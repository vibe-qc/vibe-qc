// Moment-derivative computation for far-field analytic gradient (C++ impl).
//
// OpenMP-parallelised over cells and shell pairs.

#include "vibeqc/bipole_moment_derivatives.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <omp.h>
#include <stdexcept>
#include <tuple>

namespace vibeqc {

namespace {

// Cartesian (i,j,k) power tuples for L=2, L=3, L=4.
// Must match the Python _CART_IDX_* lists in bipole_spherical_moment_buffer.py.
struct CartTuple {
    int i, j, k;
};
using CartMap = std::map<std::tuple<int,int,int>, int>;

CartMap build_cart_map(int max_L) {
    // Build the list first.
    std::vector<CartTuple> tuples;
    if (max_L >= 2) {
        tuples = {
            {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1},
            {2,0,0}, {1,1,0}, {1,0,1}, {0,2,0}, {0,1,1}, {0,0,2},
        };
    }
    if (max_L >= 3) {
        std::vector<CartTuple> extra = {
            {3,0,0}, {2,1,0}, {2,0,1}, {1,2,0}, {1,1,1},
            {1,0,2}, {0,3,0}, {0,2,1}, {0,1,2}, {0,0,3},
        };
        tuples.insert(tuples.end(), extra.begin(), extra.end());
    }
    if (max_L >= 4) {
        std::vector<CartTuple> extra = {
            {4,0,0}, {3,1,0}, {3,0,1}, {2,2,0}, {2,1,1},
            {2,0,2}, {1,3,0}, {1,2,1}, {1,1,2}, {1,0,3},
            {0,4,0}, {0,3,1}, {0,2,2}, {0,1,3}, {0,0,4},
        };
        tuples.insert(tuples.end(), extra.begin(), extra.end());
    }
    CartMap m;
    for (int idx = 0; idx < static_cast<int>(tuples.size()); ++idx) {
        m[{tuples[idx].i, tuples[idx].j, tuples[idx].k}] = idx;
    }
    return m;
}

int max_cart_L_from_n(int n_cart) {
    if (n_cart == 10) return 2;
    if (n_cart == 20) return 3;
    if (n_cart == 35) return 4;
    throw std::invalid_argument("n_cart must be 10, 20, or 35");
}

}  // namespace

MomentDerivativeBufferCpp compute_moment_derivatives_for_buffer(
    const std::vector<std::vector<Eigen::MatrixXd>>& cartesian_blocks,
    const Eigen::MatrixXd& C_matrix,
    const std::vector<std::pair<int, int>>& shell_slices,
    const std::vector<LatticeCell>& cells,
    int n_cart) {

    const int n_cells = static_cast<int>(cartesian_blocks.size());
    const int n_sh = static_cast<int>(shell_slices.size());
    const int n_sph_cart = static_cast<int>(C_matrix.rows());
    const int max_L = max_cart_L_from_n(n_cart);
    const int n_sph_deriv = max_L * max_L;  // L^2 derivative components

    CartMap idx_map = build_cart_map(max_L);

    // Build list of (i,j,k) tuples in index order.
    std::vector<CartTuple> cart_tuples;
    for (int idx = 0; idx < n_cart; ++idx) {
        // Find the tuple for this index.
        bool found = false;
        for (const auto& [tup, cidx] : idx_map) {
            if (cidx == idx) {
                cart_tuples.push_back({std::get<0>(tup), std::get<1>(tup), std::get<2>(tup)});
                found = true;
                break;
            }
        }
        if (!found) {
            // Should not happen for valid max_L.
            cart_tuples.push_back({0,0,0});
        }
    }

    // Total entries: n_sh × n_sh × n_cells (all shell pairs).
    const int total_entries = n_sh * n_sh * n_cells;

    MomentDerivativeBufferCpp result;
    result.max_cart_L = max_L;
    result.n_sph_deriv = n_sph_deriv;
    result.entries.resize(total_entries);

    // Pre-compute row stride for flat index.
    auto entry_idx = [n_sh, n_cells](int c, int s1, int s2) -> int {
        return (c * n_sh + s1) * n_sh + s2;
    };

    // Use only the first n_sph_deriv rows of C_matrix for derivative conversion.
    Eigen::MatrixXd C_deriv = C_matrix.topRows(n_sph_deriv);

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < total_entries; ++idx) {
        const int c = idx / (n_sh * n_sh);
        const int rem = idx % (n_sh * n_sh);
        const int s1 = rem / n_sh;
        const int s2 = rem % n_sh;

        const auto& [b1, n1] = shell_slices[s1];
        const auto& [b2, n2] = shell_slices[s2];

        auto& entry = result.entries[idx];
        entry.shell_1 = s1;
        entry.shell_2 = s2;
        entry.cell_index = c;
        entry.bf_count_1 = n1;
        entry.bf_count_2 = n2;
        entry.bf_offset_1 = b1;
        entry.bf_offset_2 = b2;

        // Extract Cartesian moment sub-block: shape (n1, n2, n_cart).
        // Flatten to (n1*n2, n_cart) for matrix multiply.
        Eigen::MatrixXd cart_c(n1 * n2, n_cart);
        cart_c.setZero();
        if (c < static_cast<int>(cartesian_blocks.size())) {
            for (int comp = 0; comp < n_cart; ++comp) {
                if (comp < static_cast<int>(cartesian_blocks[c].size())) {
                    const Eigen::MatrixXd& block = cartesian_blocks[c][comp];
                    for (int i = 0; i < n1; ++i) {
                        for (int j = 0; j < n2; ++j) {
                            cart_c(i * n2 + j, comp) = block(b1 + i, b2 + j);
                        }
                    }
                }
            }
        }

        // Build derivative Cartesian moments for each axis.
        // deriv_cart[a] has shape (n1*n2, n_cart).
        // dM^{(i,j,k)}/dC_x = -i * M^{(i-1,j,k)}
        Eigen::MatrixXd deriv_cart[3];
        for (int a = 0; a < 3; ++a) deriv_cart[a].setZero(n1 * n2, n_cart);

        for (int tidx = 0; tidx < n_cart; ++tidx) {
            int i = cart_tuples[tidx].i;
            int j = cart_tuples[tidx].j;
            int k = cart_tuples[tidx].k;

            // d/dx
            if (i > 0) {
                auto it = idx_map.find({i-1, j, k});
                if (it != idx_map.end()) {
                    deriv_cart[0].col(tidx) = -static_cast<double>(i)
                        * cart_c.col(it->second);
                }
            }
            // d/dy
            if (j > 0) {
                auto it = idx_map.find({i, j-1, k});
                if (it != idx_map.end()) {
                    deriv_cart[1].col(tidx) = -static_cast<double>(j)
                        * cart_c.col(it->second);
                }
            }
            // d/dz
            if (k > 0) {
                auto it = idx_map.find({i, j, k-1});
                if (it != idx_map.end()) {
                    deriv_cart[2].col(tidx) = -static_cast<double>(k)
                        * cart_c.col(it->second);
                }
            }
        }

        // Convert to spherical: sph_flat[a] = deriv_cart[a] @ C_deriv^T
        // Result: (n1*n2, n_sph_deriv) for each axis.
        const int n_flat = n1 * n2 * n_sph_deriv;
        entry.gradient_flat.resize(3 * n_flat, 0.0);

        for (int a = 0; a < 3; ++a) {
            Eigen::MatrixXd sph_a = deriv_cart[a] * C_deriv.transpose();
            // Copy row-major into gradient_flat at offset a * n_flat.
            double* dst = entry.gradient_flat.data() + a * n_flat;
            for (int r = 0; r < n1 * n2; ++r) {
                for (int sc = 0; sc < n_sph_deriv; ++sc) {
                    dst[r * n_sph_deriv + sc] = sph_a(r, sc);
                }
            }
        }
    }

    return result;
}

}  // namespace vibeqc
