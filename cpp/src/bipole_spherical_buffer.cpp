// Cartesian → spherical buffer conversion (C++ impl).

#include "vibeqc/bipole_spherical_buffer.hpp"

#include <algorithm>
#include <omp.h>
#include <stdexcept>

namespace vibeqc {

SphericalMomentBufferCpp convert_cartesian_to_spherical_buffer(
    const std::vector<std::vector<Eigen::MatrixXd>>& cartesian_blocks,
    const Eigen::MatrixXd& C_matrix,
    const std::vector<std::pair<int, int>>& shell_slices,
    const std::vector<LatticeCell>& cells,
    const std::vector<std::vector<std::array<double, 3>>>& pair_centres) {

    const int n_cells = static_cast<int>(cartesian_blocks.size());
    const int n_sh = static_cast<int>(shell_slices.size());
    const int n_cart = static_cast<int>(C_matrix.cols());  // 10, 20, or 35
    const int n_sph = static_cast<int>(C_matrix.rows());   // (L_max+1)^2

    if (n_cells == 0 || n_sh == 0) {
        SphericalMomentBufferCpp empty;
        empty.L_max = 0;
        empty.n_sph = n_sph;
        empty.cells = cells;
        empty.shell_slices = shell_slices;
        return empty;
    }

    // Pre-allocate entries list (upper triangle of shell pairs × n_cells).
    const int n_pairs = n_sh * (n_sh + 1) / 2;  // upper triangle incl. diagonal
    const int total_entries = n_pairs * n_cells;

    SphericalMomentBufferCpp result;
    result.L_max = static_cast<int>(std::sqrt(n_sph)) - 1;
    result.n_sph = n_sph;
    result.cells = cells;
    result.shell_slices = shell_slices;
    result.entries.resize(total_entries);

    // Compute flat index for (cell, s1, s2) where s2 >= s1.
    auto entry_index = [&](int c, int s1, int s2) -> int {
        // Number of upper-triangle pairs before shell s1:
        // sum_{i=0}^{s1-1} (n_sh - i) = s1 * (2*n_sh - s1 + 1) / 2
        const int before_s1 = s1 * (2 * n_sh - s1 + 1) / 2;
        const int offset_in_row = s2 - s1;
        return c * n_pairs + before_s1 + offset_in_row;
    };

    // OpenMP-parallelised over entries.
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < total_entries; ++idx) {
        // Decode flat index into (c, s1, s2).
        const int c = idx / n_pairs;
        const int local = idx % n_pairs;

        // Find s1: solve s1*(2*n_sh - s1 + 1)/2 <= local
        int s1 = 0;
        int accum = 0;
        for (; s1 < n_sh; ++s1) {
            const int row_len = n_sh - s1;
            if (accum + row_len > local) break;
            accum += row_len;
        }
        const int s2 = s1 + (local - accum);

        const auto& [b1, n1] = shell_slices[s1];
        const auto& [b2, n2] = shell_slices[s2];

        // Extract Cartesian moment sub-block: shape (n1, n2, n_cart).
        // Build a matrix of shape (n1*n2, n_cart) for the matmul.
        Eigen::MatrixXd cart_flat(n1 * n2, n_cart);
        for (int comp = 0; comp < n_cart; ++comp) {
            if (comp < static_cast<int>(cartesian_blocks[c].size())) {
                Eigen::MatrixXd block = cartesian_blocks[c][comp].block(b1, b2, n1, n2);
                // Flatten column-major: cart_flat.col(comp) = block.reshaped<Eigen::ColMajor>()
                Eigen::Map<Eigen::VectorXd> col(
                    cart_flat.col(comp).data(), n1 * n2);
                Eigen::Map<const Eigen::MatrixXd> block_map(
                    block.data(), n1, n2);
                // Copy row-major from block into column of cart_flat.
                for (int i = 0; i < n1; ++i) {
                    for (int j = 0; j < n2; ++j) {
                        cart_flat(i * n2 + j, comp) = block(i, j);
                    }
                }
            } else {
                cart_flat.col(comp).setZero();
            }
        }

        // Convert: sph_flat = cart_flat @ C^T  (n1*n2, n_cart) @ (n_cart, n_sph) → (n1*n2, n_sph)
        Eigen::MatrixXd sph_flat = cart_flat * C_matrix.transpose();

        // Store.
        auto& entry = result.entries[idx];
        entry.shell_1 = s1;
        entry.shell_2 = s2;
        entry.cell_index = c;
        entry.bf_offset_1 = b1;
        entry.bf_count_1 = n1;
        entry.bf_offset_2 = b2;
        entry.bf_count_2 = n2;
        entry.moments_flat = sph_flat;

        // Adjoined-Gaussian product centre.
        if (c < static_cast<int>(pair_centres.size()) &&
            s1 * n_sh + s2 < static_cast<int>(pair_centres[c].size())) {
            const auto& ctr = pair_centres[c][s1 * n_sh + s2];
            entry.centre_x = ctr[0];
            entry.centre_y = ctr[1];
            entry.centre_z = ctr[2];
        } else {
            entry.centre_x = 0.0;
            entry.centre_y = 0.0;
            entry.centre_z = 0.0;
        }
    }

    return result;
}

}  // namespace vibeqc
