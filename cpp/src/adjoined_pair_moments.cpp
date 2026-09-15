// Adjoined-Gaussian analytical moment extension to arbitrary L_max.
//
// Generalizes extend_lattice_moments_to_L4 to L=5 (zero) and L=6.
//
// For an isotropic s-Gaussian product distribution centered at P
// with width γ = a1·a2/(a1+a2), the even Cartesian moments are:
//
//   M_{2n} = S · (n_x-1)!! · (n_y-1)!! · (n_z-1)!! / (2γ)^n
//
// where n_x + n_y + n_z = 2n, (-1)!! ≡ 1.
//
// Odd orders (L=1,3,5) are identically zero by symmetry.

#include "vibeqc/adjoined_pair_moments.hpp"

#include <array>
#include <cmath>
#include <omp.h>
#include <vector>

namespace vibeqc {

namespace {

// Double factorial: (2k-1)!! = 1·3·5·...·(2k-1).  (-1)!! = 1.
int double_factorial(int n) {
    if (n <= 0) return 1;
    int result = 1;
    for (int k = 1; k <= n; k += 2) result *= k;
    return result;
}

// Number of Cartesian components for multipole order L.
int n_cart_for_L(int L) {
    return (L + 1) * (L + 2) * (L + 3) / 6;
}

// Generate all Cartesian monomial (i,j,k) with i+j+k = L in
// lexicographic order (i descending, then j descending).
// Returns a flat vector of 3*N values: [i0,j0,k0, i1,j1,k1, ...].
std::vector<int> monomials_of_order(int L) {
    std::vector<int> result;
    for (int i = L; i >= 0; --i) {
        for (int j = L - i; j >= 0; --j) {
            int k = L - i - j;
            result.push_back(i);
            result.push_back(j);
            result.push_back(k);
        }
    }
    return result;
}

// Isotropic moment coefficient for axis counts (nx, ny, nz).
// Returns coeff / (2γ)^(L/2) where L = nx + ny + nz.
double isotropic_coeff(int nx, int ny, int nz, double inv_2gamma_pow) {
    if ((nx % 2) != 0 || (ny % 2) != 0 || (nz % 2) != 0) return 0.0;
    return double_factorial(nx - 1) * double_factorial(ny - 1)
         * double_factorial(nz - 1) * inv_2gamma_pow;
}

}  // namespace

LatticeMultipoleSet extend_lattice_moments(
    const LatticeMultipoleSet& M_in,
    const BasisSet& basis,
    int L_target) {

    if (M_in.L_max < 3 || M_in.L_max > 4) return M_in;
    if (L_target < M_in.L_max + 1 || L_target > 6) return M_in;

    const int nbf = M_in.nbf;
    const int n_cells = static_cast<int>(M_in.cells.size());
    const int n_comp_in = n_cart_for_L(M_in.L_max);  // 20 for L=3, 35 for L=4

    const auto& shells_ref = basis.libint();
    const auto shell2bf = shells_ref.shell2bf();
    const int n_sh = static_cast<int>(shells_ref.size());

    // Extract min exponents and shell AO ranges.
    std::vector<double> min_exp(n_sh);
    std::vector<int> bf_start(n_sh), bf_size(n_sh);
    for (int s = 0; s < n_sh; ++s) {
        const auto& sh = shells_ref[s];
        double a_min = sh.alpha[0];
        for (double e : sh.alpha) { if (e < a_min) a_min = e; }
        min_exp[s] = a_min;
        bf_start[s] = static_cast<int>(shell2bf[s]);
        bf_size[s] = static_cast<int>(sh.size());
    }

    // Build output: copy input components, append zero-initialized higher orders.
    const int n_comp_out = n_cart_for_L(L_target);
    LatticeMultipoleSet M_out;
    M_out.nbf = nbf;
    M_out.L_max = L_target;
    M_out.spherical = false;
    M_out.cells = M_in.cells;
    M_out.origin = M_in.origin;
    M_out.blocks.resize(n_cells);
    for (int c = 0; c < n_cells; ++c) {
        M_out.blocks[c].resize(n_comp_out);
        for (int comp = 0; comp < n_comp_in; ++comp) {
            M_out.blocks[c][comp] = M_in.blocks[c][comp];
        }
        for (int comp = n_comp_in; comp < n_comp_out; ++comp) {
            M_out.blocks[c][comp] = Eigen::MatrixXd::Zero(nbf, nbf);
        }
    }

    // Fill higher-order components (L > M_in.L_max).
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int c = 0; c < n_cells; ++c) {
        for (int s1 = 0; s1 < n_sh; ++s1) {
            double a1 = min_exp[s1];
            int b1 = bf_start[s1], n1 = bf_size[s1];

            for (int s2 = 0; s2 < n_sh; ++s2) {
                double a2 = min_exp[s2];
                int b2 = bf_start[s2], n2 = bf_size[s2];

                double sum_a = a1 + a2;
                if (sum_a < 1e-30) continue;
                double gamma = a1 * a2 / sum_a;
                double inv_2gamma = 0.5 / gamma;  // 1/(2γ)

                // Overlap sub-block from input component 0.
                const auto& S_full = M_in.blocks[c][0];
                Eigen::MatrixXd S_sub = S_full.block(b1, b2, n1, n2);

                // For each order L from M_in.L_max+1 to L_target:
                int comp_offset = n_comp_in;
                for (int L = M_in.L_max + 1; L <= L_target; ++L) {
                    auto monomials = monomials_of_order(L);
                    int n_comp_L = n_cart_for_L(L) - n_cart_for_L(L - 1);

                    for (int m = 0; m < n_comp_L; ++m) {
                        int nx = monomials[3*m];
                        int ny = monomials[3*m + 1];
                        int nz = monomials[3*m + 2];

                        // Odd L: all zero (isotropic → odd moments vanish).
                        if (L % 2 != 0) {
                            comp_offset++;
                            continue;
                        }

                        // Even L: isotropic coefficient.
                        int n = L / 2;
                        double inv_2gamma_pow = std::pow(inv_2gamma, n);
                        double coeff = isotropic_coeff(nx, ny, nz, inv_2gamma_pow);

                        if (coeff != 0.0) {
                            M_out.blocks[c][comp_offset].block(b1, b2, n1, n2)
                                += S_sub * coeff;
                        }
                        comp_offset++;
                    }
                }
            }
        }
    }

    return M_out;
}

LatticeMultipoleSet extend_lattice_moments_to_L4(
    const LatticeMultipoleSet& M3,
    const BasisSet& basis) {
    return extend_lattice_moments(M3, basis, 4);
}

}  // namespace vibeqc
