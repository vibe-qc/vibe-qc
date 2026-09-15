#pragma once
// Moment-derivative computation for the far-field analytic gradient.
//
// Computes dM_sph/dC (spherical multipole moment derivatives with respect
// to the pair centre) for every shell pair on the lattice cell list.
// Stored alongside the regular spherical moments for use by the
// far-field gradient module.
//
// The derivative of the Cartesian moment M^{(i,j,k)}(C) w.r.t. C_a is:
//   dM^{(i,j,k)}/dC_a = -i * M^{(i-1)}(C)  (if power_a > 0)
// This follows directly from differentiating the standard binomial shift.
// Saunders et al. (1992), Sec. 5.3, Eqs. (90)-(92), derives radial-kernel
// derivatives, not this moment-buffer algorithm.

#include <vector>
#include <Eigen/Dense>
#include "vibeqc/lattice_sum.hpp"

namespace vibeqc {

struct MomentDerivativeEntry {
    int shell_1;      // bra shell index
    int shell_2;      // ket shell index
    int cell_index;   // cell index in the lattice cell list
    int bf_count_1;   // number of AOs in shell_1
    int bf_count_2;   // number of AOs in shell_2
    int bf_offset_1;  // first AO index of shell_1
    int bf_offset_2;  // first AO index of shell_2
    // Flattened derivative moments: (3 * n1 * n2 * n_sph_deriv) doubles
    // Layout: axis-major [x0_flat, x1_flat, ...; y0_flat, ...; z0_flat, ...]
    // Each axis block is n1*n2*n_sph_deriv doubles in row-major order
    // (first n_sph_deriv are for AO (0,0), then AO (0,1), ...).
    std::vector<double> gradient_flat;
};

struct MomentDerivativeBufferCpp {
    int max_cart_L;  // 2, 3, or 4
    int n_sph_deriv;  // L^2 components (4, 9, or 16)
    std::vector<MomentDerivativeEntry> entries;
};

/// Compute spherical moment derivatives dM_sph/dC_a for every shell pair.
///
/// @param cartesian_blocks  Per-cell Cartesian moment blocks:
///        cartesian_blocks[c][comp] is an (nbf, nbf) Eigen matrix.
/// @param C_matrix          Cartesian→spherical conversion matrix (n_sph, n_cart).
/// @param shell_slices      (first_bf, n_bf) for each shell.
/// @param cells             Lattice cell list.
/// @param n_cart            Number of Cartesian components (10, 20, or 35).
MomentDerivativeBufferCpp compute_moment_derivatives_for_buffer(
    const std::vector<std::vector<Eigen::MatrixXd>>& cartesian_blocks,
    const Eigen::MatrixXd& C_matrix,
    const std::vector<std::pair<int, int>>& shell_slices,
    const std::vector<LatticeCell>& cells,
    int n_cart);

}  // namespace vibeqc
