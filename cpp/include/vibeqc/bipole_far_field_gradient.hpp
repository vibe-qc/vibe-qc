#pragma once
// C++ OpenMP far-field gradient contractor.
//
// Computes the nuclear force contribution from the quartet-level bipolar
// far-field energy via the interaction-tensor gradient (dT/dR) and the
// precomputed moment-derivative terms (dM/dC), each with per-thread
// interaction-tensor caching and force accumulation.
//
// Provenance
// ----------
// Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
// doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
// Saunders et al. (1992), Sec. 5.3, Eqs. (90)-(92), supports radial
// derivatives, but not this full quartet-gradient contractor. The contractor
// is a direct derivative of the dormant implementation prototype.

#include <vector>
#include <Eigen/Dense>

#include "vibeqc/lattice_sum.hpp"

namespace vibeqc {

// ---------------------------------------------------------------------------
// Input structs (shared with bipole_contractor.hpp)
// ---------------------------------------------------------------------------

struct BipoleMomentBufferCpp;
struct BipoleQuartetEntryCpp;
struct DensityBlockCpp;

/// Compute the far-field nuclear gradient via C++ OpenMP.
///
/// For each far-field quartet q, the force on the separation vector R is:
///   f_R[a] = D_bra^T @ M_bra @ grad_T[a] @ M_ket^T @ D_ket
///
/// The force is distributed to atoms via the shell-to-atom mapping.
/// Optionally includes the sub-dominant moment-derivative terms dM/dA.
///
/// @param moment_buffer  Spherical moment buffer (moments + centres + grad).
/// @param dispatch       Far-field quartet dispatch (J or K).
/// @param density_blocks Cell-indexed density matrices.
/// @param ewald_omega    Ewald splitting parameter (>0 → erfc-screened).
/// @param n_atoms        Total number of atoms.
/// @param shell_to_atom  Flat mapping: shell_to_atom[shell_idx] = list of
///                       atom indices hosting that shell.  Empty if unavailable.
/// @param include_moment_derivative  If true, include dM/dA terms.
/// @return (n_atoms, 3) force array in Hartree/bohr.
Eigen::MatrixXd compute_bipolar_far_field_gradient_cpp(
    const BipoleMomentBufferCpp& moment_buffer,
    const std::vector<BipoleQuartetEntryCpp>& dispatch,
    const std::vector<DensityBlockCpp>& density_blocks,
    double ewald_omega,
    int n_atoms,
    const std::vector<std::vector<int>>& shell_to_atom,
    bool include_moment_derivative);

}  // namespace vibeqc
