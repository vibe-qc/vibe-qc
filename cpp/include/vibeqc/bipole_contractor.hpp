// Direct quartet-level bipolar far-field Coulomb contractor.
//
// Replaces the Python build_bipolar_coulomb_far_field with a single
// C++ OpenMP-parallel function that loops over all far-field quartets,
// retrieves pre-computed spherical multipole moments, computes (or
// looks up) the interaction tensor, and contracts moments x tensor x
// density into per-cell Fock matrix blocks.
//
// This is the per-iteration hot path when the pre-computed Fock kernel
// is not available (e.g. single-shot energy evaluation or density-matrix
// changes that invalidate the kernel).  When the kernel IS available,
// use apply_far_field_fock_kernel instead (bipole_far_field_kernel.hpp).
//
// Provenance
// ----------
// Pisani, Dovesi, and Roetti, Hartree-Fock Ab Initio Treatment of
// Crystalline Systems (1988), Ch. II.4c, Eqs. II.4.7-II.4.10,
// doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
// This contractor, its indexing, and its cache are implementation prototypes.

#pragma once

#include "bipole_multipole.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vibeqc {

// ---------------------------------------------------------------------------
// Input structs — direct equivalents of the Python dataclasses
// ---------------------------------------------------------------------------

// One shell-pair moment entry in the spherical moment buffer.
struct BipolePairMomentEntry {
    int shell_index_1;       // bra shell index
    int shell_index_2;       // ket shell index
    int cell_index;          // index into the cell list
    int bf_offset_1;         // first basis function of shell 1
    int bf_count_1;          // number of basis functions in shell 1
    int bf_offset_2;         // first basis function of shell 2
    int bf_count_2;          // number of basis functions in shell 2
    // Flat spherical moment data: (n1*n2, n_sph) row-major.
    Eigen::MatrixXd moments_flat;
    // Adjoined-Gaussian product-distribution centre (x, y, z).
    double centre_x;
    double centre_y;
    double centre_z;
};

// Spherical moment buffer — all per-pair moments and centres.
struct BipoleMomentBufferCpp {
    int L_max;               // maximum multipole order
    int n_sph;               // number of spherical components = (L_max+1)^2
    std::vector<BipolePairMomentEntry> entries;
    // Cell vectors (bohr), length n_cells.
    std::vector<std::array<double, 3>> cell_positions;
    // Cell index triplets, length n_cells.
    std::vector<std::array<int, 3>> cell_indices;
    // Shell slice map: shell_index -> (bf_offset, bf_count).
    std::vector<std::pair<int, int>> shell_slices;
};

// One quartet assigned to the dormant prototype far path.
struct BipoleQuartetEntryCpp {
    int shell_1;             // bra-product shell index 1
    int shell_2;             // bra-product shell index 2
    int cell_bra;            // bra cell index
    int shell_3;             // ket-product shell index 3
    int shell_4;             // ket-product shell index 4
    int cell_ket;            // ket cell index
    int truncation_order;    // multipole truncation order (1 to L_max)
    double bra_width;        // product-distribution width gamma_bra
    double ket_width;        // product-distribution width gamma_ket
};

// Density block for one lattice cell.
struct DensityBlockCpp {
    int cell_ix;
    int cell_iy;
    int cell_iz;
    Eigen::MatrixXd density;  // (nbf, nbf)
};

// ---------------------------------------------------------------------------
// Output struct
// ---------------------------------------------------------------------------

// Result of the far-field Coulomb contractor.
struct BipoleFarFieldResultCpp {
    // Per-cell Fock contributions: same ordering as fock_cell_*.
    std::vector<Eigen::MatrixXd> fock_blocks;
    // Cell indices for each Fock block.
    std::vector<int> fock_cell_ix;
    std::vector<int> fock_cell_iy;
    std::vector<int> fock_cell_iz;
    // Total far-field Coulomb energy (Hartree).
    double coulomb_energy_far = 0.0;
    // Number of quartets actually processed.
    int quartets_processed = 0;
};

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

// Compute the Coulomb far-field Fock contribution for all quartets in the
// penetration dispatch.
//
// For each quartet with truncation_order > 0:
//   1. Retrieve bra and ket spherical multipole moments from the buffer.
//   2. Compute the interaction tensor T(R_sep) at the product-centre
//      separation (bare Coulomb or erfc-screened, depending on ewald_omega).
//   3. Contract: F_bra += M_bra @ T @ M_ket^T @ D_ket_subblock
//   4. Accumulate Coulomb energy: E += 0.5 * Tr(D_bra_subblock @ F_contrib).
//
// Parameters
// ----------
// moment_buffer : BipoleMomentBufferCpp
//     Pre-computed per-shell-pair spherical moments and pair centres.
// dispatch : vector of BipoleQuartetEntryCpp
//     Quartet flags from the implementation-specific classifier.
// density_blocks : vector of DensityBlockCpp
//     Density matrix blocks.  Indexed by cell (ix, iy, iz).
// ewald_omega : double
//     Ewald splitting parameter.  0 = bare Coulomb; >0 = erfc-screened.
// nbf_total : int
//     Total number of basis functions (for Fock block allocation).
//
// Returns
// -------
// BipoleFarFieldResultCpp with per-cell Fock contributions and energy.
//
// Thread safety
// -------------
// OpenMP-parallel over quartets with thread-local Fock accumulation.
// The interaction tensor computation (multipole_interaction_tensor /
// multipole_erfc_interaction_tensor) is called per unique (R_sep, L_order)
// pair; results are cached per thread to avoid recomputation.
BipoleFarFieldResultCpp compute_bipolar_coulomb_far_field_cpp(
    const BipoleMomentBufferCpp& moment_buffer,
    const std::vector<BipoleQuartetEntryCpp>& dispatch,
    const std::vector<DensityBlockCpp>& density_blocks,
    double ewald_omega,
    int nbf_total);

}  // namespace vibeqc
