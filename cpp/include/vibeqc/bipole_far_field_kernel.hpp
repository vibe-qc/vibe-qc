// Pre-computed Fock kernel for the dormant BIPOLE quartet prototype.
//
// Encodes the per-quartet effective linear operator that maps the
// ket density block to the bra Fock block.  The kernel is built once per
// geometry (geometry-dependent, density-independent) and applied at each
// SCF iteration by contracting with the current density matrix.
//
// The stored buffer and sparse kernel are implementation cache strategies.
// They are not derived in the periodic quartet source.
//
// Provenance
// ----------
// Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
// doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
// The kernel construction itself is an implementation prototype.

#pragma once

#include <map>
#include <vector>

#include <Eigen/Dense>

namespace vibeqc {

// One entry in the pre-computed far-field Fock kernel.
struct FarFieldFockKernelEntry {
    int bra_cell_ix = 0, bra_cell_iy = 0, bra_cell_iz = 0;
    int bra_b1 = 0, bra_b1e = 0;
    int bra_b2 = 0, bra_b2e = 0;

    int ket_cell_ix = 0, ket_cell_iy = 0, ket_cell_iz = 0;
    int ket_b3 = 0, ket_b3e = 0;
    int ket_b4 = 0, ket_b4e = 0;

    Eigen::MatrixXd K_matrix;
};

// Result of applying the far-field Fock kernel to a density matrix.
struct FarFieldFockResult {
    // Per-cell Fock contribution matrices.
    std::vector<Eigen::MatrixXd> fock_matrices;
    // Per-cell integer lattice indices (parallel to fock_matrices).
    std::vector<int> cell_ix;
    std::vector<int> cell_iy;
    std::vector<int> cell_iz;

    double coulomb_energy_far = 0.0;
    int n_quartets_applied = 0;
};

// Apply a pre-computed far-field Fock kernel to density blocks.
//
// For each kernel entry, extracts the relevant sub-block of the ket
// density, contracts with K_matrix, and accumulates into the bra Fock
// block.  The Coulomb far-field energy is accumulated as
// E_coul_far = 1/2 trace(contribution).
//
// ``density_cell_keys`` is a flat vector [ix0, iy0, iz0, ix1, iy1, iz1, ...].
// ``density_blocks`` is a parallel list of (nbf, nbf) density matrices.
//
// OpenMP-parallelised over kernel entries with thread-local
// accumulation and map-merge at the end.
FarFieldFockResult apply_far_field_fock_kernel(
    const std::vector<FarFieldFockKernelEntry>& entries,
    const std::vector<int>& density_cell_keys_flat,
    const std::vector<Eigen::MatrixXd>& density_blocks,
    int nbf);

// Batched K-matrix builder for the far-field Fock kernel.
//
// For a group of quartets sharing the same interaction tensor T,
// computes K = bra_moments @ T @ ket_moments^T for each quartet
// via efficient Eigen matrix multiplication.
//
// ``bra_moments_2d`` and ``ket_moments_2d`` are parallel lists of
// flattened moment matrices, each of shape (n_ao_pair, n_sph_comp).
// ``T_matrix`` is the (n_sph_comp, n_sph_comp) interaction tensor
// for the group.
//
// Returns a list of K matrices, each of shape (n_bra_ao, n_ket_ao),
// parallel to the input moment lists.
//
// OpenMP-parallelised over quartets.
std::vector<Eigen::MatrixXd> build_far_field_k_matrices_batch(
    const std::vector<Eigen::MatrixXd>& bra_moments_2d,
    const std::vector<Eigen::MatrixXd>& ket_moments_2d,
    const Eigen::MatrixXd& T_matrix);

}  // namespace vibeqc
