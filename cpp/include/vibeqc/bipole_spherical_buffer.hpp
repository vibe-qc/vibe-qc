// Cartesian → spherical multipole moment conversion for the BIPOLE buffer.
//
// Converts per-shell-pair Cartesian multipole moments (from libint
// emultipole{2,3} or the merged emultipole3+sphemultipole set) to
// spherical harmonic moments using the cartesian_to_spherical_matrix
// conversion.
//
// OpenMP-parallelised over the independent (cell, shell_1, shell_2)
// triple loop.
//
// Provenance
// ----------
// The conversion uses the Stone real-solid-harmonic convention implemented
// by bipole_multipole. The buffer layout is an implementation prototype;
// Saunders et al. (1992) does not derive it.

#pragma once

#include "bipole_multipole.hpp"
#include "lattice_sum.hpp"

#include <Eigen/Dense>
#include <array>
#include <utility>
#include <vector>

namespace vibeqc {

// Spherical moment buffer in C++-friendly flat layout.
struct SphericalMomentBufferCpp {
    int L_max = 0;
    int n_sph = 0;
    // Cell lattice vectors.
    std::vector<LatticeCell> cells;
    // Shell AO index ranges: shell_index -> (bf_offset, bf_count).
    std::vector<std::pair<int, int>> shell_slices;
    // Per-shell-pair moment entries.
    // Each entry is keyed by (shell_1, shell_2, cell_index) and stores
    // the flat spherical moment matrix (n1*n2, n_sph) and the adjoined-
    // Gaussian product-distribution centre.
    struct Entry {
        int shell_1;
        int shell_2;
        int cell_index;
        int bf_offset_1;
        int bf_count_1;
        int bf_offset_2;
        int bf_count_2;
        Eigen::MatrixXd moments_flat;  // (n1*n2, n_sph)
        double centre_x;
        double centre_y;
        double centre_z;
    };
    std::vector<Entry> entries;
};

// Convert per-cell per-shell-pair Cartesian moments to spherical moments.
//
// Parameters
// ----------
// cartesian_blocks : vector of (n_cells) vectors of (n_cart) matrices (nbf, nbf)
//     Cartesian moments.  n_cart = 10 (L=2), 20 (L=3), or 35 (L=4).
// C_matrix : Eigen::MatrixXd
//     Cartesian → spherical conversion matrix, shape (n_sph, n_cart).
//     From multipole_cartesian_to_spherical_matrix(L_max).
// shell_slices : vector of (bf_offset, bf_count)
//     Shell AO index ranges.
// cells : vector of LatticeCell
//     Lattice cell vectors (used for cell key assignment).
// pair_centres : vector of (n_cells) vectors of (n_shells*n_shells)
//     flat arrays of (cx, cy, cz) for adjoined-Gaussian pair centres.
//
// Returns
// -------
// SphericalMomentBufferCpp with flat spherical moment entries.
//
// OpenMP-parallelised over (cell, shell_1, shell_2).
SphericalMomentBufferCpp convert_cartesian_to_spherical_buffer(
    const std::vector<std::vector<Eigen::MatrixXd>>& cartesian_blocks,
    const Eigen::MatrixXd& C_matrix,
    const std::vector<std::pair<int, int>>& shell_slices,
    const std::vector<LatticeCell>& cells,
    const std::vector<std::vector<std::array<double, 3>>>& pair_centres);

}  // namespace vibeqc
