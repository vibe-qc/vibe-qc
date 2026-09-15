// Parallel classifier for the dormant BIPOLE quartet far-field prototype.
//
// Enumerates shell quartets (s1, s2, c_g, s3, s4, c_lam) on a lattice
// cell list, applies an implementation-specific geometric classifier, and
// returns entries assigned a truncation order greater than zero. This is not
// a literature-certified near/far partition.
//
// OpenMP-parallelised over the (c_g, c_lam) cell-pair loop.
//
// Provenance
// ----------
// Pisani, Dovesi, and Roetti, Hartree-Fock Ab Initio Treatment of
// Crystalline Systems (1988), Ch. II.4c, Eqs. II.4.7-II.4.10,
// doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
// Saunders et al. (1992), Secs. 6.7-6.8, provides related electrostatic
// penetration terminology, but neither source derives this classifier.

#pragma once

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace vibeqc {

// One far-field quartet entry — returned by the dispatch enumeration.
struct BipolarQuartetEntry {
    int s1, s2, s3, s4;   // shell indices (0-based)
    int c_bra, c_ket;     // cell indices into the cell list
    int truncation_order;  // multipole truncation order (1..4)
    double bra_width;     // product-distribution width gamma_bra
    double ket_width;     // product-distribution width gamma_ket
};

// Shell descriptor for the penetration dispatch.
struct BipoleShellInfo {
    double min_exponent;    // smallest primitive exponent
    std::array<double, 3> origin;  // shell centre (bohr)
    int atom_index;         // owning atom (for symmetry mapping)
};

// Cell descriptor.
struct BipoleCellInfo {
    std::array<double, 3> r_cart;  // Cartesian cell vector (bohr)
    std::array<int, 3> index;      // integer lattice index
};

// Parameters for the implementation-specific prototype classifier.
struct BipoleDispatchParams {
    double cell_length_scale_inv;  // (1/V)^{1/dim}
    double dispatch_slope;         // prototype default: max_order + 1
    double overlap_threshold;      // prototype diffuse-overlap bound
    int max_multipole_order;       // prototype default: 4
};

// Enumerate far-field shell quartets on a lattice.
//
// For each (c_g, c_lam) cell pair and (s1..s4) shell quartet:
//   1. Estimate bra-pair and ket-pair overlaps.  Skip if below threshold.
//   2. Estimate product-distribution centres from diffuse exponents.
//   3. Apply the prototype geometric classifier -> truncation_order.
//   4. If truncation_order > 0, record the quartet.
//
// OpenMP-parallelised over the (c_g, c_lam) pair index.
// Each thread accumulates into a thread-local vector; reduction merges.
std::vector<BipolarQuartetEntry> compute_bipolar_penetration_dispatch(
    const std::vector<BipoleShellInfo>& shells,
    const std::vector<BipoleCellInfo>& cells,
    const BipoleDispatchParams& params);

}  // namespace vibeqc
