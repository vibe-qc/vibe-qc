// Shell-pair Cartesian multipole moments at lattice displacements.
//
// For the periodic quartet expansion of Pisani-Dovesi-Roetti (1988),
// Ch. II.4c,
// we need the multipole moments of the bra-pair density φ_μ(r) · φ_ν(r-g)
// evaluated around some expansion origin O:
//
//   M^{μν}_{xyz...}(g; O) = ⟨ χ_μ(r) | (r_x − O_x)^i (r_y − O_y)^j
//                                       (r_z − O_z)^k | χ_ν(r − g) ⟩
//
// where (i, j, k) are non-negative integers with l = i + j + k.
//
// libint exposes the operators:
//   * emultipole1 — overlap + dipole (l ≤ 1):  4 Cartesian components
//   * emultipole2 — emultipole1 + quadrupole (l ≤ 2): 10 Cartesian
//   * emultipole3 — emultipole2 + octupole (l ≤ 3):   20 Cartesian
//
// Component ordering (libint convention) for L_max = 2:
//   0  overlap S
//   1  ⟨μ | (r_x − O_x) | ν⟩         (dipole x)
//   2  ⟨μ | (r_y − O_y) | ν⟩         (dipole y)
//   3  ⟨μ | (r_z − O_z) | ν⟩         (dipole z)
//   4  ⟨μ | (r_x − O_x)² | ν⟩        (quadrupole xx)
//   5  ⟨μ | (r_x − O_x)(r_y − O_y) | ν⟩  (quadrupole xy)
//   6  ⟨μ | (r_x − O_x)(r_z − O_z) | ν⟩  (quadrupole xz)
//   7  ⟨μ | (r_y − O_y)² | ν⟩        (quadrupole yy)
//   8  ⟨μ | (r_y − O_y)(r_z − O_z) | ν⟩  (quadrupole yz)
//   9  ⟨μ | (r_z − O_z)² | ν⟩        (quadrupole zz)
//
// For L_max = 3 components 10..19 add the 10 octupole components in
// libint's lexicographic order over (i, j, k) with i+j+k = 3.
//
// The expansion origin O is a single user-supplied point (default: cell
// origin). For product-distribution expansion centres, the caller shifts
// the moments analytically via the standard polynomial-shift formula (no
// recomputation required). Pisani-Dovesi (1980), Sec. 4, supplies the
// adjoined diffuse s-Gaussian convention used for the approximate centre.

#pragma once

#include "basis.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace vibeqc {

// Number of Cartesian-multipole components for moments up through L_max.
// emultipole1 → 4, emultipole2 → 10, emultipole3 → 20.
inline int cartesian_multipole_n_components(int L_max) {
    int n = 0;
    for (int l = 0; l <= L_max; ++l) {
        n += (l + 1) * (l + 2) / 2;  // (l+1)(l+2)/2 Cartesian components of degree l
    }
    return n;
}

// Per-cell, per-component nbf × nbf matrix:
//
//   shape: { n_cells, n_components, nbf, nbf }
//
// flattened as `blocks[c][comp]` where `blocks[c]` has `n_components` and
// each entry is an Eigen::MatrixXd of shape (nbf, nbf).
struct LatticeMultipoleSet {
    int nbf = 0;
    int L_max = 0;
    bool spherical = false;  // true if output is spherical (sphemultipole)
    std::vector<LatticeCell> cells;
    std::array<double, 3> origin = {0.0, 0.0, 0.0};
    std::vector<std::vector<Eigen::MatrixXd>> blocks;  // blocks[c][comp]
};

// Compute shell-pair multipole moments for every lattice cell
// out to ``opts.cutoff_bohr``. The expansion origin is the same for all
// shell pairs (typically (0, 0, 0)); per-shell-pair shifts to Gaussian-
// product centers are handled in Python via the polynomial-shift formula.
//
// L_max 1,2,3 → Cartesian via libint emultipole{1,2,3}.
// L_max 4     → spherical via libint sphemultipole (n_comp = (L+1)^2 = 25).
LatticeMultipoleSet compute_multipole_moments_lattice(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    int L_max = 2,
    const std::array<double, 3>& origin = {0.0, 0.0, 0.0});

// Native atomic-position gradient of the EXT EL-SPHEROPOLE energy
//
//   E_sph = (π·N_e/6 / V) · Σ_g Σ_μν P_μν(g) · K_μν(g),
//   K_μν(g) = 2·Tr⟨r²⟩ − 2·(A_μ + B_ν)·⟨r⟩ + (|A_μ|² + |B_ν|²)·⟨1⟩,
//
// with the bond-symmetrised second moment built from emultipole2 moments
// about the origin O = 0 (A_μ = bra AO atom centre, B_ν = ket AO atom
// centre + lattice vector g). This entry point is disabled in the current
// build because libint emultipole2 deriv_order=1 is process-unsafe for
// some valid shell pairs. The Python BIPOLE gradient central-differences
// this energy term at fixed density until the native derivative port is
// repaired.
Eigen::MatrixXd compute_ext_el_spheropole_gradient_lattice(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const LatticeMatrixSet& P_real,
    int n_electrons);

}  // namespace vibeqc
