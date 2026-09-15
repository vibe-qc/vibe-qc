// Prototype isotropic-Gaussian extension of Cartesian moments beyond L=3.
//
// Returns a new LatticeMultipoleSet with L_max=4 containing the original
// L=3 Cartesian moments plus 15 analytically-computed hexadecapole
// components.
//
// This extension is implementation-specific. Pisani-Dovesi-Roetti (1988),
// Ch. II.4c, motivates product-distribution multipoles but does not derive
// this isotropic reconstruction.

#pragma once

#include "basis.hpp"
#include "lattice_sum.hpp"
#include "multipole_moments_lattice.hpp"
#include "periodic.hpp"

#include <Eigen/Dense>
#include <array>
#include <utility>
#include <vector>

namespace vibeqc {

// Extend a LatticeMultipoleSet from L_max=3 to L_max=L_target (4, 5, or 6).
//
// For the prototype isotropic s-Gaussian approximation, all odd-order moments
// vanish and even-order moments follow the isotropic formula:
//
//   M_{2n} = S · (n_x-1)!! · (n_y-1)!! · (n_z-1)!! / (2γ)^n
//
// where n_x + n_y + n_z = 2n, (-1)!! ≡ 1, and (2k-1)!! = 1·3·5·...·(2k-1).
//
// L=5 (21 components): all zero (odd order).
// L=6 (28 components): 15 isotropic pairings per the formula above.
//
// Returns a new LatticeMultipoleSet with L_max = L_target.
LatticeMultipoleSet extend_lattice_moments(
    const LatticeMultipoleSet& M3,
    const BasisSet& basis,
    int L_target);

// Convenience: extend to L=4 only (backward-compatible).
LatticeMultipoleSet extend_lattice_moments_to_L4(
    const LatticeMultipoleSet& M3,
    const BasisSet& basis);

}  // namespace vibeqc
