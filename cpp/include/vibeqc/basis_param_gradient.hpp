#pragma once

// Analytic derivatives of AO integrals with respect to a single basis-set
// parameter (a primitive Gaussian exponent), for the Phase-1 analytic
// SCF-energy gradient of the basis optimiser. See
// docs/basisset_dev/ENERGY_GRADIENT_DESIGN.md.
//
// The overlap derivative dS/dα is the −tr(W·∂S) Pulay term and the cleanest
// piece — it needs only second-moment (emultipole2) integrals about the
// differentiated shell's own centre, with no l+2 shell construction and no
// cartesian↔spherical transform. The kinetic / nuclear / ERI derivatives need
// the r²-weighted bra represented as l+2 Gaussians (see basis_param_gradient.cpp
// for the empirically-measured cart→(l+2) transform that handles libint's
// do_enforce=false normalisation).

#include "vibeqc/basis.hpp"
#include "vibeqc/molecule.hpp"

#include <Eigen/Dense>

#include <vector>

namespace vibeqc {

// dS/dα for the overlap matrix, where α is the ``prim_idx``-th primitive
// exponent of contracted shell ``shell_idx`` of ``basis``.
//
// The result is the full (nbf × nbf) symmetric matrix ∂S_{μν}/∂α; only the
// rows/cols belonging to ``shell_idx`` are non-zero (varying one shell's
// exponent only touches integrals involving that shell), but the dense
// matrix keeps the caller's contraction with the density trivial.
//
// Matches the finite-difference reference used by the Phase-0 gradient:
// the basis is libint-normalised (primitive *and* contracted), so the
// derivative includes the contracted-renormalisation response.
Eigen::MatrixXd overlap_exponent_derivative(const BasisSet& basis,
                                            int shell_idx, int prim_idx);

// dT/dα and dV/dα (Phase 1b): kinetic and nuclear-attraction integral
// derivatives w.r.t. the prim_idx-th primitive exponent of shell_idx.
// Same conventions/return shape as overlap_exponent_derivative.
Eigen::MatrixXd kinetic_exponent_derivative(const BasisSet& basis,
                                            int shell_idx, int prim_idx);

Eigen::MatrixXd nuclear_exponent_derivative(const BasisSet& basis,
                                            const Molecule& mol,
                                            int shell_idx, int prim_idx);

// dERI/dα (Phase 1b): two-electron integral derivative w.r.t. the
// prim_idx-th primitive exponent of shell_idx. Returns the dense flat
// (nbf⁴, row-major) tensor ∂(μν|λσ)/∂α; reshape to (nbf,nbf,nbf,nbf).
std::vector<double> eri_exponent_derivative(const BasisSet& basis,
                                            int shell_idx, int prim_idx);

}  // namespace vibeqc
