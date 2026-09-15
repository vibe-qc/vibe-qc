// Evaluate atomic-orbital basis functions and their gradients on a set of
// grid points. Consumed by DFT: need χ_μ(r_g) for every basis function /
// grid point, plus ∇χ_μ for GGA functionals and DFT gradients.

#pragma once

#include <Eigen/Dense>
#include <complex>

#include "basis.hpp"

namespace vibeqc {

// Returns an (n_points, n_basis) matrix of AO values:
//   out(g, μ) = χ_μ(r_g)
// Points are passed as an (n_points, 3) matrix of Cartesian positions in
// bohr. The basis-function ordering matches BasisSet::shell2bf().
Eigen::MatrixXd evaluate_ao(const BasisSet& basis,
                            const Eigen::MatrixX3d& points);

// Returns value plus spatial gradients:
//   values(g, μ)      = χ_μ(r_g)
//   gradients[c](g, μ) = ∂χ_μ/∂x_c at r_g  (c ∈ {0:x, 1:y, 2:z})
struct AOValues {
    Eigen::MatrixXd values;            // (n_points, n_basis)
    std::array<Eigen::MatrixXd, 3> gradients;  // each (n_points, n_basis)
};
AOValues evaluate_ao_with_gradient(const BasisSet& basis,
                                   const Eigen::MatrixX3d& points);

// Returns values, first, and second spatial derivatives of every AO at
// every grid point. Hessians are stored as the 6 unique components in
// libint-standard order: (xx, xy, xz, yy, yz, zz).
struct AOValuesWithHessian {
    Eigen::MatrixXd values;                 // (n_points, n_basis)
    std::array<Eigen::MatrixXd, 3> gradients;   // each (n_points, n_basis)
    std::array<Eigen::MatrixXd, 6> hessians;    // xx, xy, xz, yy, yz, zz
};
AOValuesWithHessian evaluate_ao_with_hessian(const BasisSet& basis,
                                             const Eigen::MatrixX3d& points);

// Bloch-summed atomic-orbital matrix on a set of grid points:
//
//   χ_μ^k(r_g) = Σ_T  e^{i k·T}  χ_μ(r_g − T)
//
// where the sum runs over the supplied lattice translations T (Cartesian,
// bohr). Returns a complex (n_points, n_basis) matrix. Contracting against
// a column of the multi-k SCF coefficient matrix recovers the Bloch
// crystalline orbital ψ_{n,k}(r) = Σ_μ C_μ(k) χ_μ^k(r). Multi-orbital plots
// at the same k reuse this matrix.
//
// Implementation: for each translation T we evaluate the unit-cell AO basis
// at the shifted points r − T (this realises χ_μ(r − T) without rebuilding
// the basis) and accumulate with the Bloch phase exp(i k·T) into separate
// real and imaginary accumulators. The expensive shell loop inside
// ``evaluate_ao`` already parallelises over grid points, so the outer loop
// over lattice translations stays serial to avoid nested OpenMP regions.
Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
evaluate_bloch_ao(
    const BasisSet& basis,
    const Eigen::MatrixX3d& points,
    const Eigen::Vector3d& k_cart,
    const Eigen::MatrixX3d& lattice_translations);

}  // namespace vibeqc
