// Multipole-multipole interaction tensors for the BIPOLE bipolar expansion.
//
// Computes the real spherical-harmonic interaction tensor T_{l1,m1; l2,m2}(R)
// from the exact Cartesian Taylor expansion of 1/|R|, following the same
// Stone-convention (Schmidt-semi-normalised) as the multipole moments.
//
// Supports bare Coulomb, erfc-screened short-range, and erf-screened
// long-range kernels.
//
// References
// ----------
// Pisani, Dovesi, and Roetti, Hartree-Fock Ab Initio Treatment of
// Crystalline Systems (1988), Ch. II.4c, Eqs. II.4.7-II.4.10,
// doi:10.1007/978-3-642-93385-1, for the periodic quartet expansion.
// Saunders et al. (1992), Sec. 5.3, Eqs. (90)-(92), for Cartesian
// derivatives of a spherically symmetric radial kernel.
//
// Jackson, Classical Electrodynamics (3rd ed.), Sec. 4.1 — Cartesian
//   multipole expansion of 1/|r-R|.
//
// Stone, The Theory of Intermolecular Forces (2nd ed., OUP 2013),
//   ch. 3 — real solid harmonic conventions.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace vibeqc {

// Number of real spherical harmonic components for L up to L_max.
inline int multipole_n_components(int L_max) {
    return (L_max + 1) * (L_max + 1);
}

// Flat index for spherical-harmonic component (l, m), m in [-l, +l].
inline int multipole_lm_index(int l, int m) {
    return l * l + l + m;
}

// Number of Cartesian multipole components for L up to L_max.
inline int multipole_n_cart_components(int L_max) {
    return (L_max + 1) * (L_max + 2) * (L_max + 3) / 6;
}

// Cartesian component multi-indices (i,j,k) for L up to L_max.
// Returns a flat vector of 3*N where entry t = (ix, iy, iz).
std::vector<int> multipole_cartesian_indices(int L_max);

// Convert Cartesian → spherical conversion matrix C (n_sph, n_cart).
// M_sph = C @ M_cart.  The Stone real-solid-harmonic convention is used,
// identical to the Python _cart_to_sph module.
Eigen::MatrixXd multipole_cartesian_to_spherical_matrix(int L_max);

// Exact Cartesian derivative d^{i,j,k} (1/|R|) for |i+j+k| <= 5.
// gamma = (ix, iy, iz) is the multi-index of partial derivatives w.r.t.
// (x, y, z); R = (rx, ry, rz) and r = |R|.
double multipole_cartesian_derivative_inverse_r(
    int ix, int iy, int iz,
    double rx, double ry, double rz);

// Exact Cartesian derivative d^{i,j,k} [erfc(sqrt(mu)*r)/r] for |i+j+k| <= 3.
// The erf component is 2*sqrt(mu/pi)*F_0(mu*r^2); erfc/r is obtained as
// 1/r minus that component. See Saunders (1992), Sec. 5.3, Eqs. (90)-(92),
// for the radial-derivative recurrence.
// This function handles the mu=0 limit (bare Coulomb) by falling back to
// multipole_cartesian_derivative_inverse_r.
double multipole_cartesian_derivative_erfc_over_r(
    int ix, int iy, int iz,
    double rx, double ry, double rz, double mu);

// Build the spherical interaction tensor T_{l1,m1; l2,m2}(R) for the bare
// Coulomb kernel 1/|R|.  T has shape (n_A, n_B) where
// n_A = (L_max_A+1)^2, n_B = (L_max_B+1)^2.
//
// Constructed from the Cartesian Taylor expansion via pseudoinverse,
// avoiding Wigner-3j normalisation ambiguities.
Eigen::MatrixXd multipole_interaction_tensor(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz);

// Build the spherical interaction tensor for the erfc-screened kernel
// erfc(sqrt(mu)*r)/r (the short-range Ewald piece).
Eigen::MatrixXd multipole_erfc_interaction_tensor(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz, double mu);

// Build the spherical interaction tensor for the erf-screened kernel
// erf(sqrt(mu)*r)/r (the long-range Ewald piece), computed as the
// difference T_bare - T_erfc.
Eigen::MatrixXd multipole_erf_interaction_tensor(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz, double mu);

// Gradient of the bare-Coulomb spherical interaction tensor.
//
// Returns a (3, n_A, n_B) array where result[axis, :, :] is the
// partial derivative ∂/∂R_axis of T_{l1,m1; l2,m2}(R).
//
// Computed from the order+1 Cartesian derivative of 1/|R|:
//   ∂/∂R_x d^{i,j,k}(1/r) = d^{i+1,j,k}(1/r)
//
// Supports L_max up to 4; the L=4 gradient uses order-5 derivatives.
std::vector<Eigen::MatrixXd> multipole_interaction_tensor_gradient(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz);

// Gradient of the erfc-screened spherical interaction tensor.
// Supports L_max ≤ 3.  Falls back to bare Coulomb gradient when mu = 0.
std::vector<Eigen::MatrixXd> multipole_erfc_interaction_tensor_gradient(
    int L_max_A, int L_max_B,
    double rx, double ry, double rz, double mu);

}  // namespace vibeqc
