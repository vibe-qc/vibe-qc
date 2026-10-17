// Internal Gaussian operands for a future consistent OMx Hamiltonian.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace vibeqc::semiempirical::nddo {

// Coefficients multiply normalized Cartesian primitives. A required s shell
// and an optional p shell share exponents (bohr^-2 before zeta^2 scaling).
// Each contraction is explicitly normalized; no basis tables are supplied.
struct OMxGaussianContraction {
    std::vector<double> exponents;
    std::vector<double> s_coefficients;
    std::vector<double> p_coefficients;
    double zeta = 1.0;
};

struct OMxGaussianPair {
    // Atom-local AO order s,px,py,pz, or s only; positive Cartesian phases.
    Eigen::MatrixXd overlap;  // A x B, dimensionless
    // Chemists' (mu_A nu_A | lambda_B sigma_B), stored as pair matrices:
    // row = mu*n_A+nu, column = lambda*n_B+sigma. All channels retained.
    Eigen::MatrixXd eri_unscaled;  // Hartree
    Eigen::MatrixXd eri_scaled;    // f_KO times eri_unscaled, Hartree
    // Negative attraction to the OTHER core, including its supplied charge.
    Eigen::MatrixXd attraction_a_unscaled, attraction_b_unscaled;
    Eigen::MatrixXd attraction_a_scaled, attraction_b_scaled;
    std::vector<double> raw_norms_a, raw_norms_b; // squared norms before renorm
    double f_ko = 0.0;
    double core_energy = 0.0; // f_KO * Zc_A * Zc_B / R, Hartree

    // Coulomb-only response sum_{lambda,sigma} P_lambda,sigma J_mu,nu,lambda,sigma.
    // P is spin-summed in the orthogonal NDDO convention: no extra factor 2.
    // These are algebraic operands, not a complete Fock matrix or SCF result.
    Eigen::MatrixXd coulomb_a(const Eigen::MatrixXd& density_b) const;
    Eigen::MatrixXd coulomb_b(const Eigen::MatrixXd& density_a) const;
};

// Dral et al., JCTC 12, 1082 (2016), DOI 10.1021/acs.jctc.5b01046,
// section 2.3 and eqs 18-20: Gaussian integrals and uniform f_KO scaling.
// gamma_ss is the caller's matched Klopman-Ohno ss integral in Hartree.
// Coordinates are bohr; core charges are dimensionless and nonnegative.
// Internal only: does not change the existing OM2/OM3 calculation path.
OMxGaussianPair evaluate_omx_gaussian_pair(
    const OMxGaussianContraction& a, const OMxGaussianContraction& b,
    const std::array<double, 3>& origin_a,
    const std::array<double, 3>& origin_b,
    double gamma_ss, double core_charge_a, double core_charge_b);

}  // namespace vibeqc::semiempirical::nddo
