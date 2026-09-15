// B97M-form semilocal XC energy density (energy-only) for the ωB97M(2)
// xDH double hybrid. See cpp/src/b97m_energy.cpp for the full provenance.
//
// The B97M meta-GGA semilocal energy is a sum of three channels (exchange,
// same-spin correlation, opposite-spin correlation), each an LDA reference
// (erf-attenuated Slater for exchange; PW92 Stoll-decomposed for
// correlation) times a 2D power series g(w,u) = Σ c_i w^{p_i} u^{q_i}.
// The (w,u) indices and coefficients are functional-specific, supplied at
// the call site, so the same evaluator serves ωB97M-V (self-check) and
// ωB97M(2). Energy-only: ωB97M(2) is xDH (non-self-consistent), so no V_xc.

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace vibeqc {

// One term of a B97M power series: coeff · w^{w_power} · u^{u_power}.
struct B97MTerm {
    double coeff = 0.0;
    int w_power = 0;
    int u_power = 0;
};

// Functional-defining parameters (no per-point data).
struct B97MParamsStatic {
    double omega = 0.0;       // range-separation ω (for the SR erf exchange)
    double gamma_x = 0.0;     // B97 inhomogeneity γ for exchange
    double gamma_ss = 0.0;    // ... same-spin correlation
    double gamma_os = 0.0;    // ... opposite-spin correlation
    std::vector<B97MTerm> terms_x;
    std::vector<B97MTerm> terms_ss;
    std::vector<B97MTerm> terms_os;
};

// Static params + the per-point kinetic-energy densities τ_σ.
struct B97MParams {
    double omega = 0.0;
    double gamma_x = 0.0, gamma_ss = 0.0, gamma_os = 0.0;
    std::vector<B97MTerm> terms_x, terms_ss, terms_os;
    double tau_a = 0.0;
    double tau_b = 0.0;
};

// Per grid point: returns the integrand ρ·ε_xc^semilocal (vibe-qc
// convention, i.e. energy density per unit volume). sigma_aa = |∇ρ_α|²,
// sigma_bb = |∇ρ_β|².
double b97m_semilocal_exc_point(double rho_a, double rho_b,
                                double sigma_aa, double sigma_bb,
                                const B97MParams& p);

// Vectorised over a grid. Returns exc(g) = ρ(g)·ε_xc^semilocal(g); the XC
// energy is Σ_g w_g·exc(g). sigma_ab is accepted for signature symmetry
// with the libxc polarised MGGA interface but is unused (B97M needs only
// the per-spin σ_σσ).
Eigen::VectorXd b97m_semilocal_exc(const Eigen::VectorXd& rho_a,
                                   const Eigen::VectorXd& rho_b,
                                   const Eigen::VectorXd& sigma_aa,
                                   const Eigen::VectorXd& sigma_ab,
                                   const Eigen::VectorXd& sigma_bb,
                                   const Eigen::VectorXd& tau_a,
                                   const Eigen::VectorXd& tau_b,
                                   const B97MParamsStatic& sp);

// Per-grid-point B97M semilocal potential (libxc vrho/vsigma/vtau
// convention, ∂(ρ ε_xc)/∂·) for the self-consistent ωB97M(2) variant.
struct B97MVxc {
    Eigen::VectorXd v_rho_a, v_rho_b;
    Eigen::VectorXd v_sigma_aa, v_sigma_bb;
    Eigen::VectorXd v_tau_a, v_tau_b;
};

// The B97M semilocal potential needed to build the Fock matrix when
// ωB97M(2) is run self-consistently (own orbitals). Per-point central
// finite-difference of the validated energy density; sigma_ab is unused.
B97MVxc b97m_semilocal_vxc(const Eigen::VectorXd& rho_a,
                           const Eigen::VectorXd& rho_b,
                           const Eigen::VectorXd& sigma_aa,
                           const Eigen::VectorXd& sigma_ab,
                           const Eigen::VectorXd& sigma_bb,
                           const Eigen::VectorXd& tau_a,
                           const Eigen::VectorXd& tau_b,
                           const B97MParamsStatic& sp);

}  // namespace vibeqc
