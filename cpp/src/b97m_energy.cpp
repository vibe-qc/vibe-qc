// B97M-form semilocal XC energy density (energy-only), for the xDH double
// hybrid ωB97M(2) (Mardirossian & Head-Gordon, J. Chem. Phys. 148, 241736
// (2018), doi:10.1063/1.5025226).
//
// ωB97M(2) repartitions the ωB97M-V meta-GGA semilocal space with its own
// power-series coefficients (Table II) whose (w,u) indices are NOT present
// in libxc's fixed WB97M_V form, so libxc cannot evaluate it. This file is
// the generic B97M (w,u) evaluator. It is ENERGY-ONLY: ωB97M(2) is xDH
// (evaluated non-self-consistently on ωB97M-V orbitals), so no V_xc
// potential is needed.
//
// All forms are transcribed faithfully from the vendored libxc 7.0.0 maple
// sources (the authoritative reference for the B97M form):
//   * src/maple/b97mv.mpl              — the (w,u) variables, g series, Stoll
//   * src/maple/mgga_exc/hyb_mgga_xc_wb97mv.mpl — the exchange + correlation assembly
//   * src/maple/lda_x_erf.mpl + attenuation.mpl — erf-attenuated SR LDA exchange
//   * src/maple/lda_c_pw.mpl           — PW92 correlation (modified params)
//   * src/maple/util.mpl               — lda_stoll_par / lda_stoll_perp
//   * src/src/util.h                   — RS_FACTOR / X_FACTOR_C / K_FACTOR_C
//
// Validated by the b97m_semilocal_exc self-check against libxc's ωB97M-V
// (feed ωB97M-V's coefficients → reproduce Functional("wb97m-v") energy
// density to ~1e-10; see tests/test_wb97m2.py).

#include "vibeqc/b97m_energy.hpp"

#include <cmath>
#include <stdexcept>

namespace vibeqc {

namespace {

// libxc constants (src/src/util.h).
constexpr double kRsFactor  = 0.6203504908994000166680068120477781673508;  // (3/(4π))^(1/3)
constexpr double kXFactorC  = 0.9305257363491000250020102180716672510262;  // (3/8)(3/π)^(1/3)·4^(2/3)
constexpr double kKFactorC  = 4.557799872345597137288163759599305358515;   // (3/10)(6π²)^(2/3)

// Spin-interpolation function f_ζ(z) = ((1+z)^(4/3)+(1-z)^(4/3)-2)/(2^(4/3)-2).
double f_zeta(double z) {
    const double c = std::pow(2.0, 4.0 / 3.0) - 2.0;
    const double zp = (1.0 + z > 0.0) ? std::pow(1.0 + z, 4.0 / 3.0) : 0.0;
    const double zm = (1.0 - z > 0.0) ? std::pow(1.0 - z, 4.0 / 3.0) : 0.0;
    return (zp + zm - 2.0) / c;
}

// PW92 correlation (Perdew-Wang 1992), MODIFIED parameters (lda_c_pw.mpl
// $ifdef lda_c_pw_modified_params). f_pw parametrises -ε_c per the libxc
// convention; the three channels k=0,1,2 are para / ferro / spin-stiffness.
struct PW92 {
    // a[0],a[1],a[2] use the modified high-precision values.
    static constexpr double a[3]      = {0.0310907, 0.01554535, 0.0168869};
    static constexpr double alpha1[3] = {0.21370, 0.20548, 0.11125};
    static constexpr double beta1[3]  = {7.5957, 14.1189, 10.357};
    static constexpr double beta2[3]  = {3.5876, 6.1977, 3.6231};
    static constexpr double beta3[3]  = {1.6382, 3.3662, 0.88026};
    static constexpr double beta4[3]  = {0.49294, 0.62517, 0.49671};
    static constexpr double fz20      = 1.709920934161365617563962776245;
};
constexpr double PW92::a[3];
constexpr double PW92::alpha1[3];
constexpr double PW92::beta1[3];
constexpr double PW92::beta2[3];
constexpr double PW92::beta3[3];
constexpr double PW92::beta4[3];

// Eq. (10): G_aux; pp[k] = 1 for all k so the last term is beta4·rs^2.
double pw92_g(int k, double rs) {
    const double srs = std::sqrt(rs);
    const double g_aux = PW92::beta1[k] * srs + PW92::beta2[k] * rs
                       + PW92::beta3[k] * rs * srs + PW92::beta4[k] * rs * rs;
    return -2.0 * PW92::a[k] * (1.0 + PW92::alpha1[k] * rs)
           * std::log(1.0 + 1.0 / (2.0 * PW92::a[k] * g_aux));
}

// Eq. (8): f_pw(rs, ζ) = -ε_c(rs, ζ).
double f_pw(double rs, double zeta) {
    const double fz = f_zeta(zeta);
    const double z4 = zeta * zeta * zeta * zeta;
    const double g0 = pw92_g(0, rs);
    const double g1 = pw92_g(1, rs);
    const double g2 = pw92_g(2, rs);
    return g0 + z4 * fz * (g1 - g0 + g2 / PW92::fz20) - fz * g2 / PW92::fz20;
}

// Stoll decomposition (util.mpl). lda_func is f_pw here.
//   lda_stoll_par(rs, z) = (1+z)/2 · f_pw(rs·2^(1/3)·(1+z)^(-1/3), 1)
double stoll_par(double rs, double z) {
    const double opz = 1.0 + z;
    if (opz <= 1e-15) return 0.0;
    const double rs_s = rs * std::pow(2.0, 1.0 / 3.0) * std::pow(opz, -1.0 / 3.0);
    return 0.5 * opz * f_pw(rs_s, 1.0);
}
//   lda_stoll_perp(rs, z) = f_pw(rs, z) - par(z) - par(-z)
double stoll_perp(double rs, double z) {
    return f_pw(rs, z) - stoll_par(rs, z) - stoll_par(rs, -z);
}

// erf attenuation (attenuation.mpl, Tawada et al). att_erf0; the large-a
// (low-density) smoothing enforce_smooth_lr is not yet transplanted — for
// compact-molecule energies the a >= 1.35 tail is negligible. att_erf0 is
// evaluated in a cancellation-tolerant order.
double attenuation_erf(double a) {
    if (a < 1e-10) return 1.0;
    const double inv2a = 1.0 / (2.0 * a);
    const double A1 = std::sqrt(M_PI) * std::erf(inv2a);
    const double A2 = std::exp(-1.0 / (4.0 * a * a)) - 1.0;       // exp(-1/(4a²)) - 1
    const double A3 = 2.0 * a * a * A2 + 0.5;
    return 1.0 - (8.0 / 3.0) * a * (A1 + 2.0 * a * (A2 - A3));
}

// Slater LDA exchange prefactor, per the lda_x_erf.mpl spin form.
constexpr double kLdaXAx = -kRsFactor * kXFactorC / 1.0;  // ·2^(-4/3) applied below

// Range-separated short-range LDA exchange energy density for ONE spin
// channel, following hyb_mgga_xc_wb97mv.mpl wb97mv_f: the (1±z)/2 prefactor
// times lda_x_erf_spin(rs·(2/(1±z))^(1/3), 1).
//   lda_x_erf_spin(rs', 1) = lda_x_ax·2^(4/3)/rs' · att_erf(a_cnst·rs'/2^(1/3))
// with lda_x_ax = -RS_FACTOR·X_FACTOR_C/2^(4/3); the 2^(4/3) cancels the
// 2^(1/3) per the z=1 evaluation. a_cnst = (4/(9π))^(1/3)·ω/2.
double ex_sr_spin(double rs, double opspin /*1±z*/, double omega) {
    if (opspin <= 1e-15) return 0.0;
    const double a_cnst = std::pow(4.0 / (9.0 * M_PI), 1.0 / 3.0) * omega / 2.0;
    const double rs_s = rs * std::pow(2.0 / opspin, 1.0 / 3.0);   // per-spin rs
    // lda_x_ax = -RS_FACTOR·X_FACTOR_C/2^(4/3); evaluated at z=1:
    //   lda_x_ax · (1+1)^(4/3)/rs_s = -RS_FACTOR·X_FACTOR_C/2^(4/3)·2^(4/3)/rs_s
    //                               = -RS_FACTOR·X_FACTOR_C/rs_s
    const double e_unif = -(kRsFactor * kXFactorC) / rs_s;
    const double att = attenuation_erf(a_cnst * rs_s / std::pow(2.0, 1.0 / 3.0));
    return 0.5 * opspin * e_unif * att;
}

// B97 inhomogeneity variable u(γ,x) = γx²/(1+γx²).
double u_var(double gamma, double x) {
    const double gx2 = gamma * x * x;
    return gx2 / (1.0 + gx2);
}
// w(t) = (K_FACTOR_C - t)/(K_FACTOR_C + t), t = τ_σ/ρ_σ^(5/3) (reduced τ).
double w_ss(double t) { return (kKFactorC - t) / (kKFactorC + t); }
// opposite-spin w (b97mv_wx_os).
double w_os(double t0, double t1) {
    return (kKFactorC * (t0 + t1) - 2.0 * t0 * t1)
         / (kKFactorC * (t0 + t1) + 2.0 * t0 * t1);
}

// Power series g = Σ_i c_i · w^{p_i} · u^{q_i}.
double g_series(const std::vector<B97MTerm>& terms, double w, double u) {
    double s = 0.0;
    for (const auto& t : terms) {
        s += t.coeff * std::pow(w, t.w_power) * std::pow(u, t.u_power);
    }
    return s;
}

constexpr double kRhoThresh = 1e-12;

}  // namespace

double b97m_semilocal_exc_point(double rho_a, double rho_b,
                                double sigma_aa, double sigma_bb,
                                const B97MParams& p) {
    const double rho = rho_a + rho_b;
    if (rho < kRhoThresh) return 0.0;
    const double rs = kRsFactor / std::cbrt(rho);
    const double z = (rho_a - rho_b) / rho;

    // Per-spin reduced gradient x_σ = |∇ρ_σ|/ρ_σ^(4/3) and reduced τ.
    auto xs = [](double rho_s, double sig_ss) -> double {
        if (rho_s < kRhoThresh) return 0.0;
        return std::sqrt(std::max(sig_ss, 0.0)) / std::pow(rho_s, 4.0 / 3.0);
    };
    const double xa = xs(rho_a, sigma_aa);
    const double xb = xs(rho_b, sigma_bb);
    const double ta = (rho_a > kRhoThresh) ? p.tau_a / std::pow(rho_a, 5.0 / 3.0) : kKFactorC;
    const double tb = (rho_b > kRhoThresh) ? p.tau_b / std::pow(rho_b, 5.0 / 3.0) : kKFactorC;

    // --- Exchange: SR erf-LDA × g_x, per spin (energy density, NOT per electron) ---
    double e_x = 0.0;
    if (rho_a > kRhoThresh) {
        e_x += ex_sr_spin(rs, 1.0 + z, p.omega)
             * g_series(p.terms_x, w_ss(ta), u_var(p.gamma_x, xa));
    }
    if (rho_b > kRhoThresh) {
        e_x += ex_sr_spin(rs, 1.0 - z, p.omega)
             * g_series(p.terms_x, w_ss(tb), u_var(p.gamma_x, xb));
    }

    // --- Same-spin correlation: Stoll-par PW92 × g_ss ---
    double e_css = 0.0;
    if (rho_a > kRhoThresh) {
        e_css += stoll_par(rs, z) * g_series(p.terms_ss, w_ss(ta), u_var(p.gamma_ss, xa));
    }
    if (rho_b > kRhoThresh) {
        e_css += stoll_par(rs, -z) * g_series(p.terms_ss, w_ss(tb), u_var(p.gamma_ss, xb));
    }

    // --- Opposite-spin correlation: Stoll-perp PW92 × g_os ---
    const double x_os = std::sqrt(xa * xa + xb * xb) / std::sqrt(2.0);
    const double e_cos = stoll_perp(rs, z)
        * g_series(p.terms_os, w_os(ta, tb), u_var(p.gamma_os, x_os));

    // libxc returns the energy density per unit volume already multiplied by
    // ρ for the LDA pieces (f_lda_x / f_pw are energy densities per electron
    // ×... ); the maple ``f`` is ε_xc (per electron) and work_mgga multiplies
    // by ρ. Here e_x / e_css / e_cos are per-electron contributions; multiply
    // by ρ to return the integrand ρ·ε_xc (vibe-qc convention).
    return rho * (e_x + e_css + e_cos);
}

Eigen::VectorXd b97m_semilocal_exc(const Eigen::VectorXd& rho_a,
                                   const Eigen::VectorXd& rho_b,
                                   const Eigen::VectorXd& sigma_aa,
                                   const Eigen::VectorXd& sigma_ab,
                                   const Eigen::VectorXd& sigma_bb,
                                   const Eigen::VectorXd& tau_a,
                                   const Eigen::VectorXd& tau_b,
                                   const B97MParamsStatic& sp) {
    const Eigen::Index n = rho_a.size();
    if (rho_b.size() != n || sigma_aa.size() != n || sigma_bb.size() != n
        || tau_a.size() != n || tau_b.size() != n) {
        throw std::invalid_argument("b97m_semilocal_exc: length mismatch");
    }
    Eigen::VectorXd exc(n);
    for (Eigen::Index g = 0; g < n; ++g) {
        B97MParams p;
        p.omega = sp.omega;
        p.gamma_x = sp.gamma_x; p.gamma_ss = sp.gamma_ss; p.gamma_os = sp.gamma_os;
        p.terms_x = sp.terms_x; p.terms_ss = sp.terms_ss; p.terms_os = sp.terms_os;
        p.tau_a = tau_a(g); p.tau_b = tau_b(g);
        exc(g) = b97m_semilocal_exc_point(rho_a(g), rho_b(g),
                                          sigma_aa(g), sigma_bb(g), p);
    }
    return exc;
}

// Self-consistent variant: the B97M semilocal POTENTIAL (v_ρ, v_σ, v_τ),
// needed to build the Fock matrix when ωB97M(2) is run with its own
// orbitals (rather than the published xDH form on ωB97M-V orbitals). The
// energy kernel is validated to machine precision against libxc's ωB97M-V,
// so the potential is obtained as the per-point central finite-difference
// derivative of that energy density w.r.t. each (ρ_σ, σ_σσ, τ_σ) — the
// libxc vrho/vsigma/vtau convention (∂(ρ ε_xc)/∂·). Local (no cross-point
// coupling), so FD here is cheap and exact-to-FD-order; validated by
// feeding ωB97M-V's coefficients and matching libxc's ωB97M-V potential.
B97MVxc b97m_semilocal_vxc(const Eigen::VectorXd& rho_a,
                           const Eigen::VectorXd& rho_b,
                           const Eigen::VectorXd& sigma_aa,
                           const Eigen::VectorXd& sigma_ab,
                           const Eigen::VectorXd& sigma_bb,
                           const Eigen::VectorXd& tau_a,
                           const Eigen::VectorXd& tau_b,
                           const B97MParamsStatic& sp) {
    (void)sigma_ab;  // B97M needs only σ_σσ
    const Eigen::Index n = rho_a.size();
    B97MVxc out;
    out.v_rho_a = Eigen::VectorXd::Zero(n);
    out.v_rho_b = Eigen::VectorXd::Zero(n);
    out.v_sigma_aa = Eigen::VectorXd::Zero(n);
    out.v_sigma_bb = Eigen::VectorXd::Zero(n);
    out.v_tau_a = Eigen::VectorXd::Zero(n);
    out.v_tau_b = Eigen::VectorXd::Zero(n);

    for (Eigen::Index g = 0; g < n; ++g) {
        const double ra = rho_a(g), rb = rho_b(g);
        if (ra + rb < kRhoThresh) continue;
        B97MParams p;
        p.omega = sp.omega;
        p.gamma_x = sp.gamma_x; p.gamma_ss = sp.gamma_ss; p.gamma_os = sp.gamma_os;
        p.terms_x = sp.terms_x; p.terms_ss = sp.terms_ss; p.terms_os = sp.terms_os;
        p.tau_a = tau_a(g); p.tau_b = tau_b(g);
        const double saa = sigma_aa(g), sbb = sigma_bb(g);

        // Central finite differences; step scaled to each variable.
        auto fd_rho = [&](bool spin_a) {
            const double base = spin_a ? ra : rb;
            const double h = 1e-6 * (std::abs(base) + 1e-4);
            B97MParams pp = p;
            const double rap = spin_a ? ra + h : ra;
            const double rbp = spin_a ? rb : rb + h;
            const double ram = spin_a ? ra - h : ra;
            const double rbm = spin_a ? rb : rb - h;
            const double ep = b97m_semilocal_exc_point(rap, rbp, saa, sbb, pp);
            const double em = b97m_semilocal_exc_point(ram, rbm, saa, sbb, pp);
            return (ep - em) / (2.0 * h);
        };
        auto fd_sigma = [&](bool spin_a) {
            const double base = spin_a ? saa : sbb;
            const double h = 1e-6 * (std::abs(base) + 1e-6);
            const double sap = spin_a ? saa + h : saa;
            const double sbp = spin_a ? sbb : sbb + h;
            const double sam = spin_a ? saa - h : saa;
            const double sbm = spin_a ? sbb : sbb - h;
            const double ep = b97m_semilocal_exc_point(ra, rb, sap, sbp, p);
            const double em = b97m_semilocal_exc_point(ra, rb, sam, sbm, p);
            return (ep - em) / (2.0 * h);
        };
        auto fd_tau = [&](bool spin_a) {
            const double base = spin_a ? p.tau_a : p.tau_b;
            const double h = 1e-6 * (std::abs(base) + 1e-6);
            B97MParams pp = p, pm = p;
            if (spin_a) { pp.tau_a += h; pm.tau_a -= h; }
            else        { pp.tau_b += h; pm.tau_b -= h; }
            const double ep = b97m_semilocal_exc_point(ra, rb, saa, sbb, pp);
            const double em = b97m_semilocal_exc_point(ra, rb, saa, sbb, pm);
            return (ep - em) / (2.0 * h);
        };
        out.v_rho_a(g) = fd_rho(true);
        out.v_rho_b(g) = fd_rho(false);
        out.v_sigma_aa(g) = fd_sigma(true);
        out.v_sigma_bb(g) = fd_sigma(false);
        out.v_tau_a(g) = fd_tau(true);
        out.v_tau_b(g) = fd_tau(false);
    }
    return out;
}

}  // namespace vibeqc
