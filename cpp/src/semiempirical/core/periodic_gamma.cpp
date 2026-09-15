#include "vibeqc/semiempirical/core/periodic_gamma.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace vibeqc {
namespace semiempirical {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPi = 1.7724538509055160273;
// Series truncation target shared by every dimension: real-space
// alpha * r_cut = sqrt(30) (erfc < 1e-14) and reciprocal |k| <= alpha
// sqrt(120) (e^{-k^2/4 alpha^2} < e^-30).
constexpr double kRealCutFactor = 5.4772255750516612;   // sqrt(30)
constexpr double kRecipCutFactor = 10.954451150103322;  // sqrt(120)
// Near-degenerate window of the Elstner unequal-tau closed form; inside it
// the exact series through delta^4 is used (validated to <= 2e-12 absolute
// at the switch against a 50-digit evaluation of the closed form).
constexpr double kElstnerDegenerateWindow = 0.02;

// ---------------------------------------------------------------------------
// Elstner 1998 short-range function
// ---------------------------------------------------------------------------

// Equal-tau branch: Elstner et al. Eq. 18 / Appendix with tau_a = tau_b.
//   S = e^{-tau R} (1/R + 11 tau/16 + 3 tau^2 R/16 + tau^3 R^2/48)
double elstner_equal(double tau, double R) {
    return std::exp(-tau * R)
        * (1.0 / R + 0.6875 * tau + 0.1875 * tau * tau * R
           + tau * tau * tau * R * R / 48.0);
}

double elstner_equal_derivative(double tau, double R) {
    const double u = std::exp(-tau * R);
    const double poly = 1.0 / R + 0.6875 * tau + 0.1875 * tau * tau * R
        + tau * tau * tau * R * R / 48.0;
    const double dpoly = -1.0 / (R * R) + 0.1875 * tau * tau
        + tau * tau * tau * R / 24.0;
    return u * (dpoly - tau * poly);
}

// One half of the unequal-tau branch (Elstner Appendix, the bracket of
// gamma = 1/R - [...]; DFTB+ gammaSubExprn_):
//   e^{-x R} [ y^4 x / (2 (x^2 - y^2)^2) - (y^6 - 3 y^4 x^2) / (R (x^2 - y^2)^3) ]
double elstner_unequal_half(double x, double y, double R) {
    const double D = x * x - y * y;
    const double c1 = y * y * y * y * x / (2.0 * D * D);
    const double c2 = (std::pow(y, 6) - 3.0 * y * y * y * y * x * x)
        / (D * D * D);
    return std::exp(-x * R) * (c1 - c2 / R);
}

double elstner_unequal_half_derivative(double x, double y, double R) {
    const double D = x * x - y * y;
    const double c1 = y * y * y * y * x / (2.0 * D * D);
    const double c2 = (std::pow(y, 6) - 3.0 * y * y * y * y * x * x)
        / (D * D * D);
    return std::exp(-x * R) * (-x * (c1 - c2 / R) + c2 / (R * R));
}

// Exact series of S in delta = (tau_a - tau_b)/2 about the mean tau, through
// delta^4:  S = S_eq(tau) + delta^2 C2 + delta^4 C4 with
//   C2 = u [12/tau + 12 R + 5 tau R^2 + tau^2 R^3 + tau^3 R^4/15] / 32
//   C4 = u [-2/tau^3 - 2 R/tau^2 + 2 R^3/3 + 19 tau R^4/60 + tau^2 R^5/20
//           + tau^3 R^6/420] / 32,  u = e^{-tau R}.
// Derived from the odd part of e^{delta R} (tau + delta)^4 B(delta) (the
// Appendix bracket); the delta^1 coefficient vanishes identically and the
// delta^3 coefficient reproduces the equal-tau branch, which is the
// consistency check on the derivation.
void elstner_series_coefficients(double tau, double R, double* c2,
                                 double* c4, double* dc2, double* dc4) {
    const double u = std::exp(-tau * R);
    const double t2 = tau * tau, t3 = t2 * tau;
    const double R2 = R * R, R3 = R2 * R, R4 = R3 * R, R5 = R4 * R,
                 R6 = R5 * R;
    const double p2 = 12.0 / tau + 12.0 * R + 5.0 * tau * R2 + t2 * R3
        + t3 * R4 / 15.0;
    const double dp2 = 12.0 + 10.0 * tau * R + 3.0 * t2 * R2
        + 4.0 * t3 * R3 / 15.0;
    const double p4 = -2.0 / t3 - 2.0 * R / t2 + 2.0 * R3 / 3.0
        + 19.0 * tau * R4 / 60.0 + t2 * R5 / 20.0 + t3 * R6 / 420.0;
    const double dp4 = -2.0 / t2 + 2.0 * R2 + 19.0 * tau * R3 / 15.0
        + t2 * R4 / 4.0 + t3 * R5 / 70.0;
    *c2 = u * p2 / 32.0;
    *c4 = u * p4 / 32.0;
    *dc2 = u * (dp2 - tau * p2) / 32.0;
    *dc4 = u * (dp4 - tau * p4) / 32.0;
}

bool elstner_degenerate(double tau_a, double tau_b) {
    return std::abs(tau_a - tau_b)
        <= kElstnerDegenerateWindow * std::max(tau_a, tau_b);
}

void require_elstner_arguments(double tau_a, double tau_b, double R) {
    if (!(tau_a > 0.0) || !(tau_b > 0.0) || !std::isfinite(tau_a)
        || !std::isfinite(tau_b)) {
        throw std::invalid_argument(
            "Elstner short-range gamma requires positive finite tau");
    }
    if (!(R > 0.0) || !std::isfinite(R)) {
        throw std::invalid_argument(
            "Elstner short-range gamma requires a positive finite distance");
    }
}

// ---------------------------------------------------------------------------
// Klopman-Ohno damping length
// ---------------------------------------------------------------------------

double klopman_ohno_eta(KlopmanOhnoAverage average, double U_a, double U_b) {
    switch (average) {
        case KlopmanOhnoAverage::HardnessMean:
            return 2.0 / (U_a + U_b);
        case KlopmanOhnoAverage::InverseHardnessMean:
            return 0.5 * (1.0 / U_a + 1.0 / U_b);
    }
    return 2.0 / (U_a + U_b);
}

void require_hardness(double U_a, double U_b) {
    if (!(U_a > 0.0) || !(U_b > 0.0) || !std::isfinite(U_a)
        || !std::isfinite(U_b)) {
        throw std::invalid_argument(
            "shell gamma requires positive finite hardnesses");
    }
}

constexpr double kElstnerTauPerHardness = 3.2;  // 16/5, Elstner Eq. 18

}  // namespace

// ---------------------------------------------------------------------------
// Scalar kernels
// ---------------------------------------------------------------------------

double elstner_short_range(double tau_a, double tau_b, double R) {
    require_elstner_arguments(tau_a, tau_b, R);
    if (tau_a == tau_b) return elstner_equal(tau_a, R);
    if (elstner_degenerate(tau_a, tau_b)) {
        const double tau = 0.5 * (tau_a + tau_b);
        const double delta = 0.5 * (tau_a - tau_b);
        double c2, c4, dc2, dc4;
        elstner_series_coefficients(tau, R, &c2, &c4, &dc2, &dc4);
        return elstner_equal(tau, R) + delta * delta * (c2 + delta * delta * c4);
    }
    return elstner_unequal_half(tau_a, tau_b, R)
        + elstner_unequal_half(tau_b, tau_a, R);
}

double elstner_short_range_derivative(double tau_a, double tau_b, double R) {
    require_elstner_arguments(tau_a, tau_b, R);
    if (tau_a == tau_b) return elstner_equal_derivative(tau_a, R);
    if (elstner_degenerate(tau_a, tau_b)) {
        const double tau = 0.5 * (tau_a + tau_b);
        const double delta = 0.5 * (tau_a - tau_b);
        double c2, c4, dc2, dc4;
        elstner_series_coefficients(tau, R, &c2, &c4, &dc2, &dc4);
        return elstner_equal_derivative(tau, R)
            + delta * delta * (dc2 + delta * delta * dc4);
    }
    return elstner_unequal_half_derivative(tau_a, tau_b, R)
        + elstner_unequal_half_derivative(tau_b, tau_a, R);
}

double shell_gamma_onsite(const ShellGammaSpec& spec, double U_a, double U_b) {
    require_hardness(U_a, U_b);
    // The on-site block is the method's published parameterisation for both
    // functional forms: the R -> 0 limit of the Klopman-Ohno kernel, which
    // for one shell per atom is exactly U (also the Elstner equal-tau limit
    // 5 tau/16 = U).
    return 1.0 / klopman_ohno_eta(spec.ko_average, U_a, U_b);
}

double shell_gamma_pair(const ShellGammaSpec& spec, double U_a, double U_b,
                        double R) {
    require_hardness(U_a, U_b);
    switch (spec.form) {
        case ShellGammaForm::KlopmanOhno: {
            const double eta = klopman_ohno_eta(spec.ko_average, U_a, U_b);
            return 1.0 / std::sqrt(R * R + eta * eta);
        }
        case ShellGammaForm::Elstner:
            return 1.0 / R
                - elstner_short_range(kElstnerTauPerHardness * U_a,
                                      kElstnerTauPerHardness * U_b, R);
    }
    return 0.0;
}

double shell_gamma_pair_derivative(const ShellGammaSpec& spec, double U_a,
                                   double U_b, double R) {
    require_hardness(U_a, U_b);
    switch (spec.form) {
        case ShellGammaForm::KlopmanOhno: {
            const double eta = klopman_ohno_eta(spec.ko_average, U_a, U_b);
            const double g = 1.0 / std::sqrt(R * R + eta * eta);
            return -R * g * g * g;
        }
        case ShellGammaForm::Elstner:
            return -1.0 / (R * R)
                - elstner_short_range_derivative(
                    kElstnerTauPerHardness * U_a,
                    kElstnerTauPerHardness * U_b, R);
    }
    return 0.0;
}

double shell_gamma_remainder(const ShellGammaSpec& spec, double U_a,
                             double U_b, double R) {
    require_hardness(U_a, U_b);
    switch (spec.form) {
        case ShellGammaForm::KlopmanOhno: {
            const double eta = klopman_ohno_eta(spec.ko_average, U_a, U_b);
            // 1/sqrt(R^2 + eta^2) - 1/R, written without cancellation:
            //   = -eta^2 / (R sqrt(R^2 + eta^2) (R + sqrt(R^2 + eta^2)))
            const double s = std::sqrt(R * R + eta * eta);
            return -eta * eta / (R * s * (R + s));
        }
        case ShellGammaForm::Elstner:
            return -elstner_short_range(kElstnerTauPerHardness * U_a,
                                        kElstnerTauPerHardness * U_b, R);
    }
    return 0.0;
}

double shell_gamma_remainder_derivative(const ShellGammaSpec& spec,
                                        double U_a, double U_b, double R) {
    require_hardness(U_a, U_b);
    switch (spec.form) {
        case ShellGammaForm::KlopmanOhno: {
            const double eta = klopman_ohno_eta(spec.ko_average, U_a, U_b);
            const double g = 1.0 / std::sqrt(R * R + eta * eta);
            return -R * g * g * g + 1.0 / (R * R);
        }
        case ShellGammaForm::Elstner:
            return -elstner_short_range_derivative(
                kElstnerTauPerHardness * U_a,
                kElstnerTauPerHardness * U_b, R);
    }
    return 0.0;
}

double shell_gamma_remainder_range(const ShellGammaSpec& spec, double U_a,
                                   double U_b, double tolerance) {
    require_hardness(U_a, U_b);
    if (!(tolerance > 0.0)) {
        throw std::invalid_argument(
            "shell_gamma_remainder_range requires a positive tolerance");
    }
    if (spec.form == ShellGammaForm::KlopmanOhno) {
        return std::numeric_limits<double>::infinity();
    }
    const double tau_a = kElstnerTauPerHardness * U_a;
    const double tau_b = kElstnerTauPerHardness * U_b;
    // S is positive and monotonically decreasing beyond a few bohr; bisect
    // |S(R)| = tolerance on [lo, hi] after bracketing.
    double lo = 1.0;
    double hi = 8.0;
    while (elstner_short_range(tau_a, tau_b, hi) > tolerance) {
        lo = hi;
        hi *= 2.0;
        if (hi > 1.0e5) return hi;
    }
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (elstner_short_range(tau_a, tau_b, mid) > tolerance) lo = mid;
        else hi = mid;
        if (hi - lo < 1.0e-6) break;
    }
    return hi;
}

// ---------------------------------------------------------------------------
// Ewald-summed lattice potential
// ---------------------------------------------------------------------------

namespace {

// Gauss-Legendre nodes and weights on [-1, 1] for the 1-D wire channels
// (the quadrature of seccm/ewald_1d.h, computed once per order).
const std::pair<std::vector<double>, std::vector<double>>&
gauss_legendre(int order) {
    static std::vector<std::pair<
        int, std::pair<std::vector<double>, std::vector<double>>>> cache;
    for (const auto& entry : cache) {
        if (entry.first == order) return entry.second;
    }
    std::vector<double> x(order), w(order);
    for (int i = 0; i < order; ++i) {
        double xi = std::cos(kPi * (i + 0.75) / (order + 0.5));
        for (int it = 0; it < 64; ++it) {
            double p0 = 1.0, p1 = xi;
            for (int j = 2; j <= order; ++j) {
                const double p2 =
                    ((2.0 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j;
                p0 = p1;
                p1 = p2;
            }
            const double dp = order * (xi * p1 - p0) / (xi * xi - 1.0);
            const double dx = p1 / dp;
            xi -= dx;
            if (std::abs(dx) < 1e-15) break;
        }
        double p0 = 1.0, p1 = xi;
        for (int j = 2; j <= order; ++j) {
            const double p2 = ((2.0 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j;
            p0 = p1;
            p1 = p2;
        }
        const double dp = order * (xi * p1 - p0) / (xi * xi - 1.0);
        x[i] = xi;
        w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
    }
    cache.emplace_back(order, std::make_pair(std::move(x), std::move(w)));
    return cache.back().second;
}

constexpr int kWireQuadratureOrder = 64;

// F(k, rho) = 2 int_0^alpha exp(-u^2 rho^2 - k^2/(4 u^2)) du/u on
// v = u/alpha in [0, 1] (the Jacobian cancels the factor 2), and its
// rho-derivative dF/drho = -2 rho alpha^2 sum_j w_j v_j exp(...).
void wire_F(double k, double rho, double alpha, double* value,
            double* d_rho) {
    const auto& gl = gauss_legendre(kWireQuadratureOrder);
    const double a2 = alpha * alpha;
    const double b = k * k / (4.0 * a2);
    double s = 0.0, ds = 0.0;
    for (int j = 0; j < kWireQuadratureOrder; ++j) {
        const double v = 0.5 * (gl.first[j] + 1.0);
        const double e = std::exp(-a2 * rho * rho * v * v - b / (v * v));
        s += gl.second[j] * e / v;
        ds += gl.second[j] * v * e;
    }
    *value = s;
    *d_rho = -2.0 * rho * a2 * ds;
}

// m0(rho): the z-averaged m = 0 channel of the background-subtracted erf
// part (seccm/ewald_1d.h convention) and its rho-derivative.
void wire_m0(double rho, double L, double alpha, double* value,
             double* d_rho) {
    const auto& gl = gauss_legendre(kWireQuadratureOrder);
    const double euler_gamma = 0.5772156649015328606;
    const double tpre = 2.0 * alpha / kSqrtPi;
    const double a2 = alpha * alpha;
    double i1 = 0.0, di1 = 0.0;
    for (int j = 0; j < kWireQuadratureOrder; ++j) {
        const double t = 0.5 * (gl.first[j] + 1.0);
        const double rt = std::sqrt(rho * rho + L * L * t * t);
        i1 += gl.second[j] * 2.0 * L * std::erf(alpha * rt) / rt;
        const double dkern = tpre * std::exp(-a2 * rt * rt) / rt
            - std::erf(alpha * rt) / (rt * rt);
        di1 += gl.second[j] * 2.0 * L * (rho / rt) * dkern;
    }
    i1 *= 0.5;
    di1 *= 0.5;
    double i2 = 0.0, di2 = 0.0;
    for (int j = 0; j < kWireQuadratureOrder; ++j) {
        const double s = 0.5 * (gl.first[j] + 1.0);
        const double t = 1.0 / s;
        const double rt = std::sqrt(rho * rho + L * L * t * t);
        const double g = 2.0 * L * (std::erf(alpha * rt) / rt - 1.0 / (L * t));
        i2 += gl.second[j] * g / (s * s);
        const double dkern = tpre * std::exp(-a2 * rt * rt) / rt
            - std::erf(alpha * rt) / (rt * rt);
        di2 += gl.second[j] * 2.0 * L * (rho / rt) * dkern / (s * s);
    }
    i2 *= 0.5;
    di2 *= 0.5;
    *value = (i1 + i2 - 2.0 * euler_gamma) / L;
    *d_rho = (di1 + di2) / L;
}

}  // namespace

EwaldCoulombKernel::EwaldCoulombKernel(
    const std::vector<Eigen::Vector3d>& translations, double alpha)
    : dim_(static_cast<int>(translations.size())),
      translations_(translations) {
    if (dim_ < 1 || dim_ > 3) {
        throw std::invalid_argument(
            "EwaldCoulombKernel requires one to three lattice translations");
    }
    lattice_.resize(dim_, 3);
    for (int axis = 0; axis < dim_; ++axis) {
        if (!translations_[static_cast<std::size_t>(axis)].allFinite()) {
            throw std::invalid_argument(
                "EwaldCoulombKernel translations must be finite");
        }
        lattice_.row(axis) =
            translations_[static_cast<std::size_t>(axis)].transpose();
    }
    const Eigen::MatrixXd gram = lattice_ * lattice_.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(lattice_);
    const double sigma_min = svd.singularValues()(dim_ - 1);
    if (!(sigma_min > 1.0e-10 * std::max(1.0, svd.singularValues()(0)))) {
        throw std::invalid_argument(
            "EwaldCoulombKernel translations must be linearly independent");
    }
    gram_inverse_ = gram.inverse();

    // Dual (reciprocal) vectors of the periodic sublattice: rows of
    // 2 pi (L L^T)^-1 L satisfy b_i . T_j = 2 pi delta_ij.
    const Eigen::MatrixXd dual = gram_inverse_ * lattice_;  // dim x 3

    if (dim_ == 3) {
        volume_ = std::abs(lattice_.determinant());
        alpha_ = alpha > 0.0 ? alpha : kSqrtPi / std::cbrt(volume_);
    } else if (dim_ == 2) {
        const Eigen::Vector3d cross =
            translations_[0].cross(translations_[1]);
        area_ = cross.norm();
        normal_ = cross / area_;
        alpha_ = alpha > 0.0 ? alpha : 0.85 * kSqrtPi / std::sqrt(area_);
    } else {
        length_ = translations_[0].norm();
        axis_ = translations_[0] / length_;
        alpha_ = alpha > 0.0 ? alpha : 4.0 / length_;
    }
    r_cut_ = kRealCutFactor / alpha_;

    // Real-space box: the reduced pair displacement satisfies
    // |f_i| <= 1/2 in fractional coordinates, so |d_reduced| is bounded by
    // half the sum of the translation lengths; any g with |d + g| <= r_cut
    // then has |g| <= r_cut + that bound, and the dual-plane inequality
    // |n_i| <= |g| |b_i| / (2 pi) bounds the coefficients.
    double d_bound = 0.0;
    for (int axis = 0; axis < dim_; ++axis) {
        d_bound += 0.5 * translations_[static_cast<std::size_t>(axis)].norm();
    }
    n_real_.assign(3, 0);
    for (int axis = 0; axis < dim_; ++axis) {
        n_real_[static_cast<std::size_t>(axis)] = static_cast<int>(std::ceil(
            (r_cut_ + d_bound) * dual.row(axis).norm())) + 1;
    }

    const double k_cut = kRecipCutFactor * alpha_;
    if (dim_ == 3) {
        const Eigen::MatrixXd recip = 2.0 * kPi * dual;  // rows = b_i
        int gmax[3];
        for (int axis = 0; axis < 3; ++axis) {
            gmax[axis] = static_cast<int>(std::ceil(
                k_cut * translations_[static_cast<std::size_t>(axis)].norm()
                / (2.0 * kPi))) + 1;
        }
        for (int i = -gmax[0]; i <= gmax[0]; ++i) {
            for (int j = -gmax[1]; j <= gmax[1]; ++j) {
                for (int k = -gmax[2]; k <= gmax[2]; ++k) {
                    if (i == 0 && j == 0 && k == 0) continue;
                    const Eigen::Vector3d g = static_cast<double>(i)
                            * recip.row(0).transpose()
                        + static_cast<double>(j) * recip.row(1).transpose()
                        + static_cast<double>(k) * recip.row(2).transpose();
                    const double g2 = g.squaredNorm();
                    if (g2 > k_cut * k_cut) continue;
                    reciprocal_.push_back({
                        g,
                        4.0 * kPi / volume_
                            * std::exp(-g2 / (4.0 * alpha_ * alpha_)) / g2,
                    });
                }
            }
        }
        // Neutralising uniform background: with it the Ewald lattice
        // potential is alpha-independent element by element (without it
        // only neutral contractions are), the tblite/xtb convention.
        background_ = -kPi / (volume_ * alpha_ * alpha_);
    } else if (dim_ == 2) {
        const Eigen::MatrixXd recip = 2.0 * kPi * dual;
        int kmax[2];
        for (int axis = 0; axis < 2; ++axis) {
            kmax[axis] = static_cast<int>(std::ceil(
                k_cut * translations_[static_cast<std::size_t>(axis)].norm()
                / (2.0 * kPi))) + 1;
        }
        for (int i = -kmax[0]; i <= kmax[0]; ++i) {
            for (int j = -kmax[1]; j <= kmax[1]; ++j) {
                if (i == 0 && j == 0) continue;
                const Eigen::Vector3d k = static_cast<double>(i)
                        * recip.row(0).transpose()
                    + static_cast<double>(j) * recip.row(1).transpose();
                if (k.squaredNorm() > k_cut * k_cut) continue;
                kvecs_.push_back(k);
            }
        }
    } else {
        m_max_ = static_cast<int>(std::ceil(k_cut * length_ / (2.0 * kPi))) + 1;
    }
}

Eigen::Vector3d EwaldCoulombKernel::reduce(const Eigen::Vector3d& d) const {
    // Fractional coordinates of the periodic-subspace projection of d, each
    // reduced to [-1/2, 1/2]; the perpendicular component is untouched.
    const Eigen::VectorXd fractional = gram_inverse_ * (lattice_ * d);
    Eigen::Vector3d reduced = d;
    for (int axis = 0; axis < dim_; ++axis) {
        const double shift = std::nearbyint(fractional(axis));
        if (shift != 0.0) {
            reduced -= shift * translations_[static_cast<std::size_t>(axis)];
        }
    }
    return reduced;
}

void EwaldCoulombKernel::real_space(
    const Eigen::Vector3d& d, bool exclude_self,
    double* value, Eigen::Vector3d* grad) const {
    const double a2 = alpha_ * alpha_;
    const double tpre = 2.0 * alpha_ / kSqrtPi;
    const int n0 = n_real_[0];
    const int n1 = dim_ >= 2 ? n_real_[1] : 0;
    const int n2 = dim_ >= 3 ? n_real_[2] : 0;
    double sum = 0.0;
    Eigen::Vector3d gsum = Eigen::Vector3d::Zero();
    for (int i = -n0; i <= n0; ++i) {
        for (int j = -n1; j <= n1; ++j) {
            for (int k = -n2; k <= n2; ++k) {
                Eigen::Vector3d r = d + static_cast<double>(i) * translations_[0];
                if (dim_ >= 2) r += static_cast<double>(j) * translations_[1];
                if (dim_ >= 3) r += static_cast<double>(k) * translations_[2];
                const double rn = r.norm();
                if (rn < 1.0e-12) {
                    if (!exclude_self) {
                        throw std::invalid_argument(
                            "EwaldCoulombKernel: coincident sites in a pair "
                            "that is not marked same-site");
                    }
                    continue;
                }
                if (rn > r_cut_) continue;
                const double erfc_term = std::erfc(alpha_ * rn) / rn;
                sum += erfc_term;
                if (grad) {
                    const double pref = (erfc_term
                        + tpre * std::exp(-a2 * rn * rn)) / (rn * rn);
                    gsum -= r * pref;
                }
            }
        }
    }
    if (value) *value += sum;
    if (grad) *grad += gsum;
}

void EwaldCoulombKernel::reciprocal_3d(
    const Eigen::Vector3d& d, double* value, Eigen::Vector3d* grad) const {
    double sum = 0.0;
    Eigen::Vector3d gsum = Eigen::Vector3d::Zero();
    for (const auto& term : reciprocal_) {
        const double phase = term.g.dot(d);
        sum += term.weight * std::cos(phase);
        if (grad) gsum -= term.weight * std::sin(phase) * term.g;
    }
    if (value) *value += sum + background_;
    if (grad) *grad += gsum;
}

void EwaldCoulombKernel::reciprocal_2d(
    const Eigen::Vector3d& d, double* value, Eigen::Vector3d* grad) const {
    // Parry/Heyes slab sum (the validated indo::_madkonst_2d arithmetic):
    //   (pi/A) sum_K cos(K.d)/K [e^{Kz} erfc(az + K/2a) + e^{-Kz} erfc(-az + K/2a)]
    //   - (2 pi/A) [z erf(az) + e^{-a^2 z^2} / (a sqrt(pi))]
    // with z = d.n.  The z-derivative of the bracket is
    //   K [e^{Kz} erfc(az + K/2a) - e^{-Kz} erfc(-az + K/2a)]
    // (the two Gaussian terms of the erfc derivatives cancel exactly), and
    // d/dz of the K = 0 term is -(2 pi/A) erf(az).
    const double z = d.dot(normal_);
    const double recip_pref = kPi / area_;
    double sum = 0.0;
    Eigen::Vector3d gsum = Eigen::Vector3d::Zero();
    for (const auto& k : kvecs_) {
        const double km = k.norm();
        const double phase = k.dot(d);
        const double plus = std::exp(km * z)
            * std::erfc(alpha_ * z + km / (2.0 * alpha_));
        const double minus = std::exp(-km * z)
            * std::erfc(-alpha_ * z + km / (2.0 * alpha_));
        sum += recip_pref * std::cos(phase) / km * (plus + minus);
        if (grad) {
            gsum -= recip_pref * std::sin(phase) / km * (plus + minus) * k;
            gsum += recip_pref * std::cos(phase) * (plus - minus) * normal_;
        }
    }
    const double az = alpha_ * z;
    const double k0_pref = 2.0 * kPi / area_;
    sum -= k0_pref * (z * std::erf(az)
                      + std::exp(-az * az) / (alpha_ * kSqrtPi));
    if (grad) gsum -= k0_pref * std::erf(az) * normal_;
    if (value) *value += sum;
    if (grad) *grad += gsum;
}

void EwaldCoulombKernel::reciprocal_1d(
    const Eigen::Vector3d& d, double* value, Eigen::Vector3d* grad) const {
    // Background-corrected Parry-type wire sum (seccm/ewald_1d.h):
    //   (2/L) sum_{m>=1} cos(k_m z) F(k_m, rho) + m0(rho),
    // z the axial and rho the radial component of d.
    const double z = d.dot(axis_);
    const Eigen::Vector3d inplane = d - z * axis_;
    const double rho = inplane.norm();
    double sum = 0.0;
    double d_z = 0.0;
    double d_rho = 0.0;
    for (int m = 1; m <= m_max_; ++m) {
        const double km = 2.0 * kPi * static_cast<double>(m) / length_;
        double F, dF;
        wire_F(km, rho, alpha_, &F, &dF);
        const double c = std::cos(km * z);
        sum += c * F;
        d_z -= km * std::sin(km * z) * F;
        d_rho += c * dF;
    }
    sum *= 2.0 / length_;
    d_z *= 2.0 / length_;
    d_rho *= 2.0 / length_;
    double m0, dm0;
    wire_m0(rho, length_, alpha_, &m0, &dm0);
    sum += m0;
    d_rho += dm0;
    if (value) *value += sum;
    if (grad) {
        *grad += d_z * axis_;
        // The radial channels vanish like rho at rho -> 0, where the
        // in-plane direction is undefined but the contribution is zero.
        if (rho > 1.0e-12) *grad += d_rho * (inplane / rho);
    }
}

double EwaldCoulombKernel::potential(
    const Eigen::Vector3d& d, bool exclude_self) const {
    const Eigen::Vector3d reduced = reduce(d);
    double value = 0.0;
    real_space(reduced, exclude_self, &value, nullptr);
    if (dim_ == 3) reciprocal_3d(reduced, &value, nullptr);
    else if (dim_ == 2) reciprocal_2d(reduced, &value, nullptr);
    else reciprocal_1d(reduced, &value, nullptr);
    if (exclude_self) value -= 2.0 * alpha_ / kSqrtPi;
    return value;
}

Eigen::Vector3d EwaldCoulombKernel::gradient(
    const Eigen::Vector3d& d, bool exclude_self) const {
    const Eigen::Vector3d reduced = reduce(d);
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    real_space(reduced, exclude_self, nullptr, &grad);
    if (dim_ == 3) reciprocal_3d(reduced, nullptr, &grad);
    else if (dim_ == 2) reciprocal_2d(reduced, nullptr, &grad);
    else reciprocal_1d(reduced, nullptr, &grad);
    return grad;
}

// ---------------------------------------------------------------------------
// Record-driven assembly
// ---------------------------------------------------------------------------

namespace {

struct SiteLayout {
    int n_atoms = 0;
    std::vector<std::vector<int>> sites_of_atom;
};

SiteLayout site_layout(const std::vector<GammaSite>& sites,
                       const std::vector<Eigen::Vector3d>& atom_positions) {
    SiteLayout layout;
    layout.n_atoms = static_cast<int>(atom_positions.size());
    layout.sites_of_atom.assign(static_cast<std::size_t>(layout.n_atoms), {});
    for (int i = 0; i < static_cast<int>(sites.size()); ++i) {
        const int atom = sites[static_cast<std::size_t>(i)].atom;
        if (atom < 0 || atom >= layout.n_atoms) {
            throw std::invalid_argument(
                "shell gamma site refers to an out-of-range atom");
        }
        layout.sites_of_atom[static_cast<std::size_t>(atom)].push_back(i);
    }
    return layout;
}

Eigen::Vector3d record_displacement(
    const ImageRecord& record,
    const std::vector<Eigen::Vector3d>& atom_positions) {
    const int n_atoms = static_cast<int>(atom_positions.size());
    if (record.a < 0 || record.a >= n_atoms || record.b < 0
        || record.b >= n_atoms) {
        throw std::invalid_argument(
            "shell gamma image record refers to an out-of-range atom");
    }
    if (!record.shift.allFinite() || !std::isfinite(record.weight)) {
        throw std::invalid_argument(
            "shell gamma image record must be finite");
    }
    return atom_positions[static_cast<std::size_t>(record.b)] + record.shift
        - atom_positions[static_cast<std::size_t>(record.a)];
}

}  // namespace

Eigen::MatrixXd build_periodic_shell_gamma(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec,
    const EwaldCoulombKernel& ewald,
    const ImageRecordSource& records) {
    const SiteLayout layout = site_layout(sites, atom_positions);
    const int n_sites = static_cast<int>(sites.size());
    const int n_atoms = layout.n_atoms;

    // Ewald point-Coulomb potential between every atom pair (lattice
    // periodic in the displacement, even under d -> -d).
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(n_atoms, n_atoms);
    for (int a = 0; a < n_atoms; ++a) {
        for (int b = a; b < n_atoms; ++b) {
            const Eigen::Vector3d d = atom_positions[static_cast<std::size_t>(b)]
                - atom_positions[static_cast<std::size_t>(a)];
            const double value = ewald.potential(d, a == b);
            phi(a, b) = value;
            phi(b, a) = value;
        }
    }

    Eigen::MatrixXd gamma(n_sites, n_sites);
    for (int i = 0; i < n_sites; ++i) {
        for (int j = 0; j < n_sites; ++j) {
            gamma(i, j) = phi(sites[static_cast<std::size_t>(i)].atom,
                              sites[static_cast<std::size_t>(j)].atom);
        }
    }

    // Short-range remainder over the directed image records.
    records([&](const ImageRecord& record) {
        const Eigen::Vector3d v = record_displacement(record, atom_positions);
        const double r = v.norm();
        if (r < 1.0e-12) {
            throw std::invalid_argument(
                "shell gamma image record has a zero pair displacement");
        }
        for (int i : layout.sites_of_atom[static_cast<std::size_t>(record.a)]) {
            for (int j :
                 layout.sites_of_atom[static_cast<std::size_t>(record.b)]) {
                gamma(i, j) += record.weight * shell_gamma_remainder(
                    spec, sites[static_cast<std::size_t>(i)].hardness,
                    sites[static_cast<std::size_t>(j)].hardness, r);
            }
        }
    });

    // On-site block.
    for (int a = 0; a < n_atoms; ++a) {
        const auto& own = layout.sites_of_atom[static_cast<std::size_t>(a)];
        for (int i : own) {
            for (int j : own) {
                gamma(i, j) += shell_gamma_onsite(
                    spec, sites[static_cast<std::size_t>(i)].hardness,
                    sites[static_cast<std::size_t>(j)].hardness);
            }
        }
    }
    return gamma;
}

Eigen::MatrixXd build_molecular_shell_gamma(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec) {
    const SiteLayout layout = site_layout(sites, atom_positions);
    const int n_sites = static_cast<int>(sites.size());
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(n_sites, n_sites);
    for (int i = 0; i < n_sites; ++i) {
        const int a = sites[static_cast<std::size_t>(i)].atom;
        for (int j = 0; j < n_sites; ++j) {
            const int b = sites[static_cast<std::size_t>(j)].atom;
            const double U_i = sites[static_cast<std::size_t>(i)].hardness;
            const double U_j = sites[static_cast<std::size_t>(j)].hardness;
            if (a == b) {
                gamma(i, j) = shell_gamma_onsite(spec, U_i, U_j);
            } else {
                const double R = (atom_positions[static_cast<std::size_t>(b)]
                    - atom_positions[static_cast<std::size_t>(a)]).norm();
                gamma(i, j) = shell_gamma_pair(spec, U_i, U_j, R);
            }
        }
    }
    (void)layout;
    return gamma;
}

Eigen::MatrixXd periodic_shell_gamma_gradient(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec,
    const EwaldCoulombKernel& ewald,
    const ImageRecordSource& records,
    const Eigen::VectorXd& dq) {
    const SiteLayout layout = site_layout(sites, atom_positions);
    const int n_sites = static_cast<int>(sites.size());
    const int n_atoms = layout.n_atoms;
    if (dq.size() != n_sites) {
        throw std::invalid_argument(
            "periodic_shell_gamma_gradient: dq length must equal the site count");
    }
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);

    // Ewald part: E_ew = 1/2 sum_ab q_a q_b Phi(R_b - R_a) with the atomic
    // charges q_a = sum_{i on a} dq_i.  For ordered a != b,
    //   dE/dR_b += 1/2 q_a q_b dPhi/dd,   dE/dR_a -= the same.
    Eigen::VectorXd q_atom = Eigen::VectorXd::Zero(n_atoms);
    for (int i = 0; i < n_sites; ++i) {
        q_atom(sites[static_cast<std::size_t>(i)].atom) += dq(i);
    }
    for (int a = 0; a < n_atoms; ++a) {
        for (int b = 0; b < n_atoms; ++b) {
            if (a == b) continue;
            const Eigen::Vector3d d = atom_positions[static_cast<std::size_t>(b)]
                - atom_positions[static_cast<std::size_t>(a)];
            const Eigen::Vector3d f = 0.5 * q_atom(a) * q_atom(b)
                * ewald.gradient(d, false);
            grad.row(b) += f.transpose();
            grad.row(a) -= f.transpose();
        }
    }

    // Remainder part: for each record, 1/2 Q_ab^w rem'(r) v/r with
    // Q_ab^w = w sum_{i on a, j on b} dq_i dq_j rem'_ij.
    records([&](const ImageRecord& record) {
        const Eigen::Vector3d v = record_displacement(record, atom_positions);
        const double r = v.norm();
        if (r < 1.0e-12) {
            throw std::invalid_argument(
                "shell gamma image record has a zero pair displacement");
        }
        double coefficient = 0.0;
        for (int i : layout.sites_of_atom[static_cast<std::size_t>(record.a)]) {
            for (int j :
                 layout.sites_of_atom[static_cast<std::size_t>(record.b)]) {
                coefficient += dq(i) * dq(j) * shell_gamma_remainder_derivative(
                    spec, sites[static_cast<std::size_t>(i)].hardness,
                    sites[static_cast<std::size_t>(j)].hardness, r);
            }
        }
        const Eigen::Vector3d f = 0.5 * record.weight * coefficient * v / r;
        grad.row(record.b) += f.transpose();
        grad.row(record.a) -= f.transpose();
    });
    return grad;
}

Eigen::Matrix3d periodic_shell_gamma_strain_derivative(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec,
    const EwaldCoulombKernel& ewald,
    const ImageRecordSource& records,
    const Eigen::VectorXd& dq) {
    const SiteLayout layout = site_layout(sites, atom_positions);
    const int n_sites = static_cast<int>(sites.size());
    const int n_atoms = layout.n_atoms;
    if (dq.size() != n_sites) {
        throw std::invalid_argument(
            "periodic_shell_gamma_strain_derivative: dq length must equal the "
            "site count");
    }
    Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();

    // Remainder part (analytic pair virial): the record displacement
    // v = R_b + shift - R_a strains homogeneously, so
    //   dE/deps_ij = 1/2 Q_ab^w rem'(r) v_i v_j / r.
    records([&](const ImageRecord& record) {
        const Eigen::Vector3d v = record_displacement(record, atom_positions);
        const double r = v.norm();
        if (r < 1.0e-12) {
            throw std::invalid_argument(
                "shell gamma image record has a zero pair displacement");
        }
        double coefficient = 0.0;
        for (int i : layout.sites_of_atom[static_cast<std::size_t>(record.a)]) {
            for (int j :
                 layout.sites_of_atom[static_cast<std::size_t>(record.b)]) {
                coefficient += dq(i) * dq(j) * shell_gamma_remainder_derivative(
                    spec, sites[static_cast<std::size_t>(i)].hardness,
                    sites[static_cast<std::size_t>(j)].hardness, r);
            }
        }
        strain += 0.5 * record.weight * coefficient / r * (v * v.transpose());
    });

    // Ewald part: central difference of the closed-form lattice potential
    // under the strained lattice and positions, alpha held fixed.
    Eigen::VectorXd q_atom = Eigen::VectorXd::Zero(n_atoms);
    for (int i = 0; i < n_sites; ++i) {
        q_atom(sites[static_cast<std::size_t>(i)].atom) += dq(i);
    }
    const auto ewald_energy = [&](const Eigen::Matrix3d& transform) {
        std::vector<Eigen::Vector3d> translations;
        translations.reserve(ewald.translations().size());
        for (const auto& t : ewald.translations()) {
            translations.push_back(transform * t);
        }
        const EwaldCoulombKernel strained(translations, ewald.alpha());
        double energy = 0.0;
        for (int a = 0; a < n_atoms; ++a) {
            for (int b = 0; b < n_atoms; ++b) {
                const Eigen::Vector3d d = transform
                    * (atom_positions[static_cast<std::size_t>(b)]
                       - atom_positions[static_cast<std::size_t>(a)]);
                energy += 0.5 * q_atom(a) * q_atom(b)
                    * strained.potential(d, a == b);
            }
        }
        return energy;
    };
    const double h = 1.0e-5;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Eigen::Matrix3d plus = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d minus = Eigen::Matrix3d::Identity();
            plus(i, j) += h;
            minus(i, j) -= h;
            strain(i, j) += (ewald_energy(plus) - ewald_energy(minus))
                / (2.0 * h);
        }
    }
    return strain;
}

}  // namespace semiempirical
}  // namespace vibeqc
