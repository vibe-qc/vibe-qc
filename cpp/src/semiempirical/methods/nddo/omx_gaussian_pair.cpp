#include "vibeqc/semiempirical/methods/nddo/omx_gaussian_pair.hpp"
#include "vibeqc/basis.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/molecule.hpp"

#include <cmath>
#include <stdexcept>

namespace vibeqc::semiempirical::nddo {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

std::vector<double> append_shells(
    const OMxGaussianContraction& contraction,
    const std::array<double, 3>& origin, int atom,
    std::vector<ShellInfo>& shells) {
    const auto& alpha = contraction.exponents;
    require(!alpha.empty(), "OMx Gaussian pair: empty exponents");
    require(std::isfinite(contraction.zeta) && contraction.zeta > 0.0,
            "OMx Gaussian pair: zeta must be finite and positive");
    require(contraction.s_coefficients.size() == alpha.size(),
            "OMx Gaussian pair: s coefficient count must match exponents");
    require(contraction.p_coefficients.empty()
            || contraction.p_coefficients.size() == alpha.size(),
            "OMx Gaussian pair: p coefficient count must match exponents");
    std::vector<double> scaled;
    for (double exponent : alpha) {
        const double value = exponent * contraction.zeta * contraction.zeta;
        require(std::isfinite(exponent) && exponent > 0.0
                && std::isfinite(value) && value > 0.0,
                "OMx Gaussian pair: scaled exponents must be finite and positive");
        scaled.push_back(value);
    }
    std::vector<double> norms;
    const double pi = std::acos(-1.0);
    const int shell_count = contraction.p_coefficients.empty() ? 1 : 2;
    for (int l = 0; l < shell_count; ++l) {
        const auto& coefficients = l == 0 ? contraction.s_coefficients
                                         : contraction.p_coefficients;
        for (double c : coefficients)
            require(std::isfinite(c), "OMx Gaussian pair: nonfinite coefficient");
        double norm = 0.0;
        for (std::size_t i = 0; i < scaled.size(); ++i)
            for (std::size_t j = 0; j < scaled.size(); ++j) {
                const double ratio = 2.0 * std::sqrt(scaled[i]) * std::sqrt(scaled[j])
                                     / (scaled[i] + scaled[j]);
                norm += coefficients[i] * coefficients[j] * std::pow(ratio, l + 1.5);
            }
        require(std::isfinite(norm) && norm > 0.0,
                "OMx Gaussian pair: contraction must have a positive finite norm");
        norms.push_back(norm);
        ShellInfo shell{atom, l, false, scaled, {}, origin};
        for (std::size_t i = 0; i < scaled.size(); ++i) {
            // Normalized Cartesian s or p_x/y/z primitives. Pass these
            // coefficients pre-normalized to avoid libint renormalizing twice.
            double primitive_norm = std::pow(2.0 * scaled[i] / pi, 0.75);
            if (l == 1) primitive_norm *= 2.0 * std::sqrt(scaled[i]);
            const double value = coefficients[i] * primitive_norm / std::sqrt(norm);
            require(std::isfinite(value), "OMx Gaussian pair: normalization overflow");
            shell.coefficients.push_back(value);
        }
        shells.push_back(std::move(shell));
    }
    return norms;
}

void validate_density(const Eigen::MatrixXd& density, Eigen::Index size) {
    require(density.rows() == size && density.cols() == size && density.allFinite(),
            "OMx Gaussian pair: density must be finite with the atom-local AO shape");
    require(density.isApprox(density.transpose(), 1e-12),
            "OMx Gaussian pair: density must be symmetric");
}

}  // namespace

// Hehre, Stewart & Pople, JCP 51, 2657 (1969), Table I, K=3 1s,
// DOI 10.1063/1.1672392. Unit Slater exponent; zeta is applied once below.
OMxGaussianContraction om2_gaussian_core(double zeta) {
    return {{2.22766, 0.405771, 0.109818},
            {0.154329, 0.535328, 0.444635}, {}, zeta};
}

OMxGaussianContraction om2_gaussian_valence(int z, double zeta) {
    if (z == 1) return om2_gaussian_core(zeta);
    // Kolb & Thiel, JCC 14, 775 (1993), p.783 Table I,
    // DOI 10.1002/jcc.540140704. Preserve printed five-decimal precision.
    // s and p coefficients multiply normalized primitives; append_shells
    // normalizes each contraction and scales primitive exponents by zeta^2.
    switch (z) {
    case 6: return {{2.64486, 0.54215, 0.14466},
                    {-0.19188, 0.61628, 0.54896},
                    {0.20259, 0.55830, 0.45514}, zeta};
    case 7: return {{3.68849, 0.77534, 0.20498},
                    {-0.19269, 0.61888, 0.54926},
                    {0.22281, 0.56032, 0.43859}, zeta};
    case 8: return {{4.78449, 0.99860, 0.25687},
                    {-0.19248, 0.66952, 0.50270},
                    {0.24158, 0.55890, 0.43160}, zeta};
    case 9: return {{6.01783, 1.25315, 0.31760},
                    {-0.18850, 0.69800, 0.47427},
                    {0.25667, 0.56013, 0.42139}, zeta};
    default: throw std::invalid_argument("OM2 Gaussian basis supports H/C/N/O/F only");
    }
}

Eigen::MatrixXd omx_gaussian_overlap(
    const OMxGaussianContraction& a, const OMxGaussianContraction& b,
    const std::array<double, 3>& origin_a, const std::array<double, 3>& origin_b) {
    for (int d = 0; d < 3; ++d)
        require(std::isfinite(origin_a[d]) && std::isfinite(origin_b[d]),
                "OMx Gaussian overlap: nonfinite coordinate");
    std::vector<ShellInfo> shells;
    append_shells(a, origin_a, 0, shells);
    append_shells(b, origin_b, 1, shells);
    const BasisSet basis(Molecule({Atom{1, origin_a}, Atom{1, origin_b}}),
                         shells, "<omx-explicit-overlap>", true);
    const Eigen::Index na = a.p_coefficients.empty() ? 1 : 4;
    const Eigen::Index nb = b.p_coefficients.empty() ? 1 : 4;
    return compute_overlap(basis).block(0, na, na, nb);
}

OMxGaussianPair evaluate_omx_gaussian_pair(
    const OMxGaussianContraction& a, const OMxGaussianContraction& b,
    const std::array<double, 3>& origin_a,
    const std::array<double, 3>& origin_b,
    double gamma_ss, double core_charge_a, double core_charge_b) {
    double r2 = 0.0;
    for (int d = 0; d < 3; ++d) {
        require(std::isfinite(origin_a[d]) && std::isfinite(origin_b[d]),
                "OMx Gaussian pair: nonfinite coordinate");
        const double delta = origin_b[d] - origin_a[d];
        r2 += delta * delta;
    }
    require(std::isfinite(r2) && r2 > 0.0,
            "OMx Gaussian pair: centers must be distinct at finite separation");
    require(std::isfinite(gamma_ss) && gamma_ss > 0.0,
            "OMx Gaussian pair: gamma_ss must be finite and positive");
    require(std::isfinite(core_charge_a) && core_charge_a >= 0.0
            && std::isfinite(core_charge_b) && core_charge_b >= 0.0,
            "OMx Gaussian pair: core charges must be finite and nonnegative");

    OMxGaussianPair out;
    std::vector<ShellInfo> shells;
    out.raw_norms_a = append_shells(a, origin_a, 0, shells);
    out.raw_norms_b = append_shells(b, origin_b, 1, shells);
    // Atomic numbers here are placeholders for the basis container only.
    // Attraction uses explicit unit nuclei below and the supplied core charges.
    const Molecule centers({Atom{1, origin_a}, Atom{1, origin_b}});
    const BasisSet basis(centers, shells, "<omx-explicit-pair>", true);
    const Eigen::Index na = a.p_coefficients.empty() ? 1 : 4;
    const Eigen::Index nb = b.p_coefficients.empty() ? 1 : 4;
    out.overlap = compute_overlap(basis).block(0, na, na, nb);
    const auto eri = compute_eri(basis);
    out.eri_unscaled.resize(na * na, nb * nb);
    for (Eigen::Index mu = 0; mu < na; ++mu)
        for (Eigen::Index nu = 0; nu < na; ++nu)
            for (Eigen::Index lam = 0; lam < nb; ++lam)
                for (Eigen::Index sig = 0; sig < nb; ++sig)
                    out.eri_unscaled(mu * na + nu, lam * nb + sig) =
                        eri(mu, nu, na + lam, na + sig);
    const double j_ssss = out.eri_unscaled(0, 0);
    require(std::isfinite(j_ssss) && j_ssss > 0.0,
            "OMx Gaussian pair: invalid unscaled ss integral");
    out.f_ko = gamma_ss / j_ssss;
    out.eri_scaled = out.f_ko * out.eri_unscaled;
    out.attraction_a_unscaled = core_charge_b * compute_nuclear(
        basis, Molecule({Atom{1, origin_b}}, 0, 2)).topLeftCorner(na, na);
    out.attraction_b_unscaled = core_charge_a * compute_nuclear(
        basis, Molecule({Atom{1, origin_a}}, 0, 2)).bottomRightCorner(nb, nb);
    out.attraction_a_scaled = out.f_ko * out.attraction_a_unscaled;
    out.attraction_b_scaled = out.f_ko * out.attraction_b_unscaled;
    out.core_energy = out.f_ko * core_charge_a * core_charge_b / std::sqrt(r2);
    if (!out.overlap.allFinite() || !out.eri_unscaled.allFinite()
        || !out.eri_scaled.allFinite() || !out.attraction_a_scaled.allFinite()
        || !out.attraction_b_scaled.allFinite() || !std::isfinite(out.core_energy))
        throw std::runtime_error("OMx Gaussian pair: nonfinite integral operand");
    return out;
}

Eigen::MatrixXd OMxGaussianPair::coulomb_a(const Eigen::MatrixXd& density_b) const {
    const Eigen::Index na = overlap.rows(), nb = overlap.cols();
    validate_density(density_b, nb);
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(na, na);
    for (Eigen::Index mu = 0; mu < na; ++mu)
        for (Eigen::Index nu = 0; nu < na; ++nu)
            for (Eigen::Index lam = 0; lam < nb; ++lam)
                for (Eigen::Index sig = 0; sig < nb; ++sig)
                    result(mu, nu) += density_b(lam, sig)
                        * eri_scaled(mu * na + nu, lam * nb + sig);
    return result;
}

Eigen::MatrixXd OMxGaussianPair::coulomb_b(const Eigen::MatrixXd& density_a) const {
    const Eigen::Index na = overlap.rows(), nb = overlap.cols();
    validate_density(density_a, na);
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(nb, nb);
    for (Eigen::Index lam = 0; lam < nb; ++lam)
        for (Eigen::Index sig = 0; sig < nb; ++sig)
            for (Eigen::Index mu = 0; mu < na; ++mu)
                for (Eigen::Index nu = 0; nu < na; ++nu)
                    result(lam, sig) += density_a(mu, nu)
                        * eri_scaled(mu * na + nu, lam * nb + sig);
    return result;
}

}  // namespace vibeqc::semiempirical::nddo
