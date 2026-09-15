#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vibeqc {

constexpr double CPCM_DIAG_ALPHA_BARE = 1.0694;
constexpr double CPCM_DIAG_ALPHA =
    CPCM_DIAG_ALPHA_BARE * 3.5449077018110318;  // sqrt(4*pi)

inline Eigen::MatrixXd build_cpcm_A_matrix(
        const Eigen::Ref<const Eigen::MatrixXd>& cavity_points,
        const Eigen::Ref<const Eigen::VectorXd>& cavity_weights) {
    if (cavity_points.cols() != 3) {
        throw std::invalid_argument("cavity_points must have shape (n_pts, 3)");
    }
    const int n = static_cast<int>(cavity_points.rows());
    if (cavity_weights.size() != n) {
        throw std::invalid_argument(
            "cavity_weights length must match cavity_points rows");
    }

    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        const double wi = cavity_weights(i);
        if (!(wi > 0.0)) {
            throw std::invalid_argument("cavity_weights must be positive");
        }
        A(i, i) = CPCM_DIAG_ALPHA / std::sqrt(wi);
        for (int j = i + 1; j < n; ++j) {
            const double r = (cavity_points.row(i) - cavity_points.row(j)).norm();
            if (!(r > 0.0)) {
                throw std::invalid_argument(
                    "CPCM cavity contains coincident off-diagonal points");
            }
            const double v = 1.0 / r;
            A(i, j) = v;
            A(j, i) = v;
        }
    }
    return A;
}

inline Eigen::MatrixXd build_cpcm_capped_A_matrix(
        const Eigen::Ref<const Eigen::MatrixXd>& cavity_points,
        const Eigen::Ref<const Eigen::VectorXd>& a_diag) {
    if (cavity_points.cols() != 3) {
        throw std::invalid_argument("cavity_points must have shape (n_pts, 3)");
    }
    const int n = static_cast<int>(cavity_points.rows());
    if (a_diag.size() != n) {
        throw std::invalid_argument("a_diag length must match cavity_points rows");
    }

    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        const double ai = a_diag(i);
        if (!(ai > 0.0)) {
            throw std::invalid_argument("a_diag entries must be positive");
        }
        A(i, i) = ai;
        for (int j = i + 1; j < n; ++j) {
            const double aj = a_diag(j);
            if (!(aj > 0.0)) {
                throw std::invalid_argument("a_diag entries must be positive");
            }
            const double r = (cavity_points.row(i) - cavity_points.row(j)).norm();
            if (!(r > 0.0)) {
                throw std::invalid_argument(
                    "CPCM cavity contains coincident off-diagonal points");
            }
            const double cap = 0.5 * std::min(ai, aj);
            const double v = std::min(1.0 / r, cap);
            A(i, j) = v;
            A(j, i) = v;
        }
    }
    return A;
}

// Klamt & Schuurmann, J. Chem. Soc. Perkin Trans. 2 799 (1993),
// doi:10.1039/P29930000799, p. 800: dielectric screening energies for a fixed
// geometry scale as f(e) = (e - 1)/(e + x), "where x is in the range 0-2".
// CPCM and COSMO are not two formulas but two points of that family:
//   x = 0    -> f = (e - 1)/e         Cossi-Rega-Scalmani-Barone 2003 (C-PCM)
//   x = 1/2  -> f = (e - 1)/(e + 1/2) Klamt-Schuurmann 1993 p. 801
// Keep this table in step with SCREENING_X in
// python/vibeqc/solvation/screening.py; the two are pinned bitwise equal by
// tests/test_solvation_screening.py so neither can be changed alone.
inline double cpcm_screening_x(const std::string& variant) {
    if (variant == "cpcm") {
        return 0.0;
    }
    if (variant == "cosmo") {
        return 0.5;
    }
    throw std::invalid_argument(
        "dielectric_factor: unknown variant (use 'cpcm' or 'cosmo')");
}

// f(e) = (e - 1)/(e + x), with the exact conductor limit. The conductor is
// the limit COSMO is derived from (Klamt 1993 eq. 2), so an infinite epsilon
// returns exactly 1 rather than the inf/inf NaN of the closed form.
inline double cpcm_screening_factor(double epsilon, double x) {
    if (!(epsilon > 1.0)) {
        throw std::invalid_argument("dielectric_factor: epsilon must be > 1");
    }
    if (std::isinf(epsilon)) {
        return 1.0;
    }
    return (epsilon - 1.0) / (epsilon + x);
}

inline double cpcm_dielectric_factor(double epsilon, const std::string& variant) {
    return cpcm_screening_factor(epsilon, cpcm_screening_x(variant));
}

struct CPCMSolveResult {
    Eigen::VectorXd q;
    Eigen::VectorXd V;
    double e_solv = 0.0;
    double epsilon = 0.0;
};

inline CPCMSolveResult solve_cpcm_apparent_charges(
        const Eigen::Ref<const Eigen::MatrixXd>& A,
        const Eigen::Ref<const Eigen::VectorXd>& V_at_cavity,
        double epsilon,
        const std::string& variant) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("solve_apparent_charges: A must be square");
    }
    if (V_at_cavity.size() != A.rows()) {
        throw std::invalid_argument(
            "solve_apparent_charges: V length does not match A shape");
    }
    const double f = cpcm_dielectric_factor(epsilon, variant);
    const Eigen::VectorXd rhs = -f * V_at_cavity;
    const Eigen::VectorXd q = A.partialPivLu().solve(rhs);
    if (!q.allFinite()) {
        throw std::runtime_error("solve_apparent_charges: native solve failed");
    }

    CPCMSolveResult out;
    out.q = q;
    out.V = V_at_cavity;
    out.e_solv = 0.5 * q.dot(V_at_cavity);
    out.epsilon = epsilon;
    return out;
}

}  // namespace vibeqc
