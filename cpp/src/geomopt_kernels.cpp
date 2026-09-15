// Geometry optimisation performance kernels — implementation.
//
// All functions operate on Eigen vectors/matrices.  pybind11 wrappers
// in bindings.cpp handle NumPy ↔ Eigen conversion automatically
// via pybind11/eigen.h.

#include "vibeqc/geomopt_kernels.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace vibeqc {
namespace geomopt {

// ---------------------------------------------------------------------------
// L-BFGS two-loop recursion (Nocedal 1980, algorithm 7.4)
// ---------------------------------------------------------------------------

std::pair<Eigen::VectorXd, bool> lbfgs_two_loop(
    const std::vector<Eigen::VectorXd>& s_list,
    const std::vector<Eigen::VectorXd>& y_list,
    const Eigen::VectorXd& g,
    double gamma0
) {
    const int m = static_cast<int>(s_list.size());
    if (m == 0) {
        // No history: steepest descent direction scaled by gamma0
        return {-gamma0 * g, true};
    }

    const int n = static_cast<int>(g.size());
    Eigen::VectorXd q = g;
    std::vector<double> alpha(m);

    // First loop: α_i = ρ_i · s_i^T · q,  q ← q - α_i · y_i
    for (int i = 0; i < m; ++i) {
        double rho_i = 1.0 / y_list[i].dot(s_list[i]);
        alpha[i] = rho_i * s_list[i].dot(q);
        q -= alpha[i] * y_list[i];
    }

    // Initial Hessian scaling: γ = s_0^T y_0 / y_0^T y_0
    double gamma = gamma0;
    if (m > 0) {
        double sy = y_list[0].dot(s_list[0]);
        double yy = y_list[0].dot(y_list[0]);
        if (yy > 1e-30) {
            gamma = sy / yy;
        }
    }

    Eigen::VectorXd r = gamma * q;

    // Second loop: β = ρ_i · y_i^T · r,  r ← r + s_i · (α_i - β)
    for (int i = m - 1; i >= 0; --i) {
        double rho_i = 1.0 / y_list[i].dot(s_list[i]);
        double beta = rho_i * y_list[i].dot(r);
        r += s_list[i] * (alpha[i] - beta);
    }

    return {-r, true};
}

// ---------------------------------------------------------------------------
// BFGS inverse-Hessian update (Sherman-Morrison)
// ---------------------------------------------------------------------------

Eigen::MatrixXd bfgs_inverse_update(
    const Eigen::MatrixXd& H_inv,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
) {
    double sy = y.dot(s);
    if (sy <= 1e-12) {
        return H_inv;
    }

    const int n = static_cast<int>(s.size());
    double rho = 1.0 / sy;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd left = I - rho * s * y.transpose();
    Eigen::MatrixXd right = I - rho * y * s.transpose();
    return left * H_inv * right + rho * s * s.transpose();
}

// ---------------------------------------------------------------------------
// Hessian update families
// ---------------------------------------------------------------------------

Eigen::MatrixXd hessian_update_bfgs(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
) {
    double sy = y.dot(s);
    if (sy <= 1e-12) return H;
    Eigen::VectorXd Hs = H * s;
    double sHs = s.dot(Hs);
    if (sHs <= 1e-12) return H;
    return H + y * y.transpose() / sy - Hs * Hs.transpose() / sHs;
}

Eigen::MatrixXd hessian_update_sr1(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
) {
    Eigen::VectorXd r = y - H * s;
    double rs = r.dot(s);
    if (std::abs(rs) < 1e-12) return H;
    return H + r * r.transpose() / rs;
}

Eigen::MatrixXd hessian_update_powell(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
) {
    Eigen::VectorXd r = y - H * s;
    double ss = s.dot(s);
    if (ss <= 1e-12) return H;
    double rs = r.dot(s);
    return H + (r * s.transpose() + s * r.transpose()) / ss
           - rs * s * s.transpose() / (ss * ss);
}

Eigen::MatrixXd hessian_update_bofill(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
) {
    Eigen::VectorXd r = y - H * s;
    double rs = r.dot(s);
    double ss = s.dot(s);
    double rr = r.dot(r);
    if (ss <= 1e-12 || rr <= 1e-12) return H;

    double phi = (rs * rs) / (rr * ss);
    phi = std::max(0.0, std::min(1.0, phi));

    // PSB update
    Eigen::MatrixXd H_psb = hessian_update_powell(H, s, y);

    // SR1 update
    Eigen::MatrixXd H_sr1 = H;
    if (std::abs(rs) >= 1e-12) {
        H_sr1 = hessian_update_sr1(H, s, y);
    }

    return phi * H_psb + (1.0 - phi) * H_sr1;
}

Eigen::MatrixXd hessian_update_ms(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
) {
    double sy = y.dot(s);
    if (sy <= 1e-12) return H;
    Eigen::VectorXd Hs = H * s;
    double sHs = s.dot(Hs);
    if (sHs <= 1e-12) return H;

    Eigen::MatrixXd H_bfgs = hessian_update_bfgs(H, s, y);
    Eigen::VectorXd w = y / sy - Hs / sHs;
    double theta = sy * sHs;
    return H_bfgs + theta * w * w.transpose();
}

// ---------------------------------------------------------------------------
// Trust-region subproblem solver (More-Sorensen)
// ---------------------------------------------------------------------------

std::tuple<Eigen::VectorXd, double, bool> trust_region_step(
    const Eigen::VectorXd& g,
    const Eigen::MatrixXd& B,
    double delta,
    int max_iter,
    double tol
) {
    const int n = static_cast<int>(g.size());

    // Try unconstrained Newton step
    Eigen::LLT<Eigen::MatrixXd> llt(B);
    Eigen::VectorXd p_newton = -llt.solve(g);

    if (llt.info() == Eigen::Success) {
        double p_norm = p_newton.norm();
        if (p_norm <= delta * (1.0 + 1e-12)) {
            return {p_newton, 0.0, true};
        }
    }

    // Boundary solve: find λ > 0 such that ||p(λ)|| = Δ
    // p(λ) = -(B + λI)⁻¹g
    // φ(λ) = 1/||p|| - 1/Δ = 0

    // Lower bound for λ: max(0, -λ_min(B))
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
    double lambda_l = std::max(0.0, -es.eigenvalues()(0) + 1e-6);

    double lam = std::max(lambda_l + 0.1, 0.1);

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd B_shifted = B + lam * Eigen::MatrixXd::Identity(n, n);
        Eigen::LLT<Eigen::MatrixXd> llt_s(B_shifted);
        if (llt_s.info() != Eigen::Success) {
            lam = std::max(lam * 2.0, lambda_l + 1e-4);
            continue;
        }

        Eigen::VectorXd p = -llt_s.solve(g);
        double p_norm = p.norm();

        if (p_norm < 1e-15) {
            return {Eigen::VectorXd::Zero(n), lam, false};
        }

        double phi = 1.0 / p_norm - 1.0 / delta;
        if (std::abs(phi * delta) < tol) {
            return {p, lam, false};
        }

        // φ'(λ) = p^T (B + λI)⁻¹ p / ||p||³
        Eigen::VectorXd q = llt_s.solve(p);
        double phi_prime = p.dot(q) / (p_norm * p_norm * p_norm);

        if (std::abs(phi_prime) < 1e-15) {
            return {p, lam, false};
        }

        double lam_new = lam - phi / phi_prime;
        if (lam_new <= lambda_l) {
            lam = (lam + lambda_l) / 2.0;
        } else {
            lam = lam_new;
        }
    }

    // Fallback: Cauchy point
    double g_norm = g.norm();
    if (g_norm < 1e-15) {
        return {Eigen::VectorXd::Zero(n), lam, false};
    }
    double gBg = g.dot(B * g);
    double tau = (gBg <= 0.0) ? 1.0 : std::min(1.0, g_norm * g_norm * g_norm / (delta * gBg));
    return {-tau * delta * g / g_norm, lam, false};
}

// ---------------------------------------------------------------------------
// Convergence metrics
// ---------------------------------------------------------------------------

double max_abs_component(const Eigen::VectorXd& v) {
    if (v.size() == 0) return 0.0;
    return v.cwiseAbs().maxCoeff();
}

double rms(const Eigen::VectorXd& v) {
    if (v.size() == 0) return 0.0;
    return std::sqrt(v.squaredNorm() / static_cast<double>(v.size()));
}

double max_atom_displacement(
    const Eigen::VectorXd& x_old,
    const Eigen::VectorXd& x_new
) {
    if (x_old.size() != x_new.size() || x_old.size() == 0) return 0.0;
    Eigen::VectorXd diff = x_new - x_old;
    int n_atoms = static_cast<int>(diff.size()) / 3;
    double max_d = 0.0;
    for (int i = 0; i < n_atoms; ++i) {
        double d = diff.segment(3 * i, 3).norm();
        if (d > max_d) max_d = d;
    }
    return max_d;
}

double rms_displacement(
    const Eigen::VectorXd& x_old,
    const Eigen::VectorXd& x_new
) {
    if (x_old.size() != x_new.size() || x_old.size() == 0) return 0.0;
    return rms(x_new - x_old);
}

} // namespace geomopt
} // namespace vibeqc
