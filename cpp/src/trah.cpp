#include "vibeqc/trah.hpp"

#include <algorithm>
#include <cmath>

#include "vibeqc/jk_builder.hpp"
#include "vibeqc/quadratic_scf.hpp"   // expm_skew
#include "vibeqc/xc_kernel.hpp"

namespace vibeqc {

namespace {

// ===========================================================================
// Closed-shell RHF / RKS orbital-Hessian matvec + preconditioned CG.
// Conceptually identical to vibeqc/newton.hpp's closed-shell path; pulled
// inline here to avoid linker coupling against newton's internal helpers.
// All routines carry a level shift λ so the same matvec drives both the
// unconstrained Newton solve (λ = 0) and the boundary trust-region solves.
// ===========================================================================

Eigen::MatrixXd build_perturbed_density(const Eigen::MatrixXd& kappa_ov,
                                         const Eigen::MatrixXd& C_occ,
                                         const Eigen::MatrixXd& C_vir) {
    const Eigen::MatrixXd T = C_vir * kappa_ov;
    const Eigen::MatrixXd D_half = T * C_occ.transpose();
    return D_half + D_half.transpose();
}

Eigen::MatrixXd ov_block(const Eigen::MatrixXd& M_ao,
                          const Eigen::MatrixXd& C_occ,
                          const Eigen::MatrixXd& C_vir) {
    return C_vir.transpose() * M_ao * C_occ;
}

// (A + λI) κ — the level-shifted orbital-Hessian action. ``xc_kernel`` is
// nullptr for HF and a pinned-at-current-density unpolarised LDA/GGA builder
// for RKS; ``alpha_hf`` scales the HF exchange piece for hybrids.
Eigen::MatrixXd orbital_hessian_action(const Eigen::MatrixXd& kappa_ov,
                                        const Eigen::MatrixXd& C_occ,
                                        const Eigen::MatrixXd& C_vir,
                                        const Eigen::MatrixXd& eps_diff,
                                        const JKBuilder& jk,
                                        const XCKernelBuilder* xc_kernel,
                                        double alpha_hf,
                                        double shift) {
    Eigen::MatrixXd out = eps_diff.cwiseProduct(kappa_ov);
    const Eigen::MatrixXd D_v = build_perturbed_density(kappa_ov, C_occ, C_vir);
    Eigen::MatrixXd G_v = jk.build_g_rhf(D_v, alpha_hf);
    if (xc_kernel != nullptr) {
        G_v.noalias() += xc_kernel->apply(D_v);
    }
    out.noalias() += 2.0 * ov_block(G_v, C_occ, C_vir);
    if (shift != 0.0) {
        out.noalias() += shift * kappa_ov;
    }
    return out;
}

struct CGResult {
    Eigen::MatrixXd x;
    int n_iter = 0;
    bool converged = false;
    bool non_pd = false;   // true ⇒ a pᵀ(A+λI)p ≤ 0 was detected
                           // (A + λI is not positive-definite — the
                           // boundary root-find escalates λ on this).
};

// Solve (A + λI) x = b by preconditioned CG. Preconditioner is the
// shifted Jacobi diagonal (ε_a − ε_i + λ).
CGResult preconditioned_cg(const Eigen::MatrixXd& b,
                            const Eigen::MatrixXd& C_occ,
                            const Eigen::MatrixXd& C_vir,
                            const Eigen::MatrixXd& eps_diff,
                            const JKBuilder& jk,
                            int max_iter,
                            double tol,
                            const XCKernelBuilder* xc_kernel,
                            double alpha_hf,
                            double shift) {
    const double b_norm = b.norm();
    CGResult res;
    res.x = Eigen::MatrixXd::Zero(b.rows(), b.cols());
    if (b_norm == 0.0) {
        res.converged = true;
        return res;
    }
    const Eigen::MatrixXd precond =
        (eps_diff.array() + shift).matrix();   // shifted Jacobi preconditioner
    Eigen::MatrixXd r = b;
    Eigen::MatrixXd z = r.cwiseQuotient(precond);
    Eigen::MatrixXd p = z;
    double rz = (r.array() * z.array()).sum();
    for (int k = 1; k <= max_iter; ++k) {
        const Eigen::MatrixXd Ap =
            orbital_hessian_action(p, C_occ, C_vir, eps_diff, jk,
                                    xc_kernel, alpha_hf, shift);
        const double pAp = (p.array() * Ap.array()).sum();
        if (pAp <= 0.0) {
            // Negative curvature: A + λI is not positive-definite. The
            // caller (boundary root-find) escalates λ; the bare-Newton
            // caller treats this as "go to boundary".
            res.n_iter = k;
            res.converged = false;
            res.non_pd = true;
            return res;
        }
        const double alpha = rz / pAp;
        res.x.noalias() += alpha * p;
        r.noalias()    -= alpha * Ap;
        if (r.norm() < tol * b_norm) {
            res.n_iter = k;
            res.converged = true;
            return res;
        }
        z = r.cwiseQuotient(precond);
        const double rz_new = (r.array() * z.array()).sum();
        const double beta = rz_new / rz;
        p = z + beta * p;
        rz = rz_new;
    }
    res.n_iter = max_iter;
    res.converged = false;
    return res;
}

// Solution of the trust-region subproblem
//   minimise  m(κ) = gᵀκ + ½ κᵀ A κ   subject to  ‖κ‖ ≤ Δ
// for the closed-shell orbital-rotation manifold.
struct TRSResult {
    Eigen::MatrixXd kappa;
    double lambda = 0.0;       // level shift used
    int cg_iter = 0;           // total inner CG iterations
    bool cg_converged = false;
    bool on_boundary = false;  // true ⇒ ‖κ‖ = Δ (boundary solution)
};

TRSResult solve_trust_region_subproblem(
    const Eigen::MatrixXd& g,
    const Eigen::MatrixXd& C_occ,
    const Eigen::MatrixXd& C_vir,
    const Eigen::MatrixXd& eps_diff,
    const JKBuilder& jk,
    const TRAHOptions& opts,
    double trust_radius,
    const XCKernelBuilder* xc_kernel,
    double alpha_hf) {
    TRSResult res;
    const Eigen::MatrixXd b = -g;
    const double g_norm = g.norm();

    // --- interior attempt: unconstrained Newton step (λ = 0) ---
    CGResult cg0 = preconditioned_cg(b, C_occ, C_vir, eps_diff, jk,
                                      opts.cg_max_iter, opts.cg_tol,
                                      xc_kernel, alpha_hf, 0.0);
    res.cg_iter += cg0.n_iter;
    if (cg0.converged && cg0.x.norm() <= trust_radius) {
        res.kappa = cg0.x;
        res.lambda = 0.0;
        res.cg_converged = true;
        res.on_boundary = false;
        return res;
    }

    // --- boundary: More-Sorensen secular root-find on the level shift λ.
    // ‖κ(λ)‖ is monotonically decreasing in λ, so bracket + safeguarded
    // Newton converges in a handful of secular iterations. ---
    res.on_boundary = true;
    double lambda_lo = 0.0;     // ‖κ(λ)‖ ≥ Δ for λ ≤ lambda_lo
    double lambda_hi = -1.0;    // ‖κ(λ)‖ ≤ Δ for λ ≥ lambda_hi (unset = -1)
    double lambda;

    if (cg0.converged && !cg0.non_pd) {
        // Seed λ with one secular Newton step from λ = 0.
        const double p_norm = cg0.x.norm();   // > Δ in this branch
        CGResult cgq = preconditioned_cg(cg0.x, C_occ, C_vir, eps_diff, jk,
                                          opts.cg_max_iter, opts.cg_tol,
                                          xc_kernel, alpha_hf, 0.0);
        res.cg_iter += cgq.n_iter;
        const double pq = (cg0.x.array() * cgq.x.array()).sum();
        lambda = (pq > 0.0)
            ? (p_norm * p_norm / pq) * (p_norm / trust_radius - 1.0)
            : g_norm / std::max(trust_radius, 1e-30);
        lambda = std::max(lambda, 1e-8);
    } else {
        // Non-PD (or non-converged) at λ = 0 — seed from a positive guess
        // and let the escalation push λ above −λ_min(A).
        lambda = std::max(g_norm / std::max(trust_radius, 1e-30), 1e-3);
    }

    Eigen::MatrixXd best_kappa;
    double best_lambda = lambda;
    for (int it = 0; it < opts.ms_max_iter; ++it) {
        CGResult cgp = preconditioned_cg(b, C_occ, C_vir, eps_diff, jk,
                                          opts.cg_max_iter, opts.cg_tol,
                                          xc_kernel, alpha_hf, lambda);
        res.cg_iter += cgp.n_iter;
        if (cgp.non_pd) {
            // A + λI still indefinite — λ below −λ_min(A). Escalate.
            lambda_lo = std::max(lambda_lo, lambda);
            lambda = (lambda_hi > 0.0)
                ? 0.5 * (lambda_lo + lambda_hi)
                : std::max(lambda * 4.0, lambda_lo + 1.0);
            continue;
        }
        const double p_norm = cgp.x.norm();
        best_kappa = cgp.x;
        best_lambda = lambda;
        if (std::abs(p_norm - trust_radius) <= opts.ms_tol * trust_radius) {
            break;   // secular equation satisfied ‖κ(λ)‖ ≈ Δ
        }
        if (p_norm > trust_radius) {
            lambda_lo = std::max(lambda_lo, lambda);
        } else {
            lambda_hi = (lambda_hi < 0.0) ? lambda
                                          : std::min(lambda_hi, lambda);
        }
        // Derivative solve q = (A + λI)⁻¹ κ for the secular Newton step.
        CGResult cgq = preconditioned_cg(cgp.x, C_occ, C_vir, eps_diff, jk,
                                          opts.cg_max_iter, opts.cg_tol,
                                          xc_kernel, alpha_hf, lambda);
        res.cg_iter += cgq.n_iter;
        const double pq = (cgp.x.array() * cgq.x.array()).sum();
        double lambda_new = lambda;
        if (pq > 0.0) {
            // Newton step on φ(λ) = 1/Δ − 1/‖κ(λ)‖.
            lambda_new = lambda
                + (p_norm * p_norm / pq) * (p_norm / trust_radius - 1.0);
        }
        // Safeguard inside the bracket [lambda_lo, lambda_hi].
        if (lambda_hi > 0.0) {
            if (lambda_new <= lambda_lo || lambda_new >= lambda_hi) {
                lambda_new = 0.5 * (lambda_lo + lambda_hi);
            }
        } else {
            lambda_new = std::max(lambda_new, lambda_lo);
        }
        lambda = lambda_new;
    }

    if (best_kappa.size() == 0) {
        // Root-find never produced a PD solve — last-ditch safety: a
        // scaled steepest-descent step on the trust-region boundary.
        const double bn = b.norm();
        if (bn > 0.0) {
            res.kappa = b * (trust_radius / bn);
        } else {
            res.kappa = Eigen::MatrixXd::Zero(b.rows(), b.cols());
        }
        res.lambda = best_lambda;
        res.cg_converged = false;
    } else {
        res.kappa = best_kappa;
        res.lambda = best_lambda;
        res.cg_converged = true;
    }
    return res;
}

}  // namespace

TRAHStepResult trah_step(const Eigen::MatrixXd& F,
                          const Eigen::MatrixXd& C_prev,
                          const Eigen::VectorXd& eps_prev,
                          int n_occ,
                          const JKBuilder& jk,
                          const TRAHOptions& opts,
                          double trust_radius,
                          const XCKernelBuilder* xc_kernel,
                          double alpha_hf) {
    const Eigen::Index n_kept = C_prev.cols();
    const Eigen::Index n_vir  = n_kept - n_occ;

    TRAHStepResult out;
    out.trust_radius_used = trust_radius;
    if (n_vir <= 0) {
        out.C = C_prev;
        out.eps = (C_prev.transpose() * F * C_prev).diagonal();
        out.cg_converged = true;
        return out;
    }

    // F in current MO basis; symmetrise.
    Eigen::MatrixXd F_mo = C_prev.transpose() * F * C_prev;
    F_mo = 0.5 * (F_mo + F_mo.transpose());
    const Eigen::MatrixXd g = F_mo.bottomLeftCorner(n_vir, n_occ);  // gradient

    // ε_a − ε_i preconditioner (defensive floor as in Newton).
    Eigen::MatrixXd eps_diff(n_vir, n_occ);
    for (Eigen::Index a = 0; a < n_vir; ++a) {
        const double eps_a = eps_prev(n_occ + a);
        for (Eigen::Index i = 0; i < n_occ; ++i) {
            eps_diff(a, i) = std::max(eps_a - eps_prev(i), 1e-6);
        }
    }

    const Eigen::MatrixXd C_occ = C_prev.leftCols(n_occ);
    const Eigen::MatrixXd C_vir = C_prev.rightCols(n_vir);

    // Solve the trust-region subproblem exactly: interior unconstrained
    // Newton step when ‖A⁻¹g‖ ≤ Δ, else the level-shifted boundary step
    // κ(λ) = −(A + λI)⁻¹g found by the More-Sorensen secular root-find.
    const TRSResult trs = solve_trust_region_subproblem(
        g, C_occ, C_vir, eps_diff, jk, opts, trust_radius,
        xc_kernel, alpha_hf);

    const Eigen::MatrixXd& kappa = trs.kappa;
    out.kappa_norm = kappa.norm();
    out.cg_iter = trs.cg_iter;
    out.cg_converged = trs.cg_converged;
    out.level_shift = trs.lambda;
    out.on_boundary = trs.on_boundary;

    // Model-predicted energy decrease for the quadratic model
    //   m(κ) = E_0 + gᵀκ + ½ κᵀ A κ.
    // With the level-shifted step (A + λI)κ = −g we have
    //   κᵀ A κ = −gᵀκ − λ‖κ‖²,
    // hence  ΔE_model = E_0 − m(κ) = −½ gᵀκ + ½ λ‖κ‖².
    // (λ = 0 recovers the plain Newton −½ gᵀκ.) The driver uses this as
    // the denominator in Powell's ρ = actual / predicted next iteration.
    const double g_dot_kappa = (g.array() * kappa.array()).sum();
    out.predicted_decrease =
        -0.5 * g_dot_kappa + 0.5 * trs.lambda * out.kappa_norm * out.kappa_norm;

    // Build full antihermitian generator and rotate.
    Eigen::MatrixXd kappa_full = Eigen::MatrixXd::Zero(n_kept, n_kept);
    kappa_full.bottomLeftCorner(n_vir, n_occ) = kappa;
    kappa_full.topRightCorner(n_occ, n_vir) = -kappa.transpose();
    const Eigen::MatrixXd U = expm_skew(kappa_full);
    out.C = C_prev * U;
    out.eps = (out.C.transpose() * F * out.C).diagonal();
    return out;
}

// ===========================================================================
// Open-shell UHF / UKS TRAH — coupled per-spin CG with the same
// level-shifted trust-region subproblem solve as the closed-shell path.
// Internals mirror cpp/src/newton.cpp's uhf_newton_step but expose the
// predicted_decrease for Powell's ρ test and solve the trust-region
// subproblem exactly (interior Newton or More-Sorensen boundary).
// ===========================================================================

namespace {

struct UHFKappa {
    Eigen::MatrixXd alpha;
    Eigen::MatrixXd beta;
};

double uhf_kappa_dot(const UHFKappa& a, const UHFKappa& b) {
    double s = 0.0;
    if (a.alpha.size() != 0) s += (a.alpha.array() * b.alpha.array()).sum();
    if (a.beta.size()  != 0) s += (a.beta.array()  * b.beta.array()).sum();
    return s;
}
double uhf_kappa_norm(const UHFKappa& a) {
    return std::sqrt(uhf_kappa_dot(a, a));
}

// (A + λI) κ on the coupled (κ_α, κ_β) manifold.
UHFKappa uhf_orbital_hessian_action(
    const UHFKappa& kappa,
    const Eigen::MatrixXd& C_occ_alpha, const Eigen::MatrixXd& C_vir_alpha,
    const Eigen::MatrixXd& C_occ_beta,  const Eigen::MatrixXd& C_vir_beta,
    const Eigen::MatrixXd& eps_diff_alpha,
    const Eigen::MatrixXd& eps_diff_beta,
    const JKBuilder& jk,
    const UHFXCKernelBuilder* xc_kernel,
    double alpha_hf,
    double shift)
{
    UHFKappa out;
    out.alpha = eps_diff_alpha.cwiseProduct(kappa.alpha);
    out.beta  = eps_diff_beta .cwiseProduct(kappa.beta);
    const bool have_a = kappa.alpha.size() != 0;
    const bool have_b = kappa.beta.size() != 0;
    Eigen::MatrixXd Da_pert, Db_pert;
    if (have_a) {
        Da_pert = build_perturbed_density(kappa.alpha, C_occ_alpha, C_vir_alpha);
    }
    if (have_b) {
        Db_pert = build_perturbed_density(kappa.beta,  C_occ_beta,  C_vir_beta);
    }
    const Eigen::Index n_bf = C_occ_alpha.size() ? C_occ_alpha.rows()
                                                  : C_occ_beta.rows();
    Eigen::MatrixXd D_tot = Eigen::MatrixXd::Zero(n_bf, n_bf);
    if (have_a) D_tot += Da_pert;
    if (have_b) D_tot += Db_pert;
    const Eigen::MatrixXd J_pert = jk.build_J(D_tot);

    UHFXCKernelBuilder::Output xc_out;
    if (xc_kernel != nullptr) {
        const Eigen::MatrixXd D_a_in = have_a ? Da_pert
            : Eigen::MatrixXd::Zero(n_bf, n_bf);
        const Eigen::MatrixXd D_b_in = have_b ? Db_pert
            : Eigen::MatrixXd::Zero(n_bf, n_bf);
        xc_out = xc_kernel->apply(D_a_in, D_b_in);
    }

    if (have_a) {
        Eigen::MatrixXd F_a = J_pert;
        if (alpha_hf != 0.0) F_a.noalias() -= alpha_hf * jk.build_K(Da_pert);
        if (xc_kernel != nullptr) F_a.noalias() += xc_out.alpha;
        out.alpha.noalias() += ov_block(F_a, C_occ_alpha, C_vir_alpha);
        if (shift != 0.0) out.alpha.noalias() += shift * kappa.alpha;
    }
    if (have_b) {
        Eigen::MatrixXd F_b = J_pert;
        if (alpha_hf != 0.0) F_b.noalias() -= alpha_hf * jk.build_K(Db_pert);
        if (xc_kernel != nullptr) F_b.noalias() += xc_out.beta;
        out.beta.noalias() += ov_block(F_b, C_occ_beta, C_vir_beta);
        if (shift != 0.0) out.beta.noalias() += shift * kappa.beta;
    }
    return out;
}

struct UHFCGResult {
    UHFKappa x;
    int n_iter = 0;
    bool converged = false;
    bool non_pd = false;
};

// Solve (A + λI) x = b on the coupled (α, β) manifold by preconditioned CG.
UHFCGResult uhf_preconditioned_cg(
    const UHFKappa& b,
    const Eigen::MatrixXd& C_occ_alpha, const Eigen::MatrixXd& C_vir_alpha,
    const Eigen::MatrixXd& C_occ_beta,  const Eigen::MatrixXd& C_vir_beta,
    const Eigen::MatrixXd& eps_diff_alpha,
    const Eigen::MatrixXd& eps_diff_beta,
    const JKBuilder& jk,
    int max_iter, double tol,
    const UHFXCKernelBuilder* xc_kernel,
    double alpha_hf,
    double shift)
{
    UHFCGResult res;
    res.x.alpha = Eigen::MatrixXd::Zero(b.alpha.rows(), b.alpha.cols());
    res.x.beta  = Eigen::MatrixXd::Zero(b.beta.rows(),  b.beta.cols());
    const double b_norm = uhf_kappa_norm(b);
    if (b_norm == 0.0) { res.converged = true; return res; }

    const Eigen::MatrixXd precond_a =
        (eps_diff_alpha.array() + shift).matrix();
    const Eigen::MatrixXd precond_b =
        (eps_diff_beta.array() + shift).matrix();
    auto precond = [&](const UHFKappa& v) {
        UHFKappa out;
        if (v.alpha.size() != 0) out.alpha = v.alpha.cwiseQuotient(precond_a);
        if (v.beta.size()  != 0) out.beta  = v.beta .cwiseQuotient(precond_b);
        return out;
    };

    UHFKappa r = {b.alpha, b.beta};
    UHFKappa z = precond(r);
    UHFKappa p = z;
    double rz = uhf_kappa_dot(r, z);
    for (int k = 1; k <= max_iter; ++k) {
        const UHFKappa Ap = uhf_orbital_hessian_action(
            p, C_occ_alpha, C_vir_alpha,
            C_occ_beta,  C_vir_beta,
            eps_diff_alpha, eps_diff_beta, jk,
            xc_kernel, alpha_hf, shift);
        const double pAp = uhf_kappa_dot(p, Ap);
        if (pAp <= 0.0) {
            res.n_iter = k; res.converged = false; res.non_pd = true;
            return res;
        }
        const double alpha_step = rz / pAp;
        if (res.x.alpha.size() != 0) res.x.alpha.noalias() += alpha_step * p.alpha;
        if (res.x.beta.size()  != 0) res.x.beta .noalias() += alpha_step * p.beta;
        if (r.alpha.size() != 0)     r.alpha   .noalias() -= alpha_step * Ap.alpha;
        if (r.beta.size()  != 0)     r.beta    .noalias() -= alpha_step * Ap.beta;
        if (uhf_kappa_norm(r) < tol * b_norm) {
            res.n_iter = k; res.converged = true; return res;
        }
        z = precond(r);
        const double rz_new = uhf_kappa_dot(r, z);
        const double beta_step = rz_new / rz;
        if (p.alpha.size() != 0) p.alpha = z.alpha + beta_step * p.alpha;
        if (p.beta.size()  != 0) p.beta  = z.beta  + beta_step * p.beta;
        rz = rz_new;
    }
    res.n_iter = max_iter; res.converged = false;
    return res;
}

struct UHFTRSResult {
    UHFKappa kappa;
    double lambda = 0.0;
    int cg_iter = 0;
    bool cg_converged = false;
    bool on_boundary = false;
};

UHFKappa uhf_scale(const UHFKappa& v, double s) {
    UHFKappa out;
    if (v.alpha.size() != 0) out.alpha = s * v.alpha;
    if (v.beta.size()  != 0) out.beta  = s * v.beta;
    return out;
}
UHFKappa uhf_negate(const UHFKappa& v) { return uhf_scale(v, -1.0); }

UHFTRSResult uhf_solve_trust_region_subproblem(
    const UHFKappa& g,
    const Eigen::MatrixXd& C_occ_alpha, const Eigen::MatrixXd& C_vir_alpha,
    const Eigen::MatrixXd& C_occ_beta,  const Eigen::MatrixXd& C_vir_beta,
    const Eigen::MatrixXd& eps_diff_alpha,
    const Eigen::MatrixXd& eps_diff_beta,
    const JKBuilder& jk,
    const TRAHOptions& opts,
    double trust_radius,
    const UHFXCKernelBuilder* xc_kernel,
    double alpha_hf) {
    UHFTRSResult res;
    const UHFKappa b = uhf_negate(g);
    const double g_norm = uhf_kappa_norm(g);

    auto run_cg = [&](const UHFKappa& rhs, double shift) {
        return uhf_preconditioned_cg(
            rhs, C_occ_alpha, C_vir_alpha, C_occ_beta, C_vir_beta,
            eps_diff_alpha, eps_diff_beta, jk,
            opts.cg_max_iter, opts.cg_tol, xc_kernel, alpha_hf, shift);
    };

    // --- interior attempt: unconstrained Newton (λ = 0) ---
    UHFCGResult cg0 = run_cg(b, 0.0);
    res.cg_iter += cg0.n_iter;
    if (cg0.converged && uhf_kappa_norm(cg0.x) <= trust_radius) {
        res.kappa = cg0.x;
        res.lambda = 0.0;
        res.cg_converged = true;
        res.on_boundary = false;
        return res;
    }

    // --- boundary: More-Sorensen secular root-find on λ ---
    res.on_boundary = true;
    double lambda_lo = 0.0;
    double lambda_hi = -1.0;
    double lambda;

    if (cg0.converged && !cg0.non_pd) {
        const double p_norm = uhf_kappa_norm(cg0.x);
        UHFCGResult cgq = run_cg(cg0.x, 0.0);
        res.cg_iter += cgq.n_iter;
        const double pq = uhf_kappa_dot(cg0.x, cgq.x);
        lambda = (pq > 0.0)
            ? (p_norm * p_norm / pq) * (p_norm / trust_radius - 1.0)
            : g_norm / std::max(trust_radius, 1e-30);
        lambda = std::max(lambda, 1e-8);
    } else {
        lambda = std::max(g_norm / std::max(trust_radius, 1e-30), 1e-3);
    }

    UHFKappa best_kappa;
    bool have_best = false;
    double best_lambda = lambda;
    for (int it = 0; it < opts.ms_max_iter; ++it) {
        UHFCGResult cgp = run_cg(b, lambda);
        res.cg_iter += cgp.n_iter;
        if (cgp.non_pd) {
            lambda_lo = std::max(lambda_lo, lambda);
            lambda = (lambda_hi > 0.0)
                ? 0.5 * (lambda_lo + lambda_hi)
                : std::max(lambda * 4.0, lambda_lo + 1.0);
            continue;
        }
        const double p_norm = uhf_kappa_norm(cgp.x);
        best_kappa = cgp.x;
        have_best = true;
        best_lambda = lambda;
        if (std::abs(p_norm - trust_radius) <= opts.ms_tol * trust_radius) {
            break;
        }
        if (p_norm > trust_radius) {
            lambda_lo = std::max(lambda_lo, lambda);
        } else {
            lambda_hi = (lambda_hi < 0.0) ? lambda
                                          : std::min(lambda_hi, lambda);
        }
        UHFCGResult cgq = run_cg(cgp.x, lambda);
        res.cg_iter += cgq.n_iter;
        const double pq = uhf_kappa_dot(cgp.x, cgq.x);
        double lambda_new = lambda;
        if (pq > 0.0) {
            lambda_new = lambda
                + (p_norm * p_norm / pq) * (p_norm / trust_radius - 1.0);
        }
        if (lambda_hi > 0.0) {
            if (lambda_new <= lambda_lo || lambda_new >= lambda_hi) {
                lambda_new = 0.5 * (lambda_lo + lambda_hi);
            }
        } else {
            lambda_new = std::max(lambda_new, lambda_lo);
        }
        lambda = lambda_new;
    }

    if (!have_best) {
        const double bn = uhf_kappa_norm(b);
        res.kappa = (bn > 0.0) ? uhf_scale(b, trust_radius / bn) : b;
        res.lambda = best_lambda;
        res.cg_converged = false;
    } else {
        res.kappa = best_kappa;
        res.lambda = best_lambda;
        res.cg_converged = true;
    }
    return res;
}

Eigen::MatrixXd uhf_eps_diff(const Eigen::VectorXd& eps,
                              Eigen::Index n_occ, Eigen::Index n_vir) {
    if (n_vir == 0 || n_occ == 0) {
        return Eigen::MatrixXd::Zero(n_vir, n_occ);
    }
    Eigen::MatrixXd out(n_vir, n_occ);
    for (Eigen::Index a = 0; a < n_vir; ++a) {
        const double eps_a = eps(n_occ + a);
        for (Eigen::Index i = 0; i < n_occ; ++i) {
            out(a, i) = std::max(eps_a - eps(i), 1e-6);
        }
    }
    return out;
}

Eigen::MatrixXd kappa_to_skew(const Eigen::MatrixXd& kappa_ov,
                               Eigen::Index n_kept,
                               Eigen::Index n_occ) {
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_kept, n_kept);
    const Eigen::Index n_vir = n_kept - n_occ;
    if (n_vir > 0 && kappa_ov.size() != 0) {
        K.bottomLeftCorner(n_vir, n_occ) = kappa_ov;
        K.topRightCorner(n_occ, n_vir) = -kappa_ov.transpose();
    }
    return K;
}

}  // namespace

UHFTRAHStepResult uhf_trah_step(const Eigen::MatrixXd& F_alpha,
                                  const Eigen::MatrixXd& F_beta,
                                  const Eigen::MatrixXd& C_alpha_prev,
                                  const Eigen::MatrixXd& C_beta_prev,
                                  const Eigen::VectorXd& eps_alpha_prev,
                                  const Eigen::VectorXd& eps_beta_prev,
                                  int n_alpha, int n_beta,
                                  const JKBuilder& jk,
                                  const TRAHOptions& opts,
                                  double trust_radius,
                                  const UHFXCKernelBuilder* xc_kernel,
                                  double alpha_hf) {
    const Eigen::Index n_kept_a = C_alpha_prev.cols();
    const Eigen::Index n_kept_b = C_beta_prev.cols();
    const Eigen::Index n_vir_a = n_kept_a - n_alpha;
    const Eigen::Index n_vir_b = n_kept_b - n_beta;

    UHFTRAHStepResult out;
    out.trust_radius_used = trust_radius;

    Eigen::MatrixXd Fmo_a = C_alpha_prev.transpose() * F_alpha * C_alpha_prev;
    Eigen::MatrixXd Fmo_b = C_beta_prev .transpose() * F_beta  * C_beta_prev;
    Fmo_a = 0.5 * (Fmo_a + Fmo_a.transpose());
    Fmo_b = 0.5 * (Fmo_b + Fmo_b.transpose());

    // Gradient g_σ = occ-vir block of F^MO_σ.
    UHFKappa g;
    if (n_vir_a > 0 && n_alpha > 0) {
        g.alpha = Fmo_a.bottomLeftCorner(n_vir_a, n_alpha);
    }
    if (n_vir_b > 0 && n_beta > 0) {
        g.beta = Fmo_b.bottomLeftCorner(n_vir_b, n_beta);
    }

    const Eigen::MatrixXd eps_diff_a = uhf_eps_diff(eps_alpha_prev, n_alpha, n_vir_a);
    const Eigen::MatrixXd eps_diff_b = uhf_eps_diff(eps_beta_prev,  n_beta,  n_vir_b);
    const Eigen::MatrixXd C_occ_a =
        (n_alpha > 0) ? C_alpha_prev.leftCols(n_alpha) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_vir_a =
        (n_vir_a > 0) ? C_alpha_prev.rightCols(n_vir_a) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_occ_b =
        (n_beta > 0) ? C_beta_prev.leftCols(n_beta) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_vir_b =
        (n_vir_b > 0) ? C_beta_prev.rightCols(n_vir_b) : Eigen::MatrixXd();

    const UHFTRSResult trs = uhf_solve_trust_region_subproblem(
        g, C_occ_a, C_vir_a, C_occ_b, C_vir_b,
        eps_diff_a, eps_diff_b, jk, opts, trust_radius,
        xc_kernel, alpha_hf);

    const UHFKappa& kappa = trs.kappa;
    out.kappa_norm = uhf_kappa_norm(kappa);
    out.cg_iter = trs.cg_iter;
    out.cg_converged = trs.cg_converged;
    out.level_shift = trs.lambda;
    out.on_boundary = trs.on_boundary;

    // Predicted ΔE = −½ gᵀκ + ½ λ‖κ‖²  (see closed-shell derivation).
    const double g_dot_kappa = uhf_kappa_dot(g, kappa);
    out.predicted_decrease =
        -0.5 * g_dot_kappa + 0.5 * trs.lambda * out.kappa_norm * out.kappa_norm;

    const Eigen::MatrixXd Ka_full = kappa_to_skew(kappa.alpha, n_kept_a, n_alpha);
    const Eigen::MatrixXd Kb_full = kappa_to_skew(kappa.beta,  n_kept_b, n_beta);
    const Eigen::MatrixXd Ua = expm_skew(Ka_full);
    const Eigen::MatrixXd Ub = expm_skew(Kb_full);
    out.C_alpha = C_alpha_prev * Ua;
    out.C_beta  = C_beta_prev  * Ub;
    out.eps_alpha = (out.C_alpha.transpose() * F_alpha * out.C_alpha).diagonal();
    out.eps_beta  = (out.C_beta .transpose() * F_beta  * out.C_beta ).diagonal();
    return out;
}

double update_trust_radius(double trust_radius,
                            double actual_decrease,
                            double predicted_decrease,
                            double kappa_norm,
                            const TRAHOptions& opts) {
    // No-op if the previous TRAH step's prediction was zero (or near it)
    // — happens on the first iter or when SCF is at the fixed point.
    if (std::abs(predicted_decrease) < 1e-12) {
        return std::clamp(trust_radius,
                          opts.min_trust_radius, opts.max_trust_radius);
    }
    const double rho = actual_decrease / predicted_decrease;
    if (rho < opts.rho_shrink) {
        trust_radius *= opts.trust_shrink_factor;
    } else if (rho > opts.rho_expand
               && kappa_norm > 0.9 * trust_radius) {
        // Only expand if the previous step actually used the budget.
        trust_radius *= opts.trust_expand_factor;
    }
    return std::clamp(trust_radius,
                      opts.min_trust_radius, opts.max_trust_radius);
}

}  // namespace vibeqc
