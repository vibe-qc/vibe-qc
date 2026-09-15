#include "vibeqc/newton.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

#include "vibeqc/jk_builder.hpp"
#include "vibeqc/quadratic_scf.hpp"
#include "vibeqc/xc_kernel.hpp"

namespace vibeqc {

namespace {

// Build the symmetric AO-basis "perturbed density"
//
//   D^κ_{μν} = Σ_{ai} κ_{ai} (C_μa C_νi + C_μi C_νa)
//
// from a κ vector in occ-vir MO basis (shape ``(n_vir, n_occ)``,
// ``[a, i]`` indexing matching ``quadratic_step``).
//
// No leading factor of 2 — the closed-shell orbital-Hessian
// coefficients ``4(ai|bj) − (ab|ji) − (aj|bi)`` already absorb the
// doubly-occupied spin algebra. The "2 ·" factor on G[D^κ] in the
// matvec gives the correct combination "2·(J − ½K)·D^κ = 4J − K"
// (in MO occ-vir basis).
Eigen::MatrixXd build_perturbed_density(const Eigen::MatrixXd& kappa_ov,
                                         const Eigen::MatrixXd& C_occ,
                                         const Eigen::MatrixXd& C_vir) {
    // T_{μi} = Σ_a κ_{ai} C_μa = (C_vir · κ_ov)_{μi}
    const Eigen::MatrixXd T = C_vir * kappa_ov;
    // D = T C_occ^T + C_occ T^T  (Hermitian).
    const Eigen::MatrixXd D_half = T * C_occ.transpose();
    return D_half + D_half.transpose();
}

// Project an AO matrix into the occ-vir block (shape (n_vir, n_occ),
// [a, i] indexing).
Eigen::MatrixXd ov_block(const Eigen::MatrixXd& M_ao,
                         const Eigen::MatrixXd& C_occ,
                         const Eigen::MatrixXd& C_vir) {
    return C_vir.transpose() * M_ao * C_occ;
}

// Closed-shell RHF / RKS orbital-Hessian times trial vector κ (shape
// (n_vir, n_occ)):
//
//   (A·κ)_{ai} = (ε_a − ε_i) κ_{ai}  +  2 · G_KS[D^κ]^{ov}_{ai}
//
// where G_KS[D] = J[D] − ½ α_HF K[D] + W^XC[D].
//
// HF case (xc_kernel == nullptr, alpha_hf default 1.0): G_KS reduces to
// J − ½K, the matvec recovers the textbook 4(ai|bj) − (ab|ji) − (aj|bi)
// closed-shell coefficient on the two-electron piece.
//
// RKS case (xc_kernel != nullptr): adds the XC kernel contribution
// W^XC[D^κ] = ∂V^XC/∂D · δD (validated against finite difference in
// `tests/test_xc_kernel.py`). alpha_hf scales the HF exchange for
// hybrids (B3LYP: 0.20, PBE0: 0.25, pure DFT: 0.0).
Eigen::MatrixXd orbital_hessian_action(const Eigen::MatrixXd& kappa_ov,
                                        const Eigen::MatrixXd& C_occ,
                                        const Eigen::MatrixXd& C_vir,
                                        const Eigen::MatrixXd& eps_diff,
                                        const JKBuilder& jk,
                                        const XCKernelBuilder* xc_kernel,
                                        double alpha_hf) {
    // Diagonal piece.
    Eigen::MatrixXd out = eps_diff.cwiseProduct(kappa_ov);
    // Two-electron piece.
    const Eigen::MatrixXd D_v = build_perturbed_density(kappa_ov, C_occ, C_vir);
    Eigen::MatrixXd G_v = jk.build_g_rhf(D_v, alpha_hf);
    if (xc_kernel != nullptr) {
        G_v.noalias() += xc_kernel->apply(D_v);
    }
    out.noalias() += 2.0 * ov_block(G_v, C_occ, C_vir);
    return out;
}

// Preconditioned conjugate gradient for ``A·κ = b`` with the
// orbital-Hessian as A and ``1/(ε_a − ε_i)`` as the preconditioner.
//
// Returns ``(κ, n_iter, converged)``. Bails out (returns the current
// iterate, converged=false) on a non-PD curvature signal — the caller
// drops back to a damped step in that case.
struct CGResult {
    Eigen::MatrixXd x;
    int n_iter = 0;
    bool converged = false;
};

CGResult preconditioned_cg(const Eigen::MatrixXd& b,
                            const Eigen::MatrixXd& C_occ,
                            const Eigen::MatrixXd& C_vir,
                            const Eigen::MatrixXd& eps_diff,
                            const JKBuilder& jk,
                            int max_iter,
                            double tol,
                            const XCKernelBuilder* xc_kernel,
                            double alpha_hf) {
    const double b_norm = b.norm();
    CGResult res;
    res.x = Eigen::MatrixXd::Zero(b.rows(), b.cols());
    if (b_norm == 0.0) {
        res.converged = true;
        return res;
    }

    Eigen::MatrixXd r = b;                                    // r = b - A·0 = b
    Eigen::MatrixXd z = r.cwiseQuotient(eps_diff);            // M^{-1} r
    Eigen::MatrixXd p = z;
    double rz = (r.array() * z.array()).sum();

    for (int k = 1; k <= max_iter; ++k) {
        const Eigen::MatrixXd Ap =
            orbital_hessian_action(p, C_occ, C_vir, eps_diff, jk,
                                    xc_kernel, alpha_hf);
        const double pAp = (p.array() * Ap.array()).sum();
        if (pAp <= 0.0) {
            // Non-PD curvature — bail before the step blows up.
            res.n_iter = k;
            res.converged = false;
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
        z = r.cwiseQuotient(eps_diff);
        const double rz_new = (r.array() * z.array()).sum();
        const double beta = rz_new / rz;
        p = z + beta * p;
        rz = rz_new;
    }
    res.n_iter = max_iter;
    res.converged = false;
    return res;
}

}  // namespace

NewtonStepResult newton_step(const Eigen::MatrixXd& F,
                           const Eigen::MatrixXd& C_prev,
                           const Eigen::VectorXd& eps_prev,
                           int n_occ,
                           const JKBuilder& jk,
                           const NewtonOptions& opts,
                           const XCKernelBuilder* xc_kernel,
                           double alpha_hf) {
    const Eigen::Index n_kept = C_prev.cols();
    const Eigen::Index n_vir = n_kept - n_occ;

    NewtonStepResult out;
    if (n_vir <= 0) {
        // No virtual space; rotation is the identity.
        out.C = C_prev;
        out.eps = (C_prev.transpose() * F * C_prev).diagonal();
        out.cg_converged = true;
        return out;
    }

    // F in current MO basis; symmetrise to absorb residual numerical drift.
    Eigen::MatrixXd F_mo = C_prev.transpose() * F * C_prev;
    F_mo = 0.5 * (F_mo + F_mo.transpose());

    // Occ-vir block of F^MO (shape (n_vir, n_occ), [a, i] indexing).
    Eigen::MatrixXd F_ov = F_mo.bottomLeftCorner(n_vir, n_occ);

    // Gradient on the orbital-rotation manifold: g_{ai} = 4 F^MO_{ai}
    // (closed-shell convention; the 4 = 2 (spin) × 2 (anti-Hermitian
    // double-counting) prefactor cancels in A·κ = -g since the same 4
    // appears on the orbital Hessian's two-electron coefficient. We can
    // equivalently solve A·κ = -F_ov (with the orbital Hessian normalised
    // accordingly) — that's what we do, since the CG residual scales the
    // same way.)
    //
    // The orbital_hessian_action already produces (A·κ)/4 in our
    // normalisation: the factor 2 on G[D^κ] paired with G = J − ½K
    // recovers 4(ai|bj) − (ab|ji) − (aj|bi)/4, and the diagonal is
    // (ε_a − ε_i) κ. Solving A·κ = -F_ov here is the rescaled Newton
    // step, identical solution.
    const Eigen::MatrixXd b = -F_ov;

    // ε_a − ε_i, shape (n_vir, n_occ), strictly positive for non-
    // pathological systems. Used both as the diagonal of the Hessian
    // and as the CG preconditioner (Jacobi).
    Eigen::MatrixXd eps_diff(n_vir, n_occ);
    for (Eigen::Index a = 0; a < n_vir; ++a) {
        const double eps_a = eps_prev(n_occ + a);
        for (Eigen::Index i = 0; i < n_occ; ++i) {
            const double diff = eps_a - eps_prev(i);
            // Defensive floor; for sane RHF references diff > 0 strictly.
            eps_diff(a, i) = std::max(diff, 1e-6);
        }
    }

    // Need the MO-basis split for the matvec.
    const Eigen::MatrixXd C_occ = C_prev.leftCols(n_occ);
    const Eigen::MatrixXd C_vir = C_prev.rightCols(n_vir);

    const CGResult cg = preconditioned_cg(b, C_occ, C_vir, eps_diff, jk,
                                            opts.cg_max_iter, opts.cg_tol,
                                            xc_kernel, alpha_hf);

    Eigen::MatrixXd kappa_ov = cg.x;
    out.kappa_norm = kappa_ov.norm();

    // Trust-region cap on ‖κ‖_F. Same simple cap as quadratic_step —
    // the augmented-Hessian / λ-shift TRAH variant is a follow-up.
    if (out.kappa_norm > opts.trust_radius) {
        kappa_ov *= (opts.trust_radius / out.kappa_norm);
    }

    // Build the full skew-symmetric κ on the kept-direction subspace.
    Eigen::MatrixXd kappa_full = Eigen::MatrixXd::Zero(n_kept, n_kept);
    kappa_full.bottomLeftCorner(n_vir, n_occ) = kappa_ov;
    kappa_full.topRightCorner(n_occ, n_vir) = -kappa_ov.transpose();

    // C_new = C_prev · exp(κ); eps_new = diag(C_new^T F C_new).
    const Eigen::MatrixXd U = expm_skew(kappa_full);
    out.C = C_prev * U;
    out.eps = (out.C.transpose() * F * out.C).diagonal();
    out.cg_iter = cg.n_iter;
    out.cg_converged = cg.converged;
    return out;
}

namespace {

// UHF orbital-Hessian times (κ_α, κ_β):
//
//   (Aκ)_{ai}^σ = (ε_a^σ − ε_i^σ) κ_{ai}^σ  +  F_σ^pert^{MOσ}_{ai}
//
// with
//   D^κ_σ      = build_perturbed_density(κ_σ, C_occ^σ, C_vir^σ)
//   F_α^pert   = J(D^κ_α + D^κ_β) − K(D^κ_α)
//   F_β^pert   = J(D^κ_α + D^κ_β) − K(D^κ_β)
//
// The α↔β cross-spin coupling lives entirely in the J build on the
// total perturbed density; K is spin-diagonal. No factor of 2 on
// F^pert (the UHF normalisation is "per-spin" already, unlike the
// closed-shell "2 · G[D^κ]").
struct UHFKappa {
    Eigen::MatrixXd alpha;
    Eigen::MatrixXd beta;
};

UHFKappa uhf_orbital_hessian_action(
    const UHFKappa& kappa,
    const Eigen::MatrixXd& C_occ_alpha, const Eigen::MatrixXd& C_vir_alpha,
    const Eigen::MatrixXd& C_occ_beta,  const Eigen::MatrixXd& C_vir_beta,
    const Eigen::MatrixXd& eps_diff_alpha,
    const Eigen::MatrixXd& eps_diff_beta,
    const JKBuilder& jk,
    const UHFXCKernelBuilder* xc_kernel,
    double alpha_hf)
{
    UHFKappa out;
    // Diagonal pieces (independent per spin).
    out.alpha = eps_diff_alpha.cwiseProduct(kappa.alpha);
    out.beta  = eps_diff_beta .cwiseProduct(kappa.beta);

    // Perturbed densities per spin (UHF convention — no factor of 2).
    // Each is identically zero when the corresponding κ is empty
    // (n_vir == 0); skip the build_perturbed_density / Fock calls in
    // that case to avoid wasted work on highly-spin-polarised systems.
    const bool have_alpha = kappa.alpha.size() != 0;
    const bool have_beta  = kappa.beta.size()  != 0;

    Eigen::MatrixXd D_alpha_pert;
    Eigen::MatrixXd D_beta_pert;
    if (have_alpha) {
        D_alpha_pert = build_perturbed_density(kappa.alpha,
                                                C_occ_alpha, C_vir_alpha);
    }
    if (have_beta) {
        D_beta_pert  = build_perturbed_density(kappa.beta,
                                                C_occ_beta,  C_vir_beta);
    }

    // Common J on the total perturbed density; per-spin K (scaled by
    // alpha_hf for hybrid functionals; 1.0 for HF). UKS XC kernel adds
    // W^XC_σ to each spin's perturbed Fock.
    // AO dimension from whichever spin actually carries a rotation
    // space: C_occ_alpha is an EMPTY matrix when the alpha channel has
    // no occupied x virtual block (e.g. N atom m=4/STO-3G, where every
    // alpha orbital is occupied), and sizing the total perturbed
    // density from it added a full-size D_beta_pert to a 0x0 matrix —
    // a latent crash the default-on stability check exposed.
    const Eigen::Index n_bf =
        (C_occ_alpha.size() != 0) ? C_occ_alpha.rows() : C_occ_beta.rows();
    Eigen::MatrixXd D_tot_pert = Eigen::MatrixXd::Zero(n_bf, n_bf);
    if (have_alpha) D_tot_pert += D_alpha_pert;
    if (have_beta)  D_tot_pert += D_beta_pert;
    const Eigen::MatrixXd J_pert = jk.build_J(D_tot_pert);

    // XC contribution per spin. Need *both* perturbed densities to
    // compute either spin's W^XC (αβ cross-coupling). Build zero-
    // matrix stand-ins where one spin's κ is empty.
    UHFXCKernelBuilder::Output xc_out;
    if (xc_kernel != nullptr) {
        const Eigen::MatrixXd D_a_in = have_alpha
            ? D_alpha_pert
            : Eigen::MatrixXd::Zero(n_bf, n_bf);
        const Eigen::MatrixXd D_b_in = have_beta
            ? D_beta_pert
            : Eigen::MatrixXd::Zero(n_bf, n_bf);
        xc_out = xc_kernel->apply(D_a_in, D_b_in);
    }

    if (have_alpha) {
        Eigen::MatrixXd F_alpha_pert = J_pert;
        if (alpha_hf != 0.0) {
            F_alpha_pert.noalias() -= alpha_hf * jk.build_K(D_alpha_pert);
        }
        if (xc_kernel != nullptr) {
            F_alpha_pert.noalias() += xc_out.alpha;
        }
        out.alpha.noalias() +=
            ov_block(F_alpha_pert, C_occ_alpha, C_vir_alpha);
    }
    if (have_beta) {
        Eigen::MatrixXd F_beta_pert = J_pert;
        if (alpha_hf != 0.0) {
            F_beta_pert.noalias() -= alpha_hf * jk.build_K(D_beta_pert);
        }
        if (xc_kernel != nullptr) {
            F_beta_pert.noalias() += xc_out.beta;
        }
        out.beta.noalias() +=
            ov_block(F_beta_pert, C_occ_beta, C_vir_beta);
    }
    return out;
}

// Inner product on stacked (κ_α, κ_β) for the CG solve.
double uhf_kappa_dot(const UHFKappa& a, const UHFKappa& b) {
    double s = 0.0;
    if (a.alpha.size() != 0) {
        s += (a.alpha.array() * b.alpha.array()).sum();
    }
    if (a.beta.size() != 0) {
        s += (a.beta.array()  * b.beta.array()).sum();
    }
    return s;
}

double uhf_kappa_norm(const UHFKappa& a) {
    return std::sqrt(uhf_kappa_dot(a, a));
}

// Preconditioned CG for the UHF Newton step. Mirrors the closed-shell
// path but iterates on stacked (κ_α, κ_β).
struct UHFCGResult {
    UHFKappa x;
    int n_iter = 0;
    bool converged = false;
};

UHFCGResult uhf_preconditioned_cg(
    const UHFKappa& b,
    const Eigen::MatrixXd& C_occ_alpha, const Eigen::MatrixXd& C_vir_alpha,
    const Eigen::MatrixXd& C_occ_beta,  const Eigen::MatrixXd& C_vir_beta,
    const Eigen::MatrixXd& eps_diff_alpha,
    const Eigen::MatrixXd& eps_diff_beta,
    const JKBuilder& jk,
    int max_iter, double tol,
    const UHFXCKernelBuilder* xc_kernel,
    double alpha_hf)
{
    UHFCGResult res;
    res.x.alpha = Eigen::MatrixXd::Zero(b.alpha.rows(), b.alpha.cols());
    res.x.beta  = Eigen::MatrixXd::Zero(b.beta.rows(),  b.beta.cols());

    const double b_norm = uhf_kappa_norm(b);
    if (b_norm == 0.0) {
        res.converged = true;
        return res;
    }

    UHFKappa r;
    r.alpha = b.alpha;
    r.beta  = b.beta;

    auto precondition = [&](const UHFKappa& v) {
        UHFKappa out;
        if (v.alpha.size() != 0) out.alpha = v.alpha.cwiseQuotient(eps_diff_alpha);
        if (v.beta.size()  != 0) out.beta  = v.beta .cwiseQuotient(eps_diff_beta);
        return out;
    };

    UHFKappa z = precondition(r);
    UHFKappa p = z;
    double rz = uhf_kappa_dot(r, z);

    for (int k = 1; k <= max_iter; ++k) {
        const UHFKappa Ap = uhf_orbital_hessian_action(
            p, C_occ_alpha, C_vir_alpha,
            C_occ_beta,  C_vir_beta,
            eps_diff_alpha, eps_diff_beta, jk,
            xc_kernel, alpha_hf);
        const double pAp = uhf_kappa_dot(p, Ap);
        if (pAp <= 0.0) {
            // Non-PD curvature — bail before the step blows up.
            res.n_iter = k;
            res.converged = false;
            return res;
        }
        const double alpha_step = rz / pAp;
        if (res.x.alpha.size() != 0) res.x.alpha.noalias() += alpha_step * p.alpha;
        if (res.x.beta.size()  != 0) res.x.beta .noalias() += alpha_step * p.beta;
        if (r.alpha.size() != 0)     r.alpha   .noalias() -= alpha_step * Ap.alpha;
        if (r.beta.size()  != 0)     r.beta    .noalias() -= alpha_step * Ap.beta;
        if (uhf_kappa_norm(r) < tol * b_norm) {
            res.n_iter = k;
            res.converged = true;
            return res;
        }
        z = precondition(r);
        const double rz_new = uhf_kappa_dot(r, z);
        const double beta_step = rz_new / rz;
        if (p.alpha.size() != 0) p.alpha = z.alpha + beta_step * p.alpha;
        if (p.beta.size()  != 0) p.beta  = z.beta  + beta_step * p.beta;
        rz = rz_new;
    }
    res.n_iter = max_iter;
    res.converged = false;
    return res;
}

// Pad a (n_vir, n_occ) κ block into a full (n_kept, n_kept)
// antihermitian rotation generator for ``expm_skew``. Empty κ
// (n_vir == 0) → identity rotation, returned as zero generator.
Eigen::MatrixXd kappa_to_full_skew(const Eigen::MatrixXd& kappa_ov,
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

// Per-spin (ε_a − ε_i) preconditioner, shape (n_vir, n_occ). Empty
// when n_vir == 0. Defensive floor matches the RHF path.
Eigen::MatrixXd eps_diff_matrix(const Eigen::VectorXd& eps,
                                  Eigen::Index n_occ,
                                  Eigen::Index n_vir) {
    Eigen::MatrixXd out(n_vir, n_occ);
    if (n_vir == 0 || n_occ == 0) {
        return Eigen::MatrixXd::Zero(n_vir, n_occ);
    }
    for (Eigen::Index a = 0; a < n_vir; ++a) {
        const double eps_a = eps(n_occ + a);
        for (Eigen::Index i = 0; i < n_occ; ++i) {
            const double diff = eps_a - eps(i);
            out(a, i) = std::max(diff, 1e-6);
        }
    }
    return out;
}

}  // namespace

UHFNewtonStepResult uhf_newton_step(const Eigen::MatrixXd& F_alpha,
                                  const Eigen::MatrixXd& F_beta,
                                  const Eigen::MatrixXd& C_alpha_prev,
                                  const Eigen::MatrixXd& C_beta_prev,
                                  const Eigen::VectorXd& eps_alpha_prev,
                                  const Eigen::VectorXd& eps_beta_prev,
                                  int n_alpha,
                                  int n_beta,
                                  const JKBuilder& jk,
                                  const NewtonOptions& opts,
                                  const UHFXCKernelBuilder* xc_kernel,
                                  double alpha_hf) {
    const Eigen::Index n_kept_alpha = C_alpha_prev.cols();
    const Eigen::Index n_kept_beta  = C_beta_prev.cols();
    const Eigen::Index n_vir_alpha = n_kept_alpha - n_alpha;
    const Eigen::Index n_vir_beta  = n_kept_beta  - n_beta;

    UHFNewtonStepResult out;

    // F in current per-spin MO basis; symmetrise to absorb numerical drift.
    Eigen::MatrixXd F_mo_alpha = C_alpha_prev.transpose() * F_alpha * C_alpha_prev;
    Eigen::MatrixXd F_mo_beta  = C_beta_prev .transpose() * F_beta  * C_beta_prev;
    F_mo_alpha = 0.5 * (F_mo_alpha + F_mo_alpha.transpose());
    F_mo_beta  = 0.5 * (F_mo_beta  + F_mo_beta .transpose());

    // RHS: -F_σ^MO occ-vir per spin. Empty when the spin has no virtual.
    UHFKappa b;
    if (n_vir_alpha > 0 && n_alpha > 0) {
        b.alpha = -F_mo_alpha.bottomLeftCorner(n_vir_alpha, n_alpha);
    }
    if (n_vir_beta > 0 && n_beta > 0) {
        b.beta = -F_mo_beta.bottomLeftCorner(n_vir_beta, n_beta);
    }

    // Preconditioner (ε_a − ε_i) per spin.
    const Eigen::MatrixXd eps_diff_alpha =
        eps_diff_matrix(eps_alpha_prev, n_alpha, n_vir_alpha);
    const Eigen::MatrixXd eps_diff_beta =
        eps_diff_matrix(eps_beta_prev,  n_beta,  n_vir_beta);

    // Per-spin occ/vir splits for the matvec.
    const Eigen::MatrixXd C_occ_alpha =
        (n_alpha > 0) ? C_alpha_prev.leftCols(n_alpha) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_vir_alpha =
        (n_vir_alpha > 0) ? C_alpha_prev.rightCols(n_vir_alpha)
                          : Eigen::MatrixXd();
    const Eigen::MatrixXd C_occ_beta =
        (n_beta > 0) ? C_beta_prev.leftCols(n_beta) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_vir_beta =
        (n_vir_beta > 0) ? C_beta_prev.rightCols(n_vir_beta)
                         : Eigen::MatrixXd();

    const UHFCGResult cg = uhf_preconditioned_cg(
        b,
        C_occ_alpha, C_vir_alpha,
        C_occ_beta,  C_vir_beta,
        eps_diff_alpha, eps_diff_beta,
        jk, opts.cg_max_iter, opts.cg_tol,
        xc_kernel, alpha_hf);

    UHFKappa kappa = cg.x;
    out.kappa_norm = uhf_kappa_norm(kappa);

    // Trust-region cap on the combined ‖(κ_α, κ_β)‖_F. Scaling is
    // applied uniformly to both spins so the per-spin gradient
    // directions stay coherent (no "tilt" between α and β rotations).
    if (out.kappa_norm > opts.trust_radius) {
        const double scale = opts.trust_radius / out.kappa_norm;
        if (kappa.alpha.size() != 0) kappa.alpha *= scale;
        if (kappa.beta.size()  != 0) kappa.beta  *= scale;
    }

    // Rotate per-spin: C_new = C_prev · exp(K_full).
    const Eigen::MatrixXd K_alpha_full =
        kappa_to_full_skew(kappa.alpha, n_kept_alpha, n_alpha);
    const Eigen::MatrixXd K_beta_full =
        kappa_to_full_skew(kappa.beta,  n_kept_beta,  n_beta);
    const Eigen::MatrixXd U_alpha = expm_skew(K_alpha_full);
    const Eigen::MatrixXd U_beta  = expm_skew(K_beta_full);
    out.C_alpha = C_alpha_prev * U_alpha;
    out.C_beta  = C_beta_prev  * U_beta;
    out.eps_alpha = (out.C_alpha.transpose() * F_alpha * out.C_alpha).diagonal();
    out.eps_beta  = (out.C_beta .transpose() * F_beta  * out.C_beta ).diagonal();
    out.cg_iter = cg.n_iter;
    out.cg_converged = cg.converged;
    return out;
}

// ---------------------------------------------------------------------------
// Internal UHF stability analysis: lowest eigenpair of the coupled
// per-spin orbital-rotation Hessian, matrix-free Davidson. See the
// header notes for the stability criterion (Lehtola 2020 Section 10 /
// Seeger-Pople 1977) and the operator definition (identical to
// uhf_orbital_hessian_action above, i.e. the same E'' the Newton /
// TRAH steps invert).
// ---------------------------------------------------------------------------
UHFStabilityResult uhf_internal_stability_lowest(
    const Eigen::MatrixXd& C_alpha,
    const Eigen::MatrixXd& C_beta,
    const Eigen::VectorXd& eps_alpha,
    const Eigen::VectorXd& eps_beta,
    int n_alpha,
    int n_beta,
    const JKBuilder& jk,
    const UHFStabilityOptions& opts,
    const UHFXCKernelBuilder* xc_kernel,
    double alpha_hf) {
    const Eigen::Index n_kept_alpha = C_alpha.cols();
    const Eigen::Index n_kept_beta  = C_beta.cols();
    const Eigen::Index n_vir_alpha = n_kept_alpha - n_alpha;
    const Eigen::Index n_vir_beta  = n_kept_beta  - n_beta;
    const Eigen::Index dim_alpha =
        (n_alpha > 0 && n_vir_alpha > 0) ? n_vir_alpha * n_alpha : 0;
    const Eigen::Index dim_beta =
        (n_beta > 0 && n_vir_beta > 0) ? n_vir_beta * n_beta : 0;
    const Eigen::Index dim = dim_alpha + dim_beta;

    UHFStabilityResult out;
    if (dim == 0) {
        // No rotation degrees of freedom at all — trivially stable.
        out.converged = true;
        out.lowest_eigenvalue = 0.0;
        return out;
    }

    const Eigen::MatrixXd C_occ_alpha =
        (dim_alpha > 0) ? C_alpha.leftCols(n_alpha) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_vir_alpha =
        (dim_alpha > 0) ? C_alpha.rightCols(n_vir_alpha) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_occ_beta =
        (dim_beta > 0) ? C_beta.leftCols(n_beta) : Eigen::MatrixXd();
    const Eigen::MatrixXd C_vir_beta =
        (dim_beta > 0) ? C_beta.rightCols(n_vir_beta) : Eigen::MatrixXd();

    // Hessian diagonal approximation for start vector + preconditioner:
    // the (eps_a - eps_i) orbital-energy differences (the two-electron
    // diagonal is not assembled matrix-free; the Davidson correction
    // only needs a rough diagonal).
    Eigen::VectorXd diag(dim);
    {
        Eigen::Index k = 0;
        for (Eigen::Index i = 0; i < n_alpha && dim_alpha > 0; ++i) {
            for (Eigen::Index a = 0; a < n_vir_alpha; ++a) {
                diag(k++) = eps_alpha(n_alpha + a) - eps_alpha(i);
            }
        }
        for (Eigen::Index i = 0; i < n_beta && dim_beta > 0; ++i) {
            for (Eigen::Index a = 0; a < n_vir_beta; ++a) {
                diag(k++) = eps_beta(n_beta + a) - eps_beta(i);
            }
        }
    }

    // Column-major (a, i) packing helpers matching Eigen's default
    // MatrixXd layout, so Map round-trips are exact.
    auto unpack = [&](const Eigen::VectorXd& v,
                      Eigen::MatrixXd& ka, Eigen::MatrixXd& kb) {
        if (dim_alpha > 0) {
            ka = Eigen::Map<const Eigen::MatrixXd>(
                v.data(), n_vir_alpha, n_alpha);
        } else {
            ka = Eigen::MatrixXd();
        }
        if (dim_beta > 0) {
            kb = Eigen::Map<const Eigen::MatrixXd>(
                v.data() + dim_alpha, n_vir_beta, n_beta);
        } else {
            kb = Eigen::MatrixXd();
        }
    };
    auto pack = [&](const Eigen::MatrixXd& ka, const Eigen::MatrixXd& kb) {
        Eigen::VectorXd v(dim);
        if (dim_alpha > 0) {
            Eigen::Map<Eigen::MatrixXd>(v.data(), n_vir_alpha, n_alpha) = ka;
        }
        if (dim_beta > 0) {
            Eigen::Map<Eigen::MatrixXd>(
                v.data() + dim_alpha, n_vir_beta, n_beta) = kb;
        }
        return v;
    };

    // NOTE: eps_diff_matrix floors (eps_a - eps_i) at 1e-6 — fine for
    // this diagonal role (it only shapes the preconditioner and the
    // diagonal part of the matvec, exactly as in the Newton step).
    const Eigen::MatrixXd eps_diff_a =
        (dim_alpha > 0) ? eps_diff_matrix(eps_alpha, n_alpha, n_vir_alpha)
                        : Eigen::MatrixXd();
    const Eigen::MatrixXd eps_diff_b =
        (dim_beta > 0) ? eps_diff_matrix(eps_beta, n_beta, n_vir_beta)
                       : Eigen::MatrixXd();

    auto matvec = [&](const Eigen::VectorXd& v) {
        UHFKappa kap;
        unpack(v, kap.alpha, kap.beta);
        const UHFKappa Hv = uhf_orbital_hessian_action(
            kap, C_occ_alpha, C_vir_alpha, C_occ_beta, C_vir_beta,
            eps_diff_a, eps_diff_b, jk, xc_kernel, alpha_hf);
        ++out.n_matvec;
        return pack(Hv.alpha, Hv.beta);
    };

    // Block-start Davidson (Davidson 1975) with thick restart. Section III,
    // step A requires a zeroth-order subspace spanning the dominant
    // components of the wanted root. A single coordinate at diag.argmin() is
    // not sufficient here: at a symmetry-block-diagonal UHF Hessian it stays
    // inside one irrep and can converge a higher root while a lower
    // instability exists in another block. Seed several dressed low-diagonal
    // directions and retain several Ritz vectors so every symmetry sector is
    // sampled while refining the global lowest root.
    if (opts.max_iter <= 0) {
        return out;
    }
    const Eigen::Index n_track = 1;
    const Eigen::Index n_seed = std::min<Eigen::Index>(
        {4, dim, static_cast<Eigen::Index>(opts.max_iter)});
    const Eigen::Index max_sub = std::min<Eigen::Index>(
        dim, std::max<Eigen::Index>(
                 n_seed, std::max<Eigen::Index>(
                              opts.max_subspace, 2 * n_track + 4)));
    std::vector<Eigen::VectorXd> V;   // orthonormal basis
    std::vector<Eigen::VectorXd> HV;  // H * basis
    auto deterministic_probe = [&](std::uint64_t state) {
        Eigen::VectorXd probe(dim);
        for (Eigen::Index k = 0; k < dim; ++k) {
            // SplitMix64: explicit integer arithmetic makes the probes
            // bit-reproducible across standard-library implementations.
            state += 0x9e3779b97f4a7c15ULL;
            std::uint64_t z = state;
            z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
            z ^= z >> 31U;
            probe(k) = static_cast<double>(z >> 11U)
                     / 9007199254740992.0 - 0.5;
        }
        return probe;
    };
    auto orthogonalize = [&](Eigen::VectorXd& probe,
                             const std::vector<Eigen::VectorXd>& extra) {
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& vp : V) {
                probe.noalias() -= vp.dot(probe) * vp;
            }
            for (const auto& vp : extra) {
                probe.noalias() -= vp.dot(probe) * vp;
            }
        }
    };
    auto append_probe = [&](Eigen::VectorXd probe) {
        static const std::vector<Eigen::VectorXd> kNoExtra;
        orthogonalize(probe, kNoExtra);
        const double norm = probe.norm();
        if (norm <= 1e-12 || out.n_matvec >= opts.max_iter) return false;
        probe /= norm;
        V.push_back(probe);
        HV.push_back(matvec(probe));
        return true;
    };
    static constexpr std::uint64_t kProbeSeeds[] = {
        0x243f6a8885a308d3ULL,
        0x13198a2e03707344ULL,
        0xa4093822299f31d0ULL,
        0x082efa98ec4e6c89ULL,
    };
    std::vector<Eigen::Index> diagonal_order(dim);
    for (Eigen::Index k = 0; k < dim; ++k) diagonal_order[k] = k;
    std::partial_sort(
        diagonal_order.begin(), diagonal_order.begin() + n_seed,
        diagonal_order.end(),
        [&](Eigen::Index lhs, Eigen::Index rhs) {
            if (diag(lhs) != diag(rhs)) return diag(lhs) < diag(rhs);
            return lhs < rhs;
        });
    for (Eigen::Index seed = 0; seed < n_seed; ++seed) {
        // Start from the lowest zeroth-order rotations, but dress every
        // coordinate with deterministic dense noise. The diagonal bias makes
        // the lowest-root solve converge within the matvec budget; the noise
        // gives each seed a component in every exact symmetry sector.
        Eigen::VectorXd probe =
            Eigen::VectorXd::Unit(dim, diagonal_order[seed]);
        probe.noalias() += 1e-3 * deterministic_probe(kProbeSeeds[seed]);
        append_probe(std::move(probe));
    }
    if (V.empty()) {
        // dim > 0 guarantees a coordinate fallback exists.
        Eigen::Index i_min = 0;
        diag.minCoeff(&i_min);
        append_probe(Eigen::VectorXd::Unit(dim, i_min));
    }

    double theta = 0.0;
    Eigen::VectorXd x = V.front();
    std::uint64_t rescue_seed = 0x452821e638d01377ULL;
    for (;;) {
        const auto m = static_cast<Eigen::Index>(V.size());
        Eigen::MatrixXd Hsub(m, m);
        for (Eigen::Index p = 0; p < m; ++p) {
            for (Eigen::Index q = 0; q < m; ++q) {
                Hsub(p, q) = V[p].dot(HV[q]);
            }
        }
        Hsub = 0.5 * (Hsub + Hsub.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hsub);
        const Eigen::VectorXd evals = es.eigenvalues();
        const Eigen::MatrixXd Y = es.eigenvectors();
        const Eigen::Index n_roots = std::min<Eigen::Index>(n_track, m);
        std::vector<Eigen::VectorXd> ritz_x;
        std::vector<Eigen::VectorXd> residuals;
        ritz_x.reserve(n_roots);
        residuals.reserve(n_roots);
        bool all_converged = true;
        for (Eigen::Index root = 0; root < n_roots; ++root) {
            Eigen::VectorXd xr = Eigen::VectorXd::Zero(dim);
            Eigen::VectorXd hxr = Eigen::VectorXd::Zero(dim);
            for (Eigen::Index p = 0; p < m; ++p) {
                xr.noalias() += Y(p, root) * V[p];
                hxr.noalias() += Y(p, root) * HV[p];
            }
            Eigen::VectorXd residual = hxr - evals(root) * xr;
            if (residual.norm() > opts.residual_tol) {
                all_converged = false;
            }
            ritz_x.push_back(std::move(xr));
            residuals.push_back(std::move(residual));
        }
        theta = evals(0);
        x = ritz_x.front();
        if (all_converged && n_roots == n_track) {
            out.converged = true;
            break;
        }
        if (out.n_matvec >= opts.max_iter) break;

        // Precondition the lowest Ritz pair. Multi-sector coverage comes from
        // the dressed starting block and the low Ritz vectors retained across
        // restarts, rather than from a symmetry-confined single coordinate.
        std::vector<Eigen::VectorXd> corrections;
        corrections.reserve(n_roots);
        for (Eigen::Index root = 0; root < n_roots; ++root) {
            const double residual_norm = residuals[root].norm();
            if (residual_norm <= opts.residual_tol) continue;
            Eigen::VectorXd t = residuals[root];
            for (Eigen::Index k = 0; k < dim; ++k) {
                double denom = diag(k) - evals(root);
                if (std::abs(denom) < 1e-2) {
                    denom = (denom < 0.0) ? -1e-2 : 1e-2;
                }
                t(k) /= denom;
            }
            const double preconditioned_norm = t.norm();
            if (preconditioned_norm > 0.0) t /= preconditioned_norm;
            orthogonalize(t, corrections);
            double norm = t.norm();
            if (norm <= 1e-10) {
                // A collapsed correction is not convergence: the residual
                // was explicitly above tolerance. Fall back to its raw
                // orthogonal direction before trying a dense rescue probe.
                t = residuals[root] / residual_norm;
                orthogonalize(t, corrections);
                norm = t.norm();
            }
            if (norm > 1e-10) {
                corrections.push_back(t / norm);
            }
        }

        if (m + static_cast<Eigen::Index>(corrections.size()) > max_sub
            || corrections.empty()) {
            // Thick restart onto several low Ritz vectors. Their H-products
            // are the same linear combinations of cached HV columns, so the
            // restart consumes no additional matrix-vector products and
            // preserves the multi-sector coverage built so far.
            const Eigen::Index keep = std::min<Eigen::Index>(
                m, std::min<Eigen::Index>(
                       max_sub, std::max(n_seed, n_track + 2)));
            std::vector<Eigen::VectorXd> kept_v;
            std::vector<Eigen::VectorXd> kept_hv;
            kept_v.reserve(keep);
            kept_hv.reserve(keep);
            for (Eigen::Index root = 0; root < keep; ++root) {
                Eigen::VectorXd xr = Eigen::VectorXd::Zero(dim);
                Eigen::VectorXd hxr = Eigen::VectorXd::Zero(dim);
                for (Eigen::Index p = 0; p < m; ++p) {
                    xr.noalias() += Y(p, root) * V[p];
                    hxr.noalias() += Y(p, root) * HV[p];
                }
                kept_v.push_back(std::move(xr));
                kept_hv.push_back(std::move(hxr));
            }
            V = std::move(kept_v);
            HV = std::move(kept_hv);

            // Pending corrections were orthogonal to the old subspace and
            // therefore to the retained one in exact arithmetic; clean them
            // up again after the finite-precision basis transformation.
            std::vector<Eigen::VectorXd> cleaned;
            cleaned.reserve(corrections.size());
            for (auto correction : corrections) {
                orthogonalize(correction, cleaned);
                const double norm = correction.norm();
                if (norm > 1e-10) cleaned.push_back(correction / norm);
            }
            corrections = std::move(cleaned);
        }
        if (corrections.empty()) {
            // Degenerate projected residuals can still leave the solve above
            // tolerance. Explore a new deterministic full-space direction;
            // never relabel that stagnation as convergence.
            for (int attempt = 0; attempt < 4 && corrections.empty(); ++attempt) {
                Eigen::VectorXd probe = deterministic_probe(rescue_seed);
                rescue_seed += 0x9e3779b97f4a7c15ULL;
                orthogonalize(probe, corrections);
                const double norm = probe.norm();
                if (norm > 1e-10) corrections.push_back(probe / norm);
            }
        }

        bool appended = false;
        for (const auto& correction : corrections) {
            if (static_cast<Eigen::Index>(V.size()) >= max_sub
                || out.n_matvec >= opts.max_iter) {
                break;
            }
            V.push_back(correction);
            HV.push_back(matvec(correction));
            appended = true;
        }
        if (!appended && out.n_matvec < opts.max_iter) break;
    }

    out.lowest_eigenvalue = theta;
    const double x_norm = x.norm();
    if (x_norm > 0.0) x /= x_norm;
    unpack(x, out.kappa_alpha, out.kappa_beta);
    return out;
}

}  // namespace vibeqc
