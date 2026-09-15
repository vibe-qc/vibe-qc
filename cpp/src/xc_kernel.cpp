// XC kernel matvec implementations. See xc_kernel.hpp for the design
// note + math derivation.
//
// Status (Phase D2c-KS-LDA + D2c-KS-GGA):
//   * `UnpolarisedLDAXCKernelBuilder` — closed-shell LDA.
//   * `UnpolarisedGGAXCKernelBuilder` — closed-shell GGA (Pople-Gill-
//     Johnson CPL 199, 557 1992 — 5 terms grouped into 3 contributions).
//   * UKS path scaffolded (polarized LDA / GGA in Phase D2c-KS-UHF).
//
// The unambiguous correctness witness for both kernels is
//
//      lim_{ε→0} (V_xc(D + ε δD) − V_xc(D − ε δD)) / (2ε) = W^XC[δD]
//
// which the dedicated `tests/test_xc_kernel.py::test_*_kernel_matches_
// finite_difference_vxc` exercise end-to-end.

#include "vibeqc/xc_kernel.hpp"

#include <exception>
#include <stdexcept>
#include <vector>

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/thread_pool.hpp"

namespace vibeqc {

namespace {

// ρ(g) = Σ_{αβ} D_{αβ} χ_α(g) χ_β(g) = diag(χ D χᵀ)(g).
//
// Efficient via (χ D) row-wise dot with χ — same one-liner the static
// build_xc in rks.cpp uses.
Eigen::VectorXd density_on_grid(const Eigen::MatrixXd& chi,
                                const Eigen::MatrixXd& D) {
    const Eigen::MatrixXd chiD = chi * D;
    return (chiD.array() * chi.array()).rowwise().sum();
}

// W^XC_{μν}[D_pert] = Σ_g w_g · v2rho2(g) · δρ(g) · χ_μ(g) χ_ν(g)
//                  = χᵀ · diag(w · v2rho2 · δρ) · χ
//
// Symmetrise on the way out to absorb numerical drift (the analytic
// expression is symmetric in μ↔ν; the matrix product would be exact in
// infinite precision but accumulates O(1e-14) drift in practice).
class UnpolarisedLDAXCKernelBuilder final : public XCKernelBuilder {
public:
    UnpolarisedLDAXCKernelBuilder(const Eigen::MatrixXd& chi,
                                  const Eigen::VectorXd& w_v2rho2)
        : chi_(chi), w_v2rho2_(w_v2rho2) {}

    Eigen::MatrixXd apply(const Eigen::MatrixXd& D_pert) const override {
        const Eigen::VectorXd delta_rho = density_on_grid(chi_, D_pert);
        const Eigen::VectorXd diag_g = w_v2rho2_.array() * delta_rho.array();
        Eigen::MatrixXd W = chi_.transpose() * diag_g.asDiagonal() * chi_;
        // Symmetrise.
        return 0.5 * (W + W.transpose());
    }

private:
    Eigen::MatrixXd chi_;
    Eigen::VectorXd w_v2rho2_;   // pre-baked grid-weight · v2rho2
};

// Unpolarised GGA fxc matvec — Pople-Gill-Johnson CPL 199, 557 (1992),
// eqs (16)-(18). The static V_xc(D) for closed-shell GGA is
//
//   V_{μν} = Σ_g w_g [ v_ρ(g) · χ_μ χ_ν
//                    + 2 v_σ(g) · ( (∇ρ·∇χ_μ) χ_ν + χ_μ (∇ρ·∇χ_ν) ) ]
//
// Linearising in D about a reference D_used:
//
//   δρ(g)   = diag(χ δD χᵀ)(g)
//   δ∇ρ(g) = 2 (χ·δD)·∇χ + symmetric counterpart  [factor 2 because
//             ∇ρ comes from a symmetric AO contraction; see the
//             closed-shell ∇ρ build in rks.cpp's build_xc]
//   δσ(g)   = 2 ∇ρ_used(g) · δ∇ρ(g)
//
// δv_ρ = v2rho2 · δρ + v2rhosigma · δσ
// δv_σ = v2rhosigma · δρ + v2sigma2 · δσ
//
// The resulting W^XC[δD] splits into three contributions:
//
//   T1: Σ_g w_g · δv_ρ · χ_μ χ_ν                 (LDA-shape; uses δσ for GGA)
//   T2: Σ_g w_g · 2 δv_σ · ( F_μ χ_ν + χ_μ F_ν ) where F_μ = ∇ρ_used·∇χ_μ
//   T3: Σ_g w_g · 2 v_σ_used · ( G_μ χ_ν + χ_μ G_ν ) where G_μ = δ∇ρ·∇χ_μ
//
// Per-apply cost: 4 matmuls (χ·δD, then three dchi[c]·δD or equivalent)
// + a few diagonal scalings + 3 outer matvecs. Still O(n_pts · n_bf²),
// same scaling as the static V_xc build.
class UnpolarisedGGAXCKernelBuilder final : public XCKernelBuilder {
public:
    UnpolarisedGGAXCKernelBuilder(
        const Eigen::MatrixXd& chi,
        const std::array<Eigen::MatrixXd, 3>& dchi,
        const Eigen::VectorXd& w,
        const std::array<Eigen::VectorXd, 3>& grho_used,
        const Eigen::VectorXd& v_sigma_used,
        const Eigen::VectorXd& v2rho2,
        const Eigen::VectorXd& v2rhosigma,
        const Eigen::VectorXd& v2sigma2)
        : chi_(chi), dchi_(dchi), w_(w),
          grho_used_(grho_used), v_sigma_used_(v_sigma_used),
          v2rho2_(v2rho2), v2rhosigma_(v2rhosigma), v2sigma2_(v2sigma2) {
        // Precompute F_used[g,μ] = Σ_c ∇ρ_used_c(g) · ∂_c χ_μ(g).
        // Reused every apply() — the reference is fixed at construction.
        F_used_ = grho_used_[0].asDiagonal() * dchi_[0]
                + grho_used_[1].asDiagonal() * dchi_[1]
                + grho_used_[2].asDiagonal() * dchi_[2];
    }

    Eigen::MatrixXd apply(const Eigen::MatrixXd& D_pert) const override {
        // (1) δρ + δ∇ρ from D_pert.
        const Eigen::MatrixXd chiD = chi_ * D_pert;          // (n_pts, n_bf)
        const Eigen::VectorXd delta_rho =
            (chiD.array() * chi_.array()).rowwise().sum();
        std::array<Eigen::VectorXd, 3> delta_grho;
        for (int c = 0; c < 3; ++c) {
            delta_grho[c] = 2.0 *
                (chiD.array() * dchi_[c].array()).rowwise().sum();
        }

        // (2) δσ = 2 ∇ρ_used · δ∇ρ.
        Eigen::VectorXd delta_sigma = Eigen::VectorXd::Zero(delta_rho.size());
        for (int c = 0; c < 3; ++c) {
            delta_sigma.array() += 2.0 * grho_used_[c].array()
                                       * delta_grho[c].array();
        }

        // (3) δv_ρ, δv_σ.
        const Eigen::VectorXd dv_rho =
            v2rho2_.array() * delta_rho.array()
          + v2rhosigma_.array() * delta_sigma.array();
        const Eigen::VectorXd dv_sigma =
            v2rhosigma_.array() * delta_rho.array()
          + v2sigma2_.array() * delta_sigma.array();

        // T1 (LDA-shape; for GGA δv_ρ also carries the δσ cross-term).
        const Eigen::VectorXd t1_diag = w_.array() * dv_rho.array();
        Eigen::MatrixXd W = chi_.transpose() * t1_diag.asDiagonal() * chi_;

        // T2 — 2 δv_σ · (F_μ χ_ν + χ_μ F_ν).
        const Eigen::VectorXd t2_diag = 2.0 * w_.array() * dv_sigma.array();
        const Eigen::MatrixXd Ft2 =
            F_used_.transpose() * t2_diag.asDiagonal() * chi_;
        W.noalias() += Ft2;
        W.noalias() += Ft2.transpose();

        // T3 — 2 v_σ_used · (G_μ χ_ν + χ_μ G_ν), G_μ = δ∇ρ · ∇χ_μ.
        const Eigen::MatrixXd G_pert =
            delta_grho[0].asDiagonal() * dchi_[0]
          + delta_grho[1].asDiagonal() * dchi_[1]
          + delta_grho[2].asDiagonal() * dchi_[2];
        const Eigen::VectorXd t3_diag = 2.0 * w_.array() * v_sigma_used_.array();
        const Eigen::MatrixXd Gt3 =
            G_pert.transpose() * t3_diag.asDiagonal() * chi_;
        W.noalias() += Gt3;
        W.noalias() += Gt3.transpose();

        // Final symmetrisation absorbs O(1e-14) drift from the
        // independent matmul accumulations.
        return 0.5 * (W + W.transpose());
    }

private:
    Eigen::MatrixXd chi_;
    std::array<Eigen::MatrixXd, 3> dchi_;
    Eigen::VectorXd w_;
    std::array<Eigen::VectorXd, 3> grho_used_;
    Eigen::VectorXd v_sigma_used_;
    Eigen::VectorXd v2rho2_;
    Eigen::VectorXd v2rhosigma_;
    Eigen::VectorXd v2sigma2_;
    Eigen::MatrixXd F_used_;     // precomputed F[g,μ] = ∇ρ_used·∇χ_μ
};

}  // namespace

std::unique_ptr<XCKernelBuilder> make_unpolarised_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D_used) {

    // ρ(g) at the reference density — needed for both LDA and GGA.
    const Eigen::VectorXd rho = density_on_grid(chi, D_used);

    if (func.kind() == XCKind::LDA) {
        // LDA: only v2rho2 enters the matvec. Cache w · v2rho2.
        const Eigen::VectorXd sigma;   // empty for LDA
        Eigen::VectorXd v2rho2, v2rhosigma, v2sigma2;
        func.eval_unpolarised_fxc(rho, sigma, v2rho2, v2rhosigma, v2sigma2);
        (void)dchi;
        Eigen::VectorXd w_v2rho2 = grid.weights.array() * v2rho2.array();
        return std::make_unique<UnpolarisedLDAXCKernelBuilder>(chi, w_v2rho2);
    }

    // GGA: need ∇ρ_used + σ_used + v_σ_used + the 3 second-deriv vectors.
    const Eigen::MatrixXd chiD_used = chi * D_used;
    std::array<Eigen::VectorXd, 3> grho_used;
    for (int c = 0; c < 3; ++c) {
        grho_used[c] = 2.0 *
            (chiD_used.array() * dchi[c].array()).rowwise().sum();
    }
    Eigen::VectorXd sigma_used =
        grho_used[0].array().square()
      + grho_used[1].array().square()
      + grho_used[2].array().square();

    // v_sigma at the reference density — needed for T3.
    Eigen::VectorXd exc, v_rho_used, v_sigma_used;
    func.eval_unpolarised(rho, sigma_used, exc, v_rho_used, v_sigma_used);

    // Second derivatives at the reference density.
    Eigen::VectorXd v2rho2, v2rhosigma, v2sigma2;
    func.eval_unpolarised_fxc(rho, sigma_used, v2rho2, v2rhosigma, v2sigma2);

    return std::make_unique<UnpolarisedGGAXCKernelBuilder>(
        chi, dchi, grid.weights,
        grho_used, v_sigma_used,
        v2rho2, v2rhosigma, v2sigma2);
}

// Open-shell polarised LDA fxc matvec — per spin σ:
//
//   W^XC_σ[δD_α, δD_β]_{μν}
//     = Σ_g w_g · χ_μ χ_ν · [ v2rho2_σα(g) δρ_α(g)
//                            + v2rho2_σβ(g) δρ_β(g) ]
//
// with v2rho2_αα = v_aa, v2rho2_αβ = v2rho2_βα = v_ab, v2rho2_ββ = v_bb
// (the αβ cross-term is symmetric by construction). The α↔β coupling
// lives entirely in v_ab — without it the two spins decouple and the
// matvec reduces to two independent LDA matvecs.
//
// Per-apply cost: 2 × (n_pts × n_bf²) matmuls (one per spin's δρ),
// plus 2 × outer matvecs. O(n_pts · n_bf²) per spin, scales like the
// per-spin V_xc build.
class PolarisedLDAXCKernelBuilder final : public UHFXCKernelBuilder {
public:
    PolarisedLDAXCKernelBuilder(const Eigen::MatrixXd& chi,
                                 const Eigen::VectorXd& w_v_aa,
                                 const Eigen::VectorXd& w_v_ab,
                                 const Eigen::VectorXd& w_v_bb)
        : chi_(chi), w_v_aa_(w_v_aa), w_v_ab_(w_v_ab), w_v_bb_(w_v_bb) {}

    Output apply(const Eigen::MatrixXd& D_pert_alpha,
                 const Eigen::MatrixXd& D_pert_beta) const override {
        const Eigen::VectorXd delta_rho_a =
            density_on_grid(chi_, D_pert_alpha);
        const Eigen::VectorXd delta_rho_b =
            density_on_grid(chi_, D_pert_beta);

        // W^XC_α = χᵀ · diag(w · v_aa · δρ_α + w · v_ab · δρ_β) · χ
        const Eigen::VectorXd diag_a =
              w_v_aa_.array() * delta_rho_a.array()
            + w_v_ab_.array() * delta_rho_b.array();
        Eigen::MatrixXd W_alpha = chi_.transpose() * diag_a.asDiagonal() * chi_;

        // W^XC_β = χᵀ · diag(w · v_ab · δρ_α + w · v_bb · δρ_β) · χ
        const Eigen::VectorXd diag_b =
              w_v_ab_.array() * delta_rho_a.array()
            + w_v_bb_.array() * delta_rho_b.array();
        Eigen::MatrixXd W_beta = chi_.transpose() * diag_b.asDiagonal() * chi_;

        // Symmetrise (drift absorbtion).
        Output out;
        out.alpha = 0.5 * (W_alpha + W_alpha.transpose());
        out.beta  = 0.5 * (W_beta  + W_beta.transpose());
        return out;
    }

private:
    Eigen::MatrixXd chi_;
    Eigen::VectorXd w_v_aa_, w_v_ab_, w_v_bb_;
};

std::unique_ptr<UHFXCKernelBuilder> make_polarised_lda_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta) {

    if (func.kind() != XCKind::LDA) {
        throw std::runtime_error(
            "make_polarised_lda_xc_kernel_builder: only LDA functionals "
            "are supported. For GGA / hybrid-GGA use "
            "make_polarised_gga_xc_kernel_builder, or "
            "make_polarised_xc_kernel_builder for automatic dispatch.");
    }

    // ρ_α(g) and ρ_β(g) at the reference density.
    const Eigen::VectorXd rho_a = density_on_grid(chi, D_used_alpha);
    const Eigen::VectorXd rho_b = density_on_grid(chi, D_used_beta);

    Eigen::VectorXd v_aa, v_ab, v_bb;
    func.eval_polarised_lda_fxc(rho_a, rho_b, v_aa, v_ab, v_bb);

    const Eigen::VectorXd w_v_aa = grid.weights.array() * v_aa.array();
    const Eigen::VectorXd w_v_ab = grid.weights.array() * v_ab.array();
    const Eigen::VectorXd w_v_bb = grid.weights.array() * v_bb.array();
    return std::make_unique<PolarisedLDAXCKernelBuilder>(
        chi, w_v_aa, w_v_ab, w_v_bb);
}

// ---------------------------------------------------------------------------
// Phase 17e — open-shell polarised GGA fxc matvec.
//
// The static UKS GGA potential for spin σ (mirrors cpp/src/uks.cpp's
// build_xc) is
//
//   V_σ,μν = Σ_g w_g [ v_ρσ · χ_μ χ_ν + U_σ · ∇(χ_μ χ_ν) ]
//
// with the per-spin "flow vector"
//
//   U_α = 2 v_σαα ∇ρ_α + v_σαβ ∇ρ_β
//   U_β = 2 v_σββ ∇ρ_β + v_σαβ ∇ρ_α
//
// and ∇(χ_μ χ_ν) = ∇χ_μ χ_ν + χ_μ ∇χ_ν.
//
// Linearising about the reference (D_α^used, D_β^used) — perturbations
// δD_α, δD_β induce
//
//   δρ_σ   = diag(χ δD_σ χᵀ)
//   δ∇ρ_σ = 2 (χ δD_σ) · ∇χ
//   δσ_αα = 2 ∇ρ_α·δ∇ρ_α
//   δσ_ββ = 2 ∇ρ_β·δ∇ρ_β
//   δσ_αβ = ∇ρ_α·δ∇ρ_β + ∇ρ_β·δ∇ρ_α
//
// The five first-derivative perturbations (δv_ρα, δv_ρβ, δv_σαα,
// δv_σαβ, δv_σββ) are linear combinations of (δρ_α, δρ_β, δσ_αα,
// δσ_αβ, δσ_ββ) with the 15 polarised-GGA fxc pieces as coefficients
// — i.e. one symmetric 5×5 fxc Hessian contracted against the
// 5-vector of density perturbations.
//
// δV_σ then splits, exactly as in the closed-shell GGA builder, into
//   T1: Σ_g w_g δv_ρσ · χ_μ χ_ν
//   T2: the "δv_σ in the flow prefactor" piece (∇ρ at the reference)
//   T3: the "δ∇ρ in the flow vector" piece (v_σ at the reference)
// ---------------------------------------------------------------------------
class PolarisedGGAXCKernelBuilder final : public UHFXCKernelBuilder {
public:
    PolarisedGGAXCKernelBuilder(
        const Eigen::MatrixXd& chi,
        const std::array<Eigen::MatrixXd, 3>& dchi,
        const Eigen::VectorXd& w,
        const std::array<Eigen::VectorXd, 3>& grho_a_used,
        const std::array<Eigen::VectorXd, 3>& grho_b_used,
        const Eigen::VectorXd& v_saa_used,
        const Eigen::VectorXd& v_sab_used,
        const Eigen::VectorXd& v_sbb_used,
        const PolarisedGGAFxc& fxc)
        : chi_(chi), dchi_(dchi), w_(w),
          v_saa_used_(v_saa_used), v_sab_used_(v_sab_used),
          v_sbb_used_(v_sbb_used), fxc_(fxc) {
        // F_σ,used[g,μ] = ∇ρ_σ,used(g) · ∇χ_μ(g) — reused every apply().
        F_a_used_ = grho_a_used[0].asDiagonal() * dchi_[0]
                  + grho_a_used[1].asDiagonal() * dchi_[1]
                  + grho_a_used[2].asDiagonal() * dchi_[2];
        F_b_used_ = grho_b_used[0].asDiagonal() * dchi_[0]
                  + grho_b_used[1].asDiagonal() * dchi_[1]
                  + grho_b_used[2].asDiagonal() * dchi_[2];
        grho_a_used_ = grho_a_used;
        grho_b_used_ = grho_b_used;
    }

    Output apply(const Eigen::MatrixXd& D_pert_alpha,
                 const Eigen::MatrixXd& D_pert_beta) const override {
        // (1) δρ_σ + δ∇ρ_σ from each spin's perturbation.
        const Eigen::MatrixXd chiDa = chi_ * D_pert_alpha;
        const Eigen::MatrixXd chiDb = chi_ * D_pert_beta;
        const Eigen::VectorXd drho_a =
            (chiDa.array() * chi_.array()).rowwise().sum();
        const Eigen::VectorXd drho_b =
            (chiDb.array() * chi_.array()).rowwise().sum();
        std::array<Eigen::VectorXd, 3> dgrho_a, dgrho_b;
        for (int c = 0; c < 3; ++c) {
            dgrho_a[c] = 2.0 *
                (chiDa.array() * dchi_[c].array()).rowwise().sum();
            dgrho_b[c] = 2.0 *
                (chiDb.array() * dchi_[c].array()).rowwise().sum();
        }

        // (2) δσ — αα and ββ from one spin each, αβ couples both.
        Eigen::VectorXd dsig_aa =
            Eigen::VectorXd::Zero(drho_a.size());
        Eigen::VectorXd dsig_ab = dsig_aa;
        Eigen::VectorXd dsig_bb = dsig_aa;
        for (int c = 0; c < 3; ++c) {
            dsig_aa.array() += 2.0 * grho_a_used_[c].array()
                                   * dgrho_a[c].array();
            dsig_bb.array() += 2.0 * grho_b_used_[c].array()
                                   * dgrho_b[c].array();
            dsig_ab.array() += grho_a_used_[c].array() * dgrho_b[c].array()
                             + grho_b_used_[c].array() * dgrho_a[c].array();
        }

        // (3) The five first-derivative perturbations — the symmetric
        //     5×5 fxc Hessian contracted against (δρ_α, δρ_β, δσ_αα,
        //     δσ_αβ, δσ_ββ).
        const auto& k = fxc_;
        const Eigen::VectorXd dv_ra =
              k.v2rho2_aa.array()       * drho_a.array()
            + k.v2rho2_ab.array()       * drho_b.array()
            + k.v2rhosigma_a_aa.array() * dsig_aa.array()
            + k.v2rhosigma_a_ab.array() * dsig_ab.array()
            + k.v2rhosigma_a_bb.array() * dsig_bb.array();
        const Eigen::VectorXd dv_rb =
              k.v2rho2_ab.array()       * drho_a.array()
            + k.v2rho2_bb.array()       * drho_b.array()
            + k.v2rhosigma_b_aa.array() * dsig_aa.array()
            + k.v2rhosigma_b_ab.array() * dsig_ab.array()
            + k.v2rhosigma_b_bb.array() * dsig_bb.array();
        const Eigen::VectorXd dv_saa =
              k.v2rhosigma_a_aa.array() * drho_a.array()
            + k.v2rhosigma_b_aa.array() * drho_b.array()
            + k.v2sigma2_aa_aa.array()  * dsig_aa.array()
            + k.v2sigma2_aa_ab.array()  * dsig_ab.array()
            + k.v2sigma2_aa_bb.array()  * dsig_bb.array();
        const Eigen::VectorXd dv_sab =
              k.v2rhosigma_a_ab.array() * drho_a.array()
            + k.v2rhosigma_b_ab.array() * drho_b.array()
            + k.v2sigma2_aa_ab.array()  * dsig_aa.array()
            + k.v2sigma2_ab_ab.array()  * dsig_ab.array()
            + k.v2sigma2_ab_bb.array()  * dsig_bb.array();
        const Eigen::VectorXd dv_sbb =
              k.v2rhosigma_a_bb.array() * drho_a.array()
            + k.v2rhosigma_b_bb.array() * drho_b.array()
            + k.v2sigma2_aa_bb.array()  * dsig_aa.array()
            + k.v2sigma2_ab_bb.array()  * dsig_ab.array()
            + k.v2sigma2_bb_bb.array()  * dsig_bb.array();

        // δ∇ρ flow fields G_σ[g,μ] = δ∇ρ_σ(g) · ∇χ_μ(g) — for T3.
        const Eigen::MatrixXd G_a =
            dgrho_a[0].asDiagonal() * dchi_[0]
          + dgrho_a[1].asDiagonal() * dchi_[1]
          + dgrho_a[2].asDiagonal() * dchi_[2];
        const Eigen::MatrixXd G_b =
            dgrho_b[0].asDiagonal() * dchi_[0]
          + dgrho_b[1].asDiagonal() * dchi_[1]
          + dgrho_b[2].asDiagonal() * dchi_[2];

        Output out;
        out.alpha = spin_block(dv_ra, dv_saa, dv_sab,
                               /*F_self=*/F_a_used_, /*F_other=*/F_b_used_,
                               /*G_self=*/G_a, /*G_other=*/G_b,
                               /*v_self_used=*/v_saa_used_,
                               /*v_cross_used=*/v_sab_used_);
        out.beta  = spin_block(dv_rb, dv_sbb, dv_sab,
                               /*F_self=*/F_b_used_, /*F_other=*/F_a_used_,
                               /*G_self=*/G_b, /*G_other=*/G_a,
                               /*v_self_used=*/v_sbb_used_,
                               /*v_cross_used=*/v_sab_used_);
        return out;
    }

private:
    // Assemble W^XC_σ for one spin. "self" is the spin's own σσ
    // channel (carries the factor 2 from ∂σ_σσ/∂∇ρ_σ); "other" is the
    // αβ cross channel (factor 1).
    Eigen::MatrixXd spin_block(const Eigen::VectorXd& dv_rho,
                               const Eigen::VectorXd& dv_sig_self,
                               const Eigen::VectorXd& dv_sig_cross,
                               const Eigen::MatrixXd& F_self,
                               const Eigen::MatrixXd& F_other,
                               const Eigen::MatrixXd& G_self,
                               const Eigen::MatrixXd& G_other,
                               const Eigen::VectorXd& v_self_used,
                               const Eigen::VectorXd& v_cross_used) const {
        // T1 — LDA-shape, δv_ρ already carries the δσ cross-terms.
        const Eigen::VectorXd t1 = w_.array() * dv_rho.array();
        Eigen::MatrixXd W = chi_.transpose() * t1.asDiagonal() * chi_;

        // T2 — δv_σ in the flow prefactor, ∇ρ pinned at the reference.
        //   U_σ = 2 v_σσσ ∇ρ_σ + v_σαβ ∇ρ_σ'
        const Eigen::VectorXd t2_self =
            2.0 * w_.array() * dv_sig_self.array();
        const Eigen::VectorXd t2_cross = w_.array() * dv_sig_cross.array();
        Eigen::MatrixXd M =
            F_self.transpose()  * t2_self.asDiagonal()  * chi_
          + F_other.transpose() * t2_cross.asDiagonal() * chi_;
        W.noalias() += M;
        W.noalias() += M.transpose();

        // T3 — δ∇ρ in the flow vector, v_σ pinned at the reference.
        const Eigen::VectorXd t3_self =
            2.0 * w_.array() * v_self_used.array();
        const Eigen::VectorXd t3_cross = w_.array() * v_cross_used.array();
        Eigen::MatrixXd N =
            G_self.transpose()  * t3_self.asDiagonal()  * chi_
          + G_other.transpose() * t3_cross.asDiagonal() * chi_;
        W.noalias() += N;
        W.noalias() += N.transpose();

        // Absorb O(1e-14) drift from the independent accumulations.
        return 0.5 * (W + W.transpose());
    }

    Eigen::MatrixXd chi_;
    std::array<Eigen::MatrixXd, 3> dchi_;
    Eigen::VectorXd w_;
    std::array<Eigen::VectorXd, 3> grho_a_used_, grho_b_used_;
    Eigen::VectorXd v_saa_used_, v_sab_used_, v_sbb_used_;
    PolarisedGGAFxc fxc_;
    Eigen::MatrixXd F_a_used_, F_b_used_;
};

std::unique_ptr<UHFXCKernelBuilder> make_polarised_gga_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta) {

    if (func.kind() == XCKind::LDA) {
        throw std::runtime_error(
            "make_polarised_gga_xc_kernel_builder: LDA functional — use "
            "make_polarised_lda_xc_kernel_builder (cheaper, no gradient "
            "terms) or make_polarised_xc_kernel_builder for dispatch.");
    }
    if (func.kind() == XCKind::MGGA) {
        throw std::runtime_error(
            "make_polarised_gga_xc_kernel_builder: meta-GGA polarised fxc "
            "(τ-dependent) is not yet plumbed. UKS-MGGA second-order SCF "
            "is restricted to the C1c quadratic-fallback / EDIIS+DIIS / "
            "level-shift convergence path.");
    }

    // ρ_σ + ∇ρ_σ at the reference density.
    const Eigen::MatrixXd chiDa = chi * D_used_alpha;
    const Eigen::MatrixXd chiDb = chi * D_used_beta;
    const Eigen::VectorXd rho_a =
        (chiDa.array() * chi.array()).rowwise().sum();
    const Eigen::VectorXd rho_b =
        (chiDb.array() * chi.array()).rowwise().sum();
    std::array<Eigen::VectorXd, 3> grho_a, grho_b;
    for (int c = 0; c < 3; ++c) {
        grho_a[c] = 2.0 * (chiDa.array() * dchi[c].array()).rowwise().sum();
        grho_b[c] = 2.0 * (chiDb.array() * dchi[c].array()).rowwise().sum();
    }
    Eigen::VectorXd sigma_aa =
        grho_a[0].array().square() + grho_a[1].array().square()
      + grho_a[2].array().square();
    Eigen::VectorXd sigma_bb =
        grho_b[0].array().square() + grho_b[1].array().square()
      + grho_b[2].array().square();
    Eigen::VectorXd sigma_ab =
        grho_a[0].array() * grho_b[0].array()
      + grho_a[1].array() * grho_b[1].array()
      + grho_a[2].array() * grho_b[2].array();

    // v_σ at the reference density — needed for T3.
    Eigen::VectorXd exc, v_rho_a, v_rho_b, v_saa, v_sab, v_sbb;
    func.eval_polarised(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                        exc, v_rho_a, v_rho_b, v_saa, v_sab, v_sbb);

    // The 15 polarised-GGA second derivatives at the reference density.
    PolarisedGGAFxc fxc;
    func.eval_polarised_gga_fxc(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                                fxc);

    return std::make_unique<PolarisedGGAXCKernelBuilder>(
        chi, dchi, grid.weights, grho_a, grho_b,
        v_saa, v_sab, v_sbb, fxc);
}

std::unique_ptr<UHFXCKernelBuilder> make_polarised_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta) {

    if (func.kind() == XCKind::LDA) {
        return make_polarised_lda_xc_kernel_builder(
            func, grid, chi, D_used_alpha, D_used_beta);
    }
    if (func.kind() == XCKind::MGGA) {
        throw std::runtime_error(
            "make_polarised_xc_kernel_builder: meta-GGA polarised fxc "
            "(τ-dependent) is not yet plumbed. UKS-MGGA second-order SCF "
            "is restricted to the C1c quadratic-fallback / EDIIS+DIIS / "
            "level-shift convergence path.");
    }
    return make_polarised_gga_xc_kernel_builder(
        func, grid, chi, dchi, D_used_alpha, D_used_beta);
}

// ---------------------------------------------------------------------------
// Memory-bounded polarised fxc matvec (default-path stability analysis).
//
// Identical per-point math to PolarisedLDAXCKernelBuilder /
// PolarisedGGAXCKernelBuilder above, regrouped so that no whole-grid
// (n_pts × n_bf) AO table is ever resident: construction and every
// apply() traverse kMolecularXcGridBatchSize-point slices and
// re-evaluate the AOs per slice, mirroring the batched SCF-loop XC
// quadrature in rks.cpp / uks.cpp. State kept across applies is
// O(n_pts) per-point vectors only (~24 vectors for GGA, 3 for LDA).
// ---------------------------------------------------------------------------
class BatchedPolarisedXCKernelBuilder final : public UHFXCKernelBuilder {
public:
    BatchedPolarisedXCKernelBuilder(const Functional& func,
                                    const Grid& grid,
                                    const BasisSet& basis,
                                    const Eigen::MatrixXd& D_used_alpha,
                                    const Eigen::MatrixXd& D_used_beta,
                                    int max_batch_workers)
        : basis_(basis),
          points_(grid.points),
          w_(grid.weights),
          is_gga_(func.kind() != XCKind::LDA),
          workers_(omp_workers_for(
              static_cast<std::size_t>(
                  (grid.points.rows() + kMolecularXcGridBatchSize - 1)
                  / kMolecularXcGridBatchSize),
              std::min(max_batch_workers, kMolecularXcKernelMaxWorkers))) {
        const Eigen::Index n_pts = points_.rows();
        Eigen::VectorXd rho_a(n_pts), rho_b(n_pts);
        if (is_gga_) {
            for (int c = 0; c < 3; ++c) {
                grho_a_[c].resize(n_pts);
                grho_b_[c].resize(n_pts);
            }
        }
        // Reference-density fields ρ_σ (+ ∇ρ_σ for GGA), slice by slice.
        // Batches write disjoint whole-grid segments, so no reduction is
        // needed during this construction pass.
        const auto build_reference_slice = [&](Eigen::Index g0) {
            const Eigen::Index count = std::min<Eigen::Index>(
                kMolecularXcGridBatchSize, n_pts - g0);
            const Eigen::MatrixX3d pts = points_.middleRows(g0, count);
            if (is_gga_) {
                const AOValues ao = evaluate_ao_with_gradient(basis_, pts);
                const Eigen::MatrixXd chiDa = ao.values * D_used_alpha;
                const Eigen::MatrixXd chiDb = ao.values * D_used_beta;
                rho_a.segment(g0, count) =
                    (chiDa.array() * ao.values.array()).rowwise().sum();
                rho_b.segment(g0, count) =
                    (chiDb.array() * ao.values.array()).rowwise().sum();
                for (int c = 0; c < 3; ++c) {
                    grho_a_[c].segment(g0, count) = 2.0 *
                        (chiDa.array() * ao.gradients[c].array())
                            .rowwise().sum();
                    grho_b_[c].segment(g0, count) = 2.0 *
                        (chiDb.array() * ao.gradients[c].array())
                            .rowwise().sum();
                }
            } else {
                const Eigen::MatrixXd chi = evaluate_ao(basis_, pts);
                rho_a.segment(g0, count) =
                    density_on_grid(chi, D_used_alpha);
                rho_b.segment(g0, count) =
                    density_on_grid(chi, D_used_beta);
            }
        };
        const std::size_t n_batches = static_cast<std::size_t>(
            (n_pts + kMolecularXcGridBatchSize - 1)
            / kMolecularXcGridBatchSize);
        for (std::size_t wave = 0; wave < n_batches;
             wave += static_cast<std::size_t>(workers_)) {
            const int wave_size = static_cast<int>(std::min<std::size_t>(
                static_cast<std::size_t>(workers_), n_batches - wave));
            if (wave_size == 1) {
                build_reference_slice(
                    static_cast<Eigen::Index>(wave)
                    * kMolecularXcGridBatchSize);
                continue;
            }
            std::vector<std::exception_ptr> errors(
                static_cast<std::size_t>(wave_size));
            int actual_workers = 1;
            #pragma omp parallel num_threads(wave_size)
            {
                #pragma omp single
                {
                    actual_workers = omp_team_threads();
                }
                #pragma omp for schedule(static, 1)
                for (int local_batch = 0;
                     local_batch < wave_size; ++local_batch) {
                    try {
                        const Eigen::Index g0 = static_cast<Eigen::Index>(
                            wave + static_cast<std::size_t>(local_batch))
                            * kMolecularXcGridBatchSize;
                        build_reference_slice(g0);
                    } catch (...) {
                        errors[static_cast<std::size_t>(local_batch)] =
                            std::current_exception();
                    }
                }
            }
            workers_used_ = std::max(workers_used_, actual_workers);
            for (const auto& error : errors) {
                if (error) std::rethrow_exception(error);
            }
        }
        if (is_gga_) {
            Eigen::VectorXd sigma_aa =
                grho_a_[0].array().square() + grho_a_[1].array().square()
              + grho_a_[2].array().square();
            Eigen::VectorXd sigma_bb =
                grho_b_[0].array().square() + grho_b_[1].array().square()
              + grho_b_[2].array().square();
            Eigen::VectorXd sigma_ab =
                grho_a_[0].array() * grho_b_[0].array()
              + grho_a_[1].array() * grho_b_[1].array()
              + grho_a_[2].array() * grho_b_[2].array();
            Eigen::VectorXd exc, v_rho_a, v_rho_b;
            func.eval_polarised(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                                exc, v_rho_a, v_rho_b,
                                v_saa_used_, v_sab_used_, v_sbb_used_);
            func.eval_polarised_gga_fxc(rho_a, rho_b,
                                        sigma_aa, sigma_ab, sigma_bb, fxc_);
        } else {
            func.eval_polarised_lda_fxc(rho_a, rho_b,
                                        v_lda_aa_, v_lda_ab_, v_lda_bb_);
        }
    }

    Output apply(const Eigen::MatrixXd& D_pert_alpha,
                 const Eigen::MatrixXd& D_pert_beta) const override {
        const Eigen::Index n_bf = D_pert_alpha.rows();
        const Eigen::Index n_pts = points_.rows();
        Eigen::MatrixXd W_a = Eigen::MatrixXd::Zero(n_bf, n_bf);
        Eigen::MatrixXd W_b = Eigen::MatrixXd::Zero(n_bf, n_bf);
        const auto apply_slice = [&](Eigen::Index g0) {
            Output part;
            part.alpha = Eigen::MatrixXd::Zero(n_bf, n_bf);
            part.beta = Eigen::MatrixXd::Zero(n_bf, n_bf);
            Eigen::MatrixXd& W_a = part.alpha;
            Eigen::MatrixXd& W_b = part.beta;
            const Eigen::Index count = std::min<Eigen::Index>(
                kMolecularXcGridBatchSize, n_pts - g0);
            const Eigen::MatrixX3d pts = points_.middleRows(g0, count);
            const Eigen::VectorXd w = w_.segment(g0, count);
            if (!is_gga_) {
                const Eigen::MatrixXd chi = evaluate_ao(basis_, pts);
                const Eigen::VectorXd drho_a =
                    density_on_grid(chi, D_pert_alpha);
                const Eigen::VectorXd drho_b =
                    density_on_grid(chi, D_pert_beta);
                const Eigen::VectorXd diag_a = w.array()
                    * (v_lda_aa_.segment(g0, count).array() * drho_a.array()
                     + v_lda_ab_.segment(g0, count).array() * drho_b.array());
                const Eigen::VectorXd diag_b = w.array()
                    * (v_lda_ab_.segment(g0, count).array() * drho_a.array()
                     + v_lda_bb_.segment(g0, count).array() * drho_b.array());
                W_a.noalias() +=
                    chi.transpose() * diag_a.asDiagonal() * chi;
                W_b.noalias() +=
                    chi.transpose() * diag_b.asDiagonal() * chi;
                return part;
            }

            const AOValues ao = evaluate_ao_with_gradient(basis_, pts);
            const Eigen::MatrixXd& chi = ao.values;
            // (1) δρ_σ + δ∇ρ_σ on this slice.
            const Eigen::MatrixXd chiDa = chi * D_pert_alpha;
            const Eigen::MatrixXd chiDb = chi * D_pert_beta;
            const Eigen::VectorXd drho_a =
                (chiDa.array() * chi.array()).rowwise().sum();
            const Eigen::VectorXd drho_b =
                (chiDb.array() * chi.array()).rowwise().sum();
            std::array<Eigen::VectorXd, 3> dgrho_a, dgrho_b;
            std::array<Eigen::VectorXd, 3> grho_a, grho_b;
            for (int c = 0; c < 3; ++c) {
                dgrho_a[c] = 2.0 *
                    (chiDa.array() * ao.gradients[c].array()).rowwise().sum();
                dgrho_b[c] = 2.0 *
                    (chiDb.array() * ao.gradients[c].array()).rowwise().sum();
                grho_a[c] = grho_a_[c].segment(g0, count);
                grho_b[c] = grho_b_[c].segment(g0, count);
            }

            // (2) δσ — αα and ββ from one spin each, αβ couples both.
            Eigen::VectorXd dsig_aa = Eigen::VectorXd::Zero(count);
            Eigen::VectorXd dsig_ab = dsig_aa;
            Eigen::VectorXd dsig_bb = dsig_aa;
            for (int c = 0; c < 3; ++c) {
                dsig_aa.array() += 2.0 * grho_a[c].array()
                                       * dgrho_a[c].array();
                dsig_bb.array() += 2.0 * grho_b[c].array()
                                       * dgrho_b[c].array();
                dsig_ab.array() += grho_a[c].array() * dgrho_b[c].array()
                                 + grho_b[c].array() * dgrho_a[c].array();
            }

            // (3) The 5×5 fxc Hessian contracted against
            //     (δρ_α, δρ_β, δσ_αα, δσ_αβ, δσ_ββ) — slice segments of
            //     the stored whole-grid fxc vectors.
            const auto& k = fxc_;
            const Eigen::VectorXd dv_ra =
                  k.v2rho2_aa.segment(g0, count).array() * drho_a.array()
                + k.v2rho2_ab.segment(g0, count).array() * drho_b.array()
                + k.v2rhosigma_a_aa.segment(g0, count).array()
                    * dsig_aa.array()
                + k.v2rhosigma_a_ab.segment(g0, count).array()
                    * dsig_ab.array()
                + k.v2rhosigma_a_bb.segment(g0, count).array()
                    * dsig_bb.array();
            const Eigen::VectorXd dv_rb =
                  k.v2rho2_ab.segment(g0, count).array() * drho_a.array()
                + k.v2rho2_bb.segment(g0, count).array() * drho_b.array()
                + k.v2rhosigma_b_aa.segment(g0, count).array()
                    * dsig_aa.array()
                + k.v2rhosigma_b_ab.segment(g0, count).array()
                    * dsig_ab.array()
                + k.v2rhosigma_b_bb.segment(g0, count).array()
                    * dsig_bb.array();
            const Eigen::VectorXd dv_saa =
                  k.v2rhosigma_a_aa.segment(g0, count).array()
                    * drho_a.array()
                + k.v2rhosigma_b_aa.segment(g0, count).array()
                    * drho_b.array()
                + k.v2sigma2_aa_aa.segment(g0, count).array()
                    * dsig_aa.array()
                + k.v2sigma2_aa_ab.segment(g0, count).array()
                    * dsig_ab.array()
                + k.v2sigma2_aa_bb.segment(g0, count).array()
                    * dsig_bb.array();
            const Eigen::VectorXd dv_sab =
                  k.v2rhosigma_a_ab.segment(g0, count).array()
                    * drho_a.array()
                + k.v2rhosigma_b_ab.segment(g0, count).array()
                    * drho_b.array()
                + k.v2sigma2_aa_ab.segment(g0, count).array()
                    * dsig_aa.array()
                + k.v2sigma2_ab_ab.segment(g0, count).array()
                    * dsig_ab.array()
                + k.v2sigma2_ab_bb.segment(g0, count).array()
                    * dsig_bb.array();
            const Eigen::VectorXd dv_sbb =
                  k.v2rhosigma_a_bb.segment(g0, count).array()
                    * drho_a.array()
                + k.v2rhosigma_b_bb.segment(g0, count).array()
                    * drho_b.array()
                + k.v2sigma2_aa_bb.segment(g0, count).array()
                    * dsig_aa.array()
                + k.v2sigma2_ab_bb.segment(g0, count).array()
                    * dsig_ab.array()
                + k.v2sigma2_bb_bb.segment(g0, count).array()
                    * dsig_bb.array();

            // Reference-density flow fields F_σ[g,μ] = ∇ρ_σ,used(g)·∇χ_μ(g)
            // and perturbation flow fields G_σ[g,μ] = δ∇ρ_σ(g)·∇χ_μ(g),
            // rebuilt per slice instead of cached whole-grid.
            const Eigen::MatrixXd F_a =
                grho_a[0].asDiagonal() * ao.gradients[0]
              + grho_a[1].asDiagonal() * ao.gradients[1]
              + grho_a[2].asDiagonal() * ao.gradients[2];
            const Eigen::MatrixXd F_b =
                grho_b[0].asDiagonal() * ao.gradients[0]
              + grho_b[1].asDiagonal() * ao.gradients[1]
              + grho_b[2].asDiagonal() * ao.gradients[2];
            const Eigen::MatrixXd G_a =
                dgrho_a[0].asDiagonal() * ao.gradients[0]
              + dgrho_a[1].asDiagonal() * ao.gradients[1]
              + dgrho_a[2].asDiagonal() * ao.gradients[2];
            const Eigen::MatrixXd G_b =
                dgrho_b[0].asDiagonal() * ao.gradients[0]
              + dgrho_b[1].asDiagonal() * ao.gradients[1]
              + dgrho_b[2].asDiagonal() * ao.gradients[2];

            accumulate_spin_block(W_a, w, chi, dv_ra, dv_saa, dv_sab,
                                  F_a, F_b, G_a, G_b,
                                  v_saa_used_.segment(g0, count),
                                  v_sab_used_.segment(g0, count));
            accumulate_spin_block(W_b, w, chi, dv_rb, dv_sbb, dv_sab,
                                  F_b, F_a, G_b, G_a,
                                  v_sbb_used_.segment(g0, count),
                                  v_sab_used_.segment(g0, count));
            return part;
        };

        const std::size_t n_batches = static_cast<std::size_t>(
            (n_pts + kMolecularXcGridBatchSize - 1)
            / kMolecularXcGridBatchSize);
        for (std::size_t wave = 0; wave < n_batches;
             wave += static_cast<std::size_t>(workers_)) {
            const int wave_size = static_cast<int>(std::min<std::size_t>(
                static_cast<std::size_t>(workers_), n_batches - wave));
            if (wave_size == 1) {
                Output part = apply_slice(
                    static_cast<Eigen::Index>(wave)
                    * kMolecularXcGridBatchSize);
                W_a.noalias() += part.alpha;
                W_b.noalias() += part.beta;
                continue;
            }

            std::vector<Output> parts(static_cast<std::size_t>(wave_size));
            std::vector<std::exception_ptr> errors(
                static_cast<std::size_t>(wave_size));
            int actual_workers = 1;
            #pragma omp parallel num_threads(wave_size)
            {
                #pragma omp single
                {
                    actual_workers = omp_team_threads();
                }
                #pragma omp for schedule(static, 1)
                for (int local_batch = 0;
                     local_batch < wave_size; ++local_batch) {
                    try {
                        const Eigen::Index g0 = static_cast<Eigen::Index>(
                            wave + static_cast<std::size_t>(local_batch))
                            * kMolecularXcGridBatchSize;
                        parts[static_cast<std::size_t>(local_batch)] =
                            apply_slice(g0);
                    } catch (...) {
                        errors[static_cast<std::size_t>(local_batch)] =
                            std::current_exception();
                    }
                }
            }
            workers_used_ = std::max(workers_used_, actual_workers);
            for (int local_batch = 0;
                 local_batch < wave_size; ++local_batch) {
                const auto slot = static_cast<std::size_t>(local_batch);
                if (errors[slot]) std::rethrow_exception(errors[slot]);
                W_a.noalias() += parts[slot].alpha;
                W_b.noalias() += parts[slot].beta;
            }
        }
        Output out;
        out.alpha = 0.5 * (W_a + W_a.transpose());
        out.beta  = 0.5 * (W_b + W_b.transpose());
        return out;
    }

    int batch_workers_used() const noexcept override {
        return workers_used_;
    }

private:
    // One spin's T1 + T2 + T3 contribution from one grid slice —
    // the slice-local restriction of PolarisedGGAXCKernelBuilder::
    // spin_block above ("self" carries the factor 2 from
    // ∂σ_σσ/∂∇ρ_σ; "other" is the αβ cross channel, factor 1). The
    // whole-matrix symmetrisation happens once at the end of apply().
    static void accumulate_spin_block(Eigen::MatrixXd& W,
                                      const Eigen::VectorXd& w,
                                      const Eigen::MatrixXd& chi,
                                      const Eigen::VectorXd& dv_rho,
                                      const Eigen::VectorXd& dv_sig_self,
                                      const Eigen::VectorXd& dv_sig_cross,
                                      const Eigen::MatrixXd& F_self,
                                      const Eigen::MatrixXd& F_other,
                                      const Eigen::MatrixXd& G_self,
                                      const Eigen::MatrixXd& G_other,
                                      const Eigen::VectorXd& v_self_used,
                                      const Eigen::VectorXd& v_cross_used) {
        // T1 — LDA-shape, δv_ρ already carries the δσ cross-terms.
        const Eigen::VectorXd t1 = w.array() * dv_rho.array();
        W.noalias() += chi.transpose() * t1.asDiagonal() * chi;

        // T2 — δv_σ in the flow prefactor, ∇ρ pinned at the reference.
        const Eigen::VectorXd t2_self =
            2.0 * w.array() * dv_sig_self.array();
        const Eigen::VectorXd t2_cross = w.array() * dv_sig_cross.array();
        const Eigen::MatrixXd M =
            F_self.transpose()  * t2_self.asDiagonal()  * chi
          + F_other.transpose() * t2_cross.asDiagonal() * chi;
        W.noalias() += M;
        W.noalias() += M.transpose();

        // T3 — δ∇ρ in the flow vector, v_σ pinned at the reference.
        const Eigen::VectorXd t3_self =
            2.0 * w.array() * v_self_used.array();
        const Eigen::VectorXd t3_cross = w.array() * v_cross_used.array();
        const Eigen::MatrixXd N =
            G_self.transpose()  * t3_self.asDiagonal()  * chi
          + G_other.transpose() * t3_cross.asDiagonal() * chi;
        W.noalias() += N;
        W.noalias() += N.transpose();
    }

    BasisSet basis_;             // by-value: the builder is self-contained
    Eigen::MatrixX3d points_;
    Eigen::VectorXd w_;
    bool is_gga_;
    int workers_ = 1;
    mutable int workers_used_ = 1;
    // GGA reference-density state (empty for LDA).
    std::array<Eigen::VectorXd, 3> grho_a_, grho_b_;
    Eigen::VectorXd v_saa_used_, v_sab_used_, v_sbb_used_;
    PolarisedGGAFxc fxc_;
    // LDA reference-density state (empty for GGA).
    Eigen::VectorXd v_lda_aa_, v_lda_ab_, v_lda_bb_;
};

std::unique_ptr<UHFXCKernelBuilder> make_batched_polarised_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const BasisSet& basis,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta,
    int max_batch_workers) {

    if (func.kind() == XCKind::MGGA) {
        throw std::runtime_error(
            "make_batched_polarised_xc_kernel_builder: meta-GGA polarised "
            "fxc (τ-dependent) is not yet plumbed. UKS-MGGA second-order "
            "SCF is restricted to the C1c quadratic-fallback / EDIIS+DIIS "
            "/ level-shift convergence path.");
    }
    return std::make_unique<BatchedPolarisedXCKernelBuilder>(
        func, grid, basis, D_used_alpha, D_used_beta,
        max_batch_workers);
}

}  // namespace vibeqc
