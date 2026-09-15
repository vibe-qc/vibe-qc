// Phase D2c-KS / D2d-KS / D2e-KS — XC kernel matvec for the Kohn-Sham
// extension of Newton / SOSCF / TRAH second-order SCF.
//
// Status: **interface scaffold only.** The Newton, SOSCF, and TRAH free
// functions in newton.hpp / soscf.hpp / trah.hpp ship today for the HF
// case (RHF + UHF). Extending them to RKS / UKS needs the XC contribution
// to the orbital-Hessian matvec — that's what this header sketches.
// Concrete implementations land in a follow-up commit (deferred to
// v0.8.x per the SCF unification chat drop-box).
//
// What the matvec needs.
// ---------------------
//
// The HF orbital Hessian on the closed-shell (RHF) manifold is
//   H·κ = (ε_a − ε_i) κ_ai + 2 · G_HF[D^κ]^{occ-vir}_{ai}
// with G_HF[D] = J[D] − ½ K[D] supplied by the JKBuilder.
//
// For RKS we extend G_HF → G_KS where
//   G_KS[D^κ] = J[D^κ] − ½ α_HF · K[D^κ] + W^XC[D^κ]
//
// The first two terms are *already* handled by ``jk.build_g_rhf(D,
// alpha_hf)`` — the JKBuilder takes an ``alpha_hf`` scale on the K
// piece, defaulting to 1.0 for HF. The third term — W^XC[D^κ] — is
// new and is what this interface builds.
//
// Math for the W^XC matvec.
// -------------------------
//
// **Unpolarized LDA** (simplest case, ship first):
//
//   δρ(g) = Σ_{μν} D_pert_{μν} χ_μ(g) χ_ν(g)
//
//   W^XC_{μν} = Σ_g w_g · v2rho2(g) · δρ(g) · χ_μ(g) χ_ν(g)
//             = χᵀ · diag(w · v2rho2 · δρ) · χ
//
// where v2rho2(g) = ∂²f_xc/∂ρ²(ρ(g)) is precomputed once at construction
// from the *current* density D_used (the reference point of the
// linearisation), and δρ(g) is computed fresh each ``apply`` call from
// the perturbed density D_pert.
//
// **Unpolarized GGA** (LDA + extra σ pieces):
//
//   δσ(g) = 2 · ∇ρ(g) · ∇δρ(g)
//
//   W^XC_{μν} = Σ_g w_g · [
//        v2rho2 · δρ · χ_μ χ_ν
//      + v2rhosigma · ( δρ · 2 ∇ρ·∇(χ_μ χ_ν)
//                     + δσ · χ_μ χ_ν )
//      + v2sigma2 · δσ · 2 ∇ρ·∇(χ_μ χ_ν) ]
//
// Plus the v_σ · ∇δρ · ∇(χ_μ χ_ν) term (the "potential is non-local on
// δρ" piece). Five terms total. Reference: Pople-Gill-Johnson 1992,
// "Kohn-Sham density-functional theory within a finite basis set",
// CPL 199, 557, eqs. (16)-(18) — closed-shell analytic GGA Hessian.
//
// **Polarized LDA** (UKS): three v2rho2 components (αα, αβ, ββ).
// Already plumbed via Functional::eval_polarised_lda_fxc. Per-spin
// W^XC_σ[D^κ_α, D^κ_β] couples the two spins via v2rho2_αβ.
//
// **Polarized GGA** (UKS GGA / hybrid): NOT YET PLUMBED in
// Functional (the eval_polarised_gga_fxc signature is reserved for
// Phase 17e — see xc.hpp). UKS GGA second-order is gated on that.
//
// Interface design.
// -----------------
//
// One abstract base class ``XCKernelBuilder`` with two subclasses
// (closed-shell RKS, open-shell UKS). Each owns the precomputed
// state (grid, χ, ∇χ, functional, current-density-evaluated fxc
// cache) so the per-iter ``apply`` call is a few matrix multiplies.
//
// The HF Newton driver passes ``XCKernelBuilder* xc = nullptr``;
// the KS Newton driver passes a non-null pointer. The matvec inside
// newton.cpp adds the XC contribution iff the pointer is non-null:
//
//   Eigen::MatrixXd D_v = build_perturbed_density(κ, C_occ, C_vir);
//   Eigen::MatrixXd G_v = jk.build_g_rhf(D_v, alpha_hf);
//   if (xc) G_v += xc->apply(D_v);
//   out += 2.0 * ov_block(G_v, C_occ, C_vir);
//
// Same pattern extends to ``soscf_step`` and ``trah_step``.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>

#include "basis.hpp"
#include "grid.hpp"
#include "xc.hpp"

namespace vibeqc {

// Closed-shell (unpolarized) XC kernel builder. Pin to a current
// reference density D_used at construction (precomputes the fxc cache
// at the grid points), then each ``apply(D_pert)`` returns the AO-basis
// W^XC[D_pert] matrix that adds to the HF G_KS matvec.
class XCKernelBuilder {
public:
    virtual ~XCKernelBuilder() = default;

    // W^XC[D_pert] in AO basis, shape (n_bf, n_bf). Hermitian.
    virtual Eigen::MatrixXd apply(const Eigen::MatrixXd& D_pert) const = 0;
};

// Open-shell (polarized) XC kernel builder. Per-spin matvec couples
// the two spins through the off-diagonal v2rho2_αβ block. Polarized
// GGA path raises (not yet plumbed in Functional — gated on the
// Phase 17e fxc extension).
class UHFXCKernelBuilder {
public:
    virtual ~UHFXCKernelBuilder() = default;

    struct Output {
        Eigen::MatrixXd alpha;  // W^XC_α[D^κ_α, D^κ_β]
        Eigen::MatrixXd beta;   // W^XC_β[D^κ_α, D^κ_β]
    };

    virtual Output apply(const Eigen::MatrixXd& D_pert_alpha,
                         const Eigen::MatrixXd& D_pert_beta) const = 0;

    virtual int batch_workers_used() const noexcept { return 1; }
};

// ---- Factories (concrete implementations land in xc_kernel.cpp) -----------

// Closed-shell unpolarized LDA / GGA kernel builder. Precomputes
// v2rho2 (+ v2rhosigma, v2sigma2 for GGA) at the grid points from
// the supplied current density D_used; the per-iter ``apply`` cost
// is O(n_pts · n_bf²). Pure-LDA functionals get the cheap LDA-only
// path automatically. Meta-GGA / range-separated raise (Phase 17e+).
//
// Status: **scaffold** — header signature is committed, implementation
// is a placeholder that ``throw``s with a clear roadmap pointer. The
// actual implementation needs ~3-5 days of focused work (math +
// finite-difference validation per functional class). Tracked in
// `.release-status/v0.8.0/scf-unification-thirsty-payne.md` and the
// roadmap as v0.8.x.
std::unique_ptr<XCKernelBuilder> make_unpolarised_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,                   // (n_pts, n_bf)
    const std::array<Eigen::MatrixXd, 3>& dchi,   // 3 × (n_pts, n_bf)
    const Eigen::MatrixXd& D_used);

// Open-shell polarized LDA-only kernel builder (Phase D2c-KS-UHF).
// Raises on non-LDA functionals — callers wanting LDA-or-GGA dispatch
// should use ``make_polarised_xc_kernel_builder`` instead.
std::unique_ptr<UHFXCKernelBuilder> make_polarised_lda_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta);

// Phase 17e — open-shell polarized GGA kernel builder. Per-spin W^XC_σ
// matvec for spin-polarised GGA / hybrid-GGA functionals: the
// open-shell analogue of UnpolarisedGGAXCKernelBuilder, with the αβ
// coupling carried through v2rho2_ab, v2rhosigma_*, v2sigma2_* and the
// σ_ab cross term. Needs the AO gradient ``dchi`` (3 × (n_pts, n_bf)).
// Raises on LDA (use the LDA builder — cheaper) and on meta-GGA.
std::unique_ptr<UHFXCKernelBuilder> make_polarised_gga_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta);

// Unified open-shell kernel-builder factory: dispatches on the
// functional kind — LDA → make_polarised_lda_xc_kernel_builder (the
// ``dchi`` argument is ignored), GGA / hybrid-GGA →
// make_polarised_gga_xc_kernel_builder, meta-GGA → raises with a
// roadmap pointer. This is the entry point the UKS Newton / TRAH
// drivers call so they need no kind-specific branching.
std::unique_ptr<UHFXCKernelBuilder> make_polarised_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const Eigen::MatrixXd& chi,
    const std::array<Eigen::MatrixXd, 3>& dchi,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta);

// Memory-bounded variant of ``make_polarised_xc_kernel_builder`` for
// default-path callers (the post-convergence UKS internal stability
// analysis). The dense builders above cache whole-grid (n_pts × n_bf)
// AO tables — acceptable for the opt-in Newton/TRAH second-order SCF,
// whose memory preflight charges for them, but ~15 concurrent dense
// extents (+4.1 GB on a 261-bf / 1.3e5-point witness) when run
// unconditionally after every converged UKS SCF. This builder stores
// only O(n_pts) per-point vectors (weights, reference ∇ρ_σ, v_σ, fxc)
// and re-evaluates AOs in ``kMolecularXcGridBatchSize``-point slices
// inside construction and every ``apply`` — the same streaming
// contract the batched SCF-loop XC uses (cpp/src/uks.cpp, "Semilocal
// XC on the full grid (batched)"). Output is algebraically identical
// to the dense builders (same per-point math, different summation
// grouping). Dispatches on the functional kind; raises on meta-GGA.
std::unique_ptr<UHFXCKernelBuilder> make_batched_polarised_xc_kernel_builder(
    const Functional& func,
    const Grid& grid,
    const BasisSet& basis,
    const Eigen::MatrixXd& D_used_alpha,
    const Eigen::MatrixXd& D_used_beta,
    int max_batch_workers = 1);

}  // namespace vibeqc
