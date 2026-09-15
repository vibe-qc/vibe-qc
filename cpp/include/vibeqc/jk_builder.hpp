// Strategy interface for the J / K Fock-build kernels used by every
// molecular SCF driver (RHF, UHF, RKS, UKS). Implementations own the
// preprocessed state (4-index ERI tensor for the direct path, B-tensor
// for DF, integration grid for COSX) so the per-iteration call is a
// cheap matrix multiply.
//
// Currently the SCF drivers dispatched between paths via a chain of
// std::function closures (see commit history for run_rhf / run_uhf /
// run_rks / run_uks). The closure pattern works but doesn't compose:
// each driver duplicates ~30 lines of `if (density_fit) { ... } else
// { ... }` boilerplate, the COSX path adds a third branch, and the
// periodic SCF can't share any of it. Promoting the strategy to a
// proper polymorphic class lets the molecular drivers shrink to a
// single ``jk = make_jk_builder(...)`` line and gives the periodic
// driver a stable interface to plug in to once a periodic-DF /
// periodic-COSX implementation lands.
//
// Design choices:
//   * Three primary methods — build_J / build_K / build_g_rhf. The
//     closed-shell fused build_g_rhf = J − ½·α·K has a default
//     implementation that calls build_J and build_K separately, so
//     concrete classes only override it when the kernel can actually
//     fuse work (DF: one B-tensor pass per call instead of two).
//   * Builders are immutable once constructed and called via a const
//     pointer — thread-safety in the SCF iteration is the caller's
//     responsibility (vibeqc currently runs SCF on a single thread
//     and the underlying libint engines are inside std::function
//     closures or per-thread engine pools).

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <utility>

#include "basis.hpp"
#include "cosx.hpp"   // CosxVariant for make_cosx_jk_builder
#include "grid.hpp"

namespace vibeqc {

// SCF Fock-build mode. Three settings:
//
//   CONVENTIONAL  In-core 4-index ERI tensor — materialised once at
//                 construction and reused across SCF iterations. Fast
//                 for small systems (≤ ~150 basis functions) because
//                 the per-iter cost is a single tensor contraction.
//                 O(n_bf⁴) memory; OOM at ~250 BF / def2-SVP.
//   DIRECT        Integral-driven — Cauchy-Schwarz-screened on-the-fly
//                 libint quartet evaluation each SCF iteration. The
//                 only path that survives at >250 BF; closes the ORCA
//                 speed gap on the ~50-atom / def2-SVP regime where
//                 the in-core path eats cache or OOMs.
// AUTO          Default. Resolves to DIRECT when the basis is large
//                 enough that in-core would be memory-bound, otherwise
//                 CONVENTIONAL. Threshold is the RHFOptions::
//                 scf_mode_auto_threshold field (default 140 BF, BUG 87).
//
// Orthogonal to density_fit / cosx — DF and RIJCOSX still apply on
// top of CONVENTIONAL or DIRECT and ignore this mode (their own
// integral-driven kernels supersede the four-index path).
enum class SCFMode {
    AUTO = 0,
    CONVENTIONAL = 1,
    DIRECT = 2,
};

// AUTO → CONVENTIONAL / DIRECT resolver shared by all four molecular
// SCF drivers (RHF / UHF / RKS / UKS). Above `auto_threshold` basis
// functions, the on-the-fly screened path is faster (in-core eats
// cache or OOMs); below, the in-core path wins (no per-iter integral
// re-evaluation). Calibrated against benchmarks/orca_vs_vibeqc_speed.md.
inline SCFMode resolve_scf_mode(SCFMode mode, int n_bf, int auto_threshold) {
    if (mode != SCFMode::AUTO) return mode;
    return (n_bf > auto_threshold) ? SCFMode::DIRECT : SCFMode::CONVENTIONAL;
}

class JKBuilder {
public:
    virtual ~JKBuilder() = default;

    // Coulomb matrix J(D)_{μν} = Σ_{λρ} D_{λρ} (μν|λρ).
    virtual Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const = 0;

    // Actual OpenMP team used by the most recent J build. Builders whose J
    // kernel is serial retain the default. The density-fitting builders
    // override this so regression/debug controls can distinguish a requested
    // allocation from work that actually entered a parallel region.
    virtual int last_j_workers_used() const noexcept { return 1; }

    // Actual OpenMP team used to form the density-fitting B tensor during
    // construction. Non-DF builders retain the serial default.
    virtual int last_df_transform_workers_used() const noexcept { return 1; }
    virtual int last_df_pack_workers_used() const noexcept { return 1; }
    virtual int last_df_unpack_workers_used() const noexcept { return 1; }

    // Exchange matrix K(D)_{μν} = Σ_{λρ} D_{λρ} (μλ|νρ).
    virtual Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const = 0;

    // Actual OpenMP team used by the most recent single-density exchange
    // build. Non-DF builders retain the serial default.
    virtual int last_k_workers_used() const noexcept { return 1; }

    // Paired open-shell exchange build. Stateless/default builders preserve
    // their ordinary sequential calls. Density fitting overrides this seam
    // so the two independent PATOM spin channels can share one exact outer
    // decomposition without requiring arbitrary JK builders to be
    // concurrently callable.
    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_K_pair(
        const Eigen::MatrixXd& D_alpha,
        const Eigen::MatrixXd& D_beta) const {
        Eigen::MatrixXd K_alpha = build_K(D_alpha);
        Eigen::MatrixXd K_beta = build_K(D_beta);
        return {std::move(K_alpha), std::move(K_beta)};
    }

    virtual int last_k_pair_workers_used() const noexcept { return 1; }

    // Slot-aware J and K builders. Used by UHF/UKS/RKS to feed the
    // incremental-Fock ΔD cache without collision when the same
    // builder is asked for K(D_α) and K(D_β) inside one SCF step
    // (audit 2026-05-18 — pre-fix, the incremental_fock flag was
    // RHF-only because ``build_K`` was stateless and could not
    // disambiguate per-spin densities). Stateless builders ignore
    // the slot and delegate to ``build_J`` / ``build_K``.
    //
    // Slot ID convention used by the molecular SCF drivers:
    //   build_J_slot(D_total, /*slot=*/0)
    //   build_K_slot(D_α,     /*slot=*/0)   // RKS reuses slot 0
    //   build_K_slot(D_β,     /*slot=*/1)   // UHF/UKS only
    // RHF/RKS make at most one call per (J, K) per iter, so slot 0
    // suffices on the K path; UHF/UKS need two K slots.
    virtual Eigen::MatrixXd build_J_slot(const Eigen::MatrixXd& D,
                                          int slot) const {
        (void)slot;
        return build_J(D);
    }

    virtual Eigen::MatrixXd build_K_slot(const Eigen::MatrixXd& D,
                                          int slot) const {
        (void)slot;
        return build_K(D);
    }

    // Long-range (erf-attenuated) exchange matrix
    // K_erf(D)_{μν} = Σ_{λσ} D_{λσ} (μλ| erf(ω·r₁₂)/r₁₂ |νσ).
    // The long-range piece of a range-separated hybrid's exact
    // exchange (ωB97X, ωB97X-D, …). A range-separated-hybrid SCF
    // assembles −½(cam_alpha·K + cam_beta·K_erf).
    //
    // Default: throw. Only the direct-SCF builder implements it —
    // density-fitting / COSX / in-core RSH need erf-attenuated 3-centre
    // or 4-index integrals that are not yet wired. The molecular SCF
    // drivers force the direct path when the functional is
    // range-separated, so users never hit this throw via run_rks /
    // run_uks; it guards a direct call on the wrong builder.
    virtual Eigen::MatrixXd build_K_erf(const Eigen::MatrixXd& D,
                                        double omega) const {
        (void)D; (void)omega;
        throw std::runtime_error(
            "JKBuilder::build_K_erf: range-separated hybrids (ωB97X, "
            "ωB97X-D, …) are only supported by the direct-SCF Fock "
            "builder. The density-fitting / COSX / in-core paths do "
            "not yet have an erf-attenuated K kernel — run with "
            "density_fit=false (the molecular SCF drivers select the "
            "direct builder automatically for RSH functionals).");
    }

    // Closed-shell fused G(D) = J(D) − ½ · α_HF · K(D). Default
    // implementation calls build_J and build_K separately; override
    // when the kernel can share work (DF amortises the B-tensor
    // contraction across both halves).
    virtual Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                        double alpha_hf = 1.0) const {
        Eigen::MatrixXd G = build_J(D);
        if (alpha_hf != 0.0) {
            G.noalias() -= 0.5 * alpha_hf * build_K(D);
        }
        return G;
    }

    // Update the per-quartet Schwarz cutoff at runtime. Used by the
    // SCF driver to coarsen-then-tighten direct SCF: start loose
    // (cheaper, ΔP cache + density envelope drop many shells), then
    // tighten once the gradient norm crosses ``schwarz_threshold_
    // tighten_at`` so the converged result hits the user's
    // ``schwarz_threshold``. This two-phase loose→tight screening is
    // the standard refinement of the Schwarz integral bound (Häser &
    // Ahlrichs, J. Comput. Chem. 10, 104 (1989)).
    //
    // No-op for builders whose screening is fixed at construction
    // (FourIndexJKBuilder, DFJKBuilder, COSXJKBuilder). Mutates
    // internal state but declared ``const`` because the JKBuilder
    // ABC is held by const reference in the SCF inner loop; the
    // concretes use ``mutable`` to back the threshold + cache.
    virtual void set_schwarz_threshold(double schwarz_threshold) const {
        (void)schwarz_threshold;
    }

    // True only for builders whose live Fock map changes when
    // ``set_schwarz_threshold`` is called.  Molecular SCF uses this
    // capability query to prevent a custom loose→tight switch threshold
    // from accepting convergence on the loose-screened map.  Builders with
    // fixed screening return false and do not pay a spurious validation row.
    virtual bool uses_runtime_schwarz_threshold() const { return false; }

    // Swap the COSX integration grid in place, rebuilding only the
    // grid-dependent caches (the overlap-fit Q-junction and the per-batch
    // AO cache) while keeping the basis/aux-only state (the RI-J B-tensor,
    // Schwarz table, Boys table, primitive-pair cache, one-centre
    // correction). This is what lets the multi-stage COSX grid progression
    // (cosx_staged.hpp) step coarse→fine on ONE builder without rebuilding
    // the expensive RI-J B-tensor per stage. ``const`` for the same reason
    // as ``set_schwarz_threshold`` (the ABC is held by const reference in
    // the SCF loop; the COSX concrete backs the grid with ``mutable``).
    // No-op for non-COSX builders.
    virtual void set_cosx_grid(Grid cosx_grid) const { (void)cosx_grid; }

    // Discard any cross-iteration state (incremental ΔP cache,
    // tightening flags). Called by the SCF driver on phase
    // transitions (threshold tighten, SOSCF→DIIS hand-off,
    // first-iter restart guards). No-op for stateless builders.
    virtual void reset_state() const {}

    // Disengage (or re-engage) the incremental ΔD cache at runtime.
    // The SCF driver disengages on a drift stall (IID 129): the ΔD
    // build accumulates Schwarz-screen bias across iterations, which
    // stalls tight-tolerance convergence at a ~1e-6 gradient floor;
    // the nonincremental full-density build removes that recursive
    // accumulation floor. No-op for
    // stateless builders and for DirectJKBuilder once disengaged
    // (``incremental_active()`` reports the live state).
    virtual void set_incremental(bool enabled) const { (void)enabled; }

    // True when the active build path currently uses the incremental
    // ΔD cache. False for stateless builders and for DirectJKBuilder
    // after ``set_incremental(false)``. Lets the SCF driver decide
    // whether a stall can be drift-limited before rotating orbitals.
    virtual bool incremental_active() const { return false; }

    // Post-convergence one-centre exchange correction.  Default no-op;
    // COSXJKBuilder overrides to replace intra-atom K blocks with
    // exact four-index ERI contractions.
    virtual void apply_one_center_correction(
        Eigen::Ref<Eigen::MatrixXd> K, const Eigen::MatrixXd& D) const {
        (void)K; (void)D;
    }

    // Whether ``apply_one_center_correction`` changes the post-SCF exchange
    // surface. Stability analysis must query the live builder rather than an
    // options flag because low-level callers can supply COSX directly.
    virtual bool has_post_scf_exchange_correction() const noexcept {
        return false;
    }
};

// ---- Factories ----------------------------------------------------------

// Direct four-index ERI builder. Materialises the full (μν|λρ) tensor
// once and reuses it for every SCF iteration. O(n_bf⁴) memory; only
// viable up to ~250 basis functions (~ 8 GB at n_bf = 250).
std::unique_ptr<JKBuilder> make_four_index_jk_builder(const BasisSet& basis);

// Direct (integral-driven) builder. Caches the Cauchy-Schwarz factor
// matrix Q at construction; per SCF iter, walks an 8-fold-symmetric
// shell-quartet loop and computes only the surviving quartets via
// libint on the fly. O(n_shells² + n_bf²) memory — the only path
// that survives at ~50-atom / def2-SVP scale where the in-core path
// OOMs. See cpp/include/vibeqc/jk_direct.hpp for the kernel.
//
// ``schwarz_threshold`` is the per-quartet skip bound (default 1e-10
// matches ORCA). Set to 0 to disable screening (useful for parity).
//
// ``incremental`` enables Almlöf-style ΔP incremental Fock builds.
// When true and ``build_g_rhf`` is called repeatedly, the builder
// caches D_prev + G_2e_prev internally; each call computes
// ΔD = D − D_prev and returns G_prev + G_2e[ΔD]. Per-shell density
// envelope for the Schwarz screen is computed from ΔD. Nonzero increments
// with max|ΔD| < 1 are amplitude-normalised for the screened build and
// scaled back afterwards; equivalently, their effective contribution
// threshold tightens in proportion to max|ΔD|, so the omitted update
// vanishes with the density change instead of leaving a fixed drift floor.
// Full rebuild every ``reset_freq`` calls remains an independent guard
// against recursive and floating-point error (Almlöf, Faegri & Korsell,
// J. Comput. Chem. 3, 385 (1982); Häser & Ahlrichs, J. Comput. Chem. 10,
// 104 (1989)).
//
// Stateful: an ``incremental=true`` builder is not thread-safe across
// concurrent build_g_rhf calls, and ``build_J`` / ``build_K`` do not
// update the cache. The intended use is the molecular SCF outer
// loop, which calls ``build_g_rhf(D, alpha_hf)`` sequentially.
std::unique_ptr<JKBuilder> make_direct_jk_builder(const BasisSet& basis,
                                                  double schwarz_threshold,
                                                  bool incremental = false,
                                                  int reset_freq = 8);

// Density-fitting (RIJK) builder. Owns its DensityFitting object — the
// V-metric Cholesky and the (n_aux, n_bf, n_bf) B-tensor are computed
// once at construction and contracted with D each iteration.
std::unique_ptr<JKBuilder> make_df_jk_builder(const BasisSet& basis,
                                              const BasisSet& aux);

// RIJCOSX builder. RI-J for the Coulomb piece (same B-tensor as the
// DFJKBuilder), seminumerical chain-of-spheres for the K piece on the
// supplied COSX grid (see compute_cosx_k). Pair with
// ``default_cosx_grid_options()`` for the standard sparse-grid tier.
//
// ``variant`` selects the K build (cosx.hpp ``CosxVariant``):
//   FITTED (default) applies the global overlap-fit Q-junction
//   (K = Q·K_naive·Qᵀ; Neese 2009 §2.4); STANDARD returns the
//   symmetrised seminumerical K with no fit. AUTO is treated as FITTED
//   here — the SCF / gradient drivers resolve AUTO to a concrete variant
//   via ``resolve_cosx_variant`` before constructing the builder. Both
//   variants apply the one-centre analytic correction.
std::unique_ptr<JKBuilder> make_cosx_jk_builder(
    const BasisSet& basis,
    const BasisSet& aux,
    Grid cosx_grid,
    CosxVariant variant = CosxVariant::FITTED);

}  // namespace vibeqc
