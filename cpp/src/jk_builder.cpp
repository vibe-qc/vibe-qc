#include "vibeqc/jk_builder.hpp"

#include <utility>

#include <libint2.hpp>

#include "vibeqc/cosx.hpp"
#include "vibeqc/cosx_one_center.hpp"
#include "vibeqc/df.hpp"
#include "vibeqc/diagnostics.hpp"
#include "vibeqc/fock.hpp"
#include "vibeqc/grid_batch.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/jk_direct.hpp"
#include "vibeqc/schwarz.hpp"

namespace vibeqc {

namespace {

// Direct four-index ERI builder. Owns the precomputed (μν|λρ) tensor.
class FourIndexJKBuilder final : public JKBuilder {
public:
    explicit FourIndexJKBuilder(const BasisSet& basis)
        : eri_(compute_eri(basis)) {}

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        return build_coulomb(eri_, D);
    }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        return build_exchange(eri_, D);
    }
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                double alpha_hf) const override {
        // The four-index path has a fused J − ½·K kernel for α_HF = 1
        // (RHF). For other α_HF values, fall back to the parent default
        // (separate J and K builds — slower but rare; α_HF ∈ (0, 1) for
        // hybrid DFT routes through RKS, which calls build_J / build_K
        // independently for the X functional dispatch).
        if (alpha_hf == 1.0) return build_fock_g(eri_, D);
        return JKBuilder::build_g_rhf(D, alpha_hf);
    }

private:
    Eri4D eri_;
};

// Direct (on-the-fly) Fock builder. Caches the Schwarz Q matrix once
// at construction; per SCF iter, walks an 8-fold-symmetric shell-
// quartet loop and evaluates only the surviving quartets via libint.
// Kernel: cpp/src/jk_direct.cpp. Memory O(n_shells² + n_bf²) instead
// of O(n_bf⁴) — the path that survives at >250 BF / def2-SVP.
//
// Optionally stateful: when ``incremental_=true``, ``build_g_rhf``
// caches D_prev + G_2e_prev and returns G_prev + G_2e[ΔD] on each
// subsequent call (Almlöf-style direct SCF). The per-shell density
// envelope inside the quartet screen is computed from ΔD. Small
// nonzero increments are amplitude-normalised before the screened
// build and scaled back afterwards, so their omitted contribution
// shrinks with |ΔD| instead of leaving a fixed absolute drift floor.
// Full rebuild every ``reset_freq_`` calls remains an independent
// guard against recursive and floating-point error.
class DirectJKBuilder final : public JKBuilder {
public:
    DirectJKBuilder(const BasisSet& basis, double schwarz_threshold,
                    bool incremental, int reset_freq)
        : basis_(&basis),
          schwarz_threshold_(schwarz_threshold),
          incremental_(incremental),
          reset_freq_(reset_freq < 1 ? 1 : reset_freq),
          Q_(build_q(basis)),
          sorted_pairs_(build_q_sorted_pairs(basis, Q_)) {
        VIBEQC_DIAG("jk", vibeqc::DiagLevel::VERBOSE,
            "direct  n_shells=%zu  schwarz=%.1e  inc=%s",
            Q_.rows(), schwarz_threshold_, (incremental ? "yes" : "no"));
    }

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        // ``build_J`` and ``build_K`` are stateless paths — they do
        // not feed or update the incremental cache. Callers mixing
        // them with the SCF outer loop's ``build_g_rhf`` should
        // expect the cache to stay out of sync. Use ``build_J_slot``
        // / ``build_K_slot`` for the cached entry points (UHF/UKS/RKS
        // do this after audit-fixes-4b 2026-05-18).
        return direct_compute_j(*basis_, D, Q_, sorted_pairs_,
                                 schwarz_threshold_);
    }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        return direct_compute_k(*basis_, D, Q_, sorted_pairs_,
                                 schwarz_threshold_);
    }
    Eigen::MatrixXd build_K_erf(const Eigen::MatrixXd& D,
                                double omega) const override {
        // Long-range (erf-attenuated) exchange for range-separated
        // hybrids. Stateless — not threaded through the incremental
        // ΔD cache (the regular K_full path still is); recomputed in
        // full each SCF iteration. The Coulomb Schwarz factors Q_ are
        // a valid conservative screen here (erf(ω·r)/r ≤ 1/r).
        return direct_compute_k_erf(*basis_, D, Q_, sorted_pairs_,
                                    schwarz_threshold_, omega);
    }
    Eigen::MatrixXd build_J_slot(const Eigen::MatrixXd& D,
                                  int slot) const override {
        if (!incremental_) return build_J(D);
        auto& s = j_slot_(slot);
        return apply_incremental(s, D, [this](const Eigen::MatrixXd& X) {
            return direct_compute_j(*basis_, X, Q_, sorted_pairs_,
                                     schwarz_threshold_);
        });
    }
    Eigen::MatrixXd build_K_slot(const Eigen::MatrixXd& D,
                                  int slot) const override {
        if (!incremental_) return build_K(D);
        auto& s = k_slot_(slot);
        return apply_incremental(s, D, [this](const Eigen::MatrixXd& X) {
            return direct_compute_k(*basis_, X, Q_, sorted_pairs_,
                                     schwarz_threshold_);
        });
    }
    void set_schwarz_threshold(double thr) const override {
        if (thr != schwarz_threshold_) {
            schwarz_threshold_ = thr;
            // Cached quartet bounds depend on the threshold — tighter
            // or looser, the screen changes, so all cached previous
            // results are stale. Force a full rebuild next call on
            // every slot.
            invalidate_all_caches();
        }
    }

    bool uses_runtime_schwarz_threshold() const override { return true; }

    void reset_state() const override {
        invalidate_all_caches();
    }

    void set_incremental(bool enabled) const override {
        if (incremental_ == enabled) return;
        incremental_ = enabled;
        // Any cached D_prev / G_prev pairs are stale under the new
        // engagement state, in both directions.
        invalidate_all_caches();
    }

    bool incremental_active() const override { return incremental_; }

    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                double alpha_hf) const override {
        if (!incremental_) {
            // Stateless fused J + K — one libint call per surviving
            // quartet, both halves accumulated. ~2× cheaper than
            // build_J + build_K. Falls through to direct_compute_j
            // when α_HF = 0 (pure GGA — no K work).
            return direct_compute_g_rhf(*basis_, D, Q_, sorted_pairs_,
                                        schwarz_threshold_, alpha_hf);
        }

        // Incremental ΔD path. Full rebuild on iter 0 (cache empty),
        // every ``reset_freq_`` calls, or on an α_HF change (the
        // cache is α_HF-specific because G_2e = J − ½·α·K).
        const bool need_full = !cache_valid_
            || iters_since_reset_ >= reset_freq_
            || cached_alpha_hf_ != alpha_hf;
        if (need_full) {
            G_prev_ = direct_compute_g_rhf(*basis_, D, Q_, sorted_pairs_,
                                           schwarz_threshold_, alpha_hf);
            D_prev_ = D;
            cached_alpha_hf_ = alpha_hf;
            cache_valid_ = true;
            iters_since_reset_ = 0;
            return G_prev_;
        }

        // ΔD build. G_2e is linear in D, so
        //   G_2e[D_new] = G_2e[D_prev] + G_2e[ΔD]
        // The scale-aware helper makes the screened update homogeneous
        // as |ΔD| tends to zero; see its derivation below.
        Eigen::MatrixXd delta_D = D - D_prev_;
        Eigen::MatrixXd delta_G = build_scale_aware_delta(
            std::move(delta_D),
            [this, alpha_hf](const Eigen::MatrixXd& X) {
                return direct_compute_g_rhf(
                    *basis_, X, Q_, sorted_pairs_, schwarz_threshold_,
                    alpha_hf);
            });
        G_prev_.noalias() += delta_G;
        D_prev_ = D;
        ++iters_since_reset_;
        return G_prev_;
    }

private:
    // Scale-homogeneous screened difference build.
    //
    // Häser & Ahlrichs, J. Comput. Chem. 10, 104 (1989), Eqs. 30--35,
    // show how fixed-threshold errors from screened difference-Fock
    // increments propagate into later iterations.  Their quartet bound
    // already uses the block amplitudes of ΔD.  The normalization here is
    // vibe-qc's scale-homogeneous extension: for
    //
    //   s = max_{μν} |ΔD_{μν}|,  0 < s < 1,
    //
    // screen ΔD/s at the configured threshold and multiply the contracted
    // result by s.  Since the direct contraction is linear for a fixed
    // retained-quartet set, this is equivalent to screening ΔD at the
    // tightened effective threshold s * schwarz_threshold, without driving
    // the floating-point threshold itself toward underflow.  The omitted
    // update therefore scales to zero with s.  For s >= 1 the configured
    // threshold is retained (never loosened).  A zero increment has exactly
    // zero J/K/G and bypasses the integral walk.
    template <typename Build>
    static Eigen::MatrixXd build_scale_aware_delta(Eigen::MatrixXd delta_D,
                                                    Build build) {
        const double scale = delta_D.cwiseAbs().maxCoeff();
        if (scale == 0.0) {
            return Eigen::MatrixXd::Zero(delta_D.rows(), delta_D.cols());
        }
        // ``!(scale < 1)`` deliberately includes NaN and infinity: preserve
        // the pre-existing direct-build path so its ordinary validation and
        // failure behaviour remain responsible for non-finite input.
        if (!(scale < 1.0)) return build(delta_D);

        delta_D /= scale;
        Eigen::MatrixXd delta_Y = build(delta_D);
        delta_Y *= scale;
        return delta_Y;
    }

    static Eigen::MatrixXd build_q(const BasisSet& basis) {
        ensure_libint_initialized();
        const auto& shells = basis.libint();
        libint2::Engine prototype(libint2::Operator::coulomb,
                                  shells.max_nprim(),
                                  shells.max_l(), 0);
        return compute_schwarz_factors(shells, prototype);
    }

    // Per-(J, K, slot) incremental cache. mutable because it mutates
    // during nominally-const build_J_slot / build_K_slot calls — the
    // JKBuilder ABC promises stateless reuse, and the incremental
    // path opts out of that contract on a per-instance basis
    // (``incremental_`` flag at construction).
    struct IncrementalSlot {
        bool valid = false;
        int iters_since_reset = 0;
        Eigen::MatrixXd D_prev;
        Eigen::MatrixXd Y_prev;  // J[D_prev] or K[D_prev]
    };

    // RHF/RKS uses slot 0 on both J and K; UHF/UKS uses
    // K slot 0 for alpha + K slot 1 for beta. Room for two more
    // future slots without growing the type.
    static constexpr int kNumSlots = 4;

    template <typename Build>
    Eigen::MatrixXd apply_incremental(IncrementalSlot& s,
                                       const Eigen::MatrixXd& D,
                                       Build build) const {
        const bool need_full = !s.valid
            || s.iters_since_reset >= reset_freq_;
        if (need_full) {
            s.Y_prev = build(D);
            s.D_prev = D;
            s.valid = true;
            s.iters_since_reset = 0;
            return s.Y_prev;
        }
        Eigen::MatrixXd delta_D = D - s.D_prev;
        Eigen::MatrixXd delta_Y = build_scale_aware_delta(
            std::move(delta_D), build);
        s.Y_prev.noalias() += delta_Y;
        s.D_prev = D;
        ++s.iters_since_reset;
        return s.Y_prev;
    }

    IncrementalSlot& j_slot_(int s) const {
        if (s < 0 || s >= kNumSlots) {
            throw std::out_of_range(
                "DirectJKBuilder::build_J_slot: slot index "
                + std::to_string(s) + " out of range [0, "
                + std::to_string(kNumSlots) + ")");
        }
        return j_slots_[s];
    }
    IncrementalSlot& k_slot_(int s) const {
        if (s < 0 || s >= kNumSlots) {
            throw std::out_of_range(
                "DirectJKBuilder::build_K_slot: slot index "
                + std::to_string(s) + " out of range [0, "
                + std::to_string(kNumSlots) + ")");
        }
        return k_slots_[s];
    }

    void invalidate_all_caches() const {
        cache_valid_ = false;
        iters_since_reset_ = 0;
        for (int i = 0; i < kNumSlots; ++i) {
            j_slots_[i].valid = false;
            j_slots_[i].iters_since_reset = 0;
            k_slots_[i].valid = false;
            k_slots_[i].iters_since_reset = 0;
        }
    }

    const BasisSet* basis_;  // borrowed; lifetime guarded by caller
    mutable double schwarz_threshold_;  // tweakable mid-SCF via set_schwarz_threshold
    mutable bool incremental_;  // disengageable mid-SCF via set_incremental (IID 129)
    int reset_freq_;
    Eigen::MatrixXd Q_;       // n_shells × n_shells, basis-only
    std::vector<DirectShellPair> sorted_pairs_;  // K-sorted, basis-only

    // Fused-G incremental cache (RHF + RKS pure-DFT — neither path
    // calls build_J/K separately, both use build_g_rhf).
    mutable bool cache_valid_ = false;
    mutable int iters_since_reset_ = 0;
    mutable double cached_alpha_hf_ = 0.0;
    mutable Eigen::MatrixXd D_prev_;
    mutable Eigen::MatrixXd G_prev_;

    // Per-(J, K, slot) incremental caches — extends the audit-fixes-1
    // RHF-only build_g_rhf cache to the UHF/UKS/RKS-hybrid paths
    // that call build_J_slot / build_K_slot separately.
    mutable IncrementalSlot j_slots_[kNumSlots];
    mutable IncrementalSlot k_slots_[kNumSlots];
};

// Density-fitting (RIJK) builder. Wraps DensityFitting; build_g_rhf
// uses the fused B-tensor contraction.
class DFJKBuilder final : public JKBuilder {
public:
    DFJKBuilder(const BasisSet& basis, const BasisSet& aux)
        : df_(basis, aux) {}

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        return df_.build_J(D);
    }
    int last_j_workers_used() const noexcept override {
        return df_.last_j_workers_used();
    }
    int last_df_transform_workers_used() const noexcept override {
        return df_.last_df_transform_workers_used();
    }
    int last_df_pack_workers_used() const noexcept override {
        return df_.last_df_pack_workers_used();
    }
    int last_df_unpack_workers_used() const noexcept override {
        return df_.last_df_unpack_workers_used();
    }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        return df_.build_K_density(D);
    }
    int last_k_workers_used() const noexcept override {
        return df_.last_k_workers_used();
    }
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_K_pair(
        const Eigen::MatrixXd& D_alpha,
        const Eigen::MatrixXd& D_beta) const override {
        return df_.build_K_density_pair(D_alpha, D_beta);
    }
    int last_k_pair_workers_used() const noexcept override {
        return df_.last_k_pair_workers_used();
    }
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                double alpha_hf) const override {
        // DensityFitting::build_g_rhf is hard-coded to α_HF = 1 (it
        // bakes the −½·K factor into one B-tensor pass). For α_HF ≠ 1,
        // fall back to separate J and K calls.
        if (alpha_hf == 1.0) return df_.build_g_rhf(D);
        return JKBuilder::build_g_rhf(D, alpha_hf);
    }

private:
    DensityFitting df_;
};

// COSX (RIJCOSX) builder. Coulomb is RI-J via the same B-tensor as
// DFJKBuilder; exchange is the seminumerical chain-of-spheres kernel.
//
// Holds the basis-only overlap-fit Q-junction matrix on the instance
// so every ``build_K`` call reuses it (Q = S · S_grid^{-1} is a
// function of the AO basis and the cosx grid only — both invariant
// for the lifetime of the builder). Rebuilding Q per K call would
// repeat two N×N solves + an N×N GEMM every SCF iteration; caching
// at construction is bit-identical and ~3 % wall on butanethiol
// HF/RIJCOSX (def2-TZVP, four-core reference run).
class COSXJKBuilder final : public JKBuilder {
public:
    COSXJKBuilder(const BasisSet& basis, const BasisSet& aux,
                  Grid cosx_grid, CosxVariant variant)
        : df_(basis, aux),
          basis_(&basis),
          variant_(variant),
          cosx_grid_(std::move(cosx_grid)),
          // FITTED needs the overlap-fit Q-junction; STANDARD does not —
          // skip the (basis-only) build entirely for the standard variant.
          cosx_q_(variant == CosxVariant::STANDARD
                      ? Eigen::MatrixXd()
                      : build_cosx_q(basis, cosx_grid_)),
          cosx_schwarz_(build_cosx_schwarz(basis)),
          shell_cutoffs_(
              compute_shell_radial_cutoffs(basis.libint(), 1e-12)),
          grid_batches_(
              build_grid_batches(basis, cosx_grid_, /*batch_size=*/4096,
                                 shell_cutoffs_, /*need_gradient=*/false)),
          boys_table_(build_boys_table(basis.libint().max_l())),
          pp_cache_(build_primitive_pair_cache(basis.libint())),
          one_center_(basis) {}

    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const override {
        return df_.build_J(D);
    }
    int last_j_workers_used() const noexcept override {
        return df_.last_j_workers_used();
    }
    int last_df_transform_workers_used() const noexcept override {
        return df_.last_df_transform_workers_used();
    }
    int last_df_pack_workers_used() const noexcept override {
        return df_.last_df_pack_workers_used();
    }
    int last_df_unpack_workers_used() const noexcept override {
        return df_.last_df_unpack_workers_used();
    }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const override {
        // STANDARD → no overlap fit (cosx_q_ is empty and unused);
        // FITTED → apply the cached global Q-junction.
        const bool fit = (variant_ != CosxVariant::STANDARD);
        Eigen::MatrixXd K_cosx = compute_cosx_k(
            *basis_, D, cosx_grid_,
            cosx_q_, cosx_schwarz_, shell_cutoffs_,
            &grid_batches_,
            /*boys_table=*/nullptr, /*pp_cache=*/nullptr,
            /*lattice=*/nullptr, /*image_cells=*/nullptr,
            /*apply_overlap_fit=*/fit);
        return K_cosx;
    }
    // No fused build_g_rhf override: COSX-K uses a different kernel
    // from RI-J, so the parent default (separate J and K) is exact.

    void apply_one_center_correction(
        Eigen::Ref<Eigen::MatrixXd> K,
        const Eigen::MatrixXd& D) const override {
        K.noalias() += one_center_.apply(
            D, grid_batches_, boys_table_, pp_cache_, shell_cutoffs_);
    }
    bool has_post_scf_exchange_correction() const noexcept override {
        return true;
    }

    // Swap the COSX grid for the multi-stage progression (cosx_staged.hpp).
    // Rebuilds only the grid-dependent caches (the overlap-fit Q-junction
    // and the per-batch AO cache); df_ (the RI-J B-tensor), the Schwarz
    // table, shell cutoffs, Boys / primitive-pair caches, and the
    // one-centre correction are basis/aux-only and are kept — so stepping
    // coarse→fine across SCF stages does not re-pay the expensive RI-J
    // build per stage.
    void set_cosx_grid(Grid cosx_grid) const override {
        cosx_grid_ = std::move(cosx_grid);
        cosx_q_ = (variant_ == CosxVariant::STANDARD)
                      ? Eigen::MatrixXd()
                      : build_cosx_q(*basis_, cosx_grid_);
        grid_batches_ = build_grid_batches(
            *basis_, cosx_grid_, /*batch_size=*/4096,
            shell_cutoffs_, /*need_gradient=*/false);
    }

private:
    DensityFitting df_;
    const BasisSet* basis_;  // borrowed; lifetime guarded by caller
    CosxVariant variant_;    // STANDARD (no fit) or FITTED (global Q)
    mutable Grid cosx_grid_;            // swappable via set_cosx_grid
    mutable Eigen::MatrixXd cosx_q_;    // empty for STANDARD; rebuilt on swap
    Eigen::MatrixXd cosx_schwarz_;
    std::vector<double> shell_cutoffs_;
    mutable GridBatches grid_batches_;  // rebuilt on grid swap
    BoysTable boys_table_;          // pre-built once at construction
    PrimitivePairCache pp_cache_;   // pre-built once at construction
    OneCenterCorrection one_center_;// pre-built once at construction
};

}  // namespace

std::unique_ptr<JKBuilder> make_four_index_jk_builder(const BasisSet& basis) {
    return std::make_unique<FourIndexJKBuilder>(basis);
}

std::unique_ptr<JKBuilder> make_direct_jk_builder(const BasisSet& basis,
                                                  double schwarz_threshold,
                                                  bool incremental,
                                                  int reset_freq) {
    return std::make_unique<DirectJKBuilder>(
        basis, schwarz_threshold, incremental, reset_freq);
}

std::unique_ptr<JKBuilder> make_df_jk_builder(const BasisSet& basis,
                                              const BasisSet& aux) {
    return std::make_unique<DFJKBuilder>(basis, aux);
}

std::unique_ptr<JKBuilder> make_cosx_jk_builder(const BasisSet& basis,
                                                const BasisSet& aux,
                                                Grid cosx_grid,
                                                CosxVariant variant) {
    return std::make_unique<COSXJKBuilder>(basis, aux, std::move(cosx_grid),
                                           variant);
}

}  // namespace vibeqc
