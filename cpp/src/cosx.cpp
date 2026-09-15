#include "vibeqc/cosx.hpp"

#include <Eigen/Dense>
#include <libint2/engine.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vibeqc/ao_eval.hpp"
#include "vibeqc/cosx_kernel.hpp"
#include "vibeqc/diagnostics.hpp"
#include "vibeqc/gradient.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/integrals.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/thread_pool.hpp"
#include <libint2/atom.h>

namespace vibeqc {

namespace {

// Threshold below which an AO value is treated as zero at a grid
// point. Two safe screening steps use this:
//
//   1. Whole-point cull: max_μ |χ_μ(r_g)| < tol → skip g entirely.
//   2. Outer-row cull:   |χ_μ(r_g)| < tol → skip the rank-1 K row.
//   3. Shell-pair screen on A: build (s1, s2) only when at least
//      one of the two shells has any |Dχ_ρ| above tol. Inactive
//      shell pairs contribute < tol² × ||A|| to F.
//
// The Dχ-based pair screen is safe because A_{νρ}(r_g) Dχ_ρ vanishes
// when Dχ_ρ does — independent of where χ_ρ has support in space. The
// alternative "drop pairs where χ_s(r_g) is small" is *not* safe
// because A_{νρ} can be sizeable even when χ_ρ(r_g) is small (the
// integral is over r', not r_g).
constexpr double AO_TOL = 1e-7;

}  // namespace

// ---------------------------------------------------------------------------
// COSX variant + grid-level auto-selection (M1 parameter surface).
// ---------------------------------------------------------------------------

int cosx_basis_cardinality_from_name(const std::string& basis_name) {
    std::string n;
    n.reserve(basis_name.size());
    for (char c : basis_name) {
        n.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    const auto has = [&](const char* s) {
        return n.find(s) != std::string::npos;
    };
    // Order matters: 5Z before the others; 6-311 (TZ) before 6-31 (DZ),
    // since "6-31" is a substring of "6-311".
    if (has("5z")) return 5;
    if (has("qz")) return 4;
    if (has("tz") || has("6-311") || has("6311")) return 3;
    if (has("dz") || has("svp") || has("sv(p)") || has("def2-sv") ||
        has("6-31") || has("631")) {
        return 2;
    }
    return 0;  // minimal / unrecognised — let the threshold decide
}

CosxVariant resolve_cosx_variant(CosxVariant variant,
                                 double thresh_cosx,
                                 int basis_cardinality,
                                 int grid_level) {
    if (variant != CosxVariant::AUTO) return variant;
    // AUTO always resolves to the robust overlap-fitted build. The
    // standard (no-fit) K is accurate per grid point on a GridX tier but
    // destabilises the SCF as a default (measured RIJCOSX convergence
    // failures, open-shell especially), so it is an explicit opt-in only.
    (void)thresh_cosx;
    (void)basis_cardinality;
    (void)grid_level;
    return CosxVariant::FITTED;
}

int resolve_cosx_grid_level(int grid_level, int basis_cardinality) {
    if (grid_level >= 1) return std::min(grid_level, 4);
    if (grid_level == 0) return 2;  // neutral default (drivers handle 0)
    // auto (grid_level <= -1): pick by basis cardinality. GridX1 is NOT
    // auto-selected (order-17/110-pt is too coarse for robust open-shell
    // SCF); it is the explicit "quick single-point" tier only.
    if (basis_cardinality >= 4) return 3;  // QZ+ → GridX3
    return 2;                               // DZ / TZ / unknown → GridX2
}

namespace {

// Helmich-Paris, de Souza, Neese & Izsák, J. Chem. Phys. 155, 104109
// (2021), Table I — the 7 "AngularGrid" levels, each a 5-region pruned
// Lebedev grid (inner-core → outer-tail point counts). The angular point
// counts are exact; the per-AG radial count is the 2021-methodology
// approximation (a single per-atom Treutler-Ahlrichs count — the paper's
// per-element radial parameters are ORCA-binary only).
struct AngularGridDef {
    int radial;
    std::array<int, 5> points;
};
constexpr AngularGridDef kAngularGrids[7] = {
    /* AG1 */ {35, {{14, 26, 50, 50, 26}}},
    /* AG2 */ {50, {{14, 26, 50, 110, 50}}},
    /* AG3 */ {65, {{26, 50, 110, 194, 110}}},
    /* AG4 */ {75, {{26, 110, 194, 302, 194}}},
    /* AG5 */ {90, {{26, 194, 302, 434, 302}}},
    /* AG6 */ {110, {{50, 302, 434, 590, 434}}},
    /* AG7 */ {130, {{110, 434, 590, 770, 590}}},
};

// DefGrid stage triples (1-based AngularGrid indices), one per COSX grid
// level 1..4: the (initial, middle, final) SCF-iteration grids of the
// multi-stage progression. Level 2 is ORCA's DefGrid2 default.
constexpr std::array<int, 3> kDefGridStages[4] = {
    {{1, 1, 2}},   // level 1 (DefGrid1)
    {{1, 2, 3}},   // level 2 (DefGrid2, default)
    {{2, 3, 4}},   // level 3 (DefGrid3)
    {{3, 4, 5}},   // level 4 (tight / 5Z)
};

// Representative Lebedev order for a target angular point count — used to
// stamp ``GridOptions::lebedev_order`` for logging / introspection (the
// quadrature actually used is the 5-region ``orca_angular_points``).
int lebedev_order_for_points(int npts) {
    switch (npts) {
        case 14:  return 5;
        case 26:  return 7;
        case 50:  return 11;
        case 110: return 17;
        case 194: return 23;
        case 302: return 29;
        case 434: return 35;
        case 590: return 41;
        case 770: return 47;
        default:  return 29;
    }
}

// Build a GridOptions for one AngularGrid level (1-based, clamped to 1..7).
GridOptions grid_from_angular_grid(int ag) {
    const AngularGridDef& d = kAngularGrids[std::clamp(ag, 1, 7) - 1];
    GridOptions g;
    g.angular = AngularScheme::Lebedev;
    g.angular_pruning = AngularPruning::None;
    g.partition = AtomicPartition::Becke;
    g.n_radial = d.radial;
    g.orca_angular_points = {d.points[0], d.points[1], d.points[2],
                             d.points[3], d.points[4]};
    g.lebedev_order = lebedev_order_for_points(
        *std::max_element(d.points.begin(), d.points.end()));
    return g;
}

}  // namespace

GridOptions cosx_grid_options_for_level(int level) {
    const int L = std::clamp(level, 1, 4);
    // The single-grid view = the FINAL (fine) stage of this level's DefGrid
    // triple — the grid on which the converged exchange matrix is built.
    // The coarse / middle SCF-iteration grids come from
    // ``cosx_grid_stages_for_level``. See grid.hpp ``orca_angular_points``
    // for the 5-region scheme and its documented approximations:
    //   level → final AngularGrid (inner-core → outer-tail point counts)
    //     1 → AG2  14/26/50/110/50    (DefGrid1 final, DZ quick)
    //     2 → AG3  26/50/110/194/110  (DefGrid2 final, default DZ/TZ)
    //     3 → AG4  26/110/194/302/194 (DefGrid3 final, QZ)
    //     4 → AG5  26/194/302/434/302 (tight / 5Z)
    return grid_from_angular_grid(kDefGridStages[L - 1][2]);
}

std::vector<GridOptions> cosx_grid_stages_for_level(int level) {
    const int L = std::clamp(level, 1, 4);
    const auto& stages = kDefGridStages[L - 1];
    std::vector<GridOptions> out;
    out.reserve(3);
    int prev_ag = 0;
    for (int s = 0; s < 3; ++s) {
        const int ag = stages[static_cast<std::size_t>(s)];
        // Collapse consecutive duplicates (DefGrid1 is AG(1,1,2)): a
        // repeated stage only costs a wasted grid + builder rebuild.
        if (ag == prev_ag) continue;
        out.push_back(grid_from_angular_grid(ag));
        prev_ag = ag;
    }
    return out;
}

// Internal helper: build Q from already-evaluated AO values.
// Both ``build_cosx_q`` (public, basis-only entry point) and
// ``compute_cosx_k`` (when no cached Q is supplied) route through
// this helper to avoid duplicating the LDLT + solve logic.
static Eigen::MatrixXd build_cosx_q_from_ao(
    const BasisSet& basis,
    const Eigen::MatrixXd& ao_values,
    const Grid& cosx_grid) {
    const int n_bf  = static_cast<int>(ao_values.cols());
    const int n_pts = static_cast<int>(cosx_grid.points.rows());

    Eigen::MatrixXd S = compute_overlap(basis);
    Eigen::MatrixXd S_grid = Eigen::MatrixXd::Zero(n_bf, n_bf);
    for (int g = 0; g < n_pts; ++g) {
        const double w_g = cosx_grid.weights(g);
        if (w_g == 0.0) continue;
        const Eigen::VectorXd chi_g = ao_values.row(g).transpose();
        S_grid.noalias() += w_g * chi_g * chi_g.transpose();
    }

    // Solve S_grid · Y = S^T (= S since S is symmetric) for Y, so
    // Y = S_grid^{-1} · S, and Q = Y^T. LDLT — S_grid is PSD by
    // construction.
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S_grid);
    if (ldlt.info() != Eigen::Success) {
        // 0×0 sentinel: caller falls back to K_naive without correction.
        return Eigen::MatrixXd();
    }
    return ldlt.solve(S).transpose();
}

Eigen::MatrixXd build_cosx_q(const BasisSet& basis,
                             const Grid& cosx_grid) {
    ensure_libint_initialized();
    const Eigen::MatrixXd ao_values = evaluate_ao(basis, cosx_grid.points);
    return build_cosx_q_from_ao(basis, ao_values, cosx_grid);
}

Eigen::MatrixXd build_cosx_schwarz(const BasisSet& basis) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const auto n_shells = static_cast<int>(shells.size());

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(), shells.max_l(), 0);
    auto engines = make_engine_pool(prototype);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n_shells, n_shells);

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto& engine = engines[
            static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto n2 = shells[s2].size();
            // (s1 s2 | s1 s2): Coulomb-self of the shell pair.
            engine.compute(shells[s1], shells[s2],
                           shells[s1], shells[s2]);
            const double* block = buf[0];
            if (block == nullptr) continue;
            const std::size_t n_elem = n1 * n2 * n1 * n2;
            double mx = 0.0;
            for (std::size_t i = 0; i < n_elem; ++i) {
                mx = std::max(mx, std::abs(block[i]));
            }
            const double q = std::sqrt(mx);
            Q(s1, s2) = q;
            if (s1 != s2) Q(s2, s1) = q;
        }
    }
    return Q;
}

Eigen::MatrixXd compute_cosx_k(const BasisSet& basis,
                               const Eigen::MatrixXd& D,
                               const Grid& cosx_grid,
                               const Eigen::MatrixXd& q_cached,
                               const Eigen::MatrixXd& schwarz_cached,
                               const std::vector<double>& shell_cutoffs,
                               const GridBatches* grid_batches,
                               const BoysTable* boys_table,
                               const PrimitivePairCache* pp_cache,
                               const Eigen::Matrix3d* lattice,
                               const std::vector<LatticeCell>* image_cells,
                               bool apply_overlap_fit) {
    ensure_libint_initialized();

    // Minimum-image distance: precompute the inverse lattice once.
    // Captured by value in the process_point lambda so the per-point
    // hot loop pays only the 3×3 mat-vec multiply, no heap access.
    const bool periodic = (lattice != nullptr);
    Eigen::Matrix3d lat;
    Eigen::Matrix3d inv_lat;
    if (periodic) {
        lat = *lattice;
        inv_lat = lat.inverse();
    }

    // Image-cell exchange summation (M3a). Active only when the caller
    // supplies a non-trivial cell list alongside the lattice; a home-
    // only list (``direct_lattice_cells`` always contains the zero
    // cell, so size() == 1 means no images within the cutoff — e.g. a
    // vacuum-padded molecular-limit box) keeps the single-pass kernel
    // with minimum-image screening and the batched-path performance.
    const bool use_image_cells = periodic && image_cells != nullptr &&
                                 image_cells->size() > 1;

    // The per-batch L2 hierarchy doesn't know about per-cell primary
    // shells yet, so image-cell mode runs the unbatched path — the
    // full-grid AO evaluation below is paid per K-build in that mode
    // (handovers/HANDOVER_RIJCOSX_M3A.md §4.5; separate optimization).
    const GridBatches* batches_use =
        use_image_cells ? nullptr : grid_batches;

    const int n_bf  = static_cast<int>(D.rows());
    const int n_pts = static_cast<int>(cosx_grid.points.rows());

    VIBEQC_DIAG("cosx", vibeqc::DiagLevel::VERBOSE,
        "n_bf=%d  n_pts=%d  batches=%s",
        n_bf, n_pts, (batches_use ? "yes" : "no"));

    // χ_μ(r_g) for every basis function at every grid point. Only built
    // for the unbatched fallback path; the batched path consumes the
    // basis-only chi cache stored on every ``GridBatch::chi_primary``,
    // which was pre-evaluated once at SCF setup by ``build_grid_batches``
    // (cpp/src/grid_batch.cpp).
    Eigen::MatrixXd ao_values;
    if (batches_use == nullptr) {
        ao_values = evaluate_ao(basis, cosx_grid.points);
    }

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const std::size_t n_shells = shells.size();

    // In-tree nuclear-attraction kernel inputs. The COSXJKBuilder
    // holds permanent copies (built once at SCF setup); callers
    // without a cache (Python free-function path) trigger a one-time
    // local build here.
    const BoysTable* boys_use = boys_table;
    const PrimitivePairCache* pp_use = pp_cache;
    BoysTable local_boys;
    PrimitivePairCache local_pp;
    if (boys_use == nullptr) {
        local_boys = build_boys_table(shells.max_l());
        boys_use = &local_boys;
    }
    if (pp_use == nullptr) {
        local_pp = build_primitive_pair_cache(shells);
        pp_use = &local_pp;
    }
    const int max_l_shell = shells.max_l();

    // No engine pool: the in-tree kernel replaces ``engine.compute``
    // for the per-point K build. ``ensure_libint_initialized()`` is
    // still called above so libint's static tables are warm for the
    // ``build_cosx_q`` path inside this same function.

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_bf, n_bf);

    const std::size_t n_threads =
        static_cast<std::size_t>(std::max(1, omp_max_threads()));
    std::vector<Eigen::MatrixXd> K_thread(
        n_threads, Eigen::MatrixXd::Zero(n_bf, n_bf));

    const bool have_schwarz = (schwarz_cached.size() != 0);
    const bool have_cutoffs = !shell_cutoffs.empty();
    // Either screen wants per-shell distance to the grid point, so we
    // compute ``shell_dist`` whenever either is on.
    const bool need_shell_dist = have_schwarz || have_cutoffs;

    // L2 (density-coupled) shell hierarchy — built once per K-build,
    // reused across all batches and grid points. The basis-only
    // shell-pair density envelope
    //   P_shell(s, s') = max |D[μ, ν]|   for μ in s, ν in s'
    // bounds how much shell s' "couples" to shell s via the density,
    // and dictates which shells contribute non-negligibly to Dχ at
    // any point in the batch. A shell s is in batch b's L2 set iff
    // P_shell(s, s') > tol for at least one s' in b.primary_shells
    // (i.e. for some s' that has chi-support somewhere in the batch).
    // Pairs (s1, s2) where BOTH shells fall outside L2_b can be
    // dropped: their contribution to F_g is bounded by ||A|| · tol
    // for each direction, and the chi values they multiply into K
    // are also bounded by the batch-relevant chi range — so the per-
    // grid-point cumulative drop is well within the existing screen
    // tolerance (Burow-Sierka 2011 § 2; Stratmann-Scuseria-Frisch
    // 1996 § 11).
    //
    // Only computed on the batched path; the unbatched path uses the
    // existing per-point dchi-max + Schwarz screens.
    Eigen::MatrixXd P_shell;
    std::vector<std::vector<char>> in_L2_per_batch;   // [batch][shell]
    if (batches_use != nullptr) {
        constexpr double kL2_TOL = 1e-10;  // matches the AO-cutoff tol
        // Per-shell-pair density envelope. Symmetric n_shells × n_shells.
        P_shell.setZero(static_cast<Eigen::Index>(n_shells),
                        static_cast<Eigen::Index>(n_shells));
        for (std::size_t s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1  = shells[s1].size();
            for (std::size_t s2 = 0; s2 <= s1; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2  = shells[s2].size();
                double mx = 0.0;
                for (std::size_t i = 0; i < n1; ++i) {
                    for (std::size_t j = 0; j < n2; ++j) {
                        const double v = std::abs(D(bf1 + i, bf2 + j));
                        if (v > mx) mx = v;
                    }
                }
                P_shell(static_cast<Eigen::Index>(s1),
                        static_cast<Eigen::Index>(s2)) = mx;
                if (s1 != s2) {
                    P_shell(static_cast<Eigen::Index>(s2),
                            static_cast<Eigen::Index>(s1)) = mx;
                }
            }
        }

        // BRIEF #6 negative-result note (2026-05-18). The next obvious
        // refinement is an L3 (Schwarz-coupled) per-batch shell-pair
        // mask:
        //   pair_alive_b(s1, s2) =
        //     schwarz(s1, s2) * max(max_dchi_b(s1), max_dchi_b(s2))
        //         > kL3_TOL
        //   where max_dchi_b(s) = max_{s' in primary} P_shell(s, s')
        // — i.e., precompute the per-point Schwarz screen with a
        // batch-wide upper bound on dχ, drop pairs whose batch-wide
        // bound falls below the AO tolerance, and skip the per-point
        // screen for those pairs. Measured locally (n-hexadecane
        // B3LYP/RIJCOSX/def2-svp, 5-iter probe, M5Max):
        //
        //   L2 only (this commit's baseline): 166 s / 163 s
        //   L2 + L3 pair mask:                176 s / 202 s   (+10–25 %)
        //
        // L3 was a regression. Energy bit-identical to L2 only — the
        // mask is mathematically correct, but the per-batch bound is
        // strictly looser than the existing per-point Schwarz screen
        // (max_batch_dchi ≥ pointwise dchi; dist_factor capped at 1),
        // so the L3 mask can only kill pairs the per-point screen
        // would also kill. The per-point screen is ~10 cycles and
        // already runs lazily — the L3 amortisation saves O(n_pts) of
        // that per dead pair, but the L3 lookup itself fires for every
        // surviving pair × every grid point, and the overhead exceeds
        // the savings unless ≳ 50 % of pairs are dead-across-batch
        // (rarely the case once L2 has already pruned).
        //
        // The headline structural lever is the two-DGEMM-per-batch
        // accumulation (Burow & Sierka 2011 § 2; Neese 2009 § 2 forms)
        // where the inner pair-loop is replaced with two BLAS-3 GEMMs.
        // L3 makes sense as a sizing input to that — bounding the
        // (L2, L3) sub-block where the analytical integrals are
        // computed — but as a standalone screen on top of the
        // existing per-point logic it doesn't pay for itself.
        //
        // Per-batch L2 membership mask. A shell is in L2_b iff its
        // density coupling to *any* shell in batch b's primary set
        // exceeds tolerance.
        const std::size_t n_batches = batches_use->batches.size();
        in_L2_per_batch.assign(n_batches,
                               std::vector<char>(n_shells, 0));
        for (std::size_t bi = 0; bi < n_batches; ++bi) {
            const auto& primary =
                batches_use->batches[bi].primary_shells;
            auto& mask = in_L2_per_batch[bi];
            for (std::size_t s = 0; s < n_shells; ++s) {
                bool any = false;
                for (int s_prim : primary) {
                    if (P_shell(static_cast<Eigen::Index>(s),
                                static_cast<Eigen::Index>(s_prim))
                        > kL2_TOL) {
                        any = true;
                        break;
                    }
                }
                mask[s] = any ? 1 : 0;
            }
        }
    }

    // Thread-local scratch: ``active_shells_local`` (per-grid-point
    // distance-prepruned shell list, only allocated when
    // ``have_cutoffs``), ``dchi_max_local``, and ``shell_dist_local``.
    // Reused across grid points within a thread to avoid the per-
    // point ``std::vector`` heap dance — the dance was the dominant
    // OMP-worker cost before the drop-A_g landing
    // (cpp/src/cosx.cpp commit history). ``schedule(guided)`` is
    // promoted from the parallel-for pragma to the parallel + for
    // pragma pair so the thread-local scratch can outlive a single
    // loop iteration.
    #pragma omp parallel
    {
        std::vector<std::size_t> active_shells_local;
        if (have_cutoffs) active_shells_local.reserve(n_shells);
        std::vector<double> dchi_max_local(n_shells, 0.0);
        std::vector<double> shell_dist_local(n_shells, 0.0);
        // Used only on the batched path — scattering primary-shell
        // chi values back into a full-basis vector so the existing
        // per-point inner kernel can be shared verbatim between the
        // batched and unbatched outer loops. Non-primary entries are
        // exactly zero by construction of the primary set (radial-
        // cutoff bound at AO tolerance ≤ 1e-12).
        Eigen::VectorXd chi_scatter;
        if (batches_use != nullptr) {
            chi_scatter.setZero(n_bf);
        }

        const auto tid =
            static_cast<std::size_t>(omp_thread_index());
        auto& K_local = K_thread[tid];

        // Per-thread workspace for the in-tree nuclear-attraction
        // kernel + the output block. Sized once per OMP worker,
        // reused across millions of pair calls. ``kernel_block`` is
        // sized for the worst-case (l_max, l_max) shell pair; each
        // call only writes the front (n1 × n2) entries.
        CosxKernelWorkspace ws;
        ws.reserve(max_l_shell, max_l_shell);
        const int max_n_per_shell =
            (max_l_shell + 1) * (max_l_shell + 2) / 2;
        std::vector<double> kernel_block(
            static_cast<std::size_t>(max_n_per_shell) * max_n_per_shell);

        // Per-grid-point inner kernel — used by both the batched and
        // unbatched outer loops. Captures the per-thread scratch +
        // engine + accumulator by reference. ``rg`` are the point's
        // Cartesian coordinates; ``w_g`` is the integration weight;
        // ``chi_g`` is the full-basis AO vector at the point (either
        // ``ao_values.row(g).transpose()`` from the unbatched path, or
        // the primary-scatter from the batched path).
        auto process_point = [&](double rg_x, double rg_y, double rg_z,
                                 double w_g,
                                 const Eigen::Ref<const Eigen::VectorXd>& chi_g,
                                 const std::vector<char>* l2_mask) {
            if (w_g == 0.0) return;
            const double chi_max = chi_g.cwiseAbs().maxCoeff();
            if (chi_max < AO_TOL) return;

            // Density-weighted basis at r_g — needed both as a screening
            // proxy (which shells of A's columns are non-negligible) and
            // as the contracting vector in F = A · Dχ.
            const Eigen::VectorXd Dchi = D * chi_g;

            // Image-cell exchange summation (M3a). The vibe-qc periodic
            // extension of the molecular chain-of-spheres form (Neese,
            // Wennmohs, Hansen & Becker, Chem. Phys. 356, 98 (2009),
            // doi:10.1016/j.chemphys.2008.10.036, §2):
            //
            //   F_ν(r_g) = Σ_R Σ_σ A_{νσ}(r_g − R) · (D · χ(r_g))_σ
            //   K_{μν}  += w_g · χ_μ(r_g) · F_ν(r_g)
            //
            // which accumulates Σ_R Σ_{λσ} D_{λσ} (μ_0 λ_0 | ν_R σ_R) —
            // the exchange coupling of the home-cell bra pair density
            // to every image-cell replica of the ket pair density. The
            // conditionally-convergent lattice sum is regularized by
            // direct truncation: the caller supplies the finite cell
            // list (``direct_lattice_cells(system, cutoff_bohr)``),
            // consistent with the codebase's DIRECT_TRUNCATED Coulomb
            // treatment. R = 0 (the home cell, always first in the
            // list) reproduces the molecular kernel; the loop starts
            // F_g at zero and accumulates over all cells, so there is
            // no special-casing and no double counting
            // (handovers/HANDOVER_RIJCOSX_M3A.md §3).
            //
            // Non-image mode (n_cells = 1, zero shift) is the exact
            // single-pass flow this loop generalizes.
            Eigen::VectorXd F_g = Eigen::VectorXd::Zero(n_bf);
            bool any_cell_active = false;
            const std::size_t n_cells =
                use_image_cells ? image_cells->size() : 1;
            for (std::size_t ci = 0; ci < n_cells; ++ci) {
                // Pseudo-nucleus position for this cell: C_R = r_g − R.
                // Evaluating the home-cell A-operator at r_g − R equals
                // evaluating the cell-R replica's A-operator at r_g.
                double cg_x = rg_x;
                double cg_y = rg_y;
                double cg_z = rg_z;
                if (use_image_cells) {
                    const auto& r_cell = (*image_cells)[ci].r_cart;
                    cg_x -= r_cell[0];
                    cg_y -= r_cell[1];
                    cg_z -= r_cell[2];
                }

                // Per-shell distance to the (cell-shifted) grid point.
                // Needed by both the Schwarz distance-prescreen and the
                // shell-cutoff active-shells filter (whenever either is on).
                //
                // Periodic systems without an image-cell list: each shell-
                // origin → grid-point displacement is folded into the first
                // unit cell via the minimum-image convention — fractional
                // coordinates rounded to nearest integer, then back to
                // Cartesian. This ensures that a shell near the +x boundary
                // correctly registers as "close" to a grid point near the
                // −x boundary of the same tight cell. Vacuum directions
                // (dim < 3) carry large padding constants so the round
                // collapses to identity.
                //
                // Image-cell mode: plain Euclidean distance from C_R — the
                // cell loop covers the images explicitly, so no minimum-
                // image folding (it would double-count the closest image).
                // Distant cells leave every shell beyond its radial cutoff
                // and are skipped wholesale by the active-shell pre-prune.
                if (need_shell_dist) {
                    if (periodic && !use_image_cells) {
                        for (std::size_t s = 0; s < n_shells; ++s) {
                            const auto& O = shells[s].O;
                            double dx = O[0] - cg_x;
                            double dy = O[1] - cg_y;
                            double dz = O[2] - cg_z;
                            // Cartesian → fractional → minimum-image → Cartesian.
                            double fx = inv_lat(0,0)*dx + inv_lat(0,1)*dy + inv_lat(0,2)*dz;
                            double fy = inv_lat(1,0)*dx + inv_lat(1,1)*dy + inv_lat(1,2)*dz;
                            double fz = inv_lat(2,0)*dx + inv_lat(2,1)*dy + inv_lat(2,2)*dz;
                            fx -= std::round(fx);
                            fy -= std::round(fy);
                            fz -= std::round(fz);
                            dx = lat(0,0)*fx + lat(0,1)*fy + lat(0,2)*fz;
                            dy = lat(1,0)*fx + lat(1,1)*fy + lat(1,2)*fz;
                            dz = lat(2,0)*fx + lat(2,1)*fy + lat(2,2)*fz;
                            shell_dist_local[s] =
                                std::sqrt(dx*dx + dy*dy + dz*dz);
                        }
                    } else {
                        for (std::size_t s = 0; s < n_shells; ++s) {
                            const auto& O = shells[s].O;
                            const double dx = O[0] - cg_x;
                            const double dy = O[1] - cg_y;
                            const double dz = O[2] - cg_z;
                            shell_dist_local[s] =
                                std::sqrt(dx*dx + dy*dy + dz*dz);
                        }
                    }
                }

                // Pre-prune to active shells via the radial-cutoff bound:
                // any shell whose origin is beyond its tabulated extent
                // contributes |χ_μ(r_g)| < AO_TOL by construction. For
                // extended systems this drops the pair-loop iteration
                // count from n_shells² to n_active² — typically a 5–50×
                // reduction at the n-hexadecane-class anchor. In image-
                // cell mode the same bound screens whole cells: a cell
                // whose C_R is beyond every shell's cutoff contributes
                // nothing and is skipped before the pair loop.
                //
                // When ``shell_cutoffs`` is empty, ``active_shells`` is
                // implicitly 0..n_shells (every shell stays in the loop)
                // and the existing dchi + Schwarz screens handle culling
                // alone.
                std::size_t n_active = n_shells;
                const std::size_t* active_ptr = nullptr;  // null = identity
                if (have_cutoffs) {
                    active_shells_local.clear();
                    for (std::size_t s = 0; s < n_shells; ++s) {
                        if (shell_dist_local[s] < shell_cutoffs[s]) {
                            active_shells_local.push_back(s);
                        }
                    }
                    n_active = active_shells_local.size();
                    if (n_active == 0) continue;  // no shell in range of C_R
                    active_ptr = active_shells_local.data();
                }
                any_cell_active = true;
                // Helper to map loop-index → shell-index (identity when
                // active_ptr is null).
                auto shell_idx = [active_ptr](std::size_t i) -> std::size_t {
                    return active_ptr ? active_ptr[i] : i;
                };

                // Per-shell max |Dχ| over the BFs in the shell. Used as a
                // baseline "shell-active" flag (m > AO_TOL) and as the
                // multiplier in the Schwarz × density-amplitude pair
                // screen. Only computed for shells in the active list —
                // inactive shells never reach the pair-loop comparison.
                // (Dχ is cell-invariant; recomputing per cell costs O(n_bf)
                // against the pair loop's O(n_active² · kernel) — noise.)
                for (std::size_t ai = 0; ai < n_active; ++ai) {
                    const std::size_t s = shell_idx(ai);
                    const auto bf0 = shell2bf[s];
                    const auto ns  = shells[s].size();
                    double m = 0.0;
                    for (std::size_t i = 0; i < ns; ++i) {
                        m = std::max(m, std::abs(Dchi(bf0 + i)));
                    }
                    dchi_max_local[s] = m;
                }

                // The in-tree nuclear-attraction kernel takes the
                // pseudo-nucleus position directly; there is no per-point
                // engine setup to do.
                const std::array<double, 3> C_g = {cg_x, cg_y, cg_z};

                // Build F_ν += (A(C_R) · D · χ)_ν with shell-pair screening,
                // contracting each (s1, s2) integral block directly against
                // Dχ on the fly — **without materialising the dense
                // n_bf × n_bf matrix A**. Outer pair loop iterates over the
                // pre-pruned active-shells list (when ``shell_cutoffs`` is
                // supplied) or all shells (when not).
                //
                // Screening tiers, applied in order:
                //   1. Radial-cutoff active-shell pre-prune (outer loop).
                //   2. Baseline dchi screen: drop pair if both shells have
                //      max|Dχ| below AO_TOL.
                //   3. Schwarz × distance prescreen (when supplied):
                //      schwarz(s1,s2) · max(|Dχ|_s1, |Dχ|_s2) · 1/d_min
                //      < AO_TOL drops the libint engine call.
                for (std::size_t a1 = 0; a1 < n_active; ++a1) {
                    const std::size_t s1 = shell_idx(a1);
                    const auto bf1 = shell2bf[s1];
                    const auto n1 = shells[s1].size();
                    for (std::size_t a2 = 0; a2 <= a1; ++a2) {
                        const std::size_t s2 = shell_idx(a2);
                        // L2 (density-coupled) shell screen — only on the
                        // batched path. The pair (s1, s2) contributes to
                        // F_g via two directions:
                        //   F_g[s1] += A · Dχ[s2]   (bounded by dchi[s2])
                        //   F_g[s2] += Aᵀ · Dχ[s1]  (bounded by dchi[s1])
                        // If BOTH shells are outside the batch's L2 set,
                        // their dχ across all batch points is bounded by
                        // the L2 tolerance, so both directions contribute
                        // below threshold and the pair is safely droppable
                        // (Burow-Sierka 2011 § 2). When only one shell is
                        // in L2_b, the in-direction contribution is
                        // non-negligible and the pair survives — the
                        // existing per-point dchi screen still drops it
                        // dynamically if that point's dchi happens to be
                        // small there.
                        if (l2_mask != nullptr
                            && (*l2_mask)[s1] == 0
                            && (*l2_mask)[s2] == 0) {
                            continue;
                        }
                        const double dchi_pair_max =
                            std::max(dchi_max_local[s1], dchi_max_local[s2]);
                        if (dchi_pair_max < AO_TOL) continue;
                        if (have_schwarz) {
                            // Distance prescreen: A_{μν}(r_g) decays as
                            // ~1/d when the pair's AO product is far from
                            // r_g. Use min(d_s1, d_s2) (closer-shell wins)
                            // and cap the factor at 1 to avoid amplifying
                            // pairs near r_g (where the integral is finite
                            // by integration but our 1/d proxy diverges).
                            const double d_min = std::min(
                                shell_dist_local[s1], shell_dist_local[s2]);
                            const double dist_factor =
                                1.0 / std::max(1.0, d_min);
                            const double bound =
                                schwarz_cached(
                                    static_cast<Eigen::Index>(s1),
                                    static_cast<Eigen::Index>(s2))
                                * dchi_pair_max * dist_factor;
                            if (bound < AO_TOL) continue;
                        }
                        const auto bf2 = shell2bf[s2];
                        const auto n2 = shells[s2].size();
                        // In-tree kernel call — replaces the per-point
                        // libint ``Operator::nuclear`` engine evaluation.
                        // The output buffer is the per-thread
                        // ``kernel_block`` (row-major, sized for the
                        // worst-case pair); the kernel writes the front
                        // (n1 × n2) entries.
                        cosx_nuclear_pair_into(
                            shells[s1], shells[s2],
                            pp_use->pairs[s1 * n_shells + s2],
                            C_g, *boys_use, ws,
                            kernel_block.data());
                        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
                                                       Eigen::Dynamic,
                                                       Eigen::RowMajor>>
                            buf_mat(kernel_block.data(),
                                    static_cast<Eigen::Index>(n1),
                                    static_cast<Eigen::Index>(n2));
                        // Block-sparse F_g contribution: see commit 899e8fb
                        // (drop-A_g) for the full algebra.
                        F_g.segment(bf1, n1).noalias() +=
                            buf_mat * Dchi.segment(bf2, n2);
                        if (s1 != s2) {
                            F_g.segment(bf2, n2).noalias() +=
                                buf_mat.transpose() * Dchi.segment(bf1, n1);
                        }
                    }
                }
            }  // end image-cell loop
            // No cell had a shell in range → F_g is exactly zero;
            // skip the K accumulation (matches the pre-M3a early
            // return on an empty active-shell list).
            if (!any_cell_active) return;

            // Accumulate K += w_g · χ_g · F_g^T.
            //
            // Two strategies, chosen per point based on the number of
            // active (non-negligible) χ rows:
            //
            //   Sparse (default, < 30 % of rows active):
            //     Row-screened rank-1 update — beats GEMM by ~41 %
            //     (commit 49a72ca) because χ is sparse-by-geometry
            //     on molecular grids.
            //
            //   Dense (≥ 30 % of rows active, e.g. near nuclei):
            //     Single GEMM K += w_g · χ_g · F_g^T — the BLAS
            //     kernel throughput beats the row loop when most
            //     rows survive screening.
            //
            // The 30 % threshold is calibrated on the n-hexadecane
            // B3LYP/RIJCOSX/def2-svp 5-iter probe; it is well above
            // the typical 5−20 % active fraction so the sparse path
            // dominates, but dense nuclear regions get the GEMM
            // speedup.
            int n_active_rows = 0;
            for (int mu = 0; mu < n_bf; ++mu)
                if (std::abs(chi_g(mu)) >= AO_TOL) ++n_active_rows;

            if (n_active_rows * 3 < n_bf) {  // < 33 % active → sparse
                for (int mu = 0; mu < n_bf; ++mu) {
                    const double cmu = chi_g(mu);
                    if (std::abs(cmu) < AO_TOL) continue;
                    K_local.row(mu).noalias() +=
                        (w_g * cmu) * F_g.transpose();
                }
            } else {
                K_local.noalias() +=
                    (w_g * chi_g) * F_g.transpose();
            }
        };  // end process_point lambda

        // Dispatch on whether the caller supplied a batched chi cache.
        // Unbatched path (default for the Python ``compute_cosx_k`` free
        // function): iterate over grid points directly, read chi from
        // the pre-built ``ao_values`` matrix. Bit-identical to the
        // pre-batch implementation (commit 50e9b6a) — the lambda
        // extraction is structure-only.
        //
        // Batched path (``COSXJKBuilder`` calls this with a cached
        // ``GridBatches``): iterate over batches; per batch, scatter
        // each point's primary-shell chi values back into a full-basis
        // vector and call into the same per-point kernel. The batched
        // outer loop is the substrate for the upcoming two-DGEMM K-
        // accumulation refactor (Neese 2009 § 2 forms; Burow-Sierka
        // 2011 § 2 hierarchical layering).
        //
        // Image-cell mode always lands on the unbatched branch —
        // ``batches_use`` was nulled at the top of the function.
        if (batches_use == nullptr) {
            #pragma omp for schedule(guided)
            for (int g = 0; g < n_pts; ++g) {
                const double w_g = cosx_grid.weights(g);
                const Eigen::VectorXd chi_g =
                    ao_values.row(g).transpose();
                process_point(cosx_grid.points(g, 0),
                              cosx_grid.points(g, 1),
                              cosx_grid.points(g, 2),
                              w_g, chi_g,
                              /*l2_mask=*/nullptr);
            }
        } else {
            // Per-point OMP parallelism (matching the unbatched
            // path's granularity for fair load distribution); chi
            // values come from the per-batch cache. Batches are
            // uniformly-sized (``batch_size_hint`` each except
            // possibly the last), so the (g → batch_index, gp)
            // mapping is a constant-time division. Per-batch
            // parallelism with ~70 outer iterations × 10 threads
            // hits load imbalance from the wide variance in n_primary
            // across batches.
            const int n_batches =
                static_cast<int>(batches_use->batches.size());
            const int batch_size_hint = batches_use->batch_size_hint;
            #pragma omp for schedule(guided)
            for (int g = 0; g < n_pts; ++g) {
                int bi = g / batch_size_hint;
                if (bi >= n_batches) bi = n_batches - 1;
                const GridBatch& batch = batches_use->batches[bi];
                const int gp = g - batch.start_index;
                const int n_primary =
                    static_cast<int>(batch.primary_bfs.size());

                chi_scatter.setZero();
                for (int c = 0; c < n_primary; ++c) {
                    chi_scatter(batch.primary_bfs[c]) =
                        batch.chi_primary(gp, c);
                }
                process_point(batch.points(gp, 0),
                              batch.points(gp, 1),
                              batch.points(gp, 2),
                              batch.weights(gp),
                              chi_scatter,
                              /*l2_mask=*/&in_L2_per_batch[bi]);
            }
        }
    }  // omp parallel

    for (const auto& Kt : K_thread) K += Kt;

    // Symmetrise — Neese 2009 §2.3.
    Eigen::MatrixXd K_naive = 0.5 * (K + K.transpose());

    // CosxVariant::STANDARD — no bra-side overlap fit. Return the
    // symmetrised seminumerical K directly (the one-centre analytic
    // correction is applied separately by the JK builder).
    if (!apply_overlap_fit) {
        return K_naive;
    }

    // ---- Overlap-fit Q-junction (Neese 2009 §2.4) ----------------------
    // Q ≡ S · S_grid^{-1} absorbs the bra-side quadrature error on
    // K_naive into the corrected K = Q · K_naive · Qᵀ. Q depends only
    // on the AO basis + the cosx grid — both invariant across the SCF.
    // The SCF driver pre-computes Q once via ``build_cosx_q`` and
    // passes it in as ``q_cached`` on every iteration; when called
    // standalone (e.g. from Python with no cache wired up) Q is
    // rebuilt here. The 0×0 sentinel from ``build_cosx_q_from_ao``
    // signals an LDLT failure → fall back to the uncorrected K.
    Eigen::MatrixXd Q_local;
    const Eigen::MatrixXd* Q_ptr = &q_cached;
    if (q_cached.size() == 0) {
        Q_local = build_cosx_q_from_ao(basis, ao_values, cosx_grid);
        Q_ptr = &Q_local;
    }
    if (Q_ptr->size() == 0) {
        return K_naive;
    }
    return (*Q_ptr) * K_naive * Q_ptr->transpose();
}


// ---------------------------------------------------------------------------
// Analytic gradient of E_K = −(α_hf/2) tr(D · K_cosx) — frozen weights /
// frozen grid / frozen Q. See cosx.hpp for the derivation.
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_cosx_k_gradient_contribution(
    const Molecule& mol,
    const BasisSet& basis,
    const Eigen::MatrixXd& D,
    const Grid& cosx_grid,
    double alpha_hf,
    const Eigen::MatrixXd& q_cached,
    bool apply_overlap_fit) {
    ensure_libint_initialized();

    const int n_bf  = static_cast<int>(D.rows());
    const int n_pts = static_cast<int>(cosx_grid.points.rows());
    const int n_atoms = static_cast<int>(mol.atoms().size());

    if (alpha_hf == 0.0) {
        return Eigen::MatrixXd::Zero(n_atoms, 3);
    }

    // χ and ∂χ/∂r at every grid point. ``ao.gradients[c]`` is the
    // Cartesian r-gradient; the atom-resolved derivative ∂χ_μ/∂R_{A,c}
    // equals −∂χ_μ/∂r_c when atom_of(μ) == A, else zero (translation
    // invariance for AOs).
    AOValues ao = evaluate_ao_with_gradient(basis, cosx_grid.points);
    const Eigen::MatrixXd& chi_all = ao.values;
    const auto& dchi = ao.gradients;  // [c] = (n_pts, n_bf)

    // Build Q or reuse the caller's cached copy.  With frozen Q,
    // dK = Q·dK_naive·Qᵀ and tr(D·dK) = tr(D'·dK_naive) with
    // D' = Qᵀ D Q.  When ``q_cached`` is supplied (non-empty), skip
    // the redundant LDLT + solve.  When ``build_cosx_q_from_ao``
    // returns 0×0 (LDLT failure on S_grid), fall through with D' = D.
    // CosxVariant::STANDARD (apply_overlap_fit == false): no Q, D' = D.
    Eigen::MatrixXd Q_local;
    const Eigen::MatrixXd* Q_ptr = &q_cached;
    if (apply_overlap_fit && q_cached.size() == 0) {
        Q_local = build_cosx_q_from_ao(basis, chi_all, cosx_grid);
        Q_ptr = &Q_local;
    }
    const bool q_ok = apply_overlap_fit && (Q_ptr->size() != 0);
    Eigen::MatrixXd D_eff = q_ok ? (Q_ptr->transpose() * D * (*Q_ptr)).eval() : D;

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const std::size_t n_shells = shells.size();

    // shell → atom index, and (per-bf) the same.
    std::vector<long> s2a;
    {
        std::vector<libint2::Atom> libint_atoms;
        libint_atoms.reserve(mol.atoms().size());
        for (const auto& a : mol.atoms()) {
            libint_atoms.push_back({a.Z, a.xyz[0], a.xyz[1], a.xyz[2]});
        }
        s2a = shells.shell2atom(libint_atoms);
    }
    // BF → atom map (handy for the χ-gradient term contraction).
    std::vector<long> bf2atom(n_bf, 0);
    for (std::size_t s = 0; s < n_shells; ++s) {
        const auto bf0 = shell2bf[s];
        const auto ns  = shells[s].size();
        for (std::size_t i = 0; i < ns; ++i) bf2atom[bf0 + i] = s2a[s];
    }

    // Two engine pools — deriv-0 for A, deriv-1 for dA. libint exposes
    // the value-only integral via a separate engine; running a single
    // deriv-1 engine doesn't return the underlying integral.
    libint2::Engine proto_d0(libint2::Operator::nuclear,
                             shells.max_nprim(), shells.max_l(), 0);
    libint2::Engine proto_d1(libint2::Operator::nuclear,
                             shells.max_nprim(), shells.max_l(), 1);
    auto engines_d0 = make_engine_pool(proto_d0);
    auto engines_d1 = make_engine_pool(proto_d1);

    const std::size_t n_threads = engines_d1.size();
    std::vector<Eigen::MatrixXd> grad_thread(
        n_threads, Eigen::MatrixXd::Zero(n_atoms, 3));
    // K_naive[D] accumulation for the overlap-fit (Q) response term
    // below — only needed on the fitted path.
    std::vector<Eigen::MatrixXd> knaive_thread(
        q_ok ? n_threads : 0, Eigen::MatrixXd::Zero(n_bf, n_bf));

    #pragma omp parallel for schedule(dynamic, 32)
    for (int g = 0; g < n_pts; ++g) {
        const double w_g = cosx_grid.weights(g);
        if (w_g == 0.0) continue;

        const Eigen::VectorXd chi_g = chi_all.row(g).transpose();
        if (chi_g.cwiseAbs().maxCoeff() < AO_TOL) continue;

        // Mixed-density bilinear form (2026-07-29 parity fix): the
        // overlap-fitted K-energy is the MIXED contraction
        //   tr(D' K_naive[D]) = Σ_g w_g (D'χ)ᵀ A (Dχ),  D' = Qᵀ D Q
        // (Izsák & Neese, J. Chem. Phys. 135, 144105 (2011), Eqs. 6/12:
        // the fit corrects the bra-side numerical index only). The
        // pre-fix code contracted D' on BOTH sides — i.e. it
        // differentiated a different functional than the SCF energy,
        // which was measured at up to 6.3e-2 Ha/bohr per component on
        // glycine/def2-TZVP heavy atoms (EVIDENCE, 2026-07-29).
        const Eigen::VectorXd Dchi  = D * chi_g;       // Dχ  (raw density)
        const Eigen::VectorXd Dpchi = D_eff * chi_g;   // D'χ (fitted side)

        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine_d0 = engines_d0[tid];
        auto& engine_d1 = engines_d1[tid];
        const auto& buf_d0 = engine_d0.results();
        const auto& buf_d1 = engine_d1.results();

        // Pseudo-nucleus at r_g with q = −1.
        const std::vector<std::pair<double, std::array<double, 3>>>
            charges_for_point{
                {-1.0,
                 {cosx_grid.points(g, 0),
                  cosx_grid.points(g, 1),
                  cosx_grid.points(g, 2)}}};
        engine_d0.set_params(charges_for_point);
        engine_d1.set_params(charges_for_point);

        // ---- Build A(r_g); F = A·Dχ and F' = A·D'χ via deriv-0 sweep ----
        Eigen::MatrixXd A_g = Eigen::MatrixXd::Zero(n_bf, n_bf);
        for (std::size_t s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            for (std::size_t s2 = 0; s2 <= s1; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                engine_d0.compute(shells[s1], shells[s2]);
                const double* block = buf_d0[0];
                if (block == nullptr) continue;
                Eigen::Map<const Eigen::Matrix<double,
                    Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    bm(block, n1, n2);
                A_g.block(bf1, bf2, n1, n2) = bm;
                if (s1 != s2) {
                    A_g.block(bf2, bf1, n2, n1) = bm.transpose();
                }
            }
        }
        const Eigen::VectorXd F_g  = A_g * Dchi;    // A·Dχ
        const Eigen::VectorXd Fp_g = A_g * Dpchi;   // A·D'χ

        // K_naive[D]_μν = Σ_g w_g χ_μ(g) (A·Dχ)_ν — needed once, for
        // the frozen-grid Q-response term after the loop.
        if (q_ok) {
            knaive_thread[tid].noalias() += w_g * chi_g * F_g.transpose();
        }

        // ---- Term 1: + ½·w_g · (D'F + DF')_μ · dχ_c[g, μ], μ on atom A ----
        // From d/dR of Σ w (D'χ)ᵀA(Dχ) through both χ factors:
        //   (D'∂χ)ᵀA(Dχ) + (D'χ)ᵀA(D∂χ) = ∂χ·(D'F + DF').
        // The translation-invariance flip ∂χ_μ/∂R_{A,c} = −dchi[c] and
        // the closed-shell prefactor combine into the +(α_hf/2)
        // multiplier applied at the end of the function (the ½ here
        // restores the α/4 energy prefactor for the two-sided sum).
        const Eigen::VectorXd T1v =
            0.5 * (D_eff * F_g + D * Fp_g);
        for (int c = 0; c < 3; ++c) {
            for (int mu = 0; mu < n_bf; ++mu) {
                grad_thread[tid](bf2atom[mu], c)
                    += w_g * T1v(mu) * dchi[c](g, mu);
            }
        }

        // ---- Term 2: −½ · w_g · D'χᵀ · ∂A_c · Dχ ----
        // libint deriv-1 nuclear-attraction returns 9 buffers per shell
        // pair: (3 dirs) × (R_bra, R_ket, R_nuc). With one pseudo-nucleus
        // ncenters = 3, so:
        //     buf_d1[0..2]:  ∂/∂R_a  (bra-shell atom)
        //     buf_d1[3..5]:  ∂/∂R_b  (ket-shell atom)
        //     buf_d1[6..8]:  ∂/∂R_n  (grid point) — frozen-grid: SKIP.
        //         NOTE (2026-07-29): assigning these to the point's
        //         parent atom WITHOUT the Becke-weight derivatives is
        //         invalid — the per-atom point-motion term is an O(1)
        //         Becke-cell boundary flux that only cancels against
        //         the weight-derivative term (measured: 0.94 Ha/bohr
        //         error on H2O/def2-svp when tried). Point motion and
        //         weight derivatives must land TOGETHER or not at all.
        //
        // The full-matrix contraction Σ_{μν} (D'χ)_μ ∂A_μν (Dχ)_ν over
        // the lower-triangular shell loop needs BOTH block orientations
        // when s1 ≠ s2 — with the mixed vectors they are no longer
        // equal:  (s1,s2): u1ᵀB v2  and  (s2,s1): u2ᵀBᵀv1 = v1ᵀB u2,
        // u = D'χ, v = Dχ.
        for (std::size_t s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            const long atom1 = s2a[s1];
            for (std::size_t s2 = 0; s2 <= s1; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                const long atom2 = s2a[s2];
                engine_d1.compute(shells[s1], shells[s2]);

                const Eigen::VectorXd u1 = Dpchi.segment(bf1, n1);
                const Eigen::VectorXd u2 = Dpchi.segment(bf2, n2);
                const Eigen::VectorXd v1 = Dchi.segment(bf1, n1);
                const Eigen::VectorXd v2 = Dchi.segment(bf2, n2);
                const bool off_diag = (s1 != s2);

                for (int c = 0; c < 3; ++c) {
                    // ∂/∂R_a contribution → atom1.  The minus sign
                    // converts the +D'χᵀ ∂A Dχ piece into the gradient
                    // sign (E_K = −α/4 tr(D' K_naive[D]) → +α/2 × (T1 − T2)).
                    if (const double* block = buf_d1[c]) {
                        Eigen::Map<const Eigen::Matrix<double,
                            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            bm(block, n1, n2);
                        double scalar = u1.dot(bm * v2);
                        if (off_diag) scalar += v1.dot(bm * u2);
                        grad_thread[tid](atom1, c) -= 0.5 * w_g * scalar;
                    }
                    // ∂/∂R_b contribution → atom2.
                    if (const double* block = buf_d1[3 + c]) {
                        Eigen::Map<const Eigen::Matrix<double,
                            Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                            bm(block, n1, n2);
                        double scalar = u1.dot(bm * v2);
                        if (off_diag) scalar += v1.dot(bm * u2);
                        grad_thread[tid](atom2, c) -= 0.5 * w_g * scalar;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);
    for (const auto& gt : grad_thread) grad += gt;

    // ---- Overlap-fit (Q) response, frozen grid (2026-07-29 parity fix) ----
    //
    // The fit matrix Q = S·S_grid⁻¹ depends on the nuclear positions
    // through BOTH the analytic overlap S and the grid-sampled overlap
    // S_grid = Σ_g w_g χ(g)χ(g)ᵀ. The reference gradient carries this
    // derivative (Izsák & Neese, J. Chem. Phys. 135, 144105 (2011),
    // Sec. II.D Eqs. 19-22 — ORCA's SCX/SCY/SCZ derivative-fit data);
    // freezing Q was measured at up to 2.8e-3 Ha/bohr per component on
    // H2O/def2-tzvp (GridX tier 2) and is a driver of the glycine/
    // def2-TZVP gradient parity defect (EVIDENCE 2026-07-29).
    //
    // With E_K = −(α/4)·tr(QᵀDQ·K_naive[D]) and dQ = (dS − Q·dS_grid)·S_grid⁻¹:
    //   dE|_dQ = −(α/2)·tr(dQᵀ·M),  M = D·Q·K_naive[D]
    //          = −(α/2)·[ tr(dS·N1) − tr(dS_grid·N2) ],  N1 = M·S_grid⁻¹,
    //            N2 = Qᵀ·N1
    // Contractions (dS, dS_grid symmetric → symmetrise N1, N2):
    //   tr(dS·N1)(A,c)      = −overlap_gradient_contribution(N1s)(A,c)
    //   tr(dS_grid·N2)(A,c) = −2·T(A,c),
    //     T(A,c) = Σ_g w_g Σ_{μ∈A} dχ_c(g,μ)·(N2s·χ_g)_μ
    // so the pre-(α/2) accumulator gains  OGC(N1s) − 2·T.
    // The grid-point-motion and Becke-weight derivatives remain
    // neglected (frozen grid), matching the ORCA default (GradType 0);
    // measured ≤ 2.8e-4 Ha/bohr on H2O/def2-tzvp.
    if (q_ok) {
        Eigen::MatrixXd K_naive = Eigen::MatrixXd::Zero(n_bf, n_bf);
        for (const auto& kt : knaive_thread) K_naive += kt;

        // S_grid = χᵀ·diag(w)·χ in one GEMM.
        const Eigen::MatrixXd wchi =
            cosx_grid.weights.asDiagonal() * chi_all;
        Eigen::MatrixXd S_grid = chi_all.transpose() * wchi;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(S_grid);
        if (ldlt.info() == Eigen::Success) {
            const Eigen::MatrixXd& Q = *Q_ptr;
            const Eigen::MatrixXd M = D * Q * K_naive;
            const Eigen::MatrixXd N1 =
                ldlt.solve(M.transpose()).transpose();   // M·S_grid⁻¹
            const Eigen::MatrixXd N1s = 0.5 * (N1 + N1.transpose());
            const Eigen::MatrixXd N2 = Q.transpose() * N1;
            const Eigen::MatrixXd N2s = 0.5 * (N2 + N2.transpose());

            // dS piece via the shared Pulay kernel
            // (overlap_gradient_contribution returns −tr(W·dS/dR)).
            grad += overlap_gradient_contribution(basis, mol, N1s);

            // dS_grid piece: T(A,c) accumulated at GEMM level.
            const Eigen::MatrixXd Y = wchi * N2s;      // (n_pts, n_bf), w folded
            for (int c = 0; c < 3; ++c) {
                const Eigen::VectorXd t =
                    (dchi[c].cwiseProduct(Y)).colwise().sum().transpose();
                for (int mu = 0; mu < n_bf; ++mu) {
                    grad(bf2atom[mu], c) -= 2.0 * t(mu);
                }
            }
        }
        // LDLT failure: fall through with the frozen-Q gradient — the
        // same degrade path the K build takes for a singular S_grid.
    }

    // dE_K/dR = (α_hf / 2) × (T1_acc − T2_acc + Q-response) for the
    // closed-shell K-energy convention E_K = −(α_hf/4)·tr(D'·K_naive[D]).
    return 0.5 * alpha_hf * grad;
}

}  // namespace vibeqc
