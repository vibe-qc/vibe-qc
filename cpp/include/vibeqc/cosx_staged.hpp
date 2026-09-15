// Multi-stage COSX grid progression driver (P2, Helmich-Paris et al. 2021
// "DefGrid" stage scheme). The SCF steps through a sequence of
// coarse → middle → fine COSX grids (``cosx_grid_stages_for_level``)
// following the published scheme of Helmich-Paris, de Souza, Neese &
// Izsák, J. Chem. Phys. 155, 104109 (2021), doi:10.1063/5.0058766,
// Sec. IV.A:
//
//   "It starts with a minimal grid, then is switched to a middle grid
//    when close to convergence, and after converging, the exchange
//    energy is recomputed using the final larger grid."
//
// Concretely: every stage before the second-to-last converges to a LOOSE
// tolerance on its coarse grid; the SECOND-TO-LAST stage converges to the
// caller's tolerance; and the FINAL (largest) grid is used for exactly ONE
// Fock rebuild + energy recompute at the converged density — the SCF never
// iterates on the final grid. (BUG87-A: an earlier revision ran a full SCF
// segment to the user's tolerance on the final grid; because the inter-grid
// exchange difference re-excites the DIIS error vector to the 1e-2-Frobenius
// class, that re-converged the SCF from scratch at the most expensive
// per-iteration price — 11 of 14.8 wall-normalized cost units on
// glycine/def2-svp were fine-grid K builds that the published scheme
// replaces with one.)
//
// The progression is orchestrated at the driver level: each stage reuses
// the tested ``run_*_scf_with_jk`` loop unchanged, called once per stage on
// the SAME COSX builder with the carried-forward density. Between stages
// only the COSX grid is swapped (``JKBuilder::set_cosx_grid``), which
// rebuilds just the grid-dependent caches (overlap-fit Q-junction + per-
// batch AO cache) and keeps the basis/aux-only RI-J B-tensor — so the
// progression does NOT re-pay the expensive RI-J build per stage.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "grid.hpp"
#include "jk_builder.hpp"
#include "molecule.hpp"

namespace vibeqc {

// Loose-convergence policy for the coarse stages that precede the
// converging (second-to-last) stage. A stage uses the looser of (this
// floor, the user's value) so a user who asks for an even looser SCF is
// honoured.
inline constexpr double kCosxStageGradTol   = 1e-4;
inline constexpr double kCosxStageEnergyTol = 1e-6;
inline constexpr int    kCosxStageMaxIter   = 30;

// Run an SCF through the multi-stage COSX grid progression.
//
//   stages      : coarse→fine COSX grids (``cosx_grid_stages_for_level``);
//                 guaranteed non-empty, last element is the fine grid.
//   staged_jk   : the COSX builder, already constructed on ``stages.front()``
//                 (it also seeds the initial guess). Reused for every stage;
//                 its grid is swapped in place between stages.
//   run_segment : (const JKBuilder& jk, const OptsT& stage_opts) -> ResultT.
//                 Runs one SCF segment with ``jk`` + ``stage_opts`` starting
//                 from the driver's carried density, and updates that
//                 carried density (captured by reference in the lambda).
//
// ResultT is given explicitly; OptsT and RunSeg are deduced. OptsT must
// expose ``conv_tol_grad`` / ``conv_tol_energy`` / ``max_iter`` (all SCF
// option structs do). ResultT must expose ``converged`` / ``n_iter`` /
// ``scf_trace`` / ``energy`` (all SCF result structs do).
//
// Returned-result semantics under the published scheme (n >= 2 stages):
//
//   * ``converged`` — whether the SCF converged to the caller's tolerance
//     on the second-to-last (converging) grid. The final-grid recompute is
//     a single non-iterated Fock build; its commutator (dominated by the
//     inter-grid exchange difference, not by SCF error) does not define
//     convergence, exactly as in the reference implementation.
//   * ``energy`` — recomputed with the final-grid exchange at the
//     converged density (the last segment's first-iteration energy).
//   * ``n_iter`` — total iterations across all stages (the final
//     recompute counts as 1).
//   * ``scf_trace`` — concatenation of every stage's trace, renumbered
//     1..n_iter, with the energy-difference seams recomputed so dE is
//     truthful across stage boundaries.
//
// Correctness fallback: a degenerate open-shell system can meet a
// COMMUTATOR FLOOR on the converging (middle) grid above the caller's
// tolerance (measured: OH·/def2-svp UHF stalls at ‖[F,DS]‖ ≈ 3e-5 on the
// middle GridX grid — the ²Π π_x/π_y degeneracy meets the grid's angular
// anisotropy — while the finer final grid's floor sits below 1e-6). When
// the converging stage fails to reach tolerance within the caller's
// ``max_iter``, the final grid therefore runs as a FULL converging
// segment (the pre-BUG87-A behaviour) instead of a single recompute, so
// floor-limited systems keep their convergence at pre-fix cost while
// every normally-converging system gets the published fast path.
template <typename ResultT, typename OptsT, typename RunSeg>
ResultT run_cosx_staged_scf(const Molecule& mol,
                            const std::vector<GridOptions>& stages,
                            const OptsT& opts,
                            const JKBuilder& staged_jk,
                            RunSeg run_segment) {
    std::optional<ResultT> result;
    const std::size_t n = stages.size();

    if (opts.max_iter == 0) {
        // NOITER is one initial-guess evaluation, not one evaluation per
        // warm-up stage.  Use the final configured grid so the returned
        // energy and operator match the staged driver's reported surface.
        if (n > 1) {
            staged_jk.set_cosx_grid(build_grid(mol, stages.back()));
        }
        return run_segment(staged_jk, opts);
    }

    // Accumulated across stages for the truthful returned result.
    decltype(std::declval<ResultT&>().scf_trace) full_trace;
    int total_iters = 0;
    bool mid_converged = false;
    bool have_prev_energy = false;
    double prev_energy = 0.0;

    for (std::size_t s = 0; s < n; ++s) {
        OptsT stage_opts = opts;
        const bool final_stage = (n >= 2) && (s + 1 == n);
        const bool converging_stage = (n == 1) || (s + 2 == n);
        if (final_stage && mid_converged) {
            // Published scheme: the final (largest) grid performs exactly
            // one Fock rebuild + energy recompute at the converged
            // density — the SCF never iterates on it.
            stage_opts.max_iter = 1;
        } else if (!final_stage && !converging_stage) {
            // Coarse warm-up stages: loose tolerance is enough — the
            // inter-grid exchange difference re-excites the error vector
            // past ~1e-4 anyway, so converging tighter here is wasted.
            stage_opts.conv_tol_grad =
                std::max(opts.conv_tol_grad, kCosxStageGradTol);
            stage_opts.conv_tol_energy =
                std::max(opts.conv_tol_energy, kCosxStageEnergyTol);
            stage_opts.max_iter = std::min(opts.max_iter, kCosxStageMaxIter);
        }
        // converging_stage — and the final stage in the commutator-floor
        // fallback — run with the caller's own tolerances.
        if (s > 0) {
            // Swap to this stage's grid on the same builder (stage 0 already
            // holds stages.front()); keeps the RI-J B-tensor, rebuilds only
            // the grid-dependent caches.
            staged_jk.set_cosx_grid(build_grid(mol, stages[s]));
        }
        result = run_segment(staged_jk, stage_opts);

        if (converging_stage) mid_converged = result->converged;

        // Splice this segment's trace into the cross-stage history:
        // renumber iterations globally and recompute dE across the stage
        // seam (each segment reports dE = 0 on its own first iteration).
        for (const auto& st : result->scf_trace) {
            auto row = st;
            row.iter = ++total_iters;
            if (have_prev_energy && st.iter == 1) {
                row.delta_e = row.energy - prev_energy;
            }
            prev_energy = row.energy;
            have_prev_energy = true;
            full_trace.push_back(std::move(row));
        }
    }

    // Fast path (mid_converged): convergence was established on the
    // converging grid; the recompute segment's own flag (its fine-grid
    // commutator vs tolerance) does not gate it. Fallback path: the
    // final segment converged (or not) on its own terms — keep its flag.
    if (n >= 2 && mid_converged) result->converged = true;
    result->n_iter = total_iters;
    result->scf_trace = std::move(full_trace);
    return std::move(*result);
}

}  // namespace vibeqc
