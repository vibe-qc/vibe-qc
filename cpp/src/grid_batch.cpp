#include "vibeqc/grid_batch.hpp"

#include "vibeqc/ao_eval.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace vibeqc {

GridBatches build_grid_batches(
    const BasisSet& basis,
    const Grid& grid,
    int batch_size,
    const std::vector<double>& shell_cutoffs,
    bool need_gradient) {
    if (batch_size < 1) {
        throw std::invalid_argument(
            "build_grid_batches: batch_size must be >= 1 (got "
            + std::to_string(batch_size) + ")");
    }

    const auto& shells = basis.libint();
    const auto n_shells = static_cast<int>(shells.size());
    const auto shell2bf = shells.shell2bf();
    const int n_bf = static_cast<int>(basis.nbasis());
    const int n_pts = static_cast<int>(grid.points.rows());

    if (static_cast<int>(shell_cutoffs.size()) != n_shells) {
        throw std::invalid_argument(
            "build_grid_batches: shell_cutoffs.size() ("
            + std::to_string(shell_cutoffs.size())
            + ") must equal basis.nshells() ("
            + std::to_string(n_shells) + ")");
    }

    GridBatches result;
    result.n_pts_total = n_pts;
    result.n_bf_total = n_bf;
    result.batch_size_hint = batch_size;
    result.batches.reserve(static_cast<std::size_t>(
        (n_pts + batch_size - 1) / batch_size));

    // Squared cutoffs — cheaper to compare than the un-squared
    // distance.
    std::vector<double> cutoff_sq(n_shells);
    for (int s = 0; s < n_shells; ++s) {
        cutoff_sq[s] = shell_cutoffs[s] * shell_cutoffs[s];
    }

    for (int batch_start = 0; batch_start < n_pts;
         batch_start += batch_size) {
        const int batch_end =
            std::min(batch_start + batch_size, n_pts);
        const int n_batch_pts = batch_end - batch_start;

        GridBatch batch;
        batch.start_index = batch_start;
        batch.n_points = n_batch_pts;
        batch.points = grid.points.middleRows(batch_start, n_batch_pts);
        batch.weights = grid.weights.segment(batch_start, n_batch_pts);

        // Primary-shell pruning (Stratmann-Scuseria-Frisch 1996 § 11
        // / Burow-Sierka 2011 § 2): include shell ``s`` whenever any
        // batch point falls inside its radial cutoff. Loop over
        // points-inside-cutoff stops at the first hit; whole-batch
        // miss is the dominant case for spatially-localised shells
        // away from the batch's region.
        batch.primary_shells.reserve(static_cast<std::size_t>(n_shells));
        for (int s = 0; s < n_shells; ++s) {
            const auto& O = shells[s].O;
            const double r_cut_sq = cutoff_sq[s];
            for (int gp = 0; gp < n_batch_pts; ++gp) {
                const double dx = O[0] - batch.points(gp, 0);
                const double dy = O[1] - batch.points(gp, 1);
                const double dz = O[2] - batch.points(gp, 2);
                const double d_sq = dx*dx + dy*dy + dz*dz;
                if (d_sq < r_cut_sq) {
                    batch.primary_shells.push_back(s);
                    break;
                }
            }
        }

        // Expand primary shells to the flat list of BF indices.
        std::size_t total_primary_bfs = 0;
        for (int s : batch.primary_shells) {
            total_primary_bfs += shells[s].size();
        }
        batch.primary_bfs.reserve(total_primary_bfs);
        for (int s : batch.primary_shells) {
            const auto bf0 = shell2bf[s];
            const auto ns = shells[s].size();
            for (std::size_t i = 0; i < ns; ++i) {
                batch.primary_bfs.push_back(
                    static_cast<int>(bf0 + i));
            }
        }

        // Evaluate AOs at batch points. The first-cut implementation
        // evaluates the full basis and column-selects to primary BFs;
        // a future commit specialises to a primary-only evaluator
        // (the column-select pattern is the standard PySCF /
        // Burow-Sierka 2011 starting point, see Stratmann 1996 § 11).
        // For the typical primary-fraction ≤ 0.5 on extended systems
        // the column-select waste is bounded and the implementation
        // stays free of basis-internals knowledge.
        const int n_primary_bfs =
            static_cast<int>(batch.primary_bfs.size());
        if (need_gradient) {
            AOValues ao = evaluate_ao_with_gradient(basis, batch.points);
            batch.chi_primary.resize(n_batch_pts, n_primary_bfs);
            for (int c = 0; c < n_primary_bfs; ++c) {
                batch.chi_primary.col(c) =
                    ao.values.col(batch.primary_bfs[c]);
            }
            for (int axis = 0; axis < 3; ++axis) {
                batch.dchi_primary[axis].resize(n_batch_pts,
                                                n_primary_bfs);
                for (int c = 0; c < n_primary_bfs; ++c) {
                    batch.dchi_primary[axis].col(c) =
                        ao.gradients[axis].col(batch.primary_bfs[c]);
                }
            }
            batch.has_gradient = true;
        } else {
            Eigen::MatrixXd ao = evaluate_ao(basis, batch.points);
            batch.chi_primary.resize(n_batch_pts, n_primary_bfs);
            for (int c = 0; c < n_primary_bfs; ++c) {
                batch.chi_primary.col(c) = ao.col(batch.primary_bfs[c]);
            }
            batch.has_gradient = false;
        }

        result.batches.push_back(std::move(batch));
    }

    return result;
}

}  // namespace vibeqc
