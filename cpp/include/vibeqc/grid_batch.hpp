// Reusable per-batch view of a numerical-integration grid.
//
// A ``GridBatch`` packages a contiguous slice of grid points
// (``start_index .. start_index + n_points``) with the minimum data
// that any seminumerical or XC-quadrature kernel needs to operate
// on the slice:
//
//   * the point coordinates + integration weights, sliced from the
//     parent ``Grid``;
//
//   * the list of basis-set shells whose contracted AOs have non-
//     negligible amplitude *somewhere* in the batch (the "primary"
//     or L1 shell set), determined by the per-shell Gaussian-extent
//     bound (see ``compute_shell_radial_cutoffs`` in
//     ``cpp/include/vibeqc/schwarz.hpp``);
//
//   * the pre-evaluated AO values ``χ_μ(g)`` for ``μ`` in the
//     primary basis-function set, at every batch point. Optionally
//     also the Cartesian AO gradients ``∇χ_μ(g)`` (needed for GGA
//     XC quadrature).
//
// Why a per-batch chi cache? In a converged SCF, ``compute_K`` and
// ``build_xc`` get called O(10) times. Re-evaluating ``χ_μ(g)`` per
// iteration is wasted work because ``χ`` is basis-only — it doesn't
// depend on the density. Storing ``χ`` densely for the full grid
// would pay the memory wall at any non-trivial system size
// (``n_pts × n_bf × 8 B`` is tens of GB on a 50-atom system at
// def2-svp). Batching + primary-shell pruning is the standard
// compromise: per-batch storage is small (the primary shell set is
// typically 10–30 % of the full basis on extended systems), and the
// downstream contractions (``F = χ · D``, ``K += χᵀ · w · F``)
// become small BLAS-3 GEMMs that run near peak throughput.
//
// This file is *infrastructure only*: it builds the batches and the
// primary chi caches; it does no SCF, no Fock build, no XC kernel
// dispatch. The COSX-K builder and the batched XC builder consume
// this primitive separately.
//
// References (mathematical, no proprietary source consulted):
//
//   * Neese, F.; Wennmohs, F.; Hansen, A.; Becker, U.,
//     *Efficient, approximate and parallel Hartree-Fock and hybrid
//     DFT calculations. A 'chain-of-spheres' algorithm for the
//     Hartree-Fock exchange*, Chem. Phys. 356, 98 (2009), § 2.
//
//   * Stratmann, R. E.; Scuseria, G. E.; Frisch, M. J.,
//     *Achieving linear scaling in exchange-correlation density
//     functional quadratures*, Chem. Phys. Lett. 257, 213 (1996),
//     § 11.
//
//   * Burow, A. M.; Sierka, M.,
//     *Linear scaling hierarchical integration scheme for the
//     exchange-correlation term in molecular and periodic systems*,
//     J. Chem. Theory Comput. 7, 3097 (2011), § 2.

#pragma once

#include <Eigen/Dense>

#include <array>
#include <vector>

#include "basis.hpp"
#include "grid.hpp"

namespace vibeqc {

// One contiguous slice of the integration grid + the per-batch caches
// needed by seminumerical / XC kernels. See file-level documentation
// for the design rationale.
struct GridBatch {
    int start_index = 0;                  // first grid index in the parent
    int n_points = 0;

    // (n_points, 3) Cartesian coordinates (bohr).
    Eigen::Matrix<double, Eigen::Dynamic, 3> points;
    // (n_points,) integration weights.
    Eigen::VectorXd weights;

    // L1 / "primary" shells — those with non-negligible χ anywhere in
    // this batch, per the Gaussian-extent radial-cutoff bound:
    //
    //   shell s is primary iff ∃ g ∈ [start, start + n_points) with
    //     |r_g − O_s|  <  r_cutoff[s].
    //
    // Indices into ``basis.libint()``.
    std::vector<int> primary_shells;
    // Basis-function indices in the primary set (flattened across
    // shells), contiguous and increasing. Length = ``chi_primary.cols()``.
    std::vector<int> primary_bfs;

    // χ_μ(g) for μ ∈ ``primary_bfs`` at every batch point.
    // Shape: (n_points, primary_bfs.size()). Basis-only / SCF-invariant
    // — built once at SCF setup and reused across iterations.
    Eigen::MatrixXd chi_primary;

    // Cartesian gradients of the primary AOs. Each component matches
    // ``chi_primary`` shape. Populated only when ``has_gradient``
    // (i.e. ``need_gradient = true`` at construction); empty for LDA
    // and other gradient-free consumers.
    std::array<Eigen::MatrixXd, 3> dchi_primary;
    bool has_gradient = false;
};

// Collection of grid batches covering the full parent ``Grid`` in
// order. Batches partition the grid: the union of their point ranges
// is exactly ``[0, n_pts_total)``, and the union of their points is
// the full grid.
struct GridBatches {
    std::vector<GridBatch> batches;
    int n_pts_total = 0;
    int n_bf_total = 0;
    int batch_size_hint = 0;   // the ``batch_size`` argument used
};

// Build a batched view of ``grid``. Each batch covers up to
// ``batch_size`` contiguous grid points (the last batch may be
// smaller). Per-batch primary-shell pruning uses ``shell_cutoffs``
// (length ``n_shells``, typically from
// ``compute_shell_radial_cutoffs`` at AO tolerance ≤ 1e-7).
//
// Throws ``std::invalid_argument`` if ``batch_size < 1`` or
// ``shell_cutoffs.size() != basis.nshells()``.
//
// Cost: O(n_pts × n_shells) for the primary-shell pruning + the cost
// of one ``evaluate_ao_with_gradient`` per batch over the primary
// columns. The per-batch AO evaluation is a one-time setup cost
// (basis-only / SCF-invariant); the storage trade is described in
// the file-level doc.
GridBatches build_grid_batches(
    const BasisSet& basis,
    const Grid& grid,
    int batch_size,
    const std::vector<double>& shell_cutoffs,
    bool need_gradient = false);

}  // namespace vibeqc
