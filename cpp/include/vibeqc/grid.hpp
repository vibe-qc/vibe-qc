// Molecular quadrature grid for DFT numerical integration.
//
// Produces a set of points (r_g ∈ R^3) with weights w_g such that for any
// well-behaved function f(r),
//     ∫ f(r) dr ≈ Σ_g w_g f(r_g).
//
// Construction follows the standard recipe used by PySCF / ORCA / NWChem:
//   1. Treutler-Ahlrichs M4 radial grid per atom (Chem. Phys. Lett. 240,
//      283 (1995)): r_k = ξ (1-x_k)^α ln(2/(1-x_k))^β with Gauss-Chebyshev
//      second-kind nodes x_k.
//   2. Angular grid per radial shell: either a Lebedev-Laikov rule of a
//      given algebraic order (the modern default — Lebedev 1976, Laikov
//      & Lebedev 1999), or the legacy Gauss-Legendre × uniform-φ product
//      grid (kept as an opt-in for the v0.7.x parity matrix). Lebedev
//      gives the same angular accuracy for ~half the points.
//   3. Becke fuzzy-cell atomic partitioning (J. Chem. Phys. 88, 2547
//      (1988)): each grid point gets a weight P_A(r) that sums to 1 over
//      atoms; for a point on the atomic grid of A, the weight going into
//      the molecular grid is (radial · angular · Becke)_A.
//
// Default resolution (legacy ``AngularScheme::ProductGaussLegendre``):
//   radial  = 75 points per atom
//   angular = 17 θ × 36 φ = 612 points per shell
//   Total  ≈ 46k points per atom.
//
// Modern Lebedev path (``AngularScheme::Lebedev`` + ``lebedev_order =
// 29``): 302 points/shell — about half the count for the same XC
// accuracy (PySCF level-3 medium). The default will switch to Lebedev
// in a follow-up commit once the parity-matrix tolerances are reviewed.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "molecule.hpp"

namespace vibeqc {

struct PeriodicSystem;

// Angular-grid family used inside ``build_grid``. ``Lebedev`` (Lebedev-
// Laikov) is the modern XC default; ``ProductGaussLegendre`` is kept
// for parity-matrix bit-reproducibility (current default) and as the
// pinned default for COSX (see ``cpp/include/vibeqc/cosx.hpp``).
enum class AngularScheme {
    Lebedev,
    ProductGaussLegendre,
};

// Angular-grid pruning scheme — reduces the Lebedev angular order at
// radial shells near the nucleus (core, ρ nearly spherical) and far
// from the nucleus (tail, ρ smooth) where high-degree spherical
// harmonics integrate trivially. Standard scheme, originally Murray-
// Handy-Laming 1993 (Mol. Phys. 78, 997) refined as SG-1 (Gill-
// Johnson-Pople 1993, Chem. Phys. Lett. 209, 506) and reproduced in
// NWChem / PySCF / Q-Chem.
//
// ``None`` (default): every radial shell uses ``GridOptions::
// lebedev_order``. Kept as default for parity-matrix bit-
// reproducibility — switches to ``NWChem`` (the modern default) in a
// follow-up commit once the parity tests are re-toleranced.
//
// ``NWChem``: 3-tier scheme — inner core uses a lower order, outer
// tail uses a lower order, the chemically-active middle band uses
// the full requested order. Typical point-count reduction 40-60 %
// with sub-µHa effect on first-row organic atomization energies
// (see Murray 1993 § 3).
enum class AngularPruning {
    None,
    NWChem,
};

// Atomic-partition cell-function family used inside the per-grid-point
// Becke partition. ``Becke`` is the original 1988 scheme (J. Chem. Phys.
// 88, 2547): k iterated p(μ) = (3μ - μ³)/2 smoothings, full O(N²)
// cost per grid point. ``Stratmann`` is the 1996 piecewise-polynomial
// variant (Stratmann, Scuseria, Frisch, Chem. Phys. Lett. 257, 213) —
// a single 7th-order polynomial on |μ| ≤ a (= 0.64 per the paper) with
// **hard cutoff outside**, so atoms whose μ_AB exceeds the cutoff
// contribute exactly 0 or 1 to the partition. The hard cutoff lets the
// per-grid-point partition cost drop from O(N²) to O(N · N_nearby),
// asymptotically linear for extended systems — the lever the 1996
// paper was built to deliver. Numerically tighter at the partition
// boundary (1e-5..1e-6 Ha less noise than 3×-iterated Becke at the same
// grid size).
enum class AtomicPartition {
    Becke,
    Stratmann,
};

// Complete atomic-grid construction profile. ``Generic`` preserves every
// existing GridOptions control below. ``PySCFLevel3`` is the molecular grid
// contract used by the official SKALA-1.1 checkpoint integration: PySCF
// 2.14.0 level 3 period-dependent radial/angular sizes, atom-specific
// Treutler-Ahlrichs radii, radius-based NWChem pruning, and the Treutler
// sqrt(Bragg-radius) correction to heteroatomic Becke cells.
//
// Keep this opt-in. It intentionally overrides ``n_radial``, ``angular``,
// ``lebedev_order``, ``angular_pruning``, ``partition``, ``becke_k``, and
// ``orca_angular_points`` as one internally consistent protocol; generic DFT
// grids retain their historical behaviour and layout.
enum class AtomicGridProfile {
    Generic,
    PySCFLevel3,
};

// Stable wire names used by external-XC capability negotiation. Keep these
// independent of pybind11's enum spelling so a callback sees the same value
// from molecular and periodic builders.
inline constexpr const char* atomic_grid_profile_name(
    AtomicGridProfile profile) noexcept {
    switch (profile) {
    case AtomicGridProfile::Generic:
        return "generic";
    case AtomicGridProfile::PySCFLevel3:
        return "pyscf-level3";
    }
    return "unknown";
}

struct GridOptions {
    AtomicGridProfile atomic_grid_profile = AtomicGridProfile::Generic;
    int n_radial = 75;     // Treutler-Ahlrichs radial points per atom

    // ---- Angular grid selector ----
    //
    // ``Lebedev`` uses a Lebedev-Laikov rule of order ``lebedev_order``
    // — 5/7/11/15/17/23/27/29/31/35/41/47/53 map to
    // 14/26/50/86/110/194/266/302/350/434/590/770/974 points per radial
    // shell. Bundled tables live in ``cpp/src/lebedev_data.cpp``
    // (Lebedev 1976; Laikov & Lebedev 1999). Order 29 (302 pts) is the
    // PySCF level-3 default for first-row atoms; order 35 (434 pts) for
    // heavier elements; order 41 for level-5 "tight".
    //
    // ``ProductGaussLegendre`` is the legacy n_θ × n_φ grid. It is the
    // current default for parity-matrix bit-reproducibility — switches
    // to ``Lebedev`` in a follow-up commit once parity tolerances are
    // reviewed.
    AngularScheme angular = AngularScheme::ProductGaussLegendre;
    int lebedev_order = 29;
    int n_theta  = 17;     // Gauss-Legendre θ nodes per shell (product only)
    int n_phi    = 36;     // Uniform φ nodes per shell (product only)

    // ---- Angular-order pruning across radial shells ----
    //
    // Applies only when ``angular == Lebedev``. The pruning rule
    // (Murray-Handy-Laming 1993 / SG-1 / NWChem) reduces the angular
    // order at radial shells in the core (ρ nearly spherical, low-ℓ
    // suffices) and tail (ρ smooth, low-ℓ suffices), keeping the full
    // requested order only in the middle band where chemistry happens.
    // ~40-60 % grid-point reduction at sub-µHa accuracy cost on
    // first-row organics. Default ``None`` for parity-matrix
    // bit-reproducibility — flip to ``NWChem`` after the parity tests
    // are re-toleranced.
    AngularPruning angular_pruning = AngularPruning::None;

    // ---- Atomic partitioning ----
    //
    // ``partition`` selects the cell-function family — ``Becke`` (1988,
    // O(N²) per grid point, current default for bit-reproducibility) or
    // ``Stratmann`` (1996, O(N · N_nearby) per grid point, modern
    // default in PySCF / NWChem / Q-Chem). Switching to Stratmann is a
    // ~µHa numerical drift at the partition boundary and a meaningful
    // speedup of the grid-build step on >20-atom systems. ``becke_k``
    // is the smoothing order for the Becke iterated polynomial — used
    // only on the Becke path.
    AtomicPartition partition = AtomicPartition::Becke;
    int becke_k  = 3;      // smoothing order for Becke cells (standard is 3)

    // ---- 2021 ORCA-style 5-region pruned-Lebedev "AngularGrid" ----
    //
    // When this holds exactly 5 entries (Lebedev point counts, ordered
    // inner-core → outer-tail, e.g. {14, 26, 50, 110, 50} for ORCA's
    // AngularGrid2), ``build_grid`` splits each atom's radial shells into
    // 5 contiguous index-bands and uses the bundled Lebedev tier whose
    // point count matches each band — OVERRIDING ``lebedev_order`` and
    // ``angular_pruning``. This reproduces the 5-region pruned angular
    // quadrature of Helmich-Paris, de Souza, Neese & Izsák, J. Chem.
    // Phys. 155, 104109 (2021), Table I (the COSX "GridX"/"DefGrid"
    // angular grids).
    //
    // The Table I angular point counts themselves are exact. The region
    // *boundaries* are taken as fixed shell-INDEX fractions
    // (``kOrcaRegionFractions`` in grid.cpp), NOT the radius multipliers
    // of the paper: ORCA's per-element Clementi-radius multipliers that
    // delimit the regions are not openly published (ORCA-binary only),
    // and a radius-based split would make the per-atom point count
    // element-dependent, breaking the uniform per-atom output layout that
    // lets the grid build run lock-free in parallel. The index-fraction
    // split is therefore a documented approximation of the 2021 scheme.
    //
    // Empty (default) = use the legacy ``lebedev_order`` / ``angular_pruning``
    // path unchanged. Only consulted when ``angular == Lebedev``.
    std::vector<int> orca_angular_points;

    // ---- VV10 nonlocal correlation grid factor ----
    //
    // When a functional requires VV10 nonlocal correlation (needs_vv10),
    // the SCF driver builds a separate, coarser integration grid for the
    // VV10 double integral.  The VV10 kernel is smooth (1/R^6 decay) and
    // converges rapidly with grid density — the semilocal XC grid is
    // needlessly dense for VV10, and the O(N^2) double sum over the full
    // grid is the dominant cost of every VV10-paired functional.
    //
    // ``vv10_grid_factor`` scales down the radial and angular resolution
    // relative to the main XC grid:
    //   vv10_n_radial  = max(5, n_radial / vv10_grid_factor)
    //   vv10_lebedev_order = next lower bundled Lebedev order
    //                         (or n_theta/n_phi scaled for product grids)
    //
    // Total grid points drop by ~factor^2, and the O(N^2) pair count by
    // ~factor^4 — a factor-3 setting yields ~81x speedup in the VV10
    // kernel.  Energy change is sub-mHa on first-row organics at factor 3
    // (the kernel double-integral is smooth; µHa convergence is already
    // reached at much coarser quadrature than semilocal XC requires).
    //
    // Default 3.0 — safe for production, validated against the full-grid
    // reference.  Set to 1.0 for bitwise-reproducible VV10 energies
    // (matching the pre-v0.16 behaviour).  Zero or negative disables the
    // separate VV10 grid (evaluates VV10 on the full XC grid).
    double vv10_grid_factor = 3.0;
};

struct Grid {
    // The complete construction protocol that produced this grid. External
    // XC providers may declare a required profile and validate this metadata
    // at the common provider boundary.
    AtomicGridProfile atomic_grid_profile = AtomicGridProfile::Generic;
    Eigen::MatrixX3d points;       // (N, 3), position in bohr
    Eigen::VectorXd  weights;      // (N,), integrator weights (bohr^3)
    // Unpartitioned radial x angular weight of the owning atom's grid.
    // Local libxc functionals do not need it, but atom-nonlocal functionals
    // such as SKALA use this second quadrature for their descriptors.  Keep
    // it distinct from ``weights``: the latter includes the molecular or
    // periodic fuzzy-cell partition and is the energy quadrature.
    Eigen::VectorXd  atomic_weights; // (N,), raw atom-grid weights (bohr^3)
    std::vector<int> atom_of_point; // (N,) which atom "owns" each point
    // One coarse point per owning atom, in the same atom order used by the
    // contiguous grid blocks.  Coordinates are in bohr.
    Eigen::MatrixX3d atom_coords;   // (n_atoms, 3)
    std::vector<int> atomic_numbers; // (n_atoms,), nuclear charges
};

// Bound the AO value/derivative tables used by molecular XC energy, Fock,
// and analytic-gradient quadratures.  The grid itself remains unchanged;
// consumers traverse contiguous slices of this many points and accumulate
// the same weighted sums.  Keep the Python memory estimator wired to this
// value through the native binding rather than duplicating the constant.
inline constexpr int kMolecularXcGridBatchSize = 4096;

// Upper bound for independent molecular XC grid batches evaluated at once.
// Each worker owns AO, libxc, and Fock-projection scratch, so allowing an
// unbounded hardware-thread count would turn the bounded-batch route back
// into an unbounded memory allocation on large nodes.  The Python preflight
// imports the same value when accounting for the concurrent workspaces.
inline constexpr int kMolecularXcGridMaxWorkers = 64;

// Stability matvec slices carry substantially more per-worker state than the
// first-order XC build, so their independent cap is deliberately smaller.
inline constexpr int kMolecularXcKernelMaxWorkers = 16;

Grid build_grid(const Molecule& mol, const GridOptions& opts = {});

// Exact number of raw atom-grid points owned by each atom, in molecule order,
// without constructing coordinates, Becke partitions, or AO tables. This is
// the cheap planning surface used by memory preflight and atom-aligned SKALA
// model chunking.
std::vector<int> grid_atomic_point_counts(
    const Molecule& mol, const GridOptions& opts = {});

// Periodic-Becke variant for solid-state DFT integration.
//
// Generates grid points around ``grid_mol`` atoms (the home unit cell),
// but applies the Becke fuzzy-cell partition over a wider set of
// ``partition_atom_positions`` that includes home + image atoms within
// some user-chosen radius. The resulting Grid has weights
//
//     w_g = w_radial × w_angular × P_A^periodic(r_g)
//
// with the partition denominator summed over every atom in
// ``partition_atom_positions``. This handles tight crystals where image
// atoms are close enough that the molecular Becke partition over
// home-cell atoms only would over-count near the unit-cell boundary.
//
// ``atom_of_point`` indexes into ``grid_mol.atoms()`` (i.e. only the
// home-cell atoms — the image atoms appear only in the partition
// denominator and never own grid points).
//
// At the molecular limit (``partition_atom_positions == grid_mol``),
// this reduces to the molecular ``build_grid`` to numerical precision.
Grid build_grid_periodic(
    const Molecule& grid_mol,
    const std::vector<Eigen::Vector3d>& partition_atom_positions,
    const GridOptions& opts = {});

// Atomic-number-aware periodic overload. The additional vector is required by
// ``AtomicGridProfile::PySCFLevel3`` so image atoms receive the same
// heteroatomic Treutler/Becke radius adjustment as home-cell atoms. Generic
// callers may continue using the overload above unchanged.
Grid build_grid_periodic(
    const Molecule& grid_mol,
    const std::vector<Eigen::Vector3d>& partition_atom_positions,
    const std::vector<int>& partition_atomic_numbers,
    const GridOptions& opts = {});

// Normalize the physical atom-image neighborhood of each grid point.
// Points and raw atomic weights retain the ordinary atom-major layout.
// Positive radii expand to twice the nearest-image distance to cover vacuum.
// Radius zero explicitly requests the molecular grid.
Grid build_periodic_point_grid(const PeriodicSystem& system,
                               double image_radius,
                               const GridOptions& opts = {});

}  // namespace vibeqc
