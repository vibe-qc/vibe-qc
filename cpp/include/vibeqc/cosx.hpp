// Chain-of-spheres exchange (COSX): seminumerical K-matrix build.
//
//   K_{μν} = Σ_{λρ} D_{λρ} (μλ|νρ)
//          ≈ Σ_g w_g χ_μ(r_g) Σ_ρ A_{νρ}(r_g) Σ_λ D_{λρ} χ_λ(r_g)
//
// where A_{νρ}(r_g) = ∫dr' χ_ν(r') χ_ρ(r') / |r' − r_g| is the
// analytic nuclear-attraction-style integral with a unit charge at
// the grid point r_g (computed via libint with q = −1, since libint's
// V_μν = Σ_A −q_A · ⟨χ_μ|1/r|χ_ν⟩).
//
// Reference: Neese, F.; Wennmohs, F.; Hansen, A.; Becker, U.,
// Chem. Phys. 2009, 356, 98 ("Efficient, approximate and parallel
// Hartree-Fock and hybrid DFT calculations. A 'chain-of-spheres'
// algorithm for the Hartree-Fock exchange").
//
// Pair with RI-J for the standard "RIJCOSX" hybrid-DFT acceleration
// (ORCA's default for hybrid SCF). The COSX K replaces O(N⁴) ERI
// assembly with O(N² · M) seminumerical work, where M is the grid
// size. Aggressive shell-pair / point screening (not yet enabled
// here) drops the prefactor further.

#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "basis.hpp"
#include "cosx_kernel.hpp"
#include "grid.hpp"
#include "grid_batch.hpp"
#include "lattice_sum.hpp"
#include "molecule.hpp"

namespace vibeqc {

// Default GridOptions for the COSX integration grid. Substantially
// sparser than the XC default (75 × 17 × 36 → ~4600 pts / atom on
// neutral organics), because COSX needs only the smooth 1/|r-r'|
// kernel through χ_μ(r) — no exchange-correlation gradient response —
// and Neese 2009 §3 reports sub-mHa per-atom accuracy on first-row
// systems with a far coarser quadrature than what's standard for XC.
//
// Tier picked here:
//   n_radial =  35   (vs 75 for XC)        — Treutler-Ahlrichs M4
//   n_theta  =   9   (vs 17)               — Gauss-Legendre
//   n_phi    =  18   (vs 36)               — uniform φ
// Total ~ 35 × 9 × 18 = 5670 / atom angularly, before Becke pruning;
// roughly 1/8 the XC grid. Matches ORCA's "default GridX" tier.
inline GridOptions default_cosx_grid_options() {
    GridOptions g;
    g.n_radial = 35;
    g.n_theta  = 9;
    g.n_phi    = 18;
    return g;
}

// COSX K-matrix variant selector (user-facing ``cosx_variant``).
//
//   STANDARD  Neese-Wennmohs-Hansen-Becker 2009 chain-of-spheres K with
//             no bra-side overlap fit: K = K_naive (symmetrised), plus
//             the one-centre analytic correction. Cheapest, and accurate
//             per grid point on a GridX tier — but NOT robust enough to be
//             the default (it destabilises some open-shell RIJCOSX SCFs);
//             explicit opt-in only.
//   FITTED    Overlap-fitted COSX: the analytic AO overlap is reproduced
//             at quadrature by the Q-junction K = Q·K_naive·Qᵀ with
//             Q = S·S_grid⁻¹ (Neese 2009 §2.4; the global-fit precursor
//             to the per-grid-point fit of Izsák & Neese, J. Chem. Phys.
//             135, 144105 (2011)). Robust on any grid; the default.
//   AUTO      Default. Resolves to FITTED (the robust choice) via
//             ``resolve_cosx_variant``; STANDARD is never auto-selected.
//
// NOTE (M3, planned): ``FITTED`` currently routes through the *global*
// overlap fit (Q = S·S_grid⁻¹). The *local* per-grid-point fit
// (Izsák-Neese 2011, Q_μg solved per grid point on the screened-shell
// overlap) is a separate kernel; when it lands, ``FITTED`` switches to
// it and the global fit stays reachable for parity.
enum class CosxVariant {
    AUTO = 0,
    STANDARD = 1,
    FITTED = 2,
};

// Heuristic orbital-basis cardinality from the basis-set name, used by
// the COSX auto-selection. Returns the zeta level: 2 = double-zeta
// (def2-SVP, cc-pVDZ, 6-31G*, …), 3 = triple-zeta (def2-TZVP,
// cc-pVTZ, …), 4 = quadruple-zeta (def2-QZVP, cc-pVQZ, …), 5 = 5-zeta
// (cc-pV5Z, …). Returns 0 ("unknown") when the name carries no
// recognisable zeta marker — the resolvers treat 0 as "do not let the
// basis force a variant/level" (threshold decides instead).
int cosx_basis_cardinality_from_name(const std::string& basis_name);

// Resolve a (possibly AUTO) COSX variant to a concrete STANDARD/FITTED.
//
// A non-AUTO input is returned unchanged. AUTO always resolves to FITTED,
// the robust overlap-fitted build. STANDARD (no fit) is accurate per grid
// point on a GridX tier but is NOT robust enough to auto-select (it caused
// RIJCOSX SCF convergence failures, open-shell especially), so it is an
// explicit opt-in only. The ``thresh_cosx`` / ``basis_cardinality`` /
// ``grid_level`` inputs are retained for API stability and a future,
// validated auto-STANDARD policy.
CosxVariant resolve_cosx_variant(CosxVariant variant,
                                 double thresh_cosx,
                                 int basis_cardinality,
                                 int grid_level);

// Resolve a (possibly auto) COSX grid level to a concrete tier in 1..4.
//   grid_level >= 1 : returned clamped to [1, 4] (explicit override).
//   grid_level <= -1: auto by basis cardinality — QZ+ → 3, else → 2.
//                     GridX1 is never auto-selected (too coarse for
//                     robust open-shell SCF); it is the explicit
//                     "quick single-point" tier only.
//   grid_level == 0 : sentinel "use the legacy GridOptions as supplied"
//                     — callers handle this before calling here; passing
//                     0 returns 2 (the neutral TZ default) defensively.
int resolve_cosx_grid_level(int grid_level, int basis_cardinality);

// COSX integration-grid presets, one per level 1..4. These follow the
// Helmich-Paris et al. 2021 *methodology* for COSX-specific grids
// (pruned Lebedev angular quadrature sized to minimise exchange-matrix
// error, paired with a Treutler-Ahlrichs radial grid) but are vibe-qc's
// own tiers parameterised from the openly-documented Krack-Köster radial
// formula and the ORCA-manual COSX accuracy tiers. They are NOT a
// verbatim copy of ORCA's globally-optimised GridX tables (those are not
// openly published). Each tier is a 5-region pruned angular set
// (``orca_angular_points``, inner-core → outer-tail Lebedev counts per
// radial band) on an ``n_radial``-shell radial grid; ``lebedev_order``
// is stamped from the peak count for logging only. The returned grid is
// the FINAL (fine) stage of the tier's DefGrid triple — see
// ``cosx_grid_stages_for_level`` for the coarse/middle SCF-iteration
// grids. Tiers, by intended use:
//   1  quick single-points (explicit only)
//        — 14/26/50/110/50,   50 radial (peak Lebedev order 17)
//   2  default / DZ / TZ — 26/50/110/194/110,  65 radial (peak order 23)
//   3  QZ                — 26/110/194/302/194, 75 radial (peak order 29)
//   4  tight / 5Z        — 26/194/302/434/302, 90 radial (peak order 35)
// ``level`` is clamped to [1, 4].
GridOptions cosx_grid_options_for_level(int level);

// Multi-stage COSX grid progression for a level 1..4: the sequence of
// (coarse → middle → fine) grids the SCF steps through, following ORCA's
// "DefGrid" stage triples (Helmich-Paris et al. 2021). Consecutive
// duplicate stages are collapsed, so the result has 2 or 3 grids; the
// LAST element equals ``cosx_grid_options_for_level(level)`` (the fine
// grid on which the converged exchange matrix is built). The SCF drivers
// run each non-final stage to a loose tolerance, carrying the density
// forward, then converge on the final grid. ``level`` is clamped to
// [1, 4]; level 0 (legacy single grid) does not use this path.
std::vector<GridOptions> cosx_grid_stages_for_level(int level);

// Smallest ``conv_tol_grad`` for which the seminumerical COSX exchange on a
// GridX grid still converges. The chain-of-spheres K carries a commutator-
// noise floor between 1e-6 and 1e-7 (measured, open-shell worst case), so a
// tighter gradient tolerance cannot be reached on a GridX grid and must
// fall back to the denser legacy grid. The default ``conv_tol_grad`` (1e-6)
// sits exactly at this bound, so default-tolerance SCFs use GridX while
// user-tightened ones (≤ 1e-7) use the legacy grid.
inline constexpr double kCosxAutoGridXMinTol = 1e-6;

// Decide whether an SCF should use a GridX tier (true) or the legacy single
// grid (false), from the (possibly auto) ``grid_level`` and the SCF's
// gradient tolerance:
//   grid_level == 0  : legacy            (explicit opt-out).
//   grid_level >= 1  : GridX             (explicit opt-in; honoured at any
//                                         tolerance — the caller chose it).
//   grid_level <= -1 : auto — GridX when ``conv_tol_grad`` is reachable on a
//                      GridX grid (``>= kCosxAutoGridXMinTol``), else legacy.
// This is the AUTO default: normal-tolerance SCFs get the accurate, faster
// GridX path; only a user-tightened ``conv_tol_grad`` falls back to legacy.
inline bool cosx_use_gridx(int grid_level, double conv_tol_grad) {
    if (grid_level == 0) return false;
    if (grid_level >= 1) return true;
    return conv_tol_grad >= kCosxAutoGridXMinTol;
}

// One-stop COSX grid resolver for the SCF / gradient drivers.
//   grid_level == 0 : keep ``legacy`` (the validated default grid) —
//                     the back-compatible default, no behaviour change.
//   otherwise       : build the tier from ``cosx_grid_options_for_level``
//                     (explicit 1..4, or auto when grid_level <= -1).
inline GridOptions resolve_cosx_grid_options(const GridOptions& legacy,
                                             int grid_level,
                                             int basis_cardinality) {
    if (grid_level == 0) return legacy;
    return cosx_grid_options_for_level(
        resolve_cosx_grid_level(grid_level, basis_cardinality));
}

// Build the basis-only Schwarz upper-bound table for COSX shell-pair
// screening:
//
//   schwarz(s1, s2) = sqrt( max_{μν ∈ s1 × s2}  |(μν|μν)| )    (Coulomb 2-e)
//
// Symmetric ``(n_shells, n_shells)`` matrix. Standard Häser-Ahlrichs
// 1989 Schwarz factor — basis-only and SCF-invariant. Compute it once
// at COSX-builder construction and pass it into every
// ``compute_cosx_k`` call so the shell-pair loop can drop pairs whose
// contribution to K is provably below tolerance, *before* paying for
// a libint nuclear-attraction call. The screen takes the form
//
//   skip pair (s1, s2)  if  schwarz(s1, s2) · max(|Dχ|_s1, |Dχ|_s2)
//                            < ``AO_TOL``  (1e-7 in current cosx.cpp)
//
// which bounds the contribution of pair (s1, s2) to ``K`` at the
// current grid point (the max is over the symmetric "A · Dχ + Aᵀ · Dχ"
// pair contribution; product-bound would mis-skip the asymmetric
// large-small case).
Eigen::MatrixXd build_cosx_schwarz(const BasisSet& basis);

// Build the overlap-fit Q-junction matrix for the COSX K corrector.
// Q ≡ S · S_grid^{-1} where S is the analytic AO overlap and
// S_grid_{μν} = Σ_g w_g χ_μ(r_g) χ_ν(r_g) is its grid-quadrature
// approximation. ``Q`` is the matrix that absorbs the bra-side
// quadrature error on K_naive into the corrected K = Q K_naive Qᵀ
// (Neese 2009 §2.4).
//
// Q depends only on the AO basis and the COSX grid — both invariant
// across a single SCF run. **Compute it once at SCF setup and reuse
// the same Q on every ``compute_cosx_k`` call**; the per-iter cost of
// recomputing it (two N×N solves + an N×N matrix-matrix product) is
// otherwise paid every iteration on top of the K build proper.
//
// Returns an (n_basis, n_basis) matrix on success. Returns a 0×0
// matrix on LDLT failure of S_grid (signals "use K_naive without
// the Q correction"; ``compute_cosx_k`` understands the 0×0 sentinel
// and falls through to the uncorrected K). LDLT failure is rare —
// it requires a degenerate basis or an unusually sparse grid.
Eigen::MatrixXd build_cosx_q(const BasisSet& basis,
                             const Grid& cosx_grid);

// Seminumerical K-matrix on the given grid. Returns an (n_basis,
// n_basis) matrix that should be ~symmetric to grid-quadrature
// precision; the caller usually symmetrises K = (K + K^T)/2 before
// folding into the Fock matrix. Density matrix ``D`` is the
// closed-shell total density (D = 2 C_occ C_occᵀ).
//
// Optional cached basis-only inputs (all four are independent and
// either-or — pass an empty container to disable):
//
// ``q_cached``: precomputed Q-junction matrix from ``build_cosx_q``.
//     Passing it lets the function skip the per-call Q-junction
//     rebuild (two N×N solves + a matrix-matrix product per invocation
//     — typically ~15 % of the K-build wall time at def2-TZVP-class
//     sizes; cumulative over ~10–15 SCF iterations).
// ``schwarz_cached``: shell-pair Schwarz table from
//     ``build_cosx_schwarz``. Tightens the per-grid-point shell-pair
//     screen.
// ``shell_cutoffs``: per-shell radial extents from
//     ``compute_shell_radial_cutoffs(basis.libint(), AO_TOL)``. When
//     non-empty, the K-build pre-prunes the shell-pair loop at each
//     grid point to the **active shells** — those whose origin is
//     within their tabulated cutoff distance of the grid point. For
//     extended systems where each grid point only sees a few nearby
//     atoms, this drops the per-point pair-loop iteration count from
//     n_shells² to n_active² — the dominant cost on the molecular
//     anchor for systems beyond ~20 atoms.
// ``grid_batches`` (optional, default ``nullptr``) routes the K-build
// through a per-batch outer loop that consumes the basis-only chi
// cache stored on every ``GridBatch::chi_primary`` (built once at
// SCF setup via ``build_grid_batches``). When supplied, this skips
// the per-call ``evaluate_ao`` over the full grid — amortising the
// AO-eval cost across all SCF iterations — and packages the per-
// point work into a per-batch outer loop, the substrate for the
// subsequent two-DGEMM K-accumulation work (Neese 2009 § 2;
// Burow-Sierka 2011 § 2). The L1-only first landing keeps the per-
// point pair loop unchanged; future commits restrict the pair loop
// to L2/L3 shell subsets.
// ``boys_table`` and ``pp_cache`` are basis-only inputs the new
// in-tree nuclear-attraction kernel (vibeqc::cosx_nuclear_pair_into)
// consumes in place of the libint ``Operator::nuclear`` engine.
// Pass ``nullptr`` to have ``compute_cosx_k`` build local copies
// per call — that path is correctness-equivalent but pays a one-
// time table + cache build per SCF iteration. The COSXJKBuilder
// holds permanent copies on the instance and passes them in for
// the hot SCF path.
//
// ``lattice`` (optional, default ``nullptr``) enables periodic
// minimum-image convention for the per-grid-point shell-distance
// screening. When non-null, the shell-origin → grid-point distances
// used by the radial-cutoff active-shell pre-prune and the Schwarz
// distance prescreen are computed with periodic boundary conditions:
// each Cartesian displacement is converted to fractional coordinates,
// rounded to the nearest integer, and converted back — so a shell
// on the far side of a tight unit cell registers the short periodic
// distance across the boundary. For molecular systems (nullptr = default)
// the plain Euclidean distance is used — zero overhead. The lattice
// matrix must be full 3×3; vacuum directions (dim < 3) should carry
// large padding constants (≥ 30 bohr) so their minimum-image round
// collapses to the identity.
//
// ``image_cells`` (optional, default ``nullptr``) enables image-cell
// exchange summation for tight periodic cells (M3a). When a non-
// trivial cell list is supplied alongside ``lattice``, the per-grid-
// point exchange kernel sums the analytic A-operator over all lattice
// cells R in the list (home cell R = 0 included):
//
//   F_ν(r_g) = Σ_R Σ_σ A_{νσ}(r_g − R) · (D · χ(r_g))_σ
//
// where A_{νσ}(C) is the home-cell nuclear-attraction-style integral
// with the pseudo-nucleus shifted to C = r_g − R. This accumulates
// the exchange coupling of the home-cell bra pair density to every
// image-cell replica of the ket pair density — the leading
// single-lattice-sum image class on top of the home-cell-only kernel
// (vibe-qc periodic COSX extension of Neese 2009). It is NOT yet the
// full Γ-folded exchange: at Γ-only sampling the density couples all
// cell pairs, generating long-range exchange classes this single sum
// does not contain (measured H-chain anchor, D = I: home-only
// ||K|| = 1.74 → image-summed 3.94, vs Γ-folded GDF 16.16; see
// handovers/HANDOVER_RIJCOSX_M3A.md § "M3a measured outcome"). Pass the
// truncated cell list from
// ``direct_lattice_cells(system, cutoff_bohr)``; the radial-cutoff
// active-shell pre-prune skips cells with no shell in range of
// r_g − R, so vacuum-padded cells (home-only list) pay zero overhead.
//
// Caveats:
//   * Requires ``lattice != nullptr``. Ignored otherwise.
//   * A trivial list (``size() <= 1``, i.e. home cell only — the
//     ``direct_lattice_cells`` result always contains the zero cell)
//     falls back to the single-pass kernel with minimum-image
//     screening, preserving the batched-path performance for
//     molecular-limit boxes.
//   * When image summation is active, ``grid_batches`` is ignored
//     and the kernel runs the unbatched path (the per-batch L2
//     hierarchy does not yet know about per-cell primary shells —
//     a separate optimization; handovers/HANDOVER_RIJCOSX_M3A.md §4.5).
Eigen::MatrixXd compute_cosx_k(const BasisSet& basis,
                               const Eigen::MatrixXd& D,
                               const Grid& cosx_grid,
                               const Eigen::MatrixXd& q_cached
                                   = Eigen::MatrixXd(),
                               const Eigen::MatrixXd& schwarz_cached
                                   = Eigen::MatrixXd(),
                               const std::vector<double>& shell_cutoffs
                                   = std::vector<double>(),
                               const GridBatches* grid_batches
                                   = nullptr,
                               const BoysTable* boys_table
                                   = nullptr,
                               const PrimitivePairCache* pp_cache
                                   = nullptr,
                               const Eigen::Matrix3d* lattice
                                   = nullptr,
                               const std::vector<LatticeCell>* image_cells
                                   = nullptr,
                               bool apply_overlap_fit = true);

// Analytic nuclear gradient of the K-energy contribution
//
//   E_K = −(α_hf / 4) · tr(D' · K_naive[D]),   D' = Qᵀ D Q
//
// for the (overlap-fitted) chain-of-spheres K — the mixed bilinear
// form Σ_g w_g (D'χ)ᵀ A(r_g) (Dχ) per Izsák & Neese, J. Chem. Phys.
// 135, 144105 (2011), Eqs. (6)/(12) (doi:10.1063/1.3646921).
//
// Differentiated terms (2026-07-29 parity fix):
//
//   1. AO derivatives of both χ factors (T1) and the bra/ket A-matrix
//      center derivatives (T2), in the MIXED D'/D contraction.
//   2. The overlap-fit response: Q = S·S_grid⁻¹ is differentiated
//      through the analytic dS and the (frozen-grid) grid-sampled
//      dS_grid — the analogue of the fitted derivative representation
//      of Izsák-Neese Sec. II.D, Eqs. (19)-(22).
//
// Remaining neglects (documented, measured):
//
//   * Grid-point motion and Becke-weight derivatives (frozen grid /
//     frozen weights — same neglect class as ORCA's default GradType).
//     Measured by the fixed-density FD decomposition
//     (examples/regression/rijcosx_gradient_fd_decompose.py):
//     ≤ 2.8e-4 Ha/bohr per component on H2O/def2-tzvp, ~3e-3 on the
//     OH radical/def2-svp legacy grid. NOT µHa-scale on general
//     systems — see EVIDENCE of the 2026-07-29 glycine parity fix.
//   * The FITTED SCF Fock is not the exact functional derivative of
//     the fitted energy (D → Q·K_naive[D]·Qᵀ is not self-adjoint), so
//     a Hellmann-Feynman response residual of first order in (Q−1)
//     survives in any frozen-density gradient, including this one.
Eigen::MatrixXd compute_cosx_k_gradient_contribution(
    const Molecule& mol,
    const BasisSet& basis,
    const Eigen::MatrixXd& D,
    const Grid& cosx_grid,
    double alpha_hf,
    const Eigen::MatrixXd& q_cached = Eigen::MatrixXd(),
    bool apply_overlap_fit = true);

}  // namespace vibeqc
