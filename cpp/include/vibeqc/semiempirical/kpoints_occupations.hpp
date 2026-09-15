// Closed-shell occupations for semiempirical k-point routes.

#pragma once

#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace vibeqc {
namespace semiempirical {

// Roundoff-degeneracy grouping of the T = 0 Aufbau fills (issues #339, #434,
// #544). Two sorted band states are "the same level" when they lie within
//
//     energy_tolerance = kZeroTemperatureDegeneracyUlps * eps * energy_scale
//
// of each other. This is a roundoff equality test, not a physical broadening:
// 256 ulps at the spectrum's scale covers the scatter LAPACK leaves inside an
// exactly degenerate manifold and is orders of magnitude below any physical
// splitting (kT at room temperature is 9.5e-4 Ha; the #434 guard's
// min_resolvable_frontier_gap is 1e-6 Ha).
//
// Grouping is TRANSITIVE (#544): a state joins the current group when it lies
// within energy_tolerance of the group's LAST member, so two adjacent groups
// are always more than energy_tolerance apart by construction and a
// roundoff-degenerate manifold cannot be chopped between two members that are
// themselves indistinguishable. A ball around the group's first member (the
// pre-#544 rule) could leave adjacent groups one ulp apart and integer-fill
// across the chop. The chain is bounded: a group's span (last - first) may
// not exceed kDegenerateGroupSpanFactor * energy_tolerance; beyond that a new
// group starts. That bound (4096 ulps) stays far above the scatter of any
// realistic exactly degenerate manifold and far below
// min_resolvable_frontier_gap, so a fill boundary landing on the bound is
// recorded as an unresolved cut by the #434 guard instead of escaping.
//
// The Python twin (vibeqc/smearing/apply.py) uses the same two numbers;
// tests/test_smearing_frontier_resolution.py pins them equal through the
// module attributes exported from the bindings.
inline constexpr double kZeroTemperatureDegeneracyUlps = 256.0;
inline constexpr double kDegenerateGroupSpanFactor = 16.0;

struct KPointOccupationOptions {
    double smearing_temperature = 0.0;
    // Smallest frontier gap (Ha) the T = 0 hard-Aufbau fill is allowed to
    // resolve by CUTTING it (issue #434).
    //
    // Deliberately NOT called a "metallic" tolerance: is_metallic already
    // ships in this translation unit with a different and incompatible
    // meaning (a fractional occupation, or a measured indirect_gap <= 0),
    // and the racing case guarded here has NEITHER -- it produces integer
    // occupations and a small POSITIVE gap, which is exactly why it slips
    // through every existing screen.
    //
    // Value: 1e-6 Ha ~ 0.027 meV, some 4 orders below room-temperature kT
    // (9.5e-4 Ha); nothing physical distinguishes such a gap from zero.
    // It is also what the tree already means by the same concept
    // (metallic_gap_tolerance_hartree, periodic.chi.properties).
    // It was checked against the real population rather than argued: the
    // six #424 wave-flip fixtures cut their frontier at 1.66e-5 .. 1.09e-3
    // Ha, so 1e-6 clears the closest of them by 16x, while 1e-5 would
    // clear it by only 1.7x and would be unsafe.
    //
    // 0.0 disables the guard explicitly (a gap is never <= 0 between two
    // distinct adjacent states, so the predicate can never fire).
    double min_resolvable_frontier_gap = 1.0e-6;
};

struct KPointOccupationResult {
    std::vector<Eigen::VectorXd> occupations_per_k;
    double fermi_level = 0.0;
    double entropy = 0.0;
    // Issue #434. True when a T = 0 global-Aufbau fill placed the
    // occupied/empty boundary between two states separated by no more than
    // min_resolvable_frontier_gap -- i.e. it CUT a frontier it cannot
    // resolve, so which state carries the electrons is decided by last-ulp
    // arithmetic and moves between hosts.
    //
    // The kernel RECORDS rather than throws, on purpose: it runs inside the
    // SCC iteration loop, and a run whose intermediate iterate transits a
    // near-degenerate frontier but whose CONVERGED frontier is well gapped
    // must not be aborted. Drivers fail closed on the accepted spectrum.
    bool frontier_cut_unresolved = false;
    // The measured gap across that boundary in Ha, NaN when no boundary was
    // cut (an equalized fractional Fermi group, or completely empty/filled
    // bands). Recorded whether or not the guard tripped, so a caller can see
    // how much margin it had.
    double frontier_gap = std::numeric_limits<double>::quiet_NaN();
};

// ---------------------------------------------------------------------------
// Measured band edges + gaps from the occupations actually used (issue #426).
// ---------------------------------------------------------------------------
//
// Classification convention (identical to the occupation masks of the
// ab-initio multi-k band-extrema renderer, periodic_runner._band_summary):
// with occupation tolerance t (default 1e-8), a state (k, i) with
// occupation n is
//
//   * a VALENCE state when      n > t          (it holds electrons), and
//   * a CONDUCTION state when   n < 2 - t      (it can accept electrons).
//
// A fractionally occupied state (t < n < 2 - t) -- a roundoff-degenerate
// zero-temperature Fermi group sharing one Weinert-Davenport occupation, or
// a finite-temperature Fermi-Dirac frontier -- belongs to BOTH classes.
// Consequences, stated so harvest screens can rely on them:
//
//   * indirect_gap = conduction_band_min - valence_band_max over ALL
//     k-points (zero-weight band-structure points included; the kernel
//     fills them from the same global step function). It is reported
//     UNCLAMPED: any fractional Fermi group forces indirect_gap <= 0, and
//     integer-occupation band overlap measures its true negative value --
//     a converged metal / zero-gap row always carries an explicit
//     0.0-or-negative measured number, never an absent one. At finite
//     smearing temperature T, states within roughly T*ln(2/t) of the
//     Fermi level are fractional under this threshold, so
//     indirect_gap <= 0 also reads "gapless at the smearing temperature
//     actually used".
//   * direct_gap = min over k of the per-k gap (conduction_min_per_k -
//     valence_max_per_k), with direct_gap_k the first k attaining it.
//     Always >= indirect_gap.
//   * NaN (with k index -1, and NaN per-k entries) appears ONLY when a
//     class is empty in the model space: no occupied state (zero
//     electrons) or no conduction-capable state (completely filled
//     valence-minimal basis). NaN never encodes a metal. On k-route
//     results the fields additionally stay NaN when the SCF did not
//     converge -- an unconverged diagnostics record measured nothing.
//
// Ties on equal energies resolve to the first k in mesh order.
//
// WHAT THE GAP MEANS AT A FRACTIONALLY OCCUPIED DEGENERATE EDGE. This is
// the one case where reasonable conventions disagree, so it is stated
// outright. When the global Fermi level cuts a degenerate manifold, every
// state in that manifold carries the same fractional occupation and is
// therefore both valence and conduction here: valence_band_max is the
// manifold's top, conduction_band_min its bottom, and indirect_gap is the
// NEGATIVE of the manifold spread -- i.e. <= 0, and zero exactly when the
// manifold is exactly degenerate. This is the physically correct value: a
// fractional occupation means E_F is pinned INSIDE a partially filled
// manifold, so the gap is zero, and the manifolds are degenerate to
// machine precision (measured spread <= 8.7e-15 Ha over 1,668 k-route
// rows; 1.9e-16 Ha on the diamond-Si 3x3x3 gate below).
//
// Two other quantities are frequently confused with it. Both are
// available here so that no consumer has to re-derive either one, and
// neither is ever called "gap":
//
//   * POOLED AUFBAU (the Gamma route's own convention: sort all states,
//     fill the lowest n_occ*n_k, report eps[n_fill] - eps[n_fill-1]).
//     This agrees with indirect_gap to machine precision on every
//     zero-temperature mesh row -- verified equal to 1.9e-16 Ha on the
//     degenerate diamond-Si edge -- because filling the globally lowest
//     states is exactly what the occupation kernel did. The two diverge
//     only where an independent per-k fill produces genuine band overlap
//     (the band-path route), which pooled aufbau cannot express at all:
//     it is >= 0 by construction. indirect_gap reports the true negative
//     value there, per CLAUDE.md section 7.
//   * gap_above_fermi_manifold = min(eps | n <= t) - max(eps | n > t),
//     the OCCUPANCY-PARTITION quantity. A fractionally occupied orbital
//     counts as occupied for the upper edge and is excluded as a lower-
//     edge candidate, so this steps over the whole partially filled
//     manifold and is strictly POSITIVE at a fractional edge. It is a
//     legitimate quantity -- "distance to the next entirely empty state"
//     -- but it is NOT the band gap, and reporting it as one is the
//     defect measured by the 2026-08-27 sec8-r4 judge: 1,022 of 1,668
//     k-route rows had a fractional edge, 1,022/1,022 reported a strictly
//     positive gap, 0/1,022 had a true gap > 1e-9, median error 8.06e-5
//     Ha and maximum 1.98e-2 Ha (0.539 eV). It is exposed here under its
//     own name precisely so it can never again be mistaken for the gap.
//
// is_metallic makes the gapless case STRUCTURALLY visible rather than
// leaving a screen to infer it from a small number: it is true exactly
// when the Fermi level is pinned inside a partially filled manifold (some
// occupation is fractional) or the measured indirect_gap is <= 0. Note
// that under a zero-temperature global aufbau fill the pooled gap can
// never be negative, so a "negative gap" screen alone is VACUOUS on mesh
// rows (the judge measured 0 negative gaps in 1,625 rows -- a tautology,
// not evidence). Screen on is_metallic, not on the sign of a float.
//
// The occupation convention is chosen because it is what the rest of the
// tree already means by "gap" (periodic_runner._band_summary), because it
// matches the Gamma route's pooled-aufbau semantics to machine precision
// so duality rungs compare like with like, and because CLAUDE.md section
// 7 requires an inverted band ordering to be surfaced rather than clamped
// away. A consumer screening "converged AND gap AND energy" reads
// gap <= 0 (or is_metallic) as "metallic / gapless as computed", never as
// a defect flag.
struct KPointBandEdges {
    double valence_band_max =
        std::numeric_limits<double>::quiet_NaN();
    double conduction_band_min =
        std::numeric_limits<double>::quiet_NaN();
    double indirect_gap = std::numeric_limits<double>::quiet_NaN();
    double direct_gap = std::numeric_limits<double>::quiet_NaN();
    // Distance to the next entirely empty state (occupancy partition).
    // NOT the band gap -- see the note above.
    double gap_above_fermi_manifold =
        std::numeric_limits<double>::quiet_NaN();
    // E_F pinned inside a partially filled manifold, or a measured
    // indirect_gap <= 0. The structural gapless flag; screen on this.
    bool is_metallic = false;
    int valence_band_max_k = -1;
    int conduction_band_min_k = -1;
    int direct_gap_k = -1;
    std::vector<double> valence_max_per_k;
    std::vector<double> conduction_min_per_k;
};

KPointBandEdges compute_kpoint_band_edges(
    const std::vector<Eigen::VectorXd>& eps_per_k,
    const std::vector<Eigen::VectorXd>& occupations_per_k,
    double occupation_tolerance = 1.0e-8);

// Copy a computed edge set onto a k-route result object carrying the
// matching flat fields (KPointDFTB0Result, KPointSCCDFTBResult,
// KPointGFN2Result). The fields stay flat on the results per the existing
// result-struct style (issue #344 pattern).
template <typename Result>
void store_kpoint_band_edges(Result& result, const KPointBandEdges& edges) {
    result.valence_band_max = edges.valence_band_max;
    result.conduction_band_min = edges.conduction_band_min;
    result.indirect_gap = edges.indirect_gap;
    result.direct_gap = edges.direct_gap;
    result.gap_above_fermi_manifold = edges.gap_above_fermi_manifold;
    result.is_metallic = edges.is_metallic;
    result.valence_band_max_k = edges.valence_band_max_k;
    result.conduction_band_min_k = edges.conduction_band_min_k;
    result.direct_gap_k = edges.direct_gap_k;
    result.valence_max_per_k = edges.valence_max_per_k;
    result.conduction_min_per_k = edges.conduction_min_per_k;
}

void validate_kpoint_occupation_options(
    const KPointOccupationOptions& options,
    const char* caller);

// Issue #434. Throw when a T = 0 fill cut a frontier it cannot resolve.
//
// Call this on the ACCEPTED spectrum only -- after an SCC has converged, or
// on a single-shot non-self-consistent fill -- and NEVER on an intermediate
// iterate: a run whose transient frontier is near-degenerate but whose
// converged frontier is well gapped must still converge. No-op when the
// result carries no unresolved cut.
void reject_unresolved_frontier_cut(
    const KPointOccupationResult& occupation,
    const KPointOccupationOptions& options,
    const char* who);

KPointOccupationResult compute_closed_shell_kpoint_occupations(
    const std::vector<Eigen::VectorXd>& eps_per_k,
    const std::vector<double>& weights,
    double n_electrons_per_cell,
    int n_occ_each,
    const KPointOccupationOptions& options = {});

// Single-spectrum Gamma occupation convention for the periodic Gamma-point
// DFTB0/SCC-DFTB drivers (issue #339). Hard Aufbau, except when the integer
// frontier cuts a roundoff-degenerate manifold: the whole manifold then
// receives equal fractional occupations (the Weinert & Davenport T -> 0
// ensemble, Phys. Rev. B 45, 13709 (1992)), so the density, energy-weighted
// density, and every analytic derivative are unique and symmetry-preserving.
// A non-degenerate frontier returns the exact hard-Aufbau vector.
//
// `max_occupation` is the full occupancy of one spatial orbital: 2.0 for a
// closed-shell spectrum (the default), 1.0 for one spin channel of an
// unrestricted spectrum (issue #439), where each channel is occupied
// independently and the same degeneracy argument applies per spin.
Eigen::VectorXd gamma_degenerate_frontier_occupations(
    const Eigen::VectorXd& eps,
    int n_occ,
    double energy_tolerance = -1.0,
    double max_occupation = 2.0);

}  // namespace semiempirical
}  // namespace vibeqc
