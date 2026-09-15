"""The T = 0 global-Aufbau fill reports frontiers it cannot resolve (#543).

`vibeqc.smearing.apply._global_aufbau_with_mu` is the Python twin of the C++
`compute_closed_shell_kpoint_occupations`: same pooled sort, same
`256*eps*energy_scale` roundoff grouping, same fill / cut / fractional branch
structure -- and, before this change, the same silent defect. It backs the
ab-initio multi-k routes (periodic_rhf_gdf, periodic_k_gdf,
periodic_gapw_augment).

When the occupied/empty boundary falls between two states closer than the fill
can resolve, the fill CUTS them: one filled, one empty. Which is "lower" is
then decided by last-ulp arithmetic, so an entire electron moves between
k-points with a rounding difference that varies between hosts -- while the SCF
still reports convergence. Issue #424 documented exactly that as cross-host
`converged=True at n_iter=1` versus `converged=False at n_iter=500` flips on
one shared binary; #434 fixed the C++ side.

The measured behaviour that motivated this, from the issue (two k-points, one
band each, half filled, split by `delta`):

      delta (Ha)   occ(k0)  occ(k1)        kind   flips when delta changes sign
        1.00e-14    1.0000   1.0000  fractional   no
        1.00e-13    2.0000   0.0000         CUT   YES
        3.20e-07    2.0000   0.0000         CUT   YES     <-- the #424 splitting
        1.00e-03    2.0000   0.0000         CUT   YES
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.smearing import (
    MIN_RESOLVABLE_FRONTIER_GAP,
    apply_smearing,
    unresolved_frontier_cut,
)


def _two_k(delta: float, eps0: float = 0.5):
    """Two k-points, one band each, half filled: exactly one state occupied."""
    return [np.array([eps0]), np.array([eps0 + delta])]


def _fill(delta: float):
    return apply_smearing(
        _two_k(delta), weights=[0.5, 0.5], n_electrons_per_cell=1.0,
        n_occ_each=1, smearing=None,
    )


# --------------------------------------------------------------------------
# The two kernels must not drift apart.
# --------------------------------------------------------------------------

def test_the_python_and_cpp_thresholds_are_the_same_number():
    """#543's third closure criterion, pinned mechanically rather than by a
    comment: the ab-initio and semiempirical multi-k fills are independent
    implementations of ONE convention, and a reader must not be able to change
    one without this failing."""
    from vibeqc._vibeqc_core import semiempirical as _se

    assert MIN_RESOLVABLE_FRONTIER_GAP == (
        _se.KPointOccupationOptions().min_resolvable_frontier_gap
    )


# --------------------------------------------------------------------------
# The defect, at the splitting that actually raced in production.
# --------------------------------------------------------------------------

def test_the_issue_424_splitting_is_reported():
    """3.2e-07 Ha is the exact pre-#316 frontier splitting #424 measured. The
    fill still cuts it -- that is aufbau -- but it no longer does so silently."""
    result = _fill(3.2e-07)
    assert result.frontier_cut_unresolved
    assert result.frontier_gap == pytest.approx(3.2e-07, rel=1e-9)


def test_the_cut_occupation_really_does_follow_an_ulp():
    """The reason this is a defect and not a preference: flipping the SIGN of a
    3.2e-07 Ha splitting moves an entire electron from one k-point to the
    other. Both fills report success."""
    forward = _fill(3.2e-07).occupations_per_k
    reversed_ = _fill(-3.2e-07).occupations_per_k
    assert float(forward[0][0]) == 2.0 and float(forward[1][0]) == 0.0
    assert float(reversed_[0][0]) == 0.0 and float(reversed_[1][0]) == 2.0


@pytest.mark.parametrize("delta", [1.0e-13, 1.0e-9, 3.2e-07])
def test_cuts_at_or_below_the_threshold_are_reported(delta):
    assert _fill(delta).frontier_cut_unresolved


def test_the_boundary_is_inclusive_and_is_tested_where_it_can_be_exact():
    """The threshold comparison is ``gap <= tolerance``, and the boundary is
    pinned HERE rather than through :func:`_fill`.

    Going through the fill cannot express the boundary: ``0.5 + 1e-6`` is not
    representable, so the realised gap is 1.0000000000287557e-06 and lands
    just ABOVE the threshold. That is correct behaviour on a gap that really
    is larger than 1e-6, but it makes the fill useless for pinning inclusivity.
    Handing the predicate exact values does pin it."""
    occ = [np.array([2.0]), np.array([0.0])]
    exact = [np.array([0.0]), np.array([MIN_RESOLVABLE_FRONTIER_GAP])]
    assert unresolved_frontier_cut(exact, occ) == pytest.approx(
        MIN_RESOLVABLE_FRONTIER_GAP, rel=1e-12
    )
    just_above = [
        np.array([0.0]),
        np.array([np.nextafter(MIN_RESOLVABLE_FRONTIER_GAP, np.inf)]),
    ]
    assert unresolved_frontier_cut(just_above, occ) is None


@pytest.mark.parametrize("delta", [1.0e-5, 1.0e-3, 0.1])
def test_a_real_gap_is_not_reported(delta):
    """A guard that fires on real gaps would be its own defect."""
    result = _fill(delta)
    assert not result.frontier_cut_unresolved
    assert np.isnan(result.frontier_gap)


# --------------------------------------------------------------------------
# What must NOT be reported: the equalized ensemble #424 depends on.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [0.0, 1.0e-16, 1.0e-14])
def test_an_equalized_degenerate_group_is_not_a_cut(delta):
    """Below the roundoff-equality tolerance the fill SHARES the frontier --
    the deterministic Weinert-Davenport T->0 ensemble. It does not race, and
    reporting it would be wrong."""
    result = _fill(delta)
    occ = np.concatenate(result.occupations_per_k)
    assert np.any((occ > 1e-8) & (occ < 2.0 - 1e-8)), "expected a shared group"
    assert not result.frontier_cut_unresolved
    assert np.isnan(result.frontier_gap)


def test_finite_temperature_never_reports_a_cut():
    """Smearing shares the frontier instead of cutting it, so there is nothing
    to resolve."""
    result = apply_smearing(
        _two_k(3.2e-07), weights=[0.5, 0.5], n_electrons_per_cell=1.0,
        n_occ_each=1, smearing=0.005,
    )
    assert not result.frontier_cut_unresolved


# --------------------------------------------------------------------------
# The predicate itself, exercised directly on its edge cases.
# --------------------------------------------------------------------------

def test_predicate_returns_the_measured_gap_not_just_a_flag():
    gap = unresolved_frontier_cut(_two_k(4.0e-07), [np.array([2.0]), np.array([0.0])])
    assert gap == pytest.approx(4.0e-07, rel=1e-9)


def test_predicate_reads_its_threshold_argument():
    """If this passed while the default-threshold tests also passed, the
    predicate could still be comparing against a literal."""
    spectra, occ = _two_k(1.0e-05), [np.array([2.0]), np.array([0.0])]
    assert unresolved_frontier_cut(spectra, occ) is None
    assert unresolved_frontier_cut(
        spectra, occ, min_resolvable_frontier_gap=1.0e-4
    ) == pytest.approx(1.0e-05, rel=1e-9)


def test_predicate_ignores_a_completely_filled_or_empty_space():
    """No occupied/empty boundary exists, so nothing was cut."""
    spectra = _two_k(1.0e-09)
    assert unresolved_frontier_cut(spectra, [np.array([2.0]), np.array([2.0])]) is None
    assert unresolved_frontier_cut(spectra, [np.array([0.0]), np.array([0.0])]) is None


def test_predicate_is_disabled_by_a_zero_threshold():
    assert unresolved_frontier_cut(
        _two_k(1.0e-09), [np.array([2.0]), np.array([0.0])],
        min_resolvable_frontier_gap=0.0,
    ) is None


def test_predicate_rejects_an_invalid_threshold():
    with pytest.raises(ValueError, match="min_resolvable_frontier_gap"):
        unresolved_frontier_cut(
            _two_k(1e-9), [np.array([2.0]), np.array([0.0])],
            min_resolvable_frontier_gap=-1.0,
        )


def test_predicate_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="k-point for k-point"):
        unresolved_frontier_cut(_two_k(1e-9), [np.array([2.0])])


# --------------------------------------------------------------------------
# Degeneracy grouping is transitive and bounded (#544)
# --------------------------------------------------------------------------
#
# Pre-#544 all three T = 0 kernels grouped by comparing each candidate to the
# group's FIRST member -- a ball, not a chain -- so states each within the
# roundoff tolerance of their neighbour could be split into two groups one
# ulp apart, and a boundary landing there integer-filled one member of an
# effectively degenerate manifold and left its twin empty. On the multi-k
# routes the #434 guard flagged that (frontier_cut_unresolved=True, an
# unresolved 4e-14 Ha cut); the Gamma helper had no guard and returned the
# chopped manifold.

from vibeqc._vibeqc_core import semiempirical as _se_kernels  # noqa: E402
from vibeqc.smearing.apply import (  # noqa: E402
    DEGENERATE_GROUP_SPAN_FACTOR,
    ZERO_TEMPERATURE_DEGENERACY_ULPS,
    _global_aufbau_with_mu,
)

_EPS = np.finfo(float).eps
_TOL_UNIT_SCALE = ZERO_TEMPERATURE_DEGENERACY_ULPS * _EPS   # 5.68e-14 at scale 1
_STEP = 4.0e-14   # below the tolerance: neighbours are indistinguishable


def test_the_grouping_constants_match_the_cpp_kernel():
    assert ZERO_TEMPERATURE_DEGENERACY_ULPS == _se_kernels.zero_temperature_degeneracy_ulps
    assert DEGENERATE_GROUP_SPAN_FACTOR == _se_kernels.degenerate_group_span_factor
    assert _STEP < _TOL_UNIT_SCALE < 2.0 * _STEP   # the fixtures below rely on this


def _chopped_manifold():
    """Three k-points, one band each, at 0, 4e-14, 8e-14 Ha; 4/3 electrons.

    Every neighbour pair is within the tolerance, but the first and last
    members are not: a ball around the first member groups {0, 4e-14} and
    starts a new group at 8e-14, and the electron count (two of three states)
    lands exactly on that split.
    """
    eps = [np.array([0.0]), np.array([_STEP]), np.array([2.0 * _STEP])]
    return eps, [1.0 / 3.0] * 3, 4.0 / 3.0


def test_cpp_kernel_shares_a_chopped_roundoff_manifold():
    """Pre-fix: occupations 2, 2, 0 with frontier_cut_unresolved=True (an
    unresolved 4e-14 Ha cut). Post-fix the whole manifold is one group and
    shares 4/3 each, nothing is flagged."""
    eps, weights, n_el = _chopped_manifold()
    res = _se_kernels.compute_closed_shell_kpoint_occupations(
        eps, weights, n_el, 1, _se_kernels.KPointOccupationOptions())
    occ = np.concatenate([np.asarray(o) for o in res.occupations_per_k])
    assert occ == pytest.approx([4.0 / 3.0] * 3, abs=1e-12), occ
    assert not res.frontier_cut_unresolved
    assert np.isnan(res.frontier_gap)
    assert res.fermi_level == pytest.approx(_STEP, abs=1e-16)


def test_python_twin_shares_a_chopped_roundoff_manifold():
    eps, weights, n_el = _chopped_manifold()
    result = apply_smearing(eps, weights=weights, n_electrons_per_cell=n_el,
                            n_occ_each=1, smearing=None)
    occ = np.concatenate([np.asarray(o) for o in result.occupations_per_k])
    assert occ == pytest.approx([4.0 / 3.0] * 3, abs=1e-12), occ
    assert not result.frontier_cut_unresolved


def test_gamma_helper_extends_the_manifold_transitively():
    """Five orbitals at 0, 4e-14, 8e-14, 1.2e-13, 1.0 Ha, two occupied. The
    frontier (4e-14) and LUMO (8e-14) are degenerate; the fourth orbital is
    4e-14 above the LUMO but 8e-14 above the frontier. Pre-fix the ball
    around the frontier excluded it: 4/3, 4/3, 4/3, 0, 0. Post-fix the
    chain includes it: 1, 1, 1, 1, 0."""
    eps = np.array([0.0, _STEP, 2.0 * _STEP, 3.0 * _STEP, 1.0])
    occ = np.asarray(_se_kernels.gamma_degenerate_frontier_occupations(eps, 2))
    assert occ == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.0], abs=1e-12), occ
    assert occ.sum() == pytest.approx(4.0, abs=1e-12)
    # per spin channel (issue #439 convention) the same manifold shares 0.5
    occ_spin = np.asarray(
        _se_kernels.gamma_degenerate_frontier_occupations(eps, 2, -1.0, 1.0))
    assert occ_spin == pytest.approx([0.5] * 4 + [0.0], abs=1e-12)


def test_gamma_helper_open_gap_is_untouched():
    eps = np.array([-1.0, -0.5, 0.0, 0.5])
    occ = np.asarray(_se_kernels.gamma_degenerate_frontier_occupations(eps, 2))
    assert occ == pytest.approx([2.0, 2.0, 0.0, 0.0])


@pytest.mark.parametrize("target_states", [3, 7, 12, 40, 100])
def test_span_cap_ladder_is_shared_or_flagged_never_silently_chopped(target_states):
    """A 200-state ladder spaced 4e-14 Ha (every neighbour within tolerance,
    total span 8e-12 Ha, far beyond the 16-tolerance cap) must not merge
    into one level, and wherever the cap ends a group the #434 guard must
    still see the sub-tolerance split. Invariant: the fill either shares the
    boundary group (fractional occupations) or reports an unresolved cut.
    A silent integer chop is the defect."""
    n = 200
    eps = [np.array([i * _STEP]) for i in range(n)]
    weights = [1.0 / n] * n
    n_el = 2.0 * target_states / n
    res = _se_kernels.compute_closed_shell_kpoint_occupations(
        eps, weights, n_el, 1, _se_kernels.KPointOccupationOptions())
    occ = np.concatenate([np.asarray(o) for o in res.occupations_per_k])
    fractional = (occ > 1e-9) & (occ < 2.0 - 1e-9)
    assert fractional.any() or res.frontier_cut_unresolved, occ
    # the cap bounds the shared group: span <= 16 tolerances = 22 steps
    max_members = int(DEGENERATE_GROUP_SPAN_FACTOR * _TOL_UNIT_SCALE / _STEP) + 1
    assert fractional.sum() <= max_members, fractional.sum()
    assert occ.sum() / n == pytest.approx(n_el, abs=1e-12)
    # Python twin: same fixture, same verdict
    result = apply_smearing(eps, weights=weights, n_electrons_per_cell=n_el,
                            n_occ_each=1, smearing=None)
    occ_py = np.concatenate([np.asarray(o) for o in result.occupations_per_k])
    assert occ_py == pytest.approx(occ, abs=1e-12)
    assert result.frontier_cut_unresolved == res.frontier_cut_unresolved


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_python_twin_matches_the_cpp_kernel_on_roundoff_clusters(seed):
    """The two kernels are independent implementations of one convention:
    random spectra with roundoff-scale clusters (widths 0..3 tolerances,
    including sub-tolerance neighbour steps) at several fillings must give
    identical occupations, Fermi levels and guard verdicts."""
    rng = np.random.default_rng(seed)
    n_k, n_band = 4, 6
    base = np.sort(rng.uniform(-1.0, 1.0, size=n_band))
    eps = []
    for _ in range(n_k):
        jitter = rng.integers(0, 4, size=n_band) * _STEP * rng.choice([-1, 1], size=n_band)
        eps.append(np.sort(base + jitter))
    weights = rng.uniform(0.5, 1.5, size=n_k)
    weights = list(weights / weights.sum())
    for n_el in (1.0, 2.5, 4.0, 7.0, 9.5):
        res = _se_kernels.compute_closed_shell_kpoint_occupations(
            eps, weights, n_el, n_band, _se_kernels.KPointOccupationOptions())
        occ_cpp = np.concatenate([np.asarray(o) for o in res.occupations_per_k])
        occ_py, mu_py = _global_aufbau_with_mu(eps, weights, n_el)
        occ_py = np.concatenate([np.asarray(o) for o in occ_py])
        assert occ_py == pytest.approx(occ_cpp, abs=1e-12), (seed, n_el)
        assert mu_py == pytest.approx(res.fermi_level, abs=1e-13), (seed, n_el)


def test_predicate_uses_the_true_maximum_occupancy_for_a_bottom_fractional_group():
    """#544 found-while: a shared group with NO fully occupied state below it
    (two of three states carrying 2/3 electrons each, third state 4e-14 Ha
    above) is a fractional fill, not a cut. Inferring the full occupancy from
    the largest value present (2/3) misread it as an integer cut at 4e-14 Ha;
    the true maximum occupancy classifies it correctly."""
    spectra = [np.array([0.0]), np.array([_STEP]), np.array([2.0 * _STEP])]
    occ = [np.array([2.0 / 3.0]), np.array([2.0 / 3.0]), np.array([0.0])]
    assert unresolved_frontier_cut(spectra, occ, max_occupation=2.0) is None
    # without the hint the legacy inference reports the ulp-scale gap
    assert unresolved_frontier_cut(spectra, occ) == pytest.approx(_STEP)
    # the fill itself passes the hint: the same fixture through apply_smearing
    # 4/9 electrons per cell on the same three states: the fill shares the
    # roundoff-degenerate group (4/9 each), and the fill passes the hint, so
    # nothing is reported.
    result = apply_smearing(spectra, weights=[1.0 / 3.0] * 3,
                            n_electrons_per_cell=4.0 / 9.0,
                            n_occ_each=1, smearing=None)
    assert not result.frontier_cut_unresolved
    assert np.concatenate(result.occupations_per_k) == pytest.approx([4.0 / 9.0] * 3)
    with pytest.raises(ValueError, match="max_occupation"):
        unresolved_frontier_cut(spectra, occ, max_occupation=0.0)
