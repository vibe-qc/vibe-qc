"""T=0 k-point fills must refuse a frontier they cannot resolve (#434).

At T = 0 with hard Aufbau on a k-point mesh, a frontier pair split by more
than the roundoff-equality tolerance but far less than a real gap is CUT by
the fill: the lower state is filled, the upper left empty.  Which of the two
is lower is then decided by last-ulp arithmetic, which differs between hosts
and between builds, so an entire electron moves with a rounding difference --
and the run reports convergence either way.  #424 documented exactly that as
cross-host ``converged=True at n_iter=1`` versus ``converged=False at
n_iter=500`` branch flips on one shared binary.

Maintainer ruling, 2026-08-28 on #434: *"Force finite-T for metallic
frontiers.  Do not accept the T=0 near-degenerate race."*  A race that can
converge to a wrong occupation and report success is what CLAUDE.md § 7
forbids treating as a convergence nuisance.

THE FIXTURE IS CONSTRUCTED, DELIBERATELY.  The issue records that the class
has no natural system, and a scan confirms it: over Z = 5 and Z = 31 zigzag
chains at a = 3-9 bohr, meshes 2/4/6 and a 30-bohr cutoff, the smallest cut
frontier gap
reachable is 2.0e-3 Ha (B, a = 9, mesh 2), three orders of magnitude ABOVE
the 1e-6 default.  Real systems therefore do not trip this guard, which is
the point of the margin tests below.  Widening the chain drives the gap down
smoothly and lands it inside the window, which is what makes a behavioural
fail-first possible:

    Z31 zigzag, mesh 2, T = 0, 30-bohr cutoff, measured pooled frontier gap
        a = 36.0 bohr -> 1.2145e-05 Ha     a = 45.0 -> 1.1614e-06
        a = 47.0      -> 6.7763e-07        a = 50.0 -> 2.9718e-07
        offset 0      -> exactly degenerate

The family and the scan were measured again under the #373 basis and the
#478 hopping (#565).  The earlier family ran at a = 9-16 bohr with a 12-bohr
cutoff.  There the Ga S(k) is now indefinite, and at a = 14-20 bohr with a
30-bohr cutoff the gap is 0.04-0.07 Ha.  The gaps above are the same at a
40-bohr cutoff to four digits.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import Atom, PeriodicSystem, monkhorst_pack
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.parameters import default_parameters

_CUTOFF = 30.0  # positive-definite Ga S(k); 12 bohr is not since #373
_Z = 31  # Ga: the #424 wave-flip element

# Measured on this fixture family (see the module docstring).
_GAP_WELL_ABOVE = 36.0   # 1.21e-05 Ha -- 12x above the 1e-6 default
_GAP_JUST_ABOVE = 45.0   # 1.16e-06 Ha -- above by 16 %
_GAP_JUST_BELOW = 47.0   # 6.78e-07 Ha -- below by 32 %
_GAP_WELL_BELOW = 50.0   # 2.97e-07 Ha


def _chain(a: float, offset: float = 0.35) -> PeriodicSystem:
    """Two-site homonuclear zigzag chain, the #424 geometry.

    ``offset`` is the transverse displacement of the second site.  At 0.35 it
    is the #424 cell; at exactly 0 the two sites become translation-equivalent
    and the frontier pair is exactly degenerate.
    """
    return PeriodicSystem(
        1,
        np.diag([a, 30.0, 30.0]),
        [Atom(_Z, [0.0, 0.0, 0.0]), Atom(_Z, [a / 2.0, offset, 0.0])],
        0,
        1,
    )


def _dftb0(a: float, *, offset: float = 0.35, mesh: int = 2,
           temperature: float = 0.0, min_gap: float | None = None):
    system = _chain(a, offset)
    options = _se.KPointOccupationOptions()
    options.smearing_temperature = temperature
    if min_gap is not None:
        options.min_resolvable_frontier_gap = min_gap
    return _se.run_dftb0_kpoints(
        system, default_parameters(), monkhorst_pack(system, (mesh, 1, 1)),
        _CUTOFF, options)


def _frontier_gap(result) -> float:
    """Pooled T=0 frontier gap actually realised, or nan if not cut."""
    eps = np.concatenate([np.asarray(e) for e in result.eps_per_k])
    occ = np.concatenate([np.asarray(o) for o in result.occupations_per_k])
    if np.any((occ > 1e-8) & (occ < 2.0 - 1e-8)):
        return float("nan")  # equalized fractional group, nothing was cut
    filled, empty = eps[occ > 1e-8], eps[occ <= 1e-8]
    if not filled.size or not empty.size:
        return float("nan")
    return float(empty.min() - filled.max())


# --------------------------------------------------------------------------
# The defect: a cut the fill cannot resolve is refused, not silently returned.
# --------------------------------------------------------------------------

def test_unresolvable_frontier_is_refused():
    """The behavioural fail-first. Before the fix this RETURNED a result --
    a raced one -- and reported success."""
    with pytest.raises(RuntimeError, match="cannot resolve"):
        _dftb0(_GAP_WELL_BELOW)


def test_the_refusal_is_actionable():
    """A refusal a user cannot act on is a worse failure than the race."""
    with pytest.raises(RuntimeError) as excinfo:
        _dftb0(_GAP_WELL_BELOW)
    message = str(excinfo.value)
    assert "run_dftb0_kpoints" in message          # which entry point
    assert "smearing_temperature" in message       # the prescribed remedy
    assert "min_resolvable_frontier_gap" in message  # the documented opt-out
    assert " Ha" in message and " eV" in message   # the measured gap, both units


def test_finite_temperature_is_the_prescribed_remedy():
    """The maintainer ruling is 'force finite-T', so finite T must work."""
    result = _dftb0(_GAP_WELL_BELOW, temperature=0.005)
    occ = np.concatenate([np.asarray(o) for o in result.occupations_per_k])
    # Smearing shares the frontier instead of cutting it.
    assert np.any((occ > 1e-8) & (occ < 2.0 - 1e-8))


def test_the_guard_can_be_disabled_explicitly():
    """0 restores the old unstable fill -- documented, and opt-in only."""
    result = _dftb0(_GAP_WELL_BELOW, min_gap=0.0)
    assert 0.0 < _frontier_gap(result) <= 1.0e-6


# --------------------------------------------------------------------------
# What must NOT change.  A guard that fires too often is its own defect.
# --------------------------------------------------------------------------

def test_a_real_frontier_gap_is_untouched():
    """1.21e-05 Ha, 12x the default: a cut the fill resolves must run
    exactly as before."""
    result = _dftb0(_GAP_WELL_ABOVE)
    assert _frontier_gap(result) == pytest.approx(1.2145e-05, rel=1e-3)


def test_an_exactly_degenerate_frontier_is_shared_not_refused():
    """The equalized T->0 ensemble is correct and deterministic -- it is what
    #424 relies on -- so it must not be caught by a guard aimed at cuts."""
    result = _dftb0(_GAP_WELL_ABOVE, offset=0.0)
    occ = np.concatenate([np.asarray(o) for o in result.occupations_per_k])
    assert np.any((occ > 1e-8) & (occ < 2.0 - 1e-8))
    assert np.isnan(_frontier_gap(result))


@pytest.mark.parametrize("a,expect_refusal", [
    (_GAP_JUST_ABOVE, False),  # 1.16e-06: above threshold by 16 %
    (_GAP_JUST_BELOW, True),   # 6.78e-07: below threshold by 32 %
])
def test_the_threshold_is_where_it_claims_to_be(a, expect_refusal):
    """Straddle 1e-6 from both sides with one fixture family, so the boundary
    is pinned rather than assumed."""
    if expect_refusal:
        with pytest.raises(RuntimeError, match="cannot resolve"):
            _dftb0(a)
    else:
        assert _frontier_gap(_dftb0(a)) > 1.0e-6


def test_the_predicate_reads_the_option_not_a_hardcoded_constant():
    """Raising the tolerance must refuse a system the default admits; if this
    passed while the previous tests also passed, the guard could still be
    comparing against a literal."""
    _dftb0(_GAP_WELL_ABOVE)  # admitted at the 1e-6 default
    with pytest.raises(RuntimeError, match="cannot resolve"):
        _dftb0(_GAP_WELL_ABOVE, min_gap=1.0e-4)


def test_an_invalid_tolerance_is_rejected():
    with pytest.raises(ValueError, match="min_resolvable_frontier_gap"):
        _dftb0(_GAP_WELL_ABOVE, min_gap=-1.0)
