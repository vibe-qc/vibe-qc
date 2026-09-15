"""Γ-CCM fail-closed guards: shell state and double hybrids (2026-08-23).

Two wrong-answer defects found by the capability audit and fixed here.
Both were reproduced live before the fix; the numbers below are those
measurements, kept so a regression is recognizable rather than merely red.

1. **Silent triplet-as-closed-shell.** The closed-shell drivers checked
   electron PARITY only, so an even-electron cluster declaring
   ``multiplicity > 1`` was solved as a closed shell. On H₂/sto-3g
   (2 e⁻, mult 3, 8-bohr box) that returned ``run_ccm_rhf``
   −1.0491709020 and ``run_ccm_rks(pbe)`` −1.1023577337, where the
   shell-aware ``run_ccm_scf`` dispatches to UHF and gives
   −0.7749672190 on ``route="four-center"`` (the 2014 union-and-weight
   lineage) and −0.7827607825 on ``route="real-gamma"`` (the neutral
   construction's direct producer).
   The route dispatcher grew this guard in 2026-07-15/16; the per-method
   entry points had not.

2. **Double hybrids failing open.** ``run_ccm_rks(ccm, "b2plyp")``
   returned −1.1446213910 — exactly the hybrid-KS SCF half of the true
   b2plyp −1.1490167438 — with no raise and no warning, i.e. a number
   that is not the requested functional.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.dft import run_ccm_rks, run_ccm_uks
from vibeqc.periodic.ccm.direct import run_ccm_uhf_direct
from vibeqc.periodic.ccm.route import run_ccm_scf
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable
from vibeqc.periodic.ccm.uhf import run_ccm_uhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _triplet_ccm():
    """Even-electron OPEN shell: 2 e⁻, multiplicity 3."""
    return CCMSystem(
        PeriodicSystem(3, np.diag([8.0, 8.0, 8.0]),
                       [Atom(1, [4.0, 4.0, 3.0]), Atom(1, [4.0, 4.0, 5.0])],
                       0, 3), (1, 1, 1), "sto-3g")


def _singlet_ccm():
    return CCMSystem(
        PeriodicSystem(3, np.diag([6.0, 6.0, 6.0]),
                       [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])],
                       0, 1), (1, 1, 1), "sto-3g")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.mark.parametrize("driver,label", [
    (lambda c: run_ccm_rhf(c), "run_ccm_rhf"),
    (lambda c: run_ccm_rhf_scalable(c), "run_ccm_rhf_scalable"),
    (lambda c: run_ccm_rks(c, "pbe"), "run_ccm_rks"),
])
def test_closed_shell_drivers_refuse_even_electron_triplet(driver, label):
    """The defect: an even-electron triplet must not be solved closed-shell."""
    with pytest.raises(ValueError, match="multiplicity"):
        _quiet(driver, _triplet_ccm())


def test_shell_aware_route_and_open_shell_drivers_still_work():
    """The guard is scoped: the route dispatcher and the open-shell drivers
    handle the same triplet. The paper-facing ``route="gamma"`` spelling has
    failed closed since ruling R1 (issue #222), so the route assertions name
    the admissible producers explicitly: the 2014 four-centre lineage
    (route −0.7749672190 via UHF) and the neutral construction's direct
    producer ``real-gamma`` (−0.7827607825, dispatched to UHF-direct)."""
    ccm = _triplet_ccm()
    via_four_center = _quiet(run_ccm_scf, ccm, route="four-center")
    assert via_four_center.energy == pytest.approx(-0.7749672190, abs=1e-6)

    via_real_gamma = _quiet(run_ccm_scf, ccm, route="real-gamma")
    assert via_real_gamma.converged
    assert via_real_gamma.energy == pytest.approx(-0.7827607825, abs=1e-6)
    # Open-shell dispatch assertion: the route reaches the unrestricted
    # direct producer, not the closed-shell driver (which would return the
    # −1.0491709020 triplet-as-singlet value), and not the 2014 lineage.
    assert type(via_real_gamma).__name__ == "CCMUHFResult"
    assert via_real_gamma.energy == pytest.approx(
        _quiet(run_ccm_uhf_direct, ccm).energy, abs=1e-12, rel=0.0
    )
    assert _quiet(run_ccm_uhf, ccm).converged
    assert _quiet(run_ccm_uks, ccm, "pbe").converged


def test_closed_shell_drivers_still_run_a_singlet():
    """No regression on the ordinary closed-shell path."""
    ccm = _singlet_ccm()
    assert _quiet(run_ccm_rhf, ccm).converged
    assert _quiet(run_ccm_rhf_scalable, ccm).converged
    assert _quiet(run_ccm_rks, ccm, "pbe0").converged


def test_odd_electron_cluster_still_refused_with_a_pointer():
    """The pre-existing parity guard survives and now names the sibling."""
    odd = CCMSystem(
        PeriodicSystem(3, np.diag([7.0, 7.0, 7.0]),
                       [Atom(3, [3.5, 3.5, 3.5])], 0, 2), (1, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="run_ccm_uhf|electrons"):
        _quiet(run_ccm_rhf, odd)


@pytest.mark.parametrize("driver,ccm_fn", [
    (lambda c: run_ccm_rks(c, "b2plyp"), _singlet_ccm),
    (lambda c: run_ccm_uks(c, "b2plyp"), _triplet_ccm),
])
def test_ks_drivers_refuse_double_hybrids(driver, ccm_fn):
    """A driver that computes only the SCF half must not return it under the
    double hybrid's name; the error points at the composed route."""
    with pytest.raises(NotImplementedError, match="double hybrid"):
        _quiet(driver, ccm_fn())


def test_plain_hybrids_and_pure_functionals_unaffected():
    """The double-hybrid guard does not catch ordinary functionals."""
    ccm = _singlet_ccm()
    for functional in ("pbe", "pbe0", "b3lyp"):
        assert _quiet(run_ccm_rks, ccm, functional).converged
