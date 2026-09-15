"""EXPERIMENTAL: DLPNO + WSC-axis parallelism on the real-Γ direct route.

Two capability gates (2026-08-21):

1. **DLPNO on the BvK-ewald direct reference.** The cyclic cluster is the
   natural home for DLPNO — Coulomb decays with distance, so pairs screen
   on the minimum-image (BvK-torus) centroid distance. The DLPNO machinery
   exists (A-line follow-on chat); what this route adds is the one-call
   entry that guarantees SCF-and-cderi come from one kernel, plus the
   ``reference="direct"`` option so the correlation does not inherit the
   strict-zero reference distortion that GROWS with cluster size
   (HANDOVER_D2_EXXDIV.md).

2. **WSC/q-axis parallelism of the fold** (``fold_threads``), the CCM
   analogue of k-point parallelism (CLAUDE.md § 15). Measured 2026-08-21
   on H₂ (3,3,1) with BLAS pinned: 67.6 s -> 17.1 s (3.95x) plain,
   25.0 s -> 6.1 s (4.11x) with the symmetry star reduction, both
   **bit-identical** to serial.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    run_ccm_dlpno_mp2_direct,
    run_ccm_rhf_direct,
)
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi, ccm_neutral_cderi_fold

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2_ccm(nrep=(2, 1, 1)):
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# --- 1. WSC/q-axis parallelism ----------------------------------------------


@pytest.mark.parametrize("symmetry", [None, True])
def test_threaded_fold_is_bit_identical(symmetry):
    """Threading the per-(k_a, k_b) fits of each q-channel changes nothing
    elementwise — including the symmetry path, where star MEMBERS
    reconstruct from representatives and the serial sweep's
    representative-before-member ordering must be preserved (builds are
    batched per channel before any reconstruction, which is what makes
    this safe)."""
    ccm = _h2_ccm((2, 2, 1))
    serial = _quiet(ccm_neutral_cderi_fold, ccm, symmetry=symmetry,
                    fold_threads=1)
    threaded = _quiet(ccm_neutral_cderi_fold, ccm, symmetry=symmetry,
                      fold_threads=4)
    assert serial.shape == threaded.shape
    assert np.array_equal(serial, threaded)


def test_fold_thread_count_resolution(monkeypatch):
    """Thread-count policy: default serial (opt-in — an outer pool on top of
    a multi-threaded BLAS oversubscribes), env override, 0 = all cores, and
    never more threads than cells."""
    from vibeqc.periodic.ccm.neutral import _fold_thread_count

    monkeypatch.delenv("VIBEQC_CCM_FOLD_THREADS", raising=False)
    assert _fold_thread_count(None, 8) == 1          # opt-in default
    monkeypatch.setenv("VIBEQC_CCM_FOLD_THREADS", "4")
    assert _fold_thread_count(None, 8) == 4          # env honoured
    assert _fold_thread_count(2, 8) == 2             # explicit wins
    assert _fold_thread_count(99, 8) == 8            # capped at n_cells
    assert _fold_thread_count(0, 8) >= 1             # 0 = all cores


# --- 2. DLPNO on the direct reference ---------------------------------------


@pytest.mark.parametrize("reference", ["direct", "neutral"])
def test_dlpno_no_truncation_equals_canonical(reference):
    """At zero truncation DLPNO-MP2 reproduces the canonical RI-MP2
    correlation of the SAME reference to machine precision (measured 4e-17
    neutral / 3e-17 direct) — the no-truncation identity that makes the
    locality screens meaningful when they are switched on."""
    ccm = _h2_ccm((2, 1, 1))
    d = _quiet(run_ccm_dlpno_mp2_direct, ccm, reference=reference)
    # Canonical comparison on the same reference + same kernel.
    L = _quiet(ccm_neutral_cderi_fold, ccm)
    if reference == "direct":
        scf = _quiet(run_ccm_rhf_direct, ccm, cderi=L)
    else:
        from vibeqc.periodic.ccm.ri import run_ccm_rhf_ri_neutral
        scf = _quiet(run_ccm_rhf_ri_neutral, ccm, cderi=L)
    m = _quiet(run_ccm_mp2, ccm, scf, cderi=L)
    assert d.e_corr == pytest.approx(m.e_correlation, abs=1e-12)
    for field in ("requested", "effective", "transport"):
        assert getattr(d.guess_selection, field) == getattr(scf.guess_selection, field)
        assert getattr(m.guess_selection, field) == getattr(scf.guess_selection, field)
    assert d.guess_selection.effective.name == "HCORE"
    assert d.n_pairs > 0


def test_dlpno_reference_choice_moves_correlation():
    """The reference is not a detail: the exchange-q=0 seam shifts every
    occupied eigenvalue, hence every MP2 denominator. Measured on compact
    H₂ (2,1,1): -25.9 mHa (neutral, strict-zero) vs -19.4 mHa (direct,
    BvK-ewald) — a 25% swing on a fixture where the D2 distortion is at its
    mildest, and it GROWS with cluster size. This gate pins that the two
    references genuinely differ, so a silent default change would be
    caught."""
    ccm = _h2_ccm((2, 1, 1))
    d_dir = _quiet(run_ccm_dlpno_mp2_direct, ccm, reference="direct")
    d_neu = _quiet(run_ccm_dlpno_mp2_direct, ccm, reference="neutral")
    assert d_dir.e_corr != pytest.approx(d_neu.e_corr, rel=1e-3)
    assert d_dir.e_corr > d_neu.e_corr        # strict-zero over-correlates


def test_dlpno_direct_boundary():
    """Boundary: unknown reference and open shells fail closed."""
    ccm = _h2_ccm((2, 1, 1))
    with pytest.raises(ValueError, match="reference"):
        _quiet(run_ccm_dlpno_mp2_direct, ccm, reference="bogus")
    odd = CCMSystem(
        PeriodicSystem(3, np.diag([7.0, 7.0, 7.0]),
                       [Atom(3, [3.5, 3.5, 3.5])], 0, 2), (1, 1, 1), "sto-3g")
    with pytest.raises(NotImplementedError, match="closed-shell"):
        _quiet(run_ccm_dlpno_mp2_direct, odd)


# --- 3. Open-shell and CCSD(T) siblings (2026-08-22) -------------------------


def _li_doublet_ccm():
    return CCMSystem(
        PeriodicSystem(3, np.diag([7.0, 7.0, 7.0]),
                       [Atom(3, [3.5, 3.5, 3.5])], 0, 2), (1, 1, 1), "sto-3g")


def test_dlpno_ump2_direct_equals_canonical_and_moves_with_reference():
    """Open-shell one-call sibling: at zero truncation DLPNO-UMP2 equals
    canonical UMP2 on the same reference (measured 2e-19), and the per-spin
    exchange-q=0 seam moves the correlation by ~36% between references
    (-2.814e-4 direct vs -2.071e-4 neutral). Open-shell ionics are where
    the reference choice bites hardest -- the seam enters each spin
    channel's denominators."""
    from vibeqc.periodic.ccm.direct import (
        run_ccm_dlpno_ump2_direct,
        run_ccm_uhf_direct,
    )
    from vibeqc.periodic.ccm.ump2 import run_ccm_ump2

    ccm = _li_doublet_ccm()
    d = _quiet(run_ccm_dlpno_ump2_direct, ccm)
    L = _quiet(ccm_neutral_cderi_fold, ccm)
    uhf = _quiet(run_ccm_uhf_direct, ccm, cderi=L)
    ref = _quiet(run_ccm_ump2, ccm, uhf, cderi=L)
    assert d.e_corr == pytest.approx(ref.e_correlation, abs=1e-12)

    neutral = _quiet(run_ccm_dlpno_ump2_direct, ccm, reference="neutral")
    assert d.e_corr != pytest.approx(neutral.e_corr, rel=1e-3)
    assert d.e_corr < neutral.e_corr    # strict-zero under-correlates here


@pytest.mark.slow
def test_dlpno_ccsd_direct_equals_canonical_ccsd():
    """CCSD(T) through the same one-call contract: the DLPNO-CCSD half
    equals canonical CCSD on the same reference at zero truncation
    (measured 9e-17), and the reference swing is ~30% (-0.02276 direct vs
    -0.03237 neutral). ((T) is ~1e-34 on this 2-electron fixture -- no
    triples exist there; the value is exercised, not gated.)"""
    from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
    from vibeqc.periodic.ccm.direct import run_ccm_dlpno_ccsd_direct

    ccm = _h2_ccm((2, 1, 1))
    c = _quiet(run_ccm_dlpno_ccsd_direct, ccm)
    assert c.converged
    L = _quiet(ccm_neutral_cderi_fold, ccm)
    scf = _quiet(run_ccm_rhf_direct, ccm, cderi=L)
    ref = _quiet(run_ccm_ccsd, ccm, scf, cderi=L)
    assert c.e_ccsd_correlation == pytest.approx(ref.e_correlation, abs=1e-12)

    neutral = _quiet(run_ccm_dlpno_ccsd_direct, ccm, reference="neutral")
    assert c.e_ccsd_correlation != pytest.approx(
        neutral.e_ccsd_correlation, rel=1e-3)


def test_dlpno_direct_shell_routing_boundary():
    """The closed/open-shell entry points refuse each other's clusters
    rather than silently mis-treating the spin state (the even-electron
    triplet trap this route already fixed once at SCF level)."""
    from vibeqc.periodic.ccm.direct import (
        run_ccm_dlpno_mp2_direct,
        run_ccm_dlpno_ump2_direct,
    )

    with pytest.raises(NotImplementedError, match="closed-shell"):
        _quiet(run_ccm_dlpno_mp2_direct, _li_doublet_ccm())
    with pytest.raises(ValueError, match="open-shell"):
        _quiet(run_ccm_dlpno_ump2_direct, _h2_ccm((2, 1, 1)))
