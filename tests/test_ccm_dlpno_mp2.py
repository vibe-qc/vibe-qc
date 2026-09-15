"""DLPNO-MP2 on the cyclic cluster (AICCM Task 2, M2).

Local-correlation DLPNO-MP2 built on the **neutral** four-center reference (the
RI-consistent ``ccm_neutral_cderi``), reusing the validated molecular DLPNO
component functions with ``S^CCM`` / ``F^CCM`` injected.

The hard gate: the no-truncation limit (``tcut_pno = tcut_mkn = 0``) must
reproduce the canonical CCM MP2 on the *same* neutral four-center to ~µHa — for
canonical *and* localized (Pipek–Mezey / Boys) occupieds (the latter exercising
the coupled LMP2 residual with the ``S^CCM``-projected neighbour amplitudes).

Reference: Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013). Derivation /
decision log (neutral reference required for correlation):
docs/aiccm2026dev_a_followon.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem, ccm_dlpno_mp2
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


# Coarse cderi for speed: the no-truncation gate is a *consistency* check
# (DLPNO vs canonical built from the SAME neutral L), so the cderi accuracy is
# irrelevant — a small vacuum + low ke_cutoff keep the FFT box tiny without
# weakening the gate. (Production cderi uses the defaults.)
_KE = 40.0
_VAC = 15.0


def _h2_chain(cell=6.0, vac=_VAC):
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], charge=0, multiplicity=1,
    )


class _Ref:
    """A neutral CCM reference (built once per cluster): SCF + canonical MP2 + cderi."""

    def __init__(self, ccm):
        self.ccm = ccm
        self.g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
        self.L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
        self.scf = run_ccm_rhf(ccm, eri=self.g)
        self.canon = run_ccm_mp2(ccm, self.scf, eri=self.g).e_correlation


@pytest.fixture(scope="module")
def ref_211():
    return _Ref(CCMSystem(_h2_chain(6.0), (2, 1, 1), "sto-3g"))


@pytest.fixture(scope="module")
def ref_411():
    return _Ref(CCMSystem(_h2_chain(6.0), (4, 1, 1), "sto-3g"))


# --------------------------------------------------------------------------- #
# The no-truncation gate: DLPNO-MP2(no trunc) == canonical CCM MP2 (neutral).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("localize", ["none", "pm", "boys"])
def test_no_truncation_reproduces_canonical(ref_211, localize):
    d = ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L,
                      localize=localize, tcut_pno=0.0, tcut_mkn=0.0)
    assert d.converged
    assert d.e_corr == pytest.approx(ref_211.canon, abs=1e-9)


def test_no_truncation_larger_cluster(ref_411):
    """10-pair (4,1,1) cluster: coupled LMP2 still collapses to canonical."""
    d = ccm_dlpno_mp2(ref_411.ccm, ref_411.scf, cderi=ref_411.L,
                      localize="pm", tcut_pno=0.0, tcut_mkn=0.0)
    assert d.converged
    assert d.e_corr == pytest.approx(ref_411.canon, abs=1e-8)
    # n_act = 4 occupieds → 10 unique pairs
    assert d.n_pairs == 10
    assert d.n_occ_active == 4


def test_e_total_and_per_atom_consistent(ref_211):
    d = ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L, localize="none")
    assert d.e_total == pytest.approx(d.e_hf + d.e_corr, abs=1e-14)
    assert d.e_corr_per_atom == pytest.approx(d.e_corr / ref_211.ccm.n_atoms, abs=1e-14)


# --------------------------------------------------------------------------- #
# PNO truncation: tighter threshold → smaller error; correction tracks it.
# --------------------------------------------------------------------------- #
def test_pno_truncation_monotone(ref_211):
    errs = []
    for tc in (1e-4, 1e-6, 1e-8):
        d = ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L,
                          localize="pm", tcut_pno=tc, tcut_mkn=0.0)
        errs.append(abs(d.e_corr - ref_211.canon))
    # tighter threshold (smaller tcut) → smaller error
    assert errs[0] >= errs[1] >= errs[2]
    assert errs[-1] < 1e-7


def test_pno_correction_tracks_truncation(ref_211):
    """The perturbative e_pno_correction is the full−truncated semicanonical gap."""
    d = ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L,
                      localize="pm", tcut_pno=1e-4, tcut_mkn=0.0)
    # with the correction applied the residual error is far below the raw gap
    assert abs(d.e_pno_correction) > 1e-6
    assert abs(d.e_corr - ref_211.canon) < abs(d.e_pno_correction)


def test_truncation_reduces_pno_count(ref_211):
    d_full = ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L,
                           localize="pm", tcut_pno=0.0)
    d_trunc = ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L,
                            localize="pm", tcut_pno=1e-4)
    assert sum(d_trunc.pno_per_pair.values()) <= sum(d_full.pno_per_pair.values())


# --------------------------------------------------------------------------- #
# API guards
# --------------------------------------------------------------------------- #
def test_unknown_localize_raises(ref_211):
    with pytest.raises(ValueError):
        ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L, localize="iao")


def test_bad_n_frozen_raises(ref_211):
    with pytest.raises(ValueError):
        ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, cderi=ref_211.L, n_frozen=99)


def test_builds_cderi_when_omitted(ref_211):
    """cderi is computed from the system when not supplied (slower path)."""
    d = ccm_dlpno_mp2(ref_211.ccm, ref_211.scf, localize="none", tcut_pno=0.0,
                      ke_cutoff=_KE)
    assert d.e_corr == pytest.approx(ref_211.canon, abs=1e-9)
