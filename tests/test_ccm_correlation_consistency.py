"""Cross-method consistency of the CCM correlation stack (integration capstone).

Every closed-shell CCM correlation method built on the **neutral** four-center must
agree in its exact limit -- dense == RI == DLPNO(no-trunc) == coupled-DLPNO(no-trunc)
== canonical -- because they are the same energy functional reached by different
(equivalent) routes. Each method has its own gate against canonical/dense elsewhere;
this ties the whole family together on one system and would catch cross-method drift
(a mismatched neutral kernel / ke_cutoff, a route that silently diverges) that no
single per-method test sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.dlpno import ccm_dlpno_mp2
from vibeqc.periodic.ccm.dlpno_ccsd import ccm_dlpno_ccsd
from vibeqc.periodic.ccm.dlpno_ccsd_coupled import ccm_dlpno_ccsd_coupled
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_KE = 40.0


@pytest.fixture(scope="module")
def _neutral():
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), (2, 1, 1), "sto-3g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    scf = run_ccm_rhf(ccm, eri=g)
    return ccm, scf, g, L


def test_mp2_family_agrees(_neutral):
    """dense MP2 == RI-MP2 == DLPNO-MP2(no-trunc), all on the neutral kernel."""
    ccm, scf, g, L = _neutral
    dense = run_ccm_mp2(ccm, scf, eri=g).e_correlation
    ri = run_ccm_mp2(ccm, scf, cderi=L).e_correlation
    dlpno = ccm_dlpno_mp2(ccm, scf, cderi=L, localize="none", tcut_pno=0.0).e_corr
    assert ri == pytest.approx(dense, abs=1e-10)
    assert dlpno == pytest.approx(dense, abs=1e-9)


def test_ccsd_family_agrees(_neutral):
    """canonical CCSD == RI-CCSD == union DLPNO-CCSD(no-trunc) ==
    coupled DLPNO-CCSD(no-trunc), all on the neutral kernel."""
    ccm, scf, g, L = _neutral
    canon = run_ccm_ccsd(ccm, scf, eri=g, compute_triples=False).e_ccsd_correlation
    ri = run_ccm_ccsd(ccm, scf, cderi=L, compute_triples=False).e_ccsd_correlation
    union = ccm_dlpno_ccsd(ccm, scf, cderi=L, g_neutral=g, localize="none",
                           tcut_pno=0.0, compute_triples=False).e_ccsd_correlation
    coupled = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="none",
                                     tcut_pno=0.0, compute_triples=False).e_ccsd_correlation
    assert ri == pytest.approx(canon, abs=1e-10)
    assert union == pytest.approx(canon, abs=1e-8)
    assert coupled == pytest.approx(canon, abs=1e-8)


def test_ccsdt_agrees(_neutral):
    """canonical CCSD(T) == coupled DLPNO-CCSD(T) (dense (T), no truncation)."""
    ccm, scf, g, L = _neutral
    canon = run_ccm_ccsd(ccm, scf, cderi=L, compute_triples=True)
    coupled = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="none", tcut_pno=0.0,
                                     compute_triples=True, triples_mode="dense")
    assert coupled.e_t == pytest.approx(canon.e_t, abs=1e-9)
    assert coupled.e_correlation == pytest.approx(canon.e_correlation, abs=1e-8)
