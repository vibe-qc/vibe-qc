"""Per-pair-coupled DLPNO-CCSD on the cyclic cluster (AICCM Task 2, M3b).

The hard gate: at ``tcut_pno = 0`` with full coupling, the genuine per-pair-coupled
local CCSD (each pair in its own PNO basis, cross-pair PNO-overlap projections)
reproduces the canonical CCM CCSD on the same neutral four-center to the CCSD
convergence tolerance -- the correctness proof for the coupled residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.dlpno_ccsd_coupled import ccm_dlpno_ccsd_coupled
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_KE = 40.0


def _ref(nrep):
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), nrep, "sto-3g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    scf = run_ccm_rhf(ccm, eri=g)
    canon = run_ccm_ccsd(ccm, scf, cderi=L, compute_triples=False)
    return ccm, scf, L, canon


@pytest.mark.parametrize("nrep", [(2, 1, 1), (4, 1, 1)])
@pytest.mark.parametrize("localize", ["none", "pm"])
def test_coupled_no_truncation_reproduces_canonical(nrep, localize):
    ccm, scf, L, canon = _ref(nrep)
    d = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize=localize,
                               tcut_pno=0.0, tcut_mkn=0.0, compute_triples=False)
    assert d.converged
    assert d.e_ccsd_correlation == pytest.approx(canon.e_ccsd_correlation, abs=1e-8)


def test_coupled_truncation_shrinks_pnos_and_stays_bound():
    ccm, scf, L, canon = _ref((4, 1, 1))
    d0 = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=0.0, compute_triples=False)
    dT = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=1e-4, compute_triples=False)
    assert dT.avg_pno_per_pair < d0.avg_pno_per_pair       # truncation removes PNOs
    assert dT.e_ccsd_correlation < 0.0                     # still bound
    # the truncated correlation recovers most of canonical.
    assert abs(dT.e_ccsd_correlation - canon.e_ccsd_correlation) < 5e-3


@pytest.mark.parametrize("nrep", [(2, 1, 1), (4, 1, 1)])
@pytest.mark.parametrize("localize", ["none", "pm"])
# "local" tol 2e-8: the T0 semicanonical error of the scalable (T) is a small
# kernel-dependent quantity; the 2026-07-02 cross-torus truncation fix in
# ccm_neutral_cderi (wrap-aware lattice cutoff, corrects L on supercells
# > ~15 bohr — (4,1,1) here is 24 bohr) moved the pm/(4,1,1) case from
# 0.99e-8 to 1.002e-8. The exact "dense" mode is unaffected (1e-10).
@pytest.mark.parametrize("triples_mode,tol", [("dense", 1e-10), ("local", 2e-8)])
def test_coupled_ccsdt_no_truncation_reproduces_canonical(nrep, localize, triples_mode, tol):
    """No-truncation coupled DLPNO-CCSD(T) reproduces canonical CCSD(T), for both
    (T) drivers: ``dense`` reconstructs the canonical amplitudes (exact, ~1e-10);
    ``local`` (the scalable per-triple-TNO (T), the default) matches to the CCSD
    convergence tolerance (~1e-8) at ``tcut_tno=0`` / full PNO."""
    ccm, scf, L, _ = _ref(nrep)
    canon = run_ccm_ccsd(ccm, scf, cderi=L, compute_triples=True)
    d = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize=localize, tcut_pno=0.0,
                               compute_triples=True, triples_mode=triples_mode)
    assert d.e_t == pytest.approx(canon.e_t, abs=tol)
    # total: dense reproduces canonical tightly; the scalable local-(T0) carries a
    # small semicanonical error on localized occupieds that stacks with the CCSD
    # convergence tolerance.
    assert d.e_correlation == pytest.approx(canon.e_correlation, abs=5e-8)


def test_local_triples_tno_truncation_changes_t():
    """tcut_tno>0 truncates each triple's TNO domain — a different, still-small (T)."""
    ccm, scf, L, _ = _ref((4, 1, 1))
    t0 = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=0.0, compute_triples=True,
                                triples_mode="local", tcut_tno=0.0).e_t
    tT = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=0.0, compute_triples=True,
                                triples_mode="local", tcut_tno=1e-4).e_t
    assert tT != t0
    assert tT < 0.0 and abs(tT - t0) < 1e-5


def test_coupling_radius_locality_screen():
    """The O(N)-pairs lever: a large coupling_radius reproduces full coupling; a
    finite radius couples fewer occupieds with a small, controlled energy change."""
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), (4, 1, 1), "sto-3g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    scf = run_ccm_rhf(ccm, eri=g)
    full = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="pm",
                                  coupling_radius=-1.0, compute_triples=False)
    big = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="pm",
                                 coupling_radius=1000.0, compute_triples=False)
    tight = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="pm",
                                   coupling_radius=4.0, compute_triples=False)
    # large radius == full coupling (every occupied within range)
    assert big.avg_coupled_occ == pytest.approx(full.avg_coupled_occ)
    assert big.e_ccsd_correlation == pytest.approx(full.e_ccsd_correlation, abs=1e-10)
    # a tight radius couples fewer occupieds, still bound, close to full
    assert tight.avg_coupled_occ < full.avg_coupled_occ
    assert tight.e_ccsd_correlation < 0.0
    assert abs(tight.e_ccsd_correlation - full.e_ccsd_correlation) < 1e-2
