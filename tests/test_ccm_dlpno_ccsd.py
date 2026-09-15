"""DLPNO-CCSD(T) on the cyclic cluster (AICCM Task 2, M3).

Subspace-projected DLPNO-CCSD(T): the per-pair PNOs are merged into a single
union virtual space ``V_trunc`` (S^CCM-orthonormal, semicanonical) and the CCM
CCSD engine runs in ``{canonical occ, V_trunc}``, on the **neutral** four-center
reference.

The hard gate: the no-truncation limit (``tcut_pno = tcut_mkn = 0``) must
reproduce the canonical CCM CCSD(T) on the *same* neutral four-center to ~µHa —
``V_trunc`` then spans the full virtual block, so CCSD(T) (invariant to unitary
rotations within occ / virt) equals the canonical result, for canonical *and*
localized occupieds.

Reference: Riplinger, Sandhoefer, Hansen, Neese, J. Chem. Phys. 139, 134101
(2013). Derivation: docs/aiccm2026dev_a_followon.md § 2.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem, ccm_dlpno_ccsd
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_KE = 40.0
_VAC = 15.0


def _h2_chain(cell=6.0, vac=_VAC):
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], charge=0, multiplicity=1,
    )


class _Ref:
    """Neutral CCM reference (once per cluster): SCF + canonical CCSD(T) + g + L."""

    def __init__(self, ccm):
        self.ccm = ccm
        self.g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
        self.L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
        self.scf = run_ccm_rhf(ccm, eri=self.g)
        self.canon = run_ccm_ccsd(ccm, self.scf, eri=self.g, compute_triples=True)


@pytest.fixture(scope="module")
def ref_211():
    return _Ref(CCMSystem(_h2_chain(6.0), (2, 1, 1), "sto-3g"))


@pytest.fixture(scope="module")
def ref_411():
    return _Ref(CCMSystem(_h2_chain(6.0), (4, 1, 1), "sto-3g"))


# --------------------------------------------------------------------------- #
# The no-truncation gate: DLPNO-CCSD(T)(no trunc) == canonical CCM CCSD(T).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("localize", ["none", "pm", "boys"])
def test_no_truncation_reproduces_canonical(ref_211, localize):
    d = ccm_dlpno_ccsd(ref_211.ccm, ref_211.scf, cderi=ref_211.L, g_neutral=ref_211.g,
                       localize=localize, tcut_pno=0.0, tcut_mkn=0.0)
    assert d.converged
    # CCSD and (T) each match canonical, hence the total.
    assert d.e_ccsd_correlation == pytest.approx(ref_211.canon.e_ccsd_correlation, abs=1e-9)
    assert d.e_t == pytest.approx(ref_211.canon.e_t, abs=1e-10)
    assert d.e_correlation == pytest.approx(ref_211.canon.e_correlation, abs=1e-9)
    # at no truncation the union PNO space is the full virtual space
    assert d.n_virtual_pno == d.n_virtual_full


def test_no_truncation_larger_cluster(ref_411):
    d = ccm_dlpno_ccsd(ref_411.ccm, ref_411.scf, cderi=ref_411.L, g_neutral=ref_411.g,
                       localize="pm", tcut_pno=0.0, tcut_mkn=0.0)
    assert d.e_correlation == pytest.approx(ref_411.canon.e_correlation, abs=1e-8)
    assert d.n_virtual_pno == d.n_virtual_full


def test_triples_toggle(ref_211):
    d = ccm_dlpno_ccsd(ref_211.ccm, ref_211.scf, cderi=ref_211.L, g_neutral=ref_211.g,
                       localize="none", compute_triples=False)
    assert d.e_t == 0.0
    assert d.e_correlation == pytest.approx(d.e_ccsd_correlation, abs=1e-14)


def test_totals_consistent(ref_211):
    d = ccm_dlpno_ccsd(ref_211.ccm, ref_211.scf, cderi=ref_211.L, g_neutral=ref_211.g,
                       localize="none")
    assert d.e_total == pytest.approx(d.e_hf + d.e_correlation, abs=1e-12)
    assert d.e_correlation_per_atom == pytest.approx(
        d.e_correlation / ref_211.ccm.n_atoms, abs=1e-14)
    assert d.n_virtual_full == ref_211.ccm.nbf - ref_211.ccm.supercell.n_electrons() // 2


def test_builds_g_from_cderi(ref_211):
    """g_neutral is built from the cderi (Σ_P L⊗L) when not supplied."""
    d = ccm_dlpno_ccsd(ref_211.ccm, ref_211.scf, cderi=ref_211.L,
                       localize="none", tcut_pno=0.0)
    assert d.e_correlation == pytest.approx(ref_211.canon.e_correlation, abs=1e-9)


def test_truncation_never_exceeds_full(ref_211):
    """PNO truncation can only shrink (or keep) the union virtual space."""
    for tc in (1e-6, 1e-3, 1e-1):
        d = ccm_dlpno_ccsd(ref_211.ccm, ref_211.scf, cderi=ref_211.L, g_neutral=ref_211.g,
                           localize="pm", tcut_pno=tc)
        assert d.n_virtual_pno <= d.n_virtual_full
        assert d.converged


# --------------------------------------------------------------------------- #
# API guards
# --------------------------------------------------------------------------- #
def test_unknown_localize_raises(ref_211):
    with pytest.raises(ValueError):
        ccm_dlpno_ccsd(ref_211.ccm, ref_211.scf, cderi=ref_211.L, g_neutral=ref_211.g,
                       localize="iao")


# --------------------------------------------------------------------------- #
# RI-CCSD(T): canonical CCSD(T) from the neutral cderi L, no dense AO four-center
# (the scalability hand-off lever that lets canonical CCSD reach 3-D).
# --------------------------------------------------------------------------- #
def test_ri_ccsd_equals_dense_on_neutral_kernel(ref_211):
    """run_ccm_ccsd(cderi=L) == dense run_ccm_ccsd(eri=g_eff=ΣLL) to machine ε,
    without ever forming the n_ref_ao**4 AO four-center."""
    ri = run_ccm_ccsd(ref_211.ccm, ref_211.scf, cderi=ref_211.L, compute_triples=True)
    assert ri.e_ccsd_correlation == pytest.approx(ref_211.canon.e_ccsd_correlation, abs=1e-10)
    assert ri.e_t == pytest.approx(ref_211.canon.e_t, abs=1e-11)
    assert ri.e_correlation == pytest.approx(ref_211.canon.e_correlation, abs=1e-10)


def test_ri_ccsd_requires_scf_reference(ref_211):
    with pytest.raises(ValueError):
        run_ccm_ccsd(ref_211.ccm, cderi=ref_211.L)
