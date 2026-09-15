"""DLPNO-UCCSD(T) on the cyclic cluster (AICCM Task D, U-DLPNO).

Subspace-projected open-shell coupled cluster on the **neutral**-reference
UHF-CCM wavefunction: per-channel PNO pairs → per-spin union virtual spaces
``V_trunc^α`` / ``V_trunc^β`` (``S^CCM``-orthonormal, semicanonical) → the
spin-orbital UCCSD(T) engine (``run_ref_uccsd``, the same one
:func:`run_ccm_uccsd` uses on the full virtual space).

The hard gate: the no-truncation limit (``tcut_pno = 0``) reproduces the canonical
CCM UCCSD(T) (:func:`run_ccm_uccsd`) on the *same* neutral four-center to ~µHa
(CCSD to machine ε; (T) to ≤1e-9) — closed-shell *and* genuine open-shell.

Reference: Riplinger, Sandhoefer, Hansen & Neese, J. Chem. Phys. 139, 134101
(2013); Saitow et al., J. Chem. Phys. 146, 164105 (2017). Derivation:
docs/aiccm2026dev_a_followon.md § D.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.dlpno_uccsd import ccm_dlpno_uccsd
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.uccsd import run_ccm_uccsd
from vibeqc.periodic.ccm.uhf import run_ccm_uhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_KE = 40.0
_VAC = 15.0


def _h2_chain(cell=6.0, vac=_VAC):
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], charge=0, multiplicity=1)


def _h_doublet(cell=1.8, vac=_VAC):
    # Neutral cderi is a 3D Coulomb route; use transverse vacuum instead of the
    # intentionally rejected dim=1 collapsed mesh.
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0])], charge=0, multiplicity=2)


class _UCCRef:
    """A neutral UHF-CCM reference: SCF + canonical UCCSD(T) + cderi (built once)."""

    def __init__(self, ccm):
        self.ccm = ccm
        self.g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
        self.L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
        self.uhf = run_ccm_uhf(ccm, eri=self.g)
        self.canon = run_ccm_uccsd(ccm, self.uhf, cderi=self.L, compute_triples=True)


@pytest.fixture(scope="module")
def cs_ref():
    return _UCCRef(CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g"))


@pytest.fixture(scope="module")
def os_ref():
    return _UCCRef(CCMSystem(_h_doublet(), (3, 1, 1), "sto-3g"))


# --------------------------------------------------------------------------- #
# The no-truncation gate: DLPNO-UCCSD(T)(no trunc) == canonical CCM UCCSD(T).
# --------------------------------------------------------------------------- #
def test_no_truncation_closed_shell(cs_ref):
    d = ccm_dlpno_uccsd(cs_ref.ccm, cs_ref.uhf, cderi=cs_ref.L,
                        tcut_pno=0.0, compute_triples=True)
    assert d.converged
    assert d.e_ccsd_correlation == pytest.approx(cs_ref.canon.e_ccsd_correlation, abs=1e-9)
    assert d.e_t == pytest.approx(cs_ref.canon.e_t, abs=1e-10)
    assert d.e_correlation == pytest.approx(cs_ref.canon.e_correlation, abs=1e-9)
    # Closed shell: V_trunc^α and V_trunc^β have equal dimension.
    assert d.n_virtual_pno_alpha == d.n_virtual_pno_beta


def test_no_truncation_open_shell(os_ref):
    assert os_ref.uhf.n_alpha != os_ref.uhf.n_beta      # genuinely open-shell
    d = ccm_dlpno_uccsd(os_ref.ccm, os_ref.uhf, cderi=os_ref.L,
                        tcut_pno=0.0, compute_triples=True)
    assert d.converged
    assert d.e_ccsd_correlation == pytest.approx(os_ref.canon.e_ccsd_correlation, abs=1e-8)
    assert d.e_t == pytest.approx(os_ref.canon.e_t, abs=1e-9)
    assert d.e_correlation == pytest.approx(os_ref.canon.e_correlation, abs=1e-8)
    assert d.e_correlation < 0.0


def test_auto_builds_uhf_reference(os_ref):
    d = ccm_dlpno_uccsd(os_ref.ccm, cderi=os_ref.L, tcut_pno=0.0,
                        compute_triples=True, ke_cutoff=_KE)
    assert d.e_correlation == pytest.approx(os_ref.canon.e_correlation, abs=1e-8)


def test_truncation_recovers_canonical():
    """6-31g doublet: PNO recovery, padding, and per-spin PAO reduction."""
    ccm = CCMSystem(_h_doublet(), (3, 1, 1), "6-31g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    uhf = run_ccm_uhf(ccm, eri=g)
    canon = run_ccm_uccsd(ccm, uhf, cderi=L, compute_triples=True).e_correlation

    # A coarse truncation runs, converges, and stays a sensible bound.
    coarse = ccm_dlpno_uccsd(ccm, uhf, cderi=L, tcut_pno=3e-4, compute_triples=True)
    assert coarse.converged
    assert coarse.e_correlation < 0.0
    # Truncation genuinely bit (>> the ~1e-9 tcut=0 roundoff floor). The absolute
    # gap is small here only because this near-trivial cell has canonical
    # e_corr ~ 8e-6 Ha; the PNO truncation effect is ~2e-7, still ~200x the floor.
    assert abs(coarse.e_correlation - canon) > 1e-8     # truncation genuinely bit
    # Exact at no truncation.
    exact = ccm_dlpno_uccsd(ccm, uhf, cderi=L, tcut_pno=0.0, compute_triples=True)
    assert exact.e_correlation == pytest.approx(canon, abs=1e-9)

    # PAO work is reduced before the pair-PNO union. On this small chain the
    # union still spans every canonical virtual direction, so UCCSD remains
    # invariant while the alpha/beta pair builds are smaller.
    pao = ccm_dlpno_uccsd(
        ccm, uhf, cderi=L, tcut_pno=0.0, tcut_mkn=1e-3,
        compute_triples=False,
    )
    assert pao.converged
    assert pao.avg_pao_alpha < pao.n_virtual_full_alpha
    assert pao.avg_pao_beta < pao.n_virtual_full_beta
    assert pao.e_correlation == pytest.approx(
        exact.e_ccsd_correlation, abs=1e-10
    )


def test_no_triples_flag(cs_ref):
    d = ccm_dlpno_uccsd(cs_ref.ccm, cs_ref.uhf, cderi=cs_ref.L,
                        tcut_pno=0.0, compute_triples=False)
    assert d.e_t == 0.0
    assert d.e_correlation == pytest.approx(d.e_ccsd_correlation, abs=1e-14)


def test_totals_and_per_atom(cs_ref):
    d = ccm_dlpno_uccsd(cs_ref.ccm, cs_ref.uhf, cderi=cs_ref.L, tcut_pno=0.0)
    assert d.e_total == pytest.approx(d.e_hf + d.e_correlation, abs=1e-12)
    assert d.e_correlation_per_atom == pytest.approx(
        d.e_correlation / cs_ref.ccm.n_atoms, abs=1e-14)
    assert d.e_total_per_atom == pytest.approx(d.e_total / cs_ref.ccm.n_atoms, abs=1e-14)


def test_requires_converged_reference(cs_ref):
    class _Bad:
        converged = False

    with pytest.raises(ValueError):
        ccm_dlpno_uccsd(cs_ref.ccm, _Bad(), cderi=cs_ref.L)


def test_pao_threshold_validation(os_ref):
    with pytest.raises(ValueError, match="tcut_mkn must be non-negative"):
        ccm_dlpno_uccsd(
            os_ref.ccm, os_ref.uhf, cderi=os_ref.L, tcut_mkn=-1e-3
        )
