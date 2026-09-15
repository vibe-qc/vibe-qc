"""Open-shell DLPNO-UMP2 on the cyclic cluster (AICCM Task D, U-DLPNO).

Local-correlation DLPNO-UMP2 on the **neutral**-reference UHF-CCM wavefunction,
resolving the correlation into the αα / ββ / αβ spin channels with per-pair PNOs
(reusing the molecular ``vibeqc.dlpno.ump2`` PNO machinery with the CCM neutral
B-tensors).

The hard gate: the no-truncation limit (``tcut_pno = 0``) reproduces the canonical
CCM UMP2 (:func:`run_ccm_ump2`) on the *same* neutral four-center to machine ε —
for a closed-shell *and* a genuine open-shell (doublet) cluster.

References: Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013), and Pinski &
Neese, J. Chem. Phys. 150, 164102 (2019). Derivation / decision log (neutral reference):
docs/aiccm2026dev_a_followon.md § D.3 / § 2.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.dlpno_ump2 import ccm_dlpno_ump2
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.uhf import run_ccm_uhf
from vibeqc.periodic.ccm.ump2 import run_ccm_ump2

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

# Coarse cderi (small vacuum + low ke_cutoff): the no-truncation gate is a
# consistency check (DLPNO vs canonical built from the SAME neutral L), so the
# cderi accuracy is irrelevant — only the FFT box size, kept tiny for speed.
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


class _URef:
    """A neutral UHF-CCM reference: SCF + canonical UMP2 + cderi (built once)."""

    def __init__(self, ccm, basis_built=True):
        self.ccm = ccm
        self.g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
        self.L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
        self.uhf = run_ccm_uhf(ccm, eri=self.g)
        self.canon = run_ccm_ump2(ccm, self.uhf, eri=self.g)


@pytest.fixture(scope="module")
def cs_ref():
    """Closed-shell H₂ chain (2,1,1): singlet, n_α = n_β."""
    return _URef(CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g"))


@pytest.fixture(scope="module")
def os_ref():
    """Genuine open-shell H chain (3,1,1): doublet, n_α ≠ n_β."""
    return _URef(CCMSystem(_h_doublet(), (3, 1, 1), "sto-3g"))


# --------------------------------------------------------------------------- #
# The no-truncation gate: DLPNO-UMP2(no trunc) == canonical CCM UMP2 (neutral).
# --------------------------------------------------------------------------- #
def test_no_truncation_closed_shell(cs_ref):
    d = ccm_dlpno_ump2(cs_ref.ccm, cs_ref.uhf, cderi=cs_ref.L, tcut_pno=0.0)
    assert d.e_corr == pytest.approx(cs_ref.canon.e_correlation, abs=1e-10)
    assert d.e_aa == pytest.approx(cs_ref.canon.e_aa, abs=1e-10)
    assert d.e_bb == pytest.approx(cs_ref.canon.e_bb, abs=1e-10)
    assert d.e_ab == pytest.approx(cs_ref.canon.e_ab, abs=1e-10)
    # Closed shell: the two same-spin channels are equal.
    assert d.e_aa == pytest.approx(d.e_bb, abs=1e-12)


def test_no_truncation_open_shell(os_ref):
    assert os_ref.uhf.n_alpha != os_ref.uhf.n_beta      # genuinely open-shell
    d = ccm_dlpno_ump2(os_ref.ccm, os_ref.uhf, cderi=os_ref.L, tcut_pno=0.0)
    assert d.e_corr == pytest.approx(os_ref.canon.e_correlation, abs=1e-10)
    assert d.e_aa == pytest.approx(os_ref.canon.e_aa, abs=1e-10)
    assert d.e_bb == pytest.approx(os_ref.canon.e_bb, abs=1e-10)
    assert d.e_ab == pytest.approx(os_ref.canon.e_ab, abs=1e-10)
    assert d.e_corr < 0.0
    # Doublet (n_α=2, n_β=1): 1 αα pair, 0 ββ pairs, 2 αβ pairs.
    assert (d.n_pairs_aa, d.n_pairs_bb, d.n_pairs_ab) == (1, 0, 2)


def test_auto_builds_uhf_reference(os_ref):
    """uhf_result=None builds the neutral UHF reference internally."""
    d = ccm_dlpno_ump2(os_ref.ccm, cderi=os_ref.L, tcut_pno=0.0, ke_cutoff=_KE)
    assert d.e_corr == pytest.approx(os_ref.canon.e_correlation, abs=1e-9)


def test_truncation_recovers_canonical():
    """6-31g doublet: PNO recovery and per-spin PAO work reduction."""
    ccm = CCMSystem(_h_doublet(), (3, 1, 1), "6-31g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    uhf = run_ccm_uhf(ccm, eri=g)
    canon = run_ccm_ump2(ccm, uhf, eri=g).e_correlation

    errs = []
    for tc in (3e-4, 1e-5, 1e-7, 0.0):
        d = ccm_dlpno_ump2(ccm, uhf, cderi=L, tcut_pno=tc)
        errs.append(abs(d.e_corr - canon))
        assert d.e_corr < 0.0
    # Recovery is monotone non-increasing in tcut and exact at tcut=0. On this
    # near-trivial cell (canonical e_corr ~ 1e-5 Ha) adjacent thresholds can keep
    # the identical PNO set (so a step can be flat); the strict ladder is thus
    # non-strict. "Truncation has a real effect" and "exact at tcut=0" are pinned
    # separately so the recovery direction is still fully constrained.
    assert errs[0] >= errs[1] >= errs[2] >= errs[-1]
    assert errs[0] > errs[-1]  # truncation genuinely changes the energy
    assert errs[-1] < 1e-9  # exact at no truncation

    # The open-shell M2 PAO route builds alpha and beta domains separately.
    # This Mulliken threshold removes two virtual directions from one alpha-beta
    # pair while retaining a negative, close-to-canonical correlation energy.
    pao = ccm_dlpno_ump2(
        ccm, uhf, cderi=L, tcut_pno=0.0, tcut_mkn=1e-3
    )
    assert pao.avg_pao_ab < 0.5 * (
        pao.pao_per_pair[("ab", 1, 0)][0]
        + pao.pao_per_pair[("ab", 1, 0)][1]
    )
    assert pao.pao_per_pair[("ab", 0, 0)] == (3, 3)
    assert pao.e_corr < 0.0
    assert abs(pao.e_corr - canon) < 1e-5


def test_pao_threshold_validation(os_ref):
    with pytest.raises(ValueError, match="tcut_mkn must be non-negative"):
        ccm_dlpno_ump2(
            os_ref.ccm, os_ref.uhf, cderi=os_ref.L, tcut_mkn=-1e-3
        )


def test_scs_scaling(cs_ref):
    """ss/os scaling recombines the unscaled per-channel energies."""
    d = ccm_dlpno_ump2(cs_ref.ccm, cs_ref.uhf, cderi=cs_ref.L,
                       tcut_pno=0.0, ss_scale=6.0 / 5.0, os_scale=1.0 / 3.0)
    expected = (1.0 / 3.0) * d.e_ab + (6.0 / 5.0) * (d.e_aa + d.e_bb)
    assert d.e_corr == pytest.approx(expected, abs=1e-14)


def test_totals_and_per_atom(cs_ref):
    d = ccm_dlpno_ump2(cs_ref.ccm, cs_ref.uhf, cderi=cs_ref.L, tcut_pno=0.0)
    assert d.e_total == pytest.approx(d.e_hf + d.e_corr, abs=1e-14)
    assert d.e_corr_per_atom == pytest.approx(d.e_corr / cs_ref.ccm.n_atoms, abs=1e-14)
    assert d.e_total_per_atom == pytest.approx(d.e_total / cs_ref.ccm.n_atoms, abs=1e-14)


def test_requires_converged_reference(cs_ref):
    class _Bad:
        converged = False

    with pytest.raises(ValueError):
        ccm_dlpno_ump2(cs_ref.ccm, _Bad(), cderi=cs_ref.L)


# --------------------------------------------------------------------------- #
# RI-UMP2: canonical UMP2 from the neutral cderi L, no dense AO four-center
# (the scalability hand-off lever that lets open-shell correlation reach 3-D).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", ["cs_ref", "os_ref"])
def test_ri_ump2_equals_dense_on_neutral_kernel(request, fixture):
    """run_ccm_ump2(cderi=L) == dense run_ccm_ump2(eri=g_eff=ΣLL) to machine ε,
    per spin channel, without ever forming the n_ref_ao**4 AO four-center."""
    ref = request.getfixturevalue(fixture)
    ri = run_ccm_ump2(ref.ccm, ref.uhf, cderi=ref.L)
    assert ri.e_correlation == pytest.approx(ref.canon.e_correlation, abs=1e-11)
    assert ri.e_aa == pytest.approx(ref.canon.e_aa, abs=1e-11)
    assert ri.e_bb == pytest.approx(ref.canon.e_bb, abs=1e-11)
    assert ri.e_ab == pytest.approx(ref.canon.e_ab, abs=1e-11)


def test_ri_ump2_requires_uhf_reference(os_ref):
    with pytest.raises(ValueError):
        run_ccm_ump2(os_ref.ccm, cderi=os_ref.L)


@pytest.mark.parametrize("fixture", ["cs_ref", "os_ref"])
def test_lean_uhf_from_cderi_matches_dense_g(request, fixture):
    """run_ccm_uhf(cderi=L) assembles J/K straight from the neutral cderi L (the
    dense n_ref**4 g is never formed) and matches run_ccm_uhf(eri=g_eff=ΣLL) to
    machine ε -- the lean open-shell neutral SCF that pairs with the open-shell RI
    correlation (run_ccm_ump2(uhf, cderi=L))."""
    ref = request.getfixturevalue(fixture)
    lean = run_ccm_uhf(ref.ccm, cderi=ref.L)
    assert lean.converged
    assert lean.energy == pytest.approx(ref.uhf.energy, abs=1e-10)
