"""Open-shell UCCSD(T) on the cyclic cluster (AICCM Task D).

`run_ccm_uccsd` runs vibe-qc's spin-orbital UCCSD(T) engine
(`vibeqc.dlpno._ccsd_ref.run_ref_uccsd`) on the UHF-CCM reference, fed the CCM
neutral cderi transformed into the α / β MO bases — the natural route, since the
neutral four-center is the RI-separable CCM kernel and the required correlation
reference.

Gates:
* **Closed-shell consistency** — for an even-electron closed-shell cluster
  ``run_ccm_uccsd`` reproduces the closed-shell ``run_ccm_ccsd`` on the same
  neutral four-center to machine ε (the UHF collapses to RHF; αα = ββ). This
  exercises the full spin-orbital machinery + the α/β B-tensor transform.
* **Open-shell run** — a genuine doublet cluster converges to sensible (negative)
  correlation.

Reference: Stanton et al., J. Chem. Phys. 94, 4334 (1991). Derivation:
docs/aiccm2026dev_a_followon.md § D.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic.ccm.uccsd import run_ccm_uccsd

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_KE = 40.0


def _h2_chain(cell=6.0, vac=15.0):
    return PeriodicSystem(
        3, np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], charge=0, multiplicity=1,
    )


@pytest.fixture(scope="module")
def closed_shell_ref():
    ccm = CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    scf = run_ccm_rhf(ccm, eri=g)
    rcc = run_ccm_ccsd(ccm, scf, eri=g, compute_triples=True)
    return ccm, L, rcc


def test_closed_shell_uccsd_equals_ccsd(closed_shell_ref):
    ccm, L, rcc = closed_shell_ref
    ucc = run_ccm_uccsd(ccm, cderi=L, compute_triples=True)
    assert ucc.converged
    assert ucc.e_ccsd_correlation == pytest.approx(rcc.e_ccsd_correlation, abs=1e-8)
    assert ucc.e_t == pytest.approx(rcc.e_t, abs=1e-9)
    assert ucc.e_correlation == pytest.approx(rcc.e_correlation, abs=1e-8)


def test_uccsd_totals_consistent(closed_shell_ref):
    ccm, L, _ = closed_shell_ref
    ucc = run_ccm_uccsd(ccm, cderi=L, compute_triples=False)
    assert ucc.e_t == 0.0
    assert ucc.e_correlation == pytest.approx(ucc.e_ccsd_correlation, abs=1e-14)
    assert ucc.e_total == pytest.approx(ucc.e_hf + ucc.e_correlation, abs=1e-12)
    assert ucc.e_correlation_per_atom == pytest.approx(
        ucc.e_correlation / ccm.n_atoms, abs=1e-14)


def test_open_shell_doublet_runs():
    """A genuine open-shell (doublet) cluster: converges with negative correlation.

    Declared ``dim=3``; the geometry is unchanged (1.8-bohr H spacing along x,
    15-bohr transverse). Open-shellness has nothing to do with dimensionality, and
    ``run_ccm_uccsd`` builds the neutral cderi, which refuses ``dim < 3`` since
    2026-07-10: its reciprocal mesh pins every non-periodic axis at ``G_perp = 0``,
    so the kernel it feeds is a transverse-uniform sheet term, not ``1/r``.
    """
    unit = PeriodicSystem(
        3, np.array([[1.8, 0, 0], [0, 15.0, 0], [0, 0, 15.0]]),
        [Atom(1, [0, 0, 0])], charge=0, multiplicity=2)
    ccm = CCMSystem(unit, (3, 1, 1), "sto-3g")
    assert ccm.supercell.n_electrons() == 3       # doublet
    ucc = run_ccm_uccsd(ccm, compute_triples=True, ke_cutoff=_KE)
    assert ucc.converged
    assert ucc.uhf.n_alpha != ucc.uhf.n_beta      # genuinely open-shell
    assert ucc.e_correlation < 0.0


def test_requires_converged_reference(closed_shell_ref):
    ccm, L, _ = closed_shell_ref

    class _Bad:
        converged = False

    with pytest.raises(ValueError):
        run_ccm_uccsd(ccm, _Bad(), cderi=L)
