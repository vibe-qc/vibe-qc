"""Four-center CCM ERIs + HF-CCM SCF — AICCM milestone 2b (work in progress).

The four-center effective ERI tensor and the HF-CCM SCF are validated *in the
molecular limit* (an isolated cluster must reproduce the ordinary molecular
RHF / ERIs). The periodic four-center weighting (eqs 18–24) does not yet
reproduce the published reference — that parity is captured as an ``xfail``
documenting the target so the gate flips to pass once the weighting is fixed.

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550, eqs (18)-(27); Table 2 (H₄ alternating chain).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, RHFOptions, compute_eri, run_rhf
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.padded import ccm_eri
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _cell(lattice, atoms):
    mult = 1 if sum(a.Z for a in atoms) % 2 == 0 else 2
    return PeriodicSystem(3, np.asarray(lattice, float), atoms, charge=0, multiplicity=mult)


def test_ccm_eri_isolated_equals_molecular():
    """Isolated (huge-cell) cluster: the effective ERI is the molecular ERI."""
    ccm = CCMSystem(_cell(np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]), (1, 1, 1), "sto-3g")
    assert np.allclose(ccm_eri(ccm), np.asarray(compute_eri(ccm.basis)), atol=1e-10)


def test_ccm_rhf_isolated_equals_molecular():
    """Isolated cluster: HF-CCM total energy equals molecular RHF."""
    ccm = CCMSystem(_cell(np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]), (1, 1, 1), "sto-3g")
    res = run_ccm_rhf(ccm)
    mol = ccm.supercell
    ref = run_rhf(mol, BasisSet(mol, "sto-3g"), RHFOptions())
    assert res.converged
    assert res.energy == pytest.approx(ref.energy, abs=1e-8)
    assert res.idempotency_error < 1e-8


def test_ccm_rhf_multicell_energy_normalization_is_explicit():
    """Issue #177: the legacy energy field is a cyclic-cluster total."""
    unit = _cell(
        np.diag([20.0, 20.0, 20.0]),
        [Atom(2, [10.0, 10.0, 10.0])],
    )
    ccm = CCMSystem(unit, (2, 1, 1), "sto-3g")
    result = run_ccm_rhf(ccm)

    assert result.converged
    assert result.normalization == "total_cyclic_cluster"
    assert result.total_cyclic_energy == pytest.approx(result.energy, abs=0.0)
    assert result.total_cyclic_energy / ccm.n_cells == pytest.approx(
        result.energy_per_atom * (ccm.n_atoms / ccm.n_cells),
        abs=1.0e-14,
    )


def test_ccm_rhf_h4_alternating_parity():
    """H₄ alternating chain (STO-3G) → −0.542875 Ha/atom (Peintinger PhD thesis
    Tab. 8.3 / JCC 2014 Tab. 2). Unit cell: 4 H at x = 0, 0.8, 2.0, 2.8 Å
    (period 4.0 Å); energy/atom converges with cluster size.

    The home-bra + bra-ket-symmetrised four-center reproduces the supercell-model
    HF energy to ~1e-5 Ha/atom (not yet bit-exact — see handovers/HANDOVER_AICCM.md M2b).
    """
    pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
    unit = _cell(np.diag([4.0 * BOHR, 40.0, 40.0]), [Atom(1, p) for p in pos])
    e8 = run_ccm_rhf(CCMSystem(unit, (8, 1, 1), "sto-3g")).energy_per_atom
    e10 = run_ccm_rhf(CCMSystem(unit, (10, 1, 1), "sto-3g")).energy_per_atom
    # Reproduces the reference to ~1e-5, and converges (10-cell tighter than 8-cell).
    assert e10 == pytest.approx(-0.542875, abs=5e-5)
    assert abs(e10 - (-0.542875)) < abs(e8 - (-0.542875)) + 1e-9
