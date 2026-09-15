"""DLPNO local virtual representation on the cyclic cluster (Task 2, PAO stage).

``ccm_pao`` builds the projected atomic orbitals — the AO basis projected out of
the CCM occupied space — the first stage of the periodic DLPNO pipeline
(PAO → pair domains → PNO → DLPNO-MP2). Pinned: PAOs are S^CCM-orthogonal to the
occupied space, number the virtual dimension (no linear dependence in a minimal
basis), and (with the occupieds) span the full space.

Reference: Pulay, Chem. Phys. Lett. 100, 151 (1983); Riplinger & Neese,
J. Chem. Phys. 138, 034106 (2013).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem, ccm_pao, pao_occupied_orthogonality
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


@pytest.fixture(scope="module")
def ccm_scf():
    unit = PeriodicSystem(3, np.diag([6.0, 30.0, 30.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(unit, (3, 1, 1), "sto-3g")
    return ccm, run_ccm_rhf(ccm, method="aiccm2026dev-a")


def test_pao_orthogonal_to_occupied(ccm_scf):
    ccm, r = ccm_scf
    C_pao, _ = ccm_pao(r, ccm)
    assert pao_occupied_orthogonality(r, ccm, C_pao) < 1e-10


def test_pao_count_is_virtual_dimension(ccm_scf):
    """STO-3G H₂ chain: no linear dependence ⇒ n_pao = nbf − n_occ."""
    ccm, r = ccm_scf
    _, n_pao = ccm_pao(r, ccm)
    n_occ = ccm.supercell.n_electrons() // 2
    assert n_pao == ccm.nbf - n_occ


def test_occupied_plus_pao_span_full_space(ccm_scf):
    ccm, r = ccm_scf
    C_pao, _ = ccm_pao(r, ccm)
    S = np.asarray(r.overlap, dtype=float)
    n_occ = ccm.supercell.n_electrons() // 2
    C_occ = np.asarray(r.mo_coeffs)[:, :n_occ]
    M = np.hstack([C_occ, C_pao])
    assert np.linalg.matrix_rank(M.T @ S @ M, tol=1e-8) == ccm.nbf
