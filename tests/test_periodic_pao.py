"""Periodic Γ-point PAO virtual space — Stage 3.

Validates :func:`vibeqc.periodic_pao.build_pao_periodic_gamma`. The virtual
basis built by projecting the occupied manifold out of the AO space must:

1. be **orthogonal to the occupied** orbitals (``C_occᵀ S V = 0``);
2. be **S-orthonormal** (``Vᵀ S V = I``);
3. be **Fock-diagonal** (``Vᵀ F V`` diagonal) — semicanonical;
4. have ``n_vir = nbf − n_occ``;
5. reproduce the **canonical virtual orbital energies** — the full-domain
   semicanonical PAO basis diagonalises the Fock matrix over the whole virtual
   space, so its eigenvalues equal the SCF virtual eigenvalues (the oracle);
6. together with the occupied manifold, be **complete** (``P_occ + P_vir = I``).

Same H₄-chain-in-a-box system as ``test_periodic_localise.py`` (core-free, so
the Γ-Ewald grid is happy).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_pao import build_pao_periodic_gamma

BOX = 30.0
CTR = BOX / 2.0
SPACING = 1.8
_Z = [CTR + SPACING * (k - 1.5) for k in range(4)]
H4_POS = [[CTR, CTR, z] for z in _Z]


@pytest.fixture(scope="module")
def periodic_h4():
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 80
    opts.use_diis = True
    atoms = [vq.Atom(1, p) for p in H4_POS]
    sysp = vq.PeriodicSystem(3, np.eye(3) * BOX, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, auto_optimize_truncation=False
    )
    assert r.converged, f"periodic H4 SCF did not converge ({r.n_iter} iters)"
    return {"system": sysp, "basis": basis, "result": r}


@pytest.fixture(scope="module")
def pao(periodic_h4):
    return build_pao_periodic_gamma(
        periodic_h4["result"], periodic_h4["basis"], periodic_h4["system"]
    )


def _occ(periodic_h4, pao):
    return np.asarray(periodic_h4["result"].mo_coeffs)[:, : pao.n_occ]


def test_pao_orthogonal_to_occupied(periodic_h4, pao):
    S = np.asarray(periodic_h4["result"].overlap)
    C_occ = _occ(periodic_h4, pao)
    np.testing.assert_allclose(C_occ.T @ S @ pao.V_semi, 0.0, atol=1e-9)


def test_pao_s_orthonormal(periodic_h4, pao):
    S = np.asarray(periodic_h4["result"].overlap)
    np.testing.assert_allclose(
        pao.V_semi.T @ S @ pao.V_semi, np.eye(pao.n_vir), atol=1e-9
    )


def test_pao_fock_diagonal(periodic_h4, pao):
    F = np.asarray(periodic_h4["result"].fock)
    Fx = pao.V_semi.T @ F @ pao.V_semi
    np.testing.assert_allclose(Fx - np.diag(np.diag(Fx)), 0.0, atol=1e-8)


def test_n_vir(periodic_h4, pao):
    nbf = np.asarray(periodic_h4["result"].overlap).shape[0]
    assert pao.n_vir == nbf - pao.n_occ


def test_eps_match_canonical_virtuals(periodic_h4, pao):
    """Oracle: full-domain semicanonical PAO energies == SCF virtual energies."""
    eps_canon = np.sort(np.asarray(periodic_h4["result"].mo_energies)[pao.n_occ :])
    np.testing.assert_allclose(np.sort(pao.eps_vir), eps_canon, atol=1e-8)


def test_occ_plus_vir_complete(periodic_h4, pao):
    """Occupied + virtual projectors sum to identity (resolution of identity)."""
    S = np.asarray(periodic_h4["result"].overlap)
    C_occ = _occ(periodic_h4, pao)
    P_occ = C_occ @ C_occ.T @ S
    P_vir = pao.V_semi @ pao.V_semi.T @ S
    np.testing.assert_allclose(P_occ + P_vir, np.eye(S.shape[0]), atol=1e-8)
