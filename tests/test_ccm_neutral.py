"""Neutral (Ewald) four-center for the CCM and the Madelung-gap resolution.

The neutral effective four-center ``g_eff = (μν|v_E|ρσ)``
(:func:`vibeqc.periodic.ccm.neutral.ccm_eri_neutral`, ``AICCM_ALGORITHM.md``
§13.7) is the RI density-fit of the charge-neutral torus Coulomb. These tests pin
its two defining properties and the scientific result that resolves the
four-center Madelung gap:

* ``g_eff`` is exactly 8-fold permutationally symmetric (the requirement of a
  single variational RHF/MP2 functional);
* in the molecular limit ``g_eff = (μν|ρσ)_full − ξ·S_μν·S_ρσ`` with
  ``ξ`` = the cell Madelung constant (``q_μν = S_μν``);
* that rank-1 background ``ξ·S⊗S`` becomes ``ξ·I⊗I`` in the MO basis and so
  contributes **exactly zero** to the correlation *numerator* **at a fixed
  reference** (the ``test_madelung_background_cancels_in_correlation`` gate uses
  one shared HF reference). This is a fixed-reference *diagnostic*, **not** an
  all-post-HF Madelung-invariance theorem: the background shifts the occ–virt
  denominators, so independent bare-vs-neutral references do not agree, and the
  neutral reference is required for ionic correlation (2026-06-22/24 round-2
  correction; the earlier "rigorously Madelung-robust" wording is retracted).

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014); Sun, Berkelbach,
McClain & Chan, J. Chem. Phys. 147, 164119 (2017) (periodic GDF).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    PeriodicSystem,
    compute_eri,
    compute_kinetic,
    compute_nuclear,
    compute_overlap,
)
from vibeqc.periodic.ccm import (
    CCMSystem,
    ccm_eri_neutral,
    ccm_neutral_background_constant,
    ccm_neutral_cderi,
    ccm_overlap,
)
from vibeqc.periodic.ccm.padded import ccm_eri_symmetric
from vibeqc.periodic.ccm.ri import (
    ccm_ri_j_neutral,
    ccm_ri_k_neutral,
    ccm_ri_tensors,
    ccm_ri_tensors_neutral,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2(box=12.0):
    return PeriodicSystem(3, np.diag([box, box, box]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)


def _rhf(basis, mol, eri, tol=1e-11):
    """Minimal spatial RHF returning (eps, C, n_occ)."""
    S = np.asarray(compute_overlap(basis))
    h = np.asarray(compute_kinetic(basis)) + np.asarray(compute_nuclear(basis, mol))
    ne = sum(int(a.Z) for a in mol.atoms)
    no = ne // 2
    w, U = np.linalg.eigh(S)
    X = U / np.sqrt(w)

    def diag(F):
        e, C = np.linalg.eigh(X.T @ F @ X)
        return e, X @ C

    eps, C = diag(h)
    D = 2 * C[:, :no] @ C[:, :no].T
    e_last = 0.0
    for _ in range(300):
        F = (h + np.einsum("mnrs,rs->mn", eri, D, optimize=True)
             - 0.5 * np.einsum("msrn,rs->mn", eri, D, optimize=True))
        E = 0.5 * np.sum(D * (h + F))
        eps, C = diag(F)
        D = 2 * C[:, :no] @ C[:, :no].T
        if abs(E - e_last) < tol:
            break
        e_last = E
    return eps, C, no


def _mp2_corr(eri, eps, C, no):
    """Closed-shell MP2 correlation from chemists' AO ERIs and MOs."""
    nmo = C.shape[1]
    mo = np.einsum("mnrs,mp,nq,ra,sb->pqab", eri, C, C, C, C, optimize=True)
    o, v = slice(0, no), slice(no, nmo)
    iajb = mo[o, v, o, v]
    D = (eps[:no, None, None, None] + eps[None, None, :no, None]
         - eps[None, no:, None, None] - eps[None, None, None, no:])
    return float(np.einsum("iajb,iajb->", iajb,
                           (2 * iajb - iajb.transpose(0, 3, 2, 1)) / D, optimize=True))


def _sym_errors(g):
    swaps = {"mn": (1, 0, 2, 3), "rs": (0, 1, 3, 2), "braket": (2, 3, 0, 1)}
    return {k: float(np.max(np.abs(g - np.transpose(g, ax)))) for k, ax in swaps.items()}


# ---------------------------------------------------------------------------
# 1. The background cancels exactly in correlation (analytic; the headline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xi", [0.0, 0.186, 0.5, 2.0])
def test_madelung_background_cancels_in_correlation(xi):
    """MP2 correlation is invariant under the exact rank-1 background ``ξ·S⊗S``.

    In the MO basis ``ξ·S⊗S → ξ·I⊗I`` (orthonormal MOs), which has no
    occupied–virtual element, so it contributes 0 to ``(ia|jb)`` and hence to the
    correlation energy — for *any* ``ξ``. Machine-precision; the resolution of
    the four-center ionic Madelung gap for post-HF.
    """
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    mol = ccm.supercell
    eri = np.asarray(compute_eri(ccm.basis))
    S = np.asarray(compute_overlap(ccm.basis))
    g = eri - xi * np.einsum("mn,rs->mnrs", S, S)
    eps, C, no = _rhf(ccm.basis, mol, eri)        # one shared (full) HF reference
    assert _mp2_corr(g, eps, C, no) == pytest.approx(_mp2_corr(eri, eps, C, no), abs=1e-12)


# ---------------------------------------------------------------------------
# 2. The neutral four-center g_eff (cderi density-fit of v_E)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_geff_eight_fold_symmetric():
    """``g_eff`` is exactly 8-fold permutationally symmetric (1-D and isolated)."""
    for unit, nrep in [(_h2(), (1, 1, 1)),
                       (PeriodicSystem(3, np.diag([6.0, 30.0, 30.0]),
                                       [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1),
                        (3, 1, 1))]:
        ccm = CCMSystem(unit, nrep, "sto-3g")
        g = ccm_eri_neutral(ccm, ke_cutoff=120.0)
        for name, err in _sym_errors(g).items():
            assert err < 1e-10, f"{name} symmetry broken: {err:.2e}"


@pytest.mark.slow
def test_geff_background_decomposition():
    """Isolated limit: ``g_eff ≈ (μν|ρσ)_full − ξ·S⊗S`` with ξ ≈ cell Madelung const."""
    ccm = CCMSystem(_h2(box=15.0), (1, 1, 1), "sto-3g")
    g = ccm_eri_neutral(ccm, ke_cutoff=200.0)
    eri = np.asarray(compute_eri(ccm.basis))
    S = np.asarray(compute_overlap(ccm.basis))
    SS = np.einsum("mn,rs->mnrs", S, S)
    xi_fit = float(np.sum((eri - g) * SS) / np.sum(SS * SS))
    xi_cell = ccm_neutral_background_constant(ccm)
    # ξ from the fit matches the cell Madelung constant to the cderi RI error
    assert xi_fit == pytest.approx(xi_cell, rel=0.05)
    # the residual after removing the rank-1 background is the (small) RI error
    assert np.max(np.abs(g - (eri - xi_fit * SS))) < 5e-3


@pytest.mark.slow
def test_neutral_cderi_ri_is_consistent_bare_union_is_not():
    """The neutral cderi RI-J/K reproduce the four-center; the bare union RI-J does not.

    RI-consistency (§13.7): the neutral kernel ``v_E`` is separable, so its cderi
    ``L`` gives ``J = LᵀL:D`` that equals the neutral four-center to machine ε.
    The bare-1/r four-center is non-separable, so the eq-13 union RI-J
    (:func:`ccm_ri_tensors`) floors at ~1e-3 off the bare four-center — the
    documented gap this resolves.
    """
    unit = PeriodicSystem(3, np.diag([6.0, 30.0, 30.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(unit, (3, 1, 1), "sto-3g")
    n = ccm.nbf
    rng = np.random.default_rng(0)
    D = rng.standard_normal((n, n))
    D = D + D.T                                   # the identity holds for any symmetric D

    # neutral cderi: RI reproduces the four-center it defines (separable)
    L = ccm_neutral_cderi(ccm, ke_cutoff=150.0)
    g_neu = np.einsum("Pmn,Prs->mnrs", L, L, optimize=True)
    j_neu_4c = np.einsum("mnrs,rs->mn", g_neu, D, optimize=True)
    k_neu_4c = np.einsum("msrn,rs->mn", g_neu, D, optimize=True)
    rel_j = np.linalg.norm(ccm_ri_j_neutral(L, D) - j_neu_4c) / np.linalg.norm(j_neu_4c)
    rel_k = np.linalg.norm(ccm_ri_k_neutral(L, D) - k_neu_4c) / np.linalg.norm(k_neu_4c)
    assert rel_j < 1e-12 and rel_k < 1e-12
    # ccm_ri_tensors_neutral returns the same RI-consistent tensor
    assert np.allclose(ccm_ri_tensors_neutral(ccm, ke_cutoff=150.0), L, atol=0, rtol=0)

    # bare-1/r union RI-J is NOT RI-consistent with the bare four-center (~1e-3)
    B, Vinv = ccm_ri_tensors(ccm)
    j_ri_bare = np.einsum("mnP,P->mn", B,
                          Vinv @ np.einsum("rsQ,rs->Q", B, D, optimize=True), optimize=True)
    j_bare_4c = np.einsum("mnrs,rs->mn", ccm_eri_symmetric(ccm), D, optimize=True)
    rel_bare = np.linalg.norm(j_ri_bare - j_bare_4c) / np.linalg.norm(j_bare_4c)
    assert rel_bare > 1e-4, f"bare union RI-J unexpectedly consistent ({rel_bare:.1e})"
    assert rel_j < rel_bare           # the neutral route is RI-consistent, the bare one is not


@pytest.mark.slow
def test_neutral_geff_correlation_matches_full():
    """MP2 correlation from the actual neutral ``g_eff`` matches the full ERIs.

    The numerical counterpart of the analytic cancellation: the only difference
    is the cderi RI fitting error, not the background.
    """
    ccm = CCMSystem(_h2(box=15.0), (1, 1, 1), "sto-3g")
    mol = ccm.supercell
    eri = np.asarray(compute_eri(ccm.basis))
    g = ccm_eri_neutral(ccm, ke_cutoff=200.0)
    eps, C, no = _rhf(ccm.basis, mol, eri)
    assert _mp2_corr(g, eps, C, no) == pytest.approx(_mp2_corr(eri, eps, C, no), abs=2e-3)


def test_neutral_cderi_jk_torus_translation_invariant():
    """The neutral cderi's implied J/K commute with the torus translation.

    Regression for the 2026-07-02 cross-torus truncation fix
    (``_supercell_wrap_lat_opts``): the Γ-supercell cderi build stores AO
    pairs at absolute supercell positions, and with the flat ~15-bohr lattice
    cutoff an 18-bohr (3,1,1) supercell lost the image cell that wraps edge
    pairs back onto the torus — the fitted kernel's J/K then broke torus
    translation symmetry at the 4e-2 level (and mismatched the multi-k GDF by
    1.1e-4 Ha/cell on complex-q meshes). The wrap-aware auto-widened cutoff
    restores invariance to machine ε; pinned here at the kernel level,
    independent of any SCF.

    The cell is declared ``dim=3``; the geometry is unchanged (6-bohr chain axis,
    15-bohr transverse). What this pins is the supercell extent outrunning the flat
    lattice cutoff, which has nothing to do with dimensionality. The neutral cderi
    refuses ``dim < 3`` outright since 2026-07-10, because its reciprocal mesh pins
    every non-periodic axis at ``G_perp = 0`` and the kernel it feeds is then a
    transverse-uniform sheet term rather than ``1/r``. That refusal is asserted here
    too, so the low-D path keeps its coverage.
    """
    from vibeqc import Atom, PeriodicSystem
    from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral

    atoms = [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])]
    cell = PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]), atoms, 0, 1)
    ccm = CCMSystem(cell, (3, 1, 1), "sto-3g")  # 18-bohr supercell: failing size

    lowd = CCMSystem(
        PeriodicSystem(1, np.diag([6.0, 15.0, 15.0]), atoms, 0, 1), (3, 1, 1), "sto-3g")
    with pytest.raises(NotImplementedError, match="transverse-collapsed"):
        ccm_neutral_cderi(lowd)
    n_mu, n_c = 2, ccm.n_cells
    n = n_c * n_mu
    P = np.zeros((n, n))
    for c in range(n_c):
        for m in range(n_mu):
            P[((c + 1) % n_c) * n_mu + m, c * n_mu + m] = 1.0
    rng = np.random.default_rng(7)
    D0 = rng.standard_normal((n, n))
    D0 = D0 + D0.T
    D_sym = sum(
        np.linalg.matrix_power(P, t) @ D0 @ np.linalg.matrix_power(P, t).T
        for t in range(n_c)) / n_c
    L = ccm_neutral_cderi(ccm)
    J = ccm_ri_j_neutral(L, D_sym)
    K = ccm_ri_k_neutral(L, D_sym)
    assert np.max(np.abs(P @ J @ P.T - J)) < 1e-10
    assert np.max(np.abs(P @ K @ P.T - K)) < 1e-10
