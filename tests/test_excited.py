"""General CIS / TDA excited-state kernel (vibeqc.excited).

The third method on the ERIProvider seam (after MP2 + GF2/OVGF).  These pin the
spin-adapted A-matrix and the solver: an exact match to vibe-qc's libint-coupled
HF TDA (vibeqc.tddft.run_tddft_tda), the singlet/triplet split, matrix symmetry,
and the analytic one-pair limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.excited import cis_excitations, cis_matrix

_A2B = 1.8897259886


def _h2o_blocks():
    """RHF/STO-3G H2O: eps, n_occ, ovov=(ia|jb), oovv=(ij|ab), + the C++ refs."""
    from vibeqc._vibeqc_core import (Atom, BasisSet, Molecule, RHFOptions,
                                     compute_eri, run_rhf)
    mol = Molecule([Atom(8, [0, 0, 0]),
                    Atom(1, [0, 0.7572 * _A2B, 0.5865 * _A2B]),
                    Atom(1, [0, -0.7572 * _A2B, 0.5865 * _A2B])], 0, 1)
    bas = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, bas, RHFOptions())
    C = np.asarray(rhf.mo_coeffs)
    eps = np.asarray(rhf.mo_energies)
    nocc = 5
    eri = np.asarray(compute_eri(bas))
    Co, Cv = C[:, :nocc], C[:, nocc:]
    ovov = np.einsum("pqrs,pi,qa,rj,sb->iajb", eri, Co, Cv, Co, Cv, optimize=True)
    oovv = np.einsum("pqrs,pi,qj,ra,sb->ijab", eri, Co, Co, Cv, Cv, optimize=True)
    return mol, bas, eps, C, nocc, ovov, oovv


def test_singlet_cis_matches_hf_tda_to_machine_precision():
    """The general singlet kernel reproduces vibe-qc's libint HF TDA exactly —
    the reference-agnostic kernel and the production CIS agree on the same
    integrals."""
    from vibeqc.tddft import run_tddft_tda
    mol, bas, eps, C, nocc, ovov, oovv = _h2o_blocks()
    ref = run_tddft_tda(mol, bas, eps, C, nocc, n_states=6, functional=None)
    ref_e = np.array([s.excitation_energy for s in ref.states])
    got = cis_excitations(eps, nocc, ovov, oovv, spin="singlet", n_states=6)
    np.testing.assert_allclose(got.excitation_energies, ref_e, atol=1e-10)


def test_triplet_below_singlet():
    """For each manifold the lowest triplet lies below the lowest singlet
    (the exchange term lowers triplets — Hund's rule)."""
    _, _, eps, _, nocc, ovov, oovv = _h2o_blocks()
    s = cis_excitations(eps, nocc, ovov, oovv, spin="singlet", n_states=3)
    t = cis_excitations(eps, nocc, ovov, oovv, spin="triplet", n_states=3)
    assert t.excitation_energies[0] < s.excitation_energies[0]
    # eV convenience + dominant-transition helper are consistent.
    assert s.excitation_energies_ev[0] == pytest.approx(
        s.excitation_energies[0] * 27.211386245988)
    occ, vir, wt = s.dominant_transition(0)
    assert 0 <= occ < nocc and 0 <= vir and 0.0 < wt <= 1.0 + 1e-12


def test_cis_matrix_symmetric():
    _, _, eps, _, nocc, ovov, oovv = _h2o_blocks()
    for spin in ("singlet", "triplet"):
        A = cis_matrix(eps, nocc, ovov, oovv, spin=spin)
        np.testing.assert_allclose(A, A.T, atol=1e-12)


def test_one_pair_analytic_limit():
    """1-occ/1-vir: singlet = Δε + 2(ia|ia) − (ii|aa); triplet = Δε − (ii|aa);
    so singlet − triplet = 2(ia|ia)."""
    eps = np.array([-0.5, 0.4])
    g = 0.12   # (ia|ia)
    h = 0.20   # (ii|aa)
    ovov = np.array([[[[g]]]])
    oovv = np.array([[[[h]]]])
    s = cis_excitations(eps, 1, ovov, oovv, spin="singlet", n_states=1)
    t = cis_excitations(eps, 1, ovov, oovv, spin="triplet", n_states=1)
    assert s.excitation_energies[0] == pytest.approx(0.9 + 2 * g - h)
    assert t.excitation_energies[0] == pytest.approx(0.9 - h)
    assert (s.excitation_energies[0] - t.excitation_energies[0]) == pytest.approx(2 * g)


def test_unknown_spin_raises():
    eps = np.array([-0.5, 0.4])
    with pytest.raises(ValueError, match="singlet.*triplet|spin"):
        cis_matrix(eps, 1, np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1, 1)),
                   spin="quintet")
