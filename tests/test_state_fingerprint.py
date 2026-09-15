"""BUG 88 — state fingerprint + multi-guess convergence tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import vibeqc as vq
from vibeqc import Atom, BasisSet, Molecule, UHFOptions, run_uhf
from vibeqc.state_fingerprint import compute_state_fingerprint


def _make_mol(atoms_bohr, charge=0, multiplicity=1):
    return Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr], charge, multiplicity)


FECL3_ATOMS = [
    (26, [0.0, 0.0, 0.0]), (17, [0.0, 0.0, 4.025916]),
    (17, [3.486550, 0.0, -2.012958]), (17, [-3.486550, 0.0, -2.012958]),
]


def test_fingerprint_same_state_is_identical():
    mol = _make_mol(FECL3_ATOMS, multiplicity=6)
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions(); opts.restart_opts.enabled = False
    r1 = run_uhf(mol, basis, opts)
    r2 = run_uhf(mol, basis, opts)
    ne = mol.n_electrons(); mult = mol.multiplicity
    na = (ne + mult - 1)//2; nb = (ne - mult + 1)//2
    fp1 = compute_state_fingerprint(energy=float(r1.energy), n_alpha=na, n_beta=nb,
                                     s_squared=float(r1.s_squared), molecule_symbols=["Fe","Cl","Cl","Cl"])
    fp2 = compute_state_fingerprint(energy=float(r2.energy), n_alpha=na, n_beta=nb,
                                     s_squared=float(r2.s_squared), molecule_symbols=["Fe","Cl","Cl","Cl"])
    assert fp1 == fp2

def test_fingerprint_different_s2_differs():
    fp1 = compute_state_fingerprint(energy=-100.0, n_alpha=10, n_beta=8, s_squared=8.76)
    fp2 = compute_state_fingerprint(energy=-100.0, n_alpha=10, n_beta=8, s_squared=3.76)
    assert fp1 != fp2

def test_fingerprint_is_64_hex_chars():
    fp = compute_state_fingerprint(energy=-100.0, n_alpha=5, n_beta=3, s_squared=3.76, molecule_symbols=["Fe"])
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)

def test_multi_guess_empty_is_single_scf():
    mol = _make_mol(FECL3_ATOMS, multiplicity=6)
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions(); opts.multi_guess_seeds = []
    assert run_uhf(mol, basis, opts).converged

def test_multi_guess_with_seeds_converges():
    mol = _make_mol(FECL3_ATOMS, multiplicity=6)
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions(); opts.max_iter = 120; opts.multi_guess_seeds = [1, 2]
    assert run_uhf(mol, basis, opts).converged

def test_multi_guess_options_roundtrip():
    o = UHFOptions()
    assert list(o.multi_guess_seeds) == []
    o.multi_guess_seeds = [42, 99]
    assert list(o.multi_guess_seeds) == [42, 99]


def test_uhf_multi_guess_value_copy_preserves_ecp_provenance(monkeypatch):
    """Alternate guesses must use the same ECP Hamiltonian as the primary."""
    mol = vq.Molecule([vq.Atom(14, [0.0, 0.0, 0.0])], 0, 3)
    basis = vq.BasisSet(mol, "def2-svp")
    opts = vq.UHFOptions()
    opts.ecp_centers = [vq.ECPCenter(Z=14, xyz=[0.0, 0.0, 0.0])]
    opts.ecp_library = "lanl2dz"
    opts.ecp_total_ncore = 10
    opts.multi_guess_seeds = [17]
    calls = []

    def fake_run(_molecule, _basis, captured):
        calls.append(captured)
        return SimpleNamespace(
            converged=True,
            energy=-3.6 - 0.01 * len(calls),
            s_squared=2.0,
            internal_instability=False,
            ecp_total_ncore=10,
        )

    monkeypatch.setattr(vq, "_run_uhf_cxx", fake_run)
    vq.run_uhf(mol, basis, opts)

    assert len(calls) == 2
    primary, alternate = calls
    assert primary is opts
    assert alternate is not opts
    assert alternate.ecp_library == "lanl2dz"
    assert alternate.ecp_total_ncore == 10
    assert len(alternate.ecp_centers) == 1
    assert alternate.ecp_centers[0].Z == 14


@pytest.mark.parametrize(
    ("requested", "expected_value", "expected_explicit"),
    [
        (None, True, False),
        (True, True, True),
        (False, False, True),
    ],
)
def test_uks_multi_guess_value_copy_preserves_stability_and_plus_u(
    monkeypatch, requested, expected_value, expected_explicit
):
    """Alternate guesses inherit the complete UKS option state."""
    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UKSOptions()
    opts.functional = "LDA"
    opts.multi_guess_seeds = [17]
    if requested is not None:
        opts.stability_check = requested

    calls = []

    def fake_run(_molecule, _basis, captured):
        calls.append(captured)
        return SimpleNamespace(
            converged=True,
            energy=-0.50 - 0.01 * len(calls),
            s_squared=0.75,
            internal_instability=False,
        )

    monkeypatch.setattr(vq, "_run_uks_cxx", fake_run)
    vq.run_uks(
        mol,
        basis,
        opts,
        dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
    )

    assert len(calls) == 2
    primary, alternate = calls
    assert primary is opts
    assert alternate is not opts
    for copied in calls:
        assert copied.stability_check is expected_value
        assert copied._stability_check_explicit is expected_explicit
        assert len(copied.dft_plus_u_sites) == 1
        site = copied.dft_plus_u_sites[0]
        assert (site.atom_index, site.l) == (0, 0)
        assert site.U_eff_au == pytest.approx(2.0 / 27.211386245988)
        assert [list(group) for group in copied.dft_plus_u_ao_groups] == [[0]]
