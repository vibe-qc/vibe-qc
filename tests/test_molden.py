"""Molden writer — round-trip validation against PySCF.

PySCF's ``tools.molden.load`` parses a molden file back into a PySCF Mole
+ MO coefficient matrix. Re-evaluating the energy from those coefficients
must reproduce the vibe-qc SCF energy to chemical-accuracy tolerance. If
either the basis listing or the MO coefficient reordering is wrong, the
round-tripped energy will drift noticeably — so this test catches format
bugs at the level that matters for actual viewer output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    InitialGuess,
    Molecule,
    UHFOptions,
    run_rhf,
    run_uhf,
    write_molden,
)


pyscf = pytest.importorskip("pyscf")
pyscf_molden = pytest.importorskip("pyscf.tools.molden")


def _h2_mol() -> Molecule:
    return Molecule([
        Atom(1, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 0.0, 1.3984]),
    ])


def _h2o_mol() -> Molecule:
    return Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])


def _run_pyscf_and_compute_energy_from_mos(
    molden_path: Path,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Re-compute RHF energy from the MOs stored in the molden file.

    PySCF reads the molden file, rebuilds Mole + basis, and hands us the
    orbital coefficients in *its* AO convention. We evaluate

        E = 2 Σ_i <i|h|i> + Σ_ij (2 J_ij − K_ij) + E_nuc

    which is equivalent to the standard RHF expression for occupied orbitals
    only. This is a fair test that the MO coefficients, energies, and the
    basis survived the round trip.
    """
    mol, mo_energy, mo_coeff, mo_occ, _, _ = pyscf_molden.load(str(molden_path))

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    S = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e", aosym="s1")

    n_occ = int(round(mo_occ.sum() / 2))
    C_occ = mo_coeff[:, :n_occ]
    dm = 2.0 * C_occ @ C_occ.T

    e_core = np.einsum("ij,ij->", dm, h_core)
    # 2J - K from density.
    J = np.einsum("ijkl,kl->ij", eri, dm)
    K = np.einsum("ikjl,kl->ij", eri, dm)
    e_ee = 0.5 * np.einsum("ij,ij->", dm, J - 0.5 * K)
    e_nuc = mol.energy_nuc()

    return e_core + e_ee + e_nuc, mo_energy, mo_coeff


def test_write_molden_h2_sto3g_roundtrip(tmp_path: Path) -> None:
    """Simplest case: H2/STO-3G, only s shells — no AO reordering needed."""
    mol = _h2_mol()
    basis = BasisSet(mol, "sto-3g")
    result = run_rhf(mol, basis)
    assert result.converged

    out = tmp_path / "h2.molden"
    write_molden(out, mol, basis, result, title="H2/STO-3G")

    assert out.is_file()
    text = out.read_text()
    assert "[Molden Format]" in text
    assert "[Atoms] (AU)" in text
    assert "[GTO]" in text
    assert "[MO]" in text
    assert "Spin= Alpha" in text

    e_rt, _, _ = _run_pyscf_and_compute_energy_from_mos(out)
    assert e_rt == pytest.approx(result.energy, abs=1e-8)


def test_write_molden_h2o_pople_roundtrip(tmp_path: Path) -> None:
    """H2O/6-31G*: exercises p and d reordering."""
    mol = _h2o_mol()
    basis = BasisSet(mol, "6-31g*")
    result = run_rhf(mol, basis)
    assert result.converged

    out = tmp_path / "h2o.molden"
    write_molden(out, mol, basis, result)
    assert "[5D]" in out.read_text()

    e_rt, mo_energy, _ = _run_pyscf_and_compute_energy_from_mos(out)
    assert e_rt == pytest.approx(result.energy, abs=1e-8)

    # Orbital energies should also survive to floating-point precision —
    # molden stores them explicitly so no recomputation drift here.
    assert np.allclose(
        np.sort(np.asarray(result.mo_energies)),
        np.sort(mo_energy),
        atol=1e-10,
    )


def test_write_molden_h2o_ccpvdz_roundtrip(tmp_path: Path) -> None:
    """H2O/cc-pVDZ: second d-shell check and more primitives."""
    mol = _h2o_mol()
    basis = BasisSet(mol, "cc-pvdz")
    result = run_rhf(mol, basis)
    assert result.converged

    out = tmp_path / "h2o_ccpvdz.molden"
    write_molden(out, mol, basis, result)

    e_rt, _, _ = _run_pyscf_and_compute_energy_from_mos(out)
    assert e_rt == pytest.approx(result.energy, abs=1e-8)


def test_write_molden_uhf_has_two_spin_blocks(tmp_path: Path) -> None:
    """UHF output must emit Alpha then Beta MO blocks."""
    # OH radical — doublet, 9 electrons; uses SAD initial guess which the
    # SCF tests elsewhere confirm converges reliably.
    A2B = 1.0 / 0.529177210903
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.97 * A2B, 0.0, 0.0]),
        ],
        multiplicity=2,
    )
    basis = BasisSet(mol, "6-31g*")
    opts = UHFOptions()
    opts.initial_guess = InitialGuess.SAD
    result = run_uhf(mol, basis, opts)
    assert result.converged

    out = tmp_path / "oh.molden"
    write_molden(out, mol, basis, result)

    text = out.read_text()
    assert text.count("Spin= Alpha") > 0
    assert text.count("Spin= Beta") > 0
    # One (Alpha) + one (Beta) block per MO — sum should equal 2*nmo.
    n_mo = len(result.mo_energies_alpha)
    assert text.count("Sym=") == 2 * n_mo


def test_write_molden_rejects_bad_result(tmp_path: Path) -> None:
    mol = _h2_mol()
    basis = BasisSet(mol, "sto-3g")

    class BogusResult:
        pass

    with pytest.raises(TypeError, match="mo_coeffs"):
        write_molden(tmp_path / "nope.molden", mol, basis, BogusResult())
