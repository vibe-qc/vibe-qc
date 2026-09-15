"""BasisSet.shells() — Python-facing basis metadata.

Exposed for external-format writers (molden, NWChem, ...). Every ShellInfo
record is a flat snapshot of one contracted Gaussian: angular momentum,
atom index, primitive exponents, contraction coefficients, origin.
"""

from __future__ import annotations

import pytest

from vibeqc import Atom, BasisSet, Molecule


def _h2() -> Molecule:
    # H-H at 0.74 Å = 1.3984 bohr.
    return Molecule([
        Atom(1, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 0.0, 1.3984]),
    ])


def _h2o() -> Molecule:
    return Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])


def test_shells_basic_sto3g_h2():
    """H2/STO-3G: 2 shells, 1 s-shell per atom, 3 primitives each."""
    mol = _h2()
    basis = BasisSet(mol, "sto-3g")

    shells = basis.shells()

    assert len(shells) == 2
    assert basis.nshells == 2
    assert basis.nbasis == 2

    for idx, shell in enumerate(shells):
        assert shell.l == 0
        assert shell.atom_index == idx
        assert len(shell.exponents) == 3
        assert len(shell.coefficients) == 3
        # STO-3G H exponents (classic Pople values).
        assert shell.exponents[0] == pytest.approx(3.42525091, rel=1e-6)


def test_shells_match_atom_positions():
    """Each shell.origin must equal the host atom's Cartesian position."""
    mol = _h2o()
    basis = BasisSet(mol, "6-31g*")
    atoms = mol.atoms

    for shell in basis.shells():
        atom = atoms[shell.atom_index]
        assert shell.origin[0] == pytest.approx(atom.xyz[0])
        assert shell.origin[1] == pytest.approx(atom.xyz[1])
        assert shell.origin[2] == pytest.approx(atom.xyz[2])


def test_shells_pure_spherical_for_d():
    """BasisSet forces pure spherical harmonics for L >= 2 (5d, 7f, ...)."""
    mol = _h2o()
    basis = BasisSet(mol, "6-31g*")

    d_shells = [s for s in basis.shells() if s.l >= 2]
    assert d_shells, "6-31g* on H2O must include at least one d shell on O"
    for shell in d_shells:
        assert shell.pure is True


def test_shells_angular_momentum_counts_match_nbasis():
    """Sum over shells of (2L+1 if pure else (L+1)(L+2)/2) == nbasis()."""
    mol = _h2o()
    basis = BasisSet(mol, "6-31g*")

    n_from_shells = 0
    for shell in basis.shells():
        if shell.pure:
            n_from_shells += 2 * shell.l + 1
        else:
            n_from_shells += (shell.l + 1) * (shell.l + 2) // 2

    assert n_from_shells == basis.nbasis


def test_shells_exponents_coefficients_same_length():
    """Every contraction: len(exponents) == len(coefficients)."""
    mol = _h2o()
    basis = BasisSet(mol, "cc-pvdz")

    for shell in basis.shells():
        assert len(shell.exponents) == len(shell.coefficients)
        assert len(shell.exponents) > 0


def test_shells_atom_index_in_range():
    """atom_index is a valid index into mol.atoms for every shell."""
    mol = _h2o()
    basis = BasisSet(mol, "def2-tzvp")
    n_atoms = len(mol.atoms)

    for shell in basis.shells():
        assert 0 <= shell.atom_index < n_atoms


def test_shellinfo_repr_is_informative():
    """repr(shell) should include L (as spdfg), atom, and n_prim."""
    mol = _h2()
    basis = BasisSet(mol, "sto-3g")
    r = repr(basis.shells()[0])
    assert "s" in r
    assert "atom=0" in r
    assert "n_prim=3" in r
