"""Analytic dT/dα and dV/dα (kinetic / nuclear exponent derivatives) vs FD.

Phase-1b of the analytic basis-optimisation energy gradient. The kinetic and
nuclear-attraction integral derivatives need the r²-weighted bra, built from a
cartesian l+2 libint shell mapped back to the spherical μ axis via a measured
(do_enforce=false-consistent) cart transform — see
docs/basisset_dev/ENERGY_GRADIENT_DESIGN.md and
cpp/src/basis_param_gradient.cpp. Needs a built vibe-qc; skipped otherwise.

Validated on single atoms (overlap/kinetic/nuclear need no SCF) so the libint
shell index equals the CrystalAtomBasis shell index: H (s + p) and C (s, p,
*and* d), covering l = 0, 1, 2 incl. contracted shells and the
contracted-renormalisation response.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")

if not (hasattr(vq, "kinetic_exponent_derivative")
        and hasattr(vq, "nuclear_exponent_derivative")):
    pytest.skip("exponent-derivative bindings not in this build",
                allow_module_level=True)

from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)

SOURCES = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"


def _atom(symbol, src_file):
    return {symbol: parse_crystal_atom_basis_file(SOURCES / src_file)}


def _mol(Z, mult):
    return vq.Molecule([vq.Atom(Z, [0.0, 0.0, 0.0])], multiplicity=mult)


def _analytic(kind, atoms, mol, shell_idx, prim_idx):
    with TempBasisLibrary() as lib:
        name = lib.write_g94(atoms, basis_name="op-an")
        basis = vq.BasisSet(mol, name)
        if kind == "T":
            return np.asarray(
                vq.kinetic_exponent_derivative(basis, shell_idx, prim_idx))
        return np.asarray(
            vq.nuclear_exponent_derivative(basis, mol, shell_idx, prim_idx))


def _fd(kind, symbol, src_file, mol, shell_idx, prim_idx, delta=1e-5):
    p = BasisParametrisation(
        atoms=_atom(symbol, src_file),
        free=[FreeSpec(symbol, shell_idx, prim_idx, "exponent",
                       transform=Transform.LINEAR)],
    )
    x0 = p.pack()
    with TempBasisLibrary() as lib:
        def M_at(x):
            atoms = p.unpack(np.asarray(x, dtype=float))
            name = lib.write_g94(atoms, basis_name="op-fd")
            b = vq.BasisSet(mol, name)
            if kind == "T":
                return np.asarray(vq.compute_kinetic(b))
            return np.asarray(vq.compute_nuclear(b, mol))
        return (M_at(x0 + np.array([delta])) - M_at(x0 - np.array([delta]))) / (
            2.0 * delta
        )


# (symbol, source file, Z, multiplicity) — single-atom systems whose shells
# span s/p/d (C carries a d polarisation shell in pob-TZVP).
_SYSTEMS = [("H", "01_H", 1, 2), ("C", "06_C", 6, 3)]


def _shell_count(src_file):
    return len(parse_crystal_atom_basis_file(SOURCES / src_file).shells)


_CASES = [
    (kind, sym, src, Z, mult, s)
    for kind in ("T", "V")
    for (sym, src, Z, mult) in _SYSTEMS
    for s in range(_shell_count(src))
]


@pytest.mark.parametrize("kind, sym, src, Z, mult, shell_idx", _CASES)
def test_op_exponent_derivative_matches_fd(kind, sym, src, Z, mult, shell_idx):
    mol = _mol(Z, mult)
    ana = _analytic(kind, _atom(sym, src), mol, shell_idx, 0)
    fd = _fd(kind, sym, src, mol, shell_idx, 0)
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-6)
