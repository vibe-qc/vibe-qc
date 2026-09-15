"""Analytic d(μν|λσ)/dα (two-electron exponent derivative) vs FD.

Phase-1b of the analytic basis-optimisation energy gradient — the 4-index
piece. `eri_exponent_derivative` reuses the validated one-electron r²-bra
machinery on the coulomb engine (differentiated shell in position 1) and
symmetrises over the 4 ERI positions. Needs a built vibe-qc; skipped
otherwise. Validated on single atoms (no SCF needed): H (s, p) and C
(s, p, d).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")

if not hasattr(vq, "eri_exponent_derivative"):
    pytest.skip("eri_exponent_derivative not in this build", allow_module_level=True)

from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)

SOURCES = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"


def _atoms(sym, src):
    return {sym: parse_crystal_atom_basis_file(SOURCES / src)}


def _mol(Z, mult):
    return vq.Molecule([vq.Atom(Z, [0.0, 0.0, 0.0])], multiplicity=mult)


def _analytic(sym, src, mol, shell_idx, prim_idx):
    with TempBasisLibrary() as lib:
        name = lib.write_g94(_atoms(sym, src), basis_name="eri-an")
        b = vq.BasisSet(mol, name)
        n = b.nbasis
        flat = np.asarray(vq.eri_exponent_derivative(b, shell_idx, prim_idx))
        return flat.reshape(n, n, n, n)


def _fd(sym, src, mol, shell_idx, prim_idx, delta=1e-5):
    p = BasisParametrisation(
        atoms=_atoms(sym, src),
        free=[FreeSpec(sym, shell_idx, prim_idx, "exponent",
                       transform=Transform.LINEAR)],
    )
    x0 = p.pack()
    with TempBasisLibrary() as lib:
        def E(x):
            ax = p.unpack(np.asarray(x, dtype=float))
            name = lib.write_g94(ax, basis_name="eri-fd")
            return np.asarray(vq.compute_eri(vq.BasisSet(mol, name)))
        return (E(x0 + np.array([delta])) - E(x0 - np.array([delta]))) / (
            2.0 * delta
        )


_SYSTEMS = [("H", "01_H", 1, 2), ("C", "06_C", 6, 3)]
_CASES = [
    (sym, src, Z, mult, s)
    for (sym, src, Z, mult) in _SYSTEMS
    for s in range(len(parse_crystal_atom_basis_file(SOURCES / src).shells))
]


@pytest.mark.parametrize("sym, src, Z, mult, shell_idx", _CASES)
def test_eri_exponent_derivative_matches_fd(sym, src, Z, mult, shell_idx):
    mol = _mol(Z, mult)
    ana = _analytic(sym, src, mol, shell_idx, 0)
    fd = _fd(sym, src, mol, shell_idx, 0)
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-6)
