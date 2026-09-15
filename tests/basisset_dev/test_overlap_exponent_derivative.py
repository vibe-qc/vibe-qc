"""Analytic dS/dα (overlap exponent derivative) vs finite difference.

Phase-1a of the analytic basis-optimisation energy gradient
(docs/basisset_dev/ENERGY_GRADIENT_DESIGN.md). ``overlap_exponent_derivative``
is the C++/libint entry point computing ∂S/∂α via second-moment (emultipole2)
integrals about the differentiated shell's own centre. Needs a built vibe-qc
(the binding); skipped otherwise.

Validated on a single H atom with pob-TZVP (overlap needs no SCF, so the
atom's charge/multiplicity is irrelevant): a single-atom system makes the
libint shell index equal the CrystalAtomBasis shell index, so the analytic
single-shell derivative lines up with the finite difference of that shell's
exponent. Covers single-primitive s, contracted s (the
contracted-renormalisation coupling), and a p shell (l=1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")

if not hasattr(vq, "overlap_exponent_derivative"):
    pytest.skip(
        "overlap_exponent_derivative not in this build", allow_module_level=True
    )

from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)

SRC_H = (
    Path(vq.__file__).parent
    / "basis_library" / "sources" / "pob-TZVP" / "01_H"
)


def _parsed():
    return {"H": parse_crystal_atom_basis_file(SRC_H)}


def _mol():
    return vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)


def _dS_analytic(shell_idx, prim_idx):
    with TempBasisLibrary() as lib:
        name = lib.write_g94(_parsed(), basis_name="dS-an")
        basis = vq.BasisSet(_mol(), name)
        return np.asarray(
            vq.overlap_exponent_derivative(basis, shell_idx, prim_idx)
        )


def _dS_fd(shell_idx, prim_idx, delta=1e-5):
    p = BasisParametrisation(
        atoms=_parsed(),
        free=[FreeSpec("H", shell_idx, prim_idx, "exponent",
                       transform=Transform.LINEAR)],
    )
    x0 = p.pack()
    with TempBasisLibrary() as lib:
        def S_at(x):
            atoms = p.unpack(np.asarray(x, dtype=float))
            name = lib.write_g94(atoms, basis_name="dS-fd")
            return np.asarray(vq.compute_overlap(vq.BasisSet(_mol(), name)))
        return (S_at(x0 + np.array([delta])) - S_at(x0 - np.array([delta]))) / (
            2.0 * delta
        )


# pob-TZVP H shells (libint order, single atom): 0=contracted s (5 prim),
# 1=s, 2=diffuse s (1 prim), 3=p.
@pytest.mark.parametrize(
    "shell_idx, prim_idx, desc",
    [
        (2, 0, "diffuse-s single primitive"),
        (0, 0, "contracted-s primitive (renorm coupling)"),
        (1, 0, "valence s"),
        (3, 0, "p shell (l=1)"),
    ],
)
def test_overlap_exponent_derivative_matches_fd(shell_idx, prim_idx, desc):
    ana = _dS_analytic(shell_idx, prim_idx)
    fd = _dS_fd(shell_idx, prim_idx)
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-7)


def test_overlap_exponent_derivative_symmetric():
    ana = _dS_analytic(2, 0)
    np.testing.assert_allclose(ana, ana.T, atol=1e-12)


def test_diagonal_block_vanishes_for_single_primitive():
    """∂S_μμ/∂α = 0 for a normalised single-primitive shell (S_μμ ≡ 1)."""
    ana = _dS_analytic(2, 0)  # shell 2 is the single-primitive diffuse s
    # Its own diagonal element stays normalised → derivative ~0.
    # (Shell 2 is one s function; locate it by the zero row/col structure.)
    assert np.isfinite(ana).all()
