"""End-to-end analytic SCF-energy gradient vs full-energy finite difference.

Phase-1 capstone: ``energy_gradient_analytic`` feeds the Pulay assembly with
the closed-form libint exponent derivatives (∂S/∂T/∂V/∂ERI) instead of
finite-differencing the integrals, and reproduces the re-SCF total-energy
central difference. Covers the three things the integral-derivative unit
tests don't:

* the optimiser-param → libint-shell mapping (single atom, 1:1),
* multi-atom summing — one exponent driving every atom of an element (H₂),
* multi-element + a d polarisation shell in the assembly (H₂O).

Needs a built vibe-qc; skipped otherwise. The build-free assembly math is
covered separately by ``test_energy_gradient_assembly.py`` (mock RHF).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")

if not hasattr(vq, "eri_exponent_derivative"):
    pytest.skip("exponent-derivative bindings not in this build",
                allow_module_level=True)

from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization import energy_gradient_analytic  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)

SRC = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"
TOL = 5e-5  # Ha per unit parameter; analytic vs full-energy FD


def _load(*syms_srcs):
    return {sym: parse_crystal_atom_basis_file(SRC / src) for sym, src in syms_srcs}


def _check(atoms, mol_factory, specs, field="exponent", transform=Transform.LOG):
    par = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec(s, i, p, field, transform=transform)
              for (s, i, p) in specs],
    )
    x0 = par.pack()
    with TempBasisLibrary() as lib:
        g_an = energy_gradient_analytic(par, mol_factory, lib, vq.run_rhf, x0)

        def etot(x):
            ax = par.unpack(np.asarray(x, dtype=float))
            nm = lib.write_g94(ax, basis_name="egrad-ref")
            m = mol_factory()
            return float(vq.run_rhf(m, vq.BasisSet(m, nm)).energy)

        # Per-parameter step: a fixed absolute step is a huge *relative*
        # perturbation for a tiny core contraction coefficient (e.g. O 1s,
        # d≈2e-4) and pushes the reference deep into the nonlinear regime.
        rel = 1e-4 if field == "exponent" else 1e-3
        g_fd = np.zeros(len(x0))
        for i in range(len(x0)):
            h = max(1e-7, rel * abs(float(x0[i])))
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            g_fd[i] = (etot(xp) - etot(xm)) / (2.0 * h)

    np.testing.assert_allclose(g_an, g_fd, atol=TOL, rtol=1e-3)


def test_single_atom_be():
    be = _load(("Be", "04_Be"))
    n = len(be["Be"].shells)
    _check(be, lambda: vq.Molecule([vq.Atom(4, [0.0, 0.0, 0.0])], multiplicity=1),
           [("Be", 0, 0), ("Be", n - 1, 0)])


def test_multi_atom_h2():
    h = _load(("H", "01_H"))
    n = len(h["H"].shells)
    _check(h, lambda: vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], multiplicity=1),
        [("H", 2, 0), ("H", n - 1, 0)])


def test_multi_element_dshell_h2o():
    wat = _load(("O", "08_O"), ("H", "01_H"))
    n = len(wat["O"].shells)
    _check(wat, lambda: vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 1.43, 1.11]),
         vq.Atom(1, [0.0, -1.43, 1.11])], multiplicity=1),
        [("O", n - 1, 0), ("O", 0, 0), ("H", 2, 0)])


def _h2o():
    return vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 1.43, 1.11]),
         vq.Atom(1, [0.0, -1.43, 1.11])], multiplicity=1)


def test_coefficient_param_h2o():
    """Contraction-coefficient gradient (augmented-basis direct term) on a
    multi-element system — exercises the AO-offset bookkeeping past O's d shell
    and the 2×H atom summing."""
    wat = _load(("O", "08_O"), ("H", "01_H"))
    # Coefficient of the first primitive of O's and H's contracted s shells.
    _check(wat, _h2o, [("O", 0, 0), ("H", 0, 0)],
           field="coeff", transform=Transform.LINEAR)
