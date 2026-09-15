"""End-to-end UHF (open-shell) analytic energy gradient vs full-energy FD.

The unrestricted counterpart of ``test_energy_gradient_analytic.py``.
``energy_gradient_analytic_uhf`` reuses the (spin-independent) closed-form
integral exponent derivatives but contracts them with per-spin densities and
energy-weighted densities (W_σ = P_σ·F_σ·P_σ), so it reaches the open-shell
atoms — C, N, O, … — the pob basis recipe needs. Validated against the re-SCF
total-energy central difference across two multiplicities (triplet, quartet)
and all of s/p/d.

Needs a built vibe-qc with the exponent-derivative bindings; skipped otherwise.
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
from vibeqc.basis_optimization import energy_gradient_analytic_uhf  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)

SRC = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"
TOL = 5e-5  # Ha per unit parameter; analytic vs full-energy UHF FD


def _check(sym, src, Z, mult, specs):
    atoms = {sym: parse_crystal_atom_basis_file(SRC / src)}
    par = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec(sym, i, p, "exponent", transform=Transform.LOG)
              for (i, p) in specs],
    )
    x0 = par.pack()

    def mol():
        return vq.Molecule([vq.Atom(Z, [0.0, 0.0, 0.0])], multiplicity=mult)

    with TempBasisLibrary() as lib:
        g_an = energy_gradient_analytic_uhf(par, mol, lib, vq.run_uhf, x0)

        def etot(x):
            ax = par.unpack(np.asarray(x, dtype=float))
            nm = lib.write_g94(ax, basis_name="uhf-ref")
            m = mol()
            return float(vq.run_uhf(m, vq.BasisSet(m, nm)).energy)

        h = 1e-4
        g_fd = np.zeros(len(x0))
        for i in range(len(x0)):
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            g_fd[i] = (etot(xp) - etot(xm)) / (2.0 * h)

    np.testing.assert_allclose(g_an, g_fd, atol=TOL, rtol=1e-3)


def test_uhf_carbon_triplet_spd():
    # C 3P: s, p, and d shells in one open-shell assembly.
    _check("C", "06_C", 6, 3, [(0, 0), (3, 0), (6, 0), (7, 0)])


def test_uhf_nitrogen_quartet():
    # N 4S: higher spin multiplicity.
    _check("N", "07_N", 7, 4, [(0, 0), (4, 0)])
