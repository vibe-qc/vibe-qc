"""End-to-end UKS (open-shell DFT) analytic basis-parameter gradient vs FD.

The spin-polarised counterpart of ``test_energy_gradient_rks.py``:
``energy_gradient_analytic_uks`` combines the UHF per-spin Pulay assembly with
the polarised grid XC term. Reaches the open-shell atoms (C, N, …) the pob
recipe optimises at DFT level.

Exponents are evaluated *off* the reference basis (the rigorous case — at x0 the
∂lnN_c reference/current distinction is invisible, and the open-shell SCF
re-convergence makes the at-x0 FD reference noisier than the analytic value);
coefficients use a valence shell (the core shell's tiny coeffs hit .g94 FD
precision). Needs a built vibe-qc with the exponent-derivative bindings.
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
from vibeqc.basis_optimization import energy_gradient_analytic_uks  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)

SRC = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"
TOL = 6e-5  # grid + open-shell-SCF accuracy


def _check(
    sym,
    src,
    Z,
    mult,
    functional,
    field,
    specs,
    transform,
    shift=0.0,
    fd_rel=None,
):
    atoms = {sym: parse_crystal_atom_basis_file(SRC / src)}
    par = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec(sym, i, p, field, transform=transform) for (i, p) in specs],
    )

    def mol():
        return vq.Molecule([vq.Atom(Z, [0.0, 0.0, 0.0])], multiplicity=mult)

    with TempBasisLibrary() as lib:
        x = par.pack() + shift
        g_an = energy_gradient_analytic_uks(par, mol, lib, x, functional=functional)

        def etot(xx):
            ax = par.unpack(np.asarray(xx, dtype=float))
            nm = lib.write_g94(ax, basis_name="uks-ref")
            m = mol()
            o = vq.UKSOptions(); o.functional = functional
            return float(vq.run_uks(m, vq.BasisSet(m, nm), o).energy)

        # rel=1e-3 keeps the central difference in the truncation-dominated
        # regime. The SCF energy carries a deterministic noise floor of a few
        # nHa (integral/AO screening-threshold discontinuities as the exponents
        # move), so at the previous rel=1e-4 the FD error on the small B3LYP
        # d-exponent component (~3e-4 Ha) reached ~6e-5 and crossed TOL while
        # the analytic value is fine: an h-sweep at tight SCF convergence shows
        # the textbook V-curve with |FD - analytic| ~ 1e-7 at rel=3e-3,
        # ~1e-5 at rel=1e-3, ~6e-5 at rel=1e-4 and ~2e-4 at rel=1e-5.
        rel = fd_rel if fd_rel is not None else 1e-3
        g_fd = np.zeros(len(x))
        for i in range(len(x)):
            h = max(1e-6, rel * abs(float(x[i])))
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            g_fd[i] = (etot(xp) - etot(xm)) / (2.0 * h)

    np.testing.assert_allclose(g_an, g_fd, atol=TOL, rtol=1e-3)


@pytest.mark.parametrize("functional", ["PBE", "B3LYP"])
def test_uks_carbon_triplet_exponent(functional):
    # C 3P valence p (shell 4) + d (shell 7) exponents, off the reference basis.
    # The B3LYP d-shell FD coordinate is step-size sensitive at 1e-4 because
    # the open-shell grid/SCF noise is comparable to the displacement.
    fd_rel = 1e-3 if functional == "B3LYP" else None
    _check("C", "06_C", 6, 3, functional, "exponent",
           [(4, 0), (7, 0)], Transform.LOG, shift=-0.3, fd_rel=fd_rel)


def test_uks_carbon_triplet_coefficient():
    # Valence S shell (shell 1) coefficients — non-tiny, clean FD.
    _check("C", "06_C", 6, 3, "PBE", "coeff", [(1, 0), (1, 1)], Transform.LINEAR)


def test_uks_nitrogen_quartet_exponent():
    _check("N", "07_N", 7, 4, "PBE", "exponent", [(4, 0)], Transform.LOG, shift=-0.3)


def test_uks_rejects_mgga():
    atoms = {"C": parse_crystal_atom_basis_file(SRC / "06_C")}
    par = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec("C", 4, 0, "exponent", transform=Transform.LOG)],
    )

    def mol():
        return vq.Molecule([vq.Atom(6, [0.0, 0.0, 0.0])], multiplicity=3)

    with TempBasisLibrary() as lib:
        with pytest.raises(NotImplementedError):
            energy_gradient_analytic_uks(par, mol, lib, par.pack(), functional="TPSS")
