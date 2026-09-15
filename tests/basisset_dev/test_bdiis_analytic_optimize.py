"""End-to-end BDIIS basis optimisation driven by the analytic gradient.

Ties the whole Phase-1 stack together: make_rhf_native_objective_gradient
builds a consistent in-process RHF (objective, analytic-gradient) pair, and
optimize_bdiis drives it to a stationary basis with no per-parameter re-SCF
finite differences. Asserts the analytic-gradient run (a) converges, (b)
lowers the energy, (c) reaches a genuinely small projected gradient, and
(d) lands on the same minimum the finite-difference-gradient run finds.

Needs a built vibe-qc with the exponent-derivative bindings; skipped
otherwise.
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
from vibeqc.basis_optimization import optimize_bdiis  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)
from vibeqc.basis_optimization.recipes.objective import (  # noqa: E402
    make_rhf_native_objective_gradient,
    make_rks_native_objective_gradient,
    make_uhf_native_objective_gradient,
    make_uks_native_objective_gradient,
)

SRC = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"


def _h2_parametrisation():
    atoms = {"H": parse_crystal_atom_basis_file(SRC / "01_H")}
    return BasisParametrisation(
        atoms=atoms,
        free=[
            FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.05, None)),
            FreeSpec("H", 3, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.05, None)),
        ],
    )


def _h2():
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], multiplicity=1
    )


def test_analytic_bdiis_converges_and_matches_fd():
    with TempBasisLibrary() as lib:
        par = _h2_parametrisation()
        x0 = par.pack()
        obj, grad = make_rhf_native_objective_gradient(par, _h2, lib)
        f0 = obj(x0)

        res_an = optimize_bdiis(obj, x0, grad=grad, bounds=par.optim_bounds(),
                                max_iter=80, tol_grad=1e-5)
        res_fd = optimize_bdiis(obj, x0, bounds=par.optim_bounds(),
                                max_iter=80, tol_grad=1e-5)

        g_star = float(np.max(np.abs(grad(res_an.x))))

    # (a) converged
    assert res_an.success, res_an.message
    # (b) energy lowered relative to the pob-TZVP starting exponents
    assert res_an.fun < f0 - 1e-6
    # (c) genuinely stationary (box-projected gradient small)
    assert g_star < 1e-4
    # (d) same minimum as the finite-difference-gradient driver
    assert abs(res_an.fun - res_fd.fun) < 1e-7


def test_analytic_gradient_coefficient_param_matches_fd():
    """Contraction-coefficient free parameters are analytic (augmented-basis
    direct term) and match the full-energy finite difference."""
    atoms = {"H": parse_crystal_atom_basis_file(SRC / "01_H")}
    par = BasisParametrisation(
        atoms=atoms,
        free=[
            FreeSpec("H", 0, 0, "coeff", transform=Transform.LINEAR),
            FreeSpec("H", 0, 1, "coeff", transform=Transform.LINEAR),
        ],
    )
    with TempBasisLibrary() as lib:
        obj, grad = make_rhf_native_objective_gradient(par, _h2, lib)
        x0 = par.pack()
        g_an = grad(x0)
        # Relative step — coefficients span orders of magnitude (tiny core
        # coeffs need a small absolute step to stay in the linear regime).
        g_fd = np.zeros(len(x0))
        for i in range(len(x0)):
            h = max(1e-7, 1e-3 * abs(float(x0[i])))
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            g_fd[i] = (obj(xp) - obj(xm)) / (2.0 * h)
    np.testing.assert_allclose(g_an, g_fd, atol=5e-5, rtol=1e-3)


def test_uhf_analytic_bdiis_open_shell_atom():
    """Open-shell (UHF) BDIIS loop on the C triplet atom — the path the pob
    recipe needs for open-shell elements. The analytic UHF gradient drives the
    optimiser to a stationary, lower-energy basis."""
    atoms = {"C": parse_crystal_atom_basis_file(SRC / "06_C")}
    par = BasisParametrisation(
        atoms=atoms,
        free=[
            FreeSpec("C", 3, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.05, None)),
            FreeSpec("C", 7, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.08, None)),
        ],
    )

    def mol():
        return vq.Molecule([vq.Atom(6, [0.0, 0.0, 0.0])], multiplicity=3)

    with TempBasisLibrary() as lib:
        obj, grad = make_uhf_native_objective_gradient(par, mol, lib)
        x0 = par.pack()
        f0 = obj(x0)
        res = optimize_bdiis(obj, x0, grad=grad, bounds=par.optim_bounds(),
                             max_iter=80, tol_grad=1e-5)
        g_star = float(np.max(np.abs(grad(res.x))))

    assert res.success, res.message
    assert res.fun < f0 - 1e-6
    assert g_star < 1e-4


def test_rks_pbe_analytic_bdiis():
    """Closed-shell RKS (PBE) BDIIS loop on the Be atom — the DFT-level path,
    driven by the analytic grid-XC gradient. Optimises diffuse valence S
    exponents (the case that exercises the off-reference ∂lnN_c path)."""
    atoms = {"Be": parse_crystal_atom_basis_file(SRC / "04_Be")}
    par = BasisParametrisation(
        atoms=atoms,
        free=[
            FreeSpec("Be", 2, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.02, None)),
            FreeSpec("Be", 3, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.02, None)),
        ],
    )

    def mol():
        return vq.Molecule([vq.Atom(4, [0.0, 0.0, 0.0])], multiplicity=1)

    with TempBasisLibrary() as lib:
        obj, grad = make_rks_native_objective_gradient(par, mol, lib, functional="PBE")
        x0 = par.pack()
        f0 = obj(x0)
        # tol_grad looser than the HF paths: the RKS gradient is grid-based
        # (DFT quadrature + the ∂ln c̃/∂α normalisation FD), accurate to ~1e-5.
        res = optimize_bdiis(obj, x0, grad=grad, bounds=par.optim_bounds(),
                             max_iter=80, tol_grad=1e-4)
        g_star = float(np.max(np.abs(grad(res.x))))

    assert res.success, res.message
    assert res.fun < f0 - 1e-6
    assert g_star < 1e-3


def test_uks_pbe_analytic_bdiis_open_shell_atom():
    """Open-shell DFT (UKS/PBE) BDIIS loop on the C ³P atom — DFT-level
    optimisation of an open-shell element, the combination the pob recipe needs.
    Optimises valence p/d exponents via the analytic polarised grid-XC gradient."""
    atoms = {"C": parse_crystal_atom_basis_file(SRC / "06_C")}
    par = BasisParametrisation(
        atoms=atoms,
        free=[
            FreeSpec("C", 4, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.05, None)),
            FreeSpec("C", 7, 0, "exponent", transform=Transform.LOG,
                     bounds=(0.05, None)),
        ],
    )

    def mol():
        return vq.Molecule([vq.Atom(6, [0.0, 0.0, 0.0])], multiplicity=3)

    with TempBasisLibrary() as lib:
        obj, grad = make_uks_native_objective_gradient(par, mol, lib, functional="PBE")
        x0 = par.pack()
        f0 = obj(x0)
        res = optimize_bdiis(obj, x0, grad=grad, bounds=par.optim_bounds(),
                             max_iter=80, tol_grad=1e-4)
        g_star = float(np.max(np.abs(grad(res.x))))

    assert res.success, res.message
    assert res.fun < f0 - 1e-6
    assert g_star < 1e-3
