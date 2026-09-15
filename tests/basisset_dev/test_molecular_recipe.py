"""The one-call native molecular basis-optimisation recipe.

Exercises ``optimize_molecular_basis`` end-to-end across the reference selectors
(RHF, and open-shell DFT/UKS), confirming it drives the analytic-gradient BDIIS
to a converged, lower-objective basis and returns the optimised
``{symbol: CrystalAtomBasis}`` dict. Needs a built vibe-qc; skipped otherwise.
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
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)
from vibeqc.basis_optimization.recipes.molecular import (  # noqa: E402
    optimize_molecular_basis,
    optimize_molecular_basis_multi,
)
from vibeqc.basis_optimization.recipes.objective import (  # noqa: E402
    make_multi_native_objective_gradient,
)
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402

SRC = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"


def _h2_at(r):
    return lambda: vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, r])], multiplicity=1
    )


def _h_par():
    return BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(SRC / "01_H")},
        free=[
            FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
            FreeSpec("H", 3, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
        ],
    )


def test_optimize_molecular_basis_rhf_h2():
    par = BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(SRC / "01_H")},
        free=[
            FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
            FreeSpec("H", 3, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
        ],
    )

    def mol():
        return vq.Molecule(
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], multiplicity=1
        )

    res = optimize_molecular_basis(par, mol)   # RHF (functional=None, closed)

    assert res.method == "RHF"
    assert res.converged, res.message
    assert res.optimal_objective < res.starting_objective - 1e-6
    # the optimised basis is returned and differs from the start
    assert set(res.optimized_atoms) == {"H"}
    assert res.optimized_atoms["H"].shells[2].exponents[0] != \
        par.atoms["H"].shells[2].exponents[0]
    assert isinstance(res.summary(), str) and "RHF" in res.summary()
    # §8: the optimiser-method citation is surfaced from user-facing output
    assert "daga_optbasis_2020" in res.citations
    assert "cite (method)" in res.summary()


def test_optimize_molecular_basis_uks_open_shell_atom():
    par = BasisParametrisation(
        atoms={"C": parse_crystal_atom_basis_file(SRC / "06_C")},
        free=[
            FreeSpec("C", 4, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
            FreeSpec("C", 7, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
        ],
    )

    def mol():
        return vq.Molecule([vq.Atom(6, [0.0, 0.0, 0.0])], multiplicity=3)

    res = optimize_molecular_basis(par, mol, functional="PBE", open_shell=True)

    assert res.method == "UKS/PBE"
    assert res.converged, res.message
    assert res.optimal_objective < res.starting_objective - 1e-6


def test_optimize_molecular_basis_fd_fallback_for_coeff():
    """analytic=False drives BDIIS with the FD gradient — the path for SP
    coefficient params (and a general escape hatch)."""
    par = BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(SRC / "01_H")},
        free=[FreeSpec("H", 0, 0, "coeff", transform=Transform.LINEAR)],
    )

    def mol():
        return vq.Molecule(
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], multiplicity=1
        )

    res = optimize_molecular_basis(par, mol, analytic=False, max_iter=40)
    assert res.optimal_objective <= res.starting_objective + 1e-9


def test_optimize_molecular_basis_multi_calibration_set():
    """Joint optimisation of one H basis against two H2 bond lengths — the
    calibration-set (multi-system) path."""
    par = _h_par()
    factories = [_h2_at(1.2), _h2_at(1.8)]
    res = optimize_molecular_basis_multi(par, factories, weights=[1.0, 1.0])

    assert "2 systems" in res.method
    assert res.converged, res.message
    assert res.optimal_objective < res.starting_objective - 1e-6


def test_multi_objective_gradient_matches_fd():
    """The joint analytic gradient (Σ_s w_s ∇E_s) matches the FD of the joint
    objective — validates the multi-system factory's summation."""
    par = _h_par()
    factories = [_h2_at(1.3), _h2_at(1.9)]
    weights = [1.0, 0.5]
    with TempBasisLibrary() as lib:
        obj, grad = make_multi_native_objective_gradient(
            par, factories, lib, weights=weights
        )
        x0 = par.pack()
        g_an = grad(x0)
        g_fd = np.zeros(len(x0))
        for i in range(len(x0)):
            h = 1e-4
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            g_fd[i] = (obj(xp) - obj(xm)) / (2.0 * h)
    np.testing.assert_allclose(g_an, g_fd, atol=5e-6, rtol=1e-4)
