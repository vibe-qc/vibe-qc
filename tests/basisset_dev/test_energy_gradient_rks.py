"""End-to-end RKS (DFT) analytic basis-parameter gradient vs full-energy FD.

The Kohn-Sham counterpart of ``test_energy_gradient_analytic.py``.
``energy_gradient_analytic_rks`` adds the explicit grid exchange-correlation
term ``Σ_g w_g [v_ρ ∂ρ/∂η + 2 v_σ ∇ρ·∂∇ρ/∂η]`` to the analytic Coulomb (+
hybrid exact-exchange) Pulay assembly, on the same grid run_rks uses. Covers
LDA, GGA (PBE), and a global hybrid (B3LYP) for both parameter kinds, a
multi-element d-shell system, AND a DIFFUSE exponent away from the reference
(the case that caught the reference-vs-current ∂lnN_c bug).

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
from vibeqc.basis_optimization import energy_gradient_analytic_rks  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
)

SRC = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"
TOL = 5e-5


def _load(*syms_srcs):
    return {sym: parse_crystal_atom_basis_file(SRC / src) for sym, src in syms_srcs}


def _check(atoms, mol_factory, specs, functional, field, transform, x_eval=None):
    par = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec(s, i, p, field, transform=transform) for (s, i, p) in specs],
    )
    x0 = par.pack() if x_eval is None else np.asarray(x_eval, dtype=float)
    with TempBasisLibrary() as lib:
        g_an = energy_gradient_analytic_rks(
            par, mol_factory, lib, x0, functional=functional
        )

        def etot(x):
            ax = par.unpack(np.asarray(x, dtype=float))
            nm = lib.write_g94(ax, basis_name="rks-ref")
            m = mol_factory()
            o = vq.RKSOptions(); o.functional = functional
            return float(vq.run_rks(m, vq.BasisSet(m, nm), o).energy)

        rel = 1e-4 if field == "exponent" else 1e-3
        g_fd = np.zeros(len(x0))
        for i in range(len(x0)):
            h = max(1e-7, rel * abs(float(x0[i])))
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            g_fd[i] = (etot(xp) - etot(xm)) / (2.0 * h)

    np.testing.assert_allclose(g_an, g_fd, atol=TOL, rtol=1e-3)


def _be():
    return vq.Molecule([vq.Atom(4, [0.0, 0.0, 0.0])], multiplicity=1)


@pytest.mark.parametrize("functional", ["LDA", "PBE", "B3LYP"])
def test_rks_be_exponent(functional):
    _check(_load(("Be", "04_Be")), _be, [("Be", 0, 0), ("Be", 1, 0)],
           functional, "exponent", Transform.LOG)


@pytest.mark.parametrize("functional", ["LDA", "PBE", "B3LYP"])
def test_rks_be_coefficient(functional):
    _check(_load(("Be", "04_Be")), _be, [("Be", 0, 0), ("Be", 0, 1)],
           functional, "coeff", Transform.LINEAR)


def test_rks_pbe_diffuse_exponent_off_reference():
    """Diffuse valence exponents evaluated AWAY from the reference basis — the
    case where ∂lnN_c/∂α must use the current (unpacked) exponent, not the
    reference. x shifts shells 2,3 to more diffuse values."""
    atoms = _load(("Be", "04_Be"))
    par = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec("Be", 2, 0, "exponent", transform=Transform.LOG),
              FreeSpec("Be", 3, 0, "exponent", transform=Transform.LOG)],
    )
    x0 = par.pack()
    x_shifted = (x0 - 0.4).tolist()   # ~0.67× the exponents (more diffuse)
    _check(atoms, _be, [("Be", 2, 0), ("Be", 3, 0)],
           "PBE", "exponent", Transform.LOG, x_eval=x_shifted)


def _h2o():
    return vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 1.43, 1.11]),
         vq.Atom(1, [0.0, -1.43, 1.11])], multiplicity=1)


def test_rks_pbe_h2o_multielement_dshell():
    wat = _load(("O", "08_O"), ("H", "01_H"))
    n = len(wat["O"].shells)
    _check(wat, _h2o, [("O", n - 1, 0), ("O", 0, 0), ("H", 2, 0)],
           "PBE", "exponent", Transform.LOG)


def test_rks_rejects_mgga():
    atoms = _load(("Be", "04_Be"))
    par = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec("Be", 0, 0, "exponent", transform=Transform.LOG)],
    )
    with TempBasisLibrary() as lib:
        with pytest.raises(NotImplementedError):
            energy_gradient_analytic_rks(par, _be, lib, par.pack(), functional="TPSS")
