"""The solvation gradient must screen with the variant that built ``q``.

Regression pin for the hard-coded ``variant="cpcm"`` in the cavity term of
:func:`vibeqc.solvation.gradient.cpcm_gradient`.

Mechanism
---------
The driver solves the apparent-surface-charge equation with the screening
factor of the *requested* model (``driver._solve_cpcm`` ->
``dielectric_factor(eps, variant=sm.variant)``), and stores the scaled
physical charge ``q = f q0`` with ``q0 = -A^-1 V``. The stationary
derivative of ``E_pol = (1/2) q^T V`` in that convention is

    dE_pol = q^T dV + (1/(2f)) q^T (dA) q,

so the ``f`` appearing in the cavity term is not a free choice: it must be
the same ``f`` that built ``q``. Re-deriving it from a fixed variant makes
the analytic gradient differentiate a *different* energy than the one the
run reports, smoothly and without any warning.

The two shipped factors are ``f_cpcm = (e - 1)/e`` (Cossi 2003) and
``f_cosmo = (e - 1)/(e + 1/2)`` (Klamt-Schuurmann 1993 p. 801). They
diverge as ``e`` falls, so the defect is largest in nonpolar solvents:
at ``e = 2.27`` (benzene) the cavity term was scaled by
``f_cosmo/f_cpcm = 0.8195``, i.e. ~18 percent too small.

Why the pre-existing FD test did not catch it: it runs water at the
*default* variant, where the hard-coded ``"cpcm"`` happens to be correct.
Only a ``variant="cosmo"`` run is inconsistent, and nothing exercised one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc.solvation.cpcm import dielectric_factor

# ---------------------------------------------------------------------
# Pure-Python contract pin -- no C++ build required.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [2.02, 2.27, 4.71, 32.61, 78.39])
@pytest.mark.parametrize("variant", ["cpcm", "cosmo"])
def test_screening_factor_follows_the_result_variant(epsilon, variant):
    """``_screening_factor`` reproduces the factor the ASC solve used.

    This is the whole defect in one assertion: the gradient's ``f`` is
    read off the result, never re-derived from a fixed variant.
    """
    from vibeqc.solvation.gradient import _screening_factor

    result = SimpleNamespace(epsilon=epsilon, solvent_variant=variant)
    assert _screening_factor(result) == pytest.approx(
        dielectric_factor(epsilon, variant=variant), rel=0, abs=0
    )


def test_screening_factor_distinguishes_cosmo_from_cpcm_at_low_epsilon():
    """A COSMO result must not be screened with the CPCM factor.

    Guards the specific silent-substitution that shipped: at benzene's
    dielectric the two factors differ by ~18 percent, so an equality here
    would mean the variant is being ignored again.
    """
    from vibeqc.solvation.gradient import _screening_factor

    eps = 2.27
    cosmo = _screening_factor(
        SimpleNamespace(epsilon=eps, solvent_variant="cosmo")
    )
    cpcm = _screening_factor(
        SimpleNamespace(epsilon=eps, solvent_variant="cpcm")
    )
    assert cosmo != cpcm
    # (e-1)/(e+0.5) vs (e-1)/e -- pinned numerically so a future edit to
    # dielectric_factor cannot quietly collapse the two branches.
    assert cosmo == pytest.approx(1.27 / 2.77, rel=1e-12)
    assert cpcm == pytest.approx(1.27 / 2.27, rel=1e-12)
    assert cosmo / cpcm == pytest.approx(0.81949458, rel=1e-6)


# ---------------------------------------------------------------------
# End-to-end: analytic vs central FD on the exact shipped COSMO energy.
# ---------------------------------------------------------------------


def _displaced_water(vq):
    """Water well away from its stationary point, in bohr.

    Non-equilibrium on purpose: at a stationary geometry every gradient
    component is near zero and a percentage error in one contribution is
    invisible against the FD floor.
    """
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.81, 0.0, 0.20]),
        vq.Atom(1, [-0.38, 1.75, 0.0]),
    ]
    return vq.Molecule(atoms, 0, 1)


def _assert_cavity_topology_stable(vq, mol, h, n_points_per_sphere):
    """Every displaced geometry in the stencil keeps the same segments.

    ``build_cavity`` drops segments below ``drop_threshold``, so a stencil
    can straddle a topology change and compare two different energy
    surfaces. A tolerance failure would then be blamed on the gradient.
    Required by the acceptance gates in
    ``handovers/HANDOVER_COSMO_COSMORS.md``.
    """
    from vibeqc.solvation.cavity import build_cavity

    pos = np.array([list(a.xyz) for a in mol.atoms], dtype=np.float64)
    Zs = [int(a.Z) for a in mol.atoms]
    counts = set()
    ref = build_cavity(
        atom_positions_bohr=pos,
        atom_numbers=Zs,
        n_points_per_sphere=n_points_per_sphere,
    )
    counts.add(ref.n_points)
    for ia in range(len(Zs)):
        for ic in range(3):
            for sign in (+1.0, -1.0):
                p = pos.copy()
                p[ia, ic] += sign * h
                counts.add(
                    build_cavity(
                        atom_positions_bohr=p,
                        atom_numbers=Zs,
                        n_points_per_sphere=n_points_per_sphere,
                    ).n_points
                )
    assert len(counts) == 1, (
        f"cavity topology changed across the h={h} stencil "
        f"(segment counts {sorted(counts)}); the analytic and FD "
        f"gradients would sample different energy surfaces"
    )


def test_cosmo_gradient_matches_fd_in_low_dielectric():
    """COSMO analytic gradient reproduces FD on the energy it reports.

    Benzene (e = 2.27) with ``variant="cosmo"``: the configuration where
    the hard-coded CPCM factor scaled the cavity term to 0.8195 of its
    correct value. Pre-fix this test fails by ~1e-3 Ha/bohr, four orders
    of magnitude above the FD floor; post-fix it agrees at the FD
    truncation level.

    Do not loosen the tolerance to make a regression pass -- the whole
    point of this file is that a smooth, plausible, wrong gradient is
    what shipped.
    """
    vq = pytest.importorskip("vibeqc")

    mol = _displaced_water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    n_pts = 50
    h = 2e-3

    sm = vq.SolventModel(
        epsilon=2.27,
        name="benzene",
        variant="cosmo",
        n_points_per_sphere=n_pts,
        max_macro_iter=40,
        tol_e_solv=1e-9,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    # The run really is COSMO -- otherwise this test proves nothing.
    assert sol.solvent_variant == "cosmo"

    _assert_cavity_topology_stable(vq, mol, h, n_pts)

    grad = np.asarray(
        vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"),
        dtype=np.float64,
    )
    grad_fd = np.asarray(
        vq.cpcm_gradient_fd(
            mol, "sto-3g", method="rhf", solvent=sm, step_bohr=h
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(grad, grad_fd, atol=1e-5, rtol=1e-3)


def test_cpcm_variant_gradient_unchanged_by_the_fix():
    """The default CPCM route keeps agreeing with FD.

    Negative control: the fix must not perturb the variant that was
    already self-consistent, so a CPCM run at the same low dielectric
    still matches its own FD reference.
    """
    vq = pytest.importorskip("vibeqc")

    mol = _displaced_water(vq)
    basis = vq.BasisSet(mol, "sto-3g")
    n_pts = 50
    h = 2e-3

    sm = vq.SolventModel(
        epsilon=2.27,
        name="benzene-cpcm",
        variant="cpcm",
        n_points_per_sphere=n_pts,
        max_macro_iter=40,
        tol_e_solv=1e-9,
    )
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    assert sol.solvent_variant == "cpcm"

    _assert_cavity_topology_stable(vq, mol, h, n_pts)

    grad = np.asarray(
        vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"),
        dtype=np.float64,
    )
    grad_fd = np.asarray(
        vq.cpcm_gradient_fd(
            mol, "sto-3g", method="rhf", solvent=sm, step_bohr=h
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(grad, grad_fd, atol=1e-5, rtol=1e-3)
