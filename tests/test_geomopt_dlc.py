"""Delocalised internal coordinate (DLC) back-transform correctness.

Before these fixes the DLC back-transform reused the *reference* Wilson
B-matrix pseudoinverse at every Newton iteration and never wrapped
torsion residuals into (-pi, pi]. For a molecule whose heavy atoms start
planar (glycine), that drove the back-transformed geometry apart until
the SCF diverged to a garbage energy (~-228 Ha vs the -279 Ha minimum).

The fixes:
* rebuild B^+ at the current geometry each back-transform iteration,
* wrap torsion residuals onto the correct 2pi branch,
* seed each step's Newton iteration from the previous step's geometry,
* project the gradient/Hessian with B^+ at the current geometry.

DLC construction still does not guarantee a complete (full-rank)
primitive set for every topology; when it is rank-deficient the
representation warns. These tests pin the transform correctness on a
full-rank system (H2O) and the fail-soft behaviour on a rank-deficient
one (glycine), not full-convergence parity for large flexible molecules.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc.geomopt import ConvergencePolicy, MolecularSCFProvider, run_geomopt
from vibeqc.geomopt.internal_coords import (
    DelocalizedInternalCoordinates,
    _cartesian_to_dlc,
    _dlc_to_cartesian,
    _wrap_primitive_residual,
)

_A2B = 1.0 / 0.529177210903

# Glycine (neutral, planar heavy-atom frame) -- the rp167 SI geometry.
_GLYCINE = [
    (7, (-2.105915, 0.392689, 0.0)),
    (6, (-0.837110, -0.332747, 0.0)),
    (6, (0.382253, 0.576152, 0.0)),
    (8, (1.501936, 0.062047, 0.0)),
    (8, (0.164156, 1.910830, 0.0)),
    (1, (-2.897845, -0.223798, 0.0)),
    (1, (-2.202618, 1.005832, 0.799602)),
    (1, (-0.782287, -0.965302, 0.901712)),
    (1, (-0.782287, -0.965302, -0.901712)),
    (1, (0.913777, 2.493046, 0.0)),
]


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.9]),
            Atom(1, [1.85, 0.0, -0.4]),
        ]
    )


def _glycine() -> Molecule:
    return Molecule(
        [Atom(z, [c * _A2B for c in xyz]) for z, xyz in _GLYCINE], 0, 1
    )


def test_h2o_dlc_is_full_rank_and_round_trips() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.9], [1.85, 0.0, -0.4]])
    dlc = DelocalizedInternalCoordinates(3, [8, 1, 1], pos)
    assert dlc._n_dlc == 3 * 3 - 6  # full rank for a 3-atom nonlinear system

    rng = np.random.RandomState(1)
    perturbed = pos + 0.05 * rng.standard_normal(pos.shape)
    s = _cartesian_to_dlc(perturbed, pos, dlc._primitives, dlc._U, dlc._B_pinv)
    back = _dlc_to_cartesian(
        s, pos, dlc._primitives, dlc._U, dlc._B_pinv, seed_positions=pos
    )

    q_target = np.array([p.value(perturbed) for p in dlc._primitives])
    q_back = np.array([p.value(back) for p in dlc._primitives])
    resid = _wrap_primitive_residual(q_target - q_back, dlc._primitives)
    assert np.max(np.abs(resid)) < 1e-8


def test_torsion_residual_wraps_across_branch_cut() -> None:
    from vibeqc.geomopt.internal_coords import _Bond, _Dihedral

    prims = [_Bond(0, 1), _Dihedral(0, 1, 2, 3)]
    # A torsion at +179 deg vs -179 deg: true difference is 2 deg, not 358.
    raw = np.array([0.1, np.deg2rad(358.0)])
    wrapped = _wrap_primitive_residual(raw, prims)
    assert wrapped[0] == pytest.approx(0.1)  # bond untouched
    assert wrapped[1] == pytest.approx(np.deg2rad(-2.0), abs=1e-9)


def test_h2o_dlc_optimization_matches_cartesian() -> None:
    results = {}
    for coords in ("cartesian", "dlc"):
        provider = MolecularSCFProvider("sto-3g", method="rhf")
        r = run_geomopt(
            _h2o(),
            provider,
            geom_opt="bfgs",
            geom_coords=coords,
            geom_conv=ConvergencePolicy(gmax=4.5e-4),
            geom_max_iter=50,
        )
        assert r.converged, f"{coords} did not converge"
        results[coords] = float(r.energy)
    assert results["dlc"] == pytest.approx(results["cartesian"], abs=1e-4)


def test_glycine_dlc_warns_incomplete_and_does_not_diverge() -> None:
    """A rank-deficient DLC set warns, and the optimizer degrades
    gracefully (bounded, descending) instead of exploding into an
    SCF-divergent geometry as it did before the back-transform fix."""
    with pytest.warns(RuntimeWarning, match="internal DOF"):
        DelocalizedInternalCoordinates(
            10,
            [z for z, _ in _GLYCINE],
            np.array([[c * _A2B for c in xyz] for _, xyz in _GLYCINE]),
        )

    provider = MolecularSCFProvider("sto-3g", method="rhf")
    e0, _ = provider(_glycine())

    # A cold reference energy for the starting geometry.
    ref = MolecularSCFProvider("sto-3g", method="rhf")
    e_start, _ = ref(_glycine())

    with pytest.warns(RuntimeWarning):
        r = run_geomopt(
            _glycine(),
            provider,
            geom_opt="bfgs",
            geom_coords="dlc",
            geom_conv=ConvergencePolicy(gmax=4.5e-4),
            geom_max_iter=40,
        )
    # Not asserting convergence (the primitive set is incomplete), but the
    # optimizer must stay in a physical basin: the energy is finite, below
    # the start, and nowhere near the pre-fix -228 Ha blow-up.
    assert np.isfinite(r.energy)
    assert r.energy <= e_start + 1e-6
    assert r.energy < -278.0
