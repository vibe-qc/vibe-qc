"""The opt-in Gaussian segment-charge kernel on the Lebedev cavity (#681).

The default point-charge cavity matrix A is indefinite: on a distorted methanol
its smallest eigenvalue runs from -6 to -28. An eigenvalue crossing zero is a
pole of A^-1, so the solvated energy and its analytic gradient spike at nearby
geometries, and at the default switching width the point kernel also overshoots
the sharp union-of-spheres answer by 17-27 %. ``charge_representation="gaussian"``
keeps the cavity and swaps the kernel for York & Karplus's Gaussian charges, whose
A is a Gram matrix (positive definite for any geometry).

Nothing here changes a default: a plain ``SolventModel`` is still point-charge.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.solvation.cavity import build_cavity
from vibeqc.solvation.cpcm import GAUSSIAN_CHARGE, POINT_CHARGE, build_A_matrix
from vibeqc.solvation.driver import solvation_citation_key

ANG = 1.8897261254578281

_WATER = [(8, (0.0, 0.0, 0.1193)), (1, (0.0, 0.7632, -0.477)), (1, (0.0, -0.7632, -0.477))]
_METHANOL = [
    (6, (-0.0471, 0.6644, 0.0)),
    (8, (-0.0471, -0.7586, 0.0)),
    (1, (-1.093, 0.9698, 0.0)),
    (1, (0.8785, -1.0485, 0.0)),
    (1, (0.4371, 1.0804, 0.8918)),
    (1, (0.4371, 1.0804, -0.8918)),
]


def _bohr(atoms):
    return np.array([p for _, p in atoms], dtype=float) * ANG


def _molecule(atoms, positions_bohr=None):
    pos = _bohr(atoms) if positions_bohr is None else positions_bohr
    return vq.Molecule([vq.Atom(z, tuple(p)) for (z, _), p in zip(atoms, pos)], 0, 1)


def _model(kernel, **kw):
    base = dict(
        epsilon=78.39,
        name="water",
        charge_representation=kernel,
        max_macro_iter=60,
        tol_e_solv=1e-11,
    )
    base.update(kw)
    return vq.SolventModel(**base)


def _spectrum_min(cavity, representation):
    A = np.asarray(
        build_A_matrix(
            cavity.points,
            cavity.weights,
            representation=representation,
            switching=cavity.switching,
            n_points_per_sphere=302,
        )
    )
    return float(np.linalg.eigvalsh(0.5 * (A + A.T)).min())


# --- the model ---------------------------------------------------------------


def test_the_default_is_still_the_point_charge_kernel():
    sm = vq.SolventModel(epsilon=78.39)
    assert sm.charge_representation is None
    assert solvation_citation_key(sm) == "cpcm"
    z = [a for a, _ in _WATER]
    cav = build_cavity(atom_positions_bohr=_bohr(_WATER), atom_numbers=z)
    assert cav.kernel == "point"
    assert cav.charge_representation == POINT_CHARGE
    assert cav.exponent_follows_area is False


@pytest.mark.parametrize("spec", [None, "point", "gaussian", "Gaussian", " POINT "])
def test_the_accepted_spellings_normalise(spec):
    sm = vq.SolventModel(epsilon=78.39, charge_representation=spec)
    assert sm.charge_representation in (None, "point", "gaussian")
    if spec is not None:
        assert sm.charge_representation == spec.strip().lower()


def test_an_unknown_kernel_is_refused_by_name():
    with pytest.raises(ValueError, match="'point'.*'gaussian'|'gaussian'"):
        vq.SolventModel(epsilon=78.39, charge_representation="slater")
    with pytest.raises(ValueError, match="charge_representation"):
        build_cavity(
            atom_positions_bohr=_bohr(_WATER),
            atom_numbers=[8, 1, 1],
            charge_representation="slater",
        )


def test_the_fine_cavity_has_no_point_charge_kernel():
    with pytest.raises(ValueError, match="fine"):
        vq.SolventModel(epsilon=78.39, cavity="fine", charge_representation="point")
    vq.SolventModel(epsilon=78.39, cavity="fine", charge_representation="gaussian")


def test_the_citation_route_names_the_kernel_that_ran():
    from vibeqc.output.citations.registry import load_default_database  # noqa: PLC0415

    db = load_default_database()
    for variant in ("cpcm", "cosmo"):
        sm = vq.SolventModel(epsilon=78.39, variant=variant, charge_representation="gaussian")
        key = solvation_citation_key(sm)
        assert key == f"{variant}_gaussian"
        entries = db.routes()["solvation"][key]
        assert "york_karplus_smooth_cosmo_1999" in entries
        assert "scalmani_frisch_csc_2010" in entries  # the switching is still theirs
        assert "klamt_fine_cavity_2018" not in entries  # and the cavity is still Lebedev


# --- the matrix --------------------------------------------------------------


def test_the_gaussian_matrix_is_positive_definite_and_the_point_matrix_is_not():
    rng = np.random.default_rng(7)
    base = _bohr(_METHANOL)
    z = [a for a, _ in _METHANOL]
    for _ in range(4):
        pos = base + rng.normal(scale=0.05, size=base.shape)
        cav = build_cavity(atom_positions_bohr=pos, atom_numbers=z)
        assert _spectrum_min(cav, GAUSSIAN_CHARGE) > 3.0  # measured 3.389-3.391
        assert _spectrum_min(cav, POINT_CHARGE) < -1.0  # measured -6 to -28


# --- the number --------------------------------------------------------------


def _model_water_esolv(sigma, representation, n_points=302):
    """E_solv of point charges on the nuclei (-0.8/+0.4/+0.4): no SCF involved."""
    pos = _bohr(_WATER)
    charges = np.array([-0.8, 0.4, 0.4])
    eps = 78.39
    cav = build_cavity(
        atom_positions_bohr=pos,
        atom_numbers=[8, 1, 1],
        n_points_per_sphere=n_points,
        switching_sigma_bohr=sigma,
        drop_threshold=1e-8,
    )
    A = np.asarray(
        build_A_matrix(
            cav.points,
            cav.weights,
            representation=representation,
            switching=cav.switching,
            n_points_per_sphere=n_points,
        )
    )
    V = sum(q / np.linalg.norm(cav.points - c, axis=1) for q, c in zip(charges, pos))
    return 0.5 * float(np.linalg.solve(A, -(eps - 1) / eps * V) @ V)


def test_at_the_default_width_the_gaussian_kernel_is_closer_to_the_sharp_cavity():
    sharp = _model_water_esolv(0.02, GAUSSIAN_CHARGE)
    # The two kernels discretise one model and agree once the switching is sharp.
    assert _model_water_esolv(0.02, POINT_CHARGE) == pytest.approx(sharp, rel=0.01)
    gauss = _model_water_esolv(0.5, GAUSSIAN_CHARGE)
    point = _model_water_esolv(0.5, POINT_CHARGE)
    assert abs(gauss / sharp - 1.0) < 0.03  # measured 2.0 %
    assert abs(point / sharp - 1.0) > 0.10  # measured 17.8 %


# --- the gradient ------------------------------------------------------------


def _fd_gradient_error(atoms, kernel, positions_bohr=None, step=1e-3):
    mol = _molecule(atoms, positions_bohr)
    basis = vq.BasisSet(mol, "sto-3g")
    sm = _model(kernel)
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)
    assert sol.converged
    g = np.asarray(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"))

    def energy(x):
        m = _molecule(atoms, x)
        r = vq.run_cpcm_scf(m, vq.BasisSet(m, "sto-3g"), method="rhf", solvent=sm)
        assert r.converged
        return float(r.energy)

    x0 = _bohr(atoms) if positions_bohr is None else np.asarray(positions_bohr)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(2):
        v = rng.normal(size=x0.shape)
        v /= np.linalg.norm(v)
        fd = (energy(x0 + step * v) - energy(x0 - step * v)) / (2 * step)
        worst = max(worst, abs(float(np.sum(v * g)) - fd))
    return worst


def test_the_analytic_gradient_matches_finite_differences_at_equilibrium():
    assert _fd_gradient_error(_WATER, "gaussian") < 2e-6  # measured 1.4e-7
    assert _fd_gradient_error(_METHANOL, "gaussian") < 2e-6  # measured 2.9e-7


def _distorted_methanol(seed):
    rng = np.random.default_rng(seed)
    return _bohr(_METHANOL) + rng.normal(scale=0.05, size=(6, 3))


# Seeds 39 and 59 are 0.05 bohr random distortions of methanol where the point
# kernel's gradient is wrong by 7e-4 and 4e-4 Ha/bohr (an eigenvalue of A is near
# zero); seed 3 is an ordinary geometry. The seeds came from a scan, #681.
@pytest.mark.parametrize("seed", [3, 39, 59])
def test_the_analytic_gradient_holds_on_distorted_methanol(seed):
    err = _fd_gradient_error(
        _METHANOL, "gaussian", positions_bohr=_distorted_methanol(seed), step=2e-4
    )
    assert err < 1e-6  # measured 4e-10 to 1.6e-9


def test_the_point_kernel_gradient_is_wrong_where_the_gaussian_one_is_not():
    """Documents #681 at a geometry where the default kernel is in resonance.

    A pin of a known defect, not of desired behaviour: delete it if the point
    kernel's gradient is ever repaired, or the default becomes the Gaussian one.
    """
    pos = _distorted_methanol(39)
    assert _fd_gradient_error(_METHANOL, "point", positions_bohr=pos, step=2e-4) > 1e-4


def test_the_gradient_is_translation_invariant():
    mol = _molecule(_METHANOL)
    basis = vq.BasisSet(mol, "sto-3g")
    sol = vq.run_cpcm_scf(mol, basis, method="rhf", solvent=_model("gaussian"))
    g = np.asarray(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"))
    assert np.abs(g.sum(axis=0)).max() < 1e-6
