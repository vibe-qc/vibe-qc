"""Periodic Gamma GFN2-xTB with the lattice-summed gamma and the Bannwarth AES.

Issue #296 closure: the periodic energy is invariant under a lattice
translation of any input atom and continuous when an atom crosses a cell face,
with no coordinate canonicalisation.  Issue #338: the analytic gradient and
all nine stress components differentiate the same energy (faithful AES with
image-resolved CAMM, Elstner or Klopman-Ohno lattice-summed gamma).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb

try:
    from vibeqc.semiempirical.methods.gfn2_params import (
        gfn2_parameter_cache_available,
        load_gfn2_params,
    )

    _HAVE_GFN2 = gfn2_parameter_cache_available()
except Exception:  # pragma: no cover - import guard
    _HAVE_GFN2 = False

requires_gfn2 = pytest.mark.skipif(
    not _HAVE_GFN2, reason="GFN2-xTB parameter cache unavailable"
)

BOHR = 1.8897261254535


@pytest.fixture(scope="module")
def params():
    return load_gfn2_params()


def _mgo(*, oxygen_image=(0, 0, 0), magnesium_image=(0, 0, 0)):
    a = 4.211 * BOHR
    lattice = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
    mg = np.zeros(3) + lattice @ np.asarray(magnesium_image, dtype=float)
    o = np.array([a / 2, 0.0, 0.0]) + lattice @ np.asarray(oxygen_image, dtype=float)
    return PeriodicSystem(3, lattice, [Atom(12, mg.tolist()), Atom(8, o.tolist())], 0, 1)


def _hbn(displacement=0.0, nitrogen_image=(0, 0)):
    """Polar h-BN monolayer: nonzero atomic dipoles/quadrupoles."""
    a = 4.732
    lattice = np.array([[a, 0.0, 0.0], [a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 30.0]])
    n = np.array([a / 2 + displacement, a * np.sqrt(3) / 6, 0.0])
    n = n + nitrogen_image[0] * lattice[:, 0] + nitrogen_image[1] * lattice[:, 1]
    return PeriodicSystem(2, lattice, [Atom(5, [0.0, 0.0, 0.0]), Atom(7, n.tolist())], 0, 1)


def _polar_hf_cell(shift=np.zeros(3)):
    return PeriodicSystem(
        3,
        np.eye(3) * 14.0,
        [Atom(1, (np.array([0.5, 0.4, 0.3]) + shift).tolist()),
         Atom(9, (np.array([3.6, 1.7, 1.0]) + shift).tolist())],
        0,
        1,
    )


def _tight(gamma_form=None, temperature=0.0):
    opts = _xtb.XTBSccOptions()
    opts.max_iter = 2000
    opts.conv_tol_charge = 1.0e-9
    opts.auto_stabilize = False
    opts.electronic_temperature = temperature
    if gamma_form is not None:
        opts.gamma_form = gamma_form
    return opts


def _components(result):
    return np.array([
        result.energy, result.free_energy, result.e_electronic, result.e_repulsive,
        result.e_scc, result.e_band0, result.e_aes, result.e_3rd,
    ])


# ---------------------------------------------------------------------------
# libint layout pin behind the multipole derivative contraction
# ---------------------------------------------------------------------------


def test_libint_emultipole2_first_derivatives_are_derivative_major():
    """Buffer d*10 + o holds d(component o)/d(centre d), pinned by FD."""
    h = 1.0e-4
    probe = np.asarray(_se._libint_emultipole2_derivative_layout_probe(h))
    derivative_buffers = probe[:6]
    plain = probe[6:]
    fd = np.zeros((6, 10))
    for d in range(6):
        fd[d] = (plain[2 * d] - plain[2 * d + 1]) / (2.0 * h)
    assert np.abs(fd).max() > 1e-3  # the probe geometry is not degenerate
    np.testing.assert_allclose(derivative_buffers, fd, rtol=0.0, atol=1e-7)
    # Translational invariance holds for the overlap component only; the
    # multipole components pick up the operator's own gradient.
    np.testing.assert_allclose(fd[:3, 0] + fd[3:, 0], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Translation covariance and continuity (issue #296)
# ---------------------------------------------------------------------------


@requires_gfn2
def test_lattice_summed_shell_gamma_is_translation_covariant(params):
    reference = np.asarray(_se._periodic_gfn2_shell_gamma(_mgo(), params, 12.0, _se.ShellGammaForm.Elstner))
    for image in ((1, 0, 0), (0, -1, 0), (2, 1, -1)):
        moved = np.asarray(_se._periodic_gfn2_shell_gamma(
            _mgo(oxygen_image=image), params, 12.0, _se.ShellGammaForm.Elstner))
        np.testing.assert_allclose(moved, reference, rtol=0.0, atol=1e-11)
    assert np.allclose(reference, reference.T, atol=1e-13)
    ko = np.asarray(_se._periodic_gfn2_shell_gamma(
        _mgo(), params, 12.0, _se.ShellGammaForm.KlopmanOhno))
    assert np.abs(ko - reference).max() > 1e-4


@requires_gfn2
@pytest.mark.parametrize("gamma_form", [_se.ShellGammaForm.Elstner, _se.ShellGammaForm.KlopmanOhno])
def test_mgo_energy_is_invariant_under_lattice_relabelling(params, gamma_form):
    """Issue #296: four energies for one crystal spanning 160 mHa."""
    # The Klopman-Ohno map at T = 0.001 converges into the over-polarised
    # shell-transfer basin the physical-basin gate rejects; at 0.005 it
    # reaches the physical state in 37 iterations (the form is not the
    # periodic default, see #444).
    temperature = 0.001 if gamma_form == _se.ShellGammaForm.Elstner else 0.005
    opts = _tight(gamma_form, temperature=temperature)
    results = [
        _xtb.run_gfn2_xtb_gamma(system, params, opts, 12.0)
        for system in (
            _mgo(),
            _mgo(oxygen_image=(1, 0, 0)),
            _mgo(magnesium_image=(0, -1, 0)),
            _mgo(oxygen_image=(-2, 1, 1), magnesium_image=(1, 1, 0)),
        )
    ]
    assert all(r.converged for r in results)
    assert results[0].gamma_form == gamma_form
    for r in results[1:]:
        np.testing.assert_allclose(_components(r), _components(results[0]), rtol=0.0, atol=1e-9)
        # The padded cell list legitimately differs between representatives;
        # only the interaction set, and hence the energy, is invariant.


@requires_gfn2
def test_polar_hbn_energy_and_moments_are_invariant_under_lattice_relabelling(params):
    """The AES channel (nonzero dipoles) must be covariant too, no canonicalisation."""
    opts = _tight()
    ref = _xtb.run_gfn2_xtb_gamma(_hbn(0.1), params, opts, 12.0)
    moved = _xtb.run_gfn2_xtb_gamma(_hbn(0.1, nitrogen_image=(1, -1)), params, opts, 12.0)
    assert ref.converged and moved.converged
    assert abs(ref.e_aes) > 1e-6
    assert np.abs(np.asarray(ref.atom_dipoles)).max() > 1e-4
    np.testing.assert_allclose(_components(moved), _components(ref), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(moved.atom_dipoles), np.asarray(ref.atom_dipoles), atol=1e-8)
    np.testing.assert_allclose(np.asarray(moved.atom_quadrupoles), np.asarray(ref.atom_quadrupoles), atol=1e-8)


@requires_gfn2
def test_energy_is_continuous_when_an_atom_crosses_a_cell_face(params):
    """The sixth #296 fix canonicalised the representative and made E jump at
    every face; the lattice-summed kernel has nothing to jump.  The energy on
    both sides of the x = 0 face of the polar HF cell must agree with the
    analytic gradient at the face, i.e. be one smooth function through it."""
    opts = _tight()
    eps = 1.0e-4

    def cell(x):
        return PeriodicSystem(
            3, np.eye(3) * 14.0,
            [Atom(1, [x, 0.4, 0.3]), Atom(9, [3.1, 1.7, 1.0])], 0, 1,
        )

    energies = {}
    for x in (-eps, 0.0, eps):
        r = _xtb.run_gfn2_xtb_gamma(cell(x), params, opts)
        assert r.converged
        energies[x] = r.energy
        if x == 0.0:
            gradient = np.asarray(_se.compute_periodic_gfn2_gradient(cell(x), r, params))
    slope = (energies[eps] - energies[-eps]) / (2.0 * eps)
    assert slope == pytest.approx(gradient[0, 0], abs=2e-6)
    assert energies[0.0] == pytest.approx(0.5 * (energies[eps] + energies[-eps]), abs=1e-9)
    # And the representative one cell over is the same crystal.
    shifted = _xtb.run_gfn2_xtb_gamma(
        PeriodicSystem(3, np.eye(3) * 14.0,
                       [Atom(1, [14.0, 0.4, 0.3]), Atom(9, [3.1, 1.7, 1.0])], 0, 1),
        params, opts)
    assert shifted.energy == pytest.approx(energies[0.0], abs=1e-9)


# ---------------------------------------------------------------------------
# Derivatives (issue #338)
# ---------------------------------------------------------------------------


def _fd_gradient(system, energy_fn, h=1e-4):
    lattice = np.asarray(system.lattice)
    atoms = list(system.unit_cell)
    grad = np.zeros((len(atoms), 3))
    for a, atom in enumerate(atoms):
        for k in range(3):
            plus = [Atom(x.Z, list(x.xyz)) for x in atoms]
            minus = [Atom(x.Z, list(x.xyz)) for x in atoms]
            xp = list(atom.xyz); xp[k] += h
            xm = list(atom.xyz); xm[k] -= h
            plus[a] = Atom(atom.Z, xp)
            minus[a] = Atom(atom.Z, xm)
            grad[a, k] = (
                energy_fn(PeriodicSystem(system.dim, lattice, plus, system.charge, system.multiplicity))
                - energy_fn(PeriodicSystem(system.dim, lattice, minus, system.charge, system.multiplicity))
            ) / (2 * h)
    return grad


def _fd_stress(system, energy_fn, h=1e-4):
    lattice0 = np.asarray(system.lattice, dtype=float)
    volume = abs(np.linalg.det(lattice0))
    atoms = list(system.unit_cell)
    stress = np.zeros((3, 3))
    for i in range(system.dim):
        for j in range(system.dim):
            eps = np.zeros((3, 3)); eps[i, j] = h
            out = []
            for sign in (1.0, -1.0):
                t = np.eye(3) + sign * eps
                new_atoms = [Atom(x.Z, list(t @ np.asarray(x.xyz))) for x in atoms]
                out.append(energy_fn(PeriodicSystem(
                    system.dim, t @ lattice0, new_atoms, system.charge, system.multiplicity)))
            stress[i, j] = (out[0] - out[1]) / (2 * h * volume)
    return stress


@requires_gfn2
@pytest.mark.parametrize("gamma_form", [_se.ShellGammaForm.Elstner, _se.ShellGammaForm.KlopmanOhno])
def test_polar_gradient_and_stress_match_finite_differences(params, gamma_form):
    opts = _tight(gamma_form)
    system = _polar_hf_cell()

    def run(candidate):
        r = _xtb.run_gfn2_xtb_gamma(candidate, params, opts)
        assert r.converged
        e = np.asarray(r.mo_energies)
        assert e[r.n_occ] - e[r.n_occ - 1] > 1e-3
        return r

    result = run(system)
    assert abs(result.e_aes) > 1e-6
    grad = np.asarray(_se.compute_periodic_gfn2_gradient(system, result, params))
    stress = np.asarray(_se.compute_periodic_gfn2_stress(system, result, params))
    np.testing.assert_allclose(grad, _fd_gradient(system, lambda s: run(s).energy), rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(stress, _fd_stress(system, lambda s: run(s).energy), rtol=0.0, atol=2e-7)


@requires_gfn2
def test_hbn_gradient_and_in_plane_stress_match_finite_differences(params):
    """2-D slab: the Parry Ewald channel plus the AES image sums."""
    opts = _tight(temperature=0.002)
    system = _hbn(0.15)

    def run(candidate):
        r = _xtb.run_gfn2_xtb_gamma(candidate, params, opts, 12.0)
        assert r.converged
        return r

    result = run(system)
    grad = np.asarray(_se.compute_periodic_gfn2_gradient(system, result, params))
    stress = np.asarray(_se.compute_periodic_gfn2_stress(system, result, params))
    np.testing.assert_allclose(grad, _fd_gradient(system, lambda s: run(s).free_energy), rtol=0.0, atol=2e-6)
    fd = _fd_stress(system, lambda s: run(s).free_energy)
    np.testing.assert_allclose(stress[:2, :2], fd[:2, :2], rtol=0.0, atol=2e-7)


@requires_gfn2
def test_elstner_gamma_converges_in_the_cutoff_and_klopman_ohno_does_not(params):
    """The periodic instance of #444 on the isotropic kernel itself.

    The driver sums the Elstner remainder over its own decay range, so the
    assembled gamma is already converged at the default overlap cutoff; the
    Klopman-Ohno remainder keeps drifting with the cutoff it is tied to.
    (The total energy is not used here because the Gamma-point overlap sum
    of Mg's diffuse 3p function is itself cutoff dependent, a property of
    the H0/S channel that this kernel does not touch.)
    """
    el = [np.asarray(_se._periodic_gfn2_shell_gamma(_mgo(), params, c, _se.ShellGammaForm.Elstner))
          for c in (12.0, 15.0, 24.0)]
    assert np.abs(el[1] - el[0]).max() < 1e-8
    assert np.abs(el[2] - el[0]).max() < 1e-8
    ko = [np.asarray(_se._periodic_gfn2_shell_gamma(_mgo(), params, c, _se.ShellGammaForm.KlopmanOhno))
          for c in (12.0, 24.0)]
    assert np.abs(ko[1] - ko[0]).max() > 1e-4
    # The decay range behind the Elstner convergence.
    spec = _se.ShellGammaSpec()
    spec.form = _se.ShellGammaForm.Elstner
    assert 20.0 < _se.shell_gamma_remainder_range(spec, 0.3448, 0.3448, 1e-10) < 30.0
    spec.form = _se.ShellGammaForm.KlopmanOhno
    assert np.isinf(_se.shell_gamma_remainder_range(spec, 0.3448, 0.3448, 1e-10))
