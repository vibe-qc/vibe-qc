"""``run_neb`` image evaluation in joblib process workers (#564).

The process backend pickles every image task, which carries the endpoint
``Molecule`` as its template, and every result, which carries the image's
SCF result as the next iteration's warm-start cache. ``Atom``, ``Molecule``
and ``PeriodicSystem`` now pickle; a molecular SCF cache travels as
``_ProcessImageCache``, which keeps the densities and guess selection and
rebuilds the restart basis from the image geometry. Inputs that still
cannot cross processes, such as native option objects, evaluate the band
serially with a warning instead of failing.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import Atom, Molecule, PeriodicSystem, run_neb
from vibeqc.neb import _evaluate_image, _process_image_cache


def _h2(z2: float) -> Molecule:
    return Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(z2)])])


def _h3(zs) -> Molecule:
    return Molecule([Atom(1, [0.0, 0.0, float(z)]) for z in zs], 0, 2)


def _positions(molecule: Molecule) -> np.ndarray:
    return np.array([atom.xyz for atom in molecule.atoms], dtype=float)


def test_molecule_and_atom_pickle_round_trip():
    molecule = Molecule(
        [Atom(8, [0.0, 0.0, 0.2]), Atom(1, [0.0, 1.4, -0.9]), Atom(1, [0.0, -1.4, -0.9])],
        charge=-1, multiplicity=2,
    )
    copy = pickle.loads(pickle.dumps(molecule))
    assert [(a.Z, list(a.xyz)) for a in copy.atoms] == [(a.Z, list(a.xyz)) for a in molecule.atoms]
    assert (copy.charge, copy.multiplicity) == (-1, 2)
    atom = pickle.loads(pickle.dumps(Atom(6, [0.1, 0.2, 0.3])))
    assert (atom.Z, list(atom.xyz)) == (6, [0.1, 0.2, 0.3])


def test_periodic_system_pickles_until_symmetry_is_attached():
    lattice = np.diag([5.0, 6.0, 30.0])
    system = PeriodicSystem(2, lattice, [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])], 0, 1)
    copy = pickle.loads(pickle.dumps(system))
    assert copy.dim == 2
    np.testing.assert_array_equal(np.asarray(copy.lattice), lattice)
    assert [(a.Z, list(a.xyz)) for a in copy.unit_cell] == [(1, [0.0, 0.0, 0.0]), (1, [1.4, 0.0, 0.0])]
    assert (copy.charge, copy.multiplicity, copy.symmetry) == (0, 1, None)
    vq.attach_symmetry(system)
    with pytest.raises(TypeError, match="attached symmetry cannot be pickled"):
        pickle.dumps(system)


@pytest.mark.parametrize(
    ("method", "start", "moved"),
    [
        ("rhf", _h2(1.4), _h2(1.5)),
        ("uhf", _h3([0.0, 1.4, 4.4]), _h3([0.0, 1.6, 4.4])),
        ("rohf", _h3([0.0, 1.4, 4.4]), _h3([0.0, 1.6, 4.4])),
    ],
)
def test_process_cache_warm_starts_like_the_scf_result(method, start, moved):
    """A pickled process cache seeds the moved image exactly as the native
    result does: same SCF iterations, energy, gradient and READ transport."""
    kwargs = dict(
        template=start, basis_name="sto-3g", method=method, functional=None,
        rhf_options=None, uhf_options=None, rks_options=None, uks_options=None,
        gradient_options=None, grid_options=None, dispersion_params=None,
    )
    _, _, native = _evaluate_image(_positions(start), **kwargs)
    portable = pickle.loads(pickle.dumps(
        _process_image_cache(_evaluate_image, native, _positions(start), kwargs)
    ))
    assert portable.restart_basis.nbasis == native.restart_basis.nbasis
    e_native, g_native, r_native = _evaluate_image(_positions(moved), **kwargs, initial_density=native)
    e_portable, g_portable, r_portable = _evaluate_image(_positions(moved), **kwargs, initial_density=portable)
    assert e_portable == pytest.approx(e_native, abs=1e-12)
    np.testing.assert_allclose(g_portable, g_native, atol=1e-10)
    assert r_portable.n_iter == r_native.n_iter
    assert r_portable.guess_selection.transport == vq.InitialGuess.READ
    assert r_portable.guess_selection.effective == r_native.guess_selection.effective


def _band(**kwargs):
    return run_neb(
        _h2(1.3), _h2(1.6), basis="sto-3g", n_images=3, method="rhf", max_iter=4,
        initial_step=0.05, **kwargs,
    )


def test_process_band_matches_the_serial_band():
    serial = _band(n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        parallel = _band(n_jobs=2)
    # Measured 0 Ha and 2e-18 bohr.
    np.testing.assert_allclose(parallel.energies, serial.energies, atol=1e-10)
    for a, b in zip(parallel.path.images, serial.path.images):
        np.testing.assert_allclose(_positions(a.system), _positions(b.system), atol=1e-10)


def test_process_band_runs_without_warm_start():
    serial = _band(n_jobs=1, warm_start=False)
    parallel = _band(n_jobs=2, warm_start=False)
    np.testing.assert_allclose(parallel.energies, serial.energies, atol=1e-10)


def test_native_options_fall_back_to_a_serial_band():
    serial = _band(n_jobs=1)
    with pytest.warns(RuntimeWarning, match="cannot round-trip through worker processes"):
        fallback = _band(n_jobs=2, rhf_options=vq.RHFOptions())
    np.testing.assert_allclose(fallback.energies, serial.energies, atol=1e-10)
