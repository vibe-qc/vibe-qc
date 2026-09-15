"""Lossless producer-owned multi-k GDF density and artifact regressions."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import json
import zipfile

import numpy as np
import pytest
import vibeqc as vq

from vibeqc.periodic_k_density import _plan_lattice_density_return
from vibeqc.periodic_runner import _fit_exact_lattice_density_to_weighted_k_matrices


def _system(spin=1):
    return vq.PeriodicSystem(
        3, np.array([[6., 0.7, 0.2], [0., 5.8, 0.4], [0., 0., 6.1]]),
        [vq.Atom(1, [2.3, 2.7, 3.]), vq.Atom(1, [3.7, 2.7, 3.])], 0, spin,
    )


def _plan(system, mesh=(3, 2, 1), shift=(0, 0, 0), channels=1, cap=2**20):
    km = vq.monkhorst_pack(system, list(mesh), list(shift), False)
    return _plan_lattice_density_return(
        system, km.kpoints, km.weights, mesh, nbf=2,
        n_channels=channels, memory_byte_cap=cap,
    )


def _state(plan):
    # Nontrivial complex Hermitian state with exact time-reversal partners;
    # convex mixing leaves a state that need not come from the final orbitals.
    rng = np.random.default_rng(142)
    densities = [None] * len(plan.kpoints)
    for i, j in enumerate(plan.partner_indices):
        if densities[i] is not None:
            continue
        c = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        d = c @ c.conj().T / 10
        if i == j:
            d = d.real.astype(complex)
        densities[i] = d
        densities[j] = d.conj()
    return densities


@pytest.mark.parametrize('mesh,shift', [
    ((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)),
    ((3, 2, 1), (0, 0, 0)), ((3, 2, 1), (1, 1, 1)),
    ((2, 2, 2), (1, 0, 1)),
])
def test_lattice_return_round_trips_accepted_complex_density(mesh, shift):
    system = _system()
    plan = _plan(system, mesh, shift)
    densities = _state(plan)
    lattice = plan.fold(densities)
    recovered = _fit_exact_lattice_density_to_weighted_k_matrices(
        lattice, plan.kpoints, plan.weights, system=system, bvk_mesh=mesh,
        label='GDF regression',
    )
    np.testing.assert_allclose(recovered, np.asarray(densities) / np.prod(mesh), rtol=0, atol=2e-14)
    keys = {tuple(cell.index) for cell in lattice.cells}
    assert all(tuple(-x for x in key) in keys for key in keys)
    assert len(keys) <= 2 * np.prod(mesh)
    for cell in lattice.cells:
        np.testing.assert_allclose(cell.r_cart, system.lattice @ cell.index, rtol=0, atol=1e-14)


def test_rejects_real_projection_that_would_change_self_inverse_k_state():
    plan = _plan(_system(), (2, 1, 1))
    densities = [np.array([[1, 0.2j], [-0.2j, 1]])] * 2
    with pytest.raises(ValueError, match='time reversal'):
        plan.fold(densities)


@pytest.mark.parametrize('dimension', [-1, 0, 4])
def test_native_lattice_cell_rejects_mutated_invalid_dimension(dimension):
    system = _system()
    system.dim = dimension
    with pytest.raises(ValueError, match='invalid periodic dimension'):
        vq._vibeqc_core.LatticeCell(system, [0, 0, 0])


def test_fold_compares_to_original_without_hermitian_projection():
    plan = _plan(_system())
    densities = _state(plan)
    # A malformed finite character set must also fail the actual round trip,
    # even if every source density passed Hermiticity and time reversal.
    corrupted = replace(plan, cells=plan.cells[::-1])
    with pytest.raises(ValueError, match='round trip'):
        corrupted.fold(densities)


def test_lattice_density_admission_precedes_cell_allocation(monkeypatch):
    system = _system()
    plan = _plan(system)
    assert _plan(system, cap=plan.reserved_peak_bytes).reserved_peak_bytes == plan.reserved_peak_bytes
    with pytest.raises(MemoryError):
        _plan(system, cap=plan.reserved_peak_bytes - 1)
    def forbidden(*args, **kwargs):
        pytest.fail('allocated cells before memory admission')
    monkeypatch.setattr(vq._vibeqc_core, 'LatticeCell', forbidden)
    with pytest.raises(MemoryError):
        _plan_lattice_density_return(
            system, [], [], (10000, 10000, 10000), nbf=100,
            n_channels=2, memory_byte_cap=2**20,
        )


def test_driver_density_admission_precedes_k_mesh_expansion(monkeypatch):
    import vibeqc.periodic_k_gdf as module

    def forbidden(*args, **kwargs):
        pytest.fail('expanded k mesh before density admission')
    monkeypatch.setattr(module, '_kmesh_to_kpoints_weights', forbidden)
    with pytest.raises(MemoryError):
        module._gdf_density_return_finalizer(
            _system(), SimpleNamespace(nbasis=100), (10000, 10000, 10000),
            True, 2, 2**20,
        )


@pytest.mark.parametrize('defect', [
    'weights', 'duplicate', 'metadata', 'non_mp', 'complex_points', 'complex_weights',
])
def test_mesh_validation_refuses_invented_bvk_metadata(defect):
    system = _system()
    plan = _plan(system)
    points, weights, mesh = plan.kpoints.copy(), plan.weights.copy(), plan.mesh
    if defect == 'weights':
        weights[0] += .01
        weights[1] -= .01
    elif defect == 'duplicate':
        points[0] = points[1]
    elif defect == 'metadata':
        mesh = (1, 1, 1)
    elif defect == 'complex_points':
        points = points.astype(complex) + 1j
    elif defect == 'complex_weights':
        weights = weights.astype(complex) + 1j
    else:
        points[0, 0] += .01
    with pytest.raises(ValueError):
        _plan_lattice_density_return(system, points, weights, mesh, nbf=2, n_channels=1, memory_byte_cap=2**20)


def test_both_spin_channels_are_certified_before_attachment():
    plan = _plan(_system(), (2, 1, 1), channels=2)
    result = SimpleNamespace(
        density_alpha=_state(plan), density_beta=[np.array([[1, .2j], [-.2j, 1]])] * 2,
        kpoints_cart=plan.kpoints, kpoint_weights=plan.weights,
    )
    with pytest.raises(ValueError, match='time reversal'):
        plan.attach(result)
    assert not hasattr(result, 'density_alpha_lattice')


@pytest.mark.parametrize('method,spin', [('rhf', 1), ('uhf', 3), ('rohf', 3)])
def test_actual_gdf_driver_returns_its_accepted_density(method, spin):
    system = _system(spin)
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    opts = vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.max_iter = 1
    opts.lattice_opts.cutoff_bohr = 12
    opts.lattice_opts.nuclear_cutoff_bohr = 18
    driver = getattr(vq, 'run_k' + method + '_periodic_gdf')
    result = driver(
        system, basis, (3, 1, 1), opts,
        rsgdf_ke_cutoff=12, rcut_strategy=None, return_lattice_density=True,
        check_energy_sanity=False, progress=False,
    )
    assert not result.converged
    channels = ('density',) if spin == 1 else ('density_alpha', 'density_beta')
    for channel in channels:
        original = getattr(result, channel)
        returned = getattr(result, channel + '_lattice')
        recovered = _fit_exact_lattice_density_to_weighted_k_matrices(
            returned, result.kpoints_cart, result.kpoint_weights,
            system=system, bvk_mesh=result.density_lattice_mesh, label=channel,
        )
        np.testing.assert_allclose(recovered, np.asarray(original) / 3, rtol=0, atol=2e-12)


def test_public_gdf_multik_qvf_and_xsf_keep_the_returned_density(tmp_path):
    from vibeqc.output.formats.qvf import validate_qvf
    system = _system()
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    output = tmp_path / 'gdf_k211'
    result = vq.run_periodic_job(
        system, basis, method='RHF', jk_method='gdf', kpoints=(2, 1, 1),
        initial_guess='HCORE', max_iter=60, convergence='off',
        rsgdf_ke_cutoff=12, output=output, output_qvf=True,
        write_density=True, write_molden_file=False, write_population_file=False,
        density_spacing_bohr=.25, citations=False, progress=False,
    )
    assert result.converged
    assert result.density_lattice is not None
    assert output.with_suffix('.xsf').is_file()
    archive = output.with_suffix('.qvf')
    assert validate_qvf(archive)['valid']
    with zipfile.ZipFile(archive) as zf:
        manifest = json.loads(zf.read('manifest.json'))
        section = next(s for s in manifest['sections'] if s['kind'] == 'volume.density')
        grid = json.loads(zf.read(section['members']['grid']['path']))
        data = np.frombuffer(zf.read(section['members']['data']['path']), dtype=np.float32)
    dvol = abs(np.linalg.det(np.asarray(grid['voxel_vectors'])))
    assert data.sum(dtype=float) * dvol == pytest.approx(2., abs=5e-3)
    # Independent direct Bloch-AO evaluation of the accepted density, not an
    # orbital refill or a transform of the newly returned lattice blocks.
    from vibeqc.periodic_density import evaluate_weighted_k_density_on_grid
    expected, shape = evaluate_weighted_k_density_on_grid(
        basis, system,
        [w * d for w, d in zip(result.kpoint_weights, result.density)],
        result.kpoints_cart, grid_shape=tuple(grid['shape']),
        spacing_bohr=.25, ao_image_radius=3,
    )
    np.testing.assert_allclose(data, np.asarray(expected).ravel(), rtol=2e-6, atol=2e-8)
