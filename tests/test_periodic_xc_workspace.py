from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core


@pytest.fixture(autouse=True)
def restore_native_thread_count():
    previous = vq.get_num_threads()
    try:
        yield
    finally:
        vq.set_num_threads(previous)


def basis_at(coords, angular):
    system = core.PeriodicSystem(3, np.eye(3) * 6., [vq.Atom(1, p) for p in coords])
    basis = core.BasisSet(system.unit_cell_molecule(), [
        core.ShellInfo(a, angular, True, [.7, 1.3], [.8, .2], p)
        for a, p in enumerate(coords)
    ], 'custom-xc-no-library-entry', coefficients_pre_normalized=False)
    return system, basis


def density_for(system, basis, opts, scale):
    cells = core.direct_lattice_cells(system, opts.cutoff_bohr)
    rng = np.random.default_rng(739)
    d = rng.normal(size=(basis.nbasis, basis.nbasis))
    d = (d @ d.T + np.eye(basis.nbasis)) * scale
    return core.make_lattice_matrix_set(basis.nbasis, cells,
        [d * np.exp(-.2 * np.dot(c.r_cart, c.r_cart)) for c in cells])


def fixed_grid(count):
    rng = np.random.default_rng(142)
    return core.Grid(rng.normal(size=(count, 3)) + [.8, .4, .5],
                                np.full(count, 3. / count))


def gradient(basis, system, grid, func, densities, opts):
    call = (core.xc_lattice_gradient_contribution if len(densities) == 1
            else core.xc_lattice_gradient_contribution_uks)
    return np.asarray(call(basis, system, grid, func, *densities, opts))


@pytest.mark.parametrize('angular', [0, 1, 2])
@pytest.mark.parametrize('functional', ['svwn', 'pbe', 'tpss'])
@pytest.mark.parametrize('spin', [1, 2])
def test_custom_periodic_basis_fixed_grid_force(angular, functional, spin):
    vq.set_num_threads(2)
    coords = np.array([[.2, .3, .4], [2.1, 1.2, .8]])
    system, basis = basis_at(coords, angular)
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = 6.2
    densities = [density_for(system, basis, opts, .05)]
    if spin == 2:
        densities.append(density_for(system, basis, opts, .03))
    grid = fixed_grid(97)
    func = core.Functional(functional, spin)
    actual = gradient(basis, system, grid, func, densities, opts)
    call = core.build_xc_periodic if spin == 1 else core.build_xc_periodic_uks

    def energy(positions):
        system, basis = basis_at(positions, angular)
        return call(basis, system, grid, func, *densities, opts).e_xc

    step = 2e-5
    expected = np.empty((2, 3))
    for atom in range(2):
        for axis in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            expected[atom, axis] = (energy(plus) - energy(minus)) / (2 * step)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('spin', [1, 2])
@pytest.mark.parametrize('functional', ['svwn', 'pbe', 'tpss'])
def test_periodic_home_block_density_keeps_image_bras_in_force(spin, functional):
    """Zero off-cell blocks do not turn a lattice density into a molecule."""
    from vibeqc.periodic_external_xc import periodic_xc_difference_closed_cells

    coords = np.array([[.2, .3, .4], [2.1, 1.2, .8]])
    system, basis = basis_at(coords, 1)
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = 6.2
    opts.becke_image_radius_bohr = 6.2
    cells = periodic_xc_difference_closed_cells(system, 6.2, 6.2)
    home = np.eye(basis.nbasis) * .1
    densities = [core.make_lattice_matrix_set(basis.nbasis, cells, [
        home * scale if np.all(np.asarray(cell.index) == 0)
        else np.zeros_like(home) for cell in cells
    ]) for scale in ([1.] if spin == 1 else [.6, .4])]
    rng = np.random.default_rng(715)
    grid = core.Grid(rng.uniform(-6., 6., (137, 3)), np.full(137, .1))
    func = core.Functional(functional, spin)
    domain = core.PeriodicXCDensityDomain.PERIODIC_LATTICE
    call = (core.xc_lattice_gradient_contribution if spin == 1
            else core.xc_lattice_gradient_contribution_uks)
    actual = np.asarray(call(basis, system, grid, func, *densities, opts, domain))
    automatic = np.asarray(call(basis, system, grid, func, *densities, opts))
    assert np.max(np.abs(actual - automatic)) > 1e-5
    value = core.build_xc_periodic if spin == 1 else core.build_xc_periodic_uks
    expected = np.empty_like(actual)
    h = 2e-5
    for atom in range(2):
        for axis in range(3):
            energies = []
            for sign in (1., -1.):
                shifted = coords.copy()
                shifted[atom, axis] += sign * h
                sd, bd = basis_at(shifted, 1)
                energies.append(value(bd, sd, grid, func, *densities, opts, domain).e_xc)
            expected[atom, axis] = (energies[0] - energies[1]) / (2 * h)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('spin', [1, 2])
@pytest.mark.parametrize('threads', [1, 2])
def test_batch_partition_preserves_same_quadrature(spin, threads):
    vq.set_num_threads(threads)
    system, basis = basis_at([[.2, .3, .4], [2.1, 1.2, .8]], 1)
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = 6.2
    densities = [density_for(system, basis, opts, .05)]
    if spin == 2:
        densities.append(density_for(system, basis, opts, .03))
    func = core.Functional('pbe', spin)
    grid = fixed_grid(4103)
    actual = gradient(basis, system, grid, func, densities, opts)
    expected = np.zeros_like(actual)
    for first in range(0, len(grid.weights), 701):
        part = core.Grid(np.asarray(grid.points)[first:first + 701],
                                    np.asarray(grid.weights)[first:first + 701])
        expected += gradient(basis, system, part, func, densities, opts)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


def test_workspace_planning_precedes_ao_allocation():
    count = core.periodic_xc_gradient_batch_size
    assert count(2, 7, True, True, 16) == 4096
    assert 0 < count(200, 503, True, True, 16) < 4096
    assert count(200, 503, True, True, 16) <= count(200, 503, True, False, 16)
    with pytest.raises(RuntimeError, match='one grid point'):
        count(10**9, 503, True, True, 16)


def test_value_memory_workspace_is_bounded_while_retained_grid_grows():
    from vibeqc.memory import estimate_periodic_xc_value

    args = dict(n_basis=100, n_atoms=4, n_cells=1000,
                functional_kind='GGA', open_shell=True, n_threads=16)
    small = estimate_periodic_xc_value(n_grid_points=10000, **args)
    large = estimate_periodic_xc_value(n_grid_points=1000000, **args)
    key = 'Periodic-XC value numerical workspace'
    assert small.by_category[key] == large.by_category[key]
    assert large.by_category[key] <= 256 * 1024**2
    assert large.dims['batch_points'] == core.periodic_xc_value_batch_size(
        100, 1000, True, True, 16,
    )
    key = 'Periodic-XC value input grid'
    assert large.by_category[key] - small.by_category[key] == 44 * 990000


@pytest.mark.parametrize('angular', [0, 1, 2])
@pytest.mark.parametrize('functional', ['svwn', 'pbe', 'tpss'])
@pytest.mark.parametrize('spin', [1, 2])
def test_value_and_fock_preserve_grid_partition(angular, functional, spin):
    vq.set_num_threads(2)
    system, basis = basis_at([[.2, .3, .4], [2.1, 1.2, .8]], angular)
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = 6.2
    densities = [density_for(system, basis, opts, .05)]
    if spin == 2:
        densities.append(density_for(system, basis, opts, .03))
    func = core.Functional(functional, spin)
    grid = fixed_grid(97)
    candidate = core.build_xc_periodic if spin == 1 else core.build_xc_periodic_uks
    domain = core.PeriodicXCDensityDomain.AUTO
    actual = candidate(basis, system, grid, func, *densities, opts, domain)
    attrs = ['V_xc'] if spin == 1 else ['V_alpha', 'V_beta']
    parts = []
    for first in range(0, len(grid.weights), 13):
        part = core.Grid(np.asarray(grid.points)[first:first + 13],
                                    np.asarray(grid.weights)[first:first + 13])
        parts.append(candidate(basis, system, part, func, *densities, opts, domain))
    assert actual.e_xc == pytest.approx(sum(p.e_xc for p in parts), rel=2e-12, abs=2e-12)
    for attr in attrs:
        blocks = sum(np.asarray(getattr(p, attr).blocks) for p in parts)
        np.testing.assert_allclose(getattr(actual, attr).blocks, blocks, atol=2e-12, rtol=2e-12)


def test_value_workspace_admits_before_allocating():
    count = core.periodic_xc_value_batch_size
    assert count(2, 7, True, True, 2) == 4096
    assert 0 < count(100, 1000, True, True, 16) < 4096
    assert count(100, 1000, True, True, 16) <= count(100, 1000, False, False, 1)
    for args in [(0, 1, True, False, 1), (1, 0, True, False, 1), (1, 1, True, False, 0)]:
        with pytest.raises(ValueError, match='dimensions'):
            count(*args)
    with pytest.raises(RuntimeError, match='one grid point'):
        count(10**9, 503, True, True, 16)


def test_padding_density_domain_does_not_expand_active_ao_workspace():
    from vibeqc.periodic_external_xc import periodic_xc_difference_closed_cells

    system, basis = basis_at([[.2, .3, .4], [2.1, 1.2, .8]], 1)
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = opts.becke_image_radius_bohr = 6.2
    domain = core.PeriodicXCDensityDomain.PERIODIC_LATTICE
    cells = periodic_xc_difference_closed_cells(system, 6.2, 6.2)
    padded = core.direct_lattice_cells(system, 25.)
    assert len(padded) > len(cells)
    assert {tuple(c.index) for c in cells} <= {tuple(c.index) for c in padded}
    counts = core.periodic_xc_domain_counts(system, cells, opts, domain)
    assert counts == core.periodic_xc_domain_counts(system, padded, opts, domain)
    home = np.eye(basis.nbasis) * .1
    densities = [core.make_lattice_matrix_set(basis.nbasis, domain_cells, [
        home if np.all(np.asarray(cell.index) == 0) else np.zeros_like(home)
        for cell in domain_cells
    ]) for domain_cells in (cells, padded)]
    grid, func = fixed_grid(79), core.Functional('tpss', 1)
    energies, gradients = [], []
    for density in densities:
        energies.append(core.build_xc_periodic(basis, system, grid, func, density, opts, domain).e_xc)
        gradients.append(core.xc_lattice_gradient_contribution(
            basis, system, grid, func, density, opts, domain,
        ))
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradients[0], gradients[1], atol=1e-12, rtol=0)
