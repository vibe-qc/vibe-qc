"""Frozen-density XC refinement, independent contractions and joint admission."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_quadrature import xc_case
from tests.test_manta_periodic_semilocal_xc import reference


def case(name='pbe', spins=2):
    system, c = xc_case(name, spins)
    grids = []
    for radial, angular, radius in ((12, 17, 8.), (16, 29, 8.), (24, 35, 12.)):
        o = manta.PeriodicQuadratureOptions()
        o.radial_points = radial; o.lebedev_order = angular; o.partition_radius_bohr = radius
        grids.append(o)
    args = {name: c[name] for name in ('basis', 'functional', 'images', 'momenta', 'weights', 'densities', 'grid_batch', 'budget')}
    args.update(system=system, grid_options=grids, reference_index=2)
    return args, c


def plan(args):
    a = dict(args); a.pop('densities'); a.pop('weights')
    a.update(images=len(a['images']), momenta=len(a['momenta']))
    return manta.plan_periodic_xc_refinement(**a)


def explicit(args, c):
    rows = []
    for options in args['grid_options']:
        g = manta.build_periodic_quadrature(args['system'], options, args['budget'])
        leaf = dict(c, points=g.points, quadrature_weights=g.weights)
        r = manta.compute_periodic_semilocal_xc(**leaf)
        e, v, pop = reference(leaf, leaf['functional'])
        np.testing.assert_allclose(r.values, v, atol=2e-11, rtol=0)
        assert r.receipt.energy == pytest.approx(e, abs=2e-11)
        rows.append((g, r, e, v, pop))
    return rows


@pytest.mark.parametrize('name,spins', [('lda', 1), ('lda', 2), ('pbe', 1), ('pbe', 2), ('blyp', 2)])
def test_refinement_compares_one_density_against_independent_energy_and_all_potentials(name, spins):
    args, c = case(name, spins); expected = explicit(args, c)
    r = manta.assess_periodic_xc_refinement(**args); target = expected[args['reference_index']]
    assert 'not-an-error-bound' in r.comparison
    assert not hasattr(r, 'converged') and not hasattr(r, 'accuracy_bound')
    assert len(r.steps) == 3 and r.plan.spins == spins
    np.testing.assert_allclose(r.reference_potential, target[3], atol=2e-11, rtol=0)
    r.reference_potential[:] = np.nan
    assert np.isfinite(r.reference_potential).all()
    for step, (grid, native, energy, potential, pop) in zip(r.steps, expected):
        assert step.quadrature.source_identity == grid.receipt.source_identity
        assert step.xc.source_identity == native.receipt.source_identity
        assert step.xc.density_identity == native.receipt.density_identity
        assert step.energy_difference == pytest.approx(energy-target[2], abs=3e-11)
        delta = potential-target[3]
        np.testing.assert_allclose(np.array(step.potential_max_abs).reshape(spins, 2),
                                   np.max(abs(delta), axis=(-1, -2)), atol=3e-11, rtol=0)
        np.testing.assert_allclose(np.array(step.potential_frobenius).reshape(spins, 2),
                                   np.linalg.norm(delta, axis=(-1, -2)), atol=3e-11, rtol=0)
        np.testing.assert_allclose(step.population_differences, pop-target[4], atol=3e-12, rtol=0)
    assert r.shell_calls == sum(row[1].receipt.shell_calls for row in expected)
    assert r.primitive_points == sum(row[1].receipt.primitive_points for row in expected)
    assert r.steps[2].energy_difference == 0 and max(r.steps[2].potential_max_abs) == 0


def test_reference_can_be_first_and_duplicate_grids_are_exactly_zero():
    args, _ = case(); args['grid_options'] = [args['grid_options'][0]]*2; args['reference_index'] = 0
    r = manta.assess_periodic_xc_refinement(**args)
    assert r.steps[1].energy_difference == 0 and max(r.steps[1].potential_max_abs) == 0
    assert r.steps[0].quadrature.source_identity == r.steps[1].quadrature.source_identity


def test_empty_beta_and_fortran_layout_preserve_frozen_state():
    args, c = case('blyp', 2); args['densities'][1] = 0
    expected = explicit(args, c); r = manta.assess_periodic_xc_refinement(**args)
    np.testing.assert_allclose(r.reference_potential, expected[2][3], atol=3e-11, rtol=0)
    assert all(step.xc.electrons_grid[1] == 0 for step in r.steps)
    for name in ('images', 'momenta', 'weights', 'densities'):
        other = manta.assess_periodic_xc_refinement(**dict(args, **{name: np.asfortranarray(args[name])}))
        assert other.source_identity == r.source_identity and other.density_identity == r.density_identity
        np.testing.assert_array_equal(other.reference_potential, r.reference_potential)


@pytest.mark.parametrize('field,quantity', [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_points'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_all_resources_are_summed_and_exact_caps_refuse_one_short_before_inputs(field, quantity):
    args, _ = case(); p = plan(args); b = args['budget']
    for name, q in [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
                   ('primitive_quartets', 'primitive_points'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')]:
        setattr(b, name, getattr(p, q))
    r = manta.assess_periodic_xc_refinement(**args)
    assert r.plan.modeled_peak_bytes == p.modeled_peak_bytes
    assert r.shell_calls <= p.shell_calls and r.primitive_points <= p.primitive_points
    setattr(b, field, getattr(p, quantity)-1); args['densities'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): plan(args)
    with pytest.raises((ValueError, RuntimeError), match='budget'): manta.assess_periodic_xc_refinement(**args)


def test_external_memory_and_retained_reference_are_in_joint_peak():
    args, _ = case(); p = plan(args)
    assert p.driver_live_bytes > args['densities'].nbytes*2
    assert p.modeled_peak_bytes == max(max(q.modeled_peak_bytes, x.modeled_peak_bytes) for q, x in zip(p.quadrature, p.xc))
    assert p.scalar_terms > sum(q.scalar_terms+x.scalar_terms for q, x in zip(p.quadrature, p.xc))
    assert p.kernel_terms > sum(q.kernel_terms+x.kernel_terms for q, x in zip(p.quadrature, p.xc))
    args['budget'].other_live_bytes = 12345
    assert plan(args).modeled_peak_bytes == p.modeled_peak_bytes+12345
    args['budget'].scalar_terms = max(x.scalar_terms for x in p.xc)
    with pytest.raises(ValueError, match='budget'): manta.assess_periodic_xc_refinement(**args)


@pytest.mark.parametrize('field', ['images', 'momenta'])
def test_count_overflow_refuses_without_array_allocation(field):
    args, _ = case()
    counts = dict(images=len(args['images']), momenta=len(args['momenta']))
    counts[field] = 2**63
    with pytest.raises(OverflowError, match='overflow'):
        manta.plan_periodic_xc_refinement(args['basis'], args['system'], args['functional'],
            counts['images'], counts['momenta'], args['grid_options'], 2, 64, args['budget'])


@pytest.mark.parametrize('fault', ['nan', 'nonhermitian', 'weights', 'images'])
def test_invalid_frozen_input_refuses_before_invalid_grid_generation(fault):
    args, _ = case(); s = args['system']
    args['system'] = vq.PeriodicSystem(3, s.lattice, [s.unit_cell[0]]*2)
    if fault == 'nan': args['densities'][0, 0, 0, 0] = np.nan
    elif fault == 'nonhermitian': args['densities'][0, 0, 0, 1] += 1j
    elif fault == 'weights': args['weights'] *= 2
    else: args['images'][0, 0] += .01
    with pytest.raises(ValueError, match='finite|Hermitian|sum to one|translations'):
        manta.assess_periodic_xc_refinement(**args)


@pytest.mark.parametrize('fault', ['empty', 'single', 'many', 'reference', 'hybrid', 'grid', 'layout', 'spin'])
def test_unadmitted_requests(fault):
    args, _ = case()
    if fault == 'empty': args['grid_options'] = []
    elif fault == 'single': args['grid_options'] = args['grid_options'][:1]
    elif fault == 'many': args['grid_options'] *= 11
    elif fault == 'reference': args['reference_index'] = 3
    elif fault == 'hybrid': args['functional'] = vq.Functional('pbe0', spin=2)
    elif fault == 'grid': args['grid_options'][1].radial_points = 0
    elif fault == 'layout': args['densities'] = args['densities'].astype(np.complex64)
    else: args['densities'] = args['densities'][:1].copy()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        manta.assess_periodic_xc_refinement(**args)


def test_source_frame_density_identity_owned_results_and_batch_independence():
    args, _ = case(); original = manta.assess_periodic_xc_refinement(**args)
    args['grid_batch'] = 1
    small = manta.assess_periodic_xc_refinement(**args)
    assert original.source_identity == small.source_identity and original.density_identity == small.density_identity
    args['grid_options'][0].radial_points += 1
    changed = manta.assess_periodic_xc_refinement(**args)
    assert changed.source_identity != original.source_identity and changed.density_identity == original.density_identity
    args['densities'] *= .9
    density = manta.assess_periodic_xc_refinement(**args)
    assert density.source_identity == changed.source_identity and density.density_identity != changed.density_identity
    saved = original.steps; identity = saved[0].quadrature.source_identity
    saved[0].options.radial_points = 999
    saved[0].potential_max_abs[0] = 888
    assert original.steps[0].options.radial_points == 12 and original.steps[0].potential_max_abs[0] != 888
    del original, args
    assert saved[0].quadrature.source_identity == identity


def test_shell_image_gauge_preserves_every_comparison_norm():
    args, c = case(); original = manta.assess_periodic_xc_refinement(**args)
    shells = args['basis'].shells(); lattice = args['system'].lattice
    shells[0].origin = np.array(shells[0].origin)+lattice[:, 0]
    args['basis'] = vq.BasisSet(vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1),
                              shells, 'xc-refinement-gauge', False)
    phase = np.ones((len(args['momenta']), 2), complex); phase[:, 0] = np.exp(1j*args['momenta']@lattice[:, 0])
    args['densities'] *= phase[:, :, None]*phase.conj()[:, None, :]
    moved = manta.assess_periodic_xc_refinement(**args)
    assert moved.source_identity != original.source_identity and moved.density_identity != original.density_identity
    for before, after in zip(original.steps, moved.steps):
        assert after.energy_difference == pytest.approx(before.energy_difference, abs=3e-12)
        np.testing.assert_allclose(after.potential_max_abs, before.potential_max_abs, atol=3e-11, rtol=0)
        np.testing.assert_allclose(after.potential_frobenius, before.potential_frobenius, atol=3e-11, rtol=0)


@pytest.mark.parametrize('name,spins,empty', [('lda', 1, False), ('lda', 2, False),
    ('pbe', 1, False), ('pbe', 2, False), ('blyp', 2, True)])
def test_vector_integration_witness_matches_weighted_native_energy_and_all_potentials(name, spins, empty):
    from benchmarks.manta_periodic_xc_reference import pointwise_xc, unpack_integral
    _, c = case(name, spins)
    if empty: c['densities'][1] = 0
    rng = np.random.default_rng(731); points = rng.random((103, 3)); weights = rng.random(103)/103
    integrated = weights@pointwise_xc(c, points)
    energy, potential, populations = unpack_integral(integrated, c['densities'].shape)
    native = manta.compute_periodic_semilocal_xc(**dict(c, points=points@c['lattice'].T,
        quadrature_weights=weights*abs(np.linalg.det(c['lattice']))))
    assert energy == pytest.approx(native.receipt.energy, abs=2e-12)
    np.testing.assert_allclose(potential, native.values, atol=2e-12, rtol=0)
    np.testing.assert_allclose(populations, native.receipt.electrons_grid, atol=2e-12, rtol=0)


def test_vector_witness_complex_density_derivative_includes_only_density_k_weight():
    from benchmarks.manta_periodic_xc_reference import pointwise_xc, unpack_integral
    _, c = case()
    rng = np.random.default_rng(617); points = rng.random((119, 3)); weights = rng.random(119)/119
    _, potential, _ = unpack_integral(weights@pointwise_xc(c, points), c['densities'].shape)
    for spin in (0, 1):
        delta = np.zeros_like(c['densities']); delta[spin, 1, 0, 1] = 1j; delta[spin, 1, 1, 0] = -1j
        h = 2e-5; energies = []
        for multiplier in (-2, -1, 1, 2):
            shifted = dict(c, densities=c['densities']+multiplier*h*delta)
            energies.append(float(weights@pointwise_xc(shifted, points)[:, 0]))
        slope = (energies[0]-8*energies[1]+8*energies[2]-energies[3])/(12*h)
        expected = c['weights'][1]*np.trace(delta[spin, 1]@potential[spin, 1]).real
        assert slope == pytest.approx(expected, abs=3e-9)
