"""Generated range KS: independent supports, joint admission and physical SCF."""
import itertools
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_ri_range_ks import (
    case, run as explicit_run, reference, as_fock_build, CAPS, MIXED_RANGE, EMPTY_RANGE,
    components, independent_scf, _nuclear_reference,
)
from tests.test_manta_periodic_ri_ks_domains import ARRAYS, run_args
from tests.test_manta_periodic_domains import _labels

PROFILES = ['J', 'K', 'JK']


def arguments(profile='JK', spins=1, grid=False, name='cam-b3lyp', **kwargs):
    a, k, old = case(profile, spins, name, **kwargs)
    args = {key: a[key] for key in ('basis', 'auxiliary', 'system', 'mesh', 'functional', 'options', 'budget')}
    args.update(reciprocal_energy_cutoff=6., xc_image_cutoff_bohr=11.)
    if grid:
        q = manta.PeriodicQuadratureOptions(); q.radial_points = 12; q.lebedev_order = 17
        q.partition_radius_bohr = 8.; args['grid_options'] = q
    else:
        args['grid_points'] = len(a['points'])
    return args, a, k, old


def function(args, verb='run', namespace=core._manta):
    suffix = 'domains' if verb == 'build' else ('with_quadrature' if 'grid_options' in args else 'from_cutoffs')
    return getattr(namespace, verb+'_periodic_ri_range_separated_ks_'+suffix)


def explicit(args, a, old):
    da = dict(args); grid = None
    if 'grid_options' in da:
        grid = manta.build_periodic_quadrature(args['system'], da.pop('grid_options'), args['budget'])
        da['grid_points'] = grid.plan.grid_points
        a.update(points=grid.points, quadrature_weights=grid.weights)
    d = function(da, 'build')(**da)
    for key in ('translations', 'reciprocal_vectors', 'exchange_vectors', 'transfer_offsets', 'xc_images'):
        a[key] = getattr(d, key)
    old.update(reciprocal_vectors=a['reciprocal_vectors'], translations=a['translations'])
    return d, grid


def check_reference(r, a, k, old):
    s, h, build = reference(a, k, old)
    pop = (r.plan.n_alpha,) if r.plan.spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
    d = independent_scf(s, h, build, pop, r.plan.spins)
    f, energy, counts = build(r.density)
    expected = sum(_nuclear_reference(old)[0])+sum(build(d)[1])
    errors = dict(S=float(np.max(abs(r.overlap-s))), H=float(np.max(abs(r.hcore-h))),
        D=float(np.max(abs(r.density-d))), F=float(np.max(abs(r.fock-f))),
        components=float(np.max(abs(np.asarray(components(r))-energy))), energy=abs(r.receipt.energy-expected),
        grid_populations=float(np.max(abs(np.asarray(r.receipt.xc.electrons_grid)-counts))))
    for key, value in errors.items():
        assert value < (5e-9 if key == 'D' else 8e-10 if key == 'F' else 8e-11), (key, value)
    return errors


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((3, 2, 1), (1, 1, 0))])
def test_independent_spheres_do_not_embed_physical_screened_zero(profile, shape, shift):
    args, _, _, _ = arguments(profile, mesh=shape, shift=shift)
    lattice = np.array([[6., 3.5, .8], [0., 4.2, 1.2], [0., 0., 6.4]])
    args['system'] = vq.PeriodicSystem(3, lattice, args['system'].unit_cell)
    args['reciprocal_energy_cutoff'] = 2.
    d = function(args, 'build')(**args); b = 2*np.pi*np.linalg.inv(lattice).T
    labels = np.array(list(itertools.product(range(-8, 9), repeat=3)))
    g = labels@b.T; images = labels@lattice.T
    keep = (np.sum(g*g, axis=1) <= 4) & np.any(labels, axis=1)
    assert _labels(d.reciprocal_vectors, np.linalg.inv(b)) == {tuple(n) for n in labels[keep]}
    for field, radius in [('translations', d.plan.translation_radius_bohr), ('xc_images', args['xc_image_cutoff_bohr'])]:
        assert _labels(getattr(d, field), np.linalg.inv(lattice)) == {
            tuple(n) for n in labels[np.sum(images*images, axis=1) <= radius**2]}
    k = np.array([args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))])@b.T
    for i, ki in enumerate(k):
        for j, kj in enumerate(k):
            lo, hi = d.transfer_offsets[i*len(k)+j:i*len(k)+j+2]
            actual = d.exchange_vectors[lo:hi]
            if profile != 'J' and i == j:
                assert len(actual) == 0
                continue
            transfer = ki-kj if profile == 'J' else kj-ki
            trial = g+transfer
            keep = (np.sum(trial*trial, axis=1) <= 4) & np.any(trial, axis=1)
            assert _labels(actual-transfer, np.linalg.inv(b)) == {tuple(n) for n in labels[keep]}
            revlo, revhi = d.transfer_offsets[j*len(k)+i:j*len(k)+i+2]
            assert {tuple(np.round(x, 12)) for x in actual} == {
                tuple(np.round(-x, 12)) for x in d.exchange_vectors[revlo:revhi]}
    assert max(np.diff(d.transfer_offsets)) <= d.plan.maximum_exchange_candidates


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('grid', [False, True])
@pytest.mark.parametrize('name', ['hse06', 'cam-b3lyp', MIXED_RANGE])
@pytest.mark.parametrize('gamma', [False, True])
def test_generated_complete_scf_matches_explicit_and_independent_sources(profile, spins, grid, name, gamma):
    kw = dict(mesh=(1, 1, 1), shift=(0, 0, 0)) if gamma else {}
    args, a, k, old = arguments(profile, spins, grid, name, **kw)
    d, q = explicit(args, a, old); p = function(args, 'plan')(**args)
    r = function(args)(**run_args(args, a)); separate = explicit_run(a)
    assert r.converged and r.receipt.confirmed
    for key in ARRAYS: np.testing.assert_array_equal(getattr(r, key), getattr(separate, key))
    for key in ('energy', 'source_identity', 'final_density_identity', 'final_fock_identity'):
        assert getattr(r.receipt, key) == getattr(separate.receipt, key)
    assert r.receipt.domain_identity == d.receipt.identity
    for _, quantity in CAPS: assert getattr(r.plan, quantity) <= getattr(p, quantity)
    if grid:
        assert r.receipt.quadrature.source_identity == q.receipt.source_identity
        assert r.plan.quadrature_live_bytes == p.grid_live_bytes
        assert p.scalar_terms == p.quadrature.scalar_terms+p.domains.scalar_terms
    else:
        assert r.receipt.quadrature is None
    if name == 'hse06': assert r.receipt.exchange_correction_energy == 0.
    fits = [c.fitting_info for c in r.receipt.exchange_components if c.fitting_info]
    assert all(len(f.metrics) == 1+len(k)*(len(k)-1) for f in fits)
    check_reference(r, a, k, old)


@pytest.mark.parametrize('profile', PROFILES)
def test_generated_empty_beta_with_long_range_correction(profile):
    args, a, k, old = arguments(profile, 2, True, EMPTY_RANGE, species='H')
    explicit(args, a, old); r = function(args)(**args)
    assert r.converged and r.receipt.confirmed and r.plan.n_beta == 0
    check_reference(r, a, k, old)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('grid', [False, True])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_all_joint_caps_precede_source_or_payload_generation(profile, grid, field, quantity):
    args, a, _, _ = arguments(profile, 2, grid, MIXED_RANGE); args['options'].ks.scf.maximum_iterations = 2
    args['budget'].other_live_bytes = 12345; p = function(args, 'plan')(**args)
    for key, value in CAPS: setattr(args['budget'], key, getattr(p, value))
    r = function(args)(**run_args(args, a)); assert r.receipt.fock_calls == 2
    setattr(args['budget'], field, getattr(p, quantity)-1); a['points'][:] = np.nan
    for verb, kw in [('plan', args), ('run', run_args(args, a))]:
        with pytest.raises((ValueError, RuntimeError), match='budget'): function(args, verb)(**kw)
    if not grid:
        with pytest.raises((ValueError, RuntimeError), match='budget'): function(args, 'build')(**args)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('grid', [False, True])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_selected_admission_includes_all_kernel_provenance(profile, grid, field, quantity, monkeypatch):
    from vibeqc import output
    from vibeqc.manta_periodic import _periodic_ri_range_ks_configuration
    args, a, _, _ = arguments(profile, 2, grid, MIXED_RANGE); args['options'].ks.scf.maximum_iterations = 2
    p = function(args, 'plan')(**args); budget = args['budget']
    engine = manta.IntegralEngineOptions(backend='manta', budget=budget)
    _, _, python_bytes = _periodic_ri_range_ks_configuration(args['options'], engine, args['mesh'], args['functional'])
    kw = run_args(args, a); kw.pop('budget'); kw['integral_engine'] = engine
    calls = []; monkeypatch.setattr(output, 'record', lambda *a, **kw: calls.append(kw))
    limit = getattr(p, quantity)+(python_bytes if field == 'memory_bytes' else 0)
    setattr(budget, field, limit); selected = function(args, namespace=manta)
    selected(**kw); assert len(calls) == 1 and budget.other_live_bytes == 0
    setattr(budget, field, limit-1); a['points'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): selected(**kw)
    assert len(calls) == 1


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('grid', [False, True])
def test_selected_output_and_shared_fock_retain_kernel_and_grid_sources(tmp_path, profile, grid):
    from vibeqc.output import OutputChannel, structured_log
    args, a, k, _ = arguments(profile, 2, grid, MIXED_RANGE); args['options'].ks.scf.maximum_iterations = 3
    kw = run_args(args, a); budget = kw.pop('budget')
    kw['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    with OutputChannel.to_file(tmp_path/'science.out'), structured_log(tmp_path/'science.jsonl'):
        r = function(args, namespace=manta)(**kw)
    evidence = json.loads((tmp_path/'science.jsonl').read_text())['integral_provenance']
    assert evidence['energy'] == r.receipt.energy and evidence['operator_kind'] == 'ks'
    assert not evidence['symmetry_reduced']
    assert evidence['domain_identity'] == r.receipt.domain_identity
    assert (evidence.get('quadrature') is not None) == grid
    assert evidence['citation_methods'][0] == 'manta_periodic_ri_range_separated_ks'
    json.dumps(evidence, allow_nan=False)
    state = as_fock_build(r, k)
    assert state.energy.total == pytest.approx(r.receipt.energy, abs=1e-13)
    for channel, fock in zip(state.channels, r.fock.reshape(-1, 3, 3)):
        np.testing.assert_allclose(channel.operator.apply(np.eye(3, dtype=complex)), fock, atol=1e-13, rtol=0)
        assert channel.operator.metadata.symmetry_block is None


@pytest.mark.parametrize('profile', PROFILES)
def test_geometry_identity_does_not_claim_a_physical_kernel_or_projector(profile):
    args, a, _, _ = arguments(profile); args['options'].ks.scf.maximum_iterations = 1
    d = function(args, 'build')(**args); original = function(args)(**run_args(args, a))
    args['functional'] = vq.Functional('hse06')
    screened = function(args)(**run_args(args, a))
    assert screened.receipt.domain_identity == original.receipt.domain_identity
    assert screened.receipt.source_identity != original.receipt.source_identity
    args['options'].ri.absolute_cutoff = 10.
    changed = function(args)(**run_args(args, a))
    assert changed.receipt.domain_identity == screened.receipt.domain_identity
    assert changed.receipt.source_identity != screened.receipt.source_identity
    assert changed.receipt.final_density_identity == screened.receipt.final_density_identity
    expected = d.exchange_vectors; bad = d.exchange_vectors; bad[:] = np.nan
    np.testing.assert_array_equal(d.exchange_vectors, expected)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('grid', [False, True])
@pytest.mark.parametrize('fault', ['pair', 'functional', 'direct-options', 'xc-radius', 'all-exact'])
def test_invalid_sources_refuse_before_payload(profile, grid, fault):
    args, a, _, _ = arguments(profile, grid=grid); a['points'][:] = np.nan
    if fault == 'pair': args['options'].ks.pair_cutoff_bohr = 2.
    if fault == 'functional': args['functional'] = vq.Functional('pbe0')
    if fault == 'direct-options': args['options'] = manta.PeriodicRangeSeparatedKSOptions()
    if fault == 'xc-radius': args['xc_image_cutoff_bohr'] = -1.
    if fault == 'all-exact': args['options'].coulomb_algorithm = args['options'].exchange_algorithm = 'exact'
    message = {'pair': 'identical explicit', 'functional': 'range-separated', 'direct-options': 'incompatible function',
               'xc-radius': 'nonnegative', 'all-exact': 'at least one RI'}[fault]
    for verb, kw in [('plan', args), ('run', run_args(args, a))]:
        with pytest.raises((ValueError, RuntimeError, TypeError), match=message): function(args, verb)(**kw)


@pytest.mark.parametrize('profile', ['K', 'JK'])
@pytest.mark.parametrize('grid', [False, True])
def test_empty_offdiagonal_fit_is_not_replaced_by_a_different_source(profile, grid):
    args, a, _, _ = arguments(profile, grid=grid); args['reciprocal_energy_cutoff'] = 1e-6
    a['points'][:] = np.nan
    for verb, kw in [('plan', args), ('run', run_args(args, a))]:
        with pytest.raises(ValueError, match='nonempty mesh/domains'): function(args, verb)(**kw)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
def test_generated_multicenter_images_use_shared_density_push_operator_pull(profile, spins):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    args, a, k, _ = arguments(profile, spins, True, MIXED_RANGE); args['options'].ks.scf.maximum_iterations = 3
    charges = (2, 1, 1) if spins == 1 else (1, 1, 1)
    atoms = [vq.Atom(z, xyz) for z, xyz in zip(charges, ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1]))]
    system = vq.PeriodicSystem(3, args['system'].lattice, atoms); system.multiplicity = spins
    shells, auxiliary = args['basis'].shells(), args['auxiliary'].shells()
    for i, shell in enumerate(shells): shell.atom_index = i; shell.origin = atoms[i].xyz
    for shell in auxiliary: shell.origin = atoms[0].xyz
    args.update(system=system, basis=vq.BasisSet(system.unit_cell_molecule(), shells, 'generated-ks', True),
                auxiliary=vq.BasisSet(system.unit_cell_molecule(), auxiliary, 'generated-ks-aux', True))
    original = function(args)(**args)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])@system.lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    for i, (shell, delta) in enumerate(zip(auxiliary, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    for field, values in (('basis', shells), ('auxiliary', auxiliary)):
        images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip(charges, values)])
        images.multiplicity = spins
        args[field] = vq.BasisSet(images.unit_cell_molecule(), values, 'generated-ks-images', True)
    moved = function(args)(**args); phase = np.exp(1j*k@shifts.T)
    transform = phase[:, :, None]*phase[:, None, :].conj()
    for name in ('density', 'fock', 'overlap', 'hcore'):
        np.testing.assert_allclose(getattr(moved, name), getattr(original, name)*transform, atol=2e-10, rtol=0)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=1e-10)
    assert moved.receipt.quadrature.source_identity == original.receipt.quadrature.source_identity
    assert moved.receipt.domain_identity != original.receipt.domain_identity
    assert np.max(abs(moved.density.imag)) > .01
    for ik in range(len(k)):
        source = SpaceIdentity('generated-ri-ks-test', original.receipt.overlap_identity, 'AO', f'original-k{ik}')
        destination = SpaceIdentity('generated-ri-ks-test', moved.receipt.overlap_identity, 'AO', f'shifted-k{ik}')
        action = BlockSpaceAction(source, destination, np.arange(4, dtype=np.int64), np.arange(3, dtype=np.int64),
            np.ascontiguousarray(phase[ik]), antiunitary=False, budget=Budget(2**20, 10**6))
        for spin in range(spins):
            np.testing.assert_allclose(action.push_density(original.density[spin, ik]), moved.density[spin, ik], atol=2e-10, rtol=0)
            np.testing.assert_allclose(action.pull_operator(moved.fock[spin, ik]), original.fock[spin, ik], atol=2e-10, rtol=0)
    state = as_fock_build(moved, k)
    assert state.energy.total == pytest.approx(original.receipt.energy, abs=1e-10)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
def test_generated_rectangular_overlap_frame_keeps_full_method(profile, spins):
    args, a, k, old = arguments(profile, spins, True, MIXED_RANGE); args['options'].ks.scf.maximum_iterations = 3
    shells = args['basis'].shells(); shells.append(shells[0])
    args['basis'] = a['basis'] = old['basis'] = vq.BasisSet(args['system'].unit_cell_molecule(), shells, 'generated-ks-rank', True)
    explicit(args, a, old); r = function(args)(**args); _, _, build = reference(a, k, old)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r), build(r.density)[1], atol=8e-11, rtol=0)
    assert r.coefficients.shape == (spins, len(k), 4, 3)
    state = as_fock_build(r, k)
    for channel in state.channels:
        assert channel.operator.frame.X.shape == (4, 3)
    assert state.energy.total == pytest.approx(r.receipt.energy, abs=1e-13)
