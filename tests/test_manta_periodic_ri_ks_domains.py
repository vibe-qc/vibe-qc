"""Fitted KS generated sources: independent supports, functional SCF and bounds."""
import itertools
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_ri_ks import case, run as explicit_run, reference, CAPS, as_fock_build
from tests.test_manta_periodic_global_hybrid_ks import components, EMPTY_HYBRID
from tests.test_manta_periodic_ks import independent_scf
from tests.test_manta_periodic_domains import _labels

PROFILES = ['pure', 'J', 'K', 'JK']
ARRAYS = ('density', 'fock', 'overlap', 'hcore', 'coefficients', 'orbital_energies')


def arguments(profile='pure', spins=1, grid=False, mesh=(2, 1, 1), shift=(1, 0, 0), species=None):
    name = ('blyp' if profile == 'pure' else EMPTY_HYBRID) if species == 'H' else None
    a, k, old = case(profile, spins, name=name, mesh=mesh, shift=shift, species=species)
    args = {key: a[key] for key in ('basis', 'auxiliary', 'system', 'mesh', 'functional', 'options', 'budget')}
    args.update(reciprocal_energy_cutoff=6., xc_image_cutoff_bohr=11.)
    if grid:
        q = manta.PeriodicQuadratureOptions(); q.radial_points = 12; q.lebedev_order = 17
        q.partition_radius_bohr = 8.; args['grid_options'] = q
    else:
        args['grid_points'] = len(a['points'])
    return args, a, k, old


def function(args, verb='run', namespace=core._manta):
    profile = 'ri_ks' if isinstance(args['options'], manta.PeriodicRIKSOptions) else 'ri_global_hybrid_ks'
    suffix = 'domains' if verb == 'build' else ('with_quadrature' if 'grid_options' in args else 'from_cutoffs')
    return getattr(namespace, verb+'_periodic_'+profile+'_'+suffix)


def run_args(args, a):
    if 'grid_options' in args: return dict(args)
    return dict({key: value for key, value in args.items() if key != 'grid_points'},
                points=a['points'], quadrature_weights=a['quadrature_weights'])


def explicit(args, a, old):
    da = dict(args); grid = None
    if 'grid_options' in args:
        grid = manta.build_periodic_quadrature(args['system'], args['grid_options'], args['budget'])
        da.pop('grid_options'); da['grid_points'] = grid.plan.grid_points
        a.update(points=grid.points, quadrature_weights=grid.weights)
    domains = function(da, 'build')(**da)
    for name in ('translations', 'reciprocal_vectors', 'xc_images', 'exchange_vectors', 'transfer_offsets'):
        if hasattr(domains, name): a[name] = getattr(domains, name)
    old.update(reciprocal_vectors=a['reciprocal_vectors'], translations=a['translations'])
    return domains, grid


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)), ((3, 2, 1), (1, 1, 0))])
def test_independent_integer_spheres_and_component_orientation(profile, shape, shift):
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
        actual = _labels(getattr(d, field), np.linalg.inv(lattice))
        assert actual == {tuple(n) for n in labels[np.sum(images*images, axis=1) <= radius**2]}
        assert actual == {tuple(-np.array(n)) for n in actual}
    if profile == 'pure':
        assert not hasattr(d, 'exchange_vectors') and not hasattr(d.plan, 'exchange_candidates')
        assert not d.plan.ks.fitting.has_k and d.plan.ks.fitting.has_j
        assert d.receipt.conjugacy_vectors == len(d.reciprocal_vectors)
        return
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
            # Both finite sources are closed under physical conjugacy.
            assert {tuple(np.round(x, 12)) for x in actual} == {tuple(np.round(-x, 12)) for x in d.exchange_vectors[revlo:revhi]}
    assert len(d.exchange_vectors) <= d.plan.exchange_candidates
    assert max(np.diff(d.transfer_offsets)) <= d.plan.maximum_exchange_candidates


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('grid', [False, True])
@pytest.mark.parametrize('gamma', [False, True])
def test_generated_scf_matches_explicit_and_independent_functional(profile, spins, grid, gamma):
    kw = dict(mesh=(1, 1, 1), shift=(0, 0, 0)) if gamma else {}
    args, a, k, old = arguments(profile, spins, grid, **kw)
    d, q = explicit(args, a, old); p = function(args, 'plan')(**args)
    r = function(args)(**run_args(args, a)); other = explicit_run(a)
    assert r.converged and r.receipt.confirmed
    for name in ARRAYS: np.testing.assert_array_equal(getattr(r, name), getattr(other, name))
    assert r.receipt.energy == other.receipt.energy
    assert r.receipt.source_identity == other.receipt.source_identity
    assert r.receipt.final_density_identity == other.receipt.final_density_identity
    assert r.receipt.final_fock_identity == other.receipt.final_fock_identity
    assert r.receipt.domain_identity == d.receipt.identity
    assert r.plan.domain_live_bytes == d.plan.domain_live_bytes
    for _, quantity in CAPS: assert getattr(r.plan, quantity) <= getattr(p, quantity)
    if grid:
        assert r.receipt.quadrature.source_identity == q.receipt.source_identity
        assert r.plan.quadrature_live_bytes == p.grid_live_bytes
        assert p.scalar_terms == p.quadrature.scalar_terms+p.domains.scalar_terms
        assert p.kernel_terms == p.quadrature.kernel_terms+p.domains.kernel_terms
    else:
        assert r.receipt.quadrature is None and r.plan.quadrature is None
    s, h, build = reference(a, k, old)
    populations = (r.plan.n_alpha,) if spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
    density = independent_scf(s, h, build, populations, spins)
    np.testing.assert_allclose(r.density, density, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.overlap, s, atol=3e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=8e-12, rtol=0)
    f, energies, counts = build(r.density)
    np.testing.assert_allclose(r.fock, f, atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r), energies, atol=8e-11, rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid, counts, atol=8e-11, rtol=0)
    assert r.receipt.energy == pytest.approx(r.receipt.nuclear_energy+sum(build(density)[1]), abs=8e-11)
    assert len(r.receipt.fitting_info.metrics) == (1+len(k)*(len(k)-1) if profile in ('K', 'JK') else 1)


@pytest.mark.parametrize('profile', PROFILES)
def test_generated_empty_spin_retains_full_matrix(profile):
    args, a, k, old = arguments(profile, 2, True, species='H'); explicit(args, a, old)
    r = function(args)(**args); _, _, build = reference(a, k, old)
    assert r.converged and r.plan.n_beta == 0
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r), build(r.density)[1], atol=8e-11, rtol=0)


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('grid', [False, True])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_all_joint_caps_and_refusal_before_payload(profile, spins, grid, field, quantity):
    args, a, _, _ = arguments(profile, spins, grid); args['options'].ks.scf.maximum_iterations = 2
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
def test_selected_caps_include_output_and_keep_caller_budget(profile, grid, field, quantity, monkeypatch):
    from vibeqc import output
    from vibeqc.manta_periodic import _periodic_ri_ks_configuration
    args, a, _, _ = arguments(profile, 2, grid); args['options'].ks.scf.maximum_iterations = 2
    p = function(args, 'plan')(**args); budget = args['budget']
    engine = manta.IntegralEngineOptions(backend='manta', budget=budget)
    _, _, python_bytes = _periodic_ri_ks_configuration(args['options'], engine, args['mesh'], profile != 'pure')
    kw = run_args(args, a); kw.pop('budget'); kw['integral_engine'] = engine
    calls = []; monkeypatch.setattr(output, 'record', lambda *a, **kw: calls.append(kw))
    limit = getattr(p, quantity)+(python_bytes if field == 'memory_bytes' else 0)
    setattr(budget, field, limit); selected = function(args, namespace=manta)
    selected(**kw); assert len(calls) == 1 and budget.other_live_bytes == 0
    setattr(budget, field, limit-1); a['points'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): selected(**kw)
    assert len(calls) == 1


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('rank_cutoff', [10., 1e9])
def test_capped_fixed_projectors_and_functional_energy(profile, spins, rank_cutoff):
    args, a, k, old = arguments(profile, spins, True); args['options'].ri.absolute_cutoff = rank_cutoff
    args['options'].ks.scf.maximum_iterations = 3; explicit(args, a, old)
    r = function(args)(**args); _, _, build = reference(a, k, old)
    assert r.receipt.fock_calls == 3 and not r.converged
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(r), build(r.density)[1], atol=8e-11, rtol=0)
    # The XC functional energy must remain distinct from an HF half trace.
    wrong = r.receipt.nuclear_energy+.5*np.einsum('skij,skji->', r.density, r.hcore+r.fock).real/len(k)
    assert abs(wrong-r.receipt.energy) > .01


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('grid', [False, True])
def test_admission_before_invalid_numeric_source_generation(profile, grid):
    args, a, _, _ = arguments(profile, grid=grid, mesh=(1, 1, 1), shift=(0, 0, 0))
    args['options'].ks.scf.maximum_iterations = 1
    args['reciprocal_energy_cutoff'] = 1e-6
    p = function(args, 'plan')(**args)
    with pytest.raises(ValueError, match='no nonzero Hartree'): function(args)(**run_args(args, a))
    args['budget'].shell_calls = p.shell_calls-1
    with pytest.raises((ValueError, RuntimeError), match='budget'): function(args)(**run_args(args, a))


@pytest.mark.parametrize('profile', PROFILES)
def test_joint_admission_before_invalid_quadrature_partition(profile):
    args, _, _, _ = arguments(profile, grid=True); s = args['system']; atom = vq.Atom(2, s.unit_cell[0].xyz)
    args['system'] = vq.PeriodicSystem(3, s.lattice, [atom, atom])
    for field in ('basis', 'auxiliary'):
        shells = args[field].shells()
        for shell in shells: shell.origin = atom.xyz
        args[field] = vq.BasisSet(args['system'].unit_cell_molecule(), shells, 'joint-fitting-admission', True)
    with pytest.raises(ValueError, match='Coincident'):
        manta.build_periodic_quadrature(args['system'], args['grid_options'], args['budget'])
    args['budget'].shell_calls = 0
    with pytest.raises((ValueError, RuntimeError), match='budget'): function(args)(**args)


@pytest.mark.parametrize('profile', PROFILES)
def test_geometry_identity_is_distinct_from_fitted_physical_source_and_exports_own_data(profile):
    args, a, _, _ = arguments(profile); args['options'].ks.scf.maximum_iterations = 1
    d = function(args, 'build')(**args); original = function(args)(**run_args(args, a))
    if profile == 'pure':
        direct = dict(args); direct.pop('auxiliary'); direct['options'] = args['options'].ks
        other = manta.build_periodic_ks_domains(**direct)
        assert other.receipt.identity == d.receipt.identity
    else:
        h = manta.PeriodicRIRHFOptions(); h.pair_cutoff_bohr = args['options'].ks.pair_cutoff_bohr
        h.fitting = args['options'].fitting; h.ri = args['options'].ri
        h.coulomb_algorithm = args['options'].coulomb_algorithm; h.exchange_algorithm = args['options'].exchange_algorithm
        hf = {n: args[n] for n in ('basis', 'auxiliary', 'system', 'mesh', 'reciprocal_energy_cutoff', 'budget')}
        other = manta.build_periodic_ri_rhf_domains(**hf, options=h)
        assert other.receipt.identity == d.receipt.pair_identity
        np.testing.assert_array_equal(other.pair_vectors, d.exchange_vectors)
    args['options'].ri.absolute_cutoff = 10.
    changed = function(args)(**run_args(args, a))
    assert changed.receipt.domain_identity == original.receipt.domain_identity
    assert changed.receipt.source_identity != original.receipt.source_identity
    assert changed.receipt.final_density_identity == original.receipt.final_density_identity
    assert changed.receipt.final_fock_identity != original.receipt.final_fock_identity
    receipt, plan = d.receipt, d.plan; expected = d.translations
    bad = d.translations; bad[:] = np.nan
    np.testing.assert_array_equal(d.translations, expected)
    del d
    assert receipt.identity and plan.ks.fitting is not None


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('grid', [False, True])
def test_selected_scientific_output_and_shared_fock_preserve_full_method(tmp_path, profile, spins, grid):
    from vibeqc.output import OutputChannel, structured_log
    from vibeqc.output.citations import load_default_database
    args, a, k, _ = arguments(profile, spins, grid); args['options'].ks.scf.maximum_iterations = 3
    kw = run_args(args, a); budget = kw.pop('budget')
    kw['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    with OutputChannel.to_file(tmp_path/'science.out'), structured_log(tmp_path/'science.jsonl'):
        r = function(args, namespace=manta)(**kw)
    event = json.loads((tmp_path/'science.jsonl').read_text()); e = event['integral_provenance']
    assert e['energy'] == r.receipt.energy and e['operator_kind'] == 'ks'
    assert not e['symmetry_reduced'] and e['fitted_fock']['component_only']
    assert e['domain_identity'] == r.receipt.domain_identity
    assert (e.get('quadrature') is not None) == grid
    json.dumps(e, allow_nan=False)
    database = load_default_database()
    keys = {citation.key for method in e['citation_methods']
            for citation in database.assemble(method=method, numerics=e['citation_numerics']).citations}
    assert {'sun_berkelbach_gdf_2017', 'ye_berkelbach_rsgdf_2021', 'libxc_2018'} <= keys
    state = as_fock_build(r, k)
    assert state.energy.total == pytest.approx(r.receipt.energy, abs=1e-13)
    for channel, density, fock in zip(state.channels, r.density.reshape(-1, 3, 3), r.fock.reshape(-1, 3, 3)):
        np.testing.assert_array_equal(channel.density, density)
        np.testing.assert_allclose(channel.operator.apply(np.eye(3, dtype=complex)), fock, atol=1e-13, rtol=0)
        assert channel.operator.metadata.kind == 'ks' and channel.operator.metadata.symmetry_block is None


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('spins', [1, 2])
def test_generated_multicenter_images_use_shared_density_push_operator_pull(profile, spins):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    args, a, k, _ = arguments(profile, spins, True); args['options'].ks.scf.maximum_iterations = 3
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
    args, a, k, old = arguments(profile, spins, True); args['options'].ks.scf.maximum_iterations = 3
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


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('grid', [False, True])
@pytest.mark.parametrize('fault', ['pair', 'range', 'meta', 'direct-options', 'xc-radius'])
def test_unqualified_sources_refuse_before_payload(profile, grid, fault):
    args, a, _, _ = arguments(profile, grid=grid); a['points'][:] = np.nan
    if fault == 'pair': args['options'].ks.pair_cutoff_bohr = 2.
    if fault == 'range': args['functional'] = vq.Functional('hse06')
    if fault == 'meta': args['functional'] = vq.Functional('scan')
    if fault == 'direct-options':
        args['options'] = manta.PeriodicKSOptions() if profile == 'pure' else manta.PeriodicGlobalHybridKSOptions()
    if fault == 'xc-radius': args['xc_image_cutoff_bohr'] = -1.
    # Function choice is independent of the intentionally invalid option type.
    prefix = 'ri_ks' if profile == 'pure' else 'ri_global_hybrid_ks'
    suffix = 'with_quadrature' if grid else 'from_cutoffs'
    message = {'pair': 'identical explicit', 'range': 'complete native', 'meta': 'complete native',
               'direct-options': 'incompatible function', 'xc-radius': 'nonnegative'}[fault]
    for verb, kw in [('plan', args), ('run', run_args(args, a))]:
        with pytest.raises((ValueError, RuntimeError, TypeError), match=message):
            getattr(core._manta, verb+'_periodic_'+prefix+'_'+suffix)(**kw)


@pytest.mark.parametrize('profile', ['K', 'JK'])
@pytest.mark.parametrize('grid', [False, True])
def test_count_only_empty_offdiagonal_fitting_box_refuses(profile, grid):
    args, a, _, _ = arguments(profile, grid=grid); args['reciprocal_energy_cutoff'] = 1e-6
    a['points'][:] = np.nan
    for verb, kw in [('plan', args), ('run', run_args(args, a))]:
        with pytest.raises(ValueError, match='nonempty mesh/domains'):
            function(args, verb)(**kw)
