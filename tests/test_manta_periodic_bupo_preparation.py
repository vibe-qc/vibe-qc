"""Staged native BUPO source ownership, cumulative admission and HF parity."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_bupo_domains import fixture
from tests.test_manta_periodic_pairs import options as pair_options
from tests.test_manta_periodic_bubbles import bubble_options
from tests.test_manta_periodic_bupo import options as bupo_options
from tests.test_manta_periodic_bupo_hf import as_fock_build


def preparation_args(a, c, cutoff=500.):
    source = manta.PeriodicBupoHFSourceOptions()
    source.pairs = pair_options(0, c['cutoff'], .2)
    source.bubbles = bubble_options()
    source.bupo = bupo_options(0.)
    source.bupo.bound_infinite_domain = source.bupo.bound_pair_images = True
    source.pair_cube_radius = int(np.max(abs(c['images'])))
    source.real_cube_radius = c['real_radius']
    source.reciprocal_cube_radius = c['reciprocal_radius']
    source.near_cube_radius = 0
    source.eta = c['eta']
    return dict(basis=c['basis'], system=a['system'], mesh=a['mesh'], reciprocal_energy_cutoff=cutoff,
                source_options=source, options=a['options'], budget=a['budget'])


def test_original_source_identities_and_cumulative_admission():
    a, c = fixture(iterations=1)
    args = preparation_args(a, c, 3.17)
    prepared = manta.prepare_periodic_bupo_hf(**args)
    p, r = prepared.plan, prepared.receipt
    assert r.pair_source_identity == c['pairs'].receipt.source_identity
    assert r.electronic_source_identity == a['electronic'].source_identity
    assert p.source_live_bytes == p.generator_live_bytes + sum(
        stage.modeled_peak_bytes for stage in (p.pairs, p.bubbles, p.ewald, p.bupo))
    assert p.source_shell_calls == sum(stage.shell_calls for stage in (p.pairs, p.bubbles, p.ewald, p.bupo))
    assert p.source_primitive_pairs == p.pairs.primitive_pairs + p.bubbles.primitive_pairs + p.ewald.primitive_products + p.bupo.primitive_products
    assert p.source_scalar_terms == p.generator_scalar_terms + sum(stage.scalar_terms for stage in (p.pairs, p.bubbles, p.ewald, p.bupo))
    assert p.source_kernel_terms == p.generator_kernel_terms + sum(stage.kernel_terms for stage in (p.pairs, p.bubbles, p.ewald, p.bupo))
    for name in ('shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms'):
        assert getattr(p, name) == getattr(p, 'source_'+name) + getattr(p.exchange_domains, name)
    limits = dict(memory_bytes=p.modeled_peak_bytes, shell_calls=p.shell_calls,
                  primitive_quartets=p.primitive_pairs, scalar_terms=p.scalar_terms, kernel_terms=p.kernel_terms)
    b = args['budget']
    for name, value in limits.items():
        setattr(b, name, value)
    assert manta.prepare_periodic_bupo_hf(**args).receipt.identity == r.identity
    for name, value in limits.items():
        setattr(b, name, value-1)
        with pytest.raises((ValueError, OverflowError), match='budget'):
            manta.prepare_periodic_bupo_hf(**args)
        setattr(b, name, value)
    assert prepared.options.exchange_bound_cube_radius > 0 and args['options'].exchange_bound_cube_radius == 0
    with pytest.raises(AttributeError):
        r.identity = 'invalid'
    with pytest.raises(AttributeError):
        p.source_live_bytes = 0


def test_nonzero_order_and_admitted_far_partition_identity():
    from tests.test_manta_periodic_bupo_hf import mixed_case
    a, c = mixed_case(iterations=1)
    a['options'].bound_exchange = True
    args = preparation_args(a, c, 3.17)
    args['source_options'].pairs = pair_options(8, .2, .16)
    args['source_options'].bupo.integral_tolerance = 1e-9
    prepared = manta.prepare_periodic_bupo_hf(**args)
    assert prepared.receipt.electronic_source_identity == a['electronic'].source_identity
    assert prepared.receipt.pair_source_identity == c['pairs'].receipt.source_identity
    assert prepared.receipt.order == 8 and prepared.receipt.integral_tolerance == 1e-9


@pytest.mark.parametrize('field', ['memory_bytes', 'shell_calls', 'primitive_quartets', 'scalar_terms', 'kernel_terms'])
def test_prefix_admission(field):
    a, c = fixture(iterations=1)
    args = preparation_args(a, c, 3.17)
    setattr(args['budget'], field, 0)
    with pytest.raises((ValueError, OverflowError), match='budget'):
        manta.prepare_periodic_bupo_hf(**args)


@pytest.mark.parametrize('field,value,match', [('eta', 0., 'positive eta'),
    ('near_cube_radius', 5, 'near cube'), ('pair_cube_radius', 2**31, 'integer extent')])
def test_invalid_source_domains(field, value, match):
    a, c = fixture(iterations=1)
    args = preparation_args(a, c)
    setattr(args['source_options'], field, value)
    with pytest.raises(ValueError, match=match):
        manta.prepare_periodic_bupo_hf(**args)


@pytest.mark.parametrize('field', ['bound_infinite_domain', 'bound_pair_images'])
def test_source_certification_is_required(field):
    a, c = fixture(iterations=1)
    args = preparation_args(a, c)
    setattr(args['source_options'].bupo, field, False)
    with pytest.raises(ValueError, match='all source/exchange bounds'):
        manta.prepare_periodic_bupo_hf(**args)


def test_physical_input_validation_before_source_construction():
    a, c = fixture(iterations=1)
    args = preparation_args(a, c)
    args['system'].charge = 1
    with pytest.raises(ValueError, match='neutral all-electron'):
        manta.prepare_periodic_bupo_hf(**args)
    args['system'].charge = 0
    args['system'].unit_cell = [vq.Atom(2, [1., 0., 0.])]
    with pytest.raises(ValueError, match='basis centers'):
        manta.prepare_periodic_bupo_hf(**args)
    args['system'].unit_cell = [vq.Atom(2, [.1, .2, -.1])]
    args['options'].maximum_exchange_energy_error = float('nan')
    with pytest.raises(ValueError, match='finite positive'):
        manta.prepare_periodic_bupo_hf(**args)


def test_captured_inputs_repeated_states_and_one_active_run():
    a, c = fixture(iterations=1)
    args = preparation_args(a, c)
    prepared = manta.prepare_periodic_bupo_hf(**args)
    control = core._manta.run_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=500.)
    # Mutating both original objects and exported policy copies cannot alter
    # the physical producer or allow an unadmitted retained-state cache.
    args['system'].lattice = np.eye(3)*19.
    args['system'].unit_cell = []
    args['options'].spin_mode = 'unrestricted'
    args['source_options'].eta = 9.
    args['budget'].memory_bytes = 0
    prepared.options.scf.maximum_iterations = 99
    prepared.source_options.eta = 7.
    assert prepared.options.scf.maximum_iterations == 1 and prepared.source_options.eta == .6
    first = prepared.run()
    for name in ('density', 'fock', 'hcore', 'overlap'):
        np.testing.assert_array_equal(getattr(first, name), getattr(control, name))
    assert first.receipt.energy == control.receipt.energy
    barrier = threading.Barrier(2)
    def run():
        barrier.wait()
        try:
            return prepared.run()
        except RuntimeError as error:
            assert 'already has an active run' in str(error)
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run(), range(2)))
    assert sum(result is None for result in results) == 1
    second = next(result for result in results if result is not None)
    np.testing.assert_array_equal(first.fock, second.fock)
    assert first.receipt.final_density_identity == second.receipt.final_density_identity
    assert first.receipt.preparation.identity == prepared.receipt.identity
    assert first.plan.source_live_bytes == prepared.plan.source_live_bytes
    assert first.plan.modeled_peak_bytes <= prepared.plan.modeled_peak_bytes
    state = as_fock_build(second, c['k'])
    assert state.energy.total == pytest.approx(second.receipt.energy, abs=3e-12)


def test_selected_provenance_and_early_ecp_refusal(monkeypatch):
    from vibeqc import output, periodic_runner
    a, c = fixture(iterations=1)
    args = preparation_args(a, c)
    budget = args.pop('budget')
    events = []
    monkeypatch.setattr(output, 'record', lambda *args, **kw: events.append(kw['integral_provenance']))
    args['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    result = manta.run_periodic_bupo_hf_from_cubes(**args)
    evidence = events[-1]
    json.dumps(evidence, allow_nan=False)
    assert evidence['preparation']['identity'] == result.receipt.preparation.identity
    assert evidence['preparation']['electronic_source_identity'] == a['electronic'].source_identity
    assert evidence['domain_selection'] == 'generated-bupo-cubes;certified-reciprocal-energy-sphere'
    assert evidence['source_live_bytes'] > 0 and evidence['source_kernel_terms'] > 0
    assert not evidence['symmetry_reduced'] and budget.other_live_bytes == 0
    assert args['options'].exchange_bound_cube_radius == 0
    monkeypatch.setattr(periodic_runner, '_resolve_ecp_data', lambda *args: (None, None, None, True))
    with pytest.raises(NotImplementedError, match='all-electron'):
        manta.run_periodic_bupo_hf_from_cubes(**args)


def test_small_generated_exchange_support_fails_accuracy():
    a, c = fixture(iterations=1)
    prepared = manta.prepare_periodic_bupo_hf(**preparation_args(a, c, 3.))
    with pytest.raises(ValueError, match='exchange accuracy target'):
        prepared.run()
    # A failed run releases its lease, without promoting a rejected state.
    with pytest.raises(ValueError, match='exchange accuracy target'):
        prepared.run()


def source_args(a, c):
    args = preparation_args(a, c)
    return {name: args[name] for name in ('basis', 'system', 'mesh', 'source_options', 'budget')}


@pytest.mark.parametrize('shifted,skew,left', [(False, False, False), (True, True, False), (True, True, True)])
def test_shared_electronic_factory_preserves_native_frames_and_responses(shifted, skew, left):
    a, c = fixture(iterations=1, shifted=shifted, skew=skew, left=left)
    prepared = manta.prepare_periodic_bupo_source(**source_args(a, c))
    assert prepared.receipt.electronic_source_identity == a['electronic'].source_identity
    assert prepared.receipt.pair_source_identity == c['pairs'].receipt.source_identity
    assert not hasattr(prepared.plan, 'exchange_domains') and not hasattr(prepared, 'run')
    n, nk = c['basis'].nbasis, len(a['mesh'])
    rng = np.random.default_rng(807)
    z = rng.normal(size=(nk, n, n)) + 1j*rng.normal(size=(nk, n, n))
    d = np.ascontiguousarray((z+z.conj().transpose(0, 2, 1))*.03)
    control = a['electronic'].build(d)
    actual = prepared.source.build(d)
    for name in ('values', 'j_error_bound', 'domain_j_error_bound', 'pair_image_j_error_bound'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(control, name))
    assert actual.receipt.energy == control.receipt.energy
    assert actual.receipt.source_identity == control.receipt.source_identity


@pytest.mark.parametrize('field,quantity', [('memory_bytes', 'modeled_peak_bytes'),
    ('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
    ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_shared_source_stage_caps_and_remaining_budget(field, quantity):
    a, c = fixture(iterations=1)
    args = source_args(a, c)
    args['budget'].other_live_bytes = 12345
    prepared = manta.prepare_periodic_bupo_source(**args)
    p, remaining = prepared.plan, prepared.remaining_budget
    assert remaining.other_live_bytes == args['budget'].other_live_bytes + p.source_live_bytes
    assert p.modeled_peak_bytes == remaining.other_live_bytes
    for cap, work in [('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
                      ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')]:
        assert getattr(remaining, cap) == getattr(args['budget'], cap) - getattr(p, work)
    value = getattr(p, quantity)
    setattr(args['budget'], field, value-1)
    with pytest.raises((ValueError, OverflowError), match='budget'):
        manta.prepare_periodic_bupo_source(**args)
    setattr(args['budget'], field, value)
    exact = manta.prepare_periodic_bupo_source(**args)
    assert exact.receipt.identity == prepared.receipt.identity
    exact.remaining_budget.memory_bytes = 0
    assert exact.remaining_budget.memory_bytes > 0


def test_electronic_factory_owns_source_after_caller_mutation_and_handle_deletion():
    import gc
    a, c = fixture(iterations=1)
    args = source_args(a, c)
    prepared = manta.prepare_periodic_bupo_source(**args)
    source = prepared.source
    identity = source.source_identity
    args['system'].charge = 12
    args['system'].multiplicity = 99
    # Neither charge nor spin is an electronic-source variable. Method
    # factories still reject invalid occupations before their own work.
    assert manta.prepare_periodic_bupo_source(**args).source.source_identity == identity
    args['system'].lattice = np.eye(3)*20
    args['source_options'].eta = 9
    args['budget'].memory_bytes = 0
    prepared.source_options.pairs.order = 6
    assert prepared.source_options.pairs.order == 0
    del prepared
    gc.collect()
    d = np.ascontiguousarray(np.ones((len(a['mesh']), c['basis'].nbasis, c['basis'].nbasis), complex)*.01)
    np.testing.assert_array_equal(source.build(d).values, a['electronic'].build(d).values)
    assert source.source_identity == identity
    assert manta.PeriodicBupoHFSourceOptions is manta.PeriodicBupoSourceOptions
    assert manta.PeriodicBupoHFPreparationReceipt is manta.PeriodicBupoSourcePreparationReceipt


@pytest.mark.parametrize('field', ['bound_infinite_domain', 'bound_pair_images'])
def test_shared_source_requires_full_coulomb_bounds(field):
    a, c = fixture(iterations=1)
    args = source_args(a, c)
    setattr(args['source_options'].bupo, field, False)
    with pytest.raises(ValueError, match='all source bounds'):
        manta.prepare_periodic_bupo_source(**args)


def test_shared_source_rejects_empty_batch():
    a, c = fixture(iterations=1)
    with pytest.raises(ValueError, match='positive reciprocal batch'):
        manta.prepare_periodic_bupo_source(**source_args(a, c), reciprocal_batch=0)
