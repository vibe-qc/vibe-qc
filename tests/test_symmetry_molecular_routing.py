"""Cost routing preserves the exact finite mask, channels and physical state."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from vibeqc.manta import ReferenceJKBuilder, KernelRequest
from vibeqc.symmetry_quartets import qualify_manta_quartets, _MantaSource
from vibeqc.symmetry_quartets_screening import QuartetScreening
from vibeqc.symmetry_quartets_ordered import ordered_quartet_jk, qualified_molecular_source
from vibeqc.symmetry_molecular_hf import ScreenedMolecularJK, _select_route
from vibeqc.symmetry_shared import Budget
from tests.test_symmetry_molecular_hf import BUDGET, method
from tests.test_symmetry_quartets import molecular_case, masked_source, dense_jk


def case(kind='water', schedule='streaming'):
    mol, basis, rotations, translations = molecular_case(kind)
    provider = ReferenceJKBuilder(basis)
    plan = qualify_manta_quartets(mol, basis, rotations, translations,
        provider=provider, source_tolerance=1e-10, budget=BUDGET,
        schedule=schedule, maximum_omitted_norm=.3 if schedule == 'streaming' else None)
    screen = (plan.screening if schedule == 'streaming' else
        QuartetScreening(plan, maximum_omitted_norm=.3, budget=BUDGET))
    return screen, provider, mol, basis


@pytest.mark.parametrize('kind', ['water', 'ammonia', 'cartesian'])
@pytest.mark.parametrize('schedule', ['stored', 'streaming'])
def test_ordered_source_complex_channels_counts_and_physical_fock(kind, schedule):
    screen, provider, mol, basis = case(kind, schedule)
    _, masked = masked_source(screen._plan, provider, screen)
    rng = np.random.default_rng(1231)
    ds = rng.normal(size=(3, basis.nbasis, basis.nbasis)).astype(complex)
    ds += 1j*rng.normal(size=ds.shape)
    result = ordered_quartet_jk(screen, ds, targets=('j', 'k', 'jk'), budget=BUDGET)
    for actual, channel in zip(result.j, (0, 2)):
        np.testing.assert_allclose(actual, dense_jk(masked, ds[channel])[0], atol=2e-11, rtol=0)
    for actual, channel in zip(result.k, (1, 2)):
        np.testing.assert_allclose(actual, dense_jk(masked, ds[channel])[1], atol=2e-11, rtol=0)
    assert result.work.shell_calls == basis.nshells**4-screen.omitted_ordered_tiles
    assert result.scalar_products == 4*(basis.nbasis**4-screen.omitted_integral_values)
    assert result.transport_scalar_products == 0
    assert result.source_identity == screen.source_identity
    states = []
    for executor in ('native', 'ordered', 'auto'):
        jk = ScreenedMolecularJK(screen, budget=BUDGET, executor=executor)
        hf, _, _ = method(jk, mol, basis, retained=True)
        states.append(hf.evaluate(tuple(.01*d.real@d.real.T for d in ds[:2]), method='UHF', generation=3))
    for other in states[1:]:
        assert other.build.source.identity == states[0].build.source.identity
        assert abs(other.build.energy.total-states[0].build.energy.total) < 2e-11
        for a, b in zip(other.build.channels, states[0].build.channels):
            np.testing.assert_allclose(a.operator.matrix, b.operator.matrix, atol=2e-11, rtol=0)
        assert other.receipt.route.qualification.policy_identity == screen.policy_identity


def test_measured_calibration_separate_work_and_bounded_signature_cache():
    screen, _, _, basis = case('ammonia')
    jk = ScreenedMolecularJK(screen, budget=BUDGET, executor='auto')
    route = jk.calibrate_routes(targets=('j', 'k', 'k'), expected_builds=1000)
    assert jk.totals.builds == 0 and jk.last_build is None
    assert len(route.samples) == 4 and route.calibration_seconds > 0
    assert route.logical_bytes > max(s.logical_bytes for s in route.samples)
    assert route.qualification.qualification_scope == 'complete-source'
    assert route.maximum_difference < 1e-10
    assert all(s.route is None for s in route.samples)
    assert tuple(s.executor for s in route.samples) == ('ordered', 'native', 'native', 'ordered')
    ordered, reduced = route.samples[:2]
    assert ordered.work.shell_calls > reduced.work.shell_calls
    assert ordered.scalar_products == reduced.scalar_products
    ds = np.random.default_rng(171).normal(size=(2, basis.nbasis, basis.nbasis))
    jk.build_jk_uhf(*ds)
    assert jk.last_build.route is route and jk.totals.builds == 1
    # Other channels must not borrow unrelated timings.
    jk.build_J(ds[0])
    assert jk.last_build.route.selected == 'ordered'
    replacement = jk.calibrate_routes(targets=('j',), expected_builds=1, repetitions=1)
    assert replacement.selected == 'ordered'
    assert jk.route_calibrations == ((('j', 'k', 'k'), route), (('j',), replacement))
    jk.build_jk_uhf(*ds)
    assert jk.last_build.route is route
    for signature in (('k',), ('jk',), ('k', 'k')):
        jk.calibrate_routes(targets=signature, expected_builds=1, repetitions=1)
    assert len(jk.route_calibrations) == 5
    with pytest.raises(ValueError, match='signature'):
        jk.calibrate_routes(targets=('jk', 'jk'), expected_builds=1)
    json.dumps(asdict(jk.last_build), allow_nan=False)


def test_cost_policy_uses_measured_interval_and_pays_for_probes():
    assert _select_route([10., 12.], [1., 2.], 25., 10)[0] == 'native'
    assert _select_route([10., 12.], [1., 2.], 25., 1)[0] == 'ordered'
    assert _select_route([1., 12.], [1., 2.], 25., 100)[0] == 'ordered'
    assert _select_route([1., 1.], [1., 1.], 1., 100)[0] == 'ordered'


def test_calibration_refuses_before_work_and_rejects_disagreeing_executor(monkeypatch):
    screen, _, _, basis = case()
    reference = ScreenedMolecularJK(screen, budget=BUDGET, executor='auto')
    required = reference.requirements(('jk',))
    jk = ScreenedMolecularJK(screen, budget=required, executor='auto')
    with pytest.raises(MemoryError):
        jk.calibrate_routes(targets=('jk',), expected_builds=10)
    assert jk.totals.builds == 0 and jk.route_calibrations == ()
    for bad in (0, -1, True):
        with pytest.raises((TypeError, ValueError)):
            reference.calibrate_routes(targets=('jk',), expected_builds=bad)
    execute = reference._execute
    def corrupt(ds, targets, executor):
        j, k, receipt = execute(ds, targets, executor)
        if executor == 'native':
            j += .01
        return j, k, receipt
    monkeypatch.setattr(reference, '_execute', corrupt)
    with pytest.raises(ValueError, match='retained source'):
        reference.calibrate_routes(targets=('jk',), expected_builds=10)
    assert reference.route_calibrations == () and reference.totals.builds == 0
    for value in (np.eye(basis.nbasis).astype(complex), np.eye(basis.nbasis).astype(np.float32)):
        with pytest.raises(TypeError):
            jk.build_J(value)


def test_qualification_rejects_declarations_kernels_and_periodic_images():
    screen, provider, _, basis = case()
    with pytest.raises(TypeError, match='certificate'):
        qualified_molecular_source(asdict(qualified_molecular_source(screen)))
    for field, value in [('kernel', 'erf'), ('scalar', 'complex'), ('derivative', 1),
                         ('images', [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])]:
        request = KernelRequest()
        setattr(request, field, value)
        with pytest.raises((ValueError, RuntimeError)):
            ReferenceJKBuilder(basis, request)
    class WrongReceipt:
        def evaluate(self, key):
            receipt = provider.evaluate(key)
            receipt['kernel'] = 'erf'
            return receipt
    source = _MantaSource(provider, screen._plan.offsets, WrongReceipt())
    with pytest.raises(ValueError, match='kernel'):
        source.evaluate((0, 0, 0, 0))
