"""Source-qualified RI blocks against native and independent integral oracles."""
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.solvers.operators import OperatorFrame
from vibeqc.symmetry_ao import build_molecular_space_action
from vibeqc.symmetry_ri import (
    prepare_abelian_molecular_ri, MolecularRIJK, _adapt_retained,
)
from vibeqc.symmetry_molecular_hf import ScreenedMolecularHF
from vibeqc.symmetry_shared import Budget, BlockSpaceAction, SpaceIdentity
from tests.test_symmetry_abelian import cyclic
from tests.test_symmetry_quartets import molecular_case

BUDGET = Budget(512 << 20, 10**12)
SPACE = SpaceIdentity('finite test molecule', 'native basis', 'AO', 'native')


def case(kind='ammonia', *, pure=False, auxiliary=None, options=None, displacement=0., factor_export='producer'):
    mol, basis, rotations, translations = molecular_case(kind, displacement=displacement)
    if kind == 'water':
        chars = np.array([[(-1)**((a & b).bit_count()) for b in range(4)] for a in range(4)], complex)
    else:
        rotations, translations = rotations[:3], translations[:3]
        chars = cyclic().characters
    if kind == 'cartesian' and pure:
        basis = vq.BasisSet(mol, [core.ShellInfo(0, l, True, [.8], [1.], [0., 0., 0.])
            for l in (0, 2)], 'pure s+d', False)
    auxiliary = basis if auxiliary is None else auxiliary
    source = prepare_abelian_molecular_ri(mol, basis, auxiliary, rotations, translations,
        chars, source_tolerance=1e-10, budget=BUDGET, ri_options=options, factor_export=factor_export)
    return source, mol, basis, auxiliary, rotations, translations


def dense(factors, density):
    # Independent full AO algebra, without irrep labels or block selection.
    return (np.einsum('Apq,Ars,rs->pq', factors, factors.conj(), density, optimize=False),
            np.einsum('Apr,rs,Aqs->pq', factors, density, factors.conj(), optimize=False))


@pytest.mark.parametrize('kind,pure', [('water', False), ('ammonia', False),
                                        ('cartesian', False), ('cartesian', True)])
def test_factors_complex_density_covariance_and_executed_work(kind, pure):
    source, mol, basis, aux, rotations, translations = case(kind, pure=pure)
    metric = np.asarray(vq.compute_2c_eri(aux))
    coupling = np.asarray(vq.compute_3c_eri(basis, aux))
    np.testing.assert_allclose(source._native.metric, metric, atol=2e-12, rtol=0)
    # Native rank/factor, but an independent libint integral implementation.
    factors = np.einsum('PA,Puv->Auv', source._native.factor, coupling)
    rng = np.random.default_rng(30218)
    trial = rng.normal(size=(basis.nbasis,)*2)+1j*rng.normal(size=(basis.nbasis,)*2)
    for d in (trial, trial+trial.conj().T):
        result = source.jk(d, budget=BUDGET)
        full = source.jk(d, budget=BUDGET, executor='dense')
        legacy = source.jk(d, budget=BUDGET, executor='numpy')
        np.testing.assert_allclose(result.j, legacy.j, atol=3e-12, rtol=0)
        np.testing.assert_allclose(result.k, legacy.k, atol=3e-12, rtol=0)
        assert result.scalar_products == legacy.scalar_products
        for actual, selected, reference in zip((result.j, result.k), (full.j, full.k), dense(factors, d)):
            np.testing.assert_allclose(actual, selected, atol=3e-12, rtol=0)
            np.testing.assert_allclose(actual, reference, atol=3e-11, rtol=0)
        for r, t in zip(rotations, translations):
            action = build_molecular_space_action(mol, basis, r, t,
                source=SPACE, target=SPACE, budget=BUDGET)
            moved = source.jk(action.push_density(d), budget=BUDGET)
            # U^dagger F[U D U^dagger] U = F[D], including nonunitary Cartesian U.
            for actual, original in ((moved.j, result.j), (moved.k, result.k)):
                np.testing.assert_allclose(action.pull_operator(actual), original, atol=4e-11, rtol=0)
        assert result.scalar_products < full.scalar_products
        assert result.transport_scalar_products == 6*basis.nbasis**3
        for target in ('j', 'k'):
            one = source.jk(d, budget=BUDGET, targets=target)
            np.testing.assert_allclose(getattr(one, target), getattr(result, target), atol=3e-12, rtol=0)
    receipt = source.receipt
    assert receipt['coupling_shell_calls'] == aux.nshells*basis.nshells**2
    assert receipt['metric_shell_calls'] == aux.nshells**2
    assert receipt['stored_factor_elements'] < receipt['dense_factor_elements']
    assert receipt['factor_array_bytes'] == 16*receipt['stored_factor_elements']
    assert receipt['native_factor_copy_bytes'] == receipt['factor_array_bytes']
    assert receipt['block_executor'] == 'native-auxiliary-panel-v1'
    assert receipt['metric_rank'] == source._native.metric_info.rank
    assert receipt['tensor_error_bound'] <= 1e-10
    json.dumps(receipt, allow_nan=False)


def test_rank_cut_preserves_complete_space_and_options_are_snapshotted():
    options = core._manta.RIOptions(); options.relative_cutoff = .1
    source, _, basis, *_ = case(options=options)
    assert 0 < source.receipt['metric_rank'] < basis.nbasis
    d = np.eye(basis.nbasis)
    expected = source._native.build(d)
    actual = source.jk(d.astype(complex), budget=BUDGET)
    np.testing.assert_allclose(actual.j, expected.J, atol=3e-11, rtol=0)
    np.testing.assert_allclose(actual.k, expected.K, atol=3e-11, rtol=0)
    identity = source.source_identity
    options.relative_cutoff = .9
    assert source.source_identity == identity
    np.testing.assert_array_equal(source.jk(d.astype(complex), budget=BUDGET).j, actual.j)
    with pytest.raises(ValueError):
        source.blocks[0][1].setflags(write=True)
    receipt = source.receipt; receipt['metric_rank'] = 0
    assert source.receipt['metric_rank'] > 0


def test_degenerate_partners_may_mix_but_cannot_be_split_or_antiunitary():
    irreps = cyclic()
    def action(theta, anti=False):
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], complex)
        return BlockSpaceAction(SPACE, SPACE, np.array([0, 2], np.int64),
            np.array([0], np.int64), r.ravel(), antiunitary=anti, budget=BUDGET)
    actions = [action(2*np.pi*g/3) for g in range(3)]
    gauge, _ = np.linalg.qr(np.random.default_rng(81).normal(size=(2, 2))+
                           1j*np.random.default_rng(94).normal(size=(2, 2)))
    for columns in (np.eye(2), gauge):
        c, labels, evidence, _ = _adapt_retained(irreps, actions, np.eye(2), columns, 1e-10, BUDGET)
        assert sorted(labels) == [1, 2]
        np.testing.assert_allclose(c@c.conj().T, np.eye(2), atol=2e-13, rtol=0)
        assert evidence['maximum_relative_leakage'] < 2e-13
    with pytest.raises(ValueError, match='not closed'):
        _adapt_retained(irreps, actions, np.eye(2), np.eye(2)[:, :1], 1e-10, BUDGET)
    with pytest.raises(ValueError, match='antiunitary'):
        _adapt_retained(irreps, [action(0, True)], np.eye(2), np.eye(2), 1e-10, BUDGET)
    with pytest.raises(ValueError, match='metric is not equivariant'):
        _adapt_retained(irreps, actions, np.diag([1., 2.]), np.diag([1., 2**-.5]), 1e-10, BUDGET)


def test_source_refusals_before_native_work_and_after_geometry():
    mol, basis, rotations, translations = molecular_case()
    def prepare(budget=BUDGET, **kw):
        return prepare_abelian_molecular_ri(mol, basis, basis, rotations[:3], translations[:3],
            cyclic().characters, source_tolerance=1e-10, budget=budget, **kw)
    with pytest.raises(MemoryError):
        prepare(Budget(1, 10**12))
    with pytest.raises((ValueError, MemoryError)):
        prepare(Budget(512 << 20, 1))
    with pytest.raises(TypeError, match='native RIOptions'):
        prepare(ri_options={})
    options = core._manta.RIOptions(); options.absolute_cutoff = 1e10
    with pytest.raises(ValueError, match='empty RI space'):
        prepare(ri_options=options)
    with pytest.raises(ValueError):
        case(displacement=.01)
    with pytest.raises(ValueError):
        # The whole C3v group has no complete table of one-dimensional irreps.
        prepare_abelian_molecular_ri(mol, basis, basis, rotations, translations,
            np.ones((6, 6), complex), source_tolerance=1e-10, budget=BUDGET)


@pytest.mark.parametrize('method', ['RHF', 'UHF'])
def test_native_scf_and_shared_fock_state(method):
    source, mol, basis, *_ = case('water')
    s = np.asarray(vq.compute_overlap(basis))
    h = np.asarray(vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol))
    if method == 'RHF':
        options = vq.RHFOptions(); driver = vq.run_rhf_scf_with_jk
        args = (basis, 10, s, h, mol.nuclear_repulsion())
    else:
        options = vq.UHFOptions(); driver = vq.run_uhf_scf_with_jk
        args = (basis, 5, 5, s, h, mol.nuclear_repulsion())
    options.initial_guess = vq.InitialGuess.HCORE
    options.conv_tol_energy = 1e-11; options.conv_tol_grad = 1e-9
    jk = MolecularRIJK(source, budget=BUDGET)
    result = driver(*args, jk, options, molecule=mol)
    full = driver(*args, source._native, options, molecule=mol)
    assert result.converged and full.converged
    assert abs(result.energy-full.energy) < 1e-10
    values, vectors = np.linalg.eigh(s)
    frame = OperatorFrame(basis_id=source.source_identity, n=basis.nbasis,
        ordering='native AO', image_gauge='molecular', overlap=s, X=vectors/np.sqrt(values))
    hf = ScreenedMolecularHF(jk, frame=frame, hcore=h, nuclear_energy=mol.nuclear_repulsion())
    d = (result.density,) if method == 'RHF' else (result.density_alpha, result.density_beta)
    state = hf.evaluate(d, method=method, generation=result.n_iter)
    assert abs(state.build.energy.total-result.energy) < 1e-10
    factors = np.einsum('PA,Puv->Auv', source._native.factor, vq.compute_3c_eri(basis, basis))
    j = dense(factors, sum(d))[0]
    for density, channel in zip(d, state.build.channels):
        k = dense(factors, density)[1]
        np.testing.assert_allclose(channel.operator.matrix, h+j-(.5 if method == 'RHF' else 1)*k,
                                   atol=3e-11, rtol=0)
    assert jk.totals.work.shell_calls == 0  # fitting setup is reported separately
    assert jk.totals.builds > 2
    with pytest.raises(TypeError, match='real float64'):
        jk.build_J(np.eye(basis.nbasis, dtype=complex))
    with pytest.raises(MemoryError):
        MolecularRIJK(source, budget=Budget(1, 10**12))
    with pytest.raises(AttributeError):
        jk.source = None


def test_non_equivariant_finite_coupling_cannot_borrow_geometry_permission(monkeypatch):
    original = core._manta.FittingTiles
    class Perturbed:
        def __init__(self, *args):
            self.provider = original(*args)
            self.source_identity = self.provider.source_identity
        def evaluate(self, key):
            tile = self.provider.evaluate(key)
            if key == (0, 0, 3):
                tile['values'] = np.asarray(tile['values'])+.001
            return tile
    monkeypatch.setattr(core._manta, 'FittingTiles', Perturbed)
    with pytest.raises(ValueError, match='fitted source violates symmetry selection'):
        case('water', factor_export='ordered-tiles')


def test_fixed_geometry_density_response_uses_actual_physical_fock():
    source, mol, basis, *_ = case()
    s = np.asarray(vq.compute_overlap(basis)); values, vectors = np.linalg.eigh(s)
    h = np.asarray(vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol))
    frame = OperatorFrame(basis_id=source.source_identity, n=basis.nbasis,
        ordering='native AO', image_gauge='molecular', overlap=s, X=vectors/np.sqrt(values))
    hf = ScreenedMolecularHF(MolecularRIJK(source, budget=BUDGET), frame=frame,
        hcore=h, nuclear_energy=mol.nuclear_repulsion())
    rng = np.random.default_rng(557)
    d = tuple(.01*a@a.T for a in rng.normal(size=(2, basis.nbasis, basis.nbasis)))
    direction = tuple(a+a.T for a in rng.normal(size=(2, basis.nbasis, basis.nbasis)))
    state = hf.evaluate(d, method='UHF', generation=0)
    step = 1e-5
    energy = [hf.evaluate(tuple(a+sign*step*b for a,b in zip(d,direction)),
        method='UHF', generation=1).build.energy.total for sign in (1,-1)]
    expected = sum(np.einsum('pq,qp', q, c.operator.matrix)
                   for q,c in zip(direction,state.build.channels))
    assert abs((energy[0]-energy[1])/(2*step)-expected) < 2e-9
    # This is a density response check; it admits no geometry derivative.


def test_native_factor_gram_is_preserved_without_whitening_or_a_new_rank_cut():
    irreps = cyclic()
    actions = []
    for g in range(3):
        angle = 2*np.pi*g/3
        u = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], complex)
        actions.append(BlockSpaceAction(SPACE, SPACE, np.array([0, 2], np.int64),
            np.array([0], np.int64), u.ravel(), antiunitary=False, budget=BUDGET))
    # A synthetic finite factor with a nonidentity Gram exposes any accidental
    # whitening: normalization would change its actual Gram tensor by 4x.
    factor = 2*np.eye(2)
    with pytest.raises(ValueError, match='metric-orthonormal'):
        _adapt_retained(irreps, actions, np.eye(2), factor, 1e-10, BUDGET)
    c, labels, evidence, rotation = _adapt_retained(irreps, actions, np.eye(2),
        factor, 1e-10, BUDGET, preserve_native_gram=True)
    np.testing.assert_allclose(c@c.conj().T, factor@factor.T, atol=2e-13, rtol=0)
    np.testing.assert_allclose(rotation@rotation.conj().T, np.eye(2), atol=2e-13, rtol=0)
    assert sorted(labels) == [1, 2] and evidence['preserved_native_gram']
    assert evidence['metric_gram_identity_residual'] == pytest.approx(3*np.sqrt(2))
    with pytest.raises(ValueError, match='not closed'):
        _adapt_retained(irreps, actions, np.eye(2), factor[:, :1], 1e-10, BUDGET,
                        preserve_native_gram=True)


@pytest.mark.parametrize('kind,orbital,auxiliary', [
    ('water', 'def2-svp', 'def2-universal-jkfit'),
    ('ammonia', 'def2-svp', 'def2-universal-jkfit'),
    ('ammonia', 'cc-pvdz', 'cc-pvdz-jkfit')])
def test_large_auxiliary_basis_keeps_native_factor_and_scientific_tolerance(kind, orbital, auxiliary):
    mol, basis, r, t = molecular_case(kind, basis_name=orbital)
    if kind == 'water':
        chars = np.array([[(-1)**((a & b).bit_count()) for b in range(4)] for a in range(4)], complex)
    else:
        r, t, chars = r[:3], t[:3], cyclic().characters
    aux = vq.BasisSet(mol, auxiliary)
    source = prepare_abelian_molecular_ri(mol, basis, aux, r, t, chars,
        source_tolerance=1e-10, budget=BUDGET)
    assert source.source_tolerance == 1e-10 and source.tensor_error_bound < 1e-10
    assert source.receipt['metric_rank'] == source._native.metric_info.rank == aux.nbasis
    if orbital == 'def2-svp':
        assert source.receipt['auxiliary_evidence']['metric_gram_identity_residual'] > 1e-10
    # Independent integral engine; preserve the actual native factor, not a
    # repaired inverse or a looser metric cutoff.
    b = np.einsum('PA,Puv->Auv', source._native.factor, vq.compute_3c_eri(basis, aux))
    rng = np.random.default_rng(919)
    d = rng.normal(size=(basis.nbasis,)*2)+1j*rng.normal(size=(basis.nbasis,)*2)
    result = source.jk(d, budget=BUDGET)
    for actual, reference in zip((result.j, result.k), dense(b, d)):
        np.testing.assert_allclose(actual, reference, atol=3e-10, rtol=0)
    for rotation, translation in zip(r, t):
        action = build_molecular_space_action(mol, basis, rotation, translation,
            source=SPACE, target=SPACE, budget=BUDGET)
        moved = source.jk(action.push_density(d), budget=BUDGET)
        for actual, original in ((moved.j, result.j), (moved.k, result.k)):
            np.testing.assert_allclose(action.pull_operator(actual), original, atol=3e-10, rtol=0)


def test_measured_routes_keep_source_state_and_probe_work_separate():
    from dataclasses import asdict
    from vibeqc.symmetry_molecular_hf import _select_route
    source, mol, basis, *_ = case()
    jk = MolecularRIJK(source, budget=BUDGET, executor='auto')
    route = jk.calibrate_routes(targets=('j', 'k', 'k'), expected_builds=100)
    assert jk.totals.builds == 0 and jk.last_build is None
    assert len(route.samples) == 4 and route.calibration_seconds > 0
    assert route.source_identity == source.source_identity
    assert all(s.route is None and s.work.shell_calls == 0 for s in route.samples)
    assert tuple(s.executor for s in route.samples) == ('ri-dense', 'ri-blocked', 'ri-blocked', 'ri-dense')
    assert route.logical_bytes > max(s.logical_bytes for s in route.samples)
    assert route.maximum_difference < source.source_tolerance
    times = {e: [s.seconds for s in route.samples if s.executor == 'ri-'+e] for e in ('dense', 'blocked')}
    assert route.selected == _select_route(times['dense'], times['blocked'],
        route.calibration_seconds, 100, baseline='dense', candidate='blocked')[0]
    d = np.random.default_rng(112).normal(size=(2, basis.nbasis, basis.nbasis))
    j, ka, kb = jk.build_jk_uhf(*d)
    assert jk.last_build.route is route and jk.totals.builds == 1
    expected = MolecularRIJK(source, budget=BUDGET, executor='dense').build_jk_uhf(*d)
    for a, b in zip((j, ka, kb), expected):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=0)
    fork = jk.fork()
    assert fork.source is source and fork.route_calibrations == jk.route_calibrations
    assert fork.totals.builds == 0 and fork.last_build is None
    fork.build_J(d[0])
    assert fork.last_build.route.selected == 'dense'
    assert len(jk.route_calibrations) == 1
    for signature in (('j',), ('k',), ('jk',), ('k', 'k')):
        jk.calibrate_routes(targets=signature, expected_builds=1)
    assert len(jk.route_calibrations) == 5 and len(fork.route_calibrations) == 1
    with pytest.raises(ValueError, match='signature'):
        jk.calibrate_routes(targets=('jk', 'jk'), expected_builds=100)
    json.dumps(asdict(route), allow_nan=False)
    # Cost evidence must not enter shared physical source identity.
    s = np.asarray(vq.compute_overlap(basis)); eigen, vec = np.linalg.eigh(s)
    frame = OperatorFrame(basis_id=source.source_identity, n=basis.nbasis,
        ordering='AO', image_gauge='molecular', overlap=s, X=vec/np.sqrt(eigen))
    h = np.asarray(vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol))
    identities = [ScreenedMolecularHF(MolecularRIJK(source, budget=BUDGET, executor=e),
        frame=frame, hcore=h, nuclear_energy=mol.nuclear_repulsion()).source.identity
        for e in ('dense', 'blocked', 'auto')]
    assert len(set(identities)) == 1


@pytest.mark.parametrize('expected,repetitions', [(1, 1), (1, 2), (2, 2), (5, 5)])
def test_short_ri_use_skips_all_calibration_contractions(expected, repetitions, monkeypatch):
    source, *_ = case()
    jk = MolecularRIJK(source, budget=BUDGET, executor='auto')
    def forbidden(*args, **kwargs):
        raise AssertionError('unprofitable calibration executed a contraction')
    monkeypatch.setattr(jk, '_execute', forbidden)
    route = jk.calibrate_routes(targets=('jk',), expected_builds=expected, repetitions=repetitions)
    assert route.selected == 'dense' and route.samples == ()
    assert route.calibration_seconds == route.modeled_work == 0
    assert jk.last_build is None and jk.totals.builds == 0


def test_ri_probe_failure_invalidates_route_and_budget_refuses_before_work(monkeypatch):
    source, _, basis, *_ = case()
    jk = MolecularRIJK(source, budget=BUDGET, executor='auto')
    jk.calibrate_routes(targets=('jk',), expected_builds=1)
    reservation = jk.requirements(('jk',))
    bounded = MolecularRIJK(source, budget=reservation, executor='auto')
    def forbidden(*args, **kwargs):
        raise AssertionError('a refused calibration executed work')
    monkeypatch.setattr(bounded, '_execute', forbidden)
    with pytest.raises(MemoryError):
        bounded.calibrate_routes(targets=('jk',), expected_builds=100)
    assert bounded.route_calibrations == ()
    execute = jk._execute
    def corrupt(ds, targets, executor):
        j, k, receipt = execute(ds, targets, executor)
        if executor == 'blocked':
            j += .001
        return j, k, receipt
    monkeypatch.setattr(jk, '_execute', corrupt)
    with pytest.raises(ValueError, match='identical selected source'):
        jk.calibrate_routes(targets=('jk',), expected_builds=100)
    assert jk.route_calibrations == () and jk.totals.builds == 0
    for bad in (0, -1, True):
        with pytest.raises((TypeError, ValueError)):
            jk.calibrate_routes(targets=('jk',), expected_builds=bad)
    with pytest.raises(ValueError, match='five paired'):
        jk.calibrate_routes(targets=('jk',), expected_builds=100, repetitions=6)
    with pytest.raises(ValueError, match='auto'):
        MolecularRIJK(source, budget=BUDGET).calibrate_routes(targets=('jk',), expected_builds=100)
    with pytest.raises(TypeError, match='float64'):
        jk.build_J(np.eye(basis.nbasis, dtype=complex))


def test_prepared_ri_budget_does_not_recharge_completed_integrals():
    source, _, basis, *_ = case()
    requested = source.requirements(budget=BUDGET)
    assert requested.maximum_work < source.required_work
    # The source setup is already complete; an execution-only work budget is
    # enough without authorizing another integral traversal or factorization.
    result = source.jk(np.eye(basis.nbasis, dtype=complex), budget=requested)
    assert result.scalar_products > 0


@pytest.mark.parametrize('targets', [('j',), ('k',), ('jk',), ('k', 'k'), ('j', 'k', 'k')])
def test_each_ri_measured_signature_obeys_actual_channel_contract(targets):
    source, _, basis, *_ = case()
    jk = MolecularRIJK(source, budget=BUDGET, executor='auto')
    route = jk.calibrate_routes(targets=targets, expected_builds=100, repetitions=1)
    d = tuple(np.random.default_rng(201+i).normal(size=(basis.nbasis,)*2) for i in range(len(targets)))
    j, k, receipt = jk.build_channels(d, targets=targets)
    reference = MolecularRIJK(source, budget=BUDGET, executor='dense').build_channels(d, targets=targets)
    for actual, expected in zip((j, k), reference[:2]):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0)
    assert receipt.route is route and receipt.targets == targets
    sample = next(s for s in route.samples if s.executor == receipt.executor)
    assert receipt.scalar_products == sample.scalar_products
    assert jk.totals.scalar_products == receipt.scalar_products
    assert all(s.source_identity == receipt.source_identity for s in route.samples)


@pytest.mark.parametrize('kind,pure', [('water', False), ('ammonia', False),
    ('cartesian', False), ('cartesian', True)])
def test_producer_export_matches_ordered_fallback_without_raw_tensor(kind, pure, monkeypatch):
    legacy, mol, basis, aux, r, t = case(kind, pure=pure, factor_export='ordered-tiles')
    def forbidden(*args, **kwargs):
        pytest.fail('producer export constructed an independent Python fitting provider')
    monkeypatch.setattr(core._manta, 'FittingTiles', forbidden)
    source, *_ = case(kind, pure=pure)
    a, b = source.receipt, legacy.receipt
    assert a['factor_export'] == 'producer' and a['raw_coupling_array_bytes'] == 0
    assert b['factor_export'] == 'ordered-tiles' and b['raw_coupling_array_bytes'] == 8*aux.nbasis*basis.nbasis**2
    for key in ('producer_identity', 'coupling_identity', 'metric_factor_identity', 'metric_rank',
                'metric_shell_calls', 'coupling_shell_calls', 'coupling_primitive_products'):
        assert a[key] == b[key]
    assert a['factor_export_receipt']['shell_calls'] == a['coupling_shell_calls']
    assert a['factor_export_receipt']['raw_coupling_array_bytes'] == 0
    # Distinct rounded exports are individually identified; both routes use
    # the same finite integrals and exact retained metric factor.
    assert source.source_identity != legacy.source_identity
    rng = np.random.default_rng(9146)
    d = rng.normal(size=(basis.nbasis,)*2)+1j*rng.normal(size=(basis.nbasis,)*2)
    x, y = source.jk(d, budget=BUDGET), legacy.jk(d, budget=BUDGET)
    np.testing.assert_allclose(x.j, y.j, atol=2e-11, rtol=0)
    np.testing.assert_allclose(x.k, y.k, atol=2e-11, rtol=0)


def test_producer_export_cannot_borrow_another_source(monkeypatch):
    original = core._manta.RIJKBuilder
    other, *_ = case('ammonia')
    foreign = other._native.export_factors(core._manta.ResourceBudget())
    class WrongExport:
        def __init__(self, *args):
            self.native = original(*args)
        def __getattr__(self, name):
            return getattr(self.native, name)
        def export_factors(self, budget):
            return foreign
    monkeypatch.setattr(core._manta, 'RIJKBuilder', WrongExport)
    with pytest.raises(ValueError, match='incompatible native RI factor export'):
        case('water')


def native_fixture():
    # Deliberately arbitrary complex factors, empty irreps, rectangular panels
    # and nonsymmetric D. No geometry or Hermiticity can hide index errors.
    o = np.array([0, 1, 1, 4], np.int64)
    a = np.array([0, 1, 3, 3], np.int64)
    keys = np.array([(0, 0, 0), (0, 2, 1), (2, 0, 1), (2, 2, 0)], np.int64)
    rng = np.random.default_rng(6049)
    factors = np.zeros((3, 4, 4), complex)
    packed = []
    for p, q, c in keys:
        shape = (a[c+1]-a[c], o[p+1]-o[p], o[q+1]-o[q])
        block = rng.normal(size=shape)+1j*rng.normal(size=shape)
        factors[a[c]:a[c+1], o[p]:o[p+1], o[q]:o[q+1]] = block
        packed.extend(block.ravel())
    values = np.asarray(packed, complex)
    d = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    return o, a, keys, values, d, factors


@pytest.mark.parametrize('j,k', [(True, False), (False, True), (True, True)])
def test_native_panels_independent_complex_reference_and_owned_payload(j, k):
    o, a, keys, values, d, factors = native_fixture()
    plan = core._SymmetryRIBlocks(4, 3, o, a, keys, values, BUDGET.native())
    expected = dense(factors, d)
    # All constructor input storage may be overwritten after compilation.
    o[:] = 0; a[:] = 0; keys[:] = 0; values[:] = 0
    result = plan.contract(d, j, k, BUDGET.native())
    for x, y, requested in zip(result[:2], expected, (j, k)):
        if requested:
            np.testing.assert_allclose(x, y, atol=3e-13, rtol=0)
        else:
            assert x is None
    # Independent shape arithmetic for the four rectangular blocks above.
    products = (44 if j else 0)+(176 if k else 0)
    calls = (12 if j else 0)+(24 if k else 0)
    assert result[2] == dict(scalar_products=products, block_contractions=calls)
    for x in result[:2]:
        if x is not None:
            x[:] = 100
    repeated = plan.contract(d, j, k, BUDGET.native())
    for x, y, requested in zip(repeated[:2], expected, (j, k)):
        if requested:
            np.testing.assert_allclose(x, y, atol=3e-13, rtol=0)
    with pytest.raises(ValueError, match='nonfinite'):
        plan.contract(np.full((4, 4), np.nan, complex), j, k, BUDGET.native())
    # A failed call exposes no partial target and does not poison immutable factors.
    for x, y, requested in zip(plan.contract(d, j, k, BUDGET.native())[:2], expected, (j, k)):
        if requested:
            np.testing.assert_allclose(x, y, atol=3e-13, rtol=0)


def test_native_blocks_reject_bad_descriptors_resources_and_payloads():
    o, a, keys, values, d, _ = native_fixture()
    def build(orbital=o, auxiliary=a, descriptor=keys, packed=values, budget=BUDGET):
        return core._SymmetryRIBlocks(4, 3, orbital, auxiliary, descriptor, packed, budget.native())
    with pytest.raises(ValueError, match='duplicate'):
        build(descriptor=np.array([keys[0], keys[0]], np.int64), packed=values[:2])
    for bad in (np.array([0, 1, -1, 4], np.int64), np.array([1, 1, 1, 4], np.int64)):
        with pytest.raises(ValueError, match='offsets'):
            build(orbital=bad)
    with pytest.raises(ValueError, match='nonempty axes'):
        build(descriptor=np.array([[1, 0, 0]], np.int64))
    with pytest.raises(ValueError, match='extent'):
        build(packed=values[:-1])
    with pytest.raises(ValueError, match='key out of range'):
        build(descriptor=np.array([[0, 0, 3]], np.int64))
    with pytest.raises(ValueError, match='nonfinite'):
        build(packed=np.full_like(values, np.nan))
    for bad in (values.real.copy(), values[::-1], values.astype(np.complex64)):
        with pytest.raises(ValueError, match='exact aligned'):
            build(packed=bad)
    for budget in (Budget(1, 10**12), Budget(10**9, 1)):
        with pytest.raises(ValueError, match='budget'):
            build(packed=np.full_like(values, np.nan), budget=budget)
    with pytest.raises(OverflowError):
        core._plan_symmetry_ri_blocks(2**63, 1, 1, 0, 0, BUDGET.native())
    plan = build()
    for bad in (d.real.copy(), d.T, d.astype(np.complex64)):
        with pytest.raises(ValueError, match='exact aligned'):
            plan.contract(bad, True, True, BUDGET.native())
    with pytest.raises(ValueError, match='density shape'):
        plan.contract(d[:2].copy(), True, True, BUDGET.native())
    with pytest.raises(ValueError, match='requires J or K'):
        plan.contract(d, False, False, BUDGET.native())
    with pytest.raises(ValueError, match='budget'):
        plan.contract(np.full_like(d, np.nan), True, True, Budget(1, 1).native())
    with pytest.raises(ValueError, match='nonfinite'):
        plan.contract(np.full_like(d, 1e308+1e308j), True, True, BUDGET.native())
    empty = core._SymmetryRIBlocks(4, 3, o, a, np.empty((0, 3), np.int64),
        np.empty(0, complex), BUDGET.native())
    zero = empty.contract(d, True, True, BUDGET.native())
    assert np.count_nonzero(zero[0]) == np.count_nonzero(zero[1]) == 0
    assert zero[2] == dict(scalar_products=0, block_contractions=0)
