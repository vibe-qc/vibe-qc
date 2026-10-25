"""Physical fitted-kernel algebra on finite Fourier data, without native builds.

The actual Python builders perform metric construction/whitening. Synthetic
Fourier integrals isolate this algebra from native integral qualification.
Native SR kernels and SCF remain subject to their separate acceptance lanes.
"""
import ast
import builtins
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional

import numpy as np
import pytest


@pytest.fixture
def fitted():
    path = Path(__file__).parents[1] / 'python/vibeqc/aux_basis.py'
    names = {'_physical_fit_kernel', '_physical_fit_reciprocal_weights',
             '_fitted_support_policy', '_immutable_fit_array', '_fit_record_json',
             '_classic_source_descriptor', '_record_classic_support',
             '_RsgdfQMetricCacheEntry', '_metric_keep_mask',
             'build_lpq_bloch_native_fft_shared_q',
             '_build_lpq_range_separated_shared_q', '_RangeSeparatedGdfBatch',
             '_RangeSeparatedGdfAdmissionError'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    vectors = np.column_stack((np.arange(5.), np.zeros((5, 2))))
    rng = np.random.default_rng(9103)
    aux_ft = rng.normal(size=(2, 5)) + 1j*rng.normal(size=(2, 5))
    pair_ft = rng.normal(size=(2, 2, 5)) + 1j*rng.normal(size=(2, 2, 5))
    # A large forbidden zero channel makes accidental zero inclusion obvious.
    aux_ft[:, 0] = pair_ft[:, :, 0] = 1.e9

    def indices(g):
        return np.rint(g[:, 0]).astype(int)

    def pair_transform(basis, g, cells, kets, **kwargs):
        return np.asarray([pair_ft[:, :, indices(g)] * np.exp(1j*k[0]) for k in kets])

    modules = {'_aopair_ft': SimpleNamespace(ao_pair_fourier_transform_bloch_multi=pair_transform),
               '_vibeqc_core': SimpleNamespace(direct_lattice_cells=lambda *a: [])}
    def import_source(name, globals=None, locals=None, fromlist=(), level=0):
        return modules[name] if level and name in modules else builtins.__import__(
            name, globals, locals, fromlist, level)
    scope = dict(np=np, json=json, Optional=Optional, Any=Any, NamedTuple=NamedTuple,
        PeriodicSystem=object, BasisSet=object, LatticeSumOptions=SimpleNamespace,
        _RangeSeparatedGdfFitState=object,
        __builtins__=dict(vars(builtins), __import__=import_source),
        _classic_build_reservation=lambda *a: None,
        _classic_pair_cells=lambda *a: np.zeros((1, 3)),
        _canonical_reciprocal_transfer=lambda s, q: q,
        _classic_reciprocal_mesh=lambda s, q, cutoff, *a: vectors[np.sum(vectors**2, axis=1) <= 2*cutoff],
        _rsgdf_shifted_sphere_boundary_tolerance=lambda *a: 0.,
        _range_separated_gdf_source_signature=lambda *a: 'synthetic-finite-source',
        _fit_basis_record=lambda b: dict(name=b.name),
        _rsgdf_aux_ft_cache_budget_bytes=lambda: 1024**2,
        rsgdf_aux_fourier_transform=lambda b, g: aux_ft[:, indices(g)],
        _rsgdf_fit_pair_keep_mask=lambda *a, **kw: None,
        _ao_scales_for_rsgdf=lambda b: np.ones(b.nbasis),
        _progress_info=lambda *a: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    scope.update(vectors=vectors, aux_ft=aux_ft, pair_ft=pair_ft, modules=modules)
    return scope


def build(fitted, kernel, omega, **kwargs):
    basis = SimpleNamespace(nbasis=2, name='synthetic')
    system = SimpleNamespace(lattice=np.eye(3)*3., dim=3)
    return fitted['build_lpq_bloch_native_fft_shared_q'](
        system, basis, basis, np.array([[.2, 0., 0.], [-.3, 0., 0.]]),
        np.array([.1, 0., 0.]), ke_cutoff=2., tail_ke_cutoff=8., tail_chunk_g=2,
        lat_opts=SimpleNamespace(cutoff_bohr=5.), support_policy='historical_index_ball',
        kernel=kernel, omega_screen=omega, **kwargs)


@pytest.mark.parametrize('kernel,omega', [('coulomb', 0.), ('erfc', .7), ('erf', .7)])
@pytest.mark.parametrize('canonical', [False, True])
def test_actual_classic_builder_whitens_matched_metric_and_three_center(fitted, kernel, omega, canonical):
    records = []
    actual = build(fitted, kernel, omega, canonical_auxiliary_basis=canonical,
                   _support_recorder=records.append)
    p2 = np.arange(1., 5.)**2
    weights = 4*np.pi/(27.*p2)
    if kernel == 'erfc':
        weights *= 1-np.exp(-p2/(4*omega**2))
    elif kernel == 'erf':
        weights *= np.exp(-p2/(4*omega**2))
    f = fitted['aux_ft'][:, 1:]
    metric = (f.conj()*weights) @ f.T
    for index, ket in enumerate((.3, -.2)):
        pairs = fitted['pair_ft'][:, :, 1:].reshape(4, 4)*np.exp(1j*ket)
        tensor = (f.conj()*weights) @ pairs.T
        # Gram equality avoids imposing an arbitrary eigenvector gauge.
        expected = tensor.conj().T @ np.linalg.solve(metric, tensor)
        factors = actual[index].reshape(2, 4)
        np.testing.assert_allclose(factors.conj().T @ factors, expected, atol=2e-13)
    record = json.loads(records[0])
    assert record['descriptor']['physical_kernel'] == kernel
    assert record['rank'] == 2
    assert record['reciprocal_base'] == [[1., 0., 0.], [2., 0., 0.]]
    assert record['reciprocal_tail'] == [[3., 0., 0.], [4., 0., 0.]]


def test_full_default_preserved_and_physical_metric_cache_is_not_reused(fitted):
    implicit = build(fitted, None, 0.)
    explicit = build(fitted, 'coulomb', 0.)
    for old, new in zip(implicit, explicit):
        np.testing.assert_array_equal(old, new)
    cache = {}
    build(fitted, 'erfc', .7, _q_metric_cache=cache)
    with pytest.raises(ValueError, match='different source'):
        build(fitted, 'erf', .7, _q_metric_cache=cache)
    with pytest.raises(ValueError, match='different source'):
        build(fitted, 'erfc', .8, _q_metric_cache=cache)


def test_weight_limits_and_complement_at_fixed_nonzero_support(fitted):
    weights = fitted['_physical_fit_reciprocal_weights']
    p2 = np.array([0., 1.e-8, 1., 9.])
    full = weights(p2, 31., 'coulomb', 0.)
    sr = weights(p2, 31., 'erfc', .3)
    lr = weights(p2, 31., 'erf', .3)
    np.testing.assert_allclose(sr+lr, full, rtol=2e-16)
    assert sr[0] == lr[0] == full[0] == 0
    np.testing.assert_allclose(sr[1], np.pi/(31*.3**2), rtol=2e-8)
    np.testing.assert_allclose(weights(p2, 31., 'erfc', 1.e-8), full)
    np.testing.assert_allclose(weights(p2, 31., 'erf', 1.e8), full)


@pytest.mark.parametrize('kernel,omega', [('bad', .1), ('coulomb', .1), ('erf', 0.),
    ('erfc', -.1), ('erf', True), ('erfc', float('nan')), ('erfc', 1.e300)])
def test_bad_physical_kernel_is_rejected(fitted, kernel, omega):
    with pytest.raises(ValueError):
        fitted['_physical_fit_kernel'](kernel, omega)


@pytest.mark.parametrize('kernel,omega', [('coulomb', 0.), ('erfc', .7), ('erf', .7)])
def test_sr_builder_receives_physical_range_and_whitens_its_own_metric(fitted, kernel, omega):
    f = fitted['aux_ft'][:, 1:]
    pairs = fitted['pair_ft'][:, :, 1:].reshape(4, 4)
    p2 = np.arange(1., 5.)**2
    weights = fitted['_physical_fit_reciprocal_weights'](p2, 27., kernel, omega)
    metric = (f.conj()*weights) @ f.T
    tensor = ((f.conj()*weights) @ pairs.T).reshape(1, 2, 2, 2)
    calls = []
    def native(*args, **kw):
        calls.append((args[6], kw))
        return metric.copy(), tensor.copy(), 4
    fitted['modules']['_vibeqc_core'].compute_gdf_range_separated_integrals = native
    fitted['_rsgdf_bounded_reciprocal_sphere'] = lambda *a, **kw: fitted['vectors'].copy()
    basis = SimpleNamespace(nbasis=2, name='synthetic')
    opts = dict(omega=.5, pair_cutoff=5., auxiliary_cutoff=6., ke_cutoff=8.,
        linear_dep_thr=1e-9, memory_byte_cap=16*1024**2, native_workspace_byte_cap=1024**2,
        image_candidate_cap=1000, reciprocal_candidate_cap=1000,
        canonical_auxiliary_basis=True, kernel=kernel, omega_screen=omega)
    records, cache = [], {}
    result = fitted['_build_lpq_range_separated_shared_q'](
        SimpleNamespace(lattice=np.eye(3)*3.), basis, basis, [[.2, 0., 0.]], [.1, 0., 0.],
        _support_recorder=records.append, _metric_state=cache, **opts)
    factors = result.factors[0].reshape(2, 4)
    t = tensor.reshape(2, 4)
    np.testing.assert_allclose(factors.conj().T @ factors,
                               t.conj().T @ np.linalg.solve(metric, t), atol=2e-13)
    assert calls == [(.5, {} if kernel == 'coulomb' else dict(kernel=kernel, omega_screen=omega))]
    if kernel != 'coulomb':
        record = json.loads(records[0])
        assert record['physical_kernel'] == kernel and record['omega_screen'] == omega
        assert record['numerical_split_omega_bohr_inv'] == .5
        assert record['rank'] == 2
        if kernel == 'erf':
            assert record['kernel']['real'] == 'none'
            assert record['selectors']['screen'] == 'none'
            assert 'no real-space' in record['selectors']['metric']
        with pytest.raises(ValueError, match='different source'):
            fitted['_build_lpq_range_separated_shared_q'](
                SimpleNamespace(lattice=np.eye(3)*3.), basis, basis, [[.2, 0., 0.]], [.1, 0., 0.],
                _metric_state=cache, **dict(opts, omega_screen=.8))
        with pytest.raises(NotImplementedError, match='derivatives'):
            fitted['_build_lpq_range_separated_shared_q'](
                SimpleNamespace(lattice=np.eye(3)*3.), basis, basis, [[.2, 0., 0.]], [.1, 0., 0.],
                retain_fit_state=True, **dict(opts, canonical_auxiliary_basis=False))


@pytest.mark.parametrize('source', ['classic', 'sr'])
def test_operator_identity_binds_physical_range_source_and_external_zero_convention(fitted, source):
    root = Path(__file__).parents[1] / 'python/vibeqc'
    desc_nodes = [node for node in ast.parse((root/'periodic_fit_source.py').read_text()).body
                  if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                  and node.name in {'RSGDFFitSource', 'factor_digest'}]
    desc = dict(dataclass=dataclass, asdict=asdict, hashlib=hashlib, json=json, np=np)
    exec(compile(ast.Module(body=desc_nodes, type_ignores=[]), '<fit-descriptor>', 'exec'), desc)
    calls = []
    def raw_factors(*args, **kw):
        calls.append(kw)
        record = json.dumps(dict(rank=2, metric_eigenvalues=[1., 2.], metric_keep=[True, True]))
        kw['_support_recorder'](record)
        factors = np.ones((len(args[3]), 2, 2, 2), dtype=complex)
        return list(factors) if source == 'classic' else SimpleNamespace(
            factors=factors, executed_support=record)
    modules = dict(aux_basis=SimpleNamespace(
        _canonical_reciprocal_transfer=lambda s, q: q,
        _basis_fingerprint=lambda b: b.name,
        _range_separated_gdf_source_signature=lambda *a: 'synthetic-source',
        _physical_fit_kernel=fitted['_physical_fit_kernel'],
        build_lpq_bloch_native_fft_shared_q=raw_factors,
        _build_lpq_range_separated_shared_q=raw_factors),
        periodic_fit_source=SimpleNamespace(**desc),
        _vibeqc_core=SimpleNamespace(LatticeSumOptions=SimpleNamespace),
        _aopair_ft=SimpleNamespace(_aopair_ft_backend=lambda: 'cxx'))
    def import_source(name, globals=None, locals=None, fromlist=(), level=0):
        return modules[name] if level and name in modules else builtins.__import__(
            name, globals, locals, fromlist, level)
    scope = dict(__builtins__=dict(vars(builtins), __import__=import_source))
    path = root/'periodic_fitted_support.py'
    exec(compile(path.read_text(), str(path), 'exec'), scope)
    control_options = {} if source == 'classic' else dict(
        auxiliary_cutoff_bohr=9., omega_bohr_inv=.5)
    controls = scope['ChiCoulombControls'](5., 6., 8., 12., 5., .6, **control_options)
    basis = SimpleNamespace(nbasis=2, name='synthetic')
    common = dict(source=source, controls=controls)
    args = (SimpleNamespace(dim=3), basis, basis, [[0., 0., 0.]], [1.])
    full = scope['build_chi_fitted_operator'](*args, **common)
    assert 'kernel' not in calls[-1] and 'omega_screen' not in calls[-1]
    short = scope['build_chi_fitted_operator'](*args, kernel='erfc', omega_screen=.7, **common)
    long = scope['build_chi_fitted_operator'](*args, kernel='erf', omega_screen=.7, **common)
    changed_range = scope['build_chi_fitted_operator'](*args, kernel='erfc', omega_screen=.8, **common)
    assert len({x.identity for x in (full, short, long, changed_range)}) == 4
    assert short.kernel == 'erfc' and short.omega_screen == .7
    assert calls[-1]['kernel'] == 'erfc' and calls[-1]['omega_screen'] == .8
    if source == 'sr':
        assert calls[-1]['omega'] == .5
    for operator, sign, madelung in ((short, 1., 0.), (long, -1., 1.)):
        zero = operator.as_dict()['exchange_zero_mode']
        assert zero['finite_constant_sign'] == sign
        assert zero['full_madelung_coefficient'] == madelung
        assert zero['included_in_factors'] is False
        assert 'equal-k weight' in zero['finite_constant']
        with pytest.raises(ValueError):
            operator[(0, 0)].setflags(write=True)
        with pytest.raises(ValueError, match='different source'):
            operator.require_identity(full.identity)
