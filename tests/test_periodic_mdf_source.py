"""Acceptance witnesses for the private MDF source; no production enabling.

Added during implementation-only work. Run with a matching rebuilt core when
numerical validation resumes; these tests were not executed during recovery.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from vibeqc import periodic_mdf as mdf
from vibeqc.aux_basis import (
    _RangeSeparatedGdfAdmissionError, _build_lpq_range_separated_shared_q,
)


def _fixture():
    system = core.PeriodicSystem(3, np.eye(3)*6., [core.Atom(2, [0., 0., 0.])])
    molecule = system.unit_cell_molecule()
    def basis(exponent):
        return core.BasisSet(molecule, [
            core.ShellInfo(0, 0, True, [exponent], [1.], [0., 0., 0.]),
        ], 'mdf-source-witness', True)
    return system, basis(.6), basis(.8)


def _options():
    return dict(omega=.8, pair_cutoff=6.2, auxiliary_cutoff=8., ke_cutoff=6.,
                plane_wave_cutoff=.7, linear_dep_thr=1e-9,
                memory_byte_cap=128*1024**2, native_workspace_byte_cap=64*1024**2,
                image_candidate_cap=100000, reciprocal_candidate_cap=100000)


@pytest.mark.parametrize('metric', [False, True])
def test_pw_projection_matches_independent_gaussian_image_sum(metric):
    system, orbital, auxiliary = _fixture()
    vectors = np.array([[0., 0., 0.], [2*np.pi/6, 0., 0.], [-2*np.pi/6, 0., 0.]])
    kets = np.array([[0., 0., 0.], [.21, -.13, .07]])
    m, t, f, count = core._compute_gdf_plane_wave_projection(
        orbital, auxiliary, system, np.zeros(3), kets, vectors, 6.2,
        1024**2, 1024**2, 100000, metric,
    )
    a, c = orbital.shells()[0].exponents[0], orbital.shells()[0].coefficients[0]
    b, d = auxiliary.shells()[0].exponents[0], auxiliary.shells()[0].coefficients[0]
    p2 = np.einsum('gi,gi->g', vectors, vectors)
    weights = np.zeros(len(vectors))
    weights[1:] = 4*np.pi/(216*p2[1:])
    aux_ft = d*(np.pi/b)**1.5*np.exp(-p2/(4*b))
    pairs = np.zeros((len(kets), len(vectors)), complex)
    for label in product(range(-1, 2), repeat=3):
        r = 6*np.asarray(label)
        if np.linalg.norm(r) > 6.2:
            continue
        spatial = c*c*(np.pi/(2*a))**1.5*np.exp(-a*np.dot(r, r)/2-p2/(8*a))
        pairs += np.exp(1j*kets@r)[:, None]*np.exp(-.5j*vectors@r)[None, :]*spatial
    np.testing.assert_allclose(f[:, :, 0, 0], pairs*np.sqrt(weights), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(t[:, 0, 0, 0], pairs@(weights*aux_ft), atol=2e-12, rtol=2e-12)
    if metric:
        np.testing.assert_allclose(m, [[np.dot(weights, aux_ft**2)]], atol=2e-12, rtol=2e-12)
    else:
        assert m.shape == (0, 0)
    assert count == 2
    np.testing.assert_array_equal(f[:, 0], 0.)


@pytest.mark.parametrize('field', ['output', 'workspace'])
def test_pw_projection_rejects_unadmitted_storage(field):
    system, orbital, auxiliary = _fixture()
    with pytest.raises((ValueError, RuntimeError), match='byte cap'):
        core._compute_gdf_plane_wave_projection(
            orbital, auxiliary, system, np.zeros(3), np.zeros((1, 3)),
            np.array([[.2, .1, .3]]), 6.2,
            1 if field == 'output' else 1024**2,
            1 if field == 'workspace' else 1024**2, 100000, True,
        )


@pytest.mark.parametrize('q', [np.zeros(3), np.array([.13, -.07, .02])])
def test_no_pw_limit_is_the_same_range_separated_gram_operator(q):
    system, orbital, auxiliary = _fixture()
    options = _options()
    options['plane_wave_cutoff'] = 0.
    bras = np.array([[0., 0., 0.], [.17, .12, -.04]])
    actual = mdf._build_mdf_shared_q(system, orbital, auxiliary, bras, q, **options)
    reference = _build_lpq_range_separated_shared_q(
        system, orbital, auxiliary, bras, q,
        **{name: value for name, value in options.items() if name != 'plane_wave_cutoff'},
    )
    np.testing.assert_allclose(actual.factors, reference.factors, atol=2e-12, rtol=2e-12)
    assert actual.plane_wave_count == 0


def test_shared_q_reuses_only_the_same_dressed_metric(monkeypatch):
    system, orbital, auxiliary = _fixture()
    options = _options()
    q = np.array([.13, -.07, .02])
    bras = np.array([[0., 0., 0.], [.17, .12, -.04]])
    full = mdf._build_mdf_shared_q(system, orbital, auxiliary, bras, q, **options)
    state, seen = {}, []
    original = core.compute_gdf_range_separated_integrals
    def observe(*args):
        seen.append(args[-1])
        return original(*args)
    monkeypatch.setattr(core, 'compute_gdf_range_separated_integrals', observe)
    parts = [mdf._build_mdf_shared_q(system, orbital, auxiliary, bra[None], q,
                                    _metric_state=state, **options).factors for bra in bras]
    np.testing.assert_allclose(np.concatenate(parts), full.factors, atol=2e-12, rtol=2e-12)
    assert seen == [True, False]
    with pytest.raises(ValueError, match='different source or PW span'):
        mdf._build_mdf_shared_q(system, orbital, auxiliary, bras, q,
                               _metric_state=state, **dict(options, plane_wave_cutoff=1.))
    assert seen == [True, False]


def test_cache_subdivision_preserves_pairs_and_bra_order(monkeypatch):
    system, orbital, auxiliary = _fixture()
    points = np.array([[0., 0., 0.], [.13, -.07, .02], [-.13, .07, -.02]])
    options = _options()
    full = mdf._build_mdf_cache(system, orbital, auxiliary, points, True, bra_rows=[2, 0], **options)
    original = mdf._build_mdf_shared_q
    def single(*args, **kwargs):
        if len(args[3]) > 1:
            raise _RangeSeparatedGdfAdmissionError('test preallocation rejection')
        return original(*args, **kwargs)
    monkeypatch.setattr(mdf, '_build_mdf_shared_q', single)
    split = mdf._build_mdf_cache(system, orbital, auxiliary, points, True, bra_rows=[2, 0], **options)
    assert set(full) == set(split) == {(2, 0), (2, 1), (2, 2), (0, 0), (0, 1), (0, 2), (1, 1)}
    for pair in full:
        np.testing.assert_allclose(full[pair], split[pair], atol=2e-12, rtol=2e-12)
    assert all(width == 1 for widths in split.q_batch_sizes for width in widths)
    assert split.retained_factor_bytes <= split.reserved_peak_bytes <= options['memory_byte_cap']


def test_allocator_failure_does_not_retry_as_a_smaller_batch(monkeypatch):
    system, orbital, auxiliary = _fixture()
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise MemoryError('actual allocator failure')
    monkeypatch.setattr(mdf, '_build_mdf_shared_q', fail)
    with pytest.raises(MemoryError, match='actual allocator'):
        mdf._build_mdf_cache(system, orbital, auxiliary, np.zeros((1, 3)), False, **_options())
    assert calls == [1]


def test_empty_nonzero_q_pw_span_is_a_gaussian_fit():
    system, orbital, auxiliary = _fixture()
    q = np.array([.13, -.07, .02])
    bras = np.zeros((1, 3))
    options = _options()
    empty = mdf._build_mdf_shared_q(
        system, orbital, auxiliary, bras, q,
        **dict(options, plane_wave_cutoff=1e-6),
    )
    gaussian = mdf._build_mdf_shared_q(
        system, orbital, auxiliary, bras, q,
        **dict(options, plane_wave_cutoff=0.),
    )
    assert empty.plane_wave_count == 0
    np.testing.assert_allclose(empty.factors, gaussian.factors, atol=2e-12, rtol=2e-12)


def test_reciprocal_allocator_failure_is_not_a_retryable_reservation(monkeypatch):
    from vibeqc import aux_basis

    system, orbital, auxiliary = _fixture()
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise MemoryError('reciprocal allocator failure')
    monkeypatch.setattr(aux_basis, '_rsgdf_bounded_reciprocal_sphere', fail)
    with pytest.raises(MemoryError, match='reciprocal allocator') as caught:
        mdf._build_mdf_shared_q(system, orbital, auxiliary, np.zeros((2, 3)),
                               np.zeros(3), **_options())
    assert not isinstance(caught.value, _RangeSeparatedGdfAdmissionError)
    assert calls == [1]


def _operator_fixture():
    from vibeqc.aux_basis import _range_separated_gdf_source_signature

    system, _, auxiliary = _fixture()
    orbital = core.BasisSet(system.unit_cell_molecule(), [
        core.ShellInfo(0, 0, True, [exponent], [1.], [0., 0., 0.])
        for exponent in (.6, .3)
    ], 'mdf-operator-witness', True)
    points = np.array([[0., 0., 0.], [.13, -.07, .02]])
    rng = np.random.default_rng(2945)
    factors = {}
    for i in range(2):
        raw = rng.normal(size=(3, 2, 2))+1j*rng.normal(size=(3, 2, 2))
        factors[(i, i)] = .5*(raw+raw.conj().transpose(0, 2, 1))
    factors[(0, 1)] = rng.normal(size=(4, 2, 2))+1j*rng.normal(size=(4, 2, 2))
    factors[(1, 0)] = factors[(0, 1)].conj().transpose(0, 2, 1)
    cache = mdf._MdfCache(
        factors, _range_separated_gdf_source_signature(system, orbital, auxiliary),
        ('mdf-sr-lr-pw-v1',), tuple(map(tuple, points)), ((2,), (1,), (1,)),
        sum(f.nbytes for f in factors.values()), 1024**2, (1, 0), True,
    )
    raw = rng.normal(size=(2, 2, 2, 2))+1j*rng.normal(size=(2, 2, 2, 2))
    density = raw@raw.conj().swapaxes(-1, -2)
    return system, orbital, auxiliary, points, cache, density


@pytest.mark.parametrize('spin_resolved', [False, True])
def test_fixed_density_operator_matches_explicit_eri_contractions(spin_resolved):
    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    density = spins if spin_resolved else spins.sum(axis=0)
    weights = np.array([.3, .7])
    result = mdf._build_mdf_jk(
        cache, system, orbital, auxiliary, points, density, weights,
        spin_resolved=spin_resolved, memory_byte_cap=16*1024**2,
        workspace_byte_cap=4096,
    )
    channels = density if spin_resolved else density[None]
    total = channels.sum(axis=0)
    jref, kref = [], []
    for i in range(2):
        matrix = np.zeros((2, 2), complex)
        for j in range(2):
            eri = np.einsum('Pmn,Prs->mnrs', cache[(i, i)], cache[(j, j)].conj())
            matrix += weights[j]*np.einsum('mnrs,rs->mn', eri, total[j])
        jref.append(matrix)
    for channel in channels:
        matrices = []
        for i in result.bra_rows:
            matrix = np.zeros((2, 2), complex)
            for j in range(2):
                eri = np.einsum('Pmr,Pns->mnrs', cache[(i, j)], cache[(i, j)].conj())
                matrix += weights[j]*np.einsum('mnrs,rs->mn', eri, channel[j])
            matrices.append(matrix)
        kref.append(matrices)
    np.testing.assert_allclose(result.coulomb, jref, atol=1e-11, rtol=1e-12)
    np.testing.assert_allclose(result.exchange, kref, atol=1e-11, rtol=1e-12)
    ej = .5*sum(w*np.trace(d@j).real for w, d, j in zip(weights, total, jref))
    ek = (-.5 if spin_resolved else -.25)*sum(
        weights[i]*np.trace(channel[i]@matrix).real
        for channel, matrices in zip(channels, kref)
        for i, matrix in zip(result.bra_rows, matrices)
    )
    assert result.coulomb_energy == pytest.approx(ej, abs=1e-11)
    assert result.exchange_energy == pytest.approx(ek, abs=1e-11)
    assert result.bra_rows == (1, 0)


def test_partial_exchange_keeps_full_ket_sum_and_omits_incomplete_energy():
    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    args = (cache, system, orbital, auxiliary, points, spins, [.3, .7])
    options = dict(spin_resolved=True, memory_byte_cap=16*1024**2, workspace_byte_cap=4096)
    full = mdf._build_mdf_jk(*args, **options)
    partial = mdf._build_mdf_jk(*args, bra_rows=[1], **options)
    np.testing.assert_allclose(partial.coulomb, full.coulomb)
    np.testing.assert_allclose(np.asarray(partial.exchange)[:, 0], np.asarray(full.exchange)[:, 0])
    assert partial.exchange_energy is None
    assert partial.coulomb_energy == full.coulomb_energy
    hartree = mdf._build_mdf_jk(*args, with_exchange=False, **options)
    assert hartree.exchange == ((), ())
    assert hartree.exchange_energy is None
    np.testing.assert_allclose(hartree.coulomb, full.coulomb)


@pytest.mark.parametrize('invalid', ['source', 'mesh', 'density', 'weights', 'pair', 'memory'])
def test_fixed_density_refuses_invalid_inputs_before_contraction(monkeypatch, invalid):
    from dataclasses import replace
    from vibeqc import aux_basis

    system, orbital, auxiliary, points, cache, density = _operator_fixture()
    weights, cap = [.3, .7], 16*1024**2
    if invalid == 'source':
        cache = replace(cache, source_signature='another-source')
    elif invalid == 'mesh':
        points = points[::-1]
    elif invalid == 'density':
        density[0, 0, 0, 1] += 1j
    elif invalid == 'weights':
        weights = [.3, .3]
    elif invalid == 'pair':
        cache = replace(cache, factors={p: f for p, f in cache.items() if p != (1, 0)})
    else:
        cap = 1
    def forbidden(*args, **kwargs):
        pytest.fail('invalid input reached contraction')
    monkeypatch.setattr(aux_basis, '_build_coulomb_from_diagonal_factors', forbidden)
    monkeypatch.setattr(aux_basis, '_accumulate_exchange_from_factors', forbidden)
    with pytest.raises((ValueError, _RangeSeparatedGdfAdmissionError)):
        mdf._build_mdf_jk(cache, system, orbital, auxiliary, points, density, weights,
                          spin_resolved=True, memory_byte_cap=cap, workspace_byte_cap=4096)


def test_built_cache_is_read_only_and_connects_to_fixed_density_operator():
    system, orbital, auxiliary = _fixture()
    points = np.array([[0., 0., 0.], [.13, -.07, .02]])
    cache = mdf._build_mdf_cache(system, orbital, auxiliary, points, True, **_options())
    with pytest.raises(TypeError):
        cache.factors[(0, 0)] = np.zeros_like(cache[(0, 0)])
    with pytest.raises(ValueError, match='read-only'):
        cache[(0, 0)][0, 0, 0] = 0.
    result = mdf._build_mdf_jk(
        cache, system, orbital, auxiliary, points, np.ones((2, 1, 1)), [.3, .7],
        memory_byte_cap=16*1024**2, workspace_byte_cap=4096,
    )
    assert result.reserved_peak_bytes <= 16*1024**2
    assert np.isfinite(result.coulomb_energy)
    assert np.isfinite(result.exchange_energy)


def test_fixed_density_operator_is_invariant_to_complex_fitting_coordinates():
    from dataclasses import replace

    system, orbital, auxiliary, points, cache, density = _operator_fixture()
    rng = np.random.default_rng(551)
    def unitary(rank):
        raw = rng.normal(size=(rank, rank))+1j*rng.normal(size=(rank, rank))
        return np.linalg.qr(raw)[0]
    zero_q = unitary(3)
    factors = {}
    for pair, factor in cache.items():
        transform = zero_q if pair[0] == pair[1] else unitary(len(factor))
        factors[pair] = np.einsum('PQ,Qmn->Pmn', transform, factor)
    changed = replace(cache, factors=factors)
    options = dict(spin_resolved=True, memory_byte_cap=16*1024**2, workspace_byte_cap=4096)
    reference = mdf._build_mdf_jk(cache, system, orbital, auxiliary, points, density, [.3, .7], **options)
    actual = mdf._build_mdf_jk(changed, system, orbital, auxiliary, points, density, [.3, .7], **options)
    np.testing.assert_allclose(actual.coulomb, reference.coulomb, atol=1e-11, rtol=1e-12)
    np.testing.assert_allclose(actual.exchange, reference.exchange, atol=1e-11, rtol=1e-12)
    assert actual.coulomb_energy == pytest.approx(reference.coulomb_energy, abs=1e-11)
    assert actual.exchange_energy == pytest.approx(reference.exchange_energy, abs=1e-11)


@pytest.mark.parametrize('spin_resolved', [False, True])
def test_fixed_density_energy_response_matches_returned_jk(spin_resolved):
    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    density = spins if spin_resolved else spins.sum(axis=0)
    direction = np.ones_like(density)
    weights = np.array([.3, .7])
    options = dict(spin_resolved=spin_resolved, memory_byte_cap=16*1024**2, workspace_byte_cap=4096)
    def evaluate(d):
        return mdf._build_mdf_jk(cache, system, orbital, auxiliary, points, d, weights, **options)
    result = evaluate(density)
    step = 1e-5
    plus, minus = evaluate(density+step*direction), evaluate(density-step*direction)
    derivative = (plus.coulomb_energy+plus.exchange_energy
                  - minus.coulomb_energy-minus.exchange_energy)/(2*step)
    channels = direction if spin_resolved else direction[None]
    expected = sum(
        weights[i]*np.trace(change[i]@(result.coulomb[i]-(1. if spin_resolved else .5)*k)).real
        for change, matrices in zip(channels, result.exchange)
        for i, k in zip(result.bra_rows, matrices)
    )
    assert derivative == pytest.approx(expected, abs=1e-6, rel=1e-8)


def _projection_gradient_fixture(displacements=None):
    centers = np.array([[.2, -.1, .3], [1.7, .4, -.2]])
    if displacements is not None:
        centers += displacements
    system = core.PeriodicSystem(3, np.eye(3)*7., [core.Atom(2, p) for p in centers])
    molecule = system.unit_cell_molecule()
    def basis(angular, exponents):
        shells = [core.ShellInfo(atom, l, True, [exponent], [1.], centers[atom])
                  for atom, (l, exponent) in enumerate(zip(angular, exponents))]
        return core.BasisSet(molecule, shells, 'mdf-projection-gradient-witness', True)
    return system, basis((0, 1), (.6, .4)), basis((1, 0), (.7, .5))


@pytest.mark.parametrize('component', ['metric', 'tensor', 'factor'])
@pytest.mark.parametrize('gamma', [False, True])
def test_pw_weighted_derivative_matches_its_value_source(component, gamma):
    system, orbital, auxiliary = _projection_gradient_fixture()
    q = np.zeros(3) if gamma else np.array([.13, -.07, .02])
    kets = np.array([[.0, .0, .0], [.21, -.13, .07]])+q
    vectors = np.array([[0., 0., 0.], [2*np.pi/7, 0., 0.], [-2*np.pi/7, 0., 0.]])+q
    rng = np.random.default_rng(852)
    def weight(shape, active, order='C'):
        array = rng.normal(size=shape)+1j*rng.normal(size=shape)
        return np.array(array if active else np.zeros(shape), dtype=complex, order=order)
    wm = weight((4, 4), component == 'metric', 'F')
    wt = weight((2, 4, 4, 4), component == 'tensor')
    wf = weight((2, 3, 4, 4), component == 'factor')
    analytic = core._compute_gdf_plane_wave_projection_gradient_weighted(
        orbital, auxiliary, system, q, kets, vectors, 7.2, wm, wt, wf,
        48, 8*1024**2, 100000,
    )
    def value(displacements):
        shifted_system, shifted_orbital, shifted_auxiliary = _projection_gradient_fixture(displacements)
        m, t, f, _ = core._compute_gdf_plane_wave_projection(
            shifted_orbital, shifted_auxiliary, shifted_system, q, kets, vectors,
            7.2, 1024**2, 8*1024**2, 100000, True,
        )
        return np.real(np.sum(wm*m)+np.sum(wt*t)+np.sum(wf*f))
    step = 2e-5
    numerical = np.zeros((2, 3))
    for atom in range(2):
        for axis in range(3):
            shift = np.zeros((2, 3))
            shift[atom, axis] = step
            numerical[atom, axis] = (value(shift)-value(-shift))/(2*step)
    np.testing.assert_allclose(analytic, numerical, atol=2e-7, rtol=2e-7)
    if component != 'factor':
        # M and T are invariant when all centers move together. F carries
        # its PW phase, so a general unconjugated F weight is not invariant.
        np.testing.assert_allclose(analytic.sum(axis=0), 0., atol=2e-10)


def test_pw_factor_derivative_keeps_global_vector_indices_across_panels():
    system, orbital, auxiliary = _projection_gradient_fixture()
    q, kets = np.array([.13, -.07, .02]), np.array([[.21, -.13, .07]])
    vector = np.array([[.13, -.07, .02]])
    wm = np.zeros((4, 4), complex, order='F')
    wt = np.zeros((1, 4, 4, 4), complex)
    wf = np.full((1, 1, 4, 4), .3+.7j)
    def derivative(vectors, weights):
        return core._compute_gdf_plane_wave_projection_gradient_weighted(
            orbital, auxiliary, system, q, kets, vectors, 7.2, wm, wt, weights,
            48, 8*1024**2, 100000,
        )
    expected = derivative(vector, wf)
    many = np.repeat(vector, 130, axis=0)
    weights = np.zeros((1, 130, 4, 4), complex)
    weights[:, -1:] = wf
    np.testing.assert_allclose(derivative(many, weights), expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(derivative(np.zeros((1, 3)), wf), np.zeros((2, 3)))


@pytest.mark.parametrize('invalid', ['output', 'workspace', 'shape', 'nonfinite', 'dtype', 'layout'])
def test_pw_derivative_rejects_invalid_storage_and_weights(invalid):
    system, orbital, auxiliary = _projection_gradient_fixture()
    wm = np.zeros((4, 4), complex, order='F')
    wt = np.zeros((1, 4, 4, 4), complex)
    wf = np.zeros((1, 2, 4, 4), complex)
    output, workspace = 48, 8*1024**2
    if invalid == 'output':
        output = 1
    elif invalid == 'workspace':
        workspace = 1
    elif invalid == 'shape':
        wf = wf[:, :1].copy()
    elif invalid == 'nonfinite':
        wf[0, 0, 0, 0] = np.nan
    elif invalid == 'dtype':
        wf = wf.real.copy()
    else:
        wf = np.asfortranarray(wf)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        core._compute_gdf_plane_wave_projection_gradient_weighted(
            orbital, auxiliary, system, np.zeros(3), np.zeros((1, 3)),
            np.array([[0., 0., 0.], [2*np.pi/7, 0., 0.]]), 7.2, wm, wt, wf,
            output, workspace, 100000,
        )


def test_pw_census_workspace_is_admitted_before_enumeration(monkeypatch):
    system, orbital, auxiliary = _fixture()
    def forbidden(*args, **kwargs):
        pytest.fail('PW census started without admitting its workspace')
    monkeypatch.setattr(mdf, '_vectors', forbidden)
    with pytest.raises(_RangeSeparatedGdfAdmissionError, match='PW census'):
        mdf._build_mdf_cache(system, orbital, auxiliary, np.zeros((1, 3)), True,
                             **dict(_options(), memory_byte_cap=8192))


def _response_batch(metric, tensor, pw, threshold=.1):
    values, vectors = np.linalg.eigh(metric)
    keep = values > threshold
    whitener = vectors[:, keep].conj().T/np.sqrt(values[keep])[:, None]
    gaussian = np.einsum('PQ,kQmn->kPmn', whitener, tensor)
    factors = np.concatenate((gaussian, pw), axis=1)
    source_key = ('synthetic-response', (0., 0., 0.), .8, 7.2, 14., 8., .4, threshold, 0.)
    state = mdf._MdfFitState(tensor, vectors, keep, np.zeros((1, 3)),
                             np.zeros((pw.shape[1], 3)), pw, np.zeros((len(tensor), 3)),
                             source_key, 100000)
    return mdf._MdfBatch(factors, np.zeros(3), values, int(keep.sum()), pw.shape[1],
                         1, 1024**2, state)


@pytest.mark.parametrize('coulomb_scale,exchange_scale', [(1., 0.), (0., 1.), (1., 2.)])
def test_mdf_response_matches_raw_metric_tensor_and_pw_energy_variations(coulomb_scale, exchange_scale):
    rng = np.random.default_rng(9254)
    def complex_array(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    rotation = np.linalg.qr(complex_array((3, 3)))[0]
    metric = (rotation*np.array([.03, 1.4, 2.1]))@rotation.conj().T
    tensor, pw = complex_array((2, 3, 2, 2)), complex_array((2, 2, 2, 2))
    raw = complex_array((2, 2, 2))
    density = raw@raw.conj().swapaxes(-1, -2)
    weights, pairs = np.array([.3, .7]), [(0, 0), (1, 1)]
    batch = _response_batch(metric, tensor, pw)
    wm, wt, wf = mdf._mdf_jk_response_weights(
        batch, pairs, density, weights, workspace_byte_cap=16*1024**2,
        coulomb_scale=coulomb_scale, exchange_scale=exchange_scale,
    )
    raw_dm = complex_array((3, 3))
    dm = .5*(raw_dm+raw_dm.conj().T)
    dt, df = complex_array(tensor.shape), complex_array(pw.shape)
    def energy(m, t, f):
        fitted = _response_batch(m, t, f).factors
        rho = sum(w*np.einsum('Pmn,mn->P', l.conj(), d)
                  for w, l, d in zip(weights, fitted, density))
        ej = .5*coulomb_scale*np.vdot(rho, rho).real
        ek = -.25*exchange_scale*sum(
            weights[i]*weights[j]*sum(np.trace(density[i]@l@density[j]@l.conj().T).real for l in factor)
            for (i, j), factor in zip(pairs, fitted)
        )
        return ej+ek
    step = 1e-5
    numerical = (energy(metric+step*dm, tensor+step*dt, pw+step*df)
                 - energy(metric-step*dm, tensor-step*dt, pw-step*df))/(2*step)
    analytic = np.real(np.sum(wm*dm)+np.sum(wt*dt)+np.sum(wf*df))
    assert analytic == pytest.approx(numerical, abs=2e-6, rel=2e-7)
    assert batch.gaussian_rank == 2  # Dropped-mode projector response is exercised.


def test_subdivided_mdf_response_uses_both_full_hartree_sources():
    metric = np.diag([.03, 1.4, 2.1]).astype(complex)
    rng = np.random.default_rng(136)
    tensor = rng.normal(size=(2, 3, 2, 2))+1j*rng.normal(size=(2, 3, 2, 2))
    pw = rng.normal(size=(2, 2, 2, 2))+1j*rng.normal(size=(2, 2, 2, 2))
    density = np.array([[[1., .2j], [-.2j, .4]], [[.8, -.1j], [.1j, .6]]])
    weights = np.array([.3, .7])
    full = mdf._mdf_jk_response_weights(
        _response_batch(metric, tensor, pw), [(0, 0), (1, 1)], density, weights,
        workspace_byte_cap=16*1024**2,
    )
    sources = tuple(sum(w*np.einsum('Pmn,nm->P', t, d) for w, t, d in zip(weights, array, density))
                    for array in (tensor, pw))
    parts = [mdf._mdf_jk_response_weights(
        _response_batch(metric, tensor[i:i+1], pw[i:i+1]), [(i, i)], density, weights,
        workspace_byte_cap=16*1024**2, coulomb_sources=sources, include_coulomb_metric=i == 0,
    ) for i in range(2)]
    np.testing.assert_allclose(sum(p[0] for p in parts), full[0], atol=1e-11, rtol=1e-12)
    for index in (1, 2):
        np.testing.assert_allclose(np.concatenate([p[index] for p in parts]), full[index], atol=1e-11, rtol=1e-12)
    with pytest.raises(ValueError, match='every q=0 diagonal'):
        mdf._mdf_jk_response_weights(
            _response_batch(metric, tensor[:1], pw[:1]), [(0, 0)], density, weights,
            workspace_byte_cap=16*1024**2,
        )


def test_response_retention_reuses_the_same_residual_eigensystem():
    system, orbital, auxiliary = _fixture()
    points = np.array([[0., 0., 0.], [.13, -.07, .02]])
    state = {}
    batches = [mdf._build_mdf_shared_q(
        system, orbital, auxiliary, point[None], np.zeros(3), _metric_state=state,
        retain_fit_state=True, **_options(),
    ) for point in points]
    assert batches[0].fit_state.eigenvectors is batches[1].fit_state.eigenvectors
    for batch in batches:
        assert np.shares_memory(batch.fit_state.pw_factors, batch.factors)
        assert not batch.fit_state.three_center.flags.writeable
        assert not batch.fit_state.eigenvectors.flags.writeable


@pytest.mark.slow
@pytest.mark.parametrize('spin_resolved', [False, True])
def test_mdf_cache_fixed_density_gradient_matches_displaced_energy(spin_resolved):
    system, orbital, auxiliary = _projection_gradient_fixture()
    points, weights = np.array([[0., 0., 0.], [.13, -.07, .02]]), np.array([.3, .7])
    rng = np.random.default_rng(994)
    raw = rng.normal(size=(2, 2, 4, 4))+1j*rng.normal(size=(2, 2, 4, 4))
    spins = raw@raw.conj().swapaxes(-1, -2)*.05
    density = spins if spin_resolved else spins.sum(axis=0)
    options = dict(_options(), pair_cutoff=7.2, auxiliary_cutoff=14., ke_cutoff=8.,
                   plane_wave_cutoff=.4, memory_byte_cap=512*1024**2)
    cache = mdf._build_mdf_cache(system, orbital, auxiliary, points, True, **options)
    gradient = mdf._compute_mdf_cache_gradient(
        cache, system, orbital, auxiliary, points, density, weights,
        spin_resolved=spin_resolved, memory_byte_cap=512*1024**2,
        native_workspace_byte_cap=64*1024**2,
    )
    direction = np.array([[.1, -.2, .3], [-.2, .1, .4]])
    def energy(step):
        s, o, a = _projection_gradient_fixture(step*direction)
        fitted = mdf._build_mdf_cache(s, o, a, points, True, **options)
        result = mdf._build_mdf_jk(fitted, s, o, a, points, density, weights,
                                  spin_resolved=spin_resolved, memory_byte_cap=512*1024**2)
        return result.coulomb_energy+result.exchange_energy
    step = 2e-5
    numerical = (energy(step)-energy(-step))/(2*step)
    assert np.sum(gradient*direction) == pytest.approx(numerical, abs=2e-6, rel=2e-6)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-8)


def test_cache_gradient_refuses_partial_exchange_before_rebuild(monkeypatch):
    from dataclasses import replace

    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    cache = replace(cache, source_parameters=('mdf-sr-lr-pw-v1', .4, ()),
                     factors={p: f for p, f in cache.items() if p != (1, 0)})
    def forbidden(*args, **kwargs):
        pytest.fail('partial exchange reached source rebuild')
    monkeypatch.setattr(mdf, '_build_mdf_shared_q', forbidden)
    with pytest.raises(ValueError, match='partial exchange'):
        mdf._compute_mdf_cache_gradient(
            cache, system, orbital, auxiliary, points, spins, [.3, .7], spin_resolved=True,
            memory_byte_cap=512*1024**2, native_workspace_byte_cap=64*1024**2,
        )


def test_gradient_assembly_subtracts_pw_projection_but_adds_direct_pw_response(monkeypatch):
    from vibeqc.aux_basis import _range_separated_gdf_source_signature

    system, orbital, auxiliary = _projection_gradient_fixture()
    q = np.array([.13, -.07, .02])
    points = np.array([np.zeros(3), q])
    source_key = (_range_separated_gdf_source_signature(system, orbital, auxiliary),
                  tuple(q), .8, 7.2, 14., 8., .4, .1, 0.)
    state = mdf._MdfFitState(np.zeros((1, 4, 4, 4), complex), np.eye(4, dtype=complex),
                             np.ones(4, bool), q[None], q[None],
                             np.zeros((1, 1, 4, 4), complex), q[None], source_key, 100000)
    batch = mdf._MdfBatch(np.zeros((1, 5, 4, 4), complex), q, np.ones(4), 4, 1, 1, 1024, state)
    wm = np.ones((4, 4), complex, order='F')
    wt, wf = np.full((1, 4, 4, 4), 2.+0j), np.full((1, 1, 4, 4), 3.+0j)
    monkeypatch.setattr(mdf, '_mdf_jk_response_weights', lambda *args, **kwargs: (wm, wt, wf))
    calls = []
    def sr(*args):
        calls.append('SR/LR')
        np.testing.assert_array_equal(args[9], 1.)
        np.testing.assert_array_equal(args[10], 2.)
        return np.ones((2, 3))
    def pw(*args):
        calls.append('PW')
        np.testing.assert_array_equal(args[7], -1.)
        np.testing.assert_array_equal(args[8], -2.)
        np.testing.assert_array_equal(args[9], 3.)
        return np.full((2, 3), 2.)
    monkeypatch.setattr(core, 'compute_gdf_range_separated_gradient_weighted', sr)
    monkeypatch.setattr(core, '_compute_gdf_plane_wave_projection_gradient_weighted', pw)
    result = mdf._compute_mdf_jk_gradient(
        system, orbital, auxiliary, batch, [(0, 1)], points, np.zeros((2, 4, 4)), [.5, .5],
        memory_byte_cap=16*1024**2, native_workspace_byte_cap=1024**2, coulomb_scale=0.,
    )
    assert calls == ['SR/LR', 'PW']
    np.testing.assert_array_equal(result, 3.)


def test_streamed_gradient_releases_previous_batch_before_rebuilding(monkeypatch):
    import weakref

    system, orbital, auxiliary = _fixture()
    points, weights = np.array([[0., 0., 0.], [.13, -.07, .02]]), [.3, .7]
    cache = mdf._build_mdf_cache(system, orbital, auxiliary, points, True, **_options())
    original, previous, widths = mdf._build_mdf_shared_q, [], []
    def single(*args, **kwargs):
        if len(args[3]) > 1:
            raise _RangeSeparatedGdfAdmissionError('forced single-pair admission')
        assert not previous or previous[-1]() is None
        batch = original(*args, **kwargs)
        previous.append(weakref.ref(batch.factors))
        widths.append(len(args[3]))
        return batch
    monkeypatch.setattr(mdf, '_build_mdf_shared_q', single)
    monkeypatch.setattr(mdf, '_compute_mdf_jk_gradient', lambda *args, **kwargs: np.zeros((1, 3)))
    mdf._compute_mdf_cache_gradient(
        cache, system, orbital, auxiliary, points, np.ones((2, 1, 1)), weights,
        memory_byte_cap=512*1024**2, native_workspace_byte_cap=64*1024**2,
    )
    assert len(widths) == 6  # Two off-diagonal batches and two passes over two diagonals.
    assert all(ref() is None for ref in previous)


@pytest.mark.parametrize('spin_resolved', [False, True])
@pytest.mark.parametrize('alpha', [0., .25, 1.])
def test_mean_field_assembly_composes_mdf_fit_once_with_existing_terms(monkeypatch, spin_resolved, alpha):
    from vibeqc import periodic_gdf_gradient as gradient

    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    weights = [.3, .7]
    overlaps = [np.eye(2, dtype=complex) for _ in points]
    energy_weighted = [np.zeros((2, 2), complex) for _ in points]
    oneel, gauge, ecp, ecp_options = object(), object(), object(), object()
    calls = []
    def fitted(cache_arg, system_arg, orbital_arg, channels, weights_arg, points_arg, **kwargs):
        assert cache_arg is cache and system_arg is system and orbital_arg is orbital
        assert len(channels) == (2 if spin_resolved else 1)
        np.testing.assert_allclose(channels, spins if spin_resolved else spins.sum(axis=0)[None])
        assert kwargs['alpha_hf'] == alpha
        calls.append('MDF')
        return np.full((1, 3), 2.)
    def one_electron(*args, **kwargs):
        np.testing.assert_allclose(args[2], spins.sum(axis=0))
        assert args[3] is energy_weighted
        assert kwargs['oneel_lat_opts'] is oneel and kwargs['gauge_lat_opts'] is gauge
        assert kwargs['ecp_context'] is ecp and kwargs['ecp_lat_opts'] is ecp_options
        assert kwargs['v_ne_ke_cutoff'] == 8.
        calls.append('one-electron')
        return np.ones((1, 3))
    def shift(*args):
        assert args[6] == alpha and args[7] == .2 and args[8] is oneel
        calls.append('Madelung')
        return np.full((1, 3), 3.)
    def legacy(*args, **kwargs):
        pytest.fail('private MDF cache reached a legacy GDF fit derivative')
    monkeypatch.setattr(mdf, '_compute_mdf_mean_field_fit_gradient', fitted)
    monkeypatch.setattr(gradient, '_compute_oneel_w_gradient_multik', one_electron)
    monkeypatch.setattr(gradient, '_compute_exxdiv_w_gradient_multik', shift)
    monkeypatch.setattr(gradient, '_compute_j_gradient_multik_rsgdf', legacy)
    monkeypatch.setattr(gradient, '_compute_k_gradient_multik_rsgdf', legacy)
    options = dict(alpha_hf=alpha, madelung=.2, oneel_lat_opts=oneel, gauge_lat_opts=gauge,
                   v_ne_ke_cutoff=8., ecp_context=ecp, ecp_lat_opts=ecp_options)
    if spin_resolved:
        result = gradient._compute_kuhf_gradient_multik(
            system, orbital, spins[0], spins[1], energy_weighted, overlaps, weights, points, cache, **options,
        )
    else:
        result = gradient._compute_krhf_gradient_multik(
            system, orbital, spins.sum(axis=0), energy_weighted, overlaps, weights, points, cache, **options,
        )
    assert calls[:2] == ['MDF', 'one-electron']
    assert calls.count('MDF') == 1
    assert calls.count('Madelung') == ((2 if spin_resolved else 1) if alpha else 0)
    np.testing.assert_array_equal(result, 3.+((12. if spin_resolved else 3.) if alpha else 0.))


@pytest.mark.parametrize('spin_resolved', [False, True])
def test_mean_field_adapter_preserves_spin_weights_and_source_controls(monkeypatch, spin_resolved):
    from dataclasses import replace

    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    cache = replace(cache, aux_basis=auxiliary, memory_byte_cap=512*1024**2,
                     native_workspace_byte_cap=64*1024**2)
    channels = tuple(list(channel) for channel in spins) if spin_resolved else (list(spins.sum(axis=0)),)
    def capture(*args, **kwargs):
        assert args[0] is cache and args[3] is auxiliary
        np.testing.assert_array_equal(args[4], points)
        np.testing.assert_allclose(args[5], spins if spin_resolved else spins.sum(axis=0))
        assert args[5].dtype == np.complex128
        assert kwargs['spin_resolved'] is spin_resolved
        assert kwargs['coulomb_scale'] == 1. and kwargs['exchange_scale'] == .25
        assert kwargs['native_workspace_byte_cap'] == cache.native_workspace_byte_cap
        assert kwargs['memory_byte_cap'] < cache.memory_byte_cap
        return np.ones((1, 3))
    monkeypatch.setattr(mdf, '_compute_mdf_cache_gradient', capture)
    mdf._compute_mdf_mean_field_fit_gradient(cache, system, orbital, channels, [.3, .7], points, alpha_hf=.25)


def test_mean_field_adapter_admits_packing_before_allocating(monkeypatch):
    from dataclasses import replace

    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    cache = replace(cache, aux_basis=auxiliary, memory_byte_cap=8192,
                     native_workspace_byte_cap=4096)
    channels = tuple(list(channel) for channel in spins)
    def forbidden(*args, **kwargs):
        pytest.fail('unadmitted density packing started')
    monkeypatch.setattr(mdf.np, 'empty', forbidden)
    with pytest.raises(_RangeSeparatedGdfAdmissionError, match='packing'):
        mdf._compute_mdf_mean_field_fit_gradient(cache, system, orbital, channels, [.3, .7], points, alpha_hf=1.)


def test_public_multik_mdf_gradient_remains_gated():
    from vibeqc.periodic_k_gdf import _reject_unsupported_multik_gradient

    with pytest.raises(NotImplementedError, match="gdf_method='rsgdf' only"):
        _reject_unsupported_multik_gradient(
            'MDF integration witness', dim=3, gdf_method='mdf', smearing_temperature=0.,
            k_exchange='gdf', screened_omega=None, functional_is_range_separated=False,
            weights=np.array([.5, .5]),
        )


def _mean_field_fixture():
    from dataclasses import replace

    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    cache = replace(cache, aux_basis=auxiliary)
    hcore = np.array([[[1., .2+.3j], [.2-.3j, -.4]],
                      [[-.3, .1-.2j], [.1+.2j, .8]]])
    overlap = np.array([[[1.1, .1+.2j], [.1-.2j, 1.2]],
                        [[.9, -.1j], [.1j, 1.3]]])
    return system, orbital, points, cache, spins, hcore, overlap


@pytest.mark.parametrize('spin_resolved', [False, True])
@pytest.mark.parametrize('alpha', [0., .25, 1.])
def test_mean_field_energy_has_physical_fock_density_derivative(spin_resolved, alpha):
    from vibeqc.madelung import exxdiv_ewald_energy_shift

    system, orbital, points, cache, spins, hcore, overlap = _mean_field_fixture()
    density = spins if spin_resolved else spins.sum(axis=0)
    weights, xi, gamma = np.array([.3, .7]), .17, .09
    # A nonlinear model XC functional makes double-counting and stale Vxc
    # visible. It is a variational algebra witness, not a physical functional.
    def evaluate(d):
        channels = d if spin_resolved else d[None]
        exc = .5*gamma*sum(w*np.trace(matrix@matrix).real
                          for channel in channels for w, matrix in zip(weights, channel))
        return mdf._evaluate_mdf_mean_field(
            cache, system, orbital, points, d, weights, hcore, overlap,
            nuclear_energy=1.3, memory_byte_cap=16*1024**2, workspace_byte_cap=4096,
            spin_resolved=spin_resolved, alpha_hf=alpha, madelung=xi,
            xc_potential=gamma*d, xc_energy=exc,
        )
    result = evaluate(density)
    channels = density if spin_resolved else density[None]
    assert result.cache is cache
    assert result.jk.bra_rows == (tuple(range(2)) if alpha else ())
    assert result.reserved_peak_bytes <= 16*1024**2
    expected_e1 = sum(w*np.trace(d@h).real for channel in channels
                      for w, d, h in zip(weights, channel, hcore))
    expected_shift = (2 if spin_resolved else 1)*sum(
        exxdiv_ewald_energy_shift(channel, overlap, xi, hf_exchange_fraction=alpha, weights=weights)
        for channel in channels
    )
    assert result.one_electron_energy == pytest.approx(expected_e1)
    assert result.exxdiv_energy == pytest.approx(expected_shift)
    assert result.electronic_energy == pytest.approx(
        expected_e1+result.jk.coulomb_energy+result.exchange_energy+expected_shift+result.xc_energy)
    assert result.total_energy == pytest.approx(result.electronic_energy+1.3)
    for spin, channel in enumerate(channels):
        for k, d in enumerate(channel):
            expected = hcore[k]+result.jk.coulomb[k]+gamma*d
            if alpha:
                expected -= alpha*(1. if spin_resolved else .5)*(result.jk.exchange[spin][k]+xi*overlap[k]@d@overlap[k])
            np.testing.assert_allclose(result.focks[spin][k], expected, atol=1e-11)
            assert not result.focks[spin][k].flags.writeable
    # Independent Hermitian variations probe every spin and k separately,
    # including imaginary off-diagonals and nonuniform mesh weights.
    direction = np.array([[.2, .13+.27j], [.13-.27j, -.1]])
    step = 2e-5
    for spin in range(len(channels)):
        for k in range(2):
            plus, minus = density.copy(), density.copy()
            target = (spin, k) if spin_resolved else k
            plus[target] += step*direction
            minus[target] -= step*direction
            difference = (evaluate(plus).total_energy-evaluate(minus).total_energy)/(2*step)
            derivative = weights[k]*np.trace(direction@result.focks[spin][k]).real
            assert difference == pytest.approx(derivative, abs=2e-7, rel=1e-8)


def test_mean_field_restricted_and_balanced_spin_hf_are_equivalent():
    system, orbital, points, cache, spins, hcore, overlap = _mean_field_fixture()
    density = spins.sum(axis=0)
    options = dict(nuclear_energy=.4, memory_byte_cap=16*1024**2,
                   workspace_byte_cap=4096, madelung=.23)
    restricted = mdf._evaluate_mdf_mean_field(
        cache, system, orbital, points, density, [.3, .7], hcore, overlap, **options)
    unrestricted = mdf._evaluate_mdf_mean_field(
        cache, system, orbital, points, np.stack([density/2, density/2]), [.3, .7],
        hcore, overlap, spin_resolved=True, **options)
    assert restricted.total_energy == pytest.approx(unrestricted.total_energy, abs=1e-11)
    for channel in unrestricted.focks:
        np.testing.assert_allclose(channel, restricted.focks[0], atol=1e-11)


@pytest.mark.parametrize('invalid', ['hcore', 'overlap', 'xc', 'xc_energy', 'xc_missing',
                                      'alpha', 'nuclear', 'memory'])
def test_mean_field_rejects_invalid_assembly_before_jk(monkeypatch, invalid):
    system, orbital, points, cache, spins, hcore, overlap = _mean_field_fixture()
    density = spins.sum(axis=0)
    options = dict(nuclear_energy=.4, memory_byte_cap=16*1024**2,
                   workspace_byte_cap=4096, alpha_hf=.25,
                   xc_potential=np.zeros_like(density), xc_energy=0.)
    if invalid == 'hcore':
        hcore[0, 0, 1] += .1
    elif invalid == 'overlap':
        overlap[0, 0, 0] = np.nan
    elif invalid == 'xc':
        options['xc_potential'] = np.zeros((2, 2))
    elif invalid == 'xc_energy':
        options['xc_energy'] = np.inf
    elif invalid == 'xc_missing':
        options['xc_potential'] = options['xc_energy'] = None
    elif invalid == 'alpha':
        options['alpha_hf'] = 1.1
    elif invalid == 'nuclear':
        options['nuclear_energy'] = .3j
    else:
        options['memory_byte_cap'] = cache.retained_cache_bytes+density.nbytes
    def forbidden(*args, **kwargs):
        pytest.fail('invalid mean-field assembly reached J/K construction')
    monkeypatch.setattr(mdf, '_build_mdf_jk', forbidden)
    with pytest.raises((ValueError, _RangeSeparatedGdfAdmissionError)):
        mdf._evaluate_mdf_mean_field(
            cache, system, orbital, points, density, [.3, .7], hcore, overlap, **options)


def test_mean_field_hartree_only_accepts_diagonal_cache(monkeypatch):
    from dataclasses import replace
    from vibeqc import aux_basis

    system, orbital, points, cache, spins, hcore, overlap = _mean_field_fixture()
    density = spins.sum(axis=0)
    cache = replace(cache, factors={p: f for p, f in cache.items() if p[0] == p[1]},
                    need_k_pairs=False, bra_rows=())
    def forbidden(*args, **kwargs):
        pytest.fail('zero-exchange mean-field assembly attempted exchange')
    monkeypatch.setattr(aux_basis, '_accumulate_exchange_from_factors', forbidden)
    result = mdf._evaluate_mdf_mean_field(
        cache, system, orbital, points, density, [.3, .7], hcore, overlap,
        nuclear_energy=.4, memory_byte_cap=16*1024**2, workspace_byte_cap=4096,
        alpha_hf=0., madelung=.23, xc_potential=np.zeros_like(density), xc_energy=0.)
    assert result.exchange_energy == result.exxdiv_energy == 0.
    assert result.jk.exchange == ((),)


def test_mean_field_admits_complete_phase_before_contraction(monkeypatch):
    from vibeqc import aux_basis

    system, orbital, points, cache, spins, hcore, overlap = _mean_field_fixture()
    density = spins.sum(axis=0)
    args = (cache, system, orbital, points, density, [.3, .7], hcore, overlap)
    options = dict(nuclear_energy=.4, workspace_byte_cap=4096, madelung=.23)
    admitted = mdf._evaluate_mdf_mean_field(*args, memory_byte_cap=16*1024**2, **options)
    def forbidden(*args, **kwargs):
        pytest.fail('unadmitted complete mean-field phase started Coulomb contraction')
    monkeypatch.setattr(aux_basis, '_build_coulomb_from_diagonal_factors', forbidden)
    with pytest.raises(_RangeSeparatedGdfAdmissionError, match='memory cap'):
        mdf._evaluate_mdf_mean_field(
            *args, memory_byte_cap=admitted.reserved_peak_bytes-1, **options)


def test_private_scf_source_receives_planned_domains_and_full_memory_cap(monkeypatch):
    from types import SimpleNamespace
    from vibeqc import aux_basis, periodic_k_gdf as driver

    system, orbital, auxiliary, points, _, _ = _operator_fixture()
    monkeypatch.setattr(aux_basis, '_plan_range_separated_gdf_cutoffs', lambda *args, **kwargs:
                        SimpleNamespace(pair_cutoff=7., auxiliary_cutoff=9., ke_cutoff=8., integral_screen_error=1e-13))
    received, sentinel = {}, object()
    def build(*args, **kwargs):
        assert args[0] is system and args[1] is orbital and args[2] is auxiliary
        np.testing.assert_array_equal(args[3], points)
        assert args[4] is True
        received.update(kwargs)
        return sentinel
    def forbidden(*args, **kwargs):
        pytest.fail('private MDF injection evaluated a Gaussian-only pair source')
    monkeypatch.setattr(mdf, '_build_mdf_cache', build)
    monkeypatch.setattr(aux_basis, '_build_lpq_range_separated_shared_q', forbidden)
    options = dict(omega=.8, pair_cutoff=6., auxiliary_cutoff=10., ke_cutoff=7.,
                   raw_integral_error=1e-9, linear_dep_thr=1e-8,
                   memory_byte_cap=128*1024**2, native_workspace_byte_cap=16*1024**2,
                   image_candidate_cap=10000, reciprocal_candidate_cap=20000,
                   integral_screen_error=1e-11)
    result = driver._build_range_separated_lpq_cache(
        system, orbital, auxiliary, points, True,
        pair_cache_builder=mdf._MdfScfSource(.7), **options)
    assert result is sentinel
    assert (received['pair_cutoff'], received['auxiliary_cutoff'], received['ke_cutoff']) == (7., 10., 8.)
    assert received['integral_screen_error'] == 1e-13
    assert received['plane_wave_cutoff'] == .7
    for key in ('omega', 'linear_dep_thr', 'memory_byte_cap', 'native_workspace_byte_cap',
                'image_candidate_cap', 'reciprocal_candidate_cap'):
        assert received[key] == options[key]
    assert tuple(received['bra_rows']) == (0, 1)


@pytest.mark.parametrize('invalid', [None, 'source', 'mesh', 'rank'])
def test_private_mdf_gradient_handoff_preserves_energy_cache(monkeypatch, invalid):
    from dataclasses import replace
    from types import SimpleNamespace
    from vibeqc import periodic_k_gdf as driver, periodic_gdf_gradient as gradient

    system, orbital, _, cache, _, _, _ = _mean_field_fixture()
    points = np.asarray(cache.kpoints_cart)
    rank = cache[(0, 0)].shape[0]
    if invalid == 'source':
        cache = replace(cache, source_signature='wrong source')
    elif invalid == 'mesh':
        points = points[::-1]
    elif invalid == 'rank':
        rank += 1
    def forbidden(*args, **kwargs):
        pytest.fail('MDF gradient handoff rebuilt a legacy Gaussian-only cache')
    monkeypatch.setattr(gradient, '_build_multik_rsgdf_gradient_cache', forbidden)
    messages = []
    def handoff():
        return driver._build_multik_gradient_cache_checked(
            system, orbital, object(), points, rsgdf_ke_cutoff=200., rsgdf_tail_ke_cutoff=None,
            lat_opts=object(), gdf_linear_dep_threshold=1e-9, n_fit_scf=rank,
            entry='private MDF witness', plog=SimpleNamespace(info=messages.append), source_cache=cache)
    if invalid:
        with pytest.raises((ValueError, RuntimeError), match='private MDF gradient source differs'):
            handoff()
    else:
        assert handoff() is cache
        assert 'private MDF source parameters' in messages[0]


@pytest.mark.parametrize('spin_resolved', [False, True])
@pytest.mark.parametrize('invalid', ['dimension', 'method', 'exchange', 'ibz'])
def test_private_mdf_scf_refuses_unsupported_routes_before_setup(monkeypatch, spin_resolved, invalid):
    from types import SimpleNamespace
    from vibeqc import periodic_k_gdf as driver

    options = dict(gdf_method='rsgdf', k_exchange='gdf', ibz_native=False,
                   _lpq_cache_builder=mdf._MdfScfSource(.7))
    dimension = 3
    if invalid == 'dimension':
        dimension = 2
    elif invalid == 'method':
        options['gdf_method'] = 'mdf'
    elif invalid == 'exchange':
        options['k_exchange'] = 'cosx'
    else:
        options['ibz_native'] = True
    def forbidden(*args, **kwargs):
        pytest.fail('unsupported private MDF route reached driver setup')
    monkeypatch.setattr(driver, '_gdf_density_return_finalizer', forbidden)
    run = driver.run_kuhf_periodic_gdf if spin_resolved else driver.run_krhf_periodic_gdf
    with pytest.raises(NotImplementedError, match='private MDF SCF injection requires'):
        run(SimpleNamespace(dim=dimension), object(), **options)


@pytest.mark.parametrize('dimension,method', [(2, 'rsgdf'), (3, 'mdf'), (3, 'compcell')])
def test_private_mdf_rohf_refuses_unsupported_routes_before_setup(monkeypatch, dimension, method):
    from types import SimpleNamespace
    from vibeqc import periodic_rohf_gdf as driver

    def forbidden(*args, **kwargs):
        pytest.fail('unsupported private MDF route reached ROHF setup')
    monkeypatch.setattr(driver, '_gdf_density_return_finalizer', forbidden)
    with pytest.raises(NotImplementedError, match='private MDF SCF injection requires'):
        driver.run_krohf_periodic_gdf(
            SimpleNamespace(dim=dimension), object(), gdf_method=method,
            _lpq_cache_builder=mdf._MdfScfSource(.7))


@pytest.mark.slow
@pytest.mark.parametrize('reference,damping,iterations,converged', [
    ('rhf', 0., 1, False), ('uhf', 0., 1, False),
    ('rohf', 0., 1, False), ('rohf', .35, 3, False), ('rohf', .35, 2, True),
])
@pytest.mark.parametrize('mesh', [(1, 1, 1), (2, 1, 1)])
def test_private_mdf_scf_assembly_matches_evaluator(
        monkeypatch, reference, damping, iterations, converged, mesh):
    """Exercise real SCF assembly with synthetic factors; source acceptance is separate."""
    from dataclasses import replace
    from types import SimpleNamespace
    import vibeqc as vq
    from vibeqc import aux_basis, periodic_k_gdf as driver

    system, orbital, auxiliary, _, template, _ = _operator_fixture()
    spin_resolved = reference != 'rhf'
    if reference == 'rohf':
        from vibeqc import periodic_rohf_gdf as driver
        # Three electrons in two spatial orbitals: both spin channels
        # contribute, with one doubly and one singly occupied orbital.
        system.charge = -1
        system.multiplicity = 2
    built = []
    def build(cell, ao, aux, points, need_pairs, **kwargs):
        assert cell is system and ao is orbital and aux is auxiliary
        factors = {p: f for p, f in template.items() if max(p) < len(points)}
        controls = {k: v for k, v in kwargs.items()
                    if k not in ('plane_wave_cutoff', 'memory_byte_cap', 'bra_rows')}
        cache = replace(
            template, factors=factors, kpoints_cart=tuple(map(tuple, points)),
            aux_basis=aux, bra_rows=tuple(range(len(points))), need_k_pairs=need_pairs,
            retained_factor_bytes=sum(f.nbytes for f in factors.values()),
            source_parameters=('mdf-sr-lr-pw-v1', kwargs['plane_wave_cutoff'], tuple(sorted(controls.items()))),
            memory_byte_cap=kwargs['memory_byte_cap'],
            native_workspace_byte_cap=kwargs['native_workspace_byte_cap'])
        built.append(cache)
        return cache
    monkeypatch.setattr(mdf, '_build_mdf_cache', build)
    monkeypatch.setattr(driver, 'make_aux_basis_set', lambda *args, **kwargs: auxiliary)
    monkeypatch.setattr(aux_basis, '_plan_range_separated_gdf_cutoffs', lambda *args, **kwargs:
                        SimpleNamespace(pair_cutoff=6.2, auxiliary_cutoff=8., ke_cutoff=6., integral_screen_error=0.))
    opts = vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.max_iter = iterations
    # Loose tolerances exercise the converged return; zero tolerances
    # force the iteration-limit return. Neither is an acceptance gate.
    opts.conv_tol_energy = opts.conv_tol_grad = 1e100 if converged else 0.
    opts.use_diis = False
    opts.damping = damping
    opts.lattice_opts.cutoff_bohr = opts.lattice_opts.nuclear_cutoff_bohr = 6.2
    run = getattr(driver, f'run_k{reference}_periodic_gdf')
    result = run(system, orbital, kmesh=mesh, options=opts, aux_basis='def2-svp-jk',
                 gdf_method='rsgdf', rsgdf_ke_cutoff=6., rcut_strategy=None,
                 _lpq_cache_builder=mdf._MdfScfSource(.7), check_energy_sanity=False, progress=False)
    assert len(built) == 1
    assert result.backend.endswith('+private-mdf-unvalidated')
    assert result.rsgdf_ke_cutoff == 6.
    assert result.n_iter == iterations
    assert result.converged == converged
    if reference == 'rohf':
        assert (result.n_alpha, result.n_beta) == (2, 1)
    density = (np.asarray([result.density_alpha, result.density_beta]) if spin_resolved
               else np.asarray(result.density))
    evaluated = mdf._evaluate_mdf_mean_field(
        built[0], system, orbital, result.kpoints_cart, density, result.kpoint_weights,
        np.asarray(result.hcore), np.asarray(result.overlap),
        nuclear_energy=result.e_nuclear, memory_byte_cap=16*1024**2, workspace_byte_cap=4096,
        spin_resolved=spin_resolved, madelung=driver._madelung_for_kmesh(system, mesh))
    focks = [result.fock_alpha, result.fock_beta] if spin_resolved else [result.fock]
    np.testing.assert_allclose(evaluated.focks, focks, atol=2e-10, rtol=0.)
    assert evaluated.total_energy == pytest.approx(result.energy, abs=2e-10, rel=0.)
    assert evaluated.jk.coulomb_energy == pytest.approx(result.e_coulomb, abs=2e-10, rel=0.)
    assert evaluated.exchange_energy + evaluated.exxdiv_energy == pytest.approx(
        result.e_hf_exchange, abs=2e-10, rel=0.)
    if reference == 'rohf':
        from vibeqc.rohf import roothaan_effective_fock
        for i, overlap in enumerate(result.overlap):
            effective = roothaan_effective_fock(
                evaluated.focks[0][i], evaluated.focks[1][i],
                density[0, i], density[1, i], overlap)
            np.testing.assert_allclose(effective, result.fock[i], atol=2e-10, rtol=0.)
            c, eps, occ = result.mo_coeffs[i], result.mo_energies[i], result.mo_occupations[i]
            np.testing.assert_allclose(c.conj().T@overlap@c, np.eye(c.shape[1]), atol=2e-10, rtol=0.)
            np.testing.assert_allclose(c.conj().T@effective@c, np.diag(eps), atol=2e-10, rtol=0.)
            assert np.count_nonzero(occ == 2.) == 1
            assert np.count_nonzero(occ == 1.) == 1
            if iterations == 1:
                # The first accepted density came from Hcore. Returning
                # canonical effective eigenpairs must not replace it by
                # a refill of that new operator.
                beta_refill = c[:, occ == 2.]@c[:, occ == 2.].conj().T
                assert np.linalg.norm(beta_refill-density[1, i]) > 1e-8


def _lagrangian_fixture(*, spin_resolved=False, degenerate=False, truncated=False):
    n, ns = (3 if truncated else 2), (2 if spin_resolved else 1)
    unitary = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    c = np.zeros((n, 2), complex)
    c[:2] = np.diag([.8, 1.2])@unitary
    s = np.diag([1/.8**2, 1/1.2**2]+([1.] if truncated else []))
    energies = np.array([[-.8, -.8 if degenerate else .3], [-.5, -.5 if degenerate else .7]])
    densities, focks, coefficients, spectra, expected = [], [], [], [], np.zeros((2, n, n), complex)
    for spin in range(ns):
        ds, fs = [], []
        for k, e in enumerate(energies):
            p = np.diag([.7-.2*spin, .2+.1*k]).astype(complex)
            if degenerate:
                p[0, 1], p[1, 0] = .12j, -.12j
            ds.append(c@p@c.conj().T)
            f = s@c@np.diag(e)@c.conj().T@s
            if truncated:
                f[-1, -1] = 1.3
            fs.append(f)
            # Stationary P commutes with epsilon, including coherent
            # occupation matrices within a degenerate eigenspace.
            expected[k] += c@(p*e[None, :])@c.conj().T
        densities.append(ds)
        focks.append(fs)
        coefficients.append([c.copy() for _ in range(2)])
        spectra.append([e.copy() for e in energies])
    factors = {(k, k): np.zeros((1, n, n), complex) for k in range(2)}
    cache = mdf._MdfCache(factors, 'state-witness', (), ((0., 0., 0.), (.1, 0., 0.)), (),
                          sum(a.nbytes for a in factors.values()), 1024**2, memory_byte_cap=16*1024**2)
    return cache, densities, focks, [s.copy(), s.copy()], coefficients, spectra, expected


@pytest.mark.parametrize('spin_resolved', [False, True])
@pytest.mark.parametrize('degenerate,truncated', [(False, False), (True, False), (False, True)])
def test_mdf_overlap_lagrangian_uses_accepted_density_and_physical_fock(spin_resolved, degenerate, truncated):
    *args, expected = _lagrangian_fixture(spin_resolved=spin_resolved, degenerate=degenerate, truncated=truncated)
    result = mdf._build_mdf_scf_energy_weighted_density(*args, [.3, .7], stationarity_tolerance=1e-10)
    np.testing.assert_allclose(result, expected, atol=2e-14, rtol=0.)
    assert all(not w.flags.writeable for w in result)
    # Shifting the orbital energy origin must shift W by lambda D_total,
    # retaining the same accepted density and each spin's physical Fock.
    shift = .41
    cache, ds, fs, ss, cs, es = args
    shifted_f = [[f+shift*s for f, s in zip(channel, ss)] for channel in fs]
    shifted_e = [[e+shift for e in channel] for channel in es]
    shifted = mdf._build_mdf_scf_energy_weighted_density(
        cache, ds, shifted_f, ss, cs, shifted_e, [.3, .7], stationarity_tolerance=1e-10)
    np.testing.assert_allclose(shifted, expected+shift*np.sum(ds, axis=0), atol=2e-14, rtol=0.)


def test_mdf_overlap_lagrangian_keeps_small_fractional_occupations():
    cache, ds, fs, ss, cs, es, _ = _lagrangian_fixture()
    for k in range(2):
        ss[k] = cs[0][k] = np.eye(2)
        es[0][k] = np.array([1., 2.])
        fs[0][k] = np.diag(es[0][k])
        ds[0][k] = np.diag([1., 1e-16])
    result = mdf._build_mdf_scf_energy_weighted_density(
        cache, ds, fs, ss, cs, es, [.3, .7], stationarity_tolerance=1e-10)
    for w in result:
        assert w[1, 1].real == pytest.approx(2e-16, abs=1e-30, rel=0.)


@pytest.mark.parametrize('invalid,match', [
    ('orbital', 'S-orthonormal'), ('spectrum', 'eigenpairs'),
    ('density', 'Hermitian'), ('support', 'outside'), ('nonfinite', 'nonfinite'),
    ('stationarity', 'not stationary'), ('weights', 'normalized'),
])
def test_mdf_overlap_lagrangian_rejects_inconsistent_states(invalid, match):
    cache, ds, fs, ss, cs, es, _ = _lagrangian_fixture(truncated=True)
    weights = [.3, .7]
    if invalid == 'orbital':
        cs[0][0][0, 0] *= 1.1
    elif invalid == 'spectrum':
        es[0][0][0] += .1
    elif invalid == 'density':
        ds[0][0][0, 1] += .1j
    elif invalid == 'support':
        ds[0][0][-1, -1] = .1
    elif invalid == 'nonfinite':
        cs[0][0][0, 0] = np.nan
    elif invalid == 'stationarity':
        ds[0][0] += cs[0][0]@np.array([[0., .1], [.1, 0.]])@cs[0][0].conj().T
    else:
        weights = [.3, .3]
    with pytest.raises(ValueError, match=match):
        mdf._build_mdf_scf_energy_weighted_density(
            cache, ds, fs, ss, cs, es, weights, stationarity_tolerance=1e-10)


def test_mdf_overlap_lagrangian_admits_storage_before_matrix_work(monkeypatch):
    from dataclasses import replace

    cache, *args, _ = _lagrangian_fixture()
    cache = replace(cache, memory_byte_cap=cache.retained_cache_bytes)
    def forbidden(*args, **kwargs):
        pytest.fail('unadmitted MDF overlap Lagrangian started matrix allocation')
    monkeypatch.setattr(mdf.np, 'zeros', forbidden)
    with pytest.raises(_RangeSeparatedGdfAdmissionError, match='memory cap'):
        mdf._build_mdf_scf_energy_weighted_density(cache, *args, [.3, .7], stationarity_tolerance=1e-10)


def test_gradient_response_admission_counts_reciprocal_copies_before_integrals(monkeypatch):
    system, orbital, auxiliary = _fixture()
    # Large LR lists can dominate the response binding copy even with one
    # Gaussian and no PW factors. Only enumeration is faked in this witness.
    vectors = np.zeros((4000, 3), dtype=np.float64)
    monkeypatch.setattr(mdf, '_vectors', lambda *args, **kwargs:
                        np.empty((0, 3)) if kwargs.get('omit_zero') else vectors)
    def forbidden(*args, **kwargs):
        pytest.fail('unadmitted gradient response reached raw integral evaluation')
    monkeypatch.setattr(core, 'compute_gdf_range_separated_integrals', forbidden)
    options = _options()
    options.update(plane_wave_cutoff=0., native_workspace_byte_cap=4096)
    without_vectors = mdf._mdf_gradient_workspace_bytes(1, 1, 1, 2, 0, 0, 1, 4096)
    with pytest.raises(_RangeSeparatedGdfAdmissionError, match='reciprocal copies'):
        mdf._build_mdf_shared_q(
            system, orbital, auxiliary, np.zeros((1, 3)), np.zeros(3),
            retain_fit_state=True, gradient_response_byte_cap=without_vectors,
            gradient_nkpoints=2, **options)


def test_gradient_response_exact_budget_reaches_integrals(monkeypatch):
    system, orbital, auxiliary = _fixture()
    vectors = np.zeros((4000, 3), dtype=np.float64)
    monkeypatch.setattr(mdf, '_vectors', lambda *args, **kwargs:
                        np.empty((0, 3)) if kwargs.get('omit_zero') else vectors)
    class Admitted(Exception):
        pass
    def stop_at_integrals(*args, **kwargs):
        raise Admitted
    monkeypatch.setattr(core, 'compute_gdf_range_separated_integrals', stop_at_integrals)
    options = _options()
    options.update(plane_wave_cutoff=0., native_workspace_byte_cap=4096)
    cap = mdf._mdf_gradient_workspace_bytes(1, 1, 1, 2, 0, len(vectors), 1, 4096)
    with pytest.raises(Admitted):
        mdf._build_mdf_shared_q(
            system, orbital, auxiliary, np.zeros((1, 3)), np.zeros(3),
            retain_fit_state=True, gradient_response_byte_cap=cap,
            gradient_nkpoints=2, **options)


def test_gradient_response_reservation_admits_real_weights_and_bindings(monkeypatch):
    from vibeqc.aux_basis import _range_separated_gdf_source_signature

    system, orbital, auxiliary = _fixture()
    vectors = np.tile([.1, 0., 0.], (4000, 1))
    points = np.zeros((1, 3))
    key = (_range_separated_gdf_source_signature(system, orbital, auxiliary),
           (0., 0., 0.), .8, 6.2, 8., 6., .7, .1, 0.)
    state = mdf._MdfFitState(np.ones((1, 1, 1, 1), complex), np.eye(1, dtype=complex),
                             np.ones(1, bool), vectors, vectors[:2],
                             np.ones((1, 2, 1, 1), complex), points, key, 100000)
    batch = mdf._MdfBatch(np.ones((1, 3, 1, 1), complex), points[0], np.ones(1), 1, 2,
                          len(vectors), 1024**2, state)
    calls = []
    def native(*args):
        calls.append(1)
        return np.zeros((1, 3))
    monkeypatch.setattr(core, 'compute_gdf_range_separated_gradient_weighted', native)
    monkeypatch.setattr(core, '_compute_gdf_plane_wave_projection_gradient_weighted', native)
    cap = mdf._mdf_gradient_workspace_bytes(1, 1, 1, 1, 2, len(vectors), 1, 4096)
    args = (system, orbital, auxiliary, batch, [(0, 0)], points, np.ones((1, 1, 1)), [1.])
    with pytest.raises(_RangeSeparatedGdfAdmissionError, match='reciprocal copies'):
        mdf._compute_mdf_jk_gradient(*args, memory_byte_cap=cap-1, native_workspace_byte_cap=4096)
    assert calls == []
    result = mdf._compute_mdf_jk_gradient(*args, memory_byte_cap=cap, native_workspace_byte_cap=4096)
    assert calls == [1, 1]
    np.testing.assert_array_equal(result, 0.)


@pytest.mark.parametrize('allocator_failure', [False, True])
def test_gradient_rebuild_retries_only_response_admission(monkeypatch, allocator_failure):
    from dataclasses import replace

    system, orbital, auxiliary, points, cache, spins = _operator_fixture()
    controls = dict(_options())
    controls.pop('memory_byte_cap')
    pw_cutoff = controls.pop('plane_wave_cutoff')
    cache = replace(cache, source_parameters=('mdf-sr-lr-pw-v1', pw_cutoff, tuple(sorted(controls.items()))))
    widths = []
    def build(system_arg, orbital_arg, auxiliary_arg, bras, q, **kwargs):
        widths.append(len(bras))
        assert kwargs['gradient_nkpoints'] == len(points)
        assert kwargs['gradient_response_byte_cap'] > 0
        if len(bras) > 1:
            if allocator_failure:
                raise MemoryError('actual allocation failed')
            raise _RangeSeparatedGdfAdmissionError('response and reciprocal copies exceed memory cap')
        i = next(i for i, p in enumerate(points) if np.allclose(p, bras[0]))
        from vibeqc.aux_basis import _canonical_reciprocal_transfer
        j = next(j for j, p in enumerate(points)
                 if np.allclose(_canonical_reciprocal_transfer(system, p-points[i]), q))
        factors = cache[(i, j)][None].copy()
        return mdf._MdfBatch(factors, q, np.ones(1), 1, len(factors[0])-1, 4000, 1024**2)
    monkeypatch.setattr(mdf, '_build_mdf_shared_q', build)
    monkeypatch.setattr(mdf, '_compute_mdf_jk_gradient', lambda *args, **kwargs: np.zeros((1, 3)))
    def gradient():
        return mdf._compute_mdf_cache_gradient(
            cache, system, orbital, auxiliary, points, spins.sum(axis=0), [.3, .7],
            memory_byte_cap=512*1024**2, native_workspace_byte_cap=4096, coulomb_scale=0.)
    if allocator_failure:
        with pytest.raises(MemoryError, match='actual allocation'):
            gradient()
        assert widths.count(2) == 1
    else:
        np.testing.assert_array_equal(gradient(), 0.)
        assert widths.count(2) == 1 and widths.count(1) == 4


@pytest.mark.parametrize('spin_resolved', [False, True])
@pytest.mark.parametrize('reciprocal_shift', [False, True])
def test_zero_transfer_exchange_pairs_do_not_enter_hartree_response(monkeypatch, spin_resolved, reciprocal_shift):
    from vibeqc.aux_basis import _range_separated_gdf_source_signature

    system, orbital, auxiliary = _fixture()
    points = np.zeros((2, 3))
    if reciprocal_shift:
        points[1] = 2*np.pi*np.linalg.inv(system.lattice).T[:, 0]
    weights = [.3, .7]
    spins = np.arange(1., 5.).reshape(2, 2, 1, 1)
    density = spins if spin_resolved else spins.sum(axis=0)
    factors = {(i, j): np.ones((2, 1, 1), complex) for i, j in product(range(2), repeat=2)}
    options = dict(_options())
    options.pop('memory_byte_cap')
    pw_cutoff = options.pop('plane_wave_cutoff')
    signature = _range_separated_gdf_source_signature(system, orbital, auxiliary)
    cache = mdf._MdfCache(
        factors, signature, ('mdf-sr-lr-pw-v1', pw_cutoff, tuple(sorted(options.items()))),
        tuple(map(tuple, points)), ((4,),), sum(f.nbytes for f in factors.values()),
        1024**2, (0, 1), True, auxiliary,
    )
    def build(cell, ao, aux, bras, q, **kwargs):
        # Force multiple response batches so the diagonal group's metric
        # and complete Hartree source must survive subdivision too.
        if len(bras) > 1:
            raise _RangeSeparatedGdfAdmissionError('single-pair witness')
        source_key = (signature, tuple(q), .8, 6.2, 8., 6., pw_cutoff, 1e-9, 0.)
        joined = np.ones((1, 2, 1, 1), complex)
        state = mdf._MdfFitState(np.ones((1, 1, 1, 1), complex), np.eye(1, dtype=complex),
                                 np.ones(1, bool), np.zeros((1, 3)), np.array([[.2, 0., 0.]]),
                                 joined[:, 1:], bras+q, source_key, 100000)
        return mdf._MdfBatch(joined, q, np.ones(1), 1, 1, 1, 1024**2, state)
    calls = []
    def differentiate(cell, ao, aux, batch, pairs, kpoints, d, w, **kwargs):
        calls.append((tuple(pairs), kwargs['coulomb_scale'], kwargs['exchange_scale'],
                      kwargs.get('include_coulomb_metric')))
        if kwargs['coulomb_scale']:
            assert all(i == j for i, j in pairs)
            total = spins.sum(axis=0)
            np.testing.assert_array_equal(d, total)
            charge = sum(w[k]*total[k, 0, 0] for k in range(2))
            for source in kwargs['coulomb_sources']:
                np.testing.assert_allclose(source, [charge])
        else:
            assert kwargs['exchange_scale'] == (2. if spin_resolved else 1.)
        return np.zeros((1, 3))
    monkeypatch.setattr(mdf, '_build_mdf_shared_q', build)
    monkeypatch.setattr(mdf, '_compute_mdf_jk_gradient', differentiate)
    mdf._compute_mdf_cache_gradient(
        cache, system, orbital, auxiliary, points, density, weights, spin_resolved=spin_resolved,
        memory_byte_cap=512*1024**2, native_workspace_byte_cap=4096)
    hartree = [call for call in calls if call[1]]
    assert [call[0] for call in hartree] == [((0, 0),), ((1, 1),)]
    assert [call[3] for call in hartree] == [True, False]
    exchange_pairs = [pair for pairs, j, _, _ in calls if not j for pair in pairs]
    expected = sorted(list(product(range(2), repeat=2))*(2 if spin_resolved else 1))
    assert sorted(exchange_pairs) == expected


@pytest.mark.slow
@pytest.mark.parametrize('reciprocal_shift', [False, True])
def test_replicated_k_mesh_preserves_fixed_density_mdf_energy_and_gradient(reciprocal_shift):
    system, orbital, auxiliary = _projection_gradient_fixture()
    replica = np.zeros(3)
    if reciprocal_shift:
        replica = 2*np.pi*np.linalg.inv(system.lattice).T[:, 0]
    rng = np.random.default_rng(1602)
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    density = .05*(raw@raw.conj().T)
    options = dict(_options(), pair_cutoff=7.2, auxiliary_cutoff=14., ke_cutoff=8.,
                   plane_wave_cutoff=.4, memory_byte_cap=512*1024**2)
    results = []
    for points, weights in ((np.zeros((1, 3)), [1.]), (np.array([np.zeros(3), replica]), [.3, .7])):
        densities = np.repeat(density[None], len(points), axis=0)
        cache = mdf._build_mdf_cache(system, orbital, auxiliary, points, True, **options)
        jk = mdf._build_mdf_jk(cache, system, orbital, auxiliary, points, densities, weights,
                              memory_byte_cap=512*1024**2)
        gradient = mdf._compute_mdf_cache_gradient(
            cache, system, orbital, auxiliary, points, densities, weights,
            memory_byte_cap=512*1024**2, native_workspace_byte_cap=64*1024**2)
        results.append((jk, gradient))
    reference, replicated = results
    assert replicated[0].coulomb_energy == pytest.approx(reference[0].coulomb_energy, abs=2e-9, rel=0.)
    assert replicated[0].exchange_energy == pytest.approx(reference[0].exchange_energy, abs=2e-9, rel=0.)
    np.testing.assert_allclose(replicated[1], reference[1], atol=2e-8, rtol=0.)
