"""Finite SR/LR retained-mode diagnostics, not production reduction acceptance."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.aux_basis import (
    _RangeSeparatedGdfBatch, _RangeSeparatedGdfFitState,
    _build_lpq_range_separated_shared_q,
)
from vibeqc.symmetry_ao import build_molecular_space_action
from vibeqc.symmetry_gdf import GdfTransportEvidence, audit_range_separated_gdf_transport
from vibeqc.symmetry_shared import BlockSpaceAction, Budget, OperatorContract, SpaceIdentity

BUDGET = Budget(256 << 20, 10**10)
SPACE = SpaceIdentity('test geometry', 'test basis', 'factor index', 'native')
CONTRACT = OperatorContract('explicit SR/LR GDF', 'test source',
    'declared finite images and reciprocal sphere', 'unscreened', 'L=lambda^-1/2 E^dagger T')


def action(u, anti=False):
    u = np.asarray(u, complex)
    return BlockSpaceAction(SPACE, SPACE, np.array([0, len(u)], np.int64),
        np.array([0], np.int64), u.ravel(), antiunitary=anti, budget=BUDGET)


def audit(source, target, aux, bra, ket, **kw):
    return audit_range_separated_gdf_transport(source, target, aux, bra, ket,
        contract=CONTRACT, probe_identity='independent target', budget=kw.pop('budget', BUDGET),
        **kw)


def batch(eigenvalues, eigenvectors, raw, threshold=.1):
    values = np.asarray(eigenvalues, float)
    vectors = np.asarray(eigenvectors, complex)
    raw = np.asarray(raw, complex)
    keep = values > threshold
    fitted = np.einsum('pr,kpab->krab', vectors[:, keep].conj(), raw)
    fitted /= np.sqrt(values[keep])[None, :, None, None]
    state = _RangeSeparatedGdfFitState(raw, vectors, keep, np.zeros((1, 3)),
        np.zeros((len(raw), 3)), (.5, 8., 10., 100000, 0., threshold), 'fixture-source')
    return _RangeSeparatedGdfBatch(fitted, np.zeros(3), values, 1, 1024, state)


def unitary(rng, n):
    return np.linalg.qr(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))[0]


@pytest.mark.parametrize('anti', [False, True])
def test_dense_gauges_nonorthogonal_ao_legs_and_complex_contractions(anti):
    rng = np.random.default_rng(9351)
    es, u = (unitary(rng, 4) for _ in range(2))
    # Three retained modes, with an independently mixed exactly-degenerate pair.
    gauge = np.eye(4, dtype=complex)
    gauge[2:, 2:] = unitary(rng, 2)
    et = u @ (es.conj() if anti else es) @ gauge
    raw = rng.normal(size=(2, 4, 3, 3)) + 1j*rng.normal(size=(2, 4, 3, 3))
    bra, ket = unitary(rng, 3) @ np.diag([.7, 1., 1.3]), unitary(rng, 3)
    transformed = np.einsum('qp,am,bn,kpmn->kqab', u, bra, ket,
                            raw.conj() if anti else raw)
    source, target = batch([0, 2, 3, 3], es, raw), batch([0, 2, 3, 3], et, transformed)
    result = audit(source, target, action(u, anti), action(bra, anti), action(ket, anti),
                   tolerance=1e-12)
    assert result.passed_probe and not result.production_reduction_authorized
    np.testing.assert_allclose(result.mixing, gauge[1:, 1:].conj().T, atol=2e-14)
    # Independent contraction oracle: pseudoinverse of target metric, not
    # the reconstructed factor, contracted against a nonsymmetric complex trial.
    d = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
    inverse = (et[:, 1:] / np.array([2, 3, 3])) @ et[:, 1:].conj().T
    for t, l in zip(transformed, result.reconstructed_factors):
        expected = np.einsum('pab,pq,qcd,cd->ab', t.conj(), inverse, t, d)
        actual = np.einsum('rab,rcd,cd->ab', l.conj(), l, d)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    for array in (result.mixing, result.reconstructed_factors):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        GdfTransportEvidence()
    source.factors[:] = 17
    assert result.passed_probe


def test_parent_metric_and_leakage_fail_even_when_tensors_pass():
    raw = np.zeros((1, 3, 2, 2), complex)
    raw[:, 2] = 1
    source = batch([0, 2, 3], np.eye(3), raw)
    target = batch([0, 2, 3], np.eye(3), raw)
    # Swap a discarded and retained mode; the only nonzero tensor row is fixed.
    u = np.eye(3)[[1, 0, 2]]
    result = audit(source, target, action(u), action(np.eye(2)), action(np.eye(2)))
    assert result.three_center.passed_probe and result.factors.passed_probe
    assert not result.retained.passed_probe and not result.metric.passed_probe
    assert not result.whitening.passed_probe and not result.passed_probe


def test_failed_factorization_and_raw_covariance_are_separate():
    source = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    target = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    target.fit_state.three_center[:, 0] *= 3  # discarded raw row
    args = (source, target, action(np.eye(3)), action(np.eye(2)), action(np.eye(2)))
    result = audit(*args)
    assert result.factors.passed_probe and result.target_factorization.passed_probe
    assert not result.three_center.passed_probe and not result.passed_probe
    target.factors[:] *= 2  # mutate array without rebinding the NamedTuple field
    result = audit(*args)
    assert not result.target_factorization.passed_probe


def test_exact_and_one_short_aggregate_admission_precedes_scans_and_callbacks(monkeypatch):
    source = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    args = (source, source, action(np.eye(3)), action(np.eye(2)), action(np.eye(2)))
    result = audit(*args)
    exact = Budget(result.required_bytes, result.required_work)
    assert audit(*args, budget=exact).passed_probe
    def forbidden(*args, **kw):
        raise AssertionError('scan or callback before admission')
    monkeypatch.setattr(np, 'isfinite', forbidden)
    monkeypatch.setattr(BlockSpaceAction, 'apply', forbidden)
    # No tolerance conversion scans are necessary at this metadata boundary;
    # its scalar isfinite check is allowed, array scans are not.
    monkeypatch.setattr(np, 'isfinite', lambda x: True if np.isscalar(x) else forbidden())
    with pytest.raises(MemoryError):
        audit(*args, budget=Budget(exact.maximum_bytes-1, exact.maximum_work))
    with pytest.raises(ValueError, match='work budget'):
        audit(*args, budget=Budget(exact.maximum_bytes, exact.maximum_work-1))


@pytest.mark.parametrize('kind', ['rank', 'mask', 'nonfinite', 'controls', 'parity', 'orthogonality', 'negative'])
def test_refusals(kind):
    source = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    target = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    aux, bra, ket = action(np.eye(3)), action(np.eye(2)), action(np.eye(2))
    if kind == 'rank':
        target = batch([0, .05, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    elif kind == 'mask':
        target.fit_state.keep[:] = [True, False, True]
    elif kind == 'nonfinite':
        target.fit_state.three_center[0, 0, 0, 0] = np.nan
    elif kind == 'controls':
        target = target._replace(fit_state=target.fit_state._replace(
            source_parameters=(.5, 9., 10., 100000, 0., .1)))
    elif kind == 'negative':
        target.metric_eigenvalues[0] = -2
    elif kind == 'parity':
        aux = action(np.eye(3), True)
    else:
        target.fit_state.eigenvectors[0, 0] = 2
    with pytest.raises(ValueError):
        audit(source, target, aux, bra, ket)


def test_stabilizer_and_noncommuting_retained_composition():
    rng = np.random.default_rng(517)
    e = unitary(rng, 3)
    raw = np.zeros((1, 3, 2, 2), complex)
    source = batch([2, 2, 2], e, raw)
    u, v = unitary(rng, 3), unitary(rng, 3)
    def witness(matrix):
        return audit(source, source, action(matrix), action(np.eye(2)), action(np.eye(2)))
    gu, gv, product = witness(u), witness(v), witness(u @ v)
    assert gu.passed_probe and gv.passed_probe and product.passed_probe
    np.testing.assert_allclose(gu.mixing @ gv.mixing, product.mixing, atol=2e-14)
    assert np.linalg.norm(gv.mixing @ gu.mixing - product.mixing) > .1
    # A non-invariant tensor fails its stabilizer relation rather than being averaged.
    source.fit_state.three_center[:] = 1
    source.factors[:] = np.einsum('pr,kpab->krab', e.conj(), source.fit_state.three_center)/np.sqrt(2)
    assert not witness(u).passed_probe


@pytest.fixture(scope='module')
def physical_source():
    centers = [[-.7, 0., 0.], [.7, 0., 0.]]
    system = vq.PeriodicSystem(3, np.eye(3)*5, [vq.Atom(1, r) for r in centers])
    molecule = system.unit_cell_molecule()
    def basis(auxiliary):
        shells = [core.ShellInfo(i, l, True, [.7 if not auxiliary else .45], [1.], r)
                  for i, r in enumerate(centers) for l in (0, 1)]
        if auxiliary:
            # Symmetry-complete exact linear dependencies at both sites.
            shells += [shells[0], shells[2]]
        return core.BasisSet(molecule, shells, 'symmetry-fit-fixture')
    orbital, auxiliary = basis(False), basis(True)
    options = dict(omega=.5, pair_cutoff=7., auxiliary_cutoff=9., ke_cutoff=5.,
        linear_dep_thr=1e-8, memory_byte_cap=64 << 20,
        native_workspace_byte_cap=8 << 20, image_candidate_cap=1000000,
        reciprocal_candidate_cap=1000000, retain_fit_state=True)
    q, bras = np.array([.13, -.09, .07]), np.array([[.1, .02, -.04], [-.04, .08, .03]])
    source = _build_lpq_range_separated_shared_q(system, orbital, auxiliary, bras, q, **options)
    return system, orbital, auxiliary, options, q, bras, source


@pytest.mark.parametrize('rotation', [np.diag([-1., -1., 1.]), -np.eye(3), np.diag([1., -1., 1.])])
@pytest.mark.parametrize('anti', [False, True])
def test_actual_finite_producer_proper_improper_antiunitary(physical_source, rotation, anti):
    system, orbital, auxiliary, options, q, bras, source = physical_source
    # These origin-fixed operations preserve both explicit finite image balls.
    # q and bra include nonzero complex phases; no SCF state is used.
    sign = -1 if anti else 1
    target = _build_lpq_range_separated_shared_q(system, orbital, auxiliary,
        sign*bras @ rotation.T, sign*rotation @ q, **options)
    def molecular(basis):
        return build_molecular_space_action(system.unit_cell_molecule(), basis,
            np.ascontiguousarray(rotation), np.zeros(3), source=SPACE, target=SPACE,
            budget=BUDGET, antiunitary=anti)
    result = audit(source, target, molecular(auxiliary), molecular(orbital), molecular(orbital),
                   tolerance=1e-9)
    assert source.factors.shape[1] < auxiliary.nbasis
    assert result.passed_probe
    assert not result.production_reduction_authorized
    rng = np.random.default_rng(962)
    d = rng.normal(size=(orbital.nbasis,)*2) + 1j*rng.normal(size=(orbital.nbasis,)*2)
    # Contract independent target factors directly, including non-Hermitian density.
    for got, ref in zip(result.reconstructed_factors, target.factors):
        np.testing.assert_allclose(np.einsum('pab,pcd,cd->ab', got.conj(), got, d),
            np.einsum('pab,pcd,cd->ab', ref.conj(), ref, d), atol=1e-9, rtol=0)
    wrong = audit(source, target, action(np.eye(auxiliary.nbasis), anti),
        action(np.eye(orbital.nbasis), anti), action(np.eye(orbital.nbasis), anti))
    assert not wrong.passed_probe


def test_action_cannot_retarget_borrowed_producer_payloads():
    source = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    target = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    class ReusedAction:
        source = target = SPACE
        dimension, antiunitary = 3, False
        scratch = None
        def apply(self, panel):
            # Simulate a backend recycling the producer's mutable storage.
            source.factors[:] = 19
            target.fit_state.three_center[:] = 23
            if self.scratch is None or self.scratch.shape != panel.shape:
                self.scratch = np.empty_like(panel)
            self.scratch[:] = panel
            return self.scratch
    result = audit(source, target, ReusedAction(), action(np.eye(2)), action(np.eye(2)))
    assert result.passed_probe


def test_changed_retained_eigenvalues_fail_whitening_even_with_same_projector():
    raw = np.ones((1, 3, 2, 2))
    source, target = batch([0, 2, 3], np.eye(3), raw), batch([0, 4, 6], np.eye(3), raw)
    result = audit(source, target, action(np.eye(3)), action(np.eye(2)), action(np.eye(2)))
    assert result.retained.passed_probe and result.factors.passed_probe
    assert not result.metric.passed_probe and not result.whitening.passed_probe
    assert not result.passed_probe


@pytest.mark.parametrize('anti', [False, True])
def test_complex_phase_legs_are_not_density_or_operator_transport(anti):
    rng = np.random.default_rng(195)
    es = unitary(rng, 3)
    u = np.diag(np.exp(1j*np.array([.7, -.2, .4])))
    bra = np.diag(np.exp(1j*np.array([.3, -.8])))
    ket = np.diag(np.exp(-1j*np.array([.5, .9])))
    raw = rng.normal(size=(1, 3, 2, 2)) + 1j*rng.normal(size=(1, 3, 2, 2))
    transformed = np.einsum('qp,am,bn,kpmn->kqab', u, bra, ket, raw.conj() if anti else raw)
    source = batch([0, 2, 3], es, raw)
    target = batch([0, 2, 3], u @ (es.conj() if anti else es), transformed)
    assert audit(source, target, action(u, anti), action(bra, anti), action(ket, anti)).passed_probe
    assert not audit(source, target, action(u, anti), action(bra, anti),
                     action(ket.conj(), anti)).passed_probe


def test_reciprocal_payload_scans_join_work_admission():
    source = batch([0, 2, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    args = (action(np.eye(3)), action(np.eye(2)), action(np.eye(2)))
    small = audit(source, source, *args)
    larger = source._replace(reciprocal_vector_count=1000,
        fit_state=source.fit_state._replace(vectors=np.zeros((1000, 3))))
    big = audit(larger, larger, *args)
    assert big.required_work - small.required_work == 16*2*999*3
    assert big.required_bytes > small.required_bytes
    assert big.source_digest != small.source_digest
    with pytest.raises(ValueError, match='work budget'):
        audit(larger, larger, *args, budget=Budget(big.required_bytes, small.required_work))


def test_exact_metric_threshold_is_discarded_without_rank_repair():
    source = batch([0, .1, 3], np.eye(3), np.ones((1, 3, 2, 2)))
    result = audit(source, source, action(np.eye(3)), action(np.eye(2)), action(np.eye(2)))
    assert result.passed_probe and result.mixing.shape == (1, 1)
    changed = batch([0, np.nextafter(.1, np.inf), 3], np.eye(3), np.ones((1, 3, 2, 2)))
    with pytest.raises(ValueError, match='retained ranks'):
        audit(source, changed, action(np.eye(3)), action(np.eye(2)), action(np.eye(2)))
