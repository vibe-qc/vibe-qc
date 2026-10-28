"""Experimental retained-auxiliary transport for the explicit SR/LR GDF source.

This diagnostic consumes independently built batches, never a solver switch.
The finite producer and its fitting convention are defined in aux_basis.py
(Sun et al., J. Chem. Phys. 147, 164119 (2017), doi:10.1063/1.4998644).
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .symmetry_shared import (
    Budget, EvidenceKind, OperatorContract, QualificationEvidence, SpaceAction,
    SpaceIdentity, SubspaceTransportEvidence, audit_subspace_panels,
    _freeze, _integer, _relative_frobenius, _text, _tolerance,
)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class GdfTransportEvidence:
    """One finite-source tensor witness, including failed parent relations."""
    retained: SubspaceTransportEvidence
    metric: QualificationEvidence
    whitening: QualificationEvidence
    source_factorization: QualificationEvidence
    target_factorization: QualificationEvidence
    three_center: QualificationEvidence
    factors: QualificationEvidence
    mixing: np.ndarray
    reconstructed_factors: np.ndarray
    source_digest: str
    target_digest: str
    required_bytes: int
    required_work: int

    def __init__(self, *args, **kwargs):
        raise TypeError("construct GDF evidence with audit_range_separated_gdf_transport")

    @property
    def passed_probe(self):
        return self.retained.passed_probe and all(getattr(self, key).passed_probe for key in (
            'metric', 'whitening', 'source_factorization', 'target_factorization',
            'three_center', 'factors',
        ))

    @property
    def production_reduction_authorized(self):
        return False


def _payload(batch):
    # Metadata only: admission must precede all scans, copies and action calls.
    from .aux_basis import _RangeSeparatedGdfBatch, _RangeSeparatedGdfFitState
    if not isinstance(batch, _RangeSeparatedGdfBatch) or not isinstance(
        batch.fit_state, _RangeSeparatedGdfFitState
    ):
        raise TypeError("GDF transport requires a batch with retain_fit_state=True")
    state = batch.fit_state
    arrays = (batch.factors, batch.metric_eigenvalues, state.three_center,
              state.eigenvectors, state.keep, state.vectors, state.ket_kpoints,
              batch.transfer)
    if any(not isinstance(a, np.ndarray) for a in arrays):
        raise TypeError("GDF transport requires NumPy source arrays")
    t, e = state.three_center, state.eigenvectors
    if t.ndim != 4 or e.ndim != 2:
        raise ValueError("invalid GDF source dimensions")
    k, n, a, b = t.shape
    if min(k, n, a) < 1 or a != b or e.shape != (n, n):
        raise ValueError("invalid GDF source dimensions")
    r = batch.factors.shape[1] if batch.factors.ndim == 4 else 0
    expected = ((k, r, a, a), (n,), (k, n, a, a), (n, n), (n,),
                (batch.reciprocal_vector_count, 3), (k, 3), (3,))
    dtypes = ('complex128', 'float64', 'complex128', 'complex128', 'bool',
              'float64', 'float64', 'float64')
    for array, shape, dtype in zip(arrays, expected, dtypes):
        if array.shape != shape or array.dtype != np.dtype(dtype) or not array.flags.aligned:
            raise ValueError("invalid GDF source array shape, dtype or alignment")
    if not 0 < r <= n:
        raise ValueError("invalid GDF retained rank")
    if not isinstance(state.source_parameters, tuple) or len(state.source_parameters) != 6:
        raise ValueError("GDF source parameters must be retained")
    _text(state.source_signature, 'GDF source signature')
    return arrays, (k, n, a, r)


def _axis(action, array, axis, *, conjugated=False):
    """Apply one tensor leg, conjugating the tensor exactly once overall."""
    panel = np.array(np.moveaxis(array, axis, 0).reshape(action.dimension, -1),
                     dtype=complex, order='C', copy=True)
    if action.antiunitary and conjugated:
        panel = panel.conj()
    result = action.apply(panel)
    if (not isinstance(result, np.ndarray) or result.dtype != np.complex128
            or result.shape != panel.shape or not np.isfinite(result).all()):
        raise ValueError("invalid GDF action output")
    shape = list(array.shape)
    shape.insert(0, shape.pop(axis))
    # Own the result before another callback may reuse its workspace.
    return np.moveaxis(result.copy().reshape(shape), 0, axis)


def audit_range_separated_gdf_transport(
    source, target, auxiliary_action: SpaceAction, bra_factor_action: SpaceAction,
    ket_factor_action: SpaceAction, *, contract: OperatorContract,
    probe_identity: str, budget: Budget, tolerance: float = 1e-9,
) -> GdfTransportEvidence:
    """Audit compact eigenmode factors against an independent full target build.

    Both inputs are `_build_lpq_range_separated_shared_q` batches retaining
    fit state, in corresponding k-pair order. Retained source controls must agree. The actual reciprocal vectors are
    digest-bound; their geometric mapping remains caller-owned.
    Each action names a *factor index*, with destination rows/source columns:
    T'[Q,a,b] = sum Uaux[Q,P] Ubra[a,m] Uket[b,n] K^s(T[P,m,n]).
    All actions have the same scalar antiunitary parity; K occurs once.
    In the classic pure-shell Bloch convention these legs carry respectively
    +q, +k_bra and -k_ket atom-image phases. Do not substitute a density
    pushforward or an operator pullback for either factor leg. Actions may
    include reciprocal wraps, translations and nonorthogonal AO transforms.
    One pair of bra/ket actions applies to every supplied batch member;
    audit separately when its k-dependent phases differ.

    The supplied action/contract are declarations, not geometric or finite
    support admission. Metric covariance, retained-space closure, whitened
    isometry, raw tensor covariance, and both original factorizations are
    tested separately. A failed parent never becomes a passing result through
    a zero tensor or discarded auxiliary leakage. Eigenvalues use the actual
    producer's strict absolute `lambda > threshold` selection; ranks must
    agree. Degenerate gauges may mix densely. Near-cut rank disagreement is
    refused, never repaired or averaged.

    For retained eigenvectors E and positive eigenvalues l, M = Et^dagger
    Uaux K(Es), and W = diag(lt^-1/2) M diag(ls^1/2). W is the compact
    factor-row action; it must also be an isometry for contractions to agree.
    No dense AO action, AO-pair Gram, global mesh or pair inventory is built.

    Budget admission covers both complete borrowed batches, snapshots,
    retained outputs and conservative explicit NumPy work/intermediates.
    Callback storage/execution, Python allocator and BLAS workspace remain
    separately owned, as in audit_subspace_panels. This is a diagnostic,
    with production_reduction_authorized always false, not a scheduler.
    """
    if not isinstance(contract, OperatorContract) or not isinstance(budget, Budget):
        raise TypeError("GDF audit requires an OperatorContract and Budget")
    _text(probe_identity, 'probe_identity')
    tolerance = _tolerance(tolerance)
    if tolerance >= 1:
        raise ValueError("GDF tolerance must be less than one")
    sa, shape = _payload(source)
    ta, other = _payload(target)
    if shape != other:
        raise ValueError("GDF batches require matching dimensions and retained ranks")
    k, n, a, r = shape
    actions = (auxiliary_action, bra_factor_action, ket_factor_action)
    for action, dim in zip(actions, (n, a, a)):
        if _integer(action.dimension, 'action dimension', positive=True) != dim:
            raise ValueError("GDF factor action dimension mismatch")
        if type(action.antiunitary) is not bool:
            raise TypeError("GDF antiunitary flag must be bool")
        if not isinstance(action.source, SpaceIdentity) or not isinstance(action.target, SpaceIdentity):
            raise TypeError("GDF factor actions require named spaces")
    if len({x.antiunitary for x in actions}) != 1:
        raise ValueError("GDF factor actions must share antiunitary parity")
    borrowed = sum(x.nbytes for x in sa + ta)
    required_bytes = 16384 + 4 * borrowed + 16 * (96*n*n + 64*n*r + 64*r*r + 24*k*n*a*a)
    required_work = 4096 + 16 * sum(x.size for x in sa + ta) + 512 * (n**3 + n*n*r + n*r*r + k*n*n*a*a + k*n*a**3)
    budget.admit(required_bytes, required_work)
    if source.fit_state.source_parameters != target.fit_state.source_parameters:
        raise ValueError("GDF batches must declare identical retained source controls")
    if any(not np.isfinite(x).all() for x in sa + ta):
        raise ValueError("nonfinite GDF source payload")
    # Freeze references before any action callback. FitState is a borrowed,
    # mutable producer payload, not an authenticated immutable owner.
    sa, ta = tuple(x.copy() for x in sa), tuple(x.copy() for x in ta)
    ls, lt = sa[1], ta[1]
    es, et = sa[3], ta[3]
    threshold = _tolerance(source.fit_state.source_parameters[-1], 'metric threshold')
    from .aux_basis import _rsgdf_metric_keep
    for batch, arrays in ((source, sa), (target, ta)):
        eigenvalues, vectors, keep = arrays[1], arrays[3], arrays[4]
        # The producer's executed decomposition (#179): 'cholesky' keeps the
        # full rank, 'eigen' the strict absolute lambda > threshold cut.
        decomposition = getattr(batch.fit_state, 'metric_decomposition', 'eigen')
        if (np.any(np.diff(eigenvalues) < 0) or np.count_nonzero(keep) != r
                or not np.array_equal(keep, _rsgdf_metric_keep(
                    eigenvalues, threshold, decomposition))):
            raise ValueError("GDF retained mask disagrees with strict metric rank policy")
        negative_tolerance = max(threshold, 64*np.finfo(float).eps*n
                                 * max(1., float(np.max(np.abs(eigenvalues)))))
        if eigenvalues[0] < -negative_tolerance:
            raise ValueError("GDF metric has negative modes outside producer tolerance")
        if np.linalg.norm(vectors.conj().T @ vectors - np.eye(n)) > tolerance:
            raise ValueError("GDF metric eigenvectors must be orthonormal")
    cs, ct = np.ascontiguousarray(es[:, sa[4]]), np.ascontiguousarray(et[:, ta[4]])
    def digest(arrays, signature):
        identity = sha256(b'vibeqc-symmetry-gdf-batch-v1\0')
        identity.update(signature.encode('utf-8'))
        identity.update(repr(source.fit_state.source_parameters).encode('ascii'))
        for value in arrays:
            identity.update(str((value.shape, value.dtype.str)).encode('ascii'))
            identity.update(value.tobytes())
        return identity.hexdigest()
    source_digest = digest(sa, source.fit_state.source_signature)
    target_digest = digest(ta, target.fit_state.source_signature)
    def retained_name(ambient, signature, numerical_digest):
        return SpaceIdentity(ambient.geometry, ambient.basis,
                             'SR/LR retained auxiliary:' + signature,
                             ambient.gauge + ':' + numerical_digest)
    retained = audit_subspace_panels(
        auxiliary_action, cs, ct, np.eye(n, dtype=complex), np.eye(n, dtype=complex),
        source_subspace=retained_name(auxiliary_action.source, source.fit_state.source_signature, source_digest),
        target_subspace=retained_name(auxiliary_action.target, target.fit_state.source_signature, target_digest),
        contract=contract, metric_tolerance=tolerance, leakage_tolerance=tolerance,
        probe_identity=probe_identity, budget=budget,
    )
    def evidence(value, reference, relation):
        residual = _relative_frobenius(value-reference, reference)
        return QualificationEvidence(contract, auxiliary_action.source, auxiliary_action.target,
            relation, EvidenceKind.NUMERICAL, probe_identity, residual, tolerance)
    sm, tm = (es * ls) @ es.conj().T, (et * lt) @ et.conj().T
    # Two-index push on the auxiliary metric, without materializing Uaux.
    moved_metric = _axis(auxiliary_action, sm, 0)
    moved_metric = _axis(auxiliary_action, moved_metric.conj().T, 0,
                         conjugated=True).conj().T
    metric = evidence(moved_metric, tm, 'full auxiliary metric covariance')
    mixing = retained.mixing * np.sqrt(ls[sa[4]])[None, :] / np.sqrt(lt[ta[4]])[:, None]
    whitening = evidence(mixing.conj().T @ mixing, np.eye(r), 'whitened auxiliary isometry')
    def factorize(arrays):
        whitener = arrays[3][:, arrays[4]].conj().T / np.sqrt(arrays[1][arrays[4]])[:, None]
        return np.einsum('rp,kpab->krab', whitener, arrays[2], optimize=True)
    source_factorization = evidence(sa[0], factorize(sa), 'source L = lambda^-1/2 E^dagger T')
    target_factorization = evidence(ta[0], factorize(ta), 'target L = lambda^-1/2 E^dagger T')
    moved_raw = _axis(auxiliary_action, sa[2], 1)
    moved_raw = _axis(bra_factor_action, moved_raw, 2, conjugated=True)
    moved_raw = _axis(ket_factor_action, moved_raw, 3, conjugated=True)
    raw = evidence(moved_raw, ta[2], 'raw three-center tensor covariance')
    factors = sa[0].conj() if auxiliary_action.antiunitary else sa[0]
    reconstructed = np.einsum('rs,ksab->krab', mixing, factors, optimize=True)
    reconstructed = _axis(bra_factor_action, reconstructed, 2, conjugated=True)
    reconstructed = _axis(ket_factor_action, reconstructed, 3, conjugated=True)
    fitted = evidence(reconstructed, ta[0], 'compact fitted tensor covariance')
    if not np.isfinite(mixing).all() or not np.isfinite(reconstructed).all():
        raise ValueError("nonfinite GDF reconstruction")
    result = object.__new__(GdfTransportEvidence)
    for key, value in dict(retained=retained, metric=metric, whitening=whitening,
        source_factorization=source_factorization, target_factorization=target_factorization,
        three_center=raw, factors=fitted, mixing=_freeze(mixing),
        reconstructed_factors=_freeze(reconstructed), source_digest=source_digest,
        target_digest=target_digest, required_bytes=required_bytes,
        required_work=required_work).items():
        object.__setattr__(result, key, value)
    return result
