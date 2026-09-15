"""Finite-source native three-center tiles; tiny analytic integral oracles."""

from __future__ import annotations

import gc
import hashlib
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _oracle as _pair_oracle
from tests.test_periodic_correlation_reciprocal_metric import (
    _accepted_metric_records, _analytic_s_fourier, _dimensions,
    _raw_metric_config, _state, _two_s_metric_basis,
)


def _bundle(*, q=1, mesh=(3, 1, 1), block=2, pairs=4, auxiliary_block=2,
            ao=None, compensation=None, workspace=None, source_audit_control=None,
            reciprocal=None, reciprocal_cutoff=2.0, auxiliary=None):
    if auxiliary is None:
        auxiliary = _two_s_metric_basis()
    if ao is None:
        ao = _basis([
            (0, (0.1, -0.2, 0.15), [0.55], [0.7], True),
            (0, (0.6, 0.3, -0.25), [0.7], [0.8], True),
        ])
    if reciprocal is None:
        reciprocal = np.diag([1.0, 2.0, 3.0])
    state = _state(mesh, reciprocal, n_basis=ao.nbasis)
    dimensions = _dimensions(mesh, n_basis=ao.nbasis, n_auxiliary=auxiliary.nbasis)
    dimensions.factor_auxiliary_block = auxiliary_block
    dimensions.factor_ao_pair_block = pairs
    budget = core._PeriodicCorrelationResourceBudget()
    budget.memory_limit_bytes = budget.scratch_limit_bytes = 10**9
    budget.mpi_ranks = budget.workers_per_rank = 1
    reference = core._make_periodic_correlation_admitted_reference(state, dimensions, budget)
    schedule = core._make_periodic_correlation_factor_stream_schedule(reference)
    config = _raw_metric_config(auxiliary, cutoff=reciprocal_cutoff, reciprocal_block=block)
    config.ao_pair_block = pairs
    config.auxiliary_block = auxiliary_block
    config.ao_basis_identity_sha256 = core._auxiliary_basis_content_identity_sha256(ao)
    config.backend_identity_sha256 = (
        core._periodic_correlation_metric_factorization_backend_identity_sha256()
    )
    backend = config.backend
    backend.exact_extra_control_bytes = (
        core._PERIODIC_CORRELATION_THREE_CENTER_SOURCE_AUDIT_CONTROL_BYTES
        if source_audit_control is None else source_audit_control
    )
    backend.exact_extra_retained_bytes = (
        16 * auxiliary.nbasis * pairs if compensation is None else compensation
    )
    backend.per_thread_fourier_transform_fixed_workspace_bytes = (
        core._PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES
        if workspace is None else workspace
    )
    config.backend = backend
    sources = [core._make_periodic_correlation_reciprocal_metric_source_manifest(
        reference, schedule, config, auxiliary, qi, 65536,
    ) for qi in range(math.prod(mesh))]
    census = core._make_periodic_correlation_factor_build_census(
        reference, schedule, config, [source.factor_build_q_record() for source in sources],
    )
    raw = core._build_periodic_correlation_reciprocal_metric(
        reference, schedule, census, sources[q], auxiliary,
    )
    whitener = core._factorize_periodic_correlation_metric(
        reference, schedule, census, raw, 1e-12,
    )
    return SimpleNamespace(
        ao=ao, auxiliary=auxiliary, state=state, reference=reference,
        schedule=schedule, config=config, census=census, sources=sources,
        whitener=whitener, q=q, reciprocal=reciprocal,
    )


def _tile(bundle, *, k_bra=0, pair_tile=0, auxiliary_tile=0, cutoff=6.5,
          candidate_cap=20000, source=None, whitener=None, ao=None, census=None):
    shape = bundle.schedule.shape
    sequence = (bundle.q * shape.tiles_per_q + k_bra * shape.tiles_per_k_bra
                + pair_tile * shape.auxiliary_tile_count + auxiliary_tile)
    return core._build_periodic_correlation_three_center_tile(
        bundle.reference, bundle.schedule,
        bundle.census if census is None else census,
        bundle.sources[bundle.q] if source is None else source,
        bundle.whitener if whitener is None else whitener,
        bundle.ao if ao is None else ao, bundle.auxiliary,
        sequence, cutoff, candidate_cap,
    )


def _oracle(bundle, tile):
    records = _accepted_metric_records(bundle.sources[bundle.q])
    vectors = np.array([row[1] for row in records])
    weights = np.array([row[3] for row in records])
    f = np.array([_analytic_s_fourier(bundle.auxiliary, row[1], row[2])
                  for row in records]).T
    lattice = 2 * np.pi * np.linalg.inv(bundle.reciprocal).T
    k = np.asarray(bundle.state.kpoint_cartesian(tile.descriptor.k_ket_index))
    rho, _ = _pair_oracle(bundle.ao, vectors, lattice, k, tile.image_cutoff_bohr, 4)
    rho = rho.reshape(bundle.ao.nbasis**2, -1)
    raw = (f.conj() * weights) @ rho.T
    # Independent principal spectral formula, rather than using native W.
    metric = (f.conj() * weights) @ f.T
    values, eigenvectors = np.linalg.eigh(metric)
    keep = values > bundle.config.metric_absolute_eigenvalue_threshold
    w = (eigenvectors[:, keep] / np.sqrt(values[keep])) @ eigenvectors[:, keep].conj().T
    d = tile.descriptor
    return (w @ raw)[d.auxiliary_begin:d.auxiliary_begin + d.auxiliary_count,
                     d.ao_pair_begin:d.ao_pair_begin + d.ao_pair_count]


@pytest.mark.parametrize("q,k_bra", [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)])
def test_actual_multi_k_tile_matches_independent_s_gaussian_oracle(q, k_bra):
    bundle = _bundle(q=q)
    tile = _tile(bundle, k_bra=k_bra)
    np.testing.assert_allclose(tile.matrix, _oracle(bundle, tile), rtol=2e-11, atol=2e-12)
    assert tile.finite_image_reference
    assert not tile.ao_image_source_certified
    assert tile.output_bytes == tile.matrix.nbytes
    assert tile.numerical_peak_bytes <= max(tile.admitted_assembly_peak_bytes,
                                           tile.admitted_whitening_peak_bytes)
    if q == 0 and k_bra == 1:
        assert np.max(np.abs(tile.matrix.imag)) > 1e-6


def test_negative_nyquist_transfer_is_a_finite_numerical_tile():
    bundle = _bundle(q=1, mesh=(2, 1, 1))
    tile = _tile(bundle, k_bra=1)
    np.testing.assert_allclose(tile.matrix, _oracle(bundle, tile), rtol=2e-11, atol=2e-12)
    assert bundle.whitener.self_conjugate_transfer


def test_same_q_retains_ket_dependence_from_image_overlap():
    bundle = _bundle(q=1)
    first, second = _tile(bundle, k_bra=0), _tile(bundle, k_bra=1)
    assert np.max(np.abs(first.matrix - second.matrix)) > 1e-6
    assert first.source_identity_sha256 == second.source_identity_sha256
    assert first.payload_identity_sha256 != second.payload_identity_sha256


def test_reciprocal_blocks_are_bit_identical_and_plan_provenance_changes():
    tiles = [_tile(_bundle(block=block)) for block in (1, 2, 5)]
    for tile in tiles[1:]:
        np.testing.assert_array_equal(tile.matrix.view(np.uint64), tiles[0].matrix.view(np.uint64))
        assert tile.payload_identity_sha256 == tiles[0].payload_identity_sha256
        assert tile.plan_identity_sha256 != tiles[0].plan_identity_sha256


def test_auxiliary_and_partial_pair_tiles_reconstruct_same_factor():
    full = _tile(_bundle())
    bundle = _bundle(pairs=3, auxiliary_block=1)
    blocks = []
    for auxiliary_tile in range(2):
        blocks.append(np.hstack([
            _tile(bundle, pair_tile=pair, auxiliary_tile=auxiliary_tile).matrix
            for pair in range(2)
        ]))
    np.testing.assert_allclose(np.vstack(blocks), full.matrix, rtol=1e-13, atol=1e-13)


def test_mixed_sp_ao_tile_has_native_normalization():
    ao = _basis([
        (0, (0.0, 0.0, 0.0), [0.65], [0.8], True),
        (1, (0.2, -0.1, 0.3), [0.8], [0.7], True),
    ])
    bundle = _bundle(ao=ao, pairs=5)
    tile = _tile(bundle, pair_tile=1, cutoff=2.5)
    np.testing.assert_allclose(tile.matrix, _oracle(bundle, tile), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("kwargs", [{"compensation": 0}, {"workspace": 0}])
def test_uncharged_compensation_or_fourier_workspace_rejected(kwargs):
    bundle = _bundle(**kwargs)
    before = bundle.whitener.matrix
    with pytest.raises(ValueError, match="compensation and Fourier workspace"):
        _tile(bundle)
    np.testing.assert_array_equal(bundle.whitener.matrix, before)


def test_wrong_source_or_foreign_state_whitener_fails_before_numerical_work():
    bundle = _bundle()
    with pytest.raises(ValueError, match="provenance"):
        _tile(bundle, source=bundle.sources[0])
    foreign = _bundle()
    with pytest.raises(ValueError, match="provenance"):
        _tile(bundle, whitener=foreign.whitener)
    config = bundle.config
    config.reciprocal_block = 3
    census = core._make_periodic_correlation_factor_build_census(
        bundle.reference, bundle.schedule, config,
        [source.factor_build_q_record() for source in bundle.sources],
    )
    with pytest.raises(ValueError, match="provenance"):
        _tile(bundle, census=census)


def test_ao_content_and_image_candidate_cap_are_enforced():
    bundle = _bundle()
    with pytest.raises(ValueError, match="basis content"):
        _tile(bundle, ao=_two_s_metric_basis())
    with pytest.raises((ValueError, OverflowError), match="source cap"):
        _tile(bundle, candidate_cap=1)
    with pytest.raises(ValueError, match="cutoff"):
        _tile(bundle, cutoff=float("nan"))


def test_cutoff_is_hashed_even_when_values_coincide():
    bundle = _bundle()
    first = _tile(bundle, cutoff=0.9)
    second = _tile(bundle, cutoff=1.0)
    np.testing.assert_array_equal(first.matrix, second.matrix)
    assert first.payload_identity_sha256 != second.payload_identity_sha256


def test_result_owns_only_its_tile_and_matrix_copy_does_not_mutate_it():
    bundle = _bundle()
    tile = _tile(bundle)
    expected = tile.matrix
    del bundle
    gc.collect()
    np.testing.assert_array_equal(tile.matrix, expected)
    copy = tile.matrix
    copy[:] = 123
    np.testing.assert_array_equal(tile.matrix, expected)


def test_payload_identity_independent_wire_oracle():
    bundle = _bundle()
    tile = _tile(bundle)
    digest = hashlib.sha256()
    def u64(value):
        digest.update(struct.pack(">Q", value))
    def string(value):
        encoded = value.encode()
        u64(len(encoded))
        digest.update(encoded)
    def real(value):
        digest.update(struct.pack(">d", 0.0 if value == 0 else value))
    string("vibeqc.periodic.correlation.three-center.payload")
    digest.update(struct.pack(">I", 1))
    for value in (tile.source_identity_sha256, tile.whitener_payload_identity_sha256,
                  tile.ao_basis_identity_sha256, tile.auxiliary_basis_identity_sha256,
                  core._PERIODIC_CORRELATION_THREE_CENTER_IMAGE_POLICY,
                  core._PERIODIC_CORRELATION_THREE_CENTER_LATTICE_POLICY):
        string(value)
    real(tile.image_cutoff_bohr)
    for value in (2 * np.pi * np.linalg.inv(bundle.reciprocal).T).flat:
        real(value)
    d = tile.descriptor
    for value in bundle.state.kpoint_cartesian(d.k_ket_index):
        real(value)
    for name in ("sequence_index", "q_index", "k_bra_index", "k_ket_index"):
        u64(getattr(d, name))
    for value in d.k_ket_reciprocal_wrap:
        digest.update(struct.pack(">i", value))
    for name in ("ao_pair_begin", "ao_pair_count", "auxiliary_begin", "auxiliary_count", "element_count"):
        u64(getattr(d, name))
    for value in (tile.image_candidate_count, tile.retained_pair_image_count,
                  tile.reciprocal_vector_count, tile.output_bytes):
        u64(value)
    for value in tile.matrix.flat:
        real(value.real)
        real(value.imag)
    assert tile.payload_identity_sha256 == digest.hexdigest()


def _stream_caps(bundle):
    caps = core._PeriodicCorrelationThreeCenterStreamCaps()
    caps.maximum_tile_count = bundle.schedule.shape.tile_count
    caps.maximum_logical_bytes = bundle.schedule.shape.logical_bytes
    caps.maximum_tile_bytes = bundle.schedule.shape.maximum_tile_bytes
    caps.maximum_reciprocal_candidates_per_q = 65536
    caps.maximum_image_candidates_per_tile = 20000
    return caps


def _stream(bundle, caps=None, **kwargs):
    return core._periodic_correlation_three_center_stream_diagnostic(
        bundle.reference, bundle.schedule, bundle.census, bundle.ao, bundle.auxiliary,
        6.5, 1e-12, _stream_caps(bundle) if caps is None else caps, **kwargs,
    )


def test_full_native_physical_stream_completion_and_independent_receipt():
    bundle = _bundle(pairs=3, auxiliary_block=1)
    events = []
    receipt = _stream(bundle, progress=events.append)
    shape = bundle.schedule.shape
    assert receipt.completed_q_count == shape.n_kpoints
    assert receipt.completed_tile_count == shape.tile_count
    assert receipt.completed_element_count == shape.logical_element_count
    assert receipt.completed_logical_bytes == shape.logical_bytes
    assert receipt.maximum_tile_bytes == shape.maximum_tile_bytes
    assert receipt.finite_image_reference and not receipt.ao_image_source_certified
    assert receipt.admitted_peak_memory_bytes <= 10**9
    stages = core._PeriodicCorrelationThreeCenterStreamStage
    assert events[0].stage == stages.SOURCE
    assert events[0].completed_tile_count == 0
    assert events[-1].stage == stages.COMPLETE
    assert events[-1].completed_tile_count == shape.tile_count
    assert [e.completed_tile_count for e in events] == sorted(
        e.completed_tile_count for e in events
    )
    assert [e.q_index for e in events if e.stage == stages.SOURCE] == list(range(3))
    assert [e.completed_q_count for e in events if e.stage == stages.Q_COMPLETE] == [1, 2, 3]

    digest = hashlib.sha256()
    def u64(value):
        digest.update(struct.pack(">Q", value))
    def string(value):
        encoded = value.encode()
        u64(len(encoded))
        digest.update(encoded)
    string("vibeqc.periodic.correlation.three-center.stream")
    digest.update(struct.pack(">I", 1))
    for value in (receipt.schedule_identity_sha256, receipt.census_identity_sha256,
                  receipt.plan_identity_sha256):
        string(value)
    digest.update(struct.pack(">dd", 6.5, 1e-12))
    for value in (shape.n_kpoints, shape.tile_count, shape.logical_element_count,
                  shape.logical_bytes):
        u64(value)
    # Reconstruct the stream's wire from separately requested physical tiles.
    # Earlier tests independently anchor these tiles to Gaussian moments.
    for q, source in enumerate(bundle.sources):
        raw = core._build_periodic_correlation_reciprocal_metric(
            bundle.reference, bundle.schedule, bundle.census, source, bundle.auxiliary,
        )
        w = core._factorize_periodic_correlation_metric(
            bundle.reference, bundle.schedule, bundle.census, raw, 1e-12,
        )
        u64(q)
        string(source.source_identity_sha256)
        string(w.payload_identity_sha256)
        for offset in range(shape.tiles_per_q):
            sequence = q * shape.tiles_per_q + offset
            tile = core._build_periodic_correlation_three_center_tile(
                bundle.reference, bundle.schedule, bundle.census, source, w,
                bundle.ao, bundle.auxiliary, sequence, 6.5, 20000,
            )
            u64(sequence)
            string(tile.payload_identity_sha256)
    assert receipt.payload_identity_sha256 == digest.hexdigest()


@pytest.mark.parametrize("field", ["maximum_tile_count", "maximum_logical_bytes",
                                  "maximum_tile_bytes"])
def test_stream_work_rejection_occurs_before_callbacks(field):
    bundle = _bundle()
    caps = _stream_caps(bundle)
    setattr(caps, field, getattr(caps, field) - 1)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="work caps"):
        _stream(bundle, caps, progress=events.append)
    assert events == []


def test_stream_missing_work_limits_and_unadmitted_receiver_reject_before_callbacks():
    bundle = _bundle()
    events = []
    with pytest.raises(ValueError, match="explicit positive work caps"):
        _stream(bundle, core._PeriodicCorrelationThreeCenterStreamCaps(), progress=events.append)
    caps = _stream_caps(bundle)
    caps.receiver_retained_numeric_bytes = 1
    with pytest.raises(ValueError, match="receiver/Fourier workspace"):
        _stream(bundle, caps, progress=events.append)
    assert not events


def test_stream_receiver_failure_has_no_completion_event_or_receipt():
    bundle = _bundle()
    events = []
    with pytest.raises(RuntimeError, match="cancelled before tile acceptance"):
        _stream(bundle, fail_before_sequence=2, progress=events.append)
    assert max(event.completed_tile_count for event in events) == 2
    assert all(event.stage != core._PeriodicCorrelationThreeCenterStreamStage.COMPLETE
               for event in events)
    # No mutable executor survives the exception. A fresh complete traversal
    # succeeds with identical immutable inputs and its own receiver.
    assert _stream(bundle).completed_tile_count == bundle.schedule.shape.tile_count


def test_stream_progress_exception_propagates_immediately():
    bundle = _bundle()
    events = []
    def cancel(event):
        events.append(event)
        raise LookupError("intentional progress cancellation")
    with pytest.raises(LookupError, match="intentional progress cancellation"):
        _stream(bundle, progress=cancel)
    assert len(events) == 1
    assert events[0].completed_tile_count == 0


def test_stream_source_candidate_cap_failure_does_not_deliver_a_tile():
    bundle = _bundle()
    caps = _stream_caps(bundle)
    caps.maximum_reciprocal_candidates_per_q = 1
    events = []
    with pytest.raises((ValueError, RuntimeError), match="candidate"):
        _stream(bundle, caps, progress=events.append)
    assert events and max(event.completed_tile_count for event in events) == 0


def test_stream_execution_identity_distinguishes_reciprocal_panel_plans():
    first = _stream(_bundle(block=1))
    second = _stream(_bundle(block=2))
    assert first.completed_logical_bytes == second.completed_logical_bytes
    assert first.census_identity_sha256 != second.census_identity_sha256
    assert first.payload_identity_sha256 != second.payload_identity_sha256


@pytest.mark.parametrize("missing", [0, 1])
def test_stream_requires_conjugate_source_control_before_callbacks(missing):
    required = core._PERIODIC_CORRELATION_THREE_CENTER_SOURCE_AUDIT_CONTROL_BYTES
    bundle = _bundle(source_audit_control=0 if missing == 0 else required - 1)
    events = []
    with pytest.raises(ValueError, match="workspace admission"):
        _stream(bundle, progress=events.append)
    assert events == []


def test_stream_rejects_nonempty_unpaired_source_before_its_metric_or_tiles():
    reciprocal = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    # The shorter third reciprocal axis keeps Gamma nonempty; the skew
    # mixed-Nyquist q=4/5 boundary mismatch is unchanged.
    b = _bundle(q=0, mesh=(2, 3, 1), reciprocal=reciprocal,
                reciprocal_cutoff=0.4027777777777386)
    assert all(s.accepted_vector_count > 0 for s in b.sources)
    assert b.sources[4].accepted_vector_count != b.sources[5].accepted_vector_count
    events = []
    with pytest.raises(RuntimeError, match="unequal accepted-vector counts"):
        _stream(b, progress=events.append)
    stage = core._PeriodicCorrelationThreeCenterStreamStage
    assert not any(e.q_index == 4 and e.stage in (stage.METRIC, stage.TILES) for e in events)
    assert not any(e.stage == stage.COMPLETE for e in events)
