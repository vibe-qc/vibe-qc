"""Actual finite reciprocal sources without manufactured SCF states."""

from __future__ import annotations

import gc
import hashlib
import math
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _system
from tests.test_periodic_auxiliary_fourier import _identity_fixture
from tests.test_periodic_gaussian_source_context import _make as _context, _options
from tests.test_periodic_correlation_reciprocal_metric import _accepted_metric_records, _manifest


_CAP_FIELDS = (
    "maximum_fixed_storage_bytes", "maximum_candidates_per_source",
    "maximum_candidate_evaluations", "maximum_source_wire_bytes",
)


def _caps():
    caps = core._PeriodicGaussianReciprocalSourceCaps()
    caps.maximum_fixed_storage_bytes = 10**6
    caps.maximum_candidates_per_source = 65536
    caps.maximum_candidate_evaluations = 400000
    caps.maximum_source_wire_bytes = 10**6
    return caps


def _source_context(mesh=(2, 3, 1), reciprocal=None, cutoff=2.0, ao=None):
    reciprocal = np.eye(3) if reciprocal is None else np.asarray(reciprocal)
    options = _options()
    options.reciprocal_energy_cutoff = cutoff
    return _context(system=_system(2 * np.pi * np.linalg.inv(reciprocal).T),
                    mesh=core._RegularKMesh(list(mesh)), options=options, ao=ao)


def _source(context=None, q=0, caps=None):
    return core._make_periodic_gaussian_reciprocal_source(
        _source_context() if context is None else context, q,
        _caps() if caps is None else caps,
    )


def _records(source):
    return core._periodic_gaussian_reciprocal_source_records(source, source.candidate_count)


def _wire_oracle(source):
    chunks = []

    def text(value):
        data = value.encode()
        chunks.extend((struct.pack(">Q", len(data)), data))

    def real(value):
        chunks.append(struct.pack(">d", 0.0 if value == 0.0 else value))

    text("vibeqc.periodic.gaussian-reciprocal-source")
    chunks.append(struct.pack(">III", 1, 1, 1))
    text(source.source_context_identity_sha256)
    chunks.append(struct.pack(">Q", source.q_index))
    for n in (*source.centered_doubled_numerator, *source.centered_reciprocal_wrap):
        chunks.append(struct.pack(">i", n))
    for x in source.reciprocal_lattice.ravel():
        real(x)
    for x in (source.reciprocal_energy_cutoff, source.maximum_reciprocal_radius,
              source.radial_boundary_tolerance, source.cell_volume_bohr3):
        real(x)
    for n in (*source.lower_bounds, *source.upper_bounds):
        chunks.append(struct.pack(">q", n))
    chunks.append(struct.pack(">QQQ", source.candidate_count, source.accepted_vector_count,
                              source.zero_mode_excluded_count))
    for labels, p, p2, weight in _accepted_metric_records(source):
        chunks.append(struct.pack(">qqq", *labels))
        for x in (*p, p2, weight):
            real(x)
    wire = b"".join(chunks)
    return hashlib.sha256(wire).hexdigest(), len(wire)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 3, 1)])
@pytest.mark.parametrize("skew", [False, True])
def test_gamma_nonself_nyquist_sources_match_independent_records_and_wires(mesh, skew):
    reciprocal = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) if skew else np.eye(3)
    context = _source_context(mesh, reciprocal)
    sources = [_source(context, q) for q in range(math.prod(mesh))]
    for source in sources:
        assert source.contract_version == 1
        assert source.reciprocal_conjugate_closure_certified
        assert not source.ao_image_source_certified
        assert source.conjugacy_audited_vector_count == source.accepted_vector_count
        assert source.source_context_identity_sha256 == context.source_context_identity_sha256
        opposite = sources[source.conjugate_q_index]
        assert source.conjugate_source_identity_sha256 == opposite.source_identity_sha256
        assert source.accepted_vector_count == opposite.accepted_vector_count
        assert source.zero_mode_excluded_count == int(source.q_index == 0)
        labels, lanes = _records(source)
        independent = _accepted_metric_records(source)
        np.testing.assert_array_equal(labels, [row[0] for row in independent])
        np.testing.assert_array_equal(lanes, [[*row[1], row[2], row[3]] for row in independent])
        assert source.source_identity_sha256 == _wire_oracle(source)[0]
        assert source.inventory.source_wire_bytes == _wire_oracle(source)[1]
        other_labels, other_lanes = _records(opposite)
        by_label = dict(zip(map(tuple, other_labels), other_lanes))
        shifts = -(np.asarray(source.centered_doubled_numerator)
                   + np.asarray(opposite.centered_doubled_numerator)) // (2 * np.asarray(mesh))
        for label, lane in zip(labels, lanes):
            partner = by_label[tuple(-label + shifts)]
            np.testing.assert_array_equal(partner[:3], -lane[:3])
            np.testing.assert_array_equal(partner[3:], lane[3:])
        assert source.self_conjugate_transfer == (source.q_index == source.conjugate_q_index)
        assert not hasattr(source, "state")
        assert not hasattr(source, "calculation_id")


@pytest.mark.parametrize("mesh,q", [((1, 1, 1), 0), ((3, 1, 1), 1), ((2, 3, 1), 4)])
def test_state_origin_v1_and_context_origin_use_bit_identical_numeric_enumeration(mesh, q):
    context = _source_context(mesh, [[1.0, 0.4, 0], [0, 1, 0.2], [0, 0, 1.2]])
    source = _source(context, q)
    # Legacy synthetic state is used ONLY as a diagnostic enum oracle, never
    # attached to the context-origin runtime or relabelled physical.
    old, *_ = _manifest(mesh, context.reciprocal_lattice, q, cutoff=2.0)
    for field in ("lower_bounds", "upper_bounds", "q_fractional", "q_cartesian"):
        np.testing.assert_array_equal(getattr(source, field), getattr(old, field))
    for field in ("maximum_reciprocal_radius", "radial_boundary_tolerance", "cell_volume_bohr3",
                  "candidate_count", "accepted_vector_count", "zero_mode_excluded_count"):
        assert getattr(source, field) == getattr(old, field)
    actual_labels, actual_lanes = _records(source)
    old_records = _accepted_metric_records(old)
    np.testing.assert_array_equal(actual_labels, [row[0] for row in old_records])
    np.testing.assert_array_equal(actual_lanes, [[*row[1], row[2], row[3]] for row in old_records])
    assert source.source_identity_sha256 != old.source_identity_sha256


def test_mixed_nyquist_skew_radial_boundary_fails_closed_without_publishing_source():
    context = _source_context((2, 3, 1), [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                              cutoff=0.4027777777777386)
    # Establish the exact witness on the context's actually derived B.
    left, *_ = _manifest((2, 3, 1), context.reciprocal_lattice, 4, cutoff=0.4027777777777386)
    right, *_ = _manifest((2, 3, 1), context.reciprocal_lattice, 5, cutoff=0.4027777777777386)
    assert left.accepted_vector_count != right.accepted_vector_count
    for q in (4, 5):
        with pytest.raises(RuntimeError, match="unequal accepted-vector counts"):
            _source(context, q)


def test_context_owner_survives_python_owner_deletion_and_sources_share_owner():
    context = _source_context()
    identity = context.source_context_identity_sha256
    first, second = _source(context, 1), _source(context, 2)
    assert first.context is context
    assert second.context is context
    del context
    gc.collect()
    assert first.context.source_context_identity_sha256 == identity
    assert first.context is second.context
    assert len(_records(first)[0]) == first.accepted_vector_count


def test_changed_actual_ao_content_and_scientific_cutoff_change_source_seal():
    first = _source()
    second = _source(_source_context(ao=_identity_fixture(first_coefficient=0.8)))
    assert first.source_identity_sha256 != second.source_identity_sha256
    np.testing.assert_array_equal(_records(first)[1], _records(second)[1])
    third = _source(_source_context(cutoff=1.9))
    assert first.source_identity_sha256 != third.source_identity_sha256


@pytest.mark.parametrize("q", [0, 1, 3, 4])
def test_exact_factory_candidate_pass_inventory_is_constant_space(q):
    source = _source(q=q)
    inv = source.inventory
    extra = 0 if source.self_conjugate_transfer else 2 * source.conjugate_candidate_count
    assert inv.variable_owned_numeric_bytes == 0
    assert inv.candidate_evaluations_upper_bound == 4 * source.candidate_count + extra
    assert inv.candidate_evaluations_performed == 3 * source.candidate_count + extra + source.accepted_vector_count
    assert inv.inventoried_fixed_storage_bytes == (inv.fixed_source_storage_bytes
        + inv.retained_context_storage_bytes + inv.factory_workspace_storage_bytes)
    assert inv.retained_context_storage_bytes == core._PERIODIC_GAUSSIAN_SOURCE_CONTEXT_STORAGE_BYTES
    assert inv.source_wire_bytes == 342 + 64 * source.accepted_vector_count
    assert inv.source_wire_bytes == core._periodic_gaussian_reciprocal_source_wire_bytes(source.accepted_vector_count)


def _exact_caps(source):
    caps = _caps()
    caps.maximum_fixed_storage_bytes = source.inventory.inventoried_fixed_storage_bytes
    caps.maximum_candidates_per_source = max(source.candidate_count, source.conjugate_candidate_count)
    caps.maximum_candidate_evaluations = source.inventory.candidate_evaluations_upper_bound
    caps.maximum_source_wire_bytes = source.inventory.source_wire_bytes
    return caps


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_every_exact_factory_cap_and_cap_minus_one(field):
    context = _source_context()
    first = _source(context, 4)
    caps = _exact_caps(first)
    assert _source(context, 4, caps).source_identity_sha256 == first.source_identity_sha256
    setattr(caps, field, getattr(caps, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _source(context, 4, caps)


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_positive_caps_required(field):
    caps = _caps()
    setattr(caps, field, 0)
    with pytest.raises(ValueError, match="positive"):
        _source(caps=caps)


def test_caps_and_basis_display_names_do_not_change_source_identity():
    context = _source_context(ao=_identity_fixture(name="first"))
    source = _source(context, 4)
    renamed = _source_context(ao=_identity_fixture(name="second"))
    assert _source(renamed, 4, _exact_caps(source)).source_identity_sha256 == source.source_identity_sha256


def test_empty_gamma_source_wrong_q_null_context_and_sha_extent_reject():
    with pytest.raises(ValueError, match="no usable vectors"):
        _source(_source_context(cutoff=1e-4), 0)
    with pytest.raises(IndexError):
        _source(q=6)
    with pytest.raises(ValueError, match="live native"):
        core._make_periodic_gaussian_reciprocal_source(None, 0, _caps())
    with pytest.raises((ValueError, RuntimeError), match="SHA"):
        core._periodic_gaussian_reciprocal_source_wire_bytes(2**61)


def test_callback_replay_matches_records_and_cap_failure_precedes_first_callback():
    source = _source(q=4)
    seen = []
    visit = core._visit_periodic_gaussian_reciprocal_source
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        visit(source, source.candidate_count - 1, lambda label, lanes: seen.append((label, lanes)))
    assert not seen
    count = visit(source, source.candidate_count, lambda label, lanes: seen.append((label, lanes)))
    assert count == source.accepted_vector_count
    labels, lanes = _records(source)
    np.testing.assert_array_equal([entry[0] for entry in seen], labels)
    np.testing.assert_array_equal([entry[1] for entry in seen], lanes)


def test_callback_failure_returns_no_success_and_source_remains_reusable():
    source = _source(q=1)
    identity = source.source_identity_sha256
    calls = []

    def fail(label, lanes):
        calls.append(label)
        if len(calls) == 2:
            raise RuntimeError("receiver cancelled")

    with pytest.raises(RuntimeError, match="receiver cancelled"):
        core._visit_periodic_gaussian_reciprocal_source(source, source.candidate_count, fail)
    assert len(calls) == 2
    assert source.source_identity_sha256 == identity
    assert core._visit_periodic_gaussian_reciprocal_source(source, source.candidate_count,
                                                         lambda label, lanes: None) == source.accepted_vector_count


def test_source_matrix_and_diagnostic_record_copies_cannot_mutate_sealed_source():
    source = _source()
    identity = source.source_identity_sha256
    matrix = source.reciprocal_lattice
    matrix[0, 0] = 123
    labels, lanes = _records(source)
    labels[:] = 123
    lanes[:] = 456
    assert source.source_identity_sha256 == identity
    assert source.reciprocal_lattice[0, 0] != 123
    assert not np.all(_records(source)[1] == 456)
