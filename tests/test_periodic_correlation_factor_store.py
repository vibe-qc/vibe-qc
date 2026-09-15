"""Tiny native, ephemeral POSIX factor-store tests; no chemistry or restart."""

from __future__ import annotations

import gc
import hashlib
import struct
import sys

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_three_center import _bundle


pytestmark = pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="POSIX store supports Linux/macOS")


def _configured(*, pairs=3, auxiliary_block=1, block=2, extra_shortfall=0,
                control_shortfall=0, codec_mismatch=False):
    b = _bundle(q=0, mesh=(2, 1, 1), pairs=pairs, auxiliary_block=auxiliary_block, block=block)
    b.whitener = None
    c = b.config
    c.publisher_mode = core._PeriodicCorrelationFactorPublisherMode.CANONICAL_SEQUENTIAL_EXACTLY_ONCE
    c.backing_mode = core._PeriodicCorrelationFactorBackingMode.DISK
    c.publisher_buffer_count = 0
    c.transpose_before_publish = False
    c.codec_identity_sha256 = core._periodic_correlation_private_factor_store_codec_identity_sha256()
    codec = core._periodic_correlation_private_factor_store_codec_inventory()
    if codec_mismatch:
        codec.fixed_header_bytes += 1
    c.codec = codec
    backend = c.backend
    backend.exact_extra_retained_bytes = (
        16 * b.auxiliary.nbasis * pairs
        + core._periodic_correlation_private_factor_store_receiver_bytes(b.schedule)
        - extra_shortfall
    )
    backend.exact_extra_control_bytes = (
        core._periodic_correlation_private_factor_store_fixed_control_bytes() - control_shortfall
    )
    c.backend = backend
    b.config = c
    b.census = core._make_periodic_correlation_factor_build_census(
        b.reference, b.schedule, c, [s.factor_build_q_record() for s in b.sources],
    )
    b.store_caps = core._PeriodicCorrelationPrivateFactorStoreCaps()
    b.store_caps.maximum_file_bytes = 1048576
    b.store_caps.maximum_tile_bytes = 8192
    b.store_caps.maximum_tile_count = 128
    b.stream_caps = core._PeriodicCorrelationThreeCenterStreamCaps()
    b.stream_caps.maximum_tile_count = 128
    b.stream_caps.maximum_logical_bytes = 65536
    b.stream_caps.maximum_tile_bytes = 8192
    b.stream_caps.maximum_reciprocal_candidates_per_q = 65536
    b.stream_caps.maximum_image_candidates_per_tile = 20000
    b.stream_caps.receiver_retained_numeric_bytes = (
        core._periodic_correlation_private_factor_store_receiver_bytes(b.schedule)
    )
    return b


def _writer(b, directory, *, cutoff=0.9):
    return core._make_periodic_correlation_private_factor_writer(
        b.reference, b.schedule, b.census, cutoff, 1e-12, str(directory), b.store_caps,
    )


def _populate(b, writer, *, cutoff=0.9, fail=2**64 - 1):
    return core._periodic_correlation_private_factor_store_stream_diagnostic(
        writer, b.reference, b.schedule, b.census, b.ao, b.auxiliary,
        cutoff, 1e-12, b.stream_caps, fail,
    )


def _tile(b, sequence, *, negative=1e-12, cutoff=0.9):
    d = b.schedule.descriptor(sequence)
    raw = core._build_periodic_correlation_reciprocal_metric(
        b.reference, b.schedule, b.census, b.sources[d.q_index], b.auxiliary,
    )
    w = core._factorize_periodic_correlation_metric(
        b.reference, b.schedule, b.census, raw, negative,
    )
    return core._build_periodic_correlation_three_center_tile(
        b.reference, b.schedule, b.census, b.sources[d.q_index], w,
        b.ao, b.auxiliary, sequence, cutoff, 20000,
    )


def test_exact_layout_partial_tiles_and_native_roundtrip(tmp_path):
    b = _configured()
    shape = b.schedule.shape
    expected_size = 1280 + 256 * shape.tile_count + shape.logical_bytes
    assert core._periodic_correlation_private_factor_store_file_bytes(b.schedule) == expected_size
    previous_end = 1024
    for sequence in range(shape.tile_count):
        d = b.schedule.descriptor(sequence)
        prefix = ((d.q_index * shape.n_kpoints + d.k_bra_index)
                  * shape.n_auxiliary * shape.n_ao_pairs
                  + d.ao_pair_begin * shape.n_auxiliary
                  + d.auxiliary_begin * d.ao_pair_count)
        expected = 1024 + sequence * 256 + 16 * prefix
        assert expected == previous_end
        assert core._periodic_correlation_private_factor_store_record_offset(b.schedule, sequence) == expected
        previous_end = expected + 256 + 16 * d.element_count
    assert previous_end + 256 == expected_size
    writer = _writer(b, tmp_path)
    assert list(tmp_path.iterdir()) == []  # file is already unlinked
    completion = _populate(b, writer)
    reader = writer.finish(completion)
    assert writer.state == core._PeriodicCorrelationPrivateFactorStoreState.FINALIZED
    assert reader.file_bytes == expected_size
    assert reader.producer_completion_identity_sha256 == completion.payload_identity_sha256
    assert reader.finite_image_reference and not reader.restart_supported
    assert not reader.ao_image_source_certified
    assert reader.image_cutoff_bohr == 0.9
    assert reader.image_policy == core._PERIODIC_CORRELATION_THREE_CENTER_IMAGE_POLICY
    assert reader.census_identity_sha256 == b.census.census_identity_sha256
    assert reader.ao_basis_identity_sha256 == b.config.ao_basis_identity_sha256
    assert reader.auxiliary_basis_identity_sha256 == b.config.auxiliary_basis_identity_sha256
    for sequence in (shape.tile_count - 1, 0, 3, 7, 1):
        expected = _tile(b, sequence).matrix
        np.testing.assert_array_equal(reader.read_tile(sequence, 8192).view(np.uint64), expected.view(np.uint64))
    assert len(reader.storage_identity_sha256) == 64


def test_native_store_admission_has_no_tile_bitmap_or_copied_publisher():
    b = _configured()
    plan = core._plan_periodic_correlation_factor_build(b.reference, b.schedule, b.census)
    c = plan.components
    assert c.publisher_bitmap_bytes == c.publisher_digest_table_bytes == c.publisher_buffer_bytes == 0
    assert c.encoded_generation_bytes == c.disk_generation_bytes == (
        1280 + 256 * b.schedule.shape.tile_count + b.schedule.shape.logical_bytes
    )
    assert c.journal_bytes == c.checkpoint_bytes == 0


def test_fixed_header_record_footer_have_independent_canonical_wire_hashes(tmp_path):
    b = _configured()
    writer = _writer(b, tmp_path)
    completion = _populate(b, writer)
    reader = writer.finish(completion)
    read = lambda at, n: core._private_factor_store_bytes_diagnostic(reader, at, n)
    def domain(name):
        encoded = name.encode()
        return struct.pack(">Q", len(encoded)) + encoded + struct.pack(">I", 1)
    header = read(0, 1024)
    assert header[:8] == b"VQPFST01"
    assert struct.unpack(">6I", header[8:32]) == (1, 1024, 256, 256, 1, 1)
    assert header[576:992] == bytes(416)
    expected = hashlib.sha256(domain("vibeqc.periodic.correlation.private-factor-store.header")
                              + header[:992]).digest()
    assert header[-32:] == expected
    root = hashlib.sha256(domain("vibeqc.periodic.correlation.private-factor-store.records")
                          + expected + struct.pack(">Q", reader.tile_count))
    for sequence in range(reader.tile_count):
        d = b.schedule.descriptor(sequence)
        at = core._periodic_correlation_private_factor_store_record_offset(b.schedule, sequence)
        record = read(at, 256)
        fields = struct.unpack(">QQQQiiiQQQQQ", record[:84])
        assert fields == (sequence, d.q_index, d.k_bra_index, d.k_ket_index,
                          *d.k_ket_reciprocal_wrap, d.ao_pair_begin, d.ao_pair_count,
                          d.auxiliary_begin, d.auxiliary_count, d.element_count)
        assert record[236:] == bytes(20)
        digest = hashlib.sha256(domain("vibeqc.periodic.correlation.private-factor-store.record")
                                + header[-32:] + record[:204] + bytes(32) + record[236:]).digest()
        assert record[204:236] == digest
        root.update(digest)
    footer = read(reader.file_bytes - 256, 256)
    assert footer[:8] == b"VQPFCM01"
    assert footer[48:80] == header[-32:]
    assert footer[80:112] == root.digest()
    assert footer[112:144].hex() == completion.payload_identity_sha256
    storage = hashlib.sha256(domain("vibeqc.periodic.correlation.private-factor-store.finalized")
                             + footer[:144] + bytes(32) + footer[176:]).hexdigest()
    assert footer[144:176].hex() == storage == reader.storage_identity_sha256


@pytest.mark.parametrize("field,value", [("publisher_buffer_count", 1), ("publisher_buffer_count", 2)])
def test_sequential_policy_requires_borrowed_zero_publisher_buffers(field, value):
    b = _configured()
    setattr(b.config, field, value)
    with pytest.raises(ValueError, match="borrowed publisher zero"):
        core._make_periodic_correlation_factor_build_census(
            b.reference, b.schedule, b.config, [s.factor_build_q_record() for s in b.sources],
        )


def test_sequential_policy_rejects_digest_table_and_legacy_unsupported_mode():
    b = _configured()
    c = b.config.codec
    c.digest_bytes_per_tile = 32
    b.config.codec = c
    with pytest.raises(ValueError, match="digest table"):
        core._make_periodic_correlation_factor_build_census(
            b.reference, b.schedule, b.config, [s.factor_build_q_record() for s in b.sources],
        )
    b.config.publisher_mode = core._PeriodicCorrelationFactorPublisherMode.SEQUENTIAL_APPEND_UNSUPPORTED
    with pytest.raises(ValueError, match="publication"):
        core._make_periodic_correlation_factor_build_census(
            b.reference, b.schedule, b.config, [s.factor_build_q_record() for s in b.sources],
        )


@pytest.mark.parametrize("changes", [{"extra_shortfall": 1}, {"control_shortfall": 1}, {"codec_mismatch": True}])
def test_store_requires_compiled_codec_and_all_receiver_control_bytes(tmp_path, changes):
    b = _configured(**changes)
    before = core._private_factor_store_live_owned_descriptors_diagnostic()
    with pytest.raises(ValueError, match="compiled codec"):
        _writer(b, tmp_path)
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ["maximum_file_bytes", "maximum_tile_bytes", "maximum_tile_count"])
def test_missing_or_insufficient_explicit_caps_open_nothing(tmp_path, field):
    b = _configured()
    before = core._private_factor_store_live_owned_descriptors_diagnostic()
    setattr(b.store_caps, field, 0)
    with pytest.raises((ValueError, OverflowError), match="caps"):
        _writer(b, tmp_path)
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before


@pytest.mark.parametrize("sequence", [1, 7])
def test_out_of_order_tile_permanently_fails_and_closes_private_storage(tmp_path, sequence):
    b = _configured()
    before = core._private_factor_store_live_owned_descriptors_diagnostic()
    writer = _writer(b, tmp_path)
    with pytest.raises(ValueError, match="sequence"):
        writer.accept(_tile(b, sequence))
    assert writer.state == core._PeriodicCorrelationPrivateFactorStoreState.FAILED
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before


def test_duplicate_and_changed_whitener_within_q_are_rejected(tmp_path):
    b = _configured()
    first = _tile(b, 0)
    writer = _writer(b, tmp_path)
    writer.accept(first)
    with pytest.raises(ValueError, match="sequence"):
        writer.accept(first)
    writer = _writer(b, tmp_path)
    writer.accept(first)
    with pytest.raises(ValueError, match="whitener changed within q"):
        writer.accept(_tile(b, 1, negative=1e-13))


def test_tile_census_and_cutoff_mismatch_are_rejected(tmp_path):
    b = _configured()
    other = _configured(block=3)
    writer = _writer(b, tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        writer.accept(_tile(other, 0))
    writer = _writer(b, tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        writer.accept(_tile(b, 0, cutoff=1.0))


@pytest.mark.parametrize("fail", [0, 2, 9])
def test_callback_failure_aborts_prefix_without_receipt_or_fd_leak(tmp_path, fail):
    b = _configured()
    before = core._private_factor_store_live_owned_descriptors_diagnostic()
    writer = _writer(b, tmp_path)
    with pytest.raises(RuntimeError, match="receiver cancelled"):
        _populate(b, writer, fail=fail)
    assert writer.state == core._PeriodicCorrelationPrivateFactorStoreState.ABORTED
    assert writer.accepted_tile_count == fail
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before
    assert list(tmp_path.iterdir()) == []


def test_receiver_inventory_declaration_is_checked_before_source(tmp_path):
    b = _configured()
    writer = _writer(b, tmp_path)
    b.stream_caps.receiver_retained_numeric_bytes = 0
    with pytest.raises(ValueError, match="receiver workspace"):
        _populate(b, writer)
    assert writer.accepted_tile_count == 0
    assert writer.state == core._PeriodicCorrelationPrivateFactorStoreState.ABORTED


@pytest.mark.parametrize("offset", [0, 1024 + 108, 1024 + 204])
def test_header_and_record_corruption_are_rejected_before_finalization(tmp_path, offset):
    b = _configured()
    before = core._private_factor_store_live_owned_descriptors_diagnostic()
    writer = _writer(b, tmp_path)
    completion = _populate(b, writer)
    core._private_factor_store_corrupt_diagnostic(writer, offset, 1)
    with pytest.raises(RuntimeError, match="corrupt|checksum"):
        writer.finish(completion)
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before


def test_payload_corruption_is_rejected_before_read_callback_and_poisons_reader(tmp_path):
    b = _configured()
    writer = _writer(b, tmp_path)
    completion = _populate(b, writer)
    core._private_factor_store_corrupt_diagnostic(writer, 1024 + 256 + 15, 1)
    reader = writer.finish(completion)
    with pytest.raises(RuntimeError, match="payload checksum"):
        reader.read_tile(0, 8192)
    with pytest.raises(RuntimeError, match="poisoned"):
        reader.read_tile(1, 8192)


def test_truncated_file_cannot_finalize(tmp_path):
    b = _configured()
    writer = _writer(b, tmp_path)
    completion = _populate(b, writer)
    core._private_factor_store_truncate_diagnostic(writer, writer.file_bytes - 1)
    with pytest.raises(RuntimeError, match="file extent"):
        writer.finish(completion)


def test_wrong_completion_and_incomplete_stream_do_not_finalize(tmp_path):
    b = _configured()
    writer = _writer(b, tmp_path)
    completed_writer = _writer(b, tmp_path)
    completion = _populate(b, completed_writer)
    with pytest.raises(ValueError, match="complete producer receipt"):
        writer.finish(completion)
    # Same resource identities/counts, different numerical cutoff stream.
    wrong = core._periodic_correlation_three_center_stream_diagnostic(
        b.reference, b.schedule, b.census, b.ao, b.auxiliary,
        1.0, 1e-12, b.stream_caps,
    )
    with pytest.raises(ValueError, match="completion checksum"):
        completed_writer.finish(wrong)


def test_reader_caps_copies_and_raii_descriptor_lifetime(tmp_path):
    b = _configured()
    before = core._private_factor_store_live_owned_descriptors_diagnostic()
    writer = _writer(b, tmp_path)
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before + 2
    reader = writer.finish(_populate(b, writer))
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before + 1
    with pytest.raises(ValueError, match="tile cap"):
        reader.read_tile(0, 1)
    original = reader.read_tile(0, 8192)
    copy = reader.read_tile(0, 8192)
    copy[:] = 123
    np.testing.assert_array_equal(reader.read_tile(0, 8192), original)
    del reader
    gc.collect()
    assert core._private_factor_store_live_owned_descriptors_diagnostic() == before
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("recursive", [False, True])
def test_reader_callback_errors_release_visit_guard_but_do_not_poison(tmp_path, recursive):
    b = _configured()
    writer = _writer(b, tmp_path)
    reader = writer.finish(_populate(b, writer))
    with pytest.raises(RuntimeError, match="callback failed|recursive"):
        core._private_factor_reader_callback_failure_diagnostic(reader, 0, recursive)
    np.testing.assert_array_equal(reader.read_tile(0, 8192), _tile(b, 0).matrix)


def test_private_directory_policy_preserves_existing_files(tmp_path):
    b = _configured()
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("user-owned", encoding="utf-8")
    writer = _writer(b, tmp_path)
    writer.abort()
    assert sentinel.read_text(encoding="utf-8") == "user-owned"
    tmp_path.chmod(0o755)
    try:
        with pytest.raises(ValueError, match="private"):
            _writer(b, tmp_path)
    finally:
        tmp_path.chmod(0o700)
    assert sentinel.read_text(encoding="utf-8") == "user-owned"
