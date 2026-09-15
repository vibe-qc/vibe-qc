"""Physical density-independent tiles for the matched native HF source."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _system, _oracle as _pair_oracle
from tests.test_periodic_correlation_reciprocal_metric import (
    _accepted_metric_records, _analytic_s_fourier, _two_s_metric_basis, _CanonicalDigest,
)
from tests.test_periodic_correlation_three_center import _bundle as _old_bundle, _tile as _old_tile
from tests.test_periodic_gaussian_metric import _live, _caps as _resource_caps, _CAP_FIELDS, _LIVE_FIELDS
from tests.test_periodic_gaussian_source_context import (
    _make as _context, _options, _caps as _basis_caps, _CAP_FIELDS as _BASIS_CAP_FIELDS,
)
from tests.test_periodic_gaussian_reciprocal_source import _source

_PLAN_FIELDS = (
    "n_basis",
    "n_auxiliary",
    "accepted_vector_count",
    "reciprocal_candidate_count",
    "reciprocal_capacity",
    "reciprocal_panel_count",
    "resident_whitener_bytes",
    "raw_panel_bytes",
    "compensation_bytes",
    "reciprocal_panel_bytes",
    "double_auxiliary_fourier_bytes",
    "ao_fourier_bytes",
    "fixed_fourier_numeric_workspace_bytes",
    "output_bytes",
    "assembly_owned_numeric_peak_bytes",
    "whitening_owned_numeric_peak_bytes",
    "owned_numeric_peak_bytes",
    "borrowed_basis_active_numeric_bytes",
    "fixed_inventoried_object_bytes",
    "per_replica_inventoried_bytes",
    "node_inventoried_bytes",
    "image_candidate_count",
    "retained_pair_image_count",
    "reciprocal_candidate_evaluations",
    "image_candidate_evaluations",
    "primitive_pair_evaluations_upper_bound",
    "raw_contraction_term_count",
    "whitening_term_count",
    "preflight_work_units_upper_bound",
    "work_units_upper_bound",
)
_DESCRIPTOR_FIELDS = ("q_index", "conjugate_q_index", "k_bra_index", "k_ket_index")
_RANGE_FIELDS = ("ao_pair_begin", "ao_pair_count", "auxiliary_begin", "auxiliary_count", "element_count")


def _bundle(*, q=1, mesh=(3, 1, 1), block=2, ao=None, auxiliary=None, lattice=None,
            cutoff=6.0, pair_begin=0, pairs=None, aux_begin=0, aux_count=None, k_bra=0):
    if ao is None:
        ao = _basis([(0, (0.1, -0.2, 0.15), [0.55], [0.7], True),
                     (0, (0.6, 0.3, -0.25), [0.7], [0.8], True)])
    auxiliary = _two_s_metric_basis() if auxiliary is None else auxiliary
    lattice = 2*np.pi*np.diag([1, 0.5, 1/3]) if lattice is None else np.asarray(lattice)
    options = _options()
    options.ao_pair_image_cutoff_bohr = cutoff
    options.metric_absolute_eigenvalue_threshold = 1e-10
    context = _context(system=_system(lattice), ao=ao, auxiliary=auxiliary,
                       mesh=core._RegularKMesh(list(mesh)), options=options)
    source = _source(context, q)
    mc = core._PeriodicGaussianMetricConfig()
    mc.reciprocal_block = block
    mc.whitener_column_block = auxiliary.nbasis
    mc.basis_verification_caps = _basis_caps()
    live, resources = _live(), _resource_caps()
    raw = core._build_periodic_gaussian_reciprocal_metric(source, ao, auxiliary, mc, live, resources)
    white = core._factorize_periodic_gaussian_metric(raw, auxiliary.nbasis, live, resources)
    config = core._PeriodicGaussianThreeCenterConfig()
    config.reciprocal_block = block
    config.basis_verification_caps = _basis_caps()
    caps = core._PeriodicGaussianThreeCenterCaps()
    caps.maximum_image_candidates = 20000
    caps.resources = resources
    selection = core._PeriodicGaussianThreeCenterSelection()
    selection.k_bra_index = k_bra
    selection.ao_pair_begin = pair_begin
    selection.ao_pair_count = ao.nbasis**2 - pair_begin if pairs is None else pairs
    selection.auxiliary_begin = aux_begin
    selection.auxiliary_count = auxiliary.nbasis - aux_begin if aux_count is None else aux_count
    return SimpleNamespace(ao=ao, auxiliary=auxiliary, context=context, source=source, whitener=white,
                           selection=selection, config=config, live=live, caps=caps)


def _call(b, *, plan=False, source=None, whitener=None, ao=None, auxiliary=None, caps=None, live=None):
    fn = (core._plan_periodic_gaussian_three_center_tile if plan
          else core._build_periodic_gaussian_three_center_tile)
    return fn(b.source if source is None else source, b.whitener if whitener is None else whitener,
              b.ao if ao is None else ao, b.auxiliary if auxiliary is None else auxiliary,
              b.selection, b.config, b.live if live is None else live, b.caps if caps is None else caps)


def _oracle(b, tile):
    records = _accepted_metric_records(b.source)
    vectors = np.asarray([r[1] for r in records])
    weights = np.asarray([r[3] for r in records])
    f = np.asarray([_analytic_s_fourier(b.auxiliary, r[1], r[2]) for r in records]).T
    ket = b.context.k_record(tile.descriptor.k_ket_index).cartesian
    rho, _ = _pair_oracle(b.ao, vectors, np.asarray(b.context.direct_lattice), ket,
                         b.context.options.ao_pair_image_cutoff_bohr, 4)
    raw = (f.conj()*weights) @ rho.reshape(b.ao.nbasis**2, -1).T
    metric = (f.conj()*weights) @ f.T
    eig, u = np.linalg.eigh(metric)
    keep = eig > b.context.options.metric_absolute_eigenvalue_threshold
    w = (u[:, keep]/np.sqrt(eig[keep])) @ u[:, keep].conj().T
    d = tile.descriptor
    return (w @ raw)[d.auxiliary_begin:d.auxiliary_begin+d.auxiliary_count,
                     d.ao_pair_begin:d.ao_pair_begin+d.ao_pair_count]


def _descriptor_wire(w, d):
    for name in _DESCRIPTOR_FIELDS:
        w.u64(getattr(d, name))
    for value in d.k_ket_reciprocal_wrap:
        w.i32(value)
    for name in _RANGE_FIELDS:
        w.u64(getattr(d, name))


def _payload_wire(t):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-three-center.payload", 1)
    for value in (t.context.source_context_identity_sha256, t.source_identity_sha256,
                  t.conjugate_source_identity_sha256, t.whitener_payload_identity_sha256):
        w.string(value)
    w.string(core._PERIODIC_GAUSSIAN_THREE_CENTER_IMAGE_POLICY)
    w.string(core._PERIODIC_GAUSSIAN_THREE_CENTER_LATTICE_POLICY)
    w.binary64(t.context.options.ao_pair_image_cutoff_bohr)
    for value in np.asarray(t.context.direct_lattice).ravel():
        w.binary64(value)
    for index in (t.descriptor.k_bra_index, t.descriptor.k_ket_index):
        k = t.context.k_record(index)
        w.u64(k.index)
        for value in k.modular_doubled_address:
            w.i32(value)
        for value in (*k.fractional, *k.cartesian):
            w.binary64(value)
    _descriptor_wire(w, t.descriptor)
    for name in ("image_candidate_count", "retained_pair_image_count", "accepted_vector_count", "output_bytes"):
        w.u64(getattr(t.plan, name))
    prefix = (648 + len("vibeqc.periodic.gaussian-three-center.payload")
        + len(core._PERIODIC_GAUSSIAN_THREE_CENTER_IMAGE_POLICY)
        + len(core._PERIODIC_GAUSSIAN_THREE_CENTER_LATTICE_POLICY))
    assert w.wire_bytes == prefix
    for z in t.matrix.ravel():
        w.complex128(z)
    return w.finish()


def _plan_wire(b, p):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-three-center.plan", 1)
    for value in (b.context.source_context_identity_sha256, b.source.source_identity_sha256,
                  b.source.conjugate_source_identity_sha256, b.whitener.payload_identity_sha256):
        w.string(value)
    _descriptor_wire(w, p.descriptor)
    for name in _PLAN_FIELDS:
        w.u64(getattr(p, name))
    w.u64(p.config.reciprocal_block)
    for name in _BASIS_CAP_FIELDS:
        w.u64(getattr(p.config.basis_verification_caps, name))
    for name in _LIVE_FIELDS:
        w.u64(getattr(p.live, name))
    w.u64(p.caps.maximum_image_candidates)
    for name in _CAP_FIELDS:
        w.u64(getattr(p.caps.resources, name))
    return w.finish()


@pytest.mark.parametrize("mesh,q,bra", [((1, 1, 1), 0, 0), ((3, 1, 1), 0, 1),
    ((3, 1, 1), 1, 0), ((3, 1, 1), 1, 2), ((3, 1, 1), 2, 1), ((2, 1, 1), 1, 1)])
def test_physical_gamma_nonself_nyquist_tiles_match_independent_s_gaussian_oracle(mesh, q, bra):
    b = _bundle(mesh=mesh, q=q, k_bra=bra)
    t = _call(b)
    np.testing.assert_allclose(t.matrix, _oracle(b, t), atol=2e-12, rtol=2e-11)
    assert t.payload_identity_sha256 == _payload_wire(t)
    assert t.plan.plan_identity_sha256 == _plan_wire(b, t.plan)
    assert t.context is b.context
    assert t.finite_image_reference and not t.ao_image_source_certified
    assert t.descriptor.k_ket_index == b.context.ket_index(bra, q)
    if q == 0 and bra == 1:
        assert np.max(np.abs(t.matrix.imag)) > 1e-6


def test_same_q_uses_native_canonical_ket_and_positive_bloch_phase():
    b = _bundle(q=1)
    first = _call(b)
    b.selection.k_bra_index = 2
    second = _call(b)
    assert second.descriptor.k_ket_index == 0
    assert list(second.descriptor.k_ket_reciprocal_wrap) == [1, 0, 0]
    assert np.max(np.abs(first.matrix-second.matrix)) > 1e-6
    assert first.source_identity_sha256 == second.source_identity_sha256
    assert first.payload_identity_sha256 != second.payload_identity_sha256


def test_partial_shell_crossing_pair_and_auxiliary_tiles_reconstruct_full_factor():
    ao = _basis([(0, (0, 0, 0), [0.65], [0.8], True),
                 (1, (0.2, -0.1, 0.3), [0.8], [0.7], True)])
    b = _bundle(ao=ao, cutoff=2.5)
    full = _call(b)
    parts = []
    for aux in range(2):
        row = []
        for begin in range(0, 16, 5):
            b.selection.auxiliary_begin, b.selection.auxiliary_count = aux, 1
            b.selection.ao_pair_begin, b.selection.ao_pair_count = begin, min(5, 16-begin)
            row.append(_call(b).matrix)
        parts.append(np.hstack(row))
    np.testing.assert_array_equal(np.vstack(parts), full.matrix)
    np.testing.assert_allclose(full.matrix, _oracle(b, full), atol=3e-12, rtol=3e-11)


def test_reciprocal_blocks_only_change_execution_plan_not_tile_payload():
    tiles = [_call(_bundle(block=block)) for block in (1, 2, 5)]
    for tile in tiles[1:]:
        np.testing.assert_array_equal(tile.matrix.view(np.uint64), tiles[0].matrix.view(np.uint64))
        assert tile.payload_identity_sha256 == tiles[0].payload_identity_sha256
        assert tile.plan.plan_identity_sha256 != tiles[0].plan.plan_identity_sha256


def test_shared_numeric_leaf_preserves_legacy_tile_on_identical_direct_lattice():
    # 2*pi*I round-trips through B=I exactly; other geometries deliberately
    # need not bit-match legacy's reconstructed rather than original A.
    b = _bundle(lattice=2*np.pi*np.eye(3))
    old = _old_bundle(ao=b.ao, auxiliary=b.auxiliary, reciprocal=b.context.reciprocal_lattice)
    expected = _old_tile(old, cutoff=6.0)
    actual = _call(b)
    np.testing.assert_array_equal(actual.matrix.view(np.uint64), expected.matrix.view(np.uint64))
    assert actual.payload_identity_sha256 != expected.payload_identity_sha256


def test_original_skew_direct_lattice_is_used_and_sealed():
    lattice = np.array([[6.0, 0.4, 0.1], [0.0, 4.7, 0.2], [0.0, 0.0, 4.1]])
    b = _bundle(lattice=lattice, cutoff=2.5, k_bra=1)
    t = _call(b)
    np.testing.assert_allclose(t.matrix, _oracle(b, t), atol=3e-12, rtol=3e-11)
    np.testing.assert_array_equal(t.context.direct_lattice, lattice)
    assert t.payload_identity_sha256 == _payload_wire(t)


def test_exact_memory_candidate_and_work_inventory_reserves_w_once():
    b = _bundle(pairs=3, aux_count=1)
    live = _live()
    live.replicas_per_node, live.other_retained_bytes_per_replica = 3, 17
    live.other_transient_bytes_per_replica, live.external_node_bytes = 23, 101
    p = _call(b, plan=True, live=live)
    a, pb, ab, n, g, k = (p.n_auxiliary, p.descriptor.ao_pair_count, p.descriptor.auxiliary_count,
                          p.accepted_vector_count, p.reciprocal_capacity, p.reciprocal_panel_count)
    assert p.resident_whitener_bytes == 16*a*a
    fourier_scratch = core._PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES
    assert p.fixed_fourier_numeric_workspace_bytes == fourier_scratch
    assert p.assembly_owned_numeric_peak_bytes == 32*a*pb+40*g+32*a*g+16*pb*g+fourier_scratch
    assert p.whitening_owned_numeric_peak_bytes == 16*a*pb+16*ab*pb
    assert p.output_bytes == 16*ab*pb
    assert p.per_replica_inventoried_bytes == (p.resident_whitener_bytes+p.owned_numeric_peak_bytes
        +p.borrowed_basis_active_numeric_bytes+p.fixed_inventoried_object_bytes+17+23+65536)
    assert p.node_inventoried_bytes == 101+3*p.per_replica_inventoried_bytes
    assert p.reciprocal_candidate_evaluations == 2*b.source.candidate_count
    assert p.image_candidate_evaluations == p.image_candidate_count*(1+k+n)
    assert p.primitive_pair_evaluations_upper_bound == n*p.retained_pair_image_count*b.context.inventory.ao.exponent_count**2
    assert p.raw_contraction_term_count == n*a*pb
    assert p.whitening_term_count == a*ab*pb
    assert p.plan_identity_sha256 == _plan_wire(b, p)
    inventory = b.context.inventory
    def scan(basis):
        return basis.borrowed_active_numeric_bytes//8 + basis.shell_count + basis.contraction_count
    scans = 2*scan(inventory.ao) + scan(inventory.auxiliary)
    lookup = inventory.ao.shell_count + inventory.ao.contraction_count + 1
    preflight = (inventory.work_units_upper_bound + 128*a*a + 1024*b.source.candidate_count
        + 128*scans + 512*pb*lookup + 128*b.caps.maximum_image_candidates + 4096)
    full = (preflight + 1024*b.source.candidate_count + 128*k*scans + 512*k*pb*lookup
        + 256*(k+1)*pb + 128*p.image_candidate_count*(k+n)
        + 1048576*p.primitive_pair_evaluations_upper_bound
        + 128*n*inventory.auxiliary.coefficient_count + 512*28*n*a
        + 64*n*a*pb + 64*n*a + 128*a*ab*pb + 128*(a*pb+ab*pb) + 4096)
    assert p.preflight_work_units_upper_bound == preflight
    assert p.work_units_upper_bound == full


def _exact_caps(b, p):
    caps = core._PeriodicGaussianThreeCenterCaps()
    caps.maximum_image_candidates = b.caps.maximum_image_candidates
    resources = _resource_caps()
    for name, value in zip(_CAP_FIELDS, (p.owned_numeric_peak_bytes, p.per_replica_inventoried_bytes,
            p.node_inventoried_bytes, p.reciprocal_candidate_evaluations, p.work_units_upper_bound)):
        setattr(resources, name, value)
    caps.resources = resources
    return caps


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_exact_cap_succeeds_and_cap_minus_one_rejects_without_consuming_w(field):
    b = _bundle()
    p = _call(b, plan=True)
    caps = _exact_caps(b, p)
    accepted = _call(b, caps=caps)
    resources = caps.resources
    setattr(resources, field, getattr(resources, field)-1)
    caps.resources = resources
    before = b.whitener.matrix
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _call(b, caps=caps)
    np.testing.assert_array_equal(b.whitener.matrix, before)
    np.testing.assert_array_equal(_call(b).matrix, accepted.matrix)


def test_image_caps_exact_and_minus_one_and_early_census_work_cap():
    b = _bundle()
    p = _call(b, plan=True)
    b.caps.maximum_image_candidates = p.image_candidate_count
    exact = _call(b)
    b.caps.maximum_image_candidates -= 1
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _call(b)
    b.caps.maximum_image_candidates = 20000
    resources = b.caps.resources
    resources.maximum_work_units = p.preflight_work_units_upper_bound-1
    b.caps.resources = resources
    with pytest.raises((ValueError, RuntimeError), match="preflight work"):
        _call(b, ao=_two_s_metric_basis())
    assert exact.plan.image_candidate_count == p.image_candidate_count


@pytest.mark.parametrize("kind", ["context", "q", "ao", "auxiliary"])
def test_foreign_context_q_or_actual_basis_mismatch_fails_closed(kind):
    b = _bundle()
    kwargs = {}
    if kind == "context":
        kwargs["whitener"] = _bundle().whitener
    elif kind == "q":
        kwargs["source"] = _source(b.context, 0)
    elif kind == "ao":
        kwargs["ao"] = _two_s_metric_basis()
    else:
        kwargs["auxiliary"] = b.ao
    with pytest.raises(ValueError, match="mismatch"):
        _call(b, **kwargs)


def test_larger_changed_basis_is_metadata_capped_before_nonfinite_hash_scan():
    b = _bundle()
    # More primitives than the authenticated stored census. NaN must not be
    # reached: generous user caps cannot expand the outer work inventory.
    ao = _basis([(0, (0.1, -0.2, 0.15), [0.55]*20, [np.nan]*20, True),
                 (0, (0.6, 0.3, -0.25), [0.7], [0.8], True)])
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _call(b, ao=ao)
    # Moving one primitive between roles keeps the combined reservation
    # unchanged. Actual digest comparison must still reject both new bases.
    auxiliary = _basis([(0, (0, 0, 0), [0.5, 1.0], [1.0, 0.75], True)])
    b = _bundle(auxiliary=auxiliary)
    shifted_ao = _basis([(0, (0.1, -0.2, 0.15), [0.55, 0.8], [0.7, 0.1], True),
                         (0, (0.6, 0.3, -0.25), [0.7], [0.8], True)])
    shifted_aux = _basis([(0, (0, 0, 0), [0.5], [1.0], True)])
    with pytest.raises(ValueError, match="content mismatch"):
        _call(b, ao=shifted_ao, auxiliary=shifted_aux)


@pytest.mark.parametrize("field,value", [("ao_pair_count", 0), ("auxiliary_count", 0),
    ("ao_pair_begin", 4), ("auxiliary_begin", 2), ("k_bra_index", 3)])
def test_invalid_selection_is_rejected(field, value):
    b = _bundle()
    setattr(b.selection, field, value)
    with pytest.raises(ValueError, match="selection"):
        _call(b)


def test_zero_controls_current_node_inventory_and_owner_lifetimes():
    b = _bundle()
    p = _call(b, plan=True)
    caps = _exact_caps(b, p)
    live = _live()
    live.other_retained_bytes_per_replica = 1
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _call(b, caps=caps, live=live)
    b.config.reciprocal_block = 0
    with pytest.raises(ValueError, match="selection"):
        _call(b)
    b.config.reciprocal_block = 2
    b.caps.maximum_image_candidates = 0
    with pytest.raises(ValueError, match="positive"):
        _call(b)
    b.caps.maximum_image_candidates = 20000
    t = _call(b)
    identity = t.payload_identity_sha256
    matrix = t.matrix
    matrix[:] = 100
    config = t.plan.config
    config.reciprocal_block = 99
    del b
    gc.collect()
    assert t.payload_identity_sha256 == identity == _payload_wire(t)
    assert t.plan.config.reciprocal_block == 2
    assert not np.all(t.matrix == 100)
    assert not hasattr(t, "state") and not hasattr(t, "calculation_id")


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_all_resource_caps_are_explicit_positive(field):
    b = _bundle()
    resources = b.caps.resources
    setattr(resources, field, 0)
    b.caps.resources = resources
    with pytest.raises(ValueError, match="positive"):
        _call(b)
