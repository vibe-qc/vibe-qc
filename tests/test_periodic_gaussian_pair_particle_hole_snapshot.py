"""Actual-source bare PH at numerical trial snapshots, not a CCSD solver.

The tiny diagnostic supplies only numerical singles and T. Pair coefficients
are borrowed from exact native warm owners. The independent oracle expands
the full common T and contracts the ORIGINAL three bare target seeds before
target projection. No trial convergence or physical singles is implied.
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_particle_hole import (
    _controls, _CAPS, _case as _frozen_case, _build as _frozen_build,
    _plan as _frozen_plan,
)
from tests.test_periodic_gaussian_pair_mp2 import (
    _physical, _run as _mp2, _controls as _mp2_controls,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _physical as _domain_physical, _run as _domain_mp2,
    _controls as _domain_mp2_controls,
)
from tests.test_periodic_gaussian_selected_local_ccsd_t import _dense_case
from tests.test_bounded_restricted_pair_ccsd_amplitudes import (
    _options as _reader_options, _caps as _base_reader_caps,
)
from tests.test_bounded_restricted_pair_ccsd_particle_hole import _original_three_seed_oracle


def _trial(case, seed=2851):
    rng = np.random.default_rng(seed)
    o, n = case.dense["o"], case.dense["v"]
    case.labels = list(combinations_with_replacement(range(o), 2))
    case.frames = {ij: case.warm.pair(*ij).coefficients_copy() for ij in case.labels}
    case.sc, case.st, case.pt = [], [], []
    for i in range(o):
        # Separate numerical singles, deliberately unrelated to diagonal PNOs.
        rank = (i+1) % (n+1)
        c = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :rank]
        case.sc.append(np.ascontiguousarray(c))
        case.st.append(np.ascontiguousarray(.01*rng.normal(size=rank)))
    for i, j in case.labels:
        r = case.frames[i, j].shape[1]
        t = .025*rng.normal(size=(r, r))
        # Exact symmetric TEST data; the native reader never repairs T.
        case.pt.append(np.ascontiguousarray(.5*(t+t.T) if i == j else t))
    return case


def _case(kind="he", nk=2, *, domain=False, cutoff=0.0):
    if domain:
        b = _domain_physical(nk, kind=kind, full=(kind == "he"))
        warm = _domain_mp2(b, _domain_mp2_controls(b, cutoff=cutoff))
        dense = b.expected_dense
    else:
        b, provider = _physical(kind, nk)
        b.provider = provider
        dense = _dense_case(b)
        warm = _mp2(b, provider, _mp2_controls(cutoff=cutoff))
    return _trial(SimpleNamespace(b=b, warm=warm, dense=dense, domain=domain))


def _reader_caps():
    caps = _base_reader_caps()
    # Whole physical ancestors are declared during diagnostic Reader creation.
    caps.maximum_per_replica_inventoried_bytes = 2**26
    caps.maximum_node_inventoried_bytes = 2**27
    return caps


def _arguments(case, i=0, j=0, controls=None, **changes):
    config, options, live, caps = _controls(case) if controls is None else controls
    b = case.b
    result = dict(hf=b.hf, reference=b.reference, wannier=b.wannier,
        common_geometry=b.builder if case.domain else b.domain,
        space=None if case.domain else b.space, basis=b.basis,
        warmstart=case.warm, target_i=i, target_j=j,
        config=config, options=options, live=live, caps=caps,
        singles_coefficients=case.sc, singles_amplitudes=case.st,
        pair_amplitudes=case.pt, reader_options=_reader_options(1e-10),
        reader_caps=_reader_caps())
    result.update(changes)
    return result


def _plan(case, i=0, j=0, controls=None, **changes):
    return core._plan_periodic_gaussian_pair_particle_hole_snapshot_diagnostic(
        **_arguments(case, i, j, controls, **changes))


def _build(case, i=0, j=0, controls=None, **changes):
    args = _arguments(case, i, j, controls)
    args.update(ao_basis=case.b.ao, auxiliary_basis=case.b.auxiliary, gauges=case.b.gauge)
    args.update(changes)
    return core._build_periodic_gaussian_pair_particle_hole_snapshot_diagnostic(**args)


def _oracle(case, i=0, j=0, pt=None):
    amplitudes = dict(zip(case.labels, case.pt if pt is None else pt))
    return _original_three_seed_oracle((case.dense["o"], case.dense["v"],
        case.dense["eri"], case.frames, amplitudes), i, j)


def _check(result, case, i=0, j=0):
    np.testing.assert_allclose(result.residual_copy(), _oracle(case, i, j), atol=5e-12, rtol=3e-10)
    p, d = result.memory.stream, result.diagnostics
    o = case.dense["o"]
    a = case.frames[min(i, j), max(i, j)].shape[1]
    ranks = [case.frames[min(l, m), max(l, m)].shape[1] for l in (i, j) for m in range(o)]
    b = max(ranks)
    assert p.target_i == i and p.target_j == j and p.target_dimension == a
    assert p.maximum_source_dimension == b
    assert p.source_slots == d.completed_source_slots == d.completed_interactions == 2*o
    assert p.domain_generated == case.domain
    assert p.borrowed_ccsd_numerical_bytes == 0
    assert p.borrowed_warmstart_numerical_bytes == (case.warm.diagnostics.retained_pair_bytes
        +case.warm.diagnostics.retained_pair_geometry_bytes+case.warm.solver.memory.output_numerical_bytes)
    assert result.memory.borrowed_reader_numerical_bytes == sum(
        x.nbytes for x in case.sc+case.st+list(case.frames.values())+case.pt)
    assert result.memory.reader_warm_coefficient_padding_bytes == sum(c.nbytes for c in case.frames.values())
    assert result.memory.borrowed_reader_table_bytes > 0 and result.memory.reader_control_storage_bytes > 0
    assert p.accumulator_owned_bytes == 16*a*a+8*a*b
    assert p.contraction_phase_bytes == p.accumulator_owned_bytes+32*a*b
    assert p.peak_owned_numerical_bytes == max(p.contraction_phase_bytes,
        p.accumulator_owned_bytes+p.maximum_interaction_owned_bytes)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.worker_bytes
    assert p.output_numerical_bytes == 8*a*a
    assert d.accumulator.scalar_products == sum(4*a*r*r+2*a*a*r for r in ranks)
    assert d.charged_work_units_upper_bound <= p.work_units
    assert result.warmstart_identity_sha256 == case.warm.identity_sha256
    assert result.origin_provider_identity_sha256 == case.b.provider.identity_sha256
    assert result.hf_reference_source_identity_sha256 == case.b.hf.reference_source_identity_sha256
    assert result.pair_spaces_identity_sha256 == case.warm.pair_spaces_identity_sha256
    for key in ("identity_sha256", "payload_sha256", "snapshot_identity_sha256",
                "consumed_interactions_identity_sha256"):
        assert len(getattr(result, key)) == 64
    for key in ("converged_snapshot_certified", "singles_physically_certified",
                "original_provider_projection_reproduced", "entire_ccsd_residual",
                "production_dlpno", "infinite_source_accuracy_certified"):
        assert not getattr(result, key)


@pytest.fixture(scope="module", params=[("he", 2, False), ("he", 3, False),
    ("he2", 2, True), ("frozen", 2, True)])
def actual(request):
    kind, nk, domain = request.param
    return _case(kind, nk, domain=domain)


@pytest.fixture(scope="module")
def k2():
    return _case()


def test_trial_snapshots_match_original_common_bare_group_and_reversal(actual):
    o = actual.dense["o"]
    results = {}
    for i, j in sorted({(0, 0), (0, o-1), (o-1, 0)}):
        result = _build(actual, i, j, snapshot_index=7)
        _check(result, actual, i, j)
        assert result.snapshot_index == 7
        results[i, j] = result.residual_copy()
    np.testing.assert_allclose(results[0, o-1], results[o-1, 0].T, atol=5e-12, rtol=3e-10)
    np.testing.assert_allclose(results[0, 0], results[0, 0].T, atol=5e-12, rtol=3e-10)


def test_linear_doubles_and_numerical_singles_have_distinct_snapshot_semantics(k2):
    initial = _build(k2, 0, 1)
    doubled = _build(k2, 0, 1, pair_amplitudes=[2*t for t in k2.pt])
    np.testing.assert_allclose(doubled.residual_copy(), 2*initial.residual_copy(), atol=2e-15, rtol=2e-13)
    assert doubled.snapshot_identity_sha256 != initial.snapshot_identity_sha256
    singles = _build(k2, 0, 1, singles_amplitudes=[t+.123 for t in k2.st])
    np.testing.assert_array_equal(singles.residual_copy(), initial.residual_copy())
    assert singles.snapshot_identity_sha256 != initial.snapshot_identity_sha256
    assert singles.identity_sha256 != initial.identity_sha256
    index = _build(k2, 0, 1, snapshot_index=19)
    assert index.snapshot_identity_sha256 == initial.snapshot_identity_sha256
    assert index.payload_sha256 == initial.payload_sha256
    assert index.identity_sha256 != initial.identity_sha256


@pytest.mark.parametrize("field,plan_field", list(_CAPS.items()))
def test_all_outer_caps_precede_physical_gauge_reads_not_prior_reader_creation(k2, field, plan_field):
    controls = _controls(k2)
    value = getattr(_plan(k2, controls=controls).stream, plan_field)
    assert value > 0
    setattr(controls[-1], field, value-1)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|exceed|insufficient"):
        _build(k2, controls=controls, gauges=np.full_like(k2.b.gauge, np.nan))


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_borrowed_numerical_bytes", "borrowed_reader_numerical_bytes"),
    ("maximum_table_bytes", "borrowed_reader_table_bytes"),
])
def test_reader_own_caps_precede_trial_payload_scan(k2, field, plan_field):
    caps = _reader_caps()
    setattr(caps, field, getattr(_plan(k2), plan_field)-1)
    bad = [np.full_like(t, np.nan) for t in k2.pt]
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|exceed"):
        _plan(k2, pair_amplitudes=bad, reader_caps=caps)


def test_explicit_caller_lives_are_not_subtracted_or_hidden(k2):
    first = _plan(k2)
    controls = _controls(k2)
    controls[2].other_live_numerical_bytes_per_worker += 1234
    second = _plan(k2, controls=controls)
    assert second.stream.worker_bytes-first.stream.worker_bytes == 1234
    assert second.stream.required_node_memory_bytes-first.stream.required_node_memory_bytes == 1234*first.stream.replicas_per_node
    assert second.borrowed_reader_numerical_bytes == first.borrowed_reader_numerical_bytes


def test_equal_authentic_frame_receipts_do_not_replace_exact_warm_C_storage(k2):
    previous = core.get_num_threads()
    try:
        core.set_num_threads(1)
        first = _mp2(k2.b, k2.b.provider)
        second = _mp2(k2.b, k2.b.provider)
    finally:
        core.set_num_threads(previous)
    case = _trial(SimpleNamespace(b=k2.b, warm=first, dense=k2.dense, domain=False))
    assert first.identity_sha256 == second.identity_sha256
    for ij in case.labels:
        assert first.pair(*ij).identity_sha256 == second.pair(*ij).identity_sha256
        np.testing.assert_array_equal(first.pair(*ij).coefficients_copy(), second.pair(*ij).coefficients_copy())
    _plan(case, coefficient_warmstart=first)
    with pytest.raises((ValueError, RuntimeError), match="exact warm coefficient storage"):
        _plan(case, coefficient_warmstart=second)


def test_unfinished_CCSD_numbers_are_valid_trial_data_but_not_a_frozen_certificate():
    case = _frozen_case("he", 2, iterations=1)
    assert not case.ccsd.converged
    with pytest.raises((ValueError, RuntimeError), match="converg|snapshot"):
        _frozen_plan(case)
    _trial(case)
    case.pt = [case.ccsd.solver.pair_copy(*ij) for ij in case.labels]
    result = _build(case, 0, 1, snapshot_index=1)
    _check(result, case, 0, 1)
    assert not result.converged_snapshot_certified


def test_same_final_T_matches_frozen_engine_without_reusing_final_owner_certificate():
    case = _frozen_case("he", 2)
    frozen = _frozen_build(case, 0, 1)
    _trial(case)
    case.sc = [case.ccsd.singles_coefficients_copy(i) for i in range(case.dense["o"])]
    case.st = [case.ccsd.solver.singles_copy(i) for i in range(case.dense["o"])]
    case.pt = [case.ccsd.solver.pair_copy(*ij) for ij in case.labels]
    current = _build(case, 0, 1)
    np.testing.assert_array_equal(current.residual_copy(), frozen.residual_copy())
    assert current.payload_sha256 == frozen.payload_sha256
    assert current.identity_sha256 != frozen.identity_sha256
    assert not hasattr(current, "ccsd_identity_sha256")


def test_generic_split_result_cannot_be_relabelled_physical_but_T_is_usable_trial_data(k2):
    from tests.test_periodic_gaussian_pair_ccsd_split import _run as _split

    case = SimpleNamespace(**vars(k2))
    case.ccsd = _split(case.b, case.b.provider, case.warm)
    assert case.ccsd.converged and case.ccsd.split_bare_particle_hole
    assert not case.ccsd.particle_hole_physical_source_certified
    with pytest.raises((ValueError, RuntimeError), match="split CCSD source is not physically certified"):
        _frozen_plan(case)
    case.pt = [case.ccsd.solver.pair_copy(*ij) for ij in case.labels]
    current = _build(case, 0, 1)
    _check(current, case, 0, 1)
    assert not current.converged_snapshot_certified


def test_zero_rank_pair_frames_keep_numerical_singles_and_all_source_visits():
    case = _case("he", 2, domain=True, cutoff=1.0)
    assert all(c.shape[1] == 0 for c in case.frames.values())
    assert sum(t.size for t in case.st) > 0
    result = _build(case)
    _check(result, case)
    assert result.residual_copy().shape == (0, 0)
    assert result.memory.reader_warm_coefficient_padding_bytes == 0
    assert result.diagnostics.accumulator.zero_rank_sources == 2*case.dense["o"]


@pytest.mark.parametrize("fault", ["dtype", "stride", "count", "nonfinite", "diagonal", "gram"])
def test_invalid_numeric_snapshot_inputs_fail_closed(k2, fault):
    changes = {}
    if fault == "count":
        changes["pair_amplitudes"] = k2.pt[:-1]
    elif fault == "gram":
        sc = [c.copy() for c in k2.sc]
        sc[0] *= 2
        changes["singles_coefficients"] = sc
    else:
        pt = [t.copy() for t in k2.pt]
        if fault == "dtype":
            pt[0] = pt[0].astype(np.float32)
        elif fault == "stride":
            pt[0] = pt[0][:, ::-1]
        elif fault == "nonfinite":
            pt[0][0, 0] = np.nan
        else:
            assert pt[0].shape[0] >= 2
            pt[0][0, 1] += .1
        changes["pair_amplitudes"] = pt
    with pytest.raises((TypeError, ValueError, RuntimeError, OverflowError)):
        _plan(k2, **changes)


def test_result_detaches_and_exposes_no_mutable_reader_or_kernel_child():
    case = _case()
    result = _build(case, 0, 1)
    expected, identity = _oracle(case, 0, 1), result.identity_sha256
    copy = result.residual_copy()
    assert not copy.flags.writeable
    copy.setflags(write=True)
    copy[:] = 123
    for name in ("reader", "warmstart", "ccsd", "bare_result"):
        assert not hasattr(result, name)
    del case
    gc.collect()
    np.testing.assert_allclose(result.residual_copy(), expected, atol=5e-12, rtol=3e-10)
    assert result.identity_sha256 == identity
