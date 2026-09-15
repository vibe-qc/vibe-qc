"""Actual frozen HF/CCSD snapshot to the complete bare particle-hole group.

The independent oracle expands T only in tiny tests, evaluates the ORIGINAL
three bare W seeds in the common tensor and projects the completed target.
Neither a full CCSD residual nor production periodic scaling is implied.
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_mp2 import (
    _physical, _run as _mp2, _controls as _mp2_controls,
)
from tests.test_periodic_gaussian_pair_ccsd import (
    _run as _ccsd, _controls as _ccsd_controls,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _physical as _domain_physical, _run as _domain_mp2,
    _controls as _domain_mp2_controls,
)
from tests.test_periodic_gaussian_domain_pair_ccsd_t import (
    _run as _domain_ccsd, _controls as _domain_ccsd_controls,
)
from tests.test_periodic_gaussian_selected_local_ccsd_t import _dense_case
from tests.test_periodic_gaussian_pair_interaction import _controls as _interaction_controls
from tests.test_bounded_restricted_pair_ccsd_particle_hole import _original_three_seed_oracle


def _case(kind="he2", nk=2, *, domain=False, cutoff=0.0, iterations=80):
    if domain:
        b = _domain_physical(nk, kind=kind, full=(kind == "he"))
        warm = _domain_mp2(b, _domain_mp2_controls(b, cutoff=cutoff))
        result = _domain_ccsd(b, warm, _domain_ccsd_controls(b, iterations=iterations))
        dense = b.expected_dense
    else:
        b, provider = _physical(kind, nk)
        b.provider = provider
        dense = _dense_case(b)
        warm = _mp2(b, provider, _mp2_controls(cutoff=cutoff))
        result = _ccsd(b, provider, warm, _ccsd_controls(provider, iterations=iterations))
    o, n = dense["o"], dense["v"]
    labels = list(combinations_with_replacement(range(o), 2))
    frames = {ij: warm.pair(*ij).coefficients_copy() for ij in labels}
    amplitudes = {ij: result.solver.pair_copy(*ij) for ij in labels}
    oracle = (o, n, dense["eri"], frames, amplitudes)
    return SimpleNamespace(b=b, warm=warm, ccsd=result, dense=dense,
        domain=domain, oracle=oracle, labels=labels)


@pytest.fixture(scope="module", params=[("he", 2, False), ("he", 3, False),
    ("he2", 2, False), ("he2", 2, True), ("frozen", 2, True)])
def actual(request):
    kind, nk, domain = request.param
    return _case(kind, nk, domain=domain)


@pytest.fixture(scope="module")
def k2():
    return _case("he2", 2, domain=True)


def _controls(case):
    config, options, _, interaction = _interaction_controls(case,
        auxiliary_block=min(2, case.b.auxiliary.nbasis),
        ao_pair_block=min(3, case.b.ao.nbasis**2))
    live = core._PeriodicGaussianPairParticleHoleLiveInventory()
    live.other_live_numerical_bytes_per_worker = 2**20
    live.other_live_control_bytes_per_worker = live.backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianPairParticleHoleCaps()
    caps.interaction = interaction
    caps.maximum_source_slots = caps.maximum_pair_count = 16
    caps.maximum_owned_numerical_bytes = caps.maximum_interaction_control_bytes = 2**22
    caps.maximum_control_storage_bytes = 2**23
    caps.maximum_worker_bytes, caps.maximum_node_bytes = 2**26, 2**27
    caps.maximum_work_units = 10**19
    caps.maximum_factor_panels, caps.maximum_tile_calls = 1024, 65536
    caps.maximum_reciprocal_candidates = caps.maximum_image_candidates = 10**18
    return config, options, live, caps


def _arguments(case, i=0, j=0, controls=None, **changes):
    config, options, live, caps = _controls(case) if controls is None else controls
    b = case.b
    args = dict(hf=b.hf, reference=b.reference, wannier=b.wannier,
        common_geometry=b.builder if case.domain else b.domain,
        space=None if case.domain else b.space, basis=b.basis,
        warmstart=case.warm, ccsd=case.ccsd, target_i=i, target_j=j,
        config=config, options=options, live=live, caps=caps)
    args.update(changes)
    return args


def _plan(case, i=0, j=0, controls=None, **changes):
    return core._plan_periodic_gaussian_pair_particle_hole(**_arguments(case, i, j, controls, **changes))


def _build(case, i=0, j=0, controls=None, **changes):
    args = _arguments(case, i, j, controls)
    args.update(ao_basis=case.b.ao, auxiliary_basis=case.b.auxiliary, gauges=case.b.gauge)
    args.update(changes)
    return core._build_periodic_gaussian_pair_particle_hole(**args)


def _check(result, case, i, j):
    expected = _original_three_seed_oracle(case.oracle, i, j)
    np.testing.assert_allclose(result.residual_copy(), expected, atol=5e-12, rtol=3e-10)
    p, d = result.memory, result.diagnostics
    o, _, _, frames, _ = case.oracle
    rank = frames[min(i, j), max(i, j)].shape[1]
    ranks = [frames[min(l, m), max(l, m)].shape[1] for l in (i, j) for m in range(o)]
    assert p.target_i == i and p.target_j == j and p.target_dimension == rank
    maximum_rank = max(ranks)
    assert p.maximum_source_dimension == maximum_rank
    assert p.source_slots == d.completed_source_slots == d.completed_interactions == 2*o
    assert p.domain_generated == case.domain
    assert p.accumulator_owned_bytes == 16*rank**2+8*rank*maximum_rank
    assert p.maximum_transpose_bytes == 8*rank*maximum_rank
    assert p.contraction_phase_bytes == p.accumulator_owned_bytes+32*rank*maximum_rank
    assert p.peak_owned_numerical_bytes == max(p.contraction_phase_bytes,
        p.accumulator_owned_bytes+p.maximum_interaction_owned_bytes)
    assert p.output_numerical_bytes == 8*rank**2
    assert d.accumulator.scalar_products == sum(4*rank*b*b+2*rank*rank*b for b in ranks)
    assert d.completed_factor_panels <= p.factor_panels and d.completed_tile_calls <= p.tile_calls
    assert d.charged_work_units_upper_bound <= p.work_units
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.worker_bytes
    assert result.ccsd_identity_sha256 == case.ccsd.identity_sha256
    assert result.ccsd_amplitude_payload_sha256 == case.ccsd.solver.payload_sha256
    assert result.warmstart_identity_sha256 == case.warm.identity_sha256
    assert result.origin_provider_identity_sha256 == case.b.provider.identity_sha256
    assert result.hf_reference_source_identity_sha256 == case.b.hf.reference_source_identity_sha256
    assert result.pair_spaces_identity_sha256 == case.warm.pair_spaces_identity_sha256
    assert result.matched_finite_gaussian_hf_recipe
    assert not result.original_provider_projection_reproduced
    assert not result.entire_ccsd_residual and not result.production_dlpno
    assert not result.infinite_source_accuracy_certified
    return expected


def test_actual_multi_k_bare_group_matches_original_common_tensor_and_reversal(actual):
    assert actual.warm.converged and actual.ccsd.converged
    o = actual.oracle[0]
    outputs = {}
    for i, j in sorted({(0, 0), (0, o-1), (o-1, 0)}):
        result = _build(actual, i, j)
        outputs[i, j] = result.residual_copy()
        _check(result, actual, i, j)
    np.testing.assert_allclose(outputs[0, o-1], outputs[o-1, 0].T, atol=5e-12, rtol=3e-10)
    np.testing.assert_allclose(outputs[0, 0], outputs[0, 0].T, atol=5e-12, rtol=3e-10)
    assert max(np.linalg.norm(value) for value in outputs.values()) > 1e-7


_CAPS = dict(maximum_source_slots="source_slots", maximum_pair_count="pair_count",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_control_storage_bytes="control_storage_reservation_bytes",
    maximum_worker_bytes="worker_bytes", maximum_node_bytes="required_node_memory_bytes",
    maximum_work_units="work_units", maximum_interaction_control_bytes="maximum_interaction_control_bytes",
    maximum_factor_panels="factor_panels", maximum_tile_calls="tile_calls",
    maximum_reciprocal_candidates="reciprocal_candidate_evaluations",
    maximum_image_candidates="image_candidate_evaluations")


@pytest.mark.parametrize("field,plan_field", list(_CAPS.items()))
def test_all_outer_caps_minus_one_precede_gauge_payload_and_source_factories(k2, field, plan_field):
    controls = _controls(k2)
    p = _plan(k2, 0, 0, controls)
    value = getattr(p, plan_field)
    assert value > 0
    setattr(controls[-1], field, value-1)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|exceed|insufficient"):
        _build(k2, 0, 0, controls, gauges=np.full_like(k2.b.gauge, np.nan))


def test_empty_pair_amplitudes_still_visit_all_physical_source_slots():
    case = _case("he", 2, domain=True, cutoff=1.0)
    assert case.ccsd.converged
    assert all(c.shape[1] == 0 for c in case.oracle[3].values())
    result = _build(case)
    _check(result, case, 0, 0)
    assert result.residual_copy().shape == (0, 0)
    assert result.diagnostics.accumulator.zero_rank_sources == 2*case.oracle[0]


def test_unconverged_snapshot_cannot_enter_the_physical_frozen_adapter():
    case = _case("he", 2, iterations=1)
    assert not case.ccsd.converged
    with pytest.raises((ValueError, RuntimeError), match="converg|snapshot"):
        _plan(case)


def test_original_source_and_gauge_checks_remain_active(k2):
    other = _case("he2", 2, domain=True)
    with pytest.raises((ValueError, RuntimeError), match="state|owner|source|context|warm"):
        _plan(k2, warmstart=other.warm)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="finite|gauge"):
        _build(k2, gauges=np.full_like(k2.b.gauge, np.nan))


def test_result_detaches_from_source_owners_without_mutable_amplitude_escape():
    case = _case("he", 2)
    expected = _original_three_seed_oracle(case.oracle, 0, 1)
    result = _build(case, 0, 1)
    before = result.residual_copy()
    assert not before.flags.writeable
    before.setflags(write=True)
    before[:] = 123
    assert not hasattr(result, "warmstart") and not hasattr(result, "ccsd")
    identity = result.identity_sha256
    del case
    gc.collect()
    np.testing.assert_allclose(result.residual_copy(), expected, atol=5e-12, rtol=3e-10)
    assert result.identity_sha256 == identity
