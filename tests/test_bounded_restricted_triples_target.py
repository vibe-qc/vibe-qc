"""Guo (2018) operator/energy leaf: tiny supplied-moment arithmetic oracles.

No physical triples producer or iterative method route is exercised. Dense
T3 appears only in the tiny independent occupied-Fock test oracle, never in
the native input contract, which borrows one neighbouring triple at a time.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _case(occupied=(0, 1, 2), o=3, v=2, dmax=3, seed=517):
    rng = np.random.default_rng(seed)
    fock = 0.1 * rng.normal(size=(o, o))
    fock = np.ascontiguousarray(fock + fock.T - np.eye(o))
    target_orbitals = np.linalg.qr(rng.normal(size=(6, v)))[0]
    sources, overlaps = [], []
    for axis in range(3):
        for l in range(o):
            if l == occupied[axis]:
                sources.append(None)
                overlaps.append(None)
                continue
            d = 1 + (axis + l) % dmax
            source_orbitals = np.linalg.qr(rng.normal(size=(6, d)))[0]
            sources.append(np.ascontiguousarray(0.1 * rng.normal(size=(d, d, d))))
            overlaps.append(np.ascontiguousarray(target_orbitals.T @ source_orbitals))
    return dict(
        o=o, v=v, dmax=dmax, occupied=occupied,
        amplitudes=np.ascontiguousarray(0.1 * rng.normal(size=(v, v, v))),
        connected=np.ascontiguousarray(0.1 * rng.normal(size=(v, v, v))),
        singles=np.ascontiguousarray(0.03 * rng.normal(size=(v, v, v))),
        energies=np.linspace(0.3, 1.2, v),
        fock_rows=np.ascontiguousarray(fock[list(occupied)]),
        sources=sources, overlaps=overlaps,
    )


def _retained(case):
    return sum(array.nbytes for array in case["sources"] + case["overlaps"] if array is not None)


def _plan(case, transient=0):
    return core._plan_bounded_restricted_triples_target(
        case["o"], case["v"], case["dmax"], _retained(case), transient
    )


def _caps(plan):
    caps = core._BoundedRestrictedTriplesTargetCaps()
    caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    caps.maximum_total_numerical_bytes = plan.total_live_numerical_bytes
    caps.maximum_provider_visits = max(1, plan.provider_visits_upper_bound)
    caps.maximum_kernel_work_units = plan.kernel_work_units_upper_bound
    return caps


def _run(case, caps=None, **kwargs):
    if caps is None:
        caps = _caps(_plan(case, kwargs.get("provider_transient_bytes", 0)))
    return core._bounded_restricted_triples_target_residual_diagnostic(
        case["amplitudes"], case["connected"], case["singles"], case["energies"],
        case["fock_rows"], case["occupied"], case["sources"], case["overlaps"],
        case["dmax"], caps, **kwargs,
    )


def _adapt(t):
    return (4 * t - 2 * t.transpose(0, 2, 1) - 2 * t.transpose(2, 1, 0)
            - 2 * t.transpose(1, 0, 2) + t.transpose(1, 2, 0) + t.transpose(2, 0, 1))


def _oracle(case):
    t, w, u, eps, rows = (case[key] for key in ("amplitudes", "connected", "singles", "energies", "fock_rows"))
    delta = eps[:, None, None] + eps[None, :, None] + eps[None, None, :]
    delta = delta - sum(rows[axis, case["occupied"][axis]] for axis in range(3))
    residual = w + delta * t
    count = 0
    for axis in range(3):
        for l in range(case["o"]):
            if l == case["occupied"][axis] or rows[axis, l] == 0:
                continue
            count += 1
            s = case["overlaps"][axis * case["o"] + l]
            source = case["sources"][axis * case["o"] + l]
            residual -= rows[axis, l] * np.einsum("ax,by,cz,xyz->abc", s, s, s, source)
    return residual, np.sum(_adapt(t) * (w + u)), count


@pytest.mark.parametrize("occupied", [(0, 1, 2), (2, 0, 1), (1, 1, 2), (1, 1, 1)])
def test_arbitrary_neighbour_spaces_match_independent_trilinear_projection(occupied):
    case = _case(occupied)
    expected, energy, visits = _oracle(case)
    result = _run(case, amplitude_snapshot_id=91)
    np.testing.assert_allclose(result.residual, expected, rtol=3e-14, atol=3e-14)
    assert result.raw_energy_contraction == pytest.approx(energy, rel=3e-14, abs=3e-14)
    assert result.provider_visits == visits == 6
    assert result.amplitude_snapshot_id == 91
    assert tuple(result.occupied) == occupied
    assert result.maximum_absolute_residual == pytest.approx(np.max(np.abs(expected)))
    assert result.residual_frobenius_norm == pytest.approx(np.linalg.norm(expected))
    # In particular, raw arbitrary moments are NOT silently multiplied by
    # zero for i=j=k, or by any unique-triple/per-cell/spin multiplicity.
    assert abs(result.raw_energy_contraction) > 1e-7


def _common_space_case(occupied, tensors, moments, fock, eps):
    o, v = tensors.shape[0], tensors.shape[3]
    sources, overlaps = [], []
    for axis in range(3):
        for l in range(o):
            if l == occupied[axis]:
                sources.append(None)
                overlaps.append(None)
            else:
                neighbour = list(occupied)
                neighbour[axis] = l
                sources.append(np.ascontiguousarray(tensors[tuple(neighbour)]))
                overlaps.append(np.eye(v))
    return dict(
        o=o, v=v, dmax=v, occupied=occupied,
        amplitudes=np.ascontiguousarray(tensors[occupied]),
        connected=np.ascontiguousarray(moments[occupied]),
        singles=np.zeros((v, v, v)), energies=np.ascontiguousarray(eps),
        fock_rows=np.ascontiguousarray(fock[list(occupied)]),
        sources=sources, overlaps=overlaps,
    )


def test_full_occupied_fock_linear_solve_satisfies_operator_but_t0_does_not():
    rng = np.random.default_rng(441)
    o = v = 2
    fock = np.array([[-1.1, 0.13], [0.13, -0.8]])
    eps = np.array([0.4, 0.9])
    moment = 0.1 * rng.normal(size=(o, o, o, v, v, v))
    # Independent dense 8x8 occupied Kronecker operator, ONLY a toy oracle.
    identity = np.eye(o)
    occupied_action = (np.kron(fock, np.kron(identity, identity))
                       + np.kron(identity, np.kron(fock, identity))
                       + np.kron(identity, np.kron(identity, fock)))
    solution = np.empty_like(moment)
    for a, b, c in itertools.product(range(v), repeat=3):
        operator = (eps[a] + eps[b] + eps[c]) * np.eye(o**3) - occupied_action
        solution[:, :, :, a, b, c] = np.linalg.solve(operator, -moment[:, :, :, a, b, c].ravel()).reshape(o, o, o)
    for occupied in ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)):
        case = _common_space_case(occupied, solution, moment, fock, eps)
        np.testing.assert_allclose(_run(case).residual, 0, atol=1e-15)
    virtual_sum = eps[:, None, None] + eps[None, :, None] + eps[None, None, :]
    t0 = np.empty_like(solution)
    for occupied in itertools.product(range(o), repeat=3):
        t0[occupied] = -moment[occupied] / (virtual_sum - sum(fock[index, index] for index in occupied))
    defect = _run(_common_space_case((0, 0, 1), t0, moment, fock, eps))
    assert defect.maximum_absolute_residual > 1e-4
    assert defect.provider_visits == 3


def _canonical_moments(t1, t2, eri, occupied):
    """Independent test-only Riplinger (2013) Eq.(8)/(9) W and Guo Eq.(1) U."""
    o, v = t1.shape
    w, u = np.zeros((v, v, v)), np.zeros((v, v, v))
    for abc in itertools.product(range(v), repeat=3):
        for order in itertools.permutations(range(3)):
            i, j, k = (occupied[index] for index in order)
            a, b, c = (abc[index] for index in order)
            w[abc] += sum(t2[k, j, c, d] * eri[i, o + a, o + b, o + d] for d in range(v))
            w[abc] -= sum(t2[i, l, a, b] * eri[k, o + c, j, l] for l in range(o))
        i, j, k = occupied
        a, b, c = abc
        u[abc] = (t1[i, a] * eri[j, o + b, k, o + c]
                  + t1[j, b] * eri[i, o + a, k, o + c]
                  + t1[k, c] * eri[j, o + b, i, o + a])
    return w, u


def test_diagonal_limit_guo_energy_matches_independent_spin_orbital_triples_anchor():
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_triples_correction

    rng = np.random.default_rng(918)
    o, v, p = 2, 3, 5
    n = o + v
    factors = 0.1 * rng.normal(size=(p, n, n))
    factors = np.ascontiguousarray(factors + factors.transpose(0, 2, 1))
    eri = np.einsum("Ppq,Prs->pqrs", factors, factors)
    t1 = np.ascontiguousarray(0.05 * rng.normal(size=(o, v)))
    raw = 0.04 * rng.normal(size=(o, o, v, v))
    t2 = np.ascontiguousarray(raw + raw.transpose(1, 0, 3, 2))
    eo, ev = np.array([-1.2, -0.8]), np.array([0.3, 0.8, 1.3])
    fock = np.diag(eo)
    guo_energy = 0.0
    for occupied in itertools.combinations_with_replacement(range(o), 3):
        w, u = _canonical_moments(t1, t2, eri, occupied)
        denominator = ev[:, None, None] + ev[None, :, None] + ev[None, None, :] - sum(eo[index] for index in occupied)
        case = dict(o=o, v=v, dmax=v, occupied=occupied, amplitudes=-w / denominator,
                    connected=w, singles=u, energies=ev, fock_rows=np.ascontiguousarray(fock[list(occupied)]),
                    sources=[None] * (3 * o), overlaps=[None] * (3 * o))
        result = _run(case)
        np.testing.assert_allclose(result.residual, 0, atol=1e-16)
        assert result.provider_visits == 0
        i, j, k = occupied
        # Guo's unique i<=j<=k multiplicity belongs to the caller, not leaf.
        guo_energy += (2 - int(i == j) - int(j == k)) * result.raw_energy_contraction
    t1so = np.zeros((2 * o, 2 * v))
    t1so[0::2, 0::2] = t1
    t1so[1::2, 1::2] = t1
    t2so = np.zeros((2 * o, 2 * o, 2 * v, 2 * v))
    for si, sj in itertools.product((0, 1), repeat=2):
        t2so[si::2, sj::2, si::2, sj::2] += t2
        t2so[si::2, sj::2, sj::2, si::2] -= t2.transpose(0, 1, 3, 2)
    spin_energy = so_triples_correction(
        np.repeat(np.r_[eo, ev], 2), _spin_orbital_eri(factors, n),
        t1so, t2so, slice(0, 2 * o), slice(2 * o, 2 * n),
    )
    assert guo_energy == pytest.approx(spin_energy, rel=2e-12, abs=2e-14)
    native_energy = core.dlpno_spatial_triples_correction(
        t1, t2.reshape(o * o, v * v),
        np.ascontiguousarray(factors[:, :o, o:].reshape(p, o * v)),
        np.ascontiguousarray(factors[:, :o, :o].reshape(p, o * o)),
        np.ascontiguousarray(factors[:, o:, o:].reshape(p, v * v)), eo, ev,
    )
    assert guo_energy == pytest.approx(native_energy, rel=2e-12, abs=2e-14)


def test_exact_peak_and_active_source_plus_declared_provider_accounting():
    o, v, d = 3, 4, 5
    p = core._plan_bounded_restricted_triples_target(o, v, d, 123, 456)
    assert p.output_bytes == p.compensation_bytes == 8 * v**3
    assert p.projection_workspace_bytes == 8 * (d * d + v * d)
    assert p.peak_owned_numerical_bytes == 8 * (2 * v**3 + d * d + v * d)
    assert p.borrowed_target_bytes == 8 * (3 * v**3 + v + 3 * o)
    assert p.active_provider_view_bytes_upper_bound == 8 * (d**3 + v * d)
    assert p.total_live_numerical_bytes == p.peak_owned_numerical_bytes + p.borrowed_target_bytes + p.active_provider_view_bytes_upper_bound + 123 + 456
    assert p.provider_visits_upper_bound == 3 * (o - 1)


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
                                   "maximum_provider_visits", "maximum_kernel_work_units"])
@pytest.mark.parametrize("zero", [False, True])
def test_caps_reject_before_any_provider_visit(field, zero):
    case = _case()
    caps = _caps(_plan(case))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(case, caps=caps, fail_before_visit=0)


@pytest.mark.parametrize("fault", [1, 2, 3, 4, 5])
def test_provider_protocol_and_snapshot_fail_closed(fault):
    with pytest.raises((ValueError, RuntimeError), match="exactly.once|snapshot mismatch|labels"):
        _run(_case(), protocol_fault=fault)


@pytest.mark.parametrize("before", [0, 2, 5])
def test_provider_failure_does_not_publish_or_mutate_inputs(before):
    case = _case()
    original = case["amplitudes"].copy()
    with pytest.raises(RuntimeError, match="injected provider failure"):
        _run(case, fail_before_visit=before)
    np.testing.assert_array_equal(case["amplitudes"], original)
    assert _run(case).provider_visits == 6


def test_only_exact_zero_couplings_may_be_skipped():
    case = _case()
    for axis in range(3):
        for l in range(case["o"]):
            if l != case["occupied"][axis]:
                case["fock_rows"][axis, l] = 0
    # A tiny nonzero value is still a required provider visit, not F_Cut.
    case["fock_rows"][0, 1] = 1e-300
    result = _run(case)
    assert result.provider_visits == 1
    assert result.exactly_zero_couplings_skipped == 5
    case["sources"][1] = None
    case["overlaps"][1] = None
    with pytest.raises(RuntimeError, match="missing nonzero-coupling"):
        _run(case)


def test_repeated_occupied_rows_must_represent_one_fock_operator():
    case = _case((0, 0, 0))
    case["fock_rows"][1, 2] += 0.01
    with pytest.raises(ValueError, match="different Fock rows"):
        _run(case, fail_before_visit=0)


def test_no_couplings_for_single_occupied_orbital_and_no_hidden_multiplicity():
    case = _case((0, 0, 0), o=1, v=2)
    result = _run(case, fail_before_visit=0)
    expected, energy, visits = _oracle(case)
    assert visits == result.provider_visits == 0
    np.testing.assert_allclose(result.residual, expected, atol=1e-14)
    assert result.raw_energy_contraction == pytest.approx(energy)
    assert abs(energy) > 1e-7


@pytest.mark.parametrize("field", ["amplitudes", "connected", "singles", "energies", "fock_rows"])
def test_nonfinite_target_inputs_fail_before_provider(field):
    case = _case()
    case[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="input must be finite"):
        _run(case, fail_before_visit=0)


def test_nonfinite_source_and_overlap_rejected_inside_borrowed_visit():
    for field in ("sources", "overlaps"):
        case = _case()
        case[field][1].flat[0] = np.nan
        with pytest.raises(ValueError, match="input must be finite"):
            _run(case)


def test_snapshot_source_size_and_provider_memory_are_not_implicit():
    case = _case()
    with pytest.raises(ValueError, match="positive amplitude snapshot"):
        _run(case, amplitude_snapshot_id=0)
    with pytest.raises((ValueError, RuntimeError), match="byte cap"):
        _run(case, caps=_caps(_plan(case)), provider_transient_bytes=1)
    _run(case, provider_transient_bytes=1)
    case["dmax"] = 1
    with pytest.raises((ValueError, RuntimeError), match="dimension exceeds"):
        _run(case)


def test_arithmetic_overflow_copy_ownership_and_dtype_checks():
    case = _case()
    result = _run(case)
    copy = result.residual
    copy[:] = 42
    assert not np.array_equal(copy, result.residual)
    case["amplitudes"][:] = 1e308
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="not finite"):
        _run(case)
    case = _case()
    case["amplitudes"] = np.asfortranarray(case["amplitudes"])
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(case)
    case["amplitudes"] = np.ascontiguousarray(case["amplitudes"], dtype=np.float32)
    with pytest.raises(ValueError, match="binary64"):
        _run(case)


@pytest.mark.parametrize("o,v,d", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (2**63, 2, 2), (2, 2**32, 1), (2, 1, 2**32)])
def test_plan_rejects_invalid_or_overflowing_extents_without_allocation(o, v, d):
    with pytest.raises((ValueError, OverflowError)):
        core._plan_bounded_restricted_triples_target(o, v, d)


def _empty_neighbour_case():
    case = _case()
    for slot, source in enumerate(case["sources"]):
        if source is not None:
            case["sources"][slot] = np.empty((0, 0, 0))
            case["overlaps"][slot] = np.empty((case["v"], 0))
    return case


def test_present_rank_zero_neighbours_contribute_zero_but_keep_every_coupling_visit():
    case = _empty_neighbour_case()
    expected, energy, visits = _oracle(case)
    result = _run(case)
    np.testing.assert_allclose(result.residual, expected, rtol=3e-14, atol=3e-14)
    assert result.raw_energy_contraction == pytest.approx(energy, rel=3e-14, abs=3e-14)
    assert result.provider_visits == visits == 6
    assert result.exactly_zero_couplings_skipped == 0
    assert result.largest_source_virtual_dimension == 0
    assert _retained(case) == 0
    # An empty subspace is not a missing neighbour; absence still rejects.
    case["sources"][1] = case["overlaps"][1] = None
    with pytest.raises(RuntimeError, match="missing nonzero-coupling"):
        _run(case)


def test_mixed_zero_and_nonzero_neighbour_ranks_match_independent_projection():
    case = _case()
    case["sources"][1] = np.empty((0, 0, 0))
    case["overlaps"][1] = np.empty((case["v"], 0))
    expected, energy, visits = _oracle(case)
    result = _run(case)
    np.testing.assert_allclose(result.residual, expected, rtol=3e-14, atol=3e-14)
    assert result.raw_energy_contraction == pytest.approx(energy, rel=3e-14, abs=3e-14)
    assert result.provider_visits == visits == 6
    assert result.largest_source_virtual_dimension == max(
        a.shape[0] for a in case["sources"] if a is not None)


@pytest.mark.parametrize("fault", range(1, 8))
def test_empty_neighbour_retains_exactly_once_label_snapshot_and_empty_extent_protocol(fault):
    with pytest.raises((ValueError, RuntimeError), match="exactly.once|snapshot mismatch|labels|exact empty"):
        _run(_empty_neighbour_case(), protocol_fault=fault)


@pytest.mark.parametrize("malformed", ["amplitude", "overlap"])
def test_empty_neighbour_diagnostic_requires_exact_zero_shapes(malformed):
    case = _empty_neighbour_case()
    if malformed == "amplitude":
        case["sources"][1] = np.empty((0, 0, 1))
    else:
        case["overlaps"][1] = np.empty((case["v"], 1))
    with pytest.raises(ValueError, match="shape mismatch"):
        _run(case)


def test_empty_neighbour_with_tiny_nonzero_fock_still_requires_a_visit():
    case = _empty_neighbour_case()
    for axis in range(3):
        for l in range(case["o"]):
            if l != case["occupied"][axis]:
                case["fock_rows"][axis, l] = 0
    case["fock_rows"][0, 1] = 1e-300
    result = _run(case)
    assert result.provider_visits == 1
    assert result.exactly_zero_couplings_skipped == 5
    assert result.largest_source_virtual_dimension == 0
