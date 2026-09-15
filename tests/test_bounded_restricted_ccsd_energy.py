"""Tiny scalar accessor energy oracles; no chemistry or solver certification."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_ccsd_target import _problem


def _inventory(**changes):
    inventory = core._BoundedRestrictedCCSDEnergyInventory()
    inventory.maximum_integral_work_units_per_query = 7
    for name, value in changes.items():
        setattr(inventory, name, value)
    return inventory


def _plan(p, inventory=None, **kwargs):
    inventory = _inventory() if inventory is None else inventory
    return core._plan_bounded_restricted_ccsd_energy(
        p["o"], p["v"], inventory,
        p["t1"].nbytes + p["t2"].nbytes, p["eri"].nbytes, **kwargs)


def _caps(plan):
    caps = core._BoundedRestrictedCCSDEnergyCaps()
    caps.maximum_total_numerical_bytes = plan.total_live_numerical_bytes
    caps.maximum_control_storage_bytes = plan.total_control_storage_bytes
    caps.maximum_singles_calls = plan.singles_calls
    caps.maximum_doubles_calls = plan.doubles_calls
    caps.maximum_integral_calls = plan.integral_calls
    caps.maximum_total_work_units = plan.total_work_units_upper_bound
    return caps


def _run(p, *, inventory=None, caps=None, fov=None, **kwargs):
    inventory = _inventory() if inventory is None else inventory
    plan_keys = ("amplitude_transient_bytes", "integral_transient_bytes", "singles_work", "doubles_work")
    if caps is None:
        caps = _caps(_plan(p, inventory, **{k: kwargs[k] for k in plan_keys if k in kwargs}))
    if fov is None:
        fov = np.ascontiguousarray(p["fock"][:p["o"], p["o"]:])
    return core._bounded_restricted_ccsd_energy_diagnostic(
        p["t1"], p["t2"], fov, p["eri"], inventory, caps, **kwargs)


def _oracle(p):
    o = p["o"]
    g = p["eri"][:o, o:, :o, o:].transpose(0, 2, 1, 3)
    tau = p["t2"] + np.einsum("ia,jb->ijab", p["t1"], p["t1"])
    singles = 2 * np.einsum("ia,ia->", p["fock"][:o, o:], p["t1"])
    doubles = np.einsum("ijab,ijab->", 2 * g - g.transpose(0, 1, 3, 2), tau)
    return singles, doubles


@pytest.mark.parametrize("o,v", [(1, 1), (1, 3), (2, 2), (2, 4), (3, 2), (4, 2)])
def test_full_common_energy_and_exact_callback_census(o, v):
    p = _problem(o, v, seed=818)
    result = _run(p, singles_work=3, doubles_work=11)
    first, second = _oracle(p)
    assert result.singles_fock_energy == pytest.approx(first, abs=2e-14)
    assert result.doubles_and_disconnected_energy == pytest.approx(second, abs=2e-14)
    assert result.correlation_energy == pytest.approx(first + second, abs=3e-14)
    plan = result.memory
    assert result.singles_calls == plan.singles_calls == o*v + 2*o**2*v**2
    assert result.doubles_calls == plan.doubles_calls == o**2*v**2
    assert result.integral_calls == plan.integral_calls == 2*o**2*v**2
    assert result.charged_amplitude_work_units == 3*result.singles_calls + 11*result.doubles_calls
    assert result.charged_integral_work_units == 7*result.integral_calls
    assert result.charged_total_work_units == plan.total_work_units_upper_bound
    assert result.charged_total_work_units == (
        result.charged_amplitude_work_units + result.charged_integral_work_units
        + plan.kernel_work_units_upper_bound)


def test_independent_antisymmetric_spin_orbital_energy():
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_energy

    p = _problem(2, 3, seed=72)
    o, v = p["o"], p["v"]
    n = o + v
    f = np.zeros((2*n, 2*n))
    f[::2, ::2] = p["fock"]
    f[1::2, 1::2] = p["fock"]
    t1 = np.zeros((2*o, 2*v))
    t1[::2, ::2] = p["t1"]
    t1[1::2, 1::2] = p["t1"]
    t2 = np.zeros((2*o, 2*o, 2*v, 2*v))
    for si in (0, 1):
        for sj in (0, 1):
            t2[si::2, sj::2, si::2, sj::2] += p["t2"]
            t2[si::2, sj::2, sj::2, si::2] -= p["t2"].transpose(0, 1, 3, 2)
    expected = so_energy(f, _spin_orbital_eri(p["factors"], n), t1, t2,
                         slice(0, 2*o), slice(2*o, 2*n))
    assert _run(p).correlation_energy == pytest.approx(expected, abs=4e-14)


def test_complete_occupied_and_virtual_rotations_preserve_energy():
    p = _problem(2, 3, seed=616)
    rng = np.random.default_rng(717)
    qo = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    qv = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotation = np.zeros((5, 5))
    rotation[:2, :2], rotation[2:, 2:] = qo, qv
    rotated = dict(p)
    rotated["fock"] = np.ascontiguousarray(rotation.T @ p["fock"] @ rotation)
    rotated["t1"] = np.ascontiguousarray(qo.T @ p["t1"] @ qv)
    rotated["t2"] = np.ascontiguousarray(np.einsum(
        "iI,jJ,aA,bB,ijab->IJAB", qo, qo, qv, qv, p["t2"]))
    rotated["eri"] = np.ascontiguousarray(np.einsum(
        "pP,qQ,rR,sS,pqrs->PQRS", rotation, rotation, rotation, rotation, p["eri"]))
    assert _run(rotated).correlation_energy == pytest.approx(_run(p).correlation_energy, abs=4e-14)


def test_singles_product_outside_target_pair_space_must_not_be_projected():
    p = _problem(1, 2)
    p["fock"][:] = 0
    p["t1"][:] = [[0.0, 0.3]]
    p["t2"][:] = 0
    p["t2"][0, 0, 0, 0] = 0.2  # connected T2 lives only in target e0
    factor = np.zeros((3, 3))
    factor[0, 1:] = [0.7, 1.1]
    factor[1:, 0] = [0.7, 1.1]
    p["eri"] = np.einsum("pq,rs->pqrs", factor, factor)
    result = _run(p)
    connected = 0.7**2 * 0.2
    outside_singles = 1.1**2 * 0.3**2
    assert result.correlation_energy == pytest.approx(connected + outside_singles, abs=1e-16)
    assert abs(result.correlation_energy - connected) > 0.1
    # Projecting tau into target e0 incorrectly discards exactly this term.
    target = np.eye(2)[:, :1]
    tau = p["t2"][0, 0] + np.outer(p["t1"][0], p["t1"][0])
    wrong_tau = target @ (target.T @ tau @ target) @ target.T
    g = p["eri"][0, 1:, 0, 1:]
    assert np.sum(g * wrong_tau) == pytest.approx(connected, abs=1e-16)


def test_all_ordered_pairs_and_no_per_cell_or_pair_divisor():
    p = _problem(2, 1)
    p["fock"][:] = 0
    p["t1"][:] = 0
    p["t2"][:] = 0
    p["t2"][0, 1, 0, 0] = p["t2"][1, 0, 0, 0] = 0.4
    factor = np.zeros((3, 3))
    factor[:2, 2] = factor[2, :2] = [1.0, 3.0]
    p["eri"] = np.einsum("pq,rs->pqrs", factor, factor)
    assert _run(p).correlation_energy == pytest.approx(2 * 3.0 * 0.4, abs=1e-15)


def test_zero_amplitudes_still_query_every_entry():
    p = _problem(2, 3)
    p["t1"][:] = p["t2"][:] = 0
    result = _run(p)
    assert result.correlation_energy == result.singles_fock_energy == result.doubles_and_disconnected_energy == 0
    assert result.singles_calls == 78
    assert result.doubles_calls == 36
    assert result.integral_calls == 72


def test_no_numerical_heap_and_explicit_all_live_roles():
    p = _problem(2, 3)
    inv = _inventory(other_live_numerical_bytes=123, other_live_control_bytes=456)
    plan = _plan(p, inv, amplitude_transient_bytes=23, integral_transient_bytes=47)
    assert plan.peak_owned_numerical_bytes == 0
    assert plan.borrowed_f_ov_bytes == 8*2*3
    assert plan.total_live_numerical_bytes == (
        8*2*3 + p["t1"].nbytes + p["t2"].nbytes + p["eri"].nbytes + 123 + 23 + 47)
    assert plan.total_control_storage_bytes == plan.fixed_inventoried_object_bytes + 456
    _run(p, inventory=inv, caps=_caps(plan), amplitude_transient_bytes=23, integral_transient_bytes=47)


@pytest.mark.parametrize("field", [
    "maximum_total_numerical_bytes", "maximum_control_storage_bytes",
    "maximum_singles_calls", "maximum_doubles_calls", "maximum_integral_calls",
    "maximum_total_work_units",
])
@pytest.mark.parametrize("zero", [False, True])
def test_each_cap_admitted_before_fov_scan_or_first_callback(field, zero):
    p = _problem(2, 2)
    caps = _caps(_plan(p))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    fov = np.full((2, 2), np.nan)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(p, caps=caps, fov=fov, fail_kind="singles")


@pytest.mark.parametrize("kind", ["singles", "doubles", "integrals"])
@pytest.mark.parametrize("before", [0, 1, 7])
def test_callback_exceptions_abort_without_mutating_input(kind, before):
    p = _problem(2, 2)
    copies = {key: p[key].copy() for key in ("t1", "t2", "fock", "eri")}
    with pytest.raises(RuntimeError, match="injected callback failure"):
        _run(p, fail_kind=kind, fail_before_call=before)
    for key, copy in copies.items():
        np.testing.assert_array_equal(p[key], copy)


@pytest.mark.parametrize("key", ["t1", "t2", "eri", "fock"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_or_callback_values_fail_closed(key, value):
    p = _problem(1, 1)
    if key in ("t1", "t2"):
        p[key].flat[0] = value
    elif key == "eri":
        p[key][0, 1, 0, 1] = value
    else:
        p[key][0, 1] = value
    with pytest.raises((ValueError, OverflowError), match="finite"):
        _run(p)


def test_fov_scan_precedes_amplitude_callback():
    p = _problem(1, 1)
    with pytest.raises(ValueError, match="Fov input must be finite"):
        _run(p, fov=np.array([[np.nan]]), fail_kind="singles")


def test_descriptors_and_controls_are_snapshotted_before_callbacks():
    p = _problem(2, 2)
    ordinary = _run(p)
    mutated = _run(p, alter_external_controls=True)
    assert ordinary.correlation_energy == mutated.correlation_energy
    assert ordinary.charged_total_work_units == mutated.charged_total_work_units


def test_compensated_sum_preserves_small_term_between_cancelling_large_terms():
    p = _problem(1, 3)
    p["t1"][:] = 1
    p["t2"][:] = 0
    p["eri"][:] = 0
    p["fock"][0, 1:] = [5e15, 0.5, -5e15]
    result = _run(p)
    assert result.correlation_energy == math.fsum([1e16, 1.0, -1e16]) == 1.0


def test_gradual_subnormal_terms_are_not_flushed():
    p = _problem(1, 1)
    tiny = np.nextafter(0.0, 1.0)
    p["t1"][:] = 1
    p["t2"][:] = 0
    p["fock"][:] = 0
    p["fock"][0, 1] = tiny
    p["eri"][:] = 0
    p["eri"][0, 1, 0, 1] = tiny
    result = _run(p)
    assert result.singles_fock_energy == 2*tiny
    assert result.doubles_and_disconnected_energy == tiny
    assert result.correlation_energy == 3*tiny


def test_nonfinite_intermediate_is_an_explicit_failure_not_an_infinite_energy():
    p = _problem(1, 1)
    p["eri"][0, 1, 0, 1] = np.finfo(float).max
    with pytest.raises(OverflowError, match="not finite"):
        _run(p)


@pytest.mark.parametrize("o,v", [(0, 1), (1, 0), (2**63, 2), (2**32, 2**32)])
def test_metadata_planner_rejects_zero_or_overflowing_dimensions(o, v):
    with pytest.raises((ValueError, OverflowError)):
        core._plan_bounded_restricted_ccsd_energy(o, v, _inventory())


@pytest.mark.parametrize("name", ["singles_work", "doubles_work", "integral_work"])
def test_positive_per_query_work_is_required(name):
    inv = _inventory(maximum_integral_work_units_per_query=0 if name == "integral_work" else 7)
    kwargs = {name: 0} if name != "integral_work" else {}
    with pytest.raises(ValueError, match="query work"):
        core._plan_bounded_restricted_ccsd_energy(1, 1, inv, **kwargs)


@pytest.mark.parametrize("mode", ["dtype", "noncontiguous", "shape", "unaligned"])
def test_diagnostic_never_casts_copies_or_reinterprets_invalid_fov(mode):
    p = _problem(2, 2)
    fov = np.ones((2, 2))
    if mode == "dtype":
        fov = fov.astype(np.complex128)
    elif mode == "noncontiguous":
        fov = np.ones((2, 4))[:, ::2]
    elif mode == "shape":
        fov = np.ones((2, 3))
    else:
        fov = np.ndarray((2, 2), dtype=np.float64, buffer=bytearray(33), offset=1)
    with pytest.raises(ValueError, match="binary64|shape"):
        _run(p, fov=fov)
