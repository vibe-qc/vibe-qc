"""Actual finite-Gaussian HF -> pair PNOs -> coupled ragged MP2 reference.

Tiny He/He2 only. Dense operators here are independent test oracles, not
runtime inputs. No automatic PAO domains or production DLPNO claim.
"""
from __future__ import annotations

import gc
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _he2_bundle, _he2_controls, _prepare_leaves, _dense_case,
)
from tests.test_periodic_gaussian_real_local_provider import _make as _provider
from tests.test_periodic_gaussian_pair_pnos import _options as _pno_options, _caps as _pno_caps


def _controls(*, cutoff=0.0, iterations=80):
    options = core._PeriodicGaussianPairMP2Options()
    options.pnos = _pno_options(occupation_cutoff=cutoff)
    projection = core._PeriodicGaussianPairSpaceOptions()
    projection.maximum_diagonal_symmetry_projection_error = 1e-11
    options.projection = projection
    solver = core._BoundedRestrictedPairMP2SolverOptions()
    solver.maximum_iterations, solver.denominator_floor = iterations, 1e-8
    solver.residual_tolerance, solver.energy_tolerance = 1e-12, 1e-13
    solver.coefficient_orthogonality_tolerance = 1e-10
    solver.maximum_diagonal_update_antisymmetry_norm = 1e-11
    options.solver, options.maximum_occupied_virtual_fock_norm = solver, 1e-10
    live = core._PeriodicGaussianPairMP2LiveInventory()
    live.other_live_bytes_per_worker = 2**20  # Other tiny fixture/oracle owners.
    live.fixed_backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianPairMP2Caps()
    caps.pnos = _pno_caps()
    projected = core._PeriodicGaussianPairSpaceCaps()
    projected.maximum_owned_numerical_bytes = 65536
    projected.maximum_per_worker_inventoried_bytes = 2**24
    projected.maximum_node_inventoried_bytes = 2**26
    projected.maximum_integral_calls, projected.maximum_work_units = 1024, 10**12
    caps.projection = projected
    sc = core._BoundedRestrictedPairMP2SolverCaps()
    sc.maximum_occupied_count = sc.maximum_common_virtual_dimension = sc.maximum_pair_rank = 4
    sc.maximum_pair_count = 10
    sc.maximum_owned_numerical_bytes = 2**20
    sc.maximum_per_replica_inventoried_bytes, sc.maximum_node_inventoried_bytes = 2**24, 2**26
    sc.maximum_coupling_slots, sc.maximum_work_units = 100000, 10**12
    caps.solver = sc
    caps.maximum_owned_numerical_bytes = 2**22
    caps.maximum_per_worker_inventoried_bytes, caps.maximum_node_inventoried_bytes = 2**24, 2**26
    caps.maximum_pair_count, caps.maximum_integral_calls = 10, 10000
    caps.maximum_progress_callbacks, caps.maximum_work_units = 256, 10**13
    return options, live, caps


def _physical(system="he", nk=1, *, full=True):
    if system == "he2":
        b = _he2_bundle((nk, 1, 1))
        _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    else:
        b = _prepare_leaves(_bundle((nk, 1, 1), full=full))
    return b, _provider(b)


def _run(b, provider, controls=None, callback=None):
    options, live, caps = _controls() if controls is None else controls
    return core._run_periodic_gaussian_pair_mp2(b.reference, b.basis, provider, options, live, caps, callback)


def _plan(b, provider, controls=None):
    options, live, caps = _controls() if controls is None else controls
    return core._plan_periodic_gaussian_pair_mp2(b.reference, b.basis, provider, options, live, caps)


def _pair_data(result):
    return [(i, j, result.pair(i, j).coefficients_copy(), result.pair(i, j).exchange_integrals_copy())
        for i, j in combinations_with_replacement(range(result.memory.occupied_count), 2)]


def _projected_oracle(b, result):
    """Assemble the independent Galerkin operator in the COMMON full basis.

    No native residual, overlap, projected-G or denominator implementation
    supplies this matrix. Source integrals come from independent AO panels.
    """
    dense = _dense_case(b)
    o, v = dense["o"], dense["v"]
    f = b.basis.fock_copy()
    foo, fvv = f[:o, :o], f[o:, o:]
    g = dense["eri"][:o, o:, :o, o:].transpose(0, 2, 1, 3)
    data = _pair_data(result)
    sizes = [c.shape[1]**2 for _, _, c, _ in data]
    offsets = np.r_[0, np.cumsum(sizes)]
    def lift(vector):
        amplitudes = np.zeros((o, o, v, v))
        for k, (i, j, c, _) in enumerate(data):
            r = c.shape[1]
            t = c@vector[offsets[k]:offsets[k+1]].reshape(r, r)@c.T
            amplitudes[i, j] = t
            if i != j:
                amplitudes[j, i] = t.T
        return amplitudes
    def flatten(values):
        return np.concatenate([(c.T@values[i, j]@c).ravel() for i, j, c, _ in data])
    def linear(vector):
        t = lift(vector)
        return flatten(np.einsum("ac,ijcb->ijab", fvv, t)
            + np.einsum("ijac,cb->ijab", t, fvv)
            - np.einsum("ki,kjab->ijab", foo, t)
            - np.einsum("kj,ikab->ijab", foo, t))
    rhs = flatten(g)
    dimension = len(rhs)
    if dimension:
        operator = np.column_stack([linear(e) for e in np.eye(dimension)])
        expected = np.linalg.solve(operator, -rhs)
    else:
        expected = np.empty(0)
    actual = np.concatenate([result.solver.pair_copy(i, j).ravel() for i, j, _, _ in data])
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-9)
    np.testing.assert_allclose(linear(actual)+rhs, 0, atol=4e-12)
    common = lift(actual)
    energy = np.einsum("ijab,ijab->", 2*g-g.transpose(0, 1, 3, 2), common)
    assert result.solver.final_snapshot.correlation_energy == pytest.approx(energy, abs=4e-12)
    for i, j, c, actual_g in data:
        np.testing.assert_allclose(actual_g, c.T@g[i, j]@c, atol=3e-12, rtol=3e-10)
        np.testing.assert_allclose(result.solver.pair_copy(i, j), result.solver.pair_copy(j, i).T, atol=0, rtol=0)
    return dense, common, energy


@pytest.mark.parametrize("system,nk", [("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def test_actual_full_pair_spaces_match_independent_canonical_mp2(system, nk):
    b, provider = _physical(system, nk)
    events = []
    result = _run(b, provider, callback=events.append)
    assert result.converged and result.periodic_energy_per_cell
    assert result.diagnostics.complete_common_finite_torus_basis and result.diagnostics.all_pairs_full_rank
    dense, common, energy = _projected_oracle(b, result)
    o, v, g, f = dense["o"], dense["v"], dense["eri"], b.basis.fock_copy()
    eo, uo = np.linalg.eigh(f[:o, :o])
    ev, uv = np.linalg.eigh(f[o:, o:])
    gij = g[:o, o:, :o, o:].transpose(0, 2, 1, 3)
    canonical = np.einsum("ijab,ip,jq,ac,bd->pqcd", gij, uo, uo, uv, uv, optimize=True)
    t = -canonical/(ev[None, None, :, None]+ev[None, None, None, :]
        -eo[:, None, None, None]-eo[None, :, None, None])
    expected_energy = np.einsum("ijab,ijab->", 2*canonical-canonical.transpose(0, 1, 3, 2), t)
    assert energy == pytest.approx(expected_energy, abs=4e-12)
    restored = np.einsum("pqcd,ip,jq,ac,bd->ijab", t, uo, uo, uv, uv, optimize=True)
    np.testing.assert_allclose(common, restored, atol=3e-12, rtol=3e-9)
    assert result.correlation_energy_per_cell == pytest.approx(expected_energy/nk, abs=4e-12)
    assert result.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell+expected_energy/nk, abs=4e-12)
    assert result.matched_finite_gaussian_hf_recipe and not result.production_dlpno
    assert not result.infinite_source_accuracy_certified
    assert result.provider_identity_sha256 == provider.identity_sha256
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert len(events) == result.diagnostics.completed_progress_callbacks <= result.memory.progress_callback_upper_bound
    assert events[0].stage == core._PeriodicGaussianPairMP2Stage.BEGIN
    assert events[-1].stage == core._PeriodicGaussianPairMP2Stage.FINISHED
    if system == "he2":
        assert abs(f[0, 1]) > 1e-3
        assert result.solver.final_snapshot.iteration > 2
        assert result.solver.final_snapshot.transposed_sources > 0


def test_actual_positive_cutoff_ragged_projection_and_zero_rank_limit():
    b, provider = _physical("he2", 2)
    full = _run(b, provider)
    occupations = np.concatenate([full.pair(i, j).original_pno_occupations_copy()
        for i, j in combinations_with_replacement(range(4), 2)])
    cutoff = float(np.median(occupations))
    assert cutoff > 0
    result = _run(b, provider, _controls(cutoff=cutoff))
    assert result.converged and not result.diagnostics.all_pairs_full_rank
    assert result.diagnostics.minimum_pair_rank < 4
    _projected_oracle(b, result)
    assert result.identity_sha256 != full.identity_sha256
    zero = _run(b, provider, _controls(cutoff=2*float(occupations.max())))
    assert zero.converged and zero.diagnostics.zero_rank_pairs == 10
    assert zero.solver.final_snapshot.iteration == 2
    assert zero.solver.final_snapshot.correlation_energy == 0
    assert zero.diagnostics.completed_integral_calls == 10*4*4  # PNO generation only.
    assert zero.correlation_energy_per_cell == 0
    assert zero.solver.memory.total_amplitude_elements == 0


def test_exact_outer_inventory_and_inner_lifetimes():
    b, provider = _physical("he2", 2)
    controls = _controls()
    p = _plan(b, provider, controls)
    result = _run(b, provider, controls)
    n, pairs = p.common_virtual_dimension, p.pair_count
    retained = pairs*(16*n*n+16*n)
    assert p.retained_pair_output_upper_bytes == retained == result.diagnostics.retained_pair_bytes
    assert p.retained_pair_seal_bytes == pairs*15*65
    assert p.pair_generation_phase_upper_bytes == (pairs-1)*(16*n*n+16*n)+48*n*n+16*n
    assert p.solver_phase_owned_upper_bytes == retained+p.solver_upper.peak_owned_numerical_bytes
    assert p.peak_owned_numerical_bytes == max(p.pair_generation_phase_upper_bytes, p.solver_phase_owned_upper_bytes)
    known = p.borrowed_basis_bytes+p.borrowed_provider_row_bytes+p.control_storage_reservation_bytes
    assert p.per_worker_inventoried_bytes == p.peak_owned_numerical_bytes+known+2**20+65536
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    assert result.solver.memory.per_replica_inventoried_bytes <= p.solver_upper.per_replica_inventoried_bytes
    controls[1].other_live_bytes_per_worker += 777
    expanded = _plan(b, provider, controls)
    assert expanded.per_worker_inventoried_bytes-p.per_worker_inventoried_bytes == 777
    assert expanded.required_node_memory_bytes-p.required_node_memory_bytes == p.replicas_per_node*777


@pytest.mark.parametrize("cap,field", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"),
    ("maximum_pair_count", "pair_count"), ("maximum_integral_calls", "integral_calls_upper_bound"),
    ("maximum_progress_callbacks", "progress_callback_upper_bound"), ("maximum_work_units", "work_units_upper_bound"),
])
def test_one_below_outer_cap_rejects_before_progress(cap, field):
    b, provider = _physical()
    controls = _controls()
    value = getattr(_plan(b, provider, controls), field)
    setattr(controls[2], cap, value)
    assert _run(b, provider, controls).converged
    setattr(controls[2], cap, value-1)
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _run(b, provider, controls, events.append)
    assert not events


def test_partial_common_basis_and_unconverged_iteration_do_not_publish_periodic_energy():
    b, provider = _physical("he", 2, full=False)
    selected = _run(b, provider)
    assert selected.converged and not selected.diagnostics.complete_common_finite_torus_basis
    assert not selected.periodic_energy_per_cell
    with pytest.raises(RuntimeError, match="complete|per.cell"):
        _ = selected.total_energy_per_cell
    b, provider = _physical("he2", 1)
    incomplete = _run(b, provider, _controls(iterations=1))
    assert not incomplete.converged and not incomplete.periodic_energy_per_cell
    assert incomplete.solver.final_snapshot.iteration == 1
    assert incomplete.solver.final_snapshot.maximum_absolute_residual > 1e-12
    with pytest.raises(RuntimeError, match="converged|per.cell"):
        _ = incomplete.correlation_energy_per_cell


def test_foreign_source_controls_snapshot_callback_cancel_and_output_ownership():
    b, provider = _physical()
    other, foreign = _physical()
    with pytest.raises(ValueError, match="owners|receipts"):
        _plan(b, foreign)
    controls = _controls()
    baseline = _run(b, provider, controls)
    def change_controls(event):
        controls[0].maximum_occupied_virtual_fock_norm = 0
        controls[2].maximum_pair_count = 0
    frozen = _run(b, provider, controls, change_controls)
    assert frozen.identity_sha256 == baseline.identity_sha256
    def cancel(event):
        if event.stage == core._PeriodicGaussianPairMP2Stage.FINISHED:
            raise RuntimeError("intentional final pair cancellation")
    with pytest.raises(RuntimeError, match="intentional final"):
        _run(b, provider, callback=cancel)
    pair, solver = baseline.pair(0, 0), baseline.solver
    del b, provider, baseline
    gc.collect()
    original = pair.coefficients_copy()
    pair.coefficients_copy()[:] = 7
    np.testing.assert_array_equal(pair.coefficients_copy(), original)
    assert solver.final_snapshot.converged
    assert not hasattr(pair, "pnos")  # A consuming factory cannot steal this nested PNO owner.


@pytest.mark.parametrize("field,value", [("maximum_occupied_virtual_fock_norm", np.nan),
    ("maximum_occupied_virtual_fock_norm", -1.0)])
def test_missing_brillouin_budget_is_not_a_default(field, value):
    b, provider = _physical()
    controls = _controls()
    setattr(controls[0], field, value)
    with pytest.raises(ValueError, match="budgets"):
        _plan(b, provider, controls)
