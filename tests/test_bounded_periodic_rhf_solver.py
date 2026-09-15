"""Bounded multi-k RHF algebra, using tiny variational linear-response models."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _options(**changes):
    out = core._BoundedPeriodicRHFOptions()
    out.maximum_iterations = 64
    out.maximum_diis_history = 0
    out.jacobi_max_sweeps = 48
    out.jacobi_relative_tolerance = 2e-15
    out.overlap_rank_absolute_floor = 1e-12
    out.overlap_rank_relative_floor = 1e-12
    out.overlap_negative_absolute_tolerance = 1e-12
    out.overlap_negative_relative_tolerance = 1e-12
    out.hermitian_absolute_tolerance = 2e-12
    out.hermitian_relative_tolerance = 2e-12
    out.maximum_overlap_projection_error = 1e-11
    out.maximum_hcore_projection_error = 1e-11
    out.maximum_fock_projection_error = 1e-11
    out.algebra_absolute_tolerance = 2e-11
    out.algebra_relative_tolerance = 2e-11
    out.eigen_relative_tolerance = 2e-11
    out.commutator_tolerance = 1e-9
    out.density_closure_absolute_tolerance = 1e-10
    out.density_closure_relative_tolerance = 1e-10
    out.energy_change_tolerance = 1e-12
    out.minimum_band_gap_hartree = 0.05
    for field, value in changes.items():
        setattr(out, field, value)
    return out


def _case(k=3, n=3, *, interaction=0.025, complex_values=True):
    """E2=lambda/2*[mean_k Tr(A_k D_k)]^2; G_k=lambda A_k*mean Tr(AD).

    This gives a self-adjoint all-k coupled variational response with a small
    contraction constant, not an actual Gaussian two-electron source.
    """
    rng = np.random.default_rng(19230 + k + n)
    overlaps, hcores, operators = [], [], []
    for point in range(k):
        x = rng.normal(size=(n, n)) * 0.08
        if complex_values:
            x = x + 0.05j * rng.normal(size=(n, n))
        factor = np.eye(n) + x
        s = factor.conj().T @ factor
        raw = rng.normal(size=(n, n))
        if complex_values:
            raw = raw + 1j * rng.normal(size=(n, n))
        rotation = np.linalg.qr(raw)[0]
        c = np.linalg.solve(factor, rotation)
        e = np.linspace(-1.4, 1.7, n) + 0.05 * point
        h = s @ c @ np.diag(e) @ c.conj().T @ s
        raw = rng.normal(size=(n, n))
        if complex_values:
            raw = raw + 1j * rng.normal(size=(n, n))
        operator = (raw + raw.conj().T) / 2
        overlaps.append((s + s.conj().T) / 2)
        hcores.append((h + h.conj().T) / 2)
        operators.append(operator)
    s = np.ascontiguousarray(overlaps, dtype=complex)
    h = np.ascontiguousarray(hcores, dtype=complex)
    a = np.array(operators)
    kernel = interaction / k * np.outer(a.ravel(), a.swapaxes(1, 2).ravel())
    return dict(s=s, h=h, kernel=np.ascontiguousarray(kernel, dtype=complex), bias=np.zeros_like(h),
                electrons=2, nuclear=0.731, operator=a, interaction=interaction)


def _plan(case, options=None, *, rank=None, other=0):
    k, n, _ = case["s"].shape
    return core._plan_bounded_periodic_rhf_linear_model(k, n, case["electrons"],
        n if rank is None else rank, _options() if options is None else options, other)


def _caps(case, options=None, **changes):
    p = _plan(case, options)
    out = core._BoundedPeriodicRHFCaps()
    out.maximum_owned_numerical_bytes = p.peak_owned_numerical_bytes
    out.maximum_total_numerical_bytes = p.total_numerical_bytes
    out.maximum_work_units = p.maximum_work_units
    out.maximum_fock_calls = 64
    out.other_live_numerical_bytes = 0
    for key, value in changes.items():
        setattr(out, key, value)
    return out


def _run(case, *, options=None, caps=None, callback=None, **extra):
    options = _options() if options is None else options
    return core._solve_bounded_periodic_rhf_linear_model(
        case["s"], case["h"], case["kernel"], case["bias"], case["electrons"], case["nuclear"],
        options, _caps(case, options) if caps is None else caps, callback, **extra)


def _herm(a):
    return (a + a.conj().swapaxes(-1, -2)) / 2


def _orthogonalizers(case, options):
    values, vectors = np.linalg.eigh(_herm(case["s"]))
    xs = []
    for d, u in zip(values, vectors):
        kept = d > max(options.overlap_rank_absolute_floor,
                       options.overlap_rank_relative_floor * d[-1])
        xs.append(u[:, kept] / np.sqrt(d[kept]))
    return np.array(xs)


def _diagonalize(f, x):
    e, u = np.linalg.eigh(_herm(x.conj().swapaxes(1, 2) @ f @ x))
    return e, x @ u


def _density(c, occupied=1):
    return 2 * c[:, :, :occupied] @ c[:, :, :occupied].conj().swapaxes(1, 2)


def _fock(case, density):
    return _herm(case["h"] + case["bias"] + (case["kernel"] @ density.ravel()).reshape(density.shape))


def _energy(case, density, fock):
    return case["nuclear"] + np.einsum("kij,kji->", density, _herm(case["h"]) + fock).real / (2 * len(density))


@pytest.mark.parametrize("k,n,complex_values", [(1, 2, False), (3, 3, True), (8, 3, True), (2, 4, False)])
def test_converged_full_k_physical_density_fock_energy_and_projected_residual_match_numpy(k, n, complex_values):
    case = _case(k, n, complex_values=complex_values)
    options = _options()
    result = _run(case, options=options)
    assert result.converged
    final = result.final_snapshot
    density, fock = result.density_copy(), result.fock_copy()
    np.testing.assert_array_equal(density, density.conj().swapaxes(1, 2))
    np.testing.assert_array_equal(np.diagonal(density, axis1=1, axis2=2).imag, 0)
    c, eps = result.coefficients_copy(), result.orbital_energies_copy()
    x = _orthogonalizers(case, options)
    np.testing.assert_allclose(fock, _fock(case, density), atol=3e-12)
    assert final.energy_per_cell == pytest.approx(_energy(case, density, fock), abs=3e-12)
    np.testing.assert_allclose(c.conj().swapaxes(1, 2) @ case["s"] @ c, np.tile(np.eye(n), (k, 1, 1)), atol=3e-12)
    np.testing.assert_allclose(fock @ c, case["s"] @ (c * eps[:, None, :]), atol=3e-12)
    r = x.conj().swapaxes(1, 2) @ (fock @ density @ case["s"] - case["s"] @ density @ fock) @ x
    assert final.commutator_frobenius_rms == pytest.approx(np.linalg.norm(r) / np.sqrt(k), abs=3e-12)
    assert final.maximum_density_closure_frobenius == pytest.approx(
        max(np.linalg.norm(d) for d in density - _density(c)), abs=3e-12)
    # Separate high-precision fixed-point reference; compare densities, not
    # arbitrary orbital signs/phases in independently diagonalized frames.
    _, initial = _diagonalize(_herm(case["h"]), x)
    reference = _density(initial)
    for _ in range(100):
        _, orbitals = _diagonalize(_fock(case, reference), x)
        updated = _density(orbitals)
        if np.linalg.norm(updated - reference) < 1e-14:
            reference = updated
            break
        reference = updated
    np.testing.assert_allclose(density, reference, atol=3e-9, rtol=3e-9)
    assert final.global_band_gap == pytest.approx(eps[:, 1].min() - eps[:, 0].max(), abs=3e-12)


def test_one_iteration_returns_evaluated_hcore_density_not_unevaluated_final_c_projector():
    case = _case()
    options = _options(maximum_iterations=1)
    result = _run(case, options=options)
    x = _orthogonalizers(case, options)
    _, initial_c = _diagonalize(_herm(case["h"]), x)
    initial_d = _density(initial_c)
    np.testing.assert_allclose(result.density_copy(), initial_d, atol=3e-12)
    assert result.final_snapshot.maximum_density_closure_frobenius > 1e-3
    assert not result.converged
    assert result.final_snapshot.status == core._BoundedPeriodicRHFStatus.ITERATION_LIMIT
    assert result.final_snapshot.energy_per_cell == pytest.approx(_energy(case, initial_d, _fock(case, initial_d)), abs=3e-12)
    assert not result.final_snapshot.has_energy_change


@pytest.mark.parametrize("kind", ["work", "calls", "cancel"])
def test_limits_return_last_evaluated_snapshot_without_a_hidden_density_update(kind):
    case = _case()
    one = _run(case, options=_options(maximum_iterations=1))
    caps = _caps(case)
    callback = None
    if kind == "work":
        caps.maximum_work_units = one.final_snapshot.charged_work_units
        status = core._BoundedPeriodicRHFStatus.WORK_LIMIT
    elif kind == "calls":
        caps.maximum_fock_calls = 1
        status = core._BoundedPeriodicRHFStatus.FOCK_CALL_LIMIT
    else:
        callback = lambda event: False
        status = core._BoundedPeriodicRHFStatus.CANCELLED
    result = _run(case, caps=caps, callback=callback)
    assert result.final_snapshot.status == status
    assert not result.converged
    for name in ("density_copy", "fock_copy", "coefficients_copy", "orbital_energies_copy"):
        np.testing.assert_array_equal(getattr(result, name)(), getattr(one, name)())
    assert result.final_snapshot.energy_per_cell == one.final_snapshot.energy_per_cell


def test_two_physical_evaluations_required_even_for_zero_interaction_exact_fixed_point():
    case = _case(interaction=0)
    one = _run(case, options=_options(maximum_iterations=1))
    assert not one.converged
    converged = _run(case)
    assert converged.converged
    assert converged.final_snapshot.evaluated_iteration == 2
    assert converged.final_snapshot.fock_calls == 2


def test_non_insulating_stationary_closed_shell_result_is_not_converged():
    case = _case(2, 2, interaction=0, complex_values=False)
    case["s"][:] = np.eye(2)
    case["h"][0] = np.diag([-1.0, -0.2])
    case["h"][1] = np.diag([0.1, 1.0])
    result = _run(case)
    assert result.final_snapshot.status == core._BoundedPeriodicRHFStatus.NON_INSULATING
    assert not result.converged
    assert result.final_snapshot.global_band_gap == pytest.approx(-0.3)


def test_rank_truncation_retains_projected_variational_solution_but_reports_full_ao_failure():
    case = _case(2, 3, interaction=0)
    case["s"][:] = np.diag([1.0, 1.0, 1e-9])
    case["h"][:] = [[-1.0, 0.2, 0.1], [0.2, 0.8, -0.15], [0.1, -0.15, 0.3]]
    options = _options(overlap_rank_absolute_floor=1e-8)
    result = _run(case, options=options)
    assert result.converged and result.memory.retained_rank == 2
    assert result.final_snapshot.maximum_full_ao_eigen_residual > 0.05
    assert result.final_snapshot.maximum_projected_eigen_relative_residual < 1e-12
    np.testing.assert_array_equal(result.coefficients_copy()[:, 2], 0)
    assert result.memory.peak_owned_numerical_bytes < _plan(case, options).peak_owned_numerical_bytes


@pytest.mark.parametrize("kind", ["different", "no_virtual", "negative"])
def test_overlap_rank_and_negative_failures_precede_first_provider_call(kind):
    case = _case(2, 3)
    case["s"][:] = np.eye(3)
    if kind == "different":
        case["s"][0, 2, 2] = 1e-14
        expected = "rank differs"
    elif kind == "no_virtual":
        case["s"][:, 1:, 1:] = 0
        expected = "occupied-plus-virtual"
    else:
        case["s"][0, 2, 2] = -0.1
        expected = "negative eigenvalue"
    with pytest.raises((ValueError, RuntimeError), match=expected):
        _run(case, fail_on_call=1)


@pytest.mark.parametrize("role", ["s", "h", "bias"])
def test_raw_hermitian_defects_are_audited_before_explicit_bounded_projection(role):
    case = _case(1, 3)
    case[role][0, 0, 1] += 2e-14j
    result = _run(case)
    d = result.diagnostics
    correction = {"s": d.maximum_overlap_projection_frobenius,
                  "h": d.maximum_hcore_projection_frobenius,
                  "bias": d.maximum_fock_projection_frobenius}[role]
    assert correction > 0
    options = _options()
    field = {"s": "maximum_overlap_projection_error", "h": "maximum_hcore_projection_error",
             "bias": "maximum_fock_projection_error"}[role]
    setattr(options, field, 0)
    with pytest.raises(ValueError, match="projection correction"):
        _run(case, options=options)
    case[role][0, 0, 1] += 1e-3j
    with pytest.raises(ValueError, match="Hermitian defect"):
        _run(case)


def test_exact_phase_and_borrowed_provider_payload_inventory_and_cap_boundaries():
    case = _case()
    result = _run(case)
    m = result.memory
    k, n, r = m.n_kpoints, m.n_basis, m.retained_rank
    assert m.output_numerical_bytes == 32 * k * n**2 + 16 * k * n * r + 8 * k * r
    assert m.peak_owned_numerical_bytes == 32 * k * n**2 + 32 * k * n * r + 8 * k * r + 32 * n**2 + 32 * n * r + 8 * n
    assert m.borrowed_input_bytes == 32 * k * n**2
    assert m.provider_retained_bytes == case["kernel"].nbytes + case["bias"].nbytes
    assert m.total_numerical_bytes == m.peak_owned_numerical_bytes + m.borrowed_input_bytes + m.provider_retained_bytes
    for field, needed in (("maximum_owned_numerical_bytes", m.peak_owned_numerical_bytes),
                          ("maximum_total_numerical_bytes", m.total_numerical_bytes)):
        assert _run(case, caps=_caps(case, **{field: needed})).converged
        with pytest.raises((ValueError, RuntimeError), match="byte cap"):
            _run(case, caps=_caps(case, **{field: needed - 1}))
    other = 713
    caps = _caps(case, other_live_numerical_bytes=other,
                 maximum_total_numerical_bytes=m.total_numerical_bytes + other)
    assert _run(case, caps=caps).memory.total_numerical_bytes == m.total_numerical_bytes + other


def test_work_before_initial_complete_evaluation_and_diis_rejection_before_bad_values():
    case = _case()
    one = _run(case, options=_options(maximum_iterations=1))
    with pytest.raises((ValueError, RuntimeError), match="initial complete evaluation"):
        _run(case, caps=_caps(case, maximum_work_units=one.final_snapshot.charged_work_units - 1))
    case["s"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="maximum_diis_history=0"):
        _run(case, options=_options(maximum_diis_history=6))


@pytest.mark.parametrize("kind", ["unwritten", "provider_failure", "progress_failure"])
def test_callback_failure_or_unwritten_output_aborts_instead_of_returning_old_energy(kind):
    case = _case()
    if kind == "unwritten":
        with pytest.raises((ValueError, RuntimeError, OverflowError), match="nonfinite"):
            _run(case, omit_last_output=True)
    elif kind == "provider_failure":
        with pytest.raises(RuntimeError, match="provider failure"):
            _run(case, fail_on_call=2)
    else:
        def callback(event):
            raise RuntimeError("cancel by exception")
        with pytest.raises(RuntimeError, match="cancel by exception"):
            _run(case, callback=callback)


def test_progress_snapshots_are_copied_scalars_and_controls_cannot_change_during_callback():
    case = _case()
    options = _options()
    events = []

    def callback(event):
        events.append(event)
        options.maximum_iterations = 1

    result = _run(case, options=options, callback=callback)
    assert result.converged
    assert len(events) == result.final_snapshot.evaluated_iteration
    assert [e.evaluated_iteration for e in events] == list(range(1, len(events) + 1))
    assert events[-1].converged
    assert events[-1].energy_per_cell == result.final_snapshot.energy_per_cell
    assert events[0].maximum_density_closure_frobenius > events[-1].maximum_density_closure_frobenius


@pytest.mark.parametrize("role", ["s", "h", "kernel", "bias"])
def test_tiny_binding_requires_exact_complex_contiguous_arrays(role):
    case = _case()
    case[role] = case[role].real.copy()
    with pytest.raises(ValueError, match="complex128"):
        _run(case)


@pytest.mark.parametrize("field", ["maximum_iterations", "jacobi_max_sweeps", "commutator_tolerance",
    "energy_change_tolerance", "eigen_relative_tolerance"])
def test_unset_controls_fail_before_numerical_work(field):
    with pytest.raises(ValueError):
        _run(_case(), options=_options(**{field: 0}))
