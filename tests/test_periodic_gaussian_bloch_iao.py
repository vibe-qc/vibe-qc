"""Tiny actual Gaussian overlaps into IAOs, not chemical SCF benchmarks."""

from __future__ import annotations

import gc
import hashlib
import itertools
import math
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_cross_fourier_panel import _oracle as _gaussian, _call as _panel
from tests.test_periodic_correlation_bloch_iao import _ao_coefficients, _ao_oracle, _options as _iao_options
from tests.test_periodic_correlation_pair_topology import _make_budget, _make_dimensions


def _options(cutoff=3.85, **changes):
    value = core._PeriodicGaussianBlochIAOOptions()
    value.image_cutoff_bohr = cutoff
    for field in ("geometry", "overlap", "structural", "projection"):
        setattr(value, field + "_absolute_tolerance", 3e-11)
        setattr(value, field + "_relative_tolerance", 3e-11)
    for field, setting in changes.items():
        setattr(value, field, setting)
    return value


def _caps(**changes):
    value = core._PeriodicGaussianBlochIAOCaps()
    value.maximum_owned_numerical_bytes = 1 << 20
    value.maximum_work_units = 2_000_000_000_000
    value.maximum_atom_count = 8
    value.maximum_shell_count = 32
    value.maximum_contraction_count = 64
    value.maximum_primitive_numeric_lanes = 1024
    value.maximum_basis_content_wire_bytes = 65536
    value.maximum_borrowed_active_numeric_bytes = 1 << 20
    value.maximum_panel_pairs = 3
    value.maximum_total_image_candidates = 50000
    for field, setting in changes.items():
        setattr(value, field, setting)
    return value


def _basis(molecule, specs, *, name="explicit-gaussian-test", scale=1.0):
    shells = []
    for atom, angular, exponent in specs:
        origin = molecule.atoms[atom].xyz
        coefficient = scale * (2 * exponent / math.pi)**0.75
        shells.append(core.ShellInfo(atom, angular, True, [exponent], [coefficient], list(origin)))
    return core.BasisSet(molecule, shells, name, True)


def _case(mesh=(3, 1, 1), *, sp=False, molecular=False, reference_scale=1.0):
    positions = [(0.11, 0.03, -0.05), (0.85, 0.17, 0.08), (-0.6, 0.7, 0.2)]
    if sp:
        positions = positions[:2]
    molecule = core.Molecule([core.Atom(2, list(position)) for position in positions])
    specs = [(0, 0, 0.7), (1, 1, 0.9)] if sp else [(0, 0, 0.7), (1, 0, 0.9), (2, 0, 1.1)]
    ao = _basis(molecule, [*specs, (0, 0, 1.9)])
    minimal = _basis(molecule, specs, name="explicit-minimal-test", scale=1.25)
    system = core.PeriodicSystem()
    system.unit_cell = molecule.atoms
    system.lattice = (np.eye(3) * 8 if molecular else
                      np.array([[3.3, 0.21, -0.12], [0.0, 3.6, 0.17], [0.0, 0.0, 3.4]]))
    cutoff = 1.9 if molecular else 3.85
    reciprocal = 2 * np.pi * np.linalg.inv(system.lattice).T
    cells = np.array(list(itertools.product(*(range(n) for n in mesh))))
    b, p = ao.nbasis, 3
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = "1" * 64
    data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    data.normalization = (core._PeriodicMeanFieldNormalizationConvention
                          .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS)
    data.periodic_dimension = 3
    data.mesh = mesh
    data.is_shift = (0, 0, 0)
    data.reciprocal_lattice = reciprocal
    data.converged = True
    data.n_basis = data.n_effective_orbitals = b
    data.electrons_per_cell = 2 * p
    data.reference_energy_per_cell = -5.0
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    energies = np.r_[np.full(p, -1.5), np.full(b - p, 0.8)]
    occupations = np.r_[np.full(p, 2.0), np.zeros(b - p)]
    matrices, coefficients, crosses, minimals = [], [], [], []
    for cell in cells:
        k = reciprocal @ (cell / mesh)
        overlap = _gaussian(ao, ao, np.zeros((1, 3)), system.lattice, k, cutoff, 0 if molecular else 2)[0][:, :, 0]
        overlap = reference_scale * (overlap + overlap.conj().T) / 2
        assert np.linalg.eigvalsh(overlap).min() > 1e-5
        c = np.linalg.inv(np.linalg.cholesky(overlap).conj().T)
        f = overlap @ c @ np.diag(energies) @ c.conj().T @ overlap
        frozen = np.zeros(b, np.uint8); frozen[1] = 1
        active = np.zeros(b, np.uint8); active[[0, 2]] = 1
        virtual = np.zeros(b, np.uint8); virtual[p:] = 1
        data.add_kpoint(k, 1 / len(cells), overlap, f, c, energies, occupations,
                        frozen.tolist(), active.tolist(), virtual.tolist())
        matrices.append(overlap); coefficients.append(c)
        crosses.append(_gaussian(ao, minimal, np.zeros((1, 3)), system.lattice, k, cutoff, 0 if molecular else 2)[0][:, :, 0])
        minimals.append(_gaussian(minimal, minimal, np.zeros((1, 3)), system.lattice, k, cutoff, 0 if molecular else 2)[0][:, :, 0])
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = _make_dimensions(mesh, 2)
    dims.n_basis = dims.n_effective_orbitals = b
    dims.n_home_total_occupied = p
    dims.n_home_virtual = b - p
    dims.n_auxiliary = 2 * b
    dims.domain_ao_support_upper_bound = dims.domain_pao_upper_bound = dims.domain_pno_upper_bound = b
    dims.domain_local_auxiliary_upper_bound = 2 * b
    reference = core._make_periodic_correlation_admitted_reference(state, dims, _make_budget())
    return dict(reference=reference, ao=ao, minimal=minimal, system=system, mesh=mesh, cells=cells,
                s=np.array(matrices), c=np.array(coefficients), cross=np.array(crosses),
                small=np.array(minimals), cutoff=cutoff, molecule=molecule, specs=specs)


def _plan(case, point=0, *, caps=None, options=None, other=0):
    return core._plan_periodic_gaussian_bloch_iao(case["reference"], case["ao"], case["minimal"],
        case["system"], point, other, _options(case["cutoff"]) if options is None else options,
        _iao_options(), _caps() if caps is None else caps)


def _make(case, point=0, *, caps=None, options=None, other=0):
    return core._make_periodic_gaussian_bloch_iao(case["reference"], case["ao"], case["minimal"],
        case["system"], point, other, _options(case["cutoff"]) if options is None else options,
        _iao_options(), _caps() if caps is None else caps)


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("mesh,point", [((1, 1, 1), 0), ((3, 1, 1), 1)])
def test_actual_gaussian_s_sp_iao_matches_independent_original_ao_equations(sp, mesh, point):
    case = _case(mesh, sp=sp)
    result = _make(case, point)
    a, g, b, d = _ao_oracle(case["c"][point], case["s"][point],
                           case["cross"][point], case["small"][point], [0, 1, 2])
    np.testing.assert_allclose(_ao_coefficients(result.iao), a, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(result.iao.metric_copy(), g, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(result.iao.occupied_coefficients_copy(), b, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(result.iao.occupied_covariant_copy(), d, atol=3e-10, rtol=3e-10)
    np.testing.assert_allclose(a @ b, case["c"][point, :, :3], atol=3e-10)
    assert result.iao.state.n_frozen_core == 1
    labels = [shell.atom_index for shell in case["minimal"].shells() for _ in range(2 * shell.l + 1)]
    assert [result.iao.atom_label(rho) for rho in range(case["minimal"].nbasis)] == labels
    assert result.diagnostics.maximum_s11_reference_residual < 4e-12
    assert result.diagnostics.maximum_cross_adjoint_residual < 4e-12
    assert not result.hf_basis_source_authenticated
    assert not result.infinite_image_tail_certified
    assert result.iao.declared_provenance.cross_overlap_identity == result.source_identity_sha256


@pytest.mark.parametrize("sp", [False, True])
def test_molecular_limit_matches_independent_libint_two_basis_overlap(sp):
    case = _case((1, 1, 1), sp=sp, molecular=True)
    cross = core.compute_overlap_two_basis(case["ao"], case["minimal"])
    minimal = core.compute_overlap_two_basis(case["minimal"], case["minimal"])
    np.testing.assert_allclose(cross, case["cross"][0], atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(minimal, case["small"][0], atol=3e-13, rtol=3e-13)
    result = _make(case)
    expected = _ao_oracle(case["c"][0], case["s"][0], cross, minimal, [0, 1, 2])
    np.testing.assert_allclose(_ao_coefficients(result.iao), expected[0], atol=2e-10)
    b, r = case["ao"].nbasis, case["minimal"].nbasis
    assert result.memory.retained_pair_image_count == b**2 + 2 * b * r + r**2
    assert result.diagnostics.maximum_trim_imaginary_magnitude == 0


def test_skew_cell_nonself_and_all_eight_trim_are_audited_without_complex_casts():
    case = _case((3, 1, 1))
    left, right = _make(case, 1), _make(case, 2)
    np.testing.assert_allclose(_ao_coefficients(left.iao), _ao_coefficients(right.iao).conj(), atol=3e-10)
    assert np.max(np.abs(case["cross"][1].imag)) > 1e-3
    assert left.diagnostics.maximum_overlap_time_reversal_residual < 3e-12
    even = _case((2, 2, 2))
    for k in range(8):
        value = _make(even, k)
        assert value.memory.conjugate_point == k
        assert value.diagnostics.maximum_trim_imaginary_magnitude < 3e-12
        assert value.diagnostics.maximum_projection_correction < 3e-12
        np.testing.assert_allclose(_ao_coefficients(value.iao).imag, 0.0, atol=3e-11)


def test_partial_pair_panel_choices_preserve_bitwise_sources_and_native_iao():
    case = _case()
    values = [_make(case, 1, caps=_caps(maximum_panel_pairs=p)) for p in (1, 3, 16)]
    for value in values[1:]:
        assert value.source_identity_sha256 == values[0].source_identity_sha256
        assert value.raw_overlap_payload_sha256 == values[0].raw_overlap_payload_sha256
        assert value.iao.iao_identity_sha256 == values[0].iao.iao_identity_sha256
        np.testing.assert_array_equal(value.iao.metric_copy(), values[0].iao.metric_copy())
    assert values[0].memory.panel_calls > values[-1].memory.panel_calls


@pytest.mark.parametrize("ranks,workers", [(1, 1), (1, 2)])
def test_exact_simultaneous_memory_borrowed_maps_and_previous_point_inventory(ranks, workers):
    case = _case((1, 1, 1))
    ref = case["reference"]
    dims, budget = ref.dimensions, ref.budget
    dims.external_bytes -= ref.state.resident_bytes
    budget.mpi_ranks, budget.workers_per_rank = ranks, workers
    case["reference"] = core._make_periodic_correlation_admitted_reference(ref.state, dims, budget)
    other = 1234
    plan = _plan(case, other=other)
    b, r = plan.n_basis, plan.n_minimal
    assert plan.overlap_and_label_bytes == 16 * (b * r + r**2) + 8 * r
    fourier_scratch = core._PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES
    assert plan.fixed_fourier_workspace_bytes == fourier_scratch
    assert plan.panel_output_bytes == 16 * plan.panel_pairs
    assert plan.peak_owned_numerical_bytes == plan.overlap_and_label_bytes + max(
        fourier_scratch + plan.panel_output_bytes, plan.iao_owned_peak_bytes)
    assert plan.borrowed_basis_numeric_bytes == (plan.ao.borrowed_active_numeric_bytes
        + plan.minimal.borrowed_active_numeric_bytes + 4 * (plan.ao.shell_count + plan.minimal.shell_count))
    assert plan.borrowed_geometry_numeric_bytes == 72 + 28 * len(case["system"].unit_cell)
    assert plan.total_borrowed_numerical_bytes == plan.borrowed_basis_numeric_bytes + plan.borrowed_geometry_numeric_bytes + other
    dims = case["reference"].dimensions
    assert plan.required_node_memory_bytes == (dims.external_bytes + dims.shared_bytes
        + ranks * (dims.per_rank_bytes + dims.localization_window_bytes_per_rank)
        + ranks * workers * (plan.peak_owned_numerical_bytes + plan.total_borrowed_numerical_bytes))
    result = _make(case, other=other)
    assert result.memory.output_numerical_bytes == result.iao.memory.output_numerical_bytes
    assert result.iao.declared_provenance.caller_live_numerical_bytes == plan.overlap_and_label_bytes + plan.total_borrowed_numerical_bytes
    assert result.diagnostics.charged_work_units == plan.maximum_work_units
    assert result.diagnostics.evaluated_panel_count == plan.panel_calls
    assert result.diagnostics.evaluated_image_candidate_count == plan.image_candidate_count


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_image_candidates", "maximum_work_units"])
def test_exact_required_caps_and_cap_minus_one_fail_before_value_processing(field):
    case = _case((1, 1, 1))
    plan = _plan(case)
    needed = {"maximum_owned_numerical_bytes": plan.peak_owned_numerical_bytes,
              "maximum_total_image_candidates": plan.image_candidate_count,
              "maximum_work_units": plan.maximum_work_units}[field]
    result = _make(case, caps=_caps(**{field: needed}))
    assert result.iao.point == 0
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _make(case, caps=_caps(**{field: needed - 1}))


def test_byte_and_work_gates_precede_invalid_s11_values():
    case = _case((1, 1, 1), reference_scale=1.01)
    plan = _plan(case)
    with pytest.raises((ValueError, RuntimeError), match="byte cap"):
        _make(case, caps=_caps(maximum_owned_numerical_bytes=plan.peak_owned_numerical_bytes - 1))
    with pytest.raises((ValueError, RuntimeError), match="preflight work"):
        _make(case, caps=_caps(maximum_work_units=plan.preflight_work_units - 1))
    with pytest.raises(ValueError, match="S11.*admitted overlap"):
        _make(case)


@pytest.mark.parametrize("field", ["maximum_atom_count", "maximum_shell_count", "maximum_contraction_count",
    "maximum_primitive_numeric_lanes", "maximum_basis_content_wire_bytes", "maximum_borrowed_active_numeric_bytes"])
def test_bounded_source_scans_reject_before_image_evaluation(field):
    case = _case((1, 1, 1))
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _plan(case, caps=_caps(**{field: 1}))


def test_actual_basis_content_not_name_controls_source_identity_and_s11_match():
    case = _case((1, 1, 1))
    base = _make(case)
    case["ao"] = _basis(case["molecule"], [*case["specs"], (0, 0, 1.9)], name="renamed-only")
    renamed = _make(case)
    assert renamed.ao_basis_identity_sha256 == base.ao_basis_identity_sha256
    assert renamed.source_identity_sha256 == base.source_identity_sha256
    case["ao"] = _basis(case["molecule"], [*case["specs"], (0, 0, 1.9)], scale=1.01)
    with pytest.raises(ValueError, match="S11.*admitted overlap"):
        _make(case)


def test_changed_explicit_minimal_basis_changes_actual_receipt_without_changing_occupied_span():
    case = _case((1, 1, 1))
    first = _make(case)
    case["minimal"] = _basis(case["molecule"], case["specs"], scale=1.4)
    second = _make(case)
    assert first.minimal_basis_identity_sha256 != second.minimal_basis_identity_sha256
    assert first.source_identity_sha256 != second.source_identity_sha256
    for value in (first, second):
        np.testing.assert_allclose(_ao_coefficients(value.iao) @ value.iao.occupied_coefficients_copy(),
                                   case["c"][0, :, :3], atol=4e-11)


@pytest.mark.parametrize("change", ["cell", "atoms", "cutoff"])
def test_geometry_atom_assignment_or_finite_image_reference_mismatch_fails_closed(change):
    case = _case((1, 1, 1))
    options = _options(case["cutoff"])
    if change == "cell":
        case["system"].lattice = case["system"].lattice * 1.01
        message = "direct cell"
    elif change == "atoms":
        case["system"].unit_cell = case["system"].unit_cell[::-1]
        message = "shell origin"
    else:
        options.image_cutoff_bohr = 1.9
        message = "S11.*admitted overlap"
    with pytest.raises(ValueError, match=message):
        _make(case, options=options)


def test_raw_overlap_digest_wire_is_independent_and_unprojected():
    case = _case((1, 1, 1))
    result = _make(case)
    b, r = case["ao"].nbasis, case["minimal"].nbasis
    domain = b"vibeqc.periodic.gaussian-bloch-iao.raw-overlaps"
    wire = struct.pack(">Q", len(domain)) + domain + struct.pack(">IQQQQI", 1, 0, 0, b, r, 4)
    for role, (left, right) in enumerate((("ao", "ao"), ("ao", "minimal"), ("minimal", "minimal"), ("minimal", "ao"))):
        panel = _panel(case[left], case[right], [[0, 0, 0]], system=case["system"], cutoff=case["cutoff"])
        values = panel["values"].reshape(-1).view(float).copy()
        values[values == 0] = 0.0
        wire += struct.pack(">QQ", role, case[left].nbasis * case[right].nbasis)
        wire += b"".join(struct.pack(">d", x) for x in values)
    assert result.raw_overlap_payload_sha256 == hashlib.sha256(wire).hexdigest()


def test_iao_reference_internal_lifetime_survives_wrapper_deletion_and_optimizer_owner_pinning():
    from tests.test_periodic_correlation_diabatic_seed import _make as seed_factory
    from tests.test_periodic_correlation_iao_optimizer import _options as opt_options, _caps as opt_caps
    from tests.test_periodic_correlation_iao_pm import _options as pm_options
    case = _case((1, 1, 1))
    seed = seed_factory(case["reference"])
    wrapper = _make(case, other=seed.memory.output_numerical_bytes)
    iao = wrapper.iao
    expected = iao.metric_copy()
    del wrapper
    gc.collect()
    np.testing.assert_array_equal(iao.metric_copy(), expected)
    options = opt_options(maximum_iterations=1)
    plan = core._plan_periodic_correlation_iao_optimizer(case["reference"], seed, case["minimal"].nbasis, options)
    points = [iao]
    del iao
    result = core._optimize_periodic_correlation_iao_pm(case["reference"], seed, points,
        pm_options(), options, opt_caps(plan), lambda event: points.clear())
    assert np.isfinite(result.diagnostics.final_objective)
    assert result.state.n_frozen_core == 1


@pytest.mark.parametrize("changes", [{"image_cutoff_bohr": np.nan}, {"image_cutoff_bohr": -1.0},
    {"structural_absolute_tolerance": 0.0, "structural_relative_tolerance": 0.0},
    {"geometry_relative_tolerance": np.inf}, {"projection_absolute_tolerance": -1.0}])
def test_explicit_finite_controls_are_required(changes):
    case = _case((1, 1, 1))
    with pytest.raises((ValueError, RuntimeError)):
        _plan(case, options=_options(**changes))
