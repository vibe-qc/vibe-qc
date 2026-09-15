"""Density-independent finite Gaussian source authentication; no SCF jobs."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _system
from tests.test_periodic_auxiliary_fourier import _identity_fixture


_OPTION_FIELDS = (
    "reciprocal_energy_cutoff", "ao_pair_image_cutoff_bohr",
    "metric_absolute_eigenvalue_threshold", "metric_negative_tolerance",
)
_CAP_FIELDS = (
    "maximum_context_storage_bytes", "maximum_kpoint_count",
    "maximum_shell_count", "maximum_contraction_count",
    "maximum_primitive_numeric_lanes", "maximum_basis_content_wire_bytes",
    "maximum_borrowed_active_numeric_bytes", "maximum_work_units",
)


def _options():
    result = core._PeriodicGaussianSourceOptions()
    for field, value in zip(_OPTION_FIELDS, (2.0, 6.0, 1e-9, 1e-12)):
        setattr(result, field, value)
    return result


def _caps():
    result = core._PeriodicGaussianSourceCaps()
    for field in _CAP_FIELDS:
        setattr(result, field, 10**8)
    return result


def _auxiliary():
    return _basis([(0, (0.2, -0.1, 0.4), [0.6, 1.3], [0.4, 0.7], True)])


def _make(*, system=None, ao=None, auxiliary=None, mesh=None, options=None, caps=None):
    return core._make_periodic_gaussian_source_context(
        _system() if system is None else system,
        _identity_fixture() if ao is None else ao,
        _auxiliary() if auxiliary is None else auxiliary,
        core._RegularKMesh([2, 3, 1]) if mesh is None else mesh,
        _options() if options is None else options,
        _caps() if caps is None else caps,
    )


def _wire_identity(context):
    chunks = []

    def text(value):
        value = value.encode("utf-8")
        chunks.append(struct.pack(">Q", len(value)))
        chunks.append(value)

    def u32(value):
        chunks.append(struct.pack(">I", value))

    text("vibeqc.periodic.gaussian-source-context")
    u32(1)
    u32(3)
    for value in (
        core._PERIODIC_GAUSSIAN_SOURCE_GEOMETRY_POLICY,
        core._PERIODIC_GAUSSIAN_SOURCE_HAMILTONIAN_POLICY,
        core._PERIODIC_CORRELATION_THREE_CENTER_IMAGE_POLICY,
        core._PERIODIC_GAUSSIAN_SOURCE_RECIPROCAL_POLICY,
    ):
        text(value)
    u32(1)  # reciprocal source contract
    u32(1)  # basis content digest
    for field in (
        "ao_basis_identity_sha256", "auxiliary_basis_identity_sha256",
        "reciprocal_producer_identity_sha256", "factorization_backend_identity_sha256",
    ):
        text(getattr(context, field))
    for value in (*context.mesh, *context.is_shift):
        u32(value)
    for value in (*context.direct_lattice.ravel(), *context.reciprocal_lattice.ravel(),
                  *(getattr(context.options, name) for name in _OPTION_FIELDS)):
        chunks.append(struct.pack(">d", 0.0 if value == 0.0 else value))
    for basis in (context.inventory.ao, context.inventory.auxiliary):
        chunks.append(struct.pack(">Q", basis.function_count))
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def test_scfindependent_native_identity_matches_independent_canonical_wire():
    context = _make()
    assert context.contract_version == 1
    assert context.density_independent
    assert not context.enumerated_source_certified
    assert not context.ao_image_source_certified
    assert not context.nuclear_hcore_certified
    assert context.source_context_identity_sha256 == _wire_identity(context)
    assert context.ao_basis_identity_sha256 == core._auxiliary_basis_content_identity_sha256(_identity_fixture())
    assert context.auxiliary_basis_identity_sha256 == core._auxiliary_basis_content_identity_sha256(_auxiliary())
    assert context.factorization_backend_identity_sha256 == core._periodic_correlation_metric_factorization_backend_identity_sha256()
    assert "exxdiv=None" in core._PERIODIC_GAUSSIAN_SOURCE_HAMILTONIAN_POLICY
    assert "G0-omit" in core._PERIODIC_GAUSSIAN_SOURCE_HAMILTONIAN_POLICY


def test_basis_names_and_signed_zero_are_not_scientific_input_changes():
    first = _make(ao=_identity_fixture(name="one", signed_zero=-0.0))
    second = _make(ao=_identity_fixture(name="two", signed_zero=0.0))
    assert first.source_context_identity_sha256 == second.source_context_identity_sha256
    second.verify_bases(_identity_fixture(name="third"), _auxiliary(), _caps())


def test_matrix_signed_zero_is_canonicalized_and_identity_is_stable():
    a = np.diag([5.0, 6.0, 7.0])
    b = a.copy()
    b[b == 0] = -0.0
    first, second = _make(system=_system(a)), _make(system=_system(b))
    assert first.source_context_identity_sha256 == second.source_context_identity_sha256
    assert not np.any(np.signbit(second.direct_lattice[second.direct_lattice == 0]))
    assert not np.any(np.signbit(second.reciprocal_lattice[second.reciprocal_lattice == 0]))


@pytest.mark.parametrize("kwargs", [
    {"first_exponent": 1.3}, {"first_coefficient": 0.8}, {"first_pure": True},
    {"second_origin": (1.3, -0.5, 0.75)}, {"reverse_shells": True},
])
def test_actual_ao_content_changes_are_sealed_and_borrowed_mismatch_rejects(kwargs):
    original = _make()
    changed = _identity_fixture(**kwargs)
    assert original.source_context_identity_sha256 != _make(ao=changed).source_context_identity_sha256
    with pytest.raises(ValueError, match="content mismatch"):
        original.verify_bases(changed, _auxiliary(), _caps())


def test_actual_auxiliary_change_and_basis_role_swap_are_sealed():
    original = _make()
    changed = _basis([(0, (0.2, -0.1, 0.4), [0.6, 1.3], [0.4, 0.8], True)])
    assert original.source_context_identity_sha256 != _make(auxiliary=changed).source_context_identity_sha256
    assert original.source_context_identity_sha256 != _make(ao=_auxiliary(), auxiliary=_identity_fixture()).source_context_identity_sha256
    with pytest.raises(ValueError, match="content mismatch"):
        original.verify_bases(_identity_fixture(), changed, _caps())


def test_nuclear_electron_and_symmetry_fields_are_not_a_false_hcore_certificate():
    system = _system()
    first = _make(system=system)
    system.unit_cell = [core.Atom(8, [5.0, 4.0, 3.0])]
    system.charge = 6
    system.multiplicity = 9
    second = _make(system=system)
    assert first.source_context_identity_sha256 == second.source_context_identity_sha256
    assert not second.nuclear_hcore_certified


def test_skew_direct_columns_and_all_gamma_nonself_nyquist_records():
    a = np.array([[5.0, 0.7, -0.3], [0.2, 6.0, 0.4], [-0.1, 0.5, 7.0]])
    context = _make(system=_system(a), mesh=core._RegularKMesh([2, 3, 2]))
    expected_b = 2 * np.pi * np.linalg.inv(a).T
    np.testing.assert_allclose(context.reciprocal_lattice, expected_b, rtol=3e-15, atol=1e-16)
    np.testing.assert_allclose(a.T @ context.reciprocal_lattice, 2 * np.pi * np.eye(3), atol=2e-15)
    mesh = np.array([2, 3, 2])
    for index in range(12):
        m = np.array(np.unravel_index(index, mesh))
        k, q = context.k_record(index), context.transfer_record(index)
        numerator = 2 * m
        centered = np.where(numerator >= mesh, numerator - 2 * mesh, numerator)
        np.testing.assert_array_equal(k.modular_doubled_address, numerator)
        np.testing.assert_array_equal(q.modular_doubled_address, numerator)
        np.testing.assert_array_equal(q.centered_doubled_numerator, centered)
        np.testing.assert_array_equal(q.centered_reciprocal_wrap, numerator >= mesh)
        np.testing.assert_array_equal(k.fractional, m / mesh)
        np.testing.assert_array_equal(q.fractional, centered / (2 * mesh))
        np.testing.assert_allclose(k.cartesian, expected_b @ (m / mesh), atol=1e-15)
        np.testing.assert_allclose(q.cartesian, expected_b @ (centered / (2 * mesh)), atol=1e-15)
        assert q.gamma == (index == 0)
        assert q.self_conjugate == bool(np.all((2 * m) % mesh == 0))
        for bra in range(12):
            mb = np.array(np.unravel_index(bra, mesh))
            expected_ket = np.ravel_multi_index(tuple((mb + m) % mesh), mesh)
            assert context.ket_index(bra, index) == expected_ket
    # Transposing a skew direct lattice is a different physical input.
    assert context.source_context_identity_sha256 != _make(system=_system(a.T), mesh=core._RegularKMesh([2, 3, 2])).source_context_identity_sha256


def test_resource_caps_are_not_scientific_identity_and_exact_inventory_is_exposed():
    context = _make()
    inv = context.inventory
    assert inv.variable_owned_numeric_bytes == 0
    assert inv.fixed_context_storage_bytes == core._PERIODIC_GAUSSIAN_SOURCE_CONTEXT_STORAGE_BYTES
    assert inv.ao.function_count == 4
    assert (inv.ao.shell_count, inv.ao.contraction_count, inv.ao.exponent_count,
            inv.ao.coefficient_count) == (2, 2, 3, 3)
    assert inv.ao.borrowed_active_numeric_bytes == 96
    assert inv.ao.content_wire_bytes == 75 + 16 + 90 + 48
    assert inv.auxiliary.borrowed_active_numeric_bytes == 56
    assert inv.combined_borrowed_active_numeric_bytes == 152
    assert inv.work_units_upper_bound == 128 * (inv.ao.content_wire_bytes + inv.auxiliary.content_wire_bytes) + 64 * 6 + 4096
    cap = _caps()
    for name in _CAP_FIELDS:
        setattr(cap, name, 2 * getattr(cap, name))
    assert context.source_context_identity_sha256 == _make(caps=cap).source_context_identity_sha256


def test_same_basis_object_has_explicit_two_role_inventory_not_silent_dedup():
    basis = _identity_fixture()
    context = _make(ao=basis, auxiliary=basis)
    inv = context.inventory
    assert inv.combined_borrowed_active_numeric_bytes == 2 * inv.ao.borrowed_active_numeric_bytes
    assert context.ao_basis_identity_sha256 == context.auxiliary_basis_identity_sha256


def _exact_caps(context):
    inv = context.inventory
    cap = _caps()
    values = (
        inv.fixed_context_storage_bytes, context.n_kpoints,
        inv.ao.shell_count + inv.auxiliary.shell_count,
        inv.ao.contraction_count + inv.auxiliary.contraction_count,
        inv.ao.exponent_count + inv.ao.coefficient_count + inv.auxiliary.exponent_count + inv.auxiliary.coefficient_count,
        inv.ao.content_wire_bytes + inv.auxiliary.content_wire_bytes,
        inv.combined_borrowed_active_numeric_bytes, inv.work_units_upper_bound,
    )
    for name, value in zip(_CAP_FIELDS, values):
        setattr(cap, name, value)
    return cap


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_every_exact_resource_boundary_and_cap_minus_one(field):
    original = _make()
    cap = _exact_caps(original)
    assert _make(caps=cap).source_context_identity_sha256 == original.source_context_identity_sha256
    original.verify_bases(_identity_fixture(), _auxiliary(), cap)
    setattr(cap, field, getattr(cap, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _make(caps=cap)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        original.verify_bases(_identity_fixture(), _auxiliary(), cap)


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_all_caps_must_be_explicit_and_positive(field):
    cap = _caps()
    setattr(cap, field, 0)
    with pytest.raises(ValueError, match="positive"):
        _make(caps=cap)


@pytest.mark.parametrize("field", _OPTION_FIELDS)
@pytest.mark.parametrize("value", [0.0, -0.0, -1.0, np.inf, np.nan])
def test_every_scientific_control_must_be_explicit_finite_positive(field, value):
    options = _options()
    setattr(options, field, value)
    with pytest.raises(ValueError, match="positive finite"):
        _make(options=options)


@pytest.mark.parametrize("field", _OPTION_FIELDS)
def test_each_valid_scientific_control_changes_source_identity(field):
    options = _options()
    setattr(options, field, 2 * getattr(options, field))
    assert _make(options=options).source_context_identity_sha256 != _make().source_context_identity_sha256


def test_negative_tolerance_cannot_exceed_rank_threshold_and_squared_cutoffs_are_finite():
    options = _options()
    options.metric_negative_tolerance = 1e-8
    with pytest.raises(ValueError):
        _make(options=options)
    for field in _OPTION_FIELDS[:2]:
        options = _options()
        setattr(options, field, np.finfo(float).max)
        with pytest.raises(OverflowError, match="squared cutoff"):
            _make(options=options)


def test_cap_failure_precedes_full_invalid_primitive_value_scan():
    bad = _identity_fixture(first_coefficient=np.nan)
    caps = _caps()
    caps.maximum_basis_content_wire_bytes = 1
    with pytest.raises((ValueError, RuntimeError), match="wire exceeds cap"):
        _make(ao=bad, caps=caps)
    with pytest.raises(ValueError, match="finite"):
        _make(ao=bad)


@pytest.mark.parametrize("basis", [
    lambda: _identity_fixture(first_exponent=0.0),
    lambda: _identity_fixture(first_exponent=np.nan),
    lambda: _basis([(1, (0, 0, 0), [0.7], [0.5], False)]),
    lambda: _basis([(7, (0, 0, 0), [0.7], [0.5], True)]),
])
def test_invalid_or_unsupported_basis_content_rejects(basis):
    with pytest.raises((ValueError, RuntimeError)):
        _make(ao=basis())


def test_only_three_dimensional_gamma_mesh_is_supported():
    for dim in (1, 2):
        system = _system()
        system.dim = dim
        with pytest.raises(ValueError, match="3D Gamma"):
            _make(system=system)
    with pytest.raises(ValueError, match="3D Gamma"):
        _make(mesh=core._RegularKMesh([2, 3, 1], [1, 0, 0]))


@pytest.mark.parametrize("lattice", [np.zeros((3, 3)), np.diag([1, 1, np.inf]), np.diag([1, 1, np.nan])])
def test_nonfinite_or_singular_geometry_rejects(lattice):
    with pytest.raises((ValueError, RuntimeError)):
        _make(system=_system(lattice))


def test_records_are_bounded_and_context_does_not_borrow_mutable_python_inputs():
    options, system = _options(), _system()
    context = _make(system=system, options=options)
    identity = context.source_context_identity_sha256
    options.reciprocal_energy_cutoff = 9.0
    system.lattice = np.eye(3) * 10
    copied_options = context.options
    copied_options.reciprocal_energy_cutoff = 7.0
    copied_lattice = context.direct_lattice
    copied_lattice[0, 0] = 100
    assert context.source_context_identity_sha256 == identity
    assert context.options.reciprocal_energy_cutoff == 2.0
    assert context.direct_lattice[0, 0] == 8.0
    for method in (context.k_record, context.transfer_record):
        with pytest.raises(IndexError):
            method(context.n_kpoints)
    for bra, q in ((6, 0), (0, 6)):
        with pytest.raises(IndexError):
            context.ket_index(bra, q)
    # Neither the Python API nor the native value offers an all-k table.
    assert not hasattr(context, "kpoints")
    assert not hasattr(context, "calculation_id")
