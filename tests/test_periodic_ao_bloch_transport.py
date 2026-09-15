"""Native Seitz coefficient transport, not an enabled correlation method.

Independent witnesses are Fourier transforms of real-torus scatters and
pointwise native Gaussian evaluation. No SCF or correlated energy is run.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from vibeqc import _vibeqc_core as core
from vibeqc.periodic.chi.symmetry import (
    build_aiccm2026dev_b_real_torus_ao_action,
    build_aiccm2026dev_b_symmetry_plan,
)


def _controls():
    options = core._PeriodicAOBlochTransportOptions()
    options.maximum_atom_mapping_residual_bohr = 1e-9
    options.maximum_basis_origin_residual_bohr = 1e-10
    options.maximum_rotation_orthogonality_residual = 1e-11
    options.maximum_polynomial_reconstruction_residual = 1e-10
    options.maximum_pure_rotation_unitarity_residual = 1e-10
    options.minimum_relative_lattice_volume = 1e-10
    inventory = core._PeriodicAOBlochTransportInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 1 << 20
    caps = core._PeriodicAOBlochTransportCaps()
    for field in (
        "maximum_atoms", "maximum_shells", "maximum_contractions",
        "maximum_basis_functions", "maximum_columns",
    ):
        setattr(caps, field, 512)
    caps.maximum_basis_numeric_lanes = 10000
    for field in (
        "maximum_borrowed_numerical_bytes", "maximum_owned_numerical_bytes",
        "maximum_control_storage_bytes", "maximum_per_replica_inventoried_bytes",
        "maximum_node_inventoried_bytes",
    ):
        setattr(caps, field, 16 << 20)
    caps.maximum_work_units = 10**9
    return options, inventory, caps


def _operation(rotation=None, translation=None):
    w = np.eye(3, dtype=np.int32) if rotation is None else np.asarray(rotation, dtype=np.int32)
    tau = np.zeros(3) if translation is None else np.asarray(translation, dtype=float)
    return core._make_periodic_ao_bloch_operation(
        np.ascontiguousarray(w), np.ascontiguousarray(tau),
    )


def _basis(system, angular=(0, 1, 2), *, pure=True):
    shells = [
        core.ShellInfo(i, l, pure, [0.8, 1.5], [0.7, 0.2], list(atom.xyz))
        for i, atom in enumerate(system.unit_cell) for l in angular
    ]
    return core.BasisSet(system.unit_cell_molecule(), shells, "transport-test", False)


def _diamond(*, unwrapped=False):
    lattice = np.array([[0, 4, 4], [4, 0, 4], [4, 4, 0]], dtype=float)
    frac = np.array([[0, 0, 0], [0.25, 0.25, 0.25]], dtype=float)
    if unwrapped:
        frac[1] += [1, -1, 2]
    return core.PeriodicSystem(3, lattice, [core.Atom(6, lattice @ f) for f in frac])


def _one_atom(*, hexagonal=False):
    lattice = np.eye(3) * 8
    if hexagonal:
        lattice = np.array([[8, 4, 0], [0, 4 * np.sqrt(3), 0], [0, 0, 8]])
    return core.PeriodicSystem(3, lattice, [core.Atom(2, [0, 0, 0])])


def _coefficients(n, columns=3):
    rng = np.random.default_rng(507)
    return rng.normal(size=(n, columns)) + 1j * rng.normal(size=(n, columns))


def _apply(system, basis, op, mesh, source, tr, c, *, controls=None):
    return core._apply_periodic_ao_bloch_operation(
        basis, system, op, mesh, source, tr, c, *(_controls() if controls is None else controls),
    )


def _fractional_oracle(mesh, source, rotation, tr):
    """Exact Fraction arithmetic; no native mapping or floating keys."""
    reciprocal = np.rint(np.linalg.inv(rotation).T).astype(int)
    q = [Fraction(a, 2 * n) for a, n in zip(mesh.address(source), mesh.mesh)]
    raw = [sum(int(reciprocal[i, j]) * q[j] for j in range(3)) for i in range(3)]
    if tr:
        raw = [-x for x in raw]
    wrap = [x.numerator // x.denominator for x in raw]
    doubled = [int((x - g) * 2 * n) for x, g, n in zip(raw, wrap, mesh.mesh)]
    return doubled, wrap


@pytest.mark.parametrize("n,tr,unwrapped", list(product((2, 3), (False, True), (False, True))))
def test_nonsymmorphic_coefficients_match_fourier_of_real_torus(n, tr, unwrapped):
    system = _diamond(unwrapped=unwrapped)
    basis = _basis(system)
    shape = (n, n, n)
    plan = build_aiccm2026dev_b_symmetry_plan(system, shape)
    candidates = [
        i for i, op in enumerate(plan.operations)
        if op.has_fractional_translation and np.any(op.atom_lattice_shifts)
    ]
    assert candidates
    cells = np.asarray(list(np.ndindex(shape)))
    grid = core._RegularKMesh(shape)
    source = len(grid) - 2
    c = _coefficients(basis.nbasis)
    phase = np.exp(2j * np.pi * (cells @ grid.fractional_at(source)))
    real_rows = (phase[:, None, None] * c[None, :, :]).reshape(-1, c.shape[1]) / len(grid)
    # This oracle is the existing UNTWISTED torus; shifted meshes are tested
    # separately against real-space Bloch functions below.
    if tr:
        real_rows = real_rows.conj()
    all_q = np.array([grid.fractional_at(i) for i in range(len(grid))])
    for index in candidates[:3]:
        source_op = plan.operations[index]
        op = _operation(source_op.rotation, source_op.translation)
        action = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, index)
        transformed = action.apply(real_rows).reshape(len(grid), basis.nbasis, -1)
        projected = np.einsum(
            "kr,raj->kaj", np.exp(-2j * np.pi * all_q @ cells.T), transformed,
        )
        result = _apply(system, basis, op, grid, source, tr, c)
        p = result.memory
        expected_address, expected_wrap = _fractional_oracle(grid, source, op.rotation, tr)
        assert list(p.target_doubled_address) == expected_address
        assert list(p.reciprocal_wrap) == expected_wrap
        assert p.target_index == grid.index(expected_address)
        assert result.coefficients_copy() == pytest.approx(projected[p.target_index], abs=2e-12)
        projected[p.target_index] = 0
        assert np.max(np.abs(projected)) < 3e-12
        assert not result.physical_orbital_sewing_certified
        assert p.retained_output_bytes == c.nbytes


@pytest.mark.parametrize("l,pure", list(product(range(7), (True, False))))
@pytest.mark.parametrize("improper", (False, True))
def test_shell_rotation_matches_native_gaussian_values(l, pure, improper):
    system = _one_atom(hexagonal=True)
    basis = _basis(system, (l,), pure=pure)
    w = np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]])
    if improper:
        w = -w
    op = _operation(w)
    grid = core._RegularKMesh([3, 3, 2])
    # Identity panels expose every rotation column without assuming its
    # Euclidean unitarity (Cartesian d and higher do NOT have it).
    c = np.eye(basis.nbasis, dtype=np.complex128)
    result = _apply(system, basis, op, grid, 4, False, c)
    rotation = np.asarray(system.lattice) @ w @ np.linalg.inv(system.lattice)
    points = np.random.default_rng(79).normal(size=(41, 3))
    evaluated = core.evaluate_ao(basis, points)
    transformed_values = core.evaluate_ao(basis, points @ rotation)
    assert evaluated @ result.coefficients_copy() == pytest.approx(transformed_values, abs=2e-11)
    if pure:
        q = result.coefficients_copy()
        assert q.conj().T @ q == pytest.approx(np.eye(q.shape[0]), abs=2e-11)
    elif l == 2:
        q = result.coefficients_copy()
        assert np.max(np.abs(q.T @ q - np.eye(q.shape[0]))) > 0.1
        overlap = core.compute_overlap(basis)
        assert q.conj().T @ overlap @ q == pytest.approx(overlap, abs=2e-12)


@pytest.mark.parametrize("shift,tr", list(product(((0, 0, 0), (1, 1, 1)), (False, True))))
def test_fractional_translation_bloch_function_covariance(shift, tr):
    system = _diamond()
    basis = _basis(system, (0, 1, 2))
    plan = build_aiccm2026dev_b_symmetry_plan(system, (3, 3, 3))
    # Some operations of the primitive reciprocal lattice do not preserve
    # the half-shifted coset. Inversion with the quarter translation does.
    selected = next(op for op in plan.operations if np.array_equal(op.rotation, -np.eye(3)))
    op = _operation(selected.rotation, selected.translation)
    grid = core._RegularKMesh([3, 3, 3], shift)
    source = 16
    c = _coefficients(basis.nbasis, 2)
    result = _apply(system, basis, op, grid, source, tr, c)
    a = np.asarray(system.lattice)
    reciprocal = 2 * np.pi * np.linalg.inv(a).T
    r = a @ op.rotation @ np.linalg.inv(a)
    translation = a @ op.translation
    points = np.random.default_rng(94).uniform(-1, 4, size=(11, 3))
    images = np.array(list(product(range(-4, 5), repeat=3))) @ a.T
    original_points = (points - translation) @ r
    lhs = core.evaluate_bloch_ao(
        basis, points, reciprocal @ grid.fractional_at(result.memory.target_index), images,
    ) @ result.coefficients_copy()
    rhs = core.evaluate_bloch_ao(
        basis, original_points, reciprocal @ grid.fractional_at(source), images,
    ) @ c
    assert lhs == pytest.approx(rhs.conj() if tr else rhs, abs=3e-12)


@pytest.mark.parametrize("n,shift,tr", list(product((2, 3), ((0, 0, 0), (1, 0, 0)), (False, True))))
def test_nonsymmorphic_composition_keeps_lattice_translation_phase(n, shift, tr):
    lattice = np.eye(3) * 8
    system = core.PeriodicSystem(3, lattice, [core.Atom(2, [0, 0, 0]), core.Atom(2, [4, 0, 0])])
    basis = _basis(system, (0, 1))
    grid = core._RegularKMesh([n, 1, 1], shift)
    op = _operation(translation=[0.5, 0, 0])
    c = _coefficients(basis.nbasis)
    first = _apply(system, basis, op, grid, n - 1, tr, c)
    second = _apply(
        system, basis, op, grid, first.memory.target_index, tr, first.coefficients_copy(),
    )
    # g is real and commutes with scalar time reversal: (g T)^2=g^2.
    expected_phase = np.exp(-2j * np.pi * grid.fractional_at(n - 1)[0])
    assert second.memory.target_index == n - 1
    assert second.coefficients_copy() == pytest.approx(expected_phase * c, abs=2e-12)


def test_exact_whole_mesh_and_shift_compatibility_not_just_gamma():
    system = _one_atom()
    basis = _basis(system, (0,))
    op = _operation([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c = _coefficients(1)
    for shape, shift in (([2, 3, 1], [0, 0, 0]), ([2, 2, 1], [1, 0, 0])):
        with pytest.raises((ValueError, RuntimeError), match="mesh|shift|coset"):
            _apply(system, basis, op, core._RegularKMesh(shape, shift), 0, False, c)


@pytest.mark.parametrize("tr", (False, True))
def test_every_diamond_operation_matches_real_torus(tr):
    system = _diamond()
    basis = _basis(system)
    shape = (2, 2, 2)
    symmetry = build_aiccm2026dev_b_symmetry_plan(system, shape)
    grid = core._RegularKMesh(shape)
    c = _coefficients(basis.nbasis, 2)
    cells = np.asarray(list(np.ndindex(shape)))
    source = 5
    rows = (
        np.exp(2j * np.pi * (cells @ grid.fractional_at(source)))[:, None, None]
        * c[None, :, :]
    ).reshape(-1, 2)
    if tr:
        rows = rows.conj()
    assert len(symmetry.operations) == 48
    for index, selected in enumerate(symmetry.operations):
        op = _operation(selected.rotation, selected.translation)
        result = _apply(system, basis, op, grid, source, tr, c)
        action = build_aiccm2026dev_b_real_torus_ao_action(system, basis, symmetry, index)
        target = grid.fractional_at(result.memory.target_index)
        expected = (
            np.exp(2j * np.pi * (cells @ target))[:, None, None]
            * result.coefficients_copy()[None, :, :]
        ).reshape(-1, 2)
        assert action.apply(rows) == pytest.approx(expected, abs=3e-12)


def test_linear_transport_respects_degenerate_band_unitary_and_empty_columns():
    system = _diamond()
    basis = _basis(system)
    op = _operation(-np.eye(3), [0.25, 0.25, 0.25])
    grid = core._RegularKMesh([2, 2, 2])
    c = _coefficients(basis.nbasis)
    u, _ = np.linalg.qr(_coefficients(3))
    base = _apply(system, basis, op, grid, 7, True, c).coefficients_copy()
    rotated = _apply(system, basis, op, grid, 7, True, np.ascontiguousarray(c @ u))
    assert rotated.coefficients_copy() == pytest.approx(base @ u.conj(), abs=2e-12)
    empty = _apply(system, basis, op, grid, 7, False, np.empty((basis.nbasis, 0), dtype=complex))
    assert empty.coefficients_copy().shape == (basis.nbasis, 0)
    assert empty.memory.retained_output_bytes == 0
    assert empty.memory.retained_mapping_bytes == empty.memory.mapping_workspace_bytes > 0
    # Inversion about tau/2 exchanges the two sites with zero cell shifts.
    # All radial/angular shell blocks follow their owning atom's destination.
    for atom in range(2):
        assert empty.atom_destination(atom) == 1 - atom
        assert empty.atom_lattice_shift(atom) == [0, 0, 0]
        for local_shell in range(3):
            assert empty.shell_destination(3 * atom + local_shell) == 3 * (1 - atom) + local_shell
    shifted_copy = empty.atom_lattice_shift(0)
    shifted_copy[0] = 17
    assert empty.atom_lattice_shift(0) == [0, 0, 0]
    for accessor, index in ((empty.atom_destination, 2), (empty.atom_lattice_shift, 2),
                            (empty.shell_destination, basis.nshells)):
        with pytest.raises(IndexError, match="index"):
            accessor(index)
    del system, basis, op, grid
    assert empty.atom_destination(1) == 0
    assert empty.shell_destination(5) == 2


@pytest.mark.parametrize("changed", ("coefficients", "exponents", "pure"))
def test_atom_species_alone_does_not_certify_basis_symmetry(changed):
    system = _diamond()
    original = _basis(system)
    shells = original.shells()
    target = next(s for s in shells if s.atom_index == 1 and s.l == 2)
    if changed == "pure":
        target.pure = False
    else:
        values = list(getattr(target, changed))
        values[0] *= 1.1
        setattr(target, changed, values)
    basis = core.BasisSet(system.unit_cell_molecule(), shells, "broken-radial-closure", True)
    with pytest.raises((ValueError, RuntimeError), match="basis|radial|shell|contraction"):
        _apply(system, basis, _operation(-np.eye(3), [.25] * 3), core._RegularKMesh([2] * 3),
               3, False, _coefficients(basis.nbasis))


@pytest.mark.parametrize("cap,field", (
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_borrowed_numerical_bytes", "borrowed_numerical_bytes"),
    ("maximum_control_storage_bytes", "control_storage_bytes"),
    ("maximum_per_replica_inventoried_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_inventoried_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
))
def test_exact_resource_caps_reject_before_coefficient_payload(cap, field):
    system = _one_atom()
    basis = _basis(system, (0, 1))
    op = _operation()
    grid = core._RegularKMesh([8, 8, 8])
    controls = _controls()
    c = _coefficients(basis.nbasis)
    plan = core._plan_periodic_ao_bloch_transport(basis, system, op, grid, 511, False, 3, *controls)
    needed = getattr(plan, field)
    assert needed > 0
    setattr(controls[2], cap, needed)
    result = _apply(system, basis, op, grid, 511, False, c, controls=controls)
    assert result.coefficients_copy() == pytest.approx(c, abs=1e-12)
    setattr(controls[2], cap, needed - 1)
    c[:] = np.nan  # If the payload is read first, the wrong error appears.
    with pytest.raises((ValueError, RuntimeError), match="cap|budget|limit"):
        _apply(system, basis, op, grid, 511, False, c, controls=controls)


def test_large_mesh_changes_no_per_k_allocation():
    system = _one_atom()
    basis = _basis(system)
    controls = _controls()
    plans = [core._plan_periodic_ao_bloch_transport(
        basis, system, _operation(), core._RegularKMesh([n] * 3), 0, False, 3, *controls,
    ) for n in (2, 8, 100000)]
    for field in ("retained_output_bytes", "peak_owned_numerical_bytes", "control_storage_bytes",
                  "work_units_upper_bound"):
        assert len({getattr(p, field) for p in plans}) == 1


def test_replica_inventory_counts_external_and_other_live_owners():
    system = _one_atom()
    basis = _basis(system)
    options, inventory, caps = _controls()
    args = (basis, system, _operation(), core._RegularKMesh([2] * 3), 0, False, 3)
    baseline = core._plan_periodic_ao_bloch_transport(*args, options, inventory, caps)
    inventory.numerical_replicas = 3
    inventory.external_node_bytes = 8192
    inventory.other_live_numerical_bytes_per_replica = 1024
    inventory.other_live_control_bytes_per_replica = 2048
    p = core._plan_periodic_ao_bloch_transport(*args, options, inventory, caps)
    assert p.per_replica_inventoried_bytes == baseline.per_replica_inventoried_bytes + 3072
    assert p.required_node_inventoried_bytes == 8192 + 3 * p.per_replica_inventoried_bytes
    inventory.numerical_replicas = 2**64 - 1
    with pytest.raises(OverflowError, match="count overflow"):
        core._plan_periodic_ao_bloch_transport(*args, options, inventory, caps)


@pytest.mark.parametrize("field", (
    "maximum_atom_mapping_residual_bohr", "maximum_basis_origin_residual_bohr",
    "maximum_rotation_orthogonality_residual", "maximum_polynomial_reconstruction_residual",
    "maximum_pure_rotation_unitarity_residual", "minimum_relative_lattice_volume",
))
@pytest.mark.parametrize("value", (np.nan, np.inf, -1.0))
def test_audit_tolerances_must_be_explicit_finite_and_valid(field, value):
    system = _one_atom()
    basis = _basis(system)
    controls = _controls()
    setattr(controls[0], field, value)
    with pytest.raises(ValueError, match="control|volume floor"):
        _apply(system, basis, _operation(), core._RegularKMesh([2] * 3), 0, False,
               _coefficients(basis.nbasis), controls=controls)


def test_floating_environment_admission_precedes_finite_value_checks():
    # Compile-mode guards cannot be toggled in an already loaded extension.
    # Pin their presence and order without changing the process-wide FP mode.
    path = Path(__file__).resolve().parents[1] / "cpp/src/periodic_ao_bloch_transport.cpp"
    source = path.read_text()
    control_body = source.split("void controls(", 1)[1].split("struct AtomMap", 1)[0]
    for token in ("__FAST_MATH__", "__FINITE_MATH_ONLY__", "FLT_EVAL_METHOD",
                  "FE_TONEAREST", "requires gradual underflow"):
        assert control_body.index(token) < control_body.index("const double tolerances")


@pytest.mark.parametrize("kind", ("float_rotation", "wide_rotation", "shape", "float_translation"))
def test_diagnostic_operation_factory_does_not_cast_descriptors(kind):
    rotation = np.eye(3, dtype=np.int32)
    translation = np.zeros(3)
    if kind == "float_rotation":
        rotation = rotation.astype(float)
    elif kind == "wide_rotation":
        rotation = rotation.astype(np.int64)
    elif kind == "shape":
        rotation = rotation.ravel()
    else:
        translation = translation.astype(np.float32)
    with pytest.raises(ValueError, match="int32.*float64"):
        core._make_periodic_ao_bloch_operation(rotation, translation)


def test_result_copy_is_bounded_and_independent():
    system = _one_atom()
    basis = _basis(system)
    c = _coefficients(basis.nbasis)
    result = _apply(system, basis, _operation(), core._RegularKMesh([2] * 3), 0, False, c)
    copied = result.coefficients_copy()
    assert not copied.flags.writeable
    copied.setflags(write=True)
    copied[:] = 0
    assert result.coefficients_copy() == pytest.approx(c, abs=1e-12)
    assert result.coefficient(0, 0) == pytest.approx(c[0, 0], abs=1e-12)
    with pytest.raises(IndexError, match="coefficient index"):
        result.coefficient(basis.nbasis, 0)


@pytest.mark.parametrize("material", ("mgo", "diamond"))
def test_pob_tzvp_rev2_one_panel_on_project_mesh_without_scf(material):
    """8^3 ADDRESSING only: a few orbital columns, never target-sized HF/CC.

    The independent real-torus witness holds roughly one MiB of coefficient
    rows, not a supercell Hamiltonian, ERIs or amplitudes.
    """
    system = _diamond()
    if material == "mgo":
        a = np.asarray(system.lattice)
        system = core.PeriodicSystem(
            3, a, [core.Atom(12, [0, 0, 0]), core.Atom(8, a @ np.array([.5, .5, .5]))]
        )
    basis = core.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    grid = core._RegularKMesh([8, 8, 8])
    plan = build_aiccm2026dev_b_symmetry_plan(system, (8, 8, 8))
    if material == "mgo":
        index = next(
            i for i, op in enumerate(plan.operations)
            if np.array_equal(op.rotation, -np.eye(3))
        )
    else:
        index = next(
            i for i, op in enumerate(plan.operations)
            if op.has_fractional_translation and np.any(op.atom_lattice_shifts)
        )
    selected = plan.operations[index]
    action = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, index)
    op = _operation(selected.rotation, selected.translation)
    c = _coefficients(basis.nbasis, 2)
    source = 319
    result = _apply(system, basis, op, grid, source, False, c)
    cells = np.asarray(list(np.ndindex((8, 8, 8))))
    source_phase = np.exp(2j * np.pi * (cells @ grid.fractional_at(source)))
    target_phase = np.exp(2j * np.pi * (cells @ grid.fractional_at(result.memory.target_index)))
    rows = (source_phase[:, None, None] * c[None, :, :]).reshape(-1, 2)
    expected = (target_phase[:, None, None] * result.coefficients_copy()[None, :, :]).reshape(-1, 2)
    assert action.apply(rows) == pytest.approx(expected, abs=8e-12)
    assert result.memory.peak_owned_numerical_bytes < (128 << 10)
    assert not result.physical_orbital_sewing_certified


@pytest.mark.parametrize("kind", ("real", "fortran", "nonfinite"))
def test_bad_coefficient_payload_rejected(kind):
    system = _one_atom()
    basis = _basis(system)
    c = _coefficients(basis.nbasis)
    if kind == "real":
        c = c.real.copy()
    elif kind == "fortran":
        c = np.asfortranarray(c)
    else:
        c[0, 0] = np.inf
    with pytest.raises(
        (ValueError, RuntimeError, OverflowError), match="complex128|finite|coefficient",
    ):
        _apply(system, basis, _operation(), core._RegularKMesh([2] * 3), 0, False, c)


@pytest.mark.parametrize("kind", ("shear", "missing_image", "wrong_origin", "singular_rotation"))
def test_incompatible_geometry_and_rotation_rejected(kind):
    system = _one_atom()
    basis = _basis(system)
    op = _operation()
    if kind == "shear":
        op = _operation([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    elif kind == "missing_image":
        op = _operation(translation=[0.25, 0, 0])
    elif kind == "wrong_origin":
        other = core.PeriodicSystem(3, system.lattice, [core.Atom(2, [1, 0, 0])])
        basis = _basis(other)
    else:
        op = _operation(np.zeros((3, 3), dtype=int))
    with pytest.raises((ValueError, RuntimeError), match="rotation|atom|origin|lattice|unimodular"):
        _apply(
            system, basis, op, core._RegularKMesh([2] * 3), 0, False, _coefficients(basis.nbasis),
        )
