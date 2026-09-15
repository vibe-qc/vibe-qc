"""Bounded common finite BIPOLE panels, not production HF or DLPNO proof.

Sun (2023), doi:10.1063/5.0155815, Eqs. 18-22, fixes the SR zero-mode
subtraction separately from probe-charge exchange corrections. Independent
Gaussian moments and polynomial Fourier products are reused as oracles;
neither native component is the sole numerical reference for composition.
"""

from __future__ import annotations

import gc
import itertools
from collections import Counter

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bipole_erfc_panel import (
    _CLOSED_IMAGES, _angular_basis, _bloch_controls, _exact_phase,
    _oracle as _erfc_oracle, _two_s,
)
from tests.test_bipole_ewald_gram import (
    _CELLS, _GRAM_LATTICE, _cell_caps, _cell_oracle, _gram_bundle, _gram_oracle, _gram_records,
)
from tests.test_periodic_aopair_fourier_panel import _basis, _system
from tests.test_periodic_ao_bloch_transport import (
    _controls as _transport_controls, _operation,
)


def _source_caps():
    caps = core._BipoleFiniteSourceCaps()
    caps.maximum_kpoints = 4096
    caps.maximum_images = caps.maximum_cells = 64
    caps.maximum_context_storage_bytes = 1 << 20
    caps.maximum_borrowed_numerical_bytes = 16 << 20
    caps.maximum_work_units = 10**11
    return caps


def _bundle(*, q=0, kl=1, kr=2, shift=(0, 0, 0), mesh=(3, 1, 1),
            basis=None, images=None, cells=None, left=(0, 4), right=(0, 4)):
    b = _gram_bundle(q=q, left=kl, right=kr, shift=shift, mesh=mesh,
                     basis=_two_s() if basis is None else basis)
    b.selection.left_pair_begin, b.selection.left_pair_count = left
    b.selection.right_pair_begin, b.selection.right_pair_count = right
    b.images = np.ascontiguousarray(
        _CLOSED_IMAGES if images is None else images, dtype=np.int64,
    ).reshape(-1, 3, 3).copy()
    b.cells = np.ascontiguousarray(
        _CELLS if cells is None else cells, dtype=np.int64,
    ).reshape(-1, 3).copy()
    options = core._BipoleFiniteSourceOptions()
    options.omega = b.options.omega
    options.reciprocal_energy_cutoff = b.options.reciprocal_energy_cutoff
    options.zero_mode = core._BipoleFiniteZeroMode.G0Omitted
    options.require_image_permutation_closure = True
    options.require_cell_inversion_closure = True
    options.require_reciprocal_conjugacy = True
    b.options = options
    b.source_caps = _source_caps()
    inventory = core._BipoleFinitePanelInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 8 << 20
    inventory.reciprocal_block_size = 3
    b.inventory = inventory
    caps = core._BipoleFinitePanelCaps()
    caps.source = _source_caps()
    caps.short_range = _bloch_controls()[3]
    caps.short_range.maximum_support_image_comparisons = 8 * 64**2
    caps.long_range = b.caps
    caps.zero_mode = _cell_caps()
    caps.maximum_borrowed_numerical_bytes = 16 << 20
    caps.maximum_owned_numerical_bytes = 128 << 20
    caps.maximum_control_storage_bytes = 32 << 20
    caps.maximum_worker_bytes = 256 << 20
    caps.maximum_node_bytes = 512 << 20
    caps.maximum_work_units = 10**11
    b.caps = caps
    return b


def _source(b, *, plan=False):
    args = (b.basis, b.mesh, b.images, b.cells, b.options, b.source_caps)
    if plan:
        return core._plan_bipole_finite_source(*args)
    return core._make_bipole_finite_source(b.basis, b.system, *args[1:])


def _panel(b, source=None, *, plan=False):
    fn = core._plan_bipole_finite_panel if plan else core._make_bipole_finite_panel
    return fn(_source(b) if source is None else source, b.basis, b.images,
              b.cells, b.selection, b.inventory, b.caps)


def _components(b, *, raw=None):
    s = b.selection
    left = s.left_pair_begin, s.left_pair_count
    right = s.right_pair_begin, s.right_pair_count
    lattice = np.asarray(b.system.lattice)
    if raw is None:
        raw = _erfc_oracle(b.basis, b.images, lattice=lattice, left=left,
                           right=right, omega=b.options.omega)
    phases = [_exact_phase(labels, tuple(b.mesh.mesh), tuple(b.mesh.is_shift),
                           s.q_index, s.left_k_index, s.right_k_index)
              for labels in b.images]
    sr = np.einsum("i,ilr->lr", phases, raw)
    lr = _gram_oracle(b)
    zero = np.zeros_like(lr)
    if s.q_index == 0:
        reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
        bl = _cell_oracle(b.basis, np.zeros((1, 3)), lattice,
                          -reciprocal @ b.mesh.fractional_at(s.left_k_index), b.cells)[:, 0]
        br = _cell_oracle(b.basis, np.zeros((1, 3)), lattice,
                          -reciprocal @ b.mesh.fractional_at(s.right_k_index), b.cells)[:, 0]
        coefficient = np.pi / (abs(np.linalg.det(lattice)) * b.options.omega**2)
        zero = coefficient * np.outer(bl[left[0]:sum(left)].conj(), br[right[0]:sum(right)])
    return sr, lr, zero


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
def test_finite_panel_matches_independent_sr_lr_and_zero_mode(q, shift):
    b = _bundle(q=q, shift=shift)
    sr, lr, zero = _components(b)
    result = _panel(b)
    np.testing.assert_allclose(result.values_copy(), sr + lr - zero, atol=6e-13, rtol=5e-12)
    assert result.memory.subtract_zero_mode == (q == 0)
    assert result.diagnostics.zero_mode_pair_values == (8 if q == 0 else 0)
    assert result.diagnostics.image_permutation_support_certified
    assert result.diagnostics.long_range.reciprocal_conjugacy_audited
    assert not result.physical_hamiltonian_certified and not result.symmetry_certified
    assert not result.source.physical_hamiltonian_certified and not result.source.symmetry_certified


@pytest.mark.parametrize("q", [0, 1])
def test_finite_panel_angular_selected_quartet(q):
    b = _bundle(q=q, basis=_angular_basis(), left=(14, 1), right=(27, 1))
    sr, lr, zero = _components(b)
    np.testing.assert_allclose(_panel(b).values_copy(), sr + lr - zero, atol=6e-13, rtol=5e-12)


def test_zero_mode_has_no_hidden_full_mesh_weight():
    b = _bundle(q=0)
    sr, lr, zero = _components(b)
    result = _panel(b)
    assert np.max(np.abs(zero)) > 1e-3
    np.testing.assert_allclose(sr + lr - result.values_copy(), zero, atol=6e-13, rtol=5e-12)
    # The same physical k points on a refined mesh must not divide this
    # selected ERI or its subtraction by Nk. J/K consumers own that weight.
    refined = _bundle(mesh=(6, 1, 1), q=0, kl=2, kr=4)
    np.testing.assert_allclose(_panel(refined).values_copy(), result.values_copy(),
                               atol=6e-13, rtol=5e-12)


def test_selected_pair_tiling_is_exact():
    b = _bundle()
    source = _source(b)
    whole = _panel(b, source).values_copy()
    pieces = []
    for begin in range(4):
        b.selection.left_pair_begin, b.selection.left_pair_count = begin, 1
        pieces.append(_panel(b, source).values_copy())
    np.testing.assert_array_equal(np.vstack(pieces), whole)
    b.selection.left_pair_begin, b.selection.left_pair_count = 0, 4
    pieces = []
    for begin in range(4):
        b.selection.right_pair_begin, b.selection.right_pair_count = begin, 1
        pieces.append(_panel(b, source).values_copy())
    np.testing.assert_array_equal(np.hstack(pieces), whole)


@pytest.mark.parametrize("block", [1, 2, 7])
def test_reciprocal_tiling_and_resource_changes_preserve_scientific_identity(block):
    b = _bundle()
    source = _source(b)
    first = _panel(b, source)
    b.inventory.reciprocal_block_size = block
    b.inventory.backend_margin_bytes_per_replica *= 2
    b.source_caps.maximum_work_units *= 2
    other_source = _source(b)
    assert source.source_identity_sha256 == other_source.source_identity_sha256
    second = _panel(b, other_source)
    np.testing.assert_array_equal(first.values_copy(), second.values_copy())
    for field in ("input_identity_sha256", "short_range_payload_identity_sha256",
                  "long_range_payload_identity_sha256", "zero_mode_identity_sha256",
                  "payload_identity_sha256"):
        value = getattr(first, field)
        assert len(value) == 64 and int(value, 16) >= 0
        assert value == getattr(second, field)


@pytest.mark.parametrize("change", ["basis", "images", "image_order", "cell_order"])
def test_changed_borrowed_content_cannot_reuse_source(change):
    b = _bundle()
    source = _source(b)
    if change == "basis":
        b.basis = _basis([(0, (0.2, -0.2, 0.15), [0.55, 1.1], [0.7, -0.12], True),
                          (0, (0.6, 0.3, -0.25), [0.7], [0.8], True)])
    elif change == "images":
        b.images[0, 0, 0] += 1
    elif change == "image_order":
        b.images = b.images[::-1].copy()
    else:
        b.cells = b.cells[::-1].copy()
    with pytest.raises(ValueError, match="identity|match|changed|content|source"):
        _panel(b, source)


@pytest.mark.parametrize("change", ["omega", "cutoff", "lattice", "shift", "image_order", "cell_order"])
def test_scientific_source_changes_are_not_resource_changes(change):
    b = _bundle()
    original = _source(b)
    if change == "omega":
        b.options.omega *= 1.1
    elif change == "cutoff":
        b.options.reciprocal_energy_cutoff *= 1.1
    elif change == "lattice":
        lattice = np.asarray(b.system.lattice).copy()
        lattice[0, 0] *= 1.1
        b.system = _system(lattice)
    elif change == "shift":
        b.mesh = core._RegularKMesh([3, 1, 1], [1, 0, 0])
    elif change == "image_order":
        b.images = b.images[::-1].copy()
    else:
        b.cells = b.cells[::-1].copy()
    assert original.source_identity_sha256 != _source(b).source_identity_sha256


def test_context_snapshots_controls_and_excludes_nonoperator_system_fields():
    b = _bundle()
    source = _source(b)
    omega = b.options.omega
    identity = source.source_identity_sha256
    b.options.omega *= 2
    assert source.options.omega == omega
    exposed_options = source.options
    exposed_options.omega *= 3
    assert source.options.omega == omega
    b.options.omega = omega
    b.system = core.PeriodicSystem(3, np.asarray(b.system.lattice), [core.Atom(8, [0.2, 0.1, 0.3])])
    assert _source(b).source_identity_sha256 == identity
    np.testing.assert_allclose(source.cell_volume, abs(np.linalg.det(_GRAM_LATTICE)), rtol=3e-15)
    np.testing.assert_allclose(source.zero_mode_coefficient,
                               np.pi / (abs(np.linalg.det(_GRAM_LATTICE)) * omega**2), rtol=3e-15)


def test_result_pins_source_and_does_not_alias_borrowed_support_or_output():
    b = _bundle()
    source = _source(b)
    result = _panel(b, source)
    identity = source.source_identity_sha256
    values = result.values_copy()
    assert not values.flags.writeable
    b.images[:] = 0
    b.cells[:] = 0
    del source, b
    gc.collect()
    assert result.source.source_identity_sha256 == identity
    copied = result.values_copy()
    assert not np.shares_memory(values, copied)
    np.testing.assert_array_equal(values, copied)
    assert result.element(1, 2) == values[1, 2]
    with pytest.raises((ValueError, IndexError)):
        result.element(4, 0)


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_kpoints", "n_kpoints"), ("maximum_images", "image_count"),
    ("maximum_cells", "cell_count"), ("maximum_context_storage_bytes", "context_storage_bytes"),
    ("maximum_borrowed_numerical_bytes", "borrowed_numerical_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_exact_context_caps_then_rejection_before_bad_labels(field, plan_field):
    b = _bundle()
    plan = _source(b, plan=True)
    amount = getattr(plan, plan_field)
    setattr(b.source_caps, field, amount)
    _source(b)
    b.images[0, 0, 0] = np.iinfo(np.int64).min
    _source(b, plan=True)  # Count-only admission does not consume label values.
    setattr(b.source_caps, field, amount - 1)
    with pytest.raises(ValueError, match="cap|work|storage|bytes"):
        _source(b)


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_borrowed_numerical_bytes", "borrowed_numerical_bytes"),
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_control_storage_bytes", "control_storage_bytes"),
    ("maximum_worker_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_bytes", "required_node_inventoried_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_exact_outer_caps_then_rejection_before_source_payload(field, plan_field):
    b = _bundle()
    source = _source(b)
    plan = _panel(b, source, plan=True)
    amount = getattr(plan, plan_field)
    setattr(b.caps, field, amount)
    _panel(b, source)
    b.images[0, 0, 0] = np.iinfo(np.int64).min
    _panel(b, source, plan=True)
    setattr(b.caps, field, amount - 1)
    with pytest.raises(ValueError, match="cap|work|storage|bytes"):
        _panel(b, source)


@pytest.mark.parametrize("support", ["images", "cells"])
def test_context_does_not_preclaim_requested_support_checks(support):
    b = _bundle()
    if support == "images":
        b.images = b.images[:-1].copy()
    else:
        b.cells = b.cells[:-1].copy()
    source = _source(b)
    assert not source.physical_hamiltonian_certified and not source.symmetry_certified
    _panel(b, source, plan=True)
    with pytest.raises(ValueError, match="multiset|inversion|closure|closed"):
        _panel(b, source)


def test_inversion_closed_common_cells_do_not_certify_screw_covariance():
    """Site reanchoring is not the same operation as R -> -R.

    The screw swaps the two s sites and shifts the second by one z cell.
    Its AO-pair action moves relative cells by a site-dependent amount, so
    the common finite window can fail even with both support checks true.
    An eventual HF/group certificate must include its actual overlap source.
    """
    lattice = np.diag([4.0, 5.0, 3.0])
    positions = np.array([[0.2, 0.4, 0.0], [-0.2, -0.4, 1.5]])
    # _basis preserves supplied coefficients; it does not normalize them.
    normalization = (0.4 / np.pi)**0.75
    basis = _basis([(0, tuple(p), [0.2], [normalization], True) for p in positions])
    b = _bundle(mesh=(1, 1, 4), kl=1, kr=1, basis=basis,
                cells=[[0, 0, 0], [0, 0, 1], [0, 0, -1]])
    b.system = core.PeriodicSystem(3, lattice, [core.Atom(2, p) for p in positions])
    rotation = np.diag([-1.0, -1.0, 1.0])
    translation = np.array([0.0, 0.0, 1.5])
    np.testing.assert_array_equal(rotation @ positions[0] + translation, positions[1])
    np.testing.assert_array_equal(rotation @ positions[1] + translation,
                                  positions[0] + lattice[:, 2])
    k = 2 * np.pi * np.linalg.inv(lattice).T @ b.mesh.fractional_at(1)
    overlap = _cell_oracle(basis, np.zeros((1, 3)), lattice, k, b.cells).reshape(2, 2)
    sewing = np.array([[0, np.exp(-1j * k @ lattice[:, 2])], [1, 0]])
    np.testing.assert_allclose(overlap, overlap.conj().T, atol=1e-14)
    np.testing.assert_allclose(overlap.diagonal(), np.ones(2), atol=1e-14)
    defect = np.max(np.abs(sewing.conj().T @ overlap @ sewing - overlap))
    # Equal normalized s exponents give the unmatched tail exp(-0.1*r^2),
    # with r^2 = 0.4^2 + 0.8^2 + 4.5^2; both quadratures contribute.
    np.testing.assert_allclose(defect, np.sqrt(2) * np.exp(-2.105),
                               atol=1e-14, rtol=1e-13)
    result = _panel(b)
    assert result.diagnostics.image_permutation_support_certified
    assert result.diagnostics.long_range.cell_inversion_closed
    assert result.diagnostics.long_range.reciprocal_conjugacy_audited
    assert not result.physical_hamiltonian_certified
    assert not result.symmetry_certified
    # Same ell convention as the native AO transport: destination = moved + A*ell.
    # For the (site0,site1 | site0,site1) quartet the second/fourth ell is -ez.
    shifts = np.array([[0, 0, 0], [0, 0, -1], [0, 0, 0], [0, 0, -1]], np.int64)
    audit = _support(b, np.diag([-1, -1, 1]).astype(np.int32), shifts)
    assert not audit.left_cells_closed and not audit.right_cells_closed
    assert tuple(audit.first_left_cell) == (0, 0, 2)
    assert not audit.images_closed
    assert not audit.symmetry_certified


def _support_caps():
    caps = core._BipoleFiniteSupportCaps()
    caps.source = _source_caps()
    caps.maximum_label_comparisons = 10**6
    caps.maximum_inventoried_bytes = 32 << 20
    caps.maximum_work_units = 10**11
    return caps


def _support(b, rotation=None, shifts=None, *, source=None, caps=None, plan=False):
    args = (_source(b) if source is None else source, b.basis, b.images, b.cells)
    caps = _support_caps() if caps is None else caps
    if plan:
        return core._plan_bipole_finite_support_action(*args, caps)
    rotation = np.eye(3, dtype=np.int32) if rotation is None else rotation
    shifts = np.zeros((4, 3), np.int64) if shifts is None else shifts
    return core._audit_bipole_finite_support_action(*args, rotation, shifts, caps)


def _support_oracle(b, rotation, shifts):
    """Compare whole multisets using Python integers, independently of native scans."""
    w = [[int(x) for x in row] for row in rotation]
    ell = [[int(x) for x in row] for row in shifts]

    def moved(row, anchor, center):
        return tuple(sum(w[d][e] * int(row[e]) for e in range(3))
                     + ell[anchor][d] - ell[center][d] for d in range(3))

    cells = Counter(tuple(int(x) for x in row) for row in b.cells)
    left = Counter(moved(row, 0, 1) for row in b.cells)
    right = Counter(moved(row, 2, 3) for row in b.cells)
    images = Counter(tuple(int(x) for x in row.flat) for row in b.images)
    transformed = Counter(tuple(x for c, row in enumerate(image, 1)
                                for x in moved(row, 0, c)) for image in b.images)
    return cells == left, cells == right, images == transformed


def _mapped_bundle(*, mesh=(1, 1, 3), shift=(0, 0, 0), angular=(0,), pure=True):
    positions = [(0.2, 0.4, 0.0), (-0.2, -0.4, 1.5)]
    basis = _basis([(l, p, [0.2], [0.7], pure) for p in positions for l in angular])
    b = _bundle(mesh=mesh, shift=shift, basis=basis, kl=0, kr=0,
                cells=[[0, 0, 0], [0, 0, 1], [0, 0, -1]], images=[[[0, 0, 0]] * 3])
    b.system = core.PeriodicSystem(3, np.diag([4., 5., 3.]),
                                   [core.Atom(2, p) for p in positions])
    return b


def _mapped_controls(shells=(0, 1, 0, 1), source_k=0, tr=False):
    options, inventory, transport_caps = _transport_controls()
    selection = core._BipoleFiniteMappedSupportSelection()
    selection.source_shells = shells
    selection.source_k_index = source_k
    selection.time_reversal = tr
    caps = core._BipoleFiniteMappedSupportCaps()
    caps.support = _support_caps()
    caps.transport = transport_caps
    caps.maximum_per_replica_inventoried_bytes = 32 << 20
    caps.maximum_node_inventoried_bytes = 64 << 20
    caps.maximum_work_units = 10**9
    return selection, options, inventory, caps


def _mapped(b, *, controls=None, op=None, source=None, plan=False):
    controls = _mapped_controls() if controls is None else controls
    op = _operation(np.diag([-1, -1, 1]), [0, 0, .5]) if op is None else op
    fn = core._plan_bipole_finite_mapped_support if plan else core._audit_bipole_finite_mapped_support
    return fn(_source(b) if source is None else source, b.basis, b.system, op,
              b.images, b.cells, *controls)


@pytest.mark.parametrize("mesh,shift", [((1, 1, 1), (0, 0, 0)),
                                       ((1, 1, 2), (0, 0, 0)),
                                       ((1, 1, 3), (0, 0, 1)),
                                       ((2, 2, 2), (1, 1, 1))])
@pytest.mark.parametrize("tr", [False, True])
def test_mapped_support_derives_screw_reanchoring_on_full_shifted_mesh(mesh, shift, tr):
    b = _mapped_bundle(mesh=mesh, shift=shift)
    # Expected mapping follows physical atom coordinates, independent of the
    # native map: site 0 -> 1 with ell=0; site 1 -> 0 with ell=-ez.
    shifts = np.array([[0, 0, 0], [0, 0, -1]] * 2, dtype=np.int64)
    for k in range(len(b.mesh)):
        result = _mapped(b, controls=_mapped_controls(source_k=k, tr=tr))
        assert result.source_atoms == [0, 1, 0, 1]
        assert result.destination_atoms == result.destination_shells == [1, 0, 1, 0]
        np.testing.assert_array_equal(np.asarray(result.atom_shifts).reshape(4, 3), shifts)
        audit = result.support
        assert (audit.left_cells_closed, audit.right_cells_closed, audit.images_closed) == (
            _support_oracle(b, np.diag([-1, -1, 1]), shifts))
        assert audit.first_left_cell == [0, 0, 2]
        assert result.transport.mapped_atoms == result.transport.mapped_shells == 2
        assert result.memory.transport.n_columns == result.memory.transport.retained_output_bytes == 0
        assert result.memory.transport.retained_mapping_bytes > 0
        assert not result.symmetry_certified and not result.physical_hamiltonian_certified


@pytest.mark.parametrize("angular,pure", [((0,), True), ((0, 1, 2), True), ((0,), False)])
def test_mapped_support_selected_closed_angular_shells_do_not_admit_other_pairs(angular, pure):
    b = _mapped_bundle(angular=angular, pure=pure)
    last = len(angular) - 1
    result = _mapped(b, controls=_mapped_controls((last,) * 4))
    assert result.destination_shells == [last + len(angular)] * 4
    assert result.support.left_cells_closed and result.support.right_cells_closed and result.support.images_closed
    assert result.transport.rotated_contractions == 2 * len(angular)
    assert not result.symmetry_certified
    other = _mapped(b, controls=_mapped_controls((last, last + len(angular), last, last)))
    assert not other.support.left_cells_closed and other.support.right_cells_closed


@pytest.mark.parametrize("defect", ["species", "position", "radial", "pure", "owner", "shear"])
def test_mapped_support_refuses_geometry_or_basis_incompatible_operation(defect):
    b = _mapped_bundle(angular=(0, 2))
    op = None
    if defect in ("species", "position"):
        atoms = list(b.system.unit_cell)
        atoms[1] = core.Atom(3 if defect == "species" else 2,
                            [-.2, -.4, 1.5 + (0.02 if defect == "position" else 0)])
        b.system = core.PeriodicSystem(3, b.system.lattice, atoms)
    elif defect == "shear":
        op = _operation([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    else:
        shells = b.basis.shells()
        if defect == "radial":
            shells[2].exponents = [.3]
        elif defect == "pure":
            # Both pure s and Cartesian s are admitted by the finite source;
            # their strict native shell descriptors still have to match.
            shells[2].pure = False
        molecule = (core.Molecule(list(reversed(b.system.unit_cell))) if defect == "owner"
                    else b.system.unit_cell_molecule())
        # BasisSet derives ownership from the molecule's order and shell
        # origins; editing the descriptive ShellInfo.atom_index would not
        # change the native map.
        b.basis = core.BasisSet(molecule, shells, "invalid-map", True)
    # A self-consistent immutable source alone cannot certify the operation.
    with pytest.raises((ValueError, RuntimeError), match="mapping|radial|basis origin|orthogonality"):
        _mapped(b, op=op)


def test_mapped_support_requires_exact_source_lattice_and_unchanged_payload():
    b = _mapped_bundle()
    source = _source(b)
    original = np.asarray(b.system.lattice).copy()
    lattice = original.copy()
    lattice[0, 0] = np.nextafter(lattice[0, 0], np.inf)
    b.system = core.PeriodicSystem(3, lattice, b.system.unit_cell)
    with pytest.raises(ValueError, match="exact finite-source 3D lattice"):
        _mapped(b, source=source)
    b.system = core.PeriodicSystem(3, original, b.system.unit_cell)
    b.cells[1, 2] = 2
    with pytest.raises(ValueError, match="source.*(identity|changed|match)|content"):
        _mapped(b, source=source)


@pytest.mark.parametrize("mesh,shift", [((2, 3, 1), (0, 0, 0)), ((2, 2, 1), (1, 0, 0))])
def test_mapped_support_rejects_whole_mesh_incompatibility(mesh, shift):
    b = _mapped_bundle(mesh=mesh, shift=shift)
    # Swapping x,y can map the requested origin, yet fail another mesh point
    # or shift parity. Admission must inspect the exact whole-mesh action.
    with pytest.raises(ValueError, match="whole.mesh"):
        _mapped(b, op=_operation([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), plan=True)


@pytest.mark.parametrize("cap_name,plan_name", [
    ("maximum_per_replica_inventoried_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_inventoried_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_mapped_support_exact_enclosing_caps_precede_payload_validation(cap_name, plan_name):
    b = _mapped_bundle()
    source = _source(b)
    controls = _mapped_controls()
    controls[2].numerical_replicas = 3
    controls[2].external_node_bytes = 8192
    plan = _mapped(b, controls=controls, plan=True)
    assert plan.required_node_inventoried_bytes == 8192 + 3 * plan.per_replica_inventoried_bytes
    assert plan.per_replica_inventoried_bytes == (plan.fixed_workspace_bytes
        + plan.support.inventoried_bytes + plan.transport.per_replica_inventoried_bytes)
    setattr(controls[3], cap_name, getattr(plan, plan_name))
    _mapped(b, controls=controls, source=source)
    setattr(controls[3], cap_name, getattr(plan, plan_name) - 1)
    b.cells[0, 0] = 2**53 + 1
    with pytest.raises((ValueError, RuntimeError), match="BIPOLE mapped support enclosing"):
        _mapped(b, controls=controls, source=source)


def test_mapped_support_receipt_binds_geometry_selection_and_tolerances_not_resources():
    b = _mapped_bundle()
    source = _source(b)
    controls = _mapped_controls()
    base = _mapped(b, controls=controls, source=source)
    controls[2].backend_margin_bytes_per_replica *= 2
    controls[3].maximum_work_units -= 1
    assert _mapped(b, controls=controls).mapping_identity_sha256 == base.mapping_identity_sha256
    controls[1].maximum_atom_mapping_residual_bohr *= 2
    assert _mapped(b, controls=controls).mapping_identity_sha256 != base.mapping_identity_sha256
    assert _mapped(b, controls=_mapped_controls(tr=True)).mapping_identity_sha256 != base.mapping_identity_sha256
    assert _mapped(b, controls=_mapped_controls(source_k=1)).mapping_identity_sha256 != base.mapping_identity_sha256
    perturbed = _mapped(b, op=_operation(np.diag([-1, -1, 1]), [0, 0, .5 + 1e-11]))
    assert perturbed.support.action_identity_sha256 == base.support.action_identity_sha256
    assert perturbed.mapping_identity_sha256 != base.mapping_identity_sha256
    # Species are absent from the two-electron source hash, but present in the
    # geometry mapping receipt. Both species-preserving maps remain valid.
    b.system = core.PeriodicSystem(3, b.system.lattice, [core.Atom(3, a.xyz) for a in b.system.unit_cell])
    changed = _mapped(b, source=source)
    assert changed.support.action_identity_sha256 == base.support.action_identity_sha256
    assert changed.mapping_identity_sha256 != base.mapping_identity_sha256
    copied = base.atom_shifts
    copied[0] = 99
    assert base.atom_shifts[0] == 0
    del source, b
    gc.collect()
    assert base.destination_atoms == [1, 0, 1, 0]


@pytest.mark.parametrize("shells", [(2, 0, 0, 0), (0, 0, 0, 2**64 - 1)])
def test_mapped_support_rejects_out_of_range_selected_shell(shells):
    with pytest.raises(IndexError, match="selected shell"):
        _mapped(_mapped_bundle(), controls=_mapped_controls(shells))


def test_mapped_support_empty_integral_support_still_validates_complete_geometry():
    b = _mapped_bundle()
    b.cells = np.empty((0, 3), dtype=np.int64)
    b.images = np.empty((0, 3, 3), dtype=np.int64)
    result = _mapped(b)
    assert result.support.left_cells_closed and result.support.right_cells_closed and result.support.images_closed
    assert result.transport.mapped_atoms == 2
    assert not result.symmetry_certified
    with pytest.raises(ValueError, match="atom mapping"):
        _mapped(b, op=_operation(np.diag([-1, -1, 1]), [0, 0, .25]))


@pytest.mark.parametrize("tr", [False, True])
def test_validated_screw_maps_feed_shared_semilinear_transport_without_support_admission(tr):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity

    b = _mapped_bundle(shift=(0, 0, 1))
    source = _source(b)
    op = _operation(np.diag([-1, -1, 1]), [0, 0, .5])
    mapped = _mapped(b, source=source, op=op, controls=_mapped_controls(source_k=0, tr=tr))
    target = mapped.memory.transport.target_index
    source_space = SpaceIdentity(mapped.mapping_identity_sha256, source.basis_identity_sha256,
                                 "AO", "native-Bloch-k-0")
    target_space = SpaceIdentity(mapped.mapping_identity_sha256, source.basis_identity_sha256,
                                 "AO", f"native-Bloch-k-{target}")
    # These are s shells, so each local rotation is scalar. The full Seitz
    # translation is already in ell; no additional global phase is applied.
    ell = np.asarray(mapped.atom_shifts[:6], dtype=np.int64).reshape(2, 3)
    phases = np.ascontiguousarray(np.exp(2j * np.pi * (ell @ b.mesh.fractional_at(target))))
    assert abs(phases[1].imag) > .8  # exercise complex antiunitary phases
    action = BlockSpaceAction(source_space, target_space, np.array([0, 1, 2], np.int64),
        np.array(mapped.destination_shells[:2], np.int64), phases,
        antiunitary=tr, budget=Budget(16 << 20, 10**8))
    rng = np.random.default_rng(851)
    coefficients = np.ascontiguousarray(rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)))
    native = core._apply_periodic_ao_bloch_operation(b.basis, b.system, op, b.mesh, 0, tr,
        coefficients, *_transport_controls())
    actual = action.apply(coefficients)
    np.testing.assert_allclose(actual, native.coefficients_copy(), atol=1e-12, rtol=1e-12)
    probe = np.ascontiguousarray(rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)))
    assert np.vdot(probe, actual).real == pytest.approx(
        np.vdot(action.adjoint(probe), coefficients).real, abs=2e-12)
    # Correct coefficient transport does not make the finite operator
    # covariant: this common-window mixed-site product still fails closure.
    assert not mapped.support.left_cells_closed
    assert not mapped.symmetry_certified and not mapped.physical_hamiltonian_certified


@pytest.mark.parametrize("mesh,shift", [((1, 1, 1), (0, 0, 0)),
                                       ((3, 1, 1), (1, 0, 0)),
                                       ((2, 2, 2), (1, 1, 1))])
def test_support_identity_and_common_cell_relabelling(mesh, shift):
    b = _bundle(mesh=mesh, shift=shift, kl=0, kr=0)
    result = _support(b)
    translated = _support(b, shifts=np.tile([2**53, -2**53, 17], (4, 1)).astype(np.int64))
    for audit in (result, translated):
        assert audit.left_cells_closed and audit.right_cells_closed and audit.images_closed
        assert audit.label_comparisons == 12 * len(b.cells)**2 + 18 * len(b.images)**2
        assert audit.label_comparisons == audit.memory.label_comparisons
        assert not audit.symmetry_certified and not audit.physical_hamiltonian_certified
    assert result.source_identity_sha256 == translated.source_identity_sha256
    # Receipts preserve declared metadata, even if its common shift cancels.
    assert result.action_identity_sha256 != translated.action_identity_sha256


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
@pytest.mark.parametrize("q", [0, 1])
def test_support_closed_inversion_matches_independent_four_index_integrals(shift, q):
    positions = [(-0.3, -0.2, 0.1), (0.3, 0.2, -0.1)]
    basis = _basis([(0, p, [0.8], [0.7], True) for p in positions])
    b = _bundle(q=q, kl=1, kr=2, shift=shift, basis=basis, images=[[[0, 0, 0]] * 3])
    b.system = core.PeriodicSystem(3, np.diag([4.0, 5.0, 6.0]), [core.Atom(2, p) for p in positions])
    audit = _support(b, -np.eye(3, dtype=np.int32))
    assert audit.left_cells_closed and audit.right_cells_closed and audit.images_closed
    for quartet in itertools.product(range(2), repeat=4):
        mapped = _mapped(b, op=_operation(-np.eye(3)), controls=_mapped_controls(quartet, source_k=1))
        assert mapped.destination_shells == [1 - s for s in quartet]
        assert mapped.support.left_cells_closed and mapped.support.right_cells_closed and mapped.support.images_closed
        assert mapped.memory.transport.target_index == (-1 - shift[0]) % 3
    before = _panel(b).values_copy()
    sr, lr, zero = _components(b)
    np.testing.assert_allclose(before, sr + lr - zero, atol=1e-12, rtol=1e-12)
    b.selection.q_index = (-q) % 3  # transfer mesh has no shift
    b.selection.left_k_index = (-1 - shift[0]) % 3
    b.selection.right_k_index = (-2 - shift[0]) % 3
    after = _panel(b).values_copy()
    # Inversion exchanges the two equal radial s shells, without an atom-cell
    # shift. Compare all four AO legs, independently of native AO transport.
    pair_permutation = [3, 2, 1, 0]
    np.testing.assert_allclose(after[np.ix_(pair_permutation, pair_permutation)],
                               before, atol=1e-12, rtol=1e-12)
    assert not audit.symmetry_certified  # a witness is not full group/HF admission


def test_support_closed_selected_screw_quartet_does_not_admit_other_pairs():
    lattice = np.diag([4.0, 5.0, 3.0])
    positions = [(0.2, 0.4, 0.0), (-0.2, -0.4, 1.5)]
    basis = _basis([(0, p, [0.2], [(0.4 / np.pi)**0.75], True) for p in positions])
    b = _bundle(q=1, mesh=(1, 1, 4), kl=1, kr=2, basis=basis,
                cells=[[0, 0, 0], [0, 0, 1], [0, 0, -1]], images=[[[0, 0, 0]] * 3])
    b.system = core.PeriodicSystem(3, lattice, [core.Atom(2, p) for p in positions])
    w = np.diag([-1, -1, 1]).astype(np.int32)
    # (0,0|0,0) maps to (1,1|1,1), all four ell values are zero.
    audit = _support(b, w)
    assert audit.left_cells_closed and audit.right_cells_closed and audit.images_closed
    mapped = _mapped(b, controls=_mapped_controls((0, 0, 0, 0), source_k=1))
    assert mapped.destination_shells == [1] * 4
    assert mapped.support.left_cells_closed and mapped.support.right_cells_closed and mapped.support.images_closed
    panel = _panel(b).values_copy()
    sr, lr, zero = _components(b)
    for component in (panel, sr, lr, zero):
        np.testing.assert_allclose(component[0, 0], component[3, 3], atol=1e-12, rtol=1e-12)
    # The same operator's off-diagonal product has unequal ell and fails.
    shifts = np.array([[0, 0, 0], [0, 0, -1], [0, 0, 0], [0, 0, -1]], np.int64)
    off_diagonal = _support(b, w, shifts)
    assert not off_diagonal.left_cells_closed and not off_diagonal.right_cells_closed
    assert not audit.symmetry_certified and not off_diagonal.symmetry_certified


@pytest.mark.parametrize("seed", range(8))
def test_support_action_matches_independent_integer_multisets(seed):
    rng = np.random.default_rng(814 + seed)
    w = np.eye(3, dtype=np.int32)[rng.permutation(3)] * rng.choice([-1, 1], size=(3, 1)).astype(np.int32)
    shifts = rng.integers(-1, 2, size=(4, 3), dtype=np.int64)
    b = _bundle(images=rng.integers(-1, 2, size=(6, 3, 3), dtype=np.int64))
    expected = _support_oracle(b, w, shifts)
    result = _support(b, w, shifts)
    assert (result.left_cells_closed, result.right_cells_closed, result.images_closed) == expected


def test_support_nonzero_relative_shifts_can_close_a_selected_affine_orbit():
    b = _bundle(cells=[[0, 0, 0], [1, 0, 0]],
                images=[[[0, 0, 0]] * 3, [[1, 0, 0], [0, 0, 0], [1, 0, 0]]])
    b.options.require_cell_inversion_closure = False
    b.options.require_image_permutation_closure = False
    w = -np.eye(3, dtype=np.int32)
    shifts = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]], np.int64)
    assert _support_oracle(b, w, shifts) == (True, True, True)
    result = _support(b, w, shifts)
    assert result.left_cells_closed and result.right_cells_closed and result.images_closed
    # Rotating the labels without the required reanchoring would reject it.
    wrong = _support(b, w)
    assert not wrong.left_cells_closed and not wrong.right_cells_closed and not wrong.images_closed


@pytest.mark.parametrize("duplicate", [False, True])
def test_support_image_multiplicity_is_not_set_membership(duplicate):
    image = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]], np.int64)
    images = np.concatenate([image, -image, image] if duplicate else [image, -image])
    b = _bundle(images=images)
    result = _support(b, -np.eye(3, dtype=np.int32))
    assert result.left_cells_closed and result.right_cells_closed
    assert result.images_closed is (not duplicate)
    if duplicate:
        assert tuple(result.first_image) == tuple((-image).flat)


def test_support_left_right_and_quartet_anchors_are_distinct():
    b = _bundle(cells=[[0, 0, 0]], images=[[[0, 0, 0]] * 3])
    # Each product is merely translated as a whole. Pair supports pass;
    # the displacement between the products changes the quartet support.
    shifts = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]], np.int64)
    result = _support(b, shifts=shifts)
    assert result.left_cells_closed and result.right_cells_closed
    assert not result.images_closed
    assert tuple(result.first_image) == (0, 0, 0, 0, -1, 0, 0, -1, 0)
    # Changing only the fourth center now breaks the right product too.
    shifts[3, 2] = 1
    result = _support(b, shifts=shifts)
    assert result.left_cells_closed and not result.right_cells_closed
    assert tuple(result.first_right_cell) == (0, 0, -1)


def test_support_never_wraps_finite_cells_modulo_kmesh():
    b = _bundle(mesh=(3, 1, 1), cells=[[0, 0, 0]], images=[[[0, 0, 0]] * 3])
    shifts = np.zeros((4, 3), np.int64)
    shifts[1, 0] = 3
    result = _support(b, shifts=shifts)
    assert not result.left_cells_closed
    assert tuple(result.first_left_cell) == (-3, 0, 0)


@pytest.mark.parametrize("radius", [1, 2, 4, 8])
def test_support_widening_common_window_cannot_make_screw_exact(radius):
    b = _bundle(cells=[[0, 0, z] for z in range(-radius, radius + 1)], images=[])
    shifts = np.array([[0, 0, 0], [0, 0, -1], [0, 0, 0], [0, 0, -1]], np.int64)
    result = _support(b, np.diag([-1, -1, 1]).astype(np.int32), shifts)
    assert not result.left_cells_closed and not result.right_cells_closed
    assert tuple(result.first_left_cell) == (0, 0, radius + 1)


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_label_comparisons", "label_comparisons"),
    ("maximum_inventoried_bytes", "inventoried_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_support_exact_resource_limits_precede_bad_payload(field, plan_field):
    b = _bundle()
    source = _source(b)
    plan = _support(b, source=source, plan=True)
    caps = _support_caps()
    setattr(caps, field, getattr(plan, plan_field))
    _support(b, source=source, caps=caps)
    setattr(caps, field, getattr(plan, plan_field) - 1)
    b.cells[0, 0] = 2**53 + 1
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _support(b, source=source, caps=caps)


def test_support_changed_borrowed_content_is_refused_and_result_is_detached():
    b = _bundle()
    source = _source(b)
    shifts = np.zeros((4, 3), np.int64)
    result = _support(b, source=source, shifts=shifts)
    receipt = result.action_identity_sha256
    shifts[1, 0] = 1
    assert result.action_identity_sha256 == receipt
    assert result.left_cells_closed
    b.images[0, 0, 0] += 1
    with pytest.raises(ValueError, match="immutable source"):
        _support(b, source=source)


def test_support_resource_changes_preserve_action_receipt():
    b = _bundle()
    source = _source(b)
    first = _support(b, source=source)
    caps = _support_caps()
    caps.maximum_label_comparisons *= 2
    caps.maximum_inventoried_bytes *= 2
    caps.maximum_work_units *= 2
    second = _support(b, source=source, caps=caps)
    assert first.action_identity_sha256 == second.action_identity_sha256
    assert first.source_identity_sha256 == source.source_identity_sha256


@pytest.mark.parametrize("field", ["maximum_label_comparisons", "maximum_inventoried_bytes", "maximum_work_units"])
def test_support_requires_explicit_resource_caps_even_for_empty_support(field):
    b = _bundle(images=[], cells=[])
    caps = _support_caps()
    setattr(caps, field, 0)
    with pytest.raises(ValueError, match="positive explicit caps"):
        _support(b, caps=caps)


@pytest.mark.parametrize("invalid", ["singular", "nonunimodular", "shift", "overflow"])
def test_support_rejects_malformed_or_overflowing_actions(invalid):
    b = _bundle(cells=[[0, 0, 0], [2**53, 2**53, 0]], images=[])
    w = np.eye(3, dtype=np.int32)
    shifts = np.zeros((4, 3), np.int64)
    if invalid == "singular":
        w[1] = w[0]
    elif invalid == "nonunimodular":
        w[0, 0] = 2
    elif invalid == "shift":
        shifts[3, 2] = np.iinfo(np.int64).min
    else:
        w[0, 1] = np.iinfo(np.int32).max  # unimodular shear, overflowing image
    with pytest.raises((ValueError, OverflowError), match="unimodular|shifts|int64"):
        _support(b, w, shifts)


def test_support_exact_large_cancellation_and_empty_supports():
    b = _bundle(cells=[[2**53, 2**53, 0]], images=[])
    # determinant +1, huge intermediate products, exact result (x,x,0).
    n = 2**30
    w = np.array([[n, 1-n, 0], [n-1, 2-n, 0], [0, 0, 1]], np.int32)
    result = _support(b, w)
    assert result.left_cells_closed and result.right_cells_closed and result.images_closed
    b = _bundle(cells=[], images=[])
    result = _support(b, shifts=np.arange(12, dtype=np.int64).reshape(4, 3))
    assert result.left_cells_closed and result.right_cells_closed and result.images_closed
    assert result.label_comparisons == 0
    assert not result.symmetry_certified


@pytest.mark.parametrize("bad", ["rotation_float", "rotation_int64", "shifts_float", "strided", "shape"])
def test_support_binding_does_not_coerce_mapping_metadata(bad):
    b = _bundle()
    w = np.eye(3, dtype=np.int32)
    shifts = np.zeros((4, 3), np.int64)
    if bad == "rotation_float":
        w = w.astype(float)
    elif bad == "rotation_int64":
        w = w.astype(np.int64)
    elif bad == "shifts_float":
        shifts = shifts.astype(float)
    elif bad == "strided":
        shifts = np.zeros((4, 6), np.int64)[:, ::2]
    else:
        shifts = shifts.ravel()
    with pytest.raises((TypeError, ValueError), match="incompatible|requires"):
        _support(b, w, shifts)


def test_support_fixed_mapping_copy_handles_unaligned_numpy_buffers():
    b = _bundle()
    w = np.ndarray((3, 3), dtype=np.int32, buffer=bytearray(37), offset=1)
    shifts = np.ndarray((4, 3), dtype=np.int64, buffer=bytearray(97), offset=1)
    w[:] = np.eye(3, dtype=np.int32)
    shifts[:] = 0
    assert not w.flags.aligned and not shifts.flags.aligned
    ordinary = _support(b)
    unaligned = _support(b, w, shifts)
    assert unaligned.action_identity_sha256 == ordinary.action_identity_sha256
    assert unaligned.left_cells_closed and unaligned.right_cells_closed and unaligned.images_closed


@pytest.mark.parametrize("omega", [1e-300, 1e300])
def test_unrepresentable_zero_coefficient_is_refused(omega):
    b = _bundle()
    b.options.omega = omega
    with pytest.raises((ValueError, OverflowError), match="coefficient|finite|zero.mode|representable"):
        _source(b)


def test_zero_mode_convention_must_be_explicit():
    b = _bundle()
    b.options.zero_mode = core._BipoleFiniteZeroMode.Unspecified
    with pytest.raises(ValueError, match="zero.mode|G0|explicit|convention"):
        _source(b)


@pytest.mark.parametrize("q", [0, 1])
@pytest.mark.parametrize("empty", ["images", "cells", "both_supports", "left", "right"])
def test_composed_empty_supports_and_selected_intervals(q, empty):
    b = _bundle(q=q)
    if empty in {"images", "both_supports"}:
        b.images = np.empty((0, 3, 3), dtype=np.int64)
    if empty in {"cells", "both_supports"}:
        b.cells = np.empty((0, 3), dtype=np.int64)
    if empty == "left":
        b.selection.left_pair_count = 0
    if empty == "right":
        b.selection.right_pair_count = 0
    result = _panel(b)
    actual = result.values_copy()
    assert actual.shape == (b.selection.left_pair_count, b.selection.right_pair_count)
    if empty in {"left", "right"}:
        assert actual.size == 0
    elif empty == "both_supports":
        np.testing.assert_array_equal(actual, np.zeros((4, 4)))
    else:
        sr, lr, zero = _components(b)
        np.testing.assert_allclose(actual, sr + lr - zero, atol=6e-13, rtol=5e-12)
    assert len(result.payload_identity_sha256) == 64
    assert not result.physical_hamiltonian_certified


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_cells", "cell_count"),
    ("maximum_pair_cell_visits", "pair_cell_visits"),
    ("maximum_output_bytes", "output_bytes"),
    ("maximum_work_units", "work_units"),
])
def test_zero_mode_child_caps_precede_invalid_source_payload(field, plan_field):
    b = _bundle(q=0)
    source = _source(b)
    plan = _panel(b, source, plan=True)
    amount = max(getattr(plan.zero_left, plan_field), getattr(plan.zero_right, plan_field))
    setattr(b.caps.zero_mode, field, amount)
    _panel(b, source)
    b.images[0, 0, 0] = np.iinfo(np.int64).min
    setattr(b.caps.zero_mode, field, amount - 1)
    with pytest.raises(ValueError, match="cap|work|storage|bytes|limit"):
        _panel(b, source)


def test_nonzero_transfer_does_not_require_zero_mode_child_resources():
    b = _bundle(q=1)
    source = _source(b)
    expected = _panel(b, source).values_copy()
    b.caps.zero_mode = core._AOPairFourierCellPanelCaps()
    np.testing.assert_array_equal(_panel(b, source).values_copy(), expected)
    b.selection.q_index = 0
    with pytest.raises(ValueError, match="positive|cap|work"):
        _panel(b, source)


def test_outer_census_includes_replicas_and_zero_mode_lifetime():
    b = _bundle()
    source = _source(b)
    b.inventory.numerical_replicas = 2
    b.inventory.external_node_bytes = 12345
    b.inventory.other_live_numerical_bytes_per_replica = 4321
    b.inventory.other_live_control_bytes_per_replica = 1234
    plan = _panel(b, source, plan=True)
    assert plan.required_node_inventoried_bytes == 2 * plan.per_replica_inventoried_bytes + 12345
    assert plan.zero_left.n_vectors == plan.zero_right.n_vectors == 1
    assert plan.zero_mode_workspace_bytes >= plan.zero_left.output_bytes + plan.zero_right.output_bytes
    assert plan.peak_owned_numerical_bytes >= plan.retained_output_bytes + plan.compensation_bytes
    b.selection.q_index = 1
    nongamma = _panel(b, source, plan=True)
    assert not nongamma.subtract_zero_mode
    assert nongamma.zero_mode_workspace_bytes == 0


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
def test_complete_hermitian_density_response_matches_independent_common_integrals(shift):
    """Twelve independent real density directions, not a symmetric fixed point.

    One common image support is used for J and crossed K. Separate finite
    O/C/D supports from the legacy HF builder would not prove this identity.
    This certifies only the tiny declared finite numerical response.
    """
    b = _bundle(shift=shift)
    source = _source(b)
    nk, nao = 3, 2
    raw = _erfc_oracle(b.basis, b.images, lattice=_GRAM_LATTICE, omega=b.options.omega)
    native, expected, zero_parts = ({}, {}), ({}, {}), ({}, {})
    for target, origin in itertools.product(range(nk), repeat=2):
        for kind in range(2):
            b.selection.q_index = 0 if kind == 0 else (target - origin) % nk
            b.selection.left_k_index = target if kind == 0 else origin
            b.selection.right_k_index = origin
            sr, lr, zero = _components(b, raw=raw)
            native[kind][target, origin] = _panel(b, source).values_copy().reshape((nao,) * 4)
            expected[kind][target, origin] = (sr + lr - zero).reshape((nao,) * 4)
            zero_parts[kind][target, origin] = zero.reshape((nao,) * 4)
    elementary = [np.diag([1, 0]), np.diag([0, 1]),
                  np.array([[0, 1], [1, 0]]), np.array([[0, 1j], [-1j, 0]])]
    densities = []
    for origin, value in itertools.product(range(nk), elementary):
        density = np.zeros((nk, nao, nao), complex)
        density[origin] = value
        densities.append(density)
    reciprocal = 2 * np.pi * np.linalg.inv(_GRAM_LATTICE).T
    overlaps = np.array([_cell_oracle(b.basis, np.zeros((1, 3)), _GRAM_LATTICE,
                                      reciprocal @ b.mesh.fractional_at(k), b.cells)[:, 0].reshape(nao, nao)
                         for k in range(nk)])
    coefficient = np.pi / (abs(np.linalg.det(_GRAM_LATTICE)) * b.options.omega**2)
    responses = [[], []]
    for density in densities:
        charge = np.einsum("kmn,knm->", density, overlaps) / nk
        for kind, axes in enumerate(("mnls,ls->mn", "mlns,ls->mn")):
            def contract(kernels):
                return np.array([sum(np.einsum(axes, kernels[target, origin], density[origin])
                                     for origin in range(nk)) / nk for target in range(nk)])
            response = contract(native[kind])
            np.testing.assert_allclose(response, contract(expected[kind]), atol=8e-13, rtol=6e-12)
            np.testing.assert_allclose(response, response.conj().transpose(0, 2, 1), atol=8e-13, rtol=6e-12)
            analytic_zero = coefficient * charge * overlaps if kind == 0 else np.array([
                coefficient / nk * overlaps[k] @ density[k] @ overlaps[k] for k in range(nk)
            ])
            np.testing.assert_allclose(contract(zero_parts[kind]), analytic_zero, atol=8e-13, rtol=6e-12)
            responses[kind].append(response)
    for group in responses:
        bilinear = np.array([[np.einsum("kmn,knm->", density, response) / nk
                              for response in group] for density in densities])
        np.testing.assert_allclose(bilinear.imag, 0, atol=1e-12)
        np.testing.assert_allclose(bilinear.real, bilinear.real.T, atol=1e-12, rtol=6e-12)

# Native full-grid density action; no SCF is run by these witnesses.
def _jk_bundle(**kwargs):
    b = _bundle(**kwargs)
    b.jk_selection = core._BipoleFiniteJKSelection()
    b.jk_selection.target_k_index = min(1, len(b.mesh) - 1)
    b.jk_selection.pair_count = b.basis.nbasis**2
    b.jk_selection.density_pair_block_size = 4
    b.jk_caps = core._BipoleFiniteJKCaps()
    b.jk_caps.panel = b.caps
    b.jk_caps.maximum_panel_calls = 32768
    b.jk_caps.maximum_work_units = 10**11
    return b


def _jk(b, density, *, source=None, plan=False):
    fn = core._plan_bipole_finite_jk if plan else core._make_bipole_finite_jk
    return fn(_source(b) if source is None else source, b.basis, b.images, b.cells,
              density, b.jk_selection, b.inventory, b.jk_caps)


def _jk_reference_data(b):
    """Independent quadrature quartets and polynomial Fourier integrals."""
    lattice = np.asarray(b.system.lattice)
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    n = b.basis.nbasis
    nk = len(b.mesh)
    raw = _erfc_oracle(b.basis, b.images, lattice=lattice,
                       left=(0, n*n), right=(0, n*n), omega=b.options.omega)
    kcart = np.array([reciprocal @ b.mesh.fractional_at(k) for k in range(nk)])
    records, fourier = [], []
    for q in range(nk):
        b.selection.q_index = q
        vectors, weights = _gram_records(b)
        records.append(weights)
        fourier.append([_cell_oracle(b.basis, vectors, lattice, -k, b.cells)
                        .reshape(n, n, len(vectors)) for k in kcart])
    overlap_minus = np.array([
        _cell_oracle(b.basis, np.zeros((1, 3)), lattice, -k, b.cells).reshape(n, n)
        for k in kcart
    ])
    return raw.reshape(len(b.images), n, n, n, n), kcart, records, fourier, overlap_minus


def _jk_reference(b, density, data):
    """Contract complex real-space density before Bloch folding; no native ERI."""
    raw, kcart, weights, fourier, overlap_minus = data
    n, nk = b.basis.nbasis, len(b.mesh)
    target = b.jk_selection.target_k_index
    expected = np.zeros((2, n, n), complex)
    lattice = np.asarray(b.system.lattice)
    for labels, eri in zip(b.images, raw):
        g, p, s = labels @ lattice.T
        dj = np.einsum("k,kcd->cd", np.exp(-1j * (kcart @ (s-p))), density) / nk
        dk = np.einsum("k,kcd->cd", np.exp(-1j * (kcart @ (s-g))), density) / nk
        expected[0] += np.exp(1j * np.dot(kcart[target], g)) * np.einsum("abcd,cd->ab", eri, dj)
        expected[1] += np.exp(1j * np.dot(kcart[target], p)) * np.einsum("acbd,cd->ab", eri, dk)
    # Each reciprocal term is an AO matrix congruence for exchange. Coulomb
    # uses an independently contracted density Fourier scalar.
    for k in range(nk):
        expected[0] += np.einsum("abv,cdv,cd,v->ab", fourier[0][target].conj(),
                                 fourier[0][k], density[k], weights[0]) / nk
        # Independent integer divisions, not the native transfer accessor.
        dims = tuple(b.mesh.mesh)
        delta = (np.array(np.unravel_index(target, dims))
                 - np.array(np.unravel_index(k, dims))) % dims
        q = np.ravel_multi_index(tuple(delta), dims)
        bm = fourier[q][k]
        expected[1] += np.einsum("acv,cd,bdv,v->ab", bm.conj(), density[k], bm, weights[q]) / nk
    alpha = np.pi / (abs(np.linalg.det(lattice)) * b.options.omega**2)
    charge = np.einsum("kcd,kcd->", overlap_minus, density) / nk
    expected[0] -= alpha * overlap_minus[target].conj() * charge
    expected[1] -= alpha / nk * (overlap_minus[target].conj() @ density[target]
                                @ overlap_minus[target].T)
    begin = b.jk_selection.pair_begin
    return expected.reshape(2, n*n)[:, begin:begin+b.jk_selection.pair_count]


@pytest.fixture(scope="module", params=[(0, 0, 0), (1, 0, 0)])
def jk_witness(request):
    b = _jk_bundle(shift=request.param)
    return b, _source(b), _jk_reference_data(b)


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
@pytest.mark.parametrize("inversion,tr", [(True, False), (False, True), (True, True)])
def test_geometry_mapped_finite_jk_covariance_at_arbitrary_complex_full_mesh_density(shift, inversion, tr):
    """Match full linear J/K actions, including antiunitarity; no SCF repair.

    This witnesses only the explicitly finite source. The production BIPOLE
    HF support/probe-charge convention remains an independent matching task.
    """
    positions = [(-.3, -.2, .1), (.3, .2, -.1)]
    basis = _basis([(0, p, [.8], [.7], True) for p in positions])
    b = _jk_bundle(basis=basis, shift=shift, images=[[[0, 0, 0]] * 3])
    b.system = core.PeriodicSystem(3, np.diag([4., 5., 6.]), [core.Atom(2, p) for p in positions])
    source = _source(b)
    op = _operation((-1 if inversion else 1) * np.eye(3))
    permutation = [1, 0] if inversion else [0, 1]
    for quartet in itertools.product(range(2), repeat=4):
        mapped = _mapped(b, source=source, op=op, controls=_mapped_controls(quartet, tr=tr))
        assert mapped.destination_shells == [permutation[s] for s in quartet]
        assert mapped.support.left_cells_closed and mapped.support.right_cells_closed and mapped.support.images_closed
    rng = np.random.default_rng(23792)
    density = np.ascontiguousarray(rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2)))
    transformed = np.empty_like(density)
    kmap = [(-k - shift[0]) % 3 if inversion != tr else k for k in range(3)]
    for k, target in enumerate(kmap):
        mapped = _mapped(b, source=source, op=op, controls=_mapped_controls(source_k=k, tr=tr))
        assert mapped.memory.transport.target_index == target
        value = density[k].conj() if tr else density[k]
        transformed[target] = value[np.ix_(permutation, permutation)]
    # The analytical k/atom permutations above are independent of the native
    # AO action, and both density actions also face an independent integral oracle.
    oracle = _jk_reference_data(b)
    before, after = [], []
    for k in range(3):
        b.jk_selection.target_k_index = k
        first = _jk(b, density, source=source).values_copy()
        second = _jk(b, transformed, source=source).values_copy()
        np.testing.assert_allclose(first, _jk_reference(b, density, oracle), atol=2e-12, rtol=8e-12)
        np.testing.assert_allclose(second, _jk_reference(b, transformed, oracle), atol=2e-12, rtol=8e-12)
        before.append(first.reshape(2, 2, 2))
        after.append(second.reshape(2, 2, 2))
    for k, target in enumerate(kmap):
        expected = before[k].conj() if tr else before[k]
        expected = expected[:, permutation, :][:, :, permutation]
        np.testing.assert_allclose(after[target], expected, atol=2e-12, rtol=8e-12)


@pytest.mark.parametrize("direction", range(24))
def test_jk_matches_all_complex_full_grid_density_directions(jk_witness, direction):
    # 24 real directions span ALL complex3x2x2 matrices, not only the
    # seven-dimensional time-reversal/Hermitian subspace of earlier SR tests.
    b, source, data = jk_witness
    density = np.zeros((3, 2, 2), np.complex128)
    density.reshape(-1)[direction // 2] = 1j if direction % 2 else 1
    actual = _jk(b, density, source=source)
    np.testing.assert_allclose(actual.values_copy(), _jk_reference(b, density, data),
                               atol=8e-13, rtol=8e-12)
    assert actual.citation_numerics == ["bipole_finite_jk"]
    assert not actual.physical_hamiltonian_certified and not actual.symmetry_certified
    assert actual.source.source_identity_sha256 == source.source_identity_sha256


@pytest.mark.parametrize("target", [0, 2])
def test_jk_complex_nontr_density_at_other_target_k(target):
    b = _jk_bundle()
    b.jk_selection.target_k_index = target
    rng = np.random.default_rng(9047)
    density = np.ascontiguousarray(rng.normal(size=(3, 2, 2)) + 1j*rng.normal(size=(3, 2, 2)))
    actual = _jk(b, density).values_copy()
    np.testing.assert_allclose(actual, _jk_reference(b, density, _jk_reference_data(b)),
                               atol=8e-13, rtol=8e-12)
    assert np.max(np.abs(actual.imag)) > 1e-3  # No real-density coercion.


@pytest.mark.parametrize("block", [1, 2, 3, 7])
def test_jk_tiling_preserves_values_and_scientific_identity(block):
    b = _jk_bundle()
    density = np.ascontiguousarray(np.arange(12).reshape(3, 2, 2) * (0.07 + 0.03j))
    first = _jk(b, density)
    b.jk_selection.density_pair_block_size = block
    b.inventory.reciprocal_block_size = 1
    second = _jk(b, density)
    np.testing.assert_array_equal(second.values_copy(), first.values_copy())
    assert second.input_identity_sha256 == first.input_identity_sha256
    assert second.payload_identity_sha256 == first.payload_identity_sha256
    b.jk_selection.pair_begin, b.jk_selection.pair_count = 1, 2
    piece = _jk(b, density)
    np.testing.assert_array_equal(piece.values_copy(), first.values_copy()[:, 1:3])


def test_jk_full_mesh_replication_has_exactly_one_mesh_weight():
    # Same finite source at the same physical k values, density zero elsewhere.
    b = _jk_bundle()
    density = np.ascontiguousarray(np.arange(12).reshape(3, 2, 2) * (0.07 + 0.03j))
    first = _jk(b, density).values_copy()
    refined = _jk_bundle(mesh=(6, 1, 1))
    refined.jk_selection.target_k_index = 2
    sparse = np.zeros((6, 2, 2), np.complex128)
    sparse[::2] = 2 * density
    np.testing.assert_allclose(_jk(refined, sparse).values_copy(), first, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("mesh,shift", [((1, 1, 1), (0, 0, 0)),
                                       ((2, 2, 1), (1, 1, 0))])
def test_jk_gamma_and_multiaxis_shifted_mesh(mesh, shift):
    b = _jk_bundle(mesh=mesh, shift=shift)
    rng = np.random.default_rng(9371)
    shape = (len(b.mesh), 2, 2)
    density = np.ascontiguousarray(rng.normal(size=shape) + 1j*rng.normal(size=shape))
    np.testing.assert_allclose(_jk(b, density).values_copy(),
                               _jk_reference(b, density, _jk_reference_data(b)),
                               atol=8e-13, rtol=8e-12)


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_borrowed_numerical_bytes", "borrowed_numerical_bytes"),
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_control_storage_bytes", "control_storage_bytes"),
    ("maximum_worker_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_bytes", "required_node_inventoried_bytes"),
])
def test_jk_enclosing_memory_exact_and_minus_one_before_nan(field, plan_field):
    b = _jk_bundle()
    density = np.zeros((3, 2, 2), np.complex128)
    p = _jk(b, density, plan=True)
    caps = b.jk_caps.panel
    setattr(caps, field, getattr(p, plan_field))
    b.jk_caps.panel = caps
    _jk(b, density)
    setattr(caps, field, getattr(p, plan_field) - 1)
    b.jk_caps.panel = caps
    density[0, 0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _jk(b, density)


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_panel_calls", "panel_calls"), ("maximum_work_units", "work_units_upper_bound"),
])
def test_jk_work_exact_and_minus_one_before_nan(field, plan_field):
    b = _jk_bundle()
    density = np.zeros((3, 2, 2), np.complex128)
    p = _jk(b, density, plan=True)
    setattr(b.jk_caps, field, getattr(p, plan_field))
    _jk(b, density)
    setattr(b.jk_caps, field, getattr(p, plan_field) - 1)
    density[0, 0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _jk(b, density)


def test_jk_replica_and_external_memory_census():
    b = _jk_bundle()
    density = np.zeros((3, 2, 2), np.complex128)
    first = _jk(b, density, plan=True)
    assert first.density_bytes == density.nbytes
    assert first.output_elements == 8 and first.retained_output_bytes == 128
    assert first.panel_calls == 36 and first.contracted_terms == 96
    b.inventory.numerical_replicas = 2
    b.inventory.external_node_bytes = 1234
    b.inventory.other_live_numerical_bytes_per_replica = 2345
    b.inventory.other_live_control_bytes_per_replica = 3456
    second = _jk(b, density, plan=True)
    assert second.per_replica_inventoried_bytes == first.per_replica_inventoried_bytes + 2345 + 3456
    assert second.required_node_inventoried_bytes == 2 * second.per_replica_inventoried_bytes + 1234


@pytest.mark.parametrize("change", ["nan", "late_nan", "basis", "images", "cells"])
def test_jk_verifies_complete_payload_before_numerical_action(change):
    b = _jk_bundle()
    source = _source(b)
    density = np.zeros((3, 2, 2), np.complex128)
    if change in {"nan", "late_nan"}:
        density.reshape(-1)[0 if change == "nan" else -1] = np.nan
    elif change == "basis":
        b.basis = _basis([(0, (0, 0, 0), [0.3], [1.0], True),
                          (0, (0.4, 0, 0), [0.3], [1.0], True)])
    elif change == "images":
        b.images = b.images[::-1].copy()
    else:
        b.cells = b.cells[::-1].copy()
    # Count-only admission deliberately does not read these payloads.
    _jk(b, density, source=source, plan=True)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="match|nonfinite"):
        _jk(b, density, source=source)


@pytest.mark.parametrize("change", ["real", "strided", "shape", "unaligned"])
def test_jk_density_binding_does_not_coerce_or_copy(change):
    b = _jk_bundle()
    density = np.zeros((3, 2, 2), np.complex128)
    if change == "real":
        density = density.real.copy()
    elif change == "strided":
        density = density[:, :, ::-1]
    elif change == "unaligned":
        storage = np.zeros(density.nbytes + 1, dtype=np.uint8)
        density = np.ndarray((3, 2, 2), dtype=np.complex128, buffer=storage, offset=1)
    else:
        density = density.reshape(3, 4)
    with pytest.raises((TypeError, ValueError)):
        _jk(b, density)


@pytest.mark.parametrize("field,value", [
    ("pair_count", 0), ("pair_begin", 4), ("pair_count", 5),
    ("target_k_index", 3), ("density_pair_block_size", 0),
])
def test_jk_rejects_invalid_selection(field, value):
    b = _jk_bundle()
    setattr(b.jk_selection, field, value)
    with pytest.raises(ValueError, match="positive|outside"):
        _jk(b, np.zeros((3, 2, 2), np.complex128), plan=True)


def test_jk_identity_owner_and_copy_lifetimes():
    b = _jk_bundle()
    density = np.zeros((3, 2, 2), np.complex128)
    first = _jk(b, density)
    density[-1, -1, -1] = 1
    second = _jk(b, density)
    assert first.input_identity_sha256 != second.input_identity_sha256
    assert first.payload_identity_sha256 != second.payload_identity_sha256
    saved = second.values_copy()
    expected = saved.copy()
    owner = second.source
    del second, b
    gc.collect()
    np.testing.assert_array_equal(saved, expected)
    assert not owner.physical_hamiltonian_certified
    with pytest.raises(ValueError, match="read-only"):
        saved[0, 0] = 0
    with pytest.raises(IndexError):
        first.element(2, 0)


# Source-bound zero-mode overlap gate; the HF overlap uses the real-space
# libint one-electron path, independent of the Fourier source recurrence.
def _overlap_controls():
    caps = core._BipoleFiniteOverlapCaps()
    caps.source, caps.fourier = _source_caps(), _cell_caps()
    caps.maximum_per_replica_inventoried_bytes = 32 << 20
    caps.maximum_node_inventoried_bytes = 128 << 20
    caps.maximum_work_units = 10**9
    inventory = core._PeriodicAOBlochTransportInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 65536
    return inventory, caps


def _hf_overlap(b, k, *, cells=None, pair_cutoff=0.0):
    labels = b.cells if cells is None else cells
    radius = max([np.linalg.norm(np.asarray(b.system.lattice) @ row) for row in labels], default=0) + 0.1
    available = {tuple(cell.index): cell for cell in core.direct_lattice_cells(b.system, radius)}
    native_cells = [available[tuple(row)] for row in labels]
    overlap = core.compute_overlap_lattice_explicit(b.basis, b.system, native_cells, pair_cutoff)
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(b.system.lattice)).T
    return np.ascontiguousarray(core.bloch_sum(overlap, reciprocal @ b.mesh.fractional_at(k)))


def _overlap_audit(b, overlap, k=0, *, tolerance=2e-12, source=None, controls=None, plan=False):
    i, c = _overlap_controls() if controls is None else controls
    fn = core._plan_bipole_finite_overlap if plan else core._audit_bipole_finite_overlap
    return fn(_source(b) if source is None else source, b.basis, b.images, b.cells,
              overlap, k, tolerance, i, c)


@pytest.mark.parametrize("mesh,shift", [
    ((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (0, 0, 0)),
    ((3, 1, 1), (0, 0, 0)), ((3, 1, 1), (1, 0, 0)),
    ((2, 2, 1), (1, 1, 0)), ((3, 2, 1), (1, 1, 1)),
])
def test_source_bound_overlap_matches_hf_and_complete_complex_zero_mode_actions(mesh, shift):
    from vibeqc.bipole_fock_ewald import exchange_q0_gauge_constant, probe_charge_madelung_supercell
    from tests.test_bipole_ewald_gram import _gram
    b = _bundle(mesh=mesh, shift=shift, kl=0, kr=0, images=[])
    source = _source(b)
    nk, n = len(b.mesh), b.basis.nbasis
    overlaps = [_hf_overlap(b, k) for k in range(nk)]
    for k, overlap in enumerate(overlaps):
        result = _overlap_audit(b, overlap, k, source=source)
        assert result.direct_matches and result.dual_matches
        assert result.maximum_direct_residual < 2e-12
        assert result.maximum_dual_residual < 2e-12
        assert result.source_identity_sha256 == source.source_identity_sha256
        assert not result.physical_hamiltonian_certified and not result.symmetry_certified
    # Obtain the subtraction from two native routes, with no SR images.
    # The independent polynomial oracle separately checks each zero tensor.
    zeros = {}
    for target, origin in itertools.product(range(nk), repeat=2):
        b.selection.q_index = 0
        b.selection.left_k_index, b.selection.right_k_index = target, origin
        gram = _gram_bundle(mesh=mesh, shift=shift, q=0, left=target, right=origin,
                            basis=b.basis, cells=b.cells)
        zero = _gram(gram).values_copy() - _panel(b, source).values_copy()
        np.testing.assert_allclose(zero, _components(b)[2], atol=6e-13, rtol=4e-12)
        zeros[target, origin] = zero.reshape((n,) * 4)
    coefficient = source.zero_mode_coefficient
    xi = probe_charge_madelung_supercell(b.system, mesh)
    for origin, row, column, factor in itertools.product(range(nk), range(n), range(n), (1, 1j)):
        density = np.zeros((n, n), complex)
        density[row, column] = factor
        charge = np.trace(density @ overlaps[origin]) / nk
        for target in range(nk):
            j = -np.einsum("mnls,ls->mn", zeros[target, origin], density) / nk
            np.testing.assert_allclose(j, -coefficient * charge * overlaps[target], atol=8e-13, rtol=5e-12)
        # Exchange q=0 couples only equal source and target k.
        kzero = -np.einsum("mlns,ls->mn", zeros[origin, origin], density) / nk
        sds = overlaps[origin] @ density @ overlaps[origin]
        for probe in (0.0, xi):
            hf = exchange_q0_gauge_constant(probe, b.options.omega, source.cell_volume, nk) * sds
            np.testing.assert_allclose(kzero + probe * sds, hf, atol=8e-13, rtol=5e-12)
        if nk > 1:
            assert np.max(np.abs(kzero + coefficient * sds)) > 1e-5


@pytest.mark.parametrize("kind", ["missing_cell", "pair_cutoff", "phase", "scale"])
def test_source_bound_overlap_refuses_other_hf_support_or_gauge(kind):
    b = _bundle(shift=(1, 0, 0))
    overlap = _hf_overlap(b, 0, cells=b.cells[:1] if kind == "missing_cell" else None,
                          pair_cutoff=0.1 if kind == "pair_cutoff" else 0.0)
    if kind == "phase":
        overlap = np.ascontiguousarray(overlap.conj())
    if kind == "scale":
        overlap *= 1.01
    result = _overlap_audit(b, overlap, 0)
    assert not result.direct_matches and not result.dual_matches
    assert max(result.maximum_direct_residual, result.maximum_dual_residual) > 1e-5


def test_overlap_direct_match_does_not_hide_nonreciprocal_finite_support():
    b = _bundle(cells=[[0, 0, 0], [1, 0, 0]])
    b.options.require_cell_inversion_closure = False
    result = _overlap_audit(b, _hf_overlap(b, 1), 1)
    assert result.direct_matches
    assert not result.dual_matches
    assert result.maximum_dual_residual > 1e-3
    assert not result.physical_hamiltonian_certified
    b.selection.left_k_index = b.selection.right_k_index = 1
    zero = _components(b)[2].reshape((2,) * 4)
    density = np.array([[0.2 + 0.7j, -0.4j], [0.8, -0.1 + 0.3j]])
    overlap = _hf_overlap(b, 1)
    direct = -np.einsum("mlns,ls->mn", zero, density) / len(b.mesh)
    hf = -_source(b).zero_mode_coefficient / len(b.mesh) * (overlap @ density @ overlap)
    assert np.max(np.abs(direct - hf)) > 1e-4


@pytest.mark.parametrize("cap,field", [
    ("maximum_per_replica_inventoried_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_inventoried_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_overlap_admission_exact_and_minus_one_precedes_nan(cap, field):
    b = _bundle(); source = _source(b); overlap = _hf_overlap(b, 0)
    i, c = _overlap_controls()
    p = _overlap_audit(b, overlap, source=source, controls=(i, c), plan=True)
    setattr(c, cap, getattr(p, field))
    assert _overlap_audit(b, overlap, source=source, controls=(i, c)).direct_matches
    setattr(c, cap, getattr(p, field) - 1)
    overlap[:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _overlap_audit(b, overlap, source=source, controls=(i, c))


@pytest.mark.parametrize("payload", ["overlap", "cells", "images"])
def test_overlap_source_validation_and_finite_payload(payload):
    b = _bundle(); source = _source(b); overlap = _hf_overlap(b, 0)
    if payload == "overlap":
        overlap[0, 0] = np.nan
    elif payload == "cells":
        b.cells[0, 0] += 1
    else:
        b.images[0, 0, 0] += 1
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        _overlap_audit(b, overlap, source=source)


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_overlap_requires_finite_nonnegative_absolute_tolerance(value):
    b = _bundle()
    with pytest.raises(ValueError, match="tolerance"):
        _overlap_audit(b, _hf_overlap(b, 0), tolerance=value)


@pytest.mark.parametrize("kind", ["real", "list", "transpose", "shape"])
def test_overlap_binding_does_not_coerce_or_copy(kind):
    b = _bundle(); overlap = _hf_overlap(b, 0)
    overlap = {"real": lambda: overlap.real.copy(), "list": lambda: overlap.tolist(),
               "transpose": lambda: overlap.T, "shape": lambda: overlap.ravel()}[kind]()
    with pytest.raises((TypeError, ValueError)):
        _overlap_audit(b, overlap)


def test_overlap_receipt_binds_values_k_tolerance_but_not_execution_caps():
    b = _bundle(); overlap = _hf_overlap(b, 1); source = _source(b)
    i, c = _overlap_controls()
    first = _overlap_audit(b, overlap, 1, source=source, controls=(i, c))
    i.numerical_replicas = 2
    i.external_node_bytes = 123
    i.other_live_numerical_bytes_per_replica = 321
    i.other_live_control_bytes_per_replica = 456
    second = _overlap_audit(b, overlap, 1, source=source, controls=(i, c))
    assert first.overlap_identity_sha256 == second.overlap_identity_sha256
    assert second.memory.required_node_inventoried_bytes == 123 + 2 * second.memory.per_replica_inventoried_bytes
    assert second.memory.per_replica_inventoried_bytes == first.memory.per_replica_inventoried_bytes + 321 + 456
    assert _overlap_audit(b, overlap, 1, tolerance=3e-12).overlap_identity_sha256 != first.overlap_identity_sha256
    assert _overlap_audit(b, overlap, 0).overlap_identity_sha256 != first.overlap_identity_sha256
    overlap[0, 0] += 1e-10
    assert _overlap_audit(b, overlap, 1).overlap_identity_sha256 != first.overlap_identity_sha256
    del b, source, overlap
    gc.collect()
    assert first.direct_matches and first.dual_matches
    copy = first.memory
    assert copy.overlap_elements == 4


@pytest.mark.parametrize("angular", range(6))
def test_overlap_gate_matches_linked_hf_pure_shells_through_l5(angular):
    basis = _basis([(angular, (0.1, -0.2, 0.15), [0.55], [0.7], True)])
    b = _bundle(basis=basis, cells=[[0, 0, 0]], images=[])
    result = _overlap_audit(b, _hf_overlap(b, 0))
    assert result.direct_matches and result.dual_matches


def test_overlap_gate_l6_refuses_tiny_work_envelope_with_valid_analytic_input():
    import math
    # The local HF one-electron libint engine stops at L=5. The source's
    # L=6 overlap has an independent same-center Gaussian radial integral:
    # (2L-1)!! pi^(3/2) / [2^L (2 alpha)^(L+3/2)]. The full 13x13 audit
    # exceeds this private diagnostic's work envelope even on valid input.
    basis = _basis([(6, (0.1, -0.2, 0.15), [0.55], [0.07], True)])
    b = _bundle(basis=basis, cells=[[0, 0, 0]], images=[])
    norm = math.prod(range(1, 12, 2)) * math.pi**1.5 * 0.07**2 / (2**6 * 1.1**7.5)
    overlap = np.ascontiguousarray(np.eye(basis.nbasis, dtype=complex) * norm)
    with pytest.raises(ValueError, match="work cap"):
        _overlap_audit(b, overlap)
    from tests.test_bipole_ewald_gram import _cell_call
    # Smaller admitted Fourier tiles still verify the source's L=6 norm;
    # they do not turn the refused whole-overlap audit into a passing gate.
    pieces = [_cell_call(basis, [[0, 0, 0]], b.cells, lattice=np.asarray(b.system.lattice),
                        begin=j, count=min(64, overlap.size - j))["values"][:, 0]
              for j in range(0, overlap.size, 64)]
    np.testing.assert_allclose(np.concatenate(pieces).reshape(overlap.shape), overlap,
                               atol=2e-12, rtol=0)


def test_overlap_tolerance_is_inclusive_without_hidden_relative_scale():
    b = _bundle(); overlap = _hf_overlap(b, 0)
    overlap[0, 0] += 0.01
    r = _overlap_audit(b, overlap, tolerance=0)
    residual = max(r.maximum_direct_residual, r.maximum_dual_residual)
    assert residual > 0.009
    at = _overlap_audit(b, overlap, tolerance=residual)
    below = _overlap_audit(b, overlap, tolerance=np.nextafter(residual, 0))
    assert at.direct_matches and at.dual_matches
    assert not (below.direct_matches and below.dual_matches)


@pytest.mark.parametrize("k", [3, 2**32])
def test_overlap_out_of_mesh_index_refused(k):
    b = _bundle()
    with pytest.raises((IndexError, ValueError), match="index"):
        _overlap_audit(b, _hf_overlap(b, 0), k)


def test_overlap_alignment_is_checked_without_copying():
    b = _bundle()
    overlap = np.ndarray((2, 2), dtype=np.complex128, buffer=bytearray(65), offset=1)
    overlap[:] = _hf_overlap(b, 0)
    with pytest.raises(ValueError, match="aligned"):
        _overlap_audit(b, overlap)


@pytest.mark.parametrize("field", ["maximum_output_bytes", "maximum_work_units"])
def test_overlap_fourier_child_admission_precedes_nonfinite_payload(field):
    b = _bundle(); overlap = _hf_overlap(b, 0)
    i, c = _overlap_controls()
    setattr(c.fourier, field, 1)
    # pybind nested structures return references; pin the refusal itself.
    overlap[:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap|work|output"):
        _overlap_audit(b, overlap, controls=(i, c))


# Exact geometric SR domains, before a production HF/source integration.
def _physical_support(lattice, centers, *, pair_radius=2.1, midpoint_radius=1.6,
                      budget=None, maximum_candidates=100000, maximum_images=4096, plan=False):
    from vibeqc._bipole_physical_support import build_quartet_support, plan_quartet_support
    from vibeqc.symmetry_shared import Budget
    function = plan_quartet_support if plan else build_quartet_support
    return function(np.ascontiguousarray(lattice, dtype=float), np.ascontiguousarray(centers, dtype=float),
                    pair_radius=pair_radius, midpoint_radius=midpoint_radius,
                    budget=Budget(8 << 20, 10**10) if budget is None else budget,
                    maximum_candidates=maximum_candidates, maximum_images=maximum_images)


def _label_set(images):
    return {tuple(int(x) for x in row.ravel()) for row in images}


def _physical_centers(b, quartet):
    shells = b.basis.shells()
    return np.array([shells[i].origin for i in quartet])


def _brute_physical_support(lattice, centers, pair_radius, midpoint_radius, extent=2):
    # Independent exhaustive g,h,p cube with direct Cartesian positions. The
    # comparison fixtures are dyadic and away from any rounded boundary.
    labels = np.array(list(itertools.product(range(-extent, extent + 1), repeat=3)))
    translations = labels @ lattice.T
    result = set()
    for g, ag in zip(labels, translations):
        left = centers[1] + ag
        if np.dot(centers[0] - left, centers[0] - left) > pair_radius**2:
            continue
        for h, ah in zip(labels, translations):
            right = centers[3] + ah
            if np.dot(centers[2] - right, centers[2] - right) > pair_radius**2:
                continue
            midpoint = (centers[0] + left - centers[2] - right) / 2
            for p, ap in zip(labels, translations):
                delta = midpoint - ap
                if np.dot(delta, delta) <= midpoint_radius**2:
                    result.add(tuple(np.concatenate([g, p, p + h])))
    return result


@pytest.mark.parametrize("lattice", [np.diag([3., 4., 5.]),
    np.array([[3., 0.5, -0.25], [0., 4., 0.5], [0.25, 0., 5.]]),
    np.array([[-3., 0.5, -0.25], [0., 4., 0.5], [0.25, 0., 5.]])])
def test_physical_quartet_domains_match_exhaustive_skew_cell_reference(lattice):
    centers = np.array([[0., 0., 0.], [.5, -.25, .75], [-.25, .5, 0.], [.25, .25, .5]])
    result = _physical_support(lattice, centers, pair_radius=2.5, midpoint_radius=2.25)
    assert _label_set(result.images) == _brute_physical_support(lattice, centers, 2.5, 2.25)
    assert len(result.images) == len(_label_set(result.images)) > 0
    assert not result.physical_hamiltonian_certified and not result.symmetry_certified


@pytest.mark.parametrize("permutation", [(0,1,2,3), (1,0,2,3), (0,1,3,2), (1,0,3,2),
                                          (2,3,0,1), (3,2,0,1), (2,3,1,0), (3,2,1,0)])
def test_physical_quartet_domains_obey_all_eight_eri_permutations(permutation):
    b = _mapped_bundle()
    quartet = (0, 1, 1, 0)
    centers = _physical_centers(b, quartet)
    source = _physical_support(b.system.lattice, centers)
    target = _physical_support(b.system.lattice, centers[list(permutation)])
    mapped = []
    for row in source.images:
        absolute = np.vstack([np.zeros(3, np.int64), row])[list(permutation)]
        mapped.append(absolute[1:] - absolute[0])
    assert _label_set(target.images) == _label_set(mapped)


def test_physical_quartet_domains_reanchor_large_site_translations_without_torus_wrapping():
    lattice = np.diag([4., 5., 3.])
    centers = np.array([[.25, .5, 0.], [-.25, -.5, 1.5], [.25, .5, 0.], [-.25, -.5, 1.5]])
    shifts = np.array([[1000, -2000, 3000], [-13, 7, 6], [43, -51, 92], [71, 12, -35]], np.int64)
    source = _physical_support(lattice, centers)
    target = _physical_support(lattice, centers + shifts @ lattice.T)
    mapped = source.images + shifts[0] - shifts[1:]
    assert _label_set(target.images) == _label_set(mapped)
    assert np.max(np.abs(target.images)) > 3000
    assert source.memory.support_identity_sha256 != target.memory.support_identity_sha256


def test_physical_quartet_exact_boundary_does_not_use_a_hidden_slack():
    lattice, centers = np.eye(3), np.zeros((4, 3))
    at = _physical_support(lattice, centers, pair_radius=0, midpoint_radius=1)
    below = _physical_support(lattice, centers, pair_radius=0, midpoint_radius=np.nextafter(1., 0))
    above = _physical_support(lattice, centers, pair_radius=0, midpoint_radius=np.nextafter(1., 2))
    assert len(at.images) == len(above.images) == 7
    assert len(below.images) == 1
    assert _label_set(at.images) == _label_set(above.images)


def test_physical_quartet_exchange_keeps_distant_output_pair():
    from tests.test_bipole_erfc_panel import _call
    lattice, centers = np.eye(3) * 2, np.zeros((4, 3))
    # Actual exchange ordering is (a,c|b,d). The (a,b) output separation
    # is p, while the supported products have g=0 and s-p=0.
    support = _physical_support(lattice, centers, pair_radius=0.5, midpoint_radius=2.1)
    label = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], np.int64)
    assert tuple(label.ravel()) in _label_set(support.images)
    assert np.linalg.norm(lattice @ label[1]) > 0.5
    basis = _basis([(0, (0., 0., 0.), [0.55], [0.7], True)])
    value = _call(basis, label[None], left=(0, 1), right=(0, 1), lattice=lattice).values_copy()[0, 0, 0]
    assert value > 1e-3


@pytest.mark.parametrize("mesh,shift", [((1,1,1),(0,0,0)), ((1,1,2),(0,0,0)),
    ((1,1,3),(0,0,0)), ((1,1,3),(0,0,1)), ((2,2,2),(1,1,1))])
def test_physical_support_screw_covariance_reaches_native_multik_sr_integrals(mesh, shift):
    from tests.test_bipole_erfc_panel import _bloch
    b = _mapped_bundle(mesh=mesh, shift=shift)
    q, kl, kr = min(1, len(b.mesh)-1), 0, len(b.mesh)-1
    moduli = b.mesh.doubled_modulus
    def rotated(index, transfer=False):
        raw = b.mesh.transfer_address(index) if transfer else b.mesh.address(index)
        wrapped = [(-raw[0]) % moduli[0], (-raw[1]) % moduli[1], raw[2]]
        return b.mesh.index_of_transfer(wrapped) if transfer else b.mesh.index(wrapped)
    qt, klt, krt = rotated(q, True), rotated(kl), rotated(kr)
    maximum_phase_imaginary = 0
    for quartet in itertools.product(range(2), repeat=4):
        mapping = _mapped(b, controls=_mapped_controls(quartet))
        destination = tuple(mapping.destination_shells)
        left = (2*quartet[0] + quartet[1], 1)
        right = (2*quartet[2] + quartet[3], 1)
        target_left = (2*destination[0] + destination[1], 1)
        target_right = (2*destination[2] + destination[3], 1)
        source = _physical_support(b.system.lattice, _physical_centers(b, quartet))
        target = _physical_support(b.system.lattice, _physical_centers(b, destination))
        ell = np.array(mapping.atom_shifts).reshape(4, 3)
        offset = ell[0] - ell[1:]
        mapped = source.images @ np.diag([-1, -1, 1]) + offset
        assert _label_set(mapped) == _label_set(target.images)
        # Source-dependent supports close for all 16 shell quartets, even
        # where the old common finite window failed its screw audit.
        actual = _bloch(b.basis, source.images, lattice=b.system.lattice, mesh=mesh, shift=shift,
                         q=q, kl=kl, kr=kr, left=left, right=right).values_copy()[0, 0]
        transformed = _bloch(b.basis, target.images, lattice=b.system.lattice, mesh=mesh, shift=shift,
                              q=qt, kl=klt, kr=krt, left=target_left, right=target_right).values_copy()[0, 0]
        phase = _exact_phase(offset, mesh, shift, qt, klt, krt)
        np.testing.assert_allclose(transformed, phase * actual, atol=2e-12, rtol=5e-12)
        # An independent quadrature oracle checks one complete operator
        # slice, including nontrivial nonsymmorphic complex phases.
        raw = _erfc_oracle(b.basis, source.images, lattice=np.asarray(b.system.lattice),
                           left=left, right=right, omega=0.6)[:, 0, 0]
        expected = sum(_exact_phase(row, mesh, shift, q, kl, kr)*float(value)
                       for row, value in zip(source.images, raw))
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=5e-12)
        maximum_phase_imaginary = max(maximum_phase_imaginary, abs(complex(phase).imag))
    if mesh == (1,1,3):
        assert maximum_phase_imaginary > 0.8


@pytest.mark.parametrize("which", ["bytes", "work", "candidates", "images"])
def test_physical_support_exact_caps_and_minus_one(which):
    from vibeqc.symmetry_shared import Budget
    lattice, centers = np.eye(3) * 2, np.zeros((4, 3))
    plan = _physical_support(lattice, centers, plan=True)
    kwargs = dict(budget=Budget(plan.inventoried_bytes, plan.work_units),
                  maximum_candidates=plan.candidate_count, maximum_images=plan.image_count)
    assert _physical_support(lattice, centers, **kwargs).memory == plan
    if which == "bytes":
        kwargs['budget'] = Budget(plan.inventoried_bytes-1, plan.work_units)
    elif which == "work":
        kwargs['budget'] = Budget(plan.inventoried_bytes, plan.work_units-1)
    elif which == "candidates":
        kwargs['maximum_candidates'] -= 1
    else:
        kwargs['maximum_images'] -= 1
    with pytest.raises((ValueError, MemoryError)):
        _physical_support(lattice, centers, **kwargs)


def test_physical_support_immutable_snapshot_and_empty_support():
    lattice, centers = np.eye(3)*4, np.zeros((4,3))
    source = _physical_support(lattice, centers)
    receipt = source.memory.support_identity_sha256
    centers[:] = 500
    lattice[:] = 0
    assert source.memory.support_identity_sha256 == receipt
    with pytest.raises(ValueError):
        source.images.flags.writeable = True
    centers[1] += 0.25
    empty = _physical_support(np.eye(3)*4, centers, pair_radius=0)
    assert empty.images.shape == (0, 3, 3)
    assert not empty.physical_hamiltonian_certified


@pytest.mark.parametrize("change", ['nan', 'singular', 'huge', 'tiny', 'negative'])
def test_physical_support_refuses_invalid_or_unadmitted_geometry(change):
    lattice, centers = np.eye(3)*4, np.zeros((4,3))
    pair_radius=2.1
    if change == 'nan': centers[0,0] = np.nan
    if change == 'singular': lattice[0] = 0
    if change == 'huge': centers[0,0] = 1e100
    if change == 'tiny': lattice[0,0] = 1e-100
    if change == 'negative': pair_radius=-1
    with pytest.raises(ValueError):
        _physical_support(lattice, centers, pair_radius=pair_radius)


def test_physical_quartet_vacuum_box_reduces_to_molecular_home_cell():
    centers = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.], [.5, .5, 0.]])
    result = _physical_support(np.eye(3)*20, centers, pair_radius=2, midpoint_radius=2)
    np.testing.assert_array_equal(result.images, np.zeros((1, 3, 3), np.int64))


@pytest.mark.parametrize("change", ['list', 'float32', 'transpose', 'radius_bool', 'cap_bool', 'cap_float'])
def test_physical_support_requires_explicit_typed_inputs(change):
    from vibeqc._bipole_physical_support import build_quartet_support
    from vibeqc.symmetry_shared import Budget
    lattice, centers = np.eye(3)*4, np.zeros((4,3))
    kw = dict(pair_radius=2.1, midpoint_radius=1.6, budget=Budget(8<<20,10**10),
              maximum_candidates=100000, maximum_images=4096)
    if change == 'list': centers = centers.tolist()
    if change == 'float32': centers = centers.astype(np.float32)
    if change == 'transpose': lattice = lattice.T
    if change == 'radius_bool': kw['pair_radius'] = True
    if change == 'cap_bool': kw['maximum_images'] = True
    if change == 'cap_float': kw['maximum_candidates'] = 100000.5
    with pytest.raises((ValueError, TypeError)):
        build_quartet_support(lattice, centers, **kw)


def test_physical_support_fixed_admission_precedes_invalid_geometry():
    from vibeqc.symmetry_shared import Budget
    with pytest.raises(MemoryError, match='budget'):
        _physical_support(np.eye(3), np.full((4,3), np.nan), budget=Budget(1,10**10))
