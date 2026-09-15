"""Small independent orbital-sewing oracles, not BIPOLE energy admission.

Synthetic stationary states let us vary phase, degeneracy, core masks and
metric independently. They are deliberately not authenticated SCF sources.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_ao_bloch_transport import (
    _basis, _controls as _ao_controls, _one_atom, _operation,
)


def _unitary(n, seed):
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]


def _fixture(*, tr=True, frozen=False, retained=4, mix_core=False,
             energy_drift=0.0, broken_subspace=False, shift=(0, 0, 0),
             nondegenerate=False, source_gauge=False, discarded_drift=0.0,
             target_mask_swap=False, target_metric_scale=1.0):
    """Analytical s+p inversion or scalar TR; no native transport oracle."""
    system = _one_atom()
    system.unit_cell = [core.Atom(4, [0, 0, 0])]
    basis = _basis(system, angular=(0, 1))
    mesh = core._RegularKMesh((3, 1, 1), shift)
    op = _operation() if tr else _operation(-np.eye(3, dtype=int))
    n = basis.nbasis
    rng = np.random.default_rng(920)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    overlap = a.conj().T @ a + np.eye(n)
    w, v = np.linalg.eigh(overlap)
    c0 = (v / np.sqrt(w)) @ v.conj().T @ _unitary(n, 921)
    energies = np.array([-1.0, -1.0, 0.5, 0.5])[:retained]
    if nondegenerate:
        energies = np.array([-1.3, -1.0, 0.5, 0.9])[:retained]
    c0 = c0[:, :retained]
    gauge = np.zeros((retained, retained), dtype=complex)
    gauge[:2, :2] = _unitary(2, 922)
    gauge[2:, 2:] = _unitary(retained - 2, 923)
    if frozen and not mix_core:
        gauge[:2, :2] = np.diag(np.exp(1j * np.array([0.2, -0.7])))
    if nondegenerate:
        gauge = np.diag(np.exp(1j * np.arange(retained) * 0.7))
    # Both operations reverse k. On a shifted odd mesh source 0 -> 2;
    # otherwise source 1 -> 2. Shift on singleton axes is not used here.
    source = 0 if shift[0] else 1
    target = 2
    ao_action = np.eye(n) if tr else np.diag([1, -1, -1, -1])
    transported = c0.conj() if tr else ao_action @ c0
    target_overlap = overlap.conj() if tr else ao_action @ overlap @ ao_action
    target_c = transported @ gauge
    target_overlap = target_overlap * target_metric_scale
    target_c = target_c / np.sqrt(target_metric_scale)
    if target_mask_swap:
        target_c = target_c[:, [1, 0] + list(range(2, retained))]
    vg = np.eye(retained, dtype=complex)
    if source_gauge:
        vg[:2, :2] = _unitary(2, 924)
        vg[2:, 2:] = _unitary(retained - 2, 925)
        c0 = c0 @ vg
    expected = gauge.conj().T @ (vg.conj() if tr else vg)
    if broken_subspace:
        # Keep each state stationary/orthonormal but mix occupied and virtual.
        turn = np.eye(retained)
        turn[1:3, 1:3] = [[0.8, -0.6], [0.6, 0.8]]
        target_c = target_c @ turn

    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = "9" * 64
    data.periodic_dimension = 3
    data.mesh = (3, 1, 1)
    data.is_shift = shift
    data.reciprocal_lattice = system.reciprocal_lattice()
    data.converged = True
    data.n_basis = n
    data.n_effective_orbitals = retained
    data.electrons_per_cell = 4
    data.reference_energy_per_cell = -3.0
    data.minimum_band_gap_hartree = 0.1
    masks = ([1, 0] + [0] * (retained - 2) if frozen else [0] * retained,
             [0, 1] + [0] * (retained - 2) if frozen else [1, 1] + [0] * (retained - 2),
             [0, 0] + [1] * (retained - 2))
    for k in range(len(mesh)):
        s, c = (target_overlap, target_c) if k == target else (overlap, c0)
        eps = energies.copy()
        if k == target:
            eps += energy_drift
        # F C = S C eps, with a harmless complement for discarded AOs.
        sc = s @ c
        complement_energy = 2.0 + (discarded_drift if k == target else 0.0)
        f = sc @ np.diag(eps) @ sc.conj().T + complement_energy * (s - sc @ sc.conj().T)
        point_masks = masks
        if k == target and target_mask_swap:
            point_masks = tuple([mask[1], mask[0]] + mask[2:] for mask in masks)
        data.add_kpoint(
            k_cartesian=system.reciprocal_lattice() @ mesh.fractional_at(k),
            weight=1 / len(mesh), overlap=s, fock=f, coefficients=c,
            orbital_energies=eps, occupations=[2, 2] + [0] * (retained - 2),
            frozen_core_mask=point_masks[0], correlated_occupied_mask=point_masks[1],
            virtual_mask=point_masks[2],
        )
    state = core._make_periodic_restricted_mean_field_state(data)
    return system, basis, op, state, source, tr, expected, transported


_TOLERANCES = (
    "maximum_reciprocal_lattice_residual", "maximum_source_metric_residual",
    "maximum_target_metric_residual", "maximum_transported_metric_residual",
    "maximum_unitarity_residual", "maximum_reconstruction_residual",
    "maximum_cross_subspace_overlap", "maximum_roothaan_residual",
    "maximum_energy_intertwining_residual",
)
_CAP_TO_PLAN = {
    "maximum_kpoints": "n_kpoints",
    "maximum_basis_functions": "n_basis",
    "maximum_effective_orbitals": "n_effective_orbitals",
    "maximum_subspace_rank": "maximum_rank",
    "maximum_transport_calls": "transport_calls",
    "maximum_borrowed_numerical_bytes": "borrowed_numerical_bytes",
    "maximum_owned_numerical_bytes": "peak_owned_numerical_bytes",
    "maximum_control_storage_bytes": "control_storage_bytes",
    "maximum_per_replica_inventoried_bytes": "per_replica_inventoried_bytes",
    "maximum_node_inventoried_bytes": "required_node_inventoried_bytes",
    "maximum_work_units": "work_units_upper_bound",
}


def _controls(*, full=False):
    ao_options, _, ao_caps = _ao_controls()
    options = core._PeriodicOrbitalSewingOptions()
    options.ao_transport = ao_options
    options.request_full_ao_scope = full
    for field in _TOLERANCES:
        setattr(options, field, 1e-10)
    inventory = core._PeriodicOrbitalSewingInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 1 << 20
    caps = core._PeriodicOrbitalSewingCaps()
    caps.ao_transport = ao_caps
    for field in _CAP_TO_PLAN:
        setattr(caps, field, 16 << 20)
    caps.maximum_work_units = 10**9
    return options, inventory, caps


def _call(case, *, controls=None, plan=False):
    system, basis, op, state, source, tr, *_ = case
    fn = core._plan_periodic_orbital_sewing if plan else core._make_periodic_orbital_sewing
    return fn(state, basis, system, op, source, tr,
              *(_controls() if controls is None else controls))


def _subspaces():
    s = core._PeriodicOrbitalSubspace
    return (s.FROZEN_CORE, s.CORRELATED_OCCUPIED, s.VIRTUAL)


@pytest.mark.parametrize("tr", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 0)])
@pytest.mark.parametrize("nondegenerate", [False, True])
def test_full_subspace_sewing_matches_independent_complex_metric_oracle(
    tr, frozen, shift, nondegenerate,
):
    case = _fixture(tr=tr, frozen=frozen, shift=shift, nondegenerate=nondegenerate)
    result = _call(case, controls=_controls(full=True))
    expected = case[6]
    groups = ([0], [1], [2, 3]) if frozen else ([], [0, 1], [2, 3])
    for subspace, group, diag in zip(_subspaces(), groups, result.diagnostics.subspaces):
        u = result.sewing_copy(subspace)
        assert u == pytest.approx(expected[np.ix_(group, group)], abs=2e-12)
        assert diag.rank == len(group)
        for i, band in enumerate(group):
            assert result.source_band(subspace, i) == band
            assert result.target_band(subspace, i) == band
            for j in range(len(group)):
                assert result.element(subspace, i, j) == u[i, j]
        assert not u.flags.writeable
        for name in ("source_metric_residual", "target_metric_residual",
                     "transported_metric_residual", "unitarity_residual",
                     "reconstruction_residual", "cross_subspace_overlap",
                     "source_roothaan_residual", "target_roothaan_residual",
                     "roothaan_residual", "energy_intertwining_residual"):
            assert getattr(diag, name) < 2e-12
    assert result.memory.target_index == 2
    assert result.memory.subspace_ranks == [len(g) for g in groups]
    assert result.diagnostics.audited_source_columns == 4
    assert result.full_ao_scope_audited
    assert not result.physical_source_symmetry_certified
    assert result.state is case[3]


@pytest.mark.parametrize("tr", [False, True])
def test_independent_source_and_target_gauges_use_correct_antiunitary_law(tr):
    # U' = V_t^dagger U V_s (unitary) or V_t^dagger U V_s* (antiunitary).
    case = _fixture(tr=tr, source_gauge=True)
    result = _call(case)
    for subspace, group in zip(_subspaces()[1:], ([0, 1], [2, 3])):
        assert result.sewing_copy(subspace) == pytest.approx(
            case[6][np.ix_(group, group)], abs=2e-12,
        )


@pytest.mark.parametrize("tr", [False, True])
def test_exactly_degenerate_core_active_mixing_is_rejected_despite_equal_density(tr):
    case = _fixture(tr=tr, frozen=True, mix_core=True)
    state, transported = case[3], case[7]
    target = state.coefficients(2)[:, :2]
    assert target @ target.conj().T == pytest.approx(
        transported[:, :2] @ transported[:, :2].conj().T, abs=2e-13,
    )
    with pytest.raises(ValueError, match="[Ss]ubspace|[Uu]nitarity|[Rr]econstruction|[Cc]ross"):
        _call(case)


@pytest.mark.parametrize("kind", ["energy", "occupied-virtual", "metric"])
def test_valid_stationary_states_without_operation_covariance_are_rejected(kind):
    case = _fixture(energy_drift=0.02) if kind == "energy" else _fixture(broken_subspace=True)
    if kind == "metric":
        case = _fixture(target_metric_scale=2.0)
    with pytest.raises(ValueError):
        _call(case)


@pytest.mark.parametrize("tr", [False, True])
def test_masks_can_select_different_native_bands_at_source_and_target(tr):
    case = _fixture(tr=tr, frozen=True, target_mask_swap=True)
    result = _call(case)
    frozen, active, _ = _subspaces()
    assert result.source_band(frozen, 0) == 0
    assert result.target_band(frozen, 0) == 1
    assert result.source_band(active, 0) == 1
    assert result.target_band(active, 0) == 0
    assert result.element(frozen, 0, 0) == pytest.approx(case[6][0, 0], abs=2e-12)
    assert result.element(active, 0, 0) == pytest.approx(case[6][1, 1], abs=2e-12)


def test_discarded_ao_fock_can_differ_only_retained_scope_is_audited():
    case = _fixture(retained=3, discarded_drift=0.7)
    state = case[3]
    assert np.max(np.abs(state.fock(2) - state.fock(1).conj())) > 0.1
    result = _call(case)
    assert not result.full_ao_scope_audited
    assert not result.memory.full_ao_scope
    assert not result.physical_source_symmetry_certified
    with pytest.raises(ValueError, match="[Ff]ull|[Rr]etained|[Rr]ank"):
        _call(case, controls=_controls(full=True))


@pytest.mark.parametrize("field", list(_CAP_TO_PLAN))
def test_each_enclosing_exact_cap_and_cap_minus_one(field):
    case = _fixture(frozen=True)
    controls = _controls()
    plan = _call(case, controls=controls, plan=True)
    value = getattr(plan, _CAP_TO_PLAN[field])
    assert value > 0
    setattr(controls[2], field, value)
    assert _call(case, controls=controls).memory.work_units_upper_bound == plan.work_units_upper_bound
    setattr(controls[2], field, value - 1)
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        _call(case, controls=controls)


@pytest.mark.parametrize("field", _TOLERANCES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_all_numerical_tolerances_are_explicit_finite_nonnegative(field, value):
    controls = _controls()
    setattr(controls[0], field, value)
    with pytest.raises((ValueError, OverflowError)):
        _call(_fixture(), controls=controls, plan=True)


@pytest.mark.parametrize("field", [
    "maximum_reciprocal_lattice_residual", "maximum_source_metric_residual",
    "maximum_target_metric_residual", "maximum_transported_metric_residual",
    "maximum_unitarity_residual", "maximum_cross_subspace_overlap",
])
def test_dimensionless_controls_cannot_accept_a_unit_sized_defect(field):
    controls = _controls()
    setattr(controls[0], field, 1.0)
    with pytest.raises(ValueError, match="less than one"):
        _call(_fixture(), controls=controls, plan=True)


@pytest.mark.parametrize("field", ["numerical_replicas", "backend_margin_bytes_per_replica"])
def test_unset_replica_or_backend_inventory_is_rejected(field):
    controls = _controls()
    setattr(controls[1], field, 0)
    with pytest.raises(ValueError, match="positive"):
        _call(_fixture(), controls=controls, plan=True)


def test_state_owner_and_readonly_copies_survive_input_lifetime():
    case = _fixture()
    state_id = case[3].state_identity_sha256
    result = _call(case)
    subspace = _subspaces()[1]
    first = result.sewing_copy(subspace)
    saved = first.copy()
    first.setflags(write=True)
    first[:] = 99
    del case
    gc.collect()
    assert result.state_identity_sha256 == state_id
    assert result.state.state_identity_sha256 == state_id
    assert result.sewing_copy(subspace) == pytest.approx(saved, abs=0)
    with pytest.raises((IndexError, ValueError)):
        result.element(subspace, 2, 0)
    with pytest.raises((IndexError, ValueError)):
        result.source_band(subspace, 2)


def test_full_state_is_charged_once_and_replicas_scale_whole_inventory():
    case = _fixture()
    controls = _controls()
    p = _call(case, controls=controls, plan=True)
    assert p.state_resident_numerical_bytes == case[3].resident_bytes
    assert p.borrowed_numerical_bytes == (
        case[3].resident_bytes + p.borrowed_basis_numeric_bytes + p.borrowed_geometry_numeric_bytes
    )
    assert p.retained_sewing_bytes == 16 * (2**2 + 2**2)
    inventory = controls[1]
    inventory.numerical_replicas = 3
    inventory.external_node_bytes = 123
    q = _call(case, controls=controls, plan=True)
    assert q.per_replica_inventoried_bytes == p.per_replica_inventoried_bytes
    assert q.required_node_inventoried_bytes == 3 * p.per_replica_inventoried_bytes + 123
    inventory.other_live_numerical_bytes_per_replica = 17
    inventory.other_live_control_bytes_per_replica = 29
    extra = _call(case, controls=controls, plan=True)
    assert extra.per_replica_inventoried_bytes == p.per_replica_inventoried_bytes + 46
    assert extra.required_node_inventoried_bytes == q.required_node_inventoried_bytes + 3 * 46
    inventory.external_node_bytes = 2**64 - 1
    with pytest.raises(OverflowError):
        _call(case, controls=controls, plan=True)


@pytest.mark.parametrize("kind", ["cell", "dimension", "basis"])
def test_detached_state_geometry_metadata_mismatch_is_rejected(kind):
    case = list(_fixture())
    if kind == "cell":
        case[0].lattice = 1.1 * case[0].lattice
    elif kind == "dimension":
        case[0].dim = 2
    else:
        case[1] = _basis(case[0], angular=(0,))
    with pytest.raises(ValueError):
        _call(case)


def test_resource_refusal_precedes_bad_reciprocal_payload_read():
    case = list(_fixture())
    case[0].lattice = np.full((3, 3), np.nan)
    controls = _controls()
    controls[2].maximum_owned_numerical_bytes = 1
    with pytest.raises((RuntimeError, ValueError), match="[Cc]ap|[Bb]ytes|[Rr]esource"):
        _call(case, controls=controls)


@pytest.fixture(scope="module", params=[(2, 1, 1), (3, 1, 1)])
def finite_gaussian_hf(request):
    # Deliberately tiny finite-Gaussian oracle. This is NOT the BIPOLE source.
    from tests.test_periodic_gaussian_rhf import _bundle, _run
    bundle = _bundle(request.param)
    result = _run(bundle)
    return bundle, result


@pytest.mark.parametrize("tr", [False, True])
def test_actual_native_full_k_hf_state_sews_without_source_relabeling(finite_gaussian_hf, tr):
    bundle, hf = finite_gaussian_hf
    op = _operation() if tr else _operation(-np.eye(3, dtype=int))
    case = (bundle.system, bundle.ao, op, hf.state, 1, tr)
    result = _call(case, controls=_controls(full=True))
    assert result.state is hf.state
    assert result.full_ao_scope_audited
    assert not result.physical_source_symmetry_certified
    for diag in result.diagnostics.subspaces:
        assert diag.roothaan_residual < 1e-10
        assert diag.reconstruction_residual < 1e-10


@pytest.mark.parametrize("tr", [False, True])
def test_nonsymmorphic_subspace_composition_retains_lattice_phase(tr):
    lattice = np.eye(3) * 8
    system = core.PeriodicSystem(3, lattice, [core.Atom(2, [0, 0, 0]), core.Atom(2, [4, 0, 0])])
    shells = [core.ShellInfo(a, 0, True, [exponent], [1.0], list(atom.xyz))
              for a, atom in enumerate(system.unit_cell) for exponent in (0.7, 1.4)]
    basis = core.BasisSet(system.unit_cell_molecule(), shells, "sewing-half-translation", False)
    op = _operation(translation=[0.5, 0, 0])
    mesh = core._RegularKMesh((3, 1, 1))
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = "8" * 64
    data.periodic_dimension = 3
    data.mesh = (3, 1, 1)
    data.reciprocal_lattice = system.reciprocal_lattice()
    data.converged = True
    data.n_basis = data.n_effective_orbitals = 4
    data.electrons_per_cell = 4
    data.reference_energy_per_cell = -3.0
    data.minimum_band_gap_hartree = 0.1
    coefficients = []
    for k in range(3):
        phase = np.exp(-1j * np.pi * k / 3)
        c = np.zeros((4, 4), dtype=complex)
        # Both +/- states of radial 0 are occupied; radial 1 is virtual.
        # The occupied class is closed under both Seitz and scalar TR.
        for radial in range(2):
            for parity in range(2):
                c[radial, 2 * radial + parity] = (-1)**parity * phase / np.sqrt(2)
                c[radial + 2, 2 * radial + parity] = 1 / np.sqrt(2)
        eps = [-1.0, -1.0, 0.5, 0.5]
        data.add_kpoint(
            k_cartesian=system.reciprocal_lattice() @ mesh.fractional_at(k), weight=1 / 3,
            overlap=np.eye(4, dtype=complex), fock=c @ np.diag(eps) @ c.conj().T,
            coefficients=c, orbital_energies=eps, occupations=[2, 2, 0, 0],
            frozen_core_mask=[0, 0, 0, 0], correlated_occupied_mask=[1, 1, 0, 0],
            virtual_mask=[0, 0, 1, 1],
        )
        coefficients.append(c)
    state = core._make_periodic_restricted_mean_field_state(data)
    first = _call((system, basis, op, state, 1, tr))
    target = 2 if tr else 1
    second = _call((system, basis, op, state, target, tr))
    gamma = np.zeros((4, 4), dtype=complex)
    gamma[:2, 2:] = np.exp(-2j * np.pi * target / 3) * np.eye(2)
    gamma[2:, :2] = np.eye(2)
    source_c = coefficients[1].conj() if tr else coefficients[1]
    expected = coefficients[target].conj().T @ gamma @ source_c
    for subspace, group in zip(_subspaces()[1:], ([0, 1], [2, 3])):
        u = first.sewing_copy(subspace)
        v = second.sewing_copy(subspace)
        assert u == pytest.approx(expected[np.ix_(group, group)], abs=2e-12)
        product = v @ (u.conj() if tr else u)
        assert product == pytest.approx(np.exp(-2j * np.pi / 3) * np.eye(2), abs=2e-12)
