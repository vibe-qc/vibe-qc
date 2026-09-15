"""Correctness gates for AICCM2026DEV-B finite-torus symmetry."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import vibeqc as vq

from vibeqc.periodic.chi.scf import (
    AICCM2026DevBExperimentalWarning,
    cyclic_gamma_mesh,
    run_aiccm2026dev_b_rhf,
)
from vibeqc.periodic.chi.symmetry import (
    AICCM2026DevBRealTorusAOAction,
    AICCM2026DevBOccupiedSymmetryAction,
    build_aiccm2026dev_b_occupied_symmetry_action,
    AICCM2026DevBTorusSymmetryGroup,
    AICCM2026DevBOccupiedGroupWitness,
    AICCM2026DevBRestrictedSnapshot,
    build_aiccm2026dev_b_restricted_snapshot,
    build_aiccm2026dev_b_torus_symmetry_group,
    build_aiccm2026dev_b_occupied_group_witness,
    build_aiccm2026dev_b_real_torus_ao_action,
    build_aiccm2026dev_b_symmetry_plan,
    gamma_matrix_symmetry_residual,
    shell_pair_orbits,
    shell_quartet_orbits,
    symmetrize_gamma_ao_matrix,
)
from vibeqc.periodic_k_gdf import PeriodicKRHFGDFResult
from vibeqc.symmetry_ao import build_ao_permutation_matrix
from vibeqc.symmetry_lattice import lattice_to_cartesian_rotation


@pytest.fixture(scope="module", params=[2, 3])
def chi_snapshot_fixture(request):
    # Two equivalent He atoms: two occupied and two virtual 6-31G bands.
    # The half-cell translation squares to a lattice translation at nonzero k.
    system = vq.PeriodicSystem(3, np.diag([8., 16., 16.]), [
        vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [4., 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "6-31g")
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-10
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system, basis, (request.param, 1, 1), options=options, progress=False,
        )
    assert result.converged
    return system, basis, result


def _chi_snapshot(case, **kwargs):
    system, basis, result = case
    controls = dict(declared_calculation_identity="a"*64, frozen_core_bands=[],
                    minimum_band_gap_hartree=1e-5)
    controls.update(kwargs)
    return build_aiccm2026dev_b_restricted_snapshot(result, system, basis, **controls)


def _chi_sewing_controls():
    core = vq._vibeqc_core
    ao = core._PeriodicAOBlochTransportOptions()
    ao.maximum_atom_mapping_residual_bohr = 1e-9
    ao.maximum_basis_origin_residual_bohr = 1e-10
    ao.maximum_rotation_orthogonality_residual = 1e-11
    ao.maximum_polynomial_reconstruction_residual = 1e-10
    ao.maximum_pure_rotation_unitarity_residual = 1e-10
    ao.minimum_relative_lattice_volume = 1e-10
    ac = core._PeriodicAOBlochTransportCaps()
    for field in ("maximum_atoms", "maximum_shells", "maximum_contractions",
                  "maximum_basis_functions", "maximum_columns"):
        setattr(ac, field, 512)
    ac.maximum_basis_numeric_lanes = 10000
    for field in ("maximum_borrowed_numerical_bytes", "maximum_owned_numerical_bytes",
                  "maximum_control_storage_bytes", "maximum_per_replica_inventoried_bytes",
                  "maximum_node_inventoried_bytes"):
        setattr(ac, field, 16 << 20)
    ac.maximum_work_units = 10**9
    options = core._PeriodicOrbitalSewingOptions()
    options.ao_transport = ao
    options.request_full_ao_scope = True
    for field in ("maximum_reciprocal_lattice_residual", "maximum_source_metric_residual",
                  "maximum_target_metric_residual", "maximum_transported_metric_residual",
                  "maximum_unitarity_residual", "maximum_reconstruction_residual",
                  "maximum_cross_subspace_overlap", "maximum_roothaan_residual",
                  "maximum_energy_intertwining_residual"):
        setattr(options, field, 1e-8)
    inventory = core._PeriodicOrbitalSewingInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 1 << 20
    caps = core._PeriodicOrbitalSewingCaps()
    caps.ao_transport = ac
    for field in ("maximum_kpoints", "maximum_basis_functions", "maximum_effective_orbitals",
                  "maximum_subspace_rank", "maximum_transport_calls",
                  "maximum_borrowed_numerical_bytes", "maximum_owned_numerical_bytes",
                  "maximum_control_storage_bytes", "maximum_per_replica_inventoried_bytes",
                  "maximum_node_inventoried_bytes"):
        setattr(caps, field, 16 << 20)
    caps.maximum_work_units = 10**9
    return options, inventory, caps


def test_chi_snapshot_preserves_actual_payload_masks_and_immutable_owner(chi_snapshot_fixture):
    system, basis, result = chi_snapshot_fixture
    snapshot = _chi_snapshot(chi_snapshot_fixture, frozen_core_bands=[1])
    state = snapshot.state
    assert not snapshot.physical_source_symmetry_certified
    assert snapshot.finite_torus_convention == result.finite_torus_convention
    assert snapshot.finite_torus_convention.exchange_q0_applicability == "active"
    assert state.n_kpoints == result.aiccm2026dev_b.n_kpoints
    assert state.reference_energy_per_cell == result.energy
    assert state.mesh == list(result.aiccm2026dev_b.mesh)
    assert state.is_shift == [0, 0, 0]
    for k in range(state.n_kpoints):
        for native, field in (("overlap", "overlap"), ("fock", "fock"), ("coefficients", "mo_coeffs"),
                              ("orbital_energies", "mo_energies"), ("occupations", "occupations")):
            np.testing.assert_array_equal(getattr(state, native)(k), getattr(result, field)[k])
        np.testing.assert_array_equal(state.kpoint_cartesian(k), result.kpoints_cart[k])
        assert state.weight(k) == result.kpoint_weights[k]
        assert state.frozen_core_mask(k) == [0, 1, 0, 0]
        assert state.correlated_occupied_mask(k) == [1, 0, 0, 0]
        assert state.virtual_mask(k) == [0, 0, 1, 1]
    assert copy.copy(snapshot) is snapshot and copy.deepcopy(snapshot) is snapshot
    with pytest.raises(TypeError, match="use build"):
        AICCM2026DevBRestrictedSnapshot()
    changed = copy.copy(result)
    changed.fock = [block.copy() for block in result.fock]
    detached = _chi_snapshot((system, basis, changed))
    original = detached.state.fock(0)
    changed.fock[0][:] = 0
    np.testing.assert_array_equal(detached.state.fock(0), original)
    output_copy = detached.state.fock(0)
    output_copy[:] = 0
    np.testing.assert_array_equal(detached.state.fock(0), original)


@pytest.mark.parametrize("frozen", [[], [0], [1]])
@pytest.mark.parametrize("tr", [False, True])
@pytest.mark.parametrize("inversion", [False, True])
def test_chi_snapshot_actual_multik_native_masked_sewing(chi_snapshot_fixture, frozen, tr, inversion):
    core = vq._vibeqc_core
    system, basis, _ = chi_snapshot_fixture
    state = _chi_snapshot(chi_snapshot_fixture, frozen_core_bands=frozen).state
    tau = [0.5, 0., 0.] if inversion else [0., 0., 0.]
    rotation = -np.eye(3, dtype=np.int32) if inversion else np.eye(3, dtype=np.int32)
    op = core._make_periodic_ao_bloch_operation(rotation, np.array(tau))
    for k in range(state.n_kpoints):
        target = (-k) % state.n_kpoints if inversion != tr else k
        c = state.coefficients(k).conj() if tr else state.coefficients(k)
        # Bond-centred inversion exchanges the two atoms without cell offsets.
        ao = np.zeros((4, 4), complex)
        ao[2:, :2] = np.eye(2)
        ao[:2, 2:] = np.eye(2)
        if not inversion:
            ao = np.eye(4)
        if frozen:
            # A single band out of a (near-)degenerate occupied pair is not
            # necessarily an invariant core space. Audit the actual C/S
            # projection, rather than demanding that geometric symmetry
            # bless an arbitrary frozen-band mask (#704/D131).
            active = [b for b in (0, 1) if b not in frozen]
            cross = state.coefficients(target)[:, active].conj().T @ state.overlap(target) @ ao @ c[:, frozen]
            if np.linalg.norm(cross) > 1e-8:
                with pytest.raises(ValueError, match="cross-subspace mask leakage"):
                    core._make_periodic_orbital_sewing(
                        state, basis, system, op, k, tr, *_chi_sewing_controls(),
                    )
                continue
        sewing = core._make_periodic_orbital_sewing(
            state, basis, system, op, k, tr, *_chi_sewing_controls(),
        )
        assert sewing.state is state
        assert sewing.full_ao_scope_audited
        assert not sewing.physical_source_symmetry_certified
        assert sewing.memory.target_index == target
        for subspace, bands in ((core._PeriodicOrbitalSubspace.FROZEN_CORE, frozen),
                               (core._PeriodicOrbitalSubspace.CORRELATED_OCCUPIED, [b for b in (0, 1) if b not in frozen]),
                               (core._PeriodicOrbitalSubspace.VIRTUAL, [2, 3])):
            expected = state.coefficients(target)[:, bands].conj().T @ state.overlap(target) @ ao @ c[:, bands]
            np.testing.assert_allclose(sewing.sewing_copy(subspace), expected, rtol=0, atol=1e-10)


def test_chi_snapshot_does_not_certify_geometric_half_translation(chi_snapshot_fixture):
    # #704/D131: displaced nuclear-image support repairs the odd-mesh case.
    # The even-mesh case retains a separate finite-support residual.
    # Neither a passing geometric D129 group nor a single sewing operation
    # is a whole-Hamiltonian/production representative-only certificate.
    core = vq._vibeqc_core
    system, basis, _ = chi_snapshot_fixture
    state = _chi_snapshot(chi_snapshot_fixture).state
    op = core._make_periodic_ao_bloch_operation(np.eye(3, dtype=np.int32), np.array([0.5, 0., 0.]))
    core._make_periodic_orbital_sewing(state, basis, system, op, 0, False, *_chi_sewing_controls())
    for k in range(1, state.n_kpoints):
        phase = np.exp(-2j*np.pi*k/state.n_kpoints)
        ao = np.zeros((4, 4), complex)
        ao[2:, :2] = np.eye(2)
        ao[:2, 2:] = phase*np.eye(2)
        f, s, c, e = state.fock(k), state.overlap(k), state.coefficients(k), state.orbital_energies(k)
        assert np.max(np.abs(f@c-s@c@np.diag(e))) < 1e-9
        assert np.max(np.abs(ao.conj().T@s@ao-s)) < 1e-8
        residual = np.max(np.abs(ao.conj().T@f@ao-f))
        if state.n_kpoints == 2:
            assert 1e-8 < residual < 1e-6
            with pytest.raises(ValueError, match="transported target Roothaan"):
                core._make_periodic_orbital_sewing(state, basis, system, op, k, False, *_chi_sewing_controls())
        else:
            assert residual < 1e-9
            sewing = core._make_periodic_orbital_sewing(
                state, basis, system, op, k, False, *_chi_sewing_controls(),
            )
            assert sewing.full_ao_scope_audited
            assert not sewing.physical_source_symmetry_certified


@pytest.mark.parametrize("cutoff", [15.0, 16.1, 30.0])
def test_half_translation_support_reuses_shared_atom_pair_domain(cutoff):
    # Dovesi (1986), doi:10.1002/qua.560290608, Eqs. (31)-(36):
    # after reanchoring the first atom, h' = h + image_b - image_a.
    # For #704's half translation the images are 0 and +1, respectively.
    # Reuse the shared geometric domain; do not replace it by a second chi
    # orbit implementation or claim that this proves J/K covariance.
    from vibeqc.pair_resolved_truncation import domain_triples, pair_resolved_domain
    from vibeqc.symmetry_lattice_c import identify_atom_pair_orbits

    core = vq._vibeqc_core
    system = vq.PeriodicSystem(3, np.diag([8., 16., 16.]), [
        vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [4., 0., 0.]),
    ])
    operations = [core._make_periodic_ao_bloch_operation(
        np.eye(3, dtype=np.int32), np.array([shift, 0., 0.]),
    ) for shift in (0., 0.5)]
    domain = pair_resolved_domain(system, cutoff, operations=operations)
    triples = domain_triples(domain)
    for a, b, h in triples:
        image_h = (h[0] + b - a, h[1], h[2])
        assert (1-a, 1-b, image_h) in triples
        displacement = np.diag([8., 16., 16.]) @ np.array(h)
        displacement[0] += 4. * (b-a)
        assert np.linalg.norm(displacement) <= cutoff + 1e-12
    orbits = identify_atom_pair_orbits(
        domain.cells, operations, system, require_closed=True, triples=triples,
    )
    assert orbits.n_orbits * 2 == len(triples)
    radial = core.direct_lattice_cells(system, cutoff)
    with pytest.raises(ValueError, match="not symmetry-closed"):
        identify_atom_pair_orbits(radial, operations, system, require_closed=True)
    if cutoff == 15.:
        # The missing radial output partner lies at cell origin 16 bohr,
        # but its atom-pair separation is just 12 bohr.
        assert (0, 1, (1, 0, 0)) in triples
        assert (1, 0, (2, 0, 0)) in triples
        assert (2, 0, 0) not in {tuple(cell.index) for cell in radial}


@pytest.mark.parametrize("internal_extent", [24., 32., 40.])
@pytest.mark.parametrize("trial_kind", ["invariant", "general"])
def test_half_translation_raw_exchange_requires_pair_output_support(
    internal_extent, trial_kind,
):
    # #704: qualify the unreconstructed operator on both invariant and
    # non-invariant densities. Geometry closure alone is insufficient.
    from vibeqc.pair_resolved_truncation import atom_pair_shell_masks, pair_resolved_domain
    from vibeqc.pbc_bipole_common import default_ewald_alpha

    core = vq._vibeqc_core
    system = vq.PeriodicSystem(3, np.diag([8., 16., 16.]), [
        vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [4., 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "6-31g")
    alpha = default_ewald_alpha(2048., real_cutoff_bohr=15., tolerance=1e-8)
    actions = []
    for phase in (1., -1.):
        action = np.zeros((4, 4))
        action[2:, :2] = np.eye(2)
        action[:2, 2:] = phase * np.eye(2)
        actions.append(action)
    if trial_kind == "invariant":
        a = np.array([[0.8, 0.07], [0.07, 0.3]])
        b = np.array([[0.04, 0.01], [0.01, 0.02]])
        z = np.zeros((2, 2))
        trial = [np.block([[a, b], [b, a]]), np.block([[a, z], [z, a]])]
    else:
        panels = np.random.default_rng(704).normal(size=(2, 4, 4))
        trial = [p @ p.T / 8. for p in panels]
    image = [u @ p @ u.T for u, p in zip(actions, trial)]
    density_change = max(np.max(np.abs(a-b)) for a, b in zip(trial, image))
    if trial_kind == "invariant":
        assert density_change < 1e-15
    else:
        assert density_change > 0.1

    internal = list(core.direct_lattice_cells(system, internal_extent))
    positions = {tuple(c.index): i for i, c in enumerate(internal)}
    density_cells = list(core.direct_lattice_cells(system, 2. * internal_extent))
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = internal_extent
    options.sr_range_screening = True
    for mode in ("radial", "pair"):
        if mode == "pair":
            domain = pair_resolved_domain(system, 15.)
            cells = list(domain.cells)
            masks = atom_pair_shell_masks(basis, domain.pairs_by_cell)
        else:
            cells = list(core.direct_lattice_cells(system, 15.))
            masks = []
        subset = [positions[tuple(c.index)] for c in cells]
        outputs = []
        for density_k in (trial, image):
            # Independent two-character inverse transform; all panels are
            # real and Hermitian, including the self-inverse edge point.
            blocks = [0.5 * (density_k[0] + (-1.)**int(c.index[0]) * density_k[1])
                      for c in density_cells]
            density = core.make_lattice_matrix_set(4, density_cells, blocks)
            jk = core.build_jk_2e_real_space_domains(
                basis, system, options, density, internal, subset, masks, alpha,
            )
            outputs.append([
                [sum(phase**int(c.index[0]) * np.asarray(lattice.blocks[p])
                     for c, p in zip(cells, subset)) for phase in (1., -1.)]
                for lattice in (jk.J, jk.K)
            ])
        residuals = [
            max(np.max(np.abs(target - u @ source @ u.T))
                for source, target, u in zip(source_k, image_k, actions))
            for source_k, image_k in zip(*outputs)
        ]
        if mode == "pair":
            assert max(residuals) < 1e-10
        elif trial_kind == "invariant":
            assert residuals[0] < 1e-10
            assert 1e-8 < residuals[1] < 1e-6


@pytest.mark.parametrize("kwargs,message", [
    ({"declared_calculation_identity": "x"*64}, "identity"),
    ({"declared_calculation_identity": "A"*64}, "identity"),
    ({"frozen_core_bands": [0, 0]}, "unique"),
    ({"frozen_core_bands": [0, 1]}, "correlated"),
    ({"frozen_core_bands": [2]}, "occupied"),
    ({"frozen_core_bands": [True]}, "integer"),
    ({"frozen_core_bands": [0.0]}, "integer"),
    ({"frozen_core_bands": np.array(0)}, "sequence"),
    ({"minimum_band_gap_hartree": float("nan")}, "gap"),
    ({"minimum_band_gap_hartree": 10**400}, "gap"),
    ({"minimum_band_gap_hartree": True}, "gap"),
    ({"minimum_band_gap_hartree": 0}, "gap"),
    ({"max_snapshot_bytes": True}, "positive integer"),
])
def test_chi_snapshot_rejects_invalid_controls(chi_snapshot_fixture, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _chi_snapshot(chi_snapshot_fixture, **kwargs)


def test_chi_snapshot_resource_refusal_precedes_native_input(chi_snapshot_fixture, monkeypatch):
    core = vq._vibeqc_core
    amount = _chi_snapshot(chi_snapshot_fixture).estimated_snapshot_bytes
    assert _chi_snapshot(chi_snapshot_fixture, max_snapshot_bytes=amount).estimated_snapshot_bytes == amount
    def forbidden():
        pytest.fail("native input created before snapshot byte admission")
    monkeypatch.setattr(core, "_PeriodicRestrictedMeanFieldInput", forbidden)
    with pytest.raises(MemoryError, match="explicit numerical bytes"):
        _chi_snapshot(chi_snapshot_fixture, max_snapshot_bytes=amount-1)


@pytest.mark.parametrize("field,value,message", [
    ("converged", False, "converged"),
    ("functional", "PBE", "RHF"),
    ("kpoints_cart", None, "recorded"),
    ("kpoint_weights", None, "recorded"),
    ("occupations", [], "complete"),
    ("finite_torus_convention", None, "convention"),
])
def test_chi_snapshot_missing_payload_is_not_synthesized(chi_snapshot_fixture, field, value, message):
    system, basis, original = chi_snapshot_fixture
    result = copy.copy(original)
    setattr(result, field, value)
    with pytest.raises(ValueError, match=message):
        _chi_snapshot((system, basis, result))


@pytest.mark.parametrize("field", ["fock", "overlap", "mo_coeffs", "mo_energies", "occupations", "kpoints_cart", "kpoint_weights"])
def test_chi_snapshot_native_checks_see_original_unrepaired_payload(chi_snapshot_fixture, field):
    system, basis, original = chi_snapshot_fixture
    result = copy.copy(original)
    values = getattr(original, field)
    if isinstance(values, list):
        values = [v.copy() for v in values]
        values[0].flat[0] += 0.05
    else:
        values = values.copy()
        values.flat[0] += 0.05
    setattr(result, field, values)
    with pytest.raises(ValueError):
        _chi_snapshot((system, basis, result))


def test_chi_snapshot_identity_binds_declared_provenance_and_exact_masks(chi_snapshot_fixture):
    a = _chi_snapshot(chi_snapshot_fixture)
    b = _chi_snapshot(chi_snapshot_fixture, declared_calculation_identity="b"*64)
    c = _chi_snapshot(chi_snapshot_fixture, frozen_core_bands=[1])
    assert a.state.numerical_payload_sha256 == b.state.numerical_payload_sha256
    assert a.state.calculation_identity != b.state.calculation_identity
    assert a.state.state_identity_sha256 != b.state.state_identity_sha256
    assert a.state.numerical_payload_sha256 != c.state.numerical_payload_sha256
    assert a.state.calculation_identity != c.state.calculation_identity


@pytest.mark.parametrize("change,message,error", [
    ({"backend": "ri"}, "four_center", NotImplementedError),
    ({"electronic_method": "RKS/PBE"}, "RHF", ValueError),
    ({"ecp_total_ncore": 2}, "all-electron", NotImplementedError),
    ({"effective_electron_count": 2}, "electron", ValueError),
    ({"smearing_temperature": 0.01}, "smearing", ValueError),
    ({"n_kpoints": 1}, "counts", ValueError),
    ({"mesh": (513, 1, 1)}, "diagnostic limit", ValueError),
    ({"mesh": (True, 1, 1)}, "mesh extent", ValueError),
])
def test_chi_snapshot_route_and_full_mesh_admission(chi_snapshot_fixture, change, message, error):
    system, basis, original = chi_snapshot_fixture
    result = copy.copy(original)
    result.aiccm2026dev_b = replace(original.aiccm2026dev_b, **change)
    with pytest.raises(error, match=message):
        _chi_snapshot((system, basis, result))


def test_chi_snapshot_rejects_reordered_k_payload_without_relabeling(chi_snapshot_fixture):
    system, basis, original = chi_snapshot_fixture
    result = copy.copy(original)
    result.kpoints_cart = original.kpoints_cart[::-1].copy()
    with pytest.raises(ValueError, match="k point"):
        _chi_snapshot((system, basis, result))


def test_chi_snapshot_native_weight_roundoff_is_canonicalized(chi_snapshot_fixture):
    system, basis, original = chi_snapshot_fixture
    result = copy.copy(original)
    result.kpoint_weights = np.nextafter(original.kpoint_weights, np.inf)
    snapshot = _chi_snapshot((system, basis, result))
    assert snapshot.state.numerical_payload_sha256 == _chi_snapshot(chi_snapshot_fixture).state.numerical_payload_sha256


@pytest.mark.parametrize("change,message", [
    (lambda b: b[:-1], "shape"),
    (lambda b: b.tolist(), "NumPy"),
    (lambda b: b.astype(str), "NumPy"),
    (lambda b: b * np.nan, "non-finite"),
])
def test_chi_snapshot_rejects_malformed_matrix_blocks(chi_snapshot_fixture, change, message):
    system, basis, original = chi_snapshot_fixture
    result = copy.copy(original)
    result.fock = [change(original.fock[0])] + original.fock[1:]
    with pytest.raises(ValueError, match=message):
        _chi_snapshot((system, basis, result))


def _cscl() -> vq.PeriodicSystem:
    lattice = np.eye(3) * 8.0
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(11, [0.0, 0.0, 0.0]), vq.Atom(17, [4.0, 4.0, 4.0])],
    )


def _diamond() -> vq.PeriodicSystem:
    lattice = np.eye(3) * 8.0
    fractional = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(6, (lattice @ position).tolist()) for position in fractional],
    )


def _operation_index(plan, rotation, translation=None) -> int:
    expected_rotation = np.asarray(rotation, dtype=np.int64)
    expected_translation = (
        None if translation is None else np.asarray(translation, dtype=float)
    )
    matches = [
        index
        for index, operation in enumerate(plan.operations)
        if np.array_equal(operation.rotation, expected_rotation)
        and (
            expected_translation is None
            or np.allclose(
                np.mod(operation.translation - expected_translation, 1.0),
                0.0,
                atol=1.0e-12,
            )
        )
    ]
    assert len(matches) == 1
    return matches[0]


def _test_ao_atom_indices(basis) -> np.ndarray:
    indices: list[int] = []
    for shell in basis.shells():
        indices.extend(
            [int(shell.atom_index)] * (2 * int(shell.l) + 1)
        )
    assert len(indices) == basis.nbasis
    return np.asarray(indices, dtype=np.int64)


def _dense_real_torus_ao_action(system, basis, plan, operation_index) -> np.ndarray:
    """Independent small-system oracle for the compact scatter."""

    operation = plan.operations[operation_index]
    mesh = tuple(int(value) for value in plan.mesh)
    n_cells = int(np.prod(mesh))
    nbasis = int(basis.nbasis)
    ao_atoms = _test_ao_atom_indices(basis)
    rotation_cart = lattice_to_cartesian_rotation(
        operation.rotation,
        np.asarray(system.lattice, dtype=float),
    )
    primitive = build_ao_permutation_matrix(
        basis,
        rotation_cart,
        operation.atom_permutation,
    )
    lattice = np.asarray(system.lattice, dtype=float)
    inverse_lattice = np.linalg.inv(lattice)
    raw_positions = np.asarray(
        [inverse_lattice @ np.asarray(atom.xyz) for atom in system.unit_cell]
    )
    physical_shifts = np.asarray(
        [
            np.rint(
                operation.rotation @ raw_positions[source_atom]
                + operation.translation
                - raw_positions[int(operation.atom_permutation[source_atom])]
            ).astype(np.int64)
            for source_atom in range(len(system.unit_cell))
        ]
    )
    dense = np.zeros((n_cells * nbasis, n_cells * nbasis))
    for source_cell, cell in enumerate(np.ndindex(mesh)):
        for source_atom in range(len(system.unit_cell)):
            source_aos = np.flatnonzero(ao_atoms == source_atom)
            image = (
                operation.rotation @ np.asarray(cell, dtype=np.int64)
                + physical_shifts[source_atom]
            )
            residue = np.mod(image, np.asarray(mesh, dtype=np.int64))
            destination_cell = int(
                (residue[0] * mesh[1] + residue[1]) * mesh[2]
                + residue[2]
            )
            destination_rows = np.arange(
                destination_cell * nbasis,
                (destination_cell + 1) * nbasis,
            )
            source_columns = source_cell * nbasis + source_aos
            dense[np.ix_(destination_rows, source_columns)] = primitive[
                :, source_aos
            ]
    return dense


def _translate_real_torus_rows(
    coefficients: np.ndarray,
    mesh: tuple[int, int, int],
    nbasis: int,
    shift: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(coefficients)
    for source_cell, cell in enumerate(np.ndindex(mesh)):
        destination = np.mod(
            np.asarray(cell, dtype=np.int64) + shift,
            np.asarray(mesh, dtype=np.int64),
        )
        destination_cell = int(
            (destination[0] * mesh[1] + destination[1]) * mesh[2]
            + destination[2]
        )
        result[
            destination_cell * nbasis : (destination_cell + 1) * nbasis
        ] = coefficients[source_cell * nbasis : (source_cell + 1) * nbasis]
    return result


def test_isotropic_cluster_preserves_full_cubic_group_and_builds_ibz() -> None:
    system = _cscl()
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    assert plan.space_group_number == 221
    assert plan.international_symbol == "Pm-3m"
    assert plan.point_group == "m-3m"
    assert plan.wyckoff_letters == ("a", "b")
    assert plan.site_symmetry_symbols == ("m-3m", "m-3m")
    assert plan.n_operations_full == 48
    assert plan.n_operations_compatible == 48
    assert plan.n_kpoints_full == 8
    assert 1 < plan.n_kpoints_irreducible < plan.n_kpoints_full
    assert plan.irreducible_weights.sum() == pytest.approx(1.0)
    counts = np.bincount(plan.full_to_irreducible)
    assert plan.irreducible_weights == pytest.approx(counts / 8.0)
    assert not plan.acceleration_applied

    # Independent native-spglib path used by the production k-point layer.
    vq.attach_symmetry(system)
    native = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=True)
    assert plan.full_to_irreducible == pytest.approx(native.ir_mapping)
    assert plan.irreducible_weights == pytest.approx(native.weights)


def test_anisotropic_cluster_uses_exact_compatible_subgroup_or_refuses() -> None:
    plan = build_aiccm2026dev_b_symmetry_plan(_cscl(), (2, 3, 4))
    assert plan.n_operations_compatible == 8
    assert len(plan.incompatible_operation_indices) == 40
    with pytest.raises(ValueError, match=r"N\^-1 W N integral"):
        build_aiccm2026dev_b_symmetry_plan(
            _cscl(),
            (2, 3, 4),
            require_full_space_group=True,
        )


def test_nonsymmorphic_diamond_maps_atoms_and_cells_without_dropping_shifts() -> None:
    system = _diamond()
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    assert plan.space_group_number == 227
    assert plan.international_symbol == "Fd-3m"
    assert plan.n_operations_full == 192
    assert plan.n_operations_compatible == 192
    nonsymmorphic = [
        index
        for index, operation in enumerate(plan.operations)
        if operation.has_fractional_translation
    ]
    assert nonsymmorphic
    operation_index = nonsymmorphic[0]
    operation = plan.operations[operation_index]
    assert operation.max_atom_mapping_residual_bohr < 1.0e-12
    for atom_index in range(len(system.unit_cell)):
        mapped_atom, mapped_cell = plan.map_cell(
            operation_index, atom_index, (1, 0, 1)
        )
        assert system.unit_cell[mapped_atom].Z == system.unit_cell[atom_index].Z
        assert all(0 <= value < 2 for value in mapped_cell)


def test_skew_column_lattice_maps_fractional_inversion_pair() -> None:
    lattice = np.asarray(
        [
            [7.0, 0.0, 0.0],
            [1.3, 6.2, 0.0],
            [0.7, 1.1, 5.4],
        ]
    )
    fractional = np.asarray([[0.13, 0.27, 0.31], [0.87, 0.73, 0.69]])
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, (lattice @ position).tolist()) for position in fractional],
    )
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    inversion = next(
        operation
        for operation in plan.operations
        if np.array_equal(operation.rotation, -np.eye(3, dtype=int))
    )
    assert inversion.atom_permutation.tolist() == [1, 0]
    assert inversion.max_atom_mapping_residual_bohr < 1.0e-12
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    pair_orbits = shell_pair_orbits(basis, plan)
    quartet_orbits = shell_quartet_orbits(basis, plan)
    assert sum(len(orbit) for orbit in pair_orbits) == 3
    assert len(pair_orbits) == 2
    assert sum(len(orbit) for orbit in quartet_orbits) == 6
    assert len(quartet_orbits) == 4


def test_compact_torus_ao_action_matches_dense_proper_and_improper_oracles() -> None:
    system = _cscl()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 3, 4))
    inversion_index = _operation_index(plan, -np.eye(3, dtype=np.int64))
    # The quarter turn is excluded by the anisotropic mesh, while inversion
    # survives.  Use the isotropic plan for the proper-rotation half.
    isotropic_plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    quarter_turn_index = _operation_index(
        isotropic_plan,
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    )

    random = np.random.default_rng(127)
    for active_plan, operation_index in (
        (plan, inversion_index),
        (isotropic_plan, quarter_turn_index),
    ):
        action = build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            active_plan,
            operation_index,
        )
        coefficients = random.normal(size=(action.torus_nbasis, 5)) + 1j * (
            random.normal(size=(action.torus_nbasis, 5))
        )
        dense = _dense_real_torus_ao_action(
            system,
            basis,
            active_plan,
            operation_index,
        )
        assert action.apply(coefficients) == pytest.approx(
            dense @ coefficients,
            abs=1.0e-13,
        )

    inversion = build_aiccm2026dev_b_real_torus_ao_action(
        system,
        basis,
        plan,
        inversion_index,
    )
    source_cell = (1 * 3 + 2) * 4 + 3
    assert inversion.cell_images[source_cell].tolist() == [17, 0]
    diagonal = np.diag(inversion.primitive_ao_action)
    assert np.any(diagonal == 1.0)
    assert np.any(diagonal == -1.0)

    quarter_turn = build_aiccm2026dev_b_real_torus_ao_action(
        system,
        basis,
        isotropic_plan,
        quarter_turn_index,
    )
    assert not np.allclose(
        quarter_turn.primitive_ao_action,
        quarter_turn.primitive_ao_action.T,
    )


def test_compact_torus_ao_action_uses_destination_by_source_ao_columns() -> None:
    system = _cscl()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    operation_index = _operation_index(
        plan,
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    )
    action = build_aiccm2026dev_b_real_torus_ao_action(
        system,
        basis,
        plan,
        operation_index,
    )
    source_cell = 5
    source_atom = 0
    # The first p shell on atom 0 occupies primitive AO rows 2:5.  Choosing
    # its first component makes the C4 action column differ from its row, so
    # this gate catches a source/destination transpose even without the dense
    # assembly helper.
    source_ao = 2
    coefficients = np.zeros((action.torus_nbasis, 1))
    coefficients[source_cell * action.nbasis + source_ao, 0] = 1.0
    transformed = action.apply(coefficients)[:, 0]
    destination_cell = int(action.cell_images[source_cell, source_atom])
    expected = np.zeros(action.torus_nbasis)
    expected[
        destination_cell * action.nbasis : (destination_cell + 1) * action.nbasis
    ] = action.primitive_ao_action[:, source_ao]
    transposed = np.zeros(action.torus_nbasis)
    transposed[
        destination_cell * action.nbasis : (destination_cell + 1) * action.nbasis
    ] = action.primitive_ao_action[source_ao, :]
    assert transformed == pytest.approx(expected, abs=1.0e-14)
    assert not np.allclose(expected, transposed)


def test_compact_torus_ao_action_obeys_identity_inverse_and_composition() -> None:
    system = _cscl()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 3, 4))
    anisotropic_rotations = {
        "identity": np.eye(3, dtype=np.int64),
        "inversion": -np.eye(3, dtype=np.int64),
    }
    actions = {
        name: build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            plan,
            _operation_index(plan, rotation),
        )
        for name, rotation in anisotropic_rotations.items()
    }
    coefficients = np.random.default_rng(9).normal(
        size=(actions["identity"].torus_nbasis, 4)
    )
    assert actions["identity"].apply(coefficients) == pytest.approx(
        coefficients,
        abs=1.0e-14,
    )
    assert actions["inversion"].apply(
        actions["inversion"].apply(coefficients)
    ) == pytest.approx(coefficients, abs=1.0e-13)

    isotropic_plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    c4z_rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c2x_rotation = np.diag([1, -1, -1])
    product_rotation = c4z_rotation @ c2x_rotation
    reverse_rotation = c2x_rotation @ c4z_rotation
    assert not np.array_equal(product_rotation, reverse_rotation)
    isotropic_actions = {
        name: build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            isotropic_plan,
            _operation_index(isotropic_plan, rotation),
        )
        for name, rotation in (
            ("c4z", c4z_rotation),
            ("c4z_inverse", c4z_rotation.T),
            ("c2x", c2x_rotation),
            ("product", product_rotation),
            ("reverse", reverse_rotation),
        )
    }
    isotropic_coefficients = coefficients[: isotropic_actions["c4z"].torus_nbasis]
    composed = isotropic_actions["c4z"].apply(
        isotropic_actions["c2x"].apply(isotropic_coefficients)
    )
    assert composed == pytest.approx(
        isotropic_actions["product"].apply(isotropic_coefficients),
        abs=1.0e-13,
    )
    assert not np.allclose(
        composed,
        isotropic_actions["reverse"].apply(isotropic_coefficients),
    )
    assert isotropic_actions["c4z_inverse"].apply(
        isotropic_actions["c4z"].apply(isotropic_coefficients)
    ) == pytest.approx(isotropic_coefficients, abs=1.0e-13)


def test_compact_torus_ao_action_carries_nonsymmorphic_cell_shifts() -> None:
    system = _diamond()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (3, 3, 3))
    operation_index = _operation_index(
        plan,
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [0.25, 0.25, 0.25],
    )
    operation = plan.operations[operation_index]
    assert operation.has_fractional_translation
    action = build_aiccm2026dev_b_real_torus_ao_action(
        system,
        basis,
        plan,
        operation_index,
    )
    source_cell = (1 * 3 + 0) * 3 + 1
    assert operation.atom_permutation[:2].tolist() == [4, 5]
    assert operation.atom_lattice_shifts[1].tolist() == [-1, 0, 0]
    assert action.cell_images[source_cell, :2].tolist() == [4, 22]
    coefficients = np.random.default_rng(83).normal(
        size=(action.torus_nbasis, 3)
    )
    dense = _dense_real_torus_ao_action(
        system,
        basis,
        plan,
        operation_index,
    )
    assert action.apply(coefficients) == pytest.approx(
        dense @ coefficients,
        abs=1.0e-13,
    )


def test_compact_torus_ao_action_corrects_unwrapped_atom_representatives() -> None:
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [8.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (3, 1, 1))
    inversion_index = _operation_index(plan, -np.eye(3, dtype=np.int64))
    action = build_aiccm2026dev_b_real_torus_ao_action(
        system,
        basis,
        plan,
        inversion_index,
    )
    assert plan.operations[inversion_index].atom_lattice_shifts.tolist() == [
        [0, 0, 0]
    ]
    assert action.atom_reference_cell_offsets.tolist() == [[1, 0, 0]]
    assert action.atom_lattice_shifts.tolist() == [[-2, 0, 0]]
    assert action.cell_images[:, 0].tolist() == [1, 0, 2]
    coefficients = np.arange(action.torus_nbasis * 2, dtype=float).reshape(
        action.torus_nbasis,
        2,
    )
    dense = _dense_real_torus_ao_action(
        system,
        basis,
        plan,
        inversion_index,
    )
    assert action.apply(coefficients) == pytest.approx(dense @ coefficients)


def test_compact_torus_ao_action_retains_nonsymmorphic_translation_cocycle() -> None:
    system = _diamond()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = (3, 3, 3)
    plan = build_aiccm2026dev_b_symmetry_plan(system, mesh)
    left_index, right_index = 1, 2
    left_operation = plan.operations[left_index]
    right_operation = plan.operations[right_index]
    assert (left_operation.full_group_index, right_operation.full_group_index) == (
        1,
        2,
    )
    product_rotation = left_operation.rotation @ right_operation.rotation
    product_translation = (
        left_operation.rotation @ right_operation.translation
        + left_operation.translation
    )
    product_index = _operation_index(
        plan,
        product_rotation,
        np.mod(product_translation, 1.0),
    )
    product_operation = plan.operations[product_index]
    cocycle = np.rint(
        product_translation - product_operation.translation
    ).astype(np.int64)
    assert product_operation.full_group_index == 3
    assert cocycle.tolist() == [-1, 0, 0]

    left = build_aiccm2026dev_b_real_torus_ao_action(
        system, basis, plan, left_index
    )
    right = build_aiccm2026dev_b_real_torus_ao_action(
        system, basis, plan, right_index
    )
    product = build_aiccm2026dev_b_real_torus_ao_action(
        system, basis, plan, product_index
    )
    coefficients = np.random.default_rng(991).normal(
        size=(left.torus_nbasis, 2)
    )
    expected = _translate_real_torus_rows(
        product.apply(coefficients),
        mesh,
        int(basis.nbasis),
        cocycle,
    )
    assert left.apply(right.apply(coefficients)) == pytest.approx(
        expected,
        abs=2.0e-13,
    )


def test_compact_torus_ao_action_preserves_an_invariant_metric_on_skew_cell() -> None:
    lattice = np.asarray(
        [
            [7.0, 0.0, 0.0],
            [1.3, 6.2, 0.0],
            [0.7, 1.1, 5.4],
        ]
    )
    fractional = np.asarray([[0.13, 0.27, 0.31], [0.87, 0.73, 0.69]])
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, (lattice @ position).tolist()) for position in fractional],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (3, 2, 1))
    operation_index = _operation_index(plan, -np.eye(3, dtype=np.int64))
    action = build_aiccm2026dev_b_real_torus_ao_action(
        system,
        basis,
        plan,
        operation_index,
    )
    dense = _dense_real_torus_ao_action(system, basis, plan, operation_index)
    assert action.atom_permutation.tolist() == [1, 0]
    assert action.atom_lattice_shifts.tolist() == [[-1, -1, -1], [-1, -1, -1]]
    assert action.cell_images[0].tolist() == [5, 5]
    assert action.cell_images[5].tolist() == [0, 0]

    random = np.random.default_rng(211)
    raw = random.normal(size=dense.shape)
    seed_metric = raw.T @ raw + np.eye(dense.shape[0])
    metric = 0.5 * (seed_metric + dense @ seed_metric @ dense.T)
    left = random.normal(size=(action.torus_nbasis, 2)) + 1j * random.normal(
        size=(action.torus_nbasis, 2)
    )
    right = random.normal(size=(action.torus_nbasis, 3)) + 1j * random.normal(
        size=(action.torus_nbasis, 3)
    )
    transformed_metric = dense.T @ metric @ dense
    assert transformed_metric == pytest.approx(metric, abs=1.0e-12)
    assert action.apply(left).conj().T @ metric @ action.apply(
        right
    ) == pytest.approx(left.conj().T @ metric @ right, abs=1.0e-12)
    assert action.apply(action.apply(left)) == pytest.approx(left, abs=1.0e-13)


def test_compact_torus_ao_action_is_factory_only_immutable_and_fail_closed() -> None:
    system = _cscl()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 3, 4))
    action = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, 1)
    repeated = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, 1)
    distinct = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, 2)

    with pytest.raises(TypeError, match="factory-only"):
        AICCM2026DevBRealTorusAOAction()
    assert not hasattr(action, "__dict__")
    with pytest.raises(TypeError):
        vars(action)
    for payload in (
        action.rotation,
        action.translation,
        action.atom_permutation,
        action.atom_lattice_shifts,
        action.atom_reference_cell_offsets,
        action.ao_atom_indices,
        action.primitive_ao_action,
        action.cell_images,
    ):
        assert not payload.flags.writeable
        with pytest.raises(ValueError):
            payload.setflags(write=True)
    assert action.fingerprint == repeated.fingerprint
    assert action.fingerprint != distinct.fingerprint
    assert len(action.fingerprint) == 64
    assert set(action.fingerprint) <= set("0123456789abcdef")
    assert action.payload_nbytes * 20 < action.torus_nbasis**2 * 8
    assert copy.copy(action) is action
    assert copy.deepcopy(action) is action

    with pytest.raises(ValueError, match="two-dimensional"):
        action.apply(np.zeros(action.torus_nbasis))
    with pytest.raises(ValueError, match="must have .* rows"):
        action.apply(np.zeros((action.torus_nbasis - 1, 2)))
    with pytest.raises(ValueError, match="must be numeric"):
        action.apply(np.full((action.torus_nbasis, 1), "x", dtype=object))
    with pytest.raises(ValueError, match="operation_index must be an integer"):
        build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, True)
    with pytest.raises(ValueError, match="outside the compatible"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            plan,
            len(plan.operations),
        )
    bad_mesh_plan = replace(plan, mesh=(2.9, 3, 4))
    with pytest.raises(ValueError, match="three positive integers"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system, basis, bad_mesh_plan, 0
        )
    inconsistent_mesh_plan = replace(plan, mesh=(4, 3, 2))
    with pytest.raises(ValueError, match="k-mesh metadata is inconsistent"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            inconsistent_mesh_plan,
            0,
        )
    huge_mesh_plan = replace(plan, mesh=(2**32, 2**32, 1))
    with pytest.raises(ValueError, match="addressable array bounds"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            huge_mesh_plan,
            0,
        )
    complex_kpoints = np.asarray(plan.full_kpoints_frac, dtype=np.complex128)
    complex_kpoints += 17.0j
    with pytest.raises(ValueError, match="k-mesh metadata is inconsistent"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            replace(plan, full_kpoints_frac=complex_kpoints),
            0,
        )
    string_kpoints = np.asarray(plan.full_kpoints_frac).astype(str)
    with pytest.raises(ValueError, match="k-mesh metadata is inconsistent"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            replace(plan, full_kpoints_frac=string_kpoints),
            0,
        )
    overflowing_rotation = np.eye(3, dtype=np.uint64)
    overflowing_rotation[0, 0] = np.iinfo(np.uint64).max
    overflowing_operation = replace(
        plan.operations[0],
        rotation=overflowing_rotation,
    )
    overflowing_plan = replace(
        plan,
        operations=(overflowing_operation, *plan.operations[1:]),
    )
    with pytest.raises(ValueError, match="outside the signed int64 range"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            overflowing_plan,
            0,
        )
    malformed_operation = replace(
        plan.operations[0],
        rotation=np.eye(2, dtype=np.int64),
    )
    malformed_plan = replace(
        plan,
        operations=(malformed_operation, *plan.operations[1:]),
    )
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            malformed_plan,
            0,
        )
    isotropic_plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    c4_operation = isotropic_plan.operations[
        _operation_index(
            isotropic_plan,
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        )
    ]
    incompatible_plan = replace(
        plan,
        operations=(c4_operation, *plan.operations[1:]),
    )
    with pytest.raises(ValueError, match="incompatible with the finite mesh"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            incompatible_plan,
            0,
        )
    forged_operation = replace(
        plan.operations[0],
        translation=np.asarray([0.1, 0.0, 0.0]),
        max_atom_mapping_residual_bohr=0.8,
        has_fractional_translation=True,
    )
    forged_plan = replace(
        plan,
        operations=(forged_operation, *plan.operations[1:]),
    )
    with pytest.raises(ValueError, match="exceeds .* admission tolerance"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            forged_plan,
            0,
        )
    complex_translation = replace(
        plan.operations[0],
        translation=np.asarray([0.0 + 9.0j, 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="real-floating 3-vector"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            replace(
                plan,
                operations=(complex_translation, *plan.operations[1:]),
            ),
            0,
        )
    string_translation = replace(
        plan.operations[0],
        translation=np.asarray(["0", "0", "0"]),
    )
    with pytest.raises(ValueError, match="real-floating 3-vector"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            replace(
                plan,
                operations=(string_translation, *plan.operations[1:]),
            ),
            0,
        )
    string_fractional_flag = replace(
        plan.operations[0],
        has_fractional_translation="false",
    )
    with pytest.raises(ValueError, match="metadata must be boolean"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            replace(
                plan,
                operations=(string_fractional_flag, *plan.operations[1:]),
            ),
            0,
        )
    string_residual = replace(
        plan.operations[0],
        max_atom_mapping_residual_bohr="0.0",
    )
    with pytest.raises(ValueError, match="residual must be a real scalar"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            replace(
                plan,
                operations=(string_residual, *plan.operations[1:]),
            ),
            0,
        )
    forged_index = replace(
        plan.operations[0],
        full_group_index=plan.n_operations_full,
    )
    forged_index_plan = replace(
        plan,
        operations=(forged_index, *plan.operations[1:]),
    )
    with pytest.raises(ValueError, match="outside the full space group"):
        build_aiccm2026dev_b_real_torus_ao_action(
            system,
            basis,
            forged_index_plan,
            0,
        )

    shifted_system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(11, [0.1, 0.0, 0.0]), vq.Atom(17, [4.0, 4.0, 4.0])],
    )
    with pytest.raises(ValueError, match="basis shell origin does not match"):
        build_aiccm2026dev_b_real_torus_ao_action(
            shifted_system,
            basis,
            plan,
            0,
        )

    pair_system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(2, [2.0, 0.0, 0.0]), vq.Atom(2, [6.0, 0.0, 0.0])],
    )
    pair_basis = vq.BasisSet(pair_system.unit_cell_molecule(), "sto-3g")
    modified_shells = []
    modified = False
    for original in pair_basis.shells():
        shell = vq.ShellInfo()
        shell.atom_index = original.atom_index
        shell.l = original.l
        shell.pure = original.pure
        shell.origin = original.origin
        exponents = list(original.exponents)
        shell.coefficients = list(original.coefficients)
        if int(shell.atom_index) == 1 and not modified:
            exponents[0] *= 1.01
            modified = True
        shell.exponents = exponents
        modified_shells.append(shell)
    asymmetric_basis = vq.BasisSet(
        pair_system.unit_cell_molecule(),
        modified_shells,
        "asymmetric-he-pair",
        True,
    )
    pair_plan = build_aiccm2026dev_b_symmetry_plan(pair_system, (2, 2, 2))
    pair_inversion = next(
        index
        for index, operation in enumerate(pair_plan.operations)
        if np.array_equal(operation.rotation, -np.eye(3, dtype=np.int64))
        and operation.atom_permutation.tolist() == [1, 0]
    )
    with pytest.raises(ValueError, match="different ordered radial AO shells"):
        build_aiccm2026dev_b_real_torus_ao_action(
            pair_system,
            asymmetric_basis,
            pair_plan,
            pair_inversion,
        )
    other_system = _diamond()
    other_basis = vq.BasisSet(other_system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="symmetry atom mapping"):
        build_aiccm2026dev_b_real_torus_ao_action(
            other_system,
            other_basis,
            plan,
            0,
        )

    cartesian_shell = vq.ShellInfo()
    cartesian_shell.atom_index = 0
    cartesian_shell.l = 1
    cartesian_shell.pure = False
    cartesian_shell.exponents = [0.8]
    cartesian_shell.coefficients = [1.0]
    cartesian_shell.origin = [0.0, 0.0, 0.0]
    one_atom_system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    cartesian_basis = vq.BasisSet(
        one_atom_system.unit_cell_molecule(),
        [cartesian_shell],
        "cartesian-p",
        False,
    )
    one_atom_plan = build_aiccm2026dev_b_symmetry_plan(
        one_atom_system,
        (1, 1, 1),
    )
    with pytest.raises(NotImplementedError, match="Cartesian p-and-higher"):
        build_aiccm2026dev_b_real_torus_ao_action(
            one_atom_system,
            cartesian_basis,
            one_atom_plan,
            0,
        )


def test_gamma_reynolds_projection_is_idempotent_and_invariant() -> None:
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(6, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (1, 1, 1))
    random = np.random.default_rng(7).normal(size=(basis.nbasis, basis.nbasis))
    matrix = random + random.T
    projected = symmetrize_gamma_ao_matrix(matrix, system, basis, plan)
    projected_twice = symmetrize_gamma_ao_matrix(projected, system, basis, plan)
    assert projected_twice == pytest.approx(projected, abs=1.0e-13)
    assert gamma_matrix_symmetry_residual(projected, system, basis, plan) < 1.0e-12


def _fake_helium_result(kpoints) -> PeriodicKRHFGDFResult:
    overlap = [np.ones((1, 1), dtype=complex)]
    density = [np.asarray([[2.0 + 0.0j]])]
    zeros = [np.zeros((1, 1), dtype=complex)]
    return PeriodicKRHFGDFResult(
        energy=-2.0,
        e_electronic=-2.0,
        e_nuclear=0.0,
        n_iter=1,
        converged=True,
        mo_energies=[np.asarray([-1.0])],
        mo_coeffs=[np.ones((1, 1), dtype=complex)],
        fock=zeros,
        overlap=overlap,
        hcore=zeros,
        density=density,
        kpoints_cart=np.asarray(kpoints.kpoints_cart),
        kpoint_weights=np.asarray(kpoints.weights),
        backend="native-multi-k-gdf-gdf-rhf",
    )


def test_diagnostic_mode_attaches_witness_without_changing_result(monkeypatch) -> None:
    lattice = np.eye(3) * 10.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def fake_gdf(_system, _basis, kpoints, _options, **_kwargs):
        return _fake_helium_result(kpoints)

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf", fake_gdf
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (1, 1, 1),
            backend="ri",
            symmetry_mode="diagnostic",
            progress=False,
        )
    assert result.energy == pytest.approx(-2.0)
    diagnostics = result.aiccm2026dev_b_symmetry
    assert diagnostics.energy_change_hartree == 0.0
    assert diagnostics.gamma_fock_residual < 1.0e-14
    assert diagnostics.gamma_density_residual < 1.0e-14
    assert diagnostics.n_unique_shell_pairs <= diagnostics.n_shell_pairs
    assert diagnostics.n_unique_shell_quartets <= diagnostics.n_shell_quartets
    assert not diagnostics.plan.acceleration_applied


def test_off_mode_is_structurally_unchanged_and_integral_mode_fails_closed(
    monkeypatch,
) -> None:
    lattice = np.eye(3) * 10.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    calls = 0

    def fake_gdf(_system, _basis, kpoints, _options, **_kwargs):
        nonlocal calls
        calls += 1
        return _fake_helium_result(kpoints)

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf", fake_gdf
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system, basis, backend="ri", symmetry_mode="off", progress=False
        )
    assert not hasattr(result, "aiccm2026dev_b_symmetry")
    assert calls == 1

    with pytest.warns(AICCM2026DevBExperimentalWarning), pytest.raises(
        NotImplementedError, match="petite-list"
    ):
        run_aiccm2026dev_b_rhf(
            system, basis, backend="ri", symmetry_mode="integrals", progress=False
        )
    assert calls == 1


def test_periodic_runner_threads_b_only_symmetry_option(monkeypatch, tmp_path) -> None:
    lattice = np.eye(3) * 10.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def fake_gdf(_system, _basis, kpoints, _options, **_kwargs):
        return _fake_helium_result(kpoints)

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf", fake_gdf
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_periodic_job(
            system,
            basis,
            method="aiccm",
            variant="chi",
            aiccm_backend="ri",
            aiccm_symmetry="diagnostic",
            convergence="off",
            output=tmp_path / "b-symmetry",
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )
    assert result.aiccm2026dev_b_symmetry.plan.space_group_number == 221
    output = (tmp_path / "b-symmetry.out").read_text()
    assert "AICCM2026DEV-B space-group diagnostic" in output
    assert "diagnostic only (not applied)" in output
    assert "shell quartets" in output


def test_low_dimensional_spglib_path_fails_closed() -> None:
    lattice = np.eye(3) * 10.0
    system = vq.PeriodicSystem(2, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    with pytest.raises(NotImplementedError, match="layer and rod groups"):
        build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 1))


def test_plan_k_mesh_matches_b_stream_full_mesh_as_a_set() -> None:
    system = _cscl()
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    kpoints = cyclic_gamma_mesh(system, (2, 2, 2))
    wrapped_plan = np.mod(plan.full_kpoints_frac, 1.0)
    wrapped_driver = np.mod(kpoints.kpoints_frac, 1.0)
    assert {
        tuple(np.round(point, 12)) for point in wrapped_plan
    } == {tuple(np.round(point, 12)) for point in wrapped_driver}


def test_real_gamma_scf_is_bit_identical_with_diagnostics_enabled() -> None:
    lattice = np.eye(3) * 16.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.max_iter = 30
    options.conv_tol_energy = 1.0e-9
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        symmetry_off = run_aiccm2026dev_b_rhf(
            system,
            basis,
            options=options,
            backend="four_center",
            symmetry_mode="off",
            progress=False,
        )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        diagnostic = run_aiccm2026dev_b_rhf(
            system,
            basis,
            options=options,
            backend="four_center",
            symmetry_mode="diagnostic",
            progress=False,
        )
    assert symmetry_off.converged and diagnostic.converged
    assert diagnostic.energy == symmetry_off.energy
    assert np.array_equal(np.asarray(diagnostic.fock), np.asarray(symmetry_off.fock))
    witness = diagnostic.aiccm2026dev_b_symmetry
    assert witness.energy_change_hartree == 0.0
    assert witness.gamma_fock_residual == 0.0
    assert witness.gamma_density_residual == 0.0


@pytest.fixture(scope="module")
def occupied_action_fixture():
    """Real four-center chi SCF -> localizer -> physical S/C, on three cells."""
    from vibeqc.periodic.chi.localization import localize_aiccm2026dev_b_occupied

    system = vq.PeriodicSystem(
        3, np.diag([8.0, 16.0, 16.0]), [vq.Atom(2, [0.0, 0.0, 0.0])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = (3, 1, 1)
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        scf = run_aiccm2026dev_b_rhf(
            system, basis, mesh, backend="four_center", progress=False,
        )
    assert scf.converged
    localized = localize_aiccm2026dev_b_occupied(scf, system, basis)
    plan = build_aiccm2026dev_b_symmetry_plan(system, mesh)
    index = _operation_index(plan, -np.eye(3, dtype=int))
    action = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, index)
    return system, basis, plan, index, action, localized


def test_occupied_action_uses_physical_overlap_and_localized_coefficients(
    occupied_action_fixture,
) -> None:
    system, basis, plan, index, action, localized = occupied_action_fixture
    result = build_aiccm2026dev_b_occupied_symmetry_action(action, localized)
    dense = _dense_real_torus_ao_action(system, basis, plan, index)
    expected = localized.coefficients.conj().T @ localized.overlap @ dense @ localized.coefficients
    np.testing.assert_allclose(result.matrix, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        dense @ localized.coefficients, localized.coefficients @ result.matrix,
        atol=1e-12, rtol=0,
    )
    np.testing.assert_allclose(result.matrix @ result.matrix, np.eye(3), atol=1e-12)
    assert result.is_monomial
    assert result.metric_covariance_residual < 1e-12
    assert result.orthonormality_residual < 1e-12
    assert result.closure_residual < 1e-12
    assert result.unitarity_residual < 1e-12
    assert result.ao_action_fingerprint == action.fingerprint


def test_occupied_action_preserves_complex_phases_and_general_mixing(
    occupied_action_fixture,
) -> None:
    *_, action, localized = occupied_action_fixture
    base = build_aiccm2026dev_b_occupied_symmetry_action(action, localized)
    phases = np.exp(1j * np.asarray([0.17, 0.73, -0.49]))
    phased = replace(localized, coefficients=localized.coefficients * phases)
    result = build_aiccm2026dev_b_occupied_symmetry_action(action, phased)
    expected = phases.conj()[:, None] * base.matrix * phases[None, :]
    np.testing.assert_allclose(result.matrix, expected, atol=1e-12)
    assert result.is_monomial
    np.testing.assert_array_equal(result.permutation, base.permutation)
    reconstructed = np.zeros((3, 3), dtype=complex)
    reconstructed[result.permutation, np.arange(3)] = result.phases
    np.testing.assert_allclose(result.matrix, reconstructed, atol=1e-12)
    assert np.max(np.abs(result.phases.imag)) > 0.1
    assert result.input_fingerprint != base.input_fingerprint
    assert result.fingerprint != base.fingerprint

    rng = np.random.default_rng(128)
    gauge, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    mixed = replace(localized, coefficients=localized.coefficients @ gauge)
    result = build_aiccm2026dev_b_occupied_symmetry_action(action, mixed)
    np.testing.assert_allclose(result.matrix, gauge.conj().T @ base.matrix @ gauge, atol=1e-12)
    assert not result.is_monomial
    assert result.permutation is None and result.phases is None
    assert result.monomial_residual > 0.1
    assert result.closure_residual < 1e-12


def test_occupied_action_snapshot_identity_and_immutable_payload(
    occupied_action_fixture,
) -> None:
    *_, action, original = occupied_action_fixture
    localized = replace(original, coefficients=original.coefficients.copy(), overlap=original.overlap.copy())
    result = build_aiccm2026dev_b_occupied_symmetry_action(action, localized)
    repeated = build_aiccm2026dev_b_occupied_symmetry_action(action, localized)
    assert repeated.fingerprint == result.fingerprint
    assert copy.copy(result) is result and copy.deepcopy(result) is result
    with pytest.raises(TypeError, match="factory-only"):
        AICCM2026DevBOccupiedSymmetryAction()
    for array in (result.matrix, result.permutation, result.phases):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    stored = result.matrix.copy()
    localized.coefficients[:] = 0
    localized.overlap[:] = 0
    np.testing.assert_array_equal(result.matrix, stored)
    scaled = replace(original, coefficients=original.coefficients / np.sqrt(2), overlap=original.overlap * 2)
    changed = build_aiccm2026dev_b_occupied_symmetry_action(action, scaled)
    np.testing.assert_allclose(changed.matrix, stored, atol=1e-12)
    assert changed.input_fingerprint != result.input_fingerprint
    assert changed.fingerprint != result.fingerprint
    from vibeqc.output.citations.registry import load_default_database

    citations = load_default_database().assemble(
        method="rhf", basis="sto-3g", numerics=result.citation_numerics,
    )
    assert "casassa_symmetry_wannier_2006" in {item.key for item in citations.citations}


@pytest.mark.parametrize("value", [True, 0, -1, 1e-3, float("nan"), float("inf"), 1j, "1e-8"])
def test_occupied_action_rejects_invalid_tolerance(occupied_action_fixture, value) -> None:
    *_, action, localized = occupied_action_fixture
    with pytest.raises(ValueError, match="tolerance"):
        build_aiccm2026dev_b_occupied_symmetry_action(action, localized, tolerance=value)


def test_occupied_action_memory_admission_precedes_transform(
    occupied_action_fixture, monkeypatch,
) -> None:
    *_, action, localized = occupied_action_fixture
    admitted = build_aiccm2026dev_b_occupied_symmetry_action(action, localized)
    exact_cap = admitted.estimated_workspace_bytes
    assert build_aiccm2026dev_b_occupied_symmetry_action(
        action, localized, max_workspace_bytes=exact_cap
    ).fingerprint == admitted.fingerprint
    def forbidden(*args, **kwargs):
        pytest.fail("AO transformation reached before memory admission")
    monkeypatch.setattr(AICCM2026DevBRealTorusAOAction, "apply", forbidden)
    with pytest.raises(MemoryError, match="estimate"):
        build_aiccm2026dev_b_occupied_symmetry_action(action, localized, max_workspace_bytes=exact_cap - 1)
    for cap in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="max_workspace_bytes"):
            build_aiccm2026dev_b_occupied_symmetry_action(action, localized, max_workspace_bytes=cap)


@pytest.mark.parametrize("mutation, message", [
    (lambda loc: replace(loc, coefficients=loc.coefficients[:, :1]), "coefficients"),
    (lambda loc: replace(loc, coefficients=loc.coefficients.tolist()), "NumPy"),
    (lambda loc: replace(loc, coefficients=loc.coefficients.astype(str)), "dtype"),
    (lambda loc: replace(loc, coefficients=loc.coefficients * np.nan), "finite"),
    (lambda loc: replace(loc, coefficients=loc.coefficients * 1.1), "orthonormal"),
    (lambda loc: replace(loc, overlap=-loc.overlap), "positive definite"),
    (lambda loc: replace(loc, overlap=np.zeros_like(loc.overlap)), "nonzero"),
    (lambda loc: replace(loc, overlap=loc.overlap + np.diag([0., 0.1, 0.3])), "preserve"),
    (lambda loc: replace(loc, overlap=loc.overlap + np.triu(np.ones((3, 3)), 1)), "Hermitian"),
    (lambda loc: replace(loc, translations=loc.translations[::-1]), "C-order"),
    (lambda loc: replace(loc, translations=loc.translations.astype(float)), "dtype"),
    (lambda loc: replace(loc, n_cells=True), "n_cells"),
    (lambda loc: replace(loc, n_cells=2), "cell count"),
    (lambda loc: replace(loc, n_occ_per_cell=2), "dimension"),
])
def test_occupied_action_rejects_invalid_localized_state(occupied_action_fixture, mutation, message) -> None:
    *_, action, localized = occupied_action_fixture
    with pytest.raises(ValueError, match=message):
        build_aiccm2026dev_b_occupied_symmetry_action(action, mutation(localized))


def test_occupied_action_rejects_an_unclosed_occupied_subspace(occupied_action_fixture) -> None:
    *_, _, template = occupied_action_fixture
    system = vq.PeriodicSystem(
        3, np.eye(3) * 16, [vq.Atom(1, [-0.7, 0, 0]), vq.Atom(1, [0.7, 0, 0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (1, 1, 1))
    action = build_aiccm2026dev_b_real_torus_ao_action(
        system, basis, plan, _operation_index(plan, -np.eye(3, dtype=int)),
    )
    # Inversion preserves S but maps the sole occupied source AO to an
    # excluded AO. Orthogonality alone must not admit this state.
    invalid = replace(template, n_cells=1, n_occ_per_cell=1,
                      coefficients=np.array([[1.], [0.]]), overlap=np.eye(2),
                      translations=np.zeros((1, 3), dtype=int))
    with pytest.raises(ValueError, match="does not close"):
        build_aiccm2026dev_b_occupied_symmetry_action(action, invalid)


def test_occupied_action_real_h2_occupied_subspace_and_metric(occupied_action_fixture) -> None:
    from vibeqc.periodic.chi.localization import localize_aiccm2026dev_b_occupied

    system = vq.PeriodicSystem(
        3, np.diag([8., 16., 16.]),
        [vq.Atom(1, [-0.7, 0., 0.]), vq.Atom(1, [0.7, 0., 0.])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = (3, 1, 1)
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        scf = run_aiccm2026dev_b_rhf(system, basis, mesh, backend="four_center", progress=False)
    assert scf.converged
    localized = localize_aiccm2026dev_b_occupied(scf, system, basis)
    assert localized.coefficients.shape == (6, 3)
    assert np.max(np.abs(localized.overlap - np.eye(6))) > 0.1
    plan = build_aiccm2026dev_b_symmetry_plan(system, mesh)
    index = _operation_index(plan, -np.eye(3, dtype=int))
    action = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, index)
    result = build_aiccm2026dev_b_occupied_symmetry_action(action, localized)
    dense = _dense_real_torus_ao_action(system, basis, plan, index)
    expected = localized.coefficients.conj().T @ localized.overlap @ dense @ localized.coefficients
    np.testing.assert_allclose(result.matrix, expected, atol=1e-12, rtol=0)
    assert result.closure_residual < 1e-12
    # Boys localization need not produce exact scalar partners. Preserve the
    # measured action and its classification rather than imposing that gauge.
    assert result.is_monomial == (result.monomial_residual <= result.tolerance)


def test_occupied_action_noncommuting_p_shell_rotations(occupied_action_fixture) -> None:
    from vibeqc._vibeqc_core import compute_overlap

    *_, template = occupied_action_fixture
    system = vq.PeriodicSystem(3, np.eye(3) * 16, [vq.Atom(10, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    # Genuine Gaussian AO overlap, complete orthonormal test subspace.
    overlap = np.asarray(compute_overlap(basis))
    eigenvalues, vectors = np.linalg.eigh(overlap)
    coefficients = (vectors * eigenvalues**-0.5) @ vectors.T
    loc = replace(template, coefficients=coefficients, overlap=overlap,
                  n_cells=1, n_occ_per_cell=basis.nbasis,
                  translations=np.zeros((1, 3), dtype=int))
    plan = build_aiccm2026dev_b_symmetry_plan(system, (1, 1, 1))
    g = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    h = np.diag([1, -1, -1])
    matrices = []
    for rotation in (g, h, g @ h, h @ g, g.T):
        index = _operation_index(plan, rotation)
        action = build_aiccm2026dev_b_real_torus_ao_action(system, basis, plan, index)
        result = build_aiccm2026dev_b_occupied_symmetry_action(action, loc)
        dense = _dense_real_torus_ao_action(system, basis, plan, index)
        np.testing.assert_allclose(result.matrix, coefficients.T @ overlap @ dense @ coefficients, atol=1e-12)
        matrices.append(result.matrix)
    np.testing.assert_allclose(matrices[0] @ matrices[1], matrices[2], atol=1e-12)
    np.testing.assert_allclose(matrices[1] @ matrices[0], matrices[3], atol=1e-12)
    np.testing.assert_allclose(matrices[0] @ matrices[4], np.eye(basis.nbasis), atol=1e-12)
    assert not np.allclose(matrices[2], matrices[3])


@pytest.fixture(scope="module")
def diamond_spatial_group():
    lattice = np.array([[0., 4., 4.], [4., 0., 4.], [4., 4., 0.]])
    system = vq.PeriodicSystem(3, lattice, [
        vq.Atom(6, [0., 0., 0.]), vq.Atom(6, lattice @ np.full(3, 0.25)),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    group = build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)
    return system, basis, plan, group


def test_spatial_group_all_diamond_products_match_independent_seitz_and_ao_scatter(
    diamond_spatial_group,
) -> None:
    system, basis, plan, group = diamond_spatial_group
    assert len(group.actions) == 48
    assert np.any(group.lattice_cocycle)
    assert group.group_products_checked == 48**2 + 48**3
    rng = np.random.default_rng(129)
    c = rng.normal(size=(80, 2)) + 1j*rng.normal(size=(80, 2))
    # Materialize independent scatters ONCE, never use the group's lookup
    # as the oracle for spatial composition.
    dense = [_dense_real_torus_ao_action(system, basis, plan, i) for i in range(48)]
    for g, ag in enumerate(group.actions):
        for h, ah in enumerate(group.actions):
            k = int(group.products[g, h])
            ell = group.lattice_cocycle[g, h]
            np.testing.assert_array_equal(ag.rotation @ ah.rotation, group.actions[k].rotation)
            np.testing.assert_allclose(ag.rotation @ ah.translation + ag.translation,
                                       group.actions[k].translation + ell, atol=1e-14)
            right = (dense[k] @ c).reshape(2, 2, 2, basis.nbasis, 2)
            right = np.roll(right, tuple(ell), axis=(0, 1, 2)).reshape(80, 2)
            np.testing.assert_allclose(dense[g] @ (dense[h] @ c), right, atol=3e-12)
    assert group.maximum_ao_product_residual < 1e-12
    assert group.maximum_seitz_residual_bohr < 1e-12


@pytest.mark.parametrize("extent", [2, 3])
def test_spatial_group_cocycle_native_multik_phase(diamond_spatial_group, extent) -> None:
    from vibeqc import _vibeqc_core as core

    system, basis, _, group = diamond_spatial_group
    if extent != 2:
        plan = build_aiccm2026dev_b_symmetry_plan(system, (extent, extent, extent))
        group = build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)
    mesh = core._RegularKMesh((extent, extent, extent))
    source = len(mesh) - 2
    rng = np.random.default_rng(128)
    c = np.ascontiguousarray(rng.normal(size=(basis.nbasis, 2))
                             + 1j*rng.normal(size=(basis.nbasis, 2)))
    # Explicit tiny native diagnostic admission; no dependency on pytest's
    # import mode or another test module's private helpers.
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
    for field in ("maximum_atoms", "maximum_shells", "maximum_contractions",
                  "maximum_basis_functions", "maximum_columns"):
        setattr(caps, field, 512)
    caps.maximum_basis_numeric_lanes = 10000
    for field in ("maximum_borrowed_numerical_bytes", "maximum_owned_numerical_bytes",
                  "maximum_control_storage_bytes", "maximum_per_replica_inventoried_bytes",
                  "maximum_node_inventoried_bytes"):
        setattr(caps, field, 16 << 20)
    caps.maximum_work_units = 10**9
    controls = options, inventory, caps
    def apply(index, source_index, coefficients):
        action = group.actions[index]
        op = core._make_periodic_ao_bloch_operation(
            np.ascontiguousarray(action.rotation, dtype=np.int32),
            np.ascontiguousarray(action.translation, dtype=np.float64),
        )
        return core._apply_periodic_ao_bloch_operation(
            basis, system, op, mesh, source_index, False,
            np.ascontiguousarray(coefficients), *controls,
        )
    nontrivial = complex_phases = 0
    for g in range(len(group.actions)):
        for h in range(len(group.actions)):
            # Test every pair carrying a nontrivial phase at this character.
            k = int(group.products[g, h])
            ell = group.lattice_cocycle[g, h]
            if not np.any(ell):
                continue
            direct = apply(k, source, c)
            q = np.asarray(mesh.fractional_at(direct.memory.target_index))
            phase = np.exp(-2j*np.pi*np.dot(q, ell))
            if abs(phase-1) < 1e-12:
                continue
            first = apply(h, source, c)
            composed = apply(g, first.memory.target_index, first.coefficients_copy())
            assert composed.memory.target_index == direct.memory.target_index
            np.testing.assert_allclose(composed.coefficients_copy(),
                                       phase*direct.coefficients_copy(), atol=3e-12)
            nontrivial += 1
            complex_phases += abs(phase.imag) > 0.1
    assert nontrivial > 100  # Not a Gamma-only or zero-cocycle witness.
    if extent == 3:
        assert complex_phases > 100  # Distinguishes exp(+ik.ell) from exp(-ik.ell).


def test_spatial_group_snapshot_and_reordered_operation_indices(diamond_spatial_group) -> None:
    system, basis, plan, group = diamond_spatial_group
    assert copy.copy(group) is group and copy.deepcopy(group) is group
    for array in (group.products, group.lattice_cocycle, group.inverses):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError, match="factory-only"):
        AICCM2026DevBTorusSymmetryGroup()
    reordered = replace(plan, operations=plan.operations[::-1])
    other = build_aiccm2026dev_b_torus_symmetry_group(system, basis, reordered)
    assert other.identity_index == 47 - group.identity_index
    np.testing.assert_array_equal(other.products, 47-group.products[::-1, ::-1])
    np.testing.assert_array_equal(other.lattice_cocycle, group.lattice_cocycle[::-1, ::-1])
    assert other.fingerprint != group.fingerprint  # binds enumeration, not abstract group


def test_spatial_group_unwrapped_representatives_preserve_cocycle(diamond_spatial_group) -> None:
    original, _, _, group = diamond_spatial_group
    lattice = np.asarray(original.lattice)
    system = vq.PeriodicSystem(3, lattice, [
        vq.Atom(6, np.asarray(original.unit_cell[0].xyz) + lattice @ [1, -1, 2]),
        vq.Atom(6, np.asarray(original.unit_cell[1].xyz) + lattice @ [-2, 1, 1]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (2, 2, 2))
    unwrapped = build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)
    np.testing.assert_array_equal(unwrapped.products, group.products)
    np.testing.assert_array_equal(unwrapped.lattice_cocycle, group.lattice_cocycle)
    assert unwrapped.fingerprint != group.fingerprint


def test_spatial_group_changed_seitz_representative_has_exact_coboundary(diamond_spatial_group) -> None:
    system, basis, plan, group = diamond_spatial_group
    offsets = np.zeros((48, 3), dtype=int)
    selected = (group.identity_index + 1) % 48
    offsets[selected] = [1, -2, 1]
    operations = tuple(replace(op, translation=op.translation + offset,
                               atom_lattice_shifts=op.atom_lattice_shifts + offset)
                       for op, offset in zip(plan.operations, offsets))
    changed = build_aiccm2026dev_b_torus_symmetry_group(system, basis, replace(plan, operations=operations))
    np.testing.assert_array_equal(changed.products, group.products)
    for g in range(48):
        for h in range(48):
            k = group.products[g, h]
            expected = group.lattice_cocycle[g, h] + offsets[g] + group.actions[g].rotation @ offsets[h] - offsets[k]
            np.testing.assert_array_equal(changed.lattice_cocycle[g, h], expected)
    assert changed.fingerprint != group.fingerprint


def test_spatial_group_rejects_incomplete_duplicate_and_unnormalized_cosets(
    occupied_action_fixture,
) -> None:
    system, basis, _, _, _, _ = occupied_action_fixture
    plan = build_aiccm2026dev_b_symmetry_plan(system, (1, 1, 1))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Use cubic geometry for a fourfold subgroup.
    system = vq.PeriodicSystem(3, np.eye(3)*8, [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (1, 1, 1))
    ops = tuple(plan.operations[_operation_index(plan, np.linalg.matrix_power(rotation, j))]
                for j in range(4))
    valid = replace(plan, operations=ops)
    assert len(build_aiccm2026dev_b_torus_symmetry_group(system, basis, valid).actions) == 4
    for selected, message in ((ops[:3], "not closed"), (ops[1:], "identity"),
                              (ops + ops[:1], "duplicate")):
        with pytest.raises(ValueError, match=message):
            build_aiccm2026dev_b_torus_symmetry_group(system, basis, replace(plan, operations=selected))
    shifted_identity = replace(ops[0], translation=np.array([1., 0., 0.]),
                               atom_lattice_shifts=ops[0].atom_lattice_shifts + [1, 0, 0])
    with pytest.raises(ValueError, match="normalized"):
        build_aiccm2026dev_b_torus_symmetry_group(
            system, basis, replace(plan, operations=(shifted_identity, *ops[1:])),
        )


def test_group_admission_precedes_action_factory(diamond_spatial_group, monkeypatch) -> None:
    import vibeqc.periodic.chi.symmetry as module

    system, basis, plan, group = diamond_spatial_group
    exact = build_aiccm2026dev_b_torus_symmetry_group(
        system, basis, plan, max_workspace_bytes=group.estimated_workspace_bytes,
        max_group_products=group.group_products_checked,
    )
    assert exact.fingerprint == group.fingerprint
    def forbidden(*args, **kwargs):
        pytest.fail("group admission reached action factory")
    monkeypatch.setattr(module, "build_aiccm2026dev_b_real_torus_ao_action", forbidden)
    with pytest.raises(MemoryError, match="estimate"):
        module.build_aiccm2026dev_b_torus_symmetry_group(
            system, basis, plan, max_workspace_bytes=group.estimated_workspace_bytes-1,
        )
    with pytest.raises(ValueError, match="max_group_products"):
        module.build_aiccm2026dev_b_torus_symmetry_group(
            system, basis, plan, max_group_products=group.group_products_checked-1,
        )


def test_occupied_group_physical_multicell_localizer_and_gauge(occupied_action_fixture) -> None:
    system, basis, plan, _, _, localized = occupied_action_fixture
    group = build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)
    result = build_aiccm2026dev_b_occupied_group_witness(group, localized)
    assert result.maximum_group_residual < 1e-12
    assert result.maximum_translation_residual < 1e-12
    assert len(result.occupied_actions) == len(plan.operations)
    assert copy.copy(result) is result and copy.deepcopy(result) is result
    with pytest.raises(TypeError, match="factory-only"):
        AICCM2026DevBOccupiedGroupWitness()
    with pytest.raises(ValueError):
        result.translation_generators.setflags(write=True)
    rng = np.random.default_rng(129)
    gauge, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3)))
    mixed = build_aiccm2026dev_b_occupied_group_witness(
        group, replace(localized, coefficients=localized.coefficients @ gauge),
    )
    assert mixed.maximum_group_residual < 1e-12
    assert any(not d.is_monomial for d in mixed.occupied_actions)
    assert mixed.fingerprint != result.fingerprint
    for base, transformed in zip(result.occupied_actions, mixed.occupied_actions):
        np.testing.assert_allclose(transformed.matrix, gauge.conj().T @ base.matrix @ gauge, atol=1e-12)
    from vibeqc.output.citations.registry import load_default_database
    citations = load_default_database().assemble(method="rhf", basis="sto-3g", numerics=result.citation_numerics)
    assert {"casassa_symmetry_wannier_2006", "dovesi_symmetry_lcao_1986"} <= {
        item.key for item in citations.citations
    }


def test_occupied_group_refuses_nontranslation_closed_state(occupied_action_fixture) -> None:
    *_, template = occupied_action_fixture
    system = vq.PeriodicSystem(3, np.eye(3)*8, [vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [2., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    plan = build_aiccm2026dev_b_symmetry_plan(system, (3, 1, 1))
    index = _operation_index(plan, np.eye(3, dtype=int), [0, 0, 0])
    group = build_aiccm2026dev_b_torus_symmetry_group(
        system, basis, replace(plan, operations=(plan.operations[index],)),
    )
    localized = replace(template, coefficients=np.eye(6)[:, :3], overlap=np.eye(6))
    build_aiccm2026dev_b_occupied_symmetry_action(group.actions[0], localized)
    with pytest.raises(ValueError, match="translation does not preserve"):
        build_aiccm2026dev_b_occupied_group_witness(group, localized)


def test_occupied_group_physical_half_translation_cocycle() -> None:
    from vibeqc.periodic.chi.localization import localize_aiccm2026dev_b_occupied

    system = vq.PeriodicSystem(3, np.diag([8., 16., 16.]), [
        vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [4., 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        scf = run_aiccm2026dev_b_rhf(system, basis, (3, 1, 1), backend="four_center", progress=False)
    assert scf.converged
    localized = localize_aiccm2026dev_b_occupied(scf, system, basis)
    plan = build_aiccm2026dev_b_symmetry_plan(system, (3, 1, 1))
    identity = _operation_index(plan, np.eye(3, dtype=int), [0, 0, 0])
    half = _operation_index(plan, np.eye(3, dtype=int), [0.5, 0, 0])
    subgroup = replace(plan, operations=(plan.operations[identity], plan.operations[half]))
    group = build_aiccm2026dev_b_torus_symmetry_group(system, basis, subgroup)
    np.testing.assert_array_equal(group.lattice_cocycle[1, 1], [1, 0, 0])
    result = build_aiccm2026dev_b_occupied_group_witness(group, localized)
    d = result.occupied_actions[1].matrix
    np.testing.assert_allclose(d @ d, result.translation_generators[0], atol=1e-12)
    assert np.linalg.norm(d @ d - np.eye(6)) > 1  # quotient inverse is NOT the torus inverse
    assert result.maximum_group_residual < 1e-12
    # The same physical Hilbert space can carry a nonmonomial complex gauge.
    rng = np.random.default_rng(130)
    gauge, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6)))
    mixed = build_aiccm2026dev_b_occupied_group_witness(
        group, replace(localized, coefficients=localized.coefficients @ gauge),
    )
    assert mixed.maximum_group_residual < 1e-12
    assert not mixed.occupied_actions[1].is_monomial
    mixed_d = mixed.occupied_actions[1].matrix
    np.testing.assert_allclose(mixed_d @ mixed_d, mixed.translation_generators[0], atol=1e-12)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 1e-3, 1j, "1e-8", 10**400])
def test_group_factories_reject_invalid_tolerance(occupied_action_fixture, value) -> None:
    system, basis, plan, _, _, localized = occupied_action_fixture
    with pytest.raises(ValueError, match="tolerance"):
        build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan, tolerance=value)
    group = build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)
    with pytest.raises(ValueError, match="tolerance"):
        build_aiccm2026dev_b_occupied_group_witness(group, localized, tolerance=value)


@pytest.mark.parametrize("name", ["max_workspace_bytes", "max_group_products"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_group_factories_reject_invalid_caps(occupied_action_fixture, name, value) -> None:
    system, basis, plan, _, _, localized = occupied_action_fixture
    with pytest.raises(ValueError, match=name):
        build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan, **{name: value})
    group = build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)
    with pytest.raises(ValueError, match=name):
        build_aiccm2026dev_b_occupied_group_witness(group, localized, **{name: value})


def test_occupied_group_memory_work_admission_and_input_mutations(occupied_action_fixture, monkeypatch) -> None:
    import vibeqc.periodic.chi.symmetry as module

    system, basis, plan, _, _, localized = occupied_action_fixture
    group = build_aiccm2026dev_b_torus_symmetry_group(system, basis, plan)
    result = build_aiccm2026dev_b_occupied_group_witness(group, localized)
    assert build_aiccm2026dev_b_occupied_group_witness(
        group, localized, max_workspace_bytes=result.estimated_workspace_bytes,
    ).fingerprint == result.fingerprint
    with pytest.raises(ValueError, match="orthonormal"):
        build_aiccm2026dev_b_occupied_group_witness(group, replace(localized, coefficients=localized.coefficients*2))
    with pytest.raises(ValueError, match="overlap"):
        build_aiccm2026dev_b_occupied_group_witness(group, replace(localized, overlap=localized.overlap[:1]))
    def forbidden(*args, **kwargs):
        pytest.fail("occupied group admission reached action factory")
    monkeypatch.setattr(module, "build_aiccm2026dev_b_occupied_symmetry_action", forbidden)
    with pytest.raises(MemoryError, match="estimate"):
        module.build_aiccm2026dev_b_occupied_group_witness(
            group, localized, max_workspace_bytes=result.estimated_workspace_bytes-1,
        )
    with pytest.raises(ValueError, match="max_group_products"):
        module.build_aiccm2026dev_b_occupied_group_witness(group, localized, max_group_products=1)


@pytest.mark.parametrize("subspace", ["correlated_occupied","virtual"])
@pytest.mark.parametrize("spatial", ["inversion","half_translation"])
def test_chi_snapshot_shared_state_bridge_and_whole_group(chi_snapshot_fixture,subspace,spatial):
    from vibeqc.symmetry_shared import Budget, FiniteGroup, audit_group_transport, audit_periodic_state_subspace
    from vibeqc.symmetry_shared import audit_periodic_state_group, audit_selected_group_transport
    from vibeqc.symmetry_shared import OperatorContract, audit_group_operators
    system,basis,_ = chi_snapshot_fixture
    state = _chi_snapshot(chi_snapshot_fixture).state
    core = vq._vibeqc_core
    budget = Budget(128 << 20,10**10)
    table = np.array([[(g%2+h%2)%2+2*((g//2)^(h//2)) for h in range(4)]
                      for g in range(4)],dtype=np.int64)
    rotations = np.tile(np.eye(3,dtype=np.int64),(4,1,1))
    images = np.zeros((4,4,3),dtype=np.int64)
    if spatial == "inversion": rotations[1::2] *= -1
    else:
        for g in range(4):
            for h in range(4): images[g,h,0] = (g%2+h%2)//2
    group = FiniteGroup.from_table(table,np.array([0,0,1,1],dtype=np.uint8),
        identity=0,identity_label=f"{spatial} times scalar time reversal",budget=budget,
        rotations=rotations,cocycle=images)
    bridges = []
    for op in range(4):
        native = core._make_periodic_ao_bloch_operation(
            np.ascontiguousarray(rotations[op],dtype=np.int32),np.array([.5*(op%2),0.,0.]))
        options,inventory,caps = _chi_sewing_controls()
        if spatial == "half_translation" and state.n_kpoints == 2 and op == 1:
            # Existing finite-support failure: geometry and group algebra
            # cannot override the actual state's transported stationarity.
            with pytest.raises(ValueError,match="transported target Roothaan"):
                audit_periodic_state_subspace(state,basis,system,native,1,False,
                    subspace=subspace,sewing_options=options,sewing_inventory=inventory,sewing_caps=caps,
                    metric_tolerance=1e-8,leakage_tolerance=1e-8,
                    probe_identity="chi rejected even-mesh half translation",budget=budget)
            return
        bridges.append(tuple(audit_periodic_state_subspace(state,basis,system,native,k,bool(op//2),
            subspace=subspace,sewing_options=options,sewing_inventory=inventory,sewing_caps=caps,
            metric_tolerance=1e-8,leakage_tolerance=1e-8,
            probe_identity="chi actual RHF state group",budget=budget) for k in range(state.n_kpoints)))
    spaces = tuple(item.transport.metric.source_space for item in bridges[0])
    dest = np.array([[item.target_index for item in row] for row in bridges],dtype=np.int64)
    rows = tuple(tuple(item.transport for item in row) for row in bridges)
    result = audit_group_transport(group,spaces,dest,rows,contract=rows[0][0].metric.contract,
        characters=tuple(item.source_character for item in bridges[0]),tolerance=1e-8,
        probe_identity="chi actual RHF full mesh",budget=budget)
    assert result.passed_probe and result.composition_residual < 1e-9
    assert not result.production_reduction_authorized
    assert all(item.state is state for row in bridges for item in row)
    bound = audit_periodic_state_group(group,tuple(bridges),seitz_tolerance=1e-10,
        composition_tolerance=1e-8,probe_identity="bound chi group",budget=budget)
    assert bound.passed_probe and bound.state is state
    assert bound.maximum_fractional_seitz_residual == 0.
    selections = tuple(np.eye(t.mixing.shape[0],dtype=complex)*np.exp(.17j*(k+1))
                       for k,t in enumerate(rows[group.identity]))
    selected = audit_selected_group_transport(bound,selections,
        selection_identity="chi full-mask complex coordinate gauge",metric_tolerance=1e-8,
        leakage_tolerance=1e-8,composition_tolerance=1e-8,
        probe_identity="chi native selected coordinate group",budget=budget)
    assert selected.passed_probe and selected.parent is bound and selected.parent.state is state
    assert selected.group_transport.characters == bound.group_transport.characters
    assert not selected.production_reduction_authorized
    fock_operators = []
    for k,item in enumerate(bridges[0]):
        c = state.coefficients(k)[:,item.source_bands]
        fock_operators.append(np.ascontiguousarray(c.conj().T @ state.fock(k) @ c))
    operator_contract = OperatorContract("projected snapshot Fock",state.state_identity_sha256,
        "native state masks","snapshot declaration","static retained orthonormal coordinates")
    for parent,operators in (
        (bound,tuple(fock_operators)),
        (selected,tuple(np.ascontiguousarray(q.conj().T @ a @ q)
                        for q,a in zip(selections,fock_operators))),
    ):
        operator_result = audit_group_operators(parent,operators,contract=operator_contract,
            tolerance=1e-8,probe_identity="chi native snapshot static Fock",budget=budget)
        assert operator_result.passed_probe and operator_result.parent is parent
        assert not operator_result.production_reduction_authorized
    np.testing.assert_array_equal(bound.group_transport.destinations,dest)
    changed_row = (bridges[1][0],bridges[0][1])+bridges[1][2:]
    changed = (bridges[0],changed_row)+tuple(bridges[2:])
    with pytest.raises(ValueError,match="operation changes"):
        audit_periodic_state_group(group,changed,seitz_tolerance=1e-10,
            composition_tolerance=1e-8,probe_identity="k-dependent operation negative",budget=budget)
    reordered = ((bridges[0][1],bridges[0][0])+bridges[0][2:],)+tuple(bridges[1:])
    with pytest.raises(ValueError,match="source order"):
        audit_periodic_state_group(group,reordered,seitz_tolerance=1e-10,
            composition_tolerance=1e-8,probe_identity="wrong k row order",budget=budget)
    if spatial == "half_translation":
        quotient = FiniteGroup.from_table(table,np.array([0,0,1,1],dtype=np.uint8),
            identity=0,identity_label="incorrectly discarded lattice images",budget=budget)
        wrong = audit_group_transport(quotient,spaces,dest,rows,contract=rows[0][0].metric.contract,
            tolerance=1e-8,probe_identity="discarded image negative",budget=budget)
        assert wrong.local_probes_passed and not wrong.passed_probe
        with pytest.raises(ValueError,match="full lattice cocycle"):
            audit_periodic_state_group(quotient,tuple(bridges),seitz_tolerance=1e-10,
                composition_tolerance=1e-8,probe_identity="missing integer images",budget=budget)
    # Different immutable numerical-state namespaces cannot be spliced into
    # this group's rows, even if only the calculation declaration changed.
    other = _chi_snapshot(chi_snapshot_fixture,declared_calculation_identity="b"*64).state
    identity = core._make_periodic_ao_bloch_operation(np.eye(3,dtype=np.int32),np.zeros(3))
    item = audit_periodic_state_subspace(other,basis,system,identity,0,False,
        subspace=subspace,sewing_options=options,sewing_inventory=inventory,sewing_caps=caps,
        metric_tolerance=1e-8,leakage_tolerance=1e-8,probe_identity="other state",budget=budget)
    mixed = ((item.transport,)+rows[0][1:],)+rows[1:]
    with pytest.raises(ValueError,match="contract or space identity"):
        audit_group_transport(group,spaces,dest,mixed,contract=rows[0][0].metric.contract,
            tolerance=1e-8,probe_identity="mixed state",budget=budget)
