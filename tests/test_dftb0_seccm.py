from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy.linalg import eigh

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc.semiempirical import (
    DFTB0_SECCM_PARAMETER_SET,
    SemiempiricalRoutePlan,
    run_dftb0_seccm,
)
from vibeqc.semiempirical.parameters import default_parameters
from vibeqc.semiempirical.seccm import (
    SECCMTopologyError,
    bind_complete_translation_action,
    bind_finite_group,
    build_seccm_topology,
)
from vibeqc.semiempirical.seccm._adapter_common import flatten_topology_records


_PRIMITIVE_TRANSLATION = np.array([8.0, 0.0, 0.0])
_PRIMITIVE_COORDS = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])


def _cyclic_h2(replicas: int, *, bind_action: bool = False):
    coords = np.vstack(
        [
            _PRIMITIVE_COORDS + cell * _PRIMITIVE_TRANSLATION
            for cell in range(replicas)
        ]
    )
    topology = build_seccm_topology(
        coords,
        [replicas * _PRIMITIVE_TRANSLATION],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[_PRIMITIVE_TRANSLATION],
        replicas=(replicas, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    if bind_action:
        topology = bind_complete_translation_action(
            topology,
            atom_primitive_site_ids=[0, 1] * replicas,
            atom_cell_labels=[
                (cell, 0, 0)
                for cell in range(replicas)
                for _ in range(2)
            ],
            atom_equivalence_keys=[("H", "a"), ("H", "b")] * replicas,
            equivalence_key_schema="dftb0-seccm-test-sites-v1",
        )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coords],
        0,
        1,
    )
    return molecule, topology, coords


def _fixed_topology_energy(topology, coords: np.ndarray) -> float:
    rebuild_options = (
        {"max_tie_score_excursion": 1.0e-3}
        if topology.has_reference_ties
        else {}
    )
    displaced_topology = topology.rebuild_displacements(coords, **rebuild_options)
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coords], 0, 1
    )
    return run_dftb0_seccm(molecule, displaced_topology).energy


def test_dftb0_seccm_molecular_limit_matches_dftb0(capsys):
    molecule, topology, _ = _cyclic_h2(1)

    result = run_dftb0_seccm(molecule, topology)
    molecular = _se_cxx.run_dftb0(molecule, default_parameters())

    assert result.energy == pytest.approx(molecular.energy, abs=1.0e-13)
    assert result.electronic_energy == pytest.approx(
        molecular.e_electronic, abs=1.0e-13
    )
    assert result.repulsive_energy == pytest.approx(
        molecular.e_repulsive, abs=1.0e-13
    )
    np.testing.assert_allclose(result.overlap, molecular.overlap, atol=1.0e-13)
    np.testing.assert_allclose(
        result.hamiltonian, molecular.hamiltonian, atol=1.0e-13
    )
    assert result.normalization == "per_primitive_cell"
    assert result.parameter_set == DFTB0_SECCM_PARAMETER_SET
    assert result.gradient is None
    assert not result.overlap.flags.writeable
    assert capsys.readouterr() == ("", "")


def test_dftb0_seccm_zero_record_one_atom_reaches_gap_gate():
    coordinates = np.zeros((1, 3))
    translation = np.array([20.0, 0.0, 0.0])
    topology = build_seccm_topology(
        coordinates,
        [translation],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[translation],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    molecule = Molecule([Atom(6, coordinates[0].tolist())], 0, 1)

    with pytest.raises(RuntimeError, match="positive finite-torus HOMO-LUMO gap"):
        run_dftb0_seccm(molecule, topology)


def test_dftb0_seccm_molecular_limit_gradient_matches_dftb0():
    molecule, topology, _ = _cyclic_h2(1)

    result = run_dftb0_seccm(molecule, topology, compute_gradient=True)
    molecular = _se_cxx.run_dftb0(molecule, default_parameters())
    expected = _se_cxx.compute_dftb0_gradient(
        molecule, molecular, default_parameters()
    )

    assert result.gradient is not None
    np.testing.assert_allclose(result.gradient, expected, atol=2.0e-12)
    assert not result.gradient.flags.writeable


@pytest.mark.parametrize("replicas", [2, 3, 4])
def test_dftb0_seccm_analytic_gradient_matches_fixed_topology_fd(replicas):
    molecule, topology, coords = _cyclic_h2(replicas, bind_action=True)
    result = run_dftb0_seccm(molecule, topology, compute_gradient=True)
    assert result.gradient is not None

    step = 1.0e-5
    finite_difference = np.zeros_like(result.gradient)
    for atom, axis in [(0, 0), (1, 0), (0, 1)]:
        plus = coords.copy()
        minus = coords.copy()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        finite_difference[atom, axis] = (
            _fixed_topology_energy(topology, plus)
            - _fixed_topology_energy(topology, minus)
        ) / (2.0 * step)

    for atom, axis in [(0, 0), (1, 0), (0, 1)]:
        assert result.gradient[atom, axis] == pytest.approx(
            finite_difference[atom, axis], abs=2.0e-9
        )
    np.testing.assert_allclose(result.gradient.sum(axis=0), 0.0, atol=2.0e-12)


@pytest.mark.parametrize("replicas", [2, 3, 4])
def test_dftb0_seccm_energy_closure_and_reverse_hermiticity(replicas):
    molecule, topology, _ = _cyclic_h2(replicas, bind_action=True)

    result = run_dftb0_seccm(molecule, topology)

    assert result.group_order == replicas
    assert result.homo_lumo_gap > 1.0e-8
    assert result.energy == pytest.approx(
        result.electronic_energy
        + result.repulsive_energy
        + result.long_range_energy
        + result.dispersion_energy
        + result.specific_energy,
        abs=1.0e-14,
    )
    assert result.total_cyclic_energy == pytest.approx(
        replicas * result.energy, abs=1.0e-13
    )
    assert result.cyclic_electronic_energy == pytest.approx(
        replicas * result.electronic_energy, abs=1.0e-13
    )
    assert result.cyclic_repulsive_energy == pytest.approx(
        replicas * result.repulsive_energy, abs=1.0e-13
    )
    np.testing.assert_allclose(result.overlap, result.overlap.T, atol=1.0e-13)
    np.testing.assert_allclose(
        result.hamiltonian, result.hamiltonian.T, atol=1.0e-13
    )
    assert np.linalg.eigvalsh(result.overlap).min() > 0.0


@pytest.mark.parametrize("replicas", [3, 4])
def test_dftb0_seccm_matches_discrete_character_decomposition(replicas):
    molecule, topology, _ = _cyclic_h2(replicas, bind_action=True)
    result = run_dftb0_seccm(molecule, topology)

    character_energies: list[float] = []
    for character in range(replicas):
        phase = 2.0 * np.pi * character / replicas
        transform = np.zeros((2 * replicas, 2), dtype=complex)
        for cell in range(replicas):
            for site in range(2):
                transform[2 * cell + site, site] = (
                    np.exp(1j * phase * cell) / np.sqrt(replicas)
                )
        h_character = transform.conj().T @ result.hamiltonian @ transform
        s_character = transform.conj().T @ result.overlap @ transform
        character_energies.extend(eigh(h_character, s_character, eigvals_only=True))

    np.testing.assert_allclose(
        np.sort(character_energies), result.mo_energies, atol=5.0e-14
    )


def test_dftb0_seccm_even_nyquist_records_keep_fractional_ownership():
    molecule, topology, _ = _cyclic_h2(2)
    tied = [
        image
        for cell in topology.cells
        for image in cell
        if image.ownership_multiplicity == 2
    ]

    assert tied
    assert {image.weight for image in tied} == {0.5}
    assert {
        image.image_shell_label[0]
        for image in tied
    } >= {-1, 0, 1}
    result = run_dftb0_seccm(molecule, topology, compute_gradient=True)
    assert result.n_records == sum(len(cell) for cell in topology.cells)
    assert result.gradient is not None
    np.testing.assert_allclose(result.gradient.sum(axis=0), 0.0, atol=2.0e-12)


def test_dftb0_seccm_group_only_defect_uses_full_cluster():
    coords = np.vstack(
        [
            _PRIMITIVE_COORDS,
            _PRIMITIVE_COORDS + 2 * _PRIMITIVE_TRANSLATION,
        ]
    )
    topology = bind_finite_group(
        build_seccm_topology(
            coords,
            [3 * _PRIMITIVE_TRANSLATION],
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=[_PRIMITIVE_TRANSLATION],
        replicas=(3, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coords], 0, 1
    )

    result = run_dftb0_seccm(molecule, topology, compute_gradient=True)

    assert not topology.complete_atom_action_bound
    assert topology.current_atom_geometry_group_covariant is None
    assert np.isfinite(result.energy)
    assert result.group_order == 3
    assert result.gradient is not None


def test_dftb0_seccm_invalidated_action_does_not_enable_orbit_reduction():
    _, topology, coords = _cyclic_h2(3, bind_action=True)
    displaced = coords.copy()
    displaced[0, 1] += 1.0e-3
    topology = topology.rebuild_displacements(displaced)
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in displaced], 0, 1
    )

    result = run_dftb0_seccm(molecule, topology, compute_gradient=True)

    assert topology.current_lattice_group_compatible
    assert topology.current_atom_geometry_group_covariant is False
    assert not topology.record_orbits_currently_usable
    assert np.isfinite(result.energy)
    assert result.gradient is not None


def test_dftb0_seccm_stale_lattice_fails_before_parameter_loading(monkeypatch):
    molecule, topology, coords = _cyclic_h2(1)
    stale = topology.rebuild_displacements(
        coords, translations=[np.array([9.0, 0.0, 0.0])]
    )
    module = importlib.import_module("vibeqc.semiempirical.seccm.dftb0")
    monkeypatch.setattr(
        module,
        "default_parameters",
        lambda: pytest.fail("parameter loading must not be reached"),
    )

    with pytest.raises(SECCMTopologyError, match="incompatible"):
        run_dftb0_seccm(molecule, stale)


def test_dftb0_seccm_scope_is_the_repulsive_pair_table():
    """Scope is the explicit repulsive-pair table, not a hardcoded
    element list (maintainer decision D2, 2026-08-28). The shipped
    vibeqc-inhouse-dftb-screening-v1 set covers H, C, N, O, F, P, S, Cl
    and all 36 of their unordered pairs, so nitrogen is in scope; an
    element with parameters but no repulsive pair (Si) is refused with
    the covered set named."""
    parameters = default_parameters()
    covered = [1, 6, 7, 8, 9, 15, 16, 17]
    for first in covered:
        for second in covered:
            assert parameters.has_repulsive_pair(first, second)
    assert parameters.has_element(14)
    assert not parameters.has_repulsive_pair(14, 14)

    coords = _PRIMITIVE_COORDS.copy()
    topology = bind_finite_group(
        build_seccm_topology(
            coords,
            [_PRIMITIVE_TRANSLATION],
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=[_PRIMITIVE_TRANSLATION],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    nitrogen = Molecule([Atom(7, c.tolist()) for c in coords])
    accepted = run_dftb0_seccm(nitrogen, topology)
    assert np.isfinite(accepted.energy)

    silicon = Molecule([Atom(14, c.tolist()) for c in coords])
    with pytest.raises(NotImplementedError, match="H, C, N, O, F, P, S, Cl"):
        run_dftb0_seccm(silicon, topology)


@pytest.mark.parametrize(
    "translations",
    [
        np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]),
        np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]),
    ],
)
def test_native_dftb0_seccm_accepts_multidimensional_records(translations):
    """The native seam accepts 2-D and 3-D cyclic topologies
    (maintainer decision D2, 2026-08-28). Every stage below the former
    gate is dimension-generic and is the same code SCC-DFTB-SECCM
    already runs in 2-D/3-D; DFTB0 is charge-free, so neither the
    even-replica SCC pathology nor the Klopman-Ohno thermodynamic-limit
    defect applies to it."""
    molecule, topology, _ = _cyclic_h2(1)
    (
        _,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        _,
        _,
    ) = flatten_topology_records(
        molecule, topology, route_name="DFTB0-SECCM native scope test"
    )
    replicas = (1, 1, 1)
    result = _se_cxx._run_dftb0_seccm_from_records(
        molecule,
        default_parameters(),
        translations,
        central,
        origin,
        np.asarray(shell_labels, dtype=np.int32),
        weights,
        multiplicities,
        np.asarray(displacements, dtype=float),
        np.asarray(translations, dtype=float),
        replicas,
        1.0e-9,
        False,
    )
    assert np.isfinite(result.energy)


@pytest.mark.parametrize(
    (
        "translations",
        "primitive_vectors",
        "replicas",
        "expected_exception",
        "match",
    ),
    [
        (
            np.array([[8.0, 0.0, 0.0]]),
            np.array([[8.0, 0.0]]),
            (1, 1, 1),
            ValueError,
            r"shape \(D, 3\)",
        ),
        (
            np.array([[8.0, 0.0, 0.0]]),
            np.array([[np.inf, 0.0, 0.0]]),
            (1, 1, 1),
            ValueError,
            "must be finite",
        ),
        (
            np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]),
            np.array([[8.0, 0.0, 0.0], [16.0, 0.0, 0.0]]),
            (1, 1, 1),
            ValueError,
            "linearly dependent",
        ),
        (
            np.array([[8.0, 0.0, 0.0]]),
            np.array([[8.0, 0.0, 0.0]]),
            (1, 1),
            ValueError,
            "exactly three",
        ),
        (
            np.array([[8.0, 0.0, 0.0]]),
            np.array([[8.0, 0.0, 0.0]]),
            (1, 0, 1),
            ValueError,
            "must be positive",
        ),
        (
            np.array([[8.0, 0.0, 0.0]]),
            np.array([[8.0, 0.0, 0.0]]),
            (1, 2, 1),
            ValueError,
            "inactive replicas",
        ),
        (
            np.array([[8.0, 0.0, 0.0]]),
            np.array([[4.0, 0.0, 0.0]]),
            (1, 1, 1),
            ValueError,
            "replica-scaled primitive vectors",
        ),
        (
            np.array([[50000.0, 0.0, 0.0], [0.0, 50000.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            (50000, 50000, 1),
            OverflowError,
            "finite-group order overflows",
        ),
    ],
)
def test_native_dftb0_seccm_rejects_structurally_invalid_finite_group(
    translations,
    primitive_vectors,
    replicas,
    expected_exception,
    match,
):
    molecule, topology, _ = _cyclic_h2(1)
    (
        _,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        _,
        _,
    ) = flatten_topology_records(
        molecule, topology, route_name="DFTB0-SECCM native attestation test"
    )

    with pytest.raises(expected_exception, match=match):
        _se_cxx._run_dftb0_seccm_from_records(
            molecule,
            default_parameters(),
            translations,
            central,
            origin,
            np.asarray(shell_labels, dtype=np.int32),
            weights,
            multiplicities,
            np.asarray(displacements, dtype=float),
            primitive_vectors,
            replicas,
            1.0e-9,
            False,
        )


@pytest.mark.parametrize("property_name", ["stress", "charges"])
def test_dftb0_seccm_route_fails_closed_for_unimplemented_properties(
    property_name,
):
    with pytest.raises(NotImplementedError, match="does not support"):
        SemiempiricalRoutePlan.from_request(
            "dftb0", boundary="seccm", properties=(property_name,)
        )


def test_dftb0_seccm_route_fails_closed_for_charge_spin_and_scc_knobs():
    with pytest.raises(NotImplementedError, match="neutral"):
        SemiempiricalRoutePlan.from_request(
            "dftb0", boundary="seccm", charge=1
        )
    with pytest.raises(NotImplementedError, match="closed-shell"):
        SemiempiricalRoutePlan.from_request(
            "dftb0", boundary="seccm", multiplicity=3
        )
    with pytest.raises(ValueError, match="MSINDO SECCM adapter"):
        SemiempiricalRoutePlan.from_request(
            "dftb0", boundary="seccm", ccm_options=object()
        )


def test_dftb0_seccm_route_plan_is_narrow_and_experimental():
    plan = SemiempiricalRoutePlan.from_request("dftb", boundary="se-ccm")

    assert plan.method_key == "dftb0"
    assert plan.variant == "dftb0"
    assert plan.boundary == "seccm_direct_torus"
    assert plan.properties == ("energy",)
    assert plan.scc == "none"
    assert plan.maturity == "experimental"
    assert plan.execution == "native"
    assert plan.status_route == "dftb0-seccm-energy"

    gradient_plan = SemiempiricalRoutePlan.from_request(
        "dftb0",
        boundary="seccm",
        properties=("energy", "gradient"),
    )
    assert gradient_plan.status_route == "dftb0-seccm-gradient-analytic"
    assert gradient_plan.execution == "native"


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("replicas", [3, 4, 5])
def test_dftb0_seccm_matches_dense_k_in_every_dimension(dimension, replicas):
    """DFTB0-SECCM reproduces the matched-mesh Bloch reference in 1-D,
    2-D and 3-D (maintainer decision D2, 2026-08-28, which relaxed the
    former 1-D-only gate).

    The fixture is an H2 molecular crystal at 8 bohr spacing: closed
    shell and gapped, so the finite-torus gate is satisfied at both
    parities. It is deliberately weakly coupled, which makes the
    Wigner-Seitz truncation error negligible and isolates the claim
    being pinned here - that the dimension-generic assembly is correct
    in every dimension, agreeing with an independent Bloch k-sum to
    machine precision. It does NOT pin the truncation convergence rate;
    a strongly coupled ladder is the separate measurement for that.

    DFTB0 carries no SCC, so neither the even-replica SCC pathology nor
    the Klopman-Ohno thermodynamic-limit defect (#444) reaches this
    route, and both even and odd replica counts are exercised."""
    import itertools

    from vibeqc._vibeqc_core import PeriodicSystem
    from vibeqc.semiempirical.periodic import _as_bloch_kmesh

    spacing = 8.0
    bond = 1.4
    basis = [np.zeros(3), np.array([bond, 0.0, 0.0])]
    primitive = [
        np.array([spacing, 0.0, 0.0]),
        np.array([0.0, spacing, 0.0]),
        np.array([0.0, 0.0, spacing]),
    ][:dimension]

    coords = []
    for index in itertools.product(*[range(replicas)] * dimension):
        origin = np.zeros(3)
        for axis, count in enumerate(index):
            origin = origin + count * primitive[axis]
        for site in basis:
            coords.append(origin + site)
    molecule = Molecule([Atom(1, c.tolist()) for c in coords])
    topology = bind_finite_group(
        build_seccm_topology(
            coords,
            [replicas * p for p in primitive],
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=primitive,
        replicas=tuple([replicas] * dimension + [1] * (3 - dimension)),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    seccm_energy = run_dftb0_seccm(molecule, topology).energy

    system = PeriodicSystem()
    lattice = np.diag([spacing, spacing, spacing]).astype(float)
    for inactive in range(dimension, 3):
        lattice[inactive, inactive] = 80.0
    system.lattice = lattice
    system.unit_cell = [Atom(1, site.tolist()) for site in basis]
    kmesh = _as_bloch_kmesh(
        system, tuple([replicas] * dimension + [1] * (3 - dimension))
    )
    reference = _se_cxx.run_dftb0_kpoints(
        system,
        default_parameters(),
        kmesh,
        40.0,
        _se_cxx.KPointOccupationOptions(),
    ).energy

    assert seccm_energy == pytest.approx(reference, abs=1.0e-11)
