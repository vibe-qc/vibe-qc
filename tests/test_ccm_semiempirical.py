"""CCM (SECCM) boundary regression tests for the semiempirical methods.

The cyclic cluster model is a boundary condition: the Hamiltonian inside
the supercell is the method's ordinary molecular Hamiltonian, assembled
with Wigner-Seitz image weights. These tests pin that contract:

* the molecular limit (one replica, large cyclic translations) reproduces
  the molecular driver for the same method;
* the 1-D chain limit converges towards the independent k-point periodic
  driver as the cluster grows;
* 2-D and odd-replica 3-D clusters are finite, converged, and
  lattice-symmetry invariant, while unvalidated 3-D parity fails closed.

Covered adapters: DFTB0 (existing, smoke-pinned here) and SCC-DFTB (new).
"""

from __future__ import annotations

import ast
import json
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vibeqc import Atom, Molecule, PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical import (
    DFTB0_SECCM_PARAMETER_SET,
    DFTB0RepulsivePlaceholderWarning,
    SemiempiricalRoutePlan,
    run_dftb0_seccm,
    run_scc_dftb_seccm,
)
from vibeqc.semiempirical.parameters import default_parameters
from vibeqc.semiempirical.seccm import (
    bind_complete_translation_action,
    bind_finite_group,
    build_seccm_topology,
)

# Independent dense-k reference for the 1-D H-Li chain below.  The 24-point
# values at 1280- and 2560-bohr lattice cutoffs are -0.107889022629041 and
# -0.107889080170082 Ha; their leading-1/R^2 Richardson limit is the value
# below.  Repeating both calculations with 48 k points moves each by 1e-15 Ha.
_HLI_CHAIN_CELL = 4.1
_HLI_SITES = np.array([[0.17, 0.31, 0.0], [1.39, -0.22, 0.0]])
_HLI_DENSE_K_LIMIT = -0.107889099350429
# The issue-313 6x6x6 comparator uses the historical 12-bohr lattice cutoff.
# It establishes the odd-subsequence direction only; unlike the 1-D value
# above, it is not claimed as an infinite-cutoff thermodynamic limit.
_HLI_3D_MATCHED_DENSE_K = -0.11893

_H2O_COORDS = np.array([[0.0, 0.0, 0.0], [1.8089, 0.0, 0.0], [-0.4576, 1.4316, 0.0]])

# DFTB0-SECCM (T3a) is gated to H/C systems, so its molecular-limit probe
# uses methane.
_CH4_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [0.63, -0.63, -0.63],
        [-0.63, 0.63, -0.63],
    ]
)


def _topology(coords, translations, *, replicas, primitive_vectors):
    topology = build_seccm_topology(
        coords,
        [np.asarray(t, dtype=float) for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p, dtype=float) for p in primitive_vectors],
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def _h2o_molecule(coords):
    return Molecule(
        [Atom(8, c.tolist()) for c in coords[:1]]
        + [Atom(1, c.tolist()) for c in coords[1:]],
        0,
        1,
    )


def _ch4_molecule(coords):
    return Molecule(
        [Atom(6, c.tolist()) for c in coords[:1]]
        + [Atom(1, c.tolist()) for c in coords[1:]],
        0,
        1,
    )


def _hli_chain_molecule(coords):
    return Molecule(
        [Atom(1 if k % 2 == 0 else 3, c.tolist()) for k, c in enumerate(coords)],
        0,
        1,
    )


def _hli_3d_case(replicas, sites=_HLI_SITES):
    replicas = tuple(int(replica) for replica in replicas)
    primitives = [
        np.array([_HLI_CHAIN_CELL, 0.0, 0.0]),
        np.array([0.0, _HLI_CHAIN_CELL, 0.0]),
        np.array([0.0, 0.0, _HLI_CHAIN_CELL]),
    ]
    coords = np.array(
        [
            sum(
                (index[axis] * primitives[axis] for axis in range(3)),
                start=np.zeros(3),
            )
            + site
            for index in product(*(range(replica) for replica in replicas))
            for site in sites
        ]
    )
    topology = _topology(
        coords,
        [replicas[axis] * primitives[axis] for axis in range(3)],
        replicas=replicas,
        primitive_vectors=primitives,
    )
    return _hli_chain_molecule(coords), topology


# ---------------------------------------------------------------------------
# DFTB0-SECCM: the existing adapter still meets the boundary contract.
# ---------------------------------------------------------------------------


def test_dftb0_seccm_molecular_limit_matches_molecular_dftb0():
    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _CH4_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _ch4_molecule(_CH4_COORDS)

    result = run_dftb0_seccm(molecule, topology)
    molecular = _se.run_dftb0(molecule, default_parameters())

    assert result.energy == pytest.approx(molecular.energy, abs=1.0e-13)
    assert result.parameter_set == DFTB0_SECCM_PARAMETER_SET
    assert result.normalization == "per_primitive_cell"


# ---------------------------------------------------------------------------
# SCC-DFTB-SECCM
# ---------------------------------------------------------------------------


def test_scc_dftb_seccm_route_plan():
    plan = SemiempiricalRoutePlan.from_request("scc_dftb", boundary="seccm")
    assert plan.method_key == "scc_dftb"
    assert plan.boundary == "seccm_direct_torus"
    assert plan.properties == ("energy",)
    assert plan.scc == "atomic"
    assert plan.maturity == "experimental"
    assert plan.execution == "native"
    assert plan.status_route == "scc-dftb-seccm-energy"

    gradient_plan = SemiempiricalRoutePlan.from_request(
        "scc_dftb", boundary="seccm", properties=("energy", "gradient")
    )
    assert gradient_plan.status_route == "scc-dftb-seccm-gradient-analytic"
    assert gradient_plan.execution == "native"
    # Charged supercells are accepted at the plan level; the engine fails
    # closed unless the opt-in Madelung embedding is active (tested below).
    charged_plan = SemiempiricalRoutePlan.from_request(
        "scc_dftb", boundary="seccm", charge=1
    )
    assert charged_plan.charge == 1
    assert charged_plan.status_route == "scc-dftb-seccm-energy"


def test_scc_dftb_seccm_molecular_limit_matches_molecular_scc_dftb():
    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)

    # Match the molecular driver's convergence tolerance so the SCC
    # iteration stops at the same fixed-point step and every energy term
    # (not just the stationary total) agrees to machine-level precision.
    result = run_scc_dftb_seccm(molecule, topology, conv_tol_charge=1.0e-6)
    molecular = _se.run_scc_dftb(molecule, default_parameters())

    assert result.converged
    assert result.energy == pytest.approx(molecular.energy, abs=1.0e-12)
    assert result.e_electronic == pytest.approx(molecular.e_electronic, abs=1.0e-12)
    assert result.e_repulsive == pytest.approx(molecular.e_repulsive, abs=1.0e-12)
    assert result.e_scc == pytest.approx(molecular.e_scc, abs=1.0e-12)
    np.testing.assert_allclose(result.charges, molecular.charges, atol=1.0e-10)
    np.testing.assert_allclose(result.overlap, molecular.overlap, atol=1.0e-12)
    np.testing.assert_allclose(result.hamiltonian, molecular.hamiltonian, atol=1.0e-10)
    assert (
        result.cyclic_electronic_energy
        + result.cyclic_repulsive_energy
        + result.cyclic_scc_energy
        + result.cyclic_madelung_energy
    ) == pytest.approx(result.total_cyclic_energy, abs=1.0e-12)
    assert result.normalization == "per_primitive_cell"


def test_scc_dftb_seccm_accepts_symmetry_broken_full_record_topology():
    primitive = np.array([8.0, 0.0, 0.0])
    sites = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    replicas = 3
    coords = np.vstack(
        [sites + cell * primitive for cell in range(replicas)]
    )
    topology = _topology(
        coords,
        [replicas * primitive],
        replicas=(replicas, 1, 1),
        primitive_vectors=[primitive],
    )
    topology = bind_complete_translation_action(
        topology,
        atom_primitive_site_ids=[0, 1] * replicas,
        atom_cell_labels=[
            (cell, 0, 0)
            for cell in range(replicas)
            for _ in range(2)
        ],
        atom_equivalence_keys=[("H", "a"), ("H", "b")] * replicas,
        equivalence_key_schema="scc-dftb-seccm-test-sites-v1",
    )
    displaced = coords.copy()
    displaced[0, 1] += 1.0e-3
    topology = topology.rebuild_displacements(displaced)
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in displaced], 0, 1
    )

    result = run_scc_dftb_seccm(molecule, topology)

    assert topology.current_lattice_group_compatible
    assert topology.current_atom_geometry_group_covariant is False
    assert not topology.record_orbits_currently_usable
    assert result.converged
    assert np.isfinite(result.energy)


def test_scc_dftb_seccm_molecular_limit_gradient_matches_molecular_scc_dftb():
    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)

    result = run_scc_dftb_seccm(
        molecule, topology, conv_tol_charge=1.0e-6, compute_gradient=True
    )
    molecular = _se.run_scc_dftb(molecule, default_parameters())
    expected = _se.compute_scc_dftb_gradient(molecule, molecular, default_parameters())

    assert result.gradient is not None
    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)


def test_scc_dftb_seccm_analytic_gradient_matches_fixed_topology_fd():
    # Fixed-topology convention (topology.rebuild_displacements): the WS
    # record set stays frozen while the displacements track the geometry.
    molecule, topology = _hli_chain(2, charge=0)
    coords = np.array([atom.xyz for atom in molecule.atoms])

    result = run_scc_dftb_seccm(molecule, topology, compute_gradient=True)
    assert result.gradient is not None

    step = 1.0e-5
    for atom in range(len(coords)):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step

            def energy(geometry):
                displaced = topology.rebuild_displacements(
                    geometry,
                    **(
                        {"max_tie_score_excursion": 1.0e-3}
                        if topology.has_reference_ties
                        else {}
                    ),
                )
                return run_scc_dftb_seccm(
                    _hli_chain_molecule(geometry), displaced
                ).energy

            finite_difference = (energy(plus) - energy(minus)) / (2.0 * step)
            assert result.gradient[atom, axis] == pytest.approx(
                finite_difference, abs=2.0e-8
            )


def test_scc_dftb_seccm_rejects_obsolete_gradient_fd_step():
    molecule, topology = _hli_chain(2, charge=0)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_scc_dftb_seccm(
            molecule,
            topology,
            compute_gradient=True,
            gradient_fd_step=1.0e-3,
        )


def _embedded_gradient_fd_oracle(molecule, topology, **run_kwargs):
    """Independent central-difference reimplementation of the embedded
    per-cell gradient: fully re-converged SCF energies at each displaced
    geometry over the fixed-topology displacements (the same contract as
    the analytic variational path)."""
    coords = np.array([atom.xyz for atom in molecule.atoms])
    step = 1.0e-4
    gradient = np.zeros((len(coords), 3))
    for atom in range(len(coords)):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step

            def energy(geometry):
                displaced = topology.rebuild_displacements(
                    geometry,
                    **(
                        {"max_tie_score_excursion": 1.0e-3}
                        if topology.has_reference_ties
                        else {}
                    ),
                )
                displaced_molecule = Molecule(
                    [
                        Atom(atom.Z, c.tolist())
                        for atom, c in zip(molecule.atoms, geometry)
                    ],
                    molecule.charge,
                    molecule.multiplicity,
                )
                return run_scc_dftb_seccm(
                    displaced_molecule, displaced, **run_kwargs
                ).energy

            gradient[atom, axis] = (energy(plus) - energy(minus)) / (2.0 * step)
    return gradient


def test_scc_dftb_seccm_embedded_variational_gradient_matches_fd():
    # The Madelung interaction enters through the same overlap-weighted
    # variational Mulliken-charge operator as ordinary SCC. Its implicit
    # charge response therefore cancels by stationarity; re-converged central
    # differences remain the independent oracle for the explicit derivative.
    # The N=3 charged polar chain sits at a near-degenerate T=0 frontier,
    # so the charge map is evaluated at electronic_temperature 0.005 with
    # the same Fermi-Dirac occupations kernel as the SCF loop.
    molecule, topology = _hli_chain(3, charge=2)

    result = run_scc_dftb_seccm(
        molecule,
        topology,
        madelung=True,
        use_diis=True,
        electronic_temperature=0.005,
        compute_gradient=True,
    )
    assert result.gradient is not None
    assert result.gradient_method == "analytic"
    # Translation invariance: the embedded per-cell force sums to zero.
    assert np.abs(result.gradient.sum(axis=0)).max() == pytest.approx(0.0, abs=1.0e-6)
    fd_gradient = _embedded_gradient_fd_oracle(
        molecule,
        topology,
        madelung=True,
        use_diis=True,
        electronic_temperature=0.005,
    )
    np.testing.assert_allclose(result.gradient, fd_gradient, atol=1.0e-6)


def test_scc_dftb_seccm_embedded_gradient_n2_t0_matches_fd():
    # T = 0 N=2 embedded chain: same variational-gradient validation as the
    # smeared N=3 case, this time without smearing.
    molecule, topology = _hli_chain(2, charge=2)

    result = run_scc_dftb_seccm(
        molecule,
        topology,
        madelung=True,
        use_diis=True,
        compute_gradient=True,
    )
    assert result.converged
    assert result.gradient is not None
    assert result.gradient_method == "analytic"
    assert np.abs(result.gradient.sum(axis=0)).max() == pytest.approx(0.0, abs=1.0e-6)
    fd_gradient = _embedded_gradient_fd_oracle(
        molecule, topology, madelung=True, use_diis=True
    )
    np.testing.assert_allclose(result.gradient, fd_gradient, atol=1.0e-6)


def test_scc_dftb_seccm_1d_chain_converges_towards_kpoint_limit():
    primitive = np.array([_HLI_CHAIN_CELL, 0.0, 0.0])
    energies = {}
    for replicas in (4, 8, 16, 32):
        coords = np.vstack([_HLI_SITES + cell * primitive for cell in range(replicas)])
        topology = _topology(
            coords,
            [replicas * primitive],
            replicas=(replicas, 1, 1),
            primitive_vectors=[primitive],
        )
        result = run_scc_dftb_seccm(_hli_chain_molecule(coords), topology)
        assert result.converged
        assert result.homo_lumo_gap > 0.0
        energies[replicas] = result.energy

    errors = {
        replicas: abs(energy - _HLI_DENSE_K_LIMIT)
        for replicas, energy in energies.items()
    }
    # The CCM environment approaches the infinite periodic limit from the
    # truncated side, and its leading finite-size term is O(R^-2): doubling
    # the replica count must quarter the per-cell error.  That ratio, not the
    # absolute bound below, is what discriminates a correct reference from a
    # displaced one.  A reference carrying its own cutoff error still decays
    # monotonically but flattens the ratios: the pre-repair
    # -0.106949485783247 Ha value yields 2.49, 1.59, 1.17 instead of ~4.
    # See issue #312.
    assert errors[8] < errors[4]
    assert errors[16] < errors[8]
    assert errors[32] < errors[16]
    ratios = (errors[4] / errors[8], errors[8] / errors[16], errors[16] / errors[32])
    for ratio in ratios:
        assert 3.5 < ratio < 4.5, ratios
    # Absolute backstop.  Measured 5.82e-5 Ha at 32 replicas, so 8.0e-5 keeps
    # ~1.4x headroom for ordinary platform-to-platform drift while still
    # rejecting the pre-repair reference (9.98e-4 Ha) by more than 12x.
    assert errors[32] < 8.0e-5


def test_scc_dftb_seccm_1d_translation_invariance():
    primitive = np.array([_HLI_CHAIN_CELL, 0.0, 0.0])
    replicas = 4
    coords = np.vstack([_HLI_SITES + cell * primitive for cell in range(replicas)])
    shifted = coords + np.array([1.0, 0.0, 0.0])

    def energy(geometry):
        topology = _topology(
            geometry,
            [replicas * primitive],
            replicas=(replicas, 1, 1),
            primitive_vectors=[primitive],
        )
        return run_scc_dftb_seccm(_hli_chain_molecule(geometry), topology).energy

    assert energy(coords) == pytest.approx(energy(shifted), abs=1.0e-12)


def test_scc_dftb_seccm_2d_sheet_axis_swap_and_convergence_trend():
    primitive = np.array([_HLI_CHAIN_CELL, 0.0, 0.0])
    primitives = [
        np.array([_HLI_CHAIN_CELL, 0.0, 0.0]),
        np.array([0.0, _HLI_CHAIN_CELL, 0.0]),
    ]

    def grid(replicas, sites):
        return np.array(
            [
                origin + site
                for idx in product(range(replicas), repeat=2)
                for origin in [idx[0] * primitives[0] + idx[1] * primitives[1]]
                for site in sites
            ]
        )

    # Lattice-symmetry invariance: swapping the two square-lattice axes
    # (with the cell geometry swapped the same way) must not move the energy.
    coords_xy = grid(2, _HLI_SITES)
    coords_yx = grid(2, _HLI_SITES[:, [1, 0, 2]])
    e_xy = run_scc_dftb_seccm(
        _hli_chain_molecule(coords_xy),
        _topology(
            coords_xy,
            [2 * p for p in primitives],
            replicas=(2, 2, 1),
            primitive_vectors=primitives,
        ),
    ).energy
    e_yx = run_scc_dftb_seccm(
        _hli_chain_molecule(coords_yx),
        _topology(
            coords_yx,
            [2 * p for p in primitives],
            replicas=(2, 2, 1),
            primitive_vectors=primitives,
        ),
    ).energy
    assert e_xy == pytest.approx(e_yx, abs=1.0e-12)

    # Even-replica subsequence: the polar 2-D cell oscillates with parity
    # (fractional WS face weights on the small torus), but the even-replica
    # values move monotonically towards the independent Gamma reference.
    system = PeriodicSystem(
        2,
        np.diag([_HLI_CHAIN_CELL, _HLI_CHAIN_CELL, 30.0]),
        [Atom(1, s.tolist()) for s in _HLI_SITES[:1]]
        + [Atom(3, s.tolist()) for s in _HLI_SITES[1:]],
        0,
        1,
    )
    gamma_options = _se.PeriodicSCCOptions()
    # 60 bohr, not 20 (#316): under pair-distance image selection the
    # truncated 1/sqrt(R^2+eta^2) SCC gamma on this polar 2-D sheet is
    # basin-fragile across cutoffs (20 and 40 bohr converge to Mulliken
    # wall states with |dq| > 1); 60 bohr sits in the sane-magnitude
    # basin (E = -0.116, comparable to the pre-#316 cutoff-60 value
    # -0.115).  The truncated-gamma fragility is recorded on issue #316
    # as an Ewald/compensated-gamma follow-up; the SECCM ladder this
    # test actually guards is unaffected by the anchor's route.
    gamma_options.cutoff_bohr = 60.0
    gamma_reference = _se.run_scc_dftb_gamma(
        system, default_parameters(), gamma_options
    ).energy
    assert gamma_reference > e_xy

    energies = [e_xy]
    for replicas in (4, 6):
        coords = grid(replicas, _HLI_SITES)
        result = run_scc_dftb_seccm(
            _hli_chain_molecule(coords),
            _topology(
                coords,
                [replicas * p for p in primitives],
                replicas=(replicas, replicas, 1),
                primitive_vectors=primitives,
            ),
        )
        assert result.converged
        energies.append(result.energy)
    assert energies[0] < energies[1] < energies[2]
    assert abs(energies[2] - gamma_reference) < abs(energies[0] - gamma_reference)


def test_scc_dftb_seccm_3d_odd_branch_converges_and_is_axis_invariant():
    def energy(replicas, sites):
        molecule, topology = _hli_3d_case(replicas, sites)
        result = run_scc_dftb_seccm(molecule, topology)
        assert result.converged
        assert result.homo_lumo_gap > 0.0
        # per-primitive-cell normalization contract.
        assert result.energy == pytest.approx(
            result.total_cyclic_energy / np.prod(replicas), abs=1.0e-12
        )
        return result.energy

    base = energy((3, 3, 3), _HLI_SITES)
    assert base == pytest.approx(-0.114596161176388, abs=1.0e-12)
    r1 = energy((1, 1, 1), _HLI_SITES)
    assert abs(base - _HLI_3D_MATCHED_DENSE_K) < abs(
        r1 - _HLI_3D_MATCHED_DENSE_K
    )
    for permutation in ([1, 0, 2], [2, 1, 0]):
        assert energy((3, 3, 3), _HLI_SITES[:, permutation]) == pytest.approx(
            base, abs=1.0e-12
        )


@pytest.mark.parametrize(
    "replicas",
    [(2, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
)
def test_scc_dftb_seccm_3d_even_replica_axes_fail_closed(replicas):
    molecule, topology = _hli_3d_case(replicas)
    with pytest.raises(
        NotImplementedError,
        match="3-D even-replica/Nyquist ownership is not validated",
    ):
        run_scc_dftb_seccm(molecule, topology)


def test_scc_dftb_seccm_native_adapter_rejects_3d_even_replicas():
    from vibeqc.semiempirical.seccm._adapter_common import (
        flatten_topology_records,
    )

    molecule, topology = _hli_3d_case((2, 2, 2))
    records = flatten_topology_records(
        molecule, topology, route_name="SCC-DFTB-SECCM"
    )
    (
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        primitive_vectors,
        replicas,
    ) = records
    group = topology.finite_group
    assert group is not None
    with pytest.raises(
        ValueError,
        match="3-D even-replica/Nyquist ownership is not validated",
    ):
        _se._run_scc_dftb_seccm_from_records(
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
            float(group.geometry_tolerance),
        )


# ---------------------------------------------------------------------------
# SCC-DFTB-SECCM Madelung/Ewald embedding and charged-cell guards
# ---------------------------------------------------------------------------


def _hli_chain(molecule_replicas: int, *, charge: int = 0):
    primitive = np.array([_HLI_CHAIN_CELL, 0.0, 0.0])
    coords = np.vstack(
        [_HLI_SITES + cell * primitive for cell in range(molecule_replicas)]
    )
    topology = _topology(
        coords,
        [molecule_replicas * primitive],
        replicas=(molecule_replicas, 1, 1),
        primitive_vectors=[primitive],
    )
    molecule = Molecule(
        [Atom(1 if k % 2 == 0 else 3, c.tolist()) for k, c in enumerate(coords)],
        charge,
        1,
    )
    return molecule, topology


def _b1_mgo100_l2_slab(*, a_angstrom: float = 4.212):
    """Genuine neutral Tasker-I B1 MgO(100) L2, not the polar stressor."""
    from vibeqc.molecule import ANGSTROM_TO_BOHR

    t1 = np.array([a_angstrom / 2.0, a_angstrom / 2.0, 0.0])
    t2 = np.array([-a_angstrom / 2.0, a_angstrom / 2.0, 0.0])
    unlike = np.array([a_angstrom / 2.0, 0.0, 0.0])
    coords_angstrom = []
    atomic_numbers = []
    for layer in range(2):
        z_offset = np.array([0.0, 0.0, layer * a_angstrom / 2.0])
        for i in range(2):
            for j in range(2):
                home = i * t1 + j * t2 + z_offset
                if layer % 2 == 0:
                    coords_angstrom.extend((home, home + unlike))
                else:
                    coords_angstrom.extend((home + unlike, home))
                atomic_numbers.extend((12, 8))

    coords_angstrom = np.asarray(coords_angstrom)
    atomic_numbers = np.asarray(atomic_numbers)

    # Issue 471 showed why a material name or neutral total formula is not
    # enough. Pin the crystallography here: orthogonal primitive surface
    # vectors of area a^2/2, equal Mg/O counts in every plane, and six unlike
    # nearest neighbours in the two-plane 3-D extension (B1, not B2 or the
    # historical single-species polar-plane stress fixture).
    assert np.dot(t1, t2) == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(t1) == pytest.approx(
        a_angstrom / np.sqrt(2.0), abs=1.0e-12
    )
    assert np.linalg.norm(t2) == pytest.approx(
        a_angstrom / np.sqrt(2.0), abs=1.0e-12
    )
    assert np.linalg.norm(np.cross(t1, t2)) == pytest.approx(
        a_angstrom**2 / 2.0, abs=1.0e-12
    )
    for plane_z in (0.0, a_angstrom / 2.0):
        plane_species = atomic_numbers[
            np.isclose(
                coords_angstrom[:, 2], plane_z, rtol=0.0, atol=1.0e-12
            )
        ]
        assert np.count_nonzero(plane_species == 12) == 4
        assert np.count_nonzero(plane_species == 8) == 4

    bulk_cell = np.asarray(
        [2.0 * t1, 2.0 * t2, [0.0, 0.0, a_angstrom]]
    )
    unlike_coordination = []
    for center_index, center in enumerate(coords_angstrom):
        neighbours = 0
        for shift_label in product((-1, 0, 1), repeat=3):
            shift = np.asarray(shift_label, dtype=float) @ bulk_cell
            for other_index, other in enumerate(coords_angstrom):
                if atomic_numbers[center_index] == atomic_numbers[other_index]:
                    continue
                if np.isclose(
                    np.linalg.norm(other + shift - center),
                    a_angstrom / 2.0,
                    rtol=0.0,
                    atol=1.0e-10,
                ):
                    neighbours += 1
        unlike_coordination.append(neighbours)
    assert set(unlike_coordination) == {6}

    coords = coords_angstrom * ANGSTROM_TO_BOHR
    primitives = [t1 * ANGSTROM_TO_BOHR, t2 * ANGSTROM_TO_BOHR]
    topology = _topology(
        coords,
        [2.0 * primitive for primitive in primitives],
        replicas=(2, 2, 1),
        primitive_vectors=primitives,
    )
    molecule = Molecule(
        [
            Atom(z, coord.tolist())
            for z, coord in zip(atomic_numbers, coords, strict=True)
        ],
        0,
        1,
    )
    return molecule, topology


def test_scc_dftb_seccm_genuine_b1_mgo100_separates_repulsive_limit():
    # Both arms use the same frozen geometry and therefore the same additive
    # placeholder repulsion, which is evaluated only after SCC.  The public
    # issue-306 warning must remain visible on the accepted and refused paths.
    # This isolates terms in the in-house screening model; it does not validate
    # an MgO energy, band gap, or causal link between the gap and convergence.
    molecule, topology = _b1_mgo100_l2_slab()
    warning = r"8-12, 12-12.*Fixed-geometry differences"

    with pytest.warns(DFTB0RepulsivePlaceholderWarning, match=warning):
        unembedded = run_scc_dftb_seccm(
            molecule,
            topology,
            max_iter=3000,
        )

    assert unembedded.converged
    assert unembedded.n_iter == 19
    assert unembedded.homo_lumo_gap == pytest.approx(
        0.00428369951148803, abs=1.0e-10
    )
    assert not unembedded.gap_guard_waived
    assert (
        unembedded.homo_lumo_gap
        > unembedded.finite_torus_gap_tolerance
    )

    with pytest.warns(DFTB0RepulsivePlaceholderWarning, match=warning):
        with pytest.raises(RuntimeError, match="did not converge"):
            run_scc_dftb_seccm(
                molecule,
                topology,
                madelung=True,
                use_diis=True,
                electronic_temperature=0.005,
                max_iter=3000,
            )


def test_scc_dftb_seccm_charged_cell_requires_embedding():
    molecule, topology = _hli_chain(2, charge=2)
    with pytest.raises(NotImplementedError, match="madelung=True"):
        run_scc_dftb_seccm(molecule, topology)


def test_scc_dftb_seccm_charged_1d_chain_with_madelung():
    # The converged background-corrected 1-D Ewald embedding exposes the
    # true physics of the odd-N charged polar chain: its SCC fixed point
    # sits at a degenerate frontier (gap ~1e-15), so the hard-Aufbau T=0
    # response is discontinuous and no mixer contracts. A small Fermi
    # temperature smooths the occupation (molecular run_scc_dftb
    # convention) and the iteration converges; the reported energy is the
    # Mermin free energy A = E - T*S. See
    # handovers/HANDOVER_SECCM_ADAPTERS.md.
    molecule, topology = _hli_chain(3, charge=2)

    embedded = run_scc_dftb_seccm(
        molecule,
        topology,
        madelung=True,
        use_diis=True,
        electronic_temperature=0.005,
    )
    neutral = run_scc_dftb_seccm(
        _hli_chain(3, charge=0)[0], topology, madelung=True, use_diis=True
    )

    assert embedded.converged
    assert neutral.converged
    assert np.isfinite(embedded.energy)
    assert embedded.e_madelung != pytest.approx(0.0, abs=1.0e-12)
    assert embedded.energy != pytest.approx(neutral.energy, abs=1.0e-12)
    assert embedded.energy == pytest.approx(
        embedded.total_cyclic_energy / 3, abs=1.0e-12
    )
    assert embedded.energy == pytest.approx(embedded.free_energy, abs=1.0e-12)
    assert embedded.smearing_temperature == pytest.approx(0.005)
    assert embedded.entropy > 0.0
    assert embedded.charges.sum() == pytest.approx(2.0, abs=1.0e-9)
    assert (
        embedded.e_electronic
        + embedded.e_repulsive
        + embedded.e_scc
        + embedded.e_madelung
    ) == pytest.approx(embedded.energy, abs=1.0e-12)
    assert (
        embedded.cyclic_electronic_energy
        + embedded.cyclic_repulsive_energy
        + embedded.cyclic_scc_energy
        + embedded.cyclic_madelung_energy
    ) == pytest.approx(embedded.total_cyclic_energy, abs=1.0e-12)


def test_scc_dftb_seccm_smeared_degenerate_frontier_records_the_gap_waiver():
    # IID 422. The positive-gap guard used to be skipped wholesale under
    # finite electronic temperature ("!smeared &&"), so this state --
    # frontier gap 1.3e-15 Ha, seven orders below the 1e-8 Ha guard epsilon
    # -- converged with converged=True and nothing on the record saying the
    # positive-gap requirement had been waived. Any screen of the form
    # "converged and gap > 0" admitted it, because 1.3e-15 > 0.
    #
    # Finite T does resolve the occupation uniquely (Fermi-Dirac), so the
    # state stays admissible; what it must not be is indistinguishable from
    # a genuinely gapped one.
    molecule, topology = _hli_chain(3, charge=2)

    embedded = run_scc_dftb_seccm(
        molecule,
        topology,
        madelung=True,
        use_diis=True,
        electronic_temperature=0.005,
    )

    assert embedded.converged
    # The applied epsilon is on the record, so a consumer can reproduce the
    # accept/reject decision without knowing the route's defaults.
    assert embedded.finite_torus_gap_tolerance == pytest.approx(1.0e-8, abs=0.0)
    # This frontier is numerically degenerate.
    assert 0.0 < embedded.homo_lumo_gap <= embedded.finite_torus_gap_tolerance
    # ... and the row says so.
    assert embedded.gap_guard_waived is True


def test_scc_dftb_seccm_gapped_smeared_state_does_not_waive_the_gap_guard():
    # IID 422 discrimination: the waiver flag must be false for an ordinary
    # gapped cell run at the same temperature, or it screens nothing.
    molecule, topology = _hli_chain(2, charge=0)

    result = run_scc_dftb_seccm(
        molecule, topology, use_diis=True, electronic_temperature=0.005
    )

    assert result.converged
    assert result.homo_lumo_gap > result.finite_torus_gap_tolerance
    assert result.gap_guard_waived is False


def test_scc_dftb_seccm_records_the_gap_epsilon_at_zero_temperature():
    # The epsilon is recorded on every result, not only on smeared ones,
    # and the T = 0 Aufbau path never waives the guard.
    molecule, topology = _hli_chain(2, charge=0)

    result = run_scc_dftb_seccm(molecule, topology, use_diis=True)

    assert result.converged
    assert result.finite_torus_gap_tolerance == pytest.approx(1.0e-8, abs=0.0)
    assert result.gap_guard_waived is False


def test_scc_dftb_seccm_1d_broyden_matches_diis():
    # Broyden is the opt-in quasi-Newton alternative to Aitken/DIIS. On this
    # ordinary two-cell chain it converges faster and must land on the same
    # fixed point.
    molecule, topology = _hli_chain(2, charge=0)

    broyden = run_scc_dftb_seccm(
        molecule, topology, use_broyden=True, max_iter=2000
    )
    diis = run_scc_dftb_seccm(
        molecule, topology, use_diis=True, max_iter=2000
    )

    assert broyden.converged
    assert broyden.n_iter < diis.n_iter
    assert broyden.energy == pytest.approx(diis.energy, abs=1.0e-9)
    assert broyden.charges.sum() == pytest.approx(0.0, abs=1.0e-9)
    assert np.allclose(broyden.charges, diis.charges, atol=1.0e-6)


def test_scc_dftb_seccm_charged_1d_chain_n4_converges_with_ewald():
    # The N=4 regression for the converged background-corrected 1-D Ewald
    # embedding: the old truncated ±2-shell bare Madelung sum left the
    # even-N charged polar chain response nearly singular (period-2 limit
    # cycle under every mixer). The converged kernel closes the gap and
    # N=4 now converges at T=0 under DIIS, and the per-cell energy sits
    # monotonically between the N=2 and N=6 ladder values. (Stock Broyden
    # still fails on this zig-zag geometry; the N=2 embedded chain keeps
    # the Broyden-vs-DIIS fixed-point parity coverage.)
    molecule, topology = _hli_chain(4, charge=2)

    diis = run_scc_dftb_seccm(
        molecule, topology, madelung=True, use_diis=True, max_iter=2000
    )
    n2 = run_scc_dftb_seccm(
        *_hli_chain(2, charge=2), madelung=True, use_diis=True, max_iter=2000
    )
    n6 = run_scc_dftb_seccm(
        *_hli_chain(6, charge=2), madelung=True, use_diis=True, max_iter=2000
    )

    assert diis.converged
    assert diis.e_madelung != pytest.approx(0.0, abs=1.0e-12)
    assert diis.charges.sum() == pytest.approx(2.0, abs=1.0e-9)
    # Even-N subsequence moves monotonically towards the dense limit.
    assert n2.energy > diis.energy > n6.energy
    assert diis.energy == pytest.approx(diis.total_cyclic_energy / 4, abs=1.0e-12)


def test_scc_dftb_seccm_charged_2d_sheet_with_madelung():
    primitives = [
        np.array([_HLI_CHAIN_CELL, 0.0, 0.0]),
        np.array([0.0, _HLI_CHAIN_CELL, 0.0]),
    ]
    replicas = 2
    coords = np.array(
        [
            origin + site
            for idx in product(range(replicas), repeat=2)
            for origin in [idx[0] * primitives[0] + idx[1] * primitives[1]]
            for site in _HLI_SITES
        ]
    )
    topology = _topology(
        coords,
        [replicas * p for p in primitives],
        replicas=(replicas, replicas, 1),
        primitive_vectors=primitives,
    )
    molecule = Molecule(
        [Atom(1 if k % 2 == 0 else 3, c.tolist()) for k, c in enumerate(coords)],
        2,
        1,
    )

    result = run_scc_dftb_seccm(molecule, topology, madelung=True, use_diis=True)
    assert result.converged
    assert np.isfinite(result.energy)
    assert result.energy == pytest.approx(
        result.total_cyclic_energy / (replicas**2), abs=1.0e-12
    )
    assert result.charges.sum() == pytest.approx(2.0, abs=1.0e-9)


def test_scc_dftb_seccm_3d_madelung_fails_closed():
    primitives = [
        np.array([_HLI_CHAIN_CELL, 0.0, 0.0]),
        np.array([0.0, _HLI_CHAIN_CELL, 0.0]),
        np.array([0.0, 0.0, _HLI_CHAIN_CELL]),
    ]
    replicas = 2
    coords = np.array(
        [
            origin + site
            for idx in product(range(replicas), repeat=3)
            for origin in [
                idx[0] * primitives[0] + idx[1] * primitives[1] + idx[2] * primitives[2]
            ]
            for site in _HLI_SITES
        ]
    )
    topology = _topology(
        coords,
        [replicas * p for p in primitives],
        replicas=(replicas, replicas, replicas),
        primitive_vectors=primitives,
    )
    molecule = Molecule(
        [Atom(1 if k % 2 == 0 else 3, c.tolist()) for k, c in enumerate(coords)],
        2,
        1,
    )

    with pytest.raises(
        NotImplementedError,
        match=r"3-D Madelung embedding.*no thermodynamic limit",
    ):
        run_scc_dftb_seccm(
            molecule, topology, madelung=True, use_diis=True
        )


def test_scc_dftb_seccm_neutral_madelung_smoke():
    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)

    result = run_scc_dftb_seccm(molecule, topology, madelung=True)
    assert result.converged
    assert np.isfinite(result.energy)
    # The 1-D embedding shells sit 40+ bohr away; the neutral-cell
    # self-energy must be a small correction to the molecular-limit value.
    assert abs(result.e_madelung) < 1.0e-2


def test_scc_dftb_seccm_madelung_uses_variational_mulliken_operator():
    from vibeqc.semiempirical.methods.msindo_ccm import (
        _ewald_ws_cells,
        _madkonst_2d,
        _madelung_potential_ewald,
    )

    translations = [
        np.array([20.0, 0.0, 0.0]),
        np.array([0.0, 20.0, 0.0]),
    ]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    plain = run_scc_dftb_seccm(molecule, topology)
    embedded = run_scc_dftb_seccm(molecule, topology, madelung=True)

    params = default_parameters()
    n_atoms = len(molecule.atoms)
    gamma = np.zeros((n_atoms, n_atoms))
    for atom, center in enumerate(molecule.atoms):
        gamma[atom, atom] = params.hubbard_u(center.Z)
        for image in topology.cells[atom]:
            origin = int(image.origin)
            ua = params.hubbard_u(center.Z)
            ub = params.hubbard_u(molecule.atoms[origin].Z)
            eta = 0.5 * (1.0 / ua + 1.0 / ub)
            distance = np.linalg.norm(image.disp)
            gamma[atom, origin] += float(image.weight) / np.sqrt(
                distance * distance + eta * eta
            )

    ao_atom = np.repeat(np.arange(n_atoms), [4, 1, 1])

    def deposit(potential):
        atom_potential = potential[ao_atom]
        return 0.5 * embedded.overlap * (
            atom_potential[:, None] + atom_potential[None, :]
        )

    h0 = plain.hamiltonian + deposit(gamma @ plain.charges)
    ewald_cells = _ewald_ws_cells(topology)
    madkonst = _madkonst_2d(
        ewald_cells, topology.translations, n_atoms
    )
    madelung = _madelung_potential_ewald(
        embedded.charges, madkonst, ewald_cells
    )
    expected = h0 - deposit(gamma @ embedded.charges + madelung)

    np.testing.assert_allclose(embedded.hamiltonian, expected, atol=1.0e-10)
    # The off-diagonal C/H overlap is nonzero, so this specifically rejects
    # the former MSINDO-only diagonal Madelung deposit.
    assert abs(deposit(madelung)[0, 4]) > 1.0e-8


# ---------------------------------------------------------------------------
# PM6-SECCM (WS-weighted supercell NDDO Fock)
# ---------------------------------------------------------------------------


def _pm6_params(molecule):
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

    return load_pm6_params_auto([int(atom.Z) for atom in molecule.atoms])


@pytest.mark.parametrize("route", ["pm6", "om2"])
def test_nddo_seccm_rejects_raw_nonhermitian_reverse_assembly(route):
    """A loose geometry gate must not let final cleanup hide raw imbalance."""
    molecule = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.0, 0.0, 0.0])],
        0,
        1,
    )
    translations = np.array([[40.0, 0.0, 0.0]])
    central = [0, 1]
    origin = [1, 0]
    shell_labels = np.zeros((2, 3), dtype=np.int32)
    weights = [1.0, 1.0]
    multiplicities = [1, 1]

    # This mutation is inside the deliberately loose geometry tolerance, so
    # common topology validation accepts both the stale displacement and its
    # imperfect reverse.  The two directed resonance integrals are evaluated
    # at different distances, however, and are materially non-Hermitian before
    # the adapter's retained roundoff cleanup.
    displacements = np.array([[1.0, 0.0, 0.0], [-1.001, 0.0, 0.0]])
    record_args = (
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        translations.copy(),
        (1, 1, 1),
        1.0e-2,
    )

    with pytest.raises(
        RuntimeError,
        match=rf"{route.upper() if route == 'pm6' else 'OMx'}-SECCM "
        r"reverse-image assembly is not Hermitian",
    ):
        if route == "pm6":
            _se._run_pm6_seccm_from_records(
                molecule,
                _pm6_params(molecule),
                *record_args,
            )
        else:
            from vibeqc.semiempirical.methods.omx_params import load_omx_params

            # The parameter set is the second positional argument of both
            # internal adapters; keep the record mutation identical.
            _se._run_omx_seccm_from_records(
                molecule,
                load_omx_params(route),
                *record_args,
            )


def test_pm6_seccm_route_plan():
    plan = SemiempiricalRoutePlan.from_request("pm6", boundary="seccm")
    assert plan.method_key == "pm6"
    assert plan.boundary == "seccm_direct_torus"
    assert plan.properties == ("energy",)
    assert plan.maturity == "experimental"
    assert plan.execution == "native"
    assert plan.status_route == "pm6-seccm-energy"

    gradient_plan = SemiempiricalRoutePlan.from_request(
        "pm6", boundary="seccm", properties=("energy", "gradient")
    )
    assert gradient_plan.status_route == "pm6-seccm-gradient-fd"
    with pytest.raises(NotImplementedError, match="neutral"):
        SemiempiricalRoutePlan.from_request("pm6", boundary="seccm", charge=1)


def test_pm6_seccm_molecular_limit_matches_molecular_pm6():
    from vibeqc.semiempirical import run_pm6_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)

    result = run_pm6_seccm(molecule, topology, max_iter=200)
    molecular = _se.nddo.run_pm6(molecule, _pm6_params(molecule), 200)

    assert result.converged
    # Directed WS-record assembly changes floating-point addition order versus
    # the unordered molecular pair loop, but both contract the same H--X tensor.
    assert result.energy == pytest.approx(molecular.energy, abs=5.0e-12)
    assert result.e_core == pytest.approx(molecular.e_core, abs=1.0e-12)
    assert result.e_electronic == pytest.approx(
        molecular.energy - molecular.e_core, abs=5.0e-12
    )
    assert result.e_electronic + result.e_core == pytest.approx(
        result.energy, abs=1.0e-14
    )
    assert result.normalization == "per_primitive_cell"


def test_pm6_seccm_molecular_limit_matches_oblique_co2():
    """The molecular limit retains all rotated heavy-heavy multipoles."""
    from vibeqc.semiempirical import run_pm6_seccm

    axis = np.array([2.2, 0.7, -0.4])
    coords = np.array([-axis, np.zeros(3), axis])
    molecule = Molecule(
        [
            Atom(8, coords[0].tolist()),
            Atom(6, coords[1].tolist()),
            Atom(8, coords[2].tolist()),
        ],
        0,
        1,
    )
    translation = np.array([40.0, 0.0, 0.0])
    topology = _topology(
        coords,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )

    result = run_pm6_seccm(molecule, topology, max_iter=200)
    molecular = _se.nddo.run_pm6(molecule, _pm6_params(molecule), 200)

    assert result.converged and molecular.converged
    assert result.energy == pytest.approx(molecular.energy, abs=5.0e-11)
    assert result.e_core == pytest.approx(molecular.e_core, abs=1.0e-12)


def test_pm6_seccm_nah_uses_the_molecular_core_charge_definition():
    """Na contributes one PM6 valence electron in both NDDO routes."""
    from vibeqc.semiempirical import run_pm6_seccm

    coords = np.array([[0.0, 0.0, 0.0], [3.6, 0.0, 0.0]])
    translation = np.array([40.0, 0.0, 0.0])
    molecule = Molecule(
        [Atom(11, coords[0].tolist()), Atom(1, coords[1].tolist())], 0, 1
    )
    topology = _topology(
        coords,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )

    result = run_pm6_seccm(molecule, topology, max_iter=200)
    molecular = _se.nddo.run_pm6(molecule, _pm6_params(molecule), 200)

    assert result.converged and molecular.converged
    assert result.n_occ == molecular.n_occ == 1
    assert result.n_basis == molecular.n_basis == 5
    assert result.energy == pytest.approx(molecular.energy, abs=1.0e-10)
    assert result.e_core == pytest.approx(molecular.e_core, abs=1.0e-12)


def test_pm6_seccm_rejects_elements_that_require_d_orbitals():
    from vibeqc.semiempirical import run_pm6_seccm

    coords = np.array([[0.0, 0.0, 0.0], [4.25, 0.0, 0.0]])
    translation = np.array([40.0, 0.0, 0.0])
    topology = _topology(
        coords,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    molecule = Molecule(
        [Atom(14, coordinate.tolist()) for coordinate in coords], 0, 1
    )

    with pytest.raises(
        ValueError,
        match=r"Z=14 requires PM6 d orbitals.*only s/p channels",
    ):
        run_pm6_seccm(molecule, topology)


def test_pm6_seccm_rejects_non_sp_parameter_records():
    """A custom heavy-atom AO count cannot bypass the s/p kernel guard."""
    molecule = Molecule(
        [Atom(6, [0.0, 0.0, 0.0]), Atom(6, [1.0, 0.0, 0.0])],
        0,
        1,
    )
    params = _pm6_params(molecule)
    element = params.element_data(6)
    assert element is not None
    element.has_d = False
    element.n_orbitals = 3
    params.add_element(element)

    with pytest.raises(
        ValueError,
        match=r"Z=6 requires PM6 d orbitals or has an unsupported AO count",
    ):
        _se._run_pm6_seccm_from_records(
            molecule,
            params,
            np.array([[40.0, 0.0, 0.0]]),
            [0, 1],
            [1, 0],
            np.zeros((2, 3), dtype=np.int32),
            [1.0, 1.0],
            [1, 1],
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.array([[40.0, 0.0, 0.0]]),
            (1, 1, 1),
            1.0e-9,
        )


def test_pm6_seccm_rejects_incomplete_placeholder_parameters():
    from vibeqc.semiempirical import run_pm6_seccm

    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    translation = np.array([40.0, 0.0, 0.0])
    topology = _topology(
        coords,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    molecule = Molecule(
        [Atom(85, coordinate.tolist()) for coordinate in coords], 0, 1
    )

    with pytest.raises(ValueError, match="incomplete placeholder"):
        run_pm6_seccm(molecule, topology)


def test_pm6_seccm_internal_seam_rejects_other_nddo_parameters():
    from vibeqc.semiempirical.methods.omx_params import load_om2_params
    from vibeqc.semiempirical.methods.pm7_params import load_pm7_params

    molecule = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.0, 0.0, 0.0])], 0, 1
    )
    for params, method_name in (
        (load_pm7_params(), "pm7"),
        (load_om2_params(), "om2"),
    ):
        with pytest.raises(
            ValueError,
            match=f"requires PM6 parameters, not {method_name}",
        ):
            _se._run_pm6_seccm_from_records(
                molecule,
                params,
                np.array([[40.0, 0.0, 0.0]]),
                [0, 1],
                [1, 0],
                np.zeros((2, 3), dtype=np.int32),
                [1.0, 1.0],
                [1, 1],
                np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
                np.array([[40.0, 0.0, 0.0]]),
                (1, 1, 1),
                1.0e-9,
            )


def test_pm6_seccm_rejects_actinide_principal_shells():
    from vibeqc.semiempirical import run_pm6_seccm

    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    translation = np.array([40.0, 0.0, 0.0])
    topology = _topology(
        coords,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    molecule = Molecule(
        [Atom(90, coordinate.tolist()) for coordinate in coords], 0, 1
    )

    with pytest.raises(ValueError, match="actinide element Z=90"):
        run_pm6_seccm(molecule, topology)


def test_pm6_seccm_fd_gradient_matches_molecular_fd():
    from vibeqc.semiempirical import run_pm6_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)

    result = run_pm6_seccm(
        molecule,
        topology,
        max_iter=200,
        conv_tol=1.0e-9,
        compute_gradient=True,
        gradient_fd_step=1.0e-4,
    )
    expected = _se.nddo.compute_pm6_gradient_fd(
        molecule,
        _pm6_params(molecule),
        h=1.0e-4,
        max_iter=200,
        conv_tol=1.0e-9,
    )

    assert result.gradient is not None
    assert result.gradient_method == "finite_difference"
    # Same energy function differentiated at the same step: agree to the
    # SCF noise of the two independent difference runs.
    np.testing.assert_allclose(result.gradient, expected, atol=5.0e-5)


def test_pm6_seccm_fd_gradient_chain_translation_invariance():
    from vibeqc.semiempirical import run_pm6_seccm

    primitive = np.array([_HLI_CHAIN_CELL, 0.0, 0.0])
    replicas = 2
    coords = np.vstack([_HLI_SITES + cell * primitive for cell in range(replicas)])
    molecule = _hli_chain_molecule(coords)
    topology = _topology(
        coords,
        [replicas * primitive],
        replicas=(replicas, 1, 1),
        primitive_vectors=[primitive],
    )

    result = run_pm6_seccm(
        molecule,
        topology,
        max_iter=200,
        compute_gradient=True,
        allow_truncated_electrostatics=True,
    )
    assert result.gradient is not None
    assert result.gradient_method == "finite_difference"
    assert np.abs(result.gradient.sum(axis=0)).max() == pytest.approx(0.0, abs=1.0e-5)


def test_pm6_seccm_1d_chain_smoke_invariants():
    from vibeqc.semiempirical import run_pm6_seccm

    primitive = np.array([_HLI_CHAIN_CELL, 0.0, 0.0])
    replicas = 4
    coords = np.vstack([_HLI_SITES + cell * primitive for cell in range(replicas)])

    def energy(geometry):
        result = run_pm6_seccm(
            _hli_chain_molecule(geometry),
            _topology(
                geometry,
                [replicas * primitive],
                replicas=(replicas, 1, 1),
                primitive_vectors=[primitive],
            ),
            max_iter=200,
            allow_truncated_electrostatics=True,
        )
        assert result.converged
        assert result.homo_lumo_gap > 0.0
        assert result.energy == pytest.approx(
            result.total_cyclic_energy / replicas, abs=1.0e-12
        )
        assert result.e_electronic + result.e_core == pytest.approx(
            result.energy, abs=1.0e-14
        )
        return result.energy

    base = energy(coords)
    # Translation invariance along the cyclic direction.
    shifted = coords + np.array([1.0, 0.0, 0.0])
    assert energy(shifted) == pytest.approx(base, abs=1.0e-10)

    # Lattice-symmetry invariance: swapping the two square-lattice axes of
    # the 2-D torus (with the cell swapped the same way) must not move the
    # energy.
    primitives = [
        np.array([_HLI_CHAIN_CELL, 0.0, 0.0]),
        np.array([0.0, _HLI_CHAIN_CELL, 0.0]),
    ]
    n2 = 2

    def sheet(sites):
        sheet_coords = np.array(
            [
                origin + site
                for idx in product(range(n2), repeat=2)
                for origin in [idx[0] * primitives[0] + idx[1] * primitives[1]]
                for site in sites
            ]
        )
        result = run_pm6_seccm(
            _hli_chain_molecule(sheet_coords),
            _topology(
                sheet_coords,
                [n2 * p for p in primitives],
                replicas=(n2, n2, 1),
                primitive_vectors=primitives,
            ),
            max_iter=200,
            allow_truncated_electrostatics=True,
        )
        assert result.converged
        return result.energy

    assert sheet(_HLI_SITES) == pytest.approx(
        sheet(_HLI_SITES[:, [1, 0, 2]]), abs=1.0e-10
    )

    # Note on conventions: the Bloch Gamma driver damps image-cell exchange
    # by S^2 so its lattice sum converges; the CCM applies the molecular NDDO
    # exchange to every WS pair (the Bredow-Jug construction). The two
    # conventions therefore agree in the molecular limit but differ at image
    # pairs, so no Gamma-vs-CCM energy comparison is asserted here.


# ---------------------------------------------------------------------------
# OMx-SECCM (WS-weighted supercell Fock; molecular limit reproduces the
# molecular driver)
# ---------------------------------------------------------------------------


_OMX_CHAIN_CELL = 6.0


def _h2o_chain_molecule(coords):
    return Molecule(
        [Atom(8 if k % 3 == 0 else 1, c.tolist()) for k, c in enumerate(coords)],
        0,
        1,
    )


def _h2o_chain(molecule_replicas: int):
    primitive = np.array([_OMX_CHAIN_CELL, 0.0, 0.0])
    coords = np.vstack(
        [_H2O_COORDS + cell * primitive for cell in range(molecule_replicas)]
    )
    topology = _topology(
        coords,
        [molecule_replicas * primitive],
        replicas=(molecule_replicas, 1, 1),
        primitive_vectors=[primitive],
    )
    return _h2o_chain_molecule(coords), topology


def test_omx_seccm_route_plans():
    for method in ("om2", "om3"):
        plan = SemiempiricalRoutePlan.from_request(method, boundary="seccm")
        assert plan.boundary == "seccm_direct_torus"
        assert plan.status_route == "omx-seccm-energy"
        assert plan.properties == ("energy",)
        assert plan.maturity == "experimental"

        runtime = plan.with_seccm_runtime(periodic_dimension=1)
        assert runtime.three_center_weighting == "peintinger_eq13"


def test_om1_seccm_fails_closed_at_plan_public_and_native_seam():
    """The incomplete OM1 core-valence ECP cannot reach a cyclic SCF."""
    from vibeqc.semiempirical import run_omx_seccm
    from vibeqc.semiempirical.methods.omx_params import load_omx_params

    reason = "published analytic core-valence ECP is not implemented"
    with pytest.raises(NotImplementedError, match=reason):
        SemiempiricalRoutePlan.from_request("om1", boundary="seccm")

    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    translation = np.array([40.0, 0.0, 0.0])
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coords], 0, 1
    )
    topology = _topology(
        coords,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    with pytest.raises(NotImplementedError, match=reason):
        run_omx_seccm(molecule, topology, variant="om1")

    with pytest.raises(ValueError, match=reason):
        _se._run_omx_seccm_from_records(
            molecule,
            load_omx_params("om1"),
            np.array([translation]),
            [0, 1],
            [1, 0],
            np.zeros((2, 3), dtype=np.int32),
            [1.0, 1.0],
            [1, 1],
            np.array([[1.4, 0.0, 0.0], [-1.4, 0.0, 0.0]]),
            np.array([translation]),
            (1, 1, 1),
            1.0e-9,
        )


@pytest.mark.parametrize("variant", ["om2", "om3"])
def test_omx_seccm_molecular_limit_matches_molecular(variant):
    from vibeqc.semiempirical import run_omx_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _CH4_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _ch4_molecule(_CH4_COORDS)

    from vibeqc.semiempirical.methods.omx_params import load_omx_params

    result = run_omx_seccm(molecule, topology, variant=variant, max_iter=200)
    redundant_ack = run_omx_seccm(
        molecule,
        topology,
        variant=variant,
        max_iter=200,
        allow_truncated_electrostatics=True,
    )
    molecular = _se.nddo.run_omx_v2(molecule, load_omx_params(variant), 200)

    assert result.converged
    assert result.energy == pytest.approx(molecular.energy, abs=1.0e-12)
    assert result.e_core == pytest.approx(molecular.e_core, abs=1.0e-12)
    assert result.e_electronic == pytest.approx(
        molecular.energy - molecular.e_core, abs=1.0e-12
    )
    assert result.e_electronic + result.e_core == pytest.approx(
        result.energy, abs=1.0e-14
    )
    assert result.normalization == "per_primitive_cell"
    assert not result.truncated_electrostatics_acknowledged
    assert not result.route_plan.truncated_electrostatics_acknowledged
    assert redundant_ack.energy == result.energy
    assert not redundant_ack.truncated_electrostatics_acknowledged
    assert not redundant_ack.route_plan.truncated_electrostatics_acknowledged


def test_omx_seccm_multi_replica_requires_truncated_electrostatics_ack(
    monkeypatch,
):
    """Unembedded OMx replicas require an explicit mechanics-only opt-in."""
    from vibeqc.semiempirical import run_omx_seccm
    from vibeqc.semiempirical.methods.omx_params import load_omx_params
    from vibeqc.semiempirical.seccm._adapter_common import (
        flatten_topology_records,
    )

    primitive = np.array([5.0, 0.0, 0.0])
    sites = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    coords = np.vstack([sites, sites + primitive])
    molecule = Molecule(
        [
            Atom(Z, coordinate.tolist())
            for Z, coordinate in zip([1, 9, 1, 9], coords, strict=True)
        ],
        0,
        1,
    )
    topology = _topology(
        coords,
        [2.0 * primitive],
        replicas=(2, 1, 1),
        primitive_vectors=[primitive],
    )

    message = "long-range Madelung/Ewald electrostatics"

    def unexpected_native_call(*args, **kwargs):
        pytest.fail("public OMx-SECCM guard reached the native solver")

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            _se,
            "_run_omx_seccm_from_records",
            unexpected_native_call,
        )
        with pytest.raises(NotImplementedError, match=message):
            run_omx_seccm(molecule, topology, variant="om2")
        with pytest.raises(ValueError, match="must be boolean"):
            run_omx_seccm(
                molecule,
                topology,
                variant="om2",
                allow_truncated_electrostatics="yes",
            )

    record_args = flatten_topology_records(
        molecule,
        topology,
        route_name="OMx-SECCM",
    )
    malformed_record_args = list(record_args)
    malformed_record_args[7] = np.zeros_like(record_args[7])
    with pytest.raises(
        ValueError,
        match="primitive vectors are linearly dependent",
    ):
        _se._run_omx_seccm_from_records(
            molecule,
            load_omx_params("om2"),
            *malformed_record_args,
            1.0e-9,
        )
    with pytest.raises(ValueError, match=message):
        _se._run_omx_seccm_from_records(
            molecule,
            load_omx_params("om2"),
            *record_args,
            1.0e-9,
        )
    native = _se._run_omx_seccm_from_records(
        molecule,
        load_omx_params("om2"),
        *record_args,
        1.0e-9,
        400,
        1.0e-7,
        "peintinger_eq13",
        True,
    )
    assert native.converged
    assert native.truncated_electrostatics_acknowledged

    result = run_omx_seccm(
        molecule,
        topology,
        variant="om2",
        max_iter=400,
        allow_truncated_electrostatics=True,
    )
    assert result.converged
    assert np.isfinite(result.energy)
    assert result.group_order == 2
    assert result.route_plan.electrostatics_kernel == "none"
    assert result.truncated_electrostatics_acknowledged
    assert result.route_plan.truncated_electrostatics_acknowledged


def test_omx_seccm_1d_chain_translation_invariance():
    from vibeqc.semiempirical import run_omx_seccm

    primitive = np.array([_OMX_CHAIN_CELL, 0.0, 0.0])
    replicas = 2
    coords = np.vstack([_H2O_COORDS + cell * primitive for cell in range(replicas)])

    def energy(geometry):
        result = run_omx_seccm(
            _h2o_chain_molecule(geometry),
            _topology(
                geometry,
                [replicas * primitive],
                replicas=(replicas, 1, 1),
                primitive_vectors=[primitive],
            ),
            variant="om2",
            max_iter=400,
            allow_truncated_electrostatics=True,
        )
        assert result.converged
        assert result.homo_lumo_gap > 0.0
        assert result.energy == pytest.approx(
            result.total_cyclic_energy / replicas, abs=1.0e-12
        )
        assert result.e_electronic + result.e_core == pytest.approx(
            result.energy, abs=1.0e-14
        )
        return result.energy

    base = energy(coords)
    # Translation invariance along the cyclic direction: the WS-weighted
    # supercell energy must not depend on where the cluster sits on the torus.
    shifted = coords + np.array([1.0, 0.0, 0.0])
    assert energy(shifted) == pytest.approx(base, abs=1.0e-10)


def test_omx_seccm_2d_sheet_axis_swap_invariance():
    from vibeqc.semiempirical import run_omx_seccm

    primitives = [
        np.array([_OMX_CHAIN_CELL, 0.0, 0.0]),
        np.array([0.0, _OMX_CHAIN_CELL, 0.0]),
    ]
    replicas = 2

    def sheet(sites):
        coords = np.array(
            [
                origin + site
                for idx in product(range(replicas), repeat=2)
                for origin in [idx[0] * primitives[0] + idx[1] * primitives[1]]
                for site in sites
            ]
        )
        result = run_omx_seccm(
            _h2o_chain_molecule(coords),
            _topology(
                coords,
                [replicas * p for p in primitives],
                replicas=(replicas, replicas, 1),
                primitive_vectors=primitives,
            ),
            variant="om2",
            max_iter=400,
            allow_truncated_electrostatics=True,
        )
        assert result.converged
        return result.energy

    # Lattice-symmetry invariance: swapping the two square-lattice axes
    # (with the cell geometry swapped the same way) must not move the energy.
    assert sheet(_H2O_COORDS) == pytest.approx(
        sheet(_H2O_COORDS[:, [1, 0, 2]]), abs=1.0e-10
    )


def test_omx_seccm_three_center_weighting_comparison():
    from vibeqc.semiempirical import run_omx_seccm

    molecule, topology = _h2o_chain(2)

    default = run_omx_seccm(
        molecule,
        topology,
        variant="om2",
        max_iter=400,
        allow_truncated_electrostatics=True,
    )
    explicit_eq13 = run_omx_seccm(
        molecule,
        topology,
        variant="om2",
        max_iter=400,
        three_center_weighting="peintinger_eq13",
        allow_truncated_electrostatics=True,
    )
    eq10 = run_omx_seccm(
        molecule,
        topology,
        variant="om2",
        max_iter=400,
        three_center_weighting="janetzko_eq10",
        allow_truncated_electrostatics=True,
    )

    # The production eq-13 scheme is the default; Janetzko's eq-10 is the
    # opt-in comparison scheme.  Both run and are finite, they are different
    # weightings, and the explicit default equals the implicit one.
    assert default.converged
    assert explicit_eq13.converged
    assert eq10.converged
    assert np.isfinite(eq10.energy)
    assert default.energy == pytest.approx(explicit_eq13.energy, abs=1.0e-14)
    assert default.energy != pytest.approx(eq10.energy, abs=1.0e-12)
    assert default.three_center_weighting == "peintinger_eq13"
    assert explicit_eq13.three_center_weighting == "peintinger_eq13"
    assert eq10.three_center_weighting == "janetzko_eq10"
    assert default.route_plan.three_center_weighting == "peintinger_eq13"
    assert explicit_eq13.route_plan.three_center_weighting == "peintinger_eq13"
    assert eq10.route_plan.three_center_weighting == "janetzko_eq10"
    assert default.truncated_electrostatics_acknowledged
    assert default.route_plan.truncated_electrostatics_acknowledged
    assert "extra_entries" not in default.route_plan.citation_assemble_kwargs
    assert eq10.route_plan.citation_assemble_kwargs["extra_entries"] == (
        "janetzko_ccm_2008",
    )

    from vibeqc.output.citations import load_default_database

    citations = load_default_database().assemble(
        **eq10.route_plan.citation_assemble_kwargs
    )
    assert "janetzko_ccm_2008" in {
        citation.key for citation in citations.citations
    }

    with pytest.raises(ValueError, match="three_center_weighting"):
        run_omx_seccm(
            molecule,
            topology,
            variant="om2",
            three_center_weighting="not_a_scheme",
        )


def test_omx_seccm_eq10_one_sided_endpoint_weight_is_exactly_zero():
    """Janetzko Eq. 10 contains both endpoint factors in the numerator."""
    weight = _se._omx_seccm_three_center_endpoint_weight

    assert weight("janetzko_eq10", 0.0, 1.0) == 0.0
    assert weight("janetzko_eq10", 0.5, 0.0) == 0.0
    assert weight("janetzko_eq10", 0.5, 1.0) == pytest.approx(1.0 / 3.0)
    assert weight("peintinger_eq13", 0.0, 1.0) == 0.5


def test_omx_seccm_eq13_weights_each_union_image_from_its_endpoints():
    """Pin Peintinger-Bredow eq. 13 on the asymmetric four-site oracle."""
    coords = np.arange(4.0)[:, None] * np.array([[1.0, 0.0, 0.0]])
    translation = np.array([4.0, 0.0, 0.0])
    topology = build_seccm_topology(
        coords,
        [translation],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )

    # Orbital pair M=0, N=1 at zero image; inspect C=2 independently of the
    # OMx implementation.  Absolute labels in WSSC(N-image) are N's local
    # labels shifted by the orbital-pair label.
    pair = next(
        image
        for image in topology.cells[0]
        if image.origin == 1 and image.image_shell_label == (0, 0, 0)
    )
    from_m = {
        (image.origin, image.image_shell_label): image.weight
        for image in topology.cells[0]
    }
    from_n = {
        (
            image.origin,
            tuple(
                local + shift
                for local, shift in zip(
                    image.image_shell_label,
                    pair.image_shell_label,
                    strict=True,
                )
            ),
        ): image.weight
        for image in topology.cells[1]
    }
    c_images = sorted(key for key in from_m.keys() | from_n.keys() if key[0] == 2)
    factors = [
        pair.weight * 0.5 * (from_m.get(key, 0.0) + from_n.get(key, 0.0))
        for key in c_images
    ]

    assert c_images == [(2, (-1, 0, 0)), (2, (0, 0, 0))]
    assert factors == [0.25, 0.75]

    # Exercise those unequal factors through production on the same record
    # geometry.  The absolute value is an integration pin for the independently
    # checked weights plus the source-derived one-centre NDDO contraction.
    from vibeqc.semiempirical import run_omx_seccm
    from vibeqc.semiempirical.methods.omx_params import load_omx_params
    from vibeqc.semiempirical.seccm._adapter_common import (
        flatten_topology_records,
    )

    bound = bind_finite_group(
        topology,
        primitive_vectors=[translation],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    molecule = Molecule(
        [
            Atom(Z, coord.tolist())
            for Z, coord in zip([7, 1, 1, 1], coords, strict=True)
        ],
        0,
        1,
    )
    message = "long-range Madelung/Ewald electrostatics"
    with pytest.raises(NotImplementedError, match=message):
        run_omx_seccm(
            molecule,
            bound,
            variant="om2",
            max_iter=400,
            three_center_weighting="peintinger_eq13",
        )
    record_args = flatten_topology_records(
        molecule,
        bound,
        route_name="OMx-SECCM",
    )
    with pytest.raises(ValueError, match=message):
        _se._run_omx_seccm_from_records(
            molecule,
            load_omx_params("om2"),
            *record_args,
            1.0e-9,
        )

    result = run_omx_seccm(
        molecule,
        bound,
        variant="om2",
        max_iter=400,
        three_center_weighting="peintinger_eq13",
        allow_truncated_electrostatics=True,
    )

    assert result.converged
    assert result.group_order == 1
    assert result.truncated_electrostatics_acknowledged
    assert result.route_plan.truncated_electrostatics_acknowledged
    assert result.energy == pytest.approx(-7.919908585032076, abs=1.0e-12)


# ---------------------------------------------------------------------------
# GFN2-xTB-SECCM (WS-weighted supercell; molecular limit delegates)
# ---------------------------------------------------------------------------


def test_gfn2_seccm_route_plan():
    plan = SemiempiricalRoutePlan.from_request("gfn2_xtb", boundary="seccm")
    assert plan.boundary == "seccm_direct_torus"
    assert plan.status_route == "gfn2-seccm-energy"
    assert plan.properties == ("energy",)
    assert plan.scc == "shell"
    assert plan.maturity == "experimental"

    with pytest.raises(NotImplementedError, match="neutral"):
        SemiempiricalRoutePlan.from_request("gfn2_xtb", boundary="seccm", charge=1)


def test_gfn2_seccm_molecular_limit_matches_molecular_gfn2():
    from vibeqc.semiempirical import run_gfn2_seccm
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)

    result = run_gfn2_seccm(molecule, topology)
    molecular = _se.xtb.run_gfn2_xtb(molecule, load_gfn2_params())

    assert result.converged
    assert result.energy == pytest.approx(molecular.energy, abs=1.0e-12)
    assert result.e_electronic == pytest.approx(molecular.e_electronic, abs=1.0e-12)
    assert result.e_repulsive == pytest.approx(molecular.e_repulsive, abs=1.0e-12)
    assert result.e_scc == pytest.approx(molecular.e_scc, abs=1.0e-12)
    assert result.e_band0 == pytest.approx(molecular.e_band0, abs=1.0e-12)
    assert result.e_aes == pytest.approx(molecular.e_aes, abs=1.0e-12)
    assert result.e_3rd == pytest.approx(molecular.e_3rd, abs=1.0e-12)
    assert result.total_cyclic_energy == pytest.approx(molecular.energy, abs=1.0e-12)
    assert result.cyclic_electronic_energy == pytest.approx(
        molecular.e_electronic, abs=1.0e-12
    )
    assert result.cyclic_repulsive_energy == pytest.approx(
        molecular.e_repulsive, abs=1.0e-12
    )
    assert result.cyclic_scc_energy == pytest.approx(molecular.e_scc, abs=1.0e-12)
    assert result.cyclic_aes_energy == pytest.approx(molecular.e_aes, abs=1.0e-12)
    assert result.cyclic_3rd_energy == pytest.approx(molecular.e_3rd, abs=1.0e-12)
    assert result.cyclic_band0_energy == pytest.approx(molecular.e_band0, abs=1.0e-12)
    np.testing.assert_allclose(result.charges, molecular.charges, atol=1.0e-12)
    np.testing.assert_allclose(result.mo_energies, molecular.mo_energies, atol=1.0e-12)
    molecular_gap = (
        molecular.mo_energies[molecular.n_occ]
        - molecular.mo_energies[molecular.n_occ - 1]
    )
    assert result.homo_lumo_gap == pytest.approx(molecular_gap, abs=1.0e-12)
    assert result.n_records == 6
    assert result.normalization == "per_primitive_cell"
    assert result.molecular_delegated
    identity = result.hamiltonian_identity
    assert result.route_plan.gfn2_hamiltonian_identity is identity
    assert identity.requested_electrostatics_family == "none"
    assert identity.resolved_electrostatics_kernel == "none"
    assert identity.requested_electronic_temperature == 0.0
    assert identity.resolved_electronic_temperature == 0.0
    assert identity.include_aes
    assert identity.molecular_delegated
    assert result.free_energy == pytest.approx(result.energy, abs=0.0)
    assert result.entropy == 0.0
    assert result.smearing_temperature == 0.0
    assert result.e_electronic + result.e_repulsive == pytest.approx(
        result.energy, abs=1.0e-14
    )


def test_gfn2_seccm_zero_record_one_atom_molecular_limit():
    from vibeqc.semiempirical import run_gfn2_seccm
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    coordinates = np.zeros((1, 3))
    translation = np.array([20.0, 0.0, 0.0])
    topology = _topology(
        coordinates,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    molecule = Molecule([Atom(2, coordinates[0].tolist())], 0, 1)

    result = run_gfn2_seccm(molecule, topology)
    molecular = _se.xtb.run_gfn2_xtb(molecule, load_gfn2_params())

    assert result.converged
    assert result.molecular_delegated
    assert result.n_records == 0
    assert result.energy == pytest.approx(molecular.energy, abs=1.0e-12)
    np.testing.assert_allclose(
        result.mo_energies,
        molecular.mo_energies,
        atol=1.0e-12,
    )


def _closed_shell_entropy_oracle(
    orbital_energies: np.ndarray,
    n_electrons: int,
    temperature: float,
) -> float:
    """Independent one-k-point Fermi entropy with two spin states."""
    energies = np.asarray(orbital_energies, dtype=float)

    def occupations(chemical_potential: float) -> np.ndarray:
        argument = np.clip((energies - chemical_potential) / temperature, -50.0, 50.0)
        return 2.0 / (1.0 + np.exp(argument))

    lower = float(energies.min() - 10.0 * temperature - 1.0)
    upper = float(energies.max() + 10.0 * temperature + 1.0)
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if float(occupations(middle).sum()) > n_electrons:
            upper = middle
        else:
            lower = middle
        if upper - lower < 1.0e-14:
            break
    occupation = occupations(0.5 * (lower + upper))
    fraction = np.clip(occupation / 2.0, 1.0e-300, 1.0 - 1.0e-15)
    return float(
        np.sum(
            -2.0
            * (fraction * np.log(fraction) + (1.0 - fraction) * np.log(1.0 - fraction))
        )
    )


def test_gfn2_seccm_trivial_group_only_cell_normalizes_per_primitive():
    """A non-covariant vacancy cell may bind a group without atom action.

    The record list is molecular in both cases, but the group-order-two
    result is still a per-primitive-cell quantity and must be divided by two.
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    molecule = _h2o_molecule(_H2O_COORDS)
    group_one = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    group_two = _topology(
        _H2O_COORDS,
        translations,
        replicas=(2, 1, 1),
        primitive_vectors=[0.5 * translations[0]],
    )

    whole = run_gfn2_seccm(molecule, group_one)
    per_primitive = run_gfn2_seccm(molecule, group_two)

    assert per_primitive.group_order == 2
    assert per_primitive.total_cyclic_energy == pytest.approx(
        whole.total_cyclic_energy, abs=0.0
    )
    assert per_primitive.energy == pytest.approx(0.5 * whole.energy, abs=1.0e-14)
    assert per_primitive.e_electronic == pytest.approx(
        0.5 * whole.e_electronic, abs=1.0e-14
    )
    assert per_primitive.e_repulsive == pytest.approx(
        0.5 * whole.e_repulsive, abs=1.0e-14
    )

    whole_hot = run_gfn2_seccm(molecule, group_one, electronic_temperature=0.1)
    primitive_hot = run_gfn2_seccm(molecule, group_two, electronic_temperature=0.1)
    assert primitive_hot.energy == pytest.approx(0.5 * whole_hot.energy, abs=1.0e-13)
    assert primitive_hot.entropy == pytest.approx(0.5 * whole_hot.entropy, abs=1.0e-13)
    assert primitive_hot.total_cyclic_energy == pytest.approx(
        whole_hot.total_cyclic_energy, abs=1.0e-13
    )
    assert primitive_hot.cyclic_electronic_energy == pytest.approx(
        whole_hot.cyclic_electronic_energy, abs=1.0e-13
    )


def test_gfn2_seccm_trivial_finite_temperature_reports_mermin_energy():
    """The delegated molecular path must expose A=E-T*S, not internal E."""
    from vibeqc.semiempirical import run_gfn2_seccm
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    temperature = 0.1
    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    result = run_gfn2_seccm(molecule, topology, electronic_temperature=temperature)

    options = _se.xtb.XTBSccOptions()
    options.electronic_temperature = temperature
    molecular = _se.xtb.run_gfn2_xtb(molecule, load_gfn2_params(), options)
    entropy = _closed_shell_entropy_oracle(
        molecular.mo_energies, 2 * molecular.n_occ, temperature
    )
    expected_free = molecular.energy - temperature * entropy

    assert molecular.entropy == pytest.approx(entropy, abs=1.0e-12)
    assert molecular.free_energy == pytest.approx(expected_free, abs=1.0e-12)
    assert molecular.smearing_temperature == pytest.approx(temperature, abs=0.0)
    assert result.energy == pytest.approx(expected_free, abs=1.0e-12)
    assert result.free_energy == pytest.approx(expected_free, abs=1.0e-12)
    assert result.total_cyclic_energy == pytest.approx(expected_free, abs=1.0e-12)
    assert result.entropy == pytest.approx(entropy, abs=1.0e-12)
    assert result.smearing_temperature == pytest.approx(temperature, abs=0.0)
    assert result.route_plan.electronic_temperature == pytest.approx(
        temperature, abs=0.0
    )
    assert result.e_electronic + result.e_repulsive == pytest.approx(
        result.energy, abs=1.0e-13
    )
    assert (
        result.cyclic_electronic_energy + result.cyclic_repulsive_energy
    ) == pytest.approx(result.total_cyclic_energy, abs=1.0e-13)

    shells = _gfn2_shell_context(molecule)
    assert result.shell_charges.size == len(shells)
    for atom in range(len(molecule.atoms)):
        shell_sum = sum(
            result.shell_charges[index]
            for index, shell in enumerate(shells)
            if int(shell.atom_idx) == atom
        )
        assert shell_sum == pytest.approx(-result.charges[atom], abs=1.0e-10)


def test_gfn2_seccm_trivial_capped_budget_fails_with_route_error():
    """A failed molecular delegate must not reconstruct empty occupations."""
    from vibeqc.semiempirical import (
        GFN2SECCMConvergenceError,
        run_gfn2_seccm,
    )

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    with pytest.raises(
        GFN2SECCMConvergenceError,
        match="did not converge within",
    ) as caught:
        run_gfn2_seccm(_h2o_molecule(_H2O_COORDS), topology, max_iter=1)
    failed = caught.value.result
    assert not failed.converged
    assert failed.selected_attempt_index is None
    assert failed.n_iter == 1
    assert failed.n_basis > 0
    assert failed.n_occ > 0
    assert len(failed.attempts) == 1
    assert failed.attempts[0].engine == "molecular_delegate"
    assert failed.attempts[0].exit_reason == "iteration_limit"
    assert failed.attempts[0].residual_trace == tuple(
        failed.scc_max_change_trace
    )


def test_gfn2_seccm_smeared_run_records_its_real_frontier_gap():
    """IID 422: a smeared GFN2-SECCM row must not report a placeholder gap.

    The gap block was gated on ``electronic_temperature <= 0.0``, so under
    finite-temperature stabilization the frontier gap was never measured and
    ``homo_lumo_gap`` kept its struct default of 0.0.  This LiF cell has a
    perfectly healthy 0.0180 Ha (0.49 eV) gap and was recorded as gapless --
    wrong in both directions: it fails a ``gap > 0`` screen and it passes a
    ``gap == 0`` metallicity screen.  (This is the wave-scale
    "113 ok rows at gap == 0.0, 87 of them from temp/temp_diis" signature.)
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    coordinates = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=float)
    molecule = Molecule(
        [Atom(3, coordinates[0].tolist()), Atom(9, coordinates[1].tolist())],
        0,
        1,
    )
    translations = [np.array([60.0, 0.0, 0.0])]
    topology = _topology(
        coordinates,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )

    result = run_gfn2_seccm(molecule, topology)

    assert result.converged
    # This row *is* finite-temperature stabilized ...
    assert result.smearing_temperature == pytest.approx(0.005, abs=0.0)
    # ... and it is genuinely gapped, so the guard is not waived.
    assert result.gap_guard_waived is False
    # The recorded gap is the real frontier gap, not a placeholder zero.
    energies = np.asarray(result.mo_energies)
    assert 0 < result.n_occ < result.n_basis
    assert result.homo_lumo_gap == pytest.approx(
        float(energies[result.n_occ] - energies[result.n_occ - 1]), abs=0.0
    )
    assert result.homo_lumo_gap > result.run_controls.finite_torus_gap_tolerance


def test_gfn2_seccm_trivial_retry_reports_actual_temperature_and_route():
    """A molecular finite-T retry must not be relabeled as a T=0 result."""
    from vibeqc.semiempirical import run_gfn2_seccm

    # LiF at 12 bohr is the molecular driver's active finite-temperature
    # stabilization fixture (also pinned in test_gfn2_xtb.py). The newer
    # simple+DIIS polyalgorithm now converges the historical BUG45 pyridine
    # case directly at T=0, so it is no longer a retry oracle.
    coordinates = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=float)
    molecule = Molecule(
        [Atom(3, coordinates[0].tolist()), Atom(9, coordinates[1].tolist())],
        0,
        1,
    )
    translations = [np.array([60.0, 0.0, 0.0])]
    topology = _topology(
        coordinates,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )

    result = run_gfn2_seccm(molecule, topology)

    assert result.molecular_delegated
    assert result.smearing_temperature == pytest.approx(0.005, abs=0.0)
    assert result.route_plan.electronic_temperature == pytest.approx(
        result.smearing_temperature, abs=0.0
    )
    assert result.route_plan.citation_assemble_kwargs[
        "electronic_temperature"
    ] == pytest.approx(0.005, abs=0.0)
    identity = result.hamiltonian_identity
    assert identity.requested_electronic_temperature == 0.0
    assert identity.resolved_electronic_temperature == pytest.approx(
        0.005, abs=0.0
    )
    assert identity.molecular_delegated
    assert result.free_energy == pytest.approx(result.energy, abs=0.0)
    assert result.e_electronic + result.e_repulsive == pytest.approx(
        result.energy, abs=1.0e-12
    )
    assert len(result.attempts) >= 2
    assert result.selected_attempt_index == len(result.attempts) - 1
    assert all(
        attempt.engine == "molecular_delegate"
        for attempt in result.attempts
    )
    assert all(
        attempt.restart_source == "neutral" for attempt in result.attempts
    )
    assert result.attempts[-1].electronic_temperature == pytest.approx(
        0.005, abs=0.0
    )
    assert sum(attempt.n_iter for attempt in result.attempts) == result.n_iter
    assert sum(
        len(attempt.residual_trace) for attempt in result.attempts
    ) == result.n_iter
    assert tuple(result.scc_max_change_trace) == tuple(
        value
        for attempt in result.attempts
        for value in attempt.residual_trace
    )


def test_gfn2_seccm_gap_retry_counts_against_total_iteration_budget():
    """Every rejected T=0 gap solve counts against the public SCC budget."""
    from vibeqc.semiempirical import (
        GFN2SECCMConvergenceError,
        run_gfn2_seccm,
    )

    # Two symmetry-equivalent H sites at half filling form a deliberately
    # tiny metallic finite torus. At 15 bohr the hard-Aufbau bonding/
    # antibonding split is below the insulating gap gate, while finite-T
    # occupations converge in one SCC step. This exercises the real native
    # charge map without relying on a long or numerically delicate bulk run.
    spacing = 15.0
    coordinates = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]], dtype=float)
    primitive = np.array([spacing, 0.0, 0.0])
    topology = _topology(
        coordinates,
        [2.0 * primitive],
        replicas=(2, 1, 1),
        primitive_vectors=[primitive],
    )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coordinates],
        0,
        1,
    )

    # Below the stabilization threshold, the converged degenerate T=0 state
    # fails at the typed gap gate instead of being mislabeled nonconverged.
    with pytest.raises(
        GFN2SECCMConvergenceError,
        match="positive finite-torus HOMO-LUMO gap",
    ) as caught:
        run_gfn2_seccm(molecule, topology, max_iter=200)
    rejected = caught.value.result
    assert rejected.homo_lumo_gap == pytest.approx(
        rejected.mo_energies[rejected.n_occ]
        - rejected.mo_energies[rejected.n_occ - 1],
        abs=0.0,
    )
    assert 0.0 < rejected.homo_lumo_gap <= (
        rejected.run_controls.finite_torus_gap_tolerance
    )

    max_iter = 2500
    result = run_gfn2_seccm(molecule, topology, max_iter=max_iter)

    assert result.converged and not result.molecular_delegated
    assert result.smearing_temperature == pytest.approx(0.005, abs=0.0)
    # One primary T=0 step, one slow-retry T=0 step rejected by the same
    # gap gate, then one converged finite-T step. The cumulative count pins
    # both exception paths and cannot exceed the caller's total budget.
    assert result.n_iter == 3
    assert result.n_iter <= max_iter
    assert [attempt.exit_reason for attempt in result.attempts] == [
        "gap_rejected",
        "gap_rejected",
        "converged",
    ]
    assert [attempt.n_iter for attempt in result.attempts] == [1, 1, 1]
    assert all(
        len(attempt.residual_trace) == attempt.n_iter
        for attempt in result.attempts
    )
    assert result.selected_attempt_index == 2
    from vibeqc.semiempirical import GFN2SECCMAttempt

    payload = json.loads(json.dumps(result.attempts[1].to_dict()))
    assert GFN2SECCMAttempt.from_dict(payload) == result.attempts[1]


def test_gfn2_seccm_bad_restart_recovers_from_neutral_fallback():
    """A failed supplied start must not leak into automatic retries."""
    from vibeqc.semiempirical import run_gfn2_seccm

    spacing = 8.0
    coordinates = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]])
    primitive = np.array([spacing, 0.0, 0.0])
    topology = _topology(
        coordinates,
        [2.0 * primitive],
        replicas=(2, 1, 1),
        primitive_vectors=[primitive],
    )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coordinates],
        0,
        1,
    )
    restart = np.array([8.0, -8.0])

    result = run_gfn2_seccm(
        molecule,
        topology,
        max_iter=2500,
        initial_shell_charges=restart,
    )

    assert result.converged and result.physical_basin
    assert result.n_iter == 501
    assert [attempt.restart_source for attempt in result.attempts] == [
        "supplied",
        "neutral",
    ]
    assert [attempt.exit_reason for attempt in result.attempts] == [
        "iteration_limit",
        "converged",
    ]
    assert result.attempts[0].n_iter == 500
    assert result.attempts[1].solver == "supercell_simple"
    assert result.selected_attempt_index == 1
    assert result.run_controls.restart.primary_source == "supplied"
    assert result.run_controls.restart.n_shell_charges == 2
    assert result.route_plan.gfn2_run_controls is result.run_controls


def test_gfn2_seccm_delegated_gap_rejection_keeps_attempt_result():
    """A molecular shortcut rejected by the SECCM gap gate stays inspectable."""
    from vibeqc.semiempirical import (
        GFN2SECCMConvergenceError,
        run_gfn2_seccm,
    )

    coordinates = np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    translation = np.array([60.0, 0.0, 0.0])
    topology = _topology(
        coordinates,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coordinates],
        0,
        1,
    )

    with pytest.raises(
        GFN2SECCMConvergenceError,
        match="positive finite-torus HOMO-LUMO gap",
    ) as caught:
        run_gfn2_seccm(molecule, topology, max_iter=200)

    rejected = caught.value.result
    assert rejected.molecular_delegated
    assert rejected.selected_attempt_index is None
    assert len(rejected.attempts) == 1
    assert rejected.attempts[0].engine == "molecular_delegate"
    assert rejected.attempts[0].exit_reason == "gap_rejected"
    assert rejected.attempts[0].scc_converged
    assert rejected.n_iter == len(rejected.scc_max_change_trace)


def test_gfn2_seccm_trivial_include_aes_false_executes_requested_model():
    """A material validation option must bypass the molecular shortcut."""
    from vibeqc.semiempirical import run_gfn2_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    complete = run_gfn2_seccm(molecule, topology)
    explicit = run_gfn2_seccm(
        molecule,
        topology,
        madelung=False,
        madelung_s_weighted=False,
        madelung_no_self=False,
        ewald_gamma=False,
        include_aes=True,
        ewald_gamma_molecular_onsite=False,
        electronic_temperature=0.0,
    )
    without_aes = run_gfn2_seccm(molecule, topology, include_aes=False)

    assert complete.molecular_delegated
    assert explicit.hamiltonian_identity == complete.hamiltonian_identity
    assert explicit.run_controls == complete.run_controls
    assert complete.route_plan.gfn2_run_controls is complete.run_controls
    assert not without_aes.molecular_delegated
    assert complete.hamiltonian_identity.include_aes
    assert not without_aes.hamiltonian_identity.include_aes
    assert without_aes.hamiltonian_identity != complete.hamiltonian_identity
    assert without_aes.e_aes == 0.0
    assert without_aes.cyclic_aes_energy == 0.0
    assert without_aes.energy != pytest.approx(complete.energy, abs=1.0e-8)


def test_gfn2_seccm_trivial_newton_fails_instead_of_silently_using_simple():
    """The custom SECCM Newton map is not a molecular-driver mixer."""
    from vibeqc.semiempirical import run_gfn2_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    with pytest.raises(ValueError, match="molecular-delegation"):
        run_gfn2_seccm(_h2o_molecule(_H2O_COORDS), topology, scc_mixer="newton")


def _h2o_grid_molecule(coords):
    """Cell-major H2O sites: atom 0 of each 3-atom cell is oxygen."""
    return Molecule(
        [Atom(8 if k % 3 == 0 else 1, c.tolist()) for k, c in enumerate(coords)],
        0,
        1,
    )


_BOHR = 1.8897259886


def _fcc_cu_2x2x2(
    a_angstrom: float,
) -> tuple[Molecule, list[np.ndarray], list[np.ndarray]]:
    """Fcc Cu 2x2x2 supercell from the rhombohedral primitive cell."""
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    translations = [2 * p for p in prim]
    molecule = Molecule(
        [Atom(29, (np.asarray(c) * _BOHR).tolist()) for c in atoms],
        0,
        1,
    )
    return molecule, atoms, translations


def _fcc_cu_topology(
    atoms: list[np.ndarray],
    translations: list[np.ndarray],
) -> object:
    prim = [0.5 * t for t in translations]
    topology = build_seccm_topology(
        [np.asarray(c) * _BOHR for c in atoms],
        [np.asarray(t) * _BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * _BOHR for p in prim],
        replicas=(2, 2, 2),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def _fcc_cu_4x4x4(
    a_angstrom: float,
) -> tuple[Molecule, list[np.ndarray], list[np.ndarray]]:
    """Fcc Cu 4x4x4 (64-atom) supercell from the rhombohedral cell."""
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(4)
        for j in range(4)
        for k in range(4)
    ]
    translations = [4 * p for p in prim]
    molecule = Molecule(
        [Atom(29, (np.asarray(c) * _BOHR).tolist()) for c in atoms],
        0,
        1,
    )
    topology = build_seccm_topology(
        [np.asarray(c) * _BOHR for c in atoms],
        [np.asarray(t) * _BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * _BOHR for p in prim],
        replicas=(4, 4, 4),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


@pytest.mark.slow
def test_gfn2_seccm_bulk_cu_4x4x4_is_metallic_and_size_converged():
    """#444: this named historical gate now pins explicit 3-D rejection.

    All energy/size/state assertions below are retained as historical
    evidence, not reachable production expectations. Original rationale:

    The 64-atom fcc Cu cyclic cluster reaches the metallic fixed
    point near the lattice minimum and is size-converged relative to
    the 8-atom cell (issue #130, pinned 2026-08-27 after the #43
    parameter-projection fix).

    Both cells converge in a handful of iterations to translation-
    symmetric, near-neutral states (issue #354 made the H0 coordination
    numbers cyclic; before it the 2x2x2 cell carried a spurious 0.39 e
    polarization that was the coordination artifact, not physics). The
    2x2x2 cell lies well above the 64-atom cell in energy per atom: the
    Wigner-Seitz truncation of the 8-atom torus is the size effect. The
    pre-#354 ladder (3.615 to 4.4 A) had its parabolic minimum at
    a = 3.7033 A, +2.4% versus the experimental 3.615 A; that ladder is
    not re-fitted here.

    Scope of the size claim: this pins that the 64-atom cell is
    symmetric, near-neutral and lower in energy per atom than the 8-atom
    one, which is what two cell sizes can show. It does NOT pin that the
    64-atom cell is converged in absolute terms. Issue #444 records that the
    ewald_gamma kernel's WS-folded Klopman-Ohno remainder has no
    thermodynamic limit, with a coefficient scaling as dq^2; Cu at
    q_rms 0.085 sits in the weakest regime of that defect, but the
    energy pinned here is expected to move when the gamma functional
    form is settled (#211/#425), and this test should be repinned then
    rather than treated as invariant across that change."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _fcc_cu_4x4x4(3.7)
    with pytest.raises(ValueError, match="3-D ewald_gamma.*no thermodynamic limit"):
        run_gfn2_seccm(
            molecule,
            topology,
            ewald_gamma=True,
            scc_mixer="broyden_eyert",
            electronic_temperature=0.001,
            max_iter=300,
        )
    # assert bulk.converged and bulk.physical_basin
    # Re-pinned with issue #354's cyclic coordination numbers
    # (-3.886828 -> -3.904530; orbit charge spread 0.456 -> 5.9e-4 e).
    # assert bulk.energy == pytest.approx(-3.904530, abs=1.0e-5)
    # assert bulk.n_iter < 60
    # bulk_q_rms = float(np.sqrt((np.asarray(bulk.charges) ** 2).mean()))
    # assert bulk_q_rms < 1.0e-3
    # assert bulk.translation_symmetry_charge_spread < 1.0e-3

    small_molecule, atoms, translations = _fcc_cu_2x2x2(3.7)
    with pytest.raises(ValueError, match="3-D ewald_gamma.*no thermodynamic limit"):
        run_gfn2_seccm(
            small_molecule,
            _fcc_cu_topology(atoms, translations),
            ewald_gamma=True,
            scc_mixer="broyden_eyert",
            electronic_temperature=0.001,
            max_iter=300,
        )
    # small_q_rms = float(np.sqrt((np.asarray(small.charges) ** 2).mean()))
    # Both cells are symmetric and near-neutral; the size effect is the
    # energy per atom (2x2x2 sits 0.12 Ha/atom above the 64-atom cell).
    # assert small.converged and small.physical_basin
    # assert small_q_rms < 1.0e-3
    # assert small.translation_symmetry_charge_spread < 1.0e-3
    # assert small.energy > bulk.energy + 0.05


def test_gfn2_seccm_bulk_cu_sane_basin_is_physical():
    """The 2x2x2 fcc Cu fixed point at the experimental lattice constant
    is physical: finite shell polarization, physical_basin True, and the
    per-cell energy pinned (IID 130 regression anchor)."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _fcc_cu_2x2x2(3.615)
    topology = _fcc_cu_topology(atoms, translations)
    result = run_gfn2_seccm(molecule, topology, electronic_temperature=0.005)
    assert result.converged
    assert result.physical_basin
    # Re-pinned after the issue #43 per-angular-momentum parameter
    # projection (pre-fix: -3.796756 on the rotated d-first parameters),
    # and again with issue #354's cyclic H0 coordination numbers
    # (-3.773069 -> -3.779587241). fcc Cu is a single translation orbit,
    # so the corrected map's fixed point is translation-symmetric.
    assert result.energy == pytest.approx(-3.779587241, abs=1.0e-3)
    assert result.translation_symmetry_charge_spread < 1.0e-3
    assert result.shell_charges.size == 24  # 8 atoms x 3 shells (s, p, d)
    assert np.isfinite(np.asarray(result.shell_charges)).all()
    shell_rms = float(np.sqrt((np.asarray(result.shell_charges) ** 2).mean()))
    # The sane metallic fixed point carries ordinary shell polarization.
    assert shell_rms < 2.5


def test_gfn2_seccm_physical_basin_gate_cannot_be_diluted_by_system_size():
    """One 11.5-electron outlier must fail even when 99 atoms dilute RMS."""
    for size in (100, 1000, 10000):
        localized = np.full(size, -11.5 / (size - 1.0))
        localized[0] = 11.5
        assert localized.sum() == pytest.approx(0.0, abs=1.0e-12)
        assert float(np.sqrt(np.mean(localized**2))) < 2.0
        assert not _se._gfn2_seccm_local_charge_state_is_physical(
            localized, localized
        )
    assert _se._gfn2_seccm_local_charge_state_is_physical(
        np.array([0.8, -0.8]), np.array([0.8, -0.8])
    )


def test_gfn2_seccm_bulk_cu_compressed_converges_physical():
    """History: on the pre-#43 tree the compressed 2x2x2 fcc Cu map had
    an over-polarized intra-atomic shell-transfer fixed point that the
    basin gate failed closed (the original IID 130 symptom). That basin
    was an artifact of the rotated d-first parameters: with the
    per-angular-momentum projection the compressed cell converges into
    the physical basin. The pin guards against the artifact's return
    (a regression would either flip physical_basin or move the energy
    by tens of mHa)."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _fcc_cu_2x2x2(3.434)
    topology = _fcc_cu_topology(atoms, translations)
    result = run_gfn2_seccm(
        molecule,
        topology,
        electronic_temperature=0.005,
        max_iter=800,
    )
    assert result.converged and result.physical_basin
    # Re-pinned with issue #354's cyclic H0 coordination numbers
    # (-3.755123 -> -3.762152203); the basin is unchanged.
    assert result.energy == pytest.approx(-3.762152203, abs=1.0e-3)
    shell_rms = float(
        np.sqrt((np.asarray(result.shell_charges) ** 2).mean())
    )
    assert shell_rms < 0.5


def _rocksalt_mgo_2x2x2(
    a_angstrom: float,
) -> tuple[Molecule, list[np.ndarray], list[np.ndarray]]:
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for site in range(2):
                    atoms.append(i * prim[0] + j * prim[1] + k * prim[2] + basis[site])
                    zs.append(12 if site == 0 else 8)
    translations = [2 * p for p in prim]
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * _BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    return molecule, atoms, translations


def _rocksalt_topology(
    atoms: list[np.ndarray], translations: list[np.ndarray]
) -> object:
    prim = [0.5 * t for t in translations]
    topology = build_seccm_topology(
        [np.asarray(c) * _BOHR for c in atoms],
        [np.asarray(t) * _BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * _BOHR for p in prim],
        replicas=(2, 2, 2),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def test_gfn2_seccm_mgo_embedding_restores_minimum_and_ionicity():
    """The opt-in Madelung/Ewald embedding (neutral ionic cells) restores
    the missing image-sum tail: the embedded 2x2x2 MgO curve has a
    minimum near a = 4.30 A (the unembedded curve decreases monotonically
    to below 3.9 A), the Mulliken charges relax to the xtb scale
    (q_rms about 0.4, xtb gives 0.41), and the self-energy is reported
    as e_madelung."""
    from vibeqc.semiempirical import run_gfn2_seccm

    def energy(a: float):
        molecule, atoms, translations = _rocksalt_mgo_2x2x2(a)
        return run_gfn2_seccm(
            molecule,
            _rocksalt_topology(atoms, translations),
            madelung=True,
        )

    e_430 = energy(4.30)
    e_4212 = energy(4.212)
    assert e_430.converged and e_430.physical_basin
    assert e_4212.converged and e_4212.physical_basin
    # Minimum near 4.30 A: both neighbours are higher.
    assert e_4212.energy > e_430.energy
    assert e_430.e_madelung > 0.0
    q_rms = float(np.sqrt((np.asarray(e_430.charges) ** 2).mean()))
    assert 0.2 < q_rms < 0.8
    # The energy decomposition exposes the embedding channel.
    assert e_430.cyclic_madelung_energy == pytest.approx(
        e_430.e_madelung * e_430.group_order, abs=1.0e-10
    )


def test_gfn2_seccm_molecular_limit_with_embedding_is_finite():
    """With madelung=True the molecular-limit delegation is skipped and
    the embedded supercell engine runs; the 40-bohr 1-D wire kernel keeps
    the H2O result finite and converged."""
    from vibeqc.semiempirical import run_gfn2_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    result = run_gfn2_seccm(molecule, topology, madelung=True)
    s_weighted = run_gfn2_seccm(
        molecule,
        topology,
        madelung=True,
        madelung_s_weighted=True,
    )
    assert result.converged
    assert result.physical_basin
    assert np.isfinite(result.energy)
    assert abs(result.e_madelung) < 0.05
    assert result.hamiltonian_identity.requested_electrostatics_family == (
        "madelung"
    )
    assert not result.hamiltonian_identity.madelung_s_weighted
    assert s_weighted.hamiltonian_identity.madelung_s_weighted
    assert s_weighted.hamiltonian_identity != result.hamiltonian_identity


def test_gfn2_seccm_madelung_no_self_uses_one_operator_for_energy_and_fock():
    """The diagnostic diagonal removal must also reach reported E_mad."""
    from vibeqc.semiempirical import run_gfn2_seccm
    from vibeqc.semiempirical.methods.msindo_ccm import (
        _ewald_ws_cells,
        _madkonst_2d,
        _madelung_potential_ewald,
    )

    translations = [
        np.array([40.0, 0.0, 0.0]),
        np.array([0.0, 40.0, 0.0]),
    ]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    result = run_gfn2_seccm(
        _h2o_molecule(_H2O_COORDS),
        topology,
        madelung=True,
        madelung_no_self=True,
    )

    ewald_cells = _ewald_ws_cells(topology)
    madkonst = _madkonst_2d(ewald_cells, topology.translations, len(result.charges))
    charges = np.asarray(result.charges)
    full_potential = _madelung_potential_ewald(charges, madkonst, ewald_cells)
    effective_potential = full_potential - np.diag(madkonst) * charges
    expected = 0.5 * float(charges @ effective_potential)
    historical_full_self = 0.5 * float(charges @ full_potential)

    assert result.e_madelung == pytest.approx(expected, abs=1.0e-10)
    assert result.cyclic_madelung_energy == pytest.approx(expected, abs=1.0e-10)
    assert abs(expected - historical_full_self) > 1.0e-3
    assert result.hamiltonian_identity.madelung_no_self
    assert result.hamiltonian_identity.resolved_electrostatics_kernel == (
        "madelung_parry_2d"
    )

    # Independent fixed-q derivative check: the same diagonal-removed
    # potential is d[1/2 q.M_eff.q]/dq, which is the quantity deposited in
    # the SCC Hamiltonian by the shared production helper.
    def energy_at(trial_charges: np.ndarray) -> float:
        potential = (
            _madelung_potential_ewald(trial_charges, madkonst, ewald_cells)
            - np.diag(madkonst) * trial_charges
        )
        return 0.5 * float(trial_charges @ potential)

    step = 1.0e-6
    derivative = np.zeros_like(charges)
    for atom in range(charges.size):
        plus = charges.copy()
        minus = charges.copy()
        plus[atom] += step
        minus[atom] -= step
        derivative[atom] = (energy_at(plus) - energy_at(minus)) / (2.0 * step)
    np.testing.assert_allclose(derivative, effective_potential, atol=1.0e-8)
    assert result.e_electronic + result.e_repulsive == pytest.approx(
        result.energy, abs=1.0e-12
    )
    assert (
        result.cyclic_electronic_energy + result.cyclic_repulsive_energy
    ) == pytest.approx(result.total_cyclic_energy, abs=1.0e-12)


def test_gfn2_seccm_bulk_cu_embedding_converges_physical():
    """History: on the pre-#43 tree the compressed-Cu over-polarized
    basin was intra-atomic shell transfer that the atomic-level Madelung
    tail could not mask, so the embedded run failed closed. The basin
    was an artifact of the rotated d-first parameters; the embedded
    compressed-Cu run now converges into the physical basin and the pin
    guards against the artifact's return."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _fcc_cu_2x2x2(3.434)
    topology = _fcc_cu_topology(atoms, translations)
    result = run_gfn2_seccm(
        molecule,
        topology,
        electronic_temperature=0.005,
        max_iter=800,
        madelung=True,
    )
    assert result.converged and result.physical_basin
    # Re-pinned with issue #354's cyclic H0 coordination numbers
    # (-3.756401160 -> -3.761968564); the basin is unchanged.
    assert result.energy == pytest.approx(-3.761968564, abs=1.0e-6)
    q_rms = float(np.sqrt((np.asarray(result.charges) ** 2).mean()))
    assert q_rms < 0.5


# ---------------------------------------------------------------------------
# GFN2-xTB-SECCM ewald_gamma (self-consistent periodic shell gamma)
# ---------------------------------------------------------------------------


def _assemble_shell_gamma_reference(
    shell_infos,
    coords_bohr: np.ndarray,
    topology,
    ewald_pair,
    eta_scale: float | None = 1.0,
) -> np.ndarray:
    """Shared Python reimplementation of the tblite get_amat_3d
    construction (docs/design_seccm_gfn2_long_range_gamma.md): the
    dimension-specific bare-Coulomb Ewald value per pair (supplied by
    ``ewald_pair(d, same_site)``) plus the WS-folded KO-minus-Coulomb
    correction plus the molecular on-site hardness. ``eta_scale=None``
    returns the bare-Coulomb Ewald alone (no correction, no on-site),
    the eta-to-zero anchor convention."""
    import math

    n_shells = len(shell_infos)
    if eta_scale is not None:
        hardness = (
            np.array([max(float(si.hardness), 1.0e-6) for si in shell_infos])
            * eta_scale
        )

    gamma = np.zeros((n_shells, n_shells))
    for a in range(n_shells):
        atom_a = int(shell_infos[a].atom_idx)
        for b in range(a, n_shells):
            atom_b = int(shell_infos[b].atom_idx)
            d = coords_bohr[atom_b] - coords_bohr[atom_a]
            same_site = atom_a == atom_b
            value = ewald_pair(d, same_site)
            if eta_scale is None:
                gamma[a, b] = value
                gamma[b, a] = value
                continue
            eta_ab = 2.0 / (hardness[a] + hardness[b])
            for image in topology.cells[atom_a]:
                if image.origin != atom_b:
                    continue
                r = float(np.linalg.norm(image.disp))
                if r < 1.0e-12:
                    continue
                f = 1.0 / math.sqrt(r * r + eta_ab * eta_ab)
                value += image.weight * (f - 1.0 / r)
            if same_site:
                value += 1.0 / eta_ab
            gamma[a, b] = value
            gamma[b, a] = value
    return gamma


def _ewald_shell_gamma_reference(
    shell_infos,
    coords_bohr: np.ndarray,
    topology,
    alpha: float,
    eta_scale: float | None = 1.0,
) -> np.ndarray:
    """3-D reference: full Ewald Coulomb (erfc real space + reciprocal
    series + self) with the same cutoffs and enumeration as the C++
    ewald_shell_gamma_3d."""
    import math

    T = np.asarray(topology.translations)
    vol = abs(np.linalg.det(T))
    rec = np.linalg.inv(T).T * 2.0 * math.pi
    g_cut = alpha * math.sqrt(120.0)
    gmax = [
        int(math.ceil(g_cut * np.linalg.norm(T[i]) / (2.0 * math.pi))) + 1
        for i in range(3)
    ]
    r_cut = math.sqrt(30.0) / alpha
    fractional = coords_bohr @ np.linalg.inv(T)
    pair_fractional_span = np.ptp(fractional, axis=0)
    nmax = [
        int(
            math.ceil(
                r_cut * np.linalg.norm(rec[i]) / (2.0 * math.pi)
                + pair_fractional_span[i]
            )
        )
        + 1
        for i in range(3)
    ]

    reciprocal: list[tuple[np.ndarray, float]] = []
    for i in range(-gmax[0], gmax[0] + 1):
        for j in range(-gmax[1], gmax[1] + 1):
            for k in range(-gmax[2], gmax[2] + 1):
                if i == j == k == 0:
                    continue
                g = i * rec[0] + j * rec[1] + k * rec[2]
                g2 = float(g @ g)
                if g2 > g_cut * g_cut:
                    continue
                reciprocal.append(
                    (
                        g,
                        4.0
                        * math.pi
                        / vol
                        * math.exp(-g2 / (4.0 * alpha * alpha))
                        / g2,
                    )
                )

    images = np.asarray(
        [
            i * T[0] + j * T[1] + k * T[2]
            for i in range(-nmax[0], nmax[0] + 1)
            for j in range(-nmax[1], nmax[1] + 1)
            for k in range(-nmax[2], nmax[2] + 1)
        ]
    )
    self_const = 2.0 * alpha / math.sqrt(math.pi)
    # Neutralising background of the 3-D Ewald sum.  Without it the G = 0
    # term of the reciprocal series is simply dropped, which leaves every
    # matrix element dependent on the arbitrary splitting parameter alpha
    # (measured on the 10-degree acute H2O cell: gamma[0, 0] runs 0.6347 ->
    # 0.5545 as alpha triples, while the shipped kernel holds 0.543268472823
    # to twelve digits).  It shifts every element by the same constant, so
    # every neutral-cell energy is identical either way -- which is why the
    # omission survived: it is invisible to any quantity the SCC consumes.
    background = -math.pi / (vol * alpha * alpha)

    def ewald_pair(d: np.ndarray, same_site: bool) -> float:
        value = background
        for shift in images:
            r = float(np.linalg.norm(d + shift))
            if r < 1.0e-12 or r > r_cut:
                continue
            value += math.erfc(alpha * r) / r
        for g, pref in reciprocal:
            value += pref * math.cos(float(g @ d))
        if same_site:
            value -= self_const
        return value

    return _assemble_shell_gamma_reference(
        shell_infos, coords_bohr, topology, ewald_pair, eta_scale
    )


def _ewald_shell_gamma_reference_2d(
    shell_infos,
    coords_bohr: np.ndarray,
    topology,
    alpha: float,
    eta_scale: float | None = 1.0,
) -> np.ndarray:
    """2-D reference: the Parry/Heyes Ewald Coulomb (erfc real space +
    reciprocal series + K=0 term + self) with the same cutoffs and
    enumeration as the C++ ewald_shell_gamma_2d (mirrors the validated
    indo::_madkonst_2d arithmetic)."""
    import math

    T = np.asarray(topology.translations)
    a1, a2 = T[0], T[1]
    normal = np.cross(a1, a2)
    area = float(np.linalg.norm(normal))
    nhat = normal / area
    twopi = 2.0 * math.pi
    b1 = twopi * np.cross(a2, nhat) / area
    b2 = twopi * np.cross(nhat, a1) / area

    k_cut = alpha * math.sqrt(120.0)
    kmax = [
        int(math.ceil(k_cut * np.linalg.norm(a1) / twopi)) + 1,
        int(math.ceil(k_cut * np.linalg.norm(a2) / twopi)) + 1,
    ]
    kvecs = [
        i * b1 + j * b2
        for i in range(-kmax[0], kmax[0] + 1)
        for j in range(-kmax[1], kmax[1] + 1)
        if not (i == j == 0)
        and float((i * b1 + j * b2) @ (i * b1 + j * b2)) <= k_cut * k_cut
    ]

    r_cut = math.sqrt(30.0) / alpha
    fractional = np.column_stack((coords_bohr @ b1 / twopi, coords_bohr @ b2 / twopi))
    pair_fractional_span = np.ptp(fractional, axis=0)
    nmax = [
        int(math.ceil(r_cut * np.linalg.norm(b1) / twopi + pair_fractional_span[0]))
        + 1,
        int(math.ceil(r_cut * np.linalg.norm(b2) / twopi + pair_fractional_span[1]))
        + 1,
    ]
    images = np.asarray(
        [
            i * a1 + j * a2
            for i in range(-nmax[0], nmax[0] + 1)
            for j in range(-nmax[1], nmax[1] + 1)
        ]
    )
    self_const = 2.0 * alpha / math.sqrt(math.pi)
    recip_pref = math.pi / area
    k0_pref = twopi / area

    def ewald_pair(d: np.ndarray, same_site: bool) -> float:
        value = 0.0
        for shift in images:
            r = float(np.linalg.norm(d + shift))
            if r < 1.0e-12 or r > r_cut:
                continue
            value += math.erfc(alpha * r) / r
        zc = float(d @ nhat)
        for k in kvecs:
            km = float(np.linalg.norm(k))
            krij = float(k @ d)
            value += (
                recip_pref
                * math.cos(krij)
                / km
                * (
                    math.exp(km * zc) * math.erfc(alpha * zc + km / (2.0 * alpha))
                    + math.exp(-km * zc) * math.erfc(-alpha * zc + km / (2.0 * alpha))
                )
            )
        alpha_z = alpha * zc
        value -= k0_pref * (
            zc * math.erf(alpha_z)
            + math.exp(-alpha_z * alpha_z) / (alpha * math.sqrt(math.pi))
        )
        if same_site:
            value -= self_const
        return value

    return _assemble_shell_gamma_reference(
        shell_infos, coords_bohr, topology, ewald_pair, eta_scale
    )


def _gfn2_shell_context(molecule):
    from vibeqc._vibeqc_core.semiempirical import (
        SemiempiricalBasis,
        gfn2_enumerate_shells,
    )
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    params = load_gfn2_params()
    basis = SemiempiricalBasis.build(molecule, params, 0)
    shells = list(gfn2_enumerate_shells(basis, molecule, params))
    return shells


def _run_native_gfn2_seccm_from_records(molecule, topology, **kwargs):
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
    from vibeqc.semiempirical.seccm._adapter_common import (
        flatten_topology_records,
        topology_length_unit_scale,
    )

    records = flatten_topology_records(
        molecule,
        topology,
        route_name="GFN2-SECCM native test",
    )
    group = topology.finite_group
    assert group is not None
    return _se._run_gfn2_seccm_from_records(
        molecule,
        load_gfn2_params(),
        records[0],
        records[1],
        records[2],
        np.asarray(records[3], dtype=np.int32),
        records[4],
        records[5],
        np.asarray(records[6], dtype=float),
        records[7],
        records[8],
        float(group.geometry_tolerance) * topology_length_unit_scale(topology),
        **kwargs,
    )


def _native_gfn2_ewald_shell_gamma(
    molecule: Molecule,
    topology,
    alpha: float,
    *,
    k0_global: bool = False,
) -> np.ndarray:
    """Call shell-gamma kernels without SCC; 3-D is diagnostic only (#444)."""
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
    from vibeqc.semiempirical.seccm._adapter_common import (
        flatten_topology_records,
    )

    records = flatten_topology_records(
        molecule, topology, route_name="GFN2-SECCM Ewald test"
    )
    return np.asarray(
        _se._gfn2_seccm_ewald_shell_gamma_from_records(
            molecule,
            load_gfn2_params(),
            records[0],
            records[1],
            records[2],
            np.asarray(records[3], dtype=np.int32),
            records[4],
            records[5],
            np.asarray(records[6], dtype=float),
            float(alpha),
            False,
            k0_global,
        )
    )


def _brute_force_ewald_shell_gamma_reference(
    shell_infos,
    coords_bohr: np.ndarray,
    topology,
    alpha: float,
    *,
    coefficient_limit: int = 36,
) -> np.ndarray:
    """Independent fixed-box oracle for acute 2-D and 3-D Ewald cells.

    The deliberately oversized coefficient cube does not reuse production's
    perpendicular-height bounds.  Boundary assertions independently prove
    that both the direct- and reciprocal-space cutoff spheres fit inside it.
    """
    import math

    from scipy.special import erfc

    translations = np.asarray(topology.translations, dtype=float)
    dimension = int(topology.dimensionality)
    coefficients = np.asarray(
        list(
            product(
                range(-coefficient_limit, coefficient_limit + 1),
                repeat=dimension,
            )
        ),
        dtype=float,
    )
    shifts = coefficients @ translations
    r_cut = math.sqrt(30.0) / alpha

    if dimension == 3:
        volume = abs(float(np.linalg.det(translations)))
        reciprocal_basis = np.linalg.inv(translations).T * (2.0 * math.pi)
        reciprocal_vectors = coefficients @ reciprocal_basis
        reciprocal_norm = np.linalg.norm(reciprocal_vectors, axis=1)
        reciprocal_mask = (reciprocal_norm > 1.0e-12) & (
            reciprocal_norm <= alpha * math.sqrt(120.0)
        )
        assert (
            np.max(np.abs(coefficients[reciprocal_mask]), initial=0.0)
            < coefficient_limit
        )
        reciprocal_vectors = reciprocal_vectors[reciprocal_mask]
        reciprocal_norm = reciprocal_norm[reciprocal_mask]
        reciprocal_weights = (
            4.0
            * math.pi
            / volume
            * np.exp(-(reciprocal_norm**2) / (4.0 * alpha**2))
            / reciprocal_norm**2
        )

        # Neutralising background: the G = 0 term of the reciprocal series.
        # See _ewald_shell_gamma_reference_3d for why omitting it leaves
        # every element alpha-dependent while changing no neutral energy.
        background = -math.pi / (volume * alpha * alpha)

        def ewald_pair(d: np.ndarray, same_site: bool) -> float:
            distances = np.linalg.norm(shifts + d, axis=1)
            direct_mask = (distances > 1.0e-12) & (distances <= r_cut)
            assert (
                np.max(np.abs(coefficients[direct_mask]), initial=0.0)
                < coefficient_limit
            )
            value = background + float(
                np.sum(erfc(alpha * distances[direct_mask]) / distances[direct_mask])
            )
            value += float(
                reciprocal_weights
                @ np.cos(reciprocal_vectors @ np.asarray(d, dtype=float))
            )
            if same_site:
                value -= 2.0 * alpha / math.sqrt(math.pi)
            return value

    elif dimension == 2:
        a1, a2 = translations
        normal = np.cross(a1, a2)
        area = float(np.linalg.norm(normal))
        nhat = normal / area
        twopi = 2.0 * math.pi
        reciprocal_basis = np.asarray(
            [
                twopi * np.cross(a2, nhat) / area,
                twopi * np.cross(nhat, a1) / area,
            ]
        )
        reciprocal_vectors = coefficients @ reciprocal_basis
        reciprocal_norm = np.linalg.norm(reciprocal_vectors, axis=1)
        reciprocal_mask = (reciprocal_norm > 1.0e-12) & (
            reciprocal_norm <= alpha * math.sqrt(120.0)
        )
        assert (
            np.max(np.abs(coefficients[reciprocal_mask]), initial=0.0)
            < coefficient_limit
        )
        reciprocal_vectors = reciprocal_vectors[reciprocal_mask]
        reciprocal_norm = reciprocal_norm[reciprocal_mask]

        def ewald_pair(d: np.ndarray, same_site: bool) -> float:
            distances = np.linalg.norm(shifts + d, axis=1)
            direct_mask = (distances > 1.0e-12) & (distances <= r_cut)
            assert (
                np.max(np.abs(coefficients[direct_mask]), initial=0.0)
                < coefficient_limit
            )
            value = float(
                np.sum(erfc(alpha * distances[direct_mask]) / distances[direct_mask])
            )
            zc = float(np.asarray(d) @ nhat)
            km = reciprocal_norm
            reciprocal_factor = np.exp(km * zc) * erfc(
                alpha * zc + km / (2.0 * alpha)
            ) + np.exp(-km * zc) * erfc(-alpha * zc + km / (2.0 * alpha))
            value += float(
                math.pi
                / area
                * np.sum(
                    np.cos(reciprocal_vectors @ np.asarray(d)) * reciprocal_factor / km
                )
            )
            value -= (
                twopi
                / area
                * (
                    zc * math.erf(alpha * zc)
                    + math.exp(-((alpha * zc) ** 2)) / (alpha * math.sqrt(math.pi))
                )
            )
            if same_site:
                value -= 2.0 * alpha / math.sqrt(math.pi)
            return value

    else:  # pragma: no cover - helper contract is pinned by its callers.
        raise AssertionError("Ewald oracle requires two or three dimensions")

    return _assemble_shell_gamma_reference(
        shell_infos, coords_bohr, topology, ewald_pair, eta_scale=1.0
    )


_GFN2_3D_GAMMA_ERROR = (
    "GFN2-SECCM 3-D ewald_gamma is unavailable with the Klopman-Ohno "
    "shell gamma on non-molecular cyclic topologies: its WS-folded "
    "remainder has no thermodynamic limit. Pass gamma_form=Elstner "
    "for a kernel that does (#444)."
)


@pytest.mark.parametrize(
    "synthetic, expected",
    [
        (True, (0.361531884674, 0.464251887966)),
        (False, (0.144635757778, 0.255217985354)),
    ],
    ids=["synthetic-cubic-mg4o4", "b1-rocksalt-control"],
)
def test_gfn2_seccm_3d_ws_ko_remainder_cluster_drift(synthetic, expected):
    """Fixed-charge diagnostic, not SCF or a two-point asymptotic fit (#444).

    The historical probe used the first species pattern below, not B1.
    On-site and Ewald point-Coulomb energy per conventional cell stay fixed;
    the entire observed size step is in the WS(KO-1/R) remainder.
    """
    import math

    sites = np.array([
        (0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5),
        (.5, .5, .5), (0, 0, .5), (0, .5, 0), (.5, 0, 0),
    ])
    species = (
        [12, 8, 8, 8, 12, 12, 12, 8] if synthetic
        else [12, 12, 12, 12, 8, 8, 8, 8]
    )
    # Preserve the synthetic provenance; nearest unlike neighbors distinguish
    # true B1. In particular site 0 and site 5 are nearest and both Mg only
    # in the synthetic pattern.
    assert (species[0] == species[5]) == synthetic
    side = 4.212 * _BOHR
    totals, remainders, onsite_energies, point_energies = [], [], [], []
    for n in (1, 2):
        coords = np.vstack([
            (sites + [i, j, k]) * side
            for i in range(n) for j in range(n) for k in range(n)
        ])
        zs = species * n**3
        molecule = Molecule(
            [Atom(z, xyz.tolist()) for z, xyz in zip(zs, coords)], 0, 1
        )
        topology = _topology(
            coords, n * side * np.eye(3), replicas=(n, n, n),
            primitive_vectors=side * np.eye(3),
        )
        shells = _gfn2_shell_context(molecule)
        dq = np.zeros(len(shells))
        hardness = np.zeros(len(zs))
        seen = set()
        for index, shell in enumerate(shells):
            atom = int(shell.atom_idx)
            if atom not in seen:
                seen.add(atom)
                dq[index] = -1.0 if zs[atom] == 12 else 1.0
                hardness[atom] = max(float(shell.hardness), 1.0e-6)
        q = np.where(np.asarray(zs) == 12, -1.0, 1.0)
        assert dq.sum() == 0.0
        # Neutrality does not cancel the pair-dependent R^-3 coefficient.
        # Four of each species per conventional cell gives 16 times the
        # bracket below, independent of site ordering and replication.
        h_mg = hardness[np.asarray(zs) == 12][0]
        h_o = hardness[np.asarray(zs) == 8][0]
        expected_coefficient = 16.0 * (
            1.0 / h_mg**2 + 1.0 / h_o**2 - 8.0 / (h_mg + h_o)**2
        )
        pair_eta = 2.0 / (hardness[:, None] + hardness[None, :])
        coefficient = float(q @ (pair_eta**2) @ q) / n**6
        assert coefficient == pytest.approx(expected_coefficient, rel=1.0e-12)
        assert coefficient > 0.0
        alpha = math.sqrt(math.pi) / (n * side)
        gamma = _native_gfn2_ewald_shell_gamma(molecule, topology, alpha)
        total = 0.5 * float(dq @ gamma @ dq) / n**3
        remainder = 0.0
        for atom, records in enumerate(topology.cells):
            for record in records:
                r = float(np.linalg.norm(record.disp))
                if r < 1.0e-12:
                    continue
                other = int(record.origin)
                eta = 2.0 / (hardness[atom] + hardness[other])
                remainder += (
                    0.5 * q[atom] * q[other] * record.weight
                    * (1.0 / math.sqrt(r*r + eta*eta) - 1.0/r)
                )
        remainder /= n**3
        onsite = 0.5 * float(hardness @ (q*q)) / n**3
        totals.append(total)
        remainders.append(remainder)
        onsite_energies.append(onsite)
        point_energies.append(total - remainder - onsite)
    np.testing.assert_allclose(totals, expected, rtol=0, atol=1.0e-8)
    assert totals[1] - totals[0] > 0.1
    assert onsite_energies[1] == pytest.approx(onsite_energies[0], abs=1.0e-12)
    assert point_energies[1] == pytest.approx(point_energies[0], abs=1.0e-9)
    assert totals[1] - totals[0] == pytest.approx(
        remainders[1] - remainders[0], abs=1.0e-9
    )


@pytest.mark.parametrize("replicas_x", [1, 2])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("options", [
    {}, {"include_aes": False}, {"ewald_gamma_molecular_onsite": True},
    {"ewald_gamma_k0_global": True},
])
def test_gfn2_seccm_3d_ewald_gamma_fails_closed(replicas_x, native, options):
    """Neither entry point nor a one-element group bypasses #444.

    The refusal is scoped to the *kernel*, not to the boundary: it is the
    Klopman-Ohno remainder's R^-3 tail that has no thermodynamic limit, so
    the default form is refused here and the Elstner form is accepted (see
    test_gfn2_seccm_3d_ewald_gamma_runs_with_the_elstner_kernel).
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    translations = np.diag([8.0, 20.0, 20.0])
    primitive = np.diag([8.0 / replicas_x, 20.0, 20.0])
    topology = _topology(
        coords, translations, replicas=(replicas_x, 1, 1),
        primitive_vectors=primitive,
    )
    molecule = Molecule([Atom(1, xyz.tolist()) for xyz in coords], 0, 1)
    run = _run_native_gfn2_seccm_from_records if native else run_gfn2_seccm
    with pytest.raises(ValueError) as caught:
        run(molecule, topology, ewald_gamma=True, max_iter=1, **options)
    assert str(caught.value) == _GFN2_3D_GAMMA_ERROR


def test_gfn2_seccm_3d_ewald_gamma_preserves_molecular_delegation():
    """An exact zero-image molecular topology still delegates in 3-D."""
    from vibeqc.semiempirical import run_gfn2_seccm

    translations = 40.0 * np.eye(3)
    topology = _topology(
        _H2O_COORDS, translations, replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    for run in (run_gfn2_seccm, _run_native_gfn2_seccm_from_records):
        plain = run(molecule, topology)
        requested = run(molecule, topology, ewald_gamma=True)
        assert requested.molecular_delegated
        assert requested.energy == plain.energy
        np.testing.assert_array_equal(requested.hamiltonian, plain.hamiltonian)
        np.testing.assert_array_equal(requested.shell_charges, plain.shell_charges)
        # Disabling AES selects the cyclic engine, not molecular delegation.
        with pytest.raises(ValueError) as caught:
            run(molecule, topology, ewald_gamma=True, include_aes=False)
        assert str(caught.value) == _GFN2_3D_GAMMA_ERROR


def test_gfn2_seccm_ewald_gamma_kernel_matches_tblite_reference():
    """Internal finite-cluster bookkeeping, not 3-D thermodynamic validity.

    The Python reference reproduces the diagnostic matrix, including its
    divergent WS remainder; production use is separately rejected (#444).
    """
    import math

    molecule, atoms, translations = _rocksalt_mgo_2x2x2(4.212)
    topology = _rocksalt_topology(atoms, translations)

    atoms_bohr = np.array([np.asarray(a.xyz) for a in molecule.atoms])
    shells = _gfn2_shell_context(molecule)

    T = np.asarray(topology.translations)
    vol = abs(np.linalg.det(T))
    alpha = math.sqrt(math.pi) / vol ** (1.0 / 3.0)
    gamma_ref = _ewald_shell_gamma_reference(shells, atoms_bohr, topology, alpha)
    gamma = _native_gfn2_ewald_shell_gamma(molecule, topology, alpha)
    np.testing.assert_allclose(gamma, gamma_ref, rtol=0, atol=1.0e-6)


def _mgo_rocksalt_cluster(n: int, a_angstrom: float = 4.212):
    """n x n x n rocksalt MgO cyclic cluster and its bound SECCM topology."""
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    coords: list[np.ndarray] = []
    zs: list[int] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for site in range(2):
                    coords.append(
                        i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                    )
                    zs.append(12 if site == 0 else 8)
    translations = [n * p for p in prim]
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * _BOHR).tolist()) for z, c in zip(zs, coords)],
        0,
        1,
    )
    topology = build_seccm_topology(
        [np.asarray(c) * _BOHR for c in coords],
        [np.asarray(t) * _BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * _BOHR for p in prim],
        replicas=(n, n, n),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def _ionic_shell_charges(molecule: Molecule, n_shells: int) -> np.ndarray:
    """+1 e on every cation, -1 e on every anion, split evenly over shells."""
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    params = load_gfn2_params()
    dq = np.zeros(n_shells)
    index = 0
    for atom in molecule.atoms:
        count = len(params.element_data(atom.Z).shells)
        dq[index : index + count] = (1.0 if atom.Z == 12 else -1.0) / count
        index += count
    assert index == n_shells
    return dq


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_seccm_shell_gamma_elements_do_not_depend_on_the_ewald_width(
    dimension: int,
):
    """Every matrix element is alpha-independent, not only neutral contractions.

    The Ewald splitting parameter is arbitrary, so a well-defined lattice
    potential cannot depend on it.  Until 2026-09-06 the 3-D SECCM kernel
    dropped the neutralising G = 0 background, and its elements moved with
    alpha (measured on the acute H2O cell: gamma[0, 0] ran 0.6347 -> 0.5545
    as alpha tripled).  That was invisible in every shipped number, because
    the omission shifts all elements by one constant and so cancels in the
    neutral contraction the SCC consumes -- which is exactly why it needs
    its own test rather than an energy test.
    """
    import math

    molecule = _h2o_molecule(_H2O_COORDS)
    topology = _acute_skew_h2o_topology(dimension, 10.0, reduced=False)
    translations = np.asarray(topology.translations)
    if dimension == 3:
        base = math.sqrt(math.pi) / abs(
            float(np.linalg.det(translations))
        ) ** (1.0 / 3.0)
    elif dimension == 2:
        area = float(np.linalg.norm(np.cross(*translations)))
        base = 0.85 * math.sqrt(math.pi) / math.sqrt(area)
    else:
        base = 4.0 / float(np.linalg.norm(translations[0]))

    reference = _native_gfn2_ewald_shell_gamma(molecule, topology, base)
    for scale in (0.7, 1.4, 2.0):
        widened = _native_gfn2_ewald_shell_gamma(
            molecule, topology, base * scale
        )
        np.testing.assert_allclose(
            widened, reference, rtol=0.0, atol=1.0e-9,
            err_msg=f"alpha scale {scale} moved the kernel",
        )


def test_gfn2_seccm_3d_ewald_gamma_runs_with_the_elstner_kernel():
    """#444 closure: the 3-D route exists, with the kernel that has a limit.

    The refusal this replaces was about the Klopman-Ohno remainder's R^-3
    tail, not about three dimensions, so selecting a remainder that decays
    exponentially opens the route rather than bypassing the gate.
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _mgo_rocksalt_cluster(2)
    result = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        gamma_form="elstner",
        max_iter=800,
        electronic_temperature=0.005,
        scc_mixer="broyden_eyert",
    )
    assert result.converged
    assert not result.molecular_delegated
    assert result.energy == pytest.approx(-4.764409516, abs=1.0e-6)
    # The same cell with the default kernel is still refused.
    with pytest.raises(ValueError) as caught:
        run_gfn2_seccm(molecule, topology, ewald_gamma=True, max_iter=1)
    assert str(caught.value) == _GFN2_3D_GAMMA_ERROR
    # And the form is only meaningful as the remainder of an Ewald split.
    with pytest.raises(ValueError, match="requires ewald_gamma=True"):
        run_gfn2_seccm(molecule, topology, gamma_form="elstner", max_iter=1)


def test_seccm_elstner_gamma_converges_faster_in_cluster_size():
    """Why the Elstner form is the one 3-D cells may use (#444).

    Both remainders are summed over the Wigner-Seitz inventory, which grows
    with the cluster.  The Klopman-Ohno remainder falls off as -eta^2/2R^3
    with a pair-dependent amplitude, so its three-dimensional sum converges
    only conditionally; the Elstner remainder is the exponentially decaying
    -S(tau_a, tau_b, R) and converges absolutely.

    Measured here on rocksalt MgO at a fixed +-1 e ionic charge pattern, as
    the electrostatic energy per formula unit (Ha):

        cells      Klopman-Ohno        Elstner
            8       0.172153245    0.128355673
           27       0.137854440    0.116311819
           64       0.141032406    0.116579996
          125       0.140160876    0.116535481

    Both settle on this cell -- the finite WS truncation is what keeps the
    Klopman-Ohno sum finite -- but the Elstner step from 64 to 125 cells is
    4.4e-5 Ha against 8.7e-4, twenty times tighter.  The claim tested here
    is that ordering, not a divergence: the divergence of the Klopman-Ohno
    remainder is a statement about the untruncated sum, pinned on the
    periodic driver by
    tests/test_periodic_gfn2_aes.py::test_elstner_gamma_converges_in_the_cutoff_and_klopman_ohno_does_not.
    """
    import math

    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
    from vibeqc.semiempirical.seccm._adapter_common import (
        flatten_topology_records,
    )

    params = load_gfn2_params()
    steps: dict[str, list[float]] = {"klopman_ohno": [], "elstner": []}
    for n in (2, 3, 4):
        molecule, topology = _mgo_rocksalt_cluster(n)
        records = flatten_topology_records(
            molecule, topology, route_name="GFN2-SECCM cluster-limit test"
        )
        translations = np.asarray(topology.translations)
        alpha = math.sqrt(math.pi) / abs(
            float(np.linalg.det(translations))
        ) ** (1.0 / 3.0)
        for name, form in (
            ("klopman_ohno", _se.ShellGammaForm.KlopmanOhno),
            ("elstner", _se.ShellGammaForm.Elstner),
        ):
            gamma = np.asarray(
                _se._gfn2_seccm_ewald_shell_gamma_from_records(
                    molecule,
                    params,
                    records[0],
                    records[1],
                    records[2],
                    np.asarray(records[3], dtype=np.int32),
                    records[4],
                    records[5],
                    np.asarray(records[6], dtype=float),
                    float(alpha),
                    False,
                    False,
                    form,
                )
            )
            dq = _ionic_shell_charges(molecule, gamma.shape[0])
            cells = len(molecule.atoms) // 2
            steps[name].append(0.5 * float(dq @ gamma @ dq) / cells)

    np.testing.assert_allclose(
        steps["klopman_ohno"],
        [0.172153245, 0.137854440, 0.141032406],
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        steps["elstner"],
        [0.128355673, 0.116311819, 0.116579996],
        rtol=0.0,
        atol=1.0e-8,
    )
    klopman_ohno_step = abs(steps["klopman_ohno"][2] - steps["klopman_ohno"][1])
    elstner_step = abs(steps["elstner"][2] - steps["elstner"][1])
    assert elstner_step < klopman_ohno_step / 5.0


def _polar_hf_chain(shift_index: int | None = None, nrep: int = 4, a: float = 6.0):
    """1-D chain of HF units: polar, so the AES channel carries real moments.

    `shift_index` retypes one atom one lattice vector over -- the same
    crystal, a different coordinate representative.
    """
    coords: list[np.ndarray] = []
    zs: list[int] = []
    for i in range(nrep):
        coords.append(np.array([i * a, 0.0, 0.0]))
        zs.append(9)
        coords.append(np.array([i * a + 1.75, 0.35, 0.0]))
        zs.append(1)
    translations = [np.array([nrep * a, 0.0, 0.0])]
    if shift_index is not None:
        coords[shift_index] = coords[shift_index] + translations[0]
    molecule = Molecule(
        [Atom(z, c.tolist()) for z, c in zip(zs, coords)], 0, 1
    )
    topology = build_seccm_topology(
        coords, translations, length_unit="bohr", geometry_quantum=1.0e-10
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.array([a, 0.0, 0.0])],
        replicas=(nrep, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def test_gfn2_seccm_faithful_aes_is_invariant_under_lattice_relabelling():
    """Issue #348: the AES was the last representative-dependent channel.

    Every other term of the route reads the Wigner-Seitz record inventory
    and so cannot see which lattice copy of an atom the caller typed: the
    overlap and H0 image blocks by construction, the coordination numbers
    since #354, the shell gamma since #444/M4a. The AES did not -- its
    multipole integrals were built from the home-cell molecule -- so the
    moments, and with them the energy, moved when a boundary site was
    retyped.

    Measured here on a 4-unit polar HF chain, three representatives of one
    crystal (Ha):

        channel     as typed        F(0) + T        H(0) + T
        ad-hoc      -5.230514880891 -5.230364176724 -5.230359290006
        faithful    -5.233747472334 -5.233747472334 -5.233747472334
        AES off     -5.231529541282 -5.231529541282 -5.231529541282

    The ad-hoc spread is 1.6e-4 Ha; with the channel switched off the three
    agree bit-for-bit, which is what isolates the defect to the AES rather
    than to any other term.

    The two paths cannot be blended, which is why this is a model flag and
    not a repair: the ad-hoc kernel consumes shell-resolved moments, and a
    pair inventory produces atom-resolved CAMM.
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    representatives = (None, 0, 1)
    faithful = []
    ad_hoc = []
    channel_off = []
    for shift_index in representatives:
        molecule, topology = _polar_hf_chain(shift_index)
        common = dict(max_iter=2000, electronic_temperature=0.0)
        faithful.append(
            run_gfn2_seccm(molecule, topology, aes_faithful=True, **common)
        )
        ad_hoc.append(run_gfn2_seccm(molecule, topology, **common))
        channel_off.append(
            run_gfn2_seccm(molecule, topology, include_aes=False, **common)
        )
    assert all(r.converged for r in faithful + ad_hoc + channel_off)

    # The fix: one energy, and one AES energy, for one crystal.
    for result in faithful[1:]:
        assert result.energy == pytest.approx(
            faithful[0].energy, abs=1.0e-12, rel=0.0
        )
        assert result.e_aes == pytest.approx(
            faithful[0].e_aes, abs=1.0e-12, rel=0.0
        )
    assert faithful[0].energy == pytest.approx(-5.233747472334, abs=1.0e-9)
    assert abs(faithful[0].e_aes) > 1.0e-4, "the AES channel must be live here"

    # The defect it replaces, pinned so a silent revert is visible.
    ad_hoc_energies = [r.energy for r in ad_hoc]
    assert max(ad_hoc_energies) - min(ad_hoc_energies) > 1.0e-4

    # And the isolation: with the channel off every term is already invariant.
    for result in channel_off[1:]:
        assert result.energy == pytest.approx(
            channel_off[0].energy, abs=1.0e-12, rel=0.0
        )


def test_gfn2_seccm_joint_state_fixed_point_is_mixer_independent():
    """Issue #409: the moments are part of the iterate, not a lag on it.

    The shipped path mixes shell charges and rebuilds the multipole channel
    from whatever density comes out, so the moments are never an iterate and
    xtb's trajectory -- which advances charges and moments together -- is not
    reachable in principle. On the faithful path the reduced state is
    ``[dq_shell; mu; theta]`` and one mixer advances all of it.

    The property that shows this is a genuine fixed-point problem rather
    than a fixed point plus a lag: every mixer must land on the *same*
    state. Measured on the polar HF chain, the faithful energies agree to
    twelve digits across simple (129 iterations), Broyden (25), DIIS (23)
    and the Eyert scheme (9), while the shipped path's mixers agree only to
    about 1e-8 because its convergence test never looks at the moments.
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    mixers = ("simple", "broyden_eyert", "broyden", "diis")
    faithful = {}
    ad_hoc = {}
    for mixer in mixers:
        molecule, topology = _polar_hf_chain()
        common = dict(
            max_iter=3000, electronic_temperature=0.0, scc_mixer=mixer
        )
        faithful[mixer] = run_gfn2_seccm(
            molecule, topology, aes_faithful=True, **common
        )
        ad_hoc[mixer] = run_gfn2_seccm(molecule, topology, **common)
    assert all(r.converged for r in faithful.values())
    assert all(r.converged for r in ad_hoc.values())

    reference = faithful["simple"].energy
    for mixer in mixers:
        assert faithful[mixer].energy == pytest.approx(
            reference, abs=1.0e-11, rel=0.0
        ), f"{mixer} reached a different faithful fixed point"
        assert faithful[mixer].e_aes == pytest.approx(
            faithful["simple"].e_aes, abs=1.0e-9, rel=0.0
        )
    assert reference == pytest.approx(-5.233747472334, abs=1.0e-9)

    # The joint residual is a stricter convergence test than the charge
    # residual at the same tolerance, so the shipped path's mixers scatter
    # where the joint one does not.
    ad_hoc_energies = [r.energy for r in ad_hoc.values()]
    assert max(ad_hoc_energies) - min(ad_hoc_energies) > 1.0e-9

    # The Eyert scheme is the one xtb uses, and it is the fastest here.
    assert faithful["broyden_eyert"].n_iter < 20
    assert faithful["simple"].n_iter > 100

    # And the relabelling invariance of M4b survives the joint mixer.
    energies = []
    for shift_index in (None, 0, 1):
        molecule, topology = _polar_hf_chain(shift_index)
        energies.append(
            run_gfn2_seccm(
                molecule,
                topology,
                aes_faithful=True,
                scc_mixer="broyden_eyert",
                max_iter=3000,
                electronic_temperature=0.0,
            ).energy
        )
    assert max(energies) - min(energies) < 1.0e-12


def test_gfn2_seccm_newton_mixer_is_refused_on_the_joint_state():
    """The Newton step's Jacobian lives on the charge subspace alone."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _polar_hf_chain()
    with pytest.raises(ValueError, match=r"newton.*aes_faithful|#409"):
        run_gfn2_seccm(
            molecule,
            topology,
            aes_faithful=True,
            scc_mixer="newton",
            max_iter=10,
        )


@pytest.mark.parametrize("aes_faithful", [False, True])
def test_gfn2_seccm_molecular_limit_delegates_with_the_requested_aes_model(
    aes_faithful: bool,
):
    """A single-replica torus is the molecule, under either AES model.

    The AES model is a choice about the Hamiltonian, not about the
    boundary, so the delegated molecular solve has to make the same one --
    otherwise asking for the Bannwarth model on a zero-image topology would
    silently return the ad-hoc energy.
    """
    from vibeqc.semiempirical import run_gfn2_seccm
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    molecule = _h2o_molecule(_H2O_COORDS)
    translations = 40.0 * np.eye(3)
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    options = _se.xtb.XTBSccOptions()
    options.aes_faithful = aes_faithful
    molecular = _se.xtb.run_gfn2_xtb(molecule, load_gfn2_params(), options)
    seccm = run_gfn2_seccm(molecule, topology, aes_faithful=aes_faithful)

    assert seccm.molecular_delegated
    assert seccm.energy == molecular.energy


def _lih_chain_topology(nrep: int = 4, a: float = 5.6):
    """Polar 1-D LiH chain with a bound cyclic group."""
    coords: list[np.ndarray] = []
    zs: list[int] = []
    for i in range(nrep):
        coords.append(np.array([i * a, 0.0, 0.0]))
        zs.append(1)
        coords.append(np.array([i * a + 2.0, 0.4, 0.0]))
        zs.append(3)
    translations = [np.array([nrep * a, 0.0, 0.0])]
    molecule = Molecule(
        [Atom(z, c.tolist()) for z, c in zip(zs, coords)], 0, 1
    )
    topology = build_seccm_topology(
        coords, translations, length_unit="bohr", geometry_quantum=1.0e-10
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.array([a, 0.0, 0.0])],
        replicas=(nrep, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def _lih_rocksalt_topology(n: int = 3, a: float = 7.6):
    """n x n x n rocksalt LiH cluster; n must be odd for the 3-D route."""
    prim = [
        np.array([0.0, a / 2, a / 2]),
        np.array([a / 2, 0.0, a / 2]),
        np.array([a / 2, a / 2, 0.0]),
    ]
    basis = [np.zeros(3), np.array([a / 2, 0.0, 0.0])]
    coords: list[np.ndarray] = []
    zs: list[int] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for site in range(2):
                    coords.append(
                        i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                    )
                    zs.append(3 if site == 0 else 1)
    translations = [n * p for p in prim]
    molecule = Molecule(
        [Atom(z, c.tolist()) for z, c in zip(zs, coords)], 0, 1
    )
    topology = build_seccm_topology(
        coords, translations, length_unit="bohr", geometry_quantum=1.0e-10
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) for p in prim],
        replicas=(n, n, n),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def test_scc_dftb_seccm_klopman_ohno_is_the_unchanged_default():
    """The staged flag must not move any shipped SCC-DFTB-SECCM number.

    `gamma_form` defaults to Klopman-Ohno, and the unembedded default route
    must therefore reproduce the untailed WS sum exactly. Elstner is a real
    change to the same route, so the two must differ -- a test that only
    checked the default would pass against a flag that does nothing.
    """
    from vibeqc.semiempirical import run_scc_dftb_seccm

    molecule, topology = _lih_chain_topology()
    default = run_scc_dftb_seccm(molecule, topology, max_iter=800)
    explicit = run_scc_dftb_seccm(
        molecule, topology, gamma_form="klopman_ohno", max_iter=800
    )
    elstner = run_scc_dftb_seccm(
        molecule, topology, gamma_form="elstner", max_iter=800
    )
    assert default.converged and explicit.converged and elstner.converged
    assert default.energy == explicit.energy
    assert default.e_scc == explicit.e_scc
    assert abs(elstner.energy - default.energy) > 1.0e-4


def test_scc_dftb_seccm_ewald_gamma_carries_the_coulomb_tail():
    """The embedded route sums the tail into the kernel, not beside it."""
    from vibeqc.semiempirical import run_scc_dftb_seccm

    molecule, topology = _lih_chain_topology()
    untailed = run_scc_dftb_seccm(
        molecule, topology, gamma_form="elstner", max_iter=800
    )
    embedded = run_scc_dftb_seccm(
        molecule,
        topology,
        gamma_form="elstner",
        ewald_gamma=True,
        max_iter=800,
    )
    assert untailed.converged and embedded.converged
    # The Ewald tail is a real contribution, not a relabelling.
    assert abs(embedded.e_scc - untailed.e_scc) > 1.0e-4
    # And it is the same channel as `madelung`, so asking for both is refused
    # rather than double-counted.
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_scc_dftb_seccm(
            molecule, topology, madelung=True, ewald_gamma=True, max_iter=1
        )


def test_scc_dftb_seccm_3d_embedding_follows_the_kernel_not_the_boundary():
    """#211: the refusal was about the Ohno tail, so Elstner opens the route.

    The 3-D Madelung embedding used to fail closed unconditionally, because
    the Wigner-Seitz truncation of the Klopman-Ohno remainder has no
    thermodynamic limit -- an `R^-3` tail with a pair-dependent amplitude.
    That is a property of the kernel, not of three dimensions, so the gate
    now names the kernel and the Elstner form passes it.
    """
    from vibeqc.semiempirical import run_scc_dftb_seccm

    molecule, topology = _lih_rocksalt_topology(3)
    with pytest.raises(NotImplementedError, match=r"Klopman-Ohno gamma.*#211"):
        run_scc_dftb_seccm(molecule, topology, madelung=True, max_iter=1)

    embedded = run_scc_dftb_seccm(
        molecule,
        topology,
        madelung=True,
        gamma_form="elstner",
        max_iter=1500,
        electronic_temperature=0.005,
    )
    assert embedded.converged
    assert embedded.energy == pytest.approx(-0.400383374, abs=1.0e-6)
    # The embedding actually contributed; a zero here would mean the gate was
    # opened onto a route that does nothing.
    assert abs(embedded.e_madelung) > 1.0e-3


def test_gfn2_seccm_ewald_gamma_eta_zero_anchor():
    """The eta-to-zero off-diagonal limit of the ewald_gamma construction
    is the bare-Coulomb Ewald kernel (the validated Madelung machinery
    convention). At eta = 0 the WS-folded KO-minus-Coulomb correction
    vanishes identically and only the Ewald dir + rec + self terms
    survive."""
    import math

    molecule, atoms, translations = _rocksalt_mgo_2x2x2(4.212)
    topology = _rocksalt_topology(atoms, translations)
    atoms_bohr = np.array([np.asarray(a.xyz) for a in molecule.atoms])
    shells = _gfn2_shell_context(molecule)
    T = np.asarray(topology.translations)
    vol = abs(np.linalg.det(T))
    alpha = math.sqrt(math.pi) / vol ** (1.0 / 3.0)

    # eta -> 0: scale the hardnesses up by a large factor.
    gamma_small = _ewald_shell_gamma_reference(
        shells, atoms_bohr, topology, alpha, eta_scale=1.0e5
    )
    n = len(shells)
    # The anchor is for pairs on different atoms: same-atom blocks carry
    # the molecular on-site 1/eta, which diverges at eta -> 0 by design.
    off = np.array(
        [
            [int(shells[a].atom_idx) != int(shells[b].atom_idx) for b in range(n)]
            for a in range(n)
        ],
        dtype=bool,
    )
    assert np.abs(gamma_small[off]).max() < 1.0
    # A second independent bare-Ewald evaluation: erfc real-space plus the
    # reciprocal series plus the self term, no on-site, no correction.
    gamma_coul = np.zeros((n, n))
    rec = np.linalg.inv(T).T * 2.0 * math.pi
    g_cut = alpha * math.sqrt(120.0)
    gmax = [
        int(math.ceil(g_cut * np.linalg.norm(T[i]) / (2.0 * math.pi))) + 1
        for i in range(3)
    ]
    r_cut = math.sqrt(30.0) / alpha
    pair_fractional_span = np.ptp(atoms_bohr @ np.linalg.inv(T), axis=0)
    nmax = [
        int(
            math.ceil(
                r_cut * np.linalg.norm(rec[i]) / (2.0 * math.pi)
                + pair_fractional_span[i]
            )
        )
        + 1
        for i in range(3)
    ]
    for a in range(n):
        atom_a = int(shells[a].atom_idx)
        for b in range(a, n):
            atom_b = int(shells[b].atom_idx)
            d = atoms_bohr[atom_b] - atoms_bohr[atom_a]
            # Neutralising background (the G = 0 reciprocal term); see
            # _ewald_shell_gamma_reference_3d.
            value = -math.pi / (vol * alpha * alpha)
            for i in range(-nmax[0], nmax[0] + 1):
                for j in range(-nmax[1], nmax[1] + 1):
                    for k in range(-nmax[2], nmax[2] + 1):
                        shift = i * T[0] + j * T[1] + k * T[2]
                        r = float(np.linalg.norm(d + shift))
                        if r < 1.0e-12 or r > r_cut:
                            continue
                        value += math.erfc(alpha * r) / r
            for i in range(-gmax[0], gmax[0] + 1):
                for j in range(-gmax[1], gmax[1] + 1):
                    for k in range(-gmax[2], gmax[2] + 1):
                        if i == j == k == 0:
                            continue
                        g = i * rec[0] + j * rec[1] + k * rec[2]
                        g2 = float(g @ g)
                        if g2 > g_cut * g_cut:
                            continue
                        value += (
                            4.0
                            * math.pi
                            / vol
                            * math.exp(-g2 / (4.0 * alpha * alpha))
                            / g2
                            * math.cos(float(g @ d))
                        )
            if atom_a == atom_b:
                value -= 2.0 * alpha / math.sqrt(math.pi)
            gamma_coul[a, b] = value
            gamma_coul[b, a] = value
    assert gamma_small[off] == pytest.approx(gamma_coul[off], abs=1.0e-6)


def test_gfn2_seccm_ewald_gamma_mgo_xtb_scale_charges():
    """#444: this named historical gate now pins explicit 3-D rejection.

    All energy/size/state assertions below are retained as historical
    evidence, not reachable production expectations. Original rationale:

    The self-consistent periodic gamma keeps the MgO 2x2x2 SCC fixed
    point at the xtb scale: converged, physical, Mulliken q_rms near
    xtb's 0.39-0.41, and the per-primitive-cell energy pinned."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _rocksalt_mgo_2x2x2(4.212)
    topology = _rocksalt_topology(atoms, translations)
    with pytest.raises(ValueError, match="3-D ewald_gamma.*no thermodynamic limit"):
        run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    # assert result.converged and result.physical_basin
    # q_rms = float(np.sqrt((np.asarray(result.charges) ** 2).mean()))
    # assert 0.2 < q_rms < 0.8
    # 2026-08-28 (issue #433): re-pinned after the full Pauling EN table
    # (Z <= 86, tblite parity) reached the Mg-O pairs; the off-site EN
    # factor (1 + 0.02*dEN^2) now uses Mg 1.31 instead of the old Z <= 10
    # fallback 1.0.  2026-09-02 (issue #354): re-pinned again with the
    # cyclic H0 coordination numbers (-4.749699725 -> -4.769076955); the
    # charge scale and the Madelung invariants below are unchanged.
    # assert result.energy == pytest.approx(-4.769076955, abs=2.0e-3)
    # assert result.e_madelung == 0.0
    # assert result.cyclic_madelung_energy == 0.0


def _aligned_plane_stress_slab(layers: int) -> tuple[Molecule, object]:
    """Synthetic aligned Mg/O plane stack used only as an SCC-map stressor.

    Its periodic extension is a two-coordinate P4/mmm lattice, neither B1
    rocksalt nor B2/CsCl. Physical B1(100) geometry is tested separately.
    """
    a = 4.212
    plane_z = [k * a / 2.0 for k in range(layers)]
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(layers):
        species = 12 if k % 2 == 0 else 8
        for i in range(2):
            for j in range(2):
                atoms.append(np.array([i * a, j * a, plane_z[k]]))
                zs.append(species)
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * _BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    translations = [
        np.array([2.0 * a, 0.0, 0.0]),
        np.array([0.0, 2.0 * a, 0.0]),
    ]
    prim = [
        np.array([a, 0.0, 0.0]),
        np.array([0.0, a, 0.0]),
    ]
    topology = _topology(
        [np.asarray(c) * _BOHR for c in atoms],
        [np.asarray(t) * _BOHR for t in translations],
        replicas=(2, 2, 1),
        primitive_vectors=[np.asarray(p) * _BOHR for p in prim],
    )
    return molecule, topology


def test_gfn2_seccm_iteration_limit_reports_unphysical_exposed_state():
    """An exhausted SCC attempt must classify the iterate it exposes (#410)."""
    from vibeqc.semiempirical import (
        GFN2SECCMConvergenceError,
        run_gfn2_seccm,
    )

    molecule, topology = _aligned_plane_stress_slab(4)
    n_shells = len(_gfn2_shell_context(molecule))
    restart = np.where(np.arange(n_shells) % 2 == 0, 5.0, -5.0)

    with pytest.raises(GFN2SECCMConvergenceError) as caught:
        run_gfn2_seccm(
            molecule,
            topology,
            ewald_gamma=True,
            max_iter=1,
            initial_shell_charges=restart,
        )

    failed = caught.value.result
    assert not failed.converged
    assert failed.selected_attempt_index is None
    assert failed.attempts[-1].exit_reason == "iteration_limit"
    assert not _se._gfn2_seccm_local_charge_state_is_physical(
        failed.charges, failed.shell_charges
    )
    assert not failed.physical_basin
    assert not failed.attempts[-1].physical_basin


def test_gfn2_seccm_ewald_gamma_options_and_molecular_delegation():
    """Conflicting kernels fail, while trivial records delegate exactly."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _rocksalt_mgo_2x2x2(4.212)
    topology = _rocksalt_topology(atoms, translations)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_gfn2_seccm(molecule, topology, madelung=True, ewald_gamma=True)

    translations_1d = [np.array([40.0, 0.0, 0.0])]
    topology_1d = _topology(
        _H2O_COORDS,
        translations_1d,
        replicas=(1, 1, 1),
        primitive_vectors=translations_1d,
    )
    molecule_1d = _h2o_molecule(_H2O_COORDS)
    exact = run_gfn2_seccm(molecule_1d, topology_1d)
    requested = run_gfn2_seccm(molecule_1d, topology_1d, ewald_gamma=True)
    assert requested.energy == pytest.approx(exact.energy, abs=0.0)
    np.testing.assert_array_equal(requested.hamiltonian, exact.hamiltonian)
    for field in (
        "e_electronic",
        "e_repulsive",
        "e_scc",
        "e_band0",
        "e_aes",
        "e_3rd",
        "cyclic_electronic_energy",
        "cyclic_repulsive_energy",
        "cyclic_scc_energy",
        "cyclic_aes_energy",
        "cyclic_3rd_energy",
        "cyclic_band0_energy",
    ):
        assert getattr(requested, field) == pytest.approx(
            getattr(exact, field), abs=0.0
        )
    np.testing.assert_array_equal(requested.shell_charges, exact.shell_charges)
    np.testing.assert_array_equal(requested.density, exact.density)
    assert requested.molecular_delegated
    assert requested.route_plan.electrostatics_kernel == "none"
    assert exact.hamiltonian_identity.requested_electrostatics_family == "none"
    assert requested.hamiltonian_identity.requested_electrostatics_family == (
        "ewald_gamma"
    )
    assert requested.hamiltonian_identity.resolved_electrostatics_kernel == (
        "none"
    )
    assert requested.hamiltonian_identity != exact.hamiltonian_identity

    # A genuinely nontrivial 1-D torus runs through the wire
    # shell-gamma construction (2026-08-27 landing) and records the
    # resolved kernel in the provenance.
    primitive = np.array([8.0, 0.0, 0.0])
    coords = np.vstack([_H2O_COORDS, _H2O_COORDS + primitive])
    nontrivial = _topology(
        coords,
        [2.0 * primitive],
        replicas=(2, 1, 1),
        primitive_vectors=[primitive],
    )
    wire = run_gfn2_seccm(
        _h2o_grid_molecule(coords), nontrivial, ewald_gamma=True
    )
    assert wire.converged and wire.physical_basin
    assert not wire.molecular_delegated
    assert wire.energy == pytest.approx(-5.031788169, abs=1.0e-6)
    assert wire.route_plan.electrostatics_kernel == "ewald_gamma_wire_1d"
    assert wire.hamiltonian_identity.resolved_electrostatics_kernel == (
        "ewald_gamma_wire_1d"
    )


@pytest.mark.parametrize(
    ("option", "parent"),
    [
        ("madelung_s_weighted", "madelung"),
        ("madelung_no_self", "madelung"),
        ("ewald_gamma_molecular_onsite", "ewald_gamma"),
    ],
)
def test_gfn2_seccm_dependent_options_require_parent_in_python_and_native(
    option, parent
):
    from vibeqc.semiempirical import run_gfn2_seccm
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
    from vibeqc.semiempirical.seccm._adapter_common import (
        flatten_topology_records,
        topology_length_unit_scale,
    )

    translation = np.array([40.0, 0.0, 0.0])
    topology = _topology(
        _H2O_COORDS,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    message = rf"{option}.*require.*{parent}"
    with pytest.raises(ValueError, match=message):
        run_gfn2_seccm(molecule, topology, **{option: True})

    records = flatten_topology_records(
        molecule, topology, route_name="GFN2-SECCM"
    )
    (
        translations,
        central,
        origin,
        labels,
        weights,
        multiplicities,
        disps,
        primitive_vectors,
        replicas,
    ) = records
    group = topology.finite_group
    assert group is not None
    with pytest.raises(ValueError, match=message):
        _se._run_gfn2_seccm_from_records(
            molecule,
            load_gfn2_params(),
            translations,
            central,
            origin,
            np.asarray(labels, dtype=np.int32),
            weights,
            multiplicities,
            np.asarray(disps, dtype=float),
            primitive_vectors,
            replicas,
            float(group.geometry_tolerance)
            * topology_length_unit_scale(topology),
            **{option: True},
        )


def test_gfn2_seccm_ewald_gamma_2d_kernel_matches_reference():
    """The 2-D ewald_gamma kernel is the Parry/Heyes construction: a
    Python reimplementation reproduces the engine's cyclic_scc_energy at
    the converged shell charges of the synthetic two-layer stress cell."""
    import math

    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    result = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    assert result.converged and result.physical_basin

    atoms_bohr = np.array([np.asarray(a.xyz) for a in molecule.atoms])
    shells = _gfn2_shell_context(molecule)
    assert len(shells) == result.shell_charges.size

    T = np.asarray(topology.translations)
    area = float(np.linalg.norm(np.cross(T[0], T[1])))
    alpha = 0.85 * math.sqrt(math.pi) / math.sqrt(area)
    gamma_ref = _ewald_shell_gamma_reference_2d(shells, atoms_bohr, topology, alpha)
    dq = np.asarray(result.shell_charges)
    e_ref = 0.5 * float(dq @ gamma_ref @ dq)
    assert result.cyclic_scc_energy == pytest.approx(e_ref, abs=1.0e-6)


def test_gfn2_seccm_ewald_gamma_2d_eta_zero_anchor():
    """The eta-to-zero off-diagonal limit of the 2-D kernel is the
    bare-Coulomb Parry/Heyes Ewald: at eta = 0 the WS correction
    vanishes and only the erfc real-space sum, the reciprocal series,
    the K=0 term, and the self term survive."""
    import math

    molecule, topology = _aligned_plane_stress_slab(2)
    atoms_bohr = np.array([np.asarray(a.xyz) for a in molecule.atoms])
    shells = _gfn2_shell_context(molecule)
    T = np.asarray(topology.translations)
    area = float(np.linalg.norm(np.cross(T[0], T[1])))
    alpha = 0.85 * math.sqrt(math.pi) / math.sqrt(area)

    gamma_small = _ewald_shell_gamma_reference_2d(
        shells, atoms_bohr, topology, alpha, eta_scale=1.0e5
    )
    gamma_coul = _ewald_shell_gamma_reference_2d(
        shells, atoms_bohr, topology, alpha, eta_scale=None
    )
    # The anchor is for pairs on different atoms: same-atom blocks carry
    # the molecular on-site 1/eta, which diverges at eta -> 0 by design.
    n = len(shells)
    off = np.array(
        [
            [int(shells[a].atom_idx) != int(shells[b].atom_idx) for b in range(n)]
            for a in range(n)
        ],
        dtype=bool,
    )
    assert np.isfinite(gamma_small[off]).all()
    assert gamma_small[off] == pytest.approx(gamma_coul[off], abs=1.0e-6)


def _acute_skew_h2o_topology(dimension: int, angle_degrees: float, *, reduced: bool):
    length = 40.0
    angle = np.deg2rad(angle_degrees)
    a1 = np.array([length, 0.0, 0.0])
    a2 = np.array([length * np.cos(angle), length * np.sin(angle), 0.0])
    translations = [a1, a2 - a1 if reduced else a2]
    if dimension == 3:
        translations.append(np.array([0.0, 0.0, length]))
    return _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("angle_degrees", [10.0, 5.0])
def test_gfn2_seccm_ewald_gamma_acute_skew_matches_complete_oracle(
    dimension: int, angle_degrees: float
):
    """Both Ewald lattices are cutoff-complete at the audited acute angles."""
    import math

    molecule = _h2o_molecule(_H2O_COORDS)
    acute = _acute_skew_h2o_topology(dimension, angle_degrees, reduced=False)
    reduced = _acute_skew_h2o_topology(dimension, angle_degrees, reduced=True)
    translations = np.asarray(acute.translations)
    if dimension == 3:
        alpha = math.sqrt(math.pi) / abs(float(np.linalg.det(translations))) ** (
            1.0 / 3.0
        )
    else:
        area = float(np.linalg.norm(np.cross(*translations)))
        alpha = 0.85 * math.sqrt(math.pi) / math.sqrt(area)

    gamma_native = _native_gfn2_ewald_shell_gamma(molecule, acute, alpha)
    gamma_reduced = _native_gfn2_ewald_shell_gamma(molecule, reduced, alpha)
    gamma_oracle = _brute_force_ewald_shell_gamma_reference(
        _gfn2_shell_context(molecule),
        np.asarray([atom.xyz for atom in molecule.atoms]),
        acute,
        alpha,
    )

    np.testing.assert_allclose(gamma_native, gamma_oracle, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(gamma_native, gamma_reduced, rtol=0.0, atol=1.0e-10)
    # Pin the quantity consumed by SCC as well as every matrix element.
    trial_charge = np.linspace(-0.75, 0.75, gamma_native.shape[0])
    trial_charge -= trial_charge.mean()
    native_energy = 0.5 * float(trial_charge @ gamma_native @ trial_charge)
    oracle_energy = 0.5 * float(trial_charge @ gamma_oracle @ trial_charge)
    reduced_energy = 0.5 * float(trial_charge @ gamma_reduced @ trial_charge)
    assert native_energy == pytest.approx(oracle_energy, abs=1.0e-10)
    assert native_energy == pytest.approx(reduced_energy, abs=1.0e-10)


@pytest.mark.parametrize("dimension", [2, 3])
def test_gfn2_seccm_ewald_gamma_is_invariant_to_atom_lattice_gauge(
    dimension: int,
):
    """A displaced pair can require a shift outside the bare cutoff sphere."""
    length = 16.0
    translations = [
        np.array([length, 0.0, 0.0]),
        np.array([0.0, length, 0.0]),
    ]
    if dimension == 3:
        translations.append(np.array([0.0, 0.0, length]))
    base_coords = np.array([[0.2, 0.3, 0.4], [1.1, 0.7, 0.5]])
    shifted_coords = base_coords.copy()
    shifted_coords[1] += 7.0 * translations[0]

    def case(coords: np.ndarray):
        molecule = Molecule([Atom(1, coord.tolist()) for coord in coords], 0, 1)
        topology = _topology(
            coords,
            translations,
            replicas=(1, 1, 1),
            primitive_vectors=translations,
        )
        return molecule, topology

    base_molecule, base_topology = case(base_coords)
    shifted_molecule, shifted_topology = case(shifted_coords)
    if dimension == 3:
        alpha = np.sqrt(np.pi) / length
    else:
        alpha = 0.85 * np.sqrt(np.pi) / length

    gamma_base = _native_gfn2_ewald_shell_gamma(base_molecule, base_topology, alpha)
    gamma_shifted = _native_gfn2_ewald_shell_gamma(
        shifted_molecule, shifted_topology, alpha
    )
    np.testing.assert_allclose(gamma_shifted, gamma_base, rtol=0.0, atol=1.0e-11)


def _strongly_skewed_equivalent_topology(molecule: Molecule, topology, shear: int = 13):
    """Unimodular a2 <- a2 + shear*a1 basis for the same finite torus."""
    translations = np.asarray(topology.translations, dtype=float).copy()
    translations[1] += shear * translations[0]
    group = topology.finite_group
    assert group is not None
    assert group.replicas[0] == group.replicas[1]
    primitive_vectors = [
        translations[axis] / group.replicas[axis]
        for axis in range(topology.dimensionality)
    ]
    coordinates = np.array([atom.xyz for atom in molecule.atoms])
    return _topology(
        coordinates,
        translations,
        replicas=group.replicas,
        primitive_vectors=primitive_vectors,
    )


def test_gfn2_seccm_ewald_gamma_2d_is_strong_skew_basis_invariant():
    """Parry real/reciprocal cutoffs include cancelling skew coefficients."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    skewed = _strongly_skewed_equivalent_topology(molecule, topology)
    reference = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    transformed = run_gfn2_seccm(molecule, skewed, ewald_gamma=True)

    assert transformed.energy == pytest.approx(reference.energy, abs=1.0e-9)
    np.testing.assert_allclose(
        transformed.shell_charges, reference.shell_charges, atol=1.0e-8
    )


def test_gfn2_seccm_ewald_gamma_3d_is_strong_skew_basis_invariant():
    """The diagnostic 3-D matrix remains cutoff-complete when skewed."""
    import math

    molecule, atoms, translations = _rocksalt_mgo_2x2x2(4.212)
    topology = _rocksalt_topology(atoms, translations)
    skewed = _strongly_skewed_equivalent_topology(molecule, topology)
    alpha = math.sqrt(math.pi) / abs(np.linalg.det(topology.translations))**(1/3)
    reference = _native_gfn2_ewald_shell_gamma(molecule, topology, alpha)
    transformed = _native_gfn2_ewald_shell_gamma(molecule, skewed, alpha)
    np.testing.assert_allclose(transformed, reference, rtol=0, atol=1.0e-9)


def test_gfn2_seccm_madelung_3d_is_strong_skew_basis_invariant():
    """The shared bare-Coulomb Madelung kernel is metric-complete in 3-D."""
    from vibeqc.semiempirical import run_gfn2_seccm

    translations = [axis * 8.0 for axis in np.eye(3)]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    skewed = _strongly_skewed_equivalent_topology(molecule, topology)

    reference = run_gfn2_seccm(molecule, topology, madelung=True)
    transformed = run_gfn2_seccm(molecule, skewed, madelung=True)

    assert reference.converged and transformed.converged
    assert reference.physical_basin and transformed.physical_basin
    assert transformed.energy == pytest.approx(reference.energy, abs=1.0e-9)
    np.testing.assert_allclose(
        transformed.charges, reference.charges, rtol=0.0, atol=1.0e-8
    )
    np.testing.assert_allclose(
        transformed.shell_charges,
        reference.shell_charges,
        rtol=0.0,
        atol=1.0e-8,
    )


def test_gfn2_seccm_ewald_gamma_2layer_stress_cell_matches_embedding():
    """The two electrostatic constructions agree on the synthetic cell."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    ewald = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    embedded = run_gfn2_seccm(molecule, topology, madelung=True)
    assert ewald.converged and ewald.physical_basin
    assert embedded.converged and embedded.physical_basin
    q_ewald = np.asarray(ewald.charges)
    q_embedded = np.asarray(embedded.charges)
    # Same physics through two independent tail constructions.
    assert ewald.energy == pytest.approx(embedded.energy, abs=2.0e-3)
    assert q_ewald == pytest.approx(q_embedded, abs=2.0e-2)
    assert float(np.sqrt((q_ewald**2).mean())) == pytest.approx(0.356, abs=0.02)
    assert 0.5 < ewald.homo_lumo_gap * 27.2114 < 2.0


def test_gfn2_seccm_ewald_gamma_4layer_stress_cell_fails_closed():
    """The measured T=0 simple-mixing trajectory does not converge.

    This establishes numerical nonconvergence on the synthetic aligned-plane
    stress map, not mathematical nonexistence of a hard-Aufbau fixed point.
    The route fails closed rather than returning the last cyclic iterate.
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(4)
    with pytest.raises(RuntimeError):
        run_gfn2_seccm(molecule, topology, ewald_gamma=True, max_iter=800)


def test_gfn2_seccm_ewald_gamma_2layer_stress_newton_matches_simple():
    """The opt-in Newton mixer lands on the same fixed point as the
    shipped simple mixing on the two-layer synthetic cell: bit-comparable
    energy, identical Mulliken charges, and ~14x fewer iterations."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    newton = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        scc_mixer="newton",
        max_iter=100,
    )
    simple = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    assert newton.converged and newton.physical_basin
    assert simple.converged and simple.physical_basin
    assert newton.energy == pytest.approx(simple.energy, abs=1.0e-8)
    np.testing.assert_allclose(
        np.asarray(newton.charges), np.asarray(simple.charges), atol=1.0e-6
    )
    assert newton.n_iter < simple.n_iter


def test_gfn2_seccm_ewald_gamma_4layer_stress_newton_converges():
    """Finite-T smoothing plus the chord/quasi-Newton step converges.

    The paired T=0 test only measures nonconvergence; this regression pins
    the observed smoothed fixed point of the synthetic aligned-plane map.
    """
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(4)
    result = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        scc_mixer="newton",
        electronic_temperature=0.002,
        max_iter=100,
    )
    assert result.converged and result.physical_basin
    # 2026-08-27 (issue #43): re-pinned after the d-shell STO-NG
    # primitive count moved from 4 to 3 (Mg carries an n=3 d shell).
    # 2026-08-28 (issue #433): re-pinned after the full Pauling EN table
    # reached the Mg-O pairs (tblite parity).
    assert result.energy == pytest.approx(-9.081743073053595, abs=1.0e-6)
    assert not result.molecular_delegated
    assert result.e_electronic + result.e_repulsive == pytest.approx(
        result.energy, abs=1.0e-12
    )
    assert (
        result.cyclic_electronic_energy + result.cyclic_repulsive_energy
    ) == pytest.approx(result.total_cyclic_energy, abs=1.0e-12)
    q = np.asarray(result.charges)
    q_rms = float(np.sqrt((q**2).mean()))
    assert 0.3 < q_rms < 0.6
    # The alternating-layer charge pattern of the repeller (per-layer
    # sums over the 2x2 in-plane cell): Mg layers positive, O layers
    # negative, magnitudes at the slab-polarized scale.
    per_layer = q.reshape(4, 4).sum(axis=1)
    assert per_layer[0] > 0.5 and per_layer[1] < -0.5
    assert per_layer[2] > 0.5 and per_layer[3] < -0.5


def test_gfn2_seccm_ewald_gamma_molecular_onsite_knob():
    """ewald_gamma_molecular_onsite=True restores the molecular on-site
    block in the periodic shell gamma (experimental validation knob for
    IID 150): the 2-layer slab still converges, the SCC energy changes
    (the on-site differs), and the shipped default (False) is unchanged.
    The knob did not cure the aligned-plane stress-map instability and stays
    only as a channel-isolation tool."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    shipped = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    hybrid = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        ewald_gamma_molecular_onsite=True,
    )
    assert hybrid.converged and hybrid.physical_basin
    assert hybrid.e_scc != pytest.approx(shipped.e_scc, abs=1.0e-12)
    assert not shipped.hamiltonian_identity.ewald_gamma_molecular_onsite
    assert hybrid.hamiltonian_identity.ewald_gamma_molecular_onsite
    assert hybrid.hamiltonian_identity != shipped.hamiltonian_identity


def test_gfn2_seccm_include_aes_validation_knob():
    """include_aes=False removes the WS-folded AES channel (experimental
    validation knob): the 2-layer slab still converges, the AES energy
    terms are exactly zero, and the per-iteration max_change trace is
    populated on the result."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    full = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    no_aes = run_gfn2_seccm(molecule, topology, ewald_gamma=True, include_aes=False)
    assert no_aes.converged and no_aes.physical_basin
    assert no_aes.e_aes == 0.0
    assert no_aes.cyclic_aes_energy == 0.0
    assert full.e_aes != 0.0
    assert full.hamiltonian_identity.include_aes
    assert not no_aes.hamiltonian_identity.include_aes
    assert no_aes.hamiltonian_identity != full.hamiltonian_identity
    assert full.scc_max_change_trace.size == full.n_iter
    assert no_aes.scc_max_change_trace.size == no_aes.n_iter


def test_parry_k0_neutral_zero_dipole_quadrupole_is_nonzero():
    """The full K=0 pair kernel is not its rank-one dipole Taylor term."""
    import math

    area = 253.416207918
    alpha = 0.094640421993
    spacing = 7.959525864
    z = np.array([-spacing / 2.0, 0.0, spacing / 2.0])
    q = np.array([1.0, -2.0, 1.0])
    dz = z[:, None] - z[None, :]
    exact_kernel = -(2.0 * math.pi / area) * (
        dz * np.vectorize(math.erf)(alpha * dz)
        + np.exp(-((alpha * dz) ** 2)) / (alpha * math.sqrt(math.pi))
    )
    exact_energy = 0.5 * float(q @ exact_kernel @ q)
    rank1_energy = 2.0 * alpha * math.sqrt(math.pi) / area * float(z @ q) ** 2

    assert float(q.sum()) == 0.0
    assert float(z @ q) == 0.0
    assert exact_energy == pytest.approx(0.005187597985, abs=1.0e-12)
    assert rank1_energy == 0.0


def test_parry_exact_pair_kernel_is_z_origin_and_alpha_invariant():
    """Exercise the complete K=0 term through the native shell-gamma kernel."""
    import math

    area = 253.416207918
    alpha0 = 0.094640421993
    spacing = 7.959525864
    side = math.sqrt(area)
    translations = [
        np.array([side, 0.0, 0.0]),
        np.array([0.0, side, 0.0]),
    ]
    z = np.array([-spacing / 2.0, 0.0, spacing / 2.0])
    trial_charge = np.array([1.0, -2.0, 1.0])

    def case(z_origin: float):
        coords = np.column_stack((np.zeros(3), np.zeros(3), z + z_origin))
        # H3+ supplies three one-shell sites with a closed-shell molecular
        # metadata state; the contracted trial charges are the Ewald oracle.
        molecule = Molecule([Atom(1, coord.tolist()) for coord in coords], 1, 1)
        topology = _topology(
            coords,
            translations,
            replicas=(1, 1, 1),
            primitive_vectors=translations,
        )
        return molecule, topology

    molecule, topology = case(0.0)
    energies = []
    for scale in (0.6, 0.8, 1.0, 1.2, 1.6):
        gamma = _native_gfn2_ewald_shell_gamma(molecule, topology, alpha0 * scale)
        energies.append(0.5 * float(trial_charge @ gamma @ trial_charge))
    assert max(energies) - min(energies) < 1.0e-10

    gamma = _native_gfn2_ewald_shell_gamma(molecule, topology, alpha0)
    compatibility = _native_gfn2_ewald_shell_gamma(
        molecule, topology, alpha0, k0_global=True
    )
    np.testing.assert_array_equal(compatibility, gamma)

    shifted_molecule, shifted_topology = case(37.25)
    shifted = _native_gfn2_ewald_shell_gamma(shifted_molecule, shifted_topology, alpha0)
    np.testing.assert_allclose(shifted, gamma, rtol=0.0, atol=1.0e-12)


def test_gfn2_seccm_k0_global_flag_keeps_exact_pair_kernel():
    """The deprecated flag is a strict compatibility no-op."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    exact = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    compatibility = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        ewald_gamma_k0_global=True,
    )
    assert compatibility.energy == pytest.approx(exact.energy, abs=0.0)
    np.testing.assert_array_equal(compatibility.shell_charges, exact.shell_charges)
    assert compatibility.hamiltonian_identity == exact.hamiltonian_identity
    assert compatibility.route_plan.route_key == exact.route_plan.route_key
    assert compatibility.run_controls != exact.run_controls
    assert compatibility.run_controls.requested_ewald_gamma_k0_global
    assert (
        compatibility.run_controls.resolved_ewald_gamma_k0_policy
        == "exact-pairwise-parry-de-leeuw"
    )


@pytest.mark.parametrize("raw_binding", [False, True], ids=["public", "raw"])
@pytest.mark.parametrize(
    ("restart_kind", "message"),
    [
        ("valid", r"initial_shell_charges.*molecular-delegation"),
        ("wrong-size", r"initial_shell_charges.*exactly n_shells"),
        ("nan", r"initial_shell_charges.*finite"),
        ("positive-infinity", r"initial_shell_charges.*finite"),
        ("negative-infinity", r"initial_shell_charges.*finite"),
    ],
    ids=["valid", "wrong-size", "nan", "positive-infinity", "negative-infinity"],
)
def test_gfn2_seccm_molecular_delegation_rejects_restart(
    raw_binding,
    restart_kind,
    message,
):
    from vibeqc.semiempirical import run_gfn2_seccm

    translation = np.array([40.0, 0.0, 0.0])
    topology = _topology(
        _H2O_COORDS,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )
    molecule = _h2o_molecule(_H2O_COORDS)
    n_shells = len(_gfn2_shell_context(molecule))
    restart = np.zeros(n_shells)
    if restart_kind == "wrong-size":
        restart = np.zeros(n_shells + 1)
    elif restart_kind == "nan":
        restart[0] = np.nan
    elif restart_kind == "positive-infinity":
        restart[0] = np.inf
    elif restart_kind == "negative-infinity":
        restart[0] = -np.inf

    with pytest.raises(ValueError, match=message):
        if raw_binding:
            _run_native_gfn2_seccm_from_records(
                molecule,
                topology,
                max_iter=1,
                initial_shell_charges=restart,
            )
        else:
            run_gfn2_seccm(
                molecule,
                topology,
                max_iter=1,
                initial_shell_charges=restart,
            )


@pytest.mark.parametrize(
    "restart",
    [
        np.array(0.0),
        np.zeros((1, 2)),
        np.zeros((2, 1)),
        np.empty((0, 0)),
    ],
    ids=["scalar", "row", "column", "two-dimensional-empty"],
)
def test_gfn2_seccm_public_restart_requires_one_dimension_before_inputs(
    restart,
):
    from vibeqc.semiempirical import run_gfn2_seccm

    with pytest.raises(
        ValueError,
        match=r"initial_shell_charges.*one-dimensional",
    ):
        run_gfn2_seccm(
            None,
            None,
            initial_shell_charges=restart,
        )


@pytest.mark.parametrize("raw_binding", [False, True], ids=["public", "raw"])
@pytest.mark.parametrize(
    "nonfinite_value",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_gfn2_seccm_nontrivial_restart_requires_finite_values(
    raw_binding,
    nonfinite_value,
):
    from vibeqc.semiempirical import run_gfn2_seccm

    coordinates = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    primitive = np.array([4.0, 0.0, 0.0])
    topology = _topology(
        coordinates,
        [2.0 * primitive],
        replicas=(2, 1, 1),
        primitive_vectors=[primitive],
    )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coordinates],
        0,
        1,
    )
    restart = np.array([nonfinite_value, 0.0])

    with pytest.raises(ValueError, match=r"initial_shell_charges.*finite"):
        if raw_binding:
            _run_native_gfn2_seccm_from_records(
                molecule,
                topology,
                max_iter=1,
                initial_shell_charges=restart,
            )
        else:
            run_gfn2_seccm(
                molecule,
                topology,
                max_iter=1,
                initial_shell_charges=restart,
            )


@pytest.mark.parametrize(
    ("restart_kind", "message"),
    [
        ("wrong-size", r"initial_shell_charges.*exactly n_shells"),
        ("nan", r"initial_shell_charges.*finite"),
        ("positive-infinity", r"initial_shell_charges.*finite"),
        ("negative-infinity", r"initial_shell_charges.*finite"),
    ],
    ids=["wrong-size", "nan", "positive-infinity", "negative-infinity"],
)
def test_gfn2_seccm_raw_restart_validation_precedes_invalid_topology(
    restart_kind,
    message,
):
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    molecule = Molecule(
        [
            Atom(1, [0.0, 0.0, -0.7]),
            Atom(1, [0.0, 0.0, 0.7]),
        ],
        0,
        1,
    )
    n_shells = len(_gfn2_shell_context(molecule))
    restart = np.zeros(n_shells)
    if restart_kind == "wrong-size":
        restart = np.zeros(n_shells + 1)
    elif restart_kind == "nan":
        restart[0] = np.nan
    elif restart_kind == "positive-infinity":
        restart[0] = np.inf
    elif restart_kind == "negative-infinity":
        restart[0] = -np.inf

    translation = np.array([[20.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=message):
        _se._run_gfn2_seccm_from_records(
            molecule,
            load_gfn2_params(),
            translation,
            [0, 1],
            [0, 1],
            np.zeros((2, 3), dtype=np.int32),
            [1.0, 1.0],
            [1, 1],
            np.zeros((2, 3)),
            np.zeros((1, 3)),
            [1, 1, 1],
            1.0e-8,
            include_aes=False,
            initial_shell_charges=restart,
        )


def test_gfn2_seccm_raw_molecular_restart_precedes_topology_svd():
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    molecule = Molecule([Atom(2, [0.0, 0.0, 0.0])], 0, 1)
    n_shells = len(_gfn2_shell_context(molecule))

    with pytest.raises(
        ValueError,
        match=r"initial_shell_charges.*molecular-delegation",
    ):
        _se._run_gfn2_seccm_from_records(
            molecule,
            load_gfn2_params(),
            np.array([[20.0, 0.0, 0.0]]),
            [],
            [],
            np.zeros((0, 3), dtype=np.int32),
            [],
            [],
            np.zeros((0, 3)),
            np.zeros((1, 3)),
            [1, 1, 1],
            1.0e-8,
            initial_shell_charges=np.zeros(n_shells),
        )


@pytest.mark.parametrize("raw_binding", [False, True], ids=["public", "raw"])
def test_gfn2_seccm_restart_preserves_missing_parameter_diagnostic(
    raw_binding,
):
    from vibeqc.semiempirical import run_gfn2_seccm

    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    molecule = Molecule(
        [
            Atom(88, coordinates[0].tolist()),
            Atom(2, coordinates[1].tolist()),
        ],
        0,
        1,
    )
    translation = np.array([20.0, 0.0, 0.0])
    topology = _topology(
        coordinates,
        [translation],
        replicas=(1, 1, 1),
        primitive_vectors=[translation],
    )

    with pytest.raises(
        ValueError,
        match=r"parameter set is missing an input element",
    ):
        if raw_binding:
            _run_native_gfn2_seccm_from_records(
                molecule,
                topology,
                initial_shell_charges=np.zeros(1),
            )
        else:
            run_gfn2_seccm(
                molecule,
                topology,
                initial_shell_charges=np.zeros(1),
            )


def test_gfn2_seccm_initial_shell_charges_restart():
    """The initial_shell_charges restart starts the SCC from a supplied
    shell-fluctuation vector instead of the neutral start. A perturbed
    start on the permitted AES-active two-layer stress slab lands on the same fixed
    point as the neutral start; a wrong-sized vector fails closed; and
    the default (None) keeps the shipped neutral start bit-for-bit."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    default = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    n_shells = len(np.asarray(default.shell_charges))
    perturbed = np.zeros(n_shells)
    perturbed[0] = 0.2
    perturbed[1] = -0.2
    restart = run_gfn2_seccm(
        molecule, topology, ewald_gamma=True, initial_shell_charges=perturbed,
    )
    assert restart.converged and restart.physical_basin
    assert restart.energy == pytest.approx(default.energy, abs=1.0e-8)
    with pytest.raises(ValueError, match="initial_shell_charges"):
        run_gfn2_seccm(
            molecule, topology, ewald_gamma=True,
            initial_shell_charges=np.zeros(n_shells + 2),
        )
    none_start = run_gfn2_seccm(
        molecule, topology, ewald_gamma=True, initial_shell_charges=None,
    )
    assert none_start.energy == pytest.approx(default.energy, abs=1.0e-12)
    assert none_start.n_iter == default.n_iter


def test_gfn2_seccm_scc_mixer_option_surface():
    """The scc_mixer option accepts the five mixer names, rejects
    unknown ones, and the default keeps the shipped simple-mixing
    behaviour (same result as an explicit 'simple')."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    default = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    explicit = run_gfn2_seccm(molecule, topology, ewald_gamma=True, scc_mixer="simple")
    assert explicit.energy == pytest.approx(default.energy, abs=1.0e-12)
    assert explicit.n_iter == default.n_iter
    with pytest.raises(ValueError, match="scc_mixer"):
        run_gfn2_seccm(molecule, topology, ewald_gamma=True, scc_mixer="anderson")
    # The tblite/xtb Eyert mixer converges the two-layer stress map to the same
    # fixed point as simple mixing (mixers select the path, not the point).
    eyert = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        scc_mixer="broyden_eyert",
        charge_mixing=0.4,
    )
    assert eyert.converged and eyert.physical_basin
    assert eyert.energy == pytest.approx(default.energy, abs=1.0e-9)
    assert eyert.attempts[0].solver == "supercell_broyden_eyert"


def test_gfn2_seccm_broyden_eyert_defaults_to_xtb_bromix():
    """An unset charge_mixing resolves per mixer: xtb's bromix default
    0.4 for broyden_eyert (tblite broyden.f90), the SECCM simple floor
    0.1 for every other mixer; an explicit value always wins. The
    resolved value is recorded in the attempt provenance."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    eyert_default = run_gfn2_seccm(
        molecule, topology, ewald_gamma=True, scc_mixer="broyden_eyert"
    )
    assert eyert_default.attempts[0].ladder_charge_mixing == pytest.approx(
        0.4
    )
    assert eyert_default.attempts[0].solver == "supercell_broyden_eyert"
    eyert_explicit = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        scc_mixer="broyden_eyert",
        charge_mixing=0.4,
    )
    assert eyert_default.energy == eyert_explicit.energy
    assert eyert_default.n_iter == eyert_explicit.n_iter
    eyert_02 = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        scc_mixer="broyden_eyert",
        charge_mixing=0.2,
    )
    assert eyert_02.attempts[0].ladder_charge_mixing == pytest.approx(0.2)
    simple_default = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    assert simple_default.attempts[0].ladder_charge_mixing == pytest.approx(
        0.1
    )


def test_gfn2_seccm_ewald_gamma_cu_simple_reaches_physical_basin():
    """#444: this named historical gate now pins explicit 3-D rejection.

    All energy/size/state assertions below are retained as historical
    evidence, not reachable production expectations. Original rationale:

    History: on the pre-#43 tree (rotated d-first parameters) this
    map's neutral simple-mixing start reached a charge-density-wave
    basin (atomic q_rms ~2.7 e) that the physical-basin gate correctly
    failed closed. With the per-angular-momentum parameter projection
    that basin is gone and plain simple mixing converges the fcc Cu
    2x2x2 ewald_gamma map inside the magnitude gates.

    With issue #354's cyclic coordination numbers the map's fixed points
    moved. Newton and broyden_eyert at this temperature agree on a
    translation-symmetric metallic state at -3.779587241 Ha (orbit
    charge spread 4.7e-4 e); plain simple mixing instead converges, in
    about 1800 iterations, into a gapped state that breaks the finite
    translation group (orbit charge spread 0.84 e, q_rms 0.36 e) and
    lies 1.0 mHa/atom above the symmetric metal. Every magnitude gate
    admits it (issue #421: the spread is reported, not gated). This
    test pins that measured state so a change in either direction -
    the gate closing on it, or the mixer path changing - is visible."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _fcc_cu_2x2x2(3.615)
    topology = _fcc_cu_topology(atoms, translations)
    with pytest.raises(ValueError, match="3-D ewald_gamma.*no thermodynamic limit"):
        run_gfn2_seccm(
            molecule,
            topology,
            ewald_gamma=True,
            electronic_temperature=0.005,
            max_iter=3000,
        )
    # assert result.converged and result.physical_basin
    # assert result.energy == pytest.approx(-3.778602023, abs=1.0e-6)
    # q_rms = float(np.sqrt((np.asarray(result.charges) ** 2).mean()))
    # assert q_rms < 0.5
    # Symmetry-broken attractor, admitted by the magnitude gates and
    # measured by the #421 diagnostic (0.8425 e on this build).
    # assert result.translation_symmetry_charge_spread == pytest.approx(
    #     0.8425, abs=5.0e-3
    # )


def test_gfn2_seccm_ewald_gamma_cu_newton_reaches_sane_basin():
    """#444: this named historical gate now pins explicit 3-D rejection.

    All energy/size/state assertions below are retained as historical
    evidence, not reachable production expectations. Original rationale:

    The Newton mixer reaches the metallic fixed point on fcc Cu
    2x2x2 in a handful of iterations at T = 0.001 Ha (IID 130 metal
    recipe). The pinned energy moved when issue #43's d-first parameter
    rotation was fixed: the pre-fix recipe pinned -3.790738 at
    T = 0.002. It moved again when issue #354 made the H0 coordination
    numbers cyclic (-3.773689041 -> -3.777337900); the fixed point is
    now translation-symmetric (orbit charge spread 9.5e-5 e, q_rms
    4e-5 e)."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _fcc_cu_2x2x2(3.615)
    topology = _fcc_cu_topology(atoms, translations)
    with pytest.raises(ValueError, match="3-D ewald_gamma.*no thermodynamic limit"):
        run_gfn2_seccm(
            molecule,
            topology,
            ewald_gamma=True,
            scc_mixer="newton",
            electronic_temperature=0.001,
            max_iter=100,
        )
    # assert result.converged and result.physical_basin
    # assert result.energy == pytest.approx(-3.777337900, abs=1.0e-6)
    # q_rms = float(np.sqrt((np.asarray(result.charges) ** 2).mean()))
    # shell_rms = float(
    #     np.sqrt((np.asarray(result.shell_charges) ** 2).mean())
    # )
    # assert q_rms < 0.5
    # assert shell_rms < 0.5
    # assert result.translation_symmetry_charge_spread < 1.0e-3


def test_gfn2_seccm_ewald_gamma_cu_eyert_reaches_sane_basin():
    """#444: this named historical gate now pins explicit 3-D rejection.

    All energy/size/state assertions below are retained as historical
    evidence, not reachable production expectations. Original rationale:

    The tblite broyden.f90 mixer (broyden_eyert) reaches the same
    fixed point as the Newton mixer on fcc Cu 2x2x2 at xtb's default
    temperature scale (T = 0.001 Ha ~ 316 K), agreeing on the pinned
    per-atom energy: a cross-mixer consistency anchor (mixers select
    the path, not the point). Re-pinned with issue #354's cyclic
    coordination numbers (-3.773689041 -> -3.777337900)."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, atoms, translations = _fcc_cu_2x2x2(3.615)
    topology = _fcc_cu_topology(atoms, translations)
    with pytest.raises(ValueError, match="3-D ewald_gamma.*no thermodynamic limit"):
        run_gfn2_seccm(
            molecule,
            topology,
            ewald_gamma=True,
            scc_mixer="broyden_eyert",
            charge_mixing=0.4,
            electronic_temperature=0.001,
            max_iter=200,
        )
    # assert result.converged and result.physical_basin
    # assert result.energy == pytest.approx(-3.777337900, abs=1.0e-6)
    # assert result.attempts[0].solver == "supercell_broyden_eyert"
    # assert result.translation_symmetry_charge_spread < 1.0e-3


def test_gfn2_seccm_ewald_gamma_cu_224_former_wall_converges():
    """#444: this named historical gate now pins explicit 3-D rejection.

    All energy/size/state assertions below are retained as historical
    evidence, not reachable production expectations. Original rationale:

    The elongated fcc Cu 2x2x4 cell was the IID 130/101 solver wall:
    on the pre-#43 tree its SCC map provably had no physical fixed
    point (every mixer x kernel x T failed; the only roots were
    6.6-8.6 e charge-density-wave states). The wall was a property of
    the d-first parameter rotation: with the per-angular-momentum
    projection the same cell converges in a few Newton iterations to a
    near-neutral metallic state. This pin guards the wall's absence.
    Re-pinned with issue #354's cyclic coordination numbers
    (-3.852071156 -> -3.865789830; orbit charge spread 3.2e-4 e)."""
    from vibeqc.semiempirical import run_gfn2_seccm

    a = 3.7958
    prim = [
        np.array([0.0, 0.5, 0.5]) * a,
        np.array([0.5, 0.0, 0.5]) * a,
        np.array([0.5, 0.5, 0.0]) * a,
    ]
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(2)
        for j in range(2)
        for k in range(4)
    ]
    translations = [2 * prim[0], 2 * prim[1], 4 * prim[2]]
    molecule = Molecule(
        [Atom(29, (np.asarray(c) * _BOHR).tolist()) for c in atoms],
        0,
        1,
    )
    topology = build_seccm_topology(
        [np.asarray(c) * _BOHR for c in atoms],
        [np.asarray(t) * _BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * _BOHR for p in prim],
        replicas=(2, 2, 4),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    with pytest.raises(ValueError, match="3-D ewald_gamma.*no thermodynamic limit"):
        run_gfn2_seccm(
            molecule,
            topology,
            ewald_gamma=True,
            scc_mixer="newton",
            electronic_temperature=0.002,
            max_iter=100,
        )
    # assert result.converged and result.physical_basin
    # assert result.energy == pytest.approx(-3.865789830, abs=1.0e-6)
    # q_rms = float(np.sqrt((np.asarray(result.charges) ** 2).mean()))
    # assert q_rms < 0.2
    # assert result.translation_symmetry_charge_spread < 1.0e-3


def _lih_chain_1d(shift_bohr: float = 0.0):
    """Two-cell LiH wire along x: primitive L = 8 bohr, Li at 0, H at L/2."""
    a = 8.0
    atoms = []
    zs = []
    for cell in range(2):
        atoms.append(np.array([cell * a + shift_bohr, 0.0, 0.0]))
        zs.append(3)
        atoms.append(np.array([cell * a + a / 2.0 + shift_bohr, 0.0, 0.0]))
        zs.append(1)
    molecule = Molecule(
        [Atom(z, c.tolist()) for z, c in zip(zs, atoms)], 0, 1
    )
    topology = build_seccm_topology(
        atoms,
        [np.array([2.0 * a, 0.0, 0.0])],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.array([a, 0.0, 0.0])],
        replicas=(2, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def test_gfn2_seccm_ewald_gamma_1d_kernel_alpha_independence_and_anchor():
    """The 1-D wire shell-gamma kernel (background-corrected Parry-type
    wire Ewald, ewald_1d.h) is alpha-independent within the converged
    series - the Ewald-identity oracle - and its pure-Coulomb part
    matches the direct background-corrected lattice sum
    Phi(d) = sum_n [1/|d+nT| - (n != 0: 1/|nT|)] evaluated by
    brute force with Richardson extrapolation (an implementation-
    independent anchor)."""
    import math

    molecule, topology = _lih_chain_1d()
    L = 16.0
    gamma_ref = _native_gfn2_ewald_shell_gamma(molecule, topology, 4.0 / L)
    for alpha in (3.0 / L, 6.0 / L):
        gamma_alt = _native_gfn2_ewald_shell_gamma(
            molecule, topology, alpha
        )
        assert np.abs(gamma_ref - gamma_alt).max() < 1.0e-12

    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    params = load_gfn2_params()
    basis = _se.SemiempiricalBasis.build(molecule, params, 0)
    shells = list(_se.gfn2_enumerate_shells(basis, molecule, params))
    hard = [max(float(s.hardness), 1.0e-6) for s in shells]
    si = 0
    sj = len([s for s in shells if int(s.atom_idx) == 0])
    d = np.array([4.0, 0.0, 0.0])  # Li(0) -> H(0) displacement (bohr)
    eta = 2.0 / (hard[si] + hard[sj])
    ws_corr = 0.0
    for image in topology.cells[0]:
        if image.origin != 1:
            continue
        r = float(np.linalg.norm(image.disp))
        if r < 1.0e-12:
            continue
        ws_corr += image.weight * (
            1.0 / math.sqrt(r * r + eta * eta) - 1.0 / r
        )
    phi_kernel = float(gamma_ref[si, sj]) - ws_corr

    T = np.array([L, 0.0, 0.0])

    def brute(n_images: int) -> float:
        total = 1.0 / float(np.linalg.norm(d))
        for n in range(1, n_images + 1):
            total += 1.0 / float(np.linalg.norm(d + n * T)) - 1.0 / (n * L)
            total += 1.0 / float(np.linalg.norm(d - n * T)) - 1.0 / (n * L)
        return total

    b1, b2 = brute(20000), brute(40000)
    extrapolated = b2 + (b2 - b1)
    assert phi_kernel == pytest.approx(extrapolated, abs=1.0e-9)


def test_gfn2_seccm_ewald_gamma_1d_chain_converges_and_is_invariant():
    """Nontrivial 1-D cells no longer fail closed: the LiH wire under
    the self-consistent 1-D wire ewald_gamma kernel converges into the
    physical basin with ionic charges, and the energy is invariant
    under a rigid translation along the chain (pinned 2026-08-27, the
    1-D kernel landing)."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _lih_chain_1d()
    result = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        electronic_temperature=0.0,
        max_iter=600,
    )
    assert result.converged and result.physical_basin
    # Re-pinned with issue #354's cyclic H0 coordination numbers
    # (-0.805035101 -> -0.804626976); the rigid-translation invariance
    # asserted below is unchanged.
    assert result.energy == pytest.approx(-0.804626976, abs=1.0e-6)
    charges = np.asarray(result.charges)
    assert charges[0] > 0.3 and charges[2] > 0.3  # Li cationic
    assert charges[1] < -0.3 and charges[3] < -0.3  # H anionic

    shifted_molecule, shifted_topology = _lih_chain_1d(shift_bohr=1.7)
    shifted = run_gfn2_seccm(
        shifted_molecule,
        shifted_topology,
        ewald_gamma=True,
        electronic_temperature=0.0,
        max_iter=600,
    )
    assert shifted.energy == pytest.approx(result.energy, abs=1.0e-10)


def test_gfn2_seccm_1d_chain_translation_invariance_and_normalization():
    from vibeqc.semiempirical import run_gfn2_seccm

    primitive = np.array([8.0, 0.0, 0.0])
    for replicas in (2, 3):
        coords = np.vstack([_H2O_COORDS + cell * primitive for cell in range(replicas)])
        shifted = coords + np.array([1.0, 0.0, 0.0])

        def energy(geometry):
            topology = _topology(
                geometry,
                [replicas * primitive],
                replicas=(replicas, 1, 1),
                primitive_vectors=[primitive],
            )
            return run_gfn2_seccm(_h2o_grid_molecule(geometry), topology)

        result = energy(coords)
        assert result.converged
        assert np.isfinite(result.energy)
        # Per-cell normalization: energy = total_cyclic_energy / group_order.
        assert result.energy == pytest.approx(
            result.total_cyclic_energy / result.group_order, abs=1.0e-15
        )
        # Cyclic components add up and normalize per cell.
        assert (
            result.cyclic_electronic_energy + result.cyclic_repulsive_energy
        ) == pytest.approx(result.total_cyclic_energy, abs=1.0e-14)
        assert result.e_scc == pytest.approx(
            result.cyclic_scc_energy / result.group_order, abs=1.0e-15
        )
        assert result.e_band0 == pytest.approx(
            result.cyclic_band0_energy / result.group_order, abs=1.0e-15
        )
        assert result.e_aes == pytest.approx(
            result.cyclic_aes_energy / result.group_order, abs=1.0e-15
        )
        assert result.e_3rd == pytest.approx(
            result.cyclic_3rd_energy / result.group_order, abs=1.0e-15
        )
        # Translation invariance along the cyclic direction (rigid shift).
        assert energy(coords).energy == pytest.approx(
            energy(shifted).energy, abs=1.0e-12
        )


def test_gfn2_seccm_2d_square_axis_swap():
    from vibeqc.semiempirical import run_gfn2_seccm

    primitives = [
        np.array([8.0, 0.0, 0.0]),
        np.array([0.0, 8.0, 0.0]),
    ]

    def grid(sites):
        return np.array(
            [
                origin + site
                for idx in product(range(2), repeat=2)
                for origin in [idx[0] * primitives[0] + idx[1] * primitives[1]]
                for site in sites
            ]
        )

    # Lattice-symmetry invariance: swapping the two square-lattice axes
    # (with the cell geometry swapped the same way) must not move the energy.
    coords_xy = grid(_H2O_COORDS)
    coords_yx = grid(_H2O_COORDS[:, [1, 0, 2]])
    e_xy = run_gfn2_seccm(
        _h2o_grid_molecule(coords_xy),
        _topology(
            coords_xy,
            [2 * p for p in primitives],
            replicas=(2, 2, 1),
            primitive_vectors=primitives,
        ),
    )
    e_yx = run_gfn2_seccm(
        _h2o_grid_molecule(coords_yx),
        _topology(
            coords_yx,
            [2 * p for p in primitives],
            replicas=(2, 2, 1),
            primitive_vectors=primitives,
        ),
    )
    assert e_xy.converged and e_yx.converged
    assert np.isfinite(e_xy.energy)
    assert e_xy.energy == pytest.approx(e_yx.energy, abs=1.0e-12)


def test_gfn2_seccm_unsupported_inputs_fail_closed():
    from vibeqc.semiempirical import run_gfn2_seccm

    translations = [np.array([40.0, 0.0, 0.0])]
    topology = _topology(
        _H2O_COORDS,
        translations,
        replicas=(1, 1, 1),
        primitive_vectors=translations,
    )
    charged = Molecule(
        [Atom(8, c.tolist()) for c in _H2O_COORDS[:1]]
        + [Atom(1, c.tolist()) for c in _H2O_COORDS[1:]],
        2,
        1,
    )
    with pytest.raises(NotImplementedError, match="neutral"):
        run_gfn2_seccm(charged, topology)
    open_shell = Molecule(
        [Atom(8, c.tolist()) for c in _H2O_COORDS[:1]]
        + [Atom(1, c.tolist()) for c in _H2O_COORDS[1:]],
        0,
        3,
    )
    with pytest.raises(NotImplementedError, match="closed-shell"):
        run_gfn2_seccm(open_shell, topology)


def test_native_broyden_history_reset_and_rollover_are_process_safe():
    script = """
import numpy as np
from vibeqc._vibeqc_core import semiempirical as se

mixer = se._BroydenMixer(memory=2, damping=0.4)
try:
    mixer.mix(np.zeros(3), np.zeros(4), 1)
except ValueError as error:
    assert "equal sizes" in str(error)
else:
    raise AssertionError("mismatched Broyden vectors were accepted")

mixed = np.zeros(4)
for iteration in range(1, 13):
    raw = np.array([
        np.sin(0.37 * iteration),
        np.cos(0.23 * iteration),
        (-1.0) ** iteration * 0.1,
        0.05 * iteration,
    ])
    mixed = np.asarray(mixer.mix(mixed, raw, iteration))
    assert np.isfinite(mixed).all()

mixer.reset()
for iteration in range(1, 8):
    raw = np.array([0.02 * iteration, -0.03 * iteration, 0.1, -0.1])
    mixed = np.asarray(mixer.mix(mixed, raw, iteration))
    assert np.isfinite(mixed).all()

# Force the coefficient-norm guard into its damped fallback. The following
# call used to write column zero of a dim-by-zero history matrix.
mixer.reset()
mixed = np.zeros(4)
mixed = np.asarray(mixer.mix(mixed, np.ones(4), 1))
mixed = np.asarray(
    mixer.mix(mixed, np.array([1.0e12, -1.0e12, 2.0e12, -2.0e12]), 2)
)
assert np.isfinite(mixed).all()
mixed = np.asarray(mixer.mix(mixed, np.array([0.2, -0.1, 0.3, -0.4]), 3))
assert np.isfinite(mixed).all()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _b1_100_checkerboard(
    n_planes: int, *, n: int = 2, remove_surface_o: bool = False
) -> tuple[Molecule, object]:
    """Primitive B1 rocksalt(100) slab fixture: 45-degree primitives
    [a/2, a/2] and [-a/2, a/2], site-swapped adjacent planes, cyclic
    2x2 in-plane translations. Optionally removes the surface O at
    (a/2, 0, 0) (the defect-ladder F0 vacancy)."""
    a = 4.212
    t1 = np.array([a / 2.0, a / 2.0, 0.0])
    t2 = np.array([-a / 2.0, a / 2.0, 0.0])
    unlike_offset = np.array([a / 2.0, 0.0, 0.0])
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(n_planes):
        z_offset = np.array([0.0, 0.0, k * a / 2.0])
        for i in range(n):
            for j in range(n):
                home = i * t1 + j * t2 + z_offset
                if k % 2 == 0:
                    atoms.extend((home, home + unlike_offset))
                    zs.extend((12, 8))
                else:
                    atoms.extend((home + unlike_offset, home))
                    zs.extend((12, 8))
    if remove_surface_o:
        target = np.array([a / 2.0, 0.0, 0.0])
        for idx, (coord, z) in enumerate(zip(atoms, zs)):
            if z == 8 and np.linalg.norm(coord - target) < 1.0e-6:
                del atoms[idx]
                del zs[idx]
                break
        else:
            raise RuntimeError("vacancy site not found")
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * _BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    supercell_vectors = [n * t1, n * t2]
    topology = _topology(
        [np.asarray(c) * _BOHR for c in atoms],
        [np.asarray(v) * _BOHR for v in supercell_vectors],
        replicas=(1, 1, 1),
        primitive_vectors=[np.asarray(v) * _BOHR for v in supercell_vectors],
    )
    return molecule, topology


def _assert_screened_metric_orthonormality(result):
    """#151: verify the actual retained eigensystem, not only its energy."""
    overlap = np.asarray(result.overlap)
    coefficients = np.asarray(result.mo_coeffs)
    energies = np.asarray(result.mo_energies)
    assert result.n_occ <= len(energies) < result.n_basis
    assert coefficients.shape == (result.n_basis, len(energies))
    assert np.isfinite(energies).all()
    np.testing.assert_allclose(coefficients.T @ overlap @ coefficients,
                               np.eye(len(energies)), rtol=0.0, atol=1.0e-10)


def test_gfn2_indefinite_overlap_screens_and_converges_b1_100_slab():
    """The WS-weighted cyclic overlap of the corrected B1(100) slab cell
    has non-positive eigenvalues (the Peintinger-Bredow C-point failure
    mode; IID 215). Canonical orthogonalization screens the non-positive
    subspace and the SCC converges to a physical slab state instead of the
    pre-screening runaway or the fail-close error. The four-plane 2x2 cell
    is the defect-ladder pristine fixture."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _b1_100_checkerboard(4)

    result = run_gfn2_seccm(
        molecule, topology, ewald_gamma=True, max_iter=3000
    )
    charges = np.asarray(result.charges)
    assert result.converged
    assert result.physical_basin
    assert result.n_iter < 600
    _assert_screened_metric_orthonormality(result)
    # Pre-screening this cell hard-failed the overlap gate; the screened
    # solve lands on the physical slab basin (molecular driver: -2.3623
    # Ha/atom, q_rms 0.480; the 2-D cyclic cell is slightly more ionic).
    # 2026-08-28 (issue #433): re-pinned after the full Pauling EN table
    # reached the Mg-O pairs (tblite parity).  2026-09-02 (issue #354):
    # re-pinned again with the cyclic H0 coordination numbers
    # (-2.365887702 -> -2.373082177).
    assert result.energy * result.group_order / len(charges) == pytest.approx(
        -2.373082177, abs=5.0e-3
    )
    assert float(np.sqrt((charges**2).mean())) == pytest.approx(0.52, abs=0.05)

    # The unscreened metric stays exposed (the cyclic S_CCM contract), and
    # the screened solve is consistent across kernel flavors.
    for kwargs in ({}, {"madelung": True}):
        other = run_gfn2_seccm(molecule, topology, max_iter=3000, **kwargs)
        assert other.converged and other.physical_basin
        _assert_screened_metric_orthonormality(other)
        other_charges = np.asarray(other.charges)
        assert float(np.sqrt((other_charges**2).mean())) < 0.8


def test_gfn2_seccm_b1_100_o_vacancy_converges():
    """The neutral surface O vacancy (F0 center) on the corrected B1(100)
    slab converges with the screened overlap (IID 155). The raw cyclic
    vacancy cost is pinned as a regression anchor; the F-center charge
    state stays on the pristine scale."""
    from vibeqc.semiempirical import run_gfn2_seccm

    pristine_mol, pristine_topology = _b1_100_checkerboard(4)
    vacancy_mol, vacancy_topology = _b1_100_checkerboard(
        4, remove_surface_o=True
    )
    pristine = run_gfn2_seccm(
        pristine_mol, pristine_topology, ewald_gamma=True, max_iter=3000
    )
    vacancy = run_gfn2_seccm(
        vacancy_mol, vacancy_topology, ewald_gamma=True, max_iter=3000
    )
    assert vacancy.converged and vacancy.physical_basin
    _assert_screened_metric_orthonormality(pristine)
    _assert_screened_metric_orthonormality(vacancy)
    vacancy_charges = np.asarray(vacancy.charges)
    pristine_charges = np.asarray(pristine.charges)
    assert float(np.sqrt((vacancy_charges**2).mean())) == pytest.approx(
        float(np.sqrt((pristine_charges**2).mean())), abs=0.1
    )
    # 2026-08-28 (issue #433): re-pinned after the full Pauling EN table
    # reached the Mg-O pairs (tblite parity).
    assert vacancy.energy - pristine.energy == pytest.approx(
        4.1226724423525525, abs=0.05
    )


@pytest.mark.parametrize(
    "method,temperature",
    [("dftb0", 0.0), ("scc_dftb", 0.0), ("scc_dftb", 0.005),
     ("gfn2", 0.0), ("gfn2", 0.005)],
)
def test_seccm_screened_occupied_space_without_lumo_is_rejected(method, temperature):
    """#151: two almost coincident H2 units retain only two occupied modes.

    This synthetic overlap stress fixture has four AOs and two occupied
    orbitals. Screening must not turn the missing third eigenvalue into an
    out-of-bounds read, or into a zero gap waived by finite temperature.
    """
    from vibeqc.semiempirical import run_gfn2_seccm
    from vibeqc.semiempirical.seccm.gfn2 import GFN2SECCMConvergenceError

    primitive = np.array([8.0, 0.0, 0.0])
    distance = 4.0e-6
    coords = np.array([[0., 0., 0.], [distance, 0., 0.],
                       [8., 0., 0.], [8. + distance, 0., 0.]])
    topology = _topology(coords, [2 * primitive], replicas=(2, 1, 1),
                         primitive_vectors=[primitive])
    molecule = Molecule([Atom(1, p.tolist()) for p in coords], 0, 1)
    with pytest.raises(RuntimeError, match="positive finite-torus HOMO-LUMO gap") as exc:
        if method == "dftb0":
            run_dftb0_seccm(molecule, topology)
        elif method == "scc_dftb":
            run_scc_dftb_seccm(molecule, topology,
                               electronic_temperature=temperature)
        else:
            run_gfn2_seccm(molecule, topology, max_iter=30,
                           electronic_temperature=temperature, include_aes=False)

    if method == "gfn2":
        assert isinstance(exc.value, GFN2SECCMConvergenceError)
        result = exc.value.result
        assert result.n_basis == 4
        assert len(result.mo_energies) == result.n_occ == 2
        assert not result.converged and not result.gap_guard_waived
        assert np.isnan(result.homo_lumo_gap)
        assert exc.value.reason == "gap_rejected"
        coefficients = np.asarray(result.mo_coeffs)
        overlap = np.asarray(result.overlap)
        np.testing.assert_allclose(coefficients.T @ overlap @ coefficients,
                                   np.eye(2), rtol=0.0, atol=1.0e-12)


def test_scc_dftb_seccm_indefinite_overlap_screens_and_runs():
    """A tightly packed H4 cyclic cell whose stitched overlap has
    non-positive modes runs through canonical orthogonalization instead of
    hard-failing. The screened result is finite (the geometry is a
    synthetic degenerate stress fixture, not a physical cell)."""
    primitive = np.array([0.2, 0.0, 0.0])
    replicas = 4
    coords = np.array(
        [[cell * primitive[0], 0.0, 0.0] for cell in range(replicas)]
    )
    topology = _topology(
        coords,
        [replicas * primitive],
        replicas=(replicas, 1, 1),
        primitive_vectors=[primitive],
    )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coords], 0, 1
    )
    result = run_scc_dftb_seccm(molecule, topology, max_iter=20)
    assert result.converged
    assert np.isfinite(result.energy)
    _assert_screened_metric_orthonormality(result)


def test_scc_dftb_seccm_indefinite_overlap_fails_closed_when_occupied_lost():
    """When canonical orthogonalization would drop the occupied manifold
    (a nearly coincident H4 cell), the adapter keeps the fail-closed
    contract instead of solving in a subspace that cannot hold the
    electrons.

    The spacing sits in the only window that reaches that contract once the
    Wigner-Seitz builders share ties within 1e-5 bohr (issues #592/#593).
    Measured on this cell: at 2e-6 bohr and below the cyclic cell (4x the
    spacing) is so much smaller than the tie tolerance that every image of
    every atom ties, and the topology fails the directed ownership rule
    before any matrix is assembled; at 6e-6 bohr and above the stitched
    overlap eigenvalues rise above the 1e-10 screening threshold, canonical
    orthogonalization keeps the whole space, and the run reaches the
    iteration budget instead.  Between them (3e-6 to 5e-6 bohr) the topology
    validates -- carrying the shared images the tolerance is there to keep --
    and the occupied manifold is still lost, which is what this test pins.
    """
    primitive = np.array([4.0e-6, 0.0, 0.0])
    replicas = 4
    coords = np.array(
        [[cell * primitive[0], 0.0, 0.0] for cell in range(replicas)]
    )
    topology = _topology(
        coords,
        [replicas * primitive],
        replicas=(replicas, 1, 1),
        primitive_vectors=[primitive],
    )
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coords], 0, 1
    )

    with pytest.raises(
        RuntimeError,
        match="SCC-DFTB-SECCM overlap matrix is not positive definite",
    ):
        run_scc_dftb_seccm(molecule, topology, max_iter=1)


@pytest.mark.parametrize(
    "relative_path",
    [
        "studies/seccm-bulk3d/probe_slab_jacobian.py",
        "studies/seccm-bulk3d/probe_slab_channel_isolation.py",
    ],
)
def test_gfn2_study_probes_bind_native_electrostatics_by_keyword(
    relative_path,
):
    source = (Path(__file__).parents[1] / relative_path).read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_run_gfn2_seccm_from_records"
    ]
    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords}
    assert {
        "madelung",
        "madelung_s_weighted",
        "madelung_no_self",
        "ewald_gamma",
        "include_aes",
    } <= keyword_names
    ewald_keyword = next(
        keyword
        for keyword in calls[0].keywords
        if keyword.arg == "ewald_gamma"
    )
    assert isinstance(ewald_keyword.value, ast.Constant)
    assert ewald_keyword.value.value is True


def test_seccm_translation_orbits_and_symmetry_charge_spread():
    """The translation-orbit invariant (issue #421, maintainer decision
    D3): atoms related by a primitive lattice vector of the bound finite
    group are the same crystal site and must carry equal charges, so the
    spread within an orbit is zero for a symmetry-respecting state and
    nonzero exactly when the state breaks the model's own translation
    symmetry.

    Pinned on the two-sublattice rocksalt case, where the invariant must
    separate the cation and anion sublattices rather than lump them: an
    ideal ionic pattern has spread 0 despite carrying charges of +-1, and
    a single perturbed site surfaces at exactly its perturbation."""
    from vibeqc.semiempirical.seccm._adapter_common import (
        translation_orbits,
        translation_symmetry_charge_spread,
    )

    molecule, atoms, translations = _rocksalt_mgo_2x2x2(4.212)
    topology = _rocksalt_topology(atoms, translations)

    orbits = translation_orbits(molecule, topology)
    species = [int(atom.Z) for atom in molecule.atoms]
    assert len(orbits) == 2
    assert sorted(len(orbit) for orbit in orbits) == [8, 8]
    for orbit in orbits:
        assert len({species[index] for index in orbit}) == 1

    ideal = np.array([1.0 if z == 12 else -1.0 for z in species])
    assert translation_symmetry_charge_spread(
        molecule, topology, ideal
    ) == pytest.approx(0.0, abs=1.0e-12)

    perturbed = ideal.copy()
    perturbed[0] += 0.37
    assert translation_symmetry_charge_spread(
        molecule, topology, perturbed
    ) == pytest.approx(0.37, abs=1.0e-12)


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("rotated", [False, True])
def test_seccm_symmetry_diagnostic_preserves_layers_and_species(dimension, rotated):
    """#421: equal projected coordinates do not make different layers equivalent."""
    from vibeqc.semiempirical.seccm._adapter_common import (
        translation_orbits,
        translation_symmetry_charge_spread,
    )

    primitive = np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]])[:dimension]
    coords = np.array([
        [0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        [0.0, 0.0, 2.0], [4.0, 0.0, 2.0],
        [0.0, 0.0, 4.0], [4.0, 0.0, 4.0],
    ])
    if rotated:
        rotation, _ = np.linalg.qr(np.array([
            [1.0, 2.0, -1.0], [2.0, -1.0, 3.0], [3.0, 1.0, 2.0],
        ]))
        coords = coords @ rotation
        primitive = primitive @ rotation
    species = [1, 1, 1, 1, 1, 2]
    molecule = Molecule([
        Atom(z, xyz.tolist()) for z, xyz in zip(species, coords)
    ], 0, 2)
    translations = primitive.copy()
    translations[0] *= 2
    topology = _topology(
        coords, translations, replicas=(2, 1, 1),
        primitive_vectors=primitive,
    )
    assert translation_orbits(molecule, topology) == (
        (0, 1), (2, 3), (4,), (5,),
    )
    charges = np.array([0.2, 0.2, -0.4, -0.4, 0.7, -0.3])
    assert translation_symmetry_charge_spread(
        molecule, topology, charges
    ) == pytest.approx(0.0, abs=1e-12)
    charges[3] += 0.15
    assert translation_symmetry_charge_spread(
        molecule, topology, charges
    ) == pytest.approx(0.15, abs=1e-12)


def test_seccm_symmetry_diagnostic_nonfinite_is_unavailable():
    """#421: NaN in a charge vector must not masquerade as zero spread."""
    from vibeqc.semiempirical.seccm._adapter_common import (
        translation_symmetry_charge_spread,
    )

    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    molecule = Molecule([Atom(1, xyz.tolist()) for xyz in coords], 0, 1)
    topology = _topology(
        coords, [[8.0, 0.0, 0.0]], replicas=(2, 1, 1),
        primitive_vectors=[[4.0, 0.0, 0.0]],
    )
    assert np.isnan(translation_symmetry_charge_spread(
        molecule, topology, [float("nan"), 0.0]
    ))


def test_gfn2_seccm_state_comparison_reports_hops_and_changed_temperature():
    """#421: a size ladder's RMS/gap jump is visible without a guessed threshold."""
    from dataclasses import replace
    from vibeqc.semiempirical import (
        compare_gfn2_seccm_states,
        run_gfn2_seccm,
    )

    # A real, permitted molecular result supplies the provenance. The
    # modified snapshots below are controlled diagnostic inputs, not new
    # SCC calculations or a claim to reproduce the historical MgO ladder.
    topology = _topology(
        _H2O_COORDS, [[40.0, 0.0, 0.0]], replicas=(1, 1, 1),
        primitive_vectors=[[40.0, 0.0, 0.0]],
    )
    result = run_gfn2_seccm(_h2o_molecule(_H2O_COORDS), topology)
    previous = replace(
        result, charges=np.array([2.0, -2.0]), group_order=1,
        homo_lumo_gap=0.35, e_scc=-0.4, free_energy=-1.0,
        translation_symmetry_charge_spread=0.0,
    )
    current = replace(
        previous, charges=np.array([1.6, -1.6] * 4), group_order=4,
        homo_lumo_gap=0.003, e_scc=0.1, free_energy=-0.98,
        translation_symmetry_charge_spread=0.7,
    )
    change = compare_gfn2_seccm_states(previous, current)
    assert change.charge_rms_change == pytest.approx(-0.4)
    assert change.homo_lumo_gap_change == pytest.approx(-0.347)
    assert change.scc_energy_change == pytest.approx(0.5)
    assert change.free_energy_change == pytest.approx(0.02)
    # Both production fields alias today, but the diagnostic must consume
    # the named thermodynamic field even on an inconsistent input snapshot.
    assert current.energy == previous.energy
    assert change.translation_symmetry_spread_change == pytest.approx(0.7)
    assert change.same_hamiltonian

    # Replication alone cannot create a charge-RMS or per-cell energy jump.
    replicated = replace(previous, charges=np.tile(previous.charges, 4))
    same = compare_gfn2_seccm_states(previous, replicated)
    assert same.charge_rms_change == 0.0
    assert same.free_energy_change == 0.0
    assert same.same_hamiltonian

    warmer = replace(
        current,
        smearing_temperature=0.005,
        hamiltonian_identity=replace(
            current.hamiltonian_identity,
            resolved_electronic_temperature=0.005,
        ),
    )
    changed = compare_gfn2_seccm_states(previous, warmer)
    assert not changed.same_hamiltonian
    assert changed.electronic_temperature_change == pytest.approx(0.005)
    assert compare_gfn2_seccm_states(
        current, previous
    ).charge_rms_change == pytest.approx(-change.charge_rms_change)
    with pytest.raises(ValueError, match="converged"):
        compare_gfn2_seccm_states(previous, replace(current, converged=False))
    with pytest.raises(ValueError, match="finite"):
        compare_gfn2_seccm_states(
            previous, replace(current, charges=np.array([np.nan]))
        )
    with pytest.raises(ValueError, match="per-primitive-cell"):
        compare_gfn2_seccm_states(
            previous, replace(current, normalization="cyclic_total")
        )


def test_gfn2_seccm_reports_translation_symmetry_charge_spread():
    """Every GFN2-SECCM result carries the orbit-spread diagnostic, and
    it is reported rather than gated (maintainer decision D3).

    This uses the permitted AES-active 2-D stress cell after #444.
    Historical Cu 3-D ewald_gamma evidence remains in the named #130 gates;
    those production calls now fail closed before a state exists."""
    from vibeqc.semiempirical import run_gfn2_seccm

    molecule, topology = _aligned_plane_stress_slab(2)
    result = run_gfn2_seccm(
        molecule,
        topology,
        ewald_gamma=True,
        scc_mixer="broyden_eyert",
        electronic_temperature=0.001,
        max_iter=300,
    )
    assert result.converged
    assert result.physical_basin
    assert np.isfinite(result.translation_symmetry_charge_spread)
    assert result.translation_symmetry_charge_spread < 1.0e-3


def _h2o_two_replica_torus():
    """Two-replica H2O torus: primitive [6, 0, 0] bohr, replicas (2, 1, 1).

    Returns the six-atom coordinates (species [8, 1, 1, 8, 1, 1]) and the
    primitive vector; callers retype a torus site by a full-torus translation
    or permute the atoms on the returned array."""
    primitive = np.array([6.0, 0.0, 0.0])
    coords = np.vstack([_H2O_COORDS, _H2O_COORDS + primitive])
    return coords, primitive


def _run_h2o_torus_no_aes(coords, species, primitive):
    from vibeqc.semiempirical import run_gfn2_seccm

    topology = _topology(
        coords,
        [2.0 * primitive],
        replicas=(2, 1, 1),
        primitive_vectors=[primitive],
    )
    molecule = Molecule(
        [Atom(int(z), c.tolist()) for z, c in zip(species, coords)], 0, 1
    )
    return run_gfn2_seccm(molecule, topology, include_aes=False)


def test_gfn2_seccm_h0_coordination_is_torus_representative_invariant():
    """Issue #354: the Eq. 18 coordination numbers that set the GFN2 H0
    self-energies are assembled from the directed Wigner-Seitz records
    (physical image distances, fractional ownership weights), so retyping a
    torus site by any full-torus translation cannot move the H0.

    Pre-fix, the CNs were recomputed molecularly from the typed home-cell
    atoms for every image block: moving one H by the full torus translation
    changed the include_aes=False total energy by -3.83e-3 Ha (e_band0
    -3.66e-3 Ha) on this fixture, and the two by-construction-equivalent
    water replicas carried charges differing by 2.6e-5 e. AES is excluded to
    isolate the H0 path (its own representative dependence is issue #348)."""
    species = [8, 1, 1, 8, 1, 1]
    base, primitive = _h2o_two_replica_torus()
    reference = _run_h2o_torus_no_aes(base, species, primitive)
    assert reference.converged and not reference.molecular_delegated
    assert reference.e_aes == 0.0

    # The two replicas are translation-equivalent by construction, so the
    # converged charges must be replica-symmetric.
    q_ref = np.asarray(reference.charges)
    np.testing.assert_allclose(q_ref[:3], q_ref[3:], atol=1.0e-9)

    fields = ("energy", "e_band0", "e_scc", "e_3rd", "e_repulsive",
              "total_cyclic_energy", "homo_lumo_gap")
    for site in (0, 3):
        shifted = base.copy()
        shifted[site] += 2.0 * primitive  # full torus translation
        relabelled = _run_h2o_torus_no_aes(shifted, species, primitive)
        assert relabelled.converged
        assert relabelled.n_iter == reference.n_iter
        for field in fields:
            assert float(getattr(relabelled, field)) == pytest.approx(
                float(getattr(reference, field)), abs=1.0e-10
            ), field
        np.testing.assert_allclose(
            np.asarray(relabelled.charges), q_ref, atol=1.0e-9
        )
        np.testing.assert_allclose(
            np.asarray(relabelled.overlap), np.asarray(reference.overlap),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(relabelled.hamiltonian),
            np.asarray(reference.hamiltonian),
            atol=1.0e-9,
        )


def test_gfn2_seccm_h0_coordination_is_atom_permutation_covariant():
    """Issue #354: one directed CN vector per atom is permutation-covariant
    (the H0 image blocks are reused, not retyped, per atom order)."""
    species = [8, 1, 1, 8, 1, 1]
    base, primitive = _h2o_two_replica_torus()
    reference = _run_h2o_torus_no_aes(base, species, primitive)
    for perm in ([3, 4, 5, 0, 1, 2], [1, 0, 2, 5, 3, 4], [2, 5, 1, 4, 0, 3]):
        permuted = _run_h2o_torus_no_aes(
            base[perm], [species[p] for p in perm], primitive
        )
        assert permuted.converged
        assert float(permuted.energy) == pytest.approx(
            float(reference.energy), abs=1.0e-10
        )
        assert float(permuted.e_band0) == pytest.approx(
            float(reference.e_band0), abs=1.0e-10
        )
        np.testing.assert_allclose(
            np.asarray(permuted.charges)[np.argsort(perm)],
            np.asarray(reference.charges),
            atol=1.0e-9,
        )


def _hf_chain_1d(nrep, cell=6.0, bond=2.2):
    """Polar H-F wire: the class the CCM Madelung embedding exists for."""
    coords = []
    zs = []
    for cell_index in range(nrep):
        coords.append(np.array([cell_index * cell, 0.0, 0.0]))
        zs.append(1)
        coords.append(np.array([cell_index * cell + bond, 0.0, 0.0]))
        zs.append(9)
    molecule = Molecule(
        [Atom(z, c.tolist()) for z, c in zip(zs, coords)], 0, 1
    )
    topology = bind_finite_group(
        build_seccm_topology(
            coords,
            [np.array([nrep * cell, 0.0, 0.0])],
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=[np.array([cell, 0.0, 0.0])],
        replicas=(nrep, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def test_pm6_seccm_truncated_electrostatics_requires_acknowledgement():
    """A nontrivial cyclic PM6 cell without the embedding is not
    quantitative and must say so (issue #217, maintainer decision D4).

    The WS-truncated monopole sum PM6's Fock carries reproduces the exact
    rocksalt Madelung constant only to -43%..+38%, with the sign flipping
    on replica parity, so a silent run would be a wrong answer rather
    than an approximation. The route therefore refuses unless the caller
    either enables the embedding or acknowledges the truncation."""
    from vibeqc.semiempirical.seccm.pm6 import run_pm6_seccm

    molecule, topology = _hf_chain_1d(3)
    with pytest.raises(ValueError, match="long-range Coulomb tail"):
        run_pm6_seccm(molecule, topology, max_iter=400)

    acknowledged = run_pm6_seccm(
        molecule, topology, max_iter=400, allow_truncated_electrostatics=True
    )
    assert acknowledged.converged
    assert acknowledged.e_madelung == 0.0


def test_pm6_seccm_madelung_embedding_1d_energy_bookkeeping():
    """The opt-in 1-D embedding runs, changes the energy, and closes.

    MSINDO convention (ccmfockcl.f + madelsum.f): the diagonal Fock
    deposit carries the electronic half of 1/2 q.V_mad and e_madelung
    the core-charge half, with the potential taken as the Ewald lattice
    sum minus the WS-internal bare 1/r. The reported components must
    still sum to the total exactly, and the embedding must move the
    energy on a polar cell."""
    from vibeqc.semiempirical.seccm.pm6 import run_pm6_seccm

    molecule, topology = _hf_chain_1d(3)
    truncated = run_pm6_seccm(
        molecule, topology, max_iter=400, allow_truncated_electrostatics=True
    )
    embedded = run_pm6_seccm(molecule, topology, max_iter=400, madelung=True)

    assert embedded.converged
    assert embedded.e_madelung != 0.0
    assert embedded.energy != truncated.energy
    # e_electronic already contains the deposit's electronic half, so the
    # two-term closure must hold exactly.
    assert embedded.energy == pytest.approx(
        embedded.e_electronic + embedded.e_core, abs=1.0e-12
    )
    assert embedded.cyclic_madelung_energy == pytest.approx(
        embedded.e_madelung * embedded.group_order, rel=1.0e-12
    )


def test_pm6_seccm_madelung_fails_closed_in_three_dimensions():
    """3-D embedding is held, not shipped (maintainer decision D4): the
    WS-folded remainder there has no thermodynamic limit, so an embedded
    3-D energy would drift with cluster size rather than converge
    (issues #211, #425, #444)."""
    from vibeqc.semiempirical.seccm.pm6 import run_pm6_seccm

    a = 6.0
    coords = [np.zeros(3), np.array([a / 2, a / 2, a / 2])]
    molecule = Molecule(
        [Atom(1, coords[0].tolist()), Atom(9, coords[1].tolist())], 0, 1
    )
    translations = [
        np.array([a, 0.0, 0.0]),
        np.array([0.0, a, 0.0]),
        np.array([0.0, 0.0, a]),
    ]
    topology = bind_finite_group(
        build_seccm_topology(
            coords,
            translations,
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=translations,
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    with pytest.raises(ValueError, match="no thermodynamic limit"):
        run_pm6_seccm(molecule, topology, max_iter=400, madelung=True)
