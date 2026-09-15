"""Fail-closed regressions for the incomplete multi-k GFN2 model (issue #351)."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import (
    Atom,
    PeriodicSystem,
    bloch_kmesh_from_lists,
    monkhorst_pack,
)
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params


def _graphene_2c() -> PeriodicSystem:
    """Graphene 2-atom unit cell (AA = 2.46 A ~ 4.65 bohr)."""
    a = 4.65
    c = 20.0
    return PeriodicSystem(
        2,
        np.array(
            [[a, 0.0, 0.0], [a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, c]]
        ),
        [
            Atom(6, [0.0, 0.0, 0.0]),
            Atom(6, [a / 2.0, a * np.sqrt(3.0) / 6.0, 0.0]),
        ],
        0,
        1,
    )


def _mgo_2c(*, oxygen_image: int = 0) -> PeriodicSystem:
    """MgO primitive with an optionally relabelled oxygen site."""
    a = 4.211 * 1.8897261254535
    lattice = np.array(
        [
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        ]
    )
    oxygen = np.array([a / 2.0, 0.0, 0.0])
    oxygen += oxygen_image * lattice[:, 0]
    return PeriodicSystem(
        3,
        lattice,
        [Atom(12, [0.0, 0.0, 0.0]), Atom(8, oxygen.tolist())],
        0,
        1,
    )


def _in_plane_reciprocal_vectors(system: PeriodicSystem):
    """First two reciprocal lattice vectors of the 2-D cell (lattice columns)."""
    lattice = np.asarray(system.lattice)
    a1, a2, a3 = lattice[:, 0], lattice[:, 1], lattice[:, 2]
    volume = float(a1.dot(np.cross(a2, a3)))
    b1 = 2.0 * np.pi * np.cross(a2, a3) / volume
    b2 = 2.0 * np.pi * np.cross(a3, a1) / volume
    return b1, b2


def _linear_path(start, end, n_points):
    return [
        np.asarray(start, dtype=float)
        + (np.asarray(end, dtype=float) - np.asarray(start, dtype=float))
        * (i / (n_points - 1))
        for i in range(n_points)
    ]


def test_single_gamma_kmesh_delegates_to_supported_gamma_driver():
    """The compatibility spelling must return the exact Gamma result."""
    system = _graphene_2c()
    params = load_gfn2_params()
    gamma_mesh = bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0])
    # Default frontier smearing: the fixture's frontier pair is nearly
    # degenerate at Gamma, and a hard-Aufbau SCC flips between its members.
    options = _xtb.XTBSccOptions()
    options.max_iter = 400
    cutoff_bohr = 12.0

    direct = _xtb.run_gfn2_xtb_gamma(system, params, options, cutoff_bohr)
    through_mesh = _xtb.run_gfn2_xtb_kpoints(
        system, params, gamma_mesh, options, cutoff_bohr
    )

    assert direct.converged
    assert through_mesh.converged
    assert through_mesh.n_kpoints == 1
    assert len(through_mesh.eps_per_k) == 1
    assert through_mesh.energy == direct.energy
    assert through_mesh.free_energy == direct.free_energy
    assert through_mesh.fermi_level == direct.fermi_level
    assert through_mesh.entropy == direct.entropy
    assert through_mesh.smearing_temperature == direct.smearing_temperature
    assert through_mesh.n_iter == direct.n_iter
    assert through_mesh.parameter_identity == direct.parameter_identity
    assert through_mesh.parameter_sha256 == direct.parameter_sha256
    assert through_mesh.n_basis == len(through_mesh.band_energies)
    assert through_mesh.n_occ > 0
    np.testing.assert_array_equal(
        np.asarray(through_mesh.eps_per_k[0]),
        np.asarray(through_mesh.band_energies),
    )


def test_single_gamma_kmesh_energy_follows_lattice_site_relabelling():
    """The supported k-point spelling inherits Gamma translation covariance.

    2026-08-28 (issue #433): the full Pauling EN table moved the MgO SCC
    basin, and the exact-zero-temperature Aufbau configuration this test
    used to pin no longer converges at any budget -- the Gamma frontier
    flaps between SCC branches, the mode the default frontier smearing
    exists to prevent. The test now runs the default smearing path with a
    1000-iteration budget; the relabelling and direct-vs-kpoint invariants
    it exists to pin are unchanged.
    """
    params = load_gfn2_params()
    gamma_mesh = bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0])
    options = _xtb.XTBSccOptions()
    options.conv_tol_charge = 1.0e-10
    options.max_iter = 1000

    reference_system = _mgo_2c(oxygen_image=0)
    translated_system = _mgo_2c(oxygen_image=1)
    reference_oxygen = np.asarray(reference_system.unit_cell[1].xyz)
    translated_oxygen = np.asarray(translated_system.unit_cell[1].xyz)
    np.testing.assert_allclose(
        translated_oxygen - np.asarray(reference_system.lattice)[:, 0],
        reference_oxygen,
        rtol=0.0,
        atol=0.0,
    )

    direct = []
    through_mesh = []
    for system in (reference_system, translated_system):
        direct.append(
            _xtb.run_gfn2_xtb_gamma(
                system, params, options, cutoff_bohr=12.0
            )
        )
        through_mesh.append(
            _xtb.run_gfn2_xtb_kpoints(
                system,
                params,
                gamma_mesh,
                options,
                cutoff_bohr=12.0,
            )
        )

    assert all(result.converged for result in direct + through_mesh)
    for gamma, kpoint in zip(direct, through_mesh):
        assert [
            kpoint.energy,
            kpoint.free_energy,
            kpoint.e_electronic,
            kpoint.e_repulsive,
            kpoint.e_scc,
            kpoint.e_band0,
            kpoint.e_aes,
            kpoint.e_3rd,
        ] == [
            gamma.energy,
            gamma.free_energy,
            gamma.e_electronic,
            gamma.e_repulsive,
            gamma.e_scc,
            gamma.e_band0,
            gamma.e_aes,
            gamma.e_3rd,
        ]
        assert kpoint.energy == kpoint.e_electronic + kpoint.e_repulsive
        assert kpoint.e_electronic == (
            kpoint.e_band0 + kpoint.e_scc + kpoint.e_aes + kpoint.e_3rd
        )

    np.testing.assert_allclose(
        [
            through_mesh[1].energy,
            through_mesh[1].free_energy,
            through_mesh[1].e_electronic,
            through_mesh[1].e_repulsive,
            through_mesh[1].e_scc,
            through_mesh[1].e_band0,
            through_mesh[1].e_aes,
            through_mesh[1].e_3rd,
            through_mesh[1].fermi_level,
            through_mesh[1].entropy,
        ],
        [
            through_mesh[0].energy,
            through_mesh[0].free_energy,
            through_mesh[0].e_electronic,
            through_mesh[0].e_repulsive,
            through_mesh[0].e_scc,
            through_mesh[0].e_band0,
            through_mesh[0].e_aes,
            through_mesh[0].e_3rd,
            through_mesh[0].fermi_level,
            through_mesh[0].entropy,
        ],
        rtol=0.0,
        atol=1.0e-9,
    )


@pytest.mark.parametrize(
    "kpoint",
    ([np.nan, 0.0, 0.0], [1.0e-15, 0.0, 0.0]),
    ids=("nan", "tiny-nonzero"),
)
def test_malformed_or_nonzero_single_point_does_not_delegate(kpoint):
    system = _graphene_2c()
    params = load_gfn2_params()
    mesh = bloch_kmesh_from_lists([kpoint], [1.0])

    with pytest.raises(RuntimeError, match=r"non-Gamma GFN2 is disabled.*#351"):
        _xtb.run_gfn2_xtb_kpoints(system, params, mesh)


def test_non_gamma_kmesh_fails_closed_before_scc():
    system = _graphene_2c()
    params = load_gfn2_params()
    non_gamma = bloch_kmesh_from_lists([[0.1, 0.0, 0.0]], [1.0])

    with pytest.raises(RuntimeError, match=r"non-Gamma GFN2 is disabled.*#351"):
        _xtb.run_gfn2_xtb_kpoints(system, params, non_gamma)


def test_nonexecuted_d4_plumbing_cannot_mask_fail_closed_route_errors():
    system = _graphene_2c()
    params = load_gfn2_params()
    params.d4_a1 = np.nan
    non_gamma = bloch_kmesh_from_lists([[0.1, 0.0, 0.0]], [1.0])

    with pytest.raises(RuntimeError, match=r"non-Gamma GFN2 is disabled.*#351"):
        _xtb.run_gfn2_xtb_kpoints(system, params, non_gamma)

    empty = bloch_kmesh_from_lists([], [])
    with pytest.raises(ValueError, match="reference k-mesh is empty"):
        _xtb.run_gfn2_xtb_bandpath(
            system,
            params,
            [np.zeros(3), np.array([0.1, 0.0, 0.0])],
            empty,
        )


def test_multi_point_kmesh_fails_closed_before_scc():
    system = _graphene_2c()
    params = load_gfn2_params()

    with pytest.raises(RuntimeError, match=r"complex Bloch phases"):
        _xtb.run_gfn2_xtb_kpoints(
            system, params, monkhorst_pack(system, (2, 2, 1))
        )


def test_gfn2_band_path_fails_closed_with_reference_mesh():
    system = _graphene_2c()
    params = load_gfn2_params()
    gamma_mesh = bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0])
    path = _linear_path(np.zeros(3), np.array([0.1, 0.0, 0.0]), 3)

    with pytest.raises(RuntimeError, match=r"band paths are disabled.*#351"):
        _xtb.run_gfn2_xtb_bandpath(system, params, path, gamma_mesh)


def test_gfn2_band_path_fails_closed_on_empty_reference_mesh():
    system = _graphene_2c()
    params = load_gfn2_params()
    path = _linear_path(np.zeros(3), np.array([0.1, 0.0, 0.0]), 3)
    empty = bloch_kmesh_from_lists([], [])

    with pytest.raises(ValueError, match="reference k-mesh is empty"):
        _xtb.run_gfn2_xtb_bandpath(system, params, path, empty)


def test_gfn2_band_path_fails_closed_on_empty_path():
    system = _graphene_2c()
    params = load_gfn2_params()
    reference_mesh = monkhorst_pack(system, (2, 2, 1))

    with pytest.raises(ValueError, match="k-path is empty"):
        _xtb.run_gfn2_xtb_bandpath(system, params, [], reference_mesh)


def _exhausted_scc_options():
    """SCC budget too small for graphene to converge in."""
    options = _xtb.XTBSccOptions()
    options.max_iter = 1
    options.auto_stabilize = False
    return options


def test_gamma_mesh_nonconvergence_raises_by_default():
    """A budget-exhausted GFN2 k-route SCF must be loud (issue #342).

    Callers that screen on "did not raise" must never receive a record whose
    ``converged`` flag is False from the default path.
    """
    system = _graphene_2c()
    params = load_gfn2_params()

    with pytest.raises(
        _xtb.SCFNonConvergenceError,
        match="did not converge after 1 iterations",
    ) as excinfo:
        _xtb.run_gfn2_xtb_kpoints(
            system,
            params,
            bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0]),
            _exhausted_scc_options(),
        )
    assert "allow_unconverged" in str(excinfo.value)


def test_gamma_mesh_nonconvergence_error_is_catchable_runtime_error():
    """Existing ``except RuntimeError`` consumers keep catching (issue #342)."""
    assert issubclass(_xtb.SCFNonConvergenceError, RuntimeError)


def test_gamma_mesh_nonconvergence_has_no_numeric_energy():
    """Flag-ignoring consumers must not read a plausible zero energy.

    Receiving the unconverged record at all now takes the explicit
    ``allow_unconverged=True`` diagnostics opt-in (issue #342); the record
    keeps ``converged=False`` and NaN energies so it cannot pass for a
    clean success even then.
    """
    system = _graphene_2c()
    params = load_gfn2_params()
    options = _exhausted_scc_options()

    result = _xtb.run_gfn2_xtb_kpoints(
        system,
        params,
        bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0]),
        options,
        allow_unconverged=True,
    )

    assert not result.converged
    assert result.n_iter == options.max_iter
    assert np.isnan(result.energy)
    assert np.isnan(result.free_energy)
