"""Focused correctness gates for native k-point DFTB routes."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import (
    Atom,
    PeriodicSystem,
    bloch_kmesh_from_lists,
    direct_lattice_cells,
    monkhorst_pack,
)
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.molecule import ANGSTROM_TO_BOHR


_CHAIN_LENGTH = 4.1
_CUTOFF = 32.0
_PRIMITIVE_ATOMS = (
    (1, np.array([0.17, 0.31, 0.0])),
    (3, np.array([1.39, -0.22, 0.0])),
)
_SI_REPEATS = 3
_SI_CUTOFF = 40.0


def _hli_chain(repeats: int = 1) -> PeriodicSystem:
    atoms = []
    for image in range(repeats):
        shift = np.array([image * _CHAIN_LENGTH, 0.0, 0.0])
        atoms.extend(Atom(z, xyz + shift) for z, xyz in _PRIMITIVE_ATOMS)

    system = PeriodicSystem()
    system.dim = 1
    system.lattice = np.diag([repeats * _CHAIN_LENGTH, 30.0, 30.0])
    system.unit_cell = atoms
    system.charge = 0
    system.multiplicity = 1
    return system


def _diamond_si(repeats: int = 1) -> PeriodicSystem:
    a = 5.43 * ANGSTROM_TO_BOHR
    primitive_lattice = np.array(
        [
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        ]
    )
    primitive_atoms = (
        np.zeros(3),
        np.full(3, a / 4.0),
    )
    atoms = []
    for image in np.ndindex((repeats, repeats, repeats)):
        shift = primitive_lattice @ np.asarray(image, dtype=float)
        atoms.extend(Atom(14, xyz + shift) for xyz in primitive_atoms)
    return PeriodicSystem(
        3,
        repeats * primitive_lattice,
        atoms,
        0,
        1,
    )


def _weighted_electron_count(kmesh, occupations) -> float:
    return sum(
        weight * float(np.asarray(occupation).sum())
        for weight, occupation in zip(kmesh.weights, occupations)
    )


def _weighted_band_energy(kmesh, result) -> float:
    return sum(
        weight * float(np.dot(occupation, energies))
        for weight, occupation, energies in zip(
            kmesh.weights,
            result.occupations_per_k,
            result.eps_per_k,
        )
    )


def _metallic_si(repeats: int = 1) -> PeriodicSystem:
    """Diamond Si with two electrons removed per primitive cell.

    Neutral diamond Si is gapped under DFTB0 since #373 and #478 (pooled
    HOMO -0.063 Ha, LUMO 0.170 Ha on 3x3x3); it was metallic only on the
    earlier basis and hopping. A charge of +2 leaves four electrons for the
    threefold valence-band top, so the 3x3x3 fill shares a 12-state Fermi
    group at 2/3, as in the #561 fixtures (#565).
    """
    system = _diamond_si(repeats)
    system.charge = 2 * repeats**3
    return system


def _assert_si_global_aufbau(kmesh, result) -> None:
    per_k_electrons = np.array(
        [
            float(np.asarray(occupation).sum())
            for occupation in result.occupations_per_k
        ]
    )
    occupations = np.concatenate(
        [np.asarray(occupation) for occupation in result.occupations_per_k]
    )
    homo = max(float(energies[result.n_occ - 1]) for energies in result.eps_per_k)
    lumo = min(float(energies[result.n_occ]) for energies in result.eps_per_k)

    # Measured: the lowest band-4 state lies 0.182 Ha below the highest
    # band-3 state, and per-point counts span 2 to 8 electrons.
    assert lumo - homo < -5.0e-5
    assert _weighted_electron_count(kmesh, result.occupations_per_k) == (
        pytest.approx(6.0, abs=1.0e-12)
    )
    assert np.ptp(per_k_electrons) > 4.0
    at_fermi = np.isclose(occupations, 2.0 / 3.0, atol=1.0e-13, rtol=0.0)
    assert np.count_nonzero(at_fermi) == 12
    fractional = (occupations > 1.0e-13) & (occupations < 2.0 - 1.0e-13)
    assert np.array_equal(fractional, at_fermi)


@pytest.fixture(scope="module")
def parameters():
    return _se.SemiempiricalParameters.dftb0_default()


def test_dftb0_complex_bloch_matches_gamma_supercell(parameters):
    primitive = _hli_chain()
    repeats = 3
    kmesh = monkhorst_pack(primitive, (repeats, 1, 1))

    kpoint = _se.run_dftb0_kpoints(primitive, parameters, kmesh, _CUTOFF)
    gamma_options = _se.PeriodicDFTB0Options()
    gamma_options.cutoff_bohr = _CUTOFF
    gamma = _se.run_dftb0_gamma(_hli_chain(repeats), parameters, gamma_options)

    assert kpoint.energy == pytest.approx(gamma.energy / repeats, abs=1.0e-10)
    assert kpoint.e_electronic == pytest.approx(
        gamma.e_electronic / repeats, abs=1.0e-10
    )
    assert kpoint.e_repulsive == pytest.approx(gamma.e_repulsive / repeats, abs=1.0e-12)


def test_dftb0_wide_gamma_supercell_folds_beyond_translation_ball(parameters):
    primitive = _hli_chain()
    repeats = 5
    supercell = _hli_chain(repeats)
    kmesh = monkhorst_pack(primitive, (repeats, 1, 1))

    # A supercell wider than the cutoff empties the |g| translation
    # ball.  Pre-#316 this silently degraded to a free-boundary
    # cluster; the gen-1 fix made it fail closed; pair-distance image
    # selection (issue #316 root cause,
    # tests/test_semiempirical_pair_image_selection.py) now retains
    # every image pair within the cutoff, so the run WORKS and folds
    # exactly onto the matched mesh.  20 bohr is the shortest cutoff at
    # which the #373 Li overlaps leave S(k) positive definite (#565).
    narrow_cutoff = 20.0
    assert len(direct_lattice_cells(supercell, narrow_cutoff)) == 1
    primitive_result = _se.run_dftb0_kpoints(
        primitive,
        parameters,
        kmesh,
        narrow_cutoff,
    )
    assert np.isfinite(primitive_result.energy)

    narrow_options = _se.PeriodicDFTB0Options()
    narrow_options.cutoff_bohr = narrow_cutoff
    narrow_gamma = _se.run_dftb0_gamma(supercell, parameters, narrow_options)
    assert narrow_gamma.n_cells > 1
    assert primitive_result.energy == pytest.approx(
        narrow_gamma.energy / repeats, abs=1.0e-10
    )

    safe_cutoff = 40.0
    kpoint = _se.run_dftb0_kpoints(primitive, parameters, kmesh, safe_cutoff)
    gamma_options = _se.PeriodicDFTB0Options()
    gamma_options.cutoff_bohr = safe_cutoff
    gamma = _se.run_dftb0_gamma(supercell, parameters, gamma_options)

    assert kpoint.energy == pytest.approx(gamma.energy / repeats, abs=1.0e-10)
    assert kpoint.e_electronic == pytest.approx(
        gamma.e_electronic / repeats, abs=1.0e-10
    )
    assert kpoint.e_repulsive == pytest.approx(gamma.e_repulsive / repeats, abs=1.0e-12)


def _rock_salt_lih() -> PeriodicSystem:
    a = 7.72  # bohr
    lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return PeriodicSystem(
        3, lattice, [Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.5 * a, 0.0, 0.0])], 0, 1
    )


@pytest.mark.parametrize("method", ["dftb0", "scc_dftb"])
def test_truncated_bloch_overlap_is_refused_with_its_remedy(parameters, method):
    """#565: the default 15-bohr pair cutoff drops Li overlaps the #373 basis
    still carries, and S(k) of rock-salt LiH is indefinite (smallest
    eigenvalue -0.105; -0.0095 at 20 bohr; positive from 25 bohr). The
    k-point drivers refuse before the SCF and name the cutoff."""
    system = _rock_salt_lih()
    kmesh = monkhorst_pack(system, (4, 4, 4))

    def run(cutoff):
        if method == "dftb0":
            return _se.run_dftb0_kpoints(system, parameters, kmesh, cutoff)
        return _se.run_scc_dftb_kpoints(system, parameters, kmesh, _se.SCCOptions(), cutoff)

    entry = "run_dftb0_kpoints" if method == "dftb0" else "run_scc_dftb_kpoints"
    with pytest.raises(RuntimeError) as excinfo:
        run(15.0)
    message = str(excinfo.value)
    assert message.startswith(entry + ": the Bloch overlap S(k)")
    assert "not positive definite" in message
    assert "cutoff_bohr = 15 bohr" in message
    assert "raise cutoff_bohr" in message
    assert np.isfinite(run(25.0).energy)


def test_dftb0_metallic_si_uses_one_global_aufbau_fill(parameters):
    primitive = _metallic_si()
    kmesh = monkhorst_pack(
        primitive,
        (_SI_REPEATS, _SI_REPEATS, _SI_REPEATS),
    )
    result = _se.run_dftb0_kpoints(
        primitive,
        parameters,
        kmesh,
        _SI_CUTOFF,
    )

    gamma_options = _se.PeriodicDFTB0Options()
    gamma_options.cutoff_bohr = _SI_CUTOFF
    gamma = _se.run_dftb0_gamma(
        _metallic_si(_SI_REPEATS),
        parameters,
        gamma_options,
    )

    _assert_si_global_aufbau(kmesh, result)
    assert result.e_electronic == pytest.approx(
        _weighted_band_energy(kmesh, result),
        abs=1.0e-13,
    )
    legacy_band_energy = sum(
        weight * 2.0 * float(np.asarray(energies)[: result.n_occ].sum())
        for weight, energies in zip(kmesh.weights, result.eps_per_k)
    )
    # Per-point Aufbau would fill three bands everywhere; the global fill
    # moves electrons between points. Measured 0.0736910666 Ha on 3x3x3.
    assert legacy_band_energy - result.e_electronic == pytest.approx(
        0.0736910666400643,
        abs=1.0e-10,
    )
    assert result.e_electronic == pytest.approx(
        gamma.e_electronic / (_SI_REPEATS**3),
        abs=1.0e-12,
    )


def _graphene_gamma_supercell(n: int, *, rows: bool = False) -> PeriodicSystem:
    """n x n graphene supercell (a = 2.46 A, 30-bohr vacuum), 2-D periodic.

    ``rows=False`` feeds the lattice vectors as COLUMNS, the PeriodicSystem
    contract; ``rows=True`` reproduces the producer bug of issue #319
    (F-TRANSPOSE), which hands the transposed matrix to the same constructor.
    """
    a = 2.46 * 1.8897261254535
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0])
    a3 = np.array([0.0, 0.0, 30.0])
    basis = (np.zeros(3), (a1 + a2) / 3.0)
    vectors = np.array([n * a1, n * a2, a3])
    lattice = vectors if rows else vectors.T
    atoms = [
        Atom(6, (i * a1 + j * a2 + site).tolist())
        for i in range(n)
        for j in range(n)
        for site in basis
    ]
    return PeriodicSystem(2, lattice, atoms, 0, 1)


def test_graphene_gamma_supercell_ladder_folds_onto_the_matching_mesh(parameters):
    """Issue #319: the 2-D repulsive lattice sum does not explode with n.

    The filed +783.76 Ha/cell at n=2 (and the non-monotone 5.42 / 1.12 /
    20.51 Ha/cell at n=3..5) came from a transposed lattice, not from the
    lattice sum: with the documented column convention E_rep per primitive
    cell is bit-identical at 0.001292104 Ha for every n.  A single n would
    miss the spike; the ladder pins n = 1..5 as the issue asked.

    The electronic energy per primitive cell is not flat in n: the Gamma
    n x n supercell samples the graphene zone on an n x n mesh, and the
    semimetal converges with it (-2.612, -2.995, -3.079, -3.115, -3.115 Ha
    since #373 and #478; it moved by under 1 mHa on the earlier basis and
    hopping).  Each rung must instead equal the primitive cell on that
    mesh, which an exploding image sum would break (#565; measured 1e-14).
    """
    primitive = _graphene_gamma_supercell(1)
    cutoff = _se.PeriodicDFTB0Options().cutoff_bohr
    repulsive = []
    for n in range(1, 6):
        result = _se.run_dftb0_gamma(_graphene_gamma_supercell(n), parameters)
        folded = _se.run_dftb0_kpoints(
            primitive, parameters, monkhorst_pack(primitive, (n, n, 1)), cutoff
        )
        repulsive.append(float(result.e_repulsive) / n**2)
        assert float(result.e_electronic) / n**2 == pytest.approx(
            folded.e_electronic, abs=1.0e-10
        )
        assert float(result.energy) / n**2 == pytest.approx(folded.energy, abs=1.0e-10)
    np.testing.assert_allclose(repulsive, repulsive[0], rtol=1.0e-12, atol=0.0)
    assert repulsive[0] == pytest.approx(0.001292104, abs=5.0e-9)

    # The same constructor with the transposed (row-fed) lattice is a sheared
    # oblique cell whose n=2 image sum is the filed explosion.  Pinning the
    # discriminator keeps the F-TRANSPOSE class attributable at a glance.
    sheared = _se.run_dftb0_gamma(_graphene_gamma_supercell(2, rows=True), parameters)
    assert float(sheared.e_repulsive) / 4.0 > 100.0


def test_dftb0_bandpath_does_not_treat_path_points_as_quadrature(parameters):
    system = _diamond_si()
    kpath = monkhorst_pack(
        system,
        (_SI_REPEATS, _SI_REPEATS, _SI_REPEATS),
    ).kpoints

    result = _se.run_dftb0_bandpath(
        system,
        parameters,
        kpath,
        _SI_CUTOFF,
    )

    for occupation in result.occupations_per_k:
        np.testing.assert_array_equal(
            occupation,
            [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        )


def test_scc_dftb_one_kpoint_matches_periodic_gamma(parameters):
    system = _hli_chain()
    cutoff = 12.0
    kmesh = monkhorst_pack(system, (1, 1, 1))

    kpoint_options = _se.SCCOptions()
    kpoint_options.max_iter = 300
    kpoint_options.conv_tol_charge = 1.0e-9
    kpoint = _se.run_scc_dftb_kpoints(system, parameters, kmesh, kpoint_options, cutoff)

    gamma_options = _se.PeriodicSCCOptions()
    gamma_options.cutoff_bohr = cutoff
    gamma_options.max_iter = kpoint_options.max_iter
    gamma_options.conv_tol_charge = kpoint_options.conv_tol_charge
    gamma = _se.run_scc_dftb_gamma(system, parameters, gamma_options)

    assert kpoint.converged
    assert gamma.converged
    assert kpoint.energy == pytest.approx(gamma.energy, abs=1.0e-12)
    assert kpoint.e_scc == pytest.approx(gamma.e_scc, abs=1.0e-12)
    np.testing.assert_allclose(kpoint.charges, gamma.charges, atol=1.0e-10)


def test_scc_dftb_weighted_complex_population_matches_supercell(parameters):
    primitive = _hli_chain()
    repeats = 3
    kmesh = monkhorst_pack(primitive, (repeats, 1, 1))

    kpoint_options = _se.SCCOptions()
    kpoint_options.max_iter = 500
    kpoint_options.conv_tol_charge = 1.0e-10
    kpoint = _se.run_scc_dftb_kpoints(
        primitive, parameters, kmesh, kpoint_options, _CUTOFF
    )

    gamma_options = _se.PeriodicSCCOptions()
    gamma_options.cutoff_bohr = _CUTOFF
    gamma_options.max_iter = kpoint_options.max_iter
    gamma_options.conv_tol_charge = kpoint_options.conv_tol_charge
    gamma = _se.run_scc_dftb_gamma(_hli_chain(repeats), parameters, gamma_options)
    gamma_charges = np.asarray(gamma.charges).reshape(repeats, 2).mean(axis=0)

    assert kpoint.converged
    assert gamma.converged
    assert kpoint.energy == pytest.approx(gamma.energy / repeats, abs=1.0e-6)
    np.testing.assert_allclose(kpoint.charges, gamma_charges, atol=1.0e-4)


def test_scc_dftb_metallic_si_density_consumes_global_occupations(parameters):
    primitive = _metallic_si()
    kmesh = monkhorst_pack(
        primitive,
        (_SI_REPEATS, _SI_REPEATS, _SI_REPEATS),
    )
    options = _se.SCCOptions()
    options.max_iter = 100
    options.conv_tol_charge = 1.0e-10
    # The two sites are equivalent, so each carries +1 and the SCC
    # potential shifts both equally: H0 - V S has the DFTB0 eigenvectors.
    # Undamped mixing reaches that fixed point on the second pass; the
    # default 0.2 needs about 100 (#565).
    options.charge_mixing = 1.0
    result = _se.run_scc_dftb_kpoints(
        primitive,
        parameters,
        kmesh,
        options,
        _SI_CUTOFF,
    )
    reference = _se.run_dftb0_kpoints(primitive, parameters, kmesh, _SI_CUTOFF)

    assert result.converged
    assert result.n_iter == 2
    _assert_si_global_aufbau(kmesh, result)
    np.testing.assert_allclose(result.charges, [1.0, 1.0], rtol=0.0, atol=1.0e-12)
    # The density is built from the global fill: the same occupations as
    # DFTB0 at every point, on a spectrum shifted uniformly by V.
    for occupation, expected in zip(result.occupations_per_k, reference.occupations_per_k):
        np.testing.assert_array_equal(np.asarray(occupation), np.asarray(expected))
    shift = np.concatenate(
        [np.asarray(e) - np.asarray(e0) for e, e0 in zip(result.eps_per_k, reference.eps_per_k)]
    )
    assert np.ptp(shift) < 1.0e-11  # measured 2.5e-13
    # e_electronic is tr(D H0): the band energy without the n_e V shift.
    assert result.e_electronic == pytest.approx(
        _weighted_band_energy(kmesh, result) - 6.0 * float(shift.mean()),
        abs=1.0e-12,
    )
    assert result.e_electronic == pytest.approx(reference.e_electronic, abs=1.0e-12)


@pytest.mark.parametrize(
    ("kpoints", "weights", "message"),
    [
        ([], [], "k-point mesh is empty"),
        ([[0.0, 0.0, 0.0]], [0.5], "weights must sum to 1"),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [0.50000000002, 0.50000000002],
            "weights must sum to 1",
        ),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [0.49999999998, 0.49999999998],
            "weights must sum to 1",
        ),
        ([[0.0, 0.0, 0.0]], [-1.0], "weights must be finite and nonnegative"),
    ],
)
@pytest.mark.parametrize("method", ["dftb0", "scc_dftb"])
def test_invalid_kmesh_fails_before_lattice_build(
    parameters, kpoints, weights, message, method
):
    system = _hli_chain()
    kmesh = bloch_kmesh_from_lists(kpoints, weights)

    with pytest.raises(ValueError, match=message):
        if method == "dftb0":
            _se.run_dftb0_kpoints(system, parameters, kmesh)
        else:
            _se.run_scc_dftb_kpoints(system, parameters, kmesh)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be >= 1"),
        ("conv_tol_charge", 0.0, "conv_tol_charge must be finite and > 0"),
        ("charge_mixing", 0.0, "charge_mixing must be finite and in"),
        ("charge_mixing", np.nan, "charge_mixing must be finite and in"),
    ],
)
def test_invalid_scc_control_fails_before_lattice_build(
    parameters, attribute, value, message
):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (1, 1, 1))
    options = _se.SCCOptions()
    setattr(options, attribute, value)

    with pytest.raises(ValueError, match=message):
        _se.run_scc_dftb_kpoints(system, parameters, kmesh, options)


def _exhausted_scc_options():
    """SCC budget too small for the H-Li charge transfer to converge."""
    options = _se.SCCOptions()
    options.max_iter = 1
    options.conv_tol_charge = 1.0e-12
    return options


def test_scc_dftb_kpoints_nonconvergence_raises_by_default(parameters):
    """A budget-exhausted full-k SCC SCF must be loud (issue #342).

    The rp246 harvest admitted 6,877 of 40,486 rows whose ``converged``
    flag was False because the k-route returned instead of raising and the
    caller screened on "did not raise".  The route now raises a specific,
    catchable error naming the iteration budget by default.
    """
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (2, 1, 1))

    with pytest.raises(
        _se.SCFNonConvergenceError,
        match="did not converge after 1 iterations",
    ) as excinfo:
        _se.run_scc_dftb_kpoints(
            system, parameters, kmesh, _exhausted_scc_options(), _CUTOFF
        )
    assert "allow_unconverged" in str(excinfo.value)


def test_scc_dftb_kpoints_nonconvergence_error_is_catchable_runtime_error():
    """Existing ``except RuntimeError`` consumers keep catching (issue #342)."""
    assert issubclass(_se.SCFNonConvergenceError, RuntimeError)


def test_scc_dftb_kpoints_allow_unconverged_returns_flagged_result(parameters):
    """Diagnostics consumers opt in explicitly and get the honest flag.

    The native return-object contract survives behind the explicit
    ``allow_unconverged=True`` opt-in (issue #342): the record still says
    ``converged=False`` so a flag-reading consumer cannot mistake it for a
    clean success.
    """
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (2, 1, 1))

    result = _se.run_scc_dftb_kpoints(
        system,
        parameters,
        kmesh,
        _exhausted_scc_options(),
        _CUTOFF,
        allow_unconverged=True,
    )

    assert not result.converged
    assert result.n_iter == 1


def test_dftb0_kpoints_has_no_silent_nonconvergence_channel(parameters):
    """DFTB0 is non-self-consistent: no ``converged`` flag exists (issue #342).

    The DFTB0 k-route completes in a single diagonalization, so it must not
    carry a convergence flag a consumer could find set to False under an
    ok-looking return.
    """
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (2, 1, 1))

    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    assert not hasattr(result, "converged")
    assert np.isfinite(result.energy)
