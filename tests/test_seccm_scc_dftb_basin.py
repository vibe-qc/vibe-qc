"""Physical-basin record of an accepted SCC-DFTB-SECCM fixed point (issue 302).

Issue 302 filed four rocksalt(100) slab rows that ``run_scc_dftb_seccm``
returned with ``converged = True`` on a state the filing called physically
impossible: Mulliken charges past the element's formal valence-electron count,
together with an effectively zero T = 0 HOMO-LUMO gap.

Measured on ``main`` at 0.15.159, three of those four rows are already handled.
Every T = 0 rung of the filing's retry ladder fails closed, the O/O L1 row
comes back with ``gap_guard_waived = True`` (commit 4e0b569a6, issue 422), and
the Mg/O L4 row no longer converges at all. Exactly one row still comes back
with no diagnostic set, and this module pins it:

    Mg/O rocksalt(100) L3, a = 4.000 A, unembedded, DIIS,
    electronic_temperature = 0.005 Ha

Its frontier holds 1.337 electrons in the LUMO and the same in the HOMO to six
decimal places -- an unresolved, equally occupied frontier manifold, i.e. the
same physical state as the already-flagged O/O L1 row. It escapes the gap guard
only because that guard compares the gap to an *absolute* epsilon
(``finite_torus_gap_tolerance``, 1e-8 Ha) while the scale that resolves the
occupation on the smeared branch is kT: the row's gap of 2.44e-8 Ha is 2.4x the
epsilon but 5e-6 of the smearing width.

The filed charge premise does not survive measurement and is recorded here as
refuted rather than gated. Mulliken, J. Chem. Phys. 23, 1833 (1955), Section 3,
page 1835, states that gross atomic populations ideally would never be negative
and never exceed 2.00 per closed sub-shell, but that small negative values and
slight excesses do occur; the invariant is the total, "necessarily an integer".
The Mg excursion here is -0.0023 e of gross population on a single-s-shell
atom, the total is exact, and the density is N-representable, so a raw
``|q| <= n_val`` acceptance gate has no basis in the source.

Post-#373/#478 (re-baselined under #423): the corrected basis and the
Wolfsberg-Helmholz hopping without the duplicated 1/2 make the L3 row an
ordinary gapped row at T = 0.005 Ha. The pins below describe that state and
the screened, indefinite Wigner-Seitz overlap it is solved on.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import (
    DFTB0RepulsivePlaceholderWarning,
    run_scc_dftb_seccm,
)
from vibeqc.semiempirical.parameters import default_parameters
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

# Issue 306 placeholder-repulsive warning raised by every Mg/O DFTB slab here.
_PLACEHOLDER = r"8-8, 8-12, 12-12|8-12, 12-12"

_HLI_SITES = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
_HLI_CHAIN_CELL = 6.0


def _rocksalt100_slab(a_angstrom: float, layers: int):
    """B1 rocksalt(100): neutral Tasker-I checkerboard planes, 2x2 cell.

    Same construction as ``studies/seccm-bulk3d/scan_mgo100_slab.py``: the
    2-D cyclic topology uses the primitive surface vectors (a/2, a/2, 0) and
    (-a/2, a/2, 0) with two atoms per primitive plane, and the finite z stack
    carries no translation.
    """
    from vibeqc.molecule import ANGSTROM_TO_BOHR

    t1 = np.array([a_angstrom / 2.0, a_angstrom / 2.0, 0.0])
    t2 = np.array([-a_angstrom / 2.0, a_angstrom / 2.0, 0.0])
    unlike = np.array([a_angstrom / 2.0, 0.0, 0.0])

    coords_angstrom: list[np.ndarray] = []
    atomic_numbers: list[int] = []
    for layer in range(layers):
        z_offset = np.array([0.0, 0.0, layer * a_angstrom / 2.0])
        for i in range(2):
            for j in range(2):
                home = i * t1 + j * t2 + z_offset
                if layer % 2 == 0:
                    coords_angstrom.extend((home, home + unlike))
                else:
                    coords_angstrom.extend((home + unlike, home))
                atomic_numbers.extend((12, 8))

    # Pin the crystallography rather than trusting the material name
    # (issue 471): orthogonal primitive surface vectors of area a^2/2 and
    # equal Mg/O counts in every plane.
    assert np.dot(t1, t2) == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(np.cross(t1, t2)) == pytest.approx(
        a_angstrom**2 / 2.0, abs=1.0e-12
    )
    assert atomic_numbers.count(12) == atomic_numbers.count(8)

    coords_bohr = [np.asarray(c) * ANGSTROM_TO_BOHR for c in coords_angstrom]
    topology = build_seccm_topology(
        coords_bohr,
        [np.asarray(2.0 * t) * ANGSTROM_TO_BOHR for t in (t1, t2)],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[
            np.asarray(t) * ANGSTROM_TO_BOHR for t in (t1, t2)
        ],
        replicas=(2, 2, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    molecule = Molecule(
        [
            Atom(z, c.tolist())
            for z, c in zip(atomic_numbers, coords_bohr)
        ],
        0,
        1,
    )
    return molecule, topology


def _hli_chain(molecule_replicas: int):
    """The gapped 1-D control chain used by the existing SECCM lanes."""
    primitive = np.array([_HLI_CHAIN_CELL, 0.0, 0.0])
    coords = np.vstack(
        [_HLI_SITES + cell * primitive for cell in range(molecule_replicas)]
    )
    topology = build_seccm_topology(
        coords,
        [molecule_replicas * primitive],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[primitive],
        replicas=(molecule_replicas, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    molecule = Molecule(
        [
            Atom(1 if k % 2 == 0 else 3, c.tolist())
            for k, c in enumerate(coords)
        ],
        0,
        1,
    )
    return molecule, topology


def _applied_occupations(molecule, result) -> np.ndarray:
    """Fermi-Dirac occupations implied by the record, independent of the route.

    Weinert and Davenport, Phys. Rev. B 45, 13709 (1992), Eq. (8):
        f_i = 1 / (exp(beta (eps_i - mu)) + 1)
    with mu fixed by particle-number conservation.  Reconstructed here from
    ``mo_energies`` and ``smearing_temperature`` alone so that the assertions
    below are a measurement of the accepted state, not a restatement of
    whatever the route chose to record.
    """
    energies = np.asarray(result.mo_energies)
    temperature = float(result.smearing_temperature)
    params = default_parameters()
    n_electrons = float(
        sum(params.valence_electrons(atom.Z) for atom in molecule.atoms)
    )
    if temperature <= 0.0:
        occupations = np.zeros_like(energies)
        occupations[: result.n_occ] = 2.0
        return occupations
    low, high = energies.min() - 5.0, energies.max() + 5.0
    for _ in range(300):
        mu = 0.5 * (low + high)
        occupations = 2.0 / (
            1.0 + np.exp(np.clip((energies - mu) / temperature, -500.0, 500.0))
        )
        if occupations.sum() > n_electrons:
            high = mu
        else:
            low = mu
    return occupations


def _aufbau_deviation(molecule, result) -> float:
    occupations = _applied_occupations(molecule, result)
    aufbau = np.where(
        np.arange(occupations.size) < result.n_occ, 2.0, 0.0
    )
    return float(np.abs(occupations - aufbau).max())


def _mgo100_l3_row():
    """The issue-302 MgO(100) L3 row (unembedded, DIIS, T = 0.005 Ha)."""
    molecule, topology = _rocksalt100_slab(4.000, 3)
    with pytest.warns(DFTB0RepulsivePlaceholderWarning, match=_PLACEHOLDER):
        result = run_scc_dftb_seccm(
            molecule,
            topology,
            madelung=False,
            use_diis=True,
            electronic_temperature=0.005,
            max_iter=5000,
        )
    return molecule, result


def test_mgo100_l3_row_is_a_gapped_smeared_row_on_a_screened_overlap():
    """Pin the accepted post-#478 state of the issue-302 L3 row (#423).

    Until #373/#478 this row was metallic: an equally occupied frontier with
    the chemical potential past the Aufbau LUMO and one sublattice past its
    valence count. With the corrected basis (#373) and the Wolfsberg-
    Helmholz hopping without the duplicated 1/2 (#478) it is an ordinary
    gapped row evaluated at finite temperature: the chemical potential stays
    inside a 0.019 Ha gap and no atom passes its valence count. The maintainer
    decision on #423 (2026-10-23) re-pins it as that row instead of keeping
    the strict xfail.

    The Wigner-Seitz-truncated cyclic overlap does not depend on H0 and stays
    indefinite (one mode at -0.455). Under the #151 decision the route screens
    it by canonical orthogonalization (Peintinger and Bredow, J. Comput. Chem.
    35, 839 (2014), eq. 5) and refuses only when the kept subspace cannot hold
    the occupied manifold, so the result records the screen and the state is
    checked in the kept metric. The criteria are state, occupation, spectrum
    and energy, not an iteration count or a roundoff-scale gap tolerance; the
    energy is mixer independent (Aitken reaches the same state to 3e-14 Ha).
    Measured 2026-10-25 on main ba24de70.
    """
    molecule, result = _mgo100_l3_row()

    assert result.converged is True
    assert result.energy == pytest.approx(-11.24499047116076, abs=1.0e-8)
    assert result.free_energy == result.energy
    assert result.smearing_temperature == pytest.approx(0.005, abs=0.0)

    # Gapped: the gap guard has nothing to waive.
    assert result.homo_lumo_gap == pytest.approx(0.019242666925, rel=1.0e-6)
    assert result.homo_lumo_gap > 1.0e3 * result.finite_torus_gap_tolerance
    assert result.gap_guard_waived is False

    # Thermally broadened, not metallic: the chemical potential lies inside
    # the gap, so the Aufbau LUMO holds less than one electron.
    occupations = _applied_occupations(molecule, result)
    assert occupations[result.n_occ - 1] == pytest.approx(1.8755227446, abs=1.0e-6)
    assert occupations[result.n_occ] == pytest.approx(0.4861060042, abs=1.0e-6)
    assert occupations[result.n_occ] < 1.0
    assert result.aufbau_occupation is False
    assert result.aufbau_occupation_deviation == pytest.approx(
        occupations[result.n_occ], abs=1.0e-9
    )

    # Mulliken populations: neutral cell, no atom past its valence count
    # (the filed excursion is gone), outer and middle planes distinct.
    params = default_parameters()
    charges = np.asarray(result.charges)
    valence = np.array(
        [params.valence_electrons(atom.Z) for atom in molecule.atoms], float
    )
    assert charges.sum() == pytest.approx(0.0, abs=1.0e-9)
    assert np.count_nonzero(charges > valence) == 0
    assert charges.max() == pytest.approx(1.822074, abs=1.0e-5)
    assert charges.min() == pytest.approx(-1.891986, abs=1.0e-5)

    # The indefinite WS overlap and its recorded screen.
    overlap = np.asarray(result.overlap)
    eigenvalues = np.linalg.eigvalsh(overlap)
    assert result.overlap_min_eigenvalue == pytest.approx(eigenvalues.min(), abs=0.0)
    assert result.overlap_min_eigenvalue == pytest.approx(-0.45470979756, abs=1.0e-8)
    assert result.overlap_screen_threshold == pytest.approx(1.0e-9, abs=0.0)
    assert result.overlap_screened_modes == 1
    assert result.overlap_screened_modes == np.count_nonzero(
        eigenvalues <= result.overlap_screen_threshold
    )

    # The spectrum lives in the kept subspace: n_basis - 1 orbitals,
    # orthonormal in S, occupations in [0, 2], and the density they build
    # carries the exact electron count. That is the N-representability check
    # an indefinite S still admits.
    coefficients = np.asarray(result.mo_coeffs)
    applied = np.asarray(result.occupations)
    assert coefficients.shape == (result.n_basis, result.n_basis - 1)
    assert np.asarray(result.mo_energies).size == result.n_basis - 1
    assert np.abs(
        coefficients.T @ overlap @ coefficients - np.eye(coefficients.shape[1])
    ).max() < 1.0e-10
    assert applied.min() >= 0.0
    assert applied.max() <= 2.0 + 1.0e-12
    density = np.asarray(result.density)
    assert np.abs(coefficients @ np.diag(applied) @ coefficients.T - density).max() < 1.0e-10
    assert np.trace(density @ overlap) == pytest.approx(valence.sum(), abs=1.0e-9)


def test_l3_smeared_row_records_that_its_occupation_is_not_aufbau():
    """The route must say on the record that this row is not an Aufbau row.

    Weinert and Davenport, Phys. Rev. B 45, 13709 (1992), Eqs. (8) and (10'):
    a fractional-occupation functional differs from the fixed-integer-
    occupation one by exactly the -T*S term this route subtracts, so a row
    whose applied occupations are not the Aufbau integers is not on the same
    energy surface as an Aufbau row and must not be compared with one.
    ``max_i |f_i - f_i^Aufbau|`` is zero exactly when the two coincide, which
    makes it the threshold-free quantity to record. Since #478 the L3 row is
    gapped (deviation 0.486 < 1: the chemical potential is inside the gap),
    but at T = 0.005 Ha it is still a Mermin row and must say so (#423).
    """
    molecule, result = _mgo100_l3_row()

    assert result.aufbau_occupation is False
    assert result.aufbau_occupation_deviation == pytest.approx(
        _aufbau_deviation(molecule, result), abs=1.0e-9
    )
    assert result.aufbau_occupation_deviation == pytest.approx(
        0.4861060041766214, abs=1.0e-6
    )
    assert result.aufbau_occupation_deviation < 1.0

    # The verdict must be reproducible from the record alone, the way
    # finite_torus_gap_tolerance already makes the gap verdict reproducible.
    assert result.aufbau_occupation_tolerance == pytest.approx(
        1.0e-8, abs=0.0
    )
    occupations = np.asarray(result.occupations)
    assert occupations.shape == (np.asarray(result.mo_energies).size,)
    assert occupations.sum() == pytest.approx(96.0, abs=1.0e-9)
    assert occupations == pytest.approx(
        _applied_occupations(molecule, result), abs=1.0e-8
    )


def test_ordinary_smeared_row_is_also_recorded_as_non_aufbau():
    """A thermally broadened gapped row is a Mermin row too, and says so.

    The HLi chain is an ordinary gapped cell: since #478 its gap is 0.131 Ha,
    26x the smearing width, so its LUMO holds only 4.2e-6 electrons. That is
    far from metallic, but it is still not the integer Aufbau occupation, so
    the row is still not comparable with an Aufbau row and must say so. The
    deviation is the LUMO population, set by the gap; checked as a relative
    pin together with the gap that sets it (#423).
    """
    molecule, topology = _hli_chain(2)

    result = run_scc_dftb_seccm(
        molecule, topology, use_diis=True, electronic_temperature=0.005
    )

    assert result.converged is True
    assert result.gap_guard_waived is False
    assert result.homo_lumo_gap == pytest.approx(0.13064707313, rel=1.0e-6)
    assert result.energy == pytest.approx(-0.439392810178657, abs=1.0e-8)
    assert result.aufbau_occupation is False
    assert result.aufbau_occupation_deviation == pytest.approx(
        4.237445748e-06, rel=1.0e-4
    )
    assert (
        result.aufbau_occupation_deviation
        > 100.0 * result.aufbau_occupation_tolerance
    )
    # Strictly below 1: the chemical potential is still inside the gap, so
    # this row is broadened, not metallic.
    assert result.aufbau_occupation_deviation < 1.0


def test_zero_temperature_rows_are_exactly_aufbau():
    """Negative control: the same route with the smearing feature off.

    The T = 0 branch fills the Aufbau manifold by construction, so the
    deviation is exactly zero -- asserted exactly, not within a band -- and
    the reported energies are unchanged by this issue's record fields.

    The MgO(100) slab is the T = 0 fixture. The HLi chain that held this
    role until #373 is now a frontier near-degeneracy under the corrected
    Stewart basis: its T = 0 integer-occupation SCC map has no stable fixed
    point (non-convergent even at 20k iterations), whereas the slab still
    fills its Aufbau manifold cleanly at T = 0.
    """
    slab_molecule, slab_topology = _rocksalt100_slab(4.212, 2)
    with pytest.warns(DFTB0RepulsivePlaceholderWarning, match=_PLACEHOLDER):
        slab = run_scc_dftb_seccm(slab_molecule, slab_topology, max_iter=10000)

    assert slab.converged is True
    # State, occupation, spectrum and energy criteria (#423); the iteration
    # count is a property of the mixer, not of the state, and is not pinned.
    assert slab.smearing_temperature == 0.0
    assert slab.aufbau_occupation is True
    assert slab.aufbau_occupation_deviation == 0.0
    # Re-measured after #478 (2026-10-25, main ba24de70): the default Aitken
    # and DIIS mixers reach the same state to roundoff.
    assert slab.energy == pytest.approx(-7.38649751980566, abs=1.0e-8)
    assert slab.homo_lumo_gap == pytest.approx(0.0872445071773, rel=1.0e-6)
    assert slab.homo_lumo_gap > 1.0e3 * slab.finite_torus_gap_tolerance
    charges = np.asarray(slab.charges)
    assert charges.sum() == pytest.approx(0.0, abs=1.0e-9)
    # Equivalent Mg (and O) sites carry one charge: no symmetry-broken state.
    assert np.ptp(charges[0::2]) < 1.0e-9
    assert np.ptp(charges[1::2]) < 1.0e-9
    assert charges[0] == pytest.approx(1.59294, abs=1.0e-5)
