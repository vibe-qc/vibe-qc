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


def _mgo100_l3_metallic_row():
    """The one issue-302 row that still returns with no diagnostic set."""
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


def test_mgo100_l3_metallic_row_is_accepted_with_a_fractional_frontier():
    """Pin the accepted state itself, independently of how it is reported.

    Every number here was measured on main at 0.15.159 and is what makes this
    row the live half of issue 302: the route accepts it, the gap guard does
    not fire on it, its frontier is equally and fractionally occupied, and one
    sublattice carries a gross Mulliken population below zero.
    """
    molecule, result = _mgo100_l3_metallic_row()

    assert result.converged is True
    assert result.n_iter == 92
    assert result.homo_lumo_gap == pytest.approx(2.441871e-08, rel=1.0e-5)

    # The gap guard does not fire: the gap is 2.4x the absolute epsilon, so
    # by that test alone the row is an ordinary gapped row.
    assert result.homo_lumo_gap > result.finite_torus_gap_tolerance
    assert result.gap_guard_waived is False

    # It is not.  The frontier is equally occupied to six decimal places,
    # i.e. the applied occupation cannot resolve HOMO from LUMO at all.
    occupations = _applied_occupations(molecule, result)
    f_homo = occupations[result.n_occ - 1]
    f_lumo = occupations[result.n_occ]
    assert f_homo == pytest.approx(1.3370094, abs=1.0e-6)
    assert f_lumo == pytest.approx(1.3370072, abs=1.0e-6)
    assert f_homo - f_lumo < 1.0e-5
    # More than one electron sits in the Aufbau LUMO: the chemical potential
    # has passed it, which no thermally broadened insulator can do.
    assert f_lumo > 1.0

    # The filed charge symptom, reproduced exactly.  Mg carries two valence
    # electrons in the deployed parameter set, and one sublattice is past it.
    params = default_parameters()
    charges = np.asarray(result.charges)
    valence = np.array(
        [params.valence_electrons(atom.Z) for atom in molecule.atoms], float
    )
    assert charges.max() == pytest.approx(2.002335, abs=1.0e-6)
    assert np.count_nonzero(charges > valence) == 8

    # It is not an arithmetic defect.  The total is exact, the WS overlap is
    # positive definite so the issue-207 screened metric is never engaged,
    # and the density is N-representable -- so the excursion is the Mulliken
    # partitioning artefact documented in Mulliken 1955, Section 3 p. 1835,
    # not a population read against the wrong metric.
    overlap = np.asarray(result.overlap)
    density = np.asarray(result.density)
    assert np.trace(density @ overlap) == pytest.approx(
        valence.sum(), abs=1.0e-9
    )
    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    assert overlap_eigenvalues.min() > 0.87
    eigenvalues, vectors = np.linalg.eigh(overlap)
    root = vectors @ np.diag(np.sqrt(eigenvalues)) @ vectors.T
    natural = np.linalg.eigvalsh(root @ density @ root)
    assert natural.min() > -1.0e-9
    assert natural.max() < 2.0 + 1.0e-9


def test_metallic_row_records_that_its_occupation_is_not_aufbau():
    """The route must say on the record that this row is not an Aufbau row.

    Weinert and Davenport, Phys. Rev. B 45, 13709 (1992), Eqs. (8) and (10'):
    a fractional-occupation functional differs from the fixed-integer-
    occupation one by exactly the -T*S term this route subtracts, so a row
    whose applied occupations are not the Aufbau integers is not on the same
    energy surface as an Aufbau row and must not be compared with one.
    ``max_i |f_i - f_i^Aufbau|`` is zero exactly when the two coincide, which
    makes it the threshold-free quantity to record.
    """
    molecule, result = _mgo100_l3_metallic_row()

    assert result.aufbau_occupation is False
    assert result.aufbau_occupation_deviation == pytest.approx(
        _aufbau_deviation(molecule, result), abs=1.0e-9
    )
    # A deviation at or above 1 means the chemical potential has reached the
    # Aufbau LUMO: the frontier is unresolved, not merely thermally broadened.
    assert result.aufbau_occupation_deviation > 1.0
    assert result.aufbau_occupation_deviation == pytest.approx(
        1.337007, abs=1.0e-5
    )

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

    This is the discrimination the record has to get right: the HLi chain is
    an ordinary gapped cell (gap 1.8e-2 Ha, 60x the smearing width), so it is
    *not* the pathology of the slab above -- but it is still evaluated with
    fractional occupations, so it is still not comparable with an Aufbau row.
    Its deviation is 0.31, well below the slab's 1.34.
    """
    molecule, topology = _hli_chain(2)

    result = run_scc_dftb_seccm(
        molecule, topology, use_diis=True, electronic_temperature=0.005
    )

    assert result.converged is True
    assert result.gap_guard_waived is False
    assert result.aufbau_occupation is False
    assert result.aufbau_occupation_deviation == pytest.approx(
        0.3113005, abs=1.0e-6
    )
    # Strictly below 1: the chemical potential is still inside the gap, so
    # this row is broadened, not metallic.
    assert result.aufbau_occupation_deviation < 1.0


def test_zero_temperature_rows_are_exactly_aufbau():
    """Negative control: the same route with the smearing feature off.

    The T = 0 branch fills the Aufbau manifold by construction, so the
    deviation is exactly zero -- asserted exactly, not within a band -- and
    the reported energies are unchanged by this issue's record fields.
    """
    molecule, topology = _hli_chain(2)
    chain = run_scc_dftb_seccm(molecule, topology, use_diis=True)

    assert chain.converged is True
    assert chain.smearing_temperature == 0.0
    assert chain.aufbau_occupation is True
    assert chain.aufbau_occupation_deviation == 0.0
    assert np.asarray(chain.occupations)[: chain.n_occ].tolist() == [2.0] * chain.n_occ
    assert np.asarray(chain.occupations)[chain.n_occ :].tolist() == [0.0] * (
        np.asarray(chain.mo_energies).size - chain.n_occ
    )
    # Bit-identical to the value this route returned before the record grew.
    assert chain.energy == pytest.approx(-0.399870251653, abs=1.0e-12)

    slab_molecule, slab_topology = _rocksalt100_slab(4.212, 2)
    with pytest.warns(DFTB0RepulsivePlaceholderWarning, match=_PLACEHOLDER):
        slab = run_scc_dftb_seccm(slab_molecule, slab_topology, max_iter=3000)

    assert slab.converged is True
    assert slab.n_iter == 19
    assert slab.aufbau_occupation is True
    assert slab.aufbau_occupation_deviation == 0.0
    assert slab.energy == pytest.approx(-7.2616647227738103, abs=1.0e-12)
    # Bit-identical to the frontier gap that
    # tests/test_ccm_semiempirical.py::
    # test_scc_dftb_seccm_genuine_b1_mgo100_separates_repulsive_limit
    # pinned on this same slab before this issue's record fields existed:
    # the accepted state is untouched, only its self-description grew.
    assert slab.homo_lumo_gap == pytest.approx(
        0.00428369951148803, abs=1.0e-16
    )
