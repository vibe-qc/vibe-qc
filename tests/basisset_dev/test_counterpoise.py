"""Counterpoise free atoms for solids (BOC-3): CRYSTAL's ``ATOMBSSE`` in vibe-qc.

The published pob cohesive energies compute every free atom in the ghost
basis of its own crystal (Vilela Oliveira, Laun, Peintinger & Bredow,
*J. Comput. Chem.* **40**, 2364 (2019)), so a cohesive energy compared
against them must do the same (#224, rung 2b). This file pins three things:

* **The cluster.** The CRYSTAL23 manual defines the ``ATOMBSSE`` cluster as
  the atom plus the basis functions of "the NSTAR sets of neighbors, when
  their distance R from the central atom is smaller than RMAX". The rocksalt
  neighbour shells are known in closed form, the stars are symmetry orbits
  rather than distance shells, and the cluster must not depend on which cell
  describes the crystal.
* **The engine contract.** The bare atom seeds the cluster SCF, the
  provenance records the cluster, and a cluster energy above the bare one is
  a failure, not a negative superposition error.
* **The number.** With no ghosts the counterpoise atom is the bare atom to
  SCF precision; with ghosts it is lower; and the ghost calculation agrees
  with PySCF's on the same cluster.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("vibe_basis")

import vibeqc as vq  # noqa: E402
from vibe_basis.engine import EngineEnergy  # noqa: E402
from vibe_basis.io.structures import Structure, StructureAtom  # noqa: E402

from vibeqc.basis_crystal import emit_crystal, parse_crystal_atom_basis  # noqa: E402
from vibeqc.basis_optimization.calculators import (  # noqa: E402
    VibeQcEngine,
    _to_periodic_system,
)
from vibeqc.basis_optimization.counterpoise import (  # noqa: E402
    ghost_cluster,
    ghost_cluster_basis,
    ghost_cluster_molecule,
)
from vibeqc.molecule import ANGSTROM_TO_BOHR  # noqa: E402

A_MGO = 4.21  # angstrom; a stand-in, not a reference geometry
HALF = A_MGO / 2.0

# STO-3G Mg and O (Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657 (1969)),
# the same blocks tests/basisset_dev/test_vibeqc_engine.py uses.
STO3G_MG = """12 3
0 0 3 2.0 1.0
 299.2374000  0.15432897
  54.5064700  0.53532814
  14.7515800  0.44463454
0 1 3 8.0 1.0
  15.1218200 -0.09996723  0.15591627
   3.5139870  0.39951283  0.60768372
   1.1428570  0.70011547  0.39195739
0 1 3 2.0 1.0
   1.3954480 -0.21962037  0.01058760
   0.3893260  0.22559543  0.59516701
   0.1523800  0.90039843  0.46200101
"""

STO3G_O = """8 2
0 0 3 2.0 1.0
 130.7093200  0.15432897
  23.8088610  0.53532814
   6.4436083  0.44463454
0 1 3 6.0 1.0
   5.0331513 -0.09996723  0.15591627
   1.1695961  0.39951283  0.60768372
   0.3803890  0.70011547  0.39195739
"""


@pytest.fixture(scope="module")
def basis_text() -> str:
    return emit_crystal(
        [parse_crystal_atom_basis(STO3G_MG), parse_crystal_atom_basis(STO3G_O)]
    )


def rocksalt_structure(a: float = A_MGO) -> Structure:
    """Conventional eight-atom rocksalt MgO, space group Fm-3m (225)."""
    cations = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    anions = [(0.5, 0.5, 0.5), (0, 0, 0.5), (0, 0.5, 0), (0.5, 0, 0)]
    return Structure(
        name="MgO",
        formula="MgO",
        spacegroup="Fm-3m",
        crystal_system="cubic",
        a=a,
        b=a,
        c=a,
        unit_cell=tuple(StructureAtom(Z=12, fxyz=s) for s in cations)
        + tuple(StructureAtom(Z=8, fxyz=s) for s in anions),
        crystal_spacegroup=225,
        crystal_asymm_unit=(
            StructureAtom(Z=12, fxyz=(0.0, 0.0, 0.0)),
            StructureAtom(Z=8, fxyz=(0.5, 0.5, 0.5)),
        ),
    )


def rocksalt_conventional(a: float = A_MGO):
    return _to_periodic_system(rocksalt_structure(a))


def rocksalt_primitive(a: float = A_MGO):
    """The same crystal in its two-atom rhombohedral primitive cell."""
    h = 0.5 * a * ANGSTROM_TO_BOHR
    lattice = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]).T  # columns
    return vq.PeriodicSystem(
        dim=3,
        lattice=lattice,
        unit_cell=[vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, [h, h, h])],
    )


def shell_distance(k: int) -> float:
    """Distance of rocksalt shell k: (a/2) * sqrt(k) angstrom."""
    return HALF * math.sqrt(k)


# ---------------------------------------------------------------------------
# The cluster
# ---------------------------------------------------------------------------


def test_mgo_cluster_is_the_rocksalt_neighbour_shells():
    """Mg's neighbours within 5 angstrom, shell by shell.

    Rocksalt shell k sits at (a/2) sqrt(k): O 6, Mg 12, O 8, Mg 6, O 24 for
    k = 1..5. These are the reference set's MgO settings, NSTAR 10 and
    RMAX 5.0, under which RMAX binds first.
    """
    cluster = ghost_cluster(rocksalt_conventional(), 0, nstar=10, rmax_angstrom=5.0)
    assert cluster.center_Z == 12
    assert cluster.star_sizes == (6, 12, 8, 6, 24)
    assert cluster.star_Z == (8, 12, 8, 12, 8)
    assert cluster.n_ghost == 56
    np.testing.assert_allclose(
        cluster.star_distances_angstrom,
        [shell_distance(k) for k in (1, 2, 3, 4, 5)],
        atol=1e-9,
    )
    radii = np.linalg.norm(cluster.ghost_positions_bohr, axis=1) / ANGSTROM_TO_BOHR
    assert radii.max() < 5.0


def test_nstar_keeps_whole_stars():
    cluster = ghost_cluster(rocksalt_conventional(), 0, nstar=2, rmax_angstrom=10.0)
    assert cluster.star_sizes == (6, 12)
    assert cluster.n_ghost == 18
    assert ghost_cluster(rocksalt_conventional(), 0, nstar=0).n_ghost == 0


def test_stars_are_symmetry_orbits_not_distance_shells():
    """Two stars can share a distance, and CRYSTAL counts them separately.

    Shell k = 9 holds the six neighbours along the axes, (a/2)(3, 0, 0), and
    the 24 at (a/2)(2, 2, 1). They are not related by the site symmetry, so
    NSTAR counts two stars there. Within 7 angstrom that makes 11 stars and
    170 neighbours where a count of distance shells would give 10.
    """
    cluster = ghost_cluster(rocksalt_conventional(), 0, nstar=30, rmax_angstrom=7.0)
    assert cluster.n_ghost == 6 + 12 + 8 + 6 + 24 + 24 + 12 + 6 + 24 + 24 + 24
    assert cluster.n_star == 11
    at_k9 = [
        size
        for size, dist in zip(cluster.star_sizes, cluster.star_distances_angstrom)
        if abs(dist - shell_distance(9)) < 1e-6
    ]
    assert sorted(at_k9) == [6, 24]


def test_cluster_does_not_depend_on_the_cell():
    """The primitive and the conventional cell describe one crystal."""
    conventional = ghost_cluster(rocksalt_conventional(), 0, nstar=30, rmax_angstrom=7.5)
    primitive = ghost_cluster(rocksalt_primitive(), 0, nstar=30, rmax_angstrom=7.5)
    assert conventional.star_sizes == primitive.star_sizes
    assert conventional.star_Z == primitive.star_Z

    def sites(cluster):
        return sorted(
            (z, *np.round(p, 6))
            for z, p in zip(cluster.ghost_Z, cluster.ghost_positions_bohr)
        )

    assert sites(conventional) == sites(primitive)


def test_rmax_excludes_a_neighbour_at_exactly_rmax():
    """The manual keeps neighbours at a distance "smaller than RMAX"."""
    system = rocksalt_conventional()
    assert ghost_cluster(system, 0, nstar=30, rmax_angstrom=HALF).n_ghost == 0
    assert ghost_cluster(system, 0, nstar=30, rmax_angstrom=HALF + 1e-3).n_ghost == 6


def test_oxygen_centre_sees_magnesium_first():
    system = rocksalt_conventional()
    oxygen = next(i for i, a in enumerate(system.unit_cell) if a.Z == 8)
    cluster = ghost_cluster(system, oxygen, nstar=1, rmax_angstrom=10.0)
    assert cluster.center_Z == 8
    assert cluster.star_Z == (12,)
    assert cluster.n_ghost == 6


def test_cluster_refuses_bad_arguments():
    system = rocksalt_conventional()
    with pytest.raises(ValueError, match="center_index"):
        ghost_cluster(system, 99)
    with pytest.raises(ValueError, match="nstar"):
        ghost_cluster(system, 0, nstar=-1)

    class Slab:
        dim = 2

    with pytest.raises(ValueError, match="3-D"):
        ghost_cluster(Slab(), 0)


def test_ghost_molecule_has_one_real_atom_and_every_ghost_function():
    """Ghosts carry basis functions, no charge and no electrons."""
    cluster = ghost_cluster(rocksalt_conventional(), 0, nstar=2, rmax_angstrom=10.0)
    mol = ghost_cluster_molecule(cluster)
    assert [a.Z for a in mol.atoms].count(0) == cluster.n_ghost
    assert mol.atoms[0].Z == 12
    assert mol.n_electrons() == 12

    basis = ghost_cluster_basis(cluster, mol, "sto-3g")
    # STO-3G: Mg has 9 functions (1s 2s 2p 3s 3p), O has 5 (1s 2s 2p).
    assert basis.nbasis == 9 + 6 * 5 + 12 * 9


# ---------------------------------------------------------------------------
# The engine contract (SCF stubbed)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, energy, converged=True, n_iter=5):
        self.energy = energy
        self.converged = converged
        self.n_iter = n_iter


def _scripted_uhf(monkeypatch, *energies, converged=(True, True)):
    """Replace run_uhf; return the calls it saw."""
    calls = []
    script = list(zip(energies, converged))

    def run(mol, basis, opts, **kwargs):
        energy, ok = script[len(calls)]
        calls.append({"mol": mol, "basis": basis, "opts": opts, **kwargs})
        return _FakeResult(energy, converged=ok)

    monkeypatch.setattr(vq, "run_uhf", run)
    return calls


def test_engine_declares_counterpoise():
    assert VibeQcEngine().supports("counterpoise") is True


def test_truncation_defaults_match_the_crystal23_engine():
    from vibe_basis.engines.crystal23 import Crystal23Engine

    engine = VibeQcEngine()
    assert engine.bsse_nstar == Crystal23Engine.bsse_nstar
    assert engine.bsse_rmax == Crystal23Engine.bsse_rmax


def test_counterpoise_atom_is_seeded_from_the_bare_atom(monkeypatch, basis_text):
    calls = _scripted_uhf(monkeypatch, -74.0, -74.001)
    engine = VibeQcEngine(bsse_nstar=1, bsse_rmax=3.0)
    result = engine.atom_energy(basis_text, 8, host=rocksalt_structure())

    assert result.ok is True
    assert result.energy == pytest.approx(-74.001)
    bare, cluster = calls
    assert len(bare["mol"].atoms) == 1
    assert "read_from" not in bare
    assert len(cluster["mol"].atoms) == 7
    assert cluster["read_from"] is not None
    assert cluster["read_from"].energy == -74.0
    assert cluster["opts"].initial_guess == vq.InitialGuess.READ

    detail = result.detail
    assert detail["atom_reference"].startswith("counterpoise_ghost_cluster")
    assert detail["host"] == "MgO"
    assert detail["bare_energy"] == -74.0
    assert detail["bsse_hartree"] == pytest.approx(0.001)
    assert detail["cluster"]["n_ghost"] == 6
    assert detail["cluster"]["star_Z"] == [12]
    assert (detail["bsse_nstar"], detail["bsse_rmax"]) == (1, 3.0)
    assert detail["cluster_nbasis"] == 5 + 6 * 9


def test_a_cluster_above_the_bare_atom_is_a_failure(monkeypatch, basis_text):
    """A larger basis cannot raise the energy of the same state."""
    _scripted_uhf(monkeypatch, -74.0, -73.99)
    result = VibeQcEngine(bsse_nstar=1, bsse_rmax=3.0).atom_energy(
        basis_text, 8, host=rocksalt_structure()
    )
    assert result.ok is False
    assert result.failure_mode == "counterpoise_above_bare"
    assert result.energy_if_ok() is None


def test_a_bare_atom_failure_is_reported_as_such(monkeypatch, basis_text):
    _scripted_uhf(monkeypatch, -73.0, converged=(False,))
    result = VibeQcEngine(bsse_nstar=1, bsse_rmax=3.0).atom_energy(
        basis_text, 8, host=rocksalt_structure()
    )
    assert result.ok is False
    assert result.failure_mode == "bare_non_converged"


def test_a_cluster_failure_is_reported(monkeypatch, basis_text):
    _scripted_uhf(monkeypatch, -74.0, -74.1, converged=(True, False))
    result = VibeQcEngine(bsse_nstar=1, bsse_rmax=3.0).atom_energy(
        basis_text, 8, host=rocksalt_structure()
    )
    assert result.ok is False
    assert result.failure_mode == "non_converged"


def test_a_host_that_contradicts_its_declaration_is_reported(basis_text):
    """The host is marshalled with its space-group check, then reported."""
    wrong = dataclasses.replace(rocksalt_structure(), crystal_spacegroup=221)
    result = VibeQcEngine().atom_energy(basis_text, 8, host=wrong)
    assert result.ok is False
    assert result.failure_mode == "bad_host"


def test_an_element_absent_from_the_host_is_a_caller_error(basis_text):
    magnesium = Structure(
        name="Mg-sc",
        formula="Mg",
        spacegroup="Pm-3m",
        crystal_system="cubic",
        a=3.0,
        b=3.0,
        c=3.0,
        unit_cell=(StructureAtom(Z=12, fxyz=(0.0, 0.0, 0.0)),),
        crystal_spacegroup=221,
        crystal_asymm_unit=(StructureAtom(Z=12, fxyz=(0.0, 0.0, 0.0)),),
    )
    with pytest.raises(ValueError, match="Z=8"):
        VibeQcEngine().atom_energy(basis_text, 8, host=magnesium)


def test_the_pipeline_applies_counterpoise_with_vibeqc(monkeypatch, basis_text):
    """The stage that used to be skipped for this engine now runs."""
    from vibe_basis.pipeline import CohesivePipeline

    engine = VibeQcEngine(bsse_nstar=1, bsse_rmax=3.0)
    monkeypatch.setattr(
        engine,
        "crystal_energy",
        lambda *a, **k: EngineEnergy(energy=-1100.0, ok=True, engine=engine.name),
    )
    seen = []

    def atom(basis, Z, method="rhf", *, host=None):
        seen.append(host)
        return EngineEnergy(energy=-100.0 - Z, ok=True, engine=engine.name)

    monkeypatch.setattr(engine, "atom_energy", atom)
    result = CohesivePipeline(
        engine, method="rhf", relax=False, zero_point=False, counterpoise=True
    ).run(basis_text, rocksalt_structure())

    assert result.ok is True
    assert result.counterpoise_applied is True
    stage = next(s for s in result.stages if s.name == "counterpoise")
    assert stage.status == "ok"
    assert seen and all(h is not None and h.name == "MgO" for h in seen)


# ---------------------------------------------------------------------------
# Real SCF: the number
# ---------------------------------------------------------------------------


def test_a_cluster_without_ghosts_is_the_bare_atom(basis_text):
    """RMAX below the nearest neighbour leaves nothing to superpose."""
    engine = VibeQcEngine(bsse_nstar=30, bsse_rmax=1.0)
    bare = engine.atom_energy(basis_text, 12)
    cp = engine.atom_energy(basis_text, 12, host=rocksalt_structure())
    assert bare.ok and cp.ok
    assert cp.detail["cluster"]["n_ghost"] == 0
    assert cp.energy == pytest.approx(bare.energy, abs=1e-9)
    assert abs(cp.detail["bsse_hartree"]) < 1e-9


@pytest.mark.parametrize("Z", [12, 8])
def test_ghost_functions_lower_the_atom_energy(basis_text, Z):
    """The superposition error is positive.

    Mg is a closed-shell singlet (RHF); O is a triplet (UHF), which also
    exercises the read guess carrying the spin state into the cluster.
    The error is large because STO-3G is minimal: its Mg atom sits 2.6 Ha
    above the Hartree-Fock limit, and six O ghosts recover 50 mHa of that
    (PySCF agrees to 1e-6 Ha). The same cluster in 6-31G* gives 0.3 mHa.
    """
    engine = VibeQcEngine(bsse_nstar=1, bsse_rmax=3.0)
    result = engine.atom_energy(basis_text, Z, host=rocksalt_structure())
    assert result.ok, result.failure_mode
    assert result.detail["cluster"]["n_ghost"] == 6
    bsse = result.detail["bsse_hartree"]
    assert 0.0 < bsse < 0.1


def test_kohn_sham_counterpoise_atom_runs_with_ghost_grids(basis_text):
    """A functional puts integration grids on the ghost centres too."""
    engine = VibeQcEngine(bsse_nstar=1, bsse_rmax=3.0)
    result = engine.atom_energy(basis_text, 12, method="pbe", host=rocksalt_structure())
    assert result.ok, result.failure_mode
    assert result.detail["scf_method"] == "RKS"
    assert 0.0 < result.detail["bsse_hartree"] < 0.1


def test_ghost_cluster_energy_matches_pyscf():
    """An independent code computes the same atom in the same ghost basis.

    Mg (a closed-shell singlet, so the state is unique) with its six nearest
    O neighbours as ghosts, RHF/STO-3G from each code's own basis library.
    Both the cluster energy and the superposition error must agree.
    """
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")

    cluster = ghost_cluster(rocksalt_conventional(), 0, nstar=1, rmax_angstrom=3.0)
    mol = ghost_cluster_molecule(cluster)
    basis = ghost_cluster_basis(cluster, mol, "sto-3g")
    ours = vq.run_rhf(mol, basis, vq.RHFOptions())
    assert ours.converged
    bare_mol = vq.Molecule([vq.Atom(12, [0.0, 0.0, 0.0])], 0, 1)
    ours_bare = vq.run_rhf(bare_mol, vq.BasisSet(bare_mol, "sto-3g"), vq.RHFOptions())
    assert ours_bare.converged

    atoms = [["Mg", (0.0, 0.0, 0.0)]] + [
        ["ghost-O", tuple(float(c) for c in p)] for p in cluster.ghost_positions_bohr
    ]
    ref_mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis={"Mg": "sto-3g", "ghost-O": gto.basis.load("sto-3g", "O")},
        cart=False,
        verbose=0,
    )
    ref = scf.RHF(ref_mol)
    ref.conv_tol = 1e-11
    e_ref = ref.kernel()
    assert ref.converged
    ref_bare = scf.RHF(gto.M(atom=[["Mg", (0.0, 0.0, 0.0)]], basis="sto-3g", verbose=0))
    ref_bare.conv_tol = 1e-11
    e_ref_bare = ref_bare.kernel()

    assert ours.energy == pytest.approx(e_ref, abs=1e-7)
    assert ours_bare.energy - ours.energy == pytest.approx(e_ref_bare - e_ref, abs=1e-7)
