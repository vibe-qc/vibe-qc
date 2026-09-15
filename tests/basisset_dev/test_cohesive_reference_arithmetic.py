"""M0 gate 1a: the pipeline reproduces published pob cohesive energies.

Gate 1 of milestone M0 asks whether `CohesivePipeline` reproduces the
`E_pob` column of the cohesive-energy reference set. That splits in two:

* **1a (this file)** -- given the reference calculation's own total
  energies, does our assembly produce the published number? This
  certifies the sign convention, the Hartree-to-kJ/mol constant, the
  formula-unit division and the stage bookkeeping.
* **1b** -- do *our emitted decks* reproduce those total energies? That
  needs CRYSTAL23 runs on the queue and is not this file's job.

1a is worth separating because it is exact, free, and fails for
completely different reasons than 1b. A discrepancy here is arithmetic;
a discrepancy there is chemistry or deck construction.

Provenance
----------
Energies below are transcribed from the reference set at
``~/gitlab/references-cohesive/<compound>/``, which is the working
directory behind the pob basis-set papers. That tree is outside this
repository and is not present on every machine, so the numbers are
pinned here as constants rather than read at runtime, per CLAUDE.md § 8
("the published target value goes immediately at the constant").

The reference protocol, read off the decks themselves:

* Free atoms use ``SYMMREMO`` + ``SPIN`` (or ``UHF``) + ``SPINLOCK``,
  i.e. **aspherical and spin-polarised**, which is what
  ``vibe_basis.backends.crystal_atom`` emits as of 0.5.0.
* Free atoms additionally use ``ATOMBSSE``, i.e. they are **counterpoise
  corrected** in the ghost basis of the crystal. Our emitter does not do
  this yet; see the module-level note below.
* Cohesive energies are **static lattice**. The published comparison
  corrects *experiment upward* by ZPE and thermal terms rather than
  correcting theory downward -- ``references-cohesive/LiF/TE.dat`` shows
  the AgCl worked example, ``E_exp = 533.2 (NIST) + 3.55 (E0) + 11.23
  (ET) = 548.0``. So a pipeline run reproducing `E_pob` must have
  ``zero_point_applied == False``.
* Sign and unit convention, from the same file's worked example:
  ``(E_atom_1 + E_atom_2 - E_bulk) * 96.485`` eV to kJ/mol, i.e.
  ``SUM E_atom - E_bulk``, positive for a bound solid. That is the
  convention `CohesivePipeline` implements.

Known gap: BSSE
---------------
The reference atoms are counterpoise-corrected and ours are not, so
gate **1b** is expected to differ by the BSSE of the pob basis on these
solids until that is addressed. Its magnitude is **not measured here**:
every atom output in the reference tree uses ``ATOMBSSE``, so there is
no uncorrected counterpart to subtract. Do not guess it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip(
    "vibe_basis.pipeline", reason="vibe-basis is an optional [basisopt] extra"
)

from vibe_basis.engine import EnergyEngine, EngineEnergy  # noqa: E402
from vibe_basis.pipeline import CohesivePipeline  # noqa: E402

# --- Reference energies, r2SCAN / pob-TZVP-REV2, Hartree per cell -----------
#
# CRYSTAL23 total energies from the last SCF cycle of each run.
# Rocksalt in space group 225 as written in these decks: the primitive
# cell holds ONE formula unit (two atoms).
#
# LiCl  ~/gitlab/references-cohesive/LiCl/
#   bulk  LiCl_pob_r2SCAN.136128.sum
#   Li    LiCl_pob_r2SCAN_atombsse_Li.136921.sum
#   Cl    LiCl_pob_r2SCAN_atombsse_Cl.136561.sum
# Published: E_pob = 688.1 kJ/mol per f.u. (references-cohesive/r2scan.dat)
LICL_BULK = -467.83953721
LICL_ATOMS = {3: -7.47616780, 17: -460.10124136}
LICL_PUBLISHED_KJ = 688.1

# MgS  ~/gitlab/references-cohesive/MgS/
#   bulk  MgS_pob_r2SCAN.137688.sum
#   Mg    MgS_pob_r2SCAN_atombsse_Mg.137912.sum
#   S     MgS_pob_r2SCAN_atombsse_S.138144.sum
# Published: E_pob = 780.0 kJ/mol per f.u. (references-cohesive/r2scan.dat)
MGS_BULK = -598.33971449
MGS_ATOMS = {12: -200.02799141, 16: -398.01462066}
MGS_PUBLISHED_KJ = 780.0

#: Gate 1 tolerance. The published column carries one decimal, so a
#: rounding half-step is 0.05; 1.0 is the milestone's stated bar.
GATE1_TOL_KJ = 1.0


@dataclass(frozen=True)
class _Atom:
    Z: int


@dataclass(frozen=True)
class _Cell:
    name: str
    unit_cell: tuple[_Atom, ...]


class _ReferenceEngine(EnergyEngine):
    """Replays the reference calculation's own total energies.

    Not a mock of convenience: the point of gate 1a is to hold the
    engine fixed at values that are known-correct and ask whether the
    assembly agrees with the published result.
    """

    name = "reference"

    def __init__(self, bulk: float, atoms: dict[int, float]) -> None:
        self._bulk = bulk
        self._atoms = atoms

    def version(self) -> str:
        return "r2SCAN/pob-TZVP-REV2 (references-cohesive)"

    def crystal_energy(self, basis_text, structure, method="rhf"):
        return EngineEnergy(energy=self._bulk, ok=True, engine=self.name)

    def atom_energy(self, basis_text, Z, method="rhf", *, host=None):
        return EngineEnergy(energy=self._atoms[Z], ok=True, engine=self.name)


def _run(bulk: float, atoms: dict[int, float], cell: _Cell):
    engine = _ReferenceEngine(bulk, atoms)
    pipeline = CohesivePipeline(
        engine,
        method="r2scan",
        relax=False,       # the reference geometries are the target
        zero_point=False,  # E_pob is a static-lattice number
    )
    return pipeline.run("<pob-TZVP-REV2>", cell)


LICL = _Cell("LiCl", (_Atom(3), _Atom(17)))
MGS = _Cell("MgS", (_Atom(12), _Atom(16)))


def test_licl_reproduces_the_published_pob_value():
    result = _run(LICL_BULK, LICL_ATOMS, LICL)
    assert result.ok
    assert result.formula_units == 1
    delta = result.cohesive_kj_per_mol - LICL_PUBLISHED_KJ
    assert abs(delta) < GATE1_TOL_KJ, (
        f"LiCl: pipeline {result.cohesive_kj_per_mol:.2f} vs published "
        f"{LICL_PUBLISHED_KJ} kJ/mol, delta {delta:+.2f}"
    )


def test_mgs_reproduces_the_published_pob_value():
    result = _run(MGS_BULK, MGS_ATOMS, MGS)
    assert result.ok
    assert result.formula_units == 1
    delta = result.cohesive_kj_per_mol - MGS_PUBLISHED_KJ
    assert abs(delta) < GATE1_TOL_KJ, (
        f"MgS: pipeline {result.cohesive_kj_per_mol:.2f} vs published "
        f"{MGS_PUBLISHED_KJ} kJ/mol, delta {delta:+.2f}"
    )


def test_the_result_is_static_lattice_and_says_so():
    """E_pob is static lattice, so a run reproducing it must not claim
    a zero-point correction it did not apply."""
    result = _run(MGS_BULK, MGS_ATOMS, MGS)
    assert result.zero_point_applied is False
    assert result.zero_point is None
    assert "static lattice" in result.summary()


def test_sign_convention_matches_the_reference_worked_example():
    """SUM E_atom - E_bulk, positive for a bound solid.

    references-cohesive/LiF/TE.dat, AgCl worked example:
        (E_Ag + E_Cl - E_bulk) * 96.485 = 473.0 kJ/mol
    """
    result = _run(MGS_BULK, MGS_ATOMS, MGS)
    assert result.cohesive_hartree > 0
    assert result.e_atoms_sum == pytest.approx(sum(MGS_ATOMS.values()))
    assert result.e_bulk == pytest.approx(MGS_BULK)
    assert result.cohesive_hartree == pytest.approx(
        sum(MGS_ATOMS.values()) - MGS_BULK
    )


def test_ev_and_hartree_conversions_agree_with_the_reference_factor():
    """The reference set converts eV to kJ/mol with 96.485.

    Our constant is Hartree-based, so the two only have to agree after
    the Hartree-to-eV step. A mismatch here would show up as a uniform
    few-per-mille bias across every compound, which is exactly the kind
    of error that gets mistaken for a basis-set effect.
    """
    from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL

    hartree_to_ev = 27.211386245988
    ev_to_kj = 96.485  # as used in references-cohesive/LiF/TE.dat
    assert HARTREE_TO_KJ_PER_MOL / hartree_to_ev == pytest.approx(ev_to_kj, rel=2e-5)
