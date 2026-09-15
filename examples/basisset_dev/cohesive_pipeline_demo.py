"""Cohesive-energy pipeline, end to end, with no SCF program required.

    .venv/bin/python examples/basisset_dev/cohesive_pipeline_demo.py

Runs the real :class:`~vibe_basis.pipeline.CohesivePipeline` against a
*scripted* engine that returns fixed energies instead of computing
them. Everything except the SCF itself is the production code path: the
stage sequencing, the skipped-stage bookkeeping, the formula-unit
arithmetic, the unit conversion, the shared free-atom cache, and the
resume journal.

Why a fake engine
-----------------
The point of the demo is the *orchestration*, and orchestration is
exactly what you cannot see when a real run takes minutes per point on a
remote queue. Substituting known energies makes every number on screen
checkable by hand, and it means this file runs on a laptop with neither
CRYSTAL installed nor vibe-qc's native core built.

Swapping in a real engine is a one-line change, which is the design
claim being demonstrated::

    from vibe_basis.engines import Crystal23Engine
    from vibe_basis.transports.vq import VqTransport
    engine = Crystal23Engine(transport=VqTransport(host="compute-reference"))

    # or, the primary engine:
    from vibeqc.basis_optimization.calculators import VibeQcEngine
    engine = VibeQcEngine()
    pipeline = CohesivePipeline(engine, relax=False)  # static lattice

``VibeQcEngine`` deliberately does not advertise relaxation while BIPOLE
variable-cell optimization is fail-closed. Select another engine with a
certified cell optimizer if the campaign requires relaxation.

See ``docs/user_guide/cohesive_energies.md`` for the concepts and
``vibe-basis/ROADMAP.md`` for where this sits in the plan.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from vibe_basis.cache import EnergyCache
from vibe_basis.engine import EnergyEngine, EngineEnergy, RelaxedStructure
from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL, CohesivePipeline

# ---------------------------------------------------------------------------
# A minimal Structure stand-in.
#
# The real one is vibe_basis.io.structures.Structure; the pipeline only
# needs `.name` and `.unit_cell`, so this keeps the demo self-contained.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    Z: int


@dataclass(frozen=True)
class Cell:
    name: str
    unit_cell: tuple[Atom, ...]


# Conventional rocksalt cells: 4 cations + 4 anions, so 4 formula units.
MGO = Cell("MgO", tuple([Atom(12)] * 4 + [Atom(8)] * 4))
CAO = Cell("CaO", tuple([Atom(20)] * 4 + [Atom(8)] * 4))

# Energies picked so every printed number is checkable by hand:
#   E_bulk(MgO) = -400.0 Ha for 4 f.u.  ->  -100.0 per f.u.
#   E(Mg) = -50.0, E(O) = -25.0         ->   -75.0 per f.u.
#   E_coh = -75.0 - (-100.0)            ->    25.0 Ha per f.u.
BULK = {"MgO": -400.0, "CaO": -800.0}
ATOMS = {8: -25.0, 12: -50.0, 20: -150.0}


class ScriptedEngine(EnergyEngine):
    """Returns fixed energies and counts how often it was asked."""

    name = "scripted"

    def __init__(self, *, with_zpe: bool = False) -> None:
        # Declaring a capability is what makes the pipeline run that
        # stage. An engine that does not declare it gets a recorded
        # skipped stage rather than a silently defaulted value.
        self.capabilities = frozenset({"relax", "zero_point"} if with_zpe else {"relax"})
        self.calls = 0

    def version(self) -> str:
        return "1.0-scripted"

    def crystal_energy(self, basis_text, structure, method="rhf"):
        self.calls += 1
        return EngineEnergy(energy=BULK[structure.name], ok=True, engine=self.name)

    def atom_energy(self, basis_text, Z, method="rhf", *, host=None):
        self.calls += 1
        return EngineEnergy(energy=ATOMS[Z], ok=True, engine=self.name)

    def relax(self, basis_text, structure, method="rhf"):
        self.calls += 1
        return RelaxedStructure(structure=structure, ok=True, engine=self.name)

    def zero_point_energy(self, basis_text, structure, method="rhf"):
        self.calls += 1
        # 0.4 Ha for the whole 4-formula-unit cell.
        return EngineEnergy(energy=0.4, ok=True, engine=self.name)


def show(result) -> None:
    print(f"  {result.summary()}")
    print(f"    formula units in cell : {result.formula_units}")
    print(f"    E_bulk (cell)         : {result.e_bulk:.4f} Ha")
    print(f"    SUM E_atom (per f.u.) : {result.e_atoms_sum:.4f} Ha")
    if result.zero_point is not None:
        print(f"    ZPE (per f.u.)        : {result.zero_point:.4f} Ha")
    print(f"    E_coh                 : {result.cohesive_hartree:.4f} Ha "
          f"= {result.cohesive_kj_per_mol:.1f} kJ/mol")
    print(f"    zero_point_applied    : {result.zero_point_applied}")
    for stage in result.stages:
        note = f"  ({stage.reason})" if stage.reason else ""
        print(f"      {stage.name:<14s} {stage.status}{note}")


def main() -> None:
    basis = "<inline basis text for Mg, Ca, O>"

    print(__doc__.strip().splitlines()[0])
    print()
    print("NOTE: the scripted energies below are round numbers chosen so the")
    print("arithmetic is checkable by hand. They are NOT physical, so the")
    print("cohesive energies printed are far too large. Section 5 shows the")
    print("real reference values for scale.")
    print()

    # -- 1. static lattice: no phonons declared -------------------------
    print("1. Engine without phonons -> static-lattice cohesive energy")
    engine = ScriptedEngine(with_zpe=False)
    pipe = CohesivePipeline(engine, method="pbe", relax=True, zero_point=True)
    static = pipe.run(basis, MGO)
    show(static)
    print()

    # -- 2. same engine, phonons available ------------------------------
    print("2. Engine with phonons -> ZPE-corrected, and it says so")
    engine = ScriptedEngine(with_zpe=True)
    pipe = CohesivePipeline(engine, method="pbe", relax=True, zero_point=True)
    corrected = pipe.run(basis, MGO)
    show(corrected)
    print()

    delta = static.cohesive_kj_per_mol - corrected.cohesive_kj_per_mol
    print(f"   Static minus ZPE-corrected: {delta:.1f} kJ/mol.")
    print("   Comparing one to a reference computed as the other would look")
    print("   like physics. Hence zero_point_applied.")
    print()

    # -- 3. the shared free-atom cache ----------------------------------
    print("3. Free atoms are shared across systems")
    engine = ScriptedEngine(with_zpe=False)
    cache = EnergyCache()
    pipe = CohesivePipeline(
        engine, method="pbe", relax=False, zero_point=False, cache=cache
    )
    pipe.run(basis, MGO)
    before = engine.calls
    pipe.run(basis, CAO)
    print(f"   MgO cost {before} engine calls (bulk + Mg + O).")
    print(f"   CaO cost {engine.calls - before} more (bulk + Ca): "
          f"oxygen came from the cache.")
    print(f"   cache: {cache.stats}")
    print()

    # -- 4. resume ------------------------------------------------------
    print("4. A killed campaign resumes from its journal")
    with tempfile.TemporaryDirectory() as tmp:
        journal = Path(tmp) / "campaign.jsonl"

        first_engine = ScriptedEngine(with_zpe=False)
        CohesivePipeline(
            first_engine, method="pbe", relax=False, zero_point=False,
            cache=EnergyCache(journal),
        ).run(basis, MGO)
        print(f"   first run : {first_engine.calls} engine calls, "
              f"journal has {len(journal.read_text().splitlines())} entries")

        # A fresh process: new engine, new cache object, same journal.
        second_engine = ScriptedEngine(with_zpe=False)
        again = CohesivePipeline(
            second_engine, method="pbe", relax=False, zero_point=False,
            cache=EnergyCache(journal),
        ).run(basis, MGO)
        print(f"   resumed   : {second_engine.calls} engine calls, "
              f"same answer {again.cohesive_kj_per_mol:.1f} kJ/mol")
    print()

    # -- 5. scale check --------------------------------------------------
    print("5. Scale check against the real reference values")
    print("   The milestone targets, from studies/pw_limit_atomization/:")
    for label, kj in (("MgO (pob, r2SCAN)", 1044.3), ("LiF (pob, r2SCAN)", 843.7)):
        ha = kj / HARTREE_TO_KJ_PER_MOL
        print(f"     {label:<20s} {kj:7.1f} kJ/mol = {ha:.5f} Ha "
              f"= {ha * 27.211386:.2f} eV per f.u.")
    print("   Both are physically sensible for these ionic solids, which is")
    print("   how you catch a units or formula-unit error before it becomes")
    print("   a mysterious few-hundred-kJ/mol disagreement.")


if __name__ == "__main__":
    main()
