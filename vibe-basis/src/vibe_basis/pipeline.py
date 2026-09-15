"""Cohesive-energy pipeline: relax -> single point -> ZPE -> atoms -> assemble.

The cohesive (atomization) energy of a solid is what this whole
endeavour optimizes against, and it is not one calculation. It is a
sequence, each stage of which can be skipped, cached, or fail
independently:

1. **Relax** the cell + internal coordinates at a working functional,
   so the basis is judged at *its own* equilibrium geometry rather than
   at one some other basis produced.
2. **Single point**, optionally at a different (hybrid) functional, on
   the relaxed geometry.
3. **Zero-point energy** of the solid, so the number is comparable to a
   physical cohesive energy rather than a static-lattice one.
4. **Isolated atoms**, one per distinct element, spin-correct and
   shared across every system in the campaign.
5. **Assemble** into kJ/mol per formula unit.

Sign convention, matching ``vibeqc.atomization.AtomizationResult``:

    E_coh = SUM_atoms E_atom  -  E_bulk_per_formula_unit  -  ZPE_per_fu

positive for a bound solid. ZPE *reduces* the cohesive energy: the
solid's true ground state sits ZPE above its static-lattice minimum,
while free atoms have no vibrations to correct.

Engine-neutrality
-----------------
There is one :class:`~vibe_basis.engine.EnergyEngine` seam and one
concrete pipeline, not a pipeline subclass per program. Which engine
runs is a constructor argument, so the same object composes vibe-qc
(primary), CRYSTAL23 or GPAW without a branch anywhere in this module.
That is what makes the M0 gate meaningful: run the *identical*
pipeline on two engines and any disagreement is the engines', not the
orchestration's.

Partial results are first-class
-------------------------------
A stage the engine cannot do (no relaxer, no phonons) is **recorded as
skipped with a reason**, not silently defaulted. A pipeline that
returned the input geometry for a missing relaxation, or zero for a
missing ZPE, would produce a number that looks finished and is not; the
distinction between "static-lattice cohesive energy" and "ZPE-corrected
cohesive energy" is tens of kJ/mol and exactly the size of the effects
being measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .cache import EnergyCache, atom_key, crystal_key
from .engine import EnergyEngine, EngineEnergy

#: Hartree -> kJ/mol. CODATA-consistent with vibe-qc's
#: ``atomization._HARTREE_TO_KCAL = 627.509474063`` via the
#: thermochemical calorie (627.509474063 * 4.184). Defined here rather
#: than imported because vibe-basis must not import vibe-qc; the two
#: are pinned equal by ``tests/test_pipeline.py``.
HARTREE_TO_KJ_PER_MOL = 2625.4996394798254


class PipelineError(RuntimeError):
    """The pipeline was asked for something it cannot do."""


# ---------------------------------------------------------------------------
# Stage bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageRecord:
    """What happened in one stage.

    ``status`` is one of:

    * ``"ok"``      -- ran and succeeded.
    * ``"skipped"`` -- deliberately not run; ``reason`` says why (the
      engine lacks the capability, or the caller disabled the stage).
      **Not** a failure, but it does change what the final number means.
    * ``"failed"``  -- ran and did not converge; ``reason`` carries the
      engine's failure mode.
    * ``"cached"``  -- served from the cache, no engine call.
    """

    name: str
    status: str
    reason: Optional[str] = None
    energy: Optional[float] = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ran(self) -> bool:
        return self.status in ("ok", "cached")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohesiveResult:
    """A cohesive energy and everything needed to trust or debug it.

    Attributes
    ----------
    system
        System identifier (``Structure.name``).
    ok
        ``True`` iff a cohesive energy was produced. A ``True`` here
        does **not** mean every stage ran -- check
        :attr:`zero_point_applied` and :attr:`stages` for what the
        number actually includes.
    cohesive_kj_per_mol
        E_coh per formula unit in kJ/mol, positive for a bound solid.
        ``None`` when :attr:`ok` is ``False``.
    cohesive_hartree
        The same quantity in Hartree per formula unit.
    e_bulk
        Total energy of the unit cell, Hartree.
    e_atoms_sum
        Sum of free-atom energies over one formula unit, Hartree.
    zero_point
        ZPE per formula unit in Hartree (positive), or ``None`` if the
        stage did not run.
    zero_point_applied
        Whether :attr:`cohesive_kj_per_mol` includes the ZPE
        correction. **Read this before comparing to a reference**: a
        static-lattice number and a ZPE-corrected one are different
        quantities, and the gap is the size of the effects being
        measured.
    counterpoise_applied
        Whether the free atoms were computed in the host's ghost basis
        (``ATOMBSSE``) rather than bare. The published pob values are
        counterpoise corrected, so a number compared against them must
        be too; the difference is the basis-set superposition error,
        which for an incomplete basis is exactly the quantity a
        basis-optimisation campaign is trying to reduce.
    formula_units
        Formula units in the unit cell that :attr:`e_bulk` describes.
    atomic_energies
        Z -> free-atom energy in Hartree.
    engine, engine_version
        Which engine produced the numbers.
    method, basis_digest
        Level of theory and a hash of the basis the run used.
    stages
        Per-stage records, in execution order.
    failure_mode
        Why :attr:`ok` is ``False``.
    """

    system: str
    ok: bool
    engine: str
    method: str
    formula_units: int
    cohesive_kj_per_mol: Optional[float] = None
    cohesive_hartree: Optional[float] = None
    e_bulk: Optional[float] = None
    e_atoms_sum: Optional[float] = None
    zero_point: Optional[float] = None
    zero_point_applied: bool = False
    counterpoise_applied: bool = False
    atomic_energies: dict[int, float] = field(default_factory=dict)
    engine_version: Optional[str] = None
    basis_digest: Optional[str] = None
    stages: tuple[StageRecord, ...] = ()
    failure_mode: Optional[str] = None

    def summary(self) -> str:
        """One-line human summary, honest about what is included."""
        if not self.ok:
            return (
                f"{self.system}: FAILED ({self.failure_mode}) "
                f"[{self.engine}/{self.method}]"
            )
        kind = "with ZPE" if self.zero_point_applied else "static lattice"
        return (
            f"{self.system}: {self.cohesive_kj_per_mol:.1f} kJ/mol per f.u. "
            f"({kind}) [{self.engine}/{self.method}]"
        )

    @classmethod
    def failed(
        cls,
        system: str,
        engine: str,
        method: str,
        failure_mode: str,
        *,
        formula_units: int = 1,
        stages: Sequence[StageRecord] = (),
        **kw: Any,
    ) -> "CohesiveResult":
        return cls(
            system=system,
            ok=False,
            engine=engine,
            method=method,
            formula_units=formula_units,
            failure_mode=failure_mode,
            stages=tuple(stages),
            **kw,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def formula_units(unit_cell: Iterable[Any]) -> int:
    """Formula units in a unit cell, from its element counts.

    The greatest common divisor of the per-element counts: rocksalt MgO
    with 4 Mg + 4 O is 4 f.u., fluorite CaF2 with 4 Ca + 8 F is 4 f.u.,
    corundum Al2O3 with 12 Al + 18 O is 6 f.u.

    This is a *stoichiometric* reading of the cell and is exactly right
    for the ordered binary and ternary solids in the pob test sets. It
    would be wrong for a cell whose formula is not the reduced one --
    a defect supercell, or a solid solution -- so those must pass
    ``formula_units=`` explicitly rather than rely on this.
    """
    counts: dict[int, int] = {}
    for atom in unit_cell:
        Z = atom.Z if hasattr(atom, "Z") else int(atom)
        counts[Z] = counts.get(Z, 0) + 1
    if not counts:
        raise PipelineError("cannot count formula units of an empty unit cell")
    n = 0
    for c in counts.values():
        n = math.gcd(n, c)
    return n


def stoichiometry(unit_cell: Iterable[Any], n_formula_units: int) -> dict[int, int]:
    """Z -> count **per formula unit**."""
    counts: dict[int, int] = {}
    for atom in unit_cell:
        Z = atom.Z if hasattr(atom, "Z") else int(atom)
        counts[Z] = counts.get(Z, 0) + 1
    per_fu = {}
    for Z, c in counts.items():
        if c % n_formula_units:
            raise PipelineError(
                f"element Z={Z} appears {c} times in a cell declared to hold "
                f"{n_formula_units} formula units; not an integer count per f.u."
            )
        per_fu[Z] = c // n_formula_units
    return per_fu


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class CohesivePipeline:
    """Compose an :class:`EnergyEngine` into a cohesive energy.

    Parameters
    ----------
    engine
        The engine that does the computing. vibe-qc is the primary;
        CRYSTAL23 and GPAW are fallbacks and cross-checks.
    method
        Level of theory for the bulk single point and the atoms.
    relax_method
        Level of theory for the relaxation, when it differs -- the
        usual pattern is a cheap GGA relaxation followed by a hybrid
        single point. Defaults to *method*.
    relax
        Run the relaxation stage. Set ``False`` to evaluate at the
        geometry as given (which is what the pob reference protocol
        does, since its geometries are already the reference ones).
    zero_point
        Run the ZPE stage.
    counterpoise
        Compute the free atoms in the host crystal's ghost basis
        (``ATOMBSSE``) instead of bare. Required to compare against the
        published pob cohesive energies, which are counterpoise
        corrected. Defaults to ``False`` because it is both more
        expensive and defeats cross-system atom caching: a counterpoise
        atom is host-specific by construction.
    cache
        Shared :class:`~vibe_basis.cache.EnergyCache`. Pass the *same*
        cache across systems so free-atom energies are computed once
        for the whole campaign.
    """

    def __init__(
        self,
        engine: EnergyEngine,
        *,
        method: str = "rhf",
        relax_method: Optional[str] = None,
        relax: bool = True,
        zero_point: bool = True,
        counterpoise: bool = False,
        cache: Optional[EnergyCache] = None,
    ) -> None:
        self.engine = engine
        self.method = method
        self.relax_method = relax_method or method
        self.do_relax = relax
        self.do_zero_point = zero_point
        self.do_counterpoise = counterpoise
        self.cache = cache if cache is not None else EnergyCache()

    # -- cached engine calls -------------------------------------------

    def _crystal_energy(
        self, basis_text: str, structure: Any, method: str
    ) -> tuple[EngineEnergy, bool]:
        key = crystal_key(self.engine.name, structure.name, basis_text, method)
        hit = self.cache.get(key)
        if hit is not None:
            return hit, True
        result = self.engine.crystal_energy(basis_text, structure, method=method)
        self.cache.put(key, result)
        return result, False

    def _atom_energy(
        self, basis_text: str, Z: int, method: str, host: Any = None
    ) -> tuple[EngineEnergy, bool]:
        host_name = getattr(host, "name", None) if host is not None else None
        key = atom_key(
            self.engine.name, Z, basis_text, method, host=host_name
        )
        hit = self.cache.get(key)
        if hit is not None:
            return hit, True
        result = self.engine.atom_energy(basis_text, Z, method=method, host=host)
        self.cache.put(key, result)
        return result, False

    # -- the run --------------------------------------------------------

    def run(
        self,
        basis_text: str,
        structure: Any,
        *,
        n_formula_units: Optional[int] = None,
    ) -> CohesiveResult:
        """Compute the cohesive energy of *structure* with *basis_text*."""
        from .cache import digest as _digest

        name = getattr(structure, "name", str(structure))
        stages: list[StageRecord] = []
        n_fu = n_formula_units or formula_units(structure.unit_cell)

        def fail(mode: str, **kw: Any) -> CohesiveResult:
            return CohesiveResult.failed(
                name,
                self.engine.name,
                self.method,
                mode,
                formula_units=n_fu,
                stages=stages,
                engine_version=self.engine.version(),
                basis_digest=_digest(basis_text),
                **kw,
            )

        # --- 1. relax -------------------------------------------------
        geometry = structure
        if not self.do_relax:
            stages.append(
                StageRecord("relax", "skipped", "disabled by caller")
            )
        elif not self.engine.supports("relax"):
            stages.append(
                StageRecord(
                    "relax", "skipped", f"engine {self.engine.name!r} has no relaxer"
                )
            )
        else:
            relaxed = self.engine.relax(
                basis_text, structure, method=self.relax_method
            )
            if not relaxed.ok:
                stages.append(
                    StageRecord("relax", "failed", relaxed.failure_mode)
                )
                return fail(f"relax_{relaxed.failure_mode or 'failed'}")
            geometry = relaxed.structure
            stages.append(
                StageRecord("relax", "ok", energy=relaxed.energy, detail=relaxed.detail)
            )

        # --- 2. bulk single point -------------------------------------
        bulk, cached = self._crystal_energy(basis_text, geometry, self.method)
        stages.append(
            StageRecord(
                "single_point",
                "cached" if cached else ("ok" if bulk.ok else "failed"),
                None if bulk.ok else bulk.failure_mode,
                energy=bulk.energy_if_ok(),
                detail=dict(bulk.detail),
            )
        )
        if not bulk.ok:
            return fail(f"bulk_{bulk.failure_mode or 'failed'}")
        e_bulk = bulk.energy
        assert e_bulk is not None  # guaranteed by EngineEnergy(ok=True)

        # --- 3. zero-point --------------------------------------------
        zpe_per_fu: Optional[float] = None
        if not self.do_zero_point:
            stages.append(
                StageRecord("zero_point", "skipped", "disabled by caller")
            )
        elif not self.engine.supports("zero_point"):
            stages.append(
                StageRecord(
                    "zero_point",
                    "skipped",
                    f"engine {self.engine.name!r} has no phonons",
                )
            )
        else:
            zpe = self.engine.zero_point_energy(
                basis_text, geometry, method=self.method
            )
            if not zpe.ok:
                # A failed ZPE does not sink the run: a static-lattice
                # cohesive energy is still a usable, well-defined
                # number, and `zero_point_applied=False` says so. What
                # would be wrong is quietly calling it ZPE-corrected.
                stages.append(
                    StageRecord("zero_point", "failed", zpe.failure_mode)
                )
            else:
                assert zpe.energy is not None
                zpe_per_fu = zpe.energy / n_fu
                stages.append(
                    StageRecord(
                        "zero_point", "ok", energy=zpe.energy, detail=dict(zpe.detail)
                    )
                )

        # --- 4. free atoms --------------------------------------------
        host = None
        if not self.do_counterpoise:
            stages.append(
                StageRecord("counterpoise", "skipped", "disabled by caller")
            )
        elif not self.engine.supports("counterpoise"):
            stages.append(
                StageRecord(
                    "counterpoise",
                    "skipped",
                    f"engine {self.engine.name!r} cannot do counterpoise",
                )
            )
        else:
            host = geometry
            stages.append(StageRecord("counterpoise", "ok"))

        per_fu = stoichiometry(geometry.unit_cell, n_fu)
        atomic: dict[int, float] = {}
        for Z in sorted(per_fu):
            atom, cached = self._atom_energy(basis_text, Z, self.method, host)
            stages.append(
                StageRecord(
                    f"atom_Z{Z}",
                    "cached" if cached else ("ok" if atom.ok else "failed"),
                    None if atom.ok else atom.failure_mode,
                    energy=atom.energy_if_ok(),
                    detail=dict(atom.detail),
                )
            )
            if not atom.ok:
                return fail(f"atom_Z{Z}_{atom.failure_mode or 'failed'}")
            assert atom.energy is not None
            atomic[Z] = atom.energy

        # --- 5. assemble ----------------------------------------------
        e_atoms_sum = sum(atomic[Z] * n for Z, n in per_fu.items())
        e_bulk_per_fu = e_bulk / n_fu
        cohesive = e_atoms_sum - e_bulk_per_fu
        if zpe_per_fu is not None:
            cohesive -= zpe_per_fu

        return CohesiveResult(
            system=name,
            ok=True,
            engine=self.engine.name,
            engine_version=self.engine.version(),
            method=self.method,
            formula_units=n_fu,
            cohesive_hartree=cohesive,
            cohesive_kj_per_mol=cohesive * HARTREE_TO_KJ_PER_MOL,
            e_bulk=e_bulk,
            e_atoms_sum=e_atoms_sum,
            zero_point=zpe_per_fu,
            zero_point_applied=zpe_per_fu is not None,
            counterpoise_applied=host is not None,
            atomic_energies=atomic,
            basis_digest=_digest(basis_text),
            stages=tuple(stages),
        )
