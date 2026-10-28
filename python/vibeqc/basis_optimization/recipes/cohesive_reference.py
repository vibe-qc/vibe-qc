"""M2 rung 2b: reproduce published pob cohesive energies through our own decks (#224).

:mod:`vibe_basis.io.cohesive_reference` reads what a published value was made
of: the runs the reference energy log cites, their decks and their outputs.
This recipe recomputes each of those runs from decks the vibe-basis emitters
write, and assembles the cohesive energy the way the log did. Agreement says
the pipeline reproduces the reference protocol; a difference localises to one
run.

What is copied from each reference run, because the reference set varies it:

* the bulk cell: the relaxed cell an ``OPTGEOM`` run ended on, as a single
  point (``SHRINK`` from its deck);
* each counterpoise atom: its deck's cell, ``ATOMBSSE IAT NSTAR RMAX`` and
  element, the label resolved in that deck's own primitive cell;
* the basis: the deck's inline block, or the bundled pob record of the same
  family when the deck names CRYSTAL's internal ``pob-tzvp-rev2``. The two
  agree shell for shell to CRYSTAL's 4-digit print precision across the 28
  systems of the M2 gate.

* the SCF input: the deck's convergence settings verbatim (``FMIXING``,
  ``ANDERSON``, ``ATOMSPIN``, ``SPINLOCK``, ``TOLDEE`` ...), its spin, and
  whether it removed symmetry. Without them the NaCl Na atom, run with the
  emitter's ``SYMMREMO`` + ``SPINLOCK`` and plain DIIS, collapsed from
  -162.23 Ha in its first cycle to -107.73 Ha and CRYSTAL still reported
  ``SCF ENDED - CONVERGENCE ON ENERGY``.

One departure is deliberate: a ``TOLDEE`` tighter than :data:`MAX_TOLDEE`
is capped at it. Several reference atom decks ask for 1e-14 Ha, which is
CRYSTAL's roundoff: recomputing the NaCl Na atom with its own input followed
the reference to 1e-10 Ha for 14 cycles and reached its energy within 3.3e-7
Ha by cycle 16, then wobbled at |dE| ~ 1e-13 without meeting the test. The
reference had stopped on an exact dE = 0 at cycle 17. Left running, ours
crept 1.3e-5 Ha lower, then the mixing broke down at cycle 45 and the run
ended on "TOO MANY CYCLES". The cap is recorded with the run.

Each deck is the input the run's output echoes, not the ``.d12`` beside it,
which in this set was sometimes edited after the run; each run records where
its deck came from and which settings the ``.d12`` file disagrees on.

The functional is the deck's (``r2SCAN``) and ``TOLINTEG`` the reference
set's ``9 9 9 18 54``, which is every deck's. A run whose deck is missing is
reported, not guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from vibe_basis.cache import EnergyCache, digest
from vibe_basis.engine import TRANSIENT_FAILURE_MODES
from vibe_basis.io.cohesive_reference import (
    ReferenceComponent,
    ReferenceSystem,
    conventional_cubic_a,
    final_primitive_cell,
)
from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL

from ...basis_crystal import emit_crystal
from ...basis_toolkit._element_data import SYMBOL
from .topology_scan import load_pob_atoms

__all__ = [
    "MAX_TOLDEE",
    "ReproducedRun",
    "ReproductionReport",
    "crystal23_reference_factory",
    "reproduce_reference_system",
]

#: The bundled source family for each CRYSTAL internal basis keyword.
_KEYWORD_FAMILY = {"pob-tzvp-rev2": "pob-TZVP-rev2", "pob-dzvp-rev2": "pob-DZVP-rev2"}


@dataclass
class ReproducedRun:
    """One reference run recomputed."""

    output_name: str
    role: str
    Z: Optional[int]
    reference_ha: float
    reference_status: str
    energy_ha: Optional[float] = None
    failure_mode: Optional[str] = None
    basis_source: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def delta_kj(self) -> Optional[float]:
        """Ours minus the reference's energy, kJ/mol."""
        if self.energy_ha is None:
            return None
        return (self.energy_ha - self.reference_ha) * HARTREE_TO_KJ_PER_MOL


@dataclass
class ReproductionReport:
    """A published value next to the same value from our decks."""

    system: str
    published_kj: float
    reference_kj: float
    reproduced_kj: Optional[float]
    families: tuple[str, ...]
    runs: list[ReproducedRun]

    @property
    def delta_kj(self) -> Optional[float]:
        """Reproduced minus reference cohesive energy, kJ/mol per formula unit."""
        return None if self.reproduced_kj is None else self.reproduced_kj - self.reference_kj

    def to_json(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "published_kj": self.published_kj,
            "reference_kj": self.reference_kj,
            "reproduced_kj": self.reproduced_kj,
            "delta_kj": self.delta_kj,
            "families": list(self.families),
            "runs": [
                {**run.__dict__, "delta_kj": run.delta_kj} for run in self.runs
            ],
        }


#: The tightest SCF energy tolerance (10^-n Ha) a reproduction asks for; see
#: the module docstring. 1e-10 Ha is 3e-7 kJ/mol.
MAX_TOLDEE = 10


def _cap_toldee(scf: tuple[str, ...], cap: Optional[int]) -> tuple[tuple[str, ...], Optional[int]]:
    """*scf* with a ``TOLDEE`` above *cap* lowered to it, and the value it replaced."""
    lines = list(scf)
    for i, line in enumerate(lines[:-1]):
        if line.upper() == "TOLDEE" and cap is not None:
            asked = int(lines[i + 1].split()[0])
            if asked > cap:
                lines[i + 1] = str(cap)
                return tuple(lines), asked
    return tuple(lines), None


def crystal23_reference_factory(
    transport: Any,
    *,
    crystal_binary: str = "crystal",
    invocation: str = "stdin",
    cpus: int = 1,
    wall_time_s: int = 7200,
) -> Callable[..., Any]:
    """``factory(run_dir, **run settings) -> Crystal23Engine``.

    The run settings are ``shrink``, ``optgeom``, ``bsse_nstar``,
    ``bsse_rmax``, ``scf_lines`` (the reference deck's SCF input) and, for a
    counterpoise atom, ``n_unpaired`` and ``symmremo``.
    """
    from vibe_basis.engines.crystal23 import Crystal23Engine

    def make(
        run_dir: Path, *, shrink=8, optgeom=False, bsse_nstar=30, bsse_rmax=10.0,
        scf_lines=(), n_unpaired=None, symmremo=None,
    ) -> Crystal23Engine:
        return Crystal23Engine(
            transport=transport,
            crystal_wrapper=crystal_binary,
            invocation=invocation,
            cpus=cpus,
            wall_time_s=wall_time_s,
            workdir_prefix=str(Path(run_dir) / "calc"),
            shrink=shrink,
            optgeom=optgeom,
            bsse_nstar=bsse_nstar,
            bsse_rmax=bsse_rmax,
            scf_lines=tuple(scf_lines),
            atom_n_unpaired=n_unpaired,
            atom_symmremo=symmremo,
            strict_version=True,
        )

    return make


def _basis_text(
    run: ReferenceComponent, system: ReferenceSystem, bundled: Optional[str] = None
) -> tuple[Optional[str], str]:
    s = run.settings
    if s is None:
        return None, "no deck"
    if bundled is not None:
        symbols = [SYMBOL[z] for z in system.counts]
        atoms = load_pob_atoms(bundled, symbols)
        return emit_crystal(list(atoms.values())), f"bundled {bundled} (replacing the deck's basis)"
    if s.inline_basis:
        return s.inline_basis, "deck inline block"
    family = _KEYWORD_FAMILY.get((s.basis_keyword or "").lower())
    if family is None:
        return None, f"keyword {s.basis_keyword!r} has no bundled family"
    symbols = [SYMBOL[z] for z in system.counts]
    atoms = load_pob_atoms(family, symbols)
    return emit_crystal(list(atoms.values())), f"bundled {family} (deck keyword {s.basis_keyword})"


def reproduce_reference_system(
    system: ReferenceSystem,
    engine_factory: Callable[..., Any],
    workdir: Path | str,
    *,
    record: bool = True,
    bundled_basis: Optional[dict[str, str]] = None,
    relax: frozenset[str] | set[str] = frozenset(),
    label: str = "reproduction",
    max_toldee: Optional[int] = MAX_TOLDEE,
) -> ReproductionReport:
    """Recompute every run behind ``system``'s published value; assemble it.

    ``engine_factory(run_dir, **run settings)`` returns the engine for one
    run (:func:`crystal23_reference_factory`).

    By default each run is reproduced as the reference made it. A consistent
    variant changes named runs only: ``bundled_basis`` maps an output name to
    the bundled family whose record replaces that deck's basis, and ``relax``
    names bulk runs to relax with ``OPTGEOM`` from their deck's starting cell
    instead of a single point at the cell the reference relaxed to, which a
    basis change invalidates. ``max_toldee`` caps a deck's ``TOLDEE``
    (``None`` keeps it). The report is written to
    ``<workdir>/<system>/<label>.json`` when ``record``.

    With ``record`` every finished run is also journalled in
    ``<workdir>/<system>/energies.jsonl``, keyed by the output it reproduces,
    the engine, the basis and every run setting, so a restarted driver
    reuses it; a failure that says nothing about the run (a lost transport,
    a killed job) is not journalled.
    """
    bundled_basis = bundled_basis or {}
    workdir = Path(workdir).resolve() / system.name
    cache = EnergyCache(workdir / "energies.jsonl") if record else None
    runs: list[ReproducedRun] = []
    for c in system.components:
        text, source = _basis_text(c, system, bundled_basis.get(c.output_name))
        run = ReproducedRun(
            output_name=c.output_name, role=c.role, Z=c.Z,
            reference_ha=c.energy_ha, reference_status=c.output_status,
            basis_source=source,
        )
        runs.append(run)
        if text is None or c.structure is None:
            run.failure_mode = "no_deck"
            continue
        method = (c.settings.functional or "hf").lower()
        run_dir = workdir / label / Path(c.output_name).stem
        run_dir.mkdir(parents=True, exist_ok=True)
        scf, toldee_asked = _cap_toldee(tuple(c.settings.scf), max_toldee)
        if c.role == "bulk":
            shrink = tuple(c.settings.shrink) if c.settings.shrink else 8
            structure, optgeom = c.structure, c.output_name in relax
            if optgeom:
                a0 = float(c.settings.lattice[0])
                structure = replace(structure, a=a0, b=a0, c=a0)
            run.settings = {"a": structure.a, "shrink": shrink, "method": method, "optgeom": optgeom}
            engine = engine_factory(run_dir, shrink=shrink, optgeom=optgeom, scf_lines=scf)

            def compute(engine=engine, structure=structure):
                return engine.crystal_energy(text, structure, method=method)
        else:
            iat, nstar, rmax = c.settings.atombsse
            run.settings = {
                "a": c.structure.a, "iat": iat, "nstar": nstar, "rmax": rmax, "method": method,
                "n_unpaired": c.settings.n_unpaired, "symmremo": c.settings.symmremo,
            }
            engine = engine_factory(
                run_dir, bsse_nstar=nstar, bsse_rmax=rmax, scf_lines=scf,
                n_unpaired=c.settings.n_unpaired, symmremo=c.settings.symmremo,
            )

            def compute(engine=engine, c=c, iat=iat):
                return engine.atom_energy(text, c.Z, method, host=c.structure, atom_label=iat)
        key = "r:" + digest(engine.name, c.output_name, c.role, text, {**run.settings, "scf": list(scf)})
        result = cache.get(key) if cache is not None else None
        if result is None:
            result = compute()
            if cache is not None and result.failure_mode not in TRANSIENT_FAILURE_MODES:
                cache.put(key, result)
        if c.role == "bulk" and run.settings["optgeom"] and result.ok:
            run.settings["relaxed_a"] = _relaxed_a(run_dir, structure, c.settings.spacegroup)
        if toldee_asked is not None:
            run.settings["toldee_capped_from"] = toldee_asked
        run.settings.update({
            "scf_lines": list(scf),
            "deck_source": c.deck_source,
            "deck_file_differs": list(c.deck_file_differs),
        })
        run.energy_ha = result.energy_if_ok()
        run.failure_mode = result.failure_mode
    reproduced = None
    if all(r.energy_ha is not None for r in runs):
        reproduced = system.cohesive_kj({r.output_name: r.energy_ha for r in runs})
    report = ReproductionReport(
        system=system.name,
        published_kj=system.published_kj,
        reference_kj=system.cohesive_kj(),
        reproduced_kj=reproduced,
        families=system.families(),
        runs=runs,
    )
    if record:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / f"{label}.json").write_text(json.dumps(report.to_json(), indent=2) + "\n")
    return report


def _relaxed_a(run_dir: Path, structure: Any, spacegroup: int) -> Optional[float]:
    """Conventional a of the final geometry in a fetched relaxation output."""
    for out in sorted(run_dir.glob(f"**/{structure.name}.out")):
        cell = final_primitive_cell(out.read_text(errors="replace"))
        if cell is not None:
            return conventional_cubic_a(cell, spacegroup)
    return None
