"""M3's topology scan on an energy engine: relax every candidate, then rank the front.

:mod:`vibe_basis.topology` is the search (Tier 1): the moves, the plateau
rule and the Pareto front, with the evaluation injected. This recipe is that
evaluation for one element and shell type in one crystal, and the driver
that runs the search and records the result (#225).

* **The candidate.** A pob topology from ``basis_library/sources/<family>``,
  edited by the scan's moves. Every other element and shell stays at its
  published values.
* **The relaxation.** The exponents of the element's uncontracted shells of
  the scanned type are optimised in log space with BOBYQA
  (:func:`vibe_basis.optimize.optimize_nlopt`), bounded below by 0.15. That
  is the lower limit the rev2 optimisation used for numerical stability
  (Vilela Oliveira, Laun, Peintinger & Bredow, J. Comput. Chem. 40, 2364
  (2019), section "Basis Set Optimization"; 0.12 in some cases). A seed
  below the bound, such as ``add-low``'s ``a_n / 2.5`` on a 0.29 exponent, is
  moved up to it and the move is recorded.
* **The score.** ``objective`` is the engine's total energy per cell in
  kJ/mol, ``cost`` the primitive count, ``instability`` log10 of the
  condition number of the element's same-centre overlap
  (:func:`..ld_diagnostics.compute_overlap_diagnostics_from_atom`).
* **Failed points.** A non-converged SCF or a transport failure goes back to
  BOBYQA as the worst energy seen so far plus a margin. An infinite value
  would poison BOBYQA's quadratic model: in a two-parameter test one
  infinite point stopped it at a wrong point with "xtol reached".
* **Resume.** Every energy is journalled through
  :class:`vibe_basis.cache.EnergyCache`, keyed by engine, system, basis text
  and method, so a scan restarted on the same work directory reuses every
  completed point. SCF failures are journalled. A point the transport loses
  is retried, and when every retry fails the scan stops with
  :class:`TransportUnavailable` rather than handing BOBYQA a penalty that is
  about the host.

With a total-energy objective, gate (c) of #225 asks whether the shipped
topology is on the front or within the plateau threshold of it
(:func:`vibe_basis.topology.front_gap`). The variational principle makes
that close to automatic: adding a function never raises the energy. A pass
shows the search runs and ranks, not that it discovers anything; that needs
the M2 cohesive objective.
"""

from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from vibe_basis.cache import EnergyCache, crystal_key
from vibe_basis.engine import EngineEnergy
from vibe_basis.optimize import optimize_nlopt
from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL
from vibe_basis.topology import (
    BasisTopology,
    Relaxed,
    ScanResult,
    ScanScore,
    Shell,
    front_gap,
    greedy_scan,
)

from ...basis_crystal import (
    CrystalAtomBasis,
    CrystalShell,
    emit_crystal,
    parse_crystal_atom_basis_file,
)
from ...basis_toolkit._element_data import SYMBOL
from ..ld_diagnostics import compute_overlap_diagnostics_from_atom

__all__ = [
    "DEFAULT_LOWER_BOUND",
    "ScanEvaluator",
    "TopologyScanReport",
    "TransportUnavailable",
    "atoms_from_topology",
    "crystal23_engine_factory",
    "load_pob_atoms",
    "run_topology_scan",
    "topology_from_atoms",
]

#: Lower bound on a relaxed exponent: the rev2 optimisation's limit.
DEFAULT_LOWER_BOUND = 0.15

#: Upper bound on a relaxed exponent. Far above any polarisation function;
#: it only keeps the log-space box finite, as BOBYQA needs.
DEFAULT_UPPER_BOUND = 100.0

#: Engine failures that say nothing about the basis, so a resumed scan
#: retries them rather than reading them back from the journal.
_TRANSIENT_FAILURES = frozenset({"transport_error", "no_output_file"})


class TransportUnavailable(RuntimeError):
    """A point could not reach its engine after every retry; the scan stops."""


# ---------------------------------------------------------------------------
# pob records <-> topologies
# ---------------------------------------------------------------------------


def _sources_dir(family: str) -> Path:
    from ... import basis_crystal

    path = Path(basis_crystal.__file__).resolve().parent / "basis_library" / "sources" / family
    if not path.is_dir():
        raise FileNotFoundError(f"no bundled source records for {family!r} at {path}")
    return path


def load_pob_atoms(
    family: str, symbols: Sequence[str]
) -> dict[str, CrystalAtomBasis]:
    """The bundled per-element CRYSTAL records of ``family``, keyed by symbol."""
    directory = _sources_dir(family)
    atoms: dict[str, CrystalAtomBasis] = {}
    for symbol in symbols:
        matches = sorted(directory.glob(f"*_{symbol}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{family}: expected one source record for {symbol}, found "
                f"{[p.name for p in matches]}"
            )
        atoms[symbol] = parse_crystal_atom_basis_file(matches[0])
    return atoms


def topology_from_atoms(atoms: Mapping[str, CrystalAtomBasis]) -> BasisTopology:
    """Transcribe parsed records into a :class:`BasisTopology`.

    The parser has already applied each shell's ``SCAL``, so every
    transcribed shell carries ``scale_factor=1.0``.
    """
    return BasisTopology({
        symbol: tuple(
            Shell(
                shell_type=s.shell_type,
                exponents=tuple(s.exponents),
                coefficients=tuple(s.coefficients),
                coefficients_p=tuple(s.coefficients_p),
                occupancy=float(s.occupancy),
                scale_factor=float(s.scale_factor),
            )
            for s in atom.shells
        )
        for symbol, atom in atoms.items()
    })


def atoms_from_topology(
    topology: BasisTopology, template: Mapping[str, CrystalAtomBasis]
) -> dict[str, CrystalAtomBasis]:
    """The records ``topology`` describes, with ``template``'s Z and ECPs.

    Keyed and ordered as ``template``. A seeded shell carries occupancy 0,
    which is what CRYSTAL needs for an added polarisation function.
    """
    out: dict[str, CrystalAtomBasis] = {}
    for symbol, atom in template.items():
        shells = [
            CrystalShell(
                shell_type=s.shell_type,
                occupancy=s.occupancy,
                scale_factor=s.scale_factor,
                exponents=list(s.exponents),
                coefficients=list(s.coefficients),
                coefficients_p=list(s.coefficients_p),
            )
            for s in topology.atoms[symbol]
        ]
        out[symbol] = replace(atom, shells=shells)
    return out


def _with_exponents(
    topology: BasisTopology, element: str, indices: Sequence[int],
    exponents: Sequence[float],
) -> BasisTopology:
    shells = list(topology.atoms[element])
    for i, a in zip(indices, exponents):
        shells[i] = replace(shells[i], exponents=(float(a),))
    atoms = dict(topology.atoms)
    atoms[element] = tuple(shells)
    return BasisTopology(atoms)


# ---------------------------------------------------------------------------
# One candidate: relax, score, record
# ---------------------------------------------------------------------------


@dataclass
class ScanEvaluator:
    """``evaluate(topology) -> Relaxed`` for :func:`vibe_basis.topology.greedy_scan`.

    ``energy(basis_text)`` returns the engine's :class:`EngineEnergy` for the
    scan's crystal; :func:`run_topology_scan` builds it with the journal and
    a work directory per point. Safe to call from several threads: each call
    keeps its own points.
    """

    element: str
    shell_type: str
    template: Mapping[str, CrystalAtomBasis]
    energy: Callable[[str], EngineEnergy]
    lower_bound: float = DEFAULT_LOWER_BOUND
    upper_bound: float = DEFAULT_UPPER_BOUND
    max_eval: int = 40
    # 1 % on an exponent: near the optimum that is ~1e-3 kJ/mol, far below
    # the 1 kJ/mol plateau threshold, and each saved point is a CRYSTAL job.
    xtol: float = 0.01
    ftol_kj: float = 0.01
    initial_step: float = 0.3
    failure_margin_kj: float = 100.0
    optimizer: Callable[..., Any] = optimize_nlopt

    def free_shells(self, topology: BasisTopology) -> tuple[int, ...]:
        """The element's uncontracted shells of the scanned type."""
        return tuple(
            i for i in topology.shell_indices(self.element, self.shell_type)
            if topology.atoms[self.element][i].is_uncontracted
        )

    def basis_text(self, topology: BasisTopology) -> str:
        return emit_crystal(list(atoms_from_topology(topology, self.template).values()))

    def instability(self, topology: BasisTopology) -> tuple[float, float]:
        """(log10 condition number, condition number) of the element's overlap."""
        atom = atoms_from_topology(topology, self.template)[self.element]
        cond = float(compute_overlap_diagnostics_from_atom(atom).condition_number)
        return (math.log10(cond) if cond > 0 and math.isfinite(cond) else math.inf), cond

    def __call__(self, topology: BasisTopology) -> Relaxed:
        indices = self.free_shells(topology)
        seeds = [topology.atoms[self.element][i].exponents[0] for i in indices]
        start = [min(max(a, self.lower_bound), self.upper_bound) for a in seeds]
        points: list[dict[str, Any]] = []

        def evaluate(exponents: Sequence[float]) -> float | None:
            exponents = [float(a) for a in exponents]
            for p in points:  # BOBYQA's first call repeats the start point
                if p["exponents"] == exponents:
                    e = p["energy_ha"]
                    return None if e is None else e * HARTREE_TO_KJ_PER_MOL
            candidate = _with_exponents(topology, self.element, indices, exponents)
            result = self.energy(self.basis_text(candidate))
            e_ha = result.energy_if_ok()
            points.append({
                "exponents": exponents,
                "energy_ha": e_ha,
                "failure_mode": result.failure_mode,
                "engine": result.engine,
                "engine_version": result.engine_version,
            })
            return None if e_ha is None else e_ha * HARTREE_TO_KJ_PER_MOL

        optimizer_message = "no free exponents"
        first = evaluate(start)
        if first is not None and indices:
            def objective(x: np.ndarray) -> float:
                value = evaluate(np.exp(x))
                if value is not None:
                    return value
                finite = [p["energy_ha"] for p in points if p["energy_ha"] is not None]
                return max(finite) * HARTREE_TO_KJ_PER_MOL + self.failure_margin_kj

            lo, hi = math.log(self.lower_bound), math.log(self.upper_bound)
            opt = self.optimizer(
                objective,
                np.log(np.asarray(start, dtype=float)),
                bounds=[(lo, hi)] * len(indices),
                tol=0.0,
                ftol_abs=self.ftol_kj,
                xtol_abs=self.xtol,
                max_eval=self.max_eval,
                initial_step=self.initial_step,
            )
            optimizer_message = f"{opt.message}; {opt.n_evaluations} evaluations"
        elif first is None:
            optimizer_message = "the starting point failed; not relaxed"

        finite = [p for p in points if p["energy_ha"] is not None]
        if finite:
            best = min(finite, key=lambda p: p["energy_ha"])
            relaxed = _with_exponents(topology, self.element, indices, best["exponents"])
            objective_kj = best["energy_ha"] * HARTREE_TO_KJ_PER_MOL
            energy_ha = best["energy_ha"]
            exponents = best["exponents"]
        else:
            relaxed, objective_kj, energy_ha, exponents = topology, math.inf, None, start
        log_cond, cond = self.instability(relaxed)
        detail = {
            "free_shells": list(indices),
            "seed_exponents": [float(a) for a in seeds],
            "start_exponents": [float(a) for a in start],
            "seed_moved_to_bounds": [float(a) for a in seeds] != [float(a) for a in start],
            "exponents": [float(a) for a in exponents],
            "energy_ha": energy_ha,
            "start_energy_ha": points[0]["energy_ha"],
            "n_points": len(points),
            "n_failed": len(points) - len(finite),
            "optimizer": optimizer_message,
            "overlap_condition_number": cond,
            "points": points,
        }
        return Relaxed(
            score=ScanScore(
                objective=objective_kj,
                cost=float(relaxed.n_primitives),
                instability=log_cond if finite else math.inf,
            ),
            topology=relaxed,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# Engines and energies
# ---------------------------------------------------------------------------


def crystal23_engine_factory(
    transport: Any,
    *,
    crystal_binary: str = "crystal",
    invocation: str = "stdin",
    cpus: int = 1,
    wall_time_s: int = 3600,
    timeout_s: float = 86_400.0,
) -> Callable[[Path], Any]:
    """``factory(point_dir) -> Crystal23Engine`` for :func:`run_topology_scan`.

    ``invocation="stdin"`` runs a serial ``crystal`` binary on the deck;
    ``"mpi"`` runs ``cpus`` ranks of a ``Pcrystal`` binary. Each point is
    staged under its own directory, and a point whose output banner is not
    CRYSTAL23 fails.
    """
    from vibe_basis.engines.crystal23 import Crystal23Engine

    def make(point_dir: Path) -> Crystal23Engine:
        return Crystal23Engine(
            transport=transport,
            crystal_wrapper=crystal_binary,
            invocation=invocation,
            cpus=cpus,
            wall_time_s=wall_time_s,
            timeout_s=timeout_s,
            workdir_prefix=str(point_dir / "calc"),
            strict_version=True,
        )

    return make


def _journalled_energy(
    engine_factory: Callable[[Path], Any],
    structure: Any,
    method: str,
    workdir: Path,
    cache: EnergyCache,
    transient_retries: int = 2,
) -> tuple[Callable[[str], EngineEnergy], str]:
    engine_name = engine_factory(workdir / "points" / "probe").name
    lock = threading.Lock()
    counter = [0]

    def energy(basis_text: str) -> EngineEnergy:
        key = crystal_key(engine_name, structure.name, basis_text, method)
        with lock:
            hit = cache.get(key)
            if hit is None:
                counter[0] += 1
                n = counter[0]
        if hit is not None:
            return hit
        point_dir = workdir / "points" / f"point{n:04d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        (point_dir / "basis.crystal").write_text(basis_text)
        # A lost submission or fetch says nothing about the basis, and handed
        # to the optimiser it would read as a penalty at that point: retry it.
        for attempt in range(transient_retries + 1):
            result = engine_factory(point_dir / f"try{attempt}").crystal_energy(
                basis_text, structure, method=method
            )
            if result.ok or result.failure_mode not in _TRANSIENT_FAILURES:
                break
        else:
            # Still unreachable: stop rather than hand the optimiser a penalty
            # that is about the host, not the basis. The journal keeps every
            # finished point, so a restart resumes here.
            raise TransportUnavailable(
                f"{point_dir.name}: {transient_retries + 1} attempts ended in "
                f"{result.failure_mode} ({result.detail.get('error', '')})"
            )
        with lock:
            cache.put(key, result)
        return result

    return energy, engine_name


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


@dataclass
class TopologyScanReport:
    """A finished scan, its settings, and gate (c)'s reading of it."""

    result: ScanResult
    settings: dict[str, Any]
    shipped_gap_kj: float
    shipped_on_front: bool
    threshold: float
    record_path: Path | None = None
    cache_stats: dict[str, int] = field(default_factory=dict)

    @property
    def shipped_within_threshold(self) -> bool:
        """Gate (c): the shipped topology is on, or within threshold of, the front."""
        return self.shipped_gap_kj <= self.threshold

    def ranked_front(self) -> list[dict[str, Any]]:
        """The front's candidates, best objective first."""
        on_front = set(self.result.front)
        rows = [c for c in self.result.candidates if c["name"] in on_front]
        return sorted(rows, key=lambda c: c["score"].objective)

    def to_json(self) -> dict[str, Any]:
        def score(s: ScanScore) -> dict[str, float]:
            return asdict(s)

        return {
            "settings": self.settings,
            "gate_c": {
                "shipped_gap_kj_per_mol": self.shipped_gap_kj,
                "shipped_on_front": self.shipped_on_front,
                "threshold_kj_per_mol": self.threshold,
                "shipped_within_threshold": self.shipped_within_threshold,
            },
            "stopped_because": self.result.stopped_because,
            "rounds": self.result.rounds,
            "best": {
                "score": score(self.result.score),
                "topology": self.result.topology.to_atom_dict(),
            },
            "front": [c["name"] for c in self.ranked_front()],
            "history": list(self.result.history),
            "candidates": [
                {**c, "score": score(c["score"])} for c in self.result.candidates
            ],
            "cache": self.cache_stats,
        }


def run_topology_scan(
    structure: Any,
    element: str,
    shell_type: str,
    engine_factory: Callable[[Path], Any],
    workdir: Path | str,
    *,
    family: str = "pob-TZVP-rev2",
    method: str = "pw1pw",
    parallel: int = 4,
    threshold: float = 1.0,
    patience: int = 2,
    max_rounds: int = 8,
    beam_width: int = 1,
    transient_retries: int = 2,
    **relax: Any,
) -> TopologyScanReport:
    """Scan ``element``'s ``shell_type`` shells in ``structure`` from ``family``.

    ``engine_factory(point_dir)`` returns the engine for one energy
    (:func:`crystal23_engine_factory`). ``parallel`` candidates of a round
    are relaxed at once; each relaxation is sequential. A point lost to the
    transport is retried ``transient_retries`` times. ``relax`` passes
    through to :class:`ScanEvaluator` (bounds, ``max_eval``, tolerances).
    The journal and the record, ``scan_record.json``, live in ``workdir``.
    """
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    symbols = list(dict.fromkeys(SYMBOL[atom.Z] for atom in structure.crystal_asymm_unit))
    if element not in symbols:
        raise ValueError(f"{structure.name} has no {element}; its elements are {symbols}")
    template = load_pob_atoms(family, symbols)
    topology = topology_from_atoms(template)

    cache = EnergyCache(workdir / "energies.jsonl")
    energy, engine_name = _journalled_energy(
        engine_factory, structure, method, workdir, cache, transient_retries
    )
    evaluator = ScanEvaluator(element, shell_type, template, energy, **relax)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        result = greedy_scan(
            topology, element, shell_type, evaluator,
            threshold=threshold, patience=patience, max_rounds=max_rounds,
            beam_width=beam_width, map_candidates=pool.map,
        )

    import vibe_basis

    from ... import __version__ as vibeqc_version

    settings = {
        "vibeqc_version": vibeqc_version,
        "vibe_basis_version": vibe_basis.__version__,
        "structure": structure.name,
        "element": element,
        "shell_type": shell_type,
        "family": family,
        "method": method,
        "engine": engine_name,
        "objective": "total energy per cell, kJ/mol",
        "threshold_kj_per_mol": threshold,
        "patience": patience,
        "max_rounds": max_rounds,
        "beam_width": beam_width,
        "parallel": parallel,
        "transient_retries": transient_retries,
        "lower_bound": evaluator.lower_bound,
        "upper_bound": evaluator.upper_bound,
        "max_eval": evaluator.max_eval,
        "xtol_log_exponent": evaluator.xtol,
        "ftol_kj_per_mol": evaluator.ftol_kj,
        "initial_step": evaluator.initial_step,
        "failure_margin_kj_per_mol": evaluator.failure_margin_kj,
    }
    report = TopologyScanReport(
        result=result,
        settings=settings,
        shipped_gap_kj=front_gap(result.scores(), "round0:baseline"),
        shipped_on_front="round0:baseline" in result.front,
        threshold=threshold,
        cache_stats=cache.stats,
    )
    report.record_path = workdir / "scan_record.json"
    report.record_path.write_text(json.dumps(report.to_json(), indent=2, default=str) + "\n")
    return report
