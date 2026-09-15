"""Basis topology and the d/f scan: M3's search layer.

CRYSTAL23's ``OPTBASIS`` optimises exponents at a **fixed topology**. This
module makes the topology itself the search space: which shells exist, at
which angular momentum, and how many primitives each carries. That is axis 2
of the roadmap, and the reason a manual d/f scan is expensive enough to be
worth automating.

The scan, for angular momentum ``l`` on element ``E`` with existing exponents
a1 > ... > an, and ``r`` the local even-tempered ratio:

===============  ===========================  ==============================
move             seed                          asks
===============  ===========================  ==============================
``add-high``     a0 = a1 * r                   core-region flexibility
``add-low``      a_{n+1} = an / r              tail / cohesion, LD-risky
``leave-one-out`` drop one shell of that l     is each function earning it?
``baseline``     unchanged                     control
===============  ===========================  ==============================

Tier 1, so this module never imports ``vibeqc`` (ROADMAP.md 3.6). Three
consequences shape the API, all following the pattern
:mod:`vibe_basis.objective` set:

* **Evaluation is injected.** Every candidate is supposed to get a full L2
  continuous relaxation before it is judged, which needs an engine. The
  driver here takes an ``evaluate(topology) -> ScanScore`` callable, so the
  search logic is testable without one.
* **The parametrisation bridge is injected.** The roadmap sketches
  ``.as_parametrisation() -> BasisParametrisation``, but that class is Tier 2
  (``vibeqc.basis_optimization.parametrise``). :meth:`BasisTopology.as_parametrisation`
  therefore takes the builder as an argument; :meth:`BasisTopology.to_atom_dict`
  gives the same content as plain data for callers that would rather convert
  themselves.
* **Topologies are immutable.** Every mutator returns a new
  :class:`BasisTopology`. A search branches, and a mutator that edited in
  place would corrupt the parent it came from.

Shell shape mirrors ``vibeqc.basis_crystal.CrystalShell`` field for field
(``shell_type``, ``exponents``, ``coefficients``, ``coefficients_p``,
``occupancy``, ``scale_factor``) so the Tier-2 adapter is a transcription and
not a translation. It is deliberately a separate class rather than an import:
see the duplicated-``OptResult`` incident in HANDOVER_BASISOPT.md 3.6, whose
lesson was that the boundary needs *declared* shapes, not shared ones.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable

__all__ = [
    "BasisTopology",
    "Move",
    "ScanResult",
    "ScanScore",
    "Shell",
    "TopologyError",
    "even_tempered_ratio",
    "greedy_scan",
    "pareto_front",
    "propose_moves",
]

# The CRYSTAL shell types, in the spelling CrystalShell uses.
SHELL_TYPES = ("S", "SP", "P", "D", "F", "G")

# Used when a shell type has a single exponent and no adjacent pair to
# measure a ratio from. 2.5 is the roadmap's stated default.
DEFAULT_RATIO = 2.5


class TopologyError(ValueError):
    """The topology, or an edit to it, is not well formed."""


# ---------------------------------------------------------------------------
# Shells and topologies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shell:
    """One contracted Gaussian shell.

    Mirrors ``vibeqc.basis_crystal.CrystalShell``. ``coefficients_p`` carries
    the p-side of an ``SP`` shell and is empty otherwise, exactly as there.
    """

    shell_type: str
    exponents: tuple[float, ...]
    coefficients: tuple[float, ...]
    coefficients_p: tuple[float, ...] = ()
    occupancy: float = 0.0
    scale_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.shell_type not in SHELL_TYPES:
            raise TopologyError(
                f"shell_type must be one of {SHELL_TYPES}, got "
                f"{self.shell_type!r}"
            )
        if not self.exponents:
            raise TopologyError(f"{self.shell_type} shell has no exponents")
        if any(not (a > 0.0) or not math.isfinite(a) for a in self.exponents):
            raise TopologyError(
                f"{self.shell_type} shell has a non-positive or non-finite "
                f"exponent: {self.exponents}"
            )
        if len(self.coefficients) != len(self.exponents):
            raise TopologyError(
                f"{self.shell_type} shell has {len(self.exponents)} exponents "
                f"but {len(self.coefficients)} coefficients"
            )
        if self.shell_type == "SP":
            if len(self.coefficients_p) != len(self.exponents):
                raise TopologyError(
                    "SP shell needs one p coefficient per exponent, got "
                    f"{len(self.coefficients_p)} for {len(self.exponents)}"
                )
        elif self.coefficients_p:
            raise TopologyError(
                f"{self.shell_type} shell carries coefficients_p, which only "
                "an SP shell has"
            )

    @property
    def n_primitives(self) -> int:
        return len(self.exponents)

    @property
    def is_uncontracted(self) -> bool:
        return len(self.exponents) == 1


def _as_shells(shells: Iterable[Shell]) -> tuple[Shell, ...]:
    out = tuple(shells)
    for s in out:
        if not isinstance(s, Shell):
            raise TopologyError(f"expected Shell, got {type(s).__name__}")
    return out


@dataclass(frozen=True)
class BasisTopology:
    """An element-keyed basis, and the edits a topology search makes to it.

    Every mutator returns a **new** topology; nothing is edited in place.
    """

    atoms: Mapping[str, tuple[Shell, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.atoms:
            raise TopologyError("topology has no atoms")
        frozen = {str(k): _as_shells(v) for k, v in self.atoms.items()}
        for element, shells in frozen.items():
            if not shells:
                raise TopologyError(f"{element}: no shells")
        object.__setattr__(self, "atoms", frozen)

    # -- inspection ---------------------------------------------------------

    @property
    def n_primitives(self) -> int:
        """Total primitive count: the cost axis the Pareto front ranks on."""
        return sum(s.n_primitives for shells in self.atoms.values() for s in shells)

    def _require(self, element: str) -> tuple[Shell, ...]:
        try:
            return self.atoms[element]
        except KeyError:
            raise TopologyError(
                f"unknown element {element!r}; have {sorted(self.atoms)}"
            ) from None

    def shell_indices(self, element: str, shell_type: str) -> tuple[int, ...]:
        """Indices of ``element``'s shells of this type, in stored order."""
        return tuple(
            i for i, s in enumerate(self._require(element))
            if s.shell_type == shell_type
        )

    def exponents(self, element: str, shell_type: str) -> tuple[float, ...]:
        """Every exponent of this angular momentum, sorted **descending**.

        Descending is the roadmap's a1 > ... > an convention, so a1 is the
        tightest function and an the most diffuse.
        """
        out: list[float] = []
        for s in self._require(element):
            if s.shell_type == shell_type:
                out.extend(s.exponents)
        return tuple(sorted(out, reverse=True))

    # -- mutators -----------------------------------------------------------

    def add_shell(
        self, element: str, shell_type: str, exponent: float,
        *, coefficient: float = 1.0, coefficient_p: float | None = None,
    ) -> "BasisTopology":
        """Seed one new **uncontracted** function of ``shell_type``.

        Placed after the element's existing shells. A seeded function is
        uncontracted on purpose: it is a new degree of freedom for the L2
        relaxation to place, not a guess at a contraction.
        """
        shells = self._require(element)
        if shell_type == "SP" and coefficient_p is None:
            raise TopologyError(
                "seeding an SP shell needs coefficient_p as well"
            )
        new = Shell(
            shell_type=shell_type,
            exponents=(float(exponent),),
            coefficients=(float(coefficient),),
            coefficients_p=(
                (float(coefficient_p),) if shell_type == "SP" else ()
            ),
        )
        return self._with(element, shells + (new,))

    def drop_shell(self, element: str, index: int) -> "BasisTopology":
        """Leave-one-out: remove one whole shell."""
        shells = self._require(element)
        if not 0 <= index < len(shells):
            raise TopologyError(
                f"{element}: shell index {index} out of range "
                f"[0, {len(shells)})"
            )
        if len(shells) == 1:
            raise TopologyError(
                f"{element}: refusing to drop its only shell, which would "
                "leave the element with no basis at all"
            )
        return self._with(element, shells[:index] + shells[index + 1:])

    def split_contraction(
        self, element: str, index: int, k: int
    ) -> "BasisTopology":
        """Decontract: split shell ``index`` after its ``k``-th primitive.

        The two halves keep their own coefficients unchanged. Renormalising
        them is a scientific decision, not a mechanical one, so it is left to
        the L2 relaxation that follows rather than guessed here.
        """
        shells = self._require(element)
        if not 0 <= index < len(shells):
            raise TopologyError(
                f"{element}: shell index {index} out of range "
                f"[0, {len(shells)})"
            )
        shell = shells[index]
        if not 0 < k < shell.n_primitives:
            raise TopologyError(
                f"{element}: split point {k} must be strictly inside the "
                f"shell's {shell.n_primitives} primitives"
            )
        head = replace(
            shell,
            exponents=shell.exponents[:k],
            coefficients=shell.coefficients[:k],
            coefficients_p=shell.coefficients_p[:k] if shell.coefficients_p else (),
        )
        tail = replace(
            shell,
            exponents=shell.exponents[k:],
            coefficients=shell.coefficients[k:],
            coefficients_p=shell.coefficients_p[k:] if shell.coefficients_p else (),
        )
        return self._with(
            element, shells[:index] + (head, tail) + shells[index + 1:]
        )

    def _with(self, element: str, shells: tuple[Shell, ...]) -> "BasisTopology":
        atoms = dict(self.atoms)
        atoms[element] = shells
        return BasisTopology(atoms)

    # -- bridges ------------------------------------------------------------

    def to_atom_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Plain-data view, one dict per shell, keyed by element.

        Field names match ``CrystalShell``, so a Tier-2 adapter can build one
        per entry without a mapping table.
        """
        return {
            element: [
                {
                    "shell_type": s.shell_type,
                    "occupancy": s.occupancy,
                    "scale_factor": s.scale_factor,
                    "exponents": list(s.exponents),
                    "coefficients": list(s.coefficients),
                    "coefficients_p": list(s.coefficients_p),
                }
                for s in shells
            ]
            for element, shells in self.atoms.items()
        }

    def as_parametrisation(self, builder: Callable[..., Any], **kwargs: Any) -> Any:
        """Hand this topology to a Tier-2 ``BasisParametrisation`` builder.

        ``builder`` receives ``to_atom_dict()`` as its first argument. It is
        injected because the target class lives on the vibe-qc side and Tier 1
        must not import it.
        """
        return builder(self.to_atom_dict(), **kwargs)


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def even_tempered_ratio(exponents: Sequence[float]) -> float:
    """Local even-tempered ratio: geometric mean of adjacent ratios.

    ``exponents`` is taken descending, so every adjacent ratio exceeds 1 and
    the geometric mean is the factor the series steps by. With a single
    exponent there is no adjacent pair and :data:`DEFAULT_RATIO` is returned,
    per the roadmap.
    """
    xs = sorted((float(a) for a in exponents), reverse=True)
    if not xs:
        raise TopologyError("no exponents to take a ratio from")
    if any(a <= 0.0 for a in xs):
        raise TopologyError(f"exponents must be positive, got {xs}")
    if len(xs) == 1:
        return DEFAULT_RATIO
    ratios = [xs[i] / xs[i + 1] for i in range(len(xs) - 1)]
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


@dataclass(frozen=True)
class Move:
    """One candidate topology and where it came from."""

    name: str
    topology: "BasisTopology"
    element: str
    shell_type: str
    provenance: str


def propose_moves(
    topology: BasisTopology, element: str, shell_type: str
) -> tuple[Move, ...]:
    """The four move kinds of the roadmap's scan table, for one (element, l).

    Always includes ``baseline`` first, as the control. ``leave-one-out``
    covers the element's shells **of this type**, one move each; it is not
    offered for the element's only shell.

    Dropping a single primitive from inside a contraction is deliberately not
    part of the scan: it changes what the contraction means, and the
    remaining coefficients would need recontracting before the candidate is
    comparable. :meth:`BasisTopology.split_contraction` exposes decontraction
    for a caller that wants it explicitly.
    """
    shells = topology._require(element)
    exps = topology.exponents(element, shell_type)
    moves: list[Move] = [
        Move("baseline", topology, element, shell_type, "unchanged control")
    ]
    if exps:
        r = even_tempered_ratio(exps)
        high = exps[0] * r
        low = exps[-1] / r
        moves.append(Move(
            "add-high",
            topology.add_shell(element, shell_type, high),
            element, shell_type,
            f"seeded a1*r = {exps[0]:.6g} * {r:.6g} = {high:.6g}",
        ))
        moves.append(Move(
            "add-low",
            topology.add_shell(element, shell_type, low),
            element, shell_type,
            f"seeded an/r = {exps[-1]:.6g} / {r:.6g} = {low:.6g}",
        ))
    for idx in topology.shell_indices(element, shell_type):
        if len(shells) == 1:
            break
        dropped = shells[idx]
        moves.append(Move(
            f"leave-one-out:{idx}",
            topology.drop_shell(element, idx),
            element, shell_type,
            "dropped the {} shell at index {} (exponents {})".format(
                shell_type, idx,
                ", ".join(f"{a:.6g}" for a in dropped.exponents),
            ),
        ))
    return tuple(moves)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanScore:
    """What one evaluated candidate is ranked on. **All minimised.**

    ``objective`` is the M2 value (held-out error is the one to quote),
    ``cost`` defaults to the primitive count, and ``instability`` is a
    conditioning measure -- the overlap condition number, or ``1/lambda_min``.
    Bigger is worse for all three, so the Pareto rule is a single direction.
    """

    objective: float
    cost: float
    instability: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.objective, self.cost, self.instability)


def pareto_front(
    scored: Mapping[str, ScanScore] | Sequence[tuple[str, ScanScore]]
) -> tuple[str, ...]:
    """Names of the non-dominated candidates, in input order.

    ``a`` dominates ``b`` when it is no worse on every axis and strictly
    better on at least one. Ties on every axis do not dominate, so exact
    duplicates both survive rather than one silently winning on dict order.
    """
    items = list(scored.items()) if isinstance(scored, Mapping) else list(scored)
    if not items:
        return ()
    keep: list[str] = []
    for name, s in items:
        a = s.as_tuple()
        dominated = False
        for other, t in items:
            if other == name:
                continue
            b = t.as_tuple()
            if all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a)):
                dominated = True
                break
        if not dominated:
            keep.append(name)
    return tuple(keep)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanResult:
    """What a scan produced, and enough provenance to defend it."""

    topology: BasisTopology
    score: ScanScore
    history: tuple[dict[str, Any], ...]
    front: tuple[str, ...]
    rounds: int
    stopped_because: str


def greedy_scan(
    topology: BasisTopology,
    element: str,
    shell_type: str,
    evaluate: Callable[[BasisTopology], ScanScore],
    *,
    threshold: float = 1.0,
    patience: int = 2,
    max_rounds: int = 8,
    beam_width: int = 1,
) -> ScanResult:
    """Hill-climb the topology, with the roadmap's plateau rule.

    Each round proposes the moves for ``(element, shell_type)`` from every
    topology currently in the beam, evaluates them, and keeps the best
    ``beam_width``. A round *improves* when the best objective beats the
    incumbent by more than ``threshold`` -- stated in the objective's own
    units, so 1.0 is the roadmap's 1 kJ/mol held-out MAD. After ``patience``
    consecutive non-improving rounds the scan stops.

    ``beam_width=1`` is the greedy default; the roadmap keeps width 3
    available for a production run. Full enumeration is not offered: the
    candidate count is exponential in the number of rounds and each candidate
    costs an L2 relaxation.

    The Pareto front is taken over **every candidate evaluated across all
    rounds**, not just the last one, because a candidate that lost on the
    objective may still be the one to ship on cost or conditioning. Reporting
    a front rather than a winner is the point: the operating point is the
    maintainer's call.
    """
    if threshold < 0.0:
        raise TopologyError(f"threshold must be >= 0, got {threshold}")
    if patience < 1:
        raise TopologyError(f"patience must be >= 1, got {patience}")
    if beam_width < 1:
        raise TopologyError(f"beam_width must be >= 1, got {beam_width}")
    if max_rounds < 0:
        raise TopologyError(f"max_rounds must be >= 0, got {max_rounds}")

    base_score = evaluate(topology)
    all_scores: list[tuple[str, ScanScore]] = [("round0:baseline", base_score)]
    history: list[dict[str, Any]] = []
    beam: list[tuple[BasisTopology, ScanScore]] = [(topology, base_score)]
    best_topology, best_score = topology, base_score
    stale = 0
    stopped = "max_rounds reached"
    rounds = 0

    for rnd in range(1, max_rounds + 1):
        rounds = rnd
        seen: list[tuple[str, BasisTopology, ScanScore, Move]] = []
        for slot, (parent, _) in enumerate(beam):
            for move in propose_moves(parent, element, shell_type):
                if move.name == "baseline" and rnd > 1:
                    continue  # the incumbent is already in the beam
                tag = f"round{rnd}:slot{slot}:{move.name}"
                score = evaluate(move.topology)
                seen.append((tag, move.topology, score, move))
                all_scores.append((tag, score))

        if not seen:
            stopped = "no moves proposed"
            break

        seen.sort(key=lambda row: row[2].objective)
        gain = best_score.objective - seen[0][2].objective
        improved = gain > threshold
        history.append({
            "round": rnd,
            "best_move": seen[0][0],
            "best_objective": seen[0][2].objective,
            "incumbent_objective": best_score.objective,
            "gain": gain,
            "improved": improved,
            "provenance": seen[0][3].provenance,
            "n_candidates": len(seen),
        })

        if improved:
            best_topology, best_score = seen[0][1], seen[0][2]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                stopped = (
                    f"{patience} consecutive rounds improved by no more than "
                    f"the {threshold:g} threshold"
                )
                break
        beam = [(t, s) for _, t, s, _ in seen[:beam_width]]

    return ScanResult(
        topology=best_topology,
        score=best_score,
        history=tuple(history),
        front=pareto_front(all_scores),
        rounds=rounds,
        stopped_because=stopped,
    )
