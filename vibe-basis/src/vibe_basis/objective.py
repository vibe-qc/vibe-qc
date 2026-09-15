"""The L1 objective: cohesive energies against a plane-wave-limit target.

This is the quantity a basis-optimisation campaign minimises::

    f(x) = sum_i w_i * loss( E_coh,i(basis(x)) - E_ref,i )  [+ lambda * penalty]

CRYSTAL23's ``OPTBASIS`` minimises a *total energy* for one crystal. This
minimises the error of a *cohesive* energy against an independent reference,
jointly over many systems, and always reports a held-out validation error
alongside the training one. That last part is what makes "this basis is
better" a claim rather than a fit.

Tier 1, so this module never imports ``vibeqc`` (ROADMAP.md § 3.6): it is
objective *algebra* and works with any engine, or with numbers typed in by
hand. Two consequences shape the API:

* **References come from the caller.** :func:`load_pw_reference` reads the
  ``pwref_merged.json`` layout, but the path is the caller's to supply --
  the file lives in vibe-qc's ``studies/`` tree, which Tier 1 cannot reach
  into.
* **Penalties are injected.** ``ld_penalty`` and
  ``condition_number_penalty`` live in ``vibeqc.basis_optimization`` (Tier
  2). Pass one through ``penalty=`` rather than importing it here.

Candidate results are duck-typed: anything with ``system``, ``ok`` and
``cohesive_kj_per_mol`` works, which :class:`vibe_basis.pipeline.CohesiveResult`
does. A plain ``{system: kJ/mol}`` mapping works too.

Sign convention throughout: E_coh is **positive for a bound solid**, matching
``CohesiveResult.cohesive_kj_per_mol``, and a residual is
``candidate - reference``. So a positive mean deviation means the candidate
overbinds relative to the reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

__all__ = [
    "CohesiveObjective",
    "ObjectiveError",
    "ObjectiveReport",
    "SystemStats",
    "load_pw_reference",
    "split_validation",
]

# Loss names accepted by ``CohesiveObjective(loss=...)``.
LOSS_NAMES = ("rmsd", "mad", "maxabs", "huber")

MissingPolicy = Literal["raise", "penalise"]


class ObjectiveError(ValueError):
    """The objective cannot be evaluated as configured."""


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------


def load_pw_reference(path: str | Path) -> dict[str, float]:
    """Read a ``pwref_merged.json``-style file into ``{system: kJ/mol}``.

    The layout is ``{"provenance": {...}, "results": [{"system_id": ...,
    "pw_limit_kjmol": ...}, ...]}``. Only ``results`` is read; the
    provenance block is the caller's to inspect and quote.

    Raises
    ------
    ObjectiveError
        If the file is not that shape, or names a system twice. A silently
        de-duplicated reference would weight one system twice in the sum.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, Mapping) or "results" not in raw:
        raise ObjectiveError(
            f"{path}: expected a mapping with a 'results' key, got "
            f"{type(raw).__name__}"
        )
    out: dict[str, float] = {}
    for i, row in enumerate(raw["results"]):
        try:
            system = str(row["system_id"])
            value = float(row["pw_limit_kjmol"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ObjectiveError(
                f"{path}: results[{i}] needs 'system_id' and a numeric "
                f"'pw_limit_kjmol' ({exc})"
            ) from exc
        if system in out:
            raise ObjectiveError(f"{path}: system {system!r} appears twice")
        out[system] = value
    if not out:
        raise ObjectiveError(f"{path}: 'results' is empty")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemStats:
    """Error statistics over one set of systems, all in kJ/mol.

    ``md`` is signed (positive: the candidate overbinds); the rest are
    magnitudes. ``n`` is how many systems contributed.
    """

    n: int
    md: float
    mad: float
    rmsd: float
    max_abs: float
    worst_system: str | None


@dataclass(frozen=True)
class ObjectiveReport:
    """Everything one objective evaluation produced.

    Attributes
    ----------
    value
        The scalar the optimiser minimises: the training loss plus the
        penalty. This is what :meth:`CohesiveObjective.__call__` returns.
    loss_value
        The training loss alone, without the penalty.
    penalty_value
        The penalty term alone (0.0 when no penalty was configured).
    train, validation
        Error statistics. ``validation`` is ``None`` when no held-out
        split was configured. **The validation number is the one to quote**;
        the training number is what was optimised against.
    residuals
        ``{system: candidate - reference}`` for every scored system.
    missing
        Systems in the reference that the candidate did not produce a
        usable number for. Under ``on_missing="penalise"`` these are
        charged :attr:`CohesiveObjective.missing_penalty` each and do not
        appear in :attr:`residuals`.
    unchecked_conventions
        Non-empty when the ZPE / counterpoise convention was not pinned,
        naming what went unchecked. A static-lattice number and a
        ZPE-corrected one are different quantities, so an unchecked
        comparison is a real caveat, not boilerplate.
    """

    value: float
    loss: str
    loss_value: float
    penalty_value: float
    train: SystemStats
    validation: SystemStats | None
    residuals: dict[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    unchecked_conventions: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------


def _stats(residuals: Mapping[str, float], weights: Mapping[str, float]) -> SystemStats:
    """Weighted error statistics over ``residuals``."""
    systems = sorted(residuals)
    if not systems:
        return SystemStats(0, 0.0, 0.0, 0.0, 0.0, None)
    total_w = sum(weights[s] for s in systems)
    if total_w <= 0.0:
        raise ObjectiveError(
            "total weight over the scored systems is not positive; "
            "every weight must be > 0 for the averages to mean anything"
        )
    md = sum(weights[s] * residuals[s] for s in systems) / total_w
    mad = sum(weights[s] * abs(residuals[s]) for s in systems) / total_w
    rmsd = math.sqrt(
        sum(weights[s] * residuals[s] ** 2 for s in systems) / total_w
    )
    worst = max(systems, key=lambda s: abs(residuals[s]))
    return SystemStats(len(systems), md, mad, rmsd, abs(residuals[worst]), worst)


class CohesiveObjective:
    """Scalar objective over cohesive energies, with a held-out split.

    Parameters
    ----------
    references
        ``{system: reference E_coh in kJ/mol}``, e.g. from
        :func:`load_pw_reference`.
    loss
        ``"rmsd"`` (default), ``"mad"``, ``"maxabs"`` or ``"huber"``.
        RMSD is the default because it is the one the optimiser's
        trust-region machinery behaves best on; ``maxabs`` is
        non-smooth and is meant for reporting or a final polish.
    weights
        Optional ``{system: w}``, default 1.0 each. Every weight must be
        strictly positive: a zero weight silently removes a system, which
        is what ``validation`` is for.
    validation
        Systems held out of :attr:`ObjectiveReport.value` and reported
        separately. They must exist in ``references`` and must not be all
        of them.
    huber_delta
        Transition point of the Huber loss, kJ/mol. Ignored for other
        losses.
    penalty
        Optional ``callable() -> float`` (or a float) added to the loss.
        This is the seam for vibe-qc's ``ld_penalty`` /
        ``condition_number_penalty``, which are Tier 2 and cannot be
        imported here.
    on_missing
        What to do when the candidate has no usable number for a
        referenced system. ``"raise"`` (default) refuses;
        ``"penalise"`` charges ``missing_penalty`` per system.

        **There is deliberately no "skip".** Dropping unconverged systems
        makes the objective *improve* as the basis gets worse enough to
        break them, so the optimiser would learn to break them. Whatever
        the policy, a missing system must cost something.
    missing_penalty
        kJ/mol charged per missing system under ``on_missing="penalise"``.
        Should exceed the worst residual you would tolerate.
    expect_zero_point, expect_counterpoise
        When set, a candidate whose ``zero_point_applied`` /
        ``counterpoise_applied`` flag disagrees is refused rather than
        compared. Leaving them ``None`` does not check, and every report
        then carries the omission in
        :attr:`ObjectiveReport.unchecked_conventions`.
    """

    def __init__(
        self,
        references: Mapping[str, float],
        *,
        loss: str = "rmsd",
        weights: Mapping[str, float] | None = None,
        validation: Iterable[str] = (),
        huber_delta: float = 20.0,
        penalty: Callable[[], float] | float | None = None,
        on_missing: MissingPolicy = "raise",
        missing_penalty: float = 1000.0,
        expect_zero_point: bool | None = None,
        expect_counterpoise: bool | None = None,
    ) -> None:
        if loss not in LOSS_NAMES:
            raise ObjectiveError(
                f"loss must be one of {LOSS_NAMES}, got {loss!r}"
            )
        if on_missing not in ("raise", "penalise"):
            raise ObjectiveError(
                "on_missing must be 'raise' or 'penalise', got "
                f"{on_missing!r}. There is no 'skip': dropping a system the "
                "candidate failed on would reward breaking it."
            )
        if not references:
            raise ObjectiveError("references is empty")
        if huber_delta <= 0.0:
            raise ObjectiveError(f"huber_delta must be > 0, got {huber_delta}")
        if missing_penalty < 0.0:
            raise ObjectiveError(
                f"missing_penalty must be >= 0, got {missing_penalty}"
            )

        self.references = dict(references)
        self.loss = loss
        self.huber_delta = float(huber_delta)
        self.penalty = penalty
        self.on_missing: MissingPolicy = on_missing
        self.missing_penalty = float(missing_penalty)
        self.expect_zero_point = expect_zero_point
        self.expect_counterpoise = expect_counterpoise

        supplied = dict(weights or {})
        unknown = sorted(set(supplied) - set(self.references))
        if unknown:
            raise ObjectiveError(
                f"weights name systems absent from references: {unknown}"
            )
        for system, w in supplied.items():
            if not w > 0.0:
                raise ObjectiveError(
                    f"weight for {system!r} is {w}; every weight must be > 0 "
                    "(use validation= to hold a system out)"
                )
        self.weights = {s: float(supplied.get(s, 1.0)) for s in self.references}

        held_out = tuple(dict.fromkeys(validation))
        unknown = sorted(set(held_out) - set(self.references))
        if unknown:
            raise ObjectiveError(
                f"validation names systems absent from references: {unknown}"
            )
        if held_out and len(held_out) == len(self.references):
            raise ObjectiveError(
                "validation holds out every system, leaving nothing to "
                "optimise against"
            )
        self.validation = held_out
        self.training = tuple(s for s in self.references if s not in set(held_out))

    # -- candidate normalisation -------------------------------------------

    def _candidate_values(
        self, results: Mapping[str, float] | Iterable[Any]
    ) -> dict[str, float]:
        """Normalise candidate results to ``{system: kJ/mol}``.

        Accepts a mapping, or an iterable of result objects carrying
        ``system`` / ``ok`` / ``cohesive_kj_per_mol``. A result with
        ``ok`` false, or a ``None`` energy, is treated as missing rather
        than as a number.
        """
        if isinstance(results, Mapping):
            return {str(k): float(v) for k, v in results.items() if v is not None}

        values: dict[str, float] = {}
        for item in results:
            try:
                system = str(item.system)
            except AttributeError as exc:
                raise ObjectiveError(
                    "candidate results must be a {system: kJ/mol} mapping or "
                    "objects with .system / .ok / .cohesive_kj_per_mol; got "
                    f"{type(item).__name__}"
                ) from exc
            if system in values:
                raise ObjectiveError(
                    f"candidate results name system {system!r} twice"
                )
            if not getattr(item, "ok", True):
                continue
            energy = getattr(item, "cohesive_kj_per_mol", None)
            if energy is None:
                continue
            self._check_conventions(system, item)
            values[system] = float(energy)
        return values

    def _check_conventions(self, system: str, item: Any) -> None:
        """Refuse a candidate whose ZPE / counterpoise flags disagree."""
        for attr, expected, label in (
            ("zero_point_applied", self.expect_zero_point, "zero-point"),
            ("counterpoise_applied", self.expect_counterpoise, "counterpoise"),
        ):
            if expected is None:
                continue
            actual = getattr(item, attr, None)
            if actual is None:
                raise ObjectiveError(
                    f"{system}: expect_{attr[:-8]} was set but the result does "
                    f"not report {attr!r}, so the {label} convention cannot be "
                    "verified"
                )
            if bool(actual) is not bool(expected):
                raise ObjectiveError(
                    f"{system}: the {label} convention does not match the "
                    f"reference ({attr}={bool(actual)}, expected "
                    f"{bool(expected)}). A corrected and an uncorrected "
                    "cohesive energy are different quantities; comparing them "
                    "would fold the correction into the basis error."
                )

    def _unchecked(self) -> tuple[str, ...]:
        out = []
        if self.expect_zero_point is None:
            out.append("zero-point (expect_zero_point not set)")
        if self.expect_counterpoise is None:
            out.append("counterpoise (expect_counterpoise not set)")
        return tuple(out)

    # -- losses -------------------------------------------------------------

    def _loss(self, stats: SystemStats, residuals: Mapping[str, float]) -> float:
        if stats.n == 0:
            raise ObjectiveError("no system was scored, so the loss is undefined")
        if self.loss == "rmsd":
            return stats.rmsd
        if self.loss == "mad":
            return stats.mad
        if self.loss == "maxabs":
            return stats.max_abs
        # Huber: quadratic within delta, linear outside, weighted like the rest.
        delta = self.huber_delta
        systems = sorted(residuals)
        total_w = sum(self.weights[s] for s in systems)
        acc = 0.0
        for s in systems:
            d = abs(residuals[s])
            per = 0.5 * d * d if d <= delta else delta * (d - 0.5 * delta)
            acc += self.weights[s] * per
        return acc / total_w

    def _penalty_value(self) -> float:
        if self.penalty is None:
            return 0.0
        value = self.penalty() if callable(self.penalty) else self.penalty
        value = float(value)
        if not math.isfinite(value):
            raise ObjectiveError(f"penalty returned a non-finite value: {value}")
        return value

    # -- evaluation ---------------------------------------------------------

    def report(
        self, results: Mapping[str, float] | Iterable[Any]
    ) -> ObjectiveReport:
        """Evaluate and return everything, not just the scalar."""
        values = self._candidate_values(results)

        missing = tuple(s for s in self.references if s not in values)
        if missing and self.on_missing == "raise":
            raise ObjectiveError(
                f"candidate has no usable cohesive energy for {list(missing)}. "
                "Either fix those runs, or set on_missing='penalise' so a "
                "failed system costs the optimiser something instead of "
                "vanishing from the average."
            )

        residuals = {s: values[s] - self.references[s] for s in self.references if s in values}
        train_res = {s: r for s, r in residuals.items() if s in set(self.training)}
        val_res = {s: r for s, r in residuals.items() if s in set(self.validation)}

        if not train_res:
            raise ObjectiveError(
                "no training system produced a usable cohesive energy; "
                f"missing: {list(missing)}"
            )

        train = _stats(train_res, self.weights)
        validation = _stats(val_res, self.weights) if self.validation else None

        loss_value = self._loss(train, train_res)
        # Missing systems are charged against the training term only: the
        # held-out split is a diagnostic and must never steer the optimiser.
        n_missing_train = sum(1 for s in missing if s in set(self.training))
        if n_missing_train:
            loss_value += self.missing_penalty * n_missing_train

        penalty_value = self._penalty_value()
        return ObjectiveReport(
            value=loss_value + penalty_value,
            loss=self.loss,
            loss_value=loss_value,
            penalty_value=penalty_value,
            train=train,
            validation=validation,
            residuals=residuals,
            missing=missing,
            unchecked_conventions=self._unchecked(),
        )

    def __call__(self, results: Mapping[str, float] | Iterable[Any]) -> float:
        """The scalar an optimiser minimises."""
        return self.report(results).value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CohesiveObjective(n_systems={len(self.references)}, "
            f"loss={self.loss!r}, n_train={len(self.training)}, "
            f"n_validation={len(self.validation)})"
        )


def split_validation(
    systems: Sequence[str], *, every: int = 4, offset: int = 0
) -> tuple[str, ...]:
    """Deterministic held-out split: every ``every``-th system, from ``offset``.

    Deterministic on purpose. A random split makes two campaign runs
    incomparable, and the whole point of the held-out number is that it can
    be compared across candidates.
    """
    if every < 2:
        raise ObjectiveError(f"every must be >= 2, got {every}")
    if not 0 <= offset < every:
        raise ObjectiveError(f"offset must be in [0, {every}), got {offset}")
    return tuple(s for i, s in enumerate(sorted(systems)) if i % every == offset)
