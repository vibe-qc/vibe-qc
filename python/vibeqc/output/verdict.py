"""One run-level verdict shared by every writer (GitLab #342, #346, #351).

The molecular SCF drivers return a complete stability verdict on the result
object: ``stability_checked``, ``stability_analysis_converged``,
``internal_instability`` and ``stability_eigenvalue`` (``cpp/include/vibeqc/
{rhf,rks,uhf,uks}.hpp``) and, since #342, why the stability solve ended
(``stability_stop_reason``), the Hessian-vector products it spent
(``stability_n_matvec``) and the wall budget that applied
(``stability_wall_budget_s``). Before this module the output layer projected
that verdict onto ``.out`` prose and one structured ``warning`` only, so a
certified-stable solution and one whose analysis produced no verdict were
byte-identical in ``.system`` (``outputs.status = "complete"``), in the QVF
(``provenance.run_status = "converged"``) and in the ``job_end`` event. A
campaign consumer therefore ingested every UNVERIFIED species as complete
(#346).

:class:`RunVerdict` is built **once** per run from the driver result by
:func:`verdict_from_result` and carried unchanged by

* ``.system``: ``[outputs].result_status`` and the fixed-shape ``[verdict]``
  section (:mod:`vibeqc.output.manifest`);
* the QVF: ``provenance.verdict`` (:mod:`vibeqc.output.formats.qvf`);
* the structured log: the ``scf_converged`` and ``job_end`` events;
* the ``.out``: the ``internal stability:`` line
  (:func:`describe_stability_stop`).

``usable`` is the single consumer gate: the SCF converged and the stability
analysis neither produced no verdict nor found an instability. ``not_checked``
(explicit opt-out), ``unsupported`` (a route whose response is not plumbed),
``not_applicable`` (no SCF stability surface) and ``not_reported`` (a writer
that recorded no stability verdict) stay visible but do not block use; a
stricter consumer gates on ``stability == "verified_stable"``.

Convergence and stability are the driver's numerical verdicts. Neither
establishes external validation of the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "NOT_USABLE_STABILITY",
    "RunVerdict",
    "SCF_STATES",
    "STABILITY_STATES",
    "STOP_REASONS",
    "describe_stability_stop",
    "usable_from_manifest",
    "verdict_from_result",
    "verdict_section",
]

SCF_STATES = ("converged", "not_converged", "not_run", "not_reported")

STABILITY_STATES = (
    "verified_stable",
    "unstable",
    "unverified",
    "not_checked",
    "unsupported",
    "not_applicable",
    "not_reported",
)

STOP_REASONS = (
    "converged",
    "matvec_budget",
    "wall_budget",
    "not_run",
    "trivial",
    "stagnated",
    "opt_out",
    "unsupported_route",
    "unreported",
    "none",
)

# ``vibeqc::StabilityStop`` (cpp/include/vibeqc/newton.hpp) as the drivers
# expose it on ``stability_stop_reason``.
_NATIVE_STOP_REASONS = {
    0: "converged",
    1: "matvec_budget",
    2: "wall_budget",
    3: "not_run",
    4: "trivial",
    5: "stagnated",
}

# The two honest-negative stability states. Everything else is either a
# certified verdict or a visible absence of one that the caller chose or the
# route documents.
NOT_USABLE_STABILITY = frozenset({"unverified", "unstable"})

_RESULT_STATUS_FOR_SCF = {
    "converged": "converged",
    "not_converged": "not_converged",
    "not_run": "not_run",
    "not_reported": "not_reported",
}

_SCF_FOR_RESULT_STATUS = {
    "converged": "converged",
    "not_converged": "not_converged",
    "not_run": "not_run",
}


@dataclass(frozen=True)
class RunVerdict:
    """The verdict of one run, in the vocabulary every writer shares.

    ``stability_eigenvalue`` is the lowest orbital-rotation Hessian
    eigenvalue at the returned solution. It is a verdict only when
    ``stability`` is ``verified_stable`` or ``unstable``; for ``unverified``
    it is the solver's last estimate, and otherwise it is ``0.0``.
    ``stability_wall_budget_s == 0.0`` means the solve was unbounded.
    """

    scf: str
    stability: str
    stability_reason: str
    stability_eigenvalue: float = 0.0
    stability_restarts: int = 0
    stability_matvec_used: int = 0
    stability_matvec_cap: int = 0
    stability_wall_s: float = 0.0
    stability_wall_budget_s: float = 0.0

    def __post_init__(self) -> None:
        if self.scf not in SCF_STATES:
            raise ValueError(f"unknown scf verdict: {self.scf!r}")
        if self.stability not in STABILITY_STATES:
            raise ValueError(f"unknown stability verdict: {self.stability!r}")
        if self.stability_reason not in STOP_REASONS:
            raise ValueError(
                f"unknown stability stop reason: {self.stability_reason!r}"
            )

    @property
    def usable(self) -> bool:
        """The single consumer gate (maintainer decision D3, 2026-09-29)."""
        return (
            self.scf == "converged"
            and self.stability not in NOT_USABLE_STABILITY
        )

    @property
    def result_status(self) -> str:
        """The ``[outputs].result_status`` value this verdict implies."""
        return _RESULT_STATUS_FOR_SCF[self.scf]

    def as_dict(self) -> dict[str, Any]:
        """Fixed-shape, TOML/JSON-native mapping. Every key is always present."""
        return {
            "scf": self.scf,
            "stability": self.stability,
            "stability_reason": self.stability_reason,
            "usable": bool(self.usable),
            "stability_eigenvalue": float(self.stability_eigenvalue),
            "stability_restarts": int(self.stability_restarts),
            "stability_matvec_used": int(self.stability_matvec_used),
            "stability_matvec_cap": int(self.stability_matvec_cap),
            "stability_wall_s": float(self.stability_wall_s),
            "stability_wall_budget_s": float(self.stability_wall_budget_s),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunVerdict":
        """Rebuild from :meth:`as_dict` output (``usable`` is recomputed)."""
        return cls(
            scf=str(data["scf"]),
            stability=str(data["stability"]),
            stability_reason=str(data.get("stability_reason", "none")),
            stability_eigenvalue=float(data.get("stability_eigenvalue", 0.0)),
            stability_restarts=int(data.get("stability_restarts", 0)),
            stability_matvec_used=int(data.get("stability_matvec_used", 0)),
            stability_matvec_cap=int(data.get("stability_matvec_cap", 0)),
            stability_wall_s=float(data.get("stability_wall_s", 0.0)),
            stability_wall_budget_s=float(
                data.get("stability_wall_budget_s", 0.0)
            ),
        )

    @classmethod
    def unreported(cls, scf: str = "not_reported") -> "RunVerdict":
        """A writer that recorded no stability verdict (e.g. periodic routes)."""
        return cls(scf=scf, stability="not_reported", stability_reason="none")


def _native_stop_reason(result: Any) -> str | None:
    raw = getattr(result, "stability_stop_reason", None)
    if raw is None:
        return None
    try:
        return _NATIVE_STOP_REASONS.get(int(raw), "unreported")
    except (TypeError, ValueError):
        return "unreported"


def verdict_from_result(
    result: Any,
    *,
    stability_requested: bool | None = None,
    matvec_cap: int | None = None,
) -> RunVerdict:
    """Build the verdict once from a driver result.

    ``stability_requested`` is the driver option ``stability_check`` when
    the caller has it: an analysis that did not run is then an explicit
    opt-out (``not_checked``) or a route the driver skipped because its
    response is not plumbed (``unsupported``). A result without any
    stability surface (post-HF solvers, MLIPs, ...) is ``not_applicable``.
    Any result-object shape is accepted; missing fields degrade to the
    honest absence, never to a certificate.
    """
    converged = getattr(result, "converged", None)
    if converged is None:
        scf = "not_reported"
    else:
        scf = "converged" if bool(converged) else "not_converged"

    checked = getattr(result, "stability_checked", None)
    eigenvalue = 0.0
    restarts = int(getattr(result, "n_stability_restarts", 0) or 0)
    matvec_used = int(getattr(result, "stability_n_matvec", 0) or 0)
    wall_s = float(getattr(result, "stability_wall_s", 0.0) or 0.0)
    budget_s = float(getattr(result, "stability_wall_budget_s", 0.0) or 0.0)

    if checked is None:
        stability, reason = "not_applicable", "none"
    elif not checked:
        if scf != "converged":
            # The drivers run the analysis only on a converged SCF.
            stability, reason = "not_applicable", "none"
        elif stability_requested is False:
            stability, reason = "not_checked", "opt_out"
        else:
            stability, reason = "unsupported", "unsupported_route"
    else:
        eigenvalue = float(getattr(result, "stability_eigenvalue", 0.0) or 0.0)
        native = _native_stop_reason(result)
        if not bool(getattr(result, "stability_analysis_converged", False)):
            stability = "unverified"
            reason = native if native not in (None, "converged") else "unreported"
        elif bool(getattr(result, "internal_instability", False)):
            stability = "unstable"
            reason = "converged"
        else:
            stability = "verified_stable"
            reason = "converged"

    return RunVerdict(
        scf=scf,
        stability=stability,
        stability_reason=reason,
        stability_eigenvalue=eigenvalue,
        stability_restarts=restarts,
        stability_matvec_used=matvec_used,
        stability_matvec_cap=int(matvec_cap or 0),
        stability_wall_s=wall_s,
        stability_wall_budget_s=budget_s,
    )


def describe_stability_stop(result: Any) -> str:
    """Prose for the ``.out`` line of a solve that produced no verdict."""
    reason = _native_stop_reason(result)
    n = int(getattr(result, "stability_n_matvec", 0) or 0)
    wall = float(getattr(result, "stability_wall_s", 0.0) or 0.0)
    budget = float(getattr(result, "stability_wall_budget_s", 0.0) or 0.0)
    if reason == "not_run":
        return "did not run (matvec cap 0)"
    if reason == "matvec_budget":
        return f"stopped at the matvec cap after {n} Hessian-vector products"
    if reason == "wall_budget":
        return (
            f"stopped by the wall budget after {n} Hessian-vector products, "
            f"{wall:.1f} s of {budget:.1f} s"
        )
    if reason == "stagnated":
        return f"stagnated after {n} Hessian-vector products"
    return "did not converge"


def verdict_section(
    verdict: Mapping[str, Any] | None, result_status: str
) -> dict[str, Any]:
    """The ``[verdict]`` body: the recorded verdict, else one derived from
    ``result_status`` with ``stability = "not_reported"``."""
    if verdict is not None:
        return RunVerdict.from_dict(verdict).as_dict()
    scf = _SCF_FOR_RESULT_STATUS.get(str(result_status), "not_reported")
    return RunVerdict.unreported(scf).as_dict()


def usable_from_manifest(manifest: Mapping[str, Any]) -> str:
    """The reference consumer gate over a parsed ``.system`` manifest.

    Returns ``"usable"``, ``"not_usable"`` or ``"unknown"``. A manifest
    without ``[verdict]`` or without a recorded ``result_status`` (every
    manifest written before this contract, and every ``not_reported`` run)
    is ``unknown``: the gate fails closed on history instead of reading
    ``outputs.status`` -- a delivery flag -- as a scientific verdict.
    """
    outputs = manifest.get("outputs") or {}
    result_status = outputs.get("result_status")
    body = manifest.get("verdict")
    if body is None or result_status in (None, "", "not_reported"):
        return "unknown"
    try:
        verdict = RunVerdict.from_dict(body)
    except (KeyError, TypeError, ValueError):
        return "unknown"
    if result_status == "converged" and verdict.usable:
        return "usable"
    return "not_usable"
