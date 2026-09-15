"""ASE optimizer availability probing + SciPy-compat shim.

Two failure modes surfaced by the glycine optimizer SI matrix
(rp167-glyopt-*, 2026-07-01) when driving the :class:`vibeqc.ase.VibeQC`
calculator with ASE's molecular optimizers on a cluster deployment:

1. **Old ASE + new SciPy.** ASE < 3.23 imports
   ``scipy.integrate.cumtrapz``, which SciPy >= 1.14 removed (renamed to
   ``cumulative_trapezoid`` in SciPy 1.6). On such an environment
   ``from ase.optimize import BFGSLineSearch`` dies with an ImportError
   *after* the job has been queued and dispatched.
   :func:`ensure_ase_scipy_compat` restores the alias when (and only
   when) it is missing, so the old-ASE line-search optimizers keep
   working on a current SciPy.

2. **Optimizer classes that exist only in newer ASE.** ``FIRE2``,
   ``ODE12r``-as-top-level, and ``RFO`` are recent additions; a runner
   generated against current ASE docs fails on an older deployed ASE
   with a bare ModuleNotFoundError -- again only after dispatch.
   :func:`resolve_ase_optimizer` resolves a canonical optimizer name
   against the *active* ASE installation and fails early with a full
   availability report (:func:`format_ase_optimizer_report`), so batch
   generators can validate their matrix before submitting anything.

This module deliberately imports ASE lazily: importing
``vibeqc.ase_optimizers`` is safe in an environment without ASE.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Optional


def ensure_ase_scipy_compat() -> None:
    """Restore SciPy aliases that old ASE (< 3.23) still imports.

    SciPy 1.14 removed ``scipy.integrate.cumtrapz`` / ``trapz`` /
    ``simps`` (deprecated since 1.6 in favour of
    ``cumulative_trapezoid`` / ``trapezoid`` / ``simpson``). ASE 3.22.x
    still imports the old names from its line-search optimizers, so on
    an old-ASE + new-SciPy environment those optimizers raise
    ImportError at import time.

    This helper re-adds each alias **only when the attribute is missing
    and its replacement exists** -- on a SciPy that still ships the old
    names, or an environment without SciPy, it is a no-op. Call it
    before importing anything from ``ase.optimize``. Idempotent, cheap,
    and safe to call unconditionally.
    """
    try:
        import scipy.integrate as _integrate
    except ImportError:  # pragma: no cover - scipy is a hard vibe-qc dep
        return
    for old, new in (
        ("cumtrapz", "cumulative_trapezoid"),
        ("trapz", "trapezoid"),
        ("simps", "simpson"),
    ):
        if not hasattr(_integrate, old) and hasattr(_integrate, new):
            setattr(_integrate, old, getattr(_integrate, new))


@dataclass(frozen=True)
class ASEOptimizerSpec:
    """One canonical ASE optimizer vibe-qc knows how to locate.

    ``candidates`` is an ordered tuple of ``(module, class_name)``
    locations; the first one that imports wins. Multiple candidates
    cover ASE versions that moved a class between modules (e.g.
    ``FIRE2`` is ``ase.optimize.FIRE2`` on current ASE but only
    ``ase.optimize.fire2.FIRE2`` on the releases that introduced it).
    """

    key: str
    candidates: tuple[tuple[str, str], ...]
    kind: str
    needs_scipy_compat: bool = False
    notes: str = ""
    aliases: tuple[str, ...] = field(default=())


_LINESEARCH_NOTE = (
    "needs scipy.integrate.cumtrapz on ASE < 3.23; vibe-qc restores the "
    "alias via ensure_ase_scipy_compat()"
)

ASE_OPTIMIZERS: dict[str, ASEOptimizerSpec] = {
    spec.key: spec
    for spec in (
        ASEOptimizerSpec(
            key="bfgs",
            candidates=(("ase.optimize", "BFGS"),),
            kind="quasi-Newton",
        ),
        ASEOptimizerSpec(
            key="lbfgs",
            candidates=(("ase.optimize", "LBFGS"),),
            kind="quasi-Newton (limited memory)",
        ),
        ASEOptimizerSpec(
            key="bfgs-linesearch",
            candidates=(("ase.optimize", "BFGSLineSearch"),),
            kind="quasi-Newton + line search",
            needs_scipy_compat=True,
            notes=_LINESEARCH_NOTE,
            aliases=("bfgslinesearch", "quasinewton", "qn"),
        ),
        ASEOptimizerSpec(
            key="lbfgs-linesearch",
            candidates=(("ase.optimize", "LBFGSLineSearch"),),
            kind="quasi-Newton (limited memory) + line search",
            needs_scipy_compat=True,
            notes=_LINESEARCH_NOTE,
            aliases=("lbfgslinesearch",),
        ),
        ASEOptimizerSpec(
            key="fire",
            candidates=(("ase.optimize", "FIRE"),),
            kind="inertial MD relaxation",
        ),
        ASEOptimizerSpec(
            key="fire2",
            candidates=(
                ("ase.optimize", "FIRE2"),
                ("ase.optimize.fire2", "FIRE2"),
            ),
            kind="inertial MD relaxation (ABC variant)",
            notes="added in ASE 3.23; not available on ASE 3.22.x",
        ),
        ASEOptimizerSpec(
            key="gpmin",
            candidates=(("ase.optimize", "GPMin"),),
            kind="Gaussian-process surrogate",
            notes=(
                "surrogate updates can propose very large steps on "
                "expensive ab initio surfaces and drive the geometry "
                "into SCF-hostile regions; consider BFGS/LBFGS for "
                "DFT-quality providers, or bound the step"
            ),
        ),
        ASEOptimizerSpec(
            key="mdmin",
            candidates=(("ase.optimize", "MDMin"),),
            kind="velocity-Verlet quench",
            notes=(
                "the default dt=0.2 is tuned for cheap force fields and "
                "overshoots on ab initio surfaces (glycine SI matrix: "
                "SCF blow-up after 7 steps); pass a smaller dt, e.g. "
                "MDMin(atoms, dt=0.05)"
            ),
        ),
        ASEOptimizerSpec(
            key="goodoldquasinewton",
            candidates=(
                ("ase.optimize", "GoodOldQuasiNewton"),
                ("ase.optimize.oldqn", "GoodOldQuasiNewton"),
            ),
            kind="quasi-Newton (legacy)",
            aliases=("oldqn",),
        ),
        ASEOptimizerSpec(
            key="ode12r",
            candidates=(
                ("ase.optimize", "ODE12r"),
                ("ase.optimize.ode", "ODE12r"),
            ),
            kind="adaptive ODE relaxation",
        ),
        ASEOptimizerSpec(
            key="rfo",
            candidates=(("ase.optimize.rfo", "RFO"),),
            kind="rational function optimization",
            notes="recent ASE addition; not available on ASE 3.22.x",
        ),
    )
}

_ALIAS_TO_KEY: dict[str, str] = {}
for _spec in ASE_OPTIMIZERS.values():
    _ALIAS_TO_KEY[_spec.key] = _spec.key
    for _alias in _spec.aliases:
        _ALIAS_TO_KEY[_alias] = _spec.key


def normalize_ase_optimizer_name(name: str) -> str:
    """Normalize a user/runner-facing optimizer name to a registry key.

    Accepts the ``ase-`` prefix used by batch matrices
    (``"ase-bfgs-linesearch"``), any capitalisation, and the class-name
    spellings (``"BFGSLineSearch"``). Raises ValueError for names not
    in the registry, listing the known keys.
    """
    key = name.strip().lower()
    if key.startswith("ase-"):
        key = key[len("ase-"):]
    key = _ALIAS_TO_KEY.get(key) or _ALIAS_TO_KEY.get(key.replace("_", "-"))
    if key is None:
        raise ValueError(
            f"Unknown ASE optimizer {name!r}. Known optimizers: "
            f"{', '.join(sorted(ASE_OPTIMIZERS))}"
        )
    return key


@dataclass(frozen=True)
class ASEOptimizerAvailability:
    """Probe result for one registry entry against the active ASE."""

    key: str
    available: bool
    location: Optional[str]  # "module.Class" that resolved, if available
    reason: Optional[str]  # import failure summary, if unavailable
    notes: str


def _probe(spec: ASEOptimizerSpec) -> ASEOptimizerAvailability:
    if spec.needs_scipy_compat:
        ensure_ase_scipy_compat()
    last_error: Optional[str] = None
    for module_name, class_name in spec.candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        return ASEOptimizerAvailability(
            key=spec.key,
            available=True,
            location=f"{module_name}.{class_name}",
            reason=None,
            notes=spec.notes,
        )
    return ASEOptimizerAvailability(
        key=spec.key,
        available=False,
        location=None,
        reason=last_error or "no candidate location importable",
        notes=spec.notes,
    )


def ase_optimizer_availability() -> dict[str, ASEOptimizerAvailability]:
    """Probe every registry entry against the active ASE installation."""
    return {key: _probe(spec) for key, spec in ASE_OPTIMIZERS.items()}


def available_ase_optimizers() -> list[str]:
    """Registry keys that resolve in the active ASE installation."""
    return sorted(
        key for key, avail in ase_optimizer_availability().items() if avail.available
    )


def format_ase_optimizer_report() -> str:
    """Human-readable availability report for the active environment."""
    try:
        import ase

        header = f"ASE {ase.__version__} optimizer availability:"
    except ImportError:
        return "ASE is not installed -- no ASE optimizers are available."
    lines = [header]
    for key, avail in sorted(ase_optimizer_availability().items()):
        if avail.available:
            lines.append(f"  {key:<22s} OK   ({avail.location})")
        else:
            lines.append(f"  {key:<22s} MISSING  ({avail.reason})")
        if avail.notes:
            lines.append(f"  {'':<22s}   note: {avail.notes}")
    return "\n".join(lines)


def resolve_ase_optimizer(name: str) -> type:
    """Resolve an optimizer name to its class in the active ASE install.

    Raises ValueError for a name outside the registry and ImportError
    (with the full availability report) for a registry entry the
    installed ASE does not provide -- both *before* any calculation
    starts, so batch runners fail at generation/validation time rather
    than mid-queue.
    """
    key = normalize_ase_optimizer_name(name)
    spec = ASE_OPTIMIZERS[key]
    avail = _probe(spec)
    if not avail.available:
        raise ImportError(
            f"ASE optimizer {name!r} ({key}) is not available in the "
            f"installed ASE: {avail.reason}\n"
            + format_ase_optimizer_report()
        )
    module_name, class_name = avail.location.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)
