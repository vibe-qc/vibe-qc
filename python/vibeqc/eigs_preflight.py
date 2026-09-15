"""EIGS-equivalent standalone overlap-matrix diagnostic.

CRYSTAL has a keyword called ``EIGS`` (manual p. 103, 398) that runs
input parsing -> integrals -> S(k) construction at every k in the
irreducible Brillouin zone -> eigenvalue decomposition -> print ->
**halt without running SCF**. The whole point is to let the user
verify the basis is suitable BEFORE committing to an expensive SCF
run, and to flag the case loudly when it isn't:

    Negative values indicate numerical linear dependence. The
    crystal program stops after the check (even if negative
    eigenvalues are not detected).

vibe-qc takes the same stance -- refuse to silently truncate
near-singular S directions and let the user run an SCF on a basis
with the wrong metric signature. The
:func:`vibeqc.scf_preflight_overlap_check` integrated into the eight
SCF drivers (committed in ``bbe2b2a``) implements the runtime
abort. This module adds the *standalone* diagnostic -- the user-
invocable

.. code-block:: python

    import vibeqc as vq

    report = vq.eigs_preflight(system, basis, kmesh)
    print(vq.format_eigs_report(report))

run before any SCF, so the basis-set conditioning check happens at a
controlled point with no expensive computation past the integrals.

What it does
------------

* Builds the lattice-sum overlap ``S_lat`` once.
* Bloch-sums to S(k) at every k-point requested.
* Diagonalises each S(k) via :func:`vibeqc.check_overlap_matrix`.
* Returns an :class:`EIGSReport` collecting per-k reports + a
  summary worst-case severity.
* No SCF is run. No integrals other than overlap are computed.

Recommended use
---------------

Run this whenever:

* Switching to a new basis set, especially a molecular-design one
  (def2, cc-pVXZ, 6-31G) on a periodic system.
* Adjusting the geometry to tighter / denser packing.
* Increasing the k-mesh density (a basis that's clean at Γ can fail
  at the zone boundary; Searle, Bernasconi, Harrison ARCHER
  eCSE04-16, 2017).
* Before publishing benchmark numbers -- establishes that the SCF
  result didn't ride on a silent canonical-orthogonalisation
  truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
    bloch_sum,
    compute_overlap_lattice,
)
from .linear_dependence import (
    DEFAULT_ERROR_THRESHOLD,
    DEFAULT_NEGATIVE_THRESHOLD,
    DEFAULT_WARN_THRESHOLD,
    LinearDependenceReport,
    check_overlap_matrix,
)


__all__ = [
    "EIGSReport",
    "eigs_preflight",
    "format_eigs_report",
    "DisambiguationReport",
    "disambiguate_critical_overlap",
    "format_disambiguation_report",
    "TruncationOptimizationReport",
    "optimize_truncation",
    "format_truncation_optimization_report",
]


# ---------------------------------------------------------------------------
# Severity ordering for the worst-case reduction.
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"ok": 0, "warn": 1, "error": 2, "critical": 3}


def _worst_severity(severities: Iterable[str]) -> str:
    seen = list(severities)
    if not seen:
        return "ok"
    return max(seen, key=lambda s: _SEVERITY_RANK.get(s, 0))


# ---------------------------------------------------------------------------
# Report struct
# ---------------------------------------------------------------------------

@dataclass
class DisambiguationReport:
    """Diagnosis output for :func:`disambiguate_critical_overlap`.

    Distinguishes the two upstream causes of a non-PSD overlap
    matrix that CRYSTAL's manual (p. 130, 398) flags as easy to
    confuse:

    * **basis_set_problem** -- the basis genuinely has too many
      diffuse primitives for the lattice geometry. Tightening
      lattice / Schwarz cutoffs doesn't help; the only fixes are at
      basis-set design time (use pob-tzvp / MOLOPT / GTH-cc-pVXZ)
      or via the runtime ``vq.make_basis(..., exp_to_discard=...)``
      filter.
    * **screening_undertight** -- the negative eigenvalues are an
      artefact of too-loose lattice-sum truncation or ERI screening
      (CRYSTAL's ITOL4 / ITOL5 in the TOLINTEG block). Tightening
      ``LatticeSumOptions.cutoff_bohr`` and / or
      ``LatticeSumOptions.schwarz_threshold`` makes the issue go
      away. Cheaper than swapping the basis.
    * **inconclusive** -- couldn't tell. Try both fixes; basis-set
      design first (it's a strict superset of the screening fix).

    The CRYSTAL manual quote that motivates this:

        the "pseudoverlap" criteria associated with the two
        computational parameters ITOL4 and ITOL5 mimic only in an
        approximate way the real behaviour of the density matrix.

    PySCF / molecular codes don't surface this distinction -- they
    just see "non-PSD S, drop the negative directions, run SCF".
    Vibe-qc surfaces it explicitly.
    """

    verdict: str   # "basis_set_problem" | "screening_undertight" | "inconclusive"
    original_min_eigenvalue: float
    tightened_min_eigenvalue: float
    original_n_negative: int
    tightened_n_negative: int
    cutoff_tighten_factor: float
    schwarz_tighten_factor: float
    original_severity: str
    tightened_severity: str
    rationale: str = ""


@dataclass
class EIGSReport:
    """Collected per-k overlap-matrix diagnostics."""

    n_basis: int
    n_kpoints: int
    k_points_cart: np.ndarray                  # (n_k, 3)
    per_k_reports: List[LinearDependenceReport] = field(default_factory=list)
    worst_severity: str = "ok"
    worst_min_eigenvalue: float = float("inf")
    worst_condition_number: float = 0.0
    warn_threshold: float = DEFAULT_WARN_THRESHOLD
    error_threshold: float = DEFAULT_ERROR_THRESHOLD
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD

    @property
    def severity(self) -> str:
        return self.worst_severity

    def is_safe_to_run_scf(self, *, allow_warn: bool = True) -> bool:
        """Convenience predicate for the typical pre-SCF check.

        Returns ``True`` if the worst k-point severity is at or below
        the user's tolerance. Default policy matches the SCF
        preflight's default (``raise_if_severe`` semantics):

        * ``ok``       -> safe.
        * ``warn``     -> safe iff ``allow_warn`` (default True).
        * ``error``    -> not safe.
        * ``critical`` -> not safe (S has lost positive-definiteness).
        """
        sev = self.worst_severity
        if sev == "ok":
            return True
        if sev == "warn":
            return allow_warn
        return False


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def eigs_preflight(
    system: PeriodicSystem,
    basis: BasisSet,
    k_points_cart: Optional[Sequence[Sequence[float]]] = None,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
) -> EIGSReport:
    """Stand-alone overlap-matrix diagnostic -- CRYSTAL's EIGS keyword.

    Builds the periodic AO overlap, Bloch-sums it at every requested
    k-point, eigenvalue-decomposes each ``S(k)``, and returns a
    structured per-k + summary report. **No SCF is run.**

    Parameters
    ----------
    system
        Periodic system.
    basis
        AO basis. For periodic SCF, the right thing to filter through
        ``vq.make_basis(..., exp_to_discard=...)`` if a prior
        EIGS run flagged critical severity.
    k_points_cart
        ``(n_k, 3)`` Cartesian k-points (1/bohr) to check. If
        ``None`` (default), only Γ = (0, 0, 0) is checked. Pass a
        Monkhorst-Pack mesh's Cartesian coordinates for the full
        IBZ scan (CRYSTAL's default).
    lattice_opts
        :class:`LatticeSumOptions`; defaults to
        ``LatticeSumOptions()`` if not provided.
    warn_threshold, error_threshold, negative_threshold
        Eigenvalue thresholds for severity classification. See
        :mod:`vibeqc.linear_dependence`.

    Returns
    -------
    :class:`EIGSReport`.
    """
    lat_opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()

    if k_points_cart is None:
        k_arr = np.zeros((1, 3), dtype=float)
    else:
        k_arr = np.asarray(k_points_cart, dtype=float).reshape(-1, 3)

    # Build the lattice-summed overlap ONCE; Bloch-summation is cheap.
    S_lat = compute_overlap_lattice(basis, system, lat_opts)

    per_k: List[LinearDependenceReport] = []
    worst_sev = "ok"
    worst_min_eig = float("inf")
    worst_cond = 0.0

    for k_idx, k in enumerate(k_arr):
        S_k = np.asarray(bloch_sum(S_lat, k))
        # Hermitise -- Bloch summation can leave ~1e-15 imaginary drift.
        S_k = 0.5 * (S_k + S_k.conj().T)
        rep = check_overlap_matrix(
            S_k,
            basis=basis,
            warn_threshold=warn_threshold,
            error_threshold=error_threshold,
            negative_threshold=negative_threshold,
            label=f"S(k={k_idx}, k_cart={k.round(4).tolist()})",
        )
        per_k.append(rep)
        if _SEVERITY_RANK[rep.severity] > _SEVERITY_RANK[worst_sev]:
            worst_sev = rep.severity
        if rep.min_eigenvalue < worst_min_eig:
            worst_min_eig = rep.min_eigenvalue
        if (
            np.isfinite(rep.condition_number)
            and rep.condition_number > worst_cond
        ):
            worst_cond = rep.condition_number
        elif not np.isfinite(rep.condition_number):
            worst_cond = float("inf")

    return EIGSReport(
        n_basis=basis.nbasis,
        n_kpoints=k_arr.shape[0],
        k_points_cart=k_arr,
        per_k_reports=per_k,
        worst_severity=worst_sev,
        worst_min_eigenvalue=float(worst_min_eig),
        worst_condition_number=float(worst_cond),
        warn_threshold=warn_threshold,
        error_threshold=error_threshold,
        negative_threshold=negative_threshold,
    )


# ---------------------------------------------------------------------------
# Item 7: TOLINTEG-style disambiguation
# ---------------------------------------------------------------------------

def disambiguate_critical_overlap(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    lattice_opts: Optional[LatticeSumOptions] = None,
    k_points_cart: Optional[Sequence[Sequence[float]]] = None,
    cutoff_tighten_factor: float = 2.0,
    schwarz_tighten_factor: float = 100.0,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
) -> DisambiguationReport:
    """Distinguish a basis-set linear-dependence problem from an
    under-converged exchange-screening problem on a non-PSD overlap.

    Strategy: run :func:`eigs_preflight` twice -- first with the user's
    settings, then with significantly tightened lattice / Schwarz
    cutoffs. Compare the worst-case minimum eigenvalue across k-points.

    * If the negative eigenvalues **persist** at roughly the same
      magnitude after tightening -> basis-set problem.
    * If the negative eigenvalues **vanish** or shrink by orders of
      magnitude -> screening-undertight problem.
    * If results are mixed (some k-points improve, others don't) ->
      inconclusive; treat as basis-set problem because the screening
      fix is a strict subset of the basis-set fix.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis to test.
    lattice_opts
        :class:`LatticeSumOptions` to start from. Defaults to a fresh
        instance.
    k_points_cart
        K-points to scan. Default Γ only.
    cutoff_tighten_factor
        Multiplier on ``lat_opts.cutoff_bohr`` and
        ``lat_opts.nuclear_cutoff_bohr`` for the second pass.
        Default 2.0 (sums roughly 8x more lattice cells).
    schwarz_tighten_factor
        Divisor on ``lat_opts.schwarz_threshold`` for the second
        pass. Default 100.0 (one hundred times tighter screening).
    negative_threshold
        Eigenvalue below which is "negative" (default -1e-6).

    Returns
    -------
    :class:`DisambiguationReport`.
    """
    base_opts = lattice_opts if lattice_opts is not None else LatticeSumOptions()

    # Pass 1: user-as-given settings.
    rep_orig = eigs_preflight(
        system, basis, k_points_cart,
        lattice_opts=base_opts,
        negative_threshold=negative_threshold,
    )

    # Pass 2: tightened settings. Make a fresh LatticeSumOptions
    # copying every field so we don't mutate the user's object.
    tight_opts = LatticeSumOptions()
    for attr in dir(base_opts):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(base_opts, attr)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            setattr(tight_opts, attr, value)
        except Exception:
            pass
    tight_opts.cutoff_bohr = (
        float(base_opts.cutoff_bohr) * float(cutoff_tighten_factor)
    )
    tight_opts.nuclear_cutoff_bohr = (
        float(base_opts.nuclear_cutoff_bohr) * float(cutoff_tighten_factor)
    )
    # Tightening Schwarz means SMALLER threshold (keep more integrals).
    if schwarz_tighten_factor > 0:
        tight_opts.schwarz_threshold = (
            float(base_opts.schwarz_threshold) / float(schwarz_tighten_factor)
        )

    rep_tight = eigs_preflight(
        system, basis, k_points_cart,
        lattice_opts=tight_opts,
        negative_threshold=negative_threshold,
    )

    # Diagnose by comparing the worst min-eigenvalue across passes.
    orig_min = rep_orig.worst_min_eigenvalue
    tight_min = rep_tight.worst_min_eigenvalue
    orig_n_neg = sum(r.n_negative for r in rep_orig.per_k_reports)
    tight_n_neg = sum(r.n_negative for r in rep_tight.per_k_reports)

    if rep_tight.severity in ("ok", "warn"):
        verdict = "screening_undertight"
        rationale = (
            f"Tightening cutoff x{cutoff_tighten_factor:g} and Schwarz "
            f"÷{schwarz_tighten_factor:g} brought severity from "
            f"'{rep_orig.severity}' to '{rep_tight.severity}'. The original "
            "non-PSD overlap was a screening artefact, not a basis-set "
            "problem. Recommended fix: tighten LatticeSumOptions."
            "cutoff_bohr / nuclear_cutoff_bohr / schwarz_threshold "
            "permanently for this geometry. CRYSTAL's TOLINTEG ITOL4/ITOL5 "
            "play the same role (manual p. 130, 398)."
        )
    elif rep_tight.severity == "critical" and tight_n_neg >= orig_n_neg * 0.8:
        # Negatives persist -- basis-set problem.
        verdict = "basis_set_problem"
        rationale = (
            f"Tightening cutoffs x{cutoff_tighten_factor:g} did not improve "
            f"PSD-ness: worst min eigenvalue {orig_min:+.3e} -> "
            f"{tight_min:+.3e}, n_negative {orig_n_neg} -> {tight_n_neg}. "
            "The negative eigenvalues are intrinsic to the basis on this "
            "lattice geometry. Switch to a basis designed for solids "
            "(pob-tzvp / pob-tzvp-rev2 / MOLOPT / GTH-cc-pVXZ) or filter "
            "diffuse primitives via vq.make_basis(..., "
            "exp_to_discard=0.1)."
        )
    else:
        # Severity dropped from critical to error/warn but not all the way:
        # mixed evidence.
        verdict = "inconclusive"
        rationale = (
            f"Tightening helped but didn't fully fix the overlap: severity "
            f"{rep_orig.severity} -> {rep_tight.severity}, worst min "
            f"eigenvalue {orig_min:+.3e} -> {tight_min:+.3e}, n_negative "
            f"{orig_n_neg} -> {tight_n_neg}. There's likely both a "
            "screening AND a basis-set component. Apply the basis-set fix "
            "first (it's strictly a superset of the screening fix): "
            "switch to pob-tzvp / pob-tzvp-rev2 or use "
            "vq.make_basis(..., exp_to_discard=0.1)."
        )

    return DisambiguationReport(
        verdict=verdict,
        original_min_eigenvalue=float(orig_min),
        tightened_min_eigenvalue=float(tight_min),
        original_n_negative=int(orig_n_neg),
        tightened_n_negative=int(tight_n_neg),
        cutoff_tighten_factor=float(cutoff_tighten_factor),
        schwarz_tighten_factor=float(schwarz_tighten_factor),
        original_severity=rep_orig.severity,
        tightened_severity=rep_tight.severity,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# optimize_truncation: bisect lattice-cutoff / Schwarz to find the LOOSEST
# settings that keep S positive-semi-definite at every k-point.
# ---------------------------------------------------------------------------

@dataclass
class TruncationOptimizationReport:
    """Outcome of :func:`optimize_truncation`.

    ``optimized_lattice_opts`` is the new :class:`LatticeSumOptions`
    you should hand to your SCF driver. The other fields document
    how it was arrived at -- useful for caching across SCF restarts
    and for logging.
    """

    optimized_lattice_opts: "LatticeSumOptions"
    starting_lattice_opts: "LatticeSumOptions"
    n_evaluations: int                   # # of preflight passes run
    final_severity: str                  # "ok" / "warn" / "error" / "critical"
    final_min_eigenvalue: float
    final_n_negative: int
    converged: bool                      # could we land at a PSD setting?
    cutoff_bohr_path: List[float] = field(default_factory=list)
    nuclear_cutoff_bohr_path: List[float] = field(default_factory=list)
    schwarz_threshold_path: List[float] = field(default_factory=list)
    notes: str = ""


def _copy_lattice_opts(src: "LatticeSumOptions") -> "LatticeSumOptions":
    """Make a deep-ish copy of a LatticeSumOptions struct.

    Pybind structs don't support ``copy.copy`` directly; walk public
    attributes and assign one by one. Skips read-only / callable
    properties.
    """
    dst = LatticeSumOptions()
    for attr in dir(src):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(src, attr)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            setattr(dst, attr, value)
        except Exception:
            pass
    return dst


def optimize_truncation(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    lattice_opts: Optional["LatticeSumOptions"] = None,
    k_points_cart: Optional[Sequence[Sequence[float]]] = None,
    target_severity: str = "ok",
    cutoff_growth_factor: float = 1.25,
    cutoff_max_bohr: float = 80.0,
    schwarz_tighten_factor: float = 100.0,
    schwarz_min: float = 1e-18,
    max_evaluations: int = 8,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    joint_growth: bool = True,
) -> TruncationOptimizationReport:
    """Find the LOOSEST lattice-cutoff / Schwarz settings that keep
    the AO overlap matrix positive-semi-definite at every requested
    k-point.

    Strategy (cheap, deterministic):

    1. Run a preflight at the user's starting settings.
    2. If severity is already at or below ``target_severity``, return
       those settings unchanged.
    3. Otherwise, **grow** ``cutoff_bohr`` and ``nuclear_cutoff_bohr``
       by ``cutoff_growth_factor`` each step. Once the preflight
       lands at the target severity, optionally tighten
       ``schwarz_threshold`` by ``schwarz_tighten_factor`` to see if
       cutoffs can come back down. Return the loosest combination
       that still satisfies the target.

    This is cheaper than a full grid search and usually adequate:
    eigenvalues are monotonic in the lattice cutoff (more cells ->
    more contributions -> more PSD), so we can grow until passing
    and then tighten the second knob in a second sweep.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis.
    lattice_opts
        Starting :class:`LatticeSumOptions`. Defaults to a fresh one.
        Will not be mutated.
    k_points_cart
        K-points to sweep across. Defaults to Γ only.
    target_severity
        Stop once the worst k-point hits this severity. ``"ok"``
        (default) requires no near-singular eigenvalues anywhere;
        ``"warn"`` accepts mild near-singularity (still no negatives);
        ``"error"`` accepts real near-singularity (still no
        negatives -- those are the
        ``"critical"`` tier). PSD-ness is the floor -- we always
        require ``n_negative == 0``.
    cutoff_growth_factor
        Multiplier on ``cutoff_bohr`` / ``nuclear_cutoff_bohr`` per
        growth step. ``1.25`` is moderate (6 steps from 10 -> ~38); use ``1.5`` for speed
        ``1.25`` for a finer grid at the cost of more evaluations.
    cutoff_max_bohr
        Hard ceiling on ``cutoff_bohr``. The function gives up if
        growth doesn't reach the target severity below this cap and
        sets ``converged = False``.
    schwarz_tighten_factor
        Per-step tightening factor on ``schwarz_threshold``. With
        ``joint_growth=True`` (default), both knobs are tightened
        together at every step so neither knob ends up far from its
        starting value while the other stays put. With
        ``joint_growth=False`` only cutoffs grow in phase 1, then
        Schwarz tightens in phase 2 (legacy behaviour).
    schwarz_min
        Floor on ``schwarz_threshold`` (default 1e-18). Prevents
        runaway tightening into denormalised territory.
    max_evaluations
        Cap on total preflight passes. Default 8 -- cheap enough that
        screening cost stays << SCF cost on any realistic system.
    joint_growth
        When True (default) cutoff_bohr x cutoff_growth_factor AND
        schwarz_threshold ÷ schwarz_tighten_factor at EVERY step,
        keeping both knobs proportional. When False, grow cutoff
        only in phase 1, tighten Schwarz only in phase 2 -- looser
        coupling between knobs but two-phase strategy.

    Returns
    -------
    :class:`TruncationOptimizationReport`.

    See also
    --------
    :func:`disambiguate_critical_overlap` -- diagnoses whether the
    fix is screening (cutoffs) or basis (pob / exp_to_discard).
    Run that *first* to confirm that the optimisation is on the
    right track; this function assumes the issue is screening.
    """
    if target_severity not in ("ok", "warn", "error"):
        raise ValueError(
            "optimize_truncation: target_severity must be one of "
            f"'ok' / 'warn' / 'error'; got {target_severity!r}. "
            "(critical is the abort tier; can't be a target.)"
        )

    base = lattice_opts if lattice_opts is not None else LatticeSumOptions()
    starting = _copy_lattice_opts(base)
    current = _copy_lattice_opts(base)

    target_rank = _SEVERITY_RANK[target_severity]
    cutoff_path: List[float] = []
    nuclear_cutoff_path: List[float] = []
    schwarz_path: List[float] = []
    n_evals = 0

    def _evaluate(opts: "LatticeSumOptions") -> EIGSReport:
        nonlocal n_evals
        n_evals += 1
        cutoff_path.append(float(opts.cutoff_bohr))
        nuclear_cutoff_path.append(float(opts.nuclear_cutoff_bohr))
        schwarz_path.append(float(opts.schwarz_threshold))
        return eigs_preflight(
            system, basis, k_points_cart,
            lattice_opts=opts,
            negative_threshold=negative_threshold,
        )

    def _meets_target(rep: EIGSReport) -> bool:
        sev = rep.severity
        if _SEVERITY_RANK[sev] > target_rank:
            return False
        # PSD is the floor regardless of target severity.
        worst_neg = sum(r.n_negative for r in rep.per_k_reports)
        return worst_neg == 0

    # ---- Phase 0: starting settings ----------------------------------
    rep = _evaluate(current)
    if _meets_target(rep):
        return TruncationOptimizationReport(
            optimized_lattice_opts=_copy_lattice_opts(current),
            starting_lattice_opts=starting,
            n_evaluations=n_evals,
            final_severity=rep.severity,
            final_min_eigenvalue=rep.worst_min_eigenvalue,
            final_n_negative=sum(r.n_negative for r in rep.per_k_reports),
            converged=True,
            cutoff_bohr_path=cutoff_path,
            nuclear_cutoff_bohr_path=nuclear_cutoff_path,
            schwarz_threshold_path=schwarz_path,
            notes=(
                "Starting settings already met target severity "
                f"({rep.severity}); no optimisation needed."
            ),
        )

    # ---- Phase 1: grow cutoffs (and Schwarz, if joint) -------------
    # User-facing rationale: keeping the two knobs proportional
    # avoids the "one knob far from default, other at default"
    # failure mode where the optimisation introduces large
    # screening errors in one direction.
    last_passing: Optional["LatticeSumOptions"] = None
    last_passing_rep: Optional[EIGSReport] = None

    while n_evals < max_evaluations:
        new_cutoff = float(current.cutoff_bohr) * cutoff_growth_factor
        new_ncutoff = float(current.nuclear_cutoff_bohr) * cutoff_growth_factor
        if joint_growth and schwarz_tighten_factor > 1.0:
            new_schwarz = (
                float(current.schwarz_threshold) / schwarz_tighten_factor
            )
        else:
            new_schwarz = float(current.schwarz_threshold)

        # Sanity-check ranges: bail if either knob hits its limit.
        if new_cutoff > cutoff_max_bohr:
            break
        if new_schwarz < schwarz_min:
            break

        current.cutoff_bohr = new_cutoff
        current.nuclear_cutoff_bohr = new_ncutoff
        if joint_growth:
            current.schwarz_threshold = new_schwarz
        rep = _evaluate(current)
        if _meets_target(rep):
            last_passing = _copy_lattice_opts(current)
            last_passing_rep = rep
            break

    if last_passing is None:
        # Couldn't reach target severity below the ceiling. Return the
        # last evaluated settings; tag converged=False.
        return TruncationOptimizationReport(
            optimized_lattice_opts=_copy_lattice_opts(current),
            starting_lattice_opts=starting,
            n_evaluations=n_evals,
            final_severity=rep.severity,
            final_min_eigenvalue=rep.worst_min_eigenvalue,
            final_n_negative=sum(r.n_negative for r in rep.per_k_reports),
            converged=False,
            cutoff_bohr_path=cutoff_path,
            nuclear_cutoff_bohr_path=nuclear_cutoff_path,
            schwarz_threshold_path=schwarz_path,
            notes=(
                f"Cutoff growth did not reach target_severity={target_severity} "
                f"below cutoff_max_bohr={cutoff_max_bohr}. "
                "This is unlikely to be a screening problem alone -- try "
                "vq.disambiguate_critical_overlap and/or vq.make_basis(..., "
                "exp_to_discard=0.1) for a basis-set fix."
            ),
        )

    # ---- Phase 2: try to push cutoffs back down by tightening Schwarz
    # Joint growth moves both knobs proportionally in Phase 1, which
    # often overshoots the cutoff (e.g. 12->18 bohr when 15 would
    # suffice).  Phase 2 bisects the cutoff back down using the
    # already-tightened Schwarz from Phase 1, clawing back ~5x Fock
    # cost on lightweight benchmarks (v0.11.x fix).
    optimized = _copy_lattice_opts(last_passing)
    notes = (
        f"Joint growth: cutoff {starting.cutoff_bohr:.2f} -> "
        f"{optimized.cutoff_bohr:.2f} bohr (x"
        f"{optimized.cutoff_bohr / starting.cutoff_bohr:.2g}), "
        f"schwarz {starting.schwarz_threshold:.0e} -> "
        f"{optimized.schwarz_threshold:.0e} "
        f"(÷{starting.schwarz_threshold / max(optimized.schwarz_threshold, 1e-300):.2g}). "
    )
    if (schwarz_tighten_factor > 1.0 and n_evals < max_evaluations
            and optimized.cutoff_bohr > starting.cutoff_bohr * 1.01):
        # Bisect cutoff_bohr down using the already-tightened Schwarz
        # from Phase 1 (joint_growth) or a fresh tighten (non-joint).
        low = float(starting.cutoff_bohr)
        high = float(optimized.cutoff_bohr)
        tight_schwarz = (
            float(optimized.schwarz_threshold) if joint_growth
            else float(starting.schwarz_threshold) / schwarz_tighten_factor
        )
        # 3-iteration bisection caps additional evaluations at ~3.
        for _ in range(3):
            if n_evals >= max_evaluations:
                break
            mid = 0.5 * (low + high)
            trial = _copy_lattice_opts(optimized)
            trial.cutoff_bohr = mid
            trial.nuclear_cutoff_bohr = mid * (
                float(optimized.nuclear_cutoff_bohr) / float(optimized.cutoff_bohr)
            )
            trial.schwarz_threshold = tight_schwarz
            rep_trial = _evaluate(trial)
            if _meets_target(rep_trial):
                optimized = _copy_lattice_opts(trial)
                last_passing_rep = rep_trial
                high = mid
            else:
                low = mid
        if optimized.cutoff_bohr < float(last_passing.cutoff_bohr):
            notes += (
                f"Phase-2 bisection: cutoff came back down to "
                f"{optimized.cutoff_bohr:.2f} bohr "
                f"(Schwarz {optimized.schwarz_threshold:.0e})."
            )

    final_rep = last_passing_rep if last_passing_rep is not None else rep
    return TruncationOptimizationReport(
        optimized_lattice_opts=optimized,
        starting_lattice_opts=starting,
        n_evaluations=n_evals,
        final_severity=final_rep.severity,
        final_min_eigenvalue=final_rep.worst_min_eigenvalue,
        final_n_negative=sum(r.n_negative for r in final_rep.per_k_reports),
        converged=True,
        cutoff_bohr_path=cutoff_path,
        nuclear_cutoff_bohr_path=nuclear_cutoff_path,
        schwarz_threshold_path=schwarz_path,
        notes=notes,
    )


def format_truncation_optimization_report(
    report: TruncationOptimizationReport,
) -> str:
    """Render a :class:`TruncationOptimizationReport` as a multi-line
    summary."""
    lines: List[str] = []
    if report.converged:
        head = "Truncation optimisation: converged"
    else:
        head = "Truncation optimisation: DID NOT CONVERGE"
    lines.append(head)
    lines.append("-" * 60)
    lines.append(
        f"  starting cutoff   : {report.starting_lattice_opts.cutoff_bohr:.3f} bohr"
    )
    lines.append(
        f"  starting nuc cutoff: {report.starting_lattice_opts.nuclear_cutoff_bohr:.3f} bohr"
    )
    lines.append(
        f"  starting schwarz   : {report.starting_lattice_opts.schwarz_threshold:.0e}"
    )
    lines.append("")
    lines.append(
        f"  final cutoff      : {report.optimized_lattice_opts.cutoff_bohr:.3f} bohr"
    )
    lines.append(
        f"  final nuc cutoff  : {report.optimized_lattice_opts.nuclear_cutoff_bohr:.3f} bohr"
    )
    lines.append(
        f"  final schwarz     : {report.optimized_lattice_opts.schwarz_threshold:.0e}"
    )
    lines.append("")
    lines.append(
        f"  evaluations: {report.n_evaluations}, "
        f"final severity: {report.final_severity}, "
        f"min eig: {report.final_min_eigenvalue:+.3e}, "
        f"n_negative: {report.final_n_negative}"
    )
    if report.notes:
        lines.append("")
        import textwrap
        for line in textwrap.wrap(report.notes, width=70):
            lines.append(f"  {line}")
    return "\n".join(lines)


def format_disambiguation_report(report: DisambiguationReport) -> str:
    """Render a :class:`DisambiguationReport` as a multi-line summary."""
    lines: List[str] = []
    headline = {
        "screening_undertight":
            "Diagnosis: under-converged screening (NOT a basis problem)",
        "basis_set_problem":
            "Diagnosis: genuine basis-set linear dependence",
        "inconclusive":
            "Diagnosis: inconclusive (likely both contributions)",
    }[report.verdict]
    lines.append(headline)
    lines.append("-" * 60)
    lines.append(
        f"  original   : severity = {report.original_severity:>8s}, "
        f"min eig = {report.original_min_eigenvalue:+.3e}, "
        f"n_negative = {report.original_n_negative}"
    )
    lines.append(
        f"  tightened  : severity = {report.tightened_severity:>8s}, "
        f"min eig = {report.tightened_min_eigenvalue:+.3e}, "
        f"n_negative = {report.tightened_n_negative}"
    )
    lines.append(
        f"  multipliers: cutoff x{report.cutoff_tighten_factor:g}, "
        f"schwarz ÷{report.schwarz_tighten_factor:g}"
    )
    lines.append("")
    lines.append("  Rationale:")
    # Word-wrap the rationale at ~70 chars for readability.
    import textwrap
    for line in textwrap.wrap(report.rationale, width=70):
        lines.append(f"    {line}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_eigs_report(
    report: EIGSReport,
    *,
    show_per_k_eigenvalues: bool = False,
    max_eigvals_per_k: int = 4,
) -> str:
    """Render an :class:`EIGSReport` as a multi-line summary.

    The default output is a one-line-per-k summary table; pass
    ``show_per_k_eigenvalues=True`` to also dump the lowest few
    eigenvalues at each k.
    """
    lines: List[str] = []
    lines.append(
        f"EIGS preflight: nbf = {report.n_basis}, "
        f"n_kpoints = {report.n_kpoints}, "
        f"worst severity = {report.worst_severity}"
    )
    cond = (
        f"{report.worst_condition_number:.2e}"
        if np.isfinite(report.worst_condition_number) else "+inf"
    )
    lines.append(
        f"  worst min eig = {report.worst_min_eigenvalue:+.3e}, "
        f"worst cond = {cond}"
    )
    lines.append(
        f"  thresholds: warn = {report.warn_threshold:.0e}, "
        f"error = {report.error_threshold:.0e}, "
        f"negative = {report.negative_threshold:+.0e}"
    )
    lines.append("-" * 60)
    lines.append(
        f"  {'k':>4}  {'k_cart':>30}  {'min eig':>12}  "
        f"{'cond':>10}  {'sev':>8}"
    )
    for k_idx, rep in enumerate(report.per_k_reports):
        k = report.k_points_cart[k_idx]
        cond_str = (
            f"{rep.condition_number:.2e}"
            if np.isfinite(rep.condition_number) else "+inf"
        )
        k_str = f"({k[0]:+.3f}, {k[1]:+.3f}, {k[2]:+.3f})"
        lines.append(
            f"  {k_idx:>4}  {k_str:>30}  "
            f"{rep.min_eigenvalue:>+12.3e}  {cond_str:>10}  "
            f"{rep.severity:>8}"
        )
        if show_per_k_eigenvalues and rep.eigenvalues.size > 0:
            n = min(max_eigvals_per_k, rep.eigenvalues.size)
            low = ", ".join(f"{e:+.3e}" for e in rep.eigenvalues[:n])
            lines.append(f"          lowest {n}: {low}")

    lines.append("")
    if report.worst_severity == "ok":
        lines.append("  ✓ Basis well-conditioned at every requested k-point.")
        lines.append("    Safe to run SCF.")
    elif report.worst_severity == "warn":
        lines.append(
            "  ! Some k-points are near-singular. SCF should still "
            "converge with canonical orthogonalisation, but consider "
            "tightening the lattice cutoff or switching to a basis "
            "designed for solids (pob-tzvp / pob-tzvp-rev2)."
        )
    elif report.worst_severity == "error":
        lines.append(
            "  ✗ Basis is linearly dependent at one or more k-points. "
            "The SCF will refuse to run by default. Suggested fixes:"
        )
        lines.append(
            "    1. vq.make_basis(mol, name, exp_to_discard=0.1) -- drop "
            "diffuse primitives at construction."
        )
        lines.append(
            "    2. Switch to a basis designed for solids "
            "('pob-tzvp', 'pob-tzvp-rev2')."
        )
    else:   # critical
        lines.append(
            "  ✗✗ CRITICAL: overlap matrix has lost positive-definiteness "
            "at one or more k-points. An AO Gram matrix must be PSD by "
            "construction; this is a basis-set or screening problem, "
            "not numerical noise. SCF will refuse to run."
        )
        lines.append(
            "    Suggested fixes (in order):"
        )
        lines.append(
            "      1. Use a basis designed for solids: "
            "'pob-tzvp' (Peintinger 2013), 'pob-tzvp-rev2' "
            "(Vilela Oliveira 2019). Both ship with vibe-qc."
        )
        lines.append(
            "      2. Tighten lattice / Schwarz cutoffs first -- the "
            "negative eigenvalues may be from under-converged "
            "exchange screening (CRYSTAL TOLINTEG, manual p. 130, 398)."
        )
        lines.append(
            "      3. Filter primitives at runtime: "
            "vq.make_basis(mol, name, exp_to_discard=0.1)."
        )

    return "\n".join(lines)
