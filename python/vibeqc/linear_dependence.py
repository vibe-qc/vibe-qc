"""Linear-dependence diagnostic for an AO basis.

Every serious QC calculation starts by computing the overlap matrix
``S_muν = <chi_mu | chi_ν>`` and forming ``S^{-1/2}`` to orthogonalize the
basis. When two (or more) AOs are *almost* the same linear
combination of primitives -- either because the basis is too diffuse
for the geometry, because two atoms sit unusually close, or because
a periodic cell has a tight packing -- the smallest eigenvalues of S
go to zero and ``S^{-1/2}`` blows up.

The SCF drivers today abort hard with

::

    RKS: AO basis is linearly dependent (min S eigenvalue 3.2e-9)

which is correct but offers the user no diagnostic trail. This
module provides a **pre-flight** check returning a structured report:
condition number, eigenvalue spectrum, the basis functions that
contribute most to the near-null space, and an actionable message.

Usage
-----

.. code-block:: python

    import vibeqc as vq

    mol = vq.Molecule.from_xyz("dimer.xyz")
    basis = vq.BasisSet(mol, "aug-cc-pvtz")
    report = vq.check_linear_dependence(basis)
    print(vq.format_linear_dependence_report(report))
    vq.raise_if_severe(report)   # optional hard-fail

The report is a plain dataclass -- callers can also inspect
``report.min_eigenvalue``, ``report.condition_number``,
``report.offenders`` directly for programmatic use.

This diagnostic does *not* modify the basis or "fix" the dependence;
canonical orthogonalisation (projecting out near-null eigenvectors
to shrink the effective basis dimension) is a follow-up that will
touch the SCF drivers directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ._vibeqc_core import BasisSet, compute_overlap


__all__ = [
    "LinearDependenceReport",
    "PeriodicLinearDependenceSummary",
    "format_periodic_linear_dependence",
    "LinearDependenceOffender",
    "LinearDependenceError",
    "check_linear_dependence",
    "check_overlap_matrix",
    "scf_preflight_overlap_check",
    "format_linear_dependence_report",
    "raise_if_severe",
    # Public thresholds -- exposed so callers can override the defaults
    # without having to remember the magic numbers.
    "DEFAULT_WARN_THRESHOLD",
    "DEFAULT_ERROR_THRESHOLD",
    "DEFAULT_NEGATIVE_THRESHOLD",
]


# Conventional thresholds matching PySCF / ORCA / Turbomole practice.
# - warn  : near-null but still recoverable with canonical orthogonalisation.
#           Matches pyscf.lib.linalg_helper's default.
# - error : well into machine-precision territory -- S^{-1/2} is numerically
#           meaningless.
DEFAULT_WARN_THRESHOLD: float = 1.0e-6
DEFAULT_ERROR_THRESHOLD: float = 1.0e-8

# An overlap matrix is mathematically positive-semi-definite by
# construction -- Gram matrices can't have negative eigenvalues. A
# negative eigenvalue more negative than this threshold is not just
# numerical noise; it means S has lost positive-definiteness, usually
# from a Bloch-summation that's mis-converged at a tight crystal
# geometry, or a periodic image cutoff that's truncating shells that
# straddle the boundary. This is a structural problem that canonical
# orthogonalisation papers over silently -- flag it loudly so the user
# isn't running SCF in a basis with the wrong metric signature.
DEFAULT_NEGATIVE_THRESHOLD: float = -1.0e-6


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LinearDependenceOffender:
    """One basis function that contributes significantly to the
    near-null space of the overlap matrix.

    ``bf_index`` is the AO's index in the basis; ``shell_l`` is its
    angular momentum; ``atom_index`` identifies the atom it sits on;
    ``min_exponent`` is the smallest primitive exponent in the shell
    (small exponents -> diffuse functions -> the usual offender);
    ``weight`` is this AO's squared amplitude summed over the
    near-null eigenvectors (larger means it contributes more to the
    collapse).
    """

    bf_index: int
    shell_index: int
    atom_index: int
    shell_l: int
    min_exponent: float
    weight: float


@dataclass
class LinearDependenceReport:
    """Structured summary of the overlap-matrix linear-dependence
    analysis."""

    n_basis: int
    eigenvalues: np.ndarray              # sorted ascending
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float              # max / min
    warn_threshold: float
    error_threshold: float
    n_below_warn: int                    # # of eigvals < warn_threshold
    n_below_error: int                   # # of eigvals < error_threshold
    # Number of eigenvalues below ``negative_threshold`` (a negative
    # number; default -1e-6). Non-zero means S has lost positive-
    # definiteness -- typically a periodic Bloch-sum / cutoff problem,
    # not a diffuse-basis issue. Always 0 for molecular S.
    n_negative: int = 0
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD
    offenders: List[LinearDependenceOffender] = field(default_factory=list)
    # Optional label so a single report can identify *which* overlap
    # matrix it describes (e.g. ``"S(Γ)"``, ``"S(k=3)"``). Helps when
    # the SCF banner emits multiple reports back-to-back.
    label: str = "AO overlap"

    @property
    def severity(self) -> str:
        """One of ``"ok"``, ``"warn"``, ``"error"``, ``"critical"``.

        ``"critical"`` is reserved for non-positive-semi-definite S
        (the periodic-Bloch-sum failure mode). It supersedes the other
        levels -- once S has negative eigenvalues, the ``warn`` /
        ``error`` near-zero counts are downstream symptoms.
        """
        if self.n_negative > 0:
            return "critical"
        if self.n_below_error > 0:
            return "error"
        if self.n_below_warn > 0:
            return "warn"
        return "ok"


class LinearDependenceError(RuntimeError):
    """Raised by :func:`raise_if_severe` when the basis is linearly
    dependent beyond the error threshold. Carries the full
    :class:`LinearDependenceReport` on ``.report`` for inspection."""

    def __init__(self, message: str, report: LinearDependenceReport):
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def check_linear_dependence(
    basis: BasisSet,
    *,
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    max_offenders: int = 10,
) -> LinearDependenceReport:
    """Analyze the overlap matrix of ``basis`` for near-linear-
    dependence and return a report.

    Parameters
    ----------
    basis
        AO basis to inspect.
    warn_threshold
        Eigenvalues of S below this are counted as "warning"
        (recoverable via canonical orthogonalisation). Default
        ``1e-6`` matches PySCF / ORCA convention.
    error_threshold
        Eigenvalues below this are "error" (numerically untreatable
        in the current SCF drivers). Default ``1e-8``.
    max_offenders
        Report at most this many contributing basis functions to keep
        the output compact.

    Returns
    -------
    :class:`LinearDependenceReport`.
    """
    S = np.asarray(compute_overlap(basis))
    return check_overlap_matrix(
        S,
        basis=basis,
        warn_threshold=warn_threshold,
        error_threshold=error_threshold,
        max_offenders=max_offenders,
        label="molecular AO overlap",
    )


def check_overlap_matrix(
    S: np.ndarray,
    *,
    basis: Optional[BasisSet] = None,
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    max_offenders: int = 10,
    label: str = "AO overlap",
) -> LinearDependenceReport:
    """Analyze a precomputed overlap matrix for linear dependence.

    Use this when you've already built ``S`` (e.g., the Bloch-summed
    ``S(Γ)`` in a periodic SCF, or ``S(k)`` per k-point); use
    :func:`check_linear_dependence` for the raw-basis case.

    Detects three failure modes, in increasing severity:

    * **near-null eigenvalues** (``warn``/``error`` levels) -- the
      classic linear-dependence case from diffuse / overcomplete bases.
      Recoverable via canonical orthogonalisation.
    * **negative eigenvalues** (``critical`` level) -- S has lost its
      positive-semi-definite structure, which happens when a periodic
      Bloch sum is mis-converged or a real-space lattice cutoff
      truncates shells that straddle the cell boundary. Canonical
      orthogonalisation will silently drop these directions but the
      remaining basis represents the wrong physical density. Loudly
      flagged so the user can fix the cutoff / sampling instead of
      running SCF on a corrupted metric.

    Parameters
    ----------
    S
        ``(n, n)`` real (or complex Hermitian) overlap matrix.
    basis
        Optional :class:`BasisSet`. When provided, the report fills in
        the per-AO offender list (which diffuse functions cause the
        near-null space). When omitted (e.g., for a Bloch-summed S(k)
        whose AO indices don't map back to a single basis), the
        offender list is left empty.
    warn_threshold, error_threshold, negative_threshold
        Eigenvalue thresholds; see module docstring.
    max_offenders, label
        Bookkeeping for the report.
    """
    if not 0 < error_threshold < warn_threshold:
        raise ValueError(
            "check_overlap_matrix: require 0 < error_threshold "
            f"(= {error_threshold}) < warn_threshold "
            f"(= {warn_threshold})"
        )
    if negative_threshold > 0:
        raise ValueError(
            "check_overlap_matrix: negative_threshold must be <= 0; "
            f"got {negative_threshold}"
        )

    S_arr = np.asarray(S)
    if S_arr.ndim != 2 or S_arr.shape[0] != S_arr.shape[1]:
        raise ValueError(
            f"check_overlap_matrix: S must be square 2D, got shape "
            f"{S_arr.shape}"
        )

    # Symmetrise (or Hermitise for complex) before eigendecomposition.
    # Floating-point accumulation in Bloch sums can leave ~1e-15
    # asymmetry on the off-diagonal that flips eigvals' sign at the
    # noise level -- symmetrising first cleans it up.
    if np.iscomplexobj(S_arr):
        S_sym = 0.5 * (S_arr + S_arr.conj().T)
    else:
        S_sym = 0.5 * (S_arr + S_arr.T)

    eigvals = np.linalg.eigvalsh(S_sym).astype(float)
    n_bf = S_sym.shape[0]

    min_eig = float(eigvals[0])
    max_eig = float(eigvals[-1])
    cond = max_eig / min_eig if min_eig > 0.0 else float("inf")

    n_below_warn = int((eigvals < warn_threshold).sum())
    n_below_error = int((eigvals < error_threshold).sum())
    n_negative = int((eigvals < negative_threshold).sum())

    offenders: List[LinearDependenceOffender] = []
    if basis is not None and (n_below_warn > 0 or n_negative > 0):
        # Recompute eigenvectors only when we need offender attribution
        # (otherwise the eigvalsh fast path above is enough).
        _, eigvecs = np.linalg.eigh(S_sym)
        threshold = warn_threshold
        offenders = _identify_offenders(
            basis, eigvals, eigvecs, threshold, max_offenders,
        )

    return LinearDependenceReport(
        n_basis=n_bf,
        eigenvalues=eigvals,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        condition_number=cond,
        warn_threshold=warn_threshold,
        error_threshold=error_threshold,
        n_below_warn=n_below_warn,
        n_below_error=n_below_error,
        n_negative=n_negative,
        negative_threshold=negative_threshold,
        offenders=offenders,
        label=label,
    )


def _identify_offenders(
    basis: BasisSet,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    threshold: float,
    max_offenders: int,
) -> List[LinearDependenceOffender]:
    """Identify basis functions whose contribution dominates the
    near-null eigenvectors.

    For each AO mu, compute the sum over near-null eigenvectors k of
    ``|C_{muk}|^2``. The AOs with the largest such weights are the
    ones "responsible" for the collapse -- usually a small cluster of
    diffuse primitives on neighboring atoms.
    """
    mask = eigvals < threshold
    if not mask.any():
        return []
    # (n_bf, n_null) slab of near-null eigenvectors; column k is the
    # k-th near-null eigenvector expressed in the AO basis.
    null_vecs = eigvecs[:, mask]
    # Per-AO contribution weight: |c_{muk}|^2 summed over near-null
    # eigenvectors. Use ``np.abs(...)**2`` so complex eigenvectors
    # (multi-k Hermitian S(k)) and real ones share one code path
    # without a ComplexWarning at the float cast on line below.
    weights = (np.abs(null_vecs) ** 2).sum(axis=1).real

    # Build shell metadata: for each AO index mu, what shell is it in,
    # what's the shell's l, what atom, smallest primitive exponent?
    shells = list(basis.shells())
    # AOs-per-shell count: for pure spherical harmonics, each shell
    # carries (2l+1) AOs sequentially.
    bf_to_shell: List[int] = []
    for s_idx, sh in enumerate(shells):
        n_bf_sh = 2 * int(sh.l) + 1
        bf_to_shell.extend([s_idx] * n_bf_sh)

    # Order AOs by weight descending; emit the top max_offenders.
    order = np.argsort(-weights)
    offenders: List[LinearDependenceOffender] = []
    for mu in order[:max_offenders]:
        if weights[mu] < 1e-6:
            break   # no more meaningful contributors
        s_idx = bf_to_shell[mu]
        sh = shells[s_idx]
        offenders.append(
            LinearDependenceOffender(
                bf_index=int(mu),
                shell_index=int(s_idx),
                atom_index=int(sh.atom_index),
                shell_l=int(sh.l),
                min_exponent=float(min(sh.exponents)),
                weight=float(weights[mu]),
            )
        )
    return offenders


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class PeriodicLinearDependenceSummary:
    """What canonical orthogonalisation actually discarded, per k-point.

    A periodic SCF builds one overlap ``S(k)`` per Brillouin-zone point
    and canonically orthogonalises each, dropping the directions whose
    eigenvalue falls below ``threshold``. That discarding is where an
    ill-conditioned periodic basis silently changes the variational
    space the SCF is allowed to use, and where the density-fitting
    ``1/sqrt(lambda)`` amplification originates.

    Every field here was already computed inside the drivers and thrown
    away. Reporting it is what lets a reader distinguish "this run
    discarded four directions at k=3" from the several other things a
    non-physical periodic energy can mean.

    ``n_fit_kept`` / ``n_aux`` describe the *auxiliary* fit rather than
    the AO overlap: the density-fitting metric is orthogonalised on its
    own threshold, and on compact dense-core cells that is the dominant
    failure mode (measured: sweeping it moved MgO MDF from -62874 Ha to
    -271.146 Ha against a -271.0499 Ha reference, with none of this
    state observable at the time).
    """

    n_basis: int
    n_kept_per_k: List[int] = field(default_factory=list)
    threshold: float = 0.0
    min_overlap_eigenvalue: float = float("nan")
    max_overlap_eigenvalue: float = float("nan")
    # Auxiliary (density-fitting) side; 0 when the route has no aux fit.
    n_aux: int = 0
    n_fit_kept: int = 0
    aux_threshold: float = 0.0

    @property
    def n_discarded_max(self) -> int:
        """Largest number of AO directions dropped at any single k."""
        if not self.n_kept_per_k:
            return 0
        return int(self.n_basis - min(self.n_kept_per_k))

    @property
    def any_discarded(self) -> bool:
        return self.n_discarded_max > 0

    @property
    def condition_number(self) -> float:
        lo = float(self.min_overlap_eigenvalue)
        hi = float(self.max_overlap_eigenvalue)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0:
            return float("inf")
        return hi / lo

    def one_line(self) -> str:
        """Compact form for an error message or a warning."""
        parts = [
            f"AO overlap: kept {min(self.n_kept_per_k, default=self.n_basis)}"
            f"-{max(self.n_kept_per_k, default=self.n_basis)} of "
            f"{self.n_basis} directions per k "
            f"(threshold {self.threshold:g})"
        ]
        if np.isfinite(self.min_overlap_eigenvalue):
            parts.append(
                f"min eig(S) = {self.min_overlap_eigenvalue:.3e}, "
                f"cond = {self.condition_number:.2e}"
            )
        if self.n_aux:
            parts.append(
                f"aux fit: kept {self.n_fit_kept} of {self.n_aux} "
                f"(threshold {self.aux_threshold:g})"
            )
        return "; ".join(parts)


def format_periodic_linear_dependence(
    summary: PeriodicLinearDependenceSummary,
    *,
    max_k_shown: int = 8,
) -> str:
    """Render the periodic summary as an ``.out`` block.

    Deliberately printed even when nothing was discarded: "0 directions
    dropped" is the fact that rules linear dependence OUT as a cause,
    and its absence is what made several periodic post-mortems ambiguous.
    """
    lines: List[str] = []
    lines.append(f"    basis functions per k     = {summary.n_basis}")
    kept = summary.n_kept_per_k
    if kept:
        lo, hi = min(kept), max(kept)
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        lines.append(f"    retained after orthog.    = {span}")
        lines.append(
            f"    directions discarded      = "
            f"{summary.n_discarded_max}"
            + (" (max over k)" if lo != hi else "")
        )
    lines.append(f"    linear-dep threshold      = {summary.threshold:g}")
    if np.isfinite(summary.min_overlap_eigenvalue):
        lines.append(
            f"    min eigenvalue of S(k)    = "
            f"{summary.min_overlap_eigenvalue:.6e}"
        )
        lines.append(
            f"    overlap condition number  = "
            f"{summary.condition_number:.3e}"
        )
    if summary.n_aux:
        lines.append(f"    auxiliary functions       = {summary.n_aux}")
        lines.append(f"    auxiliary fit retained    = {summary.n_fit_kept}")
        lines.append(
            f"    auxiliary threshold       = {summary.aux_threshold:g}"
        )
    if kept and len(kept) > 1 and min(kept) != max(kept):
        shown = kept[:max_k_shown]
        tail = " ..." if len(kept) > max_k_shown else ""
        lines.append(
            "    retained per k            = "
            + ", ".join(str(v) for v in shown)
            + tail
        )
    return "\n".join(lines) + "\n"


def format_linear_dependence_report(
    report: LinearDependenceReport,
    *,
    max_eigvals_shown: int = 6,
) -> str:
    """Render the report as a human-readable multi-line string,
    matching the house style of :func:`vibeqc.format_memory_report`."""
    lines: List[str] = []
    header = {
        "ok":       "Basis well-conditioned",
        "warn":     "Basis is near-linearly-dependent (warning)",
        "error":    "Basis is linearly dependent (error)",
        "critical": "Overlap matrix has lost positive-definiteness (CRITICAL)",
    }[report.severity]
    lines.append(f"Linear-dependence check [{report.label}]: {header}")
    lines.append("-" * 52)
    lines.append(f"  basis dimension:            {report.n_basis}")
    lines.append(f"  min eigenvalue of S:        {report.min_eigenvalue:.3e}")
    lines.append(f"  max eigenvalue of S:        {report.max_eigenvalue:.3e}")
    cond_str = (
        f"{report.condition_number:.3e}"
        if np.isfinite(report.condition_number) else "+inf"
    )
    lines.append(f"  condition number:           {cond_str}")
    lines.append(
        f"  eigenvalues below warn ({report.warn_threshold:.0e}):  "
        f"{report.n_below_warn}"
    )
    lines.append(
        f"  eigenvalues below error ({report.error_threshold:.0e}): "
        f"{report.n_below_error}"
    )
    if report.n_negative > 0:
        lines.append(
            f"  eigenvalues below {report.negative_threshold:+.0e} (NEGATIVE): "
            f"{report.n_negative}"
        )

    if max_eigvals_shown > 0 and report.n_basis > 0:
        k = min(max_eigvals_shown, report.n_basis)
        low = ", ".join(f"{e:.3e}" for e in report.eigenvalues[:k])
        lines.append(f"  lowest {k} eigenvalues:       {low}")

    if report.offenders:
        lines.append("")
        lines.append(
            f"  Top {len(report.offenders)} basis functions contributing "
            f"to the near-null space:"
        )
        lines.append(
            f"  {'bf':>4}  {'shell':>5}  {'atom':>4}  "
            f"{'l':>2}  {'a_min':>10}  {'weight':>8}"
        )
        for o in report.offenders:
            lines.append(
                f"  {o.bf_index:>4}  {o.shell_index:>5}  "
                f"{o.atom_index:>4}  {o.shell_l:>2}  "
                f"{o.min_exponent:>10.4f}  {o.weight:>8.4f}"
            )

    if report.severity == "warn":
        lines.append("")
        lines.append(
            "  Action: the SCF may still converge if you tighten damping "
            "or DIIS."
        )
        lines.append(
            "  A planned canonical-orthogonalisation path will make "
            "this case routine."
        )
    elif report.severity == "error":
        lines.append("")
        lines.append(
            "  Action: the SCF drivers will abort. Options: drop the most "
            "diffuse basis functions, switch to a less aggressive basis "
            "(e.g. def2-TZVP instead of aug-cc-pVQZ), or wait for the "
            "canonical-orthogonalisation feature."
        )
    elif report.severity == "critical":
        lines.append("")
        lines.append(
            "  Action: the overlap matrix has eigenvalues more negative "
            f"than {report.negative_threshold:+.0e}. An AO Gram matrix is "
            "mathematically positive-semi-definite, so this is *not* "
            "diffuse-basis numerical noise -- it indicates the basis is "
            "unsuitable for the geometry, OR a construction bug upstream:"
        )
        lines.append(
            "    * basis set with diffuse primitives unsuited to the "
            "tight packing of a solid (the most common cause)"
        )
        lines.append(
            "    * under-converged exchange screening (CRYSTAL's TOLINTEG "
            "ITOL4/ITOL5 -- what looks like linear dependence is actually "
            "a too-loose Schwarz / cutoff threshold; tighten "
            "``LatticeSumOptions.schwarz_threshold`` and "
            "``LatticeSumOptions.cutoff_bohr`` BEFORE touching the basis)"
        )
        lines.append(
            "    * AO image-cell sum overcounting at tight crystal "
            "geometries (v0.7 known issue, dense ionic crystals)"
        )
        lines.append(
            "    * Bloch sum mis-converged (k-mesh too coarse for the "
            "real-space cutoff)"
        )
        lines.append("")
        lines.append("  Recommended fixes (in order of preference):")
        lines.append(
            "    1. Use a basis set DESIGNED for solids -- pob-TZVP / "
            "pob-TZVP-rev2 are shipped with vibe-qc:"
        )
        lines.append(
            "         basis = vq.BasisSet(mol, 'pob-tzvp-rev2')   # "
            "Vilela Oliveira et al, J. Comput. Chem. 40, 2364 (2019)"
        )
        lines.append(
            "         # see also docs/tutorial/pob_tzvp.md and "
            "docs/tutorial/lih_pob_tzvp_solid_state.md"
        )
        lines.append(
            "    2. Filter the most diffuse primitives at runtime "
            "(Python-only, no basis swap) via the PySCF-style "
            "``exp_to_discard``:"
        )
        lines.append(
            "         basis = vq.make_basis(mol, 'sto-3g', "
            "exp_to_discard=0.1)"
        )
        lines.append(
            "    3. If you must use the original basis: bypass this check "
            "with ``allow_critical=True`` in ``raise_if_severe``. SCF "
            "WILL run but canonical orthogonalisation silently drops the "
            "wrong-signature directions -- the converged energy is "
            "physically meaningless. Not recommended."
        )
        lines.append("")
        lines.append(
            "  Other curated periodic bases (not shipped -- load externally):"
        )
        lines.append(
            "    - GTH-cc-pVXZ (Ye & Berkelbach, JCTC 18, 1595 (2022))"
        )
        lines.append(
            "    - MOLOPT (VandeVondele & Hutter, JCP 127, 114105 (2007))"
        )

    return "\n".join(lines)


def raise_if_severe(
    report: LinearDependenceReport,
    *,
    allow_warn: bool = True,
    allow_critical: bool = False,
) -> None:
    """Raise :class:`LinearDependenceError` if the report is beyond
    the safe-to-run threshold.

    Default policy:

    * ``"ok"``       -- pass through.
    * ``"warn"``     -- pass through (set ``allow_warn=False`` to abort).
    * ``"error"``    -- abort.
    * ``"critical"`` -- abort (set ``allow_critical=True`` to override
      and let canonical orthogonalisation silently drop the
      wrong-signature directions).

    The raised exception's ``.report`` attribute carries the full
    :class:`LinearDependenceReport` for post-mortem inspection.
    """
    sev = report.severity
    abort = (
        sev == "error"
        or (sev == "warn" and not allow_warn)
        or (sev == "critical" and not allow_critical)
    )
    if not abort:
        return

    if sev == "critical":
        msg = (
            f"Overlap matrix [{report.label}] has lost positive-definiteness "
            f"({report.n_negative} eigenvalues below "
            f"{report.negative_threshold:+.0e}, min = "
            f"{report.min_eigenvalue:.3e}). An AO Gram matrix must be "
            "PSD by construction; this indicates an upstream bug "
            "(lattice cutoff, image-cell summing, or Bloch sampling). "
            "See report for details."
        )
    else:
        msg = (
            f"Linear dependence in {report.label} (severity={sev}, "
            f"min eigenvalue = {report.min_eigenvalue:.3e}, "
            f"{report.n_below_error} eigenvalues below "
            f"{report.error_threshold:.0e}). See report for offending "
            f"basis functions."
        )
    raise LinearDependenceError(msg, report)


PERIODIC_OVERLAP_HINT = (
    "Periodic S(k): the one-electron lattice cutoff is auto-widened for "
    "diffuse bases and near-null directions are handled by canonical "
    "orthogonalisation. A residual critical S(k) therefore means the metric "
    "still has negative directions after cutoff growth. Filter diffuse "
    "primitives with vq.make_basis(..., exp_to_discard=0.1), raise the "
    "one-electron cutoff, or use a solid-state basis (pob-tzvp-rev2). "
    "vibeqc.eigs_preflight."
    "disambiguate_critical_overlap() reports which applies."
)
"""Remediation text for a critical periodic ``S(k)``.

Lives here rather than in one route's module because every periodic
driver that diagonalises ``S(k)`` needs it. A *truncated* Bloch sum
``S(k) = sum_g e^{ik.R_g} S(g)`` keeps ``g`` and ``-g`` together, so it
stays Hermitian --- but Hermitian is not positive-definite, and the Gram
structure that guarantees PSD only holds for the converged sum. Negative
eigenvalues at a loose cutoff are therefore an expected truncation
artefact that shrinks as the cutoff grows, not by itself evidence of a
code defect. Measured on LiF/6-31G in an 8-bohr cube at a (4,4,2) mesh,
min eig S(k) runs -1.400 (cutoff 8) -> -0.557 (12) -> -0.102 (16) ->
-3.6e-3 (20) -> +7.4e-5 (26) -> +9.6e-5 (40, converged); LiH/6-31G in a
12-bohr cube runs -0.115 (12) -> +0.021 (20) -> +0.0183 (40).

This is what CRYSTAL calls *numerical linear dependence*, and its EIGS
documentation makes the same connection: the more severe the
computational conditions, the closer to zero an eigenvalue may sit
without risk, and negative values signal numerical linear dependence
(CRYSTAL23 manual, EIGS). The converged value here really is small ---
6-31G on an 8-bohr LiF cell is close to over-complete --- which is a
basis/cell choice, not a bug.
"""


# ---------------------------------------------------------------------------
# SCF preflight integration
# ---------------------------------------------------------------------------

def scf_preflight_overlap_check(
    S: np.ndarray,
    *,
    plog,                                    # ProgressLogger; loose-typed
    label: str = "S(Γ)",
    basis: Optional[BasisSet] = None,
    warn_threshold: float = DEFAULT_WARN_THRESHOLD,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    allow_warn: bool = True,
    allow_critical: bool = False,
    raise_on_severe: bool = True,
    remediation_hint: str = "",
) -> LinearDependenceReport:
    """Inspect the overlap matrix and emit a one-line banner summary
    (plus a multi-line block if anything is wrong). Optionally raise
    when severity exceeds policy.

    Designed for the SCF startup banner. Always emits at least the
    one-line summary so users can confirm at a glance that S is
    well-conditioned. PySCF only emits this at ``DEBUG`` verbosity by
    default; vibe-qc surfaces it at ``INFO`` because the fidelity-
    versus-cost trade-off in periodic SCF is more sensitive to
    near-singular S than molecular SCF.

    Parameters
    ----------
    S
        ``(n, n)`` overlap matrix -- the Bloch-summed Γ-point S, or
        any single ``S(k)`` slice from a multi-k driver.
    plog
        :class:`vibeqc.progress.ProgressLogger`. ``plog.info`` /
        ``plog.write_raw`` / ``plog.warning`` are the surfaces used.
    label
        Human-readable identifier for this matrix (``"S(Γ)"``,
        ``"S(k=3)"``, ...). Appears in the banner line and the
        formatted report.
    basis
        Optional :class:`BasisSet` for offender attribution; passing
        ``None`` skips the per-AO breakdown.
    warn_threshold, error_threshold, negative_threshold
        Eigenvalue thresholds for severity classification.
    allow_warn, allow_critical, raise_on_severe
        SCF policy. ``raise_on_severe=False`` returns the report
        without raising even when the policy would normally abort --
        useful for callers that want to handle the situation
        themselves (e.g., fall back to canonical orthogonalisation).
    remediation_hint
        Optional caller-specific guidance appended to the
        :class:`LinearDependenceError` message when this check aborts.
        Periodic drivers pass a hint pointing at ``exp_to_discard`` /
        solid-state bases instead of the generic "upstream bug" framing.

    Returns
    -------
    :class:`LinearDependenceReport`.

    Raises
    ------
    :class:`LinearDependenceError`
        When severity exceeds policy and ``raise_on_severe=True``.
    """
    report = check_overlap_matrix(
        S,
        basis=basis,
        warn_threshold=warn_threshold,
        error_threshold=error_threshold,
        negative_threshold=negative_threshold,
        label=label,
    )

    cond_str = (
        f"{report.condition_number:.2e}"
        if np.isfinite(report.condition_number) else "+inf"
    )
    one_line = (
        f"overlap [{label}]: nbf={report.n_basis}, "
        f"min eig={report.min_eigenvalue:+.2e}, cond={cond_str}, "
        f"severity={report.severity}"
    )

    # One-line banner always; full report only when there's something
    # to look at (warn / error / critical).
    if report.severity == "ok":
        plog.info(one_line)
    else:
        # Distinguish the levels in the banner so a quick log-grep
        # ("WARN" / "ERROR" / "CRITICAL") finds them.
        prefix = {"warn": "WARN", "error": "ERROR", "critical": "CRITICAL"}[
            report.severity
        ]
        plog.info(f"[{prefix}] {one_line}")
        plog.write_raw(format_linear_dependence_report(report))

    if raise_on_severe:
        try:
            raise_if_severe(
                report,
                allow_warn=allow_warn,
                allow_critical=allow_critical,
            )
        except LinearDependenceError as exc:
            if remediation_hint:
                # A caller that supplies a hint is periodic, where a
                # negative eigenvalue is NOT prima facie a code defect: the
                # truncated Bloch sum keeps g and -g together so it stays
                # Hermitian, but the Gram structure that forces PSD holds
                # only for the converged sum. So drop the "upstream bug"
                # framing the molecular path uses -- it sends people
                # hunting for a defect when the first thing to try is a
                # larger cutoff -- and say what actually distinguishes the
                # two. See PERIODIC_OVERLAP_HINT for the measured
                # convergence data.
                text = str(exc.args[0]).replace(
                    "An AO Gram matrix must be PSD by construction; this "
                    "indicates an upstream bug (lattice cutoff, image-cell "
                    "summing, or Bloch sampling).",
                    "An AO Gram matrix is PSD only once its lattice sum has "
                    "converged, so this is usually the lattice cutoff being "
                    "too small rather than a defect: raise it and the "
                    "eigenvalue should climb toward a small positive value. "
                    "If it stays negative as the cutoff grows, then suspect "
                    "image-cell summing or Bloch sampling.",
                )
                raise LinearDependenceError(
                    f"{text} {remediation_hint}", report
                ) from None
            raise

    return report
