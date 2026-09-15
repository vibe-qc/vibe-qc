"""Production recipe -- full basis-set optimization with the qc-input-library.

This is the definitive entry point for running a basis-set
optimization.  It auto-discovers systems from the qc-input-library,
builds a multi-system joint objective with optional LD penalty,
and runs NLopt BOBYQA followed by iminuit MIGRAD/HESSE.

Usage::

    from vibeqc.basis_optimization.recipes.production import optimize_basis
    result = optimize_basis(calculator, method="rhf", basis="pob-TZVP")
    print(result.summary())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from vibe_basis.io.library_bridge import (
    LibraryEntry,
    LibraryIndex,
    discover_library,
    parse_d12_geometry,
)
from vibe_basis.optimize import OptResult, optimize_minuit, optimize_nlopt

from ...basis_crystal import parse_crystal_atom_basis_file
from vibe_basis.engine import EnergyEngine
from vibe_basis.engines.crystal23 import Crystal23Engine
from ..ld_diagnostics import (
    LDDiagnostics,
    compute_overlap_diagnostics,
    ld_penalty,
)
from ..parametrise import BasisParametrisation, FreeSpec, Transform
from .objective import make_multi_objective


@dataclass
class OptimizationResult:
    """Complete result of a basis-set optimization run."""

    method: str
    basis: str
    n_compounds: int
    n_free_params: int
    compounds: list[str]
    param_labels: list[str]
    seed_exponents: dict[str, float]
    optimized_exponents: dict[str, float]
    starting_objective: float
    optimal_objective: float
    bobyqa: OptResult
    minuit: Optional[OptResult] = None
    ld_penalty_enabled: bool = False
    ld_lambda: float = 0.0
    ld_diagnostics: Optional[LDDiagnostics] = None
    wall_seconds: float = 0.0
    calculator_label: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def objective_improvement_mha(self) -> float:
        return (self.starting_objective - self.optimal_objective) * 1000.0

    def summary(self) -> str:
        lines = [
            f"Basis-set optimization -- {self.method} on {self.basis}",
            f"  calculator: {self.calculator_label}",
            f"  compounds: {self.n_compounds}  free params: {self.n_free_params}",
            f"  objective: SE -> "
            f"{self.starting_objective:.6f} -> {self.optimal_objective:.6f} Ha",
            f"  improvement: {self.objective_improvement_mha:+.3f} mHa",
        ]

        if self.ld_penalty_enabled and self.ld_diagnostics:
            ld = self.ld_diagnostics
            lines.append(
                f"  LD penalty l_ld={self.ld_lambda:.0f}: "
                f"l_min(S)={ld.lambda_min:.2e}, "
                f"detected={ld.ld_detected}, "
                f"n_dep={ld.n_dependent}"
            )

        lines.append("  exponents (seed -> optimised):")
        for name in self.param_labels:
            s = self.seed_exponents.get(name, float("nan"))
            o = self.optimized_exponents.get(name, float("nan"))
            dpct = 100 * (o - s) / s if s else float("nan")
            lines.append(f"    {name:<28} {s:.6f} -> {o:.6f}  ({dpct:+.1f}%)")

        lines.append(
            f"  BOBYQA: {self.bobyqa.n_evaluations} evals, "
            f"{self.bobyqa.wall_seconds:.0f}s, "
            f"converged={self.bobyqa.success}"
        )
        if self.minuit:
            lines.append(
                f"  MIGRAD: {self.minuit.n_evaluations} evals, "
                f"{self.minuit.wall_seconds:.0f}s, "
                f"converged={self.minuit.success}"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-element base basis configuration
# ---------------------------------------------------------------------------

# Default elements to optimize -- the 8 elements from the pob test set.
_DEFAULT_ELEMENTS = {
    1: "pob-TZVP",  # H
    3: "pob-TZVP",  # Li
    8: "pob-TZVP",  # O
    9: "pob-TZVP",  # F
    11: "pob-TZVP",  # Na
    12: "pob-TZVP",  # Mg
    17: "pob-TZVP",  # Cl
    19: "pob-TZVP",  # K
    20: "pob-TZVP",  # Ca
    6: "pob-TZVP",  # C
    13: "pob-TZVP",  # Al
    14: "pob-TZVP",  # Si
}


def _default_free_specs(
    atom_bases: dict[str, object],
    bounds: tuple[float, float],
) -> list[FreeSpec]:
    """Auto-detect valence-shell exponents to optimize.

    For each element, free the last s-shell and last p-shell
    exponents (the outermost / most diffuse ones).
    """
    specs: list[FreeSpec] = []
    for symbol, atom in sorted(atom_bases.items()):
        # Outermost s.
        s_shells = [(i, sh) for i, sh in enumerate(atom.shells) if sh.shell_type == "S"]
        if s_shells:
            specs.append(
                FreeSpec(
                    symbol=symbol,
                    shell_idx=s_shells[-1][0],
                    prim_idx=0,
                    field="exponent",
                    transform=Transform.LOG,
                    bounds=bounds,
                    label=f"{symbol}_s_outer",
                )
            )
        # Outermost p.
        p_shells = [(i, sh) for i, sh in enumerate(atom.shells) if sh.shell_type == "P"]
        if p_shells:
            specs.append(
                FreeSpec(
                    symbol=symbol,
                    shell_idx=p_shells[-1][0],
                    prim_idx=0,
                    field="exponent",
                    transform=Transform.LOG,
                    bounds=bounds,
                    label=f"{symbol}_p_outer",
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Production optimizer
# ---------------------------------------------------------------------------


def optimize_basis(
    calculator: EnergyEngine,
    *,
    method: str = "rhf",
    basis: str = "pob-TZVP",
    library_root: str | Path | None = None,
    perturbation: float = 1.5,
    bounds: tuple[float, float] = (0.10, 5.0),
    use_ld_penalty: bool = False,
    ld_lambda: float = 1e3,
    ld_epsilon: float = 1e-7,
    max_eval: int = 100,
    run_minuit: bool = True,
    elements: dict[int, str] | None = None,
    compounds: list[str] | None = None,
    weights: list[float] | None = None,
) -> OptimizationResult:
    """Run a full basis-set optimization.

    Auto-discovers systems from the qc-input-library, builds a
    multi-system joint objective, and optimises using NLopt BOBYQA
    followed by iminuit MIGRAD+HESSE.

    Parameters
    ----------
    calculator
        :class:`~vibe_basis.engine.EnergyEngine` -- ``Crystal23Engine``
        for remote CRYSTAL runs, ``VibeQcEngine`` for in-process
        vibe-qc. vibe-qc is the primary engine; CRYSTAL23 is the
        fallback (vibe-basis/ROADMAP.md § 2).
    method
        ``"rhf"``, ``"hf"``, or a DFT functional keyword.
    basis
        Seed basis name (``"pob-TZVP"``).
    library_root
        Path to qc-input-library. Auto-discovered if None.
    perturbation
        Multiplicative perturbation from seed exponents at start.
    bounds
        (lower, upper) bounds on exponents in physical space.
    use_ld_penalty
        If True, adds ``l_ld.max(0, e-l_min)^2`` penalty to the objective
        during each evaluation. The per-element atomic overlap is built
        in-memory via :func:`ld_penalty_from_atom` (no file round-trip);
        the penalty sums over every element present in the parametrisation.
        A separate post-optimisation diagnostic at the converged point is
        still emitted into the result summary for human inspection.
    ld_lambda, ld_epsilon
        LD penalty parameters.
    max_eval
        Maximum BOBYQA evaluations.
    run_minuit
        If True, run iminuit MIGRAD+HESSE at BOBYQA optimum.
    elements
        Dict of Z -> basis_source_name for elements to include.
        Defaults to :data:`_DEFAULT_ELEMENTS` (8 common elements).
    compounds
        List of compound names to include. None = all 3D bulk
        systems from the library with the given method/basis.
    weights
        Per-compound weights. None = equal weights.

    Returns
    -------
    OptimizationResult
    """
    t0 = time.perf_counter()
    notes: list[str] = []

    # ---- Discover library and test set ----
    lib = discover_library(library_root or ".")
    if lib is None:
        raise RuntimeError(
            "qc-input-library not found. Set library_root= or clone "
            "https://vibe-qc.com/docs/basisset_dev/index.html"
        )

    if compounds:
        entries = [
            e
            for e in lib.find(method=method, basis=basis.replace("_", "-"))
            if e.compound in compounds
        ]
    else:
        entries = lib.test_set(method=method, basis=basis.replace("_", "-"))

    if not entries:
        raise RuntimeError(
            f"No systems found for method={method}, basis={basis} in {lib.root}"
        )

    structures = []
    for e in entries:
        s = parse_d12_geometry(e.input_path)
        if s:
            structures.append(s)
    if not structures:
        raise RuntimeError("No parseable geometries in test set")

    compound_names = [e.compound for e in entries if parse_d12_geometry(e.input_path)]

    # ---- Parse seed basis atoms ----
    elem_src = elements or _DEFAULT_ELEMENTS
    atom_bases: dict[str, object] = {}
    for Z, src_name in elem_src.items():
        sym = _symbol(Z)
        path = _find_source(Z, src_name)
        if path is None:
            notes.append(f"no source for {sym} (Z={Z}), skipped")
            continue
        atom_bases[sym] = parse_crystal_atom_basis_file(path)

    if not atom_bases:
        raise RuntimeError("no seed basis atoms found")

    free_specs = _default_free_specs(atom_bases, bounds)
    if not free_specs:
        raise RuntimeError("no free parameters detected")

    p = BasisParametrisation(atoms=atom_bases, free=free_specs)
    param_labels = [fs.label for fs in free_specs]

    # ---- Seed values ----
    seed_vals: dict[str, float] = {}
    for spec in free_specs:
        sh = p.atoms[spec.symbol].shells[spec.shell_idx]
        seed_vals[spec.label] = sh.exponents[0]

    # ---- Multi-system objective (+ optional in-memory LD penalty) ----
    obj = make_multi_objective(
        p,
        calculator,
        structures,
        method=method,
        weights=weights,
        use_ld_penalty=use_ld_penalty,
        ld_lambda=ld_lambda,
        ld_epsilon=ld_epsilon,
    )

    n = len(free_specs)
    x0 = p.pack() + np.full(n, math.log(perturbation))
    f_start = obj(x0)

    log_bounds = [(math.log(bounds[0]), math.log(bounds[1]))] * n

    # ---- BOBYQA ----
    bobyqa = optimize_nlopt(obj, x0, bounds=log_bounds, max_eval=max_eval)

    # ---- MIGRAD refinement ----
    minuit_result: Optional[OptResult] = None
    if run_minuit:
        try:
            minuit_result = optimize_minuit(
                obj,
                bobyqa.x,
                bounds=log_bounds,
                labels=param_labels,
                max_iter=max_eval // 2,
            )
        except RuntimeError:
            notes.append("MIGRAD skipped -- iminuit not installed")

    # ---- Final LD diagnostic for the report ----
    # The in-objective penalty (above) is what actually steers the optimiser;
    # this final check just re-runs the diagnostic at the converged point so
    # the result summary can show l_min / n_dep at the optimum.
    ld_diag: Optional[LDDiagnostics] = None
    if use_ld_penalty:
        best_x = minuit_result.x if minuit_result else bobyqa.x
        try:
            best_atoms = p.unpack(best_x)
            from ...basis_crystal import emit_crystal

            best_basis_list = [best_atoms[k] for k in p.atoms_keys]
            best_basis_text = emit_crystal(best_basis_list)
            ld_diag = compute_overlap_diagnostics(
                best_basis_text,
                next(iter(elem_src.keys())),  # first element's Z
                epsilon=ld_epsilon,
            )
            if ld_diag.ld_detected:
                notes.append(
                    f"LD detected at optimum: l_min={ld_diag.lambda_min:.2e} < "
                    f"e={ld_epsilon:.0e}"
                )
        except Exception as exc:
            notes.append(f"LD check failed: {exc}")

    # ---- Final values ----
    best_x = minuit_result.x if minuit_result else bobyqa.x
    f_best = float(obj(best_x))
    opt_atoms = p.unpack(best_x)
    opt_vals: dict[str, float] = {}
    for spec in free_specs:
        sh = opt_atoms[spec.symbol].shells[spec.shell_idx]
        opt_vals[spec.label] = sh.exponents[0]

    if f_best >= f_start:
        notes.append("WARNING: objective did not decrease")
    if not bobyqa.success:
        notes.append(f"BOBYQA: {bobyqa.message}")

    return OptimizationResult(
        method=method.upper(),
        basis=basis,
        n_compounds=len(structures),
        n_free_params=n,
        compounds=compound_names,
        param_labels=param_labels,
        seed_exponents=seed_vals,
        optimized_exponents=opt_vals,
        starting_objective=f_start,
        optimal_objective=f_best,
        bobyqa=bobyqa,
        minuit=minuit_result,
        ld_penalty_enabled=use_ld_penalty,
        ld_lambda=ld_lambda,
        ld_diagnostics=ld_diag,
        wall_seconds=time.perf_counter() - t0,
        calculator_label=getattr(calculator, "label", type(calculator).__name__),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symbol(Z: int) -> str:
    symbols = {
        1: "H",
        3: "Li",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        19: "K",
        20: "Ca",
    }
    return symbols.get(Z, f"Z{Z}")


def _find_source(Z: int, basis_name: str) -> Optional[Path]:
    """Locate a pob per-element CRYSTAL source file."""
    from .. import __file__ as bo_file

    worktree = Path(bo_file).resolve().parents[3]
    src = worktree / "python" / "vibeqc" / "basis_library" / "sources" / basis_name
    if not src.exists():
        return None
    candidates = sorted(src.glob(f"{Z:02d}_*"))
    return candidates[0] if candidates else None


if __name__ == "__main__":
    from vibe_basis.transports.local import LocalTransport

    calc = Crystal23Engine(transport=LocalTransport())
    result = optimize_basis(calc, library_root="~/gitlab/qc-input-library")
    print(result.summary())
