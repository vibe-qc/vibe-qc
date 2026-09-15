"""Stage 3 recipe: multi-compound joint HF fit.

This is the production recipe -- the same architecture that produced
pob-TZVP (Peintinger et al. 2013).  A single basis is evaluated on
many compounds simultaneously, and NLopt BOBYQA minimises the
weighted sum of total energies:

    L(x) = Sᵢ wᵢ . Eᵢ(x)

where x are the free basis parameters (valence-shell exponents
across all elements in the test set).  The resulting basis is
transferable across all bonding situations in the test set.

Stage 3 (per GOAL8_MPEI_TZVP.md):
  13 cubic ionics (PT2013 T4 + T14), ~8-16 free parameters,
  ~100 BOBYQA evals x 13 compounds = ~7 hours on compute-small.

Usage::

    from vibeqc.basis_optimization.recipes.crystal_stage3 import run_stage3_ionics
    result = run_stage3_ionics(VqTransport(host="compute-small"), ...)
    print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from vibe_basis.io.structures import in_table
from vibe_basis.optimize import OptResult, optimize_minuit, optimize_nlopt
from vibe_basis.transports.base import Transport

from ...basis_crystal import parse_crystal_atom_basis_file
from ..parametrise import BasisParametrisation, FreeSpec, Transform
from .crystal_objective import make_multi_crystal_objective

# The 13 cubic ionics from PT2013 T4 + T14.
# These compounds cover 8 elements: H, Li, Na, K, F, Cl, Mg, Ca, O.
_PT2013_T4_IONICS = [
    "LiCl",
    "NaCl",
    "LiF",
    "NaF",
    "KF",  # alkali halides
    "CaF2",
    "K2O",
    "MgO",
    "CaO",  # oxides + fluoride
    "LiH",
    "NaH",
    "KH",  # alkali hydrides
]

# Per-element basis setup for Stage 3.
# Map of Z -> (source_basis, shells_map) where shells_map is a dict
# of {label: shell_index} for the valence shells to optimize.
_ELEMENTS_IN_T4 = {
    1: ("pob-TZVP", {"s_diffuse": -1}),  # H: last s shell
    3: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # Li
    11: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # Na
    19: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # K
    9: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # F
    17: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # Cl
    12: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # Mg
    20: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # Ca
    8: ("pob-TZVP", {"s_outer": -1, "p_outer": -1}),  # O
}

ELEMENT_NAMES = {
    1: "H",
    3: "Li",
    11: "Na",
    19: "K",
    9: "F",
    17: "Cl",
    12: "Mg",
    20: "Ca",
    8: "O",
}


def _find_source(Z: int, basis: str) -> Path:
    from .. import __file__ as bo_file

    worktree = Path(bo_file).resolve().parents[3]
    src = worktree / "python" / "vibeqc" / "basis_library" / "sources" / basis
    return sorted(src.glob(f"{Z:02d}_*"))[0]


def _collect_valence_shells(atom, shell_type: str) -> list[tuple[int, object]]:
    """Return (idx, shell) pairs for shells of the given type."""
    return [(i, sh) for i, sh in enumerate(atom.shells) if sh.shell_type == shell_type]


@dataclass
class Stage3Result:
    """Outcome of a multi-compound joint optimization."""

    method: str
    n_compounds: int
    n_free_params: int
    free_param_labels: list[str]
    starting_weights: dict[str, float]
    optimal_weights: dict[str, float]
    starting_objective: float
    optimal_objective: float
    bobyqa_result: OptResult
    minuit_result: Optional[OptResult] = None
    transport: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Stage 3 -- multi-compound joint {self.method} fit",
            f"  compounds: {self.n_compounds}  free params: {self.n_free_params}",
            f"  objective: S Eᵢ -> "
            f"{self.starting_objective:.6f} -> {self.optimal_objective:.6f} Ha",
            f"  improvement: "
            f"{(self.starting_objective - self.optimal_objective) * 1000:.3f} mHa",
        ]
        lines.append("  exposures (seed -> optimised):")
        for name in self.free_param_labels:
            s = self.starting_weights.get(name, float("nan"))
            o = self.optimal_weights.get(name, float("nan"))
            dpct = 100 * (o - s) / s if s else float("nan")
            lines.append(f"    {name:<24} {s:.6f} -> {o:.6f}  ({dpct:+.1f}%)")
        lines.append(
            f"  BOBYQA: {self.bobyqa_result.n_evaluations} evals, "
            f"converged={self.bobyqa_result.success}"
        )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def run_stage3_ionics(
    transport: Transport,
    *,
    method: str = "rhf",
    perturbation: float = 1.5,
    bounds: tuple[float, float] = (0.10, 5.0),
    crystal_wrapper: str = "crystal",
    max_eval: int = 100,
    run_minuit: bool = False,
    **transport_kw,
) -> Stage3Result:
    """Run Stage 3: joint fit on 13 cubic ionics.

    Optimises valence-shell exponents for all 8 elements
    simultaneously against the weighted sum of total HF energies
    across the 13 compounds.

    Parameters
    ----------
    transport
        Transport for CRYSTAL14 submission.
    method
        ``"rhf"``, ``"hf"``, or DFT functional.
    perturbation
        Multiplicative perturbation from seed values.
    bounds
        (lower, upper) bounds on exponents in physical space.
    crystal_wrapper
        Path to CRYSTAL14 wrapper.
    max_eval
        Maximum BOBYQA evaluations.
    run_minuit
        If True, refine with MIGRAD+HESSE at BOBYQA optimum.
    **transport_kw
        cpus, wall_time_s, timeout_s.

    Returns
    -------
    Stage3Result
    """
    # Parse seed basis atoms for all 8 elements.
    atom_bases: dict[str, object] = {}
    free_specs: list[FreeSpec] = []
    param_labels: list[str] = []

    for Z, (basis_name, shell_map) in _ELEMENTS_IN_T4.items():
        symbol = ELEMENT_NAMES[Z]
        src = _find_source(Z, basis_name)
        atom = parse_crystal_atom_basis_file(src)
        atom_bases[symbol] = atom

        for label, selector in shell_map.items():
            if label == "s_diffuse" or (label.startswith("s_") and selector == -1):
                s_shells = _collect_valence_shells(atom, "S")
                if not s_shells:
                    continue
                idx = s_shells[-1][0]  # last s = most diffuse
                param_name = f"{symbol}_s_diffuse"
            elif label.startswith("s_") and selector == -1:
                s_shells = _collect_valence_shells(atom, "S")
                if not s_shells:
                    continue
                idx = s_shells[-1][0]
                param_name = f"{symbol}_s_outer"
            elif label.startswith("p_") and selector == -1:
                p_shells = _collect_valence_shells(atom, "P")
                if not p_shells:
                    continue
                idx = p_shells[-1][0]
                param_name = f"{symbol}_p_outer"
            else:
                continue

            free_specs.append(
                FreeSpec(
                    symbol=symbol,
                    shell_idx=idx,
                    prim_idx=0,
                    field="exponent",
                    transform=Transform.LOG,
                    bounds=bounds,
                    label=param_name,
                )
            )
            param_labels.append(param_name)

    if not free_specs:
        raise RuntimeError("no free parameters found for the 8 elements")

    p = BasisParametrisation(atoms=atom_bases, free=free_specs)

    # Starting values.
    starting_vals: dict[str, float] = {}
    for spec in free_specs:
        sh = p.atoms[spec.symbol].shells[spec.shell_idx]
        starting_vals[spec.label] = sh.exponents[0]

    # Multi-system objective: 13 cubic ionics.
    structures = in_table("PT2013-T4")
    obj = make_multi_crystal_objective(
        p,
        transport,
        structures,
        method=method,
        crystal_wrapper=crystal_wrapper,
        workdir_prefix="stage3_ionics",
        **transport_kw,
    )

    # Evaluate at perturbed start.
    n = len(free_specs)
    x0 = p.pack() + np.full(n, math.log(perturbation))
    f_start = obj(x0)

    notes: list[str] = []

    # BOBYQA.
    log_bounds = [(math.log(bounds[0]), math.log(bounds[1]))] * n
    bobyqa = optimize_nlopt(obj, x0, bounds=log_bounds, max_eval=max_eval)

    # MIGRAD.
    minuit_result: Optional[OptResult] = None
    if run_minuit:
        try:
            minuit_result = optimize_minuit(
                obj,
                bobyqa.x,
                bounds=log_bounds[:],
                labels=param_labels,
                max_iter=max_eval // 2,
            )
        except RuntimeError:
            notes.append("MIGRAD skipped -- iminuit not installed")

    best_x = minuit_result.x if minuit_result else bobyqa.x
    f_best = float(obj(best_x))

    # Optimal values.
    opt_atoms = p.unpack(best_x)
    optimal_vals: dict[str, float] = {}
    for spec in free_specs:
        sh = opt_atoms[spec.symbol].shells[spec.shell_idx]
        optimal_vals[spec.label] = sh.exponents[0]

    if f_best >= f_start:
        notes.append("WARNING: objective did not decrease")
    if not bobyqa.success:
        notes.append(f"BOBYQA: {bobyqa.message}")

    return Stage3Result(
        method=method.upper(),
        n_compounds=len(structures),
        n_free_params=n,
        free_param_labels=param_labels,
        starting_weights=starting_vals,
        optimal_weights=optimal_vals,
        starting_objective=f_start,
        optimal_objective=f_best,
        bobyqa_result=bobyqa,
        minuit_result=minuit_result,
        transport=type(transport).__name__,
        notes=notes,
    )


if __name__ == "__main__":
    from vibe_basis.transports.local import LocalTransport

    result = run_stage3_ionics(LocalTransport(), max_eval=5)
    print(result.summary())
