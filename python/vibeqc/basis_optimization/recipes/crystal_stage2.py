"""Stage 2 recipe: LiH one-compound HF optimization.

The publication-grade proof-of-concept from GOAL8_MPEI_TZVP.md Sec.5.
Optimises 3 valence-shell exponents (H s diffuse-most, Li outermost
s and p) for the LiH rocksalt 8-atom cubic cell using NLopt BOBYQA,
then refines with iminuit MIGRAD + HESSE for parameter uncertainties.

Acceptance gates:
* BOBYQA converges within 50 evals.
* mpei-TZVP-LiH total HF energy < pob-TZVP-rev2-LiH total HF energy
  by >= 0.1 mHa per cell.
* All converged exponents >= 0.15 (the pob-rev2 LD floor).
* HESSE error matrix is positive-definite.
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from vibe_basis.io.structures import STRUCTURES
from vibe_basis.optimize import OptResult, optimize_minuit, optimize_nlopt
from vibe_basis.transports.base import Transport

from ...basis_crystal import parse_crystal_atom_basis_file
from ..parametrise import BasisParametrisation, FreeSpec, Transform
from .crystal_objective import make_crystal_objective


@dataclass
class Stage2Result:
    """Outcome of a one-compound bulk-solid optimisation."""

    compound: str
    method: str
    free_params: list[str]
    starting_exponents: dict[str, float]
    optimal_exponents: dict[str, float]
    starting_energy: float
    optimal_energy: float
    bobyqa_result: OptResult
    minuit_result: Optional[OptResult] = None
    transport: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def energy_improvement_mha(self) -> float:
        return (self.starting_energy - self.optimal_energy) * 1000.0

    @property
    def passes_ld_floor(self) -> bool:
        """True iff all optimized exponents are >= 0.15."""
        return all(v >= 0.15 for v in self.optimal_exponents.values())

    def summary(self) -> str:
        lines = [
            f"Stage 2 -- {self.compound} {self.method}",
            f"  free params: {', '.join(self.free_params)}",
        ]
        lines.append("  exponents (seed -> optimised):")
        for name in self.free_params:
            s = self.starting_exponents.get(name, float("nan"))
            o = self.optimal_exponents.get(name, float("nan"))
            delta_pct = 100.0 * (o - s) / s if s else float("nan")
            lines.append(f"    {name:<20} {s:.6f} -> {o:.6f}  ({delta_pct:+.1f}%)")
        lines.append(
            f"  energy:  {self.starting_energy:.10f} -> "
            f"{self.optimal_energy:.10f} Ha  "
            f"({self.energy_improvement_mha:+.4f} mHa)"
        )
        lines.append(
            f"  BOBYQA:  {self.bobyqa_result.n_evaluations} evals, "
            f"{self.bobyqa_result.wall_seconds:.1f}s, "
            f"converged={self.bobyqa_result.success}"
        )
        if self.minuit_result:
            lines.append(
                f"  MIGRAD:  {self.minuit_result.n_evaluations} evals, "
                f"{self.minuit_result.wall_seconds:.1f}s, "
                f"converged={self.minuit_result.success}"
            )
        lines.append(f"  LD floor 0.15: {'PASS' if self.passes_ld_floor else 'FAIL'}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _find_pob_source(Z: int, basis: str = "pob-TZVP") -> Path:
    """Locate a pob per-element CRYSTAL source file."""
    from .. import __file__ as bo_file

    worktree = Path(bo_file).resolve().parents[3]
    src = worktree / "python" / "vibeqc" / "basis_library" / "sources" / basis
    candidates = sorted(src.glob(f"{Z:02d}_*"))
    if not candidates:
        raise FileNotFoundError(f"no source for Z={Z} in {src}")
    return candidates[0]


def run_stage2_lih(
    transport: Transport,
    *,
    basis: str = "pob-TZVP",
    method: str = "rhf",
    perturbation: float = 1.5,
    bounds: tuple[float, float] = (0.10, 5.0),
    crystal_wrapper: str = "crystal",
    max_eval: int = 50,
    run_minuit: bool = False,
    **transport_kw,
) -> Stage2Result:
    """Run Stage 2: optimise H + Li valence exponents on LiH.

    Uses pob-TZVP as the seed basis.  Three free parameters:
    * H diffuse-most s exponent (last s shell)
    * Li outermost s exponent (last s shell)
    * Li outermost p exponent (last p shell)

    Parameters
    ----------
    transport
        Transport for CRYSTAL14 submission on LiH.
    basis
        Seed basis name (default ``"pob-TZVP"``).
    method
        ``"rhf"``, ``"hf"``, or a DFT functional keyword.
    perturbation
        Multiplicative perturbation from the seed values.
    bounds
        (lower, upper) bounds on exponents in physical space.
    crystal_wrapper
        Path to CRYSTAL14 wrapper.
    max_eval
        Maximum BOBYQA evaluations.
    run_minuit
        If True, run iminuit MIGRAD+HESSE at the BOBYQA optimum
        for publication-grade uncertainties.
    **transport_kw
        cpus, wall_time_s, timeout_s -- passed through.

    Returns
    -------
    Stage2Result
    """
    # Parse seed basis for Li and H.
    h_atom = parse_crystal_atom_basis_file(_find_pob_source(1, basis=basis))
    li_atom = parse_crystal_atom_basis_file(_find_pob_source(3, basis=basis))

    # Identify free shells.
    # H: diffuse-most s = last s shell.
    h_s_shells = [(i, sh) for i, sh in enumerate(h_atom.shells) if sh.shell_type == "S"]
    h_diffuse_idx = h_s_shells[-1][0]

    # Li: outermost s and p.
    li_s_shells = [
        (i, sh) for i, sh in enumerate(li_atom.shells) if sh.shell_type == "S"
    ]
    li_p_shells = [
        (i, sh) for i, sh in enumerate(li_atom.shells) if sh.shell_type == "P"
    ]
    li_s_outer_idx = li_s_shells[-1][0] if li_s_shells else None
    li_p_outer_idx = li_p_shells[-1][0] if li_p_shells else None

    if li_s_outer_idx is None or li_p_outer_idx is None:
        raise RuntimeError(
            f"Li basis from {_find_pob_source(3, basis=basis)}: missing s or p shells"
        )

    free_specs = [
        FreeSpec(
            symbol="H",
            shell_idx=h_diffuse_idx,
            prim_idx=0,
            field="exponent",
            transform=Transform.LOG,
            bounds=bounds,
            label="H_s_diffuse",
        ),
        FreeSpec(
            symbol="Li",
            shell_idx=li_s_outer_idx,
            prim_idx=0,
            field="exponent",
            transform=Transform.LOG,
            bounds=bounds,
            label="Li_s_outer",
        ),
        FreeSpec(
            symbol="Li",
            shell_idx=li_p_outer_idx,
            prim_idx=0,
            field="exponent",
            transform=Transform.LOG,
            bounds=bounds,
            label="Li_p_outer",
        ),
    ]

    p = BasisParametrisation(
        atoms={"H": h_atom, "Li": li_atom},
        free=free_specs,
    )

    # Starting values (seed exponents).
    h_start = h_atom.shells[h_diffuse_idx].exponents[0]
    li_s_start = li_atom.shells[li_s_outer_idx].exponents[0]
    li_p_start = li_atom.shells[li_p_outer_idx].exponents[0]

    starting_values = {
        "H_s_diffuse": h_start,
        "Li_s_outer": li_s_start,
        "Li_p_outer": li_p_start,
    }

    # Objective: LiH rocksalt crystal.
    lih = STRUCTURES["LiH"]
    objective = make_crystal_objective(
        parametrisation=p,
        transport=transport,
        kind="crystal",
        structure=lih,
        method=method,
        crystal_wrapper=crystal_wrapper,
        workdir_prefix="stage2_lih",
        **transport_kw,
    )

    # Starting energy at perturbed x0.
    x0 = p.pack() + math.log(perturbation)
    e_start = objective(x0)

    notes: list[str] = []

    # BOBYQA.
    log_bounds = [(math.log(bounds[0]), math.log(bounds[1]))] * 3
    bobyqa = optimize_nlopt(
        objective,
        x0,
        bounds=log_bounds,
        max_eval=max_eval,
        tol=1e-4,
    )

    # MIGRAD refinement (optional).
    minuit_result: Optional[OptResult] = None
    if run_minuit:
        try:
            minuit_result = optimize_minuit(
                objective,
                bobyqa.x,
                bounds=[log_bounds[0], log_bounds[1], log_bounds[2]],
                labels=["H_s_diffuse", "Li_s_outer", "Li_p_outer"],
                max_iter=max_eval // 2,
            )
        except RuntimeError:
            notes.append("MIGRAD skipped -- iminuit not installed")

    # Optimal exponents.
    opt_atoms = p.unpack(minuit_result.x if minuit_result else bobyqa.x)
    opt_h = opt_atoms["H"].shells[h_diffuse_idx].exponents[0]
    opt_li_s = opt_atoms["Li"].shells[li_s_outer_idx].exponents[0]
    opt_li_p = opt_atoms["Li"].shells[li_p_outer_idx].exponents[0]

    optimal_values = {
        "H_s_diffuse": opt_h,
        "Li_s_outer": opt_li_s,
        "Li_p_outer": opt_li_p,
    }

    best_x = minuit_result.x if minuit_result else bobyqa.x
    best_fun = float(objective(best_x))

    # Acceptance checks.
    if best_fun >= e_start:
        notes.append("WARNING: energy did not decrease -- check basis/SCF convergence")
    if not bobyqa.success:
        notes.append(f"BOBYQA did not converge: {bobyqa.message}")
    for name, val in optimal_values.items():
        if val < 0.15:
            notes.append(
                f"WARNING: {name}={val:.4f} is below 0.15 LD floor -- "
                f"MIGRAD may be bunching at the bound"
            )

    return Stage2Result(
        compound="LiH",
        method=method.upper(),
        free_params=list(starting_values.keys()),
        starting_exponents=starting_values,
        optimal_exponents=optimal_values,
        starting_energy=e_start,
        optimal_energy=best_fun,
        bobyqa_result=bobyqa,
        minuit_result=minuit_result,
        transport=type(transport).__name__,
        notes=notes,
    )


if __name__ == "__main__":
    from vibe_basis.transports.local import LocalTransport

    result = run_stage2_lih(LocalTransport())
    print(result.summary())
