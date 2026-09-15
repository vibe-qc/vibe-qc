"""Stage 1 recipe: single-atom HF with CRYSTAL14 + NLopt BOBYQA.

The first non-trivial test of the CRYSTAL14-based optimisation
pipeline (Goal 8).  We take the H atom's diffuse-most s exponent
from pob-TZVP-rev2, perturb it upward by a factor of 1.5, and
let NLopt BOBYQA minimise the CRYSTAL14 HF energy.

The objective function uses :func:`vibe_basis.backends.crystal_atom.emit_input_atom`
to generate a .d12 deck with an inline basis, submits it through
the transport (local or vq), parses the output, and returns the
energy.  Failed SCF evaluations return ``np.inf`` so BOBYQA routes
around them.

This recipe lives in vibe-qc's basis_optimization (can import
from both vibeqc and vibe_basis) rather than in vibe-basis's
recipes (which cannot import vibe-qc for basis manipulation).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from vibe_basis.backends.crystal import parse_output_file

# vibe-basis imports (allowed: vibe-qc -> vibe-basis, not the reverse)
from vibe_basis.backends.crystal_atom import emit_input_atom
from vibe_basis.optimize import OptResult, optimize_nlopt, optimize_scipy
from vibe_basis.transports.base import Transport, TransportError

from ...basis_crystal import emit_crystal, parse_crystal_atom_basis_file
from ..parametrise import BasisParametrisation, FreeSpec, Transform


@dataclass
class Stage1Result:
    """Outcome of a single-atom CRYSTAL14 optimisation."""

    element: str
    method: str
    free_field: str
    starting_value: float
    optimal_value: float
    starting_energy: float
    optimal_energy: float
    optim_result: OptResult
    transport: str

    def summary(self) -> str:
        delta_e_mha = (self.starting_energy - self.optimal_energy) * 1000.0
        delta_x_pct = (
            100.0 * (self.optimal_value - self.starting_value) / self.starting_value
            if self.starting_value
            else float("nan")
        )
        return (
            f"Stage 1 -- {self.element} atom CRYSTAL14 {self.method}\n"
            f"  free param:  {self.free_field}\n"
            f"  start:       x = {self.starting_value:.6f}  "
            f"E = {self.starting_energy:.10f} Ha\n"
            f"  optimum:     x = {self.optimal_value:.6f}  "
            f"E = {self.optimal_energy:.10f} Ha\n"
            f"  Δx           {delta_x_pct:+7.2f} %\n"
            f"  ΔE           {delta_e_mha:+8.4f} mHa  (lower is better)\n"
            f"  evaluations: {self.optim_result.n_evaluations}  "
            f"wall: {self.optim_result.wall_seconds:.1f} s\n"
            f"  driver:      {self.optim_result.driver}  "
            f"converged: {self.optim_result.success}\n"
            f"  transport:   {self.transport}"
        )


def _find_pob_source(Z: int, basis: str = "pob-TZVP") -> Path:
    """Locate a pob per-element CRYSTAL source file."""
    from .. import __file__ as bo_file

    worktree = Path(bo_file).resolve().parents[3]
    src = worktree / "python" / "vibeqc" / "basis_library" / "sources" / basis
    if not src.exists():
        raise FileNotFoundError(f"pob source dir not found: {src}")
    candidates = sorted(src.glob(f"{Z:02d}_*"))
    if not candidates:
        raise FileNotFoundError(f"no source for Z={Z} in {src}")
    return candidates[0]


def make_atom_objective(
    Z: int,
    parametrisation: BasisParametrisation,
    transport: Transport,
    *,
    method: str = "rhf",
    crystal_wrapper: str = "crystal",
    cpus: int = 4,
    wall_time_s: int = 7200,
    timeout_s: float = 86_400.0,
) -> Callable[[np.ndarray], float]:
    """Build an objective function that evaluates the CRYSTAL14 energy.

    Parameters
    ----------
    Z
        Atomic number of the isolated atom.
    parametrisation
        Basis parametrisation with free parameter(s).
    transport
        Transport to use (LocalTransport or VqTransport).
    method
        CRYSTAL method: ``"rhf"``, ``"hf"``, ``"pw1pw"``, ...
    crystal_wrapper
        Path to CRYSTAL14 wrapper on the target machine.
    cpus, wall_time_s, timeout_s
        Passed to transport.

    Returns
    -------
    Callable[[np.ndarray], float]
        Objective function: parameter vector -> energy in Hartree.
        Returns ``np.inf`` on any failure (failed SCF, transport
        error, ...).
    """
    n_calls = [0]
    workdir_root = Path(f"stage1_Z{Z}")

    def objective(x: np.ndarray) -> float:
        n_calls[0] += 1
        iteration = n_calls[0]

        # Unpack parameters -> fresh basis copy.
        atoms = parametrisation.unpack(x)
        atom_basis = atoms[parametrisation.atoms_keys[0]]
        basis_text = emit_crystal([atom_basis])

        # Emit .d12 deck.
        deck = emit_input_atom(Z, basis_text, method=method)
        wd = workdir_root / f"iter_{iteration:03d}"
        if wd.exists():
            import shutil

            shutil.rmtree(wd)
        wd.mkdir(parents=True)
        (wd / "atom.d12").write_text(deck)

        # Submit -> wait -> fetch.
        try:
            job = transport.run(
                wd,
                command=[crystal_wrapper, "atom.d12"],
                dest=wd / "fetched",
                cpus=cpus,
                wall_time_s=wall_time_s,
                label=f"stage1/{Z}/iter{iteration:03d}",
                timeout_s=timeout_s,
            )
        except (TransportError, Exception):
            return float("inf")

        out_path = job.output_dir / "atom.out"
        try:
            parsed = parse_output_file(out_path)
        except FileNotFoundError:
            return float("inf")

        if not parsed.ok or parsed.energy is None:
            return float("inf")
        return parsed.energy

    return objective


def run_stage1_h_atom(
    transport: Transport,
    *,
    method: str = "rhf",
    perturbation: float = 1.5,
    bounds: tuple[float, float] = (0.05, 1.0),
    crystal_wrapper: str = "crystal",
    max_eval: int = 30,
    use_nlopt: bool = True,
    **transport_kw,
) -> Stage1Result:
    """Run Stage 1: optimise H's diffuse-most s exponent.

    Uses pob-TZVP as the seed basis (ships in the repo).  The
    diffuse-most s shell is shell[2] (0-indexed) with exponent
    ~0.1795.  We perturb it upward by *perturbation* (default
    1.5x) and let BOBYQA minimise.

    Parameters
    ----------
    transport
        Transport for CRYSTAL14 submission.
    method
        ``"rhf"`` or ``"hf"``.
    perturbation
        Multiplicative perturbation from the seed exponent.
    bounds
        (lower, upper) bounds on the exponent in physical space.
    crystal_wrapper
        Path to CRYSTAL14 wrapper on the target machine.
    max_eval
        Maximum BOBYQA evaluations.
    use_nlopt
        True -> NLopt BOBYQA; False -> scipy L-BFGS-B.
    **transport_kw
        cpus, wall_time_s, timeout_s -- passed through.

    Returns
    -------
    Stage1Result
    """
    src = _find_pob_source(1, basis="pob-TZVP")
    atom = parse_crystal_atom_basis_file(src)

    # Identify the diffuse-most s shell.
    s_shells = [(i, sh) for i, sh in enumerate(atom.shells) if sh.shell_type == "S"]
    if len(s_shells) < 3:
        raise RuntimeError(
            f"H atom from {src}: expected >=3 s shells, found {len(s_shells)}"
        )
    diffuse_idx = s_shells[-1][0]  # last s shell = most diffuse
    diffuse_shell = atom.shells[diffuse_idx]

    p = BasisParametrisation(
        atoms={"H": atom},
        free=[
            FreeSpec(
                symbol="H",
                shell_idx=diffuse_idx,
                prim_idx=0,
                field="exponent",
                transform=Transform.LOG,
                bounds=bounds,
                label="H_diffuse_s",
            )
        ],
    )

    starting_exp = diffuse_shell.exponents[0]
    x0 = p.pack() + math.log(perturbation)

    obj = make_atom_objective(
        1,
        p,
        transport,
        method=method,
        crystal_wrapper=crystal_wrapper,
        **transport_kw,
    )

    # Evaluate at start.
    e_start = obj(x0)

    # Optimise.
    if use_nlopt:
        log_bounds = [(math.log(bounds[0]), math.log(bounds[1]))]
        opt = optimize_nlopt(obj, x0, bounds=log_bounds, max_eval=max_eval)
    else:
        opt = optimize_scipy(obj, x0, bounds=[log_bounds[0]], max_iter=max_eval)

    opt_exp = p.unpack(opt.x)["H"].shells[diffuse_idx].exponents[0]
    transport_name = type(transport).__name__

    return Stage1Result(
        element="H",
        method=method.upper(),
        free_field=f"H diffuse-most s exponent (shell {diffuse_idx})",
        starting_value=starting_exp * perturbation,
        optimal_value=opt_exp,
        starting_energy=e_start,
        optimal_energy=opt.fun,
        optim_result=opt,
        transport=transport_name,
    )


if __name__ == "__main__":
    from vibe_basis.transports.local import LocalTransport

    result = run_stage1_h_atom(LocalTransport())
    print(result.summary())
