#!/usr/bin/env python3
"""Stage-1 driver comparison: every optimiser on the same single-atom problem.

The problem is the Stage-1 smoke of ``recipes/single_atom.py``: pob-TZVP H,
its diffuse-most s exponent free in log space with the physical box
(0.05, 1.0), started at ``perturbation`` times the published value, and the
objective is the UHF energy of the isolated H atom (doublet), optionally
plus the OPTBASIS ``gamma ln kappa(S)`` conditioning penalty.

Every driver in ``vibeqc.basis_optimization`` runs from that same point with
the same bounds, and the table reports what the roadmap's Phase 3 asks for:
objective evaluations (each one an SCF), analytic-gradient evaluations (each
one an SCF plus the exponent-derivative integrals), wall-clock, the final
objective and the final exponent. A driver whose backend is not installed
is reported as skipped rather than failing the run.

Usage::

    .venv/bin/python scripts/basisset_dev/driver_benchmark_stage1.py
    .venv/bin/python scripts/basisset_dev/driver_benchmark_stage1.py --penalty --json out.json

Needs a built vibe-qc; the optional backends come from the
``[basisopt-drivers]`` extra. The Markdown table it prints is what
``docs/basisset_dev/VERIFICATION_REPORT.md`` records; the numbers there carry
the vibe-qc version they were measured with.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


class Counted:
    """Wrap a closure objective so drivers can read ``n_calls``."""

    def __init__(self, fn: Callable[[np.ndarray], float], n_params: int) -> None:
        self._fn = fn
        self.n_params = n_params
        self.n_calls = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n_calls += 1
        return float(self._fn(x))


class CountedGrad:
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn
        self.n_calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        return np.asarray(self._fn(x), dtype=float)


def _row(name: str, res: Any, obj: Counted, grad: Optional[CountedGrad],
         par, wall: float) -> dict[str, Any]:
    exponent = float(par.unpack(np.asarray(res.x, dtype=float))["H"].shells[2].exponents[0])
    return {
        "driver": name,
        "objective_evals": int(obj.n_calls),
        "gradient_evals": int(grad.n_calls) if grad is not None else 0,
        "objective": float(res.fun),
        "exponent": exponent,
        "converged": bool(res.success),
        "message": str(res.message),
        "wall_s": float(wall),
    }


def _skipped(name: str, why: str) -> dict[str, Any]:
    return {
        "driver": name, "objective_evals": 0, "gradient_evals": 0,
        "objective": float("nan"), "exponent": float("nan"),
        "converged": False, "message": f"skipped: {why}", "wall_s": 0.0,
    }


def run(perturbation: float, bounds: tuple[float, float], penalty: bool,
        cond_gamma: float, seed: int) -> dict[str, Any]:
    import vibeqc as vq
    from vibeqc.basis_crystal import parse_crystal_atom_basis_file
    from vibeqc.basis_optimization import (
        optimize_bdiis, optimize_bobyqa, optimize_cmaes, optimize_minuit,
        optimize_scipy,
    )
    from vibeqc.basis_optimization.io import TempBasisLibrary
    from vibeqc.basis_optimization.parametrise import (
        BasisParametrisation, FreeSpec, Transform,
    )
    from vibeqc.basis_optimization.recipes.global_then_polish import (
        optimize_global_then_polish,
    )
    from vibeqc.basis_optimization.recipes.objective import (
        make_uhf_native_objective_gradient,
    )

    src_h = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP" / "01_H"
    par = BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(src_h)},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG,
                       bounds=bounds, label="H_diffuse_s")],
    )
    published = float(par.atoms["H"].shells[2].exponents[0])
    x0 = par.pack() + np.log(perturbation)
    box = par.optim_bounds()

    def molecule():
        return vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)

    # Import the optional backends up front so the first call of a driver is
    # not charged for loading its package (DFO-LS and Py-BOBYQA pull pandas,
    # about a second) inside the timed region.
    for backend in ("pybobyqa", "dfols", "cma", "iminuit"):
        try:
            __import__(backend)
        except ImportError:
            pass

    rows: list[dict[str, Any]] = []
    with TempBasisLibrary() as lib:
        raw_obj, raw_grad = make_uhf_native_objective_gradient(
            par, molecule, lib, use_cond_penalty=penalty, cond_gamma=cond_gamma,
        )
        e_start = float(raw_obj(x0))

        def fresh():
            return Counted(raw_obj, 1), CountedGrad(raw_grad)

        def timed(name: str, call: Callable[[Counted, CountedGrad], Any],
                  *, uses_grad: bool) -> None:
            obj, grad = fresh()
            t0 = time.perf_counter()
            try:
                res = call(obj, grad)
            except RuntimeError as exc:  # optional backend not installed
                rows.append(_skipped(name, str(exc).split(";")[0]))
                return
            rows.append(_row(name, res, obj, grad if uses_grad else None, par,
                             time.perf_counter() - t0))

        timed("scipy L-BFGS-B (FD)",
              lambda o, g: optimize_scipy(o, x0, bounds=box, tol=1e-8, max_iter=50),
              uses_grad=False)
        timed("minuit MIGRAD (FD)",
              lambda o, g: optimize_minuit(o, x0, bounds=box, labels=par.labels(),
                                           error_init=np.array([0.1]), max_iter=200),
              uses_grad=False)
        timed("bdiis (FD gradient)",
              lambda o, g: optimize_bdiis(o, x0, bounds=box, tol_energy=1e-8, tol_grad=1e-5),
              uses_grad=False)
        timed("bdiis (analytic gradient)",
              lambda o, g: optimize_bdiis(o, x0, bounds=box, grad=g, tol_energy=1e-8, tol_grad=1e-5),
              uses_grad=True)
        timed("bobyqa (Py-BOBYQA)",
              lambda o, g: optimize_bobyqa(o, x0, bounds=box, rhoend=1e-6, max_evaluations=200),
              uses_grad=False)
        timed("cmaes (pycma)",
              lambda o, g: optimize_cmaes(o, x0, bounds=box, seed=seed, tol_fun=1e-9,
                                          max_evaluations=400),
              uses_grad=False)
        timed("cmaes + bdiis (analytic)",
              lambda o, g: optimize_global_then_polish(
                  o, x0, bounds=box, grad=g, cmaes_max_evaluations=60, cmaes_seed=seed,
                  cmaes_tol_fun=1e-6, bdiis_tol_energy=1e-8, bdiis_tol_grad=1e-5),
              uses_grad=True)

    return {
        "vibeqc_version": str(vq.__version__),
        "python": platform.python_version(),
        "problem": {
            "basis": "pob-TZVP H", "free": "shell 2 primitive 0 exponent (diffuse s)",
            "reference": "UHF, isolated H atom (doublet)",
            "published_exponent": published,
            "start_exponent": published * perturbation,
            "bounds": list(bounds),
            "penalty": f"gamma ln kappa, gamma={cond_gamma}" if penalty else "none",
            "start_objective": e_start,
        },
        "rows": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    p = report["problem"]
    lines = [
        f"Problem: {p['basis']}, {p['free']}, {p['reference']}; "
        f"start exponent {p['start_exponent']:.6f} (published {p['published_exponent']:.6f}), "
        f"box {tuple(p['bounds'])}, penalty {p['penalty']}; "
        f"start objective {p['start_objective']:.9f} Ha; vibe-qc {report['vibeqc_version']}.",
        "",
        "| driver | objective evals | gradient evals | final objective (Ha) | final exponent | converged | wall (s) |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for r in report["rows"]:
        if r["message"].startswith("skipped"):
            lines.append(f"| {r['driver']} | | | {r['message']} | | | |")
            continue
        lines.append(
            f"| {r['driver']} | {r['objective_evals']} | {r['gradient_evals']} | "
            f"{r['objective']:.9f} | {r['exponent']:.6f} | "
            f"{'yes' if r['converged'] else 'no'} | {r['wall_s']:.2f} |"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--perturbation", type=float, default=1.5,
                    help="start at this multiple of the published exponent (default 1.5)")
    ap.add_argument("--bounds", type=float, nargs=2, default=(0.05, 1.0),
                    metavar=("LO", "HI"), help="physical box on the exponent")
    ap.add_argument("--penalty", action="store_true",
                    help="add the gamma ln kappa(S) conditioning penalty")
    ap.add_argument("--gamma", type=float, default=1e-3, help="penalty weight (Ha)")
    ap.add_argument("--seed", type=int, default=1, help="CMA-ES seed")
    ap.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = ap.parse_args(argv)

    report = run(args.perturbation, tuple(args.bounds), args.penalty, args.gamma, args.seed)
    print(markdown(report))
    for r in report["rows"]:
        print(f"  {r['driver']}: {r['message']}")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
