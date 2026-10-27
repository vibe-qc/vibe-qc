#!/usr/bin/env python
"""vq job: the per-cell spin sector on both neutral producers (#519 RG-14).

The smallest complete UHF and UKS/PBE calculations through the public runner
that exercise the explicit spin-state path: the Li doublet cube (STO-3G,
7 bohr) at ``(2,1,1)`` with ``aiccm_spin_state="per-cell"`` on
``variant="real-gamma"`` and ``variant="neutral-bloch"``, plus the legacy
pair for the record. The job refuses a runtime that predates the contract
instead of reporting a number from the wrong source.

Submit it on a runtime that contains the landed source, for example::

    vq submit HOST studies/aiccm-2026/spin_state_vq_job.py --program vibeqc-dev

It writes ``spin_state_rows.json`` and the runner's ``.out`` / ``.system``
files into the job workspace. The matched PySCF reference is produced by
``studies/aiccm-2026/spin_state_pyscf.py`` on the same cell and mesh.
"""
from __future__ import annotations

import json
import sys
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

import vibeqc as vq  # noqa: E402
from vibeqc import Atom, PeriodicSystem  # noqa: E402
from vibeqc._vibeqc_core import BasisSet  # noqa: E402

try:
    from vibeqc.periodic.ccm.spin_state import CCMSpinState  # noqa: F401
except ImportError:
    sys.exit("this runtime predates the explicit spin-state contract (#519 "
             "RG-14); the job needs a runtime built from main at or after "
             "the landing commit")

from vibeqc.periodic_runner import run_periodic_job  # noqa: E402

MESH = (2, 1, 1)
OUT = Path("spin_state_rows.json")


def _cell():
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)


def _runtime_identity() -> dict:
    """Version, package path and, when the runtime is a checkout, its commit."""
    import subprocess

    ident = {"vibeqc_version": vq.__version__, "vibeqc_file": vq.__file__}
    try:
        ident["source_commit"] = subprocess.run(
            ["git", "-C", str(Path(vq.__file__).resolve().parent),
             "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        ident["source_commit"] = None
    return ident


def main() -> int:
    cell = _cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    rows = []
    for policy in ("per-cell", "legacy"):
        for method, functional in (("uhf", None), ("uks", "pbe")):
            for variant in ("real-gamma", "neutral-bloch"):
                name = f"li_{method}_{variant}_{policy}"
                t0 = time.perf_counter()
                res = run_periodic_job(
                    cell, basis, method="aiccm", variant=variant,
                    functional=functional, aiccm_lattice_extension=MESH,
                    aiccm_spin_state=policy, initial_guess="HCORE",
                    progress=False, output=name)
                run = tomllib.loads(
                    Path(f"{name}.system").read_text(encoding="utf-8"))["run"]
                rows.append({
                    "name": name, "policy": policy, "method": method,
                    "variant": variant, "mesh": list(MESH),
                    "energy_per_cell": float(res.energy),
                    "converged": bool(res.converged),
                    "spin_state": run.get("aiccm_spin_state"),
                    "neutral_source_identity": run.get("neutral_source_identity"),
                    "wall_s": time.perf_counter() - t0,
                })
                print(f"{name:40s} E/cell = {float(res.energy):.10f} "
                      f"state = {run.get('aiccm_spin_state', {}).get('n_alpha')}/"
                      f"{run.get('aiccm_spin_state', {}).get('n_beta')}", flush=True)
    OUT.write_text(json.dumps({"runtime": _runtime_identity(), "rows": rows}, indent=2))
    # The same-sector pairs must agree; the legacy pairs must not be compared.
    by = {(r["policy"], r["method"], r["variant"]): r for r in rows}
    for method in ("uhf", "uks"):
        a = by[("per-cell", method, "real-gamma")]
        b = by[("per-cell", method, "neutral-bloch")]
        d = a["energy_per_cell"] - b["energy_per_cell"]
        print(f"per-cell {method}: real-gamma - bloch = {d:.3e} Ha/cell "
              f"({'PASS' if abs(d) < 1e-8 else 'FAIL'} at 1e-8, runner default convergence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
