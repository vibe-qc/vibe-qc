"""Per-iteration SCF trajectory recorder + plot — divergence forensics.

Wraps a real periodic SCF in a custom ``ProgressLogger`` subclass
that captures every iteration's (energy, dE, ||[F,DS]||, DIIS dim)
into in-memory arrays — independent of whether the SCF converges
or diverges — and then plots them on a 4-panel matplotlib figure.

Use case: "the SCF says it didn't converge, but I want to SEE what
its trajectory looked like." A wildly oscillating energy with a
flat ||[F,DS]|| → DIIS instability. A monotonically rising energy
with growing ||[F,DS]|| → bad initial guess + wrong sign on the
update. A flat ||[F,DS]|| at high value → stuck in a non-stationary
saddle.

Reads result.scf_trace if present (every Python-driven SCF
populates it) and falls back to the recorder's in-memory list
if the result.scf_trace is empty (some C++-driven paths).

Outputs:
    output/iteration-recorder/trajectory.png   — 2x2 panel:
        |dE| vs iter         (log-y; should drop monotonically)
        ||[F,DS]|| vs iter   (log-y; commutator norm)
        E vs iter            (linear-y; should plateau, not bounce)
        DIIS subspace vs iter (linear-y; should grow toward subspace size)

    output/iteration-recorder/trajectory.csv   — same data as CSV

Run:
    .venv/bin/python examples/debug/scf_iteration_recorder.py

Bring your own system: edit the build_system() call at the bottom.
The default is the same NaCl/STO-3G/RKS-LDA debug substrate the
other examples use.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

import vibeqc as vq
from vibeqc.progress import ProgressLogger

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "iteration-recorder"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrajectoryRecorder(ProgressLogger):
    """ProgressLogger that captures per-iter rows to memory.

    Subclasses ProgressLogger so the SCF can pass it as
    ``progress=recorder`` and live progress still emits to
    stdout / log_path. The recorder appends each iter's payload
    to ``rows`` regardless of the verbose level.
    """

    rows: List[dict] = field(default_factory=list)

    def iteration(self, n: int, **fields: Any) -> None:
        self.rows.append({
            "iter": int(n),
            "energy": fields.get("energy"),
            "dE": fields.get("dE"),
            "grad": fields.get("grad"),
            "diis": fields.get("diis"),
        })
        super().iteration(n, **fields)


def build_system() -> tuple[vq.PeriodicSystem, str]:
    """Default debug substrate: NaCl/STO-3G."""
    a = 5.640 / 0.529177210903
    NA = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    CL = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = (
        [vq.Atom(11, [f * a for f in p]) for p in NA] +
        [vq.Atom(17, [f * a for f in p]) for p in CL]
    )
    sys_ = vq.PeriodicSystem(3, a * np.eye(3), cell)
    return sys_, "sto-3g"


def make_opts() -> vq.PeriodicKSOptions:
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.lattice_opts.cutoff_bohr = 10.0
    o.lattice_opts.nuclear_cutoff_bohr = 18.0
    o.conv_tol_energy = 1e-7
    o.max_iter = 30           # bounded for divergence cases
    return o


def main() -> None:
    print("=" * 72)
    print(" Per-iteration recorder + trajectory plot")
    print("=" * 72)

    system, basis_name = build_system()
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    print(f"  cell:  {len(system.unit_cell)} atoms, {basis.nbasis} bf "
          f"(basis: {basis_name})")

    recorder = TrajectoryRecorder(verbose=4)

    print("\nRunning SCF (live trace below; trajectory recorded to disk):")
    try:
        with vq.crash_dump_context(OUT_DIR / "trajectory"):
            with vq.perf_log(OUT_DIR / "trajectory.perf"):
                result = vq.run_rks_periodic_scf(
                    system, basis, vq.KPoints.gamma(system),
                    make_opts(), progress=recorder,
                    spacing_bohr=0.5,
                )
        print(f"\n  E      = {result.energy:.10f} Ha")
        print(f"  iters  = {result.n_iter} "
              f"({'converged' if result.converged else 'NOT'})")
    except Exception as exc:
        print(f"\n  SCF FAILED: {type(exc).__name__}: {str(exc)[:80]}")
        print(f"  see {OUT_DIR}/trajectory.dump for last-iter state")

    if not recorder.rows:
        print("  (no iterations recorded — SCF may have failed before iter 1)")
        return

    # CSV
    csv_path = OUT_DIR / "trajectory.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iter", "energy", "dE", "grad", "diis"])
        for r in recorder.rows:
            w.writerow([r["iter"], r["energy"], r["dE"], r["grad"], r["diis"]])
    print(f"  CSV:  {csv_path.relative_to(HERE.parent.parent)}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    iters = np.asarray([r["iter"] for r in recorder.rows])
    energies = np.asarray([r["energy"] for r in recorder.rows
                           if r["energy"] is not None], dtype=float)
    iters_e = np.asarray([r["iter"] for r in recorder.rows
                          if r["energy"] is not None])
    dEs = np.asarray([abs(r["dE"]) for r in recorder.rows
                      if r["dE"] is not None], dtype=float)
    iters_dE = np.asarray([r["iter"] for r in recorder.rows
                           if r["dE"] is not None])
    grads = np.asarray([r["grad"] for r in recorder.rows
                        if r["grad"] is not None], dtype=float)
    iters_g = np.asarray([r["iter"] for r in recorder.rows
                          if r["grad"] is not None])
    diis = np.asarray([r["diis"] for r in recorder.rows
                       if r["diis"] is not None], dtype=int)
    iters_d = np.asarray([r["iter"] for r in recorder.rows
                          if r["diis"] is not None])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=130)

    if dEs.size:
        ax = axes[0, 0]
        ax.semilogy(iters_dE, dEs, marker="o", color="C0", linewidth=1.4)
        ax.axhline(1e-7, color="grey", linestyle="--", alpha=0.5,
                   label="conv_tol_energy = 1e-7")
        ax.set_xlabel("iter")
        ax.set_ylabel("|ΔE| (Ha)  [log]")
        ax.set_title("Energy convergence")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    if grads.size:
        ax = axes[0, 1]
        ax.semilogy(iters_g, grads, marker="s", color="C1", linewidth=1.4)
        ax.set_xlabel("iter")
        ax.set_ylabel("‖[F, DS]‖  [log]")
        ax.set_title("Commutator norm")
        ax.grid(alpha=0.3, which="both")

    if energies.size:
        ax = axes[1, 0]
        ax.plot(iters_e, energies, marker="o", color="C2", linewidth=1.4)
        ax.set_xlabel("iter")
        ax.set_ylabel("E (Ha)")
        ax.set_title("Total energy trajectory")
        ax.grid(alpha=0.3)

    if diis.size:
        ax = axes[1, 1]
        ax.step(iters_d, diis, where="mid", marker="d",
                color="C3", linewidth=1.4)
        ax.set_xlabel("iter")
        ax.set_ylabel("DIIS subspace dim")
        ax.set_title("DIIS subspace growth")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    fig.suptitle(f"SCF trajectory — {basis_name}, "
                 f"{len(system.unit_cell)} atoms, "
                 f"{basis.nbasis} bf",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    png_path = OUT_DIR / "trajectory.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG:  {png_path.relative_to(HERE.parent.parent)}")
    print()
    print("  Inspect the panels:")
    print("    |ΔE|       monotone drop → convergence; bounce → DIIS instability")
    print("    ‖[F,DS]‖   should approach 0; flat → stuck non-stationary")
    print("    E          should plateau; oscillation → DIIS broken")
    print("    DIIS dim   grows toward diis_subspace_size (8 by default)")


if __name__ == "__main__":
    main()
