"""OpenMP scaling for a single SCF — tutorial 18 figure.

Times the same molecular RKS/PBE calculation at thread counts
1, 2, 4, 8 and writes
``docs/_static/plots/openmp-scaling.png`` — a two-panel figure:

  Left:  wall time vs threads on a log-log axis with the ideal-scaling
         reference line.
  Right: parallel speedup vs threads, with the ideal y = x reference.

The benchmark uses a glycine zwitterion at cc-pVDZ (10 atoms, 100 basis
functions) — large enough that the per-iteration ERI build dominates
the wall time, small enough to run in well under a minute on a laptop.
Scaling flattens past ~4 threads on this size of system, exactly the
behavior discussed in the tutorial text.

Run:
    .venv/bin/python examples/plots/openmp-scaling.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "openmp-scaling.png"

# Glycine zwitterion (NH3+CH2COO-) at a sensible MP2-optimized geometry.
# 10 atoms; 220 basis functions in cc-pVTZ — heavy enough that the SCF
# spends real time in the ERI builder and the XC grid.
GLYCINE_BOHR = [
    (7, [-2.5860,  1.4488, -0.0095]),
    (1, [-2.7625,  2.5101,  1.4017]),
    (1, [-2.3517,  3.0140, -1.2941]),
    (1, [-4.2376,  0.4844, -0.0707]),
    (6, [-0.5167,  0.0090,  0.0033]),
    (1, [-0.5919, -1.1737,  1.6378]),
    (1, [-0.6080, -1.1729, -1.6238]),
    (6, [ 1.9776,  1.4569, -0.0010]),
    (8, [ 4.0042,  0.2660, -0.0098]),
    (8, [ 1.6700,  3.7715,  0.0152]),
]

THREAD_COUNTS = [1, 2, 4, 8]
N_REPEATS = 2  # warm-up + measurement; report the second


def make_molecule() -> vq.Molecule:
    return vq.Molecule([vq.Atom(z, list(p)) for z, p in GLYCINE_BOHR])


def time_one(threads: int) -> float:
    """Time a fresh SCF at the given thread count. Returns wall seconds."""
    vq.set_num_threads(threads)
    mol = make_molecule()
    basis = vq.BasisSet(mol, "cc-pvdz")

    opts = vq.RKSOptions()
    opts.functional = "PBE"

    last = None
    for rep in range(N_REPEATS):
        t0 = time.perf_counter()
        r = vq.run_rks(mol, basis, opts)
        last = time.perf_counter() - t0
        assert r.converged, f"SCF did not converge at threads={threads}"
    return last


def main() -> None:
    print(f"vibe-qc OpenMP scaling on glycine / cc-pVDZ "
          f"({len(GLYCINE_BOHR)} atoms)")

    times = []
    for n in THREAD_COUNTS:
        t = time_one(n)
        times.append(t)
        print(f"  threads={n:3d}   wall = {t:6.2f} s")

    times = np.array(times)
    threads = np.array(THREAD_COUNTS, dtype=float)
    speedup = times[0] / times
    ideal = threads / threads[0]

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_t, ax_s) = plt.subplots(
        1, 2, figsize=(8.0, 3.8), dpi=150,
    )

    # --- Wall time vs threads (log-log) --------------------------------------
    ax_t.loglog(threads, times, "o-", color="#1f77b4", markersize=8,
                linewidth=1.8, label="vibe-qc")
    # Ideal scaling: t(n) = t(1) / n
    t_ideal = times[0] / threads
    ax_t.loglog(threads, t_ideal, color="#7f7f7f", linestyle="--",
                linewidth=1.2, label="ideal (1 / N)")

    ax_t.set_xlabel("Threads")
    ax_t.set_ylabel("Wall time per SCF (s)")
    ax_t.set_title("Wall time vs OpenMP threads")
    ax_t.set_xticks(THREAD_COUNTS)
    ax_t.set_xticklabels([str(n) for n in THREAD_COUNTS])
    ax_t.minorticks_off()
    ax_t.grid(alpha=0.3, which="both", linestyle=":")
    ax_t.legend(loc="upper right", frameon=True, fontsize=9)

    # --- Speedup vs threads --------------------------------------------------
    ax_s.plot(threads, speedup, "o-", color="#1f77b4", markersize=8,
              linewidth=1.8, label="vibe-qc")
    ax_s.plot(threads, ideal, color="#7f7f7f", linestyle="--",
              linewidth=1.2, label="ideal (y = N)")

    ax_s.set_xlabel("Threads")
    ax_s.set_ylabel("Parallel speedup (× over 1 thread)")
    ax_s.set_title("Speedup vs OpenMP threads")
    ax_s.set_xticks(THREAD_COUNTS)
    ax_s.set_xlim(0.5, max(THREAD_COUNTS) + 0.5)
    ax_s.set_ylim(0, max(THREAD_COUNTS) + 1)
    ax_s.grid(alpha=0.3, linestyle=":")
    ax_s.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle(
        "vibe-qc OpenMP scaling — glycine, RKS/PBE, cc-pVDZ "
        f"({len(GLYCINE_BOHR)} atoms, ~100 bf)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"  Best speedup: {speedup.max():.2f}× at "
          f"{int(THREAD_COUNTS[int(np.argmax(speedup))])} threads")
    print(f"  Parallel efficiency at {THREAD_COUNTS[-1]} threads: "
          f"{100 * speedup[-1] / threads[-1]:.0f} %")


if __name__ == "__main__":
    main()
