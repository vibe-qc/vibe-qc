
"""H-chain Born-von-Kármán sweep: does a Γ supercell reproduce the k mesh?

Companion to the H-chain BvK tutorial. For each per-pair spacing
R_pair, runs the SAME physical chain in several unit-cell
representations and reports the per-pair energy.

Born-von-Kármán theory says a supercell of N cells at Γ and the
primitive cell on an N x 1 x 1 k mesh describe the same periodic torus,
so their per-pair energies must agree to numerical precision. Those
same-torus pairs are compared explicitly at the end of each block. The
other representations (the H2 cell at Γ, and tori of different size)
differ by genuine finite-size error, which shrinks as the pairs move
apart.

The chain is a ``dim=3`` cell with ``PAD`` bohr of vacuum on the two
transverse axes, run with k-point GDF (``run_krhf_periodic_gdf``):
vibe-qc has no route with a consistent 1-D Coulomb gauge for a
``PeriodicSystem(dim=1, ...)``. Every representation shares the same
vacuum, so it cancels from the same-torus comparison.

Run:

    .venv/bin/python examples/periodic/input-h-chain-bvk-sweep.py
    .venv/bin/python examples/periodic/input-h-chain-bvk-sweep.py --with-h32

The H16 supercell dominates the cost. ``--with-h32`` adds the H32
supercell and its 16 x 1 x 1 partner; one H32 Γ SCF alone can take
more than half an hour, so leave it for a machine with time to spare.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
OUT  = HERE / STEM

R_S  = 1.4
R_LS = [2.5, 4.0, 6.0, 10.0, 18.0]   # → R_pair = 3.9, 5.4, 7.4, 11.4, 19.4
PAD  = 15.0                          # transverse vacuum (y, z), bohr

REPRESENTATIONS = [
    (1,  (1, 1, 1)),    # H2  cell, Γ
    (1,  (4, 1, 1)),    # H2  cell, 4x1x1 k-mesh
    (1,  (8, 1, 1)),    # H2  cell, 8x1x1 k-mesh
    (4,  (1, 1, 1)),    # H8  supercell, Γ (4 H2 pairs): same torus as 4x1x1
    (8,  (1, 1, 1)),    # H16 supercell, Γ (8 H2 pairs): same torus as 8x1x1
]
SAME_TORUS = [((1, (4, 1, 1)), (4, (1, 1, 1))), ((1, (8, 1, 1)), (8, (1, 1, 1)))]
if "--with-h32" in sys.argv[1:]:
    REPRESENTATIONS += [(1, (16, 1, 1)), (16, (1, 1, 1))]
    SAME_TORUS += [((1, (16, 1, 1)), (16, (1, 1, 1)))]


def label(n_pairs: int, kmesh: tuple[int, int, int]) -> str:
    return f"H{2 * n_pairs:<2d} {kmesh[0]}x{kmesh[1]}x{kmesh[2]}"


def run(n_pairs: int, R_pair: float, kmesh: tuple[int, int, int]) -> dict:
    cell: list[vq.Atom] = []
    for i in range(n_pairs):
        cell.append(vq.Atom(1, [i * R_pair + 0.0, 0.0, 0.0]))
        cell.append(vq.Atom(1, [i * R_pair + R_S, 0.0, 0.0]))
    sysp = vq.PeriodicSystem(
        dim=3, lattice=np.diag([n_pairs * R_pair, PAD, PAD]),
        unit_cell=cell,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-9
    opts.max_iter        = 80
    t0 = time.perf_counter()
    r = vq.run_krhf_periodic_gdf(sysp, basis, kmesh, opts, progress=False)
    return dict(
        n_pairs=n_pairs, kmesh=kmesh,
        E_total=float(r.energy),
        E_per_pair=float(r.energy) / n_pairs,
        converged=bool(r.converged),
        n_iter=int(r.n_iter),
        wall_s=time.perf_counter() - t0,
    )


def main() -> None:
    out_lines: list[str] = [f"vibeqc {vq.__version__}",
                            f"transverse vacuum {PAD} bohr, sto-3g, k-point GDF", ""]

    def emit(line: str = "") -> None:
        print(line, flush=True)
        out_lines.append(line)

    sep = "-" * 88
    emit(f"{'R_pair':>9s} | {'representation':<22s} | "
         f"{'E_total (Ha)':>14s} | {'E/pair (Ha)':>14s} | conv | iter")
    emit(sep)

    summary: list[tuple[float, list[float]]] = []
    for R_L in R_LS:
        R_pair = R_S + R_L
        per_rep: dict[tuple, dict] = {}
        for n_pairs, kmesh in REPRESENTATIONS:
            res = run(n_pairs, R_pair, kmesh)
            per_rep[(n_pairs, kmesh)] = res
            emit(f"{R_pair:>9.3f} | {label(n_pairs, kmesh):<22s} | "
                 f"{res['E_total']:>14.8f} | {res['E_per_pair']:>14.8f} | "
                 f"  {'✓' if res['converged'] else '✗'}  | {res['n_iter']:>3d}")
        diffs = []
        for k_rep, g_rep in SAME_TORUS:
            a, b = per_rep[k_rep], per_rep[g_rep]
            if a["converged"] and b["converged"]:
                d = b["E_per_pair"] - a["E_per_pair"]
                diffs.append(d)
                emit(f"  → same torus, {label(*g_rep)} minus {label(*k_rep)}: "
                     f"{d:+.3e} Ha/pair ({d * 1e6:+.2f} µHa/pair)")
        summary.append((R_pair, diffs))
        emit(sep)

    emit()
    emit("=== Summary: same-torus difference (Γ supercell minus k mesh) ===")
    emit(f"{'R_pair (bohr)':>14s} | " + " | ".join(
        f"{label(*g):>10s} - {label(*k):<10s}" for k, g in SAME_TORUS))
    for R_pair, diffs in summary:
        emit(f"{R_pair:>14.3f} | " + " | ".join(
            f"{d * 1e6:>+16.2f} µHa  " for d in diffs))

    out_path = OUT.with_suffix(".out")
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
