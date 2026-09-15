"""H-chain Born-von-Kármán sweep — does v0.7's gauge fix deliver
µHa-precision BvK at all per-pair spacings?

Companion to tutorial 30. For each per-pair spacing R_pair, runs
the SAME physical chain in five different unit-cell representations
and reports per-pair energy + max drift across representations.

Born-von-Kármán theory says all representations of the same physical
chain must give identical per-pair energies. Drift is the fingerprint
of finite-precision / open-bug effects. Result on v0.7.3 (compute-reference,
32 cores, ~11 s wall total):

  * R_pair = 19.4 bohr (isolated H₂ limit) — drift 0.21 µHa/pair  ✓
  * R_pair ≤ 11.4 bohr (chemically-bonded chain) — drift
    176 mHa to 1.14 Ha/pair  ✗  ← the open multi-cell density bug

Run on compute-reference via the vq queue tool:

    vq submit --cpus 4 examples/periodic/input-h-chain-bvk-sweep.py
    vq status <jobid> -n 0       # full output

(Don't run on a laptop — the H32 / large-supercell rows can OOM
on 16 GB. compute-reference has 128 GB.)

When the v0.7.x maintenance window closes the multi-cell density
bug, the small-spacing drift will collapse from mHa to µHa — re-run
this script to see it.
"""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
OUT  = HERE / STEM

R_S  = 1.4
R_LS = [2.5, 4.0, 6.0, 10.0, 18.0]   # → R_pair = 3.9, 5.4, 7.4, 11.4, 19.4
PAD  = 30.0

REPRESENTATIONS = [
    (1,  [1, 1, 1]),    # H2  cell, Γ
    (1,  [8, 1, 1]),    # H2  cell, 8x1x1 k-mesh
    (4,  [1, 1, 1]),    # H8  supercell, Γ (4 H2 pairs)
    (8,  [1, 1, 1]),    # H16 supercell, Γ (8 H2 pairs)
    (16, [1, 1, 1]),    # H32 supercell, Γ (16 H2 pairs)
]


def run(n_pairs: int, R_pair: float, kmesh: list[int]) -> dict:
    cell: list[vq.Atom] = []
    for i in range(n_pairs):
        cell.append(vq.Atom(1, [i * R_pair + 0.0, 0.0, 0.0]))
        cell.append(vq.Atom(1, [i * R_pair + R_S, 0.0, 0.0]))
    sysp = vq.PeriodicSystem(
        dim=1, lattice=np.diag([n_pairs * R_pair, PAD, PAD]),
        unit_cell=cell,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    if kmesh == [1, 1, 1]:
        kp = vq.KPoints.gamma(sysp)
    else:
        kp = vq.KPoints.monkhorst_pack(sysp, kmesh)
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.cutoff_bohr         = 15.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.conv_tol_energy = 1e-9
    opts.max_iter        = 80
    t0 = time.perf_counter()
    r = vq.run_rhf_periodic_scf(sysp, basis, kp, opts)
    return dict(
        n_pairs=n_pairs, kmesh=tuple(kmesh),
        E_total=float(r.energy),
        E_per_pair=float(r.energy) / n_pairs,
        converged=bool(r.converged),
        n_iter=int(r.n_iter),
        wall_s=time.perf_counter() - t0,
    )


def main() -> None:
    print(f"vibeqc {vq.__version__}\n")
    out_lines: list[str] = [f"vibeqc {vq.__version__}", ""]
    sep = "-" * 88

    print(f"{'R_pair':>9s} | {'representation':<22s} | "
          f"{'E_total (Ha)':>14s} | {'E/pair (Ha)':>14s} | conv | iter")
    print(sep)
    out_lines.append(f"{'R_pair':>9s} | {'representation':<22s} | "
                     f"{'E_total (Ha)':>14s} | {'E/pair (Ha)':>14s} | conv | iter")
    out_lines.append(sep)

    summary: list[tuple[float, float]] = []
    for R_L in R_LS:
        R_pair = R_S + R_L
        per_pair = []
        for n_pairs, kmesh in REPRESENTATIONS:
            res = run(n_pairs, R_pair, kmesh)
            label = f"H{2 * n_pairs:2d} {kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
            line = (f"{R_pair:>9.3f} | {label:<22s} | "
                    f"{res['E_total']:>14.8f} | {res['E_per_pair']:>14.8f} | "
                    f"  {'✓' if res['converged'] else '✗'}  | {res['n_iter']:>3d}")
            print(line)
            out_lines.append(line)
            if res["converged"]:
                per_pair.append(res["E_per_pair"])
        if len(per_pair) >= 2:
            drift = max(per_pair) - min(per_pair)
            verdict = ("✓ µHa" if abs(drift) < 1e-5
                       else "◐ mHa" if abs(drift) < 1e-3
                       else "✗")
            line = (f"  → drift across reps: {drift:+.3e} Ha/pair "
                    f"({drift*1e6:+.2f} µHa/pair)   {verdict}")
            print(line)
            out_lines.append(line)
            summary.append((R_pair, drift))
        print(sep)
        out_lines.append(sep)

    print()
    print("=== Summary: BvK drift vs per-pair spacing ===")
    out_lines.append("")
    out_lines.append("=== Summary: BvK drift vs per-pair spacing ===")
    head = (f"{'R_pair (bohr)':>14s} | {'drift (Ha/pair)':>16s} | "
            f"{'drift (µHa/pair)':>18s}")
    print(head)
    out_lines.append(head)
    for R_pair, drift in summary:
        v = "✓ µHa" if abs(drift) < 1e-5 else "◐ mHa" if abs(drift) < 1e-3 else "✗"
        line = (f"{R_pair:>14.3f} | {drift:>+16.3e} | "
                f"{drift*1e6:>+18.2f}    {v}")
        print(line)
        out_lines.append(line)

    out_path = OUT.with_suffix(".out")
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
