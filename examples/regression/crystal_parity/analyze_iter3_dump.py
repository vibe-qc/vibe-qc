"""Decode the iter-3 collapse dump from
`diag_lih_multik_iter3_instrumented.py` and surface the three
candidate bug signals listed in the brief / handover:

  (1) D(k) eigenvalue rises above 2.0  → density loses idempotency,
      F gets corrupted.
  (2) F(k) HOMO-LUMO gap closes        → mid-SCF metallisation,
      Aufbau picks wrong orbitals.
  (3) Occupied-subspace overlap drops  → orbital flipping iter-to-
      iter, occupied/virtual swap.

Run after the compute-reference job completes:
    scp compute-reference:/home/USER/.local/share/vq/jobs/<JOBID>/lih_multik_iter3_dump.json /tmp/
    python examples/regression/crystal_parity/analyze_iter3_dump.py /tmp/lih_multik_iter3_dump.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List


def fmt_e(x: float, width: int = 11) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "       n/a "
    return f"{x: {width}.4e}"


def fmt_f(x: float, width: int = 9, prec: int = 4) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "      n/a"
    return f"{x: {width}.{prec}f}"


def compare_modes(no_aids_path: str, levshift_path: str) -> int:
    """Side-by-side energy trace from the two diagnostic modes,
    so the iter-3 collapse vs LEVSHIFT-stabilised trajectory is
    visible in one glance."""
    with open(no_aids_path) as f:
        a = json.load(f)
    with open(levshift_path) as f:
        b = json.load(f)
    print("=== iter-3 collapse comparison ===")
    print(f"  no_aids  : {no_aids_path}")
    print(f"  levshift : {levshift_path}")
    print(f"  CRYSTAL14 LiH primitive ref: -7.93817 Ha/FU")
    print()
    n = max(len(a["iters"]), len(b["iters"]))
    print(f"  {'iter':>4}  {'E (no_aids)':>13}  {'b':>5}  "
          f"{'E (levshift)':>13}  {'b':>5}  {'Δ vs ref (lshft)':>17}")
    REF = -7.93817
    for i in range(n):
        ia = a["iters"][i] if i < len(a["iters"]) else None
        ib = b["iters"][i] if i < len(b["iters"]) else None
        ea = f"{ia['E_total']:>13.6f}" if ia else " " * 13
        eb = f"{ib['E_total']:>13.6f}" if ib else " " * 13
        # b column is not directly stored — we infer from shifts schedule.
        ba_str = "  -  "
        bb_str = "  -  "
        sched = b.get("level_shift_schedule")
        if sched is not None and ib is not None:
            bb = sched[min(i, len(sched) - 1)]
            bb_str = f"{bb:>5.2f}"
        sched_a = a.get("level_shift_schedule")
        if sched_a is not None and ia is not None:
            ba = sched_a[min(i, len(sched_a) - 1)]
            ba_str = f"{ba:>5.2f}"
        delta = f"{ib['E_total'] - REF:>+17.4f}" if ib else " " * 17
        print(f"  {i+1:>4}  {ea}  {ba_str}  {eb}  {bb_str}  {delta}")
    print()
    if b["iters"]:
        last = b["iters"][-1]
        print(f"  LEVSHIFT final E_total = {last['E_total']:.6f} Ha")
        print(f"  Δ vs CRYSTAL14 -7.93817 Ha = {last['E_total'] - REF:+.4f} Ha")
        print(f"  dE last iter = {last['dE']:+.3e}, "
              f"|FDS-SDF| = {last['grad_norm_sum_k']:.3e}")
    return 0


def analyse(dump_path: str) -> int:
    with open(dump_path) as f:
        dump: Dict[str, Any] = json.load(f)

    print(f"=== {dump_path} ===")
    print(f"  system: {dump.get('system', '?')}")
    print(f"  basis:  {dump.get('basis', '?')}, n_occ={dump.get('n_occ')}, "
          f"n_bf={dump.get('n_bf')}")
    print(f"  kmesh:  {dump.get('kmesh')}, "
          f"cutoff = {dump.get('cutoff_bohr')} bohr, "
          f"n_cells_in_lattice_sum = {dump.get('n_cells_in_lattice_sum')}")
    print(f"  iters dumped: {len(dump.get('iters', []))}")
    print()

    iters: List[Dict[str, Any]] = dump.get("iters", [])
    if not iters:
        print("(no iters)")
        return 1

    n_k = len(iters[0]["per_k"])

    # ---- Energy trace
    print("--- Energy trace (per-cell, in Ha) ---")
    print(f"  {'iter':>4}  {'E_total':>13}  {'dE':>11}  {'grad_norm':>11}  "
          f"{'E_elec':>13}  {'E_nuc':>11}  {'E_mad':>11}")
    for it in iters:
        print(
            f"  {it['iter']:>4}  {it['E_total']:>13.6f}  "
            f"{fmt_e(it['dE'])}  {fmt_e(it['grad_norm_sum_k'])}  "
            f"{it['E_elec']:>13.6f}  {it['E_nuc']:>11.6f}  "
            f"{it['E_madelung_fix']:>11.6f}"
        )
    print()

    # ---- Signal (1a): AO-basis D(k) max eigenvalue per k vs iter
    print("--- Signal (1a): max AO-basis D(k) eigval per k "
          "(amplifies with small S eigvals; not a pure-purity check) ---")
    hdr = "  iter  " + "  ".join(f"k{kx:02d}_Dmax" for kx in range(n_k))
    print(hdr)
    for it in iters:
        row = f"  {it['iter']:>4}  "
        for kx in range(n_k):
            v = it["per_k"][kx]["D_eigval_max"]
            mark = "*" if v > 2.001 else " "
            row += f"{mark}{v:>8.4f} "
        print(row)
    print()

    # ---- Signal (1b): physically-meaningful S·D eigvals (purity check)
    print("--- Signal (1b): max(eigvals(S^{1/2} D S^{1/2})) per k "
          "(physical purity — should be ≤ 2.0 + ε) ---")
    hdr = "  iter  " + "  ".join(f"k{kx:02d}_SDmax" for kx in range(n_k))
    print(hdr)
    triggered_1 = False
    for it in iters:
        row = f"  {it['iter']:>4}  "
        for kx in range(n_k):
            v = it["per_k"][kx].get("SD_eigval_max")
            if v is None:
                row += "    n/a  "
                continue
            mark = "*" if v > 2.001 else " "
            row += f"{mark}{v:>8.4f} "
            if v > 2.001:
                triggered_1 = True
        print(row)
    print(f"  → signal (1b) {'TRIGGERED' if triggered_1 else 'not triggered'}")
    print()

    # ---- Signal (1c): S(k) condition — small eigvals explode the
    # canonical orthogonaliser and corrupt the inverse-Bloch density.
    print("--- Signal (1c): min S(k) eigval per k "
          "(< 1e-3 = basis ill-conditioned at this k) ---")
    hdr = "  iter  " + "  ".join(f"k{kx:02d}_Smin" for kx in range(n_k))
    print(hdr)
    triggered_1c = False
    if iters[0]["per_k"][0].get("S_eigval_min") is not None:
        # S(k) is constant across SCF iters; show iter 1 only.
        it = iters[0]
        row = f"  {it['iter']:>4}  "
        for kx in range(n_k):
            v = it["per_k"][kx]["S_eigval_min"]
            mark = "*" if v < 1e-3 else " "
            row += f"{mark}{v:>8.2e} "
            if v < 1e-3:
                triggered_1c = True
        print(row)
        print(f"  → signal (1c) {'TRIGGERED' if triggered_1c else 'not triggered'}")
    else:
        print("  (S_eigval_min not in dump — re-run with the updated diag)")
    print()

    # ---- Signal (2): HOMO-LUMO gap per k vs iter
    print("--- Signal (2): F(k) HOMO-LUMO gap per k (should be > 0 always) ---")
    hdr = "  iter  " + "  ".join(f"k{kx:02d}_gap" for kx in range(n_k))
    print(hdr)
    triggered_2 = False
    for it in iters:
        row = f"  {it['iter']:>4}  "
        for kx in range(n_k):
            v = it["per_k"][kx]["gap"]
            mark = "*" if v < 1e-3 else " "
            row += f"{mark}{v:>+8.3e} "
            if v < 1e-3:
                triggered_2 = True
        print(row)
    print(f"  → signal (2) {'TRIGGERED' if triggered_2 else 'not triggered'}")
    print()

    # ---- Signal (3): occupied-subspace overlap with previous iter
    print("--- Signal (3): occ-subspace |det(O O^†)| per k (should ≈ 1.0) ---")
    hdr = "  iter  " + "  ".join(f"k{kx:02d}_det" for kx in range(n_k))
    print(hdr)
    triggered_3 = False
    for it in iters:
        row = f"  {it['iter']:>4}  "
        has = False
        for kx in range(n_k):
            ovl = it["per_k"][kx].get("occ_overlap_vs_prev")
            if ovl is None:
                row += "    n/a  "
                continue
            has = True
            v = ovl["det_OOdag"]
            mark = "*" if v < 0.99 else " "
            row += f"{mark}{v:>7.4f} "
            if v < 0.99:
                triggered_3 = True
        if has:
            print(row)
        else:
            # First iter has no prev — skip silently.
            pass
    print(f"  → signal (3) {'TRIGGERED' if triggered_3 else 'not triggered'}")
    print()

    # ---- Min singular value of O (catches partial subspace swap before
    # the determinant goes to zero)
    print("--- Signal (3b): min σ(O) per k (subspace stability; should ≈ 1.0) ---")
    hdr = "  iter  " + "  ".join(f"k{kx:02d}_minσ" for kx in range(n_k))
    print(hdr)
    triggered_3b = False
    for it in iters:
        row = f"  {it['iter']:>4}  "
        has = False
        for kx in range(n_k):
            ovl = it["per_k"][kx].get("occ_overlap_vs_prev")
            if ovl is None:
                row += "    n/a  "
                continue
            has = True
            v = ovl["min_sigma"]
            mark = "*" if v < 0.95 else " "
            row += f"{mark}{v:>7.4f} "
            if v < 0.95:
                triggered_3b = True
        if has:
            print(row)
    print(f"  → signal (3b) {'TRIGGERED' if triggered_3b else 'not triggered'}")
    print()

    # ---- Per-cell density Frobenius (cross-cell density leak?)
    print("--- Per-cell density ‖D(g)‖_F (||D(g=0)|| ↑ relative to neighbours) ---")
    cell_sample: List[List[int]] = []
    if iters[0].get("cell_density_traces"):
        # Pick a handful of cells to show: g=0, +x_first, +y_first,
        # +z_first, and the cell with the largest frob in iter 1.
        all_cells = [c["cell_index"] for c in iters[0]["cell_density_traces"]]
        cell_sample.append([0, 0, 0])
        for idx in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            if idx in all_cells:
                cell_sample.append(idx)
        # Also pick the 3 largest-norm cells from the last iter as
        # "outliers".
        last = iters[-1]["cell_density_traces"]
        last_sorted = sorted(last, key=lambda c: -c["frobenius"])[:3]
        for c in last_sorted:
            if c["cell_index"] not in cell_sample:
                cell_sample.append(c["cell_index"])
    hdr = "  iter  " + "  ".join(f"{tuple(c)}_||F||" for c in cell_sample)
    print(hdr)
    for it in iters:
        row = f"  {it['iter']:>4}  "
        cells_dict = {tuple(c["cell_index"]): c
                      for c in it["cell_density_traces"]}
        for c in cell_sample:
            v = cells_dict.get(tuple(c), {}).get("frobenius", float("nan"))
            row += f" {v:>10.4f} "
        print(row)
    print()

    # ---- Verdict
    print("=== Verdict ===")
    print(f"  (1) D-eigval > 2.0:                 "
          f"{'TRIGGERED' if triggered_1 else 'no'}")
    print(f"  (2) HOMO-LUMO gap closes:           "
          f"{'TRIGGERED' if triggered_2 else 'no'}")
    print(f"  (3) det(occ-subspace overlap) drop: "
          f"{'TRIGGERED' if triggered_3 else 'no'}")
    print(f"  (3b) min-σ(occ-subspace) drop:      "
          f"{'TRIGGERED' if triggered_3b else 'no'}")
    print()
    print("Likely-fix path:")
    if triggered_1:
        print("  → density purification (McWeeny D ← 3D² − 2D³ on the unit-")
        print("    cell block) or Fermi-Dirac smearing to bound D eigvals.")
    if triggered_2:
        print("  → Fermi-Dirac smearing at the iter-2→3 transition; or")
        print("    iter-decreasing LEVSHIFT (a static large shift here just")
        print("    delays the gap closure rather than fixing it).")
    if triggered_3 or triggered_3b:
        print("  → Maximum-Overlap Method (MOM): track occupied orbitals by")
        print("    overlap with the previous iter's, not by energy order.")
    if not (triggered_1 or triggered_2 or triggered_3 or triggered_3b):
        print("  → none of the three documented hypotheses fired; the")
        print("    bug may be deeper (D_real inverse-Bloch fold? lattice-")
        print("    sum gauge?). Re-inspect the energy / per-cell tables.")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dump", nargs="?", default="/tmp/lih_multik_iter3_dump.json",
        help="Path to the JSON dump (default /tmp/lih_multik_iter3_dump.json)",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("NO_AIDS_PATH", "LEVSHIFT_PATH"),
        help="Side-by-side energy comparison of two dumps "
             "(e.g. no_aids vs levshift mode)",
    )
    args = parser.parse_args(argv)
    if args.compare:
        return compare_modes(args.compare[0], args.compare[1])
    return analyse(args.dump)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
