#!/usr/bin/env python
"""Aggregate AICCM-2026 ``run_case.py`` JSON results into the paper tables.

``run_case.py`` writes one ``<system>__<route>.json`` per case (energy,
energy/atom, converged, crystal_ref, …). This script collects a directory of them
into comparison tables and computes the headline deltas (in mHa/atom):

* **Single line** — per-system × per-route energies plus key route gaps. The
  Γ-CCM four-center minus neutral-GDF-control gap is attribution-qualified;
  it is not assigned to a single Coulomb or seam term:

      python compare.py results/

* **Historical head-to-head** — D89 disables ``--vs`` and direct
  ``head_to_head`` calls because route-name equality does not establish a
  Gamma-CCM/chi-CCM approach comparison. For the narrower real-Gamma
  representation control, use ``compare_b.py --real-gamma-control-results``.

* **CRYSTAL23** — add a CRYSTAL23 column from a ``{system: E_per_atom_Ha}`` JSON
  (the library ``.d12`` outputs are mostly absent; populate after running them):

      python compare.py results/ --crystal-refs crystal23.json

Markdown to stdout; ``--csv FILE`` also writes CSV. Pure stdlib.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROUTES = ["aiccm-hf", "aiccm-ks", "aiccm-ri", "aiccm-ks-ri", "aiccm-rijcosx",
          "aiccm-mp2", "aiccm-ccsd", "bipole", "gdf"]
HA2MHA = 1000.0
D89_HEAD_TO_HEAD_ERROR = (
    "D89 disables compare.py head-to-head deltas: route-name equality does "
    "not establish a Gamma-CCM/chi-CCM approach comparison. Use compare_b.py "
    "--real-gamma-control-results for the narrower real-Gamma representation "
    "control."
)


def load(results_dir):
    """{system: {route: record}} from <system>__<route>.json files."""
    data = {}
    for f in sorted(Path(results_dir).glob("*__*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        data.setdefault(rec["system"], {})[rec["route"]] = rec
    return data


def _epa(rec):
    return rec.get("energy_per_atom") if rec else None


def _delta(a, b):
    if a is None or b is None:
        return None
    return (a - b) * HA2MHA


def single_table(data, crystal_refs):
    """Per-system table: per-route E/atom + key gaps (mHa/atom)."""
    rows = []
    for sysname in sorted(data):
        r = data[sysname]
        e = {rt: _epa(r.get(rt)) for rt in ROUTES}
        ref = crystal_refs.get(sysname) if crystal_refs else None
        # 4c = aiccm-ks if present else aiccm-hf
        e4 = e["aiccm-ks"] if e["aiccm-ks"] is not None else e["aiccm-hf"]
        row = dict(
            system=sysname,
            klass=(next(iter(r.values())).get("klass") if r else ""),
            aiccm_4c=e4, aiccm_ri=e["aiccm-ri"], aiccm_ks_ri=e["aiccm-ks-ri"],
            aiccm_rijcosx=e["aiccm-rijcosx"],
            aiccm_mp2=e["aiccm-mp2"], aiccm_ccsd=e["aiccm-ccsd"],
            bipole=e["bipole"], gdf=e["gdf"], crystal23=ref,
            d_gamma_ccm_minus_gdf_control=_delta(e4, e["gdf"]),
            d_ri_gdf=_delta(e["aiccm-ri"], e["gdf"]),  # RI consistency
            d_gdf_bipole=_delta(e["gdf"], e["bipole"]),
            d_gdf_crystal=_delta(e["gdf"], ref),
        )
        rows.append(row)
    return rows


def fmt(x, nd=6):
    return "" if x is None else (f"{x:.{nd}f}" if abs(x) >= 1 else f"{x:.{nd}f}")


def fmt_d(x):
    return "" if x is None else f"{x:+.3f}"


def print_single(rows):
    hdr = ["system", "class", "aiccm-4c", "aiccm-ri", "ks-ri", "rijcosx",
           "mp2", "ccsd(t)", "bipole", "gdf", "CRYSTAL23",
           "ΔΓCCM-GDFctrl", "Δri-gdf", "Δgdf-bip", "Δgdf-X23"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        cells = [r["system"], r["klass"] or "",
                 fmt(r["aiccm_4c"]), fmt(r["aiccm_ri"]), fmt(r["aiccm_ks_ri"]),
                 fmt(r["aiccm_rijcosx"]), fmt(r["aiccm_mp2"]), fmt(r["aiccm_ccsd"]),
                 fmt(r["bipole"]), fmt(r["gdf"]), fmt(r["crystal23"]),
                 fmt_d(r["d_gamma_ccm_minus_gdf_control"]),
                 fmt_d(r["d_ri_gdf"]),
                 fmt_d(r["d_gdf_bipole"]), fmt_d(r["d_gdf_crystal"])]
        print("| " + " | ".join(cells) + " |")
    print(
        "\nEnergies in Ha/atom (mp2/ccsd(t) are totals); Δ in mHa/atom. "
        "ΔΓCCM-GDFctrl is the union-and-weight Γ-CCM construction minus the "
        "neutral fitted-torus GDF control. It is a construction/control route "
        "gap with causality unassigned, not a single-term Madelung diagnosis."
    )


def head_to_head(data_a, data_b):
    """Reject the historical unproven approach-level numeric comparison."""

    raise RuntimeError(D89_HEAD_TO_HEAD_ERROR)


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="results dir of <system>__<route>.json")
    ap.add_argument(
        "--vs",
        default=None,
        help="disabled by D89; retained only to return a fail-closed error",
    )
    ap.add_argument("--crystal-refs", default=None,
                    help="JSON {system: E_per_atom_Ha} of CRYSTAL23 references")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if args.vs:
        ap.error(D89_HEAD_TO_HEAD_ERROR)
    data = load(args.results)
    crystal_refs = json.loads(Path(args.crystal_refs).read_text()) if args.crystal_refs else {}
    print(f"# AICCM-2026 comparison — {len(data)} systems from {args.results}\n")
    rows = single_table(data, crystal_refs)
    print_single(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
