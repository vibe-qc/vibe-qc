"""CLI driver: generate GPAW plane-wave-limit atomization references and report
them against the paper's VASP@900eV values.

Out-of-process GPAW only (§10). Writes a Markdown table and a JSON sidecar
(full data + provenance) so the reference set is reproducible and auditable.

Examples
--------
    python -m examples.regression.pw_limit_atomization.run_pw_reference \
        --systems lif_rocksalt --functional r2scan --cutoffs 800,1000,1200

    python -m examples.regression.pw_limit_atomization.run_pw_reference \
        --systems validation --out pwref_r2scan.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from .pw_reference import (
    PwAtomizationResult,
    check_element_convergence,
    compute_pw_atomization,
)
from .systems import (
    BY_ID,
    HUND_MAGMOM,
    MAIN_GROUP_SET,
    NONCUBIC_SET,
    NONMAGNETIC_SET,
    TM_SYSTEMS,
    VALIDATION_SET,
)


def _select(names: str):
    if names == "validation":
        return list(VALIDATION_SET)
    if names == "main-group":
        return list(MAIN_GROUP_SET)
    if names == "noncubic":
        return list(NONCUBIC_SET)
    if names == "tm":
        return list(TM_SYSTEMS)
    if names == "nonmagnetic":
        return list(NONMAGNETIC_SET)
    if names == "all":
        return list(NONMAGNETIC_SET + NONCUBIC_SET)
    out = []
    for n in names.split(","):
        n = n.strip()
        if n not in BY_ID:
            raise SystemExit(f"unknown system {n!r}; have: {', '.join(BY_ID)}")
        out.append(BY_ID[n])
    return out


def _render_markdown(results: List[PwAtomizationResult], args) -> str:
    L: List[str] = []
    L.append(f"# GPAW plane-wave-limit atomization energies — {args.functional}")
    L.append("")
    code = results[0].code_version if results else "unknown"
    L.append(
        f"GPAW {code} (out-of-process, §10) · k-mesh {args.kmesh} · "
        f"atom box {args.atom_box} Å · cutoffs {args.cutoffs} eV · "
        f"energies kJ/mol per formula unit."
    )
    L.append("")
    L.append("## Summary — PW limit vs VASP@900eV")
    L.append("")
    L.append("| System | GPAW PW-limit | VASP@900 | Δ (GPAW−VASP) |")
    L.append("|--------|--------------:|---------:|--------------:|")
    deltas = []
    for r in results:
        v = "—" if r.vasp900_kjmol is None else f"{r.vasp900_kjmol:.1f}"
        d = r.delta_vs_vasp_kjmol
        ds = "—" if d is None else f"{d:+.1f}"
        if d is not None:
            deltas.append(d)
        L.append(f"| {r.system_id} | {r.pw_limit_kjmol:.1f} | {v} | {ds} |")
    if deltas:
        mad = sum(abs(d) for d in deltas) / len(deltas)
        md = sum(deltas) / len(deltas)
        L.append("")
        L.append(
            f"**vs VASP@900eV (n={len(deltas)}):** MD {md:+.1f}, MAD {mad:.1f} kJ/mol."
        )
    L.append("")
    L.append("## Cutoff convergence (per system)")
    for r in results:
        L.append("")
        L.append(f"### {r.system_id} — {r.functional}, a={r.geometry_a_ang} Å")
        L.append(
            f"_{r.extrapolation}; atom valence "
            + ", ".join(f"{s}:{r.atom_valence.get(s, '?')}" for s in r.atom_energies_ha)
            + "_"
        )
        L.append("")
        L.append("| cutoff (eV) | E_solid (Ha) | E_atoms (Ha) | E_at (kJ/mol) | conv |")
        L.append("|------------:|-------------:|-------------:|--------------:|:----:|")
        for p in r.points:
            ok = "✓" if (p.solid_converged and p.atoms_converged) else "✗"
            L.append(
                f"| {p.cutoff_ev:.0f} | {p.e_solid_ha:.6f} | "
                f"{p.e_atoms_ha:.6f} | {p.atomization_kjmol:.1f} | {ok} |"
            )
        L.append(f"| **PW limit** | | | **{r.pw_limit_kjmol:.1f}** | |")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--systems",
        default="validation",
        help="comma-separated system ids, or 'validation'/'all'",
    )
    ap.add_argument("--functional", default="r2scan")
    ap.add_argument(
        "--cutoffs", default="800,1000,1200", help="comma-separated PW cutoffs in eV"
    )
    ap.add_argument("--kmesh", default="6,6,6")
    ap.add_argument("--atom-box", type=float, default=12.0)
    ap.add_argument("--conv-energy", type=float, default=1e-6)
    ap.add_argument(
        "--converge-cutoff",
        action="store_true",
        help="run per-element cutoff-convergence check before atomization (slower, "
        "but warns if any element needs higher cutoff)",
    )
    ap.add_argument("--out", default=None, help="Markdown output path")
    ap.add_argument(
        "--gpaw-python",
        default=None,
        help="interpreter for the external GPAW process (sets VIBEQC_GPAW_PYTHON)",
    )
    args = ap.parse_args(argv)

    if args.gpaw_python:
        import os

        os.environ["VIBEQC_GPAW_PYTHON"] = args.gpaw_python
    cutoffs = [float(c) for c in args.cutoffs.split(",")]
    kmesh = tuple(int(x) for x in args.kmesh.split(","))
    systems = _select(args.systems)

    # --- per-element cutoff-convergence check (optional) ---
    if args.converge_cutoff:
        all_symbols: List[str] = sorted(
            set(sym for s in systems for sym in s.atom_symbols())
        )
        print(
            f"[pwref] cutoff-convergence check: {len(all_symbols)} elements "
            f"at cutoffs {cutoffs} ...",
            flush=True,
        )
        try:
            elem_results = check_element_convergence(
                all_symbols,
                HUND_MAGMOM,
                functional=args.functional,
                cutoffs_ev=cutoffs,
                atom_box_ang=args.atom_box,
                conv_energy_ha=args.conv_energy,
            )
        except RuntimeError as exc:
            print(f"[pwref] converge-check FAILED: {exc}", file=sys.stderr)
            return 1
        bad = [e for e in elem_results if not e.converged]
        for ec in elem_results:
            tag = "✓" if ec.converged else "✗"
            print(f"  {tag} {ec.recommendation}")
        if bad:
            print(
                f"[pwref] WARNING: {len(bad)} element(s) not cutoff-converged. "
                f"Consider --cutoffs with higher values.",
                flush=True,
            )
        else:
            print(f"[pwref] all {len(elem_results)} elements cutoff-converged.")

    # --- atomization calculation ---
    results: List[PwAtomizationResult] = []
    for s in systems:
        t0 = time.perf_counter()
        print(f"[pwref] {s.id} ({args.functional}) ...", flush=True)
        try:
            r = compute_pw_atomization(
                s,
                functional=args.functional,
                cutoffs_ev=cutoffs,
                kmesh=kmesh,
                atom_box_ang=args.atom_box,
                conv_energy_ha=args.conv_energy,
            )
        except RuntimeError as exc:
            print(f"[pwref] {s.id}: FAILED — {exc}", flush=True)
            continue
        d = r.delta_vs_vasp_kjmol
        ds = "" if d is None else f"  (Δ vs VASP {d:+.1f})"
        print(
            f"[pwref] {s.id}: PW-limit {r.pw_limit_kjmol:.1f} kJ/mol{ds}  "
            f"[{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )
        results.append(r)

    if not results:
        print("[pwref] no results (GPAW unavailable?)", file=sys.stderr)
        return 1

    md = _render_markdown(results, args)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        payload = {
            "provenance": {
                "external_program": "gpaw",
                "code_version": results[0].code_version,
                "boundary": "out-of-process subprocess (CLAUDE.md §10)",
                "functional": args.functional,
                "cutoffs_ev": cutoffs,
                "kmesh": list(kmesh),
                "atom_box_ang": args.atom_box,
                "reference": "VASP@900eV r2SCAN (references-cohesive/r2scan.dat)",
            },
            "results": [
                {
                    "system_id": r.system_id,
                    "pw_limit_kjmol": r.pw_limit_kjmol,
                    "vasp900_kjmol": r.vasp900_kjmol,
                    "delta_vs_vasp_kjmol": r.delta_vs_vasp_kjmol,
                    "extrapolation": r.extrapolation,
                    "points": [vars(p) for p in r.points],
                    "atom_energies_ha": r.atom_energies_ha,
                    "atom_valence": r.atom_valence,
                }
                for r in results
            ],
        }
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"[pwref] wrote {args.out} (+ .json)", flush=True)
    else:
        print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
