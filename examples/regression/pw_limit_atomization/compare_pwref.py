"""Three-way comparison: GPAW PW-limit vs pob vs VASP atomization energies.

Reads the GPAW JSON sidecar (produced by ``run_pw_reference.py``) and the
benchmark data file (``references-cohesive/r2scan.dat``), then prints a
three-column comparison table with per-system outlier identification.

Usage
-----
    python -m examples.regression.pw_limit_atomization.compare_pwref \
        pwref_r2scan.json ../references-cohesive/r2scan.dat
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Row = Dict  # {compound, E_exp, E_pob, E_VASP, E_gpaw, ...}


def _parse_r2scan_dat(path: Path) -> List[Tuple[str, float, float, float]]:
    """Parse r2scan.dat → list of (compound, E_exp, E_pob, E_VASP) in kJ/mol."""
    rows: List[Tuple[str, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(r"\ce{", "").replace("}", "").split()
        if len(parts) < 4:
            continue
        name = parts[0]  # e.g. "C-d" for diamond, "TiO2-a" for anatase
        try:
            e_exp, e_pob, e_vasp = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        rows.append((name, e_exp, e_pob, e_vasp))
    return rows


def _parse_gpaw_json(path: Path) -> Dict[str, float]:
    """Read GPAW JSON sidecar → {system_id: pw_limit_kjmol}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, float] = {}
    for r in data.get("results", []):
        mapping[r["system_id"]] = r["pw_limit_kjmol"]
    return mapping


# Map system_id to benchmark compound name.
# The r2scan.dat uses LaTeX names (C-d, TiO2-a, etc.); our system_id is
# more descriptive.
SYSTEM_TO_COMPOUND: Dict[str, str] = {
    "lif_rocksalt": "LiF",
    "nacl_rocksalt": "NaCl",
    "naf_rocksalt": "NaF",
    "licl_rocksalt": "LiCl",
    "lih_rocksalt": "LiH",
    "mgo_rocksalt": "MgO",
    "mgs_rocksalt": "MgS",
    "c_diamond": "C-d",
    "bn_zincblende": "BN",
    "agcl_rocksalt": "AgCl",
    "alas_zincblende": "AlAs",
    "aln_zincblende": "AlN",
    "alp_zincblende": "AlP",
    "bao_rocksalt": "BaO",
    "bas_rocksalt": "BaS",
    "cao_rocksalt": "CaO",
    "caf2_fluorite": "CaF2",
    "cdse_zincblende": "CdSe",
    "cscl_cscl": "CsCl",
    "gaas_zincblende": "GaAs",
    "gan_beta_zincblende": "GaN",
    "gap_zincblende": "GaP",
    "inas_zincblende": "InAs",
    "kbr_rocksalt": "KBr",
    "kcl_rocksalt": "KCl",
    "li2o_antifluorite": "Li2O",
    "sic_zincblende": "SiC",
    "zns_zincblende": "ZnS",
    "beo_wurtzite": "BeO",
    "zno_wurtzite": "ZnO",
    "al2o3_corundum": "Al2O3",
    "tio2_rutile": "TiO2-r",
}


def _identify_outlier(
    gpaw: float, pob: float, vasp: float, threshold: float = 10.0
) -> str:
    """Return which code(s) are the outlier vs the other two.

    If all three agree within threshold, return 'all agree'.
    If one disagrees with the other two beyond threshold, return that code's name.
    """
    d_gp = abs(gpaw - pob)
    d_gv = abs(gpaw - vasp)
    d_pv = abs(pob - vasp)
    if d_gp < threshold and d_gv < threshold and d_pv < threshold:
        return "all agree"
    # The pair with smallest difference is the "in-group"; the third is the outlier
    pairs = [
        (d_gp, "gpaw", "pob", "VASP"),
        (d_gv, "gpaw", "VASP", "pob"),
        (d_pv, "pob", "VASP", "GPAW"),
    ]
    pairs.sort(key=lambda x: x[0])
    closest = pairs[0][0]
    if closest < threshold:
        return pairs[0][3]  # the odd one out
    # All three disagree — report all
    return "no clear consensus"


def render_markdown(rows: List[Row]) -> str:
    """Render a Markdown comparison table with outlier column."""
    L: List[str] = []
    L.append("# GPAW PW-limit vs pob vs VASP — r2SCAN atomization energies")
    L.append("")
    L.append("Energies in kJ/mol per formula unit.  Outlier identified by")
    L.append("pairwise agreement: if two codes agree within 10 kJ/mol and the")
    L.append("third differs by more, that third is flagged.")
    L.append("")
    L.append(
        "| Compound | E_exp | GPAW | pob | VASP | Δ(GPAW−pob) | Δ(GPAW−VASP) | Outlier? |"
    )
    L.append(
        "|----------|------:|-----:|----:|-----:|------------:|-------------:|----------|"
    )

    gpaw_deltas_pob: List[float] = []
    gpaw_deltas_vasp: List[float] = []
    for r in rows:
        gpaw = r.get("E_gpaw")
        if gpaw is None:
            continue
        pob = r["E_pob"]
        vasp = r["E_VASP"]
        d_gp = gpaw - pob
        d_gv = gpaw - vasp
        outlier = _identify_outlier(gpaw, pob, vasp)
        L.append(
            f"| {r['compound']} | {r['E_exp']:.1f} | {gpaw:.1f} | "
            f"{pob:.1f} | {vasp:.1f} | {d_gp:+.1f} | {d_gv:+.1f} | {outlier} |"
        )
        gpaw_deltas_pob.append(d_gp)
        gpaw_deltas_vasp.append(d_gv)

    if gpaw_deltas_pob:
        n = len(gpaw_deltas_pob)
        md_pob = sum(gpaw_deltas_pob) / n
        mad_pob = sum(abs(d) for d in gpaw_deltas_pob) / n
        md_vasp = sum(gpaw_deltas_vasp) / n
        mad_vasp = sum(abs(d) for d in gpaw_deltas_vasp) / n
        L.append("")
        L.append(f"**vs GPAW (n={n}):**")
        L.append(f"  GPAW−pob: MD {md_pob:+.1f}, MAD {mad_pob:.1f} kJ/mol")
        L.append(f"  GPAW−VASP: MD {md_vasp:+.1f}, MAD {mad_vasp:.1f} kJ/mol")

        # Count outliers
        pob_outliers = sum(
            1
            for r in rows
            if r.get("E_gpaw") is not None
            and _identify_outlier(r["E_gpaw"], r["E_pob"], r["E_VASP"]) == "pob"
        )
        vasp_outliers = sum(
            1
            for r in rows
            if r.get("E_gpaw") is not None
            and _identify_outlier(r["E_gpaw"], r["E_pob"], r["E_VASP"]) == "VASP"
        )
        gpaw_outliers = sum(
            1
            for r in rows
            if r.get("E_gpaw") is not None
            and _identify_outlier(r["E_gpaw"], r["E_pob"], r["E_VASP"]) == "GPAW"
        )
        agree = sum(
            1
            for r in rows
            if r.get("E_gpaw") is not None
            and _identify_outlier(r["E_gpaw"], r["E_pob"], r["E_VASP"]) == "all agree"
        )
        L.append(
            f"  Outliers: pob={pob_outliers}, VASP={vasp_outliers}, "
            f"GPAW={gpaw_outliers}, agree={agree}"
        )
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "gpaw_json", type=Path, help="GPAW JSON sidecar from run_pw_reference"
    )
    ap.add_argument(
        "r2scan_dat", type=Path, help="Benchmark data (references-cohesive/r2scan.dat)"
    )
    ap.add_argument(
        "--out", default=None, help="Markdown output path (default: stdout)"
    )
    args = ap.parse_args(argv)

    if not args.gpaw_json.exists():
        print(f"error: {args.gpaw_json} not found", file=sys.stderr)
        return 1
    if not args.r2scan_dat.exists():
        print(f"error: {args.r2scan_dat} not found", file=sys.stderr)
        return 1

    benchmark = _parse_r2scan_dat(args.r2scan_dat)
    gpaw_data = _parse_gpaw_json(args.gpaw_json)

    # Reverse mapping: compound name → system_id
    compound_to_sid: Dict[str, str] = {v: k for k, v in SYSTEM_TO_COMPOUND.items()}

    rows: List[Row] = []
    for compound, e_exp, e_pob, e_vasp in benchmark:
        sid = compound_to_sid.get(compound, compound)
        e_gpaw = gpaw_data.get(sid)
        rows.append(
            {
                "compound": compound,
                "E_exp": e_exp,
                "E_pob": e_pob,
                "E_VASP": e_vasp,
                "E_gpaw": e_gpaw,
            }
        )

    n_gpaw = sum(1 for r in rows if r["E_gpaw"] is not None)
    print(
        f"Read {len(rows)} benchmark compounds; {n_gpaw} have GPAW data.",
        file=sys.stderr,
    )

    md = render_markdown(rows)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
