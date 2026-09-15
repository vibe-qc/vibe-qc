"""Fetch standard auxiliary basis sets from Basis Set Exchange.

Pulls the def2 / Weigend JKfit + JFIT + RIFIT family used by vibe-qc
for density fitting, in libint-compatible Gaussian94 (.g94) format.
Output goes to ``python/vibeqc/basis_library/custom/`` so it ships
inside the wheel; ``scripts/setup_basis_library.sh`` then merges
custom/ → basis/.

Run once when adding a new basis to the project; the resulting .g94
files are committed to the repo so end users get them via a plain
``pip install -e .`` from the vibe-qc checkout. Internet access is
required only at the maintainer's run; users never re-fetch.

Provenance: every file fetched is sourced from
https://www.basissetexchange.org with the BSE version recorded in the
written file's header (BSE pulls embed source attribution). The
shipped .g94 files are checked into ``custom/`` exactly as BSE returns
them — no manual editing.

Usage:
    python scripts/fetch_bse_aux_bases.py [--dry-run] [--list]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import basis_set_exchange as bse


# (BSE basis name, vibe-qc filename without .g94 suffix).
#
# BSE uses mixed-case names with a "FIT" suffix; vibe-qc filenames are
# lowercase and follow the BSE convention for the new aux files we ship.
# (Existing libint-bundled per-zeta JKfit files like def2-svp-jk follow
# the older ORCA "no -fit suffix" convention. We don't rename existing
# files in this script — the new family lives alongside them.)
TARGETS: list[tuple[str, str]] = [
    # ---- def2 family ------------------------------------------------------
    # Coulomb + exchange fits (Weigend, JCC 29, 167 (2008)).
    ("def2-universal-JKFIT",   "def2-universal-jkfit"),
    ("def2-universal-JFIT",    "def2-universal-jfit"),
    ("def2-SV(P)-JKFIT",       "def2-sv(p)-jkfit"),

    # MP2 / correlation RI fits, per-zeta (Weigend-Häser-Patzelt-Ahlrichs,
    # CPL 294, 143 (1998); Hellweg-Hättig-Höfener-Klopper, TCA 117, 587
    # (2007) for the diffuse variants).
    ("def2-SV(P)-RIFIT",       "def2-sv(p)-rifit"),
    ("def2-SVP-RIFIT",         "def2-svp-rifit"),
    ("def2-SVPD-RIFIT",        "def2-svpd-rifit"),
    ("def2-TZVP-RIFIT",        "def2-tzvp-rifit"),
    ("def2-TZVPD-RIFIT",       "def2-tzvpd-rifit"),
    ("def2-TZVPP-RIFIT",       "def2-tzvpp-rifit"),
    ("def2-TZVPPD-RIFIT",      "def2-tzvppd-rifit"),
    ("def2-QZVP-RIFIT",        "def2-qzvp-rifit"),
    ("def2-QZVPP-RIFIT",       "def2-qzvpp-rifit"),
    ("def2-QZVPPD-RIFIT",      "def2-qzvppd-rifit"),

    # ---- Dunning cc-pVNZ correlation-consistent family --------------------
    # MP2 RI fits (Weigend-Häser-Patzelt-Ahlrichs 1998). vibe-qc already
    # bundles libint's "cc-pvNz-ri" variant under that lowercase name; the
    # "cc-pVNZ-RIFIT" files below are the BSE-canonical Weigend names
    # (same content as the libint files for these basis sets, packaged
    # for users who type the BSE / PySCF name).
    ("cc-pVDZ-RIFIT",          "cc-pvdz-rifit"),
    ("cc-pVTZ-RIFIT",          "cc-pvtz-rifit"),
    ("cc-pVQZ-RIFIT",          "cc-pvqz-rifit"),
    ("cc-pV5Z-RIFIT",          "cc-pv5z-rifit"),
    ("cc-pV6Z-RIFIT",          "cc-pv6z-rifit"),

    # Diffuse cc-pVNZ + RI fits (aug-cc-pVNZ-RIFIT) — for MP2 with
    # aug-cc-pVNZ orbital basis.
    ("aug-cc-pVDZ-RIFIT",      "aug-cc-pvdz-rifit"),
    ("aug-cc-pVTZ-RIFIT",      "aug-cc-pvtz-rifit"),
    ("aug-cc-pVQZ-RIFIT",      "aug-cc-pvqz-rifit"),
    ("aug-cc-pV5Z-RIFIT",      "aug-cc-pv5z-rifit"),
    ("aug-cc-pV6Z-RIFIT",      "aug-cc-pv6z-rifit"),

    # ECP-aware cc-pVNZ-PP + their RI fits (Weigend / Peterson).
    ("cc-pVDZ-PP-RIFIT",       "cc-pvdz-pp-rifit"),
    ("cc-pVTZ-PP-RIFIT",       "cc-pvtz-pp-rifit"),
    ("cc-pVQZ-PP-RIFIT",       "cc-pvqz-pp-rifit"),
    ("cc-pV5Z-PP-RIFIT",       "cc-pv5z-pp-rifit"),
    ("aug-cc-pVDZ-PP-RIFIT",   "aug-cc-pvdz-pp-rifit"),
    ("aug-cc-pVTZ-PP-RIFIT",   "aug-cc-pvtz-pp-rifit"),
    ("aug-cc-pVQZ-PP-RIFIT",   "aug-cc-pvqz-pp-rifit"),
    ("aug-cc-pV5Z-PP-RIFIT",   "aug-cc-pv5z-pp-rifit"),

    # Core-correlation cc-pwCVNZ + RI fits (Peterson / Dunning).
    ("cc-pwCVDZ-RIFIT",        "cc-pwcvdz-rifit"),
    ("cc-pwCVTZ-RIFIT",        "cc-pwcvtz-rifit"),
    ("cc-pwCVQZ-RIFIT",        "cc-pwcvqz-rifit"),
    ("cc-pwCV5Z-RIFIT",        "cc-pwcv5z-rifit"),
    ("aug-cc-pwCVDZ-RIFIT",    "aug-cc-pwcvdz-rifit"),
    ("aug-cc-pwCVTZ-RIFIT",    "aug-cc-pwcvtz-rifit"),
    ("aug-cc-pwCVQZ-RIFIT",    "aug-cc-pwcvqz-rifit"),
    ("aug-cc-pwCV5Z-RIFIT",    "aug-cc-pwcv5z-rifit"),

    # Core-correlation + ECP combinations.
    ("cc-pwCVDZ-PP-RIFIT",     "cc-pwcvdz-pp-rifit"),
    ("cc-pwCVTZ-PP-RIFIT",     "cc-pwcvtz-pp-rifit"),
    ("cc-pwCVQZ-PP-RIFIT",     "cc-pwcvqz-pp-rifit"),
    ("cc-pwCV5Z-PP-RIFIT",     "cc-pwcv5z-pp-rifit"),
    ("aug-cc-pwCVDZ-PP-RIFIT", "aug-cc-pwcvdz-pp-rifit"),
    ("aug-cc-pwCVTZ-PP-RIFIT", "aug-cc-pwcvtz-pp-rifit"),
    ("aug-cc-pwCVQZ-PP-RIFIT", "aug-cc-pwcvqz-pp-rifit"),
    ("aug-cc-pwCV5Z-PP-RIFIT", "aug-cc-pwcv5z-pp-rifit"),

    # ---- Pople-style RIFIT (legacy) --------------------------------------
    ("6-31G**-RIFIT",          "6-31g**-rifit"),
    ("6-311G**-RIFIT",         "6-311g**-rifit"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be fetched without writing files.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the targets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print(f"BSE library version: {bse.version()}")
        print(f"Targets ({len(TARGETS)}):")
        for src, dst in TARGETS:
            print(f"  {src:<28s} -> {dst}.g94")
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "python" / "vibeqc" / "basis_library" / "custom"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"BSE library version: {bse.version()}")
    print(f"Output directory:    {out_dir}")

    n_written = 0
    n_skipped = 0
    for src, dst in TARGETS:
        out_path = out_dir / f"{dst}.g94"
        if args.dry_run:
            print(f"  [dry-run] {src} -> {out_path.name}")
            continue
        try:
            data = bse.get_basis(src, fmt="gaussian94", header=True)
        except KeyError as exc:
            print(f"  ! {src}: not in BSE ({exc})", file=sys.stderr)
            continue
        if out_path.exists():
            existing = out_path.read_text()
            if existing == data:
                print(f"  = {src} -> {out_path.name} (unchanged)")
                n_skipped += 1
                continue
        out_path.write_text(data)
        n_written += 1
        n_lines = data.count("\n")
        print(f"  + {src} -> {out_path.name} ({n_lines} lines)")

    print(f"\nWrote {n_written} new/updated, kept {n_skipped} unchanged.")
    print(
        "\nTo make these visible to libint at runtime:\n"
        "  cp python/vibeqc/basis_library/custom/*.g94 \\\n"
        "     python/vibeqc/basis_library/basis/\n"
        "(or run scripts/setup_basis_library.sh if libint is "
        "vendored under third_party/.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
