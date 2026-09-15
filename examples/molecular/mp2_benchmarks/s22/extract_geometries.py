"""One-shot extractor: pull S22 dimer geometries from the Psi4
``S22.py`` database file and rewrite as plain xyz files.

The geometries originate in Jurečka, Šponer, Černý, Hobza,
*PCCP* **8**, 1985 (2006) — they are facts of nature (optimized
quantum-chemistry coordinates) reported in a peer-reviewed paper, not
copyrightable expression. We re-emit them in a more portable format
(plain xyz, per-monomer) plus a small Python module recording the
S22B reference interaction energies (Marshall, Burns, Sherrill *JCP*
**135**, 194102 (2011)).

The Psi4 ``S22.py`` source is GNU LGPL v3 (Copyright (c) 2007-2025 The
Psi4 Developers). We never bundle that file in vibe-qc — this script
parses it on demand and writes our own xyz files. To re-run:

    curl -sf -o /tmp/s22_psi4.py \\
      https://raw.githubusercontent.com/psi4/psi4/master/psi4/share/psi4/databases/S22.py
    .venv/bin/python examples/molecular/mp2_benchmarks/s22/extract_geometries.py

Outputs (next to this script):
    geometries/s22-NN-<slug>.xyz           — dimer (combined)
    geometries/s22-NN-<slug>-monoA.xyz     — monomer A (counterpoise-style isolated)
    geometries/s22-NN-<slug>-monoB.xyz     — monomer B
    reference_energies.py                  — S22B values, one dict per revision
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("/tmp/s22_psi4.py")
OUT_DIR = HERE / "geometries"

# Slug per S22 index — short, kebab-case, matches Jurečka 2006 Table 1.
SLUGS = {
    1: "ammonia-dimer",
    2: "water-dimer",
    3: "formic-acid-dimer",
    4: "formamide-dimer",
    5: "uracil-dimer-hb",
    6: "2-pyridone-2-aminopyridine",
    7: "adenine-thymine-wc",
    8: "methane-dimer",
    9: "ethene-dimer",
    10: "benzene-ch4",
    11: "benzene-dimer-stack",
    12: "pyrazine-dimer",
    13: "uracil-dimer-stack",
    14: "indole-benzene-stack",
    15: "adenine-thymine-stack",
    16: "ethene-ethyne",
    17: "benzene-h2o",
    18: "benzene-nh3",
    19: "benzene-hcn",
    20: "benzene-dimer-t",
    21: "indole-benzene-t",
    22: "phenol-dimer",
}

# Subset labels per Jurečka 2006:
#   HB = hydrogen-bonded, DD = dispersion-dominated, MX = mixed-influence.
SUBSETS = {
    **{i: "HB" for i in range(1, 8)},
    **{i: "DD" for i in range(8, 16)},
    **{i: "MX" for i in range(16, 23)},
}


def parse_geos(text: str) -> dict[int, str]:
    """Return {index: raw_block} for each GEOS dimer entry."""
    pat = re.compile(
        r"^GEOS\[.*?dbse,\s*'(\d+)'.*?\]\s*=\s*qcdb\.Molecule\(\"\"\"(.*?)\"\"\"\)",
        re.DOTALL | re.MULTILINE,
    )
    out: dict[int, str] = {}
    for m in pat.finditer(text):
        out[int(m.group(1))] = m.group(2)
    return out


def split_monomers(block: str) -> list[list[str]]:
    """Split a qcdb.Molecule string into per-monomer atom blocks."""
    body = block.strip()
    body = re.sub(r"units\s+\w+", "", body).strip()
    parts = [p.strip() for p in body.split("--")]
    monomers = []
    for p in parts:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        # Drop leading "charge mult" line
        if re.match(r"^-?\d+\s+\d+\s*$", lines[0]):
            lines = lines[1:]
        monomers.append(lines)
    return monomers


def write_xyz(path: Path, atoms: list[str], comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(atoms)), comment]
    for ln in atoms:
        sym, x, y, z = ln.split()
        lines.append(f"{sym:<2s}  {float(x):14.8f}  {float(y):14.8f}  {float(z):14.8f}")
    path.write_text("\n".join(lines) + "\n")


# Reference S22B interaction energies (kcal/mol) per Marshall et al. 2011.
# Extracted directly from /tmp/s22_psi4.py (line range starts at "BIND_S22B").
def parse_bind_dict(text: str, varname: str) -> dict[int, float]:
    pat = re.compile(
        rf"^{re.escape(varname)}\['%s-%s'\s*%\s*\(dbse,\s*(\d+)\)\]\s*=\s*(-?\d+\.\d+)",
        re.MULTILINE,
    )
    return {int(m.group(1)): float(m.group(2)) for m in pat.finditer(text)}


def write_reference_module(out_path: Path, refs: dict[str, dict[int, float]]) -> None:
    lines = [
        '"""S22 reference interaction energies (kcal/mol).',
        "",
        "Extracted from the Psi4 ``S22.py`` database file (LGPL v3),",
        "which itself sources the values from:",
        "",
        '  * S220 — Jurečka, Šponer, Černý, Hobza, *PCCP* **8**, 1985 (2006).',
        '  * S22A — Takatani, Hohenstein, Malagoli, Marshall, Sherrill,',
        '    *JCP* **132**, 144104 (2010). First revision (CCSD(T)/CBS, frozen-core).',
        '  * S22B — Marshall, Burns, Sherrill, *JCP* **135**, 194102 (2011).',
        '    Second revision (CCSD(T)/CBS with corrections); current standard.',
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    for revision in ("S220", "S22A", "S22B"):
        d = refs[revision]
        lines.append(f"{revision}: dict[int, float] = {{")
        for k in sorted(d):
            lines.append(f"    {k:>2d}: {d[k]:>8.3f},")
        lines.append("}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    text = SRC.read_text()
    geos = parse_geos(text)
    if len(geos) != 22:
        raise RuntimeError(f"expected 22 S22 entries, got {len(geos)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx in sorted(geos):
        slug = SLUGS[idx]
        monos = split_monomers(geos[idx])
        if len(monos) != 2:
            raise RuntimeError(f"S22-{idx} ({slug}): expected 2 monomers, got {len(monos)}")
        a_atoms, b_atoms = monos
        dimer_atoms = a_atoms + b_atoms
        subset = SUBSETS[idx]
        stem = f"s22-{idx:02d}-{slug}"
        write_xyz(OUT_DIR / f"{stem}.xyz", dimer_atoms,
                  f"S22-{idx} ({subset}) {slug} — dimer (Jurečka 2006)")
        write_xyz(OUT_DIR / f"{stem}-monoA.xyz", a_atoms,
                  f"S22-{idx} ({subset}) {slug} — monoA (Jurečka 2006)")
        write_xyz(OUT_DIR / f"{stem}-monoB.xyz", b_atoms,
                  f"S22-{idx} ({subset}) {slug} — monoB (Jurečka 2006)")
        print(f"  S22-{idx:02d} {subset} {slug:<32s} "
              f"dimer={len(dimer_atoms):>3d} monoA={len(a_atoms):>3d} monoB={len(b_atoms):>3d}")

    refs = {name: parse_bind_dict(text, f"BIND_{name}") for name in ("S220", "S22A", "S22B")}
    if any(len(refs[k]) != 22 for k in refs):
        raise RuntimeError(f"reference revision counts: {{k: len(v) for k,v in refs.items()}}")
    write_reference_module(HERE / "reference_energies.py", refs)
    print(f"\n  Reference energies written to {HERE / 'reference_energies.py'}")


if __name__ == "__main__":
    main()
