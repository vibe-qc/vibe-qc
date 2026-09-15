"""Regenerate ``msindo_params.json`` — the bundled MSINDO parameter table.

Parses the element-indexed ``DATA`` arrays of MSINDO's ``datas.f`` (BLOCK DATA)
into a single JSON bundle (Z = 1..54).  This is *build-time tooling* (CLAUDE.md
§10): it reads the maintainer's local MSINDO source, which is not in the repo;
the committed ``msindo_params.json`` is the shipped, citable data.

The parameters are the published MSINDO parameter set (Ahlswede & Jug,
J. Comput. Chem. 20, 563 & 572 (1999) for H, C–F; Bredow, Geudtner & Jug and
later papers for the 3rd row, d shells and transition metals; © Mulliken Center
for Theoretical Chemistry, University of Bonn) — bundled by permission as
published data (docs/license.md).

Usage::

    MSINDO_SRC=/path/to/2025e .venv/bin/python \\
        examples/regression/msindo/gen_msindo_params.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SRC = (Path(os.environ.get("MSINDO_SRC", "/Users/USER/gitlab/msindo/2025e"))
       / "source" / "datas.f")
LINES = SRC.read_text().splitlines()

# 1-D element-indexed arrays we bundle (exponents, resonance, ionization, core
# exponents/potentials, shielding, valence occupations).
ARRAYS = [
    "MUS", "MUP", "MUD", "MUSE", "MUPE", "MUDE",
    "KSS", "KPS", "KPP", "KDS", "KDP", "KDD",
    "IPOTS", "IPOTP", "IPOTD",
    "TAU1S", "TAU2S", "TAU2P", "TAU3S", "TAU3P", "TAU3D",
    "TAU4S", "TAU4P", "TAU4D",
    "FCP1S", "FCP2S", "FCP2P", "FCP3S", "FCP3P", "FCP3D",
    "FCP4S", "FCP4P", "FCP4D",
    "SCP3D", "SCP4S", "SCP4P",
    "LS", "MP", "ND",
]

# AL partner-group boundaries (datas.f SHD groups) — AL is constant within each
# group, so one value per group is stored per central atom.
AL_GROUPS = [(1, 2), (3, 5), (6, 10), (11, 12), (13, 18), (19, 20),
             (21, 30), (31, 36), (37, 38), (39, 48), (49, 54)]
MAXELM = 54


def _expand(text):
    """Numeric tokens of a DATA body, expanding ``N*value`` repeats."""
    vals = []
    for tok in (t.strip() for t in text.split(",") if t.strip()):
        m = re.match(r"(\d+)\s*\*\s*(-?[\d.]+(?:[eE][-+]?\d+)?)", tok)
        if m:
            vals.extend([float(m.group(2))] * int(m.group(1)))
        else:
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return vals


def parse_1d(name):
    out = {}
    i = 0
    while i < len(LINES):
        m = re.match(rf"\s*DATA\s+{re.escape(name)}\s*/(.*)$", LINES[i])
        if not m:
            i += 1
            continue
        body = m.group(1)
        chunk = [body]
        while "/" not in body.split("!")[0]:
            i += 1
            body = LINES[i]
            chunk.append(body)
        text = ""
        for tl in chunk:
            text += " " + re.sub(r"^\s*&", "", tl.split("!")[0])
        text = text.split("/")[0]
        for z, v in enumerate(_expand(text), start=1):
            out[z] = v
        return out
    return out


def parse_AL():
    al = {}
    pat1 = re.compile(r"\s*DATA\s+AL\((\d+),(\d+):(\d+)\)\s*/\s*(.*?)/")
    pat2 = re.compile(r"\s*DATA\s*\(\s*AL\((\d+),I\)\s*,\s*I\s*=\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*(.*?)/")
    for line in LINES:
        m = pat1.match(line) or pat2.match(line)
        if not m:
            continue
        zk, lo, hi = int(m.group(1)), int(m.group(2)), int(m.group(3))
        body = m.group(4).strip()
        mm = re.match(r"(\d+)\s*\*\s*(-?[\d.]+)", body)
        val = float(mm.group(2)) if mm else float(body)
        al.setdefault(zk, {})
        for zl in range(lo, hi + 1):
            al[zk][zl] = val
    return al


def main() -> None:
    arrays = {a: parse_1d(a) for a in ARRAYS}
    al = parse_AL()
    elements = {}
    for z in range(1, MAXELM + 1):
        e = {a: arrays[a].get(z, 0.0) for a in ARRAYS}
        e["AL"] = [al.get(z, {}).get(lo, 0.0) for lo, _ in AL_GROUPS]
        elements[str(z)] = e

    bundle = {
        "_provenance": (
            "MSINDO parameter set (DATA arrays of datas.f), Z=1..54. Published "
            "data: Ahlswede & Jug, J. Comput. Chem. 20, 563 & 572 (1999) (H, "
            "C-F); Bredow, Geudtner & Jug and later papers (3rd row, d shells, "
            "transition metals). (C) Mulliken Center for Theoretical Chemistry, "
            "University of Bonn; bundled by permission as published data "
            "(docs/license.md). Regenerate with "
            "examples/regression/msindo/gen_msindo_params.py."
        ),
        "_al_groups": AL_GROUPS,
        "units": "atomic (exponents bohr^-1; energies/potentials Hartree)",
        "elements": elements,
    }
    out = (Path(__file__).resolve().parents[3] / "python" / "vibeqc"
           / "semiempirical" / "methods" / "msindo_params.json")
    out.write_text(json.dumps(bundle, indent=1) + "\n")
    n = sum(1 for e in elements.values() if e["MUS"] != 0.0)
    print(f"wrote {out} ({n} parametrized elements, Z=1..{MAXELM})")


if __name__ == "__main__":
    main()
