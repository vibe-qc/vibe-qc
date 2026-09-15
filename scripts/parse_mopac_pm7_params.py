#!/usr/bin/env python3
"""Parse MOPAC PM7 parameters (parameters_for_PM7_C.F90) into TOML.

Reads the local MOPAC source or fetches from GitHub.
Generates python/vibeqc/semiempirical/methods/pm7_mopac_params.toml
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

_F90_URL = (
    "https://raw.githubusercontent.com/openmopac/mopac/main/"
    "src/models/parameters_for_PM7_C.F90"
)
_LOCAL = Path(__file__).resolve().parents[2] / "mopac/src/models/parameters_for_PM7_C.F90"
_OUT = (
    Path(__file__).resolve().parents[1]
    / "python/vibeqc/semiempirical/methods/pm7_mopac_params.toml"
)


def dfloat(s: str) -> float:
    """Convert Fortran D-format to Python float."""
    return float(s.replace("D", "E").replace("d", "e"))


def parse_f90(text: str) -> dict:
    """Parse MOPAC PM7 Fortran parameter file."""
    arrays = {
        "uss", "upp", "udd",
        "betas", "betap", "betad",
        "zs", "zp", "zd",
        "zsn", "zpn", "zdn",
        "gss", "gsp", "gpp", "gp2", "hsp",
        "alp", "polvo", "poc_", "f0sd", "g2sd",
        "CPE_Zet", "CPE_Z0", "CPE_B", "CPE_Xlo", "CPE_Xhi",
        "gues71", "gues72", "gues73",
    }

    elem_data: dict[int, dict] = {}
    diatomic: dict[tuple[int, int], dict] = {}

    # Parse element arrays: data name7(Z) / valueD0 /
    for name in arrays:
        pattern = re.compile(
            r"data\s+" + name + r"7?\s*\(\s*(\d+)\s*\)\s*/\s*([\d\.\-\+Dd]+)D0/"
        )
        for m in pattern.finditer(text):
            z = int(m.group(1))
            if not 1 <= z <= 98:
                continue
            val = dfloat(m.group(2))
            if z not in elem_data:
                elem_data[z] = {}
            # Strip trailing "7" for PM7 array names (uss7 -> uss, etc.).
            # `poc_7` is the one array whose base name ends in an underscore,
            # so rstrip("7") leaves "poc_" and the writer -- which emits
            # "poc" -- silently dropped every fitted core radius (issue #224).
            # The PM6 parser special-cases the same array; this matches it.
            clean = "poc" if name == "poc_" else name.rstrip("7")
            elem_data[z][clean] = val

    # Parse multi-index: gues71(Z, idx)
    for name in ("gues71", "gues72", "gues73"):
        pattern = re.compile(
            r"data\s+"
            + name
            + r"7?\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*([\d\.\-\+Dd]+)D0/"
        )
        for m in pattern.finditer(text):
            z = int(m.group(1))
            if not 1 <= z <= 98:
                continue
            idx = int(m.group(2))
            val = dfloat(m.group(3))
            if z not in elem_data:
                elem_data[z] = {}
            key = name.rstrip("7")
            if key not in elem_data[z]:
                elem_data[z][key] = {}
            elem_data[z][key][idx] = val

    # Parse diatomic pairs: alpb(Z1, Z2) = value, xfac(Z1, Z2) = value
    for nm in ("alpb", "xfac"):
        pattern = re.compile(
            r"\s+" + nm + r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([\d\.\-\+Dd]+)d0"
        )
        for m in pattern.finditer(text):
            z1, z2 = int(m.group(1)), int(m.group(2))
            if not (1 <= z1 <= 98 and 1 <= z2 <= 98):
                continue
            val = dfloat(m.group(3))
            key = (min(z1, z2), max(z1, z2))
            if key not in diatomic:
                diatomic[key] = {}
            diatomic[key][nm] = val

    return {"elements": elem_data, "diatomic": diatomic}


def write_toml(data: dict, path: Path) -> None:
    """Write parsed data to TOML."""
    lines = [
        "# PM7 parameters parsed from MOPAC (parameters_for_PM7_C.F90)",
        "# License: Apache 2.0 (MOPAC)",
        f"# Source: {_F90_URL}",
        "",
        "[metadata]",
        'method = "PM7"',
        'source = "MOPAC parameters_for_PM7_C.F90"',
        'version = "pm7-mopac-2025"',
        'license = "Apache-2.0"',
        'reference = "J. J. P. Stewart, J. Mol. Model. 19, 1-32 (2013)"',
        "",
    ]

    # Elements
    for z in sorted(data["elements"]):
        d = data["elements"][z]
        lines.append("[[element]]")
        lines.append(f"Z = {z}")

        for k in ("uss", "upp", "udd", "betas", "betap", "betad"):
            if k in d:
                lines.append(f"{k} = {d[k]:.6f}")

        for k in ("zs", "zp", "zd", "zsn", "zpn", "zdn"):
            if k in d:
                lines.append(f"{k} = {d[k]:.6f}")

        for k in ("gss", "gsp", "gpp", "gp2", "hsp"):
            if k in d:
                lines.append(f"{k} = {d[k]:.6f}")

        for k in ("alp", "polvo", "poc", "f0sd", "g2sd"):
            if k in d:
                lines.append(f"{k} = {d[k]:.6f}")

        # CPE dispersion
        cpe_keys = {
            "CPE_Zet": "cpe_zet",
            "CPE_Z0": "cpe_z0",
            "CPE_B": "cpe_b",
            "CPE_Xlo": "cpe_xlo",
            "CPE_Xhi": "cpe_xhi",
        }
        for fk, tk in cpe_keys.items():
            if fk in d:
                lines.append(f"{tk} = {d[fk]:.6f}")

        # Gaussian expansion (PM7: gues71/72/73 up to 4 terms per element)
        for name in ("gues71", "gues72", "gues73"):
            if name in d:
                # Convert sparse dict to array
                max_idx = max(d[name].keys())
                vals = [d[name].get(i, 0.0) for i in range(1, max_idx + 1)]
                tag = {"gues71": "coeff", "gues72": "exponent", "gues73": "factor"}[name]
                lines.append(f"{tag} = {vals}")

        lines.append("")

    # Diatomic pairs
    for z1, z2 in sorted(data["diatomic"]):
        dp = data["diatomic"][(z1, z2)]
        lines.append("[[diatomic]]")
        lines.append(f"Z1 = {z1}")
        lines.append(f"Z2 = {z2}")
        for k in ("alpb", "xfac"):
            if k in dp:
                lines.append(f"{k} = {dp[k]:.6f}")
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--input":
        text = Path(sys.argv[2]).read_text()
    elif _LOCAL.exists():
        text = _LOCAL.read_text()
    else:
        with urllib.request.urlopen(_F90_URL) as r:
            text = r.read().decode("utf-8")

    data = parse_f90(text)
    write_toml(data, _OUT)
    print(
        f"Wrote {_OUT} ({_OUT.stat().st_size} bytes, "
        f"{len(data['elements'])} elements, {len(data['diatomic'])} diatomic pairs)"
    )


if __name__ == "__main__":
    main()
