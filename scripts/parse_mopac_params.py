#!/usr/bin/env python3
"""Parse MOPAC PM6 parameters from Fortran source and dump as TOML.

Usage:
    python parse_mopac_params.py [--output PM6_params.toml]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_F90_URL = (
    "https://raw.githubusercontent.com/openmopac/mopac/main/"
    "src/models/parameters_for_PM6_C.F90"
)


def fetch_and_parse(url: str = _F90_URL) -> dict[int, dict]:
    """Fetch the MOPAC PM6 Fortran parameter file and parse all elements."""
    import urllib.request

    with urllib.request.urlopen(url) as r:
        text = r.read().decode("utf-8")
    return fetch_and_parse_from_text(text)


def fetch_and_parse_from_text(text: str) -> dict[int, dict]:
    """Parse MOPAC PM6 element parameters from Fortran source text."""

    # Parse DATA statements:  data ARRAY(INDEX[, J])/ VALUE/
    arrays: dict[str, dict] = {}
    pat = re.compile(
        r"data\s+(\w+)\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*/\s*([-\d.]+D[+\-\d]+)"
    )
    for m in pat.finditer(text):
        name = m.group(1)
        idx = int(m.group(2))
        jdx = int(m.group(3)) if m.group(3) else 0
        val = float(m.group(4).replace("D", "E"))
        d = arrays.setdefault(name, {})
        if jdx > 0:
            d.setdefault(idx, {})[jdx] = val
        else:
            d[idx] = val

    # Parse element headers:  ! Data for Element   N  Name
    elements: dict[int, str] = {}
    for m in re.finditer(r"Data for Element\s+(\d+)\s+(\w+)", text):
        atomic_number = int(m.group(1))
        if 1 <= atomic_number <= 98:
            elements[atomic_number] = m.group(2).title()

    def _get(arr: str, Z: int, default: float = 0.0) -> float:
        entry = arrays.get(arr, {}).get(Z)
        if isinstance(entry, dict):
            return entry.get(1, default)
        return entry if entry is not None else default

    def _get_term(arr: str, Z: int, n: int) -> float:
        """Get Gaussian term N for element Z."""
        entry = arrays.get(arr, {}).get(Z, {})
        if isinstance(entry, dict):
            return entry.get(n, 0.0)
        return 0.0

    result: dict[int, dict] = {}
    for Z in sorted(elements):
        has_p = _get("upp6", Z, 0) != 0 or _get("gpp6", Z, 0) != 0
        has_d = _get("udd6", Z, 0) != 0
        n_orbitals = 9 if has_d else (4 if has_p else 1)

        gaussian_terms = []
        for n in range(1, 5):
            c = _get_term("gues61", Z, n)
            e = _get_term("gues62", Z, n)
            f = _get_term("gues63", Z, n)
            if c != 0.0 or e != 0.0:
                gaussian_terms.append({"coeff": c, "exponent": e, "factor": f})

        result[Z] = {
            "Z": Z,
            "symbol": elements[Z],
            "n_orbitals": n_orbitals,
            "has_d": has_d,
            "uss": _get("uss6", Z),
            "upp": _get("upp6", Z),
            "udd": _get("udd6", Z),
            "betas": _get("betas6", Z),
            "betap": _get("betap6", Z),
            "betad": _get("betad6", Z),
            "zs": _get("zs6", Z, 1.0),
            "zp": _get("zp6", Z, 1.0),
            "zd": _get("zd6", Z, 1.0),
            "gss": _get("gss6", Z),
            "gpp": _get("gpp6", Z),
            "gsp": _get("gsp6", Z),
            "gp2": _get("gp26", Z),
            "hsp": _get("hsp6", Z),
            "alpha": _get("alp6", Z),
            "polvo": _get("polvo6", Z),
            "poc": _get("poc_6", Z),
            "f0sd": _get("f0sd6", Z),
            "g2sd": _get("g2sd6", Z),
            "gaussian_terms": gaussian_terms,
        }

    return result


def dump_toml(params: dict[int, dict], path: Path) -> None:
    """Write parameters as TOML."""
    lines = [
        "# MOPAC PM6 parameters (open-source MOPAC 22)",
        "# Extracted from parameters_for_PM6_C.F90",
        "# DOI: 10.1007/s00894-007-0233-4",
        "",
    ]
    for Z in sorted(params):
        p = params[Z]
        lines.append("[[element]]")
        for key in [
            "Z",
            "symbol",
            "n_orbitals",
            "has_d",
            "uss",
            "upp",
            "udd",
            "betas",
            "betap",
            "betad",
            "zs",
            "zp",
            "zd",
            "gss",
            "gpp",
            "gsp",
            "gp2",
            "hsp",
            "alpha",
            "polvo",
            "poc",
            "f0sd",
            "g2sd",
        ]:
            val = p.get(key)
            if isinstance(val, float):
                lines.append(f"{key} = {val:.8f}")
            elif isinstance(val, bool):
                lines.append(f'{key} = {str(val).lower()}')
            elif isinstance(val, int):
                lines.append(f'{key} = {val}')
            else:
                lines.append(f'{key} = "{val}"')
        gt = p.get("gaussian_terms", [])
        if gt:
            lines.append("gaussian_terms = [")
            for g in gt:
                lines.append(
                    f"  {{ coeff = {g['coeff']:.8f}, "
                    f"exponent = {g['exponent']:.8f}, "
                    f"factor = {g['factor']:.8f} }},"
                )
            lines.append("]")
        lines.append("")
    path.write_text("\n".join(lines))


def parse_diatomic_pairs(text: str) -> dict[str, dict]:
    """Parse alpb_and_xfac_pm6 subroutine for diatomic pair parameters.

    Returns dict with keys 'alpb' and 'xfac', each mapping (Z1,Z2) -> value.
    Only lower-triangle (Z1 >= Z2) entries are present; the arrays are symmetric.
    """
    # Find the subroutine
    start = text.find("subroutine alpb_and_xfac_pm6")
    if start < 0:
        return {}
    end = text.find("end subroutine alpb_and_xfac_pm6", start)
    if end < 0:
        end = text.find("end subroutine", start + 500)
    block = text[start:end] if end > start else text[start:]

    pairs: dict[str, dict] = {}
    pat = re.compile(
        r"(alpb|xfac)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([-\d.]+)D([+\-\d]+)"
    )
    for m in pat.finditer(block):
        key = m.group(1)  # alpb or xfac
        z1 = int(m.group(2))
        z2 = int(m.group(3))
        if not (1 <= z1 <= 98 and 1 <= z2 <= 98):
            continue
        val = float(m.group(4) + "E" + m.group(5))
        d = pairs.setdefault(key, {})
        d[(max(z1, z2), min(z1, z2))] = val
    return pairs


def dump_toml_with_pairs(params, pairs, path):
    """Write parameters and diatomic pairs as TOML."""
    lines = [
        "# MOPAC PM6 parameters (open-source MOPAC 22)",
        "# Extracted from parameters_for_PM6_C.F90",
        "# DOI: 10.1007/s00894-007-0233-4",
        "",
    ]
    # Elements
    for Z in sorted(params):
        p = params[Z]
        lines.append("[[element]]")
        for key in [
            "Z", "symbol", "n_orbitals", "has_d",
            "uss", "upp", "udd",
            "betas", "betap", "betad",
            "zs", "zp", "zd",
            "gss", "gpp", "gsp", "gp2", "hsp",
            "alpha", "polvo", "f0sd", "g2sd",
        ]:
            val = p.get(key)
            if isinstance(val, float):
                lines.append(f"{key} = {val:.8f}")
            elif isinstance(val, bool):
                lines.append(f"{key} = {str(val).lower()}")
            elif isinstance(val, int):
                lines.append(f"{key} = {val}")
            else:
                lines.append(f'{key} = "{val}"')
        gt = p.get("gaussian_terms", [])
        if gt:
            lines.append("gaussian_terms = [")
            for g in gt:
                lines.append(
                    f"  {{ coeff = {g['coeff']:.8f}, "
                    f"exponent = {g['exponent']:.8f}, "
                    f"factor = {g['factor']:.8f} }},"
                )
            lines.append("]")
        lines.append("")

    # Diatomic pairs
    for key in ["alpb", "xfac"]:
        if key in pairs:
            lines.append(f"# Diatomic {key} parameters (lower triangle, symmetric)")
            for (z1, z2), val in sorted(pairs[key].items()):
                lines.append("[[diatomic_pair]]")
                lines.append(f"kind = \"{key}\"")
                lines.append(f"Z1 = {z1}")
                lines.append(f"Z2 = {z2}")
                lines.append(f"value = {val:.8f}")
                lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    import urllib.request

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("PM6_params.toml")
    with urllib.request.urlopen(_F90_URL) as r:
        text = r.read().decode("utf-8")
    params = fetch_and_parse_from_text(text)
    pairs = parse_diatomic_pairs(text)
    dump_toml_with_pairs(params, pairs, output)
    pair_count = sum(len(values) for values in pairs.values())
    print(f"Wrote {len(params)} elements + {pair_count} diatomic pairs to {output}")
