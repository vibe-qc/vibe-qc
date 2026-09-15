"""GFN2-xTB parameter parser — converts xTB text format to vibe-qc TOML."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# Element symbol → atomic number mapping
_SYMBOL_TO_Z = {
    "h": 1,
    "he": 2,
    "li": 3,
    "be": 4,
    "b": 5,
    "c": 6,
    "n": 7,
    "o": 8,
    "f": 9,
    "ne": 10,
    "na": 11,
    "mg": 12,
    "al": 13,
    "si": 14,
    "p": 15,
    "s": 16,
    "cl": 17,
    "ar": 18,
    "k": 19,
    "ca": 20,
    "sc": 21,
    "ti": 22,
    "v": 23,
    "cr": 24,
    "mn": 25,
    "fe": 26,
    "co": 27,
    "ni": 28,
    "cu": 29,
    "zn": 30,
    "ga": 31,
    "ge": 32,
    "as": 33,
    "se": 34,
    "br": 35,
    "kr": 36,
    "rb": 37,
    "sr": 38,
    "y": 39,
    "zr": 40,
    "nb": 41,
    "mo": 42,
    "tc": 43,
    "ru": 44,
    "rh": 45,
    "pd": 46,
    "ag": 47,
    "cd": 48,
    "in": 49,
    "sn": 50,
    "sb": 51,
    "te": 52,
    "i": 53,
    "xe": 54,
    "cs": 55,
    "ba": 56,
    "la": 57,
    "hf": 72,
    "ta": 73,
    "w": 74,
    "re": 75,
    "os": 76,
    "ir": 77,
    "pt": 78,
    "au": 79,
    "hg": 80,
    "tl": 81,
    "pb": 82,
    "bi": 83,
    "rn": 86,
}

# Angular momentum label → l value
_L_TO_NUM = {"s": 0, "p": 1, "d": 2, "f": 3}


def _parse_ao_list(ao_str: str) -> list[int]:
    """Parse '1s2p' → [0, 1] or '2s2p1d' → [0, 1, 2]."""
    result = []
    for m in re.finditer(r"(\d)([spdf])", ao_str):
        l = _L_TO_NUM.get(m.group(2))
        if l is not None:
            result.append(l)
    return result


def _parse_element_symbol(header: str) -> tuple[int, str]:
    """Parse '$Z= 1 Wed...' or '$Z=10 Tue...' or '$Z= h Wed...' -> (Z, symbol)."""
    parts = header.split()
    first = parts[0]  # "$Z=10" or "$Z="
    if "=" in first:
        token = first.split("=", 1)[1]
        if not token and len(parts) > 1:
            token = parts[1]  # "$Z= 1" format
    elif len(parts) > 1:
        token = parts[1]
    else:
        return 0, ""
    if token.isdigit():
        return int(token), ""
    sym = token.lower()
    z = _SYMBOL_TO_Z.get(sym, 0)
    return z, sym

def parse_gfn2_params(text: str) -> str:
    """Parse xTB param_gfn2-xtb.txt and return vibe-qc TOML string.

    Returns a TOML string with [[element]] and [[repulsive]] sections.
    """
    lines = text.split("\n")
    toml_lines = []
    toml_lines.append("# GFN2-xTB parameters — converted from xTB format")
    toml_lines.append("# DOI: 10.1021/acs.jctc.8b01176")
    toml_lines.append("")

    # Global parameters
    toml_lines.append("[global]")
    in_globpar = False
    global_params = {}
    for line in lines:
        stripped = line.strip()
        if stripped == "$globpar":
            in_globpar = True
            continue
        if in_globpar and stripped == "$end":
            in_globpar = False
            continue
        if in_globpar and stripped and not stripped.startswith("$"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    global_params[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    for k, v in sorted(global_params.items()):
        toml_lines.append(f"{k} = {v}")
    toml_lines.append("")

    # Element data
    in_element = False
    current_z = 0
    ao_list: list[int] = []
    lev_values: list[float] = []
    exp_values: list[float] = []
    gam_value = 0.5
    alpha_value = 1.0

    def _flush_element():
        nonlocal current_z, ao_list, lev_values, exp_values, gam_value
        if current_z == 0:
            return
        toml_lines.append("[[element]]")
        toml_lines.append(f"Z = {current_z}")
        n_val = 0
        for l in ao_list:
            n_val += 2 * (2 * l + 1)
        toml_lines.append(f"shells = [")
        for i, l in enumerate(ao_list):
            en = lev_values[i] if i < len(lev_values) else 0.0
            zeta = exp_values[i] if i < len(exp_values) else 1.0
            # Convert eV to Ha (xTB uses eV for lev)
            en_ha = en / 27.2114
            toml_lines.append(
                f"  {{ l = {l}, en = {en_ha:.6f}, zeta = {zeta:.6f}, k_en = 1.0 }},"
            )
        toml_lines.append("]")
        toml_lines.append(f"gam = {gam_value:.6f}")
        toml_lines.append("")
        current_z = 0
        ao_list = []
        lev_values = []
        exp_values = []
        gam_value = 0.5

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("$Z="):
            _flush_element()
            z, sym = _parse_element_symbol(stripped)
            current_z = z
            in_element = True
            continue
        if in_element and stripped == "$end":
            _flush_element()
            in_element = False
            continue
        if in_element:
            if stripped.startswith("ao="):
                ao_str = stripped.split("=")[1].strip()
                ao_list = _parse_ao_list(ao_str)
            elif stripped.startswith("lev="):
                parts = stripped.split("=")[1].strip().split()
                lev_values = [float(x) for x in parts]
            elif stripped.startswith("exp="):
                parts = stripped.split("=")[1].strip().split()
                exp_values = [float(x) for x in parts]
            elif stripped.startswith("GAM=") and "GAM3" not in stripped:
                try:
                    gam_value = float(stripped.split("=")[1].strip())
                except ValueError:
                    pass
    _flush_element()

    return "\n".join(toml_lines)


if __name__ == "__main__":
    # Quick test
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text()
    else:
        import urllib.request

        url = "https://raw.githubusercontent.com/grimme-lab/xtb/main/param_gfn2-xtb.txt"
        with urllib.request.urlopen(url) as r:
            text = r.read().decode("utf-8")
    toml = parse_gfn2_params(text)
    print(toml)
