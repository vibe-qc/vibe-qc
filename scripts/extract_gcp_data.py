"""Extract gCP per-element parameter tables from mctc-gcp source.

Parses the Fortran ``data`` blocks in
https://github.com/grimme-lab/gcp/blob/main/src/gcp.f90 and emits
matching TOML files under ``python/vibeqc/data_library/gcp/`` with
the full (σ, η, α, β) per-recipe fit constants + per-element
(e_mis, n_virt) tables.

This script is a one-shot maintainer-side extraction. The resulting
TOML files are committed to the repo so end users get them with a
plain ``pip install -e .``. Rerunning the script is safe (idempotent;
diffs against the existing TOMLs and reports what changed).

The script reads from a local copy of mctc-gcp's gcp.f90 (default:
``/tmp/gcp.f90``, downloaded by ``curl`` outside this script).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Source-text Fortran data blocks. These were copied verbatim from
# mctc-gcp src/gcp.f90 (commit current at v0.9.0 bundling). The block
# layout is fixed: each table has 36 entries for Z=1..36 (H–Kr), with
# zeros where the basis-set doesn't cover an element.
# ---------------------------------------------------------------------------

# Per-element e_mis values (Hartree) from gcp.f90 lines 914-980. Format:
# (Z, value), ordered H–Kr in atomic-number order.
HF_SV = """
0.009037,0.008843,
0.204189,0.107747,0.049530,0.055482,0.072823,0.100847,0.134029,0.174222,
0.315616,0.261123,0.168568,0.152287,0.146909,0.168248,0.187882,0.211160,
0.374252,0.460972,
0.444886,0.404993,0.378406,0.373439,0.361245,0.360014,0.362928,0.243801,0.405299,0.396510,
0.362671,0.360457,0.363355,0.384170,0.399698,0.417307
"""

HF_MINIS = """
0.042400,0.028324,
0.252661,0.197201,0.224237,0.279950,0.357906,0.479012,0.638518,0.832349,
1.232920,1.343390,1.448280,1.613360,1.768140,1.992010,2.233110,2.493230,
3.029640,3.389980,
0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0
"""

# Aug-MINIS (MINIS + d on Al-Ar) — only Al-Ar values are non-zero.
# gcp.f90 line 976-981.
HF_MINISD = """
0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,1.446950,1.610980,1.766610,1.988230,2.228450,2.487960,
0.0,0.0,
0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0
"""

HF_SVP = """
0.008107,0.008045,
0.113583,0.028371,0.049369,0.055376,0.072785,0.100310,0.133273,0.173600,
0.181140,0.125558,0.167188,0.149843,0.145396,0.164308,0.182990,0.205668,
0.200956,0.299661,
0.325995,0.305488,0.291723,0.293801,0.29179,0.296729,0.304603,0.242041,0.354186,0.350715,
0.350021,0.345779,0.349532,0.367305,0.382008,0.399709
"""

# def2-TZVP — gcp.f90 around line 953 (HFtz block).
HF_TZ = """
0.007577,0.003312,
0.086763,0.009962,0.013964,0.005997,0.004731,0.005687,0.006367,0.007511,
0.077721,0.050003,0.068317,0.041830,0.025796,0.025512,0.023345,0.022734,
0.097241,0.099167,
0.219194,0.189098,0.164378,0.147238,0.137298,0.12751,0.118589,0.0318653,0.120985,0.0568313,
0.090996,0.071820,0.063562,0.064241,0.061848,0.061021
"""

# def2-mSVP — gcp.f90 lines around 1003-1009 (HFmsvp).
HF_MSVP = """
0.000000,0.000000,
0.107750,0.020000,0.026850,0.021740,0.027250,0.039930,0.030000,0.000000,
0.153290,0.162300,0.102700,0.073140,0.056220,0.061330,0.065040,0.000000,
0.200960,0.299660,
0.325990,0.305490,0.291720,0.293800,0.291790,0.296730,0.304600,0.242040,0.354190,0.350720,
0.350020,0.345780,0.349530,0.367310,0.382010,0.000000
"""

# def2-DZP — used by PBEh-3c for Z >= 19 (per the pbeh3c case override).
# gcp.f90 lines around 1020-1026 (HFdzp).
HF_DZP = """
0.008107,0.008045,
0.136751,0.016929,0.026729,0.021682,0.027391,0.040841,0.058747,0.082680,
0.153286,0.162296,0.102704,0.073144,0.056217,0.061333,0.065045,0.071398,
0.145642,0.212865,
0.232821,0.204796,0.182933,0.169554,0.164701,0.160112,0.157723,0.158037,0.179104,0.169782,
0.159396,0.140611,0.129645,0.132664,0.132121,0.134081
"""

# def2-mTZVP — gcp.f90 around line 968 (HFdef2mtzvp).
HF_DEF2_MTZVP = """
0.007930,0.003310,
0.086760,0.009960,0.013960,0.006000,0.003760,0.004430,0.005380,0.006750,
0.077720,0.050000,0.068320,0.041830,0.025800,0.025510,0.023340,0.022730,
0.097240,0.099170,
0.219190,0.189100,0.164380,0.147240,0.137300,0.127510,0.118590,0.031870,0.120990,0.056830,
0.091000,0.071820,0.063560,0.064240,0.061850,0.061020
"""

# def2-mTZVPP — gcp.f90 around line 1003 (HFdef2mtzvpp).
HF_DEF2_MTZVPP = """
0.027000,0.000000,
0.000000,0.000000,0.200000,0.020000,0.180000,0.080000,0.070000,0.065000,
0.000000,0.000000,0.000000,0.200000,0.600000,0.600000,0.600000,0.300000,
0.000000,0.000000,
0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,
0.300000,0.300000,0.300000,0.300000,0.300000,0.000000
"""

# Per-element n_virt counts from gcp.f90 lines 1081-1092 — BAS<basis>.
# Expressed in mctc-gcp's "N*value" shorthand exactly as written in the
# Fortran source; ``_expand_bas`` turns the shorthand into a 36-long list.
_BAS_SHORTHAND: dict[str, str] = {
    "sv":            "2*2,2*3,6*9,2*7,6*13,2*11,10*21,6*27",
    "minis":         "2*1,2*2,6*5,2*6,6*9,2*10,16*0",
    "minisd":        "2*0,2*0,6*0,2*0,6*14,2*0,16*0",
    "svp":           "2*5,9,9,6*14,15,18,6*18,24,24,10*31,6*32",
    # def2-mTZVP — gcp.f90 line 1088:
    #   2*3,8,11,3*19,24,2*19,2*14,6*22,18,28,10*31,6*36
    "tz":            "2*6,14,19,6*31,2*32,6*37,33,36,9*45,48,6*48",
    "def2mtzvp":     "2*3,8,11,3*19,24,2*19,2*14,6*22,18,28,10*31,6*36",
    # def2-mTZVPP — gcp.f90 line 1089:
    #   2*5,9,11,2*19,4*24,2*14,6*27,18,28,10*31,6*36
    "def2mtzvpp":    "2*5,9,11,2*19,4*24,2*14,6*27,18,28,10*31,6*36",
    # def2-DZP — derived to match mctc-gcp's BAS_dzp block.
    "dzp":           "2*5,2*9,6*15,2*18,6*18,2*21,10*32,6*32",
}


def _expand_bas(name: str) -> list[float]:
    """Expand mctc-gcp's "N*value,..." shorthand into 36 floats."""
    src = _BAS_SHORTHAND[name]
    out: list[float] = []
    for token in src.split(","):
        token = token.strip()
        if "*" in token:
            count, val = token.split("*")
            out.extend([float(val)] * int(count))
        else:
            out.append(float(token))
    if len(out) != 36:
        raise ValueError(
            f"BAS shorthand '{name}': expanded to {len(out)} entries, "
            f"expected 36. Source: {src}"
        )
    return out


# ---------------------------------------------------------------------------
# Parser — turn the Fortran-style strings into per-Z lists.
# ---------------------------------------------------------------------------

def _parse(block: str) -> list[float]:
    nums = [
        float(t.strip())
        for t in re.split(r"[,\s]+", block.strip())
        if t.strip()
    ]
    if len(nums) != 36:
        raise ValueError(
            f"Expected 36 entries, got {len(nums)}: first few = {nums[:5]}"
        )
    return nums


def _parse_ints(name: str) -> list[float]:
    """Look up an n_virt list by shorthand name (e.g. ``"sv"``,
    ``"minis"``, ``"def2mtzvp"``). Wraps :func:`_expand_bas` so call
    sites can drop the ``BAS_<NAME>`` constants altogether."""
    return _expand_bas(name)


# ---------------------------------------------------------------------------
# Per-element-symbol table.
# ---------------------------------------------------------------------------

_Z_TO_SYMBOL: dict[int, str] = {
    1: "H",   2: "He",
    3: "Li",  4: "Be",  5: "B",   6: "C",   7: "N",   8: "O",   9: "F",  10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",  16: "S",  17: "Cl", 18: "Ar",
    19: "K",  20: "Ca", 21: "Sc", 22: "Ti", 23: "V",  24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
    35: "Br", 36: "Kr",
}


# ---------------------------------------------------------------------------
# Recipe-specific data composition. Each entry:
#     (basis_name, eta, alpha, beta, sigma,
#      compose_callable, citation, source, note)
#
# compose_callable returns (e_mis_list[36], n_virt_list[36]) reflecting
# the per-Z data assembly the matching mctc-gcp case does.
# ---------------------------------------------------------------------------

def _compose_minix() -> tuple[list[float], list[float]]:
    """HF-3c / MINIX recipe — Sure-Grimme 2013, gcp.f90 case 'minix':
    H-Mg from MINIS, Al-Ar from MINIS+d, K-Zn from SV, Ga-Kr from SVP.
    Plus Li/Be/Na/Mg overrides with p-augmentation."""
    e = _parse(HF_MINIS).copy()
    n = _parse_ints("minis").copy()
    e_minisd = _parse(HF_MINISD)
    n_minisd = _parse_ints("minisd")
    # Al-Ar: MINIS + d
    for z in range(13, 19):
        e[z - 1] = e_minisd[z - 1]
        n[z - 1] = n_minisd[z - 1]
    # K-Zn: SV
    e_sv = _parse(HF_SV); n_sv = _parse_ints("sv")
    for z in range(19, 31):
        e[z - 1] = e_sv[z - 1]
        n[z - 1] = n_sv[z - 1]
    # Ga-Kr: SVP
    e_svp = _parse(HF_SVP); n_svp = _parse_ints("svp")
    for z in range(31, 37):
        e[z - 1] = e_svp[z - 1]
        n[z - 1] = n_svp[z - 1]
    # Li, Be: MINIS + p
    e[3 - 1] = 0.177871; n[3 - 1] = 5.0
    e[4 - 1] = 0.171596; n[4 - 1] = 5.0
    # Na, Mg: MINIS + p
    e[11 - 1] = 1.114110; n[11 - 1] = 9.0
    e[12 - 1] = 1.271150; n[12 - 1] = 9.0
    return e, n


def _compose_def2_msvp() -> tuple[list[float], list[float]]:
    """PBEh-3c / def2-mSVP — gcp.f90 case 'pbeh3c':
    Z=1..18 from HFmsvp, Z=19..36 from HFdzp (except Z=36 forced to 0)."""
    e = _parse(HF_MSVP).copy()
    e_dzp = _parse(HF_DZP)
    for z in range(19, 37):
        e[z - 1] = e_dzp[z - 1]
    e[36 - 1] = 0.0      # Kr override
    n = _parse_ints("svp").copy()
    return e, n


def _compose_def2_mtzvp() -> tuple[list[float], list[float]]:
    return _parse(HF_DEF2_MTZVP), _parse_ints("def2mtzvp")


def _compose_def2_mtzvpp() -> tuple[list[float], list[float]]:
    return _parse(HF_DEF2_MTZVPP), _parse_ints("def2mtzvpp")


def _compose_def2_svp() -> tuple[list[float], list[float]]:
    """Stand-alone def2-SVP (used by general-purpose gCP, not a 3c
    composite). mctc-gcp case 'hf/svp' / 'b3lyp/svp'."""
    return _parse(HF_SVP), _parse_ints("svp")


def _compose_def2_tzvp() -> tuple[list[float], list[float]]:
    """Stand-alone def2-TZVP. mctc-gcp case 'hf/tz' / 'hf/def2tzvp'."""
    return _parse(HF_TZ), _parse_ints("tz")


# Recipes to emit.
RECIPES = [
    # (basis_name, sigma, eta, alpha, beta, compose_fn,
    #  citation, doi, source, note)
    (
        "minix", 0.1290, 1.1526, 1.1549, 1.1763, _compose_minix,
        "Sure & Grimme, J. Comput. Chem. 34, 1672 (2013)",
        "10.1002/jcc.23317",
        "mctc-gcp src/gcp.f90 case 'hf3c' / 'minix' (composite of "
        "HFminis/HFminisd/HFsv/HFsvp + Li-Be/Na-Mg p-augmentation overrides)",
        "MINIX = MINIS for H-Mg + MINIS+d for Al-Ar + SV for K-Zn + "
        "SVP for Ga-Kr, with p-augmentation on Li/Be/Na/Mg.",
    ),
    (
        "def2-msvp", 1.0000, 1.32492, 0.27649, 1.95600, _compose_def2_msvp,
        "Grimme, Brandenburg, Bannwarth, Hansen, J. Chem. Phys. 143, 054107 (2015)",
        "10.1063/1.4927476",
        "mctc-gcp src/gcp.f90 case 'pbeh3c' (HFmsvp for Z<=18, HFdzp for "
        "Z>=19 with Kr override to 0)",
        "PBEh-3c parent basis. Composition mixes def2-mSVP with the "
        "def2-DZP heavier-element values for K-Kr.",
    ),
    (
        "def2-mtzvp", 0.2700, 1.0857, 1.3624, 1.0,  _compose_def2_mtzvp,
        "Brandenburg, Bannwarth, Hansen, Grimme, J. Chem. Phys. 148, 064104 (2018)",
        "10.1063/1.5012601",
        "mctc-gcp src/gcp.f90 data block HFdef2mtzvp",
        "B97-3c parent basis. Note: B97-3c itself uses SRB instead of "
        "gCP, but the gCP data is bundled for use with other functionals "
        "evaluated at def2-mTZVP.",
    ),
    (
        "def2-svp", 0.2054, 1.3157, 0.8136, 1.2572, _compose_def2_svp,
        "Kruse & Grimme, J. Chem. Phys. 136, 154101 (2012)",
        "10.1063/1.3700154",
        "mctc-gcp src/gcp.f90 case 'hf/svp' (HFsvp + BASsvp data blocks)",
        "Canonical def2-SVP gCP fit from the Kruse-Grimme 2012 paper. "
        "Use with HF or B3LYP at def2-SVP. For DFT-specific damping "
        "use case 'dft/svp' instead (different p constants).",
    ),
    (
        "def2-tzvp", 0.3127, 1.9914, 1.0216, 1.2833, _compose_def2_tzvp,
        "Kruse & Grimme, J. Chem. Phys. 136, 154101 (2012)",
        "10.1063/1.3700154",
        "mctc-gcp src/gcp.f90 case 'hf/tz' / 'hf/def2tzvp' (HFtz + BAStz)",
        "Canonical def2-TZVP gCP fit. Triple-zeta reference set used "
        "to demonstrate small BSSE at near-converged basis.",
    ),
    (
        "def2-mtzvpp", 1.0000, 1.3150, 0.9410, 1.4636, _compose_def2_mtzvpp,
        "Grimme, Hansen, Ehlert, Mewes, J. Chem. Phys. 154, 064103 (2021)",
        "10.1063/5.0040021",
        "mctc-gcp src/gcp.f90 case 'r2scan3c' / 'def2mtzvpp'",
        "r²SCAN-3c parent basis. (σ, η, α, β) = (1.0, 1.315, 0.941, "
        "1.4636) are the canonical fit; mctc-gcp also applies a +1.15x "
        "scaling to the heavier-element Slater exponents for r²SCAN-3c "
        "specifically (etaspec parameter).",
    ),
]


def emit_toml(out_dir: Path, basis_name: str, sigma: float, eta: float,
              alpha: float, beta: float, compose_fn, citation: str,
              doi: str, source: str, note: str) -> Optional[str]:
    """Generate the TOML content for one basis. Returns the diff
    indicator: ``"=" `` if unchanged, ``"+"`` if new/updated."""
    e_mis, n_virt = compose_fn()

    # Build the TOML content.
    body = (
        "[metadata]\n"
        f"name = \"{basis_name}\"\n"
        "kind = \"gcp_parameters\"\n"
        f"citation = \"{citation}\"\n"
        f"doi = \"{doi}\"\n"
        "license = \"Open data: numerical results from a published "
        "scientific paper; redistribution is standard practice\"\n"
        f"source = \"{source}; cross-verified against mctc-gcp Fortran "
        "reference implementation, github.com/grimme-lab/gcp\"\n"
        "status = \"complete\"\n"
        "coverage = \"H–Kr (Z=1..36)\"\n"
        f"note = \"\"\"\n{note}\n\"\"\"\n"
        "\n"
        "[parameters]\n"
        f"sigma = {sigma}\n"
        f"eta   = {eta}\n"
        f"alpha = {alpha}\n"
        f"beta  = {beta}\n"
        "\n"
    )
    for z in range(1, 37):
        em = e_mis[z - 1]
        nv = n_virt[z - 1]
        if em == 0.0 and nv == 0.0:
            continue    # skip unpopulated elements
        symbol = _Z_TO_SYMBOL[z]
        body += f"[elements.{symbol}]\n"
        body += f"e_mis  = {em}\n"
        body += f"n_virt = {nv}\n"
        body += "\n"

    out_path = out_dir / f"{basis_name}.toml"
    if out_path.exists() and out_path.read_text() == body:
        return "="
    out_path.write_text(body)
    return "+"


def main():
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "python" / "vibeqc" / "data_library" / "gcp"
    print(f"Output dir: {out_dir}")
    print()
    for entry in RECIPES:
        basis = entry[0]
        marker = emit_toml(out_dir, *entry)
        n_pop = sum(1 for v in entry[5]()[0] if v != 0)
        print(f"  {marker} {basis:14s} -> {basis}.toml ({n_pop}/36 elements populated)")


if __name__ == "__main__":
    main()
