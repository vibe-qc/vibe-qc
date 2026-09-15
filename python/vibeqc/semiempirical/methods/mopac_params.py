"""MOPAC PM6 chemical-element parameters with diatomic pairs.

Fetches the published MOPAC PM6 Fortran parameter file from the
open-source MOPAC 22 repository, parses it, and caches as TOML.
Loads per-element parameters, Gaussian core-core correction terms
(gues61/gues62/gues63), and 933 diatomic pair parameters (alpb/xfac)
for the MOPAC PM6 core-core damping formula.

Usage:
    from vibeqc.semiempirical.methods.mopac_params import load_mopac_pm6_params
    params = load_mopac_pm6_params()
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

from vibeqc._vibeqc_core.semiempirical.nddo import NDDOElementData, PM6ParameterSet

_CACHE_DIR = Path.home() / ".cache" / "vibeqc"
_CACHE_FILE = _CACHE_DIR / "mopac_pm6_params.toml"


def _fetch_mopac_params() -> None:
    """Fetch MOPAC PM6 Fortran source, parse, and cache as TOML."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request

    _F90_URL = (
        "https://raw.githubusercontent.com/openmopac/mopac/main/"
        "src/models/parameters_for_PM6_C.F90"
    )
    try:
        with urllib.request.urlopen(_F90_URL) as r:
            text = r.read().decode("utf-8")
    except Exception:
        return  # keep existing cache or empty

    import re

    # Parse elements
    arrays: dict[str, dict] = {}
    pat = re.compile(
        r"data\s+(\w+)\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*/\s*([-\d.]+[dD][+\-\d]+)"
    )
    for m in pat.finditer(text):
        name = m.group(1)
        idx = int(m.group(2))
        jdx = int(m.group(3)) if m.group(3) else 0
        val = float(m.group(4).replace("D", "E").replace("d", "E"))
        d = arrays.setdefault(name, {})
        if jdx > 0:
            d.setdefault(idx, {})[jdx] = val
        else:
            d[idx] = val

    elements: dict[int, str] = {}
    for m in re.finditer(r"Data for Element\s+(\d+)\s+(\w+)", text):
        atomic_number = int(m.group(1))
        if 1 <= atomic_number <= 98:
            elements[atomic_number] = m.group(2).title()

    def _get(arr, Z, default=0.0):
        e = arrays.get(arr, {}).get(Z)
        return (
            e.get(1, default)
            if isinstance(e, dict)
            else (e if e is not None else default)
        )

    def _get_term(arr, Z, n):
        e = arrays.get(arr, {}).get(Z, {})
        return e.get(n, 0.0) if isinstance(e, dict) else 0.0

    # Parse diatomic pairs
    start = text.find("subroutine alpb_and_xfac_pm6")
    end = text.find("end subroutine alpb_and_xfac_pm6", start)
    block = text[start:end] if end > start else text[start:]
    pairs: dict[str, dict] = {}
    pat2 = re.compile(
        r"(alpb|xfac)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([-\d.]+[dD][+\-\d]*)"
    )
    for m in pat2.finditer(block):
        key = m.group(1)
        z1, z2 = int(m.group(2)), int(m.group(3))
        if not (1 <= z1 <= 98 and 1 <= z2 <= 98):
            continue
        val = float(m.group(4).replace("D", "E").replace("d", "E"))
        pairs.setdefault(key, {})[(max(z1, z2), min(z1, z2))] = val

    # Write TOML
    def fv(v):
        if isinstance(v, float):
            return f"{v:.8f}"
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, int):
            return str(v)
        return f'"{v}"'

    lines = [
        "# MOPAC PM6 parameters (open-source MOPAC 22)",
        "# DOI: 10.1007/s00894-007-0233-4",
        "",
    ]
    for Z in sorted(elements):
        has_p = _get("upp6", Z) != 0 or _get("gpp6", Z) != 0
        has_d = _get("udd6", Z) != 0
        no = 9 if has_d else (4 if has_p else 1)
        gt = []
        for n in range(1, 5):
            c, e, f = (
                _get_term("gues61", Z, n),
                _get_term("gues62", Z, n),
                _get_term("gues63", Z, n),
            )
            if c or e:
                gt.append({"coeff": c, "exponent": e, "factor": f})

        lines.append("[[element]]")
        for k, v in [
            ("Z", Z),
            ("symbol", elements[Z]),
            ("n_orbitals", no),
            ("has_d", has_d),
            ("uss", _get("uss6", Z)),
            ("upp", _get("upp6", Z)),
            ("udd", _get("udd6", Z)),
            ("betas", _get("betas6", Z)),
            ("betap", _get("betap6", Z)),
            ("betad", _get("betad6", Z)),
            ("zs", _get("zs6", Z, 1.0)),
            ("zp", _get("zp6", Z, 1.0)),
            ("zd", _get("zd6", Z, 1.0)),
            ("gss", _get("gss6", Z)),
            ("gpp", _get("gpp6", Z)),
            ("gsp", _get("gsp6", Z)),
            ("gp2", _get("gp26", Z)),
            ("hsp", _get("hsp6", Z)),
            ("alpha", _get("alp6", Z)),
            ("polvo", _get("polvo6", Z)),
            ("poc", _get("poc_6", Z)),
        ]:
            lines.append(f"{k} = {fv(v)}")
        # MOPAC multipole gamma terms (Priority 7).
        # When populated, the C++ diatomic_gammas path provides sigma/pi
        # decomposition of two-center two-electron integrals.  Currently
        # disabled: the MOPAC gss/gpp parameters are fitted for isotropic
        # Ohno-Klopman, not the sigma/pi-decomposed form.  Populating
        # gamma_terms changes energies in the wrong direction (-6.21->-10.53
        # for H2O).  The fix needs MOPAC's actual multipole expansion
        # coefficients (not the Slater exponents).
        #
        # To activate: generate 6 s-type + 6 p-type terms per element with
        # coefficients normalised so that sum(c_i * c_j) = 1 for same-shell
        # pairs, reproducing the Ohno-Klopman baseline.  Then refit the
        # one-center G parameters (gss/gpp/gsp/gp2/hsp) for the decomposed
        # two-center form.

        if gt:
            lines.append("gaussian_terms = [")
            for g in gt:
                lines.append(
                    f"  {{ coeff = {g['coeff']:.8f}, exponent = {g['exponent']:.8f}, factor = {g['factor']:.8f} }},"
                )
            lines.append("]")
        lines.append("")

    for kind in ["alpb", "xfac"]:
        if kind in pairs:
            lines.append(f"# Diatomic {kind}")
            for (z1, z2), val in sorted(pairs[kind].items()):
                lines.append("[[diatomic_pair]]")
                lines.append(f'kind = "{kind}"')
                lines.append(f"Z1 = {z1}")
                lines.append(f"Z2 = {z2}")
                lines.append(f"value = {val:.8f}")
                lines.append("")

    _CACHE_FILE.write_text("\n".join(lines))


def load_mopac_pm6_params(force_refetch: bool = False) -> PM6ParameterSet:
    """Load MOPAC PM6 chemical-element parameters and diatomic pairs.

    On first call, fetches and parses the Fortran parameter file from
    the open-source MOPAC 22 repository.  Includes Gaussian core-core
    correction terms and diatomic alpb/xfac pairs for the MOPAC PM6
    core-core damping formula.

    Returns a :class:`PM6ParameterSet` with full coverage.
    """
    if force_refetch or not _CACHE_FILE.exists():
        _fetch_mopac_params()

    with open(_CACHE_FILE, "rb") as f:
        data = tomllib.load(f)

    params = PM6ParameterSet()

    # Load elements
    for elem_blob in data.get("element", []):
        Z = int(elem_blob["Z"])
        if not 1 <= Z <= 98:
            continue
        ed = NDDOElementData()
        ed.Z = Z
        ed.uss = float(elem_blob.get("uss", 0.0))
        ed.upp = float(elem_blob.get("upp", 0.0))
        ed.udd = float(elem_blob.get("udd", 0.0))
        ed.betas = float(elem_blob.get("betas", 0.0))
        ed.betap = float(elem_blob.get("betap", 0.0))
        ed.betad = float(elem_blob.get("betad", 0.0))
        ed.zs = float(elem_blob.get("zs", 1.0))
        ed.zp = float(elem_blob.get("zp", 1.0))
        ed.zd = float(elem_blob.get("zd", 0.0))
        ed.zsn = float(elem_blob.get("zsn", 0.0))
        ed.zpn = float(elem_blob.get("zpn", 0.0))
        ed.zdn = float(elem_blob.get("zdn", 0.0))
        ed.gss = float(elem_blob.get("gss", 0.0))
        ed.gpp = float(elem_blob.get("gpp", 0.0))
        ed.gsp = float(elem_blob.get("gsp", 0.0))
        ed.gp2 = float(elem_blob.get("gp2", 0.0))
        ed.hsp = float(elem_blob.get("hsp", 0.0))
        ed.alpha = float(elem_blob.get("alpha", 1.0))
        ed.polvo = float(elem_blob.get("polvo", 0.0))
        ed.pcore = float(elem_blob.get("poc", 0.0))
        ed.has_d = bool(
            elem_blob.get("has_d", ed.udd != 0.0 or ed.betad != 0.0)
        )
        has_p = ed.upp != 0.0 or ed.gpp != 0.0
        default_n_orbitals = 9 if ed.has_d else (4 if has_p else 1)
        ed.n_orbitals = int(
            elem_blob.get("n_orbitals", default_n_orbitals)
        )
        for gt in elem_blob.get("gaussian_terms", []):
            ed.add_gaussian_term(
                float(gt.get("coeff", 0.0)),
                float(gt.get("exponent", 0.0)),
                float(gt.get("factor", 0.0)),
            )
        # Load STO-6G-derived multipole gamma terms
        for gt in elem_blob.get("gamma_terms", []):
            ed.add_gamma_term(
                float(gt.get("coeff", 0.0)),
                float(gt.get("exponent", 0.0)),
                float(gt.get("factor", 0.0)),
            )
        params.add_element(ed)

    # Load diatomic pairs
    alpb_map: dict[tuple[int, int], float] = {}
    xfac_map: dict[tuple[int, int], float] = {}
    for dp_blob in data.get("diatomic_pair", []):
        kind = dp_blob.get("kind", "")
        z1 = int(dp_blob.get("Z1", 0))
        z2 = int(dp_blob.get("Z2", 0))
        if not (1 <= z1 <= 98 and 1 <= z2 <= 98):
            continue
        val = float(dp_blob.get("value", 0.0))
        if kind == "alpb":
            alpb_map[(z1, z2)] = val
        elif kind == "xfac":
            xfac_map[(z1, z2)] = val

    for (z1, z2), alpb_val in alpb_map.items():
        xfac_val = xfac_map.get((z1, z2), 0.0)
        params.add_diatomic(z1, z2, alpb_val, xfac_val)

    return params
