#!/usr/bin/env python3
"""Generate MOPAC PM6 parameter cache with STO-6G-derived multipole gamma terms."""

import math
import re
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openmopac/mopac/main/src/models/parameters_for_PM6_C.F90"
CACHE = Path.home() / ".cache/vibeqc/mopac_pm6_params.toml"

# STO-6G tables from Hehre, Stewart & Pople (1969)
STO6G = {
    0: {
        "alphas": [0.168856, 0.623913, 3.42525, 8.420, 18.0, 35.5232],
        "coeffs": [0.334041, 0.534205, 0.154815, 0.0203930, 0.00198418, 0.000102952],
    },
    1: {
        "alphas": [0.181494, 0.663712, 3.62252, 8.404, 18.0, 35.5232],
        "coeffs": [0.137008, 0.504421, 0.306884, 0.0607796, 0.00650756, 0.000329759],
    },
}


def derive_gamma_terms(zeta, l, target_gamma_ha):
    if zeta <= 0 or target_gamma_ha <= 0:
        return []
    tbl = STO6G[l]
    raw = []
    for alpha_sto, c_sto in zip(tbl["alphas"], tbl["coeffs"]):
        alpha_dens = 4.0 * alpha_sto * zeta * zeta
        d = math.sqrt(math.pi / (4.0 * alpha_dens))
        raw.append((c_sto, d))
    gamma_0_raw = sum(ci * cj / (di + dj) for ci, di in raw for cj, dj in raw)
    scale = math.sqrt(target_gamma_ha / gamma_0_raw) if gamma_0_raw > 0 else 1.0
    return [{"coeff": ci * scale, "exponent": di, "factor": 0.0} for ci, di in raw]


with urllib.request.urlopen(URL) as r:
    text = r.read().decode("utf-8")

arrays = {}
pat = re.compile(
    r"data\s+(\w+)\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*/\s*([-\d.]+[dD][+\-\d]+)"
)
for m in pat.finditer(text):
    name, idx, jdx, val = (
        m.group(1),
        int(m.group(2)),
        int(m.group(3)) if m.group(3) else 0,
        float(m.group(4).replace("D", "E").replace("d", "E")),
    )
    d = arrays.setdefault(name, {})
    if jdx > 0:
        d.setdefault(idx, {})[jdx] = val
    else:
        d[idx] = val

elements = {}
for m in re.finditer(r"Data for Element\s+(\d+)\s+(\w+)", text):
    atomic_number = int(m.group(1))
    if 1 <= atomic_number <= 98:
        elements[atomic_number] = m.group(2).title()


def _get(arr, Z, default=0.0):
    e = arrays.get(arr, {}).get(Z)
    return (
        e.get(1, default) if isinstance(e, dict) else (e if e is not None else default)
    )


def _get_term(arr, Z, n):
    e = arrays.get(arr, {}).get(Z, {})
    return e.get(n, 0.0) if isinstance(e, dict) else 0.0


# Diatomic pairs
start = text.find("subroutine alpb_and_xfac_pm6")
end = text.find("end subroutine alpb_and_xfac_pm6", start)
block = text[start:end] if end > start else text[start:]
pairs = {}
pat2 = re.compile(r"(alpb|xfac)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([-\d.]+[dD][+\-\d]*)")
for m in pat2.finditer(block):
    key, z1, z2 = m.group(1), int(m.group(2)), int(m.group(3))
    if not (1 <= z1 <= 98 and 1 <= z2 <= 98):
        continue
    val = float(m.group(4).replace("D", "E").replace("d", "E"))
    pairs.setdefault(key, {})[(max(z1, z2), min(z1, z2))] = val

ev2ha = 1.0 / 27.2114


def fv(v):
    if isinstance(v, float):
        return f"{v:.8f}"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    return f'"{v}"'


lines = [
    "# MOPAC PM6 with STO-6G multipole gamma",
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

    gamma_terms = []
    for l, zk, gk in [(0, "zs6", "gss6"), (1, "zp6", "gpp6")]:
        zeta = _get(zk, Z, 0.0)
        g_ev = _get(gk, Z, 0.0)
        if zeta > 0 and g_ev > 0:
            gamma_terms.extend(derive_gamma_terms(zeta, l, g_ev * ev2ha))

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

    lines.append("gamma_terms = [")
    for t in gamma_terms:
        lines.append(
            f"  {{ coeff = {t['coeff']:.12f}, exponent = {t['exponent']:.12f}, factor = 0.0 }},"
        )
    lines.append("]")

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

CACHE.parent.mkdir(parents=True, exist_ok=True)
CACHE.write_text("\n".join(lines))
print(f"Wrote {len(elements)} elements with gamma_terms to {CACHE}")
