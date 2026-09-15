"""PM6 parameter loading — MOPAC-derived PM6 parameter set.

Parameters from the open-source MOPAC PM6 parameter file
(``src/models/parameters_for_PM6_C.F90``).

Redistribution terms (CLAUDE.md § 1, issue #440). These numbers are
**copied from MOPAC**, not transcribed from the published Stewart tables:
all 118 values below (88 element + 30 diatomic) reproduce that file
exactly at its six published decimals, and none of them occurs anywhere in
Stewart 2007. They are redistributed under MOPAC's license, and the notice
below is the Apache-2.0 § 4(c)/(d) attribution that travels with them:

  Molecular Orbital PACkage (MOPAC)
  Copyright 2021 Virginia Polytechnic Institute and State University
  Licensed under the Apache License, Version 2.0
  https://github.com/openmopac/mopac

Values in eV (converted to Ha at point of use in the C++ Fock builder).

Method reference — what the numbers parameterize, as distinct from where
this copy of them came from:
  J. J. P. Stewart, J. Mol. Model. 2007, 13, 1173-1213,
  doi:10.1007/s00894-007-0233-4.

Source-of-the-values reference:
  J. E. Moussa and J. J. P. Stewart, J. Open Source Softw. 2026, 11, 8025,
  doi:10.21105/joss.08025.
"""

from __future__ import annotations

from vibeqc._vibeqc_core.semiempirical.nddo import NDDOElementData, PM6ParameterSet

# MOPAC PM6 parameters for H, C, N, O, F.
# Source: MOPAC parameters_for_PM6_C.F90, lines 26-184.
_PM6_MOPAC_CORE = {
    1: {"uss": -11.246958, "betas": -8.352984, "zs": 1.268641,
        "gss": 14.448686, "polvo": 0.262114,
        "cpe_zet": 1.342400, "cpe_z0": 2.255100, "cpe_b": 0.856600,
        "cpe_xlo": 0.379600, "cpe_xhi": 1.379600,
        "coeff": [0.024184], "exponent": [3.055953], "factor": [1.786011]},
    6: {"uss": -51.089653, "upp": -39.937920, "betas": -15.385236,
        "betap": -7.471929, "zs": 2.047558, "zp": 1.702841,
        "gss": 13.335519, "gpp": 10.778326, "gsp": 11.528134,
        "gp2": 9.486212, "hsp": 0.717322, "polvo": 0.485071,
        "cpe_zet": 1.167040, "cpe_z0": 1.178300, "cpe_b": 0.004800,
        "cpe_xlo": 1.086200, "cpe_xhi": 2.353000,
        "coeff": [0.046302], "exponent": [2.100206], "factor": [1.333959]},
    7: {"uss": -57.784823, "upp": -49.893036, "betas": -17.979377,
        "betap": -15.055017, "zs": 2.380406, "zp": 1.999246,
        "gss": 12.357026, "gpp": 12.570756, "gsp": 9.636190,
        "gp2": 10.576425, "hsp": 2.871545, "polvo": 0.204743,
        "cpe_zet": 1.378880, "cpe_z0": 2.029200, "cpe_b": 0.323800,
        "cpe_xlo": 1.651100, "cpe_xhi": 2.292100,
        "coeff": [-0.001436], "exponent": [0.495196], "factor": [1.704857]},
    8: {"uss": -91.678761, "upp": -70.460949, "betas": -65.635137,
        "betap": -21.622604, "zs": 5.421751, "zp": 2.270960,
        "gss": 11.304042, "gpp": 13.618205, "gsp": 15.807424,
        "gp2": 10.332765, "hsp": 5.010801, "polvo": 0.154301,
        "cpe_zet": 1.585280, "cpe_z0": 4.322700, "cpe_b": 0.045100,
        "cpe_xlo": 3.483200, "cpe_xhi": 3.605000,
        "coeff": [-0.017771], "exponent": [3.05831], "factor": [1.896435]},
    9: {"uss": -140.225626, "upp": -98.778044, "betas": -69.922593,
        "betap": -30.448165, "zs": 6.043849, "zp": 2.906722,
        "gss": 12.446818, "gpp": 8.417366, "gsp": 18.496082,
        "gp2": 12.179816, "hsp": 2.604382, "polvo": 0.199611,
        "coeff": [-0.010792], "exponent": [6.004648], "factor": [1.847724]},
}

# Diatomic pair params: (Z1,Z2) -> (alpb, xfac)
_PM6_MOPAC_DIATOMIC = {
    (1,1): (3.540942, 2.243587), (1,6): (1.027806, 0.216506),
    (1,7): (0.969406, 0.175506), (1,8): (1.260942, 0.192295),
    (1,9): (3.136740, 0.815802), (6,6): (2.613713, 0.813510),
    (6,7): (2.686108, 0.859949), (6,8): (2.889607, 0.990211),
    (6,9): (3.027600, 0.732968), (7,7): (2.574502, 0.675313),
    (7,8): (2.784292, 0.764756), (7,9): (2.856646, 0.635854),
    (8,8): (2.623998, 0.535112), (8,9): (3.015444, 0.674251),
    (9,9): (3.175759, 0.681343),
}


def _populate_element(ed: NDDOElementData, d: dict) -> None:
    """Fill an NDDOElementData from a dict of MOPAC parameter values."""
    ed.Z = int(d.get("Z", ed.Z))
    ed.uss = float(d.get("uss", 0))
    ed.upp = float(d.get("upp", 0))
    ed.udd = float(d.get("udd", 0))
    ed.betas = float(d.get("betas", 0))
    ed.betap = float(d.get("betap", 0))
    ed.betad = float(d.get("betad", 0))
    ed.zs = float(d.get("zs", 0))
    ed.zp = float(d.get("zp", 0))
    ed.zd = float(d.get("zd", 0))
    ed.zsn = float(d.get("zsn", 0))
    ed.zpn = float(d.get("zpn", 0))
    ed.zdn = float(d.get("zdn", 0))
    ed.gss = float(d.get("gss", 0))
    ed.gpp = float(d.get("gpp", 0))
    ed.gsp = float(d.get("gsp", 0))
    ed.gp2 = float(d.get("gp2", 0))
    ed.hsp = float(d.get("hsp", 0))
    ed.f0sd = float(d.get("f0sd", 0))
    ed.g2sd = float(d.get("g2sd", 0))
    ed.alpha = float(d.get("alp", 0))
    ed.pcore = float(d.get("poc", 0))
    ed.polvo = float(d.get("polvo", 0))
    ed.cpe_zet = float(d.get("cpe_zet", 0))
    ed.cpe_z0 = float(d.get("cpe_z0", 0))
    ed.cpe_b = float(d.get("cpe_b", 0))
    ed.cpe_xlo = float(d.get("cpe_xlo", 0))
    ed.cpe_xhi = float(d.get("cpe_xhi", 0))
    ed.has_cpe = (ed.cpe_zet > 0 or ed.cpe_z0 > 0)
    ed.has_d = (ed.betad != 0 or ed.udd != 0)
    if ed.has_d:
        ed.n_orbitals = 9
    coeffs = d.get("coeff", [])
    expons = d.get("exponent", [])
    factors = d.get("factor", [])
    for i in range(min(len(coeffs), len(expons), len(factors))):
        c, e, f = float(coeffs[i]), float(expons[i]), float(factors[i])
        if c != 0 or e > 1e-12:
            ed.add_gaussian_term(c, e, f)


def load_pm6_params() -> PM6ParameterSet:
    """Load PM6 parameters for H, C, N, O, F from MOPAC.

    Uses inline MOPAC parameter values with diatomic pair data.
    For elements beyond F use load_pm6_mopac_params().
    """
    params = PM6ParameterSet()
    for Z, d in _PM6_MOPAC_CORE.items():
        ed = NDDOElementData()
        ed.Z = Z
        _populate_element(ed, d)
        params.add_element(ed)
    for (z1, z2), (alpb, xfac) in _PM6_MOPAC_DIATOMIC.items():
        params.add_diatomic(z1, z2, alpb, xfac)
    return params


def load_pm6_params_auto(zs=None):
    """Load MOPAC PM6 parameters. Falls back to full TOML if any element
    is outside H/C/N/O/F.

    If ``zs`` is None, returns the 5-element inline set.
    """
    if zs is None:
        return load_pm6_params()
    core = {1, 6, 7, 8, 9}
    if all(z in core for z in zs):
        return load_pm6_params()
    return load_pm6_mopac_params()


def load_pm6_mopac_params():
    """Load PM6 parameters from the full MOPAC TOML cache (75 elements,
    933 diatomic pairs).

    MOPAC's internal pseudo-atom records (Z=100--106) are not chemical
    elements and are deliberately excluded from the executable registry.
    """
    import tomllib
    from pathlib import Path

    toml_path = Path(__file__).parent / "pm6_mopac_params.toml"
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    params = PM6ParameterSet()

    for elem in data.get("element", []):
        if not 1 <= int(elem["Z"]) <= 98:
            continue
        ed = NDDOElementData()
        ed.Z = elem["Z"]
        _populate_element(ed, elem)
        params.add_element(ed)

    for dp in data.get("diatomic", []):
        if not (
            1 <= int(dp["Z1"]) <= 98 and 1 <= int(dp["Z2"]) <= 98
        ):
            continue
        params.add_diatomic(
            int(dp["Z1"]), int(dp["Z2"]),
            float(dp.get("alpb", 0)), float(dp.get("xfac", 0)),
        )

    return params
