"""PM7 parameter loading — MOPAC-derived PM7 parameter set.

Parameters from the open-source MOPAC PM7 parameter file
(``src/models/parameters_for_PM7_C.F90``).

Redistribution terms (CLAUDE.md § 1, issue #440), same as PM6: these
numbers are **copied from MOPAC**, not transcribed from the published
Stewart tables. All 95 inline values reproduce that file exactly at its six
published decimals: the 65 element values (53 scalar parameters plus the 12
Gaussian-term coefficients of H/C/N/O) from its ``data`` statements and the
30 diatomic ``alpb``/``xfac`` values from its ``alpb_and_xfac_pm7``
subroutine. The pair table shipped before 2026-09-06 (introduced in
``d333f5adc``) carried 28 values of unrecorded origin that matched neither
MOPAC nor the bundled ``pm7_mopac_params.toml`` cache; the maintainer ruled
on #440 that they do not ship, and they were replaced by MOPAC's values.
The notice below is the Apache-2.0 § 4(c)/(d) attribution that travels
with them:

  Molecular Orbital PACkage (MOPAC)
  Copyright 2021 Virginia Polytechnic Institute and State University
  Licensed under the Apache License, Version 2.0
  https://github.com/openmopac/mopac

Values in eV (converted to Ha at point of use in the C++ Fock builder).

Method reference:
  J. J. P. Stewart, J. Mol. Model. 19, 1-32 (2013),
  doi:10.1007/s00894-012-1667-x.

Source-of-the-values reference:
  J. E. Moussa and J. J. P. Stewart, J. Open Source Softw. 2026, 11, 8025,
  doi:10.21105/joss.08025.
"""

from __future__ import annotations

from vibeqc._vibeqc_core.semiempirical.nddo import NDDOElementData, PM6ParameterSet

# PM7 parameters for H, C, N, O, F from MOPAC (parameters_for_PM7_C.F90).
# These are the most common elements; for others use load_pm7_mopac_params().
_PM7_MOPAC_CORE = {
    1: {"uss": -11.070112, "betas": -8.389745, "zs": 1.260237,
        "gss": 14.149656, "polvo": 0.229769,
        "coeff": [0.177854], "exponent": [1.428710], "factor": [0.991324]},
    6: {"uss": -51.372620, "upp": -40.135421, "betas": -14.414930,
        "betap": -7.893717, "zs": 1.942244, "zp": 1.708723,
        "gss": 12.347323, "gpp": 10.452226, "gsp": 11.932801,
        "gp2": 9.385498, "hsp": 0.802634, "polvo": 0.935524,
        "coeff": [0.045888], "exponent": [5.037055], "factor": [1.588715]},
    7: {"uss": -61.623260, "upp": -48.949816, "betas": -22.110958,
        "betap": -15.464433, "zs": 2.354344, "zp": 2.028288,
        "gss": 11.881002, "gpp": 12.210140, "gsp": 9.572117,
        "gp2": 10.299947, "hsp": 2.977433, "polvo": 0.551046,
        "coeff": [0.015143], "exponent": [4.734148], "factor": [1.516714]},
    8: {"uss": -96.087736, "upp": -71.087738, "betas": -67.776339,
        "betap": -20.267981, "zs": 5.972309, "zp": 2.349017,
        "gss": 14.955121, "gpp": 12.403337, "gsp": 16.088521,
        "gp2": 10.499706, "hsp": 5.028656, "polvo": 0.312052,
        "coeff": [-0.016243], "exponent": [1.871970], "factor": [1.844360]},
    9: {"uss": -137.395658, "upp": -98.051239, "betas": -69.725688,
        "betap": -29.746124, "zs": 6.070030, "zp": 2.930631,
        "gss": 13.745134, "gpp": 9.188912, "gsp": 17.991219,
        "gp2": 12.322503, "hsp": 2.906187, "polvo": 0.121273},
}

# PM7 diatomic pair params: (Z1,Z2) -> (alpb, xfac), copied from MOPAC's
# ``alpb_and_xfac_pm7`` subroutine in parameters_for_PM7_C.F90 (#440). The
# same 15 pairs appear in the bundled pm7_mopac_params.toml cache;
# tests/test_nddo_parameter_identity.py pins the two tables against each
# other, so neither can drift from the other unnoticed.
_PM7_MOPAC_DIATOMIC = {
    (1,1): (4.051163, 2.845627), (1,6): (1.038716, 0.204582),
    (1,7): (1.049958, 0.173966), (1,8): (1.508030, 0.160602),
    (1,9): (3.104012, 0.587775), (6,6): (2.655746, 0.937818),
    (6,7): (2.734636, 0.959163), (6,8): (2.850931, 0.826848),
    (6,9): (3.230134, 0.946026), (7,7): (2.786676, 0.863430),
    (7,8): (2.841313, 0.699190), (7,9): (3.207271, 0.894011),
    (8,8): (2.457135, 0.356439), (8,9): (3.140885, 0.698033),
    (9,9): (4.492940, 3.111004),
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


def load_pm7_params() -> PM6ParameterSet:
    """Load PM7 parameters for H, C, N, O, F from MOPAC.

    Uses inline MOPAC parameter values with diatomic pair data.
    For elements beyond F use load_pm7_mopac_params().
    """
    params = PM6ParameterSet("pm7")
    for Z, d in _PM7_MOPAC_CORE.items():
        ed = NDDOElementData()
        ed.Z = Z
        _populate_element(ed, d)
        params.add_element(ed)
    for (z1, z2), (alpb, xfac) in _PM7_MOPAC_DIATOMIC.items():
        params.add_diatomic(z1, z2, alpb, xfac)
    return params


def load_pm7_params_auto(zs=None):
    """Load MOPAC PM7 parameters. Falls back to full TOML if any element
    is outside H/C/N/O/F.

    If ``zs`` is None, returns the 5-element inline set.
    """
    if zs is None:
        return load_pm7_params()
    core = {1, 6, 7, 8, 9}
    if all(z in core for z in zs):
        return load_pm7_params()
    return load_pm7_mopac_params()


def load_pm7_mopac_params():
    """Load PM7 parameters from the full MOPAC TOML cache (75 elements,
    1028 diatomic pairs).

    MOPAC's internal pseudo-atom records (Z=100--106) are not chemical
    elements and are deliberately excluded from the executable registry.
    """
    import tomllib
    from pathlib import Path

    toml_path = Path(__file__).parent / "pm7_mopac_params.toml"
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    params = PM6ParameterSet("pm7")

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
