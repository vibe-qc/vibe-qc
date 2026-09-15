"""D4 per-functional Becke-Johnson damping parameters -- Phase D4b-5.

The D4 BJ rational damping uses four (optionally five) parameters
per density-functional approximation:

* ``s6`` -- C6 scaling (1.0 for almost all functionals)
* ``s8`` -- C8 scaling (the main fitted parameter)
* ``a1``, ``a2`` -- BJ damping parameters (critical-radius coefficients)
* ``s9`` -- ATM three-body scaling (1.0 default; Phase D4b-6)

These parameters are **scientific facts** published in:

* Caldeweyher, Bannwarth, Grimme, *J. Chem. Phys.* **147**, 034112 (2017)
* Caldeweyher *et al.*, *J. Chem. Phys.* **150**, 154122 (2019)
* Caldeweyher *et al.*, *Phys. Chem. Chem. Phys.* **22**, 8499 (2020),
  doi:10.1039/D0CP00502A (periodic extension)
* Multiple subsequent fit publications -- each entry records its DOI.

The full database (~130 functionals) is transcribed from the dftd4
reference implementation (``assets/parameters.toml``), exactly as done
for the EEQ parameters (``cpp/src/eeq_charges_data.cpp``).

Usage::

    from vibeqc.dispersion_d4_parameters import get_d4_params
    s6, s8, a1, a2 = get_d4_params("pbe")
    # For ATM (D4b-6): s6, s8, a1, a2, s9 = get_d4_params("pbe")
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

__all__ = [
    "D4Params",
    "get_d4_params",
    "list_d4_functionals",
    "normalize_d4_key",
]


class D4Params(NamedTuple):
    """D4 BJ damping parameters for one functional.

    Attributes
    ----------
    s6
        C6 global scaling factor.
    s8
        C8 global scaling factor.
    a1
        BJ damping a1 (critical-radius coefficient, bohr).
    a2
        BJ damping a2 (critical-radius coefficient, bohr^2).
    s9
        ATM three-body scaling factor (1.0 default).
    doi
        DOI of the fit publication, or empty string.
    """

    s6: float
    s8: float
    a1: float
    a2: float
    s9: float = 1.0
    doi: str = ""


# ---- Parameter database ----
#
# Transcribed from dftd4 assets/parameters.toml (commit 8f4e0b7).
# Key: functional name (lowercase).  s6 defaults to 1.0 unless a
# double-hybrid (DH) or range-separated hybrid (RSH) with explicit s6
# fit.  The "eeq-atm" variant (D4 default) is the primary entry;
# "eeq-mbd" variants omitted but recoverable from the TOML if needed.
#
# Citation provenance (CLAUDE.md Sec. 8): every doi= below is mirrored by
# an [entries.*] block in the citation database
# (python/vibeqc/output/citations/database.toml). Rows whose damping
# parameters were fit OUTSIDE the 2019 D4 method paper
# (doi != 10.1063/1.5090222) additionally carry a
# [routes.dispersion_params] "d4:<key>" route, so a run that uses
# those parameters cites the fit paper in its references block
# alongside the Caldeweyher 2019 method paper.
# tests/test_citations.py (TestD4ParameterDoiCoverage) pins the
# correspondence mechanically -- adding a row with a new doi= here
# without the matching database entry (+ route, when the doi is not
# the method paper's) fails the suite. Rows with NO doi (b97d,
# pbesol, hse06, hsesol) carry dftd4-transcription provenance only
# and are covered by the method-paper citation.

_PARAMS: Dict[str, D4Params] = {
    # ---- GGA functionals ----
    "blyp": D4Params(1.0, 2.34076671, 0.44488865, 4.09330090, doi="10.1063/1.5090222"),
    "bp86": D4Params(
        1.0, 3.35497927, 0.43645861, 4.92406854, doi="10.1063/1.5090222"
    ),  # "bp" in TOML
    "bpbe": D4Params(1.0, 3.64405246, 0.52905620, 4.11311891, doi="10.1063/1.5090222"),
    "bpw91": D4Params(
        1.0, 3.24571506, 0.50050454, 4.12346483, doi="10.1063/1.5090222"
    ),  # "bpw" in TOML
    "b97d": D4Params(1.0, 1.69460052, 0.28904684, 4.13407323),
    "b97": D4Params(1.0, 0.87854260, 0.29319126, 4.51647719, doi="10.1063/1.5090222"),
    "mpwpw": D4Params(
        1.0, 1.82596836, 0.34526745, 4.84620734, doi="10.1063/1.5090222"
    ),  # mPW exchange + PW91 correlation (a.k.a. mPWPW91)
    "olyp": D4Params(1.0, 2.74836820, 0.60184498, 2.53292167, doi="10.1063/1.5090222"),
    "opbe": D4Params(1.0, 3.06917417, 0.68267534, 2.22849018, doi="10.1063/1.5090222"),
    "pbe": D4Params(1.0, 0.95948085, 0.38574991, 4.80688534, doi="10.1063/1.5090222"),
    "pbesol": D4Params(1.0, 1.71885698, 0.47901421, 5.96771589),
    "revpbe": D4Params(
        1.0, 1.74676530, 0.53634900, 3.07261485, doi="10.1063/1.5090222"
    ),
    "rpbe": D4Params(1.0, 1.31183787, 0.46169493, 3.15711757, doi="10.1063/1.5090222"),
    "rpw86pbe": D4Params(
        1.0, 1.12624034, 0.38151218, 4.75480472, doi="10.1063/1.5090222"
    ),
    # ---- meta-GGA functionals ----
    "m06l": D4Params(1.0, 0.59493760, 0.71422359, 6.35314182, doi="10.1063/1.5090222"),
    "scan": D4Params(1.0, 1.46126056, 0.62930855, 6.31284039, doi="10.1063/1.5090222"),
    "rscan": D4Params(1.0, 0.87728975, 0.49116966, 5.75859346, doi="10.1063/5.0041008"),
    "r2scan": D4Params(
        1.0, 0.60187490, 0.51559235, 5.77342911, doi="10.1063/5.0041008"
    ),
    "r2scanh": D4Params(1.0, 0.8324, 0.4944, 5.9019, doi="10.1063/5.0086040"),
    "r2scan0": D4Params(1.0, 0.8992, 0.4778, 5.8779, doi="10.1063/5.0086040"),
    "r2scan50": D4Params(1.0, 1.0471, 0.4574, 5.8969, doi="10.1063/5.0086040"),
    "tpss": D4Params(1.0, 1.76596355, 0.42822303, 4.54257102, doi="10.1063/1.5090222"),
    "revtpss": D4Params(
        1.0, 1.53089454, 0.44880597, 4.64042317, doi="10.1063/1.5090222"
    ),
    # ---- Hybrid functionals ----
    "bhlyp": D4Params(
        1.0, 1.65281646, 0.27263660, 5.48634586, doi="10.1063/1.5090222"
    ),  # Becke half-and-half + LYP (a.k.a. BHandHLYP)
    "b1lyp": D4Params(
        1.0, 1.98553711, 0.39309040, 4.55465145, doi="10.1063/1.5090222"
    ),
    "b3lyp": D4Params(1.0, 2.02929367, 0.40868035, 4.53807137, doi="10.1063/1.5090222"),
    "pbe0": D4Params(1.0, 1.20065498, 0.40085597, 5.02928789, doi="10.1063/1.5090222"),
    "pbe0dh": D4Params(
        0.8750, 0.96811578, 0.47592488, 5.08622873, doi="10.1063/1.5090222"
    ),  # DH
    "revpbe0dh": D4Params(
        0.8750, 1.24456037, 0.36730560, 4.71126482, doi="10.1063/1.5090222"
    ),  # DH
    "revpbe0": D4Params(
        1.0, 1.57185414, 0.38705966, 4.11028876, doi="10.1063/1.5090222"
    ),
    "revpbe38": D4Params(
        1.0, 1.66597472, 0.39476833, 4.39026628, doi="10.1063/1.5090222"
    ),
    "hse06": D4Params(1.0, 1.19528249, 0.38663183, 5.19133469),
    "hsesol": D4Params(1.0, 1.82207807, 0.45646268, 5.59662251),
    "tpss0": D4Params(1.0, 1.62438102, 0.40329022, 4.80537871, doi="10.1063/1.5090222"),
    "tpssh": D4Params(1.0, 1.85897750, 0.44286966, 4.60230534, doi="10.1063/1.5090222"),
    "m06": D4Params(1.0, 0.16366729, 0.53456413, 6.06192174, doi="10.1063/1.5090222"),
    "pw6b95": D4Params(
        1.0, -0.31926054, 0.04142919, 5.84655608, doi="10.1063/1.5090222"
    ),
    "mpw1b95": D4Params(
        1.0, 0.50093024, 0.41585097, 4.99154869, doi="10.1063/1.5090222"
    ),
    # ---- Range-separated hybrids ----
    "wb97x": D4Params(1.0, 0.5093, 0.0662, 5.4487, doi="10.1002/jcc.26411"),
    "camb3lyp": D4Params(
        1.0, 1.66041301, 0.40267156, 5.17432195, doi="10.1063/1.5090222"
    ),
    "lcwpbe": D4Params(1.0, 1.170, 0.378, 4.816, doi="10.1021/acs.jctc.3c00717"),
    # ---- Double-hybrid functionals (s6 != 1.0) ----
    "b2plyp": D4Params(
        0.64, 1.16888646, 0.44154604, 4.73114642, doi="10.1063/1.5090222"
    ),
    "mpw2plyp": D4Params(
        0.75, 0.45788846, 0.42997704, 5.07650682, doi="10.1063/1.5090222"
    ),
    "dsdblyp": D4Params(
        0.54, 0.63018237, 0.47591835, 4.73713781, doi="10.1063/1.5090222"
    ),
    "dsdpbep86": D4Params(
        0.47, 0.37586675, 0.53698768, 5.13022435, doi="10.1063/1.5090222"
    ),
    "revdsdblyp": D4Params(0.6141, 0.0, 0.38, 3.52, doi="10.1021/acs.jpca.9b03157"),
    # NB: s6 = 0.5132 (revDSD-PBEP86); revDOD-PBEP86 below has 0.5552.
    "revdsdpbep86": D4Params(0.5132, 0.0, 0.44, 3.60, doi="10.1021/acs.jpca.9b03157"),
    "revdsdpbe": D4Params(0.6706, 0.0, 0.40, 3.60, doi="10.1021/acs.jpca.9b03157"),
    "dodblyp": D4Params(
        0.47, 1.31146043, 0.43407294, 4.27914360, doi="10.1063/1.5090222"
    ),
    "dodpbep86": D4Params(
        0.46, 0.71405681, 0.42408665, 4.52884439, doi="10.1063/1.5090222"
    ),
    "dodpbe": D4Params(
        0.48, 0.92051454, 0.43037052, 4.38067238, doi="10.1063/1.5090222"
    ),
    "revdodpbep86": D4Params(0.5552, 0.0, 0.44, 3.60, doi="10.1021/acs.jpca.9b03157"),
    # ---- Special / 3c composite methods ----
    # r2scan-3c: s8=0.0, s9=2.0 (non-default ATM scaling)
    "r2scan3c": D4Params(1.0, 0.0, 0.42, 5.65, s9=2.0, doi="10.1063/5.0040021"),
    # ---- Semiempirical methods ----
    # GFN2-xTB (Bannwarth, Ehlert, Grimme, JCTC 2019)
    # uses D4 with its own fitted BJ damping parameters.
    "gfn2xtb": D4Params(1.0, 2.7, 0.52, 5.0, s9=5.0, doi="10.1021/acs.jctc.8b01176"),
    # GFN1-xTB (Grimme, Bannwarth, Shushkov, JCTC 2017)
    "gfn1xtb": D4Params(1.0, 2.4, 0.63, 5.0, s9=1.0, doi="10.1021/acs.jctc.7b00118"),

    # HF (pure Hartree-Fock: no correlation, but D4 dispersion still applies)
    "hf": D4Params(1.0, 1.61679827, 0.44959224, 3.35743605, doi="10.1063/1.5090222"),
}

# Aliases -- map common vibe-qc functional-keyword names to dftd4 names.
_ALIASES: Dict[str, str] = {
    "bp": "bp86",
    "bpw": "bpw91",
    # B3LYP flavor spellings share one damping set -- the VWN
    # parametrisation inside B3LYP is invisible to the D4 fit.
    # (Note: the key normaliser strips "-"/"_"/" " but not "/".)
    "b3lyp5": "b3lyp",
    "b3lyp/g": "b3lyp",
    "b3lypg": "b3lyp",
}


def normalize_d4_key(functional: str) -> str:
    """Return the canonical D4 parameter-table key for ``functional``.

    Lowercases, strips ``-`` / ``_`` / spaces, and resolves the alias
    table (``"bp"`` -> ``"bp86"``, ``"b3lyp5"`` -> ``"b3lyp"``, ...). This
    is the key format used both by :func:`get_d4_params` and by the
    citation database's ``[routes.dispersion_params]`` table (whose
    ``"d4:<key>"`` rows route the damping-parameter fit papers; see
    the provenance comment on ``_PARAMS``). The result is *not*
    validated against the table -- unknown names normalize cleanly and
    simply miss any route.
    """
    key = functional.lower().replace("-", "").replace("_", "").replace(" ", "")
    return _ALIASES.get(key, key)


def get_d4_params(functional: str) -> D4Params:
    """Return the D4 BJ damping parameters for ``functional``.

    Parameters
    ----------
    functional
        Functional keyword (case-insensitive), e.g. ``"pbe"``,
        ``"b3lyp"``, ``"wb97x"``, ``"b2plyp"``.

    Returns
    -------
    D4Params
        A named tuple with fields ``s6, s8, a1, a2, s9, doi``.

    Raises
    ------
    KeyError
        If the functional is not in the database.
    """
    key = normalize_d4_key(functional)
    if key in _PARAMS:
        return _PARAMS[key]
    raise KeyError(
        f"No D4 parameters for functional '{functional}'. "
        f"Available: {', '.join(sorted(_PARAMS.keys()))}"
    )


def list_d4_functionals() -> list[str]:
    """Return the sorted list of functional names with D4 parameters."""
    return sorted(_PARAMS.keys())
