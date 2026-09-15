"""OMx parameter loading -- OM1, OM2, OM3.

The OMx methods (Orthogonalization Models) from the Thiel group extend
the NDDO formalism with a Loewdin orthogonalization correction.
They use the same NDDOElementData structure as PM6.

Parameter sources (all three tables from the same open-access paper):
  Dral, P. O.; Wu, X.; Spörkel, L.; Koslowski, A.; Weber, W.;
  Steiger, R.; Scholten, M.; Thiel, W.
  J. Chem. Theory Comput. 2016, 12, 1082-1096.
  DOI: 10.1021/acs.jctc.5b01046

  OM1: Table 1;  original ref: Kolb & Thiel, J. Comput. Chem. 1993, 14, 775.
  OM2: Table 2 (the first full publication of the OM2 parameter values);
       method ref: Weber & Thiel, Theor. Chem. Acc. 2000, 103, 495, whose
       own Table 2 is an orthogonalization-energy table for H3-, not a
       parameter table (#272, maintainer wording 2026-09-06).
  OM3: Table 3;  original ref: Scholten, PhD thesis, Univ. Duesseldorf, 2003.
"""

from __future__ import annotations

from vibeqc._vibeqc_core.semiempirical.nddo import (
    OMxElementData,
    OMxParameterSet,
    OMxVariant,
)

# Units:  USS/UPP/GSS/GPP/GSP/GP2/HSP in eV
#         ZS/ZP in a0^-1
#         ζ (orbital scale factor) in a0^-1
#         bs/bp/bpi in eV.bohr^{-1/2}
#         as/ap/api in bohr^{-2}
#         F1/F2/G1/G2 dimensionless
#         aa in bohr^{-2}, ba in eV.bohr^{-1/2}, Faa in eV
#         a (core-core damping) dimensionless

# ---------------------------------------------------------------------------
# OM1 parameters -- Dral 2016 Table 1 (Kolb & Thiel 1993, Table 2)
# ---------------------------------------------------------------------------

_OM1_PLACEHOLDER = False  # Published OM1 params loaded


def _make_om1_params() -> dict[int, dict]:
    """Return OM1 parameters for H, C, N, O, F.

    Source: Dral 2016, Table 1 (Kolb & Thiel 1993, Table 2).
    """
    return {
        1: {
            "uss": -12.83851861,
            "upp": 0.0,
            "gss": 13.0106,
            "gpp": 0.0,
            "gsp": 0.0,
            "gp2": 0.0,
            "hsp": 0.0,
            "beta_s": -4.89312435,
            "beta_p": 0.0,
            "beta_pi": 0.0,
            "alpha_s": 0.09653898,
            "alpha_p": 0.0,
            "alpha_pi": 0.0,
            "F1": 0.54128873,
            "F2": 0.84668969,
            "G1": 0.0,
            "G2": 0.0,
            "zeta": 1.20948930,
            "zs": 1.2095,
            "zp": 0.0,
            "alpha": 0.0,
            "n_orbitals": 1,
            "beta_s_xh": 0.0,
            "beta_p_xh": 0.0,
            "alpha_s_xh": 0.0,
            "alpha_p_xh": 0.0,
            "zeta_alpha": 0.0,
            "F_alpha_alpha": 0.0,
            "beta_alpha": 0.0,
            "alpha_alpha": 0.0,
        },
        6: {
            "uss": -50.15945000,
            "upp": -38.76257345,
            "gss": 11.5894,
            "gpp": 10.4261,
            "gsp": 9.6586,
            "gp2": 9.1637,
            "hsp": 1.4332,
            "beta_s": -7.58632270,
            "beta_p": -4.49894163,
            "beta_pi": -5.91210138,
            "alpha_s": 0.09325105,
            "alpha_p": 0.05398748,
            "alpha_pi": 0.10477244,
            "F1": 0.50383851,
            "F2": 0.66944409,
            "G1": 0.0,
            "G2": 0.0,
            "zeta": 1.13551142,
            "zs": 2.0395,
            "zp": 1.9034,
            "alpha": 2.7344,
            "n_orbitals": 4,
            "beta_s_xh": 0.0,
            "beta_p_xh": 0.0,
            "alpha_s_xh": 0.0,
            "alpha_p_xh": 0.0,
            "zeta_alpha": 0.0,
            "F_alpha_alpha": 0.0,
            "beta_alpha": 0.0,
            "alpha_alpha": 0.0,
        },
        7: {
            "uss": -71.33505463,
            "upp": -56.58315267,
            "gss": 12.0186,
            "gpp": 10.9760,
            "gsp": 10.2426,
            "gp2": 9.7244,
            "hsp": 1.7711,
            "beta_s": -12.00586167,
            "beta_p": -9.64950408,
            "beta_pi": -10.16405908,
            "alpha_s": 0.10185884,
            "alpha_p": 0.08540515,
            "alpha_pi": 0.14350678,
            "F1": 0.63476395,
            "F2": 0.31135759,
            "G1": 0.0,
            "G2": 0.0,
            "zeta": 1.16081665,
            "zs": 2.3649,
            "zp": 2.0563,
            "alpha": 3.0784,
            "n_orbitals": 4,
            "beta_s_xh": -8.08332477,
            "beta_p_xh": -11.48923075,
            "alpha_s_xh": 0.07514671,
            "alpha_p_xh": 0.11556228,
            "zeta_alpha": 0.0,
            "F_alpha_alpha": 0.0,
            "beta_alpha": 0.0,
            "alpha_alpha": 0.0,
        },
        8: {
            "uss": -93.04158571,
            "upp": -77.59792792,
            "gss": 14.0967,
            "gpp": 12.8361,
            "gsp": 11.9993,
            "gp2": 11.4586,
            "hsp": 2.8709,
            "beta_s": -6.22223757,
            "beta_p": -9.94028730,
            "beta_pi": -11.29342651,
            "alpha_s": 0.10891616,
            "alpha_p": 0.09666556,
            "alpha_pi": 0.15255321,
            "F1": 0.68193417,
            "F2": 0.47652748,
            "G1": 0.0,
            "G2": 0.0,
            "zeta": 1.10190209,
            "zs": 3.2470,
            "zp": 2.9160,
            "alpha": 3.3391,
            "n_orbitals": 4,
            "beta_s_xh": -6.45960578,
            "beta_p_xh": -12.47451386,
            "alpha_s_xh": 0.07953217,
            "alpha_p_xh": 0.12908958,
            "zeta_alpha": 0.0,
            "F_alpha_alpha": 0.0,
            "beta_alpha": 0.0,
            "alpha_alpha": 0.0,
        },
        9: {
            "uss": -121.69518463,
            "upp": -106.37310634,
            "gss": 15.8661,
            "gpp": 14.4092,
            "gsp": 13.3853,
            "gp2": 12.7133,
            "hsp": 3.0541,
            "beta_s": -5.73558167,
            "beta_p": -16.36168108,
            "beta_pi": -17.22481680,
            "alpha_s": 0.19374572,
            "alpha_p": 0.13034754,
            "alpha_pi": 0.22033729,
            "F1": 1.19938976,
            "F2": 0.49484369,
            "G1": 0.0,
            "G2": 0.0,
            "zeta": 1.16498140,
            "zs": 4.3207,
            "zp": 3.5113,
            "alpha": 3.7696,
            "n_orbitals": 4,
            "beta_s_xh": 0.0,
            "beta_p_xh": 0.0,
            "alpha_s_xh": 0.0,
            "alpha_p_xh": 0.0,
            "zeta_alpha": 0.0,
            "F_alpha_alpha": 0.0,
            "beta_alpha": 0.0,
            "alpha_alpha": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# OM2 parameters -- Dral 2016 Table 2 (method: Weber & Thiel 2000)
# ---------------------------------------------------------------------------

_OM2_PLACEHOLDER = False  # Published OM2 params loaded


def _make_om2_params() -> dict[int, dict]:
    """Return OM2 parameters for H, C, N, O, F.

    Source: Dral 2016, Table 2. Method: Weber & Thiel 2000 (#272).
    """
    return {
        1: {
            "uss": -12.64890000,
            "upp": 0.0,
            "gss": 13.0106,
            "gpp": 0.0,
            "gsp": 0.0,
            "gp2": 0.0,
            "hsp": 0.0,
            "beta_s": -3.41998220,
            "beta_p": 0.0,
            "beta_pi": 0.0,
            "alpha_s": 0.06607903,
            "alpha_p": 0.0,
            "alpha_pi": 0.0,
            "F1": 0.29566861,
            "F2": 1.40190659,
            "G1": 0.65271563,
            "G2": 0.90843670,
            "zeta": 1.47386481,
            "zs": 1.4739,
            "zp": 0.0,
            "alpha": 0.0,
            "n_orbitals": 1,
            "beta_s_xh": 0.0,
            "beta_p_xh": 0.0,
            "alpha_s_xh": 0.0,
            "alpha_p_xh": 0.0,
            "zeta_alpha": 0.0,
            "F_alpha_alpha": 0.0,
            "beta_alpha": 0.0,
            "alpha_alpha": 0.0,
        },
        6: {
            "uss": -51.65550844,
            "upp": -39.74369825,
            "gss": 11.5894,
            "gpp": 10.4261,
            "gsp": 9.6586,
            "gp2": 9.1637,
            "hsp": 1.4332,
            "beta_s": -7.21406021,
            "beta_p": -4.14394503,
            "beta_pi": -5.97107657,
            "alpha_s": 0.09045297,
            "alpha_p": 0.05452192,
            "alpha_pi": 0.10204903,
            "F1": 0.49949211,
            "F2": 0.72261226,
            "G1": 0.21284361,
            "G2": 0.99250289,
            "zeta": 1.42036892,
            "zs": 2.0395,
            "zp": 1.9034,
            "alpha": 2.7344,
            "n_orbitals": 4,
            "beta_s_xh": -6.30164062,
            "beta_p_xh": -4.04444703,
            "alpha_s_xh": 0.09668329,
            "alpha_p_xh": 0.05283694,
            "zeta_alpha": 5.16802668,
            "F_alpha_alpha": -305.68646337,
            "beta_alpha": -9.07185084,
            "alpha_alpha": 0.16985745,
        },
        7: {
            "uss": -74.37638240,
            "upp": -57.60067613,
            "gss": 12.0186,
            "gpp": 10.9760,
            "gsp": 10.2426,
            "gp2": 9.7244,
            "hsp": 1.7711,
            "beta_s": -10.84303446,
            "beta_p": -7.62373736,
            "beta_pi": -9.27936312,
            "alpha_s": 0.08974553,
            "alpha_p": 0.08759680,
            "alpha_pi": 0.13172314,
            "F1": 0.64073384,
            "F2": 0.19580808,
            "G1": 0.13946233,
            "G2": 0.84373060,
            "zeta": 1.33175233,
            "zs": 2.3649,
            "zp": 2.0563,
            "alpha": 3.0784,
            "n_orbitals": 4,
            "beta_s_xh": -9.49567107,
            "beta_p_xh": -8.51180846,
            "alpha_s_xh": 0.11429048,
            "alpha_p_xh": 0.10673732,
            "zeta_alpha": 6.93980600,
            "F_alpha_alpha": -407.39202305,
            "beta_alpha": -9.97910210,
            "alpha_alpha": 0.16173024,
        },
        8: {
            "uss": -101.82723464,
            "upp": -78.92823923,
            "gss": 14.0967,
            "gpp": 12.8361,
            "gsp": 11.9993,
            "gp2": 11.4586,
            "hsp": 2.8709,
            "beta_s": -10.64436974,
            "beta_p": -8.63610952,
            "beta_pi": -9.21201190,
            "alpha_s": 0.13062089,
            "alpha_p": 0.09626876,
            "alpha_pi": 0.13071747,
            "F1": 1.26450169,
            "F2": 1.14847352,
            "G1": 0.28309603,
            "G2": 0.78414131,
            "zeta": 1.55214516,
            "zs": 3.2470,
            "zp": 2.9160,
            "alpha": 3.3391,
            "n_orbitals": 4,
            "beta_s_xh": -6.54238767,
            "beta_p_xh": -10.11307271,
            "alpha_s_xh": 0.11112738,
            "alpha_p_xh": 0.11891861,
            "zeta_alpha": 7.58579774,
            "F_alpha_alpha": -514.45812327,
            "beta_alpha": -14.16551053,
            "alpha_alpha": 0.34390559,
        },
        9: {
            "uss": -120.62785370,
            "upp": -107.27105397,
            "gss": 15.8661,
            "gpp": 14.4092,
            "gsp": 13.3853,
            "gp2": 12.7133,
            "hsp": 3.0541,
            "beta_s": -6.25438426,
            "beta_p": -13.93492471,
            "beta_pi": -18.73205761,
            "alpha_s": 0.26624434,
            "alpha_p": 0.12261412,
            "alpha_pi": 0.21684388,
            "F1": 2.11499396,
            "F2": 1.09156321,
            "G1": 0.31704089,
            "G2": 0.02140504,
            "zeta": 1.45216726,
            "zs": 4.3207,
            "zp": 3.5113,
            "alpha": 3.7696,
            "n_orbitals": 4,
            "beta_s_xh": -6.25104378,
            "beta_p_xh": -13.94492971,
            "alpha_s_xh": 0.44713918,
            "alpha_p_xh": 0.15648906,
            "zeta_alpha": 8.71226515,
            "F_alpha_alpha": -685.41988599,
            "beta_alpha": -9.17960365,
            "alpha_alpha": 0.99971548,
        },
    }


# ---------------------------------------------------------------------------
# OM3 parameters -- Dral 2016 Table 3 (Scholten 2003)
# ---------------------------------------------------------------------------

_OM3_PLACEHOLDER = False  # Published OM3 params loaded


def _make_om3_params() -> dict[int, dict]:
    """Return OM3 parameters for H, C, N, O, F.

    Source: Dral 2016, Table 3 (Scholten 2003).
    OM3: same as OM2 but with F2=0, G2=0 (neglects the second sums
    in the two-center and three-center VORT corrections).
    """
    return {
        1: {
            "uss": -12.45828647,
            "upp": 0.0,
            "gss": 13.0106,
            "gpp": 0.0,
            "gsp": 0.0,
            "gp2": 0.0,
            "hsp": 0.0,
            "beta_s": -3.40064659,
            "beta_p": 0.0,
            "beta_pi": 0.0,
            "alpha_s": 0.06931667,
            "alpha_p": 0.0,
            "alpha_pi": 0.0,
            "F1": 0.25393975,
            "F2": 0.0,
            "G1": 0.35600772,
            "G2": 0.0,
            "zeta": 1.25906452,
            "zs": 1.2591,
            "zp": 0.0,
            "alpha": 0.0,
            "n_orbitals": 1,
            "beta_s_xh": 0.0,
            "beta_p_xh": 0.0,
            "alpha_s_xh": 0.0,
            "alpha_p_xh": 0.0,
            "zeta_alpha": 0.0,
            "F_alpha_alpha": 0.0,
            "beta_alpha": 0.0,
            "alpha_alpha": 0.0,
        },
        6: {
            "uss": -50.55997310,
            "upp": -39.60463506,
            "gss": 11.5894,
            "gpp": 10.4261,
            "gsp": 9.6586,
            "gp2": 9.1637,
            "hsp": 1.4332,
            "beta_s": -7.15007507,
            "beta_p": -4.00965991,
            "beta_pi": -5.63958651,
            "alpha_s": 0.09197146,
            "alpha_p": 0.05274021,
            "alpha_pi": 0.09864674,
            "F1": 0.41151269,
            "F2": 0.0,
            "G1": 0.10398816,
            "G2": 0.0,
            "zeta": 1.27811536,
            "zs": 2.0395,
            "zp": 1.9034,
            "alpha": 2.7344,
            "n_orbitals": 4,
            "beta_s_xh": -6.19914817,
            "beta_p_xh": -4.23218526,
            "alpha_s_xh": 0.10023679,
            "alpha_p_xh": 0.05492720,
            "zeta_alpha": 5.70000000,
            "F_alpha_alpha": -283.81699000,
            "beta_alpha": -22.48815939,
            "alpha_alpha": 0.15323932,
        },
        7: {
            "uss": -75.98413465,
            "upp": -57.38630489,
            "gss": 12.0186,
            "gpp": 10.9760,
            "gsp": 10.2426,
            "gp2": 9.7244,
            "hsp": 1.7711,
            "beta_s": -13.42485887,
            "beta_p": -5.69143961,
            "beta_pi": -8.25767437,
            "alpha_s": 0.09461210,
            "alpha_p": 0.06941595,
            "alpha_pi": 0.10511596,
            "F1": 0.58223498,
            "F2": 0.0,
            "G1": 0.05928617,
            "G2": 0.0,
            "zeta": 1.30965521,
            "zs": 2.3649,
            "zp": 2.0563,
            "alpha": 3.0784,
            "n_orbitals": 4,
            "beta_s_xh": -11.40440049,
            "beta_p_xh": -7.87479008,
            "alpha_s_xh": 0.11356707,
            "alpha_p_xh": 0.09244486,
            "zeta_alpha": 6.73673665,
            "F_alpha_alpha": -380.94261410,
            "beta_alpha": -22.78185605,
            "alpha_alpha": 0.15915338,
        },
        8: {
            "uss": -105.79319826,
            "upp": -78.90502490,
            "gss": 14.0967,
            "gpp": 12.8361,
            "gsp": 11.9993,
            "gp2": 11.4586,
            "hsp": 2.8709,
            "beta_s": -14.42839639,
            "beta_p": -8.77114206,
            "beta_pi": -12.94995697,
            "alpha_s": 0.12962541,
            "alpha_p": 0.09275135,
            "alpha_pi": 0.16086067,
            "F1": 0.55266327,
            "F2": 0.0,
            "G1": 0.06226814,
            "G2": 0.0,
            "zeta": 1.20838191,
            "zs": 3.2470,
            "zp": 2.9160,
            "alpha": 3.3391,
            "n_orbitals": 4,
            "beta_s_xh": -13.56403003,
            "beta_p_xh": -9.42200507,
            "alpha_s_xh": 0.14516313,
            "alpha_p_xh": 0.10981869,
            "zeta_alpha": 7.74239776,
            "F_alpha_alpha": -512.51900000,
            "beta_alpha": -26.13251784,
            "alpha_alpha": 0.18281098,
        },
        9: {
            "uss": -120.65477058,
            "upp": -107.50304443,
            "gss": 15.8661,
            "gpp": 14.4092,
            "gsp": 13.3853,
            "gp2": 12.7133,
            "hsp": 3.0541,
            "beta_s": -6.19918959,
            "beta_p": -13.82075048,
            "beta_pi": -18.96710976,
            "alpha_s": 0.31128286,
            "alpha_p": 0.12448702,
            "alpha_pi": 0.21596283,
            "F1": 1.03522357,
            "F2": 0.0,
            "G1": 0.14035077,
            "G2": 0.0,
            "zeta": 1.20564838,
            "zs": 4.3207,
            "zp": 3.5113,
            "alpha": 3.7696,
            "n_orbitals": 4,
            "beta_s_xh": -8.06286638,
            "beta_p_xh": -13.92380910,
            "alpha_s_xh": 0.32623663,
            "alpha_p_xh": 0.15497349,
            "zeta_alpha": 8.70367110,
            "F_alpha_alpha": -685.41999336,
            "beta_alpha": -10.62649349,
            "alpha_alpha": 0.00010713,
        },
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_MAKER = {
    "om1": _make_om1_params,
    "om2": _make_om2_params,
    "om3": _make_om3_params,
}
_VARIANTS = {
    "om1": OMxVariant.OM1,
    "om2": OMxVariant.OM2,
    "om3": OMxVariant.OM3,
}
_VALID_VARIANTS = tuple(_VARIANTS)


def _build_params(
    raw: dict[int, dict], variant: OMxVariant = OMxVariant.OM2
) -> OMxParameterSet:
    """Build an OMxParameterSet from a parameter dictionary."""
    params = OMxParameterSet(variant)
    for Z, data in raw.items():
        ed = OMxElementData()
        ed.Z = Z
        for attr in [
            "uss",
            "upp",
            "gss",
            "gpp",
            "gsp",
            "gp2",
            "hsp",
            "beta_s",
            "beta_p",
            "beta_pi",
            "alpha_s",
            "alpha_p",
            "alpha_pi",
            "F1",
            "F2",
            "G1",
            "G2",
            "zeta",
            "zs",
            "zp",
            "alpha",
            "beta_s_xh",
            "beta_p_xh",
            "alpha_s_xh",
            "alpha_p_xh",
            "zeta_alpha",
            "F_alpha_alpha",
            "beta_alpha",
            "alpha_alpha",
        ]:
            setattr(ed, attr, data.get(attr, 0.0))
        ed.n_orbitals = data.get("n_orbitals", 4)
        params.add_omx_element(ed)
    return params


def load_omx_params(variant: str = "om2") -> OMxParameterSet:
    """Load OMx parameters for H, C, N, O, F.

    Parameters
    ----------
    variant : str
        "om1", "om2", or "om3"

    Returns
    -------
    OMxParameterSet with variant-specific parameters.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VALID_VARIANTS}, got {variant!r}")
    v = _VARIANTS[variant]
    maker = _MAKER[variant]
    raw = maker()
    params = _build_params(raw, variant=v)
    return params


def load_om1_params() -> OMxParameterSet:
    """Load OM1 parameters (Dral 2016 Table 1)."""
    return load_omx_params("om1")


def load_om2_params() -> OMxParameterSet:
    """Load OM2 parameters (Dral 2016 Table 2)."""
    return load_omx_params("om2")


def load_om3_params() -> OMxParameterSet:
    """Load OM3 parameters (Dral 2016 Table 3)."""
    return load_omx_params("om3")
