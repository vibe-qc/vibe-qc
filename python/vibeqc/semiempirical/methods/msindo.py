"""MSINDO semiempirical engine -- faithful port (parity-first).

MSINDO (Bredow, Geudtner & Jug; © Mulliken Center for Theoretical Chemistry,
University of Bonn) is an INDO-family SCF-MO method over Slater orbitals whose
defining feature is a symmetric (Löwdin) orthogonalization folded into the
one-electron (resonance) integrals.  This module is an independent re-
implementation in vibe-qc; it imports no MSINDO code.  It is validated against a
reference MSINDO build out-of-process (``examples/regression/msindo/``).

Status: closed-shell, **s/p/d** main-group elements (H, He, C, N, O, F, and the
3rd-row Si, P, S with their 3d polarization shell + Ne frozen core) reach
**parity** with reference MSINDO.  The general STO kernel, the frozen-core
pseudopotential (``V2CORE``), the penetration correction (``VCORRK``/``VCORRL``),
the Löwdin orthogonalization (``LMUNU``/``HORTH``), the resonance (``DELTAH``),
the local->global rotation (``ROTINT``/``TINT`` via ``HARMTR``), and the
one-center INDO Fock (``GIJ1`` 9x9 packing + ``EINZI`` hybrid d-block from
``FOCKCL``) are ported from the named MSINDO routine and validated block-by-block
against the oracle's ``PRINTOPTS`` dump.

References (the cited MSINDO routine for each formula is named inline):
  Ahlswede & Jug, J. Comput. Chem. 20, 563 & 572 (1999);
  Bredow, Geudtner & Jug, J. Comput. Chem. 22, 861 (2001);
  Nanda & Jug, Theor. Chim. Acta 57, 95 (1980).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from . import msindo_integrals as _ki

# MSINDO recomputes its Bohr radius from constituent 1986-CODATA literals in
# ``const.f`` under default REAL*8.  The result differs slightly from the
# separately tabulated 1986-CODATA Bohr radius.  Geometry is part of the
# parametrized method, so retain the reference executable's internal convention.
MSINDO_BOHR_ANGSTROM = 0.5291772575069162
ANGSTROM_TO_BOHR = 1.0 / MSINDO_BOHR_ANGSTROM

# --------------------------------------------------------------------------- #
# Parameters (datas.f).  H, He, C, N, O, F (2nd row) + Si, P, S (3rd row, 3d).
# --------------------------------------------------------------------------- #

# The published MSINDO parameter set (datas.f, Z=1..54), bundled in
# msindo_params.json (regenerate: examples/regression/msindo/gen_msindo_params.py;
# provenance: docs/license.md).  This replaces the former hand-curated dicts so
# the whole H-Xe set is available and the AL anti-penetration matrix is exact for
# every partner pair -- the old 4-bucket AL carried wrong values for some
# cross-pairs no test had exercised (e.g. C-Al, Al-C).
_PARAMS = json.loads(Path(__file__).with_name("msindo_params.json").read_text())[
    "elements"
]
_NDDO_PARAMETERS_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "vibeqc_msindo_nddo_parameters_active",
    default=False,
)


class _ParameterColumn(Mapping[int, Any]):
    """Read-only parameter column with a context-local NDDO overlay."""

    def __init__(self, name: str, values: dict[int, Any]):
        self._name = name
        self._values = values

    def __getitem__(self, atomic_number: int) -> Any:
        if (
            _NDDO_PARAMETERS_ACTIVE.get()
            and self._name in _NDDO_OVERRIDE_ARRAYS
        ):
            record = _NDDO_PARAMS.get(str(atomic_number))
            if record is not None and self._name in record:
                return record[self._name]
        return self._values[atomic_number]

    def __iter__(self) -> Iterator[int]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _col(name):
    """Per-element column from the bundle, all Z=1..54 incl. zeros (so the
    d-shell direct lookups KDS/KDP/KDD/IPOTD[z] never KeyError)."""
    values = {int(z): e[name] for z, e in _PARAMS.items()}
    return _ParameterColumn(name, values)


# Valence exponents (two-center MUS/MUP/MUD; one-center MUSE/MUPE/MUDE), bohr^-1.
MUS, MUP, MUD = _col("MUS"), _col("MUP"), _col("MUD")
MUSE, MUPE, MUDE = _col("MUSE"), _col("MUPE"), _col("MUDE")
# Resonance K parameters.
KSS, KPS, KPP = _col("KSS"), _col("KPS"), _col("KPP")
KDS, KDP, KDD = _col("KDS"), _col("KDP"), _col("KDD")
# Ionization potentials (Hartree).
IPOTS, IPOTP, IPOTD = _col("IPOTS"), _col("IPOTP"), _col("IPOTD")
# Frozen-core exponents / potentials (1s/2s/2p He & Ne cores; 3* for Ar/Kr cores).
TAU1S, TAU2S, TAU2P = _col("TAU1S"), _col("TAU2S"), _col("TAU2P")
TAU3S, TAU3P, TAU3D = _col("TAU3S"), _col("TAU3P"), _col("TAU3D")
FCP1S, FCP2S, FCP2P = _col("FCP1S"), _col("FCP2S"), _col("FCP2P")
FCP3S, FCP3P, FCP3D = _col("FCP3S"), _col("FCP3P"), _col("FCP3D")
# [Kr] / [Kr]4d¹⁰ frozen-core shells (Rb-Xe): 4s, 4p (z>36) and 4d (z>48).
TAU4S, TAU4P, TAU4D = _col("TAU4S"), _col("TAU4P"), _col("TAU4D")
FCP4S, FCP4P, FCP4D = _col("FCP4S"), _col("FCP4P"), _col("FCP4D")
SCP3D, SCP4S, SCP4P = _col("SCP3D"), _col("SCP4S"), _col("SCP4P")
# Valence occupations (DATA LS = s, MP = p, ND = d).
LS, MP, ND = _col("LS"), _col("MP"), _col("ND")

# Elements with a *validated* SCF path.  Bundled params alone don't certify an
# element -- its ENEG/ATENG/basis/core handling must reproduce the oracle -- so the
# supported set is grown explicitly as each element is parity-checked.  Validated:
# H,He (1s); Li-Ne (2nd row, s/p); Na,Mg (3rd row s/p); Al-Ar (3rd row s/p/d);
# K,Ca (4th row s/p, [Ar] 3s3p frozen core); Sc-Zn (3d transition metals --
# occupied valence 3d with 4s4p, the mixed-n d shell + d-electron penetration);
# Ga-Kr (4th-row p-block); Rb,Sr (5s); Y-Mo (4d metals); Ag,Cd (4d¹⁰5s¹/^2 -- the
# In-Xe-loop ENEG, see eneg()); In-Xe (5p, [Kr]4d¹⁰ frozen core + 5d polarization);
# and the open-d 4d metals Tc-Pd + the heavier lone-pair 5p Sb-Xe (these reach the
# reference SCF stationary point only via the Hückel-guess + WICHT path,
# :data:`_MSINDO_TRAJECTORY_SCF` / :func:`_scf_rhf_msindo`).  Full H-Xe coverage.
_SUPPORTED = set(range(1, 55))  # H-Xe (Z 1-54), all oracle-validated

# Elements whose molecular RHF needs the MSINDO-faithful Hückel-guess + WICHT SCF
# (:func:`_scf_rhf_msindo`) instead of the DIIS path (:func:`_scf_rhf`).  These
# 4d metals (Nb-Pd) and heavier lone-pair 5p (Sb-Xe) converge to a *different* SCF
# state than DIIS finds -- DIIS lands higher (Tc/Rh) or lower with the
# polarization d wrongly occupied (Pd/Sb/Xe), and for the homonuclear 4d dimers
# Nb₂/Mo₂ DIIS jumps to a spurious basin (Nb₂ higher, Mo₂ lower) while the
# Hückel+WICHT trajectory reaches the reference state.  MSINDO's default SCF
# (full-diag + WICHT density damping from the extended-Hückel guess; NOT DIIS, NOT
# Stewart pseudo-diag) reaches the reference state.  Nb/Mo were moved
# here from the DIIS path (2026-06-17) to fix Nb₂/Mo₂; their well-behaved halides
# (NbF₅, MoF₆) reproduce the oracle on WICHT too (Python+C++).  H-Zr, Ag/Cd, In/Sn
# keep Hcore/DIIS as their primary path: it is more robust for a few
# near-degenerate small molecules (e.g. AgH), where numpy's and MSINDO's LAPACK
# diagonalizers resolve the degenerate Hückel guess into different metastable
# states. Validated H-Ar RHF evaluates WICHT as a finite alternate-basin probe;
# other converged non-pinned elements retain Hcore/DIIS, while failed-primary
# recovery remains global. See :func:`_scf_rhf_molecular` and
# docs/user_guide/msindo.md.
_MSINDO_TRAJECTORY_SCF = {41, 42, 43, 44, 45, 46, 51, 52, 53, 54}  # Nb-Pd, Sb-Xe
_MSINDO_ROOT_PROBE_ELEMENTS = frozenset(range(1, 19))  # H-Ar validation scope

# AL anti-penetration matrix: one value per partner shell-group (datas.f SHD
# groups), per central atom.
_AL_GROUPS = (
    (1, 2),
    (3, 5),
    (6, 10),
    (11, 12),
    (13, 18),
    (19, 20),
    (21, 30),
    (31, 36),
    (37, 38),
    (39, 48),
    (49, 54),
)
_AL = {int(z): tuple(e["AL"]) for z, e in _PARAMS.items()}


def AL(zk: int, zl: int) -> float:
    """Anti-penetration factor AL(K,L) (datas.f), by the partner L's shell-group."""
    grp = next((i for i, (lo, hi) in enumerate(_AL_GROUPS) if lo <= zl <= hi), 0)
    table = (
        _AL_NDDO
        if _NDDO_PARAMETERS_ACTIVE.get() and zk in _AL_NDDO
        else _AL
    )
    return table.get(zk, (0.0,) * len(_AL_GROUPS))[grp]


# --------------------------------------------------------------------------- #
# NDDO parameter overrides (nddoparam.f).                                      #
# --------------------------------------------------------------------------- #
# MSINDO's optional NDDO mode is a *separate parametrization*: when the NDDO
# keyword is set, ``parset.f`` calls ``NDDOPARAM`` (nddoparam.f), which
# OVERWRITES the global parameter arrays -- valence/one-centre Slater exponents
# (MUS/MUP/MUD/MUSE/MUPE/MUDE), frozen-core exponents (TAU*), ionization
# potentials (IPOT*), the 3d shielding (SCP3D), the resonance K betas, and the
# AL anti-penetration matrix -- with NDDO-specific values for H, Li-F, Na-Cl
# (the noble gases are not parametrized).  With these exponents the *existing*
# one-centre machinery (slater_condon / one_center_2e) and ENEG reproduce the
# oracle NDDO integrals exactly (verified to 5 decimals -- e.g. NDDO MUSE(F)=
# 2.5411 -> GSS(F)=0.92313, MUSE(H)=0.9579 -> GSS(H)=0.59869).
#
# vibe-qc selects the overrides through the context-local ``_nddo_params()``
# overlay, preserving the values selected by ``IF(NDDO) CALL NDDOPARAM`` without
# mutating process-global tables. Regenerate the bundle with
# examples/regression/msindo/gen_msindo_params_nddo.py.
_NDDO_PARAMS = json.loads(
    Path(__file__).with_name("msindo_params_nddo.json").read_text()
)["elements"]
# NDDO-parametrized elements: H (1), Li-F (3-9), Na-Cl (11-17).
_SUPPORTED_NDDO = frozenset(int(z) for z in _NDDO_PARAMS)
# Scalar per-element arrays NDDOPARAM overrides.  (AL is overridden separately
# from ``_AL_NDDO`` since it is a per-central tuple, not a scalar.)
_NDDO_OVERRIDE_ARRAYS = (
    "MUS",
    "MUP",
    "MUD",
    "MUSE",
    "MUPE",
    "MUDE",
    "TAU1S",
    "TAU2S",
    "TAU2P",
    "IPOTS",
    "IPOTP",
    "IPOTD",
    "SCP3D",
    "KSS",
    "KPS",
    "KPP",
    "KDS",
    "KDP",
    "KDD",
)
# NDDO anti-penetration AL, per central Z as the 11-tuple over partner shell-
# groups (same structure as the INDO ``_AL``) -- overrides ``_AL`` under NDDO.
_AL_NDDO = {int(z): tuple(rec["AL"]) for z, rec in _NDDO_PARAMS.items()}


@contextlib.contextmanager
def _nddo_params():
    """Select the NDDO parameter overlay for the current execution context.

    Faithful to MSINDO's ``IF(NDDO) CALL NDDOPARAM`` (nddoparam.f), which
    overwrites global arrays. vibe-qc presents the same values through
    :class:`_ParameterColumn`, but a ContextVar keeps nested, threaded, and
    asynchronous calculations isolated without changing the parameterized
    equations.
    """
    token = _NDDO_PARAMETERS_ACTIVE.set(True)
    try:
        yield
    finally:
        _NDDO_PARAMETERS_ACTIVE.reset(token)


def n_principal(z: int) -> int:
    """Principal quantum number of the valence s shell (setpar.f)."""
    if z <= 2:
        return 1
    if z <= 10:
        return 2
    if z <= 18:
        return 3
    if z <= 36:
        return 4
    return 5


def n_p_principal(z: int) -> int:
    """Principal quantum number for the p shell (intdrv.f: H uses INP=2)."""
    return 2 if z == 1 else n_principal(z)


def n_d_principal(z: int) -> int:
    """Principal quantum number of the valence d shell.

    For the transition metals the d shell is the *(n-1)d* (Sc-Zn 3d, Y-Cd 4d)
    while s/p are the outer ns/np -- a mixed principal number (einzentren.f /
    calsla.f).  For the d-bearing main-group rows (Al-Ar 3d, In-Xe 5d) it equals
    ``n_principal``, so this returns the same value for every Z <= 20 (no change
    to the validated main-group path)."""
    if z <= 30:
        return 3  # Al-Ar (3d) and Sc-Zn (3d)
    if z <= 48:
        return 4  # Ga-Kr (4d) and Y-Cd (4d)
    return 5  # In-Xe (5d)


def eff_core_charge(z: int) -> int:
    """Effective core charge CZ (efflad.f)."""
    if z <= 2:
        return z
    if z <= 10:
        return z - 2
    if z <= 18:
        return z - 10
    if z <= 30:
        return z - 18
    if z <= 36:
        return z - 28
    if z <= 48:
        return z - 36
    return z - 46


def n_basis(z: int) -> int:
    """Number of valence basis functions (nbasis.f), default basis.

    H/He = 1 (1s); 2nd row + Na/Mg = 4 (s/p; the d-exponent placeholders that
    datas.f carries for C/O are NOT in the default basis); Al-Ar = 9 (s/p/d).
    For Z>18 the d shell is present iff the element carries a valence/polarization
    d exponent (``MUD`` nonzero) -- K/Ca/Rb/Sr are s/p (4); the transition rows
    and heavy p-block carry d (9)."""
    if z <= 2:
        return 1
    if z <= 12:
        return 4
    if z <= 18:
        return 9
    return 9 if MUD.get(z, 0.0) != 0.0 else 4


# --------------------------------------------------------------------------- #
# Slater-Condon factors (calsla.f) + one-center INDO integrals (gij1.f).
# --------------------------------------------------------------------------- #


def slater_condon(z: int) -> dict:
    """Slater-Condon factors via RADINT (calsla.f / atomic_reference.f).

    Returns F0SS,F0SP,F0PP,F2PP,G1SP (always) plus the d factors
    F0SD,F0PD,F0DD,G1PD,F2PD,F2DD,G2SD,G3PD,F4DD,I1SPPD,I2SDPP,I2SDDD when the
    element carries a d shell.
    """
    f = {
        k: 0.0
        for k in (
            "F0SS",
            "F0SP",
            "F0PP",
            "F2PP",
            "G1SP",
            "F0SD",
            "F0PD",
            "F0DD",
            "G1PD",
            "F2PD",
            "F2DD",
            "G2SD",
            "G3PD",
            "F4DD",
            "I1SPPD",
            "I2SDPP",
            "I2SDDD",
        )
    }
    zs, zp, zd = MUSE[z], MUPE[z], MUDE.get(z, 0.0)
    if z <= 2:  # atomic_reference.f H/He block
        f["F0SS"] = 5.0 / 8.0 * zs
        if z == 1:
            f["F0SP"] = _ki.radint(0, 2, 2, 2, 2, zs, zp, zs, zp)
            f["F0PP"] = 93.0 / 256.0 * zp
            f["G1SP"] = _ki.radint(1, 2, 2, 2, 2, zs, zp, zp, zs)
            f["F2PP"] = _ki.radint(2, 2, 2, 2, 2, zp, zp, zp, zp)
        return f
    n = n_principal(z)
    R = lambda lam, a, b, c, d: _ki.radint(lam, n, n, n, n, a, b, c, d)
    f["F0SS"] = R(0, zs, zs, zs, zs)
    f["F0SP"] = R(0, zs, zp, zs, zp)
    f["F0PP"] = R(0, zp, zp, zp, zp)
    f["G1SP"] = R(1, zs, zp, zp, zs)
    f["F2PP"] = R(2, zp, zp, zp, zp)
    if n_basis(z) >= 9:
        # The d shell may sit at a different principal number than s/p (the
        # transition metals: 3d with 4s4p).  Each RADINT carries the per-orbital
        # principal number, matching calsla.f exactly (n for s/p, nd for d).
        nd = n_d_principal(z)
        ri = _ki.radint
        f["F0SD"] = ri(0, n, nd, n, nd, zs, zd, zs, zd)
        f["F0PD"] = ri(0, n, nd, n, nd, zp, zd, zp, zd)
        f["F0DD"] = ri(0, nd, nd, nd, nd, zd, zd, zd, zd)
        f["G1PD"] = ri(1, n, nd, nd, n, zp, zd, zd, zp)
        f["F2PD"] = ri(2, n, nd, n, nd, zp, zd, zp, zd)
        f["F2DD"] = ri(2, nd, nd, nd, nd, zd, zd, zd, zd)
        f["G2SD"] = ri(2, n, nd, nd, n, zs, zd, zd, zs)
        f["G3PD"] = ri(3, n, nd, nd, n, zp, zd, zd, zp)
        f["F4DD"] = ri(4, nd, nd, nd, nd, zd, zd, zd, zd)
        f["I1SPPD"] = ri(1, n, n, n, nd, zs, zp, zp, zd)
        f["I2SDPP"] = ri(2, n, n, nd, n, zs, zp, zd, zp)
        f["I2SDDD"] = ri(2, n, nd, nd, nd, zs, zd, zd, zd)
    return f


def one_center_2e(z: int) -> dict:
    """One-center two-electron integrals (gij1.f INDO relations), s/p only.

    GSS=(ss|ss), GSP=(ss|pp), GPP=(pp|pp), GP2=(pp|p'p'),
    HSP=(sp|sp) exchange, HPP=(pp'|pp') exchange.  (For the d block use
    :func:`one_center_gmunu`.)
    """
    f = slater_condon(z)
    out = {"GSS": f["F0SS"]}
    if n_basis(z) >= 4:
        out["GSP"] = f["F0SP"]
        out["HSP"] = f["G1SP"] / 3.0
        out["GPP"] = f["F0PP"] + 4.0 / 25.0 * f["F2PP"]
        out["GP2"] = f["F0PP"] - 2.0 / 25.0 * f["F2PP"]
        out["HPP"] = 3.0 / 25.0 * f["F2PP"]
    return out


# ---------------------------------------------------------------------------
# NDDO two-centre two-electron integrals ("improved integrals" mode).
# ---------------------------------------------------------------------------
# MSINDO's optional NDDO mode keeps the two-centre *multipole* interactions
# that INDO drops -- the dipole-monopole, dipole-dipole, ... couplings between the
# charge distributions on different atoms -- in the Dewar/MNDO point-charge
# model (B. Voigt, Theor. Chim. Acta 31, 289 (1973)).  Each integral is the
# closed-form interaction of point-charge configurations whose separation is
# the s-p "dipole length" DA and whose finite size enters through the
# Klopman-Ohno additive screening RHO. These are the integral kernels
# (spss_si.f / sppp_si.f / spdd_si.f / spsp_si.f / spsp_pi.f) and their
# parameters DA/ASP/GDD (einzentren.f). The Fock assembly below follows
# nddofockcl.f + spspfockcl.f; v2core.f supplies the one-electron core term.


def nddo_da(z: int) -> float:
    """s-p charge separation (MNDO "dipole length" D1) for the multipole model.

    einzentren.f: DA = (n+1/2).(ζs.ζp)^(n+1/2) / (√3.((ζs+ζp)/2)^(2n+2)), with the
    *two-centre* exponents ζs=MUS, ζp=MUP and n the principal quantum number.
    Returns 0 for an atom with no p shell (H/He) -- it carries no s-p dipole.
    """
    zp = MUP.get(z, 0.0)
    if zp == 0.0:
        return 0.0
    zs = MUS[z]
    c = n_principal(z) + 0.5
    return c * (zs * zp) ** c / (math.sqrt(3.0) * ((zs + zp) / 2.0) ** (2 * c + 1))


def nddo_asp(z: int) -> float:
    """ASP = G1sp/3 (einzentren.f), the additive s-p term inside the screening."""
    return slater_condon(z)["G1SP"] / 3.0


def _nddo_rho_self(z: int) -> float:
    """Half the dipole's own Klopman-Ohno size, 0.5.(DA^2/ASP)^(1/3)."""
    return 0.5 * (nddo_da(z) ** 2 / nddo_asp(z)) ** (1.0 / 3.0)


def nddo_spss_si(zi: int, zj: int, r: float) -> float:
    """(s_i p_i | s_j s_j): dipole-on-i x monopole-on-j (spss_si.f).

    Two ±1/2 charges separated by DA_i interacting with the j monopole; RHO adds
    the dipole's own size and the monopole's size 1/GSS_j.  As r->inf this tends to
    the bare dipole field DA_i / r^2.
    """
    da = nddo_da(zi)
    rho = _nddo_rho_self(zi) + 0.5 / gss(zj)
    return 1.0 / math.sqrt((r - da / 2.0) ** 2 + rho**2) - 1.0 / math.sqrt(
        (r + da / 2.0) ** 2 + rho**2
    )


def nddo_dspss_si(zi: int, zj: int, r: float) -> float:
    """Radial derivative of :func:`nddo_spss_si` in bohr-based units.

    Dewar--Thiel, Theor. Chim. Acta 46, 89 (1977), Eq. 53 defines the
    monopole--dipole point-charge energy. This differentiates that same energy,
    matching MSINDO ``dspss_si.f``; it is ``d integral / dR``, not a force.
    """
    da = nddo_da(zi)
    rho = _nddo_rho_self(zi) + 0.5 / gss(zj)
    minus = r - da / 2.0
    plus = r + da / 2.0
    return -minus / (minus * minus + rho * rho) ** 1.5 + plus / (
        plus * plus + rho * rho
    ) ** 1.5


def nddo_sppp_si(zi: int, zj: int, r: float) -> float:
    """(s_i p_i | p_j p_j): dipole-on-i x p-monopole-on-j (sppp_si.f).

    As :func:`nddo_spss_si` but the j monopole has the p size 1/GPP_j.
    """
    da = nddo_da(zi)
    rho = _nddo_rho_self(zi) + 0.5 / one_center_2e(zj)["GPP"]
    return 1.0 / math.sqrt((r - da / 2.0) ** 2 + rho**2) - 1.0 / math.sqrt(
        (r + da / 2.0) ** 2 + rho**2
    )


def nddo_dsppp_si(zi: int, zj: int, r: float) -> float:
    """Radial derivative of :func:`nddo_sppp_si` (``dsppp_si.f``)."""
    da = nddo_da(zi)
    rho = _nddo_rho_self(zi) + 0.5 / one_center_2e(zj)["GPP"]
    minus = r - da / 2.0
    plus = r + da / 2.0
    return -minus / (minus * minus + rho * rho) ** 1.5 + plus / (
        plus * plus + rho * rho
    ) ** 1.5


def nddo_gdd(z: int) -> float:
    """Spherical d-shell monopole self-repulsion ``GDD`` (``einzentren.f``)."""
    f = slater_condon(z)
    return f["F0DD"] + 4.0 / 49.0 * f["F2DD"] + 36.0 / 441.0 * f["F4DD"]


def nddo_spdd_si(zi: int, zj: int, r: float) -> float:
    """(s_i p_i | d_j d_j): dipole-on-i x d-monopole-on-j.

    This is the asymmetric Ohno--Klopman/Dewar--Thiel expression in MSINDO
    ``spdd_si.f``: the first atom supplies ``DA``/``ASP`` for the dipole while
    the second supplies the spherical d-shell self-repulsion ``GDD``.
    """
    da = nddo_da(zi)
    rho = _nddo_rho_self(zi) + 0.5 / nddo_gdd(zj)
    return 1.0 / math.sqrt((r - da / 2.0) ** 2 + rho**2) - 1.0 / math.sqrt(
        (r + da / 2.0) ** 2 + rho**2
    )


def nddo_dspdd_si(zi: int, zj: int, r: float) -> float:
    """Radial derivative of :func:`nddo_spdd_si` in bohr-based units."""
    da = nddo_da(zi)
    rho = _nddo_rho_self(zi) + 0.5 / nddo_gdd(zj)
    minus = r - da / 2.0
    plus = r + da / 2.0
    # MSINDO DSPDD_SI convention: d(integral)/dR, not a force
    # (dspdd_si.f:25-28); Cartesian gradient signs are applied by the caller.
    return -minus / (minus * minus + rho * rho) ** 1.5 + plus / (
        plus * plus + rho * rho
    ) ** 1.5


def nddo_spsp_si(zi: int, zj: int, r: float) -> float:
    """(s_i p_i | s_j p_j) s: axial dipole-on-i x dipole-on-j (spsp_si.f).

    Four ±1/4 point-charge terms (the two dipoles aligned along the bond).
    As r->inf -> -2.DA_i.DA_j / r^3.
    """
    di, dj = nddo_da(zi), nddo_da(zj)
    rho = _nddo_rho_self(zi) + _nddo_rho_self(zj)
    return (
        0.25 / math.sqrt((r + di - dj) ** 2 + rho**2)
        - 0.25 / math.sqrt((r + di + dj) ** 2 + rho**2)
        - 0.25 / math.sqrt((r - di - dj) ** 2 + rho**2)
        + 0.25 / math.sqrt((r - di + dj) ** 2 + rho**2)
    )


def nddo_dspsp_si(zi: int, zj: int, r: float) -> float:
    """Radial derivative of the axial dipole--dipole integral.

    This is the derivative of the Dewar--Thiel 1977 point-charge expression
    used by the shipped energy and follows MSINDO ``dspsp_si.f``. The source
    routine returns ``d integral / dR``; Cartesian signs are applied later.
    """
    di, dj = nddo_da(zi), nddo_da(zj)
    rho = _nddo_rho_self(zi) + _nddo_rho_self(zj)
    x1 = r + di - dj
    x2 = r + di + dj
    x3 = r - di - dj
    x4 = r - di + dj
    return (
        -0.25 * x1 / (x1 * x1 + rho * rho) ** 1.5
        + 0.25 * x2 / (x2 * x2 + rho * rho) ** 1.5
        + 0.25 * x3 / (x3 * x3 + rho * rho) ** 1.5
        - 0.25 * x4 / (x4 * x4 + rho * rho) ** 1.5
    )


def nddo_spsp_pi(zi: int, zj: int, r: float) -> float:
    """(s_i p_i | s_j p_j) pi: perpendicular dipole-on-i x dipole-on-j (spsp_pi.f).

    As r->inf -> +DA_i.DA_j / r^3 (the perpendicular dipole-dipole field).
    """
    di, dj = nddo_da(zi), nddo_da(zj)
    rho = _nddo_rho_self(zi) + _nddo_rho_self(zj)
    return 0.5 / math.sqrt(r**2 + (di - dj) ** 2 + rho**2) - 0.5 / math.sqrt(
        r**2 + (di + dj) ** 2 + rho**2
    )


def nddo_dspsp_pi(zi: int, zj: int, r: float) -> float:
    """Radial derivative of the transverse dipole--dipole integral."""
    di, dj = nddo_da(zi), nddo_da(zj)
    rho = _nddo_rho_self(zi) + _nddo_rho_self(zj)
    dminus = r * r + (di - dj) ** 2 + rho * rho
    dplus = r * r + (di + dj) ** 2 + rho * rho
    return -0.5 * r / dminus**1.5 + 0.5 * r / dplus**1.5


def one_center_gmunu(z: int) -> np.ndarray:
    """Full one-center GMUNU 9x9 block (gij1.f): upper=Coulomb, lower=exchange.

    Orbital order 0=s, 1..3=px,py,pz, 4..8=dz2,dxz,dyz,dx2y2,dxy.
    """
    f = slater_condon(z)
    G = np.zeros((9, 9))
    G[0, 0] = f["F0SS"]
    nb = n_basis(z)
    if nb >= 4:
        F0SP, G1SP, F0PP, F2PP = f["F0SP"], f["G1SP"], f["F0PP"], f["F2PP"]
        for p in (1, 2, 3):
            G[0, p] = F0SP
            G[p, 0] = G1SP / 3.0
            G[p, p] = F0PP + 4.0 / 25.0 * F2PP
        for a, b in [(1, 2), (1, 3), (2, 3)]:
            G[a, b] = F0PP - 2.0 / 25.0 * F2PP
            G[b, a] = 3.0 / 25.0 * F2PP
    if nb >= 9:
        F0SD, F0PD, F0DD = f["F0SD"], f["F0PD"], f["F0DD"]
        G1PD, F2PD, F2DD = f["G1PD"], f["F2PD"], f["F2DD"]
        G2SD, G3PD, F4DD = f["G2SD"], f["G3PD"], f["F4DD"]
        for d in range(4, 9):
            G[0, d] = F0SD
            G[d, 0] = G2SD / 5.0
            G[d, d] = F0DD + 4.0 / 49.0 * F2DD + 36.0 / 441.0 * F4DD
        # p-d Coulomb (upper)
        G[1, 4] = G[2, 4] = F0PD - 2.0 / 35.0 * F2PD
        G[3, 4] = F0PD + 4.0 / 35.0 * F2PD
        for a, b in [(1, 5), (3, 5), (2, 6), (3, 6), (1, 7), (2, 7), (1, 8), (2, 8)]:
            G[a, b] = F0PD + 2.0 / 35.0 * F2PD
        for a, b in [(2, 5), (1, 6), (3, 7), (3, 8)]:
            G[a, b] = F0PD - 4.0 / 35.0 * F2PD
        # p-d exchange (lower)
        G[4, 1] = G[4, 2] = G1PD / 15.0 + 18.0 / 245.0 * G3PD
        G[4, 3] = 4.0 / 15.0 * G1PD + 27.0 / 245.0 * G3PD
        for a, b in [(5, 1), (5, 3), (6, 2), (6, 3), (7, 1), (7, 2), (8, 1), (8, 2)]:
            G[a, b] = 3.0 / 15.0 * G1PD + 24.0 / 245.0 * G3PD
        for a, b in [(5, 2), (6, 1), (7, 3), (8, 3)]:
            G[a, b] = 15.0 / 245.0 * G3PD
        # d-d Coulomb (upper)
        G[4, 5] = G[4, 6] = F0DD + 2.0 / 49.0 * F2DD - 24.0 / 441.0 * F4DD
        G[4, 7] = G[4, 8] = F0DD - 4.0 / 49.0 * F2DD + 6.0 / 441.0 * F4DD
        for a, b in [(5, 6), (5, 7), (5, 8), (6, 7), (6, 8)]:
            G[a, b] = F0DD - 2.0 / 49.0 * F2DD - 4.0 / 441.0 * F4DD
        G[7, 8] = F0DD + 4.0 / 49.0 * F2DD - 34.0 / 441.0 * F4DD
        # d-d exchange (lower)
        G[5, 4] = G[6, 4] = F2DD / 49.0 + 30.0 / 441.0 * F4DD
        G[7, 4] = G[8, 4] = 4.0 / 49.0 * F2DD + 15.0 / 441.0 * F4DD
        for a, b in [(6, 5), (7, 5), (8, 5), (7, 6), (8, 6)]:
            G[a, b] = 3.0 / 49.0 * F2DD + 20.0 / 441.0 * F4DD
        G[8, 7] = 35.0 / 441.0 * F4DD
    return G


def einzi_hyb(z: int) -> list:
    """One-center hybrid d integrals HYB(1..21) (einzi.f).  1-indexed: HYB[1..21].

    Built from the Slater-Condon factors; combine in the FOCKCL d-block.
    """
    f = slater_condon(z)
    s3, s5 = math.sqrt(3.0), math.sqrt(5.0)
    R1SPPD, R2SDPP, R2SDDD = f["I1SPPD"], f["I2SDPP"], f["I2SDDD"]
    R2PPDD, R1PDPD, R3PDPD = f["F2PD"], f["G1PD"], f["G3PD"]
    R2DDDD, R4DDDD = f["F2DD"], f["F4DD"]
    h = [0.0] * 22  # h[1..21]
    h[1] = -1.0 / 3.0 / s5 * R1SPPD
    h[2] = 1.0 / s3 / s5 * R1SPPD
    h[3] = 2.0 / 3.0 / s5 * R1SPPD
    h[4] = -1.0 / 5.0 / s5 * R2SDPP
    h[5] = 2.0 / 5.0 / s5 * R2SDPP
    h[6] = 2.0 / 7.0 / s5 * R2SDDD
    h[7] = 1.0 / 7.0 / s5 * R2SDDD
    h[8] = s3 / 5.0 / s5 * R2SDPP
    h[9] = s3 / 7.0 / s5 * R2SDDD
    h[10] = -2.0 * s3 / 35.0 * R2PPDD
    h[11] = 3.0 / 35.0 * R2PPDD
    h[12] = s3 / 35.0 * R2PPDD
    h[13] = -s3 / 15.0 * R1PDPD - 3.0 * s3 / 245.0 * R3PDPD
    h[14] = -s3 / 15.0 * R1PDPD + 12.0 * s3 / 245.0 * R3PDPD
    h[15] = 1.0 / 5.0 * R1PDPD - 6.0 / 245.0 * R3PDPD
    h[16] = 2.0 * s3 / 15.0 * R1PDPD - 9.0 * s3 / 245.0 * R3PDPD
    h[17] = 3.0 / 49.0 * R3PDPD
    h[18] = 1.0 / 5.0 * R1PDPD - 3.0 / 35.0 * R3PDPD
    h[19] = s3 / 49.0 * R2DDDD - 5.0 * s3 / 441.0 * R4DDDD
    h[20] = -2.0 * s3 / 49.0 * R2DDDD + 10.0 * s3 / 441.0 * R4DDDD
    h[21] = 3.0 / 49.0 * R2DDDD - 5.0 / 147.0 * R4DDDD
    return h


# Alkali (s¹) and group-2 (s^2, empty p) elements use distinct ENEG/ATENG forms.
_ALKALI = {3, 11, 19, 37}  # Li, Na, K, Rb -- one valence s electron
_ALKALINE_EARTH = {4, 12, 20, 38}  # Be, Mg, Ca, Sr -- s^2
# Transition metals (occupied valence d, ND!=0) use the Sc-Zn ENEG/ATENG form;
# atomic_reference.f's Y-Cd (4d) block is term-for-term identical to Sc-Zn (3d).
_TM = set(range(21, 31)) | set(range(39, 47))  # Sc-Zn (3d) + Y-Pd (4d)


def eneg(z: int) -> tuple[float, float, float]:
    """Diagonal one-electron energies (U_ss, U_pp, U_dd) -- atomic_reference.f."""
    f = slater_condon(z)
    if z == 1:
        return (IPOTS[1], IPOTP[1], 0.0)
    if z == 2:
        return (IPOTS[2] - f["F0SS"], 0.0, 0.0)
    if z in _ALKALI:
        # Li/Na, 2s¹/3s¹ (atomic_reference.f:75-77,144-146): one electron, so no
        # electron-electron correction (same shape as H).
        return (IPOTS[z], IPOTP[z], 0.0)
    if z in _TM:
        # Sc-Zn occupied-d (atomic_reference.f:234-261): SEL/PEL/DEL = s/p/d
        # valence occupations, with the 4s/4p/3d shielding potentials.
        sel, pel, de = float(LS[z]), float(MP[z]), float(ND[z])
        e1 = (
            IPOTS[z]
            - (sel - 1) * f["F0SS"]
            - pel * f["F0SP"]
            - de * f["F0SD"]
            + pel * f["G1SP"] / 6.0
            + de * f["G2SD"] / 10.0
        )
        e2 = (
            IPOTP[z]
            - (sel - 1) * f["F0SP"]
            - de * f["F0PD"]
            + (sel - 1) * f["G1SP"] / 6.0
            + de * f["G1PD"] / 15.0
            + 3.0 / 70.0 * de * f["G3PD"]
        )
        e3 = (
            IPOTD[z]
            - (de - 1) * f["F0DD"]
            + 2.0 / 63.0 * (de - 1) * (f["F2DD"] + f["F4DD"])
            - sel * f["F0SD"]
            + sel * f["G2SD"] / 10.0
            - pel * f["F0PD"]
            + pel * f["G1PD"] / 15.0
            + 3.0 / 70.0 * pel * f["G3PD"]
        )
        return (
            e1 * (1.0 - SCP4S.get(z, 0.0)),
            e2 * (1.0 - SCP4P.get(z, 0.0)),
            e3 * (1.0 - SCP3D.get(z, 0.0)),
        )
    if z in _ALKALINE_EARTH:
        # Be/Mg s^2 special case (atomic_reference.f:79-81, 148-150): the empty-p
        # ENEG is not the general (pel-1) form.
        return (IPOTS[z] - f["F0SS"], IPOTP[z] - f["F0SP"] + f["G1SP"] / 6.0, 0.0)
    if z in (47, 48):
        # Ag (4d¹⁰5s¹) and Cd (4d¹⁰5s^2): occupied-4d elements that
        # atomic_reference.f computes via the In-Xe loop (DO L=47,54, lines
        # 403-432), which runs AFTER the Y-Cd loop (DO L=39,48) and OVERWRITES
        # the Sc-Zn-style occupied-d values for L=47,48.  The *final* oracle
        # ENEG/ATENG for Ag/Cd therefore use the In-Xe p-block formula carrying
        # the OCCUPIED d shell (DEL=ND=10) -- NOT the Y-Pd occupied-d TM form
        # (which would add the -(DEL-1).F0DD d-d self-repulsion to U_dd).  This
        # is the same formula as the In-Xe branch below (and the Ga-Kr branch),
        # but with SEL=LS=1/2 and DEL=10 it does NOT reduce to the general
        # (pel-1) p-block path the way the real p-block In-Xe (SEL=2/DEL=0)
        # does.  Only SCP3D is applied (atomic_reference.f:424 -- the In-Xe loop,
        # unlike Y-Cd, does not apply SCP4S/SCP4P); all three are 0 for Ag/Cd.
        sel, pel, de = float(LS[z]), float(MP[z]), float(ND[z])
        selm1, pelm1 = sel - 1.0, pel - 1.0
        e1 = (
            IPOTS[z]
            - selm1 * f["F0SS"]
            - pel * f["F0SP"]
            - de * f["F0SD"]
            + pel * f["G1SP"] / 6.0
            + de * f["G2SD"] / 10.0
        )
        e2 = (
            IPOTP[z]
            - pelm1 * (f["F0PP"] - 2.0 / 25.0 * f["F2PP"])
            - sel * (f["F0SP"] - f["G1SP"] / 6.0)
            - de * (f["F0PD"] - f["G1PD"] / 15.0 - 3.0 / 70.0 * f["G3PD"])
        )
        e3 = (
            IPOTD[z]
            - sel * f["F0SD"]
            + sel * f["G2SD"] / 10.0
            - pelm1 * (f["F0PD"] + f["G1PD"] / 15.0 + 3.0 / 70.0 * f["G3PD"])
        )
        e3 *= 1.0 - SCP3D.get(z, 0.0)
        return (e1, e2, e3)
    pel = float(MP[z])
    pelm1 = pel - 1.0
    e1 = IPOTS[z] - f["F0SS"] - pel * f["F0SP"] + pel * f["G1SP"] / 6.0
    e2 = (
        IPOTP[z]
        - 2.0 * f["F0SP"]
        - pelm1 * f["F0PP"]
        + f["G1SP"] / 3.0
        + 2.0 / 25.0 * pelm1 * f["F2PP"]
    )
    e3 = 0.0
    if n_basis(z) >= 9:
        if 31 <= z <= 36 or 49 <= z <= 54:
            # Ga-Kr (4th-row) + In-Xe (5th-row) p-block (atomic_reference.f:300-306):
            # empty 4d/5d polarization over a filled [Ar]3d¹⁰ / [Kr]4d¹⁰ core. Same
            # shape as Al-Ar EXCEPT the G1PD and G3PD exchange terms are NEGATED.
            e3 = (
                IPOTD[z]
                - 2.0 * f["F0SD"]
                - pelm1 * f["F0PD"]
                - pelm1 * f["G1PD"] / 15.0
                + f["G2SD"] / 5.0
                - 3.0 / 70.0 * pelm1 * f["G3PD"]
            )
        else:
            # atomic_reference.f Al-Ar d block (B-Ne uses the same form).
            e3 = (
                IPOTD[z]
                - 2.0 * f["F0SD"]
                - pelm1 * f["F0PD"]
                + pelm1 * f["G1PD"] / 15.0
                + f["G2SD"] / 5.0
                + 3.0 / 70.0 * pelm1 * f["G3PD"]
            )
        e3 *= 1.0 - SCP3D.get(z, 0.0)
    return (e1, e2, e3)


_ATENG_MULT = {
    6: 3.0 / 25.0,
    7: 9.0 / 25.0,
    8: 3.0 / 25.0,
    14: 3.0 / 25.0,
    15: 9.0 / 25.0,
    16: 3.0 / 25.0,
    32: 3.0 / 25.0,
    33: 9.0 / 25.0,
    34: 3.0 / 25.0,
}  # Ge, As, Se
# atomic_reference.f extra ATENG corrections for the 3rd row (d-admixture).
_ATENG_EXTRA = {
    13: -0.0000297976,
    14: -0.0000300569,
    16: -0.0000342370,
    17: -0.0000368735,
}


def ateng(z: int) -> float:
    """Atomic reference energy ATENG (atomic_reference.f)."""
    f = slater_condon(z)
    if z == 1:
        return IPOTS[1]
    if z == 2:
        return 2.0 * (IPOTS[2] - f["F0SS"]) + f["F0SS"]
    if z in _ALKALI:  # Li/Na s¹: one-electron reference, ATENG = U_ss = IPOTS
        return IPOTS[z]
    if z in _TM:  # Sc-Zn (atomic_reference.f:265-286): occupied-d reference
        sel, pel, de = float(LS[z]), float(MP[z]), float(ND[z])
        u = eneg(z)
        a = (
            sel * u[0]
            + pel * u[1]
            + de * u[2]
            + sel * (sel - 1) / 2.0 * f["F0SS"]
            + pel * (pel - 1) / 2.0 * (f["F0PP"] - 2.0 / 25.0 * f["F2PP"])
            + de * (de - 1) / 2.0 * (f["F0DD"] - 2.0 / 63.0 * (f["F2DD"] + f["F4DD"]))
            + sel * pel * (f["F0SP"] - 1.0 / 6.0 * f["G1SP"])
            + sel * de * (f["F0SD"] - 1.0 / 10.0 * f["G2SD"])
            + pel * de * (f["F0PD"] - 1.0 / 15.0 * f["G1PD"] - 3.0 / 70.0 * f["G3PD"])
        )
        # Per-element ground-state multiplet corrections (atomic_reference.f:279-286).
        f2, f4, g2 = f["F2DD"], f["F4DD"], f["G2SD"]
        corr = {
            21: -0.0010609939,
            22: -58.0 / 441.0 * f2 + 5.0 / 441.0 * f4,
            23: -93.0 / 441.0 * f2 - 30.0 / 441.0 * f4,
            24: -25.0 / 63.0 * (f2 + f4) - g2 / 2.0,
            25: -25.0 / 63.0 * (f2 + f4),
            26: -15.0 / 63.0 * (f2 + f4),
            27: -93.0 / 441.0 * f2 - 30.0 / 441.0 * f4,
            28: -58.0 / 441.0 * f2 + 5.0 / 441.0 * f4,
        }.get(z, 0.0)
        return a + corr
    if z in _ALKALINE_EARTH:  # Be/Mg s^2: ATENG = 2.U_ss + F0SS
        return 2.0 * eneg(z)[0] + f["F0SS"]
    pel = float(MP[z])
    pelm1 = pel - 1.0
    u = eneg(z)
    a = (
        2.0 * u[0]
        + pel * u[1]
        + f["F0SS"]
        + pel * pelm1 * f["F0PP"] / 2.0
        + 2.0 * pel * f["F0SP"]
        - pel / 3.0 * f["G1SP"]
        - pel * pelm1 / 25.0 * f["F2PP"]
    )
    a -= _ATENG_MULT.get(z, 0.0) * f["F2PP"]
    a += _ATENG_EXTRA.get(z, 0.0)
    return a


# --------------------------------------------------------------------------- #
# Backward-compatible s-shell helpers (H₂ intermediate-level tests).
# --------------------------------------------------------------------------- #


def gss(z: int) -> float:
    return one_center_2e(z)["GSS"]


def overlap_1s1s(z1: float, z2: float, R: float) -> float:
    return _ki.s2int(1, 0, 0, z1, 1, 0, 0, z2, R)


def coulomb_1s1s(z1: float, z2: float, R: float) -> float:
    return _ki.c2int(1, 0, 0, z1, 1, 0, 0, z2, R)


def nuclear_1s(z: float, R: float) -> float:
    return _ki.v2int(1, 0, 0, z, R)


def lmunu_ss(zk: float, zl: float, R: float, hk: bool, hl: bool) -> float:
    S = overlap_1s1s(zk, zl, R)
    dcor = -0.5 * (zk * zk + zl * zl) / (1.0 + 0.5 * (zk + zl) * R)
    mull = dcor * S * (1.0 - S)
    if hk or hl:
        dcort = 0.5 * (zk + zl) * R
        eterm = (1.0 - math.exp(-dcort)) / (1.0 + dcort)
        mull = (mull + (-S * eterm)) / 2.0
    return mull


# --------------------------------------------------------------------------- #
# Two-center core Hamiltonian -- per-pair assembly (intdrv.f flow).
# Local orbital order: 0=s, 1=ps, 2,3=ppi, 4=ds, 5,6=dpi, 7,8=dd (HARMTR cols).
# --------------------------------------------------------------------------- #


def _vfak(z_self: int, z_other: int) -> float:
    """REVFAK ortho prefactor -- f(other's max shell): s=1.0, p=0.75, d=0.5."""
    nb = n_basis(z_other)
    if nb >= 9:
        return 0.5
    if nb >= 4:
        return 0.75
    return 1.0


@dataclass
class _CoreH:
    HSS: float = 0.0
    HPS: float = 0.0
    HPP: float = 0.0
    HDS: float = 0.0
    HDP: float = 0.0
    HDD: float = 0.0


def _rumpf_overlaps(z_val: int, z_core: int, R: float) -> dict:
    """Valence(z_val)-core(z_core) "Rumpfintegrale" (valrum.f/rumval.f).

    The core is 1s for C-F (row 1), 1s2s2p for Na-Ar (row 2).  Returns the
    valence s/ps/ds overlaps with each core orbital.
    """
    ri = {
        k: 0.0
        for k in (
            "S1S",
            "S2S",
            "S2PS",
            "PS1S",
            "PS2S",
            "PS2PS",
            "PP2PP",
            "DS1S",
            "DS2S",
            "DS2PS",
            "DP2PP",
        )
    }
    if z_core <= 2:
        return ri
    val_p = n_basis(z_val) >= 4
    # A zero d exponent (e.g. Kr's MUD=0 polarization-only d) makes the d-on-the-
    # zero-side STO integrals singular; skip the d channel (-> 0, matching MSINDO,
    # where the empty null-extent d decouples) rather than divide by zero.
    val_d = n_basis(z_val) >= 9 and MUD.get(z_val, 0.0) != 0.0
    ns, npr = n_principal(z_val), n_p_principal(z_val)
    nd = n_d_principal(z_val)
    z1s = TAU1S[z_core]
    ri["S1S"] = _ki.s2int(ns, 0, 0, MUS[z_val], 1, 0, 0, z1s, R)
    if val_p:
        ri["PS1S"] = _ki.s2int(npr, 1, 0, MUP[z_val], 1, 0, 0, z1s, R)
    if val_d:
        ri["DS1S"] = _ki.s2int(nd, 2, 0, MUD[z_val], 1, 0, 0, z1s, R)
    if z_core > 10:  # row-2 core: add 2s, 2p (Ne core)
        z2s, z2p = TAU2S[z_core], TAU2P[z_core]
        ri["S2S"] = _ki.s2int(ns, 0, 0, MUS[z_val], 2, 0, 0, z2s, R)
        ri["S2PS"] = _ki.s2int(ns, 0, 0, MUS[z_val], 2, 1, 0, z2p, R)
        if val_p:
            ri["PS2S"] = _ki.s2int(npr, 1, 0, MUP[z_val], 2, 0, 0, z2s, R)
            ri["PS2PS"] = _ki.s2int(npr, 1, 0, MUP[z_val], 2, 1, 0, z2p, R)
            ri["PP2PP"] = _ki.s2int(npr, 1, 1, MUP[z_val], 2, 1, 1, z2p, R)
        if val_d:
            ri["DS2S"] = _ki.s2int(nd, 2, 0, MUD[z_val], 2, 0, 0, z2s, R)
            ri["DS2PS"] = _ki.s2int(nd, 2, 0, MUD[z_val], 2, 1, 0, z2p, R)
            ri["DP2PP"] = _ki.s2int(nd, 2, 1, MUD[z_val], 2, 1, 1, z2p, R)
    if z_core > 18:  # row-3 core: add 3s, 3p (Ar core, K onward)
        z3s, z3p = TAU3S[z_core], TAU3P[z_core]
        ri["S3S"] = _ki.s2int(ns, 0, 0, MUS[z_val], 3, 0, 0, z3s, R)
        ri["S3PS"] = _ki.s2int(ns, 0, 0, MUS[z_val], 3, 1, 0, z3p, R)
        if val_p:
            ri["PS3S"] = _ki.s2int(npr, 1, 0, MUP[z_val], 3, 0, 0, z3s, R)
            ri["PS3PS"] = _ki.s2int(npr, 1, 0, MUP[z_val], 3, 1, 0, z3p, R)
            ri["PP3PP"] = _ki.s2int(npr, 1, 1, MUP[z_val], 3, 1, 1, z3p, R)
        if val_d:
            ri["DS3S"] = _ki.s2int(nd, 2, 0, MUD[z_val], 3, 0, 0, z3s, R)
            ri["DS3PS"] = _ki.s2int(nd, 2, 0, MUD[z_val], 3, 1, 0, z3p, R)
            ri["DP3PP"] = _ki.s2int(nd, 2, 1, MUD[z_val], 3, 1, 1, z3p, R)
    if z_core > 30:  # [Ar]3d¹⁰ core (Ga-Kr onward): add the 3d core shell
        z3d = TAU3D[z_core]
        ri["S3DS"] = _ki.s2int(ns, 0, 0, MUS[z_val], 3, 2, 0, z3d, R)
        if val_p:
            ri["PS3DS"] = _ki.s2int(npr, 1, 0, MUP[z_val], 3, 2, 0, z3d, R)
            ri["PP3DP"] = _ki.s2int(npr, 1, 1, MUP[z_val], 3, 2, 1, z3d, R)
        if val_d:
            ri["DS3DS"] = _ki.s2int(nd, 2, 0, MUD[z_val], 3, 2, 0, z3d, R)
            ri["DP3DP"] = _ki.s2int(nd, 2, 1, MUD[z_val], 3, 2, 1, z3d, R)
            ri["DD3DD"] = _ki.s2int(nd, 2, 2, MUD[z_val], 3, 2, 2, z3d, R)
    if z_core > 36:  # [Kr] core (Rb-Xe onward): add the 4s, 4p core shells
        z4s, z4p = TAU4S[z_core], TAU4P[z_core]
        ri["S4S"] = _ki.s2int(ns, 0, 0, MUS[z_val], 4, 0, 0, z4s, R)
        ri["S4PS"] = _ki.s2int(ns, 0, 0, MUS[z_val], 4, 1, 0, z4p, R)
        if val_p:
            ri["PS4S"] = _ki.s2int(npr, 1, 0, MUP[z_val], 4, 0, 0, z4s, R)
            ri["PS4PS"] = _ki.s2int(npr, 1, 0, MUP[z_val], 4, 1, 0, z4p, R)
            ri["PP4PP"] = _ki.s2int(npr, 1, 1, MUP[z_val], 4, 1, 1, z4p, R)
        if val_d:
            ri["DS4S"] = _ki.s2int(nd, 2, 0, MUD[z_val], 4, 0, 0, z4s, R)
            ri["DS4PS"] = _ki.s2int(nd, 2, 0, MUD[z_val], 4, 1, 0, z4p, R)
            ri["DP4PP"] = _ki.s2int(nd, 2, 1, MUD[z_val], 4, 1, 1, z4p, R)
    if z_core > 48:  # [Kr]4d¹⁰ core (In-Xe): add the 4d core shell
        z4d = TAU4D[z_core]
        ri["S4DS"] = _ki.s2int(ns, 0, 0, MUS[z_val], 4, 2, 0, z4d, R)
        if val_p:
            ri["PS4DS"] = _ki.s2int(npr, 1, 0, MUP[z_val], 4, 2, 0, z4d, R)
            ri["PP4DP"] = _ki.s2int(npr, 1, 1, MUP[z_val], 4, 2, 1, z4d, R)
        if val_d:
            ri["DS4DS"] = _ki.s2int(nd, 2, 0, MUD[z_val], 4, 2, 0, z4d, R)
            ri["DP4DP"] = _ki.s2int(nd, 2, 1, MUD[z_val], 4, 2, 1, z4d, R)
            ri["DD4DD"] = _ki.s2int(nd, 2, 2, MUD[z_val], 4, 2, 2, z4d, R)
    return ri


def _v2core_one_side(z_val: int, z_core: int, R: float, gam: dict) -> _CoreH:
    """Core attraction of z_val's valence onto z_core's frozen core
    (V2CORE) + penetration (VCORRK).  ``gam`` holds monopole g(shell_val,
    shell_core) used by VCORR: keys 'ps','pp','ds','dp' g(p_val,s_core) etc."""
    zc = float(eff_core_charge(z_core))
    val_p = n_basis(z_val) >= 4
    # Skip the d channel for a zero-exponent valence d (Kr) -- see _rumpf_overlaps.
    val_d = n_basis(z_val) >= 9 and MUD.get(z_val, 0.0) != 0.0
    ns, npr, nd = n_principal(z_val), n_p_principal(z_val), n_d_principal(z_val)
    h = _CoreH()
    ri = _rumpf_overlaps(z_val, z_core, R)

    def fcp_sum(key1s, key2s, key2ps, key3s, key3ps, key3d=None,
                key4s=None, key4ps=None, key4d=None):
        s = 0.0
        if z_core > 2:
            s += FCP1S[z_core] * ri[key1s] ** 2
        if z_core > 10:
            s += FCP2S[z_core] * ri[key2s] ** 2 + FCP2P[z_core] * ri[key2ps] ** 2
        if z_core > 18:
            s += FCP3S[z_core] * ri[key3s] ** 2 + FCP3P[z_core] * ri[key3ps] ** 2
        if z_core > 30 and key3d is not None:  # [Ar]3d¹⁰ core: 3d shell
            s += FCP3D[z_core] * ri[key3d] ** 2
        if z_core > 36 and key4s is not None:  # [Kr] core: 4s, 4p shells
            s += FCP4S[z_core] * ri[key4s] ** 2 + FCP4P[z_core] * ri[key4ps] ** 2
        if z_core > 48 and key4d is not None:  # [Kr]4d¹⁰ core: 4d shell
            s += FCP4D[z_core] * ri[key4d] ** 2
        return s

    h.HSS = -zc * _ki.v2int(ns, 0, 0, MUS[z_val], R) + fcp_sum(
        "S1S", "S2S", "S2PS", "S3S", "S3PS", "S3DS", "S4S", "S4PS", "S4DS"
    )
    if val_p:
        h.HPS = -zc * _ki.v2int(npr, 1, 0, MUP[z_val], R)
        h.HPP = -zc * _ki.v2int(npr, 1, 1, MUP[z_val], R)
        # frozen core: only the s component overlaps spherical cores; the pi
        # component overlaps each np core (PPnPP) for Ne/Ar cores.
        if z_core > 2:
            h.HPS += FCP1S[z_core] * ri["PS1S"] ** 2
        if z_core > 10:
            h.HPS += FCP2S[z_core] * ri["PS2S"] ** 2 + FCP2P[z_core] * ri["PS2PS"] ** 2
            h.HPP += FCP2P[z_core] * ri["PP2PP"] ** 2
        if z_core > 18:
            h.HPS += FCP3S[z_core] * ri["PS3S"] ** 2 + FCP3P[z_core] * ri["PS3PS"] ** 2
            h.HPP += FCP3P[z_core] * ri["PP3PP"] ** 2
        if z_core > 30:
            h.HPS += FCP3D[z_core] * ri["PS3DS"] ** 2
            h.HPP += FCP3D[z_core] * ri["PP3DP"] ** 2
        if z_core > 36:
            h.HPS += FCP4S[z_core] * ri["PS4S"] ** 2 + FCP4P[z_core] * ri["PS4PS"] ** 2
            h.HPP += FCP4P[z_core] * ri["PP4PP"] ** 2
        if z_core > 48:
            h.HPS += FCP4D[z_core] * ri["PS4DS"] ** 2
            h.HPP += FCP4D[z_core] * ri["PP4DP"] ** 2
    if val_d:
        h.HDS = -zc * _ki.v2int(nd, 2, 0, MUD[z_val], R)
        h.HDP = -zc * _ki.v2int(nd, 2, 1, MUD[z_val], R)
        h.HDD = -zc * _ki.v2int(nd, 2, 2, MUD[z_val], R)
        if z_core > 2:
            h.HDS += FCP1S[z_core] * ri["DS1S"] ** 2
        if z_core > 10:
            h.HDS += FCP2S[z_core] * ri["DS2S"] ** 2 + FCP2P[z_core] * ri["DS2PS"] ** 2
            h.HDP += FCP2P[z_core] * ri["DP2PP"] ** 2
        if z_core > 18:
            h.HDS += FCP3S[z_core] * ri["DS3S"] ** 2 + FCP3P[z_core] * ri["DS3PS"] ** 2
            h.HDP += FCP3P[z_core] * ri["DP3PP"] ** 2
        if z_core > 30:  # the 3d core is the first to contribute to HDD (d-d)
            h.HDS += FCP3D[z_core] * ri["DS3DS"] ** 2
            h.HDP += FCP3D[z_core] * ri["DP3DP"] ** 2
            h.HDD += FCP3D[z_core] * ri["DD3DD"] ** 2
        if z_core > 36:
            h.HDS += FCP4S[z_core] * ri["DS4S"] ** 2 + FCP4P[z_core] * ri["DS4PS"] ** 2
            h.HDP += FCP4P[z_core] * ri["DP4PP"] ** 2
        if z_core > 48:
            h.HDS += FCP4D[z_core] * ri["DS4DS"] ** 2
            h.HDP += FCP4D[z_core] * ri["DP4DP"] ** 2
            h.HDD += FCP4D[z_core] * ri["DD4DD"] ** 2

    # --- VCORRK penetration: valence p/d sees z_core's valence electrons
    #     beyond the monopole g (vcorrk.f). ---
    if val_p:
        npc = n_principal(z_core)
        coul_pss = _ki.c2int(npc, 0, 0, MUS[z_core], npr, 1, 0, MUP[z_val], R)
        coul_psp = _ki.c2int(npc, 0, 0, MUS[z_core], npr, 1, 1, MUP[z_val], R)
        h.HPS += (coul_pss - gam["ps"]) * LS[z_core]
        h.HPP += (coul_psp - gam["ps"]) * LS[z_core]
        if MP[z_core] != 0:
            npcp = n_p_principal(z_core)
            coul_pps = _ki.c2int(npcp, 0, 0, MUP[z_core], npr, 1, 0, MUP[z_val], R)
            coul_ppp = _ki.c2int(npcp, 0, 0, MUP[z_core], npr, 1, 1, MUP[z_val], R)
            h.HPS += (coul_pps - gam["pp"]) * MP[z_core]
            h.HPP += (coul_ppp - gam["pp"]) * MP[z_core]
        if ND[z_core] != 0:  # penetration by the partner's valence d electrons
            ncd = n_d_principal(z_core)
            coul_pds = _ki.c2int(ncd, 0, 0, MUD[z_core], npr, 1, 0, MUP[z_val], R)
            coul_pdp = _ki.c2int(ncd, 0, 0, MUD[z_core], npr, 1, 1, MUP[z_val], R)
            h.HPS += (coul_pds - gam["pd"]) * ND[z_core]
            h.HPP += (coul_pdp - gam["pd"]) * ND[z_core]
    if val_d:
        npc = n_principal(z_core)
        coul_dss = _ki.c2int(npc, 0, 0, MUS[z_core], nd, 2, 0, MUD[z_val], R)
        coul_dsp = _ki.c2int(npc, 0, 0, MUS[z_core], nd, 2, 1, MUD[z_val], R)
        coul_dsd = _ki.c2int(npc, 0, 0, MUS[z_core], nd, 2, 2, MUD[z_val], R)
        h.HDS += (coul_dss - gam["ds"]) * LS[z_core]
        h.HDP += (coul_dsp - gam["ds"]) * LS[z_core]
        h.HDD += (coul_dsd - gam["ds"]) * LS[z_core]
        if MP[z_core] != 0:
            npcp = n_p_principal(z_core)
            coul_dps = _ki.c2int(npcp, 0, 0, MUP[z_core], nd, 2, 0, MUD[z_val], R)
            coul_dpp = _ki.c2int(npcp, 0, 0, MUP[z_core], nd, 2, 1, MUD[z_val], R)
            coul_dpd = _ki.c2int(npcp, 0, 0, MUP[z_core], nd, 2, 2, MUD[z_val], R)
            h.HDS += (coul_dps - gam["dp"]) * MP[z_core]
            h.HDP += (coul_dpp - gam["dp"]) * MP[z_core]
            h.HDD += (coul_dpd - gam["dp"]) * MP[z_core]
        if ND[z_core] != 0:  # penetration by the partner's valence d electrons
            ncd = n_d_principal(z_core)
            coul_dds = _ki.c2int(ncd, 0, 0, MUD[z_core], nd, 2, 0, MUD[z_val], R)
            coul_ddp = _ki.c2int(ncd, 0, 0, MUD[z_core], nd, 2, 1, MUD[z_val], R)
            coul_ddd = _ki.c2int(ncd, 0, 0, MUD[z_core], nd, 2, 2, MUD[z_val], R)
            h.HDS += (coul_dds - gam["dd"]) * ND[z_core]
            h.HDP += (coul_ddp - gam["dd"]) * ND[z_core]
            h.HDD += (coul_ddd - gam["dd"]) * ND[z_core]
    return h


def _lmunu_local(zk: int, zl: int, R: float, S: dict) -> dict:
    """Löwdin auxiliary L(mu,ν), local frame (lmunu.f).  S holds reduced local
    overlaps keyed 'ss','ps','sp','pp','pipi','ds','sd','dp','pd','dd','dpi'..."""
    zsK, zpK, zdK = MUS[zk], MUP[zk], MUD.get(zk, 0.0)
    zsL, zpL, zdL = MUS[zl], MUP[zl], MUD.get(zl, 0.0)
    kp, lp = n_basis(zk) >= 4, n_basis(zl) >= 4
    kd, ld = n_basis(zk) >= 9, n_basis(zl) >= 9

    def dcor(za, zb):
        return -0.5 * (za * za + zb * zb) / (1.0 + 0.5 * (za + zb) * R)

    M = {
        k: 0.0
        for k in (
            "ss",
            "ps",
            "sp",
            "pp",
            "pipi",
            "ds",
            "sd",
            "dp",
            "dppi",
            "pd",
            "pdpi",
            "dd",
            "ddpi",
            "dddel",
        )
    }
    M["ss"] = dcor(zsK, zsL) * S["ss"] * (1.0 - S["ss"])
    if kp and zk > 2:
        M["ps"] = dcor(zpK, zsL) * S["ps"] * (1.0 - S["ps"])
    if lp and zl > 2:
        M["sp"] = dcor(zsK, zpL) * S["sp"] * (1.0 + S["sp"])
    if kp and lp and zk > 2 and zl > 2:
        M["pp"] = dcor(zpK, zpL) * S["pp"] * (1.0 - abs(S["pp"]))
        M["pipi"] = dcor(zpK, zpL) * S["pipi"] * (1.0 - abs(S["pipi"]))
    if kd:  # d(K)/s(L) [(1-S), like p/s] + d(K)/p(L) s,pi
        M["ds"] = dcor(zdK, zsL) * S["ds"] * (1.0 - S["ds"])
        if lp:
            M["dp"] = dcor(zdK, zpL) * S["dp"] * (1.0 - abs(S["dp"]))
            M["dppi"] = dcor(zdK, zpL) * S["dppi"] * (1.0 - abs(S["dppi"]))
    if ld:  # s(K)/d(L) [(1-S)] + p(K)/d(L) s,pi
        M["sd"] = dcor(zsK, zdL) * S["sd"] * (1.0 - S["sd"])
        if kp:
            M["pd"] = dcor(zpK, zdL) * S["pd"] * (1.0 - abs(S["pd"]))
            M["pdpi"] = dcor(zpK, zdL) * S["pdpi"] * (1.0 - abs(S["pdpi"]))
    if kd and ld:  # d(K)/d(L) s,pi,d
        M["dd"] = dcor(zdK, zdL) * S["dd"] * (1.0 - abs(S["dd"]))
        M["ddpi"] = dcor(zdK, zdL) * S["ddpi"] * (1.0 - abs(S["ddpi"]))
        M["dddel"] = dcor(zdK, zdL) * S["dddel"] * (1.0 - abs(S["dddel"]))

    # 1s averaging when an H/He is involved (lmunu.f tail).
    if zk > 2 and zl > 2:
        return M

    def avg(za, zb, s):
        dcort = 0.5 * (za + zb) * R
        eterm = (1.0 - math.exp(-dcort)) / (1.0 + dcort)
        return -s * eterm

    if zk == 1 and zl > 2:
        M["ss"] = (M["ss"] + avg(zsK, zsL, S["ss"])) / 2.0
        M["sp"] = (M["sp"] + avg(zsK, zpL, S["sp"])) / 2.0
        if ld:
            M["sd"] = (M["sd"] + avg(zsK, zdL, S["sd"])) / 2.0
    elif zk > 2 and zl == 1:
        M["ss"] = (M["ss"] + avg(zsK, zsL, S["ss"])) / 2.0
        M["ps"] = (M["ps"] + avg(zpK, zsL, S["ps"])) / 2.0
        if kd:
            M["ds"] = (M["ds"] + avg(zdK, zsL, S["ds"])) / 2.0
    else:
        M["ss"] = (M["ss"] + avg(zsK, zsL, S["ss"])) / 2.0
    return M


@dataclass
class _PairBlocks:
    HK1: np.ndarray
    HL1: np.ndarray
    HKL2: np.ndarray
    R: float


def _harmtr(E):
    return _ki.harmtr(3, E)


def _pair_blocks(zk, zl, rk, rl):
    """Per-pair contribution to H⁰ (intdrv.f/rotint.f) as 9x9 global blocks."""
    d = np.asarray(rl, float) - np.asarray(rk, float)
    R = float(np.linalg.norm(d))
    E = d / R
    kp, lp = n_basis(zk) >= 4, n_basis(zl) >= 4
    kd, ld = n_basis(zk) >= 9, n_basis(zl) >= 9
    nk, nl = n_principal(zk), n_principal(zl)
    npk, npl = n_p_principal(zk), n_p_principal(zl)
    nkd, nld = n_d_principal(zk), n_d_principal(zl)  # d shell (!= ns/np for TMs)

    def s2(n1, l1, m1, e1, n2, l2, m2, e2):
        return _ki.s2int(n1, l1, m1, e1, n2, l2, m2, e2, R)

    # reduced local overlaps (overlap.f)
    S = {
        k: 0.0
        for k in (
            "ss",
            "ps",
            "sp",
            "pp",
            "pipi",
            "ds",
            "sd",
            "dp",
            "pd",
            "dd",
            "dppi",
            "pdpi",
            "ddpi",
            "dddel",
        )
    }
    S["ss"] = s2(nk, 0, 0, MUS[zk], nl, 0, 0, MUS[zl])
    if kp:
        S["ps"] = s2(npk, 1, 0, MUP[zk], nl, 0, 0, MUS[zl])
    if lp:
        S["sp"] = s2(nk, 0, 0, MUS[zk], npl, 1, 0, MUP[zl])
    if kp and lp:
        S["pp"] = s2(npk, 1, 0, MUP[zk], npl, 1, 0, MUP[zl])
        S["pipi"] = s2(npk, 1, 1, MUP[zk], npl, 1, 1, MUP[zl])
    if kd:
        S["ds"] = s2(nkd, 2, 0, MUD[zk], nl, 0, 0, MUS[zl])
        if lp:
            S["dp"] = s2(nkd, 2, 0, MUD[zk], npl, 1, 0, MUP[zl])
            S["dppi"] = s2(nkd, 2, 1, MUD[zk], npl, 1, 1, MUP[zl])
    if ld:
        S["sd"] = s2(nk, 0, 0, MUS[zk], nld, 2, 0, MUD[zl])
        if kp:
            S["pd"] = s2(npk, 1, 0, MUP[zk], nld, 2, 0, MUD[zl])
            S["pdpi"] = s2(npk, 1, 1, MUP[zk], nld, 2, 1, MUD[zl])
    if kd and ld:
        S["dd"] = s2(nkd, 2, 0, MUD[zk], nld, 2, 0, MUD[zl])
        S["ddpi"] = s2(nkd, 2, 1, MUD[zk], nld, 2, 1, MUD[zl])
        S["dddel"] = s2(nkd, 2, 2, MUD[zk], nld, 2, 2, MUD[zl])

    # monopole g(shell_K, s/p of L) for the VCORR penetration term (coulom.f)
    def c2(n1, e1, n2, e2):
        return _ki.c2int(n1, 0, 0, e1, n2, 0, 0, e2, R)

    gam_k = {
        "ps": c2(npk, MUP[zk], nl, MUS[zl]) if kp else 0.0,
        "pp": c2(npk, MUP[zk], npl, MUP[zl]) if (kp and lp) else 0.0,
        "pd": c2(npk, MUP[zk], nld, MUD[zl]) if (kp and ld) else 0.0,
        "ds": c2(nkd, MUD[zk], nl, MUS[zl]) if kd else 0.0,
        "dp": c2(nkd, MUD[zk], npl, MUP[zl]) if (kd and lp) else 0.0,
        "dd": c2(nkd, MUD[zk], nld, MUD[zl]) if (kd and ld) else 0.0,
    }
    gam_l = {
        "ps": c2(npl, MUP[zl], nk, MUS[zk]) if lp else 0.0,
        "pp": c2(npl, MUP[zl], npk, MUP[zk]) if (kp and lp) else 0.0,
        "pd": c2(npl, MUP[zl], nkd, MUD[zk]) if (lp and kd) else 0.0,
        "ds": c2(nld, MUD[zl], nk, MUS[zk]) if ld else 0.0,
        "dp": c2(nld, MUD[zl], npk, MUP[zk]) if (kp and ld) else 0.0,
        "dd": c2(nld, MUD[zl], nkd, MUD[zk]) if (ld and kd) else 0.0,
    }

    hk = _v2core_one_side(zk, zl, R, gam_k)
    hl = _v2core_one_side(zl, zk, R, gam_l)

    ek, el = eneg(zk), eneg(zl)
    shk = {
        "ss": ek[0] + hk.HSS,
        "ps": ek[1] + hk.HPS,
        "pp": ek[1] + hk.HPP,
        "ds": ek[2] + hk.HDS,
        "dp": ek[2] + hk.HDP,
        "dd": ek[2] + hk.HDD,
    }
    shl = {
        "ss": el[0] + hl.HSS,
        "ps": el[1] + hl.HPS,
        "pp": el[1] + hl.HPP,
        "ds": el[2] + hl.HDS,
        "dp": el[2] + hl.HDP,
        "dd": el[2] + hl.HDD,
    }

    M = _lmunu_local(zk, zl, R, S)

    # HORTH: H** -= VFAK.S.L (horth.f)
    vk, vl = _vfak(zk, zl), _vfak(zl, zk)
    hk.HSS -= vk * S["ss"] * M["ss"]
    hl.HSS -= vl * S["ss"] * M["ss"]
    if lp:
        t = S["sp"] * M["sp"]
        hk.HSS -= vk * t
        hl.HPS -= vl * t
    if kp:
        t = S["ps"] * M["ps"]
        hk.HPS -= vk * t
        hl.HSS -= vl * t
    if kp and lp:
        hk.HPS -= vk * S["pp"] * M["pp"]
        hl.HPS -= vl * S["pp"] * M["pp"]
        hk.HPP -= vk * S["pipi"] * M["pipi"]
        hl.HPP -= vl * S["pipi"] * M["pipi"]
    if ld:  # s(K)<->d(L)
        t = S["sd"] * M["sd"]
        hk.HSS -= vk * t
        hl.HDS -= vl * t
    if kd:  # d(K)<->s(L)
        t = S["ds"] * M["ds"]
        hk.HDS -= vk * t
        hl.HSS -= vl * t
    if kp and ld:  # p(K)<->d(L): s + pi
        hk.HPS -= vk * S["pd"] * M["pd"]
        hl.HDS -= vl * S["pd"] * M["pd"]
        hk.HPP -= vk * S["pdpi"] * M["pdpi"]
        hl.HDP -= vl * S["pdpi"] * M["pdpi"]
    if kd and lp:  # d(K)<->p(L): s + pi
        hk.HDS -= vk * S["dp"] * M["dp"]
        hl.HPS -= vl * S["dp"] * M["dp"]
        hk.HDP -= vk * S["dppi"] * M["dppi"]
        hl.HPP -= vl * S["dppi"] * M["dppi"]
    if kd and ld:  # d(K)<->d(L): s + pi + d
        hk.HDS -= vk * S["dd"] * M["dd"]
        hl.HDS -= vl * S["dd"] * M["dd"]
        hk.HDP -= vk * S["ddpi"] * M["ddpi"]
        hl.HDP -= vl * S["ddpi"] * M["ddpi"]
        hk.HDD -= vk * S["dddel"] * M["dddel"]
        hl.HDD -= vl * S["dddel"] * M["dddel"]

    # DELTAH resonance (deltah.f), local frame, with +L(mu,ν).
    fack = 1.0 - math.exp(-AL(zk, zl) * R)
    facl = 1.0 - math.exp(-AL(zl, zk) * R)
    core = np.zeros((9, 9))

    def reson(kk_k, kk_l, sval, sh_k, sh_l, mval):
        return 0.25 * (kk_k + kk_l) * sval * (fack * sh_k + facl * sh_l) + mval

    core[0, 0] = reson(KSS[zk], KSS[zl], S["ss"], shk["ss"], shl["ss"], M["ss"])
    if kp:
        core[1, 0] = reson(KPS[zk], KSS[zl], S["ps"], shk["ps"], shl["ss"], M["ps"])
    if lp:
        core[0, 1] = reson(KSS[zk], KPS[zl], S["sp"], shk["ss"], shl["ps"], M["sp"])
    if kp and lp:
        core[1, 1] = reson(KPS[zk], KPS[zl], S["pp"], shk["ps"], shl["ps"], M["pp"])
        cpp = reson(KPP[zk], KPP[zl], S["pipi"], shk["pp"], shl["pp"], M["pipi"])
        core[2, 2] = core[3, 3] = cpp
    if kd:
        core[4, 0] = reson(KDS[zk], KSS[zl], S["ds"], shk["ds"], shl["ss"], M["ds"])
    if ld:
        core[0, 4] = reson(KSS[zk], KDS[zl], S["sd"], shk["ss"], shl["ds"], M["sd"])
    if kd and lp:
        core[4, 1] = reson(KDS[zk], KPS[zl], S["dp"], shk["ds"], shl["ps"], M["dp"])
        cdp = reson(KDP[zk], KPP[zl], S["dppi"], shk["dp"], shl["pp"], M["dppi"])
        core[5, 2] = core[6, 3] = cdp
    if kp and ld:
        core[1, 4] = reson(KPS[zk], KDS[zl], S["pd"], shk["ps"], shl["ds"], M["pd"])
        cpd = reson(KPP[zk], KDP[zl], S["pdpi"], shk["pp"], shl["dp"], M["pdpi"])
        core[2, 5] = core[3, 6] = cpd
    if kd and ld:
        core[4, 4] = reson(KDS[zk], KDS[zl], S["dd"], shk["ds"], shl["ds"], M["dd"])
        cddpi = reson(KDP[zk], KDP[zl], S["ddpi"], shk["dp"], shl["dp"], M["ddpi"])
        core[5, 5] = core[6, 6] = cddpi
        cdddel = reson(KDD[zk], KDD[zl], S["dddel"], shk["dd"], shl["dd"], M["dddel"])
        core[7, 7] = core[8, 8] = cdddel

    # rotate local -> global (ROTINT/TINT == T.M.Tᵀ with T=HARMTR)
    Traw = _harmtr(E)
    T = np.asarray(Traw)
    diag_k = np.diag(
        [hk.HSS, hk.HPS, hk.HPP, hk.HPP, hk.HDS, hk.HDP, hk.HDP, hk.HDD, hk.HDD]
    )
    diag_l = np.diag(
        [hl.HSS, hl.HPS, hl.HPP, hl.HPP, hl.HDS, hl.HDP, hl.HDP, hl.HDD, hl.HDD]
    )
    return _PairBlocks(T @ diag_k @ T.T, T @ diag_l @ T.T, T @ core @ T.T, R)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


@dataclass
class MsindoResult:
    total_energy: float = 0.0
    electronic_energy: float = 0.0
    binding_energy: float = 0.0
    mo_energies: np.ndarray = field(default=None)
    density: np.ndarray = field(default=None)
    n_iter: int = 0
    converged: bool = False


@lru_cache(maxsize=2)
def _cpp_msindo_kernel(*, nddo: bool = False):
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    try:
        load_params_from_json = _indo.load_params_from_json
        merge_nddo_overrides = _indo.merge_nddo_overrides
        run_msindo_full = _indo.run_msindo_full
        run_msindo_uhf = _indo.run_msindo_uhf
    except AttributeError:
        return None

    params = load_params_from_json(
        Path(__file__).with_name("msindo_params.json").read_text()
    )
    if nddo:
        merge_nddo_overrides(
            params, Path(__file__).with_name("msindo_params_nddo.json").read_text()
        )
    return run_msindo_full, run_msindo_uhf, params


def _msindo_result_from_cpp(result_obj, Z, *, nddo: bool = False) -> MsindoResult:
    total = float(result_obj.total_energy)
    if nddo:
        binding = total - sum(ateng(z) + _NDDO_ATENG_CORR.get(z, 0.0) for z in Z)
    else:
        binding = float(result_obj.binding_energy)
    return MsindoResult(
        total,
        float(result_obj.electronic_energy),
        binding,
        np.asarray(result_obj.mo_energies, dtype=float).copy(),
        np.asarray(result_obj.density, dtype=float).copy(),
        int(result_obj.n_iter),
        bool(result_obj.converged),
    )


def _atom_blocks(atomic_numbers):
    blocks, nsto = [], 0
    for z in atomic_numbers:
        nb = n_basis(z)
        blocks.append((nsto, nsto + nb))
        nsto += nb
    return blocks, nsto


def _gamma_shell(za, zb, sa, sb, R):
    """Monopole two-center g between shell ``sa`` on za and ``sb`` on zb."""

    def na(z, s):
        return (
            n_principal(z)
            if s == "s"
            else n_p_principal(z)
            if s == "p"
            else n_d_principal(z)
        )

    def exp(z, s):
        return MUS[z] if s == "s" else MUP[z] if s == "p" else MUD[z]

    return _ki.c2int(na(za, sa), 0, 0, exp(za, sa), na(zb, sb), 0, 0, exp(zb, sb), R)


def _build_core_and_gamma(Z, C, blocks, nsto, *, nddo=False):
    """Core Hamiltonian H + monopole g matrix.

    With ``nddo=True`` (call inside ``_nddo_params()``), the only addition to the
    INDO core is the NDDO **HSP** s-ps core coupling (``v2core.f:281/655``): for
    each atom K with a p shell and partner L,
    ``hsp_KL = -ZC_L.spss_si(z_K, z_L, R)`` rotated along ``E = (C_L-C_K)/R`` into
    K's one-centre (s,ps) block (summed over partners).  Everything else (ENEG,
    V2INT monopole core, VCORRK penetration, resonance) follows from the NDDO
    parameter overrides -- validated to reproduce the oracle CORE HAMILTONIAN."""
    natom = len(Z)
    H = np.zeros((nsto, nsto))
    G = np.zeros((nsto, nsto))
    for i, z in enumerate(Z):
        lo, hi = blocks[i]
        u = eneg(z)
        nb = hi - lo
        H[lo, lo] += u[0]
        if nb >= 4:
            for p in range(1, 4):
                H[lo + p, lo + p] += u[1]
        if nb >= 9:
            for d in range(4, 9):
                H[lo + d, lo + d] += u[2]
        G[lo:hi, lo:hi] = one_center_gmunu(z)[:nb, :nb]

    shells = ["s", "p", "p", "p", "d", "d", "d", "d", "d"]
    for a in range(natom):
        for b in range(a + 1, natom):
            la, ha = blocks[a]
            lb, hb = blocks[b]
            pb = _pair_blocks(Z[a], Z[b], C[a], C[b])
            na_, nb_ = ha - la, hb - lb
            H[la:ha, la:ha] += pb.HK1[:na_, :na_]
            H[lb:hb, lb:hb] += pb.HL1[:nb_, :nb_]
            H[la:ha, lb:hb] += pb.HKL2[:na_, :nb_]
            H[lb:hb, la:ha] += pb.HKL2[:na_, :nb_].T
            za, zb, R = Z[a], Z[b], pb.R
            gcache = {}
            for ia in range(na_):
                for ib in range(nb_):
                    key = (shells[ia], shells[ib])
                    if key not in gcache:
                        gcache[key] = _gamma_shell(za, zb, key[0], key[1], R)
                    G[la + ia, lb + ib] = G[lb + ib, la + ia] = gcache[key]

    if nddo:
        # NDDO HSP s-ps core coupling (the one core term beyond INDO; v2core.f).
        for a, za in enumerate(Z):
            lo, hi = blocks[a]
            if hi - lo < 4:  # no p shell (H/He) -> no dipole, no HSP
                continue
            for b, zb in enumerate(Z):
                if a == b:
                    continue
                d = C[b] - C[a]
                R = float(np.linalg.norm(d))
                E = d / R
                hsp = -eff_core_charge(zb) * nddo_spss_si(za, zb, R)
                for k in range(3):
                    H[lo + 1 + k, lo] += hsp * E[k]
                    H[lo, lo + 1 + k] += hsp * E[k]
    return H, G


def _nddo_fock_extra(Z, C, blocks):
    """Build the NDDO two-centre multipole Fock addition as a ``_scf_rhf``
    ``fock_extra`` callback ``f(P) -> (F_add, 0.0)``.

    Two additive contributions, on top of the standard (monopole) INDO Fock,
    exactly as MSINDO's ``fockcl.f:214`` does ``IF(NDDO) CALL NDDOFOCKCL; CALL
    SPSPFOCKCL``:

    * ``nddofockcl.f`` -- dipole-monopole (s-p transition density <-> s/p/d
      monopoles),
    * ``spspfockcl.f`` -- dipole-dipole (s-p <-> s-p).

    Both routines fill only the lower triangle + diagonal; we mirror lower->upper
    before returning.  Energy rides in via 1/2Tr[P(H+F)] so ``e_add = 0``.  Must be
    called inside ``_nddo_params()`` (the kernels read the NDDO exponents).

    NOTE (parity status): ``run_msindo(nddo=True)`` reproduces the HF / H₂O / CH₄ /
    N₂ / CO / AlCl reference totals to <=1 µHa. For the s/p scope, the AO
    tensor :func:`_nddo_ao_eri` contracts (J - 1/2K) to this Fock to machine
    precision. The source's d-shell ``SPDD`` terms are Coulomb-only Fock branches,
    not ordinary eightfold-symmetric ERIs; see :func:`_nddo_ao_eri`.
    """
    natom = len(Z)

    def fock(P):
        nsto = P.shape[0]
        F = np.zeros((nsto, nsto))
        for I in range(natom):
            loI, hiI = blocks[I]
            zI = Z[I]
            pI = (hiI - loI) >= 4
            dI = (hiI - loI) >= 9
            for J in range(natom):
                if I == J:
                    continue
                loJ, hiJ = blocks[J]
                zJ = Z[J]
                pJ = (hiJ - loJ) >= 4
                dJ = (hiJ - loJ) >= 9
                d = C[J] - C[I]
                R = float(np.linalg.norm(d))
                E = d / R
                # dipole-monopole integrals (first arg = dipole-bearing atom).
                SSSP = nddo_spss_si(zJ, zI, R)  # J dipole, I s-monopole
                SPSS = nddo_spss_si(zI, zJ, R)  # I dipole, J s-monopole
                SPPP = nddo_sppp_si(zI, zJ, R) if pJ else 0.0  # J p-monopole
                PPSP = nddo_sppp_si(zJ, zI, R) if pI else 0.0  # I p-monopole
                DDSP = nddo_spdd_si(zJ, zI, R) if pJ and dI else 0.0
                SPDD = nddo_spdd_si(zI, zJ, R) if pI and dJ else 0.0
                pdipJ = (
                    sum(P[loJ + 1 + a, loJ] * E[a] for a in range(3))
                    if pJ
                    else 0.0
                )
                # --- nddofockcl: diagonal (Coulomb) ---
                if pJ:
                    F[loI, loI] -= 2.0 * SSSP * pdipJ
                    if pI:
                        for a in range(3):
                            F[loI + 1 + a, loI + 1 + a] -= 2.0 * PPSP * pdipJ
                    if dI:
                        for mu in range(loI + 4, hiI):
                            F[mu, mu] -= 2.0 * DDSP * pdipJ
                # --- nddofockcl + spspfockcl: one-centre (s,ps) on I ---
                if pI:
                    PssJ = P[loJ, loJ]
                    PppJ = (
                        sum(P[loJ + 1 + b, loJ + 1 + b] for b in range(3))
                        if pJ
                        else 0.0
                    )
                    for a in range(3):
                        F[loI + 1 + a, loI] += E[a] * (SPSS * PssJ + SPPP * PppJ)
                    if dJ:
                        PddJ = sum(P[loJ + b, loJ + b] for b in range(4, 9))
                        for a in range(3):
                            F[loI + 1 + a, loI] += E[a] * SPDD * PddJ
                if pI and pJ:
                    SI = nddo_spsp_si(zJ, zI, R)
                    PI = nddo_spsp_pi(zJ, zI, R)
                    S = [
                        [
                            E[a] * E[b] * SI
                            + ((1.0 if a == b else 0.0) - E[a] * E[b]) * PI
                            for b in range(3)
                        ]
                        for a in range(3)
                    ]
                    for a in range(3):
                        F[loI + 1 + a, loI] += 2.0 * sum(
                            P[loJ + 1 + b, loJ] * S[a][b] for b in range(3)
                        )
                # --- two-centre exchange (lower triangle, I > J) ---
                if I > J:
                    if pI:
                        F[loI, loJ] -= (
                            0.5
                            * SPSS
                            * sum(E[a] * P[loI + 1 + a, loJ] for a in range(3))
                        )
                    if pJ:
                        F[loI, loJ] += (
                            0.5
                            * SSSP
                            * sum(E[a] * P[loI, loJ + 1 + a] for a in range(3))
                        )
                        for b in range(3):
                            F[loI, loJ + 1 + b] += 0.5 * SSSP * E[b] * P[loI, loJ]
                    if pI and pJ:
                        for b in range(3):
                            F[loI, loJ + 1 + b] -= (
                                0.5
                                * SPPP
                                * sum(
                                    E[a] * P[loI + 1 + a, loJ + 1 + b] for a in range(3)
                                )
                            )
                    if pI:
                        for a in range(3):
                            F[loI + 1 + a, loJ] -= 0.5 * SPSS * E[a] * P[loI, loJ]
                    if pI and pJ:
                        for a in range(3):
                            F[loI + 1 + a, loJ] += (
                                0.5
                                * PPSP
                                * sum(
                                    E[b] * P[loI + 1 + a, loJ + 1 + b] for b in range(3)
                                )
                            )
                        for a in range(3):
                            for b in range(3):
                                # nddofockcl.f:165-191 -- the SPPP term is
                                # subtracted, the PPSP term *added* (distinct
                                # signs; only fires when both atoms have p).
                                F[loI + 1 + a, loJ + 1 + b] += (
                                    -0.5 * SPPP * E[a] * P[loI, loJ + 1 + b]
                                    + 0.5 * PPSP * E[b] * P[loI + 1 + a, loJ]
                                )
                        # spspfockcl two-centre exchange.
                        F[loI, loJ] -= 0.5 * sum(
                            P[loI + 1 + a, loJ + 1 + b] * S[a][b]
                            for a in range(3)
                            for b in range(3)
                        )
                        for b in range(3):
                            F[loI, loJ + 1 + b] -= 0.5 * sum(
                                P[loI + 1 + a, loJ] * S[a][b] for a in range(3)
                            )
                        for a in range(3):
                            F[loI + 1 + a, loJ] -= 0.5 * sum(
                                P[loI, loJ + 1 + b] * S[a][b] for b in range(3)
                            )
                        for a in range(3):
                            for b in range(3):
                                F[loI + 1 + a, loJ + 1 + b] -= (
                                    0.5 * P[loI, loJ] * S[a][b]
                                )
        F = np.tril(F) + np.tril(F, -1).T  # mirror lower -> upper
        return F, 0.0

    return fock


def _core_repulsion(C, cz):
    """Point-charge core-core repulsion S_{a<b} CZ_a.CZ_b / R_ab (geometry-only)."""
    e_core = 0.0
    natom = len(cz)
    for a in range(natom):
        for b in range(a + 1, natom):
            R = float(np.linalg.norm(C[a] - C[b]))
            e_core += cz[a] * cz[b] / R
    return e_core


# NDDO atomic-energy corrections (atomic_reference.f:199, IF(NDDO)) -- applied to
# ATENG for binding energies; affects only Al/Si/S/Cl (not the H-O reference set,
# whose totals are unaffected).
_NDDO_ATENG_CORR = {
    13: -0.0049403836,
    14: -0.0048727978,
    15: 0.0,
    16: -0.0056880155,
    17: -0.0062292160,
}


def _run_nddo(Z, coords_angstrom, *, charge, max_iter, conv_tol):
    """Closed-shell (RHF) MSINDO **NDDO** SCF -- the reference program's default
    mode.  Runs entirely under ``_nddo_params()`` (the separate NDDO
    parametrization, nddoparam.f), with the HSP one-centre core term
    (``_build_core_and_gamma(nddo=True)``) and the two-centre multipole 2e Fock
    (``_nddo_fock_extra``: s-p/s-p and s-p/d-d couplings). Reproduces reference
    MSINDO NDDO total energies to <=1 µHa (HF / H₂O / N₂ / CO / CH₄ / AlCl)."""
    with _nddo_params():
        C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
        blocks, nsto = _atom_blocks(Z)
        cz = [eff_core_charge(z) for z in Z]
        nelec = sum(cz) - charge
        if nelec < 0:
            raise ValueError(f"charge={charge} exceeds the {sum(cz)} valence electrons")
        if nelec % 2 != 0:
            raise NotImplementedError(
                "MSINDO NDDO mode is closed-shell (RHF) only; the valence-electron "
                "count is odd (open-shell NDDO / UHF is not implemented)."
            )
        H, G = _build_core_and_gamma(Z, C, blocks, nsto, nddo=True)
        e_core = _core_repulsion(C, cz)
        nocc = nelec // 2
        fock = _nddo_fock_extra(Z, C, blocks)
        P, _F, e_elec, eps, converged, it = _scf_rhf(
            H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol, fock_extra=fock
        )
        total = e_elec + e_core
        binding = total - sum(ateng(z) + _NDDO_ATENG_CORR.get(z, 0.0) for z in Z)
    return MsindoResult(total, e_elec, binding, eps, P, it, converged)


def run_msindo(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=1,
    max_iter=200,
    conv_tol=1e-9,
    nddo=False,
):
    """MSINDO SCF -- closed-shell RHF (s/p/d) or open-shell UHF (s/p).

    ``multiplicity`` = 2S+1; a value > 1 (or an odd valence-electron count)
    selects the spin-unrestricted (UHF) path.  ``charge`` shifts the valence
    electron count.  ``nddo=True`` selects MSINDO's **NDDO mode** (the default
    mode of the reference program) -- a separate parametrization (nddoparam.f)
    plus the HSP one-centre core term and the two-centre multipole 2e Fock
    (including the source's asymmetric s-p/d-d ``SPDD`` terms). Closed-shell
    (RHF) only, parametrized for H, Li-F, Na-Cl. The validated scope reproduces
    reference MSINDO NDDO to <=1 µHa (HF / H₂O / N₂ / CO / CH₄ / AlCl).
    """
    from vibeqc.semiempirical.routes import SemiempiricalRoutePlan

    SemiempiricalRoutePlan.from_request(
        "msindo",
        boundary="molecule",
        charge=int(charge),
        multiplicity=int(multiplicity),
        nddo=bool(nddo),
    )
    Z = list(atomic_numbers)
    if nddo:
        bad = sorted({z for z in Z if z not in _SUPPORTED_NDDO})
        if bad:
            raise NotImplementedError(
                f"MSINDO NDDO mode is parametrized for {sorted(_SUPPORTED_NDDO)} "
                f"(H, Li-F, Na-Cl); got Z={bad}."
            )
        if multiplicity != 1:
            raise NotImplementedError(
                "MSINDO NDDO mode is closed-shell (RHF) only; open-shell NDDO "
                "(UHF) is not implemented. Use multiplicity=1."
            )
        cz = [eff_core_charge(z) for z in Z]
        nelec = sum(cz) - charge
        if nelec < 0:
            raise ValueError(f"charge={charge} exceeds the {sum(cz)} valence electrons")
        if nelec % 2 != 0:
            raise NotImplementedError(
                "MSINDO NDDO mode is closed-shell (RHF) only; the valence-electron "
                "count is odd (open-shell NDDO / UHF is not implemented)."
            )
        kernel = _cpp_msindo_kernel(nddo=True)
        if kernel is not None:
            run_msindo_full, _run_msindo_uhf, params = kernel
            result_obj = run_msindo_full(
                Z,
                coords_angstrom,
                params,
                max_iter=max_iter,
                conv_tol=conv_tol,
                nddo=True,
                charge=charge,
            )
            if getattr(result_obj, "converged", False):
                return _msindo_result_from_cpp(result_obj, Z, nddo=True)
        return _run_nddo(
            Z, coords_angstrom, charge=charge, max_iter=max_iter, conv_tol=conv_tol
        )
    missing = sorted({z for z in Z if z not in _SUPPORTED})
    if missing:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(_SUPPORTED)}; got Z={missing}. "
            "Extend the parameter tables (datas.f) -- see docs/user_guide/msindo.md."
        )
    C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(Z)
    natom = len(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec < 0:
        raise ValueError(f"charge={charge} exceeds the {sum(cz)} valence electrons")
    open_shell = multiplicity != 1 or nelec % 2 != 0
    if open_shell:
        _uhf_occupation(nelec, multiplicity)

    kernel = _cpp_msindo_kernel(nddo=False)
    if kernel is not None:
        run_msindo_full, run_msindo_uhf, params = kernel
        if open_shell:
            result_obj = run_msindo_uhf(
                Z,
                coords_angstrom,
                params,
                max_iter=max_iter,
                conv_tol=conv_tol,
                multiplicity=multiplicity,
                nddo=False,
                charge=charge,
            )
        else:
            result_obj = run_msindo_full(
                Z,
                coords_angstrom,
                params,
                max_iter=max_iter,
                conv_tol=conv_tol,
                nddo=False,
                charge=charge,
            )
        if getattr(result_obj, "converged", False):
            return _msindo_result_from_cpp(result_obj, Z, nddo=False)

    H, G = _build_core_and_gamma(Z, C, blocks, nsto)
    e_core = _core_repulsion(C, cz)

    if open_shell:
        return _run_uhf(
            Z, H, G, blocks, nelec, multiplicity, e_core, max_iter, conv_tol
        )

    nocc = nelec // 2
    P, _F, e_elec, eps, converged, it = _scf_rhf_molecular(
        H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol
    )

    total = e_elec + e_core
    binding = total - sum(ateng(z) for z in Z)
    return MsindoResult(total, e_elec, binding, eps, P, it, converged)


def _scf_rhf(
    H,
    G,
    blocks,
    Z,
    nocc,
    *,
    max_iter=200,
    conv_tol=1e-9,
    fock_extra=None,
    initial_density=None,
):
    """Closed-shell (RHF) MSINDO SCF with Pulay DIIS.

    Shared by the molecular (``run_msindo``) and periodic-CCM
    (``msindo_ccm.run_ccm``) paths -- the only difference between them is how the
    core Hamiltonian ``H``, two-electron ``G`` and (outside this function) the
    nuclear repulsion are assembled; the self-consistency machinery is identical.
    Returns ``(P, F, e_elec, eps, converged, n_iter)``.

    ``fock_extra``, when given, is a callback ``f(P) -> (F_add, e_add)`` that adds
    a density-dependent term to the Fock each iteration and a matching scalar to
    the electronic energy.  It carries the CCM Madelung embedding (a diagonal
    ``-MADELATOM`` shift + the 1/2.S q.V energy); ``None`` (the molecular default)
    leaves this path byte-for-byte unchanged.

    Pulay DIIS (Chem. Phys. Lett. 73, 393 (1980)).  The MSINDO basis is the
    symmetrically (Löwdin) orthogonalised one -- the resonance integrals fold
    S^-1/2 in -- so the overlap is the identity and the SCF error vector is the
    bare commutator e = F P - P F. DIIS finds a stationary point [F,P]=0;
    this condition alone does not certify a local minimum or a unique SCF
    root. The CCM adapter additionally checks restricted orbital stability
    when an unresolved initial frontier makes its starting projector ambiguous.
    """

    def _fock(P):
        F = _build_fock(H, G, P, blocks, Z)
        e_add = 0.0
        if fock_extra is not None:
            F_add, e_add = fock_extra(P)
            F = F + F_add
        return F, e_add

    eps, Cmo = np.linalg.eigh(H)
    P = (
        2.0 * Cmo[:, :nocc] @ Cmo[:, :nocc].T
        if initial_density is None
        else np.asarray(initial_density, dtype=float).copy()
    )
    e_elec, converged, it = 0.0, False, 0
    f_hist, e_hist = [], []
    DIIS_MAX = 8
    for it in range(1, max_iter + 1):
        F, e_add = _fock(P)
        # SCF residual = commutator [F,P] in the Löwdin-orthonormal basis.  This
        # is the true convergence measure: the step size |P_new - P_old| is
        # unreliable under DIIS (a small extrapolation step is not a converged
        # density) and would let the two backends stop ~1e-4 apart in P.
        err = F @ P - P @ F
        e_cur = 0.5 * float(np.sum(P * (H + F))) + e_add
        if np.max(np.abs(err)) < conv_tol and abs(e_cur - e_elec) < conv_tol:
            e_elec, converged = e_cur, True
            break
        e_elec = e_cur
        f_hist.append(F)
        e_hist.append(err)
        if len(f_hist) > DIIS_MAX:
            f_hist.pop(0)
            e_hist.pop(0)
        n = len(f_hist)
        F_eff = F
        if n >= 2:
            B = np.full((n + 1, n + 1), -1.0)
            B[n, n] = 0.0
            for i in range(n):
                for j in range(i, n):
                    B[i, j] = B[j, i] = float(np.sum(e_hist[i] * e_hist[j]))
            rhs = np.zeros(n + 1)
            rhs[n] = -1.0
            try:
                c = np.linalg.solve(B, rhs)
                F_eff = sum(c[i] * f_hist[i] for i in range(n))
            except np.linalg.LinAlgError:
                F_eff = F
        eps, Cmo = np.linalg.eigh(F_eff)
        P = 2.0 * Cmo[:, :nocc] @ Cmo[:, :nocc].T

    # Self-consistent energy + MO eigenvalues at the final density,
    # 0.5.Tr[P(H+F(P))].  Recomputing from the converged P's own Fock makes the
    # reported energy second-order in the SCF residual -- identical across SCF
    # accelerators and across the C++/Python backends -- and also covers the
    # non-converged max_iter exit.
    F, e_add = _fock(P)
    e_elec = 0.5 * float(np.sum(P * (H + F))) + e_add
    eps = np.linalg.eigh(F)[0]
    return P, F, e_elec, eps, converged, it


def _huckel_guess_hamiltonian(H, Z, blocks):
    """MSINDO extended-Hückel guess Hamiltonian (huckcl.f / huckop.f, IDEN=0).

    The guess Hamiltonian is the INDO core ``H`` with (a) the one-centre
    (same-atom) off-diagonal blocks zeroed and (b) the diagonal replaced by the
    *bare* ionization potentials Q (IPOTS/IPOTP/IPOTD per shell), leaving the
    two-centre resonance off-diagonals intact (huckcl.f:178-187 / huckop.f:153-160;
    the ``U(I)==U(J)`` test there is on the per-orbital atom index --
    ``setpar.f:66 U(J)=I`` -- i.e. same atom).  This is the physically decisive
    piece for the heavy elements: for the 5th row IPOT=0 (datas.f), so the metal
    s/p/d sit at 0 (high) while the electronegative ligand orbitals (IPOT<0) sit
    low -- the aufbau fill then keeps the diffuse / high-lying metal d *out* of the
    occupied space.  vibe's DIIS path starts from the Hcore guess (``eigh(H)``),
    whose deeply-bound ENEG U on the diagonal pulls that d *into* the occupied
    space, converging to a different (non-MSINDO) stationary point."""
    A = H.copy()
    for i, z in enumerate(Z):
        lo, hi = blocks[i]
        A[lo:hi, lo:hi] = 0.0  # zero the one-centre (same-atom) block
        nb = hi - lo
        A[lo, lo] = IPOTS[z]
        if nb >= 4:
            for p in range(1, 4):
                A[lo + p, lo + p] = IPOTP[z]
        if nb >= 9:
            for d in range(4, 9):
                A[lo + d, lo + d] = IPOTD[z]
    return A


def _huckel_guess_density(H, Z, blocks, nocc):
    """Closed-shell extended-Hückel start density (huckcl.f, IDEN=0): aufbau-fill
    the lowest ``nocc`` eigenvectors of the guess Hamiltonian, doubly occupied."""
    _eps, Cmo = np.linalg.eigh(_huckel_guess_hamiltonian(H, Z, blocks))
    return 2.0 * Cmo[:, :nocc] @ Cmo[:, :nocc].T


def _huckel_guess_density_uhf(H, Z, blocks, nalpha, nbeta):
    """Open-shell (UHF) extended-Hückel start densities (huckop.f, IDEN=0).

    Same guess Hamiltonian as the closed-shell huckcl.f path, but PA is built from
    the first ``nalpha`` and PB from the first ``nbeta`` columns of the *same*
    coefficient matrix (huckop.f sets CB=CA at the guess: ``CB=CA``, then
    ``DSYRK(...,OCCA,...,PA)`` / ``DSYRK(...,OCCB,...,PB)`` -- huckop.f:163,255-256),
    each singly occupied (no closed-shell factor of two).  With nalpha==nbeta this
    returns PA==PB==P/2 of the closed-shell guess, so a closed-shell molecule
    forced through the UHF path reproduces RHF.  Returns ``(PA, PB)``."""
    _eps, Cmo = np.linalg.eigh(_huckel_guess_hamiltonian(H, Z, blocks))
    PA = Cmo[:, :nalpha] @ Cmo[:, :nalpha].T
    PB = Cmo[:, :nbeta] @ Cmo[:, :nbeta].T if nbeta else np.zeros_like(PA)
    return PA, PB


# MSINDO molecular-SCF convergence threshold (eneclo.f DELEN) and initial WICHT
# damping factor for the default NAV=4 weighting (scfclo.f:78).
_MSINDO_DELEN = 1e-8
_MSINDO_DAMP0 = 3.0


def _scf_rhf_msindo(H, G, blocks, Z, nocc, *, max_iter=2000):
    """Closed-shell MSINDO RHF SCF faithful to the reference program's default
    molecular path: extended-Hückel start density (:func:`_huckel_guess_density`,
    huckcl.f) + WICHT density damping (wicht.f) + full diagonalization with aufbau
    occupation (NGIV=8) + the energy-only convergence test (eneclo.f:103).

    Used (via the :data:`_MSINDO_TRAJECTORY_SCF` dispatch in :func:`run_msindo`)
    for the heavier d/p-block elements where DIIS extrapolation jumps SCF basins:
    production DIIS lands at a higher stationary point (TcCl/RhF) or a lower one
    with the polarization d wrongly occupied (PdCl2/SbF3).  WICHT damping is a
    gentle, basin-preserving descent and the Hückel guess sets the correct
    occupation; together they reproduce the MSINDO oracle to <=1 µHa.  The
    energy-only stop (no [F,P] residual test) is essential -- these are *metastable*
    states under aufbau, so over-tightening drifts them into the wrong
    (aufbau-global) basin.

    WICHT (wicht.f): the input density for cycle n is ``(P_out + DAMP.P_in)/(1+
    DAMP)`` mixing the previous cycle's aufbau output ``P_out`` with its input
    ``P_in``; ``DAMP`` starts at 3.0 (scfclo.f NAV=4), resets to 1.0 whenever the
    energy rises, and decays x0.8 each cycle.  Returns the same tuple as
    :func:`_scf_rhf`: ``(P, F, e_elec, eps, converged, n_iter)``."""
    P = _huckel_guess_density(H, Z, blocks, nocc)
    damp = _MSINDO_DAMP0
    oldeng, epsi, p_in_prev = 0.0, 0.0, None
    converged, it = False, 0
    e_elec = 0.0
    for it in range(1, max_iter + 1):
        if it > 1:
            if epsi > 0.0:  # energy rose => re-damp hard (wicht.f:31)
                damp = 1.0
            P = (P + damp * p_in_prev) / (1.0 + damp)
            damp *= 0.8
        F = _build_fock(H, G, P, blocks, Z)
        e_elec = 0.5 * float(np.sum(P * (H + F)))
        epsi = e_elec - oldeng
        # eneclo.f:103 -- converge on a small *downward* energy step, or a tiny
        # step of either sign.  Never on cycle 1 (epsi is vs the 0.0 sentinel).
        if it > 1 and (
            (abs(epsi) < _MSINDO_DELEN and epsi <= 0.0)
            or abs(epsi) <= 0.1 * _MSINDO_DELEN
        ):
            converged = True
            break
        oldeng = e_elec
        p_in_prev = P.copy()
        _eps, Cmo = np.linalg.eigh(F)
        P = 2.0 * Cmo[:, :nocc] @ Cmo[:, :nocc].T
    F = _build_fock(H, G, P, blocks, Z)
    e_elec = 0.5 * float(np.sum(P * (H + F)))
    eps = np.linalg.eigh(F)[0]
    return P, F, e_elec, eps, converged, it


def _scf_rhf_molecular(
    H, G, blocks, Z, nocc, *, max_iter=200, conv_tol=1e-9
):
    """Select a molecular INDO RHF density from strict stationary candidates.

    Hcore/DIIS remains the primary trajectory. Validated H-Ar jobs also evaluate
    the reference extended-Hückel/WICHT path. Every converged WICHT density is
    strict-DIIS-refined, then accepted only when its stationary energy is lower
    by more than the convergence resolution. WICHT's energy-only result is never
    returned directly for those jobs. Every proactive probe retains the finite
    2000-cycle recovery floor introduced for tetrazine in fa4af4b4f, so crossing
    the caller's strict-DIIS cap cannot change which basin is considered.
    Converged jobs outside H-Ar retain their established route; failed-primary
    recovery remains available outside that scope. The trajectory-pinned
    heavy-element set keeps its direct WICHT energy path.
    """
    if any(z in _MSINDO_TRAJECTORY_SCF for z in Z):
        return _scf_rhf_msindo(
            H, G, blocks, Z, nocc, max_iter=max(max_iter, 2000)
        )

    P, F, e_elec, eps, converged, primary_it = _scf_rhf(
        H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol
    )
    proactive_probe = all(z in _MSINDO_ROOT_PROBE_ELEMENTS for z in Z)
    if converged and not proactive_probe:
        return P, F, e_elec, eps, converged, primary_it

    probe_cap = max(max_iter, 2000)
    warm_P, _warm_F, _warm_e, _warm_eps, warm_converged, warm_it = (
        _scf_rhf_msindo(H, G, blocks, Z, nocc, max_iter=probe_cap)
    )
    total_it = primary_it + warm_it

    if not warm_converged:
        return P, F, e_elec, eps, converged, total_it

    strict = _scf_rhf(
        H,
        G,
        blocks,
        Z,
        nocc,
        max_iter=max_iter,
        conv_tol=conv_tol,
        initial_density=warm_P,
    )
    strict_P, strict_F, strict_e, strict_eps, strict_converged, strict_it = strict
    total_it += strict_it

    # A failed primary preserves the exact recovery semantics of fa4af4b4f,
    # including returning the strict attempt when it also exhausts its cap.
    if not converged or (
        strict_converged and strict_e < e_elec - conv_tol
    ):
        return (
            strict_P,
            strict_F,
            strict_e,
            strict_eps,
            strict_converged,
            total_it,
        )

    return P, F, e_elec, eps, converged, total_it


def _scf_uhf_msindo(H, G, blocks, Z, nalpha, nbeta, *, max_iter=5000):
    """Open-shell (UHF) MSINDO SCF faithful to the reference program's default
    molecular path (scfopn.f / eneopn.f) -- the spin-unrestricted analog of
    :func:`_scf_rhf_msindo`: open-shell extended-Hückel start densities
    (:func:`_huckel_guess_density_uhf`, huckop.f) + WICHT density damping of
    *both* spin densities (wicht.f, the ``IF(UHF.OR.ROHF)`` block damps PA and PB
    with the same DAMP) + full diagonalization with aufbau occupation (NGIV=8) +
    the energy-only convergence test (eneopn.f:87, DELEN=1e-8).

    Used (via the :data:`_MSINDO_TRAJECTORY_SCF` dispatch in :func:`_run_uhf`) for
    open-shell radicals of the heavier d/p-block elements, where the DIIS path
    jumps SCF basins exactly as it does for the closed-shell heavy elements (see
    docs/user_guide/msindo.md and :func:`_scf_rhf_msindo`).  ``DAMP`` starts at 3.0
    (scfopn.f NAV=4), resets to 1.0 on an energy rise, decays x0.8/cycle.  The
    metastable open-shell spin states creep more slowly than closed-shell, so the
    cycle cap is higher (5000).

    Density bookkeeping mirrors eneopn.f:240-250: the WICHT-damped *input* density
    used to build the Fock this cycle (``pa_in_prev``/``pb_in_prev``, the KOPMAT
    of PA/PB into FA/FB) is what WICHT mixes the next cycle's aufbau output
    against.  Returns the same tuple as :func:`_scf_uhf`:
    ``(PA, PB, CA, CB, epsA, epsB, e_elec, converged, n_iter)``."""
    PA, PB = _huckel_guess_density_uhf(H, Z, blocks, nalpha, nbeta)
    damp = _MSINDO_DAMP0
    oldeng, epsi = 0.0, 0.0
    pa_in_prev, pb_in_prev = None, None
    converged, it = False, 0
    e_elec = 0.0
    CA = CB = None
    for it in range(1, max_iter + 1):
        if it > 1:
            if epsi > 0.0:  # energy rose => re-damp hard (wicht.f:31)
                damp = 1.0
            PA = (PA + damp * pa_in_prev) / (1.0 + damp)
            PB = (PB + damp * pb_in_prev) / (1.0 + damp)
            damp *= 0.8
        FA, FB = _build_fock_uhf(H, G, PA, PB, blocks, Z)
        e_elec = 0.5 * float(np.sum((PA + PB) * H) + np.sum(PA * FA) + np.sum(PB * FB))
        epsi = e_elec - oldeng
        # eneopn.f:87 -- converge on a small *downward* energy step, or a tiny
        # step of either sign.  Never on cycle 1 (epsi is vs the 0.0 sentinel).
        if it > 1 and (
            (abs(epsi) < _MSINDO_DELEN and epsi <= 0.0)
            or abs(epsi) <= 0.1 * _MSINDO_DELEN
        ):
            converged = True
            break
        oldeng = e_elec
        pa_in_prev, pb_in_prev = PA.copy(), PB.copy()
        _epsA, CA = np.linalg.eigh(FA)
        PA = CA[:, :nalpha] @ CA[:, :nalpha].T
        if nbeta:
            _epsB, CB = np.linalg.eigh(FB)
            PB = CB[:, :nbeta] @ CB[:, :nbeta].T
        else:
            PB = np.zeros_like(PA)
    # Self-consistent energy + MO data at the converged densities (the b set is
    # all-virtual when nbeta=0, but its orbitals still define the b manifold).
    FA, FB = _build_fock_uhf(H, G, PA, PB, blocks, Z)
    e_elec = 0.5 * float(np.sum((PA + PB) * H) + np.sum(PA * FA) + np.sum(PB * FB))
    epsA, CA = np.linalg.eigh(FA)
    epsB, CB = np.linalg.eigh(FB)
    return PA, PB, CA, CB, epsA, epsB, e_elec, converged, it


def _uhf_occupation(nelec, multiplicity):
    """(nalpha, nbeta) for a UHF reference; validates the spin/electron parity."""
    # multiplicity = 2S+1 => na-nb = M-1; na+nb = nelec.  Consistency: nelec and
    # (M-1) must share parity, i.e. (nelec + M) is odd.
    if (nelec + multiplicity) % 2 == 0:
        raise ValueError(
            f"multiplicity={multiplicity} is inconsistent with {nelec} valence "
            "electrons (need n_electrons and 2S of the same parity)."
        )
    nalpha = (nelec + multiplicity - 1) // 2
    nbeta = nelec - nalpha
    if nbeta < 0:
        raise ValueError(
            f"multiplicity={multiplicity} needs more than the {nelec} available "
            "valence electrons."
        )
    return nalpha, nbeta


def _scf_uhf(
    H, G, blocks, Z, nalpha, nbeta, *, max_iter=200, conv_tol=1e-9, fock_extra=None
):
    """Spin-unrestricted INDO SCF loop (fockop.f), s/p/d elements.

    Two spin densities PA, PB; Coulomb from the total density, exchange from the
    same-spin density.  Same Pulay-DIIS / residual-norm convergence as
    :func:`_scf_rhf`, with the a+b commutators stacked into one error vector.
    Returns ``(PA, PB, CA, CB, epsA, epsB, e_elec, converged, it)`` -- the
    converged spin densities, MO coefficients, MO energies, and the electronic
    energy ``1/2Tr[Pt.H + PA.FA + PB.FB]``.  The caller adds the core repulsion and
    enforces any element-scope guard.  Shared by :func:`_run_uhf` (energy) and
    :func:`msindo_ovgf` (open-shell quasiparticles).

    ``fock_extra``, when given, is a callback ``f(Pt) -> (F_add, e_add)`` that
    adds ``F_add`` to both FA and FB and ``e_add`` to the electronic energy."""
    epsA, Cmo = np.linalg.eigh(H)
    PA = Cmo[:, :nalpha] @ Cmo[:, :nalpha].T
    PB = Cmo[:, :nbeta] @ Cmo[:, :nbeta].T if nbeta else np.zeros_like(PA)
    e_elec, converged, it = 0.0, False, 0
    fa_hist, fb_hist, e_hist = [], [], []
    DIIS_MAX = 8
    for it in range(1, max_iter + 1):
        FA, FB = _build_fock_uhf(H, G, PA, PB, blocks, Z)
        e_add = 0.0
        if fock_extra:
            F_add, e_add = fock_extra(PA + PB)
            FA = FA + F_add
            FB = FB + F_add
        errA = FA @ PA - PA @ FA
        errB = FB @ PB - PB @ FB
        Pt = PA + PB
        e_cur = 0.5 * float(np.sum(Pt * H) + np.sum(PA * FA) + np.sum(PB * FB)) + e_add
        res = max(
            float(np.max(np.abs(errA))), float(np.max(np.abs(errB))) if nbeta else 0.0
        )
        if res < conv_tol and abs(e_cur - e_elec) < conv_tol:
            e_elec, converged = e_cur, True
            break
        e_elec = e_cur
        fa_hist.append(FA)
        fb_hist.append(FB)
        e_hist.append(np.concatenate([errA.ravel(), errB.ravel()]))
        if len(fa_hist) > DIIS_MAX:
            fa_hist.pop(0)
            fb_hist.pop(0)
            e_hist.pop(0)
        n = len(fa_hist)
        FA_eff, FB_eff = FA, FB
        if n >= 2:
            B = np.full((n + 1, n + 1), -1.0)
            B[n, n] = 0.0
            for a in range(n):
                for b in range(a, n):
                    B[a, b] = B[b, a] = float(np.dot(e_hist[a], e_hist[b]))
            rhs = np.zeros(n + 1)
            rhs[n] = -1.0
            try:
                c = np.linalg.solve(B, rhs)
                FA_eff = sum(c[a] * fa_hist[a] for a in range(n))
                FB_eff = sum(c[a] * fb_hist[a] for a in range(n))
            except np.linalg.LinAlgError:
                FA_eff, FB_eff = FA, FB
        epsA, CA = np.linalg.eigh(FA_eff)
        PA = CA[:, :nalpha] @ CA[:, :nalpha].T
        if nbeta:
            _, CB = np.linalg.eigh(FB_eff)
            PB = CB[:, :nbeta] @ CB[:, :nbeta].T
        else:
            PB = np.zeros_like(PA)

    # Self-consistent energy + MO data at the converged densities.  Both Fock
    # matrices are diagonalised (the b set is all-virtual when nbeta=0, but its
    # orbitals still define the b electron-attachment manifold for EAs).
    FA, FB = _build_fock_uhf(H, G, PA, PB, blocks, Z)
    e_add_final = 0.0
    if fock_extra:
        F_add, e_add_final = fock_extra(PA + PB)
        FA = FA + F_add
        FB = FB + F_add
    e_elec = (
        0.5 * float(np.sum((PA + PB) * H) + np.sum(PA * FA) + np.sum(PB * FB))
        + e_add_final
    )
    epsA, CA = np.linalg.eigh(FA)
    epsB, CB = np.linalg.eigh(FB)
    return PA, PB, CA, CB, epsA, epsB, e_elec, converged, it


def _run_uhf(Z, H, G, blocks, nelec, multiplicity, e_core, max_iter, conv_tol):
    """Spin-unrestricted MSINDO SCF (fockop.f), s/p/d elements.

    Open-shell radicals of the heavier d/p-block elements (any element in
    :data:`_MSINDO_TRAJECTORY_SCF`) reach the reference SCF stationary point only
    via the MSINDO-faithful Hückel-guess + WICHT-damped SCF (:func:`_scf_uhf_msindo`)
    -- the same basin-selection problem the closed-shell heavy elements have
    (docs/user_guide/msindo.md): DIIS lands on a different stationary point.  Every
    other element keeps the (faster, more robust for light near-degenerate cases)
    DIIS driver, which preserves the tight UHF(M=1)==RHF invariant on that path.
    The dispatch matches :func:`run_msindo`'s RHF dispatch, so a closed-shell
    heavy molecule forced through UHF still reduces to its RHF result."""
    nalpha, nbeta = _uhf_occupation(nelec, multiplicity)
    if any(z in _MSINDO_TRAJECTORY_SCF for z in Z):
        PA, PB, _CA, _CB, epsA, _epsB, e_elec, converged, it = _scf_uhf_msindo(
            H, G, blocks, Z, nalpha, nbeta, max_iter=max(max_iter, 5000)
        )
    else:
        PA, PB, _CA, _CB, epsA, _epsB, e_elec, converged, it = _scf_uhf(
            H, G, blocks, Z, nalpha, nbeta, max_iter=max_iter, conv_tol=conv_tol
        )
    total = e_elec + e_core
    binding = total - sum(ateng(z) for z in Z)
    return MsindoResult(total, e_elec, binding, epsA, PA + PB, it, converged)


def _build_fock_uhf(H, G, PA, PB, blocks, Z):
    """UHF Fock matrices FA, FB (fockop.f) for s/p/d elements.

    Coulomb is taken from the total density PA+PB; exchange from the same-spin
    density (no closed-shell 1/2).  The two-centre and one-centre s/p (gij1) parts
    (fockop.f:44-122) reduce to ``_build_fock`` term-for-term when PA = PB = P/2;
    the one-centre d block uses :func:`_add_einzi_dblock_uhf` (a faithful port of
    fockop.f:124-367), which reduces to ``_build_fock``'s ``_add_einzi_dblock``
    *except* for a handful of off-axis d-d couplings where fockop.f and fockcl.f
    genuinely differ -- there UHF(M=1) tracks the reference UHF value, not RHF (the
    MSINDO oracle itself shows this, e.g. AlCl₃ RHF - UHF(M=1) ≈ 2.7e-5 Ha, while
    linear/symmetric d molecules like PdCl₂ reduce exactly).
    """
    FA = H.copy()
    FB = H.copy()
    Pt = PA + PB
    nsto = H.shape[0]
    uat = np.empty(nsto, int)
    for i, (lo, hi) in enumerate(blocks):
        uat[lo:hi] = i

    # two-center: Coulomb (total density, diagonal) + exchange (same-spin)
    for i in range(nsto):
        for j in range(nsto):
            if uat[i] != uat[j]:
                FA[i, i] += Pt[j, j] * G[i, j]
                FB[i, i] += Pt[j, j] * G[i, j]
                FA[j, i] += -PA[j, i] * G[i, j]
                FB[j, i] += -PB[j, i] * G[i, j]

    # one-center INDO (gij1): Coulomb from total density, exchange from same-spin
    for ia, (lo, hi) in enumerate(blocks):
        for j in range(lo, hi):
            for k in range(lo, hi):
                jmin, jmax = min(j, k), max(j, k)
                FA[j, j] += Pt[k, k] * G[jmin, jmax] - PA[k, k] * G[jmax, jmin]
                FB[j, j] += Pt[k, k] * G[jmin, jmax] - PB[k, k] * G[jmax, jmin]
            for k in range(j + 1, hi):
                fa = (2.0 * Pt[k, j] - PA[k, j]) * G[k, j] - PA[k, j] * G[j, k]
                fb = (2.0 * Pt[k, j] - PB[k, j]) * G[k, j] - PB[k, j] * G[j, k]
                FA[k, j] += fa
                FA[j, k] += fa
                FB[k, j] += fb
                FB[j, k] += fb
        if hi - lo >= 9:
            # One-centre d Fock: fockop.f keeps Coulomb (total density) and
            # exchange (same-spin) explicit -- applying the closed-shell EINZI per
            # spin is wrong for d (see _add_einzi_dblock_uhf).
            _add_einzi_dblock_uhf(FA, FB, PA, PB, lo, Z[ia])
    return FA, FB


def _build_fock(H, G, P, blocks, Z):
    """Closed-shell Fock (fockcl.f): core + two-center Coulomb/exchange +
    one-center INDO (gij1 9x9 packing) + EINZI hybrid d-block."""
    nsto = H.shape[0]
    F = H.copy()
    uat = np.empty(nsto, int)
    for i, (lo, hi) in enumerate(blocks):
        uat[lo:hi] = i

    # two-center Coulomb (diag) + exchange (offdiag)
    for i in range(nsto):
        for j in range(nsto):
            if uat[i] != uat[j]:
                F[i, i] += P[j, j] * G[i, j]
                F[j, i] += -0.5 * P[j, i] * G[i, j]

    # one-center INDO (gij1): upper=Coulomb, lower=exchange; mirror to upper.
    for ia, (lo, hi) in enumerate(blocks):
        for j in range(lo, hi):
            for k in range(lo, hi):
                jmin, jmax = min(j, k), max(j, k)
                F[j, j] += P[k, k] * (G[jmin, jmax] - 0.5 * G[jmax, jmin])
            for k in range(j + 1, hi):
                val = 0.5 * P[k, j] * (3.0 * G[k, j] - G[j, k])
                F[k, j] += val
                F[j, k] += val
        if hi - lo >= 9:
            _add_einzi_dblock(F, P, lo, Z[ia])
    return F


def _add_einzi_dblock(F, P, lo, z):
    """EINZI hybrid one-center d Fock terms (fockcl.f L118-210), mirrored."""
    h = einzi_hyb(z)
    s3 = math.sqrt(3.0)
    L = lo  # LLIMIT (s)
    L2, L3, L4 = lo + 1, lo + 2, lo + 3  # px,py,pz
    L5, L6, L7, L8, L9 = lo + 4, lo + 5, lo + 6, lo + 7, lo + 8  # d
    X1 = 3.0 * h[1] - h[4]
    X2 = 3.0 * h[3] - h[5]
    X3 = 3.0 * h[2] - h[8]
    X4 = h[4] - h[1] / 2.0
    X5 = h[5] - h[3] / 2.0
    X6 = 2.0 * h[8] - h[2]
    X7 = h[13] - 2.0 * h[10]
    X8 = 2.0 * h[11] - (h[15] + h[17]) / 2.0
    X9 = 2.0 * h[12] - (h[14] + h[16]) / 2.0
    X10 = h[10] - 3.0 * h[13]
    X11 = 2.0 * h[14] - (h[12] + h[16]) / 2.0
    X12 = 2.0 * h[15] - (h[11] + h[17]) / 2.0
    X13 = 2.0 * h[16] - (h[12] + h[14]) / 2.0
    X14 = 2.0 * h[17] - (h[11] + h[15]) / 2.0
    X15 = 3.0 * h[19] - h[20]
    X16 = h[19] / 2.0 - h[20]

    def add(i, j, val):
        F[i, j] += val
        if i != j:
            F[j, i] += val

    add(L2, L, (P[L5, L2] - s3 * (P[L8, L2] + P[L9, L3] + P[L6, L4])) / 2.0 * X1)
    add(L3, L, (P[L5, L3] - s3 * (P[L9, L2] - P[L8, L3] + P[L7, L4])) / 2.0 * X1)
    add(L4, L, (P[L5, L4] * X2 + (P[L6, L2] + P[L7, L3]) * X3) / 2.0)
    add(
        L5,
        L,
        (P[L2, L2] + P[L3, L3]) * X4
        + P[L4, L4] * X5
        + (P[L5, L5] - P[L8, L8] - P[L9, L9]) * h[6] / 2.0
        + (P[L6, L6] + P[L7, L7]) * h[7] / 2.0,
    )
    add(L6, L, P[L6, L5] * h[7] + P[L4, L2] * X6 + (P[L8, L6] + P[L9, L7]) * h[9])
    add(L7, L, P[L4, L3] * X6 + P[L7, L5] * h[7] + (P[L9, L6] - P[L8, L7]) * h[9])
    add(
        L8,
        L,
        (P[L2, L2] - P[L3, L3]) * X6 / 2.0
        - P[L8, L5] * h[6]
        + (P[L6, L6] - P[L7, L7]) * h[9] / 2.0,
    )
    add(L9, L, P[L3, L2] * X6 - P[L9, L5] * h[6] + P[L7, L6] * h[9])
    add(L2, L2, P[L5, L] * 2.0 * X4 + P[L8, L] * X6 - P[L8, L5] * X7)
    add(L3, L2, P[L9, L] * X6 - P[L9, L5] * X7 + P[L7, L6] * X8)
    add(L4, L2, P[L6, L] * X6 + P[L6, L5] * X9 + (P[L8, L6] + P[L9, L7]) * X8)
    add(L5, L2, (P[L2, L] * X1 - (P[L8, L2] + P[L9, L3]) * X10) / 2.0 + P[L6, L4] * X11)
    add(
        L6,
        L2,
        P[L4, L] * X3 / 2.0 + P[L7, L3] * X12 + P[L5, L4] * X13 + P[L8, L4] * X14,
    )
    add(L7, L2, (P[L6, L3] + P[L9, L4]) * X14)
    add(
        L8,
        L2,
        (P[L2, L] * X3 - P[L5, L2] * X10 + 5.0 * P[L9, L3] * h[18]) / 2.0
        + P[L6, L4] * X12,
    )
    add(
        L9,
        L2,
        (P[L3, L] * X3 - P[L5, L3] * X10 - 5.0 * P[L8, L3] * h[18]) / 2.0
        + P[L7, L4] * X12,
    )
    add(L3, L3, P[L5, L] * 2.0 * X4 - P[L8, L] * X6 + P[L8, L5] * X7)
    add(L4, L3, P[L7, L] * X6 + (P[L9, L6] - P[L8, L7]) * X8 + P[L7, L5] * X9)
    add(L5, L3, (P[L3, L] * X1 + (P[L8, L3] - P[L9, L2]) * X10) / 2.0 + P[L7, L4] * X11)
    add(L6, L3, (P[L7, L2] + P[L9, L4]) * X14)
    add(
        L7,
        L3,
        P[L4, L] * X3 / 2.0 + P[L6, L2] * X12 + P[L5, L4] * X13 - P[L8, L4] * X14,
    )
    add(
        L8,
        L3,
        (P[L5, L3] * X10 - P[L3, L] * X3 - 5.0 * P[L9, L2] * h[18]) / 2.0
        - P[L7, L4] * X12,
    )
    add(
        L9,
        L3,
        (P[L2, L] * X3 - P[L5, L2] * X10 + 5.0 * P[L8, L2] * h[18]) / 2.0
        + P[L6, L4] * X12,
    )
    add(L4, L4, 2.0 * P[L5, L] * X5)
    add(L5, L4, P[L4, L] * X2 / 2.0 + (P[L6, L2] + P[L7, L3]) * X13)
    add(L6, L4, P[L2, L] * X3 / 2.0 + P[L5, L2] * X11 + (P[L8, L2] + P[L9, L3]) * X12)
    add(L7, L4, P[L3, L] * X3 / 2.0 + (P[L9, L2] - P[L8, L3]) * X12 + P[L5, L3] * X11)
    add(L8, L4, (P[L6, L2] - P[L7, L3]) * X14)
    add(L9, L4, (P[L7, L2] + P[L6, L3]) * X14)
    add(L5, L5, P[L5, L] * h[6])
    add(L6, L5, P[L6, L] * h[7] + P[L4, L2] * X9 + (P[L8, L6] + P[L9, L7]) * X15 / 2.0)
    add(L7, L5, P[L7, L] * h[7] + P[L4, L3] * X9 + (P[L9, L6] - P[L8, L7]) * X15 / 2.0)
    add(
        L8,
        L5,
        -P[L8, L] * h[6]
        + (P[L3, L3] - P[L2, L2]) * X7 / 2.0
        + (P[L7, L7] - P[L6, L6]) * X16,
    )
    add(L9, L5, -P[L9, L] * h[6] - P[L3, L2] * X7 - 2.0 * P[L7, L6] * X16)
    add(L6, L6, P[L5, L] * h[7] + P[L8, L] * h[9] - 2.0 * P[L8, L5] * X16)
    add(L7, L6, P[L9, L] * h[9] + P[L3, L2] * X8 - 2.0 * P[L9, L5] * X16)
    add(
        L8,
        L6,
        P[L6, L] * h[9]
        + P[L4, L2] * X8
        + (P[L6, L5] * X15 + 5.0 * P[L9, L7] * h[21]) / 2.0,
    )
    add(
        L9,
        L6,
        P[L7, L] * h[9]
        + P[L4, L3] * X8
        + (P[L7, L5] * X15 - 5.0 * P[L8, L7] * h[21]) / 2.0,
    )
    add(L7, L7, P[L5, L] * h[7] - P[L8, L] * h[9] + 2.0 * P[L8, L5] * X16)
    add(
        L8,
        L7,
        -P[L7, L] * h[9]
        - P[L4, L3] * X8
        - (P[L7, L5] * X15 + 5.0 * P[L9, L6] * h[21]) / 2.0,
    )
    add(
        L9,
        L7,
        P[L6, L] * h[9]
        + P[L4, L2] * X8
        + (P[L6, L5] * X15 + 5.0 * P[L8, L6] * h[21]) / 2.0,
    )
    add(L8, L8, -P[L5, L] * h[6])
    add(L9, L9, -P[L5, L] * h[6])


def _add_einzi_dblock_uhf(FA, FB, PA, PB, lo, z):
    """Open-shell one-center d Fock terms (fockop.f L124-367) -- the spin-
    unrestricted analog of :func:`_add_einzi_dblock` (fockcl.f).

    The closed-shell EINZI X-combinations fold Coulomb and exchange together
    (only valid when PA==PB), so the open-shell d Fock cannot be obtained by
    applying :func:`_add_einzi_dblock` per spin.  fockop.f instead keeps them
    explicit: the Coulomb part ``GEM`` is contracted with the *total* density
    ``Pt = PA + PB`` (spin-independent), while exchange uses the *same-spin*
    density, with fockop's own coefficient combinations (X1..X11 from the raw
    EINZI integrals, fockop.f:80-90; HYB then *doubled*, fockop.f:91-93).  Each
    a element couples the opposite-spin (b) density only through Coulomb-type
    cross-terms and the a density through exchange-type terms; FB swaps the roles.

    fockop.f writes only the lower triangle of each one-centre block (eneopn.f
    reads it row-by-row); we mirror off-diagonal terms to the upper triangle for
    the full-matrix energy 1/2(Tr[Pt.H]+Tr[PA.FA]+Tr[PB.FB]).  Setting PA==PB==P/2
    reproduces :func:`_add_einzi_dblock` term-for-term (the closed-shell limit;
    pinned in tests/test_msindo.py)."""
    h = einzi_hyb(z)  # raw EINZI hybrid integrals (1-indexed)
    # fockop.f:80-90 -- coefficient combinations from the *raw* HYB.
    X1 = h[1] + h[4]
    X2 = h[2] + h[8]
    X3 = h[3] + h[5]
    X4 = -(h[13] + h[10])
    X5 = h[17] + h[15]
    X6 = h[17] + h[11]
    X7 = h[15] + h[11]
    X8 = h[14] + h[12]
    X9 = h[14] + h[16]
    X10 = h[12] + h[16]
    X11 = h[19] + h[20]
    # fockop.f:91-93 -- HYB is doubled *after* the X combinations are formed.
    hyb = [2.0 * v for v in h]
    L = lo  # LLIMIT (s)
    L2, L3, L4 = lo + 1, lo + 2, lo + 3  # px,py,pz
    L5, L6, L7, L8, L9 = lo + 4, lo + 5, lo + 6, lo + 7, lo + 8  # d
    Pt = PA + PB  # total density (Coulomb / GEM); spin-independent

    def addA(i, j, val):
        FA[i, j] += val
        if i != j:
            FA[j, i] += val

    def addB(i, j, val):
        FB[i, j] += val
        if i != j:
            FB[j, i] += val

    # --- p-s and d-s one-centre couplings (fockop.f:136-185) ---
    gem = Pt[L5, L2] * hyb[1] + (Pt[L8, L2] + Pt[L9, L3] + Pt[L6, L4]) * hyb[2]
    addA(L2, L, gem - PA[L5, L2] * X1 - (PA[L8, L2] + PA[L9, L3] + PA[L6, L4]) * X2)
    addB(L2, L, gem - PB[L5, L2] * X1 - (PB[L8, L2] + PB[L9, L3] + PB[L6, L4]) * X2)
    gem = (Pt[L7, L4] - Pt[L8, L3] + Pt[L9, L2]) * hyb[2] + Pt[L5, L3] * hyb[1]
    addA(L3, L, gem - PA[L5, L3] * X1 - (PA[L9, L2] - PA[L8, L3] + PA[L7, L4]) * X2)
    addB(L3, L, gem - PB[L5, L3] * X1 - (PB[L9, L2] - PB[L8, L3] + PB[L7, L4]) * X2)
    gem = (Pt[L6, L2] + Pt[L7, L3]) * hyb[2] + Pt[L5, L4] * hyb[3]
    addA(L4, L, gem - (PA[L6, L2] + PA[L7, L3]) * X2 - PA[L5, L4] * X3)
    addB(L4, L, gem - (PB[L6, L2] + PB[L7, L3]) * X2 - PB[L5, L4] * X3)
    gem = ((Pt[L2, L2] + Pt[L3, L3]) * hyb[4] + Pt[L4, L4] * hyb[5]) / 2.0
    addA(
        L5,
        L,
        gem
        + (
            (PB[L5, L5] - PB[L8, L8] - PB[L9, L9]) * hyb[6]
            + (PB[L6, L6] + PB[L7, L7]) * hyb[7]
            - (PA[L2, L2] + PA[L3, L3]) * hyb[1]
            - PA[L4, L4] * hyb[3]
        )
        / 2.0,
    )
    addB(
        L5,
        L,
        gem
        + (
            (PA[L5, L5] - PA[L8, L8] - PA[L9, L9]) * hyb[6]
            + (PA[L6, L6] + PA[L7, L7]) * hyb[7]
            - (PB[L2, L2] + PB[L3, L3]) * hyb[1]
            - PB[L4, L4] * hyb[3]
        )
        / 2.0,
    )
    gem = Pt[L4, L2] * hyb[8]
    addA(L6, L, gem + PB[L6, L5] * hyb[7] + (PB[L8, L6] + PB[L9, L7]) * hyb[9] - PA[L4, L2] * hyb[2])
    addB(L6, L, gem + PA[L6, L5] * hyb[7] + (PA[L8, L6] + PA[L9, L7]) * hyb[9] - PB[L4, L2] * hyb[2])
    gem = Pt[L4, L3] * hyb[8]
    addA(L7, L, gem + PB[L7, L5] * hyb[7] + (PB[L9, L6] - PB[L8, L7]) * hyb[9] - PA[L4, L3] * hyb[2])
    addB(L7, L, gem + PA[L7, L5] * hyb[7] + (PA[L9, L6] - PA[L8, L7]) * hyb[9] - PB[L4, L3] * hyb[2])
    gem = (Pt[L2, L2] - Pt[L3, L3]) * hyb[8] / 2.0
    addA(
        L8,
        L,
        gem
        - PB[L8, L5] * hyb[6]
        - ((PA[L2, L2] - PA[L3, L3]) * hyb[2] - (PB[L6, L6] - PB[L7, L7]) * hyb[9]) / 2.0,
    )
    addB(
        L8,
        L,
        gem
        - PA[L8, L5] * hyb[6]
        - ((PB[L2, L2] - PB[L3, L3]) * hyb[2] - (PA[L6, L6] - PA[L7, L7]) * hyb[9]) / 2.0,
    )
    gem = Pt[L3, L2] * hyb[8]
    addA(L9, L, gem - PB[L9, L5] * hyb[6] + PB[L7, L6] * hyb[9] - PA[L3, L2] * hyb[2])
    addB(L9, L, gem - PA[L9, L5] * hyb[6] + PA[L7, L6] * hyb[9] - PB[L3, L2] * hyb[2])

    # --- p-p and d-p one-centre couplings (fockop.f:186-292) ---
    gem = Pt[L5, L] * hyb[4] + Pt[L8, L] * hyb[8] + Pt[L8, L5] * hyb[10]
    addA(L2, L2, gem - PA[L8, L5] * hyb[13] - PA[L5, L] * hyb[1] - PA[L8, L] * hyb[2])
    addB(L2, L2, gem - PB[L8, L5] * hyb[13] - PB[L5, L] * hyb[1] - PB[L8, L] * hyb[2])
    gem = Pt[L9, L] * hyb[8] + Pt[L7, L6] * hyb[11] + Pt[L9, L5] * hyb[10]
    addA(L3, L2, gem - PA[L9, L5] * hyb[13] - PA[L7, L6] * X5 - PA[L9, L] * hyb[2])
    addB(L3, L2, gem - PB[L9, L5] * hyb[13] - PB[L7, L6] * X5 - PB[L9, L] * hyb[2])
    gem = Pt[L6, L] * hyb[8] + (Pt[L8, L6] + Pt[L9, L7]) * hyb[11] + Pt[L6, L5] * hyb[12]
    addA(L4, L2, gem - PA[L6, L] * hyb[2] - (PA[L8, L6] + PA[L9, L7]) * X5 - PA[L6, L5] * X9)
    addB(L4, L2, gem - PB[L6, L] * hyb[2] - (PB[L8, L6] + PB[L9, L7]) * X5 - PB[L6, L5] * X9)
    gem = Pt[L2, L] * hyb[1] + (Pt[L8, L2] + Pt[L9, L3]) * hyb[13] + Pt[L6, L4] * hyb[14]
    addA(L5, L2, gem - PA[L2, L] * X1 + (PA[L8, L2] + PA[L9, L3]) * X4 - PA[L6, L4] * X10)
    addB(L5, L2, gem - PB[L2, L] * X1 + (PB[L8, L2] + PB[L9, L3]) * X4 - PB[L6, L4] * X10)
    gem = Pt[L4, L] * hyb[2] + Pt[L8, L4] * hyb[17] + Pt[L7, L3] * hyb[15] + Pt[L5, L4] * hyb[16]
    addA(L6, L2, gem - PA[L4, L] * X2 - PA[L7, L3] * X6 - PA[L8, L4] * X7 - PA[L5, L4] * X8)
    addB(L6, L2, gem - PB[L4, L] * X2 - PB[L7, L3] * X6 - PB[L8, L4] * X7 - PB[L5, L4] * X8)
    gem = (Pt[L6, L3] + Pt[L9, L4]) * hyb[17]
    addA(L7, L2, gem - (PA[L6, L3] + PA[L9, L4]) * X7)
    addB(L7, L2, gem - (PB[L6, L3] + PB[L9, L4]) * X7)
    gem = Pt[L2, L] * hyb[2] + Pt[L6, L4] * hyb[15] + Pt[L9, L3] * hyb[18] + Pt[L5, L2] * hyb[13]
    addA(L8, L2, gem - PA[L2, L] * X2 - PA[L6, L4] * X6 + PA[L9, L3] * hyb[18] / 2.0 + PA[L5, L2] * X4)
    addB(L8, L2, gem - PB[L2, L] * X2 - PB[L6, L4] * X6 + PB[L9, L3] * hyb[18] / 2.0 + PB[L5, L2] * X4)
    gem = Pt[L3, L] * hyb[2] + Pt[L5, L3] * hyb[13] + Pt[L7, L4] * hyb[15] - Pt[L8, L3] * hyb[18]
    addA(L9, L2, gem - PA[L3, L] * X2 + PA[L5, L3] * X4 - PA[L7, L4] * X6 - PA[L8, L3] * hyb[18] / 2.0)
    addB(L9, L2, gem - PB[L3, L] * X2 + PB[L5, L3] * X4 - PB[L7, L4] * X6 - PB[L8, L3] * hyb[18] / 2.0)
    gem = Pt[L5, L] * hyb[4] - Pt[L8, L] * hyb[8] - Pt[L8, L5] * hyb[10]
    addA(L3, L3, gem + PA[L8, L5] * hyb[13] - PA[L5, L] * hyb[1] + PA[L8, L] * hyb[2])
    addB(L3, L3, gem + PB[L8, L5] * hyb[13] - PB[L5, L] * hyb[1] + PB[L8, L] * hyb[2])
    gem = Pt[L7, L] * hyb[8] + (Pt[L9, L6] - Pt[L8, L7]) * hyb[11] + Pt[L7, L5] * hyb[12]
    addA(L4, L3, gem - PA[L7, L5] * X9 - (PA[L9, L6] - PA[L8, L7]) * X5 - PA[L7, L] * hyb[2])
    addB(L4, L3, gem - PB[L7, L5] * X9 - (PB[L9, L6] - PB[L8, L7]) * X5 - PB[L7, L] * hyb[2])
    gem = Pt[L3, L] * hyb[1] - (Pt[L8, L3] - Pt[L9, L2]) * hyb[13] + Pt[L7, L4] * hyb[14]
    addA(L5, L3, gem - PA[L3, L] * X1 - (PA[L8, L3] - PA[L9, L2]) * X4 - PA[L7, L4] * X10)
    addB(L5, L3, gem - PB[L3, L] * X1 - (PB[L8, L3] - PB[L9, L2]) * X4 - PB[L7, L4] * X10)
    gem = (Pt[L7, L2] + Pt[L9, L4]) * hyb[17]
    addA(L6, L3, gem - (PA[L7, L2] + PA[L9, L4]) * X7)
    addB(L6, L3, gem - (PB[L7, L2] + PB[L9, L4]) * X7)
    gem = Pt[L4, L] * hyb[2] + Pt[L6, L2] * hyb[15] + Pt[L5, L4] * hyb[16] - Pt[L8, L4] * hyb[17]
    addA(L7, L3, gem - PA[L4, L] * X2 - PA[L6, L2] * X6 + PA[L8, L4] * X7 - PA[L5, L4] * X8)
    addB(L7, L3, gem - PB[L4, L] * X2 - PB[L6, L2] * X6 + PB[L8, L4] * X7 - PB[L5, L4] * X8)
    gem = -Pt[L3, L] * hyb[2] - Pt[L9, L2] * hyb[18] - Pt[L7, L4] * hyb[15] - Pt[L5, L3] * hyb[13]
    addA(L8, L3, gem + PA[L3, L] * X2 + PA[L7, L4] * X6 - PA[L5, L3] * X4 - PA[L9, L2] * hyb[18] / 2.0)
    addB(L8, L3, gem + PB[L3, L] * X2 + PB[L7, L4] * X6 - PB[L5, L3] * X4 - PB[L9, L2] * hyb[18] / 2.0)
    gem = Pt[L2, L] * hyb[2] + Pt[L5, L2] * hyb[13] + Pt[L8, L2] * hyb[18] + Pt[L6, L4] * hyb[15]
    addA(L9, L3, gem - PA[L2, L] * X2 + PA[L5, L2] * X4 - PA[L6, L4] * X6 + PA[L8, L2] * hyb[18] / 2.0)
    addB(L9, L3, gem - PB[L2, L] * X2 + PB[L5, L2] * X4 - PB[L6, L4] * X6 + PB[L8, L2] * hyb[18] / 2.0)
    gem = Pt[L5, L] * hyb[5]
    addA(L4, L4, gem - PA[L5, L] * hyb[3])
    addB(L4, L4, gem - PB[L5, L] * hyb[3])
    gem = Pt[L4, L] * hyb[3] + (Pt[L6, L2] + Pt[L7, L3]) * hyb[16]
    addA(L5, L4, gem - PA[L4, L] * X3 - (PA[L6, L2] + PA[L7, L3]) * X8)
    addB(L5, L4, gem - PB[L4, L] * X3 - (PB[L6, L2] + PB[L7, L3]) * X8)
    gem = Pt[L2, L] * hyb[2] + (Pt[L8, L2] + Pt[L9, L3]) * hyb[15] + Pt[L5, L2] * hyb[14]
    addA(L6, L4, gem - PA[L2, L] * X2 - PA[L5, L2] * X10 - (PA[L8, L2] + PA[L9, L3]) * X6)
    addB(L6, L4, gem - PB[L2, L] * X2 - PB[L5, L2] * X10 - (PB[L8, L2] + PB[L9, L3]) * X6)
    gem = Pt[L3, L] * hyb[2] + (Pt[L9, L2] - Pt[L8, L3]) * hyb[15] + Pt[L5, L3] * hyb[14]
    addA(L7, L4, gem - PA[L3, L] * X2 - PA[L5, L3] * X10 - (PA[L9, L2] - PA[L8, L3]) * X6)
    addB(L7, L4, gem - PB[L3, L] * X2 - PB[L5, L3] * X10 - (PB[L9, L2] - PB[L8, L3]) * X6)
    gem = (Pt[L6, L2] - Pt[L7, L3]) * hyb[17]
    addA(L8, L4, gem - (PA[L6, L2] - PA[L7, L3]) * X7)
    addB(L8, L4, gem - (PB[L6, L2] - PB[L7, L3]) * X7)
    gem = (Pt[L7, L2] + Pt[L6, L3]) * hyb[17]
    addA(L9, L4, gem - (PA[L7, L2] + PA[L6, L3]) * X7)
    addB(L9, L4, gem - (PB[L7, L2] + PB[L6, L3]) * X7)

    # --- d-d one-centre couplings (fockop.f:293-367) ---
    addA(L5, L5, PB[L5, L] * hyb[6])
    addB(L5, L5, PA[L5, L] * hyb[6])
    gem = Pt[L4, L2] * hyb[12] + (Pt[L8, L6] + Pt[L9, L7]) * hyb[19]
    addA(L6, L5, gem + PB[L6, L] * hyb[7] - PA[L4, L2] * X9 - (PA[L8, L6] + PA[L9, L7]) * X11)
    addB(L6, L5, gem + PA[L6, L] * hyb[7] - PB[L4, L2] * X9 - (PB[L8, L6] + PB[L9, L7]) * X11)
    gem = Pt[L4, L3] * hyb[12] + (Pt[L9, L6] - Pt[L8, L7]) * hyb[19]
    addA(L7, L5, gem - PA[L4, L3] * X9 + (PA[L8, L7] - PA[L9, L6]) * X11 + PB[L7, L] * hyb[7])
    addB(L7, L5, gem - PB[L4, L3] * X9 + (PB[L8, L7] - PB[L9, L6]) * X11 + PA[L7, L] * hyb[7])
    gem = -((Pt[L3, L3] - Pt[L2, L2]) * hyb[10] + (Pt[L7, L7] - Pt[L6, L6]) * hyb[20]) / 2.0
    addA(
        L8,
        L5,
        gem
        - ((PA[L6, L6] - PA[L7, L7]) * hyb[19] + (PA[L2, L2] - PA[L3, L3]) * hyb[13]) / 2.0
        - PB[L8, L] * hyb[6],
    )
    addB(
        L8,
        L5,
        gem
        - ((PB[L6, L6] - PB[L7, L7]) * hyb[19] + (PB[L2, L2] - PB[L3, L3]) * hyb[13]) / 2.0
        - PA[L8, L] * hyb[6],
    )
    gem = Pt[L3, L2] * hyb[10] + Pt[L7, L6] * hyb[20]
    addA(L9, L5, gem - PB[L9, L] * hyb[6] - PA[L3, L2] * hyb[13] - PA[L7, L6] * hyb[19])
    addB(L9, L5, gem - PA[L9, L] * hyb[6] - PB[L3, L2] * hyb[13] - PB[L7, L6] * hyb[19])
    gem = Pt[L8, L5] * hyb[20]
    addA(L6, L6, gem + PB[L5, L] * hyb[7] + PB[L8, L] * hyb[9] - PA[L8, L5] * hyb[19])
    addB(L6, L6, gem + PA[L5, L] * hyb[7] + PA[L8, L] * hyb[9] - PB[L8, L5] * hyb[19])
    gem = Pt[L9, L5] * hyb[20] + Pt[L3, L2] * hyb[11]
    addA(L7, L6, gem - PA[L3, L2] * X5 + PB[L9, L] * hyb[9] - PA[L9, L5] * hyb[19])
    addB(L7, L6, gem - PB[L3, L2] * X5 + PA[L9, L] * hyb[9] - PB[L9, L5] * hyb[19])
    gem = Pt[L9, L7] * hyb[21] + Pt[L4, L2] * hyb[11] + Pt[L6, L5] * hyb[19]
    addA(L8, L6, gem + PB[L6, L] * hyb[9] - PA[L6, L5] * X11 - PA[L4, L2] * X5 + PA[L9, L7] * hyb[21] / 2.0)
    addB(L8, L6, gem + PA[L6, L] * hyb[9] - PB[L6, L5] * X11 - PB[L4, L2] * X5 + PB[L9, L7] * hyb[21] / 2.0)
    gem = -Pt[L8, L7] * hyb[21] + Pt[L4, L3] * hyb[11] + Pt[L7, L5] * hyb[19]
    addA(L9, L6, gem + PB[L7, L] * hyb[9] - PA[L4, L3] * X5 - PA[L7, L5] * X11 - PA[L8, L7] * hyb[21] / 2.0)
    addB(L9, L6, gem + PA[L7, L] * hyb[9] - PB[L4, L3] * X5 - PB[L7, L5] * X11 - PB[L8, L7] * hyb[21] / 2.0)
    gem = -Pt[L8, L5] * hyb[20]
    addA(L7, L7, gem + PB[L5, L] * hyb[7] - PB[L8, L] * hyb[9] + PA[L8, L5] * hyb[19])
    addB(L7, L7, gem + PA[L5, L] * hyb[7] - PA[L8, L] * hyb[9] + PB[L8, L5] * hyb[19])
    gem = -Pt[L4, L3] * hyb[11] - Pt[L9, L6] * hyb[21] - Pt[L7, L5] * hyb[19]
    addA(L8, L7, gem - PB[L7, L] * hyb[9] + PA[L7, L5] * X11 + PA[L4, L3] * X5 - PA[L9, L6] * hyb[21] / 2.0)
    addB(L8, L7, gem - PA[L7, L] * hyb[9] + PB[L7, L5] * X11 + PB[L4, L3] * X5 - PB[L9, L6] * hyb[21] / 2.0)
    gem = Pt[L4, L2] * hyb[11] + Pt[L8, L6] * hyb[21] + Pt[L6, L5] * hyb[19]
    addA(L9, L7, gem + PA[L4, L2] * X5 - PA[L6, L5] * X11 + PA[L8, L6] * hyb[21] / 2.0 + PB[L6, L] * hyb[9])
    addB(L9, L7, gem + PB[L4, L2] * X5 - PB[L6, L5] * X11 + PB[L8, L6] * hyb[21] / 2.0 + PA[L6, L] * hyb[9])
    addA(L8, L8, -PB[L5, L] * hyb[6])
    addB(L8, L8, -PA[L5, L] * hyb[6])
    addA(L9, L9, -PB[L5, L] * hyb[6])
    addB(L9, L9, -PA[L5, L] * hyb[6])


# --------------------------------------------------------------------------- #
# Molecular nuclear gradient + geometry optimization.                         #
# --------------------------------------------------------------------------- #


def msindo_gradient_fd(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=1,
    nddo=False,
    atoms=None,
    max_iter=200,
    conv_tol=1e-10,
    step=1e-3,
):
    """Finite-difference nuclear gradient (Ha/bohr) of the MSINDO total energy.

    Central differences of :func:`run_msindo` (INDO or, with ``nddo=True``, NDDO).
    The energy surface is oracle-validated, so the FD gradient reproduces MSINDO's
    analytic gradient (``CARTOPT ANALY``) to finite-difference accuracy.  ``step``
    is the central-difference displacement in Angstrom; ``atoms`` (default all)
    restricts the differentiation to a subset of atom indices (the rest of the
    returned gradient stays zero).  Closed-shell or open-shell, per ``run_msindo``.
    """
    Z = list(atomic_numbers)
    C0 = np.asarray(coords_angstrom, float)

    def _e(C):
        return run_msindo(
            Z,
            C,
            charge=charge,
            multiplicity=multiplicity,
            max_iter=max_iter,
            conv_tol=conv_tol,
            nddo=nddo,
        ).total_energy

    h = step  # Angstrom; convert the per-Angstrom slope to Ha/bohr below.
    which = range(len(Z)) if atoms is None else atoms
    grad = np.zeros((len(Z), 3))
    for i in which:
        for d in range(3):
            cp = C0.copy()
            cp[i, d] += h
            cm = C0.copy()
            cm[i, d] -= h
            grad[i, d] = (_e(cp) - _e(cm)) / (2.0 * h) / ANGSTROM_TO_BOHR
    return grad


def msindo_optimize(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=1,
    nddo=False,
    frozen=None,
    fmax=2e-3,
    max_steps=200,
    conv_tol=1e-10,
):
    """Relax atomic positions on the MSINDO energy surface with the
    finite-difference gradient (L-BFGS-B).  ``frozen`` = atom indices held fixed;
    the rest are relaxed.  ``fmax`` is the max-force convergence target (Ha/bohr).
    Returns ``(relaxed_coords_angstrom, MsindoResult at the relaxed geometry)``.
    """
    from scipy.optimize import minimize

    Z = list(atomic_numbers)
    C0 = np.asarray(coords_angstrom, float)
    frozen = set(frozen or [])
    free = [i for i in range(len(Z)) if i not in frozen]
    if not free:
        raise ValueError("msindo_optimize: all atoms frozen")

    def _coords(x):
        C = C0.copy()
        C[free] = x.reshape(-1, 3)
        return C

    def fun(x):
        C = _coords(x)
        r = run_msindo(
            Z, C, charge=charge, multiplicity=multiplicity, conv_tol=conv_tol, nddo=nddo
        )
        g = msindo_gradient_fd(
            Z,
            C,
            charge=charge,
            multiplicity=multiplicity,
            nddo=nddo,
            atoms=free,
            conv_tol=conv_tol,
        )
        # scipy works in Angstrom -> return Ha/Angstrom (g is Ha/bohr).
        return r.total_energy, (g[free] * ANGSTROM_TO_BOHR).ravel()

    res = minimize(
        fun,
        C0[free].ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": fmax * ANGSTROM_TO_BOHR, "maxiter": max_steps},
    )
    C = _coords(res.x)
    final = run_msindo(
        Z, C, charge=charge, multiplicity=multiplicity, conv_tol=conv_tol, nddo=nddo
    )
    return C, final


# --------------------------------------------------------------------------- #
# Post-SCF: the INDO ERIProvider + MP2 (via the general correlation kernel).   #
# --------------------------------------------------------------------------- #


def _indo_ao_eri(G, blocks):
    """INDO AO-basis 2-electron integral tensor ``(muν|ls)`` from the g / one-
    centre matrix ``G`` (vibe-qc's analogue of MSINDO's ``GMUNU``).

    In the INDO (ZDO) approximation the surviving integrals are the Coulomb
    monopole ``(mumu|νν)`` -- one-centre GSS/GSP/GPP/GP2 (``G`` upper triangle) plus
    the two-centre g (``G`` off-diagonal, symmetric) -- and the one-centre
    exchange ``(muν|muν) = (muν|νmu)`` = HSP/HPP (``G`` lower triangle, same atom).
    This tensor reproduces ``_build_fock``'s 2e Fock (J - 1/2K) exactly, so MP2 /
    CIS / OVGF over it are consistent with the SCF.  This is the semiempirical
    backend of :class:`vibeqc.correlation.ERIProvider`.
    """
    n = G.shape[0]
    atom = np.empty(n, dtype=int)
    for a, (lo, hi) in enumerate(blocks):
        atom[lo:hi] = a
    same = atom[:, None] == atom[None, :]
    coul = np.triu(G) + np.triu(G, 1).T  # symmetric Coulomb (mumu|νν)
    xch = np.tril(G, -1)
    xch = np.where(same, xch, 0.0)  # one-centre exchange only
    xch = xch + xch.T
    g = np.zeros((n, n, n, n))
    idx = np.arange(n)
    g[idx[:, None], idx[:, None], idx[None, :], idx[None, :]] = coul  # (mumu|νν)
    for mu in range(n):
        for nu in range(n):
            if mu != nu and same[mu, nu]:
                g[mu, nu, mu, nu] += xch[mu, nu]  # (muν|muν)
                g[mu, nu, nu, mu] += xch[mu, nu]  # (muν|νmu)
    return g


def _set8(g, mu, nu, la, sg, val):
    """Assign a real ERI ``(muν|ls)=val`` to all eight index permutations."""
    for a, b in ((mu, nu), (nu, mu)):
        for c, d in ((la, sg), (sg, la)):
            g[a, b, c, d] = val
            g[c, d, a, b] = val


def _nddo_ao_eri(Z, C, blocks, G):
    """NDDO AO 2-electron tensor for the source-defined s/p multipole scope.

    Built on :func:`_indo_ao_eri` (monopole ``(mumu|νν)`` + one-centre exchange),
    this adds the s-p *transition-density* couplings NDDO keeps and INDO drops,
    in the Dewar/MNDO point-charge multipole model -- the same integral kernels
    (:func:`nddo_spss_si` / :func:`nddo_sppp_si` / :func:`nddo_spsp_si` /
    :func:`nddo_spsp_pi`) that build the NDDO Fock.  For an ordered atom pair (I
    carrying a p shell, partner J) along the unit bond vector ``E = (C_J-C_I)/R``:

    * dipole(I)-s-monopole(J):  ``(s_I p_Ia | s_J s_J) = E_a . spss(I,J)``
      (``nddofockcl.f:45/76``),
    * dipole(I)-p-monopole(J):  ``(s_I p_Ia | p_Jb p_Jb) = E_a . sppp(I,J)``
      (``nddofockcl.f:53/82``),
    * dipole(I)-dipole(J):      ``(s_I p_Ia | s_J p_Jb)
      = E_a E_b . spsp_si(I,J) + (d_ab - E_a E_b) . spsp_pi(I,J)``
      (the s-p s/pi rotation, ``spspfockcl.f:43-56``).

    For s/p systems, contracting this tensor as ``J - 1/2K`` reproduces the NDDO
    2-electron Fock (``_build_fock`` + :func:`_nddo_fock_extra`) to machine
    precision. MSINDO's Al-Cl ``SPDD`` contributions are effective Coulomb-only
    Fock branches: adding ``(s_I p_I | d_J d_J)`` through :func:`_set8` would also
    create cross-atom exchange absent from ``nddofockcl.f``. They are therefore
    deliberately not represented here. Must be called inside :func:`_nddo_params`.

    **Not used by** :func:`msindo_mp2` **(nddo=True), by design.**  MSINDO's
    post-SCF integral transform (``pqrs.f`` / ``rstu.f``, driving MP2 / OVGF /
    CIS) reads only the monopole + one-centre ``GMUNU`` set -- the *INDO* tensor --
    even in NDDO mode (``mp2rhf.f`` / ``mp2uhf.f`` carry no NDDO branch); the
    multipole corrections enter post-SCF *only* through the NDDO-converged MOs.
    Validated:
    NDDO-MP2 over the INDO tensor matches the oracle to sub-µHa, while NDDO-MP2
    over this richer tensor gives a *different*, non-oracle number (off by
    1-16 mHa). This tensor is therefore kept as the verified s/p NDDO integral
    set (and the building block a future rigorous NDDO post-SCF method would
    consume), not wired into the parity MP2 path.
    """
    g = _indo_ao_eri(G, blocks).copy()
    eye3 = np.eye(3)
    for I in range(len(Z)):
        loI, hiI = blocks[I]
        if hiI - loI < 4:  # I has no p shell -> no dipole
            continue
        for J in range(len(Z)):
            if I == J:
                continue
            loJ, hiJ = blocks[J]
            pJ = (hiJ - loJ) >= 4
            d = C[J] - C[I]
            R = float(np.linalg.norm(d))
            E = d / R
            spss = nddo_spss_si(Z[I], Z[J], R)  # dipole(I)-s-monopole(J)
            for a in range(3):
                _set8(g, loI, loI + 1 + a, loJ, loJ, E[a] * spss)
            if pJ:
                sppp = nddo_sppp_si(Z[I], Z[J], R)  # dipole(I)-p-monopole(J)
                si = nddo_spsp_si(Z[I], Z[J], R)  # dipole-dipole s
                pi = nddo_spsp_pi(Z[I], Z[J], R)  # dipole-dipole pi
                for a in range(3):
                    for b in range(3):
                        _set8(
                            g, loI, loI + 1 + a, loJ + 1 + b, loJ + 1 + b, E[a] * sppp
                        )
                        s = E[a] * E[b] * si + (eye3[a, b] - E[a] * E[b]) * pi
                        _set8(g, loI, loI + 1 + a, loJ, loJ + 1 + b, s)
    return g


@dataclass
class MsindoMP2Result:
    """MSINDO MP2 result (Hartree).  ``e_corr`` is after any spin-component
    scaling; ``e_os``/``e_ss`` are the unscaled components."""

    e_scf: float
    e_corr: float
    e_total: float
    e_os: float
    e_ss: float
    variant: str
    converged: bool


def msindo_mp2(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    variant="mp2",
    nddo=False,
    max_iter=200,
    conv_tol=1e-10,
):
    """Closed-shell **INDO MP2** (and SCS-/SOS-MP2) on the MSINDO reference.

    Runs the RHF MSINDO SCF, exposes its INDO integral set as the
    :class:`~vibeqc.correlation.ERIProvider` ``(ia|jb)`` block, and evaluates the
    general :func:`vibeqc.correlation.mp2_energy` kernel -- so MSINDO-MP2 shares
    the exact same correlated code as HF/DFT-MP2.  Reproduces reference MSINDO
    MP2 to sub-µHa (SCF-convergence-limited).  ``variant`` in {"mp2","scs-mp2",
    "sos-mp2"}.

    ``nddo=True`` runs MP2 on the **NDDO** reference (the separate NDDO
    parametrization + the two-centre multipole 2e Fock, MSINDO's default mode).
    Note the integral set: MSINDO's MP2 transform (``mp2rhf.f``) reads only the
    monopole + one-centre ``GMUNU`` (INDO) integrals even under NDDO -- the
    two-centre multipoles enter MP2 *only* through the NDDO-converged MOs/orbital
    energies, not the correlation integrals.  vibe-qc reproduces that exactly
    (sub-µHa vs the oracle on HF/H₂O/CH₄/N₂/CO); the richer :func:`_nddo_ao_eri`
    multipole tensor would give a different, non-oracle number and is *not* used
    here (see its docstring).  Open-shell (odd electron count) -> use
    :func:`msindo_ump2`.
    """
    from vibeqc.correlation import mp2_energy

    Z = list(atomic_numbers)
    supported = _SUPPORTED_NDDO if nddo else _SUPPORTED
    bad = sorted({z for z in Z if z not in supported})
    if bad:
        raise NotImplementedError(
            f"MSINDO {'NDDO ' if nddo else ''}MP2 supports {sorted(supported)}; "
            f"got Z={bad}."
        )
    C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec % 2 != 0:
        raise NotImplementedError(
            "MSINDO MP2 is closed-shell (RHF reference) only; the valence "
            "electron count is odd -- use msindo_ump2 for the open-shell UMP2."
        )
    nocc = nelec // 2
    e_core = _core_repulsion(C, cz)
    with _nddo_params() if nddo else contextlib.nullcontext():
        H, G = _build_core_and_gamma(Z, C, blocks, nsto, nddo=nddo)
        fock_extra = _nddo_fock_extra(Z, C, blocks) if nddo else None
        P, _F, e_elec, _eps, converged, _it = _scf_rhf(
            H,
            G,
            blocks,
            Z,
            nocc,
            max_iter=max_iter,
            conv_tol=conv_tol,
            fock_extra=fock_extra,
        )
        e_scf = e_elec + e_core
        # Canonical MOs from the converged (NDDO or INDO) Fock.
        fock = _build_fock(H, G, P, blocks, Z)
        if fock_extra is not None:
            fock = fock + fock_extra(P)[0]
        eps, cmo = np.linalg.eigh(fock)
        # The MP2 integral transform uses the INDO GMUNU set even in NDDO mode
        # (mp2rhf.f has no NDDO branch -- _nddo_ao_eri docstring).
        g_ao = _indo_ao_eri(G, blocks)
    co, cv = cmo[:, :nocc], cmo[:, nocc:]
    ovov = np.einsum("mnls,mi,na,lj,sb->iajb", g_ao, co, cv, co, cv, optimize=True)
    mp2 = mp2_energy(eps[:nocc], eps[nocc:], ovov, variant=variant)
    return MsindoMP2Result(
        e_scf=e_scf,
        e_corr=mp2.e_corr,
        e_total=e_scf + mp2.e_corr,
        e_os=mp2.e_os,
        e_ss=mp2.e_ss,
        variant=variant,
        converged=bool(converged),
    )


@dataclass
class MsindoUMP2Result:
    """Open-shell MSINDO UMP2 result (Hartree), spin-channel resolved.

    ``e_corr`` is the *physically correct* UMP2 (after any spin-component
    scaling); ``e_aa``/``e_bb``/``e_ab`` are the unscaled aa / bb / ab channels.
    See :func:`msindo_ump2` for the MSINDO ``mp2uhf.f`` aa bug this does NOT
    reproduce."""

    e_scf: float
    e_corr: float
    e_total: float
    e_aa: float
    e_bb: float
    e_ab: float
    variant: str
    converged: bool


def msindo_ump2(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=2,
    variant="mp2",
    max_iter=200,
    conv_tol=1e-10,
):
    """Open-shell **INDO UMP2** on an unrestricted (UHF) MSINDO reference.

    Runs the UHF MSINDO SCF, then evaluates the spin-resolved unrestricted MP2
    (aa, bb, ab) over the INDO integral set via the general
    :func:`vibeqc.correlation.ump2_energy` kernel -- so MSINDO-UMP2 shares the
    correlated code with the HF/DFT UMP2 (``cpp/src/ump2.cpp``).  s/p elements
    (the open-shell INDO Fock; the d-block UHF Fock is not yet ported).
    ``variant`` in {"mp2","scs-mp2","sos-mp2"}.

    **Parity note -- MSINDO's ``mp2uhf.f`` aa bug.**  vibe-qc ships the
    *physically correct* UMP2.  Reference MSINDO's ``mp2uhf.f`` multiplies its aa
    same-spin channel by a spurious 1/4: line 62 applies ``0.25`` on the restricted
    ``i<j, a<b`` sum, where the bb channel at line 77 (identical loops) correctly
    uses ``1.0`` -- and the routine's own formula header gives factor ``1.0`` for
    *both* restricted same-spin sums, so the aa ``0.25`` is a copy-paste of the
    unrestricted prefactor.  Consequently vibe-qc's bb and ab channels reproduce
    the oracle to sub-µHa, but the total differs by 3/4.E_aa whenever an aa channel
    exists (>=2 a-occupied and >=2 a-virtual orbitals).  vibe-qc's total is instead
    consistent with its own RHF-MP2 and C++ UMP2 (verified in
    ``tests/test_msindo.py`` and ``tests/test_correlation_mp2.py``).
    """
    from vibeqc.correlation import ump2_energy

    Z = list(atomic_numbers)
    bad = sorted({z for z in Z if z not in _SUPPORTED})
    if bad:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(_SUPPORTED)}; got Z={bad}."
        )
    C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    nalpha, nbeta = _uhf_occupation(nelec, multiplicity)
    H, G = _build_core_and_gamma(Z, C, blocks, nsto)
    e_core = _core_repulsion(C, cz)
    PA, PB, CA, CB, epsA, epsB, e_elec, converged, _it = _scf_uhf(
        H, G, blocks, Z, nalpha, nbeta, max_iter=max_iter, conv_tol=conv_tol
    )
    e_scf = e_elec + e_core
    g_ao = _indo_ao_eri(G, blocks)
    CAo, CAv = CA[:, :nalpha], CA[:, nalpha:]
    CBo, CBv = CB[:, :nbeta], CB[:, nbeta:]

    def _ovov(Co, Cv, Co2, Cv2):
        # (ia|jb) MO block; a 0-width occ/vir (e.g. nb=0) yields an empty tensor
        # that the ump2_energy kernel reads as a 0 channel.
        return np.einsum(
            "mnls,mi,na,lj,sb->iajb", g_ao, Co, Cv, Co2, Cv2, optimize=True
        )

    ump2 = ump2_energy(
        epsA[:nalpha],
        epsA[nalpha:],
        epsB[:nbeta],
        epsB[nbeta:],
        _ovov(CAo, CAv, CAo, CAv),
        _ovov(CBo, CBv, CBo, CBv),
        _ovov(CAo, CAv, CBo, CBv),
        variant=variant,
    )
    return MsindoUMP2Result(
        e_scf=e_scf,
        e_corr=ump2.e_corr,
        e_total=e_scf + ump2.e_corr,
        e_aa=ump2.e_aa,
        e_bb=ump2.e_bb,
        e_ab=ump2.e_ab,
        variant=variant,
        converged=bool(converged),
    )


@dataclass(frozen=True)
class MsindoOVGFResult:
    """INDO quasiparticle (GF2 / Green's-function) result.

    ``orbitals`` are the 0-based MO indices corrected; the per-orbital arrays
    carry the SCF (Koopmans) energy, the quasiparticle energy, and the pole
    strength (all energies in Hartree).  ``homo`` is the HOMO index; the HOMO
    IP is ``-eps_qp[orbitals.index(homo)]``.  The LUMO (``homo + 1``)
    quasiparticle energy gives the electron affinity, EA = ``-eps_qp(LUMO)``
    (see :attr:`ea_lumo_ev` and its caveat).
    """

    orbitals: tuple
    eps_scf: object  # np.ndarray (Hartree)
    eps_qp: object  # np.ndarray (Hartree)
    pole_strength: object  # np.ndarray
    homo: int
    converged: bool

    @property
    def ip_homo_ev(self) -> float:
        """OVGF/GF2 ionization potential from the HOMO (eV)."""
        i = list(self.orbitals).index(self.homo)
        return float(-self.eps_qp[i] * 27.211386245988)

    @property
    def ea_lumo_ev(self) -> float:
        """GF2 electron affinity from the LUMO (eV), EA = ``-eps_qp(LUMO)``.

        **Caveat -- small-basis EAs are unreliable.**  The diagonal
        second-order (GF2) self-energy gives qualitative EAs at best in a
        minimal/valence basis: the virtual orbitals are not variationally
        optimised and the basis lacks the diffuse functions a bound anion
        needs, so a positive ``-eps_qp(LUMO)`` here does *not* establish a
        stable anion.  MSINDO's valence STO basis is minimal, so treat the
        MSINDO EA as a trend indicator only; for quantitative EAs use an
        augmented (diffuse) Gaussian basis with an HF/DFT reference.
        """
        lumo = self.homo + 1
        if lumo not in self.orbitals:
            raise ValueError(
                f"the LUMO (MO {lumo}) was not among the corrected orbitals "
                f"{self.orbitals}; pass orbitals=(..., {lumo}) to msindo_ovgf."
            )
        i = list(self.orbitals).index(lumo)
        return float(-self.eps_qp[i] * 27.211386245988)


@dataclass(frozen=True)
class MsindoUOVGFResult:
    """Open-shell (UHF reference) INDO quasiparticle (GF2) result.

    Spin-resolved sibling of :class:`MsindoOVGFResult`.  ``orbitals`` is a tuple
    of ``(spin, idx)`` pairs (``spin in {"alpha", "beta"}``, ``idx`` the 0-based
    spatial MO index) for the corrected spin-orbitals; the per-orbital arrays
    carry the Koopmans / quasiparticle energies and pole strengths (Hartree).
    ``n_alpha``/``n_beta`` are the occupied counts per spin (an orbital is
    occupied iff ``idx < n_{spin}``).  IP = ``-eps_qp`` for an occupied
    spin-orbital; the first vertical IP is the smallest over the corrected
    occupied set, the EA the value from the lowest corrected virtual.
    """

    orbitals: tuple
    eps_scf: object  # np.ndarray (Hartree)
    eps_qp: object  # np.ndarray (Hartree)
    pole_strength: object  # np.ndarray
    n_alpha: int
    n_beta: int
    converged: bool

    def _is_occupied(self, spin, idx) -> bool:
        return idx < (self.n_alpha if spin == "alpha" else self.n_beta)

    @property
    def first_ip_ev(self) -> float:
        """First vertical IP (eV): smallest ``-eps_qp`` over corrected occupied
        spin-orbitals (the most weakly bound electron, either spin)."""
        ips = [
            -float(e) * 27.211386245988
            for (sp, ix), e in zip(self.orbitals, self.eps_qp)
            if self._is_occupied(sp, ix)
        ]
        if not ips:
            raise ValueError("no occupied spin-orbital was corrected.")
        return min(ips)

    @property
    def ea_ev(self) -> float:
        """Electron affinity (eV) from the lowest corrected virtual spin-orbital,
        EA = ``-eps_qp``.  Small-basis caveat as in
        :attr:`MsindoOVGFResult.ea_lumo_ev`: qualitative only."""
        eas = [
            -float(e) * 27.211386245988
            for (sp, ix), e in zip(self.orbitals, self.eps_qp)
            if not self._is_occupied(sp, ix)
        ]
        if not eas:
            raise ValueError("no virtual spin-orbital was corrected.")
        return max(eas)  # lowest virtual => largest (least-negative) EA


def _emit_msindo_ovgf_citations(output):
    """Best-effort ``.bibtex`` / ``.references`` for an INDO GF2/OVGF run: MSINDO
    + the NDDO-coupled outer-valence Green's-function treatment (Danovich 1997)
    on top of the OVGF formalism (Cederbaum 1975, von Niessen 1984).  INDO uses
    no Gaussian integrals, so libint is suppressed; the INDO SCF is DIIS-driven
    (``uses_scf`` left on).  The ``routes.methods.ovgf`` row cannot be used here:
    it fires for the *ab-initio* ``run_job(method="ovgf")`` path, whose method
    string is ``"ovgf"`` rather than ``"msindo"``.  Non-fatal (mirrors
    ``_emit_msindo_cisd_citations``)."""
    try:
        from vibeqc.output.citations import emit_citations

        emit_citations(
            output,
            method="msindo",
            uses_integrals=False,
            extra_entries=[
                "danovich_ovgf_1997",
                "cederbaum_ovgf_1975",
                "vonniessen_ovgf_1984",
            ],
        )
    except Exception:
        pass


def msindo_ovgf(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=None,
    orbitals=None,
    renormalize=False,
    iterate=True,
    max_iter=200,
    conv_tol=1e-10,
    output=None,
):
    """INDO quasiparticle energies (GF2 / second-order electron propagator).

    Runs the MSINDO SCF, builds its *rigorous* INDO MO two-electron tensor (the
    same integral set as :func:`msindo_mp2`, via :func:`_indo_ao_eri`), and
    evaluates the general diagonal second-order self-energy from
    :mod:`vibeqc.propagator` -- so MSINDO, HF, and DFT share one Green's-function
    code path.

    Closed shell (``multiplicity == 1``, even valence count) runs the RHF
    reference and returns a :class:`MsindoOVGFResult` (spatial HOMO/LUMO by
    default).  Open shell (an odd valence count, or an explicit
    ``multiplicity > 1``) runs the UHF reference (s/p elements, H-F) and returns
    a :class:`MsindoUOVGFResult` from the *spin-resolved* second-order
    self-energy -- the electron-propagator analogue of UMP2 -- correcting the a/b
    frontier orbitals by default.

    ``orbitals``: closed-shell -- 0-based MO indices (default HOMO + LUMO);
    open-shell -- ``(spin, idx)`` pairs (default the a/b HOMOs and LUMOs).
    ``iterate``: solve the Dyson equation self-consistently (default) or take the
    non-iterative ``S(e_p)`` estimate.  ``renormalize=True`` swaps the bare
    diagonal GF2 self-energy for the **renormalized** one (full third-order
    self-energy + geometric screening -- see
    :func:`vibeqc.propagator.renormalized_quasiparticle_energies`), which trims
    the GF2 overcorrection toward experiment; small systems only (the
    third-order contractions scale steeply).

    ``output``: path-like, optional.  Write ``{output}.bibtex`` /
    ``.references`` citing MSINDO + the NDDO outer-valence Green's-function
    treatment (Danovich 1997) over the OVGF formalism (Cederbaum 1975,
    von Niessen 1984).

    Note: vibe-qc computes the *rigorous* ZDO self-energy, which differs from
    MSINDO's ``ovgfrhf_neu`` factorized-integral approximation (see
    :mod:`vibeqc.propagator` docstring).  The rigorous value is consistent with
    MSINDO-MP2.
    """
    from vibeqc.propagator import (
        quasiparticle_energy,
        renormalized_quasiparticle_energies,
        unrestricted_quasiparticle_energies,
    )

    Z = list(atomic_numbers)
    bad = sorted({z for z in Z if z not in _SUPPORTED})
    if bad:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(_SUPPORTED)}; got Z={bad}."
        )
    C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if multiplicity is None:
        multiplicity = 1 if nelec % 2 == 0 else 2
    H, G = _build_core_and_gamma(Z, C, blocks, nsto)
    g_ao = _indo_ao_eri(G, blocks)

    if multiplicity == 1 and nelec % 2 == 0:
        # Closed-shell RHF reference (spatial second-order self-energy).
        nocc = nelec // 2
        P, _F, _e_elec, _eps, converged, _it = _scf_rhf(
            H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol
        )
        eps, cmo = np.linalg.eigh(_build_fock(H, G, P, blocks, Z))
        homo = nocc - 1
        if orbitals is None:
            orbitals = (homo, nocc)  # HOMO, LUMO
        orbitals = tuple(int(o) for o in orbitals)
        if renormalize:
            # Renormalized self-energy via the spin-orbital path (Ca = Cb = cmo).
            rq = renormalized_quasiparticle_energies(
                g_ao,
                cmo,
                cmo,
                eps,
                eps,
                nocc,
                nocc,
                [("alpha", o) for o in orbitals],
                iterate=iterate,
                max_iter=max_iter,
            )
            qps = [q for _sp, q in rq]
        else:
            # Full MO two-electron tensor (pq|rs); nsto is small for INDO.
            g_mo = np.einsum(
                "mnls,mp,nq,lr,st->pqrt", g_ao, cmo, cmo, cmo, cmo, optimize=True
            )
            qps = [
                quasiparticle_energy(
                    eps, nocc, g_mo, p, iterate=iterate, max_iter=max_iter
                )
                for p in orbitals
            ]
        if output is not None:
            _emit_msindo_ovgf_citations(output)
        return MsindoOVGFResult(
            orbitals=orbitals,
            eps_scf=np.array([q.eps_scf for q in qps]),
            eps_qp=np.array([q.eps_qp for q in qps]),
            pole_strength=np.array([q.pole_strength for q in qps]),
            homo=homo,
            converged=bool(converged) and all(q.converged for q in qps),
        )

    # Open-shell UHF reference (spin-resolved second-order self-energy).
    nalpha, nbeta = _uhf_occupation(nelec, multiplicity)
    PA, PB, CA, CB, epsA, epsB, _e, converged, _it = _scf_uhf(
        H, G, blocks, Z, nalpha, nbeta, max_iter=max_iter, conv_tol=conv_tol
    )
    if orbitals is None:
        # a/b frontier orbitals: spin-HOMO (IP) + spin-LUMO (EA) per channel.
        frontier = []
        for sp, nocc_s in (("alpha", nalpha), ("beta", nbeta)):
            if nocc_s >= 1:
                frontier.append((sp, nocc_s - 1))
            if nocc_s < nsto:
                frontier.append((sp, nocc_s))
        orbitals = tuple(frontier)
    else:
        orbitals = tuple((str(sp), int(ix)) for sp, ix in orbitals)
    _qp_fn = (
        renormalized_quasiparticle_energies
        if renormalize
        else unrestricted_quasiparticle_energies
    )
    res = _qp_fn(
        g_ao,
        CA,
        CB,
        epsA,
        epsB,
        nalpha,
        nbeta,
        list(orbitals),
        iterate=iterate,
        max_iter=max_iter,
    )
    if output is not None:
        _emit_msindo_ovgf_citations(output)
    return MsindoUOVGFResult(
        orbitals=orbitals,
        eps_scf=np.array([q.eps_scf for _sp, q in res]),
        eps_qp=np.array([q.eps_qp for _sp, q in res]),
        pole_strength=np.array([q.pole_strength for _sp, q in res]),
        n_alpha=nalpha,
        n_beta=nbeta,
        converged=bool(converged) and all(q.converged for _sp, q in res),
    )


def _msindo_cis_core(
    Z,
    coords_angstrom,
    *,
    charge=0,
    spin="singlet",
    n_states=5,
    max_iter=200,
    conv_tol=1e-10,
):
    """Shared INDO CIS engine: returns ``(e_scf_total, CISResult)``.

    Runs the closed-shell RHF MSINDO SCF once and evaluates the general
    :func:`vibeqc.excited.cis_excitations` kernel over its INDO MO blocks.  Used
    by :func:`msindo_cis` (which keeps only the :class:`CISResult`) and by
    :func:`msindo_cis_stateset` / the excited-state gradient adapters (which also
    need the ground-state energy ``E_SCF`` to form excited-state *total*
    energies ``E_SCF + w``).
    """
    from vibeqc.excited import cis_excitations

    bad = sorted({z for z in Z if z not in _SUPPORTED})
    if bad:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(_SUPPORTED)}; got Z={bad}."
        )
    C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec % 2 != 0:
        raise NotImplementedError(
            "MSINDO CIS is closed-shell (RHF reference) only; the valence "
            "electron count is odd (open-shell CIS is a follow-up)."
        )
    nocc = nelec // 2
    H, G = _build_core_and_gamma(Z, C, blocks, nsto)
    e_core = _core_repulsion(C, cz)
    P, _F, e_elec, _eps, converged, _it = _scf_rhf(
        H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol
    )
    if not converged:
        raise RuntimeError("MSINDO SCF did not converge; CIS unavailable.")
    eps, cmo = np.linalg.eigh(_build_fock(H, G, P, blocks, Z))
    g_ao = _indo_ao_eri(G, blocks)
    co, cv = cmo[:, :nocc], cmo[:, nocc:]
    ovov = np.einsum("mnls,mi,na,lj,sb->iajb", g_ao, co, cv, co, cv, optimize=True)
    oovv = np.einsum("mnls,mi,nj,la,sb->ijab", g_ao, co, co, cv, cv, optimize=True)
    cis = cis_excitations(eps, nocc, ovov, oovv, spin=spin, n_states=n_states)
    return e_elec + e_core, cis


def msindo_cis(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    spin="singlet",
    n_states=5,
    max_iter=200,
    conv_tol=1e-10,
):
    """INDO CIS / Tamm-Dancoff excited states (singlet or triplet).

    Runs the closed-shell RHF MSINDO SCF, builds its INDO MO two-electron
    blocks (the same integral set as :func:`msindo_mp2` / :func:`msindo_ovgf`,
    via :func:`_indo_ao_eri`), and evaluates the general
    :func:`vibeqc.excited.cis_excitations` kernel -- so MSINDO, HF, and DFT share
    one excited-state code path.  Returns a
    :class:`vibeqc.excited.CISResult` (vertical excitation energies + CIS
    amplitudes).

    Note: this is the *rigorous* INDO CIS.  Reference MSINDO additionally
    applies empirical CIS scaling knobs (``SCALEDCIS``, ``UJCORR`` /
    ``CISCORR``) for spectroscopic fitting, so its excitation energies differ
    from the unscaled values here (cf. the OVGF factorization note in
    :mod:`vibeqc.propagator`)."""
    _e_scf, cis = _msindo_cis_core(
        list(atomic_numbers),
        coords_angstrom,
        charge=charge,
        spin=spin,
        n_states=n_states,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    return cis


def msindo_cis_stateset(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    spin="singlet",
    n_states=5,
    max_iter=200,
    conv_tol=1e-10,
):
    """INDO CIS as a :class:`vibeqc.excited_gradient.CISStateSet`.

    Same SCF + CIS as :func:`msindo_cis`, but bundles the ground-state SCF total
    energy with the excited states so callers can form excited-state *total*
    energies ``E_SCF + w`` (what nuclear gradients and the conical-intersection
    optimizer differentiate).
    """
    from vibeqc.excited_gradient import CISStateSet

    e_scf, cis = _msindo_cis_core(
        list(atomic_numbers),
        coords_angstrom,
        charge=charge,
        spin=spin,
        n_states=n_states,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    return CISStateSet(e_ground=e_scf, cis=cis)


def make_msindo_cis_energy_fn(
    atomic_numbers,
    *,
    charge=0,
    spin="singlet",
    n_states=5,
    max_iter=200,
    conv_tol=1e-10,
):
    """Build the reference-agnostic CIS energy function for MSINDO.

    Returns ``energy_fn(coords_angstrom) -> CISStateSet`` -- the MSINDO backend of
    the :data:`vibeqc.excited_gradient.CISEnergyFn` seam.  Feed it to
    :func:`vibeqc.excited_gradient.cis_state_gradients_fd` for INDO excited-state
    gradients, or to :func:`vibeqc.conical.optimize_conical_intersection`.
    """
    Z = list(atomic_numbers)

    def energy_fn(coords_angstrom):
        return msindo_cis_stateset(
            Z,
            coords_angstrom,
            charge=charge,
            spin=spin,
            n_states=n_states,
            max_iter=max_iter,
            conv_tol=conv_tol,
        )

    return energy_fn


def msindo_cis_gradient_fd(
    atomic_numbers,
    coords_angstrom,
    state,
    *,
    charge=0,
    spin="singlet",
    n_states=None,
    step=1e-3,
    max_iter=200,
    conv_tol=1e-10,
    atoms=None,
    output=None,
):
    """Finite-difference nuclear gradient (Ha/bohr) of an INDO CIS state.

    ``state`` is 0 for the ground state or ``k >= 1`` for the ``k``-th CIS root
    (spectroscopic S0, S1, ...).  The excited-state total energy ``E_SCF + w`` is
    central-differenced over displaced geometries with CIS-amplitude state
    tracking (root-flip guard); see
    :func:`vibeqc.excited_gradient.cis_state_gradients_fd`.  ``step`` is the
    displacement in Angstrom.  Returns the gradient shaped ``(natom, 3)``.

    The analytic INDO CIS gradient (CPHF/Z-vector, MSINDO ``cisgrad.f``) is a
    later refinement; this finite-difference route is its validation baseline.

    ``output`` (optional path stem): writes ``{stem}.bibtex`` / ``.references``
    citing MSINDO + the CIS method (Foresman et al. 1992), like the other
    standalone semiempirical runners.
    """
    from vibeqc.excited_gradient import cis_state_gradients_fd

    if n_states is None:
        n_states = max(5, int(state))
    elif n_states < int(state):
        raise ValueError(
            f"n_states={n_states} too small for state={state}; need n_states "
            f">= {int(state)} so the requested root is computed."
        )
    fn = make_msindo_cis_energy_fn(
        atomic_numbers,
        charge=charge,
        spin=spin,
        n_states=n_states,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    grad = cis_state_gradients_fd(
        fn, coords_angstrom, int(state), step=step, atoms=atoms
    )[int(state)]
    if output is not None and int(state) >= 1:
        _emit_msindo_cis_citations(output)
    return grad


def _emit_msindo_cis_citations(output, *, conical=False):
    """Best-effort ``.bibtex`` / ``.references`` for an INDO CIS run: MSINDO +
    the CIS method (Foresman 1992), and -- when ``conical`` -- the penalty-function
    conical-intersection optimizer (Levine-Coe-Martínez 2008).  INDO uses no
    Gaussian integrals, so libint is suppressed (``uses_integrals=False``); the
    INDO SCF is DIIS-accelerated (``uses_scf`` left on).  Non-fatal -- citation
    output never tanks a finished calculation (mirrors the runner /
    ``_emit_tddft_citations`` behaviour)."""
    try:
        from vibeqc.output.citations import emit_citations

        emit_citations(
            output,
            method="msindo",
            uses_integrals=False,
            uses_cis=True,
            uses_conical_intersection=conical,
        )
    except Exception:
        pass


def msindo_meci(
    atomic_numbers,
    coords_angstrom,
    *,
    lower_state=0,
    upper_state=1,
    charge=0,
    spin="singlet",
    step=1e-3,
    max_iter=200,
    conv_tol=1e-10,
    output=None,
    **opt_kwargs,
):
    """Optimize a minimum-energy conical intersection (MECI) at the INDO level.

    Drives two tracked CIS states (``lower_state`` / ``upper_state``; 0 = ground,
    k >= 1 = k-th CIS root -- e.g. ``0, 1`` for an S1/S0 intersection) to their
    minimum-energy degeneracy with the method-agnostic penalty-function
    optimizer :func:`vibeqc.conical.optimize_conical_intersection`, using the
    INDO finite-difference CIS gradients.  ``step`` is the FD displacement
    (Angstrom); extra optimizer knobs (``alpha``, ``sigma0``, ``gap_tol``,
    ``max_macro``, ...) pass through as ``opt_kwargs``.  Returns a
    :class:`vibeqc.conical.MECIResult`.

    ``output`` (optional path stem) writes the ``.bibtex`` / ``.references``
    citing MSINDO + CIS (Foresman 1992) + the conical-intersection penalty
    method (Levine-Coe-Martínez 2008).
    """
    from vibeqc.conical import make_cis_meci_fn, optimize_conical_intersection

    n_states = max(5, int(upper_state))
    energy_fn = make_msindo_cis_energy_fn(
        atomic_numbers,
        charge=charge,
        spin=spin,
        n_states=n_states,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    state_eg_fn = make_cis_meci_fn(energy_fn, lower_state, upper_state, step=step)
    result = optimize_conical_intersection(
        state_eg_fn, np.asarray(coords_angstrom, float), **opt_kwargs
    )
    if output is not None:
        _emit_msindo_cis_citations(output, conical=True)
    return result


# --------------------------------------------------------------------------- #
# CISD via vibe-qc's GENERIC CI solver (not MSINDO's selecting rhfcisd.f).     #
# --------------------------------------------------------------------------- #


def _msindo_mo_hamiltonian(
    Z, coords_angstrom, *, charge=0, max_iter=200, conv_tol=1e-10
):
    """Closed-shell INDO reference -> MO-basis :class:`vibeqc.solvers.Hamiltonian`.

    Scope item shared by the CI-based post-SCF methods: run the RHF MSINDO SCF,
    then express the one-electron **core** Hamiltonian and the INDO two-electron
    tensor (:func:`_indo_ao_eri`) in the canonical-MO basis as the integrals a
    generic wavefunction solver consumes -- ``h1e`` (core, *not* the Fock),
    ``h2e`` in physicist's notation, and ``nuclear_repulsion`` = the INDO core
    repulsion.  The AO->MO transform reuses
    :func:`vibeqc.solvers.transform_hamiltonian`.

    Returns ``(ham_mo, mo_energies, nocc, nelec, e_scf, converged)``.  The
    reference-determinant energy built from ``ham_mo`` equals ``e_scf`` to
    machine precision (the INDO tensor reproduces ``_build_fock``'s J-1/2K), which
    is the integral-consistency check the CI methods rely on.
    """
    from vibeqc.solvers import Hamiltonian, transform_hamiltonian
    from vibeqc.solvers._common import _chemist_to_physicist

    bad = sorted({z for z in Z if z not in _SUPPORTED})
    if bad:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(_SUPPORTED)}; got Z={bad}."
        )
    C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec % 2 != 0:
        raise NotImplementedError(
            "MSINDO CI post-SCF is closed-shell (RHF reference) only; the "
            "valence electron count is odd."
        )
    nocc = nelec // 2
    H, G = _build_core_and_gamma(Z, C, blocks, nsto)
    e_core = _core_repulsion(C, cz)
    P, _F, e_elec, _eps, converged, _it = _scf_rhf(
        H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol
    )
    eps, cmo = np.linalg.eigh(_build_fock(H, G, P, blocks, Z))
    g_ao_phys = _chemist_to_physicist(_indo_ao_eri(G, blocks))
    ham_ao = Hamiltonian(
        h1e=H, h2e=g_ao_phys, nuclear_repulsion=e_core, norb=nsto, nelec=nelec, ms2=0
    )
    ham_mo = transform_hamiltonian(ham_ao, cmo)
    return ham_mo, eps, nocc, nelec, e_elec + e_core, bool(converged)


@dataclass
class MsindoCISDResult:
    """INDO CISD result (Hartree).

    ``e_corr`` is the CISD correlation energy (``e_total - e_ref``, with
    ``e_ref`` the reference-determinant energy = SCF energy).  ``reference_weight``
    is the squared CI weight of the reference determinant.  ``ci`` is the full
    generic :class:`vibeqc.solvers.CISDResult` (CI vector, determinants, roots).
    """

    e_scf: float
    e_corr: float
    e_total: float
    reference_weight: float
    n_det: int
    converged: bool
    ci: object = None

    def dominant_configurations(self, n: int = 5, *, min_weight: float = 0.0):
        """Leading ``(description, coefficient, weight)`` configurations of the
        CISD ground state (delegates to the generic
        :meth:`vibeqc.solvers.CISDResult.dominant_configurations`)."""
        return self.ci.dominant_configurations(n, min_weight=min_weight)


def _emit_msindo_cisd_citations(output):
    """Best-effort ``.bibtex`` / ``.references`` for an INDO CISD run: MSINDO +
    the variational-CISD method (Pople-Seeger-Krishnan 1977).  INDO uses no
    Gaussian integrals, so libint is suppressed; the INDO SCF is DIIS-driven
    (``uses_scf`` left on).  Non-fatal (mirrors ``_emit_msindo_cis_citations``)."""
    try:
        from vibeqc.output.citations import emit_citations

        emit_citations(
            output,
            method="msindo",
            uses_integrals=False,
            extra_entries=["pople_cisd_1977"],
        )
    except Exception:
        pass


def msindo_cisd(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    frozen_core=0,
    nroots=1,
    max_iter=200,
    conv_tol=1e-10,
    max_det=20_000,
    output=None,
):
    """Closed-shell **INDO CISD** on the MSINDO reference, via vibe-qc's generic CI.

    Runs the RHF MSINDO SCF, expresses its INDO integral set in the MO basis
    (:func:`_msindo_mo_hamiltonian`), and diagonalises the Hamiltonian in the
    fixed singles-plus-doubles space with vibe-qc's reference-agnostic solver
    (:func:`vibeqc.solvers.cisd`) -- so MSINDO shares the *same* CI machinery
    (:func:`~vibeqc.solvers._slater_condon.build_hamiltonian_matrix_unrestricted`)
    as the ab-initio FCI / CASCI path.  This is deliberately **not** a port of
    MSINDO's selecting determinant-CI driver (``rhfcisd.f``): no empirical CIS
    scalings, no determinant selection -- a clean variational CISD over the INDO
    MOs.

    Because the space is exact for <= 2 correlated electrons, MSINDO CISD on a
    two-electron system reproduces the INDO **FCI** (``casci`` full space) to
    machine precision; in general ``E_FCI <= E_CISD <= E_SCF``.

    Parameters
    ----------
    atomic_numbers, coords_angstrom
        Molecular geometry (Z list + Å coordinates).
    charge : int
        Net charge (must give an even closed-shell electron count).
    frozen_core : int
        Doubly-occupied MOs frozen out of the correlation (default 0 -- MSINDO's
        core is already a pseudopotential, so all valence MOs are correlated).
    nroots : int
        CI roots to report energies for (``ci.e_totals``); ground state = root 0.
    max_det : int or None
        Guard on the CISD determinant-space size.
    output : path-like, optional
        Write ``{output}.bibtex`` / ``.references`` citing MSINDO + CISD
        (Pople-Seeger-Krishnan 1977).

    Returns
    -------
    MsindoCISDResult
    """
    from vibeqc.solvers import cisd as _cisd_solve

    Z = list(atomic_numbers)
    ham, _eps, _nocc, nelec, e_scf, converged = _msindo_mo_hamiltonian(
        Z, coords_angstrom, charge=charge, max_iter=max_iter, conv_tol=conv_tol
    )
    if not converged:
        raise RuntimeError("MSINDO SCF did not converge; CISD unavailable.")
    res = _cisd_solve(
        ham.h1e,
        ham.h2e,
        nelec,
        ham.norb,
        n_core=frozen_core,
        nuclear_repulsion=ham.nuclear_repulsion,
        ms2=0,
        max_excitation=2,
        nroots=nroots,
        max_det=max_det,
    )
    if output is not None:
        _emit_msindo_cisd_citations(output)
    return MsindoCISDResult(
        e_scf=e_scf,
        e_corr=res.e_corr,
        e_total=res.e_total,
        reference_weight=res.reference_weight,
        n_det=res.n_det,
        converged=converged,
        ci=res,
    )


# --------------------------------------------------------------------------- #
# Born-Oppenheimer MD: wire MSINDO into the general velocity-Verlet driver.    #
# --------------------------------------------------------------------------- #


def msindo_force_provider(
    atomic_numbers,
    *,
    charge=0,
    multiplicity=1,
    nddo=False,
    conv_tol=1e-9,
    grad_step=1e-3,
    max_iter=200,
):
    """Build an energy+gradient callable backed by the MSINDO engine.

    Returns ``fn(coords_bohr) -> (energy_Ha, gradient_Ha_per_bohr)`` -- the
    :data:`vibeqc.md.ForceProvider` interface the general MD / metadynamics
    drivers consume.  The energy comes from :func:`run_msindo` and the
    nuclear gradient from :func:`msindo_gradient_fd` (central differences).

    MSINDO is Ångström-facing; this adapter converts the bohr coordinates the
    MD driver carries.  Each evaluation costs ``6N + 1`` SCFs (the FD
    gradient), so keep trajectories short.
    """
    z = [int(a) for a in atomic_numbers]

    def provider(coords_bohr):
        coords_ang = np.asarray(coords_bohr, float) / ANGSTROM_TO_BOHR
        e = run_msindo(
            z,
            coords_ang,
            charge=charge,
            multiplicity=multiplicity,
            max_iter=max_iter,
            conv_tol=conv_tol,
            nddo=nddo,
        ).total_energy
        grad = msindo_gradient_fd(
            z,
            coords_ang,
            charge=charge,
            multiplicity=multiplicity,
            nddo=nddo,
            conv_tol=conv_tol,
            step=grad_step,
            max_iter=max_iter,
        )
        return float(e), np.asarray(grad, float)

    return provider


def _emit_msindo_md_citations(output, *, thermostat=None):
    """Best-effort ``.bibtex`` / ``.references`` for an INDO MD run: MSINDO +
    the velocity-Verlet integrator (Swope 1982) and, when a thermostat ran,
    the Berendsen (1984) or Nosé-Hoover (Nosé 1984 / Hoover 1985) NVT papers.
    INDO uses no Gaussian integrals (``uses_integrals=False``); the SCF is
    DIIS-accelerated (``uses_scf`` left on).  Non-fatal (mirrors
    ``_emit_msindo_cis_citations``)."""
    md_thermostat = None if (thermostat in (None, "nve")) else thermostat
    try:
        from vibeqc.output.citations import emit_citations

        emit_citations(
            output,
            method="msindo",
            uses_integrals=False,
            uses_md=True,
            md_thermostat=md_thermostat,
        )
    except Exception:
        pass


def run_msindo_md(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=1,
    nddo=False,
    conv_tol=1e-9,
    grad_step=1e-3,
    max_iter=200,
    timestep_fs=0.5,
    n_steps=100,
    temperature_K=300.0,
    thermostat=None,
    thermostat_tau_fs=50.0,
    velocities=None,
    remove_com=True,
    seed=None,
    record_stride=1,
    output=None,
):
    """Born-Oppenheimer molecular dynamics on the MSINDO surface.

    A thin MSINDO-aware wrapper over the general velocity-Verlet driver
    :func:`vibeqc.md.run_md`: it builds the MSINDO energy+gradient provider
    (:func:`msindo_force_provider`), takes the *initial* geometry in Ångström,
    and returns a :class:`vibeqc.md.MDTrajectory` (positions/velocities in
    atomic units -- bohr, bohr/aut -- energies in Hartree).

    ``thermostat`` selects the ensemble: ``None`` -> NVE; ``"berendsen"`` or
    ``"nose_hoover"`` -> that NVT thermostat (relaxation time
    ``thermostat_tau_fs``).  Masses are taken from the atomic numbers.

    Each step costs ``6N + 1`` MSINDO SCFs (the finite-difference gradient),
    so this is intended for short trajectories on small molecules.

    ``output`` (optional path stem) writes the ``.bibtex`` / ``.references``
    citing MSINDO + velocity-Verlet (Swope 1982) + the thermostat papers.
    """
    from vibeqc.md import run_md

    provider = msindo_force_provider(
        atomic_numbers,
        charge=charge,
        multiplicity=multiplicity,
        nddo=nddo,
        conv_tol=conv_tol,
        grad_step=grad_step,
        max_iter=max_iter,
    )
    coords_bohr = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    traj = run_md(
        provider,
        coords_bohr,
        atomic_numbers=[int(a) for a in atomic_numbers],
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        temperature_K=temperature_K,
        thermostat=thermostat,
        thermostat_tau_fs=thermostat_tau_fs,
        velocities=velocities,
        remove_com=remove_com,
        seed=seed,
        record_stride=record_stride,
    )
    if output is not None:
        _emit_msindo_md_citations(output, thermostat=traj.thermostat)
    return traj


def _emit_msindo_metadynamics_citations(output, *, thermostat=None):
    """Best-effort ``.bibtex`` / ``.references`` for an INDO well-tempered
    metadynamics run: MSINDO + velocity-Verlet (Swope 1982) + the thermostat
    + metadynamics (Laio-Parrinello 2002) + well-tempered (Barducci-Bussi-
    Parrinello 2008).  Non-fatal (mirrors ``_emit_msindo_md_citations``)."""
    md_thermostat = None if (thermostat in (None, "nve")) else thermostat
    try:
        from vibeqc.output.citations import emit_citations

        emit_citations(
            output,
            method="msindo",
            uses_integrals=False,
            uses_md=True,
            md_thermostat=md_thermostat,
            uses_metadynamics=True,
            well_tempered=True,
        )
    except Exception:
        pass


def run_msindo_metadynamics(
    atomic_numbers,
    coords_angstrom,
    collective_variables,
    *,
    charge=0,
    multiplicity=1,
    nddo=False,
    conv_tol=1e-9,
    grad_step=1e-3,
    max_iter=200,
    bias_factor=10.0,
    hill_height=1e-3,
    hill_sigma=0.1,
    deposition_stride=20,
    timestep_fs=0.5,
    n_steps=1000,
    temperature_K=300.0,
    thermostat="nose_hoover",
    thermostat_tau_fs=50.0,
    velocities=None,
    remove_com=True,
    seed=None,
    record_stride=1,
    output=None,
):
    """Well-tempered metadynamics on the MSINDO surface.

    The MSINDO-aware wrapper over :func:`vibeqc.metadynamics.run_metadynamics`:
    it builds the MSINDO energy+gradient provider (:func:`msindo_force_provider`)
    and deposits a Gaussian bias on ``collective_variables`` (a list of
    :class:`vibeqc.metadynamics.CollectiveVariable` -- e.g. ``DistanceCV`` /
    ``AngleCV``, which act on the bohr coordinates the driver carries).

    The *initial* geometry is in Ångström; the returned
    :class:`vibeqc.metadynamics.MetadynamicsResult` carries the (biased) MD
    trajectory, the CV trajectory, and the accumulated bias -- reconstruct the
    free energy with ``result.free_energy(grid)``.  Metadynamics runs NVT, so
    ``thermostat`` defaults to Nosé-Hoover.

    Each MD step costs ``6N + 1`` MSINDO SCFs (the FD gradient); metadynamics
    needs many steps to fill a basin, so this is for tiny systems / short
    demonstrations.

    ``output`` (optional path stem) writes the ``.bibtex`` / ``.references``
    citing MSINDO + velocity-Verlet + the thermostat + metadynamics
    (Laio-Parrinello 2002) + well-tempered (Barducci-Bussi-Parrinello 2008).
    """
    from vibeqc.metadynamics import run_metadynamics

    provider = msindo_force_provider(
        atomic_numbers,
        charge=charge,
        multiplicity=multiplicity,
        nddo=nddo,
        conv_tol=conv_tol,
        grad_step=grad_step,
        max_iter=max_iter,
    )
    coords_bohr = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    res = run_metadynamics(
        provider,
        coords_bohr,
        collective_variables,
        atomic_numbers=[int(a) for a in atomic_numbers],
        bias_factor=bias_factor,
        hill_height=hill_height,
        hill_sigma=hill_sigma,
        deposition_stride=deposition_stride,
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        temperature_K=temperature_K,
        thermostat=thermostat,
        thermostat_tau_fs=thermostat_tau_fs,
        velocities=velocities,
        remove_com=remove_com,
        seed=seed,
        record_stride=record_stride,
    )
    if output is not None:
        _emit_msindo_metadynamics_citations(
            output, thermostat=res.trajectory.thermostat
        )
    return res
