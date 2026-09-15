"""MSINDO per-pair integral derivatives -- local frame R-derivatives (Phase 2).

Ports the GRAD=.TRUE. branch of INTDRV (intdrv.f): for each atom pair (K,L),
computes all two-center quantities in the bond-aligned local frame AND their
derivatives w.r.t. interatomic distance R.

Fortran reference: intdrv.f, coulom.f, v2core.f, vcorrk.f, vcorrl.f,
                   overlap.f, lmunu.f, horth.f, deltah.f.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import msindo_integrals as _ki
from . import msindo_integrals_deriv as _kd
from .msindo import (
    AL,
    FCP1S,
    FCP2P,
    FCP2S,
    FCP3D,
    FCP3P,
    FCP3S,
    KDD,
    KDP,
    KDS,
    KPP,
    KPS,
    KSS,
    LS,
    MP,
    MUD,
    MUP,
    MUS,
    ND,
    TAU1S,
    TAU2P,
    TAU2S,
    TAU3D,
    TAU3P,
    TAU3S,
    _CoreH,
    _harmtr,
    _pair_blocks,
    _PairBlocks,
    _vfak,
    eff_core_charge,
    eneg,
    n_basis,
    n_d_principal,
    n_p_principal,
    n_principal,
)


@dataclass
class _PairBlocksDeriv:
    """Per-pair integrals + local-frame R-derivatives + global-frame (R,th,phi) derivatives."""

    pb: _PairBlocks
    # local-frame matrices (bond-aligned, orbital order:
    #   0=s, 1=ps, 2=ppi, 3=ppi, 4=ds, 5=dpi, 6=dpi, 7=dd, 8=dd)
    S_local: np.ndarray  # SKL(9,9)   -- overlap
    dS_local: np.ndarray  # DSKL(9,9)  -- dS/dR (OVERLAP.f GRAD)
    gamma_local: np.ndarray  # GAMM(9,9)  -- Coulomb g
    d_gamma_local: np.ndarray  # DGAMM(9,9) -- dg/dR (COULOM.f GRAD)
    M_local: np.ndarray  # MULL(9,9)  -- Löwdin Lmuν
    dM_local: np.ndarray  # DMULL(9,9) -- dM/dR (LMUNU.f GRAD)
    # local-frame diagonal H blocks (only diagonal elements non-zero)
    dHK1_local: np.ndarray  # 9x9 diag dH/dR for K
    dHL1_local: np.ndarray  # 9x9 diag dH/dR for L
    # local-frame off-diagonal core (resonance) block
    dHKL2_local: np.ndarray  # 9x9 dH_core/dR
    # --- Phase 3: global-frame (R, th, phi) derivatives (ROTINT GRAD) ---
    HK1_DR: np.ndarray = None  # 9x9 dH_global/dR for K
    HL1_DR: np.ndarray = None  # 9x9 dH_global/dR for L
    HKL2_DR: np.ndarray = None  # 9x9 dCore_global/dR
    HK1_DT: np.ndarray = None  # 9x9 dH_global/dth for K
    HL1_DT: np.ndarray = None  # 9x9 dH_global/dth for L
    HKL2_DT: np.ndarray = None  # 9x9 dCore_global/dth
    HK1_DP: np.ndarray = None  # 9x9 dH_global/dphi for K
    HL1_DP: np.ndarray = None  # 9x9 dH_global/dphi for L
    HKL2_DP: np.ndarray = None  # 9x9 dCore_global/dphi
    # bond geometry
    E: np.ndarray = None  # unit vector K->L
    R: float = None  # interatomic distance


# --------------------------------------------------------------------------- #
# d/dR helpers -- derivatives of elementary functions                         #
# --------------------------------------------------------------------------- #


def _dcor_dR(za, zb, R):
    """d/dR of dcor(z_a,z_b) = -1/2 . (z_a^2+z_b^2) / (1 + 1/2(z_a+z_b)R)."""
    num = -0.5 * (za * za + zb * zb)
    den = 1.0 + 0.5 * (za + zb) * R
    return -num * 0.5 * (za + zb) / (den * den)


# --------------------------------------------------------------------------- #
# Frozen-core overlap derivatives (valrum.f GRAD branch)                     #
# --------------------------------------------------------------------------- #


def _rumpf_overlaps_deriv(z_val: int, z_core: int, R: float) -> dict:
    """d/dR of each valence-core overlap (keys match _rumpf_overlaps)."""
    dri: dict[str, float] = {}
    if z_core <= 2:
        return dri
    ns = n_principal(z_val)
    npr = n_p_principal(z_val)
    nd = n_d_principal(z_val)
    zs, zp, zd = MUS[z_val], MUP[z_val], MUD.get(z_val, 0.0)
    kp = n_basis(z_val) >= 4
    kd = n_basis(z_val) >= 9 and zd != 0.0

    def _ds(zv, nv, lv, mv, zc, nc, lc, mc):
        return _ki.ds2int(nv, lv, mv, zv, nc, lc, mc, zc, R)

    z1s = TAU1S[z_core]
    dri["S1S"] = _ds(zs, ns, 0, 0, z1s, 1, 0, 0)
    if kp:
        dri["PS1S"] = _ds(zp, npr, 1, 0, z1s, 1, 0, 0)
    if kd:
        dri["DS1S"] = _ds(zd, nd, 2, 0, z1s, 1, 0, 0)
    if z_core > 10:
        z2s, z2p = TAU2S[z_core], TAU2P[z_core]
        dri["S2S"] = _ds(zs, ns, 0, 0, z2s, 2, 0, 0)
        dri["S2PS"] = _ds(zs, ns, 0, 0, z2p, 2, 1, 0)
        if kp:
            dri["PS2S"] = _ds(zp, npr, 1, 0, z2s, 2, 0, 0)
            dri["PS2PS"] = _ds(zp, npr, 1, 0, z2p, 2, 1, 0)
            dri["PP2PP"] = _ds(zp, npr, 1, 1, z2p, 2, 1, 1)
        if kd:
            dri["DS2S"] = _ds(zd, nd, 2, 0, z2s, 2, 0, 0)
            dri["DS2PS"] = _ds(zd, nd, 2, 0, z2p, 2, 1, 0)
            dri["DP2PP"] = _ds(zd, nd, 2, 1, z2p, 2, 1, 1)
    if z_core > 18:
        z3s, z3p = TAU3S[z_core], TAU3P[z_core]
        dri["S3S"] = _ds(zs, ns, 0, 0, z3s, 3, 0, 0)
        dri["S3PS"] = _ds(zs, ns, 0, 0, z3p, 3, 1, 0)
        if kp:
            dri["PS3S"] = _ds(zp, npr, 1, 0, z3s, 3, 0, 0)
            dri["PS3PS"] = _ds(zp, npr, 1, 0, z3p, 3, 1, 0)
            dri["PP3PP"] = _ds(zp, npr, 1, 1, z3p, 3, 1, 1)
        if kd:
            dri["DS3S"] = _ds(zd, nd, 2, 0, z3s, 3, 0, 0)
            dri["DS3PS"] = _ds(zd, nd, 2, 0, z3p, 3, 1, 0)
            dri["DP3PP"] = _ds(zd, nd, 2, 1, z3p, 3, 1, 1)
    if z_core > 30:
        z3d = TAU3D[z_core]
        dri["S3DS"] = _ds(zs, ns, 0, 0, z3d, 3, 2, 0)
        if kp:
            dri["PS3DS"] = _ds(zp, npr, 1, 0, z3d, 3, 2, 0)
            dri["PP3DP"] = _ds(zp, npr, 1, 1, z3d, 3, 2, 1)
        if kd:
            dri["DS3DS"] = _ds(zd, nd, 2, 0, z3d, 3, 2, 0)
            dri["DP3DP"] = _ds(zd, nd, 2, 1, z3d, 3, 2, 1)
            dri["DD3DD"] = _ds(zd, nd, 2, 2, z3d, 3, 2, 2)
    return dri


# --------------------------------------------------------------------------- #
# V2CORE + VCORR derivative                                                  #
# --------------------------------------------------------------------------- #


def _v2core_one_side_deriv(
    z_val: int, z_core: int, R: float, gam: dict, d_gam: dict
) -> _CoreH:
    """d/dR of V2CORE nuclear attraction + frozen-core pseudopotential + VCORR.

    Returns _CoreH with DHSS/DHPS/DHPP/DHDS/DHDP/DHDD.
    """
    zc = float(eff_core_charge(z_core))
    ns = n_principal(z_val)
    npr = n_p_principal(z_val)
    nd = n_d_principal(z_val)
    zs, zp, zd = MUS[z_val], MUP[z_val], MUD.get(z_val, 0.0)
    kp = n_basis(z_val) >= 4
    kd = n_basis(z_val) >= 9 and zd != 0.0
    dh = _CoreH()

    # Frozen-core overlap derivatives
    from .msindo import _rumpf_overlaps as _ro

    ri = _ro(z_val, z_core, R)
    dri = _rumpf_overlaps_deriv(z_val, z_core, R)

    def _dfcp(key1s, key2s, key2ps, key3s, key3ps, key3d=None):
        s = 0.0
        if z_core > 2:
            s += 2.0 * FCP1S[z_core] * dri.get(key1s, 0.0) * ri.get(key1s, 0.0)
        if z_core > 10:
            s += 2.0 * FCP2S[z_core] * dri.get(key2s, 0.0) * ri.get(key2s, 0.0)
            s += 2.0 * FCP2P[z_core] * dri.get(key2ps, 0.0) * ri.get(key2ps, 0.0)
        if z_core > 18:
            s += 2.0 * FCP3S[z_core] * dri.get(key3s, 0.0) * ri.get(key3s, 0.0)
            s += 2.0 * FCP3P[z_core] * dri.get(key3ps, 0.0) * ri.get(key3ps, 0.0)
        if z_core > 30 and key3d is not None:
            s += 2.0 * FCP3D[z_core] * dri.get(key3d, 0.0) * ri.get(key3d, 0.0)
        return s

    dh.HSS = -zc * _kd.dv2int(ns, 0, 0, zs, R) + _dfcp(
        "S1S", "S2S", "S2PS", "S3S", "S3PS", "S3DS"
    )
    if kp:
        dh.HPS = -zc * _kd.dv2int(npr, 1, 0, zp, R)
        dh.HPP = -zc * _kd.dv2int(npr, 1, 1, zp, R)
        if z_core > 2:
            dh.HPS += 2.0 * FCP1S[z_core] * dri.get("PS1S", 0.0) * ri.get("PS1S", 0.0)
        if z_core > 10:
            dh.HPS += 2.0 * FCP2S[z_core] * dri.get("PS2S", 0.0) * ri.get("PS2S", 0.0)
            dh.HPS += 2.0 * FCP2P[z_core] * dri.get("PS2PS", 0.0) * ri.get("PS2PS", 0.0)
            dh.HPP += 2.0 * FCP2P[z_core] * dri.get("PP2PP", 0.0) * ri.get("PP2PP", 0.0)
        if z_core > 18:
            dh.HPS += 2.0 * FCP3S[z_core] * dri.get("PS3S", 0.0) * ri.get("PS3S", 0.0)
            dh.HPS += 2.0 * FCP3P[z_core] * dri.get("PS3PS", 0.0) * ri.get("PS3PS", 0.0)
            dh.HPP += 2.0 * FCP3P[z_core] * dri.get("PP3PP", 0.0) * ri.get("PP3PP", 0.0)
        if z_core > 30:
            dh.HPS += 2.0 * FCP3D[z_core] * dri.get("PS3DS", 0.0) * ri.get("PS3DS", 0.0)
            dh.HPP += 2.0 * FCP3D[z_core] * dri.get("PP3DP", 0.0) * ri.get("PP3DP", 0.0)
    if kd:
        dh.HDS = -zc * _kd.dv2int(nd, 2, 0, zd, R)
        dh.HDP = -zc * _kd.dv2int(nd, 2, 1, zd, R)
        dh.HDD = -zc * _kd.dv2int(nd, 2, 2, zd, R)
        if z_core > 2:
            dh.HDS += 2.0 * FCP1S[z_core] * dri.get("DS1S", 0.0) * ri.get("DS1S", 0.0)
        if z_core > 10:
            dh.HDS += 2.0 * FCP2S[z_core] * dri.get("DS2S", 0.0) * ri.get("DS2S", 0.0)
            dh.HDS += 2.0 * FCP2P[z_core] * dri.get("DS2PS", 0.0) * ri.get("DS2PS", 0.0)
            dh.HDP += 2.0 * FCP2P[z_core] * dri.get("DP2PP", 0.0) * ri.get("DP2PP", 0.0)
        if z_core > 18:
            dh.HDS += 2.0 * FCP3S[z_core] * dri.get("DS3S", 0.0) * ri.get("DS3S", 0.0)
            dh.HDS += 2.0 * FCP3P[z_core] * dri.get("DS3PS", 0.0) * ri.get("DS3PS", 0.0)
            dh.HDP += 2.0 * FCP3P[z_core] * dri.get("DP3PP", 0.0) * ri.get("DP3PP", 0.0)
        if z_core > 30:
            dh.HDS += 2.0 * FCP3D[z_core] * dri.get("DS3DS", 0.0) * ri.get("DS3DS", 0.0)
            dh.HDP += 2.0 * FCP3D[z_core] * dri.get("DP3DP", 0.0) * ri.get("DP3DP", 0.0)
            dh.HDD += 2.0 * FCP3D[z_core] * dri.get("DD3DD", 0.0) * ri.get("DD3DD", 0.0)

    # VCORR penetration derivative (vcorrk.f/VCORRL GRAD branch)
    _vcork_deriv(z_val, z_core, R, gam, d_gam, dh, npr, nd, zp, zd, kp, kd)
    return dh


def _vcork_deriv(z_val, z_core, R, gam, d_gam, dh, npr, nd, zp, zd, kp, kd):
    """Add VCORR penetration derivatives to dh in-place (vcorrk.f GRAD)."""
    npc = n_principal(z_core)
    npcp = n_p_principal(z_core)
    ncd = n_d_principal(z_core)
    zs_c = MUS[z_core]
    zp_c = MUP[z_core]
    zd_c = MUD.get(z_core, 0.0)

    # p-orbital corrections
    # VCORR uses (s_core | p_valence) Coulomb integrals -- m=0 for ps, m=1 for ppi
    if kp:
        dc_ps_s = _kd.dc2int(npc, 0, 0, zs_c, npr, 1, 0, zp, R)  # (s_L|ps_K)
        dc_ps_p = _kd.dc2int(npc, 0, 0, zs_c, npr, 1, 1, zp, R)  # (s_L|ppi_K)
        dh.HPS += (dc_ps_s - d_gam["ps"]) * LS[z_core]
        dh.HPP += (dc_ps_p - d_gam["ps"]) * LS[z_core]
        if MP[z_core] != 0:
            dc_pp_s = _kd.dc2int(npcp, 0, 0, zp_c, npr, 1, 0, zp, R)
            dc_pp_p = _kd.dc2int(npcp, 0, 0, zp_c, npr, 1, 1, zp, R)
            dh.HPS += (dc_pp_s - d_gam["pp"]) * MP[z_core]
            dh.HPP += (dc_pp_p - d_gam["pp"]) * MP[z_core]
        if ND[z_core] != 0:
            dc_pd_s = _kd.dc2int(ncd, 0, 0, zd_c, npr, 1, 0, zp, R)
            dc_pd_p = _kd.dc2int(ncd, 0, 0, zd_c, npr, 1, 1, zp, R)
            dh.HPS += (dc_pd_s - d_gam["pd"]) * ND[z_core]
            dh.HPP += (dc_pd_p - d_gam["pd"]) * ND[z_core]
    # d-orbital corrections
    if kd:
        dc_ds_s = _kd.dc2int(npc, 0, 0, zs_c, nd, 2, 0, zd, R)  # (s|ds)
        dc_ds_p = _kd.dc2int(npc, 0, 0, zs_c, nd, 2, 1, zd, R)  # (s|dpi)
        dc_ds_d = _kd.dc2int(npc, 0, 0, zs_c, nd, 2, 2, zd, R)  # (s|dd)
        dh.HDS += (dc_ds_s - d_gam["ds"]) * LS[z_core]
        dh.HDP += (dc_ds_p - d_gam["ds"]) * LS[z_core]
        dh.HDD += (dc_ds_d - d_gam["ds"]) * LS[z_core]
        if MP[z_core] != 0:
            dc_dp_s = _kd.dc2int(npcp, 0, 0, zp_c, nd, 2, 0, zd, R)
            dc_dp_p = _kd.dc2int(npcp, 0, 0, zp_c, nd, 2, 1, zd, R)
            dc_dp_d = _kd.dc2int(npcp, 0, 0, zp_c, nd, 2, 2, zd, R)
            dh.HDS += (dc_dp_s - d_gam["dp"]) * MP[z_core]
            dh.HDP += (dc_dp_p - d_gam["dp"]) * MP[z_core]
            dh.HDD += (dc_dp_d - d_gam["dp"]) * MP[z_core]
        if ND[z_core] != 0:
            dc_dd_s = _kd.dc2int(ncd, 0, 0, zd_c, nd, 2, 0, zd, R)
            dc_dd_p = _kd.dc2int(ncd, 0, 0, zd_c, nd, 2, 1, zd, R)
            dc_dd_d = _kd.dc2int(ncd, 0, 0, zd_c, nd, 2, 2, zd, R)
            dh.HDS += (dc_dd_s - d_gam["dd"]) * ND[z_core]
            dh.HDP += (dc_dd_p - d_gam["dd"]) * ND[z_core]
            dh.HDD += (dc_dd_d - d_gam["dd"]) * ND[z_core]


# --------------------------------------------------------------------------- #
# LMUNU derivative (lmunu.f GRAD branch)                                     #
# --------------------------------------------------------------------------- #


def _lmunu_local_deriv(zk, zl, R, S, dS, dM):
    """Fill dM[9x9] in-place with d/dR of LMUNU."""
    zsK, zpK, zdK = MUS[zk], MUP[zk], MUD.get(zk, 0.0)
    zsL, zpL, zdL = MUS[zl], MUP[zl], MUD.get(zl, 0.0)
    kp, lp = n_basis(zk) >= 4, n_basis(zl) >= 4
    kd, ld = n_basis(zk) >= 9, n_basis(zl) >= 9

    def _dm_1ms(za, zb, s, ds):
        """d/dR [dcor*s*(1 - s)]."""
        dc = -0.5 * (za * za + zb * zb) / (1.0 + 0.5 * (za + zb) * R)
        ddc = _dcor_dR(za, zb, R)
        return ddc * s * (1.0 - s) + dc * ds * (1.0 - 2.0 * s)

    def _dm_1ps(za, zb, s, ds):
        """d/dR [dcor*s*(1 + s)]."""
        dc = -0.5 * (za * za + zb * zb) / (1.0 + 0.5 * (za + zb) * R)
        ddc = _dcor_dR(za, zb, R)
        return ddc * s * (1.0 + s) + dc * ds * (1.0 + 2.0 * s)

    def _dm_abs(za, zb, s, ds):
        """d/dR [dcor*s*(1 - |s|)]."""
        dc = -0.5 * (za * za + zb * zb) / (1.0 + 0.5 * (za + zb) * R)
        ddc = _dcor_dR(za, zb, R)
        as_ = abs(s)
        sgn = 1.0 if s >= 0 else -1.0
        return ddc * s * (1.0 - as_) + dc * ds * (1.0 - as_) - dc * s * sgn * ds

    # s/s
    dM[0, 0] = _dm_1ms(zsK, zsL, S["ss"], dS["ss"])
    # p/s
    if kp and zk > 2:
        dM[1, 0] = _dm_1ms(zpK, zsL, S["ps"], dS["ps"])
    # s/p
    if lp and zl > 2:
        dM[0, 1] = _dm_1ps(zsK, zpL, S["sp"], dS["sp"])
    # p/p s, pi
    if kp and lp and zk > 2 and zl > 2:
        dM[1, 1] = _dm_abs(zpK, zpL, S["pp"], dS["pp"])
        dM[2, 2] = _dm_abs(zpK, zpL, S["pipi"], dS["pipi"])
        dM[3, 3] = dM[2, 2]
    # d/s
    if kd:
        dM[4, 0] = _dm_1ms(zdK, zsL, S["ds"], dS["ds"])
        if lp:
            dM[4, 1] = _dm_abs(zdK, zpL, S["dp"], dS["dp"])
            dM[5, 2] = _dm_abs(zdK, zpL, S["dppi"], dS["dppi"])
            dM[6, 3] = dM[5, 2]
    # s/d
    if ld:
        dM[0, 4] = _dm_1ms(zsK, zdL, S["sd"], dS["sd"])
        if kp:
            dM[1, 4] = _dm_abs(zpK, zdL, S["pd"], dS["pd"])
            dM[2, 5] = _dm_abs(zpK, zdL, S["pdpi"], dS["pdpi"])
            dM[3, 6] = dM[2, 5]
    # d/d s, pi, d
    if kd and ld:
        dM[4, 4] = _dm_abs(zdK, zdL, S["dd"], dS["dd"])
        dM[5, 5] = _dm_abs(zdK, zdL, S["ddpi"], dS["ddpi"])
        dM[6, 6] = dM[5, 5]
        dM[7, 7] = _dm_abs(zdK, zdL, S["dddel"], dS["dddel"])
        dM[8, 8] = dM[7, 7]

    # 1s averaging (lmunu.f tail)
    if zk > 2 and zl > 2:
        return

    def _d_avg(za, zb, s, ds):
        dc = 0.5 * (za + zb) * R
        ddc = 0.5 * (za + zb)
        e = (1.0 - math.exp(-dc)) / (1.0 + dc)
        de = (ddc * math.exp(-dc) * (1.0 + dc) - (1.0 - math.exp(-dc)) * ddc) / (
            (1.0 + dc) ** 2
        )
        return -ds * e - s * de

    if zk == 1 and zl > 2:
        dM[0, 0] = (dM[0, 0] + _d_avg(zsK, zsL, S["ss"], dS["ss"])) / 2.0
        if lp:
            dM[0, 1] = (dM[0, 1] + _d_avg(zsK, zpL, S["sp"], dS["sp"])) / 2.0
        if ld:
            dM[0, 4] = (dM[0, 4] + _d_avg(zsK, zdL, S["sd"], dS["sd"])) / 2.0
    elif zk > 2 and zl == 1:
        dM[0, 0] = (dM[0, 0] + _d_avg(zsK, zsL, S["ss"], dS["ss"])) / 2.0
        if kp:
            dM[1, 0] = (dM[1, 0] + _d_avg(zpK, zsL, S["ps"], dS["ps"])) / 2.0
        if kd:
            dM[4, 0] = (dM[4, 0] + _d_avg(zdK, zsL, S["ds"], dS["ds"])) / 2.0
    else:
        dM[0, 0] = (dM[0, 0] + _d_avg(zsK, zsL, S["ss"], dS["ss"])) / 2.0


# --------------------------------------------------------------------------- #
# Fill helpers -- 9x9 local-frame matrices from dict keys                      #
# --------------------------------------------------------------------------- #


def _fill_S_local(Smat, S, kp, lp, kd, ld):
    """Fill 9x9 SKL from dict (matching OVERLAP.f indexing)."""
    Smat[0, 0] = S["ss"]
    if kp:
        Smat[1, 0] = S["ps"]
    if kd:
        Smat[4, 0] = S["ds"]
    if lp:
        Smat[0, 1] = S["sp"]
        if kp:
            Smat[1, 1] = S["pp"]
            Smat[2, 2] = S["pipi"]
            Smat[3, 3] = S["pipi"]
        if kd:
            Smat[4, 1] = S["dp"]
            Smat[5, 2] = S["dppi"]
            Smat[6, 3] = S["dppi"]
    if ld:
        Smat[0, 4] = S["sd"]
        if kp:
            Smat[1, 4] = S["pd"]
            Smat[2, 5] = S["pdpi"]
            Smat[3, 6] = S["pdpi"]
        if kd:
            Smat[4, 4] = S["dd"]
            Smat[5, 5] = S["ddpi"]
            Smat[6, 6] = S["ddpi"]
            Smat[7, 7] = S["dddel"]
            Smat[8, 8] = S["dddel"]


def _fill_M_local(Mmat, M, kp, lp, kd, ld):
    """Fill 9x9 MULL from dict (matching LMUNU.f indexing)."""
    Mmat[0, 0] = M["ss"]
    if kp:
        Mmat[1, 0] = M["ps"]
    if kd:
        Mmat[4, 0] = M["ds"]
    if lp:
        Mmat[0, 1] = M["sp"]
        if kp:
            Mmat[1, 1] = M["pp"]
            Mmat[2, 2] = M["pipi"]
            Mmat[3, 3] = M["pipi"]
        if kd:
            Mmat[4, 1] = M["dp"]
            Mmat[5, 2] = M["dppi"]
            Mmat[6, 3] = M["dppi"]
    if ld:
        Mmat[0, 4] = M["sd"]
        if kp:
            Mmat[1, 4] = M["pd"]
            Mmat[2, 5] = M["pdpi"]
            Mmat[3, 6] = M["pdpi"]
        if kd:
            Mmat[4, 4] = M["dd"]
            Mmat[5, 5] = M["ddpi"]
            Mmat[6, 6] = M["ddpi"]
            Mmat[7, 7] = M["dddel"]
            Mmat[8, 8] = M["dddel"]


def _fill_gamma_local(Gmat, gam_k, gam_l, kp, lp, kd, ld):
    """Fill 9x9 GAMM from gamma dicts (matching COULOM.f indexing)."""
    # Gammas are indexed by (shell_on_K, shell_on_L) in local frame.
    # COULOM.f fills the full 9x9: ss at (1,1), ps at (2:4,1), etc.
    # gam_k["ps"] = g(p_K, s_L), gam_k["ds"] = g(d_K, s_L)
    # gam_k["pp"] = g(p_K, p_L), etc.
    # gam_l has the mirrored versions: g(s_K, p_L), etc.

    # (ss|ss) = g(s_K, s_L) -- not in gam_k/gam_l dicts directly!
    # We need it; compute from gam_l["ps"] or gam_k["ps"] or from c2int directly.
    # Actually, gam_l["ps"] = g(p_L, s_K) = g(s_K, p_L) -- also not g(s_K, s_L).
    # The ss gamma isn't stored in either gam dict (they're all p/d with s/sp).
    # So we can't fill it from the dicts. But for Phase 2 validation we only need
    # the gamma values that are used in what we compute. Let's fill what we can.

    # Column 1: (something | ss), i.e., shell_on_K x s_L
    # gam_k["ps"] = g(p_K, s_L) -- fills rows 2,3,4 of col 0 (Python 0-based)
    if kp:
        Gmat[1, 0] = gam_k["ps"]
        Gmat[2, 0] = gam_k["ps"]
        Gmat[3, 0] = gam_k["ps"]
    if kd:
        Gmat[4, 0] = gam_k["ds"]
        Gmat[5, 0] = gam_k["ds"]
        Gmat[6, 0] = gam_k["ds"]
        Gmat[7, 0] = gam_k["ds"]
        Gmat[8, 0] = gam_k["ds"]

    # Column 2-4: (something | p_L)
    if lp:
        # Row 0: (s_K | p_L) = gam_k["ps"] mirrored
        # This is gam_l["ps"] = g(p_L, s_K) = g(s_K, p_L)
        Gmat[0, 1] = gam_l["ps"]
        Gmat[0, 2] = gam_l["ps"]
        Gmat[0, 3] = gam_l["ps"]
        if kp:
            Gmat[1, 1] = gam_k["pp"]
            Gmat[2, 1] = gam_k["pp"]
            Gmat[3, 1] = gam_k["pp"]
            Gmat[1, 2] = gam_k["pp"]
            Gmat[2, 2] = gam_k["pp"]
            Gmat[3, 2] = gam_k["pp"]
            Gmat[1, 3] = gam_k["pp"]
            Gmat[2, 3] = gam_k["pp"]
            Gmat[3, 3] = gam_k["pp"]
        if kd:
            Gmat[4, 1] = gam_k["dp"]
            Gmat[5, 1] = gam_k["dp"]
            Gmat[6, 1] = gam_k["dp"]
            Gmat[7, 1] = gam_k["dp"]
            Gmat[8, 1] = gam_k["dp"]
            Gmat[4, 2] = gam_k["dp"]
            Gmat[5, 2] = gam_k["dp"]
            Gmat[6, 2] = gam_k["dp"]
            Gmat[7, 2] = gam_k["dp"]
            Gmat[8, 2] = gam_k["dp"]
            Gmat[4, 3] = gam_k["dp"]
            Gmat[5, 3] = gam_k["dp"]
            Gmat[6, 3] = gam_k["dp"]
            Gmat[7, 3] = gam_k["dp"]
            Gmat[8, 3] = gam_k["dp"]

    # Column 5-9: (something | d_L)
    if ld:
        Gmat[0, 4] = gam_l["ds"]
        Gmat[0, 5] = gam_l["ds"]
        Gmat[0, 6] = gam_l["ds"]
        Gmat[0, 7] = gam_l["ds"]
        Gmat[0, 8] = gam_l["ds"]
        if kp:
            Gmat[1, 4] = gam_k["pd"]
            Gmat[2, 4] = gam_k["pd"]
            Gmat[3, 4] = gam_k["pd"]
            Gmat[1, 5] = gam_k["pd"]
            Gmat[2, 5] = gam_k["pd"]
            Gmat[3, 5] = gam_k["pd"]
            Gmat[1, 6] = gam_k["pd"]
            Gmat[2, 6] = gam_k["pd"]
            Gmat[3, 6] = gam_k["pd"]
            Gmat[1, 7] = gam_k["pd"]
            Gmat[2, 7] = gam_k["pd"]
            Gmat[3, 7] = gam_k["pd"]
            Gmat[1, 8] = gam_k["pd"]
            Gmat[2, 8] = gam_k["pd"]
            Gmat[3, 8] = gam_k["pd"]
        if kd:
            Gmat[4, 4] = gam_k["dd"]
            Gmat[5, 4] = gam_k["dd"]
            Gmat[6, 4] = gam_k["dd"]
            Gmat[7, 4] = gam_k["dd"]
            Gmat[8, 4] = gam_k["dd"]
            Gmat[4, 5] = gam_k["dd"]
            Gmat[5, 5] = gam_k["dd"]
            Gmat[6, 5] = gam_k["dd"]
            Gmat[7, 5] = gam_k["dd"]
            Gmat[8, 5] = gam_k["dd"]
            Gmat[4, 6] = gam_k["dd"]
            Gmat[5, 6] = gam_k["dd"]
            Gmat[6, 6] = gam_k["dd"]
            Gmat[7, 6] = gam_k["dd"]
            Gmat[8, 6] = gam_k["dd"]
            Gmat[4, 7] = gam_k["dd"]
            Gmat[5, 7] = gam_k["dd"]
            Gmat[6, 7] = gam_k["dd"]
            Gmat[7, 7] = gam_k["dd"]
            Gmat[8, 7] = gam_k["dd"]
            Gmat[4, 8] = gam_k["dd"]
            Gmat[5, 8] = gam_k["dd"]
            Gmat[6, 8] = gam_k["dd"]
            Gmat[7, 8] = gam_k["dd"]
            Gmat[8, 8] = gam_k["dd"]


# --------------------------------------------------------------------------- #
# HORTH derivative                                                            #
# --------------------------------------------------------------------------- #


def _horth_deriv(hk, hl, dhk, dhl, vk, vl, S, dS, M, dM, kp, lp, kd, ld):
    """Apply HORTH correction and its derivative (horth.f GRAD)."""

    def _apply(key, sk, sl, ak, al):
        s, ds = S[key], dS[key]
        m, dm = M[sk, sl], dM[sk, sl]
        setattr(hk, ak, getattr(hk, ak) - vk * s * m)
        setattr(hl, al, getattr(hl, al) - vl * s * m)
        setattr(dhk, ak, getattr(dhk, ak) - vk * (ds * m + s * dm))
        setattr(dhl, al, getattr(dhl, al) - vl * (ds * m + s * dm))

    _apply("ss", 0, 0, "HSS", "HSS")
    if lp:
        _apply("sp", 0, 1, "HSS", "HPS")
    if kp:
        _apply("ps", 1, 0, "HPS", "HSS")
    if kp and lp:
        _apply("pp", 1, 1, "HPS", "HPS")
        s, ds = S["pipi"], dS["pipi"]
        m, dm = M[2, 2], dM[2, 2]
        setattr(hk, "HPP", getattr(hk, "HPP") - vk * s * m)
        setattr(hl, "HPP", getattr(hl, "HPP") - vl * s * m)
        setattr(dhk, "HPP", getattr(dhk, "HPP") - vk * (ds * m + s * dm))
        setattr(dhl, "HPP", getattr(dhl, "HPP") - vl * (ds * m + s * dm))
    if ld:
        _apply("sd", 0, 4, "HSS", "HDS")
    if kd:
        _apply("ds", 4, 0, "HDS", "HSS")
    if kp and ld:
        _apply("pd", 1, 4, "HPS", "HDS")
        s, ds = S["pdpi"], dS["pdpi"]
        m, dm = M[2, 5], dM[2, 5]
        setattr(hk, "HPP", getattr(hk, "HPP") - vk * s * m)
        setattr(hl, "HDP", getattr(hl, "HDP") - vl * s * m)
        setattr(dhk, "HPP", getattr(dhk, "HPP") - vk * (ds * m + s * dm))
        setattr(dhl, "HDP", getattr(dhl, "HDP") - vl * (ds * m + s * dm))
    if kd and lp:
        _apply("dp", 4, 1, "HDS", "HPS")
        s, ds = S["dppi"], dS["dppi"]
        m, dm = M[5, 2], dM[5, 2]
        setattr(hk, "HDP", getattr(hk, "HDP") - vk * s * m)
        setattr(hl, "HPP", getattr(hl, "HPP") - vl * s * m)
        setattr(dhk, "HDP", getattr(dhk, "HDP") - vk * (ds * m + s * dm))
        setattr(dhl, "HPP", getattr(dhl, "HPP") - vl * (ds * m + s * dm))
    if kd and ld:
        _apply("dd", 4, 4, "HDS", "HDS")
        s, ds = S["ddpi"], dS["ddpi"]
        m, dm = M[5, 5], dM[5, 5]
        setattr(hk, "HDP", getattr(hk, "HDP") - vk * s * m)
        setattr(hl, "HDP", getattr(hl, "HDP") - vl * s * m)
        setattr(dhk, "HDP", getattr(dhk, "HDP") - vk * (ds * m + s * dm))
        setattr(dhl, "HDP", getattr(dhl, "HDP") - vl * (ds * m + s * dm))
        s, ds = S["dddel"], dS["dddel"]
        m, dm = M[7, 7], dM[7, 7]
        setattr(hk, "HDD", getattr(hk, "HDD") - vk * s * m)
        setattr(hl, "HDD", getattr(hl, "HDD") - vl * s * m)
        setattr(dhk, "HDD", getattr(dhk, "HDD") - vk * (ds * m + s * dm))
        setattr(dhl, "HDD", getattr(dhl, "HDD") - vl * (ds * m + s * dm))


# --------------------------------------------------------------------------- #
# DELTAH derivative                                                           #
# --------------------------------------------------------------------------- #


def _deltah_deriv(
    zk, zl, R, core, dcore, S, dS, M, dM, shk, shl, dshk, dshl, kp, lp, kd, ld
):
    """Compute DELTAH resonance and its R-derivative (deltah.f GRAD)."""
    fack = 1.0 - math.exp(-AL(zk, zl) * R)
    facl = 1.0 - math.exp(-AL(zl, zk) * R)
    dfack = AL(zk, zl) * math.exp(-AL(zk, zl) * R)
    dfacl = AL(zl, zk) * math.exp(-AL(zl, zk) * R)

    def _res(kk_k, kk_l, key, sh_k, sh_l, dsh_k, dsh_l, i, j):
        s, ds = S[key], dS[key]
        m, dm = M[i, j], dM[i, j]
        p = 0.25 * (kk_k + kk_l)
        core[i, j] = p * s * (fack * sh_k + facl * sh_l) + m
        dcore[i, j] = (
            p
            * (
                ds * fack * sh_k
                + s * dfack * sh_k
                + s * fack * dsh_k
                + ds * facl * sh_l
                + s * dfacl * sh_l
                + s * facl * dsh_l
            )
            + dm
        )

    _res(KSS[zk], KSS[zl], "ss", shk["ss"], shl["ss"], dshk["ss"], dshl["ss"], 0, 0)
    if kp:
        _res(KPS[zk], KSS[zl], "ps", shk["ps"], shl["ss"], dshk["ps"], dshl["ss"], 1, 0)
    if lp:
        _res(KSS[zk], KPS[zl], "sp", shk["ss"], shl["ps"], dshk["ss"], dshl["ps"], 0, 1)
    if kp and lp:
        _res(KPS[zk], KPS[zl], "pp", shk["ps"], shl["ps"], dshk["ps"], dshl["ps"], 1, 1)
        _res(
            KPP[zk], KPP[zl], "pipi", shk["pp"], shl["pp"], dshk["pp"], dshl["pp"], 2, 2
        )
        core[3, 3] = core[2, 2]
        dcore[3, 3] = dcore[2, 2]
    if kd:
        _res(KDS[zk], KSS[zl], "ds", shk["ds"], shl["ss"], dshk["ds"], dshl["ss"], 4, 0)
    if ld:
        _res(KSS[zk], KDS[zl], "sd", shk["ss"], shl["ds"], dshk["ss"], dshl["ds"], 0, 4)
    if kd and lp:
        _res(KDS[zk], KPS[zl], "dp", shk["ds"], shl["ps"], dshk["ds"], dshl["ps"], 4, 1)
        _res(
            KDP[zk], KPP[zl], "dppi", shk["dp"], shl["pp"], dshk["dp"], dshl["pp"], 5, 2
        )
        core[6, 3] = core[5, 2]
        dcore[6, 3] = dcore[5, 2]
    if kp and ld:
        _res(KPS[zk], KDS[zl], "pd", shk["ps"], shl["ds"], dshk["ps"], dshl["ds"], 1, 4)
        _res(
            KPP[zk], KDP[zl], "pdpi", shk["pp"], shl["dp"], dshk["pp"], dshl["dp"], 2, 5
        )
        core[3, 6] = core[2, 5]
        dcore[3, 6] = dcore[2, 5]
    if kd and ld:
        _res(KDS[zk], KDS[zl], "dd", shk["ds"], shl["ds"], dshk["ds"], dshl["ds"], 4, 4)
        _res(
            KDP[zk], KDP[zl], "ddpi", shk["dp"], shl["dp"], dshk["dp"], dshl["dp"], 5, 5
        )
        core[6, 6] = core[5, 5]
        dcore[6, 6] = dcore[5, 5]
        _res(
            KDD[zk],
            KDD[zl],
            "dddel",
            shk["dd"],
            shl["dd"],
            dshk["dd"],
            dshl["dd"],
            7,
            7,
        )
        core[8, 8] = core[7, 7]
        dcore[8, 8] = dcore[7, 7]


# --------------------------------------------------------------------------- #
# Derivative rotation transforms -- port of d_trans.f + rotation application   #
# --------------------------------------------------------------------------- #


def _d_trans_driver(E):
    """Port d_trans.f: compute rotation matrix T, dT/dth (TDT), and dT/dphi (TDP).

    Returns (T, TDT, TDP) as 9x9 numpy arrays.
    T matches the existing _harmtr(3, E).
    """
    cost = E[2]
    tol = 1e-12
    if abs(cost) >= 1.0 - tol:
        sint, cosp, sinp = 0.0, 1.0, 0.0
    elif abs(cost) <= tol:
        sint, cosp, sinp = 1.0, E[0], E[1]
    else:
        sint = math.sqrt(1.0 - cost**2)
        cosp, sinp = E[0] / sint, E[1] / sint

    sqrt3 = math.sqrt(3.0)
    cos2t = cost**2 - sint**2
    sin2t = 2.0 * sint * cost
    cos2p = cosp**2 - sinp**2
    sin2p = 2.0 * sinp * cosp

    # T (9x9) -- standard HARMTR
    T = np.zeros((9, 9))
    T[0, 0] = 1.0
    # p-block
    T[1, 1] = sint * cosp
    T[2, 1] = sint * sinp
    T[3, 1] = cost
    T[1, 2] = cost * cosp
    T[2, 2] = cost * sinp
    T[3, 2] = -sint
    T[1, 3] = -sinp
    T[2, 3] = cosp
    # d-block
    T[4, 4] = (3.0 * cost**2 - 1.0) * 0.5
    T[5, 4] = sqrt3 * sin2t * cosp * 0.5
    T[6, 4] = sqrt3 * sin2t * sinp * 0.5
    T[7, 4] = sqrt3 * sint**2 * cos2p * 0.5
    T[8, 4] = sqrt3 * sint**2 * sin2p * 0.5
    T[4, 5] = -sqrt3 * sin2t * 0.5
    T[5, 5] = cos2t * cosp
    T[6, 5] = cos2t * sinp
    T[7, 5] = sin2t * cos2p * 0.5
    T[8, 5] = sin2t * sin2p * 0.5
    T[5, 6] = -cost * sinp
    T[6, 6] = cost * cosp
    T[7, 6] = -sint * sin2p
    T[8, 6] = sint * cos2p
    T[4, 7] = sqrt3 * sint**2 * 0.5
    T[5, 7] = -sin2t * cosp * 0.5
    T[6, 7] = -sin2t * sinp * 0.5
    T[7, 7] = (1.0 + cost**2) * cos2p * 0.5
    T[8, 7] = (1.0 + cost**2) * sin2p * 0.5
    T[5, 8] = sint * sinp
    T[6, 8] = -sint * cosp
    T[7, 8] = -cost * sin2p
    T[8, 8] = cost * cos2p

    # TDT = dT/dth
    dcos2t = -2.0 * sin2t
    dsin2t = 2.0 * cos2t
    dcos2p = -2.0 * sin2p
    dsin2p = 2.0 * cos2p

    TDT = np.zeros((9, 9))
    # p-block th-derivatives
    TDT[1, 1] = cost * cosp
    TDT[2, 1] = cost * sinp
    TDT[3, 1] = -sint
    TDT[1, 2] = -sint * cosp
    TDT[2, 2] = -sint * sinp
    TDT[3, 2] = -cost
    # d-block th-derivatives
    TDT[4, 4] = -1.5 * sin2t
    TDT[5, 4] = sqrt3 * dsin2t * cosp * 0.5
    TDT[6, 4] = sqrt3 * dsin2t * sinp * 0.5
    TDT[7, 4] = sqrt3 * sin2t * cos2p * 0.5
    TDT[8, 4] = sqrt3 * sin2t * sin2p * 0.5
    TDT[4, 5] = -sqrt3 * dsin2t * 0.5
    TDT[5, 5] = dcos2t * cosp
    TDT[6, 5] = dcos2t * sinp
    TDT[7, 5] = dsin2t * cos2p * 0.5
    TDT[8, 5] = dsin2t * sin2p * 0.5
    TDT[5, 6] = sint * sinp
    TDT[6, 6] = -sint * cosp
    TDT[7, 6] = -cost * sin2p
    TDT[8, 6] = cost * cos2p
    TDT[4, 7] = sqrt3 * sin2t * 0.5
    TDT[5, 7] = -dsin2t * cosp * 0.5
    TDT[6, 7] = -dsin2t * sinp * 0.5
    TDT[7, 7] = -sin2t * cos2p * 0.5
    TDT[8, 7] = -sin2t * sin2p * 0.5
    TDT[5, 8] = cost * sinp
    TDT[6, 8] = -cost * cosp
    TDT[7, 8] = sint * sin2p
    TDT[8, 8] = -sint * cos2p

    # TDP = dT/dphi
    TDP = np.zeros((9, 9))
    TDP[1, 1] = -sint * sinp
    TDP[2, 1] = sint * cosp
    TDP[1, 2] = -cost * sinp
    TDP[2, 2] = cost * cosp
    TDP[1, 3] = -cosp
    TDP[2, 3] = -sinp
    TDP[5, 4] = -sqrt3 * sin2t * sinp * 0.5
    TDP[6, 4] = sqrt3 * sin2t * cosp * 0.5
    TDP[7, 4] = sqrt3 * sint**2 * dcos2p * 0.5
    TDP[8, 4] = sqrt3 * sint**2 * dsin2p * 0.5
    TDP[5, 5] = -cos2t * sinp
    TDP[6, 5] = cos2t * cosp
    TDP[7, 5] = sin2t * dcos2p * 0.5
    TDP[8, 5] = sin2t * dsin2p * 0.5
    TDP[5, 6] = -cost * cosp
    TDP[6, 6] = -cost * sinp
    TDP[7, 6] = -sint * dsin2p
    TDP[8, 6] = sint * dcos2p
    TDP[5, 7] = sin2t * sinp * 0.5
    TDP[6, 7] = -sin2t * cosp * 0.5
    TDP[7, 7] = (1.0 + cost**2) * dcos2p * 0.5
    TDP[8, 7] = (1.0 + cost**2) * dsin2p * 0.5
    TDP[5, 8] = sint * cosp
    TDP[6, 8] = sint * sinp
    TDP[7, 8] = -cost * dsin2p
    TDP[8, 8] = cost * dcos2p

    # Fix the sign for TDP[3,3] -- d/dphi of T[3,3]=0 is 0
    # TDP is lower-triangle for some entries in d_trans.f
    TDP[2, 2] = cost * cosp  # match d_trans line

    return T, TDT, TDP


def _rotate_derivatives_to_global(
    H_local, dH_local, core_local, dcore_local, T, TDT, TDP
):
    """Rotate local-frame H blocks and their derivatives to the global frame.

    Uses the matrix product rule (matching ROTINT.f GRAD):
      dH_global/dR = T @ dH_local @ T^T
      dH_global/dth = TDT @ H_local @ T^T + T @ H_local @ TDT^T
      dH_global/dphi = TDP @ H_local @ T^T + T @ H_local @ TDP^T

    Returns (HK1_DR, HK1_DT, HK1_DP), (HL1_DR, HL1_DT, HL1_DP),
            (HKL2_DR, HKL2_DT, HKL2_DP).
    """
    # R-derivatives: simple rotation
    HK1_DR = T @ dH_local[0] @ T.T
    HL1_DR = T @ dH_local[1] @ T.T
    HKL2_DR = T @ dcore_local @ T.T

    # th-derivatives: product rule
    Hk_T = T @ H_local[0] @ T.T
    Hl_T = T @ H_local[1] @ T.T
    Core_T = T @ core_local @ T.T

    HK1_DT = TDT @ H_local[0] @ T.T + T @ H_local[0] @ TDT.T
    HL1_DT = TDT @ H_local[1] @ T.T + T @ H_local[1] @ TDT.T
    HKL2_DT = TDT @ core_local @ T.T + T @ core_local @ TDT.T

    # phi-derivatives: product rule
    HK1_DP = TDP @ H_local[0] @ T.T + T @ H_local[0] @ TDP.T
    HL1_DP = TDP @ H_local[1] @ T.T + T @ H_local[1] @ TDP.T
    HKL2_DP = TDP @ core_local @ T.T + T @ core_local @ TDP.T

    return (
        (HK1_DR, HK1_DT, HK1_DP),
        (HL1_DR, HL1_DT, HL1_DP),
        (HKL2_DR, HKL2_DT, HKL2_DP),
    )


def _pair_blocks_deriv(zk: int, zl: int, rk, rl) -> _PairBlocksDeriv:
    """Compute per-pair two-center quantities AND local-frame R-derivatives.

    This is the Phase 2 output: local-frame 9x9 matrices for the SKL, GAMM,
    MULL, HK1, HL1, HKL2 blocks AND their d/dR.  The standard global-frame
    ``_PairBlocks`` is also returned (computed independently via
    ``_pair_blocks`` for correctness).
    """
    # --- geometry ---
    d = np.asarray(rl, float) - np.asarray(rk, float)
    R = float(np.linalg.norm(d))
    E = d / R

    # --- atom flags ---
    kp, lp = n_basis(zk) >= 4, n_basis(zl) >= 4
    kd, ld = n_basis(zk) >= 9, n_basis(zl) >= 9
    nk, nl = n_principal(zk), n_principal(zl)
    npk, npl = n_p_principal(zk), n_p_principal(zl)
    nkd, nld = n_d_principal(zk), n_d_principal(zl)
    zsk, zpk, zdk = MUS[zk], MUP[zk], MUD.get(zk, 0.0)
    zsl, zpl, zdl = MUS[zl], MUP[zl], MUD.get(zl, 0.0)

    # ----- helper shortcuts -----
    def _s2(n1, l1, m1, e1, n2, l2, m2, e2):
        return _ki.s2int(n1, l1, m1, e1, n2, l2, m2, e2, R)

    def _ds2(n1, l1, m1, e1, n2, l2, m2, e2):
        return _ki.ds2int(n1, l1, m1, e1, n2, l2, m2, e2, R)

    def _c2(n1, e1, n2, e2):
        return _ki.c2int(n1, 0, 0, e1, n2, 0, 0, e2, R)

    def _dc(n1, e1, n2, e2):
        return _kd.dc2int(n1, 0, 0, e1, n2, 0, 0, e2, R)

    # ----- Overlaps S and dS/dR (OVERLAP.f GRAD) -----
    S = {
        "ss": 0.0,
        "ps": 0.0,
        "sp": 0.0,
        "pp": 0.0,
        "pipi": 0.0,
        "ds": 0.0,
        "sd": 0.0,
        "dp": 0.0,
        "pd": 0.0,
        "dd": 0.0,
        "dppi": 0.0,
        "pdpi": 0.0,
        "ddpi": 0.0,
        "dddel": 0.0,
    }
    dS = dict(S)

    S["ss"] = _s2(nk, 0, 0, zsk, nl, 0, 0, zsl)
    dS["ss"] = _ds2(nk, 0, 0, zsk, nl, 0, 0, zsl)
    if kp:
        S["ps"] = _s2(npk, 1, 0, zpk, nl, 0, 0, zsl)
        dS["ps"] = _ds2(npk, 1, 0, zpk, nl, 0, 0, zsl)
    if lp:
        S["sp"] = _s2(nk, 0, 0, zsk, npl, 1, 0, zpl)
        dS["sp"] = _ds2(nk, 0, 0, zsk, npl, 1, 0, zpl)
    if kp and lp:
        S["pp"] = _s2(npk, 1, 0, zpk, npl, 1, 0, zpl)
        S["pipi"] = _s2(npk, 1, 1, zpk, npl, 1, 1, zpl)
        dS["pp"] = _ds2(npk, 1, 0, zpk, npl, 1, 0, zpl)
        dS["pipi"] = _ds2(npk, 1, 1, zpk, npl, 1, 1, zpl)
    if kd:
        S["ds"] = _s2(nkd, 2, 0, zdk, nl, 0, 0, zsl)
        dS["ds"] = _ds2(nkd, 2, 0, zdk, nl, 0, 0, zsl)
        if lp:
            S["dp"] = _s2(nkd, 2, 0, zdk, npl, 1, 0, zpl)
            S["dppi"] = _s2(nkd, 2, 1, zdk, npl, 1, 1, zpl)
            dS["dp"] = _ds2(nkd, 2, 0, zdk, npl, 1, 0, zpl)
            dS["dppi"] = _ds2(nkd, 2, 1, zdk, npl, 1, 1, zpl)
    if ld:
        S["sd"] = _s2(nk, 0, 0, zsk, nld, 2, 0, zdl)
        dS["sd"] = _ds2(nk, 0, 0, zsk, nld, 2, 0, zdl)
        if kp:
            S["pd"] = _s2(npk, 1, 0, zpk, nld, 2, 0, zdl)
            S["pdpi"] = _s2(npk, 1, 1, zpk, nld, 2, 1, zdl)
            dS["pd"] = _ds2(npk, 1, 0, zpk, nld, 2, 0, zdl)
            dS["pdpi"] = _ds2(npk, 1, 1, zpk, nld, 2, 1, zdl)
    if kd and ld:
        S["dd"] = _s2(nkd, 2, 0, zdk, nld, 2, 0, zdl)
        S["ddpi"] = _s2(nkd, 2, 1, zdk, nld, 2, 1, zdl)
        S["dddel"] = _s2(nkd, 2, 2, zdk, nld, 2, 2, zdl)
        dS["dd"] = _ds2(nkd, 2, 0, zdk, nld, 2, 0, zdl)
        dS["ddpi"] = _ds2(nkd, 2, 1, zdk, nld, 2, 1, zdl)
        dS["dddel"] = _ds2(nkd, 2, 2, zdk, nld, 2, 2, zdl)

    # ----- Coulomb g and dg/dR (COULOM.f GRAD) -----
    gam_k = {
        "ps": _c2(npk, zpk, nl, zsl) if kp else 0.0,
        "pp": _c2(npk, zpk, npl, zpl) if (kp and lp) else 0.0,
        "pd": _c2(npk, zpk, nld, zdl) if (kp and ld) else 0.0,
        "ds": _c2(nkd, zdk, nl, zsl) if kd else 0.0,
        "dp": _c2(nkd, zdk, npl, zpl) if (kd and lp) else 0.0,
        "dd": _c2(nkd, zdk, nld, zdl) if (kd and ld) else 0.0,
    }
    gam_l = {
        "ps": _c2(npl, zpl, nk, zsk) if lp else 0.0,
        "pp": _c2(npl, zpl, npk, zpk) if (kp and lp) else 0.0,
        "pd": _c2(npl, zpl, nkd, zdk) if (lp and kd) else 0.0,
        "ds": _c2(nld, zdl, nk, zsk) if ld else 0.0,
        "dp": _c2(nld, zdl, npk, zpk) if (kp and ld) else 0.0,
        "dd": _c2(nld, zdl, nkd, zdk) if (ld and kd) else 0.0,
    }
    d_gam_k = {
        k: _dc(*args) if gam_k[k] != 0.0 else 0.0
        for k, args in {
            "ps": (npk, zpk, nl, zsl),
            "pp": (npk, zpk, npl, zpl),
            "pd": (npk, zpk, nld, zdl),
            "ds": (nkd, zdk, nl, zsl),
            "dp": (nkd, zdk, npl, zpl),
            "dd": (nkd, zdk, nld, zdl),
        }.items()
    }
    d_gam_l = {
        k: _dc(*args) if gam_l[k] != 0.0 else 0.0
        for k, args in {
            "ps": (npl, zpl, nk, zsk),
            "pp": (npl, zpl, npk, zpk),
            "pd": (npl, zpl, nkd, zdk),
            "ds": (nld, zdl, nk, zsk),
            "dp": (nld, zdl, npk, zpk),
            "dd": (nld, zdl, nkd, zdk),
        }.items()
    }

    # We also need the ss gamma, which isn't in gam_k/gam_l.
    # Compute it directly: g(s_K, s_L) = c2int(nk, 0, 0, zsk, nl, 0, 0, zsl, R)
    gam_ss = _c2(nk, zsk, nl, zsl)
    d_gam_ss = _dc(nk, zsk, nl, zsl)

    # ----- V2CORE + VCORR (v2core.f, vcorrk.f, vcorrl.f) -----
    from .msindo import _lmunu_local, _v2core_one_side

    hk = _v2core_one_side(zk, zl, R, gam_k)
    hl = _v2core_one_side(zl, zk, R, gam_l)
    dhk = _v2core_one_side_deriv(zk, zl, R, gam_k, d_gam_k)
    dhl = _v2core_one_side_deriv(zl, zk, R, gam_l, d_gam_l)

    # ----- ENEG (one-electron energies) -----
    ek, el = eneg(zk), eneg(zl)

    # ----- LMUNU M and dM/dR -----
    M_dict = _lmunu_local(zk, zl, R, S)
    Mmat = np.zeros((9, 9))
    dMmat = np.zeros((9, 9))
    _fill_M_local(Mmat, M_dict, kp, lp, kd, ld)
    _lmunu_local_deriv(zk, zl, R, S, dS, dMmat)

    # ----- SH values FOR DELTAH (pre-HORTH, matching _pair_blocks) -----
    # In _pair_blocks, SH = ENEG + H (where H comes from V2CORE only, before HORTH).
    # HORTH corrections are applied separately and do NOT feed into the DELTAH formula.
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
    dshk = {
        "ss": dhk.HSS,
        "ps": dhk.HPS,
        "pp": dhk.HPP,
        "ds": dhk.HDS,
        "dp": dhk.HDP,
        "dd": dhk.HDD,
    }
    dshl = {
        "ss": dhl.HSS,
        "ps": dhl.HPS,
        "pp": dhl.HPP,
        "ds": dhl.HDS,
        "dp": dhl.HDP,
        "dd": dhl.HDD,
    }

    # ----- HORTH with derivatives -----
    # Work on mutable copies since HORTH modifies in-place.
    class _MH:
        pass

    hk_m, hl_m = _MH(), _MH()
    dhk_m, dhl_m = _MH(), _MH()
    for a in ("HSS", "HPS", "HPP", "HDS", "HDP", "HDD"):
        setattr(hk_m, a, getattr(hk, a))
        setattr(hl_m, a, getattr(hl, a))
        setattr(dhk_m, a, getattr(dhk, a))
        setattr(dhl_m, a, getattr(dhl, a))

    vk, vl = _vfak(zk, zl), _vfak(zl, zk)
    _horth_deriv(hk_m, hl_m, dhk_m, dhl_m, vk, vl, S, dS, Mmat, dMmat, kp, lp, kd, ld)

    # ----- DELTAH with derivatives -----
    core = np.zeros((9, 9))
    dcore = np.zeros((9, 9))
    _deltah_deriv(
        zk, zl, R, core, dcore, S, dS, Mmat, dMmat, shk, shl, dshk, dshl, kp, lp, kd, ld
    )

    # ----- Build local-frame 9x9 matrices -----
    S_local = np.zeros((9, 9))
    dS_local = np.zeros((9, 9))
    _fill_S_local(S_local, S, kp, lp, kd, ld)
    _fill_S_local(dS_local, dS, kp, lp, kd, ld)

    gamma_local = np.zeros((9, 9))
    d_gamma_local = np.zeros((9, 9))
    # Fill gamma: need to include ss at (0,0)
    gamma_local[0, 0] = gam_ss
    d_gamma_local[0, 0] = d_gam_ss
    _fill_gamma_local(gamma_local, gam_k, gam_l, kp, lp, kd, ld)
    _fill_gamma_local(d_gamma_local, d_gam_k, d_gam_l, kp, lp, kd, ld)

    # Diagonal HK1/HL1 blocks (local frame, only diag elements populated)
    # Non-derivative versions -- needed for th/phi product rule
    HK1_local = np.diag(
        [
            hk_m.HSS,
            hk_m.HPS,
            hk_m.HPP,
            hk_m.HPP,
            hk_m.HDS,
            hk_m.HDP,
            hk_m.HDP,
            hk_m.HDD,
            hk_m.HDD,
        ]
    )
    HL1_local = np.diag(
        [
            hl_m.HSS,
            hl_m.HPS,
            hl_m.HPP,
            hl_m.HPP,
            hl_m.HDS,
            hl_m.HDP,
            hl_m.HDP,
            hl_m.HDD,
            hl_m.HDD,
        ]
    )
    # Derivative versions
    dHK1_local = np.diag(
        [
            dhk_m.HSS,
            dhk_m.HPS,
            dhk_m.HPP,
            dhk_m.HPP,
            dhk_m.HDS,
            dhk_m.HDP,
            dhk_m.HDP,
            dhk_m.HDD,
            dhk_m.HDD,
        ]
    )
    dHL1_local = np.diag(
        [
            dhl_m.HSS,
            dhl_m.HPS,
            dhl_m.HPP,
            dhl_m.HPP,
            dhl_m.HDS,
            dhl_m.HDP,
            dhl_m.HDP,
            dhl_m.HDD,
            dhl_m.HDD,
        ]
    )

    # ----- Phase 3: rotate derivatives to global frame (ROTINT GRAD) -----
    T, TDT, TDP = _d_trans_driver(E)
    (HK1_DR, HK1_DT, HK1_DP), (HL1_DR, HL1_DT, HL1_DP), (HKL2_DR, HKL2_DT, HKL2_DP) = (
        _rotate_derivatives_to_global(
            (HK1_local, HL1_local),
            (dHK1_local, dHL1_local),
            core,
            dcore,
            T,
            TDT,
            TDP,
        )
    )

    # ----- Standard _PairBlocks (global frame) -----
    pb = _pair_blocks(zk, zl, rk, rl)

    return _PairBlocksDeriv(
        pb=pb,
        S_local=S_local,
        dS_local=dS_local,
        gamma_local=gamma_local,
        d_gamma_local=d_gamma_local,
        M_local=Mmat,
        dM_local=dMmat,
        dHK1_local=dHK1_local,
        dHL1_local=dHL1_local,
        dHKL2_local=dcore,
        HK1_DR=HK1_DR,
        HL1_DR=HL1_DR,
        HKL2_DR=HKL2_DR,
        HK1_DT=HK1_DT,
        HL1_DT=HL1_DT,
        HKL2_DT=HKL2_DT,
        HK1_DP=HK1_DP,
        HL1_DP=HL1_DP,
        HKL2_DP=HKL2_DP,
        E=E,
        R=R,
    )
