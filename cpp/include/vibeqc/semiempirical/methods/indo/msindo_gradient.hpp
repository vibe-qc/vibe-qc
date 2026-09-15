// MSINDO analytic nuclear gradient — C++ port of Phases 2-4
//
// Ported from the validated Python reference (msindo_pair_deriv.py +
// msindo_gradient_analytic.py) which was in turn ported from Fortran
// (intdrv.f, rotint.f, dedxyzk.f, dedxyzl.f).
//
// Header-only, namespace vibeqc::semiempirical::indo.
// Depends on msindo_integrals.hpp (for ds2int/dc2int/dv2int/dharris/overlp_grad)
// and indo_engine.hpp (for parameter set and existing pair-block helpers).

#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/semiempirical/methods/indo/msindo_integrals.hpp"
#include "vibeqc/semiempirical/methods/indo/indo_engine.hpp"

namespace vibeqc {
namespace semiempirical {
namespace indo {

// ========================================================================= //
// PairBlocksDeriv — per-pair integrals + local/global derivative matrices   //
// ========================================================================= //

struct PairBlocksDeriv {
    PairBlocks pb;
    // local-frame 9x9 matrices (bond-aligned)
    Eigen::Matrix<double, 9, 9> S_local, dS_local;
    Eigen::Matrix<double, 9, 9> gamma_local, d_gamma_local;
    Eigen::Matrix<double, 9, 9> M_local, dM_local;
    // local-frame diagonal H derivatives
    Eigen::Matrix<double, 9, 9> dHK1_local, dHL1_local;
    // local-frame off-diagonal core derivative
    Eigen::Matrix<double, 9, 9> dHKL2_local;
    // --- Phase 3: global-frame (R, θ, φ) derivatives ---
    Eigen::Matrix<double, 9, 9> HK1_DR, HL1_DR, HKL2_DR;
    Eigen::Matrix<double, 9, 9> HK1_DT, HL1_DT, HKL2_DT;
    Eigen::Matrix<double, 9, 9> HK1_DP, HL1_DP, HKL2_DP;
    // geometry
    std::array<double, 3> E;
    double R;

    PairBlocksDeriv() {
        S_local.setZero(); dS_local.setZero();
        gamma_local.setZero(); d_gamma_local.setZero();
        M_local.setZero(); dM_local.setZero();
        dHK1_local.setZero(); dHL1_local.setZero(); dHKL2_local.setZero();
        HK1_DR.setZero(); HL1_DR.setZero(); HKL2_DR.setZero();
        HK1_DT.setZero(); HL1_DT.setZero(); HKL2_DT.setZero();
        HK1_DP.setZero(); HL1_DP.setZero(); HKL2_DP.setZero();
        E = {0,0,0}; R = 0;
    }
};

// ========================================================================= //
// Frozen-core overlap derivatives (valrum.f GRAD)                           //
// ========================================================================= //

inline RumpfOverlaps rumpf_overlaps_deriv(int z_val, int z_core, double R,
                                          const MsindoParameterSet& p) {
    RumpfOverlaps dri{};  // all zero
    if (z_core <= 2) return dri;
    int ns = MsindoParameterSet::n_principal(z_val);
    int npr = MsindoParameterSet::n_p_principal(z_val);
    int nd = MsindoParameterSet::n_d_principal(z_val);
    double zs = p.get(p.MUS, z_val);
    double zp = p.get(p.MUP, z_val);
    double zd = p.get(p.MUD, z_val);
    if (zp == 0.0) { npr = 0; zp = 0; }
    if (zd == 0.0) { nd = 0; zd = 0; }
    bool kp = (p.n_basis(z_val) >= 4);
    bool kd = (p.n_basis(z_val) >= 9 && zd != 0.0);

    double z1s = p.get(p.TAU1S, z_core);
    dri.S1S = ds2int(ns, 0, 0, zs, 1, 0, 0, z1s, R);
    if (kp) dri.PS1S = ds2int(npr, 1, 0, zp, 1, 0, 0, z1s, R);
    if (kd) dri.DS1S = ds2int(nd, 2, 0, zd, 1, 0, 0, z1s, R);
    if (z_core > 10) {
        double z2s = p.get(p.TAU2S, z_core), z2p = p.get(p.TAU2P, z_core);
        dri.S2S = ds2int(ns, 0, 0, zs, 2, 0, 0, z2s, R);
        dri.S2PS = ds2int(ns, 0, 0, zs, 2, 1, 0, z2p, R);
        if (kp) {
            dri.PS2S = ds2int(npr, 1, 0, zp, 2, 0, 0, z2s, R);
            dri.PS2PS = ds2int(npr, 1, 0, zp, 2, 1, 0, z2p, R);
            dri.PP2PP = ds2int(npr, 1, 1, zp, 2, 1, 1, z2p, R);
        }
        if (kd) {
            dri.DS2S = ds2int(nd, 2, 0, zd, 2, 0, 0, z2s, R);
            dri.DS2PS = ds2int(nd, 2, 0, zd, 2, 1, 0, z2p, R);
            dri.DP2PP = ds2int(nd, 2, 1, zd, 2, 1, 1, z2p, R);
        }
    }
    if (z_core > 18) {
        double z3s = p.get(p.TAU3S, z_core), z3p = p.get(p.TAU3P, z_core);
        dri.S3S = ds2int(ns, 0, 0, zs, 3, 0, 0, z3s, R);
        dri.S3PS = ds2int(ns, 0, 0, zs, 3, 1, 0, z3p, R);
        if (kp) {
            dri.PS3S = ds2int(npr, 1, 0, zp, 3, 0, 0, z3s, R);
            dri.PS3PS = ds2int(npr, 1, 0, zp, 3, 1, 0, z3p, R);
            dri.PP3PP = ds2int(npr, 1, 1, zp, 3, 1, 1, z3p, R);
        }
        if (kd) {
            dri.DS3S = ds2int(nd, 2, 0, zd, 3, 0, 0, z3s, R);
            dri.DS3PS = ds2int(nd, 2, 0, zd, 3, 1, 0, z3p, R);
            dri.DP3PP = ds2int(nd, 2, 1, zd, 3, 1, 1, z3p, R);
        }
    }
    if (z_core > 30) {
        double z3d = p.get(p.TAU3D, z_core);
        dri.S3DS = ds2int(ns, 0, 0, zs, 3, 2, 0, z3d, R);
        if (kp) {
            dri.PS3DS = ds2int(npr, 1, 0, zp, 3, 2, 0, z3d, R);
            dri.PP3DP = ds2int(npr, 1, 1, zp, 3, 2, 1, z3d, R);
        }
        if (kd) {
            dri.DS3DS = ds2int(nd, 2, 0, zd, 3, 2, 0, z3d, R);
            dri.DP3DP = ds2int(nd, 2, 1, zd, 3, 2, 1, z3d, R);
            dri.DD3DD = ds2int(nd, 2, 2, zd, 3, 2, 2, z3d, R);
        }
    }
    return dri;
}

// ========================================================================= //
// dcor_dR helper                                                             //
// ========================================================================= //

inline double dcor_dR(double za, double zb, double R) {
    double num = -0.5 * (za * za + zb * zb);
    double den = 1.0 + 0.5 * (za + zb) * R;
    return -num * 0.5 * (za + zb) / (den * den);
}

// ========================================================================= //
// V2CORE + VCORR derivative (v2core.f, vcorrk.f GRAD)                       //
// ========================================================================= //

inline void vcork_deriv(int z_val, int z_core, double R,
                        const GammaCache& gam, const GammaCache& d_gam,
                        const MsindoParameterSet& p,
                        double& hps, double& hpp, double& hds, double& hdp, double& hdd,
                        bool kp, bool kd) {
    int npc = MsindoParameterSet::n_principal(z_core);
    int npcp = MsindoParameterSet::n_p_principal(z_core);
    int ncd = MsindoParameterSet::n_d_principal(z_core);
    double zs_c = p.get(p.MUS, z_core);
    double zp_c = p.get(p.MUP, z_core);
    double zd_c = p.get(p.MUD, z_core);

    // p-orbital corrections — VCORR uses (s_core | pσ/pπ_valence) integrals
    if (kp) {
        double dc_ps_s = dc2int(npc, 0, 0, zs_c, MsindoParameterSet::n_p_principal(z_val), 1, 0, p.get(p.MUP, z_val), R);
        double dc_ps_p = dc2int(npc, 0, 0, zs_c, MsindoParameterSet::n_p_principal(z_val), 1, 1, p.get(p.MUP, z_val), R);
        hps += (dc_ps_s - d_gam.ps) * p.get(p.LS, z_core);
        hpp += (dc_ps_p - d_gam.ps) * p.get(p.LS, z_core);
        if (p.iget(p.MP, z_core) != 0) {
            double dc_pp_s = dc2int(npcp, 0, 0, zp_c, MsindoParameterSet::n_p_principal(z_val), 1, 0, p.get(p.MUP, z_val), R);
            double dc_pp_p = dc2int(npcp, 0, 0, zp_c, MsindoParameterSet::n_p_principal(z_val), 1, 1, p.get(p.MUP, z_val), R);
            hps += (dc_pp_s - d_gam.pp) * p.get(p.MP, z_core);
            hpp += (dc_pp_p - d_gam.pp) * p.get(p.MP, z_core);
        }
        if (p.iget(p.ND, z_core) != 0) {
            double dc_pd_s = dc2int(ncd, 0, 0, zd_c, MsindoParameterSet::n_p_principal(z_val), 1, 0, p.get(p.MUP, z_val), R);
            double dc_pd_p = dc2int(ncd, 0, 0, zd_c, MsindoParameterSet::n_p_principal(z_val), 1, 1, p.get(p.MUP, z_val), R);
            hps += (dc_pd_s - d_gam.pd) * p.get(p.ND, z_core);
            hpp += (dc_pd_p - d_gam.pd) * p.get(p.ND, z_core);
        }
    }
    // d-orbital corrections
    if (kd) {
        int ndv = MsindoParameterSet::n_d_principal(z_val);
        double zdv = p.get(p.MUD, z_val);
        double dc_ds_s = dc2int(npc, 0, 0, zs_c, ndv, 2, 0, zdv, R);
        double dc_ds_p = dc2int(npc, 0, 0, zs_c, ndv, 2, 1, zdv, R);
        double dc_ds_d = dc2int(npc, 0, 0, zs_c, ndv, 2, 2, zdv, R);
        hds += (dc_ds_s - d_gam.ds) * p.get(p.LS, z_core);
        hdp += (dc_ds_p - d_gam.ds) * p.get(p.LS, z_core);
        hdd += (dc_ds_d - d_gam.ds) * p.get(p.LS, z_core);
        if (p.iget(p.MP, z_core) != 0) {
            double dc_dp_s = dc2int(npcp, 0, 0, zp_c, ndv, 2, 0, zdv, R);
            double dc_dp_p = dc2int(npcp, 0, 0, zp_c, ndv, 2, 1, zdv, R);
            double dc_dp_d = dc2int(npcp, 0, 0, zp_c, ndv, 2, 2, zdv, R);
            hds += (dc_dp_s - d_gam.dp) * p.get(p.MP, z_core);
            hdp += (dc_dp_p - d_gam.dp) * p.get(p.MP, z_core);
            hdd += (dc_dp_d - d_gam.dp) * p.get(p.MP, z_core);
        }
        if (p.iget(p.ND, z_core) != 0) {
            double dc_dd_s = dc2int(ncd, 0, 0, zd_c, ndv, 2, 0, zdv, R);
            double dc_dd_p = dc2int(ncd, 0, 0, zd_c, ndv, 2, 1, zdv, R);
            double dc_dd_d = dc2int(ncd, 0, 0, zd_c, ndv, 2, 2, zdv, R);
            hds += (dc_dd_s - d_gam.dd) * p.get(p.ND, z_core);
            hdp += (dc_dd_p - d_gam.dd) * p.get(p.ND, z_core);
            hdd += (dc_dd_d - d_gam.dd) * p.get(p.ND, z_core);
        }
    }
}

inline CoreH v2core_one_side_deriv(int z_val, int z_core, double R,
                                    const GammaCache& gam, const GammaCache& d_gam,
                                    const MsindoParameterSet& p) {
    double zc = p.get(p.MP, z_core); // this gets used differently — use LS+MP+ND
    zc = p.get(p.LS, z_core) + p.get(p.MP, z_core) + p.get(p.ND, z_core);
    int ns = MsindoParameterSet::n_principal(z_val);
    int npr = MsindoParameterSet::n_p_principal(z_val);
    int nd = MsindoParameterSet::n_d_principal(z_val);
    double zs = p.get(p.MUS, z_val);
    double zp = p.get(p.MUP, z_val);
    double zd = p.get(p.MUD, z_val);
    bool kp = (p.n_basis(z_val) >= 4);
    bool kd = (p.n_basis(z_val) >= 9);

    CoreH dh{};
    RumpfOverlaps ri = rumpf_overlaps(z_val, z_core, R, p);
    RumpfOverlaps dri = rumpf_overlaps_deriv(z_val, z_core, R, p);

    auto dfcp_2 = [&](double dri_val, double ri_val, double fcp_val) {
        return 2.0 * fcp_val * dri_val * ri_val;
    };

    dh.HSS = -zc * dv2int(ns, 0, 0, zs, R);
    if (z_core > 2) dh.HSS += dfcp_2(dri.S1S, ri.S1S, p.get(p.FCP1S, z_core));
    if (z_core > 10) {
        dh.HSS += dfcp_2(dri.S2S, ri.S2S, p.get(p.FCP2S, z_core));
        dh.HSS += dfcp_2(dri.S2PS, ri.S2PS, p.get(p.FCP2P, z_core));
    }
    if (z_core > 18) {
        dh.HSS += dfcp_2(dri.S3S, ri.S3S, p.get(p.FCP3S, z_core));
        dh.HSS += dfcp_2(dri.S3PS, ri.S3PS, p.get(p.FCP3P, z_core));
    }
    if (z_core > 30) dh.HSS += dfcp_2(dri.S3DS, ri.S3DS, p.get(p.FCP3D, z_core));

    if (kp) {
        dh.HPS = -zc * dv2int(npr, 1, 0, zp, R);
        dh.HPP = -zc * dv2int(npr, 1, 1, zp, R);
        if (z_core > 2) dh.HPS += dfcp_2(dri.PS1S, ri.PS1S, p.get(p.FCP1S, z_core));
        if (z_core > 10) {
            dh.HPS += dfcp_2(dri.PS2S, ri.PS2S, p.get(p.FCP2S, z_core));
            dh.HPS += dfcp_2(dri.PS2PS, ri.PS2PS, p.get(p.FCP2P, z_core));
            dh.HPP += dfcp_2(dri.PP2PP, ri.PP2PP, p.get(p.FCP2P, z_core));
        }
        if (z_core > 18) {
            dh.HPS += dfcp_2(dri.PS3S, ri.PS3S, p.get(p.FCP3S, z_core));
            dh.HPS += dfcp_2(dri.PS3PS, ri.PS3PS, p.get(p.FCP3P, z_core));
            dh.HPP += dfcp_2(dri.PP3PP, ri.PP3PP, p.get(p.FCP3P, z_core));
        }
        if (z_core > 30) {
            dh.HPS += dfcp_2(dri.PS3DS, ri.PS3DS, p.get(p.FCP3D, z_core));
            dh.HPP += dfcp_2(dri.PP3DP, ri.PP3DP, p.get(p.FCP3D, z_core));
        }
    }
    if (kd) {
        dh.HDS = -zc * dv2int(nd, 2, 0, zd, R);
        dh.HDP = -zc * dv2int(nd, 2, 1, zd, R);
        dh.HDD = -zc * dv2int(nd, 2, 2, zd, R);
        if (z_core > 2) dh.HDS += dfcp_2(dri.DS1S, ri.DS1S, p.get(p.FCP1S, z_core));
        if (z_core > 10) {
            dh.HDS += dfcp_2(dri.DS2S, ri.DS2S, p.get(p.FCP2S, z_core));
            dh.HDS += dfcp_2(dri.DS2PS, ri.DS2PS, p.get(p.FCP2P, z_core));
            dh.HDP += dfcp_2(dri.DP2PP, ri.DP2PP, p.get(p.FCP2P, z_core));
        }
        if (z_core > 18) {
            dh.HDS += dfcp_2(dri.DS3S, ri.DS3S, p.get(p.FCP3S, z_core));
            dh.HDS += dfcp_2(dri.DS3PS, ri.DS3PS, p.get(p.FCP3P, z_core));
            dh.HDP += dfcp_2(dri.DP3PP, ri.DP3PP, p.get(p.FCP3P, z_core));
        }
        if (z_core > 30) {
            dh.HDS += dfcp_2(dri.DS3DS, ri.DS3DS, p.get(p.FCP3D, z_core));
            dh.HDP += dfcp_2(dri.DP3DP, ri.DP3DP, p.get(p.FCP3D, z_core));
            dh.HDD += dfcp_2(dri.DD3DD, ri.DD3DD, p.get(p.FCP3D, z_core));
        }
    }

    // VCORR penetration derivative
    vcork_deriv(z_val, z_core, R, gam, d_gam, p, dh.HPS, dh.HPP, dh.HDS, dh.HDP, dh.HDD, kp, kd);
    return dh;
}

// ========================================================================= //
// LMUNU derivative (lmunu.f GRAD)                                           //
// ========================================================================= //

// Helper: d/dR [dcor * s * (1 - s)] — for positive s where |s| = s
inline double dm_1ms(double za, double zb, double s, double ds, double R) {
    double dc = -0.5 * (za * za + zb * zb) / (1.0 + 0.5 * (za + zb) * R);
    double ddc = dcor_dR(za, zb, R);
    return ddc * s * (1.0 - s) + dc * ds * (1.0 - 2.0 * s);
}
// d/dR [dcor * s * (1 + s)]
inline double dm_1ps(double za, double zb, double s, double ds, double R) {
    double dc = -0.5 * (za * za + zb * zb) / (1.0 + 0.5 * (za + zb) * R);
    double ddc = dcor_dR(za, zb, R);
    return ddc * s * (1.0 + s) + dc * ds * (1.0 + 2.0 * s);
}
// d/dR [dcor * s * (1 - |s|)]
inline double dm_abs(double za, double zb, double s, double ds, double R) {
    double dc = -0.5 * (za * za + zb * zb) / (1.0 + 0.5 * (za + zb) * R);
    double ddc = dcor_dR(za, zb, R);
    double as_ = std::abs(s);
    double sgn = (s >= 0) ? 1.0 : -1.0;
    return ddc * s * (1.0 - as_) + dc * ds * (1.0 - as_) - dc * s * sgn * ds;
}

inline void lmunu_local_deriv(int zk, int zl, double R, const LocalS& S, const LocalS& dS,
                               Eigen::Matrix<double, 9, 9>& dM, const MsindoParameterSet& p) {
    dM.setZero();
    double zsK = p.get(p.MUS, zk), zpK = p.get(p.MUP, zk), zdK = p.get(p.MUD, zk);
    double zsL = p.get(p.MUS, zl), zpL = p.get(p.MUP, zl), zdL = p.get(p.MUD, zl);
    bool kp = (p.n_basis(zk) >= 4), lp = (p.n_basis(zl) >= 4);
    bool kd = (p.n_basis(zk) >= 9), ld = (p.n_basis(zl) >= 9);

    dM(0,0) = dm_1ms(zsK, zsL, S.ss, dS.ss, R);
    if (kp && zk > 2) dM(1,0) = dm_1ms(zpK, zsL, S.ps, dS.ps, R);
    if (lp && zl > 2) dM(0,1) = dm_1ps(zsK, zpL, S.sp, dS.sp, R);
    if (kp && lp && zk > 2 && zl > 2) {
        dM(1,1) = dm_abs(zpK, zpL, S.pp, dS.pp, R);
        dM(2,2) = dm_abs(zpK, zpL, S.pipi, dS.pipi, R);
        dM(3,3) = dM(2,2);
    }
    if (kd) {
        dM(4,0) = dm_1ms(zdK, zsL, S.ds, dS.ds, R);
        if (lp) { dM(4,1)=dm_abs(zdK,zpL,S.dp,dS.dp,R); dM(5,2)=dm_abs(zdK,zpL,S.dppi,dS.dppi,R); dM(6,3)=dM(5,2); }
    }
    if (ld) {
        dM(0,4) = dm_1ms(zsK, zdL, S.sd, dS.sd, R);
        if (kp) { dM(1,4)=dm_abs(zpK,zdL,S.pd,dS.pd,R); dM(2,5)=dm_abs(zpK,zdL,S.pdpi,dS.pdpi,R); dM(3,6)=dM(2,5); }
    }
    if (kd && ld) {
        dM(4,4)=dm_abs(zdK,zdL,S.dd,dS.dd,R); dM(5,5)=dm_abs(zdK,zdL,S.ddpi,dS.ddpi,R); dM(6,6)=dM(5,5);
        dM(7,7)=dm_abs(zdK,zdL,S.dddel,dS.dddel,R); dM(8,8)=dM(7,7);
    }

    // 1s averaging derivative (lmunu.f tail)
    if (zk > 2 && zl > 2) return;
    auto d_avg = [&](double za, double zb, double s, double ds) {
        double dc = 0.5*(za+zb)*R, ddc = 0.5*(za+zb);
        double e = (1.0-std::exp(-dc))/(1.0+dc);
        double de = (ddc*std::exp(-dc)*(1.0+dc) - (1.0-std::exp(-dc))*ddc)/((1.0+dc)*(1.0+dc));
        return -ds*e - s*de;
    };
    if (zk == 1 && zl > 2) {
        dM(0,0) = (dM(0,0) + d_avg(zsK,zsL,S.ss,dS.ss))/2.0;
        if (lp) dM(0,1) = (dM(0,1) + d_avg(zsK,zpL,S.sp,dS.sp))/2.0;
        if (ld) dM(0,4) = (dM(0,4) + d_avg(zsK,zdL,S.sd,dS.sd))/2.0;
    } else if (zk > 2 && zl == 1) {
        dM(0,0) = (dM(0,0) + d_avg(zsK,zsL,S.ss,dS.ss))/2.0;
        if (kp) dM(1,0) = (dM(1,0) + d_avg(zpK,zsL,S.ps,dS.ps))/2.0;
        if (kd) dM(4,0) = (dM(4,0) + d_avg(zdK,zsL,S.ds,dS.ds))/2.0;
    } else {
        dM(0,0) = (dM(0,0) + d_avg(zsK,zsL,S.ss,dS.ss))/2.0;
    }
}

// ========================================================================= //
// Fill helpers — 9×9 local-frame matrices                                    //
// ========================================================================= //

inline void fill_S_local(Eigen::Matrix<double,9,9>& Smat, const LocalS& S,
                         bool kp, bool lp, bool kd, bool ld) {
    Smat.setZero();
    Smat(0,0) = S.ss;
    if (kp) Smat(1,0) = S.ps;
    if (kd) Smat(4,0) = S.ds;
    if (lp) {
        Smat(0,1) = S.sp;
        if (kp) { Smat(1,1)=S.pp; Smat(2,2)=S.pipi; Smat(3,3)=S.pipi; }
        if (kd) { Smat(4,1)=S.dp; Smat(5,2)=S.dppi; Smat(6,3)=S.dppi; }
    }
    if (ld) {
        Smat(0,4) = S.sd;
        if (kp) { Smat(1,4)=S.pd; Smat(2,5)=S.pdpi; Smat(3,6)=S.pdpi; }
        if (kd) { Smat(4,4)=S.dd; Smat(5,5)=S.ddpi; Smat(6,6)=S.ddpi; Smat(7,7)=S.dddel; Smat(8,8)=S.dddel; }
    }
}

inline void fill_M_local(Eigen::Matrix<double,9,9>& Mmat, const Lmunu& M,
                         bool kp, bool lp, bool kd, bool ld) {
    Mmat.setZero();
    Mmat(0,0)=M.ss; if(kp)Mmat(1,0)=M.ps; if(kd)Mmat(4,0)=M.ds;
    if(lp){Mmat(0,1)=M.sp; if(kp){Mmat(1,1)=M.pp;Mmat(2,2)=M.pipi;Mmat(3,3)=M.pipi;} if(kd){Mmat(4,1)=M.dp;Mmat(5,2)=M.dppi;Mmat(6,3)=M.dppi;}}
    if(ld){Mmat(0,4)=M.sd; if(kp){Mmat(1,4)=M.pd;Mmat(2,5)=M.pdpi;Mmat(3,6)=M.pdpi;} if(kd){Mmat(4,4)=M.dd;Mmat(5,5)=M.ddpi;Mmat(6,6)=M.ddpi;Mmat(7,7)=M.dddel;Mmat(8,8)=M.dddel;}}
}

inline void fill_gamma_local(Eigen::Matrix<double,9,9>& Gmat,
                             const GammaCache& gam_k, const GammaCache& gam_l,
                             bool kp, bool lp, bool kd, bool ld) {
    Gmat.setZero();
    if (kp) { Gmat(1,0)=gam_k.ps; Gmat(2,0)=gam_k.ps; Gmat(3,0)=gam_k.ps; }
    if (kd) { Gmat(4,0)=gam_k.ds; Gmat(5,0)=gam_k.ds; Gmat(6,0)=gam_k.ds; Gmat(7,0)=gam_k.ds; Gmat(8,0)=gam_k.ds; }
    if (lp) {
        Gmat(0,1)=gam_l.ps; Gmat(0,2)=gam_l.ps; Gmat(0,3)=gam_l.ps;
        if (kp) {
            for (int r=1; r<=3; ++r) for (int c=1; c<=3; ++c) Gmat(r,c) = gam_k.pp;
        }
        if (kd) {
            for (int r=4; r<=8; ++r) for (int c=1; c<=3; ++c) Gmat(r,c) = gam_k.dp;
        }
    }
    if (ld) {
        for (int c=4; c<=8; ++c) Gmat(0,c) = gam_l.ds;
        if (kp) {
            for (int r=1; r<=3; ++r) for (int c=4; c<=8; ++c) Gmat(r,c) = gam_k.pd;
        }
        if (kd) {
            for (int r=4; r<=8; ++r) for (int c=4; c<=8; ++c) Gmat(r,c) = gam_k.dd;
        }
    }
}

// ========================================================================= //
// HORTH derivative (horth.f GRAD)                                            //
// ========================================================================= //

inline void horth_deriv(CoreH& hk, CoreH& hl, CoreH& dhk, CoreH& dhl,
                        double vk, double vl,
                        const LocalS& S, const LocalS& dS,
                        const Eigen::Matrix<double,9,9>& M,
                        const Eigen::Matrix<double,9,9>& dM,
                        bool kp, bool lp, bool kd, bool ld) {
    auto apply = [&](double s, double ds, double m, double dm, double& hk_attr, double& hl_attr,
                      double& dhk_attr, double& dhl_attr) {
        hk_attr -= vk * s * m;
        hl_attr -= vl * s * m;
        dhk_attr -= vk * (ds * m + s * dm);
        dhl_attr -= vl * (ds * m + s * dm);
    };
    apply(S.ss,dS.ss,M(0,0),dM(0,0),hk.HSS,hl.HSS,dhk.HSS,dhl.HSS);
    if (lp) apply(S.sp,dS.sp,M(0,1),dM(0,1),hk.HSS,hl.HPS,dhk.HSS,dhl.HPS);
    if (kp) apply(S.ps,dS.ps,M(1,0),dM(1,0),hk.HPS,hl.HSS,dhk.HPS,dhl.HSS);
    if (kp && lp) {
        apply(S.pp,dS.pp,M(1,1),dM(1,1),hk.HPS,hl.HPS,dhk.HPS,dhl.HPS);
        double s=S.pipi,ds=dS.pipi,m=M(2,2),dm=dM(2,2);
        hk.HPP-=vk*s*m; hl.HPP-=vl*s*m; dhk.HPP-=vk*(ds*m+s*dm); dhl.HPP-=vl*(ds*m+s*dm);
    }
    if (ld) apply(S.sd,dS.sd,M(0,4),dM(0,4),hk.HSS,hl.HDS,dhk.HSS,dhl.HDS);
    if (kd) apply(S.ds,dS.ds,M(4,0),dM(4,0),hk.HDS,hl.HSS,dhk.HDS,dhl.HSS);
    if (kp && ld) {
        apply(S.pd,dS.pd,M(1,4),dM(1,4),hk.HPS,hl.HDS,dhk.HPS,dhl.HDS);
        double s=S.pdpi,ds=dS.pdpi,m=M(2,5),dm=dM(2,5);
        hk.HPP-=vk*s*m; hl.HDP-=vl*s*m; dhk.HPP-=vk*(ds*m+s*dm); dhl.HDP-=vl*(ds*m+s*dm);
    }
    if (kd && lp) {
        apply(S.dp,dS.dp,M(4,1),dM(4,1),hk.HDS,hl.HPS,dhk.HDS,dhl.HPS);
        double s=S.dppi,ds=dS.dppi,m=M(5,2),dm=dM(5,2);
        hk.HDP-=vk*s*m; hl.HPP-=vl*s*m; dhk.HDP-=vk*(ds*m+s*dm); dhl.HPP-=vl*(ds*m+s*dm);
    }
    if (kd && ld) {
        apply(S.dd,dS.dd,M(4,4),dM(4,4),hk.HDS,hl.HDS,dhk.HDS,dhl.HDS);
        {double s=S.ddpi,ds=dS.ddpi,m=M(5,5),dm=dM(5,5);hk.HDP-=vk*s*m;hl.HDP-=vl*s*m;dhk.HDP-=vk*(ds*m+s*dm);dhl.HDP-=vl*(ds*m+s*dm);}
        {double s=S.dddel,ds=dS.dddel,m=M(7,7),dm=dM(7,7);hk.HDD-=vk*s*m;hl.HDD-=vl*s*m;dhk.HDD-=vk*(ds*m+s*dm);dhl.HDD-=vl*(ds*m+s*dm);}
    }
}

// ========================================================================= //
// DELTAH derivative (deltah.f GRAD)                                         //
// ========================================================================= //

inline void deltah_deriv(int zk, int zl, double R,
                         const LocalS& S, const LocalS& dS,
                         const Eigen::Matrix<double,9,9>& M,
                         const Eigen::Matrix<double,9,9>& dM,
                         double shk_ss, double shk_ps, double shk_pp,
                         double shk_ds, double shk_dp, double shk_dd,
                         double shl_ss, double shl_ps, double shl_pp,
                         double shl_ds, double shl_dp, double shl_dd,
                         double dshk_ss, double dshk_ps, double dshk_pp,
                         double dshk_ds, double dshk_dp, double dshk_dd,
                         double dshl_ss, double dshl_ps, double dshl_pp,
                         double dshl_ds, double dshl_dp, double dshl_dd,
                         Eigen::Matrix<double,9,9>& core,
                         Eigen::Matrix<double,9,9>& dcore,
                         const MsindoParameterSet& p,
                         bool kp, bool lp, bool kd, bool ld) {
    double AL_kl = p.AL_val(zk, zl);
    double AL_lk = p.AL_val(zl, zk);
    double fack = 1.0 - std::exp(-AL_kl * R);
    double facl = 1.0 - std::exp(-AL_lk * R);
    double dfack = AL_kl * std::exp(-AL_kl * R);
    double dfacl = AL_lk * std::exp(-AL_lk * R);

    auto reson = [&](double kk_k, double kk_l, double s, double ds,
                      double sh_k, double sh_l, double dsh_k, double dsh_l,
                      double m, double dm, int i, int j) {
        double pref = 0.25 * (kk_k + kk_l);
        core(i,j) = pref * s * (fack * sh_k + facl * sh_l) + m;
        dcore(i,j) = pref * (ds * fack * sh_k + s * dfack * sh_k + s * fack * dsh_k +
                              ds * facl * sh_l + s * dfacl * sh_l + s * facl * dsh_l) + dm;
    };

    double KSSk=p.get(p.KSS,zk), KSSl=p.get(p.KSS,zl);
    double KPSk=p.get(p.KPS,zk), KPSl=p.get(p.KPS,zl);
    double KPPk=p.get(p.KPP,zk), KPPl=p.get(p.KPP,zl);
    double KDSk=p.get(p.KDS,zk), KDSl=p.get(p.KDS,zl);
    double KDPk=p.get(p.KDP,zk), KDPl=p.get(p.KDP,zl);
    double KDDk=p.get(p.KDD,zk), KDDl=p.get(p.KDD,zl);

    reson(KSSk,KSSl,S.ss,dS.ss,shk_ss,shl_ss,dshk_ss,dshl_ss,M(0,0),dM(0,0),0,0);
    if(kp) reson(KPSk,KSSl,S.ps,dS.ps,shk_ps,shl_ss,dshk_ps,dshl_ss,M(1,0),dM(1,0),1,0);
    if(lp) reson(KSSk,KPSl,S.sp,dS.sp,shk_ss,shl_ps,dshk_ss,dshl_ps,M(0,1),dM(0,1),0,1);
    if(kp&&lp){
        reson(KPSk,KPSl,S.pp,dS.pp,shk_ps,shl_ps,dshk_ps,dshl_ps,M(1,1),dM(1,1),1,1);
        reson(KPPk,KPPl,S.pipi,dS.pipi,shk_pp,shl_pp,dshk_pp,dshl_pp,M(2,2),dM(2,2),2,2);
        core(3,3)=core(2,2); dcore(3,3)=dcore(2,2);
    }
    if(kd) reson(KDSk,KSSl,S.ds,dS.ds,shk_ds,shl_ss,dshk_ds,dshl_ss,M(4,0),dM(4,0),4,0);
    if(ld) reson(KSSk,KDSl,S.sd,dS.sd,shk_ss,shl_ds,dshk_ss,dshl_ds,M(0,4),dM(0,4),0,4);
    if(kd&&lp){
        reson(KDSk,KPSl,S.dp,dS.dp,shk_ds,shl_ps,dshk_ds,dshl_ps,M(4,1),dM(4,1),4,1);
        reson(KDPk,KPPl,S.dppi,dS.dppi,shk_dp,shl_pp,dshk_dp,dshl_pp,M(5,2),dM(5,2),5,2);
        core(6,3)=core(5,2); dcore(6,3)=dcore(5,2);
    }
    if(kp&&ld){
        reson(KPSk,KDSl,S.pd,dS.pd,shk_ps,shl_ds,dshk_ps,dshl_ds,M(1,4),dM(1,4),1,4);
        reson(KPPk,KDPl,S.pdpi,dS.pdpi,shk_pp,shl_dp,dshk_pp,dshl_dp,M(2,5),dM(2,5),2,5);
        core(3,6)=core(2,5); dcore(3,6)=dcore(2,5);
    }
    if(kd&&ld){
        reson(KDSk,KDSl,S.dd,dS.dd,shk_ds,shl_ds,dshk_ds,dshl_ds,M(4,4),dM(4,4),4,4);
        reson(KDPk,KDPl,S.ddpi,dS.ddpi,shk_dp,shl_dp,dshk_dp,dshl_dp,M(5,5),dM(5,5),5,5);
        core(6,6)=core(5,5); dcore(6,6)=dcore(5,5);
        reson(KDDk,KDDl,S.dddel,dS.dddel,shk_dd,shl_dd,dshk_dd,dshl_dd,M(7,7),dM(7,7),7,7);
        core(8,8)=core(7,7); dcore(8,8)=dcore(7,7);
    }
}

// ========================================================================= //
// d_trans_driver — T, TDT, TDP from d_trans.f                                //
// ========================================================================= //

inline void d_trans_driver(const std::array<double,3>& E,
                           double T[9][9], double TDT[9][9], double TDP[9][9]) {
    double cost = E[2];
    double sint, cosp, sinp;
    const double tol = 1e-12;
    if (std::abs(cost) >= 1.0 - tol) { sint = 0; cosp = 1; sinp = 0; }
    else if (std::abs(cost) <= tol) { sint = 1; cosp = E[0]; sinp = E[1]; }
    else { sint = std::sqrt(1.0 - cost*cost); cosp = E[0]/sint; sinp = E[1]/sint; }

    double sqrt3 = std::sqrt(3.0);
    double cos2t = cost*cost - sint*sint;
    double sin2t = 2*sint*cost;
    double cos2p = cosp*cosp - sinp*sinp;
    double sin2p = 2*sinp*cosp;

    // Zero all
    for(int i=0;i<9;++i) for(int j=0;j<9;++j) T[i][j]=TDT[i][j]=TDP[i][j]=0;
    T[0][0]=1;
    // p-block
    T[1][1]=sint*cosp; T[2][1]=sint*sinp; T[3][1]=cost;
    T[1][2]=cost*cosp; T[2][2]=cost*sinp; T[3][2]=-sint;
    T[1][3]=-sinp; T[2][3]=cosp;
    // d-block
    T[4][4]=(3*cost*cost-1)*0.5; T[5][4]=sqrt3*sin2t*cosp*0.5; T[6][4]=sqrt3*sin2t*sinp*0.5;
    T[7][4]=sqrt3*sint*sint*cos2p*0.5; T[8][4]=sqrt3*sint*sint*sin2p*0.5;
    T[4][5]=-sqrt3*sin2t*0.5; T[5][5]=cos2t*cosp; T[6][5]=cos2t*sinp;
    T[7][5]=sin2t*cos2p*0.5; T[8][5]=sin2t*sin2p*0.5;
    T[5][6]=-cost*sinp; T[6][6]=cost*cosp; T[7][6]=-sint*sin2p; T[8][6]=sint*cos2p;
    T[4][7]=sqrt3*sint*sint*0.5; T[5][7]=-sin2t*cosp*0.5; T[6][7]=-sin2t*sinp*0.5;
    T[7][7]=(1+cost*cost)*cos2p*0.5; T[8][7]=(1+cost*cost)*sin2p*0.5;
    T[5][8]=sint*sinp; T[6][8]=-sint*cosp; T[7][8]=-cost*sin2p; T[8][8]=cost*cos2p;

    double dcos2t=-2*sin2t, dsin2t=2*cos2t, dcos2p=-2*sin2p, dsin2p=2*cos2p;
    // TDT
    TDT[1][1]=cost*cosp; TDT[2][1]=cost*sinp; TDT[3][1]=-sint;
    TDT[1][2]=-sint*cosp; TDT[2][2]=-sint*sinp; TDT[3][2]=-cost;
    TDT[4][4]=-1.5*sin2t; TDT[5][4]=sqrt3*dsin2t*cosp*0.5; TDT[6][4]=sqrt3*dsin2t*sinp*0.5;
    TDT[7][4]=sqrt3*sin2t*cos2p*0.5; TDT[8][4]=sqrt3*sin2t*sin2p*0.5;
    TDT[4][5]=-sqrt3*dsin2t*0.5; TDT[5][5]=dcos2t*cosp; TDT[6][5]=dcos2t*sinp;
    TDT[7][5]=dsin2t*cos2p*0.5; TDT[8][5]=dsin2t*sin2p*0.5;
    TDT[5][6]=sint*sinp; TDT[6][6]=-sint*cosp; TDT[7][6]=-cost*sin2p; TDT[8][6]=cost*cos2p;
    TDT[4][7]=sqrt3*sin2t*0.5; TDT[5][7]=-dsin2t*cosp*0.5; TDT[6][7]=-dsin2t*sinp*0.5;
    TDT[7][7]=-sin2t*cos2p*0.5; TDT[8][7]=-sin2t*sin2p*0.5;
    TDT[5][8]=cost*sinp; TDT[6][8]=-cost*cosp; TDT[7][8]=sint*sin2p; TDT[8][8]=-sint*cos2p;
    // TDP
    TDP[1][1]=-sint*sinp; TDP[2][1]=sint*cosp;
    TDP[1][2]=-cost*sinp; TDP[2][2]=cost*cosp; TDP[1][3]=-cosp; TDP[2][3]=-sinp;
    TDP[5][4]=-sqrt3*sin2t*sinp*0.5; TDP[6][4]=sqrt3*sin2t*cosp*0.5;
    TDP[7][4]=sqrt3*sint*sint*dcos2p*0.5; TDP[8][4]=sqrt3*sint*sint*dsin2p*0.5;
    TDP[5][5]=-cos2t*sinp; TDP[6][5]=cos2t*cosp;
    TDP[7][5]=sin2t*dcos2p*0.5; TDP[8][5]=sin2t*dsin2p*0.5;
    TDP[5][6]=-cost*cosp; TDP[6][6]=-cost*sinp;
    TDP[7][6]=-sint*dsin2p; TDP[8][6]=sint*dcos2p;
    TDP[5][7]=sin2t*sinp*0.5; TDP[6][7]=-sin2t*cosp*0.5;
    TDP[7][7]=(1+cost*cost)*dcos2p*0.5; TDP[8][7]=(1+cost*cost)*dsin2p*0.5;
    TDP[5][8]=sint*cosp; TDP[6][8]=sint*sinp;
    TDP[7][8]=-cost*dsin2p; TDP[8][8]=cost*dcos2p;
}

// ========================================================================= //
// Rotate derivatives to global frame                                        //
// ========================================================================= //

inline void rotate_derivatives_to_global(
    const Eigen::Matrix<double,9,9>& HK1_local, const Eigen::Matrix<double,9,9>& HL1_local,
    const Eigen::Matrix<double,9,9>& core_local,
    const Eigen::Matrix<double,9,9>& dHK1_local, const Eigen::Matrix<double,9,9>& dHL1_local,
    const Eigen::Matrix<double,9,9>& dcore_local,
    const double T[9][9], const double TDT[9][9], const double TDP[9][9],
    Eigen::Matrix<double,9,9>& HK1_DR, Eigen::Matrix<double,9,9>& HL1_DR,
    Eigen::Matrix<double,9,9>& HKL2_DR,
    Eigen::Matrix<double,9,9>& HK1_DT, Eigen::Matrix<double,9,9>& HL1_DT,
    Eigen::Matrix<double,9,9>& HKL2_DT,
    Eigen::Matrix<double,9,9>& HK1_DP, Eigen::Matrix<double,9,9>& HL1_DP,
    Eigen::Matrix<double,9,9>& HKL2_DP) {
    auto Tmat = [&]() { Eigen::Matrix<double,9,9> m; for(int i=0;i<9;++i)for(int j=0;j<9;++j)m(i,j)=T[i][j]; return m; };
    auto TDTmat = [&]() { Eigen::Matrix<double,9,9> m; for(int i=0;i<9;++i)for(int j=0;j<9;++j)m(i,j)=TDT[i][j]; return m; };
    auto TDPmat = [&]() { Eigen::Matrix<double,9,9> m; for(int i=0;i<9;++i)for(int j=0;j<9;++j)m(i,j)=TDP[i][j]; return m; };
    Eigen::Matrix<double,9,9> Tm = Tmat(), TDTm = TDTmat(), TDPm = TDPmat();

    // R-derivatives: simple rotation
    HK1_DR = Tm * dHK1_local * Tm.transpose();
    HL1_DR = Tm * dHL1_local * Tm.transpose();
    HKL2_DR = Tm * dcore_local * Tm.transpose();
    // θ-derivatives: product rule
    HK1_DT = TDTm * HK1_local * Tm.transpose() + Tm * HK1_local * TDTm.transpose();
    HL1_DT = TDTm * HL1_local * Tm.transpose() + Tm * HL1_local * TDTm.transpose();
    HKL2_DT = TDTm * core_local * Tm.transpose() + Tm * core_local * TDTm.transpose();
    // φ-derivatives
    HK1_DP = TDPm * HK1_local * Tm.transpose() + Tm * HK1_local * TDPm.transpose();
    HL1_DP = TDPm * HL1_local * Tm.transpose() + Tm * HL1_local * TDPm.transpose();
    HKL2_DP = TDPm * core_local * Tm.transpose() + Tm * core_local * TDPm.transpose();
}

// ========================================================================= //
// pair_blocks_deriv — per-pair integral derivatives (INTDRV GRAD port)      //
// ========================================================================= //

inline PairBlocksDeriv pair_blocks_deriv(int zk, int zl,
                                          const std::array<double,3>& rk,
                                          const std::array<double,3>& rl,
                                          const MsindoParameterSet& p) {
    PairBlocksDeriv result;
    std::array<double,3> d = {rl[0]-rk[0], rl[1]-rk[1], rl[2]-rk[2]};
    double R = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    result.R = R;
    result.E = {d[0]/R, d[1]/R, d[2]/R};

    // Build standard PairBlocks (delegates to existing engine)
    result.pb = pair_blocks(zk, zl, rk, rl, p);

    // Determine basis flags
    int nk = MsindoParameterSet::n_principal(zk), npk = MsindoParameterSet::n_p_principal(zk);
    int nkd = MsindoParameterSet::n_d_principal(zk);
    int nl = MsindoParameterSet::n_principal(zl), npl = MsindoParameterSet::n_p_principal(zl);
    int nld = MsindoParameterSet::n_d_principal(zl);
    double zsk = p.get(p.MUS,zk), zpk = p.get(p.MUP,zk), zdk = p.get(p.MUD,zk);
    double zsl = p.get(p.MUS,zl), zpl = p.get(p.MUP,zl), zdl = p.get(p.MUD,zl);
    bool kp = (p.n_basis(zk) >= 4), lp = (p.n_basis(zl) >= 4);
    bool kd = (p.n_basis(zk) >= 9), ld = (p.n_basis(zl) >= 9);

    // --- Overlaps S and dS/dR ---
    LocalS S{}, dS{};
    S.ss = s2int(nk,0,0,zsk,nl,0,0,zsl,R); dS.ss = ds2int(nk,0,0,zsk,nl,0,0,zsl,R);
    if(kp){S.ps=s2int(npk,1,0,zpk,nl,0,0,zsl,R);dS.ps=ds2int(npk,1,0,zpk,nl,0,0,zsl,R);}
    if(lp){S.sp=s2int(nk,0,0,zsk,npl,1,0,zpl,R);dS.sp=ds2int(nk,0,0,zsk,npl,1,0,zpl,R);}
    if(kp&&lp){S.pp=s2int(npk,1,0,zpk,npl,1,0,zpl,R);S.pipi=s2int(npk,1,1,zpk,npl,1,1,zpl,R);dS.pp=ds2int(npk,1,0,zpk,npl,1,0,zpl,R);dS.pipi=ds2int(npk,1,1,zpk,npl,1,1,zpl,R);}
    if(kd){S.ds=s2int(nkd,2,0,zdk,nl,0,0,zsl,R);dS.ds=ds2int(nkd,2,0,zdk,nl,0,0,zsl,R);if(lp){S.dp=s2int(nkd,2,0,zdk,npl,1,0,zpl,R);S.dppi=s2int(nkd,2,1,zdk,npl,1,1,zpl,R);dS.dp=ds2int(nkd,2,0,zdk,npl,1,0,zpl,R);dS.dppi=ds2int(nkd,2,1,zdk,npl,1,1,zpl,R);}}
    if(ld){S.sd=s2int(nk,0,0,zsk,nld,2,0,zdl,R);dS.sd=ds2int(nk,0,0,zsk,nld,2,0,zdl,R);if(kp){S.pd=s2int(npk,1,0,zpk,nld,2,0,zdl,R);S.pdpi=s2int(npk,1,1,zpk,nld,2,1,zdl,R);dS.pd=ds2int(npk,1,0,zpk,nld,2,0,zdl,R);dS.pdpi=ds2int(npk,1,1,zpk,nld,2,1,zdl,R);}}
    if(kd&&ld){S.dd=s2int(nkd,2,0,zdk,nld,2,0,zdl,R);S.ddpi=s2int(nkd,2,1,zdk,nld,2,1,zdl,R);S.dddel=s2int(nkd,2,2,zdk,nld,2,2,zdl,R);dS.dd=ds2int(nkd,2,0,zdk,nld,2,0,zdl,R);dS.ddpi=ds2int(nkd,2,1,zdk,nld,2,1,zdl,R);dS.dddel=ds2int(nkd,2,2,zdk,nld,2,2,zdl,R);}

    // --- Coulomb gammas ---
    auto c2 = [&](int n1, double e1, int n2, double e2){return c2int(n1,0,0,e1,n2,0,0,e2,R);};
    auto dc2 = [&](int n1, double e1, int n2, double e2){return dc2int(n1,0,0,e1,n2,0,0,e2,R);};
    GammaCache gam_k{}, gam_l{}, dgam_k{}, dgam_l{};
    if(kp){gam_k.ps=c2(npk,zpk,nl,zsl);dgam_k.ps=dc2(npk,zpk,nl,zsl);}
    if(kp&&lp){gam_k.pp=c2(npk,zpk,npl,zpl);dgam_k.pp=dc2(npk,zpk,npl,zpl);}
    if(kp&&ld){gam_k.pd=c2(npk,zpk,nld,zdl);dgam_k.pd=dc2(npk,zpk,nld,zdl);}
    if(kd){gam_k.ds=c2(nkd,zdk,nl,zsl);dgam_k.ds=dc2(nkd,zdk,nl,zsl);}
    if(kd&&lp){gam_k.dp=c2(nkd,zdk,npl,zpl);dgam_k.dp=dc2(nkd,zdk,npl,zpl);}
    if(kd&&ld){gam_k.dd=c2(nkd,zdk,nld,zdl);dgam_k.dd=dc2(nkd,zdk,nld,zdl);}
    if(lp){gam_l.ps=c2(npl,zpl,nk,zsk);dgam_l.ps=dc2(npl,zpl,nk,zsk);}
    if(lp&&kp){gam_l.pp=c2(npl,zpl,npk,zpk);dgam_l.pp=dc2(npl,zpl,npk,zpk);}
    if(lp&&kd){gam_l.pd=c2(npl,zpl,nkd,zdk);dgam_l.pd=dc2(npl,zpl,nkd,zdk);}
    if(ld){gam_l.ds=c2(nld,zdl,nk,zsk);dgam_l.ds=dc2(nld,zdl,nk,zsk);}
    if(ld&&kp){gam_l.dp=c2(nld,zdl,npk,zpk);dgam_l.dp=dc2(nld,zdl,npk,zpk);}
    if(ld&&kd){gam_l.dd=c2(nld,zdl,nkd,zdk);dgam_l.dd=dc2(nld,zdl,nkd,zdk);}

    // --- V2CORE + VCORR ---
    CoreH hk = v2core_one_side(zk, zl, R, gam_k, p);
    CoreH hl = v2core_one_side(zl, zk, R, gam_l, p);
    CoreH dhk = v2core_one_side_deriv(zk, zl, R, gam_k, dgam_k, p);
    CoreH dhl = v2core_one_side_deriv(zl, zk, R, gam_l, dgam_l, p);

    // --- ENEG ---
    auto ek = eneg(zk, p), el = eneg(zl, p);
    double ek_ss = ek[0], ek_ps = ek[1], ek_ds = ek[2];
    double el_ss = el[0], el_ps = el[1], el_ds = el[2];

    // --- SH values (pre-HORTH, for DELTAH) ---
    double shk_ss = ek_ss + hk.HSS, shk_ps = ek_ps + hk.HPS, shk_pp = ek_ps + hk.HPP;
    double shk_ds = ek_ds + hk.HDS, shk_dp = ek_ds + hk.HDP, shk_dd = ek_ds + hk.HDD;
    double shl_ss = el_ss + hl.HSS, shl_ps = el_ps + hl.HPS, shl_pp = el_ps + hl.HPP;
    double shl_ds = el_ds + hl.HDS, shl_dp = el_ds + hl.HDP, shl_dd = el_ds + hl.HDD;
    double dshk_ss = dhk.HSS, dshk_ps = dhk.HPS, dshk_pp = dhk.HPP;
    double dshk_ds = dhk.HDS, dshk_dp = dhk.HDP, dshk_dd = dhk.HDD;
    double dshl_ss = dhl.HSS, dshl_ps = dhl.HPS, dshl_pp = dhl.HPP;
    double dshl_ds = dhl.HDS, dshl_dp = dhl.HDP, dshl_dd = dhl.HDD;

    // --- LMUNU M and dM/dR ---
    Lmunu M = lmunu_local(zk, zl, R, S, p);
    Eigen::Matrix<double,9,9> Mmat, dMmat;
    fill_M_local(Mmat, M, kp, lp, kd, ld);
    lmunu_local_deriv(zk, zl, R, S, dS, dMmat, p);

    // --- HORTH with derivatives ---
    double vk = (ld?0.5:(lp?0.75:1.0)), vl = (kd?0.5:(kp?0.75:1.0));  // _vfak
    CoreH hk_m=hk, hl_m=hl, dhk_m=dhk, dhl_m=dhl;
    horth_deriv(hk_m, hl_m, dhk_m, dhl_m, vk, vl, S, dS, Mmat, dMmat, kp, lp, kd, ld);

    // --- DELTAH ---
    Eigen::Matrix<double,9,9> core, dcore;
    core.setZero(); dcore.setZero();
    deltah_deriv(zk, zl, R, S, dS, Mmat, dMmat,
                 shk_ss, shk_ps, shk_pp, shk_ds, shk_dp, shk_dd,
                 shl_ss, shl_ps, shl_pp, shl_ds, shl_dp, shl_dd,
                 dshk_ss, dshk_ps, dshk_pp, dshk_ds, dshk_dp, dshk_dd,
                 dshl_ss, dshl_ps, dshl_pp, dshl_ds, dshl_dp, dshl_dd,
                 core, dcore, p, kp, lp, kd, ld);

    // --- Build local-frame matrices ---
    fill_S_local(result.S_local, S, kp, lp, kd, ld);
    fill_S_local(result.dS_local, dS, kp, lp, kd, ld);
    double gam_ss = c2(nk,zsk,nl,zsl), dgam_ss = dc2(nk,zsk,nl,zsl);
    fill_gamma_local(result.gamma_local, gam_k, gam_l, kp, lp, kd, ld);
    fill_gamma_local(result.d_gamma_local, dgam_k, dgam_l, kp, lp, kd, ld);
    result.gamma_local(0,0) = gam_ss;   // set ss gamma AFTER fill (which zeros everything)
    result.d_gamma_local(0,0) = dgam_ss;
    result.M_local = Mmat; result.dM_local = dMmat;

    auto make_diag = [](double a,double b,double c,double d,double e,double f){
        Eigen::Matrix<double,9,9> m; m.setZero();
        m(0,0)=a;m(1,1)=b;m(2,2)=c;m(3,3)=c;m(4,4)=d;m(5,5)=e;m(6,6)=e;m(7,7)=f;m(8,8)=f;
        return m;
    };
    Eigen::Matrix<double,9,9> HK1l = make_diag(hk_m.HSS,hk_m.HPS,hk_m.HPP,hk_m.HDS,hk_m.HDP,hk_m.HDD);
    Eigen::Matrix<double,9,9> HL1l = make_diag(hl_m.HSS,hl_m.HPS,hl_m.HPP,hl_m.HDS,hl_m.HDP,hl_m.HDD);
    result.dHK1_local = make_diag(dhk_m.HSS,dhk_m.HPS,dhk_m.HPP,dhk_m.HDS,dhk_m.HDP,dhk_m.HDD);
    result.dHL1_local = make_diag(dhl_m.HSS,dhl_m.HPS,dhl_m.HPP,dhl_m.HDS,dhl_m.HDP,dhl_m.HDD);
    result.dHKL2_local = dcore;

    // --- Phase 3: rotate to global ---
    double T[9][9], TDT[9][9], TDP[9][9];
    d_trans_driver(result.E, T, TDT, TDP);
    rotate_derivatives_to_global(HK1l, HL1l, core,
                                  result.dHK1_local, result.dHL1_local, result.dHKL2_local,
                                  T, TDT, TDP,
                                  result.HK1_DR, result.HL1_DR, result.HKL2_DR,
                                  result.HK1_DT, result.HL1_DT, result.HKL2_DT,
                                  result.HK1_DP, result.HL1_DP, result.HKL2_DP);

    return result;
}

// ========================================================================= //
// msindo_gradient_analytic — DEDXYZK/DEDXYZL port (Phase 4)                 //
// ========================================================================= //

inline std::vector<std::array<double,3>> msindo_gradient_analytic_core(
    const std::vector<int>& Z,
    const std::vector<std::array<double,3>>& coords_angstrom,
    const MsindoParameterSet& p,
    int max_iter = 200, double conv_tol = 1e-10,
    int charge = 0) {

    int natom = (int)Z.size();

    // Convert to bohr
    std::vector<Eigen::Vector3d> C(natom);
    for (int i = 0; i < natom; ++i)
        C[i] = {coords_angstrom[i][0]*ANGSTROM_TO_BOHR,
                coords_angstrom[i][1]*ANGSTROM_TO_BOHR,
                coords_angstrom[i][2]*ANGSTROM_TO_BOHR};

    // Build atom blocks
    std::vector<std::array<int,2>> blocks(natom);
    int nsto = 0;
    for (int i = 0; i < natom; ++i) {
        int nb = p.n_basis(Z[i]);
        blocks[i] = {nsto, nsto + nb};
        nsto += nb;
    }

    // Compute nocc
    int nelec = 0;
    for (int z : Z) {
        auto en = eneg(z, p);
        double nval = p.get(p.LS,z)+p.get(p.MP,z)+p.get(p.ND,z);
        nelec += (int)nval;
    }
    nelec -= charge;
    if (nelec < 0 || nelec % 2 != 0) return {};
    int nocc = nelec / 2;

    // SCF
    Eigen::MatrixXd H, G;
    build_core_and_gamma(Z, C, blocks, nsto, p, false, H, G);
    bool proactive_probe = true;
    for (int z : Z) {
        if (!is_msindo_root_probe_element(z)) {
            proactive_probe = false;
            break;
        }
    }
    // Outside the validated light-element selector scope, preserve the
    // pre-existing strict-DIIS gradient route.  In particular, never feed the
    // heavy energy-only WICHT density to a variational analytic derivative.
    MsindoResult res = proactive_probe
        ? scf_rhf_molecular_driver(
            H, G, blocks, Z, nocc, p, max_iter, conv_tol)
        : scf_rhf_driver(
            H, G, blocks, Z, nocc, p, max_iter, conv_tol, nullptr);
    if (!res.converged) return {};
    Eigen::MatrixXd P = res.density;

    std::vector<std::array<double,3>> grad(natom, {0,0,0});

    for (int k = 0; k < natom; ++k) {
        for (int l = k + 1; l < natom; ++l) {
            std::array<double,3> rk = {C[k].x(), C[k].y(), C[k].z()};
            std::array<double,3> rl = {C[l].x(), C[l].y(), C[l].z()};
            double dx = rl[0]-rk[0], dy = rl[1]-rk[1], dz = rl[2]-rk[2];
            double R = std::sqrt(dx*dx+dy*dy+dz*dz);
            if (R < 1e-12) continue;
            double Ex = dx/R, Ey = dy/R, Ez = dz/R;

            PairBlocksDeriv pd = pair_blocks_deriv(Z[k], Z[l], rk, rl, p);

            // Chain rule factors
            double cost = Ez, sint = std::sqrt(std::max(0.0, 1.0-cost*cost));
            double cosphi=1, sinphi=0, dphidx=0, dphidy=0, dphidz=0;
            if (sint > 1e-12) {
                cosphi = Ex/sint; sinphi = Ey/sint;
                double c1 = sint*sint*R;
                dphidx = -Ey/c1; dphidy = Ex/c1;
            }
            double edt1 = cost*cosphi, edt2 = cost*sinphi, edt3 = -sint;
            double drdx=Ex, drdy=Ey, drdz=Ez;
            double dthx=edt1/R, dthy=edt2/R, dthz=edt3/R;

            int nk = blocks[k][1]-blocks[k][0], nl = blocks[l][1]-blocks[l][0];
            auto P_kk = P.block(blocks[k][0],blocks[k][0],nk,nk);
            auto P_ll = P.block(blocks[l][0],blocks[l][0],nl,nl);
            auto P_kl = P.block(blocks[k][0],blocks[l][0],nk,nl);

            double gx=0, gy=0, gz=0;
            // K-diag + off-diag
            for (int i=0;i<nk;++i){
                gx += P_kk(i,i)*(pd.HK1_DR(i,i)*drdx+pd.HK1_DT(i,i)*dthx+pd.HK1_DP(i,i)*dphidx);
                gy += P_kk(i,i)*(pd.HK1_DR(i,i)*drdy+pd.HK1_DT(i,i)*dthy+pd.HK1_DP(i,i)*dphidy);
                gz += P_kk(i,i)*(pd.HK1_DR(i,i)*drdz+pd.HK1_DT(i,i)*dthz+pd.HK1_DP(i,i)*dphidz);
                for (int j=i+1;j<nk;++j){
                    double Pij2 = 2*P_kk(j,i);
                    gx+=Pij2*(pd.HK1_DR(j,i)*drdx+pd.HK1_DT(j,i)*dthx+pd.HK1_DP(j,i)*dphidx);
                    gy+=Pij2*(pd.HK1_DR(j,i)*drdy+pd.HK1_DT(j,i)*dthy+pd.HK1_DP(j,i)*dphidy);
                    gz+=Pij2*(pd.HK1_DR(j,i)*drdz+pd.HK1_DT(j,i)*dthz+pd.HK1_DP(j,i)*dphidz);
                }
            }
            for (int i=0;i<nl;++i){
                gx += P_ll(i,i)*(pd.HL1_DR(i,i)*drdx+pd.HL1_DT(i,i)*dthx+pd.HL1_DP(i,i)*dphidx);
                gy += P_ll(i,i)*(pd.HL1_DR(i,i)*drdy+pd.HL1_DT(i,i)*dthy+pd.HL1_DP(i,i)*dphidy);
                gz += P_ll(i,i)*(pd.HL1_DR(i,i)*drdz+pd.HL1_DT(i,i)*dthz+pd.HL1_DP(i,i)*dphidz);
                for (int j=i+1;j<nl;++j){
                    double Pij2 = 2*P_ll(j,i);
                    gx+=Pij2*(pd.HL1_DR(j,i)*drdx+pd.HL1_DT(j,i)*dthx+pd.HL1_DP(j,i)*dphidx);
                    gy+=Pij2*(pd.HL1_DR(j,i)*drdy+pd.HL1_DT(j,i)*dthy+pd.HL1_DP(j,i)*dphidy);
                    gz+=Pij2*(pd.HL1_DR(j,i)*drdz+pd.HL1_DT(j,i)*dthz+pd.HL1_DP(j,i)*dphidz);
                }
            }
            // Pair block + gamma (2× from DEDXYZK + DEDXYZL)
            for (int i=0;i<nk;++i){
                double Pii = P_kk(i,i);
                for (int j=0;j<nl;++j){
                    double Pjj = P_ll(j,j), Pij = P_kl(i,j);
                    double Fg = 0.5*pd.d_gamma_local(i,j)*(Pii*Pjj-0.5*Pij*Pij);
                    double Fr = Pij*pd.HKL2_DR(i,j);
                    gx += 2.0*(Fg+Fr)*drdx + 2.0*Pij*pd.HKL2_DT(i,j)*dthx + 2.0*Pij*pd.HKL2_DP(i,j)*dphidx;
                    gy += 2.0*(Fg+Fr)*drdy + 2.0*Pij*pd.HKL2_DT(i,j)*dthy + 2.0*Pij*pd.HKL2_DP(i,j)*dphidy;
                    gz += 2.0*(Fg+Fr)*drdz + 2.0*Pij*pd.HKL2_DT(i,j)*dthz + 2.0*Pij*pd.HKL2_DP(i,j)*dphidz;
                }
            }
            // Nuclear (2× from both routines)
            double czk = p.get(p.LS,Z[k])+p.get(p.MP,Z[k])+p.get(p.ND,Z[k]);
            double czl = p.get(p.LS,Z[l])+p.get(p.MP,Z[l])+p.get(p.ND,Z[l]);
            double dnuc = 2.0*0.5*czk*czl*(-1.0/(R*R));
            gx += dnuc*drdx; gy += dnuc*drdy; gz += dnuc*drdz;

            grad[l][0] += gx; grad[l][1] += gy; grad[l][2] += gz;
            grad[k][0] -= gx; grad[k][1] -= gy; grad[k][2] -= gz;
        }
    }
    return grad;
}

}  // namespace indo
}  // namespace semiempirical
}  // namespace vibeqc
