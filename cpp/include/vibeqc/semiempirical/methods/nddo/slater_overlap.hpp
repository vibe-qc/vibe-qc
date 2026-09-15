// Shared Slater-type orbital overlap integrals for NDDO-family methods.
//
// Used by pm6_fock.cpp, omx_fock.cpp, periodic_pm6.cpp, periodic_omx.cpp.
// All functions are inline to avoid ODR issues when included from multiple
// translation units.

#pragma once

#include <cmath>

#include "vibeqc/semiempirical/methods/indo/msindo_integrals.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// Slater-type s-s overlap: S_ss = (1 + zR + zR²/3) · exp(-zR)
inline double slater_ss(double zR) {
    return (1.0 + zR + zR * zR / 3.0) * std::exp(-zR);
}

// d/d(zR) S_ss = -(zR/3)(1 + zR) * exp(-zR)
inline double slater_ss_deriv(double zR) {
    return -(zR / 3.0) * (1.0 + zR) * std::exp(-zR);
}

// Slater-type s-p overlap: S_sp = zR · (1 + zR/2 + zR²/6) · exp(-zR)
inline double slater_sp(double zR) {
    double r2 = zR * zR;
    return (zR + 0.5 * r2 + r2 * zR / 6.0) * std::exp(-zR);
}

// d/d(zR) S_sp = (1 - zR^2/6) * exp(-zR)
inline double slater_sp_deriv(double zR) {
    return (1.0 - zR * zR / 6.0) * std::exp(-zR);
}

// Slater-type p-p sigma overlap: S_ppσ = (1 + zR + 2zR²/5 + zR³/15) · exp(-zR)
inline double slater_pp_sigma(double zR) {
    double r2 = zR * zR;
    return (1.0 + zR + 2.0 * r2 / 5.0 + r2 * zR / 15.0) * std::exp(-zR);
}

// d/d(zR) S_pp_sigma = -(zR/5)(1 + zR/3) * exp(-zR)
inline double slater_pp_sigma_deriv(double zR) {
    return -(zR / 5.0) * (1.0 + zR / 3.0) * std::exp(-zR);
}

// Slater-type p-p pi overlap: S_ppπ = (1 + zR + zR²/5) · exp(-zR)
inline double slater_pp_pi(double zR) {
    double r2 = zR * zR;
    return (1.0 + zR + r2 / 5.0) * std::exp(-zR);
}

// d/d(zR) S_pp_pi = -(zR/5)(2 + zR) * exp(-zR)
inline double slater_pp_pi_deriv(double zR) {
    return -(zR / 5.0) * (2.0 + zR) * std::exp(-zR);
}

// Direction cosine: dx/R, dy/R, or dz/R depending on axis (0=x, 1=y, 2=z)
inline double slater_dir_cos(int axis, double dx, double dy, double dz, double R) {
    if (R < 1e-12) return 0.0;
    if (axis == 0) return dx / R;
    if (axis == 1) return dy / R;
    return dz / R;
}


// Principal quantum number of the valence s/p shell (= period number).  C/N/O/F
// are n=2, NOT n=1 — the closed-form helpers above are 1s-shaped and are wrong
// for any second-row (or heavier) atom.  Use n-dependent STO integrals instead.
inline int valence_n_sp(int Z) {
    if (Z <= 2)  return 1;
    if (Z <= 10) return 2;
    if (Z <= 18) return 3;
    if (Z <= 36) return 4;
    if (Z <= 54) return 5;
    return 6;
}

// Correct two-centre STO overlap between AO (atom A: principal n_a, type t_a,
// exponents zs_a/zp_a) and AO (atom B: n_b, t_b, zs_b/zp_b), with t: 0=s,
// 1=px, 2=py, 3=pz and dx/dy/dz = pos(A) − pos(B).  Built on the MSINDO STO
// integral kernel (indo::s2int, which is n-dependent and validated against
// numerical integration) plus a Slater–Koster direction-cosine rotation from
// the local (σ/π) frame to the global frame.
//
// NOTE (2026-06-10): commit 65ed3112 replaced this function with an s-s-shaped
// stub to "fix a build break"; msindo_integrals.hpp is header-only, so the
// include cannot break a link, and the stub silently re-broke the PM6/OMx PES
// (collapse below 1.6 bohr on water O–H).  Restored from d364ec02; the PES
// shape is now pinned by tests/test_semiempirical_pes.py — do not stub again.
//
// Sign notes (validated against numerical 3-D STO integration):
//   * s2int's local frame puts atom1 at the origin and atom2 along +z, so its
//     σ reduced overlaps already carry the correct sign (e.g. 2pσ–2pσ < 0).
//   * the p–p channel uses two direction cosines (ca·cb), so the bond-direction
//     convention (dx = A−B vs. s2int's A→B = +z) cancels.
//   * the s–p channel has a *single* cosine, so it needs an explicit minus sign
//     to undo that convention mismatch.
inline double nddo_sto_overlap(
    int n_a, int t_a, double zs_a, double zp_a,
    int n_b, int t_b, double zs_b, double zp_b,
    double dx, double dy, double dz, double R) {
    const bool sA = (t_a == 0), sB = (t_b == 0);
    if (sA && sB)
        return ::vibeqc::semiempirical::indo::s2int(n_a, 0, 0, zs_a,
                                                    n_b, 0, 0, zs_b, R);
    if (!sA && !sB) {
        const int da = t_a - 1, db = t_b - 1;
        const double ca = slater_dir_cos(da, dx, dy, dz, R);
        const double cb = slater_dir_cos(db, dx, dy, dz, R);
        const double ssig = ::vibeqc::semiempirical::indo::s2int(
            n_a, 1, 0, zp_a, n_b, 1, 0, zp_b, R);
        const double spi = ::vibeqc::semiempirical::indo::s2int(
            n_a, 1, 1, zp_a, n_b, 1, 1, zp_b, R);
        const double delta = (da == db) ? 1.0 : 0.0;
        return ca * cb * ssig + (delta - ca * cb) * spi;
    }
    if (sA) {  // A = s, B = p
        const int db = t_b - 1;
        const double ssp = ::vibeqc::semiempirical::indo::s2int(
            n_a, 0, 0, zs_a, n_b, 1, 0, zp_b, R);
        return -slater_dir_cos(db, dx, dy, dz, R) * ssp;
    }
    // A = p, B = s
    const int da = t_a - 1;
    const double sps = ::vibeqc::semiempirical::indo::s2int(
        n_a, 1, 0, zp_a, n_b, 0, 0, zs_b, R);
    return -slater_dir_cos(da, dx, dy, dz, R) * sps;
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
