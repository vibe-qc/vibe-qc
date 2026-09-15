// 1-D background-corrected Ewald kernel for the SECCM adapters.
//
// Replaces the historical truncated ±2-shell bare Coulomb sum
// (indo::_madelung_potential_1d, kept frozen for MSINDO) with the
// converged Parry-type wire Ewald. The bare 1-D image series diverges
// logarithmically, so any finite truncation leaves a kernel whose SCC
// response is nearly singular for even-N polar charge patterns (the
// documented period-2 limit cycle). The converged kernel below removes
// the harmonic divergence with a neutralizing background and is stable
// for every replica count.
//
// Convention (background-corrected periodic potential of a unit charge):
//
//     Phi(d) = sum_n [ 1/|d + nT| - (n != 0 ? 1/|nT| : 0) ]
//
// with the n = 0 self image excluded at d = 0. For neutral cells the
// background subtraction cancels identically and Phi reproduces the
// ordinary lattice sum (the alternating-chain Madelung constant 2 ln 2
// is pinned by a regression test). For charged cells the convention
// fixes the wire's harmonic image-tail divergence; the residual
// on-axis constant only shifts charged-cell energies, never the SCC
// fixed point (a uniform diagonal shift leaves the density unchanged).
//
// Ewald split (alpha = 4/L, period L = |T|, axial coordinate
// z = d.That, in-plane radius rho = |d - z That|):
//
//     Phi(d) = sum_n erfc(alpha r_n)/r_n            (real, |n| <= 8)
//            + (2/L) sum_{m=1}^{M} cos(k_m z) F(k_m)   (reciprocal, m != 0)
//            + m0(rho)                              (m = 0 channel)
//
// with r_n = |d + nT|, k_m = 2 pi m / L,
//
//     F(k) = 2 int_0^alpha exp(-u^2 rho^2 - k^2/(4 u^2)) du/u
//
// evaluated by Gauss-Legendre quadrature on v = u/alpha in [0, 1]
// (the integrand vanishes super-exponentially at v = 0), and
//
//     m0(rho) = (1/L) [ 2L int_0^1       erf(alpha r_t)/r_t dt
//                     + 2L int_1^inf ( erf(alpha r_t)/r_t - 1/(L t) ) dt
//                     - 2 gamma ]
//
// with r_t = sqrt(rho^2 + L^2 t^2); the tail integral is mapped onto
// s = 1/t in [0, 1] for quadrature. The m0 channel is the z-average of
// the background-subtracted erf part and is finite for all rho >= 0
// (its rho -> 0 limit exists). The whole kernel was validated against
// direct N -> inf background-subtracted lattice sums in a Python
// prototype (see handovers/HANDOVER_SECCM_ADAPTERS.md, item 2).
//
// This header is private to cpp/src/semiempirical/seccm/ — it is not
// public API.

#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/semiempirical/methods/indo/ccm_engine.hpp"

namespace vibeqc {
namespace semiempirical {
namespace seccm {
namespace detail {

// Gauss-Legendre nodes/weights on [-1, 1], computed once per order.
inline const std::pair<std::vector<double>, std::vector<double>>&
gauss_legendre(int order) {
    static std::vector<std::pair<int, std::pair<std::vector<double>, std::vector<double>>>> cache;
    for (const auto& entry : cache) {
        if (entry.first == order) {
            return entry.second;
        }
    }
    std::vector<double> x(order), w(order);
    for (int i = 0; i < order; ++i) {
        double xi = std::cos(M_PI * (i + 0.75) / (order + 0.5));
        for (int it = 0; it < 64; ++it) {
            double p0 = 1.0;
            double p1 = xi;
            for (int j = 2; j <= order; ++j) {
                double p2 = ((2.0 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j;
                p0 = p1;
                p1 = p2;
            }
            double dp = order * (xi * p1 - p0) / (xi * xi - 1.0);
            double dx = p1 / dp;
            xi -= dx;
            if (std::abs(dx) < 1e-15) {
                break;
            }
        }
        double p0 = 1.0;
        double p1 = xi;
        for (int j = 2; j <= order; ++j) {
            double p2 = ((2.0 * j - 1.0) * xi * p1 - (j - 1.0) * p0) / j;
            p0 = p1;
            p1 = p2;
        }
        double dp = order * (xi * p1 - p0) / (xi * xi - 1.0);
        x[i] = xi;
        w[i] = 2.0 / ((1.0 - xi * xi) * dp * dp);
    }
    cache.emplace_back(order, std::make_pair(std::move(x), std::move(w)));
    return cache.back().second;
}

// F(k) = 2 int_0^alpha exp(-u^2 rho^2 - k^2/(4 u^2)) du/u, k != 0,
// via GL quadrature on v = u/alpha in [0, 1]. The Jacobian of the
// [0, 1] map cancels the leading factor 2, so F = sum_j w_j f(v_j).
inline double wire_F(double k, double rho, double alpha, int order = 64) {
    const auto& gl = gauss_legendre(order);
    const double a2 = alpha * alpha;
    const double b = k * k / (4.0 * a2);
    double s = 0.0;
    for (int j = 0; j < order; ++j) {
        const double v = 0.5 * (gl.first[j] + 1.0);
        s += gl.second[j] * std::exp(-a2 * rho * rho * v * v - b / (v * v)) / v;
    }
    return s;
}

// m0(rho): the z-averaged m = 0 channel of the background-subtracted
// erf part. Finite for all rho >= 0.
inline double wire_m0(double rho, double L, double alpha, int order = 64) {
    const auto& gl = gauss_legendre(order);
    const double gamma = 0.5772156649015328606;
    double i1 = 0.0;
    for (int j = 0; j < order; ++j) {
        const double t = 0.5 * (gl.first[j] + 1.0);
        const double rt = std::sqrt(rho * rho + L * L * t * t);
        i1 += gl.second[j] * 2.0 * L * std::erf(alpha * rt) / rt;
    }
    i1 *= 0.5;
    double i2 = 0.0;
    for (int j = 0; j < order; ++j) {
        const double s = 0.5 * (gl.first[j] + 1.0);
        const double t = 1.0 / s;
        const double rt = std::sqrt(rho * rho + L * L * t * t);
        const double g = 2.0 * L * (std::erf(alpha * rt) / rt - 1.0 / (L * t));
        i2 += gl.second[j] * g / (s * s);
    }
    i2 *= 0.5;
    return (i1 + i2 - 2.0 * gamma) / L;
}

// WS-weighted background-corrected 1-D Madelung-constant matrix, the
// direct analogue of indo::_madkonst_2d for wires. Same contract: the
// ews record set mirrors indo::_ewald_ws_cells (WS records plus the
// atom itself at weight 1.0 and zero displacement), and the returned
// matrix feeds indo::_madelung_potential_ewald so the SMADEL short-range
// subtraction stays consistent across 1-D/2-D/3-D.
inline Eigen::MatrixXd wire_madkonst_1d(
    const std::vector<std::vector<indo::WSNeighbor>>& ews,
    const Eigen::Vector3d& T,
    int n_atoms) {
    const double L = T.norm();
    if (!(L > 0.0) || !std::isfinite(L)) {
        throw std::invalid_argument(
            "SECCM 1-D Ewald requires a finite nonzero translation vector");
    }
    const Eigen::Vector3d that = T / L;
    const double alpha = 4.0 / L;
    const double self_const = -2.0 * alpha / std::sqrt(M_PI);
    const int n_real = 8;
    const int m_max = 24;

    Eigen::MatrixXd mad = Eigen::MatrixXd::Zero(n_atoms, n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        for (const auto& nb : ews[i]) {
            const Eigen::Vector3d vij = -nb.disp;
            const double z = vij.dot(that);
            const Eigen::Vector3d inplane = vij - z * that;
            const double rho = inplane.norm();

            double direct = 0.0;
            for (int n = -n_real; n <= n_real; ++n) {
                const Eigen::Vector3d pos = vij + static_cast<double>(n) * T;
                const double dist = pos.norm();
                if (dist < 1e-8) {
                    direct += self_const;
                } else {
                    direct += std::erfc(alpha * dist) / dist;
                }
            }

            double recip = 0.0;
            for (int m = 1; m <= m_max; ++m) {
                const double km = 2.0 * M_PI * static_cast<double>(m) / L;
                recip += std::cos(km * z) * wire_F(km, rho, alpha);
            }
            recip *= 2.0 / L;

            const double m0 = wire_m0(rho, L, alpha);
            mad(i, nb.origin) += nb.weight * (direct + recip + m0);
        }
    }
    return mad;
}

// Gradient of the wire kernel Phi(v) with respect to its image argument v,
// dPhi/dv, split into the axial (that) and radial (in-plane) directions.
// Additive derivative helper for the embedded SCC-DFTB-SECCM analytic
// gradient (the kernel value semantics above are untouched); validated
// against central differences of wire_madkonst_1d itself.
//
//   direct : d/dv erfc(a r)/r = -(v + nT)/r^2 [erfc(ar)/r + 2a/sqrt(pi) e^{-a^2 r^2}]
//   recip  : d/dz  = -(2/L) sum_m k_m sin(k_m z) F(k_m, rho)      (along that)
//            dF/d(rho) = -2 rho a^2 sum_j w_j v_j exp(-a^2 rho^2 v_j^2 - b/v_j^2)
//   m0     : d/d(rho) [erf(a r)/r] = (2a/sqrt(pi)) e^{-a^2 r^2}/r - erf(a r)/r^2,
//            drt/d(rho) = rho/rt with rt = sqrt(rho^2 + L^2 t^2)
// The radial pieces all vanish like rho at rho -> 0, so the axis is skipped
// there (the in-plane direction is undefined but the contribution is zero).
inline Eigen::Vector3d wire_kernel_gradient(
    const Eigen::Vector3d& v,
    const Eigen::Vector3d& T,
    int order = 64) {
    const double L = T.norm();
    const Eigen::Vector3d that = T / L;
    const double alpha = 4.0 / L;
    const double a2 = alpha * alpha;
    const int n_real = 8;
    const int m_max = 24;
    const double z = v.dot(that);
    const Eigen::Vector3d inplane = v - z * that;
    const double rho = inplane.norm();

    Eigen::Vector3d grad = Eigen::Vector3d::Zero();

    // Direct (real-space) image sum; the self-image constant has zero
    // derivative and is skipped by the same threshold as the kernel.
    const double tpre = 2.0 * alpha / std::sqrt(M_PI);
    for (int n = -n_real; n <= n_real; ++n) {
        const Eigen::Vector3d pos = v + static_cast<double>(n) * T;
        const double dist = pos.norm();
        if (dist < 1e-8) continue;
        const double pref = (std::erfc(alpha * dist) / dist
                             + tpre * std::exp(-a2 * dist * dist))
            / (dist * dist);
        grad -= pos * pref;
    }

    // Axial part of the reciprocal channel (m != 0).
    for (int m = 1; m <= m_max; ++m) {
        const double km = 2.0 * M_PI * static_cast<double>(m) / L;
        grad += (-2.0 / L) * km * std::sin(km * z)
            * wire_F(km, rho, alpha, order) * that;
    }

    // Radial (in-plane) parts: dF/d(rho) of the reciprocal channel and
    // dm0/d(rho), both proportional to rho at rho -> 0.
    if (rho > 1.0e-12) {
        const Eigen::Vector3d rho_hat = inplane / rho;
        const auto& gl = gauss_legendre(order);
        double radial = 0.0;

        // dF(k, rho)/d(rho) with the same Jacobian-cancelled quadrature as
        // wire_F: F = sum_j w_j f(v_j), f = exp(...)/v, so the derivative
        // picks up one power of v (dF/drho = -2 rho a^2 sum_j w_j v_j f(v_j) v_j).
        for (int m = 1; m <= m_max; ++m) {
            const double km = 2.0 * M_PI * static_cast<double>(m) / L;
            const double b = km * km / (4.0 * a2);
            double df = 0.0;
            for (int j = 0; j < order; ++j) {
                const double vv = 0.5 * (gl.first[j] + 1.0);
                df += gl.second[j] * vv
                    * std::exp(-a2 * rho * rho * vv * vv - b / (vv * vv));
            }
            radial += std::cos(km * z) * (-2.0 * rho * a2 * df);
        }
        radial *= 2.0 / L;

        // dm0/d(rho): i1 quadrature (t in [0, 1]) and i2 quadrature
        // (s = 1/t), same nodes as wire_m0; each quadrature carries its own
        // 1/2 GL factor, so keep the two accumulators separate.
        double di1 = 0.0;
        for (int j = 0; j < order; ++j) {
            const double t = 0.5 * (gl.first[j] + 1.0);
            const double rt = std::sqrt(rho * rho + L * L * t * t);
            const double dkern = tpre * std::exp(-a2 * rt * rt) / rt
                - std::erf(alpha * rt) / (rt * rt);
            di1 += gl.second[j] * 2.0 * L * (rho / rt) * dkern;
        }
        di1 *= 0.5;
        double di2 = 0.0;
        for (int j = 0; j < order; ++j) {
            const double s = 0.5 * (gl.first[j] + 1.0);
            const double t = 1.0 / s;
            const double rt = std::sqrt(rho * rho + L * L * t * t);
            const double dkern = tpre * std::exp(-a2 * rt * rt) / rt
                - std::erf(alpha * rt) / (rt * rt);
            di2 += gl.second[j] * 2.0 * L * (rho / rt) * dkern / (s * s);
        }
        di2 *= 0.5;
        radial += (di1 + di2) / L;

        grad += radial * rho_hat;
    }
    return grad;
}

}  // namespace detail
}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
