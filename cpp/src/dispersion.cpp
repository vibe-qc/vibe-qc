// Native D3(BJ): CN calculator, C6 interpolation, and pairwise summation.
// Reference-table *data* and functional-parameter registry are kept in
// their own translation units so this file is small, self-contained, and
// easy to test against hand-worked intermediate quantities.

#include "vibeqc/dispersion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace vibeqc {

namespace {

// ---- Grimme's CN counting function constants ----------------------------
//
// CN counting function:
//   f_k(r) = 1 / (1 + exp(-k1 * (k2 * (r_Aco + r_Bco) / r_AB - 1)))
// with k1 = 16.0 and the damping-length rescaling k2 that multiplies the
// covalent-radius sum. We absorb k2 into the tabulated rcov() values, so
// here we see a bare (r_Aco + r_Bco) / r_AB ratio.
//
// See Grimme J. Chem. Phys. 132, 154104 (2010), Eq. 15-16.
constexpr double kCN_k1 = 16.0;

// C6 interpolation Gaussian width. Grimme uses k3 = -4.0 with the form
//     L_ij = exp(k3 * ((CN_A - CN_A_ref_i)^2 + (CN_B - CN_B_ref_j)^2))
// (negative exponent, i.e. a narrow Gaussian centered at each reference
// CN pair). We store the positive magnitude here and negate inside the
// exponent for clarity.
constexpr double kC6_k3 = 4.0;

}  // namespace

CoordinationNumbers coordination_numbers(const Molecule& mol) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());

    CoordinationNumbers out;
    out.cn = Eigen::VectorXd::Zero(n);
    out.dcn_dr.assign(static_cast<std::size_t>(n),
                       Eigen::MatrixX3d::Zero(n, 3));

    for (Eigen::Index A = 0; A < n; ++A) {
        const int ZA = atoms[A].Z;
        const double rcA = rcov(ZA);
        for (Eigen::Index B = 0; B < n; ++B) {
            if (A == B) continue;
            const int ZB = atoms[B].Z;
            const double rcB = rcov(ZB);

            const double dx = atoms[A].xyz[0] - atoms[B].xyz[0];
            const double dy = atoms[A].xyz[1] - atoms[B].xyz[1];
            const double dz = atoms[A].xyz[2] - atoms[B].xyz[2];
            const double r2 = dx*dx + dy*dy + dz*dz;
            const double r  = std::sqrt(r2);
            if (r < 1e-10) continue;

            const double r0 = rcA + rcB;      // already includes k2 scaling
            const double arg = -kCN_k1 * (r0 / r - 1.0);
            // Numerical hygiene: clip the exponent to avoid overflow/underflow
            // traps. The counting function is flat far outside [-30, 30] and
            // its derivative vanishes there.
            const double e = std::exp(std::min(std::max(arg, -60.0), 60.0));
            const double f = 1.0 / (1.0 + e);
            out.cn(A) += f;

            // dCN_A / dr_A = Σ_{B≠A} f'(r) · (r_A - r_B) / r
            // with f'(r) = -k1 * r0 / r^2 * e / (1 + e)^2
            const double fprime = -kCN_k1 * r0 / r2 * e / ((1.0 + e) * (1.0 + e));
            out.dcn_dr[static_cast<std::size_t>(A)](B, 0) += -fprime * dx / r;
            out.dcn_dr[static_cast<std::size_t>(A)](B, 1) += -fprime * dy / r;
            out.dcn_dr[static_cast<std::size_t>(A)](B, 2) += -fprime * dz / r;
            out.dcn_dr[static_cast<std::size_t>(A)](A, 0) +=  fprime * dx / r;
            out.dcn_dr[static_cast<std::size_t>(A)](A, 1) +=  fprime * dy / r;
            out.dcn_dr[static_cast<std::size_t>(A)](A, 2) +=  fprime * dz / r;
        }
    }
    return out;
}

// ---- C6 interpolation ---------------------------------------------------
//
// Grimme's CN-dependent C6 is obtained from a small grid of "reference"
// C6 values c6ab(ZA, ZB, i, j) evaluated at reference coordination numbers
// (CN_A_ref_i, CN_B_ref_j) — typically up to 5 reference CN per element,
// capturing the atom's different chemical environments (free atom,
// saturated, unsaturated, …). The fractional interpolation is
//
//   C6^AB(CN_A, CN_B)
//       = Σ_{ij} L_ij · c6ab_{ij}   /   Σ_{ij} L_ij
//   L_ij = exp(-k3 · [(CN_A - CN_A_ref_i)^2 + (CN_B - CN_B_ref_j)^2])
//
// The reference grid lives in dispersion_data.cpp. For element pairs
// outside the shipped table we return NaN; `compute_d3bj` catches that
// and promotes it to a clear exception.

// Private helpers exposed by dispersion_data.cpp via free functions.
// Returns the number of reference points for (ZA, ZB); zero if unsupported.
int c6_num_reference(int ZA, int ZB);
// Fills `out` with tuples (c6_ref, cn_ref_A, cn_ref_B) for pair (ZA, ZB).
bool c6_reference_grid(int ZA, int ZB,
                       std::vector<std::array<double, 3>>& out);

struct C6Interpolation {
    double value = std::numeric_limits<double>::quiet_NaN();
    double d_cn_a = std::numeric_limits<double>::quiet_NaN();
    double d_cn_b = std::numeric_limits<double>::quiet_NaN();
};

C6Interpolation interpolate_c6_with_derivatives(
        int ZA, int ZB, double cnA, double cnB) {
    std::vector<std::array<double, 3>> grid;
    if (!c6_reference_grid(ZA, ZB, grid) || grid.empty()) {
        return {};
    }

    // Shift all exponents by their maximum before exponentiating. This is
    // algebraically invisible after normalisation but avoids the 0/0
    // underflow that otherwise occurs for coordination numbers far beyond
    // every reference environment.
    double max_exponent = -std::numeric_limits<double>::infinity();
    for (const auto& e : grid) {
        const double da = cnA - e[1];
        const double db = cnB - e[2];
        max_exponent = std::max(
            max_exponent, -kC6_k3 * (da*da + db*db));
    }

    double num = 0.0;
    double den = 0.0;
    double dnum_a = 0.0;
    double dden_a = 0.0;
    double dnum_b = 0.0;
    double dden_b = 0.0;
    for (const auto& e : grid) {
        const double c6_ref = e[0];
        const double cnA_ref = e[1];
        const double cnB_ref = e[2];
        const double da = cnA - cnA_ref;
        const double db = cnB - cnB_ref;
        const double L = std::exp(
            -kC6_k3 * (da*da + db*db) - max_exponent);
        const double dL_a = -2.0 * kC6_k3 * da * L;
        const double dL_b = -2.0 * kC6_k3 * db * L;
        num += L * c6_ref;
        den += L;
        dnum_a += dL_a * c6_ref;
        dden_a += dL_a;
        dnum_b += dL_b * c6_ref;
        dden_b += dL_b;
    }
    if (den == 0.0) {
        return {};
    }
    const double value = num / den;
    const double inv_den = 1.0 / den;
    return {
        value,
        (dnum_a - value * dden_a) * inv_den,
        (dnum_b - value * dden_b) * inv_den,
    };
}

double interpolate_c6(int ZA, int ZB, double cnA, double cnB) {
    return interpolate_c6_with_derivatives(ZA, ZB, cnA, cnB).value;
}

// ---- Pairwise dispersion energy -----------------------------------------

DispersionResult compute_d3bj(const Molecule& mol,
                               const D3BJParams& params,
                               bool with_gradient) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());

    DispersionResult out;
    if (with_gradient) {
        out.gradient = Eigen::MatrixX3d::Zero(n, 3);
    }
    if (n < 2) return out;

    // Check that every element is covered before we start doing work that
    // would throw halfway through with misleading intermediates.
    const int maxZ = max_supported_Z();
    for (const auto& a : atoms) {
        if (a.Z < 1 || a.Z > maxZ) {
            throw std::invalid_argument(
                "compute_d3bj: element Z=" + std::to_string(a.Z) +
                " outside shipped parameter range (max Z=" +
                std::to_string(maxZ) + "). Expand dispersion_data.cpp.");
        }
    }

    const auto cnres = coordination_numbers(mol);
    const auto& cn = cnres.cn;

    // r2r4-derived Q per atom, precomputed.
    std::vector<double> Q(static_cast<std::size_t>(n));
    for (Eigen::Index A = 0; A < n; ++A) {
        const int Z = atoms[A].Z;
        Q[static_cast<std::size_t>(A)] =
            0.5 * std::sqrt(static_cast<double>(Z)) * r2r4(Z);
    }

    // Symmetric C6 matrix, built once. The two-body loop and the ATM
    // triple loop (which needs all three pair C6 of every triple) share it.
    const auto idx = [n](Eigen::Index A, Eigen::Index B) {
        return static_cast<std::size_t>(A * n + B);
    };
    std::vector<double> c6(static_cast<std::size_t>(n * n), 0.0);
    // dc6_dcn[A,B] is d C6(A,B) / d CN_A. The transposed slot carries
    // the derivative with respect to the other atom's coordination number.
    std::vector<double> dc6_dcn(static_cast<std::size_t>(n * n), 0.0);
    for (Eigen::Index A = 0; A < n; ++A) {
        for (Eigen::Index B = A; B < n; ++B) {
            const auto interp = interpolate_c6_with_derivatives(
                atoms[A].Z, atoms[B].Z, cn(A), cn(B));
            if (!std::isfinite(interp.value)) {
                throw std::invalid_argument(
                    "compute_d3bj: no C6 reference data for pair (Z=" +
                    std::to_string(atoms[A].Z) + ", Z=" +
                    std::to_string(atoms[B].Z) + ")");
            }
            c6[idx(A, B)] = interp.value;
            c6[idx(B, A)] = interp.value;
            dc6_dcn[idx(A, B)] = interp.d_cn_a;
            dc6_dcn[idx(B, A)] = interp.d_cn_b;
        }
    }

    double e_disp = 0.0;
    Eigen::VectorXd dE_dcn = Eigen::VectorXd::Zero(n);
    for (Eigen::Index A = 0; A < n; ++A) {
        for (Eigen::Index B = A + 1; B < n; ++B) {
            const double dx = atoms[A].xyz[0] - atoms[B].xyz[0];
            const double dy = atoms[A].xyz[1] - atoms[B].xyz[1];
            const double dz = atoms[A].xyz[2] - atoms[B].xyz[2];
            const double r2 = dx*dx + dy*dy + dz*dz;
            const double r  = std::sqrt(r2);
            if (r < 1e-10) continue;

            const double C6 = c6[idx(A, B)];
            const double C8 = 3.0 * C6 * std::sqrt(
                Q[static_cast<std::size_t>(A)] *
                Q[static_cast<std::size_t>(B)]);

            const double R0 = std::sqrt(C8 / C6);
            const double fd = bj_damping_radius(R0, params);
            // f_d^6 and f_d^8 used as the BJ-damping floor in the denominator.
            const double fd2 = fd * fd;
            const double fd6 = fd2 * fd2 * fd2;
            const double fd8 = fd6 * fd2;

            const double r6 = r2 * r2 * r2;
            const double r8 = r6 * r2;

            // Sign convention: dispersion is attractive, so E_disp < 0.
            const double pair_factor = params.s6 / (r6 + fd6)
                                     + params.s8 * (C8 / C6) / (r8 + fd8);
            e_disp -= C6 * pair_factor;

            if (with_gradient) {
                // Explicit pair-distance derivative at fixed C6. The
                // coordination-number chain term is accumulated separately
                // through dE_dcn and applied once after all pairs/triples.
                const double dE6_dr = 6.0 * params.s6 * C6 * r6 / r
                                       / ((r6 + fd6) * (r6 + fd6));
                const double dE8_dr = 8.0 * params.s8 * C8 * r8 / r
                                       / ((r8 + fd8) * (r8 + fd8));
                // dE/dr_A_x = (dE/dr) · (r_A_x - r_B_x) / r = dE_dr · dx / r
                // dE/dr_B_x = -(dE/dr) · dx / r          (Newton's 3rd law)
                // where dx = r_A_x - r_B_x and dE_dr = dE/dr > 0.
                const double dE_dr = dE6_dr + dE8_dr;
                const double inv_r = 1.0 / r;
                out.gradient(A, 0) += dE_dr * dx * inv_r;
                out.gradient(A, 1) += dE_dr * dy * inv_r;
                out.gradient(A, 2) += dE_dr * dz * inv_r;
                out.gradient(B, 0) -= dE_dr * dx * inv_r;
                out.gradient(B, 1) -= dE_dr * dy * inv_r;
                out.gradient(B, 2) -= dE_dr * dz * inv_r;

                dE_dcn(A) -= dc6_dcn[idx(A, B)] * pair_factor;
                dE_dcn(B) -= dc6_dcn[idx(B, A)] * pair_factor;
            }
        }
    }

    // ---- Axilrod-Teller-Muto three-body term (s9 != 0) -------------------
    //
    // E^(3) = - s9 · Σ_{A<B<C} C9 · ang · f_damp   (see dispersion.hpp).
    //
    // The explicit side-length derivative is evaluated below. C9's
    // coordination-number dependence is accumulated into dE_dcn and chained
    // through the CN Jacobian together with the two-body contribution.
    if (params.s9 != 0.0 && n >= 3) {
        // Exponent of the D3 zero-damping function: alp + 2 = 16.
        const double m_exp = kD3_alp + 2.0;
        constexpr double kFourThirds = 4.0 / 3.0;

        for (Eigen::Index A = 0; A < n; ++A) {
        for (Eigen::Index B = A + 1; B < n; ++B) {
        for (Eigen::Index C = B + 1; C < n; ++C) {
            const double abx = atoms[A].xyz[0] - atoms[B].xyz[0];
            const double aby = atoms[A].xyz[1] - atoms[B].xyz[1];
            const double abz = atoms[A].xyz[2] - atoms[B].xyz[2];
            const double bcx = atoms[B].xyz[0] - atoms[C].xyz[0];
            const double bcy = atoms[B].xyz[1] - atoms[C].xyz[1];
            const double bcz = atoms[B].xyz[2] - atoms[C].xyz[2];
            const double acx = atoms[A].xyz[0] - atoms[C].xyz[0];
            const double acy = atoms[A].xyz[1] - atoms[C].xyz[1];
            const double acz = atoms[A].xyz[2] - atoms[C].xyz[2];

            const double u2 = abx*abx + aby*aby + abz*abz;   // r_AB^2
            const double v2 = bcx*bcx + bcy*bcy + bcz*bcz;   // r_BC^2
            const double w2 = acx*acx + acy*acy + acz*acz;   // r_AC^2
            const double u = std::sqrt(u2);
            const double v = std::sqrt(v2);
            const double w = std::sqrt(w2);
            if (u < 1e-10 || v < 1e-10 || w < 1e-10) continue;

            // C9_ABC = -sqrt(|C6_AB · C6_AC · C6_BC|)  (negative).
            const double C9 = -std::sqrt(std::abs(
                c6[idx(A, B)] * c6[idx(A, C)] * c6[idx(B, C)]));

            const double rprod = u * v * w;
            const double r0prod = d3_r0ab(atoms[A].Z, atoms[B].Z)
                                * d3_r0ab(atoms[B].Z, atoms[C].Z)
                                * d3_r0ab(atoms[A].Z, atoms[C].Z);

            // t = 6 · [ (4/3)·(r0prod/rprod)^(1/3) ]^(alp+2), f_damp = 1/(1+t)
            const double t = 6.0 * std::pow(
                kFourThirds * std::cbrt(r0prod / rprod), m_exp);
            const double fdmp = 1.0 / (1.0 + t);

            // Angular factor written with the triangle side lengths:
            //   3·cosθ_A·cosθ_B·cosθ_C + 1  ->  0.375·P/rprod^2 + 1
            // so ang = 0.375·P/rprod^5 + 1/rprod^3 with
            //   P = (u²+v²-w²)(u²-v²+w²)(-u²+v²+w²).
            const double pA = u2 + v2 - w2;
            const double pB = u2 - v2 + w2;
            const double pC = -u2 + v2 + w2;
            const double P = pA * pB * pC;

            const double rp2 = rprod * rprod;
            const double rp3 = rp2 * rprod;
            const double rp5 = rp3 * rp2;
            const double ang = 0.375 * P / rp5 + 1.0 / rp3;

            const double triple_energy = -params.s9 * C9 * ang * fdmp;
            e_disp += triple_energy;

            if (with_gradient) {
                const double pref = -params.s9 * C9;

                // d f_damp / d rprod = (m/3)·t / (rprod·(1+t)²)
                // and d rprod / d(edge) = rprod / edge, so the damping
                // contribution to dE/d(edge) is  pref·ang·(m/3)·t/(1+t)²/edge.
                const double damp_fac =
                    (m_exp / 3.0) * t / ((1.0 + t) * (1.0 + t));

                // dP/du = 2u(pB·pC + pA·pC − pA·pB), and cyclically.
                const double dP_du = 2.0 * u * (pB*pC + pA*pC - pA*pB);
                const double dP_dv = 2.0 * v * (pB*pC - pA*pC + pA*pB);
                const double dP_dw = 2.0 * w * (-pB*pC + pA*pC + pA*pB);

                // dang/d(edge) = 0.375·(dP/d(edge) − 5P/edge)/rprod^5
                //                − 3/(edge·rprod^3)
                const double dang_du =
                    0.375 * (dP_du - 5.0 * P / u) / rp5 - 3.0 / (u * rp3);
                const double dang_dv =
                    0.375 * (dP_dv - 5.0 * P / v) / rp5 - 3.0 / (v * rp3);
                const double dang_dw =
                    0.375 * (dP_dw - 5.0 * P / w) / rp5 - 3.0 / (w * rp3);

                const double dE_du =
                    pref * (dang_du * fdmp + ang * damp_fac / u);
                const double dE_dv =
                    pref * (dang_dv * fdmp + ang * damp_fac / v);
                const double dE_dw =
                    pref * (dang_dw * fdmp + ang * damp_fac / w);

                // Chain each edge derivative onto its two endpoints.
                const double gu = dE_du / u;
                out.gradient(A, 0) += gu * abx;
                out.gradient(A, 1) += gu * aby;
                out.gradient(A, 2) += gu * abz;
                out.gradient(B, 0) -= gu * abx;
                out.gradient(B, 1) -= gu * aby;
                out.gradient(B, 2) -= gu * abz;

                const double gv = dE_dv / v;
                out.gradient(B, 0) += gv * bcx;
                out.gradient(B, 1) += gv * bcy;
                out.gradient(B, 2) += gv * bcz;
                out.gradient(C, 0) -= gv * bcx;
                out.gradient(C, 1) -= gv * bcy;
                out.gradient(C, 2) -= gv * bcz;

                const double gw = dE_dw / w;
                out.gradient(A, 0) += gw * acx;
                out.gradient(A, 1) += gw * acy;
                out.gradient(A, 2) += gw * acz;
                out.gradient(C, 0) -= gw * acx;
                out.gradient(C, 1) -= gw * acy;
                out.gradient(C, 2) -= gw * acz;

                const double c6_ab = c6[idx(A, B)];
                const double c6_ac = c6[idx(A, C)];
                const double c6_bc = c6[idx(B, C)];
                const double half_energy = 0.5 * triple_energy;
                dE_dcn(A) += half_energy * (
                    dc6_dcn[idx(A, B)] / c6_ab
                  + dc6_dcn[idx(A, C)] / c6_ac);
                dE_dcn(B) += half_energy * (
                    dc6_dcn[idx(B, A)] / c6_ab
                  + dc6_dcn[idx(B, C)] / c6_bc);
                dE_dcn(C) += half_energy * (
                    dc6_dcn[idx(C, A)] / c6_ac
                  + dc6_dcn[idx(C, B)] / c6_bc);
            }
        }
        }
        }
    }

    if (with_gradient) {
        for (Eigen::Index A = 0; A < n; ++A) {
            for (Eigen::Index X = 0; X < n; ++X) {
                out.gradient(X, 0) += dE_dcn(A) *
                                      cnres.dcn_dr[static_cast<std::size_t>(A)](X, 0);
                out.gradient(X, 1) += dE_dcn(A) *
                                      cnres.dcn_dr[static_cast<std::size_t>(A)](X, 1);
                out.gradient(X, 2) += dE_dcn(A) *
                                      cnres.dcn_dr[static_cast<std::size_t>(A)](X, 2);
            }
        }
    }

    out.energy = e_disp;
    return out;
}

// ======================================================================
// Chai-Head-Gordon ("CHG") dispersion — the ωB97X-D "-D" term.
//
// E_disp = − s6 · Σ_{A<B} C6_AB / R_AB^6 · f_damp(R_AB)
//   C6_AB     = sqrt(C6_A · C6_B)
//   R0_AB     = RvdW_A + RvdW_B
//   f_damp(R) = 1 / ( 1 + d · (R / R0_AB)^{-12} )
//   s6 = 1.0, d = 6.0  (ωB97X-D — Chai & Head-Gordon, PCCP 10, 6615, 2008)
//
// The atomic C6 / RvdW tables are the Grimme-2006 "DFT-D2" set
// (J. Comput. Chem. 27, 1787), in atomic units (C6 in Eh·bohr⁶, RvdW
// in bohr). Algorithm + data transcribed verbatim from Psi4's
// ``libdisp`` "CHG" scheme (psi4/src/psi4/libdisp). Index 0 is an
// unused placeholder; valid entries are Z = 1 (H) … 54 (Xe).
// ======================================================================

namespace {

constexpr double kCHG_s6 = 1.0;   // ωB97X-D global scaling
constexpr double kCHG_d  = 6.0;   // ωB97X-D damping parameter
constexpr int    kCHG_max_Z = 54;

// Grimme-2006 D2 atomic C6 coefficients (Eh·bohr⁶), Z-indexed.
static const double kCHGC6[] = {
    0.0000000000000000E+00, 2.4283353778422600E+00, 1.3876202159098600E+00, 2.7925856845186000E+01,
    2.7925856845186000E+01, 5.4290640947473300E+01, 3.0354192223028200E+01, 2.1334660819614100E+01,
    1.2141676889211300E+01, 1.3008939524155000E+01, 1.0927509200290200E+01, 9.9041392910566400E+01,
    9.9041392910566400E+01, 1.8715527662084300E+02, 1.6009668241060000E+02, 1.3598678115916600E+02,
    9.6613057532724100E+01, 8.7940431183287500E+01, 7.9961614941805800E+01, 1.8732872914783100E+02,
    1.8732872914783100E+02, 1.8732872914783100E+02, 1.8732872914783100E+02, 1.8732872914783100E+02,
    1.8732872914783100E+02, 1.8732872914783100E+02, 1.8732872914783100E+02, 1.8732872914783100E+02,
    1.8732872914783100E+02, 1.8732872914783100E+02, 1.8732872914783100E+02, 2.9469584335385700E+02,
    2.9660382115073300E+02, 2.8394178668055500E+02, 2.1924399411375800E+02, 2.1629530115495000E+02,
    2.0831648491346800E+02, 4.2790738408120400E+02, 4.2790738408120400E+02, 4.2790738408120400E+02,
    4.2790738408120400E+02, 4.2790738408120400E+02, 4.2790738408120400E+02, 4.2790738408120400E+02,
    4.2790738408120400E+02, 4.2790738408120400E+02, 4.2790738408120400E+02, 4.2790738408120400E+02,
    4.2790738408120400E+02, 6.4732483072195000E+02, 6.7143473197338400E+02, 6.6675151374468900E+02,
    5.5053832066223800E+02, 5.4637546001450800E+02, 5.2018412843920900E+02
};

// Grimme-2006 D2 atomic van-der-Waals radii (bohr), Z-indexed.
static const double kCHGRvdW[] = {
    2.07869858790000, 1.89161571498900, 1.91240270086800, 1.55902394092500,
    2.66073419251200, 2.80624309366500, 2.74388213602800, 2.63994720663300,
    2.53601227723800, 2.43207734784300, 2.34892940432700, 2.16184653141600,
    2.57758624899600, 3.09726089597100, 3.24276979712400, 3.22198281124500,
    3.18040883948700, 3.09726089597100, 3.01411295245500, 2.80624309366500,
    2.78545610778600, 2.95175199481800, 2.95175199481800, 2.95175199481800,
    2.95175199481800, 2.95175199481800, 2.95175199481800, 2.95175199481800,
    2.95175199481800, 2.95175199481800, 2.95175199481800, 3.11804788185000,
    3.26355678300300, 3.32591774064000, 3.34670472651900, 3.30513075476100,
    3.26355678300300, 3.07647391009200, 3.03489993833400, 3.09726089597100,
    3.09726089597100, 3.09726089597100, 3.09726089597100, 3.09726089597100,
    3.09726089597100, 3.09726089597100, 3.09726089597100, 3.09726089597100,
    3.09726089597100, 3.15962185360800, 3.40906568415600, 3.55457458530900,
    3.57536157118800, 3.57536157118800, 3.55457458530900
};

}  // namespace

int chg_max_supported_Z() { return kCHG_max_Z; }

DispersionResult compute_chg_dispersion(const Molecule& mol,
                                        bool with_gradient) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());

    for (Eigen::Index A = 0; A < n; ++A) {
        const int Z = atoms[A].Z;
        if (Z < 1 || Z > kCHG_max_Z) {
            throw std::invalid_argument(
                "compute_chg_dispersion: element Z=" + std::to_string(Z)
                + " is outside the ωB97X-D / DFT-D2 C6 table (Z ≤ 54, "
                  "H–Xe).");
        }
    }

    DispersionResult out;
    if (with_gradient) {
        out.gradient = Eigen::MatrixX3d::Zero(n, 3);
    }

    double e_disp = 0.0;
    for (Eigen::Index A = 0; A < n; ++A) {
        const int ZA = atoms[A].Z;
        for (Eigen::Index B = 0; B < A; ++B) {
            const int ZB = atoms[B].Z;

            const double dx = atoms[A].xyz[0] - atoms[B].xyz[0];
            const double dy = atoms[A].xyz[1] - atoms[B].xyz[1];
            const double dz = atoms[A].xyz[2] - atoms[B].xyz[2];
            const double R2 = dx*dx + dy*dy + dz*dz;
            const double R  = std::sqrt(R2);
            if (R < 1e-10) continue;

            const double C6  = std::sqrt(kCHGC6[ZA] * kCHGC6[ZB]);
            const double R0  = kCHGRvdW[ZA] + kCHGRvdW[ZB];
            const double R6  = R2 * R2 * R2;

            // CHG damping f = 1 / (1 + d·(R/R0)^{-12}).
            const double ratio_m12 = std::pow(R / R0, -12.0);
            const double f = 1.0 / (1.0 + kCHG_d * ratio_m12);

            // Pair contribution to E = −s6 · Σ C6/R^6 · f.
            e_disp += C6 / R6 * f;

            if (with_gradient) {
                // dE_AB/dR = −s6 · C6 · d/dR [ f / R^6 ]
                //   d(1/R^6)/dR = −6/R^7
                //   df/dR = −f² · d · (−12) · (R/R0)^{-13} · (1/R0)
                const double df_dR =
                    -f * f * kCHG_d * (-12.0)
                    * std::pow(R / R0, -13.0) * (1.0 / R0);
                const double dE_dR =
                    -kCHG_s6 * C6 * (df_dR / R6 - 6.0 * f / (R6 * R));
                // Chain to Cartesian: ∂R/∂r_A = (r_A − r_B)/R.
                const double gx = dE_dR * dx / R;
                const double gy = dE_dR * dy / R;
                const double gz = dE_dR * dz / R;
                out.gradient(A, 0) += gx;
                out.gradient(A, 1) += gy;
                out.gradient(A, 2) += gz;
                out.gradient(B, 0) -= gx;
                out.gradient(B, 1) -= gy;
                out.gradient(B, 2) -= gz;
            }
        }
    }

    out.energy = -kCHG_s6 * e_disp;
    return out;
}

}  // namespace vibeqc
