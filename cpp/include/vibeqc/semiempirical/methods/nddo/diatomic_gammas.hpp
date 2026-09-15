// MOPAC-style NDDO two-electron integrals with sigma/pi decomposition.
//
// In the NDDO approximation, two-center two-electron integrals
// (mu_A nu_A | lambda_B sigma_B) are computed in the diatomic frame (z-axis
// along the bond) and then rotated to the molecular frame.
//
// This header provides the sigma/pi gamma decomposition using the
// STO-6G multipole expansion already loaded as gamma_terms.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "vibeqc/semiempirical/methods/nddo/pm6_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// ---------------------------------------------------------------------------
// Diatomic gamma set: sigma and pi gammas for an atom pair
// ---------------------------------------------------------------------------

struct DiatomicGammas {
    double g_ss   = 0.0;   // (ss|ss), isotropic
    double g_sp_s = 0.0;   // (ss|p_sigma p_sigma), sigma
    double g_sp_p = 0.0;   // (ss|p_pi p_pi), pi
    double g_pp_ss = 0.0;  // (p_sigma p_sigma|ss), same as g_sp_s
    double g_pp_pp_s = 0.0; // (p_sigma p_sigma|p_sigma p_sigma)
    double g_pp_pp_p = 0.0; // (p_pi p_pi|p_pi p_pi)
    double g_pp_pp_x = 0.0; // (p_sigma p_sigma|p_pi p_pi)
};

// ---------------------------------------------------------------------------
// Compute sigma/pi gammas from multipole gamma_terms
//
// The gamma_terms are split into s-type (first 6) and p-type (next 6).
// The sigma/pi decomposition uses the quadrupole moment of the p charge
// distribution, estimated from the ratio of Slater exponents zp/zs.
// ---------------------------------------------------------------------------

inline DiatomicGammas compute_diatomic_gammas(
    double R,
    const std::vector<NDDOElementData::GammaTerm>& terms_a,
    const std::vector<NDDOElementData::GammaTerm>& terms_b,
    double zp_over_zs_a = 1.0,
    double zp_over_zs_b = 1.0) {

    DiatomicGammas dg;
    if (terms_a.empty() || terms_b.empty()) return dg;

    int n_s_a = std::min(6, (int)terms_a.size());
    int n_s_b = std::min(6, (int)terms_b.size());
    bool has_p_a = (int)terms_a.size() > 6;
    bool has_p_b = (int)terms_b.size() > 6;

    // Helper: compute gamma from subsets
    auto gamma_sub = [R](const auto& ta, int off_a, int n_a,
                          const auto& tb, int off_b, int n_b) -> double {
        double g = 0.0;
        for (int i = 0; i < n_a; ++i) {
            for (int j = 0; j < n_b; ++j) {
                double d = ta[off_a + i].exponent + tb[off_b + j].exponent;
                g += ta[off_a + i].coeff * tb[off_b + j].coeff
                   / std::sqrt(R * R + d * d);
            }
        }
        return g;
    };

    // s-s gamma (isotropic)
    dg.g_ss = gamma_sub(terms_a, 0, n_s_a, terms_b, 0, n_s_b);

    if (has_p_a && has_p_b) {
        int n_p_a = (int)terms_a.size() - n_s_a;
        int n_p_b = (int)terms_b.size() - n_s_b;

        // p-p sigma and pi from quadrupole decomposition
        double g_pp_mono = gamma_sub(terms_a, n_s_a, n_p_a,
                                      terms_b, n_s_b, n_p_b);

        // Quadrupole correction: delta/3 for sigma, -delta/6 for pi,
        // with delta proportional to (zp/zs)^2.
        double q_a = zp_over_zs_a * zp_over_zs_a;
        double q_b = zp_over_zs_b * zp_over_zs_b;
        double quad = 0.2 * (q_a + q_b) * g_pp_mono;  // empirical scaling

        dg.g_pp_pp_s = g_pp_mono + quad / 3.0;
        dg.g_pp_pp_p = g_pp_mono - quad / 6.0;
        dg.g_pp_pp_x = g_pp_mono - quad / 3.0;

        // s-p gammas
        double g_sp_mono = gamma_sub(terms_a, 0, n_s_a, terms_b, n_s_b, n_p_b);
        dg.g_sp_s = g_sp_mono + quad / 4.0;
        dg.g_sp_p = g_sp_mono - quad / 8.0;
        dg.g_pp_ss = dg.g_sp_s;  // symmetric

    } else if (has_p_b) {
        // Only B has p orbitals
        int n_p_b = (int)terms_b.size() - n_s_b;
        double g_sp_mono = gamma_sub(terms_a, 0, n_s_a, terms_b, n_s_b, n_p_b);
        dg.g_sp_s = g_sp_mono;
        dg.g_sp_p = g_sp_mono;
        dg.g_pp_ss = g_sp_mono;
        dg.g_pp_pp_s = dg.g_ss;
        dg.g_pp_pp_p = dg.g_ss;
        dg.g_pp_pp_x = dg.g_ss;
    } else if (has_p_a) {
        int n_p_a = (int)terms_a.size() - n_s_a;
        double g_sp_mono = gamma_sub(terms_a, n_s_a, n_p_a, terms_b, 0, n_s_b);
        dg.g_sp_s = g_sp_mono;
        dg.g_sp_p = g_sp_mono;
        dg.g_pp_ss = g_sp_mono;
        dg.g_pp_pp_s = dg.g_ss;
        dg.g_pp_pp_p = dg.g_ss;
        dg.g_pp_pp_x = dg.g_ss;
    }

    return dg;
}

// ---------------------------------------------------------------------------
// Rotation: compute the gamma for AO pair (mu, nu) given their types and
// directions relative to the bond axis.
//
// For s-s: gamma = g_ss (isotropic)
// For s-p: gamma = g_sp_s * cos^2(theta) + g_sp_p * sin^2(theta)
// For p-p: gamma is assembled from direction cosines of both AOs.
// ---------------------------------------------------------------------------

inline double gamma_ao_pair(
    const DiatomicGammas& dg,
    int t_mu, int t_nu,
    double cos_mu, double cos_nu) {

    if (t_mu == 0 && t_nu == 0) {
        // s-s: isotropic
        return dg.g_ss;
    }

    if (t_mu == 0) {
        // s-p: orientation of the p orbital matters
        double cos2 = cos_nu * cos_nu;
        return dg.g_sp_s * cos2 + dg.g_sp_p * (1.0 - cos2);
    }

    if (t_nu == 0) {
        // p-s: symmetric
        double cos2 = cos_mu * cos_mu;
        return dg.g_pp_ss * cos2 + dg.g_sp_p * (1.0 - cos2);
    }

    // p-p: depends on both orientations
    double cm2 = cos_mu * cos_mu;
    double cn2 = cos_nu * cos_nu;

    // Weighted combination of sigma and pi gammas
    double sigma_weight = cm2 * cn2;
    double pi_weight = (1.0 - cm2) * (1.0 - cn2);
    double cross_weight = 1.0 - sigma_weight - pi_weight;

    return dg.g_pp_pp_s * sigma_weight
         + dg.g_pp_pp_p * pi_weight
         + dg.g_pp_pp_x * cross_weight;
}

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
