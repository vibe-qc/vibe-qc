// OMx (OM1/OM2/OM3) adapter for a frozen semiempirical cyclic-cluster
// topology, WS-weighted supercell Fock.
//
// The supercell Fock is the molecular OMx Hamiltonian evaluated over the WS
// record set (the Bredow-Geudtner-Jug construction), exactly like the PM6
// adapter generalizes the molecular PM6 driver:
//   * one-center blocks per supercell atom (unchanged molecular terms),
//   * two-center terms (penetration-corrected core attraction, eq-17
//     resonance, two-electron Coulomb/exchange, core-core repulsion, and the
//     eq-8 F1/F2 one-center VORT corrections) accumulated through the WS
//     records with their fractional weights x_MN = 1/n_N,
//   * three-center terms (the eq-9 G1/G2 VORT corrections and the OM2/OM3
//     ECP) weighted per (mu, nu; C) by the Peintinger-Bredow 2014 scheme
//     (eq 13, production default) or Janetzko's original scheme (eq 10,
//     opt-in comparison).
//
// The molecular limit (one replica, zero-translation images only)
// reproduces nddo::run_omx_v2 on the same molecule.

#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <string>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/methods/nddo/omx_parameters.hpp"
#include "vibeqc/semiempirical/seccm/topology.hpp"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

// Three-center image weighting, Peintinger-Bredow 2014 (JCC 35, 839,
// doi:10.1002/jcc.23550) eqs 10-14.  For a three-center term over the
// orbital pair (mu, nu) and third center C:
//   * peintinger_eq13 (eq 13):  x_muNuC = x_muNu (x_muC + x_nuC)/2 with the
//     center weights from the union WSSC(MN) = WSSC(M) u WSSC(N); the C-sum
//     runs over the union images (interaction range extended to +-t).
//   * janetzko_eq10 (eq 10):    x_muNuC = x_muNu x_muC x_nuC/(x_muC + x_nuC)
//     with the center weights from the individual WSSCs.
enum class OMxThreeCenterWeighting {
    peintinger_eq13,
    janetzko_eq10,
};

inline OMxThreeCenterWeighting parse_omx_three_center_weighting(
    const std::string& value) {
    if (value == "peintinger_eq13")
        return OMxThreeCenterWeighting::peintinger_eq13;
    if (value == "janetzko_eq10") return OMxThreeCenterWeighting::janetzko_eq10;
    throw std::invalid_argument(
        "OMx-SECCM: three_center_weighting must be 'peintinger_eq13' "
        "(default) or 'janetzko_eq10'");
}

inline const char* omx_three_center_weighting_name(
    OMxThreeCenterWeighting weighting) {
    if (weighting == OMxThreeCenterWeighting::peintinger_eq13)
        return "peintinger_eq13";
    return "janetzko_eq10";
}

// Center factor before the orbital-pair weight x_muNu is applied. Eq. 13
// averages image-resolved endpoint ownership. Janetzko's Eq. 10 is the
// product-over-sum expression and is therefore exactly zero when the center
// image is absent from either endpoint WSSC.
inline double omx_three_center_endpoint_weight(
    OMxThreeCenterWeighting weighting,
    double x_mu_c,
    double x_nu_c) {
    if (!std::isfinite(x_mu_c) || !std::isfinite(x_nu_c)
        || x_mu_c < 0.0 || x_nu_c < 0.0) {
        throw std::invalid_argument(
            "OMx-SECCM three-center endpoint weights must be finite and "
            "non-negative");
    }
    if (weighting == OMxThreeCenterWeighting::peintinger_eq13)
        return 0.5 * (x_mu_c + x_nu_c);
    if (x_mu_c == 0.0 || x_nu_c == 0.0) return 0.0;
    return x_mu_c * x_nu_c / (x_mu_c + x_nu_c);
}

struct OMxSECCMResult : ParameterIdentifiedResult {
    double energy = 0.0;  // Total energy per primitive cell.
    // Variational electronic component 1/2 Tr[P(H+F)], per primitive cell.
    // This closes exactly with e_core to energy; orbital eigenvalues remain
    // available separately in mo_energies.
    double e_electronic = 0.0;
    double e_core = 0.0;        // WS-weighted core-core repulsion, per cell.
    double total_cyclic_energy = 0.0;
    double cyclic_core_energy = 0.0;
    double homo_lumo_gap = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    int n_basis = 0;
    int n_occ = 0;
    int n_iter = 0;
    int group_order = 0;
    int n_records = 0;
    std::string three_center_weighting;
    bool converged = false;
    // True only for a non-molecular cyclic-image topology whose caller
    // explicitly accepted omission of the unimplemented long-range
    // electrostatics.
    bool truncated_electrostatics_acknowledged = false;
};

struct OMxSECCMOptions {
    int max_iter = 100;
    double conv_tol = 1.0e-7;
    OMxThreeCenterWeighting three_center_weighting =
        OMxThreeCenterWeighting::peintinger_eq13;
    bool allow_truncated_electrostatics = false;
};

OMxSECCMResult run_omx_seccm(
    const Molecule& mol,
    const nddo::OMxParameterSet& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    const OMxSECCMOptions& opts = {},
    double gap_tolerance = 1.0e-8);

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
