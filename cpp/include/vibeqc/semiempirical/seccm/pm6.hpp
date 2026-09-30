// PM6 NDDO adapter for a frozen semiempirical cyclic-cluster topology.
//
// The supercell Fock matrix is the molecular PM6 Fock (one-center blocks per
// supercell atom, unchanged) with every directed two-center term accumulated
// through the WS record set: nuclear attraction, resonance, exchange, and
// core-core repulsion each enter with the record's fractional WS weight. The
// displacement convention mirrors the molecular driver: each record's
// (a -> b image) contribution is computed at d = R_a - (R_b + T) = -disp.
//
// The molecular limit (one replica, zero-translation images only) reproduces
// nddo::run_pm6 on the same molecule. Neutral closed-shell clusters only. The
// opt-in Madelung embedding (madelung=true; 1-D and 2-D, 3-D refused) adds
// the MSINDO CCM point-charge field to the Fock diagonal; without it a
// nontrivial cyclic topology needs the truncated-electrostatics
// acknowledgement, because the WS-truncated monopole sum is not quantitative.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_parameters.hpp"
#include "vibeqc/semiempirical/seccm/topology.hpp"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

struct PM6SECCMResult : ParameterIdentifiedResult {
    double energy = 0.0;  // Total energy per primitive cell.
    // Variational electronic component 1/2 Tr[P(H+F)], per primitive cell.
    // This closes exactly with e_core to energy; orbital eigenvalues remain
    // available separately in mo_energies.
    double e_electronic = 0.0;
    double e_core = 0.0;        // WS-weighted core-core repulsion, per cell.
    // CCM Madelung self-energy 1/2 sum_I q_I V_mad(I), per cell. Zero
    // unless the opt-in embedding is enabled. Sign and split follow
    // MSINDO: the Fock deposit carries the electronic half implicitly,
    // this field is the core-charge half (madelsum.f + ccmfockcl.f).
    double e_madelung = 0.0;
    double total_cyclic_energy = 0.0;
    double cyclic_core_energy = 0.0;
    double cyclic_madelung_energy = 0.0;
    double homo_lumo_gap = 0.0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    int n_basis = 0;
    int n_occ = 0;
    int n_iter = 0;
    int group_order = 0;
    int n_records = 0;
    bool converged = false;
};

struct PM6SECCMOptions {
    int max_iter = 100;
    double conv_tol = 1.0e-7;
    // Opt-in CCM Madelung/Ewald embedding (Bredow, Geudtner & Jug, JCC
    // 22, 89 (2001); the exact Ewald matrices are Janetzko, Bredow &
    // Jug, JCP 116, 8994 (2002)). Restores the Coulomb tail beyond the
    // Wigner-Seitz cell that a finite cyclic cluster otherwise drops.
    // Validated for 1-D and 2-D; 3-D fails closed because the
    // WS-folded remainder has no thermodynamic limit (issues #211,
    // #425, #444).
    bool madelung = false;
    // Acknowledge that a nontrivial cyclic run without the embedding
    // omits the long-range Coulomb tail and is not quantitative. The
    // WS-truncated monopole sum this replaces reproduces the exact
    // rocksalt Madelung constant only to -43%..+38% with the sign
    // flipping on replica parity, so a silent run would be a
    // quantitatively wrong answer rather than an approximation.
    bool allow_truncated_electrostatics = false;
};

PM6SECCMResult run_pm6_seccm(
    const Molecule& mol,
    const nddo::PM6ParameterSet& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    const PM6SECCMOptions& opts = {},
    double gap_tolerance = 1.0e-8);

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
