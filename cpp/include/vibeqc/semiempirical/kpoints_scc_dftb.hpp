// Periodic SCC-DFTB with k-point sampling.
//
// Extends k-point DFTB0 with charge self-consistency.
// For each SCF iteration:
//   1. Compute Mulliken charges from weighted D(k) S(k) populations
//   2. Build SCC Hamiltonian correction (k-independent)
//   3. Diagonalize at each k-point
//   4. Recompute charges, check convergence
//
// Stage 10b: k-point SCC-DFTB.

#pragma once

#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/bloch.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/kpoints_dftb0.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/scc_dftb.hpp"

namespace vibeqc {
namespace semiempirical {

struct KPointSCCDFTBResult : ParameterIdentifiedResult {
    double energy = 0.0;
    double free_energy = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    double e_scc = 0.0;
    double fermi_level = 0.0;
    double entropy = 0.0;
    double smearing_temperature = 0.0;
    std::vector<double> band_energies;
    std::vector<Eigen::VectorXd> eps_per_k;
    std::vector<Eigen::VectorXd> occupations_per_k;
    // Measured band edges + gaps from the occupations actually used
    // (issue #426); convention documented on KPointBandEdges. NaN / -1 /
    // empty means the edge does not exist in the model space -- or, on an
    // unconverged diagnostics record, that nothing was measured.
    double valence_band_max = std::numeric_limits<double>::quiet_NaN();
    double conduction_band_min = std::numeric_limits<double>::quiet_NaN();
    double indirect_gap = std::numeric_limits<double>::quiet_NaN();
    double direct_gap = std::numeric_limits<double>::quiet_NaN();
    // Distance to the next entirely empty state (occupancy partition).
    // NOT the band gap -- see KPointBandEdges (issue #426 / sec8-r4 F1).
    double gap_above_fermi_manifold =
        std::numeric_limits<double>::quiet_NaN();
    // Structural gapless flag: E_F pinned inside a partially filled
    // manifold, or a measured indirect_gap <= 0. Screen on this.
    bool is_metallic = false;
    int valence_band_max_k = -1;
    int conduction_band_min_k = -1;
    int direct_gap_k = -1;
    std::vector<double> valence_max_per_k;
    std::vector<double> conduction_min_per_k;
    Eigen::VectorXd charges;
    int n_basis = 0;
    int n_occ = 0;
    int n_kpoints = 0;
    int n_iter = 0;
    bool converged = false;
};

void validate_scc_dftb_kpoint_options(
    const SCCOptions& options,
    const char* caller);

KPointSCCDFTBResult run_scc_dftb_kpoints(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    const SCCOptions& scc_opts = {},
    double cutoff_bohr = 15.0,
    const KPointOccupationOptions& occupation_options = {});

}  // namespace semiempirical
}  // namespace vibeqc
