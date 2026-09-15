// Fail-closed GFN2-xTB k-point API (issue #351).
//
// A one-point Gamma mesh delegates to the supported Gamma driver. General
// k-point meshes and band paths are rejected until the implementation has
// full complex Bloch phases, phase-resolved energy contraction, and the same
// AES Hamiltonian/energy terms as the Gamma model.

#pragma once

#include <Eigen/Dense>
#include <limits>
#include <vector>

#include "vibeqc/bloch.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

// ---------------------------------------------------------------------------
// Multi-k GFN2-xTB result
// ---------------------------------------------------------------------------

struct KPointGFN2Result : ParameterIdentifiedResult {
    double energy = std::numeric_limits<double>::quiet_NaN();
    double free_energy = std::numeric_limits<double>::quiet_NaN();
    double e_electronic = std::numeric_limits<double>::quiet_NaN();
    double e_repulsive = std::numeric_limits<double>::quiet_NaN();
    double e_scc = std::numeric_limits<double>::quiet_NaN();
    double e_band0 = std::numeric_limits<double>::quiet_NaN();
    double e_aes = std::numeric_limits<double>::quiet_NaN();
    double e_3rd = std::numeric_limits<double>::quiet_NaN();
    double fermi_level = std::numeric_limits<double>::quiet_NaN();
    double entropy = std::numeric_limits<double>::quiet_NaN();
    double smearing_temperature = 0.0;
    std::vector<double> band_energies;        // all ε_i(k) flattened
    std::vector<Eigen::VectorXd> eps_per_k;   // ε(k) per k-point
    std::vector<Eigen::VectorXd> occupations_per_k;  // n_i(k) actually used
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
    Eigen::MatrixXd density;                  // weighted real-space density
    Eigen::MatrixXd overlap_gamma;            // real-space S summed over cells
    Eigen::VectorXd charges;                  // atomic Mulliken charges
    Eigen::VectorXd dq_shell;                 // shell charges (for gradient)
    int n_basis = 0;
    int n_occ = 0;
    int n_kpoints = 0;
    int n_shells = 0;
    int n_iter = 0;
    bool converged = false;
};

// ---------------------------------------------------------------------------
// Run the supported Gamma model through a one-point Gamma mesh spelling.
// Any non-Gamma mesh throws explicitly (issue #351).
// ---------------------------------------------------------------------------

KPointGFN2Result run_gfn2_xtb_kpoints(
    const PeriodicSystem& system,
    const GFN2ParameterSet& params,
    const BlochKMesh& kmesh,
    const XTBSccOptions& scc_opts = {},
    double cutoff_bohr = 15.0);

// ---------------------------------------------------------------------------
// Band paths fail closed until the non-Gamma GFN2 model is complete
// (issue #351). Both overloads are retained for API compatibility.
// ---------------------------------------------------------------------------

KPointGFN2Result run_gfn2_xtb_bandpath(
    const PeriodicSystem& system,
    const GFN2ParameterSet& params,
    const std::vector<Eigen::Vector3d>& kpath,
    const BlochKMesh& reference_mesh,
    const XTBSccOptions& scc_opts = {},
    double cutoff_bohr = 15.0);

KPointGFN2Result run_gfn2_xtb_bandpath(
    const PeriodicSystem& system,
    const GFN2ParameterSet& params,
    const std::vector<Eigen::Vector3d>& kpath,
    double cutoff_bohr = 15.0);

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
