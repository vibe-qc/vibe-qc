// Periodic DFTB0 with k-point sampling (beyond Gamma-point).
//
// Extends periodic DFTB0 to general k-point meshes. For each k:
//   H_{μν}(k) = Σ_g H⁰_{μν}(g) · exp(i k·g)
//   S_{μν}(k) = Σ_g S_{μν}(g) · exp(i k·g)
//
// The total energy per unit cell is:
//   E = Σ_k w_k Σ_i^{occ} n_i ε_i(k) + E_rep
//
// Native k-point prototype. The production route remains gated until k-mesh,
// runner/output, analytic-derivative, and external validation are complete.

#pragma once

#include <complex>
#include <limits>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/bloch.hpp"
#include "vibeqc/basis.hpp"
#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/parameters.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// k-point DFTB0 result
// ---------------------------------------------------------------------------

struct KPointDFTB0Result : ParameterIdentifiedResult {
    double energy = 0.0;                      // total energy per unit cell
    double free_energy = 0.0;                 // Mermin free energy per unit cell
    double e_electronic = 0.0;                // Σ_k w_k Σ_i n_i(k) ε_i(k)
    double e_repulsive = 0.0;
    double fermi_level = 0.0;
    double entropy = 0.0;
    double smearing_temperature = 0.0;
    std::vector<double> band_energies;        // all ε_i(k) flattened: k0_band0, k0_band1, ...
    std::vector<Eigen::VectorXd> eps_per_k;   // ε(k) per k-point
    std::vector<Eigen::VectorXd> occupations_per_k;
    // Measured band edges + gaps from the occupations actually used
    // (issue #426); convention documented on KPointBandEdges. NaN / -1 /
    // empty means the edge does not exist in the model space.
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
    int n_basis = 0;
    int n_occ = 0;
    int n_kpoints = 0;
};

struct DFTB0LatticeBlocks {
    std::vector<LatticeCell> cells;
    std::vector<Eigen::MatrixXd> h0;
    std::vector<Eigen::MatrixXd> overlap;
};

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

// Run periodic DFTB0 on a k-point mesh (Monkhorst-Pack or custom).
KPointDFTB0Result run_dftb0_kpoints(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    double cutoff_bohr = 15.0,
    const KPointOccupationOptions& occupation_options = {});

// Internal infrastructure shared by k-point DFTB0 and SCC-DFTB.
DFTB0LatticeBlocks build_h0_s_per_cell(
    const BasisSet&,
    const PeriodicSystem&,
    const SemiempiricalParameters&,
    double);

int validate_dftb_kpoint_request(
    const PeriodicSystem&,
    const SemiempiricalParameters&,
    const BlochKMesh&,
    double cutoff_bohr,
    const char* caller);

// Run DFTB0 along a band-structure path with independent per-point filling.
// The aggregate energy is diagnostic only; path points are not a quadrature.
KPointDFTB0Result run_dftb0_bandpath(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const std::vector<Eigen::Vector3d>& kpath,
    double cutoff_bohr = 15.0);

}  // namespace semiempirical
}  // namespace vibeqc
