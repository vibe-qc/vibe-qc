// Native finite-difference batches for full-k DFTB0 and SCC-DFTB.

#pragma once

#include <cstddef>

#include <Eigen/Dense>

#include "vibeqc/semiempirical/kpoints_dftb0.hpp"
#include "vibeqc/semiempirical/kpoints_scc_dftb.hpp"

namespace vibeqc {
namespace semiempirical {

struct KPointDFTBFDBatchOptions {
    double coordinate_step = 0.001;
    double strain_step = 0.001;
    bool compute_gradient = true;
    bool compute_stress = true;
    bool strain_positions = true;
};

struct KPointDFTBFDBatchResult : ParameterIdentifiedResult {
    Eigen::MatrixXd gradient;
    Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
    int energy_evaluations = 0;
    int system_workspace_copies = 0;
    int kmesh_workspace_copies = 0;
    int lattice_cell_setups = 0;
    std::size_t workspace_bytes = 0;
    bool memory_counters_complete = false;
    bool differentiated_free_energy = false;
};

KPointDFTBFDBatchResult compute_dftb0_kpoints_fd_batch(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    double cutoff_bohr = 15.0,
    const KPointDFTBFDBatchOptions& fd_options = {},
    const KPointOccupationOptions& occupation_options = {});

KPointDFTBFDBatchResult compute_scc_dftb_kpoints_fd_batch(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    const SCCOptions& scc_options = {},
    double cutoff_bohr = 15.0,
    const KPointDFTBFDBatchOptions& fd_options = {},
    const KPointOccupationOptions& occupation_options = {});

}  // namespace semiempirical
}  // namespace vibeqc
