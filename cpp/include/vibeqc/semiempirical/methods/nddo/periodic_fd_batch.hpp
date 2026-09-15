// Native finite-difference batches for periodic Gamma-point NDDO methods.

#pragma once

#include <cstddef>

#include <Eigen/Dense>

#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/methods/nddo/periodic_omx.hpp"
#include "vibeqc/semiempirical/methods/nddo/periodic_pm6.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

struct PeriodicNDDOFDBatchOptions {
    double coordinate_step = 0.001;
    double strain_step = 0.001;
    bool compute_gradient = true;
    bool compute_stress = true;
    bool strain_positions = true;
};

struct PeriodicNDDOFDBatchResult : ParameterIdentifiedResult {
    Eigen::MatrixXd gradient;
    Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
    int energy_evaluations = 0;
    int system_workspace_copies = 0;
    int lattice_cell_setups = 0;
    std::size_t workspace_bytes = 0;
    bool memory_counters_complete = false;
};

PeriodicNDDOFDBatchResult compute_pm6_gamma_fd_batch(
    const PeriodicSystem& system,
    const PM6ParameterSet& params,
    const PeriodicPM6Options& scf_options = {},
    const PeriodicNDDOFDBatchOptions& fd_options = {});

PeriodicNDDOFDBatchResult compute_omx_gamma_fd_batch(
    const PeriodicSystem& system,
    const OMxParameterSet& params,
    const PeriodicOMxOptions& scf_options = {},
    const PeriodicNDDOFDBatchOptions& fd_options = {});

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
