// Private, bounded diagnostics for recovery-state causality; not a method API.
#pragma once
#include "vibeqc/uks.hpp"
#include <map>
#include <string>
#include <vector>
namespace vibeqc::detail {
struct UKSRecoveryFrame {
    int iteration = 0;
    std::string phase = "iteration_evaluation";
    std::map<std::string, double> values;
    std::map<std::string, Eigen::MatrixXd> matrices;
};
struct UKSRecoveryDiagnostic {
    int event_iteration = 0, steps = 0, n_alpha = 0, n_beta = 0;
    double nuclear_energy = 0.0;
    UKSOptions requested_options, effective_options;
    Eigen::MatrixXd overlap, hcore, grid_points;
    Eigen::VectorXd grid_weights;
    UKSRecoveryFrame checkpoint;
    std::vector<UKSRecoveryFrame> prefix;
    std::vector<std::vector<UKSRecoveryFrame>> branches, certifications;
    std::vector<UKSRecoveryFrame> prefix_certifications, branch_results;
    std::vector<std::string> branch_termination, branch_errors;
    std::vector<Eigen::MatrixXd> diis_focks, diis_errors;
    std::vector<Eigen::MatrixXd> initial_branch_density_alpha, initial_branch_density_beta;
};
UKSRecoveryDiagnostic diagnose_uks_recovery(
    const Molecule&, const BasisSet&, const UKSOptions&, int event_iteration, int steps);
} // namespace vibeqc::detail
