#pragma once
#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace vibeqc {
// Greiner et al., JCTC 22, 881 (2026), doi:10.1021/acs.jctc.5c01576.
// The generic adapter owns no chemistry. Updates must commit atomically;
// trials and Hessian actions must leave the accepted reference unchanged.
struct OpenTrustRegionOptions {
    std::string subsystem_solver = "davidson";
    double gradient_rms_tolerance = 1e-8;
    double initial_trust_radius = 0.4;
    int max_micro_iterations = 50;
    int seed = 42;
    bool line_search = false;
    // "check" verifies the final manifold; "follow" additionally asks upstream
    // to escape saddles. Upstream always checks/follows an initial saddle.
    std::string stability = "check";
    int stability_max_iterations = 100;
    double stability_tolerance = 1e-4;
    double stability_residual_tolerance = 1e-8;
};
struct OrbitalModel {
    double energy = 0.0;
    Eigen::VectorXd gradient, diagonal;
};
struct OrbitalObjective {
    virtual ~OrbitalObjective() = default;
    virtual int size() const = 0;
    virtual OrbitalModel update(const Eigen::VectorXd&) = 0;
    virtual double trial(const Eigen::VectorXd&) = 0;
    virtual Eigen::VectorXd hessian(const Eigen::VectorXd&) = 0;
    virtual bool converged() const { return false; }
};
struct OpenTrustRegionReport {
    std::string backend; // empty unless executed
    std::string version;
    std::string subsystem_solver;
    std::string stability_policy;
    std::string manifold;
    std::string termination;
    std::string callback_error;
    int error_code = 0;
    int accepted_evaluations = 0;
    int trial_evaluations = 0;
    int response_evaluations = 0;
    // Upstream does not expose rejected macro/micro iteration counts in C.
    int macro_iterations = -1;
    int micro_iterations = -1;
    int fock_evaluations = 0;
    double final_residual = 0.0;
    double gradient_rms = std::numeric_limits<double>::quiet_NaN();
    bool energy_converged = false;
    bool gradient_converged = false;
    bool stability_checked = false;
    bool stability_converged = false;
    bool stable = false;
    double stability_threshold = 0.0;
    std::vector<std::string> log;
};
bool has_opentrustregion();
void validate_opentrustregion(const OpenTrustRegionOptions& options, int max_iter);
OpenTrustRegionReport optimize_orbitals(OrbitalObjective& objective,
    const OpenTrustRegionOptions& options, int max_iter);
} // namespace vibeqc
