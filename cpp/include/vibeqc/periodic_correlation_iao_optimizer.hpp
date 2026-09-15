#pragma once

/// Bounded reference IAO-PM steepest-ascent optimizer with Armijo search.
/// Zhu and Tew, doi:10.1021/acs.jpca.4c04555, Sec.2.4, Eqs.26-29.
/// U=U_seed V uses an actual TR-audited native diabatic seed. V(-k)=V(k)*
/// and V(TRIM) is real; the raw incoming Bloch gauges need not be real.
/// H=skew(U^H E), followed by the full-mesh orthogonal TR projection,
/// generates V_new=V exp(step H). Its full-mesh Frobenius norm is the
/// convergence criterion, not the nonzero ambient Euclidean norm.
/// This is neither l-BFGS nor a production scaling claim. It certifies the
/// stated algebraic audits, not an independently supplied overlap source.

#include "vibeqc/periodic_correlation_diabatic_seed.hpp"
#include "vibeqc/periodic_correlation_iao_pm.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicCorrelationIAOOptimizerContractVersion = 1;

enum class PeriodicCorrelationIAOOptimizerStatus : std::uint32_t {
    Converged = 0, IterationLimit = 1, LineSearchFailed = 2, WorkLimit = 3
};
enum class PeriodicCorrelationIAOOptimizerEvent : std::uint32_t {
    Initial = 0, Accepted = 1, Rejected = 2, Finished = 3
};

struct PeriodicCorrelationIAOOptimizerOptions {
    std::uint64_t maximum_iterations = 0, maximum_line_search_trials = 0;
    double initial_step = 0.0, minimum_step = 0.0;
    double backtracking_factor = 0.0, armijo_fraction = 0.0;
    double riemannian_gradient_tolerance = 0.0;
    double absolute_tolerance = 0.0, relative_tolerance = 0.0;
    std::uint64_t jacobi_max_sweeps = 0;
    double jacobi_relative_tolerance = 0.0;
};
struct PeriodicCorrelationIAOOptimizerCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_work_units = 0;
};

struct PeriodicCorrelationIAOOptimizerMemoryPlan {
    std::uint64_t n_points = 0, n_basis = 0, n_active = 0, n_minimal = 0;
    std::uint64_t output_numerical_bytes = 0, fixed_optimizer_bytes = 0;
    std::uint64_t exponential_workspace_bytes = 0, pm_owned_peak_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, borrowed_seed_numerical_bytes = 0;
    std::uint64_t live_iao_numerical_bytes = 0, borrowed_owner_pointer_bytes = 0;
    std::uint64_t required_node_memory_bytes = 0;
    std::uint64_t preflight_work_units = 0, projection_work_units = 0;
    std::uint64_t trial_retraction_work_units = 0, pm_work_units = 0;
    std::uint64_t initial_work_units = 0, work_units_per_trial = 0, maximum_work_units = 0;
};

struct PeriodicCorrelationIAOOptimizerDiagnostics {
    std::uint64_t charged_work_units = 0, objective_evaluations = 0;
    std::uint64_t accepted_steps = 0, line_search_trials = 0, rejected_trials = 0, jacobi_sweeps = 0;
    double initial_objective = 0.0, final_objective = 0.0;
    double riemannian_gradient_norm = 0.0, last_accepted_step = 0.0;
    double maximum_relative_unitarity_residual = 0.0, maximum_raw_unitarity_residual = 0.0;
    double maximum_charge_frame_time_reversal_residual = 0.0;
    double maximum_physical_time_reversal_residual = 0.0, maximum_trim_physical_imaginary_magnitude = 0.0;
    double maximum_trim_relative_imaginary_correction = 0.0;
    double maximum_exponential_eigen_relative_residual = 0.0;
    double maximum_exponential_eigenvector_unitarity_residual = 0.0;
    double maximum_tangent_slope_residual = 0.0;
};

/// Scalar-only live progress; objective/norm always describe the last
/// ACCEPTED state. trial_objective is meaningful only for Accepted/Rejected.
struct PeriodicCorrelationIAOOptimizerProgress {
    PeriodicCorrelationIAOOptimizerEvent event = PeriodicCorrelationIAOOptimizerEvent::Initial;
    PeriodicCorrelationIAOOptimizerStatus status = PeriodicCorrelationIAOOptimizerStatus::IterationLimit;
    std::uint64_t accepted_steps = 0, objective_evaluations = 0, line_search_trials = 0, charged_work_units = 0;
    double objective = 0.0, riemannian_gradient_norm = 0.0, step = 0.0, trial_objective = 0.0;
};
using PeriodicCorrelationIAOOptimizerCallback = void (*)(
    const PeriodicCorrelationIAOOptimizerProgress&, void*);

class PeriodicCorrelationIAOOptimizerResult {
public:
    PeriodicCorrelationIAOOptimizerResult(const PeriodicCorrelationIAOOptimizerResult&) = delete;
    PeriodicCorrelationIAOOptimizerResult& operator=(const PeriodicCorrelationIAOOptimizerResult&) = delete;
    PeriodicCorrelationIAOOptimizerResult(PeriodicCorrelationIAOOptimizerResult&&) noexcept = default;
    PeriodicCorrelationIAOOptimizerResult& operator=(PeriodicCorrelationIAOOptimizerResult&&) noexcept = default;
    ~PeriodicCorrelationIAOOptimizerResult() = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationIAOOptimizerContractVersion; }
    PeriodicCorrelationIAOOptimizerStatus status() const noexcept { return status_; }
    bool converged() const noexcept { return status_ == PeriodicCorrelationIAOOptimizerStatus::Converged; }
    const PeriodicCorrelationIAOOptimizerMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationIAOOptimizerDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicRestrictedMeanFieldState& state() const;
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }
    const std::string& seed_identity_sha256() const noexcept { return seed_identity_; }
    const std::string& source_identity_sha256() const noexcept { return source_identity_; }
    const std::string& gauge_payload_sha256() const noexcept { return gauge_digest_; }
    const std::string& optimizer_identity_sha256() const noexcept { return identity_digest_; }
    const std::complex<double>* gauges_data() const;
    std::complex<double> gauge(std::size_t point, std::size_t row, std::size_t column) const;
    std::uint64_t active_band(std::size_t point, std::size_t active) const;
private:
    PeriodicCorrelationIAOOptimizerResult() = default;
    void require_live() const;
    PeriodicCorrelationIAOOptimizerStatus status_ = PeriodicCorrelationIAOOptimizerStatus::IterationLimit;
    PeriodicCorrelationIAOOptimizerMemoryPlan memory_;
    PeriodicCorrelationIAOOptimizerDiagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::vector<std::complex<double>> gauges_;
    std::vector<std::uint64_t> active_indices_;
    std::string allocation_identity_, seed_identity_, source_identity_, gauge_digest_, identity_digest_;
    friend PeriodicCorrelationIAOOptimizerResult optimize_periodic_correlation_iao_pm(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationDiabaticSeed&,
        const PeriodicCorrelationBlochIAO* const*, std::size_t,
        const PeriodicCorrelationIAOPMOptions&, const PeriodicCorrelationIAOOptimizerOptions&,
        const PeriodicCorrelationIAOOptimizerCaps&, PeriodicCorrelationIAOOptimizerCallback, void*);
};

PeriodicCorrelationIAOOptimizerMemoryPlan plan_periodic_correlation_iao_optimizer(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationDiabaticSeed&,
    std::uint64_t n_minimal, const PeriodicCorrelationIAOOptimizerOptions&);

/// Positive exact memory and initial-work caps precede any numerical array
/// allocation. Later work/line/iteration limits return the last accepted
/// audited state, with converged=false. No unevaluated trial is exposed.
/// Invalid/numerically failed trials and callback exceptions abort, throwing
/// without a result. No retries conceal a violated algebraic audit.
/// All inputs, controls and callback context stay immutable during the call;
/// other live numerical storage (including callback-owned storage) belongs in
/// the admitted external inventory. The result retains only its state owner.
PeriodicCorrelationIAOOptimizerResult optimize_periodic_correlation_iao_pm(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationDiabaticSeed&,
    const PeriodicCorrelationBlochIAO* const* points, std::size_t point_count,
    const PeriodicCorrelationIAOPMOptions&, const PeriodicCorrelationIAOOptimizerOptions&,
    const PeriodicCorrelationIAOOptimizerCaps&,
    PeriodicCorrelationIAOOptimizerCallback callback, void* callback_context);

}  // namespace vibeqc
