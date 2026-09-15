#pragma once

// Actual finite-Gaussian-source TNO triples bridge, under development.
// The converged pair-space CCSD and its original MP2 pair-frame owner are
// supplied together with their actual-source triple spaces. This bridge
// produces the published locally projected moments and solves every retained
// occupied coupling. It does not replace them by T0 or supplied dense arrays.
#include <optional>
#include "vibeqc/periodic_gaussian_triple_spaces.hpp"
#include "vibeqc/bounded_restricted_local_triples_moments.hpp"
#include "vibeqc/bounded_restricted_local_triples_solver.hpp"

namespace vibeqc {

struct PeriodicGaussianLocalTriplesOptions {
    BoundedRestrictedLocalTriplesMomentsOptions moments;
    BoundedRestrictedLocalTriplesSolverOptions solver;
    // Physical Frobenius norm ||ORIGINAL Fov||_F, finite nonnegative and
    // explicit. Accepted Fov is replaced by zero ONLY for the Brillouin
    // triples formula; original CCSD/reference energies remain unchanged.
    double maximum_brillouin_projection_norm = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicGaussianLocalTriplesLiveInventory {
    // Beyond complete borrowed reference/basis/provider/MP2/CCSD/TNO owners.
    std::uint64_t other_live_numerical_bytes_per_worker = 0;
    std::uint64_t other_live_control_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianLocalTriplesCaps {
    BoundedRestrictedLocalTriplesMomentsCaps moments;
    BoundedRestrictedLocalTriplesSolverCaps solver;
    std::uint64_t maximum_pair_count = 0, maximum_triple_count = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes_per_worker = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0, maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_common_integral_calls = 0, maximum_progress_callbacks = 0, maximum_work_units = 0;
};
struct PeriodicGaussianLocalTriplesPlan {
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, pair_count = 0, triple_count = 0;
    std::uint64_t maximum_retained_rank = 0, total_retained_rank = 0, total_amplitude_elements = 0;
    bool moments_required = false;
    std::uint64_t complete_borrowed_owner_numerical_bytes = 0, complete_borrowed_owner_control_bytes = 0;
    std::uint64_t projected_zero_fov_bytes = 0, reader_table_bytes = 0, triple_table_bytes = 0;
    std::uint64_t reader_borrowed_numerical_bytes = 0, solver_borrowed_numerical_bytes = 0;
    std::uint64_t maximum_moment_transient_numerical_bytes = 0;
    // Explicit upper-planner reservations, NOT owned or borrowed payload.
    // Shared owner roles are counted once; max(delta_reader,delta_solver)
    // covers the two separate uniform-rank upper-admission identities.
    std::uint64_t reader_borrowed_rank_padding_bytes = 0, solver_borrowed_rank_padding_bytes = 0;
    std::uint64_t rank_padding_reservation_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, control_storage_reservation_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t common_integral_calls_upper_bound = 0, progress_callback_upper_bound = 0;
    std::uint64_t driver_work_units = 0, work_units_upper_bound = 0;
    BoundedRestrictedPairCCSDAmplitudesMemoryPlan reader_upper;
    BoundedRestrictedLocalTriplesMomentsMemoryPlan moments_upper;
    BoundedRestrictedLocalTriplesSolverMemoryPlan solver_upper;
};
enum class PeriodicGaussianLocalTriplesStage : std::uint32_t {
    Begin = 0, Solver = 1, Finished = 2
};
struct PeriodicGaussianLocalTriplesProgress {
    PeriodicGaussianLocalTriplesStage stage = PeriodicGaussianLocalTriplesStage::Begin;
    std::uint64_t callback_count = 0, common_integral_calls = 0;
    BoundedRestrictedLocalTriplesSolverProgress solver;
};
using PeriodicGaussianLocalTriplesCallback = void (*)(const PeriodicGaussianLocalTriplesProgress&,void*);
struct PeriodicGaussianLocalTriplesDiagnostics {
    std::uint64_t completed_progress_callbacks = 0, completed_common_integral_calls = 0;
    double brillouin_projection_frobenius_norm = 0;
    bool complete_common_finite_torus_basis = false;
    bool all_retained_triples_full_common_rank = false;
};
class PeriodicGaussianLocalTriplesResult {
public:
    PeriodicGaussianLocalTriplesResult(const PeriodicGaussianLocalTriplesResult&) = delete;
    PeriodicGaussianLocalTriplesResult& operator=(const PeriodicGaussianLocalTriplesResult&) = delete;
    PeriodicGaussianLocalTriplesResult(PeriodicGaussianLocalTriplesResult&&) noexcept = default;
    PeriodicGaussianLocalTriplesResult& operator=(PeriodicGaussianLocalTriplesResult&&) noexcept = delete;
    const PeriodicGaussianLocalTriplesPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianLocalTriplesDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const BoundedRestrictedLocalTriplesSolverResult& solver() const;
    bool converged() const;
    bool periodic_energy_per_cell() const;
    double triples_energy_per_cell() const;
    double correlation_energy_per_cell() const;
    double total_energy_per_cell() const;
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& ccsd_identity_sha256() const noexcept { return ccsd_; }
    const std::string& triple_spaces_identity_sha256() const noexcept { return spaces_; }
    const std::string& consumed_moments_receipt_sha256() const noexcept { return moments_; }
private:
    PeriodicGaussianLocalTriplesResult() = default;
    PeriodicGaussianLocalTriplesPlan memory_;
    PeriodicGaussianLocalTriplesDiagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::optional<BoundedRestrictedLocalTriplesSolverResult> solver_;
    double ccsd_correlation_energy_ = 0;
    std::string identity_,ccsd_,spaces_,moments_;
    friend PeriodicGaussianLocalTriplesResult run_periodic_gaussian_local_triples(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
        const PeriodicGaussianPairCCSDResult&,const PeriodicGaussianTripleSpaces&,
        const PeriodicGaussianLocalTriplesOptions&,const PeriodicGaussianLocalTriplesLiveInventory&,
        const PeriodicGaussianLocalTriplesCaps&,PeriodicGaussianLocalTriplesCallback,void*);
};

PeriodicGaussianLocalTriplesPlan plan_periodic_gaussian_local_triples(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
    const PeriodicGaussianPairCCSDResult&,const PeriodicGaussianTripleSpaces&,
    const PeriodicGaussianLocalTriplesOptions&,const PeriodicGaussianLocalTriplesLiveInventory&,
    const PeriodicGaussianLocalTriplesCaps&);
PeriodicGaussianLocalTriplesResult run_periodic_gaussian_local_triples(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
    const PeriodicGaussianPairCCSDResult&,const PeriodicGaussianTripleSpaces&,
    const PeriodicGaussianLocalTriplesOptions&,const PeriodicGaussianLocalTriplesLiveInventory&,
    const PeriodicGaussianLocalTriplesCaps&,PeriodicGaussianLocalTriplesCallback = nullptr,void* = nullptr);

} // namespace vibeqc
