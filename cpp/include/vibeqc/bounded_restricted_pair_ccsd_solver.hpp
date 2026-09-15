#pragma once

// Ragged singles/pair-space Galerkin CCSD REFERENCE. The full common-frame
// Stanton residual is evaluated from scalar amplitude accessors, and only
// the completed residual is projected. This is not the production DLPNO
// restricted interaction/domain approximation or a periodic Hamiltonian
// certificate. See doi:10.1063/1.460620 Eqs.1-13 and
// doi:10.1063/1.4773581 Sec.II.B.4, Eq.26 and the distinction in Eq.29.
#include <limits>
#include <string>
#include <vector>
#include "vibeqc/bounded_restricted_pair_ccsd_amplitudes.hpp"
#include "vibeqc/bounded_restricted_ccsd_energy.hpp"
#include "vibeqc/bounded_restricted_pair_ccsd_particle_hole.hpp"

namespace vibeqc {

struct BoundedRestrictedPairCCSDSolverMemoryPlan;

using BoundedRestrictedPairCCSDParticleHoleVisitor = void (*)(
    const BoundedRestrictedPairCCSDParticleHoleResult&, const std::string& reader_snapshot_identity, void*);
// Serial immutable producer for ONLY the complete bare particle-hole group.
// It must synchronously invoke the visitor exactly once for the supplied
// current Reader/target, including target rank zero; never retain the visitor
// or any Reader pointer. The echoed snapshot SHA is a DECLARED sequencing
// token, not a physical-source or equation certificate. No Python callback
// is registered as this producer. Native outer owners supply actual sources.
struct BoundedRestrictedPairCCSDParticleHoleProducer {
    // Exact admitted enclosing plan, borrowed only for this synchronous call.
    // This exposes authoritative ownership counts, not mutable numerical views.
    void (*produce)(const BoundedRestrictedPairCCSDAmplitudes&,
        const BoundedRestrictedPairCCSDSolverMemoryPlan&, std::uint64_t target_i,
        std::uint64_t target_j, std::uint64_t evaluated_snapshot_index,
        BoundedRestrictedPairCCSDParticleHoleVisitor, void* visitor_context, const void* producer_context) = nullptr;
    const void* context = nullptr;
    std::array<char,64> identity_sha256{};
    // Additional owners only: exclude solver input/current/candidate/Fock/
    // Reader roles and the ordinary integral-provider owners already counted.
    // Equal contents do not establish aliasing. These are caller declarations.
    std::uint64_t additional_retained_numerical_bytes = 0;
    // Includes the returned PH result AND every simultaneously live native
    // factory intermediate. No raw common residual lives during this phase.
    std::uint64_t maximum_transient_numerical_bytes_per_target = 0;
    std::uint64_t additional_control_storage_bytes = 0;
    std::uint64_t maximum_work_units_per_target = 0; // positive, includes provider's own audits
};

struct BoundedRestrictedPairCCSDSolverInput {
    BoundedRestrictedPairCCSDAmplitudesInput initial;
    BoundedRestrictedCCSDRealView f_oo, f_vv, f_ov;
};
struct BoundedRestrictedPairCCSDSolverOptions {
    std::uint64_t maximum_iterations = 0; // evaluated immutable snapshots
    double denominator_floor = 0.0;
    double singles_residual_tolerance = 0.0, doubles_residual_tolerance = 0.0;
    double energy_tolerance = 0.0, coefficient_orthogonality_tolerance = 0.0;
    // Explicit finite nonnegative cumulative norm budget for symmetrizing
    // diagonal-pair Jacobi candidates. RAW projected residuals determine
    // convergence; no asymmetric component is removed from that test.
    double maximum_diagonal_update_antisymmetry_norm = std::numeric_limits<double>::quiet_NaN();
    // Positive caller declaration, not introspection of opaque ERI callbacks.
    std::uint64_t maximum_integral_work_units_per_call = 0;
};
struct BoundedRestrictedPairCCSDSolverInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t other_live_bytes_per_replica = 0;
    std::uint64_t fixed_backend_margin_bytes_per_replica = 0;
};
struct BoundedRestrictedPairCCSDSolverCaps {
    std::uint64_t maximum_occupied_count = 0, maximum_common_virtual_dimension = 0;
    std::uint64_t maximum_pair_count = 0, maximum_singles_rank = 0, maximum_pair_rank = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0, maximum_integral_calls = 0;
    std::uint64_t maximum_singles_calls = 0, maximum_doubles_calls = 0, maximum_work_units = 0;
    std::uint64_t maximum_particle_hole_calls = 0; // split overload only
};
struct BoundedRestrictedPairCCSDSolverMemoryPlan {
    BoundedRestrictedPairCCSDAmplitudesMemoryPlan amplitudes;
    BoundedRestrictedCCSDTargetAccessorMemoryPlan target;
    BoundedRestrictedCCSDEnergyMemoryPlan energy;
    bool uniform_rank_upper_bound = false;
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0, pair_count = 0;
    std::uint64_t maximum_singles_rank = 0, maximum_pair_rank = 0, maximum_iterations = 0;
    std::uint64_t total_singles_elements = 0, total_doubles_elements = 0;
    std::uint64_t amplitude_snapshot_bytes = 0, candidate_snapshot_bytes = 0;
    std::uint64_t projected_fock_diagonal_bytes = 0, retained_record_bytes = 0;
    std::uint64_t owned_snapshot_table_bytes = 0, borrowed_input_table_bytes = 0;
    std::uint64_t borrowed_initial_numerical_bytes = 0, borrowed_fock_bytes = 0;
    std::uint64_t provider_retained_numerical_bytes = 0, provider_transient_numerical_bytes = 0;
    std::uint64_t target_owned_peak_bytes = 0, output_numerical_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, control_storage_reservation_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t metadata_work_units = 0, validation_work_units = 0, initialization_work_units = 0;
    std::uint64_t projection_work_units_per_snapshot = 0, work_units_per_snapshot = 0;
    std::uint64_t target_evaluations_upper_bound = 0, integral_calls_upper_bound = 0;
    std::uint64_t singles_calls_upper_bound = 0, doubles_calls_upper_bound = 0, work_units_upper_bound = 0;
    bool split_bare_particle_hole = false;
    std::uint64_t particle_hole_retained_numerical_bytes = 0, particle_hole_transient_numerical_bytes = 0;
    std::uint64_t particle_hole_control_storage_bytes = 0, particle_hole_calls_upper_bound = 0;
    std::uint64_t remaining_target_phase_bytes = 0, local_particle_hole_phase_bytes = 0;
    std::uint64_t particle_hole_work_units_per_target = 0, particle_hole_guard_work_units_per_target = 0;
    std::uint64_t omitted_bare_integral_calls_upper_bound = 0;
};
struct BoundedRestrictedPairCCSDSolverProgress {
    std::uint64_t iteration = 0, target_evaluations = 0, integral_calls = 0;
    std::uint64_t singles_calls = 0, doubles_calls = 0, input_checks = 0, charged_work_units = 0;
    bool has_previous_energy = false, converged = false;
    double correlation_energy = 0.0, energy_change = 0.0;
    double singles_max_residual = 0.0, doubles_max_residual = 0.0;
    double singles_residual_norm = 0.0, doubles_residual_norm = 0.0;
    double maximum_diagonal_update_antisymmetry_norm = 0.0;
    std::uint64_t particle_hole_calls = 0, particle_hole_visits = 0, particle_hole_source_slots = 0;
    // Conservative producer work PLUS enclosing before/after snapshot guards.
    std::uint64_t charged_particle_hole_work_units = 0;
};
using BoundedRestrictedPairCCSDSolverCallback = void (*)(const BoundedRestrictedPairCCSDSolverProgress&, void*);

struct BoundedRestrictedPairCCSDSolverPayloadValidationPlan {
    std::uint64_t n_occupied = 0, pair_count = 0, record_count = 0;
    std::uint64_t singles_elements = 0, doubles_elements = 0, numerical_lanes = 0;
    std::uint64_t retained_numerical_bytes = 0, retained_record_bytes = 0;
    std::uint64_t payload_message_bytes = 0, fixed_codec_payload_bytes = 0, fixed_control_storage_bytes = 0;
    std::uint64_t metadata_work_units = 0, payload_work_units = 0, work_units = 0;
};

class BoundedRestrictedPairCCSDSolverResult {
public:
    BoundedRestrictedPairCCSDSolverResult(const BoundedRestrictedPairCCSDSolverResult&) = delete;
    BoundedRestrictedPairCCSDSolverResult& operator=(const BoundedRestrictedPairCCSDSolverResult&) = delete;
    BoundedRestrictedPairCCSDSolverResult(BoundedRestrictedPairCCSDSolverResult&&) noexcept = default;
    BoundedRestrictedPairCCSDSolverResult& operator=(BoundedRestrictedPairCCSDSolverResult&&) noexcept = delete;
    const BoundedRestrictedPairCCSDSolverMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedRestrictedPairCCSDSolverProgress& final_snapshot() const noexcept { return snapshot_; }
    double minimum_denominator() const noexcept { return minimum_denominator_; }
    double maximum_denominator() const noexcept { return maximum_denominator_; }
    std::uint64_t singles_rank(std::uint64_t i) const;
    std::uint64_t pair_rank(std::uint64_t i, std::uint64_t j) const;
    double singles_amplitude(std::uint64_t i, std::uint64_t a) const;
    double doubles_amplitude(std::uint64_t i, std::uint64_t j, std::uint64_t a, std::uint64_t b) const;
    // Native immutable borrows for downstream TNO/moment producers. No
    // allocation or public Python escape; this owner stays alive and unmoved.
    // Pair borrow requires canonical i<=j; reverse scalar access stays above.
    BoundedRestrictedCCSDRealView stored_singles_view(std::uint64_t i) const;
    BoundedRestrictedCCSDRealView stored_pair_view(std::uint64_t i, std::uint64_t j) const;
    const std::string& input_identity_sha256() const noexcept { return input_identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    // Separate mode/source receipts. Legacy input and amplitude codecs stay
    // unchanged; empty strings on the original full-residual route.
    const std::string& split_execution_identity_sha256() const noexcept { return split_identity_; }
    const std::string& consumed_particle_hole_identity_sha256() const noexcept { return particle_hole_identity_; }
    bool particle_hole_physical_source_certified() const noexcept { return false; }
private:
    struct Record { std::uint64_t rank = 0, offset = 0; };
    BoundedRestrictedPairCCSDSolverResult() = default;
    BoundedRestrictedPairCCSDSolverMemoryPlan memory_;
    BoundedRestrictedPairCCSDSolverProgress snapshot_;
    double minimum_denominator_ = 0.0, maximum_denominator_ = 0.0;
    std::vector<Record> singles_, pairs_;
    std::vector<double> amplitudes_;
    std::string input_identity_, payload_, split_identity_, particle_hole_identity_;
    static BoundedRestrictedPairCCSDSolverResult solve_impl(
        const BoundedRestrictedPairCCSDSolverInput&, const BoundedRestrictedCCSDIntegralProvider&,
        const BoundedRestrictedPairCCSDSolverOptions&, const BoundedRestrictedPairCCSDSolverInventory&,
        const BoundedRestrictedPairCCSDSolverCaps&, const BoundedRestrictedPairCCSDParticleHoleProducer*,
        BoundedRestrictedPairCCSDSolverCallback, void*);
    friend BoundedRestrictedPairCCSDSolverResult bounded_restricted_pair_ccsd_solve(
        const BoundedRestrictedPairCCSDSolverInput&, const BoundedRestrictedCCSDIntegralProvider&,
        const BoundedRestrictedPairCCSDSolverOptions&, const BoundedRestrictedPairCCSDSolverInventory&,
        const BoundedRestrictedPairCCSDSolverCaps&, BoundedRestrictedPairCCSDSolverCallback, void*);
    friend BoundedRestrictedPairCCSDSolverResult bounded_restricted_pair_ccsd_solve(
        const BoundedRestrictedPairCCSDSolverInput&, const BoundedRestrictedCCSDIntegralProvider&,
        const BoundedRestrictedPairCCSDParticleHoleProducer&,
        const BoundedRestrictedPairCCSDSolverOptions&, const BoundedRestrictedPairCCSDSolverInventory&,
        const BoundedRestrictedPairCCSDSolverCaps&, BoundedRestrictedPairCCSDSolverCallback, void*);
    friend BoundedRestrictedPairCCSDSolverPayloadValidationPlan
        plan_bounded_restricted_pair_ccsd_solver_payload_validation(const BoundedRestrictedPairCCSDSolverResult&);
    friend void verify_bounded_restricted_pair_ccsd_solver_payload(
        const BoundedRestrictedPairCCSDSolverResult&, std::uint64_t maximum_work_units);
};

// O(1) count/owner-extent plan: no record traversal, floating payload scan,
// callback, or allocation. The verifier admits its positive work cap BEFORE
// traversing records/payload. It uses only the explicitly inventoried fixed
// codec/control allocation (including a 65-byte digest-string payload), never
// a size-dependent numerical allocation. It verifies the original codec exactly:
// domain/version, input_identity, all stored singles then canonical i<=j
// doubles (signed zero normalized). No snapshot or descriptor-table copy.
// The result must remain alive, unmoved and immutable throughout each call.
// This proves payload replay only, not convergence, original physical owners,
// integral correctness, or the unsealed progress/energy diagnostic fields.
BoundedRestrictedPairCCSDSolverPayloadValidationPlan
plan_bounded_restricted_pair_ccsd_solver_payload_validation(const BoundedRestrictedPairCCSDSolverResult&);
void verify_bounded_restricted_pair_ccsd_solver_payload(
    const BoundedRestrictedPairCCSDSolverResult&, std::uint64_t maximum_work_units);

// Count-only conservative plan without input tables, floating scans or
// callbacks. Both rank bounds may be zero. All occupied pairs are retained.
BoundedRestrictedPairCCSDSolverMemoryPlan plan_bounded_restricted_pair_ccsd_solver_upper(
    std::uint64_t occupied, std::uint64_t common_virtual_dimension,
    std::uint64_t maximum_singles_rank, std::uint64_t maximum_pair_rank,
    const BoundedRestrictedPairCCSDSolverOptions&, const BoundedRestrictedPairCCSDSolverInventory&,
    std::uint64_t integral_retained_bytes = 0, std::uint64_t integral_transient_bytes = 0);
// Metadata traversal is admitted before reading tables. Exact ragged caps
// are admitted before any floating payload scan, heap allocation or callback.
BoundedRestrictedPairCCSDSolverMemoryPlan plan_bounded_restricted_pair_ccsd_solver(
    const BoundedRestrictedPairCCSDSolverInput&, const BoundedRestrictedPairCCSDSolverOptions&,
    const BoundedRestrictedPairCCSDSolverInventory&, const BoundedRestrictedPairCCSDSolverCaps&,
    std::uint64_t integral_retained_bytes = 0, std::uint64_t integral_transient_bytes = 0);
// Copies the initial ragged amplitudes; denominators are the diagonals of
// C^T ORIGINAL Fvv C minus original Foo diagonals, not a replacement Fock
// operator. Singles frames need not equal diagonal-double frames. The full
// common-space energy includes the unprojected t1*t1 product. Jacobi only,
// no shifts/damping/DIIS. Callback exceptions/mutated initial inputs publish
// no result. Exhaustion returns the last evaluated (unconverged) snapshot.
BoundedRestrictedPairCCSDSolverResult bounded_restricted_pair_ccsd_solve(
    const BoundedRestrictedPairCCSDSolverInput&, const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedPairCCSDSolverOptions&, const BoundedRestrictedPairCCSDSolverInventory&,
    const BoundedRestrictedPairCCSDSolverCaps&, BoundedRestrictedPairCCSDSolverCallback = nullptr, void* = nullptr);

// Explicit opt-in replacement, not addition to a complete residual. Planning
// accepts a null produce pointer but validates all declaration bounds/identity.
// The solve requires it, copies controls before callbacks and publishes no
// result after missing/double visits, mutation, invalid output or exceptions.
// Common target ERI counts exclude exactly 6*o*n^2 calls per evaluation;
// local-source producer work is separately charged, never claimed eliminated.
BoundedRestrictedPairCCSDSolverMemoryPlan plan_bounded_restricted_pair_ccsd_solver_upper(
    std::uint64_t occupied, std::uint64_t common_virtual_dimension,
    std::uint64_t maximum_singles_rank, std::uint64_t maximum_pair_rank,
    const BoundedRestrictedPairCCSDSolverOptions&, const BoundedRestrictedPairCCSDSolverInventory&,
    const BoundedRestrictedPairCCSDParticleHoleProducer&,
    std::uint64_t integral_retained_bytes = 0, std::uint64_t integral_transient_bytes = 0);
BoundedRestrictedPairCCSDSolverMemoryPlan plan_bounded_restricted_pair_ccsd_solver(
    const BoundedRestrictedPairCCSDSolverInput&, const BoundedRestrictedPairCCSDSolverOptions&,
    const BoundedRestrictedPairCCSDSolverInventory&, const BoundedRestrictedPairCCSDSolverCaps&,
    const BoundedRestrictedPairCCSDParticleHoleProducer&,
    std::uint64_t integral_retained_bytes = 0, std::uint64_t integral_transient_bytes = 0);
BoundedRestrictedPairCCSDSolverResult bounded_restricted_pair_ccsd_solve(
    const BoundedRestrictedPairCCSDSolverInput&, const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedPairCCSDParticleHoleProducer&,
    const BoundedRestrictedPairCCSDSolverOptions&, const BoundedRestrictedPairCCSDSolverInventory&,
    const BoundedRestrictedPairCCSDSolverCaps&, BoundedRestrictedPairCCSDSolverCallback = nullptr, void* = nullptr);

} // namespace vibeqc
