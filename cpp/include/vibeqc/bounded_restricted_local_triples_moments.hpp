#pragma once

// Published local-TNO moment approximation, Riplinger et al.,
// doi:10.1063/1.4821834 Eqs.14-15 and implementation steps 6-7:
// project singles/doubles into the TNO frame BEFORE contracting W/U there.
// In particular, W's internal d sum is also truncated to that frame. This
// is NOT external projection of the full-common-space W at truncated rank.
// It is a numerical adapter, not an HF/TNO-selection/DLPNO certificate.
#include <array>
#include <string>
#include "vibeqc/bounded_restricted_pair_ccsd_amplitudes.hpp"
#include "vibeqc/bounded_restricted_triples_moments.hpp"

namespace vibeqc {

struct BoundedRestrictedLocalTriplesMomentsInput {
    std::uint64_t rank = 0; // strictly positive; empty triples skip production
    std::array<std::uint64_t,3> occupied{0,0,0}; // ordered, repeats allowed
    std::uint64_t ccsd_snapshot_id = 0; // positive sequencing, not provenance
    BoundedRestrictedCCSDRealView target_coefficients; // common_n by rank
    BoundedRestrictedCCSDRealView original_f_ov; // o by common_n, EXACT zero
};
struct BoundedRestrictedLocalTriplesMomentsOptions {
    double coefficient_orthogonality_tolerance = 0; // explicit finite (0,1)
    double amplitude_symmetry_tolerance = 0; // finite nonnegative, consumed only
    std::uint64_t maximum_integral_work_units_per_call = 0; // positive opaque cost
};
struct BoundedRestrictedLocalTriplesMomentsInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    // Include opaque original-integral owner/control and borrowed-array owner
    // bookkeeping here; only their explicit numeric provider roles and the
    // known reader/table/control are counted automatically. No RSS claim.
    std::uint64_t other_live_bytes_per_replica = 0, fixed_backend_margin_bytes_per_replica = 0;
};
struct BoundedRestrictedLocalTriplesMomentsCaps {
    std::uint64_t maximum_occupied_count = 0, maximum_common_virtual_dimension = 0, maximum_rank = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_common_singles_calls = 0, maximum_common_doubles_calls = 0;
    std::uint64_t maximum_transformed_integral_calls = 0, maximum_common_integral_calls = 0;
    std::uint64_t maximum_work_units = 0;
};
struct BoundedRestrictedLocalTriplesMomentsMemoryPlan {
    bool uniform_rank_upper_bound = false;
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0, rank = 0;
    BoundedRestrictedTriplesMomentsAccessorMemoryPlan moments;
    std::uint64_t reader_borrowed_numerical_bytes = 0, reader_borrowed_table_bytes = 0;
    std::uint64_t reader_control_storage_bytes = 0;
    std::uint64_t target_coefficient_bytes = 0, original_f_ov_bytes = 0;
    std::uint64_t integral_retained_numerical_bytes = 0, integral_maximum_transient_numerical_bytes = 0;
    std::uint64_t projected_zero_f_ov_bytes = 0, output_numerical_bytes = 0;
    // Exactly 8or+16r^3 owned numeric bytes. No projected pair table or
    // common T1/T2/ERI tensor. Target C is attributed to the projected
    // amplitude context and shared by the integral context, counted once.
    std::uint64_t peak_owned_numerical_bytes = 0, control_storage_reservation_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t common_singles_calls = 0, common_doubles_calls = 0;
    std::uint64_t transformed_integral_calls = 0, common_integral_calls = 0;
    std::uint64_t one_virtual_integral_calls = 0, two_virtual_integral_calls = 0, three_virtual_integral_calls = 0;
    std::uint64_t maximum_common_integral_calls_per_transformed_query = 0; // n^4, general callback
    std::uint64_t maximum_projected_singles_work_units_per_query = 0;
    std::uint64_t maximum_projected_doubles_work_units_per_query = 0;
    std::uint64_t validation_work_units = 0, projection_work_units = 0;
    std::uint64_t integral_work_units = 0, work_units_upper_bound = 0;
};

class BoundedRestrictedLocalTriplesMomentsResult {
public:
    BoundedRestrictedLocalTriplesMomentsResult(const BoundedRestrictedLocalTriplesMomentsResult&) = delete;
    BoundedRestrictedLocalTriplesMomentsResult& operator=(const BoundedRestrictedLocalTriplesMomentsResult&) = delete;
    BoundedRestrictedLocalTriplesMomentsResult(BoundedRestrictedLocalTriplesMomentsResult&&) noexcept = default;
    BoundedRestrictedLocalTriplesMomentsResult& operator=(BoundedRestrictedLocalTriplesMomentsResult&&) = delete;
    const BoundedRestrictedLocalTriplesMomentsMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedRestrictedTriplesMomentsAccessorResult& moments() const noexcept { return moments_; }
    std::uint64_t common_singles_calls() const noexcept { return singles_calls_; }
    std::uint64_t common_doubles_calls() const noexcept { return doubles_calls_; }
    std::uint64_t common_integral_calls() const noexcept { return integral_calls_; }
    std::uint64_t transformed_integral_calls() const noexcept { return transformed_calls_; }
    double coefficient_orthogonality_error() const noexcept { return gram_error_; }
    const std::string& reader_snapshot_identity_sha256() const noexcept { return reader_; }
    const std::string& target_coefficients_identity_sha256() const noexcept { return coefficients_; }
    const std::string& input_identity_sha256() const noexcept { return input_; }
    // Receipt of actual ordered labels/values consumed from the opaque common
    // ERI callback, not a complete Hamiltonian or integral-symmetry certificate.
    const std::string& consumed_integral_receipt_sha256() const noexcept { return integrals_; }
    const std::string& payload_identity_sha256() const noexcept { return payload_; }
private:
    BoundedRestrictedLocalTriplesMomentsResult() = default;
    BoundedRestrictedLocalTriplesMomentsMemoryPlan memory_;
    BoundedRestrictedTriplesMomentsAccessorResult moments_;
    std::uint64_t singles_calls_ = 0, doubles_calls_ = 0, integral_calls_ = 0, transformed_calls_ = 0;
    double gram_error_ = 0;
    std::string reader_,coefficients_,input_,integrals_,payload_;
    friend BoundedRestrictedLocalTriplesMomentsResult bounded_restricted_local_triples_moments(
        const BoundedRestrictedLocalTriplesMomentsInput&,const BoundedRestrictedPairCCSDAmplitudes&,
        const BoundedRestrictedCCSDIntegralProvider&,const BoundedRestrictedLocalTriplesMomentsOptions&,
        const BoundedRestrictedLocalTriplesMomentsInventory&,const BoundedRestrictedLocalTriplesMomentsCaps&);
};

// Count-only from the reader's already admitted immutable metadata. No input
// scans, validation, allocation or integral callbacks. All count arithmetic
// is checked. Reader inventory is counted by its actual payload/table/control
// roles, not by reapplying its original ambient replica/backend inventory.
BoundedRestrictedLocalTriplesMomentsMemoryPlan plan_bounded_restricted_local_triples_moments(
    const BoundedRestrictedLocalTriplesMomentsInput&,const BoundedRestrictedPairCCSDAmplitudes&,
    const BoundedRestrictedCCSDIntegralProvider&,const BoundedRestrictedLocalTriplesMomentsOptions&,
    const BoundedRestrictedLocalTriplesMomentsInventory&);

// Allocation-free pre-reader/pre-table upper planner. Derives the uniform
// reader bound internally from actual scalar counts, never fake reader views
// or a caller-labelled reader plan. It certifies no amplitude snapshot.
BoundedRestrictedLocalTriplesMomentsMemoryPlan plan_bounded_restricted_local_triples_moments_upper(
    std::uint64_t n_occupied,std::uint64_t common_virtual_dimension,std::uint64_t rank,
    std::uint64_t maximum_singles_rank,std::uint64_t maximum_pair_rank,
    std::uint64_t integral_retained_numerical_bytes,std::uint64_t integral_maximum_transient_numerical_bytes,
    const BoundedRestrictedLocalTriplesMomentsOptions&,const BoundedRestrictedLocalTriplesMomentsInventory&);

// All byte/count/work caps precede full reader validation and floating scans.
// Reader, C and ORIGINAL Fov are fully validated before and after production.
// Borrowed objects must remain alive/immutable throughout; opaque ERI context
// immutability is a caller contract, not certified by these boundary checks.
// Scalar amplitude projection uses the reader's proven pair covariance to
// canonicalize reverse/diagonal query indices, never an averaging repair.
// ERIs are transformed in original query order; no symmetry shortcut is used.
// The exact-zero original Fov gate is not an implicit Brillouin projection.
BoundedRestrictedLocalTriplesMomentsResult bounded_restricted_local_triples_moments(
    const BoundedRestrictedLocalTriplesMomentsInput&,const BoundedRestrictedPairCCSDAmplitudes&,
    const BoundedRestrictedCCSDIntegralProvider&,const BoundedRestrictedLocalTriplesMomentsOptions&,
    const BoundedRestrictedLocalTriplesMomentsInventory&,const BoundedRestrictedLocalTriplesMomentsCaps&);

} // namespace vibeqc
