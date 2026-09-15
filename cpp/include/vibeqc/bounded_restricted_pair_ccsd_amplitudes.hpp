#pragma once

// Numerical ragged-amplitude view for projected canonical restricted CCSD.
// Riplinger/Neese doi:10.1063/1.4773581 II.B.4 and II.C: singles have their
// OWN spaces; expand coupled amplitudes in a common extended domain before
// residual projection. No orbital/Hamiltonian/locality/PNO certificate.
#include <array>
#include <string>
#include "vibeqc/bounded_restricted_ccsd_target.hpp"

namespace vibeqc {

struct BoundedRestrictedPairCCSDSinglesView {
    std::uint64_t rank = 0;
    BoundedRestrictedCCSDRealView coefficients; // C_i[common_n,rank]
    BoundedRestrictedCCSDRealView amplitudes;   // connected t_i[rank]
};
struct BoundedRestrictedPairCCSDPairView {
    std::uint64_t rank = 0;
    BoundedRestrictedCCSDRealView coefficients; // C_ij[common_n,rank]
    BoundedRestrictedCCSDRealView amplitudes;   // connected alpha-beta T_ij[rank,rank]
};
struct BoundedRestrictedPairCCSDAmplitudesInput {
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0;
    const BoundedRestrictedPairCCSDSinglesView* singles = nullptr;
    std::size_t singles_count = 0; // exactly o
    const BoundedRestrictedPairCCSDPairView* pairs = nullptr;
    std::size_t pair_count = 0; // exactly o(o+1)/2, lexicographic i<=j
};
struct BoundedRestrictedPairCCSDAmplitudesOptions {
    double coefficient_orthogonality_tolerance = 0.0; // explicit finite (0,1)
};
struct BoundedRestrictedPairCCSDAmplitudesInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t other_live_bytes_per_replica = 0, fixed_backend_margin_bytes_per_replica = 0;
};
struct BoundedRestrictedPairCCSDAmplitudesCaps {
    std::uint64_t maximum_occupied_count = 0, maximum_common_virtual_dimension = 0, maximum_pair_count = 0;
    std::uint64_t maximum_singles_rank = 0, maximum_pair_rank = 0; // zero permits only zero rank
    std::uint64_t maximum_table_bytes = 0, maximum_borrowed_numerical_bytes = 0;
    // A zero numerical-byte cap is valid for the all-zero-rank snapshot.
    std::uint64_t maximum_per_replica_inventoried_bytes = 0, maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_validation_work_units = 0;
    std::uint64_t maximum_singles_work_units_per_query = 0, maximum_doubles_work_units_per_query = 0;
};
struct BoundedRestrictedPairCCSDAmplitudesMemoryPlan {
    bool uniform_rank_upper_bound = false;
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0, pair_count = 0;
    std::uint64_t maximum_singles_rank = 0, maximum_pair_rank = 0;
    std::uint64_t singles_amplitude_elements = 0, pair_amplitude_elements = 0;
    std::uint64_t singles_coefficient_elements = 0, pair_coefficient_elements = 0;
    std::uint64_t borrowed_singles_table_bytes = 0, borrowed_pair_table_bytes = 0, borrowed_table_bytes = 0;
    std::uint64_t borrowed_singles_numerical_bytes = 0, borrowed_pair_numerical_bytes = 0;
    std::uint64_t borrowed_numerical_bytes = 0, owned_numerical_bytes = 0;
    std::uint64_t fixed_control_storage_bytes = 0;
    std::uint64_t metadata_work_units = 0, validation_work_units = 0;
    std::uint64_t maximum_singles_work_units_per_query = 0, maximum_doubles_work_units_per_query = 0;
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
};
struct BoundedRestrictedPairCCSDAmplitudesDiagnostics {
    double maximum_coefficient_orthogonality_error = 0.0;
    std::uint64_t validated_singles = 0, validated_pairs = 0;
};
class BoundedRestrictedPairCCSDAmplitudes {
public:
    BoundedRestrictedPairCCSDAmplitudes(const BoundedRestrictedPairCCSDAmplitudes&) = delete;
    BoundedRestrictedPairCCSDAmplitudes& operator=(const BoundedRestrictedPairCCSDAmplitudes&) = delete;
    BoundedRestrictedPairCCSDAmplitudes(BoundedRestrictedPairCCSDAmplitudes&&) noexcept;
    BoundedRestrictedPairCCSDAmplitudes& operator=(BoundedRestrictedPairCCSDAmplitudes&&) = delete;
    const BoundedRestrictedPairCCSDAmplitudesMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedRestrictedPairCCSDAmplitudesDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    std::string snapshot_identity_sha256() const { return {snapshot_.begin(),snapshot_.end()}; }
    // No allocation, input-wide scan or callback. C*t and C*T*C^T use
    // compensated scalar loops. Pair reversal swaps BOTH occupied/virtual
    // labels. For i==j, evaluate a<=b canonically so common T is exactly
    // symmetric without averaging. Every diagonal local T must already be
    // exactly symmetric. Zero ranks return +0, not omitted/missing pairs.
    double singles(std::uint64_t i,std::uint64_t a) const;
    double doubles(std::uint64_t i,std::uint64_t j,std::uint64_t a,std::uint64_t b) const;
    // Native-only immutable descriptor copies, not numerical copies. Caller
    // keeps this reader AND its original tables/payload alive and unchanged.
    // Full descriptor/payload replay belongs at snapshot boundaries, not
    // each query. Pair borrow requires canonical i<=j; reversal is explicit.
    BoundedRestrictedPairCCSDSinglesView singles_view(std::uint64_t i) const &;
    BoundedRestrictedPairCCSDSinglesView singles_view(std::uint64_t) const && = delete;
    BoundedRestrictedPairCCSDPairView canonical_pair_view(std::uint64_t i,std::uint64_t j) const &;
    BoundedRestrictedPairCCSDPairView canonical_pair_view(std::uint64_t,std::uint64_t) const && = delete;
    BoundedRestrictedCCSDAmplitudeProvider amplitude_provider() const;
    // Full capped revalidation + payload/descriptor comparison. Call around
    // external progress callbacks; rebuild the reader for a NEW snapshot.
    // Concurrent mutation is forbidden, not cured by before/after hashing.
    void validate_immutable_snapshot() const;
private:
    BoundedRestrictedPairCCSDAmplitudes() = default;
    BoundedRestrictedPairCCSDAmplitudesInput input_;
    BoundedRestrictedPairCCSDAmplitudesOptions options_;
    BoundedRestrictedPairCCSDAmplitudesInventory inventory_;
    BoundedRestrictedPairCCSDAmplitudesCaps caps_;
    BoundedRestrictedPairCCSDAmplitudesMemoryPlan memory_;
    BoundedRestrictedPairCCSDAmplitudesDiagnostics diagnostics_;
    std::array<char,64> snapshot_{},descriptors_{};
    friend BoundedRestrictedPairCCSDAmplitudes make_bounded_restricted_pair_ccsd_amplitudes(
        const BoundedRestrictedPairCCSDAmplitudesInput&,const BoundedRestrictedPairCCSDAmplitudesOptions&,
        const BoundedRestrictedPairCCSDAmplitudesInventory&,const BoundedRestrictedPairCCSDAmplitudesCaps&);
};

// No tables, scans, callbacks or allocations. Uniform rank upper inventory
// for o>0,n>0,0<=maximum_singles_rank/maximum_pair_rank<=n.
BoundedRestrictedPairCCSDAmplitudesMemoryPlan plan_bounded_restricted_pair_ccsd_amplitudes_upper(
    std::uint64_t n_occupied,std::uint64_t common_virtual_dimension,
    std::uint64_t maximum_singles_rank,std::uint64_t maximum_pair_rank,
    const BoundedRestrictedPairCCSDAmplitudesInventory&);
// Counts/table/work caps precede table traversal; exact ragged numeric/work
// caps precede floating reads. No size-dependent heap allocation in plan
// or construction (SHA finish has fixed-size string/control storage).
BoundedRestrictedPairCCSDAmplitudesMemoryPlan plan_bounded_restricted_pair_ccsd_amplitudes(
    const BoundedRestrictedPairCCSDAmplitudesInput&,const BoundedRestrictedPairCCSDAmplitudesOptions&,
    const BoundedRestrictedPairCCSDAmplitudesInventory&,const BoundedRestrictedPairCCSDAmplitudesCaps&);
// Both descriptor tables AND all numeric views remain caller-owned, aligned,
// exact-sized, immutable and alive for the entire reader lifetime. Aliases
// are conservatively counted PER ROLE, never deduplicated by equal payload.
// Provider retained bytes = borrowed numeric roles; descriptor/control bytes
// remain separately inventoried here and must be charged by enclosing loops.
// Aggregate query work/count caps belong to the consuming energy/residual
// driver; this object guarantees only the stated bound on EACH query.
BoundedRestrictedPairCCSDAmplitudes make_bounded_restricted_pair_ccsd_amplitudes(
    const BoundedRestrictedPairCCSDAmplitudesInput&,const BoundedRestrictedPairCCSDAmplitudesOptions&,
    const BoundedRestrictedPairCCSDAmplitudesInventory&,const BoundedRestrictedPairCCSDAmplitudesCaps&);

} // namespace vibeqc
