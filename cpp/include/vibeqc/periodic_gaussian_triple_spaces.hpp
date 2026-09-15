#pragma once

#include "vibeqc/bounded_restricted_triple_natural_orbitals.hpp"
#include "vibeqc/periodic_gaussian_pair_ccsd.hpp"

namespace vibeqc {

struct PeriodicGaussianTripleSpacesLiveInventory {
    // Beyond original reference, basis, provider and COMPLETE retained MP2
    // and CCSD owners. Callback storage belongs here. These owners remain
    // borrowed/alive through every serialized triple and callback.
    std::uint64_t other_live_numerical_bytes_per_worker = 0;
    std::uint64_t other_live_control_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};

struct PeriodicGaussianTripleSpacesCaps {
    BoundedRestrictedTNOCaps geometry;
    std::uint64_t maximum_pair_count = 0, maximum_triple_count = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes_per_worker = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_progress_callbacks = 0, maximum_work_units = 0;
};

struct PeriodicGaussianTripleSpacesPlan {
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0;
    std::uint64_t pair_count = 0, triple_count = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_provider_row_bytes = 0;
    std::uint64_t borrowed_mp2_numerical_bytes = 0, borrowed_ccsd_numerical_bytes = 0;
    std::uint64_t borrowed_mp2_control_bytes = 0, borrowed_ccsd_control_bytes = 0;
    std::uint64_t retained_triple_output_upper_bytes = 0;
    std::uint64_t serialized_leaf_owned_ceiling = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t retained_triple_object_bytes = 0, retained_triple_seal_bytes = 0;
    std::uint64_t leaf_fixed_control_reservation_bytes = 0, control_storage_reservation_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t progress_callback_upper_bound = 0, driver_work_units = 0, work_units_upper_bound = 0;
    // Tightened by a no-payload/no-allocation metadata pass only AFTER the
    // enclosing cap admission. These remain upper bounds on numerical rank.
    std::uint64_t exact_rank_plan_output_upper_bytes = 0;
    std::uint64_t exact_rank_plan_peak_owned_bytes = 0;
    std::uint64_t exact_rank_plan_geometry_work_units = 0;
};

enum class PeriodicGaussianTripleSpacesStage : std::uint32_t {
    Begin = 0, TripleComplete = 1, Finished = 2
};
struct PeriodicGaussianTripleSpacesProgress {
    PeriodicGaussianTripleSpacesStage stage = PeriodicGaussianTripleSpacesStage::Begin;
    std::uint64_t callback_count = 0, completed_triples = 0;
    std::array<std::uint64_t, 3> occupied{};
    std::uint64_t union_rank = 0, retained_rank = 0, retained_numerical_bytes = 0;
    std::uint64_t geometry_work_units = 0;
};
using PeriodicGaussianTripleSpacesCallback = void (*)(const PeriodicGaussianTripleSpacesProgress&, void*);

struct PeriodicGaussianTripleSpacesDiagnostics {
    std::uint64_t completed_triples = 0, completed_progress_callbacks = 0;
    std::uint64_t retained_numerical_bytes = 0, geometry_work_units = 0;
    // Exact immutable scalar census for downstream pre-table admission.
    std::uint64_t total_retained_rank = 0, total_union_rank = 0, maximum_retained_rank = 0;
    std::uint64_t empty_union_count = 0, empty_retained_space_count = 0, all_equal_tuple_count = 0;
    bool complete_common_finite_torus_basis = false;
    bool all_unions_full_common_rank = false, all_retained_full_common_rank = false;
};

// Actual-source geometry owner, NOT a triples amplitude/energy solver.
// No raw MP2/CCSD/basis/provider references survive return. Later moment
// consumers must supply those immutable owners again and match receipts.
class PeriodicGaussianTripleSpaces {
public:
    PeriodicGaussianTripleSpaces(const PeriodicGaussianTripleSpaces&) = delete;
    PeriodicGaussianTripleSpaces& operator=(const PeriodicGaussianTripleSpaces&) = delete;
    PeriodicGaussianTripleSpaces(PeriodicGaussianTripleSpaces&&) noexcept = default;
    PeriodicGaussianTripleSpaces& operator=(PeriodicGaussianTripleSpaces&&) noexcept = delete;
    const PeriodicGaussianTripleSpacesPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianTripleSpacesDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const BoundedRestrictedTNOOptions& options() const noexcept { return options_; }
    // Canonical i<=j<=k only, native const borrow. Do not bind the mutable
    // generic result wrapper to Python; diagnostic bindings copy arrays.
    const BoundedRestrictedTNOResult& triple(std::uint64_t i, std::uint64_t j, std::uint64_t k) const;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& spaces_identity_sha256() const noexcept { return spaces_; }
    const std::string& ccsd_identity_sha256() const noexcept { return ccsd_; }
    const std::string& ccsd_amplitude_payload_sha256() const noexcept { return amplitudes_; }
    const std::string& warmstart_identity_sha256() const noexcept { return warmstart_; }
    const std::string& pair_spaces_identity_sha256() const noexcept { return pair_spaces_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
    const std::string& provider_identity_sha256() const noexcept { return provider_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& allocation_identity() const noexcept { return allocation_; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return static_cast<bool>(context_); }
    bool production_dlpno() const noexcept { return false; }
    bool includes_triples_energy() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
private:
    PeriodicGaussianTripleSpaces() = default;
    PeriodicGaussianTripleSpacesPlan memory_;
    PeriodicGaussianTripleSpacesDiagnostics diagnostics_;
    BoundedRestrictedTNOOptions options_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::vector<BoundedRestrictedTNOResult> triples_;
    std::string identity_, spaces_, ccsd_, amplitudes_, warmstart_, pair_spaces_, basis_, provider_, hf_, allocation_;
    friend PeriodicGaussianTripleSpaces make_periodic_gaussian_triple_spaces(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&, const PeriodicGaussianPairMP2Result&,
        const PeriodicGaussianPairCCSDResult&, const BoundedRestrictedTNOOptions&,
        const PeriodicGaussianTripleSpacesLiveInventory&, const PeriodicGaussianTripleSpacesCaps&,
        PeriodicGaussianTripleSpacesCallback, void*);
};

// Admit all Q=o(o+1)(o+2)/6 output upper bounds and one serialized leaf
// ceiling BEFORE pair/triple traversal, floating input scans or allocation.
// Then validate every real native frame/amplitude view with no numerical
// copies, obtain each exact geometry plan, and recheck its complete owners.
// No fake reference re-admission, reconstructed cell, store or user arrays.
PeriodicGaussianTripleSpacesPlan plan_periodic_gaussian_triple_spaces(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&, const PeriodicGaussianPairMP2Result&,
    const PeriodicGaussianPairCCSDResult&, const BoundedRestrictedTNOOptions&,
    const PeriodicGaussianTripleSpacesLiveInventory&, const PeriodicGaussianTripleSpacesCaps&);

// Metadata-only native downstream gate. The caller MUST first admit the
// complete live owners/control and O(P+o+Q) validation work using memory()
// counts. No heap allocation, callbacks, C/T/F/eps/occupation payload scan
// or numerical re-evaluation occurs. Exact native owners/lineage, all SHA
// formats and canonical triple record extents/censuses are checked. This
// relies on the private immutable source/batch types; it does not certify
// arbitrary user arrays or authenticate a forged raw numerical TNO result.
void validate_periodic_gaussian_triple_spaces(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&, const PeriodicGaussianPairMP2Result&,
    const PeriodicGaussianPairCCSDResult&, const PeriodicGaussianTripleSpaces&);

// Riplinger doi:10.1063/1.4821834 Eqs.10-15 density with CONVERGED CCSD
// connected T2; MP2 supplies pair C only, never density amplitudes. Includes
// every sorted occupied MULTISET, repeated edges included; no weak triples.
// Cutoff zero completes the union, not the common basis. Positive-cut rank
// zero stays explicit. All callbacks are scalar; throw to cancel. A failed
// callback/audit publishes no partial batch. Exact source owners stay pinned
// and immutable through return. No (T0)/(T), energy or production claim.
PeriodicGaussianTripleSpaces make_periodic_gaussian_triple_spaces(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&, const PeriodicGaussianPairMP2Result&,
    const PeriodicGaussianPairCCSDResult&, const BoundedRestrictedTNOOptions&,
    const PeriodicGaussianTripleSpacesLiveInventory&, const PeriodicGaussianTripleSpacesCaps&,
    PeriodicGaussianTripleSpacesCallback progress = nullptr, void* progress_context = nullptr);

}  // namespace vibeqc
