#pragma once

// Actual-source, streamed pair-domain preparation. Retains home occupied
// domains and ONE common geometry; constructs and releases a pair union,
// PAO geometry and real space for each request. The embedding-only factory
// releases that geometry; the bundle factory transfers it to the caller.
// No internal all-pair cache, PNO generation, extended domain or energy.
#include "vibeqc/periodic_gaussian_occupied_pao_domain.hpp"
#include "vibeqc/periodic_correlation_pair_pao_domain.hpp"
#include "vibeqc/periodic_correlation_real_pao_embedding.hpp"

namespace vibeqc {
class PeriodicGaussianPairPNOGeometryView;
class PeriodicGaussianPairDomainBuilder;
struct PeriodicGaussianPairDomainBuilderLive {
    std::uint64_t other_live_numerical_bytes_per_worker = 0;
    std::uint64_t other_live_control_bytes_per_worker = 0;
    std::uint64_t backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianPairDomainBuilderCaps {
    PeriodicGaussianOccupiedPAODomainCaps occupied;
    std::uint64_t maximum_home_domains = 0, maximum_topology_rows = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0, maximum_node_bytes = 0, maximum_work_units = 0;
};
struct PeriodicGaussianPairDomainBuilderPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_atoms = 0, n_home_occupied = 0;
    std::uint64_t common_virtual_dimension = 0, topology_rows = 0;
    std::uint64_t atom_mapping_bytes = 0, topology_row_bytes = 0;
    std::uint64_t home_domain_output_upper_bytes = 0, common_geometry_numerical_bytes = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_localization_bytes = 0, borrowed_gaussian_bytes = 0;
    std::uint64_t retained_numerical_upper_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0, retained_control_upper_bytes = 0;
    std::uint64_t input_validation_work_units = 0, domain_work_units = 0, work_units = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t worker_bytes = 0, required_node_memory_bytes = 0;
};
struct PeriodicGaussianPairDomainBuilderDiagnostics {
    std::uint64_t retained_home_domain_bytes = 0, retained_numerical_bytes = 0;
    std::uint64_t retained_control_storage_bytes = 0, constructed_home_domains = 0;
};
struct PeriodicGaussianPairDomainOptions {
    PeriodicCorrelationPAODomainOptions domain;
    PeriodicCorrelationRealPAOSpaceOptions real_space;
    PeriodicCorrelationRealPAOEmbeddingOptions embedding;
};
struct PeriodicGaussianPairDomainCaps {
    PeriodicCorrelationPairPAODomainCaps pair_union;
    std::uint64_t maximum_pao_domain_owned_numerical_bytes = 0;
    PeriodicCorrelationRealPAOSpaceCaps real_space;
    PeriodicCorrelationRealPAOEmbeddingCaps embedding;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0, maximum_node_bytes = 0, maximum_work_units = 0;
};
struct PeriodicGaussianPairDomainEmbeddingPlan {
    std::uint64_t n_cells = 0, n_basis = 0, common_virtual_dimension = 0;
    std::uint64_t maximum_generation_dimension = 0, occupied_slot_i = 0, occupied_slot_j = 0;
    std::uint64_t borrowed_builder_numerical_bytes = 0, borrowed_basis_numerical_bytes = 0;
    std::uint64_t retained_embedding_upper_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t maximum_domain_dimension = 0, retained_pair_geometry_upper_bytes = 0;
    std::uint64_t retained_geometry_control_upper_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t metadata_validation_work_units = 0, pao_domain_work_upper = 0;
    std::uint64_t child_work_upper = 0, work_units = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t worker_bytes = 0, required_node_memory_bytes = 0;
};

// Sealed native preparation owner. No coefficients or source labels can be
// supplied to its constructor. All three children move only after source
// validation succeeds; the bundle stores no pointers into a builder or HF.
// geometry_view() is a BORROWED view constructed after moves, not retained
// inside this owner. Keep the bundle alive and unmoved while using the view.
class PeriodicGaussianPairDomainGeometry {
public:
    PeriodicGaussianPairDomainGeometry(const PeriodicGaussianPairDomainGeometry&) = delete;
    PeriodicGaussianPairDomainGeometry& operator=(const PeriodicGaussianPairDomainGeometry&) = delete;
    PeriodicGaussianPairDomainGeometry(PeriodicGaussianPairDomainGeometry&&) noexcept = default;
    PeriodicGaussianPairDomainGeometry& operator=(PeriodicGaussianPairDomainGeometry&&) = delete;
    const PeriodicCorrelationPAODomain& domain() const &;
    const PeriodicCorrelationPAODomain& domain() const && = delete;
    const PeriodicCorrelationRealPAOSpace& real_space() const &;
    const PeriodicCorrelationRealPAOSpace& real_space() const && = delete;
    const PeriodicCorrelationRealPAOEmbedding& embedding() const &;
    const PeriodicCorrelationRealPAOEmbedding& embedding() const && = delete;
    PeriodicGaussianPairPNOGeometryView geometry_view() const &;
    PeriodicGaussianPairPNOGeometryView geometry_view() const && = delete;
    std::uint64_t occupied_slot_i() const noexcept { return i_; }
    std::uint64_t occupied_slot_j() const noexcept { return j_; }
    std::uint64_t retained_numerical_bytes() const;
    std::uint64_t retained_control_storage_bytes() const;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& builder_identity_sha256() const noexcept { return builder_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& allocation_identity() const noexcept { return allocation_; }
private:
    PeriodicGaussianPairDomainGeometry() = default;
    void require_live() const;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::optional<PeriodicCorrelationPAODomain> domain_;
    std::optional<PeriodicCorrelationRealPAOSpace> space_;
    std::optional<PeriodicCorrelationRealPAOEmbedding> embedding_;
    std::uint64_t i_ = 0, j_ = 0, numerical_bytes_ = 0;
    std::string identity_, builder_, basis_, hf_, allocation_;
    friend PeriodicGaussianPairDomainGeometry make_periodic_gaussian_pair_domain_geometry(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianPairDomainBuilder&, std::uint64_t, std::uint64_t,
        const PeriodicGaussianPairDomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
        const PeriodicGaussianPairDomainCaps&);
    friend PeriodicCorrelationRealPAOEmbedding make_periodic_gaussian_pair_domain_embedding(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianPairDomainBuilder&, std::uint64_t, std::uint64_t,
        const PeriodicGaussianPairDomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
        const PeriodicGaussianPairDomainCaps&);
};

class PeriodicGaussianPairDomainBuilder {
public:
    PeriodicGaussianPairDomainBuilder(const PeriodicGaussianPairDomainBuilder&) = delete;
    PeriodicGaussianPairDomainBuilder& operator=(const PeriodicGaussianPairDomainBuilder&) = delete;
    PeriodicGaussianPairDomainBuilder(PeriodicGaussianPairDomainBuilder&&) noexcept = default;
    PeriodicGaussianPairDomainBuilder& operator=(PeriodicGaussianPairDomainBuilder&&) = delete;
    const PeriodicGaussianPairDomainBuilderPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianPairDomainBuilderDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& localization_identity_sha256() const noexcept { return localization_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& allocation_identity() const noexcept { return allocation_; }
    std::uint64_t retained_numerical_bytes() const;
    std::uint64_t retained_control_storage_bytes() const;
    const PeriodicCorrelationPAODomain& common_domain() const &;
    const PeriodicCorrelationPAODomain& common_domain() const && = delete;
    const PeriodicCorrelationRealPAOSpace& common_real_space() const &;
    const PeriodicCorrelationRealPAOSpace& common_real_space() const && = delete;
    bool production_distinct_domain_scaling() const noexcept { return false; }
private:
    PeriodicGaussianPairDomainBuilder() = default;
    void require_live() const;
    PeriodicGaussianPairDomainBuilderPlan memory_;
    PeriodicGaussianPairDomainBuilderDiagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::vector<PeriodicGaussianOccupiedPAODomain> domains_;
    std::vector<std::uint64_t> atom_mapping_;
    std::optional<PeriodicCorrelationTranslationPairTopology> topology_;
    std::optional<PeriodicCorrelationPAODomain> common_domain_;
    std::optional<PeriodicCorrelationRealPAOSpace> common_space_;
    std::string basis_,hf_,localization_,identity_,payload_,allocation_;
    friend PeriodicGaussianPairDomainBuilder make_periodic_gaussian_pair_domain_builder(
        const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
        const BasisSet&, const BasisSet&, const BasisSet&, const PeriodicSystem&,
        const PeriodicGaussianLocalizationResult&, const PeriodicCorrelationRealLocalBasis&,
        PeriodicCorrelationPAODomain&&, PeriodicCorrelationRealPAOSpace&&,
        const PeriodicCorrelationOccupiedPAODomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
        const PeriodicGaussianPairDomainBuilderCaps&);
    friend PeriodicGaussianPairDomainEmbeddingPlan plan_periodic_gaussian_pair_domain_embedding(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianPairDomainBuilder&, std::uint64_t, std::uint64_t,
        const PeriodicGaussianPairDomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
        const PeriodicGaussianPairDomainCaps&);
    friend PeriodicGaussianPairDomainGeometry make_periodic_gaussian_pair_domain_geometry(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianPairDomainBuilder&, std::uint64_t, std::uint64_t,
        const PeriodicGaussianPairDomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
        const PeriodicGaussianPairDomainCaps&);
};

// Whole count-only upper admission precedes any input payload scan, native
// atom map/domain/topology allocation, or mutation. Common geometry remains
// borrowed during preparation and moves only after ALL fallible work, giving
// a strong exception guarantee. After success no original AO/minimal/auxiliary,
// cell, HF, localization or common-basis pointer is retained.
PeriodicGaussianPairDomainBuilderPlan plan_periodic_gaussian_pair_domain_builder(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianLocalizationResult&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationRealPAOSpace&,
    const PeriodicCorrelationOccupiedPAODomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
    const PeriodicGaussianPairDomainBuilderCaps&);
PeriodicGaussianPairDomainBuilder make_periodic_gaussian_pair_domain_builder(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao, const BasisSet& auxiliary, const BasisSet& minimal, const PeriodicSystem&,
    const PeriodicGaussianLocalizationResult&, const PeriodicCorrelationRealLocalBasis& common_basis,
    PeriodicCorrelationPAODomain&& common_domain, PeriodicCorrelationRealPAOSpace&& common_space,
    const PeriodicCorrelationOccupiedPAODomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
    const PeriodicGaussianPairDomainBuilderCaps&);

// This macro plan's numeric/control/work upper is ROW-INDEPENDENT: any valid
// slots, including (0,0), bound every subsequent pair for the same controls.
// It performs no topology/domain payload traversal. Actual row-specific
// source validation and native leaf plans occur only after this admission.
// Root macros can reserve the positive child cap SUM before they know any
// pair rank. The builder and basis payloads are each counted once, original
// reference state once, and caller extras once. Exact staged native plans
// remain subordinate checks, not whole-driver peak claims.
// Pair labels come from the actual basis occupied slots and topology resolver.
// Empty union/zero real-PAO rank and failure of physical common containment
// reject; no forced orbital, truncation fallback or polar repair is applied.
// Embedding-only output owns X/eps and state; its per-pair geometry is freed.
// The same plan also admits the optional owned geometry bundle.
PeriodicGaussianPairDomainEmbeddingPlan plan_periodic_gaussian_pair_domain_embedding(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairDomainBuilder&, std::uint64_t occupied_slot_i, std::uint64_t occupied_slot_j,
    const PeriodicGaussianPairDomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
    const PeriodicGaussianPairDomainCaps&);
PeriodicCorrelationRealPAOEmbedding make_periodic_gaussian_pair_domain_embedding(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairDomainBuilder&, std::uint64_t occupied_slot_i, std::uint64_t occupied_slot_j,
    const PeriodicGaussianPairDomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
    const PeriodicGaussianPairDomainCaps&);
// Same count-only plan and physical algorithm as the embedding-only API,
// but retains the actual pair domain and real-space coefficients as well.
PeriodicGaussianPairDomainGeometry make_periodic_gaussian_pair_domain_geometry(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairDomainBuilder&, std::uint64_t occupied_slot_i, std::uint64_t occupied_slot_j,
    const PeriodicGaussianPairDomainOptions&, const PeriodicGaussianPairDomainBuilderLive&,
    const PeriodicGaussianPairDomainCaps&);
} // namespace vibeqc
