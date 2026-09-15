#pragma once

// Actual-HF provenance wrapper for the translation-unique initial pair PAO
// union. No caller atom map, new localization, projector or extended domain.
#include "vibeqc/periodic_gaussian_occupied_pao_domain.hpp"
#include "vibeqc/periodic_correlation_pair_pao_domain.hpp"

namespace vibeqc {
struct PeriodicGaussianPairPAODomainCaps {
    PeriodicCorrelationPairPAODomainCaps pair_union;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_worker_bytes = 0;
    std::uint64_t maximum_node_bytes = 0, maximum_work_units = 0;
};
struct PeriodicGaussianPairPAODomainMemoryPlan {
    PeriodicCorrelationPairPAODomainMemoryPlan pair_union;
    std::uint64_t atom_mapping_bytes = 0, borrowed_gaussian_numerical_bytes = 0;
    std::uint64_t additional_control_storage_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t worker_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t physical_input_validation_work_units = 0, mapping_work_units = 0, work_units = 0;
};
class PeriodicGaussianPairPAODomain {
public:
    PeriodicGaussianPairPAODomain(const PeriodicGaussianPairPAODomain&) = delete;
    PeriodicGaussianPairPAODomain& operator=(const PeriodicGaussianPairPAODomain&) = delete;
    PeriodicGaussianPairPAODomain(PeriodicGaussianPairPAODomain&&) noexcept = default;
    PeriodicGaussianPairPAODomain& operator=(PeriodicGaussianPairPAODomain&&) = delete;
    const PeriodicCorrelationPairPAODomain& domain() const;
    const PeriodicGaussianPairPAODomainMemoryPlan& memory() const noexcept { return memory_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& localization_identity_sha256() const noexcept { return localization_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return static_cast<bool>(context_); }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
private:
    PeriodicGaussianPairPAODomain() = default;
    std::optional<PeriodicCorrelationPairPAODomain> domain_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianPairPAODomainMemoryPlan memory_;
    std::string hf_, localization_, identity_;
    friend PeriodicGaussianPairPAODomain make_periodic_gaussian_pair_pao_domain(
        const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
        const BasisSet&, const BasisSet&, const PeriodicSystem&,
        const PeriodicCorrelationTranslationPairTopology&, std::uint64_t,
        const PeriodicGaussianOccupiedPAODomain&, const PeriodicGaussianOccupiedPAODomain&,
        const PeriodicCorrelationPairPAODomainInventory&, const PeriodicGaussianPairPAODomainCaps&);
};

// Metadata-only admission, before Gaussian input scans or native map creation.
// The selector's original Gaussian localization receipt is already sealed by
// each endpoint. No original localizer/minimal basis is required to union it.
// If those owners remain live, callers must include them in other-live input.
// Endpoint payloads deduplicate by owner address, never by their equal hashes.
PeriodicGaussianPairPAODomainMemoryPlan plan_periodic_gaussian_pair_pao_domain(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationTranslationPairTopology&, std::uint64_t row_index,
    const PeriodicGaussianOccupiedPAODomain& home, const PeriodicGaussianOccupiedPAODomain& partner,
    const PeriodicCorrelationPairPAODomainInventory&, const PeriodicGaussianPairPAODomainCaps&);

// Revalidate actual HF AO/auxiliary/cell inputs and expand the actual AO shell
// labels to an owned uint64[nao] map. The native union compares that map with
// BOTH sealed occupied-domain maps. Exact HF context/state/allocation and
// global localization receipts must match. Inputs immutable, no callbacks.
// Retains only shared state/context plus its compact union, not source owners.
PeriodicGaussianPairPAODomain make_periodic_gaussian_pair_pao_domain(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao, const BasisSet& auxiliary, const PeriodicSystem&,
    const PeriodicCorrelationTranslationPairTopology&, std::uint64_t row_index,
    const PeriodicGaussianOccupiedPAODomain& home, const PeriodicGaussianOccupiedPAODomain& partner,
    const PeriodicCorrelationPairPAODomainInventory&, const PeriodicGaussianPairPAODomainCaps&);
} // namespace vibeqc
