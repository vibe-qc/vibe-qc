#pragma once

// Actual-HF connection for the bounded Mulliken/PAO-tail domain selector.
// No caller AO-to-atom mapping, gauges or localized-orbital flags. Numerical
// selection is the existing native Riplinger2013/Nejad2025 leaf, not the
// principal-domain algorithm or a CCSD extended-domain construction.
#include "vibeqc/periodic_correlation_occupied_pao_domain.hpp"
#include "vibeqc/periodic_gaussian_localization.hpp"
#include "vibeqc/periodic_gaussian_rhf.hpp"

namespace vibeqc {
struct PeriodicGaussianOccupiedPAODomainCaps {
    PeriodicCorrelationOccupiedPAODomainCaps selector;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0;
    std::uint64_t maximum_node_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianOccupiedPAODomainMemoryPlan {
    PeriodicCorrelationOccupiedPAODomainMemoryPlan selector;
    std::uint64_t atom_mapping_bytes = 0;
    // Original AO/minimal roles and geometry from localization, plus the
    // HF auxiliary basis. Role aliases are conservatively counted as roles,
    // not deduplicated by equal names or content hashes.
    std::uint64_t borrowed_gaussian_numerical_bytes = 0;
    std::uint64_t additional_control_storage_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t worker_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t physical_input_validation_work_units = 0;
    std::uint64_t mapping_work_units = 0, work_units = 0;
};
class PeriodicGaussianOccupiedPAODomain {
public:
    PeriodicGaussianOccupiedPAODomain(const PeriodicGaussianOccupiedPAODomain&) = delete;
    PeriodicGaussianOccupiedPAODomain& operator=(const PeriodicGaussianOccupiedPAODomain&) = delete;
    PeriodicGaussianOccupiedPAODomain(PeriodicGaussianOccupiedPAODomain&&) noexcept = default;
    PeriodicGaussianOccupiedPAODomain& operator=(PeriodicGaussianOccupiedPAODomain&&) = delete;
    const PeriodicCorrelationOccupiedPAODomain& domain() const;
    const PeriodicGaussianOccupiedPAODomainMemoryPlan& memory() const noexcept { return memory_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return static_cast<bool>(context_); }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& localization_identity_sha256() const noexcept { return localization_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
private:
    PeriodicGaussianOccupiedPAODomain() = default;
    std::optional<PeriodicCorrelationOccupiedPAODomain> domain_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianOccupiedPAODomainMemoryPlan memory_;
    std::string hf_, localization_, identity_;
    friend PeriodicGaussianOccupiedPAODomain select_periodic_gaussian_occupied_pao_domain(
        const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
        const BasisSet&, const BasisSet&, const BasisSet&, const PeriodicSystem&,
        const PeriodicGaussianLocalizationResult&, std::uint64_t,
        const PeriodicCorrelationOccupiedPAODomainOptions&,
        const PeriodicCorrelationOccupiedPAODomainInventory&,
        const PeriodicGaussianOccupiedPAODomainCaps&);
};

// Metadata-only admission. The atom map is a new native 8*nao owner, already
// charged as the child's borrowed caller map. Original physical-input scans
// occur only after ALL worker/node/work/owned caps pass. Original reference
// state baseline and backend allowance are counted once, without re-admission.
PeriodicGaussianOccupiedPAODomainMemoryPlan plan_periodic_gaussian_occupied_pao_domain(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianLocalizationResult&, std::uint64_t home_active_occupied_index,
    const PeriodicCorrelationOccupiedPAODomainOptions&,
    const PeriodicCorrelationOccupiedPAODomainInventory&,
    const PeriodicGaussianOccupiedPAODomainCaps&);

// Inputs stay immutable during this synchronous call; no callbacks. AO atom
// ownership is derived from the actual BasisSet shell order after checking
// both original Gaussian receipts, exact state/allocation, and AO-image
// cutoff. The owned domain does not retain HF/localization/basis pointers.
PeriodicGaussianOccupiedPAODomain select_periodic_gaussian_occupied_pao_domain(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao, const BasisSet& auxiliary, const BasisSet& minimal,
    const PeriodicSystem&, const PeriodicGaussianLocalizationResult&,
    std::uint64_t home_active_occupied_index,
    const PeriodicCorrelationOccupiedPAODomainOptions&,
    const PeriodicCorrelationOccupiedPAODomainInventory&,
    const PeriodicGaussianOccupiedPAODomainCaps&);
} // namespace vibeqc
