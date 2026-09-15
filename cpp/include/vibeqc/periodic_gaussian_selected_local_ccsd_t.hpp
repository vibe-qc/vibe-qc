#pragma once

// Native Gaussian-HF -> IAO/Wannier -> selected real PAO space -> streamed
// Gaussian factors -> CCSD + occupied-coupled (T). The selected common-space
// reference is not the pair-specific production DLPNO algorithm. The actual
// HF owner and original physical inputs are authenticated, not caller flags.
// Source equations are those of the constituent native factories (Sun 2017,
// Zhu/Tew 2024, Nejad 2025, Riplinger/Neese 2013 and Guo 2018).

#include <optional>
#include "vibeqc/periodic_gaussian_localization.hpp"
#include "vibeqc/periodic_correlation_real_pao_space.hpp"
#include "vibeqc/periodic_gaussian_real_local_provider.hpp"
#include "vibeqc/periodic_correlation_real_local_ccsd_t.hpp"
#include "vibeqc/periodic_gaussian_local_triples.hpp"
#include "vibeqc/periodic_gaussian_domain_pair_mp2.hpp"
#include "vibeqc/periodic_gaussian_pair_ccsd_physical_particle_hole.hpp"

namespace vibeqc {

struct PeriodicGaussianSelectedLocalSelection {
    // Exact interleaved uint64 views. Domain: [D,2] (cell,AO). Occupied:
    // [o,2] (active home-cell Wannier index,cell). No modular aliases or
    // duplicate rows within a list. Immutable view storage may overlap;
    // the conservative inventory still charges both borrowed roles.
    const std::uint64_t* domain_indices = nullptr;
    std::size_t domain_elements = 0, domain_count = 0;
    const std::uint64_t* occupied_indices = nullptr;
    std::size_t occupied_elements = 0, occupied_count = 0;
    // All retained PAO virtuals are used, at this explicit modular cell.
    std::uint64_t virtual_translation_cell = 0;
};
struct PeriodicGaussianSelectedLocalCCSDTOptions {
    PeriodicGaussianLocalizationOptions localization;
    PeriodicCorrelationPAODomainOptions domain;
    PeriodicCorrelationRealPAOSpaceOptions space;
    PeriodicCorrelationRealLocalBasisOptions basis;
    PeriodicGaussianRealLocalProviderConfig factors;
    PeriodicCorrelationRealLocalProviderOptions real_factors;
    PeriodicCorrelationRealLocalCCSDTOptions correlation;
    // Explicit experimental branch: native pair-PNO MP2 -> Galerkin CCSD
    // -> union TNOs -> locally projected occupied-coupled triples. This is
    // not a named production DLPNO preset. The default uses common-domain
    // PNO generation; selected_pair_domains below enables distinct PAOs.
    bool pair_local_correlation = false;
    // Additional opt-in: genuine selected atomic domains -> translated
    // PAO pair frame -> pair-sized density. Still a finite-source/common-
    // factor Galerkin reference, not the production local-CC approximation.
    bool selected_pair_domains = false;
    // Replace only the bare particle-hole group with the authenticated
    // all-q mixed-pair contraction. Requires pair_local_correlation. The
    // dressed remainder is still common-frame; production_dlpno stays false.
    bool physical_particle_hole_ccsd = false;
    PeriodicGaussianPairInteractionConfig particle_hole;
    PeriodicGaussianPairInteractionOptions particle_hole_options;
    PeriodicCorrelationOccupiedPAODomainOptions occupied_domains;
    PeriodicGaussianDomainPairMP2Options domain_pair_mp2;
    PeriodicGaussianPairMP2Options pair_mp2;
    PeriodicGaussianPairCCSDOptions pair_ccsd;
    BoundedRestrictedTNOOptions triple_spaces;
    PeriodicGaussianLocalTriplesOptions local_triples;
    // The two scalar integral work declarations are supplied by the actual
    // native provider at FactorsReady, not guessed before its rank is known.
};
struct PeriodicGaussianSelectedLocalCCSDTLiveInventory {
    // Additional storage beyond original HF state (already in reference),
    // three active basis roles, original geometry and both index views.
    // Includes caller callback storage and excess basis capacities.
    std::uint64_t other_live_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianSelectedLocalCCSDTCaps {
    PeriodicGaussianBlochIAOCaps iao_source;
    PeriodicGaussianLocalizationCaps localization;
    std::uint64_t maximum_domain_owned_numerical_bytes = 0;
    PeriodicCorrelationRealPAOSpaceCaps space;
    PeriodicCorrelationRealLocalBasisCaps basis;
    PeriodicGaussianRealLocalProviderCaps factors;
    PeriodicCorrelationRealLocalCCSDTCaps correlation;
    PeriodicGaussianPairMP2Caps pair_mp2;
    PeriodicGaussianPairDomainBuilderCaps pair_domain_builder;
    PeriodicGaussianDomainPairMP2Caps domain_pair_mp2;
    PeriodicGaussianPairCCSDCaps pair_ccsd;
    PeriodicGaussianPairParticleHoleCaps particle_hole;
    PeriodicGaussianTripleSpacesCaps triple_spaces;
    PeriodicGaussianLocalTriplesCaps local_triples;
    std::uint64_t maximum_pair_control_storage_bytes = 0;
    std::uint64_t maximum_pair_rank_padding_bytes = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_factor_control_storage_bytes = 0;
    std::uint64_t maximum_progress_callbacks = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianSelectedLocalCCSDTPlan {
    std::uint64_t n_kpoints = 0, n_basis = 0, domain_count = 0, occupied_count = 0;
    std::uint64_t caller_index_bytes = 0, auxiliary_basis_active_bytes = 0;
    std::uint64_t additional_minimal_geometry_bytes_upper_bound = 0;
    std::uint64_t borrowed_input_bytes_upper_bound = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t pair_rank_padding_reservation_bytes = 0;
    bool pair_local_correlation = false;
    bool selected_pair_domains = false;
    bool physical_particle_hole_ccsd = false;
    std::uint64_t localization_phase_owned_upper_bound = 0;
    std::uint64_t domain_phase_owned_upper_bound = 0;
    std::uint64_t space_phase_owned_upper_bound = 0;
    std::uint64_t basis_phase_owned_upper_bound = 0;
    std::uint64_t factors_phase_owned_upper_bound = 0;
    std::uint64_t pair_domain_preparation_phase_owned_upper_bound = 0;
    // Populated only for the physical particle-hole opt-in. Localization
    // and common geometry survive MP2/CCSD, but not the later TNO/(T) phase.
    std::uint64_t pair_mp2_phase_owned_upper_bound = 0;
    std::uint64_t physical_pair_ccsd_phase_owned_upper_bound = 0;
    std::uint64_t physical_pair_ccsd_metadata_work_units_upper_bound = 0;
    std::uint64_t correlation_phase_owned_upper_bound = 0;
    std::uint64_t owned_numerical_upper_bound = 0;
    std::uint64_t per_worker_inventoried_bytes_upper_bound = 0;
    std::uint64_t node_inventoried_bytes_upper_bound = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t input_check_work_units = 0, domain_work_units_upper_bound = 0;
    std::uint64_t progress_callback_upper_bound = 0, work_units_upper_bound = 0;
};
enum class PeriodicGaussianSelectedLocalCCSDTStage : std::uint32_t {
    Begin = 0, Localization = 1, DomainReady = 2, SpaceReady = 3,
    BasisReady = 4, Factors = 5, FactorsReady = 6, Correlation = 7, Finished = 8,
    PairMP2 = 9, PairCCSD = 10, TripleSpaces = 11, LocalTriples = 12, PairDomainsReady = 13
};
struct PeriodicGaussianSelectedLocalCCSDTProgress {
    PeriodicGaussianSelectedLocalCCSDTStage stage = PeriodicGaussianSelectedLocalCCSDTStage::Begin;
    std::uint64_t callback_count = 0, retained_virtual_count = 0;
    PeriodicGaussianLocalizationProgress localization;
    PeriodicGaussianRealLocalProviderProgress factors;
    PeriodicCorrelationRealLocalCCSDTProgress correlation;
    PeriodicGaussianPairMP2Progress pair_mp2;
    PeriodicGaussianPairCCSDProgress pair_ccsd;
    PeriodicGaussianTripleSpacesProgress triple_spaces;
    PeriodicGaussianLocalTriplesProgress local_triples;
};
using PeriodicGaussianSelectedLocalCCSDTCallback = void (*)(
    const PeriodicGaussianSelectedLocalCCSDTProgress&, void*);
struct PeriodicGaussianSelectedLocalCCSDTDiagnostics {
    std::uint64_t completed_progress_callbacks = 0, completed_input_checks = 0;
    std::uint64_t retained_virtual_count = 0;
    bool complete_finite_torus_basis = false;
    double hf_energy_per_cell = 0.0;
    PeriodicGaussianLocalizationMemoryPlan localization_memory;
    PeriodicGaussianLocalizationDiagnostics localization;
    PeriodicCorrelationPAODomainDiagnostics domain;
    PeriodicCorrelationRealPAOSpaceDiagnostics space;
    PeriodicCorrelationRealLocalBasisDiagnostics basis;
    PeriodicGaussianRealLocalProviderPlan factors_memory;
    PeriodicGaussianRealLocalProviderReceipt factors;
    PeriodicGaussianPairDomainBuilderPlan pair_domain_builder_memory;
    PeriodicGaussianPairDomainBuilderDiagnostics pair_domain_builder;
    bool physical_particle_hole_evaluated = false;
    PeriodicGaussianPairCCSDPhysicalParticleHolePlan physical_particle_hole_memory;
};
class PeriodicGaussianSelectedLocalCCSDTResult {
public:
    PeriodicGaussianSelectedLocalCCSDTResult(const PeriodicGaussianSelectedLocalCCSDTResult&) = delete;
    PeriodicGaussianSelectedLocalCCSDTResult& operator=(const PeriodicGaussianSelectedLocalCCSDTResult&) = delete;
    PeriodicGaussianSelectedLocalCCSDTResult(PeriodicGaussianSelectedLocalCCSDTResult&&) noexcept = default;
    PeriodicGaussianSelectedLocalCCSDTResult& operator=(PeriodicGaussianSelectedLocalCCSDTResult&&) noexcept = delete;
    const PeriodicGaussianSelectedLocalCCSDTPlan& plan() const noexcept { return plan_; }
    const PeriodicGaussianSelectedLocalCCSDTDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    bool correlation_evaluated() const noexcept { return correlation_.has_value() || pair_mp2_.has_value(); }
    bool converged() const;
    bool pair_mp2_evaluated() const noexcept { return pair_mp2_.has_value(); }
    bool pair_ccsd_evaluated() const noexcept { return pair_ccsd_.has_value(); }
    bool triple_spaces_evaluated() const noexcept { return triple_spaces_.has_value(); }
    bool local_triples_evaluated() const noexcept { return local_triples_.has_value(); }
    const PeriodicGaussianPairMP2Result& pair_mp2() const;
    const PeriodicGaussianPairCCSDResult& pair_ccsd() const;
    const PeriodicGaussianTripleSpaces& triple_spaces() const;
    const PeriodicGaussianLocalTriplesResult& local_triples() const;
    const PeriodicGaussianLocalizationResult& unfinished_localization() const;
    const PeriodicCorrelationRealLocalCCSDTResult& correlation() const;
    bool matched_finite_gaussian_hf_recipe() const noexcept { return !hf_.empty(); }
    bool bitwise_hf_factor_consumption_verified() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    // Per-cell correlation/total energy exists ONLY for a converged complete
    // finite-torus basis (all active occupied translations and full retained
    // virtual rank). A selected subsystem energy is never added to per-cell HF.
    bool periodic_energy_per_cell() const;
    double correlation_energy_per_cell() const;
    double total_energy_per_cell() const;
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& input_identity_sha256() const noexcept { return inputs_; }
    const std::string& localization_identity_sha256() const noexcept { return localization_; }
    const std::string& provider_identity_sha256() const noexcept { return provider_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
private:
    PeriodicGaussianSelectedLocalCCSDTResult() = default;
    PeriodicGaussianSelectedLocalCCSDTPlan plan_;
    PeriodicGaussianSelectedLocalCCSDTDiagnostics diagnostics_;
    std::optional<PeriodicGaussianLocalizationResult> unfinished_;
    std::optional<PeriodicCorrelationRealLocalCCSDTResult> correlation_;
    // All exact frame/amplitude owners remain available together. The outer
    // cap counts their retained numerical payload throughout later stages.
    std::optional<PeriodicGaussianPairMP2Result> pair_mp2_;
    std::optional<PeriodicGaussianPairCCSDResult> pair_ccsd_;
    std::optional<PeriodicGaussianTripleSpaces> triple_spaces_;
    std::optional<PeriodicGaussianLocalTriplesResult> local_triples_;
    std::string hf_, inputs_, localization_, provider_, identity_;
    friend PeriodicGaussianSelectedLocalCCSDTResult run_periodic_gaussian_selected_local_ccsd_t(
        const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
        const BasisSet&, const BasisSet&, const BasisSet&, const PeriodicSystem&,
        PeriodicGaussianSelectedLocalSelection, const PeriodicGaussianSelectedLocalCCSDTOptions&,
        const PeriodicGaussianSelectedLocalCCSDTLiveInventory&, const PeriodicGaussianSelectedLocalCCSDTCaps&,
        PeriodicGaussianSelectedLocalCCSDTCallback, void*);
};

// Constant-size metadata admission before basis/selection scans. Explicit
// leaf ceilings give conservative whole-call phase bounds; tighter caps may
// avoid unnecessary rejection. State/baseline is counted once, independent
// of how many immutable owners refer to it. Logical control/backend allowance
// is separate from active numerical arrays, not a process-RSS guarantee.
PeriodicGaussianSelectedLocalCCSDTPlan plan_periodic_gaussian_selected_local_ccsd_t(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    PeriodicGaussianSelectedLocalSelection, const PeriodicGaussianSelectedLocalCCSDTOptions&,
    const PeriodicGaussianSelectedLocalCCSDTLiveInventory&, const PeriodicGaussianSelectedLocalCCSDTCaps&);

// Strict same-state owner, original A/nuclei/AO/aux, frozen-mask and physical
// source checks. Minimal basis and explicit labels remain caller decisions.
// Every user callback is scalar-only and followed by bounded physical-input
// rechecks. Options/caps are copied before callbacks. Exceptions publish no
// partial success. Localization nonconvergence returns its last accepted
// iterate without constructing a correlation result. CCSD nonconvergence
// returns its evaluated status without attempting triples. No output bytes
// are emitted by C++; the frontend renders events/results.
PeriodicGaussianSelectedLocalCCSDTResult run_periodic_gaussian_selected_local_ccsd_t(
    const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao, const BasisSet& auxiliary, const BasisSet& minimal,
    const PeriodicSystem&, PeriodicGaussianSelectedLocalSelection,
    const PeriodicGaussianSelectedLocalCCSDTOptions&,
    const PeriodicGaussianSelectedLocalCCSDTLiveInventory&, const PeriodicGaussianSelectedLocalCCSDTCaps&,
    PeriodicGaussianSelectedLocalCCSDTCallback progress = nullptr, void* progress_context = nullptr);

} // namespace vibeqc
