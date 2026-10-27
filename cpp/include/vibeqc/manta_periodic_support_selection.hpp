#pragma once
#include "vibeqc/manta_periodic_domains.hpp"
#include "vibeqc/manta_periodic_nuclear_bounds.hpp"
#include "vibeqc/manta_periodic_exchange_bounds.hpp"

namespace vibeqc::manta {
struct PeriodicSupportSelectionOptions {
    Count maximum_attempts=16,maximum_completion_cube_radius=128;
    double growth_factor=1.5,maximum_pair_cutoff_bohr=32,maximum_nuclear_cutoff_bohr=64;
    double maximum_reciprocal_energy_cutoff=2048;
};
// A source request, not an HF/KS method. The method supplies the full-Coulomb
// exchange coefficient (zero when that component is not being certified).
struct PeriodicSupportRequest : PeriodicRHFOptions {
    double reciprocal_energy_cutoff=6,exchange_scale=0;
    PeriodicOneElectronAccuracyOptions one_electron_accuracy;
    PeriodicNuclearAccuracyOptions nuclear_accuracy;
    PeriodicExchangeAccuracyOptions exchange_accuracy;
};
struct PeriodicSupportProbePlan {
    BlochOneElectronPlan overlap;
    BoundedPeriodicRHFMemoryPlan density;
    PeriodicOneElectronAccuracyPlan one_electron;
    PeriodicNuclearAccuracyPlan nuclear;
    std::optional<PeriodicExchangeAccuracyPlan> exchange;
    PeriodicMadelungPlan madelung;
    Count n_alpha=0,n_beta=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicSupportDomainPlan {
    PeriodicSupportProbePlan probe;
    double reciprocal_energy_cutoff=0,pair_cutoff_bohr=0,translation_radius_bohr=0;
    Count momenta=0,translation_candidates=0,hartree_candidates=0,maximum_exchange_candidates=0,exchange_candidates=0,candidate_visits=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicSupportDomains {
    PeriodicSupportDomainPlan plan;
    PeriodicRHFDomainReceipt receipt;
    md::Matrix translations,reciprocal_vectors,exchange_vectors;
    std::vector<Count> transfer_offsets;
};
struct PeriodicSupportTrial {
    double reciprocal_energy_cutoff=0,pair_cutoff_bohr=0,nuclear_cutoff_bohr=0;
    Count ao_cube_radius=0,real_cube_radius=0,reciprocal_cube_radius=0;
    double overlap_error=0,kinetic_error=0,kinetic_energy_error=0;
    bool overlap_image_rank_requested=false,overlap_image_rank_qualified=false;
    double overlap_image_perturbation_bound=0.,overlap_image_rank_radius=0.;
    double nuclear_matrix_error=0,attraction_energy_error=0,nuclear_energy_error=0;
    double exchange_matrix_error=0,exchange_energy_error=0;
    bool qualified=false,grow_pair=false,grow_nuclear=false,grow_reciprocal=false;
    std::string overlap_identity,nuclear_identity,one_electron_certificate,nuclear_certificate,exchange_certificate,density_envelope_identity;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicSupportSelectionResult {
    PeriodicSupportRequest selected;
    PeriodicSupportSelectionOptions policy;
    std::vector<PeriodicSupportTrial> trials;
    bool qualified=false;
    std::string termination,certificate_identity;
    Count preparation_scalar_terms=0,preparation_kernel_terms=0;
    Count retained_bytes=0,modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
PeriodicSupportProbePlan plan_periodic_support_probe(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    Count ng,Count nc,Count nq,const PeriodicSupportRequest&,const ResourceBudget&);
PeriodicSupportDomainPlan plan_periodic_support_trial(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const PeriodicSupportRequest&,const ResourceBudget&);
PeriodicSupportDomains build_periodic_support_domains(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const PeriodicSupportRequest&,const ResourceBudget&);
PeriodicSupportSelectionResult select_periodic_supports(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const PeriodicSupportRequest&,const PeriodicSupportSelectionOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
