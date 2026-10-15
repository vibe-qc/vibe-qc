#pragma once
#include "vibeqc/manta_periodic_rhf.hpp"
#include "vibeqc/manta_periodic_uhf.hpp"

namespace vibeqc::manta {
// The boxes are conservative resource bounds. Membership is decided by the
// shared physical AO-pair and certified reciprocal source enumerators.
struct PeriodicRHFDomainPlan {
    PeriodicRHFPlan rhf;
    double reciprocal_energy_cutoff=0.,pair_cutoff_bohr=0.,translation_radius_bohr=0.;
    Count momenta=0,translation_candidates=0,hartree_candidates=0;
    Count maximum_exchange_candidates=0,exchange_candidates=0,candidate_visits=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicRHFDomainReceipt {
    std::string identity;
    Count translations=0,hartree_vectors=0,exchange_vectors=0,conjugacy_vectors=0;
};
struct PeriodicRHFDomains {
    PeriodicRHFDomainPlan plan;
    PeriodicRHFDomainReceipt receipt;
    md::Matrix translations,reciprocal_vectors,exchange_vectors;
    std::vector<Count> transfer_offsets;
};
struct PeriodicUHFDomainPlan {
    PeriodicUHFPlan uhf;
    double reciprocal_energy_cutoff=0.,pair_cutoff_bohr=0.,translation_radius_bohr=0.;
    Count momenta=0,translation_candidates=0,hartree_candidates=0;
    Count maximum_exchange_candidates=0,exchange_candidates=0,candidate_visits=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicUHFDomains {
    PeriodicUHFDomainPlan plan;
    PeriodicRHFDomainReceipt receipt;
    md::Matrix translations,reciprocal_vectors,exchange_vectors;
    std::vector<Count> transfer_offsets;
};
// Explicit positive physical pair cutoff required. The one reciprocal kinetic
// cutoff selects |G|^2/2 and |G+k_target-k_source|^2/2 spheres. Zero is omitted.
// All storage and worst-case SCF work are admitted before source enumeration.
PeriodicRHFDomainPlan plan_periodic_rhf_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicRHFOptions&,const ResourceBudget&);
PeriodicRHFDomains build_periodic_rhf_domains(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicRHFOptions&,const ResourceBudget&);
PeriodicRHFResult run_periodic_rhf_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicRHFOptions&,const ResourceBudget&);
PeriodicUHFDomainPlan plan_periodic_uhf_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicUHFOptions&,const ResourceBudget&);
PeriodicUHFDomains build_periodic_uhf_domains(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicUHFOptions&,const ResourceBudget&);
PeriodicUHFResult run_periodic_uhf_from_cutoffs(const BasisSet&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicUHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
