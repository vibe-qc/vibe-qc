#pragma once
#include "vibeqc/manta_periodic_bupo_hf.hpp"

namespace vibeqc::manta {
// Exchange-only selection over an existing immutable BUPO source. The AO,
// tree, near/far and electronic/nuclear Ewald domains remain source-owned.
struct PeriodicBupoHFDomainPlan {
    PeriodicBupoHFPlan hf;
    double reciprocal_energy_cutoff=0.;
    Count momenta=0,maximum_exchange_candidates=0,exchange_candidates=0,candidate_visits=0;
    Count exchange_bound_cube_radius=0,domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicBupoHFDomainReceipt {
    std::string identity;
    Count exchange_vectors=0,conjugacy_vectors=0;
};
struct PeriodicBupoHFExchangeDomains {
    PeriodicBupoHFDomainPlan plan;
    PeriodicBupoHFDomainReceipt receipt;
    md::Matrix exchange_vectors;
    std::vector<Count> transfer_offsets;
};
// Requires bound_exchange=true and exchange_bound_cube_radius=0. The factory
// derives the completion cube from certified enumeration boxes, never from
// the accepted vectors alone. A cutoff is a domain choice, not a guarantee:
// every evaluated density must still pass all unchanged HF accuracy targets.
PeriodicBupoHFDomainPlan plan_periodic_bupo_hf_from_cutoff(const PeriodicBupoSource&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicBupoHFOptions&,const ResourceBudget&);
PeriodicBupoHFExchangeDomains build_periodic_bupo_hf_exchange_domains(const PeriodicBupoSource&,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicBupoHFOptions&,const ResourceBudget&);
PeriodicBupoHFResult run_periodic_bupo_hf_from_cutoff(std::shared_ptr<const PeriodicBupoSource>,const PeriodicSystem&,
    const RegularKMesh&,double reciprocal_energy_cutoff,const PeriodicBupoHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
