#pragma once
#include "vibeqc/manta_periodic_domains.hpp"
#include "vibeqc/manta_periodic_ri_uhf.hpp"
#include "vibeqc/manta_periodic_ri_rhf.hpp"

namespace vibeqc::manta {
// Geometry uses the shared certified spheres. Admission includes auxiliary
// images, every retained metric/projector and the full fitted spin SCF.
struct PeriodicRIUHFDomainPlan {
    PeriodicRIUHFPlan uhf;
    double reciprocal_energy_cutoff=0.,pair_cutoff_bohr=0.,translation_radius_bohr=0.;
    Count momenta=0,translation_candidates=0,hartree_candidates=0;
    Count maximum_exchange_candidates=0,exchange_candidates=0,candidate_visits=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicRIUHFDomains {
    PeriodicRIUHFDomainPlan plan;
    PeriodicRHFDomainReceipt receipt;
    md::Matrix translations,reciprocal_vectors,exchange_vectors;
    std::vector<Count> transfer_offsets;
};
// One kinetic cutoff selects primed Gamma G and G+k_source-k_target spheres.
// Diagonal pair segments are empty; the immutable Gamma source owns them.
// HF and fitting pair cutoffs must agree. Nuclear/auxiliary radii, fitting
// omega, metric projection and method correction retain separate controls.
PeriodicRIUHFDomainPlan plan_periodic_ri_uhf_from_cutoffs(const BasisSet&,const BasisSet&,
    const PeriodicSystem&,const RegularKMesh&,double,const PeriodicRIUHFOptions&,const ResourceBudget&);
PeriodicRIUHFDomains build_periodic_ri_uhf_domains(const BasisSet&,const BasisSet&,
    const PeriodicSystem&,const RegularKMesh&,double,const PeriodicRIUHFOptions&,const ResourceBudget&);
PeriodicRIUHFResult run_periodic_ri_uhf_from_cutoffs(const BasisSet&,const BasisSet&,
    const PeriodicSystem&,const RegularKMesh&,double,const PeriodicRIUHFOptions&,const ResourceBudget&);
struct PeriodicRIRHFDomainPlan {
    PeriodicRIRHFPlan rhf;
    double reciprocal_energy_cutoff=0.,pair_cutoff_bohr=0.,translation_radius_bohr=0.;
    Count momenta=0,translation_candidates=0,hartree_candidates=0;
    Count maximum_exchange_candidates=0,exchange_candidates=0,candidate_visits=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicRIRHFDomains {
    PeriodicRIRHFDomainPlan plan;
    PeriodicRHFDomainReceipt receipt;
    md::Matrix translations,reciprocal_vectors,exchange_vectors;
    std::vector<Count> transfer_offsets;
};
// One kinetic cutoff selects primed Gamma G and G+k_source-k_target spheres.
// Diagonal pair segments are empty; the immutable Gamma source owns them.
// HF and fitting pair cutoffs must agree. Nuclear/auxiliary radii, fitting
// omega, metric projection and method correction retain separate controls.
PeriodicRIRHFDomainPlan plan_periodic_ri_rhf_from_cutoffs(const BasisSet&,const BasisSet&,
    const PeriodicSystem&,const RegularKMesh&,double,const PeriodicRIRHFOptions&,const ResourceBudget&);
PeriodicRIRHFDomains build_periodic_ri_rhf_domains(const BasisSet&,const BasisSet&,
    const PeriodicSystem&,const RegularKMesh&,double,const PeriodicRIRHFOptions&,const ResourceBudget&);
PeriodicRIRHFResult run_periodic_ri_rhf_from_cutoffs(const BasisSet&,const BasisSet&,
    const PeriodicSystem&,const RegularKMesh&,double,const PeriodicRIRHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
