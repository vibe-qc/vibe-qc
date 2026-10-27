#pragma once
#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/manta_periodic_exchange_bounds.hpp"

namespace vibeqc::manta {
struct PeriodicRIExchangeFittingOptions {
    // Unweighted sum over spin, k and AO entries of |D|. Optional entry
    // limits bound |D_alpha(k,i,j)|+|D_beta(k,i,j)|, not the signed spin total.
    double density_l1_limit=0.,matrix_error=0.,energy_error=0.;
    // Separate conditional source-response targets, never added to fitting error.
    double metric_assembly_matrix_error=0.,metric_assembly_energy_error=0.;
    std::vector<double> density_entry_limits;
    Count maximum_probes=10000,reciprocal_batch=64,fourier_decay_bins=1;
    Count ao_completion_cube_radius=2,reciprocal_completion_cube_radius=6;
};
struct PeriodicRIExchangeFittingPlan {
    Count n_basis=0,momenta=0,maximum_probes=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicRIExchangeFittingResult {
    PeriodicRIExchangeFittingPlan plan;
    std::string qualification="full-coulomb-exchange-on-unweighted-hermitian-spin-density-family;zero-mode-omitted;stored-integrals";
    std::string kernel="coulomb";
    double physical_omega=0.;
    std::string ri_source_identity,reference_source_identity,finite_reference_source_identity,pair_source_identity,certificate_identity;
    // Set only by automatic attachment, before conversion to the raw spin family.
    std::string canonical_density_envelope_identity;
    Count probes=0,ri_shell_calls=0,reference_shell_calls=0;
    bool probe_roundoff_included=true,source_construction_included=false,targets_requested=false,qualified=false;
    double density_l1_limit=0.,maximum_error=0.,energy_error_bound=0.;
    double finite_difference_maximum=0.,reference_remainder_maximum=0.;
    double finite_difference_energy_bound=0.,reference_remainder_energy_bound=0.;
    std::vector<double> error_bounds,finite_difference_bounds,reference_remainder_bounds;
    std::vector<double> density_entry_limits;
    bool metric_assembly_requested=false,metric_assembly_qualified=false;
    double metric_assembly_matrix_bound=0.,metric_assembly_energy_bound=0.;
    std::vector<double> metric_assembly_error_bounds;
    std::string metric_assembly_identity;
    Count metric_assembly_channels=0;
    void validate_density(const std::vector<std::complex<double>>&,Count spins) const;
};
// Optional automatic attachment to a complete method. Targets bound the raw
// full-Coulomb K component on the canonical spin-density family, not the
// BvK correction, other Fock components or relaxed-SCF energy.
struct PeriodicRIExchangeAccuracyOptions {
    double matrix_error=0.,energy_error=0.;
    double metric_assembly_matrix_error=0.,metric_assembly_energy_error=0.;
    Count reference_ao_cube_radius=1,reference_reciprocal_cube_radius=8;
    Count ao_completion_cube_radius=2,maximum_probes=10000,reciprocal_batch=64,fourier_decay_bins=1;
    bool enabled() const {return matrix_error!=0. || energy_error!=0. || metric_assembly_matrix_error!=0. || metric_assembly_energy_error!=0.;}
};
PeriodicRIExchangeFittingPlan plan_periodic_ri_exchange_accuracy(const BasisSet&,const PeriodicRIFockPlan&,
    const PeriodicRIExchangeAccuracyOptions&,Count retained_fitting_bytes,const ResourceBudget&,
    const PeriodicGDFKernelOptions& physical_kernel={});
PeriodicRIExchangeFittingResult bound_periodic_ri_exchange_accuracy(const BasisSet&,const PeriodicSystem&,
    const md::Matrix& momenta,const PeriodicRIFockBuilder&,const BoundedPeriodicDensityEnvelope&,
    const PeriodicRIExchangeAccuracyOptions&,Count retained_fitting_bytes,const ResourceBudget&,bool enforce_targets=true);
// Hermitian probes compare existing linear exchange maps, without assuming
// that separately truncated M/T define a positive variational fit residual.
// Sun et al., JCP 147,164119 (2017), II.B; verified arithmetic Rump2010.
// Admission before constructing reference pairs or reading vector payloads.
// The RI capacity is produced by the shared RI planner; it does not authorize
// execution on an owner. Actual basis/cell/k/kernel/weight checks still run.
// Counts include reference-pair preparation, but the caller separately owns
// RI setup work. retained_fitting_bytes reserves an attached J certificate.
PeriodicRIExchangeFittingPlan plan_periodic_ri_exchange_fitting_capacity(const BasisSet&,
    const PeriodicRIFockPlan&,Count reference_cells,Count reference_vectors,
    const PeriodicPairOptions&,const PeriodicRIExchangeFittingOptions&,Count retained_fitting_bytes,const ResourceBudget&);
PeriodicRIExchangeFittingPlan plan_periodic_ri_exchange_fitting(const PeriodicRIFockBuilder&,
    const PeriodicPairSource&,Count reference_vectors,const PeriodicRIExchangeFittingOptions&,const ResourceBudget&);
PeriodicRIExchangeFittingResult certify_periodic_ri_exchange_fitting(const PeriodicRIFockBuilder&,
    const PeriodicPairSource&,const md::Matrix& reference_vectors,const std::vector<Count>& offsets,
    const PeriodicRIExchangeFittingOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
