#pragma once
#include "vibeqc/manta_periodic_ri.hpp"
#include "vibeqc/product_auxiliary.hpp"
#include "vibeqc/manta_periodic_bupo_preparation.hpp"

namespace vibeqc::manta {
struct PeriodicRIFittingBoundOptions {
    double density_l1_limit=0.,j_error=0.,energy_error=0.;
    // Optional strict spectral-norm perturbation margin of the stored Gamma
    // metric. This is not an estimate of integral or Fock arithmetic error.
    std::optional<double> metric_rank_perturbation;
    double metric_action_j_error=0.,metric_action_energy_error=0.;
    std::vector<double> density_entry_limits; // |w_k sum_spin D_kij|; optional.
    Count maximum_probes=10000,probe_batch_size=8;
};
// This is a declared joint search reservation, not a value-free estimate of
// the eventual BUPO tree. Each producer stage and probe is admitted against
// the remaining reservation; discarded trials are never refunded.
struct PeriodicRIFittingReferenceOptions {
    PeriodicBupoSourceOptions initial;
    ResourceBudget budget;
    Count maximum_refinements=12,maximum_cube_radius=8;
    double growth_factor=1.5,maximum_pair_cutoff_bohr=32.,reference_fraction=.25;
    bool selective_refinement=true;
    PeriodicRIFittingReferenceOptions() {
        initial.pairs.order=0;initial.pair_cube_radius=0;
        initial.real_cube_radius=initial.reciprocal_cube_radius=1;
        initial.eta=.4;initial.bupo.integral_tolerance=0.;
        budget.memory_bytes=Count{1}<<30;budget.shell_calls=1000000000000ULL;
        budget.primitive_quartets=100000000000000ULL;
        budget.scalar_terms=100000000000000000ULL;budget.kernel_terms=1000000000000000000ULL;
    }
};
// Optional physical fitting-domain search; basis, kernel and AO frame stay
// fixed. A zero-refinement policy reproduces a selected source for controls.
struct PeriodicRIFittingDomainOptions {
    ResourceBudget budget=PeriodicRIFittingReferenceOptions{}.budget;
    std::optional<double> initial_pair_cutoff_bohr,initial_auxiliary_cutoff_bohr;
    Count initial_reciprocal_cube_radius=0,maximum_reciprocal_cube_radius=8,maximum_refinements=3;
    double growth_factor=1.5,maximum_pair_cutoff_bohr=32.,maximum_auxiliary_cutoff_bohr=32.;
};
// Public jobs prepare bounded metadata through the shared generator before
// native count admission. Native consumers may supply the same immutable sequence.
struct PeriodicRIFittingAuxiliaryOptions {
    ProductAuxiliaryOptions initial;
    ProductAuxiliaryLimits generation_limits;
    Count maximum_refinements=2;
    ResourceBudget budget=PeriodicRIFittingReferenceOptions{}.budget;
    std::shared_ptr<const ProductAuxiliarySequence> prepared;
};
void validate_periodic_ri_auxiliary_options(const PeriodicRIFittingAuxiliaryOptions&);
PeriodicRIFittingAuxiliaryOptions prepare_periodic_ri_auxiliary_selection(const BasisSet&,const BasisSet&,
    const PeriodicSystem&,const PeriodicRIFittingAuxiliaryOptions&,const ResourceBudget&);
void validate_periodic_ri_domain_options(const PeriodicRIFittingDomainOptions&);
void validate_periodic_ri_reference_options(const PeriodicRIFittingReferenceOptions&);
// Complete methods always derive their fixed family from canonical overlap.
// The immutable reference is shared, with preparation costs charged again in
// each complete-method reservation. Optional metric refinement reuses the
// selected coupling owner; optional domain search replaces it only after
// admitting the largest allowed SCF source. Auxiliary selection shares prepared
// candidate metadata and the same immutable reference/density family.
struct PeriodicRIFittingAccuracyOptions {
    std::shared_ptr<const PreparedPeriodicBupoSource> reference;
    std::optional<PeriodicRIFittingReferenceOptions> reference_selection;
    std::optional<PeriodicRIFittingDomainOptions> domain_selection;
    std::optional<PeriodicRIFittingAuxiliaryOptions> auxiliary_selection;
    double j_error=0.,energy_error=0.;
    std::optional<double> metric_rank_perturbation;
    double metric_action_j_error=0.,metric_action_energy_error=0.;
    Count maximum_probes=10000,probe_batch_size=8;
    Count maximum_metric_refinements=0;
    double metric_cutoff_reduction=.1,minimum_metric_cutoff=1e-12;
    bool enabled() const { return metric_action_j_error!=0. || metric_action_energy_error!=0. || metric_rank_perturbation || auxiliary_selection || domain_selection || reference_selection || maximum_metric_refinements || reference || j_error!=0. || energy_error!=0.; }
};
ResourceBudget periodic_ri_fitting_search_budget(const PeriodicRIFittingAccuracyOptions&);
struct PeriodicRIFittingBoundPlan {
    Count n_basis=0,momenta=0,spins=0,maximum_probes=0,maximum_trials=1;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicRIFittingBoundReceipt {
    std::string qualification="full-zero-average-coulomb-on-fixed-weighted-hermitian-density-family;stored-integrals";
    std::string ri_source_identity,reference_source_identity,pair_source_identity,certificate_identity;
    Count probes=0,reference_batches=0,reference_source_passes=0,reference_shell_calls=0,reference_majorant_pairs=0;
    double density_l1_limit=0.,j_error_bound=0.,energy_error_bound=0.;
    double finite_difference_j_bound=0.,reference_remainder_j_bound=0.;
    double finite_difference_energy_bound=0.,reference_remainder_energy_bound=0.;
    double reference_approximation_j_bound=0.,reference_approximation_energy_bound=0.;
    double reference_real_domain_j_bound=0.,reference_real_domain_energy_bound=0.;
    double reference_reciprocal_domain_j_bound=0.,reference_reciprocal_domain_energy_bound=0.;
    double reference_pair_image_j_bound=0.,reference_pair_image_energy_bound=0.;

    std::optional<double> metric_rank_perturbation;
    HermitianSpectrumCertificate metric_spectrum;
    bool metric_rank_qualified=false;
    bool rank_target_met() const { return !metric_rank_perturbation || metric_rank_qualified; }
    bool metric_action_requested=false,metric_action_qualified=false;
    std::string metric_action_coupling="stored-explicit-tiles";
    double metric_action_j_target=0.,metric_action_energy_target=0.;
    double metric_action_j_bound=0.,metric_action_energy_bound=0.;
    Count metric_action_columns=0,metric_action_shell_calls=0,metric_action_primitive_products=0;
    bool action_target_met() const { return !metric_action_requested || (metric_action_qualified &&
        metric_action_j_bound<=metric_action_j_target && metric_action_energy_bound<=metric_action_energy_target); }
    bool metric_targets_met() const { return rank_target_met() && action_target_met(); }
    bool roundoff_included=false,component_density_envelope=false,targets_requested=false;
};
// Compare the selected paired RI map with the existing BUPO source's full
// Coulomb enclosure, not a variational fit residual. Sun2017, II.B and
// Neese2025, Sec.2.1 motivate the fit; source/domain bounds retain their own
// Ewald conventions. No projection/positivity identity is assumed between
// independently truncated M/T and the unfitted operator. Hermitian basis
// probes span every admitted spin-total density without storing an AO^4 map.
class PeriodicRIFittingBoundSource {
public:
    static PeriodicRIFittingBoundPlan plan_counts(const PeriodicRIQPlan&,const PeriodicBupoPlan&,
        const PeriodicRIFittingBoundOptions&,const ResourceBudget&);
    static PeriodicRIFittingBoundPlan plan_source(const PeriodicRIQBuilder&,const PeriodicBupoSource&,
        const PeriodicRIFittingBoundOptions&,const ResourceBudget&);
    PeriodicRIFittingBoundSource(std::shared_ptr<const PeriodicRIQBuilder>,std::shared_ptr<const PeriodicBupoSource>,
        const PeriodicRIFittingBoundOptions&,const ResourceBudget&,bool enforce_targets=true);
    const PeriodicRIFittingBoundPlan& plan() const { return plan_; }
    const PeriodicRIFittingBoundReceipt& receipt() const { return receipt_; }
    const std::vector<double>& j_error_bounds() const { return j_error_bounds_; }
    const std::vector<double>& metric_action_j_bounds() const { return metric_action_j_bounds_; }
    void validate_density(const std::vector<std::complex<double>>&,const std::vector<double>&) const;
    PeriodicRIQResult build(const std::vector<std::complex<double>>&,const std::vector<double>&) const;
private:
    PeriodicRIFittingBoundPlan plan_;
    std::shared_ptr<const PeriodicRIQBuilder> ri_;
    std::shared_ptr<const PeriodicBupoSource> reference_;
    PeriodicRIFittingBoundOptions options_;
    PeriodicRIFittingBoundReceipt receipt_;
    std::vector<double> j_error_bounds_,metric_action_j_bounds_;
    void certify_metric_action(const ResourceBudget&);
};
PeriodicRIFittingBoundPlan plan_periodic_ri_fitting_accuracy(const PeriodicRIQPlan&,
    const PeriodicRIFittingAccuracyOptions&,const ResourceBudget&);
PeriodicRIFittingBoundOptions periodic_ri_fitting_family(const PeriodicRIFittingAccuracyOptions&,
    const BoundedPeriodicDensityEnvelope&);
} // namespace vibeqc::manta
