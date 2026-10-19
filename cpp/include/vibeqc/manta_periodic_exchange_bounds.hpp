#pragma once
#include "vibeqc/manta_periodic_pairs.hpp"
#include "vibeqc/manta_periodic_one_electron_bounds.hpp"

namespace vibeqc::manta {
struct PeriodicExchangeBoundPlan {
    Count n_basis=0,momenta=0,densities=0,ao_cube_radius=0,reciprocal_cube_radius=0,reciprocal_cube_cells=0;
    Count shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicExchangeBoundReceipt {
    std::string qualification="full-coulomb-exchange-at-fixed-density-and-mesh;zero-mode-omitted";
    std::string source_identity,pair_source_identity,density_identity;
    double fourier_decay=0,maximum_error=0,exchange_scale=1;
    std::vector<double> energy_error_bounds;
    bool roundoff_included=false;
};
struct PeriodicExchangeBoundResult {
    PeriodicExchangeBoundPlan plan;
    PeriodicExchangeBoundReceipt receipt;
    std::vector<double> error_bound,pair_image_error_bound,reciprocal_error_bound;
};
// Uniform over the solver's admitted canonical occupied subspaces. The
// bound is on each spin K and on the sum of unscaled spin exchange energies.
// It is a component certificate, not a Fock state or an SCF error estimate.
struct PeriodicExchangeEnvelopeResult {
    PeriodicExchangeBoundPlan plan;
    BoundedPeriodicDensityEnvelope density;
    std::string qualification="full-coulomb-exchange-on-canonical-occupied-subspaces;fixed-mesh;zero-mode-omitted";
    std::string source_identity,pair_source_identity,density_envelope_identity,certificate_identity;
    double maximum_error=0.,energy_error_bound=0.,entry_energy_error_bound=0.,l1_energy_error_bound=0.;
    bool roundoff_included=false;
    std::vector<double> error_bound,pair_image_error_bound,reciprocal_error_bound;
};
// This source consumes the original ordered AO image domain and explicit
// reversal-closed Q supports. No q=0 fitting/canonical folding is applied
// to exchange panels; only uniform absolute image amplitudes are reused.
class PeriodicExchangeBoundSource {
public:
    static PeriodicExchangeBoundPlan plan_source(const PeriodicPairSource&,Count vectors,
        Count reciprocal_cube_radius,Count densities,const ResourceBudget&);
    PeriodicExchangeBoundSource(const PeriodicPairSource&,const md::Matrix& vectors,
        const std::vector<Count>& offsets,Count reciprocal_cube_radius,const ResourceBudget&);
    static PeriodicExchangeBoundPlan plan_source(const PeriodicPairSource&,Count vectors,
        Count reciprocal_cube_radius,Count densities,Count ao_cube_radius,const ResourceBudget&);
    static PeriodicExchangeBoundPlan plan_envelope_source(const BasisSet&,Count maximum_entries,Count momenta,
        Count retained_bytes,Count vectors,Count reciprocal_cube_radius,Count ao_cube_radius,const ResourceBudget&);
    PeriodicExchangeBoundSource(const PeriodicPairSource&,const md::Matrix& vectors,
        const std::vector<Count>& offsets,Count reciprocal_cube_radius,Count ao_cube_radius,const ResourceBudget&);
    PeriodicExchangeBoundPlan plan_build(Count densities,const ResourceBudget&) const;
    PeriodicExchangeBoundResult build(const std::vector<std::complex<double>>&,Count densities,const ResourceBudget&) const;
    // The typed overload reuses an already admitted native envelope. The
    // overlap overload prepares it with the same solver leaf, once.
    PeriodicExchangeBoundPlan plan_density_envelope(const ResourceBudget&) const;
    PeriodicExchangeEnvelopeResult bound_density_envelope(const BoundedPeriodicDensityEnvelope&,const ResourceBudget&) const;
    PeriodicExchangeBoundPlan plan_occupied_subspaces(Count n_alpha,Count n_beta,
        const BoundedPeriodicRHFOptions&,const ResourceBudget&) const;
    PeriodicExchangeEnvelopeResult bound_occupied_subspaces(const std::vector<std::complex<double>>& overlap,
        Count n_alpha,Count n_beta,const BoundedPeriodicRHFOptions&,const ResourceBudget&) const;
    const PeriodicExchangeBoundPlan& plan() const {return plan_;}
    const PeriodicExchangeBoundReceipt& receipt() const {return receipt_;}
    const md::Matrix& retained_amplitudes() const {return retained_;}
    const md::Matrix& omitted_amplitudes() const {return omitted_;}
    const md::Matrix& reciprocal_coefficients() const {return reciprocal_;}
private:
    PeriodicExchangeBoundPlan plan_;
    PeriodicExchangeBoundReceipt receipt_;
    md::Matrix retained_,omitted_,reciprocal_;
    std::vector<double> weights_;
};
struct PeriodicExchangeAccuracyOptions {
    double matrix_error=0.,energy_error=0.;
    Count ao_completion_cube_radius=2,reciprocal_completion_cube_radius=6;
    bool enabled() const {return matrix_error!=0. || energy_error!=0.;}
};
struct PeriodicExchangeAccuracyPlan {
    PeriodicPairPlan pairs;
    PeriodicExchangeBoundPlan exchange;
    std::optional<PeriodicOneElectronImagePlan> images;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicExchangeAccuracyResult {
    PeriodicExchangeAccuracyOptions targets;
    PeriodicExchangeEnvelopeResult exchange;
    std::optional<PeriodicOneElectronImageResult> images;
    std::string overlap_identity,certificate_identity;
    bool qualified=false,bvk_correction=false;
    double method_scale=0.,fixed_madelung=0.,maximum_error=0.,energy_error_bound=0.;
    double maximum_correction_error=0.,correction_energy_error_bound=0.;
    Count shell_calls=0,primitive_pairs=0;
    std::vector<double> error_bound,correction_error_bound;
};
PeriodicExchangeAccuracyPlan plan_periodic_exchange_accuracy(const BasisSet&,Count cells,Count momenta,
    Count vectors,std::optional<double> pair_cutoff,bool bvk,const PeriodicExchangeAccuracyOptions&,const ResourceBudget&);
PeriodicExchangeAccuracyResult bound_periodic_exchange_accuracy(const BasisSet&,const Eigen::Matrix3d&,
    const md::Matrix& translations,const md::Matrix& momenta,const md::Matrix& vectors,const std::vector<Count>& offsets,
    std::optional<double> pair_cutoff,const std::vector<std::complex<double>>& overlap,const BoundedPeriodicDensityEnvelope&,
    bool bvk,double madelung,double method_scale,const PeriodicExchangeAccuracyOptions&,const ResourceBudget&);

// Method-owned scaling and fixed-Madelung overlap correction. The image
// certificate is native; this is not a user-provided numerical envelope.
void scale_exchange_bound(PeriodicExchangeBoundResult&,double scale);
std::vector<double> bound_bvk_overlap_correction(const PeriodicOneElectronImageResult&,
    const std::vector<std::complex<double>>& overlap,const std::vector<std::complex<double>>& densities,
    Count density_count,const std::vector<double>& weights,double scaled_madelung);
double positive_bound_sum(double,double);
double weighted_exchange_energy_bound(const std::vector<double>&,const std::vector<std::complex<double>>&,
    Count n_basis,const std::vector<double>& weights);
} // namespace vibeqc::manta
