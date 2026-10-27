#pragma once
#include "vibeqc/manta_periodic_pairs.hpp"
#include "vibeqc/bounded_periodic_rhf_solver.hpp"
#include "vibeqc/manta_ri_metric.hpp"

namespace vibeqc::manta {
struct PeriodicOneElectronImagePlan {
    Count n_basis=0,momenta=0,cube_radius=0,cube_cells=0;
    Count shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicOneElectronImageReceipt {
    std::string qualification="full-ao-image-overlap-and-kinetic-bound";
    std::string source_identity,pair_source_identity;
    Count cube_radius=0,cube_cells=0,shell_pairs_evaluated=0,primitive_pairs_evaluated=0;
    Count retained_ordered_pairs=0,omitted_ordered_pairs_in_cube=0;
    double real_metric_lower=0,maximum_overlap_error=0,maximum_kinetic_error=0;
    bool roundoff_included=false;
};
struct PeriodicOneElectronImageResult {
    PeriodicOneElectronImagePlan plan;
    PeriodicOneElectronImageReceipt receipt;
    // Nonnegative AO matrices uniform over every k. Includes physical-cutoff
    // exclusions and all translations absent from the original inventory.
    md::Matrix overlap_error_bound,kinetic_error_bound;
};
PeriodicOneElectronImagePlan plan_periodic_one_electron_images(const PeriodicPairSource&,const ResourceBudget&);
PeriodicOneElectronImageResult bound_periodic_one_electron_images(const PeriodicPairSource&,const ResourceBudget&);
// An explicit completion cube preserves a bound even when retained images
// lie outside it: the exterior sum then includes extra positive terms.
PeriodicOneElectronImagePlan plan_periodic_one_electron_images(const PeriodicPairSource&,Count cube_radius,const ResourceBudget&);
PeriodicOneElectronImageResult bound_periodic_one_electron_images(const PeriodicPairSource&,Count cube_radius,const ResourceBudget&);

// Native method helpers. Callers admit reconstruction/sorting scratch and
// the retained pair source jointly before passing physical Cartesian cells.
PeriodicPairSource make_periodic_accuracy_pair_source(const BasisSet&,const Eigen::Matrix3d&,
    const md::Matrix& translations,const md::Matrix& momenta,std::optional<double>,const ResourceBudget&);
PeriodicOneElectronImagePlan plan_periodic_one_electron_images(const BasisSet&,Count entries,Count momenta,
    Count cube_radius,Count retained_bytes,const ResourceBudget&);

// Conditional rank enclosure for the existing omitted-image source bounds.
// Integral construction and occupied-space rotation remain outside its scope.
struct PeriodicOverlapImageRankResult {
    bool qualified=false;
    double maximum_perturbation_bound=0.,minimum_stability_radius=0.;
    std::vector<double> perturbation_bounds,projection_bounds;
    std::vector<HermitianSpectrumCertificate> spectra;
    std::string certificate_identity;
};
HermitianSpectrumPlan plan_periodic_overlap_image_rank(Count n,Count nk,
    const BoundedPeriodicRHFOptions&,const ResourceBudget&);
PeriodicOverlapImageRankResult bound_periodic_overlap_image_rank(Count n,Count nk,
    const std::vector<std::complex<double>>& overlap,const md::Matrix& image_bounds,
    Count expected_rank,const BoundedPeriodicRHFOptions&,const ResourceBudget&);

// Optional complete-method contract, separate from finite RI coupling error.
struct PeriodicOneElectronAccuracyOptions {
    double overlap_error=0.,kinetic_error=0.,kinetic_energy_error=0.;
    Count completion_cube_radius=2;
    bool overlap_image_rank=false;
    bool enabled() const { return overlap_image_rank || overlap_error!=0. || kinetic_error!=0. || kinetic_energy_error!=0.; }
};
struct PeriodicOneElectronAccuracyPlan {
    PeriodicPairPlan pairs;
    PeriodicOneElectronImagePlan images;
    std::optional<HermitianSpectrumPlan> overlap_rank;
    Count modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicOneElectronAccuracyResult {
    PeriodicOneElectronImageResult images;
    PeriodicOneElectronAccuracyOptions targets;
    std::optional<PeriodicOverlapImageRankResult> overlap_rank;
    bool qualified=false;
    double kinetic_energy_error_bound=0.,density_l1_limit=0.;
    std::string density_envelope_identity,certificate_identity;
    Count shell_calls=0,primitive_pairs=0;
};
PeriodicOneElectronAccuracyPlan plan_periodic_one_electron_accuracy(const BasisSet&,Count cells,Count momenta,
    std::optional<double> pair_cutoff,const PeriodicOneElectronAccuracyOptions&,const ResourceBudget&,
    const BoundedPeriodicRHFOptions& = {});
PeriodicOneElectronAccuracyResult bound_periodic_one_electron_accuracy(const BasisSet&,const Eigen::Matrix3d&,
    const md::Matrix& translations,const md::Matrix& momenta,std::optional<double> pair_cutoff,
    const BoundedPeriodicDensityEnvelope&,const PeriodicOneElectronAccuracyOptions&,const ResourceBudget&,bool enforce_targets=true,
    const std::vector<std::complex<double>>& overlap={},const BoundedPeriodicRHFOptions& = {});
} // namespace vibeqc::manta
