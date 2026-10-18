#pragma once
#include "vibeqc/manta_periodic_pairs.hpp"

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
} // namespace vibeqc::manta
