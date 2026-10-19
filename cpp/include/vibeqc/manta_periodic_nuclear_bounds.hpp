#pragma once
#include "vibeqc/manta_periodic_nuclear.hpp"
#include "vibeqc/manta_periodic_pairs.hpp"

namespace vibeqc::manta {
struct PeriodicNuclearBoundOptions {
    Count ao_completion_cube_radius=2,real_completion_cube_radius=4,reciprocal_completion_cube_radius=6;
};
struct PeriodicNuclearBoundPlan {
    PeriodicPairPlan pairs;
    Count n_basis=0,momenta=0,vectors=0,points=0;
    Count ao_cube_cells=0,real_cube_cells=0,reciprocal_cube_cells=0,maximum_nuclear_candidates_per_pair=0;
    Count maximum_ordered_pairs=0,maximum_gaussian_terms=0,real_terms=0;
    Count shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicNuclearBoundReceipt {
    std::string qualification="full-images-pair-midpoint-sphere-nuclear-bound";
    std::string boundary="3d-zero-average-g0-omitted",nuclear_support="pair-midpoint-sphere-v1";
    std::string source_identity,nuclear_source_identity,pair_source_identity,pair_image_identity;
    Count ordered_pairs=0,majorant_primitive_pairs=0,real_terms=0;
    double alpha=0,nuclear_cutoff_bohr=0,nuclear_absolute_charge=0;
    double point_kernel_bound=0,reciprocal_weight_bound=0,maximum_potential_error=0;
    double nuclear_real_error_bound=0,nuclear_reciprocal_error_bound=0,nuclear_energy_error_bound=0;
    bool roundoff_included=false;
};
struct PeriodicNuclearBoundResult {
    PeriodicNuclearBoundPlan plan;
    PeriodicNuclearBoundOptions options;
    PeriodicNuclearBoundReceipt receipt;
    // Nonnegative AO bounds uniform in k. Independent ordered arms retain
    // the actual physical midpoint-sphere membership of the finite source.
    md::Matrix potential_error_bound,real_error_bound,reciprocal_error_bound,pair_image_error_bound;
};
PeriodicNuclearBoundPlan plan_periodic_nuclear_bounds(const BasisSet&,const PeriodicSystem&,
    Count vectors,Count cells,Count momenta,double alpha,double nuclear_cutoff,
    std::optional<double> pair_cutoff,const PeriodicNuclearBoundOptions&,const ResourceBudget&);
PeriodicNuclearBoundResult bound_periodic_nuclear(const BasisSet&,const PeriodicSystem&,
    const md::Matrix& vectors,const md::Matrix& cells,const md::Matrix& momenta,double alpha,double nuclear_cutoff,
    std::optional<double> pair_cutoff,const PeriodicNuclearBoundOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
