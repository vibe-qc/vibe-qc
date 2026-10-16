#pragma once
#include "vibeqc/manta_bloch_fourier.hpp"
#include "vibeqc/manta_bloch_one_electron.hpp"
#include "vibeqc/periodic.hpp"

namespace vibeqc::manta {
struct PeriodicNuclearPlan {
    Count n_basis=0, vectors=0, cells=0, momenta=0, reciprocal_batch=0, batches=0;
    Count maximum_nuclear_candidates_per_pair=0,nuclear_image_candidates_upper_bound=0;
    Count nuclear_shell_calls=0,overlap_shell_calls=0,fourier_shell_calls=0;
    Count shell_calls=0,primitive_pairs=0,primitive_charge_products_upper_bound=0;
    Count vector_primitive_products=0,contraction_terms=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,driver_live_bytes=0,modeled_peak_bytes=0;
};
struct PeriodicNuclearReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",normalization="libint-shell-axial-v1";
    std::string algorithm="3d-ewald-point-nuclei",boundary="3d-zero-average-g0-omitted";
    std::string nuclear_support="pair-midpoint-sphere-v1",source_identity;
    Count nuclear_shell_calls=0,overlap_shell_calls=0,fourier_shell_calls=0,skipped_shell_pairs=0;
    Count primitive_pairs=0,primitive_charge_products=0,nuclear_images_retained=0;
    Count vector_primitive_products=0,contraction_terms=0,modeled_peak_bytes=0;
    double alpha=0.,nuclear_cutoff_bohr=0.,cell_volume=0.,nuclear_charge=0.,constant_coefficient=0.;
    double cell_imaginary_residual=0.,response_hermiticity_residual=0.;
};
struct PeriodicNuclearResult {
    PeriodicNuclearPlan plan;
    PeriodicNuclearReceipt receipt;
    std::vector<double> short_range_cells,constant_cells;
    std::vector<std::complex<double>> reciprocal_cells,cells,momenta;
};
// Ewald point-nucleus attraction: real erfc images, damped G!=0 Fourier
// components and +pi*sum(Z)/(alpha^2*V) overlap. Explicit inversion-closed
// G and AO translation supports, with midpoint-centered physical nuclear
// images selected by the same selector as the existing lattice integrals.
// This is a finite all-electron operator, not a complete periodic SCF source.
PeriodicNuclearPlan plan_periodic_nuclear(const BasisSet&,const PeriodicSystem&,
    Count vectors,Count cells,Count momenta,double alpha,double nuclear_cutoff,
    Count reciprocal_batch,std::optional<double> pair_cutoff,const ResourceBudget&);
PeriodicNuclearResult compute_periodic_nuclear(const BasisSet&,const PeriodicSystem&,
    const md::Matrix& vectors,const md::Matrix& translations,const md::Matrix& momenta,
    double alpha,double nuclear_cutoff,Count reciprocal_batch,
    std::optional<double> pair_cutoff,const ResourceBudget&);
} // namespace vibeqc::manta
