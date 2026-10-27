#pragma once
#include "vibeqc/manta_periodic_bupo.hpp"
#include "manta_gaussian_domain_internal.hpp"
#include "vibeqc/manta_periodic_one_electron_bounds.hpp"

namespace vibeqc::manta::pair_image_detail {
// Native-only producer. The finite canonical inventory, original basis and
// already qualified lattice geometry determine every envelope, never density.
struct Plan {
    Count radius=0,cells=0,shell_calls=0,primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct Envelope {
    PeriodicBupoPairImageReceipt receipt;
    md::Matrix retained,omitted;
};
struct FourierEnvelope {
    md::Matrix retained,omitted;
    double decay=0.; // |B(Q)| <= amplitude exp(-decay |Q|^2).
};
Plan plan(const BasisSet&,const std::vector<PeriodicPairEntry>&,const std::vector<LatticeIndex>& original_images,
    Count momenta,const ResourceBudget&);
// A completion cube may be smaller than the retained inventory: including
// retained exterior images in the positive exterior sum is conservative.
Plan plan(const BasisSet&,Count maximum_entries,Count momenta,Count cube_radius,const ResourceBudget&);
Envelope produce(const BasisSet&,const Eigen::Matrix3d&,const std::vector<PeriodicPairEntry>&,
    const gaussian_domain_detail::Geometry&,const std::string& pair_identity,const std::string& kernel_identity,
    const Plan&);
PeriodicOneElectronImageResult one_electron(const BasisSet&,const Eigen::Matrix3d&,
    const std::vector<PeriodicPairEntry>&,const std::string& pair_identity,const Plan&);
double point_kernel_bound(const Envelope&,const gaussian_domain_detail::Geometry&);
double density_trace_bound(const md::Matrix&,const std::vector<std::complex<double>>&,const std::vector<double>&);
FourierEnvelope fourier_images(const BasisSet&,const Eigen::Matrix3d&,
    const std::vector<PeriodicPairEntry>&,const Plan&,Count decay_bin=0,Count decay_bins=1);
std::array<double,2> reciprocal_exchange_sums(const Eigen::Matrix3d&,const Eigen::Vector3d&,
    const std::vector<LatticeIndex>& sorted_labels,double fourier_decay,Count cube_radius,
    const std::string& kernel="coulomb",double omega=0.);
} // namespace vibeqc::manta::pair_image_detail
