#pragma once
#include "vibeqc/manta_fitting.hpp"
#include "vibeqc/manta_fourier.hpp"
#include "vibeqc/periodic.hpp"

namespace vibeqc::manta {
// Raw erfc pieces of Ye and Berkelbach, JCP 154,131104 (2021), Eqs.9,12.
// The existing GDF sphere/capsule selector owns the numerical image domain.
// Finite SR zero modes remain included; no jellium subtraction or exchange
// singular correction belongs to this source. No fitting solve is performed.
struct PeriodicGDFShortRangeOptions {
    double omega=0.5,pair_cutoff_bohr=6.,auxiliary_cutoff_bohr=6.;
    double integral_screen_error=0.;
    Count image_candidate_cap=1000000;
    bool compute_metric=true;
};
// Physical interaction is independent of the numerical Coulomb split above.
// erfc retains its finite p=0 coefficient; Coulomb and erf omit their poles.
struct PeriodicGDFKernelOptions {
    std::string kernel="coulomb";
    double omega=0.;
};
struct PeriodicGDFShortRangePlan {
    Count n_aux=0,n_orb=0,momenta=0;
    bool compute_metric=true;
    Count metric_image_candidates=0,pair_image_candidates=0,auxiliary_image_candidates=0;
    Count metric_shell_calls=0,three_center_shell_calls=0,shell_calls=0,primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,driver_live_bytes=0,modeled_peak_bytes=0;
    FittingPlan metric,three_center;
};
struct PeriodicGDFShortRangeReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",normalization="libint-shell-axial-v1";
    std::string algorithm="finite-3d-gdf-short-range",kernel="erfc";
    std::string convention="raw-erfc;finite-zero-included;no-jellium-subtraction;no-exchange-correction";
    std::string orbital_basis_identity,auxiliary_basis_identity;
    std::string metric_source_identity,three_center_source_identity,source_identity;
    double omega=0.,pair_cutoff_bohr=0.,auxiliary_cutoff_bohr=0.,integral_screen_error=0.;
    Count metric_shell_calls=0,three_center_shell_calls=0,pair_shell_images=0,primitive_products=0;
    Count phase_products=0,phase_accumulations=0,modeled_peak_bytes=0,threads=1;
};
struct PeriodicGDFShortRangeResult {
    PeriodicGDFShortRangePlan plan;
    PeriodicGDFShortRangeReceipt receipt;
    // Row-major [P,Q] and [ket k,P,mu,nu], individually owned.
    std::vector<std::complex<double>> metric_values,three_center_values;
};
PeriodicGDFShortRangePlan plan_periodic_gdf_short_range(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,Count momenta,const PeriodicGDFShortRangeOptions&,const ResourceBudget&);
PeriodicGDFShortRangeResult compute_periodic_gdf_short_range(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,const Eigen::Vector3d& q,const md::Matrix& ket_kpoints,
    const PeriodicGDFShortRangeOptions&,const ResourceBudget&);

// Complete finite Coulomb source, still unfactored. The reciprocal vectors
// are the caller's explicit G+q domain. The shared GDF jellium kernel removes
// the finite SR zero exactly once and omits the LR zero. No exchange seam.
// Ye and Berkelbach, JCP 154,131104 (2021), Eq.12; Bloch Fourier phases as
// Sun et al., JCP 147,164119 (2017), Eq.16.
struct PeriodicGDFPlan {
    Count n_aux=0,n_orb=0,momenta=0,vectors=0,panel_vectors=0,panel_count=0;
    bool compute_metric=true;
    Count shell_calls=0,primitive_products=0,vector_primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,modeled_peak_bytes=0;
    PeriodicGDFShortRangePlan short_range;
    FourierPlan auxiliary_fourier,ao_pair_fourier;
};
struct PeriodicGDFReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",normalization="libint-shell-axial-v1";
    std::string algorithm="finite-3d-range-separated-gdf",kernel="coulomb";
    std::string convention="jellium;finite-SR-zero-removed-once;LR-zero-omitted;no-exchange-correction";
    std::string metric_source_identity,three_center_source_identity,source_identity;
    PeriodicGDFShortRangeReceipt short_range;
    bool finite_zero_removed=false;
    bool zero_transfer=false,physical_zero_included=false,real_short_range=true;
    double physical_omega=0.;
    Count reciprocal_vectors=0,auxiliary_fourier_shell_calls=0,pair_fourier_shell_calls=0;
    Count shell_calls=0,primitive_products=0,vector_primitive_products=0;
    Count phase_products=0,phase_accumulations=0,contraction_terms=0;
    Count modeled_peak_bytes=0,threads=1;
};
struct PeriodicGDFResult {
    PeriodicGDFPlan plan;
    PeriodicGDFReceipt receipt;
    std::vector<std::complex<double>> metric_values,three_center_values;
};
PeriodicGDFPlan plan_periodic_gdf(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,Count vectors,Count momenta,const PeriodicGDFShortRangeOptions&,const ResourceBudget&,
    const PeriodicGDFKernelOptions& = {});
PeriodicGDFResult compute_periodic_gdf(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,const Eigen::Vector3d& q,const md::Matrix& ket_kpoints,const md::Matrix& vectors,
    const PeriodicGDFShortRangeOptions&,const ResourceBudget&,const PeriodicGDFKernelOptions& = {});

struct PeriodicGDFTilesPlan {
    Count n_aux=0,n_orb=0,momenta=0,vectors=0,panel_vectors=0;
    Count metric_shell_calls=0,coupling_shell_calls=0,metric_primitive_products=0,coupling_primitive_products=0;
    Count metric_scalar_terms=0,coupling_scalar_terms=0,metric_kernel_terms=0,coupling_kernel_terms=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0;
    Count maximum_tile_elements=0,retained_bytes=0,metric_peak_bytes=0,coupling_peak_bytes=0,modeled_peak_bytes=0;
    Count external_live_bytes=0;
    bool compute_metric=true;
};
struct PeriodicGDFTile {
    std::vector<Count> shape,offsets;
    std::vector<std::complex<double>> values;
    PeriodicGDFReceipt receipt;
};
PeriodicGDFTilesPlan plan_periodic_gdf_tiles(const BasisSet& orbital,const BasisSet& auxiliary,
    const PeriodicSystem&,Count vectors,Count momenta,const PeriodicGDFShortRangeOptions&,const ResourceBudget&,
    const PeriodicGDFKernelOptions& = {});

// Immutable finite source. One full dense metric or one [k,P,mu,nu] shell
// tile may be live/exported under this plan; no complete coupling tensor.
// Work bounds cover one metric and one complete shell traversal, not a
// lifetime call quota. Consumers admit all of their repeated passes jointly.
class PeriodicGDFTiles {
public:
    PeriodicGDFTiles(const BasisSet& orbital,const BasisSet& auxiliary,const PeriodicSystem&,
        const Eigen::Vector3d& q,const md::Matrix& ket_kpoints,const md::Matrix& vectors,
        const PeriodicGDFShortRangeOptions&,const ResourceBudget&,const PeriodicGDFKernelOptions& = {});
    const PeriodicGDFTilesPlan& plan() const { return plan_; }
    const PeriodicGDFReceipt& source() const { return source_; }
    const BasisSet& orbital() const { return orbital_; }
    const BasisSet& auxiliary() const { return auxiliary_; }
    const Eigen::Matrix3d& lattice() const { return system_.lattice; }
    const md::Matrix& momenta() const { return k_; }
    const Eigen::Vector3d& transfer() const { return q_; }
    PeriodicGDFTile metric() const;
    PeriodicGDFTile evaluate(Count auxiliary_shell,Count bra_shell,Count ket_shell) const;
private:
    PeriodicGDFTilesPlan plan_;
    BasisSet orbital_,auxiliary_;
    PeriodicSystem system_; // lattice/dimension only; no implicit symmetry state
    Eigen::Vector3d q_;
    md::Matrix k_,vectors_;
    PeriodicGDFShortRangeOptions options_;
    PeriodicGDFKernelOptions physical_;
    PeriodicGDFReceipt source_;
};
} // namespace vibeqc::manta
