#pragma once
#include "vibeqc/manta_periodic_bubbles.hpp"
#include "vibeqc/manta_periodic_ewald.hpp"
#include "vibeqc/manta_bupo_errors.hpp"
#include "vibeqc/manta_gaussian_majorants.hpp"

namespace vibeqc::manta {
namespace gaussian_domain_detail { struct Geometry; }
namespace pair_image_detail { struct Envelope; }
struct PeriodicBupoOptions {
    double opening=.5,integral_tolerance=1e-10;
    bool bound_infinite_domain=false;
    bool bound_pair_images=false;
};
struct PeriodicBupoPlan {
    int order=0;
    Count n_basis=0,momenta=0,pairs=0,nodes=0,moments=0,maximum_queries=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
    Count domain_majorant_terms=0,domain_pair_interactions=0;
    Count image_cube_radius=0,image_cells=0,image_primitive_products=0;
};
struct PeriodicBupoPartition {
    BubbleTraversalReceipt traversal;
    Count approximate_blocks=0,direct_blocks=0,approximate_pairs=0,direct_pairs=0;
    double maximum_integral_error=0;
    std::string identity;
};
struct PeriodicBupoDomainReceipt {
    std::string qualification="retained-pair-source-infinite-ewald-domain-bound";
    std::string source_identity,pair_source_identity,kernel_identity,envelope_identity,density_identity;
    std::string real_domain_identity,reciprocal_domain_identity;
    Count real_cube_radius=0,reciprocal_cube_radius=0,majorant_terms=0;
    Count pair_interactions=0,majorant_pairs_evaluated=0,majorant_shell_calls=0;
    double real_energy_error_bound=0,reciprocal_energy_error_bound=0,energy_error_bound=0;
    double maximum_j_error_bound=0,maximum_integral_error=0;
    double minimum_effective_eta=std::numeric_limits<double>::infinity();
    double minimum_omitted_real_distance=std::numeric_limits<double>::infinity();
    bool roundoff_included=false,pair_image_omission_included=false;
};
struct PeriodicBupoPairImageReceipt {
    std::string qualification="full-pair-image-zero-average-coulomb-bound";
    std::string source_identity,pair_source_identity,kernel_identity,density_identity;
    Count cube_radius=0,cube_cells=0,shell_pairs_evaluated=0,primitive_pairs_evaluated=0;
    Count retained_ordered_pairs=0,omitted_ordered_pairs_in_cube=0;
    double real_metric_lower=0,reciprocal_metric_lower=0,volume_lower=0,maximum_gaussian_exponent=0;
    double real_kernel_bound=0,reciprocal_kernel_bound=0,gauge_kernel_bound=0,kernel_bound=0;
    double retained_density_mass_bound=0,omitted_density_mass_bound=0;
    double maximum_retained_pair_mass=0,maximum_omitted_pair_mass=0;
    double energy_error_bound=0,maximum_j_error_bound=0;
    bool roundoff_included=false;
};
struct PeriodicBupoReceipt {
    PeriodicEwaldReceipt direct;
    PeriodicBupoPartition partition;
    std::string source_identity,pair_source_identity,tree_identity,kernel_identity,density_identity;
    std::string qualification="finite-periodic-bupo-j";
    double direct_energy=0,far_energy=0,energy=0,trace_energy=0;
    double energy_error_bound=0,maximum_j_error_bound=0,charge=0,hermiticity_residual=0;
    double finite_energy_error_bound=0,finite_maximum_j_error_bound=0;
    std::optional<PeriodicBupoDomainReceipt> domain;
    std::optional<PeriodicBupoPairImageReceipt> pair_images;
    Count forward_shell_calls=0,adjoint_shell_calls=0,translation_calls=0,far_matrix_terms=0;
    Count modeled_peak_bytes=0;
    bool roundoff_included=false;
};
struct PeriodicBupoResult {
    PeriodicBupoPlan plan;
    PeriodicBupoReceipt receipt;
    std::vector<std::complex<double>> values;
    // Uniform over k: absolute phases are one; both canonical scatter arms
    // contribute, including a nonzero-image diagonal AO component twice.
    md::Matrix j_error_bound;
    std::optional<md::Matrix> domain_j_error_bound;
    std::optional<md::Matrix> pair_image_j_error_bound,retained_pair_mass_bound,omitted_pair_mass_bound;
};
class PeriodicBupoSource {
public:
    static PeriodicBupoPlan plan_source(const PeriodicBubbleTree&,const PeriodicEwaldSource&,
        Count near_cells,const PeriodicBupoOptions&,const ResourceBudget&);
    PeriodicBupoSource(std::shared_ptr<const PeriodicBubbleTree>,std::shared_ptr<const PeriodicEwaldSource>,
        const std::vector<LatticeIndex>& near,const PeriodicBupoOptions&,const ResourceBudget&);
    const PeriodicBupoPlan& plan() const { return plan_; }
    const PeriodicBupoPartition& partition() const { return partition_; }
    const std::string& source_identity() const { return source_identity_; }
    PeriodicBupoResult build(const std::vector<std::complex<double>>&) const;
private:
    friend struct PeriodicBupoHFDriver;
    friend class PeriodicElectrostaticSource;
    struct Far { Eigen::MatrixXd values; double error=0; };
    bool accepted(const BubbleBlock&,Far*) const;
    void bound_domain(const std::vector<std::complex<double>>&,PeriodicBupoResult&) const;
    void bound_pair_images(const std::vector<std::complex<double>>&,PeriodicBupoResult&) const;
    PeriodicBupoPlan plan_;
    std::shared_ptr<const PeriodicBubbleTree> tree_;
    std::shared_ptr<const PeriodicEwaldSource> direct_;
    std::shared_ptr<const BubbleErrorModel> errors_;
    std::vector<LatticeIndex> near_;
    PeriodicBupoOptions options_;
    ResourceBudget workspace_budget_;
    PeriodicBupoPartition partition_;
    std::string source_identity_;
    std::shared_ptr<const gaussian_domain_detail::Geometry> domain_geometry_;
    std::vector<GaussianMajorant> majorants_;
    std::optional<PeriodicBupoDomainReceipt> domain_receipt_;
    std::shared_ptr<const pair_image_detail::Envelope> image_envelope_;
};
} // namespace vibeqc::manta
