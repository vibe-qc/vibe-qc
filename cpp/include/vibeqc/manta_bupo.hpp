#pragma once
#include "vibeqc/manta_bupo_errors.hpp"
#include "vibeqc/manta_moments.hpp"

namespace vibeqc::manta {
struct BupoOptions {
    int order=6;
    bool collect_timings=false;
    double opening=.5,integral_tolerance=1e-10;
    std::string near_backend="manta";
    ExtentOptions extents;
    BubbleOptions bubbles;
};
struct BupoPlan {
    Count n_basis=0,shell_pairs=0,nodes=0,moments=0;
    Count near_shell_calls=0,near_primitive_quartets=0,far_blocks=0;
    Count moment_panels=0,moment_primitive_pairs=0,traversal_visits=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
    double maximum_integral_error=0.,maximum_compact_error=0.,maximum_penetration_error=0.,maximum_moment_tail_error=0.;
    std::string partition_identity;
    bool timings_collected=false;
    double setup_wall_seconds=0.,provider_setup_wall_seconds=0.,extent_setup_wall_seconds=0.;
    double tree_setup_wall_seconds=0.,error_model_setup_wall_seconds=0.,partition_setup_wall_seconds=0.;
};
struct BupoReceipt {
    std::string source_identity,density_identity,partition_identity,near_provider,moment_provider="manta";
    std::string algorithm="bupo-j-cartesian",normalization="libint-shell-axial-v1";
    Count near_shell_calls=0,near_primitive_quartets=0,near_null_tiles=0,far_blocks=0;
    Count moment_panels=0,moment_primitive_pairs=0,traversal_visits=0,modeled_peak_bytes=0;
    bool near_primitive_evaluations_measured=false;
    double density_l1=0.,coulomb_energy=0.,approximation_j_bound=0.;
    double compact_j_bound=0.,penetration_j_bound=0.,moment_tail_j_bound=0.;
    double wall_seconds=0.;
    bool timings_collected=false;
    double validation_wall_seconds=0.,leaf_moments_wall_seconds=0.,upward_wall_seconds=0.;
    double far_admission_wall_seconds=0.,far_apply_wall_seconds=0.,near_integrals_wall_seconds=0.;
    double near_digest_wall_seconds=0.,downward_wall_seconds=0.,reverse_moments_wall_seconds=0.;
    double finalize_wall_seconds=0.;
};
struct BupoResult { Eigen::MatrixXd J; BupoReceipt receipt; };
class BupoJBuilder final : public JKBuilder {
public:
    BupoJBuilder(const BasisSet&,const BupoOptions&,const ResourceBudget&);
    const BupoPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_; }
    BupoResult build(const Eigen::MatrixXd&) const;
    Eigen::MatrixXd build_J(const Eigen::MatrixXd& d) const override { return build(d).J; }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd&) const override;
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd&,double alpha=1.) const override;
private:
    BupoOptions options_;
    ResourceBudget budget_;
    BupoPlan plan_;
    std::string source_;
    std::shared_ptr<const IntegralProvider> provider_;
    std::shared_ptr<const BubbleTree> tree_;
    std::shared_ptr<const BubbleErrorModel> errors_;
    std::vector<Eigen::Index> offsets_;
    bool accepted(const BubbleBlock&,BubbleInteraction* output=nullptr) const;
};
} // namespace vibeqc::manta
