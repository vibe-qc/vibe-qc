#pragma once
#include "vibeqc/manta_bupo_errors.hpp"
#include "vibeqc/manta_moments.hpp"
#include "vibeqc/manta_fitting.hpp"
#include <functional>

namespace vibeqc::manta {
struct BupoFittingOptions {
    int order=6;
    double opening=.5,coupling_tolerance=1e-10;
    ExtentOptions extents;
    BubbleOptions bubbles;
};
struct BupoFittingPlan {
    Count n_basis=0,n_auxiliary=0,shell_pairs=0,nodes=0,moments=0;
    // Executed counts per pass; aggregate budgets admit setup + both passes.
    Count near_shell_calls=0,near_primitive_products=0,far_blocks=0,traversal_visits=0;
    Count pair_moment_panels=0,auxiliary_moment_panels=0,moment_primitive_products=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
    double maximum_exact_coupling=0.,maximum_coupling_error=0.,maximum_compact_error=0.,maximum_penetration_error=0.,maximum_moment_tail_error=0.;
    std::string partition_identity;
};
struct BupoFittingReceipt {
    std::string source_identity,input_identity,partition_identity,direction;
    std::string provider="manta",algorithm="bupo-paired-three-center",normalization="libint-shell-axial-v1";
    Count near_shell_calls=0,near_primitive_products=0,far_blocks=0,traversal_visits=0;
    Count pair_moment_panels=0,auxiliary_moment_panels=0,moment_primitive_products=0,modeled_peak_bytes=0;
    double input_l1=0.,approximation_bound=0.,compact_bound=0.,penetration_bound=0.,moment_tail_bound=0.;
};
struct BupoFittingForward { Eigen::VectorXd values; BupoFittingReceipt receipt; };
struct BupoFittingAdjoint { Eigen::MatrixXd values; BupoFittingReceipt receipt; };
// Fixed molecular full-Coulomb B~ and its adjoint. Auxiliary shells are atomic
// leaves; pair bubbles use upward moments and the transposed downward map.
// No global auxiliary/AO-pair tensor or rectangular interaction list is stored.
class BupoFittingCoupling {
public:
    BupoFittingCoupling(const BasisSet&,const BasisSet&,const BupoFittingOptions&,const ResourceBudget&);
    const BupoFittingPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_; }
    BupoFittingForward forward(const Eigen::MatrixXd&) const;
    BupoFittingAdjoint adjoint(const Eigen::VectorXd&) const;
private:
    BupoFittingOptions options_;
    ResourceBudget budget_;
    BupoFittingPlan plan_;
    std::string source_,partition_seed_;
    std::shared_ptr<const FittingTiles> provider_;
    std::shared_ptr<const BubbleTree> tree_;
    std::shared_ptr<const BubbleErrorModel> errors_;
    AuxiliaryExtents auxiliary_;
    using Consume=std::function<void(Count,Count,const BubbleInteraction*)>;
    std::string walk(const Consume&,Count& visits,Count* queries=nullptr) const;
    void finish(BupoFittingReceipt&) const;
};
} // namespace vibeqc::manta
