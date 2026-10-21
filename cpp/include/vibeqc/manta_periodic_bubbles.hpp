#pragma once
#include "vibeqc/manta_bubbles.hpp"
#include "vibeqc/manta_periodic_pairs.hpp"
#include "vibeqc/manta_multipole.hpp"

namespace vibeqc::manta {
struct PeriodicBubblePlan {
    BubblePlan geometry;
    Count pairs=0,maximum_nodes=0,moments=0,maximum_translation_calls=0;
    Count shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicBubbleDensity {
    md::Matrix moments,full_moment_bounds,tail_moment_bounds;
    std::vector<double> coefficient_l1;
    PeriodicPairDensityReceipt pair_receipt;
    std::string source_identity,density_identity;
    std::string qualification="periodic-pair-image-hierarchy-and-adjoint";
    Count translation_calls=0,translation_terms=0,modeled_peak_bytes=0;
};
struct PeriodicBubbleAdjoint {
    PeriodicPairAdjoint pair;
    std::string source_identity,potential_identity;
    Count translation_calls=0,translation_terms=0,modeled_peak_bytes=0;
};

// The only input is a native immutable pair-image producer. A molecular
// triangular inventory cannot supply periodic image ownership implicitly.
class PeriodicBubbleTree {
public:
    static PeriodicBubblePlan plan_source(const PeriodicPairSource&,const BubbleOptions&,const ResourceBudget&);
    PeriodicBubbleTree(std::shared_ptr<const PeriodicPairSource>,const BubbleOptions&,const ResourceBudget&);
    const PeriodicBubblePlan& plan() const { return plan_; }
    const PeriodicPairPlan& pair_plan() const { return source_->plan(); }
    const std::vector<BubbleNode>& nodes() const { return tree_->nodes(); }
    const std::vector<Count>& pair_order() const { return tree_->pair_order(); }
    const std::string& source_identity() const { return tree_->source_identity(); }
    const std::string& pair_source_identity() const { return source_->receipt().source_identity; }
    int depth() const { return tree_->depth(); }
    PeriodicBubbleDensity contract(const std::vector<std::complex<double>>&) const;
    PeriodicBubbleAdjoint adjoint(const md::Matrix&) const;
    std::unique_ptr<BubbleTraversal> traversal(double,const ResourceBudget&) const;
private:
    friend class PeriodicBupoSource;
    PeriodicBubblePlan plan_;
    std::shared_ptr<const PeriodicPairSource> source_;
    std::shared_ptr<const BubbleTree> tree_;
    ResourceBudget workspace_budget_;
};
} // namespace vibeqc::manta
