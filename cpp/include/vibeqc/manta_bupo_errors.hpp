#pragma once
#include "vibeqc/manta_bubbles.hpp"
#include "vibeqc/manta_multipole.hpp"

namespace vibeqc::manta {
struct BubbleErrorPlan {
    int order=0;
    Count nodes=0,moments=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct BubbleEnvelope {
    std::vector<double> full_moments,tail_moments;
    double potential=0.;
};
struct BubbleErrorBound {
    double compact_remainder=0.,tail_penetration=0.,full_moment_tail=0.,total=0.;
    std::string source_identity;
};
struct BubbleInteraction {
    MultipoleMatrix matrix;
    BubbleErrorBound error;
};
BubbleErrorPlan plan_bubble_errors(const BubbleTree&,int order,const ResourceBudget&);
// Uniform per-AO-product envelopes. These are maxima over descendants, not
// charge/density sums. Density weights belong to the consuming J build.
class BubbleErrorModel {
public:
    BubbleErrorModel(std::shared_ptr<const BubbleTree>,int order,const ResourceBudget&);
    const BubbleErrorPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_; }
    const std::vector<BubbleEnvelope>& envelopes() const { return envelopes_; }
    BubbleInteraction interaction(Count target,Count source) const;
    // Pair-bubble target and single-auxiliary-shell source. Their geometric
    // supports must be disjoint; the inventories have distinct physical roles.
    BubbleInteraction auxiliary_interaction(Count target,const AuxiliaryExtent& source) const;
    BubbleErrorBound bound(Count target,Count source) const { return interaction(target,source).error; }
private:
    std::shared_ptr<const BubbleTree> tree_;
    BubbleErrorPlan plan_;
    ResourceBudget budget_;
    std::string source_;
    std::vector<BubbleEnvelope> envelopes_;
};
} // namespace vibeqc::manta
