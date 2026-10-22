#pragma once
#include "vibeqc/manta_bupo_errors.hpp"
// Internal native producer adapter. The caller supplies exactly the images
// admitted by its finite-domain selector, never a canonical AO-pair inventory.
namespace vibeqc::manta::bupo_detail {
// An ordered finite producer supplies every qualified pair-image extent.
// This adapter never substitutes the molecular/canonical Ewald inventory.
struct OrderedPairImageBubbles {
    static BubblePlan plan(Count pairs,int order,const ResourceBudget&);
    static std::shared_ptr<const BubbleTree> build(const PairExtents&,const ResourceBudget&);
};
struct AuxiliaryImageBubbles {
    static BubblePlan plan(Count images,int order,const ResourceBudget&);
    static std::shared_ptr<const BubbleTree> build(const AuxiliaryExtent&,
        const std::vector<md::Point>& centers,const std::string& source,const ResourceBudget&);
};
}
