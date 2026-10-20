#pragma once
#include "vibeqc/manta_extents.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <memory>
#include <optional>

namespace vibeqc::manta {
namespace bupo_detail { struct AuxiliaryImageBubbles; struct OrderedPairImageBubbles; }
class PeriodicPairSource;
class PeriodicBubbleTree;
struct BubbleOptions {
    Count leaf_capacity=150;
    int branching=3,maximum_depth=32;
};
struct BubblePlan {
    Count pairs=0,maximum_nodes=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct BubbleNode {
    std::array<double,3> center{};
    double radius=0.;
    Count begin=0,end=0,parent=0;
    std::array<Count,8> children{};
    int child_count=0,depth=0;
    bool always_near=false;
};
struct BubbleBlock {
    Count target=0,source=0,pair_interactions=0;
    bool far_candidate=false;
    double separation_ratio=0.;
};
struct BubbleTraversalReceipt {
    Count visited_node_pairs=0,near_blocks=0,far_blocks=0,pair_interactions=0;
    Count refined_candidates=0,maximum_stack=0,modeled_peak_bytes=0;
    bool complete=false,failed=false;
    std::string source_identity,partition_identity;
};
class BubbleTree {
public:
    BubbleTree(const PairExtents&,const BubbleOptions&,const ResourceBudget&);
    const std::vector<BubbleNode>& nodes() const { return nodes_; }
    const std::vector<Count>& pair_order() const { return pair_order_; }
    const std::vector<PairExtent>& pairs() const { return pairs_; }
    const BubblePlan& plan() const { return plan_; }
    const std::string& source_identity() const { return source_identity_; }
    int depth() const { return depth_; }
    Count retained_bytes() const { return retained_bytes_; }
private:
    friend class PeriodicBubbleTree;
    friend struct bupo_detail::AuxiliaryImageBubbles;
    friend struct bupo_detail::OrderedPairImageBubbles;
    BubbleTree(const PairExtents&,const BubbleOptions&,const ResourceBudget&,bool molecular,const std::string& domain="");
    static BubblePlan plan_periodic(const PeriodicPairSource&,const BubbleOptions&,const ResourceBudget&);
    std::vector<BubbleNode> nodes_;
    std::vector<Count> pair_order_;
    std::vector<PairExtent> pairs_;
    BubbleOptions options_;
    BubblePlan plan_;
    std::string source_identity_;
    int depth_=0;
    Count retained_bytes_=0;
    Count construct(Count begin,Count end,Count parent,int depth);
};
BubblePlan plan_bubbles(const PairExtents&,const BubbleOptions&,const ResourceBudget&);
// Geometry-only candidates: a consuming algorithm must additionally certify
// its source-specific error before accept(). refine() descends a candidate.
class BubbleTraversal {
public:
    BubbleTraversal(std::shared_ptr<const BubbleTree>,double opening,const ResourceBudget&);
    std::optional<BubbleBlock> peek();
    void accept();
    void refine();
    const BubbleTraversalReceipt& receipt() const { return receipt_; }
private:
    std::shared_ptr<const BubbleTree> tree_;
    ResourceBudget budget_;
    double opening_;
    std::vector<std::array<Count,2>> stack_;
    std::optional<BubbleBlock> pending_;
    BubbleTraversalReceipt receipt_;
    Count stack_limit_=0;
    vibeqc::detail::Sha256 partition_hash_;
    void finish();
    void expand(Count a,Count b);
    void healthy() const;
};
} // namespace vibeqc::manta
