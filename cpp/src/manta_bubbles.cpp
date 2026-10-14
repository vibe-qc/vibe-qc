#include "vibeqc/manta_bubbles.hpp"
#include "vibeqc/manta_periodic_pairs.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>
#include <numeric>

namespace vibeqc::manta {
namespace {
using namespace admission;
void validate_options(const BubbleOptions& o) {
    if(!o.leaf_capacity || o.branching<2 || o.branching>8 || o.maximum_depth<0 || o.maximum_depth>64)
        throw std::invalid_argument("MANTA bubble leaf capacity, branching or depth is invalid");
}
void hash_word(vibeqc::detail::Sha256& hash,std::uint64_t word) {
    std::array<std::uint8_t,8> bytes{};
    for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(word>>(56-8*k));
    hash.update(bytes.data(),bytes.size());
}
void hash_real(vibeqc::detail::Sha256& hash,double value) {
    if(value==0.) value=0.; std::uint64_t word; std::memcpy(&word,&value,sizeof(word)); hash_word(hash,word);
}
void hash_text(vibeqc::detail::Sha256& hash,const std::string& text) {
    hash_word(hash,text.size()); hash.update(reinterpret_cast<const std::uint8_t*>(text.data()),text.size());
}
double distance(const std::array<double,3>& a,const std::array<double,3>& b) {
    const auto value=std::hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
    if(!std::isfinite(value)) throw std::overflow_error("MANTA bubble geometry exceeds binary64");
    return value;
}
double enclosing_radius(double distance,double radius) {
    const auto sum=distance+radius;
    if(!std::isfinite(sum)) throw std::overflow_error("MANTA bubble radius exceeds binary64");
    if(sum==0.) return 0.;
    const auto value=std::nextafter(sum*(1+16*std::numeric_limits<double>::epsilon()),std::numeric_limits<double>::infinity());
    if(!std::isfinite(value)) throw std::overflow_error("MANTA bubble radius exceeds binary64");
    return value;
}
template<class Get> Count tree_bytes(Count count,Get pair,Count maximum_nodes) {
    Count bytes=mul(maximum_nodes,sizeof(BubbleNode));
    bytes=add(bytes,mul(count,sizeof(PairExtent)+sizeof(Count)+64));
    for(Count i=0;i<count;++i) {
        const auto& p=pair(i);
        bytes=add(bytes,add(mul(add(p.tail_moment_bounds.size(),p.full_moment_bounds.size()),8),p.reason.size()));
    }
    return add(bytes,1024);
}
template<class Get> BubblePlan plan_inventory(Count count,Get pair_at,const BubbleOptions& options,const ResourceBudget& budget,bool molecular) {
    using namespace admission;
    validate_options(options); margin(budget);
    BubblePlan plan; plan.pairs=count;
    if(!plan.pairs) throw std::invalid_argument("MANTA bubble tree requires nonempty extents");
    plan.maximum_nodes=mul(2,plan.pairs)-1;
    plan.scalar_terms=add(mul(plan.maximum_nodes,24),mul(plan.pairs,8));
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA bubble scalar budget exceeded");
    // At most maximum_depth deterministic sorts. N^2 is not a storage/work model:
    // sorting comparisons are bounded by a conservative N*ceil(log2(N)).
    Count logarithm=0; for(Count n=plan.pairs;n;n/=2) ++logarithm;
    plan.kernel_terms=mul(256,mul(plan.pairs,mul(options.maximum_depth+1,logarithm+1)));
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA bubble construction work budget exceeded");
    plan.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(mul(3,tree_bytes(count,pair_at,plan.maximum_nodes)),mul(plan.pairs,64)));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA bubble memory budget exceeded");
    // The pair IDs must be the complete canonical triangular inventory. This
    // prevents an omitted/duplicated shell pair from disappearing in the tree.
    Count a=0,b=0;
    for(Count i=0;i<count;++i) {
        const auto& pair=pair_at(i);
        if(molecular && (pair.shells[0]!=a || pair.shells[1]!=b))
            throw std::invalid_argument("MANTA bubble extents require a complete canonical shell-pair inventory");
        for(auto x:pair.center) require_finite(x);
        require_finite(pair.radius);
        if(pair.radius<0) throw std::invalid_argument("MANTA bubble radius must be nonnegative");
        if(++b>a) { ++a; b=0; }
    }
    if(molecular && b) throw std::invalid_argument("MANTA bubble shell-pair inventory is incomplete");
    return plan;
}
} // namespace
BubblePlan plan_bubbles(const PairExtents& input,const BubbleOptions& options,const ResourceBudget& budget) {
    return plan_inventory(input.pairs.size(),[&](Count i)->const PairExtent& { return input.pairs[i]; },options,budget,true);
}
BubblePlan BubbleTree::plan_periodic(const PeriodicPairSource& source,const BubbleOptions& options,const ResourceBudget& budget) {
    return plan_inventory(source.entries().size(),[&](Count i)->const PairExtent& { return source.entries()[i].extent; },options,budget,false);
}
BubbleTree::BubbleTree(const PairExtents& input,const BubbleOptions& options,const ResourceBudget& budget)
    :BubbleTree(input,options,budget,true) {}
BubbleTree::BubbleTree(const PairExtents& input,const BubbleOptions& options,const ResourceBudget& budget,bool molecular)
    :options_(options),plan_(plan_inventory(input.pairs.size(),[&](Count i)->const PairExtent& { return input.pairs[i]; },options,budget,molecular)) {
    retained_bytes_=tree_bytes(input.pairs.size(),[&](Count i)->const PairExtent& { return input.pairs[i]; },plan_.maximum_nodes);
    pairs_=input.pairs; pair_order_.resize(plan_.pairs); std::iota(pair_order_.begin(),pair_order_.end(),0);
    nodes_.reserve(plan_.maximum_nodes); construct(0,plan_.pairs,0,0);
    vibeqc::detail::Sha256 hash;
    hash_text(hash,molecular?"manta.bubbles.v1;stable-median;enclosing-children;":"manta.periodic-bubbles.v1;native-pair-image-inventory;stable-median;enclosing-children;");
    hash_text(hash,input.receipt.source_identity);
    hash_word(hash,options.leaf_capacity); hash_word(hash,options.branching); hash_word(hash,options.maximum_depth);
    for(const auto& p:pairs_) {
        hash_word(hash,p.shells[0]); hash_word(hash,p.shells[1]);
        for(auto x:p.center) hash_real(hash,x);
        hash_real(hash,p.radius); hash_word(hash,p.always_near);
    }
    source_identity_=hash.finish_hex();
}
Count BubbleTree::construct(Count begin,Count end,Count parent,int depth) {
    // BUPO groups shell-pair distributions into enclosing bubbles (Neese
    // et al., 2025, DOI 10.1021/acs.jpca.4c07415). This deterministic median
    // hierarchy and explicit always-near propagation are MANTA choices.
    const Count id=nodes_.size(); nodes_.emplace_back();
    nodes_[id].begin=begin; nodes_[id].end=end; nodes_[id].parent=parent; nodes_[id].depth=depth;
    depth_=std::max(depth_,depth);
    std::array<double,3> low=pairs_[pair_order_[begin]].center,high=low;
    for(Count k=begin;k<end;++k) {
        const auto& pair=pairs_[pair_order_[k]];
        nodes_[id].always_near|=pair.always_near;
        for(int axis=0;axis<3;++axis) { low[axis]=std::min(low[axis],pair.center[axis]); high[axis]=std::max(high[axis],pair.center[axis]); }
    }
    int axis=0;
    for(int k=0;k<3;++k) {
        nodes_[id].center[k]=low[k]/2+high[k]/2;
        if(high[k]-low[k]>high[axis]-low[axis]) axis=k;
    }
    if(end-begin>options_.leaf_capacity && depth<options_.maximum_depth) {
        std::sort(pair_order_.begin()+begin,pair_order_.begin()+end,[&](Count a,Count b) {
            return pairs_[a].center[axis]<pairs_[b].center[axis] ||
                (pairs_[a].center[axis]==pairs_[b].center[axis] && a<b);
        });
        const int count=static_cast<int>(std::min<Count>(options_.branching,end-begin));
        nodes_[id].child_count=count;
        const Count quotient=(end-begin)/count,remainder=(end-begin)%count;
        Count first=begin;
        for(int k=0;k<count;++k) {
            const Count last=first+quotient+(Count(k)<remainder);
            const auto child=construct(first,last,id,depth+1); nodes_[id].children[k]=child;
            nodes_[id].radius=std::max(nodes_[id].radius,enclosing_radius(distance(nodes_[id].center,nodes_[child].center),nodes_[child].radius));
            first=last;
        }
    } else {
        for(Count k=begin;k<end;++k) {
            const auto& pair=pairs_[pair_order_[k]];
            nodes_[id].radius=std::max(nodes_[id].radius,enclosing_radius(distance(nodes_[id].center,pair.center),pair.radius));
        }
    }
    return id;
}
BubbleTraversal::BubbleTraversal(std::shared_ptr<const BubbleTree> tree,double opening,const ResourceBudget& budget)
    :tree_(std::move(tree)),budget_(budget),opening_(opening) {
    using namespace admission;
    margin(budget); require_finite(opening);
    if(!tree_ || opening<0 || opening>=1.) throw std::invalid_argument("MANTA bubble opening must be in [0,1)");
    // Every expansion inserts at most 36 children and increases the combined
    // target/source depth. Thus depth-first live siblings are linearly bounded.
    stack_limit_=add(1,mul(70,tree_->depth()));
    receipt_.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(tree_->retained_bytes(),add(mul(stack_limit_,sizeof(std::array<Count,2>)),1024)));
    cap(receipt_.modeled_peak_bytes,budget.memory_bytes,"MANTA bubble traversal memory budget exceeded");
    stack_.reserve(stack_limit_); stack_.push_back({0,0}); receipt_.maximum_stack=1;
    vibeqc::detail::Sha256 hash; hash_text(hash,tree_->source_identity()); hash_real(hash,opening);
    hash_text(hash,"unordered-paired-partition.v1"); receipt_.source_identity=hash.finish_hex();
    hash_text(partition_hash_,receipt_.source_identity);
}
void BubbleTraversal::finish() {
    if(!receipt_.complete) receipt_.partition_identity=partition_hash_.finish_hex();
    receipt_.complete=true;
}
void BubbleTraversal::healthy() const {
    if(receipt_.failed) throw std::runtime_error("MANTA bubble traversal cannot resume after a budget or geometry failure");
}
void BubbleTraversal::expand(Count a,Count b) {
    const auto& na=tree_->nodes()[a]; const auto& nb=tree_->nodes()[b];
    auto push=[&](Count x,Count y) {
        if(stack_.size()>=stack_limit_) throw std::logic_error("MANTA bubble traversal stack bound exceeded");
        if(x>y) std::swap(x,y); stack_.push_back({x,y});
    };
    if(a==b) {
        if(!na.child_count) throw std::logic_error("MANTA cannot refine a terminal self block");
        for(int i=na.child_count-1;i>=0;--i) for(int j=i;j>=0;--j) push(na.children[i],na.children[j]);
    } else if(na.child_count && (!nb.child_count || na.radius>=nb.radius)) {
        for(int i=na.child_count-1;i>=0;--i) push(na.children[i],b);
    } else if(nb.child_count) {
        for(int i=nb.child_count-1;i>=0;--i) push(a,nb.children[i]);
    } else throw std::logic_error("MANTA cannot refine a terminal block");
    receipt_.maximum_stack=std::max<Count>(receipt_.maximum_stack,stack_.size());
}
std::optional<BubbleBlock> BubbleTraversal::peek() {
    healthy(); if(pending_) return pending_;
    try {
        while(!stack_.empty()) {
            admission::cap(admission::add(receipt_.visited_node_pairs,1),budget_.shell_calls,"MANTA bubble node-pair visit budget exceeded");
            admission::cap(admission::mul(admission::add(receipt_.visited_node_pairs,1),128),budget_.kernel_terms,"MANTA bubble traversal work budget exceeded");
            const auto [a,b]=stack_.back(); stack_.pop_back(); ++receipt_.visited_node_pairs;
            const auto& na=tree_->nodes()[a]; const auto& nb=tree_->nodes()[b];
            BubbleBlock block; block.target=a; block.source=b;
            bool separated=false;
            if(a!=b && opening_>0 && !na.always_near && !nb.always_near) {
                const double d=distance(na.center,nb.center);
                const double radii=na.radius+nb.radius;
                if(d>0 && std::isfinite(radii)) {
                    block.separation_ratio=std::nextafter(radii/d,std::numeric_limits<double>::infinity());
                    separated=block.separation_ratio<opening_;
                }
            }
            if(separated || (!na.child_count && !nb.child_count)) {
                const auto n=na.end-na.begin,m=nb.end-nb.begin;
                block.pair_interactions=a==b?admission::mul(n,admission::add(n,1))/2:admission::mul(n,m);
                block.far_candidate=separated; pending_=block; return pending_;
            }
            expand(a,b);
        }
        finish(); return std::nullopt;
    } catch(...) { receipt_.failed=true; throw; }
}
void BubbleTraversal::accept() {
    healthy(); if(!pending_) throw std::logic_error("MANTA bubble accept requires a pending block");
    try {
        const auto count=admission::add(receipt_.pair_interactions,pending_->pair_interactions);
        admission::cap(count,budget_.scalar_terms,"MANTA bubble represented-pair budget exceeded");
        receipt_.pair_interactions=count;
        if(pending_->far_candidate) ++receipt_.far_blocks; else ++receipt_.near_blocks;
        hash_word(partition_hash_,pending_->target); hash_word(partition_hash_,pending_->source);
        hash_word(partition_hash_,pending_->far_candidate);
        pending_.reset();
        if(stack_.empty()) finish();
    } catch(...) { receipt_.failed=true; throw; }
}
void BubbleTraversal::refine() {
    healthy(); if(!pending_ || !pending_->far_candidate) throw std::logic_error("MANTA bubble refine requires a far candidate");
    try {
        const auto& na=tree_->nodes()[pending_->target]; const auto& nb=tree_->nodes()[pending_->source];
        if(!na.child_count && !nb.child_count) {
            // A refused terminal far candidate becomes an exact near block.
            pending_->far_candidate=false; ++receipt_.refined_candidates; return;
        }
        expand(pending_->target,pending_->source); pending_.reset(); ++receipt_.refined_candidates;
    } catch(...) { receipt_.failed=true; throw; }
}
} // namespace vibeqc::manta
