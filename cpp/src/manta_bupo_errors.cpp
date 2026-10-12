#include "vibeqc/manta_bupo_errors.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using Real=long double;
const Real minus_inf=-std::numeric_limits<Real>::infinity();
Real logadd(Real a,Real b) {
    if(a==minus_inf) return b;
    if(b==minus_inf) return a;
    if(a<b) std::swap(a,b);
    return a+std::log1p(std::exp(b-a));
}
Real choose(int n,int k) {
    Real v=1.; for(int j=1;j<=k;++j) v*=Real(n-j+1)/j; return v;
}
Real logpower(Real x,int n) { return n==0?0.:x==0?minus_inf:n*std::log(x); }
double upper(Real logarithm) {
    if(logarithm==minus_inf) return 0.;
    if(!std::isfinite(logarithm)) throw std::overflow_error("MANTA BUPO nonfinite error envelope");
    const Real value=std::exp(logarithm)*(1+512*std::numeric_limits<double>::epsilon());
    if(!std::isfinite(value) || value>std::numeric_limits<double>::max())
        throw std::overflow_error("MANTA BUPO error envelope exceeds binary64");
    const auto result=std::nextafter(static_cast<double>(value),std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA BUPO error envelope exceeds binary64");
    return result;
}
double distance(const md::Point& a,const md::Point& b) {
    const double value=std::hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
    if(!std::isfinite(value)) throw std::overflow_error("MANTA BUPO center separation exceeds binary64");
    return value;
}
void include(std::vector<double>& target,const std::vector<double>& source,double shift) {
    for(std::size_t n=0;n<target.size();++n) {
        Real sum=minus_inf;
        for(std::size_t k=0;k<=n;++k) if(source[k]>0.)
            sum=logadd(sum,std::log(choose(n,k))+logpower(shift,n-k)+std::log(Real(source[k])));
        target[n]=std::max(target[n],upper(sum));
    }
}
std::string hash(const std::string& payload) {
    vibeqc::detail::Sha256 h; h.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size()); return h.finish_hex();
}
BubbleInteraction evaluate_interaction(const md::Point& a_center,double a_radius,const BubbleEnvelope& ea,
    const md::Point& b_center,double b_radius,const BubbleEnvelope& eb,int order,const ResourceBudget& budget) {
    const double d=distance(a_center,b_center);
    if(d<=a_radius+b_radius) throw std::invalid_argument("MANTA BUPO far bubbles require disjoint support spheres");
    // The approximation ledger is separate from floating-point error. Refuse
    // an extreme scale where even the common highest-order inverse radial
    // power cannot be represented; a consumer can refine or evaluate directly.
    const Real log_power=-(order+1)*std::log(Real(d));
    if(log_power<std::log(Real(std::numeric_limits<double>::min())) || log_power>std::log(Real(std::numeric_limits<double>::max())))
        throw std::overflow_error("MANTA BUPO multipole radial scale requires direct evaluation");
    md::Point displacement{}; for(int axis=0;axis<3;++axis) displacement[axis]=a_center[axis]-b_center[axis];
    BubbleInteraction result; result.matrix=multipole_coulomb(displacement,order,budget);
    auto& error=result.error;
    error.compact_remainder=compact_multipole_remainder(d,a_radius,b_radius,ea.full_moments[0],eb.full_moments[0],order);
    Real penetration=minus_inf,tail=minus_inf;
    auto logproduct=[](double x,double y) { return x==0. || y==0.?minus_inf:std::log(Real(x))+std::log(Real(y)); };
    penetration=logadd(logproduct(ea.tail_moments[0],eb.potential),logproduct(eb.tail_moments[0],ea.potential));
    error.tail_penetration=upper(penetration);
    // BUPO near/far separation (Neese et al., 2025, DOI 10.1021/acs.jpca.4c07415)
    // with an explicit MANTA Gaussian-tail ledger. The compact remainder alone
    // is insufficient: our contraction uses FULL Gaussian Cartesian moments.
    // |M_A^T T M_B - m_A^T T m_B| <= |tail_A|^T |T| |M_B|_abs
    //                                     + |M_A|_abs^T |T| |tail_B|.
    const auto& indices=result.matrix.indices;
    for(std::size_t i=0;i<indices.size();++i) for(std::size_t j=0;j<indices.size();++j) {
        const double coefficient=std::abs(result.matrix.values(i,j)); if(coefficient==0.) continue;
        const int ai=indices[i][0]+indices[i][1]+indices[i][2],bj=indices[j][0]+indices[j][1]+indices[j][2];
        const Real weight=logadd(logproduct(ea.tail_moments[ai],eb.full_moments[bj]),logproduct(ea.full_moments[ai],eb.tail_moments[bj]));
        tail=logadd(tail,std::log(Real(coefficient))+weight);
    }
    error.full_moment_tail=upper(tail);
    Real total=minus_inf;
    for(double x:{error.compact_remainder,error.tail_penetration,error.full_moment_tail}) if(x>0.) total=logadd(total,std::log(Real(x)));
    error.total=upper(total);
    return result;
}
std::string auxiliary_envelope_identity(const AuxiliaryExtent& extent,int order) {
    vibeqc::detail::Sha256 h;
    const std::string prefix="manta.auxiliary-error-envelope.v1;";
    h.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    auto word=[&](std::uint64_t bits) {
        std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(bits>>(56-8*k));
        h.update(bytes.data(),bytes.size());
    };
    auto number=[&](double value) {
        if(value==0.) value=0.;
        std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits)); word(bits);
    };
    word(extent.shell); word(order);
    for(double x:extent.center) number(x);
    number(extent.radius); number(extent.potential_bound);
    for(int i=0;i<=order;++i) { number(extent.full_moment_bounds[i]); number(extent.tail_moment_bounds[i]); }
    return h.finish_hex();
}
} // namespace
BubbleErrorPlan plan_bubble_errors(const BubbleTree& tree,int order,const ResourceBudget& budget) {
    using namespace admission;
    const auto multipole=plan_multipoles(order,budget);
    BubbleErrorPlan plan; plan.order=order; plan.nodes=tree.nodes().size(); plan.moments=multipole.moments;
    const Count radial=mul(plan.nodes,order+1),edges=add(plan.nodes,tree.pairs().size());
    plan.scalar_terms=add(multipole.scalar_terms,mul(2,radial));
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA BUPO envelope scalar budget exceeded");
    plan.kernel_terms=add(multipole.kernel_terms,add(mul(256,mul(edges,square(order+1))),mul(128,multipole.scalar_terms)));
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA BUPO envelope work budget exceeded");
    plan.modeled_peak_bytes=add(multipole.modeled_peak_bytes,add(tree.retained_bytes(),add(mul(3,add(mul(plan.nodes,sizeof(BubbleEnvelope)),mul(16,radial))),4096)));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA BUPO envelope memory budget exceeded");
    for(const auto& pair:tree.pairs()) {
        if(pair.full_moment_bounds.size()<=Count(order) || pair.tail_moment_bounds.size()<=Count(order))
            throw std::invalid_argument("MANTA BUPO error model requires retained full and tail moments through the requested order");
        for(int k=0;k<=order;++k) for(auto v:{pair.full_moment_bounds[k],pair.tail_moment_bounds[k]}) {
            require_finite(v); if(v<0.) throw std::invalid_argument("MANTA BUPO envelopes must be nonnegative");
        }
        require_finite(pair.potential_bound);
        if(pair.potential_bound<0.) throw std::invalid_argument("MANTA BUPO potential envelope must be nonnegative");
    }
    return plan;
}
BubbleErrorModel::BubbleErrorModel(std::shared_ptr<const BubbleTree> tree,int order,const ResourceBudget& budget)
    :tree_(std::move(tree)),budget_(budget) {
    if(!tree_) throw std::invalid_argument("MANTA BUPO error model requires a tree");
    plan_=plan_bubble_errors(*tree_,order,budget_);
    envelopes_.resize(plan_.nodes);
    const auto& nodes=tree_->nodes(); const auto& pairs=tree_->pairs(); const auto& permutation=tree_->pair_order();
    // Triangle inequality and the binomial theorem translate absolute radial
    // moments. Parent spheres enclose children, so outside(parent) is a subset
    // of outside(child). Maxima preserve a bound for every descendant AO pair.
    for(Count reverse=plan_.nodes;reverse>0;--reverse) {
        const auto id=reverse-1; auto& envelope=envelopes_[id]; const auto& node=nodes[id];
        envelope.full_moments.assign(order+1,0.); envelope.tail_moments.assign(order+1,0.);
        if(node.child_count) {
            for(int i=0;i<node.child_count;++i) {
                const auto child=node.children[i]; const auto& source=envelopes_[child];
                const double d=distance(nodes[child].center,node.center);
                include(envelope.full_moments,source.full_moments,d);
                include(envelope.tail_moments,source.tail_moments,d);
                envelope.potential=std::max(envelope.potential,source.potential);
            }
        } else {
            for(Count k=node.begin;k<node.end;++k) {
                const auto& pair=pairs[permutation[k]]; const double d=distance(pair.center,node.center);
                include(envelope.full_moments,pair.full_moment_bounds,d);
                include(envelope.tail_moments,pair.tail_moment_bounds,d);
                envelope.potential=std::max(envelope.potential,pair.potential_bound);
            }
        }
    }
    source_=hash("manta.bupo-error.v1;absolute-radial-maxima;compact-plus-penetration-plus-full-moment-tail;"+tree_->source_identity()+";"+std::to_string(order));
}
BubbleInteraction BubbleErrorModel::interaction(Count target,Count source) const {
    if(target>=plan_.nodes || source>=plan_.nodes) throw std::out_of_range("MANTA BUPO bubble index");
    const auto& a=tree_->nodes()[target]; const auto& b=tree_->nodes()[source];
    if(a.always_near || b.always_near) throw std::invalid_argument("MANTA BUPO always-near bubbles require direct evaluation");
    if(std::max(a.begin,b.begin)<std::min(a.end,b.end))
        throw std::invalid_argument("MANTA BUPO far bubbles must have disjoint pair inventories");
    auto result=evaluate_interaction(a.center,a.radius,envelopes_[target],b.center,b.radius,envelopes_[source],plan_.order,budget_);
    auto& error=result.error;
    error.source_identity=hash(source_+";"+std::to_string(target)+";"+std::to_string(source)+";"+result.matrix.source_identity);
    return result;
}
BubbleInteraction BubbleErrorModel::auxiliary_interaction(Count target,const AuxiliaryExtent& source) const {
    using namespace admission;
    if(target>=plan_.nodes) throw std::out_of_range("MANTA BUPO bubble index");
    const auto& node=tree_->nodes()[target];
    if(node.always_near || source.always_near)
        throw std::invalid_argument("MANTA BUPO always-near auxiliary interactions require direct evaluation");
    for(double x:source.center) require_finite(x);
    require_finite(source.radius); require_finite(source.potential_bound);
    if(source.radius<0. || source.potential_bound<0.)
        throw std::invalid_argument("MANTA BUPO auxiliary radius and potential envelope must be nonnegative");
    if(source.full_moment_bounds.size()<=Count(plan_.order) || source.tail_moment_bounds.size()<=Count(plan_.order))
        throw std::invalid_argument("MANTA BUPO auxiliary interaction requires retained full and tail moments through the requested order");
    BubbleEnvelope envelope; envelope.potential=source.potential_bound;
    for(int i=0;i<=plan_.order;++i) {
        const double full=source.full_moment_bounds[i],tail=source.tail_moment_bounds[i];
        require_finite(full); require_finite(tail);
        if(full<0. || tail<0.) throw std::invalid_argument("MANTA BUPO auxiliary envelopes must be nonnegative");
        envelope.full_moments.push_back(full); envelope.tail_moments.push_back(tail);
    }
    auto result=evaluate_interaction(node.center,node.radius,envelopes_[target],source.center,source.radius,envelope,plan_.order,budget_);
    result.error.source_identity=hash(source_+";auxiliary;"+std::to_string(target)+";"
        +auxiliary_envelope_identity(source,plan_.order)+";"+result.matrix.source_identity);
    return result;
}
} // namespace vibeqc::manta
