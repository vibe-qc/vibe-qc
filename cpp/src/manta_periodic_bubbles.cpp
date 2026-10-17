#include "vibeqc/manta_periodic_bubbles.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using namespace admission;
using Real=long double;
constexpr Real minus_inf=-std::numeric_limits<Real>::infinity();
ResourceBudget child_budget(const ResourceBudget& budget) {
    auto out=budget; out.other_live_bytes=0;
    out.memory_bytes=out.shell_calls=out.primitive_quartets=out.scalar_terms=out.kernel_terms=std::numeric_limits<Count>::max();
    return out; // Only after the joint count-only admission below.
}
const PeriodicPairSource& required(const std::shared_ptr<const PeriodicPairSource>& p) {
    if(!p) throw std::invalid_argument("MANTA periodic bubbles require a native pair source");
    return *p;
}
std::string identity(const std::string& source,const std::string& input,const md::Matrix* values=nullptr) {
    vibeqc::detail::Sha256 hash;
    auto word=[&](Count x) { std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=x>>(56-8*k); hash.update(bytes.data(),bytes.size()); };
    for(const auto& text:{std::string("manta.periodic-bubble-action.v1"),source,input}) {
        word(text.size()); hash.update(reinterpret_cast<const std::uint8_t*>(text.data()),text.size());
    }
    if(values) {
        word(values->rows()); word(values->cols());
        for(Eigen::Index i=0;i<values->rows();++i) for(Eigen::Index j=0;j<values->cols();++j) {
            double x=(*values)(i,j); if(x==0) x=0; Count bits; std::memcpy(&bits,&x,8); word(bits);
        }
    }
    return hash.finish_hex();
}
md::Point displacement(const md::Point& a,const md::Point& b) {
    md::Point d; for(int k=0;k<3;++k) { d[k]=a[k]-b[k]; require_finite(d[k]); } return d;
}
double radius(const md::Point& d) {
    const double r=std::hypot(d[0],d[1],d[2]); require_finite(r);
    if(r==0) return 0;
    const double upper=std::nextafter(r*(1+64*std::numeric_limits<double>::epsilon()),std::numeric_limits<double>::infinity());
    require_finite(upper); return upper;
}
template<class T> void accumulate(T x,T& sum,T& error) {
    const T y=x-error,next=sum+y; error=(next-sum)-y; sum=next;
}
void translate(const Eigen::MatrixXd& w,const md::Matrix& input,Count from,md::Matrix& output,
    md::Matrix& correction,Count to,bool transpose,Count& terms) {
    for(Eigen::Index i=0;i<w.rows();++i) {
        Real sum=0,error=0;
        for(Eigen::Index j=0;j<w.cols();++j) {
            const double coefficient=transpose?w(j,i):w(i,j);
            if(coefficient==0) continue;
            accumulate(Real(coefficient)*input(from,j),sum,error); ++terms;
        }
        const double x=static_cast<double>(sum); require_finite(x);
        accumulate(x,output(to,i),correction(to,i)); require_finite(output(to,i));
    }
}
Real logadd(Real a,Real b) {
    if(a==minus_inf) return b;
    if(b==minus_inf) return a;
    const Real high=std::max(a,b); return high+std::log1p(std::exp(std::min(a,b)-high));
}
double upper(Real logarithm) {
    if(logarithm==minus_inf) return 0;
    if(!std::isfinite(logarithm)) throw std::overflow_error("MANTA periodic bubble bound is not finite");
    const Real x=std::exp(logarithm)*(1+256*std::numeric_limits<double>::epsilon());
    const double out=std::nextafter(static_cast<double>(x),std::numeric_limits<double>::infinity());
    if(!std::isfinite(out)) throw std::overflow_error("MANTA periodic bubble bound exceeds binary64");
    return out;
}
double positive_add(double a,double b) { return upper(logadd(a==0?minus_inf:std::log(Real(a)),b==0?minus_inf:std::log(Real(b)))); }
template<class Get> void shifted_bounds(md::Matrix& output,Count to,int order,double shift,double weight,Get value) {
    if(weight==0) return;
    for(int k=0;k<=order;++k) {
        Real total=minus_inf,choose=1;
        for(int t=0;t<=k;++t) {
            if(t) choose=choose*(k-t+1)/t;
            const double v=value(t);
            if(v==0 || (shift==0 && k!=t)) continue;
            const Real logshift=k==t?0:(k-t)*std::log(Real(shift));
            total=logadd(total,std::log(Real(weight))+std::log(choose)+logshift+std::log(Real(v)));
        }
        output(to,k)=positive_add(output(to,k),upper(total));
    }
}
} // namespace

PeriodicBubblePlan PeriodicBubbleTree::plan_source(const PeriodicPairSource& source,const BubbleOptions& options,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget); PeriodicBubblePlan p;
    // No extent export, tree copy, moment allocation or array payload read.
    p.geometry=BubbleTree::plan_periodic(source,options,budget);
    p.pairs=source.receipt().pairs; p.maximum_nodes=p.geometry.maximum_nodes; p.moments=source.plan().moments;
    const auto child=child_budget(budget); const auto mp=plan_multipoles(source.order(),child);
    p.maximum_translation_calls=mul(2,add(p.pairs,p.maximum_nodes-1));
    p.shell_calls=mul(2,p.pairs); p.primitive_pairs=mul(2,source.receipt().primitive_pairs);
    const Count nm=mul(p.maximum_nodes,p.moments),radial=mul(p.maximum_nodes,source.order()+1);
    cap(nm,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic bubble moments are not addressable");
    p.scalar_terms=add(add(source.plan().scalar_terms,p.geometry.scalar_terms),
        add(mul(8,add(nm,radial)),mul(p.maximum_translation_calls,mp.scalar_terms)));
    p.kernel_terms=add(add(source.plan().kernel_terms,p.geometry.kernel_terms),
        add(mul(p.maximum_translation_calls,add(mp.kernel_terms,mul(16,mp.scalar_terms))),mul(512,mul(add(p.pairs,p.maximum_nodes),square(source.order()+1)))));
    // The source's retained reservation and one complete forward/adjoint cycle,
    // node result/input/correction matrices, exports and one translation panel.
    p.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(source.plan().modeled_peak_bytes,add(p.geometry.modeled_peak_bytes,
        add(mp.modeled_peak_bytes,add(mul(64,nm),add(mul(64,radial),add(mul(64,p.maximum_nodes),8192)))))));
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic bubble shell budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic bubble primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic bubble scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic bubble work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic bubble memory budget exceeded");
    return p;
}
PeriodicBubbleTree::PeriodicBubbleTree(std::shared_ptr<const PeriodicPairSource> source,const BubbleOptions& options,const ResourceBudget& budget)
    :plan_(plan_source(required(source),options,budget)),source_(std::move(source)),workspace_budget_(child_budget(budget)) {
    tree_=std::shared_ptr<const BubbleTree>(new BubbleTree(source_->extents(),options,workspace_budget_,false));
}
std::unique_ptr<BubbleTraversal> PeriodicBubbleTree::traversal(double opening,const ResourceBudget& budget) const {
    auto limits=budget; limits.other_live_bytes=add(limits.other_live_bytes,source_->plan().modeled_peak_bytes);
    return std::make_unique<BubbleTraversal>(tree_,opening,limits);
}
PeriodicBubbleDensity PeriodicBubbleTree::contract(const std::vector<std::complex<double>>& d) const {
    const auto paired=source_->contract(d); const auto& ns=nodes(); const auto order=source_->order();
    PeriodicBubbleDensity out; out.pair_receipt=paired.receipt; out.source_identity=source_identity();
    out.density_identity=identity(source_identity(),paired.receipt.density_identity); out.modeled_peak_bytes=plan_.modeled_peak_bytes;
    out.moments=md::Matrix::Zero(ns.size(),plan_.moments);
    md::Matrix correction=out.moments;
    out.full_moment_bounds=out.tail_moment_bounds=md::Matrix::Zero(ns.size(),order+1);
    out.coefficient_l1.assign(ns.size(),0.);
    // Neese et al. (2025), BUPO, doi:10.1021/acs.jpca.4c07415:
    // full raw Cartesian moment translations, here over explicit periodic
    // pair-image leaves. The weighted Bloch contraction stays with its owner.
    for(Count id=ns.size();id-->0;) {
        const auto& node=ns[id];
        if(!node.child_count) for(Count index=node.begin;index<node.end;++index) {
            const auto p=pair_order()[index]; const auto& e=source_->entries()[p].extent;
            const auto shift=displacement(e.center,node.center);
            const auto w=multipole_translation(shift,order,workspace_budget_);
            translate(w.values,paired.moments,p,out.moments,correction,id,false,out.translation_terms); ++out.translation_calls;
            const auto norm=paired.pair_coefficient_l1[p];
            out.coefficient_l1[id]=positive_add(out.coefficient_l1[id],norm);
            shifted_bounds(out.full_moment_bounds,id,order,radius(shift),norm,[&](int k) { return e.full_moment_bounds[k]; });
            shifted_bounds(out.tail_moment_bounds,id,order,radius(shift),norm,[&](int k) { return e.tail_moment_bounds[k]; });
        }
        if(id) {
            const auto parent=node.parent; const auto shift=displacement(node.center,ns[parent].center);
            const auto w=multipole_translation(shift,order,workspace_budget_);
            translate(w.values,out.moments,id,out.moments,correction,parent,false,out.translation_terms); ++out.translation_calls;
            out.coefficient_l1[parent]=positive_add(out.coefficient_l1[parent],out.coefficient_l1[id]);
            // Parent balls enclose child balls. Outside a parent, every child
            // tail bound applies; the same binomial shift bounds tail moments.
            shifted_bounds(out.full_moment_bounds,parent,order,radius(shift),1.,[&](int k) { return out.full_moment_bounds(id,k); });
            shifted_bounds(out.tail_moment_bounds,parent,order,radius(shift),1.,[&](int k) { return out.tail_moment_bounds(id,k); });
        }
    }
    return out;
}
PeriodicBubbleAdjoint PeriodicBubbleTree::adjoint(const md::Matrix& input) const {
    if(input.rows()!=static_cast<Eigen::Index>(nodes().size()) || input.cols()!=static_cast<Eigen::Index>(plan_.moments) || !input.allFinite())
        throw std::invalid_argument("MANTA periodic bubble potentials must be finite and match (nodes,moments)");
    PeriodicBubbleAdjoint out; out.source_identity=source_identity();
    out.potential_identity=identity(source_identity(),"node-potentials",&input); out.modeled_peak_bytes=plan_.modeled_peak_bytes;
    md::Matrix local=input,correction=md::Matrix::Zero(input.rows(),input.cols());
    md::Matrix leaves=md::Matrix::Zero(plan_.pairs,plan_.moments),leaf_error=leaves;
    const auto& ns=nodes(); const auto order=source_->order();
    for(Count id=0;id<ns.size();++id) {
        const auto& node=ns[id];
        if(id) {
            const auto w=multipole_translation(displacement(node.center,ns[node.parent].center),order,workspace_budget_);
            translate(w.values,local,node.parent,local,correction,id,true,out.translation_terms); ++out.translation_calls;
        }
        if(!node.child_count) for(Count index=node.begin;index<node.end;++index) {
            const auto p=pair_order()[index];
            const auto w=multipole_translation(displacement(source_->entries()[p].extent.center,node.center),order,workspace_budget_);
            translate(w.values,local,id,leaves,leaf_error,p,true,out.translation_terms); ++out.translation_calls;
        }
    }
    out.pair=source_->adjoint(leaves); return out;
}
} // namespace vibeqc::manta
