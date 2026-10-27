#include "vibeqc/manta_lattice_errors.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace vibeqc::manta {
namespace {
using Real=long double;
constexpr Real pi=3.141592653589793238462643383279502884L;
constexpr Real minus_inf=-std::numeric_limits<Real>::infinity();
Real logadd(Real a,Real b) {
    if(a==minus_inf) return b;
    if(b==minus_inf) return a;
    const Real high=std::max(a,b);
    return high+std::log1p(std::exp(std::min(a,b)-high));
}
Real logproduct(double a,double b) {
    return a==0 || b==0?minus_inf:std::log(Real(a))+std::log(Real(b));
}
double upper(Real logarithm) {
    if(logarithm==minus_inf) return 0;
    if(!std::isfinite(logarithm)) throw std::overflow_error("MANTA lattice Gaussian nonfinite error envelope");
    const Real value=std::exp(logarithm)*(1+8192*std::numeric_limits<double>::epsilon());
    if(!std::isfinite(value) || value>std::numeric_limits<double>::max())
        throw std::overflow_error("MANTA lattice Gaussian error envelope exceeds finite binary64");
    const double result=std::nextafter(static_cast<double>(value),std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA lattice Gaussian error envelope exceeds finite binary64");
    return result;
}
struct Hash {
    vibeqc::detail::Sha256 hash;
    void word(std::uint64_t x) {
        std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(x>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    }
    void number(double x) {
        if(x==0) x=0;
        std::uint64_t bits; std::memcpy(&bits,&x,sizeof(bits)); word(bits);
    }
    void text(const std::string& s) {
        word(s.size()); hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
    }
};
void role(Hash& h,const PairExtent& e) {
    h.text("orbital-product"); h.word(e.shells[0]); h.word(e.shells[1]);
}
void role(Hash& h,const AuxiliaryExtent& e) { h.text("auxiliary"); h.word(e.shell); }
template<class Extent> std::string envelope_identity(const Extent& e,int order) {
    Hash h; h.text("manta.lattice-gaussian-envelope.v1;absolute-radial"); role(h,e); h.word(order);
    for(double x:e.center) h.number(x);
    h.number(e.radius); h.number(e.potential_bound);
    for(int i=0;i<=order;++i) { h.number(e.full_moment_bounds[i]); h.number(e.tail_moment_bounds[i]); }
    return h.hash.finish_hex();
}
template<class Extent> void validate(const Extent& e,int order) {
    if(e.always_near) throw std::invalid_argument("MANTA lattice Gaussian always-near envelopes require direct evaluation");
    if(e.full_moment_bounds.size()<Count(order+1) || e.tail_moment_bounds.size()<Count(order+1)
        || e.full_moment_bounds.size()>13 || e.tail_moment_bounds.size()>13)
        throw std::invalid_argument("MANTA lattice Gaussian requires native retained full and tail moments through the requested order");
    for(double x:e.center) admission::require_finite(x);
    auto nonnegative=[](double x) {
        admission::require_finite(x);
        if(x<0) throw std::invalid_argument("MANTA lattice Gaussian envelope values must be nonnegative");
    };
    nonnegative(e.radius); nonnegative(e.potential_bound);
    for(int i=0;i<=order;++i) { nonnegative(e.full_moment_bounds[i]); nonnegative(e.tail_moment_bounds[i]); }
}
// On an interior sphere, integral |x|^j |rho(x)| is bounded by
// min_{0<=k<=min(j,p)} radius^(j-k) full_absolute_moment[k]. Thus the
// (p+1)-st relative-displacement moment is bounded by their binomial
// convolution. This reuses native full moments through p, including p=12;
// it never assumes compact Gaussian support or invents a p+1 moment.
template<class Extent> Real interior_moment(const Extent& e,int j,int order) {
    Real best=std::numeric_limits<Real>::infinity();
    for(int k=0;k<=std::min(j,order);++k) {
        const double moment=e.full_moment_bounds[k];
        const Real power=j==k?0:e.radius==0?minus_inf:Real(j-k)*std::log(Real(e.radius));
        best=std::min(best,moment==0?minus_inf:std::log(Real(moment))+power);
    }
    return best;
}
template<class A,class B> double compact_moment_scale(const A& a,const B& b,int order) {
    const int n=order+1;const Real radius=Real(a.radius)+b.radius;
    if(radius==0 || a.full_moment_bounds[0]==0 || b.full_moment_bounds[0]==0) return 1.;
    Real moment=minus_inf;
    for(int j=0;j<=n;++j) {
        const Real x=interior_moment(a,j,order),y=interior_moment(b,n-j,order);
        moment=logadd(moment,std::lgamma(n+1.L)-std::lgamma(j+1.L)-std::lgamma(n-j+1.L)+x+y);
    }
    const Real denominator=n*std::log(radius)+logproduct(a.full_moment_bounds[0],b.full_moment_bounds[0]);
    return std::min(1.,upper(std::min(0.L,moment-denominator)));
}
template<class A,class B> LatticeGaussianInteraction evaluate(const Eigen::Matrix3d& cell,double eta,
    const std::vector<LatticeIndex>& rs,const std::vector<LatticeIndex>& gs,const std::vector<LatticeIndex>& ns,
    const A& a,const B& b,int order,const ResourceBudget& budget,bool isolated=false) {
    LatticeGaussianInteraction result;
    result.plan=isolated?plan_erfc_gaussian_interaction(order,budget):plan_lattice_gaussian_interaction(order,rs.size(),gs.size(),ns.size(),budget);
    validate(a,order); validate(b,order);
    md::Point displacement{};
    for(int axis=0;axis<3;++axis) {
        displacement[axis]=a.center[axis]-b.center[axis];
        admission::require_finite(displacement[axis]);
    }
    // Check support before the more expensive coefficient construction. Both
    // components use exactly the same finite domains and center displacement.
    if(isolated) {
        result.compact=erfc_remainder(displacement,eta,order,a.radius,b.radius,a.full_moment_bounds[0],b.full_moment_bounds[0],budget);
        result.matrix=multipole_erfc(displacement,eta,order,budget);
        result.error.qualification="isolated-erfc-gaussian-approximation";
    } else {
        result.compact=lattice_tail_remainder(cell,displacement,eta,rs,gs,ns,order,
            a.radius,b.radius,a.full_moment_bounds[0],b.full_moment_bounds[0],budget);
        result.matrix=multipole_lattice_tail(cell,displacement,eta,rs,gs,ns,order,budget);
    }
    if(result.compact.expansion_identity!=result.matrix.source_identity)
        throw std::logic_error("MANTA lattice Gaussian coefficient/remainder identity mismatch");
    auto& error=result.error;
    error.compact_remainder=result.compact.total;
    if(isolated) {
        error.compact_moment_scale=compact_moment_scale(a,b,order);
        error.compact_remainder=std::min(error.compact_remainder,
            upper(logproduct(result.compact.total,error.compact_moment_scale)));
    }
    error.target_envelope_identity=envelope_identity(a,order);
    error.source_envelope_identity=envelope_identity(b,order);

    // Ewald (1921), doi:10.1002/andp.19213690304, with the finite complementary
    // partition derived in scientific contract 5.2. Every near -erf/r term,
    // reciprocal cosine term and constant gauge has a global absolute bound.
    // Non-near erfc/r is positive and no larger than bare Coulomb. The global
    // producer potential bound therefore controls each translated tail.
    Real bounded=isolated?minus_inf:std::log(pi)-2*std::log(Real(eta))-std::log(Real(result.matrix.cell_volume));
    bounded=logadd(bounded,std::log(Real(ns.size()))+std::log(2.L*eta)-.5L*std::log(pi));
    const Eigen::Matrix3d dual=cell.inverse().transpose();
    auto sorted_gs=gs; std::sort(sorted_gs.begin(),sorted_gs.end());
    for(const auto& index:sorted_gs) {
        std::array<Real,3> g{};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) g[i]+=2*pi*Real(dual(i,j))*index[j];
        const Real norm=std::hypot(g[0],g[1],g[2]);
        if(!std::isfinite(norm) || norm<=0) throw std::overflow_error("MANTA lattice Gaussian invalid reciprocal scale");
        const Real lognorm=std::log(norm),logt=2*(lognorm-std::log(Real(eta))-std::log(2.L));
        const Real t=std::exp(std::min(logt,std::log(1.e6L)));
        bounded=logadd(bounded,std::log(4*pi)-std::log(Real(result.matrix.cell_volume))-2*lognorm-t);
        ++error.reciprocal_terms_evaluated;
    }
    error.bounded_kernel_sup=upper(bounded);
    const Real mass=logadd(logproduct(a.tail_moment_bounds[0],b.full_moment_bounds[0]),
        logproduct(a.full_moment_bounds[0],b.tail_moment_bounds[0]));
    Real penetration=mass==minus_inf?minus_inf:bounded+mass;
    if(rs.size()>ns.size()) {
        const Real potential=logadd(logproduct(a.tail_moment_bounds[0],b.potential_bound),
            logproduct(b.tail_moment_bounds[0],a.potential_bound));
        if(potential!=minus_inf) penetration=logadd(penetration,std::log(Real(rs.size()-ns.size()))+potential);
    }
    error.tail_penetration=upper(penetration);

    // The same full-moment substitution identity as the molecular Gaussian
    // BUPO ledger applies to this NON-harmonic matrix, including trace and
    // gauge entries. It does not import the molecular Coulomb remainder.
    Real substitution=minus_inf;
    const auto& indices=result.matrix.indices;
    for(std::size_t i=0;i<indices.size();++i) for(std::size_t j=0;j<indices.size();++j) {
        const double coefficient=std::abs(result.matrix.values(i,j));
        if(coefficient==0) continue;
        const int ai=indices[i][0]+indices[i][1]+indices[i][2],bj=indices[j][0]+indices[j][1]+indices[j][2];
        const Real moments=logadd(logproduct(a.tail_moment_bounds[ai],b.full_moment_bounds[bj]),
            logproduct(a.full_moment_bounds[ai],b.tail_moment_bounds[bj]));
        if(moments!=minus_inf) substitution=logadd(substitution,std::log(Real(coefficient))+moments);
        ++error.moment_products_evaluated;
    }
    error.full_moment_tail=upper(substitution);
    Real total=minus_inf;
    for(double x:{error.compact_remainder,error.tail_penetration,error.full_moment_tail})
        if(x>0) total=logadd(total,std::log(Real(x)));
    error.total=upper(total);
    Hash h; h.text(isolated?"manta.erfc-gaussian-error.v2;joint-radial-and-interior-moments;compact-plus-penetration-plus-full-moment-tail":
        "manta.lattice-gaussian-error.v1;finite;compact-plus-penetration-plus-full-moment-tail");
    h.text(result.matrix.source_identity); h.text(result.compact.source_identity);
    h.text(error.target_envelope_identity); h.text(error.source_envelope_identity);
    error.source_identity=h.hash.finish_hex();
    return result;
}
} // namespace

static LatticeGaussianPlan gaussian_plan(int order,Count nr,Count ng,Count nn,const ResourceBudget& budget,bool isolated=false) {
    using namespace admission;
    LatticeGaussianPlan p;
    p.coefficients=isolated?plan_erfc_multipole(order,budget):plan_lattice_tail(order,nr,ng,nn,budget);
    p.compact=isolated?plan_erfc_remainder(order,budget):plan_lattice_tail_remainder(order,nr,ng,nn,budget);
    p.scalar_terms=add(add(p.coefficients.scalar_terms,p.compact.scalar_terms),mul(2,p.coefficients.scalar_terms));
    Count levels=1; for(auto n=ng;n>1;n>>=1) ++levels;
    p.kernel_terms=add(add(p.coefficients.kernel_terms,p.compact.kernel_terms),
        add(mul(256,p.coefficients.scalar_terms),add(mul(512,ng),mul(128,mul(ng,levels)))));
    if(isolated) {
        // Binomial convolution and the interior-moment caps, including logs
        // and gamma coefficients. No additional producer data or matrix.
        const Count work=mul(16,square(Count(order)+2));
        p.scalar_terms=add(p.scalar_terms,work);p.kernel_terms=add(p.kernel_terms,mul(256,work));
    }
    p.envelope_bytes=4096; // two borrowed native envelopes, metadata and hashing
    p.workspace_bytes=add(mul(sizeof(LatticeIndex),ng),8192); // canonical reciprocal copy and scalar work
    const auto base=add(budget.other_live_bytes,budget.backend_margin_bytes);
    p.modeled_peak_bytes=add(add(p.coefficients.modeled_peak_bytes,p.compact.modeled_peak_bytes-base),
        add(p.envelope_bytes,p.workspace_bytes));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA lattice Gaussian scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA lattice Gaussian work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA lattice Gaussian memory budget exceeded");
    return p;
}
LatticeGaussianPlan plan_lattice_gaussian_interaction(int order,Count nr,Count ng,Count nn,const ResourceBudget& budget) {
    return gaussian_plan(order,nr,ng,nn,budget);
}
LatticeGaussianPlan plan_erfc_gaussian_interaction(int order,const ResourceBudget& budget) {
    return gaussian_plan(order,1,0,0,budget,true);
}
#define MANTA_ERFC_INTERACTION(A,B) \
LatticeGaussianInteraction erfc_gaussian_interaction(double omega,const A& target,const B& source, \
    int order,const ResourceBudget& budget) { \
    return evaluate(Eigen::Matrix3d::Identity(),omega,{LatticeIndex{}},{},{},target,source,order,budget,true); }
MANTA_ERFC_INTERACTION(PairExtent,PairExtent)
MANTA_ERFC_INTERACTION(PairExtent,AuxiliaryExtent)
MANTA_ERFC_INTERACTION(AuxiliaryExtent,PairExtent)
MANTA_ERFC_INTERACTION(AuxiliaryExtent,AuxiliaryExtent)
#undef MANTA_ERFC_INTERACTION
#define MANTA_LATTICE_INTERACTION(A,B) \
LatticeGaussianInteraction lattice_gaussian_interaction(const Eigen::Matrix3d& cell,double eta, \
    const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal,const std::vector<LatticeIndex>& near, \
    const A& target,const B& source,int order,const ResourceBudget& budget) { \
    return evaluate(cell,eta,real,reciprocal,near,target,source,order,budget); }
MANTA_LATTICE_INTERACTION(PairExtent,PairExtent)
MANTA_LATTICE_INTERACTION(PairExtent,AuxiliaryExtent)
MANTA_LATTICE_INTERACTION(AuxiliaryExtent,PairExtent)
MANTA_LATTICE_INTERACTION(AuxiliaryExtent,AuxiliaryExtent)
#undef MANTA_LATTICE_INTERACTION
} // namespace vibeqc::manta
