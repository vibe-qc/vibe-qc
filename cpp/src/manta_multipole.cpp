#include "vibeqc/manta_multipole.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using Real=long double;
constexpr int maximum_order=12;
int degree(const md::Index& a) { return a[0]+a[1]+a[2]; }
void validate_order(int order) {
    if(order<0 || order>maximum_order) throw std::invalid_argument("MANTA Cartesian multipole order must be zero through twelve");
}
Real choose(int n,int k) {
    Real value=1.; for(int j=1;j<=k;++j) value*=Real(n-j+1)/j; return value;
}
void add(Real x,Real& sum,Real& correction) {
    const auto adjusted=x-correction,next=sum+adjusted;
    correction=(next-sum)-adjusted; sum=next;
}
std::string identity(const char* operation,const md::Point& displacement,int order) {
    vibeqc::detail::Sha256 hash;
    const std::string prefix=std::string("manta.cartesian-multipole.v1;raw-moments;total-degree;")+operation;
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    for(double x:{displacement[0],displacement[1],displacement[2],double(order)}) {
        if(x==0.) x=0.; std::uint64_t word; std::memcpy(&word,&x,sizeof(word));
        std::array<std::uint8_t,8> bytes{}; for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(word>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    }
    return hash.finish_hex();
}
MultipoleMatrix prepare(const char* operation,const md::Point& displacement,int order,const ResourceBudget& budget) {
    const auto plan=plan_multipoles(order,budget);
    for(auto x:displacement) admission::require_finite(x);
    MultipoleMatrix result; result.indices=md::hermites(order);
    result.values=Eigen::MatrixXd::Zero(plan.moments,plan.moments);
    result.operation=operation; result.source_identity=identity(operation,displacement,order);
    result.modeled_peak_bytes=plan.modeled_peak_bytes; return result;
}
void check(const MultipoleMatrix& result) {
    if(!result.values.allFinite()) throw std::overflow_error("MANTA multipole coefficients exceed finite binary64");
}
} // namespace
MultipolePlan plan_multipoles(int order,const ResourceBudget& budget) {
    using namespace admission;
    validate_order(order); margin(budget);
    MultipolePlan plan; plan.order=order;
    plan.moments=Count(order+1)*(order+2)*(order+3)/6;
    plan.scalar_terms=square(plan.moments);
    plan.kernel_terms=add(mul(128,plan.scalar_terms),mul(256,mul(square(order+1),plan.moments)));
    plan.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(mul(24,plan.scalar_terms),mul(512,plan.moments)));
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA multipole matrix-element budget exceeded");
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA multipole work budget exceeded");
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA multipole memory budget exceeded");
    return plan;
}
MultipoleMatrix multipole_translation(const md::Point& shift,int order,const ResourceBudget& budget) {
    auto result=prepare("translation",shift,order,budget);
    for(std::size_t i=0;i<result.indices.size();++i) for(std::size_t j=0;j<result.indices.size();++j) {
        const auto& n=result.indices[i]; const auto& a=result.indices[j];
        if(a[0]>n[0] || a[1]>n[1] || a[2]>n[2]) continue;
        Real value=1;
        for(int axis=0;axis<3;++axis) value*=choose(n[axis],a[axis])*std::pow(Real(shift[axis]),n[axis]-a[axis]);
        result.values(i,j)=static_cast<double>(value); ++result.coefficients_evaluated;
    }
    check(result); return result;
}
MultipoleMatrix multipole_coulomb(const md::Point& displacement,int order,const ResourceBudget& budget) {
    auto result=prepare("coulomb",displacement,order,budget);
    const Real radius=std::hypot(Real(displacement[0]),Real(displacement[1]),Real(displacement[2]));
    if(radius==0 || !std::isfinite(radius)) throw std::invalid_argument("MANTA multipole Coulomb separation must be finite and nonzero");
    const int side=order+1;
    std::vector<int> lookup(side*side*side,-1);
    auto offset=[side](const md::Index& a) { return (a[0]*side+a[1])*side+a[2]; };
    const auto n=result.indices.size();
    for(std::size_t i=0;i<n;++i) lookup[offset(result.indices[i])]=static_cast<int>(i);
    std::vector<Real> power(n,0.),next(n),coefficients(n,0.),errors(n,0.),next_errors(n);
    power[0]=1.; Real binomial=1.;
    // Cartesian Taylor form of the Coulomb multipole expansion. With u=R/|R|
    // and t=h/|R|, expand (1+2*u.t+t.t)^(-1/2)/|R|. The total-degree cutoff
    // keeps complete homogeneous orders, including all trace cancellations.
    // This normalization differs from BUPO's solid harmonics (Neese et al.,
    // 2025, DOI 10.1021/acs.jpca.4c07415); raw moments and factorials are explicit.
    for(int k=0;k<=order;++k) {
        for(std::size_t i=0;i<n;++i) add(binomial*power[i],coefficients[i],errors[i]);
        if(k==order) break;
        std::fill(next.begin(),next.end(),0.); std::fill(next_errors.begin(),next_errors.end(),0.);
        for(std::size_t i=0;i<n;++i) if(power[i]!=0.) for(int axis=0;axis<3;++axis) {
            auto a=result.indices[i];
            if(degree(a)+1<=order) {
                ++a[axis]; const auto j=lookup[offset(a)];
                add(power[i]*(2*(Real(displacement[axis])/radius)),next[j],next_errors[j]);
            }
            a=result.indices[i];
            if(degree(a)+2<=order) {
                a[axis]+=2; const auto j=lookup[offset(a)]; add(power[i],next[j],next_errors[j]);
            }
        }
        power.swap(next); binomial*=(-.5L-k)/(k+1);
    }
    std::array<Real,maximum_order+2> inverse_powers{}; inverse_powers[0]=1.;
    for(int k=1;k<=order+1;++k) inverse_powers[k]=inverse_powers[k-1]/radius;
    for(std::size_t i=0;i<n;++i) for(std::size_t j=0;j<n;++j) {
        const auto& a=result.indices[i]; const auto& b=result.indices[j];
        md::Index total{a[0]+b[0],a[1]+b[1],a[2]+b[2]}; const int d=degree(total);
        if(d>order) continue;
        Real factor=(degree(b)%2)?-1.:1.;
        for(int axis=0;axis<3;++axis) factor*=choose(total[axis],a[axis]);
        result.values(i,j)=static_cast<double>(factor*coefficients[lookup[offset(total)]]*inverse_powers[d+1]);
        ++result.coefficients_evaluated;
    }
    check(result); return result;
}
double compact_multipole_remainder(double distance,double a,double b,double qa,double qb,int order) {
    validate_order(order);
    for(auto x:{distance,a,b,qa,qb}) admission::require_finite(x);
    const Real radius=Real(a)+b,gap=Real(distance)-radius;
    if(a<0 || b<0 || qa<0 || qb<0 || gap<=0)
        throw std::invalid_argument("MANTA compact remainder requires disjoint nonnegative supports and norms");
    if(qa==0 || qb==0 || radius==0) return 0.;
    // |P_l(cos theta)| <= 1 and the geometric-series remainder, integrated
    // against absolute charge norms. Gaussian penetration is a separate term.
    const Real logarithm=std::log(Real(qa))+std::log(Real(qb))
        +(order+1)*(std::log(radius)-std::log(Real(distance)))-std::log(gap);
    const Real value=std::exp(logarithm)*(1+256*std::numeric_limits<double>::epsilon());
    if(!std::isfinite(value) || value>std::numeric_limits<double>::max())
        throw std::overflow_error("MANTA compact remainder exceeds binary64");
    const auto result=std::nextafter(static_cast<double>(value),std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA compact remainder exceeds binary64");
    return result;
}
} // namespace vibeqc::manta
