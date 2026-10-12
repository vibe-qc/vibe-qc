#include "vibeqc/manta_extents.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_md.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using Real=long double;
constexpr int maximum_degree=12,search_steps=64;
const Real pi=std::acos(Real(-1)),minus_inf=-std::numeric_limits<Real>::infinity();
Real logadd(Real a,Real b) {
    if(a==minus_inf) return b;
    if(b==minus_inf) return a;
    if(a<b) std::swap(a,b);
    return a+std::log1p(std::exp(b-a));
}
Real logpower(Real x,int power) {
    return power==0?0.:x==0?minus_inf:power*std::log(x);
}
Real choose(int n,int k) {
    Real value=1.; for(int j=1;j<=k;++j) value*=Real(n-j+1)/j; return value;
}
// Positive upper-incomplete-gamma recurrence. For half integers start with
// Gamma(1/2,x) <= min(sqrt(pi), exp(-x)/sqrt(x)); this follows directly from
// the Gaussian Mills inequality. Integer orders start with Gamma(1,x)=exp(-x).
// No subtraction of nearly equal complete/lower gamma values is needed.
Real log_radial_tail(int power,Real exponent,Real radius) {
    const Real shape=Real(power+1)/2,x=exponent*radius*radius;
    if(!std::isfinite(x)) throw std::overflow_error("MANTA extent radial exponent overflow");
    Real a=(power%2)?1.:.5;
    Real value=(a==1.)?-x:(x==0?.5*std::log(pi):std::min(.5*std::log(pi),-x-.5*std::log(x)));
    while(a<shape) {
        value=logadd(std::log(a)+value,x==0?minus_inf:a*std::log(x)-x);
        a+=1.;
    }
    return -std::log(Real(2))-shape*std::log(exponent)+value;
}
// Report analytic envelopes with outward FP64 headroom. This is not interval
// arithmetic: mathematical tail error and floating-point evaluation remain
// separate error sources. Positive underflows are rounded upward, never zero.
double upper(Real logarithm) {
    if(logarithm==minus_inf) return 0.;
    if(!std::isfinite(logarithm)) throw std::overflow_error("MANTA extent nonfinite bound");
    const Real padded=std::exp(logarithm)*(1+256*std::numeric_limits<double>::epsilon());
    if(padded>std::numeric_limits<double>::max()) throw std::overflow_error("MANTA extent bound exceeds binary64");
    const double value=static_cast<double>(padded);
    const double result=std::nextafter(value,std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA extent bound exceeds binary64");
    return result;
}
void validate_options(const ExtentOptions& options) {
    for(double x:{options.pointwise_tolerance,options.tail_mass_tolerance,options.minimum_radius,options.maximum_radius})
        admission::require_finite(x);
    if(options.pointwise_tolerance<=0 || options.tail_mass_tolerance<=0 || options.minimum_radius<0
       || options.maximum_radius<options.minimum_radius || options.maximum_radius<=0
       || options.moment_order<0 || options.moment_order>12)
        throw std::invalid_argument("MANTA extent tolerances/radii/order are invalid");
}
struct Primitive {
    Real exponent=0.,weight=0.,shift=0.;
    int degree=0;
    std::array<Real,maximum_degree+1> polynomial;
};
Real representation_norm(const libint2::Shell& shell) {
    const auto matrix=md::representation(shell.contr[0].l,shell.contr[0].pure);
    Real value=0;
    for(Eigen::Index row=0;row<matrix.rows();++row) {
        Real sum=0; for(Eigen::Index col=0;col<matrix.cols();++col) sum+=std::abs(matrix(row,col));
        value=std::max(value,sum);
    }
    return value;
}
std::vector<Primitive> primitives(const libint2::Shell& a,const libint2::Shell& b,const std::array<double,3>& center) {
    const int la=a.contr[0].l,lb=b.contr[0].l;
    const Real norm=representation_norm(a)*representation_norm(b);
    const Real distance=std::hypot(Real(a.O[0])-b.O[0],Real(a.O[1])-b.O[1],Real(a.O[2])-b.O[2]);
    if(!std::isfinite(distance)) throw std::overflow_error("MANTA extent center separation overflow");
    std::vector<Primitive> result; result.reserve(a.alpha.size()*b.alpha.size());
    for(std::size_t i=0;i<a.alpha.size();++i) for(std::size_t j=0;j<b.alpha.size();++j) {
        const Real ca=std::abs(a.contr[0].coeff[i]),cb=std::abs(b.contr[0].coeff[j]);
        if(ca==0 || cb==0) continue;
        Primitive p; p.polynomial.fill(minus_inf);
        const Real alpha=a.alpha[i],beta=b.alpha[j]; p.exponent=alpha+beta; p.degree=la+lb;
        if(!std::isfinite(p.exponent)) throw std::overflow_error("MANTA extent exponent sum overflow");
        const Real da=beta/p.exponent*distance,db=alpha/p.exponent*distance;
        std::array<Real,3> displacement{};
        for(int axis=0;axis<3;++axis)
            displacement[axis]=Real(a.O[axis])+beta/p.exponent*(Real(b.O[axis])-a.O[axis])-center[axis];
        p.shift=std::hypot(displacement[0],displacement[1],displacement[2]);
        p.weight=std::log(ca)+std::log(cb)+std::log(norm)-(alpha/p.exponent)*beta*distance*distance;
        if(!std::isfinite(p.weight) || !std::isfinite(p.shift))
            throw std::overflow_error("MANTA extent primitive envelope overflow");
        // Every Cartesian product is bounded by (r+|P-A|)^la (r+|P-B|)^lb.
        // Taking absolute contraction weights also covers nodal/signed shells.
        for(int k=0;k<=la;++k) for(int l=0;l<=lb;++l)
            p.polynomial[k+l]=logadd(p.polynomial[k+l],std::log(choose(la,k)*choose(lb,l))+logpower(da,la-k)+logpower(db,lb-l));
        result.push_back(p);
    }
    return result;
}
Real log_sup_monomial(int degree,Real exponent,Real lower) {
    const Real radius=std::max(lower,std::sqrt((Real(degree)/2)/exponent));
    return logpower(radius,degree)-exponent*radius*radius;
}
struct Bounds {
    Real point=minus_inf;
    std::vector<Real> tails;
};
Bounds evaluate(const std::vector<Primitive>& primitives,double radius,int moment_order) {
    Bounds result; result.tails.assign(moment_order+1,minus_inf);
    for(const auto& p:primitives) {
        const Real lower=std::max(Real(0),Real(radius)-p.shift);
        std::array<Real,maximum_degree+12+3> radial{};
        for(int n=0;n<=p.degree+moment_order;++n) radial[n]=log_radial_tail(n+2,p.exponent,lower);
        for(int n=0;n<=p.degree;++n) {
            const Real weight=p.weight+p.polynomial[n];
            if(weight==minus_inf) continue;
            result.point=logadd(result.point,weight+log_sup_monomial(n,p.exponent,lower));
            // Outside the pair sphere, r_P >= max(0,R-|P-C|).
            // |r-C|^k <= (r_P+|P-C|)^k gives tail moments at the same center.
            for(int k=0;k<=moment_order;++k) for(int t=0;t<=k;++t)
                result.tails[k]=logadd(result.tails[k],std::log(4*pi)+weight+std::log(choose(k,t))
                    +logpower(p.shift,k-t)+radial[n+t]);
        }
    }
    return result;
}
bool meets(const Bounds& b,const ExtentOptions& options) {
    return upper(b.point)<=options.pointwise_tolerance && upper(b.tails[0])<=options.tail_mass_tolerance;
}
template<class Extent> void fill_extent(Extent& pair,const std::vector<Primitive>& terms,
    const ExtentOptions& options,Count& evaluations,Count& always_near) {
    auto probe=[&](double radius) { ++evaluations; return evaluate(terms,radius,options.moment_order); };
    const auto complete=probe(0.);
    pair.l1_norm_bound=upper(complete.tails[0]); pair.linfinity_bound=upper(complete.point);
    if(options.retain_full_moments)
        for(auto value:complete.tails) pair.full_moment_bounds.push_back(upper(value));
    // Splitting the Coulomb integral at a and minimizing
    // 2*pi*||rho||_inf*a^2 + ||rho||_1/a gives 3*(pi/2)^(1/3) M^(1/3) Q^(2/3).
    pair.potential_bound=upper(std::log(Real(3))+std::log(pi/2)/3+(complete.point+2*complete.tails[0])/3);
    double lower=options.minimum_radius,high=lower;
    auto bounds=probe(high);
    int steps=0;
    while(!meets(bounds,options) && high<options.maximum_radius && steps++<search_steps) {
        lower=high;
        high=std::min(options.maximum_radius,std::max(1.,high*2));
        bounds=probe(high);
    }
    if(meets(bounds,options)) {
        for(int step=0;step<search_steps && high>lower;++step) {
            const double middle=lower+(high-lower)/2;
            if(middle==lower || middle==high) break;
            auto trial=probe(middle);
            if(meets(trial,options)) { high=middle; bounds=std::move(trial); }
            else lower=middle;
        }
    }
    pair.radius=high; pair.pointwise_bound=upper(bounds.point); pair.tail_mass_bound=upper(bounds.tails[0]);
    for(auto value:bounds.tails) pair.tail_moment_bounds.push_back(upper(value));
    pair.always_near=!meets(bounds,options);
    if(pair.always_near) {
        pair.reason=high>=options.maximum_radius?"radius-cap":"search-cap";
        ++always_near;
    }
}
std::vector<Primitive> auxiliary_primitives(const libint2::Shell& a) {
    const int l=a.contr[0].l;
    const Real norm=representation_norm(a);
    std::vector<Primitive> result; result.reserve(a.alpha.size());
    for(std::size_t i=0;i<a.alpha.size();++i) {
        const Real coefficient=std::abs(a.contr[0].coeff[i]);
        if(coefficient==0.) continue;
        Primitive p; p.polynomial.fill(minus_inf); p.polynomial[l]=0.;
        p.exponent=a.alpha[i]; p.degree=l; p.weight=std::log(coefficient)+std::log(norm);
        result.push_back(p);
    }
    return result;
}
std::string identity(const BasisSet& basis,const ExtentOptions& options,bool auxiliary=false) {
    vibeqc::detail::Sha256 hash;
    const auto prefix=std::string(auxiliary?"manta.auxiliary-extents.v1;absolute-cartesian-polynomial;shell-center;":
        "manta.pair-extents.v2;absolute-cartesian-polynomial;geometric-midpoint;")+auxiliary_basis_content_identity_sha256(basis);
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    for(double value:{options.pointwise_tolerance,options.tail_mass_tolerance,options.minimum_radius,options.maximum_radius,double(options.moment_order),double(options.retain_full_moments)}) {
        if(value==0.) value=0.; std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits));
        std::array<std::uint8_t,8> bytes{}; for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(bits>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    }
    return hash.finish_hex();
}
} // namespace
ExtentPlan plan_pair_extents(const BasisSet& basis,const ExtentOptions& options,const ResourceBudget& budget) {
    using namespace admission;
    validate_options(options); margin(budget);
    ExtentPlan p;
    p.shell_pairs=mul(basis.nshells(),add(basis.nshells(),1))/2;
    cap(p.shell_pairs,budget.shell_calls,"MANTA extent shell-pair budget exceeded");
    p.modeled_peak_bytes=add(budget.other_live_bytes,budget.backend_margin_bytes);
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(p.shell_pairs,3*(sizeof(PairExtent)+8*(options.moment_order+1)*(1+options.retain_full_moments)+128)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA extent output memory budget exceeded");
    const auto basis_counts=census(basis,6);
    Count diagonal=0,max_primitives=0;
    for(const auto& shell:basis.libint()) {
        diagonal=add(diagonal,square(shell.alpha.size())); max_primitives=std::max<Count>(max_primitives,shell.alpha.size());
    }
    p.primitive_pairs=add(square(basis_counts.primitives),diagonal)/2;
    p.maximum_pair_primitives=square(max_primitives);
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA extent primitive-pair budget exceeded");
    const Count degree=2*basis_counts.maximum_l,moments=options.moment_order+1;
    const Count per_evaluation=add(mul(128,square(degree+moments+3)),mul(128,mul(degree+1,square(moments))));
    p.kernel_terms=mul(p.primitive_pairs,add(4096,mul(2*search_steps+4,per_evaluation)));
    cap(p.kernel_terms,budget.kernel_terms,"MANTA extent work budget exceeded");
    p.scalar_terms=mul(p.shell_pairs,admission::add(12,mul(moments,1+options.retain_full_moments)));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA extent scalar-output budget exceeded");
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(basis_counts.bytes,mul(p.maximum_pair_primitives,512)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA extent primitive workspace budget exceeded");
    validate_basis(basis); return p;
}
PairExtents compute_pair_extents(const BasisSet& basis,const ExtentOptions& options,const ResourceBudget& budget) {
    const auto plan=plan_pair_extents(basis,options,budget);
    PairExtents result; result.pairs.reserve(plan.shell_pairs);
    result.receipt.source_identity=identity(basis,options);
    result.receipt.shell_pairs=plan.shell_pairs; result.receipt.primitive_pairs=plan.primitive_pairs;
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    const auto& shells=basis.libint();
    // Neese et al., BUPO (2025), DOI 10.1021/acs.jpca.4c07415, Sec. 2.5:
    // extents separate near/far fields. This conservative MANTA envelope is
    // an explicit alternative to that paper's axial walkers and signed center.
    for(std::size_t a=0;a<shells.size();++a) for(std::size_t b=0;b<=a;++b) {
        PairExtent pair; pair.shells={a,b};
        for(int axis=0;axis<3;++axis) pair.center[axis]=static_cast<double>((Real(shells[a].O[axis])+shells[b].O[axis])/2);
        const auto terms=primitives(shells[a],shells[b],pair.center);
        fill_extent(pair,terms,options,result.receipt.envelope_evaluations,result.receipt.always_near_pairs);
        result.pairs.push_back(std::move(pair));
    }
    return result;
}
AuxiliaryExtentPlan plan_auxiliary_extents(const BasisSet& basis,const ExtentOptions& options,const ResourceBudget& budget) {
    using namespace admission;
    validate_options(options); margin(budget);
    AuxiliaryExtentPlan plan; plan.shells=basis.nshells();
    cap(plan.shells,budget.shell_calls,"MANTA auxiliary extent shell budget exceeded");
    plan.modeled_peak_bytes=add(budget.other_live_bytes,budget.backend_margin_bytes);
    plan.modeled_peak_bytes=add(plan.modeled_peak_bytes,mul(plan.shells,
        3*(sizeof(AuxiliaryExtent)+8*(options.moment_order+1)*(1+options.retain_full_moments)+128)));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA auxiliary extent output memory budget exceeded");
    const auto counts=census(basis,7);
    plan.primitives=counts.primitives;
    for(const auto& shell:basis.libint())
        plan.maximum_shell_primitives=std::max<Count>(plan.maximum_shell_primitives,shell.alpha.size());
    cap(plan.primitives,budget.primitive_quartets,"MANTA auxiliary extent primitive budget exceeded");
    const Count degree=counts.maximum_l,moments=options.moment_order+1;
    const Count per_evaluation=add(mul(128,square(degree+moments+3)),mul(128,mul(degree+1,square(moments))));
    plan.kernel_terms=mul(plan.primitives,add(4096,mul(2*search_steps+4,per_evaluation)));
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA auxiliary extent work budget exceeded");
    plan.scalar_terms=mul(plan.shells,add(12,mul(moments,1+options.retain_full_moments)));
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA auxiliary extent output budget exceeded");
    plan.modeled_peak_bytes=add(plan.modeled_peak_bytes,add(counts.bytes,mul(plan.maximum_shell_primitives,512)));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA auxiliary extent workspace budget exceeded");
    validate_basis(basis); return plan;
}
AuxiliaryExtents compute_auxiliary_extents(const BasisSet& basis,const ExtentOptions& options,const ResourceBudget& budget) {
    const auto plan=plan_auxiliary_extents(basis,options,budget);
    AuxiliaryExtents result; result.shells.reserve(plan.shells);
    result.receipt.source_identity=identity(basis,options,true);
    result.receipt.shells=plan.shells; result.receipt.primitives=plan.primitives;
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    // Atomic auxiliary leaves in RI-BUPO (Neese et al., 2025,
    // DOI 10.1021/acs.jpca.4c07415). These absolute radial envelopes are
    // MANTA's conservative alternative to the paper's numerical extent scan.
    for(std::size_t shell=0;shell<basis.nshells();++shell) {
        const auto& a=basis.libint()[shell];
        AuxiliaryExtent extent; extent.shell=shell; extent.center=a.O;
        const auto terms=auxiliary_primitives(a);
        fill_extent(extent,terms,options,result.receipt.envelope_evaluations,result.receipt.always_near_shells);
        result.shells.push_back(std::move(extent));
    }
    return result;
}
} // namespace vibeqc::manta
