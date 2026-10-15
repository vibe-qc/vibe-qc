#include "vibeqc/manta_lattice_tail.hpp"
#include "manta_lattice_geometry.hpp"
#include "vibeqc/manta_gaussian_majorants.hpp"
#include "manta_gaussian_domain_internal.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <algorithm>
#include <cstring>

namespace vibeqc::manta {
namespace {
using Real=long double;
constexpr Real pi=3.141592653589793238462643383279502884L;
constexpr int maximum_order=12;
int degree(const md::Index& a) { return a[0]+a[1]+a[2]; }
Real finite(Real x) {
    if(!std::isfinite(x)) throw std::overflow_error("MANTA lattice-tail nonfinite intermediate");
    return x;
}
double binary64(Real x) {
    const auto result=static_cast<double>(finite(x));
    if(!std::isfinite(result)) throw std::overflow_error("MANTA lattice-tail exceeds finite binary64");
    return result;
}
void accumulate(Real value,Real& sum,Real& correction) {
    const auto adjusted=finite(value)-correction,next=finite(sum+adjusted);
    correction=(next-sum)-adjusted; sum=next;
}
using lattice_detail::Hash;
using lattice_detail::Geometry;
Real logadd(Real a,Real b) {
    if(a==-std::numeric_limits<Real>::infinity()) return b;
    if(b==-std::numeric_limits<Real>::infinity()) return a;
    const Real high=std::max(a,b); return high+std::log1p(std::exp(std::min(a,b)-high));
}
// log Gamma(1/2,t). The large-t asymptotic series is used only for t>=64,
// where its terms reach floating-point accuracy well before the least term.
// Keeping this logarithmic avoids erfc underflow before radial rescaling.
Real log_upper_half(Real t) {
    if(std::isinf(t)) return -std::numeric_limits<Real>::infinity();
    if(t<64) return .5L*std::log(pi)+std::log(std::erfc(std::sqrt(t)));
    Real term=1,sum=1;
    for(int k=1;k<=64;++k) {
        term*=-(2*k-1)/(2*t); sum+=term;
        if(std::abs(term)<std::numeric_limits<Real>::epsilon()*.25L*std::abs(sum)) break;
    }
    return -t-.5L*std::log(t)+std::log(sum);
}
std::vector<Real> radial_seeds(Real radius,Real eta,int order,bool near) {
    std::vector<Real> result(order+1);
    if(radius==0 && !near)
        throw std::invalid_argument("MANTA lattice-tail expansion meets a singularity outside the near domain");
    const Real t=(eta*radius)*(eta*radius),log2=std::log(2.L),logpi=std::log(pi);
    // Near cells use -erf(eta*r)/r directly, including the removable origin.
    // d^n/d(r^2)^n [erf(eta*r)/r] = 2*(-1)^n*eta^(2n+1)*F_n(t)/sqrt(pi).
    // The Cartesian Hermite recurrence requires 2^n times these derivatives.
    if(near && t<=order+30) {
        const auto boys=md::boys(order,static_cast<double>(t));
        for(int n=0;n<=order;++n) {
            const Real logvalue=(n+1)*log2+(2*n+1)*std::log(eta)-.5L*logpi+std::log(Real(boys[n]));
            result[n]=finite((n%2?1.L:-1.L)*std::exp(logvalue));
        }
        return result;
    }
    // erfc seeds = (-2)^n Gamma(n+1/2,t)/(sqrt(pi)*r^(2n+1)).
    // The positive upper-gamma recurrence avoids Coulomb-minus-erf loss.
    // For large-t erf, the lower-gamma complement is well conditioned.
    Real upper=log_upper_half(t);
    for(int n=0;n<=order;++n) {
        Real gamma=upper;
        if(near) {
            const Real complete=std::lgamma(n+.5L);
            gamma=complete+std::log1p(-std::exp(upper-complete));
        }
        const Real logvalue=n*log2+gamma-.5L*logpi-(2*n+1)*std::log(radius);
        result[n]=finite(((n%2)!=near?-1.L:1.L)*std::exp(logvalue));
        if(n<order && !std::isinf(t))
            upper=logadd(std::log(n+.5L)+upper,(n+.5L)*std::log(t)-t);
    }
    return result;
}
struct Taylor {
    int order,side;
    std::vector<md::Index> indices;
    std::vector<int> lookup;
    std::vector<Real> factorial,divisors,radial,coefficients,errors;
    explicit Taylor(int p):order(p),side(p+1),indices(md::hermites(p)),
        lookup(side*side*side,-1),factorial(side,1),divisors(indices.size()),
        radial(side*indices.size()),coefficients(indices.size(),0),errors(indices.size(),0) {
        for(int i=1;i<=p;++i) factorial[i]=factorial[i-1]*i;
        for(std::size_t i=0;i<indices.size();++i) {
            lookup[offset(indices[i])]=static_cast<int>(i);
            divisors[i]=factorial[indices[i][0]]*factorial[indices[i][1]]*factorial[indices[i][2]];
        }
    }
    int offset(const md::Index& a) const { return (a[0]*side+a[1])*side+a[2]; }
    void real_term(const std::array<Real,3>& x,Real eta,bool near) {
        const auto seeds=radial_seeds(std::hypot(x[0],x[1],x[2]),eta,order,near);
        const auto size=indices.size();
        for(int n=0;n<=order;++n) radial[n*size]=seeds[n];
        for(std::size_t i=1;i<size;++i) {
            auto lower=indices[i]; int axis=0; while(lower[axis]==0) ++axis;
            const int d=degree(lower); --lower[axis]; const int one=lookup[offset(lower)];
            const int multiple=lower[axis]; int two=0;
            if(multiple) { --lower[axis]; two=lookup[offset(lower)]; }
            for(int n=0;n<=order-d;++n)
                radial[n*size+i]=finite(x[axis]*radial[(n+1)*size+one]
                    +(multiple?multiple*radial[(n+1)*size+two]:0));
        }
        for(std::size_t i=0;i<size;++i)
            accumulate(radial[i]/divisors[i],coefficients[i],errors[i]);
    }
    void reciprocal_term(const std::array<Real,3>& g,const md::Point& r,Real eta,Real volume) {
        const Real norm=finite(std::hypot(g[0],g[1],g[2]));
        if(norm==0) throw std::invalid_argument("MANTA lattice-tail reciprocal vector vanished");
        const Real t=(norm/eta*.5L)*(norm/eta*.5L);
        const Real logamplitude=std::log(4*pi)-std::log(volume)-2*std::log(norm)-t;
        const Real phase=finite(g[0]*r[0]+g[1]*r[1]+g[2]*r[2]);
        const Real c=std::cos(phase),s=std::sin(phase);
        const std::array<Real,4> cycle{c,-s,-c,s};
        for(std::size_t i=0;i<indices.size();++i) {
            Real logvalue=logamplitude,sign=1;
            bool zero=false;
            for(int axis=0;axis<3;++axis) if(indices[i][axis]) {
                if(g[axis]==0) { zero=true; break; }
                logvalue+=indices[i][axis]*std::log(std::abs(g[axis]));
                if(g[axis]<0 && indices[i][axis]%2) sign=-sign;
            }
            if(!zero) accumulate(sign*std::exp(logvalue)*cycle[degree(indices[i])%4]/divisors[i],coefficients[i],errors[i]);
        }
    }
};
} // namespace

static LatticeTailPlan plan_coefficients(int order,Count nr,Count ng,Count nn,const ResourceBudget& budget,bool isolated=false) {
    using namespace admission;
    if(order<0 || order>maximum_order) throw std::invalid_argument("MANTA lattice-tail order must be zero through twelve");
    if(!nr || (!isolated && !nn) || nn>nr) throw std::invalid_argument("MANTA lattice-tail requires nonempty real and near domains with near contained in real");
    margin(budget);
    LatticeTailPlan p; p.order=order; p.real_cells=nr; p.reciprocal_vectors=ng; p.near_cells=nn;
    p.moments=Count(order+1)*(order+2)*(order+3)/6;
    p.scalar_terms=square(p.moments);
    const auto count=add(add(nr,ng),nn),terms=add(nr,ng);
    Count levels=1; for(auto n=count;n>1;n>>=1) ++levels;
    p.kernel_terms=add(add(mul(128,p.scalar_terms),mul(128,mul(count,levels))),
        mul(terms,add(mul(256,mul(order+1,p.moments)),mul(8192,order+1))));
    p.output_bytes=add(add(mul(8,p.scalar_terms),mul(12,p.moments)),4096);
    p.domain_bytes=mul(96,count); // borrowed inputs, binding copies, canonical snapshots, sort margin
    p.workspace_bytes=add(add(mul(16,mul(order+1,p.moments)),mul(96,p.moments)),
        add(mul(8,mul(square(order+1),order+1)),8192));
    p.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(add(mul(2,p.output_bytes),p.domain_bytes),p.workspace_bytes));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA lattice-tail matrix-element budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA lattice-tail work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA lattice-tail memory budget exceeded");
    return p;
}

LatticeTailPlan plan_lattice_tail(int order,Count nr,Count ng,Count nn,const ResourceBudget& budget) {
    return plan_coefficients(order,nr,ng,nn,budget);
}
LatticeTailPlan plan_erfc_multipole(int order,const ResourceBudget& budget) {
    return plan_coefficients(order,1,0,0,budget,true);
}
namespace {
void erfc_identity(LatticeTailMatrix& result,const md::Point& displacement,double omega,int order) {
    if(!std::isfinite(omega) || omega<=0.) throw std::invalid_argument("MANTA erfc multipole omega must be finite positive");
    for(double x:displacement) admission::require_finite(x);
    result.eta=omega;result.boundary="isolated-erfc";result.qualification="isolated-erfc-coefficients";
    Hash kernel;kernel.text("manta.erfc-kernel.v1;raw-cartesian;total-degree;binary64");kernel.number(omega);
    result.kernel_identity=kernel.value.finish_hex();
    Hash source;source.text(result.kernel_identity);source.word(order);
    for(double x:displacement) source.number(x);
    result.source_identity=source.value.finish_hex();
}
void coefficient_matrix(const Taylor& t,LatticeTailMatrix& result) {
    result.indices=t.indices;
    result.values=Eigen::MatrixXd::Zero(t.indices.size(),t.indices.size());
    for(std::size_t i=0;i<t.indices.size();++i) for(std::size_t j=0;j<t.indices.size();++j) {
        const auto& a=t.indices[i]; const auto& b=t.indices[j];
        const md::Index total{a[0]+b[0],a[1]+b[1],a[2]+b[2]};
        if(degree(total)>t.order) continue;
        Real factor=degree(b)%2?-1:1;
        for(int axis=0;axis<3;++axis) factor*=t.factorial[total[axis]]/(t.factorial[a[axis]]*t.factorial[b[axis]]);
        result.values(i,j)=binary64(factor*t.coefficients[t.lookup[t.offset(total)]]);
        ++result.matrix_coefficients_evaluated;
    }
}
}
LatticeTailMatrix multipole_erfc(const md::Point& displacement,double omega,int order,const ResourceBudget& budget) {
    LatticeTailMatrix result;result.plan=plan_erfc_multipole(order,budget);
    erfc_identity(result,displacement,omega,order);
    Taylor t(order);
    t.real_term({Real(displacement[0]),Real(displacement[1]),Real(displacement[2])},omega,false);
    result.radial_derivatives_evaluated=t.indices.size();
    coefficient_matrix(t,result);return result;
}

LatticeTailMatrix multipole_lattice_tail(const Eigen::Matrix3d& cell,const md::Point& displacement,
    double eta,const std::vector<LatticeIndex>& real_input,const std::vector<LatticeIndex>& reciprocal_input,
    const std::vector<LatticeIndex>& near_input,int order,const ResourceBudget& budget) {
    LatticeTailMatrix result;
    result.plan=plan_lattice_tail(order,real_input.size(),reciprocal_input.size(),near_input.size(),budget);
    const Geometry geometry(cell,displacement,eta,real_input,reciprocal_input,near_input,order);
    const auto& real=geometry.real; const auto& reciprocal=geometry.reciprocal; const auto& near=geometry.near;
    const double volume=geometry.volume;
    result.eta=eta; result.cell_volume=volume;
    result.gauge_constant=binary64(-std::exp(std::log(pi)-2*std::log(Real(eta))-std::log(Real(volume))));
    result.real_domain_identity=geometry.real_identity;
    result.reciprocal_domain_identity=geometry.reciprocal_identity;
    result.near_domain_identity=geometry.near_identity;
    result.kernel_identity=geometry.kernel_identity;
    result.source_identity=geometry.expansion_identity;

    // Ewald (1921), doi:10.1002/andp.19213690304. Periodic FMM context:
    // Kudin/Scuseria (2000), doi:10.1103/PhysRevB.61.16440. The complementary
    // tail and raw Cartesian convention are derived in the MANTA scientific
    // contract, section 5.2; this is not the molecular BUPO harmonic kernel.
    Taylor t(order);
    for(const auto& n:real) {
        std::array<Real,3> x{};
        for(int i=0;i<3;++i) {
            x[i]=displacement[i]; for(int j=0;j<3;++j) x[i]+=Real(cell(i,j))*n[j];
            finite(x[i]);
        }
        t.real_term(x,eta,std::binary_search(near.begin(),near.end(),n));
        result.radial_derivatives_evaluated+=t.indices.size();
    }
    const Eigen::Matrix3d dual=cell.inverse().transpose();
    for(const auto& n:reciprocal) {
        std::array<Real,3> g{};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) g[i]+=2*pi*Real(dual(i,j))*n[j];
        t.reciprocal_term(g,displacement,eta,volume);
        result.reciprocal_coefficients_evaluated+=t.indices.size();
    }
    accumulate(result.gauge_constant,t.coefficients[0],t.errors[0]);
    coefficient_matrix(t,result);
    return result;
}

namespace {
constexpr Real minus_inf=-std::numeric_limits<Real>::infinity();
Real logpower(Real x,int n) { return n==0?0:x==0?minus_inf:n*std::log(x); }
// Only true algebraic zeros return zero. Positive contributions smaller than
// binary64's subnormal range round UP to its smallest positive value. As for
// the molecular error ledger, this guard is not an interval-libm certificate
// and does not include coefficient, moment or contraction roundoff.
double remainder_upper(Real logarithm) {
    if(logarithm==minus_inf) return 0.;
    finite(logarithm);
    const Real value=std::exp(logarithm)*(1+8192*std::numeric_limits<double>::epsilon());
    if(!std::isfinite(value) || value>std::numeric_limits<double>::max())
        throw std::overflow_error("MANTA lattice-tail remainder exceeds finite binary64");
    const double result=std::nextafter(static_cast<double>(value),std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA lattice-tail remainder exceeds finite binary64");
    return result;
}
Real complete_gamma(int m) {
    Real value=.5L*std::log(pi);
    for(int j=0;j<m;++j) value+=std::log(j+.5L);
    return value;
}
// Upper bound on log integral s^(2m) exp(-d^2 s^2) ds, on [0,eta]
// (near) or [eta,infinity) (non-near). Positive series/recurrences avoid
// cancellation and special-function underflow. This is deliberately an upper
// envelope, not a high-accuracy incomplete-gamma evaluation.
Real log_integral_bound(int m,Real eta,Real d,bool near) {
    const Real power=2*m+1,shape=m+.5L,logeta=std::log(eta);
    if(d==0) {
        if(!near) throw std::invalid_argument("MANTA lattice-tail remainder support meets a non-near singularity");
        return power*logeta-std::log(power);
    }
    const Real logd=std::log(d),logx=2*(logeta+logd);
    // Replacing an enormous x by a smaller one only increases the upper
    // incomplete gamma. It keeps positive tiny bounds distinct from zero.
    const Real x=std::exp(std::min(logx,std::log(1.e6L)));
    if(near) {
        if(logx>std::log(shape+32))
            return complete_gamma(m)-std::log(2.L)-power*logd;
        // gamma(shape,x) = x^shape exp(-x) sum_{j>=0}
        //  x^j/[shape*(shape+1)*...*(shape+j)]. After the ratios fall below
        // one, a geometric series bounds ALL remaining positive terms.
        Real term=1/shape,sum=term;
        for(int j=0;j<512;++j) {
            const Real ratio=x/(shape+j+1);
            if(ratio<1) {
                const Real tail=term*ratio/(1-ratio);
                if(tail<=std::numeric_limits<Real>::epsilon()*.125L*sum)
                    return power*logeta-x-std::log(2.L)+std::log(sum+tail);
            }
            term*=ratio; sum+=term;
        }
        throw std::runtime_error("MANTA lattice-tail remainder series exceeded admitted work");
    }
    // Gamma(1/2,x) <= min(sqrt(pi), exp(-x)/sqrt(x)); integration
    // by parts then gives positive upper bounds through m+1/2.
    const Real used_logx=std::min(logx,std::log(1.e6L));
    Real upper=std::min(.5L*std::log(pi),-x-.5L*used_logx);
    for(int j=0;j<m;++j)
        upper=logadd(std::log(j+.5L)+upper,(j+.5L)*used_logx-x);
    return upper-std::log(2.L)-power*logd;
}
} // namespace

static LatticeTailRemainderPlan plan_remainder(int order,Count nr,Count ng,Count nn,
    const ResourceBudget& budget,bool isolated=false) {
    using namespace admission;
    if(order<0 || order>maximum_order) throw std::invalid_argument("MANTA lattice-tail order must be zero through twelve");
    if(!nr || (!isolated && !nn) || nn>nr) throw std::invalid_argument("MANTA lattice-tail requires nonempty real and near domains with near contained in real");
    margin(budget);
    LatticeTailRemainderPlan p; p.order=order; p.real_cells=nr; p.reciprocal_vectors=ng; p.near_cells=nn;
    const auto count=add(add(nr,ng),nn);
    Count levels=1; for(auto n=count;n>1;n>>=1) ++levels;
    p.radial_integrals=mul(nr,(order+1)/2+1);
    p.scalar_terms=add(p.radial_integrals,ng);
    p.kernel_terms=add(mul(128,mul(count,levels)),add(mul(65536,p.radial_integrals),mul(512,ng)));
    p.output_bytes=4096; p.workspace_bytes=8192; p.domain_bytes=mul(96,count);
    p.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(add(mul(2,p.output_bytes),p.domain_bytes),p.workspace_bytes));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA lattice-tail remainder scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA lattice-tail remainder work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA lattice-tail remainder memory budget exceeded");
    return p;
}

LatticeTailRemainderPlan plan_lattice_tail_remainder(int order,Count nr,Count ng,Count nn,const ResourceBudget& budget) {
    return plan_remainder(order,nr,ng,nn,budget);
}
LatticeTailRemainderPlan plan_erfc_remainder(int order,const ResourceBudget& budget) {
    return plan_remainder(order,1,0,0,budget,true);
}
namespace {
Real real_remainder_term(const std::array<Real,3>& x,const std::array<Real,3>& absolute,
    double eta,int n,Real rho,bool zero,bool near,LatticeTailRemainder& result) {
    const Real distance=finite(std::hypot(x[0],x[1],x[2]));
    const Real guard=finite(64*std::numeric_limits<double>::epsilon()*std::hypot(absolute[0],absolute[1],absolute[2]));
    const Real d=std::max(0.L,distance-guard-rho),maximum=finite(distance+guard+rho);
    if(!near) {
        if(d<=0) throw std::invalid_argument("MANTA lattice-tail remainder support meets a non-near singularity");
        result.minimum_non_near_distance=std::min(result.minimum_non_near_distance,static_cast<double>(d));
    }
    if(zero) return minus_inf; // Support is validated even for zero charge/radius.
    Real term=minus_inf;
    for(int k=0;k<=n/2;++k) {
        const int power=n-2*k;
        term=logadd(term,(power+1)*std::log(2.L)-.5L*std::log(pi)+n*std::log(rho)
            +logpower(maximum,power)+log_integral_bound(n-k,eta,d,near)
            -std::lgamma(k+1.L)-std::lgamma(power+1.L));
        ++result.radial_integrals_evaluated;
    }
    return term;
}
}
LatticeTailRemainder erfc_remainder(const md::Point& displacement,double omega,int order,
    double ra,double rb,double qa,double qb,const ResourceBudget& budget) {
    LatticeTailRemainder result;result.plan=plan_erfc_remainder(order,budget);
    for(double x:{ra,rb,qa,qb}) {
        admission::require_finite(x);
        if(x<0) throw std::invalid_argument("MANTA erfc radii and absolute norms must be nonnegative");
    }
    LatticeTailMatrix identity;erfc_identity(identity,displacement,omega,order);
    result.expansion_identity=identity.source_identity;result.kernel_identity=identity.kernel_identity;
    result.qualification="isolated-erfc-compact-taylor";
    result.target_radius=ra;result.source_radius=rb;result.target_norm=qa;result.source_norm=qb;
    Hash hash;hash.text("manta.erfc-remainder.v1;directional-gaussian;compact;positive-integral-envelopes");
    hash.text(result.expansion_identity);for(double x:{ra,rb,qa,qb}) hash.number(x);
    result.source_identity=hash.value.finish_hex();
    std::array<Real,3> x{},absolute{};
    for(int i=0;i<3;++i) {x[i]=displacement[i];absolute[i]=std::abs(x[i]);}
    const Real rho=finite((Real(ra)+rb)*(1+8*std::numeric_limits<double>::epsilon()));
    const Real term=real_remainder_term(x,absolute,omega,order+1,rho,rho==0 || qa==0 || qb==0,false,result);
    const Real charge=qa==0 || qb==0?0:std::log(Real(qa))+std::log(Real(qb));
    result.real_remainder=result.total=remainder_upper(term+charge);return result;
}

LatticeTailRemainder lattice_tail_remainder(const Eigen::Matrix3d& cell,const md::Point& displacement,
    double eta,const std::vector<LatticeIndex>& rs,const std::vector<LatticeIndex>& gs,
    const std::vector<LatticeIndex>& ns,int order,double ra,double rb,double qa,double qb,const ResourceBudget& budget) {
    LatticeTailRemainder result;
    result.plan=plan_lattice_tail_remainder(order,rs.size(),gs.size(),ns.size(),budget);
    for(double x:{ra,rb,qa,qb}) {
        admission::require_finite(x);
        if(x<0) throw std::invalid_argument("MANTA lattice-tail radii and absolute charge norms must be nonnegative");
    }
    const Geometry geometry(cell,displacement,eta,rs,gs,ns,order);
    result.target_radius=ra; result.source_radius=rb; result.target_norm=qa; result.source_norm=qb;
    result.expansion_identity=geometry.expansion_identity; result.kernel_identity=geometry.kernel_identity;
    result.real_domain_identity=geometry.real_identity; result.reciprocal_domain_identity=geometry.reciprocal_identity;
    result.near_domain_identity=geometry.near_identity;
    Hash identity; identity.text("manta.lattice-tail.remainder.v1;directional-gaussian;compact;positive-integral-envelopes");
    identity.text(result.expansion_identity);
    for(double x:{ra,rb,qa,qb}) identity.number(x);
    result.source_identity=identity.value.finish_hex();
    const Real eps=std::numeric_limits<double>::epsilon();
    const Real rho=finite((Real(ra)+rb)*(1+8*eps));
    const bool zero=rho==0 || qa==0 || qb==0;
    const int n=order+1;
    Real near_sum=minus_inf,real_sum=minus_inf,reciprocal_sum=minus_inf;
    // Directional Taylor theorem applied to exp(-s^2 |x+t*h|^2).
    // See scientific contract 5.2: the same finite Ewald-minus-near source
    // as Ewald (1921), DOI 10.1002/andp.19213690304. This derivation is NOT
    // the harmonic Coulomb remainder from the molecular BUPO paper.
    for(const auto& index:geometry.real) {
        std::array<Real,3> x{},absolute{};
        for(int i=0;i<3;++i) {
            x[i]=displacement[i]; absolute[i]=std::abs(x[i]);
            for(int j=0;j<3;++j) { const Real v=Real(cell(i,j))*index[j]; x[i]+=v; absolute[i]+=std::abs(v); }
            finite(x[i]); finite(absolute[i]);
        }
        const bool near=std::binary_search(geometry.near.begin(),geometry.near.end(),index);
        const Real term=real_remainder_term(x,absolute,eta,n,rho,zero,near,result);
        if(near) near_sum=logadd(near_sum,term); else real_sum=logadd(real_sum,term);
    }
    const Eigen::Matrix3d dual=cell.inverse().transpose();
    for(const auto& index:geometry.reciprocal) {
        std::array<Real,3> g{};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) g[i]+=2*pi*Real(dual(i,j))*index[j];
        const Real norm=finite(std::hypot(g[0],g[1],g[2]));
        if(norm==0) throw std::invalid_argument("MANTA lattice-tail reciprocal vector vanished");
        if(zero) continue;
        const Real lognorm=std::log(norm),logt=2*(lognorm-std::log(Real(eta))-std::log(2.L));
        const Real t=std::exp(std::min(logt,std::log(1.e6L)));
        reciprocal_sum=logadd(reciprocal_sum,std::log(4*pi)-std::log(Real(geometry.volume))-2*lognorm-t
            +n*(std::log(rho)+lognorm)-std::lgamma(n+1.L));
        ++result.reciprocal_terms_evaluated;
    }
    const Real logcharge=qa==0 || qb==0?0:std::log(Real(qa))+std::log(Real(qb));
    result.near_remainder=remainder_upper(near_sum+logcharge);
    result.real_remainder=remainder_upper(real_sum+logcharge);
    result.reciprocal_remainder=remainder_upper(reciprocal_sum+logcharge);
    Real total=minus_inf;
    for(double x:{result.near_remainder,result.real_remainder,result.reciprocal_remainder})
        if(x>0) total=logadd(total,std::log(Real(x)));
    result.total=remainder_upper(total);
    return result;
}
namespace {
Count complete_cube(const std::vector<LatticeIndex>& indices,bool origin) {
    Count radius=0;
    for(const auto& n:indices) for(auto x:n) radius=std::max(radius,static_cast<Count>(std::abs(x)));
    const Count width=admission::add(admission::mul(2,radius),1);
    const Count required=admission::mul(admission::square(width),width)-(origin?0:1);
    // Geometry already proved uniqueness, origin convention and finite index
    // range. Count plus the maximum coordinate therefore proves completeness.
    if(required!=indices.size())
        throw std::invalid_argument("MANTA lattice-domain remainder requires complete centered integer cubes");
    return radius;
}
Real frobenius(const Eigen::Matrix3d& a) {
    Real norm=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) norm=std::hypot(norm,Real(a(i,j)));
    return finite(norm);
}
Real real_metric(const Eigen::Matrix3d& a) {
    const Eigen::Matrix3d b=a.inverse();
    const Real eps=std::numeric_limits<double>::epsilon();
    Real residual=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        Real x=-(i==j?1.L:0.L),absolute=std::abs(x);
        for(int k=0;k<3;++k) {
            const Real term=finite(Real(b(i,k))*a(k,j)); x+=term; absolute+=std::abs(term);
        }
        residual=std::hypot(residual,std::abs(finite(x))+64*eps*finite(absolute));
    }
    residual=finite(residual*(1+64*eps));
    if(residual>=1) throw std::invalid_argument("MANTA lattice-domain inverse residual cannot certify a positive metric bound");
    // ||x|| <= ||B||_F ||A x|| + ||I-BA||_F ||x||. This checks
    // the computed inverse instead of trusting its norm for an ill-conditioned A.
    return finite((1-residual)/(frobenius(b)*(1+64*eps))*(1-64*eps));
}
double lower_scalar(Real x) {
    const double out=std::nextafter(binary64(x),0.);
    if(out<=0) throw std::overflow_error("MANTA lattice-domain positive distance/metric is not representable");
    return out;
}
Real decay(Real logarithm) {
    // Replacing an enormous positive Gaussian argument by 1e6 only
    // increases the remainder and preserves positive subnormal bounds.
    return std::exp(std::min(logarithm,std::log(1.e6L)));
}
Real real_domain_log(Real m,Real logeta,Real logd,Real logs) {
    return std::log(24*m*m+2)-.5L*std::log(pi)-logeta-2*logd
        -decay(2*(logeta+logd))+logadd(0,-std::log(2.L)-2*logeta-logs-logd);
}
Real reciprocal_domain_log(Real n,Real logeta,Real logg,double volume) {
    const Real loga=logg-std::log(2.L)-logeta;
    return std::log(4*pi)-std::log(Real(volume))-2*logg
        +std::log(24+2/(n*n))-decay(2*(loga+std::log(n)))
        +logadd(0,-std::log(2.L)-2*loga-std::log(n));
}
} // namespace

LatticeDomainPlan plan_lattice_domain_remainder(Count nr,Count ng,Count nn,const ResourceBudget& budget) {
    using namespace admission;
    if(!nr || !nn || nn>nr) throw std::invalid_argument("MANTA lattice-domain remainder requires nonempty real/near domains");
    margin(budget);
    LatticeDomainPlan p; p.real_cells=nr; p.reciprocal_vectors=ng; p.near_cells=nn;
    const auto count=add(add(nr,ng),nn); Count levels=1;
    for(auto n=count;n>1;n>>=1) ++levels;
    p.scalar_terms=add(mul(3,count),256);
    p.kernel_terms=add(mul(256,mul(count,levels)),8192);
    p.output_bytes=4096; p.workspace_bytes=8192; p.domain_bytes=mul(96,count);
    p.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(add(mul(2,p.output_bytes),p.domain_bytes),p.workspace_bytes));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA lattice-domain remainder scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA lattice-domain remainder work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA lattice-domain remainder memory budget exceeded");
    return p;
}

LatticeDomainRemainder lattice_domain_remainder(const Eigen::Matrix3d& cell,const md::Point& r,
    double eta,const std::vector<LatticeIndex>& rs,const std::vector<LatticeIndex>& gs,
    const std::vector<LatticeIndex>& ns,double ra,double rb,double qa,double qb,const ResourceBudget& budget) {
    LatticeDomainRemainder out;
    out.plan=plan_lattice_domain_remainder(rs.size(),gs.size(),ns.size(),budget);
    for(double x:{ra,rb,qa,qb}) {
        admission::require_finite(x);
        if(x<0) throw std::invalid_argument("MANTA lattice-domain radii and absolute charge norms must be nonnegative");
    }
    const Geometry geometry(cell,r,eta,rs,gs,ns,0);
    out.real_cube_radius=complete_cube(geometry.real,true);
    out.reciprocal_cube_radius=complete_cube(geometry.reciprocal,false);
    out.target_radius=ra; out.source_radius=rb; out.target_norm=qa; out.source_norm=qb;
    out.kernel_identity=geometry.kernel_identity; out.real_domain_identity=geometry.real_identity;
    out.reciprocal_domain_identity=geometry.reciprocal_identity; out.near_domain_identity=geometry.near_identity;
    Hash identity; identity.text("manta.lattice-domain-remainder.v1;compact;complete-cubes;frobenius-shell-envelopes");
    identity.text(out.kernel_identity);
    for(double x:r) identity.number(x);
    for(double x:{ra,rb,qa,qb}) identity.number(x);
    out.source_identity=identity.value.finish_hex();
    const Real eps=std::numeric_limits<double>::epsilon();
    out.real_metric_lower=lower_scalar(real_metric(cell));
    out.reciprocal_metric_lower=lower_scalar(2*pi/(frobenius(cell)*(1+64*eps))*(1-64*eps));
    const Real span=finite((std::hypot(Real(r[0]),Real(r[1]),Real(r[2]))+ra+rb)*(1+64*eps));
    out.displacement_bound=span==0?0:std::nextafter(binary64(span),std::numeric_limits<double>::infinity());
    const Real m=out.real_cube_radius+1,n=out.reciprocal_cube_radius+1;
    const Real distance=finite(Real(out.real_metric_lower)*m-out.displacement_bound);
    if(distance<=0) throw std::invalid_argument("MANTA lattice-domain omitted real images are not separated from the compact support");
    out.minimum_omitted_real_distance=lower_scalar(distance*(1-64*eps));
    if(qa==0 || qb==0) return out; // All domain/support checks still apply.
    const Real logeta=std::log(Real(eta)),logd=std::log(Real(out.minimum_omitted_real_distance));
    const Real logs=std::log(Real(out.real_metric_lower)),logg=std::log(Real(out.reciprocal_metric_lower));
    const Real logcharge=std::log(Real(qa))+std::log(Real(qb));
    // Ewald (1921), doi:10.1002/andp.19213690304, split Green function.
    // A max-norm integer shell m has 24*m^2+2 points. For d_m=s*m-B>0,
    // erfc(eta*d_m)/d_m <= exp(-eta^2*d_m^2)/(sqrt(pi)*eta*d_m^2).
    // The shell prefactor decreases with m; the remaining Gaussian series
    // is bounded by its first term plus its integral. See contract section5.2.
    const Real real_log=real_domain_log(m,logeta,logd,logs);
    // Reciprocal phases have modulus one. sigma_G >= 2*pi/||A||_F,
    // with shell prefactor (24+2/n^2)/sigma_G^2 and Gaussian width 2*eta.
    const Real reciprocal_log=reciprocal_domain_log(n,logeta,logg,geometry.volume);
    out.real_remainder=remainder_upper(real_log+logcharge);
    out.reciprocal_remainder=remainder_upper(reciprocal_log+logcharge);
    out.total=remainder_upper(logadd(std::log(Real(out.real_remainder)),std::log(Real(out.reciprocal_remainder))));
    return out;
}

LatticeGaussianDomainPlan plan_lattice_gaussian_domain_remainder(Count nt,Count ns,Count nr,Count ng,
    Count nn,const ResourceBudget& budget) {
    using namespace admission;
    LatticeGaussianDomainPlan p; p.target_terms=nt; p.source_terms=ns;
    p.primitive_products=mul(nt,ns);
    cap(p.primitive_products,budget.primitive_quartets,"MANTA Gaussian domain primitive budget exceeded");
    p.domain=plan_lattice_domain_remainder(nr,ng,nn,budget);
    p.scalar_terms=add(p.domain.scalar_terms,add(mul(5,add(nt,ns)),32));
    p.kernel_terms=add(p.domain.kernel_terms,mul(4096,p.primitive_products));
    p.majorant_bytes=add(mul(3*sizeof(GaussianMajorantTerm),add(nt,ns)),8192);
    p.modeled_peak_bytes=add(p.domain.modeled_peak_bytes,p.majorant_bytes);
    cap(p.scalar_terms,budget.scalar_terms,"MANTA Gaussian domain scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA Gaussian domain work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA Gaussian domain memory budget exceeded");
    return p;
}
LatticeGaussianDomainRemainder lattice_gaussian_domain_remainder(const Eigen::Matrix3d& cell,double eta,
    const std::vector<LatticeIndex>& rs,const std::vector<LatticeIndex>& gs,const std::vector<LatticeIndex>& ns,
    const GaussianMajorant& target,const GaussianMajorant& source,const ResourceBudget& budget) {
    plan_lattice_gaussian_domain_remainder(target.terms().size(),source.terms().size(),rs.size(),gs.size(),ns.size(),budget);
    return gaussian_domain_detail::evaluate(gaussian_domain_detail::prepare(cell,eta,rs,gs,ns),target,source,budget);
}
namespace gaussian_domain_detail {
double real_metric_lower(const Eigen::Matrix3d& cell) { return lower_scalar(real_metric(cell)); }
Geometry prepare(const Eigen::Matrix3d& cell,double eta,const std::vector<LatticeIndex>& rs,
    const std::vector<LatticeIndex>& gs,const std::vector<LatticeIndex>& ns) {
    const lattice_detail::Geometry source(cell,{},eta,rs,gs,ns,0);
    Geometry out; out.real_cells=rs.size(); out.reciprocal_vectors=gs.size(); out.near_cells=ns.size();
    out.real_cube_radius=complete_cube(source.real,true);
    out.reciprocal_cube_radius=complete_cube(source.reciprocal,false);
    out.eta=eta; out.volume=source.volume; out.kernel_identity=source.kernel_identity;
    out.real_identity=source.real_identity; out.reciprocal_identity=source.reciprocal_identity; out.near_identity=source.near_identity;
    const Real eps=std::numeric_limits<double>::epsilon();
    out.metric=lower_scalar(real_metric(cell));
    out.reciprocal_metric=lower_scalar(2*pi/(frobenius(cell)*(1+64*eps))*(1-64*eps));
    return out;
}
LatticeGaussianDomainRemainder evaluate(const Geometry& geometry,const GaussianMajorant& target,
    const GaussianMajorant& source,const ResourceBudget& budget) {
    LatticeGaussianDomainRemainder out; const double eta=geometry.eta;
    out.plan=plan_lattice_gaussian_domain_remainder(target.terms().size(),source.terms().size(),
        geometry.real_cells,geometry.reciprocal_vectors,geometry.near_cells,budget);
    out.real_cube_radius=geometry.real_cube_radius;
    out.reciprocal_cube_radius=geometry.reciprocal_cube_radius;
    out.physical_eta=eta; out.kernel_identity=geometry.kernel_identity;
    out.real_domain_identity=geometry.real_identity; out.reciprocal_domain_identity=geometry.reciprocal_identity;
    out.near_domain_identity=geometry.near_identity;
    out.target_majorant_identity=target.source_identity(); out.source_majorant_identity=source.source_identity();
    Hash identity; identity.text("manta.gaussian-domain-remainder.v1;full-gaussian-convolution;complete-cubes");
    identity.text(out.kernel_identity); identity.text(out.target_majorant_identity); identity.text(out.source_majorant_identity);
    out.source_identity=identity.value.finish_hex();
    const Real eps=std::numeric_limits<double>::epsilon();
    const Real metric=geometry.metric,reciprocal_metric=geometry.reciprocal_metric;
    Real sum=-std::numeric_limits<Real>::infinity();
    for(const auto& a:target.terms()) for(const auto& b:source.terms()) {
        const Real span=finite(std::hypot(Real(a.center[0])-b.center[0],Real(a.center[1])-b.center[1],
            Real(a.center[2])-b.center[2])*(1+64*eps));
        const Real guard=span==0?0:std::nextafter(binary64(span),std::numeric_limits<double>::infinity());
        const Real distance=finite(metric*(out.real_cube_radius+1)-guard);
        if(distance<=0) throw std::invalid_argument("MANTA Gaussian domain omitted real images are not separated from majorant centers");
        const double d=lower_scalar(distance*(1-64*eps));
        // Convolution of two normalized positive Gaussians with erfc(eta*r)/r
        // is [erf(sqrt(beta)*d)-erf(effective_eta*d)]/d, beta^-1=a^-1+b^-1.
        // Bounding erf(sqrt(beta)*d) by one reuses the point-kernel shell bound.
        // eta here is an envelope decay only; the physical Ewald split is unchanged.
        const Real log_eta=-.5L*logadd(-2*std::log(Real(eta)),
            logadd(-std::log(Real(a.exponent)),-std::log(Real(b.exponent))));
        const double effective=lower_scalar(std::exp(log_eta)*(1-256*eps));
        out.minimum_effective_eta=std::min(out.minimum_effective_eta,effective);
        out.minimum_omitted_real_distance=std::min(out.minimum_omitted_real_distance,d);
        sum=logadd(sum,real_domain_log(out.real_cube_radius+1,std::log(Real(effective)),std::log(Real(d)),std::log(metric))
            +std::log(Real(a.charge_bound))+std::log(Real(b.charge_bound)));
        ++out.majorant_pairs_evaluated;
    }
    if(!out.majorant_pairs_evaluated) return out; // Geometry/cube admission still applies.
    out.real_remainder=remainder_upper(sum);
    const Real reciprocal_log=reciprocal_domain_log(out.reciprocal_cube_radius+1,std::log(Real(eta)),
        std::log(reciprocal_metric),geometry.volume);
    out.reciprocal_remainder=remainder_upper(reciprocal_log+std::log(Real(target.charge_bound()))+std::log(Real(source.charge_bound())));
    out.total=remainder_upper(logadd(std::log(Real(out.real_remainder)),std::log(Real(out.reciprocal_remainder))));
    return out;
}
namespace {
Real point_real_log(const Geometry& g,const md::Point& a,const md::Point& b,Real log_inverse_exponent) {
    const Real eps=std::numeric_limits<double>::epsilon();
    const Real span=finite(std::hypot(Real(a[0])-b[0],Real(a[1])-b[1],Real(a[2])-b[2])*(1+64*eps));
    const Real guard=span==0?0:std::nextafter(binary64(span),std::numeric_limits<double>::infinity());
    const Real distance=finite(Real(g.metric)*(g.real_cube_radius+1)-guard);
    if(distance<=0) throw std::invalid_argument("MANTA point-domain omitted real images are not separated from source centers");
    const double d=lower_scalar(distance*(1-64*eps));
    // One normalized Gaussian convolved with a point has beta=alpha; two
    // points have beta=infinity. The physical Ewald eta is unchanged.
    const Real logeta=-.5L*logadd(-2*std::log(Real(g.eta)),log_inverse_exponent);
    const double effective=lower_scalar(std::exp(logeta)*(1-256*eps));
    return real_domain_log(g.real_cube_radius+1,std::log(Real(effective)),std::log(Real(d)),std::log(Real(g.metric)));
}
Real point_reciprocal_log(const Geometry& g) {
    return reciprocal_domain_log(g.reciprocal_cube_radius+1,std::log(Real(g.eta)),std::log(Real(g.reciprocal_metric)),g.volume);
}
PointRemainder point_result(Real real,Real reciprocal) {
    PointRemainder out; out.real=remainder_upper(real); out.reciprocal=remainder_upper(reciprocal);
    out.total=remainder_upper(logadd(real,reciprocal)); return out;
}
}
PointRemainder point_pair(const Geometry& g,const md::Point& a,const md::Point& b) {
    return point_result(point_real_log(g,a,b,minus_inf),point_reciprocal_log(g));
}
PointRemainder gaussian_point(const Geometry& g,const GaussianMajorant& a,const md::Point& b) {
    Real real=minus_inf;
    for(const auto& t:a.terms()) real=logadd(real,point_real_log(g,t.center,b,-std::log(Real(t.exponent)))+std::log(Real(t.charge_bound)));
    return point_result(real,a.charge_bound()?point_reciprocal_log(g)+std::log(Real(a.charge_bound())):minus_inf);
}
} // namespace gaussian_domain_detail
} // namespace vibeqc::manta
