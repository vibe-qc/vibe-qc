#include "vibeqc/manta_lattice_tail.hpp"
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
struct Hash {
    vibeqc::detail::Sha256 value;
    void word(std::uint64_t x) {
        std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(x>>(56-8*k));
        value.update(bytes.data(),bytes.size());
    }
    void number(double x) {
        if(x==0.) x=0.;
        std::uint64_t bits; std::memcpy(&bits,&x,sizeof(bits)); word(bits);
    }
    void text(const std::string& s) {
        word(s.size()); value.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
    }
};
std::vector<LatticeIndex> domain(const std::vector<LatticeIndex>& input,bool origin) {
    auto result=input;
    for(const auto& n:result) for(auto x:n)
        if(x < -2147483647LL || x > 2147483647LL)
            throw std::invalid_argument("MANTA lattice-tail indices exceed the signed 31-bit range");
    std::sort(result.begin(),result.end());
    if(std::adjacent_find(result.begin(),result.end())!=result.end())
        throw std::invalid_argument("MANTA lattice-tail domains must contain unique indices");
    if(std::binary_search(result.begin(),result.end(),LatticeIndex{})!=origin)
        throw std::invalid_argument("MANTA lattice-tail real/near domains require origin; reciprocal domain excludes zero");
    for(const auto& n:result)
        if(!std::binary_search(result.begin(),result.end(),LatticeIndex{-n[0],-n[1],-n[2]}))
            throw std::invalid_argument("MANTA lattice-tail domains must be inversion closed");
    return result;
}
std::string domain_identity(const char* kind,const std::vector<LatticeIndex>& indices) {
    Hash h; h.text("manta.lattice-tail.domain.v1"); h.text(kind); h.word(indices.size());
    for(const auto& n:indices) for(auto x:n) h.word(static_cast<std::uint64_t>(x));
    return h.value.finish_hex();
}
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

LatticeTailPlan plan_lattice_tail(int order,Count nr,Count ng,Count nn,const ResourceBudget& budget) {
    using namespace admission;
    if(order<0 || order>maximum_order) throw std::invalid_argument("MANTA lattice-tail order must be zero through twelve");
    if(!nr || !nn || nn>nr) throw std::invalid_argument("MANTA lattice-tail requires nonempty real and near domains with near contained in real");
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

LatticeTailMatrix multipole_lattice_tail(const Eigen::Matrix3d& cell,const md::Point& displacement,
    double eta,const std::vector<LatticeIndex>& real_input,const std::vector<LatticeIndex>& reciprocal_input,
    const std::vector<LatticeIndex>& near_input,int order,const ResourceBudget& budget) {
    LatticeTailMatrix result;
    result.plan=plan_lattice_tail(order,real_input.size(),reciprocal_input.size(),near_input.size(),budget);
    const double volume=std::abs(cell.determinant());
    if(!cell.allFinite() || !std::isfinite(volume) || volume<=0 || !cell.inverse().allFinite())
        throw std::invalid_argument("MANTA lattice-tail requires a finite nonsingular 3D cell");
    if(!std::isfinite(eta) || eta<=0) throw std::invalid_argument("MANTA lattice-tail eta must be finite and positive");
    for(auto x:displacement) admission::require_finite(x);
    const auto real=domain(real_input,true),reciprocal=domain(reciprocal_input,false),near=domain(near_input,true);
    if(!std::includes(real.begin(),real.end(),near.begin(),near.end()))
        throw std::invalid_argument("MANTA lattice-tail near domain must be contained in real domain");
    result.eta=eta; result.cell_volume=volume;
    result.gauge_constant=binary64(-std::exp(std::log(pi)-2*std::log(Real(eta))-std::log(Real(volume))));
    result.real_domain_identity=domain_identity("real",real);
    result.reciprocal_domain_identity=domain_identity("reciprocal",reciprocal);
    result.near_domain_identity=domain_identity("near",near);
    Hash kernel; kernel.text("manta.lattice-tail.v1;finite-ewald-minus-near;columns;raw-cartesian;total-degree;binary64");
    kernel.text(result.boundary); kernel.number(eta);
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) kernel.number(cell(i,j));
    kernel.text(result.real_domain_identity); kernel.text(result.reciprocal_domain_identity); kernel.text(result.near_domain_identity);
    result.kernel_identity=kernel.value.finish_hex();
    Hash source; source.text(result.kernel_identity); source.word(order);
    for(auto x:displacement) source.number(x);
    result.source_identity=source.value.finish_hex();

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
    result.indices=t.indices;
    result.values=Eigen::MatrixXd::Zero(t.indices.size(),t.indices.size());
    for(std::size_t i=0;i<t.indices.size();++i) for(std::size_t j=0;j<t.indices.size();++j) {
        const auto& a=t.indices[i]; const auto& b=t.indices[j];
        const md::Index total{a[0]+b[0],a[1]+b[1],a[2]+b[2]};
        if(degree(total)>order) continue;
        Real factor=degree(b)%2?-1:1;
        for(int axis=0;axis<3;++axis) factor*=t.factorial[total[axis]]/(t.factorial[a[axis]]*t.factorial[b[axis]]);
        result.values(i,j)=binary64(factor*t.coefficients[t.lookup[t.offset(total)]]);
        ++result.matrix_coefficients_evaluated;
    }
    return result;
}
} // namespace vibeqc::manta
