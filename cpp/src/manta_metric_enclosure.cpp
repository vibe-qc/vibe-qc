#include "manta_metric_enclosure.hpp"
#include "manta_radial_enclosure.hpp"
#include "manta_verified_math.hpp"
#include "manta_positive_radial.hpp"
#include "manta_representation_enclosure.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/manta_shells.hpp"
#include <functional>

namespace vibeqc::manta {
namespace {
namespace si=spectral_interval;namespace vm=verified_math;
using I=si::Interval;
I point(double v) {return {v,v};}
I positive(I a) {a.lo=std::max(0.,a.lo);if(a.hi<a.lo)throw std::logic_error("MANTA radial enclosure lost positivity");return a;}
I power(I x,int n) {I out{1.,1.};for(int j=0;j<n;++j)out=si::product(out,x);return out;}
// F_n(T)=exp(-T)/(2n+1) sum_k (2T)^k/product_j(2n+2j+1).
// Positive terms; a decreasing future ratio bounds the omitted geometric tail.
// At very large T use the rigorous [0,infinity) gamma integral upper bound.
// Rump2010; McMurchieDavidson1978; Neese2023 Eq.54; contract 5.63.
std::vector<I> boys_enclosure(int n,I t) {
    t=positive(t);std::vector<I> out(n+1);
    if(t.hi>512.) {
        if(t.lo<=0.) {for(int j=0;j<=n;++j)out[j]={0.,vm::divide(point(1.),point(2*j+1)).hi};return out;}
        auto gamma=vm::divide(vm::sqrt_positive(vm::pi()),si::product(point(2.),vm::sqrt_positive(t)));
        for(int j=0;j<=n;++j) {
            out[j]={0.,std::min(vm::divide(point(1.),point(2*j+1)).hi,gamma.hi)};
            if(j<n)gamma=vm::divide(si::product(gamma,point(2*j+1)),si::product(point(2.),t));
        }
        return out;
    }
    const auto decay=vm::exp_negative(t);I term{1.,1.},sum=term;bool finished=false;
    for(int k=0;k<2048;++k) {
        const auto next=vm::divide(si::product(term,si::product(point(2.),t)),point(2*n+2*k+3));
        const auto ratio=vm::divide(si::product(point(2.),t),point(2*n+2*k+5));
        if(ratio.hi<1.) {
            const auto tail=vm::divide(next,si::sub(point(1.),ratio));
            if(tail.hi<=sum.hi*1e-18) {sum=si::add(sum,{0.,tail.hi});finished=true;break;}
        }
        sum=si::add(sum,next);term=next;
    }
    if(!finished)throw std::runtime_error("MANTA radial enclosure work limit");
    out[n]=positive(vm::divide(si::product(decay,sum),point(2*n+1)));
    for(int j=n;j>0;--j)out[j-1]=positive(vm::divide(si::add(si::product(si::product(point(2.),t),out[j]),decay),point(2*j-1)));
    return out;
}
std::vector<I> radial(int n,I t,I rho,md::RadialKernel kernel,double omega) {
    if(kernel==md::RadialKernel::Erfc && omega>0. && std::isfinite(omega))return positive_erfc_enclosure(n,t,rho,omega);
    auto full=boys_enclosure(n,t);
    if(kernel==md::RadialKernel::Coulomb || omega==0. || std::isinf(omega)) {
        if((omega==0. && kernel==md::RadialKernel::Erf) || (std::isinf(omega) && kernel==md::RadialKernel::Erfc))return std::vector<I>(n+1);
        return full;
    }
    const auto w2=si::product(omega,omega);
    auto kappa=positive(vm::divide(w2,si::add(w2,rho)));kappa.hi=std::min(1.,kappa.hi);
    auto factor=vm::sqrt_positive(kappa);auto screened=boys_enclosure(n,si::product(kappa,t));
    for(int j=0;j<=n;++j) {
        const auto long_range=si::product(factor,screened[j]);
        full[j]=long_range;
        factor=si::product(factor,kappa);
    }
    return full;
}
struct Term {md::Index index;I value;};
std::vector<std::vector<Term>> expansion(int l,double a) {
    std::vector<std::vector<Term>> out;
    // Exact single-center x^l Gaussian Hermite coefficients, with parity.
    // E_(l,t)=l!/[((l-t)/2)! t! 2^l a^((l+t)/2)].
    double factorial[15];factorial[0]=1.;for(int i=1;i<15;++i)factorial[i]=factorial[i-1]*i;
    const auto axis=[&](int degree,int t) {
        auto denominator=si::product(point(factorial[(degree-t)/2]),point(factorial[t]));
        denominator=si::product(denominator,point(std::ldexp(1.,degree)));
        denominator=si::product(denominator,power(point(a),(degree+t)/2));
        return vm::divide(point(factorial[degree]),denominator);
    };
    for(const auto& p:md::cartesians(l)) {
        std::vector<Term> terms;
        for(int x=p[0]%2;x<=p[0];x+=2)for(int y=p[1]%2;y<=p[1];y+=2)for(int z=p[2]%2;z<=p[2];z+=2)
            terms.push_back({{x,y,z},si::product(si::product(axis(p[0],x),axis(p[1],y)),axis(p[2],z))});
        out.push_back(std::move(terms));
    }
    return out;
}
}
std::vector<spectral_interval::Interval> radial_kernel_enclosure(int n,spectral_interval::Interval t,
    spectral_interval::Interval rho,md::RadialKernel kernel,double omega) {
    return radial(n,t,rho,kernel,omega);
}
std::vector<I> coulomb_boys_enclosure(int n,I t) {
    if(n<0 || n>md::maximum_order || !std::isfinite(t.lo) || !std::isfinite(t.hi) || t.lo<0. || t.lo>t.hi)
        throw std::invalid_argument("MANTA invalid Coulomb radial enclosure input");
    return boys_enclosure(n,t);
}

MetricConstructionPlan plan_metric_construction(const BasisSet& bra,const BasisSet& ket,const KernelRequest& request,const ResourceBudget& budget) {
    using namespace admission;margin(budget);const auto f=plan_fitting({&bra,&ket},request,ResourceBudget{});
    const auto a=census(bra,7),b=census(ket,7);const Count degree=a.maximum_l+b.maximum_l;
    const Count rec=fourth(degree+1),cart=mul(a.maximum_cartesian,b.maximum_cartesian);
    // Each Cartesian polynomial has <=64 nonzero Hermite terms through l7.
    // At most nine panels and 64 positive-series terms for each radial order;
    // two Bernstein linear products, scaled coefficients and interval means.
    const Count radial_work=mul(64*9*64,mul(2*degree+131,degree+1));
    const Count work=add(add(1048576,radial_work),add(mul(64,rec),mul(65536,cart)));
    MetricConstructionPlan p;p.shell_calls=f.shell_calls;p.primitive_products=f.primitive_products;
    p.scalar_terms=add(mul(f.primitive_products,work),mul(f.shell_calls,mul(256,mul(cart,add(a.maximum_cartesian,b.maximum_cartesian)))));
    p.scalar_terms=add(p.scalar_terms,mul(f.shell_calls,2097152));
    p.kernel_terms=add(f.kernel_terms,mul(8,p.scalar_terms));
    p.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),add(f.modeled_peak_bytes,
        add(1048576,add(mul(64,rec),add(mul(256,cart),mul(128,f.integral_elements))))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA metric construction memory budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA metric construction scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA metric construction work budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA metric construction shell budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA metric construction primitive budget exceeded");return p;
}
std::vector<I> metric_shell_enclosure(const libint2::Shell& a,const libint2::Shell& b,md::RadialKernel kernel,double omega,const Eigen::Vector3d& translation) {
    si::require_arithmetic();const int la=a.contr[0].l,lb=b.contr[0].l,l=la+lb,width=l+1;
    const auto ca=md::cartesians(la).size(),cb=md::cartesians(lb).size();std::vector<I> contracted(ca*cb);
    std::array<I,3> displacement;I squared;
    for(int axis=0;axis<3;++axis) {
        displacement[axis]=si::sub(point(a.O[axis]),si::add(point(b.O[axis]),point(translation[axis])));
        squared=si::add(squared,si::product(displacement[axis],displacement[axis]));
    }
    squared=positive(squared);const auto pi=vm::pi();const auto prefactor=si::product(point(2.),si::product(si::product(pi,pi),vm::sqrt_positive(pi)));
    for(Count ia=0;ia<a.alpha.size();++ia)for(Count ib=0;ib<b.alpha.size();++ib) {
        const double pa=a.alpha[ia],pb=b.alpha[ib];const auto sum=si::add(point(pa),point(pb));
        const auto rho=vm::divide(si::product(pa,pb),sum);
        const auto scale=vm::divide(prefactor,si::product(si::product(pa,pb),vm::sqrt_positive(sum)));
        auto seeds=radial(l,si::product(rho,squared),rho,kernel,omega);I factor{1.,1.};
        for(int n=0;n<=l;++n) {seeds[n]=si::product(seeds[n],factor);if(n<l)factor=si::product(factor,si::product(point(-2.),rho));}
        std::vector<I> cache(admission::fourth(width));std::vector<bool> known(cache.size());
        std::function<I(int,int,int,int)> get=[&](int x,int y,int z,int n)->I {
            if(!x && !y && !z)return seeds[n];
            const Count index=((x*width+y)*width+z)*width+n;if(known[index])return cache[index];
            const int axis=x?0:y?1:2,degree=axis==0?x:axis==1?y:z;
            int v[3]={x,y,z};--v[axis];I value=si::product(displacement[axis],get(v[0],v[1],v[2],n+1));
            if(degree>1) {--v[axis];value=si::add(value,si::product(point(degree-1),get(v[0],v[1],v[2],n+1)));}
            known[index]=true;return cache[index]=value;
        };
        const auto ea=expansion(la,pa),eb=expansion(lb,pb);
        const auto weight=si::product(scale,si::product(a.contr[0].coeff[ia],b.contr[0].coeff[ib]));
        for(Count i=0;i<ca;++i)for(Count j=0;j<cb;++j) {
            I value;
            for(const auto& t:ea[i])for(const auto& u:eb[j]) {
                auto term=si::product(si::product(t.value,u.value),get(t.index[0]+u.index[0],t.index[1]+u.index[1],t.index[2]+u.index[2],0));
                if((u.index[0]+u.index[1]+u.index[2])%2)term={-term.hi,-term.lo};value=si::add(value,term);
            }
            contracted[i*cb+j]=si::add(contracted[i*cb+j],si::product(weight,value));
        }
    }
    const auto ta=representation_enclosure(la,a.contr[0].pure),tb=representation_enclosure(lb,b.contr[0].pure);
    std::vector<I> intermediate(a.size()*cb),out(a.size()*b.size());
    for(Count i=0;i<a.size();++i)for(Count j=0;j<cb;++j)for(Count k=0;k<ca;++k)
        intermediate[i*cb+j]=si::add(intermediate[i*cb+j],si::product(ta[i*ca+k],contracted[k*cb+j]));
    for(Count i=0;i<a.size();++i)for(Count j=0;j<b.size();++j)for(Count k=0;k<cb;++k)
        out[i*b.size()+j]=si::add(out[i*b.size()+j],si::product(intermediate[i*cb+k],tb[j*cb+k]));
    return out;
}
MetricConstructionCertificate certify_metric_construction(const BasisSet& bra,const BasisSet& ket,const KernelRequest& request,const ResourceBudget& budget) {
    const auto plan=plan_metric_construction(bra,ket,request,budget);const FittingTiles source({&bra,&ket},request,ResourceBudget{});
    MetricConstructionCertificate out;out.evaluated=md::Matrix::Zero(bra.nbasis(),ket.nbasis());out.lower=out.upper=out.error_bounds=out.evaluated;
    const auto a=bra.libint().shell2bf(),b=ket.libint().shell2bf();
    for(Count i=0;i<bra.nshells();++i)for(Count j=0;j<ket.nshells();++j) {
        const auto numerical=source.evaluate({i,j});const auto proof=metric_shell_enclosure(bra.libint()[i],ket.libint()[j],radial_kernel(request),request.omega);
        for(Count x=0;x<numerical.shape[0];++x)for(Count y=0;y<numerical.shape[1];++y) {
            const auto z=proof[x*numerical.shape[1]+y];const double value=numerical.values(x,y);const auto error=si::sub(point(value),z);
            out.evaluated(a[i]+x,b[j]+y)=value;out.lower(a[i]+x,b[j]+y)=z.lo;out.upper(a[i]+x,b[j]+y)=z.hi;
            out.error_bounds(a[i]+x,b[j]+y)=std::max(std::abs(error.lo),std::abs(error.hi));
        }
    }
    return out;
}
} // namespace vibeqc::manta
