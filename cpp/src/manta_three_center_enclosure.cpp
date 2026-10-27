#include "manta_three_center_enclosure.hpp"
#include "manta_radial_enclosure.hpp"
#include "manta_verified_math.hpp"
#include "manta_representation_enclosure.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_radial.hpp"
#include <functional>

namespace vibeqc::manta {
namespace {
namespace si=spectral_interval;namespace vm=verified_math;using I=si::Interval;
I point(double x) {return {x,x};}
I positive(I x) {x.lo=std::max(0.,x.lo);return x;}
bool zero(I x) {return x.lo==0. && x.hi==0.;}
using Term=std::pair<Count,I>;
// Gaussian product Hermite raising identity, independently enclosed at every
// operation. Neese2023 / McMurchieDavidson1978; contract 5.67. The common
// exp(-a*b/(a+b)*|A-B|^2) is included once in the primitive prefactor.
std::vector<std::vector<Term>> expansion(int la,int lb,I p,
    const std::array<I,3>& pa,const std::array<I,3>& pb) {
    const int width=la+lb+2;const auto half=vm::divide(point(1.),si::product(point(2.),p));
    std::array<std::vector<I>,3> axes;
    const auto index=[&](int i,int j,int t) {return (i*(lb+1)+j)*width+t;};
    for(int axis=0;axis<3;++axis) {
        auto& e=axes[axis];e.resize((la+1)*(lb+1)*width);e[0]=point(1.);
        for(int i=0;i<=la;++i)for(int j=0;j<=lb;++j) {
            if(!i && !j)continue;const int a=i?i-1:i,b=i?j:j-1;const auto shift=i?pa[axis]:pb[axis];
            for(int t=0;t<=i+j;++t) {
                I v=si::product(shift,e[index(a,b,t)]);
                if(t)v=si::add(v,si::product(half,e[index(a,b,t-1)]));
                v=si::add(v,si::product(point(t+1),e[index(a,b,t+1)]));e[index(i,j,t)]=v;
            }
        }
    }
    const auto& ca=md::cartesians(la);const auto& cb=md::cartesians(lb);const auto& h=md::hermites(la+lb);
    std::vector<std::vector<Term>> out(ca.size()*cb.size());
    for(Count i=0;i<ca.size();++i)for(Count j=0;j<cb.size();++j)for(Count u=0;u<h.size();++u) {
        I v=point(1.);
        for(int axis=0;axis<3;++axis) {
            if(h[u][axis]>ca[i][axis]+cb[j][axis]) {v={};break;}
            v=si::product(v,axes[axis][index(ca[i][axis],cb[j][axis],h[u][axis])]);
        }
        if(!zero(v))out[i*cb.size()+j].push_back({u,v});
    }
    return out;
}
}
MetricConstructionPlan plan_three_center_construction(const BasisSet& aux,const BasisSet& bra,const BasisSet& ket,
    const KernelRequest& request,const ResourceBudget& budget) {
    using namespace admission;margin(budget);const auto f=plan_fitting({&aux,&bra,&ket},request,ResourceBudget{});
    const auto a=census(aux,7),b=census(bra,6),c=census(ket,6);
    const Count la=a.maximum_l,lb=b.maximum_l+c.maximum_l,l=la+lb;
    const Count ha=(la+1)*(la+2)*(la+3)/6,hb=(lb+1)*(lb+2)*(lb+3)/6;
    const Count pair=mul(b.maximum_cartesian,c.maximum_cartesian),cart=mul(a.maximum_cartesian,pair),rec=fourth(l+1);
    const Count radial=mul(64*9*64,mul(2*l+131,l+1));
    const Count contraction=mul(128,add(mul(mul(ha,hb),pair),mul(cart,ha)));
    const Count transforms=mul(256,mul(cart,add(a.maximum_cartesian,add(b.maximum_cartesian,c.maximum_cartesian))));
    MetricConstructionPlan p;p.shell_calls=f.shell_calls;p.primitive_products=f.primitive_products;
    p.scalar_terms=add(f.integral_elements,add(mul(f.primitive_products,add(2097152,add(radial,add(mul(128,rec),contraction)))),
        mul(f.shell_calls,add(3145728,transforms))));
    p.kernel_terms=add(f.kernel_terms,mul(8,p.scalar_terms));
    // Per shell tile only: numerical tile, four certificate arrays, sparse
    // E tables, derivative cache, Hermite kernel and successive transforms.
    p.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),add(f.modeled_peak_bytes,
        add(2097152,add(mul(256,cart),mul(128,add(rec,add(mul(ha,hb),add(mul(pair,add(ha,hb)),mul(a.maximum_cartesian,ha)))))))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA three-center construction memory budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA three-center construction scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA three-center construction work budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA three-center construction shell budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA three-center construction primitive budget exceeded");return p;
}
std::vector<I> three_center_shell_enclosure(const libint2::Shell& a,const libint2::Shell& b,const libint2::Shell& c,
    md::RadialKernel kernel,double omega) {
    return three_center_shell_enclosure(a,b,c,kernel,omega,enclosed_center(a),enclosed_center(b),enclosed_center(c));
}
std::vector<I> three_center_shell_enclosure(const libint2::Shell& a,const libint2::Shell& b,const libint2::Shell& c,
    md::RadialKernel kernel,double omega,const EnclosedPoint& ac,const EnclosedPoint& bc,const EnclosedPoint& cc) {
    si::require_arithmetic();const int la=a.contr[0].l,lb=b.contr[0].l,lc=c.contr[0].l,l=la+lb+lc,width=l+1;
    const auto na=md::cartesians(la).size(),nb=md::cartesians(lb).size(),nc=md::cartesians(lc).size(),np=nb*nc;
    const auto& ha=md::hermites(la);const auto& hb=md::hermites(lb+lc);
    std::vector<I> contracted(na*np);const auto pi=vm::pi();
    const auto prefactor=si::product(point(2.),si::product(si::product(pi,pi),vm::sqrt_positive(pi)));
    std::array<I,3> pair_delta;I pair_squared;
    for(int axis=0;axis<3;++axis) {
        pair_delta[axis]=si::sub(cc[axis],bc[axis]);
        pair_squared=si::add(pair_squared,si::product(pair_delta[axis],pair_delta[axis]));
    }
    pair_squared=positive(pair_squared);
    for(Count ia=0;ia<a.alpha.size();++ia)for(Count ib=0;ib<b.alpha.size();++ib)for(Count ic=0;ic<c.alpha.size();++ic) {
        const auto alpha=point(a.alpha[ia]),beta=point(b.alpha[ib]),gamma=point(c.alpha[ic]);
        const auto p=si::add(beta,gamma),fraction=vm::divide(gamma,p),sum=si::add(alpha,p);
        const auto rho=vm::divide(si::product(alpha,p),sum);
        std::array<I,3> pb,pc,delta;I squared;
        for(int axis=0;axis<3;++axis) {
            pb[axis]=si::product(fraction,pair_delta[axis]);pc[axis]=si::sub(pb[axis],pair_delta[axis]);
            delta[axis]=si::sub(ac[axis],si::add(bc[axis],pb[axis]));
            squared=si::add(squared,si::product(delta[axis],delta[axis]));
        }
        const auto decay=vm::exp_negative(positive(si::product(si::product(beta,fraction),pair_squared)));
        const auto scale=si::product(decay,vm::divide(prefactor,si::product(si::product(alpha,p),vm::sqrt_positive(sum))));
        const auto weight=si::product(scale,si::product(si::product(a.contr[0].coeff[ia],b.contr[0].coeff[ib]),point(c.contr[0].coeff[ic])));
        auto seeds=radial_kernel_enclosure(l,positive(si::product(rho,positive(squared))),rho,kernel,omega);I factor=point(1.);
        for(int n=0;n<=l;++n) {seeds[n]=si::product(seeds[n],factor);if(n<l)factor=si::product(factor,si::product(point(-2.),rho));}
        std::vector<I> cache(admission::fourth(width));std::vector<bool> known(cache.size());
        std::function<I(int,int,int,int)> get=[&](int x,int y,int z,int n)->I {
            if(!x && !y && !z)return seeds[n];
            const Count idx=((x*width+y)*width+z)*width+n;if(known[idx])return cache[idx];
            const int axis=x?0:y?1:2,degree=axis==0?x:axis==1?y:z;int v[3]={x,y,z};--v[axis];
            I value=si::product(delta[axis],get(v[0],v[1],v[2],n+1));
            if(degree>1) {--v[axis];value=si::add(value,si::product(point(degree-1),get(v[0],v[1],v[2],n+1)));}
            known[idx]=true;return cache[idx]=value;
        };
        const auto ea=expansion(la,0,alpha,{},{}),eb=expansion(lb,lc,p,pb,pc);
        std::vector<I> r(ha.size()*hb.size());
        for(Count u=0;u<ha.size();++u)for(Count v=0;v<hb.size();++v) {
            auto z=get(ha[u][0]+hb[v][0],ha[u][1]+hb[v][1],ha[u][2]+hb[v][2],0);
            if((hb[v][0]+hb[v][1]+hb[v][2])%2)z={-z.hi,-z.lo};r[u*hb.size()+v]=z;
        }
        std::vector<I> intermediate(ha.size()*np);
        for(Count u=0;u<ha.size();++u)for(Count j=0;j<np;++j)for(const auto& v:eb[j])
            intermediate[u*np+j]=si::add(intermediate[u*np+j],si::product(r[u*hb.size()+v.first],v.second));
        for(Count i=0;i<na;++i)for(Count j=0;j<np;++j) {
            I z;for(const auto& u:ea[i])z=si::add(z,si::product(u.second,intermediate[u.first*np+j]));
            contracted[i*np+j]=si::add(contracted[i*np+j],si::product(weight,z));
        }
    }
    const auto ta=representation_enclosure(la,a.contr[0].pure),tb=representation_enclosure(lb,b.contr[0].pure),tc=representation_enclosure(lc,c.contr[0].pure);
    std::vector<I> first(a.size()*np),second(a.size()*b.size()*nc),out(a.size()*b.size()*c.size());
    for(Count i=0;i<a.size();++i)for(Count j=0;j<np;++j)for(Count k=0;k<na;++k)
        first[i*np+j]=si::add(first[i*np+j],si::product(ta[i*na+k],contracted[k*np+j]));
    for(Count i=0;i<a.size();++i)for(Count j=0;j<b.size();++j)for(Count k=0;k<nc;++k)for(Count v=0;v<nb;++v)
        second[(i*b.size()+j)*nc+k]=si::add(second[(i*b.size()+j)*nc+k],si::product(tb[j*nb+v],first[(i*nb+v)*nc+k]));
    for(Count i=0;i<a.size()*b.size();++i)for(Count j=0;j<c.size();++j)for(Count k=0;k<nc;++k)
        out[i*c.size()+j]=si::add(out[i*c.size()+j],si::product(tc[j*nc+k],second[i*nc+k]));
    return out;
}
ThreeCenterConstructionTiles::ThreeCenterConstructionTiles(const BasisSet& a,const BasisSet& b,const BasisSet& c,
    const KernelRequest& request,const ResourceBudget& budget)
    :plan_(plan_three_center_construction(a,b,c,request,budget)),source_({&a,&b,&c},request,ResourceBudget{}) {}
ThreeCenterConstructionTile ThreeCenterConstructionTiles::evaluate(const std::vector<std::size_t>& indices) const {
    ThreeCenterConstructionTile out;out.evaluated=source_.evaluate(indices); // Validate arity/indices first.
    const auto& request=source_.kernel_request();
    const auto proof=three_center_shell_enclosure(source_.basis(0).libint()[indices[0]],source_.basis(1).libint()[indices[1]],
        source_.basis(2).libint()[indices[2]],radial_kernel(request),request.omega);
    out.lower.resize(proof.size());out.upper.resize(proof.size());out.error_bounds.resize(proof.size());
    for(Count i=0;i<proof.size();++i) {
        const auto z=proof[i],error=si::sub(point(out.evaluated.values.data()[i]),z);
        out.lower[i]=z.lo;out.upper[i]=z.hi;out.error_bounds[i]=std::max(std::abs(error.lo),std::abs(error.hi));
    }
    return out;
}
} // namespace vibeqc::manta
