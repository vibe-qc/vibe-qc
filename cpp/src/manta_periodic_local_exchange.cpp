#include "manta_periodic_xc_source.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/ao_eval_subset.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <Eigen/LU>
#include <cstring>

namespace vibeqc::manta {
using namespace periodic_xc_detail;

PeriodicLocalExchangePlan plan_periodic_local_exchange(const BasisSet& basis,Count ni,Count nk,Count ng,
    Count ns,Count batch,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget); const auto c=census(basis,4);
    if(!ni || !nk || !ng || !batch || (ns!=1 && ns!=2))
        throw std::invalid_argument("MANTA local exchange requires nonempty images/k/grid/batch and one or two spins");
    cap(basis.nshells(),std::numeric_limits<int>::max(),"MANTA local exchange shell index overflow");
    PeriodicLocalExchangePlan p;
    p.n_basis=basis.nbasis(); p.spins=ns; p.images=ni; p.momenta=nk; p.grid_points=ng; p.grid_batch=std::min(ng,batch);
    const Count n=p.n_basis,b=p.grid_batch,e=mul(mul(ns,nk),square(n));
    for(Count count:{ni,nk,ng})
        cap(count,std::numeric_limits<Eigen::Index>::max()/3,"MANTA local exchange source is not addressable");
    cap(mul(b,c.maximum_size),std::numeric_limits<Eigen::Index>::max(),"MANTA local exchange AO panel is not addressable");
    // Caller arrays plus binding copies; the basis is borrowed. Include one
    // exported V owner, compensated V accumulation, shell/gauge metadata,
    // bounded complex AO panel and serial AO evaluator scratch. Backend
    // allowance covers allocator/runtime overhead; this is not a RSS bound.
    p.input_bytes=mul(2,add(add(72,mul(24,ni)),add(mul(32,nk),add(mul(32,ng),mul(16,e)))));
    p.output_bytes=mul(48,e);
    p.panel_bytes=add(mul(16,mul(mul(b,nk),n)),add(mul(8,mul(b,c.maximum_size)),mul(64,b)));
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(budget.backend_margin_bytes,
        add(c.bytes,add(p.input_bytes,add(p.output_bytes,add(p.panel_bytes,add(mul(128,basis.nshells()),65536)))))));
    const Count panels=add(ng/b,ng%b!=0);
    p.shell_calls=mul(mul(panels,ni),basis.nshells());
    p.primitive_points=mul(mul(ng,ni),c.primitives);
    const Count contractions=mul(mul(mul(ng,nk),ns),square(n));
    p.scalar_terms=add(mul(128,contractions),add(mul(128,mul(mul(mul(ng,ni),nk),n)),mul(64,square(ni))));
    p.kernel_terms=add(p.scalar_terms,mul(4096,mul(p.primitive_points,add(c.maximum_cartesian,1))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA local exchange memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA local exchange shell-call budget exceeded");
    cap(p.primitive_points,budget.primitive_quartets,"MANTA local exchange primitive-point budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA local exchange scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA local exchange kernel-work budget exceeded");
    return p;
}
PeriodicLocalExchangeResult compute_periodic_local_exchange(const BasisSet& basis,const Eigen::MatrixXd& lattice,
    const Eigen::MatrixXd& images,const Eigen::MatrixXd& k,const std::vector<double>& weights,
    const Eigen::MatrixXd& points,const std::vector<double>& qw,const std::vector<Complex>& d,
    Count ns,Count batch,const ResourceBudget& budget) {
    using namespace admission;
    PeriodicLocalExchangeResult result;
    result.plan=plan_periodic_local_exchange(basis,images.rows(),k.rows(),points.rows(),ns,batch,budget);
    const auto& p=result.plan; auto& r=result.receipt;
    const Count n=p.n_basis,nk=p.momenta,ng=p.grid_points,e=mul(mul(ns,nk),square(n));
    const auto shifts=periodic_xc_detail::initialize(basis,lattice,images,k,weights,points,qw,d,ns,p,r,
        "manta.periodic-local-exchange.source.v1");
    r.minimum_density=std::numeric_limits<double>::infinity();
    result.values.assign(e,0.); std::vector<Complex> errors(e,0.);
    std::vector<Sum> electrons(ns); Sum energy;
    // Dirac, Proc. Cambridge Philos. Soc. 26, 376 (1930),
    // doi:10.1017/S0305004100016108. Spin scaling: Ex[a,b]=(Ex[2a]+Ex[2b])/2.
    const double coefficient=.75*std::cbrt((ns==2 ? 6. : 3.)/std::acos(-1.));
    for(Count start=0;start<ng;start+=p.grid_batch) {
        const Count b=std::min(p.grid_batch,ng-start);
        const auto panel=periodic_xc_detail::evaluate_panel(basis,images,k,points,shifts,start,b,false,r);
        const auto& bloch=panel.values;
        for(Count g=0;g<b;++g) for(Count spin=0;spin<ns;++spin) {
            Sum rho,absolute;
            for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                const auto term=weights[ik]*bloch[(g*nk+ik)*n+i]*d[((spin*nk+ik)*n+i)*n+j]*std::conj(bloch[(g*nk+ik)*n+j]);
                finite(term); rho.add(term.real()); absolute.add(std::abs(term));
            }
            r.minimum_density=std::min(r.minimum_density,rho.value);
            if(rho.value < -1e-12*std::max(1.,absolute.value))
                throw std::invalid_argument("MANTA local exchange requires nonnegative grid spin density");
            if(rho.value<0) ++r.rounded_negative_points;
            const double density_value=std::max(0.,rho.value),root=std::cbrt(density_value),w=qw[start+g];
            electrons[spin].add(w*rho.value); energy.add(-w*coefficient*density_value*root);
            const double potential=-w*(4./3.)*coefficient*root;
            for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                const Count index=((spin*nk+ik)*n+i)*n+j;
                const Complex term=potential*std::conj(bloch[(g*nk+ik)*n+i])*bloch[(g*nk+ik)*n+j]; finite(term);
                const auto y=term-errors[index],next=result.values[index]+y;
                errors[index]=(next-result.values[index])-y; result.values[index]=next; finite(next);
            }
        }
        ++r.panel_calls;
    }
    r.energy=energy.value; for(auto x:electrons) r.electrons_grid.push_back(x.value);
    r.modeled_peak_bytes=p.modeled_peak_bytes;
    Digest snapshot; snapshot.text(r.density_identity); snapshot.real(r.energy);
    for(double x:r.electrons_grid) snapshot.real(x); for(auto x:result.values) snapshot.value(x);
    r.snapshot_identity=snapshot.hash.finish_hex(); return result;
}
} // namespace vibeqc::manta
