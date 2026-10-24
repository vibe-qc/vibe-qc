#include "manta_ewald_internal.hpp"

namespace vibeqc::manta {
using namespace ewald_detail;
Count periodic_density_batch_workspace(Count n,Count nk,Count pairs,Count reciprocal_batch) {
    // Input snapshots and exported copies, compensated response arrays,
    // simultaneous coefficient/contraction panels, reciprocal rho/sums,
    // bound matrices and per-pair norms, vector headers and receipt strings.
    return add(16384,add(mul(256,mul(nk,square(n))),add(mul(2048,square(n)),
        add(mul(512,reciprocal_batch),mul(128,pairs)))));
}
PeriodicDensityBatchPlan PeriodicEwaldSource::plan_batch(Count count,const ResourceBudget& b) const {
    margin(b);
    if(!count || count>32)throw std::invalid_argument("MANTA density batch requires 1 to 32 densities");
    PeriodicDensityBatchPlan p;p.densities=count;p.source_passes=1;
    p.shell_calls=mul(count,plan_.shell_calls);p.primitive_products=mul(count,plan_.primitive_products);
    const Count setup=mul(count,add(4096,mul(256,mul(plan_.momenta,square(plan_.n_basis)))));
    p.scalar_terms=add(setup,mul(count,plan_.scalar_terms));p.kernel_terms=add(mul(8,setup),mul(count,plan_.kernel_terms));
    p.modeled_peak_bytes=add(plan_.modeled_peak_bytes,add(b.other_live_bytes,add(b.backend_margin_bytes,
        mul(count,periodic_density_batch_workspace(plan_.n_basis,plan_.momenta,plan_.pairs,plan_.reciprocal_batch)))));
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA density batch memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA density batch shell budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA density batch primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA density batch scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA density batch kernel budget exceeded");return p;
}
PeriodicEwaldBatchResult PeriodicEwaldSource::build_batch(const std::vector<std::vector<Complex>>& densities,
    const ResourceBudget& budget) const {
    PeriodicEwaldBatchResult out;out.plan=plan_batch(densities.size(),budget);
    // Validate every input before the first numerical kernel or result allocation.
    for(const auto& d:densities){auto r=receipt_;validate_density(*source_,d,r);}
    const Count count=densities.size(),n=plan_.n_basis,mesh=plan_.momenta*n*n;
    out.results.resize(count);std::vector<std::vector<Complex>> correction(count,std::vector<Complex>(mesh));
    const auto& entries=source_->entries_;const auto& k=source_->momenta_;const auto& weights=source_->weights_;
    PeriodicEwaldReceipt shared=receipt_;
    for(Count b=0;b<count;++b) {
        auto& result=out.results[b];result.plan=plan_;result.receipt=receipt_;result.values.resize(mesh);
        validate_density(*source_,densities[b],result.receipt);
    }
    auto panel=[&](Count b,Count p){++out.results[b].receipt.coefficient_panels;
        return coefficients(entries[p],n,k,weights,densities[b]);};
    std::vector<Real> energy(count),energy_error(count);
    // Same Ewald1921 real quadratic form as scalar build. Stream each immutable
    // kernel across the batch while preserving every density's reduction order.
    for(Count p=0;p<entries.size();++p) {
        std::vector<std::vector<Coefficient>> cp(count);
        for(Count b=0;b<count;++b)cp[b]=panel(b,p);
        for(Count q=p;q<entries.size();++q) {
            std::vector<std::vector<Coefficient>> cq(count);
            std::vector<std::vector<Real>> vp(count),ep(count),vq(count),eq(count);
            for(Count b=0;b<count;++b){cq[b]=panel(b,q);vp[b].resize(cp[b].size());ep[b].resize(cp[b].size());
                vq[b].resize(cq[b].size());eq[b].resize(cq[b].size());}
            for(const auto& index:real_) {
                const auto tile=real_tile(p,q,index,shared);
                for(Count b=0;b<count;++b)for(Count i=0;i<cp[b].size();++i)for(Count j=0;j<cq[b].size();++j) {
                    if(p==q) {
                        accumulate(Real(.5)*cq[b][j].times(tile(i,j)),vp[b][i],ep[b][i]);
                        accumulate(Real(.5)*cp[b][i].times(tile(i,j)),vp[b][j],ep[b][j]);
                    } else {
                        accumulate(cq[b][j].times(tile(i,j)),vp[b][i],ep[b][i]);
                        accumulate(cp[b][i].times(tile(i,j)),vq[b][j],eq[b][j]);
                    }
                }
            }
            for(Count b=0;b<count;++b) {
                for(Count i=0;i<cp[b].size();++i)accumulate((p==q?Real(.5):Real(1))*cp[b][i].times(number(vp[b][i])),energy[b],energy_error[b]);
                scatter(entries[p],n,k,vp[b],out.results[b].values,correction[b]);
                if(p!=q)scatter(entries[q],n,k,vq[b],out.results[b].values,correction[b]);
            }
        }
    }
    for(Count b=0;b<count;++b)out.results[b].receipt.real_energy=number(energy[b]);
    std::fill(energy.begin(),energy.end(),0);std::fill(energy_error.begin(),energy_error.end(),0);
    // Sun2017 Eqs.16-17: the same two streamed passes, with independent rho.
    for(Count start=0;start<reciprocal_.size();start+=plan_.reciprocal_batch) {
        const Count ng=std::min<Count>(plan_.reciprocal_batch,reciprocal_.size()-start);
        std::vector<double> lambda;const auto g=reciprocal_tile(start,ng,lambda);
        std::vector<std::vector<Real>> rr(count,std::vector<Real>(ng)),ri=rr,er=rr,ei=rr;
        for(Count p=0;p<entries.size();++p) {
            const auto f=fourier_panel(p,g,shared);
            for(Count b=0;b<count;++b) {
                const auto cp=panel(b,p);
                for(Count i=0;i<cp.size();++i)for(Count ig=0;ig<ng;++ig) {
                    accumulate(cp[i].times(f(i,ig).real()),rr[b][ig],er[b][ig]);
                    accumulate(cp[i].times(f(i,ig).imag()),ri[b][ig],ei[b][ig]);
                }
            }
        }
        std::vector<std::vector<Complex>> rho(count,std::vector<Complex>(ng));
        for(Count b=0;b<count;++b)for(Count ig=0;ig<ng;++ig) {
            rho[b][ig]={number(rr[b][ig]),number(ri[b][ig])};
            accumulate(Real(.5)*lambda[ig]*(rr[b][ig]*rr[b][ig]+ri[b][ig]*ri[b][ig]),energy[b],energy_error[b]);
        }
        for(Count p=0;p<entries.size();++p) {
            const auto f=fourier_panel(p,g,shared);
            for(Count b=0;b<count;++b) {
                std::vector<Real> v(f.rows()),e(f.rows());
                for(Eigen::Index i=0;i<f.rows();++i)for(Count ig=0;ig<ng;++ig) {
                    const Real x=Real(lambda[ig])*(Real(rho[b][ig].real())*f(i,ig).real()+Real(rho[b][ig].imag())*f(i,ig).imag());
                    accumulate(x,v[i],e[i]);
                }
                scatter(entries[p],n,k,v,out.results[b].values,correction[b]);
            }
        }
    }
    for(Count b=0;b<count;++b)out.results[b].receipt.reciprocal_energy=number(energy[b]);
    std::vector<Real> charge(count),charge_error(count);
    for(Count p=0;p<entries.size();++p) {
        const auto s=overlap_panel(p,shared);
        for(Count b=0;b<count;++b) {
            const auto cp=panel(b,p);
            for(Count i=0;i<cp.size();++i)accumulate(cp[i].times(s.data()[i]),charge[b],charge_error[b]);
        }
    }
    for(Count b=0;b<count;++b) {
        auto& r=out.results[b].receipt;r.charge=number(charge[b]);
        r.gauge_energy=number(Real(.5)*r.gauge_constant*charge[b]*charge[b]);
    }
    for(Count p=0;p<entries.size();++p) {
        const auto s=overlap_panel(p,shared);
        for(Count b=0;b<count;++b) {
            std::vector<Real> v(s.size());
            for(Eigen::Index i=0;i<s.size();++i)v[i]=Real(receipt_.gauge_constant)*charge[b]*s.data()[i];
            scatter(entries[p],n,k,v,out.results[b].values,correction[b]);
        }
    }
    auto& actual=out.receipt;actual.source_identity=source_identity();actual.source_passes=1;
    actual.real_shell_calls=shared.real_shell_calls;actual.fourier_shell_calls=shared.fourier_shell_calls;
    actual.overlap_shell_calls=shared.overlap_shell_calls;actual.primitive_products=shared.primitive_products;
    actual.vector_primitive_products=shared.vector_primitive_products;
    for(Count b=0;b<count;++b) {
        auto& result=out.results[b];auto& r=result.receipt;actual.coefficient_panels+=r.coefficient_panels;
        r.real_shell_calls=shared.real_shell_calls;r.fourier_shell_calls=shared.fourier_shell_calls;
        r.overlap_shell_calls=shared.overlap_shell_calls;r.primitive_products=shared.primitive_products;
        r.vector_primitive_products=shared.vector_primitive_products;
        r.energy=number(Real(r.real_energy)+r.reciprocal_energy+r.gauge_energy);
        Real trace=0,trace_error=0;double scale=0;
        for(Count ik=0;ik<plan_.momenta;++ik)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
            const auto v=result.values[(ik*n+i)*n+j];finite(v);require_finite(std::abs(v));scale=std::max(scale,std::abs(v));
            r.response_hermiticity_residual=std::max(r.response_hermiticity_residual,std::abs(v-std::conj(result.values[(ik*n+j)*n+i])));
            const auto x=densities[b][(ik*n+j)*n+i];
            accumulate(Real(.5)*weights[ik]*(Real(x.real())*v.real()-Real(x.imag())*v.imag()),trace,trace_error);
        }
        r.trace_energy=number(trace);require_finite(r.response_hermiticity_residual);
        if(r.response_hermiticity_residual>1e-11*std::max(1.,scale))throw std::runtime_error("MANTA periodic Ewald response violates Hermiticity");
    }
    return out;
}
} // namespace vibeqc::manta
