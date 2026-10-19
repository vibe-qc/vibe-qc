#include "vibeqc/manta_periodic_ewald.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_fourier.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_shells.hpp"
#include "manta_lattice_geometry.hpp"
#include "manta_ewald_internal.hpp"

namespace vibeqc::manta {
using namespace ewald_detail;

PeriodicEwaldPlan PeriodicEwaldSource::plan_source(const PeriodicPairSource& source,
    Count nr,Count ng,Count batch,const ResourceBudget& budget) {
    margin(budget);
    if(!nr || !(nr%2) || ng%2 || !batch)
        throw std::invalid_argument("MANTA periodic Ewald needs odd positive real count, even reciprocal count and positive batch");
    PeriodicEwaldPlan p; p.n_basis=source.plan_.n_basis; p.momenta=source.plan_.momenta;
    p.pairs=source.entries_.size(); p.real_cells=nr; p.reciprocal_vectors=ng;
    p.reciprocal_batch=std::min(batch,ng); p.reciprocal_batches=ng?1+(ng-1)/p.reciprocal_batch:0;
    p.pair_interactions=mul(p.pairs,add(p.pairs,1))/2;
    p.real_shell_calls=mul(p.pair_interactions,nr);
    p.fourier_shell_calls=mul(mul(2,p.pairs),p.reciprocal_batches);
    p.overlap_shell_calls=mul(2,p.pairs);
    p.shell_calls=add(p.real_shell_calls,add(p.fourier_shell_calls,p.overlap_shell_calls));
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic Ewald shell-call budget exceeded");
    const Count dn=mul(p.momenta,square(p.n_basis));
    cap(dn,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic Ewald density is not addressable");
    const Count domains=add(nr,ng); Count levels=1; for(auto m=domains;m>1;m>>=1) ++levels;
    const Count inputs=mul(96,domains); // input, copy, canonical inventory and sorting
    p.modeled_peak_bytes=add(source.plan_.modeled_peak_bytes,add(budget.other_live_bytes,
        add(budget.backend_margin_bytes,add(inputs,add(mul(96,dn),mul(128,p.reciprocal_batch))))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic Ewald driver memory budget exceeded");
    const auto leaf=unlimited(budget);
    KernelRequest request; request.kernel="erfc"; request.omega=1.;
    const auto mp=plan_md_tiles(source.basis_,request,leaf);
    const auto fp=plan_fourier(source.basis_,&source.basis_,std::max<Count>(1,p.reciprocal_batch),leaf);
    const auto c=census(source.basis_,6); const Count m=square(c.maximum_size);
    const Count coefficient_panels=add(add(p.pairs,p.pair_interactions),add(p.fourier_shell_calls,p.overlap_shell_calls));
    p.primitive_products=add(mul(p.real_shell_calls,mp.primitive_quartets_upper_bound),
        mul(add(p.fourier_shell_calls,p.overlap_shell_calls),square(mp.maximum_primitives)));
    p.scalar_terms=add(mul(16,dn),add(mul(16,mul(p.real_shell_calls,square(m))),
        add(mul(32,mul(coefficient_panels,mul(m,p.momenta))),add(mul(16,mul(m,mul(p.pairs,ng))),mul(128,mul(domains,levels))))));
    // The existing full-basis Fourier plan is a conservative upper bound for
    // each streamed shell panel. It reserves no all-pair Fourier cache here.
    p.kernel_terms=add(mul(p.real_shell_calls,mp.kernel_terms),
        add(mul(add(p.fourier_shell_calls,p.overlap_shell_calls),fp.kernel_terms),mul(64,p.scalar_terms)));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(add(mul(256,m),add(mul(8,c.bytes),32768)),
        std::max(mp.modeled_peak_bytes,fp.modeled_peak_bytes)-budget.backend_margin_bytes));
    cap(p.primitive_products,budget.primitive_quartets,"MANTA periodic Ewald primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic Ewald scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic Ewald work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic Ewald memory budget exceeded");
    return p;
}

PeriodicEwaldSource::PeriodicEwaldSource(std::shared_ptr<const PeriodicPairSource> source,double eta,
    const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal,
    Count batch,const ResourceBudget& budget)
    :plan_(plan_source(required(source),real.size(),reciprocal.size(),batch,budget)),source_(std::move(source)) {
    // Reuse exactly the lattice-tail domain convention. The full Ewald hash
    // deliberately excludes a near partition; this direct source has none.
    const lattice_detail::Geometry g(source_->lattice_,{},eta,real,reciprocal,{{0,0,0}},0);
    real_=g.real; reciprocal_=g.reciprocal;
    auto& r=receipt_; r.eta=eta; r.cell_volume=g.volume;
    r.gauge_constant=number(-pi/Real(eta)/eta/g.volume);
    r.real_domain_identity=g.real_identity; r.reciprocal_domain_identity=g.reciprocal_identity;
    r.pair_source_identity=source_->receipt_.source_identity; r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    lattice_detail::Hash kernel; kernel.text("manta.periodic-ewald-kernel.v1;finite;columns;binary64");
    kernel.text(r.boundary); kernel.number(eta);
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) kernel.number(source_->lattice_(i,j));
    kernel.text(g.real_identity); kernel.text(g.reciprocal_identity); r.kernel_identity=kernel.value.finish_hex();
    lattice_detail::Hash hash; hash.text("manta.periodic-ewald-source.v1;canonical-quadratic;no-screening");
    hash.text(r.pair_source_identity); hash.text(r.kernel_identity); hash.text(r.normalization);
    r.source_identity=hash.value.finish_hex();
}

md::Matrix PeriodicEwaldSource::real_tile(Count p,Count q,const LatticeIndex& index,PeriodicEwaldReceipt& r) const {
    const auto& entries=source_->entries_; const auto& shells=source_->basis_.libint();
    const auto& a=shells[entries[p].shells[0]];
    const auto b=shifted(shells[entries[p].shells[1]],entries[p].translation);
    auto shift=vector(source_->lattice_,index); for(auto& x:shift) x=-x;
    const auto c=shifted(shells[entries[q].shells[0]],shift);
    const auto e=shifted(shifted(shells[entries[q].shells[1]],entries[q].translation),shift);
    auto tile=md::shell_coulomb({&a,&b,&c,&e},md::RadialKernel::Erfc,r.eta);
    if(!tile.allFinite()) throw std::overflow_error("MANTA periodic Ewald integral overflow");
    ++r.real_shell_calls; r.primitive_products+=a.alpha.size()*b.alpha.size()*c.alpha.size()*e.alpha.size();
    return tile;
}
md::ComplexMatrix PeriodicEwaldSource::fourier_panel(Count p,const md::Matrix& g,PeriodicEwaldReceipt& r) const {
    const auto& e=source_->entries_[p]; const auto& a=source_->basis_.libint()[e.shells[0]];
    const auto b=shifted(source_->basis_.libint()[e.shells[1]],e.translation);
    auto f=md::shell_fourier_panel(&a,&b,g);
    if(!f.allFinite()) throw std::overflow_error("MANTA periodic Ewald Fourier overflow");
    ++r.fourier_shell_calls; const Count primitives=a.alpha.size()*b.alpha.size();
    r.primitive_products+=primitives; r.vector_primitive_products+=primitives*g.rows(); return f;
}
md::Matrix PeriodicEwaldSource::overlap_panel(Count p,PeriodicEwaldReceipt& r) const {
    const auto& e=source_->entries_[p]; const auto& a=source_->basis_.libint()[e.shells[0]];
    const auto b=shifted(source_->basis_.libint()[e.shells[1]],e.translation);
    auto s=md::shell_one_electron(a,b,"overlap");
    if(!s.allFinite()) throw std::overflow_error("MANTA periodic Ewald overlap overflow");
    ++r.overlap_shell_calls; r.primitive_products+=a.alpha.size()*b.alpha.size(); return s;
}
md::Matrix PeriodicEwaldSource::reciprocal_tile(Count start,Count ng,std::vector<double>& lambda) const {
    const auto& r=receipt_;
    const Eigen::Matrix3d reciprocal=number(2*pi)*source_->lattice_.inverse().transpose();
    md::Matrix g(ng,3); lambda.resize(ng);
    for(Count ig=0;ig<ng;++ig) {
        const auto v=vector(reciprocal,reciprocal_[start+ig]);
        for(int axis=0;axis<3;++axis) g(ig,axis)=v[axis];
        const double norm=std::hypot(v[0],v[1],v[2]);
        if(!std::isfinite(norm) || norm<=0) throw std::overflow_error("MANTA periodic Ewald reciprocal vector is not representable");
        const Real t=Real(norm)/r.eta/2;
        lambda[ig]=number(std::exp(std::log(4*pi)-std::log(Real(r.cell_volume))-2*std::log(Real(norm))-t*t));
    }
    return g;
}
std::array<md::Matrix,3> PeriodicEwaldSource::interaction(Count p,Count q,PeriodicEwaldReceipt& receipt) const {
    // Private block fallback for the source-bound periodic bubble consumer.
    // Reuse the identical primitive kernels, phases, gauge and finite domains.
    const auto& a=source_->entries_.at(p); const auto& b=source_->entries_.at(q);
    const Count na=a.sizes[0]*a.sizes[1],nb=b.sizes[0]*b.sizes[1];
    std::array<md::Matrix,3> values;
    for(auto& m:values) m=md::Matrix::Zero(na,nb);
    md::Matrix error=md::Matrix::Zero(na,nb);
    for(const auto& index:real_) {
        const auto tile=real_tile(p,q,index,receipt);
        for(Count i=0;i<na;++i) for(Count j=0;j<nb;++j) accumulate(tile(i,j),values[0](i,j),error(i,j));
    }
    error.setZero();
    for(Count start=0;start<reciprocal_.size();start+=plan_.reciprocal_batch) {
        const Count ng=std::min<Count>(plan_.reciprocal_batch,reciprocal_.size()-start);
        std::vector<double> lambda; const auto g=reciprocal_tile(start,ng,lambda);
        const auto fa=fourier_panel(p,g,receipt),fb=fourier_panel(q,g,receipt);
        for(Count i=0;i<na;++i) for(Count j=0;j<nb;++j) for(Count ig=0;ig<ng;++ig) {
            const Real x=Real(lambda[ig])*(Real(fa(i,ig).real())*fb(j,ig).real()+Real(fa(i,ig).imag())*fb(j,ig).imag());
            accumulate(number(x),values[1](i,j),error(i,j));
        }
    }
    const auto qa=overlap_panel(p,receipt),qb=overlap_panel(q,receipt);
    for(Count i=0;i<na;++i) for(Count j=0;j<nb;++j)
        values[2](i,j)=number(Real(receipt_.gauge_constant)*qa.data()[i]*qb.data()[j]);
    if(p==q) for(auto& m:values) for(Count i=0;i<na;++i) for(Count j=0;j<i;++j)
        m(i,j)=m(j,i)=number((Real(m(i,j))+m(j,i))/2);
    return values;
}

PeriodicEwaldResult PeriodicEwaldSource::build(const std::vector<Complex>& d) const {
    PeriodicEwaldResult out; out.plan=plan_; out.receipt=receipt_; auto& r=out.receipt;
    validate_density(*source_,d,r);
    const auto& entries=source_->entries_;
    const auto& k=source_->momenta_; const auto& weights=source_->weights_; const Count n=plan_.n_basis;
    out.values.resize(plan_.momenta*n*n); std::vector<Complex> correction(out.values.size());
    auto panel=[&](Count p) { ++r.coefficient_panels; return coefficients(entries[p],n,k,weights,d); };
    Real energy=0,energy_error=0;
    // Disjoint unordered canonical pair blocks. For a diagonal block the
    // finite even kernel equals (I+I^T)/2. Apply that identity while digesting
    // each image, so the numerical action has a fixed symmetric Hessian.
    for(Count p=0;p<entries.size();++p) {
        const auto cp=panel(p);
        for(Count q=p;q<entries.size();++q) {
            const auto cq=panel(q);
            std::vector<Real> vp(cp.size()),ep(cp.size()),vq(cq.size()),eq(cq.size());
            for(const auto& index:real_) {
                const auto tile=real_tile(p,q,index,r);
                for(Count i=0;i<cp.size();++i) for(Count j=0;j<cq.size();++j) {
                    if(p==q) {
                        accumulate(Real(.5)*cq[j].times(tile(i,j)),vp[i],ep[i]);
                        accumulate(Real(.5)*cp[i].times(tile(i,j)),vp[j],ep[j]);
                    } else {
                        accumulate(cq[j].times(tile(i,j)),vp[i],ep[i]);
                        accumulate(cp[i].times(tile(i,j)),vq[j],eq[j]);
                    }
                }
            }
            for(Count i=0;i<cp.size();++i) accumulate((p==q?Real(.5):Real(1))*cp[i].times(number(vp[i])),energy,energy_error);
            scatter(entries[p],n,k,vp,out.values,correction);
            if(p!=q) scatter(entries[q],n,k,vq,out.values,correction);
        }
    }
    r.real_energy=number(energy);
    // Stream one reciprocal tile twice: rho(G), then its AO-pair adjoint.
    // Sun2017 Eqs.16-17; no global pair/G or four-index array is retained.
    energy=energy_error=0;
    for(Count start=0;start<reciprocal_.size();start+=plan_.reciprocal_batch) {
        const Count ng=std::min<Count>(plan_.reciprocal_batch,reciprocal_.size()-start);
        std::vector<double> lambda; const auto g=reciprocal_tile(start,ng,lambda);
        std::vector<Real> rr(ng),ri(ng),er(ng),ei(ng);
        for(Count p=0;p<entries.size();++p) {
            const auto cp=panel(p); const auto f=fourier_panel(p,g,r);
            for(Count i=0;i<cp.size();++i) for(Count ig=0;ig<ng;++ig) {
                accumulate(cp[i].times(f(i,ig).real()),rr[ig],er[ig]);
                accumulate(cp[i].times(f(i,ig).imag()),ri[ig],ei[ig]);
            }
        }
        std::vector<Complex> rho(ng);
        for(Count ig=0;ig<ng;++ig) {
            rho[ig]={number(rr[ig]),number(ri[ig])};
            accumulate(Real(.5)*lambda[ig]*(rr[ig]*rr[ig]+ri[ig]*ri[ig]),energy,energy_error);
        }
        for(Count p=0;p<entries.size();++p) {
            const auto f=fourier_panel(p,g,r); std::vector<Real> v(f.rows()),e(f.rows());
            for(Eigen::Index i=0;i<f.rows();++i) for(Count ig=0;ig<ng;++ig) {
                const Real x=Real(lambda[ig])*(Real(rho[ig].real())*f(i,ig).real()+Real(rho[ig].imag())*f(i,ig).imag());
                accumulate(x,v[i],e[i]);
            }
            scatter(entries[p],n,k,v,out.values,correction);
        }
    }
    r.reciprocal_energy=number(energy);
    // The constant term belongs to this charged electronic component once.
    // It is not an extra whole-cell charge correction or nuclear point self.
    Real charge=0,charge_error=0;
    for(Count p=0;p<entries.size();++p) {
        const auto cp=panel(p); const auto s=overlap_panel(p,r);
        for(Count i=0;i<cp.size();++i) accumulate(cp[i].times(s.data()[i]),charge,charge_error);
    }
    r.charge=number(charge); r.gauge_energy=number(Real(.5)*r.gauge_constant*charge*charge);
    for(Count p=0;p<entries.size();++p) {
        const auto s=overlap_panel(p,r); std::vector<Real> v(s.size());
        for(Eigen::Index i=0;i<s.size();++i) v[i]=Real(r.gauge_constant)*charge*s.data()[i];
        scatter(entries[p],n,k,v,out.values,correction);
    }
    r.energy=number(Real(r.real_energy)+r.reciprocal_energy+r.gauge_energy);
    Real trace=0,trace_error=0; double scale=0;
    for(Count ik=0;ik<plan_.momenta;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto v=out.values[(ik*n+i)*n+j]; finite(v); require_finite(std::abs(v));
        scale=std::max(scale,std::abs(v));
        r.response_hermiticity_residual=std::max(r.response_hermiticity_residual,std::abs(v-std::conj(out.values[(ik*n+j)*n+i])));
        const auto x=d[(ik*n+j)*n+i];
        accumulate(Real(.5)*weights[ik]*(Real(x.real())*v.real()-Real(x.imag())*v.imag()),trace,trace_error);
    }
    r.trace_energy=number(trace); require_finite(r.response_hermiticity_residual);
    if(r.response_hermiticity_residual>1e-11*std::max(1.,scale))
        throw std::runtime_error("MANTA periodic Ewald response violates Hermiticity");
    return out;
}
} // namespace vibeqc::manta
