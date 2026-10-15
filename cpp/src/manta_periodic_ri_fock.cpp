#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
using Index=std::array<std::int64_t,3>;
using namespace admission;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double v) { if(v==0)v=0.;Count bits;std::memcpy(&bits,&v,8);word(bits); }
    void value(Complex z) { real(z.real());real(z.imag()); }
};
ResourceBudget unlimited(ResourceBudget b) {
    b.other_live_bytes=0;b.memory_bytes=b.shell_calls=b.primitive_quartets=b.scalar_terms=b.kernel_terms=std::numeric_limits<Count>::max();return b;
}
Count maximum_segment(const std::vector<Count>& offsets,Count nk,Count vectors,bool make_k) {
    const Count pairs=square(nk);
    if(offsets.size()!=add(pairs,1) || offsets.front()!=0 || offsets.back()!=vectors)
        throw std::invalid_argument("MANTA RI Fock offsets must delimit every target/source pair");
    Count maximum=0;
    for(Count i=0;i<nk;++i)for(Count j=0;j<nk;++j) {
        const Count p=i*nk+j;
        if(offsets[p+1]<offsets[p] || offsets[p+1]>vectors)
            throw std::invalid_argument("MANTA RI Fock offsets must be monotone and bounded");
        const Count n=offsets[p+1]-offsets[p];
        if(!make_k && n)
            throw std::invalid_argument("MANTA RI Fock J-only requires an empty pair domain");
        if(make_k && ((i==j && n) || (i!=j && !n)))
            throw std::invalid_argument("MANTA RI Fock K needs empty diagonal and nonempty off-diagonal pair segments");
        maximum=std::max(maximum,n);
    }
    return maximum;
}
std::vector<Index> labels(const md::Matrix& lattice,const md::Matrix& v,const Eigen::Vector3d& q) {
    std::vector<Index> result(v.rows());
    for(Eigen::Index g=0;g<v.rows();++g) {
        const Eigen::Vector3d f=lattice.transpose()*(v.row(g).transpose()-q)/(2*std::acos(-1.));
        for(int d=0;d<3;++d) {
            if(!std::isfinite(f[d]) || std::abs(f[d])>0x1p40 || std::abs(f[d]-std::round(f[d]))>1e-10)
                throw std::invalid_argument("MANTA RI Fock requires G+ket-bra reciprocal vectors");
            result[g][d]=static_cast<std::int64_t>(std::round(f[d]));
        }
    }
    std::sort(result.begin(),result.end());
    for(std::size_t g=1;g<result.size();++g)if(result[g]==result[g-1])
        throw std::invalid_argument("MANTA RI Fock reciprocal domain contains duplicate vectors");
    return result;
}
void reversal(const std::vector<Index>& a,const std::vector<Index>& b) {
    if(a.size()!=b.size()) throw std::invalid_argument("MANTA RI Fock reverse-transfer domains differ");
    for(const auto& n:a)if(!std::binary_search(b.begin(),b.end(),Index{-n[0],-n[1],-n[2]}))
        throw std::invalid_argument("MANTA RI Fock reciprocal domains must close under transfer reversal");
}
double conjugacy(const Eigen::MatrixXcd& a,const Eigen::MatrixXcd& b) {
    double scale=1.,error=0.;
    for(Eigen::Index i=0;i<a.rows();++i)for(Eigen::Index j=0;j<a.cols();++j) {
        scale=std::max({scale,std::abs(a(i,j)),std::abs(b(i,j))});error=std::max(error,std::abs(a(i,j)-std::conj(b(i,j))));
    }
    require_finite(scale);require_finite(error);return error/scale;
}
void confirm_reverse(const PeriodicRIQBuilder& a,const PeriodicRIQBuilder& b,PeriodicRIFockInfo& info) {
    if(a.metric_info().rank!=b.metric_info().rank)
        throw std::runtime_error("MANTA RI Fock reverse-transfer metric ranks differ");
    info.reverse_metric_residual=std::max(info.reverse_metric_residual,conjugacy(a.metric(),b.metric()));
    info.reverse_projector_residual=std::max(info.reverse_projector_residual,conjugacy(a.projector(),b.projector()));
    const Eigen::MatrixXcd ai=a.factor()*a.factor().adjoint(),bi=b.factor()*b.factor().adjoint();
    info.reverse_inverse_residual=std::max(info.reverse_inverse_residual,conjugacy(ai,bi));
    if(std::max({info.reverse_metric_residual,info.reverse_projector_residual,info.reverse_inverse_residual})>1e-10)
        throw std::runtime_error("MANTA RI Fock reverse-transfer metric/projector/inverse failed conjugacy");
}
double hermitian(const std::vector<Complex>& values,Count n,double tolerance=1e-12) {
    double maximum=0.;
    for(Count start=0;start<values.size();start+=n*n) {
        double scale=1.,error=0.;
        for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
            const auto z=values[start+i*n+j];require_finite(z.real());require_finite(z.imag());
            scale=std::max(scale,std::abs(z));error=std::max(error,std::abs(z-std::conj(values[start+j*n+i])));
        }
        require_finite(scale);require_finite(error);
        if(error>tolerance*scale) throw std::invalid_argument("MANTA RI Fock requires Hermitian finite matrices without repair");
        maximum=std::max(maximum,error);
    }
    return maximum;
}
void accumulate(Complex z,Complex& sum,Complex& correction) {
    require_finite(z.real());require_finite(z.imag());const auto y=z-correction,t=sum+y;correction=(t-sum)-y;sum=t;
    require_finite(sum.real());require_finite(sum.imag());
}
double real_energy(Complex e) {
    require_finite(e.real());require_finite(e.imag());
    if(std::abs(e.imag())>1e-10*std::max(1.,std::abs(e.real()))) throw std::runtime_error("MANTA RI Fock energy is not real");
    return e.real();
}
}

PeriodicRIFockPlan plan_periodic_ri_fock(const BasisSet& orbital,const BasisSet& auxiliary,const PeriodicSystem& system,
    Count ng,Count nq,Count nk,const PeriodicGDFShortRangeOptions& o,const RIOptions& ri,const ResourceBudget& budget,
    bool make_j,bool make_k,const PeriodicGDFKernelOptions& physical) {
    using namespace admission;
    margin(budget);
    if(!make_j && !make_k) throw std::invalid_argument("MANTA RI Fock requires a Coulomb or exchange target");
    if((o.coupling_screen_error>0. || o.coupling_multipole_error>0.) && (make_k || physical.kernel!="coulomb"))
        throw std::invalid_argument("MANTA coupling-only approximation requires J-only fitting and independent exchange");
    if(physical.kernel!="coulomb" && make_j)
        throw std::invalid_argument("MANTA physical range RI Fock requires K-only selection and a separate Coulomb J source");
    if(!make_k && nq) throw std::invalid_argument("MANTA RI Fock J-only requires an empty pair domain");
    if(!nk || !ng || (make_k && nk>1 && !nq) || !o.compute_metric)
        throw std::invalid_argument("MANTA RI Fock requires nonempty mesh/domains and an enabled metric");
    const auto leaf=unlimited(budget);const Count off=make_k?mul(nk,nk-1):0,channels=add(1,off);
    const auto gamma=plan_periodic_gdf_tiles(orbital,auxiliary,system,ng,nk,o,leaf,physical);
    PeriodicGDFTilesPlan pair;
    PeriodicRIFockPlan p;p.n_basis=orbital.nbasis();p.n_auxiliary=auxiliary.nbasis();p.momenta=nk;p.channels=channels;
    p.has_j=make_j;p.has_k=make_k;
    p.gamma=plan_periodic_ri_q(gamma,2,ri,leaf);
    if(off) {pair=plan_periodic_gdf_tiles(orbital,auxiliary,system,nq,1,o,leaf,physical);p.pair=plan_periodic_ri_q(pair,2,ri,leaf);}
    const Count n2=square(p.n_basis),m=p.n_auxiliary,m2=square(m);
    // Snapshot owners include every fixed metric, factor and projector. Only
    // one child executes at a time; its peak includes conservative spare owners.
    p.retained_source_bytes=add(gamma.retained_bytes,add(mul(off,pair.retained_bytes),mul(channels,add(4096,mul(16,add(mul(3,m2),mul(64,m)))))));
    p.driver_live_bytes=add(65536,add(mul(640,mul(nk,n2)),add(mul(4096,channels),
        mul(256,add(add(ng,mul(off,nq)),add(nk,square(nk)))))));
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(p.retained_source_bytes,add(p.driver_live_bytes,
        std::max(p.gamma.modeled_peak_bytes,p.pair.modeled_peak_bytes))));
    p.metric_shell_calls=add(gamma.metric_shell_calls,mul(off,pair.metric_shell_calls));
    p.metric_primitive_products=add(gamma.metric_primitive_products,mul(off,pair.metric_primitive_products));
    p.shell_calls=add(p.gamma.shell_calls,mul(off,p.pair.shell_calls));
    p.primitive_products=add(p.gamma.primitive_products,mul(off,p.pair.primitive_products));
    p.build_shell_calls=p.shell_calls-p.metric_shell_calls;p.build_primitive_products=p.primitive_products-p.metric_primitive_products;
    const Count driver=mul(512,add(mul(channels,mul(m,m2)),add(mul(nk,n2),add(ng,mul(off,nq)))));
    p.scalar_terms=add(driver,add(p.gamma.scalar_terms,mul(off,p.pair.scalar_terms)));
    p.kernel_terms=add(mul(16,driver),add(p.gamma.kernel_terms,mul(off,p.pair.kernel_terms)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA RI Fock aggregate memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA RI Fock aggregate shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA RI Fock aggregate primitive-product budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA RI Fock aggregate scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA RI Fock aggregate kernel-work budget exceeded");
    return p;
}
PeriodicRIFockBuilder::PeriodicRIFockBuilder(const BasisSet& orbital,const BasisSet& auxiliary,const PeriodicSystem& system,
    const md::Matrix& k,const std::vector<double>& w,const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,
    const PeriodicGDFShortRangeOptions& o,const RIOptions& ri,const ResourceBudget& budget,bool make_j,bool make_k,
    const PeriodicGDFKernelOptions& physical)
    :plan_([&] {
        if(k.cols()!=3 || g.cols()!=3 || q.cols()!=3)throw std::invalid_argument("MANTA RI Fock vectors need three components");
        return plan_periodic_ri_fock(orbital,auxiliary,system,g.rows(),maximum_segment(offsets,k.rows(),q.rows(),make_k),k.rows(),o,ri,budget,make_j,make_k,physical);
    }()),weights_(w) {
    const Count nk=plan_.momenta;
    info_.has_j=make_j;info_.has_k=make_k;
    info_.exchange_kernel=make_k?physical.kernel:"none";info_.physical_omega=physical.omega;
    if(w.size()!=nk || !k.allFinite() || !g.allFinite() || !q.allFinite())throw std::invalid_argument("MANTA RI Fock momenta/weights must be finite and match the mesh");
    double total=0.;for(double value:w) {require_finite(value);if(value<0.)throw std::invalid_argument("MANTA RI Fock weights must be nonnegative");total+=value;}
    if(!std::isfinite(total) || std::abs(total-1.)>1e-12)throw std::invalid_argument("MANTA RI Fock weights must sum to one");
    const auto gamma_labels=labels(system.lattice,g,Eigen::Vector3d::Zero());reversal(gamma_labels,gamma_labels);
    for(Count i=0;i<nk;++i)for(Count j=i+1;j<nk;++j) {
        const Eigen::Vector3d transfer=(k.row(j)-k.row(i)).transpose();
        const Eigen::Vector3d fractional=system.lattice.transpose()*transfer/(2*std::acos(-1.));
        if((fractional-fractional.array().round().matrix()).cwiseAbs().maxCoeff()<1e-12)
            throw std::invalid_argument("MANTA RI Fock mesh contains duplicate Bloch momenta");
        if(make_k) {
            const Count a=i*nk+j,b=j*nk+i;
            reversal(labels(system.lattice,q.middleRows(offsets[a],offsets[a+1]-offsets[a]),transfer),
                labels(system.lattice,q.middleRows(offsets[b],offsets[b+1]-offsets[b]),-transfer));
        }
    }
    const auto leaf=unlimited(budget);
    auto make=[&](const Eigen::Vector3d& transfer,const md::Matrix& ket,const md::Matrix& vectors) {
        return std::make_unique<PeriodicRIQBuilder>(std::make_shared<PeriodicGDFTiles>(orbital,auxiliary,system,transfer,ket,vectors,o,leaf,physical),2,ri,leaf);
    };
    gamma_=make(Eigen::Vector3d::Zero(),k,g);if(make_k)pairs_.resize(square(nk));
    confirm_reverse(*gamma_,*gamma_,info_);
    info_.frame_identity=gamma_->source_frame_identity();info_.metrics.reserve(plan_.channels);
    auto record=[&](const PeriodicRIQBuilder& b) {
        info_.metrics.push_back(b.metric_info());info_.metric_shell_calls=add(info_.metric_shell_calls,b.metric_info().shell_calls);
        info_.metric_primitive_products=add(info_.metric_primitive_products,b.metric_info().primitive_products);
    };
    record(*gamma_);
    if(make_k)for(Count i=0;i<nk;++i)for(Count j=0;j<nk;++j)if(i!=j) {
        const Count p=i*nk+j;pairs_[p]=make((k.row(j)-k.row(i)).transpose(),k.middleRows(j,1),q.middleRows(offsets[p],offsets[p+1]-offsets[p]));
        record(*pairs_[p]);
    }
    if(make_k)for(Count i=0;i<nk;++i)for(Count j=i+1;j<nk;++j)confirm_reverse(*pairs_[i*nk+j],*pairs_[j*nk+i],info_);
    Digest weights;weights.text("manta.periodic-ri-weights.v1");weights.word(nk);for(double value:w)weights.real(value);info_.weight_identity=weights.hash.finish_hex();
    Digest source;source.text("manta.periodic-ri-fock-source.v1;explicit-mesh;no-exchange-correction");
    if(!make_j || !make_k) {source.text("selected-targets.v1");source.word(make_j);source.word(make_k);}
    source.text(info_.frame_identity);source.text(info_.weight_identity);
    for(const auto& metric:info_.metrics)source.text(metric.source_identity);info_.source_identity=source.hash.finish_hex();
    cap(info_.metric_shell_calls,plan_.metric_shell_calls,"MANTA RI Fock executed metric shell bound exceeded");
    cap(info_.metric_primitive_products,plan_.metric_primitive_products,"MANTA RI Fock executed metric primitive bound exceeded");
}
PeriodicRIFockResult PeriodicRIFockBuilder::build(const std::vector<Complex>& h,const std::vector<Complex>& d,double scale) const {
    const Count n=plan_.n_basis,nk=plan_.momenta,n2=square(n),mesh=mul(nk,n2);
    if(h.size()!=mesh || d.size()!=mul(2,mesh))throw std::invalid_argument("MANTA RI Fock H/D dimensions disagree");
    require_finite(scale);PeriodicRIFockResult result;result.n_basis=n;result.momenta=nk;auto& r=result.receipt;
    r.has_j=plan_.has_j;r.has_k=plan_.has_k;
    r.exchange_kernel=info_.exchange_kernel;r.physical_omega=info_.physical_omega;
    // A coefficient cannot select an unadmitted target. Conversely zero
    // scale on a J/K builder still evaluates and exports its selected K.
    if(!r.has_k)scale=0.;
    r.hcore_hermiticity_residual=hermitian(h,n);hermitian(d,n);r.exchange_scale=scale;r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    r.frame_identity=info_.frame_identity;r.weight_identity=info_.weight_identity;result.components.reserve(plan_.channels);
    auto record=[&](const PeriodicRIQReceipt& c) {
        if(c.has_j) {
            r.reference_coupling_identity=c.reference_coupling_identity;r.coupling_error_bound=c.coupling_error_bound;
            r.approximation_j_bound=c.approximation_j_bound;r.approximation_energy_bound=c.approximation_energy_bound;
            r.coupling_multipole_trials=c.coupling_multipole_trials;r.coupling_multipole_images=c.coupling_multipole_images;
        }
        result.components.push_back(c);r.shell_calls=add(r.shell_calls,c.shell_calls);
        r.primitive_products=add(r.primitive_products,c.primitive_products);r.coupling_passes=add(r.coupling_passes,c.coupling_passes);
    };
    {auto gamma=gamma_->build(d,weights_,r.has_j,r.has_k);result.J=std::move(gamma.J);result.K=std::move(gamma.K);r.coulomb_energy=gamma.receipt.coulomb_energy;record(gamma.receipt);}
    std::vector<Complex> correction(result.K.size()),pair_density(r.has_k?mul(2,n2):0);
    if(r.has_k)for(Count i=0;i<nk;++i)for(Count j=0;j<nk;++j)if(i!=j) {
        for(Count spin=0;spin<2;++spin)std::copy_n(d.data()+(spin*nk+j)*n2,n2,pair_density.data()+spin*n2);
        const auto part=pairs_[i*nk+j]->build(pair_density,{weights_[j]},false,true);record(part.receipt);
        for(Count spin=0;spin<2;++spin)for(Count pair=0;pair<n2;++pair) {
            const Count out=(spin*nk+i)*n2+pair;accumulate(part.K[spin*n2+pair],result.K[out],correction[out]);
        }
    }
    result.values.resize(d.size());Complex one{},one_error{},exchange{},exchange_error{};
    for(Count spin=0;spin<2;++spin)for(Count ik=0;ik<nk;++ik)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
        const Count pair=i*n+j,out=(spin*nk+ik)*n2+pair;
        result.values[out]=h[ik*n2+pair]+(r.has_j?result.J[ik*n2+pair]:Complex{})-(r.has_k?scale*result.K[out]:Complex{});
        const auto weighted=weights_[ik]*d[(spin*nk+ik)*n2+j*n+i];
        accumulate(weighted*h[ik*n2+pair],one,one_error);
        if(r.has_k)accumulate(-.5*scale*weighted*result.K[out],exchange,exchange_error);
    }
    r.one_electron_energy=real_energy(one);r.exchange_energy=real_energy(exchange);r.electronic_energy=r.one_electron_energy+r.coulomb_energy+r.exchange_energy;
    require_finite(r.electronic_energy);r.fock_hermiticity_residual=hermitian(result.values,n,1e-10);
    Digest hd;hd.text("manta.periodic-ri-hcore.v1");hd.text(info_.frame_identity);for(auto z:h)hd.value(z);r.one_electron_identity=hd.hash.finish_hex();
    Digest source;source.text("manta.periodic-ri-fock.v1");source.text(info_.source_identity);source.text(r.one_electron_identity);source.real(scale);r.source_identity=source.hash.finish_hex();
    Digest density;density.text("manta.periodic-ri-spin-density.v1");density.text(info_.frame_identity);for(auto z:d)density.value(z);r.density_identity=density.hash.finish_hex();
    Digest snapshot;snapshot.text("manta.periodic-ri-fock-snapshot.v1");snapshot.text(r.source_identity);snapshot.text(r.density_identity);
    for(auto z:result.values)snapshot.value(z);snapshot.real(r.one_electron_energy);snapshot.real(r.coulomb_energy);snapshot.real(r.exchange_energy);r.snapshot_identity=snapshot.hash.finish_hex();
    cap(r.shell_calls,plan_.build_shell_calls,"MANTA RI Fock executed shell bound exceeded");
    cap(r.primitive_products,plan_.build_primitive_products,"MANTA RI Fock executed primitive bound exceeded");return result;
}
} // namespace vibeqc::manta
