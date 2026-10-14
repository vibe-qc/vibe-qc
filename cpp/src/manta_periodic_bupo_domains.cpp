#include "vibeqc/manta_periodic_bupo_domains.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/periodic_reciprocal_source.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using namespace admission;
using Numeric=detail::PeriodicReciprocalNumericPlan;
using Label=std::array<std::int64_t,3>;
struct Transfer { Label numerator{},wrap{}; };
Transfer transfer(const RegularKMesh& mesh,Count target,Count source) {
    const auto t=mesh.address(target),s=mesh.address(source);Transfer out;
    for(int d=0;d<3;++d) {
        const std::int64_t raw=static_cast<std::int64_t>(t.doubled[d])-s.doubled[d];
        out.wrap[d]=raw>=mesh.mesh()[d]?1:(raw < -mesh.mesh()[d]?-1:0);
        out.numerator[d]=raw-out.wrap[d]*mesh.doubled_modulus()[d];
    }
    return out;
}
Numeric numeric(const PeriodicSystem& system,const RegularKMesh& mesh,Count target,Count source,double cutoff) {
    // Sun et al. (2017), Sec.II, DOI 10.1063/1.4998644: Gaussian Bloch
    // reciprocal exchange. Use the shared certified fixed-FMA predicate.
    // The enumerator owns the centered transfer convention. Exact integer
    // wraps translate its labels into PeriodicExchangeBoundSource's raw
    // target-minus-source frame without changing a physical vector.
    const auto q=transfer(mesh,target,source);Eigen::Vector3d fraction;
    for(int d=0;d<3;++d)fraction[d]=static_cast<double>(q.numerator[d])/mesh.doubled_modulus()[d];
    return detail::prepare_periodic_reciprocal_numeric_source(system.reciprocal_lattice(),fraction,
        cutoff,target==source,std::numeric_limits<Count>::max());
}
PeriodicBupoHFOptions effective(const PeriodicBupoHFOptions& o,const PeriodicBupoHFDomainPlan& p) {
    auto out=o;out.exchange_bound_cube_radius=p.exchange_bound_cube_radius;return out;
}
ResourceBudget remaining(const ResourceBudget& b,const PeriodicBupoHFDomainPlan& p) {
    auto out=b;out.other_live_bytes=add(b.other_live_bytes,p.domain_live_bytes);
    out.scalar_terms-=p.domain_scalar_terms;out.kernel_terms-=p.domain_kernel_terms;return out;
}
struct Digest {
    detail::Sha256 hash;
    void word(Count n) { std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=n>>(56-8*i);hash.update(b.data(),8); }
    void real(double x) { if(x==0)x=0.;Count n;std::memcpy(&n,&x,8);word(n); }
    void text(const std::string& s) {word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());}
};
void ignore(const Label&,const std::array<double,5>&,void*) {}
PeriodicBupoHFExchangeDomains generate(const PeriodicBupoSource& source,const PeriodicSystem& system,
    const RegularKMesh& mesh,const PeriodicBupoHFDomainPlan& plan) {
    PeriodicBupoHFExchangeDomains out;out.plan=plan;
    const Count nk=mesh.size(),pairs=square(nk);
    std::vector<Numeric> plans;plans.reserve(pairs);std::vector<Count> counts(pairs);
    out.transfer_offsets.resize(add(pairs,1));out.transfer_offsets[0]=0;
    for(Count target=0;target<nk;++target)for(Count origin=0;origin<nk;++origin) {
        const Count pair=target*nk+origin;
        plans.push_back(numeric(system,mesh,target,origin,plan.reciprocal_energy_cutoff));
        counts[pair]=detail::visit_periodic_reciprocal_numeric_source(plans.back().geometry,ignore,nullptr);
        out.transfer_offsets[pair+1]=add(out.transfer_offsets[pair],counts[pair]);
    }
    for(Count target=0;target<nk;++target)for(Count origin=0;origin<nk;++origin) {
        const Count pair=target*nk+origin,opposite=origin*nk+target;
        if(counts[pair]!=counts[opposite])throw std::runtime_error("MANTA BUPO exchange cutoff has unequal conjugate counts");
        const auto a=transfer(mesh,target,origin),b=transfer(mesh,origin,target);Label shift{};
        for(int d=0;d<3;++d)shift[d]=-(a.numerator[d]+b.numerator[d])/mesh.doubled_modulus()[d];
        out.receipt.conjugacy_vectors=add(out.receipt.conjugacy_vectors,
            detail::require_periodic_reciprocal_numeric_conjugacy(plans[pair].geometry,plans[opposite].geometry,
                shift,counts[pair]));
    }
    cap(out.transfer_offsets.back(),plan.exchange_candidates,"MANTA BUPO exchange domain exceeded admitted vector count");
    out.exchange_vectors.resize(out.transfer_offsets.back(),3);
    for(Count pair=0;pair<pairs;++pair) {
        struct State { md::Matrix& matrix;Count position,end; } state{out.exchange_vectors,out.transfer_offsets[pair],out.transfer_offsets[pair+1]};
        const auto count=detail::visit_periodic_reciprocal_numeric_source(plans[pair].geometry,
            [](const Label&,const std::array<double,5>& lanes,void* opaque) {
                auto& s=*static_cast<State*>(opaque);
                if(s.position>=s.end)throw std::logic_error("MANTA BUPO exchange domain replay exceeded admitted count");
                for(int d=0;d<3;++d)s.matrix(s.position,d)=lanes[d];++s.position;
            },&state);
        if(count!=counts[pair] || state.position!=state.end)throw std::logic_error("MANTA BUPO exchange domain replay changed count");
    }
    Digest digest;digest.word(1);digest.text(source.source_identity());digest.real(plan.reciprocal_energy_cutoff);
    digest.word(plan.exchange_bound_cube_radius);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)digest.real(system.lattice(i,j));
    for(auto n:mesh.mesh())digest.word(n);for(auto n:mesh.is_shift())digest.word(n);
    for(auto n:out.transfer_offsets)digest.word(n);
    for(Eigen::Index i=0;i<out.exchange_vectors.rows();++i)for(int j=0;j<3;++j)digest.real(out.exchange_vectors(i,j));
    out.receipt.identity="manta.periodic-bupo-exchange-cutoff.v1:"+digest.hash.finish_hex();
    out.receipt.exchange_vectors=out.exchange_vectors.rows();return out;
}
} // namespace

PeriodicBupoHFDomainPlan plan_periodic_bupo_hf_from_cutoff(const PeriodicBupoSource& source,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicBupoHFOptions& options,const ResourceBudget& budget) {
    margin(budget);
    if(system.dim!=3 || !system.lattice.allFinite() || !system.lattice.inverse().allFinite()
       || !std::isfinite(system.lattice.determinant()) || system.lattice.determinant()==0)
        throw std::invalid_argument("MANTA BUPO exchange cutoff requires a finite nonsingular 3D lattice");
    if(!std::isfinite(cutoff) || cutoff<=0 || cutoff>std::numeric_limits<double>::max()/2)
        throw std::invalid_argument("MANTA BUPO exchange cutoff must be finite positive");
    if(!options.bound_exchange || options.exchange_bound_cube_radius!=0)
        throw std::invalid_argument("MANTA BUPO exchange cutoff requires bound_exchange=true and a factory-owned zero input cube radius");
    PeriodicBupoHFDomainPlan p;p.reciprocal_energy_cutoff=cutoff;p.momenta=mesh.size();const Count pairs=square(p.momenta);
    const Count preliminary=add(32768,mul(8192,pairs));
    cap(preliminary,budget.scalar_terms,"MANTA BUPO exchange domain planning scalar budget exceeded");
    cap(mul(32,preliminary),budget.kernel_terms,"MANTA BUPO exchange domain planning kernel budget exceeded");
    cap(add(budget.other_live_bytes,add(32768,mul(2048,pairs))),budget.memory_bytes,
        "MANTA BUPO exchange domain planning memory budget exceeded");
    for(Count target=0;target<p.momenta;++target)for(Count origin=0;origin<p.momenta;++origin) {
        const auto n=numeric(system,mesh,target,origin,cutoff);const auto count=n.enumeration.candidate_count;
        const auto q=transfer(mesh,target,origin);
        p.exchange_candidates=add(p.exchange_candidates,count);p.maximum_exchange_candidates=std::max(p.maximum_exchange_candidates,count);
        for(int d=0;d<3;++d)for(auto edge:{n.geometry.lower_bounds[d],n.geometry.upper_bounds[d]}) {
            if(edge<=-std::numeric_limits<int>::max() || edge>=std::numeric_limits<int>::max())
                throw std::invalid_argument("MANTA BUPO exchange completion cube exceeds the supported integer extent");
            // Q = B(n_centered + q_centered) = B(n_raw + q_raw),
            // q_raw=q_centered+wrap, hence n_raw=n_centered-wrap.
            p.exchange_bound_cube_radius=std::max(p.exchange_bound_cube_radius,static_cast<Count>(std::abs(edge-q.wrap[d])));
        }
    }
    // Count, opposite-source audit and replay. Include digesting, metadata,
    // native/owned Python vectors, offsets and their simultaneous lifetimes.
    p.candidate_visits=mul(3,p.exchange_candidates);
    p.domain_scalar_terms=add(preliminary,mul(512,p.candidate_visits));p.domain_kernel_terms=mul(32,p.domain_scalar_terms);
    p.domain_live_bytes=add(32768,add(mul(2048,pairs),add(mul(32,add(pairs,1)),mul(96,p.exchange_candidates))));
    cap(p.domain_scalar_terms,budget.scalar_terms,"MANTA BUPO exchange domain scalar budget exceeded");
    cap(p.domain_kernel_terms,budget.kernel_terms,"MANTA BUPO exchange domain kernel budget exceeded");
    cap(add(budget.other_live_bytes,p.domain_live_bytes),budget.memory_bytes,"MANTA BUPO exchange domain memory budget exceeded");
    p.hf=plan_periodic_bupo_hf(source,system,mesh,p.maximum_exchange_candidates,effective(options,p),remaining(budget,p));
    p.modeled_peak_bytes=p.hf.modeled_peak_bytes;p.shell_calls=p.hf.shell_calls;p.primitive_pairs=p.hf.primitive_pairs;
    p.scalar_terms=add(p.domain_scalar_terms,p.hf.scalar_terms);p.kernel_terms=add(p.domain_kernel_terms,p.hf.kernel_terms);
    return p;
}
PeriodicBupoHFExchangeDomains build_periodic_bupo_hf_exchange_domains(const PeriodicBupoSource& source,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicBupoHFOptions& options,const ResourceBudget& budget) {
    return generate(source,system,mesh,plan_periodic_bupo_hf_from_cutoff(source,system,mesh,cutoff,options,budget));
}
PeriodicBupoHFResult run_periodic_bupo_hf_from_cutoff(std::shared_ptr<const PeriodicBupoSource> source,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicBupoHFOptions& options,const ResourceBudget& budget) {
    if(!source)throw std::invalid_argument("MANTA BUPO exchange cutoff requires a native electronic source");
    const auto domains=build_periodic_bupo_hf_exchange_domains(*source,system,mesh,cutoff,options,budget);
    const auto& p=domains.plan;
    auto out=run_periodic_bupo_hf(std::move(source),system,mesh,domains.exchange_vectors,domains.transfer_offsets,effective(options,p),remaining(budget,p));
    out.plan.domain_live_bytes=p.domain_live_bytes;out.plan.domain_scalar_terms=p.domain_scalar_terms;out.plan.domain_kernel_terms=p.domain_kernel_terms;
    out.plan.scalar_terms=add(out.plan.scalar_terms,p.domain_scalar_terms);out.plan.kernel_terms=add(out.plan.kernel_terms,p.domain_kernel_terms);
    auto& r=out.receipt;r.domain_selection="original-bupo-source;certified-reciprocal-energy-sphere";r.domain_identity=domains.receipt.identity;
    r.reciprocal_energy_cutoff=cutoff;r.domain_translations=out.plan.cells;r.domain_exchange_vectors=domains.receipt.exchange_vectors;
    r.domain_conjugacy_vectors=domains.receipt.conjugacy_vectors;return out;
}
} // namespace vibeqc::manta
