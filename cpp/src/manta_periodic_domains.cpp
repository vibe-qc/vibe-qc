#include "vibeqc/manta_periodic_domains.hpp"
#include "vibeqc/manta_periodic_ks_domains.hpp"
#include "vibeqc/manta_periodic_ri_domains.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/periodic_reciprocal_source.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using namespace admission;
using Numeric=detail::PeriodicReciprocalNumericPlan;
using Label=std::array<std::int64_t,3>;
struct Digest {
    detail::Sha256 hash;
    void word(Count n) { std::array<std::uint8_t,8> b{}; for(int i=0;i<8;++i) b[i]=n>>(56-8*i); hash.update(b.data(),8); }
    void real(double x) { if(x==0) x=0.; Count n; std::memcpy(&n,&x,8); word(n); }
    void text(const std::string& s) { word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void matrix(const md::Matrix& a) {
        word(a.rows()); word(a.cols());
        for(Eigen::Index i=0;i<a.rows();++i) for(Eigen::Index j=0;j<a.cols();++j) real(a(i,j));
    }
};
Label centered(const RegularKMesh& mesh,Count q) {
    const auto address=mesh.transfer_address(q); Label out{};
    for(int d=0;d<3;++d) out[d]=static_cast<std::int64_t>(address.doubled[d])
        -(address.doubled[d]>=mesh.mesh()[d] ? mesh.doubled_modulus()[d] : 0);
    return out;
}
Numeric numeric(const Eigen::Matrix3d& b,const RegularKMesh& mesh,Count q,double cutoff) {
    const auto numerator=centered(mesh,q); Eigen::Vector3d f;
    for(int d=0;d<3;++d) f[d]=static_cast<double>(numerator[d])/mesh.doubled_modulus()[d];
    return detail::prepare_periodic_reciprocal_numeric_source(b,f,cutoff,q==0,std::numeric_limits<Count>::max());
}
Count real_box(const Eigen::Matrix3d& a,double radius) {
    Count result=1; const double volume=std::abs(a.determinant());
    for(int d=0;d<3;++d) {
        const double cross=a.col((d+1)%3).cross(a.col((d+2)%3)).norm(),distance=volume/cross;
        const double extent=std::ceil(radius/distance);
        if(!std::isfinite(cross) || cross<1e-14 || !std::isfinite(distance) || distance<1e-14
           || !std::isfinite(extent) || extent>=std::numeric_limits<int>::max())
            throw std::invalid_argument("MANTA periodic domain lattice/image extent is unsupported");
        result=mul(result,add(mul(2,static_cast<Count>(extent)),1));
    }
    return result;
}
template<class Plan> ResourceBudget remaining(const ResourceBudget& b,const Plan& p) {
    auto out=b; out.other_live_bytes=add(b.other_live_bytes,p.domain_live_bytes);
    out.scalar_terms-=p.domain_scalar_terms; out.kernel_terms-=p.domain_kernel_terms; return out;
}
void ignore(const Label&,const std::array<double,5>&,void*) {}
void fill(const Numeric& p,md::Matrix& out,Count begin,Count count,double sign=1.) {
    struct State { md::Matrix& out; Count position,end; double sign; } state{out,begin,add(begin,count),sign};
    const auto accepted=detail::visit_periodic_reciprocal_numeric_source(p.geometry,
        [](const Label&,const std::array<double,5>& lanes,void* opaque) {
            auto& s=*static_cast<State*>(opaque);
            if(s.position>=s.end) throw std::logic_error("MANTA periodic domain replay exceeded admitted extent");
            for(int d=0;d<3;++d) s.out(s.position,d)=s.sign*lanes[d];
            ++s.position;
        },&state);
    if(accepted!=count || state.position!=state.end)
        throw std::logic_error("MANTA periodic domain replay changed accepted count");
}
template<class Domains, class Plan> Domains generate(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
                            const Plan& plan,bool fitted=false) {
    Domains out; out.plan=plan; const Count nk=mesh.size();
    const auto reciprocal=system.reciprocal_lattice();
    std::vector<Numeric> plans; plans.reserve(nk); std::vector<Count> counts(nk);
    for(Count q=0;q<nk;++q) {
        plans.push_back(numeric(reciprocal,mesh,q,plan.reciprocal_energy_cutoff));
        counts[q]=detail::visit_periodic_reciprocal_numeric_source(plans.back().geometry,ignore,nullptr);
    }
    if(counts[0]==0) throw std::invalid_argument("MANTA periodic reciprocal cutoff contains no nonzero Hartree vectors");
    for(Count q=0;q<nk;++q) {
        const Count opposite=mesh.transfer_index(mesh.negate(mesh.transfer_address(q)));
        if(counts[q]!=counts[opposite])
            throw std::runtime_error("MANTA periodic cutoff domains have unequal conjugate vector counts");
        const auto a=centered(mesh,q),b=centered(mesh,opposite); Label shift{};
        for(int d=0;d<3;++d) shift[d]=-(a[d]+b[d])/mesh.doubled_modulus()[d];
        out.receipt.conjugacy_vectors=add(out.receipt.conjugacy_vectors,
            detail::require_periodic_reciprocal_numeric_conjugacy(
                plans[q].geometry,plans[opposite].geometry,shift,counts[q]));
    }
    // Shared pair-complete image source. Individual kernels retain the same
    // physical ||O_mu-O_nu-R|| cutoff, including unwrapped atomic images.
    const auto cells=pair_complete_cells(system,plan.pair_cutoff_bohr,basis.libint(),basis.libint());
    cap(cells.size(),plan.translation_candidates,"MANTA periodic translations exceeded admitted extent");
    out.translations.resize(cells.size(),3);
    for(Count i=0;i<cells.size();++i) out.translations.row(i)=cells[i].r_cart.transpose();
    out.reciprocal_vectors.resize(counts[0],3); fill(plans[0],out.reciprocal_vectors,0,counts[0]);
    const Count pairs=square(nk); out.transfer_offsets.resize(add(pairs,1)); out.transfer_offsets[0]=0;
    for(Count target=0;target<nk;++target) for(Count source=0;source<nk;++source) {
        const Count pair=target*nk+source,q=mesh.transfer_index(source,target);
        const Count count=fitted && target==source ? 0 : counts[q];
        if(fitted && target!=source && !count)
            throw std::invalid_argument("MANTA periodic RI cutoff contains an empty off-diagonal pair sphere");
        out.transfer_offsets[pair+1]=add(out.transfer_offsets[pair],count);
    }
    cap(out.transfer_offsets.back(),plan.exchange_candidates,"MANTA periodic exchange vectors exceeded admitted extent");
    out.exchange_vectors.resize(out.transfer_offsets.back(),3);
    for(Count target=0;target<nk;++target) for(Count source=0;source<nk;++source) {
        const Count pair=target*nk+source,q=mesh.transfer_index(source,target);
        if(fitted && target==source) continue;
        // Direct exchange uses target-minus-source; fitted three-center
        // sources use ket-minus-bra. Negate the complete physical vector,
        // including its reciprocal label, even for wrapped mesh transfers.
        fill(plans[q],out.exchange_vectors,out.transfer_offsets[pair],counts[q],fitted ? -1. : 1.);
    }
    Digest digest; digest.word(1); digest.matrix(system.lattice);
    for(auto n:mesh.mesh()) digest.word(n);
    for(auto n:mesh.is_shift()) digest.word(n);
    digest.real(plan.reciprocal_energy_cutoff); digest.real(plan.pair_cutoff_bohr);
    digest.word(basis.nshells()); for(const auto& shell:basis.libint()) for(auto x:shell.O) digest.real(x);
    digest.matrix(out.translations); digest.matrix(out.reciprocal_vectors); digest.matrix(out.exchange_vectors);
    for(auto n:out.transfer_offsets) digest.word(n);
    out.receipt.identity=std::string(fitted ? "manta.periodic-ri-cutoff-domains.v1:" : "manta.periodic-cutoff-domains.v1:")+digest.hash.finish_hex();
    out.receipt.translations=out.translations.rows(); out.receipt.hartree_vectors=out.reciprocal_vectors.rows();
    out.receipt.exchange_vectors=out.exchange_vectors.rows(); return out;
}
PeriodicRHFPlan& physical(PeriodicRHFDomainPlan& p) { return p.rhf; }
PeriodicUHFPlan& physical(PeriodicUHFDomainPlan& p) { return p.uhf; }
PeriodicRIUHFPlan& physical(PeriodicRIUHFDomainPlan& p) { return p.uhf; }
PeriodicRIRHFPlan& physical(PeriodicRIRHFDomainPlan& p) { return p.rhf; }
template<class Options> PeriodicKSPlan& physical(PeriodicHybridKSDomainPlan<Options>& p) { return p.ks; }

template<class Plan> void additional_domains(Plan&,const PeriodicSystem&) {}
template<class Options> void additional_domains(PeriodicHybridKSDomainPlan<Options>& p,const PeriodicSystem& system) {
    p.xc_image_candidates=real_box(system.lattice,p.xc_image_cutoff_bohr);
    Count depth=1;for(Count n=p.xc_image_candidates;n>1;n>>=1)++depth;
    p.candidate_visits=add(p.candidate_visits,p.xc_image_candidates);
    p.domain_scalar_terms=add(p.domain_scalar_terms,add(mul(512,p.xc_image_candidates),mul(256,mul(depth,p.xc_image_candidates))));
    p.domain_kernel_terms=mul(32,p.domain_scalar_terms);
    p.domain_live_bytes=add(p.domain_live_bytes,add(32768,mul(256,p.xc_image_candidates)));
}
template<class Plan> void additional_work(Plan&) {}
template<class Options> void additional_work(PeriodicHybridKSDomainPlan<Options>& p) {
    p.primitive_points=p.ks.primitive_points;p.primitive_work=p.ks.primitive_work;
}

template<class Plan, class Options, class Planner> Plan plan_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const Options& options,const ResourceBudget& budget,Planner planner,Plan p={},bool fitted=false) {
    if(system.dim!=3 || !system.lattice.allFinite() || !system.lattice.inverse().allFinite()
       || !std::isfinite(system.lattice.determinant()) || system.lattice.determinant()==0)
        throw std::invalid_argument("MANTA periodic cutoff domains require a finite nonsingular 3D lattice");
    if(!std::isfinite(cutoff) || cutoff<=0 || cutoff>std::numeric_limits<double>::max()/2
       || !options.pair_cutoff_bohr || !std::isfinite(*options.pair_cutoff_bohr) || *options.pair_cutoff_bohr<=0)
        throw std::invalid_argument("MANTA periodic cutoff domains require positive finite reciprocal and physical pair cutoffs");
    const Count nk=mesh.size(),pairs=square(nk);
    p.momenta=nk; p.reciprocal_energy_cutoff=cutoff; p.pair_cutoff_bohr=*options.pair_cutoff_bohr;
    // Admit the count-only metadata scans before O(nshell^2) center work or
    // O(nk) certified-box planning. No source arrays exist at this point.
    const Count preliminary=add(32768,add(mul(8192,nk),add(mul(128,square(basis.nshells())),mul(128,pairs))));
    cap(preliminary,budget.scalar_terms,"MANTA periodic domain planning scalar-work budget exceeded");
    cap(mul(32,preliminary),budget.kernel_terms,"MANTA periodic domain planning kernel-work budget exceeded");
    cap(add(budget.other_live_bytes,add(32768,mul(2048,nk))),budget.memory_bytes,
        "MANTA periodic domain planning memory budget exceeded");
    census(basis,6); validate_basis(basis);
    p.translation_radius_bohr=p.pair_cutoff_bohr+max_shell_offset(basis.libint(),basis.libint());
    if(!std::isfinite(p.translation_radius_bohr*p.translation_radius_bohr))
        throw std::invalid_argument("MANTA periodic domain translation radius is not representable");
    p.translation_candidates=real_box(system.lattice,p.translation_radius_bohr);
    const auto reciprocal=system.reciprocal_lattice(); Count sum=0;
    for(Count q=0;q<nk;++q) {
        const Count count=numeric(reciprocal,mesh,q,cutoff).enumeration.candidate_count;
        if(q==0) p.hartree_candidates=count;
        if(!fitted || q!=0) p.maximum_exchange_candidates=std::max(p.maximum_exchange_candidates,count);
        sum=add(sum,count);
    }
    p.exchange_candidates=mul(nk,fitted ? sum-p.hartree_candidates : sum);
    // Count, exact conjugacy audit, one G replay and nk replays per transfer.
    p.candidate_visits=add(mul(3,sum),add(p.hartree_candidates,p.exchange_candidates));
    Count sort_depth=1; for(Count n=p.translation_candidates;n>1;n>>=1) ++sort_depth;
    p.domain_scalar_terms=add(preliminary,add(mul(512,p.candidate_visits),
        mul(256,mul(p.translation_candidates,sort_depth))));
    p.domain_kernel_terms=mul(32,p.domain_scalar_terms);
    // Include candidate-reserved shared cell storage and stable-sort scratch,
    // native domain owners, owned Python exports and per-transfer plans.
    p.domain_live_bytes=add(32768,add(mul(2048,nk),add(mul(32,add(pairs,1)),
        add(mul(256,p.translation_candidates),mul(96,add(p.hartree_candidates,p.exchange_candidates))))));
    additional_domains(p,system);
    cap(p.domain_scalar_terms,budget.scalar_terms,"MANTA periodic domain scalar-work budget exceeded");
    cap(p.domain_kernel_terms,budget.kernel_terms,"MANTA periodic domain kernel-work budget exceeded");
    cap(add(budget.other_live_bytes,p.domain_live_bytes),budget.memory_bytes,"MANTA periodic domain memory budget exceeded");
    // Box counts also bound every downstream shell call and complete SCF.
    auto& scf=physical(p);scf=planner(p,remaining(budget,p));
    p.modeled_peak_bytes=scf.modeled_peak_bytes;p.shell_calls=scf.shell_calls;p.primitive_pairs=scf.primitive_pairs;
    p.scalar_terms=add(p.domain_scalar_terms,scf.scalar_terms);p.kernel_terms=add(p.domain_kernel_terms,scf.kernel_terms);
    additional_work(p);
    return p;
}
} // namespace

PeriodicRHFDomainPlan plan_periodic_rhf_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicRHFOptions& options,const ResourceBudget& budget) {
    return plan_domains<PeriodicRHFDomainPlan>(basis,system,mesh,cutoff,options,budget,[&](const auto& p,const auto& leaf) {
        return plan_periodic_rhf(basis,system,mesh,p.hartree_candidates,p.maximum_exchange_candidates,p.translation_candidates,options,leaf);
    });
}
PeriodicUHFDomainPlan plan_periodic_uhf_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicUHFOptions& options,const ResourceBudget& budget) {
    return plan_domains<PeriodicUHFDomainPlan>(basis,system,mesh,cutoff,options,budget,[&](const auto& p,const auto& leaf) {
        return plan_periodic_uhf(basis,system,mesh,p.hartree_candidates,p.maximum_exchange_candidates,p.translation_candidates,options,leaf);
    });
}
PeriodicRIUHFDomainPlan plan_periodic_ri_uhf_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,double cutoff,const PeriodicRIUHFOptions& options,const ResourceBudget& budget) {
    return plan_domains<PeriodicRIUHFDomainPlan>(basis,system,mesh,cutoff,options,budget,[&](const auto& p,const auto& leaf) {
        return plan_periodic_ri_uhf(basis,auxiliary,system,mesh,p.hartree_candidates,p.maximum_exchange_candidates,
            p.translation_candidates,options,leaf);
    },PeriodicRIUHFDomainPlan{},options.exchange_algorithm=="ri");
}
PeriodicRIUHFDomains build_periodic_ri_uhf_domains(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,double cutoff,const PeriodicRIUHFOptions& options,const ResourceBudget& budget) {
    return generate<PeriodicRIUHFDomains>(basis,system,mesh,
        plan_periodic_ri_uhf_from_cutoffs(basis,auxiliary,system,mesh,cutoff,options,budget),options.exchange_algorithm=="ri");
}
PeriodicRIUHFResult run_periodic_ri_uhf_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,double cutoff,const PeriodicRIUHFOptions& options,const ResourceBudget& budget) {
    const auto domains=build_periodic_ri_uhf_domains(basis,auxiliary,system,mesh,cutoff,options,budget);
    const auto& p=domains.plan; const auto& d=domains.receipt;
    auto result=run_periodic_ri_uhf(basis,auxiliary,system,mesh,domains.reciprocal_vectors,domains.exchange_vectors,
        domains.transfer_offsets,domains.translations,options,remaining(budget,p));
    result.plan.domain_live_bytes=p.domain_live_bytes; result.plan.domain_scalar_terms=p.domain_scalar_terms;
    result.plan.domain_kernel_terms=p.domain_kernel_terms;
    result.plan.scalar_terms=add(result.plan.scalar_terms,p.domain_scalar_terms);
    result.plan.kernel_terms=add(result.plan.kernel_terms,p.domain_kernel_terms);
    auto& r=result.receipt;
    r.domain_selection=options.exchange_algorithm=="ri"
        ? "physical-pair-sphere;certified-reciprocal-energy-sphere;ri-ket-minus-bra;empty-pair-diagonal"
        : "physical-pair-sphere;certified-reciprocal-energy-sphere;exact-target-minus-source;explicit-pair-diagonal";
    r.domain_identity=d.identity; r.reciprocal_energy_cutoff=cutoff;
    r.domain_translations=d.translations; r.domain_hartree_vectors=d.hartree_vectors;
    r.domain_exchange_vectors=d.exchange_vectors; r.domain_conjugacy_vectors=d.conjugacy_vectors;
    return result;
}
PeriodicRIRHFDomainPlan plan_periodic_ri_rhf_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,double cutoff,const PeriodicRIRHFOptions& options,const ResourceBudget& budget) {
    return plan_domains<PeriodicRIRHFDomainPlan>(basis,system,mesh,cutoff,options,budget,[&](const auto& p,const auto& leaf) {
        return plan_periodic_ri_rhf(basis,auxiliary,system,mesh,p.hartree_candidates,p.maximum_exchange_candidates,
            p.translation_candidates,options,leaf);
    },PeriodicRIRHFDomainPlan{},options.exchange_algorithm=="ri");
}
PeriodicRIRHFDomains build_periodic_ri_rhf_domains(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,double cutoff,const PeriodicRIRHFOptions& options,const ResourceBudget& budget) {
    return generate<PeriodicRIRHFDomains>(basis,system,mesh,
        plan_periodic_ri_rhf_from_cutoffs(basis,auxiliary,system,mesh,cutoff,options,budget),options.exchange_algorithm=="ri");
}
PeriodicRIRHFResult run_periodic_ri_rhf_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,double cutoff,const PeriodicRIRHFOptions& options,const ResourceBudget& budget) {
    const auto domains=build_periodic_ri_rhf_domains(basis,auxiliary,system,mesh,cutoff,options,budget);
    const auto& p=domains.plan; const auto& d=domains.receipt;
    auto result=run_periodic_ri_rhf(basis,auxiliary,system,mesh,domains.reciprocal_vectors,domains.exchange_vectors,
        domains.transfer_offsets,domains.translations,options,remaining(budget,p));
    result.plan.domain_live_bytes=p.domain_live_bytes; result.plan.domain_scalar_terms=p.domain_scalar_terms;
    result.plan.domain_kernel_terms=p.domain_kernel_terms;
    result.plan.scalar_terms=add(result.plan.scalar_terms,p.domain_scalar_terms);
    result.plan.kernel_terms=add(result.plan.kernel_terms,p.domain_kernel_terms);
    auto& r=result.receipt;
    r.domain_selection=options.exchange_algorithm=="ri"
        ? "physical-pair-sphere;certified-reciprocal-energy-sphere;ri-ket-minus-bra;empty-pair-diagonal"
        : "physical-pair-sphere;certified-reciprocal-energy-sphere;exact-target-minus-source;explicit-pair-diagonal";
    r.domain_identity=d.identity; r.reciprocal_energy_cutoff=cutoff;
    r.domain_translations=d.translations; r.domain_hartree_vectors=d.hartree_vectors;
    r.domain_exchange_vectors=d.exchange_vectors; r.domain_conjugacy_vectors=d.conjugacy_vectors;
    return result;
}
PeriodicRHFDomains build_periodic_rhf_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicRHFOptions& options,const ResourceBudget& budget) {
    return generate<PeriodicRHFDomains>(basis,system,mesh,plan_periodic_rhf_from_cutoffs(basis,system,mesh,cutoff,options,budget));
}
PeriodicRHFResult run_periodic_rhf_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicRHFOptions& options,const ResourceBudget& budget) {
    const auto domains=build_periodic_rhf_domains(basis,system,mesh,cutoff,options,budget);
    const auto& p=domains.plan; const auto& d=domains.receipt;
    auto result=run_periodic_rhf(basis,system,mesh,domains.reciprocal_vectors,domains.exchange_vectors,
        domains.transfer_offsets,domains.translations,options,remaining(budget,p));
    result.plan.domain_live_bytes=p.domain_live_bytes; result.plan.domain_scalar_terms=p.domain_scalar_terms;
    result.plan.domain_kernel_terms=p.domain_kernel_terms;
    result.plan.scalar_terms=add(result.plan.scalar_terms,p.domain_scalar_terms);
    result.plan.kernel_terms=add(result.plan.kernel_terms,p.domain_kernel_terms);
    auto& r=result.receipt; r.domain_selection="physical-pair-sphere;certified-reciprocal-energy-sphere";
    r.domain_identity=d.identity; r.reciprocal_energy_cutoff=cutoff;
    r.domain_translations=d.translations; r.domain_hartree_vectors=d.hartree_vectors;
    r.domain_exchange_vectors=d.exchange_vectors; r.domain_conjugacy_vectors=d.conjugacy_vectors;
    return result;
}
PeriodicUHFDomains build_periodic_uhf_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicUHFOptions& options,const ResourceBudget& budget) {
    return generate<PeriodicUHFDomains>(basis,system,mesh,plan_periodic_uhf_from_cutoffs(basis,system,mesh,cutoff,options,budget));
}
PeriodicUHFResult run_periodic_uhf_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,double cutoff,const PeriodicUHFOptions& options,const ResourceBudget& budget) {
    const auto domains=build_periodic_uhf_domains(basis,system,mesh,cutoff,options,budget);
    const auto& p=domains.plan; const auto& d=domains.receipt;
    auto result=run_periodic_uhf(basis,system,mesh,domains.reciprocal_vectors,domains.exchange_vectors,
        domains.transfer_offsets,domains.translations,options,remaining(budget,p));
    result.plan.domain_live_bytes=p.domain_live_bytes; result.plan.domain_scalar_terms=p.domain_scalar_terms;
    result.plan.domain_kernel_terms=p.domain_kernel_terms;
    result.plan.scalar_terms=add(result.plan.scalar_terms,p.domain_scalar_terms);
    result.plan.kernel_terms=add(result.plan.kernel_terms,p.domain_kernel_terms);
    auto& r=result.receipt; r.domain_selection="physical-pair-sphere;certified-reciprocal-energy-sphere";
    r.domain_identity=d.identity; r.reciprocal_energy_cutoff=cutoff;
    r.domain_translations=d.translations; r.domain_hartree_vectors=d.hartree_vectors;
    r.domain_exchange_vectors=d.exchange_vectors; r.domain_conjugacy_vectors=d.conjugacy_vectors;
    return result;
}
namespace {
template<class Options,class Planner>
PeriodicHybridKSDomainPlan<Options> plan_hybrid_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const Options& options,const ResourceBudget& budget,Planner planner,bool fitted=false) {
    if(!std::isfinite(xc_radius) || xc_radius<0 || !std::isfinite(xc_radius*xc_radius))
        throw std::invalid_argument("MANTA hybrid KS requires a nonnegative finite XC image radius");
    PeriodicHybridKSDomainPlan<Options> initial;initial.xc_image_cutoff_bohr=xc_radius;
    return plan_domains(basis,system,mesh,cutoff,options.ks,budget,[&](const auto& p,const auto& leaf) {
        return planner(basis,system,mesh,f,p.hartree_candidates,p.maximum_exchange_candidates,
            p.translation_candidates,p.xc_image_candidates,ngrid,options,leaf);
    },initial,fitted);
}
template<class Options,class Planner>
PeriodicHybridKSDomains<Options> build_hybrid_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const Options& options,const ResourceBudget& budget,Planner planner,const char* identity_prefix,bool fitted=false) {
    const auto plan=planner(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget);
    auto out=generate<PeriodicHybridKSDomains<Options>>(basis,system,mesh,plan,fitted);
    const auto images=direct_lattice_cells(system,xc_radius);
    cap(images.size(),plan.xc_image_candidates,"MANTA hybrid XC images exceeded admitted extent");
    out.xc_images.resize(images.size(),3);
    for(Count i=0;i<images.size();++i)out.xc_images.row(i)=images[i].r_cart.transpose();
    // Preserve the identical HF pair/G/Q geometry identity as a child source.
    out.receipt.pair_identity=out.receipt.identity;out.receipt.xc_images=images.size();
    Digest xc;xc.word(1);xc.matrix(system.lattice);xc.real(xc_radius);xc.matrix(out.xc_images);
    out.receipt.xc_image_identity="manta.periodic-xc-image-domain.v1:"+xc.hash.finish_hex();
    Digest combined;combined.word(1);combined.text(out.receipt.pair_identity);combined.text(out.receipt.xc_image_identity);
    out.receipt.identity=identity_prefix+combined.hash.finish_hex();
    return out;
}
template<class Options,class Builder,class Runner>
PeriodicKSResult run_hybrid_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,const Eigen::MatrixXd& points,
    const std::vector<double>& weights,const Options& options,const ResourceBudget& budget,Builder builder,Runner runner) {
    const auto domains=builder(basis,system,mesh,f,cutoff,xc_radius,points.rows(),options,budget);
    const auto& p=domains.plan;const auto& d=domains.receipt;
    auto result=runner(basis,system,mesh,f,domains.reciprocal_vectors,domains.exchange_vectors,
        domains.transfer_offsets,domains.translations,domains.xc_images,points,weights,options,remaining(budget,p));
    result.plan.domain_live_bytes=p.domain_live_bytes;result.plan.domain_scalar_terms=p.domain_scalar_terms;
    result.plan.domain_kernel_terms=p.domain_kernel_terms;
    result.plan.scalar_terms=add(result.plan.scalar_terms,p.domain_scalar_terms);
    result.plan.kernel_terms=add(result.plan.kernel_terms,p.domain_kernel_terms);
    auto& r=result.receipt;r.domain_selection="physical-pair-sphere;certified-reciprocal-energy-sphere;canonical-xc-image-sphere";
    r.domain_identity=d.identity;r.reciprocal_energy_cutoff=cutoff;r.xc_image_cutoff_bohr=xc_radius;
    r.domain_translations=d.translations;r.domain_hartree_vectors=d.hartree_vectors;r.domain_exchange_vectors=d.exchange_vectors;
    r.domain_conjugacy_vectors=d.conjugacy_vectors;r.domain_xc_images=d.xc_images;return result;
}
} // namespace
PeriodicGlobalHybridKSDomainPlan plan_periodic_global_hybrid_ks_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    return plan_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,plan_periodic_global_hybrid_ks);
}
PeriodicGlobalHybridKSDomains build_periodic_global_hybrid_ks_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    return build_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,
        plan_periodic_global_hybrid_ks_from_cutoffs,"manta.periodic-global-hybrid-ks-cutoff-domains.v1:");
}
PeriodicKSResult run_periodic_global_hybrid_ks_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,const Eigen::MatrixXd& points,
    const std::vector<double>& weights,const PeriodicGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    return run_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,points,weights,options,budget,
        build_periodic_global_hybrid_ks_domains,run_periodic_global_hybrid_ks);
}
PeriodicRangeSeparatedKSDomainPlan plan_periodic_range_separated_ks_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    return plan_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,plan_periodic_range_separated_ks);
}
PeriodicRangeSeparatedKSDomains build_periodic_range_separated_ks_domains(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    return build_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,
        plan_periodic_range_separated_ks_from_cutoffs,"manta.periodic-range-separated-ks-cutoff-domains.v1:");
}
PeriodicKSResult run_periodic_range_separated_ks_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,const Eigen::MatrixXd& points,
    const std::vector<double>& weights,const PeriodicRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    return run_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,points,weights,options,budget,
        build_periodic_range_separated_ks_domains,run_periodic_range_separated_ks);
}
PeriodicRIGlobalHybridKSDomainPlan plan_periodic_ri_global_hybrid_ks_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRIGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto planner=[&](const BasisSet& b,auto&&... args){return plan_periodic_ri_global_hybrid_ks(b,auxiliary,args...);};
    return plan_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,planner,options.exchange_algorithm=="ri");
}
PeriodicRIGlobalHybridKSDomains build_periodic_ri_global_hybrid_ks_domains(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRIGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto planner=[&](const BasisSet& b,auto&&... args){return plan_periodic_ri_global_hybrid_ks_from_cutoffs(b,auxiliary,args...);};
    return build_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,planner,
        "manta.periodic-ri-global-hybrid-ks-cutoff-domains.v1:",options.exchange_algorithm=="ri");
}
PeriodicKSResult run_periodic_ri_global_hybrid_ks_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const Eigen::MatrixXd& points,const std::vector<double>& qw,const PeriodicRIGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto builder=[&](const BasisSet& b,auto&&... args){return build_periodic_ri_global_hybrid_ks_domains(b,auxiliary,args...);};
    const auto runner=[&](const BasisSet& b,auto&&... args){return run_periodic_ri_global_hybrid_ks(b,auxiliary,args...);};
    auto result=run_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,points,qw,options,budget,builder,runner);
    result.receipt.domain_selection+=options.exchange_algorithm=="ri"
        ? ";ri-ket-minus-bra;empty-pair-diagonal" : ";exact-target-minus-source;explicit-pair-diagonal";
    return result;
}
PeriodicRIRangeSeparatedKSDomainPlan plan_periodic_ri_range_separated_ks_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRIRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto planner=[&](const BasisSet& b,auto&&... args){return plan_periodic_ri_range_separated_ks(b,auxiliary,args...);};
    return plan_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,planner,options.exchange_algorithm=="ri");
}
PeriodicRIRangeSeparatedKSDomains build_periodic_ri_range_separated_ks_domains(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRIRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto planner=[&](const BasisSet& b,auto&&... args){return plan_periodic_ri_range_separated_ks_from_cutoffs(b,auxiliary,args...);};
    return build_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,planner,
        "manta.periodic-ri-range-separated-ks-cutoff-domains.v1:",options.exchange_algorithm=="ri");
}
PeriodicKSResult run_periodic_ri_range_separated_ks_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const Eigen::MatrixXd& points,const std::vector<double>& qw,const PeriodicRIRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto builder=[&](const BasisSet& b,auto&&... args){return build_periodic_ri_range_separated_ks_domains(b,auxiliary,args...);};
    const auto runner=[&](const BasisSet& b,auto&&... args){return run_periodic_ri_range_separated_ks(b,auxiliary,args...);};
    auto result=run_hybrid_domains(basis,system,mesh,f,cutoff,xc_radius,points,qw,options,budget,builder,runner);
    result.receipt.domain_selection+=options.exchange_algorithm=="ri"
        ? ";ri-ket-minus-bra;empty-pair-diagonal" : ";exact-target-minus-source;explicit-pair-diagonal";
    return result;
}
} // namespace vibeqc::manta
