#include "vibeqc/manta_periodic_ks_domains.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/periodic_reciprocal_source.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using namespace admission;
using Label=std::array<std::int64_t,3>;
using Numeric=detail::PeriodicReciprocalNumericPlan;
Numeric numeric(const PeriodicSystem& s,double cutoff) {
    return detail::prepare_periodic_reciprocal_numeric_source(s.reciprocal_lattice(),Eigen::Vector3d::Zero(),
        cutoff,true,std::numeric_limits<Count>::max());
}
Count real_box(const Eigen::Matrix3d& a,double radius) {
    const double volume=std::abs(a.determinant());Count count=1;
    for(int i=0;i<3;++i) {
        const double cross=a.col((i+1)%3).cross(a.col((i+2)%3)).norm(),distance=volume/cross,extent=std::ceil(radius/distance);
        if(!std::isfinite(cross) || cross<1e-14 || !std::isfinite(distance) || distance<1e-14
           || !std::isfinite(extent) || extent>=std::numeric_limits<int>::max())
            throw std::invalid_argument("MANTA KS domain lattice/image extent is unsupported");
        count=mul(count,add(mul(2,static_cast<Count>(extent)),1));
    }
    return count;
}
ResourceBudget remaining(const ResourceBudget& b,const PeriodicKSDomainPlan& p) {
    auto result=b;result.other_live_bytes=add(b.other_live_bytes,p.domain_live_bytes);
    result.scalar_terms-=p.domain_scalar_terms;result.kernel_terms-=p.domain_kernel_terms;return result;
}
void ignore(const Label&,const std::array<double,5>&,void*) {}
struct Digest {
    detail::Sha256 hash;
    void word(Count v) {std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8);}
    void real(double v) {if(v==0)v=0.;Count bits;std::memcpy(&bits,&v,8);word(bits);}
    template<class Matrix> void matrix(const Matrix& a) {
        word(a.rows());word(a.cols());for(Eigen::Index i=0;i<a.rows();++i)for(Eigen::Index j=0;j<a.cols();++j)real(a(i,j));
    }
};
PeriodicKSDomains generate(const BasisSet& basis,const PeriodicSystem& system,const PeriodicKSDomainPlan& plan) {
    PeriodicKSDomains out;out.plan=plan;
    const auto reciprocal=numeric(system,plan.reciprocal_energy_cutoff);
    const Count ng=detail::visit_periodic_reciprocal_numeric_source(reciprocal.geometry,ignore,nullptr);
    if(!ng) throw std::invalid_argument("MANTA KS reciprocal cutoff contains no nonzero Hartree vectors");
    out.receipt.conjugacy_vectors=detail::require_periodic_reciprocal_numeric_conjugacy(reciprocal.geometry,reciprocal.geometry,{0,0,0},ng);
    const auto cells=pair_complete_cells(system,plan.pair_cutoff_bohr,basis.libint(),basis.libint());
    const auto images=direct_lattice_cells(system,plan.xc_image_cutoff_bohr);
    cap(cells.size(),plan.translation_candidates,"MANTA KS pair domain exceeded admitted extent");
    cap(images.size(),plan.xc_image_candidates,"MANTA KS XC image domain exceeded admitted extent");
    cap(ng,plan.hartree_candidates,"MANTA KS reciprocal domain exceeded admitted extent");
    out.translations.resize(cells.size(),3);out.xc_images.resize(images.size(),3);out.reciprocal_vectors.resize(ng,3);
    for(Count i=0;i<cells.size();++i)out.translations.row(i)=cells[i].r_cart.transpose();
    for(Count i=0;i<images.size();++i)out.xc_images.row(i)=images[i].r_cart.transpose();
    struct State { md::Matrix& out;Count next=0; } state{out.reciprocal_vectors};
    const Count filled=detail::visit_periodic_reciprocal_numeric_source(reciprocal.geometry,
        [](const Label&,const std::array<double,5>& lanes,void* opaque) {
            auto& state=*static_cast<State*>(opaque);
            if(state.next>=static_cast<Count>(state.out.rows())) throw std::logic_error("MANTA KS reciprocal replay exceeded admitted extent");
            for(int i=0;i<3;++i)state.out(state.next,i)=lanes[i];++state.next;
        },&state);
    if(filled!=ng || state.next!=ng)throw std::logic_error("MANTA KS reciprocal replay changed accepted count");
    Digest digest;digest.word(1);digest.matrix(system.lattice);digest.real(plan.pair_cutoff_bohr);
    digest.real(plan.reciprocal_energy_cutoff);digest.real(plan.xc_image_cutoff_bohr);
    digest.word(basis.nshells());for(const auto& shell:basis.libint())for(double x:shell.O)digest.real(x);
    digest.matrix(out.translations);digest.matrix(out.reciprocal_vectors);digest.matrix(out.xc_images);
    out.receipt.identity="manta.periodic-ks-cutoff-domains.v1:"+digest.hash.finish_hex();
    out.receipt.translations=cells.size();out.receipt.hartree_vectors=ng;out.receipt.xc_images=images.size();return out;
}
template<class Planner>
PeriodicKSDomainPlan plan_pure_domains(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    const Functional& f,double cutoff,double xc_radius,Count ngrid,const PeriodicKSOptions& options,const ResourceBudget& budget,Planner planner) {
    if(system.dim!=3 || !system.lattice.allFinite() || !system.lattice.inverse().allFinite()
       || !std::isfinite(system.lattice.determinant()) || system.lattice.determinant()==0)
        throw std::invalid_argument("MANTA KS cutoff domains require a finite nonsingular 3D lattice");
    if(!std::isfinite(cutoff) || cutoff<=0 || cutoff>std::numeric_limits<double>::max()/2
       || !options.pair_cutoff_bohr || !std::isfinite(*options.pair_cutoff_bohr) || *options.pair_cutoff_bohr<=0
       || !std::isfinite(xc_radius) || xc_radius<0 || !std::isfinite(xc_radius*xc_radius))
        throw std::invalid_argument("MANTA KS domains require positive reciprocal/pair cutoffs and a nonnegative finite XC image radius");
    PeriodicKSDomainPlan p;p.reciprocal_energy_cutoff=cutoff;p.pair_cutoff_bohr=*options.pair_cutoff_bohr;p.xc_image_cutoff_bohr=xc_radius;
    // Bound center scans before O(nshell^2) work, then bound every enumeration
    // and the complete physical solver before any numerical source allocation.
    const Count preliminary=add(32768,mul(128,square(basis.nshells())));
    cap(preliminary,budget.scalar_terms,"MANTA KS domain planning scalar-work budget exceeded");
    cap(mul(32,preliminary),budget.kernel_terms,"MANTA KS domain planning kernel-work budget exceeded");
    cap(add(budget.other_live_bytes,32768),budget.memory_bytes,"MANTA KS domain planning memory budget exceeded");
    census(basis,4);validate_basis(basis);
    p.translation_radius_bohr=p.pair_cutoff_bohr+max_shell_offset(basis.libint(),basis.libint());
    if(!std::isfinite(p.translation_radius_bohr*p.translation_radius_bohr))throw std::invalid_argument("MANTA KS pair domain radius is not representable");
    p.translation_candidates=real_box(system.lattice,p.translation_radius_bohr);
    p.xc_image_candidates=real_box(system.lattice,xc_radius);
    p.hartree_candidates=numeric(system,cutoff).enumeration.candidate_count;
    p.candidate_visits=add(mul(3,p.hartree_candidates),add(p.translation_candidates,p.xc_image_candidates));
    Count depth=1;for(Count n=std::max(p.translation_candidates,p.xc_image_candidates);n>1;n>>=1)++depth;
    p.domain_scalar_terms=add(preliminary,add(mul(512,p.candidate_visits),mul(256,mul(depth,add(p.translation_candidates,p.xc_image_candidates)))));
    p.domain_kernel_terms=mul(32,p.domain_scalar_terms);
    // Reserved cell boxes, sorting scratch, numerical source owners and owned
    // exports can coexist. No K-transfer tables or k^2 metadata are needed.
    p.domain_live_bytes=add(32768,add(mul(256,add(p.translation_candidates,p.xc_image_candidates)),mul(96,p.hartree_candidates)));
    cap(p.domain_scalar_terms,budget.scalar_terms,"MANTA KS domain scalar-work budget exceeded");
    cap(p.domain_kernel_terms,budget.kernel_terms,"MANTA KS domain kernel-work budget exceeded");
    cap(add(budget.other_live_bytes,p.domain_live_bytes),budget.memory_bytes,"MANTA KS domain memory budget exceeded");
    p.ks=planner(p,remaining(budget,p));
    p.modeled_peak_bytes=p.ks.modeled_peak_bytes;p.shell_calls=p.ks.shell_calls;p.primitive_work=p.ks.primitive_work;
    p.scalar_terms=add(p.domain_scalar_terms,p.ks.scalar_terms);p.kernel_terms=add(p.domain_kernel_terms,p.ks.kernel_terms);return p;
}
} // namespace
PeriodicKSDomainPlan plan_periodic_ks_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    const Functional& f,double cutoff,double xc_radius,Count ngrid,const PeriodicKSOptions& options,const ResourceBudget& budget) {
    return plan_pure_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget,[&](const auto& p,const auto& leaf) {
        return plan_periodic_ks(basis,system,mesh,f,p.hartree_candidates,p.translation_candidates,p.xc_image_candidates,ngrid,options,leaf);
    });
}
PeriodicKSDomains build_periodic_ks_domains(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    const Functional& f,double cutoff,double xc_radius,Count ngrid,const PeriodicKSOptions& options,const ResourceBudget& budget) {
    return generate(basis,system,plan_periodic_ks_from_cutoffs(basis,system,mesh,f,cutoff,xc_radius,ngrid,options,budget));
}
namespace {
template<class Options,class Builder,class Runner>
PeriodicKSResult run_pure_domains(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    const Functional& f,double cutoff,double xc_radius,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const Options& options,const ResourceBudget& budget,Builder builder,Runner runner) {
    const auto domains=builder(basis,system,mesh,f,cutoff,xc_radius,points.rows(),options,budget);
    const auto& p=domains.plan;const auto& d=domains.receipt;
    auto result=runner(basis,system,mesh,f,domains.reciprocal_vectors,domains.translations,domains.xc_images,points,qw,options,remaining(budget,p));
    result.plan.domain_live_bytes=p.domain_live_bytes;result.plan.domain_scalar_terms=p.domain_scalar_terms;result.plan.domain_kernel_terms=p.domain_kernel_terms;
    result.plan.scalar_terms=add(result.plan.scalar_terms,p.domain_scalar_terms);result.plan.kernel_terms=add(result.plan.kernel_terms,p.domain_kernel_terms);
    auto& r=result.receipt;r.domain_selection="physical-pair-sphere;certified-reciprocal-energy-sphere;canonical-xc-image-sphere";
    r.domain_identity=d.identity;r.reciprocal_energy_cutoff=cutoff;r.xc_image_cutoff_bohr=xc_radius;
    r.domain_translations=d.translations;r.domain_hartree_vectors=d.hartree_vectors;r.domain_xc_images=d.xc_images;r.domain_conjugacy_vectors=d.conjugacy_vectors;
    return result;
}
} // namespace
PeriodicKSResult run_periodic_ks_from_cutoffs(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    const Functional& f,double cutoff,double xc_radius,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const PeriodicKSOptions& options,const ResourceBudget& budget) {
    return run_pure_domains(basis,system,mesh,f,cutoff,xc_radius,points,qw,options,budget,build_periodic_ks_domains,run_periodic_ks);
}
PeriodicKSDomainPlan plan_periodic_ri_ks_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRIKSOptions& options,const ResourceBudget& budget) {
    return plan_pure_domains(basis,system,mesh,f,cutoff,xc_radius,ngrid,options.ks,budget,[&](const auto& p,const auto& leaf) {
        return plan_periodic_ri_ks(basis,auxiliary,system,mesh,f,p.hartree_candidates,p.translation_candidates,
            p.xc_image_candidates,ngrid,options,leaf);
    });
}
PeriodicKSDomains build_periodic_ri_ks_domains(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,Count ngrid,
    const PeriodicRIKSOptions& options,const ResourceBudget& budget) {
    return generate(basis,system,plan_periodic_ri_ks_from_cutoffs(basis,auxiliary,system,mesh,f,cutoff,xc_radius,ngrid,options,budget));
}
PeriodicKSResult run_periodic_ri_ks_from_cutoffs(const BasisSet& basis,const BasisSet& auxiliary,
    const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,double cutoff,double xc_radius,
    const Eigen::MatrixXd& points,const std::vector<double>& qw,const PeriodicRIKSOptions& options,const ResourceBudget& budget) {
    const auto builder=[&](const BasisSet& b,auto&&... args){return build_periodic_ri_ks_domains(b,auxiliary,args...);};
    const auto runner=[&](const BasisSet& b,auto&&... args){return run_periodic_ri_ks(b,auxiliary,args...);};
    return run_pure_domains(basis,system,mesh,f,cutoff,xc_radius,points,qw,options,budget,builder,runner);
}
} // namespace vibeqc::manta
