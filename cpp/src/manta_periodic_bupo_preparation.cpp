#include "vibeqc/manta_periodic_bupo_preparation.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "manta_periodic_bupo_hf_inputs.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using namespace admission;
ResourceBudget unlimited(const ResourceBudget& b) {
    auto out=b;out.other_live_bytes=0;
    out.memory_bytes=out.shell_calls=out.primitive_quartets=out.scalar_terms=out.kernel_terms=std::numeric_limits<Count>::max();
    return out;
}
Count cube_cells(Count radius) {
    if(radius>=static_cast<Count>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("MANTA BUPO preparation cube exceeds the supported integer extent");
    const Count side=add(mul(2,radius),1);return mul(square(side),side);
}
std::vector<LatticeIndex> cube(Count radius,bool nonzero=false) {
    std::vector<LatticeIndex> out;out.reserve(cube_cells(radius)-(nonzero?1:0));const auto r=static_cast<int>(radius);
    for(int i=-r;i<=r;++i)for(int j=-r;j<=r;++j)for(int k=-r;k<=r;++k)
        if(!nonzero || i || j || k)out.push_back({i,j,k});
    return out;
}
void charge(PeriodicBupoSourcePreparationPlan& p,const ResourceBudget& b,Count bytes,Count shells,Count primitives,Count scalar,Count kernel) {
    // Existing stage plans include conservative parent/workspace reservations.
    // Summing them deliberately overcounts shared storage, but never drops an
    // owner. Keep that reservation alive through the complete HF run.
    const Count live=add(p.source_live_bytes,bytes),calls=add(p.source_shell_calls,shells),products=add(p.source_primitive_pairs,primitives);
    const Count terms=add(p.source_scalar_terms,scalar),work=add(p.source_kernel_terms,kernel);
    cap(add(b.other_live_bytes,live),b.memory_bytes,"MANTA BUPO preparation cumulative memory budget exceeded");
    cap(calls,b.shell_calls,"MANTA BUPO preparation cumulative shell-call budget exceeded");
    cap(products,b.primitive_quartets,"MANTA BUPO preparation cumulative primitive budget exceeded");
    cap(terms,b.scalar_terms,"MANTA BUPO preparation cumulative scalar budget exceeded");
    cap(work,b.kernel_terms,"MANTA BUPO preparation cumulative kernel budget exceeded");
    p.source_live_bytes=live;p.source_shell_calls=calls;p.source_primitive_pairs=products;p.source_scalar_terms=terms;p.source_kernel_terms=work;
}
ResourceBudget remaining(const ResourceBudget& b,const PeriodicBupoSourcePreparationPlan& p) {
    auto out=b;out.other_live_bytes=add(b.other_live_bytes,p.source_live_bytes);
    out.shell_calls-=p.source_shell_calls;out.primitive_quartets-=p.source_primitive_pairs;
    out.scalar_terms-=p.source_scalar_terms;out.kernel_terms-=p.source_kernel_terms;return out;
}
struct Hash {
    detail::Sha256 value;
    void word(Count n) {std::array<std::uint8_t,8> bytes{};for(int i=0;i<8;++i)bytes[i]=n>>(56-8*i);value.update(bytes.data(),8);}
    void number(double x) {if(x==0)x=0.;Count bits;std::memcpy(&bits,&x,8);word(bits);}
    void text(const std::string& s) {word(s.size());value.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());}
};
} // namespace

PreparedPeriodicBupoSource::PreparedPeriodicBupoSource(const BasisSet& basis,const PeriodicSystem& system,
    const RegularKMesh& mesh,const PeriodicBupoSourceOptions& source_options,Count reciprocal_batch,const ResourceBudget& budget) {
    margin(budget);const auto& o=source_options;
    if(system.dim!=3 || !system.lattice.allFinite() || !system.lattice.inverse().allFinite()
       || !std::isfinite(system.lattice.determinant()) || system.lattice.determinant()==0)
        throw std::invalid_argument("MANTA BUPO preparation requires a finite nonsingular 3D lattice");
    if(!std::isfinite(o.eta) || o.eta<=0 || o.near_cube_radius>o.real_cube_radius)
        throw std::invalid_argument("MANTA BUPO preparation requires positive eta and near cube within real cube");
    if(!o.bupo.bound_infinite_domain || !o.bupo.bound_pair_images)
        throw std::invalid_argument("MANTA BUPO source preparation requires all source bounds");
    if(!reciprocal_batch)throw std::invalid_argument("MANTA BUPO preparation requires a positive reciprocal batch");
    const auto leaf=unlimited(budget);auto& p=plan_;
    const Count ni=cube_cells(o.pair_cube_radius),nr=cube_cells(o.real_cube_radius),ng=cube_cells(o.reciprocal_cube_radius)-1,nn=cube_cells(o.near_cube_radius);
    const Count labels=add(add(ni,nr),add(ng,nn)),nk=mesh.size();
    p.generator_live_bytes=add(65536,add(mul(128,labels),add(mul(256,nk),mul(512,system.unit_cell.size()))));
    p.generator_scalar_terms=add(32768,add(mul(256,labels),add(mul(8192,nk),mul(1024,add(system.unit_cell.size(),basis.nshells())))));
    p.generator_kernel_terms=mul(32,p.generator_scalar_terms);
    charge(p,budget,p.generator_live_bytes,0,0,p.generator_scalar_terms,p.generator_kernel_terms);
    source_options_=source_options;
    // Extents, moments, Ewald coefficients and the immutable admitted partition
    // use the original native producers (contract Sections 5.7-5.14). The
    // Neese2025 molecular BUPO paper does not imply a new periodic derivation.
    p.pairs=plan_periodic_pairs(basis,ni,nk,o.pairs,leaf);
    charge(p,budget,p.pairs.modeled_peak_bytes,p.pairs.shell_calls,p.pairs.primitive_pairs,p.pairs.scalar_terms,p.pairs.kernel_terms);
    const auto images=cube(o.pair_cube_radius),real=cube(o.real_cube_radius),reciprocal=cube(o.reciprocal_cube_radius,true),near=cube(o.near_cube_radius);
    md::Matrix k(nk,3);std::vector<double> weights(nk,1./nk);
    for(Count i=0;i<nk;++i)k.row(i)=(system.reciprocal_lattice()*mesh.fractional_at(i)).transpose();
    auto pairs=std::make_shared<const PeriodicPairSource>(basis,system.lattice,images,k,weights,o.pairs,leaf);
    p.bubbles=PeriodicBubbleTree::plan_source(*pairs,o.bubbles,leaf);
    charge(p,budget,p.bubbles.modeled_peak_bytes,p.bubbles.shell_calls,p.bubbles.primitive_pairs,p.bubbles.scalar_terms,p.bubbles.kernel_terms);
    auto tree=std::make_shared<const PeriodicBubbleTree>(pairs,o.bubbles,leaf);
    p.ewald=PeriodicEwaldSource::plan_source(*pairs,nr,ng,reciprocal_batch,leaf);
    charge(p,budget,p.ewald.modeled_peak_bytes,p.ewald.shell_calls,p.ewald.primitive_products,p.ewald.scalar_terms,p.ewald.kernel_terms);
    auto ewald=std::make_shared<const PeriodicEwaldSource>(pairs,o.eta,real,reciprocal,reciprocal_batch,leaf);
    p.bupo=PeriodicBupoSource::plan_source(*tree,*ewald,nn,o.bupo,leaf);
    charge(p,budget,p.bupo.modeled_peak_bytes,p.bupo.shell_calls,p.bupo.primitive_products,p.bupo.scalar_terms,p.bupo.kernel_terms);
    electronic_=std::make_shared<PeriodicBupoSource>(tree,ewald,near,o.bupo,leaf);
    runtime_budget_=remaining(budget,p);
    p.modeled_peak_bytes=add(budget.other_live_bytes,p.source_live_bytes);
    p.shell_calls=p.source_shell_calls;p.primitive_pairs=p.source_primitive_pairs;
    p.scalar_terms=p.source_scalar_terms;p.kernel_terms=p.source_kernel_terms;
    auto& r=receipt_;r.pair_source_identity=pairs->receipt().source_identity;r.tree_identity=tree->source_identity();
    r.kernel_identity=ewald->kernel_identity();r.electronic_source_identity=electronic_->source_identity();
    r.pair_cube_radius=o.pair_cube_radius;r.real_cube_radius=o.real_cube_radius;r.reciprocal_cube_radius=o.reciprocal_cube_radius;r.near_cube_radius=o.near_cube_radius;
    r.pairs=pairs->receipt().pairs;r.nodes=tree->nodes().size();r.leaf_capacity=o.bubbles.leaf_capacity;r.order=o.pairs.order;
    r.branching=o.bubbles.branching;r.maximum_depth=o.bubbles.maximum_depth;r.eta=o.eta;r.opening=o.bupo.opening;r.integral_tolerance=o.bupo.integral_tolerance;
    r.pair_cutoff_bohr=o.pairs.pair_cutoff_bohr;r.pointwise_tolerance=o.pairs.extents.pointwise_tolerance;r.tail_mass_tolerance=o.pairs.extents.tail_mass_tolerance;
    r.minimum_radius=o.pairs.extents.minimum_radius;r.maximum_radius=o.pairs.extents.maximum_radius;
    Hash hash;hash.text("manta.periodic-prepared-bupo-source.v1;fixed-source;no-method-or-symmetry-authority");
    hash.text(r.electronic_source_identity);hash.word(reciprocal_batch);
    r.identity=hash.value.finish_hex();
}

PreparedPeriodicBupoHF::PreparedPeriodicBupoHF(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    double cutoff,const PeriodicBupoHFSourceOptions& source_options,const PeriodicBupoHFOptions& options,const ResourceBudget& budget)
    :mesh_(mesh) {
    // Method validation still precedes source allocation. Electronic-source
    // preparation itself has no occupations, exchange or nuclear ownership.
    margin(budget);
    if(!source_options.bupo.bound_infinite_domain || !source_options.bupo.bound_pair_images || !options.bound_exchange || options.exchange_bound_cube_radius!=0)
        throw std::invalid_argument("MANTA BUPO preparation requires all source/exchange bounds and a factory-owned zero input exchange cube");
    if((options.spin_mode!="restricted" && options.spin_mode!="unrestricted")
       || (options.exchange_q0!="strict-zero-mode" && options.exchange_q0!="BvK-ewald"))
        throw std::invalid_argument("MANTA BUPO preparation spin and exchange zero-mode policies are invalid");
    if(!std::isfinite(cutoff) || cutoff<=0 || cutoff>std::numeric_limits<double>::max()/2)
        throw std::invalid_argument("MANTA BUPO preparation exchange cutoff must be finite positive");
    bupo_hf_detail::validate_inputs(basis,system,options);
    PreparedPeriodicBupoSource prepared(basis,system,mesh,source_options,options.reciprocal_batch,budget);
    auto& p=plan_;static_cast<PeriodicBupoSourcePreparationPlan&>(p)=prepared.plan();
    electronic_=prepared.source();receipt_=prepared.receipt();
    system_.dim=system.dim;system_.lattice=system.lattice;system_.unit_cell=system.unit_cell;
    system_.charge=system.charge;system_.multiplicity=system.multiplicity;
    options_=options;source_options_=source_options;
    runtime_budget_=prepared.remaining_budget();
    domains_=build_periodic_bupo_hf_exchange_domains(*electronic_,system_,mesh_,cutoff,options_,runtime_budget_);
    p.exchange_domains=domains_.plan;const auto& d=p.exchange_domains;
    p.modeled_peak_bytes=d.modeled_peak_bytes;p.shell_calls=add(p.source_shell_calls,d.shell_calls);p.primitive_pairs=add(p.source_primitive_pairs,d.primitive_pairs);
    p.scalar_terms=add(p.source_scalar_terms,d.scalar_terms);p.kernel_terms=add(p.source_kernel_terms,d.kernel_terms);
    runtime_budget_.other_live_bytes=add(runtime_budget_.other_live_bytes,d.domain_live_bytes);
    runtime_budget_.scalar_terms-=d.domain_scalar_terms;runtime_budget_.kernel_terms-=d.domain_kernel_terms;
    options_.exchange_bound_cube_radius=d.exchange_bound_cube_radius;
    auto& r=receipt_;
    Hash hash;hash.text("manta.periodic-prepared-bupo-hf.v1;fixed-source;no-state-or-symmetry-authority");hash.text(r.electronic_source_identity);hash.text(domains_.receipt.identity);
    hash.word(system.unit_cell.size());for(const auto& atom:system.unit_cell){hash.word(atom.Z);for(auto x:atom.xyz)hash.number(x);}
    hash.word(system.charge);hash.word(system.multiplicity);hash.text(options.spin_mode);hash.text(options.exchange_q0);
    hash.word(options.madelung_alpha.has_value());if(options.madelung_alpha)hash.number(*options.madelung_alpha);hash.number(options.madelung_tolerance);
    r.identity=hash.value.finish_hex();
}

PeriodicBupoHFResult PreparedPeriodicBupoHF::run() const {
    std::unique_lock<std::mutex> lease(run_mutex_,std::try_to_lock);
    if(!lease.owns_lock())throw std::runtime_error("MANTA prepared BUPO HF source already has an active run");
    auto out=run_periodic_bupo_hf(electronic_,system_,mesh_,domains_.exchange_vectors,domains_.transfer_offsets,options_,runtime_budget_);
    const auto& p=plan_;const auto& d=p.exchange_domains;
    out.plan.source_live_bytes=p.source_live_bytes;out.plan.source_shell_calls=p.source_shell_calls;out.plan.source_primitive_pairs=p.source_primitive_pairs;
    out.plan.source_scalar_terms=p.source_scalar_terms;out.plan.source_kernel_terms=p.source_kernel_terms;
    out.plan.shell_calls=add(out.plan.shell_calls,p.source_shell_calls);out.plan.primitive_pairs=add(out.plan.primitive_pairs,p.source_primitive_pairs);
    out.plan.domain_live_bytes=d.domain_live_bytes;out.plan.domain_scalar_terms=d.domain_scalar_terms;out.plan.domain_kernel_terms=d.domain_kernel_terms;
    out.plan.scalar_terms=add(out.plan.scalar_terms,add(p.source_scalar_terms,d.domain_scalar_terms));out.plan.kernel_terms=add(out.plan.kernel_terms,add(p.source_kernel_terms,d.domain_kernel_terms));
    auto& r=out.receipt;r.preparation=receipt_;r.domain_selection="generated-bupo-cubes;certified-reciprocal-energy-sphere";
    r.domain_identity=domains_.receipt.identity;r.reciprocal_energy_cutoff=d.reciprocal_energy_cutoff;r.domain_translations=out.plan.cells;
    r.domain_exchange_vectors=domains_.receipt.exchange_vectors;r.domain_conjugacy_vectors=domains_.receipt.conjugacy_vectors;return out;
}
} // namespace vibeqc::manta
