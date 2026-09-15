#include "vibeqc/periodic_gaussian_pair_space.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
namespace local=periodic_correlation_local_detail;
using I=std::uint64_t;
using real::add;
using real::mul;
using real::Digest;
using PNO=PeriodicGaussianPairPNOResult;
using EmbeddedPNO=PeriodicGaussianEmbeddedPairPNOResult;
using GramPNO=PeriodicGaussianGramPairPNOResult;
using Frame=PeriodicGaussianPairPNOFrame;
using FrameView=PeriodicGaussianPairPNOFrameView;
using Space=PeriodicGaussianPairSpace;
using Storage=PeriodicGaussianPairSpaceStorage;
using Plan=PeriodicGaussianPairSpacePlan;
using Options=PeriodicGaussianPairSpaceOptions;
using Live=PeriodicGaussianPairSpaceLiveInventory;
using Caps=PeriodicGaussianPairSpaceCaps;
using Overlap=PeriodicGaussianPairOverlap;
using OverlapPlan=PeriodicGaussianPairOverlapPlan;
using OverlapCaps=PeriodicGaussianPairOverlapCaps;
constexpr char policy[]="Nejad2025-Eqs37-44;common-PAO;Gprime=C^TGC;row-streamed;PNO-owner-transfer;explicit-diagonal-roundoff-projection;no-energy-multiplicity";
constexpr char overlap_policy[]="Nejad2025-Eqs42-44;O=Ctarget^TCsource;same-certified-common-basis;no-pair-translation-or-multiplicity";
constexpr char direct_policy[]="Nejad2025-Eqs37-44;direct-PAO-Gram;retain-authentic-D-and-C=XD;borrow-native-GPNO;no-common-integral-replay;no-second-projection;final-owner-transfer";
constexpr char direct_overlap_policy[]="Nejad2025-Eq42;O=Ctarget^TCsource;same-actual-common-basis-HF-context;independent-selected-Gram-receipts;no-mixed-legacy-source";
static_assert(sizeof(double)==8 && FLT_EVAL_METHOD==0,"Gaussian pair spaces require binary64 evaluation");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian pair spaces do not support fast-math"
#endif
void limit(I x,I cap,const char* message) { if(!cap || x>cap) throw std::length_error(message); }
void live_valid(const Live& live) {
    if(!live.fixed_backend_margin_bytes_per_worker)
        throw std::invalid_argument("Gaussian pair space backend margin must be explicit and positive");
}
void options_valid(const Options& options) {
    if(!std::isfinite(options.maximum_diagonal_symmetry_projection_error)
        || options.maximum_diagonal_symmetry_projection_error<0)
        throw std::invalid_argument("Gaussian pair space diagonal projection budget must be explicit and finite");
}
void accumulate(double term,double& sum,double& correction) {
    real::finite(term); const auto next=real::finite(sum+term);
    const auto tail=std::abs(sum)>=std::abs(term)
        ? real::finite(real::finite(sum-next)+term) : real::finite(real::finite(term-next)+sum);
    correction=real::finite(correction+tail); sum=next;
}
double product(double a,double b,I& underflows) {
    const auto x=real::finite(a*b);
    if(x==0 && a!=0 && b!=0) ++underflows;
    return x;
}
double average(double a,double b) {
    if(a==b) return a; // Preserve equal subnormals exactly.
    const auto half=std::numeric_limits<double>::max()*0.5;
    if(std::signbit(a)!=std::signbit(b) || (std::abs(a)<=half && std::abs(b)<=half))
        return real::finite((a+b)*0.5);
    return real::finite(a*0.5+b*0.5);
}
bool same_label(PeriodicCorrelationPlacedOccupied a,PeriodicCorrelationPlacedOccupied b) {
    return a.occupied_index==b.occupied_index && a.cell==b.cell;
}
void objects(const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicGaussianRealLocalProvider& wrapper,FrameView pno) {
    if(pno.is_direct_gram())
        throw std::invalid_argument("direct PAO-Gram pair space does not accept a legacy provider");
    local::validate_reference(ref);
    const auto& provider=wrapper.provider(); const auto& context=wrapper.context_handle();
    (void)pno.coefficients(); // Constant-size moved-owner and shape guard, not a payload scan.
    if(!context || pno.state_handle().get()!=ref.state_handle().get()
        || pno.context_handle().get()!=context.get() || basis.state_handle().get()!=ref.state_handle().get()
        || provider.state_handle().get()!=ref.state_handle().get()
        || basis.allocation_identity()!=ref.dimensions().allocation_identity
        || !provider.matched_finite_gaussian_hf_recipe() || !wrapper.matched_finite_gaussian_hf_recipe()
        || provider.identity_sha256()!=wrapper.identity_sha256()
        || provider.basis_certificate_identity_sha256()!=basis.identity_sha256()
        || provider.local_basis_identity_sha256()!=basis.local_basis_identity_sha256()
        || pno.basis_identity_sha256()!=basis.identity_sha256()
        || pno.provider_identity_sha256()!=wrapper.identity_sha256()
        || pno.hf_reference_source_identity_sha256()!=wrapper.hf_reference_source_identity_sha256()
        || provider.hf_reference_source_identity_sha256()!=wrapper.hf_reference_source_identity_sha256()
        || provider.source_context_identity_sha256()!=context->source_context_identity_sha256()
        || wrapper.source_context_identity_sha256()!=context->source_context_identity_sha256())
        throw std::invalid_argument("Gaussian pair space requires identical actual source/basis/state/PNO owners and receipts");
    const auto& b=basis.memory(); const auto& p=provider.memory();
    const auto callback=provider.integral_provider(basis);
    if(!b.occupied_count || !b.virtual_count || b.orbital_count!=add(b.occupied_count,b.virtual_count)
        || b.occupied_count!=p.occupied_count || b.virtual_count!=p.virtual_count || b.orbital_count!=p.orbital_count
        || pno.occupied_count()!=b.occupied_count || pno.common_virtual_dimension()!=b.virtual_count
        || pno.occupied_slot_i()>=b.occupied_count || pno.occupied_slot_j()>=b.occupied_count
        || p.n_cells!=ref.state().n_kpoints() || p.n_auxiliary!=ref.dimensions().n_auxiliary
        || context->mesh().mesh()!=ref.state().mesh() || context->mesh().is_shift()!=ref.state().is_shift()
        || context->inventory().ao.function_count!=ref.state().n_basis()
        || context->inventory().auxiliary.function_count!=p.n_auxiliary
        || wrapper.memory().retained_row_bytes!=p.retained_row_bytes
        || wrapper.memory().scalar_work_units!=p.scalar_work_units
        || callback.retained_numerical_bytes!=add(p.retained_row_bytes,b.retained_index_bytes)
        || callback.maximum_transient_numerical_bytes)
        throw std::invalid_argument("Gaussian pair space source/PNO shape or provider inventory mismatch");
    if(!same_label(pno.occupied_i(),basis.occupied(pno.occupied_slot_i()))
        || !same_label(pno.occupied_j(),basis.occupied(pno.occupied_slot_j())))
        throw std::invalid_argument("Gaussian pair space physical occupied labels differ from its common basis");
}
std::string integrals_hash(const std::vector<double>& values,I n,I r) {
    Digest h("vibeqc.periodic.gaussian-pair-space.integrals"); h.u64(n); h.u64(r); h.u64(values.size());
    for(double value:values) h.real(value);
    return h.finish();
}
std::string payload_hash(const std::string& pno,const std::string& integrals) {
    Digest h("vibeqc.periodic.gaussian-pair-space.payload"); h.string(pno); h.string(integrals); return h.finish();
}
void storage_wire(Digest& h,const Storage& p) {
    for(auto x:{p.virtual_count,p.retained_dimension,p.integral_calls,p.construction_owned_bytes,p.borrowed_pno_bytes,
        p.retained_output_bytes,p.peak_owned_numerical_bytes,p.construction_live_numerical_bytes,
        p.numerical_work_units,p.fixed_control_storage_bytes}) h.u64(x);
}
void plan_wire(Digest& h,const Plan& p) {
    storage_wire(h,p);
    for(auto x:{p.occupied_count,p.occupied_slot_i,p.occupied_slot_j,p.provider_work_units,p.work_units,
        p.borrowed_basis_bytes,p.borrowed_provider_row_bytes,p.state_resident_bytes,p.replicas_per_node,
        p.reference_base_node_bytes,p.per_worker_inventoried_bytes,p.required_node_memory_bytes}) h.u64(x);
    h.real(p.options.maximum_diagonal_symmetry_projection_error);
    h.u64(p.live.other_live_bytes_per_worker); h.u64(p.live.fixed_backend_margin_bytes_per_worker);
    for(auto x:{p.caps.maximum_owned_numerical_bytes,p.caps.maximum_per_worker_inventoried_bytes,
        p.caps.maximum_node_inventoried_bytes,p.caps.maximum_integral_calls,p.caps.maximum_work_units}) h.u64(x);
}
template<class P> void reference_memory(const PeriodicCorrelationAdmittedReference& ref,P& p) {
    const auto& d=ref.dimensions(); const auto& b=ref.budget();
    p.state_resident_bytes=ref.state_resident_bytes();
    if(p.state_resident_bytes!=ref.state().resident_bytes() || d.external_bytes<p.state_resident_bytes)
        throw std::logic_error("Gaussian pair space reference state inventory changed");
    p.replicas_per_node=mul(b.mpi_ranks,b.workers_per_rank);
    p.reference_base_node_bytes=add(add(d.external_bytes,d.shared_bytes),
        mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
}
template<class P> void memory_caps(const PeriodicCorrelationAdmittedReference& ref,P& p,bool allow_empty_owned=false) {
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    if(!allow_empty_owned || p.peak_owned_numerical_bytes || p.caps.maximum_owned_numerical_bytes)
        limit(p.peak_owned_numerical_bytes,p.caps.maximum_owned_numerical_bytes,"Gaussian pair space owned memory cap exceeded");
    limit(p.per_worker_inventoried_bytes,p.caps.maximum_per_worker_inventoried_bytes,"Gaussian pair space worker memory cap exceeded");
    limit(p.required_node_memory_bytes,p.caps.maximum_node_inventoried_bytes,"Gaussian pair space node memory cap exceeded");
    limit(p.required_node_memory_bytes,ref.budget().memory_limit_bytes,"Gaussian pair space admitted-reference memory cap exceeded");
    limit(p.work_units,p.caps.maximum_work_units,"Gaussian pair space work cap exceeded");
}
void overlap_objects(const PeriodicCorrelationAdmittedReference& ref,const Space& target,const Space& source) {
    local::validate_reference(ref); const auto a=target.frame(),b=source.frame();
    (void)target.exchange_integrals(); (void)source.exchange_integrals();
    const bool direct=a.is_direct_gram();
    if(direct!=b.is_direct_gram())
        throw std::invalid_argument("Gaussian pair overlap does not qualify mixed direct PAO-Gram and legacy sources");
    // Individual direct pair Gram selections need not agree: the common
    // C basis, physical HF source and exact state/context owners must agree.
    const bool source_matches=direct || (target.consumed_sources_identity_sha256()==source.consumed_sources_identity_sha256()
        && a.provider_identity_sha256()==b.provider_identity_sha256());
    const auto target_owned=direct ? a.retained_numerical_bytes()
        : add(a.retained_numerical_bytes(),mul(8,target.exchange_integrals().size()));
    const auto source_owned=direct ? b.retained_numerical_bytes()
        : add(b.retained_numerical_bytes(),mul(8,source.exchange_integrals().size()));
    if(target.state_handle().get()!=ref.state_handle().get() || source.state_handle().get()!=ref.state_handle().get()
        || target.context_handle().get()!=source.context_handle().get()
        || target.allocation_identity()!=ref.dimensions().allocation_identity
        || source.allocation_identity()!=target.allocation_identity()
        || target.local_basis_identity_sha256()!=source.local_basis_identity_sha256()
        || !source_matches
        || a.basis_identity_sha256()!=b.basis_identity_sha256()
        || a.hf_reference_source_identity_sha256()!=b.hf_reference_source_identity_sha256()
        || target.memory().virtual_count!=source.memory().virtual_count
        || target.memory().occupied_count!=source.memory().occupied_count
        || target.memory().generation_dimension!=a.generation_dimension()
        || source.memory().generation_dimension!=b.generation_dimension()
        || target.memory().retained_output_bytes!=target_owned
        || source.memory().retained_output_bytes!=source_owned)
        throw std::invalid_argument("Gaussian pair overlap requires the same certified common basis/provider/context/state origin");
}
void space_payload(const Space& space) {
    space.frame().verify_payload();
    const auto g=integrals_hash(space.exchange_integrals(),space.memory().virtual_count,space.memory().retained_dimension);
    if(g!=space.exchange_integral_identity_sha256()
        || payload_hash(space.frame().payload_sha256(),g)!=space.payload_sha256())
        throw std::invalid_argument("Gaussian pair overlap input payload digest mismatch");
}
} // namespace

namespace {
Storage frame_storage(I n,I m,I r,bool embedded) {
    if(!n || !m || m>n || r>m)
        throw std::invalid_argument("Gaussian pair space dimensions require 0<=r<=m<=n with n and m positive");
    Storage p; p.virtual_count=n; p.generation_dimension=m; p.retained_dimension=r;
    const auto nn=mul(n,n),rr=mul(r,r),nr=mul(n,r);
    p.integral_calls=r ? nn : 0;
    p.construction_owned_bytes=add(mul(16,rr),mul(16,r));
    p.borrowed_pno_bytes=mul(8,add(add(nr,r),m));
    if(embedded) p.borrowed_pno_bytes=add(p.borrowed_pno_bytes,mul(8,mul(m,r)));
    p.retained_output_bytes=add(p.borrowed_pno_bytes,mul(8,rr));
    p.peak_owned_numerical_bytes=std::max(p.construction_owned_bytes,p.retained_output_bytes);
    p.construction_live_numerical_bytes=add(p.borrowed_pno_bytes,p.construction_owned_bytes);
    // Covers all scalar contractions, finalization, raw/projection/final
    // scans and SHA lanes, with fixed owner/control allowance. Not timing.
    p.numerical_work_units=add(65536,mul(1024,add(add(mul(p.integral_calls,add(r,1)),mul(n,rr)),
        add(add(add(nr,m),r),add(rr,1)))));
    // Shared tagged payload verifier, precharged before its first scan.
    p.numerical_work_units=add(p.numerical_work_units,mul(512,add(p.borrowed_pno_bytes/8,1024)));
    p.fixed_control_storage_bytes=65536+sizeof(Plan)+sizeof(Space)+sizeof(Options)+sizeof(Live)+sizeof(Caps)
        +sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationRealLocalBasis)
        +sizeof(PeriodicGaussianRealLocalProvider)+sizeof(PeriodicGaussianSourceContext)+sizeof(Frame)+sizeof(FrameView)
        +14*65; // Maximum source-tag receipt payload; no occupation padding.
    for(auto bytes:{p.construction_owned_bytes,p.borrowed_pno_bytes,p.retained_output_bytes,
        p.construction_live_numerical_bytes,mul(8,p.integral_calls)}) real::extent(bytes);
    if(rr>std::vector<double>().max_size() || r>std::vector<double>().max_size())
        throw std::length_error("Gaussian pair space vector extent exceeds cap");
    return p;
}
} // namespace
Storage periodic_gaussian_pair_space_storage(I n,I r) { return frame_storage(n,n,r,false); }
Storage periodic_gaussian_pair_space_storage(I n,I m,I r) { return frame_storage(n,m,r,true); }
Storage periodic_gaussian_direct_gram_pair_space_storage(I n,I m,I r) {
    if(!n || !m || m>n || r>m)
        throw std::invalid_argument("direct PAO-Gram pair space requires 0<=r<=m<=n with positive n and m");
    Storage p; p.virtual_count=n; p.generation_dimension=m; p.retained_dimension=r;
    const auto rr=mul(r,r);
    p.borrowed_pno_bytes=mul(8,add(add(add(mul(n,r),mul(m,r)),add(r,m)),rr));
    p.retained_output_bytes=p.borrowed_pno_bytes;
    p.peak_owned_numerical_bytes=p.borrowed_pno_bytes; // Final transfer, not a new allocation.
    p.construction_live_numerical_bytes=p.borrowed_pno_bytes;
    // One full source-specific payload validation plus a separate retained
    // G digest/exact diagonal-symmetry scan and fixed receipt verification.
    // No Fock/PNO reconstruction, no integral query, no numerical workspace.
    p.numerical_work_units=add(131072,add(mul(512,add(p.borrowed_pno_bytes/8,1024)),mul(1024,add(rr,1))));
    p.fixed_control_storage_bytes=65536+sizeof(Plan)+sizeof(Space)+sizeof(Options)+sizeof(Live)+sizeof(Caps)
        +sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationRealLocalBasis)
        +sizeof(PeriodicGaussianSourceContext)+sizeof(GramPNO)+sizeof(FrameView)+19*65;
    real::extent(p.borrowed_pno_bytes);
    return p;
}

namespace {
void sha_format(const std::string& text) {
    if(text.size()!=64 || !std::all_of(text.begin(),text.end(),[](char c) {
        return (c>='0' && c<='9') || (c>='a' && c<='f');
    })) throw std::invalid_argument("direct PAO-Gram pair space source receipt is malformed");
}
void direct_objects(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const GramPNO& pno) {
    local::validate_reference(ref);
    const FrameView view(pno); const auto& context=view.context_handle();
    const auto& b=basis.memory(); const auto& p=pno.memory();
    if(basis.contract_version()!=kPeriodicCorrelationRealLocalBasisContractVersion
        || basis.state_handle().get()!=ref.state_handle().get()
        || view.state_handle().get()!=ref.state_handle().get()
        || basis.allocation_identity()!=ref.dimensions().allocation_identity
        || !pno.matched_finite_gaussian_hf_recipe()
        || pno.basis_identity_sha256()!=basis.identity_sha256()
        || pno.source_context_identity_sha256()!=context->source_context_identity_sha256()
        || context->mesh().mesh()!=ref.state().mesh() || context->mesh().is_shift()!=ref.state().is_shift()
        || context->inventory().ao.function_count!=ref.state().n_basis()
        || context->inventory().auxiliary.function_count!=ref.dimensions().n_auxiliary
        || p.n_cells!=ref.state().n_kpoints() || p.n_basis!=ref.state().n_basis()
        || !b.occupied_count || !b.virtual_count || b.orbital_count!=add(b.occupied_count,b.virtual_count)
        || view.occupied_count()!=b.occupied_count || view.common_virtual_dimension()!=b.virtual_count
        || pno.diagonal_pair()!=view.diagonal_pair()
        || !same_label(view.occupied_i(),basis.occupied(view.occupied_slot_i()))
        || !same_label(view.occupied_j(),basis.occupied(view.occupied_slot_j()))
        || pno.retained_numerical_bytes()!=view.retained_numerical_bytes()
        || pno.diagnostics().retained_output_bytes!=view.retained_numerical_bytes()
        || pno.diagnostics().retained_frame_bytes!=view.retained_numerical_bytes()-mul(8,mul(view.retained_dimension(),view.retained_dimension())))
        throw std::invalid_argument("direct PAO-Gram pair space requires the identical actual common basis/state/context lineage");
    const auto& d=pno.diagnostics();
    if(!std::isfinite(d.maximum_raw_retained_transpose_defect) || d.maximum_raw_retained_transpose_defect<0
        || !std::isfinite(d.retained_diagonal_projection_frobenius_upper_bound)
        || d.retained_diagonal_projection_frobenius_upper_bound<0
        || (!view.diagonal_pair() && d.retained_diagonal_projection_frobenius_upper_bound!=0))
        throw std::invalid_argument("direct PAO-Gram pair space retained projection diagnostics are malformed");
}
void direct_receipts(const GramPNO& pno) {
    for(const auto* text:{&pno.identity_sha256(),&pno.payload_sha256(),&pno.basis_identity_sha256(),
        &pno.local_basis_identity_sha256(),&pno.hf_reference_source_identity_sha256(),
        &pno.source_context_identity_sha256(),&pno.geometry_identity_sha256(),&pno.embedding_identity_sha256(),
        &pno.gram_identity_sha256(),&pno.gram_payload_sha256(),&pno.gram_consumed_sources_identity_sha256(),
        &pno.raw_exchange_identity_sha256(),&pno.projected_exchange_identity_sha256(),&pno.local_fock_identity_sha256(),
        &pno.raw_retained_exchange_identity_sha256(),&pno.retained_exchange_identity_sha256(),
        &pno.initial_amplitude_identity_sha256(),&pno.density_identity_sha256(),&pno.source_payload_receipt_sha256()})
        sha_format(*text);
}
Plan plan_frame(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    FrameView pno,const Options& options,const Live& live,const Caps& caps) {
    options_valid(options); live_valid(live); real::float_environment(); objects(ref,basis,provider,pno);
    Plan p; static_cast<Storage&>(p)=frame_storage(
        pno.common_virtual_dimension(),pno.generation_dimension(),pno.retained_dimension(),pno.is_embedded());
    p.options=options; p.live=live; p.caps=caps; p.occupied_count=pno.occupied_count();
    p.occupied_slot_i=pno.occupied_slot_i(); p.occupied_slot_j=pno.occupied_slot_j();
    p.provider_work_units=mul(p.integral_calls,provider.memory().scalar_work_units);
    p.work_units=add(p.provider_work_units,p.numerical_work_units);
    p.borrowed_basis_bytes=basis.memory().retained_output_bytes;
    p.borrowed_provider_row_bytes=provider.memory().retained_row_bytes;
    if(p.borrowed_pno_bytes!=pno.retained_numerical_bytes())
        throw std::logic_error("Gaussian pair space PNO live inventory differs from payload");
    const auto validation=plan_periodic_gaussian_pair_pno_frame_validation(pno);
    if(validation.numerical_lanes!=p.borrowed_pno_bytes/8
        || validation.work_units!=mul(512,add(validation.numerical_lanes,1024))
        || validation.control_storage_bytes>65536)
        throw std::logic_error("Gaussian pair space tagged validation exceeds its admitted work/control allowance");
    reference_memory(ref,p);
    p.per_worker_inventoried_bytes=add(p.construction_live_numerical_bytes,add(p.borrowed_basis_bytes,
        add(p.borrowed_provider_row_bytes,add(p.fixed_control_storage_bytes,
        add(live.other_live_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker)))));
    memory_caps(ref,p);
    limit(p.integral_calls,caps.maximum_integral_calls,"Gaussian pair space integral call cap exceeded");
    Digest h("vibeqc.periodic.gaussian-pair-space.plan");
    for(const auto& s:{ref.state().state_identity_sha256(),ref.dimensions().allocation_identity,
        basis.identity_sha256(),provider.identity_sha256(),provider.hf_reference_source_identity_sha256(),pno.identity_sha256()}) h.string(s);
    plan_wire(h,p); h.string(policy);
    // Preserve the legacy wire. Embedded receipts add their distinct
    // generation dimension even when m=n, without relabelling the source.
    if(pno.is_embedded()) { h.string("embedded-pair-generation"); h.u64(p.generation_dimension); }
    const auto sha=h.finish();
    std::copy(sha.begin(),sha.end(),p.identity_ascii.begin()); return p;
}
} // namespace
Plan plan_periodic_gaussian_pair_space(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PNO& pno,const Options& options,const Live& live,const Caps& caps) {
    return plan_frame(ref,basis,provider,FrameView(pno),options,live,caps);
}
Plan plan_periodic_gaussian_pair_space(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const GramPNO& pno,
    const Options& options,const Live& live,const Caps& caps) {
    options_valid(options); live_valid(live); real::float_environment();
    if(!ref.state_handle()) throw std::invalid_argument("direct PAO-Gram pair space reference is consumed");
    const FrameView view(pno);
    Plan p; static_cast<Storage&>(p)=periodic_gaussian_direct_gram_pair_space_storage(
        view.common_virtual_dimension(),view.generation_dimension(),view.retained_dimension());
    p.options=options; p.live=live; p.caps=caps;
    p.occupied_count=view.occupied_count(); p.occupied_slot_i=view.occupied_slot_i(); p.occupied_slot_j=view.occupied_slot_j();
    p.borrowed_basis_bytes=basis.memory().retained_output_bytes;
    p.work_units=p.numerical_work_units;
    reference_memory(ref,p);
    p.per_worker_inventoried_bytes=add(p.construction_live_numerical_bytes,add(p.borrowed_basis_bytes,
        add(p.fixed_control_storage_bytes,add(live.other_live_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    memory_caps(ref,p); // Counts before all source receipt/payload traversals.
    direct_objects(ref,basis,pno); direct_receipts(pno);
    const auto validation=plan_periodic_gaussian_pair_pno_frame_validation(view);
    if(validation.numerical_lanes!=p.borrowed_pno_bytes/8
        || validation.work_units!=mul(512,add(validation.numerical_lanes,1024))
        || validation.control_storage_bytes>65536)
        throw std::logic_error("direct PAO-Gram frame validation exceeds admitted memory/work");
    if(pno.diagnostics().retained_diagonal_projection_frobenius_upper_bound>options.maximum_diagonal_symmetry_projection_error)
        throw std::invalid_argument("direct PAO-Gram retained diagonal projection exceeds pair-space budget");
    Digest h("vibeqc.periodic.gaussian-direct-gram-pair-space.plan");
    for(const auto* text:{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &basis.identity_sha256(),&basis.local_basis_identity_sha256(),&pno.identity_sha256(),
        &pno.hf_reference_source_identity_sha256(),&pno.source_context_identity_sha256(),
        &pno.gram_identity_sha256(),&pno.gram_payload_sha256(),&pno.gram_consumed_sources_identity_sha256(),
        &pno.retained_exchange_identity_sha256()}) h.string(*text);
    plan_wire(h,p); h.u64(p.generation_dimension); h.string(direct_policy); h.string(real::kFloatPolicy);
    const auto sha=h.finish(); std::copy(sha.begin(),sha.end(),p.identity_ascii.begin()); return p;
}
Plan plan_periodic_gaussian_pair_space(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const EmbeddedPNO& pno,const Options& options,const Live& live,const Caps& caps) {
    return plan_frame(ref,basis,provider,FrameView(pno),options,live,caps);
}

FrameView Space::frame() const {
    if(!pnos_) throw std::logic_error("Gaussian pair space is moved or incomplete");
    const auto view=pnos_->view(); (void)view.coefficients(); return view;
}
const PNO& Space::pnos() const {
    return frame().legacy();
}
const EmbeddedPNO& Space::embedded_pnos() const {
    return frame().embedded();
}
const GramPNO& Space::direct_gram_pnos() const {
    return frame().direct_gram();
}
const std::vector<double>& Space::exchange_integrals() const {
    const auto view=frame();
    if(view.is_direct_gram()) {
        if(!exchange_.empty() || view.direct_gram().exchange_integrals().size()!=mul(memory_.retained_dimension,memory_.retained_dimension)
            || memory_.retained_output_bytes!=view.retained_numerical_bytes())
            throw std::logic_error("direct PAO-Gram pair space integral owner is moved or malformed");
        return view.direct_gram().exchange_integrals();
    }
    if(exchange_.size()!=mul(memory_.retained_dimension,memory_.retained_dimension))
        throw std::logic_error("Gaussian pair space integral payload is moved or malformed");
    return exchange_;
}

Space Space::build(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& wrapper,
    FrameView pno,const Options& options,const Live& live,const Caps& caps) {
    const auto p=plan_frame(ref,basis,wrapper,pno,options,live,caps);
    pno.verify_payload(); // All caps before the first full input scan/allocation.
    const auto n=p.virtual_count,r=p.retained_dimension,rr=mul(r,r),o=p.occupied_count;
    const auto i=p.occupied_slot_i,j=p.occupied_slot_j;
    const auto& coefficients=pno.coefficients(); const auto& provider=wrapper.provider();
    Space result; result.memory_=p; result.allocation_=ref.dimensions().allocation_identity;
    result.local_basis_=basis.local_basis_identity_sha256(); result.sources_=wrapper.consumed_sources_identity_sha256();
    auto& d=result.diagnostics_;
    d.common_basis_integral_projection_error_bound=provider.diagnostics().maximum_eri_projection_error_bound;
    result.exchange_.resize(static_cast<std::size_t>(rr));
    {
        std::vector<double> correction(static_cast<std::size_t>(rr));
        std::vector<double> row(static_cast<std::size_t>(r)),row_correction(static_cast<std::size_t>(r));
        Digest source("vibeqc.periodic.gaussian-pair-pnos.integrals"); source.u64(n); source.u64(mul(n,n));
        if(r) for(I a=0;a<n;++a) {
            std::fill(row.begin(),row.end(),0.0); std::fill(row_correction.begin(),row_correction.end(),0.0);
            for(I b=0;b<n;++b) {
                const auto integral=provider.integral(i,o+a,j,o+b);
                const auto value=real::finite(integral.value); source.real(value);
                d.maximum_scalar_integral_roundoff_error=std::max(d.maximum_scalar_integral_roundoff_error,
                    real::finite(integral.roundoff_error_bound));
                ++d.completed_integral_calls;
                for(I beta=0;beta<r;++beta)
                    accumulate(product(value,coefficients[b*r+beta],d.product_underflow_count),row[beta],row_correction[beta]);
            }
            for(I beta=0;beta<r;++beta) row[beta]=real::finite(row[beta]+row_correction[beta]);
            for(I alpha=0;alpha<r;++alpha) for(I beta=0;beta<r;++beta)
                accumulate(product(coefficients[a*r+alpha],row[beta],d.product_underflow_count),
                    result.exchange_[alpha*r+beta],correction[alpha*r+beta]);
        }
        if(r) {
            if(source.finish()!=pno.common_exchange_integral_identity_sha256())
                throw std::invalid_argument("Gaussian pair space source-integral replay differs from generating PNOs");
            d.source_integrals_replayed=true;
        }
        for(I at=0;at<rr;++at) result.exchange_[at]=real::finite(result.exchange_[at]+correction[at]);
    } // All numerical scratch is gone before ownership transfer.
    if(d.completed_integral_calls!=p.integral_calls)
        throw std::logic_error("Gaussian pair space completed calls differ from admitted count");
    result.raw_integrals_=integrals_hash(result.exchange_,n,r);
    double projection_squared=0;
    if(pno.diagonal_pair()) {
        for(I a=0;a<r;++a) for(I b=a+1;b<r;++b) {
            const auto x=result.exchange_[a*r+b],y=result.exchange_[b*r+a],value=average(x,y);
            d.maximum_raw_diagonal_asymmetry=std::max(d.maximum_raw_diagonal_asymmetry,real::difference_up(x,y));
            real::norm_lane(projection_squared,real::difference_up(x,value));
            real::norm_lane(projection_squared,real::difference_up(y,value));
            result.exchange_[a*r+b]=result.exchange_[b*r+a]=value;
        }
        d.diagonal_symmetry_projection_frobenius_bound=real::sqrt_up(projection_squared);
        if(d.diagonal_symmetry_projection_frobenius_bound>options.maximum_diagonal_symmetry_projection_error)
            throw std::invalid_argument("Gaussian pair space diagonal roundoff projection exceeds explicit budget");
        d.diagonal_symmetry_projection_applied=d.diagonal_symmetry_projection_frobenius_bound!=0;
    }
    result.integrals_=integrals_hash(result.exchange_,n,r);
    result.payload_=payload_hash(pno.payload_sha256(),result.integrals_);
    Digest h("vibeqc.periodic.gaussian-pair-space.identity");
    for(const auto& s:{ref.state().state_identity_sha256(),result.allocation_,
        pno.context_handle()->source_context_identity_sha256(),pno.hf_reference_source_identity_sha256(),
        basis.identity_sha256(),result.local_basis_,pno.provider_identity_sha256(),result.sources_,
        pno.identity_sha256(),result.raw_integrals_,result.integrals_,result.payload_}) h.string(s);
    for(auto x:{i,j,pno.occupied_i().occupied_index,pno.occupied_i().cell,
        pno.occupied_j().occupied_index,pno.occupied_j().cell,n,r,d.completed_integral_calls,d.product_underflow_count}) h.u64(x);
    h.u32(d.source_integrals_replayed); h.u32(d.diagonal_symmetry_projection_applied);
    for(double x:{options.maximum_diagonal_symmetry_projection_error,d.maximum_scalar_integral_roundoff_error,
        d.common_basis_integral_projection_error_bound,d.maximum_raw_diagonal_asymmetry,
        d.diagonal_symmetry_projection_frobenius_bound}) h.real(x);
    h.string(policy); h.string(real::kFloatPolicy); result.identity_=h.finish();
    return result;
}
Space make_periodic_gaussian_pair_space(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& wrapper,
    PNO&& pno,const Options& options,const Live& live,const Caps& caps) {
    auto result=Space::build(ref,basis,wrapper,FrameView(pno),options,live,caps);
    // Nothing that can throw remains after the noexcept move of the sole
    // coefficient/energy owner. No PNO copy and no dangling borrowed view.
    result.pnos_.emplace(std::move(pno)); return result;
}
Space make_periodic_gaussian_pair_space(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& wrapper,
    EmbeddedPNO&& pno,const Options& options,const Live& live,const Caps& caps) {
    auto result=Space::build(ref,basis,wrapper,FrameView(pno),options,live,caps);
    result.pnos_.emplace(std::move(pno)); return result;
}

Space make_periodic_gaussian_pair_space(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,GramPNO&& pno,
    const Options& options,const Live& live,const Caps& caps) {
    const auto p=plan_periodic_gaussian_pair_space(ref,basis,pno,options,live,caps);
    const FrameView view(pno);
    view.verify_payload(); // Admitted full source scan includes the owned G_PNO.
    const auto n=p.virtual_count,r=p.retained_dimension;
    const auto& integrals=pno.exchange_integrals();
    // This is a consuming adapter, not a second numerical projection. The
    // source explicitly gated/symmetrized physical diagonal G_PNO already.
    if(view.diagonal_pair()) for(I a=0;a<r;++a) for(I b=a+1;b<r;++b)
        if(integrals[a*r+b]!=integrals[b*r+a])
            throw std::invalid_argument("direct PAO-Gram diagonal retained integrals are not exactly symmetric");
    Space result; result.memory_=p;
    result.allocation_=ref.dimensions().allocation_identity;
    result.local_basis_=basis.local_basis_identity_sha256();
    result.sources_=pno.gram_consumed_sources_identity_sha256();
    auto& d=result.diagnostics_;
    d.maximum_raw_diagonal_asymmetry=pno.diagonal_pair() ? pno.diagnostics().maximum_raw_retained_transpose_defect : 0.0;
    d.diagonal_symmetry_projection_frobenius_bound=pno.diagnostics().retained_diagonal_projection_frobenius_upper_bound;
    d.diagonal_symmetry_projection_applied=d.diagonal_symmetry_projection_frobenius_bound!=0;
    // COMMON-provider diagnostics remain zero: there was no provider replay.
    // Actual selected-Gram error diagnostics stay on the typed source owner.
    result.raw_integrals_=pno.raw_retained_exchange_identity_sha256();
    result.integrals_=integrals_hash(integrals,n,r);
    result.payload_=payload_hash(pno.payload_sha256(),result.integrals_);
    Digest h("vibeqc.periodic.gaussian-direct-gram-pair-space.identity");
    const std::array<const std::string*,15> receipts{{&ref.state().state_identity_sha256(),&result.allocation_,
        &basis.identity_sha256(),&result.local_basis_,&pno.source_context_identity_sha256(),
        &pno.hf_reference_source_identity_sha256(),&pno.identity_sha256(),&pno.payload_sha256(),
        &pno.gram_identity_sha256(),&pno.gram_payload_sha256(),&result.sources_,
        &pno.retained_exchange_identity_sha256(),&result.raw_integrals_,&result.integrals_,&result.payload_}};
    for(const auto* text:receipts) h.string(*text);
    for(auto x:{p.occupied_slot_i,p.occupied_slot_j,view.occupied_i().occupied_index,view.occupied_i().cell,
        view.occupied_j().occupied_index,view.occupied_j().cell,n,p.generation_dimension,r}) h.u64(x);
    h.real(options.maximum_diagonal_symmetry_projection_error);
    h.real(d.maximum_raw_diagonal_asymmetry); h.real(d.diagonal_symmetry_projection_frobenius_bound);
    h.u32(d.diagonal_symmetry_projection_applied); h.string(direct_policy); h.string(real::kFloatPolicy);
    result.identity_=h.finish();
    // All scans, allocations and possible exceptions precede this transfer.
    // The empty exchange_ vector does not duplicate the source integral owner.
    result.pnos_.emplace(std::move(pno)); return result;
}

OverlapPlan plan_periodic_gaussian_pair_overlap(const PeriodicCorrelationAdmittedReference& ref,
    const Space& target,const Space& source,const Live& live,const OverlapCaps& caps) {
    live_valid(live); real::float_environment(); overlap_objects(ref,target,source);
    OverlapPlan p; p.live=live; p.caps=caps;
    p.virtual_count=target.memory().virtual_count; p.target_dimension=target.memory().retained_dimension;
    p.source_dimension=source.memory().retained_dimension; p.same_object_borrower=&target==&source;
    const auto count=mul(p.target_dimension,p.source_dimension);
    p.peak_owned_numerical_bytes=mul(8,count);
    p.borrowed_pair_space_bytes=target.memory().retained_output_bytes;
    if(!p.same_object_borrower) p.borrowed_pair_space_bytes=add(p.borrowed_pair_space_bytes,source.memory().retained_output_bytes);
    p.work_units=add(65536,mul(1024,add(add(mul(p.virtual_count,count),count),add(p.borrowed_pair_space_bytes/8,1))));
    const auto target_validation=plan_periodic_gaussian_pair_pno_frame_validation(target.frame());
    const auto source_validation=plan_periodic_gaussian_pair_pno_frame_validation(source.frame());
    if(target_validation.control_storage_bytes>65536 || source_validation.control_storage_bytes>65536)
        throw std::logic_error("Gaussian pair overlap tagged validation exceeds fixed control allowance");
    p.work_units=add(p.work_units,target_validation.work_units);
    if(!p.same_object_borrower) p.work_units=add(p.work_units,source_validation.work_units);
    p.fixed_control_storage_bytes=65536+sizeof(OverlapPlan)+sizeof(Overlap)+2*sizeof(Space)+2*sizeof(FrameView)
        +sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicGaussianSourceContext)
        +target.frame().retained_receipt_payload_bytes();
    if(!p.same_object_borrower)
        p.fixed_control_storage_bytes=add(p.fixed_control_storage_bytes,source.frame().retained_receipt_payload_bytes());
    reference_memory(ref,p);
    p.per_worker_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_pair_space_bytes,
        add(p.fixed_control_storage_bytes,add(live.other_live_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    real::extent(p.peak_owned_numerical_bytes); real::extent(p.borrowed_pair_space_bytes);
    if(count>std::vector<double>().max_size()) throw std::length_error("Gaussian pair overlap vector extent exceeds cap");
    memory_caps(ref,p,target.frame().is_direct_gram());
    Digest h("vibeqc.periodic.gaussian-pair-overlap.plan");
    for(const auto& s:{ref.state().state_identity_sha256(),ref.dimensions().allocation_identity,
        target.identity_sha256(),source.identity_sha256()}) h.string(s);
    for(auto x:{p.virtual_count,p.target_dimension,p.source_dimension,p.peak_owned_numerical_bytes,
        p.borrowed_pair_space_bytes,p.work_units,p.fixed_control_storage_bytes,p.state_resident_bytes,
        p.replicas_per_node,p.reference_base_node_bytes,p.per_worker_inventoried_bytes,p.required_node_memory_bytes}) h.u64(x);
    h.u32(p.same_object_borrower); h.u64(live.other_live_bytes_per_worker); h.u64(live.fixed_backend_margin_bytes_per_worker);
    for(auto x:{caps.maximum_owned_numerical_bytes,caps.maximum_per_worker_inventoried_bytes,
        caps.maximum_node_inventoried_bytes,caps.maximum_work_units}) h.u64(x);
    h.string(target.frame().is_direct_gram() ? direct_overlap_policy : overlap_policy);
    const auto sha=h.finish(); std::copy(sha.begin(),sha.end(),p.identity_ascii.begin()); return p;
}
const std::vector<double>& Overlap::overlaps() const {
    if(!state_ || !context_ || values_.size()!=mul(memory_.target_dimension,memory_.source_dimension))
        throw std::logic_error("Gaussian pair overlap is moved or malformed");
    return values_;
}
Overlap make_periodic_gaussian_pair_overlap(const PeriodicCorrelationAdmittedReference& ref,
    const Space& target,const Space& source,const Live& live,const OverlapCaps& caps) {
    const auto p=plan_periodic_gaussian_pair_overlap(ref,target,source,live,caps);
    space_payload(target); if(!p.same_object_borrower) space_payload(source);
    const auto r=p.target_dimension,s=p.source_dimension,n=p.virtual_count;
    Overlap result; result.memory_=p; result.state_=ref.state_handle(); result.context_=target.context_handle();
    result.target_=target.identity_sha256(); result.source_=source.identity_sha256();
    result.values_.resize(static_cast<std::size_t>(mul(r,s)));
    const auto& a=target.coefficients(); const auto& b=source.coefficients();
    for(I i=0;i<r;++i) for(I j=0;j<s;++j) {
        double sum=0,correction=0;
        for(I k=0;k<n;++k) accumulate(product(a[k*r+i],b[k*s+j],result.underflows_),sum,correction);
        result.values_[i*s+j]=real::finite(sum+correction);
    }
    Digest payload("vibeqc.periodic.gaussian-pair-overlap.payload"); payload.u64(n); payload.u64(r); payload.u64(s);
    for(double x:result.values_) payload.real(x);
    result.payload_=payload.finish();
    Digest h("vibeqc.periodic.gaussian-pair-overlap.identity");
    if(target.frame().is_direct_gram()) {
        // Keep each source's distinct selected-Gram lineage. No invented
        // global provider identity and no requirement that pair selections match.
        for(const auto& x:{ref.state().state_identity_sha256(),ref.dimensions().allocation_identity,
            target.context_handle()->source_context_identity_sha256(),target.local_basis_identity_sha256(),
            target.frame().basis_identity_sha256(),target.frame().hf_reference_source_identity_sha256(),
            target.frame().direct_gram().gram_identity_sha256(),source.frame().direct_gram().gram_identity_sha256(),
            target.consumed_sources_identity_sha256(),source.consumed_sources_identity_sha256(),
            result.target_,result.source_,result.payload_}) h.string(x);
    } else {
        for(const auto& x:{ref.state().state_identity_sha256(),ref.dimensions().allocation_identity,
            target.context_handle()->source_context_identity_sha256(),target.local_basis_identity_sha256(),
            target.frame().provider_identity_sha256(),target.frame().hf_reference_source_identity_sha256(),
            result.target_,result.source_,result.payload_}) h.string(x);
    }
    h.u64(result.underflows_); h.string(target.frame().is_direct_gram() ? direct_overlap_policy : overlap_policy); h.string(real::kFloatPolicy);
    result.identity_=h.finish(); return result;
}

} // namespace vibeqc
