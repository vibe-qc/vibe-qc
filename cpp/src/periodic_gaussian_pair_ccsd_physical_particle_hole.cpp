#include "vibeqc/periodic_gaussian_pair_ccsd_physical_particle_hole.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace real = periodic_correlation_real_local_detail;
namespace local = periodic_correlation_local_detail;
using U = std::uint64_t;
using C = std::complex<double>;
using real::add;
using real::mul;
using real::Digest;
using Plan = PeriodicGaussianPairCCSDPhysicalParticleHolePlan;
using Result = PeriodicGaussianPairCCSDResult;
using Producer = BoundedRestrictedPairCCSDParticleHoleProducer;
using Reader = BoundedRestrictedPairCCSDAmplitudes;
using SolverPlan = BoundedRestrictedPairCCSDSolverMemoryPlan;
using Config = PeriodicGaussianPairInteractionConfig;
using InteractionOptions = PeriodicGaussianPairInteractionOptions;
using PHCaps = PeriodicGaussianPairParticleHoleCaps;
using Options = PeriodicGaussianPairCCSDOptions;
using Live = PeriodicGaussianPairCCSDLiveInventory;
using Caps = PeriodicGaussianPairCCSDCaps;
static_assert(sizeof(C)==16 && FLT_EVAL_METHOD==0,"physical Gaussian pair CCSD requires binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "physical Gaussian pair CCSD forbids fast/finite-only math"
#endif
constexpr char policy[] = "native-current-reader;all-q-original-auxiliary-Gaussian-KJO;"
    "bare-W1-W2-WX-replaced-before-contraction;all-two-occupied-legs;"
    "remaining-dressed-groups-common-frame;no-screened-coupling;"
    "new-real-projection-not-origin-provider-bitwise-proof;not-production-DLPNO";
U subtract(U a,U b) {
    if(b>a) throw std::logic_error("physical Gaussian pair CCSD ownership subtraction is inconsistent");
    return a-b;
}
U triangle(U n) { return n%2 ? mul(n,add(n,1)/2) : mul(n/2,add(n,1)); }
void limit(U value,U cap,const char* message) { if(value>cap) throw std::length_error(message); }
void positive(U value) {
    if(!value) throw std::invalid_argument("physical Gaussian pair CCSD requires positive explicit controls");
}
void sha(const std::string& value) {
    if(value.size()!=64) throw std::invalid_argument("physical Gaussian pair CCSD requires native SHA256 receipts");
    for(char c:value) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("physical Gaussian pair CCSD receipt is malformed");
}
struct Sources {
    const PeriodicGaussianRHFResult& hf;
    const PeriodicCorrelationAdmittedReference& ref;
    const PeriodicCorrelationWannier& wannier;
    const PeriodicCorrelationPAODomain& domain;
    const PeriodicCorrelationPAOSpace& space;
    const PeriodicCorrelationRealLocalBasis& basis;
    const PeriodicGaussianRealLocalProvider& provider;
    const PeriodicGaussianPairMP2Result& warm;
};
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& c) {
    const auto& v=c.inventory(); const auto& a=v.ao; const auto& b=v.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),c.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),
        add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),v.combined_borrowed_active_numeric_bytes,v.work_units_upper_bound};
}
void options_valid(const Config& config,const InteractionOptions& options,const PHCaps& caps) {
    positive(config.auxiliary_block); positive(config.panel.ao_pair_block);
    const auto& p=options.real_projection;
    real::tolerance_pair(p.reversal_absolute_tolerance,p.reversal_relative_tolerance);
    real::tolerance_pair(p.conjugacy_absolute_tolerance,p.conjugacy_relative_tolerance);
    real::tolerance_pair(p.self_q_absolute_tolerance,p.self_q_relative_tolerance);
    for(double x:{p.maximum_eri_projection_error,p.maximum_scalar_roundoff_error,options.maximum_overlap_imaginary_norm})
        if(!std::isfinite(x) || x<=0)
            throw std::invalid_argument("physical Gaussian pair CCSD requires positive finite projection controls");
    for(U x:{caps.maximum_source_slots,caps.maximum_pair_count,caps.maximum_owned_numerical_bytes,
        caps.maximum_control_storage_bytes,caps.maximum_worker_bytes,caps.maximum_node_bytes,
        caps.maximum_work_units,caps.maximum_interaction_control_bytes,caps.maximum_factor_panels,
        caps.maximum_tile_calls,caps.maximum_reciprocal_candidates,caps.maximum_image_candidates}) positive(x);
    for(const auto* resources:{&caps.interaction.resources,&caps.interaction.metric,
        &caps.interaction.panel.resources,&caps.interaction.panel.tile.resources})
        for(U x:{resources->maximum_owned_numeric_bytes,resources->maximum_per_replica_inventoried_bytes,
            resources->maximum_node_inventoried_bytes,resources->maximum_candidate_evaluations,resources->maximum_work_units}) positive(x);
    for(U x:{caps.interaction.maximum_factor_panels,caps.interaction.maximum_tile_calls,
        caps.interaction.maximum_image_candidate_evaluations,caps.interaction.maximum_leaf_control_bytes,
        caps.interaction.panel.maximum_tile_calls,caps.interaction.panel.maximum_image_candidate_evaluations,
        caps.interaction.panel.tile.maximum_image_candidates}) positive(x);
}
void metadata(const Sources& s,const Plan& p) {
    // Owner-liveness checks precede all borrowed-buffer access. No payload scan.
    local::validate_wannier(s.ref,s.wannier); local::validate_pao(s.ref,s.domain,s.space);
    const auto state=s.ref.state_handle(); const auto context=s.hf.context_handle();
    const auto& native=s.provider.provider();
    (void)native.integral_provider(s.basis);
    if(!context || !s.hf.converged() || !s.hf.matched_finite_gaussian_hf_source()
        || s.hf.state_handle()!=state || s.basis.state_handle()!=state || native.state_handle()!=state
        || s.provider.context_handle()!=context || s.warm.context_handle()!=context || s.warm.state_handle()!=state
        || !s.warm.converged() || s.basis.allocation_identity()!=s.ref.dimensions().allocation_identity
        || s.basis.memory().occupied_count!=p.occupied_count || s.basis.memory().virtual_count!=p.common_virtual_dimension
        || s.warm.memory().occupied_count!=p.occupied_count || s.warm.memory().common_virtual_dimension!=p.common_virtual_dimension
        || s.warm.memory().pair_count!=p.pair_count || s.warm.diagnostics().completed_pairs!=p.pair_count
        || s.warm.domain_generated()!=p.domain_generated
        || s.warm.provider_identity_sha256()!=s.provider.identity_sha256()
        || s.warm.hf_reference_source_identity_sha256()!=s.hf.reference_source_identity_sha256()
        || s.provider.hf_reference_source_identity_sha256()!=s.hf.reference_source_identity_sha256())
        throw std::invalid_argument("physical Gaussian pair CCSD requires identical HF/reference/basis/provider/warm owners");
    const auto& domain=s.domain.pao_domain_identity_sha256(); const auto& space=s.space.pao_space_identity_sha256();
    sha(domain); sha(space);
    if(!std::equal(domain.begin(),domain.end(),s.basis.virtual_domain_identity_ascii().begin())
        || !std::equal(space.begin(),space.end(),s.basis.virtual_space_identity_ascii().begin()))
        throw std::invalid_argument("physical Gaussian pair CCSD common domain or space differs from original basis");
    const auto actual=real::local_basis_identity(s.ref,s.wannier,s.wannier.gauge_payload_sha256(),s.domain,s.space,
        s.basis.occupied_indices_data(),p.occupied_count,s.basis.virtual_selection());
    if(actual!=s.basis.local_basis_identity_sha256())
        throw std::invalid_argument("physical Gaussian pair CCSD Wannier/common selection differs from original basis");
}
void options_wire(Digest& h,const Config& c,const InteractionOptions& o) {
    h.u64(c.auxiliary_block); h.u64(c.panel.ao_pair_block);
    const auto& r=o.real_projection;
    for(double x:{r.reversal_absolute_tolerance,r.reversal_relative_tolerance,
        r.conjugacy_absolute_tolerance,r.conjugacy_relative_tolerance,r.self_q_absolute_tolerance,
        r.self_q_relative_tolerance,r.maximum_eri_projection_error,r.maximum_scalar_roundoff_error,
        o.maximum_overlap_imaginary_norm}) h.real(x);
}
std::string source_identity(const Sources& s,const Config& config,const InteractionOptions& options) {
    Digest h("vibeqc.periodic.gaussian-pair-ccsd.physical-PH-source");
    for(const auto* value:std::initializer_list<const std::string*>{&s.ref.state().state_identity_sha256(),
        &s.ref.dimensions().allocation_identity,&s.wannier.wannier_identity_sha256(),&s.wannier.gauge_payload_sha256(),
        &s.domain.pao_domain_identity_sha256(),&s.space.pao_space_identity_sha256(),&s.basis.identity_sha256(),
        &s.provider.identity_sha256(),&s.provider.consumed_sources_identity_sha256(),&s.warm.identity_sha256(),
        &s.warm.pair_spaces_identity_sha256()}) { sha(*value); h.string(*value); }
    h.string(s.hf.reference_source_identity_sha256()); h.string(s.hf.context_handle()->source_context_identity_sha256());
    options_wire(h,config,options); h.string(policy); return h.finish();
}
struct Pins {
    const std::uint64_t* occupied = nullptr;
    const double *foo=nullptr,*fvv=nullptr,*fov=nullptr,*eps=nullptr,*seigen=nullptr;
    const C *wannier=nullptr,*overlap=nullptr,*fock=nullptr,*coefficients=nullptr;
    static Pins capture(const Sources& s) {
        return {s.basis.occupied_indices_data(),s.basis.f_oo_data(),s.basis.f_vv_data(),s.basis.f_ov_data(),
            s.space.energies_data(),s.space.overlap_eigenvalues_data(),s.wannier.cell_coefficients(0),
            s.domain.overlap_data(),s.domain.fock_data(),s.space.coefficients_data()};
    }
    void check(const Sources& s) const {
        const auto p=capture(s);
        if(occupied!=p.occupied || foo!=p.foo || fvv!=p.fvv || fov!=p.fov || eps!=p.eps || seigen!=p.seigen
            || wannier!=p.wannier || overlap!=p.overlap || fock!=p.fock || coefficients!=p.coefficients)
            throw std::invalid_argument("physical Gaussian pair CCSD source owner storage changed at callback");
    }
};
// Numerical content replay uses the actual admitted owners, not exported
// reconstructed arrays. It detects callback mutation; it does not replace
// the original constructors' physical proofs with caller-labelled digests.
std::string physical_payload(const Sources& s,const BasisSet& ao,const BasisSet& aux,const C* gauges,std::size_t count) {
    const auto context=s.hf.context_handle();
    context->verify_bases(ao,aux,exact_basis_caps(*context));
    const auto gauge=local::validate_gauges(s.wannier,gauges,count);
    Digest h("vibeqc.periodic.gaussian-pair-ccsd.physical-PH-borrowed-payload"); h.string(gauge);
    const U k=s.wannier.n_cells(),nm=mul(s.wannier.n_basis(),s.wannier.n_home_occupied());
    h.u64(k); h.u64(nm);
    for(U cell=0;cell<k;++cell) {
        const auto* values=s.wannier.cell_coefficients(cell);
        for(U x=0;x<nm;++x) h.complex(values[x]);
    }
    const U d=s.domain.domain_dimension(),r=s.space.retained_dimension(); h.u64(d); h.u64(r);
    for(U x=0;x<d;++x) { const auto column=s.domain.column(x); h.u64(column.cell); h.u64(column.ao); }
    for(U x=0;x<mul(d,d);++x) { h.complex(s.domain.overlap_data()[x]); h.complex(s.domain.fock_data()[x]); }
    for(U x=0;x<mul(d,r);++x) h.complex(s.space.coefficients_data()[x]);
    for(U x=0;x<r;++x) h.real(s.space.energies_data()[x]);
    for(U x=0;x<d;++x) h.real(s.space.overlap_eigenvalues_data()[x]);
    const U o=s.basis.memory().occupied_count,n=s.basis.memory().virtual_count;
    for(U x=0;x<mul(2,o);++x) h.u64(s.basis.occupied_indices_data()[x]);
    for(U x=0;x<mul(o,o);++x) h.real(s.basis.f_oo_data()[x]);
    for(U x=0;x<mul(n,n);++x) h.real(s.basis.f_vv_data()[x]);
    for(U x=0;x<mul(o,n);++x) h.real(s.basis.f_ov_data()[x]);
    return h.finish();
}
struct Context {
    Sources sources;
    const BasisSet* ao=nullptr; const BasisSet* auxiliary=nullptr;
    const C* gauges=nullptr; std::size_t gauge_count=0;
    Config config; InteractionOptions interaction; PHCaps caps; Options options; Live live;
    Plan plan; Pins pins;
    std::string source,payload;
    PeriodicGaussianPairCCSDCallback progress=nullptr; void* progress_context=nullptr;
    mutable Digest consumed{"vibeqc.periodic.gaussian-pair-ccsd.physical-PH-consumption"};
    mutable U calls=0,source_checks=0,panels=0,tiles=0,reciprocal=0,images=0;
    void verify() const {
        if(source_checks>=plan.source_validation_passes_upper_bound)
            throw std::length_error("physical Gaussian pair CCSD source-validation count cap exceeded");
        ++source_checks;
        real::float_environment(); metadata(sources,plan); pins.check(sources);
        if(source_identity(sources,config,interaction)!=source
            || physical_payload(sources,*ao,*auxiliary,gauges,gauge_count)!=payload)
            throw std::invalid_argument("physical Gaussian pair CCSD immutable source changed at callback");
    }
    static void event(const PeriodicGaussianPairCCSDProgress& p,void* raw) {
        auto& self=*static_cast<Context*>(raw); self.verify();
        if(self.progress) self.progress(p,self.progress_context);
        // Before returning to the enclosing Events/generic Fock scans.
        self.verify();
    }
    static void produce(const Reader& reader,const SolverPlan& exact,U i,U j,U snapshot,
        BoundedRestrictedPairCCSDParticleHoleVisitor visitor,void* visitor_context,const void* raw) {
        const auto& self=*static_cast<const Context*>(raw); const auto& p=self.plan; const auto& s=self.sources;
        if(self.calls>=p.particle_hole_calls_upper_bound)
            throw std::length_error("physical Gaussian pair CCSD particle-hole count cap exceeded");
        self.verify();
        if(!snapshot || !visitor || !exact.split_bare_particle_hole
            || exact.n_occupied!=p.occupied_count || exact.common_virtual_dimension!=p.common_virtual_dimension
            || exact.pair_count!=p.pair_count || reader.memory().n_occupied!=p.occupied_count
            || reader.memory().common_virtual_dimension!=p.common_virtual_dimension
            || mul(8,reader.memory().pair_coefficient_elements)!=p.pair_coefficient_bytes
            || exact.particle_hole_retained_numerical_bytes!=p.additional_retained_numerical_bytes
            || exact.particle_hole_transient_numerical_bytes!=p.producer_transient_upper_bytes
            || exact.per_replica_inventoried_bytes>p.ccsd.solver_upper.per_replica_inventoried_bytes)
            throw std::invalid_argument("physical Gaussian pair CCSD exact solver/source census differs");
        const auto& dims=s.ref.dimensions(); const auto& budget=s.ref.budget();
        const U replicas=mul(budget.mpi_ranks,budget.workers_per_rank);
        const U reference_base=add(add(dims.external_bytes,dims.shared_bytes),
            mul(budget.mpi_ranks,add(dims.per_rank_bytes,dims.localization_window_bytes_per_rank)));
        if(replicas!=p.ccsd.replicas_per_node || reference_base!=p.ccsd.reference_base_node_bytes
            || exact.required_node_inventoried_bytes!=add(reference_base,mul(replicas,exact.per_replica_inventoried_bytes))
            || exact.required_node_inventoried_bytes>p.ccsd.solver_upper.required_node_inventoried_bytes
            || exact.required_node_inventoried_bytes>budget.memory_limit_bytes)
            throw std::invalid_argument("physical Gaussian pair CCSD admitted reference resource owner changed");
        const U phase=std::max(exact.remaining_target_phase_bytes,exact.local_particle_hole_phase_bytes);
        const U resident=subtract(exact.per_replica_inventoried_bytes,phase);
        // Preserve our own Context/sealing controls. Remove only the PH leaf
        // control ceiling; it is re-added by that leaf, bounded by this cap.
        const U base=subtract(subtract(resident,p.additional_retained_numerical_bytes),self.caps.maximum_control_storage_bytes);
        const U known=subtract(add(p.borrowed_warmstart_numerical_bytes,
            add(p.borrowed_basis_numerical_bytes,reader.memory().borrowed_numerical_bytes)),p.pair_coefficient_bytes);
        PeriodicGaussianPairParticleHoleLiveInventory live;
        // This mixed channel includes remaining solver/control owners. It is
        // a declared live inventory, not a claim those bytes are a new array.
        live.other_live_numerical_bytes_per_worker=subtract(subtract(base,known),self.live.fixed_backend_margin_bytes_per_worker);
        live.backend_margin_bytes_per_worker=self.live.fixed_backend_margin_bytes_per_worker;
        // Enforce the exact parent's envelope INSIDE the child's preflight,
        // not only after its factories have allocated and returned.
        auto child_caps=self.caps;
        child_caps.maximum_worker_bytes=std::min(child_caps.maximum_worker_bytes,exact.per_replica_inventoried_bytes);
        child_caps.maximum_node_bytes=std::min(child_caps.maximum_node_bytes,exact.required_node_inventoried_bytes);
        auto result=build_periodic_gaussian_pair_particle_hole_snapshot(s.hf,s.ref,*self.ao,*self.auxiliary,
            s.wannier,self.gauges,self.gauge_count,s.domain,s.space,s.basis,s.warm,reader,i,j,snapshot,
            self.config,self.interaction,live,child_caps);
        const auto& m=result.memory(); const auto& d=result.diagnostics(); const auto& stream=m.stream;
        if(result.state_handle()!=s.ref.state_handle() || result.context_handle()!=s.hf.context_handle()
            || result.snapshot_index()!=snapshot || result.snapshot_identity_sha256()!=reader.snapshot_identity_sha256()
            || result.warmstart_identity_sha256()!=s.warm.identity_sha256()
            || result.origin_provider_identity_sha256()!=s.provider.identity_sha256()
            || result.hf_reference_source_identity_sha256()!=s.hf.reference_source_identity_sha256()
            || result.pair_spaces_identity_sha256()!=s.warm.pair_spaces_identity_sha256()
            || m.reader_warm_coefficient_padding_bytes!=p.pair_coefficient_bytes
            || stream.borrowed_warmstart_numerical_bytes!=p.borrowed_warmstart_numerical_bytes
            || stream.replicas_per_node!=p.ccsd.replicas_per_node
            || stream.reference_base_node_bytes!=p.ccsd.reference_base_node_bytes
            || stream.worker_bytes>exact.per_replica_inventoried_bytes
            || stream.required_node_memory_bytes>exact.required_node_inventoried_bytes
            || stream.peak_owned_numerical_bytes>self.caps.maximum_owned_numerical_bytes
            || stream.control_storage_reservation_bytes>self.caps.maximum_control_storage_bytes
            || stream.work_units>self.caps.maximum_work_units)
            throw std::logic_error("physical Gaussian pair CCSD consumed PH source exceeds exact solver admission");
        self.verify();
        self.consumed.u64(snapshot); self.consumed.u64(i); self.consumed.u64(j);
        for(const auto* receipt:{&result.identity_sha256(),&result.payload_sha256(),
            &result.consumed_interactions_identity_sha256(),&result.snapshot_identity_sha256()}) {
            sha(*receipt); self.consumed.string(*receipt);
        }
        self.panels=add(self.panels,d.completed_factor_panels); self.tiles=add(self.tiles,d.completed_tile_calls);
        self.reciprocal=add(self.reciprocal,d.reciprocal_candidate_evaluations); self.images=add(self.images,d.image_candidate_evaluations);
        limit(self.panels,p.factor_panels_upper_bound,"physical Gaussian pair CCSD aggregate factor-panel cap exceeded");
        limit(self.tiles,p.tile_calls_upper_bound,"physical Gaussian pair CCSD aggregate tile cap exceeded");
        limit(self.reciprocal,p.reciprocal_candidates_upper_bound,"physical Gaussian pair CCSD aggregate reciprocal cap exceeded");
        limit(self.images,p.image_candidates_upper_bound,"physical Gaussian pair CCSD aggregate image cap exceeded");
        ++self.calls;
        visitor(result.bare_result(),result.snapshot_identity_sha256(),visitor_context);
    }
};
constexpr U fixed_controls=131072U+sizeof(Context)+3U*sizeof(Plan)+sizeof(Sources)+sizeof(Pins)
    +2U*sizeof(Producer)+4U*sizeof(Digest)+24U*65U;
Producer producer(const Plan& p,const PHCaps& caps,const std::string& identity,const Context* context) {
    sha(identity); Producer out;
    out.produce=&Context::produce; out.context=context;
    std::copy(identity.begin(),identity.end(),out.identity_sha256.begin());
    out.additional_retained_numerical_bytes=p.additional_retained_numerical_bytes;
    out.maximum_transient_numerical_bytes_per_target=p.producer_transient_upper_bytes;
    out.additional_control_storage_bytes=p.producer_additional_control_storage_bytes;
    out.maximum_work_units_per_target=add(caps.maximum_work_units,
        add(mul(2,p.source_validation_work_units_per_pass),mul(16384,add(1024,mul(2,p.occupied_count)))));
    return out;
}
void pair_census(const Sources& s,Plan& p) {
    U frames=0,geometry=0,coefficients=0,ranks=0,maximum_rank=0;
    for(U i=0;i<p.occupied_count;++i) for(U j=i;j<p.occupied_count;++j) {
        const auto& pair=s.warm.pair(i,j); const auto frame=pair.frame();
        const U r=frame.retained_dimension(),n=p.common_virtual_dimension;
        if(r>n || pair.memory().retained_dimension!=r || pair.coefficients().size()!=mul(n,r)
            || frame.state_handle()!=s.ref.state_handle() || frame.context_handle()!=s.hf.context_handle()
            || frame.basis_identity_sha256()!=s.basis.identity_sha256()
            || frame.provider_identity_sha256()!=s.provider.identity_sha256()
            || frame.occupied_slot_i()!=i || frame.occupied_slot_j()!=j)
            throw std::invalid_argument("physical Gaussian pair CCSD authentic pair census differs");
        coefficients=add(coefficients,mul(8,mul(n,r))); ranks=add(ranks,r); maximum_rank=std::max(maximum_rank,r);
        frames=add(frames,pair.memory().retained_output_bytes);
        p.maximum_frame_numerical_bytes=std::max(p.maximum_frame_numerical_bytes,frame.retained_numerical_bytes());
        if(p.domain_generated) {
            const auto& g=s.warm.pair_generation_geometry(i,j);
            geometry=add(geometry,g.retained_numerical_bytes());
            p.maximum_geometry_numerical_bytes=std::max(p.maximum_geometry_numerical_bytes,g.retained_numerical_bytes());
        }
    }
    if(frames!=s.warm.diagnostics().retained_pair_bytes || geometry!=s.warm.diagnostics().retained_pair_geometry_bytes
        || maximum_rank!=s.warm.diagnostics().maximum_pair_rank)
        throw std::logic_error("physical Gaussian pair CCSD complete warm-owner census differs");
    p.maximum_pair_rank=maximum_rank; p.pair_coefficient_bytes=coefficients;
    p.borrowed_warmstart_numerical_bytes=add(add(frames,geometry),s.warm.solver().memory().output_numerical_bytes);
    if(coefficients!=mul(8,mul(p.common_virtual_dimension,ranks)))
        throw std::logic_error("physical Gaussian pair CCSD coefficient census overflows");
}
Plan plan_impl(const Sources& s,const Config& config,const InteractionOptions& interaction,const PHCaps& phcaps,
    const Options& options,const Live& live,const Caps& caps) {
    real::float_environment(); options_valid(config,interaction,phcaps); positive(live.fixed_backend_margin_bytes_per_worker);
    Plan p; p.occupied_count=s.basis.memory().occupied_count; p.common_virtual_dimension=s.basis.memory().virtual_count;
    positive(p.occupied_count); positive(p.common_virtual_dimension); p.pair_count=triangle(p.occupied_count);
    p.domain_generated=s.warm.domain_generated();
    limit(p.occupied_count,caps.solver.maximum_occupied_count,"physical Gaussian pair CCSD occupied cap exceeded");
    limit(p.common_virtual_dimension,caps.solver.maximum_common_virtual_dimension,"physical Gaussian pair CCSD common cap exceeded");
    limit(p.pair_count,caps.maximum_pair_count,"physical Gaussian pair CCSD pair count cap exceeded");
    limit(p.pair_count,phcaps.maximum_pair_count,"physical Gaussian pair CCSD PH pair count cap exceeded");
    limit(mul(2,p.occupied_count),phcaps.maximum_source_slots,"physical Gaussian pair CCSD PH source-slot cap exceeded");
    p.wrapper_fixed_control_storage_bytes=fixed_controls;
    p.producer_additional_control_storage_bytes=add(phcaps.maximum_control_storage_bytes,fixed_controls);
    limit(p.producer_additional_control_storage_bytes,caps.maximum_per_worker_inventoried_bytes,
        "physical Gaussian pair CCSD startup control cap exceeded");
    // Fixed conservative pre-admission covers our metadata/receipt walks and
    // the extra count-only CCSD plan performed by this enclosing factory.
    p.metadata_work_units=mul(1048576,add(1024,add(p.pair_count,p.occupied_count)));
    limit(p.metadata_work_units,caps.maximum_work_units,"physical Gaussian pair CCSD startup metadata work cap exceeded");
    metadata(s,p); pair_census(s,p);
    p.borrowed_basis_numerical_bytes=s.basis.memory().retained_output_bytes;
    p.additional_retained_numerical_bytes=add(s.wannier.memory().caller_gauge_bytes,
        add(s.wannier.memory().retained_coefficient_bytes,add(s.domain.memory().retained_domain_index_bytes,
        add(s.domain.memory().retained_matrix_bytes,add(s.space.memory().output_numerical_bytes,
        s.hf.context_handle()->inventory().combined_borrowed_active_numeric_bytes)))));
    p.duplicate_frame_geometry_padding_bytes=mul(2,add(p.maximum_frame_numerical_bytes,p.maximum_geometry_numerical_bytes));
    p.duplicate_t_role_padding_bytes=mul(8,mul(p.maximum_pair_rank,p.maximum_pair_rank));
    p.producer_transient_upper_bytes=add(phcaps.maximum_owned_numerical_bytes,
        add(p.pair_coefficient_bytes,std::max(p.duplicate_frame_geometry_padding_bytes,p.duplicate_t_role_padding_bytes)));
    const U lanes=add(p.additional_retained_numerical_bytes,p.borrowed_basis_numerical_bytes)/8;
    p.source_validation_work_units_per_pass=add(s.hf.context_handle()->inventory().work_units_upper_bound,
        mul(4096,add(2048,add(lanes,add(p.pair_count,mul(p.occupied_count,p.occupied_count))))));
    const auto identity=source_identity(s,config,interaction);
    const auto ph=producer(p,phcaps,identity,nullptr);
    p.ccsd=plan_periodic_gaussian_pair_ccsd(s.ref,s.basis,s.provider,s.warm,ph,options,live,caps);
    if(p.ccsd.borrowed_mp2_numerical_bytes!=p.borrowed_warmstart_numerical_bytes
        || p.ccsd.borrowed_basis_bytes!=p.borrowed_basis_numerical_bytes
        || p.ccsd.borrowed_particle_hole_additional_numerical_bytes!=p.additional_retained_numerical_bytes)
        throw std::logic_error("physical Gaussian pair CCSD enclosing source inventory differs");
    p.particle_hole_calls_upper_bound=p.ccsd.solver_upper.particle_hole_calls_upper_bound;
    p.source_validation_passes_upper_bound=add(4,mul(2,add(p.ccsd.progress_callback_upper_bound,p.particle_hole_calls_upper_bound)));
    const U amplitudes=add(p.ccsd.solver_upper.total_singles_elements,p.ccsd.solver_upper.total_doubles_elements);
    p.finalization_work_units=add(mul(512,add(2048,add(add(p.occupied_count,p.pair_count),amplitudes))),65536);
    // Per-target pre/post source replays belong to producer work above.
    // Startup/finalization and normal progress replays are separate here.
    p.wrapper_work_units_upper_bound=add(p.metadata_work_units,add(p.finalization_work_units,
        mul(add(4,mul(2,p.ccsd.progress_callback_upper_bound)),p.source_validation_work_units_per_pass)));
    p.work_units_upper_bound=add(p.ccsd.work_units_upper_bound,p.wrapper_work_units_upper_bound);
    p.factor_panels_upper_bound=mul(p.particle_hole_calls_upper_bound,phcaps.maximum_factor_panels);
    p.tile_calls_upper_bound=mul(p.particle_hole_calls_upper_bound,phcaps.maximum_tile_calls);
    p.reciprocal_candidates_upper_bound=mul(p.particle_hole_calls_upper_bound,phcaps.maximum_reciprocal_candidates);
    p.image_candidates_upper_bound=mul(p.particle_hole_calls_upper_bound,phcaps.maximum_image_candidates);
    p.peak_owned_numerical_bytes=p.ccsd.peak_owned_numerical_bytes;
    p.per_worker_inventoried_bytes=p.ccsd.per_worker_inventoried_bytes;
    p.required_node_memory_bytes=p.ccsd.required_node_memory_bytes;
    limit(p.work_units_upper_bound,caps.maximum_work_units,"physical Gaussian pair CCSD complete wrapper work cap exceeded");
    for(U bytes:{p.producer_transient_upper_bytes,p.additional_retained_numerical_bytes,
        p.per_worker_inventoried_bytes,p.required_node_memory_bytes}) real::extent(bytes);
    return p;
}
} // namespace

// No public promotion operation. This TU alone creates the actual producer,
// checks every physical receipt/owner and re-seals its consumed source chain.
struct PeriodicGaussianPairCCSDPhysicalParticleHoleAccess {
    static void qualify(Result& result,const std::string& receipt) {
        if(!result.split_bare_particle_hole() || result.physical_particle_hole_source_
            || !result.physical_particle_hole_source_identity_.empty())
            throw std::logic_error("physical Gaussian pair CCSD qualification requires a fresh generic split result");
        sha(receipt); sha(result.identity_); sha(result.solver().split_execution_identity_sha256());
        sha(result.solver().consumed_particle_hole_identity_sha256());
        Digest identity("vibeqc.periodic.gaussian-pair-ccsd.physical-PH-qualified");
        identity.string(result.identity_); identity.string(receipt); identity.string(policy);
        const auto sealed=identity.finish();
        result.physical_particle_hole_source_identity_=receipt; result.identity_=sealed;
        result.physical_particle_hole_source_=true;
    }
};

Plan plan_periodic_gaussian_pair_ccsd_physical_particle_hole(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm,const Config& config,const InteractionOptions& interaction,const PHCaps& phcaps,
    const Options& options,const Live& live,const Caps& caps) {
    return plan_impl({hf,ref,wannier,domain,space,basis,provider,warm},config,interaction,phcaps,options,live,caps);
}

Result run_periodic_gaussian_pair_ccsd_physical_particle_hole(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,const BasisSet& ao,const BasisSet& auxiliary,
    const PeriodicCorrelationWannier& wannier,const C* gauges,std::size_t gauge_count,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& warm,const Config& supplied_config,const InteractionOptions& supplied_interaction,
    const PHCaps& supplied_phcaps,const Options& supplied_options,const Live& supplied_live,const Caps& supplied_caps,
    PeriodicGaussianPairCCSDCallback progress,void* progress_context) {
    const auto config=supplied_config; const auto interaction=supplied_interaction; const auto phcaps=supplied_phcaps;
    const auto options=supplied_options; const auto live=supplied_live; const auto caps=supplied_caps;
    const Sources sources{hf,ref,wannier,domain,space,basis,provider,warm};
    const auto p=plan_impl(sources,config,interaction,phcaps,options,live,caps);
    const auto pins=Pins::capture(sources);
    const auto source=source_identity(sources,config,interaction);
    const auto payload=physical_payload(sources,ao,auxiliary,gauges,gauge_count);
    Context context{sources,&ao,&auxiliary,gauges,gauge_count,config,interaction,phcaps,options,live,p,pins,
        source,payload,progress,progress_context};
    context.source_checks=1;
    const auto ph=producer(p,phcaps,source,&context);
    auto result=run_periodic_gaussian_pair_ccsd(ref,basis,provider,warm,ph,options,live,caps,&Context::event,&context);
    context.verify();
    const auto& last=result.solver().final_snapshot();
    if(result.state_handle()!=ref.state_handle() || result.context_handle()!=hf.context_handle()
        || result.warmstart_identity_sha256()!=warm.identity_sha256() || result.provider_identity_sha256()!=provider.identity_sha256()
        || result.pair_spaces_identity_sha256()!=warm.pair_spaces_identity_sha256()
        || result.basis_identity_sha256()!=basis.identity_sha256()
        || !result.split_bare_particle_hole() || result.particle_hole_physical_source_certified()
        || context.calls!=last.particle_hole_calls || context.calls!=last.particle_hole_visits
        || context.calls!=mul(last.iteration,p.pair_count)
        || result.memory().per_worker_inventoried_bytes!=p.ccsd.per_worker_inventoried_bytes)
        throw std::logic_error("physical Gaussian pair CCSD completed execution/source receipts differ");
    const auto validation=plan_bounded_restricted_pair_ccsd_solver_payload_validation(result.solver());
    limit(validation.work_units,p.finalization_work_units,"physical Gaussian pair CCSD final payload work cap exceeded");
    limit(validation.fixed_control_storage_bytes,p.wrapper_fixed_control_storage_bytes,
        "physical Gaussian pair CCSD final payload control cap exceeded");
    verify_bounded_restricted_pair_ccsd_solver_payload(result.solver(),validation.work_units);
    Digest receipt("vibeqc.periodic.gaussian-pair-ccsd.physical-PH-completed-source");
    receipt.string(source); receipt.string(payload); receipt.string(context.consumed.finish());
    receipt.string(result.solver().split_execution_identity_sha256());
    receipt.string(result.solver().consumed_particle_hole_identity_sha256());
    receipt.string(result.solver().payload_sha256()); receipt.u64(context.calls);
    receipt.u64(context.panels); receipt.u64(context.tiles); receipt.u64(context.reciprocal); receipt.u64(context.images);
    receipt.string(policy);
    PeriodicGaussianPairCCSDPhysicalParticleHoleAccess::qualify(result,receipt.finish());
    return result;
}
} // namespace vibeqc
