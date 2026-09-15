#include "vibeqc/periodic_gaussian_pair_particle_hole.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
namespace local=periodic_correlation_local_detail;
using U=std::uint64_t;
using real::add;using real::mul;using real::Digest;
using Plan=PeriodicGaussianPairParticleHolePlan;
using Result=PeriodicGaussianPairParticleHoleResult;
using Caps=PeriodicGaussianPairParticleHoleCaps;
using Live=PeriodicGaussianPairParticleHoleLiveInventory;
using Frame=PeriodicGaussianPairPNOFrameView;
using Geometry=PeriodicGaussianPairPNOGeometryView;
using Kernel=BoundedRestrictedPairCCSDParticleHoleAccumulator;
static_assert(FLT_EVAL_METHOD==0,"Gaussian particle-hole requires binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian particle-hole forbids fast/finite-only math"
#endif
void limit(U x,U cap,const char* message) {if(x>cap) throw std::length_error(message);}
U triangle(U o) {return o%2?mul(o,add(o,1)/2):mul(o/2,add(o,1));}
void sha(const std::string& s) {
    if(s.size()!=64) throw std::invalid_argument("Gaussian particle-hole receipt extent differs");
    for(char c:s) if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))
        throw std::invalid_argument("Gaussian particle-hole receipt is not lowercase SHA256");
}
bool placed_same(PeriodicCorrelationPlacedOccupied a,PeriodicCorrelationPlacedOccupied b) {
    return a.occupied_index==b.occupied_index && a.cell==b.cell;
}
struct Sources {
    const PeriodicGaussianRHFResult& hf;const PeriodicCorrelationAdmittedReference& ref;
    const PeriodicCorrelationWannier& wannier;const PeriodicCorrelationPAODomain& domain;
    const PeriodicCorrelationPAOSpace& space;const PeriodicCorrelationRealLocalBasis& basis;
    const PeriodicGaussianPairMP2Result& warm;const PeriodicGaussianPairCCSDResult* cc = nullptr;
    const BoundedRestrictedPairCCSDAmplitudes* reader = nullptr;
};
void basic(const Sources& s,Plan& p,U i,U j,const Live& live,const Caps& caps) {
    real::float_environment();
    p.occupied_count=s.basis.memory().occupied_count;p.common_virtual_dimension=s.basis.memory().virtual_count;
    p.source_slots=mul(2,p.occupied_count);p.pair_count=triangle(p.occupied_count);
    // These precede any owner record/source traversal.
    limit(p.source_slots,caps.maximum_source_slots,"Gaussian particle-hole source-slot cap exceeded");
    limit(p.pair_count,caps.maximum_pair_count,"Gaussian particle-hole pair-count cap exceeded");
    for(U v:{p.occupied_count,p.common_virtual_dimension,live.backend_margin_bytes_per_worker,
        caps.maximum_owned_numerical_bytes,caps.maximum_control_storage_bytes,caps.maximum_worker_bytes,
        caps.maximum_node_bytes,caps.maximum_work_units,caps.maximum_interaction_control_bytes})
        if(!v) throw std::invalid_argument("Gaussian particle-hole requires positive explicit resource controls");
    if(i>=p.occupied_count || j>=p.occupied_count) throw std::out_of_range("Gaussian particle-hole target occupied slot out of range");
    p.target_i=i;p.target_j=j;p.domain_generated=s.warm.domain_generated();
    local::validate_wannier(s.ref,s.wannier);local::validate_pao(s.ref,s.domain,s.space);
    const auto state=s.ref.state_handle();const auto context=s.hf.context_handle();
    const U o=p.occupied_count,n=p.common_virtual_dimension,P=p.pair_count;
    if(!context || !s.hf.converged() || !s.hf.matched_finite_gaussian_hf_source()
        || s.hf.state_handle()!=state || s.basis.state_handle()!=state
        || s.warm.state_handle()!=state || s.warm.context_handle()!=context || !s.warm.converged()
        || s.basis.allocation_identity()!=s.ref.dimensions().allocation_identity
        || s.warm.memory().occupied_count!=o || s.warm.memory().common_virtual_dimension!=n
        || s.warm.memory().pair_count!=P || s.warm.diagnostics().completed_pairs!=P
        || s.warm.hf_reference_source_identity_sha256()!=s.hf.reference_source_identity_sha256())
        throw std::invalid_argument("Gaussian particle-hole requires exact matching converged HF/MP2 owners");
    if(s.cc && s.cc->split_bare_particle_hole() && !s.cc->particle_hole_physical_source_certified())
        throw std::invalid_argument("Gaussian particle-hole frozen split CCSD source is not physically certified");
    if(s.cc && (s.cc->state_handle()!=state || s.cc->context_handle()!=context || !s.cc->converged()
        || s.cc->memory().occupied_count!=o || s.cc->memory().common_virtual_dimension!=n
        || s.cc->memory().pair_count!=P || s.cc->diagnostics().completed_singles!=o
        || s.cc->memory().domain_generated!=p.domain_generated
        || s.cc->warmstart_identity_sha256()!=s.warm.identity_sha256()
        || s.cc->pair_spaces_identity_sha256()!=s.warm.pair_spaces_identity_sha256()
        || s.cc->basis_identity_sha256()!=s.basis.identity_sha256()
        || s.cc->provider_identity_sha256()!=s.warm.provider_identity_sha256()
        || s.cc->hf_reference_source_identity_sha256()!=s.hf.reference_source_identity_sha256()))
        throw std::invalid_argument("Gaussian particle-hole requires exact matching converged HF/MP2/CCSD owners");
    const auto hf_identity=s.hf.reference_source_identity_sha256();
    for(const auto* value:{&s.warm.identity_sha256(),&s.warm.pair_spaces_identity_sha256(),
        &s.warm.provider_identity_sha256(),&hf_identity,&s.basis.identity_sha256()}) sha(*value);
    if(s.cc) for(const auto* value:{&s.cc->identity_sha256(),&s.cc->solver().payload_sha256(),
        &s.cc->solver().input_identity_sha256()}) sha(*value);
    if(s.warm.solver().memory().n_occupied!=o || s.warm.solver().memory().pair_count!=P
        || (s.cc && (s.cc->solver().memory().n_occupied!=o || s.cc->solver().memory().pair_count!=P)))
        throw std::invalid_argument("Gaussian particle-hole final snapshot dimensions differ");
    if((s.cc==nullptr)==(s.reader==nullptr)) throw std::logic_error("Gaussian particle-hole requires exactly one native snapshot source");
    if(s.reader && (s.reader->memory().n_occupied!=o || s.reader->memory().common_virtual_dimension!=n
        || s.reader->memory().pair_count!=P))
        throw std::invalid_argument("Gaussian particle-hole current Reader dimensions differ");
}
void frame_source(const Sources& s,const Frame& f,U i,U j) {
    if(f.state_handle()!=s.ref.state_handle() || f.context_handle()!=s.hf.context_handle()
        || f.basis_identity_sha256()!=s.basis.identity_sha256()
        || f.provider_identity_sha256()!=s.warm.provider_identity_sha256()
        || f.hf_reference_source_identity_sha256()!=s.hf.reference_source_identity_sha256()
        || f.occupied_count()!=s.basis.memory().occupied_count
        || f.common_virtual_dimension()!=s.basis.memory().virtual_count
        || f.occupied_slot_i()!=i || f.occupied_slot_j()!=j || f.is_embedded()!=s.warm.domain_generated()
        || !placed_same(f.occupied_i(),s.basis.occupied(i)) || !placed_same(f.occupied_j(),s.basis.occupied(j)))
        throw std::invalid_argument("Gaussian particle-hole frame source or occupied labels differ");
}
constexpr U fixed_controls=131072+3*sizeof(Plan)+2*sizeof(Result)+sizeof(Caps)+sizeof(Live)
    +sizeof(PeriodicGaussianPairInteractionConfig)+sizeof(PeriodicGaussianPairInteractionOptions)
    +sizeof(PeriodicGaussianPairInteractionBlocks)+3*sizeof(PeriodicGaussianPairInteractionPlan)
    +sizeof(Kernel)+2*sizeof(Geometry)+4*sizeof(Frame)+8*sizeof(Digest)+20*65;
// First bounded metadata pass: complete retained owner census, no floats.
void owners(const Sources& s,Plan& p) {
    const U o=p.occupied_count,n=p.common_virtual_dimension,P=p.pair_count;
    U pairs=0,singles=0,geometry=0,geometry_controls=0,pair_seals=0,single_seals=0,t2=0,t1=0,maximum_rank=0,visited_maximum_rank=0;
    for(U i=0;i<o;++i) {
        if(s.cc) {
        const auto f=s.cc->singles_frame(i);frame_source(s,f,i,i);
        const auto t=s.cc->solver().stored_singles_view(i);
        if(t.element_count!=f.retained_dimension() || s.cc->solver().singles_rank(i)!=f.retained_dimension())
            throw std::invalid_argument("Gaussian particle-hole singles extent differs");
        t1=add(t1,t.element_count);singles=add(singles,f.retained_numerical_bytes());
        single_seals=add(single_seals,f.retained_receipt_payload_bytes());
        p.frame_validation_work_units=add(p.frame_validation_work_units,plan_periodic_gaussian_pair_pno_frame_validation(f).work_units);
        } else {
            const auto q=s.reader->singles_view(i);
            if(q.rank>n || q.amplitudes.element_count!=q.rank || q.coefficients.element_count!=mul(n,q.rank))
                throw std::invalid_argument("Gaussian particle-hole numerical singles Reader extent differs");
            t1=add(t1,q.rank);
        }
        for(U j=i;j<o;++j) {
            const auto& pair=s.warm.pair(i,j);const auto f2=pair.frame();frame_source(s,f2,i,j);
            const U r=f2.retained_dimension();
            const auto t2view=s.cc?s.cc->solver().stored_pair_view(i,j):s.reader->canonical_pair_view(i,j).amplitudes;
            if(s.reader) {
                const auto q=s.reader->canonical_pair_view(i,j);const auto& c=pair.coefficients();
                if(q.rank!=r || q.coefficients.element_count!=c.size() || q.coefficients.data!=c.data())
                    throw std::invalid_argument("Gaussian particle-hole current Reader pair C must use exact warm coefficient storage");
            }
            if(r>n || pair.memory().retained_dimension!=r || (s.cc && s.cc->solver().pair_rank(i,j)!=r)
                || s.warm.solver().pair_rank(i,j)!=r || t2view.element_count!=mul(r,r)
                || pair.memory().retained_output_bytes!=add(f2.retained_numerical_bytes(),mul(8,mul(r,r)))
                || pair.allocation_identity()!=s.ref.dimensions().allocation_identity
                || pair.local_basis_identity_sha256()!=s.basis.local_basis_identity_sha256())
                throw std::invalid_argument("Gaussian particle-hole pair frame/snapshot extent differs");
            pairs=add(pairs,pair.memory().retained_output_bytes);t2=add(t2,mul(r,r));maximum_rank=std::max(maximum_rank,r);
            if(i==p.target_i || j==p.target_i || i==p.target_j || j==p.target_j)
                visited_maximum_rank=std::max(visited_maximum_rank,r);
            pair_seals=add(pair_seals,add(7*65,f2.retained_receipt_payload_bytes()));
            p.frame_validation_work_units=add(p.frame_validation_work_units,plan_periodic_gaussian_pair_pno_frame_validation(f2).work_units);
            if(p.domain_generated) {
                const auto& g=s.warm.pair_generation_geometry(i,j);
                geometry=add(geometry,g.retained_numerical_bytes());geometry_controls=add(geometry_controls,g.retained_control_storage_bytes());
                const auto view=s.warm.pair_generation_geometry_view(i,j);
                if(view.embedding().identity_sha256()!=f2.embedded().embedding_identity_sha256())
                    throw std::invalid_argument("Gaussian particle-hole retained generation geometry differs");
            }
        }
    }
    const auto& mm=s.warm.solver().memory();
    if(pairs!=s.warm.diagnostics().retained_pair_bytes
        || geometry!=s.warm.diagnostics().retained_pair_geometry_bytes
        || geometry_controls!=s.warm.diagnostics().retained_pair_geometry_control_bytes
        || pair_seals!=s.warm.memory().retained_pair_seal_bytes || maximum_rank!=s.warm.diagnostics().maximum_pair_rank
        || mm.total_amplitude_elements!=t2 || mm.output_numerical_bytes!=add(mul(8,t2),mul(16,P)))
        throw std::logic_error("Gaussian particle-hole complete source owner inventory differs");
    p.borrowed_warmstart_numerical_bytes=add(add(pairs,geometry),mm.output_numerical_bytes);
    if(s.cc) {
    const auto& cm=s.cc->solver().memory();
    if(singles!=s.cc->diagnostics().retained_singles_bytes || cm.total_singles_elements!=t1 || cm.total_doubles_elements!=t2
        || cm.output_numerical_bytes!=add(mul(8,add(t1,t2)),mul(16,add(o,P))))
        throw std::logic_error("Gaussian particle-hole CCSD owner inventory differs");
    p.borrowed_ccsd_numerical_bytes=add(singles,cm.output_numerical_bytes);
    p.borrowed_owner_control_bytes=add(add(sizeof(PeriodicGaussianPairMP2Result),sizeof(PeriodicGaussianPairCCSDResult)),
        add(add(mul(P,sizeof(PeriodicGaussianPairSpace)),mul(o,sizeof(PeriodicGaussianPairPNOFrame))),
        add(add(pair_seals,single_seals),add(geometry_controls,17*65))));
    for(const auto* extra:{&s.cc->physical_particle_hole_source_identity_sha256(),
        &s.cc->solver().split_execution_identity_sha256(),&s.cc->solver().consumed_particle_hole_identity_sha256()})
        if(!extra->empty()) {
            sha(*extra);
            p.borrowed_owner_control_bytes=add(p.borrowed_owner_control_bytes,add(extra->size(),1));
        }
    } else {
        const auto& r=s.reader->memory();
        if(r.singles_amplitude_elements!=t1 || r.pair_amplitude_elements!=t2)
            throw std::logic_error("Gaussian particle-hole Reader amplitude census differs");
        p.borrowed_owner_control_bytes=add(sizeof(PeriodicGaussianPairMP2Result),
            add(mul(P,sizeof(PeriodicGaussianPairSpace)),add(pair_seals,add(geometry_controls,8*65))));
    }
    p.borrowed_common_numerical_bytes=add(s.wannier.memory().caller_gauge_bytes,
        add(s.wannier.memory().retained_coefficient_bytes,add(s.domain.memory().retained_domain_index_bytes,
        add(s.domain.memory().retained_matrix_bytes,add(s.space.memory().output_numerical_bytes,s.basis.memory().retained_output_bytes)))));
    p.borrowed_basis_active_numeric_bytes=s.hf.context_handle()->inventory().combined_borrowed_active_numeric_bytes;
    p.frame_validation_work_units=mul(2,p.frame_validation_work_units);
    p.target_dimension=s.warm.pair(std::min(p.target_i,p.target_j),std::max(p.target_i,p.target_j)).frame().retained_dimension();
    // Every source has one occupied endpoint in the target; unrelated
    // pair ranks remain retained owners but do not inflate kernel scratch.
    p.maximum_source_dimension=visited_maximum_rank;
    if(p.maximum_source_dimension>n) throw std::logic_error("Gaussian particle-hole source rank census differs");
}
U owner_bytes(const Plan& p) {return add(p.borrowed_warmstart_numerical_bytes,p.borrowed_ccsd_numerical_bytes);}
PeriodicGaussianLocalOrbitalFactorLiveInventory child_live(const Plan& p,const Live& live) {
    // Whole owners retained; child frame/geometry roles are intentionally
    // duplicated and named in the outer plan. This channel carries logical
    // outer controls too because the child has no separate control channel.
    return {add(add(owner_bytes(p),p.accumulator_owned_bytes),add(live.other_live_numerical_bytes_per_worker,p.outer_control_storage_bytes)),
        0,live.backend_margin_bytes_per_worker};
}
BoundedRestrictedPairCCSDParticleHoleInventory kernel_inventory(const Plan& p,const Live& live,U kernel_fixed) {
    const U original_k=p.maximum_transpose_bytes;
    return {p.replicas_per_node,p.reference_base_node_bytes,
        add(owner_bytes(p),add(p.borrowed_common_numerical_bytes,add(p.borrowed_basis_active_numeric_bytes,
        add(original_k,live.other_live_numerical_bytes_per_worker)))),
        p.control_storage_reservation_bytes-kernel_fixed,live.backend_margin_bytes_per_worker};
}
BoundedRestrictedPairCCSDParticleHoleCaps kernel_caps(const BoundedRestrictedPairCCSDParticleHoleMemoryPlan& p) {
    return {p.target_dimension,p.maximum_source_dimension,p.occupied_count,p.maximum_borrowed_numerical_bytes,
        p.peak_owned_numerical_bytes,p.total_control_storage_bytes,p.per_replica_inventoried_bytes,
        p.required_node_inventoried_bytes,p.scalar_products_upper_bound,p.work_units_upper_bound};
}
void verify_payloads(const Sources& s,const Plan& p) {
    if(s.cc) verify_bounded_restricted_pair_ccsd_solver_payload(s.cc->solver(),p.snapshot_validation_work_units/2);
    else s.reader->validate_immutable_snapshot();
    for(U i=0;i<p.occupied_count;++i) {
        if(s.cc) s.cc->singles_frame(i).verify_payload();
        for(U j=i;j<p.occupied_count;++j) s.warm.pair(i,j).frame().verify_payload();
    }
}
Plan plan_stream(const Sources& s,U i,U j,const PeriodicGaussianPairInteractionConfig& config,
    const PeriodicGaussianPairInteractionOptions& options,const Live& live,const Caps& caps) {
    const auto& hf=s.hf;const auto& ref=s.ref;const auto& wannier=s.wannier;const auto& domain=s.domain;
    const auto& space=s.space;const auto& basis=s.basis;const auto& warm=s.warm;
    Plan p;basic(s,p,i,j,live,caps);
    // Fixed conservative metadata bound is admitted before both walks.
    p.metadata_work_units=mul(1048576,add(1024,add(p.source_slots,add(p.pair_count,p.occupied_count))));
    limit(p.metadata_work_units,caps.maximum_work_units,"Gaussian particle-hole metadata work cap exceeded");
    owners(s,p);const U A=p.target_dimension,B=p.maximum_source_dimension;
    const U validation_work=s.cc?plan_bounded_restricted_pair_ccsd_solver_payload_validation(s.cc->solver()).work_units:s.reader->memory().validation_work_units;
    const U validation_controls=s.cc?plan_bounded_restricted_pair_ccsd_solver_payload_validation(s.cc->solver()).fixed_control_storage_bytes:0;
    p.snapshot_validation_work_units=mul(2,validation_work);
    const auto bare=plan_bounded_restricted_pair_ccsd_particle_hole(A,p.occupied_count,B,{1,0,0,0,live.backend_margin_bytes_per_worker});
    p.accumulator_owned_bytes=bare.peak_owned_numerical_bytes;p.output_numerical_bytes=mul(8,mul(A,A));
    p.maximum_transpose_bytes=mul(8,mul(A,B));
    p.contraction_phase_bytes=add(p.accumulator_owned_bytes,mul(4,p.maximum_transpose_bytes));
    p.duplicate_t_role_padding_bytes=mul(8,mul(B,B));
    p.outer_control_storage_bytes=add(fixed_controls,add(p.borrowed_owner_control_bytes,
        add(validation_controls,add(bare.fixed_control_storage_bytes,live.other_live_control_bytes_per_worker))));
    p.control_storage_reservation_bytes=add(p.outer_control_storage_bytes,caps.maximum_interaction_control_bytes);
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes,"Gaussian particle-hole control cap exceeded");
    const auto& dims=ref.dimensions();const auto& budget=ref.budget();p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);
    if(!p.replicas_per_node) throw std::invalid_argument("Gaussian particle-hole replica census is zero");
    p.reference_base_node_bytes=add(add(dims.external_bytes,dims.shared_bytes),mul(budget.mpi_ranks,add(dims.per_rank_bytes,dims.localization_window_bytes_per_rank)));
    const auto target=warm.pair(std::min(i,j),std::max(i,j)).frame();
    std::optional<Geometry> ga;if(p.domain_generated) ga.emplace(warm.pair_generation_geometry_view(std::min(i,j),std::max(i,j)));
    const auto cl=child_live(p,live);
    // Second metadata pass: one exact child plan at a time; no plan table.
    for(U leg=0;leg<2;++leg) for(U m=0;m<p.occupied_count;++m) {
        const U l=leg?j:i,r=leg?i:j,x=std::min(l,m),y=std::max(l,m);
        const auto source=warm.pair(x,y).frame();std::optional<Geometry> gb;
        if(p.domain_generated) gb.emplace(warm.pair_generation_geometry_view(x,y));
        const auto q=plan_periodic_gaussian_pair_interaction_blocks(hf,ref,wannier,domain,space,basis,target,source,r,m,
            config,options,cl,caps.interaction,ga?&*ga:nullptr,gb?&*gb:nullptr);
        if(q.source_dimension>B || q.borrowed_local_numerical_bytes!=p.borrowed_common_numerical_bytes
            || q.borrowed_basis_active_numeric_bytes!=p.borrowed_basis_active_numeric_bytes)
            throw std::logic_error("Gaussian particle-hole nested source inventory differs");
        limit(q.control_storage_reservation_bytes,caps.maximum_interaction_control_bytes,"Gaussian particle-hole interaction control cap exceeded");
        p.maximum_interaction_control_bytes=std::max(p.maximum_interaction_control_bytes,q.control_storage_reservation_bytes);
        p.maximum_interaction_owned_bytes=std::max(p.maximum_interaction_owned_bytes,q.peak_owned_numerical_bytes);
        p.duplicate_frame_geometry_padding_bytes=std::max(p.duplicate_frame_geometry_padding_bytes,
            add(q.borrowed_frame_numerical_bytes,q.borrowed_geometry_numerical_bytes));
        p.interaction_work_units=add(p.interaction_work_units,q.work_units);
        p.factor_panels=add(p.factor_panels,q.factor_panels);p.tile_calls=add(p.tile_calls,q.tile_calls);
        p.reciprocal_candidate_evaluations=add(p.reciprocal_candidate_evaluations,q.reciprocal_candidate_evaluations_upper_bound);
        p.image_candidate_evaluations=add(p.image_candidate_evaluations,q.image_candidate_evaluations_upper_bound);
        p.transpose_work_units=add(p.transpose_work_units,mul(8,mul(A,q.source_dimension)));
    }
    p.peak_owned_numerical_bytes=std::max(add(p.accumulator_owned_bytes,p.maximum_interaction_owned_bytes),p.contraction_phase_bytes);
    const U phases=std::max(add(p.maximum_interaction_owned_bytes,p.duplicate_frame_geometry_padding_bytes),
        add(mul(4,p.maximum_transpose_bytes),p.duplicate_t_role_padding_bytes));
    p.worker_bytes=add(add(p.accumulator_owned_bytes,phases),add(owner_bytes(p),add(p.borrowed_common_numerical_bytes,
        add(p.borrowed_basis_active_numeric_bytes,add(p.control_storage_reservation_bytes,
        add(live.other_live_numerical_bytes_per_worker,live.backend_margin_bytes_per_worker))))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.worker_bytes));
    p.accumulator=plan_bounded_restricted_pair_ccsd_particle_hole(A,p.occupied_count,B,kernel_inventory(p,live,bare.fixed_control_storage_bytes));
    if(p.accumulator.per_replica_inventoried_bytes>p.worker_bytes || p.accumulator.required_node_inventoried_bytes>p.required_node_memory_bytes)
        throw std::logic_error("Gaussian particle-hole accumulator exceeds enclosing inventory");
    p.work_units=add(p.metadata_work_units,add(p.snapshot_validation_work_units,add(p.frame_validation_work_units,
        add(p.interaction_work_units,add(p.transpose_work_units,p.accumulator.work_units_upper_bound)))));
    for(U bytes:{p.peak_owned_numerical_bytes,p.worker_bytes,p.required_node_memory_bytes,p.control_storage_reservation_bytes}) real::extent(bytes);
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian particle-hole owned cap exceeded");
    limit(p.worker_bytes,caps.maximum_worker_bytes,"Gaussian particle-hole worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_bytes,"Gaussian particle-hole node cap exceeded");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian particle-hole reference node cap exceeded");
    limit(p.work_units,caps.maximum_work_units,"Gaussian particle-hole work cap exceeded");
    limit(p.factor_panels,caps.maximum_factor_panels,"Gaussian particle-hole factor panel cap exceeded");
    limit(p.tile_calls,caps.maximum_tile_calls,"Gaussian particle-hole tile cap exceeded");
    limit(p.reciprocal_candidate_evaluations,caps.maximum_reciprocal_candidates,"Gaussian particle-hole reciprocal cap exceeded");
    limit(p.image_candidate_evaluations,caps.maximum_image_candidates,"Gaussian particle-hole image cap exceeded");
    return p;
}

struct StreamConsumption {
    BoundedRestrictedPairCCSDParticleHoleResult residual;
    PeriodicGaussianPairParticleHoleDiagnostics diagnostics;
    std::string receipt;
};
StreamConsumption consume_stream(const Sources& s,const BasisSet& ao,const BasisSet& auxiliary,
    const std::complex<double>* gauges,std::size_t gauge_count,const Plan& p,U i,U j,
    const PeriodicGaussianPairInteractionConfig& config,const PeriodicGaussianPairInteractionOptions& options,
    const Live& live,const Caps& caps) {
    const auto& hf=s.hf;const auto& ref=s.ref;const auto& wannier=s.wannier;const auto& domain=s.domain;
    const auto& space=s.space;const auto& basis=s.basis;const auto& warm=s.warm;
    const auto target=warm.pair(std::min(i,j),std::max(i,j)).frame();
    std::optional<Geometry> ga;if(p.domain_generated) ga.emplace(warm.pair_generation_geometry_view(std::min(i,j),std::max(i,j)));
    const auto inventory=kernel_inventory(p,live,p.accumulator.fixed_control_storage_bytes);
    Kernel accumulator(p.target_dimension,p.occupied_count,i,j,p.maximum_source_dimension,inventory,kernel_caps(p.accumulator));
    Digest consumed("vibeqc.periodic.gaussian-pair-particle-hole.interactions");consumed.u64(p.source_slots);
    PeriodicGaussianPairParticleHoleDiagnostics d;const auto cl=child_live(p,live);
    for(U leg=0;leg<2;++leg) for(U m=0;m<p.occupied_count;++m) {
        const U l=leg?j:i,r=leg?i:j,x=std::min(l,m),y=std::max(l,m);
        const auto source=warm.pair(x,y).frame();const U B=source.retained_dimension(),A=p.target_dimension;
        std::optional<Geometry> gb;if(p.domain_generated) gb.emplace(warm.pair_generation_geometry_view(x,y));
        auto blocks=build_periodic_gaussian_pair_interaction_blocks(hf,ref,ao,auxiliary,wannier,gauges,gauge_count,
            domain,space,basis,target,source,r,m,config,options,cl,caps.interaction,ga?&*ga:nullptr,gb?&*gb:nullptr);
        const auto& bp=blocks.memory();const auto& bd=blocks.diagnostics();
        if(bp.peak_owned_numerical_bytes>p.maximum_interaction_owned_bytes
            || bp.control_storage_reservation_bytes>p.maximum_interaction_control_bytes
            || bp.per_worker_inventoried_bytes>p.worker_bytes || bp.required_node_memory_bytes>p.required_node_memory_bytes
            || blocks.state_handle()!=ref.state_handle() || blocks.context_handle()!=hf.context_handle()
            || blocks.target_frame_identity_sha256()!=target.identity_sha256() || blocks.source_frame_identity_sha256()!=source.identity_sha256())
            throw std::logic_error("Gaussian particle-hole interaction differs from admitted source");
        std::vector<double> kt(mul(A,B));
        for(U a=0;a<A;++a) for(U b=0;b<B;++b) kt[b*A+a]=real::finite(blocks.k_ab_data()[a*B+b]);
        const auto t=s.cc?s.cc->solver().stored_pair_view(x,y):s.reader->canonical_pair_view(x,y).amplitudes;
        accumulator.accumulate({leg,m,B,{kt.data(),kt.size()},
            {blocks.j_ba_data(),static_cast<std::size_t>(mul(A,B))},
            {blocks.overlap_ba_data(),static_cast<std::size_t>(mul(A,B))},t,l>m});
        consumed.u64(leg);consumed.u64(m);consumed.u64(x);consumed.u64(y);consumed.u64(l>m);
        consumed.string(source.identity_sha256());consumed.string(blocks.identity_sha256());
        consumed.string(blocks.payload_sha256());consumed.string(blocks.consumed_sources_identity_sha256());
        ++d.completed_interactions;++d.completed_source_slots;
        d.completed_factor_panels=add(d.completed_factor_panels,bd.completed_factor_panels);
        d.completed_tile_calls=add(d.completed_tile_calls,bd.completed_tile_calls);
        d.reciprocal_candidate_evaluations=add(d.reciprocal_candidate_evaluations,bd.reciprocal_candidate_evaluations);
        d.image_candidate_evaluations=add(d.image_candidate_evaluations,bd.image_candidate_evaluations);
        d.maximum_integral_roundoff_error=std::max(d.maximum_integral_roundoff_error,bd.maximum_integral_roundoff_error);
        d.maximum_overlap_imaginary_norm=std::max(d.maximum_overlap_imaginary_norm,bd.overlap_imaginary_frobenius_upper_bound);
    }
    verify_payloads(s,p);auto residual=accumulator.finish();d.accumulator=residual.diagnostics();
    if(d.completed_source_slots!=p.source_slots || d.completed_factor_panels!=p.factor_panels || d.completed_tile_calls!=p.tile_calls
        || d.reciprocal_candidate_evaluations>p.reciprocal_candidate_evaluations || d.image_candidate_evaluations>p.image_candidate_evaluations)
        throw std::logic_error("Gaussian particle-hole completed work census differs");
    d.charged_work_units_upper_bound=p.work_units;return {std::move(residual),d,consumed.finish()};
}

} // namespace

Plan plan_periodic_gaussian_pair_particle_hole(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairMP2Result& warm,
    const PeriodicGaussianPairCCSDResult& cc,U i,U j,const PeriodicGaussianPairInteractionConfig& config,
    const PeriodicGaussianPairInteractionOptions& options,const Live& live,const Caps& caps) {
    const Sources s{hf,ref,wannier,domain,space,basis,warm,&cc};return plan_stream(s,i,j,config,options,live,caps);
}


Result build_periodic_gaussian_pair_particle_hole(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const BasisSet& ao,const BasisSet& auxiliary,
    const PeriodicCorrelationWannier& wannier,const std::complex<double>* gauges,std::size_t gauge_count,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairMP2Result& warm,
    const PeriodicGaussianPairCCSDResult& cc,U i,U j,const PeriodicGaussianPairInteractionConfig& supplied_config,
    const PeriodicGaussianPairInteractionOptions& supplied_options,const Live& supplied_live,const Caps& supplied_caps) {
    const auto config=supplied_config;const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_pair_particle_hole(hf,ref,wannier,domain,space,basis,warm,cc,i,j,config,options,live,caps);
    const Sources sources{hf,ref,wannier,domain,space,basis,warm,&cc};verify_payloads(sources,p);
    Result result;result.memory_=p;result.state_=ref.state_handle();result.context_=hf.context_handle();
    result.warm_=warm.identity_sha256();result.ccsd_=cc.identity_sha256();result.amplitudes_=cc.solver().payload_sha256();
    result.provider_=warm.provider_identity_sha256();result.hf_=hf.reference_source_identity_sha256();result.pairs_=warm.pair_spaces_identity_sha256();
    const auto target=warm.pair(std::min(i,j),std::max(i,j)).frame();result.target_=target.identity_sha256();
    auto stream=consume_stream(sources,ao,auxiliary,gauges,gauge_count,p,i,j,config,options,live,caps);
    result.residual_.emplace(std::move(stream.residual));result.diagnostics_=stream.diagnostics;
    result.consumed_=std::move(stream.receipt);
    Digest identity("vibeqc.periodic.gaussian-pair-particle-hole.identity");
    for(const auto* x:std::initializer_list<const std::string*>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,&basis.identity_sha256(),
        &result.hf_,&result.warm_,&result.ccsd_,&result.amplitudes_,&result.provider_,&result.pairs_,&result.target_,&result.consumed_,
        &result.residual_->identity_sha256()}) identity.string(*x);
    identity.u64(i);identity.u64(j);identity.u64(p.domain_generated);identity.u64(cc.solver().final_snapshot().iteration);
    identity.string("frozen-converged-CCSD-snapshot;all-two-occupied-legs;actual-finite-Gaussian-KJO;"
        "new-real-projection-not-origin-provider-bitwise-proof;bare-particle-hole-only;not-full-residual-or-energy");
    result.identity_=identity.finish();return result;
}
void Result::require_live() const {
    if(!state_ || !context_ || !residual_ || identity_.size()!=64) throw std::logic_error("Gaussian particle-hole result is moved or incomplete");
    (void)residual_->residual_data();
}
const Plan& Result::memory() const {require_live();return memory_;}
const PeriodicGaussianPairParticleHoleDiagnostics& Result::diagnostics() const {require_live();return diagnostics_;}
const double* Result::residual_data() const {require_live();return residual_->residual_data();}
const std::string& Result::payload_sha256() const {require_live();return residual_->payload_identity_sha256();}
#define GPH_GET(name,field) const std::string& Result::name() const {require_live();return field;}
GPH_GET(identity_sha256,identity_) GPH_GET(consumed_interactions_identity_sha256,consumed_)
GPH_GET(ccsd_identity_sha256,ccsd_) GPH_GET(ccsd_amplitude_payload_sha256,amplitudes_)
GPH_GET(warmstart_identity_sha256,warm_) GPH_GET(origin_provider_identity_sha256,provider_)
GPH_GET(hf_reference_source_identity_sha256,hf_) GPH_GET(pair_spaces_identity_sha256,pairs_)
GPH_GET(target_frame_identity_sha256,target_)
#undef GPH_GET
const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& Result::state_handle() const {require_live();return state_;}
const std::shared_ptr<const PeriodicGaussianSourceContext>& Result::context_handle() const {require_live();return context_;}

namespace {
using SnapshotPlan=PeriodicGaussianPairParticleHoleSnapshotPlan;
using SnapshotResult=PeriodicGaussianPairParticleHoleSnapshotResult;
Live snapshot_live(const BoundedRestrictedPairCCSDAmplitudes& reader,const Live& supplied) {
    auto live=supplied;const auto& p=reader.memory();
    live.other_live_numerical_bytes_per_worker=add(live.other_live_numerical_bytes_per_worker,p.borrowed_numerical_bytes);
    live.other_live_control_bytes_per_worker=add(live.other_live_control_bytes_per_worker,
        add(p.borrowed_table_bytes,add(p.fixed_control_storage_bytes,
        2*sizeof(SnapshotPlan)+sizeof(SnapshotResult)+16*65)));
    return live;
}
}
SnapshotPlan plan_periodic_gaussian_pair_particle_hole_snapshot(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairMP2Result& warm,
    const BoundedRestrictedPairCCSDAmplitudes& reader,U i,U j,const PeriodicGaussianPairInteractionConfig& config,
    const PeriodicGaussianPairInteractionOptions& options,const Live& live,const Caps& caps) {
    SnapshotPlan p;const auto& r=reader.memory();
    p.borrowed_reader_numerical_bytes=r.borrowed_numerical_bytes;p.borrowed_reader_table_bytes=r.borrowed_table_bytes;
    p.reader_control_storage_bytes=r.fixed_control_storage_bytes;
    p.reader_warm_coefficient_padding_bytes=mul(8,r.pair_coefficient_elements);
    const Sources sources{hf,ref,wannier,domain,space,basis,warm,nullptr,&reader};
    p.stream=plan_stream(sources,i,j,config,options,snapshot_live(reader,live),caps);return p;
}
SnapshotResult build_periodic_gaussian_pair_particle_hole_snapshot(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const BasisSet& ao,const BasisSet& auxiliary,
    const PeriodicCorrelationWannier& wannier,const std::complex<double>* gauges,std::size_t gauge_count,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairMP2Result& warm,
    const BoundedRestrictedPairCCSDAmplitudes& reader,U i,U j,U snapshot_index,
    const PeriodicGaussianPairInteractionConfig& supplied_config,const PeriodicGaussianPairInteractionOptions& supplied_options,
    const Live& supplied_live,const Caps& supplied_caps) {
    const auto config=supplied_config;const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_pair_particle_hole_snapshot(hf,ref,wannier,domain,space,basis,warm,reader,i,j,config,options,live,caps);
    const Sources sources{hf,ref,wannier,domain,space,basis,warm,nullptr,&reader};verify_payloads(sources,p.stream);
    SnapshotResult result;result.memory_=p;result.state_=ref.state_handle();result.context_=hf.context_handle();result.index_=snapshot_index;
    result.snapshot_=reader.snapshot_identity_sha256();result.warm_=warm.identity_sha256();
    result.provider_=warm.provider_identity_sha256();result.hf_=hf.reference_source_identity_sha256();result.pairs_=warm.pair_spaces_identity_sha256();
    auto stream=consume_stream(sources,ao,auxiliary,gauges,gauge_count,p.stream,i,j,config,options,snapshot_live(reader,live),caps);
    result.residual_.emplace(std::move(stream.residual));result.diagnostics_=stream.diagnostics;result.consumed_=std::move(stream.receipt);
    if(reader.snapshot_identity_sha256()!=result.snapshot_) throw std::invalid_argument("Gaussian particle-hole current Reader snapshot changed");
    Digest identity("vibeqc.periodic.gaussian-pair-particle-hole.current-snapshot.identity-v1");
    for(const auto* s:std::initializer_list<const std::string*>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &basis.identity_sha256(),&result.hf_,&result.warm_,&result.provider_,&result.pairs_,&result.snapshot_,&result.consumed_,
        &result.residual_->identity_sha256()}) identity.string(*s);
    identity.u64(i);identity.u64(j);identity.u64(snapshot_index);identity.u64(p.stream.domain_generated);
    identity.string("immutable-numerical-current-reader;exact-warm-pair-C-storage;singles-numerical-only;"
        "snapshot-index-is-sequencing-not-provenance;all-two-occupied-legs;"
        "new-real-projection-not-origin-provider-bitwise-proof;bare-only;no-convergence-or-energy-certificate");
    result.identity_=identity.finish();return result;
}
void SnapshotResult::require_live() const {
    if(!state_ || !context_ || !residual_ || identity_.size()!=64 || snapshot_.size()!=64)
        throw std::logic_error("Gaussian particle-hole snapshot result is moved or incomplete");
    (void)residual_->residual_data();
}
const SnapshotPlan& SnapshotResult::memory() const {require_live();return memory_;}
const PeriodicGaussianPairParticleHoleDiagnostics& SnapshotResult::diagnostics() const {require_live();return diagnostics_;}
const double* SnapshotResult::residual_data() const {require_live();return residual_->residual_data();}
const BoundedRestrictedPairCCSDParticleHoleResult& SnapshotResult::bare_result() const & {require_live();return *residual_;}
U SnapshotResult::snapshot_index() const {require_live();return index_;}
const std::string& SnapshotResult::payload_sha256() const {require_live();return residual_->payload_identity_sha256();}
#define GPH_SGET(name,field) const std::string& SnapshotResult::name() const {require_live();return field;}
GPH_SGET(snapshot_identity_sha256,snapshot_) GPH_SGET(identity_sha256,identity_)
GPH_SGET(consumed_interactions_identity_sha256,consumed_) GPH_SGET(warmstart_identity_sha256,warm_)
GPH_SGET(origin_provider_identity_sha256,provider_) GPH_SGET(hf_reference_source_identity_sha256,hf_)
GPH_SGET(pair_spaces_identity_sha256,pairs_)
#undef GPH_SGET
const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& SnapshotResult::state_handle() const {require_live();return state_;}
const std::shared_ptr<const PeriodicGaussianSourceContext>& SnapshotResult::context_handle() const {require_live();return context_;}
} // namespace vibeqc
