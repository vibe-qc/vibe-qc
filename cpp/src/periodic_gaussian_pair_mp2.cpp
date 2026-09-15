#include "vibeqc/periodic_gaussian_pair_mp2.hpp"
#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
using I=std::uint64_t;
using real::add;
using real::mul;
using real::Digest;
using Options=PeriodicGaussianPairMP2Options;
using Live=PeriodicGaussianPairMP2LiveInventory;
using Caps=PeriodicGaussianPairMP2Caps;
using Plan=PeriodicGaussianPairMP2Plan;
using Result=PeriodicGaussianPairMP2Result;
using SolverInventory=BoundedRestrictedPairMP2SolverInventory;
using SolverPlan=BoundedRestrictedPairMP2SolverMemoryPlan;
using PairView=BoundedRestrictedPairMP2PairView;
static_assert(sizeof(double)==8 && FLT_EVAL_METHOD==0,"Gaussian pair MP2 requires binary64 evaluation");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian pair MP2 forbids fast/finite-only math"
#endif
void limit(I value,I cap,const char* message) {
    if(!cap || value>cap) throw std::length_error(message);
}
I subtract(I a,I b) {
    if(b>a) throw std::logic_error("Gaussian pair MP2 ownership subtraction is inconsistent");
    return a-b;
}
void scientific(const Options& o) {
    for(double x:{o.maximum_occupied_virtual_fock_norm,o.projection.maximum_diagonal_symmetry_projection_error,
        o.solver.maximum_diagonal_update_antisymmetry_norm})
        if(!std::isfinite(x) || x<0)
            throw std::invalid_argument("Gaussian pair MP2 norm budgets must be explicit finite nonnegative");
    for(double x:{o.solver.denominator_floor,o.solver.residual_tolerance,o.solver.energy_tolerance,
        o.solver.coefficient_orthogonality_tolerance})
        if(!std::isfinite(x) || x<=0)
            throw std::invalid_argument("Gaussian pair MP2 solver controls must be explicit positive finite");
    if(!o.solver.maximum_iterations || o.solver.coefficient_orthogonality_tolerance>=1
        || o.pnos.denominator_floor!=o.solver.denominator_floor)
        throw std::invalid_argument("Gaussian pair MP2 requires consistent denominator floors and valid solver controls");
}
void solver_caps(const SolverPlan& p,const BoundedRestrictedPairMP2SolverCaps& c) {
    limit(p.n_occupied,c.maximum_occupied_count,"Gaussian pair MP2 solver occupied cap exceeded");
    limit(p.common_virtual_dimension,c.maximum_common_virtual_dimension,"Gaussian pair MP2 solver common dimension cap exceeded");
    limit(p.pair_count,c.maximum_pair_count,"Gaussian pair MP2 solver pair count cap exceeded");
    limit(p.maximum_pair_rank,c.maximum_pair_rank,"Gaussian pair MP2 solver rank cap exceeded");
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"Gaussian pair MP2 solver owned cap exceeded");
    limit(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"Gaussian pair MP2 solver worker cap exceeded");
    limit(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"Gaussian pair MP2 solver node cap exceeded");
    limit(p.coupling_slots_upper_bound,c.maximum_coupling_slots,"Gaussian pair MP2 solver coupling cap exceeded");
    limit(p.work_units_upper_bound,c.maximum_work_units,"Gaussian pair MP2 solver work cap exceeded");
}
SolverInventory solver_inventory(const Plan& p,const Live& live,I fixed,I table) {
    SolverInventory inventory;
    inventory.numerical_replicas=p.replicas_per_node;
    inventory.external_node_bytes=p.reference_base_node_bytes;
    inventory.fixed_backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
    // The generic solver counts Foo, C/G/eps, its table and its own control.
    // Keep original PNO occupations, the rest of basis F/labels, provider
    // rows and outer controls exactly once. Rank truncation changes C/G/eps,
    // not the common-dimension original occupation arrays.
    inventory.other_live_bytes_per_replica=add(live.other_live_bytes_per_worker,
        add(p.borrowed_provider_row_bytes,add(subtract(p.borrowed_basis_bytes,mul(8,mul(p.occupied_count,p.occupied_count))),
        add(mul(8,mul(p.pair_count,p.common_virtual_dimension)),
        subtract(p.control_storage_reservation_bytes,add(fixed,table))))));
    return inventory;
}
void options_wire(Digest& h,const Options& o) {
    for(double x:{o.pnos.occupation_cutoff,o.pnos.denominator_floor,
        o.pnos.maximum_initial_fvv_offdiagonal_norm,o.pnos.semicanonical_orthonormality_tolerance,
        o.projection.maximum_diagonal_symmetry_projection_error,o.maximum_occupied_virtual_fock_norm,
        o.solver.denominator_floor,o.solver.residual_tolerance,o.solver.energy_tolerance,
        o.solver.coefficient_orthogonality_tolerance,o.solver.maximum_diagonal_update_antisymmetry_norm}) h.real(x);
    for(const auto& e:{o.pnos.pno_eigensolver,o.pnos.semicanonical_eigensolver}) {
        h.u64(e.max_sweeps); h.real(e.relative_offdiagonal_tolerance);
    }
    h.u64(o.solver.maximum_iterations);
}
struct Events {
    PeriodicGaussianPairMP2Progress event;
    PeriodicGaussianPairMP2Callback callback=nullptr;
    void* context=nullptr;
    I maximum=0;
    void emit(PeriodicGaussianPairMP2Stage stage) {
        if(event.callback_count>=maximum) throw std::length_error("Gaussian pair MP2 progress cap exceeded");
        event.stage=stage; ++event.callback_count;
        if(callback) callback(event,context);
    }
    static void solver(const BoundedRestrictedPairMP2SolverProgress& progress,void* context) {
        auto& self=*static_cast<Events*>(context); self.event.solver=progress;
        self.emit(PeriodicGaussianPairMP2Stage::Solver);
    }
};
} // namespace

Plan plan_periodic_gaussian_pair_mp2(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const Options& options,const Live& live,const Caps& caps) {
    scientific(options); real::float_environment();
    if(!live.fixed_backend_margin_bytes_per_worker)
        throw std::invalid_argument("Gaussian pair MP2 backend margin must be explicit and positive");
    // This count-only leaf plan also authenticates exact state/source/basis
    // owners and scientific PNO controls, without scanning F or factor rows.
    PeriodicGaussianPairPNOLiveInventory initial_live{0,live.fixed_backend_margin_bytes_per_worker};
    const auto pno=plan_periodic_gaussian_pair_pnos(ref,basis,provider,0,0,options.pnos,initial_live,caps.pnos);
    Plan p; p.occupied_count=pno.occupied_count; p.common_virtual_dimension=pno.virtual_count;
    const auto o=p.occupied_count,n=p.common_virtual_dimension;
    p.pair_count=mul(o,add(o,1))/2;
    limit(p.pair_count,caps.maximum_pair_count,"Gaussian pair MP2 outer pair count cap exceeded");
    const auto space=periodic_gaussian_pair_space_storage(n,n);
    p.borrowed_basis_bytes=pno.borrowed_basis_bytes; p.borrowed_provider_row_bytes=pno.borrowed_provider_row_bytes;
    p.replicas_per_node=pno.replicas_per_node; p.reference_base_node_bytes=pno.reference_base_node_bytes;
    SolverInventory initial_inventory{p.replicas_per_node,p.reference_base_node_bytes,0,
        live.fixed_backend_margin_bytes_per_worker};
    const auto solver=plan_bounded_restricted_pair_mp2_solver_upper(o,n,n,options.solver.maximum_iterations,initial_inventory);
    // Seven space and eight PNO strings are sealed 64-character digests
    // (allocation identity is validated as lowercase SHA-256 too). Include
    // their active payload and terminators per retained pair, not merely
    // sizeof(string) handles. Allocator capacity is a separate allowance.
    p.retained_pair_seal_bytes=mul(p.pair_count,15*65);
    p.control_storage_reservation_bytes=add(65536+sizeof(Plan)+sizeof(Result)+sizeof(Options)+sizeof(Live)+sizeof(Caps)
        +sizeof(Events)+sizeof(PeriodicCorrelationAdmittedReference),
        add(add(mul(p.pair_count,sizeof(PeriodicGaussianPairSpace)),p.retained_pair_seal_bytes),
        add(solver.borrowed_pair_table_bytes,add(pno.fixed_control_storage_bytes,
        add(space.fixed_control_storage_bytes,solver.fixed_control_storage_bytes)))));
    p.retained_pair_output_upper_bytes=mul(p.pair_count,space.retained_output_bytes);
    const auto previous=mul(p.pair_count-1,space.retained_output_bytes);
    p.pair_generation_phase_upper_bytes=add(previous,std::max(pno.peak_owned_numerical_bytes,space.construction_live_numerical_bytes));
    p.solver_phase_owned_upper_bytes=add(p.retained_pair_output_upper_bytes,solver.peak_owned_numerical_bytes);
    p.peak_owned_numerical_bytes=std::max(p.pair_generation_phase_upper_bytes,p.solver_phase_owned_upper_bytes);
    const auto borrowed=add(p.borrowed_basis_bytes,p.borrowed_provider_row_bytes);
    const auto extras=add(live.other_live_bytes_per_worker,
        add(p.control_storage_reservation_bytes,live.fixed_backend_margin_bytes_per_worker));
    p.per_worker_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(borrowed,extras));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    p.integral_calls_upper_bound=mul(p.pair_count,add(pno.integral_calls,space.integral_calls));
    p.progress_callback_upper_bound=add(add(p.pair_count,options.solver.maximum_iterations),2);
    p.driver_work_units=mul(4096,add(add(p.pair_count,add(o,n)),1));
    p.work_units_upper_bound=add(p.driver_work_units,add(solver.work_units_upper_bound,
        mul(p.pair_count,add(pno.work_units,add(space.numerical_work_units,
        mul(space.integral_calls,provider.memory().scalar_work_units))))));
    for(I bytes:{p.peak_owned_numerical_bytes,p.control_storage_reservation_bytes,p.retained_pair_output_upper_bytes}) real::extent(bytes);
    if(p.pair_count>std::vector<PeriodicGaussianPairSpace>().max_size()
        || p.pair_count>std::vector<PairView>().max_size())
        throw std::length_error("Gaussian pair MP2 pair table exceeds native extent");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian pair MP2 outer owned cap exceeded");
    limit(p.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"Gaussian pair MP2 outer worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"Gaussian pair MP2 outer node cap exceeded");
    limit(p.required_node_memory_bytes,ref.budget().memory_limit_bytes,"Gaussian pair MP2 admitted reference node cap exceeded");
    limit(p.integral_calls_upper_bound,caps.maximum_integral_calls,"Gaussian pair MP2 outer integral cap exceeded");
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gaussian pair MP2 outer progress cap exceeded");
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian pair MP2 outer work cap exceeded");
    // Admit worst-case simultaneous owners against EVERY inner cap before
    // allocating even the pair table. No invented numerical PNO owner.
    initial_live.other_live_bytes_per_worker=add(previous,add(live.other_live_bytes_per_worker,
        subtract(p.control_storage_reservation_bytes,pno.fixed_control_storage_bytes)));
    (void) plan_periodic_gaussian_pair_pnos(ref,basis,provider,0,0,options.pnos,initial_live,caps.pnos);
    const auto space_worker=add(previous,add(space.construction_live_numerical_bytes,add(borrowed,extras)));
    limit(space.peak_owned_numerical_bytes,caps.projection.maximum_owned_numerical_bytes,"Gaussian pair MP2 projection owned cap exceeded");
    limit(space_worker,caps.projection.maximum_per_worker_inventoried_bytes,"Gaussian pair MP2 projection worker cap exceeded");
    limit(add(p.reference_base_node_bytes,mul(p.replicas_per_node,space_worker)),
        caps.projection.maximum_node_inventoried_bytes,"Gaussian pair MP2 projection node cap exceeded");
    limit(space.integral_calls,caps.projection.maximum_integral_calls,"Gaussian pair MP2 projection integral cap exceeded");
    limit(add(space.numerical_work_units,mul(space.integral_calls,provider.memory().scalar_work_units)),
        caps.projection.maximum_work_units,"Gaussian pair MP2 projection work cap exceeded");
    const auto inventory=solver_inventory(p,live,solver.fixed_control_storage_bytes,solver.borrowed_pair_table_bytes);
    p.solver_upper=plan_bounded_restricted_pair_mp2_solver_upper(o,n,n,options.solver.maximum_iterations,inventory);
    solver_caps(p.solver_upper,caps.solver);
    if(p.solver_upper.per_replica_inventoried_bytes!=add(p.solver_phase_owned_upper_bytes,add(borrowed,extras)))
        throw std::logic_error("Gaussian pair MP2 enclosing/solver inventory identity changed");
    return p;
}

const BoundedRestrictedPairMP2SolverResult& Result::solver() const {
    if(!state_ || !context_ || !solver_ || pairs_.size()!=memory_.pair_count
        || pair_geometry_.size()!=(domain_generated()?memory_.pair_count:0))
        throw std::logic_error("Gaussian pair MP2 result is moved or incomplete");
    return *solver_;
}
const std::string& Result::provider_identity_sha256() const {
    if(direct_gram_source()) throw std::logic_error("DirectPAOGram pair MP2 has no common factor provider receipt");
    return provider_;
}
const PeriodicGaussianPairSpace& Result::pair(I i,I j) const {
    (void) solver(); const auto o=memory_.occupied_count;
    if(i>=o || j>=o) throw std::out_of_range("Gaussian pair MP2 pair index out of range");
    if(i>j) std::swap(i,j);
    const auto index=mul(i,2*o-i+1)/2+j-i;
    return pairs_.at(static_cast<std::size_t>(index));
}
const PeriodicCorrelationRealPAOEmbedding& Result::diagonal_generation_embedding(I i) const {
    return pair_generation_geometry(i,i).embedding();
}
const PeriodicGaussianPairDomainGeometry& Result::pair_generation_geometry(I i,I j) const {
    (void)solver();const I o=memory_.occupied_count;
    if(!domain_generated()) throw std::logic_error("Gaussian pair MP2 has no retained generation geometry");
    if(i>j || j>=o) throw std::out_of_range("Gaussian pair MP2 generation geometry requires canonical occupied slots");
    const I index=mul(i,add(subtract(mul(2,o),i),1))/2+j-i;
    const auto& g=pair_geometry_.at(static_cast<std::size_t>(index));
    const auto f=pairs_.at(static_cast<std::size_t>(index)).frame();
    if(g.state_handle()!=state_ || g.context_handle()!=context_ || g.builder_identity_sha256()!=builder_
        || g.hf_reference_source_identity_sha256()!=hf_ || g.occupied_slot_i()!=i || g.occupied_slot_j()!=j
        || g.basis_identity_sha256()!=f.basis_identity_sha256()
        || g.allocation_identity()!=pairs_.at(static_cast<std::size_t>(index)).allocation_identity()
        || (direct_gram_source() ? (!f.is_direct_gram()
            || g.embedding().identity_sha256()!=f.direct_gram().embedding_identity_sha256())
            : (!f.is_embedded() || g.embedding().identity_sha256()!=f.embedded().embedding_identity_sha256()))
        || g.embedding().memory().pair_dimension!=f.generation_dimension())
        throw std::invalid_argument("Gaussian pair MP2 retained generation source differs from its PNO frame");
    return g;
}
PeriodicGaussianPairPNOGeometryView Result::pair_generation_geometry_view(I i,I j) const & {
    return pair_generation_geometry(i,j).geometry_view();
}
bool Result::converged() const { return solver().final_snapshot().converged; }
bool Result::periodic_energy_per_cell() const {
    return diagnostics_.complete_common_finite_torus_basis && converged();
}
double Result::correlation_energy_per_cell() const {
    if(!periodic_energy_per_cell()) throw std::logic_error("Gaussian pair MP2 per-cell energy requires a complete common basis and converged pair solve");
    return real::finite(solver().final_snapshot().correlation_energy/static_cast<double>(state_->n_kpoints()));
}
double Result::total_energy_per_cell() const {
    const double correlation=correlation_energy_per_cell();
    return real::finite(state_->reference_energy_per_cell()+correlation);
}

Result run_periodic_gaussian_pair_mp2(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const Options& input_options,const Live& input_live,const Caps& input_caps,
    PeriodicGaussianPairMP2Callback callback,void* callback_context) {
    const auto options=input_options; const auto live=input_live; const auto caps=input_caps;
    const auto p=plan_periodic_gaussian_pair_mp2(ref,basis,provider,options,live,caps);
    Result result; result.memory_=p; result.state_=ref.state_handle(); result.context_=provider.context_handle();
    result.provider_=provider.identity_sha256(); result.hf_=provider.hf_reference_source_identity_sha256();
    auto& d=result.diagnostics_; const auto o=p.occupied_count,n=p.common_virtual_dimension;
    double squared=0;
    for(I at=0;at<mul(o,n);++at) real::norm_lane(squared,std::abs(real::finite(basis.f_ov_data()[at])));
    d.occupied_virtual_fock_norm_upper_bound=real::sqrt_up(squared);
    if(d.occupied_virtual_fock_norm_upper_bound>options.maximum_occupied_virtual_fock_norm)
        throw std::invalid_argument("Gaussian pair MP2 occupied-virtual Fock norm exceeds explicit Brillouin budget");
    d.minimum_pair_rank=n; d.all_pairs_full_rank=true;
    d.generation_dimension_sum=mul(p.pair_count,n);
    d.complete_common_finite_torus_basis=o==mul(ref.state().n_kpoints(),ref.state().n_correlated_occupied())
        && n==mul(ref.state().n_kpoints(),ref.state().n_virtual());
    Events events{{},callback,callback_context,p.progress_callback_upper_bound};
    events.emit(PeriodicGaussianPairMP2Stage::Begin);
    result.pairs_.reserve(static_cast<std::size_t>(p.pair_count));
    Digest pairs("vibeqc.periodic.gaussian-pair-mp2.spaces"); pairs.u64(p.pair_count);
    const auto pno_controls=plan_periodic_gaussian_pair_pnos(ref,basis,provider,0,0,options.pnos,
        {0,live.fixed_backend_margin_bytes_per_worker},caps.pnos).fixed_control_storage_bytes;
    const auto space_controls=periodic_gaussian_pair_space_storage(n,n).fixed_control_storage_bytes;
    for(I i=0;i<o;++i) for(I j=i;j<o;++j) {
        const auto prior=d.retained_pair_bytes;
        PeriodicGaussianPairPNOLiveInventory pno_live;
        pno_live.fixed_backend_margin_bytes_per_worker=live.fixed_backend_margin_bytes_per_worker;
        pno_live.other_live_bytes_per_worker=add(prior,add(live.other_live_bytes_per_worker,
            subtract(p.control_storage_reservation_bytes,pno_controls)));
        auto pno=make_periodic_gaussian_pair_pnos(ref,basis,provider,i,j,options.pnos,pno_live,caps.pnos);
        d.completed_integral_calls=add(d.completed_integral_calls,pno.diagnostics().completed_integral_calls);
        PeriodicGaussianPairSpaceLiveInventory space_live;
        space_live.fixed_backend_margin_bytes_per_worker=live.fixed_backend_margin_bytes_per_worker;
        space_live.other_live_bytes_per_worker=add(prior,add(live.other_live_bytes_per_worker,
            subtract(p.control_storage_reservation_bytes,space_controls)));
        auto space=make_periodic_gaussian_pair_space(ref,basis,provider,std::move(pno),options.projection,space_live,caps.projection);
        d.completed_integral_calls=add(d.completed_integral_calls,space.diagnostics().completed_integral_calls);
        d.retained_pair_bytes=add(d.retained_pair_bytes,space.memory().retained_output_bytes);
        const auto rank=space.memory().retained_dimension;
        d.minimum_pair_rank=std::min(d.minimum_pair_rank,rank); d.maximum_pair_rank=std::max(d.maximum_pair_rank,rank);
        if(!rank) ++d.zero_rank_pairs;
        d.all_pairs_full_rank=d.all_pairs_full_rank && rank==n;
        d.maximum_diagonal_integral_projection_norm=std::max(d.maximum_diagonal_integral_projection_norm,
            space.diagnostics().diagonal_symmetry_projection_frobenius_bound);
        if(d.retained_pair_bytes>p.retained_pair_output_upper_bytes || d.completed_integral_calls>p.integral_calls_upper_bound)
            throw std::logic_error("Gaussian pair MP2 retained pair payload exceeds admission");
        pairs.u64(i); pairs.u64(j); pairs.string(space.identity_sha256());
        result.pairs_.push_back(std::move(space)); ++d.completed_pairs;
        events.event.completed_pairs=d.completed_pairs; events.event.occupied_i=i; events.event.occupied_j=j;
        events.event.pair_rank=rank; events.emit(PeriodicGaussianPairMP2Stage::PairComplete);
    }
    result.pairs_sha_=pairs.finish();
    std::vector<PairView> views; views.reserve(static_cast<std::size_t>(p.pair_count));
    for(const auto& space:result.pairs_) {
        const auto& g=space.exchange_integrals(); const auto& eps=space.energies(); const auto& c=space.coefficients();
        views.push_back({space.memory().retained_dimension,{g.data(),g.size()},{eps.data(),eps.size()},{c.data(),c.size()}});
    }
    const BoundedRestrictedPairMP2SolverInput input{o,n,{basis.f_oo_data(),static_cast<std::size_t>(mul(o,o))},views.data(),views.size()};
    const auto inventory=solver_inventory(p,live,p.solver_upper.fixed_control_storage_bytes,p.solver_upper.borrowed_pair_table_bytes);
    result.solver_.emplace(bounded_restricted_pair_mp2_solve(input,options.solver,inventory,caps.solver,&Events::solver,&events));
    Digest identity("vibeqc.periodic.gaussian-pair-mp2.identity");
    for(const auto& s:{ref.state().state_identity_sha256(),ref.dimensions().allocation_identity,basis.identity_sha256(),
        result.context_->source_context_identity_sha256(),result.hf_,result.provider_,result.pairs_sha_,
        result.solver_->input_identity_sha256(),result.solver_->payload_sha256()}) identity.string(s);
    options_wire(identity,options); identity.real(d.occupied_virtual_fock_norm_upper_bound);
    identity.u64(d.complete_common_finite_torus_basis); identity.u64(d.all_pairs_full_rank);
    identity.string("all-unordered-pairs;derived-reverse-transpose;common-real-PAO;periodic-Eq39;coupled-MP2;no-weak-pair-correction;no-symmetry-reduction");
    result.identity_=identity.finish(); events.emit(PeriodicGaussianPairMP2Stage::Finished);
    d.completed_progress_callbacks=events.event.callback_count;
    return result;
}

} // namespace vibeqc
