#include "vibeqc/periodic_gaussian_domain_pair_mp2.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian domain pair MP2 forbids fast/finite-only math"
#endif
static_assert(FLT_EVAL_METHOD==0,"Gaussian domain pair MP2 requires binary64 evaluation");

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
using real::add; using real::mul; using real::Digest;
using I=std::uint64_t;
using Options=PeriodicGaussianDomainPairMP2Options;
using Caps=PeriodicGaussianDomainPairMP2Caps;
using Live=PeriodicGaussianPairMP2LiveInventory;
using Plan=PeriodicGaussianPairMP2Plan;
using Result=PeriodicGaussianPairMP2Result;
using Builder=PeriodicGaussianPairDomainBuilder;
using SolverInventory=BoundedRestrictedPairMP2SolverInventory;
using SolverPlan=BoundedRestrictedPairMP2SolverMemoryPlan;
using PairView=BoundedRestrictedPairMP2PairView;
void limit(I value,I cap,const char* message) {
    if (!cap || value>cap) throw std::length_error(message);
}
I subtract(I a,I b) {
    if (b>a) throw std::logic_error("Gaussian domain pair MP2 inventory subtraction is inconsistent");
    return a-b;
}
void scientific(const Options& o,const Live& live) {
    for (double x : {o.maximum_occupied_virtual_fock_norm,o.projection.maximum_diagonal_symmetry_projection_error,
            o.solver.maximum_diagonal_update_antisymmetry_norm})
        if (!std::isfinite(x) || x<0)
            throw std::invalid_argument("Gaussian domain pair MP2 norm budgets must be explicit finite nonnegative");
    for (double x : {o.solver.denominator_floor,o.solver.residual_tolerance,o.solver.energy_tolerance,
            o.solver.coefficient_orthogonality_tolerance})
        if (!std::isfinite(x) || x<=0)
            throw std::invalid_argument("Gaussian domain pair MP2 solver controls must be explicit finite positive");
    if (!o.solver.maximum_iterations || o.solver.coefficient_orthogonality_tolerance>=1
        || o.pnos.pno.denominator_floor!=o.solver.denominator_floor || !live.fixed_backend_margin_bytes_per_worker)
        throw std::invalid_argument("Gaussian domain pair MP2 denominator/iteration/backend controls differ");
}
void owners(const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicGaussianRealLocalProvider& provider,const Builder& builder) {
    real::float_environment();
    const auto& p=provider.provider(); const auto& ctx=provider.context_handle();
    if (!ref.state_handle() || !ctx || basis.state_handle()!=ref.state_handle()
        || p.state_handle()!=ref.state_handle() || builder.state_handle()!=ref.state_handle()
        || builder.context_handle()!=ctx || builder.basis_identity_sha256()!=basis.identity_sha256()
        || builder.hf_reference_source_identity_sha256()!=provider.hf_reference_source_identity_sha256()
        || p.basis_certificate_identity_sha256()!=basis.identity_sha256()
        || p.local_basis_identity_sha256()!=basis.local_basis_identity_sha256()
        || p.identity_sha256()!=provider.identity_sha256()
        || !provider.matched_finite_gaussian_hf_recipe() || !p.matched_finite_gaussian_hf_recipe()
        || basis.allocation_identity()!=ref.dimensions().allocation_identity
        || builder.allocation_identity()!=ref.dimensions().allocation_identity
        || basis.memory().occupied_count!=provider.memory().occupied_count
        || basis.memory().virtual_count!=provider.memory().virtual_count
        || basis.memory().orbital_count!=provider.memory().orbital_count
        || provider.memory().retained_row_bytes!=p.memory().retained_row_bytes
        || !provider.memory().scalar_work_units || provider.memory().scalar_work_units!=p.memory().scalar_work_units)
        throw std::invalid_argument("Gaussian domain pair MP2 exact state/context/basis/provider/builder lineage differs");
    (void)basis.f_oo_data(); (void)basis.occupied_indices_data(); (void)p.rows_data();
    (void)builder.retained_numerical_bytes(); (void)builder.retained_control_storage_bytes();
}
std::string source_payload(const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicGaussianRealLocalProvider& provider,const Builder& builder) {
    owners(ref,basis,provider,builder);
    Digest h("vibeqc.periodic.gaussian-domain-pair-mp2.sources");
    for (const auto* s : {&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
            &basis.identity_sha256(),&basis.payload_sha256(),&provider.identity_sha256(),
            &provider.provider().payload_sha256(),&builder.identity_sha256(),&builder.payload_sha256()}) h.string(*s);
    const auto* labels=basis.occupied_indices_data();
    for (I at=0;at<mul(2,basis.memory().occupied_count);++at) h.u64(labels[at]);
    const auto* f=basis.f_oo_data();
    for (I at=0;at<basis.memory().retained_fock_bytes/8;++at) h.real(f[at]);
    const auto* rows=provider.provider().rows_data();
    for (I at=0;at<provider.memory().retained_row_bytes/8;++at) h.real(rows[at]);
    return h.finish();
}
void solver_caps(const SolverPlan& p,const BoundedRestrictedPairMP2SolverCaps& c) {
    limit(p.n_occupied,c.maximum_occupied_count,"Gaussian domain MP2 solver occupied cap");
    limit(p.common_virtual_dimension,c.maximum_common_virtual_dimension,"Gaussian domain MP2 solver common dimension cap");
    limit(p.pair_count,c.maximum_pair_count,"Gaussian domain MP2 solver pair cap");
    limit(p.maximum_pair_rank,c.maximum_pair_rank,"Gaussian domain MP2 solver rank cap");
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"Gaussian domain MP2 solver owned cap");
    limit(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"Gaussian domain MP2 solver worker cap");
    limit(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"Gaussian domain MP2 solver node cap");
    limit(p.coupling_slots_upper_bound,c.maximum_coupling_slots,"Gaussian domain MP2 solver coupling cap");
    limit(p.work_units_upper_bound,c.maximum_work_units,"Gaussian domain MP2 solver work cap");
}
SolverInventory solver_inventory(const Plan& p,const Live& live,I fixed,I table,I occupations,I embeddings,I generation_coefficients) {
    SolverInventory out;
    out.numerical_replicas=p.replicas_per_node; out.external_node_bytes=p.reference_base_node_bytes;
    out.fixed_backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
    out.other_live_bytes_per_replica=add(live.other_live_bytes_per_worker,add(p.borrowed_domain_builder_bytes,
        add(p.borrowed_provider_row_bytes,add(subtract(p.borrowed_basis_bytes,mul(8,mul(p.occupied_count,p.occupied_count))),
        add(add(mul(8,occupations),generation_coefficients),add(embeddings,subtract(p.control_storage_reservation_bytes,add(fixed,table))))))));
    return out;
}
struct Events {
    const PeriodicCorrelationAdmittedReference& ref;
    const PeriodicCorrelationRealLocalBasis& basis;
    const PeriodicGaussianRealLocalProvider& provider;
    const Builder& builder;
    const std::string& source;
    const double* fock;
    const std::uint64_t* labels;
    const double* rows;
    PeriodicGaussianPairMP2Progress event;
    PeriodicGaussianPairMP2Callback callback;
    void* context;
    I maximum;
    void emit(PeriodicGaussianPairMP2Stage stage) {
        if (event.callback_count>=maximum) throw std::length_error("Gaussian domain MP2 progress cap exceeded");
        event.stage=stage; ++event.callback_count;
        if (callback) callback(event,context);
        owners(ref,basis,provider,builder);
        if (basis.f_oo_data()!=fock || basis.occupied_indices_data()!=labels || provider.provider().rows_data()!=rows
            || source_payload(ref,basis,provider,builder)!=source)
            throw std::invalid_argument("Gaussian domain MP2 native source changed during progress");
    }
    static void solver(const BoundedRestrictedPairMP2SolverProgress& p,void* context) {
        auto& self=*static_cast<Events*>(context); self.event.solver=p;
        self.emit(PeriodicGaussianPairMP2Stage::Solver);
    }
};
void phase(const Plan& outer,I worker,I node) {
    limit(worker,outer.per_worker_inventoried_bytes,"Gaussian domain MP2 inner worker exceeds macro cap");
    limit(node,outer.required_node_memory_bytes,"Gaussian domain MP2 inner node exceeds macro cap");
}
} // namespace

Plan plan_periodic_gaussian_domain_pair_mp2(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const Builder& builder,const Options& options,const Live& live,const Caps& caps) {
    scientific(options,live); owners(ref,basis,provider,builder);
    const PeriodicGaussianPairDomainBuilderLive domain_live{0,0,live.fixed_backend_margin_bytes_per_worker};
    const auto domain=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,0,0,options.domain,domain_live,caps.domain);
    const I o=basis.memory().occupied_count,n=basis.memory().virtual_count;
    const PeriodicGaussianEmbeddedPairPNOLiveInventory pno_live{0,0,live.fixed_backend_margin_bytes_per_worker};
    const auto pno=plan_periodic_gaussian_embedded_pair_pno_counts(
        {o,n,n,provider.memory().scalar_work_units,provider.memory().retained_row_bytes},options.pnos,pno_live);
    const auto space=periodic_gaussian_pair_space_storage(n,n,n);
    Plan p; p.domain_generated=true; p.occupied_count=o; p.common_virtual_dimension=n; p.pair_count=mul(o,add(o,1))/2;
    limit(p.pair_count,caps.maximum_pair_count,"Gaussian domain MP2 pair count cap exceeded");
    p.borrowed_basis_bytes=basis.memory().retained_output_bytes;
    p.borrowed_provider_row_bytes=provider.memory().retained_row_bytes;
    p.borrowed_domain_builder_bytes=builder.retained_numerical_bytes();
    p.replicas_per_node=domain.replicas_per_node; p.reference_base_node_bytes=domain.reference_base_node_bytes;
    const SolverInventory initial{p.replicas_per_node,p.reference_base_node_bytes,0,live.fixed_backend_margin_bytes_per_worker};
    const auto solver=plan_bounded_restricted_pair_mp2_solver_upper(o,n,n,options.solver.maximum_iterations,initial);
    p.retained_pair_seal_bytes=mul(p.pair_count,21*65);
    p.control_storage_reservation_bytes=add(65536+sizeof(Plan)+sizeof(Result)+sizeof(Options)+sizeof(Live)+sizeof(Caps)+sizeof(Events),
        add(mul(p.pair_count,sizeof(PeriodicGaussianPairSpace)),add(p.retained_pair_seal_bytes,
        add(mul(p.pair_count,domain.retained_geometry_control_upper_bytes),add(domain.control_storage_reservation_bytes,
        add(pno.control_storage_reservation_bytes,add(space.fixed_control_storage_bytes,
        add(solver.fixed_control_storage_bytes,solver.borrowed_pair_table_bytes))))))));
    p.retained_pair_output_upper_bytes=mul(p.pair_count,space.retained_output_bytes);
    p.retained_generation_embedding_upper_bytes=mul(o,pno.borrowed_embedding_bytes);
    p.retained_pair_geometry_upper_bytes=mul(p.pair_count,domain.retained_pair_geometry_upper_bytes);
    p.retained_pair_geometry_control_upper_bytes=mul(p.pair_count,domain.retained_geometry_control_upper_bytes);
    // Conservative reservation includes every final pair and every complete
    // generation geometry even while a current pair is still being generated. This avoids
    // assuming its rank or allocation lifetime before native admission.
    const I retained=add(p.retained_pair_output_upper_bytes,p.retained_pair_geometry_upper_bytes);
    p.pair_generation_phase_upper_bytes=add(retained,std::max(domain.peak_owned_numerical_bytes,
        add(domain.retained_pair_geometry_upper_bytes,std::max(pno.peak_owned_numerical_bytes,space.construction_live_numerical_bytes))));
    p.solver_phase_owned_upper_bytes=add(retained,solver.peak_owned_numerical_bytes);
    p.peak_owned_numerical_bytes=std::max(p.pair_generation_phase_upper_bytes,p.solver_phase_owned_upper_bytes);
    const I borrowed=add(p.borrowed_basis_bytes,add(p.borrowed_provider_row_bytes,p.borrowed_domain_builder_bytes));
    p.per_worker_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(borrowed,add(p.control_storage_reservation_bytes,
        add(live.other_live_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    p.progress_callback_upper_bound=add(add(mul(2,p.pair_count),options.solver.maximum_iterations),2);
    p.integral_calls_upper_bound=mul(p.pair_count,add(pno.integral_calls,space.integral_calls));
    const I source_scan=mul(512,add(4096,add(p.borrowed_basis_bytes/8,p.borrowed_provider_row_bytes/8)));
    p.driver_work_units=add(mul(add(p.progress_callback_upper_bound,2),source_scan),mul(4096,add(p.pair_count,add(o,n))));
    p.work_units_upper_bound=add(p.driver_work_units,add(solver.work_units_upper_bound,mul(p.pair_count,
        add(domain.work_units,add(pno.work_units,add(space.numerical_work_units,
            mul(space.integral_calls,provider.memory().scalar_work_units)))))));
    for (I bytes : {p.peak_owned_numerical_bytes,p.control_storage_reservation_bytes,p.retained_generation_embedding_upper_bytes,
            p.retained_pair_geometry_upper_bytes,p.retained_pair_geometry_control_upper_bytes}) real::extent(bytes);
    if (p.pair_count>std::vector<PeriodicGaussianPairSpace>().max_size() || p.pair_count>std::vector<PairView>().max_size()
        || p.pair_count>std::vector<PeriodicGaussianPairDomainGeometry>().max_size())
        throw std::length_error("Gaussian domain MP2 native owner-table extent exceeded");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian domain MP2 owned cap");
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes,"Gaussian domain MP2 control cap");
    limit(p.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"Gaussian domain MP2 worker cap");
    limit(p.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"Gaussian domain MP2 node cap");
    limit(p.required_node_memory_bytes,ref.budget().memory_limit_bytes,"Gaussian domain MP2 admitted node cap");
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gaussian domain MP2 progress cap");
    limit(p.integral_calls_upper_bound,caps.maximum_integral_calls,"Gaussian domain MP2 integral cap");
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian domain MP2 work cap");
    // The enclosing peak is also a conservative upper for EACH leaf worker,
    // with fixed/backend/reference reservations shared, not readmitted.
    for (I cap : {caps.domain.maximum_worker_bytes,caps.pnos.maximum_per_worker_inventoried_bytes,
            caps.projection.maximum_per_worker_inventoried_bytes})
        limit(p.per_worker_inventoried_bytes,cap,"Gaussian domain MP2 subordinate worker cap");
    for (I cap : {caps.domain.maximum_node_bytes,caps.pnos.maximum_node_inventoried_bytes,
            caps.projection.maximum_node_inventoried_bytes})
        limit(p.required_node_memory_bytes,cap,"Gaussian domain MP2 subordinate node cap");
    for (I cap : {caps.domain.maximum_control_storage_bytes,caps.pnos.maximum_control_storage_bytes_per_worker})
        limit(p.control_storage_reservation_bytes,cap,"Gaussian domain MP2 subordinate control cap");
    limit(n,caps.pnos.maximum_common_dimension,"Gaussian domain MP2 PNO common dimension cap");
    limit(n,caps.pnos.maximum_generation_dimension,"Gaussian domain MP2 PNO generation dimension cap");
    limit(pno.peak_owned_numerical_bytes,caps.pnos.maximum_owned_numerical_bytes,"Gaussian domain MP2 PNO owned cap");
    limit(pno.integral_calls,caps.pnos.maximum_integral_calls,"Gaussian domain MP2 PNO integral cap");
    limit(pno.work_units,caps.pnos.maximum_work_units,"Gaussian domain MP2 PNO work cap");
    limit(space.peak_owned_numerical_bytes,caps.projection.maximum_owned_numerical_bytes,"Gaussian domain MP2 projection owned cap");
    limit(space.integral_calls,caps.projection.maximum_integral_calls,"Gaussian domain MP2 projection integral cap");
    limit(add(space.numerical_work_units,mul(space.integral_calls,provider.memory().scalar_work_units)),
        caps.projection.maximum_work_units,"Gaussian domain MP2 projection work cap");
    p.solver_upper=plan_bounded_restricted_pair_mp2_solver_upper(o,n,n,options.solver.maximum_iterations,
        solver_inventory(p,live,solver.fixed_control_storage_bytes,solver.borrowed_pair_table_bytes,
            mul(p.pair_count,n),p.retained_pair_geometry_upper_bytes,mul(8,mul(p.pair_count,mul(n,n)))));
    solver_caps(p.solver_upper,caps.solver);
    phase(p,p.solver_upper.per_replica_inventoried_bytes,p.solver_upper.required_node_inventoried_bytes);
    return p;
}

Result run_periodic_gaussian_domain_pair_mp2(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const Builder& builder,const Options& supplied_options,const Live& supplied_live,const Caps& supplied_caps,
    PeriodicGaussianPairMP2Callback callback,void* callback_context) {
    const auto options=supplied_options; const auto live=supplied_live; const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_domain_pair_mp2(ref,basis,provider,builder,options,live,caps);
    const I o=p.occupied_count,n=p.common_virtual_dimension;
    Result result; result.memory_=p; result.state_=ref.state_handle(); result.context_=provider.context_handle();
    result.hf_=provider.hf_reference_source_identity_sha256(); result.provider_=provider.identity_sha256();
    result.builder_=builder.identity_sha256();
    auto& d=result.diagnostics_; d.domain_generated=true; d.minimum_pair_rank=n; d.all_pairs_full_rank=true;
    d.complete_common_finite_torus_basis=o==mul(ref.state().n_kpoints(),ref.state().n_correlated_occupied())
        && n==mul(ref.state().n_kpoints(),ref.state().n_virtual());
    double squared=0;
    for (I at=0;at<mul(o,n);++at) real::norm_lane(squared,std::abs(real::finite(basis.f_ov_data()[at])));
    d.occupied_virtual_fock_norm_upper_bound=real::sqrt_up(squared);
    if (d.occupied_virtual_fock_norm_upper_bound>options.maximum_occupied_virtual_fock_norm)
        throw std::invalid_argument("Gaussian domain MP2 occupied-virtual Fock exceeds explicit Brillouin budget");
    const auto source=source_payload(ref,basis,provider,builder);
    Events events{ref,basis,provider,builder,source,basis.f_oo_data(),basis.occupied_indices_data(),
        provider.provider().rows_data(),{},callback,callback_context,p.progress_callback_upper_bound};
    events.emit(PeriodicGaussianPairMP2Stage::Begin);
    result.pairs_.reserve(static_cast<std::size_t>(p.pair_count));
    result.pair_geometry_.reserve(static_cast<std::size_t>(p.pair_count));
    Digest pair_seal("vibeqc.periodic.gaussian-pair-mp2.spaces"); pair_seal.u64(p.pair_count);
    for (I i=0;i<o;++i) for (I j=i;j<o;++j) {
        const I retained=add(d.retained_pair_bytes,d.retained_pair_geometry_bytes);
        PeriodicGaussianPairDomainBuilderLive dl{add(live.other_live_bytes_per_worker,
            add(retained,p.borrowed_provider_row_bytes)),0,live.fixed_backend_margin_bytes_per_worker};
        const auto pre=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options.domain,dl,caps.domain);
        dl.other_live_control_bytes_per_worker=subtract(p.control_storage_reservation_bytes,pre.control_storage_reservation_bytes);
        const auto dp=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options.domain,dl,caps.domain);
        phase(p,dp.worker_bytes,dp.required_node_memory_bytes);
        auto geometry=make_periodic_gaussian_pair_domain_geometry(ref,basis,builder,i,j,options.domain,dl,caps.domain);
        const auto& embedding=geometry.embedding();
        const I embedding_bytes=embedding.memory().output_numerical_bytes;
        const I geometry_bytes=geometry.retained_numerical_bytes();
        events.event.occupied_i=i; events.event.occupied_j=j; events.event.pair_rank=embedding.memory().pair_dimension;
        events.emit(PeriodicGaussianPairMP2Stage::PairDomainReady);
        PeriodicGaussianEmbeddedPairPNOLiveInventory pl{add(live.other_live_bytes_per_worker,
            add(retained,add(p.borrowed_domain_builder_bytes,subtract(geometry_bytes,embedding_bytes)))),0,live.fixed_backend_margin_bytes_per_worker};
        const auto pp0=plan_periodic_gaussian_embedded_pair_pnos(ref,basis,provider,embedding,i,j,options.pnos,pl,caps.pnos);
        pl.other_live_control_bytes_per_worker=subtract(p.control_storage_reservation_bytes,pp0.control_storage_reservation_bytes);
        const auto pp=plan_periodic_gaussian_embedded_pair_pnos(ref,basis,provider,embedding,i,j,options.pnos,pl,caps.pnos);
        phase(p,pp.per_worker_inventoried_bytes,pp.required_node_memory_bytes);
        auto pno=make_periodic_gaussian_embedded_pair_pnos(ref,basis,provider,embedding,i,j,options.pnos,pl,caps.pnos);
        d.completed_integral_calls=add(d.completed_integral_calls,pno.diagnostics().generation.completed_integral_calls);
        const I m=pno.memory().generation_dimension;
        const auto storage=periodic_gaussian_pair_space_storage(n,m,pno.diagnostics().retained_dimension);
        const PeriodicGaussianPairSpaceLiveInventory sl{
            add(live.other_live_bytes_per_worker,add(retained,add(p.borrowed_domain_builder_bytes,
                add(geometry_bytes,subtract(p.control_storage_reservation_bytes,storage.fixed_control_storage_bytes))))),
            live.fixed_backend_margin_bytes_per_worker};
        const auto sp=plan_periodic_gaussian_pair_space(ref,basis,provider,pno,options.projection,sl,caps.projection);
        phase(p,sp.per_worker_inventoried_bytes,sp.required_node_memory_bytes);
        auto space=make_periodic_gaussian_pair_space(ref,basis,provider,std::move(pno),options.projection,sl,caps.projection);
        d.completed_integral_calls=add(d.completed_integral_calls,space.diagnostics().completed_integral_calls);
        d.retained_pair_bytes=add(d.retained_pair_bytes,space.memory().retained_output_bytes);
        d.generation_dimension_sum=add(d.generation_dimension_sum,m);
        d.retained_generation_coefficient_bytes=add(d.retained_generation_coefficient_bytes,
            mul(8,mul(m,space.memory().retained_dimension)));
        const I rank=space.memory().retained_dimension;
        d.minimum_pair_rank=std::min(d.minimum_pair_rank,rank); d.maximum_pair_rank=std::max(d.maximum_pair_rank,rank);
        d.zero_rank_pairs+=!rank; d.all_pairs_full_rank=d.all_pairs_full_rank && rank==n;
        d.maximum_diagonal_integral_projection_norm=std::max(d.maximum_diagonal_integral_projection_norm,
            space.diagnostics().diagonal_symmetry_projection_frobenius_bound);
        if (i==j) {
            d.diagonal_generation_dimension_sum=add(d.diagonal_generation_dimension_sum,m);
            d.retained_generation_embedding_bytes=add(d.retained_generation_embedding_bytes,embedding_bytes);
        }
        d.retained_pair_geometry_bytes=add(d.retained_pair_geometry_bytes,geometry_bytes);
        d.retained_pair_geometry_control_bytes=add(d.retained_pair_geometry_control_bytes,geometry.retained_control_storage_bytes());
        if (d.retained_pair_bytes>p.retained_pair_output_upper_bytes
            || d.retained_generation_embedding_bytes>p.retained_generation_embedding_upper_bytes
            || d.retained_pair_geometry_bytes>p.retained_pair_geometry_upper_bytes
            || d.retained_pair_geometry_control_bytes>p.retained_pair_geometry_control_upper_bytes
            || d.completed_integral_calls>p.integral_calls_upper_bound)
            throw std::logic_error("Gaussian domain MP2 exact retained/integral census exceeds admission");
        pair_seal.u64(i); pair_seal.u64(j); pair_seal.string(space.identity_sha256());
        result.pair_geometry_.push_back(std::move(geometry));
        result.pairs_.push_back(std::move(space)); ++d.completed_pairs;
        events.event.completed_pairs=d.completed_pairs; events.event.pair_rank=rank;
        events.emit(PeriodicGaussianPairMP2Stage::PairComplete);
    }
    result.pairs_sha_=pair_seal.finish();
    std::vector<PairView> views; views.reserve(static_cast<std::size_t>(p.pair_count));
    for (const auto& space : result.pairs_) {
        const auto& g=space.exchange_integrals(); const auto& eps=space.energies(); const auto& c=space.coefficients();
        views.push_back({space.memory().retained_dimension,{g.data(),g.size()},{eps.data(),eps.size()},{c.data(),c.size()}});
    }
    const BoundedRestrictedPairMP2SolverInput input{o,n,{basis.f_oo_data(),static_cast<std::size_t>(mul(o,o))},views.data(),views.size()};
    const auto inventory=solver_inventory(p,live,p.solver_upper.fixed_control_storage_bytes,p.solver_upper.borrowed_pair_table_bytes,
        d.generation_dimension_sum,d.retained_pair_geometry_bytes,d.retained_generation_coefficient_bytes);
    result.solver_.emplace(bounded_restricted_pair_mp2_solve(input,options.solver,inventory,caps.solver,&Events::solver,&events));
    Digest identity("vibeqc.periodic.gaussian-domain-pair-mp2.identity");
    for (const auto* s : std::array<const std::string*,10>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,&basis.identity_sha256(),
            &result.hf_,&result.provider_,&result.builder_,&result.pairs_sha_,&source,
            &result.solver_->input_identity_sha256(),&result.solver_->payload_sha256()}) identity.string(*s);
    for (const auto& g : result.pair_geometry_) identity.string(g.identity_sha256());
    identity.u64(d.generation_dimension_sum); identity.u64(d.diagonal_generation_dimension_sum);
    identity.u64(d.retained_generation_coefficient_bytes);
    identity.u64(d.retained_pair_geometry_bytes);identity.u64(d.retained_pair_geometry_control_bytes);
    identity.real(d.occupied_virtual_fock_norm_upper_bound); identity.real(options.maximum_occupied_virtual_fock_norm);
    identity.u64(d.complete_common_finite_torus_basis); identity.u64(d.all_pairs_full_rank);
    identity.string("actual-HF-selected-domains;streamed-pair-PAO;pair-frame-Eq39;all-authentic-pair-geometries-retained;"
        "all-placed-pairs;coupled-MP2;no-weak-pair-correction;no-space-group-reduction;not-production-DLPNO");
    result.identity_=identity.finish(); events.emit(PeriodicGaussianPairMP2Stage::Finished);
    d.completed_progress_callbacks=events.event.callback_count;
    return result;
}
namespace {
using GramConfig=PeriodicGaussianGramDomainPairMP2Config;
using GramOptions=PeriodicGaussianGramDomainPairMP2Options;
using GramCaps=PeriodicGaussianGramDomainPairMP2Caps;
using Ref=PeriodicCorrelationAdmittedReference;
using Basis=PeriodicCorrelationRealLocalBasis;
using Wannier=PeriodicCorrelationWannier;
using HF=PeriodicGaussianRHFResult;
using Complex=std::complex<double>;
namespace local=periodic_correlation_local_detail;

I gram_sources_bytes(const Plan& p) {
    return add(p.borrowed_gaussian_bytes,add(p.borrowed_wannier_bytes,p.borrowed_gauge_bytes));
}
void gram_scientific(const GramConfig& c,const GramOptions& o,const Live& live,const GramCaps& caps) {
    // Reuse the unchanged coupled-solver convention check, not a numerical
    // adapter or a provider-backed source. All remaining seed controls are
    // scalar and can be validated before a genuine pair geometry exists.
    Options old;old.projection=o.projection;old.solver=o.solver;
    old.maximum_occupied_virtual_fock_norm=o.maximum_occupied_virtual_fock_norm;
    old.pnos.pno.denominator_floor=o.pnos.pno.pno.denominator_floor;
    scientific(old,live);
    const auto& a=o.pnos;
    real::tolerance_pair(a.local_basis.coefficient_tr_absolute_tolerance,a.local_basis.coefficient_tr_relative_tolerance);
    real::tolerance_pair(a.local_basis.orthonormality_absolute_tolerance,a.local_basis.orthonormality_relative_tolerance);
    real::tolerance_pair(a.local_basis.fock_absolute_tolerance,a.local_basis.fock_relative_tolerance);
    real::tolerance_pair(a.gram.reversal_absolute_tolerance,a.gram.reversal_relative_tolerance);
    real::tolerance_pair(a.gram.conjugacy_absolute_tolerance,a.gram.conjugacy_relative_tolerance);
    real::tolerance_pair(a.gram.self_q_absolute_tolerance,a.gram.self_q_relative_tolerance);
    for(double x:{a.local_basis.maximum_fock_projection_error,a.gram.maximum_eri_projection_error,
        a.gram.maximum_scalar_roundoff_error,a.maximum_occupied_fock_difference,a.pno.maximum_exported_gram_error,
        a.pno.maximum_exported_fock_error,a.pno.maximum_exported_subspace_error,a.pno.pno.semicanonical_orthonormality_tolerance})
        if(!std::isfinite(x) || x<=0) throw std::invalid_argument("Gram domain MP2 scientific tolerances must be finite positive");
    for(double x:{a.pno.pno.occupation_cutoff,a.pno.pno.maximum_initial_fvv_offdiagonal_norm,
        a.pno.maximum_diagonal_exchange_projection_norm,a.pno.maximum_fock_symmetry_projection_norm,
        a.maximum_retained_diagonal_projection_norm})
        if(!std::isfinite(x) || x<0) throw std::invalid_argument("Gram domain MP2 projection budgets must be finite nonnegative");
    if(a.pno.pno.semicanonical_orthonormality_tolerance>=1)
        throw std::invalid_argument("Gram domain MP2 orthonormality tolerance must be below one");
    for(const auto& e:{a.pno.pno.pno_eigensolver,a.pno.pno.semicanonical_eigensolver})
        if(!e.max_sweeps || !std::isfinite(e.relative_offdiagonal_tolerance)
            || e.relative_offdiagonal_tolerance<=0 || e.relative_offdiagonal_tolerance>=1)
            throw std::invalid_argument("Gram domain MP2 requires explicit eigensolver controls");
    for(I x:{c.pnos.gram.auxiliary_block,c.pnos.gram.panel.ao_pair_block,
        caps.maximum_pno_leaf_control_storage_bytes,caps.pnos.maximum_owned_numerical_bytes,
        caps.pnos.maximum_work_units,caps.pnos.maximum_gram_control_storage_bytes,
        caps.pnos.gram.maximum_factor_panels,caps.pnos.gram.maximum_tile_calls})
        if(!x) throw std::invalid_argument("Gram domain MP2 requires positive explicit resource controls");
}
void gram_owners(const HF& hf,const Ref& ref,const Wannier& w,const Basis& b,const Builder& builder) {
    real::float_environment();
    if(!ref.state_handle() || !hf.converged() || !hf.matched_finite_gaussian_hf_source()
        || !hf.context_handle() || hf.state_handle()!=ref.state_handle()
        || b.state_handle()!=ref.state_handle() || builder.state_handle()!=ref.state_handle()
        || builder.context_handle()!=hf.context_handle() || b.allocation_identity()!=ref.dimensions().allocation_identity
        || builder.allocation_identity()!=ref.dimensions().allocation_identity
        || builder.basis_identity_sha256()!=b.identity_sha256()
        || builder.hf_reference_source_identity_sha256()!=hf.reference_source_identity_sha256())
        throw std::invalid_argument("Gram domain MP2 requires exact original HF/reference/basis/builder owners");
    local::validate_wannier(ref,w);
    const auto& d=builder.common_domain();const auto& s=builder.common_real_space().space();
    local::validate_pao(ref,d,s);
    const auto& bm=b.memory();const auto& ctx=*hf.context_handle();
    if(!bm.occupied_count || !bm.virtual_count || bm.orbital_count!=add(bm.occupied_count,bm.virtual_count)
        || bm.n_cells!=ref.state().n_kpoints() || bm.n_basis!=ref.state().n_basis()
        || bm.occupied_count>mul(bm.n_cells,ref.state().n_correlated_occupied())
        || bm.virtual_count>mul(bm.n_cells,ref.state().n_virtual())
        || ctx.inventory().ao.function_count!=bm.n_basis
        || ctx.inventory().auxiliary.function_count!=ref.dimensions().n_auxiliary
        || ctx.mesh().mesh()!=ref.state().mesh() || ctx.mesh().is_shift()!=ref.state().is_shift())
        throw std::invalid_argument("Gram domain MP2 original source dimensions differ");
    (void)b.occupied_indices_data();(void)b.f_oo_data();(void)b.f_vv_data();(void)b.f_ov_data();
    (void)w.cell_coefficients(0);(void)d.overlap_data();(void)d.fock_data();
    (void)s.coefficients_data();(void)s.energies_data();(void)s.overlap_eigenvalues_data();
    (void)builder.retained_numerical_bytes();(void)builder.retained_control_storage_bytes();
}
void gram_reference(const Ref& ref,const Plan& p) {
    const auto& d=ref.dimensions();const auto& b=ref.budget();
    if(ref.state_resident_bytes()!=ref.state().resident_bytes() || d.external_bytes<ref.state_resident_bytes()
        || mul(b.mpi_ranks,b.workers_per_rank)!=p.replicas_per_node
        || add(add(d.external_bytes,d.shared_bytes),mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)))
            !=p.reference_base_node_bytes || p.required_node_memory_bytes>b.memory_limit_bytes)
        throw std::invalid_argument("Gram domain MP2 admitted reference resource owner changed");
}
PeriodicGaussianSourceCaps gram_basis_caps(const PeriodicGaussianSourceContext& c) {
    const auto& v=c.inventory();const auto& a=v.ao;const auto& b=v.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),c.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),v.combined_borrowed_active_numeric_bytes,v.work_units_upper_bound};
}
std::string gram_source_payload(const HF& hf,const Ref& ref,const BasisSet& ao,const BasisSet& aux,
    const Wannier& w,const Complex* gauges,std::size_t count,const Basis& b,const Builder& builder) {
    gram_owners(hf,ref,w,b,builder);
    hf.context_handle()->verify_bases(ao,aux,gram_basis_caps(*hf.context_handle()));
    const auto gauge=local::validate_gauges(w,gauges,count);
    const auto& domain=builder.common_domain();const auto& space=builder.common_real_space().space();
    real::indices(b.occupied_indices_data(),mul(2,b.memory().occupied_count),b.memory().occupied_count,ref.state());
    if(real::local_basis_identity(ref,w,gauge,domain,space,b.occupied_indices_data(),b.memory().occupied_count,
        b.virtual_selection())!=b.local_basis_identity_sha256())
        throw std::invalid_argument("Gram domain MP2 Wannier/gauge differs from original common basis localization");
    Digest h("vibeqc.periodic.gaussian-gram-domain-pair-mp2.sources");
    h.string(hf.original_input_identity_sha256());h.string(hf.one_electron_source_identity_sha256());
    h.string(hf.final_fock_source_identity_sha256());h.string(hf.reference_source_identity_sha256());
    h.string(hf.context_handle()->source_context_identity_sha256());
    for(const auto* s:{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &b.identity_sha256(),&b.payload_sha256(),&builder.identity_sha256(),&builder.payload_sha256(),
        &w.wannier_identity_sha256(),&w.coefficient_payload_sha256(),&gauge}) h.string(*s);
    Digest bp("vibeqc.periodic.correlation.real-local-basis.payload");
    bp.u64(b.memory().occupied_count);bp.u64(b.memory().virtual_count);
    for(I x=0;x<mul(2,b.memory().occupied_count);++x) {bp.u64(b.occupied_indices_data()[x]);h.u64(b.occupied_indices_data()[x]);}
    for(I x=0;x<b.memory().retained_fock_bytes/8;++x) {bp.real(b.f_oo_data()[x]);h.real(b.f_oo_data()[x]);}
    if(bp.finish()!=b.payload_sha256()) throw std::invalid_argument("Gram domain MP2 original common F payload changed");
    static_assert(kPeriodicCorrelationWannierContractVersion==1,"Gram domain MP2 Wannier codec changed");
    Digest wp("vibeqc.periodic.correlation.wannier.coefficients");
    for(int x:ref.state().mesh()) wp.u32(x);wp.u64(w.n_basis());wp.u64(w.n_home_occupied());
    for(I cell=0;cell<w.n_cells();++cell) for(I x=0;x<mul(w.n_basis(),w.n_home_occupied());++x) {
        const auto z=w.cell_coefficients(cell)[x];wp.complex(z);h.complex(z);
    }
    if(wp.finish()!=w.coefficient_payload_sha256()) throw std::invalid_argument("Gram domain MP2 Wannier payload changed");
    const I D=domain.domain_dimension(),r=space.retained_dimension();
    Digest di("vibeqc.periodic.correlation.pao.domain");for(int x:ref.state().mesh()) di.u32(x);
    di.u64(ref.state().n_basis());di.u64(D);
    for(I x=0;x<D;++x) {const auto c=domain.column(x);di.u64(c.cell);di.u64(c.ao);h.u64(c.cell);h.u64(c.ao);}
    Digest dm("vibeqc.periodic.correlation.pao.matrices");dm.u64(D);
    for(const auto* a:{domain.overlap_data(),domain.fock_data()}) for(I x=0;x<mul(D,D);++x) {dm.complex(a[x]);h.complex(a[x]);}
    Digest sp("vibeqc.periodic.correlation.pao.space.payload");sp.u64(D);sp.u64(r);
    for(I x=0;x<mul(D,r);++x) {sp.complex(space.coefficients_data()[x]);h.complex(space.coefficients_data()[x]);}
    for(I x=0;x<r;++x) {sp.real(space.energies_data()[x]);h.real(space.energies_data()[x]);}
    for(I x=0;x<D;++x) {sp.real(space.overlap_eigenvalues_data()[x]);h.real(space.overlap_eigenvalues_data()[x]);}
    if(di.finish()!=domain.domain_index_sha256() || dm.finish()!=domain.matrix_payload_sha256() || sp.finish()!=space.payload_sha256())
        throw std::invalid_argument("Gram domain MP2 original common geometry payload changed");
    return h.finish();
}
struct GramEvents {
    const HF& hf;const Ref& ref;const BasisSet& ao;const BasisSet& aux;const Wannier& w;
    const Complex* gauges;std::size_t gauge_count;const Basis& basis;const Builder& builder;
    const Plan& plan;const std::string& source;
    const PeriodicRestrictedMeanFieldState* state;const PeriodicGaussianSourceContext* context_owner;
    const I* labels;const double* foo;const double* fvv;const double* fov;
    const Complex* wannier;const Complex* overlap;const Complex* fock;const Complex* coefficients;
    const double* energies;const double* metric_values;
    PeriodicGaussianPairMP2Progress event;
    PeriodicGaussianPairMP2Callback callback;void* context;
    void check() const {
        // Getter/owner checks precede every captured raw-pointer scan. This
        // detects equal-content move replacements as well as source mutation.
        gram_owners(hf,ref,w,basis,builder);gram_reference(ref,plan);
        const auto& d=builder.common_domain();const auto& s=builder.common_real_space().space();
        if(ref.state_handle().get()!=state || hf.context_handle().get()!=context_owner
            || basis.occupied_indices_data()!=labels || basis.f_oo_data()!=foo || basis.f_vv_data()!=fvv || basis.f_ov_data()!=fov
            || w.cell_coefficients(0)!=wannier || d.overlap_data()!=overlap || d.fock_data()!=fock
            || s.coefficients_data()!=coefficients || s.energies_data()!=energies || s.overlap_eigenvalues_data()!=metric_values)
            throw std::invalid_argument("Gram domain MP2 source storage changed during progress");
        if(gram_source_payload(hf,ref,ao,aux,w,gauges,gauge_count,basis,builder)!=source)
            throw std::invalid_argument("Gram domain MP2 original source payload changed during progress");
    }
    void emit(PeriodicGaussianPairMP2Stage stage) {
        if(event.callback_count>=plan.progress_callback_upper_bound) throw std::length_error("Gram domain MP2 progress cap exceeded");
        event.stage=stage;++event.callback_count;if(callback) callback(event,context);check();
    }
    static void solver(const BoundedRestrictedPairMP2SolverProgress& progress,void* context) {
        auto& self=*static_cast<GramEvents*>(context);self.event.solver=progress;self.emit(PeriodicGaussianPairMP2Stage::Solver);
    }
};
SolverInventory gram_solver_inventory(const Plan& p,const Live& live,I occupations,I geometries,I local_coefficients) {
    auto extra=live;extra.other_live_bytes_per_worker=add(extra.other_live_bytes_per_worker,gram_sources_bytes(p));
    return solver_inventory(p,extra,p.solver_upper.fixed_control_storage_bytes,p.solver_upper.borrowed_pair_table_bytes,
        occupations,geometries,local_coefficients);
}
void clamp_metric(PeriodicGaussianMetricCaps& c,I owned,I worker,I node,I work) {
    c.maximum_owned_numeric_bytes=std::min(c.maximum_owned_numeric_bytes,owned);
    c.maximum_per_replica_inventoried_bytes=std::min(c.maximum_per_replica_inventoried_bytes,worker);
    c.maximum_node_inventoried_bytes=std::min(c.maximum_node_inventoried_bytes,node);
    c.maximum_work_units=std::min(c.maximum_work_units,work);
}
PeriodicGaussianGramPairPNOCaps gram_seed_caps(const GramCaps& caps,const Plan& p,I retained,I geometry) {
    auto c=caps.pnos;
    c.maximum_owned_numerical_bytes=std::min(c.maximum_owned_numerical_bytes,
        subtract(p.pair_generation_phase_upper_bytes,add(retained,geometry)));
    c.maximum_control_storage_bytes=std::min(c.maximum_control_storage_bytes,p.control_storage_reservation_bytes);
    c.maximum_worker_bytes=std::min(c.maximum_worker_bytes,p.per_worker_inventoried_bytes);
    c.maximum_node_bytes=std::min(c.maximum_node_bytes,p.required_node_memory_bytes);
    c.maximum_work_units=std::min(c.maximum_work_units,p.work_units_upper_bound);
    c.local_basis.maximum_owned_numerical_bytes=std::min(c.local_basis.maximum_owned_numerical_bytes,c.maximum_owned_numerical_bytes);
    c.local_basis.maximum_work_units=std::min(c.local_basis.maximum_work_units,c.maximum_work_units);
    for(auto* metric:{&c.gram.resources,&c.gram.metric,&c.gram.panel.resources,&c.gram.panel.tile.resources})
        clamp_metric(*metric,c.maximum_owned_numerical_bytes,c.maximum_worker_bytes,c.maximum_node_bytes,c.maximum_work_units);
    return c;
}
} // namespace

Plan plan_periodic_gaussian_domain_pair_mp2(const HF& hf,const Ref& ref,const Wannier& w,const Basis& basis,
    const Builder& builder,const GramConfig& config,const GramOptions& options,const Live& live,const GramCaps& caps) {
    gram_scientific(config,options,live,caps);gram_owners(hf,ref,w,basis,builder);
    const PeriodicGaussianPairDomainBuilderLive dl{0,0,live.fixed_backend_margin_bytes_per_worker};
    const auto domain=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,0,0,options.domain,dl,caps.domain);
    const I o=basis.memory().occupied_count,n=basis.memory().virtual_count;
    Plan p;p.source_kind=PeriodicGaussianPairMP2SourceKind::DirectPAOGram;p.domain_generated=true;
    p.occupied_count=o;p.common_virtual_dimension=n;p.pair_count=mul(o,add(o,1))/2;
    limit(p.pair_count,caps.maximum_pair_count,"Gram domain MP2 pair count cap exceeded");
    p.borrowed_basis_bytes=basis.memory().retained_output_bytes;p.borrowed_domain_builder_bytes=builder.retained_numerical_bytes();
    p.borrowed_gaussian_bytes=hf.context_handle()->inventory().combined_borrowed_active_numeric_bytes;
    p.borrowed_wannier_bytes=w.memory().retained_coefficient_bytes;p.borrowed_gauge_bytes=w.memory().caller_gauge_bytes;
    p.replicas_per_node=domain.replicas_per_node;p.reference_base_node_bytes=domain.reference_base_node_bytes;
    const SolverInventory si{p.replicas_per_node,p.reference_base_node_bytes,0,live.fixed_backend_margin_bytes_per_worker};
    const auto solver=plan_bounded_restricted_pair_mp2_solver_upper(o,n,n,options.solver.maximum_iterations,si);
    const auto space=periodic_gaussian_direct_gram_pair_space_storage(n,n,n);
    // Count-only seed admission uses an explicit bare-control reservation:
    // no fabricated Geometry or PNO owner is constructed just for planning.
    p.retained_pair_seal_bytes=mul(p.pair_count,26*65);
    p.retained_pair_geometry_control_upper_bytes=mul(p.pair_count,domain.retained_geometry_control_upper_bytes);
    p.control_storage_reservation_bytes=add(131072+3*sizeof(Plan)+2*sizeof(Result)+2*sizeof(GramOptions)+2*sizeof(GramCaps)
        +2*sizeof(GramConfig)+2*sizeof(Live)+sizeof(GramEvents)+sizeof(HF)+sizeof(Ref)+sizeof(Wannier)+sizeof(Basis)
        +sizeof(PeriodicGaussianSourceContext)+12*sizeof(Digest)+32*65,
        add(mul(p.pair_count,sizeof(PeriodicGaussianPairSpace)+sizeof(PeriodicGaussianPairDomainGeometry)),
        add(p.retained_pair_seal_bytes,add(p.retained_pair_geometry_control_upper_bytes,
        add(domain.control_storage_reservation_bytes,add(caps.maximum_pno_leaf_control_storage_bytes,
        add(space.fixed_control_storage_bytes,add(solver.fixed_control_storage_bytes,solver.borrowed_pair_table_bytes))))))));
    p.retained_pair_output_upper_bytes=mul(p.pair_count,space.retained_output_bytes);
    p.retained_generation_embedding_upper_bytes=mul(o,domain.retained_embedding_upper_bytes);
    p.retained_pair_geometry_upper_bytes=mul(p.pair_count,domain.retained_pair_geometry_upper_bytes);
    const I retained=add(p.retained_pair_output_upper_bytes,p.retained_pair_geometry_upper_bytes);
    p.pair_generation_phase_upper_bytes=add(retained,std::max(domain.peak_owned_numerical_bytes,
        add(domain.retained_pair_geometry_upper_bytes,std::max(caps.pnos.maximum_owned_numerical_bytes,space.construction_live_numerical_bytes))));
    p.solver_phase_owned_upper_bytes=add(retained,solver.peak_owned_numerical_bytes);
    p.peak_owned_numerical_bytes=std::max(p.pair_generation_phase_upper_bytes,p.solver_phase_owned_upper_bytes);
    const I borrowed=add(p.borrowed_basis_bytes,add(p.borrowed_domain_builder_bytes,gram_sources_bytes(p)));
    p.per_worker_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(borrowed,add(p.control_storage_reservation_bytes,
        add(live.other_live_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    p.progress_callback_upper_bound=add(add(mul(2,p.pair_count),options.solver.maximum_iterations),2);
    p.gram_builds_upper_bound=p.pair_count;
    p.factor_panels_upper_bound=mul(p.pair_count,caps.pnos.gram.maximum_factor_panels);
    p.tile_calls_upper_bound=mul(p.pair_count,caps.pnos.gram.maximum_tile_calls);
    const I scans=add(hf.context_handle()->inventory().work_units_upper_bound,
        mul(4096,add(65536,add(borrowed/8,add(mul(o,o),mul(w.n_cells(),mul(w.n_home_occupied(),mul(w.n_home_occupied(),w.n_home_occupied()))))))));
    // Two external count-only domain/seed plans per pair, plus the native
    // factory's self-plan already charged within its complete work ceiling.
    const I metadata=mul(4*1048576,add(65536,add(domain.maximum_domain_dimension,add(n,add(o,p.pair_count)))));
    p.driver_work_units=add(mul(add(p.progress_callback_upper_bound,2),scans),mul(p.pair_count,metadata));
    p.work_units_upper_bound=add(p.driver_work_units,add(solver.work_units_upper_bound,
        mul(p.pair_count,add(domain.work_units,add(caps.pnos.maximum_work_units,mul(2,space.numerical_work_units))))));
    for(I bytes:{p.peak_owned_numerical_bytes,p.control_storage_reservation_bytes,p.retained_pair_geometry_upper_bytes,
        p.retained_pair_output_upper_bytes,borrowed}) real::extent(bytes);
    if(p.pair_count>std::vector<PeriodicGaussianPairSpace>().max_size() || p.pair_count>std::vector<PairView>().max_size()
        || p.pair_count>std::vector<PeriodicGaussianPairDomainGeometry>().max_size())
        throw std::length_error("Gram domain MP2 owner table extent exceeded");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gram domain MP2 owned cap exceeded");
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes,"Gram domain MP2 control cap exceeded");
    limit(p.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"Gram domain MP2 worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"Gram domain MP2 node cap exceeded");
    limit(p.required_node_memory_bytes,ref.budget().memory_limit_bytes,"Gram domain MP2 admitted node cap exceeded");
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gram domain MP2 progress cap exceeded");
    limit(p.gram_builds_upper_bound,caps.maximum_gram_builds,"Gram domain MP2 Gram build cap exceeded");
    limit(p.factor_panels_upper_bound,caps.maximum_factor_panels,"Gram domain MP2 factor panel cap exceeded");
    limit(p.tile_calls_upper_bound,caps.maximum_tile_calls,"Gram domain MP2 tile cap exceeded");
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gram domain MP2 work cap exceeded");
    for(I cap:{caps.domain.maximum_worker_bytes,caps.pnos.maximum_worker_bytes,caps.projection.maximum_per_worker_inventoried_bytes})
        limit(p.per_worker_inventoried_bytes,cap,"Gram domain MP2 child worker cap exceeded");
    for(I cap:{caps.domain.maximum_node_bytes,caps.pnos.maximum_node_bytes,caps.projection.maximum_node_inventoried_bytes})
        limit(p.required_node_memory_bytes,cap,"Gram domain MP2 child node cap exceeded");
    for(I cap:{caps.domain.maximum_control_storage_bytes,caps.pnos.maximum_control_storage_bytes})
        limit(p.control_storage_reservation_bytes,cap,"Gram domain MP2 child enclosing control cap exceeded");
    limit(n,caps.pnos.maximum_common_dimension,"Gram domain MP2 common dimension cap exceeded");
    limit(n,caps.pnos.maximum_generation_dimension,"Gram domain MP2 generation dimension cap exceeded");
    limit(space.retained_output_bytes,caps.projection.maximum_owned_numerical_bytes,"Gram domain MP2 pair-space owned cap exceeded");
    limit(space.numerical_work_units,caps.projection.maximum_work_units,"Gram domain MP2 pair-space work cap exceeded");
    // Populate only the fixed/table fields needed to form the final upper
    // inventory; the second solver plan owns no arrays or PairView table.
    p.solver_upper=solver;
    p.solver_upper=plan_bounded_restricted_pair_mp2_solver_upper(o,n,n,options.solver.maximum_iterations,
        gram_solver_inventory(p,live,mul(p.pair_count,n),p.retained_pair_geometry_upper_bytes,mul(8,mul(p.pair_count,mul(n,n)))));
    solver_caps(p.solver_upper,caps.solver);phase(p,p.solver_upper.per_replica_inventoried_bytes,p.solver_upper.required_node_inventoried_bytes);
    gram_reference(ref,p);return p;
}

Result run_periodic_gaussian_domain_pair_mp2(const HF& hf,const Ref& ref,const BasisSet& ao,const BasisSet& aux,
    const Wannier& w,const Complex* gauges,std::size_t gauge_count,const Basis& basis,const Builder& builder,
    const GramConfig& supplied_config,const GramOptions& supplied_options,const Live& supplied_live,const GramCaps& supplied_caps,
    PeriodicGaussianPairMP2Callback callback,void* callback_context) {
    const auto config=supplied_config;const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_domain_pair_mp2(hf,ref,w,basis,builder,config,options,live,caps);
    const I o=p.occupied_count,n=p.common_virtual_dimension;
    // Authenticate original Gaussian/Wannier/common geometry before even
    // Begin or PairDomainReady can expose a misleading progress event.
    const auto source=gram_source_payload(hf,ref,ao,aux,w,gauges,gauge_count,basis,builder);
    const auto& common_domain=builder.common_domain();const auto& common_space=builder.common_real_space().space();
    GramEvents events{hf,ref,ao,aux,w,gauges,gauge_count,basis,builder,p,source,
        ref.state_handle().get(),hf.context_handle().get(),basis.occupied_indices_data(),basis.f_oo_data(),basis.f_vv_data(),basis.f_ov_data(),
        w.cell_coefficients(0),common_domain.overlap_data(),common_domain.fock_data(),common_space.coefficients_data(),
        common_space.energies_data(),common_space.overlap_eigenvalues_data(),{},callback,callback_context};
    Result result;result.memory_=p;result.state_=ref.state_handle();result.context_=hf.context_handle();
    result.hf_=hf.reference_source_identity_sha256();result.builder_=builder.identity_sha256();
    auto& d=result.diagnostics_;d.source_kind=p.source_kind;d.domain_generated=true;d.minimum_pair_rank=n;d.all_pairs_full_rank=true;
    d.complete_common_finite_torus_basis=o==mul(ref.state().n_kpoints(),ref.state().n_correlated_occupied())
        && n==mul(ref.state().n_kpoints(),ref.state().n_virtual());
    double squared=0;for(I at=0;at<mul(o,n);++at) real::norm_lane(squared,std::abs(real::finite(basis.f_ov_data()[at])));
    d.occupied_virtual_fock_norm_upper_bound=real::sqrt_up(squared);
    if(d.occupied_virtual_fock_norm_upper_bound>options.maximum_occupied_virtual_fock_norm)
        throw std::invalid_argument("Gram domain MP2 occupied-virtual Fock exceeds explicit Brillouin budget");
    events.emit(PeriodicGaussianPairMP2Stage::Begin);
    result.pairs_.reserve(static_cast<std::size_t>(p.pair_count));result.pair_geometry_.reserve(static_cast<std::size_t>(p.pair_count));
    Digest pairs("vibeqc.periodic.gaussian-gram-domain-pair-mp2.spaces");pairs.u64(p.pair_count);
    Digest grams("vibeqc.periodic.gaussian-gram-domain-pair-mp2.consumed-grams");grams.u64(p.pair_count);
    for(I i=0;i<o;++i) for(I j=i;j<o;++j) {
        const I retained=add(d.retained_pair_bytes,d.retained_pair_geometry_bytes);
        auto dc=caps.domain;
        dc.maximum_owned_numerical_bytes=std::min(dc.maximum_owned_numerical_bytes,subtract(p.pair_generation_phase_upper_bytes,retained));
        dc.maximum_worker_bytes=std::min(dc.maximum_worker_bytes,p.per_worker_inventoried_bytes);
        dc.maximum_node_bytes=std::min(dc.maximum_node_bytes,p.required_node_memory_bytes);
        dc.maximum_control_storage_bytes=std::min(dc.maximum_control_storage_bytes,p.control_storage_reservation_bytes);
        dc.maximum_work_units=std::min(dc.maximum_work_units,p.work_units_upper_bound);
        PeriodicGaussianPairDomainBuilderLive dl{add(live.other_live_bytes_per_worker,add(retained,gram_sources_bytes(p))),0,
            live.fixed_backend_margin_bytes_per_worker};
        const auto dp0=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options.domain,dl,dc);
        dl.other_live_control_bytes_per_worker=subtract(p.control_storage_reservation_bytes,dp0.control_storage_reservation_bytes);
        const auto dp=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options.domain,dl,dc);
        phase(p,dp.worker_bytes,dp.required_node_memory_bytes);
        auto geometry=make_periodic_gaussian_pair_domain_geometry(ref,basis,builder,i,j,options.domain,dl,dc);
        const I geometry_bytes=geometry.retained_numerical_bytes(),embedding_bytes=geometry.embedding().memory().output_numerical_bytes;
        events.event.occupied_i=i;events.event.occupied_j=j;events.event.pair_rank=geometry.embedding().memory().pair_dimension;
        events.emit(PeriodicGaussianPairMP2Stage::PairDomainReady);
        auto pc=gram_seed_caps(caps,p,retained,geometry_bytes);
        PeriodicGaussianGramPairPNOLiveInventory pl{add(live.other_live_bytes_per_worker,add(retained,p.borrowed_domain_builder_bytes)),0,
            live.fixed_backend_margin_bytes_per_worker};
        const auto pp0=plan_periodic_gaussian_gram_pair_pnos(hf,ref,w,basis,geometry,config.pnos,options.pnos,pl,pc);
        limit(pp0.control_storage_reservation_bytes,caps.maximum_pno_leaf_control_storage_bytes,"Gram domain MP2 bare PNO control reservation exceeded");
        pl.other_live_control_bytes_per_worker=subtract(p.control_storage_reservation_bytes,pp0.control_storage_reservation_bytes);
        const auto pp=plan_periodic_gaussian_gram_pair_pnos(hf,ref,w,basis,geometry,config.pnos,options.pnos,pl,pc);
        phase(p,pp.worker_bytes,pp.required_node_memory_bytes);
        auto pno=make_periodic_gaussian_gram_pair_pnos(hf,ref,ao,aux,w,gauges,gauge_count,basis,geometry,config.pnos,options.pnos,pl,pc);
        const I m=pno.memory().generation_dimension,rank=pno.diagnostics().retained_dimension;
        grams.u64(i);grams.u64(j);grams.string(pno.identity_sha256());grams.string(pno.payload_sha256());
        grams.string(pno.gram_identity_sha256());grams.string(pno.gram_payload_sha256());
        grams.string(pno.gram_consumed_sources_identity_sha256());grams.string(pno.source_payload_receipt_sha256());
        ++d.completed_gram_builds;
        d.completed_factor_panels=add(d.completed_factor_panels,pno.diagnostics().gram.completed_factor_panels);
        d.completed_tile_calls=add(d.completed_tile_calls,pno.diagnostics().gram.completed_tile_calls);
        const auto storage=periodic_gaussian_direct_gram_pair_space_storage(n,m,rank);
        PeriodicGaussianPairSpaceLiveInventory sl{add(live.other_live_bytes_per_worker,add(retained,
            add(p.borrowed_domain_builder_bytes,add(geometry_bytes,add(gram_sources_bytes(p),
            subtract(p.control_storage_reservation_bytes,storage.fixed_control_storage_bytes)))))),live.fixed_backend_margin_bytes_per_worker};
        auto sc=caps.projection;
        sc.maximum_owned_numerical_bytes=std::min(sc.maximum_owned_numerical_bytes,subtract(p.pair_generation_phase_upper_bytes,add(retained,geometry_bytes)));
        sc.maximum_per_worker_inventoried_bytes=std::min(sc.maximum_per_worker_inventoried_bytes,p.per_worker_inventoried_bytes);
        sc.maximum_node_inventoried_bytes=std::min(sc.maximum_node_inventoried_bytes,p.required_node_memory_bytes);
        sc.maximum_work_units=std::min(sc.maximum_work_units,p.work_units_upper_bound);
        const auto sp=plan_periodic_gaussian_pair_space(ref,basis,pno,options.projection,sl,sc);
        phase(p,sp.per_worker_inventoried_bytes,sp.required_node_memory_bytes);
        auto space=make_periodic_gaussian_pair_space(ref,basis,std::move(pno),options.projection,sl,sc);
        d.retained_pair_bytes=add(d.retained_pair_bytes,space.memory().retained_output_bytes);
        d.generation_dimension_sum=add(d.generation_dimension_sum,m);
        d.retained_generation_coefficient_bytes=add(d.retained_generation_coefficient_bytes,mul(8,mul(m,rank)));
        d.minimum_pair_rank=std::min(d.minimum_pair_rank,rank);d.maximum_pair_rank=std::max(d.maximum_pair_rank,rank);
        d.zero_rank_pairs+=!rank;d.all_pairs_full_rank=d.all_pairs_full_rank && rank==n;
        d.maximum_diagonal_integral_projection_norm=std::max(d.maximum_diagonal_integral_projection_norm,
            space.diagnostics().diagonal_symmetry_projection_frobenius_bound);
        if(i==j) {d.diagonal_generation_dimension_sum=add(d.diagonal_generation_dimension_sum,m);
            d.retained_generation_embedding_bytes=add(d.retained_generation_embedding_bytes,embedding_bytes);}
        d.retained_pair_geometry_bytes=add(d.retained_pair_geometry_bytes,geometry_bytes);
        d.retained_pair_geometry_control_bytes=add(d.retained_pair_geometry_control_bytes,geometry.retained_control_storage_bytes());
        if(d.retained_pair_bytes>p.retained_pair_output_upper_bytes || d.retained_generation_embedding_bytes>p.retained_generation_embedding_upper_bytes
            || d.retained_pair_geometry_bytes>p.retained_pair_geometry_upper_bytes || d.retained_pair_geometry_control_bytes>p.retained_pair_geometry_control_upper_bytes
            || d.completed_gram_builds>p.gram_builds_upper_bound || d.completed_factor_panels>p.factor_panels_upper_bound || d.completed_tile_calls>p.tile_calls_upper_bound)
            throw std::logic_error("Gram domain MP2 exact retained/source census exceeds admission");
        pairs.u64(i);pairs.u64(j);pairs.string(space.identity_sha256());
        result.pairs_.push_back(std::move(space));result.pair_geometry_.push_back(std::move(geometry));++d.completed_pairs;
        events.event.completed_pairs=d.completed_pairs;events.event.pair_rank=rank;events.emit(PeriodicGaussianPairMP2Stage::PairComplete);
    }
    result.pairs_sha_=pairs.finish();result.gram_sources_=grams.finish();
    std::vector<PairView> views;views.reserve(static_cast<std::size_t>(p.pair_count));
    for(const auto& space:result.pairs_) {
        const auto& g=space.exchange_integrals();const auto& eps=space.energies();const auto& c=space.coefficients();
        views.push_back({space.memory().retained_dimension,{g.data(),g.size()},{eps.data(),eps.size()},{c.data(),c.size()}});
    }
    const BoundedRestrictedPairMP2SolverInput input{o,n,{basis.f_oo_data(),static_cast<std::size_t>(mul(o,o))},views.data(),views.size()};
    const auto inventory=gram_solver_inventory(p,live,d.generation_dimension_sum,d.retained_pair_geometry_bytes,d.retained_generation_coefficient_bytes);
    auto solver_cap=caps.solver;
    solver_cap.maximum_owned_numerical_bytes=std::min(solver_cap.maximum_owned_numerical_bytes,
        subtract(p.solver_phase_owned_upper_bytes,add(d.retained_pair_bytes,d.retained_pair_geometry_bytes)));
    solver_cap.maximum_per_replica_inventoried_bytes=std::min(solver_cap.maximum_per_replica_inventoried_bytes,p.per_worker_inventoried_bytes);
    solver_cap.maximum_node_inventoried_bytes=std::min(solver_cap.maximum_node_inventoried_bytes,p.required_node_memory_bytes);
    solver_cap.maximum_work_units=std::min(solver_cap.maximum_work_units,p.work_units_upper_bound);
    result.solver_.emplace(bounded_restricted_pair_mp2_solve(input,options.solver,inventory,solver_cap,&GramEvents::solver,&events));
    Digest identity("vibeqc.periodic.gaussian-gram-domain-pair-mp2.identity");
    for(const auto* s:std::array<const std::string*,10>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &basis.identity_sha256(),&result.hf_,&result.builder_,&result.pairs_sha_,&result.gram_sources_,&source,
        &result.solver_->input_identity_sha256(),&result.solver_->payload_sha256()}) identity.string(*s);
    for(const auto& g:result.pair_geometry_) identity.string(g.identity_sha256());
    identity.u64(static_cast<I>(p.source_kind));identity.u64(d.generation_dimension_sum);identity.u64(d.diagonal_generation_dimension_sum);
    identity.u64(d.retained_generation_coefficient_bytes);identity.u64(d.retained_pair_geometry_bytes);
    identity.u64(d.retained_pair_geometry_control_bytes);identity.real(d.occupied_virtual_fock_norm_upper_bound);
    identity.real(options.maximum_occupied_virtual_fock_norm);identity.u64(d.complete_common_finite_torus_basis);identity.u64(d.all_pairs_full_rank);
    identity.string("DirectPAOGram;actual-HF-native-pair-domains;Eq39-local-density;coupled-MP2;original-geometries-retained;"
        "all-placed-unordered-pairs;per-cell-divide-once;no-common-factor-provider;no-weak-pair-correction;not-production-DLPNO");
    result.identity_=identity.finish();events.emit(PeriodicGaussianPairMP2Stage::Finished);
    d.completed_progress_callbacks=events.event.callback_count;return result;
}
} // namespace vibeqc
