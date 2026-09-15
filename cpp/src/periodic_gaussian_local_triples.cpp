#include "vibeqc/periodic_gaussian_local_triples.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian local triples forbid fast/finite-only math"
#endif

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
using U=std::uint64_t;
using real::add;using real::mul;using real::Digest;
using Plan=PeriodicGaussianLocalTriplesPlan;
using Result=PeriodicGaussianLocalTriplesResult;
using Options=PeriodicGaussianLocalTriplesOptions;
using Live=PeriodicGaussianLocalTriplesLiveInventory;
using Caps=PeriodicGaussianLocalTriplesCaps;
using Reader=BoundedRestrictedPairCCSDAmplitudes;
using ReaderPlan=BoundedRestrictedPairCCSDAmplitudesMemoryPlan;
using ReaderInventory=BoundedRestrictedPairCCSDAmplitudesInventory;
using ReaderCaps=BoundedRestrictedPairCCSDAmplitudesCaps;
using SolverPlan=BoundedRestrictedLocalTriplesSolverMemoryPlan;
using SolverInventory=BoundedRestrictedLocalTriplesSolverInventory;
using MomentPlan=BoundedRestrictedLocalTriplesMomentsMemoryPlan;
using MomentInventory=BoundedRestrictedLocalTriplesMomentsInventory;
using MomentProvider=BoundedRestrictedLocalTriplesMomentProvider;
using IntegralProvider=BoundedRestrictedCCSDIntegralProvider;
using SingleView=BoundedRestrictedPairCCSDSinglesView;
using PairView=BoundedRestrictedPairCCSDPairView;
using TripleView=BoundedRestrictedLocalTriplesSpaceView;
static_assert(sizeof(double)==8 && FLT_EVAL_METHOD==0,
    "Gaussian local triples require binary64 evaluation");
constexpr char policy[]="actual-finite-Gaussian;converged-CCSD-independent-singles-and-pair-PNOs;"
    "Riplinger2013-local-TNO-amplitude-projection-BEFORE-moments;Guo2018-all-occupied-couplings;"
    "all-unordered-multisets;no-weak-screening;explicit-original-Fov-Brillouin-projection;"
    "original-Foo-Fvv;no-T0-fallback;not-production-DLPNO;no-infinite-source-error-certificate";
U subtract(U a,U b) {
    if (b>a) throw std::logic_error("Gaussian local triples owner subtraction is inconsistent");
    return a-b;
}
U cube(U a) { return mul(mul(a,a),a); }
U pair_count(U o) { return o%2 ? mul(o,add(o,1)/2) : mul(o/2,add(o,1)); }
U triple_count(U o) {
    std::array<U,3> a{o,add(o,1),add(o,2)};
    for (U divisor:{2U,3U}) for (U& x:a) if (x%divisor==0) {x/=divisor;break;}
    return mul(mul(a[0],a[1]),a[2]);
}
void limit(U value,U cap,const char* text) { if (value>cap) throw std::length_error(text); }
void finite_nonnegative(double x) {
    if (!std::isfinite(x) || x<0) throw std::invalid_argument("Gaussian local triples explicit budgets must be finite nonnegative");
}
void options_valid(const Options& options,const Live& live,const PeriodicGaussianRealLocalProvider& provider) {
    finite_nonnegative(options.maximum_brillouin_projection_norm);
    finite_nonnegative(options.moments.amplitude_symmetry_tolerance);
    if (!std::isfinite(options.moments.coefficient_orthogonality_tolerance)
        || options.moments.coefficient_orthogonality_tolerance<=0
        || options.moments.coefficient_orthogonality_tolerance>=1
        || !live.fixed_backend_margin_bytes_per_worker
        || !options.moments.maximum_integral_work_units_per_call
        || options.moments.maximum_integral_work_units_per_call!=provider.memory().scalar_work_units)
        throw std::invalid_argument("Gaussian local triples require explicit Gram/backend controls and actual native integral-work bound");
}
struct Sources {
    const PeriodicCorrelationAdmittedReference& ref;
    const PeriodicCorrelationRealLocalBasis& basis;
    const PeriodicGaussianRealLocalProvider& provider;
    const PeriodicGaussianPairMP2Result& mp2;
    const PeriodicGaussianPairCCSDResult& ccsd;
    const PeriodicGaussianTripleSpaces& spaces;
};
void validate_sources(const Sources& s) {
    validate_periodic_gaussian_triple_spaces(s.ref,s.basis,s.provider,s.mp2,s.ccsd,s.spaces);
}
U reader_control(const Plan& p) { return add(p.reader_table_bytes,p.reader_upper.fixed_control_storage_bytes); }
ReaderInventory reader_inventory(const Plan& p,const Live& live) {
    return {p.replicas_per_node,p.reference_base_node_bytes,
        add(live.other_live_numerical_bytes_per_worker,
            add(subtract(p.complete_borrowed_owner_numerical_bytes,p.reader_borrowed_numerical_bytes),
                add(p.projected_zero_fov_bytes,subtract(p.control_storage_reservation_bytes,reader_control(p))))),
        live.fixed_backend_margin_bytes_per_worker};
}
SolverInventory solver_inventory(const Plan& p,const Live& live) {
    return {p.replicas_per_node,p.reference_base_node_bytes,
        add(live.other_live_numerical_bytes_per_worker,
            add(subtract(p.complete_borrowed_owner_numerical_bytes,p.solver_borrowed_numerical_bytes),
                add(p.projected_zero_fov_bytes,subtract(p.control_storage_reservation_bytes,p.solver_upper.control_storage_reservation_bytes)))),
        live.fixed_backend_margin_bytes_per_worker};
}
MomentInventory moment_inventory(const Plan& p,const Live& live,U rank,U integral_bytes,U controls) {
    const U roles=add(p.reader_borrowed_numerical_bytes,add(mul(8,mul(p.common_virtual_dimension,rank)),integral_bytes));
    return {p.replicas_per_node,p.reference_base_node_bytes,
        add(live.other_live_numerical_bytes_per_worker,
            add(subtract(p.complete_borrowed_owner_numerical_bytes,roles),
                add(p.solver_upper.peak_owned_numerical_bytes,subtract(p.control_storage_reservation_bytes,controls)))),
        live.fixed_backend_margin_bytes_per_worker};
}
ReaderCaps reader_caps(const ReaderPlan& p) {
    return {p.n_occupied,p.common_virtual_dimension,p.pair_count,p.maximum_singles_rank,p.maximum_pair_rank,
        p.borrowed_table_bytes,p.borrowed_numerical_bytes,p.per_replica_inventoried_bytes,p.required_node_inventoried_bytes,
        p.validation_work_units,p.maximum_singles_work_units_per_query,p.maximum_doubles_work_units_per_query};
}
void solver_caps(const SolverPlan& p,const BoundedRestrictedLocalTriplesSolverCaps& c) {
    limit(p.n_occupied,c.maximum_occupied_count,"Gaussian local triples solver occupied cap");
    limit(p.common_virtual_dimension,c.maximum_common_virtual_dimension,"Gaussian local triples solver common cap");
    limit(p.triple_count,c.maximum_triple_count,"Gaussian local triples solver triple cap");
    limit(p.maximum_rank,c.maximum_rank,"Gaussian local triples solver rank cap");
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"Gaussian local triples solver owned cap");
    limit(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"Gaussian local triples solver worker cap");
    limit(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"Gaussian local triples solver node cap");
    limit(p.moment_visits_upper_bound,c.maximum_moment_visits,"Gaussian local triples solver moment visit cap");
    limit(p.work_units_upper_bound,c.maximum_work_units,"Gaussian local triples solver work cap");
}
void moment_caps(const MomentPlan& p,const BoundedRestrictedLocalTriplesMomentsCaps& c) {
    limit(p.n_occupied,c.maximum_occupied_count,"Gaussian local triples moment occupied cap");
    limit(p.common_virtual_dimension,c.maximum_common_virtual_dimension,"Gaussian local triples moment common cap");
    limit(p.rank,c.maximum_rank,"Gaussian local triples moment rank cap");
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"Gaussian local triples moment owned cap");
    limit(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"Gaussian local triples moment worker cap");
    limit(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"Gaussian local triples moment node cap");
    limit(p.common_singles_calls,c.maximum_common_singles_calls,"Gaussian local triples moment singles cap");
    limit(p.common_doubles_calls,c.maximum_common_doubles_calls,"Gaussian local triples moment doubles cap");
    limit(p.transformed_integral_calls,c.maximum_transformed_integral_calls,"Gaussian local triples transformed integral cap");
    limit(p.common_integral_calls,c.maximum_common_integral_calls,"Gaussian local triples common integral visit cap");
    limit(p.work_units_upper_bound,c.maximum_work_units,"Gaussian local triples moment work cap");
}
U visit_work(const Plan& p) {
    return p.moments_required ? add(p.moments_upper.work_units_upper_bound,
        add(32768,mul(128,p.moments_upper.common_integral_calls))) : 1;
}
MomentProvider moment_provider_metadata(const Plan& p) {
    MomentProvider provider;
    provider.maximum_transient_numerical_bytes=p.maximum_moment_transient_numerical_bytes;
    provider.maximum_work_units_per_visit=visit_work(p);
    return provider;
}
void assert_worker(const Plan& p,U actual,U padding) {
    const U expected=add(subtract(p.per_worker_inventoried_bytes,p.rank_padding_reservation_bytes),padding);
    if (actual!=expected) throw std::logic_error("Gaussian local triples nested owner/rank-padding identity differs");
}
void scalar_counts(const Sources& s,const Plan& p) {
    validate_sources(s);
    U total_rank=0,total_cube=0,maximum=0;
    for (U i=0;i<p.occupied_count;++i) for (U j=i;j<p.occupied_count;++j) for (U k=j;k<p.occupied_count;++k) {
        const auto& g=s.spaces.triple(i,j,k);
        total_rank=add(total_rank,g.retained_rank);total_cube=add(total_cube,cube(g.retained_rank));
        maximum=std::max(maximum,g.retained_rank);
    }
    if (total_rank!=p.total_retained_rank || maximum!=p.maximum_retained_rank
        || total_cube!=p.total_amplitude_elements
        || s.ccsd.solver().memory().amplitudes.borrowed_numerical_bytes!=p.reader_borrowed_numerical_bytes)
        throw std::invalid_argument("Gaussian local triples actual owner rank census changed");
}
void options_wire(Digest& h,const Options& options) {
    for (double x:{options.maximum_brillouin_projection_norm,options.moments.coefficient_orthogonality_tolerance,
        options.moments.amplitude_symmetry_tolerance,options.solver.denominator_floor,
        options.solver.residual_tolerance,options.solver.energy_tolerance,
        options.solver.coefficient_orthogonality_tolerance,options.solver.maximum_projected_fock_error,
        options.solver.maximum_repeated_moment_defect_norm,options.solver.maximum_repeated_update_defect_norm}) h.real(x);
    h.u64(options.moments.maximum_integral_work_units_per_call);h.u64(options.solver.maximum_iterations);
}
}  // namespace

Plan plan_periodic_gaussian_local_triples(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& mp2,const PeriodicGaussianPairCCSDResult& ccsd,
    const PeriodicGaussianTripleSpaces& spaces,const Options& options,const Live& live,const Caps& caps) {
    real::float_environment();options_valid(options,live,provider);
    Plan p;const U o=p.occupied_count=basis.memory().occupied_count,n=p.common_virtual_dimension=basis.memory().virtual_count;
    if (!o || !n) throw std::invalid_argument("Gaussian local triples dimensions must be positive");
    const U P=p.pair_count=pair_count(o),Q=p.triple_count=triple_count(o),nn=mul(n,n);
    limit(P,caps.maximum_pair_count,"Gaussian local triples pair count cap");
    limit(Q,caps.maximum_triple_count,"Gaussian local triples triple count cap");
    const auto& gd=spaces.diagnostics();const auto& gp=spaces.memory();
    const auto& cc=ccsd.solver().memory();
    const U R=p.maximum_retained_rank=gd.maximum_retained_rank;
    p.total_retained_rank=gd.total_retained_rank;p.moments_required=R!=0;
    if (R>n || p.total_retained_rank>mul(Q,R) || gd.total_union_rank>mul(Q,n)
        || gp.triple_count!=Q || gp.pair_count!=P || gp.occupied_count!=o || gp.common_virtual_dimension!=n
        || cc.n_occupied!=o || cc.common_virtual_dimension!=n || cc.pair_count!=P)
        throw std::invalid_argument("Gaussian local triples immutable dimension/rank metadata differs");
    const U M=add(mp2.diagnostics().retained_pair_geometry_bytes,
        add(mp2.diagnostics().retained_pair_bytes,mp2.solver().memory().output_numerical_bytes));
    const U C=add(ccsd.diagnostics().retained_singles_bytes,cc.output_numerical_bytes);
    const U T=mul(8,add(mul(add(n,1),p.total_retained_rank),gd.total_union_rank));
    if (T!=gd.retained_numerical_bytes)
        throw std::logic_error("Gaussian local triples retained triple byte census differs");
    p.complete_borrowed_owner_numerical_bytes=add(add(basis.memory().retained_output_bytes,provider.memory().retained_row_bytes),add(add(M,C),T));
    p.complete_borrowed_owner_control_bytes=sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationRealLocalBasis)
        +sizeof(PeriodicGaussianRealLocalProvider)+sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianTripleSpaces)+10U*65U;
    p.complete_borrowed_owner_control_bytes=add(p.complete_borrowed_owner_control_bytes,
        add(add(gp.borrowed_mp2_control_bytes,gp.borrowed_ccsd_control_bytes),add(gp.retained_triple_object_bytes,gp.retained_triple_seal_bytes)));
    p.projected_zero_fov_bytes=mul(8,mul(o,n));
    p.reader_borrowed_numerical_bytes=cc.amplitudes.borrowed_numerical_bytes;
    p.solver_borrowed_numerical_bytes=mul(8,add(add(mul(o,o),nn),mul(add(n,1),p.total_retained_rank)));
    const auto original=provider.provider().integral_provider(basis);
    if (original.retained_numerical_bytes!=add(provider.memory().retained_row_bytes,mul(16,o))
        || original.maximum_transient_numerical_bytes)
        throw std::logic_error("Gaussian local triples actual integral provider inventory differs");
    const auto& dimensions=ref.dimensions();const auto& budget=ref.budget();
    p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);
    p.reference_base_node_bytes=add(add(dimensions.external_bytes,dimensions.shared_bytes),
        mul(budget.mpi_ranks,add(dimensions.per_rank_bytes,dimensions.localization_window_bytes_per_rank)));
    const ReaderInventory subset{1,0,0,live.fixed_backend_margin_bytes_per_worker};
    p.reader_upper=plan_bounded_restricted_pair_ccsd_amplitudes_upper(o,n,cc.maximum_singles_rank,cc.maximum_pair_rank,subset);
    p.reader_table_bytes=p.reader_upper.borrowed_table_bytes;
    p.reader_borrowed_rank_padding_bytes=subtract(p.reader_upper.borrowed_numerical_bytes,p.reader_borrowed_numerical_bytes);
    if (R) {
        const MomentInventory minv{1,0,0,live.fixed_backend_margin_bytes_per_worker};
        p.moments_upper=plan_bounded_restricted_local_triples_moments_upper(o,n,R,cc.maximum_singles_rank,
            cc.maximum_pair_rank,original.retained_numerical_bytes,0,options.moments,minv);
        p.maximum_moment_transient_numerical_bytes=p.moments_upper.peak_owned_numerical_bytes;
        if (p.maximum_moment_transient_numerical_bytes!=add(mul(8,mul(o,R)),mul(16,cube(R))))
            throw std::logic_error("Gaussian local triples moment workspace formula changed");
    }
    const SolverInventory sinv{1,0,0,live.fixed_backend_margin_bytes_per_worker};
    const auto moment_metadata=moment_provider_metadata(p);
    p.solver_upper=plan_bounded_restricted_local_triples_solver_upper(o,n,R,moment_metadata,options.solver,sinv);
    p.triple_table_bytes=p.solver_upper.borrowed_space_table_bytes;
    p.solver_borrowed_rank_padding_bytes=subtract(p.solver_upper.borrowed_numerical_bytes,p.solver_borrowed_numerical_bytes);
    if (p.solver_borrowed_rank_padding_bytes!=mul(8,mul(add(n,1),subtract(mul(Q,R),p.total_retained_rank))))
        throw std::logic_error("Gaussian local triples rank padding formula changed");
    p.rank_padding_reservation_bytes=std::max(p.reader_borrowed_rank_padding_bytes,p.solver_borrowed_rank_padding_bytes);
    p.peak_owned_numerical_bytes=add(p.projected_zero_fov_bytes,
        add(p.solver_upper.peak_owned_numerical_bytes,p.maximum_moment_transient_numerical_bytes));
    const U incremental_moment_control=R ? subtract(p.moments_upper.control_storage_reservation_bytes,reader_control(p)) : 0;
    const U bridge_control=131072U+3U*sizeof(Plan)+2U*sizeof(Result)+2U*sizeof(Options)+2U*sizeof(Live)+2U*sizeof(Caps)
        +4U*sizeof(Reader)+4U*sizeof(BoundedRestrictedLocalTriplesMomentsResult)+4U*sizeof(Digest)+16U*sizeof(std::vector<double>)+4U*65U;
    p.control_storage_reservation_bytes=add(add(p.complete_borrowed_owner_control_bytes,bridge_control),
        add(reader_control(p),add(p.solver_upper.control_storage_reservation_bytes,
            add(incremental_moment_control,live.other_live_control_bytes_per_worker))));
    p.per_worker_inventoried_bytes=add(p.complete_borrowed_owner_numerical_bytes,
        add(p.peak_owned_numerical_bytes,add(p.rank_padding_reservation_bytes,
            add(p.control_storage_reservation_bytes,add(live.other_live_numerical_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker)))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    p.common_integral_calls_upper_bound=R ? mul(p.solver_upper.moment_visits_upper_bound,p.moments_upper.common_integral_calls) : 0;
    p.progress_callback_upper_bound=add(options.solver.maximum_iterations,2);
    const U checks=add(mul(2,p.progress_callback_upper_bound),8);
    const U one_check=add(mul(32768,add(add(add(add(P,o),Q),n),1)),p.reader_upper.validation_work_units);
    p.driver_work_units=add(mul(checks,one_check),add(mul(4,p.reader_upper.validation_work_units),mul(4096,add(mul(o,n),1))));
    p.work_units_upper_bound=add(p.driver_work_units,p.solver_upper.work_units_upper_bound);
    for (U bytes:{p.peak_owned_numerical_bytes,p.control_storage_reservation_bytes,p.complete_borrowed_owner_numerical_bytes,
        p.reader_table_bytes,p.triple_table_bytes,p.rank_padding_reservation_bytes}) real::extent(bytes);
    if (o>std::vector<SingleView>().max_size() || P>std::vector<PairView>().max_size()
        || Q>std::vector<TripleView>().max_size() || mul(o,n)>std::vector<double>().max_size())
        throw std::length_error("Gaussian local triples native table extent exceeded");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian local triples outer owned cap");
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes_per_worker,"Gaussian local triples outer control cap");
    limit(p.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"Gaussian local triples outer worker cap");
    limit(p.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"Gaussian local triples outer node cap");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian local triples admitted reference node cap");
    limit(p.common_integral_calls_upper_bound,caps.maximum_common_integral_calls,"Gaussian local triples aggregate integral cap");
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gaussian local triples progress cap");
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian local triples outer work cap");
    // Full owner and per-triple metadata validation is finally admitted.
    const Sources sources{ref,basis,provider,mp2,ccsd,spaces};validate_sources(sources);
    for (U i=0;i<o;++i) for (U j=i;j<o;++j) for (U k=j;k<o;++k)
        p.total_amplitude_elements=add(p.total_amplitude_elements,cube(spaces.triple(i,j,k).retained_rank));
    p.reader_upper=plan_bounded_restricted_pair_ccsd_amplitudes_upper(o,n,cc.maximum_singles_rank,cc.maximum_pair_rank,reader_inventory(p,live));
    p.solver_upper=plan_bounded_restricted_local_triples_solver_upper(o,n,R,moment_metadata,options.solver,solver_inventory(p,live));
    solver_caps(p.solver_upper,caps.solver);
    assert_worker(p,p.solver_upper.per_replica_inventoried_bytes,p.solver_borrowed_rank_padding_bytes);
    if (R) {
        const auto inv=moment_inventory(p,live,R,original.retained_numerical_bytes,p.moments_upper.control_storage_reservation_bytes);
        p.moments_upper=plan_bounded_restricted_local_triples_moments_upper(o,n,R,cc.maximum_singles_rank,
            cc.maximum_pair_rank,original.retained_numerical_bytes,0,options.moments,inv);
        moment_caps(p.moments_upper,caps.moments);
        assert_worker(p,p.moments_upper.per_replica_inventoried_bytes,p.reader_borrowed_rank_padding_bytes);
    }
    scalar_counts(sources,p);
    return p;
}

namespace {
struct CountedIntegrals {
    IntegralProvider original;
    U calls=0,maximum_calls=0;
    static double value(U p,U q,U r,U s,void* context) {
        auto& self=*static_cast<CountedIntegrals*>(context);
        if (self.calls>=self.maximum_calls)
            throw std::length_error("Gaussian local triples aggregate common integral calls exhausted");
        ++self.calls; // charge BEFORE the actual native provider invocation
        const double result=self.original.value(p,q,r,s,self.original.context);
        real::float_environment();return real::finite(result);
    }
    IntegralProvider provider() {
        return {&CountedIntegrals::value,this,original.retained_numerical_bytes,original.maximum_transient_numerical_bytes};
    }
};
struct MomentContext {
    const Sources& sources;const Plan& plan;const Options& options;const Live& live;const Caps& caps;
    const Reader& reader;const std::vector<double>& zero_fov;
    CountedIntegrals& integrals;
    U snapshot=0,visits=0;
    Digest receipts{"vibeqc.periodic.gaussian-local-triples.consumed-moments"};
    static void visit(const BoundedRestrictedLocalTriplesMomentRequest& request,
        BoundedRestrictedLocalTriplesMomentReceiver receiver,void* receiver_context,void* context) {
        auto& self=*static_cast<MomentContext*>(context);
        if (!receiver || !self.plan.moments_required || !request.rank || request.ccsd_snapshot_id!=self.snapshot
            || self.visits>=self.plan.solver_upper.moment_visits_upper_bound)
            throw std::invalid_argument("Gaussian local triples moment request/snapshot/census is invalid");
        const auto& tuple=request.occupied;
        const auto& geometry=self.sources.spaces.triple(tuple[0],tuple[1],tuple[2]);
        if (request.rank!=geometry.retained_rank)
            throw std::invalid_argument("Gaussian local triples request does not match its exact native TNO frame");
        const auto& C=geometry.semicanonical.coefficients;
        const BoundedRestrictedLocalTriplesMomentsInput input{request.rank,tuple,self.snapshot,
            {C.data(),C.size()},{self.zero_fov.data(),self.zero_fov.size()}};
        const auto common=self.integrals.provider();
        const auto inv=moment_inventory(self.plan,self.live,request.rank,common.retained_numerical_bytes,
            self.plan.moments_upper.control_storage_reservation_bytes);
        const auto p=plan_bounded_restricted_local_triples_moments(input,self.reader,common,self.options.moments,inv);
        moment_caps(p,self.caps.moments);
        limit(p.required_node_inventoried_bytes,self.plan.required_node_memory_bytes,"Gaussian local triples moment exceeds enclosing node cap");
        limit(p.peak_owned_numerical_bytes,self.plan.maximum_moment_transient_numerical_bytes,
            "Gaussian local triples moment exceeds declared provider transient");
        limit(add(p.work_units_upper_bound,add(32768,mul(128,p.common_integral_calls))),visit_work(self.plan),
            "Gaussian local triples moment exceeds declared per-visit work");
        const U previous_calls=self.integrals.calls;++self.visits;
        auto result=bounded_restricted_local_triples_moments(input,self.reader,common,self.options.moments,inv,self.caps.moments);
        if (result.common_integral_calls()!=subtract(self.integrals.calls,previous_calls)
            || result.common_integral_calls()!=p.common_integral_calls)
            throw std::logic_error("Gaussian local triples actual common integral census differs from native moment receipt");
        const auto& moment=result.moments().moments;
        if (moment.occupied!=tuple || moment.amplitude_snapshot_id!=self.snapshot
            || moment.connected.size()!=cube(request.rank) || moment.singles.size()!=cube(request.rank))
            throw std::logic_error("Gaussian local triples native moment payload/request differs");
        for (U label:tuple) self.receipts.u64(label);
        self.receipts.u64(request.rank);self.receipts.u64(self.snapshot);
        self.receipts.string(result.reader_snapshot_identity_sha256());
        self.receipts.string(result.input_identity_sha256());
        self.receipts.string(result.consumed_integral_receipt_sha256());
        self.receipts.string(result.payload_identity_sha256());
        self.receipts.u64(result.common_integral_calls());
        // W/U owner remains alive throughout the one synchronous receiver.
        // Receiver failures are not swallowed. No global moment cube/cache.
        const BoundedRestrictedLocalTriplesMomentView view{request,
            {moment.connected.data(),moment.connected.size()},{moment.singles.data(),moment.singles.size()}};
        receiver(view,receiver_context);
        real::float_environment();
    }
};
struct Events {
    const Sources& sources;const Plan& plan;const Reader& reader;const MomentContext& moments;
    const CountedIntegrals& integrals;
    PeriodicGaussianLocalTriplesCallback callback=nullptr;void* context=nullptr;
    PeriodicGaussianLocalTriplesProgress event;
    std::array<char,64> ccsd{},spaces{},provider{},reader_sha{};
    // The basis has a native move-assignment operator. Equal-content owner
    // replacement must still abort before the solver touches old F pointers.
    std::array<const void*,4> basis_views{};
    void pin() {
        const auto snapshot=reader.snapshot_identity_sha256();
        if (snapshot.size()!=64 || sources.ccsd.identity_sha256().size()!=64
            || sources.spaces.identity_sha256().size()!=64 || sources.provider.identity_sha256().size()!=64)
            throw std::logic_error("Gaussian local triples source receipt extent differs");
        std::copy(snapshot.begin(),snapshot.end(),reader_sha.begin());
        std::copy(sources.ccsd.identity_sha256().begin(),sources.ccsd.identity_sha256().end(),ccsd.begin());
        std::copy(sources.spaces.identity_sha256().begin(),sources.spaces.identity_sha256().end(),spaces.begin());
        std::copy(sources.provider.identity_sha256().begin(),sources.provider.identity_sha256().end(),provider.begin());
        basis_views={sources.basis.occupied_indices_data(),sources.basis.f_oo_data(),
            sources.basis.f_vv_data(),sources.basis.f_ov_data()};
    }
    void check() const {
        real::float_environment();
        // A progress callback may move a native owner. Check native ownership
        // BEFORE dereferencing the reader's still-borrowed C/T pointers.
        scalar_counts(sources,plan);
        if (basis_views!=std::array<const void*,4>{sources.basis.occupied_indices_data(),sources.basis.f_oo_data(),
            sources.basis.f_vv_data(),sources.basis.f_ov_data()})
            throw std::invalid_argument("Gaussian local triples original basis storage changed across progress");
        if (!std::equal(ccsd.begin(),ccsd.end(),sources.ccsd.identity_sha256().begin())
            || !std::equal(spaces.begin(),spaces.end(),sources.spaces.identity_sha256().begin())
            || !std::equal(provider.begin(),provider.end(),sources.provider.identity_sha256().begin()))
            throw std::invalid_argument("Gaussian local triples physical owner changed across progress");
        reader.validate_immutable_snapshot();const auto snapshot=reader.snapshot_identity_sha256();
        if (snapshot.size()!=64 || !std::equal(reader_sha.begin(),reader_sha.end(),snapshot.begin()))
            throw std::invalid_argument("Gaussian local triples CCSD reader snapshot changed");
    }
    void emit(PeriodicGaussianLocalTriplesStage stage) {
        check();
        if (event.callback_count>=plan.progress_callback_upper_bound)
            throw std::length_error("Gaussian local triples progress count exceeds admission");
        event.stage=stage;++event.callback_count;event.common_integral_calls=integrals.calls;
        if (callback) callback(event,context);
        check();
    }
    static void solver(const BoundedRestrictedLocalTriplesSolverProgress& progress,void* context) {
        auto& self=*static_cast<Events*>(context);
        if (progress.moment_visits!=self.moments.visits)
            throw std::logic_error("Gaussian local triples solver/producer visit census differs");
        self.event.solver=progress;self.emit(PeriodicGaussianLocalTriplesStage::Solver);
    }
};
}  // namespace

const BoundedRestrictedLocalTriplesSolverResult& Result::solver() const {
    if (!state_ || !context_ || !solver_ || ccsd_.size()!=64 || spaces_.size()!=64)
        throw std::logic_error("Gaussian local triples result is moved or incomplete");
    return *solver_;
}
bool Result::converged() const { return solver().final_snapshot().converged; }
bool Result::periodic_energy_per_cell() const {
    return diagnostics_.complete_common_finite_torus_basis && converged();
}
double Result::triples_energy_per_cell() const {
    if (!periodic_energy_per_cell())
        throw std::logic_error("Gaussian local triples per-cell energy requires complete common basis and convergence");
    return real::finite(solver().final_snapshot().triples_energy/static_cast<double>(state_->n_kpoints()));
}
double Result::correlation_energy_per_cell() const {
    const double triples=triples_energy_per_cell();
    return real::finite(real::finite(ccsd_correlation_energy_/static_cast<double>(state_->n_kpoints()))+triples);
}
double Result::total_energy_per_cell() const {
    return real::finite(correlation_energy_per_cell()+state_->reference_energy_per_cell());
}

Result run_periodic_gaussian_local_triples(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& mp2,const PeriodicGaussianPairCCSDResult& ccsd,
    const PeriodicGaussianTripleSpaces& spaces,const Options& supplied_options,const Live& supplied_live,
    const Caps& supplied_caps,PeriodicGaussianLocalTriplesCallback callback,void* callback_context) {
    const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_local_triples(ref,basis,provider,mp2,ccsd,spaces,options,live,caps);
    const Sources sources{ref,basis,provider,mp2,ccsd,spaces};scalar_counts(sources,p);
    const U o=p.occupied_count,n=p.common_virtual_dimension,P=p.pair_count,Q=p.triple_count;
    // The ORIGINAL physical block is norm-gated before allocating/passing a
    // separate zero operator. This never modifies the CCSD/reference Fock.
    double fov_squared=0;
    const double* original_fov=basis.f_ov_data();
    for (U x=0;x<mul(o,n);++x) real::norm_lane(fov_squared,std::abs(real::finite(original_fov[x])));
    const double fov_norm=real::sqrt_up(fov_squared);
    if (fov_norm>options.maximum_brillouin_projection_norm)
        throw std::invalid_argument("Gaussian local triples original Fov exceeds explicit Brillouin projection budget");
    std::vector<double> zero_fov(static_cast<std::size_t>(mul(o,n)),0.0);
    std::vector<SingleView> singles(static_cast<std::size_t>(o));
    std::vector<PairView> pairs(static_cast<std::size_t>(P));
    std::vector<TripleView> triples(static_cast<std::size_t>(Q));
    U pair_at=0,triple_at=0;
    for (U i=0;i<o;++i) {
        const auto& C=ccsd.singles_frame(i).coefficients();const auto t=ccsd.solver().stored_singles_view(i);
        singles[i]={ccsd.solver().singles_rank(i),{C.data(),C.size()},t};
        for (U j=i;j<o;++j) {
            const auto& pair_C=mp2.pair(i,j).coefficients();const auto T=ccsd.solver().stored_pair_view(i,j);
            pairs[pair_at++]={ccsd.solver().pair_rank(i,j),{pair_C.data(),pair_C.size()},T};
            for (U k=j;k<o;++k) {
                const auto& geometry=spaces.triple(i,j,k);
                const auto& tc=geometry.semicanonical.coefficients;const auto& eps=geometry.semicanonical.energies;
                triples[triple_at++]={geometry.retained_rank,{tc.data(),tc.size()},{eps.data(),eps.size()}};
            }
        }
    }
    if (pair_at!=P || triple_at!=Q)
        throw std::logic_error("Gaussian local triples descriptor census differs");
    const BoundedRestrictedPairCCSDAmplitudesInput reader_input{o,n,singles.data(),singles.size(),pairs.data(),pairs.size()};
    const BoundedRestrictedPairCCSDAmplitudesOptions reader_options{options.moments.coefficient_orthogonality_tolerance};
    const auto rinv=reader_inventory(p,live);const auto rcaps=reader_caps(p.reader_upper);
    const auto rplan=plan_bounded_restricted_pair_ccsd_amplitudes(reader_input,reader_options,rinv,rcaps);
    if (rplan.borrowed_numerical_bytes!=p.reader_borrowed_numerical_bytes || rplan.borrowed_table_bytes!=p.reader_table_bytes)
        throw std::logic_error("Gaussian local triples realized reader roles differ from owner census");
    auto reader=make_bounded_restricted_pair_ccsd_amplitudes(reader_input,reader_options,rinv,rcaps);
    CountedIntegrals integrals{provider.provider().integral_provider(basis),0,p.common_integral_calls_upper_bound};
    const U snapshot=ccsd.solver().final_snapshot().iteration;
    if (!snapshot) throw std::invalid_argument("Gaussian local triples requires an evaluated converged CCSD snapshot");
    MomentContext moments{sources,p,options,live,caps,reader,zero_fov,integrals,snapshot,0,
        Digest("vibeqc.periodic.gaussian-local-triples.consumed-moments")};
    auto moment_provider=moment_provider_metadata(p);moment_provider.visit=&MomentContext::visit;moment_provider.context=&moments;
    Events events{sources,p,reader,moments,integrals,callback,callback_context,{},{},{},{},{},{}};events.pin();
    const BoundedRestrictedLocalTriplesSolverInput input{o,n,triples.data(),triples.size(),
        {basis.f_oo_data(),static_cast<std::size_t>(mul(o,o))},{basis.f_vv_data(),static_cast<std::size_t>(mul(n,n))},snapshot};
    const auto sinv=solver_inventory(p,live);
    const auto actual=plan_bounded_restricted_local_triples_solver(input,moment_provider,options.solver,sinv,caps.solver);
    if (actual.borrowed_numerical_bytes!=p.solver_borrowed_numerical_bytes
        || actual.total_rank!=p.total_retained_rank || actual.total_amplitude_elements!=p.total_amplitude_elements
        || actual.peak_owned_numerical_bytes>p.solver_upper.peak_owned_numerical_bytes
        || actual.work_units_upper_bound>p.solver_upper.work_units_upper_bound)
        throw std::logic_error("Gaussian local triples realized solver plan exceeds enclosing rank census");
    Result result;result.memory_=p;result.state_=ref.state_handle();result.context_=provider.context_handle();
    result.ccsd_=ccsd.identity_sha256();result.spaces_=spaces.identity_sha256();
    result.ccsd_correlation_energy_=real::finite(ccsd.solver().final_snapshot().correlation_energy);
    result.diagnostics_.brillouin_projection_frobenius_norm=fov_norm;
    result.diagnostics_.complete_common_finite_torus_basis=spaces.diagnostics().complete_common_finite_torus_basis;
    result.diagnostics_.all_retained_triples_full_common_rank=spaces.diagnostics().all_retained_full_common_rank;
    events.emit(PeriodicGaussianLocalTriplesStage::Begin);
    result.solver_.emplace(bounded_restricted_local_triples_solve(input,moment_provider,options.solver,sinv,caps.solver,&Events::solver,&events));
    events.check();
    if (integrals.calls>p.common_integral_calls_upper_bound || integrals.calls>caps.maximum_common_integral_calls
        || moments.visits!=result.solver_->final_snapshot().moment_visits)
        throw std::logic_error("Gaussian local triples completed producer census exceeds admission");
    moments.receipts.u64(moments.visits);moments.receipts.u64(integrals.calls);
    result.moments_=moments.receipts.finish();
    result.diagnostics_.completed_common_integral_calls=integrals.calls;
    Digest identity("vibeqc.periodic.gaussian-local-triples.identity");
    for (const auto* s:std::array<const std::string*,10>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &result.ccsd_,&result.spaces_,&provider.identity_sha256(),&provider.hf_reference_source_identity_sha256(),
        &provider.source_context_identity_sha256(),&result.moments_,&result.solver_->input_identity_sha256(),&result.solver_->payload_sha256()}) identity.string(*s);
    identity.real(fov_norm);identity.real(result.ccsd_correlation_energy_);options_wire(identity,options);
    identity.u64(result.diagnostics_.complete_common_finite_torus_basis);
    identity.u64(result.diagnostics_.all_retained_triples_full_common_rank);identity.string(policy);
    result.identity_=identity.finish();
    events.emit(PeriodicGaussianLocalTriplesStage::Finished);
    result.diagnostics_.completed_progress_callbacks=events.event.callback_count;
    return result;
}
}  // namespace vibeqc
