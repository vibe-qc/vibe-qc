#include "vibeqc/periodic_correlation_real_local_ccsd_t.hpp"

#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace arithmetic = periodic_correlation_real_local_detail;
using U = std::uint64_t;
using arithmetic::add;
using arithmetic::mul;
using Options = PeriodicCorrelationRealLocalCCSDTOptions;
using Plan = PeriodicCorrelationRealLocalCCSDTMemoryPlan;

void controls(const Options& options) {
    if (!options.ccsd.maximum_iterations || !options.triples.maximum_iterations)
        throw std::invalid_argument("real-local CCSD(T) iteration counts must be positive");
    for (double x : {options.ccsd.denominator_floor, options.ccsd.singles_residual_tolerance,
        options.ccsd.doubles_residual_tolerance, options.ccsd.energy_tolerance,
        options.triples.denominator_floor, options.triples.residual_tolerance, options.triples.energy_tolerance})
        if (!std::isfinite(x) || x <= 0.0)
            throw std::invalid_argument("real-local CCSD(T) convergence controls must be positive finite");
    for (double x : {options.ccsd.input_symmetry_tolerance, options.triples.input_symmetry_tolerance,
                    options.maximum_additional_fock_projection_norm})
        if (!std::isfinite(x) || x < 0.0)
            throw std::invalid_argument("real-local CCSD(T) audit controls must be nonnegative finite");
}
struct Events {
    PeriodicCorrelationRealLocalCCSDTProgressCallback callback;
    void* context;
    BoundedRestrictedCCSDSolverProgress final_ccsd;
    static void ccsd(const BoundedRestrictedCCSDSolverProgress& p, void* opaque) {
        auto& self = *static_cast<Events*>(opaque);
        self.final_ccsd = p;
        if (self.callback) {
            PeriodicCorrelationRealLocalCCSDTProgress event;
            event.stage = PeriodicCorrelationRealLocalCCSDTStage::CCSD; event.ccsd = p;
            self.callback(event, self.context);
        }
    }
    static void triples(const BoundedRestrictedTriplesSolverProgress& p, void* opaque) {
        const auto& self = *static_cast<const Events*>(opaque);
        if (self.callback) {
            PeriodicCorrelationRealLocalCCSDTProgress event;
            event.stage = PeriodicCorrelationRealLocalCCSDTStage::Triples;
            event.ccsd = self.final_ccsd; event.triples = p;
            self.callback(event, self.context);
        }
    }
};
void admit(const Plan& p, const PeriodicCorrelationRealLocalCCSDTCaps& caps,
           const PeriodicCorrelationAdmittedReference& reference) {
    if (!caps.maximum_owned_numerical_bytes || caps.maximum_owned_numerical_bytes < p.peak_owned_numerical_bytes
        || !caps.maximum_total_numerical_bytes || caps.maximum_total_numerical_bytes < p.total_live_numerical_bytes
        || !caps.maximum_node_numerical_bytes || caps.maximum_node_numerical_bytes < p.required_node_numerical_bytes
        || !reference.budget().memory_limit_bytes || reference.budget().memory_limit_bytes < p.required_node_numerical_bytes)
        throw std::length_error("real-local CCSD(T) numerical memory cap is missing or exceeded");
    if (!caps.maximum_integral_calls || caps.maximum_integral_calls < p.integral_calls_upper_bound
        || !caps.maximum_work_units || caps.maximum_work_units < p.work_units_upper_bound)
        throw std::length_error("real-local CCSD(T) total provider-call or work cap is missing or exceeded");
}
void option_wire(arithmetic::Digest& d, const Options& o) {
    d.u64(o.ccsd.maximum_iterations); d.u64(o.triples.maximum_iterations);
    for (double x : {o.ccsd.denominator_floor, o.ccsd.singles_residual_tolerance,
        o.ccsd.doubles_residual_tolerance, o.ccsd.energy_tolerance, o.ccsd.input_symmetry_tolerance,
        o.triples.denominator_floor, o.triples.residual_tolerance, o.triples.energy_tolerance,
        o.triples.input_symmetry_tolerance, o.maximum_additional_fock_projection_norm}) d.real(x);
}
} // namespace

const BoundedRestrictedTriplesSolverResult& PeriodicCorrelationRealLocalCCSDTResult::triples() const {
    if (!triples_evaluated_) throw std::logic_error("real-local CCSD did not converge; triples were not evaluated");
    return triples_;
}

PeriodicCorrelationRealLocalCCSDTMemoryPlan plan_periodic_correlation_real_local_ccsd_t(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicCorrelationRealLocalProvider& provider, const Options& options,
    const PeriodicCorrelationRealLocalCCSDTInventory& inventory) {
    periodic_correlation_local_detail::validate_reference(reference);
    controls(options);
    if (basis.state_handle().get() != reference.state_handle().get()
        || provider.state_handle().get() != reference.state_handle().get()
        || basis.allocation_identity() != reference.dimensions().allocation_identity)
        throw std::invalid_argument("real-local CCSD(T) requires the same admitted native reference and allocation");
    const auto callback = provider.integral_provider(basis); // also validates live owners/selection
    Plan p;
    p.occupied_count = basis.memory().occupied_count; p.virtual_count = basis.memory().virtual_count;
    const U o = p.occupied_count, v = p.virtual_count;
    p.projected_fock_bytes = mul(8, add(add(mul(o,o),mul(v,v)),mul(o,v)));
    p.borrowed_basis_bytes = basis.memory().retained_output_bytes;
    p.borrowed_factor_row_bytes = provider.memory().retained_row_bytes;
    if (!o || !v || provider.memory().occupied_count != o || provider.memory().virtual_count != v
        || basis.memory().retained_fock_bytes != p.projected_fock_bytes
        || p.borrowed_basis_bytes != add(mul(16,o),p.projected_fock_bytes)
        || callback.retained_numerical_bytes != add(p.borrowed_factor_row_bytes,mul(16,o))
        || callback.maximum_transient_numerical_bytes != 0 || !provider.memory().scalar_work_units)
        throw std::invalid_argument("real-local CCSD(T) owner inventory differs from its numerical layout");
    p.other_live_numerical_bytes_per_replica = inventory.other_live_numerical_bytes_per_replica;
    const auto& b = reference.budget(); const auto& d = reference.dimensions();
    p.numerical_replicas = mul(b.mpi_ranks,b.workers_per_rank);
    p.external_node_numerical_bytes = add(add(d.external_bytes,d.shared_bytes),
        mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    const U retained = add(add(p.borrowed_factor_row_bytes,p.borrowed_basis_bytes),
        p.other_live_numerical_bytes_per_replica);
    p.ccsd = plan_bounded_restricted_ccsd_solver(o,v,options.ccsd.maximum_iterations,false,
        {p.external_node_numerical_bytes,p.numerical_replicas},retained,0);
    p.triples = plan_bounded_restricted_triples_solver(o,v,options.triples.maximum_iterations,
        {p.external_node_numerical_bytes,p.numerical_replicas},retained,0);
    p.peak_owned_numerical_bytes = add(p.projected_fock_bytes,std::max(p.ccsd.peak_owned_numerical_bytes,
        add(p.ccsd.amplitude_snapshot_bytes,p.triples.peak_owned_numerical_bytes)));
    p.retained_output_bytes_upper_bound = add(p.ccsd.amplitude_snapshot_bytes,p.triples.amplitude_snapshot_bytes);
    p.total_live_numerical_bytes = std::max(p.ccsd.total_live_numerical_bytes,p.triples.total_live_numerical_bytes);
    p.required_node_numerical_bytes = add(p.external_node_numerical_bytes,mul(p.numerical_replicas,p.total_live_numerical_bytes));
    if (p.total_live_numerical_bytes != add(p.peak_owned_numerical_bytes,retained))
        throw std::logic_error("real-local CCSD(T) sequential phase accounting is inconsistent");
    p.integral_calls_upper_bound = add(p.ccsd.integral_calls_upper_bound,p.triples.integral_calls_upper_bound);
    p.provider_work_units_upper_bound = mul(p.integral_calls_upper_bound,provider.memory().scalar_work_units);
    p.work_units_upper_bound = add(add(p.ccsd.kernel_work_units_upper_bound,p.triples.kernel_work_units_upper_bound),
        add(p.provider_work_units_upper_bound,mul(256,add(1,add(p.projected_fock_bytes/8,
            p.retained_output_bytes_upper_bound/8)))));
    arithmetic::extent(p.projected_fock_bytes); arithmetic::extent(p.retained_output_bytes_upper_bound);
    if (p.projected_fock_bytes/8 > std::vector<double>().max_size())
        throw std::length_error("real-local CCSD(T) projected Fock exceeds native vector extent");
    return p;
}

PeriodicCorrelationRealLocalCCSDTResult solve_periodic_correlation_real_local_ccsd_t(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicCorrelationRealLocalProvider& provider, const Options& options,
    const PeriodicCorrelationRealLocalCCSDTInventory& inventory, const PeriodicCorrelationRealLocalCCSDTCaps& caps,
    PeriodicCorrelationRealLocalCCSDTProgressCallback progress, void* progress_context) {
    const auto p = plan_periodic_correlation_real_local_ccsd_t(reference,basis,provider,options,inventory);
    admit(p,caps,reference); arithmetic::float_environment();
    const U o=p.occupied_count,v=p.virtual_count,o2=o*o,v2=v*v,ov=o*v;
    const double* oo=basis.f_oo_data(); const double* vv=basis.f_vv_data(); const double* vo=basis.f_ov_data();
    double squared=0.0;
    for (U i=0;i<o2;++i) arithmetic::finite(oo[i]);
    for (U a=0;a<v;++a) for (U b=0;b<v;++b) {
        const double x=arithmetic::finite(vv[a*v+b]);
        if (a!=b) arithmetic::norm_lane(squared,std::abs(x));
    }
    for (U i=0;i<ov;++i) {
        const double x=std::abs(arithmetic::finite(vo[i]));
        arithmetic::norm_lane(squared,x); arithmetic::norm_lane(squared,x); // OV and VO
    }
    const double projection=arithmetic::sqrt_up(squared);
    if (projection>options.maximum_additional_fock_projection_norm)
        throw std::invalid_argument("real-local CCSD(T) quasi-canonical Fock projection exceeds its explicit error budget");

    PeriodicCorrelationRealLocalCCSDTResult result;
    result.memory_=p; result.additional_projection_=projection;
    result.total_projection_=arithmetic::add_up(projection,basis.diagnostics().fock_projection_frobenius_upper_bound);
    result.basis_=basis.identity_sha256(); result.provider_=provider.identity_sha256();
    std::vector<double> fock(static_cast<std::size_t>(p.projected_fock_bytes/8),0.0);
    std::copy_n(oo,static_cast<std::size_t>(o2),fock.data());
    for (U a=0;a<v;++a) fock[o2+a*v+a]=vv[a*v+a];
    const BoundedRestrictedCCSDRealView foo{fock.data(),o2},fvv{fock.data()+o2,v2},fov{fock.data()+o2+v2,ov};
    auto callback=provider.integral_provider(basis);
    // The generic leaves cannot discover these outer live input owners.
    // Original F, original labels, rows and declared other arrays are counted
    // here; their separate borrowed F views count our projected F exactly once.
    callback.retained_numerical_bytes=add(add(p.borrowed_basis_bytes,p.borrowed_factor_row_bytes),
        p.other_live_numerical_bytes_per_replica);
    Events events{progress,progress_context,{}};
    const auto& c=p.ccsd;
    result.ccsd_=bounded_restricted_ccsd_solve({o,v,foo,fvv,fov,{},{}},callback,options.ccsd,
        {p.external_node_numerical_bytes,p.numerical_replicas},
        {c.peak_owned_numerical_bytes,c.total_live_numerical_bytes,c.required_node_numerical_bytes,
         c.integral_calls_upper_bound,c.kernel_work_units_upper_bound},&Events::ccsd,&events);
    if (result.ccsd_.final_snapshot.converged) {
        const auto& t=p.triples;
        result.triples_=bounded_restricted_triples_solve({o,v,
            {result.ccsd_.t1.data(),result.ccsd_.t1.size()}, {result.ccsd_.t2.data(),result.ccsd_.t2.size()},
            foo,fvv,fov,result.ccsd_.final_snapshot.iteration},callback,options.triples,
            {p.external_node_numerical_bytes,p.numerical_replicas},
            {t.peak_owned_numerical_bytes,t.total_live_numerical_bytes,t.required_node_numerical_bytes,
             t.integral_calls_upper_bound,t.kernel_work_units_upper_bound},&Events::triples,&events);
        result.triples_evaluated_=true;
    }
    arithmetic::Digest payload("vibeqc.periodic.correlation.real-local-ccsd-t.payload");
    payload.u64(o); payload.u64(v); payload.u32(result.triples_evaluated_);
    payload.u64(result.ccsd_.final_snapshot.iteration); payload.u32(result.ccsd_.final_snapshot.converged);
    payload.real(result.ccsd_.final_snapshot.correlation_energy);
    for (double x:result.ccsd_.t1) payload.real(x);
    for (double x:result.ccsd_.t2) payload.real(x);
    if (result.triples_evaluated_) {
        payload.u64(result.triples_.final_snapshot.iteration); payload.u32(result.triples_.final_snapshot.converged);
        payload.real(result.triples_.final_snapshot.triples_energy);
        for (double x:result.triples_.amplitudes) payload.real(x);
    }
    result.payload_=payload.finish();
    arithmetic::Digest identity("vibeqc.periodic.correlation.real-local-ccsd-t.identity");
    identity.string(reference.state().state_identity_sha256()); identity.string(result.basis_);
    identity.string(result.provider_); identity.string(result.payload_); identity.string(arithmetic::kFloatPolicy);
    identity.string("unweighted-selected-finite-system;global-ri;coupled-triples;hf-source-and-image-tail-uncertified");
    identity.real(result.additional_projection_); identity.real(result.total_projection_);
    for (double x:fock) identity.real(x);
    option_wire(identity,options); result.identity_=identity.finish();
    return result;
}

} // namespace vibeqc
