#include "vibeqc/periodic_gaussian_real_local_provider.hpp"

#include <cfloat>
#include <optional>
#include "periodic_correlation_real_local_provider_internal.hpp"

namespace vibeqc {
namespace {
namespace real = periodic_correlation_real_local_detail;
namespace local = periodic_correlation_local_detail;
using real::add;
using real::mul;
using Complex = std::complex<double>;
using Plan = PeriodicGaussianRealLocalProviderPlan;
using Config = PeriodicGaussianRealLocalProviderConfig;
using Live = PeriodicGaussianRealLocalProviderLiveInventory;
using Caps = PeriodicGaussianRealLocalProviderCaps;
using Options = PeriodicCorrelationRealLocalProviderOptions;
using Stage = PeriodicGaussianRealLocalProviderStage;
using Panel = PeriodicGaussianLocalOrbitalFactorPanel;
static_assert(sizeof(Complex) == 16 && FLT_EVAL_METHOD == 0,
              "Gaussian real provider requires binary64 complex and no excess evaluation precision");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian real provider does not support fast/finite-only math"
#endif

std::uint64_t ceil_div(std::uint64_t a,std::uint64_t b) { return a/b+(a%b != 0); }
void limit(std::uint64_t value,std::uint64_t cap,const char* message) {
    if (!cap || value > cap) throw std::length_error(message);
}
void resource_caps(const PeriodicGaussianMetricCaps& c) {
    if (!c.maximum_owned_numeric_bytes || !c.maximum_per_replica_inventoried_bytes
        || !c.maximum_node_inventoried_bytes || !c.maximum_candidate_evaluations || !c.maximum_work_units)
        throw std::invalid_argument("Gaussian real provider resource caps must be positive");
}
std::uint64_t source_work(std::uint64_t evaluations) { return add(mul(2048,evaluations),65536); }
std::uint64_t known_local_bytes(const Plan& p) {
    return add(p.caller_gauge_bytes,add(p.live_wannier_bytes,
        add(p.live_domain_bytes,add(p.live_space_bytes,p.live_basis_bytes))));
}
std::uint64_t explicit_live_bytes(const Live& live) {
    return add(live.other_retained_bytes_per_worker,
        add(live.other_transient_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker));
}
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& context,
                                           const PeriodicGaussianSourceCaps& supplied) {
    const auto& a=context.inventory().ao; const auto& b=context.inventory().auxiliary;
    PeriodicGaussianSourceCaps c;
    c.maximum_context_storage_bytes=sizeof(PeriodicGaussianSourceContext);
    c.maximum_kpoint_count=context.mesh().size(); c.maximum_shell_count=add(a.shell_count,b.shell_count);
    c.maximum_contraction_count=add(a.contraction_count,b.contraction_count);
    c.maximum_primitive_numeric_lanes=add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count));
    c.maximum_basis_content_wire_bytes=add(a.content_wire_bytes,b.content_wire_bytes);
    c.maximum_borrowed_active_numeric_bytes=context.inventory().combined_borrowed_active_numeric_bytes;
    c.maximum_work_units=context.inventory().work_units_upper_bound;
    const std::array<std::uint64_t,8> values{{c.maximum_context_storage_bytes,c.maximum_kpoint_count,
        c.maximum_shell_count,c.maximum_contraction_count,c.maximum_primitive_numeric_lanes,
        c.maximum_basis_content_wire_bytes,c.maximum_borrowed_active_numeric_bytes,c.maximum_work_units}};
    const std::array<std::uint64_t,8> ceilings{{supplied.maximum_context_storage_bytes,supplied.maximum_kpoint_count,
        supplied.maximum_shell_count,supplied.maximum_contraction_count,supplied.maximum_primitive_numeric_lanes,
        supplied.maximum_basis_content_wire_bytes,supplied.maximum_borrowed_active_numeric_bytes,supplied.maximum_work_units}};
    for (std::size_t i=0;i<values.size();++i)
        limit(values[i],ceilings[i],"Gaussian real provider basis census exceeds verification cap");
    return c;
}
void validate_inputs(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis) {
    real::float_environment(); local::validate_wannier(reference,wannier); local::validate_pao(reference,domain,space);
    if (!hf.converged() || !hf.context_handle() || hf.state_handle().get()!=reference.state_handle().get())
        throw std::invalid_argument("Gaussian real provider requires actual converged HF and identical reference state owner");
    const auto& context=*hf.context_handle();
    if (basis.state_handle().get()!=reference.state_handle().get()
        || basis.allocation_identity()!=reference.dimensions().allocation_identity
        || basis.contract_version()!=kPeriodicCorrelationRealLocalBasisContractVersion
        || context.mesh().size()!=reference.state().n_kpoints()
        || context.inventory().ao.function_count!=reference.state().n_basis()
        || context.inventory().auxiliary.function_count!=reference.dimensions().n_auxiliary)
        throw std::invalid_argument("Gaussian real provider source, real-basis or allocation provenance differs");
    (void) basis.occupied_indices_data(); (void) basis.f_oo_data();
    if (basis.identity_sha256().size()!=64 || basis.local_basis_identity_sha256().size()!=64)
        throw std::logic_error("Gaussian real provider lacks native real-basis identities");
}
void resource_wire(real::Digest& h,const PeriodicGaussianMetricCaps& c) {
    for (auto value:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) h.u64(value);
}
void controls_wire(real::Digest& h,const Config& c,const Live& live,const Caps& caps) {
    h.u64(c.auxiliary_block); h.u64(c.panel.ao_pair_block);
    for (auto value:{live.other_retained_bytes_per_worker,live.other_transient_bytes_per_worker,
        live.fixed_backend_margin_bytes_per_worker}) h.u64(value);
    resource_wire(h,caps.resources); resource_wire(h,caps.metric); resource_wire(h,caps.panel.resources);
    resource_wire(h,caps.panel.tile.resources); h.u64(caps.panel.tile.maximum_image_candidates);
    h.u64(caps.panel.maximum_tile_calls); h.u64(caps.panel.maximum_image_candidate_evaluations);
    for (auto value:{caps.maximum_factor_panels,caps.maximum_tile_calls,caps.maximum_image_candidate_evaluations,
        caps.maximum_progress_callbacks,caps.maximum_scalar_work_units}) h.u64(value);
}
void plan_wire(real::Digest& h,const Plan& p) {
    for (auto value:{p.n_cells,p.n_basis,p.n_auxiliary,p.occupied_count,p.virtual_count,p.orbital_count,
        p.density_count,p.row_count,p.self_inverse_q_count,p.auxiliary_block_count,p.factor_panels,p.tile_calls,
        p.retained_row_bytes,p.norm_workspace_bytes,p.whitener_bytes,p.maximum_live_whitener_bytes,
        p.retained_partner_panel_bytes,p.panel_driver_owned_bytes,p.metric_phase_owned_upper_bound,
        p.panel_phase_owned_upper_bound,p.peak_owned_numerical_bytes,p.borrowed_basis_active_numeric_bytes,
        p.caller_gauge_bytes,p.live_wannier_bytes,p.live_domain_bytes,p.live_space_bytes,p.live_basis_bytes,
        p.basis_index_alias_bytes,p.macro_fixed_object_bytes,p.maximum_leaf_fixed_object_bytes,
        p.replicas_per_node,p.reference_base_node_bytes,p.per_replica_inventoried_bytes,p.required_node_memory_bytes,
        p.source_factory_calls,p.metric_calls,p.whitening_calls,p.reciprocal_candidate_evaluations_upper_bound,
        p.image_candidate_evaluations_upper_bound,p.progress_callback_upper_bound,p.input_check_work_units,
        p.driver_work_units,p.work_units,p.scalar_work_units}) h.u64(value);
    controls_wire(h,p.config,p.live,p.caps);
}
void ascii(std::array<char,64>& output,const std::string& value) {
    if (value.size()!=output.size()) throw std::logic_error("Gaussian real provider digest length is invalid");
    std::copy(value.begin(),value.end(),output.begin());
}
} // namespace

Plan plan_periodic_gaussian_real_local_provider(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Config& config,const Live& live,const Caps& caps) {
    validate_inputs(hf,reference,wannier,domain,space,basis);
    resource_caps(caps.resources); resource_caps(caps.metric); resource_caps(caps.panel.resources);
    resource_caps(caps.panel.tile.resources);
    if (!live.fixed_backend_margin_bytes_per_worker || !caps.maximum_factor_panels || !caps.maximum_tile_calls
        || !caps.maximum_image_candidate_evaluations || !caps.maximum_progress_callbacks
        || !caps.maximum_scalar_work_units || !caps.panel.maximum_tile_calls
        || !caps.panel.maximum_image_candidate_evaluations || !caps.panel.tile.maximum_image_candidates)
        throw std::invalid_argument("Gaussian real provider controls must be positive");
    const auto& context=*hf.context_handle(); const auto& recipe=hf.plan().fock.config;
    (void) exact_basis_caps(context,recipe.metric.basis_verification_caps);
    (void) exact_basis_caps(context,recipe.tile.basis_verification_caps);
    Plan p; p.config=config; p.live=live; p.caps=caps;
    p.n_cells=reference.state().n_kpoints(); p.n_basis=reference.state().n_basis();
    p.n_auxiliary=context.inventory().auxiliary.function_count;
    p.occupied_count=basis.memory().occupied_count; p.virtual_count=basis.memory().virtual_count;
    p.orbital_count=add(p.occupied_count,p.virtual_count);
    const auto n=p.n_basis,a=p.n_auxiliary,k=p.n_cells,m=p.orbital_count,n2=mul(n,n),m2=mul(m,m);
    if (!config.auxiliary_block || config.auxiliary_block>a || !config.panel.ao_pair_block
        || config.panel.ao_pair_block>n2 || !p.occupied_count || !p.virtual_count
        || basis.memory().orbital_count!=m || basis.memory().n_cells!=k)
        throw std::invalid_argument("Gaussian real provider shape or block is invalid");
    p.density_count=mul(m,add(m,1))/2; p.row_count=mul(k,a);
    p.self_inverse_q_count=1;
    for (const auto value:reference.state().mesh())
        p.self_inverse_q_count=mul(p.self_inverse_q_count,value%2==0?2:1);
    const bool nonself=p.self_inverse_q_count!=k;
    p.auxiliary_block_count=ceil_div(a,config.auxiliary_block);
    p.factor_panels=mul(k,p.auxiliary_block_count);
    const auto tiles_per_panel=mul(k,ceil_div(n2,config.panel.ao_pair_block));
    p.tile_calls=mul(p.factor_panels,tiles_per_panel);
    limit(tiles_per_panel,caps.panel.maximum_tile_calls,"Gaussian real provider panel tile calls exceed cap");
    p.retained_row_bytes=mul(8,mul(p.row_count,p.density_count));
    p.norm_workspace_bytes=mul(24,p.density_count); p.whitener_bytes=mul(16,mul(a,a));
    p.maximum_live_whitener_bytes=mul(nonself?2:1,p.whitener_bytes);
    const auto ab=config.auxiliary_block;
    p.retained_partner_panel_bytes=nonself?add(mul(16,mul(ab,m2)),mul(16,p.occupied_count)):0;
    p.panel_driver_owned_bytes=add(mul(32,mul(ab,m2)),add(mul(32,mul(n,m)),add(mul(32,n),mul(16,p.occupied_count))));
    const auto retained=add(p.retained_row_bytes,p.norm_workspace_bytes);
    p.metric_phase_owned_upper_bound=add(retained,add(nonself?p.whitener_bytes:0,caps.metric.maximum_owned_numeric_bytes));
    const auto building=add(p.panel_driver_owned_bytes,caps.panel.tile.resources.maximum_owned_numeric_bytes);
    limit(building,caps.panel.resources.maximum_owned_numeric_bytes,"Gaussian real provider panel-owned ceiling is insufficient");
    limit(add(mul(32,mul(a,a)),mul(8,a)),caps.metric.maximum_owned_numeric_bytes,
        "Gaussian real provider principal-whitener owned cap is insufficient");
    p.panel_phase_owned_upper_bound=add(retained,add(p.maximum_live_whitener_bytes,
        add(p.retained_partner_panel_bytes,building)));
    p.peak_owned_numerical_bytes=std::max(p.metric_phase_owned_upper_bound,p.panel_phase_owned_upper_bound);
    p.borrowed_basis_active_numeric_bytes=context.inventory().combined_borrowed_active_numeric_bytes;
    p.caller_gauge_bytes=wannier.memory().caller_gauge_bytes;
    p.live_wannier_bytes=wannier.memory().retained_coefficient_bytes;
    p.live_domain_bytes=add(domain.memory().retained_domain_index_bytes,domain.memory().retained_matrix_bytes);
    p.live_space_bytes=space.memory().output_numerical_bytes; p.live_basis_bytes=basis.memory().retained_output_bytes;
    p.basis_index_alias_bytes=basis.memory().retained_index_bytes;
    // Logical object inventories are deliberately conservative, separate
    // from exact variable numerical payload and caller backend allowance.
    p.macro_fixed_object_bytes=65536+sizeof(PeriodicGaussianRealLocalProvider)+sizeof(Plan)+sizeof(Config)+sizeof(Live)
        +sizeof(Caps)+sizeof(Options)+sizeof(PeriodicGaussianRealLocalProviderReceipt)
        +sizeof(PeriodicGaussianRealLocalProviderProgress)+sizeof(PeriodicGaussianRHFResult)
        +sizeof(PeriodicCorrelationWannier)+sizeof(PeriodicCorrelationPAODomain)
        +sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis);
    p.maximum_leaf_fixed_object_bytes=add(mul(2,recipe.source_caps.maximum_fixed_storage_bytes),
        sizeof(PeriodicGaussianReciprocalMetric)+2*sizeof(PeriodicGaussianMetricWhitener)
        +sizeof(PeriodicGaussianMetricPlan)+2*sizeof(Panel)+sizeof(PeriodicGaussianLocalOrbitalFactorPlan)
        +sizeof(PeriodicGaussianThreeCenterPlan)+sizeof(PeriodicGaussianThreeCenterTile)+sizeof(PeriodicSystem));
    // Source factory caps describe a different phase. Even an exact-minimum
    // source cap must leave room for the standalone panel's complete fixed
    // inventory plus the sibling source/W/panel counted in make_panel.
    // Keep this explicit lower bound in step with the panel plan; numerical
    // source/W/row owners remain separately counted exactly once above.
    const auto panel_fixed=sizeof(Panel)+sizeof(PeriodicGaussianLocalOrbitalFactorPlan)
        +sizeof(PeriodicGaussianLocalOrbitalFactorSelection)+sizeof(PeriodicGaussianLocalOrbitalFactorConfig)
        +sizeof(PeriodicGaussianLocalOrbitalFactorLiveInventory)+sizeof(PeriodicGaussianLocalOrbitalFactorCaps)
        +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicCorrelationAdmittedReference)
        +sizeof(PeriodicCorrelationWannier)+sizeof(PeriodicCorrelationPAODomain)
        +sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis);
    const auto tile_fixed=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)
        +sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianThreeCenterPlan)
        +sizeof(PeriodicGaussianThreeCenterTile)+sizeof(PeriodicSystem);
    const auto sibling_fixed=sizeof(PeriodicGaussianReciprocalSource)+sizeof(PeriodicGaussianMetricWhitener)+sizeof(Panel);
    p.maximum_leaf_fixed_object_bytes=std::max(p.maximum_leaf_fixed_object_bytes,
        add(sibling_fixed,add(panel_fixed,tile_fixed)));
    const auto& dimensions=reference.dimensions(); const auto& budget=reference.budget();
    p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);
    p.reference_base_node_bytes=add(add(dimensions.external_bytes,dimensions.shared_bytes),
        mul(budget.mpi_ranks,add(dimensions.per_rank_bytes,dimensions.localization_window_bytes_per_rank)));
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,
        add(p.borrowed_basis_active_numeric_bytes,add(known_local_bytes(p),add(p.macro_fixed_object_bytes,
        add(p.maximum_leaf_fixed_object_bytes,explicit_live_bytes(live))))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_replica_inventoried_bytes));
    p.source_factory_calls=k; p.metric_calls=k; p.whitening_calls=k;
    p.progress_callback_upper_bound=add(2,add(mul(3,k),add(p.factor_panels,(k+p.self_inverse_q_count)/2)));
    p.input_check_work_units=add(context.inventory().work_units_upper_bound,
        mul(512,add(p.caller_gauge_bytes/16,mul(p.occupied_count,p.occupied_count))));
    p.driver_work_units=add(mul(add(1,p.progress_callback_upper_bound),p.input_check_work_units),
        add(mul(512,add(mul(p.row_count,add(m2,p.density_count)),p.factor_panels)),65536));
    p.work_units=add(p.driver_work_units,add(mul(k,add(source_work(recipe.source_caps.maximum_candidate_evaluations),
        mul(2,caps.metric.maximum_work_units))),mul(p.factor_panels,caps.panel.resources.maximum_work_units)));
    p.reciprocal_candidate_evaluations_upper_bound=add(mul(k,add(recipe.source_caps.maximum_candidate_evaluations,
        caps.metric.maximum_candidate_evaluations)),mul(p.factor_panels,caps.panel.resources.maximum_candidate_evaluations));
    p.image_candidate_evaluations_upper_bound=mul(p.factor_panels,caps.panel.maximum_image_candidate_evaluations);
    p.scalar_work_units=real::dot_interval_work(p.row_count);
    for (const auto bytes:{p.retained_row_bytes,p.norm_workspace_bytes,p.peak_owned_numerical_bytes,
        mul(1024,add(p.factor_panels,k))}) real::extent(bytes);
    if (p.retained_row_bytes/8>std::vector<double>().max_size() || p.norm_workspace_bytes/8>std::vector<double>().max_size())
        throw std::length_error("Gaussian real provider payload exceeds vector extent");
    limit(p.peak_owned_numerical_bytes,caps.resources.maximum_owned_numeric_bytes,"Gaussian real provider owned numerical cap exceeded");
    limit(p.per_replica_inventoried_bytes,caps.resources.maximum_per_replica_inventoried_bytes,"Gaussian real provider per-replica cap exceeded");
    limit(p.required_node_memory_bytes,caps.resources.maximum_node_inventoried_bytes,"Gaussian real provider node cap exceeded");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian real provider admitted-reference node cap exceeded");
    limit(p.factor_panels,caps.maximum_factor_panels,"Gaussian real provider factor-panel cap exceeded");
    limit(p.tile_calls,caps.maximum_tile_calls,"Gaussian real provider tile-call cap exceeded");
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gaussian real provider progress cap exceeded");
    limit(p.work_units,caps.resources.maximum_work_units,"Gaussian real provider work cap exceeded");
    limit(p.scalar_work_units,caps.maximum_scalar_work_units,"Gaussian real provider scalar work cap exceeded");
    limit(p.reciprocal_candidate_evaluations_upper_bound,caps.resources.maximum_candidate_evaluations,
        "Gaussian real provider cumulative reciprocal candidate cap exceeded");
    limit(p.image_candidate_evaluations_upper_bound,caps.maximum_image_candidate_evaluations,
        "Gaussian real provider cumulative image candidate cap exceeded");
    real::Digest digest("vibeqc.periodic.gaussian-real-local-provider.plan");
    digest.string(context.source_context_identity_sha256()); digest.string(hf.reference_source_identity_sha256());
    digest.string(reference.dimensions().allocation_identity); digest.string(basis.identity_sha256());
    plan_wire(digest,p); ascii(p.identity_ascii,digest.finish()); return p;
}

PeriodicGaussianRealLocalProvider::PeriodicGaussianRealLocalProvider(PeriodicCorrelationRealLocalProvider&& provider)
    : provider_(std::move(provider)) {}
const PeriodicCorrelationRealLocalProvider& PeriodicGaussianRealLocalProvider::provider() const {
    if (!context_) throw std::logic_error("Gaussian real provider owner is consumed");
    (void) provider_.rows_data(); return provider_;
}

PeriodicGaussianRealLocalProvider make_periodic_gaussian_real_local_provider(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
    const Complex* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Config& config,const Options& options,const Live& live,const Caps& caps,
    PeriodicGaussianRealLocalProviderCallback progress,void* progress_context) {
    const auto plan=plan_periodic_gaussian_real_local_provider(hf,reference,wannier,domain,space,basis,config,live,caps);
    const auto settings=options;
    real::tolerance_pair(settings.reversal_absolute_tolerance,settings.reversal_relative_tolerance);
    real::tolerance_pair(settings.conjugacy_absolute_tolerance,settings.conjugacy_relative_tolerance);
    real::tolerance_pair(settings.self_q_absolute_tolerance,settings.self_q_relative_tolerance);
    if (!std::isfinite(settings.maximum_eri_projection_error) || settings.maximum_eri_projection_error<=0
        || !std::isfinite(settings.maximum_scalar_roundoff_error) || settings.maximum_scalar_roundoff_error<=0)
        throw std::invalid_argument("Gaussian real provider error budgets must be finite positive");
    const auto context=hf.context_handle(); const auto& recipe=hf.plan().fock.config;
    const auto basis_caps=exact_basis_caps(*context,recipe.metric.basis_verification_caps);
    context->verify_bases(ao,auxiliary,basis_caps);
    const auto* labels=basis.occupied_indices_data();
    real::indices(labels,mul(2,plan.occupied_count),plan.occupied_count,reference.state());
    const auto gauge_identity=local::validate_gauges(wannier,gauges,gauge_count);
    if (real::local_basis_identity(reference,wannier,gauge_identity,domain,space,labels,plan.occupied_count,
        basis.virtual_selection())!=basis.local_basis_identity_sha256())
        throw std::invalid_argument("Gaussian real provider actual selected orbitals differ from real-basis certificate");
    // The common numerical view has no reader/store, so every reader-specific
    // field remains zero. Gaussian construction/work inventories are NEW
    // named fields on the wrapper, not renamed reader counters.
    PeriodicCorrelationRealLocalProviderMemoryPlan core_memory;
    core_memory.n_cells=plan.n_cells; core_memory.n_auxiliary=plan.n_auxiliary;
    core_memory.occupied_count=plan.occupied_count; core_memory.virtual_count=plan.virtual_count;
    core_memory.orbital_count=plan.orbital_count; core_memory.density_count=plan.density_count;
    core_memory.row_count=plan.row_count; core_memory.self_inverse_q_count=plan.self_inverse_q_count;
    core_memory.retained_row_bytes=plan.retained_row_bytes; core_memory.norm_workspace_bytes=plan.norm_workspace_bytes;
    core_memory.retained_partner_panel_bytes=plan.retained_partner_panel_bytes;
    core_memory.building_panel_peak_bytes=add(plan.panel_driver_owned_bytes,plan.caps.panel.tile.resources.maximum_owned_numeric_bytes);
    core_memory.peak_owned_numerical_bytes=plan.peak_owned_numerical_bytes;
    core_memory.live_basis_bytes=plan.live_basis_bytes; core_memory.basis_index_alias_bytes=plan.basis_index_alias_bytes;
    core_memory.caller_gauge_bytes=plan.caller_gauge_bytes; core_memory.live_wannier_bytes=plan.live_wannier_bytes;
    core_memory.live_domain_bytes=plan.live_domain_bytes; core_memory.live_space_bytes=plan.live_space_bytes;
    core_memory.factor_panels=plan.factor_panels; core_memory.work_units=plan.work_units;
    core_memory.scalar_work_units=plan.scalar_work_units; core_memory.required_node_memory_bytes=plan.required_node_memory_bytes;
    PeriodicGaussianRealLocalProvider result(real::ProviderAccess::gaussian(reference,core_memory,settings,basis,
        context->source_context_identity_sha256(),hf.reference_source_identity_sha256()));
    result.context_=context; result.memory_=plan; result.context_sha_=context->source_context_identity_sha256();
    result.hf_=hf.reference_source_identity_sha256();
    auto& receipt=result.receipt_; auto& diagnostic=real::ProviderAccess::diagnostics(result.provider_);
    auto& rows=real::ProviderAccess::rows(result.provider_);
    receipt.charged_work_units_upper_bound=plan.driver_work_units;
    const auto retained=add(plan.retained_row_bytes,plan.norm_workspace_bytes);
    const auto sibling_controls=sizeof(PeriodicGaussianReciprocalSource)+sizeof(PeriodicGaussianMetricWhitener)+sizeof(Panel);
    real::Digest sources("vibeqc.periodic.gaussian-real-local-provider.consumed-sources");
    sources.string(result.context_sha_); sources.string(result.hf_); sources.u64(plan.n_cells);
    real::Digest panels("vibeqc.periodic.gaussian-real-local-provider.consumed-panels");
    panels.string(result.context_sha_); panels.string(result.hf_); panels.string(basis.identity_sha256());
    panels.u64(plan.factor_panels);
    const auto notify=[&](Stage stage,std::uint64_t q=0,std::uint64_t begin=0) {
        if (!progress) return;
        limit(add(receipt.progress_callback_count,1),plan.caps.maximum_progress_callbacks,
            "Gaussian real provider progress count exceeded");
        const PeriodicGaussianRealLocalProviderProgress event{stage,q,begin,receipt.completed_source_count,
            receipt.completed_factor_panels,receipt.completed_tile_calls,receipt.charged_work_units_upper_bound};
        progress(event,progress_context);
        real::float_environment(); context->verify_bases(ao,auxiliary,basis_caps);
        if (local::validate_gauges(wannier,gauges,gauge_count)!=gauge_identity)
            throw std::invalid_argument("Gaussian real provider gauge changed during progress callback");
        ++receipt.progress_callback_count;
    };
    const auto reserve=[&](std::uint64_t work,std::uint64_t candidates,std::uint64_t images=0) {
        limit(add(receipt.charged_work_units_upper_bound,work),plan.caps.resources.maximum_work_units,
            "Gaussian real provider remaining work cap exceeded");
        limit(add(receipt.reciprocal_candidate_evaluations,candidates),plan.caps.resources.maximum_candidate_evaluations,
            "Gaussian real provider remaining reciprocal candidate cap exceeded");
        limit(add(receipt.image_candidate_evaluations,images),plan.caps.maximum_image_candidate_evaluations,
            "Gaussian real provider remaining image candidate cap exceeded");
    };
    const auto observe=[&](std::uint64_t owned,std::uint64_t replica) {
        if (owned>plan.peak_owned_numerical_bytes || replica>plan.per_replica_inventoried_bytes)
            throw std::logic_error("Gaussian real provider nested live inventory exceeds enclosing admission");
        receipt.maximum_observed_owned_numerical_bytes=std::max(receipt.maximum_observed_owned_numerical_bytes,owned);
        receipt.maximum_observed_per_replica_inventoried_bytes=std::max(receipt.maximum_observed_per_replica_inventoried_bytes,replica);
    };
    const auto metric_live=[&](std::uint64_t sibling_whitener) {
        PeriodicGaussianMetricLiveInventory l;
        l.replicas_per_node=plan.replicas_per_node; l.external_node_bytes=plan.reference_base_node_bytes;
        l.other_retained_bytes_per_replica=add(plan.live.other_retained_bytes_per_worker,
            add(retained,add(sibling_whitener,add(known_local_bytes(plan),add(plan.macro_fixed_object_bytes,sibling_controls)))));
        l.other_transient_bytes_per_replica=plan.live.other_transient_bytes_per_worker;
        l.fixed_backend_margin_bytes_per_replica=plan.live.fixed_backend_margin_bytes_per_worker;
        return l;
    };
    const auto make_source=[&](std::uint64_t q,std::uint64_t sibling_whitener) {
        notify(Stage::Source,q);
        reserve(source_work(recipe.source_caps.maximum_candidate_evaluations),recipe.source_caps.maximum_candidate_evaluations);
        auto source=make_periodic_gaussian_reciprocal_source(context,q,recipe.source_caps);
        const auto evaluations=source.inventory().candidate_evaluations_performed;
        receipt.source_factory_candidate_evaluations=add(receipt.source_factory_candidate_evaluations,evaluations);
        receipt.reciprocal_candidate_evaluations=add(receipt.reciprocal_candidate_evaluations,evaluations);
        receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,source_work(evaluations));
        ++receipt.completed_source_count;
        const auto owned=add(retained,sibling_whitener);
        const auto replica=add(owned,add(plan.borrowed_basis_active_numeric_bytes,add(known_local_bytes(plan),
            add(plan.macro_fixed_object_bytes,add(add(source.inventory().inventoried_fixed_storage_bytes,sibling_controls),
            explicit_live_bytes(plan.live))))));
        observe(owned,replica); return source;
    };
    const auto make_whitener=[&](const PeriodicGaussianReciprocalSource& source,std::uint64_t sibling_whitener) {
        const auto l=metric_live(sibling_whitener);
        notify(Stage::Metric,source.q_index());
        reserve(plan.caps.metric.maximum_work_units,plan.caps.metric.maximum_candidate_evaluations);
        auto raw=build_periodic_gaussian_reciprocal_metric(source,ao,auxiliary,recipe.metric,l,plan.caps.metric);
        receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,raw.plan().work_units_upper_bound);
        receipt.reciprocal_candidate_evaluations=add(receipt.reciprocal_candidate_evaluations,raw.plan().candidate_evaluations);
        observe(add(retained,add(sibling_whitener,raw.plan().owned_numeric_peak_bytes)),raw.plan().per_replica_inventoried_bytes);
        ++receipt.completed_metric_count;
        sources.u64(source.q_index()); sources.string(source.source_identity_sha256());
        sources.string(source.conjugate_source_identity_sha256()); sources.string(raw.payload_identity_sha256());
        notify(Stage::Whitening,source.q_index());
        reserve(plan.caps.metric.maximum_work_units,0);
        auto w=factorize_periodic_gaussian_metric(std::move(raw),recipe.metric.whitener_column_block,l,plan.caps.metric);
        receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,w.plan().work_units_upper_bound);
        observe(add(retained,add(sibling_whitener,w.plan().owned_numeric_peak_bytes)),w.plan().per_replica_inventoried_bytes);
        sources.string(w.payload_identity_sha256()); ++receipt.completed_whitening_count; return w;
    };
    const auto make_panel=[&](const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        std::uint64_t begin,std::uint64_t count,std::uint64_t sibling_whitener,std::uint64_t partner_panel) {
        notify(Stage::Panel,source.q_index(),begin);
        limit(add(receipt.completed_factor_panels,1),plan.caps.maximum_factor_panels,
            "Gaussian real provider factor-panel count exceeded");
        reserve(plan.caps.panel.resources.maximum_work_units,plan.caps.panel.resources.maximum_candidate_evaluations,
            plan.caps.panel.maximum_image_candidate_evaluations);
        PeriodicGaussianLocalOrbitalFactorLiveInventory l;
        l.other_retained_bytes_per_worker=add(plan.live.other_retained_bytes_per_worker,
            add(retained,add(sibling_whitener,add(partner_panel,add(plan.macro_fixed_object_bytes,sibling_controls)))));
        l.other_transient_bytes_per_worker=plan.live.other_transient_bytes_per_worker;
        l.fixed_backend_margin_bytes_per_worker=plan.live.fixed_backend_margin_bytes_per_worker;
        auto panel=build_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,ao,auxiliary,
            wannier,gauges,gauge_count,domain,space,basis,begin,count,plan.config.panel,l,plan.caps.panel);
        const auto selection=panel.selection();
        if (panel.context_handle().get()!=context.get() || panel.state_handle().get()!=reference.state_handle().get()
            || panel.hf_reference_source_identity_sha256()!=result.hf_
            || panel.local_basis_identity_sha256()!=basis.local_basis_identity_sha256()
            || panel.basis_certificate_identity_sha256()!=basis.identity_sha256()
            || panel.q_index()!=source.q_index() || panel.auxiliary_begin()!=begin || panel.auxiliary_count()!=count
            || panel.orbital_count()!=plan.orbital_count || selection.left_begin || selection.right_begin
            || selection.left_count!=plan.orbital_count || selection.right_count!=plan.orbital_count
            || panel.source_identity_sha256()!=source.source_identity_sha256()
            || panel.conjugate_source_identity_sha256()!=source.conjugate_source_identity_sha256()
            || panel.whitener_payload_identity_sha256()!=w.payload_identity_sha256())
            throw std::logic_error("Gaussian real provider native local-panel source or shape differs");
        const auto& d=panel.diagnostics();
        receipt.charged_work_units_upper_bound=add(receipt.charged_work_units_upper_bound,d.charged_work_units_upper_bound);
        receipt.reciprocal_candidate_evaluations=add(receipt.reciprocal_candidate_evaluations,d.reciprocal_candidate_evaluations);
        receipt.image_candidate_evaluations=add(receipt.image_candidate_evaluations,d.image_candidate_evaluations);
        receipt.completed_tile_calls=add(receipt.completed_tile_calls,d.completed_tile_calls);
        limit(receipt.completed_tile_calls,plan.caps.maximum_tile_calls,"Gaussian real provider cumulative tile count exceeded");
        observe(add(retained,add(plan.whitener_bytes,add(sibling_whitener,
            add(partner_panel,d.maximum_observed_owned_numerical_bytes)))),d.maximum_observed_per_replica_inventoried_bytes);
        panels.u64(source.q_index()); panels.u64(begin); panels.u64(count); panels.string(panel.identity_sha256());
        panels.string(panel.source_identity_sha256()); panels.string(panel.whitener_payload_identity_sha256());
        ++receipt.completed_factor_panels; ++diagnostic.factor_panels_built; return panel;
    };
    {
        std::vector<double> norms(static_cast<std::size_t>(plan.norm_workspace_bytes/8));
        notify(Stage::Begin);
        for (std::uint64_t q=0;q<plan.n_cells;++q) {
            const auto qbar=real::negative_cell(q,reference.state().mesh());
            if (q>qbar) continue;
            const auto source=make_source(q,0); const auto w=make_whitener(source,0);
            std::optional<PeriodicGaussianReciprocalSource> opposite;
            std::optional<PeriodicGaussianMetricWhitener> opposite_w;
            if (q!=qbar) {
                opposite.emplace(make_source(qbar,plan.whitener_bytes));
                opposite_w.emplace(make_whitener(*opposite,plan.whitener_bytes));
                if (source.conjugate_source_identity_sha256()!=opposite->source_identity_sha256()
                    || opposite->conjugate_source_identity_sha256()!=source.source_identity_sha256())
                    throw std::logic_error("Gaussian real provider opposite source identities disagree");
            }
            const auto sibling=q==qbar?0:plan.whitener_bytes;
            for (std::uint64_t begin=0;begin<plan.n_auxiliary;begin+=std::min(plan.config.auxiliary_block,plan.n_auxiliary-begin)) {
                const auto count=std::min(plan.config.auxiliary_block,plan.n_auxiliary-begin);
                const auto panel=make_panel(source,w,begin,count,sibling,0);
                std::optional<Panel> partner;
                if (opposite) partner.emplace(make_panel(*opposite,*opposite_w,begin,count,sibling,panel.memory().retained_output_bytes));
                real::convert_provider_panel_pair(panel,partner?&*partner:static_cast<const Panel*>(nullptr),
                    q,qbar,begin,count,core_memory,settings,diagnostic,rows.data(),norms.data());
            }
            notify(Stage::PairComplete,q);
        }
        real::finish_provider_norms(core_memory,diagnostic,norms.data());
    }
    if (receipt.completed_source_count!=plan.n_cells || receipt.completed_metric_count!=plan.n_cells
        || receipt.completed_whitening_count!=plan.n_cells || receipt.completed_factor_panels!=plan.factor_panels
        || receipt.completed_tile_calls!=plan.tile_calls || receipt.charged_work_units_upper_bound>plan.work_units
        || receipt.reciprocal_candidate_evaluations>plan.reciprocal_candidate_evaluations_upper_bound
        || receipt.image_candidate_evaluations>plan.image_candidate_evaluations_upper_bound)
        throw std::logic_error("Gaussian real provider completed traversal differs from admission");
    real::finish_provider_projection(core_memory,settings,diagnostic);
    result.sources_=sources.finish(); const auto consumed=panels.finish();
    real::Digest payload("vibeqc.periodic.correlation.real-local-provider.rows");
    payload.u64(plan.row_count); payload.u64(plan.density_count);
    for (const auto value:rows) payload.real(value);
    const auto row_payload=payload.finish();
    real::Digest identity("vibeqc.periodic.gaussian-real-local-provider.identity");
    for (const auto& value:{result.context_sha_,result.hf_,reference.state().state_identity_sha256(),
        reference.dimensions().allocation_identity,basis.local_basis_identity_sha256(),basis.identity_sha256(),
        result.sources_,consumed,row_payload}) identity.string(value);
    identity.string(real::kFloatPolicy);
    identity.string("actual-finite-Gaussian-HF-recipe;bitwise-HF-factor-consumption-and-infinite-source-uncertified");
    for (const auto value:{settings.reversal_absolute_tolerance,settings.reversal_relative_tolerance,
        settings.conjugacy_absolute_tolerance,settings.conjugacy_relative_tolerance,settings.self_q_absolute_tolerance,
        settings.self_q_relative_tolerance,settings.maximum_eri_projection_error,settings.maximum_scalar_roundoff_error}) identity.real(value);
    identity.u64(diagnostic.factor_panels_built);
    for (const auto value:{diagnostic.maximum_reversal_error,diagnostic.maximum_conjugacy_error,
        diagnostic.maximum_self_q_imaginary_magnitude,diagnostic.maximum_original_density_norm,
        diagnostic.maximum_reversal_norm,diagnostic.maximum_covariance_projection_norm,
        diagnostic.maximum_real_row_conversion_norm,diagnostic.orientation_error_bound,
        diagnostic.covariance_error_bound,diagnostic.conversion_error_bound,diagnostic.maximum_eri_projection_error_bound}) identity.real(value);
    result.identity_=identity.finish();
    real::ProviderAccess::seal(result.provider_,consumed,row_payload,result.identity_);
    notify(Stage::Complete); return result;
}

} // namespace vibeqc
