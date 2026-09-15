#include "vibeqc/periodic_gaussian_pair_interaction.hpp"

#include <cfloat>
#include <optional>
#include "periodic_correlation_real_local_provider_internal.hpp"
#include "periodic_gaussian_mixed_pair_factors_internal.hpp"

namespace vibeqc {
namespace {
namespace real = periodic_correlation_real_local_detail;
namespace local = periodic_correlation_local_detail;
namespace mixed = periodic_gaussian_mixed_pair_detail;
using U = std::uint64_t;
using C = std::complex<double>;
using real::add;
using real::mul;
using real::Digest;
using Plan = PeriodicGaussianPairInteractionPlan;
using Result = PeriodicGaussianPairInteractionBlocks;
using Config = PeriodicGaussianPairInteractionConfig;
using Options = PeriodicGaussianPairInteractionOptions;
using Caps = PeriodicGaussianPairInteractionCaps;
using Live = PeriodicGaussianLocalOrbitalFactorLiveInventory;
using Frame = PeriodicGaussianPairPNOFrameView;
using Panel = PeriodicGaussianMixedPairFactorPanel;
static_assert(sizeof(C)==16 && FLT_EVAL_METHOD==0,"Gaussian pair interactions require binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian pair interactions forbid fast/finite-only math"
#endif
void limit(U value,U cap,const char* message) { if(value>cap) throw std::length_error(message); }
void positive(U value) { if(!value) throw std::invalid_argument("Gaussian pair interaction requires positive explicit controls"); }
U ceil_div(U a,U b) { return a/b+(a%b!=0); }
U triangle(U m) { return m%2 ? mul(m,add(m,1)/2) : mul(m/2,add(m,1)); }
U source_work(U n) { return add(mul(2048,n),65536); }
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& c) {
    const auto& v=c.inventory();const auto& a=v.ao;const auto& b=v.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),c.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),
        add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),v.combined_borrowed_active_numeric_bytes,v.work_units_upper_bound};
}
const void* frame_address(const Frame& f) {
    if(f.is_direct_gram()) return static_cast<const void*>(&f.direct_gram());
    return f.is_embedded() ? static_cast<const void*>(&f.embedded()) : static_cast<const void*>(&f.legacy());
}
U frame_controls(const Frame& f) {
    return add(f.is_direct_gram()?sizeof(PeriodicGaussianGramPairPNOResult):
        f.is_embedded()?sizeof(PeriodicGaussianEmbeddedPairPNOResult):sizeof(PeriodicGaussianPairPNOResult),
        f.retained_receipt_payload_bytes());
}
void direct_receipt(const std::string& s) {
    if(s.size()!=64) throw std::invalid_argument("Gaussian pair interaction direct Gram receipt is missing");
    for(char c:s) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("Gaussian pair interaction direct Gram receipt is malformed");
}
void resources(const PeriodicGaussianMetricCaps& c) {
    for(auto x:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) positive(x);
}
void options_valid(const Options& o) {
    const auto& p=o.real_projection;
    real::tolerance_pair(p.reversal_absolute_tolerance,p.reversal_relative_tolerance);
    real::tolerance_pair(p.conjugacy_absolute_tolerance,p.conjugacy_relative_tolerance);
    real::tolerance_pair(p.self_q_absolute_tolerance,p.self_q_relative_tolerance);
    for(double x:{p.maximum_eri_projection_error,p.maximum_scalar_roundoff_error,o.maximum_overlap_imaginary_norm})
        if(!std::isfinite(x) || x<=0) throw std::invalid_argument("Gaussian pair interaction requires positive finite error budgets");
}
void objects(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Frame& a,const Frame& b,U i,U k) {
    if(a.is_direct_gram()!=b.is_direct_gram())
        throw std::invalid_argument("Gaussian pair interaction direct Gram and legacy source kinds cannot be mixed");
    real::float_environment();local::validate_wannier(ref,wannier);local::validate_pao(ref,domain,space);
    if(!hf.converged() || !hf.context_handle() || !hf.matched_finite_gaussian_hf_source()
        || hf.state_handle().get()!=ref.state_handle().get() || basis.state_handle().get()!=ref.state_handle().get()
        || basis.allocation_identity()!=ref.dimensions().allocation_identity)
        throw std::invalid_argument("Gaussian pair interaction requires identical actual HF/reference/basis owners");
    const auto& c=*hf.context_handle();const auto& bm=basis.memory();const auto sel=basis.virtual_selection();
    if(!bm.occupied_count || !bm.virtual_count || bm.orbital_count!=add(bm.occupied_count,bm.virtual_count)
        || bm.n_cells!=ref.state().n_kpoints() || bm.n_basis!=ref.state().n_basis()
        || c.mesh().mesh()!=ref.state().mesh() || c.mesh().is_shift()!=ref.state().is_shift()
        || c.inventory().ao.function_count!=bm.n_basis || c.inventory().auxiliary.function_count!=ref.dimensions().n_auxiliary
        || sel.count!=bm.virtual_count || sel.begin>=space.retained_dimension()
        || sel.count>space.retained_dimension()-sel.begin || sel.translation_cell>=bm.n_cells
        || i>=bm.occupied_count || k>=bm.occupied_count)
        throw std::invalid_argument("Gaussian pair interaction common frame or occupied slot mismatch");
    (void)basis.occupied_indices_data();(void)basis.f_oo_data();
    for(const auto* f:{&a,&b}) {
        if(f->state_handle().get()!=ref.state_handle().get() || f->context_handle().get()!=hf.context_handle().get()
            || f->common_virtual_dimension()!=bm.virtual_count || f->occupied_count()!=bm.occupied_count
            || f->basis_identity_sha256()!=basis.identity_sha256()
            || f->hf_reference_source_identity_sha256()!=hf.reference_source_identity_sha256())
            throw std::invalid_argument("Gaussian pair interaction PNO frame actual source mismatch");
        for(const auto* s:{&f->identity_sha256(),&f->payload_sha256()})
            if(s->size()!=64) throw std::logic_error("Gaussian pair interaction frame receipt missing");
        if(f->is_direct_gram()) {
            const auto& g=f->direct_gram();
            for(const auto* s:{&g.gram_identity_sha256(),&g.gram_payload_sha256(),
                &g.gram_consumed_sources_identity_sha256(),&g.source_context_identity_sha256()}) direct_receipt(*s);
            if(g.source_context_identity_sha256()!=c.source_context_identity_sha256())
                throw std::invalid_argument("Gaussian pair interaction direct Gram context differs from actual HF source");
        } else if(f->provider_identity_sha256().size()!=64)
            throw std::logic_error("Gaussian pair interaction frame receipt missing");
    }
}
U known(const Plan& p) { return add(p.borrowed_geometry_numerical_bytes,
    add(p.borrowed_local_numerical_bytes,p.borrowed_frame_numerical_bytes)); }
U extra(const Live& l) {
    return add(l.other_retained_bytes_per_worker,add(l.other_transient_bytes_per_worker,l.fixed_backend_margin_bytes_per_worker));
}
constexpr U macro_controls=65536+sizeof(Result)+3*sizeof(Plan)+sizeof(Config)+sizeof(Options)+sizeof(Caps)
    +sizeof(Live)+2*sizeof(Frame)+4*sizeof(Digest)+8*sizeof(std::vector<double>)+4*sizeof(std::vector<C>)
    +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicCorrelationAdmittedReference)
    +sizeof(PeriodicCorrelationWannier)+sizeof(PeriodicCorrelationPAODomain)
    +sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis);
U frame_work(const Frame& a,const Frame& b) {
    auto n=plan_periodic_gaussian_pair_pno_frame_validation(a).work_units;
    if(frame_address(a)!=frame_address(b)) n=add(n,plan_periodic_gaussian_pair_pno_frame_validation(b).work_units);
    return n;
}
void verify_frames(const Frame& a,const Frame& b) {
    a.verify_payload();if(frame_address(a)!=frame_address(b)) b.verify_payload();
}
void option_wire(Digest& h,const Options& o) {
    const auto& p=o.real_projection;
    for(double x:{p.reversal_absolute_tolerance,p.reversal_relative_tolerance,p.conjugacy_absolute_tolerance,
        p.conjugacy_relative_tolerance,p.self_q_absolute_tolerance,p.self_q_relative_tolerance,
        p.maximum_eri_projection_error,p.maximum_scalar_roundoff_error,o.maximum_overlap_imaginary_norm}) h.real(x);
}
// Streaming compensated dot AND outward enclosure of the represented real
// row products. Interval accumulation deliberately spans every q/aux slab.
void include_product(double x,double y,double& sum,double& correction,double& lower,double& upper) {
    real::finite(x);real::finite(y);if(x==0 || y==0) return;
    const double product=real::finite(x*y),next=real::finite(sum+product);
    correction=real::finite(correction+(std::abs(sum)>=std::abs(product)?(sum-next)+product:(product-next)+sum));
    sum=next;lower=real::down(real::finite(lower+real::down(product)));upper=real::up(real::finite(upper+real::up(product)));
}
} // namespace

Plan plan_periodic_gaussian_pair_interaction_blocks(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Frame& a,const Frame& b,U i,U k,const Config& config,const Options& options,const Live& live,const Caps& caps,
    const PeriodicGaussianPairPNOGeometryView* ga,const PeriodicGaussianPairPNOGeometryView* gb) {
    options_valid(options);resources(caps.resources);resources(caps.metric);resources(caps.panel.resources);resources(caps.panel.tile.resources);
    for(U x:{config.auxiliary_block,config.panel.ao_pair_block,live.fixed_backend_margin_bytes_per_worker,
        caps.maximum_factor_panels,caps.maximum_tile_calls,caps.maximum_image_candidate_evaluations,
        caps.maximum_leaf_control_bytes,caps.panel.maximum_tile_calls,caps.panel.maximum_image_candidate_evaluations,
        caps.panel.tile.maximum_image_candidates}) positive(x);
    objects(hf,ref,wannier,domain,space,basis,a,b,i,k);
    if(a.is_direct_gram() && (!ga || !gb))
        throw std::invalid_argument("Gaussian pair interaction direct Gram frames require both original pair geometries");
    Plan p;const auto& state=ref.state();const auto& context=*hf.context_handle();const auto& recipe=hf.plan().fock.config;
    p.direct_gram_frames=a.is_direct_gram();p.source_receipt_control_bytes=p.direct_gram_frames?2*65:0;
    const auto geometry=mixed::plan_geometry(ref,basis,a,b,domain,space,ga,gb);
    p.local_geometry_a=ga!=nullptr;p.local_geometry_b=gb!=nullptr;
    p.borrowed_geometry_numerical_bytes=geometry.numerical_bytes;
    p.borrowed_geometry_control_bytes=geometry.control_bytes;p.geometry_validation_work_units=geometry.validation_work_units;
    p.n_cells=state.n_kpoints();p.n_basis=state.n_basis();p.n_auxiliary=ref.dimensions().n_auxiliary;
    p.target_dimension=a.retained_dimension();p.source_dimension=b.retained_dimension();p.occupied_slot_i=i;p.occupied_slot_k=k;
    const auto A=p.target_dimension,B=p.source_dimension,AB=mul(A,B),n=p.n_basis,K=p.n_cells;
    if(config.auxiliary_block>p.n_auxiliary || config.panel.ao_pair_block>mul(n,n))
        throw std::invalid_argument("Gaussian pair interaction auxiliary/AO block is invalid");
    const U maximum_domain=std::max({domain.domain_dimension(),ga?ga->domain().domain_dimension():0,
        gb?gb->domain().domain_dimension():0});
    auto storage=periodic_gaussian_mixed_pair_factor_storage(K,n,state.n_effective_orbitals(),
        basis.memory().virtual_count,maximum_domain,A,B,config.auxiliary_block,config.panel.ao_pair_block);
    if(ga || gb) storage.coefficient_work_units_per_kpoint=mul(2,storage.coefficient_work_units_per_kpoint);
    p.local_orbital_count=storage.orbital_count;p.density_count=triangle(p.local_orbital_count);p.row_count=mul(K,p.n_auxiliary);
    p.self_inverse_q_count=1;for(auto axis:state.mesh()) p.self_inverse_q_count=mul(p.self_inverse_q_count,axis%2?1:2);
    const bool nonself=p.self_inverse_q_count!=K;
    p.auxiliary_block_count=ceil_div(p.n_auxiliary,config.auxiliary_block);p.factor_panels=mul(K,p.auxiliary_block_count);
    p.tile_calls=mul(p.factor_panels,storage.tile_calls);
    p.retained_output_bytes=mul(24,AB);p.integral_accumulator_bytes=mul(48,AB);
    p.norm_workspace_bytes=mul(24,p.density_count);p.row_slab_bytes=mul(nonself?16:8,mul(config.auxiliary_block,p.density_count));
    p.factor_phase_retained_bytes=add(add(p.retained_output_bytes,p.integral_accumulator_bytes),add(p.norm_workspace_bytes,p.row_slab_bytes));
    p.overlap_phase_bytes=add(add(p.retained_output_bytes,mul(32,AB)),
        add(mul(16,mul(n,p.local_orbital_count)),storage.coefficient_helper_workspace_bytes));
    p.whitener_bytes=mul(16,mul(p.n_auxiliary,p.n_auxiliary));p.maximum_live_whitener_bytes=mul(nonself?2:1,p.whitener_bytes);
    p.retained_partner_panel_bytes=nonself?storage.retained_output_bytes:0;
    p.metric_phase_upper_bytes=add(p.factor_phase_retained_bytes,add(nonself?p.whitener_bytes:0,caps.metric.maximum_owned_numeric_bytes));
    const auto building=add(storage.driver_owned_numerical_bytes,caps.panel.tile.resources.maximum_owned_numeric_bytes);
    limit(building,caps.panel.resources.maximum_owned_numeric_bytes,"Gaussian pair interaction panel owned cap insufficient");
    limit(add(mul(32,mul(p.n_auxiliary,p.n_auxiliary)),mul(8,p.n_auxiliary)),caps.metric.maximum_owned_numeric_bytes,
        "Gaussian pair interaction whitener owned cap insufficient");
    p.panel_phase_upper_bytes=add(p.factor_phase_retained_bytes,add(p.maximum_live_whitener_bytes,
        add(p.retained_partner_panel_bytes,building)));
    p.peak_owned_numerical_bytes=std::max({p.overlap_phase_bytes,p.metric_phase_upper_bytes,p.panel_phase_upper_bytes});
    p.borrowed_local_numerical_bytes=add(wannier.memory().caller_gauge_bytes,add(wannier.memory().retained_coefficient_bytes,
        add(add(domain.memory().retained_domain_index_bytes,domain.memory().retained_matrix_bytes),
        add(space.memory().output_numerical_bytes,basis.memory().retained_output_bytes))));
    p.borrowed_frame_numerical_bytes=a.retained_numerical_bytes();auto fc=frame_controls(a);
    if(frame_address(a)!=frame_address(b)) { p.borrowed_frame_numerical_bytes=add(p.borrowed_frame_numerical_bytes,b.retained_numerical_bytes());fc=add(fc,frame_controls(b)); }
    // All nested logical inventories are bounded before any factory or
    // payload scan, even if the original source fixed cap has no slack.
    // Panel source-dependent numerical counts are independently capped;
    // its fixed-object census depends only on these types and frame owners.
    const auto validation_control=std::max(plan_periodic_gaussian_pair_pno_frame_validation(a).control_storage_bytes,
        plan_periodic_gaussian_pair_pno_frame_validation(b).control_storage_bytes);
    const U sibling_controls=sizeof(PeriodicGaussianReciprocalSource)+sizeof(PeriodicGaussianMetricWhitener)+sizeof(Panel);
    const auto panel_controls=add(add(fc,geometry.control_bytes),add(validation_control,65536+17*65+sizeof(Panel)+sizeof(PeriodicGaussianMixedPairFactorPlan)
        +sizeof(PeriodicGaussianLocalOrbitalFactorConfig)+sizeof(Live)+sizeof(PeriodicGaussianLocalOrbitalFactorCaps)+2*sizeof(Frame)
        +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationWannier)
        +sizeof(PeriodicCorrelationPAODomain)+sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis)));
    const U tile_controls=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)+sizeof(PeriodicGaussianMetricWhitener)
        +sizeof(PeriodicGaussianThreeCenterPlan)+sizeof(PeriodicGaussianThreeCenterTile)+sizeof(PeriodicSystem);
    const U metric_controls=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)+sizeof(PeriodicGaussianReciprocalMetric)
        +sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianMetricPlan);
    p.minimum_leaf_control_bytes=add(sibling_controls,std::max({recipe.source_caps.maximum_fixed_storage_bytes,metric_controls,
        add(panel_controls,tile_controls)}));
    limit(p.minimum_leaf_control_bytes,caps.maximum_leaf_control_bytes,"Gaussian pair interaction leaf control cap insufficient");
    p.borrowed_basis_active_numeric_bytes=context.inventory().combined_borrowed_active_numeric_bytes;
    p.control_storage_reservation_bytes=add(add(add(add(macro_controls,fc),geometry.control_bytes),
        caps.maximum_leaf_control_bytes),p.source_receipt_control_bytes);
    const auto& dims=ref.dimensions();const auto& budget=ref.budget();
    p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);positive(p.replicas_per_node);
    p.reference_base_node_bytes=add(add(dims.external_bytes,dims.shared_bytes),mul(budget.mpi_ranks,add(dims.per_rank_bytes,dims.localization_window_bytes_per_rank)));
    p.per_worker_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(known(p),
        add(p.borrowed_basis_active_numeric_bytes,add(p.control_storage_reservation_bytes,extra(live)))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    p.source_factory_calls=p.metric_calls=p.whitening_calls=K;
    p.driver_work_units=add(mul(2,add(context.inventory().work_units_upper_bound,frame_work(a,b))),
        add(mul(K,storage.coefficient_work_units_per_kpoint),mul(4096,add(add(mul(K,mul(mul(n,n),AB)),
        mul(p.row_count,add(mul(p.local_orbital_count,p.local_orbital_count),mul(2,AB)))),
        add(add(wannier.memory().caller_gauge_bytes,mul(basis.memory().occupied_count,basis.memory().occupied_count)),add(p.factor_panels,1024))))));
    // The enclosing driver also asks each leaf for its immutable plan
    // before construction. Cover that additional fixed-size metadata/SHA
    // traversal separately from the leaf's own construction work ceiling.
    p.driver_work_units=add(p.driver_work_units,mul(1048576,p.factor_panels));
    p.driver_work_units=add(p.driver_work_units,geometry.validation_work_units);
    if(p.direct_gram_frames) p.driver_work_units=add(p.driver_work_units,512*1024);
    p.work_units=add(p.driver_work_units,add(mul(K,add(source_work(recipe.source_caps.maximum_candidate_evaluations),
        mul(2,caps.metric.maximum_work_units))),mul(p.factor_panels,caps.panel.resources.maximum_work_units)));
    p.reciprocal_candidate_evaluations_upper_bound=add(mul(K,add(recipe.source_caps.maximum_candidate_evaluations,
        caps.metric.maximum_candidate_evaluations)),mul(p.factor_panels,caps.panel.resources.maximum_candidate_evaluations));
    p.image_candidate_evaluations_upper_bound=mul(p.factor_panels,caps.panel.maximum_image_candidate_evaluations);
    for(U bytes:{p.peak_owned_numerical_bytes,p.per_worker_inventoried_bytes,p.required_node_memory_bytes,p.retained_output_bytes}) real::extent(bytes);
    limit(p.peak_owned_numerical_bytes/8,std::vector<double>().max_size(),"Gaussian pair interaction numerical extent exceeds vector cap");
    limit(storage.tile_calls,caps.panel.maximum_tile_calls,"Gaussian pair interaction leaf tile count cap exceeded");
    limit(p.peak_owned_numerical_bytes,caps.resources.maximum_owned_numeric_bytes,"Gaussian pair interaction owned cap exceeded");
    limit(p.per_worker_inventoried_bytes,caps.resources.maximum_per_replica_inventoried_bytes,"Gaussian pair interaction worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.resources.maximum_node_inventoried_bytes,"Gaussian pair interaction node cap exceeded");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian pair interaction reference node cap exceeded");
    limit(p.factor_panels,caps.maximum_factor_panels,"Gaussian pair interaction factor panel count cap exceeded");
    limit(p.tile_calls,caps.maximum_tile_calls,"Gaussian pair interaction tile count cap exceeded");
    limit(p.work_units,caps.resources.maximum_work_units,"Gaussian pair interaction work cap exceeded");
    limit(p.reciprocal_candidate_evaluations_upper_bound,caps.resources.maximum_candidate_evaluations,"Gaussian pair interaction candidate cap exceeded");
    limit(p.image_candidate_evaluations_upper_bound,caps.maximum_image_candidate_evaluations,"Gaussian pair interaction image cap exceeded");
    return p;
}

void Result::require_live() const {
    const auto count=mul(memory_.target_dimension,memory_.source_dimension);
    if(!state_ || !context_ || k_.size()!=count || j_.size()!=count || overlap_.size()!=count || payload_.size()!=64
        || (memory_.direct_gram_frames && (target_gram_.size()!=64 || source_gram_.size()!=64
            || !memory_.local_geometry_a || !memory_.local_geometry_b || memory_.source_receipt_control_bytes!=2*65)))
        throw std::logic_error("Gaussian pair interaction result is consumed or malformed");
}
const double* Result::k_ab_data() const { require_live();return k_.data(); }
const double* Result::j_ba_data() const { require_live();return j_.data(); }
const double* Result::overlap_ba_data() const { require_live();return overlap_.data(); }
const std::string& Result::target_gram_identity_sha256() const {
    require_live();
    if(!direct_gram_frames()) throw std::logic_error("legacy pair interaction has no direct Gram generation receipt");
    return target_gram_;
}
const std::string& Result::source_gram_identity_sha256() const {
    require_live();
    if(!direct_gram_frames()) throw std::logic_error("legacy pair interaction has no direct Gram generation receipt");
    return source_gram_;
}

Result build_periodic_gaussian_pair_interaction_blocks(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
    const C* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Frame& a,const Frame& b,U i,U k,const Config& supplied_config,const Options& supplied_options,
    const Live& supplied_live,const Caps& supplied_caps,
    const PeriodicGaussianPairPNOGeometryView* ga,const PeriodicGaussianPairPNOGeometryView* gb) {
    const auto config=supplied_config;const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_pair_interaction_blocks(hf,ref,wannier,domain,space,basis,a,b,i,k,config,options,live,caps,ga,gb);
    const auto context=hf.context_handle();const auto& state=ref.state();const auto& recipe=hf.plan().fock.config;
    const auto basis_caps=exact_basis_caps(*context);
    context->verify_bases(ao,auxiliary,basis_caps);
    const auto* labels=basis.occupied_indices_data();real::indices(labels,mul(2,basis.memory().occupied_count),basis.memory().occupied_count,state);
    const auto gauge_id=local::validate_gauges(wannier,gauges,gauge_count);
    if(real::local_basis_identity(ref,wannier,gauge_id,domain,space,labels,basis.memory().occupied_count,basis.virtual_selection())!=basis.local_basis_identity_sha256())
        throw std::invalid_argument("Gaussian pair interaction actual common coefficients differ from their certificate");
    verify_frames(a,b);
    mixed::validate_geometry(ref,basis,a,b,ga,gb);
    Result result;result.memory_=p;result.target_=a.identity_sha256();result.source_=b.identity_sha256();result.hf_=hf.reference_source_identity_sha256();
    if(p.direct_gram_frames) {
        result.target_gram_=a.direct_gram().gram_identity_sha256();
        result.source_gram_=b.direct_gram().gram_identity_sha256();
    }
    const auto A=p.target_dimension,B=p.source_dimension,AB=mul(A,B),n=p.n_basis,M=p.local_orbital_count;
    result.k_.resize(static_cast<std::size_t>(AB));result.j_.resize(static_cast<std::size_t>(AB));result.overlap_.resize(static_cast<std::size_t>(AB));
    auto& d=result.diagnostics_;d.charged_work_units_upper_bound=p.driver_work_units;
    const auto observe=[&](U owned,U worker) {
        if(owned>p.peak_owned_numerical_bytes || worker>p.per_worker_inventoried_bytes)
            throw std::logic_error("Gaussian pair interaction nested lifetime exceeded enclosing admission");
        d.maximum_observed_owned_numerical_bytes=std::max(d.maximum_observed_owned_numerical_bytes,owned);
        d.maximum_observed_per_worker_inventoried_bytes=std::max(d.maximum_observed_per_worker_inventoried_bytes,worker);
    };
    const auto worker=[&](U owned,U control) { return add(owned,add(known(p),add(p.borrowed_basis_active_numeric_bytes,add(control,extra(live))))); };
    const auto storage=periodic_gaussian_mixed_pair_factor_storage(p.n_cells,n,state.n_effective_orbitals(),basis.memory().virtual_count,
        domain.domain_dimension(),A,B,config.auxiliary_block,config.panel.ao_pair_block);
    {
        std::vector<C> overlap(static_cast<std::size_t>(AB)),correction(static_cast<std::size_t>(AB));
        std::vector<C> coefficients(static_cast<std::size_t>(mul(n,M))),scratch(static_cast<std::size_t>(storage.coefficient_helper_workspace_bytes/16));
        observe(p.overlap_phase_bytes,worker(p.overlap_phase_bytes,p.control_storage_reservation_bytes));
        for(U at=0;at<p.n_cells;++at) {
            mixed::fill_coefficients(state,gauges,domain,space,basis,a,b,i,k,at,coefficients.data(),scratch.data(),ga,gb);
            const auto& S=state.overlap(at);
            for(U c=0;c<B;++c) for(U aa=0;aa<A;++aa) {
                C sum{},comp{};
                for(U mu=0;mu<n;++mu) for(U nu=0;nu<n;++nu)
                    real::accumulate((std::conj(coefficients[(2+A+c)*n+mu])*S(mu,nu))*coefficients[(2+aa)*n+nu],sum,comp);
                real::accumulate(real::finite((sum+comp)/static_cast<double>(p.n_cells)),overlap[c*A+aa],correction[c*A+aa]);
            }
        }
        double imag_squared=0;
        for(U x=0;x<AB;++x) {
            const auto value=real::finite(overlap[x]+correction[x]);real::norm_lane(imag_squared,std::abs(value.imag()));result.overlap_[x]=value.real();
        }
        d.overlap_imaginary_frobenius_upper_bound=real::sqrt_up(imag_squared);
        if(d.overlap_imaginary_frobenius_upper_bound>options.maximum_overlap_imaginary_norm)
            throw std::invalid_argument("Gaussian pair interaction physical overlap imaginary norm exceeds budget");
    }
    Digest consumed("vibeqc.periodic.gaussian-pair-interaction.consumed");
    consumed.string(result.hf_);consumed.string(context->source_context_identity_sha256());consumed.u64(p.factor_panels);
    const auto reserve=[&](U work,U candidates,U images=0) {
        limit(add(d.charged_work_units_upper_bound,work),p.work_units,"Gaussian pair interaction remaining work cap exceeded");
        limit(add(d.reciprocal_candidate_evaluations,candidates),p.reciprocal_candidate_evaluations_upper_bound,"Gaussian pair interaction remaining candidate cap exceeded");
        limit(add(d.image_candidate_evaluations,images),p.image_candidate_evaluations_upper_bound,"Gaussian pair interaction remaining image cap exceeded");
    };
    const U siblings=sizeof(PeriodicGaussianReciprocalSource)+sizeof(PeriodicGaussianMetricWhitener)+sizeof(Panel);
    const auto control_base=p.control_storage_reservation_bytes-caps.maximum_leaf_control_bytes;
    const auto metric_live=[&](U sibling) {
        PeriodicGaussianMetricLiveInventory l;l.replicas_per_node=p.replicas_per_node;l.external_node_bytes=p.reference_base_node_bytes;
        l.other_retained_bytes_per_replica=add(live.other_retained_bytes_per_worker,
            add(p.factor_phase_retained_bytes,add(sibling,add(known(p),add(control_base,siblings)))));
        l.other_transient_bytes_per_replica=live.other_transient_bytes_per_worker;l.fixed_backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
        return l;
    };
    const auto make_source=[&](U q,U sibling) {
        reserve(source_work(recipe.source_caps.maximum_candidate_evaluations),recipe.source_caps.maximum_candidate_evaluations);
        auto s=make_periodic_gaussian_reciprocal_source(context,q,recipe.source_caps);
        const auto count=s.inventory().candidate_evaluations_performed;
        d.reciprocal_candidate_evaluations=add(d.reciprocal_candidate_evaluations,count);d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,source_work(count));
        limit(add(siblings,s.inventory().inventoried_fixed_storage_bytes),caps.maximum_leaf_control_bytes,"Gaussian pair interaction source logical control cap exceeded");
        observe(add(p.factor_phase_retained_bytes,sibling),worker(add(p.factor_phase_retained_bytes,sibling),add(control_base,add(siblings,s.inventory().inventoried_fixed_storage_bytes))));
        ++d.completed_source_count;return s;
    };
    const auto make_w=[&](const PeriodicGaussianReciprocalSource& s,U sibling) {
        const auto l=metric_live(sibling);reserve(caps.metric.maximum_work_units,caps.metric.maximum_candidate_evaluations);
        auto raw=build_periodic_gaussian_reciprocal_metric(s,ao,auxiliary,recipe.metric,l,caps.metric);
        d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,raw.plan().work_units_upper_bound);
        d.reciprocal_candidate_evaluations=add(d.reciprocal_candidate_evaluations,raw.plan().candidate_evaluations);
        observe(add(p.factor_phase_retained_bytes,add(sibling,raw.plan().owned_numeric_peak_bytes)),raw.plan().per_replica_inventoried_bytes);
        consumed.u64(s.q_index());consumed.string(s.source_identity_sha256());consumed.string(s.conjugate_source_identity_sha256());consumed.string(raw.payload_identity_sha256());++d.completed_metric_count;
        reserve(caps.metric.maximum_work_units,0);
        auto w=factorize_periodic_gaussian_metric(std::move(raw),recipe.metric.whitener_column_block,l,caps.metric);
        d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,w.plan().work_units_upper_bound);
        observe(add(p.factor_phase_retained_bytes,add(sibling,w.plan().owned_numeric_peak_bytes)),w.plan().per_replica_inventoried_bytes);
        consumed.string(w.payload_identity_sha256());++d.completed_whitening_count;return w;
    };
    const auto make_panel=[&](const PeriodicGaussianReciprocalSource& s,const PeriodicGaussianMetricWhitener& w,U begin,U count,U sibling,U prior) {
        reserve(caps.panel.resources.maximum_work_units,caps.panel.resources.maximum_candidate_evaluations,caps.panel.maximum_image_candidate_evaluations);
        Live l=live;l.other_retained_bytes_per_worker=add(l.other_retained_bytes_per_worker,
            add(p.factor_phase_retained_bytes,add(sibling,add(prior,add(control_base,siblings)))));
        const auto leaf=plan_periodic_gaussian_mixed_pair_factor_panel(hf,ref,s,w,wannier,domain,space,basis,a,b,i,k,begin,count,config.panel,l,caps.panel,ga,gb);
        // The leaf inventories its own complete frame controls; they already
        // occur in control_base, so that conservative duplication is covered
        // by the explicit leaf-control ceiling, not silently subtracted.
        limit(add(siblings,add(leaf.macro_fixed_object_bytes,leaf.tile_fixed_object_bytes)),
            caps.maximum_leaf_control_bytes,"Gaussian pair interaction panel logical control cap exceeded");
        if(leaf.peak_owned_numerical_bytes>add(storage.driver_owned_numerical_bytes,caps.panel.tile.resources.maximum_owned_numeric_bytes))
            throw std::logic_error("Gaussian pair interaction actual panel exceeds count-only bound");
        auto panel=build_periodic_gaussian_mixed_pair_factor_panel(hf,ref,s,w,ao,auxiliary,wannier,gauges,gauge_count,
            domain,space,basis,a,b,i,k,begin,count,config.panel,l,caps.panel,ga,gb);
        const auto& dp=panel.diagnostics();
        if(panel.context_handle().get()!=context.get() || panel.state_handle().get()!=ref.state_handle().get()
            || panel.orbital_count()!=M || panel.rank_a()!=A || panel.rank_b()!=B || panel.q_index()!=s.q_index()
            || panel.auxiliary_begin()!=begin || panel.auxiliary_count()!=count
            || panel.frame_a_identity_sha256()!=result.target_ || panel.frame_b_identity_sha256()!=result.source_
            || panel.hf_reference_source_identity_sha256()!=result.hf_
            || panel.source_identity_sha256()!=s.source_identity_sha256() || panel.whitener_payload_identity_sha256()!=w.payload_identity_sha256())
            throw std::logic_error("Gaussian pair interaction native panel source mismatch");
        d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,dp.charged_work_units_upper_bound);
        d.reciprocal_candidate_evaluations=add(d.reciprocal_candidate_evaluations,dp.reciprocal_candidate_evaluations);
        d.image_candidate_evaluations=add(d.image_candidate_evaluations,dp.image_candidate_evaluations);d.completed_tile_calls=add(d.completed_tile_calls,dp.completed_tile_calls);
        observe(add(p.factor_phase_retained_bytes,add(p.whitener_bytes,add(sibling,add(prior,dp.maximum_observed_owned_numerical_bytes)))),dp.maximum_observed_per_replica_inventoried_bytes);
        consumed.u64(s.q_index());consumed.u64(begin);consumed.u64(count);consumed.string(panel.identity_sha256());++d.completed_factor_panels;return panel;
    };
    {
        std::vector<double> accum(static_cast<std::size_t>(p.integral_accumulator_bytes/8));
        std::vector<double> rows(static_cast<std::size_t>(p.row_slab_bytes/8)),norms(static_cast<std::size_t>(p.norm_workspace_bytes/8));
        PeriodicCorrelationRealLocalProviderMemoryPlan global;global.n_cells=p.n_cells;global.n_auxiliary=p.n_auxiliary;
        global.orbital_count=M;global.density_count=p.density_count;global.row_count=p.row_count;global.self_inverse_q_count=p.self_inverse_q_count;
        for(U q=0;q<p.n_cells;++q) {
            const auto qbar=real::negative_cell(q,state.mesh());if(q>qbar) continue;
            const auto source=make_source(q,0);const auto w=make_w(source,0);
            std::optional<PeriodicGaussianReciprocalSource> other;std::optional<PeriodicGaussianMetricWhitener> other_w;
            if(q!=qbar) {
                other.emplace(make_source(qbar,p.whitener_bytes));other_w.emplace(make_w(*other,p.whitener_bytes));
                if(source.conjugate_source_identity_sha256()!=other->source_identity_sha256()
                    || other->conjugate_source_identity_sha256()!=source.source_identity_sha256())
                    throw std::logic_error("Gaussian pair interaction actual qbar source mismatch");
            }
            const U sibling=other?p.whitener_bytes:0;
            for(U begin=0;begin<p.n_auxiliary;begin+=std::min(config.auxiliary_block,p.n_auxiliary-begin)) {
                const auto count=std::min(config.auxiliary_block,p.n_auxiliary-begin);
                const auto panel=make_panel(source,w,begin,count,sibling,0);std::optional<Panel> partner;
                if(other) partner.emplace(make_panel(*other,*other_w,begin,count,sibling,panel.memory().retained_output_bytes));
                // This private arithmetic descriptor specifies ONLY slab
                // offsets, not physical q identities. Actual source q/qbar
                // and every owner are authenticated above; global norms span
                // all original q/aux rows, and use the global census below.
                auto slab=global;slab.n_auxiliary=count;
                real::convert_provider_panel_pair(panel,partner?&*partner:static_cast<const Panel*>(nullptr),
                    0,1,0,count,slab,options.real_projection,d.real_projection,rows.data(),norms.data());
                for(U row=0;row<mul(partner?2:1,count);++row) {
                    const auto* values=rows.data()+row*p.density_count;
                    for(U aa=0;aa<A;++aa) for(U c=0;c<B;++c) {
                        const auto x=aa*B+c,y=c*A+aa;
                        include_product(values[real::density_index(0,2+aa,M)],values[real::density_index(1,2+A+c,M)],
                            result.k_[x],accum[x],accum[2*AB+x],accum[3*AB+x]);
                        include_product(values[real::density_index(0,1,M)],values[real::density_index(2+A+c,2+aa,M)],
                            result.j_[y],accum[AB+y],accum[4*AB+y],accum[5*AB+y]);
                    }
                }
            }
        }
        d.real_projection.factor_panels_built=d.completed_factor_panels;
        real::finish_provider_norms(global,d.real_projection,norms.data());real::finish_provider_projection(global,options.real_projection,d.real_projection);
        for(U x=0;x<AB;++x) for(unsigned side=0;side<2;++side) {
            auto& value=side?result.j_[x]:result.k_[x];value=real::finite(value+accum[side*AB+x]);
            const auto lower=accum[(side?4:2)*AB+x],upper=accum[(side?5:3)*AB+x];
            const auto error=std::max(real::difference_up(value,lower),real::difference_up(value,upper));
            d.maximum_integral_roundoff_error=std::max(d.maximum_integral_roundoff_error,error);
            if(error>options.real_projection.maximum_scalar_roundoff_error)
                throw std::invalid_argument("Gaussian pair interaction streamed integral roundoff exceeds budget");
        }
    }
    if(d.completed_source_count!=p.n_cells || d.completed_metric_count!=p.n_cells || d.completed_whitening_count!=p.n_cells
        || d.completed_factor_panels!=p.factor_panels || d.completed_tile_calls!=p.tile_calls
        || d.charged_work_units_upper_bound>p.work_units || d.reciprocal_candidate_evaluations>p.reciprocal_candidate_evaluations_upper_bound
        || d.image_candidate_evaluations>p.image_candidate_evaluations_upper_bound)
        throw std::logic_error("Gaussian pair interaction completed traversal differs from admission");
    objects(hf,ref,wannier,domain,space,basis,a,b,i,k);verify_frames(a,b);
    context->verify_bases(ao,auxiliary,basis_caps);
    if(local::validate_gauges(wannier,gauges,gauge_count)!=gauge_id)
        throw std::invalid_argument("Gaussian pair interaction gauge changed during native construction");
    mixed::validate_geometry(ref,basis,a,b,ga,gb);
    result.consumed_=consumed.finish();Digest payload("vibeqc.periodic.gaussian-pair-interaction.payload");payload.u64(A);payload.u64(B);
    for(const auto* v:{&result.k_,&result.j_,&result.overlap_}) for(double value:*v) payload.real(value);
    result.payload_=payload.finish();Digest identity("vibeqc.periodic.gaussian-pair-interaction.identity");
    for(const auto* s:std::array<const std::string*,6>{&result.hf_,&result.target_,&result.source_,&result.consumed_,&result.payload_,&basis.identity_sha256()}) identity.string(*s);
    identity.u64(i);identity.u64(k);option_wire(identity,options);
    identity.string("original-S-cross-overlap;native-distinct-PNO-columns;all-q-original-auxiliary;measured-real-projection;no-origin-provider-replay;no-energy");
    if(ga || gb) {
        identity.string("optional-authenticated-local-PAO-geometry;original-generation-D");
        identity.u64(ga!=nullptr);identity.u64(gb!=nullptr);
        for(const auto* g:{ga,gb}) if(g) identity.string(g->embedding().identity_sha256());
    }
    if(p.direct_gram_frames) {
        identity.string("DirectPAOGram;homogeneous-generation;mandatory-original-PAO-geometry;no-common-provider");
        identity.string(result.target_gram_);identity.string(result.source_gram_);
    }
    identity.string(real::kFloatPolicy);result.identity_=identity.finish();result.state_=ref.state_handle();result.context_=context;return result;
}
} // namespace vibeqc
