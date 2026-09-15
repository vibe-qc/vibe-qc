#include "vibeqc/periodic_gaussian_density_gram.hpp"

#include <cfloat>
#include <optional>
#include "periodic_correlation_real_local_provider_internal.hpp"

namespace vibeqc {
namespace {
namespace real = periodic_correlation_real_local_detail;
namespace local = periodic_correlation_local_detail;
using U = std::uint64_t;
using C = std::complex<double>;
using real::add;
using real::mul;
using real::Digest;
using Plan = PeriodicGaussianDensityGramPlan;
using Result = PeriodicGaussianDensityGramBlock;
using Selection = PeriodicGaussianDensityGramSelection;
using Range = PeriodicGaussianDensityRange;
using Config = PeriodicGaussianDensityGramConfig;
using Options = PeriodicGaussianDensityGramOptions;
using Live = PeriodicGaussianDensityGramLiveInventory;
using Caps = PeriodicGaussianDensityGramCaps;
using Panel = PeriodicGaussianLocalOrbitalFactorPanel;
using Rectangle = PeriodicGaussianLocalOrbitalFactorSelection;
static_assert(sizeof(C)==16 && sizeof(U)==8 && FLT_EVAL_METHOD==0,
    "Gaussian density Gram requires binary64 and uint64 storage");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian density Gram forbids fast/finite-only math"
#endif
constexpr char policy[] = "selected-packed-upper-common-densities;actual-both-oriented-q-qbar;"
    "representative-q-cosine-sine-slab;compensated-dot-with-outward-interval;"
    "selected-densities-only;not-stored-provider-bitwise;no-energy-or-production-DLPNO";
void limit(U value,U cap,const char* message) { if(value>cap) throw std::length_error(message); }
void positive(U value) { if(!value) throw std::invalid_argument("Gaussian density Gram requires positive explicit controls"); }
U triangle(U n) { return n%2 ? mul(n,add(n,1)/2) : mul(n/2,add(n,1)); }
U ceil_div(U n,U d) { return n/d+(n%d!=0); }
U source_work(U n) { return add(mul(2048,n),65536); }
void sha(const std::string& value) {
    if(value.size()!=64) throw std::invalid_argument("Gaussian density Gram requires native SHA256 receipts");
    for(char c:value) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("Gaussian density Gram receipt is malformed");
}
U row_start(U p,U m,U h) { return h-triangle(m-p); }
std::array<U,2> decode(U x,U m,U h) {
    if(!m || x>=h) throw std::out_of_range("Gaussian density Gram packed density is out of range");
    U lo=0,hi=m;
    while(hi-lo>1) {
        const U mid=lo+(hi-lo)/2;
        if(row_start(mid,m,h)<=x) lo=mid; else hi=mid;
    }
    return {lo,add(lo,x-row_start(lo,m,h))};
}
struct RunShape { U count=0,maximum=0; };
RunShape run_shape(Range range,U m,U h) {
    const auto first=decode(range.begin,m,h),last=decode(add(range.begin,range.count-1),m,h);
    if(first[0]==last[0]) return {1,range.count};
    RunShape out{last[0]-first[0]+1,std::max(m-first[1],last[1]-last[0]+1)};
    if(last[0]-first[0]>1) out.maximum=std::max(out.maximum,m-first[0]-1);
    return out;
}
void resources(const PeriodicGaussianMetricCaps& c) {
    for(U x:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) positive(x);
}
void options_valid(const Options& o) {
    real::tolerance_pair(o.reversal_absolute_tolerance,o.reversal_relative_tolerance);
    real::tolerance_pair(o.conjugacy_absolute_tolerance,o.conjugacy_relative_tolerance);
    real::tolerance_pair(o.self_q_absolute_tolerance,o.self_q_relative_tolerance);
    for(double x:{o.maximum_eri_projection_error,o.maximum_scalar_roundoff_error})
        if(!std::isfinite(x) || x<=0)
            throw std::invalid_argument("Gaussian density Gram requires positive finite error budgets");
}
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& c) {
    const auto& v=c.inventory();const auto& a=v.ao;const auto& b=v.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),c.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),
        add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),v.combined_borrowed_active_numeric_bytes,v.work_units_upper_bound};
}
void objects(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis) {
    real::float_environment();local::validate_wannier(ref,wannier);local::validate_pao(ref,domain,space);
    if(!hf.converged() || !hf.context_handle() || !hf.matched_finite_gaussian_hf_source()
        || hf.state_handle()!=ref.state_handle() || basis.state_handle()!=ref.state_handle()
        || basis.contract_version()!=kPeriodicCorrelationRealLocalBasisContractVersion
        || basis.allocation_identity()!=ref.dimensions().allocation_identity)
        throw std::invalid_argument("Gaussian density Gram requires exact actual HF/reference/basis owners");
    const auto& b=basis.memory();const auto& context=*hf.context_handle();const auto v=basis.virtual_selection();
    if(!b.occupied_count || !b.virtual_count || b.orbital_count!=add(b.occupied_count,b.virtual_count)
        || b.n_cells!=ref.state().n_kpoints() || b.n_basis!=ref.state().n_basis()
        || b.occupied_count>mul(b.n_cells,ref.state().n_correlated_occupied())
        || context.mesh().mesh()!=ref.state().mesh() || context.mesh().is_shift()!=ref.state().is_shift()
        || context.inventory().ao.function_count!=b.n_basis
        || context.inventory().auxiliary.function_count!=ref.dimensions().n_auxiliary
        || v.count!=b.virtual_count || v.begin>=space.retained_dimension()
        || v.count>space.retained_dimension()-v.begin || v.translation_cell>=b.n_cells)
        throw std::invalid_argument("Gaussian density Gram common-frame dimensions or source mesh differ");
    const auto& did=domain.pao_domain_identity_sha256();const auto& sid=space.pao_space_identity_sha256();
    sha(did);sha(sid);
    if(!std::equal(did.begin(),did.end(),basis.virtual_domain_identity_ascii().begin())
        || !std::equal(sid.begin(),sid.end(),basis.virtual_space_identity_ascii().begin()))
        throw std::invalid_argument("Gaussian density Gram common PAO owners differ from the native basis");
    (void)basis.occupied_indices_data();(void)basis.f_oo_data();(void)basis.f_vv_data();(void)basis.f_ov_data();
}
U extra(const Live& live) {
    return add(live.other_retained_bytes_per_worker,
        add(live.other_transient_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker));
}
constexpr U macro_controls=65536+sizeof(Result)+3*sizeof(Plan)+sizeof(Config)+sizeof(Options)+sizeof(Caps)
    +sizeof(Live)+3*sizeof(Selection)+2*sizeof(Range)+6*sizeof(Digest)+4*sizeof(std::vector<double>)
    +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicCorrelationAdmittedReference)
    +sizeof(PeriodicCorrelationWannier)+sizeof(PeriodicCorrelationPAODomain)
    +sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis)+32*65;
constexpr U sibling_controls=2*sizeof(PeriodicGaussianReciprocalSource)
    +2*sizeof(PeriodicGaussianMetricWhitener)+2*sizeof(Panel);
void selection_wire(Digest& h,const Selection& s) {
    for(U x:{s.left.begin,s.left.count,s.right.begin,s.right.count}) h.u64(x);
}
void option_wire(Digest& h,const Options& o) {
    for(double x:{o.reversal_absolute_tolerance,o.reversal_relative_tolerance,
        o.conjugacy_absolute_tolerance,o.conjugacy_relative_tolerance,
        o.self_q_absolute_tolerance,o.self_q_relative_tolerance,
        o.maximum_eri_projection_error,o.maximum_scalar_roundoff_error}) h.real(x);
}
std::string input_payload(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
    const C* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis) {
    objects(hf,ref,wannier,domain,space,basis);
    hf.context_handle()->verify_bases(ao,auxiliary,exact_basis_caps(*hf.context_handle()));
    const U o=basis.memory().occupied_count,n=basis.memory().virtual_count;
    real::indices(basis.occupied_indices_data(),mul(2,o),o,ref.state());
    const auto gauge=local::validate_gauges(wannier,gauges,gauge_count);
    if(real::local_basis_identity(ref,wannier,gauge,domain,space,basis.occupied_indices_data(),o,basis.virtual_selection())
        !=basis.local_basis_identity_sha256())
        throw std::invalid_argument("Gaussian density Gram actual common coefficients differ from their certificate");
    Digest h("vibeqc.periodic.gaussian-density-gram.inputs");
    const auto hf_identity=hf.reference_source_identity_sha256();
    for(const auto* value:std::array<const std::string*,8>{&hf_identity,
        &ref.dimensions().allocation_identity,&wannier.wannier_identity_sha256(),&gauge,
        &domain.pao_domain_identity_sha256(),&space.pao_space_identity_sha256(),
        &basis.identity_sha256(),&basis.local_basis_identity_sha256()}) {sha(*value);h.string(*value);}
    h.string(hf.context_handle()->source_context_identity_sha256());
    const U k=wannier.n_cells(),na=wannier.n_basis(),no=wannier.n_home_occupied();
    for(U cell=0;cell<k;++cell) {
        const auto* values=wannier.cell_coefficients(cell);
        for(U x=0;x<mul(na,no);++x) h.complex(values[x]);
    }
    const U d=domain.domain_dimension(),r=space.retained_dimension();
    for(U x=0;x<d;++x) {const auto column=domain.column(x);h.u64(column.cell);h.u64(column.ao);}
    for(U x=0;x<mul(d,d);++x) {h.complex(domain.overlap_data()[x]);h.complex(domain.fock_data()[x]);}
    for(U x=0;x<mul(d,r);++x) h.complex(space.coefficients_data()[x]);
    for(U x=0;x<r;++x) h.real(space.energies_data()[x]);
    for(U x=0;x<d;++x) h.real(space.overlap_eigenvalues_data()[x]);
    for(U x=0;x<mul(2,o);++x) h.u64(basis.occupied_indices_data()[x]);
    for(U x=0;x<mul(o,o);++x) h.real(basis.f_oo_data()[x]);
    for(U x=0;x<mul(n,n);++x) h.real(basis.f_vv_data()[x]);
    for(U x=0;x<mul(o,n);++x) h.real(basis.f_ov_data()[x]);
    return h.finish();
}
void include_product(double x,double y,double& sum,double& correction,double& lower,double& upper) {
    real::finite(x);real::finite(y);if(x==0 || y==0) return;
    const double product=real::finite(x*y),next=real::finite(sum+product);
    correction=real::finite(correction+(std::abs(sum)>=std::abs(product)?(sum-next)+product:(product-next)+sum));
    sum=next;lower=real::down(real::finite(lower+real::down(product)));
    upper=real::up(real::finite(upper+real::up(product)));
}
void reversal(C x,C reversed,const Options& options,PeriodicCorrelationRealLocalProviderDiagnostics& diagnostic) {
    diagnostic.maximum_reversal_error=std::max(diagnostic.maximum_reversal_error,real::defect_up(x,reversed));
    if(!real::within(x,reversed,options.reversal_absolute_tolerance,options.reversal_relative_tolerance))
        throw std::invalid_argument("Gaussian density Gram actual selected same-q reversal exceeds tolerance");
}
void conjugacy(C x,C opposite,const Options& options,PeriodicCorrelationRealLocalProviderDiagnostics& diagnostic) {
    diagnostic.maximum_conjugacy_error=std::max(diagnostic.maximum_conjugacy_error,real::defect_up(opposite,std::conj(x)));
    if(!real::within(opposite,std::conj(x),options.conjugacy_absolute_tolerance,options.conjugacy_relative_tolerance))
        throw std::invalid_argument("Gaussian density Gram actual selected opposite-q conjugacy exceeds tolerance");
}
} // namespace

Plan plan_periodic_gaussian_density_gram(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const Selection& selection,const Config& config,
    const Options& options,const Live& live,const Caps& caps) {
    // All request-shape bounds precede any source/payload inspection. This
    // phase reads fixed scalar metadata only, including the native basis m.
    for(U x:{selection.left.count,selection.right.count,caps.maximum_left_density_count,
        caps.maximum_right_density_count,caps.maximum_output_elements,config.auxiliary_block,config.panel.ao_pair_block,
        live.fixed_backend_margin_bytes_per_worker,caps.maximum_factor_panels,caps.maximum_tile_calls,
        caps.maximum_image_candidate_evaluations,caps.maximum_leaf_control_bytes,
        caps.panel.maximum_tile_calls,caps.panel.maximum_image_candidate_evaluations,
        caps.panel.tile.maximum_image_candidates}) positive(x);
    resources(caps.resources);resources(caps.metric);resources(caps.panel.resources);resources(caps.panel.tile.resources);
    limit(selection.left.count,caps.maximum_left_density_count,"Gaussian density Gram left density cap exceeded");
    limit(selection.right.count,caps.maximum_right_density_count,"Gaussian density Gram right density cap exceeded");
    Plan p;p.selection=selection;p.orbital_count=basis.memory().orbital_count;
    positive(p.orbital_count);p.common_density_count=triangle(p.orbital_count);
    for(const auto range:{selection.left,selection.right})
        if(range.begin>=p.common_density_count || range.count>p.common_density_count-range.begin)
            throw std::out_of_range("Gaussian density Gram range exceeds packed common densities");
    p.output_elements=mul(selection.left.count,selection.right.count);
    limit(p.output_elements,caps.maximum_output_elements,"Gaussian density Gram output element cap exceeded");
    p.selected_density_count=add(selection.left.count,selection.right.count);
    p.retained_output_bytes=mul(8,p.output_elements);
    real::extent(p.retained_output_bytes);
    limit(p.retained_output_bytes,caps.resources.maximum_owned_numeric_bytes,"Gaussian density Gram output owned cap exceeded");
    // Two O(log m) integer decoders and fixed source metadata are covered
    // before evaluating even the scientific-control or owner checks.
    limit(U{1048576},caps.resources.maximum_work_units,"Gaussian density Gram startup metadata work cap exceeded");
    const auto left=run_shape(selection.left,p.orbital_count,p.common_density_count);
    const auto right=run_shape(selection.right,p.orbital_count,p.common_density_count);
    p.left_run_count=left.count;p.right_run_count=right.count;p.maximum_run_length=std::max(left.maximum,right.maximum);
    options_valid(options);objects(hf,ref,wannier,domain,space,basis);
    const auto& state=ref.state();const auto& context=*hf.context_handle();const auto& recipe=hf.plan().fock.config;
    p.n_cells=state.n_kpoints();p.n_basis=state.n_basis();p.n_auxiliary=ref.dimensions().n_auxiliary;
    p.occupied_count=basis.memory().occupied_count;p.virtual_count=basis.memory().virtual_count;
    const U K=p.n_cells,a=p.n_basis,A=p.n_auxiliary,o=p.occupied_count,b=config.auxiliary_block,t=p.maximum_run_length;
    positive(K);positive(a);positive(A);
    if(b>A || config.panel.ao_pair_block>mul(a,a))
        throw std::invalid_argument("Gaussian density Gram auxiliary/AO block exceeds the actual source");
    p.row_count=mul(K,A);p.self_inverse_q_count=1;
    for(auto axis:state.mesh()) p.self_inverse_q_count=mul(p.self_inverse_q_count,axis%2?1:2);
    const bool nonself=p.self_inverse_q_count!=K;
    p.auxiliary_block_count=ceil_div(A,b);
    p.factor_panels=mul(mul(mul(2,K),p.auxiliary_block_count),add(p.left_run_count,p.right_run_count));
    const U tile_per_panel=mul(K,ceil_div(mul(a,a),config.panel.ao_pair_block));
    p.tile_calls=mul(p.factor_panels,tile_per_panel);
    p.integral_accumulator_bytes=mul(24,p.output_elements);
    p.norm_workspace_bytes=mul(24,p.selected_density_count);
    p.row_slab_bytes=mul(nonself?16:8,mul(b,p.selected_density_count));
    p.factor_phase_retained_bytes=add(add(p.retained_output_bytes,p.integral_accumulator_bytes),
        add(p.norm_workspace_bytes,p.row_slab_bytes));
    p.whitener_bytes=mul(16,mul(A,A));p.maximum_live_whitener_bytes=mul(nonself?2:1,p.whitener_bytes);
    p.retained_panel_upper_bytes=add(mul(16,mul(b,t)),mul(16,o));
    p.panel_driver_owned_upper_bytes=add(mul(32,mul(b,t)),add(mul(16,mul(a,add(t,1))),add(mul(32,a),mul(16,o))));
    const U building=add(p.panel_driver_owned_upper_bytes,caps.panel.tile.resources.maximum_owned_numeric_bytes);
    limit(building,caps.panel.resources.maximum_owned_numeric_bytes,"Gaussian density Gram panel owned cap insufficient");
    limit(add(mul(32,mul(A,A)),mul(8,A)),caps.metric.maximum_owned_numeric_bytes,
        "Gaussian density Gram principal-whitener owned cap insufficient");
    p.metric_phase_upper_bytes=add(p.factor_phase_retained_bytes,add(nonself?p.whitener_bytes:0,caps.metric.maximum_owned_numeric_bytes));
    p.panel_phase_upper_bytes=add(p.factor_phase_retained_bytes,add(p.maximum_live_whitener_bytes,
        add(mul(nonself?2:1,p.retained_panel_upper_bytes),building)));
    p.peak_owned_numerical_bytes=std::max(p.metric_phase_upper_bytes,p.panel_phase_upper_bytes);
    p.borrowed_local_numerical_bytes=add(wannier.memory().caller_gauge_bytes,
        add(wannier.memory().retained_coefficient_bytes,add(domain.memory().retained_domain_index_bytes,
        add(domain.memory().retained_matrix_bytes,add(space.memory().output_numerical_bytes,basis.memory().retained_output_bytes)))));
    p.borrowed_basis_active_numeric_bytes=context.inventory().combined_borrowed_active_numeric_bytes;
    const U panel_controls=sizeof(Panel)+sizeof(PeriodicGaussianLocalOrbitalFactorPlan)+sizeof(Rectangle)
        +sizeof(PeriodicGaussianLocalOrbitalFactorConfig)+sizeof(Live)+sizeof(PeriodicGaussianLocalOrbitalFactorCaps)
        +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationWannier)
        +sizeof(PeriodicCorrelationPAODomain)+sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis);
    const U tile_controls=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)
        +sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianThreeCenterPlan)
        +sizeof(PeriodicGaussianThreeCenterTile)+sizeof(PeriodicSystem);
    const U metric_controls=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)
        +sizeof(PeriodicGaussianReciprocalMetric)+sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianMetricPlan);
    p.macro_control_storage_bytes=macro_controls;
    p.minimum_leaf_control_bytes=add(sibling_controls,std::max({recipe.source_caps.maximum_fixed_storage_bytes,
        metric_controls,add(panel_controls,tile_controls)}));
    limit(p.minimum_leaf_control_bytes,caps.maximum_leaf_control_bytes,"Gaussian density Gram leaf control cap insufficient");
    p.control_storage_reservation_bytes=add(macro_controls,caps.maximum_leaf_control_bytes);
    const auto& dimensions=ref.dimensions();const auto& budget=ref.budget();
    p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);positive(p.replicas_per_node);
    p.reference_base_node_bytes=add(add(dimensions.external_bytes,dimensions.shared_bytes),
        mul(budget.mpi_ranks,add(dimensions.per_rank_bytes,dimensions.localization_window_bytes_per_rank)));
    p.per_worker_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_local_numerical_bytes,
        add(p.borrowed_basis_active_numeric_bytes,add(p.control_storage_reservation_bytes,extra(live)))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    p.source_factory_calls=p.metric_calls=p.whitening_calls=K;
    p.input_validation_work_units=add(context.inventory().work_units_upper_bound,
        mul(4096,add(4096,add(p.borrowed_local_numerical_bytes/8,mul(o,o)))));
    p.driver_work_units=add(U{1048576},add(mul(2,p.input_validation_work_units),
        add(mul(4096,add(mul(p.row_count,add(p.selected_density_count,p.output_elements)),p.selected_density_count)),
        mul(1048576,p.factor_panels))));
    p.work_units=add(p.driver_work_units,add(mul(K,add(source_work(recipe.source_caps.maximum_candidate_evaluations),
        mul(2,caps.metric.maximum_work_units))),mul(p.factor_panels,caps.panel.resources.maximum_work_units)));
    p.reciprocal_candidate_evaluations_upper_bound=add(mul(K,add(recipe.source_caps.maximum_candidate_evaluations,
        caps.metric.maximum_candidate_evaluations)),mul(p.factor_panels,caps.panel.resources.maximum_candidate_evaluations));
    p.image_candidate_evaluations_upper_bound=mul(p.factor_panels,caps.panel.maximum_image_candidate_evaluations);
    for(U bytes:{p.peak_owned_numerical_bytes,p.per_worker_inventoried_bytes,p.required_node_memory_bytes,
        p.integral_accumulator_bytes,p.norm_workspace_bytes,p.row_slab_bytes,p.retained_panel_upper_bytes}) real::extent(bytes);
    limit(p.peak_owned_numerical_bytes/8,std::vector<double>().max_size(),"Gaussian density Gram numerical vector extent exceeded");
    limit(tile_per_panel,caps.panel.maximum_tile_calls,"Gaussian density Gram leaf tile count cap insufficient");
    limit(p.peak_owned_numerical_bytes,caps.resources.maximum_owned_numeric_bytes,"Gaussian density Gram owned cap exceeded");
    limit(p.per_worker_inventoried_bytes,caps.resources.maximum_per_replica_inventoried_bytes,"Gaussian density Gram worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.resources.maximum_node_inventoried_bytes,"Gaussian density Gram node cap exceeded");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian density Gram admitted-reference node cap exceeded");
    limit(p.factor_panels,caps.maximum_factor_panels,"Gaussian density Gram factor panel cap exceeded");
    limit(p.tile_calls,caps.maximum_tile_calls,"Gaussian density Gram tile call cap exceeded");
    limit(p.work_units,caps.resources.maximum_work_units,"Gaussian density Gram work cap exceeded");
    limit(p.reciprocal_candidate_evaluations_upper_bound,caps.resources.maximum_candidate_evaluations,
        "Gaussian density Gram reciprocal candidate cap exceeded");
    limit(p.image_candidate_evaluations_upper_bound,caps.maximum_image_candidate_evaluations,"Gaussian density Gram image cap exceeded");
    return p;
}

void Result::require_live() const {
    if(!state_ || !context_ || values_.size()!=memory_.output_elements
        || memory_.output_elements!=mul(memory_.selection.left.count,memory_.selection.right.count)
        || payload_.size()!=64 || identity_.size()!=64)
        throw std::logic_error("Gaussian density Gram output is consumed or malformed");
}
const double* Result::data() const {require_live();return values_.data();}
double Result::element(std::size_t left,std::size_t right) const {
    require_live();
    if(left>=memory_.selection.left.count || right>=memory_.selection.right.count)
        throw std::out_of_range("Gaussian density Gram output index is out of range");
    return values_[left*memory_.selection.right.count+right];
}
std::array<U,2> Result::left_density(std::size_t index) const {
    require_live();if(index>=memory_.selection.left.count) throw std::out_of_range("Gaussian density Gram left density index is out of range");
    return decode(add(memory_.selection.left.begin,index),memory_.orbital_count,memory_.common_density_count);
}
std::array<U,2> Result::right_density(std::size_t index) const {
    require_live();if(index>=memory_.selection.right.count) throw std::out_of_range("Gaussian density Gram right density index is out of range");
    return decode(add(memory_.selection.right.begin,index),memory_.orbital_count,memory_.common_density_count);
}

Result build_periodic_gaussian_density_gram(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const BasisSet& ao,const BasisSet& auxiliary,
    const PeriodicCorrelationWannier& wannier,const C* gauges,std::size_t gauge_count,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const Selection& supplied_selection,const Config& supplied_config,
    const Options& supplied_options,const Live& supplied_live,const Caps& supplied_caps) {
    const auto selection=supplied_selection;const auto config=supplied_config;const auto options=supplied_options;
    const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_density_gram(hf,ref,wannier,domain,space,basis,selection,config,options,live,caps);
    const auto context=hf.context_handle();const auto& state=ref.state();const auto& recipe=hf.plan().fock.config;
    const auto initial=input_payload(hf,ref,ao,auxiliary,wannier,gauges,gauge_count,domain,space,basis);
    Result result;result.memory_=p;result.hf_=hf.reference_source_identity_sha256();
    result.context_sha_=context->source_context_identity_sha256();result.basis_=basis.identity_sha256();
    result.values_.resize(static_cast<std::size_t>(p.output_elements));
    auto& d=result.diagnostics_;d.charged_work_units_upper_bound=p.driver_work_units;
    const auto worker=[&](U owned,U control) {
        return add(owned,add(p.borrowed_local_numerical_bytes,
            add(p.borrowed_basis_active_numeric_bytes,add(control,extra(live)))));
    };
    const auto observe=[&](U owned,U observed_worker) {
        if(owned>p.peak_owned_numerical_bytes || observed_worker>p.per_worker_inventoried_bytes)
            throw std::logic_error("Gaussian density Gram actual lifetime exceeds enclosing admission");
        d.maximum_observed_owned_numerical_bytes=std::max(d.maximum_observed_owned_numerical_bytes,owned);
        d.maximum_observed_per_worker_inventoried_bytes=std::max(d.maximum_observed_per_worker_inventoried_bytes,observed_worker);
    };
    const auto reserve=[&](U work,U candidates,U images=0) {
        limit(add(d.charged_work_units_upper_bound,work),p.work_units,"Gaussian density Gram remaining work cap exceeded");
        limit(add(d.reciprocal_candidate_evaluations,candidates),p.reciprocal_candidate_evaluations_upper_bound,
            "Gaussian density Gram remaining reciprocal cap exceeded");
        limit(add(d.image_candidate_evaluations,images),p.image_candidate_evaluations_upper_bound,
            "Gaussian density Gram remaining image cap exceeded");
    };
    auto metric_caps=caps.metric;
    metric_caps.maximum_per_replica_inventoried_bytes=std::min(metric_caps.maximum_per_replica_inventoried_bytes,p.per_worker_inventoried_bytes);
    metric_caps.maximum_node_inventoried_bytes=std::min(metric_caps.maximum_node_inventoried_bytes,p.required_node_memory_bytes);
    auto panel_caps=caps.panel;
    for(auto* c:{&panel_caps.resources,&panel_caps.tile.resources}) {
        c->maximum_per_replica_inventoried_bytes=std::min(c->maximum_per_replica_inventoried_bytes,p.per_worker_inventoried_bytes);
        c->maximum_node_inventoried_bytes=std::min(c->maximum_node_inventoried_bytes,p.required_node_memory_bytes);
    }
    auto metric_config=recipe.metric;
    metric_config.basis_verification_caps=exact_basis_caps(*context);
    Digest consumed("vibeqc.periodic.gaussian-density-gram.consumed");
    consumed.string(result.hf_);consumed.string(result.context_sha_);consumed.string(initial);
    selection_wire(consumed,selection);consumed.u64(p.factor_panels);
    const auto metric_live=[&](U sibling) {
        PeriodicGaussianMetricLiveInventory l;
        l.replicas_per_node=p.replicas_per_node;l.external_node_bytes=p.reference_base_node_bytes;
        l.other_retained_bytes_per_replica=add(live.other_retained_bytes_per_worker,
            add(p.factor_phase_retained_bytes,add(sibling,
                add(p.borrowed_local_numerical_bytes,add(p.macro_control_storage_bytes,sibling_controls)))));
        l.other_transient_bytes_per_replica=live.other_transient_bytes_per_worker;
        l.fixed_backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
        return l;
    };
    const auto make_source=[&](U q,U sibling) {
        reserve(source_work(recipe.source_caps.maximum_candidate_evaluations),recipe.source_caps.maximum_candidate_evaluations);
        auto source=make_periodic_gaussian_reciprocal_source(context,q,recipe.source_caps);
        if(source.context_handle()!=context || source.q_index()!=q
            || source.conjugate_q_index()!=real::negative_cell(q,state.mesh()))
            throw std::logic_error("Gaussian density Gram actual reciprocal source differs");
        const U count=source.inventory().candidate_evaluations_performed;
        d.reciprocal_candidate_evaluations=add(d.reciprocal_candidate_evaluations,count);
        d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,source_work(count));
        const U controls=add(sibling_controls,source.inventory().inventoried_fixed_storage_bytes);
        limit(controls,caps.maximum_leaf_control_bytes,"Gaussian density Gram source control cap exceeded");
        observe(add(p.factor_phase_retained_bytes,sibling),
            worker(add(p.factor_phase_retained_bytes,sibling),add(p.macro_control_storage_bytes,controls)));
        ++d.completed_source_count;return source;
    };
    const auto make_w=[&](const PeriodicGaussianReciprocalSource& source,U sibling) {
        const auto l=metric_live(sibling);
        reserve(metric_caps.maximum_work_units,metric_caps.maximum_candidate_evaluations);
        auto metric=build_periodic_gaussian_reciprocal_metric(source,ao,auxiliary,metric_config,l,metric_caps);
        limit(add(sibling_controls,metric.plan().fixed_inventoried_object_bytes),caps.maximum_leaf_control_bytes,
            "Gaussian density Gram metric control cap exceeded");
        if(metric.context_handle()!=context || metric.q_index()!=source.q_index()
            || metric.source_identity_sha256()!=source.source_identity_sha256()
            || metric.conjugate_source_identity_sha256()!=source.conjugate_source_identity_sha256()
            || metric.plan().matrix_bytes!=p.whitener_bytes)
            throw std::logic_error("Gaussian density Gram actual metric source differs");
        d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,metric.plan().work_units_upper_bound);
        d.reciprocal_candidate_evaluations=add(d.reciprocal_candidate_evaluations,metric.plan().candidate_evaluations);
        observe(add(p.factor_phase_retained_bytes,add(sibling,metric.plan().owned_numeric_peak_bytes)),metric.plan().per_replica_inventoried_bytes);
        consumed.u64(source.q_index());consumed.string(source.source_identity_sha256());
        consumed.string(source.conjugate_source_identity_sha256());consumed.string(metric.payload_identity_sha256());
        ++d.completed_metric_count;
        reserve(metric_caps.maximum_work_units,0);
        auto w=factorize_periodic_gaussian_metric(std::move(metric),recipe.metric.whitener_column_block,l,metric_caps);
        limit(add(sibling_controls,w.plan().fixed_inventoried_object_bytes),caps.maximum_leaf_control_bytes,
            "Gaussian density Gram whitener control cap exceeded");
        if(w.context_handle()!=context || w.q_index()!=source.q_index()
            || w.source_identity_sha256()!=source.source_identity_sha256()
            || w.conjugate_source_identity_sha256()!=source.conjugate_source_identity_sha256()
            || mul(16,w.matrix_row_major().size())!=p.whitener_bytes)
            throw std::logic_error("Gaussian density Gram actual whitener source differs");
        d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,w.plan().work_units_upper_bound);
        observe(add(p.factor_phase_retained_bytes,add(sibling,w.plan().owned_numeric_peak_bytes)),w.plan().per_replica_inventoried_bytes);
        consumed.string(w.payload_identity_sha256());++d.completed_whitening_count;return w;
    };
    const auto make_panel=[&](const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        U begin,U count,Rectangle rectangle,U sibling,U prior) {
        if(d.completed_factor_panels>=p.factor_panels)
            throw std::length_error("Gaussian density Gram factor-panel count exhausted");
        reserve(panel_caps.resources.maximum_work_units,panel_caps.resources.maximum_candidate_evaluations,
            panel_caps.maximum_image_candidate_evaluations);
        Live l=live;
        l.other_retained_bytes_per_worker=add(l.other_retained_bytes_per_worker,
            add(p.factor_phase_retained_bytes,add(sibling,add(prior,add(p.macro_control_storage_bytes,sibling_controls)))));
        const auto leaf=plan_periodic_gaussian_local_orbital_factor_panel(hf,ref,source,w,wannier,domain,space,basis,
            begin,count,rectangle,config.panel,l,panel_caps);
        limit(add(sibling_controls,add(leaf.macro_fixed_object_bytes,leaf.tile_fixed_object_bytes)),
            caps.maximum_leaf_control_bytes,"Gaussian density Gram panel control cap exceeded");
        if(leaf.driver_owned_numerical_bytes>p.panel_driver_owned_upper_bytes
            || leaf.retained_output_bytes>p.retained_panel_upper_bytes
            || leaf.per_replica_inventoried_bytes>p.per_worker_inventoried_bytes
            || leaf.required_node_memory_bytes>p.required_node_memory_bytes)
            throw std::logic_error("Gaussian density Gram exact rectangle exceeds count-only admission");
        auto panel=build_periodic_gaussian_local_orbital_factor_panel(hf,ref,source,w,ao,auxiliary,wannier,gauges,gauge_count,
            domain,space,basis,begin,count,rectangle,config.panel,l,panel_caps);
        const auto s=panel.selection();const auto& pd=panel.diagnostics();
        if(panel.context_handle()!=context || panel.state_handle()!=ref.state_handle()
            || panel.q_index()!=source.q_index() || panel.auxiliary_begin()!=begin || panel.auxiliary_count()!=count
            || panel.orbital_count()!=p.orbital_count || s.left_begin!=rectangle.left_begin || s.left_count!=rectangle.left_count
            || s.right_begin!=rectangle.right_begin || s.right_count!=rectangle.right_count
            || panel.basis_certificate_identity_sha256()!=result.basis_
            || panel.local_basis_identity_sha256()!=basis.local_basis_identity_sha256()
            || panel.hf_reference_source_identity_sha256()!=result.hf_
            || panel.source_identity_sha256()!=source.source_identity_sha256()
            || panel.conjugate_source_identity_sha256()!=source.conjugate_source_identity_sha256()
            || panel.whitener_payload_identity_sha256()!=w.payload_identity_sha256())
            throw std::logic_error("Gaussian density Gram returned rectangle source or orientation differs");
        sha(panel.identity_sha256());sha(panel.payload_sha256());sha(panel.consumed_tiles_identity_sha256());
        d.charged_work_units_upper_bound=add(d.charged_work_units_upper_bound,pd.charged_work_units_upper_bound);
        d.reciprocal_candidate_evaluations=add(d.reciprocal_candidate_evaluations,pd.reciprocal_candidate_evaluations);
        d.image_candidate_evaluations=add(d.image_candidate_evaluations,pd.image_candidate_evaluations);
        d.completed_tile_calls=add(d.completed_tile_calls,pd.completed_tile_calls);
        observe(add(p.factor_phase_retained_bytes,add(p.whitener_bytes,
            add(sibling,add(prior,pd.maximum_observed_owned_numerical_bytes)))),pd.maximum_observed_per_replica_inventoried_bytes);
        consumed.u64(source.q_index());consumed.u64(begin);consumed.u64(count);
        for(U x:{rectangle.left_begin,rectangle.left_count,rectangle.right_begin,rectangle.right_count}) consumed.u64(x);
        consumed.string(panel.identity_sha256());consumed.string(panel.payload_sha256());consumed.string(panel.consumed_tiles_identity_sha256());
        ++d.completed_factor_panels;return panel;
    };
    {
        const U L=selection.left.count,R=selection.right.count,LR=p.output_elements,D=p.selected_density_count;
        std::vector<double> accum(static_cast<std::size_t>(p.integral_accumulator_bytes/8));
        std::vector<double> rows(static_cast<std::size_t>(p.row_slab_bytes/8));
        std::vector<double> norms(static_cast<std::size_t>(p.norm_workspace_bytes/8));
        observe(p.factor_phase_retained_bytes,worker(p.factor_phase_retained_bytes,p.control_storage_reservation_bytes));
        auto* original=norms.data();auto* reverse_norm=original+D;auto* covariance=reverse_norm+D;
        for(U q=0;q<p.n_cells;++q) {
            const U qbar=real::negative_cell(q,state.mesh());if(q>qbar) continue;
            const auto source=make_source(q,0);const auto w=make_w(source,0);
            std::optional<PeriodicGaussianReciprocalSource> opposite;
            std::optional<PeriodicGaussianMetricWhitener> opposite_w;
            if(q!=qbar) {
                opposite.emplace(make_source(qbar,p.whitener_bytes));opposite_w.emplace(make_w(*opposite,p.whitener_bytes));
                if(source.conjugate_source_identity_sha256()!=opposite->source_identity_sha256()
                    || opposite->conjugate_source_identity_sha256()!=source.source_identity_sha256())
                    throw std::logic_error("Gaussian density Gram original opposite sources do not match");
            }
            const U sibling=opposite?p.whitener_bytes:0;
            for(U begin=0;begin<p.n_auxiliary;begin+=std::min(config.auxiliary_block,p.n_auxiliary-begin)) {
                const U count=std::min(config.auxiliary_block,p.n_auxiliary-begin);
                const auto fill_range=[&](Range range,U offset) {
                    auto density=decode(range.begin,p.orbital_count,p.common_density_count);
                    U remaining=range.count,completed=0,runs=0;
                    while(remaining) {
                        const U length=std::min(remaining,p.orbital_count-density[1]);
                        const Rectangle forward{density[0],1,density[1],length};
                        const Rectangle reversed{density[1],length,density[0],1};
                        const auto panel=make_panel(source,w,begin,count,forward,sibling,0);
                        std::optional<Panel> opposite_reverse;
                        {
                            const auto reverse_panel=make_panel(source,w,begin,count,reversed,sibling,panel.memory().retained_output_bytes);
                            for(U a=0;a<count;++a) for(U z=0;z<length;++z) {
                                const U index=add(offset,add(completed,z));
                                const C x=panel.element(a,0,z),xr=reverse_panel.element(a,z,0);
                                reversal(x,xr,options,d.real_projection);
                                real::norm_value(original[index],x);real::norm_difference(reverse_norm[index],xr,x);
                                if(opposite) {
                                    constexpr double sqrt_two=0x1.6a09e667f3bcdp+0;
                                    rows[a*D+index]=real::finite(sqrt_two*x.real());
                                    rows[(count+a)*D+index]=real::finite(sqrt_two*x.imag());
                                } else {
                                    for(C value:{x,xr}) {
                                        d.real_projection.maximum_self_q_imaginary_magnitude=std::max(
                                            d.real_projection.maximum_self_q_imaginary_magnitude,std::abs(value.imag()));
                                        if(!real::within(value,C(value.real(),0),options.self_q_absolute_tolerance,options.self_q_relative_tolerance))
                                            throw std::invalid_argument("Gaussian density Gram selected self-q imaginary lane exceeds tolerance");
                                    }
                                    real::norm_lane(covariance[index],std::abs(x.imag()));rows[a*D+index]=x.real();
                                }
                            }
                            if(opposite) {
                                // Build reversed qbar BEFORE forward qbar: its
                                // actual conjugacy is audited while reverse q
                                // still exists, with at most three panels live.
                                opposite_reverse.emplace(make_panel(*opposite,*opposite_w,begin,count,reversed,sibling,
                                    add(panel.memory().retained_output_bytes,reverse_panel.memory().retained_output_bytes)));
                                for(U a=0;a<count;++a) for(U z=0;z<length;++z)
                                    conjugacy(reverse_panel.element(a,z,0),opposite_reverse->element(a,z,0),options,d.real_projection);
                            }
                        } // Reverse q is released before forward qbar allocates.
                        if(opposite) {
                            const auto opposite_panel=make_panel(*opposite,*opposite_w,begin,count,forward,sibling,
                                add(panel.memory().retained_output_bytes,opposite_reverse->memory().retained_output_bytes));
                            for(U a=0;a<count;++a) for(U z=0;z<length;++z) {
                                const U index=add(offset,add(completed,z));
                                const C x=panel.element(a,0,z),y=opposite_panel.element(a,0,z),yr=opposite_reverse->element(a,z,0);
                                reversal(y,yr,options,d.real_projection);conjugacy(x,y,options,d.real_projection);
                                real::norm_value(original[index],y);real::norm_difference(reverse_norm[index],yr,y);
                                real::norm_difference(covariance[index],y,std::conj(x));
                            }
                        }
                        completed=add(completed,length);remaining-=length;++runs;
                        if(remaining) {++density[0];density[1]=density[0];}
                    }
                    const auto expected=run_shape(range,p.orbital_count,p.common_density_count);
                    if(completed!=range.count || runs!=expected.count)
                        throw std::logic_error("Gaussian density Gram packed-range traversal differs from admission");
                };
                fill_range(selection.left,0);fill_range(selection.right,L);
                for(U row=0;row<mul(opposite?2:1,count);++row) {
                    const auto* values=rows.data()+row*D;
                    for(U i=0;i<L;++i) for(U j=0;j<R;++j) {
                        const U x=i*R+j;
                        include_product(values[i],values[L+j],result.values_[x],accum[x],accum[LR+x],accum[2*LR+x]);
                    }
                }
            }
        }
        // Arithmetic-only descriptor for the selected norm vectors. It is
        // not a physical provider or a certificate of unrequested densities.
        PeriodicCorrelationRealLocalProviderMemoryPlan selected;
        selected.n_cells=p.n_cells;selected.n_auxiliary=p.n_auxiliary;selected.density_count=D;
        selected.row_count=p.row_count;selected.self_inverse_q_count=p.self_inverse_q_count;
        d.real_projection.factor_panels_built=d.completed_factor_panels;
        real::finish_provider_norms(selected,d.real_projection,norms.data());
        real::finish_provider_projection(selected,options,d.real_projection);
        for(U x=0;x<LR;++x) {
            result.values_[x]=real::finite(result.values_[x]+accum[x]);
            const double error=std::max(real::difference_up(result.values_[x],accum[LR+x]),
                real::difference_up(result.values_[x],accum[2*LR+x]));
            d.maximum_integral_roundoff_error=std::max(d.maximum_integral_roundoff_error,error);
            if(error>options.maximum_scalar_roundoff_error)
                throw std::invalid_argument("Gaussian density Gram streamed integral roundoff exceeds its budget");
        }
    } // Only the Gram output and fixed metadata survive these workspaces.
    if(d.completed_source_count!=p.source_factory_calls || d.completed_metric_count!=p.metric_calls
        || d.completed_whitening_count!=p.whitening_calls || d.completed_factor_panels!=p.factor_panels
        || d.completed_tile_calls!=p.tile_calls || d.charged_work_units_upper_bound>p.work_units
        || d.reciprocal_candidate_evaluations>p.reciprocal_candidate_evaluations_upper_bound
        || d.image_candidate_evaluations>p.image_candidate_evaluations_upper_bound)
        throw std::logic_error("Gaussian density Gram completed traversal differs from admission");
    if(input_payload(hf,ref,ao,auxiliary,wannier,gauges,gauge_count,domain,space,basis)!=initial)
        throw std::invalid_argument("Gaussian density Gram original borrowed payload changed during construction");
    result.consumed_=consumed.finish();
    Digest payload("vibeqc.periodic.gaussian-density-gram.payload");selection_wire(payload,selection);
    payload.u64(p.orbital_count);for(double value:result.values_) payload.real(value);result.payload_=payload.finish();
    Digest identity("vibeqc.periodic.gaussian-density-gram.identity");
    for(const auto* value:std::array<const std::string*,7>{&result.hf_,&result.context_sha_,&result.basis_,
        &initial,&result.consumed_,&result.payload_,&ref.dimensions().allocation_identity}) {sha(*value);identity.string(*value);}
    selection_wire(identity,selection);identity.u64(config.auxiliary_block);identity.u64(config.panel.ao_pair_block);
    option_wire(identity,options);identity.string(policy);identity.string(real::kFloatPolicy);
    result.identity_=identity.finish();result.state_=ref.state_handle();result.context_=context;return result;
}

} // namespace vibeqc
