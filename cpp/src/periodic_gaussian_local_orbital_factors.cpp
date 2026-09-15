#include "vibeqc/periodic_gaussian_local_orbital_factors.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace local = periodic_correlation_local_detail;
namespace real = periodic_correlation_real_local_detail;
using I = std::uint64_t;
using C = std::complex<double>;
using Plan = PeriodicGaussianLocalOrbitalFactorPlan;
using Selection = PeriodicGaussianLocalOrbitalFactorSelection;
using Config = PeriodicGaussianLocalOrbitalFactorConfig;
using Live = PeriodicGaussianLocalOrbitalFactorLiveInventory;
using Caps = PeriodicGaussianLocalOrbitalFactorCaps;
using Panel = PeriodicGaussianLocalOrbitalFactorPanel;
using Digest = local::Digest;
using real::add;
using real::mul;
void limit(I n,I cap,const char* message) { if(n>cap) throw std::length_error(message); }
void positive(I n) { if(!n) throw std::invalid_argument("Gaussian local factors require positive explicit caps"); }
void resources(const PeriodicGaussianMetricCaps& c) {
    for(auto n:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) positive(n);
}
I ceil_div(I a,I b) { return a/b+(a%b!=0); }
Selection full_selection(const PeriodicCorrelationRealLocalBasis& basis) {
    const auto m=basis.memory().orbital_count;
    return {0,m,0,m};
}
bool full_selection(const Selection& s,I m) {
    return s.left_begin==0 && s.left_count==m && s.right_begin==0 && s.right_count==m;
}
void selection_wire(Digest& h,const Plan& p) {
    if(full_selection(p.selection,p.orbital_count)) return;
    h.string("rectangular-common-orbital-ranges-v1");
    for(auto n:{p.selection.left_begin,p.selection.left_count,p.selection.right_begin,p.selection.right_count}) h.u64(n);
}
I selected_occupied(I begin,I count,I occupied) {
    return begin<occupied ? std::min(count,occupied-begin) : 0;
}
I local_owners(const Plan& p) {
    return add(p.caller_gauge_bytes,add(p.live_wannier_bytes,
        add(p.live_domain_bytes,add(p.live_space_bytes,p.live_basis_bytes))));
}
void extent(I bytes) {
    real::extent(bytes);
    limit(bytes/16,std::vector<C>().max_size(),"Gaussian local factor vector exceeds addressable extent");
}
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& c) {
    const auto& v=c.inventory(); const auto& a=v.ao; const auto& b=v.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),c.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),
        add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),v.combined_borrowed_active_numeric_bytes,v.work_units_upper_bound};
}
void objects(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis) {
    real::float_environment();
    if(!hf.converged() || !hf.context_handle() || !hf.matched_finite_gaussian_hf_source())
        throw std::invalid_argument("Gaussian local factors require an actual converged Gaussian HF result");
    local::validate_wannier(reference,wannier); local::validate_pao(reference,domain,space);
    if(hf.state_handle().get()!=reference.state_handle().get()
        || basis.state_handle().get()!=reference.state_handle().get()
        || basis.allocation_identity()!=reference.dimensions().allocation_identity)
        throw std::invalid_argument("Gaussian local factor HF/reference/local-basis owners disagree");
    if(source.context_handle().get()!=hf.context_handle().get() || w.context_handle().get()!=hf.context_handle().get())
        throw std::invalid_argument("Gaussian local factor source or whitener context differs from actual HF");
    const auto& c=*hf.context_handle(); const auto& recipe=hf.plan().fock.config;
    const auto a=c.inventory().auxiliary.function_count;
    if(reference.dimensions().n_auxiliary!=a || reference.state().n_basis()!=c.inventory().ao.function_count
        || reference.state().mesh()!=c.mesh().mesh() || reference.state().is_shift()!=c.mesh().is_shift()
        || source.q_index()!=w.q_index() || source.conjugate_q_index()!=w.conjugate_q_index()
        || source.source_identity_sha256()!=w.source_identity_sha256()
        || source.conjugate_source_identity_sha256()!=w.conjugate_source_identity_sha256()
        || w.plan().phase!=PeriodicGaussianMetricPhase::PrincipalWhitening
        || w.plan().n_auxiliary!=a || w.matrix_row_major().size()!=mul(a,a)
        || w.diagnostics().n_auxiliary!=a || !w.diagnostics().retained_rank
        || w.plan().config.reciprocal_block!=recipe.metric.reciprocal_block
        || w.plan().config.whitener_column_block!=recipe.metric.whitener_column_block)
        throw std::invalid_argument("Gaussian local factor source/whitener recipe or shape differs from actual HF");
    const auto& b=basis.memory();
    if(b.n_cells!=reference.state().n_kpoints() || b.n_basis!=reference.state().n_basis()
        || !b.occupied_count || !b.virtual_count || b.orbital_count!=add(b.occupied_count,b.virtual_count)
        || b.occupied_count>mul(b.n_cells,reference.state().n_correlated_occupied())
        || b.virtual_count!=basis.virtual_selection().count)
        throw std::invalid_argument("Gaussian local factor real-basis shape mismatch");
    (void) basis.occupied_indices_data(); (void) basis.f_oo_data();
}
void resource_wire(Digest& h,const PeriodicGaussianMetricCaps& c) {
    for(auto n:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) h.u64(n);
}
void plan_wire(Digest& h,const Plan& p) {
    for(auto n:{p.n_cells,p.n_basis,p.n_auxiliary,p.occupied_count,p.virtual_count,p.orbital_count,
        p.q_index,p.auxiliary_begin,p.auxiliary_count,p.ao_pair_block,p.tile_calls,
        p.retained_factor_bytes,p.retained_index_bytes,p.retained_output_bytes,p.compensation_bytes,
        p.coefficient_panel_bytes,p.coefficient_scratch_bytes,p.driver_owned_numerical_bytes,
        p.maximum_tile_owned_numerical_bytes,p.peak_owned_numerical_bytes,p.resident_whitener_bytes,
        p.borrowed_basis_active_numeric_bytes,p.caller_gauge_bytes,p.live_wannier_bytes,p.live_domain_bytes,
        p.live_space_bytes,p.live_basis_bytes,p.basis_index_alias_bytes,p.state_resident_bytes,
        p.macro_fixed_object_bytes,p.tile_fixed_object_bytes,p.replicas_per_node,p.reference_base_node_bytes,
        p.per_replica_inventoried_bytes,p.required_node_memory_bytes,p.reciprocal_candidate_evaluations,
        p.image_candidate_evaluations_upper_bound,p.coefficient_work_units,p.contraction_term_count,
        p.driver_work_units,p.work_units}) h.u64(n);
    h.u64(p.config.ao_pair_block);
    for(auto n:{p.live.other_retained_bytes_per_worker,p.live.other_transient_bytes_per_worker,
        p.live.fixed_backend_margin_bytes_per_worker}) h.u64(n);
    resource_wire(h,p.caps.resources); resource_wire(h,p.caps.tile.resources);
    h.u64(p.caps.tile.maximum_image_candidates); h.u64(p.caps.maximum_tile_calls);
    h.u64(p.caps.maximum_image_candidate_evaluations); h.u64(p.tile_config.reciprocal_block);
    const auto& c=p.tile_config.basis_verification_caps;
    for(auto n:{c.maximum_context_storage_bytes,c.maximum_kpoint_count,c.maximum_shell_count,
        c.maximum_contraction_count,c.maximum_primitive_numeric_lanes,c.maximum_basis_content_wire_bytes,
        c.maximum_borrowed_active_numeric_bytes,c.maximum_work_units}) h.u64(n);
    selection_wire(h,p);
}
} // namespace

Plan plan_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    I begin,I count,const Config& config,const Live& live,const Caps& caps) {
    return plan_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,wannier,domain,space,basis,
        begin,count,full_selection(basis),config,live,caps);
}

Plan plan_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    I begin,I count,const Selection& selection,const Config& config,const Live& live,const Caps& caps) {
    resources(caps.resources); resources(caps.tile.resources);
    positive(caps.maximum_tile_calls); positive(caps.maximum_image_candidate_evaluations);
    positive(caps.tile.maximum_image_candidates); positive(live.fixed_backend_margin_bytes_per_worker);
    objects(hf,reference,source,w,wannier,domain,space,basis);
    const auto& context=*hf.context_handle(); const auto& state=reference.state();
    const auto& dims=reference.dimensions(); const auto& budget=reference.budget();
    Plan p; p.selection=selection; p.config=config; p.live=live; p.caps=caps; p.tile_config=hf.plan().fock.config.tile;
    p.n_cells=state.n_kpoints(); p.n_basis=state.n_basis(); p.n_auxiliary=dims.n_auxiliary;
    p.occupied_count=basis.memory().occupied_count; p.virtual_count=basis.memory().virtual_count;
    p.orbital_count=add(p.occupied_count,p.virtual_count); p.q_index=source.q_index();
    p.auxiliary_begin=begin; p.auxiliary_count=count; p.ao_pair_block=config.ao_pair_block;
    const auto n=p.n_basis,m=p.orbital_count,o=p.occupied_count,v=p.virtual_count,nn=mul(n,n);
    const auto selected=basis.virtual_selection();
    if(!count || begin>=p.n_auxiliary || count>p.n_auxiliary-begin || !config.ao_pair_block || config.ao_pair_block>nn
        || selected.begin>=space.retained_dimension() || selected.count>space.retained_dimension()-selected.begin
        || selected.translation_cell>=p.n_cells)
        throw std::invalid_argument("Gaussian local factor auxiliary/virtual slice or AO block is invalid");
    const auto& s=p.selection;
    if(!s.left_count || !s.right_count || s.left_begin>=m || s.right_begin>=m
        || s.left_count>m-s.left_begin || s.right_count>m-s.right_begin)
        throw std::invalid_argument("Gaussian local factor common-orbital rectangle is invalid");
    const auto left=s.left_count,right=s.right_count;
    p.tile_calls=mul(p.n_cells,ceil_div(nn,config.ao_pair_block));
    p.retained_factor_bytes=mul(16,mul(count,mul(left,right))); p.retained_index_bytes=mul(16,o);
    p.retained_output_bytes=add(p.retained_factor_bytes,p.retained_index_bytes);
    p.compensation_bytes=p.retained_factor_bytes; p.coefficient_panel_bytes=mul(16,mul(n,add(left,right)));
    p.coefficient_scratch_bytes=mul(32,n);
    p.driver_owned_numerical_bytes=add(p.retained_output_bytes,add(p.compensation_bytes,
        add(p.coefficient_panel_bytes,p.coefficient_scratch_bytes)));
    p.maximum_tile_owned_numerical_bytes=caps.tile.resources.maximum_owned_numeric_bytes;
    p.peak_owned_numerical_bytes=add(p.driver_owned_numerical_bytes,p.maximum_tile_owned_numerical_bytes);
    p.resident_whitener_bytes=mul(16,mul(p.n_auxiliary,p.n_auxiliary));
    p.borrowed_basis_active_numeric_bytes=context.inventory().combined_borrowed_active_numeric_bytes;
    const auto wm=plan_periodic_correlation_wannier(state.mesh(),n,state.n_correlated_occupied());
    p.caller_gauge_bytes=wm.caller_gauge_bytes; p.live_wannier_bytes=wm.retained_coefficient_bytes;
    const auto d=domain.domain_dimension(),r=space.retained_dimension();
    p.live_domain_bytes=add(mul(16,d),mul(32,mul(d,d)));
    p.live_space_bytes=add(mul(16,mul(d,r)),mul(8,add(d,r)));
    p.live_basis_bytes=add(mul(16,o),mul(8,add(add(mul(o,o),mul(v,v)),mul(o,v))));
    p.basis_index_alias_bytes=p.retained_index_bytes; p.state_resident_bytes=reference.state_resident_bytes();
    if(p.live_wannier_bytes!=wannier.memory().retained_coefficient_bytes
        || p.live_domain_bytes!=add(domain.memory().retained_domain_index_bytes,domain.memory().retained_matrix_bytes)
        || p.live_space_bytes!=space.memory().output_numerical_bytes || p.live_basis_bytes!=basis.memory().retained_output_bytes
        || p.state_resident_bytes!=state.resident_bytes() || dims.external_bytes<p.state_resident_bytes)
        throw std::logic_error("Gaussian local factor live owner inventory disagrees with sealed payloads");
    p.macro_fixed_object_bytes=sizeof(Panel)+sizeof(Plan)+sizeof(Selection)+sizeof(Config)+sizeof(Live)+sizeof(Caps)
        +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicCorrelationAdmittedReference)
        +sizeof(PeriodicCorrelationWannier)+sizeof(PeriodicCorrelationPAODomain)
        +sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis);
    p.tile_fixed_object_bytes=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)
        +sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianThreeCenterPlan)
        +sizeof(PeriodicGaussianThreeCenterTile)+sizeof(PeriodicSystem);
    p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank); positive(p.replicas_per_node);
    p.reference_base_node_bytes=add(add(dims.external_bytes,dims.shared_bytes),
        mul(budget.mpi_ranks,add(dims.per_rank_bytes,dims.localization_window_bytes_per_rank)));
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.resident_whitener_bytes,
        add(p.borrowed_basis_active_numeric_bytes,add(local_owners(p),add(p.macro_fixed_object_bytes,
        add(p.tile_fixed_object_bytes,add(live.other_retained_bytes_per_worker,
        add(live.other_transient_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))))))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_replica_inventoried_bytes));
    const auto candidate_per_tile=mul(2,source.candidate_count());
    p.reciprocal_candidate_evaluations=mul(p.tile_calls,candidate_per_tile);
    const auto accepted=source.accepted_vector_count();
    if(!accepted || !p.tile_config.reciprocal_block) throw std::invalid_argument("Gaussian local factor finite source is empty");
    const auto images_per_tile=mul(caps.tile.maximum_image_candidates,
        add(add(1,accepted),ceil_div(accepted,p.tile_config.reciprocal_block)));
    p.image_candidate_evaluations_upper_bound=mul(p.tile_calls,images_per_tile);
    const auto virtual_work=add(mul(n,d),add(nn,add(mul(mul(2,n),state.n_effective_orbitals()),mul(4,n))));
    const auto selected_o=add(selected_occupied(s.left_begin,left,o),selected_occupied(s.right_begin,right,o));
    const auto selected_v=add(left,right)-selected_o;
    p.coefficient_work_units=mul(128,mul(p.n_cells,
        add(mul(selected_v,virtual_work),mul(selected_o,mul(n,add(state.n_effective_orbitals(),1))))));
    p.contraction_term_count=mul(mul(p.n_cells,nn),mul(count,mul(left,right)));
    p.driver_work_units=context.inventory().work_units_upper_bound;
    for(auto work:{p.coefficient_work_units,mul(128,p.contraction_term_count),mul(256,p.caller_gauge_bytes),
        mul(128,mul(o,o)),mul(128,add(p.retained_output_bytes,mul(n,m))),mul(4096,p.tile_calls),I{65536}})
        p.driver_work_units=add(p.driver_work_units,work);
    p.work_units=add(p.driver_work_units,mul(p.tile_calls,caps.tile.resources.maximum_work_units));
    for(auto bytes:{p.retained_output_bytes,p.compensation_bytes,p.coefficient_panel_bytes,p.coefficient_scratch_bytes,
        p.caller_gauge_bytes,mul(80,p.tile_calls)}) extent(bytes);
    limit(mul(2,o),std::vector<I>().max_size(),"Gaussian local factor index extent exceeds cap");
    limit(candidate_per_tile,caps.tile.resources.maximum_candidate_evaluations,"Gaussian local factor leaf reciprocal cap exceeded");
    limit(p.peak_owned_numerical_bytes,caps.resources.maximum_owned_numeric_bytes,"Gaussian local factor owned memory cap exceeded");
    limit(p.per_replica_inventoried_bytes,caps.resources.maximum_per_replica_inventoried_bytes,"Gaussian local factor replica cap exceeded");
    limit(p.required_node_memory_bytes,caps.resources.maximum_node_inventoried_bytes,"Gaussian local factor node cap exceeded");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian local factor admitted reference node cap exceeded");
    limit(p.tile_calls,caps.maximum_tile_calls,"Gaussian local factor tile calls cap exceeded");
    limit(p.reciprocal_candidate_evaluations,caps.resources.maximum_candidate_evaluations,"Gaussian local factor reciprocal cap exceeded");
    limit(p.image_candidate_evaluations_upper_bound,caps.maximum_image_candidate_evaluations,"Gaussian local factor image cap exceeded");
    limit(p.work_units,caps.resources.maximum_work_units,"Gaussian local factor work cap exceeded");
    Digest digest("vibeqc.periodic.gaussian-local-orbital-factors.plan");
    for(const auto& s:{hf.reference_source_identity_sha256(),source.source_identity_sha256(),
        source.conjugate_source_identity_sha256(),w.payload_identity_sha256(),basis.identity_sha256(),
        reference.dimensions().allocation_identity,wannier.wannier_identity_sha256(),
        domain.pao_domain_identity_sha256(),space.pao_space_identity_sha256()}) digest.string(s);
    plan_wire(digest,p); const auto identity=digest.finish();
    std::copy(identity.begin(),identity.end(),p.identity_ascii.begin()); return p;
}

const C* Panel::data() const {
    if(!context_ || !state_ || values_.size()!=memory_.retained_factor_bytes/16
        || indices_.size()!=mul(2,memory_.occupied_count))
        throw std::logic_error("Gaussian local factor panel is moved or malformed");
    return values_.data();
}
PeriodicCorrelationPlacedOccupied Panel::occupied(std::size_t i) const {
    (void) data();
    if(i>=memory_.occupied_count) throw std::out_of_range("Gaussian local occupied index out of range");
    return {indices_[2*i],indices_[2*i+1]};
}
C Panel::element(std::size_t p,std::size_t l,std::size_t r) const {
    const auto& s=memory_.selection;
    if(p>=auxiliary_count() || l>=s.left_count || r>=s.right_count)
        throw std::out_of_range("Gaussian local factor element out of range");
    return data()[(p*s.left_count+l)*s.right_count+r];
}

Panel build_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
    const C* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    I begin,I count,const Config& config,const Live& live,const Caps& caps) {
    return build_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,ao,auxiliary,wannier,
        gauges,gauge_count,domain,space,basis,begin,count,full_selection(basis),config,live,caps);
}

Panel build_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
    const C* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    I begin,I count,const Selection& selection,const Config& config,const Live& live,const Caps& caps) {
    const auto p=plan_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,wannier,domain,space,basis,
        begin,count,selection,config,live,caps);
    const auto context=hf.context_handle(); const auto& state=reference.state();
    context->verify_bases(ao,auxiliary,exact_basis_caps(*context));
    const auto* indices=basis.occupied_indices_data();
    real::indices(indices,mul(2,p.occupied_count),p.occupied_count,state);
    const auto gauge_identity=local::validate_gauges(wannier,gauges,gauge_count);
    const auto selected=basis.virtual_selection();
    const auto basis_identity=real::local_basis_identity(reference,wannier,gauge_identity,domain,space,
        indices,p.occupied_count,selected);
    if(basis_identity!=basis.local_basis_identity_sha256())
        throw std::invalid_argument("Gaussian local factor basis certificate does not match actual orbital inputs");
    Panel result; result.memory_=p; result.context_=context; result.state_=reference.state_handle(); result.virtual_=selected;
    result.hf_=hf.reference_source_identity_sha256(); result.basis_=basis_identity; result.certificate_=basis.identity_sha256();
    result.source_=source.source_identity_sha256(); result.opposite_=source.conjugate_source_identity_sha256();
    result.whitener_=w.payload_identity_sha256();
    result.indices_.assign(indices,indices+mul(2,p.occupied_count));
    result.values_.resize(static_cast<std::size_t>(p.retained_factor_bytes/16));
    auto& diagnostic=result.diagnostics_; diagnostic.charged_work_units_upper_bound=p.driver_work_units;
    Digest consumed("vibeqc.periodic.gaussian-local-orbital-factors.consumed-tiles");
    for(const auto& s:{result.hf_,result.source_,result.opposite_,result.whitener_}) consumed.string(s);
    consumed.u64(p.tile_calls);
    {
        std::vector<C> correction(result.values_.size());
        std::vector<C> coefficients(static_cast<std::size_t>(p.coefficient_panel_bytes/16));
        std::vector<C> scratch(static_cast<std::size_t>(p.coefficient_scratch_bytes/16));
        PeriodicGaussianMetricLiveInventory nested;
        nested.replicas_per_node=p.replicas_per_node; nested.external_node_bytes=p.reference_base_node_bytes;
        nested.other_retained_bytes_per_replica=add(p.driver_owned_numerical_bytes,
            add(local_owners(p),add(p.macro_fixed_object_bytes,live.other_retained_bytes_per_worker)));
        nested.other_transient_bytes_per_replica=live.other_transient_bytes_per_worker;
        nested.fixed_backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
        const auto n=p.n_basis,left=p.selection.left_count,right_count=p.selection.right_count,nn=mul(n,n);
        for(I k=0;k<p.n_cells;++k) {
            const auto ket=context->ket_index(k,source.q_index());
            for(unsigned side=0;side<2;++side) {
                auto* panel=coefficients.data()+(side==0 ? 0 : n*left);
                const auto momentum=side==0 ? k : ket;
                const auto first=side==0 ? p.selection.left_begin : p.selection.right_begin;
                const auto columns=side==0 ? left : right_count;
                const auto occupied=selected_occupied(first,columns,p.occupied_count);
                for(I i=0;i<occupied;++i) {
                    const auto index=first+i;
                    local::fill_occupied_column(state,gauges,indices[2*index],indices[2*index+1],momentum,panel+i*n);
                }
                if(columns>occupied) {
                    const auto virtual_begin=add(selected.begin,std::max(first,p.occupied_count)-p.occupied_count);
                    local::fill_virtual_columns(state,domain,space,virtual_begin,columns-occupied,
                        selected.translation_cell,momentum,panel+occupied*n,scratch.data());
                }
            }
            for(I pair=0;pair<nn;pair+=std::min(p.ao_pair_block,nn-pair)) {
                limit(add(diagnostic.charged_work_units_upper_bound,caps.tile.resources.maximum_work_units),
                    caps.resources.maximum_work_units,"Gaussian local factor remaining work cap exceeded");
                const auto pairs=std::min(p.ao_pair_block,nn-pair);
                const PeriodicGaussianThreeCenterSelection tile_selection{k,pair,pairs,begin,count};
                const auto tile=build_periodic_gaussian_three_center_tile(source,w,ao,auxiliary,tile_selection,p.tile_config,nested,caps.tile);
                const auto& descriptor=tile.descriptor(); const auto& memory=tile.plan();
                if(tile.context_handle().get()!=context.get() || descriptor.k_bra_index!=k || descriptor.k_ket_index!=ket
                    || descriptor.q_index!=source.q_index() || descriptor.ao_pair_begin!=pair || descriptor.ao_pair_count!=pairs
                    || descriptor.auxiliary_begin!=begin || descriptor.auxiliary_count!=count || descriptor.element_count!=mul(pairs,count)
                    || tile.source_identity_sha256()!=result.source_ || tile.conjugate_source_identity_sha256()!=result.opposite_
                    || tile.whitener_payload_identity_sha256()!=result.whitener_)
                    throw std::logic_error("Gaussian local factor native tile provenance/descriptor mismatch");
                diagnostic.charged_work_units_upper_bound=add(diagnostic.charged_work_units_upper_bound,memory.work_units_upper_bound);
                diagnostic.reciprocal_candidate_evaluations=add(diagnostic.reciprocal_candidate_evaluations,memory.reciprocal_candidate_evaluations);
                diagnostic.image_candidate_evaluations=add(diagnostic.image_candidate_evaluations,memory.image_candidate_evaluations);
                const auto owned=add(p.driver_owned_numerical_bytes,memory.owned_numeric_peak_bytes);
                if(owned>p.peak_owned_numerical_bytes || memory.per_replica_inventoried_bytes>p.per_replica_inventoried_bytes)
                    throw std::logic_error("Gaussian local factor nested lifetime exceeded admission");
                diagnostic.maximum_observed_owned_numerical_bytes=std::max(diagnostic.maximum_observed_owned_numerical_bytes,owned);
                diagnostic.maximum_observed_per_replica_inventoried_bytes=std::max(
                    diagnostic.maximum_observed_per_replica_inventoried_bytes,memory.per_replica_inventoried_bytes);
                const auto* right=coefficients.data()+n*left;
                for(I at=0;at<pairs;++at) {
                    const auto mu=(pair+at)/n,nu=(pair+at)%n;
                    for(I auxiliary_row=0;auxiliary_row<count;++auxiliary_row) {
                        const auto factor=tile.matrix_row_major()[auxiliary_row*pairs+at];
                        for(I l=0;l<left;++l) for(I r=0;r<right_count;++r) {
                            const auto output=(auxiliary_row*left+l)*right_count+r;
                            real::accumulate((std::conj(coefficients[l*n+mu])*right[r*n+nu])*factor,
                                result.values_[output],correction[output]);
                        }
                    }
                }
                consumed.u64(diagnostic.completed_tile_calls); consumed.string(tile.payload_identity_sha256());
                ++diagnostic.completed_tile_calls;
            }
        }
        const auto nk=static_cast<double>(p.n_cells); const auto normalization=(1.0/nk)/std::sqrt(nk);
        if(!std::isfinite(normalization) || normalization<=0) throw std::overflow_error("Gaussian local BvK normalization invalid");
        for(std::size_t at=0;at<result.values_.size();++at)
            result.values_[at]=real::finite((result.values_[at]+correction[at])*normalization);
    }
    if(diagnostic.completed_tile_calls!=p.tile_calls || diagnostic.reciprocal_candidate_evaluations!=p.reciprocal_candidate_evaluations
        || diagnostic.image_candidate_evaluations>p.image_candidate_evaluations_upper_bound
        || diagnostic.charged_work_units_upper_bound>p.work_units)
        throw std::logic_error("Gaussian local factor completed count exceeds admitted plan");
    (void) local::validate_gauges(wannier,gauges,gauge_count);
    result.consumed_=consumed.finish();
    Digest payload("vibeqc.periodic.gaussian-local-orbital-factors.payload");
    for(auto n:{p.q_index,p.auxiliary_begin,p.auxiliary_count,p.orbital_count}) payload.u64(n);
    selection_wire(payload,p);
    for(auto value:result.values_) payload.complex(value);
    result.payload_=payload.finish();
    Digest identity("vibeqc.periodic.gaussian-local-orbital-factors.identity");
    for(const auto& s:{result.hf_,context->source_context_identity_sha256(),result.basis_,result.certificate_,
        result.source_,result.opposite_,result.whitener_,result.consumed_,result.payload_}) identity.string(s);
    identity.string("actual-HF-context-recipe;original-direct-A;principal-original-aux;Nk^-3/2;all-ordered-complex;no-bitwise-HF-trace-claim");
    selection_wire(identity,p);
    result.identity_=identity.finish(); return result;
}

} // namespace vibeqc
