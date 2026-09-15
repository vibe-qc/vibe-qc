#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"

#include <algorithm>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include "periodic_gaussian_mixed_pair_factors_internal.hpp"
#include "periodic_correlation_real_local_internal.hpp"
#include "periodic_correlation_real_pao_embedding_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian mixed-pair factors require finite-aware non-fast-math arithmetic"
#endif

namespace vibeqc {
namespace {
namespace local = periodic_correlation_local_detail;
namespace real = periodic_correlation_real_local_detail;
using U = std::uint64_t;
using C = std::complex<double>;
using Frame = PeriodicGaussianPairPNOFrameView;
using Geometry = PeriodicGaussianPairPNOGeometryView;
using Storage = PeriodicGaussianMixedPairFactorStorage;
using Plan = PeriodicGaussianMixedPairFactorPlan;
using Panel = PeriodicGaussianMixedPairFactorPanel;
using Config = PeriodicGaussianLocalOrbitalFactorConfig;
using Live = PeriodicGaussianLocalOrbitalFactorLiveInventory;
using Caps = PeriodicGaussianLocalOrbitalFactorCaps;
using Digest = local::Digest;
using real::add;
using real::mul;
static_assert(sizeof(double)==8 && FLT_EVAL_METHOD==0,"mixed-pair factors require binary64 evaluation");
void limit(U value,U cap,const char* message) { if(value>cap) throw std::length_error(message); }
void positive(U value) { if(!value) throw std::invalid_argument("Gaussian mixed-pair factors require positive explicit controls"); }
U ceil_div(U a,U b) { return a/b+(a%b!=0); }
void extent(U bytes) {
    real::extent(bytes);
    limit(bytes/16,std::vector<C>().max_size(),"Gaussian mixed-pair factor extent exceeds vector capacity");
}
void resources(const PeriodicGaussianMetricCaps& c) {
    for(U v:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) positive(v);
}
void sha(const std::string& s) {
    if(s.size()!=64) throw std::invalid_argument("Gaussian mixed-pair source SHA length is invalid");
    for(char c:s) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("Gaussian mixed-pair source SHA is malformed");
}
bool same_owner(const Frame& a,const Frame& b) {
    if(a.kind()!=b.kind()) return false;
    if(a.is_direct_gram()) return &a.direct_gram()==&b.direct_gram();
    return a.is_embedded() ? &a.embedded()==&b.embedded() : &a.legacy()==&b.legacy();
}
U frame_control(const Frame& f) {
    return add(f.is_direct_gram()?sizeof(PeriodicGaussianGramPairPNOResult):
        f.is_embedded()?sizeof(PeriodicGaussianEmbeddedPairPNOResult):sizeof(PeriodicGaussianPairPNOResult),
        f.retained_receipt_payload_bytes());
}
const std::vector<double>& original_generation_coefficients(const Frame& f) {
    return f.is_direct_gram()?f.direct_gram().generation_coefficients():f.embedded().generation_coefficients();
}
const std::string& generation_source_receipt(const Frame& f) {
    return f.is_direct_gram()?f.direct_gram().gram_identity_sha256():f.provider_identity_sha256();
}
void compatible_source_kinds(const Frame& a,const Frame& b) {
    if(a.is_direct_gram()!=b.is_direct_gram())
        throw std::invalid_argument("Gaussian mixed-pair direct Gram and legacy source kinds cannot be mixed");
}
U known_local(const Plan& p) {
    return add(p.borrowed_geometry_numerical_bytes,add(p.borrowed_frame_numerical_bytes,add(p.caller_gauge_bytes,
        add(p.live_wannier_bytes,add(p.live_domain_bytes,add(p.live_space_bytes,p.live_basis_bytes))))));
}
void geometry_metadata(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const Frame& frame,const Geometry* g) {
    if(!g) {
        if(frame.is_direct_gram())
            throw std::invalid_argument("direct Gram PNO factors require authenticated original pair geometry");
        return;
    }
    if(!frame.is_embedded() && !frame.is_direct_gram())
        throw std::invalid_argument("local PNO geometry requires an embedded or direct Gram generation owner");
    const auto& origin=frame.is_direct_gram()?frame.direct_gram().embedding_identity_sha256():
        frame.embedded().embedding_identity_sha256();
    const auto& e=g->embedding();const auto& s=g->space().space();const auto v=e.pair_selection();
    local::validate_pao(ref,g->domain(),s);
    if(e.state_handle()!=ref.state_handle() || e.allocation_identity()!=ref.dimensions().allocation_identity
        || e.identity_sha256()!=origin
        || e.common_basis_identity_sha256()!=basis.identity_sha256()
        || e.memory().common_dimension!=frame.common_virtual_dimension()
        || e.memory().pair_dimension!=frame.generation_dimension() || v.count!=frame.generation_dimension()
        || v.begin>=s.retained_dimension() || v.count>s.retained_dimension()-v.begin
        || v.translation_cell>=ref.state().n_kpoints()
        || g->space().memory().compact.output_numerical_bytes!=s.memory().output_numerical_bytes)
        throw std::invalid_argument("local PNO geometry source, generation or selection differs");
    (void)e.coefficients_data();(void)e.energies_data();
    (void)original_generation_coefficients(frame);
}
void geometry_inventory(Plan& p,const Geometry* ga,const Geometry* gb,
    const PeriodicCorrelationPAODomain& common_domain,const PeriodicCorrelationPAOSpace& common_space) {
    p.local_geometry_a=ga!=nullptr;p.local_geometry_b=gb!=nullptr;
    if(!ga && !gb) return;
    for(U side=0;side<2;++side) {
        const auto* g=side?gb:ga;
        if(!g) continue;
        const auto& d=g->domain();const auto& r=g->space();const auto& s=r.space();const auto& e=g->embedding();
        const bool second=side==1 && ga!=nullptr;
        const U dn=d.domain_dimension(),sn=s.retained_dimension(),n=e.memory().common_dimension,m=e.memory().pair_dimension;
        const U db=add(mul(16,dn),mul(32,mul(dn,dn))),sb=add(mul(16,mul(dn,sn)),mul(8,add(dn,sn)));
        const U eb=mul(8,add(mul(n,m),m));
        if(db!=add(d.memory().retained_domain_index_bytes,d.memory().retained_matrix_bytes)
            || sb!=s.memory().output_numerical_bytes || eb!=e.memory().output_numerical_bytes)
            throw std::logic_error("local PNO geometry numerical inventory differs");
        if(&d!=&common_domain && !(second && &d==&ga->domain())) {
            p.borrowed_geometry_numerical_bytes=add(p.borrowed_geometry_numerical_bytes,db);
            p.borrowed_geometry_control_bytes=add(p.borrowed_geometry_control_bytes,sizeof(d)+4*65);
        }
        if(&s!=&common_space && !(second && &s==&ga->space().space())) {
            p.borrowed_geometry_numerical_bytes=add(p.borrowed_geometry_numerical_bytes,sb);
            p.borrowed_geometry_control_bytes=add(p.borrowed_geometry_control_bytes,sizeof(r)+7*65);
        } else if(&s==&common_space && !(second && &r==&ga->space())) {
            p.borrowed_geometry_control_bytes=add(p.borrowed_geometry_control_bytes,sizeof(r)+65);
        }
        if(!(second && &e==&ga->embedding())) {
            p.borrowed_geometry_numerical_bytes=add(p.borrowed_geometry_numerical_bytes,eb);
            p.borrowed_geometry_control_bytes=add(p.borrowed_geometry_control_bytes,sizeof(e)+9*65);
        }
        // Both before/after scans are charged even when physical owners alias.
        p.geometry_validation_work_units=add(p.geometry_validation_work_units,
            mul(1024,add(add(db,sb),add(eb,4096))));
    }
    p.borrowed_geometry_control_bytes=add(p.borrowed_geometry_control_bytes,2*sizeof(Geometry)+8192);
}
void geometry_validate(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const Frame& frame,const Geometry* g) {
    if(!g) return;
    geometry_metadata(ref,basis,frame,g);
    const auto& d=g->domain();const auto& s=g->space().space();const auto& e=g->embedding();
    const U dn=d.domain_dimension(),sn=s.retained_dimension();
    if(periodic_correlation_real_pao_embedding_detail::pair_frame_identity(ref,d,g->space(),e.pair_selection())
        !=e.pair_frame_identity_sha256()) throw std::invalid_argument("local PNO geometry does not match the authenticated pair frame");
    Digest di("vibeqc.periodic.correlation.pao.domain");
    for(int x:ref.state().mesh()) di.u32(x);
    di.u64(ref.state().n_basis());di.u64(dn);
    for(U a=0;a<dn;++a) {const auto c=d.column(a);di.u64(c.cell);di.u64(c.ao);}
    Digest dm("vibeqc.periodic.correlation.pao.matrices");dm.u64(dn);
    for(U a=0;a<dn;++a) for(U b=0;b<dn;++b) dm.complex(d.overlap(a,b));
    for(U a=0;a<dn;++a) for(U b=0;b<dn;++b) dm.complex(d.fock(a,b));
    Digest sp("vibeqc.periodic.correlation.pao.space.payload");sp.u64(dn);sp.u64(sn);
    for(U a=0;a<dn;++a) for(U b=0;b<sn;++b) sp.complex(s.coefficient(a,b));
    for(U b=0;b<sn;++b) sp.real(s.energy(b));
    for(U a=0;a<dn;++a) sp.real(s.overlap_eigenvalue(a));
    Digest ep("vibeqc.periodic.correlation.real-pao-embedding.payload");
    const auto n=e.memory().common_dimension,m=e.memory().pair_dimension;ep.u64(n);ep.u64(m);
    for(U a=0;a<n;++a) for(U b=0;b<m;++b) ep.real(e.coefficient(a,b));
    for(U b=0;b<m;++b) ep.real(e.energy(b));
    if(di.finish()!=d.domain_index_sha256() || dm.finish()!=d.matrix_payload_sha256()
        || sp.finish()!=s.payload_sha256() || ep.finish()!=e.payload_sha256())
        throw std::invalid_argument("local PNO geometry numerical payload differs from its native receipt");
}
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& c) {
    const auto& v=c.inventory();const auto& a=v.ao;const auto& b=v.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),c.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),
        add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),v.combined_borrowed_active_numeric_bytes,v.work_units_upper_bound};
}
// Metadata-only: all input floats and numerical frame payloads remain unread.
void objects(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Frame& a,const Frame& b,U i,U k) {
    compatible_source_kinds(a,b);
    real::float_environment();local::validate_wannier(ref,wannier);local::validate_pao(ref,domain,space);
    if(!hf.converged() || !hf.context_handle() || !hf.matched_finite_gaussian_hf_source()
        || hf.state_handle()!=ref.state_handle() || basis.state_handle()!=ref.state_handle()
        || basis.allocation_identity()!=ref.dimensions().allocation_identity
        || source.context_handle()!=hf.context_handle() || w.context_handle()!=hf.context_handle())
        throw std::invalid_argument("Gaussian mixed-pair factors require exact actual HF/common-frame owners");
    const auto& bm=basis.memory();const auto& state=ref.state();const auto& ctx=*hf.context_handle();
    const auto& recipe=hf.plan().fock.config;const auto naux=ctx.inventory().auxiliary.function_count;
    if(!bm.occupied_count || !bm.virtual_count || bm.orbital_count!=add(bm.occupied_count,bm.virtual_count)
        || i>=bm.occupied_count || k>=bm.occupied_count || bm.n_basis!=state.n_basis()
        || bm.occupied_count>mul(state.n_kpoints(),state.n_correlated_occupied())
        || bm.n_cells!=state.n_kpoints() || state.n_basis()!=ctx.inventory().ao.function_count
        || state.mesh()!=ctx.mesh().mesh() || state.is_shift()!=ctx.mesh().is_shift()
        || ref.dimensions().n_auxiliary!=naux || source.q_index()!=w.q_index()
        || source.conjugate_q_index()!=w.conjugate_q_index()
        || w.plan().phase!=PeriodicGaussianMetricPhase::PrincipalWhitening || w.plan().n_auxiliary!=naux
        || w.matrix_row_major().size()!=mul(naux,naux) || w.diagnostics().n_auxiliary!=naux
        || !w.diagnostics().retained_rank
        || w.plan().config.reciprocal_block!=recipe.metric.reciprocal_block
        || w.plan().config.whitener_column_block!=recipe.metric.whitener_column_block)
        throw std::invalid_argument("Gaussian mixed-pair source, shape or whitener recipe differs from actual HF");
    const auto v=basis.virtual_selection();
    if(v.count!=bm.virtual_count || v.begin>=space.retained_dimension()
        || v.count>space.retained_dimension()-v.begin || v.translation_cell>=state.n_kpoints())
        throw std::invalid_argument("Gaussian mixed-pair common virtual selection is invalid");
    (void)basis.occupied_indices_data();(void)basis.f_oo_data();
    for(const Frame* frame:{&a,&b}) {
        if(frame->state_handle()!=ref.state_handle() || frame->context_handle()!=hf.context_handle()
            || frame->common_virtual_dimension()!=bm.virtual_count || frame->occupied_count()!=bm.occupied_count
            || frame->basis_identity_sha256()!=basis.identity_sha256()
            || frame->hf_reference_source_identity_sha256()!=hf.reference_source_identity_sha256())
            throw std::invalid_argument("Gaussian mixed-pair PNO frame source or common basis differs");
        const auto x=basis.occupied(frame->occupied_slot_i()),y=basis.occupied(frame->occupied_slot_j());
        const auto fx=frame->occupied_i(),fy=frame->occupied_j();
        if(x.occupied_index!=fx.occupied_index || x.cell!=fx.cell
            || y.occupied_index!=fy.occupied_index || y.cell!=fy.cell)
            throw std::invalid_argument("Gaussian mixed-pair PNO occupied generation labels differ from common basis");
    }
}
// Fixed receipt comparisons/hashes only, after the complete resource gate.
void source_receipts(const PeriodicGaussianRHFResult& hf,const PeriodicGaussianReciprocalSource& source,
    const PeriodicGaussianMetricWhitener& w,const PeriodicCorrelationRealLocalBasis& basis,
    const Frame& a,const Frame& b) {
    if(source.source_identity_sha256()!=w.source_identity_sha256()
        || source.conjugate_source_identity_sha256()!=w.conjugate_source_identity_sha256())
        throw std::invalid_argument("Gaussian mixed-pair whitener/source receipts differ");
    for(const auto& s:{hf.reference_source_identity_sha256(),source.source_identity_sha256(),
        source.conjugate_source_identity_sha256(),w.payload_identity_sha256(),basis.identity_sha256(),
        basis.local_basis_identity_sha256(),basis.payload_sha256()}) sha(s);
    for(const Frame* f:{&a,&b}) {
        for(const auto* s:{&f->identity_sha256(),&f->payload_sha256(),&generation_source_receipt(*f),
            &f->basis_identity_sha256(),&f->hf_reference_source_identity_sha256(),
            &f->generation_exchange_integral_identity_sha256(),
            &f->initial_amplitude_identity_sha256(),&f->density_identity_sha256()}) sha(*s);
        if(f->is_direct_gram()) {
            sha(f->direct_gram().gram_payload_sha256());
            sha(f->direct_gram().gram_consumed_sources_identity_sha256());
        } else sha(f->common_exchange_integral_identity_sha256());
    }
}
void resource_wire(Digest& h,const PeriodicGaussianMetricCaps& c) {
    for(U v:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) h.u64(v);
}
void plan_wire(Digest& h,const Plan& p) {
    for(U v:{p.orbital_count,p.pno_column_count,p.tile_calls,p.contraction_term_count,
        p.retained_factor_bytes,p.retained_output_bytes,p.compensation_bytes,p.coefficient_panel_bytes,
        p.coefficient_helper_workspace_bytes,p.driver_owned_numerical_bytes,
        p.coefficient_work_units_per_kpoint,p.coefficient_work_units,
        p.n_cells,p.n_basis,p.n_auxiliary,p.occupied_count,p.common_virtual_dimension,p.common_domain_dimension,
        p.rank_a,p.rank_b,p.occupied_slot_i,p.occupied_slot_k,p.q_index,p.auxiliary_begin,p.auxiliary_count,p.ao_pair_block,
        U(p.same_frame_owner),p.borrowed_frame_numerical_bytes,p.borrowed_frame_control_bytes,p.frame_validation_work_units,
        U(p.local_geometry_a),U(p.local_geometry_b),p.borrowed_geometry_numerical_bytes,
        p.borrowed_geometry_control_bytes,p.geometry_validation_work_units,
        p.maximum_tile_owned_numerical_bytes,p.peak_owned_numerical_bytes,p.resident_whitener_bytes,
        p.borrowed_basis_active_numeric_bytes,p.caller_gauge_bytes,p.live_wannier_bytes,
        p.live_domain_bytes,p.live_space_bytes,p.live_basis_bytes,p.state_resident_bytes,
        p.macro_fixed_object_bytes,p.tile_fixed_object_bytes,p.replicas_per_node,p.reference_base_node_bytes,
        p.per_replica_inventoried_bytes,p.required_node_memory_bytes,p.reciprocal_candidate_evaluations,
        p.image_candidate_evaluations_upper_bound,p.driver_work_units,p.work_units}) h.u64(v);
    h.u64(p.config.ao_pair_block);
    for(U v:{p.live.other_retained_bytes_per_worker,p.live.other_transient_bytes_per_worker,
        p.live.fixed_backend_margin_bytes_per_worker}) h.u64(v);
    resource_wire(h,p.caps.resources);resource_wire(h,p.caps.tile.resources);
    h.u64(p.caps.tile.maximum_image_candidates);h.u64(p.caps.maximum_tile_calls);
    h.u64(p.caps.maximum_image_candidate_evaluations);h.u64(p.tile_config.reciprocal_block);
    const auto& c=p.tile_config.basis_verification_caps;
    for(U v:{c.maximum_context_storage_bytes,c.maximum_kpoint_count,c.maximum_shell_count,
        c.maximum_contraction_count,c.maximum_primitive_numeric_lanes,c.maximum_basis_content_wire_bytes,
        c.maximum_borrowed_active_numeric_bytes,c.maximum_work_units}) h.u64(v);
}
} // namespace

Storage periodic_gaussian_mixed_pair_factor_storage(U nk,U n,U effective,U common,U domain,
    U ra,U rb,U ab,U pair_block) {
    for(U v:{nk,n,effective,common,domain,ab,pair_block}) positive(v);
    if(effective>n || ra>common || rb>common || pair_block>mul(n,n))
        throw std::invalid_argument("Gaussian mixed-pair storage dimensions are inconsistent");
    Storage p;p.pno_column_count=add(ra,rb);p.orbital_count=add(2,p.pno_column_count);
    p.tile_calls=mul(nk,ceil_div(mul(n,n),pair_block));
    p.contraction_term_count=mul(mul(nk,mul(n,n)),mul(ab,mul(p.orbital_count,p.orbital_count)));
    p.retained_factor_bytes=mul(16,mul(ab,mul(p.orbital_count,p.orbital_count)));
    p.retained_output_bytes=p.retained_factor_bytes;p.compensation_bytes=p.retained_factor_bytes;
    p.coefficient_panel_bytes=mul(32,mul(n,p.orbital_count));
    p.coefficient_helper_workspace_bytes=p.pno_column_count?mul(16,mul(n,add(p.pno_column_count,3))):0;
    p.driver_owned_numerical_bytes=add(add(p.retained_output_bytes,p.compensation_bytes),
        add(p.coefficient_panel_bytes,p.coefficient_helper_workspace_bytes));
    const auto occupied_work=mul(2,mul(n,add(effective,1)));
    const auto virtual_work=add(mul(n,domain),add(mul(n,n),add(mul(mul(2,n),effective),mul(4,n))));
    // Includes finite checks, compensated complex-by-real accumulations,
    // initialization/final folding, and all repeated frame metadata access.
    const auto projected_work=p.pno_column_count?
        add(mul(common,add(virtual_work,mul(n,p.pno_column_count))),mul(n,add(p.pno_column_count,3))):0;
    p.coefficient_work_units_per_kpoint=mul(512,add(add(occupied_work,projected_work),1024));
    p.coefficient_work_units=mul(mul(2,nk),p.coefficient_work_units_per_kpoint);
    for(U bytes:{p.retained_output_bytes,p.compensation_bytes,p.coefficient_panel_bytes,
        p.coefficient_helper_workspace_bytes}) extent(bytes);
    return p;
}

namespace periodic_gaussian_mixed_pair_detail {
GeometryInventory plan_geometry(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const Frame& a,const Frame& b,
    const PeriodicCorrelationPAODomain& d,const PeriodicCorrelationPAOSpace& s,
    const Geometry* ga,const Geometry* gb) {
    compatible_source_kinds(a,b);
    geometry_metadata(ref,basis,a,ga);geometry_metadata(ref,basis,b,gb);
    Plan p;geometry_inventory(p,ga,gb,d,s);
    return {p.borrowed_geometry_numerical_bytes,p.borrowed_geometry_control_bytes,p.geometry_validation_work_units};
}
void validate_geometry(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const Frame& a,const Frame& b,
    const Geometry* ga,const Geometry* gb) {
    geometry_validate(ref,basis,a,ga);geometry_validate(ref,basis,b,gb);
}
void fill_coefficients(const PeriodicRestrictedMeanFieldState& state,const C* gauges,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis,const Frame& a,const Frame& b,
    U i,U k,U momentum,C* output,C* workspace,const Geometry* ga,const Geometry* gb) {
    const auto n=state.n_basis(),ra=a.retained_dimension(),rb=b.retained_dimension(),r=add(ra,rb);
    if(!output || (r && !workspace) || !gauges || i>=basis.memory().occupied_count
        || k>=basis.memory().occupied_count || momentum>=state.n_kpoints())
        throw std::invalid_argument("Gaussian mixed-pair coefficient helper view or index is invalid");
    const auto oi=basis.occupied(i),ok=basis.occupied(k);
    local::fill_occupied_column(state,gauges,oi.occupied_index,oi.cell,momentum,output);
    local::fill_occupied_column(state,gauges,ok.occupied_index,ok.cell,momentum,output+n);
    if(!r) return;
    auto* projected=output+2*n;auto* column=workspace+n*r;auto* scratch=column+n;
    std::fill_n(projected,n*r,C{});std::fill_n(workspace,n*r,C{});
    if(!ga && !gb) {
        const auto selected=basis.virtual_selection();const auto& ca=a.coefficients();const auto& cb=b.coefficients();
        for(U v=0;v<selected.count;++v) {
            local::fill_virtual_columns(state,domain,space,selected.begin+v,1,selected.translation_cell,
                momentum,column,scratch);
            for(U t=0;t<r;++t) {
                const auto weight=t<ra?ca[v*ra+t]:cb[v*rb+(t-ra)];
                for(U mu=0;mu<n;++mu)
                    real::accumulate(column[mu]*weight,projected[t*n+mu],workspace[t*n+mu]);
            }
        }
        for(U at=0;at<n*r;++at) projected[at]=real::finite(projected[at]+workspace[at]);
        return;
    }
    // Independent physical PAO frames are not projected into one another.
    // With no local geometry, retain the established common-C compatibility
    // route. With geometry, use the preserved ORIGINAL D[m,r] directly.
    for(U side=0;side<2;++side) {
        const auto& frame=side?b:a;const auto* g=side?gb:ga;
        const U rank=side?rb:ra,offset=side?ra:0;
        if(!rank) continue;
        const auto selected=g?g->embedding().pair_selection():basis.virtual_selection();
        const auto& d=g?g->domain():domain;const auto& s=g?g->space().space():space;
        const auto& c=g?original_generation_coefficients(frame):frame.coefficients();
        for(U v=0;v<selected.count;++v) {
            local::fill_virtual_columns(state,d,s,selected.begin+v,1,selected.translation_cell,momentum,column,scratch);
            for(U t=0;t<rank;++t) for(U mu=0;mu<n;++mu)
                real::accumulate(column[mu]*c[v*rank+t],projected[(offset+t)*n+mu],workspace[(offset+t)*n+mu]);
        }
    }
    for(U at=0;at<n*r;++at) projected[at]=real::finite(projected[at]+workspace[at]);
}
} // namespace periodic_gaussian_mixed_pair_detail

Plan plan_periodic_gaussian_mixed_pair_factor_panel(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Frame& a,const Frame& b,U i,U k,U begin,U count,const Config& config,const Live& live,const Caps& caps,
    const Geometry* ga,const Geometry* gb) {
    resources(caps.resources);resources(caps.tile.resources);
    for(U v:{caps.maximum_tile_calls,caps.maximum_image_candidate_evaluations,caps.tile.maximum_image_candidates,
        live.fixed_backend_margin_bytes_per_worker}) positive(v);
    objects(hf,ref,source,w,wannier,domain,space,basis,a,b,i,k);
    geometry_metadata(ref,basis,a,ga);geometry_metadata(ref,basis,b,gb);
    const auto& state=ref.state();const auto& ctx=*hf.context_handle();
    Plan p;p.config=config;p.live=live;p.caps=caps;p.tile_config=hf.plan().fock.config.tile;
    p.direct_gram_frames=a.is_direct_gram();
    p.n_cells=state.n_kpoints();p.n_basis=state.n_basis();p.n_auxiliary=ref.dimensions().n_auxiliary;
    p.occupied_count=basis.memory().occupied_count;p.common_virtual_dimension=basis.memory().virtual_count;
    p.common_domain_dimension=domain.domain_dimension();p.rank_a=a.retained_dimension();p.rank_b=b.retained_dimension();
    p.occupied_slot_i=i;p.occupied_slot_k=k;p.q_index=source.q_index();
    p.auxiliary_begin=begin;p.auxiliary_count=count;p.ao_pair_block=config.ao_pair_block;
    if(!count || begin>=p.n_auxiliary || count>p.n_auxiliary-begin)
        throw std::invalid_argument("Gaussian mixed-pair auxiliary slice is invalid");
    static_cast<Storage&>(p)=periodic_gaussian_mixed_pair_factor_storage(p.n_cells,p.n_basis,state.n_effective_orbitals(),
        p.common_virtual_dimension,p.common_domain_dimension,p.rank_a,p.rank_b,count,config.ao_pair_block);
    if(ga || gb) {
        const U maximum_domain=std::max({p.common_domain_dimension,
            ga?ga->domain().domain_dimension():0,gb?gb->domain().domain_dimension():0});
        const auto upper=periodic_gaussian_mixed_pair_factor_storage(p.n_cells,p.n_basis,state.n_effective_orbitals(),
            p.common_virtual_dimension,maximum_domain,p.rank_a,p.rank_b,count,config.ao_pair_block);
        // Independent geometry traversals can each span the full common
        // generation upper. This is conservative work, not a timing claim.
        p.coefficient_work_units_per_kpoint=mul(2,upper.coefficient_work_units_per_kpoint);
        p.coefficient_work_units=mul(mul(2,p.n_cells),p.coefficient_work_units_per_kpoint);
    }
    geometry_inventory(p,ga,gb,domain,space);
    p.same_frame_owner=same_owner(a,b);
    p.borrowed_frame_numerical_bytes=add(a.retained_numerical_bytes(),p.same_frame_owner?0:b.retained_numerical_bytes());
    p.borrowed_frame_control_bytes=add(frame_control(a),p.same_frame_owner?0:frame_control(b));
    const auto va=plan_periodic_gaussian_pair_pno_frame_validation(a),vb=plan_periodic_gaussian_pair_pno_frame_validation(b);
    p.frame_validation_work_units=mul(2,add(va.work_units,p.same_frame_owner?0:vb.work_units));
    p.maximum_tile_owned_numerical_bytes=caps.tile.resources.maximum_owned_numeric_bytes;
    p.peak_owned_numerical_bytes=add(p.driver_owned_numerical_bytes,p.maximum_tile_owned_numerical_bytes);
    p.resident_whitener_bytes=mul(16,mul(p.n_auxiliary,p.n_auxiliary));
    p.borrowed_basis_active_numeric_bytes=ctx.inventory().combined_borrowed_active_numeric_bytes;
    const auto wm=plan_periodic_correlation_wannier(state.mesh(),p.n_basis,state.n_correlated_occupied());
    p.caller_gauge_bytes=wm.caller_gauge_bytes;p.live_wannier_bytes=wm.retained_coefficient_bytes;
    p.live_domain_bytes=add(mul(16,p.common_domain_dimension),mul(32,mul(p.common_domain_dimension,p.common_domain_dimension)));
    p.live_space_bytes=add(mul(16,mul(p.common_domain_dimension,space.retained_dimension())),
        mul(8,add(p.common_domain_dimension,space.retained_dimension())));
    const auto o=p.occupied_count,v=p.common_virtual_dimension;
    p.live_basis_bytes=add(mul(16,o),mul(8,add(add(mul(o,o),mul(v,v)),mul(o,v))));
    p.state_resident_bytes=ref.state_resident_bytes();
    if(p.live_wannier_bytes!=wannier.memory().retained_coefficient_bytes
        || p.live_domain_bytes!=add(domain.memory().retained_domain_index_bytes,domain.memory().retained_matrix_bytes)
        || p.live_space_bytes!=space.memory().output_numerical_bytes || p.live_basis_bytes!=basis.memory().retained_output_bytes
        || p.state_resident_bytes!=state.resident_bytes() || ref.dimensions().external_bytes<p.state_resident_bytes)
        throw std::logic_error("Gaussian mixed-pair live payload census differs from native owners");
    // Includes retained receipt payloads, several simultaneous fixed digests,
    // temporary metadata strings and all source-view validation controls.
    p.macro_fixed_object_bytes=add(add(p.borrowed_frame_control_bytes,p.borrowed_geometry_control_bytes),
        65536+17*65+sizeof(Panel)+sizeof(Plan)+sizeof(Config)+sizeof(Live)+sizeof(Caps)+2*sizeof(Frame)
        +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicCorrelationAdmittedReference)
        +sizeof(PeriodicCorrelationWannier)+sizeof(PeriodicCorrelationPAODomain)
        +sizeof(PeriodicCorrelationPAOSpace)+sizeof(PeriodicCorrelationRealLocalBasis)
        +std::max(va.control_storage_bytes,vb.control_storage_bytes));
    p.tile_fixed_object_bytes=sizeof(PeriodicGaussianSourceContext)+sizeof(PeriodicGaussianReciprocalSource)
        +sizeof(PeriodicGaussianMetricWhitener)+sizeof(PeriodicGaussianThreeCenterPlan)
        +sizeof(PeriodicGaussianThreeCenterTile)+sizeof(PeriodicSystem);
    const auto& budget=ref.budget();const auto& dims=ref.dimensions();
    p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);positive(p.replicas_per_node);
    p.reference_base_node_bytes=add(add(dims.external_bytes,dims.shared_bytes),
        mul(budget.mpi_ranks,add(dims.per_rank_bytes,dims.localization_window_bytes_per_rank)));
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.resident_whitener_bytes,
        add(p.borrowed_basis_active_numeric_bytes,add(known_local(p),add(p.macro_fixed_object_bytes,
        add(p.tile_fixed_object_bytes,add(live.other_retained_bytes_per_worker,
        add(live.other_transient_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))))))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_replica_inventoried_bytes));
    const auto per_tile=mul(2,source.candidate_count()),accepted=source.accepted_vector_count();
    if(!accepted || !p.tile_config.reciprocal_block) throw std::invalid_argument("Gaussian mixed-pair source is empty");
    p.reciprocal_candidate_evaluations=mul(p.tile_calls,per_tile);
    p.image_candidate_evaluations_upper_bound=mul(p.tile_calls,mul(caps.tile.maximum_image_candidates,
        add(add(1,accepted),ceil_div(accepted,p.tile_config.reciprocal_block))));
    p.driver_work_units=add(ctx.inventory().work_units_upper_bound,
        add(p.frame_validation_work_units,p.geometry_validation_work_units));
    for(U work:{p.coefficient_work_units,mul(128,p.contraction_term_count),mul(256,p.caller_gauge_bytes),
        mul(128,mul(o,o)),mul(256,p.retained_output_bytes),mul(4096,p.tile_calls),U{131072}})
        p.driver_work_units=add(p.driver_work_units,work);
    p.work_units=add(p.driver_work_units,mul(p.tile_calls,caps.tile.resources.maximum_work_units));
    for(U bytes:{p.borrowed_frame_numerical_bytes,p.caller_gauge_bytes,mul(80,p.tile_calls)}) extent(bytes);
    limit(per_tile,caps.tile.resources.maximum_candidate_evaluations,"Gaussian mixed-pair tile reciprocal cap exceeded");
    limit(p.peak_owned_numerical_bytes,caps.resources.maximum_owned_numeric_bytes,"Gaussian mixed-pair owned cap exceeded");
    limit(p.per_replica_inventoried_bytes,caps.resources.maximum_per_replica_inventoried_bytes,"Gaussian mixed-pair replica cap exceeded");
    limit(p.required_node_memory_bytes,caps.resources.maximum_node_inventoried_bytes,"Gaussian mixed-pair node cap exceeded");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian mixed-pair reference node cap exceeded");
    limit(p.tile_calls,caps.maximum_tile_calls,"Gaussian mixed-pair tile calls cap exceeded");
    limit(p.reciprocal_candidate_evaluations,caps.resources.maximum_candidate_evaluations,"Gaussian mixed-pair reciprocal cap exceeded");
    limit(p.image_candidate_evaluations_upper_bound,caps.maximum_image_candidate_evaluations,"Gaussian mixed-pair image cap exceeded");
    limit(p.work_units,caps.resources.maximum_work_units,"Gaussian mixed-pair work cap exceeded");
    source_receipts(hf,source,w,basis,a,b);
    Digest digest("vibeqc.periodic.gaussian-mixed-pair-factors.plan");
    for(const auto& s:{hf.reference_source_identity_sha256(),source.source_identity_sha256(),
        source.conjugate_source_identity_sha256(),w.payload_identity_sha256(),basis.identity_sha256(),
        dims.allocation_identity,wannier.wannier_identity_sha256(),domain.pao_domain_identity_sha256(),
        space.pao_space_identity_sha256(),a.identity_sha256(),b.identity_sha256(),a.payload_sha256(),b.payload_sha256(),
        generation_source_receipt(a),generation_source_receipt(b)}) digest.string(s);
    if(p.direct_gram_frames) digest.string("DirectPAOGram;mandatory-original-PAO-geometry;no-common-provider");
    for(const auto* g:{ga,gb}) if(g) {
        digest.string(g->domain().pao_domain_identity_sha256());
        digest.string(g->space().identity_sha256());digest.string(g->embedding().identity_sha256());
    }
    plan_wire(digest,p);const auto identity=digest.finish();
    std::copy(identity.begin(),identity.end(),p.identity_ascii.begin());return p;
}

const C* Panel::data() const {
    if(!context_ || !state_ || values_.size()!=memory_.retained_factor_bytes/16)
        throw std::logic_error("Gaussian mixed-pair panel is consumed or malformed");
    return values_.data();
}
const std::string& Panel::frame_a_provider_identity_sha256() const {
    if(direct_gram_frames()) throw std::logic_error("direct Gram mixed factors have no legacy provider receipt");
    return provider_a_;
}
const std::string& Panel::frame_b_provider_identity_sha256() const {
    if(direct_gram_frames()) throw std::logic_error("direct Gram mixed factors have no legacy provider receipt");
    return provider_b_;
}
const std::string& Panel::frame_a_gram_identity_sha256() const {
    if(!direct_gram_frames()) throw std::logic_error("legacy mixed factors have no direct Gram receipt");
    return gram_a_;
}
const std::string& Panel::frame_b_gram_identity_sha256() const {
    if(!direct_gram_frames()) throw std::logic_error("legacy mixed factors have no direct Gram receipt");
    return gram_b_;
}
C Panel::element(std::size_t p,std::size_t l,std::size_t r) const {
    if(p>=auxiliary_count() || l>=orbital_count() || r>=orbital_count())
        throw std::out_of_range("Gaussian mixed-pair factor element is out of range");
    return data()[(p*orbital_count()+l)*orbital_count()+r];
}

Panel build_periodic_gaussian_mixed_pair_factor_panel(
    const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
    const C* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
    const Frame& a,const Frame& b,U i,U k,U begin,U count,const Config& config,const Live& live,const Caps& caps,
    const Geometry* ga,const Geometry* gb) {
    const auto p=plan_periodic_gaussian_mixed_pair_factor_panel(hf,ref,source,w,wannier,domain,space,basis,
        a,b,i,k,begin,count,config,live,caps,ga,gb);
    const auto context=hf.context_handle();const auto& state=ref.state();
    context->verify_bases(ao,auxiliary,exact_basis_caps(*context));
    const auto* indices=basis.occupied_indices_data();
    real::indices(indices,mul(2,p.occupied_count),p.occupied_count,state);
    const auto gauge=local::validate_gauges(wannier,gauges,gauge_count);
    const auto actual_basis=real::local_basis_identity(ref,wannier,gauge,domain,space,indices,
        p.occupied_count,basis.virtual_selection());
    if(actual_basis!=basis.local_basis_identity_sha256())
        throw std::invalid_argument("Gaussian mixed-pair actual common coefficients differ from basis certificate");
    a.verify_payload();if(!p.same_frame_owner) b.verify_payload();
    geometry_validate(ref,basis,a,ga);geometry_validate(ref,basis,b,gb);
    Panel result;result.memory_=p;result.context_=context;result.state_=ref.state_handle();
    result.i_=basis.occupied(i);result.k_=basis.occupied(k);result.hf_=hf.reference_source_identity_sha256();
    result.basis_=actual_basis;result.certificate_=basis.identity_sha256();
    result.source_=source.source_identity_sha256();result.opposite_=source.conjugate_source_identity_sha256();
    result.whitener_=w.payload_identity_sha256();result.frame_a_=a.identity_sha256();result.frame_b_=b.identity_sha256();
    result.frame_a_payload_=a.payload_sha256();result.frame_b_payload_=b.payload_sha256();
    if(p.direct_gram_frames) {
        result.gram_a_=generation_source_receipt(a);result.gram_b_=generation_source_receipt(b);
    } else {
        result.provider_a_=a.provider_identity_sha256();result.provider_b_=b.provider_identity_sha256();
    }
    result.values_.resize(static_cast<std::size_t>(p.retained_factor_bytes/16));
    auto& diagnostic=result.diagnostics_;diagnostic.charged_work_units_upper_bound=p.driver_work_units;
    Digest consumed("vibeqc.periodic.gaussian-mixed-pair-factors.consumed-tiles");
    for(const auto* s:{&result.hf_,&result.source_,&result.opposite_,&result.whitener_}) consumed.string(*s);
    consumed.u64(p.tile_calls);
    {
        std::vector<C> correction(result.values_.size());
        std::vector<C> coefficients(static_cast<std::size_t>(p.coefficient_panel_bytes/16));
        std::vector<C> workspace(static_cast<std::size_t>(p.coefficient_helper_workspace_bytes/16));
        PeriodicGaussianMetricLiveInventory nested;
        nested.replicas_per_node=p.replicas_per_node;nested.external_node_bytes=p.reference_base_node_bytes;
        nested.other_retained_bytes_per_replica=add(p.driver_owned_numerical_bytes,
            add(known_local(p),add(p.macro_fixed_object_bytes,live.other_retained_bytes_per_worker)));
        nested.other_transient_bytes_per_replica=live.other_transient_bytes_per_worker;
        nested.fixed_backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
        const auto n=p.n_basis,m=p.orbital_count,nn=mul(n,n);
        for(U momentum=0;momentum<p.n_cells;++momentum) {
            const auto ket=context->ket_index(momentum,source.q_index());
            periodic_gaussian_mixed_pair_detail::fill_coefficients(state,gauges,domain,space,basis,a,b,i,k,
                momentum,coefficients.data(),workspace.data(),ga,gb);
            periodic_gaussian_mixed_pair_detail::fill_coefficients(state,gauges,domain,space,basis,a,b,i,k,
                ket,coefficients.data()+n*m,workspace.data(),ga,gb);
            for(U pair=0;pair<nn;pair+=std::min(p.ao_pair_block,nn-pair)) {
                limit(add(diagnostic.charged_work_units_upper_bound,caps.tile.resources.maximum_work_units),
                    caps.resources.maximum_work_units,"Gaussian mixed-pair remaining tile work cap exceeded");
                const auto pairs=std::min(p.ao_pair_block,nn-pair);
                const PeriodicGaussianThreeCenterSelection selection{momentum,pair,pairs,begin,count};
                const auto tile=build_periodic_gaussian_three_center_tile(source,w,ao,auxiliary,selection,p.tile_config,nested,caps.tile);
                const auto& descriptor=tile.descriptor();const auto& memory=tile.plan();
                if(tile.context_handle()!=context || descriptor.k_bra_index!=momentum || descriptor.k_ket_index!=ket
                    || descriptor.q_index!=source.q_index() || descriptor.ao_pair_begin!=pair || descriptor.ao_pair_count!=pairs
                    || descriptor.auxiliary_begin!=begin || descriptor.auxiliary_count!=count || descriptor.element_count!=mul(pairs,count)
                    || tile.source_identity_sha256()!=result.source_ || tile.conjugate_source_identity_sha256()!=result.opposite_
                    || tile.whitener_payload_identity_sha256()!=result.whitener_)
                    throw std::logic_error("Gaussian mixed-pair native tile provenance or shape differs");
                diagnostic.charged_work_units_upper_bound=add(diagnostic.charged_work_units_upper_bound,memory.work_units_upper_bound);
                diagnostic.reciprocal_candidate_evaluations=add(diagnostic.reciprocal_candidate_evaluations,memory.reciprocal_candidate_evaluations);
                diagnostic.image_candidate_evaluations=add(diagnostic.image_candidate_evaluations,memory.image_candidate_evaluations);
                const auto owned=add(p.driver_owned_numerical_bytes,memory.owned_numeric_peak_bytes);
                if(owned>p.peak_owned_numerical_bytes || memory.per_replica_inventoried_bytes>p.per_replica_inventoried_bytes)
                    throw std::logic_error("Gaussian mixed-pair nested tile lifetime exceeds admission");
                diagnostic.maximum_observed_owned_numerical_bytes=std::max(diagnostic.maximum_observed_owned_numerical_bytes,owned);
                diagnostic.maximum_observed_per_replica_inventoried_bytes=std::max(
                    diagnostic.maximum_observed_per_replica_inventoried_bytes,memory.per_replica_inventoried_bytes);
                const auto* right=coefficients.data()+n*m;
                for(U at=0;at<pairs;++at) {
                    const auto mu=(pair+at)/n,nu=(pair+at)%n;
                    for(U row=0;row<count;++row) {
                        const auto factor=tile.matrix_row_major()[row*pairs+at];
                        for(U l=0;l<m;++l) for(U r=0;r<m;++r) {
                            const auto index=(row*m+l)*m+r;
                            real::accumulate((std::conj(coefficients[l*n+mu])*right[r*n+nu])*factor,
                                result.values_[index],correction[index]);
                        }
                    }
                }
                consumed.u64(diagnostic.completed_tile_calls);consumed.string(tile.payload_identity_sha256());
                ++diagnostic.completed_tile_calls;
            }
        }
        const auto nk=static_cast<double>(p.n_cells);const auto normalization=(1.0/nk)/std::sqrt(nk);
        if(!std::isfinite(normalization) || normalization<=0) throw std::overflow_error("Gaussian mixed-pair normalization is invalid");
        for(std::size_t at=0;at<result.values_.size();++at)
            result.values_[at]=real::finite((result.values_[at]+correction[at])*normalization);
    }
    if(diagnostic.completed_tile_calls!=p.tile_calls || diagnostic.reciprocal_candidate_evaluations!=p.reciprocal_candidate_evaluations
        || diagnostic.image_candidate_evaluations>p.image_candidate_evaluations_upper_bound || diagnostic.charged_work_units_upper_bound>p.work_units)
        throw std::logic_error("Gaussian mixed-pair completed traversal exceeds admission");
    objects(hf,ref,source,w,wannier,domain,space,basis,a,b,i,k);
    source_receipts(hf,source,w,basis,a,b);
    if(local::validate_gauges(wannier,gauges,gauge_count)!=gauge || a.identity_sha256()!=result.frame_a_
        || b.identity_sha256()!=result.frame_b_ || a.payload_sha256()!=result.frame_a_payload_
        || b.payload_sha256()!=result.frame_b_payload_)
        throw std::invalid_argument("Gaussian mixed-pair inputs changed during construction");
    a.verify_payload();if(!p.same_frame_owner) b.verify_payload();
    geometry_validate(ref,basis,a,ga);geometry_validate(ref,basis,b,gb);
    result.consumed_=consumed.finish();
    Digest payload("vibeqc.periodic.gaussian-mixed-pair-factors.payload");
    for(U v:{p.q_index,p.auxiliary_begin,p.auxiliary_count,p.orbital_count,p.rank_a,p.rank_b,p.occupied_slot_i,p.occupied_slot_k}) payload.u64(v);
    for(C value:result.values_) payload.complex(value);result.payload_=payload.finish();
    Digest identity("vibeqc.periodic.gaussian-mixed-pair-factors.identity");
    for(const auto& s:{result.hf_,context->source_context_identity_sha256(),result.basis_,result.certificate_,
        result.source_,result.opposite_,result.whitener_,result.consumed_,result.payload_,result.frame_a_,result.frame_b_,
        result.frame_a_payload_,result.frame_b_payload_,result.provider_a_,result.provider_b_}) identity.string(s);
    if(p.direct_gram_frames) {
        identity.string("actual-HF-context-recipe;original-direct-A;principal-original-aux;Nk^-3/2;all-ordered-complex;i,k,A,B;independent-frames;DirectPAOGram;no-common-provider");
        identity.string(result.gram_a_);identity.string(result.gram_b_);
    } else identity.string("actual-HF-context-recipe;original-direct-A;principal-original-aux;Nk^-3/2;all-ordered-complex;i,k,A,B;independent-frames;common-C-stream;provider-lineage-only");
    if(ga || gb) {
        identity.string("optional-authenticated-local-PAO-geometry;original-generation-D;no-XtC-reconstruction");
        identity.u64(ga!=nullptr);identity.u64(gb!=nullptr);
        for(const auto* g:{ga,gb}) if(g) {
            identity.string(g->domain().pao_domain_identity_sha256());
            identity.string(g->space().identity_sha256());identity.string(g->embedding().identity_sha256());
        }
    }
    result.identity_=identity.finish();return result;
}

} // namespace vibeqc
