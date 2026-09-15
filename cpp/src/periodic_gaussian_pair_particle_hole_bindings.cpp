// Tiny diagnostic seams. Frozen input uses a native CCSD owner; the separate
// current-snapshot diagnostic borrows numerical trial T in exact warm frames.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstring>
#include "vibeqc/periodic_gaussian_pair_particle_hole.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif
namespace periodic_gaussian_pair_particle_hole_python {
using namespace vibeqc;
using Live=PeriodicGaussianPairParticleHoleLiveInventory;
using Caps=PeriodicGaussianPairParticleHoleCaps;
using U=std::uint64_t;
U add(U x,U y) {
    if(y>std::numeric_limits<U>::max()-x) throw std::overflow_error("particle-hole binding inventory overflows");return x+y;
}
U multiply(U x,U y) {
    if(x && y>std::numeric_limits<U>::max()/x) throw std::overflow_error("particle-hole binding inventory overflows");return x*y;
}
struct Common {
    const PeriodicCorrelationPAODomain* domain;
    const PeriodicCorrelationPAOSpace* space;
    Live live;
};
Common common(const py::object& owner,const py::object& space,Live live) {
    if(py::isinstance<PeriodicGaussianPairDomainBuilder>(owner)) {
        if(!space.is_none()) throw py::type_error("particle-hole builder common_geometry requires space=None");
        const auto& b=owner.cast<const PeriodicGaussianPairDomainBuilder&>();
        const auto& d=b.common_domain();const auto& s=b.common_real_space().space();
        const auto counted=add(d.memory().retained_domain_index_bytes,add(d.memory().retained_matrix_bytes,s.memory().output_numerical_bytes));
        const auto whole=b.retained_numerical_bytes();
        if(counted>whole) throw std::logic_error("particle-hole builder common geometry exceeds owner inventory");
        live.other_live_numerical_bytes_per_worker=add(live.other_live_numerical_bytes_per_worker,whole-counted);
        live.other_live_control_bytes_per_worker=add(live.other_live_control_bytes_per_worker,b.retained_control_storage_bytes());
        return {&d,&s,live};
    }
    if(!py::isinstance<PeriodicCorrelationPAODomain>(owner) || !py::isinstance<PeriodicCorrelationPAOSpace>(space))
        throw py::type_error("particle-hole common_geometry requires native PAODomain/PAOSpace or Builder/None");
    return {&owner.cast<const PeriodicCorrelationPAODomain&>(),&space.cast<const PeriodicCorrelationPAOSpace&>(),live};
}
void tiny(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationRealLocalBasis& basis,const Caps& caps) {
    if(!hf.context_handle() || hf.plan().n_kpoints>8 || hf.plan().n_basis>4
        || hf.context_handle()->inventory().auxiliary.function_count>8
        || basis.memory().occupied_count>8 || basis.memory().virtual_count>8
        || hf.plan().fock.config.source_caps.maximum_candidate_evaluations>65536
        || caps.maximum_owned_numerical_bytes>16U*1024*1024
        || caps.maximum_node_bytes>128U*1024*1024)
        throw std::length_error("Gaussian particle-hole diagnostic exceeds tiny shape/source/memory bounds");
}
py::array_t<double> copy(const PeriodicGaussianPairParticleHoleResult& r) {
    const auto n=r.memory().target_dimension;
    if(n>8) throw std::length_error("Gaussian particle-hole copy exceeds tiny rank");
    py::array_t<double> out({static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)});
    if(n) std::memcpy(out.mutable_data(),r.residual_data(),8*n*n);
    out.attr("setflags")(false);return out;
}
using Reader=BoundedRestrictedPairCCSDAmplitudes;
using ReaderOptions=BoundedRestrictedPairCCSDAmplitudesOptions;
using ReaderCaps=BoundedRestrictedPairCCSDAmplitudesCaps;
struct SnapshotBorrow {
    U o=0,n=0,P=0;
    std::array<BoundedRestrictedPairCCSDSinglesView,4> singles;
    std::array<BoundedRestrictedPairCCSDPairView,10> pairs;
    std::array<py::array,18> pins;
    BoundedRestrictedPairCCSDAmplitudesInput input() const {
        return {o,n,singles.data(),static_cast<std::size_t>(o),pairs.data(),static_cast<std::size_t>(P)};
    }
};
void exact_array(const py::array& a,int ndim) {
    if(!a.dtype().is(py::dtype::of<double>()) || !(a.flags()&py::array::c_style) || a.ndim()!=ndim
        || reinterpret_cast<std::uintptr_t>(a.data())%alignof(double))
        throw py::type_error("particle-hole snapshot requires exact aligned contiguous binary64 arrays");
}
U warm_numeric(const PeriodicGaussianPairMP2Result& w) {
    return add(w.diagnostics().retained_pair_bytes,add(w.diagnostics().retained_pair_geometry_bytes,w.solver().memory().output_numerical_bytes));
}
U warm_controls(const PeriodicGaussianPairMP2Result& w) {
    const auto P=w.memory().pair_count;
    if(P>10) throw std::length_error("particle-hole snapshot diagnostic exceeds tiny warm pair count");
    return add(sizeof(PeriodicGaussianPairMP2Result)+8*65,add(P*sizeof(PeriodicGaussianPairSpace),
        add(w.memory().retained_pair_seal_bytes,w.diagnostics().retained_pair_geometry_control_bytes)));
}
void snapshot_shape(const PeriodicGaussianPairMP2Result& warm,const py::list& sc,const py::list& st,
    const py::list& pt,const ReaderCaps& caps,const Caps& physical) {
    const auto o=warm.memory().occupied_count,n=warm.memory().common_virtual_dimension,P=warm.memory().pair_count;
    if(!o || o>4 || !n || n>4 || P!=o*(o+1)/2 || P>10
        || sc.size()!=o || st.size()!=o || pt.size()!=P)
        throw std::length_error("particle-hole snapshot diagnostic exceeds tiny descriptor counts");
    if(2*o>physical.maximum_source_slots || P>physical.maximum_pair_count
        || o>caps.maximum_occupied_count || n>caps.maximum_common_virtual_dimension || P>caps.maximum_pair_count
        || o*sizeof(BoundedRestrictedPairCCSDSinglesView)+P*sizeof(BoundedRestrictedPairCCSDPairView)>caps.maximum_table_bytes)
        throw std::length_error("particle-hole snapshot diagnostic source/table cap exceeded");
}
SnapshotBorrow borrow_snapshot(const PeriodicGaussianPairMP2Result& coefficient_warm,const py::list& sc,
    const py::list& st,const py::list& pt) {
    SnapshotBorrow b;b.o=coefficient_warm.memory().occupied_count;b.n=coefficient_warm.memory().common_virtual_dimension;
    b.P=coefficient_warm.memory().pair_count;
    const auto view=[](const py::array& a)->BoundedRestrictedCCSDRealView {
        return {static_cast<const double*>(a.data()),static_cast<std::size_t>(a.size())};
    };
    for(U i=0;i<b.o;++i) {
        if(!py::isinstance<py::array>(sc[i]) || !py::isinstance<py::array>(st[i]))
            throw py::type_error("particle-hole snapshot singles require arrays without implicit conversion");
        const auto c=py::reinterpret_borrow<py::array>(sc[i]),t=py::reinterpret_borrow<py::array>(st[i]);
        exact_array(c,2);exact_array(t,1);
        if(c.shape(0)!=static_cast<py::ssize_t>(b.n) || c.shape(1)!=t.shape(0) || t.size()>static_cast<py::ssize_t>(b.n))
            throw py::type_error("particle-hole snapshot singles dimensions differ");
        b.singles[i]={static_cast<U>(t.size()),view(c),view(t)};b.pins[2*i]=c;b.pins[2*i+1]=t;
    }
    U k=0;
    for(U i=0;i<b.o;++i) for(U j=i;j<b.o;++j,++k) {
        const auto& pair=coefficient_warm.pair(i,j);const auto& c=pair.coefficients();const auto r=pair.frame().retained_dimension();
        if(!py::isinstance<py::array>(pt[k])) throw py::type_error("particle-hole snapshot doubles require arrays without conversion");
        const auto t=py::reinterpret_borrow<py::array>(pt[k]);exact_array(t,2);
        if(t.shape(0)!=static_cast<py::ssize_t>(r) || t.shape(1)!=static_cast<py::ssize_t>(r))
            throw py::type_error("particle-hole snapshot pair dimensions differ from native frame");
        b.pairs[k]={r,{c.data(),c.size()},view(t)};b.pins[2*b.o+k]=t;
    }
    return b;
}
BoundedRestrictedPairCCSDAmplitudesInventory reader_inventory(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationWannier& w,const PeriodicCorrelationRealLocalBasis& basis,const Common& c,
    const PeriodicGaussianRHFResult& hf,const PeriodicGaussianPairMP2Result& warm,const Live& live) {
    const auto& d=ref.dimensions();const auto& b=ref.budget();
    const U replicas=multiply(b.mpi_ranks,b.workers_per_rank);
    const U baseline=add(add(d.external_bytes,d.shared_bytes),multiply(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    const U common_bytes=add(w.memory().caller_gauge_bytes,add(w.memory().retained_coefficient_bytes,
        add(c.domain->memory().retained_domain_index_bytes,add(c.domain->memory().retained_matrix_bytes,
        add(c.space->memory().output_numerical_bytes,basis.memory().retained_output_bytes)))));
    const U other=add(warm_numeric(warm),add(warm_controls(warm),add(common_bytes,
        add(hf.context_handle()->inventory().combined_borrowed_active_numeric_bytes,
        add(live.other_live_numerical_bytes_per_worker,live.other_live_control_bytes_per_worker)))));
    return {replicas,baseline,other,live.backend_margin_bytes_per_worker};
}
py::array_t<double> snapshot_copy(const PeriodicGaussianPairParticleHoleSnapshotResult& r) {
    const auto n=r.memory().stream.target_dimension;if(n>4) throw std::length_error("particle-hole snapshot copy exceeds tiny rank");
    py::array_t<double> out({static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)});
    if(n) std::memcpy(out.mutable_data(),r.residual_data(),8*n*n);out.attr("setflags")(false);return out;
}
py::object snapshot_diagnostic(bool build,const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationWannier& wannier,const py::object& common_geometry,const py::object& space,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairMP2Result& warm,
    U i,U j,const PeriodicGaussianPairInteractionConfig& config,const PeriodicGaussianPairInteractionOptions& options,
    Live live,const Caps& caps,const py::list& sc,const py::list& st,const py::list& pt,
    const ReaderOptions& ro,const ReaderCaps& rc,const py::object& coefficient_owner,
    const BasisSet* ao=nullptr,const BasisSet* auxiliary=nullptr,const py::array* gauges=nullptr,U index=0) {
    tiny(hf,basis,caps);snapshot_shape(warm,sc,st,pt,rc,caps);
    const auto common_owner=common(common_geometry,space,live);live=common_owner.live;
    const auto* coefficient_warm=&warm;
    if(!coefficient_owner.is_none()) {
        if(!py::isinstance<PeriodicGaussianPairMP2Result>(coefficient_owner))
            throw py::type_error("particle-hole coefficient_warmstart must be an authentic native MP2 owner");
        coefficient_warm=&coefficient_owner.cast<const PeriodicGaussianPairMP2Result&>();
        snapshot_shape(*coefficient_warm,sc,st,pt,rc,caps);
        if(coefficient_warm->memory().occupied_count!=warm.memory().occupied_count
            || coefficient_warm->memory().common_virtual_dimension!=warm.memory().common_virtual_dimension)
            throw py::type_error("particle-hole coefficient_warmstart dimensions differ");
        if(coefficient_warm!=&warm) {
            live.other_live_numerical_bytes_per_worker=add(live.other_live_numerical_bytes_per_worker,warm_numeric(*coefficient_warm));
            live.other_live_control_bytes_per_worker=add(live.other_live_control_bytes_per_worker,warm_controls(*coefficient_warm));
        }
    }
    live.other_live_control_bytes_per_worker=add(live.other_live_control_bytes_per_worker,
        sizeof(SnapshotBorrow)+2*sizeof(ReaderOptions)+2*sizeof(ReaderCaps)+4096);
    const auto pinned=borrow_snapshot(*coefficient_warm,sc,st,pt);
    const auto inv=reader_inventory(ref,wannier,basis,common_owner,hf,warm,live);
    const auto rp=plan_bounded_restricted_pair_ccsd_amplitudes(pinned.input(),ro,inv,rc);
    if(rp.borrowed_numerical_bytes>1024*1024 || rp.required_node_inventoried_bytes>128U*1024*1024
        || rp.validation_work_units>1000000000ULL)
        throw std::length_error("particle-hole diagnostic Reader exceeds tiny memory/work");
    // Reader construction has its OWN earlier admission. Do not claim the
    // later PH cap precedes these numerical trial-T/singles reads.
    const auto reader=make_bounded_restricted_pair_ccsd_amplitudes(pinned.input(),ro,inv,rc);
    if(!build) return py::cast(plan_periodic_gaussian_pair_particle_hole_snapshot(hf,ref,wannier,
        *common_owner.domain,*common_owner.space,basis,warm,reader,i,j,config,options,live,caps));
    using C=std::complex<double>;
    if(!ao || !auxiliary || !gauges || !gauges->dtype().is(py::dtype::of<C>()) || !(gauges->flags()&py::array::c_style)
        || gauges->ndim()!=3 || gauges->shape(0)!=static_cast<py::ssize_t>(hf.state_handle()->n_kpoints())
        || gauges->shape(1)!=static_cast<py::ssize_t>(hf.state_handle()->n_correlated_occupied())
        || gauges->shape(2)!=gauges->shape(1) || reinterpret_cast<std::uintptr_t>(gauges->data())%alignof(C))
        throw py::type_error("Gaussian particle-hole snapshot gauges require contiguous complex128 [K,nocc,nocc]");
    return py::cast(build_periodic_gaussian_pair_particle_hole_snapshot(hf,ref,*ao,*auxiliary,wannier,
        static_cast<const C*>(gauges->data()),gauges->size(),*common_owner.domain,*common_owner.space,basis,warm,
        reader,i,j,index,config,options,live,caps));
}
} // namespace periodic_gaussian_pair_particle_hole_python

void bind_periodic_gaussian_pair_particle_hole(py::module_& m) {
    using namespace vibeqc;
    using U=std::uint64_t;
    namespace binding=periodic_gaussian_pair_particle_hole_python;
    using Live=PeriodicGaussianPairParticleHoleLiveInventory;using Caps=PeriodicGaussianPairParticleHoleCaps;
    using Plan=PeriodicGaussianPairParticleHolePlan;using Diagnostics=PeriodicGaussianPairParticleHoleDiagnostics;
    using Result=PeriodicGaussianPairParticleHoleResult;
    py::class_<Live>(m,"_PeriodicGaussianPairParticleHoleLiveInventory").def(py::init<>())
        .def_readwrite("other_live_numerical_bytes_per_worker",&Live::other_live_numerical_bytes_per_worker)
        .def_readwrite("other_live_control_bytes_per_worker",&Live::other_live_control_bytes_per_worker)
        .def_readwrite("backend_margin_bytes_per_worker",&Live::backend_margin_bytes_per_worker);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianPairParticleHoleCaps");caps.def(py::init<>());
#define GPH_CAP(f) caps.def_readwrite(#f,&Caps::f)
    GPH_CAP(interaction);GPH_CAP(maximum_source_slots);GPH_CAP(maximum_pair_count);GPH_CAP(maximum_owned_numerical_bytes);
    GPH_CAP(maximum_control_storage_bytes);GPH_CAP(maximum_worker_bytes);GPH_CAP(maximum_node_bytes);GPH_CAP(maximum_work_units);
    GPH_CAP(maximum_interaction_control_bytes);GPH_CAP(maximum_factor_panels);GPH_CAP(maximum_tile_calls);
    GPH_CAP(maximum_reciprocal_candidates);GPH_CAP(maximum_image_candidates);
#undef GPH_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianPairParticleHolePlan");
#define GPH_PLAN(f) plan.def_readonly(#f,&Plan::f)
    GPH_PLAN(occupied_count);GPH_PLAN(common_virtual_dimension);GPH_PLAN(pair_count);GPH_PLAN(target_i);GPH_PLAN(target_j);
    GPH_PLAN(target_dimension);GPH_PLAN(maximum_source_dimension);GPH_PLAN(source_slots);GPH_PLAN(domain_generated);
    GPH_PLAN(borrowed_warmstart_numerical_bytes);GPH_PLAN(borrowed_ccsd_numerical_bytes);GPH_PLAN(borrowed_common_numerical_bytes);
    GPH_PLAN(borrowed_basis_active_numeric_bytes);GPH_PLAN(borrowed_owner_control_bytes);
    GPH_PLAN(duplicate_frame_geometry_padding_bytes);GPH_PLAN(duplicate_t_role_padding_bytes);
    GPH_PLAN(accumulator_owned_bytes);GPH_PLAN(maximum_interaction_owned_bytes);GPH_PLAN(maximum_transpose_bytes);
    GPH_PLAN(contraction_phase_bytes);GPH_PLAN(output_numerical_bytes);GPH_PLAN(peak_owned_numerical_bytes);
    GPH_PLAN(outer_control_storage_bytes);GPH_PLAN(control_storage_reservation_bytes);GPH_PLAN(maximum_interaction_control_bytes);
    GPH_PLAN(replicas_per_node);GPH_PLAN(reference_base_node_bytes);GPH_PLAN(worker_bytes);GPH_PLAN(required_node_memory_bytes);
    GPH_PLAN(metadata_work_units);GPH_PLAN(snapshot_validation_work_units);GPH_PLAN(frame_validation_work_units);
    GPH_PLAN(interaction_work_units);GPH_PLAN(transpose_work_units);GPH_PLAN(work_units);GPH_PLAN(factor_panels);GPH_PLAN(tile_calls);
    GPH_PLAN(reciprocal_candidate_evaluations);GPH_PLAN(image_candidate_evaluations);
#undef GPH_PLAN
    plan.def_property_readonly("accumulator",[](const Plan& p){return p.accumulator;});
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianPairParticleHoleDiagnostics");
#define GPH_DIAG(f) diagnostics.def_readonly(#f,&Diagnostics::f)
    GPH_DIAG(completed_source_slots);GPH_DIAG(completed_interactions);GPH_DIAG(completed_factor_panels);GPH_DIAG(completed_tile_calls);
    GPH_DIAG(reciprocal_candidate_evaluations);GPH_DIAG(image_candidate_evaluations);GPH_DIAG(charged_work_units_upper_bound);
    GPH_DIAG(maximum_integral_roundoff_error);GPH_DIAG(maximum_overlap_imaginary_norm);
#undef GPH_DIAG
    diagnostics.def_property_readonly("accumulator",[](const Diagnostics& d){return d.accumulator;});
    auto result=py::class_<Result>(m,"_PeriodicGaussianPairParticleHoleResult");
    result.def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("state",&Result::state_handle).def_property_readonly("context",&Result::context_handle)
        .def("residual_copy",&binding::copy);
#define GPH_RESULT(f) result.def_property_readonly(#f,&Result::f)
    GPH_RESULT(identity_sha256);GPH_RESULT(payload_sha256);GPH_RESULT(consumed_interactions_identity_sha256);
    GPH_RESULT(ccsd_identity_sha256);GPH_RESULT(ccsd_amplitude_payload_sha256);GPH_RESULT(warmstart_identity_sha256);
    GPH_RESULT(origin_provider_identity_sha256);GPH_RESULT(hf_reference_source_identity_sha256);GPH_RESULT(pair_spaces_identity_sha256);
    GPH_RESULT(target_frame_identity_sha256);GPH_RESULT(matched_finite_gaussian_hf_recipe);GPH_RESULT(original_provider_projection_reproduced);
    GPH_RESULT(entire_ccsd_residual);GPH_RESULT(production_dlpno);GPH_RESULT(infinite_source_accuracy_certified);
#undef GPH_RESULT
    using SnapshotPlan=PeriodicGaussianPairParticleHoleSnapshotPlan;
    using SnapshotResult=PeriodicGaussianPairParticleHoleSnapshotResult;
    py::class_<SnapshotPlan>(m,"_PeriodicGaussianPairParticleHoleSnapshotPlan")
        .def_property_readonly("stream",[](const SnapshotPlan& p){return p.stream;})
        .def_readonly("borrowed_reader_numerical_bytes",&SnapshotPlan::borrowed_reader_numerical_bytes)
        .def_readonly("borrowed_reader_table_bytes",&SnapshotPlan::borrowed_reader_table_bytes)
        .def_readonly("reader_control_storage_bytes",&SnapshotPlan::reader_control_storage_bytes)
        .def_readonly("reader_warm_coefficient_padding_bytes",&SnapshotPlan::reader_warm_coefficient_padding_bytes);
    auto snapshot=py::class_<SnapshotResult>(m,"_PeriodicGaussianPairParticleHoleSnapshotResult");
    snapshot.def_property_readonly("memory",[](const SnapshotResult& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const SnapshotResult& r){return r.diagnostics();})
        .def_property_readonly("state",&SnapshotResult::state_handle).def_property_readonly("context",&SnapshotResult::context_handle)
        .def("residual_copy",&binding::snapshot_copy);
#define GPH_SRESULT(f) snapshot.def_property_readonly(#f,&SnapshotResult::f)
    GPH_SRESULT(snapshot_index);GPH_SRESULT(snapshot_identity_sha256);GPH_SRESULT(identity_sha256);GPH_SRESULT(payload_sha256);
    GPH_SRESULT(consumed_interactions_identity_sha256);GPH_SRESULT(warmstart_identity_sha256);GPH_SRESULT(origin_provider_identity_sha256);
    GPH_SRESULT(hf_reference_source_identity_sha256);GPH_SRESULT(pair_spaces_identity_sha256);GPH_SRESULT(converged_snapshot_certified);
    GPH_SRESULT(singles_physically_certified);GPH_SRESULT(original_provider_projection_reproduced);GPH_SRESULT(entire_ccsd_residual);
    GPH_SRESULT(production_dlpno);GPH_SRESULT(infinite_source_accuracy_certified);
#undef GPH_SRESULT
    m.def("_plan_periodic_gaussian_pair_particle_hole_snapshot_diagnostic",[](const PeriodicGaussianRHFResult& hf,
        const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationWannier& wannier,
        const py::object& common_geometry,const py::object& space,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianPairMP2Result& warm,U i,U j,const PeriodicGaussianPairInteractionConfig& config,
        const PeriodicGaussianPairInteractionOptions& options,const Live& live,const Caps& caps,
        const py::list& sc,const py::list& st,const py::list& pt,const binding::ReaderOptions& ro,
        const binding::ReaderCaps& rc,const py::object& coefficient_warm) {
        return binding::snapshot_diagnostic(false,hf,ref,wannier,common_geometry,space,basis,warm,i,j,config,options,live,caps,
            sc,st,pt,ro,rc,coefficient_warm);
    },py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("common_geometry"),py::arg("space"),py::arg("basis"),
        py::arg("warmstart"),py::arg("target_i"),py::arg("target_j"),py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"),
        py::arg("singles_coefficients"),py::arg("singles_amplitudes"),py::arg("pair_amplitudes"),py::arg("reader_options"),py::arg("reader_caps"),
        py::arg("coefficient_warmstart")=py::none());
    m.def("_build_periodic_gaussian_pair_particle_hole_snapshot_diagnostic",[](const PeriodicGaussianRHFResult& hf,
        const PeriodicCorrelationAdmittedReference& ref,const BasisSet& ao,const BasisSet& auxiliary,
        const PeriodicCorrelationWannier& wannier,const py::array& gauges,const py::object& common_geometry,
        const py::object& space,const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairMP2Result& warm,
        U i,U j,const PeriodicGaussianPairInteractionConfig& config,const PeriodicGaussianPairInteractionOptions& options,
        const Live& live,const Caps& caps,const py::list& sc,const py::list& st,const py::list& pt,
        const binding::ReaderOptions& ro,const binding::ReaderCaps& rc,U index,const py::object& coefficient_warm) {
        return binding::snapshot_diagnostic(true,hf,ref,wannier,common_geometry,space,basis,warm,i,j,config,options,live,caps,
            sc,st,pt,ro,rc,coefficient_warm,&ao,&auxiliary,&gauges,index);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("wannier"),py::arg("gauges"),
        py::arg("common_geometry"),py::arg("space"),py::arg("basis"),py::arg("warmstart"),py::arg("target_i"),py::arg("target_j"),
        py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"),py::arg("singles_coefficients"),py::arg("singles_amplitudes"),
        py::arg("pair_amplitudes"),py::arg("reader_options"),py::arg("reader_caps"),py::arg("snapshot_index")=0,
        py::arg("coefficient_warmstart")=py::none());
    m.def("_plan_periodic_gaussian_pair_particle_hole",[](const PeriodicGaussianRHFResult& hf,
        const PeriodicCorrelationAdmittedReference& reference,const PeriodicCorrelationWannier& wannier,
        const py::object& common_geometry,const py::object& space,const PeriodicCorrelationRealLocalBasis& basis,
        const PeriodicGaussianPairMP2Result& warmstart,const PeriodicGaussianPairCCSDResult& ccsd,
        std::uint64_t target_i,std::uint64_t target_j,const PeriodicGaussianPairInteractionConfig& config,
        const PeriodicGaussianPairInteractionOptions& options,const Live& live,const Caps& caps) {
        binding::tiny(hf,basis,caps);const auto c=binding::common(common_geometry,space,live);
        return plan_periodic_gaussian_pair_particle_hole(hf,reference,wannier,*c.domain,*c.space,basis,warmstart,ccsd,
            target_i,target_j,config,options,c.live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("common_geometry"),py::arg("space"),py::arg("basis"),
        py::arg("warmstart"),py::arg("ccsd"),py::arg("target_i"),py::arg("target_j"),py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"));
    m.def("_build_periodic_gaussian_pair_particle_hole",[](const PeriodicGaussianRHFResult& hf,
        const PeriodicCorrelationAdmittedReference& reference,const BasisSet& ao,const BasisSet& auxiliary,
        const PeriodicCorrelationWannier& wannier,const py::array& gauges,const py::object& common_geometry,
        const py::object& space,const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianPairMP2Result& warmstart,
        const PeriodicGaussianPairCCSDResult& ccsd,std::uint64_t target_i,std::uint64_t target_j,
        const PeriodicGaussianPairInteractionConfig& config,const PeriodicGaussianPairInteractionOptions& options,
        const Live& live,const Caps& caps) {
        binding::tiny(hf,basis,caps);const auto c=binding::common(common_geometry,space,live);
        using C=std::complex<double>;
        if(!gauges.dtype().is(py::dtype::of<C>()) || !(gauges.flags()&py::array::c_style) || gauges.ndim()!=3
            || gauges.shape(0)!=static_cast<py::ssize_t>(hf.state_handle()->n_kpoints())
            || gauges.shape(1)!=static_cast<py::ssize_t>(hf.state_handle()->n_correlated_occupied())
            || gauges.shape(2)!=gauges.shape(1) || reinterpret_cast<std::uintptr_t>(gauges.data())%alignof(C))
            throw py::type_error("Gaussian particle-hole gauges require exact contiguous complex128 [K,nocc,nocc]");
        // No callbacks, held GIL, pinned Python owners; no coefficient/T
        // copies or native child escape. Native API borrows immutable views.
        return build_periodic_gaussian_pair_particle_hole(hf,reference,ao,auxiliary,wannier,
            static_cast<const C*>(gauges.data()),gauges.size(),*c.domain,*c.space,basis,warmstart,ccsd,
            target_i,target_j,config,options,c.live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("wannier"),py::arg("gauges"),
        py::arg("common_geometry"),py::arg("space"),py::arg("basis"),py::arg("warmstart"),py::arg("ccsd"),
        py::arg("target_i"),py::arg("target_j"),py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"));
}
