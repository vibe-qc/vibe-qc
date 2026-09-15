// Tiny diagnostic boundary: physical factors come only from native owners.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_pair_interaction.hpp"
#include "vibeqc/periodic_gaussian_pair_space.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif
namespace periodic_gaussian_pair_interaction_python {
using namespace vibeqc;
PeriodicGaussianPairPNOFrameView frame(const py::object& owner) {
    if(py::isinstance<PeriodicGaussianPairPNOResult>(owner)) return PeriodicGaussianPairPNOFrameView(owner.cast<const PeriodicGaussianPairPNOResult&>());
    if(py::isinstance<PeriodicGaussianEmbeddedPairPNOResult>(owner)) return PeriodicGaussianPairPNOFrameView(owner.cast<const PeriodicGaussianEmbeddedPairPNOResult&>());
    if(py::isinstance<PeriodicGaussianGramPairPNOResult>(owner)) return PeriodicGaussianPairPNOFrameView(owner.cast<const PeriodicGaussianGramPairPNOResult&>());
    if(py::isinstance<PeriodicGaussianPairSpace>(owner)) return owner.cast<const PeriodicGaussianPairSpace&>().frame();
    throw py::type_error("pair interaction frame requires a native PNO or pair-space owner");
}
PeriodicGaussianLocalOrbitalFactorLiveInventory live_with_wrappers(
    const py::object& a,const py::object& b,PeriodicGaussianLocalOrbitalFactorLiveInventory live) {
    const auto add=[](std::uint64_t x,std::uint64_t y) {
        if(y>std::numeric_limits<std::uint64_t>::max()-x) throw std::overflow_error("pair interaction wrapper inventory overflows");return x+y;
    };
    const PeriodicGaussianPairSpace* previous=nullptr;
    for(const auto* value:{&a,&b}) {
        if(!py::isinstance<PeriodicGaussianPairSpace>(*value)) continue;
        const auto* space=&value->cast<const PeriodicGaussianPairSpace&>();if(space==previous) continue;previous=space;
        // Direct Gram frames already own their retained exchange matrix.
        // Legacy frames do not; their immediate PairSpace owns it separately.
        const auto count=space->frame().is_direct_gram()?0:space->exchange_integrals().size();
        if(count>std::numeric_limits<std::uint64_t>::max()/8) throw std::overflow_error("pair interaction wrapper extent overflows");
        live.other_retained_bytes_per_worker=add(live.other_retained_bytes_per_worker,
            add(8*count,sizeof(PeriodicGaussianPairSpace)+7*65));
    }
    return live;
}
void tiny(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicGaussianPairPNOFrameView& a,const PeriodicGaussianPairPNOFrameView& b,
    const PeriodicGaussianPairInteractionConfig& config) {
    if(!hf.context_handle() || hf.plan().n_kpoints>8 || hf.plan().n_basis>4
        || hf.context_handle()->inventory().auxiliary.function_count>8
        || basis.memory().occupied_count>8 || basis.memory().virtual_count>8
        || a.retained_dimension()>8 || b.retained_dimension()>8 || config.auxiliary_block>8
        || hf.plan().fock.config.source_caps.maximum_candidate_evaluations>65536)
        throw std::length_error("Gaussian pair interaction diagnostic exceeds tiny shape/source bounds");
}
py::array_t<double> copy(const PeriodicGaussianPairInteractionBlocks& r,unsigned which) {
    const auto& p=r.memory();
    if(p.target_dimension>8 || p.source_dimension>8) throw std::length_error("Gaussian pair interaction copy exceeds tiny shape");
    const auto* values=which==0?r.k_ab_data():which==1?r.j_ba_data():r.overlap_ba_data();
    const auto rows=which==0?p.target_dimension:p.source_dimension,cols=which==0?p.source_dimension:p.target_dimension;
    py::array_t<double> out({static_cast<py::ssize_t>(rows),static_cast<py::ssize_t>(cols)});
    if(rows && cols) std::memcpy(out.mutable_data(),values,8*rows*cols);
    out.attr("setflags")(false);return out;
}
} // namespace periodic_gaussian_pair_interaction_python

void bind_periodic_gaussian_pair_interaction(py::module_& m) {
    using namespace vibeqc;
    namespace binding=periodic_gaussian_pair_interaction_python;
    using Config=PeriodicGaussianPairInteractionConfig;
    using Options=PeriodicGaussianPairInteractionOptions;
    using Caps=PeriodicGaussianPairInteractionCaps;
    using Live=PeriodicGaussianLocalOrbitalFactorLiveInventory;
    using Plan=PeriodicGaussianPairInteractionPlan;
    using Diagnostics=PeriodicGaussianPairInteractionDiagnostics;
    using Result=PeriodicGaussianPairInteractionBlocks;
    py::class_<Config>(m,"_PeriodicGaussianPairInteractionConfig").def(py::init<>())
        .def_readwrite("auxiliary_block",&Config::auxiliary_block).def_readwrite("panel",&Config::panel);
    py::class_<Options>(m,"_PeriodicGaussianPairInteractionOptions").def(py::init<>())
        .def_readwrite("real_projection",&Options::real_projection)
        .def_readwrite("maximum_overlap_imaginary_norm",&Options::maximum_overlap_imaginary_norm);
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianPairInteractionCaps");caps.def(py::init<>());
#define GPINT_CAP(f) caps.def_readwrite(#f,&Caps::f)
    GPINT_CAP(resources);GPINT_CAP(metric);GPINT_CAP(panel);GPINT_CAP(maximum_factor_panels);GPINT_CAP(maximum_tile_calls);
    GPINT_CAP(maximum_image_candidate_evaluations);GPINT_CAP(maximum_leaf_control_bytes);
#undef GPINT_CAP
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianPairInteractionPlan");
#define GPINT_PLAN(f) plan.def_readonly(#f,&Plan::f)
    GPINT_PLAN(n_cells);GPINT_PLAN(n_basis);GPINT_PLAN(n_auxiliary);GPINT_PLAN(target_dimension);GPINT_PLAN(source_dimension);
    GPINT_PLAN(direct_gram_frames);GPINT_PLAN(source_receipt_control_bytes);
    GPINT_PLAN(local_orbital_count);GPINT_PLAN(occupied_slot_i);GPINT_PLAN(occupied_slot_k);GPINT_PLAN(density_count);GPINT_PLAN(row_count);
    GPINT_PLAN(self_inverse_q_count);GPINT_PLAN(auxiliary_block_count);GPINT_PLAN(factor_panels);GPINT_PLAN(tile_calls);
    GPINT_PLAN(retained_output_bytes);GPINT_PLAN(integral_accumulator_bytes);GPINT_PLAN(row_slab_bytes);GPINT_PLAN(norm_workspace_bytes);
    GPINT_PLAN(overlap_phase_bytes);GPINT_PLAN(factor_phase_retained_bytes);GPINT_PLAN(whitener_bytes);GPINT_PLAN(maximum_live_whitener_bytes);
    GPINT_PLAN(retained_partner_panel_bytes);GPINT_PLAN(metric_phase_upper_bytes);GPINT_PLAN(panel_phase_upper_bytes);GPINT_PLAN(peak_owned_numerical_bytes);
    GPINT_PLAN(borrowed_local_numerical_bytes);GPINT_PLAN(borrowed_frame_numerical_bytes);GPINT_PLAN(borrowed_basis_active_numeric_bytes);
    GPINT_PLAN(local_geometry_a);GPINT_PLAN(local_geometry_b);GPINT_PLAN(borrowed_geometry_numerical_bytes);
    GPINT_PLAN(borrowed_geometry_control_bytes);GPINT_PLAN(geometry_validation_work_units);
    GPINT_PLAN(minimum_leaf_control_bytes);GPINT_PLAN(control_storage_reservation_bytes);GPINT_PLAN(replicas_per_node);GPINT_PLAN(reference_base_node_bytes);
    GPINT_PLAN(per_worker_inventoried_bytes);GPINT_PLAN(required_node_memory_bytes);GPINT_PLAN(source_factory_calls);GPINT_PLAN(metric_calls);
    GPINT_PLAN(whitening_calls);GPINT_PLAN(reciprocal_candidate_evaluations_upper_bound);GPINT_PLAN(image_candidate_evaluations_upper_bound);
    GPINT_PLAN(driver_work_units);GPINT_PLAN(work_units);
#undef GPINT_PLAN
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianPairInteractionDiagnostics");
#define GPINT_DIAG(f) diagnostics.def_readonly(#f,&Diagnostics::f)
    GPINT_DIAG(completed_source_count);GPINT_DIAG(completed_metric_count);GPINT_DIAG(completed_whitening_count);
    GPINT_DIAG(completed_factor_panels);GPINT_DIAG(completed_tile_calls);GPINT_DIAG(reciprocal_candidate_evaluations);
    GPINT_DIAG(image_candidate_evaluations);GPINT_DIAG(charged_work_units_upper_bound);GPINT_DIAG(maximum_observed_owned_numerical_bytes);
    GPINT_DIAG(maximum_observed_per_worker_inventoried_bytes);GPINT_DIAG(overlap_imaginary_frobenius_upper_bound);GPINT_DIAG(maximum_integral_roundoff_error);
#undef GPINT_DIAG
    diagnostics.def_property_readonly("real_projection",[](const Diagnostics& d){return d.real_projection;});
    auto result=py::class_<Result>(m,"_PeriodicGaussianPairInteractionBlocks");
    result.def_property_readonly("memory",[](const Result& r){return r.memory();})
        .def_property_readonly("diagnostics",[](const Result& r){return r.diagnostics();})
        .def_property_readonly("state",&Result::state_handle).def_property_readonly("context",&Result::context_handle)
        .def("k_ab_copy",[](const Result& r){return binding::copy(r,0);})
        .def("j_ba_copy",[](const Result& r){return binding::copy(r,1);})
        .def("overlap_ba_copy",[](const Result& r){return binding::copy(r,2);});
#define GPINT_RESULT(f) result.def_property_readonly(#f,&Result::f)
    GPINT_RESULT(identity_sha256);GPINT_RESULT(payload_sha256);GPINT_RESULT(target_frame_identity_sha256);GPINT_RESULT(source_frame_identity_sha256);
    GPINT_RESULT(hf_reference_source_identity_sha256);GPINT_RESULT(consumed_sources_identity_sha256);GPINT_RESULT(production_dlpno);
    GPINT_RESULT(infinite_source_accuracy_certified);GPINT_RESULT(original_provider_projection_reproduced);
    GPINT_RESULT(direct_gram_frames);GPINT_RESULT(target_gram_identity_sha256);GPINT_RESULT(source_gram_identity_sha256);
#undef GPINT_RESULT
    m.def("_plan_periodic_gaussian_pair_interaction_blocks",[](const PeriodicGaussianRHFResult& hf,
        const PeriodicCorrelationAdmittedReference& reference,const PeriodicCorrelationWannier& wannier,
        const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
        const py::object& target_frame,const py::object& source_frame,std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_k,
        const Config& config,const Options& options,const Live& live,const Caps& caps,
        const PeriodicGaussianPairPNOGeometryView* ga,const PeriodicGaussianPairPNOGeometryView* gb) {
        const auto a=binding::frame(target_frame),b=binding::frame(source_frame);binding::tiny(hf,basis,a,b,config);
        return plan_periodic_gaussian_pair_interaction_blocks(hf,reference,wannier,domain,space,basis,a,b,occupied_slot_i,occupied_slot_k,
            config,options,binding::live_with_wrappers(target_frame,source_frame,live),caps,ga,gb);
    },py::arg("hf"),py::arg("reference"),py::arg("wannier"),py::arg("domain"),py::arg("space"),py::arg("basis"),
        py::arg("target_frame"),py::arg("source_frame"),py::arg("occupied_slot_i"),py::arg("occupied_slot_k"),
        py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"),
        py::arg("geometry_a")=nullptr,py::arg("geometry_b")=nullptr);
    m.def("_build_periodic_gaussian_pair_interaction_blocks",[](const PeriodicGaussianRHFResult& hf,
        const PeriodicCorrelationAdmittedReference& reference,const BasisSet& ao,const BasisSet& auxiliary,
        const PeriodicCorrelationWannier& wannier,const py::array& gauges,const PeriodicCorrelationPAODomain& domain,
        const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
        const py::object& target_frame,const py::object& source_frame,std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_k,
        const Config& config,const Options& options,const Live& live,const Caps& caps,
        const PeriodicGaussianPairPNOGeometryView* ga,const PeriodicGaussianPairPNOGeometryView* gb) {
        const auto a=binding::frame(target_frame),b=binding::frame(source_frame);binding::tiny(hf,basis,a,b,config);
        using C=std::complex<double>;
        if(!gauges.dtype().is(py::dtype::of<C>()) || !(gauges.flags()&py::array::c_style)
            || gauges.ndim()!=3 || gauges.shape(0)!=static_cast<py::ssize_t>(hf.state_handle()->n_kpoints())
            || gauges.shape(1)!=static_cast<py::ssize_t>(hf.state_handle()->n_correlated_occupied()) || gauges.shape(2)!=gauges.shape(1)
            || reinterpret_cast<std::uintptr_t>(gauges.data())%alignof(C))
            throw py::type_error("Gaussian pair interaction gauges require exact contiguous complex128 [K,nocc,nocc]");
        // Pinned py::object owners and held GIL prohibit an escaping borrow
        // or another Python thread consuming the frame during this call.
        return build_periodic_gaussian_pair_interaction_blocks(hf,reference,ao,auxiliary,wannier,
            static_cast<const C*>(gauges.data()),gauges.size(),domain,space,basis,a,b,
            occupied_slot_i,occupied_slot_k,config,options,binding::live_with_wrappers(target_frame,source_frame,live),caps,ga,gb);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),py::arg("wannier"),py::arg("gauges"),
        py::arg("domain"),py::arg("space"),py::arg("basis"),py::arg("target_frame"),py::arg("source_frame"),
        py::arg("occupied_slot_i"),py::arg("occupied_slot_k"),py::arg("config"),py::arg("options"),py::arg("live"),py::arg("caps"),
        py::arg("geometry_a")=nullptr,py::arg("geometry_b")=nullptr);
}
