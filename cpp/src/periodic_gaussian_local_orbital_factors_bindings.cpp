// Included by bindings.cpp. Tiny actual-HF-origin diagnostic, no array-factor route.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include "vibeqc/periodic_gaussian_local_orbital_factors.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_local_orbital_factors(py::module_& m) {
    using namespace vibeqc;
    using Config=PeriodicGaussianLocalOrbitalFactorConfig;
    using Live=PeriodicGaussianLocalOrbitalFactorLiveInventory;
    using Caps=PeriodicGaussianLocalOrbitalFactorCaps;
    using Plan=PeriodicGaussianLocalOrbitalFactorPlan;
    using Diagnostics=PeriodicGaussianLocalOrbitalFactorDiagnostics;
    using Panel=PeriodicGaussianLocalOrbitalFactorPanel;
    using Selection=PeriodicGaussianLocalOrbitalFactorSelection;
    using C=std::complex<double>;
    py::class_<Selection>(m,"_PeriodicGaussianLocalOrbitalFactorSelection")
        .def(py::init<>())
        .def_readwrite("left_begin",&Selection::left_begin)
        .def_readwrite("left_count",&Selection::left_count)
        .def_readwrite("right_begin",&Selection::right_begin)
        .def_readwrite("right_count",&Selection::right_count);
    py::class_<Config>(m,"_PeriodicGaussianLocalOrbitalFactorConfig").def(py::init<>())
        .def_readwrite("ao_pair_block",&Config::ao_pair_block);
    py::class_<Live>(m,"_PeriodicGaussianLocalOrbitalFactorLiveInventory").def(py::init<>())
        .def_readwrite("other_retained_bytes_per_worker",&Live::other_retained_bytes_per_worker)
        .def_readwrite("other_transient_bytes_per_worker",&Live::other_transient_bytes_per_worker)
        .def_readwrite("fixed_backend_margin_bytes_per_worker",&Live::fixed_backend_margin_bytes_per_worker);
    py::class_<Caps>(m,"_PeriodicGaussianLocalOrbitalFactorCaps").def(py::init<>())
        .def_readwrite("resources",&Caps::resources).def_readwrite("tile",&Caps::tile)
        .def_readwrite("maximum_tile_calls",&Caps::maximum_tile_calls)
        .def_readwrite("maximum_image_candidate_evaluations",&Caps::maximum_image_candidate_evaluations);
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianLocalOrbitalFactorPlan");
#define GLOCAL_PLAN(field) plan.def_readonly(#field,&Plan::field)
    GLOCAL_PLAN(n_cells); GLOCAL_PLAN(n_basis); GLOCAL_PLAN(n_auxiliary);
    GLOCAL_PLAN(occupied_count); GLOCAL_PLAN(virtual_count); GLOCAL_PLAN(orbital_count);
    GLOCAL_PLAN(q_index); GLOCAL_PLAN(auxiliary_begin); GLOCAL_PLAN(auxiliary_count);
    GLOCAL_PLAN(ao_pair_block); GLOCAL_PLAN(tile_calls); GLOCAL_PLAN(retained_factor_bytes);
    GLOCAL_PLAN(retained_index_bytes); GLOCAL_PLAN(retained_output_bytes); GLOCAL_PLAN(compensation_bytes);
    GLOCAL_PLAN(coefficient_panel_bytes); GLOCAL_PLAN(coefficient_scratch_bytes);
    GLOCAL_PLAN(driver_owned_numerical_bytes); GLOCAL_PLAN(maximum_tile_owned_numerical_bytes);
    GLOCAL_PLAN(peak_owned_numerical_bytes); GLOCAL_PLAN(resident_whitener_bytes);
    GLOCAL_PLAN(borrowed_basis_active_numeric_bytes); GLOCAL_PLAN(caller_gauge_bytes);
    GLOCAL_PLAN(live_wannier_bytes); GLOCAL_PLAN(live_domain_bytes); GLOCAL_PLAN(live_space_bytes);
    GLOCAL_PLAN(live_basis_bytes); GLOCAL_PLAN(basis_index_alias_bytes); GLOCAL_PLAN(state_resident_bytes);
    GLOCAL_PLAN(macro_fixed_object_bytes); GLOCAL_PLAN(tile_fixed_object_bytes);
    GLOCAL_PLAN(replicas_per_node); GLOCAL_PLAN(reference_base_node_bytes);
    GLOCAL_PLAN(per_replica_inventoried_bytes); GLOCAL_PLAN(required_node_memory_bytes);
    GLOCAL_PLAN(reciprocal_candidate_evaluations); GLOCAL_PLAN(image_candidate_evaluations_upper_bound);
    GLOCAL_PLAN(coefficient_work_units); GLOCAL_PLAN(contraction_term_count);
    GLOCAL_PLAN(driver_work_units); GLOCAL_PLAN(work_units);
#undef GLOCAL_PLAN
    plan.def_property_readonly("config",[](const Plan& p){return p.config;})
        .def_property_readonly("selection",[](const Plan& p){return p.selection;})
        .def_property_readonly("live",[](const Plan& p){return p.live;})
        .def_property_readonly("caps",[](const Plan& p){return p.caps;})
        .def_property_readonly("tile_config",[](const Plan& p){return p.tile_config;})
        .def_property_readonly("plan_identity_sha256",&Plan::plan_identity_sha256);
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianLocalOrbitalFactorDiagnostics");
#define GLOCAL_DIAG(field) diagnostics.def_readonly(#field,&Diagnostics::field)
    GLOCAL_DIAG(completed_tile_calls); GLOCAL_DIAG(reciprocal_candidate_evaluations);
    GLOCAL_DIAG(image_candidate_evaluations); GLOCAL_DIAG(charged_work_units_upper_bound);
    GLOCAL_DIAG(maximum_observed_owned_numerical_bytes);
    GLOCAL_DIAG(maximum_observed_per_replica_inventoried_bytes);
#undef GLOCAL_DIAG
    py::class_<Panel>(m,"_PeriodicGaussianLocalOrbitalFactorPanel")
        .def_property_readonly("context",&Panel::context_handle)
        .def_property_readonly("state",&Panel::state_handle)
        .def_property_readonly("memory",[](const Panel& p){return p.memory();})
        .def_property_readonly("diagnostics",[](const Panel& p){return p.diagnostics();})
        .def_property_readonly("q_index",&Panel::q_index)
        .def_property_readonly("auxiliary_begin",&Panel::auxiliary_begin)
        .def_property_readonly("auxiliary_count",&Panel::auxiliary_count)
        .def_property_readonly("orbital_count",&Panel::orbital_count)
        .def_property_readonly("selection",&Panel::selection)
        .def_property_readonly("virtual_selection",&Panel::virtual_selection)
        .def("occupied",[](const Panel& p,std::size_t i){
            const auto label=p.occupied(i); return py::make_tuple(label.occupied_index,label.cell);
        }).def("element",&Panel::element)
        .def_property_readonly("finite_image_reference",&Panel::finite_image_reference)
        .def_property_readonly("ao_image_source_certified",&Panel::ao_image_source_certified)
        .def_property_readonly("density_symmetry_certified",&Panel::density_symmetry_certified)
        .def_property_readonly("matched_finite_gaussian_hf_recipe",&Panel::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("bitwise_hf_factor_consumption_verified",&Panel::bitwise_hf_factor_consumption_verified)
        .def_property_readonly("hf_reference_source_identity_sha256",&Panel::hf_reference_source_identity_sha256)
        .def_property_readonly("identity_sha256",&Panel::identity_sha256)
        .def_property_readonly("payload_sha256",&Panel::payload_sha256)
        .def_property_readonly("local_basis_identity_sha256",&Panel::local_basis_identity_sha256)
        .def_property_readonly("basis_certificate_identity_sha256",&Panel::basis_certificate_identity_sha256)
        .def_property_readonly("source_identity_sha256",&Panel::source_identity_sha256)
        .def_property_readonly("conjugate_source_identity_sha256",&Panel::conjugate_source_identity_sha256)
        .def_property_readonly("whitener_payload_identity_sha256",&Panel::whitener_payload_identity_sha256)
        .def_property_readonly("consumed_tiles_identity_sha256",&Panel::consumed_tiles_identity_sha256)
        .def("tensor_copy",[](const Panel& p){
            const auto& plan=p.memory();
            if(plan.n_cells>8 || plan.n_basis>4 || plan.n_auxiliary>4 || plan.orbital_count>8)
                throw std::length_error("Gaussian local factor diagnostic copy exceeds tiny shape");
            const auto* data=p.data();
            py::array_t<C> values({static_cast<py::ssize_t>(plan.auxiliary_count),
                static_cast<py::ssize_t>(plan.selection.left_count),static_cast<py::ssize_t>(plan.selection.right_count)});
            std::memcpy(values.mutable_data(),data,plan.retained_factor_bytes); return values;
        });
    const auto tiny=[](const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationRealLocalBasis& basis,
                       const PeriodicGaussianReciprocalSource& source){
        if(!hf.context_handle() || hf.plan().n_kpoints>8 || hf.plan().n_basis>4
            || hf.context_handle()->inventory().auxiliary.function_count>4
            || basis.memory().orbital_count>8 || basis.memory().occupied_count>4
            || source.candidate_count()>65536)
            throw std::length_error("Gaussian local factor diagnostic exceeds tiny shape/source bounds");
    };
    const auto bounded=[](const Plan& p){
        if(p.peak_owned_numerical_bytes>(16U<<20) || p.tile_calls>512)
            throw std::length_error("Gaussian local factor diagnostic exceeds tiny memory/tile bounds");
    };
    m.def("_plan_periodic_gaussian_local_orbital_factor_panel",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
        const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
        std::uint64_t begin,std::uint64_t count,Config config,Live live,Caps caps){
        tiny(hf,basis,source);
        auto p=plan_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,wannier,domain,space,basis,
            begin,count,config,live,caps); bounded(p); return p;
    },py::arg("hf"),py::arg("reference"),py::arg("source"),py::arg("whitener"),py::arg("wannier"),
      py::arg("domain"),py::arg("space"),py::arg("basis"),py::arg("auxiliary_begin"),py::arg("auxiliary_count"),
      py::arg("config"),py::arg("live"),py::arg("caps"));
    m.def("_build_periodic_gaussian_local_orbital_factor_panel",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
        py::array gauges,const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
        const PeriodicCorrelationRealLocalBasis& basis,std::uint64_t begin,std::uint64_t count,
        Config config,Live live,Caps caps){
        tiny(hf,basis,source);
        const auto p=plan_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,wannier,domain,space,basis,
            begin,count,config,live,caps); bounded(p);
        if(!gauges.dtype().is(py::dtype::of<C>()) || !(gauges.flags()&py::array::c_style)
            || gauges.ndim()!=3 || gauges.shape(0)!=static_cast<py::ssize_t>(p.n_cells)
            || gauges.shape(1)!=static_cast<py::ssize_t>(reference.state().n_correlated_occupied())
            || gauges.shape(2)!=gauges.shape(1))
            throw std::invalid_argument("Gaussian local gauge requires contiguous complex128 [Nk,no,no]");
        // Hold GIL, immutable borrowed view: no hidden NumPy forcecast/copy,
        // arbitrary integral callback, fake source flags or legacy reader.
        return build_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,ao,auxiliary,wannier,
            static_cast<const C*>(gauges.data()),static_cast<std::size_t>(gauges.size()),domain,space,basis,
            begin,count,config,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("source"),py::arg("whitener"),py::arg("ao_basis"),
      py::arg("auxiliary_basis"),py::arg("wannier"),py::arg("gauges").noconvert(),py::arg("domain"),
      py::arg("space"),py::arg("basis"),py::arg("auxiliary_begin"),py::arg("auxiliary_count"),
      py::arg("config"),py::arg("live"),py::arg("caps"));
    m.def("_plan_periodic_gaussian_local_orbital_factor_block",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        const PeriodicCorrelationWannier& wannier,const PeriodicCorrelationPAODomain& domain,
        const PeriodicCorrelationPAOSpace& space,const PeriodicCorrelationRealLocalBasis& basis,
        std::uint64_t begin,std::uint64_t count,Selection selection,Config config,Live live,Caps caps){
        tiny(hf,basis,source);
        auto p=plan_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,wannier,domain,space,basis,
            begin,count,selection,config,live,caps); bounded(p); return p;
    },py::arg("hf"),py::arg("reference"),py::arg("source"),py::arg("whitener"),py::arg("wannier"),
      py::arg("domain"),py::arg("space"),py::arg("basis"),py::arg("auxiliary_begin"),py::arg("auxiliary_count"),
      py::arg("selection"),py::arg("config"),py::arg("live"),py::arg("caps"));
    m.def("_build_periodic_gaussian_local_orbital_factor_block",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicGaussianReciprocalSource& source,const PeriodicGaussianMetricWhitener& w,
        const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier& wannier,
        py::array gauges,const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationPAOSpace& space,
        const PeriodicCorrelationRealLocalBasis& basis,std::uint64_t begin,std::uint64_t count,
        Selection selection,Config config,Live live,Caps caps){
        tiny(hf,basis,source);
        const auto p=plan_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,wannier,domain,space,basis,
            begin,count,selection,config,live,caps); bounded(p);
        if(!gauges.dtype().is(py::dtype::of<C>()) || !(gauges.flags()&py::array::c_style)
            || gauges.ndim()!=3 || gauges.shape(0)!=static_cast<py::ssize_t>(p.n_cells)
            || gauges.shape(1)!=static_cast<py::ssize_t>(reference.state().n_correlated_occupied())
            || gauges.shape(2)!=gauges.shape(1))
            throw std::invalid_argument("Gaussian local gauge requires contiguous complex128 [Nk,no,no]");
        return build_periodic_gaussian_local_orbital_factor_panel(hf,reference,source,w,ao,auxiliary,wannier,
            static_cast<const C*>(gauges.data()),static_cast<std::size_t>(gauges.size()),domain,space,basis,
            begin,count,selection,config,live,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("source"),py::arg("whitener"),py::arg("ao_basis"),
      py::arg("auxiliary_basis"),py::arg("wannier"),py::arg("gauges").noconvert(),py::arg("domain"),
      py::arg("space"),py::arg("basis"),py::arg("auxiliary_begin"),py::arg("auxiliary_count"),
      py::arg("selection"),py::arg("config"),py::arg("live"),py::arg("caps"));
}
