// Tiny connected native localization diagnostics; included by bindings.cpp.
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_gaussian_localization.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_localization(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicGaussianLocalizationOptions;
    using Caps=PeriodicGaussianLocalizationCaps;
    using Memory=PeriodicGaussianLocalizationMemoryPlan;
    using Diagnostics=PeriodicGaussianLocalizationDiagnostics;
    using Progress=PeriodicGaussianLocalizationProgress;
    using Stage=PeriodicGaussianLocalizationStage;
    using Result=PeriodicGaussianLocalizationResult;
    auto options=py::class_<Options>(m,"_PeriodicGaussianLocalizationOptions"); options.def(py::init<>());
#define GLS_OPTION(field) options.def_readwrite(#field,&Options::field)
    GLS_OPTION(source); GLS_OPTION(iao); GLS_OPTION(seed); GLS_OPTION(pm); GLS_OPTION(optimizer); GLS_OPTION(wannier);
#undef GLS_OPTION
    auto caps=py::class_<Caps>(m,"_PeriodicGaussianLocalizationCaps"); caps.def(py::init<>());
#define GLS_CAP(field) caps.def_readwrite(#field,&Caps::field)
    GLS_CAP(maximum_owned_numerical_bytes); GLS_CAP(maximum_control_storage_bytes);
    GLS_CAP(maximum_point_owners); GLS_CAP(maximum_work_units);
    GLS_CAP(maximum_total_image_candidates); GLS_CAP(maximum_optimizer_work_units);
#undef GLS_CAP
    auto memory=py::class_<Memory>(m,"_PeriodicGaussianLocalizationMemoryPlan");
#define GLS_MEMORY(field) memory.def_readonly(#field,&Memory::field)
    GLS_MEMORY(n_points); GLS_MEMORY(n_basis); GLS_MEMORY(n_minimal); GLS_MEMORY(n_active);
    GLS_MEMORY(borrowed_gaussian_numerical_bytes); GLS_MEMORY(other_live_numerical_bytes);
    GLS_MEMORY(point_plan_record_bytes); GLS_MEMORY(point_owner_record_bytes);
    GLS_MEMORY(source_identity_character_bytes); GLS_MEMORY(fixed_control_reservation_bytes);
    GLS_MEMORY(peak_control_storage_bytes); GLS_MEMORY(seed_phase_owned_bytes); GLS_MEMORY(source_phase_owned_bytes);
    GLS_MEMORY(optimizer_phase_owned_bytes); GLS_MEMORY(wannier_phase_owned_bytes);
    GLS_MEMORY(peak_owned_numerical_bytes); GLS_MEMORY(output_numerical_bytes);
    GLS_MEMORY(seed_required_node_bytes); GLS_MEMORY(source_required_node_bytes);
    GLS_MEMORY(optimizer_required_node_bytes); GLS_MEMORY(wannier_required_node_bytes); GLS_MEMORY(required_node_memory_bytes);
    GLS_MEMORY(planned_source_image_candidates); GLS_MEMORY(planned_total_image_visits);
    GLS_MEMORY(source_planning_work_units); GLS_MEMORY(source_factory_work_reservation);
    GLS_MEMORY(input_verification_work_units); GLS_MEMORY(callback_work_reservation);
    GLS_MEMORY(wannier_work_reservation); GLS_MEMORY(optimizer_work_cap);
#undef GLS_MEMORY
    auto diagnostics=py::class_<Diagnostics>(m,"_PeriodicGaussianLocalizationDiagnostics");
#define GLS_DIAGNOSTIC(field) diagnostics.def_readonly(#field,&Diagnostics::field)
    GLS_DIAGNOSTIC(charged_work_units); GLS_DIAGNOSTIC(image_candidate_visits);
    GLS_DIAGNOSTIC(constructed_iao_points); GLS_DIAGNOSTIC(callback_input_checks); GLS_DIAGNOSTIC(source);
#undef GLS_DIAGNOSTIC
    py::enum_<Stage>(m,"_PeriodicGaussianLocalizationStage")
        .value("SEED_READY",Stage::SeedReady).value("POINT_READY",Stage::PointReady)
        .value("OPTIMIZATION",Stage::Optimization).value("WANNIER_READY",Stage::WannierReady).value("FINISHED",Stage::Finished);
    py::class_<Progress>(m,"_PeriodicGaussianLocalizationProgress")
        .def_readonly("stage",&Progress::stage).def_readonly("completed_points",&Progress::completed_points)
        .def_readonly("point_count",&Progress::point_count).def_readonly("charged_work_units",&Progress::charged_work_units)
        .def_readonly("optimizer",&Progress::optimizer);
    py::class_<Result>(m,"_PeriodicGaussianLocalizationResult")
        .def_property_readonly("contract_version",&Result::contract_version)
        .def_property_readonly("optimizer",&Result::optimizer,py::return_value_policy::reference_internal)
        .def_property_readonly("wannier",&Result::wannier,py::return_value_policy::reference_internal)
        .def_property_readonly("converged",&Result::converged)
        .def_property_readonly("hf_basis_source_authenticated",&Result::hf_basis_source_authenticated)
        .def_property_readonly("infinite_image_tail_certified",&Result::infinite_image_tail_certified)
        .def_property_readonly("memory",[](const Result& r) { return r.memory(); })
        .def_property_readonly("diagnostics",[](const Result& r) { return r.diagnostics(); })
        .def_property_readonly("gaussian_input_identity_sha256",&Result::gaussian_input_identity_sha256)
        .def_property_readonly("ordered_source_identity_sha256",&Result::ordered_source_identity_sha256)
        .def_property_readonly("ordered_reference_match_identity_sha256",&Result::ordered_reference_match_identity_sha256)
        .def_property_readonly("source_image_cutoff_bohr",&Result::source_image_cutoff_bohr)
        .def_property_readonly("localization_identity_sha256",&Result::localization_identity_sha256);
    m.def("_verify_periodic_gaussian_localization_inputs",[](
        const Result& result,const PeriodicCorrelationAdmittedReference& reference,
        const BasisSet& ao,const BasisSet& minimal,const PeriodicSystem& system,
        std::uint64_t maximum_work_units) {
        if (!reference.state_handle() || reference.state().n_kpoints()>8
            || ao.nbasis()>12 || minimal.nbasis()>12 || ao.nshells()>16 || minimal.nshells()>16
            || system.unit_cell.size()>8 || maximum_work_units>40000000000000ULL)
            throw std::length_error("Gaussian localization input verification exceeds tiny diagnostic limits");
        if (!maximum_work_units || result.memory().input_verification_work_units>maximum_work_units)
            throw std::length_error("Gaussian localization input verification work cap exceeded");
        result.verify_original_inputs(reference,ao,minimal,system);
    },py::arg("localization"),py::arg("reference"),py::arg("ao_basis"),py::arg("minimal_basis"),
      py::arg("system"),py::arg("maximum_work_units"));
    m.def("_localize_periodic_gaussian_occupied",[](
        const PeriodicCorrelationAdmittedReference& reference,const BasisSet& ao,const BasisSet& minimal,
        const PeriodicSystem& system,std::uint64_t other,Options options,
        PeriodicGaussianBlochIAOCaps source,Caps caps,const py::object& callback) {
        if (!reference.state_handle()) throw std::invalid_argument("Gaussian localization diagnostic requires a live reference");
        if (reference.state().n_kpoints()>8 || reference.state().n_correlated_occupied()>4
            || ao.nbasis()>12 || minimal.nbasis()>12 || ao.nshells()>16 || minimal.nshells()>16
            || system.unit_cell.size()>8 || other>(8U<<20)
            || options.optimizer.maximum_iterations>256 || options.optimizer.maximum_line_search_trials>16
            || options.optimizer.jacobi_max_sweeps>64 || options.seed.jacobi_max_sweeps>64 || options.iao.jacobi_max_sweeps>64
            || caps.maximum_owned_numerical_bytes>(8U<<20) || caps.maximum_control_storage_bytes>(4U<<20)
            || caps.maximum_point_owners>8 || caps.maximum_work_units>40000000000000ULL
            || caps.maximum_total_image_candidates>2000000 || caps.maximum_optimizer_work_units>10000000000ULL
            || source.maximum_owned_numerical_bytes>(8U<<20) || source.maximum_work_units>20000000000000ULL
            || source.maximum_atom_count>8 || source.maximum_shell_count>32 || source.maximum_contraction_count>64
            || source.maximum_primitive_numeric_lanes>1024 || source.maximum_basis_content_wire_bytes>65536
            || source.maximum_borrowed_active_numeric_bytes>(8U<<20) || source.maximum_panel_pairs>64
            || source.maximum_total_image_candidates>250000)
            throw std::length_error("Gaussian localization diagnostic exceeds tiny shape/source/iteration caps");
        if (!callback.is_none() && !PyCallable_Check(callback.ptr()))
            throw std::invalid_argument("Gaussian localization callback must be callable or None");
        const auto notify=[](const Progress& event,void* context) {
            (*static_cast<const py::object*>(context))(py::cast(Progress(event),py::return_value_policy::move));
        };
        // Hold the GIL and actual argument owners; controls are COPIED. The
        // core rechecks the original cell/basis after callbacks. Scalar Python
        // snapshots/reference bookkeeping are separately bounded diagnostic
        // control storage, not additional native numerical matrix copies.
        return localize_periodic_gaussian_occupied(reference,ao,minimal,system,other,options,source,caps,
            callback.is_none()?nullptr:+notify,callback.is_none()?nullptr:const_cast<py::object*>(&callback));
    },py::arg("reference"),py::arg("ao_basis"),py::arg("minimal_basis"),py::arg("system"),
      py::arg("other_live_numerical_bytes"),py::arg("options"),py::arg("source_caps"),py::arg("caps"),
      py::arg("callback")=py::none());
}
