// Included by bindings.cpp. Tiny actual-HF diagnostics, no caller mapping.
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_gaussian_occupied_pao_domain.hpp"
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py=pybind11;
#endif

void bind_periodic_gaussian_occupied_pao_domain(py::module_& m) {
    using namespace vibeqc;
    using Options=PeriodicCorrelationOccupiedPAODomainOptions;
    using Inventory=PeriodicCorrelationOccupiedPAODomainInventory;
    using Caps=PeriodicGaussianOccupiedPAODomainCaps;
    using Plan=PeriodicGaussianOccupiedPAODomainMemoryPlan;
    using Result=PeriodicGaussianOccupiedPAODomain;
    py::class_<Caps>(m,"_PeriodicGaussianOccupiedPAODomainCaps").def(py::init<>())
        .def_readwrite("selector",&Caps::selector)
        .def_readwrite("maximum_owned_numerical_bytes",&Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_worker_bytes",&Caps::maximum_worker_bytes)
        .def_readwrite("maximum_node_bytes",&Caps::maximum_node_bytes)
        .def_readwrite("maximum_work_units",&Caps::maximum_work_units);
    auto plan=py::class_<Plan>(m,"_PeriodicGaussianOccupiedPAODomainMemoryPlan");
#define GDOMAIN_PLAN(field) plan.def_readonly(#field,&Plan::field)
    GDOMAIN_PLAN(selector); GDOMAIN_PLAN(atom_mapping_bytes); GDOMAIN_PLAN(borrowed_gaussian_numerical_bytes);
    GDOMAIN_PLAN(additional_control_storage_bytes); GDOMAIN_PLAN(peak_owned_numerical_bytes);
    GDOMAIN_PLAN(worker_bytes); GDOMAIN_PLAN(required_node_memory_bytes);
    GDOMAIN_PLAN(physical_input_validation_work_units); GDOMAIN_PLAN(mapping_work_units); GDOMAIN_PLAN(work_units);
#undef GDOMAIN_PLAN
    py::class_<Result>(m,"_PeriodicGaussianOccupiedPAODomain")
        .def_property_readonly("domain",&Result::domain,py::return_value_policy::reference_internal)
        .def_property_readonly("memory",[](const Result& r) { return r.memory(); })
        .def_property_readonly("context",&Result::context_handle)
        .def_property_readonly("matched_finite_gaussian_hf_recipe",&Result::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("production_dlpno",&Result::production_dlpno)
        .def_property_readonly("infinite_source_accuracy_certified",&Result::infinite_source_accuracy_certified)
        .def_property_readonly("hf_reference_source_identity_sha256",&Result::hf_reference_source_identity_sha256)
        .def_property_readonly("localization_identity_sha256",&Result::localization_identity_sha256)
        .def_property_readonly("identity_sha256",&Result::identity_sha256);
    const auto tiny=[](const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
                      const PeriodicGaussianLocalizationResult& localization) {
        if (!ref.state_handle() || ref.state().n_kpoints()>8 || ref.state().n_basis()>12
            || ref.state().n_correlated_occupied()>4 || hf.plan().atom_count>8
            || localization.memory().n_minimal>12)
            throw std::length_error("Gaussian occupied PAO domain exceeds tiny diagnostic dimensions");
    };
    const auto bounded=[](const Plan& p) {
        if (p.peak_owned_numerical_bytes>(8U<<20) || p.required_node_memory_bytes>(128U<<20)
            || p.work_units>100000000000000ULL)
            throw std::length_error("Gaussian occupied PAO domain exceeds tiny diagnostic memory/work limits");
    };
    m.def("_plan_periodic_gaussian_occupied_pao_domain",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
        const PeriodicGaussianLocalizationResult& localization,std::uint64_t i,
        Options options,Inventory inventory,Caps caps) {
        tiny(hf,ref,localization);
        const auto p=plan_periodic_gaussian_occupied_pao_domain(hf,ref,localization,i,options,inventory,caps);
        bounded(p); return p;
    },py::arg("hf"),py::arg("reference"),py::arg("localization"),py::arg("home_active_occupied_index"),
      py::arg("options"),py::arg("inventory"),py::arg("caps"));
    m.def("_select_periodic_gaussian_occupied_pao_domain",[tiny,bounded](
        const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
        const BasisSet& ao,const BasisSet& auxiliary,const BasisSet& minimal,const PeriodicSystem& system,
        const PeriodicGaussianLocalizationResult& localization,std::uint64_t i,
        Options options,Inventory inventory,Caps caps) {
        tiny(hf,ref,localization);
        bounded(plan_periodic_gaussian_occupied_pao_domain(hf,ref,localization,i,options,inventory,caps));
        // GIL held; C++ borrows the actual argument owners. No callback or
        // NumPy conversion can replace their numerical payloads mid-call.
        return select_periodic_gaussian_occupied_pao_domain(hf,ref,ao,auxiliary,minimal,system,
            localization,i,options,inventory,caps);
    },py::arg("hf"),py::arg("reference"),py::arg("ao_basis"),py::arg("auxiliary_basis"),
      py::arg("minimal_basis"),py::arg("system"),py::arg("localization"),py::arg("home_active_occupied_index"),
      py::arg("options"),py::arg("inventory"),py::arg("caps"));
}
