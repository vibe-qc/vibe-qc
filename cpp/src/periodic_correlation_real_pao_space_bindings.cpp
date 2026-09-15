// Tiny diagnostic access to the native real-PAO preparation owner.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include "vibeqc/periodic_correlation_real_pao_space.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_real_pao_space(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationRealPAOSpaceOptions;
    using Caps = PeriodicCorrelationRealPAOSpaceCaps;
    using Memory = PeriodicCorrelationRealPAOSpaceMemoryPlan;
    using Diagnostics = PeriodicCorrelationRealPAOSpaceDiagnostics;
    using Owner = PeriodicCorrelationRealPAOSpace;
    m.attr("_PERIODIC_CORRELATION_REAL_PAO_SPACE_CONTRACT_VERSION") = py::int_(1);
#define REAL_PAO_OPTION(name) .def_readwrite(#name, &Options::name)
    py::class_<Options>(m,"_PeriodicCorrelationRealPAOSpaceOptions").def(py::init<>())
        REAL_PAO_OPTION(algebra) REAL_PAO_OPTION(maximum_overlap_projection_error)
        REAL_PAO_OPTION(maximum_fock_projection_error) REAL_PAO_OPTION(maximum_overlap_spectral_uncertainty)
        REAL_PAO_OPTION(maximum_original_metric_error) REAL_PAO_OPTION(maximum_original_fock_error)
        REAL_PAO_OPTION(maximum_original_projector_relation_error)
        REAL_PAO_OPTION(coefficient_tr_absolute_tolerance) REAL_PAO_OPTION(coefficient_tr_relative_tolerance);
#undef REAL_PAO_OPTION
    py::class_<Caps>(m,"_PeriodicCorrelationRealPAOSpaceCaps").def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes",&Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_work_units",&Caps::maximum_work_units);
#define REAL_PAO_MEMORY(name) .def_readonly(#name, &Memory::name)
    py::class_<Memory>(m,"_PeriodicCorrelationRealPAOSpaceMemoryPlan")
        REAL_PAO_MEMORY(compact) REAL_PAO_MEMORY(n_cells) REAL_PAO_MEMORY(n_basis)
        REAL_PAO_MEMORY(coefficient_tr_phase_bytes) REAL_PAO_MEMORY(peak_owned_numerical_bytes)
        REAL_PAO_MEMORY(work_units) REAL_PAO_MEMORY(required_node_memory_bytes);
#undef REAL_PAO_MEMORY
#define REAL_PAO_DIAGNOSTIC(name) .def_readonly(#name, &Diagnostics::name)
    py::class_<Diagnostics>(m,"_PeriodicCorrelationRealPAOSpaceDiagnostics")
        REAL_PAO_DIAGNOSTIC(overlap_projection_frobenius_upper_bound)
        REAL_PAO_DIAGNOSTIC(fock_projection_frobenius_upper_bound)
        REAL_PAO_DIAGNOSTIC(overlap_eigenvector_gram_error_upper_bound)
        REAL_PAO_DIAGNOSTIC(overlap_reconstruction_error_upper_bound)
        REAL_PAO_DIAGNOSTIC(overlap_polar_distance_upper_bound)
        REAL_PAO_DIAGNOSTIC(overlap_eigenvalue_error_upper_bound)
        REAL_PAO_DIAGNOSTIC(rank_cutoff_lower_bound) REAL_PAO_DIAGNOSTIC(rank_cutoff_upper_bound)
        REAL_PAO_DIAGNOSTIC(negative_tolerance_lower_bound) REAL_PAO_DIAGNOSTIC(minimum_rank_margin_lower_bound)
        REAL_PAO_DIAGNOSTIC(projected_projector_reconstruction_error_upper_bound)
        REAL_PAO_DIAGNOSTIC(original_metric_frobenius_error_upper_bound)
        REAL_PAO_DIAGNOSTIC(original_fock_frobenius_error_upper_bound)
        REAL_PAO_DIAGNOSTIC(original_projector_relation_frobenius_error_upper_bound)
        REAL_PAO_DIAGNOSTIC(inspected_kpoints) REAL_PAO_DIAGNOSTIC(inspected_trim_points)
        REAL_PAO_DIAGNOSTIC(maximum_coefficient_tr_error) REAL_PAO_DIAGNOSTIC(coefficient_tr_frobenius_upper_bound);
#undef REAL_PAO_DIAGNOSTIC
    py::class_<Owner>(m,"_PeriodicCorrelationRealPAOSpace")
        .def_property_readonly("contract_version",&Owner::contract_version)
        .def_property_readonly("space",&Owner::space,py::return_value_policy::reference_internal)
        .def_property_readonly("options",[](const Owner& o) { return o.options(); })
        .def_property_readonly("memory",[](const Owner& o) { return o.memory(); })
        .def_property_readonly("diagnostics",[](const Owner& o) { return o.diagnostics(); })
        .def_property_readonly("identity_sha256",&Owner::identity_sha256)
        .def_property_readonly("original_operators_unchanged_certified",&Owner::original_operators_unchanged_certified);
    m.def("_plan_periodic_correlation_real_pao_space",&plan_periodic_correlation_real_pao_space,
        py::arg("reference"),py::arg("domain"),py::arg("retained_dimension"),py::arg("options"));
    m.def("_make_periodic_correlation_real_pao_space",[](
        const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationPAODomain& domain,
        const Options& options, const Caps& caps) {
        if (reference.state().n_kpoints() > 8 || reference.state().n_basis() > 8 || domain.domain_dimension() > 16
            || caps.maximum_owned_numerical_bytes > 1048576 || caps.maximum_work_units > 1000000000)
            throw std::length_error("real PAO diagnostic exceeds tiny shape or work cap");
        const auto settings = options; const auto limits = caps;
        py::gil_scoped_release release;
        return make_periodic_correlation_real_pao_space(reference,domain,settings,limits);
    },py::arg("reference"),py::arg("domain"),py::arg("options"),py::arg("caps"));
}
