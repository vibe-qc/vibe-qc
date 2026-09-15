// Tiny physical-overlap source diagnostics; included by bindings.cpp.
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_gaussian_bloch_iao.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_gaussian_bloch_iao(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicGaussianBlochIAOOptions;
    using Caps = PeriodicGaussianBlochIAOCaps;
    using Plan = PeriodicGaussianBlochIAOMemoryPlan;
    using Diagnostics = PeriodicGaussianBlochIAODiagnostics;
    using Point = PeriodicGaussianBlochIAOPoint;
    auto options = py::class_<Options>(m, "_PeriodicGaussianBlochIAOOptions"); options.def(py::init<>());
#define GBIAO_OPTION(field) options.def_readwrite(#field, &Options::field)
    GBIAO_OPTION(image_cutoff_bohr); GBIAO_OPTION(geometry_absolute_tolerance); GBIAO_OPTION(geometry_relative_tolerance);
    GBIAO_OPTION(overlap_absolute_tolerance); GBIAO_OPTION(overlap_relative_tolerance);
    GBIAO_OPTION(structural_absolute_tolerance); GBIAO_OPTION(structural_relative_tolerance);
    GBIAO_OPTION(projection_absolute_tolerance); GBIAO_OPTION(projection_relative_tolerance);
#undef GBIAO_OPTION
    auto caps = py::class_<Caps>(m, "_PeriodicGaussianBlochIAOCaps"); caps.def(py::init<>());
#define GBIAO_CAP(field) caps.def_readwrite(#field, &Caps::field)
    GBIAO_CAP(maximum_owned_numerical_bytes); GBIAO_CAP(maximum_work_units); GBIAO_CAP(maximum_atom_count);
    GBIAO_CAP(maximum_shell_count); GBIAO_CAP(maximum_contraction_count); GBIAO_CAP(maximum_primitive_numeric_lanes);
    GBIAO_CAP(maximum_basis_content_wire_bytes); GBIAO_CAP(maximum_borrowed_active_numeric_bytes);
    GBIAO_CAP(maximum_panel_pairs); GBIAO_CAP(maximum_total_image_candidates);
#undef GBIAO_CAP
    auto plan = py::class_<Plan>(m, "_PeriodicGaussianBlochIAOMemoryPlan");
#define GBIAO_PLAN(field) plan.def_readonly(#field, &Plan::field)
    GBIAO_PLAN(n_basis); GBIAO_PLAN(n_effective); GBIAO_PLAN(n_occupied); GBIAO_PLAN(n_minimal);
    GBIAO_PLAN(point); GBIAO_PLAN(conjugate_point); GBIAO_PLAN(panel_pairs); GBIAO_PLAN(panel_calls);
    GBIAO_PLAN(ao); GBIAO_PLAN(minimal); GBIAO_PLAN(borrowed_basis_numeric_bytes); GBIAO_PLAN(borrowed_geometry_numeric_bytes);
    GBIAO_PLAN(other_live_numerical_bytes); GBIAO_PLAN(total_borrowed_numerical_bytes); GBIAO_PLAN(overlap_and_label_bytes);
    GBIAO_PLAN(panel_output_bytes); GBIAO_PLAN(fixed_fourier_workspace_bytes); GBIAO_PLAN(iao_owned_peak_bytes);
    GBIAO_PLAN(peak_owned_numerical_bytes); GBIAO_PLAN(output_numerical_bytes); GBIAO_PLAN(required_node_memory_bytes);
    GBIAO_PLAN(image_candidate_count); GBIAO_PLAN(retained_pair_image_count); GBIAO_PLAN(basis_scan_work_units);
    GBIAO_PLAN(preflight_work_units); GBIAO_PLAN(overlap_evaluation_work_units); GBIAO_PLAN(iao_work_units);
    GBIAO_PLAN(maximum_work_units);
#undef GBIAO_PLAN
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicGaussianBlochIAODiagnostics");
#define GBIAO_DIAGNOSTIC(field) diagnostics.def_readonly(#field, &Diagnostics::field)
    GBIAO_DIAGNOSTIC(charged_work_units); GBIAO_DIAGNOSTIC(evaluated_panel_count);
    GBIAO_DIAGNOSTIC(evaluated_image_candidate_count); GBIAO_DIAGNOSTIC(evaluated_pair_image_count);
    GBIAO_DIAGNOSTIC(maximum_geometry_residual); GBIAO_DIAGNOSTIC(maximum_s11_reference_residual);
    GBIAO_DIAGNOSTIC(maximum_s11_reference_conjugacy_residual); GBIAO_DIAGNOSTIC(maximum_cross_adjoint_residual);
    GBIAO_DIAGNOSTIC(maximum_overlap_time_reversal_residual); GBIAO_DIAGNOSTIC(maximum_minimal_hermitian_defect);
    GBIAO_DIAGNOSTIC(maximum_trim_imaginary_magnitude); GBIAO_DIAGNOSTIC(maximum_projection_correction);
#undef GBIAO_DIAGNOSTIC
    py::class_<Point>(m, "_PeriodicGaussianBlochIAOPoint")
        .def_property_readonly("iao", &Point::iao, py::return_value_policy::reference_internal)
        .def_property_readonly("contract_version", &Point::contract_version)
        .def_property_readonly("hf_basis_source_authenticated", &Point::hf_basis_source_authenticated)
        .def_property_readonly("infinite_image_tail_certified", &Point::infinite_image_tail_certified)
        .def_property_readonly("memory", [](const Point& x) { return x.memory(); })
        .def_property_readonly("diagnostics", [](const Point& x) { return x.diagnostics(); })
        .def_property_readonly("options", [](const Point& x) { return x.options(); })
        .def_property_readonly("ao_basis_identity_sha256", &Point::ao_basis_identity_sha256)
        .def_property_readonly("minimal_basis_identity_sha256", &Point::minimal_basis_identity_sha256)
        .def_property_readonly("source_identity_sha256", &Point::source_identity_sha256)
        .def_property_readonly("raw_overlap_payload_sha256", &Point::raw_overlap_payload_sha256)
        .def_property_readonly("reference_match_identity_sha256", &Point::reference_match_identity_sha256);
    const auto tiny = [](const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao,
                         const BasisSet& minimal, const PeriodicSystem& system, std::uint64_t other,
                         const PeriodicCorrelationBlochIAOOptions& io, const Caps& c) {
        if (!reference.state_handle()) throw std::invalid_argument("Gaussian Bloch IAO diagnostic requires a live reference");
        if (reference.state().n_kpoints() > 8 || ao.nbasis() > 12 || minimal.nbasis() > 12
            || system.unit_cell.size() > 8 || ao.nshells() > 16 || minimal.nshells() > 16
            || other > (8U << 20) || io.jacobi_max_sweeps > 64
            || c.maximum_owned_numerical_bytes > (8U << 20) || c.maximum_work_units > 20000000000000ULL
            || c.maximum_atom_count > 8 || c.maximum_shell_count > 32 || c.maximum_contraction_count > 64
            || c.maximum_primitive_numeric_lanes > 1024 || c.maximum_basis_content_wire_bytes > 65536
            || c.maximum_borrowed_active_numeric_bytes > (8U << 20) || c.maximum_panel_pairs > 64
            || c.maximum_total_image_candidates > 250000)
            throw std::length_error("Gaussian Bloch IAO diagnostic exceeds tiny shape/source/work caps");
    };
    m.def("_plan_periodic_gaussian_bloch_iao", [tiny](
        const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao, const BasisSet& minimal,
        const PeriodicSystem& system, std::size_t point, std::uint64_t other,
        const Options& o, const PeriodicCorrelationBlochIAOOptions& io, const Caps& c) {
        tiny(reference, ao, minimal, system, other, io, c);
        return plan_periodic_gaussian_bloch_iao(reference, ao, minimal, system, point, other, o, io, c);
    }, py::arg("reference"), py::arg("ao_basis"), py::arg("minimal_basis"), py::arg("system"),
       py::arg("point"), py::arg("other_live_numerical_bytes"), py::arg("options"),
       py::arg("iao_options"), py::arg("caps"));
    m.def("_make_periodic_gaussian_bloch_iao", [tiny](
        const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao, const BasisSet& minimal,
        const PeriodicSystem& system, std::size_t point, std::uint64_t other,
        const Options& o, const PeriodicCorrelationBlochIAOOptions& io, const Caps& c) {
        tiny(reference, ao, minimal, system, other, io, c);
        // No GIL release or callbacks: the borrowed basis/cell/control owners
        // stay unchanged and alive throughout this bounded synchronous call.
        return make_periodic_gaussian_bloch_iao(reference, ao, minimal, system, point, other, o, io, c);
    }, py::arg("reference"), py::arg("ao_basis"), py::arg("minimal_basis"), py::arg("system"),
       py::arg("point"), py::arg("other_live_numerical_bytes"), py::arg("options"),
       py::arg("iao_options"), py::arg("caps"));
}
