// Tiny one-k algebra diagnostic; input provenance remains caller-declared.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>

#include "vibeqc/periodic_correlation_bloch_iao.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_bloch_iao(py::module_& m) {
    using Options = vibeqc::PeriodicCorrelationBlochIAOOptions;
    using Provenance = vibeqc::PeriodicCorrelationBlochIAOInputProvenance;
    using Memory = vibeqc::PeriodicCorrelationBlochIAOMemoryPlan;
    using Diagnostics = vibeqc::PeriodicCorrelationBlochIAODiagnostics;
    using Result = vibeqc::PeriodicCorrelationBlochIAO;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Z = std::complex<double>;
    m.attr("_PERIODIC_CORRELATION_BLOCH_IAO_CONTRACT_VERSION") = py::int_(vibeqc::kPeriodicCorrelationBlochIAOContractVersion);
    auto o = py::class_<Options>(m, "_PeriodicCorrelationBlochIAOOptions");
    o.def(py::init<>());
#define BLOCH_IAO_OPTION(field) o.def_readwrite(#field, &Options::field)
    BLOCH_IAO_OPTION(minimal_rank_absolute_floor); BLOCH_IAO_OPTION(minimal_rank_relative_floor);
    BLOCH_IAO_OPTION(depolarized_rank_absolute_floor); BLOCH_IAO_OPTION(depolarized_rank_relative_floor);
    BLOCH_IAO_OPTION(iao_rank_absolute_floor); BLOCH_IAO_OPTION(iao_rank_relative_floor);
    BLOCH_IAO_OPTION(schur_negative_absolute_tolerance); BLOCH_IAO_OPTION(schur_negative_relative_tolerance);
    BLOCH_IAO_OPTION(validation_absolute_tolerance); BLOCH_IAO_OPTION(validation_relative_tolerance);
    BLOCH_IAO_OPTION(jacobi_max_sweeps); BLOCH_IAO_OPTION(jacobi_relative_tolerance);
    BLOCH_IAO_OPTION(maximum_work_units);
#undef BLOCH_IAO_OPTION
    py::class_<Provenance>(m, "_PeriodicCorrelationBlochIAOInputProvenance")
        .def(py::init<>())
        .def_readwrite("cross_overlap_identity", &Provenance::cross_overlap_identity)
        .def_readwrite("minimal_basis_identity", &Provenance::minimal_basis_identity)
        .def_readwrite("caller_live_numerical_bytes", &Provenance::caller_live_numerical_bytes);
    auto memory = py::class_<Memory>(m, "_PeriodicCorrelationBlochIAOMemoryPlan");
#define BLOCH_IAO_MEMORY(field) memory.def_readonly(#field, &Memory::field)
    BLOCH_IAO_MEMORY(n_basis); BLOCH_IAO_MEMORY(n_effective); BLOCH_IAO_MEMORY(n_occupied); BLOCH_IAO_MEMORY(n_minimal);
    BLOCH_IAO_MEMORY(borrowed_input_bytes); BLOCH_IAO_MEMORY(output_numerical_bytes);
    BLOCH_IAO_MEMORY(projection_workspace_bytes); BLOCH_IAO_MEMORY(matrix_workspace_bytes);
    BLOCH_IAO_MEMORY(scalar_workspace_bytes); BLOCH_IAO_MEMORY(peak_owned_numerical_bytes);
    BLOCH_IAO_MEMORY(input_preflight_work_units); BLOCH_IAO_MEMORY(projection_work_units);
    BLOCH_IAO_MEMORY(construction_work_units); BLOCH_IAO_MEMORY(validation_work_units);
    BLOCH_IAO_MEMORY(maximum_work_units);
#undef BLOCH_IAO_MEMORY
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicCorrelationBlochIAODiagnostics");
#define BLOCH_IAO_DIAGNOSTIC(field) diagnostics.def_readonly(#field, &Diagnostics::field)
    BLOCH_IAO_DIAGNOSTIC(required_node_memory_bytes); BLOCH_IAO_DIAGNOSTIC(charged_work_units);
    BLOCH_IAO_DIAGNOSTIC(jacobi_sweeps); BLOCH_IAO_DIAGNOSTIC(maximum_retained_metric_residual);
    BLOCH_IAO_DIAGNOSTIC(maximum_eigensystem_relative_residual);
    BLOCH_IAO_DIAGNOSTIC(maximum_derived_hermitian_defect); BLOCH_IAO_DIAGNOSTIC(maximum_derived_hermitization_correction);
    BLOCH_IAO_DIAGNOSTIC(minimum_schur_eigenvalue); BLOCH_IAO_DIAGNOSTIC(schur_negative_boundary);
    BLOCH_IAO_DIAGNOSTIC(minimum_minimal_eigenvalue); BLOCH_IAO_DIAGNOSTIC(minimal_rank_boundary);
    BLOCH_IAO_DIAGNOSTIC(minimum_depolarized_gram_eigenvalue); BLOCH_IAO_DIAGNOSTIC(depolarized_rank_boundary);
    BLOCH_IAO_DIAGNOSTIC(minimum_iao_gram_eigenvalue); BLOCH_IAO_DIAGNOSTIC(iao_rank_boundary);
    BLOCH_IAO_DIAGNOSTIC(maximum_depolarized_metric_residual); BLOCH_IAO_DIAGNOSTIC(maximum_depolarized_span_residual);
    BLOCH_IAO_DIAGNOSTIC(maximum_ao_iao_metric_residual); BLOCH_IAO_DIAGNOSTIC(maximum_covariant_residual);
    BLOCH_IAO_DIAGNOSTIC(maximum_occupied_metric_residual); BLOCH_IAO_DIAGNOSTIC(maximum_retained_reconstruction_residual);
    BLOCH_IAO_DIAGNOSTIC(maximum_ao_reconstruction_residual); BLOCH_IAO_DIAGNOSTIC(maximum_occupied_projector_residual);
#undef BLOCH_IAO_DIAGNOSTIC
    const auto copy_matrix = [](const Z* source, std::uint64_t rows, std::uint64_t columns) {
        if (rows > 64U || columns > 16U) throw std::length_error("Bloch IAO diagnostic copy is limited to 64 rows and 16 columns");
        py::array_t<Z> answer({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
        std::memcpy(answer.mutable_data(), source, rows * columns * sizeof(Z)); return answer;
    };
    py::class_<Result>(m, "_PeriodicCorrelationBlochIAO")
        .def_property_readonly("contract_version", &Result::contract_version)
        .def_property_readonly("point", &Result::point)
        .def_property_readonly("state", [](const Result& value) -> std::shared_ptr<State> {
            (void) value.state(); return std::const_pointer_cast<State>(value.state_handle());
        })
        .def_property_readonly("memory", [](const Result& value) { return value.memory(); })
        .def_property_readonly("options", [](const Result& value) { return value.options(); })
        .def_property_readonly("diagnostics", [](const Result& value) { return value.diagnostics(); })
        .def_property_readonly("declared_provenance", [](const Result& value) { return value.declared_provenance(); })
        .def_property_readonly("input_payload_sha256", &Result::input_payload_sha256)
        .def_property_readonly("output_payload_sha256", &Result::output_payload_sha256)
        .def_property_readonly("iao_identity_sha256", &Result::iao_identity_sha256)
        .def_property_readonly("allocation_identity", &Result::allocation_identity)
        .def("atom_label", &Result::atom_label, py::arg("minimal"))
        .def("occupied_band", &Result::occupied_band, py::arg("occupied"))
        .def("ao_coefficient", &Result::ao_coefficient, py::arg("ao"), py::arg("minimal"))
        .def("retained_coefficients_copy", [copy_matrix](const Result& v) { return copy_matrix(v.retained_coefficients_data(), v.memory().n_effective, v.memory().n_minimal); })
        .def("metric_copy", [copy_matrix](const Result& v) { return copy_matrix(v.metric_data(), v.memory().n_minimal, v.memory().n_minimal); })
        .def("occupied_coefficients_copy", [copy_matrix](const Result& v) { return copy_matrix(v.occupied_coefficients_data(), v.memory().n_minimal, v.memory().n_occupied); })
        .def("occupied_covariant_copy", [copy_matrix](const Result& v) { return copy_matrix(v.occupied_covariant_data(), v.memory().n_minimal, v.memory().n_occupied); });
    m.def("_plan_periodic_correlation_bloch_iao", &vibeqc::plan_periodic_correlation_bloch_iao,
        py::arg("n_basis"), py::arg("n_effective"), py::arg("n_occupied"), py::arg("n_minimal"), py::arg("jacobi_max_sweeps"));
    m.def("_make_periodic_correlation_bloch_iao",
        [](const Reference& reference, std::size_t point, const py::array& cross, const py::array& minimal,
           const py::array& atoms, const Provenance& provenance, std::uint64_t cap, const Options& options) {
            const auto& s = reference.state();
            if (s.n_basis() > 64U || s.n_effective_orbitals() > 64U
                || s.n_frozen_core() + s.n_correlated_occupied() > 16U || options.jacobi_max_sweeps > 256U) {
                throw std::length_error("Bloch IAO diagnostic is limited to 64 AOs, 16 occupied/minimal labels and 256 sweeps");
            }
            if (!cross.dtype().is(py::dtype::of<Z>()) || cross.ndim() != 2 || !(cross.flags() & py::array::c_style)
                || cross.shape(0) != static_cast<py::ssize_t>(s.n_basis())
                || !minimal.dtype().is(py::dtype::of<Z>()) || minimal.ndim() != 2 || !(minimal.flags() & py::array::c_style)
                || minimal.shape(0) != cross.shape(1) || minimal.shape(1) != cross.shape(1)
                || !atoms.dtype().is(py::dtype::of<std::uint64_t>()) || atoms.ndim() != 1
                || !(atoms.flags() & py::array::c_style) || atoms.shape(0) != cross.shape(1)) {
                throw std::invalid_argument("Bloch IAO needs existing C-contiguous complex128 S12[B,r]/S22[r,r] and uint64 atom labels[r]");
            }
            if (cross.shape(1) > 16) throw std::length_error("Bloch IAO diagnostic minimal dimension is limited to 16");
            const auto r = static_cast<std::size_t>(cross.shape(1));
            const auto* s12 = static_cast<const Z*>(cross.data()); const auto* s22 = static_cast<const Z*>(minimal.data());
            const auto* map = static_cast<const std::uint64_t*>(atoms.data());
            // This capped diagnostic deliberately keeps the GIL: Options and
            // provenance are mutable Python wrappers and must not change while
            // native gates/hash construction are in flight. Native producers
            // have the ordinary immutable-borrowed-input call contract.
            return vibeqc::make_periodic_correlation_bloch_iao(reference, point, s12, cross.size(), s22, minimal.size(),
                map, atoms.size(), r, provenance, cap, options);
        }, py::arg("reference"), py::arg("point"), py::arg("cross_overlap").noconvert(), py::arg("minimal_overlap").noconvert(),
        py::arg("atom_labels").noconvert(), py::arg("declared_provenance"), py::arg("owned_numerical_byte_cap"), py::arg("options"),
        "One-k unorthogonalized Bloch-IAO algebra from caller-declared overlaps. No physical-source certification or localization optimizer.");
}
