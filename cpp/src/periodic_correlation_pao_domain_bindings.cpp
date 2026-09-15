// Tiny diagnostic boundary; production consumers retain the native owner.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "vibeqc/periodic_correlation_pao_domain.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_pao_domain(py::module_& m) {
    using Options = vibeqc::PeriodicCorrelationPAODomainOptions;
    using Memory = vibeqc::PeriodicCorrelationPAODomainMemoryPlan;
    using Diagnostics = vibeqc::PeriodicCorrelationPAODomainDiagnostics;
    using Domain = vibeqc::PeriodicCorrelationPAODomain;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;
    using Complex = std::complex<double>;

    m.attr("_PERIODIC_CORRELATION_PAO_DOMAIN_CONTRACT_VERSION") =
        py::int_(vibeqc::kPeriodicCorrelationPAODomainContractVersion);
    py::class_<Options>(m, "_PeriodicCorrelationPAODomainOptions")
        .def(py::init<>())
        .def_readwrite("hermitian_absolute_tolerance", &Options::hermitian_absolute_tolerance)
        .def_readwrite("hermitian_relative_tolerance", &Options::hermitian_relative_tolerance)
        .def_readwrite("time_reversal_absolute_tolerance", &Options::time_reversal_absolute_tolerance)
        .def_readwrite("time_reversal_relative_tolerance", &Options::time_reversal_relative_tolerance)
        .def_readwrite("real_absolute_tolerance", &Options::real_absolute_tolerance)
        .def_readwrite("real_relative_tolerance", &Options::real_relative_tolerance)
        .def_readwrite("require_time_reversal", &Options::require_time_reversal)
        .def_readwrite("require_real_matrices", &Options::require_real_matrices);
    py::class_<Memory>(m, "_PeriodicCorrelationPAODomainMemoryPlan")
        .def_readonly("n_cells", &Memory::n_cells)
        .def_readonly("n_basis", &Memory::n_basis)
        .def_readonly("domain_dimension", &Memory::domain_dimension)
        .def_readonly("maximum_unique_domain_columns", &Memory::maximum_unique_domain_columns)
        .def_readonly("matrix_element_count", &Memory::matrix_element_count)
        .def_readonly("caller_domain_index_bytes", &Memory::caller_domain_index_bytes)
        .def_readonly("retained_domain_index_bytes", &Memory::retained_domain_index_bytes)
        .def_readonly("retained_matrix_bytes", &Memory::retained_matrix_bytes)
        .def_readonly("temporary_column_bytes", &Memory::temporary_column_bytes)
        .def_readonly("peak_owned_numerical_bytes", &Memory::peak_owned_numerical_bytes);
    py::class_<Diagnostics>(m, "_PeriodicCorrelationPAODomainDiagnostics")
        .def_readonly("maximum_input_overlap_hermitian_defect", &Diagnostics::maximum_input_overlap_hermitian_defect)
        .def_readonly("maximum_input_fock_hermitian_defect", &Diagnostics::maximum_input_fock_hermitian_defect)
        .def_readonly("maximum_overlap_time_reversal_residual", &Diagnostics::maximum_overlap_time_reversal_residual)
        .def_readonly("maximum_fock_time_reversal_residual", &Diagnostics::maximum_fock_time_reversal_residual)
        .def_readonly("maximum_selected_projector_time_reversal_residual", &Diagnostics::maximum_selected_projector_time_reversal_residual)
        .def_readonly("maximum_raw_overlap_hermitian_defect", &Diagnostics::maximum_raw_overlap_hermitian_defect)
        .def_readonly("maximum_raw_fock_hermitian_defect", &Diagnostics::maximum_raw_fock_hermitian_defect)
        .def_readonly("maximum_overlap_hermitization_correction", &Diagnostics::maximum_overlap_hermitization_correction)
        .def_readonly("maximum_fock_hermitization_correction", &Diagnostics::maximum_fock_hermitization_correction)
        .def_readonly("maximum_overlap_imaginary_magnitude", &Diagnostics::maximum_overlap_imaginary_magnitude)
        .def_readonly("maximum_fock_imaginary_magnitude", &Diagnostics::maximum_fock_imaginary_magnitude)
        .def_readonly("time_reversal_compatible", &Diagnostics::time_reversal_compatible)
        .def_readonly("real_matrices_compatible", &Diagnostics::real_matrices_compatible)
        .def_readonly("required_node_memory_bytes", &Diagnostics::required_node_memory_bytes);
    py::class_<Domain>(m, "_PeriodicCorrelationPAODomain")
        .def_property_readonly("contract_version", &Domain::contract_version)
        .def_property_readonly("domain_dimension", &Domain::domain_dimension)
        .def_property_readonly("state", [](const Domain& value) -> std::shared_ptr<State> {
            return std::const_pointer_cast<State>(value.state_handle());
        })
        .def_property_readonly("state_identity_sha256", [](const Domain& value) {
            return value.state().state_identity_sha256();
        })
        .def_property_readonly("calculation_identity", [](const Domain& value) {
            return value.state().calculation_identity();
        })
        .def_property_readonly("allocation_identity", &Domain::allocation_identity)
        .def_property_readonly("domain_index_sha256", &Domain::domain_index_sha256)
        .def_property_readonly("matrix_payload_sha256", &Domain::matrix_payload_sha256)
        .def_property_readonly("pao_domain_identity_sha256", &Domain::pao_domain_identity_sha256)
        .def_property_readonly("memory", [](const Domain& value) { return value.memory(); })
        .def_property_readonly("options", [](const Domain& value) { return value.options(); })
        .def_property_readonly("diagnostics", [](const Domain& value) { return value.diagnostics(); })
        .def("column", [](const Domain& value, std::size_t index) {
            const auto column = value.column(index);
            return py::make_tuple(column.cell, column.ao);
        }, py::arg("index"))
        .def("overlap", &Domain::overlap, py::arg("row"), py::arg("col"))
        .def("fock", &Domain::fock, py::arg("row"), py::arg("col"))
        .def("overlap_copy", [](const Domain& value) {
            const auto d = value.domain_dimension();
            if (d > 32U) throw std::length_error("PAO domain diagnostic copy is limited to 32 columns");
            py::array_t<Complex> answer({static_cast<py::ssize_t>(d), static_cast<py::ssize_t>(d)});
            if (d != 0U) std::memcpy(answer.mutable_data(), value.overlap_data(), d * d * sizeof(Complex));
            return answer;
        })
        .def("fock_copy", [](const Domain& value) {
            const auto d = value.domain_dimension();
            if (d > 32U) throw std::length_error("PAO domain diagnostic copy is limited to 32 columns");
            py::array_t<Complex> answer({static_cast<py::ssize_t>(d), static_cast<py::ssize_t>(d)});
            if (d != 0U) std::memcpy(answer.mutable_data(), value.fock_data(), d * d * sizeof(Complex));
            return answer;
        });
    m.def("_plan_periodic_correlation_pao_domain", &vibeqc::plan_periodic_correlation_pao_domain,
          py::arg("mesh"), py::arg("n_basis"), py::arg("domain_dimension"));
    m.def("_make_periodic_correlation_pao_domain",
        [](const Reference& reference, const py::array& columns,
           std::uint64_t owned_numerical_byte_cap, const Options& options) {
            if (!reference.state_handle()) throw std::invalid_argument("PAO domain requires a live reference");
            if (reference.state().n_kpoints() > 32U || reference.state().n_basis() > 32U) {
                throw std::length_error("PAO domain diagnostic is limited to 32 k points and 32 AOs");
            }
            if (!columns.dtype().is(py::dtype::of<std::uint64_t>()) || columns.ndim() != 2
                || !(columns.flags() & py::array::c_style) || columns.shape(1) != 2) {
                throw std::invalid_argument("PAO domain requires existing C-contiguous uint64 [D,2] indices");
            }
            if (columns.shape(0) > 32) {
                throw std::length_error("PAO domain diagnostic is limited to 32 columns");
            }
            const auto* data = static_cast<const std::uint64_t*>(columns.data());
            const auto count = static_cast<std::size_t>(columns.size());
            const auto d = static_cast<std::size_t>(columns.shape(0));
            py::gil_scoped_release release;
            return vibeqc::make_periodic_correlation_pao_domain(
                reference, data, count, d, owned_numerical_byte_cap, options);
        }, py::arg("reference"), py::arg("columns").noconvert(),
        py::arg("owned_numerical_byte_cap"), py::arg("options"),
        "Bounded selected PAO S/F geometry, not a correlation calculation. "
        "The borrowed uint64 index view must remain immutable during the call.");
}
