// Tiny diagnostic binding; production callers consume the C++ block owner.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "vibeqc/periodic_correlation_occupied_fock.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_occupied_fock(py::module_& m) {
    using Options = vibeqc::PeriodicCorrelationOccupiedFockOptions;
    using Memory = vibeqc::PeriodicCorrelationOccupiedFockMemoryPlan;
    using Diagnostics = vibeqc::PeriodicCorrelationOccupiedFockDiagnostics;
    using Fock = vibeqc::PeriodicCorrelationOccupiedFock;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;
    using Wannier = vibeqc::PeriodicCorrelationWannier;
    using Complex = std::complex<double>;

    m.attr("_PERIODIC_CORRELATION_OCCUPIED_FOCK_CONTRACT_VERSION") =
        py::int_(vibeqc::kPeriodicCorrelationOccupiedFockContractVersion);
    py::class_<Options>(m, "_PeriodicCorrelationOccupiedFockOptions")
        .def(py::init<>())
        .def_readwrite("hermiticity_absolute_tolerance", &Options::hermiticity_absolute_tolerance)
        .def_readwrite("hermiticity_relative_tolerance", &Options::hermiticity_relative_tolerance)
        .def_readwrite("time_reversal_absolute_tolerance", &Options::time_reversal_absolute_tolerance)
        .def_readwrite("time_reversal_relative_tolerance", &Options::time_reversal_relative_tolerance)
        .def_readwrite("real_absolute_tolerance", &Options::real_absolute_tolerance)
        .def_readwrite("real_relative_tolerance", &Options::real_relative_tolerance)
        .def_readwrite("require_time_reversal", &Options::require_time_reversal)
        .def_readwrite("require_real_blocks", &Options::require_real_blocks);
    py::class_<Memory>(m, "_PeriodicCorrelationOccupiedFockMemoryPlan")
        .def_readonly("n_cells", &Memory::n_cells)
        .def_readonly("n_basis", &Memory::n_basis)
        .def_readonly("n_home_occupied", &Memory::n_home_occupied)
        .def_readonly("element_count", &Memory::element_count)
        .def_readonly("retained_block_bytes", &Memory::retained_block_bytes)
        .def_readonly("projection_workspace_bytes", &Memory::projection_workspace_bytes)
        .def_readonly("fourier_workspace_bytes", &Memory::fourier_workspace_bytes)
        .def_readonly("peak_owned_numerical_bytes", &Memory::peak_owned_numerical_bytes)
        .def_readonly("caller_gauge_bytes", &Memory::caller_gauge_bytes)
        .def_readonly("live_wannier_bytes", &Memory::live_wannier_bytes);
    py::class_<Diagnostics>(m, "_PeriodicCorrelationOccupiedFockDiagnostics")
        .def_readonly("maximum_projected_hermiticity_residual", &Diagnostics::maximum_projected_hermiticity_residual)
        .def_readonly("maximum_translation_hermiticity_residual", &Diagnostics::maximum_translation_hermiticity_residual)
        .def_readonly("maximum_projected_time_reversal_residual", &Diagnostics::maximum_projected_time_reversal_residual)
        .def_readonly("maximum_block_imaginary_magnitude", &Diagnostics::maximum_block_imaginary_magnitude)
        .def_readonly("maximum_canonical_projection_discrepancy", &Diagnostics::maximum_canonical_projection_discrepancy)
        .def_readonly("canonical_projection_discrepancy_frobenius", &Diagnostics::canonical_projection_discrepancy_frobenius)
        .def_readonly("time_reversal_compatible", &Diagnostics::time_reversal_compatible)
        .def_readonly("real_blocks_compatible", &Diagnostics::real_blocks_compatible)
        .def_readonly("required_node_memory_bytes", &Diagnostics::required_node_memory_bytes);
    py::class_<Fock>(m, "_PeriodicCorrelationOccupiedFock")
        .def_property_readonly("contract_version", &Fock::contract_version)
        .def_property_readonly("mesh", &Fock::mesh)
        .def_property_readonly("n_cells", &Fock::n_cells)
        .def_property_readonly("n_home_occupied", &Fock::n_home_occupied)
        .def_property_readonly("state", [](const Fock& value) -> std::shared_ptr<State> {
            return std::const_pointer_cast<State>(value.state_handle());
        })
        .def_property_readonly("allocation_identity", &Fock::allocation_identity)
        .def_property_readonly("wannier_identity_sha256", &Fock::wannier_identity_sha256)
        .def_property_readonly("gauge_payload_sha256", &Fock::gauge_payload_sha256)
        .def_property_readonly("block_payload_sha256", &Fock::block_payload_sha256)
        .def_property_readonly("occupied_fock_identity_sha256", &Fock::occupied_fock_identity_sha256)
        .def_property_readonly("memory", [](const Fock& value) { return value.memory(); })
        .def_property_readonly("options", [](const Fock& value) { return value.options(); })
        .def_property_readonly("diagnostics", [](const Fock& value) { return value.diagnostics(); })
        .def("element", &Fock::element, py::arg("translation"), py::arg("i"), py::arg("j"))
        .def("placed_element", &Fock::placed_element,
             py::arg("bra_cell"), py::arg("ket_cell"), py::arg("i"), py::arg("j"))
        .def("block_copy", [](const Fock& value, std::size_t translation) {
            const auto* source = value.block(translation);
            const auto n = value.n_home_occupied();
            if (n > 32U) throw std::length_error("occupied Fock diagnostic block copy is limited to rank 32");
            py::array_t<Complex> result({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            std::memcpy(result.mutable_data(), source, n * n * sizeof(Complex));
            return result;
        }, py::arg("translation"));
    m.def("_plan_periodic_correlation_occupied_fock", &vibeqc::plan_periodic_correlation_occupied_fock,
          py::arg("mesh"), py::arg("n_basis"), py::arg("n_home_occupied"));
    m.def("_make_periodic_correlation_occupied_fock",
        [](const Reference& reference, const Wannier& wannier, const py::array& gauges,
           std::uint64_t owned_numerical_byte_cap, const Options& options) {
            const auto nk = reference.state().n_kpoints();
            const auto nocc = reference.state().n_correlated_occupied();
            if (nk > 64U || reference.state().n_basis() > 64U || nocc > 32U) {
                throw std::length_error("occupied Fock diagnostic is limited to 64 k points, 64 AOs and rank 32");
            }
            if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim() != 3
                || !(gauges.flags() & py::array::c_style)
                || gauges.shape(0) != static_cast<py::ssize_t>(nk)
                || gauges.shape(1) != static_cast<py::ssize_t>(nocc)
                || gauges.shape(2) != static_cast<py::ssize_t>(nocc)) {
                throw std::invalid_argument("occupied Fock gauges require existing C-contiguous complex128 [Nk,nactive,nactive] storage");
            }
            const auto* data = static_cast<const Complex*>(gauges.data());
            const auto count = static_cast<std::size_t>(gauges.size());
            const Options controls = options;
            py::gil_scoped_release release;
            return vibeqc::make_periodic_correlation_occupied_fock(
                reference, wannier, data, count, owned_numerical_byte_cap, controls);
        }, py::arg("reference"), py::arg("wannier"), py::arg("gauges").noconvert(),
        py::arg("owned_numerical_byte_cap"), py::arg("options"),
        "Direct physical occupied Fock and in-place finite-torus Fourier diagnostic. "
        "The gauge input must remain immutable for this call. "
        "No eigensolver, canonical-eigenvalue substitution or full placed matrix is used.");
}
