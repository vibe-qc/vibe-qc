// Tiny diagnostic boundary. Production consumers call the C++ owner directly.

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

#include "vibeqc/periodic_correlation_wannier.hpp"
#include "vibeqc/periodic_correlation_diabatic_seed.hpp"
#include "vibeqc/periodic_correlation_iao_optimizer.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_wannier(py::module_& m) {
    using Options = vibeqc::PeriodicCorrelationWannierOptions;
    using Memory = vibeqc::PeriodicCorrelationWannierMemoryPlan;
    using Diagnostics = vibeqc::PeriodicCorrelationWannierDiagnostics;
    using Wannier = vibeqc::PeriodicCorrelationWannier;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;
    using Complex = std::complex<double>;

    m.attr("_PERIODIC_CORRELATION_WANNIER_CONTRACT_VERSION") =
        py::int_(vibeqc::kPeriodicCorrelationWannierContractVersion);
    py::class_<Options>(m, "_PeriodicCorrelationWannierOptions")
        .def(py::init<>())
        .def_readwrite("gauge_unitarity_tolerance", &Options::gauge_unitarity_tolerance)
        .def_readwrite("time_reversal_absolute_tolerance", &Options::time_reversal_absolute_tolerance)
        .def_readwrite("time_reversal_relative_tolerance", &Options::time_reversal_relative_tolerance)
        .def_readwrite("real_absolute_tolerance", &Options::real_absolute_tolerance)
        .def_readwrite("real_relative_tolerance", &Options::real_relative_tolerance)
        .def_readwrite("require_time_reversal", &Options::require_time_reversal)
        .def_readwrite("require_real_home_coefficients", &Options::require_real_home_coefficients);
    py::class_<Memory>(m, "_PeriodicCorrelationWannierMemoryPlan")
        .def_readonly("n_cells", &Memory::n_cells)
        .def_readonly("n_basis", &Memory::n_basis)
        .def_readonly("n_home_occupied", &Memory::n_home_occupied)
        .def_readonly("coefficient_count", &Memory::coefficient_count)
        .def_readonly("gauge_element_count", &Memory::gauge_element_count)
        .def_readonly("caller_gauge_bytes", &Memory::caller_gauge_bytes)
        .def_readonly("retained_coefficient_bytes", &Memory::retained_coefficient_bytes)
        .def_readonly("temporary_gauged_coefficient_bytes", &Memory::temporary_gauged_coefficient_bytes)
        .def_readonly("peak_owned_numerical_bytes", &Memory::peak_owned_numerical_bytes);
    py::class_<Diagnostics>(m, "_PeriodicCorrelationWannierDiagnostics")
        .def_readonly("maximum_left_unitarity_residual", &Diagnostics::maximum_left_unitarity_residual)
        .def_readonly("maximum_right_unitarity_residual", &Diagnostics::maximum_right_unitarity_residual)
        .def_readonly("maximum_overlap_time_reversal_residual", &Diagnostics::maximum_overlap_time_reversal_residual)
        .def_readonly("maximum_fock_time_reversal_residual", &Diagnostics::maximum_fock_time_reversal_residual)
        .def_readonly("maximum_coefficient_time_reversal_residual", &Diagnostics::maximum_coefficient_time_reversal_residual)
        .def_readonly("maximum_home_imaginary_magnitude", &Diagnostics::maximum_home_imaginary_magnitude)
        .def_readonly("maximum_home_coefficient_magnitude", &Diagnostics::maximum_home_coefficient_magnitude)
        .def_readonly("time_reversal_compatible", &Diagnostics::time_reversal_compatible)
        .def_readonly("real_home_coefficients_compatible", &Diagnostics::real_home_coefficients_compatible)
        .def_readonly("required_node_memory_bytes", &Diagnostics::required_node_memory_bytes)
        .def_readonly("live_diabatic_seed_index_bytes", &Diagnostics::live_diabatic_seed_index_bytes)
        .def_readonly("live_optimizer_index_bytes", &Diagnostics::live_optimizer_index_bytes);
    py::class_<Wannier>(m, "_PeriodicCorrelationWannier")
        .def_property_readonly("contract_version", &Wannier::contract_version)
        .def_property_readonly("mesh", &Wannier::mesh)
        .def_property_readonly("n_cells", &Wannier::n_cells)
        .def_property_readonly("n_basis", &Wannier::n_basis)
        .def_property_readonly("n_home_occupied", &Wannier::n_home_occupied)
        .def_property_readonly("state", [](const Wannier& value) -> std::shared_ptr<State> {
            return std::const_pointer_cast<State>(value.state_handle());
        })
        .def_property_readonly("state_identity_sha256", [](const Wannier& value) {
            return value.state().state_identity_sha256();
        })
        .def_property_readonly("calculation_identity", [](const Wannier& value) {
            return value.state().calculation_identity();
        })
        .def_property_readonly("allocation_identity", &Wannier::allocation_identity)
        .def_property_readonly("gauge_payload_sha256", &Wannier::gauge_payload_sha256)
        .def_property_readonly("coefficient_payload_sha256", &Wannier::coefficient_payload_sha256)
        .def_property_readonly("wannier_identity_sha256", &Wannier::wannier_identity_sha256)
        .def_property_readonly("localization_identity_sha256", &Wannier::localization_identity_sha256)
        .def_property_readonly("memory", [](const Wannier& value) { return value.memory(); })
        .def_property_readonly("options", [](const Wannier& value) { return value.options(); })
        .def_property_readonly("diagnostics", [](const Wannier& value) { return value.diagnostics(); })
        .def("coefficient", &Wannier::coefficient,
             py::arg("cell"), py::arg("ao"), py::arg("occupied"))
        .def("translated_coefficient", &Wannier::translated_coefficient,
             py::arg("cell"), py::arg("orbital_cell"), py::arg("ao"), py::arg("occupied"))
        .def("cell_coefficients_copy", [](const Wannier& value, std::size_t cell) {
            const auto* source = value.cell_coefficients(cell);
            if (value.n_basis() > 64U || value.n_home_occupied() > 32U) {
                throw std::length_error("Wannier diagnostic cell copy exceeds 64 AOs or 32 occupied orbitals");
            }
            py::array_t<Complex> result({static_cast<py::ssize_t>(value.n_basis()),
                                          static_cast<py::ssize_t>(value.n_home_occupied())});
            std::memcpy(result.mutable_data(), source,
                value.n_basis() * value.n_home_occupied() * sizeof(Complex));
            return result;
        }, py::arg("cell"));
    m.def("_plan_periodic_correlation_wannier", &vibeqc::plan_periodic_correlation_wannier,
          py::arg("mesh"), py::arg("n_basis"), py::arg("n_home_occupied"));
    m.def("_make_periodic_correlation_wannier",
        [](const Reference& reference, const py::array& gauges,
           std::uint64_t owned_numerical_byte_cap, const Options& options) {
            const auto nk = reference.state().n_kpoints();
            const auto nocc = reference.state().n_correlated_occupied();
            if (nk > 64U || reference.state().n_basis() > 64U || nocc > 32U) {
                throw std::length_error("Wannier diagnostic is limited to 64 k points, 64 AOs and 32 occupied orbitals");
            }
            if (!gauges.dtype().is(py::dtype::of<Complex>())
                || gauges.ndim() != 3 || !(gauges.flags() & py::array::c_style)
                || gauges.shape(0) != static_cast<py::ssize_t>(nk)
                || gauges.shape(1) != static_cast<py::ssize_t>(nocc)
                || gauges.shape(2) != static_cast<py::ssize_t>(nocc)) {
                throw std::invalid_argument("Wannier gauges require existing C-contiguous complex128 [Nk,nactive,nactive] storage");
            }
            const auto* data = static_cast<const Complex*>(gauges.data());
            const auto count = static_cast<std::size_t>(gauges.size());
            py::gil_scoped_release release;
            return vibeqc::make_periodic_correlation_wannier(
                reference, data, count, owned_numerical_byte_cap, options);
        }, py::arg("reference"), py::arg("gauges").noconvert(),
        py::arg("owned_numerical_byte_cap"), py::arg("options"),
        "Bounded native Fourier/gauge diagnostic, not a localization optimizer. "
        "The non-owning input gauge must remain immutable during this call. "
        "No full placed-orbital matrix or all-cell copy is exposed.");
    m.def("_make_periodic_correlation_wannier_from_diabatic_seed",
        [](const Reference& reference, const vibeqc::PeriodicCorrelationDiabaticSeed& seed,
           std::uint64_t owned_numerical_byte_cap, const Options& options) {
            if (reference.state().n_kpoints() > 64U || reference.state().n_basis() > 64U
                || reference.state().n_correlated_occupied() > 32U) {
                throw std::length_error("Wannier diagnostic is limited to 64 k points, 64 AOs and 32 occupied orbitals");
            }
            py::gil_scoped_release release;
            return vibeqc::make_periodic_correlation_wannier_from_diabatic_seed(
                reference, seed, owned_numerical_byte_cap, options);
        }, py::arg("reference"), py::arg("seed"), py::arg("owned_numerical_byte_cap"),
        py::arg("options"), "Borrow native seed gauges without copying; admit all live seed storage.");
    m.def("_make_periodic_correlation_wannier_from_iao_optimizer",
        [](const Reference& reference, const vibeqc::PeriodicCorrelationIAOOptimizerResult& optimizer,
           std::uint64_t cap, const Options& options) {
            if (reference.state().n_kpoints() > 64U || reference.state().n_basis() > 64U
                || reference.state().n_correlated_occupied() > 32U)
                throw std::length_error("Wannier diagnostic is limited to 64 k points, 64 AOs and 32 occupied orbitals");
            const auto controls = options;
            py::gil_scoped_release release;
            return vibeqc::make_periodic_correlation_wannier_from_iao_optimizer(reference, optimizer, cap, controls);
        }, py::arg("reference"), py::arg("optimizer"), py::arg("owned_numerical_byte_cap"), py::arg("options"),
        "Borrow converged native optimizer gauges, preserve its receipt and admit all retained optimizer bytes.");
}
