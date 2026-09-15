// Tiny diagnostic only; no production-sized orbital copies or optimizer.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>

#include "vibeqc/periodic_correlation_diabatic_seed.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_diabatic_seed(py::module_& m) {
    using Options = vibeqc::PeriodicCorrelationDiabaticSeedOptions;
    using Memory = vibeqc::PeriodicCorrelationDiabaticSeedMemoryPlan;
    using Diagnostics = vibeqc::PeriodicCorrelationDiabaticSeedDiagnostics;
    using Seed = vibeqc::PeriodicCorrelationDiabaticSeed;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Complex = std::complex<double>;
    m.attr("_PERIODIC_CORRELATION_DIABATIC_SEED_CONTRACT_VERSION") =
        py::int_(vibeqc::kPeriodicCorrelationDiabaticSeedContractVersion);
    auto options = py::class_<Options>(m, "_PeriodicCorrelationDiabaticSeedOptions");
    options.def(py::init<>());
#define DIABATIC_OPTION(field) options.def_readwrite(#field, &Options::field)
    DIABATIC_OPTION(absolute_tolerance);
    DIABATIC_OPTION(relative_tolerance);
    DIABATIC_OPTION(unitarity_tolerance);
    DIABATIC_OPTION(cholesky_absolute_floor);
    DIABATIC_OPTION(cholesky_relative_floor);
    DIABATIC_OPTION(singular_absolute_floor);
    DIABATIC_OPTION(singular_relative_floor);
    DIABATIC_OPTION(jacobi_max_sweeps);
    DIABATIC_OPTION(jacobi_relative_tolerance);
    DIABATIC_OPTION(maximum_work_units);
#undef DIABATIC_OPTION
    auto memory = py::class_<Memory>(m, "_PeriodicCorrelationDiabaticSeedMemoryPlan");
#define DIABATIC_MEMORY(field) memory.def_readonly(#field, &Memory::field)
    DIABATIC_MEMORY(n_cells); DIABATIC_MEMORY(n_basis); DIABATIC_MEMORY(n_active);
    DIABATIC_MEMORY(n_effective_orbitals); DIABATIC_MEMORY(self_inverse_count);
    DIABATIC_MEMORY(gauge_count); DIABATIC_MEMORY(active_index_count);
    DIABATIC_MEMORY(retained_gauge_bytes); DIABATIC_MEMORY(retained_index_bytes);
    DIABATIC_MEMORY(frame_workspace_bytes); DIABATIC_MEMORY(polar_workspace_bytes);
    DIABATIC_MEMORY(scalar_workspace_bytes); DIABATIC_MEMORY(peak_owned_numerical_bytes);
    DIABATIC_MEMORY(output_numerical_bytes); DIABATIC_MEMORY(reference_preflight_work_units);
    DIABATIC_MEMORY(anchor_work_units); DIABATIC_MEMORY(polar_work_units_per_point);
    DIABATIC_MEMORY(physical_work_units_per_point); DIABATIC_MEMORY(maximum_work_units);
#undef DIABATIC_MEMORY
    auto diagnostics = py::class_<Diagnostics>(m, "_PeriodicCorrelationDiabaticSeedDiagnostics");
#define DIABATIC_DIAGNOSTIC(field) diagnostics.def_readonly(#field, &Diagnostics::field)
    DIABATIC_DIAGNOSTIC(required_node_memory_bytes); DIABATIC_DIAGNOSTIC(charged_work_units);
    DIABATIC_DIAGNOSTIC(polar_point_count); DIABATIC_DIAGNOSTIC(sewn_pair_count);
    DIABATIC_DIAGNOSTIC(self_inverse_point_count); DIABATIC_DIAGNOSTIC(jacobi_sweeps);
    DIABATIC_DIAGNOSTIC(maximum_overlap_time_reversal_residual);
    DIABATIC_DIAGNOSTIC(maximum_fock_time_reversal_residual);
    DIABATIC_DIAGNOSTIC(maximum_active_projector_time_reversal_residual);
    DIABATIC_DIAGNOSTIC(maximum_frozen_projector_time_reversal_residual);
    DIABATIC_DIAGNOSTIC(maximum_anchor_density_residual);
    DIABATIC_DIAGNOSTIC(maximum_anchor_imaginary_correction);
    DIABATIC_DIAGNOSTIC(minimum_cholesky_pivot);
    DIABATIC_DIAGNOSTIC(minimum_similarity_singular_value);
    DIABATIC_DIAGNOSTIC(maximum_similarity_singular_value);
    DIABATIC_DIAGNOSTIC(maximum_dilation_eigen_residual);
    DIABATIC_DIAGNOSTIC(maximum_procrustes_trace_residual);
    DIABATIC_DIAGNOSTIC(maximum_unitarity_residual);
    DIABATIC_DIAGNOSTIC(maximum_metric_orthonormality_residual);
    DIABATIC_DIAGNOSTIC(maximum_physical_reconstruction_residual);
    DIABATIC_DIAGNOSTIC(maximum_physical_time_reversal_residual);
    DIABATIC_DIAGNOSTIC(maximum_self_inverse_imaginary_magnitude);
#undef DIABATIC_DIAGNOSTIC
    py::class_<Seed>(m, "_PeriodicCorrelationDiabaticSeed")
        .def_property_readonly("contract_version", &Seed::contract_version)
        .def_property_readonly("state", [](const Seed& value) -> std::shared_ptr<State> {
            (void) value.state(); return std::const_pointer_cast<State>(value.state_handle());
        })
        .def_property_readonly("memory", [](const Seed& value) { return value.memory(); })
        .def_property_readonly("options", [](const Seed& value) { return value.options(); })
        .def_property_readonly("diagnostics", [](const Seed& value) { return value.diagnostics(); })
        .def_property_readonly("allocation_identity", &Seed::allocation_identity)
        .def_property_readonly("gauge_payload_sha256", &Seed::gauge_payload_sha256)
        .def_property_readonly("seed_identity_sha256", &Seed::seed_identity_sha256)
        .def("active_band", &Seed::active_band, py::arg("point"), py::arg("active"))
        .def("gamma_pivot", &Seed::gamma_pivot, py::arg("active"))
        .def("gauge", &Seed::gauge, py::arg("point"), py::arg("row"), py::arg("column"))
        .def("point_gauge_copy", [](const Seed& value, std::size_t point) {
            const auto n = value.memory().n_active;
            if (n > 16U) throw std::length_error("Diabatic seed diagnostic copy is limited to active rank 16");
            const auto* source = value.point_gauge(point);
            py::array_t<Complex> result({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            std::memcpy(result.mutable_data(), source, n * n * sizeof(Complex));
            return result;
        }, py::arg("point"));
    m.def("_plan_periodic_correlation_diabatic_seed", &vibeqc::plan_periodic_correlation_diabatic_seed,
        py::arg("mesh"), py::arg("n_basis"), py::arg("n_active"),
        py::arg("n_effective_orbitals"), py::arg("jacobi_max_sweeps"));
    m.def("_make_periodic_correlation_diabatic_seed",
        [](const Reference& reference, std::uint64_t cap, const Options& controls) {
            const auto& state = reference.state();
            if (state.n_kpoints() > 64U || state.n_basis() > 64U
                || state.n_correlated_occupied() > 16U || controls.jacobi_max_sweeps > 256U) {
                throw std::length_error("Diabatic seed diagnostic is limited to 64 k points, 64 AOs, rank 16 and 256 sweeps");
            }
            py::gil_scoped_release release;
            return vibeqc::make_periodic_correlation_diabatic_seed(reference, cap, controls);
        }, py::arg("reference"), py::arg("owned_numerical_byte_cap"), py::arg("options"),
        "Native bounded diabatic starting gauge only, not a localization optimizer or convergence certificate.");
}
