// Tiny inspection bindings for the finite-source real-row reference provider.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include "vibeqc/periodic_correlation_real_local_provider.hpp"
#include "periodic_correlation_real_local_provider_internal.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_real_local_provider(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationRealLocalProviderOptions;
    using Caps = PeriodicCorrelationRealLocalProviderCaps;
    using Memory = PeriodicCorrelationRealLocalProviderMemoryPlan;
    using Diagnostics = PeriodicCorrelationRealLocalProviderDiagnostics;
    using Provider = PeriodicCorrelationRealLocalProvider;
    using Integral = PeriodicCorrelationRealLocalIntegral;
    m.attr("_PERIODIC_CORRELATION_REAL_LOCAL_PROVIDER_CONTRACT_VERSION") = py::int_(1);
#define REAL_PROVIDER_OPTION(name) .def_readwrite(#name, &Options::name)
    py::class_<Options>(m, "_PeriodicCorrelationRealLocalProviderOptions").def(py::init<>())
        REAL_PROVIDER_OPTION(reversal_absolute_tolerance) REAL_PROVIDER_OPTION(reversal_relative_tolerance)
        REAL_PROVIDER_OPTION(conjugacy_absolute_tolerance) REAL_PROVIDER_OPTION(conjugacy_relative_tolerance)
        REAL_PROVIDER_OPTION(self_q_absolute_tolerance) REAL_PROVIDER_OPTION(self_q_relative_tolerance)
        REAL_PROVIDER_OPTION(maximum_eri_projection_error) REAL_PROVIDER_OPTION(maximum_scalar_roundoff_error);
#undef REAL_PROVIDER_OPTION
#define REAL_PROVIDER_CAP(name) .def_readwrite(#name, &Caps::name)
    py::class_<Caps>(m, "_PeriodicCorrelationRealLocalProviderCaps").def(py::init<>())
        REAL_PROVIDER_CAP(maximum_owned_numerical_bytes) REAL_PROVIDER_CAP(maximum_work_units)
        REAL_PROVIDER_CAP(maximum_factor_panels) REAL_PROVIDER_CAP(maximum_tile_visits)
        REAL_PROVIDER_CAP(maximum_reader_tile_bytes) REAL_PROVIDER_CAP(maximum_scalar_work_units);
#undef REAL_PROVIDER_CAP
#define REAL_PROVIDER_MEMORY(name) .def_readonly(#name, &Memory::name)
    py::class_<Memory>(m, "_PeriodicCorrelationRealLocalProviderMemoryPlan")
        REAL_PROVIDER_MEMORY(n_cells) REAL_PROVIDER_MEMORY(n_auxiliary) REAL_PROVIDER_MEMORY(occupied_count)
        REAL_PROVIDER_MEMORY(virtual_count) REAL_PROVIDER_MEMORY(orbital_count) REAL_PROVIDER_MEMORY(density_count)
        REAL_PROVIDER_MEMORY(row_count) REAL_PROVIDER_MEMORY(self_inverse_q_count)
        REAL_PROVIDER_MEMORY(retained_row_bytes) REAL_PROVIDER_MEMORY(norm_workspace_bytes)
        REAL_PROVIDER_MEMORY(retained_partner_panel_bytes) REAL_PROVIDER_MEMORY(building_panel_peak_bytes)
        REAL_PROVIDER_MEMORY(peak_owned_numerical_bytes) REAL_PROVIDER_MEMORY(live_basis_bytes)
        REAL_PROVIDER_MEMORY(basis_index_alias_bytes) REAL_PROVIDER_MEMORY(caller_gauge_bytes)
        REAL_PROVIDER_MEMORY(live_wannier_bytes) REAL_PROVIDER_MEMORY(live_domain_bytes) REAL_PROVIDER_MEMORY(live_space_bytes)
        REAL_PROVIDER_MEMORY(live_reader_numeric_bytes) REAL_PROVIDER_MEMORY(live_reader_control_bytes)
        REAL_PROVIDER_MEMORY(maximum_reader_tile_bytes) REAL_PROVIDER_MEMORY(factor_panels)
        REAL_PROVIDER_MEMORY(tile_visits) REAL_PROVIDER_MEMORY(work_units) REAL_PROVIDER_MEMORY(scalar_work_units)
        REAL_PROVIDER_MEMORY(required_node_memory_bytes);
#undef REAL_PROVIDER_MEMORY
#define REAL_PROVIDER_DIAGNOSTIC(name) .def_readonly(#name, &Diagnostics::name)
    py::class_<Diagnostics>(m, "_PeriodicCorrelationRealLocalProviderDiagnostics")
        REAL_PROVIDER_DIAGNOSTIC(factor_panels_built) REAL_PROVIDER_DIAGNOSTIC(tile_visits)
        REAL_PROVIDER_DIAGNOSTIC(maximum_reversal_error) REAL_PROVIDER_DIAGNOSTIC(maximum_conjugacy_error)
        REAL_PROVIDER_DIAGNOSTIC(maximum_self_q_imaginary_magnitude) REAL_PROVIDER_DIAGNOSTIC(maximum_original_density_norm)
        REAL_PROVIDER_DIAGNOSTIC(maximum_reversal_norm) REAL_PROVIDER_DIAGNOSTIC(maximum_covariance_projection_norm)
        REAL_PROVIDER_DIAGNOSTIC(maximum_real_row_conversion_norm) REAL_PROVIDER_DIAGNOSTIC(orientation_error_bound)
        REAL_PROVIDER_DIAGNOSTIC(covariance_error_bound) REAL_PROVIDER_DIAGNOSTIC(conversion_error_bound)
        REAL_PROVIDER_DIAGNOSTIC(maximum_eri_projection_error_bound);
#undef REAL_PROVIDER_DIAGNOSTIC
    py::class_<Integral>(m, "_PeriodicCorrelationRealLocalIntegral")
        .def_readonly("value", &Integral::value)
        .def_readonly("roundoff_error_bound", &Integral::roundoff_error_bound);
    m.def("_diagnostic_real_local_dot_interval", [](
        const py::array& a, const py::array& b, std::uint64_t maximum_work, double maximum_error) {
        if (!a.dtype().is(py::dtype::of<double>()) || !b.dtype().is(py::dtype::of<double>())
            || a.ndim() != 1 || b.ndim() != 1 || !(a.flags() & py::array::c_style)
            || !(b.flags() & py::array::c_style) || a.size() != b.size())
            throw std::invalid_argument("nonphysical scalar-dot diagnostic requires equal contiguous float64 vectors");
        if (a.size() > 256 || maximum_work > 65600)
            throw std::length_error("nonphysical scalar-dot diagnostic exceeds tiny lane or work cap");
        const auto count = static_cast<std::size_t>(a.size());
        return periodic_correlation_real_local_detail::dot_interval(
            static_cast<const double*>(a.data()),count,1,static_cast<const double*>(b.data()),count,1,
            count,maximum_work,maximum_error);
    },py::arg("left").noconvert(),py::arg("right").noconvert(),py::arg("maximum_work"),py::arg("maximum_error"));
    py::class_<Provider>(m, "_PeriodicCorrelationRealLocalProvider")
        .def_property_readonly("contract_version", &Provider::contract_version)
        .def_property_readonly("memory", [](const Provider& p) { return p.memory(); })
        .def_property_readonly("options", [](const Provider& p) { return p.options(); })
        .def_property_readonly("diagnostics", [](const Provider& p) { return p.diagnostics(); })
        .def_property_readonly("identity_sha256", &Provider::identity_sha256)
        .def_property_readonly("payload_sha256", &Provider::payload_sha256)
        .def_property_readonly("local_basis_identity_sha256", &Provider::local_basis_identity_sha256)
        .def_property_readonly("basis_certificate_identity_sha256", &Provider::basis_certificate_identity_sha256)
        .def_property_readonly("store_identity_sha256", &Provider::store_identity_sha256)
        .def_property_readonly("consumed_panels_identity_sha256", &Provider::consumed_panels_identity_sha256)
        .def_property_readonly("finite_image_reference", &Provider::finite_image_reference)
        .def_property_readonly("hf_hamiltonian_match_certified", &Provider::hf_hamiltonian_match_certified)
        .def_property_readonly("ao_image_source_certified", &Provider::ao_image_source_certified)
        .def_property_readonly("matched_finite_gaussian_hf_recipe", &Provider::matched_finite_gaussian_hf_recipe)
        .def_property_readonly("bitwise_hf_factor_consumption_verified", &Provider::bitwise_hf_factor_consumption_verified)
        .def_property_readonly("source_context_identity_sha256", &Provider::source_context_identity_sha256)
        .def_property_readonly("hf_reference_source_identity_sha256", &Provider::hf_reference_source_identity_sha256)
        .def("row", &Provider::row)
        .def("integral", &Provider::integral)
        .def("provider_inventory", [](const Provider& p, const PeriodicCorrelationRealLocalBasis& basis) {
            const auto view = p.integral_provider(basis);
            return py::make_tuple(view.retained_numerical_bytes, view.maximum_transient_numerical_bytes);
        })
        .def("rows_copy", [](const Provider& p) {
            const auto rows = p.memory().row_count, densities = p.memory().density_count;
            if (rows > 64 || densities > 136)
                throw std::length_error("real-local provider diagnostic exceeds tiny row shape cap");
            py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(densities)});
            std::memcpy(out.mutable_data(), p.rows_data(), p.memory().retained_row_bytes);
            return out;
        });
    m.def("_plan_periodic_correlation_real_local_provider", &plan_periodic_correlation_real_local_provider,
        py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
        py::arg("domain"), py::arg("space"), py::arg("basis"));
    m.def("_make_periodic_correlation_real_local_provider", [](
        const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
        const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
        const py::array& gauges, const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
        const PeriodicCorrelationRealLocalBasis& basis, const Options& options, const Caps& caps) {
        using Complex = std::complex<double>;
        const auto& state = reference.state();
        if (state.n_kpoints() > 8 || state.n_basis() > 8 || state.n_correlated_occupied() > 4
            || schedule.shape().n_auxiliary > 8 || domain.domain_dimension() > 32 || basis.memory().orbital_count > 16
            || schedule.shape().tile_count > 128 || caps.maximum_owned_numerical_bytes > 1048576
            || caps.maximum_work_units > 100000000 || caps.maximum_scalar_work_units > 1000000
            || caps.maximum_factor_panels > 64 || caps.maximum_tile_visits > 128 || caps.maximum_reader_tile_bytes > 8192)
            throw std::length_error("real-local provider diagnostic exceeds tiny shape or work cap");
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim() != 3 || !(gauges.flags() & py::array::c_style)
            || gauges.shape(0) != static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1) != static_cast<py::ssize_t>(state.n_correlated_occupied())
            || gauges.shape(2) != static_cast<py::ssize_t>(state.n_correlated_occupied()))
            throw std::invalid_argument("real-local provider gauges require C-contiguous complex128 [Nk,nactive,nactive]");
        const auto* g = static_cast<const Complex*>(gauges.data()); const auto ng = static_cast<std::size_t>(gauges.size());
        const auto settings = options; const auto limits = caps;
        py::gil_scoped_release release;
        return make_periodic_correlation_real_local_provider(
            reference,schedule,reader,wannier,g,ng,domain,space,basis,settings,limits);
    }, py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
       py::arg("gauges").noconvert(), py::arg("domain"), py::arg("space"), py::arg("basis"),
       py::arg("options"), py::arg("caps"));
}
