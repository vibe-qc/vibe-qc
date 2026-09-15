// Bounded internal diagnostics. Scientific production consumes native results.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>
#include "vibeqc/periodic_correlation_metric_factorization.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {
py::array_t<std::complex<double>> metric_factorization_matrix_copy(
    const std::vector<std::complex<double>>& matrix, std::uint64_t n) {
    if (n == 0U || n > 64U || matrix.size() != n * n) {
        throw std::length_error("metric factorization matrix diagnostic requires a live result of at most 64 rows");
    }
    py::array_t<std::complex<double>> result({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
    std::memcpy(result.mutable_data(), matrix.data(), matrix.size() * sizeof(std::complex<double>));
    return result;
}
}  // namespace

void bind_periodic_correlation_metric_factorization(py::module_& m) {
    using Result = vibeqc::PeriodicCorrelationMetricFactorizationResult;
    using Diagnostics = vibeqc::PeriodicCorrelationMetricFactorizationDiagnostics;
    py::class_<Diagnostics>(m, "_PeriodicCorrelationMetricFactorizationDiagnostics")
        .def_readonly("n_auxiliary", &Diagnostics::n_auxiliary)
        .def_readonly("retained_rank", &Diagnostics::retained_rank)
        .def_readonly("sweeps", &Diagnostics::sweeps)
        .def_readonly("rotations", &Diagnostics::rotations)
        .def_readonly("rank_cutoff", &Diagnostics::rank_cutoff)
        .def_readonly("negative_tolerance", &Diagnostics::negative_tolerance)
        .def_readonly("minimum_eigenvalue", &Diagnostics::minimum_eigenvalue)
        .def_readonly("maximum_eigenvalue", &Diagnostics::maximum_eigenvalue)
        .def_readonly("smallest_retained_eigenvalue", &Diagnostics::smallest_retained_eigenvalue)
        .def_readonly("largest_discarded_eigenvalue", &Diagnostics::largest_discarded_eigenvalue)
        .def_readonly("input_scale", &Diagnostics::input_scale)
        .def_readonly("scaled_initial_frobenius_norm", &Diagnostics::scaled_initial_frobenius_norm)
        .def_readonly("scaled_final_offdiagonal_norm", &Diagnostics::scaled_final_offdiagonal_norm)
        .def_readonly("orthogonality_frobenius_error", &Diagnostics::orthogonality_frobenius_error)
        .def_readonly("maximum_self_conjugate_imaginary_residual", &Diagnostics::maximum_self_conjugate_imaginary_residual);
    py::class_<Result>(m, "_PeriodicCorrelationMetricFactorizationResult")
        .def_property_readonly("contract_version", &Result::contract_version)
        .def_property_readonly("q_index", &Result::q_index)
        .def_property_readonly("conjugate_q_index", &Result::conjugate_q_index)
        .def_property_readonly("self_conjugate_transfer", &Result::self_conjugate_transfer)
        .def_property_readonly("diagnostics", &Result::diagnostics)
        .def_property_readonly("source_identity_sha256", &Result::source_identity_sha256)
        .def_property_readonly("input_payload_identity_sha256", &Result::input_payload_identity_sha256)
        .def_property_readonly("payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly("census_identity_sha256", &Result::census_identity_sha256)
        .def_property_readonly("plan_identity_sha256", &Result::plan_identity_sha256)
        .def_property_readonly("auxiliary_basis_identity_sha256", &Result::auxiliary_basis_identity_sha256)
        .def_property_readonly("admitted_factorization_peak_bytes", &Result::admitted_factorization_peak_bytes)
        .def_property_readonly("admitted_whitener_peak_bytes", &Result::admitted_whitener_peak_bytes)
        .def_property_readonly("numerical_peak_bytes", &Result::numerical_peak_bytes)
        .def_property_readonly("matrix", [](const Result& r) {
            return metric_factorization_matrix_copy(r.matrix_row_major(), r.diagnostics().n_auxiliary);
        })
        .def_property_readonly("state", [](const Result& r) {
            if (!r.state_handle()) throw std::logic_error("consumed metric factorization has no state");
            return std::const_pointer_cast<vibeqc::PeriodicRestrictedMeanFieldState>(r.state_handle());
        });
    m.def("_periodic_correlation_metric_factorization_backend_identity_sha256",
          &vibeqc::periodic_correlation_metric_factorization_backend_identity_sha256);
    m.def("_factorize_periodic_correlation_metric", [](
        const vibeqc::PeriodicCorrelationAdmittedReference& reference,
        const vibeqc::PeriodicCorrelationFactorStreamSchedule& schedule,
        const vibeqc::PeriodicCorrelationFactorBuildCensus& census,
        vibeqc::PeriodicCorrelationReciprocalMetricResult& raw, double negative_tolerance) {
        if (raw.n_auxiliary() > 64U) {
            throw std::length_error("metric factorization diagnostic is limited to 64 auxiliary functions");
        }
        return vibeqc::factorize_periodic_correlation_metric(
            reference, schedule, census, std::move(raw), negative_tolerance);
    }, py::arg("reference"), py::arg("schedule"), py::arg("census"),
       py::arg("raw"), py::arg("negative_tolerance"));
    m.def("_metric_principal_square_root_diagnostic", [](
        const py::array& input, double rank_cutoff, double negative_tolerance, bool self_conjugate) {
        if (input.ndim() != 2 || input.shape(0) != input.shape(1)
            || input.shape(0) < 1 || input.shape(0) > 16) {
            throw std::length_error("metric square-root diagnostic requires a square matrix of 1..16 rows");
        }
        if (!input.dtype().is(py::dtype::of<std::complex<double>>())
            || !(input.flags() & py::array::c_style)) {
            throw std::invalid_argument("metric square-root diagnostic requires C-contiguous complex128 input");
        }
        const auto n = static_cast<std::uint64_t>(input.shape(0));
        std::vector<std::complex<double>> matrix(n * n);
        std::memcpy(matrix.data(), input.data(), matrix.size() * sizeof(std::complex<double>));
        auto result = vibeqc::detail::metric_principal_square_root_diagnostic(
            std::move(matrix), n, rank_cutoff, negative_tolerance, self_conjugate);
        py::dict output;
        output["matrix"] = metric_factorization_matrix_copy(result.whitener, n);
        output["diagnostics"] = py::cast(result.diagnostics);
        return output;
    }, py::arg("matrix"), py::arg("rank_cutoff"), py::arg("negative_tolerance"),
       py::arg("self_conjugate") = false);
}
