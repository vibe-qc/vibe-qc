// Bounded internal diagnostics for the native one-q reciprocal source and raw
// auxiliary metric. Production periodic code consumes the C++ results directly.

#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include "vibeqc/periodic_correlation_reciprocal_metric.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {

constexpr std::uint64_t
    kReciprocalMetricSourceDiagnosticMaximumCandidates = 65536U;
constexpr std::uint64_t
    kReciprocalMetricDiagnosticMaximumAuxiliary = 64U;
constexpr std::uint64_t
    kReciprocalMetricDiagnosticMaximumPanelVectors = 4096U;
constexpr std::uint64_t
    kReciprocalMetricDiagnosticMaximumPanelElements = 65536U;
constexpr std::uint64_t
    kReciprocalMetricDiagnosticMaximumAccumulationTerms = 2000000U;

std::shared_ptr<vibeqc::PeriodicRestrictedMeanFieldState>
reciprocal_metric_source_state_for_binding(
    const vibeqc::PeriodicCorrelationReciprocalMetricSourceManifest&
        manifest) {
    const auto& state = manifest.state_handle();
    if (!state) {
        throw std::logic_error(
            "moved-from reciprocal metric source manifest has no "
            "mean-field state");
    }
    return std::const_pointer_cast<
        vibeqc::PeriodicRestrictedMeanFieldState>(state);
}

std::shared_ptr<vibeqc::PeriodicRestrictedMeanFieldState>
reciprocal_metric_result_state_for_binding(
    const vibeqc::PeriodicCorrelationReciprocalMetricResult& result) {
    const auto& state = result.state_handle();
    if (!state) {
        throw std::logic_error(
            "moved-from reciprocal metric result has no mean-field state");
    }
    return std::const_pointer_cast<
        vibeqc::PeriodicRestrictedMeanFieldState>(state);
}

py::array_t<std::complex<double>> reciprocal_metric_matrix_copy(
    const vibeqc::PeriodicCorrelationReciprocalMetricResult& result) {
    const std::uint64_t n = result.n_auxiliary();
    if (n != 0U && n > std::numeric_limits<std::uint64_t>::max() / n) {
        throw std::overflow_error(
            "reciprocal metric matrix extent overflows uint64");
    }
    const std::uint64_t expected = n * n;
    const auto& matrix = result.matrix_row_major();
    if (expected
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())
        || matrix.size() != static_cast<std::size_t>(expected)) {
        throw std::logic_error(
            "reciprocal metric result has an inconsistent matrix extent");
    }
    if (n > static_cast<std::uint64_t>(
                std::numeric_limits<py::ssize_t>::max())) {
        throw std::overflow_error(
            "reciprocal metric dimension exceeds Python ssize_t");
    }
    const auto python_n = static_cast<py::ssize_t>(n);
    py::array_t<std::complex<double>> copy({python_n, python_n});
    std::memcpy(
        copy.mutable_data(),
        matrix.data(),
        matrix.size() * sizeof(std::complex<double>));
    return copy;
}

void require_reciprocal_metric_diagnostic_caps(
    const vibeqc::PeriodicCorrelationFactorBuildCensus& census,
    const vibeqc::PeriodicCorrelationReciprocalMetricSourceManifest& source) {
    const std::uint64_t n = census.shape().n_auxiliary;
    const std::uint64_t panel = census.config().reciprocal_block;
    const std::uint64_t vectors = source.accepted_vector_count();
    if (source.candidate_count()
        > kReciprocalMetricSourceDiagnosticMaximumCandidates) {
        throw std::length_error(
            "internal reciprocal metric diagnostic is limited to 65536 "
            "Cartesian candidates");
    }
    if (n > kReciprocalMetricDiagnosticMaximumAuxiliary) {
        throw std::length_error(
            "internal reciprocal metric diagnostic auxiliary extent is "
            "limited to 64");
    }
    if (panel > kReciprocalMetricDiagnosticMaximumPanelVectors) {
        throw std::length_error(
            "internal reciprocal metric diagnostic panel is limited to "
            "4096 vectors");
    }
    if (n != 0U
        && panel > kReciprocalMetricDiagnosticMaximumPanelElements / n) {
        throw std::length_error(
            "internal reciprocal metric diagnostic Fourier panel is "
            "limited to 65536 complex elements");
    }
    if (n != 0U
        && n > kReciprocalMetricDiagnosticMaximumAccumulationTerms / n) {
        throw std::length_error(
            "internal reciprocal metric diagnostic accumulation is "
            "limited to 2000000 scalar terms");
    }
    const std::uint64_t square = n * n;
    if (vectors != 0U
        && square
            > kReciprocalMetricDiagnosticMaximumAccumulationTerms / vectors) {
        throw std::length_error(
            "internal reciprocal metric diagnostic accumulation is "
            "limited to 2000000 scalar terms");
    }
}

}  // namespace

void bind_periodic_correlation_reciprocal_metric(py::module_& m) {
    using Config = vibeqc::PeriodicCorrelationFactorBuildConfig;
    using EnumerationPlan =
        vibeqc::PeriodicCorrelationReciprocalMetricEnumerationPlan;
    using Manifest =
        vibeqc::PeriodicCorrelationReciprocalMetricSourceManifest;
    using Census = vibeqc::PeriodicCorrelationFactorBuildCensus;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;
    using Result = vibeqc::PeriodicCorrelationReciprocalMetricResult;
    using Schedule = vibeqc::PeriodicCorrelationFactorStreamSchedule;

    m.def("_require_periodic_correlation_reciprocal_source_conjugacy",
        [](const Manifest& source, const Manifest& opposite, std::uint64_t cap) {
            if (cap > kReciprocalMetricSourceDiagnosticMaximumCandidates) {
                throw std::length_error("reciprocal conjugacy diagnostic exceeds tiny candidate limit");
            }
            py::gil_scoped_release release;
            return vibeqc::require_periodic_correlation_reciprocal_source_conjugacy(source, opposite, cap);
        }, py::arg("source"), py::arg("conjugate_source"),
        py::arg("maximum_candidates_per_source"));

    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_SOURCE_CONTRACT_VERSION") =
        py::int_(
            vibeqc::
                kPeriodicCorrelationReciprocalMetricSourceContractVersion);
    m.attr("_PERIODIC_CORRELATION_RECIPROCAL_METRIC_PRODUCER_VERSION") =
        py::int_(
            vibeqc::kPeriodicCorrelationReciprocalMetricProducerVersion);
    m.attr("_PERIODIC_CORRELATION_RECIPROCAL_METRIC_PRODUCER_ID") =
        py::str(vibeqc::kPeriodicCorrelationReciprocalMetricProducerId);
    m.attr("_PERIODIC_CORRELATION_RECIPROCAL_METRIC_SOURCE_MANIFEST_BYTES") =
        py::int_(
            vibeqc::
                kPeriodicCorrelationReciprocalMetricSourceManifestBytes);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_SOURCE_FIXED_PREFIX_WIRE_BYTES") =
        py::int_(
            vibeqc::
                kPeriodicCorrelationReciprocalMetricSourceFixedPrefixWireBytes);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_ACCEPTED_VECTOR_WIRE_BYTES") =
        py::int_(
            vibeqc::
                kPeriodicCorrelationReciprocalMetricAcceptedVectorWireBytes);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_CANDIDATES") =
        py::int_(kReciprocalMetricSourceDiagnosticMaximumCandidates);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_RESULT_CONTRACT_VERSION") =
        py::int_(
            vibeqc::
                kPeriodicCorrelationReciprocalMetricResultContractVersion);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_PAYLOAD_FIXED_PREFIX_WIRE_BYTES") =
        py::int_(
            vibeqc::
                kPeriodicCorrelationReciprocalMetricPayloadFixedPrefixWireBytes);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_AUXILIARY") =
        py::int_(kReciprocalMetricDiagnosticMaximumAuxiliary);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_PANEL_VECTORS") =
        py::int_(kReciprocalMetricDiagnosticMaximumPanelVectors);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_PANEL_ELEMENTS") =
        py::int_(kReciprocalMetricDiagnosticMaximumPanelElements);
    m.attr(
        "_PERIODIC_CORRELATION_RECIPROCAL_METRIC_DIAGNOSTIC_MAX_ACCUMULATION_TERMS") =
        py::int_(kReciprocalMetricDiagnosticMaximumAccumulationTerms);

    py::class_<EnumerationPlan>(
        m, "_PeriodicCorrelationReciprocalMetricEnumerationPlan")
        .def_readonly(
            "normalization_scale", &EnumerationPlan::normalization_scale)
        .def_readonly(
            "determinant_lower_bound",
            &EnumerationPlan::determinant_lower_bound)
        .def_readonly(
            "determinant_upper_bound",
            &EnumerationPlan::determinant_upper_bound)
        .def_readonly(
            "normalized_matrix_infinity_norm_upper_bound",
            &EnumerationPlan::
                normalized_matrix_infinity_norm_upper_bound)
        .def_readonly(
            "inverse_row_one_norm_upper_bounds",
            &EnumerationPlan::inverse_row_one_norm_upper_bounds)
        .def_readonly(
            "inverse_row_two_norm_upper_bounds",
            &EnumerationPlan::inverse_row_two_norm_upper_bounds)
        .def_readonly(
            "dot_relative_error_upper_bound",
            &EnumerationPlan::dot_relative_error_upper_bound)
        .def_readonly(
            "dot_absolute_error_upper_bound",
            &EnumerationPlan::dot_absolute_error_upper_bound)
        .def_readonly(
            "accepted_vector_radius_upper_bound",
            &EnumerationPlan::accepted_vector_radius_upper_bound)
        .def_readonly(
            "matvec_feedback_upper_bound",
            &EnumerationPlan::matvec_feedback_upper_bound)
        .def_readonly(
            "matvec_error_upper_bound",
            &EnumerationPlan::matvec_error_upper_bound)
        .def_readonly(
            "shifted_vector_infinity_norm_upper_bound",
            &EnumerationPlan::
                shifted_vector_infinity_norm_upper_bound)
        .def_readonly(
            "stored_shift_component_upper_bounds",
            &EnumerationPlan::stored_shift_component_upper_bounds)
        .def_readonly(
            "exact_shift_component_upper_bounds",
            &EnumerationPlan::exact_shift_component_upper_bounds)
        .def_readonly("lower_bounds", &EnumerationPlan::lower_bounds)
        .def_readonly("upper_bounds", &EnumerationPlan::upper_bounds)
        .def_readonly(
            "candidate_count", &EnumerationPlan::candidate_count);

    py::class_<Manifest>(
        m, "_PeriodicCorrelationReciprocalMetricSourceManifest")
        .def_property_readonly(
            "contract_version", &Manifest::contract_version)
        .def_property_readonly(
            "state", &reciprocal_metric_source_state_for_binding)
        .def_property_readonly(
            "source_identity_sha256", &Manifest::source_identity_sha256)
        .def_property_readonly(
            "source_wire_bytes",
            [](const Manifest& manifest) {
                return vibeqc::
                    periodic_correlation_reciprocal_metric_source_wire_bytes(
                        manifest.accepted_vector_count());
            })
        .def_property_readonly(
            "periodic_dimension", &Manifest::periodic_dimension)
        .def_property_readonly("mesh", &Manifest::mesh)
        .def_property_readonly("is_shift", &Manifest::is_shift)
        .def_property_readonly("q_index", &Manifest::q_index)
        .def_property_readonly(
            "centered_doubled_numerator",
            &Manifest::centered_doubled_numerator)
        .def_property_readonly(
            "centered_reciprocal_wrap",
            &Manifest::centered_reciprocal_wrap)
        .def_property_readonly(
            "q_fractional", &Manifest::q_fractional)
        .def_property_readonly(
            "q_cartesian", &Manifest::q_cartesian)
        .def_property_readonly(
            "reciprocal_lattice", &Manifest::reciprocal_lattice)
        .def_property_readonly(
            "reciprocal_energy_cutoff",
            &Manifest::reciprocal_energy_cutoff)
        .def_property_readonly(
            "maximum_reciprocal_radius",
            &Manifest::maximum_reciprocal_radius)
        .def_property_readonly(
            "radial_boundary_tolerance",
            &Manifest::radial_boundary_tolerance)
        .def_property_readonly(
            "cell_volume_bohr3", &Manifest::cell_volume_bohr3)
        .def_property_readonly(
            "lower_bounds", &Manifest::lower_bounds)
        .def_property_readonly(
            "upper_bounds", &Manifest::upper_bounds)
        .def_property_readonly(
            "candidate_count", &Manifest::candidate_count)
        .def_property_readonly(
            "accepted_vector_count", &Manifest::accepted_vector_count)
        .def_property_readonly(
            "zero_mode_excluded_count",
            &Manifest::zero_mode_excluded_count)
        .def(
            "factor_build_q_record", &Manifest::factor_build_q_record);

    py::class_<Result>(m, "_PeriodicCorrelationReciprocalMetricResult")
        .def_property_readonly(
            "contract_version", &Result::contract_version)
        .def_property_readonly(
            "state", &reciprocal_metric_result_state_for_binding)
        .def_property_readonly(
            "source_identity_sha256", &Result::source_identity_sha256)
        .def_property_readonly(
            "census_identity_sha256", &Result::census_identity_sha256)
        .def_property_readonly(
            "plan_identity_sha256", &Result::plan_identity_sha256)
        .def_property_readonly(
            "payload_identity_sha256", &Result::payload_identity_sha256)
        .def_property_readonly(
            "auxiliary_basis_identity_sha256",
            &Result::auxiliary_basis_identity_sha256)
        .def_property_readonly("q_index", &Result::q_index)
        .def_property_readonly(
            "conjugate_q_index", &Result::conjugate_q_index)
        .def_property_readonly("n_auxiliary", &Result::n_auxiliary)
        .def_property_readonly(
            "accepted_vector_count", &Result::accepted_vector_count)
        .def_property_readonly(
            "reciprocal_panel_capacity",
            &Result::reciprocal_panel_capacity)
        .def_property_readonly(
            "completed_panel_count", &Result::completed_panel_count)
        .def_property_readonly("metric_bytes", &Result::metric_bytes)
        .def_property_readonly(
            "reciprocal_panel_bytes", &Result::reciprocal_panel_bytes)
        .def_property_readonly(
            "maximum_auxiliary_fourier_panel_bytes",
            &Result::maximum_auxiliary_fourier_panel_bytes)
        .def_property_readonly(
            "maximum_weighted_fourier_panel_bytes",
            &Result::maximum_weighted_fourier_panel_bytes)
        .def_property_readonly(
            "admitted_reciprocal_metric_peak_bytes",
            &Result::admitted_reciprocal_metric_peak_bytes)
        .def_property_readonly(
            "self_conjugate_transfer", &Result::self_conjugate_transfer)
        .def_property_readonly(
            "maximum_self_conjugate_imaginary_residual",
            &Result::maximum_self_conjugate_imaginary_residual)
        .def("matrix_copy", &reciprocal_metric_matrix_copy);

    m.def(
        "_periodic_correlation_reciprocal_metric_producer_identity_sha256",
        &vibeqc::
            periodic_correlation_reciprocal_metric_producer_identity_sha256,
        "Internal diagnostic for the compiled reciprocal-metric producer "
        "identity.");
    m.def(
        "_periodic_correlation_reciprocal_metric_source_wire_bytes",
        &vibeqc::
            periodic_correlation_reciprocal_metric_source_wire_bytes,
        py::arg("accepted_vector_count"),
        "O(1) preflight for the exact canonical source-wire extent.");
    m.def(
        "_periodic_correlation_reciprocal_metric_payload_wire_bytes",
        &vibeqc::
            periodic_correlation_reciprocal_metric_payload_wire_bytes,
        py::arg("n_auxiliary"),
        "O(1) preflight for the exact canonical metric-payload extent.");
    m.def(
        "_plan_periodic_correlation_reciprocal_metric_enumeration",
        &vibeqc::
            plan_periodic_correlation_reciprocal_metric_enumeration,
        py::arg("reciprocal_lattice"),
        py::arg("centered_transfer_fractional"),
        py::arg("squared_membership_limit"),
        py::call_guard<py::gil_scoped_release>(),
        "O(1) internal diagnostic for the certified skew-safe integer "
        "enumeration box.");
    m.def(
        "_make_periodic_correlation_reciprocal_metric_source_manifest",
        [](const Reference& reference,
           const Schedule& schedule,
           const Config& config,
           const vibeqc::BasisSet& auxiliary_basis,
           std::uint64_t q_index,
           std::uint64_t candidate_count_cap) {
            if (candidate_count_cap
                > kReciprocalMetricSourceDiagnosticMaximumCandidates) {
                throw std::length_error(
                    "internal reciprocal metric source diagnostic is "
                    "limited to 65536 Cartesian candidates");
            }
            Config native_config = config;
            py::gil_scoped_release release;
            return vibeqc::
                make_periodic_correlation_reciprocal_metric_source_manifest(
                    reference,
                    schedule,
                    native_config,
                    auxiliary_basis,
                    q_index,
                    candidate_count_cap);
        },
        py::arg("reference"),
        py::arg("schedule"),
        py::arg("config"),
        py::arg("auxiliary_basis"),
        py::arg("q_index"),
        py::arg("candidate_count_cap"),
        "Bounded internal diagnostic for the native constant-space one-q "
        "reciprocal-metric source manifest.");
    m.def(
        "_build_periodic_correlation_reciprocal_metric",
        [](const Reference& reference,
           const Schedule& schedule,
           const Census& census,
           const Manifest& source,
           const vibeqc::BasisSet& auxiliary_basis) {
            require_reciprocal_metric_diagnostic_caps(census, source);
            py::gil_scoped_release release;
            return vibeqc::build_periodic_correlation_reciprocal_metric(
                reference, schedule, census, source, auxiliary_basis);
        },
        py::arg("reference"),
        py::arg("schedule"),
        py::arg("census"),
        py::arg("source"),
        py::arg("auxiliary_basis"),
        "Bounded internal diagnostic for one admitted raw reciprocal "
        "metric.");
}
