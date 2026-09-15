#pragma once

// PRIVATE numerical leaves. Source-origin wrappers own authentication and
// admission. These helpers neither certify a Hamiltonian nor accept a state.
#include "vibeqc/detail/periodic_reciprocal_source.hpp"
#include "vibeqc/periodic_correlation_metric_factorization.hpp"

namespace vibeqc {
namespace detail {

using MetricReciprocalVisitor = std::uint64_t (*)(
    PeriodicReciprocalRecordCallback, void* callback_user, const void* source_user);
struct ReciprocalMetricNumericalResult {
    std::vector<std::complex<double>> matrix;
    std::uint64_t completed_panel_count = 0;
    std::uint64_t maximum_fourier_panel_bytes = 0;
    double maximum_self_conjugate_imaginary_residual = 0.0;
};
ReciprocalMetricNumericalResult contract_reciprocal_metric_numeric(
    const BasisSet& auxiliary_basis, std::uint64_t n_auxiliary,
    std::uint64_t accepted_count, std::uint64_t panel_capacity,
    bool self_conjugate, std::uint64_t numerical_byte_cap,
    MetricReciprocalVisitor visitor, const void* source_user);

// Same kernel as the existing v1 factorizer and tiny diagnostic, but no
// diagnostic dimension cap. The authenticating wrapper must admit memory,
// shapes, work and scientific policies before consuming its sealed input.
MetricPrincipalSquareRootDiagnostic metric_principal_square_root_admitted(
    std::vector<std::complex<double>>&& matrix, std::uint64_t dimension,
    double rank_cutoff, double negative_tolerance, bool self_conjugate,
    std::uint64_t column_block);

} // namespace detail
} // namespace vibeqc
