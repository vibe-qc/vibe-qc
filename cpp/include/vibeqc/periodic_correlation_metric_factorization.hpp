#pragma once

/// Native, one-q principal pseudoinverse square root in original auxiliary
/// AO rows. M = U diag(lambda) U^H, W = U diag(w) U^H, with
/// w = 1/sqrt(lambda) iff lambda > rank_cutoff, and zero otherwise.
/// This is the symmetric transformation of Lowdin, DOI 10.1063/1.1747632,
/// with the metric rank truncation used in Sun et al., DOI 10.1063/1.4998644.
/// No compact eigenvector gauge, k weight, spin factor, or Madelung term is
/// introduced. Negative eigenvalues below -negative_tolerance are errors;
/// negative_tolerance is explicit, positive, and no larger than rank_cutoff.
/// An empty retained subspace is rejected, never published as zero energy.
/// These classifications apply to the converged numerical eigenvalues, not
/// interval-certified exact eigenvalues. Very ill-conditioned spectra near a
/// cutoff require a threshold/spectral convergence study, like the finite
/// reciprocal cutoff itself. No numerical regularization is added to M.

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_reciprocal_metric.hpp"

namespace vibeqc {

inline constexpr std::uint32_t
    kPeriodicCorrelationMetricFactorizationContractVersion = 1U;

/// Compiled scalar-Jacobi reference backend. It requires zero dynamic
/// eigensolver/whitener workspace beyond the two matrices and eigenvalues
/// already charged by factor-build admission. Other producer workspaces are
/// not certified by this identity and must be checked by their consumers.
std::string periodic_correlation_metric_factorization_backend_identity_sha256();

struct PeriodicCorrelationMetricFactorizationDiagnostics {
    std::uint64_t n_auxiliary = 0U;
    std::uint64_t retained_rank = 0U;
    std::uint64_t sweeps = 0U;
    std::uint64_t rotations = 0U;
    double rank_cutoff = 0.0;
    double negative_tolerance = 0.0;
    double minimum_eigenvalue = 0.0;
    double maximum_eigenvalue = 0.0;
    double smallest_retained_eigenvalue = 0.0;
    /// Meaningful iff retained_rank < n_auxiliary; zero otherwise.
    double largest_discarded_eigenvalue = 0.0;
    double input_scale = 0.0;
    double scaled_initial_frobenius_norm = 0.0;
    double scaled_final_offdiagonal_norm = 0.0;
    double orthogonality_frobenius_error = 0.0;
    double maximum_self_conjugate_imaginary_residual = 0.0;
};

/// Immutable move-only numerical result. The input metric is consumed and
/// freed before W is allocated. Eigenvectors and eigenvalues are freed before
/// return. Successful production factorization retains only W and O(1)
/// diagnostics/identities in addition to the shared mean-field state.
///
/// Payload v1 wire: length-prefixed domain
/// "vibeqc.periodic.correlation.metric-factorization.payload", u32 version,
/// length-prefixed raw-payload SHA, u64 q and conjugate-q, then the diagnostics
/// in declaration order (first four u64, remaining eleven binary64), u64 W
/// element count, and row-major W real/imaginary binary64 lanes. Integers and
/// IEEE-754 bits are big-endian; signed zero is normalized. Backend identity,
/// resource plan and column-block size are execution provenance, not payload.
class PeriodicCorrelationMetricFactorizationResult {
public:
    PeriodicCorrelationMetricFactorizationResult(
        const PeriodicCorrelationMetricFactorizationResult&) = delete;
    PeriodicCorrelationMetricFactorizationResult& operator=(
        const PeriodicCorrelationMetricFactorizationResult&) = delete;
    PeriodicCorrelationMetricFactorizationResult(
        PeriodicCorrelationMetricFactorizationResult&&) noexcept = default;
    PeriodicCorrelationMetricFactorizationResult& operator=(
        PeriodicCorrelationMetricFactorizationResult&&) noexcept = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationMetricFactorizationContractVersion;
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
    state_handle() const noexcept { return state_; }
    std::uint64_t q_index() const noexcept { return q_index_; }
    std::uint64_t conjugate_q_index() const noexcept { return conjugate_q_index_; }
    bool self_conjugate_transfer() const noexcept {
        return q_index_ == conjugate_q_index_;
    }
    const PeriodicCorrelationMetricFactorizationDiagnostics& diagnostics()
        const noexcept { return diagnostics_; }
    const std::vector<std::complex<double>>& matrix_row_major()
        const noexcept { return whitener_; }
    std::string source_identity_sha256() const;
    std::string input_payload_identity_sha256() const;
    std::string payload_identity_sha256() const;
    std::string census_identity_sha256() const;
    std::string plan_identity_sha256() const;
    std::string auxiliary_basis_identity_sha256() const;
    PeriodicCorrelationByteCount admitted_factorization_peak_bytes() const noexcept {
        return admitted_factorization_peak_bytes_;
    }
    PeriodicCorrelationByteCount admitted_whitener_peak_bytes() const noexcept {
        return admitted_whitener_peak_bytes_;
    }
    PeriodicCorrelationByteCount numerical_peak_bytes() const noexcept {
        return numerical_peak_bytes_;
    }

private:
    PeriodicCorrelationMetricFactorizationResult() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::array<char, 64> source_identity_{};
    std::array<char, 64> input_payload_identity_{};
    std::array<char, 64> payload_identity_{};
    std::array<char, 64> census_identity_{};
    std::array<char, 64> plan_identity_{};
    std::array<char, 64> auxiliary_basis_identity_{};
    std::uint64_t q_index_ = 0U;
    std::uint64_t conjugate_q_index_ = 0U;
    PeriodicCorrelationByteCount admitted_factorization_peak_bytes_ = 0U;
    PeriodicCorrelationByteCount admitted_whitener_peak_bytes_ = 0U;
    PeriodicCorrelationByteCount numerical_peak_bytes_ = 0U;
    PeriodicCorrelationMetricFactorizationDiagnostics diagnostics_;
    std::vector<std::complex<double>> whitener_;

    friend PeriodicCorrelationMetricFactorizationResult
    factorize_periodic_correlation_metric(
        const PeriodicCorrelationAdmittedReference&,
        const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationFactorBuildCensus&,
        PeriodicCorrelationReciprocalMetricResult&&, double);
};

/// Validate the live admission, compiled backend, state/source/payload and
/// matrix shape before consuming raw. There is no input matrix copy. On an
/// exception after eigensolve starts, raw is consumed and must not be reused.
PeriodicCorrelationMetricFactorizationResult
factorize_periodic_correlation_metric(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    PeriodicCorrelationReciprocalMetricResult&& raw,
    double negative_tolerance);

namespace detail {
/// Tiny test seam, structurally capped at 16 auxiliary functions. The same
/// numerical kernel is used after production admission. No fabricated source
/// provenance is attached to this diagnostic result.
struct MetricPrincipalSquareRootDiagnostic {
    PeriodicCorrelationMetricFactorizationDiagnostics diagnostics;
    std::vector<std::complex<double>> whitener;
};
MetricPrincipalSquareRootDiagnostic metric_principal_square_root_diagnostic(
    std::vector<std::complex<double>> matrix, std::uint64_t dimension,
    double rank_cutoff, double negative_tolerance, bool self_conjugate);
}  // namespace detail

}  // namespace vibeqc
