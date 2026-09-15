#include "vibeqc/periodic_correlation_metric_factorization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/detail/periodic_gaussian_metric_numeric.hpp"
#include "vibeqc/hermitian_jacobi.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
constexpr std::uint64_t kMaximumSweeps = 64U;
constexpr double kRelativeTolerance = 8.0 * std::numeric_limits<double>::epsilon();
constexpr char kBackendDomain[] = "vibeqc.periodic.correlation.metric-factorization.backend";
constexpr char kBackendId[] = "vibeqc.native.scalar-hermitian-cyclic-jacobi";
constexpr char kPayloadDomain[] = "vibeqc.periodic.correlation.metric-factorization.payload";

// The wire is deliberately independent of native structs, Eigen, and BLAS.
// Strings are length-prefixed; integers/binary64 are big-endian, zero signed
// lanes are normalized. All extents are preflighted before payload traversal.
class FactorizationDigest {
public:
    FactorizationDigest(const char* domain, std::uint32_t version) {
        string(domain);
        u32(version);
    }
    void u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> bytes{};
        for (unsigned i = 0; i < 4U; ++i) bytes[i] = value >> (24U - 8U * i);
        hash_.update(bytes.data(), bytes.size());
    }
    void u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned i = 0; i < 8U; ++i) bytes[i] = value >> (56U - 8U * i);
        hash_.update(bytes.data(), bytes.size());
    }
    void real(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("metric factorization hash has a non-finite lane");
        }
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void string(const std::string& value) {
        u64(value.size());
        hash_.update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    }
    void complex(Complex value) { real(value.real()); real(value.imag()); }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
};

std::array<char, 64> identity_array(const std::string& value) {
    if (value.size() != 64U || !std::all_of(value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) {
        throw std::invalid_argument("metric factorization requires a lowercase SHA-256 identity");
    }
    std::array<char, 64> result{};
    std::copy(value.begin(), value.end(), result.begin());
    return result;
}

std::size_t checked_square(std::uint64_t n) {
    // Reserve ample constant prefix space within SHA-256's message domain.
    constexpr auto maximum = (std::numeric_limits<std::uint64_t>::max() / 8U - 4096U) / 16U;
    if (n == 0U || n > maximum / n) {
        throw std::length_error("metric factorization matrix extent is empty or overflows");
    }
    const auto square = n * n;
    if (square > std::vector<Complex>().max_size()
        || n > std::vector<double>().max_size()) {
        throw std::length_error("metric factorization allocation exceeds vector max_size");
    }
    return static_cast<std::size_t>(square);
}

void require_cutoffs(double rank, double negative) {
    if (!std::isfinite(rank) || !std::isfinite(negative)
        || rank <= 0.0 || negative <= 0.0 || negative > rank) {
        throw std::invalid_argument(
            "metric factorization requires 0 < negative_tolerance <= rank_cutoff, both finite");
    }
}

void require_matrix(const std::vector<Complex>& matrix, std::uint64_t n, bool self) {
    const auto square = checked_square(n);
    if (matrix.size() != square) {
        throw std::invalid_argument("metric factorization matrix extent is inconsistent or consumed");
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            const auto z = matrix[i * n + j];
            if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
                throw std::invalid_argument("metric factorization input is non-finite");
            }
            if (z != std::conj(matrix[j * n + i])) {
                throw std::invalid_argument("metric factorization input is not exactly Hermitian");
            }
            if (self && z.imag() != 0.0) {
                throw std::invalid_argument("self-conjugate metric factorization requires the exact real AO gauge");
            }
        }
    }
}

void neumaier(double term, double& sum, double& correction) {
    const double next = sum + term;
    correction += std::abs(sum) >= std::abs(term)
        ? (sum - next) + term : (term - next) + sum;
    sum = next;
}

detail::MetricPrincipalSquareRootDiagnostic principal_square_root(
    std::vector<Complex>&& matrix, std::uint64_t dimension, double rank_cutoff,
    double negative_tolerance, bool self_conjugate, std::uint64_t column_block) {
    require_cutoffs(rank_cutoff, negative_tolerance);
    require_matrix(matrix, dimension, self_conjugate);
    const auto n = static_cast<std::size_t>(dimension);
    const auto square = checked_square(dimension);
    if (column_block == 0U || column_block > dimension) {
        throw std::invalid_argument("metric factorization column block is outside 1..n_auxiliary");
    }
    std::vector<Complex> vectors(square);
    std::vector<double> values(n);
    HermitianJacobiOptions options;
    options.max_sweeps = kMaximumSweeps;
    options.relative_offdiagonal_tolerance = kRelativeTolerance;
    const auto solve = hermitian_jacobi_in_place(
        matrix.data(), matrix.size(), vectors.data(), vectors.size(),
        values.data(), values.size(), n, options);
    if (solve.status != HermitianJacobiStatus::Success || solve.scaling_underflow_components != 0U) {
        throw std::runtime_error("native Hermitian metric eigensolver failed; no whitener was published");
    }
    if (!std::isfinite(solve.orthogonality_frobenius_error)
        || solve.orthogonality_frobenius_error
            > 128.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(n)) {
        throw std::runtime_error("native metric eigenvectors failed the orthogonality gate");
    }
    detail::MetricPrincipalSquareRootDiagnostic result;
    auto& d = result.diagnostics;
    d.n_auxiliary = dimension;
    d.rank_cutoff = rank_cutoff;
    d.negative_tolerance = negative_tolerance;
    d.minimum_eigenvalue = values.front();
    d.maximum_eigenvalue = values.back();
    if (d.minimum_eigenvalue < -negative_tolerance) {
        throw std::runtime_error("correlation metric is indefinite beyond negative_tolerance");
    }
    const auto first = static_cast<std::size_t>(
        std::upper_bound(values.begin(), values.end(), rank_cutoff) - values.begin());
    d.retained_rank = n - first;
    if (d.retained_rank == 0U) {
        throw std::runtime_error("correlation metric retained rank is zero; refusing an empty fitting space");
    }
    d.smallest_retained_eigenvalue = values[first];
    d.largest_discarded_eigenvalue = first == 0U ? 0.0 : values[first - 1U];
    d.sweeps = solve.sweeps;
    d.rotations = solve.rotations;
    d.input_scale = solve.input_scale;
    d.scaled_initial_frobenius_norm = solve.scaled_input_frobenius_norm;
    d.scaled_final_offdiagonal_norm = solve.scaled_offdiagonal_frobenius_norm;
    d.orthogonality_frobenius_error = solve.orthogonality_frobenius_error;

    // Consume M before allocating W. No solver object retains an implicit
    // matrix/work buffer; eigenvalues are reused for reciprocal square roots.
    std::vector<Complex>().swap(matrix);
    for (std::size_t k = first; k < n; ++k) values[k] = 1.0 / std::sqrt(values[k]);
    result.whitener.resize(square);
    double maximum_magnitude = 0.0;
    for (std::size_t begin = 0; begin < n; begin += column_block) {
        const auto end = std::min(n, begin + static_cast<std::size_t>(column_block));
        for (std::size_t j = begin; j < end; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                double real = 0.0, imag = 0.0, real_c = 0.0, imag_c = 0.0;
                for (std::size_t k = first; k < n; ++k) {
                    const Complex term = (vectors[i * n + k] * values[k])
                        * std::conj(vectors[j * n + k]);
                    neumaier(term.real(), real, real_c);
                    neumaier(term.imag(), imag, imag_c);
                }
                real += real_c;
                imag += imag_c;
                if (!std::isfinite(real) || !std::isfinite(imag)) {
                    throw std::overflow_error("metric principal square root is non-finite");
                }
                if (self_conjugate) {
                    d.maximum_self_conjugate_imaginary_residual = std::max(
                        d.maximum_self_conjugate_imaginary_residual, std::abs(imag));
                }
                // Diagonal terms are real analytically; exact-real input
                // must also stay in the real gauge under the scalar solve.
                if (i == j || self_conjugate) imag = 0.0;
                const Complex value(real == 0.0 ? 0.0 : real, imag == 0.0 ? 0.0 : imag);
                maximum_magnitude = std::max(maximum_magnitude, std::abs(value));
                result.whitener[i * n + j] = value;
                result.whitener[j * n + i] = Complex(
                    value.real(), value.imag() == 0.0 ? 0.0 : -value.imag());
            }
        }
    }
    if (self_conjugate && d.maximum_self_conjugate_imaginary_residual
        > 64.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(n)
            * maximum_magnitude) {
        throw std::runtime_error("metric whitener failed the self-conjugate real-gauge gate");
    }
    return result;
}

PeriodicCorrelationByteCount phase_peak(
    const PeriodicCorrelationFactorBuildPlan& plan, PeriodicCorrelationFactorBuildPhase phase) {
    for (const auto& row : plan.phases) if (row.phase == phase) return row.peak_memory_bytes;
    throw std::logic_error("metric factorization admission omits its numerical phase");
}

void add_diagnostics(FactorizationDigest& hash,
                     const PeriodicCorrelationMetricFactorizationDiagnostics& d) {
    hash.u64(d.n_auxiliary);
    hash.u64(d.retained_rank);
    hash.u64(d.sweeps);
    hash.u64(d.rotations);
    hash.real(d.rank_cutoff);
    hash.real(d.negative_tolerance);
    hash.real(d.minimum_eigenvalue);
    hash.real(d.maximum_eigenvalue);
    hash.real(d.smallest_retained_eigenvalue);
    hash.real(d.largest_discarded_eigenvalue);
    hash.real(d.input_scale);
    hash.real(d.scaled_initial_frobenius_norm);
    hash.real(d.scaled_final_offdiagonal_norm);
    hash.real(d.orthogonality_frobenius_error);
    hash.real(d.maximum_self_conjugate_imaginary_residual);
}

}  // namespace

std::string periodic_correlation_metric_factorization_backend_identity_sha256() {
    FactorizationDigest hash(kBackendDomain, 1U);
    hash.string(kBackendId);
    hash.u64(kMaximumSweeps);
    hash.real(kRelativeTolerance);
    hash.u64(0U);  // Extra heap eigensolver workspace, beyond M/U/lambda.
    hash.u64(0U);  // Extra heap principal-whitener workspace.
    return hash.finish();
}

#define VIBEQC_METRIC_IDENTITY_ACCESSOR(name, field) \
std::string PeriodicCorrelationMetricFactorizationResult::name() const { \
    return std::string(field.begin(), field.end()); \
}
VIBEQC_METRIC_IDENTITY_ACCESSOR(source_identity_sha256, source_identity_)
VIBEQC_METRIC_IDENTITY_ACCESSOR(input_payload_identity_sha256, input_payload_identity_)
VIBEQC_METRIC_IDENTITY_ACCESSOR(payload_identity_sha256, payload_identity_)
VIBEQC_METRIC_IDENTITY_ACCESSOR(census_identity_sha256, census_identity_)
VIBEQC_METRIC_IDENTITY_ACCESSOR(plan_identity_sha256, plan_identity_)
VIBEQC_METRIC_IDENTITY_ACCESSOR(auxiliary_basis_identity_sha256, auxiliary_basis_identity_)
#undef VIBEQC_METRIC_IDENTITY_ACCESSOR

PeriodicCorrelationMetricFactorizationResult factorize_periodic_correlation_metric(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    PeriodicCorrelationReciprocalMetricResult&& raw, double negative_tolerance) {
    const auto& config = census.config();
    require_cutoffs(config.metric_absolute_eigenvalue_threshold, negative_tolerance);
    if (config.backend_identity_sha256
            != periodic_correlation_metric_factorization_backend_identity_sha256()
        || config.backend.eigensolver_workspace_bytes != 0U
        || config.backend.whitener_workspace_bytes != 0U) {
        throw std::invalid_argument("metric factorization requires the compiled scalar-Jacobi backend inventory");
    }
    PeriodicCorrelationMetricFactorizationResult result;
    {
        const auto plan = plan_periodic_correlation_factor_build(reference, schedule, census);
        if (plan.admission != PeriodicCorrelationFactorBuildAdmissionCode::Admitted) {
            throw std::runtime_error("metric factorization resource plan is not admitted");
        }
        if (!raw.state_ || raw.state_.get() != reference.state_handle().get()
            || raw.census_identity_sha256() != census.census_identity_sha256()
            || raw.plan_identity_sha256() != plan.plan_identity_sha256
            || raw.auxiliary_basis_identity_sha256() != config.auxiliary_basis_identity_sha256
            || raw.n_auxiliary_ != census.shape().n_auxiliary
            || raw.q_index_ >= census.shape().n_kpoints) {
            throw std::invalid_argument("metric factorization input does not match its admitted provenance");
        }
        const auto square = checked_square(raw.n_auxiliary_);
        const auto matrix_bytes = static_cast<std::uint64_t>(square) * sizeof(Complex);
        const auto value_bytes = raw.n_auxiliary_ * sizeof(double);
        if (plan.components.metric_matrix_bytes != matrix_bytes
            || plan.components.metric_eigenvector_bytes != matrix_bytes
            || plan.components.metric_eigenvalue_bytes != value_bytes
            || plan.components.whitener_matrix_bytes != matrix_bytes
            || raw.metric_bytes_ != matrix_bytes) {
            throw std::logic_error("metric factorization allocation differs from its admitted components");
        }
        result.admitted_factorization_peak_bytes_ = phase_peak(
            plan, PeriodicCorrelationFactorBuildPhase::MetricFactorization);
        result.admitted_whitener_peak_bytes_ = phase_peak(
            plan, PeriodicCorrelationFactorBuildPhase::MetricWhitener);
        result.numerical_peak_bytes_ = 2U * matrix_bytes + value_bytes;
        result.plan_identity_ = identity_array(plan.plan_identity_sha256);
    }
    const RegularKMesh addressing(census.mesh(), census.is_shift());
    const auto conjugate = addressing.transfer_index(
        addressing.negate(addressing.transfer_address(raw.q_index_)));
    if (raw.conjugate_q_index_ != conjugate
        || raw.self_conjugate_transfer_ != (raw.q_index_ == conjugate)) {
        throw std::invalid_argument("metric factorization transfer/conjugation provenance is inconsistent");
    }
    require_matrix(raw.matrix_row_major_, raw.n_auxiliary_, raw.self_conjugate_transfer_);
    FactorizationDigest input("vibeqc.periodic.correlation.reciprocal-metric.payload", 1U);
    input.string(raw.source_identity_sha256());
    input.u64(raw.q_index_);
    input.u64(raw.n_auxiliary_);
    input.u64(raw.matrix_row_major_.size());
    input.u64(raw.metric_bytes_);
    for (const auto z : raw.matrix_row_major_) input.complex(z);
    if (input.finish() != raw.payload_identity_sha256()) {
        throw std::invalid_argument("metric factorization raw payload identity mismatch");
    }
    result.state_ = raw.state_;
    result.source_identity_ = raw.source_identity_ascii_;
    result.input_payload_identity_ = raw.payload_identity_ascii_;
    result.census_identity_ = raw.census_identity_ascii_;
    result.auxiliary_basis_identity_ = raw.auxiliary_basis_identity_ascii_;
    result.q_index_ = raw.q_index_;
    result.conjugate_q_index_ = raw.conjugate_q_index_;
    // Move into an owning local before invoking the kernel. Exceptions after
    // this point cannot leave a mutated matrix attached to a sealed raw SHA.
    auto matrix = std::move(raw.matrix_row_major_);
    raw.state_.reset();
    auto factor = detail::metric_principal_square_root_admitted(
        std::move(matrix), raw.n_auxiliary_, config.metric_absolute_eigenvalue_threshold,
        negative_tolerance, raw.self_conjugate_transfer_, config.whitener_column_block);
    result.diagnostics_ = factor.diagnostics;
    result.whitener_ = std::move(factor.whitener);
    FactorizationDigest payload(kPayloadDomain, result.contract_version());
    payload.string(result.input_payload_identity_sha256());
    payload.u64(result.q_index_);
    payload.u64(result.conjugate_q_index_);
    add_diagnostics(payload, result.diagnostics_);
    payload.u64(result.whitener_.size());
    for (const auto z : result.whitener_) payload.complex(z);
    result.payload_identity_ = identity_array(payload.finish());
    return result;
}

namespace detail {
MetricPrincipalSquareRootDiagnostic metric_principal_square_root_admitted(
    std::vector<Complex>&& matrix, std::uint64_t dimension, double rank_cutoff,
    double negative_tolerance, bool self_conjugate, std::uint64_t column_block) {
    return principal_square_root(std::move(matrix), dimension, rank_cutoff,
                                 negative_tolerance, self_conjugate, column_block);
}

MetricPrincipalSquareRootDiagnostic metric_principal_square_root_diagnostic(
    std::vector<Complex> matrix, std::uint64_t dimension, double rank_cutoff,
    double negative_tolerance, bool self_conjugate) {
    if (dimension > 16U) {
        throw std::length_error("metric square-root diagnostic is limited to 16 auxiliary functions");
    }
    return principal_square_root(std::move(matrix), dimension, rank_cutoff,
                                 negative_tolerance, self_conjugate, dimension);
}
}  // namespace detail
}  // namespace vibeqc
