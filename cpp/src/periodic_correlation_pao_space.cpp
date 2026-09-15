#include "vibeqc/periodic_correlation_pao_space.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace {
using Complex = std::complex<double>;
using Options = PeriodicCorrelationPAOSpaceOptions;
using Diagnostics = PeriodicCorrelationPAOSpaceDiagnostics;
static_assert(sizeof(Complex) == 16U && sizeof(double) == 8U
              && std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits == 53,
              "PAO space requires IEEE binary64 and complex128");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "PAO space forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "PAO space requires binary64 evaluation"
#endif

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("PAO space byte or count product overflow");
    }
    return a * b;
}
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("PAO space byte sum overflow");
    }
    return a + b;
}
bool finite(Complex value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}
Complex scaled(Complex value, int exponent) {
    return Complex(std::scalbn(value.real(), exponent), std::scalbn(value.imag(), exponent));
}
double magnitude(Complex value) {
    const auto result = std::abs(value);
    if (!finite(value) || !std::isfinite(result)) {
        throw std::overflow_error("PAO space contraction or diagnostic is non-finite");
    }
    return result;
}

struct ComplexSum {
    double real = 0.0, imag = 0.0, rc = 0.0, ic = 0.0;
    static void lane(double term, double& value, double& correction) {
        const double next = value + term;
        correction += std::abs(value) >= std::abs(term)
            ? (value - next) + term : (term - next) + value;
        value = next;
        if (!std::isfinite(value) || !std::isfinite(correction)) {
            throw std::overflow_error("PAO space compensated sum overflow");
        }
    }
    void add(Complex value) {
        lane(value.real(), real, rc); lane(value.imag(), imag, ic);
    }
    Complex result() const {
        const Complex value(real + rc, imag + ic);
        if (!finite(value)) throw std::overflow_error("PAO space sum is non-finite");
        return value;
    }
};

struct Frobenius {
    double scale = 0.0, squares = 1.0;
    void add(double value) {
        value = std::abs(value);
        if (!std::isfinite(value)) throw std::overflow_error("PAO space norm is non-finite");
        if (value == 0.0) return;
        if (scale < value) {
            const double ratio = scale / value;
            squares = 1.0 + squares * ratio * ratio;
            scale = value;
        } else {
            const double ratio = value / scale;
            squares += ratio * ratio;
        }
    }
    void add(Complex value) { add(value.real()); add(value.imag()); }
    double norm() const {
        const double result = scale == 0.0 ? 0.0 : scale * std::sqrt(squares);
        if (!std::isfinite(result)) throw std::overflow_error("PAO space norm overflow");
        return result;
    }
};

double safe_relative(double numerator, double denominator) {
    const double answer = numerator / (denominator == 0.0 ? 1.0 : denominator);
    if (!std::isfinite(answer)) throw std::overflow_error("PAO space relative residual overflow");
    return answer;
}

void gate(Complex left, Complex right, const Options& options, const char* message) {
    const double error = magnitude(left - right);
    const double bound = options.validation_absolute_tolerance
        + options.validation_relative_tolerance * std::max(magnitude(left), magnitude(right));
    if (!std::isfinite(bound) || error > bound) throw std::runtime_error(message);
}

void require_options(const Options& options) {
    const std::array<double, 2> absolutes{options.rank_absolute_cutoff,
                                        options.negative_absolute_tolerance};
    for (double value : absolutes) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("PAO space absolute rank/negative controls must be finite nonnegative");
        }
    }
    const std::array<double, 4> fractions{options.rank_relative_cutoff,
        options.negative_relative_tolerance, options.validation_absolute_tolerance,
        options.validation_relative_tolerance};
    for (double value : fractions) {
        if (!std::isfinite(value) || value < 0.0 || value >= 1.0) {
            throw std::invalid_argument("PAO space relative/validation controls must be finite in [0,1)");
        }
    }
    if ((options.rank_absolute_cutoff == 0.0 && options.rank_relative_cutoff == 0.0)
        || (options.negative_absolute_tolerance == 0.0 && options.negative_relative_tolerance == 0.0)
        || (options.validation_absolute_tolerance == 0.0 && options.validation_relative_tolerance == 0.0)) {
        throw std::invalid_argument("PAO space rank, negative and validation pairs need a positive control");
    }
    if (options.eigensolver.max_sweeps == 0U
        || !std::isfinite(options.eigensolver.relative_offdiagonal_tolerance)
        || options.eigensolver.relative_offdiagonal_tolerance <= 0.0
        || options.eigensolver.relative_offdiagonal_tolerance >= 1.0) {
        throw std::invalid_argument("PAO space eigensolver controls must be explicit and positive");
    }
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0.0) || std::fma(tiny, one, zero) != tiny) {
        throw std::invalid_argument("PAO space requires round-to-nearest and gradual underflow");
    }
}

void require_inputs(const PeriodicCorrelationAdmittedReference& reference,
                    const PeriodicCorrelationPAODomain& domain) {
    if (!reference.state_handle() || !domain.state_handle()) {
        throw std::invalid_argument("PAO space requires live admitted reference and domain owners");
    }
    const auto& state = reference.state();
    const auto& dims = reference.dimensions();
    const auto& plan = reference.plan();
    if (reference.contract_version() != kPeriodicCorrelationAdmittedReferenceContractVersion
        || domain.contract_version() != kPeriodicCorrelationPAODomainContractVersion
        || state.contract_version() != kPeriodicRestrictedMeanFieldStateContractVersion
        // The node inventory charges this immutable state allocation once.
        // Equal content in a different owner would be an uncharged copy.
        || domain.state_handle().get() != reference.state_handle().get()
        || domain.state().state_identity_sha256() != state.state_identity_sha256()
        || domain.allocation_identity() != dims.allocation_identity
        || domain.state().calculation_identity() != dims.calculation_identity
        || plan.allocation_identity != dims.allocation_identity
        || plan.calculation_identity != dims.calculation_identity
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus) {
        throw std::invalid_argument("PAO space reference/domain identity or admission mismatch");
    }
    const auto expected = plan_periodic_correlation_pao_domain(
        state.mesh(), state.n_basis(), domain.domain_dimension());
    if (domain.memory().retained_matrix_bytes != expected.retained_matrix_bytes
        || domain.memory().retained_domain_index_bytes != expected.retained_domain_index_bytes
        || domain.memory().matrix_element_count != expected.matrix_element_count) {
        throw std::logic_error("PAO space received inconsistent domain inventory");
    }
}

std::uint64_t admit(const PeriodicCorrelationAdmittedReference& reference,
                    const PeriodicCorrelationPAOSpaceMemoryPlan& memory,
                    std::uint64_t cap) {
    if (memory.peak_owned_numerical_bytes > cap) {
        throw std::length_error("PAO space numerical byte cap is missing or exceeded");
    }
    const auto& dims = reference.dimensions();
    const auto& budget = reference.budget();
    const auto required = add(add(dims.external_bytes, dims.shared_bytes),
        add(mul(budget.mpi_ranks, add(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank),
                add(memory.borrowed_domain_bytes, memory.peak_owned_numerical_bytes))));
    if (budget.memory_limit_bytes == 0U || required > budget.memory_limit_bytes) {
        throw std::length_error("PAO space and live borrowed domain exceed admitted node memory");
    }
    return required;
}

int audit_matrix(const Complex* matrix, std::size_t n) {
    double maximum = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            const auto value = matrix[row * n + col];
            if (!finite(value) || value != std::conj(matrix[col * n + row])) {
                throw std::invalid_argument("PAO space requires finite exactly Hermitian domain matrices");
            }
            maximum = std::max({maximum, std::abs(value.real()), std::abs(value.imag())});
        }
    }
    return maximum == 0.0 ? 0 : std::ilogb(maximum);
}

void audit_eigensystem(const Complex* original, const Complex* u, const double* eigenvalues,
                      std::size_t n, int scale_exponent, const Options& options,
                      double& relative_residual, double& orthogonality) {
    Frobenius residual, matrix_norm, gram_error;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            matrix_norm.add(scaled(original[row * n + col], -scale_exponent));
            ComplexSum product, gram;
            for (std::size_t k = 0; k < n; ++k) {
                product.add(scaled(original[row * n + k], -scale_exponent) * u[k * n + col]);
                gram.add(std::conj(u[k * n + row]) * u[k * n + col]);
            }
            residual.add(product.result() - u[row * n + col] * std::scalbn(eigenvalues[col], -scale_exponent));
            gram_error.add(gram.result() - (row == col ? 1.0 : 0.0));
        }
    }
    relative_residual = safe_relative(residual.norm(), matrix_norm.norm());
    orthogonality = gram_error.norm();
    const double tolerance = options.validation_absolute_tolerance + options.validation_relative_tolerance;
    if (relative_residual > tolerance || orthogonality > tolerance) {
        throw std::runtime_error("PAO space independent eigensystem residual or orthogonality gate failed");
    }
}

void apply_column(const Complex* matrix, int scale_exponent, const Complex* c,
                  std::size_t n, std::size_t r, std::size_t column, Complex* target) {
    for (std::size_t row = 0; row < n; ++row) {
        ComplexSum value;
        for (std::size_t k = 0; k < n; ++k) {
            value.add(scaled(matrix[row * n + k], -scale_exponent) * c[k * r + column]);
        }
        target[row] = value.result();
    }
}

Complex dot_column(const Complex* c, const Complex* column, std::size_t n,
                   std::size_t r, std::size_t index) {
    ComplexSum value;
    for (std::size_t k = 0; k < n; ++k) value.add(std::conj(c[k * r + index]) * column[k]);
    return value.result();
}

double audit_metric(const Complex* s, const Complex* c, std::size_t n, std::size_t r,
                    Complex* column, const Options& options) {
    Frobenius error;
    for (std::size_t b = 0; b < r; ++b) {
        apply_column(s, 0, c, n, r, b, column);
        for (std::size_t a = 0; a < r; ++a) {
            const auto value = dot_column(c, column, n, r, a);
            const Complex target(a == b ? 1.0 : 0.0, 0.0);
            gate(value, target, options, "PAO space C^H S C metric gate failed");
            error.add(value - target);
        }
    }
    return error.norm();
}

class Digest {
public:
    explicit Digest(const char* domain) { string(domain); u32(kPeriodicCorrelationPAOSpaceContractVersion); }
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
        if (!std::isfinite(value)) throw std::overflow_error("PAO space digest is non-finite");
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); u64(bits);
    }
    void complex(Complex value) { real(value.real()); real(value.imag()); }
    void string(const std::string& value) {
        u64(value.size());
        hash_.update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
};

}  // namespace

PeriodicCorrelationPAOSpaceMemoryPlan plan_periodic_correlation_pao_space(
    std::uint64_t n, std::uint64_t r) {
    if (r > n) throw std::invalid_argument("PAO space retained rank exceeds domain dimension");
    PeriodicCorrelationPAOSpaceMemoryPlan p;
    p.domain_dimension = n; p.retained_dimension = r;
    const auto nn = mul(n, n), nr = mul(n, r), rr = mul(r, r);
    p.borrowed_domain_index_bytes = mul(16U, n);
    p.borrowed_domain_matrix_bytes = mul(32U, nn);
    p.borrowed_domain_bytes = add(p.borrowed_domain_index_bytes, p.borrowed_domain_matrix_bytes);
    p.overlap_factorization_phase_bytes = add(mul(32U, nn), mul(8U, n));
    if (r != 0U) {
        p.compact_orthogonalizer_phase_bytes = add(add(mul(16U, nn), mul(16U, nr)), mul(8U, n));
        p.projected_fock_phase_bytes = add(add(mul(16U, nr), mul(16U, rr)), mul(24U, n));
        p.fock_factorization_phase_bytes = add(add(mul(16U, nr), mul(48U, rr)), add(mul(8U, n), mul(8U, r)));
        p.rotation_phase_bytes = add(add(mul(32U, nr), mul(16U, rr)), add(mul(8U, n), mul(8U, r)));
        p.validation_phase_bytes = add(mul(32U, nr), add(mul(24U, n), mul(8U, r)));
    }
    p.output_numerical_bytes = add(mul(16U, nr), add(mul(8U, n), mul(8U, r)));
    p.peak_owned_numerical_bytes = std::max({p.overlap_factorization_phase_bytes,
        p.compact_orthogonalizer_phase_bytes, p.projected_fock_phase_bytes,
        p.fock_factorization_phase_bytes, p.rotation_phase_bytes, p.validation_phase_bytes});
    constexpr auto hash_limit = std::numeric_limits<std::uint64_t>::max() / 8U - 4096U;
    const auto address_limit = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (p.output_numerical_bytes > hash_limit || p.borrowed_domain_bytes > hash_limit) {
        throw std::overflow_error("PAO space exceeds SHA-256 bit-length bounds");
    }
    if (nn > std::vector<Complex>().max_size() || nr > std::vector<Complex>().max_size()
        || n > std::vector<double>().max_size() || p.borrowed_domain_matrix_bytes > address_limit
        || p.output_numerical_bytes > address_limit) {
        throw std::length_error("PAO space exceeds native vector/address extents");
    }
    return p;
}

const Complex* PeriodicCorrelationPAOSpace::coefficients_data() const {
    if (!state_ || coefficients_.size() != memory_.domain_dimension * memory_.retained_dimension) {
        throw std::logic_error("PAO space coefficients requested from a consumed owner");
    }
    return coefficients_.data();
}
const double* PeriodicCorrelationPAOSpace::energies_data() const {
    if (!state_ || energies_.size() != memory_.retained_dimension) {
        throw std::logic_error("PAO space energies requested from a consumed owner");
    }
    return energies_.data();
}
const double* PeriodicCorrelationPAOSpace::overlap_eigenvalues_data() const {
    if (!state_ || overlap_eigenvalues_.size() != memory_.domain_dimension) {
        throw std::logic_error("PAO space overlap eigenvalues requested from a consumed owner");
    }
    return overlap_eigenvalues_.data();
}
Complex PeriodicCorrelationPAOSpace::coefficient(std::size_t row, std::size_t orbital) const {
    const auto* data = coefficients_data();
    if (row >= domain_dimension() || orbital >= retained_dimension()) throw std::out_of_range("PAO space coefficient index out of range");
    return data[row * retained_dimension() + orbital];
}
double PeriodicCorrelationPAOSpace::energy(std::size_t orbital) const {
    const auto* data = energies_data();
    if (orbital >= retained_dimension()) throw std::out_of_range("PAO space energy index out of range");
    return data[orbital];
}
double PeriodicCorrelationPAOSpace::overlap_eigenvalue(std::size_t index) const {
    const auto* data = overlap_eigenvalues_data();
    if (index >= domain_dimension()) throw std::out_of_range("PAO space overlap index out of range");
    return data[index];
}

PeriodicCorrelationPAOSpace make_periodic_correlation_pao_space(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationPAODomain& domain, std::uint64_t cap, const Options& options) {
    require_inputs(reference, domain);
    require_options(options);
    const auto n = static_cast<std::size_t>(domain.domain_dimension());
    auto memory = plan_periodic_correlation_pao_space(n, 0U);
    Diagnostics diagnostics;
    diagnostics.required_node_memory_bytes = admit(reference, memory, cap);
    PeriodicCorrelationPAOSpace result;
    result.state_ = reference.state_handle();
    result.state_digest_ = reference.state().state_identity_sha256();
    result.domain_index_digest_ = domain.domain_index_sha256();
    result.domain_digest_ = domain.pao_domain_identity_sha256();
    result.allocation_identity_ = reference.dimensions().allocation_identity;
    result.options_ = options;
    if (n != 0U) {
        // Getter guards consumed owners before the raw immutable views are used.
        (void) domain.overlap(n - 1U, n - 1U); (void) domain.fock(n - 1U, n - 1U);
        const auto* s = domain.overlap_data();
        const auto* f = domain.fock_data();
        const int s_exponent = audit_matrix(s, n);
        const int f_exponent = audit_matrix(f, n);
        diagnostics.fock_scale_exponent = f_exponent;
        for (std::size_t index = 0; index < n * n; ++index) {
            const auto value = scaled(f[index], -f_exponent);
            diagnostics.fock_scaling_underflow_components += f[index].real() != 0.0 && value.real() == 0.0;
            diagnostics.fock_scaling_underflow_components += f[index].imag() != 0.0 && value.imag() == 0.0;
        }
        result.overlap_eigenvalues_.resize(n);
        std::vector<Complex> u(n * n);
        {
            std::vector<Complex> work(s, s + n * n);
            diagnostics.overlap_eigensolver = hermitian_jacobi_in_place(work.data(), work.size(),
                u.data(), u.size(), result.overlap_eigenvalues_.data(), n, n, options.eigensolver);
            if (diagnostics.overlap_eigensolver.status != HermitianJacobiStatus::Success) {
                throw std::runtime_error("PAO space overlap eigensolver failed to converge or represent its result");
            }
        }
        audit_eigensystem(s, u.data(), result.overlap_eigenvalues_.data(), n, s_exponent, options,
            diagnostics.overlap_eigensystem_relative_residual,
            diagnostics.overlap_eigenvector_orthogonality_error);
        const auto* lambda = result.overlap_eigenvalues_.data();
        diagnostics.minimum_overlap_eigenvalue = lambda[0];
        diagnostics.maximum_overlap_eigenvalue = lambda[n - 1U];
        const double maximum_abs = std::max(std::abs(lambda[0]), std::abs(lambda[n - 1U]));
        diagnostics.effective_rank_cutoff = std::max(options.rank_absolute_cutoff,
            options.rank_relative_cutoff * std::max(lambda[n - 1U], 0.0));
        diagnostics.effective_negative_tolerance = options.negative_absolute_tolerance
            + options.negative_relative_tolerance * maximum_abs;
        if (!std::isfinite(diagnostics.effective_negative_tolerance)
            || !std::isfinite(diagnostics.effective_rank_cutoff)) {
            throw std::overflow_error("PAO space effective overlap thresholds overflowed");
        }
        if (lambda[0] < -diagnostics.effective_negative_tolerance) {
            throw std::runtime_error("PAO space overlap is materially indefinite");
        }
        std::size_t first = n;
        for (std::size_t index = 0; index < n; ++index) {
            diagnostics.negative_overlap_eigenvalue_count += lambda[index] < 0.0;
            if (first == n && lambda[index] > diagnostics.effective_rank_cutoff) first = index;
        }
        const auto r = n - first;
        if (r == 0U) throw std::runtime_error("PAO space nonempty domain has zero retained virtual rank");
        memory = plan_periodic_correlation_pao_space(n, r);
        diagnostics.required_node_memory_bytes = admit(reference, memory, cap);
        std::vector<Complex> x(n * r);
        for (std::size_t b = 0; b < r; ++b) {
            const double factor = 1.0 / std::sqrt(lambda[first + b]);
            if (!std::isfinite(factor)) throw std::overflow_error("PAO space inverse overlap square root overflow");
            for (std::size_t a = 0; a < n; ++a) {
                x[a * r + b] = u[a * n + first + b] * factor;
                if (!finite(x[a * r + b])) throw std::overflow_error("PAO space orthogonalizer is non-finite");
            }
        }
        std::vector<Complex>().swap(u);
        std::vector<Complex> h(r * r);
        {
            std::vector<Complex> column(n);
            diagnostics.canonical_metric_frobenius_residual = audit_metric(s, x.data(), n, r, column.data(), options);
            // Populate both triangles independently before auditing/correcting.
            for (std::size_t b = 0; b < r; ++b) {
                apply_column(f, f_exponent, x.data(), n, r, b, column.data());
                for (std::size_t a = 0; a < r; ++a) h[a * r + b] = dot_column(x.data(), column.data(), n, r, a);
            }
        }
        for (std::size_t a = 0; a < r; ++a) {
            for (std::size_t b = a; b < r; ++b) {
                const auto ab = h[a * r + b], ba = h[b * r + a];
                const auto defect = magnitude(ab - std::conj(ba));
                diagnostics.maximum_projected_fock_hermitian_defect = std::max(
                    diagnostics.maximum_projected_fock_hermitian_defect, defect);
                gate(ab, std::conj(ba), options, "PAO space projected Fock Hermitian audit failed");
                const auto value = a == b ? Complex(ab.real(), 0.0) : ab * 0.5 + std::conj(ba) * 0.5;
                diagnostics.maximum_projected_fock_hermitization_correction = std::max({
                    diagnostics.maximum_projected_fock_hermitization_correction,
                    magnitude(value - ab), magnitude(std::conj(value) - ba)});
                h[a * r + b] = value; h[b * r + a] = std::conj(value);
            }
        }
        result.energies_.resize(r);
        {
            std::vector<Complex> v(r * r);
            {
                std::vector<Complex> work(h.begin(), h.end());
                diagnostics.fock_eigensolver = hermitian_jacobi_in_place(work.data(), work.size(),
                    v.data(), v.size(), result.energies_.data(), r, r, options.eigensolver);
                if (diagnostics.fock_eigensolver.status != HermitianJacobiStatus::Success) {
                    throw std::runtime_error("PAO space Fock eigensolver failed to converge or represent its result");
                }
            }
            const int h_exponent = audit_matrix(h.data(), r);
            audit_eigensystem(h.data(), v.data(), result.energies_.data(), r, h_exponent, options,
                diagnostics.fock_eigensystem_relative_residual,
                diagnostics.fock_eigenvector_orthogonality_error);
            std::vector<Complex>().swap(h);
            result.coefficients_.resize(n * r);
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = 0; b < r; ++b) {
                    ComplexSum value;
                    for (std::size_t k = 0; k < r; ++k) value.add(x[a * r + k] * v[k * r + b]);
                    result.coefficients_[a * r + b] = value.result();
                }
            }
        }
        // Deterministic complex phase: largest component is nonnegative real.
        // Normalize locally so a complex subnormal pivot still has unit phase.
        for (std::size_t b = 0; b < r; ++b) {
            std::size_t pivot = 0;
            double largest = magnitude(result.coefficients_[b]);
            for (std::size_t a = 1; a < n; ++a) {
                const double value = magnitude(result.coefficients_[a * r + b]);
                if (value > largest) { largest = value; pivot = a; }
            }
            if (largest == 0.0) throw std::runtime_error("PAO space rotation produced a zero orbital");
            const auto value = result.coefficients_[pivot * r + b];
            const double component = std::max(std::abs(value.real()), std::abs(value.imag()));
            const Complex reduced(value.real() / component, value.imag() / component);
            const Complex phase = std::conj(reduced) / std::abs(reduced);
            for (std::size_t a = 0; a < n; ++a) result.coefficients_[a * r + b] *= phase;
        }
        {
            std::vector<Complex> column(n);
            const auto* c = result.coefficients_.data();
            diagnostics.final_metric_frobenius_residual = audit_metric(s, c, n, r, column.data(), options);
            Frobenius fock_error, fock_target, projector_error;
            for (std::size_t b = 0; b < r; ++b) {
                apply_column(f, f_exponent, c, n, r, b, column.data());
                fock_target.add(result.energies_[b]);
                for (std::size_t a = 0; a < r; ++a) {
                    const auto value = dot_column(c, column.data(), n, r, a);
                    const Complex target(a == b ? result.energies_[b] : 0.0, 0.0);
                    gate(value, target, options, "PAO space C^H F C semicanonical gate failed");
                    fock_error.add(value - target);
                }
            }
            diagnostics.final_projected_fock_relative_residual = safe_relative(fock_error.norm(), fock_target.norm());
            // C C^H S has the metric on its RIGHT. The target U_r U_r^H is
            // reconstructed from the unrotated X and original retained lambda.
            // Only r of the existing n scratch entries are needed here.
            for (std::size_t b = 0; b < n; ++b) {
                for (std::size_t k = 0; k < r; ++k) {
                    ComplexSum value;
                    for (std::size_t nu = 0; nu < n; ++nu) value.add(std::conj(c[nu * r + k]) * s[nu * n + b]);
                    column[k] = value.result();
                }
                for (std::size_t a = 0; a < n; ++a) {
                    ComplexSum actual, target;
                    for (std::size_t k = 0; k < r; ++k) {
                        actual.add(c[a * r + k] * column[k]);
                        target.add((x[a * r + k] * lambda[first + k]) * std::conj(x[b * r + k]));
                    }
                    gate(actual.result(), target.result(), options, "PAO space retained-projector preservation gate failed");
                    projector_error.add(actual.result() - target.result());
                }
            }
            diagnostics.retained_projector_relative_residual = safe_relative(projector_error.norm(), std::sqrt(static_cast<double>(r)));
        }
        for (auto& energy : result.energies_) {
            const double value = std::scalbn(energy, f_exponent);
            if (!std::isfinite(value)) throw std::overflow_error("PAO space semicanonical energy is not representable");
            diagnostics.energy_rescaling_underflow_count += energy != 0.0 && value == 0.0;
            energy = value;
        }
    }
    result.memory_ = memory;
    result.diagnostics_ = diagnostics;
    // Fixed big-endian wire, signed zeros canonicalized. The payload is n,r,
    // row-major complex C, ascending eps, then all ascending overlap lambdas.
    Digest payload("vibeqc.periodic.correlation.pao.space.payload");
    payload.u64(memory.domain_dimension); payload.u64(memory.retained_dimension);
    for (const auto value : result.coefficients_) payload.complex(value);
    for (const auto value : result.energies_) payload.real(value);
    for (const auto value : result.overlap_eigenvalues_) payload.real(value);
    result.payload_digest_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.pao.space.identity");
    identity.string(result.state_digest_); identity.string(result.domain_index_digest_);
    identity.string(result.domain_digest_); identity.string(result.allocation_identity_);
    identity.string(result.payload_digest_);
    identity.string("canonical-X=U-lambda^-1/2;strict-rank;C=XV;complex;projector=CCHS");
    identity.real(options.rank_absolute_cutoff); identity.real(options.rank_relative_cutoff);
    identity.real(options.negative_absolute_tolerance); identity.real(options.negative_relative_tolerance);
    identity.real(options.validation_absolute_tolerance); identity.real(options.validation_relative_tolerance);
    identity.u64(options.eigensolver.max_sweeps);
    identity.real(options.eigensolver.relative_offdiagonal_tolerance);
    result.identity_digest_ = identity.finish();
    return result;
}

}  // namespace vibeqc
