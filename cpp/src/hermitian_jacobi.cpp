#include "vibeqc/hermitian_jacobi.hpp"

#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <utility>

namespace vibeqc {
namespace {

using Complex = std::complex<double>;

static_assert(std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::radix == 2
              && std::numeric_limits<double>::digits == 53
              && std::numeric_limits<double>::max_exponent == 1024
              && std::numeric_limits<double>::min_exponent == -1021
              && std::numeric_limits<double>::has_denorm == std::denorm_present
              && std::numeric_limits<double>::round_style == std::round_to_nearest
              && sizeof(double) == 8U,
              "Hermitian Jacobi requires IEEE-754 binary64");
static_assert(sizeof(Complex) == 16U,
              "Hermitian Jacobi requires 16-byte complex128");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "Hermitian Jacobi requires size_t to fit uint64");

#if defined(__FAST_MATH__)
#error "Hermitian Jacobi forbids fast math"
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0
#error "Hermitian Jacobi requires IEEE infinities"
#endif
#if FLT_EVAL_METHOD != 0
#error "Hermitian Jacobi requires binary64 evaluation"
#endif

bool supported_floating_point_environment() noexcept {
    if (std::fegetround() != FE_TONEAREST) {
        return false;
    }
    volatile double smallest = std::numeric_limits<double>::denorm_min();
    volatile double zero = 0.0;
    volatile double one = 1.0;
    volatile double twice = smallest + smallest;
    volatile double expected_twice = std::scalbn(smallest, 1);
    volatile double identity = std::fma(smallest, one, zero);
    return smallest > 0.0 && twice == expected_twice && identity == smallest
        && std::nextafter(0.0, std::numeric_limits<double>::infinity()) == smallest;
}

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool address_range(const void* pointer, std::size_t count,
                   std::size_t element_bytes, std::size_t alignment,
                   AddressRange& range) noexcept {
    if (pointer == nullptr || count == 0
        || count > static_cast<std::size_t>(
                       std::numeric_limits<std::ptrdiff_t>::max())
                       / element_bytes) {
        return false;
    }
    const auto bytes = count * element_bytes;
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    if (start % alignment != 0
        || bytes > std::numeric_limits<std::uintptr_t>::max() - start) {
        return false;
    }
    range = {start, start + bytes};
    return true;
}

bool overlaps(const AddressRange& left, const AddressRange& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

bool finite(Complex value) noexcept {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

// Scaled sum-of-squares accumulation avoids forming squares of unscaled
// subnormal/large numbers. Only a fixed number of scalar accumulators is live.
struct FrobeniusAccumulator {
    double scale = 0.0;
    double sum = 1.0;

    void add(double value) noexcept {
        const double magnitude = std::abs(value);
        if (magnitude == 0.0) {
            return;
        }
        if (magnitude > scale) {
            const double ratio = scale / magnitude;
            sum = std::fma(sum, ratio * ratio, 1.0);
            scale = magnitude;
        } else {
            const double ratio = magnitude / scale;
            sum = std::fma(ratio, ratio, sum);
        }
    }

    double norm() const noexcept {
        return scale == 0.0 ? 0.0 : scale * std::sqrt(sum);
    }
};

struct CompensatedSum {
    double sum = 0.0;
    double correction = 0.0;

    void add(double value) noexcept {
        const double next = sum + value;
        correction += std::abs(sum) >= std::abs(value)
            ? (sum - next) + value : (value - next) + sum;
        sum = next;
    }

    double value() const noexcept {
        return sum + correction;
    }
};

double matrix_norm(const Complex* matrix, std::size_t n,
                   bool offdiagonal_only) noexcept {
    FrobeniusAccumulator accumulator;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            if (offdiagonal_only && row == column) {
                continue;
            }
            const auto value = matrix[row * n + column];
            if (!finite(value)) {
                return std::numeric_limits<double>::infinity();
            }
            accumulator.add(value.real());
            accumulator.add(value.imag());
        }
    }
    return accumulator.norm();
}

double orthogonality_error(const Complex* vectors, std::size_t n) noexcept {
    FrobeniusAccumulator accumulator;
    for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t q = p; q < n; ++q) {
            CompensatedSum real;
            CompensatedSum imag;
            for (std::size_t row = 0; row < n; ++row) {
                const auto left = vectors[row * n + p];
                const auto right = vectors[row * n + q];
                real.add(left.real() * right.real());
                real.add(left.imag() * right.imag());
                imag.add(left.real() * right.imag());
                imag.add(-left.imag() * right.real());
            }
            const double real_error = real.value() - (p == q ? 1.0 : 0.0);
            const double imag_error = imag.value();
            if (!std::isfinite(real_error) || !std::isfinite(imag_error)) {
                return std::numeric_limits<double>::infinity();
            }
            accumulator.add(real_error);
            accumulator.add(imag_error);
            if (p != q) {
                accumulator.add(real_error);
                accumulator.add(imag_error);
            }
        }
    }
    return accumulator.norm();
}

// Apply the right rotation J=[[c,-s],[conj(s),c]] to one row pair.
// Scalar FMA expressions avoid complex matrix products and their workspaces.
void rotate_pair(Complex& x, Complex& y, double c, Complex s) noexcept {
    const double xr = x.real();
    const double xi = x.imag();
    const double yr = y.real();
    const double yi = y.imag();
    const double sr = s.real();
    const double si = s.imag();
    x = {std::fma(c, xr, std::fma(sr, yr, si * yi)),
         std::fma(c, xi, std::fma(sr, yi, -si * yr))};
    y = {std::fma(c, yr, std::fma(-sr, xr, si * xi)),
         std::fma(c, yi, std::fma(-sr, xi, -si * xr))};
}

void sort_eigensystem(Complex* matrix, Complex* vectors, double* values,
                      std::size_t n) noexcept {
    // Stable insertion sort uses no index vector, temporary column, or heap.
    // The same permutation acts on both sides of the residual work matrix.
    for (std::size_t current = 1; current < n; ++current) {
        std::size_t q = current;
        while (q > 0 && values[q] < values[q - 1]) {
            const auto p = q - 1;
            std::swap(values[p], values[q]);
            for (std::size_t row = 0; row < n; ++row) {
                std::swap(vectors[row * n + p], vectors[row * n + q]);
                std::swap(matrix[row * n + p], matrix[row * n + q]);
            }
            for (std::size_t column = 0; column < n; ++column) {
                std::swap(matrix[p * n + column], matrix[q * n + column]);
            }
            --q;
        }
    }
}

}  // namespace

HermitianJacobiResult hermitian_jacobi_in_place(
    Complex* matrix,
    std::size_t matrix_elements,
    Complex* eigenvectors,
    std::size_t eigenvector_elements,
    double* eigenvalues,
    std::size_t eigenvalue_elements,
    std::size_t dimension,
    const HermitianJacobiOptions& options) noexcept {
    HermitianJacobiResult result;
    const auto n = dimension;
    if (!supported_floating_point_environment()
        || n == 0 || n > std::numeric_limits<std::size_t>::max() / n
        || options.max_sweeps == 0
        || !std::isfinite(options.relative_offdiagonal_tolerance)
        || options.relative_offdiagonal_tolerance <= 0.0
        || options.relative_offdiagonal_tolerance >= 1.0) {
        return result;
    }
    const auto square = n * n;
    const auto pairs = n * (n - 1) / 2;
    if (matrix_elements < square || eigenvector_elements < square
        || eigenvalue_elements < n
        || square > std::numeric_limits<std::uint64_t>::max() / 2
        || (pairs != 0
            && options.max_sweeps
                > std::numeric_limits<std::uint64_t>::max() / pairs)) {
        return result;
    }
    AddressRange matrix_range;
    AddressRange vectors_range;
    AddressRange values_range;
    if (!address_range(matrix, square, sizeof(Complex), alignof(Complex),
                       matrix_range)
        || !address_range(eigenvectors, square, sizeof(Complex),
                          alignof(Complex), vectors_range)
        || !address_range(eigenvalues, n, sizeof(double), alignof(double),
                          values_range)
        || overlaps(matrix_range, vectors_range)
        || overlaps(matrix_range, values_range)
        || overlaps(vectors_range, values_range)) {
        return result;
    }

    double largest_component = 0.0;
    for (std::size_t element = 0; element < square; ++element) {
        const auto value = matrix[element];
        if (!finite(value)) {
            result.status = HermitianJacobiStatus::NonFiniteInput;
            return result;
        }
        largest_component = std::fmax(
            largest_component, std::fmax(std::abs(value.real()),
                                         std::abs(value.imag())));
    }
    for (std::size_t row = 0; row < n; ++row) {
        if (matrix[row * n + row].imag() != 0.0) {
            result.status = HermitianJacobiStatus::NotHermitian;
            return result;
        }
        for (std::size_t column = row + 1; column < n; ++column) {
            if (matrix[row * n + column]
                != std::conj(matrix[column * n + row])) {
                result.status = HermitianJacobiStatus::NotHermitian;
                return result;
            }
        }
    }

    if (largest_component != 0.0) {
        result.input_scale_exponent = std::ilogb(largest_component);
        result.input_scale = std::scalbn(1.0, result.input_scale_exponent);
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            const auto index = row * n + column;
            const auto value = matrix[index];
            const double real = std::scalbn(
                value.real(), -result.input_scale_exponent);
            const double imag = std::scalbn(
                value.imag(), -result.input_scale_exponent);
            result.scaling_underflow_components +=
                (value.real() != 0.0 && real == 0.0) ? 1U : 0U;
            result.scaling_underflow_components +=
                (value.imag() != 0.0 && imag == 0.0) ? 1U : 0U;
            matrix[index] = {real == 0.0 ? 0.0 : real,
                             imag == 0.0 ? 0.0 : imag};
            eigenvectors[index] = {row == column ? 1.0 : 0.0, 0.0};
        }
    }
    result.scaled_input_frobenius_norm = matrix_norm(matrix, n, false);
    result.scaled_offdiagonal_frobenius_norm = matrix_norm(matrix, n, true);
    const double target = options.relative_offdiagonal_tolerance
        * result.scaled_input_frobenius_norm;

    while (result.scaled_offdiagonal_frobenius_norm > target
           && result.sweeps < options.max_sweeps) {
        for (std::size_t p = 0; p < n - 1; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const auto pivot = matrix[p * n + q];
                const double magnitude = std::hypot(pivot.real(), pivot.imag());
                if (magnitude == 0.0) {
                    continue;
                }
                const double a = matrix[p * n + p].real();
                const double d = matrix[q * n + q].real();
                const double pivot_scale = std::fmax(
                    std::abs(pivot.real()), std::abs(pivot.imag()));
                // Normalize the phase independently: |b| may be subnormal,
                // and b/hypot(b.real,b.imag) then need not have unit norm.
                const double phase_real = pivot.real() / pivot_scale;
                const double phase_imag = pivot.imag() / pivot_scale;
                const double phase_norm = std::hypot(phase_real, phase_imag);
                const Complex phase(phase_real / phase_norm,
                                    phase_imag / phase_norm);
                // The globally scaled diagonal difference cannot overflow.
                // Local scaling before halving also preserves the angle when
                // a, d and b all happen to be subnormal in the global scale.
                const double difference = a - d;
                const double rotation_scale = std::fmax(
                    std::abs(difference), pivot_scale);
                const double scaled_magnitude = std::hypot(
                    pivot.real() / rotation_scale,
                    pivot.imag() / rotation_scale);
                if (scaled_magnitude == 0.0) {
                    continue;
                }
                const double delta = 0.5 * (difference / rotation_scale);
                const double absolute_delta = std::abs(delta);

                // Hahn (2007), Eqs. 10-17, Hermitian specialization:
                // t = sign(delta)*|b|/(|delta|+hypot(delta,|b|));
                // c = 1/sqrt(1+t*t), s=c*t*b/|b|,
                // J=[[c,-s],[conj(s),c]], M'=J^H M J, U'=U J.
                // Ratio branches avoid both overflow in the denominator and
                // loss of a tiny pivot through squaring. At delta=0 choose
                // t=+1 deterministically. No source code is imported.
                double tangent;
                if (absolute_delta >= scaled_magnitude) {
                    const double ratio = scaled_magnitude / absolute_delta;
                    tangent = ratio / (1.0 + std::hypot(1.0, ratio));
                } else {
                    const double ratio = absolute_delta / scaled_magnitude;
                    tangent = 1.0 / (ratio + std::hypot(1.0, ratio));
                }
                if (delta < 0.0) {
                    tangent = -tangent;
                }
                const double cosine = 1.0 / std::hypot(1.0, tangent);
                const auto sine = phase * (tangent * cosine);
                if (sine == Complex(0.0, 0.0)) {
                    continue;
                }
                for (std::size_t row = 0; row < n; ++row) {
                    if (row != p && row != q) {
                        auto x = matrix[row * n + p];
                        auto y = matrix[row * n + q];
                        rotate_pair(x, y, cosine, sine);
                        matrix[row * n + p] = x;
                        matrix[p * n + row] = std::conj(x);
                        matrix[row * n + q] = y;
                        matrix[q * n + row] = std::conj(y);
                    }
                    rotate_pair(eigenvectors[row * n + p],
                                eigenvectors[row * n + q], cosine, sine);
                }
                matrix[p * n + p] = {std::fma(tangent, magnitude, a), 0.0};
                matrix[q * n + q] = {std::fma(-tangent, magnitude, d), 0.0};
                matrix[p * n + q] = {0.0, 0.0};
                matrix[q * n + p] = {0.0, 0.0};
                ++result.rotations;
            }
        }
        ++result.sweeps;
        result.scaled_offdiagonal_frobenius_norm = matrix_norm(matrix, n, true);
        if (!std::isfinite(result.scaled_offdiagonal_frobenius_norm)) {
            result.status = HermitianJacobiStatus::NumericalFailure;
            return result;
        }
    }

    result.orthogonality_frobenius_error = orthogonality_error(eigenvectors, n);
    if (!std::isfinite(result.orthogonality_frobenius_error)) {
        result.status = HermitianJacobiStatus::NumericalFailure;
        return result;
    }
    if (result.scaled_offdiagonal_frobenius_norm > target) {
        result.status = HermitianJacobiStatus::NoConvergence;
        return result;
    }
    for (std::size_t index = 0; index < n; ++index) {
        const double value = std::scalbn(matrix[index * n + index].real(),
                                        result.input_scale_exponent);
        if (!std::isfinite(value)) {
            result.status = HermitianJacobiStatus::NumericalFailure;
            return result;
        }
        eigenvalues[index] = value == 0.0 ? 0.0 : value;
    }
    sort_eigensystem(matrix, eigenvectors, eigenvalues, n);
    result.status = HermitianJacobiStatus::Success;
    return result;
}

}  // namespace vibeqc
