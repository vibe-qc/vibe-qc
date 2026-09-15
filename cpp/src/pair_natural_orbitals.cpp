#include "vibeqc/pair_natural_orbitals.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vibeqc {
namespace {

constexpr double kPairPNOEpsilon = std::numeric_limits<double>::epsilon();
constexpr double kDiagonalPairSymmetryTolerance = 64.0 * kPairPNOEpsilon;

std::uint64_t pair_pno_checked_product(
    std::uint64_t a, std::uint64_t b) {
    if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("pair PNO numerical extent overflows uint64");
    }
    return a * b;
}

std::uint64_t pair_pno_checked_sum(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("pair PNO numerical extent overflows uint64");
    }
    return a + b;
}

struct PairPNOCompensatedSum {
    double value = 0.0;
    double correction = 0.0;

    void add(double term) noexcept {
        const double next = value + term;
        correction += (std::abs(value) >= std::abs(term))
            ? (value - next) + term : (term - next) + value;
        value = next;
    }

    double total() const noexcept { return value + correction; }
};

void require_pair_pno_finite(double value, const char* quantity) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(
            std::string("pair PNO ") + quantity + " is not finite");
    }
}

void check_pair_pno_eigensystem(
    RestrictedPairPNOResult& result,
    const std::vector<std::complex<double>>& eigenvectors,
    const HermitianJacobiOptions& options) {
    const std::size_t n = result.domain_dimension;
    double density_scale = 0.0;
    for (double value : result.density) {
        density_scale = std::max(density_scale, std::abs(value));
    }
    // The real input never needs a complex eigenbasis. Reject a backend
    // regression rather than silently discarding a phase or imaginary part.
    for (const auto& value : eigenvectors) {
        if (!std::isfinite(value.real()) || value.imag() != 0.0) {
            throw std::runtime_error(
                "pair PNO eigensolver returned a non-finite or non-real basis");
        }
    }

    double scaled_density_norm = 0.0;
    if (density_scale > 0.0) {
        for (double value : result.density) {
            scaled_density_norm = std::hypot(
                scaled_density_norm, value / density_scale);
        }
    }
    double residual_norm = 0.0;
    double orthogonality_norm = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < n; ++b) {
            PairPNOCompensatedSum overlap;
            PairPNOCompensatedSum applied;
            for (std::size_t c = 0; c < n; ++c) {
                overlap.add(eigenvectors[c * n + a].real()
                            * eigenvectors[c * n + b].real());
                if (density_scale > 0.0) {
                    applied.add((result.density[a * n + c] / density_scale)
                                * eigenvectors[c * n + b].real());
                }
            }
            orthogonality_norm = std::hypot(
                orthogonality_norm,
                overlap.total() - (a == b ? 1.0 : 0.0));
            if (density_scale > 0.0) {
                residual_norm = std::hypot(
                    residual_norm,
                    applied.total() - eigenvectors[a * n + b].real()
                        * (result.occupations[b] / density_scale));
            }
        }
    }
    result.eigensystem_relative_residual = scaled_density_norm > 0.0
        ? residual_norm / scaled_density_norm : 0.0;
    result.eigenvector_orthogonality_error = orthogonality_norm;
    const double dimension = static_cast<double>(n);
    // These are explicit conservative numerical gates, not formal error
    // certificates or scientific occupation cutoffs. The residual is checked
    // against the original density, independently of Jacobi's internal norm.
    const double residual_gate =
        8.0 * options.relative_offdiagonal_tolerance
        + 512.0 * dimension * kPairPNOEpsilon;
    const double orthogonality_gate =
        512.0 * dimension * kPairPNOEpsilon;
    if (!std::isfinite(result.eigensystem_relative_residual)
        || result.eigensystem_relative_residual > residual_gate
        || !std::isfinite(orthogonality_norm)
        || orthogonality_norm > orthogonality_gate) {
        throw std::runtime_error(
            "pair PNO eigensystem failed its independent numerical audit");
    }
    const double negative_scaled_gate =
        256.0 * dimension * kPairPNOEpsilon * scaled_density_norm;
    for (double occupation : result.occupations) {
        if (!std::isfinite(occupation)
            || (density_scale == 0.0 && occupation != 0.0)
            || (density_scale > 0.0
                && occupation / density_scale < -negative_scaled_gate)) {
            throw std::runtime_error(
                "pair PNO density has a materially negative or invalid occupation");
        }
        if (occupation < 0.0) {
            ++result.negative_occupation_count;
        }
    }
}

}  // namespace

RestrictedPairMP2MemoryPlan plan_restricted_pair_semicanonical_mp2(
    std::uint64_t domain_dimension) {
    if (domain_dimension == 0U) {
        throw std::invalid_argument("pair MP2 domain dimension must be positive");
    }
    const auto square = pair_pno_checked_product(domain_dimension, domain_dimension);
    RestrictedPairMP2MemoryPlan plan;
    plan.domain_dimension = domain_dimension;
    plan.peak_owned_numerical_bytes = pair_pno_checked_product(square, sizeof(double));
    plan.input_bytes = pair_pno_checked_sum(plan.peak_owned_numerical_bytes,
        pair_pno_checked_product(domain_dimension, sizeof(double)));
    if (square > std::vector<double>().max_size()
        || plan.input_bytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("pair MP2 numerical extent exceeds address space");
    }
    return plan;
}

RestrictedPairMP2Result restricted_pair_semicanonical_mp2(
    const double* exchange_integrals, std::size_t integral_elements,
    const double* virtual_energies, std::size_t energy_elements,
    std::size_t domain_dimension, double occupied_fock_ii,
    double occupied_fock_jj, double denominator_floor,
    std::uint64_t numerical_byte_cap) {
    const auto plan = plan_restricted_pair_semicanonical_mp2(domain_dimension);
    if (numerical_byte_cap == 0U || numerical_byte_cap < plan.peak_owned_numerical_bytes) {
        throw std::length_error("pair MP2 numerical byte cap is insufficient");
    }
    if (!std::isfinite(occupied_fock_ii) || !std::isfinite(occupied_fock_jj)
        || !std::isfinite(denominator_floor) || denominator_floor <= 0.0) {
        throw std::invalid_argument("pair MP2 requires finite occupied Fock entries and a positive denominator floor");
    }
    const auto n = domain_dimension;
    const auto square = n * n;
    const auto valid_view = [](const double* pointer, std::size_t extent,
                               std::size_t required) {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        return pointer != nullptr && extent >= required
            && address % alignof(double) == 0U
            && required * sizeof(double) - 1U
                <= std::numeric_limits<std::uintptr_t>::max() - address;
    };
    if (!valid_view(exchange_integrals, integral_elements, square)
        || !valid_view(virtual_energies, energy_elements, n)) {
        throw std::invalid_argument("pair MP2 numerical input view is invalid");
    }
    for (std::size_t a = 0; a < n; ++a) {
        if (!std::isfinite(virtual_energies[a])) {
            throw std::invalid_argument("pair MP2 virtual energies must be finite");
        }
    }
    for (std::size_t ab = 0; ab < square; ++ab) {
        if (!std::isfinite(exchange_integrals[ab])) {
            throw std::invalid_argument("pair MP2 exchange integrals must be finite");
        }
    }
    const auto denominator = [&](std::size_t a, std::size_t b) {
        // Power-of-two scaling preserves subnormal operands; raw half-sums
        // would silently round e.g. eps=3*denorm_min before cancellation.
        const double operands[4] = {virtual_energies[a], virtual_energies[b],
                                     -occupied_fock_ii, -occupied_fock_jj};
        double largest = 0.0;
        for (double x : operands) largest = std::max(largest, std::abs(x));
        int exponent = 0;
        if (largest != 0.0) std::frexp(largest, &exponent);
        PairPNOCompensatedSum sum;
        for (double x : operands) {
            const double scaled = std::scalbn(x, -exponent);
            if (std::scalbn(scaled, exponent) != x) {
                throw std::overflow_error("pair MP2 denominator input scaling loses range");
            }
            sum.add(scaled);
        }
        const double delta = std::scalbn(sum.total(), exponent);
        if (!std::isfinite(delta) || delta <= denominator_floor) {
            throw std::invalid_argument("pair MP2 denominator must be finite and strictly exceed its floor");
        }
        return delta;
    };
    // Preflight every denominator before allocating the output. This is
    // intentionally separate from the bounded numerical solve.
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) (void) denominator(a, b);
    }
    RestrictedPairMP2Result result;
    result.domain_dimension = n;
    result.memory = plan;
    result.minimum_denominator = std::numeric_limits<double>::infinity();
    result.amplitudes.resize(square);
    PairPNOCompensatedSum energy;
    const auto amplitude = [&](std::size_t ab, double delta) {
        const double g = exchange_integrals[ab];
        const double t = -g / delta;
        if (!std::isfinite(t) || (t == 0.0 && g != 0.0)) {
            throw std::overflow_error("pair MP2 amplitude overflow or underflow");
        }
        result.amplitudes[ab] = t;
        result.maximum_absolute_amplitude = std::max(result.maximum_absolute_amplitude, std::abs(t));
        const double residual = std::abs(std::fma(delta, t, g));
        result.maximum_residual = std::max(result.maximum_residual, residual);
        result.residual_frobenius_norm = std::hypot(result.residual_frobenius_norm, residual);
    };
    const auto add_energy = [&](double gab, double gba, double delta, bool diagonal) {
        const double largest = std::max(std::abs(gab), std::abs(gba));
        if (largest == 0.0) return;
        int g_exponent = 0, delta_exponent = 0;
        std::frexp(largest, &g_exponent);
        const double d = std::frexp(delta, &delta_exponent);
        const double x = std::scalbn(gab, -g_exponent);
        const double y = std::scalbn(gba, -g_exponent);
        if (std::scalbn(x, g_exponent) != gab) ++result.energy_input_scaling_underflow_count;
        if (!diagonal && std::scalbn(y, g_exponent) != gba) ++result.energy_input_scaling_underflow_count;
        const double s = 0.5 * x + 0.5 * y;
        const double anti = 0.5 * x - 0.5 * y;
        // Include weights BEFORE restoring the exponent: a subnormal square
        // may round to zero even though six times that square is representable.
        const double scaled = diagonal ? x * x : 2.0 * s * s + 6.0 * anti * anti;
        const double term = std::scalbn(scaled / d, 2 * g_exponent - delta_exponent);
        if (!std::isfinite(term)) throw std::overflow_error("pair MP2 energy term overflow");
        if (term == 0.0) ++result.energy_term_underflow_count;
        energy.add(-term);
    };
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) {
            const double delta = denominator(a, b);
            result.minimum_denominator = std::min(result.minimum_denominator, delta);
            result.maximum_denominator = std::max(result.maximum_denominator, delta);
            amplitude(a * n + b, delta);
            if (a == b) {
                add_energy(exchange_integrals[a * n + a], exchange_integrals[a * n + a], delta, true);
            } else {
                amplitude(b * n + a, delta);
                // sum_ab (2G_ab-G_ba)T_ab = -sum_ab(S_ab^2+3A_ab^2)/Delta_ab.
                // Summing the nonpositive form avoids artificial cancellation.
                add_energy(exchange_integrals[a * n + b], exchange_integrals[b * n + a], delta, false);
            }
        }
    }
    result.ordered_pair_energy = energy.total();
    if (!std::isfinite(result.ordered_pair_energy)
        || !std::isfinite(result.residual_frobenius_norm)) {
        throw std::overflow_error("pair MP2 result is non-finite");
    }
    return result;
}

RestrictedPairPNOMemoryPlan plan_restricted_pair_pnos(
    std::uint64_t domain_dimension) {
    if (domain_dimension == 0U) {
        throw std::invalid_argument("pair PNO domain dimension must be positive");
    }
    const auto square = pair_pno_checked_product(
        domain_dimension, domain_dimension);
    const auto matrix_bytes = pair_pno_checked_product(square, sizeof(double));
    const auto eigenvalue_bytes = pair_pno_checked_product(
        domain_dimension, sizeof(double));
    RestrictedPairPNOMemoryPlan plan;
    plan.domain_dimension = domain_dimension;
    plan.input_bytes = matrix_bytes;
    plan.output_bytes_upper_bound = pair_pno_checked_sum(
        pair_pno_checked_product(2U, matrix_bytes), eigenvalue_bytes);
    plan.eigensolver_workspace_bytes = pair_pno_checked_product(
        square, 2U * sizeof(std::complex<double>));
    plan.peak_owned_numerical_bytes = pair_pno_checked_sum(
        pair_pno_checked_sum(matrix_bytes, eigenvalue_bytes),
        plan.eigensolver_workspace_bytes);
    if (plan.peak_owned_numerical_bytes
            > std::numeric_limits<std::size_t>::max()
        || plan.peak_owned_numerical_bytes
            > static_cast<std::uint64_t>(
                std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("pair PNO numerical extent exceeds address space");
    }
    return plan;
}

RestrictedPairPNOResult restricted_pair_pnos(
    const double* amplitudes,
    std::size_t amplitude_elements,
    std::size_t domain_dimension,
    RestrictedPairKind pair_kind,
    double occupation_cutoff,
    std::uint64_t numerical_byte_cap,
    const HermitianJacobiOptions& eigensolver_options) {
    const auto plan = plan_restricted_pair_pnos(domain_dimension);
    if (numerical_byte_cap == 0U
        || numerical_byte_cap < plan.peak_owned_numerical_bytes) {
        throw std::length_error(
            "pair PNO numerical byte cap is insufficient for the admitted peak");
    }
    if (pair_kind != RestrictedPairKind::Diagonal
        && pair_kind != RestrictedPairKind::OffDiagonal) {
        throw std::invalid_argument("pair PNO kind is not a restricted pair kind");
    }
    if (!std::isfinite(occupation_cutoff) || occupation_cutoff < 0.0) {
        throw std::invalid_argument("pair PNO occupation cutoff must be finite and nonnegative");
    }
    if (eigensolver_options.max_sweeps == 0U
        || !std::isfinite(eigensolver_options.relative_offdiagonal_tolerance)
        || eigensolver_options.relative_offdiagonal_tolerance <= 0.0
        || eigensolver_options.relative_offdiagonal_tolerance >= 1.0) {
        throw std::invalid_argument("pair PNO eigensolver options are invalid");
    }
    const std::size_t n = domain_dimension;
    const std::size_t square = n * n;  // checked by the count-only plan
    const auto address = reinterpret_cast<std::uintptr_t>(amplitudes);
    if (amplitudes == nullptr || amplitude_elements < square
        || address % alignof(double) != 0U
        || plan.input_bytes - 1U
            > std::numeric_limits<std::uintptr_t>::max() - address) {
        throw std::invalid_argument("pair PNO amplitude storage is invalid");
    }

    double largest_amplitude = 0.0;
    for (std::size_t index = 0; index < square; ++index) {
        if (!std::isfinite(amplitudes[index])) {
            throw std::invalid_argument("pair PNO amplitudes must be finite");
        }
        largest_amplitude = std::max(largest_amplitude, std::abs(amplitudes[index]));
    }
    int amplitude_exponent = 0;
    if (largest_amplitude > 0.0) {
        std::frexp(largest_amplitude, &amplitude_exponent);
    }
    auto scaled_amplitude = [&](std::size_t a, std::size_t b) noexcept {
        return std::scalbn(amplitudes[a * n + b], -amplitude_exponent);
    };
    double amplitude_norm = 0.0;
    double antisymmetric_norm = 0.0;
    std::uint64_t scaling_underflows = 0U;
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < n; ++b) {
            const double value = scaled_amplitude(a, b);
            amplitude_norm = std::hypot(amplitude_norm, value);
            antisymmetric_norm = std::hypot(
                antisymmetric_norm, 0.5 * value - 0.5 * scaled_amplitude(b, a));
            if (value == 0.0 && amplitudes[a * n + b] != 0.0) {
                ++scaling_underflows;
            }
        }
    }
    const double relative_antisymmetric_norm = amplitude_norm > 0.0
        ? antisymmetric_norm / amplitude_norm : 0.0;
    const bool diagonal = pair_kind == RestrictedPairKind::Diagonal;
    if (diagonal
        && relative_antisymmetric_norm > kDiagonalPairSymmetryTolerance) {
        throw std::invalid_argument(
            "diagonal pair PNO amplitudes are not symmetric within the numerical tolerance");
    }

    RestrictedPairPNOResult result;
    result.pair_kind = pair_kind;
    result.domain_dimension = n;
    result.occupation_cutoff = occupation_cutoff;
    result.memory = plan;
    result.input_relative_antisymmetric_norm = relative_antisymmetric_norm;
    result.amplitude_scaling_underflow_count = scaling_underflows;
    if (diagonal) {
        result.discarded_antisymmetric_norm = std::scalbn(
            antisymmetric_norm, amplitude_exponent);
        require_pair_pno_finite(result.discarded_antisymmetric_norm,
                                "discarded antisymmetric norm");
    }
    result.density.resize(square);
    // Riplinger and Neese, JCP 138, 034106 (2013), Eq. (23):
    // D = Ttilde*T^T + Ttilde^T*T, Ttilde=(4*T-2*T^T)/(1+delta_ij).
    // For real T=S+A, S^T=S, A^T=-A, expansion gives
    // D=(4*S*S^T+12*A*A^T)/(1+delta_ij). Evaluate this Gram sum to
    // avoid subtracting large spin terms. Accepted diagonal pairs use S;
    // their discarded roundoff antisymmetry is reported above.
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) {
            PairPNOCompensatedSum entry;
            for (std::size_t c = 0; c < n; ++c) {
                const double ac = scaled_amplitude(a, c);
                const double ca = scaled_amplitude(c, a);
                const double bc = scaled_amplitude(b, c);
                const double cb = scaled_amplitude(c, b);
                const double symmetric_ac = 0.5 * ac + 0.5 * ca;
                const double symmetric_bc = 0.5 * bc + 0.5 * cb;
                entry.add(4.0 * symmetric_ac * symmetric_bc);
                if (!diagonal) {
                    const double antisymmetric_ac = 0.5 * ac - 0.5 * ca;
                    const double antisymmetric_bc = 0.5 * bc - 0.5 * cb;
                    entry.add(12.0 * antisymmetric_ac * antisymmetric_bc);
                }
            }
            const double scaled_density = entry.total() / (diagonal ? 2.0 : 1.0);
            double value = std::scalbn(scaled_density, 2 * amplitude_exponent);
            require_pair_pno_finite(value, "density element");
            if (value == 0.0) {
                if (scaled_density != 0.0) {
                    result.density_underflow_entry_count += a == b ? 1U : 2U;
                }
                value = 0.0;
            }
            result.density[a * n + b] = value;
            result.density[b * n + a] = value;
        }
    }
    PairPNOCompensatedSum trace;
    for (std::size_t a = 0; a < n; ++a) {
        trace.add(result.density[a * n + a]);
    }
    result.density_trace = trace.total();
    require_pair_pno_finite(result.density_trace, "density trace");

    result.occupations.resize(n);
    std::vector<std::complex<double>> eigenvectors(square);
    {
        // At this point the exact live payload is density (8*n*n),
        // occupations (8*n), and two complex solver arrays (32*n*n).
        // Release the consumed matrix before allocating the PNO output.
        std::vector<std::complex<double>> matrix(square);
        for (std::size_t index = 0; index < square; ++index) {
            matrix[index] = {result.density[index], 0.0};
        }
        result.eigensolver = hermitian_jacobi_in_place(
            matrix.data(), matrix.size(), eigenvectors.data(), eigenvectors.size(),
            result.occupations.data(), result.occupations.size(), n,
            eigensolver_options);
        if (result.eigensolver.status != HermitianJacobiStatus::Success) {
            throw std::runtime_error(
                "pair PNO eigensolver failed with status "
                + std::to_string(static_cast<std::uint32_t>(result.eigensolver.status)));
        }
    }
    check_pair_pno_eigensystem(result, eigenvectors, eigensolver_options);
    result.minimum_occupation = result.occupations.front();
    std::reverse(result.occupations.begin(), result.occupations.end());
    PairPNOCompensatedSum discarded;
    for (double occupation : result.occupations) {
        if (occupation_cutoff == 0.0 || occupation > occupation_cutoff) {
            ++result.retained_dimension;
        } else {
            discarded.add(occupation);
        }
    }
    result.discarded_occupation_sum = discarded.total();
    require_pair_pno_finite(result.discarded_occupation_sum,
                            "discarded occupation sum");
    const std::size_t retained = result.retained_dimension;
    result.coefficients.resize(n * retained);
    for (std::size_t column = 0; column < retained; ++column) {
        const std::size_t source = n - 1U - column;
        std::size_t pivot = 0U;
        for (std::size_t row = 1; row < n; ++row) {
            if (std::abs(eigenvectors[row * n + source].real())
                > std::abs(eigenvectors[pivot * n + source].real())) {
                pivot = row;
            }
        }
        const double sign = eigenvectors[pivot * n + source].real() < 0.0
            ? -1.0 : 1.0;
        for (std::size_t row = 0; row < n; ++row) {
            result.coefficients[row * retained + column] =
                sign * eigenvectors[row * n + source].real();
        }
    }
    result.output_numerical_bytes = pair_pno_checked_sum(
        pair_pno_checked_sum(plan.input_bytes,
                            pair_pno_checked_product(n, sizeof(double))),
        pair_pno_checked_product(
            pair_pno_checked_product(n, retained), sizeof(double)));
    return result;
}

namespace {

void require_semicanonical_storage(const double* data, std::size_t elements,
                                  std::size_t active_elements) {
    if (active_elements == 0U) {
        return;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    const auto bytes = active_elements * sizeof(double);  // count-only plan checked
    if (data == nullptr || elements < active_elements
        || address % alignof(double) != 0U
        || bytes - 1U > std::numeric_limits<std::uintptr_t>::max() - address) {
        throw std::invalid_argument("pair semicanonical input storage is invalid");
    }
}

double semicanonical_orthogonality(const double* coefficients,
                                 std::size_t n, std::size_t r) {
    double norm = 0.0;
    for (std::size_t a = 0; a < r; ++a) {
        for (std::size_t b = 0; b < r; ++b) {
            PairPNOCompensatedSum value;
            for (std::size_t row = 0; row < n; ++row) {
                value.add(coefficients[row * r + a] * coefficients[row * r + b]);
            }
            norm = std::hypot(norm, value.total() - (a == b ? 1.0 : 0.0));
        }
    }
    return norm;
}

void semicanonical_apply_column(const double* fock, const double* coefficients,
                               double* applied, std::size_t n, std::size_t r,
                               std::size_t column, int exponent) {
    for (std::size_t row = 0; row < n; ++row) {
        PairPNOCompensatedSum value;
        for (std::size_t inner = 0; inner < n; ++inner) {
            value.add(std::scalbn(fock[row * n + inner], -exponent)
                      * coefficients[inner * r + column]);
        }
        applied[row] = value.total();
    }
}

}  // namespace

RestrictedPairSemicanonicalMemoryPlan plan_restricted_pair_semicanonicalization(
    std::uint64_t domain_dimension, std::uint64_t selected_dimension) {
    const auto n = domain_dimension;
    const auto r = selected_dimension;
    if (n == 0U || r > n) {
        throw std::invalid_argument(
            "pair semicanonical dimensions require n>0 and 0<=r<=n");
    }
    const auto fock_bytes = pair_pno_checked_product(
        pair_pno_checked_product(n, n), sizeof(double));
    const auto coefficient_bytes = pair_pno_checked_product(
        pair_pno_checked_product(n, r), sizeof(double));
    const auto projected_bytes = pair_pno_checked_product(
        pair_pno_checked_product(r, r), sizeof(double));
    const auto energy_bytes = pair_pno_checked_product(r, sizeof(double));
    const auto column_bytes = pair_pno_checked_product(n, sizeof(double));
    RestrictedPairSemicanonicalMemoryPlan plan;
    plan.domain_dimension = n;
    plan.selected_dimension = r;
    plan.input_bytes = pair_pno_checked_sum(fock_bytes, coefficient_bytes);
    plan.output_bytes = pair_pno_checked_sum(coefficient_bytes, energy_bytes);
    if (r != 0U) {
        plan.projection_phase_bytes = pair_pno_checked_sum(projected_bytes, column_bytes);
        plan.factorization_phase_bytes = pair_pno_checked_sum(
            pair_pno_checked_product(5U, projected_bytes), energy_bytes);
        plan.rotation_phase_bytes = pair_pno_checked_sum(
            plan.output_bytes, pair_pno_checked_product(2U, projected_bytes));
        plan.validation_phase_bytes = pair_pno_checked_sum(plan.output_bytes, column_bytes);
        plan.peak_owned_numerical_bytes = std::max({
            plan.projection_phase_bytes, plan.factorization_phase_bytes,
            plan.rotation_phase_bytes, plan.validation_phase_bytes});
    }
    const auto address_limit = static_cast<std::uint64_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    if (plan.input_bytes > std::numeric_limits<std::size_t>::max()
        || plan.input_bytes > address_limit
        || plan.peak_owned_numerical_bytes > std::numeric_limits<std::size_t>::max()
        || plan.peak_owned_numerical_bytes > address_limit) {
        throw std::overflow_error("pair semicanonical numerical extent exceeds address space");
    }
    return plan;
}

RestrictedPairSemicanonicalResult restricted_pair_semicanonicalize(
    const double* coefficients,
    std::size_t coefficient_elements,
    const double* virtual_fock,
    std::size_t fock_elements,
    std::size_t domain_dimension,
    std::size_t selected_dimension,
    double input_orthonormality_tolerance,
    std::uint64_t numerical_byte_cap,
    const HermitianJacobiOptions& eigensolver_options) {
    using Complex = std::complex<double>;
    const auto n = domain_dimension;
    const auto r = selected_dimension;
    const auto plan = plan_restricted_pair_semicanonicalization(n, r);
    if (numerical_byte_cap < plan.peak_owned_numerical_bytes) {
        throw std::length_error("pair semicanonical numerical byte cap is insufficient");
    }
    if (!std::isfinite(input_orthonormality_tolerance)
        || input_orthonormality_tolerance <= 0.0 || input_orthonormality_tolerance >= 1.0
        || eigensolver_options.max_sweeps == 0U
        || !std::isfinite(eigensolver_options.relative_offdiagonal_tolerance)
        || eigensolver_options.relative_offdiagonal_tolerance <= 0.0
        || eigensolver_options.relative_offdiagonal_tolerance >= 1.0) {
        throw std::invalid_argument("pair semicanonical numerical tolerances are invalid");
    }
    require_semicanonical_storage(coefficients, coefficient_elements, n * r);
    require_semicanonical_storage(virtual_fock, fock_elements, n * n);
    double largest_fock = 0.0;
    for (std::size_t element = 0; element < n * n; ++element) {
        if (!std::isfinite(virtual_fock[element])) {
            throw std::invalid_argument("pair semicanonical Fock input must be finite");
        }
        largest_fock = std::max(largest_fock, std::abs(virtual_fock[element]));
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = row + 1; column < n; ++column) {
            if (virtual_fock[row * n + column] != virtual_fock[column * n + row]) {
                throw std::invalid_argument("pair semicanonical Fock input must be exactly symmetric");
            }
        }
    }
    const double coefficient_bound = std::sqrt(1.0 + input_orthonormality_tolerance);
    for (std::size_t element = 0; element < n * r; ++element) {
        if (!std::isfinite(coefficients[element])) {
            throw std::invalid_argument("pair semicanonical coefficients must be finite");
        }
        if (std::abs(coefficients[element]) > coefficient_bound) {
            throw std::invalid_argument("pair semicanonical input basis is not orthonormal");
        }
    }
    RestrictedPairSemicanonicalResult result;
    result.domain_dimension = n;
    result.selected_dimension = r;
    result.memory = plan;
    result.input_orthonormality_error = semicanonical_orthogonality(coefficients, n, r);
    if (!std::isfinite(result.input_orthonormality_error)
        || result.input_orthonormality_error > input_orthonormality_tolerance) {
        throw std::invalid_argument("pair semicanonical input basis is not orthonormal");
    }
    if (r == 0U) {
        return result;
    }
    if (largest_fock > 0.0) {
        result.fock_scale_exponent = std::ilogb(largest_fock);
    }
    double scaled_fock_norm = 0.0;
    for (std::size_t element = 0; element < n * n; ++element) {
        const double scaled = std::scalbn(virtual_fock[element], -result.fock_scale_exponent);
        scaled_fock_norm = std::hypot(scaled_fock_norm, scaled);
        if (scaled == 0.0 && virtual_fock[element] != 0.0) {
            ++result.fock_scaling_underflow_entries;
        }
    }

    // Riplinger and Neese (2013), text after Eqs.24-25: recanonicalize the
    // retained PNOs against Fock. F_selected=C^T F C, C_final=C U, where
    // F_selected U=U eps. The full Fock operator may couple outside span(C).
    std::vector<double> projected(r * r);
    {
        std::vector<double> applied(n);
        for (std::size_t b = 0; b < r; ++b) {
            semicanonical_apply_column(virtual_fock, coefficients, applied.data(),
                                      n, r, b, result.fock_scale_exponent);
            for (std::size_t a = 0; a <= b; ++a) {
                PairPNOCompensatedSum value;
                for (std::size_t row = 0; row < n; ++row) {
                    value.add(coefficients[row * r + a] * applied[row]);
                }
                require_pair_pno_finite(value.total(), "projected Fock element");
                projected[a * r + b] = projected[b * r + a] = value.total();
            }
        }
    }
    double projected_norm = 0.0;
    for (double value : projected) {
        projected_norm = std::hypot(projected_norm, value);
    }
    result.energies.resize(r);
    std::vector<Complex> eigenvectors(r * r);
    {
        std::vector<Complex> work(r * r);
        for (std::size_t element = 0; element < r * r; ++element) {
            work[element] = {projected[element], 0.0};
        }
        result.eigensolver = hermitian_jacobi_in_place(
            work.data(), work.size(), eigenvectors.data(), eigenvectors.size(),
            result.energies.data(), result.energies.size(), r, eigensolver_options);
        result.eigensolver_performed = true;
        if (result.eigensolver.status != HermitianJacobiStatus::Success) {
            throw std::runtime_error("pair semicanonical eigensolver failed");
        }
        for (auto value : eigenvectors) {
            if (!std::isfinite(value.real()) || value.imag() != 0.0) {
                throw std::runtime_error("pair semicanonical eigensolver returned a non-real basis");
            }
        }
        double reduced_residual = 0.0;
        for (std::size_t a = 0; a < r; ++a) {
            for (std::size_t b = 0; b < r; ++b) {
                PairPNOCompensatedSum value;
                for (std::size_t inner = 0; inner < r; ++inner) {
                    value.add(projected[a * r + inner] * eigenvectors[inner * r + b].real());
                }
                reduced_residual = std::hypot(reduced_residual,
                    value.total() - eigenvectors[a * r + b].real() * result.energies[b]);
            }
        }
        result.reduced_eigensystem_relative_residual = reduced_residual
            / (projected_norm > 0.0 ? projected_norm : 1.0);
        const double reduced_gate = 8.0 * eigensolver_options.relative_offdiagonal_tolerance
            + 512.0 * static_cast<double>(r) * kPairPNOEpsilon;
        if (!std::isfinite(result.reduced_eigensystem_relative_residual)
            || result.reduced_eigensystem_relative_residual > reduced_gate) {
            throw std::runtime_error("pair semicanonical reduced eigensystem failed its numerical audit");
        }
    }
    // Release both raw matrices before allocating C_final. clear() alone
    // would retain capacity and violate the admitted phase lifetime.
    std::vector<double>().swap(projected);
    result.coefficients.resize(n * r);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t b = 0; b < r; ++b) {
            PairPNOCompensatedSum value;
            for (std::size_t a = 0; a < r; ++a) {
                value.add(coefficients[row * r + a] * eigenvectors[a * r + b].real());
            }
            result.coefficients[row * r + b] = value.total();
        }
    }
    std::vector<Complex>().swap(eigenvectors);
    for (std::size_t b = 0; b < r; ++b) {
        std::size_t pivot = 0;
        for (std::size_t row = 1; row < n; ++row) {
            if (std::abs(result.coefficients[row * r + b])
                > std::abs(result.coefficients[pivot * r + b])) {
                pivot = row;
            }
        }
        if (result.coefficients[pivot * r + b] < 0.0) {
            for (std::size_t row = 0; row < n; ++row) {
                result.coefficients[row * r + b] = -result.coefficients[row * r + b];
            }
        }
    }
    result.output_orthonormality_error = semicanonical_orthogonality(
        result.coefficients.data(), n, r);
    double projector_error = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = row; column < n; ++column) {
            PairPNOCompensatedSum difference;
            for (std::size_t orbital = 0; orbital < r; ++orbital) {
                difference.add(coefficients[row * r + orbital] * coefficients[column * r + orbital]);
                difference.add(-result.coefficients[row * r + orbital]
                               * result.coefficients[column * r + orbital]);
            }
            projector_error = std::hypot(projector_error, difference.total());
            if (row != column) {
                projector_error = std::hypot(projector_error, difference.total());
            }
        }
    }
    result.subspace_projector_frobenius_error = projector_error;
    const double geometry_gate = 512.0 * static_cast<double>(n)
        * static_cast<double>(r) * kPairPNOEpsilon;
    if (!std::isfinite(result.output_orthonormality_error)
        || result.output_orthonormality_error > result.input_orthonormality_error + geometry_gate
        || !std::isfinite(projector_error)
        || projector_error > geometry_gate * (1.0 + result.input_orthonormality_error)) {
        throw std::runtime_error("pair semicanonical rotated space failed its orthogonality or span audit");
    }
    {
        std::vector<double> applied(n);
        double projected_residual = 0.0;
        double full_residual = 0.0;
        for (std::size_t b = 0; b < r; ++b) {
            semicanonical_apply_column(virtual_fock, result.coefficients.data(),
                                      applied.data(), n, r, b, result.fock_scale_exponent);
            for (std::size_t row = 0; row < n; ++row) {
                full_residual = std::hypot(full_residual,
                    applied[row] - result.coefficients[row * r + b] * result.energies[b]);
            }
            for (std::size_t a = 0; a < r; ++a) {
                PairPNOCompensatedSum value;
                for (std::size_t row = 0; row < n; ++row) {
                    value.add(result.coefficients[row * r + a] * applied[row]);
                }
                projected_residual = std::hypot(projected_residual,
                    value.total() - (a == b ? result.energies[b] : 0.0));
            }
        }
        result.projected_fock_relative_residual = projected_residual
            / (projected_norm > 0.0 ? projected_norm : 1.0);
        result.full_space_relative_residual = full_residual
            / (scaled_fock_norm > 0.0 ? scaled_fock_norm : 1.0);
        // Audit Galerkin/selected-space diagonalization, not invariance of
        // the truncated subspace under the full-domain Fock operator. The
        // absolute roundoff term uses ||F|| when projection causes cancellation.
        const double projected_gate =
            8.0 * eigensolver_options.relative_offdiagonal_tolerance * projected_norm
            + geometry_gate * scaled_fock_norm;
        if (!std::isfinite(projected_residual) || projected_residual > projected_gate
            || !std::isfinite(result.projected_fock_relative_residual)
            || !std::isfinite(result.full_space_relative_residual)) {
            throw std::runtime_error("pair semicanonical projected Fock failed its numerical audit");
        }
    }
    for (double& energy : result.energies) {
        energy = std::scalbn(energy, result.fock_scale_exponent);
        require_pair_pno_finite(energy, "semicanonical energy");
        if (energy == 0.0) energy = 0.0;
    }
    return result;
}

}  // namespace vibeqc
