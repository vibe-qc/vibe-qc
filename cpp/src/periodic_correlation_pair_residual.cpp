#include "vibeqc/periodic_correlation_pair_residual.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vibeqc {
namespace {

static_assert(sizeof(double) == 8U && std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits == 53,
              "Pair residual requires IEEE binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Pair residual forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Pair residual requires binary64 evaluation"
#endif

std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("Pair residual count, work or byte product overflows uint64");
    }
    return a * b;
}
std::uint64_t sum(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("Pair residual count, work or byte sum overflows uint64");
    }
    return a + b;
}

void require_finite(double value, const char* message) {
    if (!std::isfinite(value)) throw std::overflow_error(message);
}

void compensated_add(double term, double& value, double& correction) {
    const double next = value + term;
    correction += std::abs(value) >= std::abs(term)
        ? (value - next) + term : (term - next) + value;
    value = next;
    require_finite(value, "Pair residual compensated sum overflowed");
    require_finite(correction, "Pair residual compensation overflowed");
}

struct ScalarSum {
    double value = 0.0, correction = 0.0;
    void add(double term) { compensated_add(term, value, correction); }
    double total() const {
        const double result = value + correction;
        require_finite(result, "Pair residual contraction is non-finite");
        return result;
    }
};

struct AddressRange { std::uintptr_t begin = 0, end = 0; };

AddressRange require_storage(const double* pointer, std::size_t extent,
                             std::uint64_t required) {
    if (extent < required) throw std::invalid_argument("Pair residual view has insufficient accessible extent");
    if (required == 0U) return {};
    const auto bytes = product(required, sizeof(double));
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || address % alignof(double) != 0U
        || bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
        throw std::invalid_argument("Pair residual view has an invalid pointer, alignment or address range");
    }
    return {address, address + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(AddressRange a, AddressRange b) {
    return a.begin < b.end && b.begin < a.end;
}

std::uint64_t source_work(std::uint64_t n, std::uint64_t m) {
    if (m == 0U) return 0U;
    const auto nn = product(n, n);
    return sum(sum(product(n, product(m, m)), product(nn, m)), nn);
}

std::uint64_t source_input_bytes(std::uint64_t n, std::uint64_t m) {
    return product(sizeof(double), sum(product(m, m), product(n, m)));
}

double checked_product(double a, double b, PairResidualDiagnostics& diagnostics) {
    const double result = a * b;
    require_finite(result, "Pair residual scalar product overflowed");
    if (result == 0.0 && a != 0.0 && b != 0.0) {
        diagnostics.product_underflow_count = sum(diagnostics.product_underflow_count, 1U);
    }
    return result;
}

double denominator(double ea, double eb, double fii, double fjj, double floor) {
    // Same scalar convention as the native semicanonical MP2 initializer:
    // frexp power-of-two scale, Neumaier sum, then rescale. Half sums would
    // discard subnormal inputs before a physically important cancellation.
    const std::array<double, 4> operands{ea, eb, -fii, -fjj};
    double largest = 0.0;
    for (const double value : operands) largest = std::max(largest, std::abs(value));
    int exponent = 0;
    if (largest != 0.0) std::frexp(largest, &exponent);
    ScalarSum total;
    for (const double value : operands) {
        const double scaled = std::scalbn(value, -exponent);
        if (std::scalbn(scaled, exponent) != value) {
            throw std::overflow_error("Pair residual denominator input scaling loses range");
        }
        total.add(scaled);
    }
    const double delta = std::scalbn(total.total(), exponent);
    if (!std::isfinite(delta) || delta <= floor) {
        throw std::invalid_argument("Pair residual denominator must be finite and strictly exceed its floor");
    }
    return delta;
}

void require_environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0.0) || std::fma(tiny, one, zero) != tiny) {
        throw std::invalid_argument("Pair residual requires round-to-nearest and gradual underflow");
    }
}

}  // namespace

PairResidualMemoryPlan plan_pair_residual(
    std::uint64_t n, std::uint64_t m, std::uint64_t couplings) {
    if (n == 0U) throw std::invalid_argument("Pair residual target dimension must be positive");
    PairResidualMemoryPlan plan;
    plan.target_dimension = n;
    plan.maximum_source_dimension = m;
    plan.expected_coupling_count = couplings;
    const auto nn = product(n, n), nm = product(n, m), mm = product(m, m);
    plan.borrowed_target_input_bytes = sum(product(16U, nn), product(8U, n));
    plan.maximum_borrowed_source_input_bytes = source_input_bytes(n, m);
    plan.residual_bytes = product(8U, nn);
    plan.compensation_bytes = plan.residual_bytes;
    plan.projection_workspace_bytes = product(8U, nm);
    plan.peak_owned_numerical_bytes = sum(product(2U, plan.residual_bytes), plan.projection_workspace_bytes);
    plan.output_numerical_bytes = plan.residual_bytes;
    plan.maximum_scalar_products_per_source = source_work(n, m);
    plan.maximum_total_scalar_products = product(plan.maximum_scalar_products_per_source, couplings);
    const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (nn > std::vector<double>().max_size() || nm > std::vector<double>().max_size()
        || mm > std::vector<double>().max_size() || plan.borrowed_target_input_bytes > maximum
        || plan.maximum_borrowed_source_input_bytes > maximum
        || plan.projection_workspace_bytes > maximum) {
        throw std::length_error("Pair residual inputs or workspace exceed native vector/address extents");
    }
    return plan;
}

const double* PairResidualResult::residual_data() const {
    if (memory_.target_dimension == 0U
        || residual_.size() != memory_.target_dimension * memory_.target_dimension) {
        throw std::logic_error("Pair residual data requested from a consumed result");
    }
    return residual_.data();
}

double PairResidualResult::residual(std::size_t row, std::size_t column) const {
    const auto* data = residual_data();
    if (row >= target_dimension() || column >= target_dimension()) {
        throw std::out_of_range("Pair residual matrix index out of range");
    }
    return data[row * target_dimension() + column];
}

bool PairResidualAccumulator::is_open() const noexcept {
    return status_ == Status::Open && memory_.target_dimension != 0U
        && residual_.size() == memory_.target_dimension * memory_.target_dimension
        && compensation_.size() == residual_.size()
        && workspace_.size() == memory_.target_dimension * memory_.maximum_source_dimension;
}

void PairResidualAccumulator::abort() noexcept {
    status_ = Status::Aborted;
    std::vector<double>().swap(residual_);
    std::vector<double>().swap(compensation_);
    std::vector<double>().swap(workspace_);
}

PairResidualAccumulator initialize_pair_residual(
    const double* g, std::size_t g_elements, const double* t, std::size_t t_elements,
    const double* eps, std::size_t energy_elements, std::size_t n, double fii,
    double fjj, const PairResidualControls& controls, std::uint64_t cap) {
    const auto plan = plan_pair_residual(n, controls.maximum_source_dimension,
                                         controls.expected_coupling_count);
    if (cap == 0U || cap < plan.peak_owned_numerical_bytes) {
        throw std::length_error("Pair residual numerical byte cap is missing or exceeded");
    }
    if (controls.occupied_label_count == 0U
        || controls.target_first >= controls.occupied_label_count
        || controls.target_second >= controls.occupied_label_count
        || controls.expected_coupling_count > product(2U, controls.occupied_label_count - 1U)) {
        throw std::invalid_argument("Pair residual occupied slots or coupling count are invalid");
    }
    if (!std::isfinite(fii) || !std::isfinite(fjj)
        || !std::isfinite(controls.denominator_floor) || controls.denominator_floor <= 0.0
        || (controls.target_first == controls.target_second && fii != fjj)) {
        throw std::invalid_argument("Pair residual occupied diagonals or positive denominator floor are invalid");
    }
    require_environment();
    const auto nn = n * n;
    (void) require_storage(g, g_elements, nn);
    (void) require_storage(t, t_elements, nn);
    (void) require_storage(eps, energy_elements, n);
    for (std::size_t index = 0; index < nn; ++index) {
        if (!std::isfinite(g[index]) || !std::isfinite(t[index])) {
            throw std::invalid_argument("Pair residual target G/T must be finite");
        }
    }
    for (std::size_t a = 0; a < n; ++a) {
        if (!std::isfinite(eps[a])) throw std::invalid_argument("Pair residual virtual energies must be finite");
    }
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) (void) denominator(eps[a], eps[b], fii, fjj, controls.denominator_floor);
    }
    PairResidualAccumulator result;
    result.memory_ = plan;
    result.controls_ = controls;
    result.diagnostics_.minimum_denominator = std::numeric_limits<double>::infinity();
    try {
        result.residual_.resize(nn);
        result.compensation_.resize(nn);
        result.workspace_.resize(static_cast<std::size_t>(plan.projection_workspace_bytes / sizeof(double)));
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = a; b < n; ++b) {
                const double delta = denominator(eps[a], eps[b], fii, fjj, controls.denominator_floor);
                result.diagnostics_.minimum_denominator = std::min(result.diagnostics_.minimum_denominator, delta);
                result.diagnostics_.maximum_denominator = std::max(result.diagnostics_.maximum_denominator, delta);
                const auto fill = [&](std::size_t index) {
                    const double value = std::fma(delta, t[index], g[index]);
                    require_finite(value, "Pair residual initialization overflowed");
                    result.residual_[index] = value;
                    if (t[index] != 0.0 && std::fma(delta, t[index], 0.0) == 0.0) {
                        result.diagnostics_.product_underflow_count = sum(result.diagnostics_.product_underflow_count, 1U);
                    }
                };
                fill(a * n + b);
                if (a != b) fill(b * n + a);
            }
        }
        result.status_ = PairResidualAccumulator::Status::Open;
        return result;
    } catch (...) {
        result.abort();
        throw;
    }
}

void PairResidualAccumulator::accumulate(const PairResidualSource& source) {
    try {
        if (!is_open()) throw std::logic_error("Pair residual accumulator is moved, aborted or finished");
        require_environment();
        const auto leg = static_cast<std::uint32_t>(source.leg);
        const auto k = source.substituted_occupied;
        if (leg > 1U || k >= controls_.occupied_label_count
            || source.stored_first >= controls_.occupied_label_count
            || source.stored_second >= controls_.occupied_label_count
            || diagnostics_.accepted_coupling_count >= controls_.expected_coupling_count) {
            throw std::invalid_argument("Pair residual coupling slot, source pair or count is invalid");
        }
        const auto replaced = source.leg == PairResidualOccupiedLeg::First
            ? controls_.target_first : controls_.target_second;
        if (k == replaced) throw std::invalid_argument("Pair residual diagonal occupied term is already in the target denominator");
        if (has_previous_slot_ && (leg < static_cast<std::uint32_t>(previous_leg_)
            || (leg == static_cast<std::uint32_t>(previous_leg_) && k <= previous_occupied_))) {
            throw std::invalid_argument("Pair residual coupling slots must be strictly ascending without duplicates");
        }
        const auto first = source.leg == PairResidualOccupiedLeg::First ? k : controls_.target_first;
        const auto second = source.leg == PairResidualOccupiedLeg::First ? controls_.target_second : k;
        const bool direct = source.stored_first == first && source.stored_second == second;
        const bool reversed = source.stored_first == second && source.stored_second == first;
        if (!direct && !reversed) throw std::invalid_argument("Pair residual source occupied labels do not match the arithmetic coupling slot");
        const bool transpose = !direct && reversed;
        const auto n = static_cast<std::size_t>(memory_.target_dimension);
        const auto m = source.source_dimension;
        if (m > controls_.maximum_source_dimension) throw std::length_error("Pair residual source rank exceeds its admitted bound");
        const auto work = source_work(n, m);
        const auto next_work = sum(diagnostics_.scalar_product_count, work);
        if (next_work > controls_.maximum_scalar_products) {
            throw std::length_error("Pair residual scalar-product work budget exceeded");
        }
        if (!std::isfinite(source.occupied_fock_coupling)) throw std::invalid_argument("Pair residual occupied coupling must be finite");
        const auto mm = product(m, m), nm = product(n, m);
        const auto trange = require_storage(source.amplitudes, source.amplitude_elements, mm);
        const auto orange = require_storage(source.target_source_overlap, source.overlap_elements, nm);
        const auto rr = require_storage(residual_.data(), residual_.size(), n * n);
        const auto cr = require_storage(compensation_.data(), compensation_.size(), n * n);
        const auto wr = require_storage(workspace_.data(), workspace_.size(), product(n, controls_.maximum_source_dimension));
        for (const auto input : {trange, orange}) {
            for (const auto owned : {rr, cr, wr}) {
                if (overlaps(input, owned)) throw std::invalid_argument("Pair residual source aliases a mutable owned buffer");
            }
        }
        for (std::size_t index = 0; index < mm; ++index) {
            if (!std::isfinite(source.amplitudes[index])) throw std::invalid_argument("Pair residual source amplitudes must be finite");
        }
        for (std::size_t index = 0; index < nm; ++index) {
            if (!std::isfinite(source.target_source_overlap[index])) throw std::invalid_argument("Pair residual source overlap must be finite");
        }
        if (m != 0U) {
            // O is ALWAYS target-by-source. Stored pair reversal transposes
            // amplitudes, not O; no transpose matrix is allocated.
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t d = 0; d < m; ++d) {
                    ScalarSum value;
                    for (std::size_t c = 0; c < m; ++c) {
                        const double t = source.amplitudes[transpose ? d * m + c : c * m + d];
                        value.add(checked_product(source.target_source_overlap[a * m + c], t, diagnostics_));
                    }
                    workspace_[a * m + d] = value.total();
                }
            }
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = 0; b < n; ++b) {
                    ScalarSum value;
                    for (std::size_t d = 0; d < m; ++d) {
                        value.add(checked_product(workspace_[a * m + d], source.target_source_overlap[b * m + d], diagnostics_));
                    }
                    const double term = checked_product(-source.occupied_fock_coupling, value.total(), diagnostics_);
                    compensated_add(term, residual_[a * n + b], compensation_[a * n + b]);
                }
            }
        }
        diagnostics_.scalar_product_count = next_work;
        diagnostics_.accepted_coupling_count = sum(diagnostics_.accepted_coupling_count, 1U);
        diagnostics_.transposed_source_count += transpose;
        diagnostics_.zero_rank_source_count += m == 0U;
        diagnostics_.maximum_observed_source_dimension = std::max<std::uint64_t>(diagnostics_.maximum_observed_source_dimension, m);
        diagnostics_.maximum_observed_source_input_bytes = std::max(diagnostics_.maximum_observed_source_input_bytes, source_input_bytes(n, m));
        has_previous_slot_ = true;
        previous_leg_ = source.leg;
        previous_occupied_ = k;
    } catch (...) {
        abort();
        throw;
    }
}

PairResidualResult PairResidualAccumulator::finish() {
    try {
        if (!is_open()) throw std::logic_error("Pair residual accumulator is moved, aborted or finished");
        require_environment();
        if (diagnostics_.accepted_coupling_count != controls_.expected_coupling_count) {
            throw std::runtime_error("Pair residual cannot finish with an incomplete coupling stream");
        }
        for (std::size_t index = 0; index < residual_.size(); ++index) {
            residual_[index] += compensation_[index];
            require_finite(residual_[index], "Pair residual final compensated value overflowed");
            diagnostics_.maximum_absolute_residual = std::max(diagnostics_.maximum_absolute_residual, std::abs(residual_[index]));
            diagnostics_.residual_frobenius_norm = std::hypot(diagnostics_.residual_frobenius_norm, residual_[index]);
        }
        require_finite(diagnostics_.residual_frobenius_norm, "Pair residual final norm overflowed");
        std::vector<double>().swap(workspace_);
        std::vector<double>().swap(compensation_);
        PairResidualResult result;
        result.memory_ = memory_;
        result.controls_ = controls_;
        result.diagnostics_ = diagnostics_;
        result.residual_ = std::move(residual_);
        status_ = Status::Finished;
        return result;
    } catch (...) {
        abort();
        throw;
    }
}

}  // namespace vibeqc
