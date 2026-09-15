#include "vibeqc/bounded_restricted_ccsd_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using View = BoundedRestrictedCCSDRealView;
static_assert(sizeof(double) == 8, "bounded CCSD solver requires binary64");
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("bounded CCSD solver count addition overflows");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("bounded CCSD solver count multiplication overflows");
    return a * b;
}
double finite(double value) {
    if (!std::isfinite(value))
        throw std::overflow_error("bounded CCSD solver arithmetic is not finite");
    return value;
}
struct Sum {
    double value = 0.0, correction = 0.0;
    void include(double term) {
        finite(term);
        const double next = finite(value + term);
        correction = finite(correction + (std::abs(value) >= std::abs(term)
            ? (value - next) + term : (term - next) + value));
        value = next;
    }
    double total() const { return finite(value + correction); }
};
void view_extent(const View& view, U count) {
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    const U bytes = mul(8, count);
    if (!view.data || view.element_count < count || address % alignof(double)
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("bounded CCSD solver borrowed input view is invalid");
}
void scan(const View& view, U count) {
    for (U i = 0; i < count; ++i)
        if (!std::isfinite(view.data[i]))
            throw std::invalid_argument("bounded CCSD solver inputs must be finite");
}
void symmetric(double left, double right, double tolerance) {
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    if (std::abs(left / scale - right / scale) > tolerance)
        throw std::invalid_argument("bounded CCSD solver input symmetry audit failed");
}
double gap(double ea, double eb, double ei, double ej, double floor) {
    // Common power-of-two scaling avoids premature overflow/half-subnormal
    // loss. Reject any operand whose scaling cannot round-trip exactly.
    const double values[4] = {ea, eb, -ei, -ej};
    double largest = 0.0;
    for (double x : values) largest = std::max(largest, std::abs(x));
    int exponent = 0;
    if (largest != 0.0) std::frexp(largest, &exponent);
    Sum sum;
    for (double x : values) {
        const double scaled = std::scalbn(x, -exponent);
        if (std::scalbn(scaled, exponent) != x)
            throw std::overflow_error("bounded CCSD solver denominator scaling loses range");
        sum.include(scaled);
    }
    const double result = std::scalbn(sum.total(), exponent);
    if (!std::isfinite(result) || result <= floor)
        throw std::invalid_argument("bounded CCSD solver denominator must be finite and strictly exceed its floor");
    return result;
}
struct IntegralCounter {
    const BoundedRestrictedCCSDIntegralProvider* provider;
    U calls = 0, limit = 0;
    static double value(U p, U q, U r, U s, void* context) {
        auto& c = *static_cast<IntegralCounter*>(context);
        if (c.calls >= c.limit)
            throw std::length_error("bounded CCSD solver exhausted its integral call cap");
        ++c.calls;
        return finite(c.provider->value(p, q, r, s, c.provider->context));
    }
};
}  // namespace

BoundedRestrictedCCSDSolverMemoryPlan plan_bounded_restricted_ccsd_solver(
    U o, U v, U iterations, bool supplied,
    const BoundedRestrictedCCSDSolverInventory& inventory, U retained, U transient) {
    if (!iterations || !inventory.numerical_replicas)
        throw std::invalid_argument("bounded CCSD solver iterations and numerical replicas must be positive");
    BoundedRestrictedCCSDSolverMemoryPlan p;
    p.target = plan_bounded_restricted_ccsd_target(o, v, retained, transient);
    p.n_occupied = o; p.n_virtual = v; p.maximum_iterations = iterations;
    p.supplied_initial_amplitudes = supplied;
    const U ov = mul(o, v), o2 = mul(o, o), v2 = mul(v, v), t2 = mul(o2, v2);
    const U amplitudes = add(ov, t2);
    p.amplitude_snapshot_bytes = mul(8, amplitudes);
    p.candidate_snapshot_bytes = p.amplitude_snapshot_bytes;
    p.borrowed_fock_bytes = mul(8, add(add(o2, v2), ov));
    p.borrowed_initial_amplitude_bytes = supplied ? p.amplitude_snapshot_bytes : 0;
    p.peak_owned_numerical_bytes = add(mul(2, p.amplitude_snapshot_bytes), p.target.peak_owned_numerical_bytes);
    p.total_live_numerical_bytes = add(p.peak_owned_numerical_bytes,
        add(add(p.borrowed_fock_bytes, p.borrowed_initial_amplitude_bytes), add(retained, transient)));
    p.external_node_numerical_bytes = inventory.external_node_numerical_bytes;
    p.numerical_replicas = inventory.numerical_replicas;
    p.required_node_numerical_bytes = add(p.external_node_numerical_bytes,
        mul(p.numerical_replicas, p.total_live_numerical_bytes));
    p.target_evaluations_upper_bound = mul(iterations, o2);
    p.integral_calls_upper_bound = mul(iterations,
        add(mul(o2, p.target.integral_calls_upper_bound), mul(2, t2)));
    // Leaf work includes scanning its entire borrowed T1/T2 on EVERY target.
    // Initial 128-unit allowance covers input scans, symmetry/gap preflights,
    // zero/copy of both snapshots. Per-snapshot 256 units/amplitude cover
    // compensated energy, gap recomputation, norms, updates and scalar events.
    const U initial = mul(128, add(add(amplitudes, p.borrowed_fock_bytes / 8),
        add(p.borrowed_initial_amplitude_bytes / 8, 1)));
    const U per_snapshot = add(mul(o2, p.target.kernel_work_units_upper_bound),
        mul(256, add(add(amplitudes, o2), 1)));
    p.kernel_work_units_upper_bound = add(initial, mul(iterations, per_snapshot));
    const U limit = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max());
    if (p.peak_owned_numerical_bytes > limit || p.borrowed_fock_bytes > limit
        || p.amplitude_snapshot_bytes / 8 > std::vector<double>().max_size())
        throw std::overflow_error("bounded CCSD solver exceeds native numerical address extent");
    return p;
}

BoundedRestrictedCCSDSolverResult bounded_restricted_ccsd_solve(
    const BoundedRestrictedCCSDSolverInput& in, const BoundedRestrictedCCSDIntegralProvider& provider,
    const BoundedRestrictedCCSDSolverOptions& options, const BoundedRestrictedCCSDSolverInventory& inventory,
    const BoundedRestrictedCCSDSolverCaps& caps,
    BoundedRestrictedCCSDSolverProgressCallback progress, void* progress_context) {
    const bool supplied = in.initial_t1.data || in.initial_t1.element_count
        || in.initial_t2.data || in.initial_t2.element_count;
    const auto p = plan_bounded_restricted_ccsd_solver(in.n_occupied, in.n_virtual,
        options.maximum_iterations, supplied, inventory,
        provider.retained_numerical_bytes, provider.maximum_transient_numerical_bytes);
    if (!caps.maximum_owned_numerical_bytes || caps.maximum_owned_numerical_bytes < p.peak_owned_numerical_bytes
        || !caps.maximum_total_numerical_bytes || caps.maximum_total_numerical_bytes < p.total_live_numerical_bytes
        || !caps.maximum_node_numerical_bytes || caps.maximum_node_numerical_bytes < p.required_node_numerical_bytes)
        throw std::length_error("bounded CCSD solver numerical byte cap is missing or exceeded");
    if (!caps.maximum_integral_calls || caps.maximum_integral_calls < p.integral_calls_upper_bound
        || !caps.maximum_kernel_work_units || caps.maximum_kernel_work_units < p.kernel_work_units_upper_bound)
        throw std::length_error("bounded CCSD solver call or work cap is missing or exceeded");
    for (const double x : {options.denominator_floor, options.singles_residual_tolerance,
                           options.doubles_residual_tolerance, options.energy_tolerance})
        if (!std::isfinite(x) || x <= 0.0)
            throw std::invalid_argument("bounded CCSD solver convergence tolerances and denominator floor must be positive finite");
    if (!std::isfinite(options.input_symmetry_tolerance) || options.input_symmetry_tolerance < 0.0)
        throw std::invalid_argument("bounded CCSD solver input symmetry tolerance must be nonnegative finite");
    if (!provider.value)
        throw std::invalid_argument("bounded CCSD solver requires an integral callback");
    const U o = p.n_occupied, v = p.n_virtual, ov = o * v, o2 = o * o, v2 = v * v, nt2 = o2 * v2;
    view_extent(in.f_oo, o2); view_extent(in.f_vv, v2); view_extent(in.f_ov, ov);
    if (supplied) { view_extent(in.initial_t1, ov); view_extent(in.initial_t2, nt2); }
    scan(in.f_oo, o2); scan(in.f_vv, v2); scan(in.f_ov, ov);
    if (supplied) { scan(in.initial_t1, ov); scan(in.initial_t2, nt2); }
    for (U i = 0; i < o; ++i) for (U j = 0; j < i; ++j)
        symmetric(in.f_oo.data[i * o + j], in.f_oo.data[j * o + i], options.input_symmetry_tolerance);
    for (U a = 0; a < v; ++a) for (U b = 0; b < a; ++b)
        symmetric(in.f_vv.data[a * v + b], in.f_vv.data[b * v + a], options.input_symmetry_tolerance);
    if (supplied) for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j)
        for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b)
            symmetric(in.initial_t2.data[(i * o + j) * v2 + a * v + b],
                      in.initial_t2.data[(j * o + i) * v2 + b * v + a], options.input_symmetry_tolerance);
    const auto gap1 = [&](U i, U a) {
        return gap(in.f_vv.data[a * v + a], 0.0, in.f_oo.data[i * o + i], 0.0, options.denominator_floor);
    };
    const auto gap2 = [&](U i, U j, U a, U b) {
        return gap(in.f_vv.data[a * v + a], in.f_vv.data[b * v + b],
                   in.f_oo.data[i * o + i], in.f_oo.data[j * o + j], options.denominator_floor);
    };
    double minimum = std::numeric_limits<double>::infinity(), maximum = 0.0;
    const auto record_gap = [&](double x) { minimum = std::min(minimum, x); maximum = std::max(maximum, x); };
    for (U i = 0; i < o; ++i) for (U a = 0; a < v; ++a) record_gap(gap1(i, a));
    for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j)
        for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) record_gap(gap2(i, j, a, b));

    BoundedRestrictedCCSDSolverResult result;
    result.memory = p; result.minimum_denominator = minimum; result.maximum_denominator = maximum;
    result.t1.resize(static_cast<std::size_t>(ov)); result.t2.resize(static_cast<std::size_t>(nt2));
    if (supplied) {
        std::copy_n(in.initial_t1.data, static_cast<std::size_t>(ov), result.t1.data());
        std::copy_n(in.initial_t2.data, static_cast<std::size_t>(nt2), result.t2.data());
    }
    std::vector<double> candidate1(static_cast<std::size_t>(ov)), candidate2(static_cast<std::size_t>(nt2));
    IntegralCounter counter{&provider, 0, p.integral_calls_upper_bound};
    const BoundedRestrictedCCSDIntegralProvider counted{&IntegralCounter::value, &counter,
        provider.retained_numerical_bytes, provider.maximum_transient_numerical_bytes};
    const BoundedRestrictedCCSDTargetCaps target_caps{p.target.peak_owned_numerical_bytes,
        p.target.total_live_numerical_bytes, p.target.integral_calls_upper_bound, p.target.kernel_work_units_upper_bound};
    double previous_energy = 0.0;
    U target_count = 0;
    for (U iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        BoundedRestrictedCCSDSolverProgress record;
        record.iteration = iteration;
        const bool update = iteration < options.maximum_iterations;
        BoundedRestrictedCCSDTargetInput target{o, v, 0, 0,
            {result.t1.data(), result.t1.size()}, {result.t2.data(), result.t2.size()},
            in.f_oo, in.f_vv, in.f_ov};
        for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j) {
            target.target_i = i; target.target_j = j;
            // The complete leaf owner dies before the next target is built.
            const auto residual = bounded_restricted_ccsd_target_residual(target, counted, target_caps);
            ++target_count;
            if (j == 0) for (U a = 0; a < v; ++a) {
                const double r = residual.singles[a];
                record.singles_max_residual = std::max(record.singles_max_residual, std::abs(r));
                record.singles_residual_norm = finite(std::hypot(record.singles_residual_norm, r));
                if (update) candidate1[i * v + a] = finite(result.t1[i * v + a] - finite(r / gap1(i, a)));
            }
            for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) {
                const U index = (i * o + j) * v2 + a * v + b;
                const double r = residual.doubles[a * v + b];
                record.doubles_max_residual = std::max(record.doubles_max_residual, std::abs(r));
                record.doubles_residual_norm = finite(std::hypot(record.doubles_residual_norm, r));
                if (update) candidate2[index] = finite(result.t2[index] - finite(r / gap2(i, j, a, b)));
            }
        }
        Sum energy;
        for (U ia = 0; ia < ov; ++ia)
            energy.include(finite(2.0 * finite(in.f_ov.data[ia] * result.t1[ia])));
        for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j)
            for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) {
                const double direct = IntegralCounter::value(i, o + a, j, o + b, &counter);
                const double exchange = IntegralCounter::value(i, o + b, j, o + a, &counter);
                const double tau = finite(result.t2[(i * o + j) * v2 + a * v + b]
                    + finite(result.t1[i * v + a] * result.t1[j * v + b]));
                energy.include(finite(finite(2.0 * direct - exchange) * tau));
            }
        record.correlation_energy = energy.total();
        record.has_previous_energy = iteration > 1;
        record.energy_change = record.has_previous_energy ? finite(record.correlation_energy - previous_energy) : 0.0;
        record.converged = record.has_previous_energy
            && std::abs(record.energy_change) <= options.energy_tolerance
            && record.singles_max_residual <= options.singles_residual_tolerance
            && record.doubles_max_residual <= options.doubles_residual_tolerance;
        record.integral_calls = counter.calls; record.target_evaluations = target_count;
        result.final_snapshot = record;
        if (progress) progress(record, progress_context);
        if (record.converged || !update) return result;
        previous_energy = record.correlation_energy;
        result.t1.swap(candidate1); result.t2.swap(candidate2);
    }
    throw std::logic_error("bounded CCSD solver snapshot loop did not terminate");
}
}  // namespace vibeqc
