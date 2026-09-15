#include "vibeqc/bounded_restricted_triples_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using View = BoundedRestrictedCCSDRealView;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "bounded triples solver requires IEEE binary64");
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("bounded triples solver count overflow");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("bounded triples solver extent overflow");
    return a * b;
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("bounded triples solver arithmetic is not finite");
    return x;
}
struct Sum {
    double value = 0.0, correction = 0.0;
    void include(double term) {
        finite(term);
        const auto next = finite(value + term);
        correction = finite(correction + (std::abs(value) >= std::abs(term)
            ? (value - next) + term : (term - next) + value));
        value = next;
    }
    double total() const { return finite(value + correction); }
};
void view(const View& x, U count) {
    const auto address = reinterpret_cast<std::uintptr_t>(x.data);
    const auto bytes = mul(8, count);
    if (!x.data || x.element_count < count || address % alignof(double)
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("bounded triples solver invalid borrowed input view");
}
void scan(const View& x, U count) {
    for (U n = 0; n < count; ++n)
        if (!std::isfinite(x.data[n])) throw std::invalid_argument("bounded triples solver input must be finite");
}
void symmetry(double x, double y, double tolerance) {
    const auto scale = std::max({1.0, std::abs(x), std::abs(y)});
    if (std::abs(x / scale - y / scale) > tolerance)
        throw std::invalid_argument("bounded triples solver input symmetry audit failed");
}
double denominator(const double values[6], double floor) {
    double largest = 0.0;
    for (unsigned n = 0; n < 6; ++n) largest = std::max(largest, std::abs(values[n]));
    int exponent = 0;
    if (largest != 0.0) std::frexp(largest, &exponent);
    Sum s;
    for (unsigned n = 0; n < 6; ++n) {
        const auto scaled = std::scalbn(values[n], -exponent);
        if (std::scalbn(scaled, exponent) != values[n])
            throw std::overflow_error("bounded triples solver denominator scaling loses range");
        s.include(scaled);
    }
    const auto gap = std::scalbn(s.total(), exponent);
    if (!std::isfinite(gap) || gap <= floor)
        throw std::invalid_argument("bounded triples solver denominator must be finite and strictly exceed floor");
    return gap;
}
struct IntegralCounter {
    const BoundedRestrictedCCSDIntegralProvider& provider;
    U calls = 0, limit = 0;
    static double value(U p, U q, U r, U s, void* context) {
        auto& self = *static_cast<IntegralCounter*>(context);
        if (self.calls >= self.limit) throw std::length_error("bounded triples solver exhausted integral call cap");
        ++self.calls;
        return finite(self.provider.value(p, q, r, s, self.provider.context));
    }
};
struct Neighbours {
    U o = 0, v = 0, snapshot = 0;
    const double* amplitudes = nullptr;
    const double* identity = nullptr;
    static void visit(const BoundedRestrictedTriplesNeighbourRequest& request,
                      BoundedRestrictedTriplesNeighbourReceiver receiver, void* receiving, void* context) {
        const auto& self = *static_cast<Neighbours*>(context);
        if (request.amplitude_snapshot_id != self.snapshot || request.replaced_axis >= 3
            || request.replacement_occupied >= self.o)
            throw std::logic_error("bounded triples solver internal snapshot request mismatch");
        for (auto index : request.ordered_occupied)
            if (index >= self.o) throw std::logic_error("bounded triples solver internal occupied request mismatch");
        const auto& x = request.ordered_occupied;
        const auto cube = self.v * self.v * self.v;
        BoundedRestrictedTriplesNeighbourView source;
        source.ordered_occupied = x; source.amplitude_snapshot_id = self.snapshot;
        source.source_virtual_dimension = self.v;
        source.amplitudes = {self.amplitudes + ((x[0] * self.o + x[1]) * self.o + x[2]) * cube,
                             static_cast<std::size_t>(cube)};
        source.target_source_overlap = {self.identity, static_cast<std::size_t>(self.v * self.v)};
        receiver(source, receiving);
    }
};
} // namespace

BoundedRestrictedTriplesSolverMemoryPlan plan_bounded_restricted_triples_solver(
    U o, U v, U iterations, const BoundedRestrictedTriplesSolverInventory& inventory,
    U retained, U transient) {
    if (!iterations || !inventory.numerical_replicas)
        throw std::invalid_argument("bounded triples solver iterations and replicas must be positive");
    BoundedRestrictedTriplesSolverMemoryPlan p;
    p.moments = plan_bounded_restricted_triples_moments(o, v, retained, transient);
    p.target = plan_bounded_restricted_triples_target(o, v, v);
    p.n_occupied = o; p.n_virtual = v; p.maximum_iterations = iterations;
    const auto o2 = mul(o, o), o3 = mul(o2, o), v2 = mul(v, v), v3 = mul(v2, v), ov = mul(o, v);
    const auto count = mul(o3, v3);
    p.amplitude_snapshot_bytes = mul(8, count); p.candidate_snapshot_bytes = p.amplitude_snapshot_bytes;
    p.orbital_workspace_bytes = mul(8, add(add(v2, v), mul(3, o)));
    p.borrowed_input_bytes = mul(8, add(add(add(o2, v2), mul(2, ov)), mul(o2, v2)));
    p.peak_owned_numerical_bytes = add(mul(2, p.amplitude_snapshot_bytes), add(p.orbital_workspace_bytes,
        add(p.moments.peak_owned_numerical_bytes, p.target.peak_owned_numerical_bytes)));
    p.total_live_numerical_bytes = add(p.peak_owned_numerical_bytes,
        add(p.borrowed_input_bytes, add(retained, transient)));
    p.external_node_numerical_bytes = inventory.external_node_numerical_bytes;
    p.numerical_replicas = inventory.numerical_replicas;
    p.required_node_numerical_bytes = add(p.external_node_numerical_bytes,
        mul(p.numerical_replicas, p.total_live_numerical_bytes));
    p.target_evaluations_upper_bound = mul(iterations, o3);
    p.neighbour_visits_upper_bound = mul(p.target_evaluations_upper_bound, p.target.provider_visits_upper_bound);
    p.integral_calls_upper_bound = mul(p.target_evaluations_upper_bound, p.moments.integral_calls);
    const auto initial = mul(256, add(add(p.borrowed_input_bytes / 8, count), add(p.orbital_workspace_bytes / 8, 1)));
    const auto per_target = add(add(p.moments.kernel_work_units_upper_bound, p.target.kernel_work_units_upper_bound),
        mul(512, add(add(v3, o), 1)));
    p.kernel_work_units_upper_bound = add(initial, mul(p.target_evaluations_upper_bound, per_target));
    if (p.peak_owned_numerical_bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || p.borrowed_input_bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || count > std::vector<double>().max_size())
        throw std::overflow_error("bounded triples solver exceeds native address extent");
    return p;
}

BoundedRestrictedTriplesSolverResult bounded_restricted_triples_solve(
    const BoundedRestrictedTriplesSolverInput& in, const BoundedRestrictedCCSDIntegralProvider& provider,
    const BoundedRestrictedTriplesSolverOptions& options, const BoundedRestrictedTriplesSolverInventory& inventory,
    const BoundedRestrictedTriplesSolverCaps& caps,
    BoundedRestrictedTriplesSolverProgressCallback progress, void* progress_context) {
    const auto p = plan_bounded_restricted_triples_solver(in.n_occupied, in.n_virtual,
        options.maximum_iterations, inventory, provider.retained_numerical_bytes, provider.maximum_transient_numerical_bytes);
    if (!caps.maximum_owned_numerical_bytes || caps.maximum_owned_numerical_bytes < p.peak_owned_numerical_bytes
        || !caps.maximum_total_numerical_bytes || caps.maximum_total_numerical_bytes < p.total_live_numerical_bytes
        || !caps.maximum_node_numerical_bytes || caps.maximum_node_numerical_bytes < p.required_node_numerical_bytes)
        throw std::length_error("bounded triples solver numerical byte cap is missing or exceeded");
    if (!caps.maximum_integral_calls || caps.maximum_integral_calls < p.integral_calls_upper_bound
        || !caps.maximum_kernel_work_units || caps.maximum_kernel_work_units < p.kernel_work_units_upper_bound)
        throw std::length_error("bounded triples solver call or work cap is missing or exceeded");
    for (double x : {options.denominator_floor, options.residual_tolerance, options.energy_tolerance})
        if (!std::isfinite(x) || x <= 0.0)
            throw std::invalid_argument("bounded triples solver tolerances and denominator floor must be positive finite");
    if (!std::isfinite(options.input_symmetry_tolerance) || options.input_symmetry_tolerance < 0.0)
        throw std::invalid_argument("bounded triples solver symmetry tolerance must be nonnegative finite");
    if (!in.ccsd_snapshot_id || !provider.value)
        throw std::invalid_argument("bounded triples solver needs a CCSD snapshot label and integral callback");
    const auto o = p.n_occupied, v = p.n_virtual, o2 = o * o, v2 = v * v, v3 = v2 * v, ov = o * v;
    view(in.t1, ov); view(in.t2, o2 * v2); view(in.f_oo, o2); view(in.f_vv, v2); view(in.f_ov, ov);
    scan(in.t1, ov); scan(in.t2, o2 * v2); scan(in.f_oo, o2); scan(in.f_vv, v2); scan(in.f_ov, ov);
    for (U at = 0; at < ov; ++at)
        if (in.f_ov.data[at] != 0.0) throw std::invalid_argument("bounded triples solver requires exactly zero Fov");
    for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b)
        if (a != b && in.f_vv.data[a * v + b] != 0.0)
            throw std::invalid_argument("bounded triples solver requires exactly diagonal Fvv");
    for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j) {
        symmetry(in.f_oo.data[i * o + j], in.f_oo.data[j * o + i], options.input_symmetry_tolerance);
        for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b)
            symmetry(in.t2.data[(i * o + j) * v2 + a * v + b],
                     in.t2.data[(j * o + i) * v2 + b * v + a], options.input_symmetry_tolerance);
    }
    const auto gap = [&](U i, U j, U k, U a, U b, U c) {
        const double values[6] = {in.f_vv.data[a * v + a], in.f_vv.data[b * v + b], in.f_vv.data[c * v + c],
            -in.f_oo.data[i * o + i], -in.f_oo.data[j * o + j], -in.f_oo.data[k * o + k]};
        return denominator(values, options.denominator_floor);
    };
    BoundedRestrictedTriplesSolverResult result;
    result.memory = p; result.ccsd_snapshot_id = in.ccsd_snapshot_id;
    result.minimum_denominator = std::numeric_limits<double>::infinity();
    for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j) for (U k = 0; k < o; ++k)
        for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) for (U c = 0; c < v; ++c) {
            const auto d = gap(i, j, k, a, b, c);
            result.minimum_denominator = std::min(result.minimum_denominator, d);
            result.maximum_denominator = std::max(result.maximum_denominator, d);
        }
    result.amplitudes.resize(static_cast<std::size_t>(p.amplitude_snapshot_bytes / 8));
    std::vector<double> candidate(result.amplitudes.size());
    std::vector<double> orbital(static_cast<std::size_t>(p.orbital_workspace_bytes / 8));
    auto* identity = orbital.data(); auto* energies = identity + v2; auto* rows = energies + v;
    for (U a = 0; a < v; ++a) { identity[a * v + a] = 1.0; energies[a] = in.f_vv.data[a * v + a]; }
    IntegralCounter counter{provider, 0, p.integral_calls_upper_bound};
    const BoundedRestrictedCCSDIntegralProvider counted{&IntegralCounter::value, &counter,
        provider.retained_numerical_bytes, provider.maximum_transient_numerical_bytes};
    const BoundedRestrictedTriplesMomentsCaps moment_caps{p.moments.peak_owned_numerical_bytes,
        p.moments.total_live_numerical_bytes, p.moments.integral_calls, p.moments.kernel_work_units_upper_bound};
    const BoundedRestrictedTriplesTargetCaps target_caps{p.target.peak_owned_numerical_bytes,
        p.target.total_live_numerical_bytes, std::max<U>(1, p.target.provider_visits_upper_bound),
        p.target.kernel_work_units_upper_bound};
    double previous_energy = 0.0;
    U targets = 0, visits = 0;
    for (U iteration = 1; iteration <= options.maximum_iterations; ++iteration) {
        Neighbours context{o, v, iteration, result.amplitudes.data(), identity};
        const BoundedRestrictedTriplesNeighbourProvider neighbours{&Neighbours::visit, &context, v, 0, 0};
        BoundedRestrictedTriplesSolverProgress record;
        record.iteration = iteration;
        Sum energy;
        for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j) for (U k = 0; k < o; ++k) {
            const auto start = ((i * o + j) * o + k) * v3;
            const std::array<U, 3> occupied{i, j, k};
            for (U axis = 0; axis < 3; ++axis)
                std::copy_n(in.f_oo.data + occupied[axis] * o, o, rows + axis * o);
            const BoundedRestrictedTriplesMomentsInput moments_input{o, v, occupied, in.ccsd_snapshot_id,
                in.t1, in.t2, in.f_ov, options.input_symmetry_tolerance};
            const auto moments = bounded_restricted_triples_moments(moments_input, counted, moment_caps);
            const BoundedRestrictedTriplesTargetInput target{o, v, occupied, iteration,
                {result.amplitudes.data() + start, static_cast<std::size_t>(v3)},
                {moments.connected.data(), moments.connected.size()}, {moments.singles.data(), moments.singles.size()},
                {energies, static_cast<std::size_t>(v)}, {rows, static_cast<std::size_t>(3 * o)}};
            const auto residual = bounded_restricted_triples_target_residual(target, neighbours, target_caps);
            ++targets; visits = add(visits, residual.provider_visits);
            std::copy(residual.residual.begin(), residual.residual.end(), candidate.data() + start);
            record.maximum_absolute_residual = std::max(record.maximum_absolute_residual, residual.maximum_absolute_residual);
            record.residual_frobenius_norm = finite(std::hypot(record.residual_frobenius_norm, residual.residual_frobenius_norm));
            if (i <= j && j <= k)
                energy.include((2 - static_cast<int>(i == j) - static_cast<int>(j == k)) * residual.raw_energy_contraction);
        }
        record.integral_calls = counter.calls; record.target_evaluations = targets; record.neighbour_visits = visits;
        record.triples_energy = energy.total(); record.has_previous_energy = iteration > 1;
        record.energy_change = record.has_previous_energy ? finite(record.triples_energy - previous_energy) : 0.0;
        record.converged = record.has_previous_energy && record.maximum_absolute_residual <= options.residual_tolerance
            && std::abs(record.energy_change) <= options.energy_tolerance;
        result.final_snapshot = record;
        if (progress) progress(record, progress_context);
        if (record.converged || iteration == options.maximum_iterations) return result;
        for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j) for (U k = 0; k < o; ++k) {
            const auto start = ((i * o + j) * o + k) * v3;
            for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) for (U c = 0; c < v; ++c) {
                const auto at = start + (a * v + b) * v + c;
                candidate[at] = finite(result.amplitudes[at] - candidate[at] / gap(i, j, k, a, b, c));
            }
        }
        previous_energy = record.triples_energy;
        result.amplitudes.swap(candidate);
    }
    throw std::logic_error("bounded triples solver exited without evaluated snapshot");
}

} // namespace vibeqc
