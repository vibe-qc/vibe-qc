#include "vibeqc/bounded_restricted_triples_target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vibeqc {
namespace {

using U = std::uint64_t;
static_assert(sizeof(double) == 8, "bounded triples target requires binary64");
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("bounded triples target count overflows uint64");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("bounded triples target extent overflows uint64");
    return a * b;
}
double finite(double x) {
    if (!std::isfinite(x))
        throw std::overflow_error("bounded triples target numerical value is not finite");
    return x;
}
void accumulate(double term, double& value, double& correction) {
    finite(term);
    const double next = finite(value + term);
    correction = finite(correction + (std::abs(value) >= std::abs(term)
        ? (value - next) + term : (term - next) + value));
    value = next;
}
struct Sum {
    double value = 0, correction = 0;
    void include(double x) { accumulate(x, value, correction); }
    double total() const { return finite(value + correction); }
};
void require_view(const BoundedRestrictedTriplesRealView& view, U count) {
    const U bytes = mul(8, count);
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    if (!view.data || view.element_count < count || address % alignof(double) != 0
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("bounded triples target borrowed view is invalid");
}
void scan(const BoundedRestrictedTriplesRealView& view, U count) {
    for (U n = 0; n < count; ++n)
        if (!std::isfinite(view.data[n]))
            throw std::invalid_argument("bounded triples target input must be finite");
}

struct ProjectionReceiver {
    BoundedRestrictedTriplesNeighbourRequest expected;
    U target_dimension = 0, maximum_source_dimension = 0;
    U visits = 0, largest_source_dimension = 0;
    bool receiver_failed = false;
    double fock = 0;
    double* residual = nullptr;
    double* correction = nullptr;
    double* first = nullptr; // [d_max,d_max] allocated; compact [d,d] used
    double* second = nullptr; // [v,d_max] allocated; compact [v,d] used

    static void receive(const BoundedRestrictedTriplesNeighbourView& source, void* context) {
        auto& self = *static_cast<ProjectionReceiver*>(context);
        self.receiver_failed = true;
        if (++self.visits != 1)
            throw std::runtime_error("bounded triples provider must visit exactly once");
        if (source.ordered_occupied != self.expected.ordered_occupied
            || source.amplitude_snapshot_id != self.expected.amplitude_snapshot_id)
            throw std::invalid_argument("bounded triples source ordered labels or amplitude snapshot mismatch");
        const U d = source.source_virtual_dimension, v = self.target_dimension;
        if (d > self.maximum_source_dimension)
            throw std::length_error("bounded triples source virtual dimension exceeds admitted maximum");
        if (!d) {
            if (source.amplitudes.element_count || source.target_source_overlap.element_count)
                throw std::invalid_argument("bounded triples empty source requires exact empty views");
            self.receiver_failed = false;
            return; // A present, explicitly empty retained space contributes zero.
        }
        const U d2 = mul(d, d), d3 = mul(d2, d), vd = mul(v, d);
        require_view(source.amplitudes, d3); require_view(source.target_source_overlap, vd);
        scan(source.amplitudes, d3); scan(source.target_source_overlap, vd);
        const double* t = source.amplitudes.data;
        const double* s = source.target_source_overlap.data;
        // Apply S[a,x] S[b,y] S[c,z] T[x,y,z] one target-a slab at a time.
        // There is no explicit six-index overlap or complete transformed T3.
        for (U a = 0; a < v; ++a) {
            for (U y = 0; y < d; ++y) for (U z = 0; z < d; ++z) {
                Sum value;
                for (U x = 0; x < d; ++x)
                    value.include(s[a * d + x] * t[(x * d + y) * d + z]);
                self.first[y * d + z] = value.total();
            }
            for (U b = 0; b < v; ++b) for (U z = 0; z < d; ++z) {
                Sum value;
                for (U y = 0; y < d; ++y)
                    value.include(s[b * d + y] * self.first[y * d + z]);
                self.second[b * d + z] = value.total();
            }
            for (U b = 0; b < v; ++b) for (U c = 0; c < v; ++c) {
                Sum value;
                for (U z = 0; z < d; ++z)
                    value.include(s[c * d + z] * self.second[b * d + z]);
                const U abc = (a * v + b) * v + c;
                accumulate(-self.fock * value.total(), self.residual[abc], self.correction[abc]);
            }
        }
        self.largest_source_dimension = d;
        self.receiver_failed = false;
    }
};

}  // namespace

BoundedRestrictedTriplesTargetMemoryPlan plan_bounded_restricted_triples_target(
    U o, U v, U d, U retained, U transient) {
    if (!o || !v || !d)
        throw std::invalid_argument("bounded triples target dimensions must be positive");
    const U v2 = mul(v, v), v3 = mul(v2, v), d2 = mul(d, d), d3 = mul(d2, d), vd = mul(v, d);
    BoundedRestrictedTriplesTargetMemoryPlan p;
    p.n_occupied = o; p.n_virtual = v; p.maximum_source_virtual_dimension = d;
    const U target_elements = add(mul(3, v3), add(v, mul(3, o)));
    const U view_elements = add(d3, vd);
    p.borrowed_target_bytes = mul(8, target_elements);
    p.active_provider_view_bytes_upper_bound = mul(8, view_elements);
    p.provider_retained_numerical_bytes = retained;
    p.provider_maximum_transient_numerical_bytes = transient;
    p.output_bytes = mul(8, v3); p.compensation_bytes = p.output_bytes;
    p.projection_workspace_bytes = mul(8, add(d2, vd));
    p.peak_owned_numerical_bytes = add(add(p.output_bytes, p.compensation_bytes), p.projection_workspace_bytes);
    p.total_live_numerical_bytes = add(add(p.peak_owned_numerical_bytes, p.borrowed_target_bytes),
        add(p.active_provider_view_bytes_upper_bound, add(retained, transient)));
    p.provider_visits_upper_bound = mul(3, o - 1);
    // Scan one active source and perform its three mode products. The factor
    // 64 includes scalar compensated accumulation, finite checks and indexing;
    // the target factor includes diagonal construction, spin adaptation,
    // energy contraction and residual norms. Provider internals are excluded.
    const U per_visit = add(view_elements, mul(64,
        add(add(mul(v, d3), mul(v2, d2)), add(mul(v3, d), v3))));
    p.kernel_work_units_upper_bound = add(target_elements,
        add(mul(128, add(v3, o)), mul(p.provider_visits_upper_bound, per_visit)));
    const U limit = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max());
    if (p.borrowed_target_bytes > limit || p.active_provider_view_bytes_upper_bound > limit
        || p.peak_owned_numerical_bytes > limit
        || v3 > std::vector<double>().max_size()
        || add(v3, add(d2, vd)) > std::vector<double>().max_size())
        throw std::overflow_error("bounded triples target exceeds native numerical address extent");
    return p;
}

BoundedRestrictedTriplesTargetResult bounded_restricted_triples_target_residual(
    const BoundedRestrictedTriplesTargetInput& in,
    const BoundedRestrictedTriplesNeighbourProvider& provider,
    const BoundedRestrictedTriplesTargetCaps& caps) {
    const auto plan = plan_bounded_restricted_triples_target(in.n_occupied, in.n_virtual,
        provider.maximum_source_virtual_dimension, provider.retained_numerical_bytes,
        provider.maximum_transient_numerical_bytes);
    if (!caps.maximum_owned_numerical_bytes || caps.maximum_owned_numerical_bytes < plan.peak_owned_numerical_bytes
        || !caps.maximum_total_numerical_bytes || caps.maximum_total_numerical_bytes < plan.total_live_numerical_bytes)
        throw std::length_error("bounded triples target numerical byte cap is missing or exceeded");
    if (!caps.maximum_provider_visits || caps.maximum_provider_visits < plan.provider_visits_upper_bound
        || !caps.maximum_kernel_work_units || caps.maximum_kernel_work_units < plan.kernel_work_units_upper_bound)
        throw std::length_error("bounded triples target visit or kernel work cap is missing or exceeded");
    const U o = in.n_occupied, v = in.n_virtual, v3 = v * v * v;
    if (!in.amplitude_snapshot_id)
        throw std::invalid_argument("bounded triples target requires a defined positive amplitude snapshot");
    for (U occupied : in.occupied)
        if (occupied >= o) throw std::out_of_range("bounded triples target occupied label is out of range");
    require_view(in.amplitudes, v3); require_view(in.connected_moment, v3);
    require_view(in.singles_moment, v3); require_view(in.virtual_energies, v);
    require_view(in.occupied_fock_rows, 3 * o);
    scan(in.amplitudes, v3); scan(in.connected_moment, v3); scan(in.singles_moment, v3);
    scan(in.virtual_energies, v); scan(in.occupied_fock_rows, 3 * o);
    U nonzero_couplings = 0;
    for (U axis = 0; axis < 3; ++axis) {
        for (U l = 0; l < o; ++l)
            if (l != in.occupied[axis] && in.occupied_fock_rows.data[axis * o + l] != 0.0)
                ++nonzero_couplings;
        for (U other = 0; other < axis; ++other)
            if (in.occupied[axis] == in.occupied[other])
                for (U l = 0; l < o; ++l)
                    if (in.occupied_fock_rows.data[axis * o + l] != in.occupied_fock_rows.data[other * o + l])
                        throw std::invalid_argument("bounded triples repeated occupied labels have different Fock rows");
    }
    if (nonzero_couplings && !provider.visit)
        throw std::invalid_argument("bounded triples nonzero coupling requires a neighbour provider");

    BoundedRestrictedTriplesTargetResult result;
    result.memory = plan; result.occupied = in.occupied; result.amplitude_snapshot_id = in.amplitude_snapshot_id;
    result.residual.resize(static_cast<std::size_t>(v3));
    const U dmax = provider.maximum_source_virtual_dimension;
    std::vector<double> workspace(static_cast<std::size_t>(v3 + dmax * dmax + v * dmax));
    double* correction = workspace.data();
    double* first = correction + v3;
    double* second = first + dmax * dmax;
    for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) for (U c = 0; c < v; ++c) {
        const U abc = (a * v + b) * v + c;
        Sum diagonal;
        diagonal.include(in.virtual_energies.data[a]); diagonal.include(in.virtual_energies.data[b]);
        diagonal.include(in.virtual_energies.data[c]);
        for (U axis = 0; axis < 3; ++axis)
            diagonal.include(-in.occupied_fock_rows.data[axis * o + in.occupied[axis]]);
        // fma preserves a small residual when the diagonal action cancels W.
        result.residual[abc] = finite(std::fma(diagonal.total(), in.amplitudes.data[abc], in.connected_moment.data[abc]));
    }
    for (U axis = 0; axis < 3; ++axis) for (U l = 0; l < o; ++l) {
        if (l == in.occupied[axis]) continue;
        const double fock = in.occupied_fock_rows.data[axis * o + l];
        if (fock == 0.0) { ++result.exactly_zero_couplings_skipped; continue; }
        if (result.provider_visits >= caps.maximum_provider_visits
            || result.provider_visits >= plan.provider_visits_upper_bound)
            throw std::length_error("bounded triples exhausted its neighbour visit cap");
        BoundedRestrictedTriplesNeighbourRequest request;
        request.replaced_axis = axis; request.replacement_occupied = l;
        request.ordered_occupied = in.occupied; request.ordered_occupied[axis] = l;
        request.amplitude_snapshot_id = in.amplitude_snapshot_id;
        ProjectionReceiver receiver{request, v, dmax, 0, 0, false, fock,
            result.residual.data(), correction, first, second};
        ++result.provider_visits;
        provider.visit(request, &ProjectionReceiver::receive, &receiver, provider.context);
        if (receiver.visits != 1 || receiver.receiver_failed)
            throw std::runtime_error("bounded triples provider violated exactly-once receiver protocol");
        result.largest_source_virtual_dimension = std::max(result.largest_source_virtual_dimension,
            receiver.largest_source_dimension);
    }
    if (result.provider_visits != nonzero_couplings)
        throw std::logic_error("bounded triples did not evaluate every nonzero occupied coupling");

    Sum energy;
    const auto t = [&](U a, U b, U c) { return in.amplitudes.data[(a * v + b) * v + c]; };
    for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) for (U c = 0; c < v; ++c) {
        const U abc = (a * v + b) * v + c;
        result.residual[abc] = finite(result.residual[abc] + correction[abc]);
        result.maximum_absolute_residual = std::max(result.maximum_absolute_residual, std::abs(result.residual[abc]));
        result.residual_frobenius_norm = finite(std::hypot(result.residual_frobenius_norm, result.residual[abc]));
        Sum adapted;
        adapted.include(4.0 * t(a, b, c)); adapted.include(-2.0 * t(a, c, b));
        adapted.include(-2.0 * t(c, b, a)); adapted.include(-2.0 * t(b, a, c));
        adapted.include(t(c, a, b)); adapted.include(t(b, c, a));
        energy.include(adapted.total() * finite(in.connected_moment.data[abc] + in.singles_moment.data[abc]));
    }
    result.raw_energy_contraction = energy.total();
    return result;
}

}  // namespace vibeqc
