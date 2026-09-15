#include "vibeqc/bounded_restricted_triples_moments.hpp"

#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cfloat>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
static_assert(sizeof(double) == 8, "bounded triples moments require binary64");
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("bounded triples moments count addition overflows");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("bounded triples moments count multiplication overflows");
    return a * b;
}
double finite(double value) {
    if (!std::isfinite(value))
        throw std::overflow_error("bounded triples moments arithmetic or integral is not finite");
    return value;
}
struct Sum {
    double sum = 0.0, correction = 0.0;
    void include(double term) {
        finite(term);
        const double next = finite(sum + term);
        correction = finite(correction + (std::abs(sum) >= std::abs(term)
            ? (sum - next) + term : (term - next) + sum));
        sum = next;
    }
    double total() const { return finite(sum + correction); }
};
void extent(const BoundedRestrictedCCSDRealView& view, U count) {
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    const U bytes = mul(8, count);
    if (!view.data || view.element_count < count || address % alignof(double)
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("bounded triples moments borrowed view is invalid");
}
void accessor_environment() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
    throw std::runtime_error("bounded triples accessor requires strict binary64 evaluation");
#else
    if (!std::numeric_limits<double>::is_iec559
        || std::numeric_limits<double>::digits != 53
        || std::numeric_limits<double>::max_exponent != 1024
        || FLT_RADIX != 2 || std::fegetround() != FE_TONEAREST)
        throw std::runtime_error("bounded triples accessor requires nearest IEEE binary64 arithmetic");
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double twice = tiny + tiny;
    volatile double recovered = twice * 0.5;
    if (tiny == 0.0 || twice == 0.0 || recovered != tiny)
        throw std::runtime_error("bounded triples accessor requires gradual underflow");
#endif
}

struct DenseAmplitudes {
    const double* t1;
    const double* t2;
    U o, v;
    double singles(U i, U a) const { return t1[i * v + a]; }
    double doubles(U i, U j, U a, U b) const {
        return t2[(i * o + j) * v * v + a * v + b];
    }
    void after_integral() const {}
};

struct AccessorAmplitudes {
    BoundedRestrictedCCSDAmplitudeProvider provider;
    U o, v, singles_limit, doubles_limit, work_limit;
    double tolerance;
    U singles_calls = 0, doubles_calls = 0, work = 0, audits = 0;
    double maximum_symmetry_error = 0.0;
    void charge(U& count, U limit, U cost) {
        if (count >= limit || cost > work_limit - work)
            throw std::length_error("bounded triples accessor exhausted amplitude call or work census");
        ++count;
        work += cost;
    }
    double singles(U i, U a) {
        if (i >= o || a >= v)
            throw std::out_of_range("bounded triples accessor singles index is out of range");
        charge(singles_calls, singles_limit, provider.maximum_singles_work_units_per_query);
        const double x = provider.singles(i, a, provider.context);
        accessor_environment();
        return finite(x);
    }
    double raw_doubles(U i, U j, U a, U b) {
        if (i >= o || j >= o || a >= v || b >= v)
            throw std::out_of_range("bounded triples accessor doubles index is out of range");
        charge(doubles_calls, doubles_limit, provider.maximum_doubles_work_units_per_query);
        const double x = provider.doubles(i, j, a, b, provider.context);
        accessor_environment();
        return finite(x);
    }
    double doubles(U i, U j, U a, U b) {
        const double x = raw_doubles(i, j, a, b);
        const double y = raw_doubles(j, i, b, a);
        const double scale = std::max({1.0, std::abs(x), std::abs(y)});
        const double error = std::abs(x / scale - y / scale);
        maximum_symmetry_error = std::max(maximum_symmetry_error, error);
        ++audits;
        if (error > tolerance)
            throw std::invalid_argument("bounded triples accessor consumed amplitude symmetry audit failed");
        return x; // Audit only; never average or replace the supplied amplitude.
    }
    void after_integral() const { accessor_environment(); }
};

template<class Amplitudes>
BoundedRestrictedTriplesMomentsResult moments_arithmetic(
    const BoundedRestrictedTriplesMomentsMemoryPlan& p,
    const std::array<U, 3>& occupied, U snapshot,
    const BoundedRestrictedCCSDIntegralProvider& provider, Amplitudes& amplitude) {
    const U o = p.n_occupied, v = p.n_virtual, v2 = v * v;
    BoundedRestrictedTriplesMomentsResult result;
    result.memory = p; result.occupied = occupied; result.amplitude_snapshot_id = snapshot;
    result.connected.resize(static_cast<std::size_t>(p.connected_output_bytes / 8));
    result.singles.resize(static_cast<std::size_t>(p.singles_output_bytes / 8));
    const auto eri = [&](U a, U b, U c, U d) {
        if (result.integral_calls >= p.integral_calls)
            throw std::length_error("bounded triples moments exhausted integral call census");
        ++result.integral_calls;
        const double x = provider.value(a, b, c, d, provider.context);
        amplitude.after_integral();
        return finite(x);
    };
    // Explicit six simultaneous permutations, in Riplinger Eq.(9) order.
    constexpr std::array<std::array<unsigned, 3>, 6> permutations{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{2, 0, 1}},
        {{2, 1, 0}}, {{1, 2, 0}}, {{1, 0, 2}},
    }};
    for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) for (U c = 0; c < v; ++c) {
        const std::array<U, 3> abc{a, b, c};
        Sum w;
        for (const auto& permutation : permutations) {
            const U i = occupied[permutation[0]], j = occupied[permutation[1]], k = occupied[permutation[2]];
            const U aa = abc[permutation[0]], bb = abc[permutation[1]], cc = abc[permutation[2]];
            for (U d = 0; d < v; ++d)
                w.include(amplitude.doubles(k, j, cc, d) * eri(i, o + aa, o + bb, o + d));
            for (U l = 0; l < o; ++l)
                w.include(-amplitude.doubles(i, l, aa, bb) * eri(k, o + cc, j, l));
        }
        const U i = occupied[0], j = occupied[1], k = occupied[2];
        Sum u;
        u.include(amplitude.singles(i, a) * eri(j, o + b, k, o + c));
        u.include(amplitude.singles(j, b) * eri(i, o + a, k, o + c));
        u.include(amplitude.singles(k, c) * eri(j, o + b, i, o + a));
        const U index = a * v2 + b * v + c;
        result.connected[index] = w.total(); result.singles[index] = u.total();
    }
    if (result.integral_calls != p.integral_calls)
        throw std::logic_error("bounded triples moments integral census is incomplete");
    return result;
}

}  // namespace

BoundedRestrictedTriplesMomentsMemoryPlan plan_bounded_restricted_triples_moments(
    U o, U v, U retained, U transient) {
    if (!o || !v)
        throw std::invalid_argument("bounded triples moments local dimensions must be positive");
    const U ov = mul(o, v), o2v2 = mul(ov, ov), v3 = mul(mul(v, v), v);
    (void) add(o, v);
    BoundedRestrictedTriplesMomentsMemoryPlan p;
    p.n_occupied = o; p.n_virtual = v;
    p.borrowed_input_bytes = mul(8, add(mul(2, ov), o2v2));
    p.provider_retained_numerical_bytes = retained;
    p.provider_maximum_transient_numerical_bytes = transient;
    p.connected_output_bytes = p.singles_output_bytes = mul(8, v3);
    p.peak_owned_numerical_bytes = mul(16, v3);
    p.total_live_numerical_bytes = add(add(p.peak_owned_numerical_bytes, p.borrowed_input_bytes), add(retained, transient));
    p.integral_calls = mul(v3, add(mul(6, add(o, v)), 3));
    p.kernel_work_units_upper_bound = add(add(mul(2, ov), mul(2, o2v2)),
        add(mul(128, p.integral_calls), add(mul(64, v3), 256)));
    const U limit = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max());
    if (p.borrowed_input_bytes > limit || p.peak_owned_numerical_bytes > limit
        || v3 > std::vector<double>().max_size())
        throw std::overflow_error("bounded triples moments exceeds native numerical address extent");
    return p;
}

BoundedRestrictedTriplesMomentsResult bounded_restricted_triples_moments(
    const BoundedRestrictedTriplesMomentsInput& in,
    const BoundedRestrictedCCSDIntegralProvider& provider,
    const BoundedRestrictedTriplesMomentsCaps& caps) {
    const auto p = plan_bounded_restricted_triples_moments(in.n_occupied, in.n_virtual,
        provider.retained_numerical_bytes, provider.maximum_transient_numerical_bytes);
    if (!caps.maximum_owned_numerical_bytes || caps.maximum_owned_numerical_bytes < p.peak_owned_numerical_bytes
        || !caps.maximum_total_numerical_bytes || caps.maximum_total_numerical_bytes < p.total_live_numerical_bytes)
        throw std::length_error("bounded triples moments numerical byte cap is missing or exceeded");
    if (!caps.maximum_integral_calls || caps.maximum_integral_calls < p.integral_calls
        || !caps.maximum_kernel_work_units || caps.maximum_kernel_work_units < p.kernel_work_units_upper_bound)
        throw std::length_error("bounded triples moments call or work cap is missing or exceeded");
    if (!provider.value)
        throw std::invalid_argument("bounded triples moments require an integral callback");
    if (!in.amplitude_snapshot_id)
        throw std::invalid_argument("bounded triples moments require a positive snapshot sequencing label");
    if (!std::isfinite(in.amplitude_symmetry_tolerance) || in.amplitude_symmetry_tolerance < 0.0)
        throw std::invalid_argument("bounded triples moments symmetry tolerance must be nonnegative finite");
    const U o = p.n_occupied, v = p.n_virtual, ov = o * v, v2 = v * v, nt2 = ov * ov;
    for (const U label : in.occupied)
        if (label >= o) throw std::out_of_range("bounded triples moments occupied label is out of range");
    extent(in.t1, ov); extent(in.t2, nt2); extent(in.f_ov, ov);
    for (U ia = 0; ia < ov; ++ia) {
        if (!std::isfinite(in.t1.data[ia]) || !std::isfinite(in.f_ov.data[ia]))
            throw std::invalid_argument("bounded triples moments input must be finite");
        if (in.f_ov.data[ia] != 0.0)
            throw std::invalid_argument("bounded triples moments require exactly zero Fov, not a general non-HF reference");
    }
    for (U x = 0; x < nt2; ++x)
        if (!std::isfinite(in.t2.data[x]))
            throw std::invalid_argument("bounded triples moments input must be finite");
    const auto amplitude = [&](U i, U j, U a, U b) {
        return in.t2.data[(i * o + j) * v2 + a * v + b];
    };
    for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j)
        for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) {
            const double left = amplitude(i, j, a, b), right = amplitude(j, i, b, a);
            const double scale = std::max({1.0, std::abs(left), std::abs(right)});
            if (std::abs(left / scale - right / scale) > in.amplitude_symmetry_tolerance)
                throw std::invalid_argument("bounded triples moments amplitude symmetry audit failed");
        }
    DenseAmplitudes reader{in.t1.data, in.t2.data, o, v};
    return moments_arithmetic(p, in.occupied, in.amplitude_snapshot_id, provider, reader);
}

BoundedRestrictedTriplesMomentsAccessorMemoryPlan plan_bounded_restricted_triples_moments_accessor(
    U o, U v, const BoundedRestrictedCCSDAmplitudeProvider& supplied,
    U retained, U transient) {
    const auto provider = supplied;
    if (!o || !v)
        throw std::invalid_argument("bounded triples moments local dimensions must be positive");
    if (!provider.maximum_singles_work_units_per_query || !provider.maximum_doubles_work_units_per_query)
        throw std::invalid_argument("bounded triples accessor requires positive amplitude query work declarations");
    const U ov = mul(o, v), v3 = mul(mul(v, v), v);
    BoundedRestrictedTriplesMomentsAccessorMemoryPlan p;
    auto& k = p.kernel;
    k.n_occupied = o; k.n_virtual = v;
    k.borrowed_input_bytes = mul(8, ov);
    k.provider_retained_numerical_bytes = retained;
    k.provider_maximum_transient_numerical_bytes = transient;
    k.connected_output_bytes = k.singles_output_bytes = mul(8, v3);
    k.peak_owned_numerical_bytes = mul(16, v3);
    p.amplitude_retained_numerical_bytes = provider.retained_numerical_bytes;
    p.amplitude_maximum_transient_numerical_bytes = provider.maximum_transient_numerical_bytes;
    k.total_live_numerical_bytes = add(add(k.peak_owned_numerical_bytes, k.borrowed_input_bytes),
        add(add(retained, transient), add(p.amplitude_retained_numerical_bytes,
            p.amplitude_maximum_transient_numerical_bytes)));
    p.singles_calls = mul(3, v3);
    p.contraction_doubles_calls = p.reverse_audit_doubles_calls = mul(mul(6, v3), add(o, v));
    p.doubles_calls = mul(2, p.contraction_doubles_calls);
    k.integral_calls = add(p.singles_calls, p.contraction_doubles_calls);
    p.amplitude_work_units_upper_bound = add(mul(p.singles_calls, provider.maximum_singles_work_units_per_query),
        mul(p.doubles_calls, provider.maximum_doubles_work_units_per_query));
    k.kernel_work_units_upper_bound = add(ov, add(mul(128, k.integral_calls),
        add(mul(64, v3), add(256, mul(64, add(p.singles_calls, p.doubles_calls))))));
    // Fixed nonrecursive scalar frames, copied descriptors and vector controls;
    // 4096 additional bytes conservatively cover ABI stack bookkeeping. This
    // explicit reservation excludes allocator/backend internals and is not RSS.
    p.fixed_control_storage_bytes = 4096 + sizeof(AccessorAmplitudes)
        + 2 * sizeof(BoundedRestrictedTriplesMomentsAccessorMemoryPlan)
        + 2 * sizeof(BoundedRestrictedTriplesMomentsResult)
        + sizeof(BoundedRestrictedTriplesMomentsAccessorResult)
        + sizeof(BoundedRestrictedTriplesMomentsOperatorInput)
        + sizeof(BoundedRestrictedTriplesMomentsAccessorCaps)
        + 2 * sizeof(BoundedRestrictedCCSDAmplitudeProvider)
        + sizeof(BoundedRestrictedCCSDIntegralProvider);
    const U limit = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max());
    if (k.borrowed_input_bytes > limit || k.peak_owned_numerical_bytes > limit
        || v3 > std::vector<double>().max_size())
        throw std::overflow_error("bounded triples accessor exceeds native numerical address extent");
    return p;
}

BoundedRestrictedTriplesMomentsAccessorResult bounded_restricted_triples_moments_accessor(
    const BoundedRestrictedTriplesMomentsOperatorInput& supplied_input,
    const BoundedRestrictedCCSDAmplitudeProvider& supplied_amplitudes,
    const BoundedRestrictedCCSDIntegralProvider& supplied_integrals,
    const BoundedRestrictedTriplesMomentsAccessorCaps& supplied_caps) {
    // Opaque callbacks cannot change these local control snapshots. Borrowed
    // payload/context immutability remains a caller contract, not a certificate.
    const auto in = supplied_input;
    const auto amplitudes = supplied_amplitudes;
    const auto integrals = supplied_integrals;
    const auto caps = supplied_caps;
    const auto p = plan_bounded_restricted_triples_moments_accessor(in.n_occupied, in.n_virtual,
        amplitudes, integrals.retained_numerical_bytes, integrals.maximum_transient_numerical_bytes);
    const auto& k = p.kernel;
    if (!caps.kernel.maximum_owned_numerical_bytes || caps.kernel.maximum_owned_numerical_bytes < k.peak_owned_numerical_bytes
        || !caps.kernel.maximum_total_numerical_bytes || caps.kernel.maximum_total_numerical_bytes < k.total_live_numerical_bytes)
        throw std::length_error("bounded triples accessor numerical byte cap is missing or exceeded");
    if (!caps.kernel.maximum_integral_calls || caps.kernel.maximum_integral_calls < k.integral_calls
        || !caps.kernel.maximum_kernel_work_units || caps.kernel.maximum_kernel_work_units < k.kernel_work_units_upper_bound)
        throw std::length_error("bounded triples accessor integral call or kernel work cap is missing or exceeded");
    if (!caps.maximum_singles_calls || caps.maximum_singles_calls < p.singles_calls
        || !caps.maximum_doubles_calls || caps.maximum_doubles_calls < p.doubles_calls
        || !caps.maximum_amplitude_work_units || caps.maximum_amplitude_work_units < p.amplitude_work_units_upper_bound
        || !caps.maximum_control_storage_bytes || caps.maximum_control_storage_bytes < p.fixed_control_storage_bytes)
        throw std::length_error("bounded triples accessor amplitude call, work or control cap is missing or exceeded");
    if (!amplitudes.singles || !amplitudes.doubles || !integrals.value)
        throw std::invalid_argument("bounded triples accessor requires singles, doubles and integral callbacks");
    if (!in.amplitude_snapshot_id)
        throw std::invalid_argument("bounded triples moments require a positive snapshot sequencing label");
    if (!std::isfinite(in.amplitude_symmetry_tolerance) || in.amplitude_symmetry_tolerance < 0.0)
        throw std::invalid_argument("bounded triples moments symmetry tolerance must be nonnegative finite");
    for (const U label : in.occupied)
        if (label >= k.n_occupied)
            throw std::out_of_range("bounded triples moments occupied label is out of range");
    const U ov = k.n_occupied * k.n_virtual;
    extent(in.f_ov, ov);
    accessor_environment();
    for (U ia = 0; ia < ov; ++ia) {
        if (!std::isfinite(in.f_ov.data[ia]))
            throw std::invalid_argument("bounded triples moments input must be finite");
        if (in.f_ov.data[ia] != 0.0)
            throw std::invalid_argument("bounded triples moments require exactly zero Fov, not a general non-HF reference");
    }
    AccessorAmplitudes reader{amplitudes, k.n_occupied, k.n_virtual,
        p.singles_calls, p.doubles_calls, p.amplitude_work_units_upper_bound, in.amplitude_symmetry_tolerance};
    BoundedRestrictedTriplesMomentsAccessorResult result;
    result.memory = p;
    result.moments = moments_arithmetic(k, in.occupied, in.amplitude_snapshot_id, integrals, reader);
    if (reader.singles_calls != p.singles_calls || reader.doubles_calls != p.doubles_calls
        || reader.audits != p.reverse_audit_doubles_calls || reader.work != p.amplitude_work_units_upper_bound)
        throw std::logic_error("bounded triples accessor amplitude census is incomplete");
    result.singles_calls = reader.singles_calls; result.doubles_calls = reader.doubles_calls;
    result.reverse_symmetry_audits = reader.audits; result.charged_amplitude_work_units = reader.work;
    result.maximum_consumed_amplitude_symmetry_error = reader.maximum_symmetry_error;
    return result;
}
}  // namespace vibeqc
