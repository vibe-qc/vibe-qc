#include "vibeqc/bounded_restricted_ccsd_target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vibeqc {
namespace {

using U = std::uint64_t;
static_assert(sizeof(double) == 8, "bounded CCSD target requires binary64");

U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("bounded CCSD target count overflows uint64");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("bounded CCSD target extent overflows uint64");
    return a * b;
}
double finite(double x) {
    if (!std::isfinite(x))
        throw std::overflow_error("bounded CCSD target numerical value is not finite");
    return x;
}
struct Sum {
    double value = 0.0, correction = 0.0;
    void include(double x) {
        finite(x);
        const double next = finite(value + x);
        correction = finite(correction + (std::abs(value) >= std::abs(x)
            ? (value - next) + x : (x - next) + value));
        value = next;
    }
    double total() const { return finite(value + correction); }
};

void view_extent(const BoundedRestrictedCCSDRealView& view, U count) {
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    const U bytes = mul(8, count);
    if (!view.data || view.element_count < count
        || address % alignof(double) != 0
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("bounded CCSD target borrowed input view is invalid");
}
void scan(const BoundedRestrictedCCSDRealView& view, U count) {
    for (U k = 0; k < count; ++k) {
        if (!std::isfinite(view.data[k]))
            throw std::invalid_argument("bounded CCSD target input must be finite");
    }
}

BoundedRestrictedCCSDTargetMemoryPlan plan_kernel(
    U o, U v, U retained, U transient, bool dense_amplitudes) {
    if (!o || !v)
        throw std::invalid_argument("bounded CCSD target local dimensions must be positive");
    const U o2 = mul(o, o), v2 = mul(v, v), ov = mul(o, v);
    const U v3 = mul(v2, v), v4 = mul(v3, v), o2v2 = mul(o2, v2);
    (void) add(o, v);
    BoundedRestrictedCCSDTargetMemoryPlan p;
    p.n_occupied = o; p.n_virtual = v;
    const U inputs = dense_amplitudes
        ? add(add(o2v2, mul(2, ov)), add(o2, v2))
        : add(ov, add(o2, v2));
    p.borrowed_input_bytes = mul(8, inputs);
    p.provider_retained_numerical_bytes = retained;
    p.provider_maximum_transient_numerical_bytes = transient;
    p.output_bytes = mul(8, add(v2, v));
    p.workspace_bytes = mul(8, add(add(v2, mul(2, o)), mul(3, ov)));
    p.peak_owned_numerical_bytes = add(p.output_bytes, p.workspace_bytes);
    p.total_live_numerical_bytes = add(add(p.peak_owned_numerical_bytes,
        p.borrowed_input_bytes), add(retained, transient));
    // Both target-column passes are retained even for i==j, giving one
    // target-independent bound. The shared-ring implementation uses fewer
    // calls than this bound by reusing integral values inside each m,e cell.
    p.integral_calls_upper_bound = add(add(
        mul(o2, add(add(1, mul(6, v)), add(mul(14, v2), mul(6, v3)))),
        mul(o, add(mul(10, v2), mul(13, v3)))),
        add(mul(2, v3), mul(v4, add(1, mul(2, o)))));
    const U loops = add(add(add(o2v2, mul(o, v2)), add(v3, ov)),
                        add(add(v2, o), v));
    p.kernel_work_units_upper_bound = add(inputs, add(
        mul(128, p.integral_calls_upper_bound), mul(64, loops)));
    const U extent_limit = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max());
    if (p.borrowed_input_bytes > extent_limit || p.peak_owned_numerical_bytes > extent_limit
        || p.output_bytes / 8 > std::vector<double>().max_size()
        || p.workspace_bytes / 8 > std::vector<double>().max_size())
        throw std::overflow_error("bounded CCSD target exceeds native numerical address extent");
    return p;
}

}  // namespace

BoundedRestrictedCCSDTargetMemoryPlan plan_bounded_restricted_ccsd_target(
    U o, U v, U retained, U transient) {
    return plan_kernel(o, v, retained, transient, true);
}

BoundedRestrictedCCSDTargetAccessorMemoryPlan plan_bounded_restricted_ccsd_target_accessor(
    U o, U v, const BoundedRestrictedCCSDAmplitudeProvider& amplitudes,
    U retained, U transient) {
    if (!amplitudes.maximum_singles_work_units_per_query
        || !amplitudes.maximum_doubles_work_units_per_query)
        throw std::invalid_argument("bounded CCSD amplitude per-query work ceilings must be positive");
    BoundedRestrictedCCSDTargetAccessorMemoryPlan p;
    p.kernel = plan_kernel(o, v, retained, transient, false);
    p.amplitude_retained_numerical_bytes = amplitudes.retained_numerical_bytes;
    p.amplitude_maximum_transient_numerical_bytes = amplitudes.maximum_transient_numerical_bytes;
    p.kernel.total_live_numerical_bytes = add(p.kernel.total_live_numerical_bytes,
        add(p.amplitude_retained_numerical_bytes, p.amplitude_maximum_transient_numerical_bytes));
    const U o2 = mul(o, o), v2 = mul(v, v), v3 = mul(v2, v), v4 = mul(v3, v);
    // Exact source-level query census of the common arithmetic below. Both
    // occupied-column and ring passes execute even for equal target indices.
    // Single queries by degree: 2ov4+2v4+10o2v3+15ov3+15o2v2+
    // 5ov2+4o2v+5ov+2v3+v2. Double queries: 9o2v3+6o2v2+
    // 8ov3+4ov2+v4+2v3. No zero-amplitude/coupling shortcut is taken.
    for (U term : {mul(2, mul(o, v4)), mul(2, v4), mul(10, mul(o2, v3)),
                   mul(15, mul(o, v3)), mul(15, mul(o2, v2)), mul(5, mul(o, v2)),
                   mul(4, mul(o2, v)), mul(5, mul(o, v)), mul(2, v3), v2})
        p.singles_calls_upper_bound = add(p.singles_calls_upper_bound, term);
    for (U term : {mul(9, mul(o2, v3)), mul(6, mul(o2, v2)), mul(8, mul(o, v3)),
                   mul(4, mul(o, v2)), v4, mul(2, v3)})
        p.doubles_calls_upper_bound = add(p.doubles_calls_upper_bound, term);
    p.amplitude_work_units_upper_bound = add(
        mul(p.singles_calls_upper_bound, amplitudes.maximum_singles_work_units_per_query),
        mul(p.doubles_calls_upper_bound, amplitudes.maximum_doubles_work_units_per_query));
    // Count/extent/cap/finite checks in each accessor wrapper are native work,
    // additional to declared callback work and the unchanged algebra bound.
    p.kernel.kernel_work_units_upper_bound = add(p.kernel.kernel_work_units_upper_bound,
        mul(16, add(p.singles_calls_upper_bound, p.doubles_calls_upper_bound)));
    return p;
}

namespace {

void exclude_bare_particle_hole(BoundedRestrictedCCSDTargetMemoryPlan& plan) {
    const U calls = mul(6, mul(plan.n_occupied, mul(plan.n_virtual, plan.n_virtual)));
    const U work = mul(128, calls);
    if (calls > plan.integral_calls_upper_bound || work > plan.kernel_work_units_upper_bound)
        throw std::logic_error("bounded CCSD bare particle-hole census exceeds full target plan");
    plan.bare_particle_hole_excluded = true;
    plan.omitted_bare_integral_calls = calls; plan.omitted_bare_kernel_work_units = work;
    plan.integral_calls_upper_bound -= calls; plan.kernel_work_units_upper_bound -= work;
}

// Arithmetic only. Dense and accessor paths deliberately retain the same
// expressions, loop/sum order and integral-provider invocation sites.
// The true specialization is the original FULL target arithmetic. The false
// specialization excludes only the seed block explicitly marked below.
template<bool IncludeBareParticleHole, class Amplitudes>
BoundedRestrictedCCSDTargetResult target_arithmetic(
    const BoundedRestrictedCCSDTargetOperatorInput& in,
    const BoundedRestrictedCCSDIntegralProvider& provider,
    const BoundedRestrictedCCSDTargetCaps& caps,
    const BoundedRestrictedCCSDTargetMemoryPlan& plan,
    Amplitudes& amplitudes) {
    const U o = in.n_occupied, v = in.n_virtual, ov = o * v, v2 = v * v;
    const U i = in.target_i, j = in.target_j;

    BoundedRestrictedCCSDTargetResult result;
    result.memory = plan; result.target_i = i; result.target_j = j;
    result.singles.resize(static_cast<std::size_t>(v));
    result.doubles.resize(static_cast<std::size_t>(v2));
    std::vector<double> workspace(static_cast<std::size_t>(plan.workspace_bytes / 8));
    double* fa = workspace.data();                  // [a,e], later Fh
    double* fi = fa + v2;                           // two [m] columns
    double* scratch = fi + 2 * o;                   // Fme, then W1/W2/WX
    const auto t1 = [&](U m, U a) { return amplitudes.singles(m, a); };
    const auto t2 = [&](U m, U n, U a, U b) {
        return amplitudes.doubles(m, n, a, b);
    };
    const auto tau = [&](U m, U n, U a, U b) {
        return finite(t2(m, n, a, b) + finite(t1(m, a) * t1(n, b)));
    };
    const auto taut = [&](U m, U n, U a, U b) {
        return finite(t2(m, n, a, b) + 0.5 * finite(t1(m, a) * t1(n, b)));
    };
    const auto eri = [&](U p, U q, U r, U s) {
        if (result.integral_calls >= caps.maximum_integral_calls
            || result.integral_calls >= plan.integral_calls_upper_bound)
            throw std::length_error("bounded CCSD target exhausted its integral callback count");
        ++result.integral_calls;
        return finite(provider.value(p, q, r, s, provider.context));
    };
    const auto accumulate_r2 = [&](U a, U b, double value) {
        auto& output = result.doubles[a * v + b];
        output = finite(output + finite(value));
    };

    // SGWB (3)-(5), closed-shell spin integration. Keep only two F_mi columns.
    for (U m = 0; m < o; ++m) for (U e = 0; e < v; ++e) {
        Sum value; value.include(in.f_ov.data[m * v + e]);
        for (U n = 0; n < o; ++n) for (U f = 0; f < v; ++f)
            value.include(t1(n, f) * (2.0 * eri(m, o + e, n, o + f)
                                      - eri(m, o + f, n, o + e)));
        scratch[m * v + e] = value.total();
    }
    for (U a = 0; a < v; ++a) for (U e = 0; e < v; ++e) {
        Sum value; value.include(in.f_vv.data[a * v + e]);
        for (U m = 0; m < o; ++m) {
            value.include(-0.5 * in.f_ov.data[m * v + e] * t1(m, a));
            for (U f = 0; f < v; ++f)
                value.include(t1(m, f) * (2.0 * eri(m, o + f, o + a, o + e)
                                         - eri(m, o + e, o + a, o + f)));
        }
        for (U m = 0; m < o; ++m) for (U n = 0; n < o; ++n) for (U f = 0; f < v; ++f)
            value.include(-taut(m, n, a, f) * (2.0 * eri(m, o + e, n, o + f)
                                              - eri(m, o + f, n, o + e)));
        fa[a * v + e] = value.total();
    }
    for (U column = 0; column < 2; ++column) {
        const U target = column == 0 ? i : j;
        for (U m = 0; m < o; ++m) {
            Sum value; value.include(in.f_oo.data[m * o + target]);
            for (U e = 0; e < v; ++e)
                value.include(0.5 * t1(target, e) * in.f_ov.data[m * v + e]);
            for (U n = 0; n < o; ++n) for (U e = 0; e < v; ++e)
                value.include(t1(n, e) * (2.0 * eri(m, target, n, o + e)
                                         - eri(n, target, m, o + e)));
            for (U n = 0; n < o; ++n) for (U e = 0; e < v; ++e) for (U f = 0; f < v; ++f)
                value.include(taut(target, n, e, f) * (2.0 * eri(m, o + e, n, o + f)
                                                      - eri(m, o + f, n, o + e)));
            fi[column * o + m] = value.total();
        }
    }

    // SGWB (1), one singles row, before overwriting F with its doubles form.
    for (U a = 0; a < v; ++a) {
        Sum value; value.include(in.f_ov.data[i * v + a]);
        for (U e = 0; e < v; ++e) value.include(t1(i, e) * fa[a * v + e]);
        for (U m = 0; m < o; ++m) value.include(-t1(m, a) * fi[m]);
        for (U m = 0; m < o; ++m) for (U e = 0; e < v; ++e)
            value.include((2.0 * t2(i, m, a, e) - t2(m, i, a, e)) * scratch[m * v + e]);
        for (U n = 0; n < o; ++n) for (U f = 0; f < v; ++f)
            value.include(t1(n, f) * (2.0 * eri(n, o + f, i, o + a)
                                     - eri(n, i, o + a, o + f)));
        for (U m = 0; m < o; ++m) for (U e = 0; e < v; ++e) for (U f = 0; f < v; ++f)
            value.include((2.0 * t2(i, m, e, f) - t2(m, i, e, f))
                          * eri(m, o + f, o + a, o + e));
        for (U m = 0; m < o; ++m) for (U n = 0; n < o; ++n) for (U e = 0; e < v; ++e)
            value.include(-(2.0 * t2(m, n, a, e) - t2(n, m, a, e))
                          * eri(m, i, n, o + e));
        result.singles[a] = value.total();
    }
    for (U b = 0; b < v; ++b) for (U e = 0; e < v; ++e) {
        Sum value; value.include(fa[b * v + e]);
        for (U m = 0; m < o; ++m) value.include(-0.5 * t1(m, b) * scratch[m * v + e]);
        fa[b * v + e] = value.total();
    }
    for (U column = 0; column < 2; ++column) {
        const U target = column == 0 ? i : j;
        for (U m = 0; m < o; ++m) {
            Sum value; value.include(fi[column * o + m]);
            for (U e = 0; e < v; ++e) value.include(0.5 * t1(target, e) * scratch[m * v + e]);
            fi[column * o + m] = value.total();
        }
    }

    // Merge the two half-tau terms from W_mnij and W_abef before contraction:
    // sum_mn tau_mn^ab [H_mnij + sum_ef tau_ij^ef (me|nf)]. No W/tau tensor.
    for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b)
        result.doubles[a * v + b] = eri(i, o + a, j, o + b);
    for (U m = 0; m < o; ++m) for (U n = 0; n < o; ++n) {
        Sum value; value.include(eri(m, i, n, j));
        for (U e = 0; e < v; ++e) {
            value.include(t1(j, e) * eri(m, i, n, o + e));
            value.include(t1(i, e) * eri(n, j, m, o + e));
        }
        for (U e = 0; e < v; ++e) for (U f = 0; f < v; ++f)
            value.include(tau(i, j, e, f) * eri(m, o + e, n, o + f));
        const double intermediate = value.total();
        for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b)
            accumulate_r2(a, b, tau(m, n, a, b) * intermediate);
    }
    // Remaining particle-particle ladder, generated and consumed scalarwise.
    for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) {
        Sum ladder;
        for (U e = 0; e < v; ++e) for (U f = 0; f < v; ++f) {
            Sum value; value.include(eri(o + a, o + e, o + b, o + f));
            for (U m = 0; m < o; ++m) {
                value.include(-t1(m, b) * eri(m, o + f, o + a, o + e));
                value.include(-t1(m, a) * eri(m, o + e, o + b, o + f));
            }
            ladder.include(tau(i, j, e, f) * value.total());
        }
        accumulate_r2(a, b, ladder.total());
    }

    // Fme is dead. Reuse its allocation for one-b ring columns, including
    // WX's DIFFERENT target occupied index. The two Ph halves write directly
    // into (a,b) and (b,a); no full-shaped or even target-sized half tensor.
    double* w1 = scratch; double* w2 = scratch + ov; double* wx = scratch + 2 * ov;
    for (U pass = 0; pass < 2; ++pass) {
        const U left = pass == 0 ? i : j, right = pass == 0 ? j : i;
        const U right_column = pass == 0 ? 1 : 0;
        for (U b = 0; b < v; ++b) {
            for (U m = 0; m < o; ++m) for (U e = 0; e < v; ++e) {
                Sum s1, s2, sx;
                if constexpr (IncludeBareParticleHole) {
                    const double direct = eri(m, o + e, right, o + b);
                    const double right_exchange = eri(m, right, o + b, o + e);
                    s1.include(direct); s2.include(direct); s2.include(-right_exchange);
                    sx.include(-eri(m, left, o + b, o + e));
                }
                for (U f = 0; f < v; ++f) {
                    const double vd = eri(m, o + e, o + b, o + f);
                    const double vx = eri(m, o + f, o + b, o + e);
                    s1.include(t1(right, f) * vd);
                    s2.include(t1(right, f) * (vd - vx));
                    sx.include(-t1(left, f) * vx);
                }
                for (U n = 0; n < o; ++n) {
                    const double vd = eri(n, right, m, o + e);
                    const double vx = eri(m, right, n, o + e);
                    s1.include(-t1(n, b) * vd);
                    s2.include(-t1(n, b) * (vd - vx));
                    sx.include(t1(n, b) * eri(m, left, n, o + e));
                }
                for (U n = 0; n < o; ++n) for (U f = 0; f < v; ++f) {
                    const double t_njfb = t2(n, right, f, b);
                    const double t_jnfb = t2(right, n, f, b);
                    const double t_jnbf = t2(right, n, b, f);
                    const double t1t1 = finite(t1(right, f) * t1(n, b));
                    const double vd = eri(m, o + e, n, o + f);
                    const double vx = eri(m, o + f, n, o + e);
                    s1.include((t_njfb - 0.5 * t_jnfb - t1t1) * vd - 0.5 * t_njfb * vx);
                    s2.include(-(0.5 * (t_jnfb - t_jnbf) + t1t1) * (vd - vx) + 0.5 * t_njfb * vd);
                    sx.include((0.5 * t2(left, n, f, b) + finite(t1(left, f) * t1(n, b))) * vx);
                }
                w1[m * v + e] = s1.total(); w2[m * v + e] = s2.total(); wx[m * v + e] = sx.total();
            }
            for (U a = 0; a < v; ++a) {
                Sum value;
                for (U e = 0; e < v; ++e) value.include(t2(left, right, a, e) * fa[b * v + e]);
                for (U m = 0; m < o; ++m) value.include(-t2(left, m, a, b) * fi[right_column * o + m]);
                for (U m = 0; m < o; ++m) for (U e = 0; e < v; ++e) {
                    const double t_im = t2(left, m, a, e);
                    value.include((t_im - t2(m, left, a, e)) * w1[m * v + e]);
                    value.include(t_im * w2[m * v + e]);
                    value.include(t2(m, right, a, e) * wx[m * v + e]);
                    value.include(-t1(left, e) * t1(m, a) * eri(m, o + e, right, o + b));
                    value.include(-t1(right, e) * t1(m, a) * eri(m, left, o + b, o + e));
                }
                for (U e = 0; e < v; ++e)
                    value.include(t1(left, e) * eri(right, o + b, o + a, o + e));
                for (U m = 0; m < o; ++m)
                    value.include(-t1(m, a) * eri(m, left, right, o + b));
                if (pass == 0) accumulate_r2(a, b, value.total());
                else accumulate_r2(b, a, value.total());
            }
        }
    }
    return result;
}

void cap_kernel(const BoundedRestrictedCCSDTargetMemoryPlan& plan,
                const BoundedRestrictedCCSDTargetCaps& caps) {
    if (!caps.maximum_owned_numerical_bytes
        || caps.maximum_owned_numerical_bytes < plan.peak_owned_numerical_bytes
        || !caps.maximum_total_numerical_bytes
        || caps.maximum_total_numerical_bytes < plan.total_live_numerical_bytes)
        throw std::length_error("bounded CCSD target numerical byte cap is missing or exceeded");
    if (!caps.maximum_integral_calls || caps.maximum_integral_calls < plan.integral_calls_upper_bound
        || !caps.maximum_kernel_work_units
        || caps.maximum_kernel_work_units < plan.kernel_work_units_upper_bound)
        throw std::length_error("bounded CCSD target integral or kernel work cap is missing or exceeded");
}
void operator_extents(const BoundedRestrictedCCSDTargetOperatorInput& in) {
    const U o = in.n_occupied, v = in.n_virtual;
    if (in.target_i >= o || in.target_j >= o)
        throw std::out_of_range("bounded CCSD target occupied index is out of range");
    view_extent(in.f_oo, o * o); view_extent(in.f_vv, v * v); view_extent(in.f_ov, o * v);
}
void scan_operator(const BoundedRestrictedCCSDTargetOperatorInput& in) {
    const U o = in.n_occupied, v = in.n_virtual;
    scan(in.f_oo, o * o); scan(in.f_vv, v * v); scan(in.f_ov, o * v);
}
struct DenseAmplitudes {
    const double* t1;
    const double* t2;
    U o, v, v2;
    double singles(U i, U a) const { return t1[i * v + a]; }
    double doubles(U i, U j, U a, U b) const { return t2[(i * o + j) * v2 + a * v + b]; }
};
struct AccessorAmplitudes {
    const BoundedRestrictedCCSDAmplitudeProvider provider;
    const BoundedRestrictedCCSDTargetAccessorCaps caps;
    const BoundedRestrictedCCSDTargetAccessorMemoryPlan plan;
    BoundedRestrictedCCSDTargetAccessorResult& result;
    void charge(U& count, U cap, U bound, U work) {
        if (count >= cap || count >= bound)
            throw std::length_error("bounded CCSD target exhausted its amplitude callback count");
        const U next = add(result.charged_amplitude_work_units, work);
        if (next > caps.maximum_amplitude_work_units || next > plan.amplitude_work_units_upper_bound)
            throw std::length_error("bounded CCSD target exhausted its amplitude work cap");
        ++count;
        result.charged_amplitude_work_units = next;
    }
    double singles(U i, U a) {
        if (i >= plan.kernel.n_occupied || a >= plan.kernel.n_virtual)
            throw std::out_of_range("bounded CCSD amplitude singles query is out of range");
        charge(result.singles_calls, caps.maximum_singles_calls, plan.singles_calls_upper_bound,
               provider.maximum_singles_work_units_per_query);
        return finite(provider.singles(i, a, provider.context));
    }
    double doubles(U i, U j, U a, U b) {
        if (i >= plan.kernel.n_occupied || j >= plan.kernel.n_occupied
            || a >= plan.kernel.n_virtual || b >= plan.kernel.n_virtual)
            throw std::out_of_range("bounded CCSD amplitude doubles query is out of range");
        charge(result.doubles_calls, caps.maximum_doubles_calls, plan.doubles_calls_upper_bound,
               provider.maximum_doubles_work_units_per_query);
        return finite(provider.doubles(i, j, a, b, provider.context));
    }
};

}  // namespace

BoundedRestrictedCCSDTargetResult bounded_restricted_ccsd_target_residual(
    const BoundedRestrictedCCSDTargetInput& supplied_input,
    const BoundedRestrictedCCSDIntegralProvider& supplied_provider,
    const BoundedRestrictedCCSDTargetCaps& supplied_caps) {
    const auto in = supplied_input;
    const auto provider = supplied_provider;
    const auto caps = supplied_caps;
    const auto plan = plan_bounded_restricted_ccsd_target(in.n_occupied, in.n_virtual,
        provider.retained_numerical_bytes, provider.maximum_transient_numerical_bytes);
    cap_kernel(plan, caps);
    if (!provider.value)
        throw std::invalid_argument("bounded CCSD target requires an integral callback");
    const U o = in.n_occupied, v = in.n_virtual, ov = o * v, v2 = v * v;
    if (in.target_i >= o || in.target_j >= o)
        throw std::out_of_range("bounded CCSD target occupied index is out of range");
    view_extent(in.t1, ov); view_extent(in.t2, o * o * v2);
    const BoundedRestrictedCCSDTargetOperatorInput op{
        o, v, in.target_i, in.target_j, in.f_oo, in.f_vv, in.f_ov};
    operator_extents(op);
    scan(in.t1, ov); scan(in.t2, o * o * v2);
    scan_operator(op);
    DenseAmplitudes amplitudes{in.t1.data, in.t2.data, o, v, v2};
    return target_arithmetic<true>(op, provider, caps, plan, amplitudes);
}

BoundedRestrictedCCSDTargetAccessorResult bounded_restricted_ccsd_target_residual_accessor(
    const BoundedRestrictedCCSDTargetOperatorInput& supplied_input,
    const BoundedRestrictedCCSDAmplitudeProvider& supplied_amplitudes,
    const BoundedRestrictedCCSDIntegralProvider& supplied_integrals,
    const BoundedRestrictedCCSDTargetAccessorCaps& supplied_caps) {
    const auto in = supplied_input;
    const auto amplitudes = supplied_amplitudes;
    const auto integrals = supplied_integrals;
    const auto caps = supplied_caps;
    const auto plan = plan_bounded_restricted_ccsd_target_accessor(in.n_occupied, in.n_virtual,
        amplitudes, integrals.retained_numerical_bytes, integrals.maximum_transient_numerical_bytes);
    cap_kernel(plan.kernel, caps.kernel);
    if (!caps.maximum_singles_calls || caps.maximum_singles_calls < plan.singles_calls_upper_bound
        || !caps.maximum_doubles_calls || caps.maximum_doubles_calls < plan.doubles_calls_upper_bound
        || !caps.maximum_amplitude_work_units
        || caps.maximum_amplitude_work_units < plan.amplitude_work_units_upper_bound)
        throw std::length_error("bounded CCSD target amplitude count or work cap is missing or exceeded");
    if (!amplitudes.singles || !amplitudes.doubles)
        throw std::invalid_argument("bounded CCSD target requires singles and doubles amplitude callbacks");
    if (!integrals.value)
        throw std::invalid_argument("bounded CCSD target requires an integral callback");
    operator_extents(in);
    scan_operator(in);
    BoundedRestrictedCCSDTargetAccessorResult result;
    result.memory = plan;
    AccessorAmplitudes reader{amplitudes, caps, plan, result};
    result.target = target_arithmetic<true>(in, integrals, caps.kernel, plan.kernel, reader);
    return result;
}

BoundedRestrictedCCSDTargetMemoryPlan plan_bounded_restricted_ccsd_target_without_bare_particle_hole(
    U o, U v, U retained, U transient) {
    auto plan = plan_bounded_restricted_ccsd_target(o, v, retained, transient);
    exclude_bare_particle_hole(plan); return plan;
}

BoundedRestrictedCCSDTargetAccessorMemoryPlan plan_bounded_restricted_ccsd_target_accessor_without_bare_particle_hole(
    U o, U v, const BoundedRestrictedCCSDAmplitudeProvider& amplitudes, U retained, U transient) {
    auto plan = plan_bounded_restricted_ccsd_target_accessor(o, v, amplitudes, retained, transient);
    exclude_bare_particle_hole(plan.kernel); return plan;
}

BoundedRestrictedCCSDTargetResult bounded_restricted_ccsd_target_residual_without_bare_particle_hole(
    const BoundedRestrictedCCSDTargetInput& supplied_input,
    const BoundedRestrictedCCSDIntegralProvider& supplied_provider,
    const BoundedRestrictedCCSDTargetCaps& supplied_caps) {
    // Same admission contract and dense reader as the full API, without
    // changing that API's dispatch, default plans or invocation order.
    const auto in = supplied_input;
    const auto provider = supplied_provider;
    const auto caps = supplied_caps;
    const auto plan = plan_bounded_restricted_ccsd_target_without_bare_particle_hole(in.n_occupied, in.n_virtual,
        provider.retained_numerical_bytes, provider.maximum_transient_numerical_bytes);
    cap_kernel(plan, caps);
    if (!provider.value) throw std::invalid_argument("bounded CCSD target requires an integral callback");
    const U o = in.n_occupied, v = in.n_virtual, ov = o * v, v2 = v * v;
    if (in.target_i >= o || in.target_j >= o)
        throw std::out_of_range("bounded CCSD target occupied index is out of range");
    view_extent(in.t1, ov); view_extent(in.t2, o * o * v2);
    const BoundedRestrictedCCSDTargetOperatorInput op{
        o, v, in.target_i, in.target_j, in.f_oo, in.f_vv, in.f_ov};
    operator_extents(op);
    scan(in.t1, ov); scan(in.t2, o * o * v2); scan_operator(op);
    DenseAmplitudes amplitudes{in.t1.data, in.t2.data, o, v, v2};
    return target_arithmetic<false>(op, provider, caps, plan, amplitudes);
}

BoundedRestrictedCCSDTargetAccessorResult bounded_restricted_ccsd_target_residual_accessor_without_bare_particle_hole(
    const BoundedRestrictedCCSDTargetOperatorInput& supplied_input,
    const BoundedRestrictedCCSDAmplitudeProvider& supplied_amplitudes,
    const BoundedRestrictedCCSDIntegralProvider& supplied_integrals,
    const BoundedRestrictedCCSDTargetAccessorCaps& supplied_caps) {
    const auto in = supplied_input;
    const auto amplitudes = supplied_amplitudes;
    const auto integrals = supplied_integrals;
    const auto caps = supplied_caps;
    const auto plan = plan_bounded_restricted_ccsd_target_accessor_without_bare_particle_hole(in.n_occupied, in.n_virtual,
        amplitudes, integrals.retained_numerical_bytes, integrals.maximum_transient_numerical_bytes);
    cap_kernel(plan.kernel, caps.kernel);
    if (!caps.maximum_singles_calls || caps.maximum_singles_calls < plan.singles_calls_upper_bound
        || !caps.maximum_doubles_calls || caps.maximum_doubles_calls < plan.doubles_calls_upper_bound
        || !caps.maximum_amplitude_work_units
        || caps.maximum_amplitude_work_units < plan.amplitude_work_units_upper_bound)
        throw std::length_error("bounded CCSD target amplitude count or work cap is missing or exceeded");
    if (!amplitudes.singles || !amplitudes.doubles)
        throw std::invalid_argument("bounded CCSD target requires singles and doubles amplitude callbacks");
    if (!integrals.value) throw std::invalid_argument("bounded CCSD target requires an integral callback");
    operator_extents(in); scan_operator(in);
    BoundedRestrictedCCSDTargetAccessorResult result; result.memory = plan;
    AccessorAmplitudes reader{amplitudes, caps, plan, result};
    result.target = target_arithmetic<false>(in, integrals, caps.kernel, plan.kernel, reader);
    return result;
}

}  // namespace vibeqc
