#include "vibeqc/bounded_restricted_ccsd_energy.hpp"

#include <cfenv>
#include <cfloat>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "bounded CCSD energy does not support fast/finite-only math"
#endif

namespace vibeqc {
namespace {
using U = std::uint64_t;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::radix == 2
    && std::numeric_limits<double>::digits == 53 && FLT_EVAL_METHOD == 0,
    "bounded CCSD energy requires binary64 without excess evaluation precision");

U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("bounded CCSD energy count overflows uint64");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("bounded CCSD energy extent overflows uint64");
    return a * b;
}
double finite(double x) {
    if (!std::isfinite(x))
        throw std::overflow_error("bounded CCSD energy numerical value is not finite");
    return x;
}
void environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double one = 1.0, half = 0.5;
    volatile double normal = std::numeric_limits<double>::min();
    if (std::fegetround() != FE_TONEAREST || tiny * one != tiny
        || !(normal * half > 0.0))
        throw std::invalid_argument("bounded CCSD energy requires round-to-nearest and gradual underflow");
}
struct Sum {
    double value = 0.0, correction = 0.0;
    void include(double x) {
        finite(x);
        const double next = finite(value + x);
        const double error = std::abs(value) >= std::abs(x)
            ? finite(finite(value - next) + x) : finite(finite(x - next) + value);
        correction = finite(correction + error);
        value = next;
    }
    double total() const { return finite(value + correction); }
};
struct Frame {
    BoundedRestrictedCCSDEnergyInput input;
    BoundedRestrictedCCSDAmplitudeProvider amplitudes;
    BoundedRestrictedCCSDIntegralProvider integrals;
    BoundedRestrictedCCSDEnergyInventory inventory;
    BoundedRestrictedCCSDEnergyCaps caps;
    BoundedRestrictedCCSDEnergyResult result;
    Sum energy, singles_energy, doubles_energy;
};
void limit(U amount, U cap, const char* message) {
    if (!cap || amount > cap) throw std::length_error(message);
}
void view_extent(const BoundedRestrictedCCSDRealView& view, U count) {
    const U bytes = mul(8, count);
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    if (!view.data || view.element_count != count || address % alignof(double)
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("bounded CCSD energy Fov view is invalid");
}
void charge(Frame& f, U& calls, U call_cap, U work, U& category_work) {
    const U next_calls = add(calls, 1), next_work = add(f.result.charged_total_work_units, work);
    limit(next_calls, call_cap, "bounded CCSD energy callback count exceeds cap");
    limit(next_work, f.caps.maximum_total_work_units, "bounded CCSD energy callback work exceeds cap");
    // Both count and work are committed before invoking the opaque provider.
    calls = next_calls;
    category_work = add(category_work, work);
    f.result.charged_total_work_units = next_work;
}
double singles(Frame& f, U i, U a) {
    charge(f, f.result.singles_calls, f.caps.maximum_singles_calls,
        f.amplitudes.maximum_singles_work_units_per_query, f.result.charged_amplitude_work_units);
    const double value = f.amplitudes.singles(i, a, f.amplitudes.context);
    environment();
    return finite(value);
}
double doubles(Frame& f, U i, U j, U a, U b) {
    charge(f, f.result.doubles_calls, f.caps.maximum_doubles_calls,
        f.amplitudes.maximum_doubles_work_units_per_query, f.result.charged_amplitude_work_units);
    const double value = f.amplitudes.doubles(i, j, a, b, f.amplitudes.context);
    environment();
    return finite(value);
}
double integral(Frame& f, U i, U a, U j, U b) {
    charge(f, f.result.integral_calls, f.caps.maximum_integral_calls,
        f.inventory.maximum_integral_work_units_per_query, f.result.charged_integral_work_units);
    const double value = f.integrals.value(i, f.input.n_occupied + a,
        j, f.input.n_occupied + b, f.integrals.context);
    environment();
    return finite(value);
}
}  // namespace

BoundedRestrictedCCSDEnergyMemoryPlan plan_bounded_restricted_ccsd_energy(
    U o, U v, const BoundedRestrictedCCSDAmplitudeProvider& amplitudes,
    const BoundedRestrictedCCSDIntegralProvider& integrals,
    const BoundedRestrictedCCSDEnergyInventory& inventory) {
    if (!o || !v || !amplitudes.maximum_singles_work_units_per_query
        || !amplitudes.maximum_doubles_work_units_per_query
        || !inventory.maximum_integral_work_units_per_query)
        throw std::invalid_argument("bounded CCSD energy dimensions and query work must be positive");
    (void) add(o, v);
    const U ov = mul(o, v), quartets = mul(ov, ov);
    BoundedRestrictedCCSDEnergyMemoryPlan p;
    p.n_occupied = o; p.n_virtual = v;
    p.borrowed_f_ov_bytes = mul(8, ov);
    if (p.borrowed_f_ov_bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::overflow_error("bounded CCSD energy Fov exceeds native address extent");
    p.amplitude_retained_numerical_bytes = amplitudes.retained_numerical_bytes;
    p.amplitude_maximum_transient_numerical_bytes = amplitudes.maximum_transient_numerical_bytes;
    p.integral_retained_numerical_bytes = integrals.retained_numerical_bytes;
    p.integral_maximum_transient_numerical_bytes = integrals.maximum_transient_numerical_bytes;
    p.other_live_numerical_bytes = inventory.other_live_numerical_bytes;
    p.total_live_numerical_bytes = p.borrowed_f_ov_bytes;
    for (U bytes : {p.amplitude_retained_numerical_bytes, p.amplitude_maximum_transient_numerical_bytes,
                   p.integral_retained_numerical_bytes, p.integral_maximum_transient_numerical_bytes,
                   p.other_live_numerical_bytes})
        p.total_live_numerical_bytes = add(p.total_live_numerical_bytes, bytes);
    // Snapshot frame, metadata planner, return slot, plus a fixed 1024-byte
    // conservative scalar/counter reservation. This does not claim to measure
    // compiler ABI frames or opaque callbacks' stack/allocator bookkeeping.
    p.fixed_inventoried_object_bytes = sizeof(Frame) + sizeof(p)
        + sizeof(BoundedRestrictedCCSDEnergyResult) + 1024U;
    p.other_live_control_bytes = inventory.other_live_control_bytes;
    p.total_control_storage_bytes = add(p.fixed_inventoried_object_bytes, p.other_live_control_bytes);
    p.singles_calls = add(ov, mul(2, quartets));
    p.doubles_calls = quartets;
    p.integral_calls = mul(2, quartets);
    p.kernel_work_units_upper_bound = add(512, add(mul(128, ov), mul(512, quartets)));
    p.amplitude_work_units_upper_bound = add(
        mul(p.singles_calls, amplitudes.maximum_singles_work_units_per_query),
        mul(p.doubles_calls, amplitudes.maximum_doubles_work_units_per_query));
    p.integral_work_units_upper_bound = mul(p.integral_calls, inventory.maximum_integral_work_units_per_query);
    p.total_work_units_upper_bound = add(p.kernel_work_units_upper_bound,
        add(p.amplitude_work_units_upper_bound, p.integral_work_units_upper_bound));
    return p;
}

BoundedRestrictedCCSDEnergyResult bounded_restricted_ccsd_energy(
    const BoundedRestrictedCCSDEnergyInput& input,
    const BoundedRestrictedCCSDAmplitudeProvider& amplitudes,
    const BoundedRestrictedCCSDIntegralProvider& integrals,
    const BoundedRestrictedCCSDEnergyInventory& inventory,
    const BoundedRestrictedCCSDEnergyCaps& caps) {
    Frame f{input, amplitudes, integrals, inventory, caps, {}, {}, {}, {}};
    f.result.memory = plan_bounded_restricted_ccsd_energy(f.input.n_occupied,
        f.input.n_virtual, f.amplitudes, f.integrals, f.inventory);
    const auto& p = f.result.memory;
    if (!f.amplitudes.singles || !f.amplitudes.doubles || !f.integrals.value)
        throw std::invalid_argument("bounded CCSD energy requires amplitude and integral callbacks");
    view_extent(f.input.f_ov, p.borrowed_f_ov_bytes / 8);
    limit(p.total_live_numerical_bytes, f.caps.maximum_total_numerical_bytes,
        "bounded CCSD energy numerical inventory exceeds cap");
    limit(p.total_control_storage_bytes, f.caps.maximum_control_storage_bytes,
        "bounded CCSD energy control inventory exceeds cap");
    limit(p.singles_calls, f.caps.maximum_singles_calls, "bounded CCSD energy singles calls exceed cap");
    limit(p.doubles_calls, f.caps.maximum_doubles_calls, "bounded CCSD energy doubles calls exceed cap");
    limit(p.integral_calls, f.caps.maximum_integral_calls, "bounded CCSD energy integral calls exceed cap");
    limit(p.total_work_units_upper_bound, f.caps.maximum_total_work_units,
        "bounded CCSD energy total work exceeds cap");
    environment();
    for (U x = 0; x < p.borrowed_f_ov_bytes / 8; ++x)
        if (!std::isfinite(f.input.f_ov.data[x]))
            throw std::invalid_argument("bounded CCSD energy Fov input must be finite");
    f.result.charged_total_work_units = p.kernel_work_units_upper_bound;
    const U o = f.input.n_occupied, v = f.input.n_virtual;
    for (U i = 0; i < o; ++i) for (U a = 0; a < v; ++a) {
        const double t = singles(f, i, a);
        const double term = finite(2.0 * finite(f.input.f_ov.data[i*v+a] * t));
        f.singles_energy.include(term); f.energy.include(term);
    }
    for (U i = 0; i < o; ++i) for (U j = 0; j < o; ++j)
        for (U a = 0; a < v; ++a) for (U b = 0; b < v; ++b) {
            const double direct = integral(f, i, a, j, b);
            const double exchange = integral(f, i, b, j, a);
            const double t = doubles(f, i, j, a, b);
            // Deliberately separate queries and sequenced statements: no
            // unspecified callback order and no target-PNO projection of t1*t1.
            const double ti = singles(f, i, a), tj = singles(f, j, b);
            const double tau = finite(t + finite(ti * tj));
            const double term = finite(finite(finite(2.0 * direct) - exchange) * tau);
            f.doubles_energy.include(term); f.energy.include(term);
        }
    if (f.result.singles_calls != p.singles_calls || f.result.doubles_calls != p.doubles_calls
        || f.result.integral_calls != p.integral_calls
        || f.result.charged_total_work_units != p.total_work_units_upper_bound)
        throw std::logic_error("bounded CCSD energy callback census differs from admitted plan");
    environment();
    f.result.correlation_energy = f.energy.total();
    f.result.singles_fock_energy = f.singles_energy.total();
    f.result.doubles_and_disconnected_energy = f.doubles_energy.total();
    return f.result;
}
}  // namespace vibeqc
