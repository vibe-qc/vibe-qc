#include "vibeqc/bounded_restricted_pair_ccsd_interaction.hpp"
#include "vibeqc/detail/sha256.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "bounded pair CCSD interaction forbids fast/finite-only math"
#endif

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Input = BoundedRestrictedPairCCSDInteractionInput;
using Inventory = BoundedRestrictedPairCCSDInteractionInventory;
using Caps = BoundedRestrictedPairCCSDInteractionCaps;
using Plan = BoundedRestrictedPairCCSDInteractionMemoryPlan;
using Diagnostics = BoundedRestrictedPairCCSDInteractionDiagnostics;
using Result = BoundedRestrictedPairCCSDInteractionResult;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53
    && FLT_EVAL_METHOD == 0, "bounded pair CCSD interaction requires exact binary64 evaluation");

U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("pair CCSD interaction count overflows uint64");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("pair CCSD interaction extent overflows uint64");
    return a * b;
}
void limit(U value, U cap, const char* message) {
    if (value > cap) throw std::length_error(message);
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("pair CCSD interaction arithmetic is nonfinite");
    return x;
}
void environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double normal = std::numeric_limits<double>::min(), one = 1.0, half = 0.5;
    if (std::fegetround() != FE_TONEAREST || tiny * one != tiny || !(normal * half > 0.0))
        throw std::invalid_argument("pair CCSD interaction requires nearest rounding and gradual underflow");
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
class Digest {
public:
    explicit Digest(const char* domain) { string(domain); u64(1); }
    void u64(U x) {
        std::array<std::uint8_t, 8> bytes{};
        for (unsigned i = 0; i < 8; ++i) bytes[i] = x >> (56 - 8 * i);
        hash_.update(bytes.data(), bytes.size());
    }
    void real(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("pair CCSD interaction input/payload is nonfinite");
        if (x == 0.0) x = 0.0;
        U bits = 0; std::memcpy(&bits, &x, sizeof(bits)); u64(bits);
    }
    void string(const std::string& x) {
        u64(x.size()); hash_.update(reinterpret_cast<const std::uint8_t*>(x.data()), x.size());
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
};
void view(const BoundedRestrictedCCSDRealView& v, U count) {
    if (v.element_count != count)
        throw std::invalid_argument("pair CCSD interaction view requires its exact element extent");
    if (!count) return;
    const auto address = reinterpret_cast<std::uintptr_t>(v.data);
    const U bytes = mul(8, count);
    if (!v.data || address % alignof(double)
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("pair CCSD interaction view pointer/alignment is invalid");
}
std::string input_hash(const Input& in) {
    Digest h("vibeqc.bounded.pair-ccsd-interaction.input");
    h.u64(in.target_dimension); h.u64(in.source_dimension); h.u64(in.source_transposed ? 1 : 0);
    for (const auto* v : {&in.k_ab, &in.j_ba, &in.overlap_ba, &in.t_bb})
        for (std::size_t i = 0; i < v->element_count; ++i) h.real(v->data[i]);
    return h.finish();
}
double product(double x, double y, Diagnostics& d, const Plan& p) {
    if (d.scalar_products >= p.scalar_products)
        throw std::logic_error("pair CCSD interaction multiplication census exhausted");
    ++d.scalar_products;
    // Store the multiplication before it participates in a sum. No fused
    // multiply-add/BLAS contraction or extended-precision accumulator.
    const double z = finite(x * y);
    if (x != 0.0 && y != 0.0 && z == 0.0) ++d.product_underflow_count;
    return z;
}
void admit(const Plan& p, const Caps& c) {
    if (!c.maximum_control_storage_bytes || !c.maximum_per_replica_inventoried_bytes
        || !c.maximum_node_inventoried_bytes || !c.maximum_work_units)
        throw std::invalid_argument("pair CCSD interaction control/worker/node/work caps must be positive");
    limit(p.target_dimension, c.maximum_target_dimension, "pair CCSD interaction target dimension cap exceeded");
    limit(p.source_dimension, c.maximum_source_dimension, "pair CCSD interaction source dimension cap exceeded");
    limit(p.borrowed_numerical_bytes, c.maximum_borrowed_numerical_bytes, "pair CCSD interaction borrowed byte cap exceeded");
    limit(p.peak_owned_numerical_bytes, c.maximum_owned_numerical_bytes, "pair CCSD interaction owned byte cap exceeded");
    limit(p.total_control_storage_bytes, c.maximum_control_storage_bytes, "pair CCSD interaction control byte cap exceeded");
    limit(p.per_replica_inventoried_bytes, c.maximum_per_replica_inventoried_bytes, "pair CCSD interaction worker byte cap exceeded");
    limit(p.required_node_inventoried_bytes, c.maximum_node_inventoried_bytes, "pair CCSD interaction node byte cap exceeded");
    limit(p.scalar_products, c.maximum_scalar_products, "pair CCSD interaction scalar product cap exceeded");
    limit(p.work_units_upper_bound, c.maximum_work_units, "pair CCSD interaction work cap exceeded");
}
} // namespace

Plan plan_bounded_restricted_pair_ccsd_interaction(U a, U b, const Inventory& inventory) {
    if (!inventory.numerical_replicas || !inventory.backend_margin_bytes_per_replica)
        throw std::invalid_argument("pair CCSD interaction replicas and backend margin must be positive");
    Plan p; p.target_dimension = a; p.source_dimension = b;
    const U aa = mul(a, a), ab = mul(a, b), bb = mul(b, b);
    p.borrowed_input_elements = add(mul(3, ab), bb);
    p.borrowed_numerical_bytes = mul(8, p.borrowed_input_elements);
    p.output_bytes = mul(8, aa); p.intermediate_bytes = mul(8, ab);
    p.peak_owned_numerical_bytes = add(p.output_bytes, p.intermediate_bytes);
    p.intermediate_contraction_terms = mul(ab, b); p.residual_contraction_terms = mul(aa, b);
    p.scalar_products = mul(2, add(p.intermediate_contraction_terms, p.residual_contraction_terms));
    p.input_scan_elements = mul(2, p.borrowed_input_elements); p.output_scan_elements = aa;
    // Fixed return/snapshot/plan/digest/scalar slots plus 8192 bytes for the
    // small receipt strings, scalar call frames and codec controls. It is a
    // logical reservation, not measurement of stack or allocator internals.
    p.fixed_control_storage_bytes = 8192 + sizeof(Input) + sizeof(Inventory) + sizeof(Caps)
        + 2 * sizeof(Plan) + sizeof(Result) + 3 * sizeof(Digest) + 4 * sizeof(Sum);
    p.total_control_storage_bytes = add(p.fixed_control_storage_bytes, inventory.other_live_control_bytes_per_replica);
    p.other_live_numerical_bytes_per_replica = inventory.other_live_numerical_bytes_per_replica;
    p.backend_margin_bytes_per_replica = inventory.backend_margin_bytes_per_replica;
    p.numerical_replicas = inventory.numerical_replicas; p.external_node_bytes = inventory.external_node_bytes;
    p.per_replica_inventoried_bytes = add(add(p.peak_owned_numerical_bytes, p.borrowed_numerical_bytes),
        add(add(p.other_live_numerical_bytes_per_replica, p.total_control_storage_bytes), p.backend_margin_bytes_per_replica));
    p.required_node_inventoried_bytes = add(p.external_node_bytes,
        mul(p.numerical_replicas, p.per_replica_inventoried_bytes));
    p.work_units_upper_bound = add(mul(512, add(4096, add(p.input_scan_elements, p.output_scan_elements))),
        mul(128, add(p.intermediate_contraction_terms, p.residual_contraction_terms)));
    const U maximum_extent = static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max());
    for (U bytes : {p.borrowed_numerical_bytes, p.output_bytes, p.intermediate_bytes, p.peak_owned_numerical_bytes})
        if (bytes > maximum_extent) throw std::overflow_error("pair CCSD interaction exceeds native address extent");
    if (aa > std::vector<double>().max_size() || ab > std::vector<double>().max_size())
        throw std::overflow_error("pair CCSD interaction exceeds native vector extent");
    // SHA-256 counts message bits in uint64. Include generous fixed codec
    // metadata before multiplying, not an unchecked size-dependent update.
    (void) mul(8, add(4096, std::max(p.borrowed_numerical_bytes, p.output_bytes)));
    return p;
}

Result::BoundedRestrictedPairCCSDInteractionResult(Result&& other) noexcept
    : live_(std::exchange(other.live_, false)), source_transposed_(other.source_transposed_),
      memory_(other.memory_), diagnostics_(other.diagnostics_), residual_(std::move(other.residual_)),
      input_(std::move(other.input_)), payload_(std::move(other.payload_)), identity_(std::move(other.identity_)) {}
void Result::require_live() const {
    if (!live_ || residual_.size() != memory_.output_bytes / 8)
        throw std::invalid_argument("pair CCSD interaction result has been consumed");
}
const Plan& Result::memory() const { require_live(); return memory_; }
const Diagnostics& Result::diagnostics() const { require_live(); return diagnostics_; }
bool Result::source_transposed() const { require_live(); return source_transposed_; }
U Result::target_dimension() const { require_live(); return memory_.target_dimension; }
U Result::source_dimension() const { require_live(); return memory_.source_dimension; }
const double* Result::residual_data() const { require_live(); return residual_.data(); }
double Result::residual(std::size_t a, std::size_t b) const {
    require_live();
    if (a >= memory_.target_dimension || b >= memory_.target_dimension)
        throw std::out_of_range("pair CCSD interaction residual index out of range");
    return residual_[a * memory_.target_dimension + b];
}
const std::string& Result::input_payload_identity_sha256() const { require_live(); return input_; }
const std::string& Result::payload_identity_sha256() const { require_live(); return payload_; }
const std::string& Result::identity_sha256() const { require_live(); return identity_; }

Result bounded_restricted_pair_ccsd_interaction(const Input& input,
    const Inventory& supplied_inventory, const Caps& supplied_caps) {
    const Input in = input; const Inventory inventory = supplied_inventory; const Caps caps = supplied_caps;
    limit(in.target_dimension, caps.maximum_target_dimension, "pair CCSD interaction target dimension cap exceeded");
    limit(in.source_dimension, caps.maximum_source_dimension, "pair CCSD interaction source dimension cap exceeded");
    const Plan p = plan_bounded_restricted_pair_ccsd_interaction(in.target_dimension, in.source_dimension, inventory);
    admit(p, caps);
    const U a = p.target_dimension, b = p.source_dimension, ab = mul(a, b);
    view(in.k_ab, ab); view(in.j_ba, ab); view(in.overlap_ba, ab); view(in.t_bb, mul(b, b));
    environment();
    const auto original_input = input_hash(in);
    Result result; result.memory_ = p; result.source_transposed_ = in.source_transposed;
    auto& d = result.diagnostics_; d.charged_work_units = p.work_units_upper_bound;
    result.residual_.resize(static_cast<std::size_t>(p.output_bytes / 8), 0.0);
    {
        std::vector<double> intermediate(static_cast<std::size_t>(ab));
        const auto t = [&](U c, U e) { return in.t_bb.data[in.source_transposed ? e * b + c : c * b + e]; };
        // Riplinger/Neese Eq.(26), first stage. The contracted source index
        // remains in B; O only maps the final d leg to target b. Projecting
        // T to A here instead would discard non-nested source components.
        for (U c = 0; c < b; ++c) for (U target = 0; target < a; ++target) {
            Sum value;
            for (U e = 0; e < b; ++e) {
                Sum spin; spin.include(product(2.0, t(c, e), d, p)); spin.include(-t(e, c));
                value.include(product(spin.total(), in.overlap_ba.data[e * a + target], d, p));
                ++d.intermediate_contraction_terms;
            }
            const double x = value.total(); intermediate[c * a + target] = x;
            d.maximum_absolute_intermediate = std::max(d.maximum_absolute_intermediate, std::abs(x));
        }
        for (U left = 0; left < a; ++left) for (U right = 0; right < a; ++right) {
            Sum value;
            for (U c = 0; c < b; ++c) {
                Sum integral; integral.include(in.k_ab.data[left * b + c]);
                integral.include(-product(0.5, in.j_ba.data[c * a + left], d, p));
                value.include(product(integral.total(), intermediate[c * a + right], d, p));
                ++d.residual_contraction_terms;
            }
            const double x = value.total(); result.residual_[left * a + right] = x;
            d.maximum_absolute_residual = std::max(d.maximum_absolute_residual, std::abs(x));
        }
    }
    if (d.scalar_products != p.scalar_products || d.intermediate_contraction_terms != p.intermediate_contraction_terms
        || d.residual_contraction_terms != p.residual_contraction_terms)
        throw std::logic_error("pair CCSD interaction completed arithmetic census mismatch");
    environment();
    if (input_hash(in) != original_input)
        throw std::invalid_argument("pair CCSD interaction input payload changed during evaluation");
    result.input_ = original_input;
    Digest payload("vibeqc.bounded.pair-ccsd-interaction.payload"); payload.u64(a); payload.u64(b);
    for (double x : result.residual_) payload.real(x);
    result.payload_ = payload.finish();
    Digest identity("vibeqc.bounded.pair-ccsd-interaction.identity");
    identity.string(result.input_); identity.string(result.payload_);
    identity.string("Riplinger2013.Eq26;real-binary64;nearest-gradual;compensated-source-first;no-weights;no-source-certificate");
    result.identity_ = identity.finish(); result.live_ = true;
    return result;
}
} // namespace vibeqc
