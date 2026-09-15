#include "vibeqc/bounded_restricted_pair_ccsd_particle_hole.hpp"
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
#error "bounded pair CCSD particle-hole forbids fast/finite-only math"
#endif

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Source = BoundedRestrictedPairCCSDParticleHoleSource;
using Inventory = BoundedRestrictedPairCCSDParticleHoleInventory;
using Caps = BoundedRestrictedPairCCSDParticleHoleCaps;
using SourcePlan = BoundedRestrictedPairCCSDParticleHoleSourcePlan;
using Plan = BoundedRestrictedPairCCSDParticleHoleMemoryPlan;
using Diagnostics = BoundedRestrictedPairCCSDParticleHoleDiagnostics;
using Result = BoundedRestrictedPairCCSDParticleHoleResult;
using Accumulator = BoundedRestrictedPairCCSDParticleHoleAccumulator;
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53
    && FLT_EVAL_METHOD == 0, "particle-hole requires exact binary64 evaluation");
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("particle-hole count overflows uint64");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("particle-hole extent overflows uint64");
    return a * b;
}
void limit(U value, U cap, const char* message) {
    if (value > cap) throw std::length_error(message);
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("particle-hole arithmetic is nonfinite");
    return x;
}
void environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double normal = std::numeric_limits<double>::min(), one = 1.0, half = 0.5;
    if (std::fegetround() != FE_TONEAREST || tiny * one != tiny || !(normal * half > 0.0))
        throw std::invalid_argument("particle-hole requires nearest rounding and gradual underflow");
}
void include(double& value, double& correction, double x) {
    finite(x);
    const double next = finite(value + x);
    const double error = std::abs(value) >= std::abs(x)
        ? finite(finite(value - next) + x) : finite(finite(x - next) + value);
    correction = finite(correction + error); value = next;
}
struct Sum {
    double value = 0.0, correction = 0.0;
    void include(double x) { ::vibeqc::include(value, correction, x); }
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
        if (!std::isfinite(x)) throw std::invalid_argument("particle-hole input/payload is nonfinite");
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
        throw std::invalid_argument("particle-hole view requires its exact element extent");
    if (!count) return;
    const auto address = reinterpret_cast<std::uintptr_t>(v.data);
    const U bytes = mul(8, count);
    if (!v.data || address % alignof(double)
        || bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("particle-hole view pointer/alignment is invalid");
}
std::string source_hash(const Source& s, U a) {
    Digest h("vibeqc.bounded.pair-ccsd-particle-hole.source");
    h.u64(a); h.u64(s.source_dimension); h.u64(s.leg); h.u64(s.occupied_index);
    h.u64(s.source_transposed ? 1 : 0);
    for (const auto* v : {&s.k_ba, &s.j_ba, &s.overlap_ba, &s.t_bb})
        for (std::size_t i = 0; i < v->element_count; ++i) h.real(v->data[i]);
    return h.finish();
}
double product(double x, double y, Diagnostics& d, U stop) {
    if (d.scalar_products >= stop) throw std::logic_error("particle-hole multiplication census exhausted");
    ++d.scalar_products;
    const double z = finite(x * y);
    if (x != 0.0 && y != 0.0 && z == 0.0) ++d.product_underflow_count;
    return z;
}
void admit(const Plan& p, const Caps& c) {
    if (!c.maximum_occupied_count || !c.maximum_control_storage_bytes
        || !c.maximum_per_replica_inventoried_bytes || !c.maximum_node_inventoried_bytes || !c.maximum_work_units)
        throw std::invalid_argument("particle-hole occupied/control/worker/node/work caps must be positive");
    limit(p.target_dimension, c.maximum_target_dimension, "particle-hole target dimension cap exceeded");
    limit(p.maximum_source_dimension, c.maximum_source_dimension, "particle-hole source dimension cap exceeded");
    limit(p.occupied_count, c.maximum_occupied_count, "particle-hole occupied count cap exceeded");
    limit(p.maximum_borrowed_numerical_bytes, c.maximum_borrowed_numerical_bytes, "particle-hole borrowed byte cap exceeded");
    limit(p.peak_owned_numerical_bytes, c.maximum_owned_numerical_bytes, "particle-hole owned byte cap exceeded");
    limit(p.total_control_storage_bytes, c.maximum_control_storage_bytes, "particle-hole control byte cap exceeded");
    limit(p.per_replica_inventoried_bytes, c.maximum_per_replica_inventoried_bytes, "particle-hole worker byte cap exceeded");
    limit(p.required_node_inventoried_bytes, c.maximum_node_inventoried_bytes, "particle-hole node byte cap exceeded");
    limit(p.scalar_products_upper_bound, c.maximum_scalar_products, "particle-hole scalar product cap exceeded");
    limit(p.work_units_upper_bound, c.maximum_work_units, "particle-hole work cap exceeded");
}
void address_extent(U bytes) {
    if (bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::overflow_error("particle-hole exceeds native address extent");
}
} // namespace

SourcePlan plan_bounded_restricted_pair_ccsd_particle_hole_source(U a, U b) {
    SourcePlan p; p.target_dimension = a; p.source_dimension = b;
    const U ab = mul(a, b), bb = mul(b, b);
    p.borrowed_input_elements = add(mul(3, ab), bb);
    p.borrowed_numerical_bytes = mul(8, p.borrowed_input_elements);
    p.intermediate_contraction_terms = mul(ab, b);
    p.projection_contraction_terms = mul(mul(a, a), b);
    p.scalar_products = add(mul(4, p.intermediate_contraction_terms), mul(2, p.projection_contraction_terms));
    p.input_scan_elements = mul(2, p.borrowed_input_elements);
    p.work_units_upper_bound = add(mul(512, add(4096, p.input_scan_elements)),
        mul(128, add(p.scalar_products, add(ab, mul(a, a)))));
    address_extent(p.borrowed_numerical_bytes);
    (void) mul(8, add(4096, p.borrowed_numerical_bytes)); // SHA message bit extent
    return p;
}

Plan plan_bounded_restricted_pair_ccsd_particle_hole(U a, U o, U maximum_rank, const Inventory& inventory) {
    if (!o || !inventory.numerical_replicas || !inventory.backend_margin_bytes_per_replica)
        throw std::invalid_argument("particle-hole occupied count, replicas and backend margin must be positive");
    const auto source = plan_bounded_restricted_pair_ccsd_particle_hole_source(a, maximum_rank);
    Plan p; p.target_dimension = a; p.occupied_count = o; p.maximum_source_dimension = maximum_rank;
    p.source_slots = mul(2, o); p.maximum_borrowed_numerical_bytes = source.borrowed_numerical_bytes;
    const U aa = mul(a, a), am = mul(a, maximum_rank);
    p.output_bytes = mul(8, aa); p.compensation_bytes = p.output_bytes; p.intermediate_bytes = mul(8, am);
    p.peak_owned_numerical_bytes = add(mul(2, p.output_bytes), p.intermediate_bytes);
    p.scalar_products_upper_bound = mul(p.source_slots, source.scalar_products);
    p.input_scan_elements_upper_bound = mul(p.source_slots, source.input_scan_elements);
    p.finish_work_units = mul(512, add(4096, aa));
    p.work_units_upper_bound = add(p.finish_work_units, mul(p.source_slots, source.work_units_upper_bound));
    // Fixed scalar/codec/receipt/control reservation; not exact OS RSS or
    // allocator bookkeeping. No O(o) source metadata, hashes or pointer table.
    p.fixed_control_storage_bytes = 8192 + sizeof(Accumulator) + sizeof(Result)
        + 2 * sizeof(Source) + sizeof(Inventory) + sizeof(Caps) + 2 * sizeof(Plan)
        + 2 * sizeof(SourcePlan) + 4 * sizeof(Digest) + 4 * sizeof(Sum);
    p.total_control_storage_bytes = add(p.fixed_control_storage_bytes, inventory.other_live_control_bytes_per_replica);
    p.other_live_numerical_bytes_per_replica = inventory.other_live_numerical_bytes_per_replica;
    p.backend_margin_bytes_per_replica = inventory.backend_margin_bytes_per_replica;
    p.numerical_replicas = inventory.numerical_replicas; p.external_node_bytes = inventory.external_node_bytes;
    p.per_replica_inventoried_bytes = add(add(p.peak_owned_numerical_bytes, p.maximum_borrowed_numerical_bytes),
        add(add(p.other_live_numerical_bytes_per_replica, p.total_control_storage_bytes), p.backend_margin_bytes_per_replica));
    p.required_node_inventoried_bytes = add(p.external_node_bytes, mul(p.numerical_replicas, p.per_replica_inventoried_bytes));
    for (U bytes : {p.output_bytes, p.intermediate_bytes, p.peak_owned_numerical_bytes}) address_extent(bytes);
    if (aa > std::vector<double>().max_size() || am > std::vector<double>().max_size())
        throw std::overflow_error("particle-hole exceeds native vector extent");
    (void) mul(8, add(4096, p.output_bytes));
    return p;
}

Result::BoundedRestrictedPairCCSDParticleHoleResult(Result&& other) noexcept
    : live_(std::exchange(other.live_, false)), i_(other.i_), j_(other.j_), memory_(other.memory_),
      diagnostics_(other.diagnostics_), residual_(std::move(other.residual_)), input_(std::move(other.input_)),
      payload_(std::move(other.payload_)), identity_(std::move(other.identity_)) {}
void Result::require_live() const {
    if (!live_ || residual_.size() != memory_.output_bytes / 8)
        throw std::invalid_argument("particle-hole result has been consumed");
}
const Plan& Result::memory() const { require_live(); return memory_; }
const Diagnostics& Result::diagnostics() const { require_live(); return diagnostics_; }
U Result::target_i() const { require_live(); return i_; }
U Result::target_j() const { require_live(); return j_; }
const double* Result::residual_data() const { require_live(); return residual_.data(); }
double Result::residual(std::size_t a, std::size_t b) const {
    require_live();
    if (a >= memory_.target_dimension || b >= memory_.target_dimension)
        throw std::out_of_range("particle-hole residual index out of range");
    return residual_[a * memory_.target_dimension + b];
}
const std::string& Result::input_stream_identity_sha256() const { require_live(); return input_; }
const std::string& Result::payload_identity_sha256() const { require_live(); return payload_; }
const std::string& Result::identity_sha256() const { require_live(); return identity_; }

Accumulator::BoundedRestrictedPairCCSDParticleHoleAccumulator(U a, U o, U i, U j, U maximum_rank,
    const Inventory& supplied_inventory, const Caps& supplied_caps) : caps_(supplied_caps) {
    const Inventory inventory = supplied_inventory;
    limit(a, caps_.maximum_target_dimension, "particle-hole target dimension cap exceeded");
    limit(o, caps_.maximum_occupied_count, "particle-hole occupied count cap exceeded");
    limit(maximum_rank, caps_.maximum_source_dimension, "particle-hole source dimension cap exceeded");
    if (i >= o || j >= o) throw std::invalid_argument("particle-hole target occupied index is invalid");
    result_.memory_ = plan_bounded_restricted_pair_ccsd_particle_hole(a, o, maximum_rank, inventory);
    admit(result_.memory_, caps_); environment();
    result_.i_ = i; result_.j_ = j;
    Digest stream("vibeqc.bounded.pair-ccsd-particle-hole.stream");
    stream.u64(a); stream.u64(o); stream.u64(i); stream.u64(j);
    result_.input_ = stream.finish();
    result_.residual_.resize(static_cast<std::size_t>(result_.memory_.output_bytes / 8), 0.0);
    compensation_.resize(result_.residual_.size(), 0.0);
    intermediate_.resize(static_cast<std::size_t>(result_.memory_.intermediate_bytes / 8), 0.0);
    active_ = true;
}
Accumulator::BoundedRestrictedPairCCSDParticleHoleAccumulator(Accumulator&& other) noexcept
    : active_(std::exchange(other.active_, false)), failed_(other.failed_), finished_(other.finished_), caps_(other.caps_),
      result_(std::move(other.result_)), compensation_(std::move(other.compensation_)), intermediate_(std::move(other.intermediate_)) {}
void Accumulator::require_active() const {
    if (!active_ || failed_ || finished_) throw std::invalid_argument("particle-hole stream is failed, finished or consumed");
}
const Plan& Accumulator::memory() const { require_active(); return result_.memory_; }

void Accumulator::accumulate(const Source& supplied_source) {
    try {
        require_active();
        const Source s = supplied_source;
        const Plan& p = result_.memory_; auto& d = result_.diagnostics_;
        if (d.visited_sources >= p.source_slots || s.leg != d.visited_sources / p.occupied_count
            || s.occupied_index != d.visited_sources % p.occupied_count)
            throw std::invalid_argument("particle-hole source requires the exact next leg/occupied slot");
        limit(s.source_dimension, p.maximum_source_dimension, "particle-hole source rank exceeds admitted maximum");
        const SourcePlan source = plan_bounded_restricted_pair_ccsd_particle_hole_source(p.target_dimension, s.source_dimension);
        // All per-source extents, scans and numerical work are admitted BEFORE
        // touching its floats. The uniform upper already reserves all 2o slots.
        limit(source.borrowed_numerical_bytes, p.maximum_borrowed_numerical_bytes, "particle-hole source bytes exceed plan");
        const U stop = add(d.scalar_products, source.scalar_products);
        limit(stop, p.scalar_products_upper_bound, "particle-hole stream product work exceeds plan");
        const U charged = add(d.charged_work_units, source.work_units_upper_bound);
        limit(add(charged, p.finish_work_units), p.work_units_upper_bound, "particle-hole stream work exceeds plan");
        const U a = p.target_dimension, b = s.source_dimension, ab = mul(a, b);
        view(s.k_ba, ab); view(s.j_ba, ab); view(s.overlap_ba, ab); view(s.t_bb, mul(b, b));
        environment(); const auto before = source_hash(s, a);
        const auto t = [&](U c, U e) { return s.t_bb.data[s.source_transposed ? e * b + c : c * b + e]; };
        const auto accumulate = [&](U left, U right, double x) {
            const U index = s.leg == 0 ? left * a + right : right * a + left;
            include(result_.residual_[index], compensation_[index], x);
        };
        // The complete original restricted W1/W2 bare contribution from
        // ordered T_lm, retaining the INTERNAL e index in its own frame B.
        for (U c = 0; c < b; ++c) for (U right = 0; right < a; ++right) {
            Sum value;
            for (U e = 0; e < b; ++e) {
                Sum spin; spin.include(product(2.0, t(c, e), d, stop)); spin.include(-t(e, c));
                value.include(product(spin.total(), s.k_ba.data[e * a + right], d, stop));
                value.include(-product(t(c, e), s.j_ba.data[e * a + right], d, stop));
            }
            const double x = value.total(); intermediate_[c * a + right] = x;
            d.maximum_absolute_intermediate = std::max(d.maximum_absolute_intermediate, std::abs(x));
        }
        for (U left = 0; left < a; ++left) for (U right = 0; right < a; ++right) {
            Sum value;
            for (U c = 0; c < b; ++c)
                value.include(product(s.overlap_ba.data[c * a + left], intermediate_[c * a + right], d, stop));
            accumulate(left, right, value.total());
        }
        // Reindex the OTHER pass's WX term using T_ml=T_lm^T, then
        // transpose its target indices. This exchange is not the Eq.(26)
        // semi-joint term alone and must not be omitted or counted twice.
        for (U c = 0; c < b; ++c) for (U right = 0; right < a; ++right) {
            Sum value;
            for (U e = 0; e < b; ++e)
                value.include(product(t(e, c), s.j_ba.data[e * a + right], d, stop));
            const double x = value.total(); intermediate_[c * a + right] = x;
            d.maximum_absolute_intermediate = std::max(d.maximum_absolute_intermediate, std::abs(x));
        }
        for (U left = 0; left < a; ++left) for (U right = 0; right < a; ++right) {
            Sum value;
            for (U c = 0; c < b; ++c)
                value.include(product(s.overlap_ba.data[c * a + left], intermediate_[c * a + right], d, stop));
            accumulate(right, left, -value.total());
        }
        if (d.scalar_products != stop) throw std::logic_error("particle-hole completed multiplication census mismatch");
        environment();
        if (source_hash(s, a) != before)
            throw std::invalid_argument("particle-hole source payload changed during consumption");
        Digest chain("vibeqc.bounded.pair-ccsd-particle-hole.chain");
        chain.string(result_.input_); chain.string(before); result_.input_ = chain.finish();
        d.input_scan_elements = add(d.input_scan_elements, source.input_scan_elements);
        d.charged_work_units = charged; ++d.visited_sources;
        if (!b) ++d.zero_rank_sources;
    } catch (...) { failed_ = true; active_ = false; throw; }
}

Result Accumulator::finish() {
    try {
        require_active(); auto& d = result_.diagnostics_; const auto& p = result_.memory_;
        if (d.visited_sources != p.source_slots)
            throw std::invalid_argument("particle-hole finish requires every leg/occupied source slot");
        environment();
        d.charged_work_units = add(d.charged_work_units, p.finish_work_units);
        limit(d.charged_work_units, p.work_units_upper_bound, "particle-hole finish exceeds work plan");
        Digest payload("vibeqc.bounded.pair-ccsd-particle-hole.payload");
        payload.u64(p.target_dimension); payload.u64(p.occupied_count); payload.u64(result_.i_); payload.u64(result_.j_);
        for (std::size_t x = 0; x < result_.residual_.size(); ++x) {
            const double value = finite(result_.residual_[x] + compensation_[x]); result_.residual_[x] = value;
            d.maximum_absolute_residual = std::max(d.maximum_absolute_residual, std::abs(value)); payload.real(value);
        }
        result_.payload_ = payload.finish();
        Digest identity("vibeqc.bounded.pair-ccsd-particle-hole.identity");
        identity.string(result_.input_); identity.string(result_.payload_);
        identity.string("restricted-bare-W1-W2-WX;both-legs;source-first;binary64-nearest-gradual;compensated;no-weights;no-source-certificate");
        result_.identity_ = identity.finish(); result_.live_ = true;
        std::vector<double>().swap(compensation_); std::vector<double>().swap(intermediate_);
        finished_ = true; active_ = false; return std::move(result_);
    } catch (...) { failed_ = true; active_ = false; throw; }
}
} // namespace vibeqc
