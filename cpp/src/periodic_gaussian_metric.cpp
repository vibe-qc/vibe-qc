#include "vibeqc/periodic_gaussian_metric.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "vibeqc/detail/periodic_gaussian_metric_numeric.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"

namespace vibeqc {
namespace {
using Complex = std::complex<double>;
constexpr char kRawDomain[] = "vibeqc.periodic.gaussian-metric.payload";
constexpr char kWhiteDomain[] = "vibeqc.periodic.gaussian-whitener.payload";
constexpr std::uint64_t kShaMax = std::numeric_limits<std::uint64_t>::max() / 8U;
constexpr std::uint64_t kRawPrefix = 268U + sizeof(kRawDomain) - 1U;
constexpr std::uint64_t kWhitePrefix = 228U + sizeof(kWhiteDomain) - 1U;

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) throw std::overflow_error("Gaussian metric count overflow");
    return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::overflow_error("Gaussian metric count overflow");
    return a * b;
}
void limit(std::uint64_t a, std::uint64_t cap, const char* message) {
    if (a > cap) throw std::length_error(message);
}
std::array<char, 64> ascii(const std::string& s) {
    if (s.size() != 64U || !std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    })) throw std::invalid_argument("Gaussian metric requires a native lowercase SHA identity");
    std::array<char, 64> r{};
    std::copy(s.begin(), s.end(), r.begin());
    return r;
}
class Digest {
public:
    void bytes(const std::uint8_t* p, std::size_t n) {
        if (n > kShaMax - extent_) throw std::length_error("Gaussian metric SHA extent overflow");
        hash_.update(p, n); extent_ += n;
    }
    void u32(std::uint32_t n) {
        std::array<std::uint8_t, 4> b{};
        for (unsigned i = 0; i != 4; ++i) b[i] = n >> (24U - 8U * i);
        bytes(b.data(), b.size());
    }
    void u64(std::uint64_t n) {
        std::array<std::uint8_t, 8> b{};
        for (unsigned i = 0; i != 8; ++i) b[i] = n >> (56U - 8U * i);
        bytes(b.data(), b.size());
    }
    void real(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("Gaussian metric payload is nonfinite");
        if (x == 0.0) x = 0.0;
        std::uint64_t n = 0; std::memcpy(&n, &x, 8U); u64(n);
    }
    void text(const std::string& s) {
        u64(s.size()); bytes(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
    }
    std::uint64_t extent() const noexcept { return extent_; }
    std::array<char, 64> finish() { return ascii(hash_.finish_hex()); }
private:
    detail::Sha256 hash_;
    std::uint64_t extent_ = 0;
};

void basis_caps(Digest& h, const PeriodicGaussianSourceCaps& b) {
    h.u64(b.maximum_context_storage_bytes); h.u64(b.maximum_kpoint_count);
    h.u64(b.maximum_shell_count); h.u64(b.maximum_contraction_count);
    h.u64(b.maximum_primitive_numeric_lanes); h.u64(b.maximum_basis_content_wire_bytes);
    h.u64(b.maximum_borrowed_active_numeric_bytes); h.u64(b.maximum_work_units);
}
PeriodicGaussianSourceCaps require_basis_caps(
    const PeriodicGaussianSourceContext& c, const PeriodicGaussianSourceCaps& b) {
    const auto& a = c.inventory().ao;
    const auto& x = c.inventory().auxiliary;
    const std::array<std::uint64_t, 8> limits{{b.maximum_context_storage_bytes, b.maximum_kpoint_count,
        b.maximum_shell_count, b.maximum_contraction_count, b.maximum_primitive_numeric_lanes,
        b.maximum_basis_content_wire_bytes, b.maximum_borrowed_active_numeric_bytes, b.maximum_work_units}};
    const std::array<std::uint64_t, 8> counts{{sizeof(PeriodicGaussianSourceContext), c.mesh().size(),
        add(a.shell_count, x.shell_count), add(a.contraction_count, x.contraction_count),
        add(add(a.exponent_count, a.coefficient_count), add(x.exponent_count, x.coefficient_count)),
        add(a.content_wire_bytes, x.content_wire_bytes), c.inventory().combined_borrowed_active_numeric_bytes,
        c.inventory().work_units_upper_bound}};
    for (std::size_t i = 0; i != counts.size(); ++i) {
        if (limits[i] == 0U) throw std::invalid_argument("Gaussian metric basis verification caps must be positive");
        limit(counts[i], limits[i], "Gaussian metric basis verification exceeds cap");
    }
    // Caller caps authorize the immutable expected census, but may not let
    // changed actual inputs expand this wrapper's scan/work reservation.
    auto bounded = b;
    bounded.maximum_context_storage_bytes = counts[0];
    bounded.maximum_kpoint_count = counts[1];
    bounded.maximum_shell_count = counts[2];
    bounded.maximum_contraction_count = counts[3];
    bounded.maximum_primitive_numeric_lanes = counts[4];
    bounded.maximum_basis_content_wire_bytes = counts[5];
    bounded.maximum_borrowed_active_numeric_bytes = counts[6];
    bounded.maximum_work_units = counts[7];
    return bounded;
}

void payload_extent(std::uint64_t elements, std::uint64_t prefix) {
    if (elements > (kShaMax - prefix) / 16U) throw std::length_error("Gaussian metric payload exceeds SHA extent");
}
void hash_plan(Digest& h, const PeriodicGaussianMetricPlan& p) {
    h.u32(static_cast<std::uint32_t>(p.phase));
    for (auto n : {p.n_auxiliary, p.accepted_vector_count, p.candidate_count,
        p.reciprocal_panel_capacity, p.auxiliary_validation_passes,
        p.matrix_bytes, p.reciprocal_panel_bytes, p.double_fourier_panel_bytes,
        p.metric_owned_numeric_peak_bytes, p.factorization_owned_numeric_peak_bytes,
        p.owned_numeric_peak_bytes, p.returned_numeric_bytes, p.borrowed_basis_active_numeric_bytes,
        p.fixed_inventoried_object_bytes, p.per_replica_inventoried_bytes, p.node_inventoried_bytes,
        p.metric_product_count, p.fourier_radial_term_count, p.harmonic_term_count_upper_bound,
        p.jacobi_rotation_count_upper_bound, p.whitener_term_count_upper_bound,
        p.candidate_evaluations, p.work_units_upper_bound}) h.u64(n);
    h.u64(p.config.reciprocal_block); h.u64(p.config.whitener_column_block);
    basis_caps(h, p.config.basis_verification_caps);
    for (auto n : {p.live.replicas_per_node, p.live.other_retained_bytes_per_replica,
        p.live.other_transient_bytes_per_replica, p.live.fixed_backend_margin_bytes_per_replica,
        p.live.external_node_bytes, p.caps.maximum_owned_numeric_bytes,
        p.caps.maximum_per_replica_inventoried_bytes, p.caps.maximum_node_inventoried_bytes,
        p.caps.maximum_candidate_evaluations, p.caps.maximum_work_units}) h.u64(n);
}

PeriodicGaussianMetricPlan make_plan(
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context,
    const std::string& source, const std::string& opposite, std::uint64_t q, std::uint64_t qbar,
    std::uint64_t accepted, std::uint64_t candidates,
    const PeriodicGaussianMetricConfig& config, const PeriodicGaussianMetricLiveInventory& live,
    const PeriodicGaussianMetricCaps& caps, PeriodicGaussianMetricPhase phase) {
    if (!context) throw std::invalid_argument("Gaussian metric requires a live source context");
    if (caps.maximum_owned_numeric_bytes == 0 || caps.maximum_per_replica_inventoried_bytes == 0
        || caps.maximum_node_inventoried_bytes == 0 || caps.maximum_candidate_evaluations == 0
        || caps.maximum_work_units == 0 || live.replicas_per_node == 0
        || live.fixed_backend_margin_bytes_per_replica == 0) {
        throw std::invalid_argument("Gaussian metric caps, replicas and declared backend margin must be positive");
    }
    if (phase != PeriodicGaussianMetricPhase::RawMetric && phase != PeriodicGaussianMetricPhase::PrincipalWhitening) {
        throw std::invalid_argument("Gaussian metric phase is unsupported");
    }
    const auto a = context->inventory().auxiliary.function_count;
    if (a == 0 || accepted == 0 || candidates == 0 || config.reciprocal_block == 0
        || config.whitener_column_block == 0 || config.whitener_column_block > a
        || q >= context->mesh().size() || context->mesh().negate_index(q) != qbar) {
        throw std::invalid_argument("Gaussian metric shape, blocks or q addresses are invalid");
    }
    require_basis_caps(*context, config.basis_verification_caps);
    if (context->factorization_backend_identity_sha256()
        != periodic_correlation_metric_factorization_backend_identity_sha256()) {
        throw std::invalid_argument("Gaussian metric context does not match compiled principal-root backend");
    }
    ascii(source); ascii(opposite);
    PeriodicGaussianMetricPlan p;
    p.phase = phase; p.config = config; p.live = live; p.caps = caps;
    p.n_auxiliary = a; p.accepted_vector_count = accepted; p.candidate_count = candidates;
    const auto square = mul(a, a), triangle = mul(a, add(a, 1U)) / 2U;
    const auto cube = mul(square, a), g = std::min(config.reciprocal_block, accepted);
    payload_extent(square, kRawPrefix); payload_extent(square, kWhitePrefix);
    limit(square, std::vector<Complex>().max_size(), "Gaussian metric square exceeds vector extent");
    limit(mul(a, g), std::vector<Complex>().max_size(), "Gaussian metric Fourier extent exceeds vector limit");
    limit(mul(5U, g), std::vector<double>().max_size(), "Gaussian metric reciprocal extent exceeds vector limit");
    p.reciprocal_panel_capacity = g;
    p.auxiliary_validation_passes = add(accepted / g + (accepted % g != 0), 1U);
    p.matrix_bytes = mul(16U, square); p.reciprocal_panel_bytes = mul(40U, g);
    p.double_fourier_panel_bytes = mul(32U, mul(a, g));
    p.metric_owned_numeric_peak_bytes = add(add(p.matrix_bytes, p.reciprocal_panel_bytes), p.double_fourier_panel_bytes);
    p.factorization_owned_numeric_peak_bytes = add(mul(32U, square), mul(8U, a));
    p.owned_numeric_peak_bytes = phase == PeriodicGaussianMetricPhase::RawMetric
        ? p.metric_owned_numeric_peak_bytes : p.factorization_owned_numeric_peak_bytes;
    p.returned_numeric_bytes = p.matrix_bytes;
    p.borrowed_basis_active_numeric_bytes = context->inventory().combined_borrowed_active_numeric_bytes;
    p.fixed_inventoried_object_bytes = sizeof(PeriodicGaussianSourceContext)
        + sizeof(PeriodicGaussianReciprocalSource) + sizeof(PeriodicGaussianReciprocalMetric)
        + sizeof(PeriodicGaussianMetricWhitener) + sizeof(PeriodicGaussianMetricPlan);
    p.per_replica_inventoried_bytes = add(add(p.owned_numeric_peak_bytes, p.borrowed_basis_active_numeric_bytes),
        add(p.fixed_inventoried_object_bytes, add(live.other_retained_bytes_per_replica,
            add(live.other_transient_bytes_per_replica, live.fixed_backend_margin_bytes_per_replica))));
    p.node_inventoried_bytes = add(live.external_node_bytes, mul(live.replicas_per_node, p.per_replica_inventoried_bytes));
    p.metric_product_count = mul(accepted, triangle);
    p.fourier_radial_term_count = mul(accepted, context->inventory().auxiliary.coefficient_count);
    p.harmonic_term_count_upper_bound = mul(28U, mul(accepted, a));
    p.jacobi_rotation_count_upper_bound = mul(64U, mul(a, a - 1U) / 2U);
    p.whitener_term_count_upper_bound = mul(a, triangle);
    p.candidate_evaluations = phase == PeriodicGaussianMetricPhase::RawMetric ? mul(2U, candidates) : 0U;
    if (phase == PeriodicGaussianMetricPhase::RawMetric) {
        const auto& b = context->inventory().auxiliary;
        const auto scan = add(b.borrowed_active_numeric_bytes / 8U, add(b.shell_count, b.contraction_count));
        p.work_units_upper_bound = context->inventory().work_units_upper_bound;
        p.work_units_upper_bound = add(p.work_units_upper_bound, mul(128U, mul(p.auxiliary_validation_passes, scan)));
        for (auto term : {mul(128U, p.fourier_radial_term_count), mul(512U, p.harmonic_term_count_upper_bound),
            mul(64U, p.metric_product_count), mul(64U, mul(accepted, a)), mul(2048U, candidates),
            mul(128U, square), std::uint64_t(4096U)}) p.work_units_upper_bound = add(p.work_units_upper_bound, term);
    } else {
        p.work_units_upper_bound = add(add(mul(32768U, cube), mul(1024U, square)), 4096U);
    }
    limit(p.owned_numeric_peak_bytes, caps.maximum_owned_numeric_bytes, "Gaussian metric owned numerical memory exceeds cap");
    limit(p.per_replica_inventoried_bytes, caps.maximum_per_replica_inventoried_bytes, "Gaussian metric per-replica inventory exceeds cap");
    limit(p.node_inventoried_bytes, caps.maximum_node_inventoried_bytes, "Gaussian metric node inventory exceeds cap");
    limit(p.candidate_evaluations, caps.maximum_candidate_evaluations, "Gaussian metric candidate evaluations exceed cap");
    limit(p.work_units_upper_bound, caps.maximum_work_units, "Gaussian metric work exceeds cap");
    Digest hash;
    hash.text("vibeqc.periodic.gaussian-metric.plan"); hash.u32(1U);
    hash.text(context->source_context_identity_sha256()); hash.text(source); hash.text(opposite);
    hash.u64(q); hash.u64(qbar); hash_plan(hash, p); p.identity_ascii = hash.finish();
    return p;
}

std::array<char, 64> raw_payload(
    const PeriodicGaussianSourceContext& context, const std::string& source, const std::string& opposite,
    std::uint64_t q, std::uint64_t qbar, std::uint64_t a, const std::vector<Complex>& m) {
    if (m.size() != mul(a, a)) throw std::invalid_argument("Gaussian metric raw matrix extent is inconsistent or consumed");
    payload_extent(m.size(), kRawPrefix);
    Digest hash; hash.text(kRawDomain); hash.u32(1U);
    hash.text(context.source_context_identity_sha256()); hash.text(source); hash.text(opposite);
    hash.u64(q); hash.u64(qbar); hash.u64(a); hash.u64(m.size()); hash.u64(mul(16U, m.size()));
    if (hash.extent() != kRawPrefix) throw std::logic_error("Gaussian metric raw prefix wire changed");
    for (const auto z : m) { hash.real(z.real()); hash.real(z.imag()); }
    return hash.finish();
}
void diagnostics(Digest& hash, const PeriodicCorrelationMetricFactorizationDiagnostics& d) {
    hash.u64(d.n_auxiliary); hash.u64(d.retained_rank); hash.u64(d.sweeps); hash.u64(d.rotations);
    for (double x : {d.rank_cutoff, d.negative_tolerance, d.minimum_eigenvalue, d.maximum_eigenvalue,
        d.smallest_retained_eigenvalue, d.largest_discarded_eigenvalue, d.input_scale,
        d.scaled_initial_frobenius_norm, d.scaled_final_offdiagonal_norm, d.orthogonality_frobenius_error,
        d.maximum_self_conjugate_imaginary_residual}) hash.real(x);
}

} // namespace

PeriodicGaussianMetricPlan plan_periodic_gaussian_metric(
    const PeriodicGaussianReciprocalSource& source, const PeriodicGaussianMetricConfig& config,
    const PeriodicGaussianMetricLiveInventory& live, const PeriodicGaussianMetricCaps& caps,
    PeriodicGaussianMetricPhase phase) {
    if (!source.reciprocal_conjugate_closure_certified()) throw std::invalid_argument("Gaussian metric needs a certified live reciprocal source");
    return make_plan(source.context_handle(), source.source_identity_sha256(), source.conjugate_source_identity_sha256(),
        source.q_index(), source.conjugate_q_index(), source.accepted_vector_count(), source.candidate_count(),
        config, live, caps, phase);
}

PeriodicGaussianReciprocalMetric build_periodic_gaussian_reciprocal_metric(
    const PeriodicGaussianReciprocalSource& source, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicGaussianMetricConfig& config, const PeriodicGaussianMetricLiveInventory& live,
    const PeriodicGaussianMetricCaps& caps) {
    const auto plan = plan_periodic_gaussian_metric(source, config, live, caps, PeriodicGaussianMetricPhase::RawMetric);
    const auto verified = source.context_handle()->verify_bases(ao, auxiliary,
        require_basis_caps(*source.context_handle(), config.basis_verification_caps));
    if (verified.auxiliary.function_count != plan.n_auxiliary) throw std::logic_error("Gaussian metric auxiliary extent changed");
    // Full payload replay, rounding-environment check and angular preflight
    // precede every variable numerical allocation.
    visit_periodic_gaussian_reciprocal_source(source, source.candidate_count(),
        [](const std::array<std::int64_t, 3>&, const std::array<double, 5>&, void*) {}, nullptr);
    const auto empty = auxiliary_gaussian_fourier_panel(auxiliary, AuxiliaryFourierVectorView{}, 1U);
    if (!empty.data.empty()) throw std::logic_error("Gaussian metric basis preflight allocated output");
    auto numeric = detail::contract_reciprocal_metric_numeric(auxiliary, plan.n_auxiliary,
        source.accepted_vector_count(), plan.reciprocal_panel_capacity, source.self_conjugate_transfer(),
        plan.owned_numeric_peak_bytes,
        [](detail::PeriodicReciprocalRecordCallback callback, void* user, const void* pointer) {
            const auto& s = *static_cast<const PeriodicGaussianReciprocalSource*>(pointer);
            return visit_periodic_gaussian_reciprocal_source(s, s.candidate_count(), callback, user);
        }, &source);
    PeriodicGaussianReciprocalMetric result;
    result.context_ = source.context_handle(); result.plan_ = plan;
    result.q_ = source.q_index(); result.qbar_ = source.conjugate_q_index();
    result.source_ = ascii(source.source_identity_sha256()); result.opposite_ = ascii(source.conjugate_source_identity_sha256());
    result.completed_panels_ = numeric.completed_panel_count;
    result.imaginary_residual_ = numeric.maximum_self_conjugate_imaginary_residual;
    result.matrix_ = std::move(numeric.matrix);
    result.payload_ = raw_payload(*result.context_, result.source_identity_sha256(), result.conjugate_source_identity_sha256(),
        result.q_, result.qbar_, plan.n_auxiliary, result.matrix_);
    return result;
}

PeriodicGaussianMetricWhitener factorize_periodic_gaussian_metric(
    PeriodicGaussianReciprocalMetric&& raw, std::uint64_t columns,
    const PeriodicGaussianMetricLiveInventory& live, const PeriodicGaussianMetricCaps& caps) {
    if (!raw.context_) throw std::invalid_argument("Gaussian metric factorization input is consumed");
    auto config = raw.plan_.config; config.whitener_column_block = columns;
    const auto plan = make_plan(raw.context_, raw.source_identity_sha256(), raw.conjugate_source_identity_sha256(),
        raw.q_, raw.qbar_, raw.plan_.accepted_vector_count, raw.plan_.candidate_count,
        config, live, caps, PeriodicGaussianMetricPhase::PrincipalWhitening);
    if (raw_payload(*raw.context_, raw.source_identity_sha256(), raw.conjugate_source_identity_sha256(),
        raw.q_, raw.qbar_, plan.n_auxiliary, raw.matrix_) != raw.payload_) {
        throw std::invalid_argument("Gaussian metric raw payload identity mismatch");
    }
    const auto a = plan.n_auxiliary;
    for (std::size_t i = 0; i < a; ++i) for (std::size_t j = i; j < a; ++j) {
        const auto z = raw.matrix_[i * a + j];
        if (z != std::conj(raw.matrix_[j * a + i]) || (raw.q_ == raw.qbar_ && z.imag() != 0.0)) {
            throw std::invalid_argument("Gaussian metric raw Hermitian/real gauge mismatch");
        }
    }
    PeriodicGaussianMetricWhitener result;
    result.context_ = raw.context_; result.plan_ = plan; result.q_ = raw.q_; result.qbar_ = raw.qbar_;
    result.source_ = raw.source_; result.opposite_ = raw.opposite_; result.input_ = raw.payload_;
    auto matrix = std::move(raw.matrix_);
    raw.context_.reset(); // invalidate sealed owner before any solver mutation
    const auto& options = result.context_->options();
    auto factor = detail::metric_principal_square_root_admitted(std::move(matrix), a,
        options.metric_absolute_eigenvalue_threshold, options.metric_negative_tolerance,
        result.q_ == result.qbar_, columns);
    result.diagnostics_ = factor.diagnostics; result.matrix_ = std::move(factor.whitener);
    Digest payload; payload.text(kWhiteDomain); payload.u32(1U);
    payload.text(result.input_payload_identity_sha256()); payload.u64(result.q_); payload.u64(result.qbar_);
    diagnostics(payload, result.diagnostics_); payload.u64(result.matrix_.size());
    if (payload.extent() != kWhitePrefix) throw std::logic_error("Gaussian whitener prefix wire changed");
    for (const auto z : result.matrix_) { payload.real(z.real()); payload.real(z.imag()); }
    result.payload_ = payload.finish();
    return result;
}

#define VIBEQC_GAUSSIAN_ID(Class, Name, Field) \
std::string Class::Name() const { return {Field.begin(), Field.end()}; }
VIBEQC_GAUSSIAN_ID(PeriodicGaussianReciprocalMetric, source_identity_sha256, source_)
VIBEQC_GAUSSIAN_ID(PeriodicGaussianReciprocalMetric, conjugate_source_identity_sha256, opposite_)
VIBEQC_GAUSSIAN_ID(PeriodicGaussianReciprocalMetric, payload_identity_sha256, payload_)
VIBEQC_GAUSSIAN_ID(PeriodicGaussianMetricWhitener, source_identity_sha256, source_)
VIBEQC_GAUSSIAN_ID(PeriodicGaussianMetricWhitener, conjugate_source_identity_sha256, opposite_)
VIBEQC_GAUSSIAN_ID(PeriodicGaussianMetricWhitener, input_payload_identity_sha256, input_)
VIBEQC_GAUSSIAN_ID(PeriodicGaussianMetricWhitener, payload_identity_sha256, payload_)
#undef VIBEQC_GAUSSIAN_ID

} // namespace vibeqc
