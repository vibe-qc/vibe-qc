#include "vibeqc/periodic_gaussian_reciprocal_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace {

constexpr char kDomain[] = "vibeqc.periodic.gaussian-reciprocal-source";
constexpr std::uint64_t kPrefixBytes = 300U + sizeof(kDomain) - 1U;
constexpr std::uint64_t kShaMaximumBytes = std::numeric_limits<std::uint64_t>::max() / 8U;
struct FactoryWorkspace {
    detail::PeriodicReciprocalNumericPlan own;
    detail::PeriodicReciprocalNumericPlan opposite;
    PeriodicGaussianTransferRecord q;
    PeriodicGaussianTransferRecord qbar;
};

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("Gaussian reciprocal source count overflow");
    }
    return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("Gaussian reciprocal source count overflow");
    }
    return a * b;
}
void require_limit(std::uint64_t value, std::uint64_t cap, const char* message) {
    if (value > cap) throw std::length_error(message);
}
void require_context(const std::shared_ptr<const PeriodicGaussianSourceContext>& context) {
    if (!context || context->contract_version() != kPeriodicGaussianSourceContextVersion) {
        throw std::invalid_argument("Gaussian reciprocal source requires a live native source context");
    }
}

class Hasher {
public:
    void bytes(const std::uint8_t* p, std::size_t n) {
        if (n > kShaMaximumBytes - extent_) {
            throw std::length_error("Gaussian reciprocal source exceeds SHA message domain");
        }
        hash_.update(p, n);
        extent_ += n;
    }
    void u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> b{};
        for (unsigned i = 0; i != 4U; ++i) b[i] = value >> (24U - 8U * i);
        bytes(b.data(), b.size());
    }
    void u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> b{};
        for (unsigned i = 0; i != 8U; ++i) b[i] = value >> (56U - 8U * i);
        bytes(b.data(), b.size());
    }
    void real(double value) {
        if (!std::isfinite(value)) throw std::invalid_argument("Gaussian reciprocal source has nonfinite wire lane");
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void text(const std::string& text) {
        u64(text.size());
        bytes(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    }
    void record(const std::array<std::int64_t, 3>& label, const std::array<double, 5>& lanes) {
        for (auto n : label) u64(static_cast<std::uint64_t>(n));
        for (double value : lanes) real(value);
    }
    std::uint64_t extent() const noexcept { return extent_; }
    std::array<char, 64> finish() {
        const auto text = hash_.finish_hex();
        if (text.size() != 64U) throw std::logic_error("Gaussian reciprocal SHA length");
        std::array<char, 64> result{};
        std::copy(text.begin(), text.end(), result.begin());
        return result;
    }
private:
    detail::Sha256 hash_;
    std::uint64_t extent_ = 0;
};

void prefix(Hasher& hash, const PeriodicGaussianSourceContext& context,
            const detail::PeriodicReciprocalNumericPlan& numeric,
            const PeriodicGaussianTransferRecord& q, std::uint64_t accepted) {
    hash.text(kDomain);
    hash.u32(kPeriodicGaussianReciprocalSourceVersion);
    hash.u32(kPeriodicCorrelationReciprocalMetricSourceContractVersion);
    hash.u32(context.contract_version());
    hash.text(context.source_context_identity_sha256());
    hash.u64(q.index);
    for (auto n : q.centered_doubled_numerator) hash.u32(static_cast<std::uint32_t>(n));
    for (auto n : q.centered_reciprocal_wrap) hash.u32(static_cast<std::uint32_t>(n));
    for (int r = 0; r != 3; ++r) {
        for (int c = 0; c != 3; ++c) hash.real(numeric.geometry.reciprocal_lattice(r, c));
    }
    hash.real(context.options().reciprocal_energy_cutoff);
    hash.real(numeric.maximum_radius);
    hash.real(numeric.boundary_tolerance);
    hash.real(numeric.geometry.cell_volume);
    for (auto n : numeric.geometry.lower_bounds) hash.u64(static_cast<std::uint64_t>(n));
    for (auto n : numeric.geometry.upper_bounds) hash.u64(static_cast<std::uint64_t>(n));
    hash.u64(numeric.enumeration.candidate_count);
    hash.u64(accepted);
    hash.u64(q.gamma ? 1U : 0U);
    if (hash.extent() != kPrefixBytes) throw std::logic_error("Gaussian reciprocal source prefix wire changed");
}

std::uint64_t count_source(const detail::PeriodicReciprocalNumericPlan& numeric) {
    return detail::visit_periodic_reciprocal_numeric_source(
        numeric.geometry,
        [](const std::array<std::int64_t, 3>&, const std::array<double, 5>&, void*) {}, nullptr);
}

std::array<char, 64> hash_source(
    const PeriodicGaussianSourceContext& context,
    const detail::PeriodicReciprocalNumericPlan& numeric,
    const PeriodicGaussianTransferRecord& q, std::uint64_t expected_count) {
    Hasher hash;
    prefix(hash, context, numeric, q, expected_count);
    const auto count = detail::visit_periodic_reciprocal_numeric_source(
        numeric.geometry,
        [](const std::array<std::int64_t, 3>& label, const std::array<double, 5>& lanes, void* user) {
            static_cast<Hasher*>(user)->record(label, lanes);
        }, &hash);
    if (count != expected_count || hash.extent() != periodic_gaussian_reciprocal_source_wire_bytes(count)) {
        throw std::logic_error("Gaussian reciprocal source changed between count and hash");
    }
    return hash.finish();
}

detail::PeriodicReciprocalNumericPlan prepare(
    const PeriodicGaussianSourceContext& context,
    const PeriodicGaussianTransferRecord& q, std::uint64_t cap) {
    Eigen::Vector3d fractional;
    for (int d = 0; d != 3; ++d) fractional[d] = q.fractional[d];
    return detail::prepare_periodic_reciprocal_numeric_source(
        context.reciprocal_lattice(), fractional,
        context.options().reciprocal_energy_cutoff, q.gamma, cap);
}

} // namespace

std::uint64_t periodic_gaussian_reciprocal_source_wire_bytes(std::uint64_t n) {
    if (n > (kShaMaximumBytes - kPrefixBytes) / 64U) {
        throw std::length_error("Gaussian reciprocal source exceeds SHA message domain");
    }
    return kPrefixBytes + 64U * n;
}

PeriodicGaussianReciprocalSource make_periodic_gaussian_reciprocal_source(
    std::shared_ptr<const PeriodicGaussianSourceContext> context,
    std::uint64_t q_index, const PeriodicGaussianReciprocalSourceCaps& caps) {
    if (caps.maximum_fixed_storage_bytes == 0 || caps.maximum_candidates_per_source == 0
        || caps.maximum_candidate_evaluations == 0 || caps.maximum_source_wire_bytes == 0) {
        throw std::invalid_argument("Gaussian reciprocal source caps must be positive");
    }
    PeriodicGaussianReciprocalSourceInventory inventory;
    inventory.fixed_source_storage_bytes = sizeof(PeriodicGaussianReciprocalSource);
    inventory.retained_context_storage_bytes = sizeof(PeriodicGaussianSourceContext);
    inventory.factory_workspace_storage_bytes = sizeof(FactoryWorkspace);
    inventory.inventoried_fixed_storage_bytes = add(add(inventory.fixed_source_storage_bytes,
        inventory.retained_context_storage_bytes), inventory.factory_workspace_storage_bytes);
    require_limit(inventory.inventoried_fixed_storage_bytes, caps.maximum_fixed_storage_bytes,
                  "Gaussian reciprocal source fixed storage exceeds cap");
    require_context(context);
    if (q_index >= context->mesh().size()) throw std::out_of_range("Gaussian reciprocal source q index");
    FactoryWorkspace work;
    work.q = context->transfer_record(q_index);
    const auto qbar_index = context->mesh().negate_index(static_cast<std::size_t>(q_index));
    work.qbar = context->transfer_record(qbar_index);
    const bool self = qbar_index == q_index;
    work.own = prepare(*context, work.q, caps.maximum_candidates_per_source);
    work.opposite = self ? work.own : prepare(*context, work.qbar, caps.maximum_candidates_per_source);
    const auto own_candidates = work.own.enumeration.candidate_count;
    const auto other_candidates = work.opposite.enumeration.candidate_count;
    inventory.candidate_evaluations_upper_bound = add(mul(4U, own_candidates),
        self ? 0U : mul(2U, other_candidates));
    require_limit(inventory.candidate_evaluations_upper_bound, caps.maximum_candidate_evaluations,
                  "Gaussian reciprocal source candidate evaluations exceed cap");
    // No traversal above. Both exact boxes and the complete pass count have
    // been admitted before counting either physical reciprocal source.
    const auto own_count = count_source(work.own);
    const auto other_count = self ? own_count : count_source(work.opposite);
    if (own_count == 0U || other_count == 0U) {
        throw std::invalid_argument("Gaussian reciprocal source has no usable vectors at this q/cutoff");
    }
    if (own_count != other_count || work.q.gamma != work.qbar.gamma) {
        throw std::runtime_error("Gaussian reciprocal conjugacy has unequal accepted-vector counts");
    }
    inventory.source_wire_bytes = periodic_gaussian_reciprocal_source_wire_bytes(own_count);
    inventory.conjugate_source_wire_bytes = periodic_gaussian_reciprocal_source_wire_bytes(other_count);
    require_limit(inventory.source_wire_bytes, caps.maximum_source_wire_bytes,
                  "Gaussian reciprocal source wire exceeds cap");
    require_limit(inventory.conjugate_source_wire_bytes, caps.maximum_source_wire_bytes,
                  "Gaussian reciprocal conjugate source wire exceeds cap");
    const auto own_digest = hash_source(*context, work.own, work.q, own_count);
    const auto other_digest = self ? own_digest : hash_source(*context, work.opposite, work.qbar, other_count);
    std::array<std::int64_t, 3> shift{};
    for (int d = 0; d != 3; ++d) {
        const auto numerator = static_cast<std::int64_t>(work.q.centered_doubled_numerator[d])
            + work.qbar.centered_doubled_numerator[d];
        const auto denominator = 2 * static_cast<std::int64_t>(context->mesh().mesh()[d]);
        if (numerator % denominator != 0
            || (numerator / denominator != 0 && numerator / denominator != -1)) {
            throw std::logic_error("Gaussian reciprocal conjugacy centered addresses are inconsistent");
        }
        shift[d] = -numerator / denominator;
    }
    if (work.own.geometry.cell_volume != work.opposite.geometry.cell_volume) {
        throw std::logic_error("Gaussian reciprocal opposite source volume mismatch");
    }
    const auto audited = detail::require_periodic_reciprocal_numeric_conjugacy(
        work.own.geometry, work.opposite.geometry, shift, own_count);
    inventory.candidate_evaluations_performed = add(add(mul(3U, own_candidates),
        self ? 0U : mul(2U, other_candidates)), audited);
    if (inventory.candidate_evaluations_performed > inventory.candidate_evaluations_upper_bound) {
        throw std::logic_error("Gaussian reciprocal source exceeded admitted traversal count");
    }
    PeriodicGaussianReciprocalSource result;
    result.context_ = std::move(context);
    result.numeric_ = work.own;
    result.q_ = work.q;
    result.conjugate_q_index_ = qbar_index;
    result.conjugate_candidate_count_ = other_candidates;
    result.accepted_count_ = own_count;
    result.audited_count_ = audited;
    result.source_digest_ = own_digest;
    result.conjugate_digest_ = other_digest;
    result.inventory_ = inventory;
    return result;
}

double PeriodicGaussianReciprocalSource::reciprocal_energy_cutoff() const {
    require_context(context_);
    return context_->options().reciprocal_energy_cutoff;
}
std::string PeriodicGaussianReciprocalSource::source_identity_sha256() const {
    return {source_digest_.begin(), source_digest_.end()};
}
std::string PeriodicGaussianReciprocalSource::conjugate_source_identity_sha256() const {
    return {conjugate_digest_.begin(), conjugate_digest_.end()};
}
std::string PeriodicGaussianReciprocalSource::source_context_identity_sha256() const {
    require_context(context_);
    return context_->source_context_identity_sha256();
}

std::uint64_t visit_periodic_gaussian_reciprocal_source(
    const PeriodicGaussianReciprocalSource& source, std::uint64_t maximum_candidates,
    detail::PeriodicReciprocalRecordCallback callback, void* user) {
    require_context(source.context_);
    if (callback == nullptr || maximum_candidates == 0U) {
        throw std::invalid_argument("Gaussian reciprocal replay requires callback and positive candidate cap");
    }
    require_limit(source.candidate_count(), maximum_candidates,
                  "Gaussian reciprocal replay candidates exceed cap");
    Hasher hash;
    prefix(hash, *source.context_, source.numeric_, source.q_, source.accepted_count_);
    struct CallbackState {
        Hasher* hash;
        detail::PeriodicReciprocalRecordCallback callback;
        void* user;
    } state{&hash, callback, user};
    const auto count = detail::visit_periodic_reciprocal_numeric_source(
        source.numeric_.geometry,
        [](const std::array<std::int64_t, 3>& label, const std::array<double, 5>& lanes, void* opaque) {
            auto& callback_state = *static_cast<CallbackState*>(opaque);
            callback_state.hash->record(label, lanes);
            callback_state.callback(label, lanes, callback_state.user);
        }, &state);
    if (count != source.accepted_count_
        || hash.extent() != source.inventory_.source_wire_bytes
        || hash.finish() != source.source_digest_) {
        throw std::runtime_error("Gaussian reciprocal replay changed the sealed source");
    }
    return count;
}

} // namespace vibeqc
