#include "vibeqc/periodic_correlation_factor_stream.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {

namespace {

constexpr std::array<int, 3> kGammaCentered = {0, 0, 0};
constexpr std::uint64_t kComplexBytes = sizeof(std::complex<double>);
constexpr std::uint64_t kCanonicalDescriptorBytes = 84U;
constexpr std::uint64_t kSha256MaximumMessageBytes =
    std::numeric_limits<std::uint64_t>::max() / 8U;
constexpr char kScheduleDomain[] =
    "vibeqc.periodic.correlation.factor-stream.schedule";
constexpr char kSyntheticSourceDomain[] =
    "vibeqc.periodic.correlation.factor-stream.synthetic-source";
constexpr char kSyntheticPayloadDomain[] =
    "vibeqc.periodic.correlation.factor-stream.synthetic-payload";
constexpr char kSyntheticReceiptDomain[] =
    "vibeqc.periodic.correlation.factor-stream.synthetic-receipt";
constexpr char kSyntheticPatternName[] =
    "synthetic-nonphysical-test-pattern";
constexpr char kSyntheticPatternEncoding[] =
    "splitmix64-source-bound-to-exact-binary-fractions";

static_assert(sizeof(std::complex<double>) == 16U,
              "factor stream v1 requires two IEEE-754 binary64 components");
static_assert(std::numeric_limits<double>::is_iec559,
              "factor stream v1 requires IEEE-754 binary64");
static_assert(sizeof(double) == sizeof(std::uint64_t),
              "factor stream v1 requires IEEE-754 binary64");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "factor stream v1 requires size_t to fit in uint64_t");

class CanonicalFactorHasher {
public:
    CanonicalFactorHasher(const char* domain, std::uint32_t version) {
        add_string(domain);
        add_u32(version);
    }

    void add_u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(
                value >> (24U - 8U * static_cast<unsigned>(index)));
        }
        add_bytes(encoded.data(), encoded.size());
    }

    void add_i32(std::int32_t value) {
        add_u32(static_cast<std::uint32_t>(value));
    }

    void add_u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(
                value >> (56U - 8U * static_cast<unsigned>(index)));
        }
        add_bytes(encoded.data(), encoded.size());
    }

    void add_double(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "factor stream digest cannot encode a non-finite value");
        }
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        add_u64(bits);
    }

    void add_complex(const std::complex<double>& value) {
        add_double(value.real());
        add_double(value.imag());
    }

    void add_string(const std::string& value) {
        add_u64(static_cast<std::uint64_t>(value.size()));
        add_bytes(
            reinterpret_cast<const std::uint8_t*>(value.data()),
            value.size());
    }

    std::string finish_hex() { return hasher_.finish_hex(); }

private:
    void add_bytes(const std::uint8_t* data, std::size_t size) {
        const auto count = static_cast<std::uint64_t>(size);
        if (count > kSha256MaximumMessageBytes - wire_bytes_) {
            throw std::length_error(
                "periodic factor-stream canonical digest exceeds "
                "SHA-256's less-than-2^64-bit message-length domain");
        }
        hasher_.update(data, size);
        wire_bytes_ += count;
    }

    detail::Sha256 hasher_;
    std::uint64_t wire_bytes_ = 0;
};

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            std::string("periodic factor-stream arithmetic overflow: ")
            + context);
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* context) {
    if (left != 0U
        && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            std::string("periodic factor-stream arithmetic overflow: ")
            + context);
    }
    return left * right;
}

std::uint64_t checked_ceil_divide(std::uint64_t numerator,
                                  std::uint64_t denominator,
                                  const char* context) {
    if (denominator == 0U) {
        throw std::invalid_argument(
            std::string("periodic factor-stream zero divisor: ") + context);
    }
    return numerator / denominator
        + static_cast<std::uint64_t>(numerator % denominator != 0U);
}

std::size_t checked_size_t(std::uint64_t value, const char* context) {
    if (value
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            std::string("periodic factor-stream size_t overflow: ")
            + context);
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t canonical_string_wire_bytes(const char* value,
                                          const char* context) {
    return checked_add(
        8U,
        static_cast<std::uint64_t>(std::strlen(value)),
        context);
}

std::uint64_t hex_nibble(char value) {
    if (value >= '0' && value <= '9') {
        return static_cast<std::uint64_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<std::uint64_t>(value - 'a' + 10);
    }
    throw std::logic_error(
        "periodic synthetic factor source identity is not lowercase hex");
}

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t synthetic_source_seed(const std::string& source_identity) {
    if (source_identity.size() != 64U) {
        throw std::logic_error(
            "periodic synthetic factor source identity must be a SHA-256");
    }
    std::array<std::uint64_t, 4> words{};
    for (std::size_t word = 0; word < words.size(); ++word) {
        for (std::size_t nibble = 0; nibble < 16U; ++nibble) {
            words[word] = (words[word] << 4U)
                | hex_nibble(source_identity[16U * word + nibble]);
        }
    }

    std::uint64_t seed = 0x6a09e667f3bcc909ULL;
    seed ^= splitmix64(words[0] ^ 0xbb67ae8584caa73bULL);
    seed ^= splitmix64(words[1] ^ 0x3c6ef372fe94f82bULL);
    seed ^= splitmix64(words[2] ^ 0xa54ff53a5f1d36f1ULL);
    seed ^= splitmix64(words[3] ^ 0x510e527fade682d1ULL);
    return seed;
}

bool same_descriptor(
    const PeriodicCorrelationFactorTileDescriptor& left,
    const PeriodicCorrelationFactorTileDescriptor& right) noexcept {
    return left.sequence_index == right.sequence_index
        && left.q_index == right.q_index
        && left.k_bra_index == right.k_bra_index
        && left.k_ket_index == right.k_ket_index
        && left.k_ket_reciprocal_wrap == right.k_ket_reciprocal_wrap
        && left.ao_pair_begin == right.ao_pair_begin
        && left.ao_pair_count == right.ao_pair_count
        && left.auxiliary_begin == right.auxiliary_begin
        && left.auxiliary_count == right.auxiliary_count
        && left.element_count == right.element_count;
}

void add_descriptor(
    CanonicalFactorHasher& digest,
    const PeriodicCorrelationFactorTileDescriptor& descriptor) {
    digest.add_u64(descriptor.sequence_index);
    digest.add_u64(descriptor.q_index);
    digest.add_u64(descriptor.k_bra_index);
    digest.add_u64(descriptor.k_ket_index);
    for (const int component : descriptor.k_ket_reciprocal_wrap) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    digest.add_u64(descriptor.ao_pair_begin);
    digest.add_u64(descriptor.ao_pair_count);
    digest.add_u64(descriptor.auxiliary_begin);
    digest.add_u64(descriptor.auxiliary_count);
    digest.add_u64(descriptor.element_count);
}

PeriodicCorrelationFactorTileDescriptor make_descriptor(
    const PeriodicCorrelationFactorStreamShape& shape,
    const RegularKMesh& addressing,
    std::uint64_t sequence_index) {
    if (sequence_index >= shape.tile_count) {
        throw std::out_of_range(
            "periodic factor-stream sequence index is out of range");
    }

    std::uint64_t quotient = sequence_index;
    const std::uint64_t auxiliary_tile =
        quotient % shape.auxiliary_tile_count;
    quotient /= shape.auxiliary_tile_count;
    const std::uint64_t ao_pair_tile =
        quotient % shape.ao_pair_tile_count;
    quotient /= shape.ao_pair_tile_count;
    const std::uint64_t k_bra_index = quotient % shape.n_kpoints;
    const std::uint64_t q_index = quotient / shape.n_kpoints;
    if (q_index >= shape.n_kpoints) {
        throw std::logic_error(
            "periodic factor-stream sequence decomposition escaped q range");
    }

    const std::size_t k_bra = checked_size_t(
        k_bra_index, "descriptor k_bra index");
    const std::size_t q = checked_size_t(q_index, "descriptor q index");
    const auto translated = addressing.add_transfer_with_wrap(
        addressing.address(k_bra), addressing.transfer_address(q));
    const std::size_t k_ket = addressing.index(translated.address);
    if (k_ket != addressing.add_transfer_index(k_bra, q)
        || addressing.transfer_index(k_bra, k_ket) != q) {
        throw std::logic_error(
            "periodic factor-stream exact k-transfer round trip failed");
    }

    const std::uint64_t ao_pair_begin = checked_multiply(
        ao_pair_tile, shape.ao_pair_block, "AO-pair tile begin");
    const std::uint64_t auxiliary_begin = checked_multiply(
        auxiliary_tile, shape.auxiliary_block, "auxiliary tile begin");
    if (ao_pair_begin >= shape.n_ao_pairs
        || auxiliary_begin >= shape.n_auxiliary) {
        throw std::logic_error(
            "periodic factor-stream tile begin escaped its extent");
    }
    const std::uint64_t ao_pair_count = std::min(
        shape.ao_pair_block, shape.n_ao_pairs - ao_pair_begin);
    const std::uint64_t auxiliary_count = std::min(
        shape.auxiliary_block, shape.n_auxiliary - auxiliary_begin);
    const std::uint64_t element_count = checked_multiply(
        auxiliary_count, ao_pair_count, "descriptor element count");

    PeriodicCorrelationFactorTileDescriptor descriptor;
    descriptor.sequence_index = sequence_index;
    descriptor.q_index = q_index;
    descriptor.k_bra_index = k_bra_index;
    descriptor.k_ket_index = static_cast<std::uint64_t>(k_ket);
    for (int component = 0; component < 3; ++component) {
        descriptor.k_ket_reciprocal_wrap[component] =
            translated.wrap[component];
    }
    descriptor.ao_pair_begin = ao_pair_begin;
    descriptor.ao_pair_count = ao_pair_count;
    descriptor.auxiliary_begin = auxiliary_begin;
    descriptor.auxiliary_count = auxiliary_count;
    descriptor.element_count = element_count;
    return descriptor;
}

void require_admitted_reference_consistency(
    const PeriodicCorrelationAdmittedReference& reference) {
    if (!reference.state_handle()) {
        throw std::invalid_argument(
            "periodic factor-stream schedule requires a live admitted "
            "mean-field state");
    }
    const auto& dimensions = reference.dimensions();
    const auto& plan = reference.plan();
    const auto& state = reference.state();
    if (reference.contract_version()
            != kPeriodicCorrelationAdmittedReferenceContractVersion
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission
            != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || plan.allocation_contract_version
            != dimensions.allocation_contract_version
        || !plan.census_identity.empty()
        || plan.pair_domain_census_complete
        || plan.triple_domain_census_complete) {
        throw std::logic_error(
            "periodic factor-stream schedule received an inconsistent "
            "static admission contract");
    }
    if (dimensions.symmetry_reduction_requested
        || !dimensions.symmetry_mapping_identity.empty()
        || dimensions.symmetry_representative_count
            != dimensions.n_kpoints
        || dimensions.symmetry_weight_sum != dimensions.n_kpoints) {
        throw std::invalid_argument(
            "periodic factor-stream contract v1 requires an unreduced "
            "full-mesh symmetry inventory");
    }
    if (dimensions.is_shift != kGammaCentered
        || state.is_shift() != kGammaCentered) {
        throw std::invalid_argument(
            "periodic factor-stream contract v1 requires the "
            "Gamma-centered k-mesh convention");
    }
    if (dimensions.factor_q_block != 1U
        || dimensions.factor_k_bra_block != 1U
        || dimensions.factor_k_ket_block != 1U) {
        throw std::invalid_argument(
            "periodic factor-stream contract v1 requires unit q and k "
            "blocks");
    }
    if (dimensions.n_spin_channels != 1U
        || reference.budget().mpi_ranks != 1U) {
        throw std::invalid_argument(
            "periodic factor-stream contract v1 is sequential, "
            "single-rank, and closed-shell");
    }
    if (plan.calculation_identity != dimensions.calculation_identity
        || plan.allocation_identity != dimensions.allocation_identity
        || state.calculation_identity() != dimensions.calculation_identity
        || state.state_identity_sha256().empty()
        || state.mesh() != dimensions.mesh
        || state.is_shift() != dimensions.is_shift
        || static_cast<std::uint64_t>(state.n_kpoints())
            != dimensions.n_kpoints
        || state.n_basis() != dimensions.n_basis) {
        throw std::logic_error(
            "periodic factor-stream schedule received inconsistent "
            "admitted-reference provenance or dimensions");
    }
}

std::string make_schedule_identity_sha256(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamShape& shape) {
    CanonicalFactorHasher digest(
        kScheduleDomain,
        kPeriodicCorrelationFactorStreamContractVersion);
    digest.add_u32(reference.contract_version());
    digest.add_u32(reference.dimensions().allocation_contract_version);
    digest.add_u32(reference.state().digest_version());
    digest.add_string(reference.state().state_identity_sha256());
    digest.add_string(reference.dimensions().calculation_identity);
    digest.add_string(reference.dimensions().allocation_identity);
    for (const int component : reference.dimensions().mesh) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    for (const int component : reference.dimensions().is_shift) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    digest.add_string("gamma-centered-full-mesh-unreduced");
    digest.add_string("q,k-bra,ao-pair-tile,auxiliary-tile");
    digest.add_string("ao-pair=mu*n-basis+nu;nu-fastest");
    digest.add_string(
        "synthetic-payload[opaque-row,ao-pair];ao-pair-fastest");
    digest.add_string("complex-ieee754-binary64");
    digest.add_string(
        "q-index=regular-mesh-modular-residue;physical-q-gauge-undefined");
    digest.add_string("q=k-ket-k-bra;k-ket=k-bra+q-mod-G");
    digest.add_u64(shape.n_kpoints);
    digest.add_u64(shape.n_basis);
    digest.add_u64(shape.n_auxiliary);
    digest.add_u64(shape.n_ao_pairs);
    digest.add_u64(shape.auxiliary_block);
    digest.add_u64(shape.ao_pair_block);
    digest.add_u64(shape.auxiliary_tile_count);
    digest.add_u64(shape.ao_pair_tile_count);
    digest.add_u64(shape.tiles_per_k_bra);
    digest.add_u64(shape.tiles_per_q);
    digest.add_u64(shape.tile_count);
    digest.add_u64(shape.logical_element_count);
    digest.add_u64(shape.logical_bytes);
    digest.add_u64(shape.maximum_tile_element_count);
    digest.add_u64(shape.maximum_tile_bytes);
    digest.add_u64(shape.two_buffer_workspace_bytes);
    return digest.finish_hex();
}

std::string make_source_identity_sha256(const std::string& schedule_identity) {
    CanonicalFactorHasher digest(
        kSyntheticSourceDomain,
        kPeriodicCorrelationSyntheticFactorPatternVersion);
    digest.add_string(schedule_identity);
    digest.add_string(kSyntheticPatternName);
    digest.add_string(kSyntheticPatternEncoding);
    return digest.finish_hex();
}

std::complex<double> synthetic_value_unchecked(std::uint64_t source_seed,
                                               std::uint64_t q_index,
                                               std::uint64_t k_bra_index,
                                               std::uint64_t auxiliary,
                                               std::uint64_t ao_pair) {
    std::uint64_t key = source_seed;
    key ^= splitmix64(q_index ^ 0x243f6a8885a308d3ULL);
    key ^= splitmix64(k_bra_index ^ 0x13198a2e03707344ULL);
    key ^= splitmix64(auxiliary ^ 0xa4093822299f31d0ULL);
    key ^= splitmix64(ao_pair ^ 0x082efa98ec4e6c89ULL);
    key = splitmix64(key);

    const std::uint64_t real_code = (key & 0xfffffULL) + 1U;
    const std::uint64_t imaginary_code = ((key >> 20U) & 0xfffffULL) + 1U;
    const double real = std::ldexp(static_cast<double>(real_code), -20);
    double imaginary =
        std::ldexp(static_cast<double>(imaginary_code), -20);
    if (((key >> 40U) & 1U) != 0U) imaginary = -imaginary;
    return {real, imaginary};
}

std::string make_receipt_identity_sha256(
    const PeriodicCorrelationSyntheticFactorReceipt& receipt) {
    CanonicalFactorHasher digest(
        kSyntheticReceiptDomain,
        kPeriodicCorrelationSyntheticFactorTransactionVersion);
    digest.add_u32(receipt.transaction_contract_version);
    digest.add_u32(receipt.synthetic_pattern_version);
    digest.add_string(kSyntheticPatternName);
    digest.add_u32(receipt.synthetic_nonphysical ? 1U : 0U);
    digest.add_string(receipt.schedule_identity_sha256);
    digest.add_string(receipt.source_identity_sha256);
    digest.add_string(receipt.payload_identity_sha256);
    digest.add_u64(receipt.committed_tile_count);
    digest.add_u64(receipt.committed_element_count);
    digest.add_u64(receipt.committed_logical_bytes);
    return digest.finish_hex();
}

}  // namespace

std::uint64_t
estimate_periodic_correlation_synthetic_payload_wire_bytes(
    const PeriodicCorrelationFactorStreamShape& shape) {
    const std::uint64_t encoded_complex_bytes = checked_multiply(
        shape.logical_element_count,
        kComplexBytes,
        "synthetic payload canonical complex bytes");
    if (encoded_complex_bytes != shape.logical_bytes) {
        throw std::invalid_argument(
            "periodic synthetic factor payload shape has inconsistent "
            "logical byte counts");
    }

    std::uint64_t wire_bytes = canonical_string_wire_bytes(
        kSyntheticPayloadDomain, "synthetic payload domain wire bytes");
    wire_bytes = checked_add(
        wire_bytes, 4U, "synthetic payload version wire bytes");
    wire_bytes = checked_add(
        wire_bytes, 8U + 64U, "synthetic payload schedule identity bytes");
    wire_bytes = checked_add(
        wire_bytes, 8U + 64U, "synthetic payload source identity bytes");
    wire_bytes = checked_add(
        wire_bytes,
        canonical_string_wire_bytes(
            kSyntheticPatternName, "synthetic payload pattern wire bytes"),
        "synthetic payload prefix wire bytes");
    wire_bytes = checked_add(
        wire_bytes, 3U * 8U, "synthetic payload shape-count wire bytes");
    wire_bytes = checked_add(
        wire_bytes,
        checked_multiply(
            shape.tile_count,
            kCanonicalDescriptorBytes,
            "synthetic payload descriptor wire bytes"),
        "synthetic payload descriptor-prefix wire bytes");
    wire_bytes = checked_add(
        wire_bytes,
        encoded_complex_bytes,
        "complete synthetic payload wire bytes");
    if (wire_bytes > kSha256MaximumMessageBytes) {
        throw std::length_error(
            "periodic synthetic factor payload exceeds SHA-256's "
            "less-than-2^64-bit message-length domain");
    }
    return wire_bytes;
}

PeriodicCorrelationFactorStreamShape
estimate_periodic_correlation_factor_stream_shape(
    std::array<int, 3> mesh,
    std::uint64_t n_basis,
    std::uint64_t n_auxiliary,
    std::uint64_t auxiliary_block,
    std::uint64_t ao_pair_block) {
    if (n_basis == 0U) {
        throw std::invalid_argument(
            "periodic factor-stream shape requires n_basis > 0");
    }
    if (n_auxiliary == 0U) {
        throw std::invalid_argument(
            "periodic factor-stream shape requires n_auxiliary > 0");
    }
    if (auxiliary_block == 0U || auxiliary_block > n_auxiliary) {
        throw std::invalid_argument(
            "periodic factor-stream auxiliary block must be in "
            "[1, n_auxiliary]");
    }

    RegularKMesh addressing({1, 1, 1});
    try {
        addressing = RegularKMesh(mesh);
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument(
            std::string("invalid periodic factor-stream RegularKMesh: ")
            + error.what());
    }
    const std::uint64_t n_kpoints =
        static_cast<std::uint64_t>(addressing.size());
    const std::uint64_t n_ao_pairs = checked_multiply(
        n_basis, n_basis, "AO-pair extent");
    const std::uint64_t resolved_ao_pair_block =
        ao_pair_block == 0U ? n_ao_pairs : ao_pair_block;
    if (resolved_ao_pair_block > n_ao_pairs) {
        throw std::invalid_argument(
            "periodic factor-stream AO-pair block cannot exceed "
            "n_basis squared");
    }

    PeriodicCorrelationFactorStreamShape shape;
    shape.n_kpoints = n_kpoints;
    shape.n_basis = n_basis;
    shape.n_auxiliary = n_auxiliary;
    shape.n_ao_pairs = n_ao_pairs;
    shape.auxiliary_block = auxiliary_block;
    shape.ao_pair_block = resolved_ao_pair_block;
    shape.auxiliary_tile_count = checked_ceil_divide(
        n_auxiliary, auxiliary_block, "auxiliary tile count");
    shape.ao_pair_tile_count = checked_ceil_divide(
        n_ao_pairs, resolved_ao_pair_block, "AO-pair tile count");
    shape.tiles_per_k_bra = checked_multiply(
        shape.ao_pair_tile_count,
        shape.auxiliary_tile_count,
        "tiles per k_bra");
    shape.tiles_per_q = checked_multiply(
        n_kpoints, shape.tiles_per_k_bra, "tiles per q");
    shape.tile_count = checked_multiply(
        n_kpoints, shape.tiles_per_q, "total tile count");

    const std::uint64_t k_pair_count = checked_multiply(
        n_kpoints, n_kpoints, "logical k-pair count");
    std::uint64_t logical_elements = checked_multiply(
        k_pair_count, n_auxiliary, "logical k-pair auxiliary extent");
    logical_elements = checked_multiply(
        logical_elements, n_ao_pairs, "logical factor element count");
    shape.logical_element_count = logical_elements;
    shape.logical_bytes = checked_multiply(
        logical_elements, kComplexBytes, "logical factor bytes");
    shape.maximum_tile_element_count = checked_multiply(
        auxiliary_block,
        resolved_ao_pair_block,
        "maximum tile element count");
    shape.maximum_tile_bytes = checked_multiply(
        shape.maximum_tile_element_count,
        kComplexBytes,
        "maximum tile bytes");
    shape.two_buffer_workspace_bytes = checked_multiply(
        2U, shape.maximum_tile_bytes, "two-buffer workspace bytes");
    return shape;
}

PeriodicCorrelationFactorStreamSchedule::
    PeriodicCorrelationFactorStreamSchedule(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::string calculation_identity,
        std::string allocation_identity,
        std::string state_identity_sha256,
        std::string schedule_identity_sha256,
        std::array<int, 3> mesh,
        std::array<int, 3> is_shift,
        PeriodicCorrelationFactorStreamShape shape)
    : state_(std::move(state)),
      calculation_identity_(std::move(calculation_identity)),
      allocation_identity_(std::move(allocation_identity)),
      state_identity_sha256_(std::move(state_identity_sha256)),
      schedule_identity_sha256_(std::move(schedule_identity_sha256)),
      mesh_(mesh),
      is_shift_(is_shift),
      shape_(shape) {}

const PeriodicRestrictedMeanFieldState&
PeriodicCorrelationFactorStreamSchedule::state() const {
    if (!state_) {
        throw std::logic_error(
            "moved-from periodic factor-stream schedule has no state");
    }
    return *state_;
}

PeriodicCorrelationFactorTileDescriptor
PeriodicCorrelationFactorStreamSchedule::descriptor(
    std::uint64_t sequence_index) const {
    if (!state_) {
        throw std::logic_error(
            "moved-from periodic factor-stream schedule has no descriptor");
    }
    return make_descriptor(
        shape_, RegularKMesh(mesh_, is_shift_), sequence_index);
}

std::uint64_t PeriodicCorrelationFactorStreamSchedule::ao_pair_index(
    std::uint64_t mu,
    std::uint64_t nu) const {
    if (!state_) {
        throw std::logic_error(
            "moved-from periodic factor-stream schedule has no AO-pair map");
    }
    if (mu >= shape_.n_basis || nu >= shape_.n_basis) {
        throw std::out_of_range(
            "periodic factor-stream AO index is out of range");
    }
    return checked_add(
        checked_multiply(mu, shape_.n_basis, "AO-pair row begin"),
        nu,
        "AO-pair index");
}

std::array<std::uint64_t, 2>
PeriodicCorrelationFactorStreamSchedule::ao_pair_indices(
    std::uint64_t ao_pair_index_value) const {
    if (!state_) {
        throw std::logic_error(
            "moved-from periodic factor-stream schedule has no AO-pair map");
    }
    if (ao_pair_index_value >= shape_.n_ao_pairs) {
        throw std::out_of_range(
            "periodic factor-stream AO-pair index is out of range");
    }
    return {
        ao_pair_index_value / shape_.n_basis,
        ao_pair_index_value % shape_.n_basis,
    };
}

std::uint64_t PeriodicCorrelationFactorStreamSchedule::payload_offset(
    const PeriodicCorrelationFactorTileDescriptor& tile,
    std::uint64_t auxiliary,
    std::uint64_t ao_pair) const {
    const auto expected = descriptor(tile.sequence_index);
    if (!same_descriptor(tile, expected)) {
        throw std::invalid_argument(
            "periodic factor-stream payload offset received a descriptor "
            "that is not canonical for this schedule");
    }
    if (auxiliary < tile.auxiliary_begin
        || auxiliary - tile.auxiliary_begin >= tile.auxiliary_count) {
        throw std::out_of_range(
            "periodic factor-stream auxiliary index is outside the tile");
    }
    if (ao_pair < tile.ao_pair_begin
        || ao_pair - tile.ao_pair_begin >= tile.ao_pair_count) {
        throw std::out_of_range(
            "periodic factor-stream AO-pair index is outside the tile");
    }
    return checked_add(
        checked_multiply(
            auxiliary - tile.auxiliary_begin,
            tile.ao_pair_count,
            "tile payload auxiliary row"),
        ao_pair - tile.ao_pair_begin,
        "tile payload offset");
}

PeriodicCorrelationFactorStreamSchedule
make_periodic_correlation_factor_stream_schedule(
    const PeriodicCorrelationAdmittedReference& reference) {
    require_admitted_reference_consistency(reference);
    const auto& dimensions = reference.dimensions();
    const auto shape = estimate_periodic_correlation_factor_stream_shape(
        dimensions.mesh,
        dimensions.n_basis,
        dimensions.n_auxiliary,
        dimensions.factor_auxiliary_block,
        dimensions.factor_ao_pair_block);
    if (shape.n_kpoints != dimensions.n_kpoints) {
        throw std::logic_error(
            "periodic factor-stream shape disagrees with the admitted "
            "k-point extent");
    }
    const std::string schedule_identity =
        make_schedule_identity_sha256(reference, shape);
    return PeriodicCorrelationFactorStreamSchedule(
        reference.state_handle(),
        dimensions.calculation_identity,
        dimensions.allocation_identity,
        reference.state().state_identity_sha256(),
        schedule_identity,
        dimensions.mesh,
        dimensions.is_shift,
        shape);
}

std::complex<double> periodic_correlation_synthetic_factor_value(
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    std::uint64_t q_index,
    std::uint64_t k_bra_index,
    std::uint64_t auxiliary,
    std::uint64_t ao_pair) {
    if (!schedule.state_handle()) {
        throw std::logic_error(
            "moved-from periodic factor-stream schedule has no synthetic "
            "source");
    }
    const auto& shape = schedule.shape();
    if (q_index >= shape.n_kpoints
        || k_bra_index >= shape.n_kpoints
        || auxiliary >= shape.n_auxiliary
        || ao_pair >= shape.n_ao_pairs) {
        throw std::out_of_range(
            "periodic synthetic factor source index is out of range");
    }
    const std::uint64_t source_seed = synthetic_source_seed(
        make_source_identity_sha256(schedule.schedule_identity_sha256()));
    return synthetic_value_unchecked(
        source_seed, q_index, k_bra_index, auxiliary, ao_pair);
}

void fill_periodic_correlation_synthetic_factor_tile(
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorTileDescriptor& tile,
    std::complex<double>* payload,
    std::uint64_t payload_count) {
    const auto expected = schedule.descriptor(tile.sequence_index);
    if (!same_descriptor(tile, expected)) {
        throw std::invalid_argument(
            "periodic synthetic factor source received a noncanonical "
            "descriptor");
    }
    if (payload_count != tile.element_count) {
        throw std::invalid_argument(
            "periodic synthetic factor source payload count does not match "
            "the descriptor");
    }
    if (payload == nullptr) {
        throw std::invalid_argument(
            "periodic synthetic factor source requires a non-null payload");
    }
    const std::uint64_t source_seed = synthetic_source_seed(
        make_source_identity_sha256(schedule.schedule_identity_sha256()));

    for (std::uint64_t auxiliary_local = 0;
         auxiliary_local < tile.auxiliary_count;
         ++auxiliary_local) {
        const std::uint64_t auxiliary =
            tile.auxiliary_begin + auxiliary_local;
        for (std::uint64_t ao_pair_local = 0;
             ao_pair_local < tile.ao_pair_count;
             ++ao_pair_local) {
            const std::uint64_t ao_pair =
                tile.ao_pair_begin + ao_pair_local;
            const std::uint64_t offset =
                auxiliary_local * tile.ao_pair_count + ao_pair_local;
            payload[checked_size_t(offset, "synthetic source offset")] =
                synthetic_value_unchecked(
                    source_seed,
                    tile.q_index,
                    tile.k_bra_index,
                    auxiliary,
                    ao_pair);
        }
    }
}

struct PeriodicCorrelationSyntheticFactorTransaction::Impl {
    explicit Impl(const PeriodicCorrelationFactorStreamSchedule& schedule)
        : state_handle(schedule.state_handle()),
          shape(schedule.shape()),
          mesh(schedule.mesh()),
          is_shift(schedule.is_shift()),
          schedule_identity(schedule.schedule_identity_sha256()),
          source_identity(make_source_identity_sha256(schedule_identity)),
          source_seed(synthetic_source_seed(source_identity)),
          payload_digest(
              kSyntheticPayloadDomain,
              kPeriodicCorrelationSyntheticFactorPatternVersion) {
        (void)estimate_periodic_correlation_synthetic_payload_wire_bytes(
            shape);
        payload_digest.add_string(schedule_identity);
        payload_digest.add_string(source_identity);
        payload_digest.add_string(kSyntheticPatternName);
        payload_digest.add_u64(shape.tile_count);
        payload_digest.add_u64(shape.logical_element_count);
        payload_digest.add_u64(shape.logical_bytes);
    }

    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_handle;
    PeriodicCorrelationFactorStreamShape shape;
    std::array<int, 3> mesh = {0, 0, 0};
    std::array<int, 3> is_shift = {0, 0, 0};
    std::string schedule_identity;
    std::string source_identity;
    std::uint64_t source_seed = 0;
    PeriodicCorrelationSyntheticTransactionState transaction_state =
        PeriodicCorrelationSyntheticTransactionState::Open;
    std::uint64_t next_sequence = 0;
    std::uint64_t accepted_elements = 0;
    CanonicalFactorHasher payload_digest;
    std::optional<PeriodicCorrelationSyntheticFactorReceipt> receipt;
};

PeriodicCorrelationSyntheticFactorTransaction::
    PeriodicCorrelationSyntheticFactorTransaction(
        std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

PeriodicCorrelationSyntheticFactorTransaction::
    PeriodicCorrelationSyntheticFactorTransaction(
        PeriodicCorrelationSyntheticFactorTransaction&&) noexcept = default;

PeriodicCorrelationSyntheticFactorTransaction&
PeriodicCorrelationSyntheticFactorTransaction::operator=(
    PeriodicCorrelationSyntheticFactorTransaction&& other) noexcept {
    if (this != &other) {
        abort();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

PeriodicCorrelationSyntheticFactorTransaction::
    ~PeriodicCorrelationSyntheticFactorTransaction() {
    abort();
}

PeriodicCorrelationSyntheticTransactionState
PeriodicCorrelationSyntheticFactorTransaction::state() const noexcept {
    return impl_ ? impl_->transaction_state
                 : PeriodicCorrelationSyntheticTransactionState::Aborted;
}

std::uint64_t
PeriodicCorrelationSyntheticFactorTransaction::next_sequence_index()
    const noexcept {
    return impl_ ? impl_->next_sequence : 0U;
}

std::uint64_t
PeriodicCorrelationSyntheticFactorTransaction::accepted_tile_count()
    const noexcept {
    return impl_ ? impl_->next_sequence : 0U;
}

std::uint64_t
PeriodicCorrelationSyntheticFactorTransaction::accepted_element_count()
    const noexcept {
    return impl_ ? impl_->accepted_elements : 0U;
}

const std::string&
PeriodicCorrelationSyntheticFactorTransaction::schedule_identity_sha256()
    const noexcept {
    static const std::string empty;
    return impl_ ? impl_->schedule_identity : empty;
}

const std::string&
PeriodicCorrelationSyntheticFactorTransaction::source_identity_sha256()
    const noexcept {
    static const std::string empty;
    return impl_ ? impl_->source_identity : empty;
}

void PeriodicCorrelationSyntheticFactorTransaction::accept(
    const PeriodicCorrelationFactorTileDescriptor& tile,
    const std::complex<double>* payload,
    std::uint64_t payload_count,
    std::uint64_t max_payload_bytes) {
    if (!impl_) {
        throw std::logic_error(
            "moved-from periodic synthetic factor transaction");
    }
    if (impl_->transaction_state
        != PeriodicCorrelationSyntheticTransactionState::Open) {
        throw std::logic_error(
            "periodic synthetic factor transaction is not open");
    }

    try {
        const RegularKMesh addressing(impl_->mesh, impl_->is_shift);
        const auto expected = make_descriptor(
            impl_->shape, addressing, impl_->next_sequence);
        if (!same_descriptor(tile, expected)) {
            throw std::invalid_argument(
                "periodic synthetic factor transaction received a tile "
                "out of order or with a mismatched descriptor");
        }
        if (payload_count != expected.element_count) {
            throw std::invalid_argument(
                "periodic synthetic factor transaction payload count does "
                "not match the descriptor");
        }
        if (max_payload_bytes == 0U) {
            throw std::invalid_argument(
                "periodic synthetic factor transaction requires an "
                "explicit nonzero payload-byte cap");
        }
        if (payload_count > max_payload_bytes / kComplexBytes) {
            throw std::length_error(
                "periodic synthetic factor transaction payload exceeds "
                "its explicit byte cap");
        }
        if (payload == nullptr) {
            throw std::invalid_argument(
                "periodic synthetic factor transaction requires a non-null "
                "payload");
        }

        for (std::uint64_t auxiliary_local = 0;
             auxiliary_local < expected.auxiliary_count;
             ++auxiliary_local) {
            const std::uint64_t auxiliary =
                expected.auxiliary_begin + auxiliary_local;
            for (std::uint64_t ao_pair_local = 0;
                 ao_pair_local < expected.ao_pair_count;
                 ++ao_pair_local) {
                const std::uint64_t ao_pair =
                    expected.ao_pair_begin + ao_pair_local;
                const std::uint64_t offset =
                    auxiliary_local * expected.ao_pair_count + ao_pair_local;
                const auto& value = payload[checked_size_t(
                    offset, "synthetic sink validation offset")];
                const auto expected_value = synthetic_value_unchecked(
                    impl_->source_seed,
                    expected.q_index,
                    expected.k_bra_index,
                    auxiliary,
                    ao_pair);
                if (!std::isfinite(value.real())
                    || !std::isfinite(value.imag())
                    || value != expected_value) {
                    throw std::invalid_argument(
                        "periodic synthetic factor transaction payload value "
                        "does not match the nonphysical test pattern");
                }
            }
        }

        // Hash only after the entire tile has validated, so a rejected tile
        // never leaves a partially accepted digest prefix.
        add_descriptor(impl_->payload_digest, expected);
        for (std::uint64_t offset = 0; offset < payload_count; ++offset) {
            impl_->payload_digest.add_complex(payload[checked_size_t(
                offset, "synthetic sink digest offset")]);
        }
        impl_->accepted_elements = checked_add(
            impl_->accepted_elements,
            payload_count,
            "accepted synthetic element count");
        impl_->next_sequence = checked_add(
            impl_->next_sequence, 1U, "accepted synthetic tile count");
    } catch (...) {
        impl_->transaction_state =
            PeriodicCorrelationSyntheticTransactionState::Failed;
        throw;
    }
}

PeriodicCorrelationSyntheticFactorReceipt
PeriodicCorrelationSyntheticFactorTransaction::commit() {
    if (!impl_) {
        throw std::logic_error(
            "moved-from periodic synthetic factor transaction");
    }
    if (impl_->transaction_state
        == PeriodicCorrelationSyntheticTransactionState::Committed) {
        return *impl_->receipt;
    }
    if (impl_->transaction_state
        != PeriodicCorrelationSyntheticTransactionState::Open) {
        throw std::logic_error(
            "periodic synthetic factor transaction cannot commit from its "
            "terminal state");
    }
    if (impl_->next_sequence != impl_->shape.tile_count
        || impl_->accepted_elements != impl_->shape.logical_element_count) {
        impl_->transaction_state =
            PeriodicCorrelationSyntheticTransactionState::Failed;
        throw std::logic_error(
            "periodic synthetic factor transaction cannot commit before the "
            "complete canonical stream has been accepted");
    }

    try {
        PeriodicCorrelationSyntheticFactorReceipt receipt;
        receipt.transaction_contract_version =
            kPeriodicCorrelationSyntheticFactorTransactionVersion;
        receipt.synthetic_pattern_version =
            kPeriodicCorrelationSyntheticFactorPatternVersion;
        receipt.payload_kind = PeriodicCorrelationFactorPayloadKind::
            SyntheticNonPhysicalTestPattern;
        receipt.synthetic_nonphysical = true;
        receipt.schedule_identity_sha256 = impl_->schedule_identity;
        receipt.source_identity_sha256 = impl_->source_identity;
        receipt.payload_identity_sha256 =
            impl_->payload_digest.finish_hex();
        receipt.committed_tile_count = impl_->next_sequence;
        receipt.committed_element_count = impl_->accepted_elements;
        receipt.committed_logical_bytes = checked_multiply(
            impl_->accepted_elements,
            kComplexBytes,
            "committed synthetic logical bytes");
        if (receipt.committed_logical_bytes != impl_->shape.logical_bytes) {
            throw std::logic_error(
                "periodic synthetic factor receipt byte count disagrees "
                "with the sealed schedule");
        }
        receipt.receipt_identity_sha256 =
            make_receipt_identity_sha256(receipt);
        impl_->receipt = receipt;
        impl_->transaction_state =
            PeriodicCorrelationSyntheticTransactionState::Committed;
        return receipt;
    } catch (...) {
        impl_->transaction_state =
            PeriodicCorrelationSyntheticTransactionState::Failed;
        throw;
    }
}

void PeriodicCorrelationSyntheticFactorTransaction::abort() noexcept {
    if (impl_
        && impl_->transaction_state
            == PeriodicCorrelationSyntheticTransactionState::Open) {
        impl_->transaction_state =
            PeriodicCorrelationSyntheticTransactionState::Aborted;
    }
}

PeriodicCorrelationSyntheticFactorTransaction
make_periodic_correlation_synthetic_factor_transaction(
    const PeriodicCorrelationFactorStreamSchedule& schedule) {
    if (!schedule.state_handle()) {
        throw std::invalid_argument(
            "periodic synthetic factor transaction requires a live "
            "schedule state");
    }
    return PeriodicCorrelationSyntheticFactorTransaction(
        std::make_unique<
            PeriodicCorrelationSyntheticFactorTransaction::Impl>(schedule));
}

PeriodicCorrelationSyntheticFactorReceipt
execute_periodic_correlation_synthetic_factor_stream(
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationSyntheticFactorExecutionCaps& caps) {
    if (!schedule.state_handle()) {
        throw std::invalid_argument(
            "periodic synthetic factor execution requires a live "
            "schedule state");
    }
    if (caps.max_tile_bytes == 0U
        || caps.max_workspace_bytes == 0U
        || caps.max_tile_count == 0U
        || caps.max_logical_bytes == 0U) {
        throw std::invalid_argument(
            "periodic synthetic factor execution requires explicit nonzero "
            "tile-byte, workspace-byte, tile-count, and logical-byte caps");
    }
    const auto& shape = schedule.shape();
    if (shape.maximum_tile_bytes > caps.max_tile_bytes) {
        throw std::length_error(
            "periodic synthetic factor maximum tile exceeds its cap");
    }
    if (shape.two_buffer_workspace_bytes > caps.max_workspace_bytes) {
        throw std::length_error(
            "periodic synthetic factor two-buffer workspace exceeds its "
            "cap");
    }
    if (shape.tile_count > caps.max_tile_count) {
        throw std::length_error(
            "periodic synthetic factor tile count exceeds its cap");
    }
    if (shape.logical_bytes > caps.max_logical_bytes) {
        throw std::length_error(
            "periodic synthetic factor logical byte count exceeds its cap");
    }

    const std::size_t buffer_elements = checked_size_t(
        shape.maximum_tile_element_count,
        "synthetic executor buffer element count");
    if (buffer_elements
        > std::numeric_limits<std::size_t>::max()
            / sizeof(std::complex<double>)) {
        throw std::overflow_error(
            "periodic synthetic factor buffer byte count exceeds size_t");
    }

    // Validate the state and complete hash domain before any payload buffer is
    // allocated. The transaction itself owns only fixed-size control state.
    auto transaction =
        make_periodic_correlation_synthetic_factor_transaction(schedule);

    // These are the only size-dependent payload allocations in the executor.
    auto source_buffer =
        std::make_unique<std::complex<double>[]>(buffer_elements);
    auto sink_staging_buffer =
        std::make_unique<std::complex<double>[]>(buffer_elements);

    for (std::uint64_t sequence = 0; sequence < shape.tile_count; ++sequence) {
        const auto tile = schedule.descriptor(sequence);
        fill_periodic_correlation_synthetic_factor_tile(
            schedule, tile, source_buffer.get(), tile.element_count);
        std::copy_n(
            source_buffer.get(),
            checked_size_t(tile.element_count, "synthetic tile copy count"),
            sink_staging_buffer.get());
        transaction.accept(
            tile,
            sink_staging_buffer.get(),
            tile.element_count,
            caps.max_tile_bytes);
    }
    return transaction.commit();
}

static_assert(
    !std::is_copy_constructible<
        PeriodicCorrelationFactorStreamSchedule>::value,
    "a periodic factor-stream schedule must not copy its retained state");
static_assert(
    std::is_nothrow_move_constructible<
        PeriodicCorrelationFactorStreamSchedule>::value,
    "a periodic factor-stream schedule must be cheaply movable");
static_assert(
    !std::is_copy_constructible<
        PeriodicCorrelationSyntheticFactorTransaction>::value,
    "a periodic synthetic factor transaction must not be copied");
static_assert(
    std::is_nothrow_move_constructible<
        PeriodicCorrelationSyntheticFactorTransaction>::value,
    "a periodic synthetic factor transaction must be cheaply movable");

}  // namespace vibeqc
