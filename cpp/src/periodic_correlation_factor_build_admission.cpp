#include "vibeqc/periodic_correlation_factor_build_admission.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {

namespace {

constexpr std::array<int, 3> kGammaCentered = {0, 0, 0};
constexpr PeriodicCorrelationByteCount kComplexBytes =
    kPeriodicCorrelationFactorBuildComplexBytes;
constexpr PeriodicCorrelationByteCount kRealBytes =
    kPeriodicCorrelationFactorBuildRealBytes;
constexpr PeriodicCorrelationByteCount kReciprocalVectorRecordBytes =
    kPeriodicCorrelationFactorBuildReciprocalVectorRecordBytes;
constexpr char kCensusDomain[] =
    "vibeqc.periodic.correlation.factor-build.census";
constexpr char kPlanDomain[] =
    "vibeqc.periodic.correlation.factor-build.plan";

static_assert(sizeof(std::complex<double>) == kComplexBytes,
              "factor-build admission requires complex binary64");
static_assert(sizeof(double) == kRealBytes,
              "factor-build admission requires binary64");
static_assert(std::numeric_limits<double>::is_iec559,
              "factor-build admission requires IEEE-754 binary64");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "factor-build admission requires size_t to fit uint64");

class FactorBuildArithmeticOverflow : public std::overflow_error {
public:
    explicit FactorBuildArithmeticOverflow(const std::string& label)
        : std::overflow_error(
              "periodic factor-build arithmetic overflow: " + label) {}
};

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw FactorBuildArithmeticOverflow(label);
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* label) {
    if (left != 0U
        && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw FactorBuildArithmeticOverflow(label);
    }
    return left * right;
}

std::uint64_t checked_ceil_divide(std::uint64_t numerator,
                                  std::uint64_t denominator,
                                  const char* label) {
    if (denominator == 0U) {
        throw std::invalid_argument(std::string(label) + " divisor is zero");
    }
    return numerator / denominator
        + static_cast<std::uint64_t>(numerator % denominator != 0U);
}

std::uint64_t checked_scale_ceil(std::uint64_t value,
                                 std::uint64_t numerator,
                                 std::uint64_t denominator,
                                 const char* label) {
    const auto quotient = value / denominator;
    const auto remainder = value % denominator;
    return checked_add(
        checked_multiply(quotient, numerator, label),
        checked_ceil_divide(
            checked_multiply(remainder, numerator, label),
            denominator,
            label),
        label);
}

std::uint64_t required_memory(std::uint64_t modeled) {
    return checked_scale_ceil(modeled, 3U, 2U, "memory headroom");
}

std::uint64_t required_scratch(std::uint64_t modeled) {
    return checked_scale_ceil(modeled, 5U, 4U, "scratch headroom");
}

bool is_lower_sha256(const std::string& value) {
    if (value.size() != 64U) return false;
    return std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f');
    });
}

void require_sha256(const std::string& value, const char* label) {
    if (!is_lower_sha256(value)) {
        throw std::invalid_argument(
            std::string(label) + " must be a lowercase SHA-256");
    }
}

void require_finite_positive(double value, const char* label) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(
            std::string(label) + " must be finite and positive");
    }
}

class CanonicalFactorBuildHasher {
public:
    CanonicalFactorBuildHasher(const char* domain, std::uint32_t version) {
        add_string(domain);
        add_u32(version);
    }

    void add_u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(
                value >> (24U - 8U * static_cast<unsigned>(index)));
        }
        hasher_.update(encoded.data(), encoded.size());
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
        hasher_.update(encoded.data(), encoded.size());
    }

    void add_double(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "factor-build digest cannot encode a non-finite value");
        }
        if (value == 0.0) value = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        add_u64(bits);
    }

    void add_string(const std::string& value) {
        add_u64(static_cast<std::uint64_t>(value.size()));
        hasher_.update(
            reinterpret_cast<const std::uint8_t*>(value.data()),
            value.size());
    }

    std::string finish_hex() { return hasher_.finish_hex(); }

private:
    detail::Sha256 hasher_;
};

template <typename Enum>
void add_enum(CanonicalFactorBuildHasher& digest, Enum value) {
    digest.add_u32(static_cast<std::uint32_t>(value));
}

void add_shape(CanonicalFactorBuildHasher& digest,
               const PeriodicCorrelationFactorBuildShape& shape) {
    digest.add_u64(shape.n_kpoints);
    digest.add_u64(shape.n_basis);
    digest.add_u64(shape.n_auxiliary);
    digest.add_u64(shape.n_ao_pairs);
    digest.add_u64(shape.ao_pair_block);
    digest.add_u64(shape.auxiliary_block);
    digest.add_u64(shape.reciprocal_block);
    digest.add_u64(shape.whitener_column_block);
    digest.add_u64(shape.tile_count);
    digest.add_u64(shape.logical_element_count);
    digest.add_u64(shape.logical_bytes);
    digest.add_u64(shape.maximum_tile_bytes);
}

void add_backend_inventory(
    CanonicalFactorBuildHasher& digest,
    const PeriodicCorrelationFactorBuildBackendInventory& backend) {
    digest.add_u32(backend.complete ? 1U : 0U);
    digest.add_u64(
        backend.per_thread_short_range_fixed_workspace_bytes);
    digest.add_u64(
        backend.per_thread_fourier_transform_fixed_workspace_bytes);
    digest.add_u64(backend.eigensolver_workspace_bytes);
    digest.add_u64(backend.whitener_workspace_bytes);
    digest.add_u64(backend.gemm_workspace_bytes);
    digest.add_u64(backend.publisher_workspace_bytes);
    digest.add_u64(backend.short_range_staging_memory_bytes);
    digest.add_u64(backend.folded_source_memory_bytes);
    digest.add_u64(backend.short_range_staging_scratch_bytes);
    digest.add_u64(backend.folded_source_scratch_bytes);
    digest.add_u64(backend.exact_extra_retained_bytes);
    digest.add_u64(backend.exact_extra_control_bytes);
    digest.add_u64(backend.exact_extra_scratch_bytes);
}

void add_codec_inventory(
    CanonicalFactorBuildHasher& digest,
    const PeriodicCorrelationFactorBuildCodecInventory& codec) {
    digest.add_u32(codec.complete ? 1U : 0U);
    digest.add_u64(codec.fixed_header_bytes);
    digest.add_u64(codec.fixed_footer_bytes);
    digest.add_u64(codec.fixed_manifest_bytes);
    digest.add_u64(codec.integrity_bytes_per_tile);
    digest.add_u64(codec.rank_bytes_per_q);
    digest.add_u64(codec.digest_bytes_per_tile);
    digest.add_u64(codec.journal_header_bytes);
    digest.add_u64(codec.journal_record_bytes_per_tile);
    digest.add_u64(codec.checkpoint_record_bytes);
    digest.add_u64(codec.checkpoint_records_per_generation);
    digest.add_u64(codec.existing_generation_count);
}

void add_config(CanonicalFactorBuildHasher& digest,
                const PeriodicCorrelationFactorBuildConfig& config) {
    add_enum(digest, config.producer_mode);
    add_enum(digest, config.short_range_policy);
    add_enum(digest, config.row_convention);
    add_enum(digest, config.transfer_convention);
    add_enum(digest, config.backing_mode);
    add_enum(digest, config.publisher_mode);
    digest.add_string(config.ao_basis_identity_sha256);
    digest.add_string(config.auxiliary_basis_identity_sha256);
    digest.add_string(config.producer_identity_sha256);
    digest.add_string(config.backend_identity_sha256);
    digest.add_string(config.codec_identity_sha256);
    digest.add_double(config.metric_absolute_eigenvalue_threshold);
    digest.add_double(config.integral_absolute_screening_threshold);
    digest.add_double(config.reciprocal_energy_cutoff);
    digest.add_double(config.short_range_real_space_cutoff);
    digest.add_double(config.range_separation_omega);
    digest.add_u64(config.ao_pair_block);
    digest.add_u64(config.auxiliary_block);
    digest.add_u64(config.reciprocal_block);
    digest.add_u64(config.whitener_column_block);
    digest.add_u64(config.q_concurrency);
    digest.add_u64(config.native_threads);
    digest.add_u32(config.ao_pair_fourier_staging_required ? 1U : 0U);
    digest.add_u32(config.transpose_before_publish ? 1U : 0U);
    digest.add_u64(config.publisher_buffer_count);
    add_backend_inventory(digest, config.backend);
    add_codec_inventory(digest, config.codec);
}

void add_q_record(CanonicalFactorBuildHasher& digest,
                  const PeriodicCorrelationFactorBuildQRecord& record) {
    digest.add_u64(record.q_index);
    for (const int component : record.centered_doubled_numerator) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    for (const int component : record.centered_reciprocal_wrap) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    digest.add_u64(record.base_reciprocal_vector_count);
    digest.add_u64(record.tail_reciprocal_vector_count);
    digest.add_u64(record.zero_mode_excluded_count);
    digest.add_u64(record.short_range_metric_cell_count);
    digest.add_u64(record.short_range_metric_task_count);
    digest.add_u64(record.short_range_three_center_cell_pair_count);
    digest.add_u64(record.short_range_three_center_task_count);
    digest.add_u64(record.source_manifest_bytes);
}

bool same_stream_shape(
    const PeriodicCorrelationFactorBuildShape& build,
    const PeriodicCorrelationFactorStreamShape& stream) noexcept {
    return build.n_kpoints == stream.n_kpoints
        && build.n_basis == stream.n_basis
        && build.n_auxiliary == stream.n_auxiliary
        && build.n_ao_pairs == stream.n_ao_pairs
        && build.ao_pair_block == stream.ao_pair_block
        && build.auxiliary_block == stream.auxiliary_block
        && build.tile_count == stream.tile_count
        && build.logical_element_count == stream.logical_element_count
        && build.logical_bytes == stream.logical_bytes
        && build.maximum_tile_bytes == stream.maximum_tile_bytes;
}

PeriodicCorrelationFactorBuildShape make_build_shape(
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildConfig& config) {
    const auto& stream = schedule.shape();
    PeriodicCorrelationFactorBuildShape shape;
    shape.n_kpoints = stream.n_kpoints;
    shape.n_basis = stream.n_basis;
    shape.n_auxiliary = stream.n_auxiliary;
    shape.n_ao_pairs = stream.n_ao_pairs;
    shape.ao_pair_block = config.ao_pair_block;
    shape.auxiliary_block = config.auxiliary_block;
    shape.reciprocal_block = config.reciprocal_block;
    shape.whitener_column_block = config.whitener_column_block;
    shape.tile_count = stream.tile_count;
    shape.logical_element_count = stream.logical_element_count;
    shape.logical_bytes = stream.logical_bytes;
    shape.maximum_tile_bytes = stream.maximum_tile_bytes;
    return shape;
}

void require_reference_schedule_consistency(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule) {
    if (!reference.state_handle() || !schedule.state_handle()) {
        throw std::invalid_argument(
            "factor-build admission requires live reference and schedule "
            "state handles");
    }
    if (reference.state_handle().get() != schedule.state_handle().get()) {
        throw std::invalid_argument(
            "factor-build reference and schedule do not share the same "
            "mean-field state object");
    }
    const auto& dimensions = reference.dimensions();
    const auto& plan = reference.plan();
    const auto& state = reference.state();
    if (reference.contract_version()
            != kPeriodicCorrelationAdmittedReferenceContractVersion
        || schedule.contract_version()
            != kPeriodicCorrelationFactorStreamContractVersion
        || plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission
            != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus
        || plan.allocation_contract_version
            != dimensions.allocation_contract_version) {
        throw std::logic_error(
            "factor-build admission received an inconsistent upstream "
            "contract");
    }
    if (dimensions.symmetry_reduction_requested
        || !dimensions.symmetry_mapping_identity.empty()
        || dimensions.symmetry_representative_count
            != dimensions.n_kpoints
        || dimensions.symmetry_weight_sum != dimensions.n_kpoints) {
        throw std::invalid_argument(
            "factor-build contract v1 requires an unreduced full mesh");
    }
    if (dimensions.is_shift != kGammaCentered
        || schedule.is_shift() != kGammaCentered
        || state.is_shift() != kGammaCentered) {
        throw std::invalid_argument(
            "factor-build contract v1 requires a Gamma-centered mesh");
    }
    if (reference.budget().mpi_ranks != 1U) {
        throw std::invalid_argument(
            "factor-build contract v1 requires exactly one MPI rank");
    }
    if (plan.calculation_identity != dimensions.calculation_identity
        || plan.allocation_identity != dimensions.allocation_identity
        || schedule.calculation_identity()
            != dimensions.calculation_identity
        || schedule.allocation_identity()
            != dimensions.allocation_identity
        || schedule.state_identity_sha256()
            != state.state_identity_sha256()
        || schedule.mesh() != dimensions.mesh
        || schedule.is_shift() != dimensions.is_shift
        || static_cast<std::uint64_t>(state.n_kpoints())
            != dimensions.n_kpoints
        || schedule.shape().n_kpoints != dimensions.n_kpoints
        || schedule.shape().n_basis != dimensions.n_basis
        || schedule.shape().n_auxiliary != dimensions.n_auxiliary) {
        throw std::logic_error(
            "factor-build upstream identities or dimensions disagree");
    }
    require_sha256(
        state.state_identity_sha256(), "mean-field state identity");
    require_sha256(
        schedule.schedule_identity_sha256(), "factor-stream identity");
}

void require_configuration(
    const PeriodicCorrelationFactorBuildConfig& config,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    std::uint64_t maximum_native_threads) {
    require_sha256(config.ao_basis_identity_sha256, "AO basis identity");
    require_sha256(
        config.auxiliary_basis_identity_sha256,
        "auxiliary basis identity");
    require_sha256(config.producer_identity_sha256, "producer identity");
    require_sha256(config.backend_identity_sha256, "backend identity");
    require_sha256(config.codec_identity_sha256, "codec identity");
    require_finite_positive(
        config.metric_absolute_eigenvalue_threshold,
        "metric_absolute_eigenvalue_threshold");
    require_finite_positive(
        config.integral_absolute_screening_threshold,
        "integral_absolute_screening_threshold");
    require_finite_positive(
        config.reciprocal_energy_cutoff,
        "reciprocal_energy_cutoff");

    if (config.row_convention
        != PeriodicCorrelationFactorRowConvention::
            OriginalAuxiliaryAoHermitianPrincipalPseudoinverseSquareRoot) {
        throw std::invalid_argument(
            "factor-build contract v1 rejects compact metric eigenmodes");
    }
    if (config.transfer_convention
        != PeriodicCorrelationTransferRepresentativeConvention::
            CenteredHalfOpenNegativeNyquist) {
        throw std::invalid_argument(
            "factor-build contract v1 requires centred [-1/2,1/2) "
            "transfers with negative Nyquist");
    }
    const bool sequential_publisher = config.publisher_mode
        == PeriodicCorrelationFactorPublisherMode::CanonicalSequentialExactlyOnce;
    if (config.publisher_mode != PeriodicCorrelationFactorPublisherMode::RandomAccessExactlyOnce
        && !sequential_publisher) {
        throw std::invalid_argument(
            "factor-build contract requires random-access exactly-once or "
            "canonical sequential exactly-once publication");
    }
    if (sequential_publisher && (config.publisher_buffer_count != 0U
        || config.codec.digest_bytes_per_tile != 0U)) {
        throw std::invalid_argument(
            "canonical sequential publication requires borrowed publisher zero "
            "buffers and no in-memory digest table");
    }
    if (config.backing_mode != PeriodicCorrelationFactorBackingMode::Disk
        && config.backing_mode
            != PeriodicCorrelationFactorBackingMode::Memory) {
        throw std::invalid_argument("unknown factor backing mode");
    }
    if (!config.backend.complete) {
        throw std::invalid_argument(
            "factor-build backend byte inventory is incomplete");
    }
    if (!config.codec.complete) {
        throw std::invalid_argument(
            "factor-build codec byte inventory is incomplete");
    }
    if (config.ao_pair_block == 0U
        || config.auxiliary_block == 0U
        || config.reciprocal_block == 0U
        || config.whitener_column_block == 0U
        || config.q_concurrency != 1U
        || config.native_threads == 0U
        || config.native_threads > maximum_native_threads
        || (!sequential_publisher && config.publisher_buffer_count == 0U)) {
        throw std::invalid_argument(
            "factor-build blocks, native threads, and publisher buffers "
            "must be positive, native threads must fit the admitted worker "
            "limit, and q_concurrency must equal one");
    }
    if (config.ao_pair_block != schedule.shape().ao_pair_block
        || config.auxiliary_block != schedule.shape().auxiliary_block
        || config.whitener_column_block > schedule.shape().n_auxiliary) {
        throw std::invalid_argument(
            "factor-build Pb/Ab/Rb blocks do not match or fit the stream");
    }

    const auto& backend = config.backend;
    if (config.producer_mode
        == PeriodicCorrelationFactorProducerMode::
            FullCoulombAllReciprocalReference) {
        if (config.short_range_policy
                != PeriodicCorrelationShortRangePolicy::
                    DisabledAllReciprocal
            || config.short_range_real_space_cutoff != 0.0
            || config.range_separation_omega != 0.0) {
            throw std::invalid_argument(
                "the full-Coulomb reference requires disabled short range "
                "and zero range-separation parameters");
        }
        if (backend.per_thread_short_range_fixed_workspace_bytes != 0U
            || backend.short_range_staging_memory_bytes != 0U
            || backend.short_range_staging_scratch_bytes != 0U
            || backend.folded_source_memory_bytes != 0U
            || backend.folded_source_scratch_bytes != 0U) {
            throw std::invalid_argument(
                "an all-reciprocal producer cannot inventory short-range "
                "workspaces");
        }
    } else if (config.producer_mode
               == PeriodicCorrelationFactorProducerMode::RangeSeparatedGdf) {
        if (config.short_range_policy
                != PeriodicCorrelationShortRangePolicy::
                    DoubleCellRecomputePerKPairReference
            && config.short_range_policy
                != PeriodicCorrelationShortRangePolicy::
                    DoubleCellBvkBlockedTransform) {
            throw std::invalid_argument(
                "range-separated GDF requires a supported double-cell "
                "short-range policy; all-reciprocal RSGDF is forbidden");
        }
        require_finite_positive(
            config.short_range_real_space_cutoff,
            "short_range_real_space_cutoff");
        require_finite_positive(
            config.range_separation_omega,
            "range_separation_omega");
        if (config.short_range_policy
                == PeriodicCorrelationShortRangePolicy::
                    DoubleCellRecomputePerKPairReference
            && (backend.folded_source_memory_bytes != 0U
                || backend.folded_source_scratch_bytes != 0U)) {
            throw std::invalid_argument(
                "per-k-pair short-range recomputation cannot claim a "
                "folded BvK source");
        }
        if (config.short_range_policy
                == PeriodicCorrelationShortRangePolicy::
                    DoubleCellBvkBlockedTransform
            && backend.folded_source_memory_bytes == 0U) {
            throw std::invalid_argument(
                "blocked BvK short-range transformation requires a "
                "retained folded source inventory");
        }
    } else {
        throw std::invalid_argument("unknown factor producer mode");
    }
    if (config.short_range_policy
        == PeriodicCorrelationShortRangePolicy::
            UnsupportedSingleCellGammaSum) {
        throw std::invalid_argument(
            "single-cell Gamma-summed short-range GDF is unsupported");
    }

    const auto& codec = config.codec;
    if (codec.fixed_header_bytes == 0U
        || codec.fixed_footer_bytes == 0U
        || (!sequential_publisher && codec.fixed_manifest_bytes == 0U)
        || codec.integrity_bytes_per_tile == 0U
        || (!sequential_publisher && codec.rank_bytes_per_q == 0U)) {
        throw std::invalid_argument(
            "factor-build codec must inventory nonzero fixed, per-tile, "
            "and per-q metadata");
    }
    if (!sequential_publisher
        && config.backing_mode == PeriodicCorrelationFactorBackingMode::Disk
        && (codec.journal_header_bytes == 0U
            || codec.journal_record_bytes_per_tile == 0U
            || codec.checkpoint_record_bytes == 0U
            || codec.checkpoint_records_per_generation == 0U)) {
        throw std::invalid_argument(
            "disk factor publication requires complete journal and "
            "checkpoint record inventories");
    }
}

std::pair<std::array<int, 3>, std::array<int, 3>> centered_transfer(
    const RegularKMesh& addressing,
    std::uint64_t q_index) {
    const auto q = addressing.transfer_address(
        static_cast<std::size_t>(q_index));
    std::array<int, 3> numerator{};
    std::array<int, 3> wrap{};
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        const int modular = q.doubled[axis];
        const int divisions = addressing.mesh()[axis];
        if (modular >= divisions) {
            numerator[axis] = modular - 2 * divisions;
            wrap[axis] = 1;
        } else {
            numerator[axis] = modular;
            wrap[axis] = 0;
        }
    }
    return {numerator, wrap};
}

void require_q_records(
    const std::vector<PeriodicCorrelationFactorBuildQRecord>& records,
    const PeriodicCorrelationFactorBuildConfig& config,
    const PeriodicCorrelationFactorBuildShape& shape,
    const std::array<int, 3>& mesh) {
    if (records.size() != static_cast<std::size_t>(shape.n_kpoints)) {
        throw std::invalid_argument(
            "factor-build census requires exactly one q record per "
            "full-mesh transfer");
    }
    const RegularKMesh addressing(mesh);
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        if (record.q_index != static_cast<std::uint64_t>(index)) {
            throw std::invalid_argument(
                "factor-build q records must be in strict q-index order");
        }
        const auto expected = centered_transfer(addressing, record.q_index);
        if (record.centered_doubled_numerator != expected.first
            || record.centered_reciprocal_wrap != expected.second) {
            throw std::invalid_argument(
                "factor-build q record violates the centred transfer "
                "or negative-Nyquist convention");
        }
        const bool is_zero = std::all_of(
            expected.first.begin(), expected.first.end(),
            [](int value) { return value == 0; });
        const std::uint64_t expected_zero_count = is_zero ? 1U : 0U;
        if (record.zero_mode_excluded_count != expected_zero_count) {
            throw std::invalid_argument(
                "factor-build q record has an incorrect G+q zero-mode "
                "exclusion count");
        }
        if ((record.base_reciprocal_vector_count == 0U
             && record.tail_reciprocal_vector_count == 0U)
            || record.tail_reciprocal_vector_count
                > std::numeric_limits<std::uint64_t>::max()
                    - record.base_reciprocal_vector_count
            || record.source_manifest_bytes == 0U) {
            throw std::invalid_argument(
                "factor-build q record requires a nonempty reciprocal "
                "source and source manifest");
        }

        const bool short_range = config.producer_mode
            == PeriodicCorrelationFactorProducerMode::RangeSeparatedGdf;
        const bool any_short_range =
            record.short_range_metric_cell_count != 0U
            || record.short_range_metric_task_count != 0U
            || record.short_range_three_center_cell_pair_count != 0U
            || record.short_range_three_center_task_count != 0U;
        if (!short_range && any_short_range) {
            throw std::invalid_argument(
                "all-reciprocal q records cannot contain short-range work");
        }
        if (short_range
            && (record.short_range_metric_cell_count == 0U
                || record.short_range_metric_task_count == 0U
                || record.short_range_three_center_cell_pair_count == 0U
                || record.short_range_three_center_task_count == 0U
                || record.short_range_metric_task_count
                    < record.short_range_metric_cell_count
                || record.short_range_three_center_task_count
                    < record.short_range_three_center_cell_pair_count)) {
            throw std::invalid_argument(
                "range-separated q records require complete double-cell "
                "short-range counts");
        }
    }
}

std::string make_census_identity(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildShape& shape,
    const PeriodicCorrelationFactorBuildConfig& config,
    const std::vector<PeriodicCorrelationFactorBuildQRecord>& records) {
    CanonicalFactorBuildHasher digest(
        kCensusDomain,
        kPeriodicCorrelationFactorBuildAdmissionContractVersion);
    digest.add_u32(reference.contract_version());
    digest.add_u32(schedule.contract_version());
    digest.add_u32(reference.state().digest_version());
    digest.add_u32(reference.dimensions().allocation_contract_version);
    digest.add_string(reference.state().state_identity_sha256());
    digest.add_string(reference.dimensions().calculation_identity);
    digest.add_string(reference.dimensions().allocation_identity);
    digest.add_string(schedule.schedule_identity_sha256());
    for (const int component : reference.dimensions().mesh) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    for (const int component : reference.dimensions().is_shift) {
        digest.add_i32(static_cast<std::int32_t>(component));
    }
    add_shape(digest, shape);
    add_config(digest, config);
    digest.add_u64(static_cast<std::uint64_t>(records.size()));
    for (const auto& record : records) add_q_record(digest, record);
    return digest.finish_hex();
}

void require_census_consistency(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census) {
    require_reference_schedule_consistency(reference, schedule);
    if (!census.state_handle()) {
        throw std::invalid_argument(
            "moved-from factor-build census has no mean-field state");
    }
    if (census.contract_version()
            != kPeriodicCorrelationFactorBuildAdmissionContractVersion
        || census.state_handle().get() != reference.state_handle().get()
        || census.state_handle().get() != schedule.state_handle().get()) {
        throw std::invalid_argument(
            "factor-build census does not belong to the live admitted "
            "reference and schedule");
    }
    if (census.state_identity_sha256()
            != reference.state().state_identity_sha256()
        || census.calculation_identity()
            != reference.dimensions().calculation_identity
        || census.allocation_identity()
            != reference.dimensions().allocation_identity
        || census.schedule_identity_sha256()
            != schedule.schedule_identity_sha256()
        || census.mesh() != reference.dimensions().mesh
        || census.is_shift() != reference.dimensions().is_shift
        || !same_stream_shape(census.shape(), schedule.shape())) {
        throw std::logic_error(
            "factor-build census provenance or shape no longer matches "
            "its upstream contracts");
    }
    require_configuration(
        census.config(), schedule, reference.budget().workers_per_rank);
    require_q_records(
        census.q_records(), census.config(), census.shape(), census.mesh());
    const std::string expected = make_census_identity(
        reference,
        schedule,
        census.shape(),
        census.config(),
        census.q_records());
    if (census.census_identity_sha256() != expected) {
        throw std::logic_error(
            "factor-build census canonical identity verification failed");
    }
}

void add_component_inventory(
    CanonicalFactorBuildHasher& digest,
    const PeriodicCorrelationFactorBuildComponents& components) {
    digest.add_u64(components.admitted_baseline_retained_bytes);
    digest.add_u64(components.in_memory_output_bytes);
    digest.add_u64(components.source_manifest_total_bytes);
    digest.add_u64(components.source_manifest_maximum_bytes);
    digest.add_u64(components.publisher_bitmap_bytes);
    digest.add_u64(components.publisher_digest_table_bytes);
    digest.add_u64(components.publisher_control_bytes);
    digest.add_u64(components.folded_source_retained_bytes);
    digest.add_u64(components.reciprocal_g_panel_bytes);
    digest.add_u64(components.auxiliary_fourier_double_panel_bytes);
    digest.add_u64(components.ao_pair_fourier_panel_bytes);
    digest.add_u64(components.ao_pair_fourier_staging_bytes);
    digest.add_u64(components.metric_matrix_bytes);
    digest.add_u64(components.metric_eigenvector_bytes);
    digest.add_u64(components.metric_eigenvalue_bytes);
    digest.add_u64(components.whitener_matrix_bytes);
    digest.add_u64(components.whitener_column_panel_bytes);
    digest.add_u64(components.raw_three_center_panel_bytes);
    digest.add_u64(components.whitening_input_panel_bytes);
    digest.add_u64(components.whitening_output_panel_bytes);
    digest.add_u64(components.transpose_staging_bytes);
    digest.add_u64(components.publisher_buffer_bytes);
    digest.add_u64(components.threaded_short_range_workspace_bytes);
    digest.add_u64(components.threaded_fourier_workspace_bytes);
    digest.add_u64(components.encoded_generation_bytes);
    digest.add_u64(components.journal_bytes);
    digest.add_u64(components.checkpoint_bytes);
    digest.add_u64(components.disk_generation_bytes);
}

void add_phase(
    PeriodicCorrelationFactorBuildPlan& plan,
    PeriodicCorrelationFactorBuildPhase phase,
    std::uint64_t retained,
    std::initializer_list<std::uint64_t> extras) {
    std::uint64_t extra = 0;
    for (const auto value : extras) {
        extra = checked_add(extra, value, "factor-build phase extras");
    }
    PeriodicCorrelationFactorBuildPhaseEstimate estimate;
    estimate.phase = phase;
    estimate.retained_bytes = retained;
    estimate.phase_extra_bytes = extra;
    estimate.peak_memory_bytes = checked_add(
        retained, extra, "factor-build phase peak memory");
    plan.phases.push_back(estimate);
}

std::string make_plan_identity(
    const PeriodicCorrelationResourceBudget& budget,
    const PeriodicCorrelationFactorBuildPlan& plan) {
    CanonicalFactorBuildHasher digest(
        kPlanDomain,
        kPeriodicCorrelationFactorBuildAdmissionContractVersion);
    digest.add_string(plan.census_identity_sha256);
    digest.add_u64(budget.memory_limit_bytes);
    digest.add_u64(budget.scratch_limit_bytes);
    digest.add_u64(budget.mpi_ranks);
    digest.add_u64(budget.workers_per_rank);
    add_enum(digest, plan.admission);
    digest.add_string(plan.calculation_identity);
    digest.add_string(plan.allocation_identity);
    digest.add_string(plan.schedule_identity_sha256);
    add_shape(digest, plan.shape);
    digest.add_u64(plan.total_base_reciprocal_vectors);
    digest.add_u64(plan.total_tail_reciprocal_vectors);
    digest.add_u64(plan.maximum_reciprocal_vectors_per_q);
    digest.add_u64(plan.total_short_range_metric_tasks);
    digest.add_u64(plan.total_short_range_three_center_tasks);
    digest.add_u64(plan.maximum_short_range_metric_tasks_per_q);
    digest.add_u64(plan.maximum_short_range_three_center_tasks_per_q);
    digest.add_u64(plan.logical_factor_bytes);
    digest.add_u64(plan.encoded_generation_bytes);
    digest.add_u64(plan.modeled_peak_memory_bytes);
    digest.add_u64(plan.required_memory_bytes);
    digest.add_u64(plan.modeled_scratch_bytes);
    digest.add_u64(plan.required_scratch_bytes);
    add_component_inventory(digest, plan.components);
    digest.add_u64(static_cast<std::uint64_t>(plan.phases.size()));
    for (const auto& phase : plan.phases) {
        add_enum(digest, phase.phase);
        digest.add_u64(phase.retained_bytes);
        digest.add_u64(phase.phase_extra_bytes);
        digest.add_u64(phase.peak_memory_bytes);
    }
    return digest.finish_hex();
}

void populate_counts(
    PeriodicCorrelationFactorBuildPlan& plan,
    const PeriodicCorrelationFactorBuildCensus& census) {
    for (const auto& record : census.q_records()) {
        plan.total_base_reciprocal_vectors = checked_add(
            plan.total_base_reciprocal_vectors,
            record.base_reciprocal_vector_count,
            "total base reciprocal-vector count");
        plan.total_tail_reciprocal_vectors = checked_add(
            plan.total_tail_reciprocal_vectors,
            record.tail_reciprocal_vector_count,
            "total tail reciprocal-vector count");
        const auto reciprocal_count = checked_add(
            record.base_reciprocal_vector_count,
            record.tail_reciprocal_vector_count,
            "per-q reciprocal-vector count");
        plan.maximum_reciprocal_vectors_per_q = std::max(
            plan.maximum_reciprocal_vectors_per_q,
            reciprocal_count);
        plan.total_short_range_metric_tasks = checked_add(
            plan.total_short_range_metric_tasks,
            record.short_range_metric_task_count,
            "total short-range metric tasks");
        plan.total_short_range_three_center_tasks = checked_add(
            plan.total_short_range_three_center_tasks,
            record.short_range_three_center_task_count,
            "total short-range three-centre tasks");
        plan.maximum_short_range_metric_tasks_per_q = std::max(
            plan.maximum_short_range_metric_tasks_per_q,
            record.short_range_metric_task_count);
        plan.maximum_short_range_three_center_tasks_per_q = std::max(
            plan.maximum_short_range_three_center_tasks_per_q,
            record.short_range_three_center_task_count);
        plan.components.source_manifest_total_bytes = checked_add(
            plan.components.source_manifest_total_bytes,
            record.source_manifest_bytes,
            "total source-manifest bytes");
        plan.components.source_manifest_maximum_bytes = std::max(
            plan.components.source_manifest_maximum_bytes,
            record.source_manifest_bytes);
    }
}

void populate_components(
    PeriodicCorrelationFactorBuildPlan& plan,
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorBuildCensus& census) {
    const auto& dimensions = reference.dimensions();
    const auto& config = census.config();
    const auto& backend = config.backend;
    const auto& codec = config.codec;
    const auto& shape = census.shape();
    auto& components = plan.components;

    auto baseline = checked_add(
        dimensions.external_bytes,
        dimensions.shared_bytes,
        "admitted external and shared bytes");
    baseline = checked_add(
        baseline,
        dimensions.per_rank_bytes,
        "admitted single-rank retained bytes");
    components.admitted_baseline_retained_bytes = checked_add(
        baseline,
        backend.exact_extra_retained_bytes,
        "factor-build retained bytes");
    components.in_memory_output_bytes =
        config.backing_mode == PeriodicCorrelationFactorBackingMode::Memory
        ? shape.logical_bytes
        : 0U;
    components.publisher_bitmap_bytes = config.publisher_mode
        == PeriodicCorrelationFactorPublisherMode::CanonicalSequentialExactlyOnce
        ? 0U : checked_ceil_divide(shape.tile_count, 8U, "exactly-once publisher bitmap");
    components.publisher_digest_table_bytes = checked_multiply(
        shape.tile_count,
        codec.digest_bytes_per_tile,
        "publisher digest table");
    components.publisher_control_bytes = checked_add(
        checked_add(
            components.publisher_bitmap_bytes,
            components.publisher_digest_table_bytes,
            "publisher bitmap and digest table"),
        backend.exact_extra_control_bytes,
        "publisher and backend control bytes");
    components.folded_source_retained_bytes =
        backend.folded_source_memory_bytes;

    components.reciprocal_g_panel_bytes = checked_multiply(
        kReciprocalVectorRecordBytes,
        shape.reciprocal_block,
        "reciprocal G-vector panel");
    components.auxiliary_fourier_double_panel_bytes = checked_multiply(
        2U * kComplexBytes,
        checked_multiply(
            shape.n_auxiliary,
            shape.reciprocal_block,
            "auxiliary Fourier panel elements"),
        "double-buffered auxiliary Fourier panels");
    components.ao_pair_fourier_panel_bytes = checked_multiply(
        kComplexBytes,
        checked_multiply(
            shape.ao_pair_block,
            shape.reciprocal_block,
            "AO-pair Fourier panel elements"),
        "AO-pair Fourier panel");
    components.ao_pair_fourier_staging_bytes =
        config.ao_pair_fourier_staging_required
        ? components.ao_pair_fourier_panel_bytes
        : 0U;
    const auto auxiliary_square = checked_multiply(
        shape.n_auxiliary,
        shape.n_auxiliary,
        "full auxiliary square");
    components.metric_matrix_bytes = checked_multiply(
        kComplexBytes, auxiliary_square, "complex Coulomb metric");
    components.metric_eigenvector_bytes = components.metric_matrix_bytes;
    components.metric_eigenvalue_bytes = checked_multiply(
        kRealBytes, shape.n_auxiliary, "metric eigenvalues");
    components.whitener_matrix_bytes = components.metric_matrix_bytes;
    components.whitener_column_panel_bytes = checked_multiply(
        kComplexBytes,
        checked_multiply(
            shape.n_auxiliary,
            shape.whitener_column_block,
            "whitener column-panel elements"),
        "whitener column panel");
    const auto full_auxiliary_pair_panel = checked_multiply(
        shape.n_auxiliary,
        shape.ao_pair_block,
        "full-auxiliary AO-pair panel elements");
    components.raw_three_center_panel_bytes = checked_multiply(
        kComplexBytes,
        full_auxiliary_pair_panel,
        "full-A raw three-centre panel");
    components.whitening_input_panel_bytes = checked_multiply(
        kComplexBytes,
        full_auxiliary_pair_panel,
        "full-A whitening input panel");
    components.whitening_output_panel_bytes = checked_multiply(
        kComplexBytes,
        full_auxiliary_pair_panel,
        "full-A whitening output panel");
    components.transpose_staging_bytes = config.transpose_before_publish
        ? components.whitening_output_panel_bytes
        : 0U;
    components.publisher_buffer_bytes = checked_multiply(
        config.publisher_buffer_count,
        shape.maximum_tile_bytes,
        "publisher tile buffers");
    components.threaded_short_range_workspace_bytes = checked_multiply(
        config.native_threads,
        backend.per_thread_short_range_fixed_workspace_bytes,
        "threaded short-range fixed workspace");
    components.threaded_fourier_workspace_bytes = checked_multiply(
        config.native_threads,
        backend.per_thread_fourier_transform_fixed_workspace_bytes,
        "threaded Fourier fixed workspace");

    auto encoded = checked_add(
        shape.logical_bytes,
        codec.fixed_header_bytes,
        "encoded factor header");
    encoded = checked_add(
        encoded, codec.fixed_footer_bytes, "encoded factor footer");
    encoded = checked_add(
        encoded, codec.fixed_manifest_bytes, "encoded factor manifest");
    encoded = checked_add(
        encoded,
        checked_multiply(
            shape.tile_count,
            codec.integrity_bytes_per_tile,
            "encoded per-tile integrity bytes"),
        "encoded factor tile integrity");
    encoded = checked_add(
        encoded,
        checked_multiply(
            shape.n_kpoints,
            codec.rank_bytes_per_q,
            "encoded per-q retained ranks"),
        "encoded factor q ranks");
    components.encoded_generation_bytes = encoded;
    plan.encoded_generation_bytes = encoded;

    components.journal_bytes = checked_add(
        codec.journal_header_bytes,
        checked_multiply(
            shape.tile_count,
            codec.journal_record_bytes_per_tile,
            "publisher journal records"),
        "publisher journal");
    const auto scratch_generations = checked_add(
        codec.existing_generation_count,
        1U,
        "existing plus temporary factor generations");
    components.checkpoint_bytes = checked_multiply(
        checked_multiply(
            codec.checkpoint_record_bytes,
            codec.checkpoint_records_per_generation,
            "checkpoint bytes per generation"),
        scratch_generations,
        "checkpoint bytes across generations");
    components.disk_generation_bytes =
        config.backing_mode == PeriodicCorrelationFactorBackingMode::Disk
        ? checked_multiply(
              encoded,
              scratch_generations,
              "existing and temporary encoded generations")
        : 0U;
}

void populate_phases_and_caps(
    PeriodicCorrelationFactorBuildPlan& plan,
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorBuildCensus& census) {
    const auto& config = census.config();
    const auto& backend = config.backend;
    const auto& c = plan.components;
    auto retained = checked_add(
        c.admitted_baseline_retained_bytes,
        c.in_memory_output_bytes,
        "baseline and in-memory factor output");
    retained = checked_add(
        retained,
        c.publisher_control_bytes,
        "retained publisher control");
    retained = checked_add(
        retained,
        c.folded_source_retained_bytes,
        "retained folded reciprocal source");
    const auto manifest = c.source_manifest_maximum_bytes;

    add_phase(
        plan,
        PeriodicCorrelationFactorBuildPhase::ReciprocalMetric,
        retained,
        {manifest,
         c.metric_matrix_bytes,
         c.reciprocal_g_panel_bytes,
         c.auxiliary_fourier_double_panel_bytes,
         c.threaded_fourier_workspace_bytes});
    if (config.producer_mode
        == PeriodicCorrelationFactorProducerMode::RangeSeparatedGdf) {
        add_phase(
            plan,
            PeriodicCorrelationFactorBuildPhase::ShortRangeMetric,
            retained,
            {manifest,
             c.metric_matrix_bytes,
             c.threaded_short_range_workspace_bytes,
             backend.short_range_staging_memory_bytes});
    }
    add_phase(
        plan,
        PeriodicCorrelationFactorBuildPhase::MetricFactorization,
        retained,
        {manifest,
         c.metric_matrix_bytes,
         c.metric_eigenvector_bytes,
         c.metric_eigenvalue_bytes,
         backend.eigensolver_workspace_bytes});
    add_phase(
        plan,
        PeriodicCorrelationFactorBuildPhase::MetricWhitener,
        retained,
        {manifest,
         c.metric_eigenvector_bytes,
         c.metric_eigenvalue_bytes,
         c.whitener_matrix_bytes,
         c.whitener_column_panel_bytes,
         c.whitener_column_panel_bytes,
         backend.whitener_workspace_bytes});
    add_phase(
        plan,
        PeriodicCorrelationFactorBuildPhase::ReciprocalThreeCenter,
        retained,
        {manifest,
         c.whitener_matrix_bytes,
         c.raw_three_center_panel_bytes,
         c.reciprocal_g_panel_bytes,
         c.auxiliary_fourier_double_panel_bytes,
         c.ao_pair_fourier_panel_bytes,
         c.ao_pair_fourier_staging_bytes,
         c.threaded_fourier_workspace_bytes});
    if (config.producer_mode
        == PeriodicCorrelationFactorProducerMode::RangeSeparatedGdf) {
        add_phase(
            plan,
            PeriodicCorrelationFactorBuildPhase::ShortRangeThreeCenter,
            retained,
            {manifest,
             c.whitener_matrix_bytes,
             c.raw_three_center_panel_bytes,
             c.threaded_short_range_workspace_bytes,
             backend.short_range_staging_memory_bytes});
    }
    add_phase(
        plan,
        PeriodicCorrelationFactorBuildPhase::Whitening,
        retained,
        {manifest,
         c.whitener_matrix_bytes,
         c.whitener_column_panel_bytes,
         c.whitening_input_panel_bytes,
         c.whitening_output_panel_bytes,
         backend.gemm_workspace_bytes});
    if (config.transpose_before_publish) {
        add_phase(
            plan,
            PeriodicCorrelationFactorBuildPhase::Transpose,
            retained,
            {manifest,
             c.whitener_matrix_bytes,
             c.whitening_output_panel_bytes,
             c.transpose_staging_bytes});
    }
    add_phase(
        plan,
        PeriodicCorrelationFactorBuildPhase::Publish,
        retained,
        {manifest,
         c.whitener_matrix_bytes,
         c.whitening_output_panel_bytes,
         c.transpose_staging_bytes,
         c.publisher_buffer_bytes,
         backend.publisher_workspace_bytes});

    for (const auto& phase : plan.phases) {
        plan.modeled_peak_memory_bytes = std::max(
            plan.modeled_peak_memory_bytes, phase.peak_memory_bytes);
    }
    plan.required_memory_bytes = required_memory(
        plan.modeled_peak_memory_bytes);

    auto scratch = checked_add(
        backend.short_range_staging_scratch_bytes,
        backend.folded_source_scratch_bytes,
        "factor source scratch");
    scratch = checked_add(
        scratch,
        backend.exact_extra_scratch_bytes,
        "factor backend extra scratch");
    if (config.backing_mode == PeriodicCorrelationFactorBackingMode::Disk) {
        scratch = checked_add(
            scratch,
            c.disk_generation_bytes,
            "factor generation scratch");
        scratch = checked_add(
            scratch, c.journal_bytes, "factor journal scratch");
        scratch = checked_add(
            scratch, c.checkpoint_bytes, "factor checkpoint scratch");
    }
    plan.modeled_scratch_bytes = scratch;
    plan.required_scratch_bytes = required_scratch(scratch);

    const auto& budget = reference.budget();
    if (budget.memory_limit_bytes == 0U) {
        plan.admission =
            PeriodicCorrelationFactorBuildAdmissionCode::MissingMemoryLimit;
    } else if (plan.required_memory_bytes > budget.memory_limit_bytes) {
        plan.admission =
            PeriodicCorrelationFactorBuildAdmissionCode::MemoryExceeded;
    } else if (plan.modeled_scratch_bytes != 0U
               && budget.scratch_limit_bytes == 0U) {
        plan.admission =
            PeriodicCorrelationFactorBuildAdmissionCode::MissingScratchLimit;
    } else if (plan.required_scratch_bytes > budget.scratch_limit_bytes) {
        plan.admission =
            PeriodicCorrelationFactorBuildAdmissionCode::ScratchExceeded;
    } else {
        plan.admission =
            PeriodicCorrelationFactorBuildAdmissionCode::Admitted;
    }
}

}  // namespace

PeriodicCorrelationFactorBuildCensus::
    PeriodicCorrelationFactorBuildCensus(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::string state_identity_sha256,
        std::string calculation_identity,
        std::string allocation_identity,
        std::string schedule_identity_sha256,
        std::string census_identity_sha256,
        std::array<int, 3> mesh,
        std::array<int, 3> is_shift,
        PeriodicCorrelationFactorBuildShape shape,
        PeriodicCorrelationFactorBuildConfig config,
        std::vector<PeriodicCorrelationFactorBuildQRecord> q_records)
    : state_(std::move(state)),
      state_identity_sha256_(std::move(state_identity_sha256)),
      calculation_identity_(std::move(calculation_identity)),
      allocation_identity_(std::move(allocation_identity)),
      schedule_identity_sha256_(std::move(schedule_identity_sha256)),
      census_identity_sha256_(std::move(census_identity_sha256)),
      mesh_(mesh),
      is_shift_(is_shift),
      shape_(shape),
      config_(std::move(config)),
      q_records_(std::move(q_records)) {}

PeriodicCorrelationFactorBuildCensus
make_periodic_correlation_factor_build_census(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    PeriodicCorrelationFactorBuildConfig config,
    std::vector<PeriodicCorrelationFactorBuildQRecord> q_records) {
    require_reference_schedule_consistency(reference, schedule);
    require_configuration(
        config, schedule, reference.budget().workers_per_rank);
    const auto shape = make_build_shape(schedule, config);
    if (!same_stream_shape(shape, schedule.shape())) {
        throw std::logic_error(
            "factor-build shape does not reproduce the stream schedule");
    }
    require_q_records(
        q_records, config, shape, reference.dimensions().mesh);
    const std::string census_identity = make_census_identity(
        reference, schedule, shape, config, q_records);
    return PeriodicCorrelationFactorBuildCensus(
        reference.state_handle(),
        reference.state().state_identity_sha256(),
        reference.dimensions().calculation_identity,
        reference.dimensions().allocation_identity,
        schedule.schedule_identity_sha256(),
        census_identity,
        reference.dimensions().mesh,
        reference.dimensions().is_shift,
        shape,
        std::move(config),
        std::move(q_records));
}

PeriodicCorrelationFactorBuildPlan
plan_periodic_correlation_factor_build(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census) {
    require_census_consistency(reference, schedule, census);

    PeriodicCorrelationFactorBuildPlan plan;
    plan.calculation_identity = census.calculation_identity();
    plan.allocation_identity = census.allocation_identity();
    plan.schedule_identity_sha256 = census.schedule_identity_sha256();
    plan.census_identity_sha256 = census.census_identity_sha256();
    plan.shape = census.shape();
    plan.logical_factor_bytes = census.shape().logical_bytes;
    try {
        populate_counts(plan, census);
        populate_components(plan, reference, census);
        populate_phases_and_caps(plan, reference, census);
        plan.plan_identity_sha256 = make_plan_identity(
            reference.budget(), plan);
    } catch (const FactorBuildArithmeticOverflow& error) {
        plan.admission =
            PeriodicCorrelationFactorBuildAdmissionCode::ArithmeticOverflow;
        plan.failure_detail = error.what();
        plan.plan_identity_sha256.clear();
    }
    return plan;
}

static_assert(
    !std::is_copy_constructible<
        PeriodicCorrelationFactorBuildCensus>::value,
    "a factor-build census must not copy its retained state");
static_assert(
    std::is_nothrow_move_constructible<
        PeriodicCorrelationFactorBuildCensus>::value,
    "a factor-build census must be cheaply movable");

}  // namespace vibeqc
