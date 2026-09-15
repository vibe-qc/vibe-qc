#pragma once

/// \file periodic_correlation_factor_build_admission.hpp
/// \brief Count-only admission contract for numerical periodic DF factors.
///
/// Contract version 1 sizes a future native numerical factor producer before
/// it allocates a Coulomb metric, a three-centre panel, or an output store. It
/// does not evaluate integrals and is not a numerical factor producer. The
/// only admitted output row convention is the original auxiliary-AO basis,
/// with the Coulomb metric transformed by its Hermitian principal
/// pseudoinverse square root. A compact retained-eigenmode row axis is
/// deliberately rejected because its q-dependent gauge is not yet defined.
///
/// Transfer momenta use the unique centred representative in [-1/2, 1/2) on
/// every reciprocal axis. An even-mesh Nyquist point is therefore represented
/// on the negative face. The census factory checks every supplied q record
/// against RegularKMesh; supplied records must be in increasing q-index order.
/// The records are counts and provenance only. Production must obtain them
/// from a native reciprocal/real-space source inventory, not from this API.
/// Metric diagonalisation and construction of the principal whitener are
/// separate sequential phases. The completed full-A-by-A whitener remains
/// live through three-centre assembly, whitening, transpose, and publication;
/// publication also retains the full-A-by-Pb output panel. Any backend-reported
/// folded reciprocal source memory is retained across every phase.
/// The reciprocal work panel is exactly five structure-of-arrays binary64
/// lanes per active vector: three Cartesian G+q components, |G+q| squared,
/// and the producer's screened Coulomb-kernel weight. It has no padding.
/// The two admitted auxiliary Fourier panels are one native F_P(G+q) panel
/// and one equally sized weighted-conjugate panel used by metric/three-centre
/// contractions; they are simultaneous allocations, not a ping-pong pair.
/// The raw full-A-by-Pb three-centre allocation is reclassified in place as
/// `whitening_input_panel_bytes` after source assembly; those two component
/// names describe consecutive lifetimes of one allocation, not two live
/// panels. Whitening additionally requires a distinct full-A-by-Pb output.
///
/// Canonical SHA-256 wire
/// ---------------------
/// Integers are unsigned big-endian u32/u64; i32 uses its two's-complement
/// bits as u32. Booleans and enums are u32. Strings are a u64 byte length
/// followed by bytes. Finite binary64 values are their big-endian IEEE-754
/// bits after normalising either signed zero to +0.0. There is no padding or
/// hashing of native object representation.
///
/// The census domain is
/// "vibeqc.periodic.correlation.factor-build.census" followed by schema
/// version 1, admitted-reference and factor-stream contract versions, state
/// digest version, allocation contract version, state/calculation/allocation/
/// schedule identities, mesh and shift, the
/// PeriodicCorrelationFactorBuildShape fields in declaration order, all
/// configuration, backend-inventory and codec-inventory fields in declaration
/// order, then the q-record count and every q record in declaration order.
///
/// The plan domain is
/// "vibeqc.periodic.correlation.factor-build.plan" followed by schema version
/// 1, census identity, the two resource limits and concurrency fields, the
/// admission code, calculation/allocation/schedule identities, the
/// complete PeriodicCorrelationFactorBuildShape, all plan scalar counts
/// through required scratch bytes, the component fields in declaration order,
/// then every phase record in order.
/// Arithmetic-overflow plans intentionally have an empty plan identity because
/// their canonical scalar inventory could not be completed.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_admitted_reference.hpp"
#include "vibeqc/periodic_correlation_factor_stream.hpp"

namespace vibeqc {

inline constexpr std::uint32_t
    kPeriodicCorrelationFactorBuildAdmissionContractVersion = 1;
inline constexpr PeriodicCorrelationByteCount
    kPeriodicCorrelationFactorBuildComplexBytes = 16U;
inline constexpr PeriodicCorrelationByteCount
    kPeriodicCorrelationFactorBuildRealBytes = 8U;
inline constexpr PeriodicCorrelationByteCount
    kPeriodicCorrelationFactorBuildReciprocalVectorRecordBytes = 40U;

enum class PeriodicCorrelationFactorProducerMode : std::uint32_t {
    FullCoulombAllReciprocalReference = 0,
    RangeSeparatedGdf = 1,
};

enum class PeriodicCorrelationShortRangePolicy : std::uint32_t {
    DisabledAllReciprocal = 0,
    DoubleCellRecomputePerKPairReference = 1,
    DoubleCellBvkBlockedTransform = 2,
    UnsupportedSingleCellGammaSum = 3,
};

enum class PeriodicCorrelationFactorRowConvention : std::uint32_t {
    OriginalAuxiliaryAoHermitianPrincipalPseudoinverseSquareRoot = 0,
    CompactMetricEigenmodesUnsupported = 1,
};

enum class PeriodicCorrelationTransferRepresentativeConvention
    : std::uint32_t {
    CenteredHalfOpenNegativeNyquist = 0,
    UnspecifiedUnsupported = 1,
};

enum class PeriodicCorrelationFactorBackingMode : std::uint32_t {
    Disk = 0,
    Memory = 1,
};

enum class PeriodicCorrelationFactorPublisherMode : std::uint32_t {
    RandomAccessExactlyOnce = 0,
    SequentialAppendUnsupported = 1,
    /// Canonical sequential borrowed tiles: no RAM bitmap/digest table and
    /// no copied publisher buffer. A concrete sink requires its own gate.
    CanonicalSequentialExactlyOnce = 2,
};

/// Exact source work counts for one transfer momentum.
struct PeriodicCorrelationFactorBuildQRecord {
    std::uint64_t q_index = 0;
    std::array<int, 3> centered_doubled_numerator = {0, 0, 0};
    /// Integer wrap satisfying modular_q = centered_q + wrap.
    std::array<int, 3> centered_reciprocal_wrap = {0, 0, 0};
    std::uint64_t base_reciprocal_vector_count = 0;
    std::uint64_t tail_reciprocal_vector_count = 0;
    /// Exactly one only when G + q = 0 is excluded, otherwise zero.
    std::uint64_t zero_mode_excluded_count = 0;
    std::uint64_t short_range_metric_cell_count = 0;
    std::uint64_t short_range_metric_task_count = 0;
    std::uint64_t short_range_three_center_cell_pair_count = 0;
    std::uint64_t short_range_three_center_task_count = 0;
    /// Maximum live native source-manifest payload for this q.
    PeriodicCorrelationByteCount source_manifest_bytes = 0;
};

/// Exact backend-owned bytes not derivable from mathematical panel shapes.
/// Eigensolver workspace belongs to MetricFactorization, whitener workspace
/// to MetricWhitener, and GEMM workspace to Whitening. Per-thread source
/// workspaces belong to their corresponding metric or three-centre source
/// phases. Publisher workspace is live only during Publish.
struct PeriodicCorrelationFactorBuildBackendInventory {
    bool complete = false;
    PeriodicCorrelationByteCount
        per_thread_short_range_fixed_workspace_bytes = 0;
    PeriodicCorrelationByteCount
        per_thread_fourier_transform_fixed_workspace_bytes = 0;
    PeriodicCorrelationByteCount eigensolver_workspace_bytes = 0;
    PeriodicCorrelationByteCount whitener_workspace_bytes = 0;
    PeriodicCorrelationByteCount gemm_workspace_bytes = 0;
    PeriodicCorrelationByteCount publisher_workspace_bytes = 0;
    PeriodicCorrelationByteCount short_range_staging_memory_bytes = 0;
    /// Required and retained for DoubleCellBvkBlockedTransform; required to
    /// be zero for the per-k-pair recompute and all-reciprocal policies.
    PeriodicCorrelationByteCount folded_source_memory_bytes = 0;
    PeriodicCorrelationByteCount short_range_staging_scratch_bytes = 0;
    PeriodicCorrelationByteCount folded_source_scratch_bytes = 0;
    PeriodicCorrelationByteCount exact_extra_retained_bytes = 0;
    PeriodicCorrelationByteCount exact_extra_control_bytes = 0;
    PeriodicCorrelationByteCount exact_extra_scratch_bytes = 0;
};

/// Exact serialized-store overheads. Logical complex payload is added by the
/// planner. `existing_generation_count` counts complete generations already
/// kept on disk; disk admission additionally reserves one temporary encoded
/// generation for atomic publication.
struct PeriodicCorrelationFactorBuildCodecInventory {
    bool complete = false;
    PeriodicCorrelationByteCount fixed_header_bytes = 0;
    PeriodicCorrelationByteCount fixed_footer_bytes = 0;
    PeriodicCorrelationByteCount fixed_manifest_bytes = 0;
    PeriodicCorrelationByteCount integrity_bytes_per_tile = 0;
    PeriodicCorrelationByteCount rank_bytes_per_q = 0;
    /// Zero disables the in-memory per-tile digest table. Nonzero values are
    /// multiplied by the exact tile count.
    PeriodicCorrelationByteCount digest_bytes_per_tile = 0;
    PeriodicCorrelationByteCount journal_header_bytes = 0;
    PeriodicCorrelationByteCount journal_record_bytes_per_tile = 0;
    PeriodicCorrelationByteCount checkpoint_record_bytes = 0;
    std::uint64_t checkpoint_records_per_generation = 0;
    std::uint64_t existing_generation_count = 0;
};

struct PeriodicCorrelationFactorBuildConfig {
    PeriodicCorrelationFactorProducerMode producer_mode =
        PeriodicCorrelationFactorProducerMode::
            FullCoulombAllReciprocalReference;
    PeriodicCorrelationShortRangePolicy short_range_policy =
        PeriodicCorrelationShortRangePolicy::DisabledAllReciprocal;
    PeriodicCorrelationFactorRowConvention row_convention =
        PeriodicCorrelationFactorRowConvention::
            OriginalAuxiliaryAoHermitianPrincipalPseudoinverseSquareRoot;
    PeriodicCorrelationTransferRepresentativeConvention transfer_convention =
        PeriodicCorrelationTransferRepresentativeConvention::
            CenteredHalfOpenNegativeNyquist;
    PeriodicCorrelationFactorBackingMode backing_mode =
        PeriodicCorrelationFactorBackingMode::Disk;
    PeriodicCorrelationFactorPublisherMode publisher_mode =
        PeriodicCorrelationFactorPublisherMode::RandomAccessExactlyOnce;

    std::string ao_basis_identity_sha256;
    std::string auxiliary_basis_identity_sha256;
    std::string producer_identity_sha256;
    std::string backend_identity_sha256;
    std::string codec_identity_sha256;

    double metric_absolute_eigenvalue_threshold = 0.0;
    double integral_absolute_screening_threshold = 0.0;
    double reciprocal_energy_cutoff = 0.0;
    /// Positive only for RangeSeparatedGdf; exactly zero for the full-Coulomb
    /// all-reciprocal reference producer.
    double short_range_real_space_cutoff = 0.0;
    double range_separation_omega = 0.0;

    /// Pb, Ab, G, and Rb respectively. Pb/Ab must match the stream schedule.
    std::uint64_t ao_pair_block = 0;
    std::uint64_t auxiliary_block = 0;
    std::uint64_t reciprocal_block = 0;
    std::uint64_t whitener_column_block = 0;
    std::uint64_t q_concurrency = 1;
    std::uint64_t native_threads = 1;
    bool ao_pair_fourier_staging_required = false;
    bool transpose_before_publish = true;
    std::uint64_t publisher_buffer_count = 2;

    PeriodicCorrelationFactorBuildBackendInventory backend;
    PeriodicCorrelationFactorBuildCodecInventory codec;
};

/// Checked shape copied from, and required to equal, the existing stream.
struct PeriodicCorrelationFactorBuildShape {
    std::uint64_t n_kpoints = 0;
    std::uint64_t n_basis = 0;
    std::uint64_t n_auxiliary = 0;
    std::uint64_t n_ao_pairs = 0;
    std::uint64_t ao_pair_block = 0;
    std::uint64_t auxiliary_block = 0;
    std::uint64_t reciprocal_block = 0;
    std::uint64_t whitener_column_block = 0;
    std::uint64_t tile_count = 0;
    std::uint64_t logical_element_count = 0;
    PeriodicCorrelationByteCount logical_bytes = 0;
    PeriodicCorrelationByteCount maximum_tile_bytes = 0;
};

/// Immutable resource census tied to one admitted state and stream. Its
/// identity seals source counts and configuration identities, not the exact
/// numerical G-vector/cell manifests or factor values. Numerical restart or
/// reuse therefore requires a separate source/payload identity contract.
class PeriodicCorrelationFactorBuildCensus {
public:
    PeriodicCorrelationFactorBuildCensus(
        const PeriodicCorrelationFactorBuildCensus&) = delete;
    PeriodicCorrelationFactorBuildCensus& operator=(
        const PeriodicCorrelationFactorBuildCensus&) = delete;
    PeriodicCorrelationFactorBuildCensus(
        PeriodicCorrelationFactorBuildCensus&&) noexcept = default;
    PeriodicCorrelationFactorBuildCensus& operator=(
        PeriodicCorrelationFactorBuildCensus&&) noexcept = default;
    ~PeriodicCorrelationFactorBuildCensus() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationFactorBuildAdmissionContractVersion;
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
    state_handle() const noexcept { return state_; }
    const std::string& state_identity_sha256() const noexcept {
        return state_identity_sha256_;
    }
    const std::string& calculation_identity() const noexcept {
        return calculation_identity_;
    }
    const std::string& allocation_identity() const noexcept {
        return allocation_identity_;
    }
    const std::string& schedule_identity_sha256() const noexcept {
        return schedule_identity_sha256_;
    }
    const std::string& census_identity_sha256() const noexcept {
        return census_identity_sha256_;
    }
    const std::array<int, 3>& mesh() const noexcept { return mesh_; }
    const std::array<int, 3>& is_shift() const noexcept { return is_shift_; }
    const PeriodicCorrelationFactorBuildShape& shape() const noexcept {
        return shape_;
    }
    const PeriodicCorrelationFactorBuildConfig& config() const noexcept {
        return config_;
    }
    const std::vector<PeriodicCorrelationFactorBuildQRecord>&
    q_records() const noexcept { return q_records_; }

private:
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
        std::vector<PeriodicCorrelationFactorBuildQRecord> q_records);

    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::string state_identity_sha256_;
    std::string calculation_identity_;
    std::string allocation_identity_;
    std::string schedule_identity_sha256_;
    std::string census_identity_sha256_;
    std::array<int, 3> mesh_ = {0, 0, 0};
    std::array<int, 3> is_shift_ = {0, 0, 0};
    PeriodicCorrelationFactorBuildShape shape_;
    PeriodicCorrelationFactorBuildConfig config_;
    std::vector<PeriodicCorrelationFactorBuildQRecord> q_records_;

    friend PeriodicCorrelationFactorBuildCensus
    make_periodic_correlation_factor_build_census(
        const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationFactorStreamSchedule& schedule,
        PeriodicCorrelationFactorBuildConfig config,
        std::vector<PeriodicCorrelationFactorBuildQRecord> q_records);
};

PeriodicCorrelationFactorBuildCensus
make_periodic_correlation_factor_build_census(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    PeriodicCorrelationFactorBuildConfig config,
    std::vector<PeriodicCorrelationFactorBuildQRecord> q_records);

enum class PeriodicCorrelationFactorBuildPhase : std::uint32_t {
    ReciprocalMetric = 0,
    ShortRangeMetric = 1,
    MetricFactorization = 2,
    MetricWhitener = 3,
    ReciprocalThreeCenter = 4,
    ShortRangeThreeCenter = 5,
    Whitening = 6,
    Transpose = 7,
    Publish = 8,
};

enum class PeriodicCorrelationFactorBuildAdmissionCode : std::uint32_t {
    Admitted = 0,
    MissingMemoryLimit = 1,
    MissingScratchLimit = 2,
    MemoryExceeded = 3,
    ScratchExceeded = 4,
    ArithmeticOverflow = 5,
};

/// Auditable scalar decomposition. z=16 bytes per complex, r=8 bytes per real
/// binary64 value, and five real lanes (40 bytes) per reciprocal work record
/// are fixed by contract version 1.
struct PeriodicCorrelationFactorBuildComponents {
    PeriodicCorrelationByteCount admitted_baseline_retained_bytes = 0;
    PeriodicCorrelationByteCount in_memory_output_bytes = 0;
    PeriodicCorrelationByteCount source_manifest_total_bytes = 0;
    PeriodicCorrelationByteCount source_manifest_maximum_bytes = 0;
    PeriodicCorrelationByteCount publisher_bitmap_bytes = 0;
    PeriodicCorrelationByteCount publisher_digest_table_bytes = 0;
    PeriodicCorrelationByteCount publisher_control_bytes = 0;
    PeriodicCorrelationByteCount folded_source_retained_bytes = 0;
    PeriodicCorrelationByteCount reciprocal_g_panel_bytes = 0;
    PeriodicCorrelationByteCount auxiliary_fourier_double_panel_bytes = 0;
    PeriodicCorrelationByteCount ao_pair_fourier_panel_bytes = 0;
    PeriodicCorrelationByteCount ao_pair_fourier_staging_bytes = 0;
    PeriodicCorrelationByteCount metric_matrix_bytes = 0;
    PeriodicCorrelationByteCount metric_eigenvector_bytes = 0;
    PeriodicCorrelationByteCount metric_eigenvalue_bytes = 0;
    PeriodicCorrelationByteCount whitener_matrix_bytes = 0;
    PeriodicCorrelationByteCount whitener_column_panel_bytes = 0;
    PeriodicCorrelationByteCount raw_three_center_panel_bytes = 0;
    PeriodicCorrelationByteCount whitening_input_panel_bytes = 0;
    PeriodicCorrelationByteCount whitening_output_panel_bytes = 0;
    PeriodicCorrelationByteCount transpose_staging_bytes = 0;
    PeriodicCorrelationByteCount publisher_buffer_bytes = 0;
    PeriodicCorrelationByteCount threaded_short_range_workspace_bytes = 0;
    PeriodicCorrelationByteCount threaded_fourier_workspace_bytes = 0;
    PeriodicCorrelationByteCount encoded_generation_bytes = 0;
    PeriodicCorrelationByteCount journal_bytes = 0;
    PeriodicCorrelationByteCount checkpoint_bytes = 0;
    PeriodicCorrelationByteCount disk_generation_bytes = 0;
};

struct PeriodicCorrelationFactorBuildPhaseEstimate {
    PeriodicCorrelationFactorBuildPhase phase =
        PeriodicCorrelationFactorBuildPhase::ReciprocalMetric;
    PeriodicCorrelationByteCount retained_bytes = 0;
    PeriodicCorrelationByteCount phase_extra_bytes = 0;
    PeriodicCorrelationByteCount peak_memory_bytes = 0;
};

struct PeriodicCorrelationFactorBuildPlan {
    std::uint32_t contract_version =
        kPeriodicCorrelationFactorBuildAdmissionContractVersion;
    PeriodicCorrelationFactorBuildAdmissionCode admission =
        PeriodicCorrelationFactorBuildAdmissionCode::MissingMemoryLimit;
    std::string calculation_identity;
    std::string allocation_identity;
    std::string schedule_identity_sha256;
    std::string census_identity_sha256;
    std::string plan_identity_sha256;
    PeriodicCorrelationFactorBuildShape shape;
    std::uint64_t total_base_reciprocal_vectors = 0;
    std::uint64_t total_tail_reciprocal_vectors = 0;
    std::uint64_t maximum_reciprocal_vectors_per_q = 0;
    std::uint64_t total_short_range_metric_tasks = 0;
    std::uint64_t total_short_range_three_center_tasks = 0;
    std::uint64_t maximum_short_range_metric_tasks_per_q = 0;
    std::uint64_t maximum_short_range_three_center_tasks_per_q = 0;
    PeriodicCorrelationByteCount logical_factor_bytes = 0;
    PeriodicCorrelationByteCount encoded_generation_bytes = 0;
    PeriodicCorrelationByteCount modeled_peak_memory_bytes = 0;
    PeriodicCorrelationByteCount required_memory_bytes = 0;
    PeriodicCorrelationByteCount modeled_scratch_bytes = 0;
    PeriodicCorrelationByteCount required_scratch_bytes = 0;
    PeriodicCorrelationFactorBuildComponents components;
    std::vector<PeriodicCorrelationFactorBuildPhaseEstimate> phases;
    /// Set only for ArithmeticOverflow; the C++ core emits no output.
    std::string failure_detail;
};

/// Re-validate all provenance against the live reference and schedule, then
/// apply the reference's node budget. Semantic contract violations throw;
/// checked byte/count overflow returns ArithmeticOverflow.
PeriodicCorrelationFactorBuildPlan
plan_periodic_correlation_factor_build(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census);

}  // namespace vibeqc
