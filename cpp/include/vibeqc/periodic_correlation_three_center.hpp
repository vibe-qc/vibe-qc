#pragma once

// Actual finite-source three-center factors, not the synthetic stream pattern.
// Sun et al. (2017), doi:10.1063/1.4998644, Eqs. 13, 16, 21:
//   T_P,mu nu = sum_p [4*pi/(Omega*p^2)] conj(F_P(p)) rho_mu nu(p;k_ket),
//   L = W T,  W = the original-auxiliary-AO principal metric pseudoinverse root.
// The exact reciprocal source is shared with W; q = k_ket - k_bra mod G.
// AO images use the explicitly finite numerical policy of aopair_ft.hpp.
// This is not a certification of the infinite-image tail or production DLPNO.

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_metric_factorization.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationThreeCenterTileContractVersion = 1U;
inline constexpr char kPeriodicCorrelationThreeCenterImagePolicy[] =
    "finite-pair-separation-cutoff;padded-long-double-box;lexicographic;binary64-v1";
inline constexpr char kPeriodicCorrelationThreeCenterLatticePolicy[] =
    "direct-lattice=2*pi*inverse(sealed-reciprocal-lattice).transpose;binary64-v1";

class PeriodicCorrelationThreeCenterTile {
public:
    PeriodicCorrelationThreeCenterTile(const PeriodicCorrelationThreeCenterTile&) = delete;
    PeriodicCorrelationThreeCenterTile& operator=(const PeriodicCorrelationThreeCenterTile&) = delete;
    PeriodicCorrelationThreeCenterTile(PeriodicCorrelationThreeCenterTile&&) noexcept = default;
    PeriodicCorrelationThreeCenterTile& operator=(PeriodicCorrelationThreeCenterTile&&) noexcept = default;

    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationThreeCenterTileContractVersion; }
    bool finite_image_reference() const noexcept { return true; }
    bool ao_image_source_certified() const noexcept { return false; }
    const PeriodicCorrelationFactorTileDescriptor& descriptor() const noexcept { return descriptor_; }
    const std::vector<std::complex<double>>& matrix_row_major() const noexcept { return values_; }
    double image_cutoff_bohr() const noexcept { return image_cutoff_bohr_; }
    std::uint64_t image_candidate_count() const noexcept { return image_candidate_count_; }
    std::uint64_t retained_pair_image_count() const noexcept { return retained_pair_image_count_; }
    std::uint64_t reciprocal_vector_count() const noexcept { return reciprocal_vector_count_; }
    std::uint64_t output_bytes() const noexcept { return output_bytes_; }
    std::uint64_t numerical_peak_bytes() const noexcept { return numerical_peak_bytes_; }
    std::uint64_t admitted_assembly_peak_bytes() const noexcept { return admitted_assembly_peak_bytes_; }
    std::uint64_t admitted_whitening_peak_bytes() const noexcept { return admitted_whitening_peak_bytes_; }
    std::string source_identity_sha256() const;
    std::string whitener_payload_identity_sha256() const;
    std::string ao_basis_identity_sha256() const;
    std::string auxiliary_basis_identity_sha256() const;
    std::string census_identity_sha256() const;
    std::string plan_identity_sha256() const;
    std::string payload_identity_sha256() const;

private:
    PeriodicCorrelationThreeCenterTile() = default;
    PeriodicCorrelationFactorTileDescriptor descriptor_;
    double image_cutoff_bohr_ = 0.0;
    std::uint64_t image_candidate_count_ = 0;
    std::uint64_t retained_pair_image_count_ = 0;
    std::uint64_t reciprocal_vector_count_ = 0;
    std::uint64_t output_bytes_ = 0;
    std::uint64_t numerical_peak_bytes_ = 0;
    std::uint64_t admitted_assembly_peak_bytes_ = 0;
    std::uint64_t admitted_whitening_peak_bytes_ = 0;
    std::array<char, 64> source_identity_{};
    std::array<char, 64> whitener_identity_{};
    std::array<char, 64> ao_identity_{};
    std::array<char, 64> auxiliary_identity_{};
    std::array<char, 64> census_identity_{};
    std::array<char, 64> plan_identity_{};
    std::array<char, 64> payload_identity_{};
    std::vector<std::complex<double>> values_;

    friend PeriodicCorrelationThreeCenterTile build_periodic_correlation_three_center_tile(
        const PeriodicCorrelationAdmittedReference&,
        const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationFactorBuildCensus&,
        const PeriodicCorrelationReciprocalMetricSourceManifest&,
        const PeriodicCorrelationMetricFactorizationResult&,
        const BasisSet&, const BasisSet&, std::uint64_t, double, std::uint64_t);
};

// Recompute and require admission before any size-dependent allocation.
// The configuration must charge a full A*Pb complex compensation panel in
// backend.exact_extra_retained_bytes, plus the AO-pair fixed numerical scratch
// in backend.per_thread_fourier_transform_fixed_workspace_bytes. W remains
// resident throughout. Assembly holds full-A raw and compensation panels;
// reciprocal/Fourier/compensation storage is freed before W*T. The only
// retained output is the descriptor's Ab*Pb row-major tile. Each auxiliary
// slice recomputes T: intentionally an unoptimized numerical reference.
//
// Payload v1: length-prefixed domain "vibeqc.periodic.correlation.three-center.payload",
// u32 version; source/W-payload/AO-basis/auxiliary-basis SHA strings; image and
// reconstructed-lattice policy strings; cutoff binary64; direct lattice in
// row-major order; stored canonical ket Cartesian coordinates; descriptor in
// declaration order (wrap components i32, other fields u64); image candidate,
// retained pair-image, reciprocal-vector and output-byte counts u64; then
// row-major output complex values. Big-endian integers/binary64, signed zero
// normalized. Plan/census/block sizes are validated execution provenance and
// stored separately; candidate cap is a resource policy, not payload content.
PeriodicCorrelationThreeCenterTile build_periodic_correlation_three_center_tile(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    const PeriodicCorrelationMetricFactorizationResult& whitener,
    const BasisSet& ao_basis,
    const BasisSet& auxiliary_basis,
    std::uint64_t sequence_index,
    double image_cutoff_bohr,
    std::uint64_t maximum_image_candidates);

/// Explicit work limits for the complete physical stream. Zero means unset,
/// except receiver_retained_numeric_bytes (a genuinely allocation-free sink).
/// Receiver/progress-callback numerical storage must be included here AND in the native
/// backend's exact_extra_retained_bytes above the compensated raw panel. This
/// declaration does not inspect arbitrary callback allocations. Production
/// sinks must separately audit their own lifetime inventory.
struct PeriodicCorrelationThreeCenterStreamCaps {
    std::uint64_t maximum_tile_count = 0;
    std::uint64_t maximum_logical_bytes = 0;
    std::uint64_t maximum_tile_bytes = 0;
    std::uint64_t maximum_reciprocal_candidates_per_q = 0;
    std::uint64_t maximum_image_candidates_per_tile = 0;
    std::uint64_t receiver_retained_numeric_bytes = 0;
};

enum class PeriodicCorrelationThreeCenterStreamStage : std::uint32_t {
    Source = 0,
    Metric = 1,
    Factorization = 2,
    Tiles = 3,
    QComplete = 4,
    Complete = 5,
};

/// Numbers only: callers route live rendering through the output module.
/// Notifications precede each expensive q stage and follow each delivered
/// tile; completed counts include only callbacks that returned successfully.
struct PeriodicCorrelationThreeCenterStreamProgress {
    PeriodicCorrelationThreeCenterStreamStage stage =
        PeriodicCorrelationThreeCenterStreamStage::Source;
    std::uint64_t q_index = 0;
    std::uint64_t completed_q_count = 0;
    std::uint64_t completed_tile_count = 0;
    std::uint64_t completed_logical_bytes = 0;
    std::uint64_t total_q_count = 0;
    std::uint64_t total_tile_count = 0;
    std::uint64_t total_logical_bytes = 0;
};

/// Completion evidence, NOT proof of a durable/transactional sink commit.
/// No receipt is returned on any source/numerical/receiver/progress exception.
struct PeriodicCorrelationThreeCenterStreamReceipt {
    std::uint32_t contract_version = 1;
    bool finite_image_reference = true;
    bool ao_image_source_certified = false;
    std::uint64_t completed_q_count = 0;
    std::uint64_t completed_tile_count = 0;
    std::uint64_t completed_element_count = 0;
    std::uint64_t completed_logical_bytes = 0;
    std::uint64_t maximum_tile_bytes = 0;
    std::uint64_t admitted_peak_memory_bytes = 0;
    std::string schedule_identity_sha256;
    std::string census_identity_sha256;
    std::string plan_identity_sha256;
    std::string payload_identity_sha256;
};

/// Native full-schedule executor. Only one q source/whitener and one output
/// tile are live, with no all-q matrix or descriptor/payload collection.
/// Before its metric, each q source is checked against one constant-size
/// conjugate-source manifest, released before metric allocation. This proves
/// reciprocal membership/weight closure, not AO-image or whitener covariance.
/// Backend exact_extra_control_bytes must include sizeof that extra manifest;
/// the separate 64-byte source-identity wire is not its in-memory size.
/// Raw M is consumed/freed before W; W and its last tile are destroyed before
/// the next q metric. Borrowed tile/progress references expire at callback
/// return. Receivers must copy only into their separately admitted storage.
/// Callbacks may throw to cancel immediately; side-effecting sinks must own
/// their own rollback/RAII transaction. This routine opens no files.
///
/// Receipt payload v1 wire: length-prefixed domain
/// "vibeqc.periodic.correlation.three-center.stream", u32 version,
/// schedule/census/plan SHA strings, cutoff and negative tolerance binary64,
/// total q/tile/element/logical-byte counts u64; for each q, q index u64,
/// source and W-payload SHA strings, then each tile sequence u64 and payload
/// SHA string. Signed zero is normalized; numbers are big-endian. Work caps
/// and callback state do not enter the wire. This completion digest includes
/// execution provenance: both tiling and reciprocal block changes affect it
/// through census/plan identities, even if individual numerical tiles agree.
PeriodicCorrelationThreeCenterStreamReceipt stream_periodic_correlation_three_center_tiles(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    const BasisSet& ao_basis, const BasisSet& auxiliary_basis,
    double image_cutoff_bohr, double negative_tolerance,
    const PeriodicCorrelationThreeCenterStreamCaps& caps,
    void (*receive)(const PeriodicCorrelationThreeCenterTile&, void*), void* receiver,
    void (*progress)(const PeriodicCorrelationThreeCenterStreamProgress&, void*) = nullptr,
    void* progress_context = nullptr);

}  // namespace vibeqc
