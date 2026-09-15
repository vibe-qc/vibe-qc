#pragma once

// Private numerical scratch, not a restart or user-facing output format.
// One canonical sequential writer, one immutable single-threaded reader,
// no descriptor table, bitmap, mmap, complete factor tensor or visible path.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_three_center.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationPrivateFactorStoreVersion = 1;
inline constexpr std::uint64_t kPeriodicCorrelationPrivateFactorStoreHeaderBytes = 1024;
inline constexpr std::uint64_t kPeriodicCorrelationPrivateFactorStoreRecordBytes = 256;
inline constexpr std::uint64_t kPeriodicCorrelationPrivateFactorStoreFooterBytes = 256;
inline constexpr std::uint64_t kPeriodicCorrelationPrivateFactorStoreCodecPageBytes = 65536;

struct PeriodicCorrelationPrivateFactorStoreCaps {
    std::uint64_t maximum_file_bytes = 0;
    std::uint64_t maximum_tile_bytes = 0;
    std::uint64_t maximum_tile_count = 0;
};

enum class PeriodicCorrelationPrivateFactorStoreState : std::uint32_t {
    Open = 0, Failed = 1, Aborted = 2, Finalized = 3,
};

std::string periodic_correlation_private_factor_store_codec_identity_sha256();
PeriodicCorrelationFactorBuildCodecInventory
periodic_correlation_private_factor_store_codec_inventory();
// Conservative fixed native control allowance, independent of Nq/Ntiles.
std::uint64_t periodic_correlation_private_factor_store_fixed_control_bytes() noexcept;
// Page plus one maximum reader tile. The page is resident during production;
// the extra reader tile is conservatively retained in every phase as well.
std::uint64_t periodic_correlation_private_factor_store_receiver_bytes(
    const PeriodicCorrelationFactorStreamSchedule& schedule);
std::uint64_t periodic_correlation_private_factor_store_file_bytes(
    const PeriodicCorrelationFactorStreamSchedule& schedule);
std::uint64_t periodic_correlation_private_factor_store_record_offset(
    const PeriodicCorrelationFactorStreamSchedule& schedule, std::uint64_t sequence);

struct PeriodicCorrelationPrivateFactorTileView {
    PeriodicCorrelationFactorTileDescriptor descriptor;
    const std::complex<double>* data = nullptr;
    std::uint64_t element_count = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t image_candidate_count = 0;
    std::uint64_t retained_pair_image_count = 0;
    std::uint64_t reciprocal_vector_count = 0;
    std::string source_identity_sha256;
    std::string whitener_payload_identity_sha256;
    std::string payload_identity_sha256;
    // Borrowed payload and this view expire when the callback returns.
};

namespace detail {
struct PrivateFactorStoreImpl;
}
class PeriodicCorrelationPrivateFactorWriter;

class PeriodicCorrelationPrivateFactorReader {
public:
    PeriodicCorrelationPrivateFactorReader(const PeriodicCorrelationPrivateFactorReader&) = delete;
    PeriodicCorrelationPrivateFactorReader& operator=(const PeriodicCorrelationPrivateFactorReader&) = delete;
    PeriodicCorrelationPrivateFactorReader(PeriodicCorrelationPrivateFactorReader&&) noexcept;
    PeriodicCorrelationPrivateFactorReader& operator=(PeriodicCorrelationPrivateFactorReader&&) noexcept;
    ~PeriodicCorrelationPrivateFactorReader();

    std::uint64_t file_bytes() const;
    std::uint64_t tile_count() const;
    std::string storage_identity_sha256() const;
    std::string producer_completion_identity_sha256() const;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const;
    std::string state_identity_sha256() const;
    std::string calculation_identity() const;
    std::string allocation_identity() const;
    std::string schedule_identity_sha256() const;
    std::string census_identity_sha256() const;
    std::string plan_identity_sha256() const;
    std::string ao_basis_identity_sha256() const;
    std::string auxiliary_basis_identity_sha256() const;
    double image_cutoff_bohr() const;
    bool finite_image_reference() const noexcept { return true; }
    bool ao_image_source_certified() const noexcept { return false; }
    const char* image_policy() const noexcept { return kPeriodicCorrelationThreeCenterImagePolicy; }
    PeriodicCorrelationFactorTileDescriptor descriptor(std::uint64_t sequence) const;
    // Verifies exact file extent, fixed metadata, record and full numerical
    // tile SHA before invoking receiver. No recursive or concurrent visits.
    // On corruption the reader is permanently poisoned. Callback exceptions
    // propagate but do not imply corruption. No tile survives callback return.
    void visit_tile(std::uint64_t sequence, std::uint64_t maximum_tile_bytes,
                    void (*receiver)(const PeriodicCorrelationPrivateFactorTileView&, void*),
                    void* context) const;

private:
    explicit PeriodicCorrelationPrivateFactorReader(std::unique_ptr<detail::PrivateFactorStoreImpl>) noexcept;
    std::unique_ptr<detail::PrivateFactorStoreImpl> impl_;
    friend class PeriodicCorrelationPrivateFactorWriter;
    friend std::vector<std::uint8_t> private_factor_store_bytes_diagnostic(
        const PeriodicCorrelationPrivateFactorReader&, std::uint64_t, std::uint64_t);
};

class PeriodicCorrelationPrivateFactorWriter {
public:
    PeriodicCorrelationPrivateFactorWriter(const PeriodicCorrelationPrivateFactorWriter&) = delete;
    PeriodicCorrelationPrivateFactorWriter& operator=(const PeriodicCorrelationPrivateFactorWriter&) = delete;
    PeriodicCorrelationPrivateFactorWriter(PeriodicCorrelationPrivateFactorWriter&&) noexcept;
    PeriodicCorrelationPrivateFactorWriter& operator=(PeriodicCorrelationPrivateFactorWriter&&) noexcept;
    ~PeriodicCorrelationPrivateFactorWriter();

    PeriodicCorrelationPrivateFactorStoreState state() const noexcept;
    std::uint64_t accepted_tile_count() const noexcept;
    std::uint64_t file_bytes() const;
    void accept(const PeriodicCorrelationThreeCenterTile& tile);
    PeriodicCorrelationPrivateFactorReader finish(
        const PeriodicCorrelationThreeCenterStreamReceipt& completion);
    void abort() noexcept;

private:
    explicit PeriodicCorrelationPrivateFactorWriter(std::unique_ptr<detail::PrivateFactorStoreImpl>) noexcept;
    std::unique_ptr<detail::PrivateFactorStoreImpl> impl_;
    bool finalized_ = false;
    friend PeriodicCorrelationPrivateFactorWriter make_periodic_correlation_private_factor_writer(
        const PeriodicCorrelationAdmittedReference&,
        const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationFactorBuildCensus&,
        double, double, const std::string&,
        const PeriodicCorrelationPrivateFactorStoreCaps&);
    friend void private_factor_store_corrupt_diagnostic(
        PeriodicCorrelationPrivateFactorWriter&, std::uint64_t, std::uint8_t);
    friend void private_factor_store_truncate_diagnostic(
        PeriodicCorrelationPrivateFactorWriter&, std::uint64_t);
};

// Only Linux/macOS with actual storage reservation are supported. Directory
// must be existing, caller-owned and private (no group/other permission).
// An exclusive 0600 regular file is opened with close-on-exec and immediately
// unlinked after obtaining matching read/write descriptors. Nothing existing
// is overwritten or removed; abort/reader destruction reclaim owned scratch.
// No filesystem durability across process loss or restart is advertised.
PeriodicCorrelationPrivateFactorWriter make_periodic_correlation_private_factor_writer(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    double image_cutoff_bohr, double negative_tolerance,
    const std::string& private_scratch_directory,
    const PeriodicCorrelationPrivateFactorStoreCaps& caps);

// Tiny diagnostic seams; never expose native descriptors or paths. Changes
// affect only this writer's unlinked private inode, before finalization.
void private_factor_store_corrupt_diagnostic(
    PeriodicCorrelationPrivateFactorWriter&, std::uint64_t offset, std::uint8_t xor_mask);
void private_factor_store_truncate_diagnostic(
    PeriodicCorrelationPrivateFactorWriter&, std::uint64_t size);
std::uint64_t private_factor_store_live_owned_descriptors_diagnostic() noexcept;
std::vector<std::uint8_t> private_factor_store_bytes_diagnostic(
    const PeriodicCorrelationPrivateFactorReader&, std::uint64_t offset, std::uint64_t count);

}  // namespace vibeqc
