#pragma once

/// \file periodic_correlation_pair_topology.hpp
/// \brief Translation-unique occupied-pair topology for periodic correlation.
///
/// Contract version 1 enumerates correlated occupied pairs with one orbital
/// in the reference cell.  It accepts only a statically admitted, complete
/// full-mesh RHF reference and retains that reference through immutable shared
/// ownership.  No row is screened or assigned a local domain here: every row
/// remains explicitly Unclassified for the subsequent pair-domain census.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_admitted_reference.hpp"

namespace vibeqc {

inline constexpr std::uint32_t
    kPeriodicCorrelationTranslationPairTopologyContractVersion = 1;

enum class PeriodicCorrelationPairClassification {
    Unclassified,
};

/// One translation-unique occupied-pair candidate.
///
/// The row denotes {home_orbital in cell 0, partner_orbital in cell L}.
/// translation_linear_index follows RegularKMesh's last-axis-fast order and
/// placed_multiplicity is the number of unordered pairs in the finite Born-von
/// Karman supercell represented by this row.  Classification is deliberately
/// a zero-storage contract property so the compact row stays exactly 32 bytes.
struct PeriodicCorrelationTranslationPair {
    std::uint64_t home_orbital = 0;
    std::uint64_t partner_orbital = 0;
    std::uint64_t translation_linear_index = 0;
    std::uint64_t placed_multiplicity = 0;

    constexpr PeriodicCorrelationPairClassification classification()
        const noexcept {
        return PeriodicCorrelationPairClassification::Unclassified;
    }
};

/// Exact native payload bytes per structural row. The resource planner keeps
/// a larger 64-byte allowance for the later classification/domain metadata.
inline constexpr std::uint64_t
    kPeriodicCorrelationTranslationPairRowBytes =
        sizeof(PeriodicCorrelationTranslationPair);

static_assert(kPeriodicCorrelationTranslationPairRowBytes == 32,
              "translation-pair rows must remain exactly 32 bytes");

/// Immutable reference-cell occupied-pair work list.
///
/// Rows are ordered lexicographically by (home_orbital, partner_orbital,
/// translation_linear_index), with home_orbital <= partner_orbital.  For a
/// same-orbital row, only the canonical representative L <= -L modulo the
/// mesh is retained.  The topology digest is a versioned SHA-256 over the
/// state, calculation, and allocation identities followed by the exact mesh,
/// counts, classification token, and ordered row fields.
class PeriodicCorrelationTranslationPairTopology {
public:
    PeriodicCorrelationTranslationPairTopology(
        const PeriodicCorrelationTranslationPairTopology&) = delete;
    PeriodicCorrelationTranslationPairTopology& operator=(
        const PeriodicCorrelationTranslationPairTopology&) = delete;
    PeriodicCorrelationTranslationPairTopology(
        PeriodicCorrelationTranslationPairTopology&&) noexcept = default;
    PeriodicCorrelationTranslationPairTopology& operator=(
        PeriodicCorrelationTranslationPairTopology&&) noexcept = default;
    ~PeriodicCorrelationTranslationPairTopology() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationTranslationPairTopologyContractVersion;
    }
    const PeriodicRestrictedMeanFieldState& state() const noexcept {
        return *state_;
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
    state_handle() const noexcept {
        return state_;
    }
    const std::string& calculation_identity() const noexcept {
        return calculation_identity_;
    }
    const std::string& allocation_identity() const noexcept {
        return allocation_identity_;
    }
    const std::string& state_identity_sha256() const noexcept {
        return state_identity_sha256_;
    }
    const std::string& topology_identity_sha256() const noexcept {
        return topology_identity_sha256_;
    }
    const std::array<int, 3>& mesh() const noexcept { return mesh_; }
    const std::array<int, 3>& is_shift() const noexcept { return is_shift_; }
    std::uint64_t n_cells() const noexcept { return n_cells_; }
    std::uint64_t n_home_occupied() const noexcept {
        return n_home_occupied_;
    }
    std::uint64_t self_inverse_translation_count() const noexcept {
        return self_inverse_translation_count_;
    }
    std::uint64_t row_count() const noexcept {
        return static_cast<std::uint64_t>(rows_.size());
    }
    std::uint64_t placed_pair_count() const noexcept {
        return placed_pair_count_;
    }

    const PeriodicCorrelationTranslationPair& row(std::size_t index) const;
    PeriodicCorrelationPairClassification classification(
        std::size_t index) const;

private:
    PeriodicCorrelationTranslationPairTopology(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::string calculation_identity,
        std::string allocation_identity,
        std::string state_identity_sha256,
        std::string topology_identity_sha256,
        std::array<int, 3> mesh,
        std::array<int, 3> is_shift,
        std::uint64_t n_cells,
        std::uint64_t n_home_occupied,
        std::uint64_t self_inverse_translation_count,
        std::uint64_t placed_pair_count,
        std::vector<PeriodicCorrelationTranslationPair> rows);

    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::string calculation_identity_;
    std::string allocation_identity_;
    std::string state_identity_sha256_;
    std::string topology_identity_sha256_;
    std::array<int, 3> mesh_ = {0, 0, 0};
    std::array<int, 3> is_shift_ = {0, 0, 0};
    std::uint64_t n_cells_ = 0;
    std::uint64_t n_home_occupied_ = 0;
    std::uint64_t self_inverse_translation_count_ = 0;
    std::uint64_t placed_pair_count_ = 0;
    std::vector<PeriodicCorrelationTranslationPair> rows_;

    friend PeriodicCorrelationTranslationPairTopology
    make_periodic_correlation_translation_pair_topology(
        const PeriodicCorrelationAdmittedReference& reference);
};

/// Enumerate the unscreened translation-unique occupied-pair candidates from
/// an admitted static reference. No alternative dimensions or plan can be
/// supplied to this entry point.
PeriodicCorrelationTranslationPairTopology
make_periodic_correlation_translation_pair_topology(
    const PeriodicCorrelationAdmittedReference& reference);

/// Exact routing of an ordered placed pair to the translation-unique row.
/// Translate that row's domain by common_translation_cell. If transpose is
/// true, also transpose its two virtual amplitude axes; the overlap matrix
/// itself is not transposed. Cell arithmetic is modular, not Cartesian.
/// In a nonzero self-inverse same-orbital row, reversing the endpoints may
/// instead change the common translation: do not infer transpose from an
/// unordered row key alone. Nejad doi:10.1063/5.0290816, Eqs.(42)-(44).
struct PeriodicCorrelationPlacedPairResolution {
    std::uint64_t row_index = 0;
    std::uint64_t common_translation_cell = 0;
    bool transpose = false;
};

/// Constant workspace, O(log(row_count)) lookup. Rejects noncanonical cells,
/// inactive occupied labels and consumed topology; no placed-pair table.
PeriodicCorrelationPlacedPairResolution resolve_periodic_correlation_placed_pair(
    const PeriodicCorrelationTranslationPairTopology& topology,
    std::uint64_t first_orbital, std::uint64_t first_cell,
    std::uint64_t second_orbital, std::uint64_t second_cell);

/// Multiply an UNWEIGHTED ORDERED spatial-pair energy by this exact integer
/// to obtain the representative's contribution per primitive cell. It is
/// placed_multiplicity*(2-delta_placed)/Nk, not universally 2-delta_home.
/// A same-orbital nonzero self-inverse translation has weight 1, not 2.
/// No pair screening or discarded-PNO correction is included.
std::uint64_t periodic_correlation_translation_pair_energy_weight(
    const PeriodicCorrelationTranslationPairTopology& topology,
    std::uint64_t row_index);

}  // namespace vibeqc
