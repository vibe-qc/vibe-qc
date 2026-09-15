#include "vibeqc/periodic_correlation_pair_topology.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {

static_assert(
    sizeof(std::size_t) <= sizeof(std::uint64_t),
    "translation-pair topology requires size_t to fit in uint64_t");

namespace {

constexpr std::array<int, 3> kGammaCentered = {0, 0, 0};

class CanonicalTopologyHasher {
public:
    CanonicalTopologyHasher() {
        add_string(
            "vibeqc.periodic.correlation.translation-pair-topology");
        add_u32(kPeriodicCorrelationTranslationPairTopologyContractVersion);
    }

    void add_u32(std::uint32_t value) {
        std::array<std::uint8_t, 4> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::uint8_t>(
                value >> (24U - 8U * static_cast<unsigned>(i)));
        }
        hasher_.update(bytes.data(), bytes.size());
    }

    void add_u64(std::uint64_t value) {
        std::array<std::uint8_t, 8> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::uint8_t>(
                value >> (56U - 8U * static_cast<unsigned>(i)));
        }
        hasher_.update(bytes.data(), bytes.size());
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

std::uint64_t checked_add(std::uint64_t left,
                          std::uint64_t right,
                          const char* context) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(context);
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left,
                               std::uint64_t right,
                               const char* context) {
    if (left != 0
        && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(context);
    }
    return left * right;
}

std::uint64_t checked_unordered_pair_count(std::uint64_t count) {
    // Divide the even factor first so count * (count + 1) cannot wrap before
    // the exact division by two.
    if (count == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "periodic correlation placed occupied-pair count overflows "
            "uint64");
    }
    const std::uint64_t successor = count + 1U;
    const std::uint64_t left = (count % 2U == 0U) ? count / 2U : count;
    const std::uint64_t right =
        (count % 2U == 0U) ? successor : successor / 2U;
    return checked_multiply(
        left, right,
        "periodic correlation placed occupied-pair count overflows "
        "uint64");
}

void require_admitted_reference_consistency(
    const PeriodicCorrelationAdmittedReference& reference) {
    const auto& dimensions = reference.dimensions();
    const auto& plan = reference.plan();
    const auto& state = reference.state();
    if (plan.stage != PeriodicCorrelationEstimateStage::StaticPreflight
        || plan.admission
            != PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus) {
        throw std::logic_error(
            "translation-pair topology requires a StaticPreflight plan "
            "admitted as ReadyForPairDomainCensus");
    }
    if (reference.contract_version()
            != kPeriodicCorrelationAdmittedReferenceContractVersion
        || plan.allocation_contract_version
            != dimensions.allocation_contract_version
        || !plan.census_identity.empty()
        || plan.pair_domain_census_complete
        || plan.triple_domain_census_complete) {
        throw std::logic_error(
            "translation-pair topology received an inconsistent static "
            "admission contract");
    }
    if (dimensions.symmetry_reduction_requested
        || !dimensions.symmetry_mapping_identity.empty()
        || dimensions.symmetry_representative_count
            != dimensions.n_kpoints
        || dimensions.symmetry_weight_sum != dimensions.n_kpoints) {
        throw std::invalid_argument(
            "translation-pair topology contract v1 requires an unreduced "
            "full-mesh symmetry inventory");
    }
    if (dimensions.is_shift != kGammaCentered
        || state.is_shift() != kGammaCentered) {
        throw std::invalid_argument(
            "translation-pair topology contract v1 requires the "
            "Gamma-centered k-mesh convention");
    }
    if (plan.calculation_identity != dimensions.calculation_identity
        || plan.allocation_identity != dimensions.allocation_identity
        || state.calculation_identity() != dimensions.calculation_identity
        || state.mesh() != dimensions.mesh
        || state.is_shift() != dimensions.is_shift
        || static_cast<std::uint64_t>(state.n_kpoints())
            != dimensions.n_kpoints
        || state.n_correlated_occupied() != dimensions.n_home_occupied) {
        throw std::logic_error(
            "translation-pair topology received inconsistent admitted "
            "reference seals");
    }
}

std::size_t checked_row_reserve(std::uint64_t count) {
    if (count
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "translation-pair topology row count exceeds size_t");
    }
    if (count
        > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()
            / sizeof(PeriodicCorrelationTranslationPair))) {
        throw std::overflow_error(
            "translation-pair topology byte count exceeds size_t");
    }
    return static_cast<std::size_t>(count);
}

std::string make_topology_identity_sha256(
    const PeriodicCorrelationAdmittedReference& reference,
    std::uint64_t n_cells,
    std::uint64_t n_home_occupied,
    std::uint64_t self_inverse_translation_count,
    std::uint64_t placed_pair_count,
    const std::vector<PeriodicCorrelationTranslationPair>& rows) {
    CanonicalTopologyHasher digest;
    digest.add_u32(reference.contract_version());
    digest.add_u32(reference.dimensions().allocation_contract_version);
    digest.add_u32(reference.state().digest_version());
    digest.add_string(reference.state().state_identity_sha256());
    digest.add_string(reference.dimensions().calculation_identity);
    digest.add_string(reference.dimensions().allocation_identity);
    for (int component : reference.dimensions().mesh) {
        digest.add_u32(static_cast<std::uint32_t>(component));
    }
    for (int component : reference.dimensions().is_shift) {
        digest.add_u32(static_cast<std::uint32_t>(component));
    }
    digest.add_u64(n_cells);
    digest.add_u64(n_home_occupied);
    digest.add_u64(self_inverse_translation_count);
    digest.add_u64(static_cast<std::uint64_t>(rows.size()));
    digest.add_u64(placed_pair_count);
    digest.add_string("unclassified");
    for (const auto& row : rows) {
        digest.add_u64(row.home_orbital);
        digest.add_u64(row.partner_orbital);
        digest.add_u64(row.translation_linear_index);
        digest.add_u64(row.placed_multiplicity);
    }
    return digest.finish_hex();
}

}  // namespace

PeriodicCorrelationTranslationPairTopology::
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
        std::vector<PeriodicCorrelationTranslationPair> rows)
    : state_(std::move(state)),
      calculation_identity_(std::move(calculation_identity)),
      allocation_identity_(std::move(allocation_identity)),
      state_identity_sha256_(std::move(state_identity_sha256)),
      topology_identity_sha256_(std::move(topology_identity_sha256)),
      mesh_(mesh),
      is_shift_(is_shift),
      n_cells_(n_cells),
      n_home_occupied_(n_home_occupied),
      self_inverse_translation_count_(self_inverse_translation_count),
      placed_pair_count_(placed_pair_count),
      rows_(std::move(rows)) {}

const PeriodicCorrelationTranslationPair&
PeriodicCorrelationTranslationPairTopology::row(std::size_t index) const {
    if (index >= rows_.size()) {
        throw std::out_of_range(
            "translation-pair topology row index is out of range");
    }
    return rows_[index];
}

PeriodicCorrelationPairClassification
PeriodicCorrelationTranslationPairTopology::classification(
    std::size_t index) const {
    return row(index).classification();
}

PeriodicCorrelationPlacedPairResolution resolve_periodic_correlation_placed_pair(
    const PeriodicCorrelationTranslationPairTopology& topology,
    std::uint64_t first_orbital, std::uint64_t first_cell,
    std::uint64_t second_orbital, std::uint64_t second_cell) {
    if (!topology.state_handle() || !topology.row_count())
        throw std::logic_error("placed-pair resolution requires a live topology");
    if (first_orbital >= topology.n_home_occupied() || second_orbital >= topology.n_home_occupied()
        || first_cell >= topology.n_cells() || second_cell >= topology.n_cells())
        throw std::out_of_range("placed-pair occupied index or canonical cell is out of range");
    const auto difference = [&](std::uint64_t left, std::uint64_t right) {
        std::array<std::uint64_t, 3> delta{};
        for (int axis = 2; axis >= 0; --axis) {
            const auto n = static_cast<std::uint64_t>(topology.mesh()[axis]);
            delta[axis] = (left % n + n - right % n) % n;
            left /= n; right /= n;
        }
        return (delta[0] * topology.mesh()[1] + delta[1]) * topology.mesh()[2] + delta[2];
    };
    auto translation = difference(second_cell, first_cell);
    const auto inverse = difference(first_cell, second_cell);
    PeriodicCorrelationPlacedPairResolution result;
    result.transpose = first_orbital > second_orbital
        || (first_orbital == second_orbital && translation > inverse);
    result.common_translation_cell = result.transpose ? second_cell : first_cell;
    if (result.transpose) {
        std::swap(first_orbital, second_orbital);
        translation = inverse;
    }
    const auto key = std::make_tuple(first_orbital, second_orbital, translation);
    std::uint64_t lo = 0, hi = topology.row_count();
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        const auto& row = topology.row(static_cast<std::size_t>(mid));
        const auto candidate = std::make_tuple(row.home_orbital, row.partner_orbital,
                                               row.translation_linear_index);
        if (candidate < key) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= topology.row_count())
        throw std::logic_error("placed-pair canonical row is missing");
    const auto& row = topology.row(static_cast<std::size_t>(lo));
    if (std::make_tuple(row.home_orbital, row.partner_orbital, row.translation_linear_index) != key)
        throw std::logic_error("placed-pair canonical row key is inconsistent");
    result.row_index = lo;
    return result;
}

std::uint64_t periodic_correlation_translation_pair_energy_weight(
    const PeriodicCorrelationTranslationPairTopology& topology, std::uint64_t index) {
    if (!topology.state_handle() || !topology.n_cells())
        throw std::logic_error("pair energy weight requires a live topology");
    if (index >= topology.row_count())
        throw std::out_of_range("pair energy weight row index is out of range");
    const auto& row = topology.row(static_cast<std::size_t>(index));
    const bool diagonal = row.home_orbital == row.partner_orbital && row.translation_linear_index == 0;
    const auto ordered = checked_multiply(row.placed_multiplicity, diagonal ? 1 : 2,
        "pair energy placed multiplicity overflows uint64");
    if (ordered % topology.n_cells() || ordered / topology.n_cells() > 2
        || ordered / topology.n_cells() == 0)
        throw std::logic_error("pair energy primitive-cell multiplicity is inconsistent");
    return ordered / topology.n_cells();
}

PeriodicCorrelationTranslationPairTopology
make_periodic_correlation_translation_pair_topology(
    const PeriodicCorrelationAdmittedReference& reference) {
    require_admitted_reference_consistency(reference);
    const auto& dimensions = reference.dimensions();

    const PeriodicCorrelationTranslationPairCounts expected =
        estimate_periodic_correlation_translation_pair_counts(
            dimensions.mesh, dimensions.n_home_occupied);
    if (dimensions.expected_pair_candidate_count
        != expected.candidate_count) {
        throw std::logic_error(
            "translation-pair topology candidate count disagrees with "
            "the admitted static inventory");
    }
    if (expected.cell_count != dimensions.n_kpoints) {
        throw std::logic_error(
            "translation-pair topology cell count disagrees with the "
            "admitted static inventory");
    }

    const std::size_t reserve_count =
        checked_row_reserve(expected.candidate_count);
    std::vector<PeriodicCorrelationTranslationPair> rows;
    if (reserve_count > rows.max_size()) {
        throw std::overflow_error(
            "translation-pair topology row count exceeds vector max_size");
    }
    rows.reserve(reserve_count);

    const std::uint64_t n_cells = expected.cell_count;
    const std::uint64_t n_home_occupied = dimensions.n_home_occupied;
    const RegularKMesh addressing(dimensions.mesh, dimensions.is_shift);
    std::uint64_t total_placed_multiplicity = 0;

    for (std::uint64_t i = 0; i < n_home_occupied; ++i) {
        for (std::uint64_t j = i; j < n_home_occupied; ++j) {
            for (std::uint64_t translation = 0;
                 translation < n_cells;
                 ++translation) {
                const auto translation_index =
                    static_cast<std::size_t>(translation);
                bool nonzero_self_inverse = false;
                if (i == j) {
                    const auto inverse_index = addressing.transfer_index(
                        addressing.negate(
                            addressing.transfer_address(translation_index)));
                    if (translation_index > inverse_index) {
                        continue;
                    }
                    nonzero_self_inverse =
                        translation != 0U
                        && translation_index == inverse_index;
                    if (nonzero_self_inverse && n_cells % 2U != 0U) {
                        throw std::logic_error(
                            "a nonzero self-inverse translation requires "
                            "an even cell count");
                    }
                }
                const std::uint64_t multiplicity =
                    nonzero_self_inverse ? n_cells / 2U : n_cells;
                if (rows.size() >= reserve_count) {
                    throw std::logic_error(
                        "translation-pair topology enumeration exceeded "
                        "the exact reserved row count");
                }
                total_placed_multiplicity = checked_add(
                    total_placed_multiplicity,
                    multiplicity,
                    "translation-pair topology multiplicity sum overflows "
                    "uint64");
                rows.push_back({i, j, translation, multiplicity});
            }
        }
    }

    if (rows.size() != reserve_count) {
        throw std::logic_error(
            "translation-pair topology enumeration count disagrees with "
            "the exact estimator");
    }
    if (total_placed_multiplicity
        != expected.placed_pair_count) {
        throw std::logic_error(
            "translation-pair topology multiplicity sum disagrees with "
            "the exact estimator");
    }
    const std::uint64_t placed_occupied = checked_multiply(
        n_cells,
        n_home_occupied,
        "translation-pair topology placed occupied count overflows uint64");
    if (total_placed_multiplicity
        != checked_unordered_pair_count(placed_occupied)) {
        throw std::logic_error(
            "translation-pair topology does not cover every unordered "
            "placed occupied pair exactly once");
    }

    const std::string topology_identity_sha256 =
        make_topology_identity_sha256(
            reference,
            n_cells,
            n_home_occupied,
            expected.self_inverse_translation_count,
            total_placed_multiplicity,
            rows);
    return PeriodicCorrelationTranslationPairTopology(
        reference.state_handle(),
        dimensions.calculation_identity,
        dimensions.allocation_identity,
        reference.state().state_identity_sha256(),
        topology_identity_sha256,
        dimensions.mesh,
        dimensions.is_shift,
        n_cells,
        n_home_occupied,
        expected.self_inverse_translation_count,
        total_placed_multiplicity,
        std::move(rows));
}

static_assert(
    !std::is_copy_constructible<
        PeriodicCorrelationTranslationPairTopology>::value,
    "a translation-pair topology must not copy its state or row storage");
static_assert(
    std::is_nothrow_move_constructible<
        PeriodicCorrelationTranslationPairTopology>::value,
    "a translation-pair topology must be cheaply movable");

}  // namespace vibeqc
