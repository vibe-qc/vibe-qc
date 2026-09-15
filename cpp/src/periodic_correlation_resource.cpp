#include "vibeqc/periodic_correlation_resource.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vibeqc/kmesh_address.hpp"

namespace vibeqc {

namespace {

constexpr PeriodicCorrelationByteCount kComplexBytes = 16;
constexpr PeriodicCorrelationByteCount kMiB = 1024 * 1024;
constexpr PeriodicCorrelationByteCount kSinglesMetadataBytes = 8;
constexpr PeriodicCorrelationByteCount kPairCandidateMetadataBytes = 64;
constexpr PeriodicCorrelationByteCount kFactorDomainMetadataBytes = 128;
constexpr PeriodicCorrelationByteCount kTripleCandidateMetadataBytes = 64;
constexpr PeriodicCorrelationByteCount kCheckpointProvenanceMinimumBytes = 256;

bool is_lower_sha256(const std::string& value);

class ResourceArithmeticOverflow : public std::overflow_error {
public:
    explicit ResourceArithmeticOverflow(const std::string& label)
        : std::overflow_error("resource arithmetic overflow: " + label) {}
};

PeriodicCorrelationByteCount checked_add(
    PeriodicCorrelationByteCount a,
    PeriodicCorrelationByteCount b,
    const char* label) {
    constexpr auto limit =
        std::numeric_limits<PeriodicCorrelationByteCount>::max();
    if (b > limit - a) {
        throw ResourceArithmeticOverflow(label);
    }
    return a + b;
}

PeriodicCorrelationByteCount checked_multiply(
    PeriodicCorrelationByteCount a,
    PeriodicCorrelationByteCount b,
    const char* label) {
    constexpr auto limit =
        std::numeric_limits<PeriodicCorrelationByteCount>::max();
    if (a != 0 && b > limit / a) {
        throw ResourceArithmeticOverflow(label);
    }
    return a * b;
}

PeriodicCorrelationByteCount checked_square(
    PeriodicCorrelationByteCount value,
    const char* label) {
    return checked_multiply(value, value, label);
}

PeriodicCorrelationByteCount checked_cube(
    PeriodicCorrelationByteCount value,
    const char* label) {
    return checked_multiply(
        checked_square(value, label), value, label);
}

PeriodicCorrelationByteCount checked_ceil_divide(
    PeriodicCorrelationByteCount numerator,
    PeriodicCorrelationByteCount denominator,
    const char* label) {
    if (denominator == 0) {
        throw std::invalid_argument(std::string(label) + " denominator is zero");
    }
    return numerator / denominator
        + static_cast<PeriodicCorrelationByteCount>(
            numerator % denominator != 0);
}

/// Exact ceil(value * numerator / denominator) for the small rational
/// headroom factors used by this planner. Dividing first avoids the usual
/// value * numerator intermediate overflow.
PeriodicCorrelationByteCount checked_scale_ceil(
    PeriodicCorrelationByteCount value,
    PeriodicCorrelationByteCount numerator,
    PeriodicCorrelationByteCount denominator,
    const char* label) {
    if (denominator == 0) {
        throw std::invalid_argument(std::string(label) + " denominator is zero");
    }
    const auto quotient = value / denominator;
    const auto remainder = value % denominator;
    const auto whole = checked_multiply(quotient, numerator, label);
    const auto fractional_numerator =
        checked_multiply(remainder, numerator, label);
    return checked_add(
        whole,
        checked_ceil_divide(fractional_numerator, denominator, label),
        label);
}

PeriodicCorrelationByteCount required_memory(
    PeriodicCorrelationByteCount modeled) {
    return checked_scale_ceil(modeled, 3, 2, "memory headroom");
}

PeriodicCorrelationByteCount required_scratch(
    PeriodicCorrelationByteCount modeled) {
    return checked_scale_ceil(modeled, 5, 4, "scratch headroom");
}

void require_positive(
    PeriodicCorrelationByteCount value,
    const char* name) {
    if (value == 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

PeriodicCorrelationByteCount minimum_pair_metadata_bytes(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    return checked_add(
        checked_multiply(
            dimensions.n_home_occupied,
            kSinglesMetadataBytes,
            "minimum singles metadata"),
        checked_multiply(
            dimensions.expected_pair_candidate_count,
            kPairCandidateMetadataBytes,
            "minimum pair-candidate metadata"),
        "minimum pair-domain metadata");
}

PeriodicCorrelationByteCount minimum_pair_census_metadata_bytes(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationDomainCensus& census) {
    return checked_add(
        minimum_pair_metadata_bytes(dimensions),
        checked_multiply(
            census.factor_domains.size(),
            kFactorDomainMetadataBytes,
            "minimum exact factor-domain metadata"),
        "minimum exact pair-domain metadata");
}

PeriodicCorrelationByteCount minimum_triple_metadata_bytes(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    return checked_multiply(
        dimensions.expected_triple_candidate_count,
        kTripleCandidateMetadataBytes,
        "minimum triple-candidate metadata");
}

void validate_static_dimensions(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationResourceBudget& budget) {
    if (dimensions.allocation_contract_version
        != kPeriodicCorrelationStreamedResourceContractVersion) {
        throw std::invalid_argument(
            "unsupported periodic-correlation allocation contract version");
    }
    if (!is_lower_sha256(dimensions.calculation_identity)) {
        throw std::invalid_argument(
            "calculation_identity must be a lowercase SHA-256 digest");
    }
    if (!is_lower_sha256(dimensions.allocation_identity)) {
        throw std::invalid_argument(
            "allocation_identity must be a lowercase SHA-256 digest");
    }
    if (!dimensions.static_inventory_complete) {
        throw std::invalid_argument(
            "static periodic-correlation inventory is incomplete");
    }
    if (dimensions.periodic_dimension < 1
        || dimensions.periodic_dimension > 3) {
        throw std::invalid_argument(
            "periodic-correlation periodic_dimension must be 1, 2, or 3");
    }
    for (int axis = dimensions.periodic_dimension; axis < 3; ++axis) {
        if (dimensions.mesh[static_cast<std::size_t>(axis)] != 1
            || dimensions.is_shift[static_cast<std::size_t>(axis)] != 0) {
            throw std::invalid_argument(
                "periodic-correlation inactive mesh axes must have extent "
                "one and zero shift");
        }
    }
    require_positive(dimensions.n_kpoints, "n_kpoints");
    std::size_t exact_kpoint_count = 0;
    try {
        exact_kpoint_count =
            RegularKMesh(dimensions.mesh, dimensions.is_shift).size();
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument(
            std::string("invalid periodic-correlation RegularKMesh: ")
            + error.what());
    }
    if (static_cast<std::uint64_t>(exact_kpoint_count)
        != dimensions.n_kpoints) {
        throw std::invalid_argument(
            "n_kpoints must equal the exact RegularKMesh extent");
    }
    require_positive(dimensions.n_basis, "n_basis");
    require_positive(
        dimensions.n_effective_orbitals, "n_effective_orbitals");
    require_positive(dimensions.n_auxiliary, "n_auxiliary");
    require_positive(
        dimensions.n_home_total_occupied, "n_home_total_occupied");
    require_positive(dimensions.n_home_occupied, "n_home_occupied");
    require_positive(dimensions.n_home_virtual, "n_home_virtual");
    if (dimensions.n_home_occupied > dimensions.n_home_total_occupied) {
        throw std::invalid_argument(
            "correlated occupied count cannot exceed total occupied count");
    }
    if (dimensions.n_effective_orbitals > dimensions.n_basis
        || dimensions.n_home_total_occupied > dimensions.n_effective_orbitals
        || dimensions.n_home_virtual > dimensions.n_effective_orbitals
        || checked_add(
               dimensions.n_home_total_occupied,
               dimensions.n_home_virtual,
               "occupied and virtual orbital count")
            != dimensions.n_effective_orbitals) {
        throw std::invalid_argument(
            "total occupied plus full virtual must equal the effective MO rank");
    }
    require_positive(
        dimensions.symmetry_representative_count,
        "symmetry_representative_count");
    if (dimensions.symmetry_representative_count > dimensions.n_kpoints
        || dimensions.symmetry_weight_sum != dimensions.n_kpoints) {
        throw std::invalid_argument(
            "static symmetry inventory does not cover the full k-point mesh");
    }
    if (dimensions.symmetry_reduction_requested) {
        if (!is_lower_sha256(dimensions.symmetry_mapping_identity)) {
            throw std::invalid_argument(
                "symmetry reduction requires a mapping digest");
        }
    } else if (!dimensions.symmetry_mapping_identity.empty()
               || dimensions.symmetry_representative_count
                   != dimensions.n_kpoints) {
        throw std::invalid_argument(
            "unreduced preflight must enumerate the full k-point mesh");
    }
    const auto pair_counts =
        estimate_periodic_correlation_translation_pair_counts(
            dimensions.mesh, dimensions.n_home_occupied);
    if (dimensions.expected_pair_candidate_count
        != pair_counts.candidate_count) {
        throw std::invalid_argument(
            "expected_pair_candidate_count must equal the exact "
            "translation-pair topology count");
    }
    require_positive(
        dimensions.domain_ao_support_upper_bound,
        "domain_ao_support_upper_bound");
    require_positive(
        dimensions.domain_pao_upper_bound,
        "domain_pao_upper_bound");
    require_positive(
        dimensions.domain_pno_upper_bound,
        "domain_pno_upper_bound");
    require_positive(
        dimensions.domain_local_occupied_upper_bound,
        "domain_local_occupied_upper_bound");
    require_positive(
        dimensions.domain_local_auxiliary_upper_bound,
        "domain_local_auxiliary_upper_bound");
    require_positive(
        dimensions.pair_domain_metadata_upper_bytes,
        "pair_domain_metadata_upper_bytes");
    if (dimensions.pair_domain_metadata_upper_bytes
        < minimum_pair_metadata_bytes(dimensions)) {
        throw std::invalid_argument(
            "pair-domain metadata bound is below its structural minimum");
    }
    if (dimensions.domain_pno_upper_bound
            > dimensions.domain_pao_upper_bound
        || dimensions.domain_pao_upper_bound
            > dimensions.domain_ao_support_upper_bound) {
        throw std::invalid_argument(
            "domain PNO/PAO bounds exceed their admitted AO support");
    }
    if (dimensions.n_spin_channels != 1) {
        throw std::invalid_argument(
            "streamed resource contract version 1 is closed-shell only");
    }
    require_positive(
        dimensions.factor_k_bra_block, "factor_k_bra_block");
    require_positive(
        dimensions.factor_k_ket_block, "factor_k_ket_block");
    require_positive(
        dimensions.factor_auxiliary_block, "factor_auxiliary_block");
    require_positive(dimensions.factor_q_block, "factor_q_block");
    if (dimensions.factor_auxiliary_block > dimensions.n_auxiliary) {
        throw std::invalid_argument(
            "factor auxiliary block cannot exceed n_auxiliary");
    }
    if (dimensions.factor_k_bra_block != 1
        || dimensions.factor_k_ket_block != 1
        || dimensions.factor_q_block != 1) {
        throw std::invalid_argument(
            "streamed resource contract version 1 is one-q-at-a-time");
    }
    require_positive(budget.mpi_ranks, "mpi_ranks");
    require_positive(budget.workers_per_rank, "workers_per_rank");
    if (budget.mpi_ranks != 1) {
        throw std::invalid_argument(
            "streamed resource contract version 1 supports one MPI rank");
    }
    if (dimensions.triples_requested) {
        require_positive(
            dimensions.expected_triple_candidate_count,
            "expected_triple_candidate_count");
        require_positive(
            dimensions.triple_virtual_support_upper_bound,
            "triple_virtual_support_upper_bound");
        require_positive(
            dimensions.triple_union_pno_upper_bound,
            "triple_union_pno_upper_bound");
        require_positive(
            dimensions.triple_tno_upper_bound,
            "triple_tno_upper_bound");
        require_positive(
            dimensions.triple_local_occupied_upper_bound,
            "triple_local_occupied_upper_bound");
        require_positive(
            dimensions.triple_local_auxiliary_upper_bound,
            "triple_local_auxiliary_upper_bound");
        require_positive(
            dimensions.triple_domain_metadata_upper_bytes,
            "triple_domain_metadata_upper_bytes");
        if (dimensions.triple_domain_metadata_upper_bytes
            < minimum_triple_metadata_bytes(dimensions)) {
            throw std::invalid_argument(
                "triple-domain metadata bound is below its structural minimum");
        }
        if (dimensions.triple_tno_upper_bound
                > dimensions.triple_union_pno_upper_bound
            || dimensions.triple_union_pno_upper_bound
                > dimensions.triple_virtual_support_upper_bound) {
            throw std::invalid_argument(
                "triple TNO/union bounds exceed their virtual support");
        }
    } else if (dimensions.expected_triple_candidate_count != 0
               || dimensions.triple_virtual_support_upper_bound != 0
               || dimensions.triple_union_pno_upper_bound != 0
               || dimensions.triple_tno_upper_bound != 0
               || dimensions.triple_local_occupied_upper_bound != 0
               || dimensions.triple_local_auxiliary_upper_bound != 0
               || dimensions.triple_domain_metadata_upper_bytes != 0) {
        throw std::invalid_argument(
            "triple-domain bounds require triples_requested");
    }
}

PeriodicCorrelationByteCount total_worker_limit(
    const PeriodicCorrelationResourceBudget& budget) {
    return checked_multiply(
        budget.mpi_ranks,
        budget.workers_per_rank,
        "MPI ranks times workers per rank");
}

PeriodicCorrelationByteCount mean_field_payload_per_rank(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto n_basis_squared =
        checked_square(dimensions.n_basis, "mean-field basis square");
    const auto spin_buffers = checked_multiply(
        2, dimensions.n_spin_channels, "mean-field spin buffers");
    const auto matrix_count =
        checked_add(4, spin_buffers, "mean-field matrix count");
    auto elements = checked_multiply(
        dimensions.n_kpoints,
        n_basis_squared,
        "mean-field k-point matrices");
    elements = checked_multiply(
        elements, matrix_count, "mean-field resident matrices");
    return checked_multiply(
        kComplexBytes, elements, "mean-field resident bytes");
}

PeriodicCorrelationByteCount localization_payload_per_rank(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto coefficients = checked_multiply(
        dimensions.n_basis,
        dimensions.n_home_occupied,
        "localization coefficient matrix");
    const auto occupied_square = checked_square(
        dimensions.n_home_occupied,
        "localization occupied square");
    auto per_k = checked_add(
        coefficients, occupied_square, "localization per-k elements");
    per_k = checked_multiply(
        dimensions.n_kpoints, per_k, "localization k-point elements");
    auto bytes = checked_multiply(
        kComplexBytes, per_k, "localization matrix bytes");
    return checked_add(
        bytes,
        dimensions.localization_window_bytes_per_rank,
        "localization window bytes");
}

PeriodicCorrelationByteCount factor_metric_payload_per_rank(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    auto metric_elements = checked_multiply(
        2,
        checked_square(
            dimensions.n_auxiliary, "factor-build auxiliary metric square"),
        "factor-build raw and factorized metrics");
    metric_elements = checked_multiply(
        dimensions.factor_q_block,
        metric_elements,
        "factor-build q-sector metrics");
    return checked_multiply(
        kComplexBytes, metric_elements, "factor-build metric bytes");
}

PeriodicCorrelationByteCount domain_build_workspace_bytes(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto pao = dimensions.domain_pao_upper_bound;
    const auto pno = dimensions.domain_pno_upper_bound;
    const auto ao_support = dimensions.domain_ao_support_upper_bound;
    const auto local_auxiliary =
        dimensions.domain_local_auxiliary_upper_bound;
    const auto full_virtual = checked_multiply(
        dimensions.n_kpoints,
        dimensions.n_home_virtual,
        "domain-build full periodic virtual support");
    const auto basis_pao = checked_multiply(
        ao_support, pao, "domain-build AO-PAO coefficients");
    const auto pao_pno = checked_multiply(
        pao, pno, "domain-build PAO-PNO coefficients");
    const auto auxiliary_pao = checked_multiply(
        local_auxiliary,
        pao,
        "domain-build auxiliary-PAO factors");
    const auto pao_square = checked_square(
        pao, "domain-build PAO square");
    const auto pno_square = checked_square(
        pno, "domain-build PNO square");
    const auto full_virtual_square = checked_square(
        full_virtual, "domain-build full virtual square");
    auto elements = checked_add(
        basis_pao, pao_pno, "domain-build coefficient elements");
    elements = checked_add(
        elements,
        checked_multiply(
            2, auxiliary_pao, "domain-build paired auxiliary factors"),
        "domain-build factor elements");
    elements = checked_add(
        elements,
        checked_multiply(
            4, pao_square, "domain-build PAO eigensolver matrices"),
        "domain-build PAO elements");
    elements = checked_add(
        elements,
        checked_multiply(
            3, pno_square, "domain-build PNO eigensolver matrices"),
        "domain-build total elements");
    elements = checked_add(
        elements,
        checked_multiply(
            3,
            full_virtual_square,
            "domain-build full-virtual K/T/denominator matrices"),
        "domain-build pair-screening elements");
    elements = checked_add(
        elements,
        checked_multiply(
            2,
            checked_multiply(
                dimensions.factor_auxiliary_block,
                full_virtual,
                "domain-build streamed screening factors"),
            "domain-build paired screening-factor panels"),
        "domain-build streamed screening elements");
    return checked_multiply(
        kComplexBytes, elements, "domain-build workspace bytes");
}

PeriodicCorrelationByteCount triple_domain_build_workspace_bytes(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto support = dimensions.triple_virtual_support_upper_bound;
    const auto union_pno = dimensions.triple_union_pno_upper_bound;
    const auto tno = dimensions.triple_tno_upper_bound;
    const auto occupied = dimensions.triple_local_occupied_upper_bound;
    const auto auxiliary_panel = std::min(
        dimensions.factor_auxiliary_block,
        dimensions.triple_local_auxiliary_upper_bound);
    const auto union_square = checked_square(
        union_pno, "triple-domain union-PNO square");
    const auto tno_square = checked_square(
        tno, "triple-domain TNO square");
    const auto occupied_square = checked_square(
        occupied, "triple-domain occupied square");

    auto elements = checked_multiply(
        support, union_pno, "triple-domain concatenated union matrix");
    elements = checked_add(
        elements,
        checked_multiply(
            8, union_square, "triple-domain SVD/eigensolver workspace"),
        "triple-domain union construction");
    elements = checked_add(
        elements,
        checked_multiply(
            support, tno, "triple-domain retained TNO coefficients"),
        "triple-domain retained transformations");
    elements = checked_add(
        elements,
        checked_multiply(
            4, tno_square, "triple-domain density/Fock workspace"),
        "triple-domain TNO construction");
    elements = checked_add(
        elements,
        checked_multiply(
            occupied, tno, "triple-domain projected singles"),
        "triple-domain projected amplitudes");
    elements = checked_add(
        elements,
        checked_multiply(
            occupied_square,
            tno_square,
            "triple-domain projected pair amplitudes"),
        "triple-domain projected amplitudes");
    auto factor_panel_elements = checked_add(
        checked_multiply(
            occupied, tno, "triple-domain factor OV panel"),
        occupied_square,
        "triple-domain factor OV+OO panel");
    factor_panel_elements = checked_add(
        factor_panel_elements,
        tno_square,
        "triple-domain factor OV+OO+VV panel");
    elements = checked_add(
        elements,
        checked_multiply(
            auxiliary_panel,
            factor_panel_elements,
            "triple-domain transformed factor panel"),
        "triple-domain total workspace");
    return checked_multiply(
        kComplexBytes, elements, "triple-domain workspace bytes");
}

PeriodicCorrelationByteCount streamed_factor_tile_bytes(
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto n_basis_squared =
        checked_square(dimensions.n_basis, "factor AO-pair extent");
    const auto ao_pair_block = dimensions.factor_ao_pair_block == 0
        ? n_basis_squared
        : dimensions.factor_ao_pair_block;
    if (ao_pair_block > n_basis_squared) {
        throw std::invalid_argument(
            "factor_ao_pair_block cannot exceed n_basis squared");
    }
    return estimate_periodic_streamed_factor_tile_bytes(
        dimensions.factor_k_bra_block,
        dimensions.factor_k_ket_block,
        dimensions.factor_auxiliary_block,
        ao_pair_block);
}

PeriodicCorrelationByteCount node_retained_bytes(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationResourceBudget& budget,
    PeriodicCorrelationByteCount payload_per_rank,
    PeriodicCorrelationByteCount extra_node_bytes = 0) {
    const auto rank_payload = checked_add(
        dimensions.per_rank_bytes,
        payload_per_rank,
        "per-rank retained bytes");
    const auto replicated = checked_multiply(
        budget.mpi_ranks, rank_payload, "MPI-replicated retained bytes");
    auto total = checked_add(
        dimensions.external_bytes,
        dimensions.shared_bytes,
        "external and shared bytes");
    total = checked_add(total, extra_node_bytes, "node-extra retained bytes");
    return checked_add(total, replicated, "node retained bytes");
}

struct WorkerChoice {
    PeriodicCorrelationByteCount concurrent_bytes = 0;
    std::uint64_t active_workers = 0;
};

bool candidate_fits(
    PeriodicCorrelationByteCount retained,
    PeriodicCorrelationByteCount worker_bytes,
    std::uint64_t workers,
    PeriodicCorrelationByteCount memory_limit) {
    try {
        const auto concurrent = checked_multiply(
            worker_bytes, workers, "concurrent uniform worker bytes");
        const auto peak = checked_add(
            retained, concurrent, "uniform-worker phase peak");
        return required_memory(peak) <= memory_limit;
    } catch (const ResourceArithmeticOverflow&) {
        return false;
    }
}

WorkerChoice choose_uniform_workers(
    PeriodicCorrelationByteCount retained,
    PeriodicCorrelationByteCount worker_bytes,
    std::uint64_t maximum_workers,
    PeriodicCorrelationByteCount memory_limit) {
    require_positive(maximum_workers, "maximum_workers");
    std::uint64_t selected = maximum_workers;
    if (memory_limit != 0) {
        std::uint64_t low = 1;
        std::uint64_t high = maximum_workers;
        selected = 0;
        while (low <= high) {
            const auto midpoint = low + (high - low) / 2;
            if (candidate_fits(
                    retained, worker_bytes, midpoint, memory_limit)) {
                selected = midpoint;
                if (midpoint == maximum_workers) {
                    break;
                }
                low = midpoint + 1;
            } else {
                high = midpoint - 1;
            }
        }
        // Preserve the irreducible one-worker peak so failure reports remain
        // truthful when even one worker cannot fit.
        if (selected == 0) {
            selected = 1;
        }
    }
    return {
        checked_multiply(
            worker_bytes, selected, "selected uniform worker bytes"),
        selected,
    };
}

WorkerChoice choose_variable_workers(
    PeriodicCorrelationByteCount retained,
    std::vector<PeriodicCorrelationByteCount> worker_bytes,
    std::uint64_t maximum_workers,
    PeriodicCorrelationByteCount memory_limit) {
    if (worker_bytes.empty()) {
        return {};
    }
    std::sort(worker_bytes.begin(), worker_bytes.end(), std::greater<>());
    const auto candidate_count = std::min<std::uint64_t>(
        maximum_workers, worker_bytes.size());
    require_positive(candidate_count, "candidate worker count");

    PeriodicCorrelationByteCount prefix = 0;
    PeriodicCorrelationByteCount selected_prefix = 0;
    std::uint64_t selected = 0;
    for (std::uint64_t index = 0; index < candidate_count; ++index) {
        try {
            prefix = checked_add(
                prefix, worker_bytes[index], "top-worker workspace sum");
            const auto peak = checked_add(
                retained, prefix, "variable-worker phase peak");
            if (memory_limit == 0 || required_memory(peak) <= memory_limit) {
                selected = index + 1;
                selected_prefix = prefix;
            } else {
                break;
            }
        } catch (const ResourceArithmeticOverflow&) {
            break;
        }
    }
    if (selected == 0) {
        selected = 1;
        selected_prefix = worker_bytes.front();
    }
    return {selected_prefix, selected};
}

void retain_largest_workspace(
    std::vector<PeriodicCorrelationByteCount>& largest,
    PeriodicCorrelationByteCount candidate,
    std::uint64_t limit) {
    require_positive(limit, "top-workspace limit");
    const auto comparator = std::greater<PeriodicCorrelationByteCount>{};
    if (largest.size() < limit) {
        largest.push_back(candidate);
        std::push_heap(largest.begin(), largest.end(), comparator);
    } else if (candidate > largest.front()) {
        std::pop_heap(largest.begin(), largest.end(), comparator);
        largest.back() = candidate;
        std::push_heap(largest.begin(), largest.end(), comparator);
    }
}

PeriodicCorrelationPhaseEstimate make_phase(
    PeriodicCorrelationResourcePhase phase,
    PeriodicCorrelationByteCount retained,
    WorkerChoice workers,
    PeriodicCorrelationByteCount scratch = 0) {
    PeriodicCorrelationPhaseEstimate result;
    result.phase = phase;
    result.retained_bytes = retained;
    result.concurrent_worker_bytes = workers.concurrent_bytes;
    result.active_workers = workers.active_workers;
    result.peak_memory_bytes = checked_add(
        retained, workers.concurrent_bytes, "phase peak memory");
    result.scratch_bytes = scratch;
    return result;
}

void finish_plan(
    PeriodicCorrelationResourcePlan& plan,
    const PeriodicCorrelationResourceBudget& budget,
    PeriodicCorrelationAdmissionCode success_code) {
    for (const auto& phase : plan.phases) {
        plan.modeled_peak_memory_bytes = std::max(
            plan.modeled_peak_memory_bytes, phase.peak_memory_bytes);
        plan.modeled_scratch_bytes = std::max(
            plan.modeled_scratch_bytes, phase.scratch_bytes);
    }
    plan.required_memory_bytes = required_memory(
        plan.modeled_peak_memory_bytes);
    plan.required_scratch_bytes = required_scratch(
        plan.modeled_scratch_bytes);

    if (budget.memory_limit_bytes == 0) {
        plan.admission =
            PeriodicCorrelationAdmissionCode::MissingMemoryLimit;
    } else if (plan.required_memory_bytes > budget.memory_limit_bytes) {
        plan.admission = PeriodicCorrelationAdmissionCode::MemoryExceeded;
    } else if (plan.modeled_scratch_bytes != 0
               && budget.scratch_limit_bytes == 0) {
        plan.admission =
            PeriodicCorrelationAdmissionCode::MissingScratchLimit;
    } else if (plan.required_scratch_bytes > budget.scratch_limit_bytes) {
        plan.admission = PeriodicCorrelationAdmissionCode::ScratchExceeded;
    } else {
        plan.admission = success_code;
    }
}

PeriodicCorrelationResourcePlan make_static_plan(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationResourceBudget& budget,
    PeriodicCorrelationEstimateStage stage) {
    PeriodicCorrelationResourcePlan plan;
    plan.stage = stage;
    plan.allocation_contract_version =
        kPeriodicCorrelationStreamedResourceContractVersion;
    plan.calculation_identity = dimensions.calculation_identity;
    plan.allocation_identity = dimensions.allocation_identity;

    const auto mean_field_payload =
        mean_field_payload_per_rank(dimensions);
    const auto mean_field_retained = node_retained_bytes(
        dimensions, budget, mean_field_payload);
    const auto factor_tile = streamed_factor_tile_bytes(dimensions);
    const auto max_workers = total_worker_limit(budget);
    const auto factor_build_retained = node_retained_bytes(
        dimensions,
        budget,
        factor_metric_payload_per_rank(dimensions));
    const auto factor_build_workers = choose_uniform_workers(
        factor_build_retained,
        checked_multiply(
            2, factor_tile, "factor-build double-buffered tile"),
        max_workers,
        budget.memory_limit_bytes);
    plan.active_factor_build_workers =
        factor_build_workers.active_workers;
    plan.phases.push_back(make_phase(
        PeriodicCorrelationResourcePhase::FactorBuild,
        factor_build_retained,
        factor_build_workers));

    const auto mean_field_workers = choose_uniform_workers(
        mean_field_retained,
        factor_tile,
        max_workers,
        budget.memory_limit_bytes);
    plan.active_mean_field_workers = mean_field_workers.active_workers;
    plan.phases.push_back(make_phase(
        PeriodicCorrelationResourcePhase::MeanField,
        mean_field_retained,
        mean_field_workers));

    const auto localization_payload =
        localization_payload_per_rank(dimensions);
    plan.phases.push_back(make_phase(
        PeriodicCorrelationResourcePhase::Localization,
        node_retained_bytes(dimensions, budget, localization_payload),
        {}));

    const auto domain_reference_payload = checked_add(
        mean_field_payload,
        localization_payload,
        "domain-build reference and localization payload");
    const auto domain_build_retained = node_retained_bytes(
        dimensions,
        budget,
        domain_reference_payload,
        dimensions.pair_domain_metadata_upper_bytes);
    const auto domain_build_workers = choose_uniform_workers(
        domain_build_retained,
        domain_build_workspace_bytes(dimensions),
        max_workers,
        budget.memory_limit_bytes);
    plan.active_domain_build_workers = domain_build_workers.active_workers;
    plan.phases.push_back(make_phase(
        PeriodicCorrelationResourcePhase::DomainBuild,
        domain_build_retained,
        domain_build_workers));
    return plan;
}

PeriodicCorrelationByteCount sum_singles_elements(
    const PeriodicCorrelationDomainCensus& census) {
    PeriodicCorrelationByteCount total = 0;
    for (const auto count : census.singles_pno_counts) {
        total = checked_add(total, count, "singles-PNO element sum");
    }
    return total;
}

PeriodicCorrelationByteCount sum_pair_amplitude_elements(
    const PeriodicCorrelationDomainCensus& census) {
    PeriodicCorrelationByteCount total = 0;
    for (const auto& pair : census.pair_domains) {
        if (pair.n_pno > pair.n_extended_virtual) {
            throw std::invalid_argument(
                "pair PNO rank cannot exceed its extended virtual rank");
        }
        total = checked_add(
            total,
            checked_square(pair.n_pno, "pair-PNO amplitude square"),
            "pair-PNO amplitude element sum");
    }
    return total;
}

/// Allocation contract for one blocked extended-domain residual.  The largest
/// virtual ladder is B_v * e^3, never e^4, while local RI factors are consumed
/// in fixed auxiliary panels.  If the implementation changes this inventory,
/// this formula and its synthetic regression must change in the same commit.
PeriodicCorrelationByteCount pair_workspace_bytes(
    const PeriodicCorrelationPairDomain& pair,
    const PeriodicCorrelationDomainCensus& census,
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto occupied = pair.n_local_occupied;
    const auto extended = pair.n_extended_virtual;
    const auto panel = std::min(
        dimensions.factor_auxiliary_block, pair.n_local_auxiliary);
    const auto virtual_block = std::min(
        census.pair_virtual_block, extended);

    const auto occupied_square = checked_square(
        occupied, "pair-workspace occupied square");
    const auto extended_square = checked_square(
        extended, "pair-workspace virtual square");
    const auto occupied_virtual = checked_multiply(
        occupied, extended, "pair-workspace occupied-virtual product");
    auto panel_elements = checked_add(
        occupied_virtual,
        occupied_square,
        "pair-workspace panel OV+OO");
    panel_elements = checked_add(
        panel_elements,
        extended_square,
        "pair-workspace panel OV+OO+VV");
    panel_elements = checked_multiply(
        panel, panel_elements, "pair-workspace factor panel");

    const auto occupied2_virtual2 = checked_multiply(
        occupied_square,
        extended_square,
        "pair-workspace OOVV product");
    const auto occupied_fourth = checked_square(
        occupied_square, "pair-workspace occupied fourth power");
    const auto extended_cube = checked_cube(
        extended, "pair-workspace virtual cube");
    auto residual_elements = checked_multiply(
        6, occupied2_virtual2, "pair-workspace OOVV intermediates");
    residual_elements = checked_add(
        residual_elements,
        checked_multiply(
            2, occupied_virtual, "pair-workspace OV intermediates"),
        "pair-workspace residual inventory");
    residual_elements = checked_add(
        residual_elements,
        checked_multiply(
            3, extended_square, "pair-workspace VV intermediates"),
        "pair-workspace residual inventory");
    residual_elements = checked_add(
        residual_elements,
        checked_multiply(
            2, occupied_square, "pair-workspace OO intermediates"),
        "pair-workspace residual inventory");
    residual_elements = checked_add(
        residual_elements,
        occupied_fourth,
        "pair-workspace residual inventory");
    residual_elements = checked_add(
        residual_elements,
        extended,
        "pair-workspace residual inventory");
    const auto blocked_ladder = checked_multiply(
        2,
        checked_multiply(
            virtual_block,
            extended_cube,
            "pair-workspace blocked virtual ladder"),
        "pair-workspace double-buffered virtual ladder");
    residual_elements = checked_add(
        residual_elements,
        blocked_ladder,
        "pair-workspace residual plus ladder");
    return checked_multiply(
        kComplexBytes,
        checked_add(
            panel_elements,
            residual_elements,
            "pair-workspace total elements"),
        "pair-workspace bytes");
}

/// Allocation contract for one TNO-blocked triples task.  Cubic TNO tensors
/// are blocked in their leading virtual index; the exact persistent u^3
/// amplitudes for an iterative correction are accounted separately.
PeriodicCorrelationByteCount triple_workspace_bytes(
    const PeriodicCorrelationTripleDomain& triple,
    const PeriodicCorrelationDomainCensus& census,
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto occupied = triple.n_local_occupied;
    const auto tno = triple.n_tno;
    const auto panel = std::min(
        dimensions.factor_auxiliary_block, triple.n_local_auxiliary);
    const auto virtual_block = std::min(
        census.triple_virtual_block, tno);
    const auto occupied_square = checked_square(
        occupied, "triple-workspace occupied square");
    const auto tno_square = checked_square(
        tno, "triple-workspace TNO square");
    const auto occupied_tno = checked_multiply(
        occupied, tno, "triple-workspace occupied-TNO product");

    auto panel_elements = checked_add(
        occupied_tno,
        occupied_square,
        "triple-workspace panel OV+OO");
    panel_elements = checked_add(
        panel_elements,
        tno_square,
        "triple-workspace panel OV+OO+VV");
    panel_elements = checked_multiply(
        panel, panel_elements, "triple-workspace factor panel");

    auto contraction_elements = checked_multiply(
        2,
        checked_multiply(
            occupied_square,
            tno_square,
            "triple-workspace OOVV product"),
        "triple-workspace OOVV buffers");
    contraction_elements = checked_add(
        contraction_elements,
        occupied_tno,
        "triple-workspace contraction inventory");
    const auto blocked_tno_cube = checked_multiply(
        virtual_block,
        tno_square,
        "triple-workspace blocked TNO cube");
    contraction_elements = checked_add(
        contraction_elements,
        checked_multiply(
            occupied,
            blocked_tno_cube,
            "triple-workspace occupied blocked cube"),
        "triple-workspace contraction inventory");
    contraction_elements = checked_add(
        contraction_elements,
        checked_multiply(
            checked_cube(occupied, "triple-workspace occupied cube"),
            tno,
            "triple-workspace OOO-V product"),
        "triple-workspace contraction inventory");
    contraction_elements = checked_add(
        contraction_elements,
        checked_multiply(
            4,
            blocked_tno_cube,
            "triple-workspace streamed TNO buffers"),
        "triple-workspace contraction inventory");
    contraction_elements = checked_add(
        contraction_elements,
        tno_square,
        "triple-workspace contraction inventory");
    return checked_multiply(
        kComplexBytes,
        checked_add(
            panel_elements,
            contraction_elements,
            "triple-workspace total elements"),
        "triple-workspace bytes");
}

/// One direct AO-to-local factor realization. Two raw AO tiles overlap two
/// half-transform/output panels; completed domains are written directly to the
/// selected backing store and are accounted separately as retained state.
PeriodicCorrelationByteCount factor_transform_workspace_bytes(
    const PeriodicCorrelationFactorDomain& factor,
    const PeriodicCorrelationStaticDimensions& dimensions) {
    const auto panel = std::min(
        dimensions.factor_auxiliary_block, factor.n_auxiliary);
    const auto local_rank = checked_add(
        factor.n_occupied,
        factor.n_virtual,
        "factor-transform local rank");
    const auto half_transform = checked_multiply(
        dimensions.domain_ao_support_upper_bound,
        local_rank,
        "factor-transform AO-local panel");
    auto output_matrices = checked_multiply(
        factor.n_occupied,
        factor.n_virtual,
        "factor-transform OV output");
    output_matrices = checked_add(
        output_matrices,
        checked_square(
            factor.n_occupied, "factor-transform OO output"),
        "factor-transform OV+OO output");
    output_matrices = checked_add(
        output_matrices,
        checked_square(factor.n_virtual, "factor-transform VV output"),
        "factor-transform OV+OO+VV output");
    const auto local_panel_elements = checked_multiply(
        2,
        checked_multiply(
            panel,
            checked_add(
                half_transform,
                output_matrices,
                "factor-transform half/output panel"),
            "factor-transform auxiliary panel"),
        "factor-transform double-buffered local panels");
    const auto local_panel_bytes = checked_multiply(
        kComplexBytes,
        local_panel_elements,
        "factor-transform local-panel bytes");
    return checked_add(
        checked_multiply(
            2,
            streamed_factor_tile_bytes(dimensions),
            "factor-transform double-buffered AO tiles"),
        local_panel_bytes,
        "factor-transform total workspace");
}

bool is_lower_sha256(const std::string& value) {
    if (value.size() != 64) {
        return false;
    }
    return std::all_of(
        value.begin(), value.end(), [](char digit) {
            return (digit >= '0' && digit <= '9')
                || (digit >= 'a' && digit <= 'f');
        });
}

std::pair<PeriodicCorrelationByteCount, std::uint64_t>
factor_store_inventory(const PeriodicCorrelationDomainCensus& census) {
    const auto& domains = census.factor_domains;
    PeriodicCorrelationByteCount bytes = 0;
    std::uint64_t unique_count = 0;
    for (std::size_t index = 0; index < domains.size(); ++index) {
        if (!is_lower_sha256(domains[index].storage_identity)) {
            throw std::invalid_argument(
                "factor storage identity must be a lowercase SHA-256 digest");
        }
        if (index != 0
            && domains[index - 1].storage_identity
                >= domains[index].storage_identity) {
            throw std::invalid_argument(
                "factor-domain identities must be unique and canonically ordered");
        }
        const auto occupied_virtual = checked_multiply(
            domains[index].n_occupied,
            domains[index].n_virtual,
            "factor-store occupied-virtual product");
        auto matrix_elements = checked_add(
            occupied_virtual,
            checked_square(
                domains[index].n_occupied,
                "factor-store occupied square"),
            "factor-store OV+OO elements");
        matrix_elements = checked_add(
            matrix_elements,
            checked_square(
                domains[index].n_virtual,
                "factor-store virtual square"),
            "factor-store OV+OO+VV elements");
        auto domain_bytes = checked_multiply(
            domains[index].n_auxiliary,
            matrix_elements,
            "factor-store three-index elements");
        domain_bytes = checked_multiply(
            kComplexBytes, domain_bytes, "factor-store complex bytes");
        domain_bytes = checked_add(
            domain_bytes,
            domains[index].header_bytes,
            "factor-store domain header");
        bytes = checked_add(bytes, domain_bytes, "factor-store byte sum");
        unique_count = checked_add(
            unique_count, 1, "unique factor-domain count");
    }
    return {bytes, unique_count};
}

PeriodicCorrelationByteCount triples_amplitude_inventory(
    const PeriodicCorrelationDomainCensus& census) {
    if (!census.iterative_triples
        || !census.triple_domain_census_complete) {
        return 0;
    }
    PeriodicCorrelationByteCount elements = 0;
    for (const auto& triple : census.triple_domains) {
        elements = checked_add(
            elements,
            checked_cube(triple.n_tno, "iterative-triples TNO cube"),
            "iterative-triples amplitude element sum");
    }
    return checked_multiply(
        kComplexBytes, elements, "iterative-triples amplitude bytes");
}

PeriodicCorrelationByteCount add_many(
    std::initializer_list<PeriodicCorrelationByteCount> values,
    const char* label) {
    PeriodicCorrelationByteCount total = 0;
    for (const auto value : values) {
        total = checked_add(total, value, label);
    }
    return total;
}

void validate_census(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationDomainCensus& census) {
    if (census.calculation_identity != dimensions.calculation_identity) {
        throw std::invalid_argument(
            "domain census calculation identity does not match preflight");
    }
    if (census.allocation_identity != dimensions.allocation_identity) {
        throw std::invalid_argument(
            "domain census allocation identity does not match preflight");
    }
    if (!is_lower_sha256(census.census_identity)) {
        throw std::invalid_argument(
            "census_identity must be a lowercase SHA-256 digest");
    }
    if (!census.pair_domain_census_complete) {
        throw std::invalid_argument("pair-domain census is incomplete");
    }
    if (census.symmetry_reduction_used
            != dimensions.symmetry_reduction_requested
        || census.symmetry_mapping_identity
            != dimensions.symmetry_mapping_identity
        || census.symmetry_full_kpoint_count != dimensions.n_kpoints
        || census.symmetry_representative_count
            != dimensions.symmetry_representative_count
        || census.symmetry_weight_sum != dimensions.symmetry_weight_sum) {
        throw std::invalid_argument(
            "symmetry census does not match the sealed preflight mapping");
    }
    require_positive(
        census.symmetry_representative_count,
        "symmetry_representative_count");
    if (census.symmetry_representative_count
        > census.symmetry_full_kpoint_count) {
        throw std::invalid_argument(
            "symmetry representative count exceeds the full mesh");
    }
    if (census.symmetry_reduction_used) {
        if (!census.symmetry_mapping_verified
            || !is_lower_sha256(census.symmetry_mapping_identity)) {
            throw std::invalid_argument(
                "a symmetry-reduced census requires a verified mapping digest");
        }
    } else if (!census.symmetry_mapping_identity.empty()
               || census.symmetry_representative_count
                   != census.symmetry_full_kpoint_count) {
        throw std::invalid_argument(
            "an unreduced census must enumerate the complete k-point mesh");
    }
    if (census.triples_requested != dimensions.triples_requested) {
        throw std::invalid_argument(
            "census triples request does not match static preflight");
    }
    if (!census.triples_requested
        && (census.triple_domain_census_complete
            || census.iterative_triples
            || census.disk_backed_triples
            || !census.triple_domains.empty()
            || census.triple_candidate_count != 0
            || census.evaluated_triple_count != 0
            || census.screened_triple_count != 0
            || census.triple_domain_metadata_bytes != 0)) {
        throw std::invalid_argument(
            "triples census/options require triples_requested");
    }
    if (!census.triple_domain_census_complete
        && (!census.triple_domains.empty()
            || census.triple_candidate_count != 0
            || census.evaluated_triple_count != 0
            || census.screened_triple_count != 0
            || census.triple_domain_metadata_bytes != 0
            || census.final_checkpoint_payload_bytes != 0)) {
        throw std::invalid_argument(
            "triple-domain inventory requires a complete triples census");
    }
    if (census.iterative_triples || census.disk_backed_triples
        || census.triples_amplitude_replicas != 0) {
        throw std::invalid_argument(
            "streamed resource contract version 1 excludes iterative triples");
    }
    if (census.singles_pno_counts.size() != dimensions.n_home_occupied) {
        throw std::invalid_argument(
            "singles-PNO census does not match correlated occupied count");
    }
    require_positive(census.pair_candidate_count, "pair_candidate_count");
    if (census.pair_candidate_count
        != dimensions.expected_pair_candidate_count) {
        throw std::invalid_argument(
            "pair candidate count does not match sealed preflight");
    }
    const auto classified_pairs = checked_add(
        checked_add(
            census.strong_pair_count,
            census.weak_pair_count,
            "strong and weak pair counts"),
        census.neglected_pair_count,
        "classified pair count");
    if (classified_pairs != census.pair_candidate_count
        || census.strong_pair_count != census.pair_domains.size()) {
        throw std::invalid_argument(
            "pair census completeness counts are inconsistent");
    }
    require_positive(census.strong_pair_count, "strong_pair_count");
    if (!census.factor_manifest_verified
        || !is_lower_sha256(census.factor_manifest_identity)) {
        throw std::invalid_argument(
            "factor-domain census requires a verified native manifest");
    }
    if (census.expected_factor_domain_count
        != census.factor_domains.size()) {
        throw std::invalid_argument(
            "factor-domain census count is incomplete");
    }
    if (!census.pair_domains.empty() && census.factor_domains.empty()) {
        throw std::invalid_argument(
            "strong pair domains require stored or streamed factor domains");
    }
    require_positive(
        census.pair_domain_metadata_bytes,
        "pair_domain_metadata_bytes");
    if (census.pair_domain_metadata_bytes
        > dimensions.pair_domain_metadata_upper_bytes) {
        throw std::invalid_argument(
            "pair-domain metadata exceeds its admitted upper bound");
    }
    if (census.pair_domain_metadata_bytes
        < minimum_pair_census_metadata_bytes(dimensions, census)) {
        throw std::invalid_argument(
            "pair-domain metadata is below its structural minimum");
    }
    for (const auto count : census.singles_pno_counts) {
        require_positive(count, "singles PNO rank");
        if (count > dimensions.domain_pno_upper_bound) {
            throw std::invalid_argument(
                "singles PNO rank exceeds its admitted upper bound");
        }
    }
    for (const auto& pair : census.pair_domains) {
        require_positive(pair.n_pno, "pair PNO rank");
        require_positive(pair.n_local_occupied, "pair local occupied rank");
        require_positive(pair.n_extended_virtual, "pair extended virtual rank");
        require_positive(pair.n_local_auxiliary, "pair local auxiliary rank");
        if (pair.n_pno > pair.n_extended_virtual) {
            throw std::invalid_argument(
                "pair PNO rank cannot exceed its extended virtual rank");
        }
        if (pair.n_pno > dimensions.domain_pno_upper_bound
            || pair.n_extended_virtual
                > dimensions.domain_pao_upper_bound
            || pair.n_local_occupied
                > dimensions.domain_local_occupied_upper_bound
            || pair.n_local_auxiliary
                > dimensions.domain_local_auxiliary_upper_bound) {
            throw std::invalid_argument(
                "pair-domain rank exceeds its admitted upper bound");
        }
    }
    if (census.triple_domain_census_complete) {
        if (census.triple_candidate_count
            != dimensions.expected_triple_candidate_count) {
            throw std::invalid_argument(
                "triple candidate count does not match sealed preflight");
        }
        const auto classified_triples = checked_add(
            census.evaluated_triple_count,
            census.screened_triple_count,
            "classified triple count");
        if (classified_triples != census.triple_candidate_count
            || census.evaluated_triple_count
                != census.triple_domains.size()) {
            throw std::invalid_argument(
                "triple census completeness counts are inconsistent");
        }
        require_positive(
            census.triple_domain_metadata_bytes,
            "triple_domain_metadata_bytes");
        if (census.triple_domain_metadata_bytes
            > dimensions.triple_domain_metadata_upper_bytes) {
            throw std::invalid_argument(
                "triple-domain metadata exceeds its admitted upper bound");
        }
        if (census.triple_domain_metadata_bytes
            < minimum_triple_metadata_bytes(dimensions)) {
            throw std::invalid_argument(
                "triple-domain metadata is below its structural minimum");
        }
    }
    for (const auto& triple : census.triple_domains) {
        require_positive(triple.n_tno, "triple TNO rank");
        require_positive(
            triple.n_local_occupied, "triple local occupied rank");
        require_positive(
            triple.n_local_auxiliary, "triple local auxiliary rank");
        if (triple.n_tno > dimensions.triple_tno_upper_bound
            || triple.n_local_occupied
                > dimensions.triple_local_occupied_upper_bound
            || triple.n_local_auxiliary
                > dimensions.triple_local_auxiliary_upper_bound) {
            throw std::invalid_argument(
                "triple-domain rank exceeds its admitted upper bound");
        }
    }
    for (const auto& factor : census.factor_domains) {
        require_positive(factor.n_auxiliary, "factor auxiliary rank");
        require_positive(factor.n_occupied, "factor occupied rank");
        require_positive(factor.n_virtual, "factor virtual rank");
        if (factor.n_auxiliary
                > dimensions.domain_local_auxiliary_upper_bound
            || factor.n_occupied
                > dimensions.domain_local_occupied_upper_bound
            || factor.n_virtual > dimensions.domain_pao_upper_bound) {
            throw std::invalid_argument(
                "factor-domain rank exceeds its admitted upper bound");
        }
    }
    if (census.amplitude_replicas != 1
        || census.factor_store_replicas != 1) {
        throw std::invalid_argument(
            "resource contract version 1 requires one state/factor replica");
    }
    if (!census.pair_domains.empty()) {
        require_positive(census.pair_virtual_block, "pair_virtual_block");
    }
    if (!census.triple_domains.empty()) {
        require_positive(
            census.triple_virtual_block, "triple_virtual_block");
    }
    if (census.checkpoint_generations == 1) {
        throw std::invalid_argument(
            "checkpoint generations must be zero or at least two");
    }
    if (census.checkpoint_generations == 0) {
        if (census.checkpoint_replicas != 0
            || census.checkpoint_inventory_complete
            || census.checkpoint_includes_factor_store
            || census.checkpoint_provenance_bytes != 0
            || census.factor_regeneration_recipe_bytes != 0
            || census.pair_checkpoint_payload_bytes != 0
            || census.final_checkpoint_payload_bytes != 0
            || census.checkpoint_write_buffer_extra_bytes != 0) {
            throw std::invalid_argument(
                "checkpoint inventory must be zero when checkpointing is disabled");
        }
    } else {
        if (!census.checkpoint_inventory_complete) {
            throw std::invalid_argument(
                "checkpoint inventory is incomplete");
        }
        if (census.checkpoint_replicas != 1) {
            throw std::invalid_argument(
                "resource contract version 1 requires one checkpoint replica");
        }
        require_positive(
            census.pair_checkpoint_payload_bytes,
            "pair_checkpoint_payload_bytes");
        if (census.checkpoint_provenance_bytes
            < kCheckpointProvenanceMinimumBytes) {
            throw std::invalid_argument(
                "checkpoint provenance is below its structural minimum");
        }
        if (!census.checkpoint_includes_factor_store) {
            require_positive(
                census.factor_regeneration_recipe_bytes,
                "factor_regeneration_recipe_bytes");
        } else if (census.factor_regeneration_recipe_bytes != 0) {
            throw std::invalid_argument(
                "serialized factors cannot also claim a regeneration recipe");
        }
        if (census.triples_requested
            && census.triple_domain_census_complete) {
            require_positive(
                census.final_checkpoint_payload_bytes,
                "final_checkpoint_payload_bytes");
        } else if (census.final_checkpoint_payload_bytes != 0) {
            throw std::invalid_argument(
                "final checkpoint payload requires a complete triples census");
        }
    }
}

}  // namespace

PeriodicCorrelationByteCount estimate_periodic_dense_gdf_cache_bytes(
    std::uint64_t n_kpairs,
    std::uint64_t n_auxiliary,
    std::uint64_t n_basis) {
    auto elements = checked_multiply(
        n_kpairs, n_auxiliary, "dense GDF k-pair auxiliary extent");
    elements = checked_multiply(
        elements,
        checked_square(n_basis, "dense GDF basis square"),
        "dense GDF factor elements");
    return checked_multiply(
        kComplexBytes, elements, "dense GDF cache bytes");
}

PeriodicCorrelationByteCount estimate_periodic_streamed_factor_tile_bytes(
    std::uint64_t k_bra_block,
    std::uint64_t k_ket_block,
    std::uint64_t auxiliary_block,
    std::uint64_t ao_pair_block) {
    require_positive(k_bra_block, "k_bra_block");
    require_positive(k_ket_block, "k_ket_block");
    require_positive(auxiliary_block, "auxiliary_block");
    require_positive(ao_pair_block, "ao_pair_block");
    auto elements = checked_multiply(
        k_bra_block, k_ket_block, "streamed factor k blocks");
    elements = checked_multiply(
        elements, auxiliary_block, "streamed factor auxiliary block");
    elements = checked_multiply(
        elements, ao_pair_block, "streamed factor AO-pair block");
    return checked_multiply(
        kComplexBytes, elements, "streamed factor tile bytes");
}

PeriodicCorrelationTranslationPairCounts
estimate_periodic_correlation_translation_pair_counts(
    std::array<int, 3> mesh,
    std::uint64_t n_home_occupied) {
    require_positive(n_home_occupied, "n_home_occupied");
    const RegularKMesh addressing(mesh);
    static_assert(
        sizeof(std::size_t) <= sizeof(std::uint64_t),
        "translation-pair counts require size_t to fit in uint64_t");
    const auto cell_count =
        static_cast<std::uint64_t>(addressing.size());

    std::uint64_t self_inverse_translation_count = 1;
    for (const int axis_size : mesh) {
        if (axis_size % 2 == 0) {
            self_inverse_translation_count = checked_multiply(
                self_inverse_translation_count,
                2,
                "self-inverse translation count");
        }
    }

    // cell_count and its two-torsion count have the same parity. Splitting
    // both terms before summing computes (cell_count + t) / 2 without an
    // overflowing cell_count + t intermediate.
    auto same_orbital_translation_count = checked_add(
        cell_count / 2,
        self_inverse_translation_count / 2,
        "same-orbital translation representatives");
    if (cell_count % 2 != 0) {
        same_orbital_translation_count = checked_add(
            same_orbital_translation_count,
            1,
            "same-orbital translation representatives");
    }

    const auto occupied_pair_count = n_home_occupied % 2 == 0
        ? checked_multiply(
              n_home_occupied / 2,
              n_home_occupied - 1,
              "home occupied pair count")
        : checked_multiply(
              n_home_occupied,
              (n_home_occupied - 1) / 2,
              "home occupied pair count");
    auto candidate_count = checked_multiply(
        n_home_occupied,
        same_orbital_translation_count,
        "same-orbital translation-pair candidates");
    candidate_count = checked_add(
        candidate_count,
        checked_multiply(
            occupied_pair_count,
            cell_count,
            "different-orbital translation-pair candidates"),
        "translation-pair candidate count");

    const auto placed_orbital_count = checked_multiply(
        n_home_occupied,
        cell_count,
        "placed correlated occupied count");
    const auto placed_pair_count = placed_orbital_count % 2 == 0
        ? checked_multiply(
              placed_orbital_count / 2,
              checked_add(
                  placed_orbital_count, 1, "placed occupied pairs n+1"),
              "placed occupied pair count")
        : checked_multiply(
              placed_orbital_count,
              checked_add(
                  placed_orbital_count / 2,
                  1,
                  "placed occupied pairs half n+1"),
              "placed occupied pair count");

    PeriodicCorrelationTranslationPairCounts counts;
    counts.cell_count = cell_count;
    counts.self_inverse_translation_count =
        self_inverse_translation_count;
    counts.candidate_count = candidate_count;
    counts.placed_pair_count = placed_pair_count;
    return counts;
}

std::uint64_t periodic_occupied_triple_count(std::uint64_t n_occupied) {
    std::array<std::uint64_t, 3> factors = {
        n_occupied,
        checked_add(n_occupied, 1, "occupied triples n+1"),
        checked_add(n_occupied, 2, "occupied triples n+2"),
    };
    for (auto& factor : factors) {
        if (factor % 2 == 0) {
            factor /= 2;
            break;
        }
    }
    for (auto& factor : factors) {
        if (factor % 3 == 0) {
            factor /= 3;
            break;
        }
    }
    return checked_multiply(
        checked_multiply(
            factors[0], factors[1], "occupied triples first product"),
        factors[2],
        "occupied triples count");
}

std::uint64_t periodic_correlation_bytes_to_mib_ceil(
    PeriodicCorrelationByteCount bytes) {
    return checked_ceil_divide(bytes, kMiB, "bytes-to-MiB conversion");
}

PeriodicCorrelationResourcePlan plan_periodic_correlation_static_resources(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationResourceBudget& budget) {
    bool dimensions_validated = false;
    try {
        validate_static_dimensions(dimensions, budget);
        dimensions_validated = true;
        auto plan = make_static_plan(
            dimensions,
            budget,
            PeriodicCorrelationEstimateStage::StaticPreflight);
        plan.pair_domain_census_complete = false;
        plan.triple_domain_census_complete = false;
        finish_plan(
            plan,
            budget,
            PeriodicCorrelationAdmissionCode::ReadyForPairDomainCensus);
        return plan;
    } catch (const ResourceArithmeticOverflow& error) {
        PeriodicCorrelationResourcePlan plan;
        plan.stage = PeriodicCorrelationEstimateStage::StaticPreflight;
        if (dimensions_validated) {
            plan.allocation_contract_version =
                dimensions.allocation_contract_version;
            plan.calculation_identity = dimensions.calculation_identity;
            plan.allocation_identity = dimensions.allocation_identity;
        }
        plan.admission =
            PeriodicCorrelationAdmissionCode::ArithmeticOverflow;
        plan.failure_detail = error.what();
        return plan;
    }
}

PeriodicCorrelationResourcePlan plan_periodic_correlation_domain_resources(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationDomainCensus& census,
    const PeriodicCorrelationResourceBudget& budget) {
    bool dimensions_validated = false;
    bool census_validated = false;
    try {
        validate_static_dimensions(dimensions, budget);
        dimensions_validated = true;
        validate_census(dimensions, census);
        census_validated = true;
        const auto stage = census.triple_domain_census_complete
            ? PeriodicCorrelationEstimateStage::TripleDomainCensus
            : PeriodicCorrelationEstimateStage::PairDomainCensus;
        auto plan = make_static_plan(
            dimensions,
            budget,
            stage);
        plan.pair_domain_census_complete = true;
        plan.triple_domain_census_complete =
            census.triple_domain_census_complete;
        plan.census_identity = census.census_identity;

        const auto singles_elements = sum_singles_elements(census);
        const auto pair_elements = sum_pair_amplitude_elements(census);
        plan.amplitude_bytes = checked_multiply(
            kComplexBytes,
            checked_add(
                singles_elements,
                pair_elements,
                "local amplitude element inventory"),
            "local amplitude bytes");
        const auto factor_inventory = factor_store_inventory(census);
        plan.factor_store_bytes = factor_inventory.first;
        plan.unique_factor_domains = factor_inventory.second;
        plan.triples_amplitude_bytes = triples_amplitude_inventory(census);
        if (census.checkpoint_generations != 0) {
            const auto factor_checkpoint_component =
                census.checkpoint_includes_factor_store
                ? plan.factor_store_bytes
                : census.factor_regeneration_recipe_bytes;
            const auto pair_checkpoint_minimum = add_many(
                {
                    plan.amplitude_bytes,
                    census.pair_domain_metadata_bytes,
                    census.checkpoint_provenance_bytes,
                    factor_checkpoint_component,
                },
                "minimum pair checkpoint payload");
            if (census.pair_checkpoint_payload_bytes
                < pair_checkpoint_minimum) {
                throw std::invalid_argument(
                    "pair checkpoint payload is below its known inventory");
            }
            if (census.triples_requested
                && census.triple_domain_census_complete) {
                const auto final_checkpoint_minimum = add_many(
                    {
                        census.pair_checkpoint_payload_bytes,
                        census.triple_domain_metadata_bytes,
                        plan.triples_amplitude_bytes,
                    },
                    "minimum final checkpoint payload");
                if (census.final_checkpoint_payload_bytes
                    < final_checkpoint_minimum) {
                    throw std::invalid_argument(
                        "final checkpoint payload is below its known inventory");
                }
            }
        }

        const auto mean_field_payload =
            mean_field_payload_per_rank(dimensions);
        const auto localization_payload =
            localization_payload_per_rank(dimensions);
        const auto reference_payload = checked_add(
            mean_field_payload,
            localization_payload,
            "post-HF reference and localization payload");
        const auto physical_factor_store = checked_multiply(
            plan.factor_store_bytes,
            census.factor_store_replicas,
            "physical factor-store bytes");
        const auto factor_memory = census.disk_backed_factors
            ? 0
            : physical_factor_store;
        const auto factor_scratch = census.disk_backed_factors
            ? physical_factor_store
            : 0;
        const auto max_workers = total_worker_limit(budget);

        // Local factor realization overlaps the q metric, pair-domain maps,
        // a growing completed store (when in memory), and two streamed tiles.
        // Disk-backed factors are written directly to their final store.
        const auto factor_store_build_payload = checked_add(
            reference_payload,
            factor_metric_payload_per_rank(dimensions),
            "factor-store reference and q-metric payload");
        const auto factor_store_build_retained = node_retained_bytes(
            dimensions,
            budget,
            factor_store_build_payload,
            add_many(
                {
                    factor_memory,
                    census.pair_domain_metadata_bytes,
                    census.retained_extra_bytes,
                },
                "factor-store-build node-extra retained bytes"));
        std::vector<PeriodicCorrelationByteCount> factor_workspaces;
        for (const auto& factor : census.factor_domains) {
            retain_largest_workspace(
                factor_workspaces,
                factor_transform_workspace_bytes(factor, dimensions),
                max_workers);
        }
        const auto factor_store_build_workers = choose_variable_workers(
            factor_store_build_retained,
            std::move(factor_workspaces),
            max_workers,
            budget.memory_limit_bytes);
        plan.active_factor_store_build_workers =
            factor_store_build_workers.active_workers;
        plan.phases.push_back(make_phase(
            PeriodicCorrelationResourcePhase::FactorStoreBuild,
            factor_store_build_retained,
            factor_store_build_workers,
            checked_add(
                factor_scratch,
                census.scratch_extra_bytes,
                "factor-store-build scratch extent")));

        const auto diis_vectors = checked_multiply(
            2, census.diis_depth, "pair DIIS vector count");
        const auto in_memory_state_copies = checked_add(
            4, diis_vectors, "in-memory pair-state copy count");
        constexpr PeriodicCorrelationByteCount disk_state_copies = 4;
        const auto pair_state_one_copy = checked_multiply(
            census.disk_backed_diis
                ? disk_state_copies
                : in_memory_state_copies,
            plan.amplitude_bytes,
            "pair-solve amplitude state");
        const auto pair_state_node = checked_multiply(
            pair_state_one_copy,
            census.amplitude_replicas,
            "physical pair-solve amplitude state");
        const auto retained_node_extra = add_many(
            {
                factor_memory,
                pair_state_node,
                census.pair_domain_metadata_bytes,
                census.retained_extra_bytes,
            },
            "pair-solve node-extra retained bytes");
        const auto pair_retained = node_retained_bytes(
            dimensions,
            budget,
            reference_payload,
            retained_node_extra);

        std::vector<PeriodicCorrelationByteCount> pair_workspaces;
        for (const auto& pair : census.pair_domains) {
            retain_largest_workspace(
                pair_workspaces,
                pair_workspace_bytes(pair, census, dimensions),
                max_workers);
        }
        const auto pair_workers = choose_variable_workers(
            pair_retained,
            std::move(pair_workspaces),
            max_workers,
            budget.memory_limit_bytes);
        plan.active_pair_workers = pair_workers.active_workers;

        if (census.disk_backed_diis) {
            const auto logical_diis = checked_multiply(
                diis_vectors,
                plan.amplitude_bytes,
                "disk-backed DIIS logical extent");
            plan.diis_scratch_bytes = checked_multiply(
                logical_diis,
                census.amplitude_replicas,
                "physical DIIS scratch");
        }
        const auto pair_checkpoint_payload = checked_multiply(
            census.pair_checkpoint_payload_bytes,
            census.checkpoint_replicas,
            "physical pair checkpoint payload");
        const auto pair_checkpoint_steady = checked_multiply(
            census.checkpoint_generations,
            pair_checkpoint_payload,
            "retained pair checkpoint generations");
        const auto pair_scratch = add_many(
            {
                factor_scratch,
                plan.diis_scratch_bytes,
                pair_checkpoint_steady,
                census.scratch_extra_bytes,
            },
            "pair-phase scratch extent");
        plan.phases.push_back(make_phase(
            PeriodicCorrelationResourcePhase::PairSolve,
            pair_retained,
            pair_workers,
            pair_scratch));

        auto final_state_retained = pair_retained;
        auto final_noncheckpoint_scratch = add_many(
            {
                factor_scratch,
                plan.diis_scratch_bytes,
                census.scratch_extra_bytes,
            },
            "pair non-checkpoint scratch extent");
        auto final_checkpoint_payload = pair_checkpoint_payload;

        if (census.triples_requested) {
            const auto triple_domain_build_retained = checked_add(
                pair_retained,
                dimensions.triple_domain_metadata_upper_bytes,
                "triple-domain-build retained metadata upper bound");
            const auto triple_domain_build_workers = choose_uniform_workers(
                triple_domain_build_retained,
                triple_domain_build_workspace_bytes(dimensions),
                max_workers,
                budget.memory_limit_bytes);
            plan.active_triple_domain_build_workers =
                triple_domain_build_workers.active_workers;
            plan.phases.push_back(make_phase(
                PeriodicCorrelationResourcePhase::TripleDomainBuild,
                triple_domain_build_retained,
                triple_domain_build_workers,
                pair_scratch));
            final_state_retained = triple_domain_build_retained;
            if (census.checkpoint_generations != 0) {
                const auto physical_triple_metadata_upper = checked_multiply(
                    dimensions.triple_domain_metadata_upper_bytes,
                    census.checkpoint_replicas,
                    "physical triple-domain checkpoint metadata upper bound");
                final_checkpoint_payload = checked_add(
                    pair_checkpoint_payload,
                    physical_triple_metadata_upper,
                    "post-TNO checkpoint payload upper bound");
            }
        }

        if (census.triple_domain_census_complete) {
            const auto retained_pair_amplitudes_one_copy = checked_multiply(
                2,
                plan.amplitude_bytes,
                "triples retained pair amplitudes");
            auto triples_state_node = checked_multiply(
                retained_pair_amplitudes_one_copy,
                census.amplitude_replicas,
                "physical triples pair-amplitude state");
            const auto triples_node_extra = add_many(
                {
                    factor_memory,
                    triples_state_node,
                    census.pair_domain_metadata_bytes,
                    census.triple_domain_metadata_bytes,
                    census.retained_extra_bytes,
                },
                "triples node-extra retained bytes");
            const auto triples_retained = node_retained_bytes(
                dimensions,
                budget,
                reference_payload,
                triples_node_extra);
            std::vector<PeriodicCorrelationByteCount> triple_workspaces;
            for (const auto& triple : census.triple_domains) {
                retain_largest_workspace(
                    triple_workspaces,
                    triple_workspace_bytes(triple, census, dimensions),
                    max_workers);
            }
            const auto triple_workers = choose_variable_workers(
                triples_retained,
                std::move(triple_workspaces),
                max_workers,
                budget.memory_limit_bytes);
            plan.active_triple_workers = triple_workers.active_workers;
            const auto final_payload = checked_multiply(
                census.final_checkpoint_payload_bytes,
                census.checkpoint_replicas,
                "physical final checkpoint payload");
            const auto final_checkpoint_steady = checked_multiply(
                census.checkpoint_generations,
                final_payload,
                "retained final checkpoint generations");
            const auto triples_scratch = add_many(
                {
                    factor_scratch,
                    final_checkpoint_steady,
                    census.scratch_extra_bytes,
                },
                "triples-phase scratch extent");
            plan.phases.push_back(make_phase(
                PeriodicCorrelationResourcePhase::Triples,
                triples_retained,
                triple_workers,
                triples_scratch));
            final_state_retained = triples_retained;
            final_noncheckpoint_scratch = checked_add(
                factor_scratch,
                census.scratch_extra_bytes,
                "triples non-checkpoint scratch extent");
            final_checkpoint_payload = final_payload;
        }

        // Contract version 1 writes directly to the final backing store and
        // makes every in-memory/disk decision before allocation. During the
        // transition the live solver state overlaps one complete serialization
        // payload plus any implementation-measured write buffer.
        if (census.checkpoint_generations != 0) {
            const auto checkpoint_transition_bytes = checked_add(
                final_checkpoint_payload,
                census.checkpoint_write_buffer_extra_bytes,
                "checkpoint transition buffer");
            const auto atomic_generation_count = checked_add(
                census.checkpoint_generations,
                1,
                "atomic checkpoint generation count");
            plan.checkpoint_scratch_bytes = checked_multiply(
                atomic_generation_count,
                final_checkpoint_payload,
                "atomic checkpoint scratch extent");
            plan.phases.push_back(make_phase(
                PeriodicCorrelationResourcePhase::Checkpoint,
                checked_add(
                    final_state_retained,
                    checkpoint_transition_bytes,
                    "checkpoint transition retained bytes"),
                {},
                checked_add(
                    final_noncheckpoint_scratch,
                    plan.checkpoint_scratch_bytes,
                    "checkpoint transition scratch extent")));
        }

        const auto success_code =
            census.triples_requested
                && !census.triple_domain_census_complete
            ? PeriodicCorrelationAdmissionCode::ReadyForTripleDomainCensus
            : PeriodicCorrelationAdmissionCode::Admitted;
        finish_plan(plan, budget, success_code);
        return plan;
    } catch (const ResourceArithmeticOverflow& error) {
        PeriodicCorrelationResourcePlan plan;
        plan.stage = census_validated && census.triple_domain_census_complete
            ? PeriodicCorrelationEstimateStage::TripleDomainCensus
            : PeriodicCorrelationEstimateStage::PairDomainCensus;
        if (dimensions_validated) {
            plan.allocation_contract_version =
                dimensions.allocation_contract_version;
            plan.calculation_identity = dimensions.calculation_identity;
            plan.allocation_identity = dimensions.allocation_identity;
        }
        plan.pair_domain_census_complete = census_validated;
        plan.triple_domain_census_complete =
            census_validated && census.triple_domain_census_complete;
        if (census_validated) {
            plan.census_identity = census.census_identity;
        }
        plan.admission =
            PeriodicCorrelationAdmissionCode::ArithmeticOverflow;
        plan.failure_detail = error.what();
        return plan;
    }
}

}  // namespace vibeqc
