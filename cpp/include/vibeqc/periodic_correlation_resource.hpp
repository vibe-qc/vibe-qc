#pragma once

/// \file periodic_correlation_resource.hpp
/// \brief Tensor-allocation-free resource admission for periodic correlation.
///
/// The periodic correlation route is planned in three passes. The static pass
/// sizes the streamed mean-field and localization phases without constructing
/// k-pair tensors. The pair-census pass consumes exact PNO and local-factor
/// dimensions before pair amplitudes are allocated. A later triples-domain
/// pass is necessary because its TNO ranks depend on converged pair amplitudes;
/// triples are re-admitted from a separate exact census before allocation.
///
/// Contract version 1 describes a future native executor with preselected
/// in-memory or disk backing, bounded factor/virtual tiles, and direct writes
/// to final checkpoint storage. It does not authorize the existing dense
/// periodic or molecular kernels, and it permits no allocation-time fallback
/// to an unmodeled in-memory tensor.
///
/// All byte counts are node-wide unless a field explicitly says "per rank".
/// A zero memory or scratch limit means "unknown", not "unlimited"; production
/// admission therefore fails closed when a required limit is missing.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vibeqc {

using PeriodicCorrelationByteCount = std::uint64_t;

inline constexpr std::uint32_t
    kPeriodicCorrelationStreamedResourceContractVersion = 1;

enum class PeriodicCorrelationEstimateStage {
    StaticPreflight,
    PairDomainCensus,
    TripleDomainCensus,
};

enum class PeriodicCorrelationResourcePhase {
    FactorBuild,
    MeanField,
    Localization,
    DomainBuild,
    FactorStoreBuild,
    PairSolve,
    TripleDomainBuild,
    Triples,
    Checkpoint,
};

enum class PeriodicCorrelationAdmissionCode {
    ReadyForPairDomainCensus,
    ReadyForTripleDomainCensus,
    Admitted,
    MissingMemoryLimit,
    MissingScratchLimit,
    MemoryExceeded,
    ScratchExceeded,
    ArithmeticOverflow,
};

/// Resource limits and maximum concurrency for one compute node.
struct PeriodicCorrelationResourceBudget {
    PeriodicCorrelationByteCount memory_limit_bytes = 0;
    PeriodicCorrelationByteCount scratch_limit_bytes = 0;
    std::uint64_t mpi_ranks = 1;
    std::uint64_t workers_per_rank = 1;
};

/// Dimensions known before localization and pair-domain construction.
struct PeriodicCorrelationStaticDimensions {
    /// Must name the allocation layout actually implemented by the caller.
    /// Version 1 is the blocked, streamed local-factor contract documented in
    /// this header; existing dense Python DLPNO kernels do not satisfy it.
    std::uint32_t allocation_contract_version = 0;
    /// Lowercase SHA-256 over the physical calculation: periodic dimension,
    /// cell and basis, auxiliary basis, exact k coordinates/convention,
    /// Hamiltonian, frozen-core orbital identities, thresholds, symmetry map,
    /// and spin/root.
    std::string calculation_identity;
    /// Lowercase SHA-256 over allocation-contract version, streamed block
    /// layout, canonical work-list extents, bounds, and ownership policy.
    std::string allocation_identity;
    /// Assertion that all caller-owned live objects were included in the byte
    /// inventories below. Required even when a legitimate inventory is zero.
    bool static_inventory_complete = false;
    /// Physical periodicity: 1 for polymers, 2 for slabs, or 3 for bulk.
    /// Zero is an unset, fail-closed assembly default.
    int periodic_dimension = 0;
    /// Full Brillouin-zone mesh size.  Static admission is deliberately
    /// unreduced; symmetry representatives enter only in the verified census.
    /// mesh and is_shift carry the exact RegularKMesh convention; n_kpoints
    /// remains an explicit audited extent and must equal their product.
    std::array<int, 3> mesh = {0, 0, 0};
    std::array<int, 3> is_shift = {0, 0, 0};
    std::uint64_t n_kpoints = 0;
    std::uint64_t n_basis = 0;
    /// Rank of the retained orthonormal home-cell MO space after any linear-
    /// dependence removal. Total occupied plus full virtual must equal it.
    std::uint64_t n_effective_orbitals = 0;
    std::uint64_t n_auxiliary = 0;
    /// Total and correlated occupied bands in the home cell. The difference
    /// is the explicit frozen-core count sealed into calculation_identity.
    std::uint64_t n_home_total_occupied = 0;
    std::uint64_t n_home_occupied = 0;
    std::uint64_t n_home_virtual = 0;
    std::uint64_t n_spin_channels = 1;
    bool triples_requested = false;
    bool symmetry_reduction_requested = false;
    std::string symmetry_mapping_identity;
    std::uint64_t symmetry_representative_count = 0;
    std::uint64_t symmetry_weight_sum = 0;
    /// Native translation-unique occupied-pair extent. The caller repeats the
    /// value so it is sealed into the allocation identity; static validation
    /// recomputes it exactly before any domain screening or row allocation.
    std::uint64_t expected_pair_candidate_count = 0;
    /// Independently enumerated triples extent for the later amplitude-
    /// dependent TNO stage.
    std::uint64_t expected_triple_candidate_count = 0;

    /// Fixed streamed-factor tile.  The tile, rather than the full number of
    /// k pairs or auxiliary functions, is resident in each worker.
    std::uint64_t factor_k_bra_block = 1;
    std::uint64_t factor_k_ket_block = 1;
    std::uint64_t factor_q_block = 1;
    std::uint64_t factor_auxiliary_block = 1;
    /// AO-pair functions in a tile.  Zero selects the full n_basis^2 panel.
    std::uint64_t factor_ao_pair_block = 0;

    /// Memory outside the modeled tensors: once per node, shared once per
    /// node, and replicated by every MPI rank, respectively.  A caller that
    /// constructs PeriodicCorrelationAdmittedReference must exclude the
    /// retained mean-field state's numerical payload from external_bytes;
    /// that factory adds the state payload exactly once before planning.
    PeriodicCorrelationByteCount external_bytes = 0;
    PeriodicCorrelationByteCount shared_bytes = 0;
    PeriodicCorrelationByteCount per_rank_bytes = 0;
    PeriodicCorrelationByteCount localization_window_bytes_per_rank = 0;

    /// Conservative single-domain construction bounds. Contract version 1
    /// permits only one streamed PAO/PNO build per worker and does not retain
    /// factor domains until the pair census has been re-admitted.
    std::uint64_t domain_ao_support_upper_bound = 0;
    std::uint64_t domain_pao_upper_bound = 0;
    std::uint64_t domain_pno_upper_bound = 0;
    std::uint64_t domain_local_occupied_upper_bound = 0;
    std::uint64_t domain_local_auxiliary_upper_bound = 0;
    PeriodicCorrelationByteCount pair_domain_metadata_upper_bytes = 0;

    /// Bounds for one amplitude-dependent TNO-domain construction. Required
    /// only when triples_requested is true. The support/union dimensions may
    /// span translated cells and therefore are never inferred from n_basis.
    std::uint64_t triple_virtual_support_upper_bound = 0;
    std::uint64_t triple_union_pno_upper_bound = 0;
    std::uint64_t triple_tno_upper_bound = 0;
    std::uint64_t triple_local_occupied_upper_bound = 0;
    std::uint64_t triple_local_auxiliary_upper_bound = 0;
    PeriodicCorrelationByteCount triple_domain_metadata_upper_bytes = 0;
};

/// Exact translation quotient of all unordered correlated occupied pairs.
///
/// This is a structural count before pair classification or point-group
/// reduction. `candidate_count` is the number of reference-cell rows after
/// identifying `(i, 0; i, L)` with `(i, 0; i, -L)`. The sum of those rows'
/// translation-orbit multiplicities is `placed_pair_count`, the number of
/// unordered pairs among all `cell_count * n_home_occupied` placed orbitals.
struct PeriodicCorrelationTranslationPairCounts {
    std::uint64_t cell_count = 0;
    std::uint64_t self_inverse_translation_count = 0;
    std::uint64_t candidate_count = 0;
    std::uint64_t placed_pair_count = 0;
};

/// Exact retained dimensions for one strong occupied pair.
struct PeriodicCorrelationPairDomain {
    std::uint64_t n_pno = 0;
    std::uint64_t n_local_occupied = 0;
    std::uint64_t n_extended_virtual = 0;
    std::uint64_t n_local_auxiliary = 0;
};

/// Exact retained dimensions for one evaluated occupied triple.
struct PeriodicCorrelationTripleDomain {
    std::uint64_t n_tno = 0;
    std::uint64_t n_local_occupied = 0;
    std::uint64_t n_local_auxiliary = 0;
};

/// One unique local three-index factor domain.
///
/// storage_identity is a lowercase SHA-256 digest of the full domain identity.
/// The caller must supply identities in strictly increasing lexical order;
/// duplicates or noncanonical order are rejected without a planner-side copy.
struct PeriodicCorrelationFactorDomain {
    std::string storage_identity;
    std::uint64_t n_auxiliary = 0;
    std::uint64_t n_occupied = 0;
    std::uint64_t n_virtual = 0;
    PeriodicCorrelationByteCount header_bytes = 0;
};

/// Geometry-dependent inventory produced after localization/domain screening.
struct PeriodicCorrelationDomainCensus {
    /// Both must exactly match the static preflight identities.
    std::string calculation_identity;
    std::string allocation_identity;
    /// Native-builder digest over every ordered dynamic census record and
    /// backing/checkpoint choice. Copied to the plan for executor comparison.
    std::string census_identity;
    bool pair_domain_census_complete = false;
    /// Retained singles-PNO rank for each localized occupied orbital.
    std::vector<std::uint64_t> singles_pno_counts;
    std::vector<PeriodicCorrelationPairDomain> pair_domains;
    std::vector<PeriodicCorrelationTripleDomain> triple_domains;
    std::vector<PeriodicCorrelationFactorDomain> factor_domains;

    /// A reduced census is admissible only after the external symmetry module
    /// has verified every representative, weight, and reconstruction map.
    bool symmetry_reduction_used = false;
    bool symmetry_mapping_verified = false;
    /// Opaque digest of the externally verified representative/weight/map
    /// table. Required for reduction and sealed into calculation_identity.
    std::string symmetry_mapping_identity;
    std::uint64_t symmetry_full_kpoint_count = 0;
    std::uint64_t symmetry_representative_count = 0;
    std::uint64_t symmetry_weight_sum = 0;

    /// Pair ranks are always complete on entry. Triples are requested by the
    /// method but become complete only after their amplitude-dependent TNO
    /// domains have been constructed and this planner is called again.
    bool triples_requested = false;
    bool triple_domain_census_complete = false;

    /// Completeness seals supplied by the domain builders. Strong/evaluated
    /// counts must match the corresponding vectors, and all classifications
    /// must sum to their candidate count.
    std::uint64_t pair_candidate_count = 0;
    std::uint64_t strong_pair_count = 0;
    std::uint64_t weak_pair_count = 0;
    std::uint64_t neglected_pair_count = 0;
    /// Seal emitted by the native pair-domain builder over canonical
    /// pair-to-factor references. The resource planner verifies its declared
    /// factor extent; production must not synthesize this in Python.
    bool factor_manifest_verified = false;
    std::string factor_manifest_identity;
    std::uint64_t expected_factor_domain_count = 0;
    std::uint64_t triple_candidate_count = 0;
    std::uint64_t evaluated_triple_count = 0;
    std::uint64_t screened_triple_count = 0;

    std::uint64_t diis_depth = 6;
    bool disk_backed_diis = false;
    bool disk_backed_factors = true;
    bool iterative_triples = false;
    bool disk_backed_triples = false;

    /// Explicit physical copies across the node. Contract version 1 supports
    /// one MPI rank only, so active stores have exactly one replica.
    std::uint64_t amplitude_replicas = 0;
    std::uint64_t factor_store_replicas = 0;
    std::uint64_t triples_amplitude_replicas = 0;
    std::uint64_t checkpoint_replicas = 0;

    /// Blocking of the extended-virtual and TNO contractions.
    std::uint64_t pair_virtual_block = 32;
    std::uint64_t triple_virtual_block = 32;

    /// Valid restart generations retained. During an atomic write the planner
    /// also reserves one temporary generation. Payloads are exact logical
    /// one-copy serialized sizes and include amplitudes, domain transforms,
    /// orbital energies, symmetry/frozen-core provenance, and either factors
    /// or their complete deterministic regeneration recipe.
    std::uint64_t checkpoint_generations = 2;
    bool checkpoint_inventory_complete = false;
    bool checkpoint_includes_factor_store = false;
    /// Exact serialized symmetry/frozen-core/orbital-energy provenance.
    PeriodicCorrelationByteCount checkpoint_provenance_bytes = 0;
    /// Required when factors are regenerated rather than serialized.
    PeriodicCorrelationByteCount factor_regeneration_recipe_bytes = 0;
    PeriodicCorrelationByteCount pair_checkpoint_payload_bytes = 0;
    PeriodicCorrelationByteCount final_checkpoint_payload_bytes = 0;
    /// Exact node-wide persistent census storage, maps, transformations, and
    /// metadata. These include the census vectors passed to this planner.
    PeriodicCorrelationByteCount pair_domain_metadata_bytes = 0;
    PeriodicCorrelationByteCount triple_domain_metadata_bytes = 0;
    PeriodicCorrelationByteCount retained_extra_bytes = 0;
    /// Additional implementation-owned serialization workspace. The planner
    /// also retains one complete checkpoint payload during the write phase.
    PeriodicCorrelationByteCount checkpoint_write_buffer_extra_bytes = 0;
    PeriodicCorrelationByteCount scratch_extra_bytes = 0;
};

struct PeriodicCorrelationPhaseEstimate {
    PeriodicCorrelationResourcePhase phase =
        PeriodicCorrelationResourcePhase::MeanField;
    PeriodicCorrelationByteCount retained_bytes = 0;
    PeriodicCorrelationByteCount concurrent_worker_bytes = 0;
    std::uint64_t active_workers = 0;
    PeriodicCorrelationByteCount peak_memory_bytes = 0;
    PeriodicCorrelationByteCount scratch_bytes = 0;
};

/// Immutable-by-convention result returned before any large allocation.
struct PeriodicCorrelationResourcePlan {
    PeriodicCorrelationEstimateStage stage =
        PeriodicCorrelationEstimateStage::StaticPreflight;
    PeriodicCorrelationAdmissionCode admission =
        PeriodicCorrelationAdmissionCode::MissingMemoryLimit;
    std::uint32_t allocation_contract_version = 0;
    std::string calculation_identity;
    std::string allocation_identity;
    std::string census_identity;
    bool pair_domain_census_complete = false;
    bool triple_domain_census_complete = false;

    PeriodicCorrelationByteCount modeled_peak_memory_bytes = 0;
    PeriodicCorrelationByteCount required_memory_bytes = 0;
    PeriodicCorrelationByteCount modeled_scratch_bytes = 0;
    PeriodicCorrelationByteCount required_scratch_bytes = 0;

    /// Logical (one-copy) inventories retained for auditability. Physical
    /// node totals apply the explicit replica counts from the census.
    PeriodicCorrelationByteCount amplitude_bytes = 0;
    PeriodicCorrelationByteCount factor_store_bytes = 0;
    PeriodicCorrelationByteCount triples_amplitude_bytes = 0;
    PeriodicCorrelationByteCount diis_scratch_bytes = 0;
    PeriodicCorrelationByteCount checkpoint_scratch_bytes = 0;
    std::uint64_t unique_factor_domains = 0;

    std::uint64_t active_factor_build_workers = 0;
    std::uint64_t active_mean_field_workers = 0;
    std::uint64_t active_domain_build_workers = 0;
    std::uint64_t active_factor_store_build_workers = 0;
    std::uint64_t active_pair_workers = 0;
    std::uint64_t active_triple_domain_build_workers = 0;
    std::uint64_t active_triple_workers = 0;
    std::vector<PeriodicCorrelationPhaseEstimate> phases;

    /// Set only for ArithmeticOverflow; no output is emitted by the core.
    std::string failure_detail;
};

/// Static, full-mesh admission before localization/domain construction.
PeriodicCorrelationResourcePlan plan_periodic_correlation_static_resources(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationResourceBudget& budget);

/// Exact post-domain admission before allocating local amplitudes.
PeriodicCorrelationResourcePlan plan_periodic_correlation_domain_resources(
    const PeriodicCorrelationStaticDimensions& dimensions,
    const PeriodicCorrelationDomainCensus& census,
    const PeriodicCorrelationResourceBudget& budget);

/// Diagnostic helpers for proving the dense-to-streamed scaling change.
PeriodicCorrelationByteCount estimate_periodic_dense_gdf_cache_bytes(
    std::uint64_t n_kpairs,
    std::uint64_t n_auxiliary,
    std::uint64_t n_basis);

PeriodicCorrelationByteCount estimate_periodic_streamed_factor_tile_bytes(
    std::uint64_t k_bra_block,
    std::uint64_t k_ket_block,
    std::uint64_t auxiliary_block,
    std::uint64_t ao_pair_block);

/// Allocation-free exact count for the translation-unique occupied-pair
/// topology. Throws on an invalid mesh, zero occupied rank, or uint64 overflow.
PeriodicCorrelationTranslationPairCounts
estimate_periodic_correlation_translation_pair_counts(
    std::array<int, 3> mesh,
    std::uint64_t n_home_occupied);

/// Safe number of unordered occupied triples i <= j <= k.
std::uint64_t periodic_occupied_triple_count(std::uint64_t n_occupied);

/// Scheduler conversion that never rounds a positive byte count down.
std::uint64_t periodic_correlation_bytes_to_mib_ceil(
    PeriodicCorrelationByteCount bytes);

}  // namespace vibeqc
