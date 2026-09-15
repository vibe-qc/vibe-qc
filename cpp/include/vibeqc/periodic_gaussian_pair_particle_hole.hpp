#pragma once

// Frozen actual-source bare particle-hole contribution, not an iterative
// CCSD driver. Stanton1991 doi:10.1063/1.460620 and Riplinger2013
// doi:10.1063/1.4773581 Eq.(7): replaces the three bare W1/W2/WX seeds
// in BOTH particle-hole passes; never add to an unchanged full residual.
#include <optional>
#include "vibeqc/periodic_gaussian_pair_interaction.hpp"
#include "vibeqc/periodic_gaussian_pair_ccsd.hpp"
#include "vibeqc/bounded_restricted_pair_ccsd_particle_hole.hpp"

namespace vibeqc {
struct PeriodicGaussianPairParticleHoleLiveInventory {
    std::uint64_t other_live_numerical_bytes_per_worker = 0;
    std::uint64_t other_live_control_bytes_per_worker = 0;
    std::uint64_t backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianPairParticleHoleCaps {
    PeriodicGaussianPairInteractionCaps interaction;
    std::uint64_t maximum_source_slots = 0, maximum_pair_count = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0, maximum_node_bytes = 0, maximum_work_units = 0;
    std::uint64_t maximum_interaction_control_bytes = 0;
    std::uint64_t maximum_factor_panels = 0, maximum_tile_calls = 0;
    std::uint64_t maximum_reciprocal_candidates = 0, maximum_image_candidates = 0;
};
struct PeriodicGaussianPairParticleHolePlan {
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, pair_count = 0;
    std::uint64_t target_i = 0, target_j = 0, target_dimension = 0, maximum_source_dimension = 0;
    std::uint64_t source_slots = 0;
    bool domain_generated = false;
    std::uint64_t borrowed_warmstart_numerical_bytes = 0, borrowed_ccsd_numerical_bytes = 0;
    std::uint64_t borrowed_common_numerical_bytes = 0, borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t borrowed_owner_control_bytes = 0;
    // Explicit conservative overlap in nested role inventories, not extra
    // allocations. Whole source owners are never silently subtracted.
    std::uint64_t duplicate_frame_geometry_padding_bytes = 0, duplicate_t_role_padding_bytes = 0;
    std::uint64_t accumulator_owned_bytes = 0, maximum_interaction_owned_bytes = 0;
    // Contraction phase includes the resident accumulator H plus K/J/O
    // and explicit K transpose (32ABmax), not just its transient part.
    std::uint64_t maximum_transpose_bytes = 0, contraction_phase_bytes = 0;
    std::uint64_t output_numerical_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t outer_control_storage_bytes = 0, control_storage_reservation_bytes = 0;
    std::uint64_t maximum_interaction_control_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t worker_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t metadata_work_units = 0, snapshot_validation_work_units = 0;
    std::uint64_t frame_validation_work_units = 0, interaction_work_units = 0;
    std::uint64_t transpose_work_units = 0, work_units = 0;
    std::uint64_t factor_panels = 0, tile_calls = 0;
    std::uint64_t reciprocal_candidate_evaluations = 0, image_candidate_evaluations = 0;
    BoundedRestrictedPairCCSDParticleHoleMemoryPlan accumulator;
};
struct PeriodicGaussianPairParticleHoleDiagnostics {
    std::uint64_t completed_source_slots = 0, completed_interactions = 0;
    std::uint64_t completed_factor_panels = 0, completed_tile_calls = 0;
    std::uint64_t reciprocal_candidate_evaluations = 0, image_candidate_evaluations = 0;
    std::uint64_t charged_work_units_upper_bound = 0;
    double maximum_integral_roundoff_error = 0.0, maximum_overlap_imaginary_norm = 0.0;
    BoundedRestrictedPairCCSDParticleHoleDiagnostics accumulator;
};
class PeriodicGaussianPairParticleHoleResult {
public:
    PeriodicGaussianPairParticleHoleResult(const PeriodicGaussianPairParticleHoleResult&) = delete;
    PeriodicGaussianPairParticleHoleResult& operator=(const PeriodicGaussianPairParticleHoleResult&) = delete;
    PeriodicGaussianPairParticleHoleResult(PeriodicGaussianPairParticleHoleResult&&) noexcept = default;
    PeriodicGaussianPairParticleHoleResult& operator=(PeriodicGaussianPairParticleHoleResult&&) = delete;
    const PeriodicGaussianPairParticleHolePlan& memory() const;
    const PeriodicGaussianPairParticleHoleDiagnostics& diagnostics() const;
    const double* residual_data() const;
    const std::string& identity_sha256() const;
    const std::string& payload_sha256() const;
    const std::string& consumed_interactions_identity_sha256() const;
    const std::string& ccsd_identity_sha256() const;
    const std::string& ccsd_amplitude_payload_sha256() const;
    const std::string& warmstart_identity_sha256() const;
    const std::string& origin_provider_identity_sha256() const;
    const std::string& hf_reference_source_identity_sha256() const;
    const std::string& pair_spaces_identity_sha256() const;
    const std::string& target_frame_identity_sha256() const;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const;
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const;
    bool matched_finite_gaussian_hf_recipe() const noexcept { return true; }
    bool original_provider_projection_reproduced() const noexcept { return false; }
    bool entire_ccsd_residual() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
private:
    PeriodicGaussianPairParticleHoleResult() = default;
    void require_live() const;
    PeriodicGaussianPairParticleHolePlan memory_;
    PeriodicGaussianPairParticleHoleDiagnostics diagnostics_;
    std::optional<BoundedRestrictedPairCCSDParticleHoleResult> residual_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::string identity_,consumed_,ccsd_,amplitudes_,warm_,provider_,hf_,pairs_,target_;
    friend PeriodicGaussianPairParticleHoleResult build_periodic_gaussian_pair_particle_hole(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
        const BasisSet&,const BasisSet&,const PeriodicCorrelationWannier&,
        const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianPairMP2Result&,const PeriodicGaussianPairCCSDResult&,
        std::uint64_t,std::uint64_t,const PeriodicGaussianPairInteractionConfig&,
        const PeriodicGaussianPairInteractionOptions&,const PeriodicGaussianPairParticleHoleLiveInventory&,
        const PeriodicGaussianPairParticleHoleCaps&);
};
// Count/shape caps precede two bounded metadata walks, then the complete
// inventory precedes payload scans/allocations/factories. No plan table.
// One K/J/O owner is live. H=16A^2+8ABmax; owned peak is
// H+max(max interaction peak,32ABmax). T is borrowed, never copied.
// Frozen, converged source owners must remain alive and unmoved throughout.
// Same finite-source recipe does NOT prove the new real-row projection is
// bitwise identical to the origin common provider. Both lineages are sealed.
PeriodicGaussianPairParticleHolePlan plan_periodic_gaussian_pair_particle_hole(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairMP2Result&,const PeriodicGaussianPairCCSDResult&,
    std::uint64_t target_i,std::uint64_t target_j,const PeriodicGaussianPairInteractionConfig&,
    const PeriodicGaussianPairInteractionOptions&,const PeriodicGaussianPairParticleHoleLiveInventory&,
    const PeriodicGaussianPairParticleHoleCaps&);
PeriodicGaussianPairParticleHoleResult build_periodic_gaussian_pair_particle_hole(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairMP2Result&,const PeriodicGaussianPairCCSDResult&,
    std::uint64_t target_i,std::uint64_t target_j,const PeriodicGaussianPairInteractionConfig&,
    const PeriodicGaussianPairInteractionOptions&,const PeriodicGaussianPairParticleHoleLiveInventory&,
    const PeriodicGaussianPairParticleHoleCaps&);

// Current numerical trial amplitudes in authentic warm pair frames. No
// final CCSD owner/convergence is required or claimed. The Reader remains
// borrowed, immutable and alive; native pair C pointers/extents MUST be
// those of the exact warm PairSpace owners. Singles are only numerically
// validated/inventoried, not physically certified by this contribution.
struct PeriodicGaussianPairParticleHoleSnapshotPlan {
    PeriodicGaussianPairParticleHolePlan stream;
    std::uint64_t borrowed_reader_numerical_bytes = 0, borrowed_reader_table_bytes = 0;
    std::uint64_t reader_control_storage_bytes = 0;
    // Complete Reader roles are retained alongside complete warm owners.
    // Pair C occurs in both; no equality-based or silent subtraction.
    std::uint64_t reader_warm_coefficient_padding_bytes = 0;
};
class PeriodicGaussianPairParticleHoleSnapshotResult {
public:
    PeriodicGaussianPairParticleHoleSnapshotResult(const PeriodicGaussianPairParticleHoleSnapshotResult&) = delete;
    PeriodicGaussianPairParticleHoleSnapshotResult& operator=(const PeriodicGaussianPairParticleHoleSnapshotResult&) = delete;
    PeriodicGaussianPairParticleHoleSnapshotResult(PeriodicGaussianPairParticleHoleSnapshotResult&&) noexcept = default;
    PeriodicGaussianPairParticleHoleSnapshotResult& operator=(PeriodicGaussianPairParticleHoleSnapshotResult&&) = delete;
    const PeriodicGaussianPairParticleHoleSnapshotPlan& memory() const;
    const PeriodicGaussianPairParticleHoleDiagnostics& diagnostics() const;
    const double* residual_data() const;
    // Native synchronous visitor seam only; do not expose a mutable/movable
    // child at the Python boundary. Keep this result alive and unmoved.
    const BoundedRestrictedPairCCSDParticleHoleResult& bare_result() const &;
    const BoundedRestrictedPairCCSDParticleHoleResult& bare_result() const && = delete;
    std::uint64_t snapshot_index() const;
    const std::string& snapshot_identity_sha256() const;
    const std::string& identity_sha256() const;
    const std::string& payload_sha256() const;
    const std::string& consumed_interactions_identity_sha256() const;
    const std::string& warmstart_identity_sha256() const;
    const std::string& origin_provider_identity_sha256() const;
    const std::string& hf_reference_source_identity_sha256() const;
    const std::string& pair_spaces_identity_sha256() const;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const;
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const;
    bool converged_snapshot_certified() const noexcept { return false; }
    bool singles_physically_certified() const noexcept { return false; }
    bool original_provider_projection_reproduced() const noexcept { return false; }
    bool entire_ccsd_residual() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
private:
    PeriodicGaussianPairParticleHoleSnapshotResult() = default;
    void require_live() const;
    PeriodicGaussianPairParticleHoleSnapshotPlan memory_;
    PeriodicGaussianPairParticleHoleDiagnostics diagnostics_;
    std::optional<BoundedRestrictedPairCCSDParticleHoleResult> residual_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::uint64_t index_ = 0;
    std::string snapshot_,identity_,consumed_,warm_,provider_,hf_,pairs_;
    friend PeriodicGaussianPairParticleHoleSnapshotResult build_periodic_gaussian_pair_particle_hole_snapshot(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,const BasisSet&,const BasisSet&,
        const PeriodicCorrelationWannier&,const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairMP2Result&,
        const BoundedRestrictedPairCCSDAmplitudes&,std::uint64_t,std::uint64_t,std::uint64_t,
        const PeriodicGaussianPairInteractionConfig&,const PeriodicGaussianPairInteractionOptions&,
        const PeriodicGaussianPairParticleHoleLiveInventory&,const PeriodicGaussianPairParticleHoleCaps&);
};
// Reader construction/validation is a prior numerical operation with its
// own admission. These plans admit subsequent source/frame/Reader replays
// before any physical AO/gauge scan or factor factory. Remaining enclosing
// solver current/candidate arrays, singles owners/provider rows and controls
// not in Reader or warm are explicitly declared in Live by that caller.
PeriodicGaussianPairParticleHoleSnapshotPlan plan_periodic_gaussian_pair_particle_hole_snapshot(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationWannier&,
    const PeriodicCorrelationPAODomain&,const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairMP2Result&,const BoundedRestrictedPairCCSDAmplitudes&,
    std::uint64_t target_i,std::uint64_t target_j,const PeriodicGaussianPairInteractionConfig&,
    const PeriodicGaussianPairInteractionOptions&,const PeriodicGaussianPairParticleHoleLiveInventory&,
    const PeriodicGaussianPairParticleHoleCaps&);
PeriodicGaussianPairParticleHoleSnapshotResult build_periodic_gaussian_pair_particle_hole_snapshot(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,const BasisSet&,const BasisSet&,
    const PeriodicCorrelationWannier&,const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairMP2Result&,
    const BoundedRestrictedPairCCSDAmplitudes&,std::uint64_t target_i,std::uint64_t target_j,std::uint64_t snapshot_index,
    const PeriodicGaussianPairInteractionConfig&,const PeriodicGaussianPairInteractionOptions&,
    const PeriodicGaussianPairParticleHoleLiveInventory&,const PeriodicGaussianPairParticleHoleCaps&);
} // namespace vibeqc
