#pragma once

// Actual finite-Gaussian mixed pair integrals for Riplinger/Neese (2013),
// doi:10.1063/1.4773581 Sec.II.C Eq.26. The two authentic PNO frames remain
// independent; their concatenation is NOT an orthonormal common basis.
// Nejad (2025), doi:10.1063/5.0290816 Eqs.2-10,26-28 fixes the inherited
// BvK normalization and physical AO overlap. This is one interaction's
// immutable K/J/O data, not a complete CCSD residual or an energy driver.
#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"
#include "vibeqc/periodic_correlation_real_local_provider.hpp"

namespace vibeqc {
struct PeriodicGaussianPairInteractionConfig {
    std::uint64_t auxiliary_block = 0;
    PeriodicGaussianLocalOrbitalFactorConfig panel;
};
struct PeriodicGaussianPairInteractionOptions {
    PeriodicCorrelationRealLocalProviderOptions real_projection;
    // Explicit positive norm gate before taking Re of the ORIGINAL-S
    // cross overlap. Not an assumed Euclidean overlap of exported columns.
    double maximum_overlap_imaginary_norm = 0.0;
};
struct PeriodicGaussianPairInteractionCaps {
    PeriodicGaussianMetricCaps resources, metric;
    PeriodicGaussianLocalOrbitalFactorCaps panel;
    std::uint64_t maximum_factor_panels = 0, maximum_tile_calls = 0;
    std::uint64_t maximum_image_candidate_evaluations = 0;
    // Explicit ceiling for concurrently live nested logical objects. This
    // is distinct from the exact variable numerical payload and OS RSS.
    std::uint64_t maximum_leaf_control_bytes = 0;
};
struct PeriodicGaussianPairInteractionPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_auxiliary = 0;
    std::uint64_t target_dimension = 0, source_dimension = 0, local_orbital_count = 0;
    std::uint64_t occupied_slot_i = 0, occupied_slot_k = 0;
    std::uint64_t density_count = 0, row_count = 0, self_inverse_q_count = 0;
    std::uint64_t auxiliary_block_count = 0, factor_panels = 0, tile_calls = 0;
    std::uint64_t retained_output_bytes = 0, integral_accumulator_bytes = 0;
    std::uint64_t row_slab_bytes = 0, norm_workspace_bytes = 0;
    std::uint64_t overlap_phase_bytes = 0, factor_phase_retained_bytes = 0;
    std::uint64_t whitener_bytes = 0, maximum_live_whitener_bytes = 0;
    std::uint64_t retained_partner_panel_bytes = 0;
    std::uint64_t metric_phase_upper_bytes = 0, panel_phase_upper_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t borrowed_local_numerical_bytes = 0, borrowed_frame_numerical_bytes = 0;
    bool direct_gram_frames = false;
    // Two retained Gram-generation SHA strings on the direct branch only.
    // The frame owner census already includes G_PNO, never add it again.
    std::uint64_t source_receipt_control_bytes = 0;
    bool local_geometry_a = false, local_geometry_b = false;
    std::uint64_t borrowed_geometry_numerical_bytes = 0, borrowed_geometry_control_bytes = 0;
    std::uint64_t geometry_validation_work_units = 0;
    std::uint64_t borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t minimum_leaf_control_bytes = 0, control_storage_reservation_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t source_factory_calls = 0, metric_calls = 0, whitening_calls = 0;
    std::uint64_t reciprocal_candidate_evaluations_upper_bound = 0;
    std::uint64_t image_candidate_evaluations_upper_bound = 0;
    std::uint64_t driver_work_units = 0, work_units = 0;
};
struct PeriodicGaussianPairInteractionDiagnostics {
    std::uint64_t completed_source_count = 0, completed_metric_count = 0;
    std::uint64_t completed_whitening_count = 0, completed_factor_panels = 0;
    std::uint64_t completed_tile_calls = 0, reciprocal_candidate_evaluations = 0;
    std::uint64_t image_candidate_evaluations = 0, charged_work_units_upper_bound = 0;
    std::uint64_t maximum_observed_owned_numerical_bytes = 0;
    std::uint64_t maximum_observed_per_worker_inventoried_bytes = 0;
    PeriodicCorrelationRealLocalProviderDiagnostics real_projection;
    double overlap_imaginary_frobenius_upper_bound = 0.0;
    // Bounds only dot rounding of the represented projected real rows.
    // The independent real-projection bound is above; neither includes
    // finite-source truncation or local/common orbital-generation error.
    double maximum_integral_roundoff_error = 0.0;
};
class PeriodicGaussianPairInteractionBlocks {
public:
    PeriodicGaussianPairInteractionBlocks(const PeriodicGaussianPairInteractionBlocks&) = delete;
    PeriodicGaussianPairInteractionBlocks& operator=(const PeriodicGaussianPairInteractionBlocks&) = delete;
    PeriodicGaussianPairInteractionBlocks(PeriodicGaussianPairInteractionBlocks&&) noexcept = default;
    PeriodicGaussianPairInteractionBlocks& operator=(PeriodicGaussianPairInteractionBlocks&&) = delete;
    const PeriodicGaussianPairInteractionPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianPairInteractionDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    // Row-major K[A,B]=(i a_A|k c_B), J[B,A]=(ik|c_B a_A),
    // O[B,A]=(1/K) sum_k C_B(k)^H S(k) C_A(k).
    const double* k_ab_data() const;
    const double* j_ba_data() const;
    const double* overlap_ba_data() const;
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& target_frame_identity_sha256() const noexcept { return target_; }
    const std::string& source_frame_identity_sha256() const noexcept { return source_; }
    bool direct_gram_frames() const noexcept { return memory_.direct_gram_frames; }
    // Generation receipts, not a common provider or a claim of bitwise
    // reproduction of generation's independently rounded real projection.
    // Both throw on the legacy/common-provider generation branch.
    const std::string& target_gram_identity_sha256() const;
    const std::string& source_gram_identity_sha256() const;
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& consumed_sources_identity_sha256() const noexcept { return consumed_; }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    bool original_provider_projection_reproduced() const noexcept { return false; }
private:
    PeriodicGaussianPairInteractionBlocks() = default;
    void require_live() const;
    PeriodicGaussianPairInteractionPlan memory_;
    PeriodicGaussianPairInteractionDiagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::vector<double> k_,j_,overlap_;
    std::string identity_,payload_,target_,source_,hf_,consumed_,target_gram_,source_gram_;
    friend PeriodicGaussianPairInteractionBlocks build_periodic_gaussian_pair_interaction_blocks(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
        const BasisSet&,const BasisSet&,const PeriodicCorrelationWannier&,
        const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOFrameView&,
        std::uint64_t,std::uint64_t,const PeriodicGaussianPairInteractionConfig&,
        const PeriodicGaussianPairInteractionOptions&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
        const PeriodicGaussianPairInteractionCaps&,
        const PeriodicGaussianPairPNOGeometryView*,const PeriodicGaussianPairPNOGeometryView*);
};
// Structural/count admission precedes source construction, frame/gauge/basis
// scans and allocations. Same-address PNO owners count once; equal receipts
// on different owners do not. Original shared reference state counts once.
// The caller declares other owners (including any still-live generation
// provider) in live. No provider/factor-store owner or user factor array is
// accepted or retained by this factory. All q/original auxiliary rows are
// traversed, retaining only a two-q row SLAB and local K/J/O outputs.
// Homogeneous DirectPAOGram frames are accepted ONLY with BOTH original
// authenticated PAO geometries. Their retained D[m,r] is applied directly;
// no common-C reprojection, fabricated provider receipt or mixed direct/old
// generation source is accepted. Other source-kind behavior is unchanged.
PeriodicGaussianPairInteractionPlan plan_periodic_gaussian_pair_interaction_blocks(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOFrameView&,
    std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_k,
    const PeriodicGaussianPairInteractionConfig&,const PeriodicGaussianPairInteractionOptions&,
    const PeriodicGaussianLocalOrbitalFactorLiveInventory&,const PeriodicGaussianPairInteractionCaps&,
    const PeriodicGaussianPairPNOGeometryView* geometry_a = nullptr,
    const PeriodicGaussianPairPNOGeometryView* geometry_b = nullptr);
PeriodicGaussianPairInteractionBlocks build_periodic_gaussian_pair_interaction_blocks(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const BasisSet&,const BasisSet&,const PeriodicCorrelationWannier&,
    const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOFrameView&,
    std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_k,
    const PeriodicGaussianPairInteractionConfig&,const PeriodicGaussianPairInteractionOptions&,
    const PeriodicGaussianLocalOrbitalFactorLiveInventory&,const PeriodicGaussianPairInteractionCaps&,
    const PeriodicGaussianPairPNOGeometryView* geometry_a = nullptr,
    const PeriodicGaussianPairPNOGeometryView* geometry_b = nullptr);
} // namespace vibeqc
