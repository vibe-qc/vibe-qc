#pragma once

// Bounded compatibility bridge, not removal of the common-factor bottleneck.
// Nejad doi:10.1063/5.0290816 Eqs.37-40: initial SC-MP2 and periodic pair
// density in an admitted PAO pair frame, followed by original-Fock rotation
// and export into the common real virtual frame. The existing common-space
// PeriodicGaussianPairPNOResult and its occupation-size invariant are unchanged.
#include "vibeqc/periodic_gaussian_pair_pnos.hpp"
#include "vibeqc/periodic_correlation_real_pao_embedding.hpp"

namespace vibeqc {
struct PeriodicGaussianEmbeddedPairPNOOptions {
    PeriodicGaussianPairPNOOptions pno;
    // Explicit finite nonnegative Frobenius correction budgets. Ordered raw
    // projections are evaluated and hashed first. Only diagonal-pair G and
    // the projected Fock matrix are symmetrized after their measured gates.
    double maximum_diagonal_exchange_projection_norm = std::numeric_limits<double>::quiet_NaN();
    double maximum_fock_symmetry_projection_norm = std::numeric_limits<double>::quiet_NaN();
    // Finite strictly positive final common-frame COMPUTED-residual budgets.
    // Compensated contractions are rounded to binary64 before differencing;
    // these do not bound exact dot products of the represented input arrays.
    double maximum_exported_gram_error = 0.0;
    double maximum_exported_fock_error = 0.0;
    double maximum_exported_subspace_error = 0.0;
};
struct PeriodicGaussianEmbeddedPairPNOLiveInventory {
    // Beyond the original reference and complete actual basis/provider plus
    // embedding output. Original Gaussian/localization/domain owners, if
    // still live, belong here. No equal-hash ownership deduction is made.
    std::uint64_t other_live_numerical_bytes_per_worker = 0;
    std::uint64_t other_live_control_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianEmbeddedPairPNOCaps {
    std::uint64_t maximum_common_dimension = 0, maximum_generation_dimension = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes_per_worker = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0, maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_integral_calls = 0, maximum_work_units = 0;
};
// Count-only scientific/resource census, NOT an authenticated source plan.
// In an enclosing macro planner m=n is a monotone full-generation upper;
// the actual factory still verifies all owners and the exact embedding.
struct PeriodicGaussianEmbeddedPairPNOCountInput {
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, generation_dimension = 0;
    std::uint64_t provider_scalar_work_units = 0, provider_retained_row_bytes = 0;
};
struct PeriodicGaussianEmbeddedPairPNOCountPlan {
    std::uint64_t integral_calls = 0, provider_work_units = 0, numerical_work_units = 0, work_units = 0;
    std::uint64_t projection_phase_bytes = 0, amplitude_phase_bytes = 0, density_phase_bytes = 0;
    std::uint64_t semicanonical_phase_upper_bytes = 0, export_phase_upper_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, retained_output_upper_bytes = 0;
    std::uint64_t retained_generation_coefficient_upper_bytes = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_provider_row_bytes = 0, borrowed_embedding_bytes = 0;
    std::uint64_t complete_borrowed_numerical_bytes = 0, control_storage_reservation_bytes = 0;
};
// No owner access, payload scan or allocation. Includes caller extra CONTROL
// in the control reservation, but neither caller extra NUMERICAL/backend nor
// reference/node inventory: the enclosing planner must add those explicitly.
PeriodicGaussianEmbeddedPairPNOCountPlan plan_periodic_gaussian_embedded_pair_pno_counts(
    const PeriodicGaussianEmbeddedPairPNOCountInput&, const PeriodicGaussianEmbeddedPairPNOOptions&,
    const PeriodicGaussianEmbeddedPairPNOLiveInventory&);
struct PeriodicGaussianEmbeddedPairPNOPlan {
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, generation_dimension = 0;
    std::uint64_t occupied_slot_i = 0, occupied_slot_j = 0;
    std::uint64_t integral_calls = 0, provider_work_units = 0, numerical_work_units = 0, work_units = 0;
    std::uint64_t projection_phase_bytes = 0, amplitude_phase_bytes = 0, density_phase_bytes = 0;
    std::uint64_t semicanonical_phase_upper_bytes = 0, export_phase_upper_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, retained_output_upper_bytes = 0;
    std::uint64_t retained_generation_coefficient_upper_bytes = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_provider_row_bytes = 0, borrowed_embedding_bytes = 0;
    std::uint64_t complete_borrowed_numerical_bytes = 0, control_storage_reservation_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
};
struct PeriodicGaussianEmbeddedPairPNODiagnostics {
    // Dimension-m generation diagnostics. Its retained_output_bytes counts
    // the LOCAL D[m,t], eps[t], occupations[m], not the complete output.
    // Its integral_projection_error_bound and maximum_scalar_integral_roundoff_error
    // are COMMON scalar-provider diagnostics, not error bounds propagated through
    // X^T G X. That projection can amplify errors by column one-norm factors.
    PeriodicGaussianPairPNODiagnostics generation;
    std::uint64_t retained_dimension = 0, retained_output_bytes = 0;
    std::uint64_t retained_generation_coefficient_bytes = 0;
    std::uint64_t actual_semicanonical_phase_bytes = 0, actual_export_phase_bytes = 0;
    double maximum_raw_exchange_transpose_defect = 0.0;
    double diagonal_exchange_projection_frobenius_upper_bound = 0.0;
    double maximum_raw_fock_transpose_defect = 0.0;
    double fock_symmetry_projection_frobenius_upper_bound = 0.0;
    // Outward norm accumulation of COMPUTED, rounded residual entries only.
    // Dot-product rounding is not propagated as an interval; zero does not
    // prove an exactly zero represented-array contraction residual.
    double exported_gram_frobenius_upper_bound = 0.0;
    double exported_fock_frobenius_upper_bound = 0.0;
    // ||Ccommon-X X^T Ccommon||_F. Together with the existing local
    // semicanonical selected-projector audit, this checks export containment;
    // it is NOT a full-space Fock invariant-subspace residual.
    double exported_subspace_frobenius_upper_bound = 0.0;
};
class PeriodicGaussianEmbeddedPairPNOResult {
public:
    PeriodicGaussianEmbeddedPairPNOResult(const PeriodicGaussianEmbeddedPairPNOResult&) = delete;
    PeriodicGaussianEmbeddedPairPNOResult& operator=(const PeriodicGaussianEmbeddedPairPNOResult&) = delete;
    PeriodicGaussianEmbeddedPairPNOResult(PeriodicGaussianEmbeddedPairPNOResult&&) noexcept = default;
    PeriodicGaussianEmbeddedPairPNOResult& operator=(PeriodicGaussianEmbeddedPairPNOResult&&) = delete;
    const PeriodicGaussianEmbeddedPairPNOPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianEmbeddedPairPNODiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const PeriodicGaussianEmbeddedPairPNOOptions& options() const noexcept { return options_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    PeriodicCorrelationPlacedOccupied occupied_i() const noexcept { return i_; }
    PeriodicCorrelationPlacedOccupied occupied_j() const noexcept { return j_; }
    bool diagonal_pair() const noexcept { return memory_.occupied_slot_i==memory_.occupied_slot_j; }
    bool complete_generation_pair_space() const;
    bool complete_common_virtual_space() const;
    // Row-major common C[n,t], authentic generation-frame D[m,t], ascending
    // semicanonical eps[t], and ALL m pre-rotation density occupations.
    // D is moved from the original-Fock semicanonical result, never recovered
    // as X^T C. The common compatibility export is C=X D, without repair.
    // No occupation padding to n. All borrows require an unmoved live owner.
    const std::vector<double>& coefficients() const;
    const std::vector<double>& generation_coefficients() const;
    const std::vector<double>& energies() const;
    const std::vector<double>& original_pno_occupations() const;
    bool coupled_mp2_solution() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
    const std::string& provider_identity_sha256() const noexcept { return provider_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& embedding_identity_sha256() const noexcept { return embedding_; }
    // Original COMMON ordered G before pair projection; exactly the legacy
    // pair-PNO integral digest wire, streamed without retaining common n*n G.
    const std::string& common_exchange_integral_identity_sha256() const noexcept { return common_g_; }
    const std::string& raw_exchange_identity_sha256() const noexcept { return raw_g_; }
    const std::string& projected_exchange_identity_sha256() const noexcept { return g_; }
    const std::string& raw_fock_identity_sha256() const noexcept { return raw_f_; }
    const std::string& projected_fock_identity_sha256() const noexcept { return f_; }
    const std::string& initial_amplitude_identity_sha256() const noexcept { return amplitudes_; }
    const std::string& density_identity_sha256() const noexcept { return density_; }
    const std::string& source_payload_receipt_sha256() const noexcept { return sources_; }
private:
    PeriodicGaussianEmbeddedPairPNOResult() = default;
    void require_live() const;
    PeriodicGaussianEmbeddedPairPNOPlan memory_;
    PeriodicGaussianEmbeddedPairPNODiagnostics diagnostics_;
    PeriodicGaussianEmbeddedPairPNOOptions options_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicCorrelationPlacedOccupied i_,j_;
    std::vector<double> coefficients_, generation_coefficients_, energies_, occupations_;
    std::string identity_,payload_,basis_,provider_,hf_,embedding_,common_g_,raw_g_,g_,raw_f_,f_,amplitudes_,density_,sources_;
    friend PeriodicGaussianEmbeddedPairPNOResult make_periodic_gaussian_embedded_pair_pnos(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&, const PeriodicCorrelationRealPAOEmbedding&,
        std::uint64_t, std::uint64_t, const PeriodicGaussianEmbeddedPairPNOOptions&,
        const PeriodicGaussianEmbeddedPairPNOLiveInventory&, const PeriodicGaussianEmbeddedPairPNOCaps&);
};

// No callbacks, user arrays, common n*n G/T/D, or hidden backend workspace.
// n^2 common scalar ERIs are called exactly once each, charged BEFORE calls.
// With m=pair generation rank and t=retained rank: formation24m^2;
// G/F/eps/T24m^2+8m; density56m^2+8m; local semicanonical upper56m^2+16m;
// export upper8nm+8m^2+16m; retained8nt+8mt+8t+8m. Retaining the actual
// local coefficients changes final ownership, not the already admitted export
// peak (both local D and exported C were already live). Full source owners and
// logical control/backend reservations are additional, pre-admitted once.
// Cutoff0 retains the COMPLETE PAIR frame, which need not be common-complete.
PeriodicGaussianEmbeddedPairPNOPlan plan_periodic_gaussian_embedded_pair_pnos(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&, const PeriodicCorrelationRealPAOEmbedding&,
    std::uint64_t occupied_slot_i, std::uint64_t occupied_slot_j,
    const PeriodicGaussianEmbeddedPairPNOOptions&, const PeriodicGaussianEmbeddedPairPNOLiveInventory&,
    const PeriodicGaussianEmbeddedPairPNOCaps&);
PeriodicGaussianEmbeddedPairPNOResult make_periodic_gaussian_embedded_pair_pnos(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&, const PeriodicCorrelationRealPAOEmbedding&,
    std::uint64_t occupied_slot_i, std::uint64_t occupied_slot_j,
    const PeriodicGaussianEmbeddedPairPNOOptions&, const PeriodicGaussianEmbeddedPairPNOLiveInventory&,
    const PeriodicGaussianEmbeddedPairPNOCaps&);
} // namespace vibeqc
