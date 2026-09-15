#pragma once

// Actual finite-Gaussian-source pair PNO generation, not coupled MP2 or a
// production DLPNO energy. Nejad doi:10.1063/5.0290816, Sec.IV.C Eqs.37-40:
// T=-G/Delta, U=2T-T^T, D=2(TU^T+T^TU)=4SS^T+12AA^T.
// There is NO 1/(1+delta_ij), including diagonal pairs (Eq.34: D=4TT^T).
// The legacy Riplinger doi:10.1063/1.4773581 Eq.23 primitive is unchanged.

#include <limits>
#include "vibeqc/pair_natural_orbitals.hpp"
#include "vibeqc/periodic_gaussian_real_local_provider.hpp"

namespace vibeqc {

struct PeriodicGaussianPairPNOOptions {
    // NaN deliberately requires an explicit scientific choice. Zero cutoff
    // is the documented complete-domain diagnostic; positive cuts retain >.
    double occupation_cutoff = std::numeric_limits<double>::quiet_NaN();
    double denominator_floor = 0.0;
    double maximum_initial_fvv_offdiagonal_norm = std::numeric_limits<double>::quiet_NaN();
    double semicanonical_orthonormality_tolerance = 0.0;
    HermitianJacobiOptions pno_eigensolver, semicanonical_eigensolver;
};
struct PeriodicGaussianPairPNOLiveInventory {
    // Beyond the reference baseline, real-basis F/indices, provider rows and
    // constant owners below. Original Gaussian bases/localization objects,
    // if the caller still owns them, belong here; no alias inference occurs.
    std::uint64_t other_live_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianPairPNOCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_integral_calls = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianPairPNOPlan {
    std::uint64_t occupied_count = 0, virtual_count = 0;
    std::uint64_t occupied_slot_i = 0, occupied_slot_j = 0;
    std::uint64_t integral_calls = 0, provider_work_units = 0, numerical_work_units = 0, work_units = 0;
    std::uint64_t integral_amplitude_phase_bytes = 0, density_phase_bytes = 0;
    std::uint64_t semicanonical_phase_upper_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t retained_output_upper_bytes = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_provider_row_bytes = 0;
    std::uint64_t state_resident_bytes = 0, fixed_control_storage_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    PeriodicGaussianPairPNOOptions options;
    PeriodicGaussianPairPNOLiveInventory live;
    PeriodicGaussianPairPNOCaps caps;
    std::array<char,64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(),identity_ascii.end()}; }
};
struct PeriodicGaussianPairPNODiagnostics {
    std::uint64_t completed_integral_calls = 0, retained_dimension = 0, retained_output_bytes = 0;
    std::uint64_t actual_semicanonical_phase_bytes = 0;
    double initial_fvv_offdiagonal_norm_upper_bound = 0.0;
    double ignored_occupied_offdiagonal_norm_upper_bound = 0.0;
    double maximum_scalar_integral_roundoff_error = 0.0, integral_projection_error_bound = 0.0;
    double minimum_denominator = 0.0, maximum_denominator = 0.0;
    double maximum_absolute_initial_amplitude = 0.0, maximum_initial_residual = 0.0;
    double initial_residual_frobenius_norm = 0.0;
    double maximum_diagonal_integral_asymmetry = 0.0, maximum_diagonal_amplitude_asymmetry = 0.0;
    double density_trace = 0.0, discarded_occupation_sum = 0.0, minimum_occupation = 0.0;
    std::uint64_t negative_occupation_count = 0, amplitude_scaling_underflow_count = 0;
    std::uint64_t density_underflow_entry_count = 0, semicanonical_fock_scaling_underflow_count = 0;
    double density_eigensystem_relative_residual = 0.0, density_eigenvector_orthogonality_error = 0.0;
    double semicanonical_input_orthonormality_error = 0.0, semicanonical_output_orthonormality_error = 0.0;
    double semicanonical_reduced_relative_residual = 0.0, semicanonical_projected_fock_relative_residual = 0.0;
    double semicanonical_subspace_projector_error = 0.0, semicanonical_full_space_relative_residual = 0.0;
    HermitianJacobiResult pno_eigensolver, semicanonical_eigensolver;
};
class PeriodicGaussianPairPNOResult {
public:
    PeriodicGaussianPairPNOResult(const PeriodicGaussianPairPNOResult&) = delete;
    PeriodicGaussianPairPNOResult& operator=(const PeriodicGaussianPairPNOResult&) = delete;
    PeriodicGaussianPairPNOResult(PeriodicGaussianPairPNOResult&&) noexcept = default;
    PeriodicGaussianPairPNOResult& operator=(PeriodicGaussianPairPNOResult&&) = delete;
    const PeriodicGaussianPairPNOPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianPairPNODiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    PeriodicCorrelationPlacedOccupied occupied_i() const noexcept { return i_; }
    PeriodicCorrelationPlacedOccupied occupied_j() const noexcept { return j_; }
    bool diagonal_pair() const noexcept { return memory_.occupied_slot_i==memory_.occupied_slot_j; }
    // Row-major [n,r], ascending final virtual energies. The original PNO
    // occupations are BEFORE this rotation, NOT columnwise labels for C'.
    const std::vector<double>& coefficients() const;
    const std::vector<double>& energies() const;
    const std::vector<double>& original_pno_occupations() const;
    bool semicanonical_generation_approximation() const noexcept { return true; }
    bool coupled_mp2_solution() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return static_cast<bool>(context_); }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
    const std::string& provider_identity_sha256() const noexcept { return provider_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& exchange_integral_identity_sha256() const noexcept { return integrals_; }
    const std::string& initial_amplitude_identity_sha256() const noexcept { return amplitudes_; }
    const std::string& density_identity_sha256() const noexcept { return density_; }
private:
    PeriodicGaussianPairPNOResult() = default;
    PeriodicGaussianPairPNOPlan memory_;
    PeriodicGaussianPairPNODiagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicCorrelationPlacedOccupied i_,j_;
    std::vector<double> coefficients_,energies_,occupations_;
    std::string identity_,payload_,basis_,provider_,hf_,integrals_,amplitudes_,density_;
    friend PeriodicGaussianPairPNOResult make_periodic_gaussian_pair_pnos(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,std::uint64_t,std::uint64_t,
        const PeriodicGaussianPairPNOOptions&,const PeriodicGaussianPairPNOLiveInventory&,
        const PeriodicGaussianPairPNOCaps&);
};

// Count/owner/control admission before F/G/row scans or numerical allocation.
// Exactly n² scalar integrals; no user callback or four-index tensor. Exact
// peak numerical phases: G+eps+T=16n²+8n; T+eps+PNO=48n²+16n;
// after freeing T/eps/density: C8nr+occupations8n+semicanonical_leaf(n,r),
// bounded above by48n²+16n. Final output8nr+8r+8n; no G/T/D retained.
// Known borrowed real-basis F/labels and provider rows are separate; state
// is already in reference external bytes. Fixed controls/backend allowance
// are not exact allocator/stack/OS-RSS claims. Work is conservative, not time.
PeriodicGaussianPairPNOPlan plan_periodic_gaussian_pair_pnos(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_j,
    const PeriodicGaussianPairPNOOptions&,const PeriodicGaussianPairPNOLiveInventory&,
    const PeriodicGaussianPairPNOCaps&);

// Initial T uses diagonal Fvv only after a measured explicit offdiagonal
// norm gate. Occupied-Fock offdiagonal couplings are ignored for Eq.37 and
// their norm is reported, not claimed to bound MP2 error. Final rotation is
// against ORIGINAL symmetric Fvv, not an unannounced second projection.
// Diagonal G/T must be exactly symmetric; no amplitude repair/rescaling.
// Periodic Eq.39 uses the legacy OffDiagonal density algebra for BOTH pair
// types. Rank zero is valid for positive cutoff. No energy/multiplicity.
PeriodicGaussianPairPNOResult make_periodic_gaussian_pair_pnos(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_j,
    const PeriodicGaussianPairPNOOptions&,const PeriodicGaussianPairPNOLiveInventory&,
    const PeriodicGaussianPairPNOCaps&);

} // namespace vibeqc
