#pragma once

// Actual-source PAO-domain PNO seed. Nejad2025 doi:10.1063/5.0290816
// Eqs.37-40, followed by the original-Fock rotation of Riplinger2013
// doi:10.1063/1.4773581 Eq.25. This source is NOT the legacy common
// real-row provider. No provider-row or common-G replay receipt exists.
#include "vibeqc/periodic_gaussian_density_gram.hpp"
#include "vibeqc/periodic_gaussian_pair_domain_builder.hpp"
#include "vibeqc/periodic_gaussian_embedded_pair_pnos.hpp"

namespace vibeqc {

struct PeriodicGaussianGramPairPNOConfig {
    PeriodicGaussianDensityGramConfig gram;
};
struct PeriodicGaussianGramPairPNOOptions {
    PeriodicCorrelationRealLocalBasisOptions local_basis;
    PeriodicGaussianDensityGramOptions gram;
    // Existing explicit SC-MP2, density, rotation and export controls.
    // Its Fock-symmetry budget additionally gates the native local basis's
    // measured raw-complex to real-symmetric Fock projection.
    PeriodicGaussianEmbeddedPairPNOOptions pno;
    double maximum_occupied_fock_difference = 0.0;
    double maximum_retained_diagonal_projection_norm = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicGaussianGramPairPNOLiveInventory {
    // Beyond shared reference, actual AO/aux, gauges/Wannier, common basis
    // and the COMPLETE borrowed pair-geometry owner. No equal-hash dedup.
    std::uint64_t other_live_numerical_bytes_per_worker = 0;
    std::uint64_t other_live_control_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianGramPairPNOCaps {
    PeriodicCorrelationRealLocalBasisCaps local_basis;
    PeriodicGaussianDensityGramCaps gram;
    // Positive explicit reservation for the actual Gram child's complete
    // fixed/leaf control census; verified against its exact plan before run.
    std::uint64_t maximum_gram_control_storage_bytes = 0;
    std::uint64_t maximum_common_dimension = 0, maximum_generation_dimension = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0, maximum_node_bytes = 0, maximum_work_units = 0;
};
struct PeriodicGaussianGramPairPNOPlan {
    std::uint64_t n_cells = 0, n_basis = 0, occupied_count = 0;
    std::uint64_t common_virtual_dimension = 0, generation_dimension = 0;
    std::uint64_t occupied_slot_i = 0, occupied_slot_j = 0, local_occupied_count = 0;
    PeriodicGaussianDensityGramSelection gram_selection;
    PeriodicCorrelationRealLocalBasisMemoryPlan local_basis;
    // gram_phase includes retained local basis B; the following algebraic
    // phases exclude B. Whole owned=max(local-basis peak,Gram phase,
    // B+max(algebraic phases)). Gram owned/work ceilings occur only once.
    std::uint64_t gram_phase_upper_bytes = 0, copy_phase_bytes = 0, amplitude_phase_bytes = 0;
    std::uint64_t density_phase_bytes = 0, semicanonical_phase_upper_bytes = 0;
    std::uint64_t retained_integral_phase_upper_bytes = 0, export_phase_upper_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, retained_output_upper_bytes = 0;
    std::uint64_t borrowed_common_basis_bytes = 0, borrowed_geometry_bytes = 0;
    std::uint64_t borrowed_wannier_bytes = 0, borrowed_gauge_bytes = 0, borrowed_gaussian_bytes = 0;
    std::uint64_t complete_borrowed_numerical_bytes = 0, fixed_control_storage_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t input_validation_work_units = 0, numerical_work_units = 0;
    std::uint64_t gram_work_units_upper_bound = 0, work_units = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t worker_bytes = 0, required_node_memory_bytes = 0;
};
struct PeriodicGaussianGramPairPNODiagnostics {
    PeriodicCorrelationRealLocalBasisDiagnostics local_basis;
    PeriodicGaussianDensityGramPlan gram_memory;
    PeriodicGaussianDensityGramDiagnostics gram;
    // m-dimensional generation diagnostics. Density normalization is
    // Nejad Eq.39 even for physical diagonal pairs (no 1+delta division).
    PeriodicGaussianPairPNODiagnostics generation;
    std::uint64_t retained_dimension = 0, retained_frame_bytes = 0, retained_output_bytes = 0;
    std::uint64_t retained_generation_coefficient_bytes = 0;
    std::uint64_t actual_semicanonical_phase_bytes = 0, actual_retained_integral_phase_bytes = 0;
    std::uint64_t actual_export_phase_bytes = 0;
    double maximum_raw_exchange_transpose_defect = 0.0;
    double diagonal_exchange_projection_frobenius_upper_bound = 0.0;
    double maximum_raw_retained_transpose_defect = 0.0;
    double retained_diagonal_projection_frobenius_upper_bound = 0.0;
    double occupied_fock_frobenius_upper_bound = 0.0;
    double exported_gram_frobenius_upper_bound = 0.0;
    double exported_fock_frobenius_upper_bound = 0.0;
    double exported_subspace_frobenius_upper_bound = 0.0;
};

struct PeriodicGaussianGramPairPNOPayloadValidationPlan {
    std::uint64_t numerical_lanes = 0, work_units = 0, control_storage_bytes = 0;
};
class PeriodicGaussianGramPairPNOResult;
PeriodicGaussianGramPairPNOPayloadValidationPlan plan_periodic_gaussian_gram_pair_pno_payload_validation(
    const PeriodicGaussianGramPairPNOResult&);
// Caller's work cap is checked before scans. Fixed digest/string controls
// are published separately; no numerical heap or descriptor table is made.
void verify_periodic_gaussian_gram_pair_pno_payload(
    const PeriodicGaussianGramPairPNOResult&, std::uint64_t maximum_work_units);

class PeriodicGaussianGramPairPNOResult {
public:
    PeriodicGaussianGramPairPNOResult(const PeriodicGaussianGramPairPNOResult&) = delete;
    PeriodicGaussianGramPairPNOResult& operator=(const PeriodicGaussianGramPairPNOResult&) = delete;
    PeriodicGaussianGramPairPNOResult(PeriodicGaussianGramPairPNOResult&&) noexcept = default;
    PeriodicGaussianGramPairPNOResult& operator=(PeriodicGaussianGramPairPNOResult&&) = delete;
    const PeriodicGaussianGramPairPNOPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianGramPairPNODiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const PeriodicGaussianGramPairPNOOptions& options() const noexcept { return options_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    PeriodicCorrelationPlacedOccupied occupied_i() const noexcept { return i_; }
    PeriodicCorrelationPlacedOccupied occupied_j() const noexcept { return j_; }
    bool diagonal_pair() const noexcept { return i_.occupied_index==j_.occupied_index && i_.cell==j_.cell; }
    bool complete_generation_pair_space() const;
    bool complete_common_virtual_space() const;
    const std::vector<double>& coefficients() const; // compatibility C[n,r]=X D
    const std::vector<double>& generation_coefficients() const; // authentic D[m,r]
    const std::vector<double>& energies() const;
    const std::vector<double>& original_pno_occupations() const;
    const std::vector<double>& exchange_integrals() const; // D^T G D, no duplicate on transfer
    std::uint64_t retained_numerical_bytes() const;
    std::uint64_t retained_receipt_payload_bytes() const noexcept { return 19U*65U; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; } // COMMON
    const std::string& local_basis_identity_sha256() const noexcept { return local_basis_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& source_context_identity_sha256() const noexcept { return context_sha_; }
    const std::string& geometry_identity_sha256() const noexcept { return geometry_; }
    const std::string& embedding_identity_sha256() const noexcept { return embedding_; }
    const std::string& gram_identity_sha256() const noexcept { return gram_; }
    const std::string& gram_payload_sha256() const noexcept { return gram_payload_; }
    const std::string& gram_consumed_sources_identity_sha256() const noexcept { return gram_sources_; }
    const std::string& raw_exchange_identity_sha256() const noexcept { return raw_g_; }
    const std::string& projected_exchange_identity_sha256() const noexcept { return g_; }
    const std::string& local_fock_identity_sha256() const noexcept { return f_; }
    const std::string& raw_retained_exchange_identity_sha256() const noexcept { return raw_retained_g_; }
    const std::string& retained_exchange_identity_sha256() const noexcept { return retained_g_; }
    const std::string& initial_amplitude_identity_sha256() const noexcept { return amplitudes_; }
    const std::string& density_identity_sha256() const noexcept { return density_; }
    const std::string& source_payload_receipt_sha256() const noexcept { return sources_; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return state_ && context_; }
    bool original_provider_projection_reproduced_bitwise() const noexcept { return false; }
    bool coupled_mp2_solution() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
private:
    PeriodicGaussianGramPairPNOResult() = default;
    void require_live() const;
    PeriodicGaussianGramPairPNOPlan memory_;
    PeriodicGaussianGramPairPNODiagnostics diagnostics_;
    PeriodicGaussianGramPairPNOOptions options_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicCorrelationPlacedOccupied i_,j_;
    std::vector<double> coefficients_,generation_coefficients_,energies_,occupations_,exchange_;
    std::string identity_,payload_,basis_,local_basis_,hf_,context_sha_,geometry_,embedding_;
    std::string gram_,gram_payload_,gram_sources_,raw_g_,g_,f_,raw_retained_g_,retained_g_,amplitudes_,density_,sources_;
    friend PeriodicGaussianGramPairPNOResult make_periodic_gaussian_gram_pair_pnos(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,const BasisSet&,const BasisSet&,
        const PeriodicCorrelationWannier&,const std::complex<double>*,std::size_t,
        const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairDomainGeometry&,
        const PeriodicGaussianGramPairPNOConfig&,const PeriodicGaussianGramPairPNOOptions&,
        const PeriodicGaussianGramPairPNOLiveInventory&,const PeriodicGaussianGramPairPNOCaps&);
};

// Metadata/count-only whole-owner admission. No original payload scan,
// local-basis/Gram construction, reference relabelling or numerical heap.
// q=1 for equal PHYSICAL labels, otherwise2; slots come from the genuine
// geometry and original common basis. All scientific controls are explicit.
PeriodicGaussianGramPairPNOPlan plan_periodic_gaussian_gram_pair_pnos(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairDomainGeometry&,const PeriodicGaussianGramPairPNOConfig&,
    const PeriodicGaussianGramPairPNOOptions&,const PeriodicGaussianGramPairPNOLiveInventory&,
    const PeriodicGaussianGramPairPNOCaps&);
// Callback-free. Original inputs remain alive and immutable through return.
// The owner retains only C,D,eps,occupations,G_PNO plus immutable source
// handles/receipts. Original pair geometry remains borrowed and unconsumed.
// Numeric algebra phases (excluding the still-live local basis): raw Gram
// copy16m^2; initial16m^2+8m; density56m^2+8m; local rotation upper
// 56m^2+16m; retained-G upper32m^2+40m. Final retained output is exactly
// 8(nr+mr+r+m+r^2); all m original occupations survive, including rank0.
// No raw Gram/G/T/density workspace or old common-factor rows survive.
// Raw/projected/local/common diagnostics are measured separately and are
// not a propagated PNO/correlation-energy error certificate.
PeriodicGaussianGramPairPNOResult make_periodic_gaussian_gram_pair_pnos(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,const BasisSet& ao,const BasisSet& auxiliary,
    const PeriodicCorrelationWannier&,const std::complex<double>* gauges,std::size_t gauge_count,
    const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairDomainGeometry&,
    const PeriodicGaussianGramPairPNOConfig&,const PeriodicGaussianGramPairPNOOptions&,
    const PeriodicGaussianGramPairPNOLiveInventory&,const PeriodicGaussianGramPairPNOCaps&);

} // namespace vibeqc
