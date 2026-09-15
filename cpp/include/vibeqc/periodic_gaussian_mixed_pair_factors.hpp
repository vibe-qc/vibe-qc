#pragma once

// Actual finite-Gaussian mixed-frame density factors. Riplinger
// doi:10.1063/1.4773581 Sec.II.C Eq.26; Sun doi:10.1063/1.4998644
// Eqs.3,4,13,16,21. Local columns are [occupied i, occupied k, PNO A, PNO B].
// Each PNO frame is genuine, but their concatenation is NOT an orthonormal
// orbital basis. All ordered complex density factors are independently
// formed with Nk^-3/2 normalization; no reversal/qbar/real projection here.

#include "vibeqc/periodic_gaussian_local_orbital_factors.hpp"
#include "vibeqc/periodic_gaussian_pair_pno_frame.hpp"

namespace vibeqc {

// Optional physical generation geometry. This is a non-owning INPUT, not
// a certificate: the factor factory authenticates all three owners against
// the embedded PNO and admits their complete live payloads before scans.
// All owners must stay alive, at their addresses and unmoved during use.
// No caller coefficient arrays and no reconstruction D = X^T C are accepted.
class PeriodicGaussianPairPNOGeometryView {
public:
    PeriodicGaussianPairPNOGeometryView(const PeriodicCorrelationPAODomain& domain,
        const PeriodicCorrelationRealPAOSpace& space,const PeriodicCorrelationRealPAOEmbedding& embedding)
        : domain_(&domain),space_(&space),embedding_(&embedding) {}
    PeriodicGaussianPairPNOGeometryView(const PeriodicCorrelationPAODomain&&,
        const PeriodicCorrelationRealPAOSpace&,const PeriodicCorrelationRealPAOEmbedding&) = delete;
    PeriodicGaussianPairPNOGeometryView(const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationRealPAOSpace&&,const PeriodicCorrelationRealPAOEmbedding&) = delete;
    PeriodicGaussianPairPNOGeometryView(const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationRealPAOSpace&,const PeriodicCorrelationRealPAOEmbedding&&) = delete;
    const PeriodicCorrelationPAODomain& domain() const { return *domain_; }
    const PeriodicCorrelationRealPAOSpace& space() const { return *space_; }
    const PeriodicCorrelationRealPAOEmbedding& embedding() const { return *embedding_; }
private:
    const PeriodicCorrelationPAODomain* domain_;
    const PeriodicCorrelationRealPAOSpace* space_;
    const PeriodicCorrelationRealPAOEmbedding* embedding_;
};

struct PeriodicGaussianMixedPairFactorStorage {
    std::uint64_t orbital_count = 0, pno_column_count = 0;
    std::uint64_t tile_calls = 0, contraction_term_count = 0;
    std::uint64_t retained_factor_bytes = 0, retained_output_bytes = 0;
    std::uint64_t compensation_bytes = 0, coefficient_panel_bytes = 0;
    std::uint64_t coefficient_helper_workspace_bytes = 0;
    std::uint64_t driver_owned_numerical_bytes = 0;
    std::uint64_t coefficient_work_units_per_kpoint = 0, coefficient_work_units = 0;
};
// Pure count-only inventory, not source authentication. All dimensions are
// positive except the independently permitted zero PNO ranks. n_common is
// the retained common virtual dimension and d_common the original PAO-domain
// dimension. No source/W construction, owners, hashes or payload scans.
// Helper work bounds one kpoint's complete [i,k,A,B] coefficient fill.
PeriodicGaussianMixedPairFactorStorage periodic_gaussian_mixed_pair_factor_storage(
    std::uint64_t n_cells,std::uint64_t n_basis,std::uint64_t n_effective_orbitals,
    std::uint64_t n_common,std::uint64_t d_common,
    std::uint64_t rank_a,std::uint64_t rank_b,
    std::uint64_t auxiliary_count,std::uint64_t ao_pair_block);

struct PeriodicGaussianMixedPairFactorPlan : PeriodicGaussianMixedPairFactorStorage {
    std::uint64_t n_cells = 0, n_basis = 0, n_auxiliary = 0;
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, common_domain_dimension = 0;
    std::uint64_t rank_a = 0, rank_b = 0, occupied_slot_i = 0, occupied_slot_k = 0;
    std::uint64_t q_index = 0, auxiliary_begin = 0, auxiliary_count = 0, ao_pair_block = 0;
    bool same_frame_owner = false;
    // Direct Gram generation requires authenticated original geometry on
    // both sides. It does not have common-factor provider receipts.
    bool direct_gram_frames = false;
    std::uint64_t borrowed_frame_numerical_bytes = 0, borrowed_frame_control_bytes = 0;
    std::uint64_t frame_validation_work_units = 0;
    bool local_geometry_a = false, local_geometry_b = false;
    std::uint64_t borrowed_geometry_numerical_bytes = 0, borrowed_geometry_control_bytes = 0;
    std::uint64_t geometry_validation_work_units = 0;
    std::uint64_t maximum_tile_owned_numerical_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t resident_whitener_bytes = 0, borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0, live_wannier_bytes = 0;
    std::uint64_t live_domain_bytes = 0, live_space_bytes = 0, live_basis_bytes = 0;
    std::uint64_t state_resident_bytes = 0, macro_fixed_object_bytes = 0, tile_fixed_object_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t reciprocal_candidate_evaluations = 0, image_candidate_evaluations_upper_bound = 0;
    std::uint64_t driver_work_units = 0, work_units = 0;
    PeriodicGaussianLocalOrbitalFactorConfig config;
    PeriodicGaussianLocalOrbitalFactorLiveInventory live;
    PeriodicGaussianLocalOrbitalFactorCaps caps;
    PeriodicGaussianThreeCenterConfig tile_config;
    std::array<char,64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(),identity_ascii.end()}; }
};
using PeriodicGaussianMixedPairFactorDiagnostics = PeriodicGaussianLocalOrbitalFactorDiagnostics;

class PeriodicGaussianMixedPairFactorPanel {
public:
    PeriodicGaussianMixedPairFactorPanel(const PeriodicGaussianMixedPairFactorPanel&) = delete;
    PeriodicGaussianMixedPairFactorPanel& operator=(const PeriodicGaussianMixedPairFactorPanel&) = delete;
    PeriodicGaussianMixedPairFactorPanel(PeriodicGaussianMixedPairFactorPanel&&) noexcept = default;
    PeriodicGaussianMixedPairFactorPanel& operator=(PeriodicGaussianMixedPairFactorPanel&&) = delete;
    const auto& context_handle() const noexcept { return context_; }
    const auto& state_handle() const noexcept { return state_; }
    const PeriodicGaussianMixedPairFactorPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianMixedPairFactorDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    std::uint64_t q_index() const noexcept { return memory_.q_index; }
    std::uint64_t auxiliary_begin() const noexcept { return memory_.auxiliary_begin; }
    std::uint64_t auxiliary_count() const noexcept { return memory_.auxiliary_count; }
    std::uint64_t orbital_count() const noexcept { return memory_.orbital_count; }
    std::uint64_t rank_a() const noexcept { return memory_.rank_a; }
    std::uint64_t rank_b() const noexcept { return memory_.rank_b; }
    PeriodicCorrelationPlacedOccupied occupied_i() const noexcept { return i_; }
    PeriodicCorrelationPlacedOccupied occupied_k() const noexcept { return k_; }
    const std::complex<double>* data() const;
    std::complex<double> element(std::size_t auxiliary,std::size_t left,std::size_t right) const;
    bool finite_image_reference() const noexcept { return true; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return true; }
    bool jointly_orthonormal_basis_certified() const noexcept { return false; }
    bool density_symmetry_certified() const noexcept { return false; }
    bool bitwise_origin_provider_factors_verified() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& local_basis_identity_sha256() const noexcept { return basis_; }
    const std::string& basis_certificate_identity_sha256() const noexcept { return certificate_; }
    const std::string& source_identity_sha256() const noexcept { return source_; }
    const std::string& conjugate_source_identity_sha256() const noexcept { return opposite_; }
    const std::string& whitener_payload_identity_sha256() const noexcept { return whitener_; }
    const std::string& consumed_tiles_identity_sha256() const noexcept { return consumed_; }
    const std::string& frame_a_identity_sha256() const noexcept { return frame_a_; }
    const std::string& frame_b_identity_sha256() const noexcept { return frame_b_; }
    const std::string& frame_a_payload_sha256() const noexcept { return frame_a_payload_; }
    const std::string& frame_b_payload_sha256() const noexcept { return frame_b_payload_; }
    // Immutable generation lineage, NOT a retained/provider callback or a
    // claim that the present factors equal its projected real rows bitwise.
    bool direct_gram_frames() const noexcept { return memory_.direct_gram_frames; }
    const std::string& frame_a_provider_identity_sha256() const;
    const std::string& frame_b_provider_identity_sha256() const;
    const std::string& frame_a_gram_identity_sha256() const;
    const std::string& frame_b_gram_identity_sha256() const;
private:
    PeriodicGaussianMixedPairFactorPanel() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicGaussianMixedPairFactorPlan memory_;
    PeriodicGaussianMixedPairFactorDiagnostics diagnostics_;
    PeriodicCorrelationPlacedOccupied i_,k_;
    std::vector<std::complex<double>> values_;
    std::string identity_,payload_,hf_,basis_,certificate_,source_,opposite_,whitener_,consumed_;
    std::string frame_a_,frame_b_,frame_a_payload_,frame_b_payload_,provider_a_,provider_b_;
    std::string gram_a_,gram_b_;
    friend PeriodicGaussianMixedPairFactorPanel build_periodic_gaussian_mixed_pair_factor_panel(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
        const PeriodicGaussianReciprocalSource&,const PeriodicGaussianMetricWhitener&,
        const BasisSet&,const BasisSet&,const PeriodicCorrelationWannier&,
        const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOFrameView&,
        std::uint64_t,std::uint64_t,std::uint64_t,std::uint64_t,
        const PeriodicGaussianLocalOrbitalFactorConfig&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
        const PeriodicGaussianLocalOrbitalFactorCaps&,
        const PeriodicGaussianPairPNOGeometryView*,const PeriodicGaussianPairPNOGeometryView*);
};

// O(1) source/owner/dimension census and complete caps precede numerical
// scans/allocation. Live owners include common geometry/Fock/gauges and both
// complete PNO payloads; exact same frame-owner address is counted once.
// A parent PairSpace/MP2/CCSD owner may retain additional data not reachable
// from these views; the caller MUST declare that remainder in live.
// No all-common factor rows or provider owner is accepted or required here.
//
// R=rA+rB, M=2+R: output16*Ab*M², compensation16*Ab*M², two-kpoint
// coefficient panels32*nAO*M, helper workspace16*nAO*(R+3) for R>0
// (zero for R=0). All arrays coexist with at most one native tile whose
// explicitly capped peak is additional. Borrowed W16*A² is counted once;
// qbar W/previous panel and other external owners belong in live once.
// Rank0 keeps the two occupied columns, without forcing a PNO or dropping
// source validation. No real cast, orthogonalization or normalization repair.
// DirectPAOGram pairs may instead supply both authentic original geometries
// and use their retained PAO-to-PNO rotations without a common-factor store.
// Mixing that source kind with an older frame remains unqualified. The
// common auxiliary metric is unchanged; this is not a production scaling claim.
PeriodicGaussianMixedPairFactorPlan plan_periodic_gaussian_mixed_pair_factor_panel(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianReciprocalSource&,const PeriodicGaussianMetricWhitener&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOFrameView&,
    std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_k,
    std::uint64_t auxiliary_begin,std::uint64_t auxiliary_count,
    const PeriodicGaussianLocalOrbitalFactorConfig&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
    const PeriodicGaussianLocalOrbitalFactorCaps&,
    const PeriodicGaussianPairPNOGeometryView* geometry_a = nullptr,
    const PeriodicGaussianPairPNOGeometryView* geometry_b = nullptr);
PeriodicGaussianMixedPairFactorPanel build_periodic_gaussian_mixed_pair_factor_panel(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianReciprocalSource&,const PeriodicGaussianMetricWhitener&,
    const BasisSet&,const BasisSet&,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOFrameView&,
    std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_k,
    std::uint64_t auxiliary_begin,std::uint64_t auxiliary_count,
    const PeriodicGaussianLocalOrbitalFactorConfig&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
    const PeriodicGaussianLocalOrbitalFactorCaps&,
    const PeriodicGaussianPairPNOGeometryView* geometry_a = nullptr,
    const PeriodicGaussianPairPNOGeometryView* geometry_b = nullptr);

} // namespace vibeqc
