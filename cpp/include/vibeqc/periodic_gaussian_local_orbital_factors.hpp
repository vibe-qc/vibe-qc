#pragma once

// Actual Gaussian-HF-origin selected-local factors on the finite BvK torus.
// Sun doi:10.1063/1.4998644 Eqs.3,4,13,16,21; Nejad
// doi:10.1063/5.0290816 Eqs.2-10,26-28:
// L_lr(q,P)=Nk^(-3/2) sum_k,mu,nu conj(C_l(mu,k))*C_r(nu,k+q)*B_P(mu,nu;k,q).
// The full wrapper computes every ordered OO/OV/VO/VV density; the block
// overload computes exact rectangular subranges of that same certified
// common basis. No real projection, q-conjugacy shortcut, full AO store or
// all-k coefficient cache exists.

#include "vibeqc/periodic_gaussian_rhf.hpp"
#include "vibeqc/periodic_correlation_real_local_basis.hpp"

namespace vibeqc {

struct PeriodicGaussianLocalOrbitalFactorSelection {
    // Common orbital order is occupied first, followed by the certified
    // virtual selection. Both counts must be positive; ranges may cross
    // that boundary. These are indices, never caller-supplied orbitals.
    std::uint64_t left_begin = 0, left_count = 0;
    std::uint64_t right_begin = 0, right_count = 0;
};
struct PeriodicGaussianLocalOrbitalFactorConfig {
    std::uint64_t ao_pair_block = 0;
};
struct PeriodicGaussianLocalOrbitalFactorLiveInventory {
    // Extra numerical owners not already in the admitted reference or the
    // explicitly counted gauge/Wannier/domain/space/real-basis/source/W.
    // A provider's retained q-partner W/panel and real-row output go here.
    std::uint64_t other_retained_bytes_per_worker = 0;
    std::uint64_t other_transient_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianLocalOrbitalFactorCaps {
    PeriodicGaussianMetricCaps resources;
    PeriodicGaussianThreeCenterCaps tile;
    std::uint64_t maximum_tile_calls = 0;
    std::uint64_t maximum_image_candidate_evaluations = 0;
};
struct PeriodicGaussianLocalOrbitalFactorPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_auxiliary = 0;
    std::uint64_t occupied_count = 0, virtual_count = 0, orbital_count = 0;
    std::uint64_t q_index = 0, auxiliary_begin = 0, auxiliary_count = 0;
    std::uint64_t ao_pair_block = 0, tile_calls = 0;
    std::uint64_t retained_factor_bytes = 0, retained_index_bytes = 0, retained_output_bytes = 0;
    std::uint64_t compensation_bytes = 0, coefficient_panel_bytes = 0, coefficient_scratch_bytes = 0;
    std::uint64_t driver_owned_numerical_bytes = 0, maximum_tile_owned_numerical_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, resident_whitener_bytes = 0;
    std::uint64_t borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0, live_wannier_bytes = 0;
    std::uint64_t live_domain_bytes = 0, live_space_bytes = 0, live_basis_bytes = 0;
    // Occupied-index view aliases the explicitly counted real-basis owner;
    // informational only, never added to the inventory a second time.
    std::uint64_t basis_index_alias_bytes = 0, state_resident_bytes = 0;
    std::uint64_t macro_fixed_object_bytes = 0, tile_fixed_object_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t reciprocal_candidate_evaluations = 0;
    std::uint64_t image_candidate_evaluations_upper_bound = 0;
    std::uint64_t coefficient_work_units = 0, contraction_term_count = 0;
    std::uint64_t driver_work_units = 0, work_units = 0;
    PeriodicGaussianLocalOrbitalFactorSelection selection;
    PeriodicGaussianLocalOrbitalFactorConfig config;
    PeriodicGaussianLocalOrbitalFactorLiveInventory live;
    PeriodicGaussianLocalOrbitalFactorCaps caps;
    // Derived from actual HF, never substituted by caller scientific labels.
    PeriodicGaussianThreeCenterConfig tile_config;
    std::array<char,64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(),identity_ascii.end()}; }
};
struct PeriodicGaussianLocalOrbitalFactorDiagnostics {
    std::uint64_t completed_tile_calls = 0, reciprocal_candidate_evaluations = 0;
    std::uint64_t image_candidate_evaluations = 0, charged_work_units_upper_bound = 0;
    std::uint64_t maximum_observed_owned_numerical_bytes = 0;
    std::uint64_t maximum_observed_per_replica_inventoried_bytes = 0;
};
class PeriodicGaussianLocalOrbitalFactorPanel {
public:
    PeriodicGaussianLocalOrbitalFactorPanel(const PeriodicGaussianLocalOrbitalFactorPanel&) = delete;
    PeriodicGaussianLocalOrbitalFactorPanel& operator=(const PeriodicGaussianLocalOrbitalFactorPanel&) = delete;
    PeriodicGaussianLocalOrbitalFactorPanel(PeriodicGaussianLocalOrbitalFactorPanel&&) noexcept = default;
    PeriodicGaussianLocalOrbitalFactorPanel& operator=(PeriodicGaussianLocalOrbitalFactorPanel&&) = delete;
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicGaussianLocalOrbitalFactorPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianLocalOrbitalFactorDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    std::uint64_t q_index() const noexcept { return memory_.q_index; }
    std::uint64_t auxiliary_begin() const noexcept { return memory_.auxiliary_begin; }
    std::uint64_t auxiliary_count() const noexcept { return memory_.auxiliary_count; }
    // Global common-basis dimension, not either local rectangular extent.
    std::uint64_t orbital_count() const noexcept { return memory_.orbital_count; }
    PeriodicGaussianLocalOrbitalFactorSelection selection() const noexcept { return memory_.selection; }
    PeriodicCorrelationPlacedOccupied occupied(std::size_t) const;
    PeriodicCorrelationVirtualBlockSelection virtual_selection() const noexcept { return virtual_; }
    const std::complex<double>* data() const;
    // Local rectangular indices: global columns are left_begin+left and
    // right_begin+right. Row-major storage is [auxiliary,left_count,right_count].
    std::complex<double> element(std::size_t auxiliary,std::size_t left,std::size_t right) const;
    bool finite_image_reference() const noexcept { return true; }
    bool ao_image_source_certified() const noexcept { return false; }
    bool density_symmetry_certified() const noexcept { return false; }
    // The genuine HF factory and exact native source recipe are matched.
    // Its aggregate final-Fock trace does not expose per-q W/tile hashes;
    // this is NOT a comparison to each bitwise final-HF consumed factor.
    bool matched_finite_gaussian_hf_recipe() const noexcept { return true; }
    bool bitwise_hf_factor_consumption_verified() const noexcept { return false; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& local_basis_identity_sha256() const noexcept { return basis_; }
    const std::string& basis_certificate_identity_sha256() const noexcept { return certificate_; }
    const std::string& source_identity_sha256() const noexcept { return source_; }
    const std::string& conjugate_source_identity_sha256() const noexcept { return opposite_; }
    const std::string& whitener_payload_identity_sha256() const noexcept { return whitener_; }
    const std::string& consumed_tiles_identity_sha256() const noexcept { return consumed_; }
private:
    PeriodicGaussianLocalOrbitalFactorPanel() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicGaussianLocalOrbitalFactorPlan memory_;
    PeriodicGaussianLocalOrbitalFactorDiagnostics diagnostics_;
    PeriodicCorrelationVirtualBlockSelection virtual_;
    std::vector<std::uint64_t> indices_;
    std::vector<std::complex<double>> values_;
    std::string hf_,identity_,payload_,basis_,certificate_,source_,opposite_,whitener_,consumed_;
    friend PeriodicGaussianLocalOrbitalFactorPanel build_periodic_gaussian_local_orbital_factor_panel(
        const PeriodicGaussianRHFResult&, const PeriodicCorrelationAdmittedReference&,
        const PeriodicGaussianReciprocalSource&, const PeriodicGaussianMetricWhitener&,
        const BasisSet&, const BasisSet&, const PeriodicCorrelationWannier&,
        const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
        std::uint64_t,std::uint64_t,const PeriodicGaussianLocalOrbitalFactorSelection&,
        const PeriodicGaussianLocalOrbitalFactorConfig&,
        const PeriodicGaussianLocalOrbitalFactorLiveInventory&,const PeriodicGaussianLocalOrbitalFactorCaps&);
};

// O(1) structural/source/recipe/owner admission precedes any gauge/index/
// basis scan or numerical allocation. Requires exact HF/reference/basis
// state identity and exact HF/source/W context pointer identity. Source/W
// remain externally owned, so a provider can reuse them across aux slices.
// No source factory, metric or W build is performed or charged here.
//
// With o occupied,v virtual,m=o+v,nAO=n,A original auxiliary,Ab selected
// auxiliary rows, L/R selected common-orbital columns and Pb AO-pair block,
// the driver owns 32*Ab*L*R +16*n*(L+R) +32*n +16*o bytes:
// output+compensation, two per-k coefficient panels, two reused AO scratch
// columns, retained indices. The full wrapper uses L=R=m, exactly the
// existing 32*Ab*m²+32*n*m+32*n+16*o inventory and contraction order.
// Peak adds the explicit tile-owned ceiling; resident W16*A² is counted
// ONCE as borrowed. Output retains only16*Ab*L*R+16*o. No all-k coefficients.
// Known live gauges/Wannier/domain/space/real-basis and both active bases
// are separately counted. Selected occupied indices alias real-basis storage.
// The reference's external inventory ALREADY includes the exact HF state;
// no second state payload is charged through HF/reference/basis shared owners.
//
// Node=reference external+shared+ranks*(per_rank+localization_window)
//      +ranks*workers*(owned_peak+W+known_live+fixed_objects+explicit_live).
// The new caps AND the old admitted-reference memory budget must both cover
// this value. Fixed logical objects and a mandatory positive backend margin
// are not exact stack/allocator/OS RSS claims. A caller-owned qbar W/panel or
// row-provider allocation is additionally declared in live exactly once.
//
// Tile calls=Nk*ceil(n²/Pb), source replay candidates=2*C*tile_calls.
// Images conservatively <=tile_calls*Icap*(1+ceil(N/g)+N), from the native
// selected-tile admission. Work preflight sums explicit native tile ceilings
// plus all coefficient, contraction, gauge/basis/hash scans. Returned tile
// plans charge actual counts before the next tile. Loose ceilings may reject
// affordable work; counts are not wall-clock or FLOP predictions.
//
// A rectangle alone does not certify density reversal or time reversal:
// a consumer must build the swapped rectangle and actual qbar source as
// needed before any real-row projection. This primitive does not change
// the existing provider's retained full factor rows or auxiliary space.
// Full-range plan/payload/identity wire schemas are unchanged; only nonfull
// blocks append explicitly tagged common-orbital range metadata. Logical
// object inventory reflects the current sizeof of these public structures.
PeriodicGaussianLocalOrbitalFactorPlan plan_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianReciprocalSource&,const PeriodicGaussianMetricWhitener&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    std::uint64_t auxiliary_begin,std::uint64_t auxiliary_count,
    const PeriodicGaussianLocalOrbitalFactorConfig&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
    const PeriodicGaussianLocalOrbitalFactorCaps&);
PeriodicGaussianLocalOrbitalFactorPlan plan_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianReciprocalSource&,const PeriodicGaussianMetricWhitener&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    std::uint64_t auxiliary_begin,std::uint64_t auxiliary_count,
    const PeriodicGaussianLocalOrbitalFactorSelection&,
    const PeriodicGaussianLocalOrbitalFactorConfig&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
    const PeriodicGaussianLocalOrbitalFactorCaps&);
PeriodicGaussianLocalOrbitalFactorPanel build_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianReciprocalSource&,const PeriodicGaussianMetricWhitener&,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    std::uint64_t auxiliary_begin,std::uint64_t auxiliary_count,
    const PeriodicGaussianLocalOrbitalFactorConfig&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
    const PeriodicGaussianLocalOrbitalFactorCaps&);
PeriodicGaussianLocalOrbitalFactorPanel build_periodic_gaussian_local_orbital_factor_panel(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicGaussianReciprocalSource&,const PeriodicGaussianMetricWhitener&,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    std::uint64_t auxiliary_begin,std::uint64_t auxiliary_count,
    const PeriodicGaussianLocalOrbitalFactorSelection&,
    const PeriodicGaussianLocalOrbitalFactorConfig&,const PeriodicGaussianLocalOrbitalFactorLiveInventory&,
    const PeriodicGaussianLocalOrbitalFactorCaps&);

} // namespace vibeqc
