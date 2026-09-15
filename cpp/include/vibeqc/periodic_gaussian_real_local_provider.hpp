#pragma once

// Actual Gaussian-HF-origin bounded real-row provider. The finite-source
// q/qbar conversion is shared with the store-backed reference and retains
// both complex lanes; no AO store, all-q complex tensor or ERI cache exists.
// Sun doi:10.1063/1.4998644 Eqs.13,16,21 and Nejad
// doi:10.1063/5.0290816 Eqs.2-10 fix the inherited finite-torus convention.

#include "vibeqc/periodic_correlation_real_local_provider.hpp"
#include "vibeqc/periodic_gaussian_local_orbital_factors.hpp"

namespace vibeqc {

struct PeriodicGaussianRealLocalProviderConfig {
    std::uint64_t auxiliary_block = 0;
    PeriodicGaussianLocalOrbitalFactorConfig panel;
};
struct PeriodicGaussianRealLocalProviderLiveInventory {
    // Beyond the exact admitted reference and explicitly counted AO/aux,
    // gauges/Wannier/domain/space/real-basis owners. No deduplication of
    // unrelated producers' caller-live fields is inferred.
    std::uint64_t other_retained_bytes_per_worker = 0;
    std::uint64_t other_transient_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianRealLocalProviderCaps {
    PeriodicGaussianMetricCaps resources;
    PeriodicGaussianMetricCaps metric;
    PeriodicGaussianLocalOrbitalFactorCaps panel;
    std::uint64_t maximum_factor_panels = 0;
    std::uint64_t maximum_tile_calls = 0;
    std::uint64_t maximum_image_candidate_evaluations = 0;
    std::uint64_t maximum_progress_callbacks = 0;
    std::uint64_t maximum_scalar_work_units = 0;
};
struct PeriodicGaussianRealLocalProviderPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_auxiliary = 0;
    std::uint64_t occupied_count = 0, virtual_count = 0, orbital_count = 0;
    std::uint64_t density_count = 0, row_count = 0, self_inverse_q_count = 0;
    std::uint64_t auxiliary_block_count = 0, factor_panels = 0, tile_calls = 0;
    std::uint64_t retained_row_bytes = 0, norm_workspace_bytes = 0;
    std::uint64_t whitener_bytes = 0, maximum_live_whitener_bytes = 0;
    std::uint64_t retained_partner_panel_bytes = 0, panel_driver_owned_bytes = 0;
    std::uint64_t metric_phase_owned_upper_bound = 0, panel_phase_owned_upper_bound = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t borrowed_basis_active_numeric_bytes = 0, caller_gauge_bytes = 0;
    std::uint64_t live_wannier_bytes = 0, live_domain_bytes = 0, live_space_bytes = 0;
    std::uint64_t live_basis_bytes = 0, basis_index_alias_bytes = 0;
    std::uint64_t macro_fixed_object_bytes = 0, maximum_leaf_fixed_object_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t source_factory_calls = 0, metric_calls = 0, whitening_calls = 0;
    std::uint64_t reciprocal_candidate_evaluations_upper_bound = 0;
    std::uint64_t image_candidate_evaluations_upper_bound = 0;
    std::uint64_t progress_callback_upper_bound = 0, input_check_work_units = 0;
    std::uint64_t driver_work_units = 0, work_units = 0, scalar_work_units = 0;
    PeriodicGaussianRealLocalProviderConfig config;
    PeriodicGaussianRealLocalProviderLiveInventory live;
    PeriodicGaussianRealLocalProviderCaps caps;
    std::array<char,64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(),identity_ascii.end()}; }
};
enum class PeriodicGaussianRealLocalProviderStage : std::uint32_t {
    Begin = 0, Source = 1, Metric = 2, Whitening = 3,
    Panel = 4, PairComplete = 5, Complete = 6
};
struct PeriodicGaussianRealLocalProviderProgress {
    PeriodicGaussianRealLocalProviderStage stage = PeriodicGaussianRealLocalProviderStage::Begin;
    std::uint64_t q_index = 0, auxiliary_begin = 0;
    std::uint64_t completed_source_count = 0, completed_factor_panels = 0;
    std::uint64_t completed_tile_calls = 0, charged_work_units_upper_bound = 0;
};
using PeriodicGaussianRealLocalProviderCallback = void (*)(const PeriodicGaussianRealLocalProviderProgress&, void*);
struct PeriodicGaussianRealLocalProviderReceipt {
    std::uint64_t completed_source_count = 0, completed_metric_count = 0, completed_whitening_count = 0;
    std::uint64_t completed_factor_panels = 0, completed_tile_calls = 0;
    std::uint64_t progress_callback_count = 0, source_factory_candidate_evaluations = 0;
    std::uint64_t reciprocal_candidate_evaluations = 0, image_candidate_evaluations = 0;
    std::uint64_t charged_work_units_upper_bound = 0;
    std::uint64_t maximum_observed_owned_numerical_bytes = 0;
    std::uint64_t maximum_observed_per_replica_inventoried_bytes = 0;
};

/// The numerical core owns only real rows and the immutable reference state.
/// This enclosing owner adds actual Gaussian source/recipe/work receipts,
/// but no numerical row copy and no retained HF/Wannier/domain/panel owner.
/// provider() borrows this owner's member; it cannot outlive the wrapper.
class PeriodicGaussianRealLocalProvider {
public:
    PeriodicGaussianRealLocalProvider(const PeriodicGaussianRealLocalProvider&) = delete;
    PeriodicGaussianRealLocalProvider& operator=(const PeriodicGaussianRealLocalProvider&) = delete;
    PeriodicGaussianRealLocalProvider(PeriodicGaussianRealLocalProvider&&) noexcept = default;
    PeriodicGaussianRealLocalProvider& operator=(PeriodicGaussianRealLocalProvider&&) = delete;
    const PeriodicCorrelationRealLocalProvider& provider() const;
    const PeriodicGaussianRealLocalProviderPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianRealLocalProviderReceipt& receipt() const noexcept { return receipt_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const std::string& source_context_identity_sha256() const noexcept { return context_sha_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& consumed_sources_identity_sha256() const noexcept { return sources_; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return static_cast<bool>(context_); }
    bool bitwise_hf_factor_consumption_verified() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
private:
    explicit PeriodicGaussianRealLocalProvider(PeriodicCorrelationRealLocalProvider&&);
    PeriodicCorrelationRealLocalProvider provider_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianRealLocalProviderPlan memory_;
    PeriodicGaussianRealLocalProviderReceipt receipt_;
    std::string context_sha_,hf_,identity_,sources_;
    friend PeriodicGaussianRealLocalProvider make_periodic_gaussian_real_local_provider(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
        const BasisSet&,const BasisSet&,const PeriodicCorrelationWannier&,
        const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProviderConfig&,const PeriodicCorrelationRealLocalProviderOptions&,
        const PeriodicGaussianRealLocalProviderLiveInventory&,const PeriodicGaussianRealLocalProviderCaps&,
        PeriodicGaussianRealLocalProviderCallback,void*);
};

// With K cells,A auxiliary,n AO,m=o+v and h=m(m+1)/2, retained rows R=8KAh,
// norm workspace N=24h, W=16A², partner panel P=16Ab*m²+16o and building
// driver D=32Ab*m²+32nm+32n+16o. A nonself pair has at most two W owners.
// Metric phase <=R+N+(nonself?W:0)+metric-owned-cap; panel phase
// <=R+N+(nonself?2W:W)+(nonself?P:0)+D+tile-owned-cap.
// The exact variable formulas and declared leaf ceilings are distinct.
// All known borrowed local/basis payloads, fixed logical controls, and live
// extras are additionally admitted. Exact reference/state baseline is once.
// These bounds are not stack/allocator/OS RSS or source-accuracy certificates.
//
// Plan performs structural checks only, with conservative source/metric/
// panel work ceilings before any numerical output, gauge or basis scan.
// Every actual leaf independently re-admits its true live owners. Scientific
// source/metric/tile controls derive from the immutable HF recipe; only
// explicit block/resource choices here may vary. No reconstructed cell A.
PeriodicGaussianRealLocalProviderPlan plan_periodic_gaussian_real_local_provider(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProviderConfig&,const PeriodicGaussianRealLocalProviderLiveInventory&,
    const PeriodicGaussianRealLocalProviderCaps&);

// Inputs remain immutable through this synchronous call, including progress.
// Progress throws to cancel; callback-boundary gauge/basis rechecks are in
// the admitted work bound. Callers declare callback storage in live. No
// provider or success receipt survives any failed native source/audit/call.
// The enclosing physical-system factory must additionally invoke the HF
// owner's original-system content recheck before constructing local spaces.
PeriodicGaussianRealLocalProvider make_periodic_gaussian_real_local_provider(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProviderConfig&,const PeriodicCorrelationRealLocalProviderOptions&,
    const PeriodicGaussianRealLocalProviderLiveInventory&,const PeriodicGaussianRealLocalProviderCaps&,
    PeriodicGaussianRealLocalProviderCallback progress = nullptr,void* progress_context = nullptr);

} // namespace vibeqc
