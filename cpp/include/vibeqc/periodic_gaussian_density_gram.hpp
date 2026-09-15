#pragma once

// Selected common-density Gram blocks from the actual finite Gaussian HF
// source. Sun doi:10.1063/1.4998644 Eqs.3,13,16,21 and Nejad
// doi:10.1063/5.0290816 Eqs.2-10 fix the inherited BvK normalization.
// No PNO owner, prior real-row provider, caller factor array, full AO store,
// all-q row store, energy driver or scalar-integral adapter is accepted.
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "vibeqc/periodic_gaussian_local_orbital_factors.hpp"
#include "vibeqc/periodic_correlation_real_local_provider.hpp"

namespace vibeqc {

struct PeriodicGaussianDensityRange {
    // Packed upper common-density order: x=p*m-p*(p+1)/2+q, p<=q,
    // m=occupied_count+virtual_count. Common orbitals are occupied first.
    std::uint64_t begin = 0, count = 0;
};
struct PeriodicGaussianDensityGramSelection {
    // Both ranges are nonempty and within m*(m+1)/2. They may overlap;
    // left and right are explicitly inventoried roles, not deduplicated.
    PeriodicGaussianDensityRange left, right;
};
struct PeriodicGaussianDensityGramConfig {
    std::uint64_t auxiliary_block = 0;
    PeriodicGaussianLocalOrbitalFactorConfig panel;
};
using PeriodicGaussianDensityGramOptions = PeriodicCorrelationRealLocalProviderOptions;
using PeriodicGaussianDensityGramLiveInventory = PeriodicGaussianLocalOrbitalFactorLiveInventory;
struct PeriodicGaussianDensityGramCaps {
    PeriodicGaussianMetricCaps resources, metric;
    PeriodicGaussianLocalOrbitalFactorCaps panel;
    std::uint64_t maximum_left_density_count = 0, maximum_right_density_count = 0;
    std::uint64_t maximum_output_elements = 0;
    std::uint64_t maximum_factor_panels = 0, maximum_tile_calls = 0;
    std::uint64_t maximum_image_candidate_evaluations = 0;
    std::uint64_t maximum_leaf_control_bytes = 0;
};
struct PeriodicGaussianDensityGramPlan {
    PeriodicGaussianDensityGramSelection selection;
    std::uint64_t n_cells = 0, n_basis = 0, n_auxiliary = 0;
    std::uint64_t occupied_count = 0, virtual_count = 0, orbital_count = 0;
    std::uint64_t common_density_count = 0, selected_density_count = 0, output_elements = 0;
    std::uint64_t left_run_count = 0, right_run_count = 0, maximum_run_length = 0;
    std::uint64_t row_count = 0, self_inverse_q_count = 0, auxiliary_block_count = 0;
    std::uint64_t factor_panels = 0, tile_calls = 0;
    std::uint64_t retained_output_bytes = 0, integral_accumulator_bytes = 0;
    std::uint64_t norm_workspace_bytes = 0, row_slab_bytes = 0, factor_phase_retained_bytes = 0;
    std::uint64_t whitener_bytes = 0, maximum_live_whitener_bytes = 0;
    std::uint64_t retained_panel_upper_bytes = 0, panel_driver_owned_upper_bytes = 0;
    std::uint64_t metric_phase_upper_bytes = 0, panel_phase_upper_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t borrowed_local_numerical_bytes = 0, borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t macro_control_storage_bytes = 0, minimum_leaf_control_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t source_factory_calls = 0, metric_calls = 0, whitening_calls = 0;
    std::uint64_t reciprocal_candidate_evaluations_upper_bound = 0;
    std::uint64_t image_candidate_evaluations_upper_bound = 0;
    std::uint64_t input_validation_work_units = 0, driver_work_units = 0, work_units = 0;
};
struct PeriodicGaussianDensityGramDiagnostics {
    std::uint64_t completed_source_count = 0, completed_metric_count = 0;
    std::uint64_t completed_whitening_count = 0, completed_factor_panels = 0;
    std::uint64_t completed_tile_calls = 0, reciprocal_candidate_evaluations = 0;
    std::uint64_t image_candidate_evaluations = 0, charged_work_units_upper_bound = 0;
    std::uint64_t maximum_observed_owned_numerical_bytes = 0;
    std::uint64_t maximum_observed_per_worker_inventoried_bytes = 0;
    // These maxima and error bounds cover ONLY the two selected ranges,
    // including all actual q/original-auxiliary rows and reversed densities.
    // Nothing is certified about unrequested common densities.
    PeriodicCorrelationRealLocalProviderDiagnostics real_projection;
    // Outward interval of represented projected-real-row products, including
    // compensated-sum rounding. Separate from the projection bound above.
    double maximum_integral_roundoff_error = 0.0;
};

class PeriodicGaussianDensityGramBlock {
public:
    PeriodicGaussianDensityGramBlock(const PeriodicGaussianDensityGramBlock&) = delete;
    PeriodicGaussianDensityGramBlock& operator=(const PeriodicGaussianDensityGramBlock&) = delete;
    PeriodicGaussianDensityGramBlock(PeriodicGaussianDensityGramBlock&&) noexcept = default;
    PeriodicGaussianDensityGramBlock& operator=(PeriodicGaussianDensityGramBlock&&) = delete;
    const PeriodicGaussianDensityGramPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianDensityGramDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    // Row-major [selection.left.count,selection.right.count], no 1/K energy
    // normalization. Each entry is the selected finite-torus ERI (pq|rs).
    const double* data() const;
    double element(std::size_t left, std::size_t right) const;
    std::array<std::uint64_t,2> left_density(std::size_t index) const;
    std::array<std::uint64_t,2> right_density(std::size_t index) const;
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& source_context_identity_sha256() const noexcept { return context_sha_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
    const std::string& consumed_sources_identity_sha256() const noexcept { return consumed_; }
    bool matched_finite_gaussian_hf_recipe() const noexcept { return state_ && context_; }
    bool original_provider_projection_reproduced_bitwise() const noexcept { return false; }
    bool all_common_densities_audited() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    bool production_dlpno() const noexcept { return false; }
private:
    PeriodicGaussianDensityGramBlock() = default;
    void require_live() const;
    PeriodicGaussianDensityGramPlan memory_;
    PeriodicGaussianDensityGramDiagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::vector<double> values_;
    std::string identity_,payload_,hf_,context_sha_,basis_,consumed_;
    friend PeriodicGaussianDensityGramBlock build_periodic_gaussian_density_gram(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
        const BasisSet&,const BasisSet&,const PeriodicCorrelationWannier&,
        const std::complex<double>*,std::size_t,const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianDensityGramSelection&,const PeriodicGaussianDensityGramConfig&,
        const PeriodicGaussianDensityGramOptions&,const PeriodicGaussianDensityGramLiveInventory&,
        const PeriodicGaussianDensityGramCaps&);
};

// Scalar selection/count/output admission precedes source construction,
// gauge/basis/payload scans or numerical allocation. Packed-range decoding
// is integer-only O(log m); no density-sized metadata table is constructed.
// Each range splits into maximal same-left/right-contiguous rectangles.
// Reversed rectangles and actual qbar sources are evaluated explicitly.
// The source and whiteners are reused for every run and auxiliary slice.
//
// L/R are density counts, d=L+R, b=auxiliary_block, t=max run length,
// a=nAO, o=occupied_count, A=original auxiliary dimension, W=16A^2.
// P=16bt+16o is one retained panel; D=32bt+16a(t+1)+32a+16o is its driver.
// Base=32LR+24d+(nonself?16:8)bd: output+compensations+interval endpoints,
// three norm vectors, and one/two-q real slab. Metric peak is bounded by
// Base+(nonself?W:0)+metric-owned-cap; panel peak by
// Base+(nonself?2W:W)+(nonself?2P:P)+D+tile-owned-cap. In a nonself
// pair the build order is q forward, q reverse, qbar reverse; release q
// reverse, then qbar forward. Both oriented conjugacies are audited while
// at most two previously retained panels coexist with the current driver.
// Only8LR numerical bytes survive return. All original local/Gaussian
// owners, fixed logical controls, caller extras and positive backend margin
// are independently inventoried; the shared reference baseline is once.
// These are explicit logical numerical/control bounds, not OS RSS claims.
PeriodicGaussianDensityGramPlan plan_periodic_gaussian_density_gram(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianDensityGramSelection&,const PeriodicGaussianDensityGramConfig&,
    const PeriodicGaussianDensityGramOptions&,const PeriodicGaussianDensityGramLiveInventory&,
    const PeriodicGaussianDensityGramCaps&);

// Callback-free synchronous factory. Every borrowed original owner/view
// remains immutable and alive through return. Successful output retains
// only the block, copied scalar metadata/receipts and shared state/context.
// Representative-q-first slab dot order can differ from a stored provider's
// canonical q-index row order; explicit rounding/projection diagnostics are
// supplied, not a bitwise-equivalence assertion. No full-factor-store owner.
PeriodicGaussianDensityGramBlock build_periodic_gaussian_density_gram(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianDensityGramSelection&,const PeriodicGaussianDensityGramConfig&,
    const PeriodicGaussianDensityGramOptions&,const PeriodicGaussianDensityGramLiveInventory&,
    const PeriodicGaussianDensityGramCaps&);

} // namespace vibeqc
