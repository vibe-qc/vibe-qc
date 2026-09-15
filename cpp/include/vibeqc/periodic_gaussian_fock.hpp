#pragma once

// Native, density-independent finite-Gaussian two-electron response only.
// Sun et al. (2017), doi:10.1063/1.4998644, Eqs.13,16,18,21.
// B_P(k_bra,q) has AO rows mu, columns nu and q=k_ket-k_bra.
// z_P=(1/Nk)sum_k Tr[D(k)B_P(k,0)], J(k)=sum_P B_P(k,0)^dagger z_P;
// K(k_ket)=(1/Nk)sum_k_bra,P B_P^dagger D(k_bra) B_P; G[D]=J-K/2.
// D is spin-summed (e.g. 2*Cocc*Cocc^dagger), without extra spin/k weights.
// No nuclear/Hcore, HF convergence, image-tail, qbar-rank covariance or
// periodic DLPNO accuracy claim is made by computing this response.

#include "vibeqc/periodic_gaussian_three_center.hpp"

namespace vibeqc {

struct PeriodicGaussianFockDensityView {
    // Borrowed immutable for the entire call, including all callbacks.
    // Row-major (Nk,nAO,nAO). No positivity/idempotency requirement: this
    // linear operator also accepts Hermitian density differences. Concurrent
    // mutation/data races are forbidden; callback-boundary hashes detect
    // ordinary mutation. Input must be finite and EXACTLY Hermitian.
    const std::complex<double>* data = nullptr;
    std::uint64_t element_count = 0;
};
struct PeriodicGaussianFockConfig {
    std::uint64_t auxiliary_block = 0, ao_column_block = 0;
    PeriodicGaussianReciprocalSourceCaps source_caps;
    PeriodicGaussianMetricConfig metric;
    PeriodicGaussianMetricCaps metric_caps;
    PeriodicGaussianThreeCenterConfig tile;
    PeriodicGaussianThreeCenterCaps tile_caps;
};
struct PeriodicGaussianFockCaps {
    // Cumulative source candidate bound includes factories, metric replays
    // and every tile replay; image traversal bound is separate below.
    PeriodicGaussianMetricCaps resources;
    std::uint64_t maximum_tile_calls = 0;
    std::uint64_t maximum_progress_callbacks = 0;
    std::uint64_t maximum_image_candidate_evaluations = 0;
};
struct PeriodicGaussianFockPlan {
    std::uint64_t n_kpoints = 0, n_basis = 0, n_auxiliary = 0;
    std::uint64_t auxiliary_block = 0, ao_column_block = 0;
    std::uint64_t auxiliary_block_count = 0, ao_column_block_count = 0;
    std::uint64_t density_element_count = 0, borrowed_density_bytes = 0;
    std::uint64_t response_bytes = 0, response_compensation_bytes = 0;
    std::uint64_t hartree_vector_bytes = 0, exchange_double_panel_bytes = 0;
    std::uint64_t resident_whitener_bytes = 0;
    std::uint64_t metric_phase_owned_numeric_upper_bound = 0;
    std::uint64_t tile_phase_owned_numeric_upper_bound = 0;
    std::uint64_t owned_numeric_upper_bound = 0;
    std::uint64_t borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t macro_fixed_object_bytes = 0, maximum_leaf_fixed_inventory_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, node_inventoried_bytes = 0;
    std::uint64_t hartree_tile_calls = 0, exchange_tile_calls = 0, total_tile_calls = 0;
    std::uint64_t progress_callback_upper_bound = 0, driver_contraction_terms = 0;
    std::uint64_t reciprocal_candidate_evaluations_upper_bound = 0;
    std::uint64_t image_candidate_evaluations_upper_bound = 0;
    std::uint64_t driver_work_units_upper_bound = 0, work_units_upper_bound = 0;
    PeriodicGaussianFockConfig config;
    PeriodicGaussianMetricLiveInventory live;
    PeriodicGaussianFockCaps caps;
    std::array<char, 64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(), identity_ascii.end()}; }
};
struct PeriodicGaussianFockMatrixDiagnostics {
    double maximum_magnitude = 0.0;
    double maximum_hermiticity_residual = 0.0, relative_hermiticity_residual = 0.0;
    double maximum_time_reversal_residual = 0.0, relative_time_reversal_residual = 0.0;
    double maximum_diagonal_imaginary = 0.0;
};
enum class PeriodicGaussianFockStage : std::uint32_t {
    Begin = 0, Source = 1, Metric = 2, Whitening = 3,
    HartreeDensity = 4, HartreeApply = 5, Exchange = 6, QComplete = 7, Complete = 8
};
struct PeriodicGaussianFockProgress {
    PeriodicGaussianFockStage stage = PeriodicGaussianFockStage::Begin;
    std::uint64_t q_index = 0, auxiliary_begin = 0, k_bra_index = 0, k_ket_index = 0;
    std::uint64_t sigma_begin = 0, completed_q_count = 0, completed_tile_count = 0;
    std::uint64_t charged_work_units_upper_bound = 0;
};
struct PeriodicGaussianFockReceipt {
    std::uint64_t completed_q_count = 0, completed_tile_count = 0, progress_callback_count = 0;
    std::uint64_t source_factory_candidate_evaluations = 0, reciprocal_candidate_evaluations = 0;
    std::uint64_t image_candidate_evaluations = 0, charged_work_units_upper_bound = 0;
    std::uint64_t maximum_observed_owned_numeric_bytes = 0;
    std::uint64_t maximum_observed_per_replica_inventoried_bytes = 0;
};
class PeriodicGaussianFockResult {
public:
    PeriodicGaussianFockResult(const PeriodicGaussianFockResult&) = delete;
    PeriodicGaussianFockResult& operator=(const PeriodicGaussianFockResult&) = delete;
    PeriodicGaussianFockResult(PeriodicGaussianFockResult&&) noexcept = default;
    PeriodicGaussianFockResult& operator=(PeriodicGaussianFockResult&&) = delete;
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const PeriodicGaussianFockPlan& plan() const noexcept { return plan_; }
    const PeriodicGaussianFockReceipt& receipt() const noexcept { return receipt_; }
    const PeriodicGaussianFockMatrixDiagnostics& density_diagnostics() const noexcept { return density_diagnostics_; }
    const PeriodicGaussianFockMatrixDiagnostics& response_diagnostics() const noexcept { return response_diagnostics_; }
    const std::vector<std::complex<double>>& matrix_row_major() const noexcept { return values_; }
    std::string density_identity_sha256() const;
    std::string consumed_factor_identity_sha256() const;
    std::string payload_identity_sha256() const;
private:
    PeriodicGaussianFockResult() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianFockPlan plan_;
    PeriodicGaussianFockReceipt receipt_;
    PeriodicGaussianFockMatrixDiagnostics density_diagnostics_, response_diagnostics_;
    std::array<char, 64> density_{}, consumed_{}, payload_{};
    std::vector<std::complex<double>> values_;
    friend PeriodicGaussianFockResult build_periodic_gaussian_fock(
        std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet&, const BasisSet&,
        PeriodicGaussianFockDensityView, PeriodicGaussianFockConfig,
        PeriodicGaussianMetricLiveInventory, PeriodicGaussianFockCaps,
        void (*)(const PeriodicGaussianFockProgress&, void*), void*);
};

// Explicit conservative macro admission, before output allocation. It uses
// exact call counts and configured leaf ceilings, with no all-q plan table.
// Loose leaf caps may reject an otherwise affordable run. Work and fixed
// object inventories are NOT elapsed-time/instruction/stack/RSS estimates.
// Leaf plans still re-admit all live owners; returned actual counts are
// charged before the next call. Neither this plain plan nor caller labels
// can authenticate numerical factors. All factors are built here natively.
PeriodicGaussianFockPlan plan_periodic_gaussian_fock(
    const PeriodicGaussianSourceContext&, const PeriodicGaussianFockConfig&,
    const PeriodicGaussianMetricLiveInventory&, const PeriodicGaussianFockCaps&);
// All controls are copied; D and physical bases are borrowed immutable.
// Progress can throw to cancel. Every callback is followed by a full D hash
// check, charged in the work bound. No receipt/result survives a failure.
// Progress references expire on callback return. Callbacks must declare
// their retained/transient storage in live; arbitrary callback allocations
// are not certified by this driver. The tiny Python adapter copies events.
// One q source/W and at most one transient tile are live. Return only complex
// G[D]; no Hermiticity/TR averaging or hidden real projection is performed.
//
// Full owned output+compensation is32*Nk*nAO². Metric/whitener phases add
// the explicit metric-owned ceiling. Tile phases add resident W16*A² ONCE,
// max(32*Ab,32*Ab*nAO*b) driver scratch and the tile-owned ceiling (which
// includes its queried fixed Fourier workspace). The leaves reserve the active bases
// once per role; driver declares D, output, scratch and macro controls only.
// The Python diagnostic additionally reserves its second simultaneously live
// caller/snapshot D buffer before copying, and snapshots mutable controls.
//
// Wires use big-endian integers, u64-length-prefixed strings, finite binary64
// with signed zeros normalized. Density v1 domain
// "vibeqc.periodic.gaussian-fock.density": context SHA; Nk,nAO,elements,bytes
// u64; row-major complex D. Consumed-factors v1 domain
// "vibeqc.periodic.gaussian-fock.consumed-factors": context and D SHA;
// Nk,nAO,A,tile_count u64; each q index then source/opposite/raw/W SHA, then
// each tile's monotone sequence u64 and payload SHA, in actual call order.
// Payload v1 domain "vibeqc.periodic.gaussian-fock.payload": context,D,
// consumed-factor SHA; Nk,nAO,elements,bytes; D then G diagnostics in their
// declaration order; row-major complex G. All domains start with u32 v1
// after the length-prefixed domain. Exact prefixes are116/188/356 plus the
// respective domain lengths; consumed records are296/q and80/tile bytes.
// Plan v1 has its own domain and hashes context, all plan counts in
// declaration order, then config/live/caps. It is not a numerical identity.
PeriodicGaussianFockResult build_periodic_gaussian_fock(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet& ao, const BasisSet& auxiliary,
    PeriodicGaussianFockDensityView, PeriodicGaussianFockConfig,
    PeriodicGaussianMetricLiveInventory, PeriodicGaussianFockCaps,
    void (*progress)(const PeriodicGaussianFockProgress&, void*) = nullptr, void* progress_context = nullptr);

} // namespace vibeqc
