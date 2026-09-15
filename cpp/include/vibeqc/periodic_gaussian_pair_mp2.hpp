#pragma once

// Actual finite-Gaussian HF-origin pair-PNO coupled-MP2 reference.
// Nejad doi:10.1063/5.0290816 Eqs.24,37-44. Every unordered occupied
// pair is built from one certified common real PAO basis; pair ranks vary.
// No automatic distinct PAO domains, local-auxiliary approximation, weak-pair
// correction, translation/symmetry reduction or production DLPNO is implied.
#include <optional>
#include "vibeqc/periodic_gaussian_pair_space.hpp"
#include "vibeqc/bounded_restricted_pair_mp2_solver.hpp"
#include "vibeqc/periodic_gaussian_pair_domain_builder.hpp"

namespace vibeqc {
class PeriodicGaussianPairDomainBuilder;
struct PeriodicGaussianDomainPairMP2Options;
struct PeriodicGaussianDomainPairMP2Caps;
struct PeriodicGaussianGramDomainPairMP2Config;
struct PeriodicGaussianGramDomainPairMP2Options;
struct PeriodicGaussianGramDomainPairMP2Caps;
enum class PeriodicGaussianPairMP2SourceKind : std::uint32_t {
    CommonFactorRows = 0, DirectPAOGram = 1
};

struct PeriodicGaussianPairMP2Options {
    PeriodicGaussianPairPNOOptions pnos;
    PeriodicGaussianPairSpaceOptions projection;
    BoundedRestrictedPairMP2SolverOptions solver;
    // Brillouin block discarded by the MP2 model. Explicit measured norm
    // gate, not a convergence fix; the original certified F is unchanged.
    double maximum_occupied_virtual_fock_norm = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicGaussianPairMP2LiveInventory {
    // Beyond the exact reference, common-basis F/labels and provider rows.
    // Includes still-live Gaussian/localization owners and callback storage.
    std::uint64_t other_live_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianPairMP2Caps {
    PeriodicGaussianPairPNOCaps pnos;
    PeriodicGaussianPairSpaceCaps projection;
    BoundedRestrictedPairMP2SolverCaps solver;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_pair_count = 0, maximum_integral_calls = 0;
    std::uint64_t maximum_progress_callbacks = 0, maximum_work_units = 0;
};
struct PeriodicGaussianPairMP2Plan {
    PeriodicGaussianPairMP2SourceKind source_kind = PeriodicGaussianPairMP2SourceKind::CommonFactorRows;
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, pair_count = 0;
    std::uint64_t retained_pair_output_upper_bytes = 0, pair_generation_phase_upper_bytes = 0;
    std::uint64_t solver_phase_owned_upper_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_provider_row_bytes = 0;
    std::uint64_t borrowed_gaussian_bytes = 0, borrowed_wannier_bytes = 0, borrowed_gauge_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t retained_pair_seal_bytes = 0;
    bool domain_generated = false;
    std::uint64_t borrowed_domain_builder_bytes = 0, retained_generation_embedding_upper_bytes = 0;
    // Complete geometry owner, INCLUDING the diagonal embedding subset.
    std::uint64_t retained_pair_geometry_upper_bytes = 0, retained_pair_geometry_control_upper_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t integral_calls_upper_bound = 0, progress_callback_upper_bound = 0;
    std::uint64_t gram_builds_upper_bound = 0, factor_panels_upper_bound = 0, tile_calls_upper_bound = 0;
    std::uint64_t driver_work_units = 0, work_units_upper_bound = 0;
    BoundedRestrictedPairMP2SolverMemoryPlan solver_upper;
};
enum class PeriodicGaussianPairMP2Stage : std::uint32_t {
    Begin = 0, PairComplete = 1, Solver = 2, Finished = 3, PairDomainReady = 4
};
struct PeriodicGaussianPairMP2Progress {
    PeriodicGaussianPairMP2Stage stage = PeriodicGaussianPairMP2Stage::Begin;
    std::uint64_t callback_count = 0, completed_pairs = 0;
    std::uint64_t occupied_i = 0, occupied_j = 0, pair_rank = 0;
    BoundedRestrictedPairMP2SolverProgress solver;
};
using PeriodicGaussianPairMP2Callback = void (*)(const PeriodicGaussianPairMP2Progress&, void*);
struct PeriodicGaussianPairMP2Diagnostics {
    PeriodicGaussianPairMP2SourceKind source_kind = PeriodicGaussianPairMP2SourceKind::CommonFactorRows;
    std::uint64_t completed_pairs = 0, completed_integral_calls = 0;
    std::uint64_t completed_progress_callbacks = 0, retained_pair_bytes = 0;
    std::uint64_t completed_gram_builds = 0, completed_factor_panels = 0, completed_tile_calls = 0;
    std::uint64_t minimum_pair_rank = 0, maximum_pair_rank = 0, zero_rank_pairs = 0;
    bool domain_generated = false;
    // All P pre-truncation density dimensions, never padded to common n.
    std::uint64_t generation_dimension_sum = 0;
    // Authentic embedded D[m_ij,r_ij] retained in pair owners, never an
    // amplitude-reader input role. Exactly zero on the legacy common path.
    std::uint64_t retained_generation_coefficient_bytes = 0;
    // Diagonal SUBSET of the complete geometry below, for independently
    // cut CCSD singles. Never add this subset to a whole-owner inventory.
    std::uint64_t diagonal_generation_dimension_sum = 0, retained_generation_embedding_bytes = 0;
    // Complete sum over canonical pairs: 16d+32d²+16dm+8(d+m)
    // for actual PAO domain/real space, plus 8(nm+m) for its embedding.
    // All zero on the legacy common-generation branch.
    std::uint64_t retained_pair_geometry_bytes = 0, retained_pair_geometry_control_bytes = 0;
    bool complete_common_finite_torus_basis = false, all_pairs_full_rank = false;
    double occupied_virtual_fock_norm_upper_bound = 0.0;
    double maximum_diagonal_integral_projection_norm = 0.0;
};
class PeriodicGaussianPairMP2Result {
public:
    PeriodicGaussianPairMP2Result(const PeriodicGaussianPairMP2Result&) = delete;
    PeriodicGaussianPairMP2Result& operator=(const PeriodicGaussianPairMP2Result&) = delete;
    PeriodicGaussianPairMP2Result(PeriodicGaussianPairMP2Result&&) noexcept = default;
    PeriodicGaussianPairMP2Result& operator=(PeriodicGaussianPairMP2Result&&) noexcept = delete;
    const PeriodicGaussianPairMP2Plan& memory() const noexcept { return memory_; }
    const PeriodicGaussianPairMP2Diagnostics& diagnostics() const noexcept { return diagnostics_; }
    const BoundedRestrictedPairMP2SolverResult& solver() const;
    const PeriodicGaussianPairSpace& pair(std::uint64_t i,std::uint64_t j) const;
    bool domain_generated() const noexcept { return diagnostics_.domain_generated; }
    PeriodicGaussianPairMP2SourceKind source_kind() const noexcept { return memory_.source_kind; }
    bool direct_gram_source() const noexcept { return source_kind()==PeriodicGaussianPairMP2SourceKind::DirectPAOGram; }
    const PeriodicCorrelationRealPAOEmbedding& diagonal_generation_embedding(std::uint64_t i) const;
    // Canonical i<=j only. Native children remain borrowed from this result.
    const PeriodicGaussianPairDomainGeometry& pair_generation_geometry(std::uint64_t i,std::uint64_t j) const;
    PeriodicGaussianPairPNOGeometryView pair_generation_geometry_view(std::uint64_t i,std::uint64_t j) const &;
    PeriodicGaussianPairPNOGeometryView pair_generation_geometry_view(std::uint64_t,std::uint64_t) const && = delete;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    bool converged() const;
    bool matched_finite_gaussian_hf_recipe() const noexcept { return static_cast<bool>(context_); }
    bool production_dlpno() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    // Includes explicit PNO truncation; not an untruncated energy certificate.
    // No per-cell number for an incomplete common occupied/virtual basis or
    // an unconverged pair solve. No additive missing-pair/PNO correction.
    bool periodic_energy_per_cell() const;
    double correlation_energy_per_cell() const;
    double total_energy_per_cell() const;
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& pair_spaces_identity_sha256() const noexcept { return pairs_sha_; }
    // DirectPAOGram has no common provider; asking for that receipt throws.
    const std::string& provider_identity_sha256() const;
    const std::string& gram_sources_identity_sha256() const noexcept { return gram_sources_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
    const std::string& domain_builder_identity_sha256() const noexcept { return builder_; }
private:
    PeriodicGaussianPairMP2Result() = default;
    PeriodicGaussianPairMP2Plan memory_;
    PeriodicGaussianPairMP2Diagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::vector<PeriodicGaussianPairSpace> pairs_;
    std::vector<PeriodicGaussianPairDomainGeometry> pair_geometry_;
    std::optional<BoundedRestrictedPairMP2SolverResult> solver_;
    std::string identity_,pairs_sha_,provider_,hf_,builder_,gram_sources_;
    friend PeriodicGaussianPairMP2Result run_periodic_gaussian_pair_mp2(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Options&,
        const PeriodicGaussianPairMP2LiveInventory&,const PeriodicGaussianPairMP2Caps&,
        PeriodicGaussianPairMP2Callback,void*);
    friend PeriodicGaussianPairMP2Result run_periodic_gaussian_domain_pair_mp2(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairDomainBuilder&,
        const PeriodicGaussianDomainPairMP2Options&,const PeriodicGaussianPairMP2LiveInventory&,
        const PeriodicGaussianDomainPairMP2Caps&,PeriodicGaussianPairMP2Callback,void*);
    friend PeriodicGaussianPairMP2Result run_periodic_gaussian_domain_pair_mp2(
        const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,const BasisSet&,const BasisSet&,
        const PeriodicCorrelationWannier&,const std::complex<double>*,std::size_t,
        const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairDomainBuilder&,
        const PeriodicGaussianGramDomainPairMP2Config&,const PeriodicGaussianGramDomainPairMP2Options&,
        const PeriodicGaussianPairMP2LiveInventory&,const PeriodicGaussianGramDomainPairMP2Caps&,
        PeriodicGaussianPairMP2Callback,void*);
};

// Constant-size metadata admission using conservative full-rank bounds,
// BEFORE floating scans, pair tables, PNO construction or progress. All
// leaves independently re-admit their exact live owners; reference/state is
// counted once. Control/backend allowance is not a stack/allocator/RSS bound.
PeriodicGaussianPairMP2Plan plan_periodic_gaussian_pair_mp2(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Options&,
    const PeriodicGaussianPairMP2LiveInventory&,const PeriodicGaussianPairMP2Caps&);
// No G/F/C/T or overlap inputs. Actual PNOs and projected G are built natively
// for all lexicographic i<=j. Completed spaces remain owned for the ragged
// coupled solver and later reference consumers. Controls are copied before
// scalar callbacks; all object inputs must stay immutable until return.
// A callback exception publishes no result; iteration exhaustion preserves
// the last evaluated solver state without claiming energy convergence.
PeriodicGaussianPairMP2Result run_periodic_gaussian_pair_mp2(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Options&,
    const PeriodicGaussianPairMP2LiveInventory&,const PeriodicGaussianPairMP2Caps&,
    PeriodicGaussianPairMP2Callback progress=nullptr,void* progress_context=nullptr);

} // namespace vibeqc
