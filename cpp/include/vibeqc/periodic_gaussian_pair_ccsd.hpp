#pragma once

// Actual finite-Gaussian-source pair-space Galerkin CCSD REFERENCE. The
// complete common-frame Stanton residual is projected only at its output,
// not the production DLPNO interaction/domain approximation (Riplinger,
// doi:10.1063/1.4773581 Eqs.26-29). No triples/TNO contribution is present.
// A separately named overload replaces ONLY the bare particle-hole group;
// its generic producer is not certified by the original Gaussian lineage.
#include <optional>
#include "vibeqc/periodic_gaussian_pair_mp2.hpp"
#include "vibeqc/periodic_gaussian_embedded_pair_pnos.hpp"
#include "vibeqc/periodic_gaussian_pair_pno_frame.hpp"
#include "vibeqc/bounded_restricted_pair_ccsd_solver.hpp"

namespace vibeqc {

// Only the separately implemented native physical-source wrapper may qualify
// the optional producer after checking its original Gaussian source receipts.
struct PeriodicGaussianPairCCSDPhysicalParticleHoleAccess;

struct PeriodicGaussianPairCCSDEmbeddingOptions {
    // Used only for an MP2 owner generated in distinct PAO pair frames.
    // The original diagonal embedding is retained by that owner; singles
    // are regenerated INSIDE it, with Options.singles as the sole PNO
    // controls. No common-density or already-truncated-PNO fallback.
    double maximum_diagonal_exchange_projection_norm = std::numeric_limits<double>::quiet_NaN();
    double maximum_fock_symmetry_projection_norm = std::numeric_limits<double>::quiet_NaN();
    double maximum_exported_gram_error = std::numeric_limits<double>::quiet_NaN();
    double maximum_exported_fock_error = std::numeric_limits<double>::quiet_NaN();
    double maximum_exported_subspace_error = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicGaussianPairCCSDOptions {
    // Independently generated diagonal-pair singles spaces, using periodic
    // Nejad doi:10.1063/5.0290816 Eq.39 occupations. The cutoff is explicit;
    // no unvalidated transfer of molecular 0.03*TCutPNO is made. Separate
    // cutoff=0 complete-space diagnostics and rank-zero spaces are allowed.
    PeriodicGaussianPairPNOOptions singles;
    PeriodicGaussianPairCCSDEmbeddingOptions singles_embedding;
    // Integral work per call must equal the actual native provider bound.
    // Denominator floor must equal the singles generation floor. Original
    // Foo/Fvv/Fov are retained, including off-diagonal occupied couplings.
    BoundedRestrictedPairCCSDSolverOptions solver;
};
struct PeriodicGaussianPairCCSDLiveInventory {
    // Beyond exact reference, basis, Gaussian provider and borrowed MP2.
    std::uint64_t other_live_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicGaussianPairCCSDCaps {
    PeriodicGaussianPairPNOCaps singles;
    PeriodicGaussianEmbeddedPairPNOCaps embedded_singles;
    BoundedRestrictedPairCCSDSolverCaps solver;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_pair_count = 0, maximum_integral_calls = 0;
    std::uint64_t maximum_progress_callbacks = 0, maximum_work_units = 0;
};
struct PeriodicGaussianPairCCSDPlan {
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, pair_count = 0;
    bool domain_generated = false;
    bool split_bare_particle_hole = false;
    // Producer-owned numeric objects beyond every source already counted by
    // this outer wrapper. These remain live during singles generation too.
    std::uint64_t borrowed_particle_hole_additional_numerical_bytes = 0;
    std::uint64_t singles_generation_dimension_sum = 0;
    std::uint64_t borrowed_generation_embedding_bytes = 0;
    std::uint64_t borrowed_basis_bytes = 0, borrowed_provider_row_bytes = 0;
    std::uint64_t borrowed_mp2_numerical_bytes = 0, borrowed_mp2_pair_ct_bytes = 0;
    std::uint64_t borrowed_mp2_control_bytes = 0;
    std::uint64_t singles_output_upper_bytes = 0, zero_singles_upper_bytes = 0;
    std::uint64_t singles_generation_phase_upper_bytes = 0;
    std::uint64_t solver_phase_owned_upper_bytes = 0, peak_owned_numerical_bytes = 0;
    // Uniform full-rank solver planning overcounts borrowed warm-start C/T
    // at truncated ranks. Explicit conservative reservation, NOT owned data.
    std::uint64_t solver_rank_padding_upper_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t integral_calls_upper_bound = 0, progress_callback_upper_bound = 0;
    std::uint64_t driver_work_units = 0, work_units_upper_bound = 0;
    PeriodicGaussianPairPNOPlan singles_upper;
    // Owner-free monotone m=n census, not a fabricated physical leaf plan.
    PeriodicGaussianEmbeddedPairPNOCountPlan embedded_singles_upper;
    BoundedRestrictedPairCCSDSolverMemoryPlan solver_upper;
};
enum class PeriodicGaussianPairCCSDStage : std::uint32_t {
    Begin = 0, SinglesComplete = 1, Solver = 2, Finished = 3
};
struct PeriodicGaussianPairCCSDProgress {
    PeriodicGaussianPairCCSDStage stage = PeriodicGaussianPairCCSDStage::Begin;
    std::uint64_t callback_count = 0, completed_singles = 0, occupied_i = 0, singles_rank = 0;
    BoundedRestrictedPairCCSDSolverProgress solver;
};
using PeriodicGaussianPairCCSDCallback = void (*)(const PeriodicGaussianPairCCSDProgress&, void*);
struct PeriodicGaussianPairCCSDDiagnostics {
    std::uint64_t completed_singles = 0, completed_integral_calls = 0, completed_progress_callbacks = 0;
    std::uint64_t retained_singles_bytes = 0, zero_initial_singles_bytes = 0;
    std::uint64_t retained_singles_generation_coefficient_bytes = 0;
    std::uint64_t minimum_singles_rank = 0, maximum_singles_rank = 0, zero_rank_singles = 0;
    std::uint64_t singles_generation_dimension_sum = 0;
    bool complete_common_finite_torus_basis = false, all_singles_full_rank = false, all_pairs_full_rank = false;
    bool split_bare_particle_hole = false;
    std::uint64_t completed_particle_hole_calls = 0, completed_particle_hole_visits = 0;
    std::uint64_t charged_particle_hole_work_units = 0;
};
class PeriodicGaussianPairCCSDResult {
public:
    PeriodicGaussianPairCCSDResult(const PeriodicGaussianPairCCSDResult&) = delete;
    PeriodicGaussianPairCCSDResult& operator=(const PeriodicGaussianPairCCSDResult&) = delete;
    PeriodicGaussianPairCCSDResult(PeriodicGaussianPairCCSDResult&&) noexcept = default;
    PeriodicGaussianPairCCSDResult& operator=(PeriodicGaussianPairCCSDResult&&) noexcept = delete;
    const PeriodicGaussianPairCCSDPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianPairCCSDDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const BoundedRestrictedPairCCSDSolverResult& solver() const;
    PeriodicGaussianPairPNOFrameView singles_frame(std::uint64_t i) const;
    // Legacy native getter remains strict: embedded generations cannot be
    // presented as an old common-generation owner with padded occupations.
    const PeriodicGaussianPairPNOResult& singles(std::uint64_t i) const;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    bool converged() const;
    bool periodic_energy_per_cell() const;
    double correlation_energy_per_cell() const;
    double total_energy_per_cell() const;
    // Structural lineage of the original HF/basis/ordinary integral provider
    // only. A generic replacement producer has no physical-source certificate.
    bool matched_finite_gaussian_hf_recipe() const noexcept { return static_cast<bool>(context_); }
    bool split_bare_particle_hole() const noexcept { return memory_.split_bare_particle_hole; }
    bool particle_hole_physical_source_certified() const noexcept { return physical_particle_hole_source_; }
    const std::string& physical_particle_hole_source_identity_sha256() const noexcept {
        return physical_particle_hole_source_identity_;
    }
    const std::string& split_execution_identity_sha256() const;
    const std::string& consumed_particle_hole_identity_sha256() const;
    bool production_dlpno() const noexcept { return false; }
    bool includes_triples() const noexcept { return false; }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& warmstart_identity_sha256() const noexcept { return warmstart_; }
    const std::string& pair_spaces_identity_sha256() const noexcept { return pairs_; }
    const std::string& singles_spaces_identity_sha256() const noexcept { return singles_sha_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
    const std::string& provider_identity_sha256() const noexcept { return provider_; }
    const std::string& hf_reference_source_identity_sha256() const noexcept { return hf_; }
private:
    PeriodicGaussianPairCCSDResult() = default;
    PeriodicGaussianPairCCSDPlan memory_;
    PeriodicGaussianPairCCSDDiagnostics diagnostics_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::vector<PeriodicGaussianPairPNOFrame> singles_;
    std::optional<BoundedRestrictedPairCCSDSolverResult> solver_;
    std::string identity_,warmstart_,pairs_,singles_sha_,basis_,provider_,hf_;
    bool physical_particle_hole_source_ = false;
    std::string physical_particle_hole_source_identity_;
    static PeriodicGaussianPairCCSDResult run_impl(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
        const BoundedRestrictedPairCCSDParticleHoleProducer*,
        const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
        const PeriodicGaussianPairCCSDCaps&,PeriodicGaussianPairCCSDCallback,void*);
    friend struct PeriodicGaussianPairCCSDPhysicalParticleHoleAccess;
    friend PeriodicGaussianPairCCSDResult run_periodic_gaussian_pair_ccsd(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
        const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
        const PeriodicGaussianPairCCSDCaps&,PeriodicGaussianPairCCSDCallback,void*);
    friend PeriodicGaussianPairCCSDResult run_periodic_gaussian_pair_ccsd(
        const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
        const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
        const BoundedRestrictedPairCCSDParticleHoleProducer&,
        const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
        const PeriodicGaussianPairCCSDCaps&,PeriodicGaussianPairCCSDCallback,void*);
};

// Count-only full-rank admission, including every inner cap, before pair
// metadata traversal, floating scans, singles owners/tables or callbacks.
// Known owners are counted once; the separately named rank padding is an
// upper reservation, not a physically allocated amplitude tensor.
PeriodicGaussianPairCCSDPlan plan_periodic_gaussian_pair_ccsd(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
    const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
    const PeriodicGaussianPairCCSDCaps&);
// Requires a CONVERGED immutable actual-source MP2 warm start, exact common
// basis/provider/ref/context receipts, and no user numerical arrays. Borrows
// all pair C/T; copies only amplitudes inside the iterative solver. Warmstart
// remains alive/unmoved through return but is NOT retained by the result.
// Returned data owns singles spaces, local CCSD T and state/context only.
// Future pair-frame consumers must supply the original MP2 owner again and
// verify warmstart/pair-frame receipts. Exceptions publish no partial result;
// limits preserve the last evaluated unconverged snapshot, with no periodic
// total-energy claim. No missing-pair/PNO correction or (T) is added.
PeriodicGaussianPairCCSDResult run_periodic_gaussian_pair_ccsd(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
    const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
    const PeriodicGaussianPairCCSDCaps&,PeriodicGaussianPairCCSDCallback progress=nullptr,
    void* progress_context=nullptr);

// Explicit generic bare-particle-hole replacement. The producer is NOT an
// assertion of Gaussian source correctness: complete common-space basis and
// convergence alone cannot authorize physical per-cell totals on this route.
// It is copied before any callback and used by the split generic solver in
// place of ONLY the original W1/W2/WX bare seeds. Both singles-generation
// routes and exact ragged admission count its additional owners once.
// Additional retained roles must exclude ref/basis/provider/complete MP2,
// prepared singles and solver input/current/candidate roles counted here.
// No callback, producer context or warm-start owner is retained on return.
// Exceptions propagate; production_dlpno() remains false. Original APIs and
// identity codecs above remain unchanged. Split identities additionally seal
// execution mode and the consumed native particle-hole result stream.
PeriodicGaussianPairCCSDPlan plan_periodic_gaussian_pair_ccsd(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
    const BoundedRestrictedPairCCSDParticleHoleProducer&,
    const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
    const PeriodicGaussianPairCCSDCaps&);
PeriodicGaussianPairCCSDResult run_periodic_gaussian_pair_ccsd(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairMP2Result&,
    const BoundedRestrictedPairCCSDParticleHoleProducer&,
    const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
    const PeriodicGaussianPairCCSDCaps&,PeriodicGaussianPairCCSDCallback progress=nullptr,
    void* progress_context=nullptr);

} // namespace vibeqc
