#pragma once

// Connected finite-image Gaussian IAO -> diabatic seed -> TR-preserving
// IAO-PM optimization -> converged-only home-cell Wannier construction.
// Sun et al., doi:10.1063/1.4998644, Eqs.10/16; Zhu and Tew,
// doi:10.1021/acs.jpca.4c04555, Eqs.13-29. This is a bounded reference
// localizer, not a production-scaling or global-maximum certificate.

#include <optional>
#include "vibeqc/periodic_gaussian_bloch_iao.hpp"
#include "vibeqc/periodic_correlation_iao_optimizer.hpp"
#include "vibeqc/periodic_correlation_wannier.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicGaussianLocalizationVersion = 1U;

struct PeriodicGaussianLocalizationOptions {
    PeriodicGaussianBlochIAOOptions source;
    PeriodicCorrelationBlochIAOOptions iao;
    PeriodicCorrelationDiabaticSeedOptions seed;
    PeriodicCorrelationIAOPMOptions pm;
    PeriodicCorrelationIAOOptimizerOptions optimizer;
    PeriodicCorrelationWannierOptions wannier;
};
struct PeriodicGaussianLocalizationCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_point_owners = 0;
    std::uint64_t maximum_work_units = 0;
    // ALL image candidate visits: preliminary and repeated count-only
    // plans, source-factory preflights, and both value-panel image walks.
    std::uint64_t maximum_total_image_candidates = 0;
    std::uint64_t maximum_optimizer_work_units = 0;
};
struct PeriodicGaussianLocalizationMemoryPlan {
    std::uint64_t n_points = 0, n_basis = 0, n_minimal = 0, n_active = 0;
    std::uint64_t borrowed_gaussian_numerical_bytes = 0, other_live_numerical_bytes = 0;
    std::uint64_t point_plan_record_bytes = 0, point_owner_record_bytes = 0;
    std::uint64_t source_identity_character_bytes = 0, fixed_control_reservation_bytes = 0;
    std::uint64_t peak_control_storage_bytes = 0;
    std::uint64_t seed_phase_owned_bytes = 0, source_phase_owned_bytes = 0;
    std::uint64_t optimizer_phase_owned_bytes = 0, wannier_phase_owned_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, output_numerical_bytes = 0;
    std::uint64_t seed_required_node_bytes = 0, source_required_node_bytes = 0;
    std::uint64_t optimizer_required_node_bytes = 0, wannier_required_node_bytes = 0;
    std::uint64_t required_node_memory_bytes = 0;
    std::uint64_t planned_source_image_candidates = 0, planned_total_image_visits = 0;
    std::uint64_t source_planning_work_units = 0, source_factory_work_reservation = 0;
    std::uint64_t input_verification_work_units = 0, callback_work_reservation = 0;
    std::uint64_t wannier_work_reservation = 0, optimizer_work_cap = 0;
};
struct PeriodicGaussianLocalizationDiagnostics {
    std::uint64_t charged_work_units = 0, image_candidate_visits = 0;
    std::uint64_t constructed_iao_points = 0, callback_input_checks = 0;
    PeriodicGaussianBlochIAODiagnostics source;
};
enum class PeriodicGaussianLocalizationStage : std::uint32_t {
    SeedReady = 0, PointReady = 1, Optimization = 2, WannierReady = 3, Finished = 4
};
struct PeriodicGaussianLocalizationProgress {
    PeriodicGaussianLocalizationStage stage = PeriodicGaussianLocalizationStage::SeedReady;
    std::uint64_t completed_points = 0, point_count = 0, charged_work_units = 0;
    // Meaningful only for Optimization. Scalar snapshot, never a borrowed
    // matrix/owner. Its objective always names the last accepted iterate.
    PeriodicCorrelationIAOOptimizerProgress optimizer;
};
using PeriodicGaussianLocalizationCallback = void (*)(
    const PeriodicGaussianLocalizationProgress&, void*);

class PeriodicGaussianLocalizationResult {
public:
    PeriodicGaussianLocalizationResult(const PeriodicGaussianLocalizationResult&) = delete;
    PeriodicGaussianLocalizationResult& operator=(const PeriodicGaussianLocalizationResult&) = delete;
    PeriodicGaussianLocalizationResult(PeriodicGaussianLocalizationResult&&) noexcept = default;
    PeriodicGaussianLocalizationResult& operator=(PeriodicGaussianLocalizationResult&&) noexcept = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicGaussianLocalizationVersion; }
    const PeriodicCorrelationIAOOptimizerResult& optimizer() const;
    const PeriodicCorrelationWannier& wannier() const;
    bool converged() const;
    bool hf_basis_source_authenticated() const noexcept { return false; }
    bool infinite_image_tail_certified() const noexcept { return false; }
    const PeriodicGaussianLocalizationMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicGaussianLocalizationDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::string& gaussian_input_identity_sha256() const noexcept { return input_identity_; }
    const std::string& ordered_source_identity_sha256() const noexcept { return source_identity_; }
    const std::string& ordered_reference_match_identity_sha256() const noexcept { return match_identity_; }
    const std::string& localization_identity_sha256() const noexcept { return identity_; }
    double source_image_cutoff_bohr() const noexcept { return image_cutoff_bohr_; }
    // Recheck the original AO/minimal basis content, shell-to-atom map and
    // cell against this native producer's receipt. The exact admitted state
    // and allocation must also match. Callers reserve the existing
    // memory().input_verification_work_units BEFORE invoking this scan.
    // This does not itself authenticate an HF Hamiltonian: an actual HF
    // consumer must separately verify its own AO/auxiliary/cell inputs and
    // require the same AO-image policy and captured state owner.
    void verify_original_inputs(const PeriodicCorrelationAdmittedReference&,
        const BasisSet& ao, const BasisSet& minimal, const PeriodicSystem&) const;
private:
    PeriodicGaussianLocalizationResult() = default;
    std::optional<PeriodicCorrelationIAOOptimizerResult> optimizer_;
    std::optional<PeriodicCorrelationWannier> wannier_;
    PeriodicGaussianLocalizationMemoryPlan memory_;
    PeriodicGaussianLocalizationDiagnostics diagnostics_;
    std::string input_identity_, source_identity_, match_identity_, identity_;
    // Fixed-size original census only, no Gaussian arrays or borrowed input
    // pointers. Covered by the producer's fixed logical-control reservation.
    PeriodicGaussianBlochIAOMemoryPlan source_plan_;
    std::uint64_t atom_count_ = 0;
    double image_cutoff_bohr_ = 0.0;
    friend PeriodicGaussianLocalizationResult localize_periodic_gaussian_occupied(
        const PeriodicCorrelationAdmittedReference&, const BasisSet&, const BasisSet&,
        const PeriodicSystem&, std::uint64_t, const PeriodicGaussianLocalizationOptions&,
        const PeriodicGaussianBlochIAOCaps&, const PeriodicGaussianLocalizationCaps&,
        PeriodicGaussianLocalizationCallback, void*);
};

// One original immutable state/basis/minimal/cell input for the entire call.
// No supplied point-owner list, full-supercell matrix, or all-k overlap cache.
// Lower leaves report PARTIAL inventories. This enclosing driver adds all
// still-borrowed Gaussian payload and other_live for every worker replica,
// plus every simultaneously live numerical owner and logical control record.
// The original admitted reference/allocation identity is never manufactured
// or re-admitted. State/baseline is counted exactly once by its own contract.
//
// Seed-first staged admission is deliberate: the optimizer planner needs an
// actual audited seed. Each allocation phase is admitted BEFORE allocation;
// failure before/after seed returns no fabricated plan or partial result.
// Positive work caps cover conservative scalar-loop reservations, not FLOPs
// or callback user code. Callback mutation of basis/cell fails closed after
// bounded census/content checks. Native callers must keep all inputs immutable.
// Caller-owned callback storage/capacities belong in other_live/reference.
//
// Records/identity characters are separately reported from numerical arrays.
// A fixed 64 KiB reservation covers scalar/control/short-string working data;
// allocator overhead, capacity slack and Python wrapper costs are external
// backend inventory, NOT an assertion of exact process RSS.
//
// Nonconvergence preserves the optimizer's last ACCEPTED gauge/status; no
// Wannier owner is constructed. Success releases IAOs/seed BEFORE Wannier,
// but retains optimizer + Wannier outputs. Actual finite-image source receipts
// are kept separately from their numerical reference-match receipts: there is
// still no authenticated AO-basis receipt from the mean-field producer.
PeriodicGaussianLocalizationResult localize_periodic_gaussian_occupied(
    const PeriodicCorrelationAdmittedReference&, const BasisSet& ao, const BasisSet& minimal,
    const PeriodicSystem& original_system, std::uint64_t other_live_numerical_bytes,
    const PeriodicGaussianLocalizationOptions&, const PeriodicGaussianBlochIAOCaps&,
    const PeriodicGaussianLocalizationCaps&,
    PeriodicGaussianLocalizationCallback callback, void* callback_context);
}  // namespace vibeqc
