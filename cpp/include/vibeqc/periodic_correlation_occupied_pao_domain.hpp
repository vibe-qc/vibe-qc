#pragma once

// Riplinger and Neese, doi:10.1063/1.4773581, Sec.II.B.1/II.B.3;
// Nejad et al., doi:10.1063/5.0290816, Eqs.25-28. One home occupied
// orbital's signed gross Mulliken atom domain, followed by ONE PAO-tail
// expansion from its original seeds. This is not the principal-domain
// algorithm, an extended CCSD domain, pair screening, or an HF source proof.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_iao_optimizer.hpp"
#include "vibeqc/periodic_correlation_wannier.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicCorrelationOccupiedPAODomainVersion = 1;

struct PeriodicCorrelationAtomCell {
    std::uint64_t cell = 0, atom = 0;
};

struct PeriodicCorrelationOccupiedPAODomainOptions {
    // Full mode deliberately selects every atom-cell and skips PAO tails.
    // Otherwise a seed satisfies SIGNED p > mulliken_population_cutoff.
    // Expansion is strict pAB > pao_tail_cutoff, not a recursive closure.
    bool full_domain = false;
    double mulliken_population_cutoff = std::numeric_limits<double>::quiet_NaN();
    double pao_tail_cutoff = std::numeric_limits<double>::quiet_NaN();
    double normalization_tolerance = std::numeric_limits<double>::quiet_NaN();
    double maximum_population_imaginary_magnitude = std::numeric_limits<double>::quiet_NaN();
    double maximum_pao_imaginary_magnitude = std::numeric_limits<double>::quiet_NaN();
    // All three population budgets are explicit finite nonnegative numbers, not
    // inferred zero/unlimited defaults. Signed populations are never clipped.
    double maximum_negative_absolute_population = std::numeric_limits<double>::quiet_NaN();
    double maximum_seed_omitted_absolute_population = std::numeric_limits<double>::quiet_NaN();
    double maximum_expanded_omitted_absolute_population = std::numeric_limits<double>::quiet_NaN();
};

struct PeriodicCorrelationOccupiedPAODomainInventory {
    // Excludes state (already admitted), optimizer gauges/active indices,
    // Wannier coefficients, AO mapping, and this leaf's logical controls.
    std::uint64_t other_live_numerical_bytes = 0;
    std::uint64_t other_live_control_bytes = 0;
    // Explicit positive allowance for allocator/container capacity, wrappers,
    // backend and runtime overhead not promised by logical payload counts.
    std::uint64_t backend_allowance_bytes = 0;
};
struct PeriodicCorrelationOccupiedPAODomainCaps {
    std::uint64_t maximum_atom_cells = 0;
    std::uint64_t maximum_seed_atom_cells = 0;
    std::uint64_t maximum_expanded_atom_cells = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    // Gates fixed + borrowed-owner + inventory.other_live_control_bytes.
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicCorrelationOccupiedPAODomainMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_atoms = 0, n_active = 0;
    std::uint64_t atom_cell_count = 0, seed_count_upper = 0, expanded_count_upper = 0;
    std::uint64_t borrowed_optimizer_numerical_bytes = 0;
    std::uint64_t borrowed_wannier_numerical_bytes = 0, caller_mapping_bytes = 0;
    std::uint64_t retained_population_bytes = 0, retained_domain_bytes_upper = 0;
    std::uint64_t coefficient_panel_bytes = 0, shared_scratch_bytes = 0;
    std::uint64_t selection_mark_bytes = 0, tail_score_bytes_upper = 0;
    std::uint64_t population_phase_owned_bytes = 0, tail_phase_owned_bytes_upper = 0;
    std::uint64_t output_phase_owned_bytes_upper = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t fixed_control_storage_bytes = 0, borrowed_owner_control_bytes = 0;
    std::uint64_t worker_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t validation_work_units = 0, population_work_units = 0;
    std::uint64_t tail_work_units_upper = 0, planned_work_units = 0;
};
struct PeriodicCorrelationOccupiedPAODomainDiagnostics {
    std::uint64_t seed_count = 0, expanded_count = 0;
    std::uint64_t retained_numerical_bytes = 0, actual_peak_owned_numerical_bytes = 0;
    std::uint64_t charged_work_units = 0, projected_pao_columns = 0;
    double population_sum = 0.0, normalization_residual = 0.0;
    double maximum_population_imaginary_magnitude = 0.0;
    double maximum_pao_imaginary_magnitude = 0.0;
    double negative_absolute_population = 0.0;
    double seed_omitted_absolute_population = 0.0;
    double expanded_omitted_absolute_population = 0.0;
    double maximum_pao_tail_strength = 0.0;
};

class PeriodicCorrelationOccupiedPAODomain {
public:
    PeriodicCorrelationOccupiedPAODomain(const PeriodicCorrelationOccupiedPAODomain&) = delete;
    PeriodicCorrelationOccupiedPAODomain& operator=(const PeriodicCorrelationOccupiedPAODomain&) = delete;
    PeriodicCorrelationOccupiedPAODomain(PeriodicCorrelationOccupiedPAODomain&&) noexcept = default;
    PeriodicCorrelationOccupiedPAODomain& operator=(PeriodicCorrelationOccupiedPAODomain&&) noexcept = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationOccupiedPAODomainVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    std::uint64_t home_occupied_index() const noexcept { return occupied_; }
    const PeriodicCorrelationOccupiedPAODomainOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationOccupiedPAODomainMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationOccupiedPAODomainDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    bool hf_basis_source_authenticated() const noexcept { return false; }
    bool extended_ccsd_domain() const noexcept { return false; }
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }
    const std::string& optimizer_identity_sha256() const noexcept { return optimizer_identity_; }
    const std::string& wannier_identity_sha256() const noexcept { return wannier_identity_; }
    const std::string& mapping_identity_sha256() const noexcept { return mapping_identity_; }
    const std::string& payload_identity_sha256() const noexcept { return payload_identity_; }
    const std::string& occupied_pao_domain_identity_sha256() const noexcept { return identity_; }
    double population(std::size_t cell, std::size_t atom) const;
    PeriodicCorrelationAtomCell seed(std::size_t index) const;
    PeriodicCorrelationAtomCell expanded(std::size_t index) const;
    // Const compact rows, sorted cell-major then atom-major. No full-torus
    // projector, AO matrix, translated-orbital matrix, or mutable view.
    const double* populations_data() const;
    const PeriodicCorrelationAtomCell* seeds_data() const;
    const PeriodicCorrelationAtomCell* expanded_data() const;
private:
    PeriodicCorrelationOccupiedPAODomain() = default;
    void require_live() const;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::uint64_t occupied_ = 0;
    PeriodicCorrelationOccupiedPAODomainOptions options_;
    PeriodicCorrelationOccupiedPAODomainMemoryPlan memory_;
    PeriodicCorrelationOccupiedPAODomainDiagnostics diagnostics_;
    std::vector<double> populations_;
    std::vector<PeriodicCorrelationAtomCell> seeds_, expanded_;
    std::string allocation_identity_, optimizer_identity_, wannier_identity_;
    std::string mapping_identity_, payload_identity_, identity_;
    friend PeriodicCorrelationOccupiedPAODomain select_periodic_correlation_occupied_pao_domain(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationIAOOptimizerResult&,
        const PeriodicCorrelationWannier&, const std::uint64_t*, std::size_t,
        std::uint64_t, std::uint64_t, const PeriodicCorrelationOccupiedPAODomainOptions&,
        const PeriodicCorrelationOccupiedPAODomainInventory&,
        const PeriodicCorrelationOccupiedPAODomainCaps&);
};

// Metadata-only, allocation-free numerical planning. Uses capped seed/output
// upper counts; does not read mapping/gauges/Wannier/state numerical payloads.
// Exact byte/work/candidate overflows and node/worker caps precede scans.
PeriodicCorrelationOccupiedPAODomainMemoryPlan plan_periodic_correlation_occupied_pao_domain(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationIAOOptimizerResult&,
    const PeriodicCorrelationWannier&, std::uint64_t atom_count,
    const PeriodicCorrelationOccupiedPAODomainOptions&,
    const PeriodicCorrelationOccupiedPAODomainInventory&,
    const PeriodicCorrelationOccupiedPAODomainCaps&);

// Exact Gamma mesh only. Genuine converged optimizer + its own Wannier must
// name the SAME immutable state/allocation/gauge payload, not equal-content
// replacement owners. The caller mapping has exactly nao uint64 entries,
// every entry < atom_count, and every atom has at least one AO. Root physical
// wrappers must authenticate it against actual BasisSet atom labels.
//
// beta[R,mu]=W0[R,mu,i], eta=(1/K)sum_k exp(+ikR)S(k)C_active(k)U_i(k).
// p[A,R]=Re sum_mu_in_A conj(beta)*eta, with no spin factor. PAO columns
// L[R,nu;mu]=(1/K)sum_k exp(+ikR)[Cvirt Cvirt^H S]nu,mu use ONLY retained
// virtuals, excluding all occupied (including frozen) and discarded SCF
// directions. An ORIGINAL seed(row cellA,atomA) extends to column(cellB,atomB)
// iff sum_{nu in A,mu in B}|L[cellA-cellB,nu;mu]| > cutoff. The direction
// matters: this AO metric projector is generally not an ordinary Hermitian
// matrix. Complex lanes are measured against explicit gates, not silently
// discarded or repaired. A thresholded empty domain is valid.
//
// One Kn complex panel and one K*seed_count double score/compensation panel
// are reused. No quadratic full AO projector is allocated. All controls are
// copied before scans. Inputs must remain immutable throughout; receipts and
// gauge/mapping payloads are rechecked after construction, not a promise of
// safety against unsynchronized concurrent writes. No user callbacks.
PeriodicCorrelationOccupiedPAODomain select_periodic_correlation_occupied_pao_domain(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationIAOOptimizerResult&,
    const PeriodicCorrelationWannier&, const std::uint64_t* ao_to_atom,
    std::size_t accessible_mapping_count, std::uint64_t atom_count,
    std::uint64_t home_active_occupied_index,
    const PeriodicCorrelationOccupiedPAODomainOptions&,
    const PeriodicCorrelationOccupiedPAODomainInventory&,
    const PeriodicCorrelationOccupiedPAODomainCaps&);
}  // namespace vibeqc
