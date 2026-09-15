#pragma once

// Initial pair PAO domain: union of two translated occupied atom domains,
// Riplinger/Neese doi:10.1063/1.4773581 Sec.II.B.1; periodic PAOs and exact
// torus translations: Nejad et al. doi:10.1063/5.0290816 Eqs.25-28/42-44.
// This is NOT a CCSD extended domain, PNO selection or an HF source proof.

#include "vibeqc/periodic_correlation_occupied_pao_domain.hpp"
#include "vibeqc/periodic_correlation_pair_topology.hpp"
#include "vibeqc/periodic_correlation_pao_domain.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicCorrelationPairPAODomainVersion = 1;

struct PeriodicCorrelationPairPAODomainInventory {
    // Exclude admitted state, exact distinct endpoint domain owners, topology
    // rows, caller AO mapping and their logical controls; all counted here.
    std::uint64_t other_live_numerical_bytes = 0, other_live_control_bytes = 0;
    std::uint64_t backend_allowance_bytes = 0;  // Explicit positive capacity/runtime allowance.
};
struct PeriodicCorrelationPairPAODomainCaps {
    std::uint64_t maximum_atom_cells = 0, maximum_union_atom_cells = 0;
    std::uint64_t maximum_ao_columns = 0, maximum_topology_rows = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_control_storage_bytes = 0;
    std::uint64_t maximum_worker_bytes = 0, maximum_work_units = 0;
};
struct PeriodicCorrelationPairPAODomainMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_atoms = 0, atom_cell_count = 0;
    std::uint64_t union_atom_count_upper = 0, ao_column_count_upper = 0;
    std::uint64_t distinct_domain_owners = 0, borrowed_domain_numerical_bytes = 0;
    std::uint64_t borrowed_topology_row_bytes = 0, caller_mapping_bytes = 0;
    std::uint64_t mark_bytes = 0, retained_atom_bytes_upper = 0, retained_ao_bytes_upper = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, fixed_control_storage_bytes = 0;
    std::uint64_t borrowed_owner_control_bytes = 0, worker_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t validation_work_units = 0, union_work_units = 0, planned_work_units = 0;
};
struct PeriodicCorrelationPairPAODomainDiagnostics {
    std::uint64_t union_atom_count = 0, ao_column_count = 0;
    std::uint64_t retained_numerical_bytes = 0, actual_peak_owned_numerical_bytes = 0;
    std::uint64_t charged_work_units = 0;
};

class PeriodicCorrelationPairPAODomain {
public:
    PeriodicCorrelationPairPAODomain(const PeriodicCorrelationPairPAODomain&) = delete;
    PeriodicCorrelationPairPAODomain& operator=(const PeriodicCorrelationPairPAODomain&) = delete;
    PeriodicCorrelationPairPAODomain(PeriodicCorrelationPairPAODomain&&) noexcept = default;
    PeriodicCorrelationPairPAODomain& operator=(PeriodicCorrelationPairPAODomain&&) noexcept = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationPairPAODomainVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    std::uint64_t row_index() const noexcept { return row_index_; }
    PeriodicCorrelationTranslationPair row() const noexcept { return row_; }
    std::uint64_t energy_weight() const noexcept { return energy_weight_; }
    const PeriodicCorrelationPairPAODomainMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationPairPAODomainDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    bool extended_ccsd_domain() const noexcept { return false; }
    bool hf_basis_source_authenticated() const noexcept { return false; }
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }
    const std::string& topology_identity_sha256() const noexcept { return topology_identity_; }
    const std::string& row_identity_sha256() const noexcept { return row_identity_; }
    const std::string& home_domain_identity_sha256() const noexcept { return home_identity_; }
    const std::string& partner_domain_identity_sha256() const noexcept { return partner_identity_; }
    const std::string& mapping_identity_sha256() const noexcept { return mapping_identity_; }
    const std::string& payload_identity_sha256() const noexcept { return payload_identity_; }
    const std::string& pair_pao_domain_identity_sha256() const noexcept { return identity_; }
    PeriodicCorrelationAtomCell atom(std::size_t index) const;
    PeriodicPAODomainColumn column(std::size_t index) const;
    // Translate every column by an existing placed-pair resolver's common
    // translation. The scalar ordering is retained, not resorted; its AO set
    // stays unique. The resolver's transpose flag concerns amplitude axes.
    PeriodicPAODomainColumn translated_column(std::size_t index, std::uint64_t common_translation_cell) const;
    const PeriodicCorrelationAtomCell* atom_cells_data() const;
    // Exactly 2*ao_column_count aligned uint64 elements. Borrow directly into
    // make_periodic_correlation_pao_domain; no struct pointer type-punning.
    const std::uint64_t* cell_ao_indices_data() const;
private:
    PeriodicCorrelationPairPAODomain() = default;
    void require_live() const;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::uint64_t row_index_ = 0, energy_weight_ = 0;
    PeriodicCorrelationTranslationPair row_;
    PeriodicCorrelationPairPAODomainMemoryPlan memory_;
    PeriodicCorrelationPairPAODomainDiagnostics diagnostics_;
    std::vector<PeriodicCorrelationAtomCell> atoms_;
    std::vector<std::uint64_t> columns_;
    std::string allocation_identity_, topology_identity_, row_identity_, home_identity_, partner_identity_;
    std::string mapping_identity_, payload_identity_, identity_;
    friend PeriodicCorrelationPairPAODomain make_periodic_correlation_pair_pao_domain(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationTranslationPairTopology&,
        std::uint64_t, const PeriodicCorrelationOccupiedPAODomain&, const PeriodicCorrelationOccupiedPAODomain&,
        const std::uint64_t*, std::size_t, std::uint64_t,
        const PeriodicCorrelationPairPAODomainInventory&, const PeriodicCorrelationPairPAODomainCaps&);
};

// Metadata-only count/byte/work admission before any domain, mapping or full
// topology scan. One byte per atom-cell is the ONLY temporary numeric array.
// Borrowed endpoint payloads/controls deduplicate by C++ OWNER ADDRESS, never
// by numerical hash. Distinct equal-content owners count twice. State counts
// once in the admitted reference; no optimizer/Wannier owner is retained by
// the endpoints. Other live localizers remain the caller's inventory duty.
PeriodicCorrelationPairPAODomainMemoryPlan plan_periodic_correlation_pair_pao_domain(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationTranslationPairTopology&,
    std::uint64_t row_index, const PeriodicCorrelationOccupiedPAODomain& home,
    const PeriodicCorrelationOccupiedPAODomain& partner, std::uint64_t atom_count,
    const PeriodicCorrelationPairPAODomainInventory&, const PeriodicCorrelationPairPAODomainCaps&);

// Native endpoints must match the canonical row's active labels and exact
// state/allocation/global optimizer/Wannier receipts. Different occupied
// orbitals may use different scientific cuts; BOTH endpoint identities are
// sealed. A same-orbital row requires identical domain identities to preserve
// exchange/translation covariance, especially at nonzero self-inverse cells.
// Equal-content distinct owners are allowed, but not budget-deduplicated.
//
// Union = home.expanded UNION [partner.expanded + row.L] modulo the mesh.
// Expand each selected atom to ALL its mapped AOs. Sorted unique atom and AO
// rows use last-axis-fast cell indices. Empty union is valid; no forced PAO.
// Map is aligned uint64[nao], with an identity matching BOTH native endpoint
// receipts. Full source payload/topology/map hashes are audited before/after.
// No callbacks; caller inputs must remain immutable, not concurrently written.
// The returned owner retains only state and compact outputs, not source owners.
PeriodicCorrelationPairPAODomain make_periodic_correlation_pair_pao_domain(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationTranslationPairTopology&,
    std::uint64_t row_index, const PeriodicCorrelationOccupiedPAODomain& home,
    const PeriodicCorrelationOccupiedPAODomain& partner, const std::uint64_t* ao_to_atom,
    std::size_t accessible_mapping_count, std::uint64_t atom_count,
    const PeriodicCorrelationPairPAODomainInventory&, const PeriodicCorrelationPairPAODomainCaps&);
}  // namespace vibeqc
