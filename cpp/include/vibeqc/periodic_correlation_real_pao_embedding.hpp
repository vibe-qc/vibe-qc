#pragma once

// Bounded physical common-to-pair PAO embedding. Nejad (2025),
// doi:10.1063/5.0290816, Eqs.(25)-(28),(42), and Riplinger/Neese (2013),
// doi:10.1063/1.4773581, Sec.II.B.3. This is a reference geometry bridge,
// not automatic domain selection or a removal of the common-factor bottleneck.
#include "vibeqc/periodic_correlation_pao_overlap.hpp"
#include "vibeqc/periodic_correlation_real_pao_space.hpp"
#include "vibeqc/periodic_correlation_real_local_basis.hpp"

namespace vibeqc {

struct PeriodicCorrelationRealPAOEmbeddingOptions {
    // All budgets are finite strictly positive and explicit. They are
    // Frobenius norms, except containment's physical Hilbert-space norm.
    double maximum_cross_overlap_imaginary_norm = 0.0;
    double maximum_embedding_gram_error = 0.0;
    double maximum_pair_metric_error = 0.0;
    double maximum_containment_norm = 0.0;
    // Fock budgets are in Hartree matrix units. Original physical F(k)
    // is inspected, never replaced by either owner's semicanonical eps.
    double maximum_pair_fock_error = 0.0;
    double maximum_common_fock_error = 0.0;
    double maximum_embedded_fock_error = 0.0;
};
struct PeriodicCorrelationRealPAOEmbeddingLiveInventory {
    std::uint64_t other_live_numerical_bytes_per_worker = 0;
    std::uint64_t other_live_control_bytes_per_worker = 0;
    std::uint64_t fixed_backend_margin_bytes_per_worker = 0;
};
struct PeriodicCorrelationRealPAOEmbeddingCaps {
    std::uint64_t maximum_common_dimension = 0, maximum_pair_dimension = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes_per_worker = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicCorrelationRealPAOEmbeddingMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0, common_dimension = 0, pair_dimension = 0;
    std::uint64_t unique_domain_owners = 0, unique_space_owners = 0, unique_real_wrapper_owners = 0;
    std::uint64_t live_domain_bytes = 0, live_space_bytes = 0, live_basis_bytes = 0;
    std::uint64_t complete_borrowed_numerical_bytes = 0;
    std::uint64_t output_numerical_bytes = 0;
    std::uint64_t overlap_phase_bytes = 0, conversion_phase_bytes = 0, physical_audit_phase_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t replicas_per_node = 0, reference_base_node_bytes = 0;
    std::uint64_t per_worker_inventoried_bytes = 0, required_node_memory_bytes = 0;
    std::uint64_t source_payload_validation_work_units = 0, physical_audit_work_units = 0;
    std::uint64_t work_units = 0;
    // Child inventory is only a subset. The enclosing gate above counts
    // all live owners/control/backend against the ORIGINAL reference.
    PeriodicCorrelationPAOOverlapMemoryPlan overlap;
};
struct PeriodicCorrelationRealPAOEmbeddingDiagnostics {
    double cross_overlap_imaginary_frobenius_upper_bound = 0.0;
    double embedding_gram_frobenius_upper_bound = 0.0;
    double pair_metric_frobenius_upper_bound = 0.0;
    // Signed compensated trace of e^H S e, e=Vp-Vc X. Never clipped.
    double containment_metric_quadratic_real = 0.0;
    double containment_metric_quadratic_imaginary = 0.0;
    // sqrt[(1/K) sum_k,a |e_a(k)|^T |S(k)| |e_a(k)|], evaluated
    // outward for the represented reconstructed columns. This conservative
    // bound, NOT a clipped quadratic, controls containment admission.
    double containment_s_absolute_upper_bound = 0.0;
    double pair_original_fock_frobenius_upper_bound = 0.0;
    double common_projected_fock_frobenius_upper_bound = 0.0;
    double embedded_original_fock_frobenius_upper_bound = 0.0;
};

class PeriodicCorrelationRealPAOEmbedding {
public:
    PeriodicCorrelationRealPAOEmbedding(const PeriodicCorrelationRealPAOEmbedding&) = delete;
    PeriodicCorrelationRealPAOEmbedding& operator=(const PeriodicCorrelationRealPAOEmbedding&) = delete;
    PeriodicCorrelationRealPAOEmbedding(PeriodicCorrelationRealPAOEmbedding&&) noexcept = default;
    PeriodicCorrelationRealPAOEmbedding& operator=(PeriodicCorrelationRealPAOEmbedding&&) noexcept = delete;
    const PeriodicCorrelationRealPAOEmbeddingMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationRealPAOEmbeddingDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const PeriodicCorrelationRealPAOEmbeddingOptions& options() const noexcept { return options_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    PeriodicCorrelationVirtualBlockSelection common_selection() const noexcept { return common_; }
    PeriodicCorrelationVirtualBlockSelection pair_selection() const noexcept { return pair_; }
    // Row-major X[common,pair] and copied pair energies. No repair,
    // renormalization, recanonicalization or borrowed numeric output.
    const double* coefficients_data() const;
    const double* energies_data() const;
    double coefficient(std::size_t common, std::size_t pair) const;
    double energy(std::size_t pair) const;
    const std::string& common_basis_identity_sha256() const noexcept { return basis_; }
    const std::string& common_frame_identity_sha256() const noexcept { return common_frame_; }
    const std::string& pair_frame_identity_sha256() const noexcept { return pair_frame_; }
    const std::string& overlap_identity_sha256() const noexcept { return overlap_; }
    const std::string& source_payload_receipt_sha256() const noexcept { return sources_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& allocation_identity() const noexcept { return allocation_; }
    bool production_distinct_domain_scaling() const noexcept { return false; }
private:
    PeriodicCorrelationRealPAOEmbedding() = default;
    void require_live() const;
    PeriodicCorrelationRealPAOEmbeddingMemoryPlan memory_;
    PeriodicCorrelationRealPAOEmbeddingDiagnostics diagnostics_;
    PeriodicCorrelationRealPAOEmbeddingOptions options_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationVirtualBlockSelection common_, pair_;
    std::vector<double> coefficients_, energies_;
    std::string basis_, common_frame_, pair_frame_, overlap_, sources_, payload_, identity_, allocation_;
    friend PeriodicCorrelationRealPAOEmbedding make_periodic_correlation_real_pao_embedding(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicCorrelationPAODomain&, const PeriodicCorrelationRealPAOSpace&,
        const PeriodicCorrelationVirtualBlockSelection&, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationRealPAOSpace&, const PeriodicCorrelationVirtualBlockSelection&,
        const PeriodicCorrelationRealPAOEmbeddingOptions&, const PeriodicCorrelationRealPAOEmbeddingLiveInventory&,
        const PeriodicCorrelationRealPAOEmbeddingCaps&);
};

// Mandatory native common-basis receipts establish the actual virtual frame,
// not a caller label. Positive dimensions only: an empty/consumed real PAO
// owner is not a usable initial pair space. Equal hashes never deduplicate
// distinct live domain/compact-space objects. No callbacks or source caches.
PeriodicCorrelationRealPAOEmbeddingMemoryPlan plan_periodic_correlation_real_pao_embedding(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationRealPAOSpace&,
    const PeriodicCorrelationVirtualBlockSelection&, const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationRealPAOSpace&, const PeriodicCorrelationVirtualBlockSelection&,
    const PeriodicCorrelationRealPAOEmbeddingOptions&, const PeriodicCorrelationRealPAOEmbeddingLiveInventory&,
    const PeriodicCorrelationRealPAOEmbeddingCaps&);
PeriodicCorrelationRealPAOEmbedding make_periodic_correlation_real_pao_embedding(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationRealPAOSpace&,
    const PeriodicCorrelationVirtualBlockSelection&, const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationRealPAOSpace&, const PeriodicCorrelationVirtualBlockSelection&,
    const PeriodicCorrelationRealPAOEmbeddingOptions&, const PeriodicCorrelationRealPAOEmbeddingLiveInventory&,
    const PeriodicCorrelationRealPAOEmbeddingCaps&);

} // namespace vibeqc
