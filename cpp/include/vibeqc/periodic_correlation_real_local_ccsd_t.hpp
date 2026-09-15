#pragma once

/// Native connected CCSD + coupled (T) REFERENCE calculation for one selected
/// real local system made from finite multi-k Gaussian factors. This is not
/// the translation-weighted, pair-specific production DLPNO driver. In
/// particular, neither a per-cell energy nor matching HF/RI Hamiltonians nor
/// infinite-image convergence is certified here.
///
/// A measured, explicitly budgeted projection makes Fov exactly zero and Fvv
/// exactly diagonal, as required by Guo doi:10.1063/1.5011798 Eqs.(1)-(2).
/// Both CCSD and (T) use THAT SAME projected Fock and native real-row provider.
/// Foo, including every offdiagonal occupied coupling, is unchanged. This is
/// not T0. No caller-supplied ERI or amplitude arrays enter this connection.

#include "vibeqc/bounded_restricted_ccsd_solver.hpp"
#include "vibeqc/bounded_restricted_triples_solver.hpp"
#include "vibeqc/periodic_correlation_real_local_provider.hpp"

namespace vibeqc {

struct PeriodicCorrelationRealLocalCCSDTOptions {
    BoundedRestrictedCCSDSolverOptions ccsd;
    BoundedRestrictedTriplesSolverOptions triples;
    /// Outward Frobenius bound on the ADDITIONAL real Fock projection.
    /// Finite nonnegative, zero means require no change. The earlier complex
    /// to real basis projection retains its independent receipt/error bound.
    double maximum_additional_fock_projection_norm = 0.0;
};
struct PeriodicCorrelationRealLocalCCSDTInventory {
    /// Prior gauges, domains, reader, seeds, callback storage, etc. are NOT
    /// discoverable from the two immutable numerical input owners. Release
    /// them, account them in reference baseline, or declare them here (once).
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
};
struct PeriodicCorrelationRealLocalCCSDTCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_node_numerical_bytes = 0;
    std::uint64_t maximum_integral_calls = 0;
    /// Includes BOTH solver kernels, total scalar-provider work, Fock audit
    /// and result sealing. These are conservative units, not timings/FLOPs.
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicCorrelationRealLocalCCSDTMemoryPlan {
    std::uint64_t occupied_count = 0, virtual_count = 0;
    std::uint64_t projected_fock_bytes = 0, borrowed_basis_bytes = 0, borrowed_factor_row_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, retained_output_bytes_upper_bound = 0;
    std::uint64_t total_live_numerical_bytes = 0, required_node_numerical_bytes = 0;
    std::uint64_t numerical_replicas = 0, external_node_numerical_bytes = 0;
    std::uint64_t integral_calls_upper_bound = 0, provider_work_units_upper_bound = 0;
    std::uint64_t work_units_upper_bound = 0;
    BoundedRestrictedCCSDSolverMemoryPlan ccsd;
    BoundedRestrictedTriplesSolverMemoryPlan triples;
};
enum class PeriodicCorrelationRealLocalCCSDTStage : std::uint32_t { CCSD = 1, Triples = 2 };
struct PeriodicCorrelationRealLocalCCSDTProgress {
    PeriodicCorrelationRealLocalCCSDTStage stage = PeriodicCorrelationRealLocalCCSDTStage::CCSD;
    BoundedRestrictedCCSDSolverProgress ccsd;
    BoundedRestrictedTriplesSolverProgress triples;
};
using PeriodicCorrelationRealLocalCCSDTProgressCallback =
    void (*)(const PeriodicCorrelationRealLocalCCSDTProgress&, void*);

class PeriodicCorrelationRealLocalCCSDTResult {
public:
    PeriodicCorrelationRealLocalCCSDTResult(const PeriodicCorrelationRealLocalCCSDTResult&) = delete;
    PeriodicCorrelationRealLocalCCSDTResult& operator=(const PeriodicCorrelationRealLocalCCSDTResult&) = delete;
    PeriodicCorrelationRealLocalCCSDTResult(PeriodicCorrelationRealLocalCCSDTResult&&) noexcept = default;
    PeriodicCorrelationRealLocalCCSDTResult& operator=(PeriodicCorrelationRealLocalCCSDTResult&&) noexcept = default;
    const PeriodicCorrelationRealLocalCCSDTMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedRestrictedCCSDSolverResult& ccsd() const noexcept { return ccsd_; }
    const BoundedRestrictedTriplesSolverResult& triples() const;
    bool triples_evaluated() const noexcept { return triples_evaluated_; }
    bool converged() const noexcept { return triples_evaluated_ && triples_.final_snapshot.converged; }
    bool hf_hamiltonian_match_certified() const noexcept { return false; }
    bool periodic_energy_per_cell() const noexcept { return false; }
    double additional_fock_projection_norm_bound() const noexcept { return additional_projection_; }
    double total_fock_projection_norm_bound() const noexcept { return total_projection_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& provider_identity_sha256() const noexcept { return provider_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_; }
private:
    PeriodicCorrelationRealLocalCCSDTResult() = default;
    PeriodicCorrelationRealLocalCCSDTMemoryPlan memory_;
    BoundedRestrictedCCSDSolverResult ccsd_;
    BoundedRestrictedTriplesSolverResult triples_;
    bool triples_evaluated_ = false;
    double additional_projection_ = 0.0, total_projection_ = 0.0;
    std::string identity_, payload_, provider_, basis_;
    friend PeriodicCorrelationRealLocalCCSDTResult solve_periodic_correlation_real_local_ccsd_t(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
        const PeriodicCorrelationRealLocalProvider&, const PeriodicCorrelationRealLocalCCSDTOptions&,
        const PeriodicCorrelationRealLocalCCSDTInventory&, const PeriodicCorrelationRealLocalCCSDTCaps&,
        PeriodicCorrelationRealLocalCCSDTProgressCallback, void*);
};

/// Allocation-free plan from actual native owners, with the admitted state
/// counted once and ranks*workers replicas. All numerical controls/stack,
/// allocator, Python and OS costs need independent runtime/RSS admission.
PeriodicCorrelationRealLocalCCSDTMemoryPlan plan_periodic_correlation_real_local_ccsd_t(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicCorrelationRealLocalProvider&, const PeriodicCorrelationRealLocalCCSDTOptions&,
    const PeriodicCorrelationRealLocalCCSDTInventory&);

/// Admit BOTH stages before Fock allocation, scalar-provider use or progress.
/// Unconverged CCSD returns its last evaluated result WITHOUT attempting (T).
/// Callback/arithmetic failures throw, never publishing a partial owner.
/// Final amplitudes belong to this result; no input owner need outlive return.
PeriodicCorrelationRealLocalCCSDTResult solve_periodic_correlation_real_local_ccsd_t(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationRealLocalBasis&,
    const PeriodicCorrelationRealLocalProvider&, const PeriodicCorrelationRealLocalCCSDTOptions&,
    const PeriodicCorrelationRealLocalCCSDTInventory&, const PeriodicCorrelationRealLocalCCSDTCaps&,
    PeriodicCorrelationRealLocalCCSDTProgressCallback progress = nullptr, void* progress_context = nullptr);

} // namespace vibeqc
