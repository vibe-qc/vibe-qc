#pragma once

// Coupled real pair-MP2 REFERENCE algebra, Nejad et al., JCP 163,214107
// (2025), doi:10.1063/5.0290816, Eqs.(24),(37),(42)-(44). All unordered
// occupied pairs are included; reverse amplitudes are derived by transpose.
// No torus, Eq.(39) PNO provenance, Hamiltonian or periodic-energy certificate
// is inferred from numerical input views. The physical adapter owns those.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace vibeqc {

struct BoundedRestrictedPairMP2RealView {
    const double* data = nullptr;
    std::size_t element_count = 0;
};
struct BoundedRestrictedPairMP2PairView {
    std::uint64_t rank = 0;
    BoundedRestrictedPairMP2RealView integrals;        // G[r,r], row major
    BoundedRestrictedPairMP2RealView virtual_energies; // eps[r]
    BoundedRestrictedPairMP2RealView coefficients;     // C[common_n,r]
};
struct BoundedRestrictedPairMP2SolverInput {
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0;
    BoundedRestrictedPairMP2RealView occupied_fock;    // exact symmetric [o,o]
    // Exact table count o(o+1)/2, lexicographic i=0..o-1,j=i..o-1.
    // All views remain immutable until return; aliases are counted by role.
    const BoundedRestrictedPairMP2PairView* pairs = nullptr;
    std::size_t pair_count = 0;
};
struct BoundedRestrictedPairMP2SolverOptions {
    std::uint64_t maximum_iterations = 0; // evaluated snapshots, initial included
    double denominator_floor = 0.0;
    double residual_tolerance = 0.0, energy_tolerance = 0.0;
    double coefficient_orthogonality_tolerance = 0.0; // Explicit finite 0 < tolerance < 1.
    // Explicit finite nonnegative bound on the Frobenius norm discarded
    // when diagonal-pair candidate T is symmetrized. RAW full residuals
    // determine convergence, never their symmetric projection.
    double maximum_diagonal_update_antisymmetry_norm = std::numeric_limits<double>::quiet_NaN();
};
struct BoundedRestrictedPairMP2SolverInventory {
    std::uint64_t numerical_replicas = 0;
    std::uint64_t external_node_bytes = 0;
    std::uint64_t other_live_bytes_per_replica = 0;
    std::uint64_t fixed_backend_margin_bytes_per_replica = 0;
};
struct BoundedRestrictedPairMP2SolverCaps {
    std::uint64_t maximum_occupied_count = 0, maximum_common_virtual_dimension = 0;
    std::uint64_t maximum_pair_count = 0, maximum_pair_rank = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_coupling_slots = 0, maximum_work_units = 0;
};
struct BoundedRestrictedPairMP2SolverMemoryPlan {
    bool uniform_rank_upper_bound = false;
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0, pair_count = 0;
    std::uint64_t maximum_pair_rank = 0, maximum_iterations = 0, total_amplitude_elements = 0;
    std::uint64_t amplitude_snapshot_bytes = 0, candidate_snapshot_bytes = 0;
    std::uint64_t retained_pair_record_bytes = 0, borrowed_pair_table_bytes = 0;
    std::uint64_t borrowed_fock_bytes = 0, borrowed_pair_numeric_bytes = 0;
    std::uint64_t output_numerical_bytes = 0, initialization_phase_owned_bytes = 0;
    std::uint64_t residual_workspace_bytes = 0, overlap_workspace_bytes = 0;
    std::uint64_t iteration_phase_owned_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t fixed_control_storage_bytes = 0, per_replica_inventoried_bytes = 0;
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t metadata_work_units = 0, validation_work_units = 0;
    std::uint64_t initialization_work_units = 0, work_units_per_snapshot = 0, work_units_upper_bound = 0;
    std::uint64_t target_evaluations_upper_bound = 0, coupling_slots_upper_bound = 0;
    std::uint64_t overlap_scalar_products_upper_bound = 0, residual_scalar_products_upper_bound = 0;
};
struct BoundedRestrictedPairMP2SolverProgress {
    std::uint64_t iteration = 0, target_evaluations = 0, coupling_slots = 0;
    std::uint64_t transposed_sources = 0, zero_rank_sources = 0, zero_rank_targets = 0;
    std::uint64_t overlap_scalar_products = 0, residual_scalar_products = 0;
    std::uint64_t input_checks = 0, charged_work_units = 0, arithmetic_underflow_count = 0;
    bool has_previous_energy = false, converged = false;
    double correlation_energy = 0.0, energy_change = 0.0;
    double maximum_absolute_residual = 0.0, residual_frobenius_norm = 0.0;
    double maximum_diagonal_update_antisymmetry_norm = 0.0;
};
using BoundedRestrictedPairMP2SolverCallback = void (*)(const BoundedRestrictedPairMP2SolverProgress&, void*);

class BoundedRestrictedPairMP2SolverResult {
public:
    BoundedRestrictedPairMP2SolverResult(const BoundedRestrictedPairMP2SolverResult&) = delete;
    BoundedRestrictedPairMP2SolverResult& operator=(const BoundedRestrictedPairMP2SolverResult&) = delete;
    BoundedRestrictedPairMP2SolverResult(BoundedRestrictedPairMP2SolverResult&&) noexcept = default;
    BoundedRestrictedPairMP2SolverResult& operator=(BoundedRestrictedPairMP2SolverResult&&) noexcept = delete;
    const BoundedRestrictedPairMP2SolverMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedRestrictedPairMP2SolverProgress& final_snapshot() const noexcept { return snapshot_; }
    double minimum_denominator() const noexcept { return minimum_denominator_; }
    double maximum_denominator() const noexcept { return maximum_denominator_; }
    std::uint64_t pair_rank(std::uint64_t i, std::uint64_t j) const;
    double amplitude(std::uint64_t i, std::uint64_t j, std::uint64_t a, std::uint64_t b) const;
    // Native-only immutable borrow for a downstream solver. Canonical i<=j
    // only: reverse-pair transpose is not a contiguous borrowed view. The
    // result owner must remain alive and unmoved until the consumer returns.
    BoundedRestrictedPairMP2RealView stored_amplitudes_view(std::uint64_t i, std::uint64_t j) const;
    const std::string& input_identity_sha256() const noexcept { return input_identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
private:
    struct PairRecord { std::uint64_t rank = 0, offset = 0; };
    BoundedRestrictedPairMP2SolverResult() = default;
    BoundedRestrictedPairMP2SolverMemoryPlan memory_;
    BoundedRestrictedPairMP2SolverProgress snapshot_;
    double minimum_denominator_ = 0.0, maximum_denominator_ = 0.0;
    std::vector<PairRecord> records_;
    std::vector<double> amplitudes_;
    std::string input_identity_, payload_;
    friend BoundedRestrictedPairMP2SolverResult bounded_restricted_pair_mp2_solve(
        const BoundedRestrictedPairMP2SolverInput&, const BoundedRestrictedPairMP2SolverOptions&,
        const BoundedRestrictedPairMP2SolverInventory&, const BoundedRestrictedPairMP2SolverCaps&,
        BoundedRestrictedPairMP2SolverCallback, void*);
};

// Allocation-free uniform-rank upper bound, no views or invented pair owners.
// Requires o>0,common_n>0,0<=max_rank<=common_n,iterations>0. An all-zero
// rank table is valid and has zero numerical amplitudes/energy.
BoundedRestrictedPairMP2SolverMemoryPlan plan_bounded_restricted_pair_mp2_solver_upper(
    std::uint64_t n_occupied, std::uint64_t common_virtual_dimension,
    std::uint64_t maximum_pair_rank, std::uint64_t maximum_iterations,
    const BoundedRestrictedPairMP2SolverInventory&);

// Caps cover metadata traversal before reading the pair table, then the
// exact ragged numerical/storage/work inventory before any floating scan.
BoundedRestrictedPairMP2SolverMemoryPlan plan_bounded_restricted_pair_mp2_solver(
    const BoundedRestrictedPairMP2SolverInput&, const BoundedRestrictedPairMP2SolverOptions&,
    const BoundedRestrictedPairMP2SolverInventory&, const BoundedRestrictedPairMP2SolverCaps&);

// No input projection: Foo and diagonal G must be exactly symmetric; C
// columns are independently checked orthonormal. eps defines each local
// semicanonical operator; its physical/original-F provenance is upstream.
// Initial semicanonical T=-G/Delta. Coupling slots are all visited in strict
// (leg,k) order, even exact-zero F entries and omitted rank-zero spaces.
// Energy is sum_i<=j (2-delta_ij) sum_ab(2Gab-Gba)Tab, no cell weight or
// truncation correction. Convergence requires raw residual AND energy change
// at one complete snapshot, including >=2 evaluated snapshots. Exhaustion
// returns the last evaluated state; callback/arithmetic errors publish none.
BoundedRestrictedPairMP2SolverResult bounded_restricted_pair_mp2_solve(
    const BoundedRestrictedPairMP2SolverInput&, const BoundedRestrictedPairMP2SolverOptions&,
    const BoundedRestrictedPairMP2SolverInventory&, const BoundedRestrictedPairMP2SolverCaps&,
    BoundedRestrictedPairMP2SolverCallback progress = nullptr, void* progress_context = nullptr);

} // namespace vibeqc
