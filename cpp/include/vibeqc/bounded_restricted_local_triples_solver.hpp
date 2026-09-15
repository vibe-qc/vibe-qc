#pragma once

// Ragged, occupied-coupled restricted triples Jacobi solver. Guo et al.,
// doi:10.1063/1.5011798 Eqs.1-3. Each unordered occupied MULTISET has its
// own orthonormal, quasi-canonical virtual frame. All occupied couplings
// remain present; reverse triples permute all three virtual axes. This is
// a numerical solver, not a physical moment producer or a DLPNO certificate.
#include <array>
#include <limits>
#include <string>
#include <vector>
#include "vibeqc/bounded_restricted_triples_target.hpp"

namespace vibeqc {

struct BoundedRestrictedLocalTriplesSpaceView {
    std::uint64_t rank = 0;
    BoundedRestrictedTriplesRealView coefficients; // common_n by rank
    BoundedRestrictedTriplesRealView energies; // rank, ascending not required
};
struct BoundedRestrictedLocalTriplesSolverInput {
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0;
    // Exactly o(o+1)(o+2)/6 entries, lexicographic i<=j<=k. Rank zero is
    // a valid empty retained space, not a missing neighbour.
    const BoundedRestrictedLocalTriplesSpaceView* spaces = nullptr;
    std::size_t space_count = 0;
    BoundedRestrictedTriplesRealView f_oo, f_vv; // ORIGINAL symmetric operators
    std::uint64_t ccsd_snapshot_id = 0; // caller sequencing, not provenance
};
struct BoundedRestrictedLocalTriplesMomentRequest {
    std::array<std::uint64_t,3> occupied{0,0,0}; // canonical multiset
    std::uint64_t rank = 0, ccsd_snapshot_id = 0;
};
struct BoundedRestrictedLocalTriplesMomentView {
    BoundedRestrictedLocalTriplesMomentRequest request;
    BoundedRestrictedTriplesRealView connected, singles; // W,U rank^3
};
using BoundedRestrictedLocalTriplesMomentReceiver = void (*)(
    const BoundedRestrictedLocalTriplesMomentView&,void*);
struct BoundedRestrictedLocalTriplesMomentProvider {
    // Exactly one synchronous receiver call. Views live throughout that
    // call. Do not swallow receiver failures or retain receiver/context.
    // Each visit must produce identical moments for its CCSD snapshot and
    // frame. The solver hashes moments across iterations to enforce this.
    void (*visit)(const BoundedRestrictedLocalTriplesMomentRequest&,
        BoundedRestrictedLocalTriplesMomentReceiver,void*,void*) = nullptr;
    void* context = nullptr;
    // Inclusive: retained plus transient inventory includes active W/U
    // views and producer scratch. W/U is NOT counted a second time.
    std::uint64_t retained_numerical_bytes = 0, maximum_transient_numerical_bytes = 0;
    std::uint64_t maximum_work_units_per_visit = 0;
};
struct BoundedRestrictedLocalTriplesSolverOptions {
    std::uint64_t maximum_iterations = 0;
    double denominator_floor = 0, residual_tolerance = 0, energy_tolerance = 0;
    double coefficient_orthogonality_tolerance = 0;
    // ||C^T ORIGINAL Fvv C-diag(eps)||_F, explicit physical Hartree budget.
    double maximum_projected_fock_error = std::numeric_limits<double>::quiet_NaN();
    // Repeated occupied axes must have permutation-covariant W and U.
    // The raw moments/residual are NOT repaired. Each moment's defect norm
    // is measured against its stabilizer average and bounded separately.
    double maximum_repeated_moment_defect_norm = std::numeric_limits<double>::quiet_NaN();
    // Cumulative Frobenius norm of candidate projection onto repeated-axis
    // symmetry. Applied ONLY when another Jacobi step is actually needed.
    double maximum_repeated_update_defect_norm = std::numeric_limits<double>::quiet_NaN();
};
struct BoundedRestrictedLocalTriplesSolverInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t other_live_bytes_per_replica = 0, fixed_backend_margin_bytes_per_replica = 0;
};
struct BoundedRestrictedLocalTriplesSolverCaps {
    std::uint64_t maximum_occupied_count = 0, maximum_common_virtual_dimension = 0;
    std::uint64_t maximum_triple_count = 0, maximum_rank = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_moment_visits = 0, maximum_work_units = 0;
};
struct BoundedRestrictedLocalTriplesSolverMemoryPlan {
    bool uniform_rank_upper_bound = false;
    std::uint64_t n_occupied = 0, common_virtual_dimension = 0, triple_count = 0;
    std::uint64_t nonempty_triple_count = 0, maximum_rank = 0, maximum_iterations = 0;
    std::uint64_t total_amplitude_elements = 0, total_rank = 0;
    std::uint64_t amplitude_snapshot_bytes = 0, candidate_snapshot_bytes = 0;
    std::uint64_t retained_record_bytes = 0, copied_space_table_bytes = 0, borrowed_space_table_bytes = 0;
    std::uint64_t borrowed_numerical_bytes = 0, neighbour_workspace_bytes = 0, occupied_row_bytes = 0;
    std::uint64_t target_owned_peak_bytes = 0, moment_digest_bytes = 0;
    std::uint64_t output_numerical_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t moment_visits_upper_bound = 0, neighbour_visits_upper_bound = 0;
    std::uint64_t validation_work_units = 0, work_units_upper_bound = 0;
};
struct BoundedRestrictedLocalTriplesSolverProgress {
    std::uint64_t iteration = 0, moment_visits = 0, neighbour_visits = 0;
    bool has_previous_energy = false, converged = false;
    double triples_energy = 0, energy_change = 0;
    // Max is unweighted. Frobenius norm includes all ORDERED occupied
    // triples: canonical block squared norms carry multiplicities 1/3/6.
    double maximum_absolute_residual = 0, residual_frobenius_norm = 0;
    double repeated_update_defect_norm = 0, maximum_repeated_moment_defect_norm = 0;
};
using BoundedRestrictedLocalTriplesSolverCallback = void (*)(
    const BoundedRestrictedLocalTriplesSolverProgress&,void*);
class BoundedRestrictedLocalTriplesSolverResult {
public:
    BoundedRestrictedLocalTriplesSolverResult(const BoundedRestrictedLocalTriplesSolverResult&) = delete;
    BoundedRestrictedLocalTriplesSolverResult& operator=(const BoundedRestrictedLocalTriplesSolverResult&) = delete;
    BoundedRestrictedLocalTriplesSolverResult(BoundedRestrictedLocalTriplesSolverResult&&) noexcept = default;
    BoundedRestrictedLocalTriplesSolverResult& operator=(BoundedRestrictedLocalTriplesSolverResult&&) noexcept = delete;
    const BoundedRestrictedLocalTriplesSolverMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedRestrictedLocalTriplesSolverProgress& final_snapshot() const noexcept { return snapshot_; }
    double minimum_denominator() const noexcept { return minimum_denominator_; }
    double maximum_denominator() const noexcept { return maximum_denominator_; }
    std::uint64_t rank(std::uint64_t i,std::uint64_t j,std::uint64_t k) const;
    double amplitude(std::uint64_t i,std::uint64_t j,std::uint64_t k,
        std::uint64_t a,std::uint64_t b,std::uint64_t c) const;
    const std::string& input_identity_sha256() const noexcept { return input_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
private:
    struct Record { std::uint64_t rank = 0, offset = 0; };
    BoundedRestrictedLocalTriplesSolverResult() = default;
    BoundedRestrictedLocalTriplesSolverMemoryPlan memory_;
    BoundedRestrictedLocalTriplesSolverProgress snapshot_;
    double minimum_denominator_ = 0, maximum_denominator_ = 0;
    std::vector<Record> records_;
    std::vector<double> amplitudes_;
    std::string input_,payload_;
    friend BoundedRestrictedLocalTriplesSolverResult bounded_restricted_local_triples_solve(
        const BoundedRestrictedLocalTriplesSolverInput&,const BoundedRestrictedLocalTriplesMomentProvider&,
        const BoundedRestrictedLocalTriplesSolverOptions&,const BoundedRestrictedLocalTriplesSolverInventory&,
        const BoundedRestrictedLocalTriplesSolverCaps&,BoundedRestrictedLocalTriplesSolverCallback,void*);
};

BoundedRestrictedLocalTriplesSolverMemoryPlan plan_bounded_restricted_local_triples_solver_upper(
    std::uint64_t occupied,std::uint64_t common_virtual_dimension,std::uint64_t maximum_rank,
    const BoundedRestrictedLocalTriplesMomentProvider&,const BoundedRestrictedLocalTriplesSolverOptions&,
    const BoundedRestrictedLocalTriplesSolverInventory&);
// Admit the count-only metadata pass before reading the space table; then
// exact ragged memory/work caps before floating scans or allocation.
BoundedRestrictedLocalTriplesSolverMemoryPlan plan_bounded_restricted_local_triples_solver(
    const BoundedRestrictedLocalTriplesSolverInput&,const BoundedRestrictedLocalTriplesMomentProvider&,
    const BoundedRestrictedLocalTriplesSolverOptions&,const BoundedRestrictedLocalTriplesSolverInventory&,
    const BoundedRestrictedLocalTriplesSolverCaps&);
// Zero start, two immutable snapshots; raw residual and energy-change tests.
// Guo Eq.1 unique-triple weight 2-delta_ij-delta_jk, NO per-cell divisor.
// iii amplitudes are retained and coupled even though their energy weight
// is zero. Empty spaces contribute zero without inventing a missing moment.
// Provider/progress errors, mutated input, or inconsistent CONSUMED moment
// snapshots publish no result. Opaque provider state is not introspected
// after its last visit; physical owners must supply immutable source receipts.
BoundedRestrictedLocalTriplesSolverResult bounded_restricted_local_triples_solve(
    const BoundedRestrictedLocalTriplesSolverInput&,const BoundedRestrictedLocalTriplesMomentProvider&,
    const BoundedRestrictedLocalTriplesSolverOptions&,const BoundedRestrictedLocalTriplesSolverInventory&,
    const BoundedRestrictedLocalTriplesSolverCaps&,BoundedRestrictedLocalTriplesSolverCallback = nullptr,void* = nullptr);

} // namespace vibeqc
