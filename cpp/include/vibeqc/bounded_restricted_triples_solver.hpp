#pragma once

/// Coupled common-space REFERENCE (T) iteration, Guo et al.,
/// doi:10.1063/1.5011798, Eqs.(1)-(2), using the native W/U and residual
/// leaves. Every occupied-Fock coupling is retained. This is NOT the T0
/// diagonal-occupied approximation or a pair-specific TNO/periodic driver.
/// The supplied real orthonormal virtual space must be quasi-canonical;
/// Fvv offdiagonals and Fov must be exactly zero. The caller establishes
/// physical integrals, converged CCSD amplitudes and the Brillouin reference.
///
/// Two explicitly capped o^3*v^3 snapshots are retained. W/U are rebuilt
/// one target at a time, not cached for all triples. This deliberately small
/// oracle is NOT a target-size storage strategy. All output is native data.

#include "vibeqc/bounded_restricted_triples_moments.hpp"
#include "vibeqc/bounded_restricted_triples_target.hpp"

namespace vibeqc {

struct BoundedRestrictedTriplesSolverInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    BoundedRestrictedCCSDRealView t1, t2; // [o,v], [o,o,v,v], immutable CCSD
    BoundedRestrictedCCSDRealView f_oo, f_vv, f_ov; // [o,o], [v,v], [o,v]
    std::uint64_t ccsd_snapshot_id = 0; // sequencing label, not certification
};
struct BoundedRestrictedTriplesSolverOptions {
    std::uint64_t maximum_iterations = 0; // evaluated snapshots, including zero
    double denominator_floor = 0.0;
    double residual_tolerance = 0.0, energy_tolerance = 0.0;
    double input_symmetry_tolerance = 0.0; // finite nonnegative; never repair
};
struct BoundedRestrictedTriplesSolverInventory {
    std::uint64_t external_node_numerical_bytes = 0;
    std::uint64_t numerical_replicas = 0;
};
struct BoundedRestrictedTriplesSolverCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_node_numerical_bytes = 0;
    std::uint64_t maximum_integral_calls = 0;
    std::uint64_t maximum_kernel_work_units = 0;
};
struct BoundedRestrictedTriplesSolverMemoryPlan {
    std::uint64_t n_occupied = 0, n_virtual = 0, maximum_iterations = 0;
    std::uint64_t amplitude_snapshot_bytes = 0, candidate_snapshot_bytes = 0;
    std::uint64_t borrowed_input_bytes = 0, orbital_workspace_bytes = 0;
    /// 16*o^3*v^3 + 32*v^3 + 24*v^2 + 8*v + 24*o.
    /// Fixed control/stack and allocator/runtime overhead excluded. No hidden
    /// BLAS workspace, file, thread, all-target moment or integral allocation.
    std::uint64_t peak_owned_numerical_bytes = 0, total_live_numerical_bytes = 0;
    std::uint64_t external_node_numerical_bytes = 0, numerical_replicas = 0;
    std::uint64_t required_node_numerical_bytes = 0;
    std::uint64_t target_evaluations_upper_bound = 0, neighbour_visits_upper_bound = 0;
    std::uint64_t integral_calls_upper_bound = 0, kernel_work_units_upper_bound = 0;
    BoundedRestrictedTriplesMomentsMemoryPlan moments;
    BoundedRestrictedTriplesTargetMemoryPlan target;
};
struct BoundedRestrictedTriplesSolverProgress {
    std::uint64_t iteration = 0, integral_calls = 0, target_evaluations = 0, neighbour_visits = 0;
    double triples_energy = 0.0, energy_change = 0.0;
    double maximum_absolute_residual = 0.0, residual_frobenius_norm = 0.0;
    bool has_previous_energy = false, converged = false;
};
struct BoundedRestrictedTriplesSolverResult {
    BoundedRestrictedTriplesSolverMemoryPlan memory;
    BoundedRestrictedTriplesSolverProgress final_snapshot;
    std::uint64_t ccsd_snapshot_id = 0;
    double minimum_denominator = 0.0, maximum_denominator = 0.0;
    std::vector<double> amplitudes; // [o,o,o,v,v,v], ordered axes
};
using BoundedRestrictedTriplesSolverProgressCallback =
    void (*)(const BoundedRestrictedTriplesSolverProgress&, void*);

BoundedRestrictedTriplesSolverMemoryPlan plan_bounded_restricted_triples_solver(
    std::uint64_t n_occupied, std::uint64_t n_virtual, std::uint64_t maximum_iterations,
    const BoundedRestrictedTriplesSolverInventory&,
    std::uint64_t provider_retained_numerical_bytes = 0,
    std::uint64_t provider_maximum_transient_numerical_bytes = 0);

/// Admit every cap, view, input lane, symmetry and positive gap before any
/// allocation or integral callback. Jacobi observes ONE immutable T3 across
/// all ordered targets; candidate first holds residuals and is converted to
/// T-R/gap only after the current global convergence verdict. Thus exhaustion
/// and success both return the last EVALUATED snapshot. Energy is Guo Eq.(1)
/// over i<=j<=k with weight 2-delta_ij-delta_jk, no translation/per-cell weight.
/// Both max residual and successive energy change must pass (>=2 snapshots).
/// Progress gets one scalar record per evaluated state; callback exceptions
/// abort without a partial result. Opaque provider work/live allocations and
/// caller callback storage require independent accounting in the inventory.
BoundedRestrictedTriplesSolverResult bounded_restricted_triples_solve(
    const BoundedRestrictedTriplesSolverInput&, const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedTriplesSolverOptions&, const BoundedRestrictedTriplesSolverInventory&,
    const BoundedRestrictedTriplesSolverCaps&,
    BoundedRestrictedTriplesSolverProgressCallback progress = nullptr,
    void* progress_context = nullptr);

} // namespace vibeqc
