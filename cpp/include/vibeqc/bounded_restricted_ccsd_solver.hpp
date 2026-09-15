#pragma once

/// Bounded Jacobi REFERENCE CCSD iteration in one supplied real orthonormal
/// LOCAL extended domain. This is not the pair-specific DLPNO or periodic
/// driver, a Hamiltonian certificate, or a claim of scalable convergence.
/// All coupled occupied/virtual amplitudes inside that domain are retained.
/// Residuals use Stanton et al., doi:10.1063/1.460620, Eqs.(1)-(13), through
/// bounded_restricted_ccsd_target. The energy is the closed-shell spin sum of
/// Purvis and Bartlett, doi:10.1063/1.443164, Eq.(15):
/// 2 sum_ia f_ia t_ia + sum_ijab [2(ia|jb)-(ib|ja)](t_ijab+t_ia*t_jb).
/// It is an unweighted energy of the WHOLE SELECTED SYSTEM, not per cell.

#include "vibeqc/bounded_restricted_ccsd_target.hpp"

namespace vibeqc {

struct BoundedRestrictedCCSDSolverInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    BoundedRestrictedCCSDRealView f_oo, f_vv, f_ov;
    /// Both initial views must be absent (nullptr,0), or both supplied.
    /// Borrowed immutable until return; copied only after complete admission.
    BoundedRestrictedCCSDRealView initial_t1, initial_t2;
};

struct BoundedRestrictedCCSDSolverOptions {
    /// Number of EVALUATED amplitude snapshots, including the initial one.
    /// At least two snapshots are required to establish energy convergence.
    std::uint64_t maximum_iterations = 0;
    double denominator_floor = 0.0;
    double singles_residual_tolerance = 0.0;
    double doubles_residual_tolerance = 0.0;
    double energy_tolerance = 0.0;
    /// Finite nonnegative relative-to-max(1,|x|,|y|) audit tolerance.
    /// Checks F_oo/F_vv transpose symmetry and initial t_ijab=t_jiba.
    /// No averaging or repair of inputs is performed; zero requests exact.
    double input_symmetry_tolerance = 0.0;
};

/// Declared numerical inventory, not RSS or introspection of opaque owners.
/// Other caller/provider/progress allocations, allocator and runtime overhead
/// require their own admission. Progress callbacks must not grow unbudgeted
/// history; their retained/transient numerical bytes belong in this external
/// node declaration. Provider retained/transient storage is counted separately
/// per replica. No actual threads or ranks are launched by this serial leaf.
struct BoundedRestrictedCCSDSolverInventory {
    std::uint64_t external_node_numerical_bytes = 0;
    std::uint64_t numerical_replicas = 0;
};

struct BoundedRestrictedCCSDSolverCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_node_numerical_bytes = 0;
    std::uint64_t maximum_integral_calls = 0;
    std::uint64_t maximum_kernel_work_units = 0;
};

struct BoundedRestrictedCCSDSolverMemoryPlan {
    std::uint64_t n_occupied = 0, n_virtual = 0, maximum_iterations = 0;
    bool supplied_initial_amplitudes = false;
    std::uint64_t amplitude_snapshot_bytes = 0, candidate_snapshot_bytes = 0;
    std::uint64_t borrowed_fock_bytes = 0, borrowed_initial_amplitude_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, total_live_numerical_bytes = 0;
    std::uint64_t external_node_numerical_bytes = 0, numerical_replicas = 0;
    std::uint64_t required_node_numerical_bytes = 0;
    std::uint64_t target_evaluations_upper_bound = 0;
    std::uint64_t integral_calls_upper_bound = 0, kernel_work_units_upper_bound = 0;
    BoundedRestrictedCCSDTargetMemoryPlan target;
};

struct BoundedRestrictedCCSDSolverProgress {
    std::uint64_t iteration = 0, integral_calls = 0, target_evaluations = 0;
    double correlation_energy = 0.0, energy_change = 0.0;
    bool has_previous_energy = false, converged = false;
    double singles_max_residual = 0.0, doubles_max_residual = 0.0;
    double singles_residual_norm = 0.0, doubles_residual_norm = 0.0;
};

struct BoundedRestrictedCCSDSolverResult {
    BoundedRestrictedCCSDSolverMemoryPlan memory;
    BoundedRestrictedCCSDSolverProgress final_snapshot;
    double minimum_denominator = 0.0, maximum_denominator = 0.0;
    std::vector<double> t1;  // [o,v]
    std::vector<double> t2;  // [o,o,v,v], alpha-beta amplitudes
};

using BoundedRestrictedCCSDSolverProgressCallback =
    void (*)(const BoundedRestrictedCCSDSolverProgress&, void*);

/// Allocation-free checked plan. Counts current/candidate amplitudes once,
/// one entire target numerical peak, Fock/optional initial views and declared
/// provider storage. Includes the leaf's repeated all-amplitude scans.
BoundedRestrictedCCSDSolverMemoryPlan plan_bounded_restricted_ccsd_solver(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    std::uint64_t maximum_iterations, bool supplied_initial_amplitudes,
    const BoundedRestrictedCCSDSolverInventory&,
    std::uint64_t provider_retained_numerical_bytes = 0,
    std::uint64_t provider_maximum_transient_numerical_bytes = 0);

/// All positive caps, views, finite values, input symmetries and strictly
/// positive gaps greater than the floor pass before allocation/provider use.
/// Jacobi update Tnew=T-R/gap uses immutable T across all ordered targets.
/// No DIIS, damping, shifts, screening, global tensors or output. Callback
/// and arithmetic exceptions abort without publishing a partial result.
/// Convergence requires both max residuals AND successive energy change.
/// Exhaustion returns the last fully evaluated state with converged=false;
/// an unevaluated candidate is never returned. Progress is one scalar record
/// after each completed snapshot; its reference must not be retained.
BoundedRestrictedCCSDSolverResult bounded_restricted_ccsd_solve(
    const BoundedRestrictedCCSDSolverInput&, const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedCCSDSolverOptions&, const BoundedRestrictedCCSDSolverInventory&,
    const BoundedRestrictedCCSDSolverCaps&,
    BoundedRestrictedCCSDSolverProgressCallback progress = nullptr,
    void* progress_context = nullptr);

}  // namespace vibeqc
