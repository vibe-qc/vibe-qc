#pragma once

// Bounded closed-shell uniform-full-k RHF NUMERICAL leaf. Pulay (1982),
// doi:10.1002/jcc.540030413, Eq.4: evaluate the physical commutator using
// the SAME spin-summed D used to construct F[D]. No physical-source receipt,
// mean-field state factory, guessed basis/mesh, damping or level shifting.
// Version 1 requires maximum_diis_history=0 explicitly; nonzero fails before
// allocation. A later bounded adapter to existing DIIS is separate work.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vibeqc {
inline constexpr std::uint32_t kBoundedPeriodicRHFSolverVersion = 1U;

struct BoundedPeriodicRHFInput {
    std::uint64_t n_kpoints = 0, n_basis = 0, electrons_per_cell = 0;
    double nuclear_repulsion_energy_per_cell = 0.0;
    // Immutable aligned row-major complex128 [K,n,n], exact active extent;
    // accessible extents may be larger. No symmetry/mesh inference is made.
    const std::complex<double>* overlap = nullptr;
    std::size_t overlap_elements = 0;
    const std::complex<double>* hcore = nullptr;
    std::size_t hcore_elements = 0;
};

using BoundedPeriodicRHFFockCallback = void (*)(
    const std::complex<double>* density, std::size_t density_elements,
    std::complex<double>* two_electron_output, std::size_t output_elements, void*);
struct BoundedPeriodicRHFTwoElectronProvider {
    BoundedPeriodicRHFFockCallback evaluate = nullptr;
    void* context = nullptr;
    // G[D]=J[D]-K[D]/2 for spin-summed D. Every output element must be
    // written; unfilled/nonfinite lanes fail. D and output are disjoint.
    // These are callback resource contracts, NOT physical provenance claims.
    std::uint64_t retained_numerical_bytes = 0;
    // Excludes borrowed D/output; INCLUDES any native transient response
    // owner the adapter creates before copying into the output view.
    std::uint64_t peak_workspace_numerical_bytes = 0;
    std::uint64_t maximum_work_units_per_call = 0;
};

struct BoundedPeriodicRHFOptions {
    std::uint64_t maximum_iterations = 0, maximum_diis_history = 0;
    std::uint64_t jacobi_max_sweeps = 0;
    double jacobi_relative_tolerance = 0.0;
    // Strict keep lambda > max(abs,rel*lambda_max), same retained rank at
    // every k, at least nocc+1. Negative eigenvalues below -(abs+rel*scale)
    // fail; no discarded positive-overlap direction re-enters via identity.
    double overlap_rank_absolute_floor = 0.0, overlap_rank_relative_floor = 0.0;
    double overlap_negative_absolute_tolerance = 0.0, overlap_negative_relative_tolerance = 0.0;
    double hermitian_absolute_tolerance = 0.0, hermitian_relative_tolerance = 0.0;
    // Per-point Frobenius correction caps; finite nonnegative, zero is
    // exact-only. S/H are projected by an audited scalar accessor, not copied.
    double maximum_overlap_projection_error = 0.0;
    double maximum_hcore_projection_error = 0.0;
    double maximum_fock_projection_error = 0.0;
    double algebra_absolute_tolerance = 0.0, algebra_relative_tolerance = 0.0;
    double eigen_relative_tolerance = 0.0;
    double commutator_tolerance = 0.0;
    double density_closure_absolute_tolerance = 0.0, density_closure_relative_tolerance = 0.0;
    double energy_change_tolerance = 0.0;
    double minimum_band_gap_hartree = 0.0;
};

struct BoundedPeriodicRHFCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_work_units = 0, maximum_fock_calls = 0;
    // Other still-live producer/caller/callback arrays, counted ONCE.
    // Node replication, allocator/control/RSS margins remain outer duties.
    std::uint64_t other_live_numerical_bytes = 0;
};

struct BoundedPeriodicRHFMemoryPlan {
    std::uint64_t n_kpoints = 0, n_basis = 0, n_occupied = 0, retained_rank = 0;
    std::uint64_t overlap_discovery_owned_bytes = 0;
    std::uint64_t orthogonalizer_bytes = 0, density_bytes = 0, fock_bytes = 0;
    std::uint64_t coefficient_bytes = 0, orbital_energy_bytes = 0;
    std::uint64_t jacobi_workspace_bytes = 0, contraction_workspace_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, output_numerical_bytes = 0;
    std::uint64_t borrowed_input_bytes = 0, provider_retained_bytes = 0, provider_workspace_bytes = 0;
    std::uint64_t other_live_numerical_bytes = 0, total_numerical_bytes = 0;
    std::uint64_t input_validation_work_units = 0, overlap_discovery_work_units = 0;
    std::uint64_t orthogonalizer_work_units = 0, initial_density_work_units = 0;
    std::uint64_t work_units_per_evaluated_iteration = 0, maximum_work_units = 0;
};

enum class BoundedPeriodicRHFStatus : std::uint32_t {
    Converged = 0, IterationLimit = 1, WorkLimit = 2, FockCallLimit = 3,
    Cancelled = 4, NonInsulating = 5
};
struct BoundedPeriodicRHFSnapshot {
    BoundedPeriodicRHFStatus status = BoundedPeriodicRHFStatus::IterationLimit;
    std::uint64_t evaluated_iteration = 0, fock_calls = 0, charged_work_units = 0;
    bool converged = false, has_energy_change = false;
    double energy_per_cell = 0.0, electronic_energy_per_cell = 0.0, energy_change = 0.0;
    double raw_energy_per_cell = 0.0, energy_projection_change = 0.0;
    // RMS over k of retained-space e'=X^H(FDS-SDF)X, plus max element.
    double commutator_frobenius_rms = 0.0, maximum_commutator_element = 0.0;
    double maximum_density_closure_frobenius = 0.0, maximum_density_closure_relative = 0.0;
    double maximum_metric_idempotency_error = 0.0, maximum_electron_count_error = 0.0;
    double maximum_coefficient_metric_error = 0.0, maximum_projected_eigen_relative_residual = 0.0;
    // Diagnostic ONLY for truncated overlap space. Existing MF v1 capture
    // independently requires its full-AO FC=SCepsilon condition and S floor.
    double maximum_full_ao_eigen_residual = 0.0;
    double global_homo = 0.0, global_lumo = 0.0, global_band_gap = 0.0;
};
struct BoundedPeriodicRHFDiagnostics {
    std::uint64_t jacobi_sweeps = 0;
    double maximum_raw_overlap_hermitian_defect = 0.0, maximum_raw_hcore_hermitian_defect = 0.0;
    double maximum_raw_fock_hermitian_defect = 0.0;
    double maximum_overlap_projection_frobenius = 0.0, maximum_hcore_projection_frobenius = 0.0;
    double maximum_fock_projection_frobenius = 0.0;
    double maximum_reduced_operator_hermitian_defect = 0.0;
    double maximum_reduced_operator_projection_frobenius = 0.0;
    double minimum_overlap_eigenvalue = 0.0, minimum_retained_overlap_eigenvalue = 0.0;
    double maximum_overlap_eigen_relative_residual = 0.0, maximum_orthogonalizer_metric_error = 0.0;
};

// Return false to cancel AFTER this evaluated snapshot. Callback exceptions
// abort without a result; no progress callback receives matrices or owners.
using BoundedPeriodicRHFProgressCallback = bool (*)(const BoundedPeriodicRHFSnapshot&, void*);

class BoundedPeriodicRHFResult {
public:
    BoundedPeriodicRHFResult(const BoundedPeriodicRHFResult&) = delete;
    BoundedPeriodicRHFResult& operator=(const BoundedPeriodicRHFResult&) = delete;
    BoundedPeriodicRHFResult(BoundedPeriodicRHFResult&&) noexcept = default;
    BoundedPeriodicRHFResult& operator=(BoundedPeriodicRHFResult&&) noexcept = default;
    const BoundedPeriodicRHFMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedPeriodicRHFDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const BoundedPeriodicRHFSnapshot& final_snapshot() const noexcept { return snapshot_; }
    bool converged() const noexcept { return snapshot_.converged; }
    // Borrowed read-only row-major D/F[K,n,n], C[K,n,r], eps[K,r]. Last
    // E/F correspond to LAST evaluated D, NOT the freshly diagonalized C
    // density. The measured closure is retained, including at convergence.
    const std::complex<double>* density_data() const;
    const std::complex<double>* fock_data() const;
    const std::complex<double>* coefficients_data() const;
    const double* orbital_energies_data() const;
private:
    BoundedPeriodicRHFResult() = default;
    void require_live() const;
    BoundedPeriodicRHFMemoryPlan memory_;
    BoundedPeriodicRHFDiagnostics diagnostics_;
    BoundedPeriodicRHFSnapshot snapshot_;
    std::vector<std::complex<double>> density_, fock_, coefficients_;
    std::vector<double> energies_;
    friend BoundedPeriodicRHFResult solve_bounded_periodic_rhf(
        const BoundedPeriodicRHFInput&, const BoundedPeriodicRHFTwoElectronProvider&,
        const BoundedPeriodicRHFOptions&, const BoundedPeriodicRHFCaps&,
        BoundedPeriodicRHFProgressCallback, void*);
};

// Count-only. retained_rank=0 is the overlap-discovery phase, not an empty
// solution. r=n is a conservative full-rank reservation for an outer driver.
// For positive r the exact owned peak is
// 32Kn²+32Knr+8Kr+32n²+32nr+8n; output32Kn²+16Knr+8Kr.
// No Eigen/BLAS workspace, full-supercell matrices, or DIIS history exists.
BoundedPeriodicRHFMemoryPlan plan_bounded_periodic_rhf_solver(
    std::uint64_t n_kpoints, std::uint64_t n_basis, std::uint64_t electrons_per_cell,
    std::uint64_t retained_rank, const BoundedPeriodicRHFTwoElectronProvider&,
    const BoundedPeriodicRHFOptions&, std::uint64_t other_live_numerical_bytes);

// Hcore Aufbau start; one physical F[D] evaluation per iteration. A fresh
// physical canonicalization, SAME-D energy/commutator/idempotency/electron
// checks and density-vs-C closure precede convergence. At least two physical
// energy evaluations are required. Gap failures at a stationary solution
// return NonInsulating. Limits retain the last evaluated snapshot and its
// D/F/C/eps; no unevaluated density update is published. Initial caps must
// permit the overlap preparation and first complete F[D] evaluation.
BoundedPeriodicRHFResult solve_bounded_periodic_rhf(
    const BoundedPeriodicRHFInput&, const BoundedPeriodicRHFTwoElectronProvider&,
    const BoundedPeriodicRHFOptions&, const BoundedPeriodicRHFCaps&,
    BoundedPeriodicRHFProgressCallback progress = nullptr, void* progress_context = nullptr);
} // namespace vibeqc
