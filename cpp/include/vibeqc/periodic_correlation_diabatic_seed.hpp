#pragma once

/// Bounded diabatic STARTING gauge, not a localization optimizer.
/// Nejad et al., JCP 163, 214107 (2025), doi:10.1063/5.0290816,
/// Eqs.17-23; Zhu and Tew, JPCA 128, 8570 (2024),
/// doi:10.1021/acs.jpca.4c04555, Eqs.22-25. A real rank-revealing
/// Cholesky factor L of the Gamma ACTIVE density C C^H supplies an anchor.
/// U(k)=polar(C_active(k)^H L) minimizes coefficient-space distance.
/// No overlap metric is inserted in that Procrustes similarity.
/// Every active band is retained; deficient similarities fail closed.
///
/// Full-rank polar equivariance fixes arbitrary incoming canonical gauges.
/// At every self-inverse k point, the physical C(k)U(k), NOT raw U(k),
/// must be real. Non-self partners are sewn by metric projection and audited.
/// Only exact Gamma-centered full meshes and restricted TR-compatible states
/// are accepted. Frozen-core and active projector TR are checked separately.

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_admitted_reference.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationDiabaticSeedContractVersion = 1;

struct PeriodicCorrelationDiabaticSeedOptions {
    /// Audit/relative controls lie in [0,1); unitarity and Jacobi tolerance
    /// are positive. Absolute rank floors are finite and nonnegative, with
    /// at least one positive absolute/relative floor for each rank decision.
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
    double unitarity_tolerance = 0.0;
    double cholesky_absolute_floor = 0.0;
    double cholesky_relative_floor = 0.0;
    double singular_absolute_floor = 0.0;
    double singular_relative_floor = 0.0;
    std::uint64_t jacobi_max_sweeps = 0;
    double jacobi_relative_tolerance = 0.0;
    /// Explicit conservative scalar-loop work budget, not a FLOP estimate.
    /// Charged before each stage. No hidden optimizer/iteration follows.
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicCorrelationDiabaticSeedMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0, n_active = 0;
    std::uint64_t n_effective_orbitals = 0, self_inverse_count = 0;
    std::uint64_t gauge_count = 0, active_index_count = 0;
    std::uint64_t retained_gauge_bytes = 0, retained_index_bytes = 0;
    std::uint64_t frame_workspace_bytes = 0, polar_workspace_bytes = 0;
    std::uint64_t scalar_workspace_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, output_numerical_bytes = 0;
    std::uint64_t reference_preflight_work_units = 0;
    std::uint64_t anchor_work_units = 0, polar_work_units_per_point = 0;
    std::uint64_t physical_work_units_per_point = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicCorrelationDiabaticSeedDiagnostics {
    std::uint64_t required_node_memory_bytes = 0, charged_work_units = 0;
    std::uint64_t polar_point_count = 0, sewn_pair_count = 0;
    std::uint64_t self_inverse_point_count = 0, jacobi_sweeps = 0;
    double maximum_overlap_time_reversal_residual = 0.0;
    double maximum_fock_time_reversal_residual = 0.0;
    double maximum_active_projector_time_reversal_residual = 0.0;
    double maximum_frozen_projector_time_reversal_residual = 0.0;
    double maximum_anchor_density_residual = 0.0;
    double maximum_anchor_imaginary_correction = 0.0;
    double minimum_cholesky_pivot = 0.0;
    double minimum_similarity_singular_value = 0.0;
    double maximum_similarity_singular_value = 0.0;
    double maximum_dilation_eigen_residual = 0.0;
    double maximum_procrustes_trace_residual = 0.0;
    double maximum_unitarity_residual = 0.0;
    double maximum_metric_orthonormality_residual = 0.0;
    double maximum_physical_reconstruction_residual = 0.0;
    double maximum_physical_time_reversal_residual = 0.0;
    double maximum_self_inverse_imaginary_magnitude = 0.0;
};

/// Sole numerical outputs: gauges [K,o,o], active band indices [K,o] and
/// Gamma Cholesky AO pivots [o]. The immutable reference allocation is shared,
/// not copied. Getters reject a consumed/moved-from owner.
class PeriodicCorrelationDiabaticSeed {
public:
    PeriodicCorrelationDiabaticSeed(const PeriodicCorrelationDiabaticSeed&) = delete;
    PeriodicCorrelationDiabaticSeed& operator=(const PeriodicCorrelationDiabaticSeed&) = delete;
    PeriodicCorrelationDiabaticSeed(PeriodicCorrelationDiabaticSeed&&) noexcept = default;
    PeriodicCorrelationDiabaticSeed& operator=(PeriodicCorrelationDiabaticSeed&&) noexcept = default;
    ~PeriodicCorrelationDiabaticSeed() = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationDiabaticSeedContractVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicRestrictedMeanFieldState& state() const;
    const PeriodicCorrelationDiabaticSeedMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationDiabaticSeedOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationDiabaticSeedDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }
    const std::string& gauge_payload_sha256() const noexcept { return gauge_digest_; }
    const std::string& seed_identity_sha256() const noexcept { return identity_digest_; }
    const std::complex<double>* gauges_data() const;
    const std::complex<double>* point_gauge(std::size_t point) const;
    std::complex<double> gauge(std::size_t point, std::size_t row, std::size_t column) const;
    std::uint64_t active_band(std::size_t point, std::size_t active) const;
    std::uint64_t gamma_pivot(std::size_t active) const;
private:
    PeriodicCorrelationDiabaticSeed() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationDiabaticSeedMemoryPlan memory_;
    PeriodicCorrelationDiabaticSeedOptions options_;
    PeriodicCorrelationDiabaticSeedDiagnostics diagnostics_;
    std::string allocation_identity_, gauge_digest_, identity_digest_;
    std::vector<std::complex<double>> gauges_;
    std::vector<std::uint64_t> active_indices_, pivots_;
    friend PeriodicCorrelationDiabaticSeed make_periodic_correlation_diabatic_seed(
        const PeriodicCorrelationAdmittedReference&, std::uint64_t,
        const PeriodicCorrelationDiabaticSeedOptions&);
};

/// Count-only plan. All owned arrays have fixed extents, no density B^2 or
/// all-placed (K*B)*(K*o) payload. Borrowed state/baseline are charged separately
/// by make(), including admitted MPI/worker replicas. The work plan is a
/// conservative upper bound in scalar-loop units, deliberately not FLOPs.
PeriodicCorrelationDiabaticSeedMemoryPlan plan_periodic_correlation_diabatic_seed(
    std::array<int, 3> mesh, std::uint64_t n_basis, std::uint64_t n_active,
    std::uint64_t n_effective_orbitals, std::uint64_t jacobi_max_sweeps);

PeriodicCorrelationDiabaticSeed make_periodic_correlation_diabatic_seed(
    const PeriodicCorrelationAdmittedReference& reference,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationDiabaticSeedOptions& options);

}  // namespace vibeqc
