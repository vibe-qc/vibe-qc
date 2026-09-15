#pragma once

// Numerical one-Seitz orbital-subspace sewing, NOT physical Hamiltonian
// symmetry authentication. No mean-field/correlation admission gate changes.
// Dovesi (1986), doi:10.1002/qua.560290608, Eqs.37-43; Casassa et al.
// (2006), doi:10.1007/s00214-006-0119-z, Sec.2 Eq.3: a symmetry operation
// acts through a full matrix within an invariant orbital subspace, not
// generally a signed permutation. No localization/IRREP construction here.
//
// Each X=frozen/active/virtual is processed separately:
//   T_X = Q_g conjugate_if_TR(C_source,X)
//   U_X = C_target,X^dagger S_target T_X.
// U_X is retained exactly as computed. Metric, reconstruction, cross-mask,
// target Roothaan and energy-intertwining audits never repair/project U.
// The three masks are exact supplied STATE masks, including within an
// exactly degenerate occupied manifold. Rank-reduced states give only a
// retained-space audit. Full-AO scope does not certify a finite Hamiltonian
// for other densities, an actual Gaussian/BIPOLE source, or an IBZ expansion.

#include "vibeqc/periodic_ao_bloch_transport.hpp"
#include "vibeqc/periodic_mean_field_state.hpp"
#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace vibeqc {

enum class PeriodicOrbitalSubspace : std::uint32_t {
    FrozenCore = 0, CorrelatedOccupied = 1, Virtual = 2
};

struct PeriodicOrbitalSewingOptions {
    PeriodicAOBlochTransportOptions ao_transport;
    bool request_full_ao_scope = false;
    // Every residual below is the maximum absolute complex element, without
    // a hidden dimension multiplier, relative tolerance or degeneracy cutoff.
    // The reciprocal audit is dimensionless max|A^T B/(2*pi)-I|.
    // Reciprocal, metric, unitarity and cross-subspace overlap budgets must
    // be finite, nonnegative and strictly less than one. Reconstruction,
    // Roothaan and energy-intertwining budgets must be finite and nonnegative.
    double maximum_reciprocal_lattice_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_source_metric_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_target_metric_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_transported_metric_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_unitarity_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_reconstruction_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_cross_subspace_overlap = std::numeric_limits<double>::quiet_NaN();
    double maximum_roothaan_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_energy_intertwining_residual = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicOrbitalSewingInventory {
    std::uint64_t numerical_replicas = 0;
    std::uint64_t external_node_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    std::uint64_t backend_margin_bytes_per_replica = 0;
};
struct PeriodicOrbitalSewingCaps {
    PeriodicAOBlochTransportCaps ao_transport;
    std::uint64_t maximum_kpoints = 0, maximum_basis_functions = 0;
    std::uint64_t maximum_effective_orbitals = 0, maximum_subspace_rank = 0;
    std::uint64_t maximum_transport_calls = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicOrbitalSewingPlan {
    std::uint64_t n_basis = 0, n_effective_orbitals = 0, n_kpoints = 0;
    std::uint64_t source_index = 0, target_index = 0, maximum_rank = 0;
    std::array<std::uint64_t,3> subspace_ranks = {0,0,0};
    std::array<int,3> target_doubled_address = {0,0,0};
    std::array<std::int64_t,3> reciprocal_wrap = {0,0,0};
    bool time_reversal = false, full_ao_scope = false;
    std::uint64_t state_resident_numerical_bytes = 0, state_control_storage_bytes = 0;
    std::uint64_t borrowed_basis_numeric_bytes = 0, borrowed_geometry_numeric_bytes = 0;
    std::uint64_t borrowed_numerical_bytes = 0, retained_sewing_bytes = 0;
    std::uint64_t index_workspace_bytes = 0, maximum_packed_source_bytes = 0;
    std::uint64_t column_workspace_bytes = 0, fixed_scalar_numerical_bytes = 0;
    std::uint64_t transport_phase_owned_upper_bound = 0, audit_phase_owned_upper_bound = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, control_storage_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t transport_calls = 0, transport_work_units_upper_bound = 0;
    std::uint64_t validation_work_units_upper_bound = 0, work_units_upper_bound = 0;
    // Conservative maximum-rank leaf plan, with the actual enclosing live
    // state/U/index/control owners already charged. Smaller calls re-admit.
    PeriodicAOBlochTransportPlan transport_upper;
};
struct PeriodicOrbitalSubspaceSewingDiagnostics {
    std::uint64_t rank = 0;
    double source_metric_residual = 0, target_metric_residual = 0;
    double transported_metric_residual = 0, unitarity_residual = 0;
    double reconstruction_residual = 0, cross_subspace_overlap = 0;
    double source_roothaan_residual = 0, target_roothaan_residual = 0;
    double roothaan_residual = 0, energy_intertwining_residual = 0;
    PeriodicAOBlochTransportDiagnostics transport;
};
struct PeriodicOrbitalSewingDiagnostics {
    double reciprocal_lattice_residual = 0;
    std::uint64_t completed_transport_calls = 0, audited_source_columns = 0;
    std::array<PeriodicOrbitalSubspaceSewingDiagnostics,3> subspaces;
};

class PeriodicOrbitalSewingResult {
public:
    PeriodicOrbitalSewingResult(const PeriodicOrbitalSewingResult&) = delete;
    PeriodicOrbitalSewingResult& operator=(const PeriodicOrbitalSewingResult&) = delete;
    PeriodicOrbitalSewingResult(PeriodicOrbitalSewingResult&&) noexcept = default;
    PeriodicOrbitalSewingResult& operator=(PeriodicOrbitalSewingResult&&) noexcept = default;
    const PeriodicOrbitalSewingPlan& memory() const noexcept { return plan_; }
    const PeriodicOrbitalSewingDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const;
    const std::string& state_identity_sha256() const;
    bool full_ao_scope_audited() const;
    bool physical_source_symmetry_certified() const noexcept { return false; }
    // Row-major U[target ordinal,source ordinal]. Ordinals follow increasing
    // native band index within each exact state mask, never energy regrouping.
    const std::complex<double>* sewing_data(PeriodicOrbitalSubspace) const;
    std::complex<double> element(PeriodicOrbitalSubspace, std::uint64_t target_ordinal,
                                 std::uint64_t source_ordinal) const;
    std::uint64_t source_band(PeriodicOrbitalSubspace, std::uint64_t ordinal) const;
    std::uint64_t target_band(PeriodicOrbitalSubspace, std::uint64_t ordinal) const;
private:
    PeriodicOrbitalSewingResult() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicOrbitalSewingPlan plan_;
    PeriodicOrbitalSewingDiagnostics diagnostics_;
    std::array<std::vector<std::complex<double>>,3> sewing_;
    friend PeriodicOrbitalSewingResult make_periodic_orbital_sewing(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState>, const BasisSet&,
        const PeriodicSystem&, const SymmetryOp&, std::uint64_t, bool,
        const PeriodicOrbitalSewingOptions&, const PeriodicOrbitalSewingInventory&,
        const PeriodicOrbitalSewingCaps&);
};

// Count/shape planning only. No masks, C, S/F, primitive values or reciprocal
// floating payloads are scanned until complete outer and nested admission.
// The whole immutable state's resident numerical bytes are charged ONCE;
// there is no all-k copy or dense AO rotation. Descriptor controls are a
// conservative role census plus explicit backend margin, not an RSS claim.
PeriodicOrbitalSewingPlan plan_periodic_orbital_sewing(
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&, const BasisSet&,
    const PeriodicSystem&, const SymmetryOp&, std::uint64_t source_index, bool time_reversal,
    const PeriodicOrbitalSewingOptions&, const PeriodicOrbitalSewingInventory&,
    const PeriodicOrbitalSewingCaps&);
PeriodicOrbitalSewingResult make_periodic_orbital_sewing(
    std::shared_ptr<const PeriodicRestrictedMeanFieldState>, const BasisSet&,
    const PeriodicSystem&, const SymmetryOp&, std::uint64_t source_index, bool time_reversal,
    const PeriodicOrbitalSewingOptions&, const PeriodicOrbitalSewingInventory&,
    const PeriodicOrbitalSewingCaps&);

} // namespace vibeqc
