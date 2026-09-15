#pragma once

// Bounded numerical Seitz action on one AO coefficient panel. This is NOT
// a mean-field/orbital-sewing certificate and changes no correlation gate.
//
// Native AO Bloch sums use exp(+i k.T). With fractional atom coordinates
// f_b = W f_a + tau + ell_a, ell_a an INTEGER lattice vector, the action is
//   C'(b alpha,j) = exp(+2 pi i q'.ell_a) sum_beta D_l(R)[alpha,beta]
//                  conjugate_if_time_reversal(C(a beta,j)), R=A W A^-1.
// q' = (+/-) W^-T q = target_fractional + reciprocal_wrap exactly.
// ell includes the FULL fractional Seitz translation. A separate global
// translation phase must not be added in this AO-center Bloch convention.
// See Zicovich-Wilson/Dovesi (1998), construction Eqs.21-26, and Dovesi
// (1986), doi:10.1002/qua.560290608, with Bloch-sign conventions reconciled.
//
// Supports native Cartesian AND pure shells, L=0..6. Pure rows follow
// libint STANDARD m=-L..L; Cartesian rows follow native FOR_CART ordering
// and shell-wide axial normalization. Cartesian rotations are NOT assumed
// Euclidean-unitary. Cartesian blocks use direct monomial substitution;
// pure-shell blocks undergo polynomial reconstruction and Euclidean
// unitarity audits.

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "basis.hpp"
#include "kmesh_address.hpp"
#include "periodic.hpp"

namespace vibeqc {

inline constexpr int kPeriodicAOBlochTransportMaximumAngularMomentum = 6;

struct PeriodicAOBlochTransportOptions {
    // Position errors are Cartesian Euclidean norms. Matrix/polynomial
    // audits below are maximum absolute element residuals (not Frobenius).
    double maximum_atom_mapping_residual_bohr = std::numeric_limits<double>::quiet_NaN();
    double maximum_basis_origin_residual_bohr = std::numeric_limits<double>::quiet_NaN();
    double maximum_rotation_orthogonality_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_polynomial_reconstruction_residual = std::numeric_limits<double>::quiet_NaN();
    double maximum_pure_rotation_unitarity_residual = std::numeric_limits<double>::quiet_NaN();
    // |det(A / max_ij |A_ij|)|, strictly positive and less than one.
    double minimum_relative_lattice_volume = std::numeric_limits<double>::quiet_NaN();
};

struct PeriodicAOBlochTransportInventory {
    std::uint64_t numerical_replicas = 0;
    std::uint64_t external_node_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    std::uint64_t backend_margin_bytes_per_replica = 0;
};

struct PeriodicAOBlochTransportCaps {
    std::uint64_t maximum_atoms = 0, maximum_shells = 0, maximum_contractions = 0;
    std::uint64_t maximum_basis_functions = 0, maximum_columns = 0;
    std::uint64_t maximum_basis_numeric_lanes = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicAOBlochTransportPlan {
    std::uint64_t n_atoms = 0, n_shells = 0, n_contractions = 0, n_basis = 0, n_columns = 0;
    std::uint64_t basis_numeric_lanes = 0, n_kpoints = 0, source_index = 0, target_index = 0;
    std::array<int,3> mesh = {1,1,1}, is_shift = {0,0,0}, target_doubled_address = {0,0,0};
    std::array<std::int64_t,3> reciprocal_wrap = {0,0,0};
    bool time_reversal = false;
    std::uint64_t retained_output_bytes = 0, borrowed_coefficient_bytes = 0;
    std::uint64_t retained_mapping_bytes = 0;
    std::uint64_t borrowed_basis_numeric_bytes = 0, borrowed_geometry_numeric_bytes = 0;
    std::uint64_t borrowed_numerical_bytes = 0, mapping_workspace_bytes = 0;
    std::uint64_t fixed_numerical_workspace_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t control_storage_bytes = 0, per_replica_inventoried_bytes = 0;
    std::uint64_t required_node_inventoried_bytes = 0, work_units_upper_bound = 0;
};

struct PeriodicAOBlochTransportDiagnostics {
    std::uint64_t mapped_atoms = 0, mapped_shells = 0, rotated_contractions = 0;
    std::uint64_t pure_rotation_blocks = 0, cartesian_rotation_blocks = 0;
    double maximum_atom_mapping_residual_bohr = 0;
    double maximum_basis_origin_residual_bohr = 0;
    double rotation_orthogonality_residual = 0;
    double maximum_polynomial_reconstruction_residual = 0;
    double maximum_pure_rotation_unitarity_residual = 0;
    double maximum_output_magnitude = 0;
};

namespace detail {
struct PeriodicAOAtomMap {
    std::uint64_t destination = 0;
    std::array<std::int64_t,3> ell = {0,0,0};
};
struct PeriodicAOShellMap {
    std::uint64_t destination = 0, source_ao = 0, destination_ao = 0;
};
} // namespace detail

class PeriodicAOBlochTransportResult {
public:
    PeriodicAOBlochTransportResult(const PeriodicAOBlochTransportResult&) = delete;
    PeriodicAOBlochTransportResult& operator=(const PeriodicAOBlochTransportResult&) = delete;
    PeriodicAOBlochTransportResult(PeriodicAOBlochTransportResult&&) noexcept = default;
    PeriodicAOBlochTransportResult& operator=(PeriodicAOBlochTransportResult&&) noexcept = default;
    const PeriodicAOBlochTransportPlan& memory() const noexcept { return plan_; }
    const PeriodicAOBlochTransportDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::complex<double>* data() const;
    std::complex<double> coefficient(std::uint64_t ao, std::uint64_t column) const;
    // Validated single-operation maps, including for zero coefficient columns.
    // Scalar/fixed-size copies expose no mutable mapping storage. These do not
    // establish finite integral support closure or a group representation.
    std::uint64_t atom_destination(std::uint64_t atom) const;
    std::array<std::int64_t,3> atom_lattice_shift(std::uint64_t atom) const;
    std::uint64_t shell_destination(std::uint64_t shell) const;
    bool physical_orbital_sewing_certified() const noexcept { return false; }
private:
    PeriodicAOBlochTransportResult() = default;
    PeriodicAOBlochTransportPlan plan_;
    PeriodicAOBlochTransportDiagnostics diagnostics_;
    std::vector<std::complex<double>> coefficients_;
    std::vector<detail::PeriodicAOAtomMap> atoms_;
    std::vector<detail::PeriodicAOShellMap> shells_;
    friend PeriodicAOBlochTransportResult apply_periodic_ao_bloch_operation(
        const BasisSet&, const PeriodicSystem&, const SymmetryOp&, const RegularKMesh&,
        std::uint64_t, bool, const std::complex<double>*, std::uint64_t, std::uint64_t,
        const PeriodicAOBlochTransportOptions&, const PeriodicAOBlochTransportInventory&,
        const PeriodicAOBlochTransportCaps&);
};

// Count/metadata-only admission; never reads AO primitive values or C.
// Whole-mesh compatibility is exact, including shift parity, not merely a
// check that the requested source point happens to map onto the grid.
// No all-k table, dense all-AO rotation, shell copy or BasisSet clone exists.
// Numerical inventory includes fixed stack arrays; vector controls and all
// borrowed descriptor roles are separate. Zero columns are valid (empty C),
// but still require complete geometry, basis and operation validation.
// Integer atom shifts must be exactly representable in binary64 (|ell|<=2^52).
// Logical payloads are charged by role; full descriptor sizeofs conservatively
// duplicate inline numeric members. Backend margin is not allocator/RSS proof.
PeriodicAOBlochTransportPlan plan_periodic_ao_bloch_transport(
    const BasisSet&, const PeriodicSystem&, const SymmetryOp&, const RegularKMesh&,
    std::uint64_t source_index, bool time_reversal, std::uint64_t n_columns,
    const PeriodicAOBlochTransportOptions&, const PeriodicAOBlochTransportInventory&,
    const PeriodicAOBlochTransportCaps&);

// Exact aligned complex128 row-major [nao,n_columns], extent supplied in
// complex elements. All inputs are immutable borrowed views for the call.
// Both roles are charged even if caller storage aliases. Finite checks and
// strict exact radial matching precede coefficient output allocation.
// No spin rotation: time_reversal means scalar complex conjugation only.
PeriodicAOBlochTransportResult apply_periodic_ao_bloch_operation(
    const BasisSet&, const PeriodicSystem&, const SymmetryOp&, const RegularKMesh&,
    std::uint64_t source_index, bool time_reversal,
    const std::complex<double>* coefficients, std::uint64_t coefficient_count,
    std::uint64_t n_columns, const PeriodicAOBlochTransportOptions&,
    const PeriodicAOBlochTransportInventory&, const PeriodicAOBlochTransportCaps&);

} // namespace vibeqc
