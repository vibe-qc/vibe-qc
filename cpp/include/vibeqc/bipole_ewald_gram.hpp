#pragma once

// Selected reciprocal-Ewald numerical Gram, not a BIPOLE SCF/correlation
// source certificate. Sun2017, doi:10.1063/1.4998644, Eqs.16/22; finite-k
// zero-mode distinction: Sundararaman/Arias2013, doi:10.1103/PhysRevB.87.165122,
// pp.2-4. With explicitly supplied product cells and the native AO convention:
//   B^-_{mu,nu}(p;k) = sum_R exp(-i k.R) FT[chi_mu chi_nu,R](p)
//   I[l,r] = sum_{p=G+q != 0} (4*pi/Omega)/p^2
//                       exp(-p^2/(4*omega^2)) conj(B^-_l) B^-_r.
// No 1/Nk, spin, Madelung/background, SR, density-adjoint, Hermiticity or
// permutation repair is applied. q is a centered TRANSFER of RegularKMesh;
// left/right k are GRID points (possibly shifted). Product-cell truncation
// and this enumerator's reciprocal boundary are explicit numerical policies,
// not proof of equality to an existing finite SCF operator.

#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/kmesh_address.hpp"
#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace vibeqc {

struct BipoleEwaldGramOptions {
    double omega = std::numeric_limits<double>::quiet_NaN();
    double reciprocal_energy_cutoff = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t reciprocal_block_size = 0;
    // No averaging. If requested, exact reciprocal conjugacy fails closed.
    // Cell inversion closure is measured independently even if not required.
    bool require_reciprocal_conjugacy = false;
    bool require_cell_inversion_closure = false;
};
struct BipoleEwaldGramSelection {
    std::uint64_t q_index = 0, left_k_index = 0, right_k_index = 0;
    std::uint64_t left_pair_begin = 0, left_pair_count = 0;
    std::uint64_t right_pair_begin = 0, right_pair_count = 0;
};
struct BipoleEwaldGramInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    std::uint64_t backend_margin_bytes_per_replica = 0;
};
struct BipoleEwaldGramCaps {
    AOPairFourierCellPanelCaps ao_panel;
    std::uint64_t maximum_kpoints = 0, maximum_reciprocal_candidates = 0;
    std::uint64_t maximum_accepted_vectors = 0, maximum_reciprocal_blocks = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0, maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0, maximum_work_units = 0;
};
struct BipoleEwaldGramPlan {
    // cell_count inventories borrowed labels: identical pointer/extents are
    // counted once; other views are conservatively counted separately.
    std::uint64_t n_basis = 0, n_kpoints = 0, cell_count = 0;
    std::uint64_t left_cell_count = 0, right_cell_count = 0;
    bool shared_cell_storage = false;
    BipoleEwaldGramSelection selection;
    std::uint64_t opposite_q_index = 0;
    std::array<int,3> centered_q_doubled = {0,0,0};
    std::array<int,3> centered_q_wrap = {0,0,0};
    std::uint64_t reciprocal_candidates = 0, opposite_reciprocal_candidates = 0;
    std::uint64_t accepted_vectors_upper_bound = 0, reciprocal_blocks_upper_bound = 0;
    std::uint64_t block_vectors = 0, retained_output_bytes = 0, compensation_bytes = 0;
    std::uint64_t reciprocal_buffer_bytes = 0, maximum_ao_panel_bytes = 0;
    std::uint64_t fixed_numerical_workspace_bytes = 0;
    std::uint64_t borrowed_basis_numeric_bytes = 0, borrowed_cell_bytes = 0;
    std::uint64_t borrowed_geometry_numeric_bytes = 0, borrowed_numerical_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, control_storage_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t reciprocal_traversal_work_units = 0, contraction_work_units = 0;
    std::uint64_t validation_work_units = 0, work_units_upper_bound = 0;
    AOPairFourierCellPanelPlan left_panel, right_panel;
};
struct BipoleEwaldGramDiagnostics {
    std::uint64_t accepted_vectors = 0, evaluated_blocks = 0, zero_weight_vectors = 0;
    bool cell_inversion_closed = false, reciprocal_conjugacy_audited = false;
    bool left_cell_inversion_closed = false, right_cell_inversion_closed = false;
    double maximum_weight = 0, maximum_integral_magnitude = 0;
};
class BipoleEwaldGramResult {
public:
    BipoleEwaldGramResult(const BipoleEwaldGramResult&) = delete;
    BipoleEwaldGramResult& operator=(const BipoleEwaldGramResult&) = delete;
    BipoleEwaldGramResult(BipoleEwaldGramResult&&) noexcept = default;
    BipoleEwaldGramResult& operator=(BipoleEwaldGramResult&&) noexcept = default;
    const BipoleEwaldGramPlan& memory() const noexcept { return plan_; }
    const BipoleEwaldGramDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::complex<double>* data() const;
    std::complex<double> element(std::uint64_t left, std::uint64_t right) const;
    const std::string& input_identity_sha256() const;
    const std::string& reciprocal_source_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
private:
    BipoleEwaldGramResult() = default;
    BipoleEwaldGramPlan plan_;
    BipoleEwaldGramDiagnostics diagnostics_;
    std::vector<std::complex<double>> values_;
    std::string input_identity_, reciprocal_identity_, payload_identity_;
    friend BipoleEwaldGramResult make_bipole_ewald_product_gram(
        const BasisSet&, const PeriodicSystem&, const RegularKMesh&,
        AOPairFourierCellView, AOPairFourierCellView, const BipoleEwaldGramSelection&, const BipoleEwaldGramOptions&,
        const BipoleEwaldGramInventory&, const BipoleEwaldGramCaps&);
};

// No numerical allocation or cell/primitive scans. O(1) lattice preparation
// follows shape/leaf metadata preflight; all candidate traversals, hashes and
// numerical allocations occur only after the returned complete admission.
// The accepted-count upper bound is the exact candidate box count; conservative
// work admission may therefore reject a sparse spherical source deliberately.
BipoleEwaldGramPlan plan_bipole_ewald_gram(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&, AOPairFourierCellView,
    const BipoleEwaldGramSelection&, const BipoleEwaldGramOptions&,
    const BipoleEwaldGramInventory&, const BipoleEwaldGramCaps&);
BipoleEwaldGramResult make_bipole_ewald_gram(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&, AOPairFourierCellView,
    const BipoleEwaldGramSelection&, const BipoleEwaldGramOptions&,
    const BipoleEwaldGramInventory&, const BipoleEwaldGramCaps&);

// Independent ordered supports for the left and right product panels. Each
// list applies to every selected pair on its own axis; for shell-pair-specific
// domains the caller must select only pairs belonging to that domain. This is
// not a ragged all-pair source. No domain is expanded to the other list's union.
// The existing shared-cell entry points delegate here with the same view twice.
// Both views are validated before reciprocal traversal, even for empty output.
// Per-list inversion is diagnostic, not the reversal relation C_ba = -C_ab
// needed by physical pair domains. require_cell_inversion_closure requires BOTH
// lists separately; off-diagonal physical domains generally disable this check.
// Identical ordered contents retain the legacy shared-cell scientific identity,
// independently of storage aliasing; otherwise both ordered lists are hashed in
// a distinct product-domain identity. No geometry-policy or HF provenance claim.
BipoleEwaldGramPlan plan_bipole_ewald_product_gram(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&,
    AOPairFourierCellView left_cells, AOPairFourierCellView right_cells,
    const BipoleEwaldGramSelection&, const BipoleEwaldGramOptions&,
    const BipoleEwaldGramInventory&, const BipoleEwaldGramCaps&);
BipoleEwaldGramResult make_bipole_ewald_product_gram(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&,
    AOPairFourierCellView left_cells, AOPairFourierCellView right_cells,
    const BipoleEwaldGramSelection&, const BipoleEwaldGramOptions&,
    const BipoleEwaldGramInventory&, const BipoleEwaldGramCaps&);

// Scalar arithmetic regression seam over the SAME private weighted-product
// helper used above. No Gaussian input, resource admission or provenance is
// supplied by this diagnostic. Requires finite factors and weight >= 0.
std::complex<double> bipole_ewald_weighted_product_diagnostic(
    std::complex<double> left, std::complex<double> right, double weight);

namespace detail {
// Shared scaled arithmetic for the reciprocal arm and its explicit zero-mode
// subtraction. Computes weight*conj(left)*right, without phase/Nk conventions.
std::complex<double> bipole_ewald_weighted_product(
    std::complex<double> left, std::complex<double> right, double weight);
}

} // namespace vibeqc
