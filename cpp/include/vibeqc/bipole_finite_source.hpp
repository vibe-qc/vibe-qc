#pragma once

// One explicit finite electron-electron source for the selected BIPOLE arms.
// Sun2023, doi:10.1063/5.0155815, Eqs.18-22/45-47; finite-mesh kernel
// distinction: Sundararaman/Arias2013, doi:10.1103/PhysRevB.87.165122, pp.2-4.
// In the inverse-Bloch convention of bipole_erfc_bloch / bipole_ewald_gram:
//   I = I_SR + I_LR - delta(q,0)*pi/(Omega*omega^2)*conj(Bminus_L(0))*Bminus_R(0).
// Only the explicitly chosen G0-omitted convention is implemented. No probe
// charge, Nk/spin factor, supplied overlap, averaging, HF or space-group
// certificate is added. Finite cutoffs need not yield omega-invariant values.

#include "vibeqc/bipole_erfc_bloch.hpp"
#include "vibeqc/periodic_ao_bloch_transport.hpp"
#include <array>
#include <memory>

namespace vibeqc {

enum class BipoleFiniteZeroMode : std::uint32_t { Unspecified = 0, G0Omitted = 1 };

struct BipoleFiniteSourceOptions {
    double omega = std::numeric_limits<double>::quiet_NaN();
    double reciprocal_energy_cutoff = std::numeric_limits<double>::quiet_NaN();
    BipoleFiniteZeroMode zero_mode = BipoleFiniteZeroMode::Unspecified;
    bool require_image_permutation_closure = false;
    bool require_cell_inversion_closure = false;
    bool require_reciprocal_conjugacy = false;
};
struct BipoleFiniteSourceCaps {
    std::uint64_t maximum_kpoints = 0, maximum_images = 0, maximum_cells = 0;
    std::uint64_t maximum_context_storage_bytes = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0, maximum_work_units = 0;
};
// Immutable ownership of AO-pair ranges for two declared product domains.
// Pair rows use mu*Nao+nu. This binds labels to rows, not a geometric cutoff
// certificate; a caller generating physical domains must use the right centers.
struct BipoleFiniteProductDomain {
    std::uint64_t left_pair_begin = 0, left_pair_count = 0;
    std::uint64_t right_pair_begin = 0, right_pair_count = 0;
};
struct BipoleFiniteSourcePlan {
    std::uint64_t n_basis = 0, n_shells = 0, n_kpoints = 0;
    std::uint64_t image_count = 0, cell_count = 0;
    std::uint64_t left_cell_count = 0, right_cell_count = 0;
    bool product_resolved = false;
    BipoleFiniteProductDomain product_domain;
    std::uint64_t context_storage_bytes = 0, borrowed_basis_numeric_bytes = 0;
    std::uint64_t borrowed_image_bytes = 0, borrowed_cell_bytes = 0;
    std::uint64_t borrowed_numerical_bytes = 0, work_units_upper_bound = 0;
    AOPairFourierCellPanelPlan basis_census;
};

class BipoleFiniteSource {
public:
    BipoleFiniteSource(const BipoleFiniteSource&) = delete;
    BipoleFiniteSource& operator=(const BipoleFiniteSource&) = delete;
    const Eigen::Matrix3d& direct_lattice() const noexcept { return direct_; }
    const Eigen::Matrix3d& reciprocal_lattice() const noexcept { return reciprocal_; }
    const RegularKMesh& mesh() const noexcept { return mesh_; }
    const BipoleFiniteSourceOptions& options() const noexcept { return options_; }
    const BipoleFiniteSourcePlan& memory() const noexcept { return plan_; }
    double cell_volume() const noexcept { return volume_; }
    double zero_mode_coefficient() const noexcept { return zero_coefficient_; }
    const std::string& source_identity_sha256() const noexcept { return source_identity_; }
    const std::string& basis_identity_sha256() const noexcept { return basis_identity_; }
    const std::string& images_identity_sha256() const noexcept { return images_identity_; }
    const std::string& cells_identity_sha256() const noexcept { return cells_identity_; }
    bool product_resolved() const noexcept { return plan_.product_resolved; }
    const BipoleFiniteProductDomain& product_domain() const noexcept { return plan_.product_domain; }
    const std::string& right_cells_identity_sha256() const noexcept { return right_cells_identity_; }
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
    // No borrowed pointers are retained. Every numerical call must match the
    // original basis content and exact ordered supports before allocation.
    void verify_inputs(const BasisSet&, BipoleErfcImageView, AOPairFourierCellView,
                       const BipoleFiniteSourceCaps&) const;
    void verify_product_inputs(const BasisSet&, BipoleErfcImageView,
                               AOPairFourierCellView, AOPairFourierCellView,
                               const BipoleFiniteSourceCaps&) const;
private:
    BipoleFiniteSource() = default;
    Eigen::Matrix3d direct_, reciprocal_;
    RegularKMesh mesh_{{1,1,1}};
    BipoleFiniteSourceOptions options_;
    BipoleFiniteSourcePlan plan_;
    double volume_ = 0, zero_coefficient_ = 0;
    std::string source_identity_, basis_identity_, images_identity_, cells_identity_, right_cells_identity_;
    friend std::shared_ptr<BipoleFiniteSource> make_bipole_finite_source(
        const BasisSet&, const PeriodicSystem&, const RegularKMesh&,
        BipoleErfcImageView, AOPairFourierCellView,
        const BipoleFiniteSourceOptions&, const BipoleFiniteSourceCaps&);
    friend std::shared_ptr<BipoleFiniteSource> make_bipole_finite_product_source(
        const BasisSet&, const PeriodicSystem&, const RegularKMesh&,
        BipoleErfcImageView, AOPairFourierCellView, AOPairFourierCellView,
        const BipoleFiniteProductDomain&, const BipoleFiniteSourceOptions&,
        const BipoleFiniteSourceCaps&);
};

// Context planning inspects shapes and incrementally capped basis metadata
// only. Geometry, primitive values and exact labels are checked after complete
// context admission. Atoms/charge/symmetry on System are not two-electron
// source inputs: only dim and the original lattice are read/retained.
BipoleFiniteSourcePlan plan_bipole_finite_source(
    const BasisSet&, const RegularKMesh&, BipoleErfcImageView, AOPairFourierCellView,
    const BipoleFiniteSourceOptions&, const BipoleFiniteSourceCaps&);
std::shared_ptr<BipoleFiniteSource> make_bipole_finite_source(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&,
    BipoleErfcImageView, AOPairFourierCellView,
    const BipoleFiniteSourceOptions&, const BipoleFiniteSourceCaps&);

// Source with independent left/right product cells and immutable AO ranges.
// maximum_cells applies to each list; borrowed storage counts an identical
// pointer/extent once, otherwise both views conservatively. No borrowed pointer
// is retained, and every evaluation revalidates both exact ordered contents.
// The common-source support/overlap/J/K consumers refuse this source; they
// cannot infer pair reversal or all-quartet support from one declared domain.
BipoleFiniteSourcePlan plan_bipole_finite_product_source(
    const BasisSet&, const RegularKMesh&, BipoleErfcImageView,
    AOPairFourierCellView, AOPairFourierCellView, const BipoleFiniteProductDomain&,
    const BipoleFiniteSourceOptions&, const BipoleFiniteSourceCaps&);
std::shared_ptr<BipoleFiniteSource> make_bipole_finite_product_source(
    const BasisSet&, const PeriodicSystem&, const RegularKMesh&, BipoleErfcImageView,
    AOPairFourierCellView, AOPairFourierCellView, const BipoleFiniteProductDomain&,
    const BipoleFiniteSourceOptions&, const BipoleFiniteSourceCaps&);

// Exact support audit for ONE DECLARED four-center Seitz mapping. In the
// AO transport convention f_destination = W f_source + tau + ell_source,
// a relative cell transforms as W R + ell_anchor - ell_center. The four
// ell rows refer to (mu,nu,lambda,sigma), in that order. A common shift
// cancels. No BvK modulo reduction is allowed for finite real-space supports.
//
// This accepts mapping metadata, not a validated geometry/AO action. It
// proves only multiset equality for the declared map, separately for the
// left/right Fourier-product cells and SR quartet images. It does NOT
// certify radial basis matching, mesh action, reciprocal support, numerical
// covariance, a full group, HF matching or a production Hamiltonian.
struct BipoleFiniteSupportAction {
    std::array<std::int32_t,9> rotation{}; // row-major integer W
    std::array<std::int64_t,12> atom_shifts{}; // ell, row-major [4,3]
};
struct BipoleFiniteSupportCaps {
    BipoleFiniteSourceCaps source;
    std::uint64_t maximum_label_comparisons = 0;
    std::uint64_t maximum_inventoried_bytes = 0, maximum_work_units = 0;
};
struct BipoleFiniteSupportPlan {
    BipoleFiniteSourcePlan source;
    std::uint64_t label_comparisons = 0, fixed_workspace_bytes = 0;
    std::uint64_t inventoried_bytes = 0, work_units_upper_bound = 0;
};
struct BipoleFiniteSupportDiagnostics {
    BipoleFiniteSupportPlan memory;
    bool left_cells_closed = false, right_cells_closed = false, images_closed = false;
    // First mismatch in source order; meaningful only if that audit is false.
    // Fixed-size labels plus closure flags avoid allocating a support table.
    std::array<std::int64_t,3> first_left_cell{}, first_right_cell{};
    std::array<std::int64_t,9> first_image{};
    std::uint64_t label_comparisons = 0;
    std::string source_identity_sha256, action_identity_sha256;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
};
// Planning reads shapes/metadata only. Auditing verifies the immutable source
// and complete action before inspecting support; malformed data throws, while
// a valid non-closed support returns the first missing/multiplicity witness.
BipoleFiniteSupportPlan plan_bipole_finite_support_action(
    const BipoleFiniteSource&, const BasisSet&, BipoleErfcImageView,
    AOPairFourierCellView, const BipoleFiniteSupportCaps&);
BipoleFiniteSupportDiagnostics audit_bipole_finite_support_action(
    const BipoleFiniteSource&, const BasisSet&, BipoleErfcImageView,
    AOPairFourierCellView, const BipoleFiniteSupportAction&, const BipoleFiniteSupportCaps&);

// Geometry-bound selected SHELL quartet audit. The existing AO transport
// validator derives atom shifts and a radial/angular shell bijection using
// zero coefficient columns, and validates the whole source k mesh. All AOs
// within each selected shell share its lattice shift. This does not assume
// that an angular shell rotation is an AO permutation. The source lattice
// must match exactly; no geometry or support is substituted or repaired.
struct BipoleFiniteMappedSupportSelection {
    std::array<std::uint64_t,4> source_shells{}; // mu,nu,lambda,sigma
    std::uint64_t source_k_index = 0;
    bool time_reversal = false; // scalar, no spin action
};
struct BipoleFiniteMappedSupportCaps {
    BipoleFiniteSupportCaps support;
    PeriodicAOBlochTransportCaps transport;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0, maximum_work_units = 0;
};
struct BipoleFiniteMappedSupportPlan {
    BipoleFiniteSupportPlan support;
    PeriodicAOBlochTransportPlan transport;
    std::uint64_t fixed_workspace_bytes = 0, per_replica_inventoried_bytes = 0;
    std::uint64_t required_node_inventoried_bytes = 0, work_units_upper_bound = 0;
};
struct BipoleFiniteMappedSupportDiagnostics {
    BipoleFiniteMappedSupportPlan memory;
    BipoleFiniteSupportDiagnostics support;
    PeriodicAOBlochTransportDiagnostics transport;
    std::array<std::uint64_t,4> source_shells{}, destination_shells{};
    std::array<std::uint64_t,4> source_atoms{}, destination_atoms{};
    std::array<std::int64_t,12> atom_shifts{};
    // Binds source, atom geometry/species, shell ownership, operation,
    // selection and numerical audit tolerances; execution caps are excluded.
    std::string mapping_identity_sha256;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
};
// Both complete child envelopes plus fixed output/hash workspace are charged
// before payload validation. Duplicate borrowed roles are conservative. No
// dense AO rotation, coefficient panel, support copy or all-k table is built.
BipoleFiniteMappedSupportPlan plan_bipole_finite_mapped_support(
    const BipoleFiniteSource&, const BasisSet&, const PeriodicSystem&, const SymmetryOp&,
    BipoleErfcImageView, AOPairFourierCellView, const BipoleFiniteMappedSupportSelection&,
    const PeriodicAOBlochTransportOptions&, const PeriodicAOBlochTransportInventory&,
    const BipoleFiniteMappedSupportCaps&);
BipoleFiniteMappedSupportDiagnostics audit_bipole_finite_mapped_support(
    const BipoleFiniteSource&, const BasisSet&, const PeriodicSystem&, const SymmetryOp&,
    BipoleErfcImageView, AOPairFourierCellView, const BipoleFiniteMappedSupportSelection&,
    const PeriodicAOBlochTransportOptions&, const PeriodicAOBlochTransportInventory&,
    const BipoleFiniteMappedSupportCaps&);

// Selected-k overlap gate for expressing this source's zero mode in the HF
// S D S form. B = Bminus(0;k), computed from the immutable source cells.
// Both S_ab = conj(B_ab) and S_ab = B_ba must hold within the caller's
// absolute element tolerance. No Hermitization or support repair is applied.
// A passing selected-k gate does not validate HF provenance, other k points,
// SR/LR support, probe-charge physics or a complete Hamiltonian/group.
struct BipoleFiniteOverlapCaps {
    BipoleFiniteSourceCaps source;
    AOPairFourierCellPanelCaps fourier;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0, maximum_work_units = 0;
};
struct BipoleFiniteOverlapPlan {
    BipoleFiniteSourcePlan source;
    AOPairFourierCellPanelPlan fourier;
    std::uint64_t k_index = 0, overlap_elements = 0, borrowed_overlap_bytes = 0;
    std::uint64_t fixed_workspace_bytes = 0, per_replica_inventoried_bytes = 0;
    std::uint64_t required_node_inventoried_bytes = 0, work_units_upper_bound = 0;
};
struct BipoleFiniteOverlapDiagnostics {
    BipoleFiniteOverlapPlan memory;
    double absolute_tolerance = 0, maximum_direct_residual = 0, maximum_dual_residual = 0;
    bool direct_matches = false, dual_matches = false;
    std::string source_identity_sha256, overlap_identity_sha256;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
};
// Borrowed S is an aligned row-major Nao^2 complex panel. Full admission
// precedes source/S payload validation and the one Nao^2 Fourier allocation.
BipoleFiniteOverlapPlan plan_bipole_finite_overlap(
    const BipoleFiniteSource&, const BasisSet&, BipoleErfcImageView, AOPairFourierCellView,
    const std::complex<double>*, std::uint64_t, std::uint64_t, double,
    const PeriodicAOBlochTransportInventory&, const BipoleFiniteOverlapCaps&);
BipoleFiniteOverlapDiagnostics audit_bipole_finite_overlap(
    const BipoleFiniteSource&, const BasisSet&, BipoleErfcImageView, AOPairFourierCellView,
    const std::complex<double>*, std::uint64_t, std::uint64_t, double,
    const PeriodicAOBlochTransportInventory&, const BipoleFiniteOverlapCaps&);

struct BipoleFinitePanelInventory : BipoleErfcBlochInventory {
    // Execution tiling is outside the immutable scientific source identity.
    std::uint64_t reciprocal_block_size = 0;
};
struct BipoleFinitePanelCaps {
    BipoleFiniteSourceCaps source;
    BipoleErfcBlochCaps short_range;
    BipoleEwaldGramCaps long_range;
    AOPairFourierCellPanelCaps zero_mode;
    std::uint64_t maximum_borrowed_numerical_bytes = 0, maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0, maximum_worker_bytes = 0;
    std::uint64_t maximum_node_bytes = 0, maximum_work_units = 0;
};
struct BipoleFinitePanelPlan {
    BipoleEwaldGramSelection selection;
    BipoleFiniteSourcePlan source;
    BipoleErfcBlochPlan short_range;
    BipoleEwaldGramPlan long_range;
    AOPairFourierCellPanelPlan zero_left, zero_right;
    bool subtract_zero_mode = false;
    std::uint64_t output_elements = 0, retained_output_bytes = 0, compensation_bytes = 0;
    std::uint64_t fixed_numeric_workspace_bytes = 0, wrapper_control_storage_bytes = 0;
    std::uint64_t zero_mode_workspace_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t borrowed_numerical_bytes = 0, control_storage_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t wrapper_work_units = 0, work_units_upper_bound = 0;
};
struct BipoleFinitePanelDiagnostics {
    BipoleErfcBlochDiagnostics short_range;
    BipoleEwaldGramDiagnostics long_range;
    std::uint64_t zero_mode_pair_values = 0;
    bool image_permutation_support_certified = false;
    double maximum_zero_mode_magnitude = 0, maximum_integral_magnitude = 0;
};
class BipoleFinitePanelResult {
public:
    BipoleFinitePanelResult(const BipoleFinitePanelResult&) = delete;
    BipoleFinitePanelResult& operator=(const BipoleFinitePanelResult&) = delete;
    BipoleFinitePanelResult(BipoleFinitePanelResult&&) noexcept = default;
    BipoleFinitePanelResult& operator=(BipoleFinitePanelResult&&) noexcept = default;
    const BipoleFinitePanelPlan& memory() const noexcept { return plan_; }
    const BipoleFinitePanelDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const BipoleFiniteSource& source() const;
    const std::complex<double>* data() const;
    std::complex<double> element(std::uint64_t left, std::uint64_t right) const;
    const std::string& input_identity_sha256() const;
    const std::string& short_range_payload_identity_sha256() const;
    const std::string& long_range_payload_identity_sha256() const;
    const std::string& zero_mode_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
private:
    BipoleFinitePanelResult() = default;
    void require_live() const;
    std::shared_ptr<const BipoleFiniteSource> source_;
    BipoleFinitePanelPlan plan_;
    BipoleFinitePanelDiagnostics diagnostics_;
    std::vector<std::complex<double>> values_;
    std::string input_identity_, sr_identity_, lr_identity_, zero_identity_, payload_identity_;
    friend BipoleFinitePanelResult make_bipole_finite_product_panel(
        std::shared_ptr<const BipoleFiniteSource>, const BasisSet&, BipoleErfcImageView,
        AOPairFourierCellView, AOPairFourierCellView, const BipoleEwaldGramSelection&,
        const BipoleFinitePanelInventory&, const BipoleFinitePanelCaps&);
};

// All child memory/work plus the retained outer output/compensation, context
// and union of borrowed owners are admitted before source payload validation
// or numerical allocation. SR and LR results never coexist. q=0 selected
// overlap vectors come from the same explicit-cell native Fourier primitive,
// with -k and G=0, not an external overlap source or an all-k AO matrix.
BipoleFinitePanelPlan plan_bipole_finite_panel(
    const BipoleFiniteSource&, const BasisSet&, BipoleErfcImageView, AOPairFourierCellView,
    const BipoleEwaldGramSelection&, const BipoleFinitePanelInventory&, const BipoleFinitePanelCaps&);
BipoleFinitePanelResult make_bipole_finite_panel(
    std::shared_ptr<const BipoleFiniteSource>, const BasisSet&, BipoleErfcImageView,
    AOPairFourierCellView, const BipoleEwaldGramSelection&,
    const BipoleFinitePanelInventory&, const BipoleFinitePanelCaps&);

// Same SR/LR/zero-mode composition with separate product supports. Every
// selected pair interval must lie inside its source-owned domain. At q=0 the
// left and right Fourier factors use their respective LR cell lists; q!=0 has
// no zero-mode factor allocation. No additional Nk/spin/probe-charge factor.
// The shared-source wrapper delegates with an identical cell view twice.
BipoleFinitePanelPlan plan_bipole_finite_product_panel(
    const BipoleFiniteSource&, const BasisSet&, BipoleErfcImageView,
    AOPairFourierCellView, AOPairFourierCellView, const BipoleEwaldGramSelection&,
    const BipoleFinitePanelInventory&, const BipoleFinitePanelCaps&);
BipoleFinitePanelResult make_bipole_finite_product_panel(
    std::shared_ptr<const BipoleFiniteSource>, const BasisSet&, BipoleErfcImageView,
    AOPairFourierCellView, AOPairFourierCellView, const BipoleEwaldGramSelection&,
    const BipoleFinitePanelInventory&, const BipoleFinitePanelCaps&);

// Bounded linear density action of THIS finite source, not a production HF
// builder or an admission certificate. D[k,lambda,sigma] is in C*C^dagger
// AO convention, in exact full-grid order. It may be non-Hermitian for linear
// operator witnesses; neither D nor the output is projected or repaired.
// Sun2023 Eqs.23/45-47 in the inverse-Bloch integral convention above:
//   J_ab(t) = sum_k,cd I(0,t,k)[ab,cd] D_k[cd] / Nk
//   K_ab(t) = sum_k,cd I(t-k,k,k)[ac,bd] D_k[cd] / Nk.
// Exactly one uniform full-mesh weight, no spin factor or Madelung term.
// A restricted Fock consumer would form H+J-K/2 from a spin-summed D;
// this routine deliberately does not choose a spin model or form a Fock.
struct BipoleFiniteDensityView {
    const std::complex<double>* values = nullptr;
    std::uint64_t n_kpoints = 0, n_basis = 0, value_count = 0;
};
struct BipoleFiniteJKSelection {
    std::uint64_t target_k_index = 0, pair_begin = 0, pair_count = 0;
    // Execution tiling, excluded from scientific input identity.
    std::uint64_t density_pair_block_size = 0;
};
struct BipoleFiniteJKCaps {
    // These memory ceilings cover the enclosing action too, not just its
    // child panel. Borrowed full-grid density and live output are charged.
    BipoleFinitePanelCaps panel;
    std::uint64_t maximum_panel_calls = 0, maximum_work_units = 0;
};
struct BipoleFiniteJKPlan {
    BipoleFiniteJKSelection selection;
    std::uint64_t n_basis = 0, n_kpoints = 0, density_bytes = 0;
    std::uint64_t panel_calls = 0, contracted_terms = 0, output_elements = 0;
    std::uint64_t retained_output_bytes = 0, compensation_bytes = 0;
    std::uint64_t fixed_numeric_workspace_bytes = 0, wrapper_control_storage_bytes = 0;
    std::uint64_t borrowed_numerical_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t control_storage_bytes = 0, per_replica_inventoried_bytes = 0;
    std::uint64_t required_node_inventoried_bytes = 0, wrapper_work_units = 0;
    std::uint64_t work_units_upper_bound = 0;
};
class BipoleFiniteJKResult {
public:
    BipoleFiniteJKResult(const BipoleFiniteJKResult&) = delete;
    BipoleFiniteJKResult& operator=(const BipoleFiniteJKResult&) = delete;
    BipoleFiniteJKResult(BipoleFiniteJKResult&&) noexcept = default;
    BipoleFiniteJKResult& operator=(BipoleFiniteJKResult&&) noexcept = default;
    const BipoleFiniteJKPlan& memory() const noexcept { return plan_; }
    const BipoleFiniteSource& source() const;
    // Row 0=J, row 1=K; columns follow the selected row-major AO pairs.
    const std::complex<double>* data() const;
    std::complex<double> element(std::uint64_t kind, std::uint64_t pair) const;
    const std::string& input_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
private:
    BipoleFiniteJKResult() = default;
    void require_live() const;
    std::shared_ptr<const BipoleFiniteSource> source_;
    BipoleFiniteJKPlan plan_;
    std::vector<std::complex<double>> values_;
    std::string input_identity_, payload_identity_;
    friend BipoleFiniteJKResult make_bipole_finite_jk(
        std::shared_ptr<const BipoleFiniteSource>, const BasisSet&, BipoleErfcImageView,
        AOPairFourierCellView, BipoleFiniteDensityView, const BipoleFiniteJKSelection&,
        const BipoleFinitePanelInventory&, const BipoleFiniteJKCaps&);
};
// Count-only preflight visits a capped descriptor-free schedule. A single
// selected ERI panel is live at once. No Nao^4 or all-q tensor is allocated.
// Complete memory/work admission precedes reading any density/image payload.
BipoleFiniteJKPlan plan_bipole_finite_jk(
    const BipoleFiniteSource&, const BasisSet&, BipoleErfcImageView,
    AOPairFourierCellView, BipoleFiniteDensityView, const BipoleFiniteJKSelection&,
    const BipoleFinitePanelInventory&, const BipoleFiniteJKCaps&);
BipoleFiniteJKResult make_bipole_finite_jk(
    std::shared_ptr<const BipoleFiniteSource>, const BasisSet&, BipoleErfcImageView,
    AOPairFourierCellView, BipoleFiniteDensityView, const BipoleFiniteJKSelection&,
    const BipoleFinitePanelInventory&, const BipoleFiniteJKCaps&);


// Complete native density reduction of a streamed sequence of immutable
// singleton product panels. Order: AO quartet (a,b,c,d), density k, J then K.
// This does not generate supports or run the panels. It admits their declared
// numerical envelope up front, checks every descriptor and common context,
// and refuses an incomplete result. The generating-policy digest is recorded
// as a caller declaration, not independently authenticated geometry evidence.
struct BipoleProductJKStreamCaps {
    std::uint64_t maximum_panel_calls = 0, maximum_density_bytes = 0;
    std::uint64_t maximum_state_bytes = 0, maximum_panel_inventoried_bytes = 0;
    std::uint64_t maximum_panel_work_units = 0;
    std::uint64_t maximum_node_bytes = 0, maximum_work_units = 0;
};
struct BipoleProductJKStreamPlan {
    std::uint64_t n_basis = 0, n_kpoints = 0, target_k_index = 0;
    std::uint64_t quartet_count = 0, panel_calls = 0, density_bytes = 0;
    std::uint64_t output_elements = 0, state_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t wrapper_work_units = 0, work_units_upper_bound = 0;
};
class BipoleProductJKStream {
public:
    BipoleProductJKStream(const BipoleProductJKStream&) = delete;
    BipoleProductJKStream& operator=(const BipoleProductJKStream&) = delete;
    const BipoleProductJKStreamPlan& memory() const noexcept { return plan_; }
    const BipoleFiniteSource& declaration() const noexcept { return *declaration_; }
    std::uint64_t accepted_panels() const noexcept { return accepted_; }
    bool complete() const noexcept { return accepted_ == plan_.panel_calls; }
    bool finalized() const noexcept { return finalized_; }
    BipoleEwaldGramSelection next_selection() const;
    // All rejection paths leave the stream unchanged and retryable.
    void consume(const BipoleFinitePanelResult&);
    void finalize();
    const std::complex<double>* data() const;
    std::complex<double> element(std::uint64_t kind, std::uint64_t pair) const;
    const std::string& declared_policy_identity_sha256() const noexcept { return policy_identity_; }
    const std::string& input_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    bool physical_hamiltonian_certified() const noexcept { return false; }
    bool symmetry_certified() const noexcept { return false; }
private:
    BipoleProductJKStream() = default;
    void require_finalized() const;
    std::shared_ptr<const BipoleFiniteSource> declaration_;
    BipoleProductJKStreamPlan plan_;
    BipoleProductJKStreamCaps caps_;
    std::vector<std::complex<double>> density_, values_, correction_;
    // O(Nao^2) consistency receipts, shared across both product roles.
    std::vector<std::array<char,65>> pair_identities_;
    std::string policy_identity_, chain_identity_, domain_identity_, payload_identity_;
    std::uint64_t accepted_ = 0;
    bool finalized_ = false;
    friend std::shared_ptr<BipoleProductJKStream> make_bipole_product_jk_stream(
        std::shared_ptr<const BipoleFiniteSource>, BipoleFiniteDensityView,
        std::uint64_t, const std::string&, const BipoleFinitePanelInventory&,
        const BipoleProductJKStreamCaps&);
};
BipoleProductJKStreamPlan plan_bipole_product_jk_stream(
    const BipoleFiniteSource&, BipoleFiniteDensityView, std::uint64_t target_k_index,
    const BipoleFinitePanelInventory&, const BipoleProductJKStreamCaps&);
std::shared_ptr<BipoleProductJKStream> make_bipole_product_jk_stream(
    std::shared_ptr<const BipoleFiniteSource>, BipoleFiniteDensityView,
    std::uint64_t target_k_index, const std::string& declared_policy_identity,
    const BipoleFinitePanelInventory&, const BipoleProductJKStreamCaps&);

} // namespace vibeqc
