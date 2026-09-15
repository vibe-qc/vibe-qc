#pragma once

// Bounded density-independent, all-electron nuclear operator. Ewald1921
// doi:10.1002/andp.19213690304 Eqs.38-40 (neutral lattice); the nonneutral
// background below is the explicit zero-average Green-function convention.
// Obara-Saika1986 doi:10.1063/1.450106 Appendix A14-A20 and Table II;
// Ahlrichs2006 doi:10.1039/b605188j Eqs.40/52. Finite nuclear cuts are
// independent approximation controls, NOT an RI identity or tail estimate.

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "vibeqc/periodic_gaussian_source_context.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicGaussianNuclearVersion = 1U;
inline constexpr char kPeriodicGaussianNuclearPolicy[] =
    "3d;all-electron-actual-Z;atomic-units;zero-average-tinfoil-G0-omit;"
    "Vshort=-sum-Z-erfc(alpha*r)/r;Vlong=negative-reciprocal-G-nonzero;"
    "Vbackground=positive-pi-Qn/(Omega*alpha^2)*S;"
    "positive-ket-phase;no-Nk-spin-factor;"
    "nuclei=native-Ewald-centered-representatives;"
    "fractional=inverse-A*raw-xyz;fractional-=nearbyint;xyz=A*fractional;"
    "AO-images=source-context-pair-separation-cut;"
    "short-nuclear-images=primitive-product-centered-closed-sphere;"
    "explicit-independent-nuclear-cuts;raw-complex;"
    "independent-reverse-pair-and-opposite-k-audits-v1";
inline constexpr char kPeriodicGaussianNuclearBoysPolicy[] =
    "Ahlrichs-positive-top-series;top=2*L+6<=18;iterations<=512;"
    "positive-downward-recurrence;nmax=2*L<=12;"
    "Tmax=100;step=0.01;grid=10001;Taylor-order=6;"
    "existing-native-OS-erfc-difference;no-propagated-error-certificate-v1";
inline constexpr char kPeriodicGaussianNuclearFloatingPointPolicy[] =
    "IEEE754-binary64;FLT_EVAL_METHOD=0;round-to-nearest;"
    "gradual-underflow;no-fast-math;native-transcendentals-v1";

struct PeriodicGaussianNuclearOptions {
    // Required finite positive controls, with no implicit automatic rule.
    double alpha = 0.0;
    double real_cutoff_bohr = 0.0;
    double reciprocal_cutoff_bohr_inverse = 0.0;
    // Finite in [0,1), not both zero. Raw components and total are audited;
    // none is averaged, conjugated into place, or silently made real.
    double structural_absolute_tolerance = 0.0;
    double structural_relative_tolerance = 0.0;
    // Used only for AO panels. Scalar Ewald does not read Gaussian bases.
    PeriodicGaussianSourceCaps basis_verification_caps;
};
struct PeriodicGaussianNuclearLiveInventory {
    std::uint64_t replicas_per_node = 0;
    std::uint64_t other_retained_bytes_per_replica = 0;
    std::uint64_t other_transient_bytes_per_replica = 0;
    // Positive allowance for allocator, scalar/control and native OpenMP
    // worker overhead. Inventoried payload is not a certified RSS bound.
    std::uint64_t fixed_backend_margin_bytes_per_replica = 0;
    std::uint64_t external_node_bytes = 0;
};
struct PeriodicGaussianNuclearCaps {
    std::uint64_t maximum_owned_numeric_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_atom_count = 0;
    std::uint64_t maximum_pair_count = 0;
    std::uint64_t maximum_ao_image_candidates = 0;
    std::uint64_t maximum_nuclear_image_candidates = 0;
    std::uint64_t maximum_reciprocal_candidates = 0;
    std::uint64_t maximum_ewald_pair_candidates = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianNuclearPlan {
    std::uint64_t n_basis = 0, atom_count = 0;
    std::uint64_t k_index = 0, opposite_k_index = 0;
    std::uint64_t pair_begin = 0, pair_count = 0;
    std::uint64_t output_numeric_bytes = 0; // exactly 64*pair_count
    std::uint64_t boys_table_numeric_bytes = 0;
    std::uint64_t os_workspace_numeric_bytes = 0;
    std::uint64_t selected_shell_numeric_bytes_upper_bound = 0;
    std::uint64_t primitive_and_shell_output_numeric_bytes = 0;
    std::uint64_t fourier_numeric_workspace_bytes = 0;
    std::uint64_t owned_numeric_peak_bytes = 0;
    std::uint64_t borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t borrowed_system_active_numeric_bytes = 0;
    std::uint64_t fixed_inventoried_object_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, node_inventoried_bytes = 0;
    // Includes original/reverse/opposite-k numerical constructions and any
    // admitted census passes. Rejected integer candidates are charged too.
    std::uint64_t ao_image_candidates_upper_bound = 0;
    std::uint64_t primitive_pair_evaluations_upper_bound = 0;
    std::uint64_t nuclear_image_candidates_upper_bound = 0;
    std::uint64_t reciprocal_box_candidates = 0;
    std::uint64_t reciprocal_candidate_evaluations_upper_bound = 0;
    std::uint64_t ao_fourier_calls_upper_bound = 0;
    std::uint64_t basis_scan_work_units = 0;
    std::uint64_t preflight_work_units = 0, work_units_upper_bound = 0;
    std::uint32_t maximum_selected_angular_momentum = 0;
    PeriodicGaussianNuclearOptions options;
    PeriodicGaussianNuclearLiveInventory live;
    PeriodicGaussianNuclearCaps caps;
};
struct PeriodicGaussianNuclearDiagnostics {
    std::uint64_t completed_ao_image_candidates = 0;
    std::uint64_t retained_ao_images = 0;
    std::uint64_t completed_nuclear_image_candidates = 0;
    std::uint64_t retained_nuclear_images = 0;
    std::uint64_t retained_reciprocal_vectors = 0;
    std::uint64_t completed_ao_fourier_calls = 0;
    std::uint32_t maximum_boys_series_iterations = 0;
    // Native erfc seeding subtracts positive Boys terms. These measured
    // magnitudes/ratios describe cancellation, NOT propagated error bounds.
    double maximum_erfc_seed_term_magnitude = 0.0;
    double minimum_nonzero_erfc_seed_difference_ratio = 1.0;
    std::uint64_t rounded_zero_erfc_seed_differences = 0;
    // Order: short, reciprocal-long, background, total.
    std::array<double, 4> maximum_hermitian_error{};
    std::array<double, 4> maximum_time_reversal_error{};
    double maximum_diagonal_imaginary_magnitude = 0.0;
    double maximum_trim_imaginary_magnitude = 0.0;
};
class PeriodicGaussianNuclearPanel {
public:
    PeriodicGaussianNuclearPanel(const PeriodicGaussianNuclearPanel&) = delete;
    PeriodicGaussianNuclearPanel& operator=(const PeriodicGaussianNuclearPanel&) = delete;
    PeriodicGaussianNuclearPanel(PeriodicGaussianNuclearPanel&&) noexcept = default;
    PeriodicGaussianNuclearPanel& operator=(PeriodicGaussianNuclearPanel&&) = delete;
    std::uint32_t contract_version() const noexcept { return kPeriodicGaussianNuclearVersion; }
    bool infinite_image_tail_certified() const noexcept { return false; }
    bool whole_hf_hamiltonian_certified() const noexcept { return false; }
    bool propagated_roundoff_error_certified() const noexcept { return false; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const PeriodicGaussianNuclearPlan& plan() const noexcept { return plan_; }
    const PeriodicGaussianNuclearDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    // [component,pair]: 0 short, 1 reciprocal-long, 2 background, 3 total.
    const std::vector<std::complex<double>>& values() const noexcept { return values_; }
    std::complex<double> element(std::uint32_t component, std::uint64_t pair) const;
    std::string nuclei_policy_identity_sha256() const;
    std::string panel_source_identity_sha256() const;
    std::string payload_identity_sha256() const;
private:
    PeriodicGaussianNuclearPanel() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianNuclearPlan plan_;
    PeriodicGaussianNuclearDiagnostics diagnostics_;
    std::array<char, 64> nuclei_{}, source_{}, payload_{};
    std::vector<std::complex<double>> values_;
    friend PeriodicGaussianNuclearPanel build_periodic_gaussian_nuclear_panel(
        std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet&, const BasisSet&,
        const PeriodicSystem&, std::uint64_t, std::uint64_t, std::uint64_t,
        const PeriodicGaussianNuclearOptions&, const PeriodicGaussianNuclearLiveInventory&,
        const PeriodicGaussianNuclearCaps&);
};

struct PeriodicGaussianNuclearEwaldPlan {
    std::uint64_t atom_count = 0;
    std::uint64_t packed_input_numeric_bytes = 0;
    std::uint64_t periodic_positions_numeric_bytes = 0;
    // Existing native Ewald reserves the FULL reciprocal integer box.
    std::uint64_t reciprocal_reserved_numeric_bytes = 0;
    std::uint64_t owned_numeric_peak_bytes = 0;
    std::uint64_t borrowed_system_active_numeric_bytes = 0;
    std::uint64_t fixed_inventoried_object_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, node_inventoried_bytes = 0;
    std::uint64_t atom_pair_count = 0;
    std::uint64_t real_pair_candidates_upper_bound = 0;
    std::uint64_t reciprocal_box_candidates = 0;
    std::uint64_t preflight_work_units = 0, work_units_upper_bound = 0;
    PeriodicGaussianNuclearOptions options;
    PeriodicGaussianNuclearLiveInventory live;
    PeriodicGaussianNuclearCaps caps;
};
class PeriodicGaussianNuclearEwald {
public:
    PeriodicGaussianNuclearEwald(const PeriodicGaussianNuclearEwald&) = delete;
    PeriodicGaussianNuclearEwald& operator=(const PeriodicGaussianNuclearEwald&) = delete;
    PeriodicGaussianNuclearEwald(PeriodicGaussianNuclearEwald&&) noexcept = default;
    PeriodicGaussianNuclearEwald& operator=(PeriodicGaussianNuclearEwald&&) = delete;
    std::uint32_t contract_version() const noexcept { return kPeriodicGaussianNuclearVersion; }
    bool infinite_image_tail_certified() const noexcept { return false; }
    bool whole_hf_hamiltonian_certified() const noexcept { return false; }
    bool cross_rank_bitwise_replay_certified() const noexcept { return false; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const PeriodicGaussianNuclearEwaldPlan& plan() const noexcept { return plan_; }
    double energy() const noexcept { return energy_; }
    std::string nuclei_policy_identity_sha256() const;
    std::string scalar_source_identity_sha256() const;
    std::string payload_identity_sha256() const;
private:
    PeriodicGaussianNuclearEwald() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianNuclearEwaldPlan plan_;
    double energy_ = 0.0;
    std::array<char, 64> nuclei_{}, source_{}, payload_{};
    friend PeriodicGaussianNuclearEwald build_periodic_gaussian_nuclear_ewald(
        std::shared_ptr<const PeriodicGaussianSourceContext>, const PeriodicSystem&,
        const PeriodicGaussianNuclearOptions&, const PeriodicGaussianNuclearLiveInventory&,
        const PeriodicGaussianNuclearCaps&);
};

// Both factories repeat planning and validate immutable-context consistency;
// plans do not authenticate mutable borrowed input. The common nuclei seal
// binds context, ordered actual Z/xyz, original cell, charge/multiplicity,
// alpha/cuts and compiled physical policies. It excludes k/pair selection,
// numerical audit tolerances, caps and live budgets. Optional space-group
// annotations do not define this unreduced operator. No owner pointer is
// hashed. Neither result retains borrowed system/basis pointers.
PeriodicGaussianNuclearPlan plan_periodic_gaussian_nuclear_panel(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet&, const BasisSet&,
    const PeriodicSystem&, std::uint64_t k_index, std::uint64_t pair_begin, std::uint64_t pair_count,
    const PeriodicGaussianNuclearOptions&, const PeriodicGaussianNuclearLiveInventory&,
    const PeriodicGaussianNuclearCaps&);
PeriodicGaussianNuclearPanel build_periodic_gaussian_nuclear_panel(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet&, const BasisSet&,
    const PeriodicSystem&, std::uint64_t k_index, std::uint64_t pair_begin, std::uint64_t pair_count,
    const PeriodicGaussianNuclearOptions&, const PeriodicGaussianNuclearLiveInventory&,
    const PeriodicGaussianNuclearCaps&);

// Ewald charges are actual Z, not inferred ECP/effective charges. Before the
// legacy numerical kernel is called, admit its complete reciprocal reserve,
// packed/periodic positions, all integer ranges and traversal/work products.
// Distinct periodically coincident nonzero nuclei reject instead of reaching
// the legacy near-zero self skip. No automatic alpha or cutoffs are used.
PeriodicGaussianNuclearEwaldPlan plan_periodic_gaussian_nuclear_ewald(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const PeriodicSystem&,
    const PeriodicGaussianNuclearOptions&, const PeriodicGaussianNuclearLiveInventory&,
    const PeriodicGaussianNuclearCaps&);
PeriodicGaussianNuclearEwald build_periodic_gaussian_nuclear_ewald(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const PeriodicSystem&,
    const PeriodicGaussianNuclearOptions&, const PeriodicGaussianNuclearLiveInventory&,
    const PeriodicGaussianNuclearCaps&);
} // namespace vibeqc
