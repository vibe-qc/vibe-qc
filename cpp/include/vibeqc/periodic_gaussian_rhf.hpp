#pragma once

// Actual finite-source Gaussian RHF producer. All S/T/V/G[D] are native
// consumers of the SAME original Gaussian source context. No supplied
// matrices, Fock callbacks, calculation labels or converged-state flags.
// A final F[2*Cocc*Cocc^H] reconstruction precedes native state capture.
// This is an experimental full-k reference, NOT production DLPNO or an
// infinite-source/auxiliary-quality/symmetry-reconstruction certificate.

#include <optional>
#include "vibeqc/bounded_periodic_rhf_solver.hpp"
#include "vibeqc/periodic_gaussian_fock.hpp"
#include "vibeqc/periodic_gaussian_nuclear.hpp"
#include "vibeqc/periodic_gaussian_one_electron.hpp"
#include "vibeqc/periodic_mean_field_state.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicGaussianRHFVersion = 1U;
struct PeriodicGaussianRHFOptions {
    std::uint64_t one_electron_pair_block = 0;
    PeriodicGaussianOneElectronOptions one_electron;
    PeriodicGaussianNuclearOptions nuclear;
    BoundedPeriodicRHFOptions scf;
};
struct PeriodicGaussianRHFCaps {
    PeriodicGaussianOneElectronCaps one_electron;
    PeriodicGaussianNuclearCaps nuclear;
    PeriodicGaussianFockCaps fock;
    BoundedPeriodicRHFCaps scf;
    std::uint64_t maximum_owned_numeric_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_state_numeric_bytes = 0;
    std::uint64_t maximum_one_electron_panel_calls = 0;
    std::uint64_t maximum_progress_callbacks = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianRHFFrozenSelection {
    // Binary [K,nocc], in increasing canonical occupied-band order at
    // every point. Explicit all-zero masks are valid. At least one active
    // occupied band, and the same frozen rank at each k, are required.
    const std::uint8_t* data = nullptr;
    std::size_t elements = 0;
};
struct PeriodicGaussianRHFPlan {
    std::uint64_t n_kpoints = 0, n_basis = 0, electrons_per_cell = 0;
    std::uint64_t atom_count = 0, frozen_selection_bytes = 0;
    std::uint64_t overlap_hcore_bytes = 0, state_bytes_upper_bound = 0;
    std::uint64_t state_validation_workspace_reservation_bytes = 0;
    std::uint64_t one_electron_panel_calls = 0;
    std::uint64_t one_electron_owned_upper_bound = 0;
    std::uint64_t scf_owned_upper_bound = 0, capture_owned_upper_bound = 0;
    std::uint64_t owned_numeric_upper_bound = 0;
    std::uint64_t borrowed_input_numeric_bytes = 0;
    std::uint64_t control_storage_reservation_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, node_inventoried_bytes = 0;
    std::uint64_t progress_callback_upper_bound = 0;
    std::uint64_t input_check_work_units = 0, work_units_upper_bound = 0;
    PeriodicGaussianFockPlan fock;
    BoundedPeriodicRHFMemoryPlan scf;
};
enum class PeriodicGaussianRHFStage : std::uint32_t {
    Begin = 0, NuclearEnergy = 1, OneElectron = 2, TwoElectron = 3,
    SCF = 4, CaptureRebuild = 5, Captured = 6, Finished = 7
};
struct PeriodicGaussianRHFProgress {
    PeriodicGaussianRHFStage stage = PeriodicGaussianRHFStage::Begin;
    std::uint64_t k_index = 0, pair_begin = 0, completed_panel_calls = 0;
    std::uint64_t fock_call = 0;
    PeriodicGaussianFockProgress fock;
    BoundedPeriodicRHFSnapshot scf;
};
using PeriodicGaussianRHFCallback = void (*)(const PeriodicGaussianRHFProgress&, void*);
struct PeriodicGaussianRHFDiagnostics {
    std::uint64_t completed_panel_calls = 0, completed_fock_calls = 0;
    std::uint64_t completed_progress_callbacks = 0;
    double nuclear_energy_per_cell = 0.0, captured_energy_per_cell = 0.0;
    double maximum_capture_fock_hermiticity_defect = 0.0;
    double maximum_capture_fock_change = 0.0;
    double capture_energy_change = 0.0;
    double capture_commutator_frobenius_rms = 0.0;
    double maximum_capture_projected_eigen_relative_residual = 0.0;
    double maximum_capture_coefficient_metric_error = 0.0;
    BoundedPeriodicRHFSnapshot last_scf_snapshot;
    BoundedPeriodicRHFDiagnostics scf;
};
class PeriodicGaussianRHFResult {
public:
    PeriodicGaussianRHFResult(const PeriodicGaussianRHFResult&) = delete;
    PeriodicGaussianRHFResult& operator=(const PeriodicGaussianRHFResult&) = delete;
    PeriodicGaussianRHFResult(PeriodicGaussianRHFResult&&) noexcept = default;
    PeriodicGaussianRHFResult& operator=(PeriodicGaussianRHFResult&&) = delete;
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const;
    const BoundedPeriodicRHFResult& unconverged_numerical_result() const;
    bool converged() const noexcept { return static_cast<bool>(state_); }
    bool matched_finite_gaussian_hf_source() const noexcept { return static_cast<bool>(state_); }
    bool infinite_source_accuracy_certified() const noexcept { return false; }
    const PeriodicGaussianRHFPlan& plan() const noexcept { return plan_; }
    const PeriodicGaussianRHFDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    std::string original_input_identity_sha256() const;
    std::string one_electron_source_identity_sha256() const;
    std::string final_fock_source_identity_sha256() const;
    std::string reference_source_identity_sha256() const;
    // Bounded original-input recheck for downstream native consumers. The
    // frozen mask is read from the actual captured state, not resupplied or
    // copied into an all-k buffer. Work is covered by input_check_work_units;
    // callers must reserve that scan before invoking it. No owner/array is
    // created, and equal dimensions or basis names are not authentication.
    void verify_physical_inputs(const BasisSet& ao, const BasisSet& auxiliary,
                                const PeriodicSystem& original_system) const;
private:
    PeriodicGaussianRHFResult() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::optional<BoundedPeriodicRHFResult> unfinished_;
    PeriodicGaussianRHFPlan plan_;
    PeriodicGaussianRHFDiagnostics diagnostics_;
    std::array<char,64> input_{}, one_electron_{}, final_fock_{}, reference_{};
    friend PeriodicGaussianRHFResult run_periodic_gaussian_rhf(
        std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet&, const BasisSet&,
        const PeriodicSystem&, PeriodicGaussianRHFFrozenSelection,
        const PeriodicGaussianRHFOptions&, const PeriodicGaussianFockConfig&,
        const PeriodicGaussianMetricLiveInventory&, const PeriodicGaussianRHFCaps&,
        PeriodicGaussianRHFCallback, void*);
};

// Conservative whole-driver admission using explicit leaf ceilings. Loose
// leaf caps may over-reserve; no all-k/all-q planning table is constructed.
// The original system/selection metadata are bounded before their scans.
// Numerical array bounds are separate from a 64 KiB base logical-control
// reservation, per-k owner controls, lower Fock fixed reservations, and an
// explicit positive caller backend allowance. This is
// not an allocator, OpenMP stack, process RSS or OS-page-cache certificate.
PeriodicGaussianRHFPlan plan_periodic_gaussian_rhf(
    const PeriodicGaussianSourceContext&, const PeriodicSystem&,
    PeriodicGaussianRHFFrozenSelection, const PeriodicGaussianRHFOptions&,
    const PeriodicGaussianFockConfig&, const PeriodicGaussianMetricLiveInventory&,
    const PeriodicGaussianRHFCaps&);

// Native all-electron, singlet, insulating, full-mesh path only. No symmetry
// reduction or scalar weights substitute for actual full-k AO matrices.
// Nonconvergence returns the last evaluated numerical result WITHOUT a
// mean-field owner. Callback exceptions abort without a partial receipt.
// On apparent convergence, reconstruct D from actual C, rebuild G[D] with
// the same producer, recompute the energy, gate raw Fock defects, and invoke
// the existing native state's independent full-AO validation. MF contract
// v1 still rejects general rank-deficient/full-AO-incompatible captures.
// All borrowed inputs must stay immutable, including across callbacks.
PeriodicGaussianRHFResult run_periodic_gaussian_rhf(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicSystem&, PeriodicGaussianRHFFrozenSelection,
    const PeriodicGaussianRHFOptions&, const PeriodicGaussianFockConfig&,
    const PeriodicGaussianMetricLiveInventory&, const PeriodicGaussianRHFCaps&,
    PeriodicGaussianRHFCallback progress = nullptr, void* progress_context = nullptr);
} // namespace vibeqc
