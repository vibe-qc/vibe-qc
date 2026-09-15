#pragma once

// Bounded one-k overlap and kinetic panels from actual Gaussian bases.
// Sun2017 doi:10.1063/1.4998644 Eqs.10/16; McMurchie-Davidson1978
// doi:10.1016/0021-9991(78)90092-X Eqs.2.20-2.22,2.32,2.37,3.3-3.4.
// Positive ket Bloch phase, no Nk/volume/spin factor. Original-shell
// normalization multiplies shifted derivative POLYNOMIALS, never newly
// normalized raised shells. No nuclear operator, Hcore, SCF or tail claim.

#include <complex>
#include <memory>
#include <vector>
#include "vibeqc/periodic_gaussian_source_context.hpp"

namespace vibeqc {
inline constexpr std::uint32_t kPeriodicGaussianOneElectronVersion = 1U;
inline constexpr char kPeriodicGaussianOneElectronPolicy[] =
    "overlap=identity;kinetic=negative-half-laplacian;gradient-product-MD;"
    "atomic-units;positive-ket-phase;no-Nk-volume-spin-factor;"
    "original-libint-shell-normalization;pure-L<=6-cartesian-s;"
    "raw-complex;independent-reverse-pair-and-opposite-k-audits-v1";

struct PeriodicGaussianOneElectronOptions {
    PeriodicGaussianSourceCaps basis_verification_caps;
    // Both finite in [0,1), not both zero; no implicit scientific default.
    double structural_absolute_tolerance = 0.0;
    double structural_relative_tolerance = 0.0;
};
struct PeriodicGaussianOneElectronLiveInventory {
    std::uint64_t replicas_per_node = 0;
    std::uint64_t other_retained_bytes_per_replica = 0;
    std::uint64_t other_transient_bytes_per_replica = 0;
    // Positive caller allowance for remaining scalar stack/control/allocator
    // costs, not a certified RSS or operating-system memory bound.
    std::uint64_t fixed_backend_margin_bytes_per_replica = 0;
    std::uint64_t external_node_bytes = 0;
};
struct PeriodicGaussianOneElectronCaps {
    std::uint64_t maximum_owned_numeric_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_pair_count = 0;
    std::uint64_t maximum_candidate_evaluations = 0;
    std::uint64_t maximum_work_units = 0;
};
struct PeriodicGaussianOneElectronPlan {
    std::uint64_t n_basis = 0, k_index = 0, opposite_k_index = 0;
    std::uint64_t pair_begin = 0, pair_count = 0;
    std::uint64_t output_numeric_bytes = 0, fixed_numeric_workspace_bytes = 0;
    std::uint64_t owned_numeric_peak_bytes = 0;
    std::uint64_t borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t fixed_inventoried_object_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, node_inventoried_bytes = 0;
    // Includes all THREE sequential evaluations: original/reverse/opposite.
    std::uint64_t candidate_evaluations = 0;
    std::uint64_t primitive_pair_evaluations_upper_bound = 0;
    std::uint64_t md_table_cells_upper_bound = 0;
    std::uint64_t cartesian_pair_terms_upper_bound = 0;
    std::uint64_t image_identity_wire_bytes_upper_bound = 0;
    std::uint64_t basis_scan_work_units = 0, preflight_work_units = 0;
    std::uint64_t work_units_upper_bound = 0;
    PeriodicGaussianOneElectronOptions options;
    PeriodicGaussianOneElectronLiveInventory live;
    PeriodicGaussianOneElectronCaps caps;
};
struct PeriodicGaussianOneElectronDiagnostics {
    std::uint64_t retained_primary_pair_images = 0;
    std::uint64_t retained_audit_pair_images = 0;
    std::uint64_t completed_candidate_evaluations = 0;
    std::uint64_t completed_primitive_pair_evaluations = 0;
    std::uint64_t image_identity_wire_bytes = 0;
    double maximum_overlap_hermitian_error = 0.0;
    double maximum_kinetic_hermitian_error = 0.0;
    double maximum_overlap_time_reversal_error = 0.0;
    double maximum_kinetic_time_reversal_error = 0.0;
    double maximum_diagonal_imaginary_magnitude = 0.0;
    double maximum_trim_imaginary_magnitude = 0.0;
};
class PeriodicGaussianOneElectronPanel {
public:
    PeriodicGaussianOneElectronPanel(const PeriodicGaussianOneElectronPanel&) = delete;
    PeriodicGaussianOneElectronPanel& operator=(const PeriodicGaussianOneElectronPanel&) = delete;
    PeriodicGaussianOneElectronPanel(PeriodicGaussianOneElectronPanel&&) noexcept = default;
    PeriodicGaussianOneElectronPanel& operator=(PeriodicGaussianOneElectronPanel&&) = delete;
    std::uint32_t contract_version() const noexcept { return kPeriodicGaussianOneElectronVersion; }
    bool nuclear_hcore_certified() const noexcept { return false; }
    bool infinite_image_tail_certified() const noexcept { return false; }
    bool ao_image_source_certified() const noexcept { return false; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const PeriodicGaussianOneElectronPlan& plan() const noexcept { return plan_; }
    const PeriodicGaussianOneElectronDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    // [operator, pair], operator0=S, operator1=T. No projection or averaging.
    const std::vector<std::complex<double>>& values() const noexcept { return values_; }
    std::complex<double> element(std::uint32_t op, std::uint64_t pair) const;
    std::string image_source_identity_sha256() const;
    std::string operator_source_identity_sha256() const;
    std::string payload_identity_sha256() const;
private:
    PeriodicGaussianOneElectronPanel() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianOneElectronPlan plan_;
    PeriodicGaussianOneElectronDiagnostics diagnostics_;
    std::array<char, 64> images_{}, source_{}, payload_{};
    std::vector<std::complex<double>> values_;
    friend PeriodicGaussianOneElectronPanel build_periodic_gaussian_one_electron_panel(
        std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet&, const BasisSet&,
        const PeriodicSystem&, std::uint64_t, std::uint64_t, std::uint64_t,
        const PeriodicGaussianOneElectronOptions&, const PeriodicGaussianOneElectronLiveInventory&,
        const PeriodicGaussianOneElectronCaps&);
};

// No variable numerical allocation or image walk. Census/hash verification
// is bounded by the immutable context's EXACT counts after checking caller
// caps cover them; oversized substituted inputs cannot exploit generous caps.
// Pair boxes/work are counted before numerical evaluation. Original direct
// lattice must exactly equal the source context (signed zero canonicalized).
// Every build repeats planning; plans do not authenticate mutable inputs.
PeriodicGaussianOneElectronPlan plan_periodic_gaussian_one_electron_panel(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicSystem&, std::uint64_t k_index, std::uint64_t pair_begin, std::uint64_t pair_count,
    const PeriodicGaussianOneElectronOptions&, const PeriodicGaussianOneElectronLiveInventory&,
    const PeriodicGaussianOneElectronCaps&);

// Exact owned numerical peak =32*pair_count+23040 bytes. Original, reversed
// and opposite-k pair sums reuse that ONE fixed recurrence workspace. Basis
// active lanes are inventoried per role even if the two bases alias; caller
// declares excess capacities/control/other live owners. No input mutation or
// concurrent mutation is permitted during these synchronous calls.
// Abstract conservative work (not FLOP/time certification): context Wscan
// +256*P*(AO shells+contractions)+4096+128*(32P)
// +4096*C+512*Nprimitive+32*Nmdcells+512*Ncartesian, all3 traversals included.
// Metadata lookup during both planning/evaluation is included in the first
// term; all image SHA scans fit the candidate term. C includes rejected
// candidates, while primitive/MD/Cartesian terms use that conservative C.
PeriodicGaussianOneElectronPanel build_periodic_gaussian_one_electron_panel(
    std::shared_ptr<const PeriodicGaussianSourceContext>, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicSystem&, std::uint64_t k_index, std::uint64_t pair_begin, std::uint64_t pair_count,
    const PeriodicGaussianOneElectronOptions&, const PeriodicGaussianOneElectronLiveInventory&,
    const PeriodicGaussianOneElectronCaps&);
} // namespace vibeqc
