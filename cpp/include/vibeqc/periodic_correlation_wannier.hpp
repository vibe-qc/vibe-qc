#pragma once

/// \file periodic_correlation_wannier.hpp
/// \brief Bounded, exact finite-torus occupied Fourier/gauge transform.
///
/// Nejad et al., J. Chem. Phys. 163, 214107 (2025),
/// doi:10.1063/5.0290816, Eqs. (2)-(10): with unnormalized AO Bloch
/// sums, the home-cell orbital coefficients are
/// W0[R,mu,i] = (1/Nk) sum_k exp(+i k.R) [C_active(k) U(k)]_{mu,i}.
/// A translated orbital L uses W0[(R-L) mod mesh,mu,i]. This is a Fourier
/// primitive, NOT a gauge-localization optimizer or a locality guarantee.
/// No (Nk*nao) by (Nk*nocc) placed-orbital matrix is ever constructed.

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_admitted_reference.hpp"

namespace vibeqc {

class PeriodicCorrelationDiabaticSeed;
class PeriodicCorrelationIAOOptimizerResult;

inline constexpr std::uint32_t kPeriodicCorrelationWannierContractVersion = 1;

/// All controls are explicit: zero-initialized options are not admissible.
/// Unitarity is the largest element of both U^H U-I and U U^H-I.
/// TR and real gates use abs_tolerance + rel_tolerance * element magnitude;
/// no global scale hides an offending small element. Tolerances must lie in
/// [0,1), with a positive unitarity tolerance. Complex coefficients are kept
/// even when require_real_home_coefficients succeeds; no imaginary discard.
struct PeriodicCorrelationWannierOptions {
    double gauge_unitarity_tolerance = 0.0;
    double time_reversal_absolute_tolerance = 0.0;
    double time_reversal_relative_tolerance = 0.0;
    double real_absolute_tolerance = 0.0;
    double real_relative_tolerance = 0.0;
    bool require_time_reversal = false;
    bool require_real_home_coefficients = false;
};

struct PeriodicCorrelationWannierMemoryPlan {
    std::uint64_t n_cells = 0;
    std::uint64_t n_basis = 0;
    std::uint64_t n_home_occupied = 0;
    std::uint64_t coefficient_count = 0;
    std::uint64_t gauge_element_count = 0;
    std::uint64_t caller_gauge_bytes = 0;
    std::uint64_t retained_coefficient_bytes = 0;
    std::uint64_t temporary_gauged_coefficient_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
};

struct PeriodicCorrelationWannierDiagnostics {
    double maximum_left_unitarity_residual = 0.0;
    double maximum_right_unitarity_residual = 0.0;
    double maximum_overlap_time_reversal_residual = 0.0;
    double maximum_fock_time_reversal_residual = 0.0;
    double maximum_coefficient_time_reversal_residual = 0.0;
    double maximum_home_imaginary_magnitude = 0.0;
    double maximum_home_coefficient_magnitude = 0.0;
    bool time_reversal_compatible = true;
    bool real_home_coefficients_compatible = true;
    /// Includes admitted external/shared/per-rank inventory and worst-case
    /// MPI*worker replicas of this call's owned peak plus caller gauges.
    std::uint64_t required_node_memory_bytes = 0;
    /// Additional live seed indices/pivots charged by the owner-aware bridge.
    /// Zero for a raw gauge view; this is peak accounting, not retained output.
    std::uint64_t live_diabatic_seed_index_bytes = 0;
    std::uint64_t live_optimizer_index_bytes = 0;
};

/// Immutable result with exactly one numerical payload, in R,mu,i order.
/// Mask rows of U(k) name the ascending set bits of correlated_occupied_mask
/// at that k, not the first nocc columns. Frozen and virtual columns never
/// enter this transform. No eigenvalue-based partition is inferred here.
class PeriodicCorrelationWannier {
public:
    PeriodicCorrelationWannier(const PeriodicCorrelationWannier&) = delete;
    PeriodicCorrelationWannier& operator=(const PeriodicCorrelationWannier&) = delete;
    PeriodicCorrelationWannier(PeriodicCorrelationWannier&&) noexcept = default;
    PeriodicCorrelationWannier& operator=(PeriodicCorrelationWannier&&) noexcept = default;
    ~PeriodicCorrelationWannier() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationWannierContractVersion;
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle()
        const noexcept { return state_; }
    const PeriodicRestrictedMeanFieldState& state() const noexcept { return *state_; }
    const std::array<int, 3>& mesh() const noexcept { return state_->mesh(); }
    std::uint64_t n_cells() const noexcept { return memory_.n_cells; }
    std::uint64_t n_basis() const noexcept { return memory_.n_basis; }
    std::uint64_t n_home_occupied() const noexcept { return memory_.n_home_occupied; }
    const PeriodicCorrelationWannierMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationWannierOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationWannierDiagnostics& diagnostics() const noexcept {
        return diagnostics_;
    }
    const std::string& gauge_payload_sha256() const noexcept { return gauge_digest_; }
    const std::string& coefficient_payload_sha256() const noexcept { return coefficient_digest_; }
    const std::string& wannier_identity_sha256() const noexcept { return identity_digest_; }
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }
    /// Nonempty only for an actually converged native IAO optimizer owner.
    /// This receipt is separate from the numerical gauge/Fourier identity;
    /// it does not certify a global maximum or the HF/overlap source.
    const std::string& localization_identity_sha256() const noexcept { return localization_identity_; }

    std::complex<double> coefficient(std::size_t cell, std::size_t ao,
                                     std::size_t occupied) const;
    std::complex<double> translated_coefficient(
        std::size_t cell, std::size_t orbital_cell, std::size_t ao,
        std::size_t occupied) const;
    /// Read-only row-major nao*nactive view, valid while this owner lives.
    /// No all-placed-orbitals view or copy API is provided.
    const std::complex<double>* cell_coefficients(std::size_t cell) const;

private:
    PeriodicCorrelationWannier() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationWannierMemoryPlan memory_;
    PeriodicCorrelationWannierOptions options_;
    PeriodicCorrelationWannierDiagnostics diagnostics_;
    std::string gauge_digest_, coefficient_digest_, identity_digest_;
    std::string allocation_identity_;
    std::string localization_identity_;
    std::vector<std::complex<double>> coefficients_;

    friend PeriodicCorrelationWannier make_periodic_correlation_wannier(
        const PeriodicCorrelationAdmittedReference&, const std::complex<double>*,
        std::size_t, std::uint64_t, const PeriodicCorrelationWannierOptions&);
    friend PeriodicCorrelationWannier make_periodic_correlation_wannier_from_diabatic_seed(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationDiabaticSeed&,
        std::uint64_t, const PeriodicCorrelationWannierOptions&);
    friend PeriodicCorrelationWannier make_periodic_correlation_wannier_from_iao_optimizer(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationIAOOptimizerResult&,
        std::uint64_t, const PeriodicCorrelationWannierOptions&);
};

/// Count-only preflight: validates mesh/counts, candidate and byte overflow,
/// vector/address extents, and SHA-256's bit-length domain without allocating.
PeriodicCorrelationWannierMemoryPlan plan_periodic_correlation_wannier(
    std::array<int, 3> mesh, std::uint64_t n_basis,
    std::uint64_t n_home_occupied);

/// Gauges are a non-owning, aligned, row-major complex view [Nk,nactive,nactive]
/// that must remain immutable for this call. A cap of zero is unknown, not
/// unlimited. The cap covers exactly the two owned home-sized payloads at
/// peak; caller gauges and admitted reference inventory are counted separately
/// in the node budget. Finite/shape/unitarity checks precede native allocation.
/// Only the exact Gamma-centered full regular mesh is accepted in version 1.
PeriodicCorrelationWannier make_periodic_correlation_wannier(
    const PeriodicCorrelationAdmittedReference& reference,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationWannierOptions& options);

/// Borrow immutable native seed gauges without copying. Requires the exact
/// admitted state allocation, not merely equal content. The node preflight
/// includes the live seed's gauges AND indices/pivots for every worker replica;
/// only the Wannier transform's own buffers count against the owned byte cap.
/// The seed can be destroyed after return. This still is not a localizer.
PeriodicCorrelationWannier make_periodic_correlation_wannier_from_diabatic_seed(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationDiabaticSeed& seed,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationWannierOptions& options);

/// Consume the convergence receipt of a native optimizer without copying its
/// gauge payload or retaining its lifetime. Failed/limited optimizations are
/// rejected. Exact state/allocation and active-band order are mandatory;
/// require_time_reversal and require_real_home_coefficients must both be true.
/// Count retained optimizer gauges and active indices throughout construction.
/// Other live seed/IAO owners remain the caller's external-inventory duty.
PeriodicCorrelationWannier make_periodic_correlation_wannier_from_iao_optimizer(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationIAOOptimizerResult& optimizer,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationWannierOptions& options);

}  // namespace vibeqc
