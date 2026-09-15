#pragma once

/// \file periodic_correlation_occupied_fock.hpp
/// \brief Physical occupied Fock blocks on an exact finite translation torus.
///
/// Nejad et al., J. Chem. Phys. 163, 214107 (2025),
/// doi:10.1063/5.0290816, Eqs. (15)-(16). With this implementation's
/// Wannier convention, f[R] = Nk^-1 sum_k exp(+ik.R) Focc(k), and
/// <i,Lbra|F|j,Lket> = f[(Lbra-Lket) mod mesh,i,j].
/// We project the actual admitted AO Fock, Focc=(C_active U)^H F(C_active U),
/// NOT a diagonal-eigenvalue replacement. The latter differs by the finite
/// Roothaan residual and is reported only as an independent diagnostic.

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_wannier.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationOccupiedFockContractVersion = 1;

/// Explicit controls; zero-initialized options are inadmissible. All scalar
/// tolerances lie in [0,1), and at least one Hermiticity tolerance is positive.
/// Absolute tolerances have Hartree units; relative tolerances are unitless.
/// Gates are per element: residual <= abs + rel*max(|left|,|right|).
/// Hermiticity is mandatory; TR/real gates are independently selectable.
/// No Hermitian averaging, real projection, or numerical clipping occurs.
struct PeriodicCorrelationOccupiedFockOptions {
    double hermiticity_absolute_tolerance = 0.0;
    double hermiticity_relative_tolerance = 0.0;
    double time_reversal_absolute_tolerance = 0.0;
    double time_reversal_relative_tolerance = 0.0;
    double real_absolute_tolerance = 0.0;
    double real_relative_tolerance = 0.0;
    bool require_time_reversal = false;
    bool require_real_blocks = false;
};

struct PeriodicCorrelationOccupiedFockMemoryPlan {
    std::uint64_t n_cells = 0;
    std::uint64_t n_basis = 0;
    std::uint64_t n_home_occupied = 0;
    std::uint64_t element_count = 0;
    std::uint64_t retained_block_bytes = 0;
    std::uint64_t projection_workspace_bytes = 0;
    std::uint64_t fourier_workspace_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0;
    std::uint64_t live_wannier_bytes = 0;
};

struct PeriodicCorrelationOccupiedFockDiagnostics {
    double maximum_projected_hermiticity_residual = 0.0;
    double maximum_translation_hermiticity_residual = 0.0;
    double maximum_projected_time_reversal_residual = 0.0;
    double maximum_block_imaginary_magnitude = 0.0;
    double maximum_canonical_projection_discrepancy = 0.0;
    double canonical_projection_discrepancy_frobenius = 0.0;
    bool time_reversal_compatible = true;
    bool real_blocks_compatible = true;
    std::uint64_t required_node_memory_bytes = 0;
};

/// Move-only owner of exactly Nk*nactive^2 complex translation entries.
/// State ownership is immutable; Wannier/gauge content identities are sealed
/// at construction. The source Wannier object need not outlive this result.
/// This is the occupied Hamiltonian primitive, not a local-CC solver.
class PeriodicCorrelationOccupiedFock {
public:
    PeriodicCorrelationOccupiedFock(const PeriodicCorrelationOccupiedFock&) = delete;
    PeriodicCorrelationOccupiedFock& operator=(const PeriodicCorrelationOccupiedFock&) = delete;
    PeriodicCorrelationOccupiedFock(PeriodicCorrelationOccupiedFock&&) noexcept = default;
    PeriodicCorrelationOccupiedFock& operator=(PeriodicCorrelationOccupiedFock&&) noexcept = default;
    ~PeriodicCorrelationOccupiedFock() = default;

    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationOccupiedFockContractVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicRestrictedMeanFieldState& state() const noexcept { return *state_; }
    const std::array<int, 3>& mesh() const noexcept { return state_->mesh(); }
    std::uint64_t n_cells() const noexcept { return memory_.n_cells; }
    std::uint64_t n_home_occupied() const noexcept { return memory_.n_home_occupied; }
    const PeriodicCorrelationOccupiedFockMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationOccupiedFockOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationOccupiedFockDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::string& wannier_identity_sha256() const noexcept { return wannier_identity_; }
    const std::string& gauge_payload_sha256() const noexcept { return gauge_identity_; }
    const std::string& block_payload_sha256() const noexcept { return block_identity_; }
    const std::string& occupied_fock_identity_sha256() const noexcept { return identity_; }
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }

    std::complex<double> element(std::size_t translation, std::size_t i, std::size_t j) const;
    std::complex<double> placed_element(std::size_t bra_cell, std::size_t ket_cell,
                                        std::size_t i, std::size_t j) const;
    /// Read-only nactive*nactive row-major block; no all-placed matrix API.
    const std::complex<double>* block(std::size_t translation) const;

private:
    PeriodicCorrelationOccupiedFock() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationOccupiedFockMemoryPlan memory_;
    PeriodicCorrelationOccupiedFockOptions options_;
    PeriodicCorrelationOccupiedFockDiagnostics diagnostics_;
    std::string wannier_identity_, gauge_identity_, block_identity_, identity_;
    std::string allocation_identity_;
    std::vector<std::complex<double>> blocks_;

    friend PeriodicCorrelationOccupiedFock make_periodic_correlation_occupied_fock(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t, std::uint64_t,
        const PeriodicCorrelationOccupiedFockOptions&);
};

/// Allocation-free shape, candidate-count, address, hash and byte preflight.
PeriodicCorrelationOccupiedFockMemoryPlan plan_periodic_correlation_occupied_fock(
    std::array<int, 3> mesh, std::uint64_t n_basis, std::uint64_t n_home_occupied);

/// Exact same immutable state and gauge content as the source Wannier are
/// required. Gauges are a non-owning aligned row-major [Nk,nactive,nactive]
/// view, immutable for this call. The positive cap covers native-owned peak:
/// output + 16*max(2*nao,max(mesh)); the projection and DFT scratch lifetimes
/// do not overlap. Node admission separately includes the live Wannier and
/// gauges for every worker plus the reference's other declared inventory.
PeriodicCorrelationOccupiedFock make_periodic_correlation_occupied_fock(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationWannier& wannier,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationOccupiedFockOptions& options);

}  // namespace vibeqc
