#pragma once

/// Numerical certificate for an explicitly selected real local basis on the
/// finite BvK torus. Nejad doi:10.1063/5.0290816 Eqs.(2)-(10),(26)-(28).
/// Actual expanded C(-k)=conj(C(k)), including every TRIM, and
/// (1/Nk)sum C^H S C=I are inspected; real domain matrices are insufficient.
/// F is projected directly from the physical stored F(k), never substituted
/// by canonical energies. The final real symmetric F conversion is measured
/// against the raw complex projection and has an explicit error gate.
/// This does NOT certify matching HF/factor Hamiltonians or image convergence.

#include <array>
#include "vibeqc/periodic_correlation_local_orbital_factors.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationRealLocalBasisContractVersion = 1;

struct PeriodicCorrelationRealLocalBasisOptions {
    double coefficient_tr_absolute_tolerance = 0.0;
    double coefficient_tr_relative_tolerance = 0.0;
    double orthonormality_absolute_tolerance = 0.0;
    double orthonormality_relative_tolerance = 0.0;
    double fock_absolute_tolerance = 0.0;
    double fock_relative_tolerance = 0.0;
    double maximum_fock_projection_error = 0.0;
};

struct PeriodicCorrelationRealLocalBasisCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicCorrelationRealLocalBasisMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0, occupied_count = 0, virtual_count = 0, orbital_count = 0;
    std::uint64_t retained_index_bytes = 0, caller_index_bytes = 0, retained_fock_bytes = 0;
    std::uint64_t retained_output_bytes = 0, projection_matrix_bytes = 0;
    std::uint64_t coefficient_panel_bytes = 0, coefficient_scratch_bytes = 0;
    /// 16o+64m^2+32nao*m+32nao, m=o+v. Fixed metadata and allocator
    /// bookkeeping excluded, no hidden numerical backend workspace.
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t caller_gauge_bytes = 0, live_wannier_bytes = 0, live_domain_bytes = 0, live_space_bytes = 0;
    std::uint64_t work_units = 0, required_node_memory_bytes = 0;
};

struct PeriodicCorrelationRealLocalBasisDiagnostics {
    std::uint64_t inspected_kpoints = 0, inspected_trim_points = 0;
    double maximum_coefficient_tr_error = 0.0;
    double coefficient_tr_frobenius_upper_bound = 0.0;
    double maximum_orthonormality_error = 0.0;
    double orthonormality_frobenius_upper_bound = 0.0;
    double maximum_raw_fock_imaginary_magnitude = 0.0;
    double maximum_raw_fock_hermitian_error = 0.0;
    double maximum_fock_projection_error = 0.0;
    double fock_projection_frobenius_upper_bound = 0.0;
};

/// Move-only index/Fock owner. Fock is packed OO, VV, OV in that order;
/// transposed VO is the same explicitly gated real-symmetric projection.
/// Indices are physically stored as uint64 [o,2], not aliased POD structs.
/// All controls must be explicit: abs/relative pairs finite in [0,1), at
/// least one positive, and maximum_fock_projection_error finite positive.
/// Outward bounds assume correctly rounded binary64 basic arithmetic/sqrt,
/// round-to-nearest, gradual underflow and no fast-math reassociation.
class PeriodicCorrelationRealLocalBasis {
public:
    PeriodicCorrelationRealLocalBasis(const PeriodicCorrelationRealLocalBasis&) = delete;
    PeriodicCorrelationRealLocalBasis& operator=(const PeriodicCorrelationRealLocalBasis&) = delete;
    PeriodicCorrelationRealLocalBasis(PeriodicCorrelationRealLocalBasis&&) noexcept = default;
    PeriodicCorrelationRealLocalBasis& operator=(PeriodicCorrelationRealLocalBasis&&) noexcept = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationRealLocalBasisContractVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicCorrelationRealLocalBasisMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationRealLocalBasisDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const PeriodicCorrelationRealLocalBasisOptions& options() const noexcept { return options_; }
    PeriodicCorrelationVirtualBlockSelection virtual_selection() const noexcept { return virtual_; }
    PeriodicCorrelationPlacedOccupied occupied(std::size_t index) const;
    const std::uint64_t* occupied_indices_data() const;
    const double* f_oo_data() const;
    const double* f_vv_data() const;
    const double* f_ov_data() const;
    double fock(std::size_t left, std::size_t right) const;
    bool hf_hamiltonian_match_certified() const noexcept { return false; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& local_basis_identity_sha256() const noexcept { return basis_; }
    const std::string& allocation_identity() const noexcept { return allocation_; }
    // Native-only component receipts for independently matching the virtual
    // frame. The existing local_basis_identity already commits both hashes;
    // these fixed metadata copies add no coefficient owner or new policy.
    const std::array<char,64>& virtual_domain_identity_ascii() const noexcept { return virtual_domain_; }
    const std::array<char,64>& virtual_space_identity_ascii() const noexcept { return virtual_space_; }
private:
    PeriodicCorrelationRealLocalBasis() = default;
    void require_live() const;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationRealLocalBasisMemoryPlan memory_;
    PeriodicCorrelationRealLocalBasisDiagnostics diagnostics_;
    PeriodicCorrelationRealLocalBasisOptions options_;
    PeriodicCorrelationVirtualBlockSelection virtual_;
    std::vector<std::uint64_t> indices_;
    std::vector<double> fock_;
    std::string identity_, payload_, basis_, allocation_;
    std::array<char,64> virtual_domain_{}, virtual_space_{};
    friend PeriodicCorrelationRealLocalBasis make_periodic_correlation_real_local_basis(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const std::uint64_t*, std::size_t, std::size_t,
        const PeriodicCorrelationVirtualBlockSelection&, const PeriodicCorrelationRealLocalBasisOptions&,
        const PeriodicCorrelationRealLocalBasisCaps&);
};

PeriodicCorrelationRealLocalBasisMemoryPlan plan_periodic_correlation_real_local_basis(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationWannier&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&, std::uint64_t occupied_count,
    const PeriodicCorrelationVirtualBlockSelection&);
PeriodicCorrelationRealLocalBasis make_periodic_correlation_real_local_basis(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges, std::size_t gauge_count, const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&, const std::uint64_t* occupied_indices,
    std::size_t accessible_count, std::size_t occupied_count, const PeriodicCorrelationVirtualBlockSelection&,
    const PeriodicCorrelationRealLocalBasisOptions&, const PeriodicCorrelationRealLocalBasisCaps&);

}  // namespace vibeqc
