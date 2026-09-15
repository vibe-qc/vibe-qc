#pragma once

/// Selected periodic PAO geometry, without a full finite-supercell matrix.
///
/// Nejad et al., J. Chem. Phys. 163, 214107 (2025),
/// doi:10.1063/5.0290816, Eqs. (2)-(3), (25)-(28), defines PAOs by the
/// retained virtual-space projector. With the state's unnormalized AO Bloch
/// matrices and uniform full-mesh weights, this module evaluates
///
/// Q(k) = C_virtual(k) C_virtual(k)^H S(k),
/// M_domain[a,b] = (1/Nk) sum_k exp(+ik.(R_a-R_b))
///                 [Q(k)^H M(k) Q(k)]_{mu_a,mu_b}, M = S or F.
///
/// Q = P_retained - P_all_occupied. It equals I-P_all_occupied only when
/// the retained SCF space is complete. Both frozen and correlated occupied
/// orbitals are excluded, with no spin-occupation factor of two. Discarded
/// positive-overlap directions must not re-enter through I-P_occupied.
/// Domain selection is an explicit caller decision; this module neither
/// selects atoms, removes PAO redundancy, nor constructs extended domains.
/// See Riplinger and Neese, doi:10.1063/1.4773581, Sec. II.B.1 and II.B.3.

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_admitted_reference.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationPAODomainContractVersion = 1;

/// A unique canonical modular cell (last-axis-fast) and home-cell AO index.
/// Ordering is retained verbatim; aliases and duplicates are rejected.
struct PeriodicPAODomainColumn {
    std::uint64_t cell = 0;
    std::uint64_t ao = 0;
};

/// All tolerances are explicit, finite and in [0,1). Hermitian controls must
/// not both be zero. Gates are elementwise abs + rel*max(|left|,|right|).
/// TR checks S/F on the full mesh and Q columns named by this domain; it
/// does not compare individual, gauge-dependent virtual MO columns.
/// Complex lanes are always retained, even when require_real_matrices passes.
struct PeriodicCorrelationPAODomainOptions {
    double hermitian_absolute_tolerance = 0.0;
    double hermitian_relative_tolerance = 0.0;
    double time_reversal_absolute_tolerance = 0.0;
    double time_reversal_relative_tolerance = 0.0;
    double real_absolute_tolerance = 0.0;
    double real_relative_tolerance = 0.0;
    bool require_time_reversal = false;
    bool require_real_matrices = false;
};

struct PeriodicCorrelationPAODomainMemoryPlan {
    std::uint64_t n_cells = 0;
    std::uint64_t n_basis = 0;
    std::uint64_t domain_dimension = 0;
    std::uint64_t maximum_unique_domain_columns = 0;
    std::uint64_t matrix_element_count = 0;
    std::uint64_t caller_domain_index_bytes = 0;
    std::uint64_t retained_domain_index_bytes = 0;
    std::uint64_t retained_matrix_bytes = 0;
    std::uint64_t temporary_column_bytes = 0;
    /// D>0: 16*D + 32*D*D + 48*nao. D=0: zero.
    /// Counts every owned numerical/index payload, excluding fixed objects,
    /// strings and allocator bookkeeping; no Eigen/BLAS or matrix copies.
    std::uint64_t peak_owned_numerical_bytes = 0;
};

struct PeriodicCorrelationPAODomainDiagnostics {
    double maximum_input_overlap_hermitian_defect = 0.0;
    double maximum_input_fock_hermitian_defect = 0.0;
    double maximum_overlap_time_reversal_residual = 0.0;
    double maximum_fock_time_reversal_residual = 0.0;
    double maximum_selected_projector_time_reversal_residual = 0.0;
    double maximum_raw_overlap_hermitian_defect = 0.0;
    double maximum_raw_fock_hermitian_defect = 0.0;
    double maximum_overlap_hermitization_correction = 0.0;
    double maximum_fock_hermitization_correction = 0.0;
    double maximum_overlap_imaginary_magnitude = 0.0;
    double maximum_fock_imaginary_magnitude = 0.0;
    bool time_reversal_compatible = true;
    bool real_matrices_compatible = true;
    std::uint64_t required_node_memory_bytes = 0;
};

/// Immutable, in-process geometry. Empty geometry is valid, not a claim of
/// usable correlation space. A nonempty geometry may also have PAO rank zero;
/// this stage intentionally does not diagonalize or guess its rank.
class PeriodicCorrelationPAODomain {
public:
    PeriodicCorrelationPAODomain(const PeriodicCorrelationPAODomain&) = delete;
    PeriodicCorrelationPAODomain& operator=(const PeriodicCorrelationPAODomain&) = delete;
    PeriodicCorrelationPAODomain(PeriodicCorrelationPAODomain&&) noexcept = default;
    PeriodicCorrelationPAODomain& operator=(PeriodicCorrelationPAODomain&&) noexcept = default;
    ~PeriodicCorrelationPAODomain() = default;

    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationPAODomainContractVersion; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicRestrictedMeanFieldState& state() const noexcept { return *state_; }
    std::uint64_t domain_dimension() const noexcept { return memory_.domain_dimension; }
    const PeriodicCorrelationPAODomainMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationPAODomainOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationPAODomainDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::string& domain_index_sha256() const noexcept { return domain_digest_; }
    const std::string& matrix_payload_sha256() const noexcept { return matrix_digest_; }
    const std::string& pao_domain_identity_sha256() const noexcept { return identity_digest_; }
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }
    PeriodicPAODomainColumn column(std::size_t index) const;
    std::complex<double> overlap(std::size_t row, std::size_t col) const;
    std::complex<double> fock(std::size_t row, std::size_t col) const;
    /// Const row-major D*D views, valid while this owner lives. Empty arrays
    /// may have null data; no full-supercell reconstruction API is provided.
    const std::complex<double>* overlap_data() const noexcept { return overlap_.data(); }
    const std::complex<double>* fock_data() const noexcept { return fock_.data(); }

private:
    PeriodicCorrelationPAODomain() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationPAODomainMemoryPlan memory_;
    PeriodicCorrelationPAODomainOptions options_;
    PeriodicCorrelationPAODomainDiagnostics diagnostics_;
    std::string domain_digest_, matrix_digest_, identity_digest_, allocation_identity_;
    std::vector<PeriodicPAODomainColumn> columns_;
    std::vector<std::complex<double>> overlap_, fock_;
    friend PeriodicCorrelationPAODomain make_periodic_correlation_pao_domain(
        const PeriodicCorrelationAdmittedReference&, const std::uint64_t*,
        std::size_t, std::size_t, std::uint64_t,
        const PeriodicCorrelationPAODomainOptions&);
};

/// Allocation-free checked counts, native extents and digest-length bounds.
PeriodicCorrelationPAODomainMemoryPlan plan_periodic_correlation_pao_domain(
    std::array<int, 3> mesh, std::uint64_t n_basis, std::uint64_t domain_dimension);

/// Scalar pair-major, full-k sum; only three nao-length complex columns are
/// live. The caller's interleaved uint64 [D,2] (cell,AO) view must stay
/// immutable during this call; its accessible count is in uint64 elements.
/// Caps and the admitted node budget precede domain reads or allocations.
/// Native numerical payload cap zero is valid only for D=0. Version 1 uses
/// the exact Gamma-centered full mesh, with no IBZ or shifted/twisted cells.
/// D=0 still audits the reference, masks, S/F Hermiticity and time reversal;
/// it allocates no size-dependent buffers, not a promise of zero arithmetic.
/// Raw AB and BA contractions are separately evaluated and audited before
/// explicit Hermitization. The correction is reported, not concealed.
PeriodicCorrelationPAODomain make_periodic_correlation_pao_domain(
    const PeriodicCorrelationAdmittedReference& reference,
    const std::uint64_t* cell_ao_indices, std::size_t accessible_index_count,
    std::size_t domain_dimension, std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationPAODomainOptions& options);

}  // namespace vibeqc
