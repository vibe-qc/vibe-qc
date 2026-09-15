#pragma once

/// Compact complex canonical PAO orthogonalization and semicanonicalization.
/// Löwdin, Advances in Quantum Chemistry 5, 185 (1970), Sec. II.C,
/// doi:10.1016/S0065-3276(08)60339-1, Eqs. (35)-(37): X=U_r lambda_r^-1/2.
/// Riplinger and Neese, doi:10.1063/1.4773581, Sec. II.B.3, p. 034106-7:
/// diagonalize the Fock operator within the nonredundant local PAO domain.
/// This is canonical orthogonalization, not a square symmetric inverse root.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/hermitian_jacobi.hpp"
#include "vibeqc/periodic_correlation_pao_domain.hpp"

namespace vibeqc {

class PeriodicCorrelationRealPAOSpace;
struct PeriodicCorrelationRealPAOSpaceOptions;
struct PeriodicCorrelationRealPAOSpaceCaps;

inline constexpr std::uint32_t kPeriodicCorrelationPAOSpaceContractVersion = 1;

/// Scientific rank controls are separate from numerical solver tolerances.
/// Retain strictly lambda > max(rank_abs, rank_rel*max(lambda_max,0)).
/// Reject lambda_min < -(negative_abs + negative_rel*max_abs_lambda).
/// Absolute rank/negative controls are finite nonnegative; relative controls
/// lie in [0,1). Each pair must contain a positive value. There is no inferred
/// chemical/default threshold and no forced orbital. Validation controls are
/// finite in [0,1), not both zero; scalar matrix gates are elementwise
/// abs + rel*max(|left|,|right|). Fock gates operate on F/2^fock_scale_exponent.
/// Independent eigensystem relative residuals and unitary errors must be
/// <= validation_abs+validation_rel. Complex lanes are never discarded.
struct PeriodicCorrelationPAOSpaceOptions {
    double rank_absolute_cutoff = 0.0;
    double rank_relative_cutoff = 0.0;
    double negative_absolute_tolerance = 0.0;
    double negative_relative_tolerance = 0.0;
    double validation_absolute_tolerance = 0.0;
    double validation_relative_tolerance = 0.0;
    HermitianJacobiOptions eigensolver;
};

struct PeriodicCorrelationPAOSpaceMemoryPlan {
    std::uint64_t domain_dimension = 0;
    std::uint64_t retained_dimension = 0;
    std::uint64_t borrowed_domain_index_bytes = 0;
    std::uint64_t borrowed_domain_matrix_bytes = 0;
    std::uint64_t borrowed_domain_bytes = 0;
    std::uint64_t overlap_factorization_phase_bytes = 0;
    std::uint64_t compact_orthogonalizer_phase_bytes = 0;
    std::uint64_t projected_fock_phase_bytes = 0;
    std::uint64_t fock_factorization_phase_bytes = 0;
    std::uint64_t rotation_phase_bytes = 0;
    std::uint64_t validation_phase_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t output_numerical_bytes = 0;
};

struct PeriodicCorrelationPAOSpaceDiagnostics {
    double effective_rank_cutoff = 0.0;
    double effective_negative_tolerance = 0.0;
    double minimum_overlap_eigenvalue = 0.0;
    double maximum_overlap_eigenvalue = 0.0;
    std::uint64_t negative_overlap_eigenvalue_count = 0;
    double overlap_eigensystem_relative_residual = 0.0;
    double overlap_eigenvector_orthogonality_error = 0.0;
    double canonical_metric_frobenius_residual = 0.0;
    double maximum_projected_fock_hermitian_defect = 0.0;
    double maximum_projected_fock_hermitization_correction = 0.0;
    double fock_eigensystem_relative_residual = 0.0;
    double fock_eigenvector_orthogonality_error = 0.0;
    double final_metric_frobenius_residual = 0.0;
    double final_projected_fock_relative_residual = 0.0;
    /// ||C C^H S - U_r U_r^H||_F / sqrt(r). The reference projector is
    /// independently reconstructed as X diag(lambda_r) X^H, not C C^H.
    double retained_projector_relative_residual = 0.0;
    int fock_scale_exponent = 0;
    std::uint64_t fock_scaling_underflow_components = 0;
    std::uint64_t energy_rescaling_underflow_count = 0;
    std::uint64_t required_node_memory_bytes = 0;
    HermitianJacobiResult overlap_eigensolver;
    HermitianJacobiResult fock_eigensolver;
};

/// In-process, move-only coefficients relative to the ordered input domain.
/// The domain is borrowed only during construction, never copied or retained;
/// callers expanding these coefficients must match its complete identity.
/// State lifetime is retained. D=0 returns usable=false and empty payloads.
/// Nonempty D with zero retained overlap rank is an explicit error, never a
/// successful zero correlation energy. No MP2 amplitudes/energy occur here.
class PeriodicCorrelationPAOSpace {
public:
    PeriodicCorrelationPAOSpace(const PeriodicCorrelationPAOSpace&) = delete;
    PeriodicCorrelationPAOSpace& operator=(const PeriodicCorrelationPAOSpace&) = delete;
    PeriodicCorrelationPAOSpace(PeriodicCorrelationPAOSpace&&) noexcept = default;
    PeriodicCorrelationPAOSpace& operator=(PeriodicCorrelationPAOSpace&&) noexcept = default;
    ~PeriodicCorrelationPAOSpace() = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationPAOSpaceContractVersion; }
    bool usable() const noexcept { return state_ && memory_.retained_dimension != 0U; }
    std::uint64_t domain_dimension() const noexcept { return memory_.domain_dimension; }
    std::uint64_t retained_dimension() const noexcept { return memory_.retained_dimension; }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& state_handle() const noexcept { return state_; }
    const PeriodicCorrelationPAOSpaceMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationPAOSpaceOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationPAOSpaceDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::string& state_identity_sha256() const noexcept { return state_digest_; }
    const std::string& domain_index_sha256() const noexcept { return domain_index_digest_; }
    const std::string& pao_domain_identity_sha256() const noexcept { return domain_digest_; }
    const std::string& payload_sha256() const noexcept { return payload_digest_; }
    const std::string& pao_space_identity_sha256() const noexcept { return identity_digest_; }
    const std::string& allocation_identity() const noexcept { return allocation_identity_; }
    std::complex<double> coefficient(std::size_t row, std::size_t orbital) const;
    double energy(std::size_t orbital) const;
    double overlap_eigenvalue(std::size_t index) const;
    /// Exact active views, throwing for consumed owners; no input matrix views.
    const std::complex<double>* coefficients_data() const;
    const double* energies_data() const;
    const double* overlap_eigenvalues_data() const;

private:
    PeriodicCorrelationPAOSpace() = default;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    PeriodicCorrelationPAOSpaceMemoryPlan memory_;
    PeriodicCorrelationPAOSpaceOptions options_;
    PeriodicCorrelationPAOSpaceDiagnostics diagnostics_;
    std::string state_digest_, domain_index_digest_, domain_digest_, payload_digest_;
    std::string identity_digest_, allocation_identity_;
    std::vector<std::complex<double>> coefficients_;
    std::vector<double> energies_, overlap_eigenvalues_;
    friend PeriodicCorrelationPAOSpace make_periodic_correlation_pao_space(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain&,
        std::uint64_t, const PeriodicCorrelationPAOSpaceOptions&);
    friend PeriodicCorrelationRealPAOSpace make_periodic_correlation_real_pao_space(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationRealPAOSpaceOptions&, const PeriodicCorrelationRealPAOSpaceCaps&);
};

/// Exact owned numerical phases for n=D,r>0:
/// overlap 32n^2+8n; compact 16n^2+16nr+8n; projected F 16nr+16r^2+24n;
/// F solve 16nr+48r^2+8n+8r; rotate 32nr+16r^2+8n+8r;
/// validation 32nr+24n+8r. Borrowed geometry 32n^2+16n is separate.
/// r=0 has only the overlap phase. D=0 has no numerical payload at all.
/// No allocation; no Eigen/BLAS or implementation-dependent hidden scratch.
PeriodicCorrelationPAOSpaceMemoryPlan plan_periodic_correlation_pao_space(
    std::uint64_t domain_dimension, std::uint64_t retained_dimension);

/// Two-stage admission: before data reads/buffers, admit overlap solve and
/// live borrowed geometry; after rank selection, admit the exact remaining
/// phase peak before allocating X. Reference inventory is charged once,
/// while known domain and worker-owned phase payloads use MPI*worker replicas.
/// Reference and domain must share the identical immutable state allocation,
/// not merely equal state contents; an additional state copy is not charged.
/// Other live caller allocations must already be in the admitted inventory.
/// Independent C^H S C, C^H F C and retained-projector gates do not demand
/// F C=S C eps outside the truncated space. Ascending energies, complex C.
PeriodicCorrelationPAOSpace make_periodic_correlation_pao_space(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationPAODomain& domain,
    std::uint64_t owned_numerical_byte_cap,
    const PeriodicCorrelationPAOSpaceOptions& options);

}  // namespace vibeqc
