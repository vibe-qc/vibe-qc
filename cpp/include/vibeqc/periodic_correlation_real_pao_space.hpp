#pragma once

/// Explicit real-gauge preparation, separate from the default complex PAO
/// path. Canonical X=U_r lambda_r^-1/2 is Lowdin (1970), Sec.II.C,
/// Eqs.(35)-(37), doi:10.1016/S0065-3276(08)60339-1; local Fock rotation is
/// Riplinger/Neese (2013), Sec.II.B.3, doi:10.1063/1.4773581. Expanded PAOs
/// inherit Nejad (2025), Eqs.(26)-(28), doi:10.1063/5.0290816.
///
/// The original domain must already have qualified its actual projected
/// virtual columns' time reversal and its S/F reality. S0=Re(S), F0=Re(F)
/// are nevertheless explicit, measured operator projections, not casts
/// presumed exact. Exactly real eigensolver inputs preserve a real gauge in
/// degenerate spaces. Final metric, Fock and retained-projector relations
/// are inspected against ORIGINAL complex S/F; actual expanded virtual
/// coefficients are independently tested for TR, including every TRIM.
///
/// Rank decisions use outward original-S eigenvalue intervals. If
/// g>=||U^T U-I||_F<1, e>=1-sqrt(1-g), t>=||S0-U Lambda U^T||_F and
/// deltaS>=||S-S0||_F, polar decomposition followed by Weyl gives
/// epsilon=deltaS+t+max|lambda|*(2e+e^2). Each original eigenvalue lies in
/// [lambda-epsilon,lambda+epsilon]. Propagating lambda_max's interval
/// through the strict scientific cutoff permits only unambiguous keep or
/// discard decisions; uncertain rank/negative boundaries fail closed.
/// U_r belongs to PROJECTED S0. C C^T S-U_r U_r^T is a measured relation,
/// not a claim that U_r is the original-S spectral projector.

#include <utility>
#include "vibeqc/periodic_correlation_pao_space.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationRealPAOSpaceContractVersion = 1;

struct PeriodicCorrelationRealPAOSpaceOptions {
    /// Existing explicit rank, negative, algebra-validation and Jacobi
    /// controls. The complex factory and its wire are not modified.
    PeriodicCorrelationPAOSpaceOptions algebra;
    /// All Frobenius/error budgets are finite strictly positive. Fock
    /// budgets use physical Hartree matrix units, not scaled solver units.
    double maximum_overlap_projection_error = 0.0;
    double maximum_fock_projection_error = 0.0;
    double maximum_overlap_spectral_uncertainty = 0.0;
    double maximum_original_metric_error = 0.0;
    double maximum_original_fock_error = 0.0;
    double maximum_original_projector_relation_error = 0.0;
    /// Finite [0,1), not both zero. These test actual expanded C(k), not
    /// merely the real-valued domain coefficient array.
    double coefficient_tr_absolute_tolerance = 0.0;
    double coefficient_tr_relative_tolerance = 0.0;
};

struct PeriodicCorrelationRealPAOSpaceCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicCorrelationRealPAOSpaceMemoryPlan {
    /// Transparent phase inventories, including borrowed original domain
    /// and the standard compact PAOSpace output payload. Original-matrix
    /// interval validation uses 32Dr+40D+8r bytes (32D column scratch).
    PeriodicCorrelationPAOSpaceMemoryPlan compact;
    std::uint64_t n_cells = 0, n_basis = 0;
    /// After unrotated X is freed: standard output plus two nao coefficient
    /// columns and two nao scratch columns, 64*nao extra bytes.
    std::uint64_t coefficient_tr_phase_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    /// Conservative bounded scalar/Jacobi work, not a timing promise.
    /// r=0 admits only the initial overlap/rank stage. The discovered-r
    /// plan includes that already executed stage and all remaining work.
    std::uint64_t work_units = 0;
    std::uint64_t required_node_memory_bytes = 0;
};

struct PeriodicCorrelationRealPAOSpaceDiagnostics {
    double overlap_projection_frobenius_upper_bound = 0.0;
    double fock_projection_frobenius_upper_bound = 0.0;
    double overlap_eigenvector_gram_error_upper_bound = 0.0;
    double overlap_reconstruction_error_upper_bound = 0.0;
    double overlap_polar_distance_upper_bound = 0.0;
    double overlap_eigenvalue_error_upper_bound = 0.0;
    double rank_cutoff_lower_bound = 0.0, rank_cutoff_upper_bound = 0.0;
    double negative_tolerance_lower_bound = 0.0;
    double minimum_rank_margin_lower_bound = 0.0;
    double projected_projector_reconstruction_error_upper_bound = 0.0;
    double original_metric_frobenius_error_upper_bound = 0.0;
    double original_fock_frobenius_error_upper_bound = 0.0;
    double original_projector_relation_frobenius_error_upper_bound = 0.0;
    std::uint64_t inspected_kpoints = 0, inspected_trim_points = 0;
    double maximum_coefficient_tr_error = 0.0;
    double coefficient_tr_frobenius_upper_bound = 0.0;
};

/// Owns exactly the ordinary compact PAOSpace payload, not an additional
/// coefficient copy. space() is a borrowed immutable view; both C++ and
/// Python callers must keep this owner alive (reference_internal at the
/// diagnostic Python seam). No move-out accessor can detach the proof.
class PeriodicCorrelationRealPAOSpace {
public:
    PeriodicCorrelationRealPAOSpace(const PeriodicCorrelationRealPAOSpace&) = delete;
    PeriodicCorrelationRealPAOSpace& operator=(const PeriodicCorrelationRealPAOSpace&) = delete;
    PeriodicCorrelationRealPAOSpace(PeriodicCorrelationRealPAOSpace&&) noexcept = default;
    PeriodicCorrelationRealPAOSpace& operator=(PeriodicCorrelationRealPAOSpace&&) noexcept = default;
    std::uint32_t contract_version() const noexcept { return kPeriodicCorrelationRealPAOSpaceContractVersion; }
    const PeriodicCorrelationPAOSpace& space() const;
    const PeriodicCorrelationRealPAOSpaceOptions& options() const noexcept { return options_; }
    const PeriodicCorrelationRealPAOSpaceMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationRealPAOSpaceDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
    bool original_operators_unchanged_certified() const noexcept { return false; }
private:
    explicit PeriodicCorrelationRealPAOSpace(PeriodicCorrelationPAOSpace&& space) : space_(std::move(space)) {}
    PeriodicCorrelationPAOSpace space_;
    PeriodicCorrelationRealPAOSpaceOptions options_;
    PeriodicCorrelationRealPAOSpaceMemoryPlan memory_;
    PeriodicCorrelationRealPAOSpaceDiagnostics diagnostics_;
    std::string identity_;
    friend PeriodicCorrelationRealPAOSpace make_periodic_correlation_real_pao_space(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationRealPAOSpaceOptions&, const PeriodicCorrelationRealPAOSpaceCaps&);
};

/// r=0 is an initial-stage plan, not a successful empty-space result. Empty
/// domains, zero retained rank, and ambiguous interval decisions are errors.
PeriodicCorrelationRealPAOSpaceMemoryPlan plan_periodic_correlation_real_pao_space(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain&,
    std::uint64_t retained_dimension, const PeriodicCorrelationRealPAOSpaceOptions&);
PeriodicCorrelationRealPAOSpace make_periodic_correlation_real_pao_space(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationRealPAOSpaceOptions&, const PeriodicCorrelationRealPAOSpaceCaps&);

}  // namespace vibeqc
