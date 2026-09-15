#pragma once

/// Bounded all-k p=4 Pipek-Mezey objective and REAL Euclidean derivative.
/// Zhu and Tew, doi:10.1021/acs.jpca.4c04555, Eqs.(19),(20),(26),(27).
/// Each native IAO owner supplies unorthogonalized B(k)=G(k)^-1 D(k) and
/// D(k)=A(k)^H S(k) C_occ(k). Only explicit active occupied columns rotate.
/// v_R=Nk^-1 sum_k exp(ik.R) B_active(k) U(k), w_R likewise using D(k).
/// Q_An(R)=Re sum_{rho on A} conj(v_R[rho,n])*w_R[rho,n], f=sum Q^4.
/// The returned E obeys df=Re sum_k Tr(E(k)^H dU(k)). This convention is
/// explicit: it is NOT half this derivative under a Wirtinger convention.
/// For physically real orbitals this is the real PM charge; complex inputs
/// use the stated real Mulliken extension. No time-reversal, physical overlap
/// source, localization convergence or optimizer certificate is implied.

#include "vibeqc/periodic_correlation_bloch_iao.hpp"

namespace vibeqc {

struct PeriodicCorrelationIAOPMOptions {
    double gauge_unitarity_tolerance = 0.0;
    double charge_normalization_tolerance = 0.0;
};

struct PeriodicCorrelationIAOPMCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicCorrelationIAOPMMemoryPlan {
    std::uint64_t n_points = 0, n_minimal = 0, n_active = 0, n_occupied = 0;
    std::uint64_t gradient_bytes = 0, gradient_compensation_bytes = 0;
    std::uint64_t cell_workspace_bytes = 0, active_index_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0, borrowed_gauge_bytes = 0;
    std::uint64_t borrowed_owner_pointer_bytes = 0, live_iao_numerical_bytes = 0;
    std::uint64_t required_node_memory_bytes = 0, work_units = 0;
};

struct PeriodicCorrelationIAOPMResult {
    PeriodicCorrelationIAOPMMemoryPlan memory;
    double objective = 0.0;
    double maximum_charge_normalization_residual = 0.0;
    double maximum_charge_imaginary_magnitude = 0.0;
    double maximum_unitarity_residual = 0.0;
    /// Ambient Euclidean norm, NOT a unitary-manifold convergence criterion.
    /// Even the stationary one-orbital phase problem has this norm equal to 8.
    double gradient_frobenius_norm = 0.0;
    std::vector<std::complex<double>> gradient; // [Nk,nactive,nactive]
};

/// Count-only admission includes every live one-k IAO numerical output,
/// borrowed pointers/gauges, the reference once and worker replicas. Other
/// live caller storage must be in the admitted inventory. No all-cell charge
/// tensor or full placed-orbital coefficients are retained. Direct finite
/// Fourier sums are a bounded reference algorithm, not an FFT scaling claim.
PeriodicCorrelationIAOPMMemoryPlan plan_periodic_correlation_iao_pm(
    const PeriodicCorrelationAdmittedReference&, std::uint64_t n_minimal);

/// points[k] must be a live immutable native IAO for exactly k, all sharing
/// this reference allocation, minimal-basis declaration and atomic labeling.
/// Positive caps precede gauge scans or allocation. The raw pointer array and
/// gauges are borrowed for the call; no input pointer is retained. A caller
/// must not mutate any input or controls while evaluation is running.
PeriodicCorrelationIAOPMResult evaluate_periodic_correlation_iao_pm(
    const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationBlochIAO* const* points, std::size_t point_count,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    const PeriodicCorrelationIAOPMOptions&, const PeriodicCorrelationIAOPMCaps&);

}  // namespace vibeqc
