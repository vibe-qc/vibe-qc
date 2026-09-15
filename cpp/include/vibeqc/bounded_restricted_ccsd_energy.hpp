#pragma once

/// Allocation-transparent REAL closed-shell CCSD correlation-energy leaf.
/// Riplinger and Neese, JCP 138, 034106 (2013), doi:10.1063/1.4773581,
/// Eq. (3), with the spatial amplitude convention used by Stanton et al.,
/// JCP 94, 4334 (1991), doi:10.1063/1.460620. This is numerical algebra,
/// not a converged solver, physical Hamiltonian certificate, or DLPNO method.
///
/// E_c = 2 sum_ia F_ia t_i^a
///     + sum_ijab [2(ia|jb)-(ib|ja)] [t_ij^ab+t_i^a t_j^b].
/// ALL ordered occupied indices and the full supplied common orthonormal
/// virtual space are used. There is no pair multiplicity or per-cell divisor.
/// In particular t_i*t_j must NOT be projected into a truncated target-pair
/// PNO space. A ragged provider reconstructs each amplitude in the COMMON
/// frame, and retains its complete immutable snapshot until this call ends.

#include "vibeqc/bounded_restricted_ccsd_target.hpp"

namespace vibeqc {

struct BoundedRestrictedCCSDEnergyInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    BoundedRestrictedCCSDRealView f_ov;  // exact o*v borrowed elements, binary64 row-major
};

struct BoundedRestrictedCCSDEnergyInventory {
    /// Other simultaneously live numerical storage, e.g. Foo/Fvv and owners
    /// not already included in Fov or either provider's declared roles.
    std::uint64_t other_live_numerical_bytes = 0;
    /// Other live fixed objects/descriptor tables. No ownership introspection
    /// or alias deduction is performed; provider numerical roles add by role.
    std::uint64_t other_live_control_bytes = 0;
    /// Positive callback-work ceiling, excluding this leaf's native overhead.
    /// The legacy shared integral-provider type has no work declaration.
    std::uint64_t maximum_integral_work_units_per_query = 0;
};

struct BoundedRestrictedCCSDEnergyCaps {
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_singles_calls = 0, maximum_doubles_calls = 0;
    std::uint64_t maximum_integral_calls = 0;
    std::uint64_t maximum_total_work_units = 0;
};

struct BoundedRestrictedCCSDEnergyMemoryPlan {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    std::uint64_t borrowed_f_ov_bytes = 0;
    std::uint64_t amplitude_retained_numerical_bytes = 0;
    std::uint64_t amplitude_maximum_transient_numerical_bytes = 0;
    std::uint64_t integral_retained_numerical_bytes = 0;
    std::uint64_t integral_maximum_transient_numerical_bytes = 0;
    std::uint64_t other_live_numerical_bytes = 0;
    /// Exactly zero: no vectors, integral/amplitude cache, residual or tau.
    /// The scalar result and compensated sums are fixed-size stack/control.
    std::uint64_t peak_owned_numerical_bytes = 0;
    /// Conservatively includes BOTH providers' declared transient ceilings.
    std::uint64_t total_live_numerical_bytes = 0;
    /// Published fixed leaf-object/scalar reservation plus caller control.
    /// Not allocator overhead, Python metadata, or arbitrary callback stacks.
    std::uint64_t fixed_inventoried_object_bytes = 0;
    std::uint64_t other_live_control_bytes = 0;
    std::uint64_t total_control_storage_bytes = 0;
    /// Exact successful-call census; zeros do not bypass any callback.
    std::uint64_t singles_calls = 0, doubles_calls = 0, integral_calls = 0;
    /// Conservative native scalar/check work, not FLOPs or elapsed time.
    std::uint64_t kernel_work_units_upper_bound = 0;
    std::uint64_t amplitude_work_units_upper_bound = 0;
    std::uint64_t integral_work_units_upper_bound = 0;
    std::uint64_t total_work_units_upper_bound = 0;
};

struct BoundedRestrictedCCSDEnergyResult {
    BoundedRestrictedCCSDEnergyMemoryPlan memory;
    double correlation_energy = 0.0;
    double singles_fock_energy = 0.0;
    double doubles_and_disconnected_energy = 0.0;
    std::uint64_t singles_calls = 0, doubles_calls = 0, integral_calls = 0;
    std::uint64_t charged_amplitude_work_units = 0;
    std::uint64_t charged_integral_work_units = 0;
    std::uint64_t charged_total_work_units = 0;
};

/// Metadata-only checked plan: positive o/v and per-query work ceilings; no
/// input scan, callback, context access or numerical allocation. Callback
/// pointers need only be valid when evaluating, not when planning.
BoundedRestrictedCCSDEnergyMemoryPlan plan_bounded_restricted_ccsd_energy(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    const BoundedRestrictedCCSDAmplitudeProvider&,
    const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedCCSDEnergyInventory&);

/// Snapshots every descriptor/provider/inventory/cap before any callback.
/// All extents and positive caps are admitted before Fov scans or callbacks.
/// Every callback is counted and its full declared work precharged BEFORE
/// invocation; exceptions/nonfinite arithmetic abort without a partial result.
/// Context payload immutability, provider work and physical provenance remain
/// caller responsibilities. Requires binary64, round-to-nearest, gradual
/// underflow and no fast/finite-only math; no console/files or hidden BLAS.
BoundedRestrictedCCSDEnergyResult bounded_restricted_ccsd_energy(
    const BoundedRestrictedCCSDEnergyInput&,
    const BoundedRestrictedCCSDAmplitudeProvider&,
    const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedCCSDEnergyInventory&,
    const BoundedRestrictedCCSDEnergyCaps&);

}  // namespace vibeqc
