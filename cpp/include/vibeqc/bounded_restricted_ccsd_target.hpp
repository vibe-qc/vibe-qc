#pragma once

/// A bounded REAL spatial CCSD residual numerical leaf, not an energy method
/// or a Hamiltonian certificate. The equations are the validated closed-shell
/// spin integration in ccsd.cpp of Stanton et al., JCP 94, 4334 (1991),
/// doi:10.1063/1.460620, Eqs. (1)-(13). Fock diagonal terms remain on the RHS,
/// so these are true residuals, not Jacobi numerators.
///
/// The occupied set and common orthonormal virtual space must be prepared by
/// the caller. For a local calculation they are the selected coupled occupied
/// set and its independently constructed EXTENDED virtual domain. All coupled
/// amplitudes must be represented there when consumed by this leaf; project the
/// returned residual to target PNOs only afterwards. Projecting the coupled
/// inputs into the target pair's PNO space loses mixed-pair terms (Riplinger
/// and Neese, JCP 138, 034106 (2013), doi:10.1063/1.4773581, Eq. (26)).
///
/// The dense API borrows T2 within the LOCAL occupied/extended virtual space.
/// The separate accessor API reads common-frame amplitudes scalarwise, without
/// a dense T2 allocation or scan. Both APIs use the SAME arithmetic kernel.
/// The caller must admit all actual amplitude/integral-provider storage.
/// This callback reference path deliberately trades repeated integral calls
/// for bounded memory; it is not a claim of scalable DLPNO implementation.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vibeqc {

struct BoundedRestrictedCCSDRealView {
    const double* data = nullptr;
    std::size_t element_count = 0;
};

/// All views are aligned contiguous row-major binary64 and borrowed immutable
/// until return. T2 uses alpha-beta amplitudes [i,j,a,b], not antisymmetrized
/// spin-orbital amplitudes. Extra supplied elements are not read or admitted.
struct BoundedRestrictedCCSDTargetInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    std::uint64_t target_i = 0, target_j = 0;
    BoundedRestrictedCCSDRealView t1;    // [o,v]
    BoundedRestrictedCCSDRealView t2;    // [o,o,v,v]
    BoundedRestrictedCCSDRealView f_oo;  // [o,o]
    BoundedRestrictedCCSDRealView f_vv;  // [v,v]
    BoundedRestrictedCCSDRealView f_ov;  // [o,v]
};

/// Operator-only common-frame input for the scalar-amplitude path. The caller
/// keeps the Fock views immutable until return; no amplitude array is present.
struct BoundedRestrictedCCSDTargetOperatorInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    std::uint64_t target_i = 0, target_j = 0;
    BoundedRestrictedCCSDRealView f_oo, f_vv, f_ov;
};

/// Common EXTENDED-frame spatial alpha-beta amplitudes. Implementations may
/// reconstruct a scalar from ragged pair coefficients, but must not project
/// coupled amplitudes to the target pair's truncated PNO space. Calls are
/// serial/synchronous and individually counted, precharged and finite-checked.
/// Const context expresses a contract, not mutation detection or provenance.
/// Numerical storage/work are caller declarations, not callback introspection;
/// both positive per-query work ceilings exclude this leaf's native overhead.
struct BoundedRestrictedCCSDAmplitudeProvider {
    double (*singles)(std::uint64_t i, std::uint64_t a, const void* context) = nullptr;
    double (*doubles)(std::uint64_t i, std::uint64_t j, std::uint64_t a,
                      std::uint64_t b, const void* context) = nullptr;
    const void* context = nullptr;
    std::uint64_t retained_numerical_bytes = 0;
    std::uint64_t maximum_transient_numerical_bytes = 0;
    std::uint64_t maximum_singles_work_units_per_query = 0;
    std::uint64_t maximum_doubles_work_units_per_query = 0;
};

/// Returns the REAL chemists' integral (pq|rs), combined orbital labels:
/// occupied [0,o), extended virtual [o,o+v). Calls are serial and synchronous.
/// No permutation or caching shortcut is applied by the leaf. Every returned
/// scalar is checked finite and every invocation counted/capped. Exceptions
/// abort the evaluation; no partial result is published.
///
/// The provider must independently establish real-orbital/Fock and integral
/// permutation validity before a physical use. In particular complex periodic
/// factors cannot be passed through a molecular B^T B contraction or made real
/// by dropping their imaginary lanes. Memory below is a CALLER DECLARATION,
/// not introspection or certification of opaque callback allocation. Internal
/// callback work requires its own cap; this leaf caps callback count only.
struct BoundedRestrictedCCSDIntegralProvider {
    double (*value)(std::uint64_t p, std::uint64_t q, std::uint64_t r,
                    std::uint64_t s, void* context) = nullptr;
    void* context = nullptr;
    std::uint64_t retained_numerical_bytes = 0;
    std::uint64_t maximum_transient_numerical_bytes = 0;
};

struct BoundedRestrictedCCSDTargetCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_integral_calls = 0;
    std::uint64_t maximum_kernel_work_units = 0;
};

struct BoundedRestrictedCCSDTargetMemoryPlan {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    std::uint64_t borrowed_input_bytes = 0;
    std::uint64_t provider_retained_numerical_bytes = 0;
    std::uint64_t provider_maximum_transient_numerical_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t workspace_bytes = 0;
    /// Exactly 8*(2v^2+v+3ov+2o) explicitly owned heap-numerical bytes.
    /// Fixed-size scalar/stack control and allocator bookkeeping are excluded.
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t total_live_numerical_bytes = 0;
    /// Conservative bound, not a promise of callback FLOPs or elapsed time.
    std::uint64_t integral_calls_upper_bound = 0;
    /// Dense API: input scalars plus 128*integral_calls_upper_bound + 64*L, where
    /// L=o^2*v^2+o*v^2+v^3+o*v+v^2+o+v. This conservatively bounds native
    /// scalar loop/contraction work, excluding opaque callback internals.
    /// The accessor API counts only Fock input scans and additionally charges
    /// 16*(singles_calls_upper_bound+doubles_calls_upper_bound) for its checks.
    std::uint64_t kernel_work_units_upper_bound = 0;
    /// True ONLY for explicitly named remainder APIs below. Their doubles
    /// exclude all bare W1/W2/WX seeds in BOTH particle-hole passes, not the
    /// dressed intermediates or explicit T1 products. Never treat that
    /// remainder alone as a complete target residual.
    bool bare_particle_hole_excluded = false;
    /// Exact removed callback count 6*o*v^2 and corresponding conservative
    /// kernel-bound decrement 128 times that count. Not a FLOP measurement.
    std::uint64_t omitted_bare_integral_calls = 0, omitted_bare_kernel_work_units = 0;
};

struct BoundedRestrictedCCSDTargetResult {
    BoundedRestrictedCCSDTargetMemoryPlan memory;
    std::uint64_t target_i = 0, target_j = 0;
    std::uint64_t integral_calls = 0;
    std::vector<double> singles;  // [v], target_i only
    std::vector<double> doubles;  // [v,v], target_i,target_j only
    bool complete_target_residual() const noexcept { return !memory.bare_particle_hole_excluded; }
};

struct BoundedRestrictedCCSDTargetAccessorCaps {
    BoundedRestrictedCCSDTargetCaps kernel;
    std::uint64_t maximum_singles_calls = 0, maximum_doubles_calls = 0;
    std::uint64_t maximum_amplitude_work_units = 0;
};
struct BoundedRestrictedCCSDTargetAccessorMemoryPlan {
    /// Same owned kernel bytes/integral bound; borrowed_input_bytes includes
    /// only Foo/Fvv/Fov. Total includes BOTH providers' retained/transient roles.
    BoundedRestrictedCCSDTargetMemoryPlan kernel;
    std::uint64_t amplitude_retained_numerical_bytes = 0;
    std::uint64_t amplitude_maximum_transient_numerical_bytes = 0;
    /// Exact source-level traversal census, independent of target and values.
    std::uint64_t singles_calls_upper_bound = 0, doubles_calls_upper_bound = 0;
    std::uint64_t amplitude_work_units_upper_bound = 0;
};
struct BoundedRestrictedCCSDTargetAccessorResult {
    BoundedRestrictedCCSDTargetAccessorMemoryPlan memory;
    BoundedRestrictedCCSDTargetResult target;
    std::uint64_t singles_calls = 0, doubles_calls = 0;
    std::uint64_t charged_amplitude_work_units = 0;
};

/// Allocation-free overflow/address-extent plan; no input scan or callback.
BoundedRestrictedCCSDTargetMemoryPlan plan_bounded_restricted_ccsd_target(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    std::uint64_t provider_retained_numerical_bytes = 0,
    std::uint64_t provider_maximum_transient_numerical_bytes = 0);

/// All extents, declared provider inventory and positive caps are checked
/// before size-dependent scans, callbacks or allocations. No integral blocks,
/// full-shaped residuals, tau tensors, Eigen/BLAS workspace, files or output.
BoundedRestrictedCCSDTargetResult bounded_restricted_ccsd_target_residual(
    const BoundedRestrictedCCSDTargetInput&,
    const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedCCSDTargetCaps&);

/// Allocation-free count/inventory plan. No callbacks, context inspection or
/// amplitude-array extents. Positive per-query work declarations are required.
BoundedRestrictedCCSDTargetAccessorMemoryPlan plan_bounded_restricted_ccsd_target_accessor(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    const BoundedRestrictedCCSDAmplitudeProvider&,
    std::uint64_t integral_provider_retained_numerical_bytes = 0,
    std::uint64_t integral_provider_maximum_transient_numerical_bytes = 0);

/// Copies input descriptors/providers/caps before any callback. Context data
/// immutability is the caller's responsibility; no silent state certificate.
BoundedRestrictedCCSDTargetAccessorResult bounded_restricted_ccsd_target_residual_accessor(
    const BoundedRestrictedCCSDTargetOperatorInput&,
    const BoundedRestrictedCCSDAmplitudeProvider&,
    const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedCCSDTargetAccessorCaps&);

/// Explicitly INCOMPLETE target-remainder APIs for native local-frame
/// replacement of the complete bare particle-hole group. The existing APIs
/// above remain full, with unchanged arithmetic, query order and counts.
/// These skip ONLY three amplitude-independent ERIs and four seed additions
/// per (pass,b,m,e), BEFORE W1/W2/WX are contracted with amplitudes. They do
/// not evaluate/subtract a bare residual, truncate any coupled amplitude,
/// change singles, or remove any T1/T2-dependent dressing. All amplitude
/// queries and allocations remain unchanged; exactly 6*o*v^2 fewer integral
/// callbacks occur even for equal occupied targets and zero amplitudes.
/// Project the COMPLETE remainder to the target frame, release it, then add
/// one complete local-frame bare group before residual norms or updates.
/// This is neither a complete residual nor an enabled local CCSD solver.
BoundedRestrictedCCSDTargetMemoryPlan plan_bounded_restricted_ccsd_target_without_bare_particle_hole(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    std::uint64_t provider_retained_numerical_bytes = 0,
    std::uint64_t provider_maximum_transient_numerical_bytes = 0);
BoundedRestrictedCCSDTargetAccessorMemoryPlan plan_bounded_restricted_ccsd_target_accessor_without_bare_particle_hole(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    const BoundedRestrictedCCSDAmplitudeProvider&,
    std::uint64_t integral_provider_retained_numerical_bytes = 0,
    std::uint64_t integral_provider_maximum_transient_numerical_bytes = 0);
BoundedRestrictedCCSDTargetResult bounded_restricted_ccsd_target_residual_without_bare_particle_hole(
    const BoundedRestrictedCCSDTargetInput&, const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedCCSDTargetCaps&);
BoundedRestrictedCCSDTargetAccessorResult bounded_restricted_ccsd_target_residual_accessor_without_bare_particle_hole(
    const BoundedRestrictedCCSDTargetOperatorInput&, const BoundedRestrictedCCSDAmplitudeProvider&,
    const BoundedRestrictedCCSDIntegralProvider&, const BoundedRestrictedCCSDTargetAccessorCaps&);

}  // namespace vibeqc
