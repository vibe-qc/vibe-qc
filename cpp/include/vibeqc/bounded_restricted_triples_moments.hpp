#pragma once

/// Real restricted triples moments in one supplied common orthonormal LOCAL
/// virtual space. Riplinger et al., doi:10.1063/1.4821834, Eqs.(8)-(9):
/// W_ijk^abc = P6 [sum_d t_kj^cd (ia|bd) - sum_l t_il^ab (kc|jl)].
/// P6 simultaneously permutes occupied and virtual axes, including all six
/// terms for repeated labels. Guo et al., doi:10.1063/1.5011798, Eq.(1):
/// U_ijk^abc = t_i^a(jb|kc)+t_j^b(ia|kc)+t_k^c(jb|ia).
/// U uses CCSD(T)'s singles coefficient 1, not QCISD(T)'s coefficient 2.
/// Neither denominator nor unique-triple/virtual/spin/cell weight is applied.
///
/// Scientific scope: real closed-shell Brillouin reference F_ov=0. This is
/// NOT generic non-HF triples, a pair-specific TNO projection, or a complete
/// periodic/DLPNO triples driver. Local T1/T2 must already be represented in
/// the same common space, with every retained occupied coupling included.
/// Input providers must independently establish orbital, integral permutation,
/// Hamiltonian and converged-amplitude validity. This leaf only computes the
/// stated moments of supplied numbers; it does not certify those conditions.

#include <array>
#include "vibeqc/bounded_restricted_ccsd_target.hpp"

namespace vibeqc {

struct BoundedRestrictedTriplesMomentsInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    std::array<std::uint64_t, 3> occupied = {0, 0, 0};
    /// Positive caller sequencing label, not payload provenance/certification.
    std::uint64_t amplitude_snapshot_id = 0;
    BoundedRestrictedCCSDRealView t1;   // [o,v]
    BoundedRestrictedCCSDRealView t2;   // [o,o,v,v], restricted alpha-beta
    BoundedRestrictedCCSDRealView f_ov; // [o,v], must be EXACTLY zero
    /// Finite nonnegative relative-to-max(1,|x|,|y|) input audit tolerance
    /// for t_ijab=t_jiba. Zero requests exact equality; no averaging/repair.
    double amplitude_symmetry_tolerance = 0.0;
};

struct BoundedRestrictedTriplesMomentsCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_integral_calls = 0;
    std::uint64_t maximum_kernel_work_units = 0;
};

struct BoundedRestrictedTriplesMomentsMemoryPlan {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    std::uint64_t borrowed_input_bytes = 0;
    std::uint64_t provider_retained_numerical_bytes = 0;
    std::uint64_t provider_maximum_transient_numerical_bytes = 0;
    std::uint64_t connected_output_bytes = 0, singles_output_bytes = 0;
    /// Exactly two v^3 output arrays: 16*v^3 explicit heap numerical bytes.
    /// Compensated contraction uses scalar stack state, not a cube workspace.
    /// Fixed-size control/stack and allocator bookkeeping are excluded.
    std::uint64_t peak_owned_numerical_bytes = 0, total_live_numerical_bytes = 0;
    /// EXACT v^3*(6*(v+o)+3) calls: no zero or repeated-label shortcut.
    std::uint64_t integral_calls = 0;
    /// 2ov+2o^2v^2+128*calls+64v^3+256 conservative scalar units.
    /// Includes finite/Fov-zero/symmetry audits; opaque callback work needs
    /// its own bound. Provider memory is declared, not introspected.
    std::uint64_t kernel_work_units_upper_bound = 0;
};

struct BoundedRestrictedTriplesMomentsResult {
    BoundedRestrictedTriplesMomentsMemoryPlan memory;
    std::array<std::uint64_t, 3> occupied = {0, 0, 0};
    std::uint64_t amplitude_snapshot_id = 0, integral_calls = 0;
    std::vector<double> connected; // W, row-major [v,v,v]
    std::vector<double> singles;   // U, row-major [v,v,v]
};

BoundedRestrictedTriplesMomentsMemoryPlan plan_bounded_restricted_triples_moments(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    std::uint64_t provider_retained_numerical_bytes = 0,
    std::uint64_t provider_maximum_transient_numerical_bytes = 0);

/// All positive caps, overflow/extents and scalar options are admitted before
/// scans/allocation/provider calls. Borrowed aligned contiguous arrays remain
/// immutable through return. Exact nonzero Fov, nonfinite values, invalid
/// labels, callback errors and arithmetic overflow abort with no partial
/// output. No full integral/T2 copies, files, C++ output or hidden workers.
BoundedRestrictedTriplesMomentsResult bounded_restricted_triples_moments(
    const BoundedRestrictedTriplesMomentsInput&,
    const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedTriplesMomentsCaps&);

/// Scalar-accessor counterpart of the dense input. No common T1/T2 array is
/// required or materialized. The provider represents amplitudes in this SAME
/// common orthonormal space; any projection belongs to its independently
/// validated source, not to the moments contractions. Fov must still be exactly
/// zero: an upstream physical driver must explicitly budget a Brillouin
/// projection rather than silently passing a non-HF operator here.
struct BoundedRestrictedTriplesMomentsOperatorInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    std::array<std::uint64_t, 3> occupied = {0, 0, 0};
    std::uint64_t amplitude_snapshot_id = 0; // sequencing only, never provenance
    BoundedRestrictedCCSDRealView f_ov;
    /// Relative-to-max(1,|x|,|y|) reverse-symmetry audit for CONSUMED pairs only.
    double amplitude_symmetry_tolerance = 0.0;
};
struct BoundedRestrictedTriplesMomentsAccessorCaps {
    BoundedRestrictedTriplesMomentsCaps kernel;
    std::uint64_t maximum_singles_calls = 0, maximum_doubles_calls = 0;
    std::uint64_t maximum_amplitude_work_units = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
};
struct BoundedRestrictedTriplesMomentsAccessorMemoryPlan {
    /// Unchanged two-cube heap, 16v^3 bytes. Borrowed input here is only 8ov
    /// (Fov); total_live also includes both declared amplitude-provider roles.
    BoundedRestrictedTriplesMomentsMemoryPlan kernel;
    std::uint64_t amplitude_retained_numerical_bytes = 0;
    std::uint64_t amplitude_maximum_transient_numerical_bytes = 0;
    std::uint64_t singles_calls = 0; // S=3v^3, exact (also for zero amplitudes)
    /// Each is D=6v^3(o+v). Every contraction query additionally evaluates its
    /// reverse, including coincident labels: 2D doubles calls, no cache.
    std::uint64_t contraction_doubles_calls = 0, reverse_audit_doubles_calls = 0;
    std::uint64_t doubles_calls = 0, amplitude_work_units_upper_bound = 0;
    /// Conservative fixed C++ control/stack reservation, admitted separately;
    /// not allocator/backend RSS. Kernel work is ov+128I+64v^3+256+64(S+2D),
    /// I=D+S. Opaque callback work is charged separately as S*w1+2D*w2.
    std::uint64_t fixed_control_storage_bytes = 0;
};
struct BoundedRestrictedTriplesMomentsAccessorResult {
    BoundedRestrictedTriplesMomentsAccessorMemoryPlan memory;
    BoundedRestrictedTriplesMomentsResult moments;
    std::uint64_t singles_calls = 0, doubles_calls = 0;
    std::uint64_t reverse_symmetry_audits = 0, charged_amplitude_work_units = 0;
    /// Largest scaled difference over consumed pairs and their reverse queries.
    /// This does NOT audit unused amplitudes or prove global snapshot covariance.
    double maximum_consumed_amplitude_symmetry_error = 0.0;
};
BoundedRestrictedTriplesMomentsAccessorMemoryPlan plan_bounded_restricted_triples_moments_accessor(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    const BoundedRestrictedCCSDAmplitudeProvider&,
    std::uint64_t integral_retained_numerical_bytes = 0,
    std::uint64_t integral_maximum_transient_numerical_bytes = 0);
/// Explicitly CONSUMED-ONLY audit: each T2 query also requests T_ji^ba and
/// rejects a difference above the supplied tolerance, returning the ORIGINAL
/// forward value unchanged. Unqueried T1/T2 values are neither scanned nor
/// certified. The dense API above retains its original global array audit;
/// an immutable ragged reader must independently validate global frames and
/// pair covariance before exposing them as a physical amplitude source.
///
/// All metadata/byte/control/work/call limits precede scans/allocation/callbacks.
/// Dimensions, descriptors, providers and controls are copied before callbacks,
/// not their opaque context/payload. Per-query count/work precedes invocation;
/// returned values must be finite. Borrowed data/context must stay immutable,
/// but this algebra leaf does not claim to certify opaque context immutability.
/// Requires nearest IEEE binary64, FLT_EVAL_METHOD=0, no fast math and gradual
/// underflow; the environment is rechecked after each opaque callback. Errors
/// and callback exceptions abort without publishing partial cubes.
BoundedRestrictedTriplesMomentsAccessorResult bounded_restricted_triples_moments_accessor(
    const BoundedRestrictedTriplesMomentsOperatorInput&,
    const BoundedRestrictedCCSDAmplitudeProvider&,
    const BoundedRestrictedCCSDIntegralProvider&,
    const BoundedRestrictedTriplesMomentsAccessorCaps&);

}  // namespace vibeqc
