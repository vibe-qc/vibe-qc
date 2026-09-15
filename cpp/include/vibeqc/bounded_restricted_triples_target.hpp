#pragma once

/// A real restricted local-triples operator/energy NUMERICAL LEAF, not an
/// iterative driver, moment producer or completed DLPNO-(T) method.
/// Guo et al., JCP 148, 011101 (2018), doi:10.1063/1.5011798, Eqs. (1)-(3):
///
/// R_ijk^abc = W_ijk^abc + (eps_a+eps_b+eps_c) T_ijk^abc
///             - sum_l [F_il P(T_ljk) + F_jl P(T_ilk) + F_kl P(T_ijl)].
/// P is the product of three target/source one-orbital overlaps. All nonzero
/// offdiagonal occupied-Fock couplings within the supplied occupied set are
/// kept, with no F_Cut or semicanonical/T0 replacement. Diagonal contributions
/// use the supplied target amplitude directly. Target virtuals must already
/// be quasi-canonical; no virtual offdiagonal terms are present in this model.
///
/// The input moments are EXPLICITLY SUPPLIED, not generated or certified here.
/// W is the connected moment of Riplinger et al., JCP 139, 134101 (2013),
/// doi:10.1063/1.4821834, Eqs. (8)-(9). U is the singles energy moment
/// t_i^a(jb|kc)+t_j^b(ia|kc)+t_k^c(jb|ia), with the CCSD(T), not QCISD(T),
/// singles coefficient (Raghavachari et al., CPL 157, 479 (1989),
/// doi:10.1016/S0009-2614(89)87395-6, Eq. (14)). This is the restricted
/// Brillouin F_ov=0 reference formula, not an open-shell/general-reference
/// triples claim. Physical input providers must establish these conditions.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vibeqc {

struct BoundedRestrictedTriplesRealView {
    const double* data = nullptr;
    std::size_t element_count = 0;
};

struct BoundedRestrictedTriplesTargetInput {
    std::uint64_t n_occupied = 0, n_virtual = 0;
    /// Ordered occupied-axis labels in [0,n_occupied), repeated labels allowed.
    /// They need not be sorted. No canonicalization or multiplicity is applied.
    std::array<std::uint64_t, 3> occupied = {0, 0, 0};
    /// Positive caller-defined identity for one immutable observation state.
    /// It can designate a Jacobi snapshot or a defined Gauss-Seidel frontier;
    /// raw equality is only a consistency check, not provenance certification.
    std::uint64_t amplitude_snapshot_id = 0;
    BoundedRestrictedTriplesRealView amplitudes;       // [v,v,v]
    BoundedRestrictedTriplesRealView connected_moment; // W [v,v,v]
    BoundedRestrictedTriplesRealView singles_moment;   // U [v,v,v]
    BoundedRestrictedTriplesRealView virtual_energies; // [v]
    BoundedRestrictedTriplesRealView occupied_fock_rows; // [3,o], F_[i,j,k],l
};

struct BoundedRestrictedTriplesNeighbourRequest {
    std::uint64_t replaced_axis = 0, replacement_occupied = 0;
    std::array<std::uint64_t, 3> ordered_occupied = {0, 0, 0};
    std::uint64_t amplitude_snapshot_id = 0;
};

struct BoundedRestrictedTriplesNeighbourView {
    std::array<std::uint64_t, 3> ordered_occupied = {0, 0, 0};
    std::uint64_t amplitude_snapshot_id = 0;
    // Zero is a PRESENT empty retained space. Both views then have exact
    // zero extent and contribute zero without skipping the provider visit.
    std::uint64_t source_virtual_dimension = 0;
    BoundedRestrictedTriplesRealView amplitudes; // [d,d,d], REQUESTED axis order
    /// Row-major [v,d], S[a,x]=<target virtual a|source virtual x>.
    /// One common orbital space is used on each of a,b,c within each triple.
    BoundedRestrictedTriplesRealView target_source_overlap;
};

using BoundedRestrictedTriplesNeighbourReceiver = void(*)(
    const BoundedRestrictedTriplesNeighbourView&, void* receiver_context);

/// Visit exactly once, synchronously, with an immutable borrowed view whose
/// lifetime contains the receiver call. The leaf never retains either pointer.
/// Missing/duplicate visits, label/snapshot mismatch and exceptions abort
/// without publishing any result. The provider must not swallow receiver
/// errors, retain the receiver/context, or invoke it from another thread.
///
/// The provider must certify physical orbital/Fock/moment/overlap validity,
/// complete retained-neighbour coverage and amplitude snapshot semantics.
/// The declared memory inventory is NOT introspection of opaque allocation.
/// Active source views are separately conservatively charged below; declared
/// retained/transient inventory is additional (aliasing may overcount safely).
/// Provider internal work requires its own bound; this leaf caps visits only.
struct BoundedRestrictedTriplesNeighbourProvider {
    void (*visit)(const BoundedRestrictedTriplesNeighbourRequest&,
                  BoundedRestrictedTriplesNeighbourReceiver,
                  void* receiver_context, void* provider_context) = nullptr;
    void* context = nullptr;
    std::uint64_t maximum_source_virtual_dimension = 0;
    std::uint64_t retained_numerical_bytes = 0;
    std::uint64_t maximum_transient_numerical_bytes = 0;
};

struct BoundedRestrictedTriplesTargetCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_total_numerical_bytes = 0;
    std::uint64_t maximum_provider_visits = 0;
    std::uint64_t maximum_kernel_work_units = 0;
};

struct BoundedRestrictedTriplesTargetMemoryPlan {
    std::uint64_t n_occupied = 0, n_virtual = 0, maximum_source_virtual_dimension = 0;
    std::uint64_t borrowed_target_bytes = 0;
    std::uint64_t active_provider_view_bytes_upper_bound = 0;
    std::uint64_t provider_retained_numerical_bytes = 0;
    std::uint64_t provider_maximum_transient_numerical_bytes = 0;
    std::uint64_t output_bytes = 0, compensation_bytes = 0, projection_workspace_bytes = 0;
    /// Exactly 8*(2v^3+d_max^2+v*d_max) explicit heap-numerical bytes.
    /// Fixed-size stack/control and allocator bookkeeping are excluded.
    std::uint64_t peak_owned_numerical_bytes = 0, total_live_numerical_bytes = 0;
    std::uint64_t provider_visits_upper_bound = 0; // 3*(o-1)
    /// Conservative scalar scan/contraction units, not provider FLOPs/time.
    std::uint64_t kernel_work_units_upper_bound = 0;
};

struct BoundedRestrictedTriplesTargetResult {
    BoundedRestrictedTriplesTargetMemoryPlan memory;
    std::array<std::uint64_t, 3> occupied = {0, 0, 0};
    std::uint64_t amplitude_snapshot_id = 0;
    std::uint64_t provider_visits = 0, exactly_zero_couplings_skipped = 0;
    std::uint64_t largest_source_virtual_dimension = 0;
    std::vector<double> residual; // [v,v,v] in the target space
    double maximum_absolute_residual = 0.0, residual_frobenius_norm = 0.0;
    /// Guo Eq.(1) SUM ONLY: sum_abc T_hat*(W+U), evaluated on INPUT T.
    /// T_hat=4Tabc-2Tacb-2Tcba-2Tbac+Tcab+Tbca. No unique-triple,
    /// spin-orbital, translation, k-point or per-cell multiplicity is applied.
    /// This is not an energy-convergence verdict.
    double raw_energy_contraction = 0.0;
};

BoundedRestrictedTriplesTargetMemoryPlan plan_bounded_restricted_triples_target(
    std::uint64_t n_occupied, std::uint64_t n_virtual,
    std::uint64_t maximum_source_virtual_dimension,
    std::uint64_t provider_retained_numerical_bytes = 0,
    std::uint64_t provider_maximum_transient_numerical_bytes = 0);

BoundedRestrictedTriplesTargetResult bounded_restricted_triples_target_residual(
    const BoundedRestrictedTriplesTargetInput&,
    const BoundedRestrictedTriplesNeighbourProvider&,
    const BoundedRestrictedTriplesTargetCaps&);

}  // namespace vibeqc
