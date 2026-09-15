#pragma once

/// Bounded real target-pair coupled-MP2 residual algebra.
/// Nejad et al., JCP 163, 214107 (2025), doi:10.1063/5.0290816,
/// Eqs. (37), (42), (44), with BOTH occupied diagonal terms extracted:
/// R_ij = G_ij + (eps_a+eps_b-f_ii-f_jj) T_ij
///        - sum_{k!=i} f_ki O_(ij,kj) T_kj O_(ij,kj)^T
///        - sum_{k!=j} f_kj O_(ij,ik) T_ik O_(ij,ik)^T.
///
/// This is numerical algebra, not a torus topology selector, a converged
/// solver or an energy. Inputs must be real, with orthonormal pair virtual
/// spaces and semicanonical target eps; upstream code must establish that
/// physical contract and a real-space gate. Double-only views do not prove
/// those facts, and no complex-to-real conversion occurs here. Integer labels
/// certify only arithmetic occupied slots/orientation, not physical provenance.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vibeqc {

enum class PairResidualOccupiedLeg : std::uint32_t { First = 0, Second = 1 };

struct PairResidualControls {
    std::uint64_t occupied_label_count = 0;
    std::uint64_t target_first = 0;
    std::uint64_t target_second = 0;
    std::uint64_t expected_coupling_count = 0;
    std::uint64_t maximum_source_dimension = 0;
    /// Exact budget for scalar multiplications in streamed projections and
    /// weighted residual updates. A source rank m>0 costs n*m*m+n*n*m+n*n;
    /// an explicit zero-rank source costs zero. Checked before source reads.
    std::uint64_t maximum_scalar_products = 0;
    /// Positive finite rejection boundary, not a shift or clipping rule.
    double denominator_floor = 0.0;
};

struct PairResidualMemoryPlan {
    std::uint64_t target_dimension = 0;
    std::uint64_t maximum_source_dimension = 0;
    std::uint64_t expected_coupling_count = 0;
    std::uint64_t borrowed_target_input_bytes = 0;
    std::uint64_t maximum_borrowed_source_input_bytes = 0;
    std::uint64_t residual_bytes = 0;
    std::uint64_t compensation_bytes = 0;
    std::uint64_t projection_workspace_bytes = 0;
    /// Exactly 16*n*n+8*n*M, including cross-source compensation. Fixed
    /// metadata/allocator bookkeeping and caller-owned inputs are separate.
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t output_numerical_bytes = 0;
    std::uint64_t maximum_scalar_products_per_source = 0;
    std::uint64_t maximum_total_scalar_products = 0;
};

struct PairResidualSource {
    PairResidualOccupiedLeg leg = PairResidualOccupiedLeg::First;
    std::uint64_t substituted_occupied = 0;
    std::uint64_t stored_first = 0;
    std::uint64_t stored_second = 0;
    std::size_t source_dimension = 0;
    double occupied_fock_coupling = 0.0;
    const double* amplitudes = nullptr;
    std::size_t amplitude_elements = 0;
    /// Row-major [n,m], O_ac=<target virtual a|stored source virtual c>.
    const double* target_source_overlap = nullptr;
    std::size_t overlap_elements = 0;
};

struct PairResidualDiagnostics {
    std::uint64_t accepted_coupling_count = 0;
    std::uint64_t scalar_product_count = 0;
    std::uint64_t transposed_source_count = 0;
    std::uint64_t zero_rank_source_count = 0;
    std::uint64_t maximum_observed_source_dimension = 0;
    std::uint64_t maximum_observed_source_input_bytes = 0;
    /// Nonzero products rounded to zero are counted, not a residual error
    /// certificate. This primitive never decides iterative convergence.
    std::uint64_t product_underflow_count = 0;
    double minimum_denominator = 0.0;
    double maximum_denominator = 0.0;
    double maximum_absolute_residual = 0.0;
    double residual_frobenius_norm = 0.0;
};

class PairResidualAccumulator;

/// Immutable finished residual; no partial-result view exists.
class PairResidualResult {
public:
    PairResidualResult(const PairResidualResult&) = delete;
    PairResidualResult& operator=(const PairResidualResult&) = delete;
    PairResidualResult(PairResidualResult&&) noexcept = default;
    PairResidualResult& operator=(PairResidualResult&&) noexcept = default;
    ~PairResidualResult() = default;
    std::uint64_t target_dimension() const noexcept { return memory_.target_dimension; }
    const PairResidualMemoryPlan& memory() const noexcept { return memory_; }
    const PairResidualControls& controls() const noexcept { return controls_; }
    const PairResidualDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    double residual(std::size_t row, std::size_t column) const;
    const double* residual_data() const;
private:
    PairResidualResult() = default;
    PairResidualMemoryPlan memory_;
    PairResidualControls controls_;
    PairResidualDiagnostics diagnostics_;
    std::vector<double> residual_;
    friend class PairResidualAccumulator;
};

/// Owns only R[n,n], compensation[n,n] and scratch[n,M]. Source callbacks
/// and their lifetime are deliberately outside this primitive. A provider
/// exception must abort/discard the accumulator; abort() is explicit.
/// Any accumulate/finish error poisons and releases the unfinished result.
/// A moved-from, aborted or finished accumulator cannot be reused.
/// The native caller must give this mutable owner exclusive access; methods
/// are not concurrently callable. Source views must remain stable per call.
class PairResidualAccumulator {
public:
    PairResidualAccumulator(const PairResidualAccumulator&) = delete;
    PairResidualAccumulator& operator=(const PairResidualAccumulator&) = delete;
    PairResidualAccumulator(PairResidualAccumulator&&) noexcept = default;
    PairResidualAccumulator& operator=(PairResidualAccumulator&&) noexcept = default;
    ~PairResidualAccumulator() = default;
    bool is_open() const noexcept;
    std::uint64_t accepted_coupling_count() const noexcept { return diagnostics_.accepted_coupling_count; }
    const PairResidualMemoryPlan& memory() const noexcept { return memory_; }
    /// Strict ascending (leg,k), with k!=i on First and k!=j on Second,
    /// rejects duplicate/self-diagonal terms. Stored source pair must equal
    /// (k,j)/(i,k) or its reversal; reversal derives T^T, never O^T input.
    /// Finite source views are borrowed only during this call. Rank-zero
    /// sources are explicit omitted spaces and still consume their slot.
    void accumulate(const PairResidualSource& source);
    void abort() noexcept;
    /// Requires exact expected term count; folds compensation, releases
    /// scratch/compensation and moves the sole R payload into the result.
    PairResidualResult finish();
private:
    PairResidualAccumulator() = default;
    enum class Status { Constructing, Open, Aborted, Finished };
    Status status_ = Status::Constructing;
    PairResidualMemoryPlan memory_;
    PairResidualControls controls_;
    PairResidualDiagnostics diagnostics_;
    bool has_previous_slot_ = false;
    PairResidualOccupiedLeg previous_leg_ = PairResidualOccupiedLeg::First;
    std::uint64_t previous_occupied_ = 0;
    std::vector<double> residual_, compensation_, workspace_;
    friend PairResidualAccumulator initialize_pair_residual(
        const double*, std::size_t, const double*, std::size_t,
        const double*, std::size_t, std::size_t, double, double,
        const PairResidualControls&, std::uint64_t);
};

/// Checked count-only plan; n>0, M>=0, no numerical allocation.
PairResidualMemoryPlan plan_pair_residual(
    std::uint64_t target_dimension, std::uint64_t maximum_source_dimension,
    std::uint64_t expected_coupling_count);

/// Borrowed row-major G[n,n], T[n,n], eps[n]. Exact positive owned cap and
/// finite/input/denominator preflights precede every size-dependent allocation.
/// Denominators use scaled compensated four-term sums, preserving subnormals.
/// No external target/source storage or resource envelope is inferred.
PairResidualAccumulator initialize_pair_residual(
    const double* exchange_integrals, std::size_t integral_elements,
    const double* target_amplitudes, std::size_t amplitude_elements,
    const double* virtual_energies, std::size_t energy_elements,
    std::size_t target_dimension, double occupied_fock_ii,
    double occupied_fock_jj, const PairResidualControls& controls,
    std::uint64_t owned_numerical_byte_cap);

}  // namespace vibeqc
