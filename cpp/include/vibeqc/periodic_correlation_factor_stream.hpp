#pragma once

/// \file periodic_correlation_factor_stream.hpp
/// \brief Constant-space q-major factor-stream contract for periodic work.
///
/// Contract version 1 describes only the ordering, identity, and bounded
/// transport of three-index factor tiles.  Its public source is an explicitly
/// synthetic, nonphysical test pattern.  It does not construct density-fitting
/// integrals and must not be used as a numerical RSGDF producer.
///
/// A v1 tile stores a synthetic payload[opaque_row, ao_pair] in row-major
/// logical order: the flattened AO-pair index is fastest and is itself
/// mu * n_basis + nu, with nu fastest.  The n_auxiliary fields name only the
/// admitted opaque-row extent used for sizing.  They do not choose between an
/// original auxiliary-AO axis and a retained metric-eigenvector axis, and v1
/// defines no numerical factor gauge.  Tiles are visited in the canonical
/// order
///
///     q, k_bra, AO-pair tile, auxiliary tile.
///
/// The schedule stores no descriptor table. Shape construction, descriptor
/// lookup, and schedule identity construction are independent of the logical
/// number of factor elements.
///
/// Canonical SHA-256 wire
/// ---------------------
/// Every digest starts with its domain string encoded as described below,
/// followed by a big-endian u32 schema version. u32/u64 are unsigned
/// big-endian integers; i32 is its two's-complement bits as a big-endian u32.
/// A string is u64 byte length followed by its bytes. A double is finite
/// IEEE-754 binary64 bits as a big-endian u64 after normalizing either signed
/// zero to +0.0. A complex value is real double followed by imaginary double.
/// No padding, native object bytes, or std::hash value enters a digest.
/// Every complete canonical message is rejected unless its byte length is at
/// most floor((2^64 - 1) / 8), preserving SHA-256's strict less-than-2^64-bit
/// input domain. The synthetic payload preflights descriptor overhead as well
/// as logical complex-value bytes before any tile is accepted.
///
/// The schedule domain is
/// "vibeqc.periodic.correlation.factor-stream.schedule". After version 1 its
/// fields are admitted-reference contract u32, allocation contract u32, state
/// digest version u32, state/calculation/allocation identity strings, three
/// mesh i32, three is_shift i32, these exact strings in order:
/// "gamma-centered-full-mesh-unreduced",
/// "q,k-bra,ao-pair-tile,auxiliary-tile",
/// "ao-pair=mu*n-basis+nu;nu-fastest",
/// "synthetic-payload[opaque-row,ao-pair];ao-pair-fastest",
/// "complex-ieee754-binary64",
/// "q-index=regular-mesh-modular-residue;physical-q-gauge-undefined",
/// "q=k-ket-k-bra;k-ket=k-bra+q-mod-G", then every Shape field in
/// declaration order as u64.  In particular, v1 does not choose a centered
/// reciprocal-space representative or a Nyquist-plane convention for a
/// future numerical G+q factor producer.
///
/// The source domain is
/// "vibeqc.periodic.correlation.factor-stream.synthetic-source". After
/// pattern version 1 it contains the schedule identity and the exact strings
/// "synthetic-nonphysical-test-pattern" and
/// "splitmix64-source-bound-to-exact-binary-fractions".
///
/// The payload domain is
/// "vibeqc.periodic.correlation.factor-stream.synthetic-payload". After
/// pattern version 1 it contains schedule identity, source identity, the
/// exact synthetic-pattern string above, tile count, logical element count,
/// and logical bytes. Each accepted tile then contributes descriptor fields
/// in declaration order (the three wrap components are i32), followed by its
/// complex values in payload order.
///
/// The receipt domain is
/// "vibeqc.periodic.correlation.factor-stream.synthetic-receipt". After
/// transaction version 1 it contains transaction version u32, pattern version
/// u32, the exact synthetic-pattern string, synthetic_nonphysical as u32 1,
/// schedule/source/payload identity strings, then committed tile, element,
/// and logical-byte counts as u64. Its own identity is not self-hashed.

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>

#include "vibeqc/periodic_correlation_admitted_reference.hpp"

namespace vibeqc {

inline constexpr std::uint32_t
    kPeriodicCorrelationFactorStreamContractVersion = 1;
inline constexpr std::uint32_t
    kPeriodicCorrelationSyntheticFactorPatternVersion = 1;
inline constexpr std::uint32_t
    kPeriodicCorrelationSyntheticFactorTransactionVersion = 1;

/// Checked, allocation-free scalar shape of the complete factor stream.
struct PeriodicCorrelationFactorStreamShape {
    std::uint64_t n_kpoints = 0;
    std::uint64_t n_basis = 0;
    std::uint64_t n_auxiliary = 0;
    std::uint64_t n_ao_pairs = 0;

    /// Resolved positive block sizes. A zero factor_ao_pair_block in the
    /// admitted resource dimensions is resolved to n_ao_pairs here.
    std::uint64_t auxiliary_block = 0;
    std::uint64_t ao_pair_block = 0;
    std::uint64_t auxiliary_tile_count = 0;
    std::uint64_t ao_pair_tile_count = 0;

    std::uint64_t tiles_per_k_bra = 0;
    std::uint64_t tiles_per_q = 0;
    std::uint64_t tile_count = 0;
    std::uint64_t logical_element_count = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t maximum_tile_element_count = 0;
    std::uint64_t maximum_tile_bytes = 0;
    /// Exact size-dependent payload workspace of the one-shot executor: one
    /// source tile and one sink-staging tile. Fixed control metadata is not
    /// included.
    std::uint64_t two_buffer_workspace_bytes = 0;
};

/// Allocation-free checked shape for a Gamma-centred full-mesh schedule.
/// ao_pair_block == 0 selects the complete n_basis^2 AO-pair panel.
PeriodicCorrelationFactorStreamShape
estimate_periodic_correlation_factor_stream_shape(
    std::array<int, 3> mesh,
    std::uint64_t n_basis,
    std::uint64_t n_auxiliary,
    std::uint64_t auxiliary_block,
    std::uint64_t ao_pair_block);

/// One O(1)-derived tile in the canonical q-major stream.
struct PeriodicCorrelationFactorTileDescriptor {
    std::uint64_t sequence_index = 0;
    std::uint64_t q_index = 0;
    std::uint64_t k_bra_index = 0;
    std::uint64_t k_ket_index = 0;
    /// Exact reciprocal-lattice wrap G satisfying k_bra + q = k_ket + G.
    std::array<int, 3> k_ket_reciprocal_wrap = {0, 0, 0};
    std::uint64_t ao_pair_begin = 0;
    std::uint64_t ao_pair_count = 0;
    std::uint64_t auxiliary_begin = 0;
    std::uint64_t auxiliary_count = 0;
    std::uint64_t element_count = 0;
};

/// Immutable, descriptor-table-free stream schedule retained against the
/// certified mean-field state from which it was admitted.
class PeriodicCorrelationFactorStreamSchedule {
public:
    PeriodicCorrelationFactorStreamSchedule(
        const PeriodicCorrelationFactorStreamSchedule&) = delete;
    PeriodicCorrelationFactorStreamSchedule& operator=(
        const PeriodicCorrelationFactorStreamSchedule&) = delete;
    PeriodicCorrelationFactorStreamSchedule(
        PeriodicCorrelationFactorStreamSchedule&&) noexcept = default;
    PeriodicCorrelationFactorStreamSchedule& operator=(
        PeriodicCorrelationFactorStreamSchedule&&) noexcept = default;
    ~PeriodicCorrelationFactorStreamSchedule() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationFactorStreamContractVersion;
    }
    const PeriodicRestrictedMeanFieldState& state() const;
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
    state_handle() const noexcept {
        return state_;
    }
    const std::string& calculation_identity() const noexcept {
        return calculation_identity_;
    }
    const std::string& allocation_identity() const noexcept {
        return allocation_identity_;
    }
    const std::string& state_identity_sha256() const noexcept {
        return state_identity_sha256_;
    }
    const std::string& schedule_identity_sha256() const noexcept {
        return schedule_identity_sha256_;
    }
    const std::array<int, 3>& mesh() const noexcept { return mesh_; }
    const std::array<int, 3>& is_shift() const noexcept { return is_shift_; }
    const PeriodicCorrelationFactorStreamShape& shape() const noexcept {
        return shape_;
    }

    PeriodicCorrelationFactorTileDescriptor descriptor(
        std::uint64_t sequence_index) const;

    /// AO-pair flattening fixed by v1: mu * n_basis + nu (nu fastest).
    std::uint64_t ao_pair_index(std::uint64_t mu,
                                std::uint64_t nu) const;
    std::array<std::uint64_t, 2> ao_pair_indices(
        std::uint64_t ao_pair_index) const;

    /// Offset within descriptor payload for global indices. Layout is
    /// opaque-row-major (named auxiliary in the v1 fields), with AO pair
    /// fastest:
    /// (auxiliary - begin) * ao_pair_count + (ao_pair - begin).
    std::uint64_t payload_offset(
        const PeriodicCorrelationFactorTileDescriptor& descriptor,
        std::uint64_t auxiliary,
        std::uint64_t ao_pair) const;

private:
    PeriodicCorrelationFactorStreamSchedule(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::string calculation_identity,
        std::string allocation_identity,
        std::string state_identity_sha256,
        std::string schedule_identity_sha256,
        std::array<int, 3> mesh,
        std::array<int, 3> is_shift,
        PeriodicCorrelationFactorStreamShape shape);

    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::string calculation_identity_;
    std::string allocation_identity_;
    std::string state_identity_sha256_;
    std::string schedule_identity_sha256_;
    std::array<int, 3> mesh_ = {0, 0, 0};
    std::array<int, 3> is_shift_ = {0, 0, 0};
    PeriodicCorrelationFactorStreamShape shape_;

    friend PeriodicCorrelationFactorStreamSchedule
    make_periodic_correlation_factor_stream_schedule(
        const PeriodicCorrelationAdmittedReference& reference);
};

/// Construct a Gamma-centred, unreduced full-mesh schedule from the sole v1
/// native correlation entry point. No pair topology or factor tensor is
/// constructed by this factory.
PeriodicCorrelationFactorStreamSchedule
make_periodic_correlation_factor_stream_schedule(
    const PeriodicCorrelationAdmittedReference& reference);

/// Deterministic nonphysical value for source/sink protocol tests only.
/// Every uint64 operation below is modulo 2^64. Define
///
///   mix(x):
///     x = x + 0x9e3779b97f4a7c15
///     x = (x xor (x >> 30)) * 0xbf58476d1ce4e5b9
///     x = (x xor (x >> 27)) * 0x94d049bb133111eb
///     return x xor (x >> 31)
///
/// Let s0, s1, s2, s3 be the four consecutive big-endian u64 words of the
/// source-identity SHA-256. Then
///
///   key = 0x6a09e667f3bcc909
///         xor mix(s0 xor 0xbb67ae8584caa73b)
///         xor mix(s1 xor 0x3c6ef372fe94f82b)
///         xor mix(s2 xor 0xa54ff53a5f1d36f1)
///         xor mix(s3 xor 0x510e527fade682d1)
///         xor mix(q         xor 0x243f6a8885a308d3)
///         xor mix(k_bra     xor 0x13198a2e03707344)
///         xor mix(auxiliary xor 0xa4093822299f31d0)
///         xor mix(ao_pair   xor 0x082efa98ec4e6c89)
///   key = mix(key)
///   real_code = (key & 0xfffff) + 1
///   imag_code = ((key >> 20) & 0xfffff) + 1
///   real = real_code * 2^-20
///   imag = imag_code * 2^-20, negated when ((key >> 40) & 1) != 0.
///
/// The codes fit exactly in binary64 and ldexp supplies the exact power-of-two
/// scaling. The wire never uses std::hash or native object representation.
std::complex<double> periodic_correlation_synthetic_factor_value(
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    std::uint64_t q_index,
    std::uint64_t k_bra_index,
    std::uint64_t auxiliary,
    std::uint64_t ao_pair);

/// Fill exactly one canonical tile with the deterministic nonphysical pattern.
/// The descriptor must exactly match schedule.descriptor(sequence_index), and
/// payload_count must equal descriptor.element_count.
void fill_periodic_correlation_synthetic_factor_tile(
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorTileDescriptor& descriptor,
    std::complex<double>* payload,
    std::uint64_t payload_count);

enum class PeriodicCorrelationSyntheticTransactionState {
    Open,
    Failed,
    Aborted,
    Committed,
};

enum class PeriodicCorrelationFactorPayloadKind {
    SyntheticNonPhysicalTestPattern,
};

/// Receipt returned only after every canonical tile has been validated and
/// committed. Both the type and payload kind deliberately advertise that this
/// is synthetic evidence, not a numerical density-fitting result.
struct PeriodicCorrelationSyntheticFactorReceipt {
    std::uint32_t transaction_contract_version = 0;
    std::uint32_t synthetic_pattern_version = 0;
    PeriodicCorrelationFactorPayloadKind payload_kind =
        PeriodicCorrelationFactorPayloadKind::
            SyntheticNonPhysicalTestPattern;
    bool synthetic_nonphysical = true;
    std::string schedule_identity_sha256;
    std::string source_identity_sha256;
    std::string payload_identity_sha256;
    std::string receipt_identity_sha256;
    std::uint64_t committed_tile_count = 0;
    std::uint64_t committed_element_count = 0;
    std::uint64_t committed_logical_bytes = 0;
};

/// Sequential, in-memory synthetic sink. It retains no tile payload and has no
/// disk, restart, retry, or concurrency contract. Any core
/// descriptor/count/cap/value validation failure while open permanently
/// transitions the transaction to Failed. abort() and a successful commit()
/// are idempotent terminal operations.
class PeriodicCorrelationSyntheticFactorTransaction {
public:
    PeriodicCorrelationSyntheticFactorTransaction(
        const PeriodicCorrelationSyntheticFactorTransaction&) = delete;
    PeriodicCorrelationSyntheticFactorTransaction& operator=(
        const PeriodicCorrelationSyntheticFactorTransaction&) = delete;
    PeriodicCorrelationSyntheticFactorTransaction(
        PeriodicCorrelationSyntheticFactorTransaction&&) noexcept;
    PeriodicCorrelationSyntheticFactorTransaction& operator=(
        PeriodicCorrelationSyntheticFactorTransaction&&) noexcept;
    ~PeriodicCorrelationSyntheticFactorTransaction();

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationSyntheticFactorTransactionVersion;
    }
    PeriodicCorrelationSyntheticTransactionState state() const noexcept;
    std::uint64_t next_sequence_index() const noexcept;
    std::uint64_t accepted_tile_count() const noexcept;
    std::uint64_t accepted_element_count() const noexcept;
    const std::string& schedule_identity_sha256() const noexcept;
    const std::string& source_identity_sha256() const noexcept;

    void accept(
        const PeriodicCorrelationFactorTileDescriptor& descriptor,
        const std::complex<double>* payload,
        std::uint64_t payload_count,
        std::uint64_t max_payload_bytes);
    PeriodicCorrelationSyntheticFactorReceipt commit();
    void abort() noexcept;

private:
    struct Impl;
    explicit PeriodicCorrelationSyntheticFactorTransaction(
        std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend PeriodicCorrelationSyntheticFactorTransaction
    make_periodic_correlation_synthetic_factor_transaction(
        const PeriodicCorrelationFactorStreamSchedule& schedule);
};

PeriodicCorrelationSyntheticFactorTransaction
make_periodic_correlation_synthetic_factor_transaction(
    const PeriodicCorrelationFactorStreamSchedule& schedule);

/// Explicit fail-closed caps for the synthetic one-shot executor. Zero means
/// missing, never unlimited.
struct PeriodicCorrelationSyntheticFactorExecutionCaps {
    std::uint64_t max_tile_bytes = 0;
    std::uint64_t max_workspace_bytes = 0;
    std::uint64_t max_tile_count = 0;
    std::uint64_t max_logical_bytes = 0;
};

/// Exact canonical synthetic-payload SHA-256 message length. This is
/// allocation-free and rejects a shape whose complete message is outside
/// SHA-256's strict less-than-2^64-bit length domain.
std::uint64_t
estimate_periodic_correlation_synthetic_payload_wire_bytes(
    const PeriodicCorrelationFactorStreamShape& shape);

/// Execute the synthetic source -> copied staging tile -> validating sink path
/// with exactly two size-dependent payload buffers. Every cap is checked before
/// allocation or iteration.
PeriodicCorrelationSyntheticFactorReceipt
execute_periodic_correlation_synthetic_factor_stream(
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationSyntheticFactorExecutionCaps& caps);

}  // namespace vibeqc
