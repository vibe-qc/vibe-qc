#pragma once

/// \file periodic_correlation_reciprocal_metric.hpp
/// \brief Exact one-q reciprocal source manifest and admitted raw metric.
///
/// Contract version 1 covers only the full-Coulomb, all-reciprocal reference
/// producer.  It enumerates the shifted sphere
///
///     p = B (n + f_q),    |p| <= sqrt(2 E_cut),    n in Z^3,
///
/// for one centred transfer representative f_q in [-1/2, 1/2).  The
/// Gamma G=0 term is omitted by its exact integer labels, never by comparing
/// a floating-point norm with zero.  A non-Gamma boundary receives the fixed
/// radial slack
///
///     128 epsilon max(sqrt(2 E_cut) + |q|, 1),
///
/// while exact Gamma receives no slack.  Enumeration is lexicographic with
/// n_0 outermost and n_2 innermost.  The factory makes two constant-space
/// passes: one to count accepted vectors and one to stream their canonical
/// records into SHA-256.  It never materialises the reciprocal-vector list.
///
/// Skew-safe bounds are certified around the exact binary64 predicate rather
/// than padded heuristically. The implementation evaluates every three-term
/// dot product and squared norm in a fixed `std::fma` order. Outward binary64
/// intervals enclose all cofactors and independent determinant expansions of
/// `B / max(abs(B))`; certified inverse row bounds then close the matvec-error
/// feedback loop. A determinant interval containing zero or a feedback bound
/// that is not strictly below one fails closed with a distinct diagnostic.
/// Gradual-underflow allowances are derived from the three rounded operations.
/// The final ceil/floor bounds also enclose rounding in `n_i + f_i`; there is
/// no fixed integer pad. Labels outside binary64's exact integer range are
/// rejected because silently rounding an integer label would change the
/// declared source.
/// The resulting Cartesian-box product is checked against both uint64 and a
/// caller-supplied positive candidate cap before either enumeration pass.
///
/// Canonical source-identity wire
/// --------------------------------
/// Integers are big-endian.  Signed i32/i64 values use their two's-complement
/// bits.  Strings are a u64 byte length followed by bytes.  Finite binary64
/// values are encoded as big-endian IEEE-754 bits after normalising either
/// signed zero to +0.0.  There is no native-object padding.
///
/// The source wire is, in order:
///
///   * domain "vibeqc.periodic.correlation.reciprocal-metric.source" and
///     source-contract version;
///   * compiled producer ID string, producer version, and producer-identity
///     SHA-256;
///   * admitted-reference, factor-stream, mean-field-state-digest, and
///     allocation-contract versions;
///   * state, calculation, allocation, and schedule identity strings;
///   * auxiliary-basis digest version and content-identity SHA-256;
///   * physical periodic dimension as u32; contract version 1 requires 3;
///   * mesh and shift as three i32 values each;
///   * q index as u64, then centred doubled numerator and reciprocal wrap as
///     three i32 values each;
///   * reciprocal lattice B in row-major order as nine binary64 values;
///   * E_cut, p_max, radial boundary tolerance, and cell volume Omega as four
///     binary64 values;
///   * lower and upper enumeration bounds as three i64 values each, followed
///     by candidate, accepted, and exact-zero-exclusion counts as u64;
///   * for each accepted vector in production order: n_0,n_1,n_2 as signed
///     i64, followed by px, py, pz, p2, and w as binary64, where each p row is
///     accumulated from index 2 to 0 with `std::fma`,
///     `p2 = fma(px,px,fma(py,py,fma(pz,pz,0)))`, and
///     `w = (4*pi/Omega)/p2`.
/// `Omega` is evaluated canonically as
/// `(2*pi/sigma)^3 / abs(det_fma(B/sigma))`, where `sigma=max(abs(B))`,
/// every division is stored in binary64, and the determinant uses the same
/// fixed index-2-to-0 FMA accumulation.
///
/// The caller's safety cap, reciprocal panel block, resource-census identity,
/// and resource-plan identity are deliberately absent: none changes the exact
/// mathematical source.  The manifest payload charged to factor-build
/// admission is the 64-byte lowercase hexadecimal source identity itself.

#include <Eigen/Core>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/basis.hpp"
#include "vibeqc/periodic_correlation_admitted_reference.hpp"
#include "vibeqc/periodic_correlation_factor_build_admission.hpp"
#include "vibeqc/periodic_correlation_factor_stream.hpp"

namespace vibeqc {

class PeriodicCorrelationMetricFactorizationResult;

inline constexpr std::uint32_t
    kPeriodicCorrelationReciprocalMetricSourceContractVersion = 1U;
inline constexpr std::uint32_t
    kPeriodicCorrelationReciprocalMetricProducerVersion = 1U;
inline constexpr char kPeriodicCorrelationReciprocalMetricProducerId[] =
    "vibeqc.native.full-coulomb-all-reciprocal-metric";
inline constexpr PeriodicCorrelationByteCount
    kPeriodicCorrelationReciprocalMetricSourceManifestBytes = 64U;
inline constexpr std::uint64_t
    kPeriodicCorrelationReciprocalMetricSourceFixedPrefixWireBytes = 812U;
inline constexpr std::uint64_t
    kPeriodicCorrelationReciprocalMetricAcceptedVectorWireBytes = 64U;
inline constexpr std::uint32_t
    kPeriodicCorrelationReciprocalMetricResultContractVersion = 1U;
inline constexpr std::uint64_t
    kPeriodicCorrelationReciprocalMetricPayloadFixedPrefixWireBytes = 169U;

/// SHA-256 over the compiled producer ID and version, using the same canonical
/// integer/string encoding documented above.  A full-Coulomb factor-build
/// configuration must carry this exact value in producer_identity_sha256.
std::string
periodic_correlation_reciprocal_metric_producer_identity_sha256();

/// Exact canonical source-wire extent, excluding SHA-256 padding.  This O(1)
/// preflight rejects counts outside SHA-256's less-than-2^64-bit message
/// domain before a source-hash traversal begins.
std::uint64_t periodic_correlation_reciprocal_metric_source_wire_bytes(
    std::uint64_t accepted_vector_count);

/// Exact canonical numerical-payload extent, excluding SHA-256 padding.
/// The v1 payload has a 169-byte prefix followed by one row-major complex128
/// value for every element of the square auxiliary metric.
std::uint64_t periodic_correlation_reciprocal_metric_payload_wire_bytes(
    std::uint64_t n_auxiliary);

/// Constant-space outward certificate for the integer enumeration box.
/// Certificate diagnostics are deliberately absent from the canonical source
/// wire: the resulting bounds and candidate count are already sealed there.
struct PeriodicCorrelationReciprocalMetricEnumerationPlan {
    double normalization_scale = 0.0;
    /// Outward determinant interval for B / normalization_scale.
    double determinant_lower_bound = 0.0;
    double determinant_upper_bound = 0.0;
    double normalized_matrix_infinity_norm_upper_bound = 0.0;
    /// Row norms of (B / normalization_scale)^-1.
    std::array<double, 3> inverse_row_one_norm_upper_bounds = {0.0, 0.0, 0.0};
    std::array<double, 3> inverse_row_two_norm_upper_bounds = {0.0, 0.0, 0.0};
    double dot_relative_error_upper_bound = 0.0;
    double dot_absolute_error_upper_bound = 0.0;
    double accepted_vector_radius_upper_bound = 0.0;
    double matvec_feedback_upper_bound = 0.0;
    double matvec_error_upper_bound = 0.0;
    double shifted_vector_infinity_norm_upper_bound = 0.0;
    std::array<double, 3> stored_shift_component_upper_bounds = {0.0, 0.0, 0.0};
    std::array<double, 3> exact_shift_component_upper_bounds = {0.0, 0.0, 0.0};
    std::array<std::int64_t, 3> lower_bounds = {0, 0, 0};
    std::array<std::int64_t, 3> upper_bounds = {0, 0, 0};
    std::uint64_t candidate_count = 0U;
};

/// Certify, without enumeration or a reciprocal-vector list, a box containing
/// every integer label accepted by the fixed binary64 predicate for one stored
/// centred transfer. The squared membership limit is the exact finite
/// comparator used by the traversal. This O(1) seam is also the resource-
/// estimation and skew-cell regression surface.
PeriodicCorrelationReciprocalMetricEnumerationPlan
plan_periodic_correlation_reciprocal_metric_enumeration(
    const Eigen::Matrix3d& reciprocal_lattice,
    const Eigen::Vector3d& centered_transfer_fractional,
    double squared_membership_limit);

/// Immutable O(1)-resident description and identity of one exact q source.
class PeriodicCorrelationReciprocalMetricSourceManifest {
public:
    PeriodicCorrelationReciprocalMetricSourceManifest(
        const PeriodicCorrelationReciprocalMetricSourceManifest&) = delete;
    PeriodicCorrelationReciprocalMetricSourceManifest& operator=(
        const PeriodicCorrelationReciprocalMetricSourceManifest&) = delete;
    PeriodicCorrelationReciprocalMetricSourceManifest(
        PeriodicCorrelationReciprocalMetricSourceManifest&&) noexcept =
        default;
    PeriodicCorrelationReciprocalMetricSourceManifest& operator=(
        PeriodicCorrelationReciprocalMetricSourceManifest&&) noexcept =
        default;
    ~PeriodicCorrelationReciprocalMetricSourceManifest() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationReciprocalMetricSourceContractVersion;
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
    state_handle() const noexcept {
        return state_;
    }
    const std::array<char, 64>& source_identity_ascii() const noexcept {
        return source_identity_ascii_;
    }
    /// Diagnostic/binding convenience. The manifest itself retains no
    /// dynamically allocated identity string; this accessor returns a copy.
    std::string source_identity_sha256() const;
    int periodic_dimension() const noexcept { return periodic_dimension_; }
    const std::array<int, 3>& mesh() const noexcept { return mesh_; }
    const std::array<int, 3>& is_shift() const noexcept { return is_shift_; }
    std::uint64_t q_index() const noexcept { return q_index_; }
    const std::array<int, 3>& centered_doubled_numerator() const noexcept {
        return centered_doubled_numerator_;
    }
    const std::array<int, 3>& centered_reciprocal_wrap() const noexcept {
        return centered_reciprocal_wrap_;
    }
    const Eigen::Vector3d& q_fractional() const noexcept {
        return q_fractional_;
    }
    const Eigen::Vector3d& q_cartesian() const noexcept {
        return q_cartesian_;
    }
    const Eigen::Matrix3d& reciprocal_lattice() const noexcept {
        return reciprocal_lattice_;
    }
    double reciprocal_energy_cutoff() const noexcept {
        return reciprocal_energy_cutoff_;
    }
    double maximum_reciprocal_radius() const noexcept {
        return maximum_reciprocal_radius_;
    }
    double radial_boundary_tolerance() const noexcept {
        return radial_boundary_tolerance_;
    }
    double cell_volume_bohr3() const noexcept { return cell_volume_bohr3_; }
    const std::array<std::int64_t, 3>& lower_bounds() const noexcept {
        return lower_bounds_;
    }
    const std::array<std::int64_t, 3>& upper_bounds() const noexcept {
        return upper_bounds_;
    }
    std::uint64_t candidate_count() const noexcept {
        return candidate_count_;
    }
    std::uint64_t accepted_vector_count() const noexcept {
        return accepted_vector_count_;
    }
    std::uint64_t zero_mode_excluded_count() const noexcept {
        return zero_mode_excluded_count_;
    }

    /// Convert this exact source census into the count-only record consumed by
    /// factor-build admission.  The source payload is the 64 ASCII bytes of
    /// source_identity_sha256; full-Coulomb v1 has no tail or short-range work.
    PeriodicCorrelationFactorBuildQRecord factor_build_q_record() const;

private:
    PeriodicCorrelationReciprocalMetricSourceManifest(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::array<char, 64> source_identity_ascii,
        int periodic_dimension,
        std::array<int, 3> mesh,
        std::array<int, 3> is_shift,
        std::uint64_t q_index,
        std::array<int, 3> centered_doubled_numerator,
        std::array<int, 3> centered_reciprocal_wrap,
        Eigen::Vector3d q_fractional,
        Eigen::Vector3d q_cartesian,
        Eigen::Matrix3d reciprocal_lattice,
        double reciprocal_energy_cutoff,
        double maximum_reciprocal_radius,
        double radial_boundary_tolerance,
        double cell_volume_bohr3,
        std::array<std::int64_t, 3> lower_bounds,
        std::array<std::int64_t, 3> upper_bounds,
        std::uint64_t candidate_count,
        std::uint64_t accepted_vector_count,
        std::uint64_t zero_mode_excluded_count);

    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    /// The only retained source-manifest payload: exactly 64 ASCII bytes.
    std::array<char, 64> source_identity_ascii_{};
    int periodic_dimension_ = 0;
    std::array<int, 3> mesh_ = {0, 0, 0};
    std::array<int, 3> is_shift_ = {0, 0, 0};
    std::uint64_t q_index_ = 0U;
    std::array<int, 3> centered_doubled_numerator_ = {0, 0, 0};
    std::array<int, 3> centered_reciprocal_wrap_ = {0, 0, 0};
    Eigen::Vector3d q_fractional_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d q_cartesian_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d reciprocal_lattice_ = Eigen::Matrix3d::Zero();
    double reciprocal_energy_cutoff_ = 0.0;
    double maximum_reciprocal_radius_ = 0.0;
    double radial_boundary_tolerance_ = 0.0;
    double cell_volume_bohr3_ = 0.0;
    std::array<std::int64_t, 3> lower_bounds_ = {0, 0, 0};
    std::array<std::int64_t, 3> upper_bounds_ = {0, 0, 0};
    std::uint64_t candidate_count_ = 0U;
    std::uint64_t accepted_vector_count_ = 0U;
    std::uint64_t zero_mode_excluded_count_ = 0U;

    friend PeriodicCorrelationReciprocalMetricSourceManifest
    make_periodic_correlation_reciprocal_metric_source_manifest(
        const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationFactorStreamSchedule& schedule,
        const PeriodicCorrelationFactorBuildConfig& config,
        const BasisSet& auxiliary_basis,
        std::uint64_t q_index,
        std::uint64_t candidate_count_cap);
};

/// Build the exact one-q source identity without allocating the G-vector list.
/// candidate_count_cap is a mandatory caller policy bound on the guarded
/// Cartesian enumeration box, not part of the mathematical source identity.
PeriodicCorrelationReciprocalMetricSourceManifest
make_periodic_correlation_reciprocal_metric_source_manifest(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildConfig& config,
    const BasisSet& auxiliary_basis,
    std::uint64_t q_index,
    std::uint64_t candidate_count_cap);

/// Immutable raw A-by-A Hermitian Coulomb metric for one transfer q.
///
/// For the native auxiliary transform
///
///     F_P(p) = integral chi_P(r) exp(-i p.r) dr,
///
/// the full-Coulomb/all-reciprocal result is
///
///     M_PQ(q) = sum_G [4*pi/(Omega*|G+q|^2)]
///                         conj(F_P(G+q)) F_Q(G+q).
///
/// There is no k-point weight, spin factor, or factor one half.  The builder
/// first recomputes factor-build admission, then replays and verifies the
/// source SHA without allocating.  A second constant-source-space pass fills
/// admitted reciprocal and Fourier panels.  Scalar compensated accumulation
/// follows canonical source order and is invariant to reciprocal panel size.
/// Only the upper triangle is accumulated; final lower entries are exact
/// conjugates and diagonal imaginary lanes are +0.0.  Exact self-conjugate
/// transfers, including Gamma and negative-Nyquist combinations, first pass an
/// allocation-free p -> -p partner audit.  They are projected to the real
/// auxiliary-AO gauge only after both partners have been evaluated through one
/// canonical-momentum Fourier parity and the discarded imaginary residual
/// satisfies a scale-aware compensated-summation bound; the accepted residual
/// is retained as a diagnostic.  A positive-semidefinite zero metric is valid
/// here and is left for the later rank-revealing factorization stage to
/// classify.
///
/// The v1 payload identity hashes, in order: the length-prefixed domain
/// "vibeqc.periodic.correlation.reciprocal-metric.payload", result contract
/// version, length-prefixed 64-byte source identity, q index, auxiliary extent,
/// square element count, matrix bytes, then every row-major complex value as
/// real and imaginary binary64 lanes.  Integers are big-endian and either sign
/// of binary64 zero is encoded as +0.0.  Metric-census, metric-plan, backend,
/// and reciprocal-panel identities are execution provenance and deliberately
/// absent, so changing only reciprocal_block cannot change the numerical
/// payload identity.  The embedded source identity continues to bind its own
/// source-generation schedule, including upstream blocking choices.
class PeriodicCorrelationReciprocalMetricResult {
public:
    PeriodicCorrelationReciprocalMetricResult(
        const PeriodicCorrelationReciprocalMetricResult&) = delete;
    PeriodicCorrelationReciprocalMetricResult& operator=(
        const PeriodicCorrelationReciprocalMetricResult&) = delete;
    PeriodicCorrelationReciprocalMetricResult(
        PeriodicCorrelationReciprocalMetricResult&&) noexcept = default;
    PeriodicCorrelationReciprocalMetricResult& operator=(
        PeriodicCorrelationReciprocalMetricResult&&) noexcept = default;
    ~PeriodicCorrelationReciprocalMetricResult() = default;

    std::uint32_t contract_version() const noexcept {
        return kPeriodicCorrelationReciprocalMetricResultContractVersion;
    }
    const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
    state_handle() const noexcept {
        return state_;
    }
    std::string source_identity_sha256() const;
    std::string census_identity_sha256() const;
    std::string plan_identity_sha256() const;
    std::string payload_identity_sha256() const;
    std::string auxiliary_basis_identity_sha256() const;
    std::uint64_t q_index() const noexcept { return q_index_; }
    std::uint64_t conjugate_q_index() const noexcept {
        return conjugate_q_index_;
    }
    std::uint64_t n_auxiliary() const noexcept { return n_auxiliary_; }
    std::uint64_t accepted_vector_count() const noexcept {
        return accepted_vector_count_;
    }
    std::uint64_t reciprocal_panel_capacity() const noexcept {
        return reciprocal_panel_capacity_;
    }
    std::uint64_t completed_panel_count() const noexcept {
        return completed_panel_count_;
    }
    PeriodicCorrelationByteCount metric_bytes() const noexcept {
        return metric_bytes_;
    }
    PeriodicCorrelationByteCount reciprocal_panel_bytes() const noexcept {
        return reciprocal_panel_bytes_;
    }
    PeriodicCorrelationByteCount
    maximum_auxiliary_fourier_panel_bytes() const noexcept {
        return maximum_auxiliary_fourier_panel_bytes_;
    }
    PeriodicCorrelationByteCount
    maximum_weighted_fourier_panel_bytes() const noexcept {
        return maximum_weighted_fourier_panel_bytes_;
    }
    PeriodicCorrelationByteCount
    admitted_reciprocal_metric_peak_bytes() const noexcept {
        return admitted_reciprocal_metric_peak_bytes_;
    }
    bool self_conjugate_transfer() const noexcept {
        return self_conjugate_transfer_;
    }
    double maximum_self_conjugate_imaginary_residual() const noexcept {
        return maximum_self_conjugate_imaginary_residual_;
    }
    const std::vector<std::complex<double>>& matrix_row_major()
        const noexcept {
        return matrix_row_major_;
    }

private:
    PeriodicCorrelationReciprocalMetricResult(
        std::shared_ptr<const PeriodicRestrictedMeanFieldState> state,
        std::array<char, 64> source_identity_ascii,
        std::array<char, 64> census_identity_ascii,
        std::array<char, 64> plan_identity_ascii,
        std::array<char, 64> payload_identity_ascii,
        std::array<char, 64> auxiliary_basis_identity_ascii,
        std::uint64_t q_index,
        std::uint64_t conjugate_q_index,
        std::uint64_t n_auxiliary,
        std::uint64_t accepted_vector_count,
        std::uint64_t reciprocal_panel_capacity,
        std::uint64_t completed_panel_count,
        PeriodicCorrelationByteCount metric_bytes,
        PeriodicCorrelationByteCount reciprocal_panel_bytes,
        PeriodicCorrelationByteCount maximum_auxiliary_fourier_panel_bytes,
        PeriodicCorrelationByteCount maximum_weighted_fourier_panel_bytes,
        PeriodicCorrelationByteCount admitted_reciprocal_metric_peak_bytes,
        bool self_conjugate_transfer,
        double maximum_self_conjugate_imaginary_residual,
        std::vector<std::complex<double>> matrix_row_major);

    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::array<char, 64> source_identity_ascii_{};
    std::array<char, 64> census_identity_ascii_{};
    std::array<char, 64> plan_identity_ascii_{};
    std::array<char, 64> payload_identity_ascii_{};
    std::array<char, 64> auxiliary_basis_identity_ascii_{};
    std::uint64_t q_index_ = 0U;
    std::uint64_t conjugate_q_index_ = 0U;
    std::uint64_t n_auxiliary_ = 0U;
    std::uint64_t accepted_vector_count_ = 0U;
    std::uint64_t reciprocal_panel_capacity_ = 0U;
    std::uint64_t completed_panel_count_ = 0U;
    PeriodicCorrelationByteCount metric_bytes_ = 0U;
    PeriodicCorrelationByteCount reciprocal_panel_bytes_ = 0U;
    PeriodicCorrelationByteCount maximum_auxiliary_fourier_panel_bytes_ = 0U;
    PeriodicCorrelationByteCount maximum_weighted_fourier_panel_bytes_ = 0U;
    PeriodicCorrelationByteCount admitted_reciprocal_metric_peak_bytes_ = 0U;
    bool self_conjugate_transfer_ = false;
    double maximum_self_conjugate_imaginary_residual_ = 0.0;
    std::vector<std::complex<double>> matrix_row_major_;

    friend PeriodicCorrelationReciprocalMetricResult
    build_periodic_correlation_reciprocal_metric(
        const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationFactorStreamSchedule& schedule,
        const PeriodicCorrelationFactorBuildCensus& census,
        const PeriodicCorrelationReciprocalMetricSourceManifest& source,
        const BasisSet& auxiliary_basis);

    friend PeriodicCorrelationMetricFactorizationResult
    factorize_periodic_correlation_metric(
        const PeriodicCorrelationAdmittedReference&,
        const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationFactorBuildCensus&,
        PeriodicCorrelationReciprocalMetricResult&&, double);
};

/// Build one admitted raw reciprocal metric.  The factor-build plan is
/// recomputed from the immutable census and must be admitted before any
/// size-dependent allocation.  The source is verified against its exact census
/// record and canonical SHA before matrix/panel allocation.
PeriodicCorrelationReciprocalMetricResult
build_periodic_correlation_reciprocal_metric(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    const BasisSet& auxiliary_basis);

/// Allocation-free replay of an immutable source for native panel consumers.
/// The callback receives px,py,pz,p2,w in the source's canonical order. Its
/// lane reference is transient and must not be retained. No G list or callback
/// wrapper is allocated. A null callback or moved-from source is rejected;
/// callback exceptions propagate. The final count must equal the sealed count.
/// This visits the source only: a numerical consumer must separately validate
/// its live reference/schedule/census and source identity before allocating.
std::uint64_t visit_periodic_correlation_reciprocal_metric_source(
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    void (*callback)(const std::array<double, 5>&, void*), void* context);

/// Allocation-free inversion audit for two immutable q/-q source manifests.
/// Both must share the identical state owner, cutoff, volume and lattice.
/// Equal cardinality plus an injective exact integer-label inversion proves
/// two-way closure; each partner must satisfy the OTHER source's predicate
/// and have bitwise negative momentum and identical p2/Coulomb weight.
/// The positive per-source candidate cap is checked before enumeration.
/// Returns the common accepted count; does not change either source or its
/// version-1 radial slack. A q-dependent cutoff boundary can fail this gate.
/// This certifies reciprocal source closure ONLY, not AO-image covariance,
/// auxiliary-rank/projector covariance or an infinite-image error bound.
std::uint64_t require_periodic_correlation_reciprocal_source_conjugacy(
    const PeriodicCorrelationReciprocalMetricSourceManifest& source,
    const PeriodicCorrelationReciprocalMetricSourceManifest& conjugate_source,
    std::uint64_t maximum_candidates_per_source);

}  // namespace vibeqc
