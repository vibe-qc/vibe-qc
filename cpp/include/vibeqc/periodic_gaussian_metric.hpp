#pragma once

// One-q metric/whitener before SCF, from an authenticated finite source.
// Shares the native v1 compensated Fourier contraction and principal root.
// No mean-field owner, schedule, caller-labelled matrix, qbar-projector
// covariance, infinite-image/tail accuracy, nuclear or SCF claim.

#include "vibeqc/periodic_gaussian_reciprocal_source.hpp"
#include "vibeqc/periodic_correlation_metric_factorization.hpp"

namespace vibeqc {

struct PeriodicGaussianMetricConfig {
    std::uint64_t reciprocal_block = 0;
    std::uint64_t whitener_column_block = 0;
    PeriodicGaussianSourceCaps basis_verification_caps;
};
struct PeriodicGaussianMetricLiveInventory {
    std::uint64_t replicas_per_node = 0; // positive
    std::uint64_t other_retained_bytes_per_replica = 0;
    std::uint64_t other_transient_bytes_per_replica = 0;
    // Explicit positive caller allowance, not a certified stack/RSS bound.
    std::uint64_t fixed_backend_margin_bytes_per_replica = 0;
    std::uint64_t external_node_bytes = 0;
};
struct PeriodicGaussianMetricCaps {
    std::uint64_t maximum_owned_numeric_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0;
    std::uint64_t maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_candidate_evaluations = 0;
    std::uint64_t maximum_work_units = 0;
};
enum class PeriodicGaussianMetricPhase : std::uint32_t { RawMetric = 0, PrincipalWhitening = 1 };

struct PeriodicGaussianMetricPlan {
    PeriodicGaussianMetricPhase phase = PeriodicGaussianMetricPhase::RawMetric;
    std::uint64_t n_auxiliary = 0, accepted_vector_count = 0, candidate_count = 0;
    std::uint64_t reciprocal_panel_capacity = 0, auxiliary_validation_passes = 0;
    std::uint64_t matrix_bytes = 0, reciprocal_panel_bytes = 0, double_fourier_panel_bytes = 0;
    std::uint64_t metric_owned_numeric_peak_bytes = 0, factorization_owned_numeric_peak_bytes = 0;
    std::uint64_t owned_numeric_peak_bytes = 0, returned_numeric_bytes = 0;
    std::uint64_t borrowed_basis_active_numeric_bytes = 0, fixed_inventoried_object_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, node_inventoried_bytes = 0;
    std::uint64_t metric_product_count = 0, fourier_radial_term_count = 0;
    std::uint64_t harmonic_term_count_upper_bound = 0, jacobi_rotation_count_upper_bound = 0;
    std::uint64_t whitener_term_count_upper_bound = 0, candidate_evaluations = 0;
    std::uint64_t work_units_upper_bound = 0;
    PeriodicGaussianMetricConfig config;
    PeriodicGaussianMetricLiveInventory live;
    PeriodicGaussianMetricCaps caps;
    std::array<char, 64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(), identity_ascii.end()}; }
};

class PeriodicGaussianMetricWhitener;
class PeriodicGaussianReciprocalMetric {
public:
    PeriodicGaussianReciprocalMetric(const PeriodicGaussianReciprocalMetric&) = delete;
    PeriodicGaussianReciprocalMetric& operator=(const PeriodicGaussianReciprocalMetric&) = delete;
    PeriodicGaussianReciprocalMetric(PeriodicGaussianReciprocalMetric&&) noexcept = default;
    PeriodicGaussianReciprocalMetric& operator=(PeriodicGaussianReciprocalMetric&&) = delete;
    bool consumed() const noexcept { return !context_; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const PeriodicGaussianMetricPlan& plan() const noexcept { return plan_; }
    std::uint64_t q_index() const noexcept { return q_; }
    std::uint64_t conjugate_q_index() const noexcept { return qbar_; }
    bool self_conjugate_transfer() const noexcept { return q_ == qbar_; }
    std::uint64_t completed_panel_count() const noexcept { return completed_panels_; }
    double maximum_self_conjugate_imaginary_residual() const noexcept { return imaginary_residual_; }
    const std::vector<std::complex<double>>& matrix_row_major() const noexcept { return matrix_; }
    std::string source_identity_sha256() const;
    std::string conjugate_source_identity_sha256() const;
    std::string payload_identity_sha256() const;
private:
    PeriodicGaussianReciprocalMetric() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianMetricPlan plan_;
    std::uint64_t q_ = 0, qbar_ = 0, completed_panels_ = 0;
    double imaginary_residual_ = 0.0;
    std::array<char, 64> source_{}, opposite_{}, payload_{};
    std::vector<std::complex<double>> matrix_;
    friend PeriodicGaussianReciprocalMetric build_periodic_gaussian_reciprocal_metric(
        const PeriodicGaussianReciprocalSource&, const BasisSet&, const BasisSet&,
        const PeriodicGaussianMetricConfig&, const PeriodicGaussianMetricLiveInventory&,
        const PeriodicGaussianMetricCaps&);
    friend PeriodicGaussianMetricWhitener factorize_periodic_gaussian_metric(
        PeriodicGaussianReciprocalMetric&&, std::uint64_t,
        const PeriodicGaussianMetricLiveInventory&, const PeriodicGaussianMetricCaps&);
};

class PeriodicGaussianMetricWhitener {
public:
    PeriodicGaussianMetricWhitener(const PeriodicGaussianMetricWhitener&) = delete;
    PeriodicGaussianMetricWhitener& operator=(const PeriodicGaussianMetricWhitener&) = delete;
    PeriodicGaussianMetricWhitener(PeriodicGaussianMetricWhitener&&) noexcept = default;
    PeriodicGaussianMetricWhitener& operator=(PeriodicGaussianMetricWhitener&&) = delete;
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const PeriodicGaussianMetricPlan& plan() const noexcept { return plan_; }
    std::uint64_t q_index() const noexcept { return q_; }
    std::uint64_t conjugate_q_index() const noexcept { return qbar_; }
    bool self_conjugate_transfer() const noexcept { return q_ == qbar_; }
    const PeriodicCorrelationMetricFactorizationDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const std::vector<std::complex<double>>& matrix_row_major() const noexcept { return matrix_; }
    std::string source_identity_sha256() const;
    std::string conjugate_source_identity_sha256() const;
    std::string input_payload_identity_sha256() const;
    std::string payload_identity_sha256() const;
private:
    PeriodicGaussianMetricWhitener() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianMetricPlan plan_;
    std::uint64_t q_ = 0, qbar_ = 0;
    std::array<char, 64> source_{}, opposite_{}, input_{}, payload_{};
    PeriodicCorrelationMetricFactorizationDiagnostics diagnostics_;
    std::vector<std::complex<double>> matrix_;
    friend PeriodicGaussianMetricWhitener factorize_periodic_gaussian_metric(
        PeriodicGaussianReciprocalMetric&&, std::uint64_t,
        const PeriodicGaussianMetricLiveInventory&, const PeriodicGaussianMetricCaps&);
};

// Exact variable numerical peaks: raw 16*A²+40*g+32*A*g; factorization and
// W formation 32*A²+8*A. M is freed before W is allocated. Both return16*A².
// Fixed object inventories are not allocator/stack/RSS certification. Basis
// active payload is charged per role (no implicit same-pointer deduction);
// caller declares remaining owner/capacity/control/live bytes and margin.
// Basis active payload remains conservatively reserved during whitening even
// if a caller releases those objects early. Do not add that reserved active
// component again in other_retained_bytes; add only excess/other live storage.
// Work is an abstract conservative gate, not timing or FLOP certification.
// Supplied basis caps must cover the stored context census; actual rechecks
// are then capped to that census, so changed larger inputs fail before hash
// scans can exceed the plan's active-payload/work reservation.
// For raw: W_context +128*(ceil(N/g)+1)*(aux_lanes+S+C)
// +128*N*P_aux +512*(28*N*A) +64*N*A*(A+1)/2 +64*N*A
// +2048*source_candidates +128*A² +4096. For principal whitening:
// 32768*A³+1024*A²+4096 covers the compiled <=64 cyclic sweeps, sorting,
// orthogonality, W formation, finite checks and payload scans.
// No public plan object is trusted as numerical admission: builders re-plan.
PeriodicGaussianMetricPlan plan_periodic_gaussian_metric(
    const PeriodicGaussianReciprocalSource&, const PeriodicGaussianMetricConfig&,
    const PeriodicGaussianMetricLiveInventory&, const PeriodicGaussianMetricCaps&,
    PeriodicGaussianMetricPhase phase);
PeriodicGaussianReciprocalMetric build_periodic_gaussian_reciprocal_metric(
    const PeriodicGaussianReciprocalSource&, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicGaussianMetricConfig&, const PeriodicGaussianMetricLiveInventory&,
    const PeriodicGaussianMetricCaps&);
// Current live inventory is re-admitted before inspecting/consuming payload.
// No fresh basis is supplied or evaluated. Raw's owner is invalidated before
// eigensolver mutation; exceptions thereafter leave raw consumed. Thresholds
// are exclusively the immutable context's explicit scientific controls.
PeriodicGaussianMetricWhitener factorize_periodic_gaussian_metric(
    PeriodicGaussianReciprocalMetric&& raw, std::uint64_t whitener_column_block,
    const PeriodicGaussianMetricLiveInventory&, const PeriodicGaussianMetricCaps&);

} // namespace vibeqc
