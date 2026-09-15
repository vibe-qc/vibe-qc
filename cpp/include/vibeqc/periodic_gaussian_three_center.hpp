#pragma once

// Density-independent physical finite-source factors for the matched native
// HF consumer. Sun (2017), doi:10.1063/1.4998644, Eqs.13,16,21:
// T_P,mu nu=sum_p w_p conj(F_P(p))*rho_mu nu(p;k_ket), B=W*T.
// No extra 1/Nk, volume or spin factor. The row index remains original aux AO.
// These tiles do not certify image tails, qbar rank/projector covariance,
// nuclear/Hcore compatibility, converged HF, or production DLPNO accuracy.
#include "vibeqc/periodic_gaussian_metric.hpp"

namespace vibeqc {

inline constexpr char kPeriodicGaussianThreeCenterLatticePolicy[] =
    "direct-lattice=authenticated-original-context-direct-columns;binary64-v1";
inline constexpr char kPeriodicGaussianThreeCenterImagePolicy[] =
    "finite-pair-separation-cutoff;padded-long-double-box;lexicographic;binary64-v1";

struct PeriodicGaussianThreeCenterSelection {
    std::uint64_t k_bra_index = 0;
    std::uint64_t ao_pair_begin = 0, ao_pair_count = 0;
    std::uint64_t auxiliary_begin = 0, auxiliary_count = 0;
};
struct PeriodicGaussianThreeCenterDescriptor {
    std::uint64_t q_index = 0, conjugate_q_index = 0;
    std::uint64_t k_bra_index = 0, k_ket_index = 0;
    std::array<int, 3> k_ket_reciprocal_wrap{};
    std::uint64_t ao_pair_begin = 0, ao_pair_count = 0;
    std::uint64_t auxiliary_begin = 0, auxiliary_count = 0;
    std::uint64_t element_count = 0;
};
struct PeriodicGaussianThreeCenterConfig {
    std::uint64_t reciprocal_block = 0;
    PeriodicGaussianSourceCaps basis_verification_caps;
};
struct PeriodicGaussianThreeCenterCaps {
    // Positive, explicit. Image cap is the sum of selected pair box sizes,
    // not multiplied by N. The total traversals are admitted separately.
    std::uint64_t maximum_image_candidates = 0;
    PeriodicGaussianMetricCaps resources;
};
struct PeriodicGaussianThreeCenterPlan {
    PeriodicGaussianThreeCenterDescriptor descriptor;
    std::uint64_t n_basis = 0, n_auxiliary = 0;
    std::uint64_t accepted_vector_count = 0, reciprocal_candidate_count = 0;
    std::uint64_t reciprocal_capacity = 0, reciprocal_panel_count = 0;
    std::uint64_t resident_whitener_bytes = 0, raw_panel_bytes = 0, compensation_bytes = 0;
    std::uint64_t reciprocal_panel_bytes = 0, double_auxiliary_fourier_bytes = 0, ao_fourier_bytes = 0;
    std::uint64_t fixed_fourier_numeric_workspace_bytes = 0, output_bytes = 0;
    std::uint64_t assembly_owned_numeric_peak_bytes = 0, whitening_owned_numeric_peak_bytes = 0;
    std::uint64_t owned_numeric_peak_bytes = 0, borrowed_basis_active_numeric_bytes = 0;
    std::uint64_t fixed_inventoried_object_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, node_inventoried_bytes = 0;
    std::uint64_t image_candidate_count = 0, retained_pair_image_count = 0;
    std::uint64_t reciprocal_candidate_evaluations = 0, image_candidate_evaluations = 0;
    std::uint64_t primitive_pair_evaluations_upper_bound = 0;
    std::uint64_t raw_contraction_term_count = 0, whitening_term_count = 0;
    std::uint64_t preflight_work_units_upper_bound = 0, work_units_upper_bound = 0;
    PeriodicGaussianThreeCenterConfig config;
    PeriodicGaussianMetricLiveInventory live;
    PeriodicGaussianThreeCenterCaps caps;
    std::array<char, 64> identity_ascii{};
    std::string plan_identity_sha256() const { return {identity_ascii.begin(), identity_ascii.end()}; }
};

class PeriodicGaussianThreeCenterTile {
public:
    PeriodicGaussianThreeCenterTile(const PeriodicGaussianThreeCenterTile&) = delete;
    PeriodicGaussianThreeCenterTile& operator=(const PeriodicGaussianThreeCenterTile&) = delete;
    PeriodicGaussianThreeCenterTile(PeriodicGaussianThreeCenterTile&&) noexcept = default;
    PeriodicGaussianThreeCenterTile& operator=(PeriodicGaussianThreeCenterTile&&) = delete;
    bool finite_image_reference() const noexcept { return true; }
    bool ao_image_source_certified() const noexcept { return false; }
    const std::shared_ptr<const PeriodicGaussianSourceContext>& context_handle() const noexcept { return context_; }
    const PeriodicGaussianThreeCenterPlan& plan() const noexcept { return plan_; }
    const PeriodicGaussianThreeCenterDescriptor& descriptor() const noexcept { return plan_.descriptor; }
    const std::vector<std::complex<double>>& matrix_row_major() const noexcept { return values_; }
    std::string source_identity_sha256() const;
    std::string conjugate_source_identity_sha256() const;
    std::string whitener_payload_identity_sha256() const;
    std::string payload_identity_sha256() const;
private:
    PeriodicGaussianThreeCenterTile() = default;
    std::shared_ptr<const PeriodicGaussianSourceContext> context_;
    PeriodicGaussianThreeCenterPlan plan_;
    std::array<char, 64> source_{}, opposite_{}, whitener_{}, payload_{};
    std::vector<std::complex<double>> values_;
    friend PeriodicGaussianThreeCenterTile build_periodic_gaussian_three_center_tile(
        const PeriodicGaussianReciprocalSource&, const PeriodicGaussianMetricWhitener&,
        const BasisSet&, const BasisSet&, const PeriodicGaussianThreeCenterSelection&,
        const PeriodicGaussianThreeCenterConfig&, const PeriodicGaussianMetricLiveInventory&,
        const PeriodicGaussianThreeCenterCaps&);
};

// Both APIs perform allocation-free physical preflight, not just shape math:
// admit known memory/SHA extents and conservative census work first; verify
// actual AO+aux content, immutable context/q/source/W, replay source SHA and
// census AO images; then admit full work before any numerical panel exists.
// Basis rechecks clamp caller caps to the authenticated combined census.
// The returned plain plan is diagnostic only, never accepted as admission.
// With A aux, Pb selected pairs, Ab output rows, g=min(block,N): resident W
// 16A² is counted ONCE, separately from owned phase peaks. Assembly owns
// 32A*Pb+40g+32Ag+16Pb*g+F, with F queried from the Fourier evaluator;
// whitening owns16A*Pb+16Ab*Pb.
// Fourier/reciprocal/compensation storage is dead before output allocation.
// Only16Ab*Pb remains in the result; each aux slice recomputes raw T.
// Active borrowed bases are reserved once per role; caller declares excess
// capacity/control and all other retained/transient owners. Fixed object
// inventory + explicit positive backend margin is NOT a stack/RSS bound.
//
// Source candidate traversals=2C. For K=ceil(N/g), actual image candidate
// traversals=I*(1+K+N). Primitive pair eval upper=N*R*Pexp_AO². Full scans
// include context verification, K+1 aux and 2(K+1) AO basis scans. Abstract
// work counts are conservative gates, not timing/FLOP predictions. Census
// is first charged using Icap, then actual I/R enter the final work bound.
// The L<=6 primitive contribution is charged 2^20 abstract units per bounded
// primitive-pair evaluation; all basis scans, AO lookups, image traversals,
// reciprocal SHA replays and raw/whitening contractions are also charged.
//
// Numerical payload v1: length-prefixed domain
// "vibeqc.periodic.gaussian-three-center.payload", u32 version; context,
// source, opposite-source and W-payload SHA strings; image/lattice policy
// strings; cutoff binary64; original direct A row-major; bra then ket full
// records (u64 index, i32[3] doubled address, binary64[3] fractional and
// Cartesian); descriptor declaration order (wrap i32[3], other fields u64);
// image/retained/reciprocal/output-byte counts u64; complex values row-major.
// Big-endian numbers and u64 string lengths, signed zero normalized to +0.
// Context SHA binds actual AO/aux contents and q-source/threshold policies.
// Plan v1 has a distinct domain and seals context/source/opposite/W SHA,
// descriptor, all plan counts in declaration order, then config/live/caps.
// Reciprocal block and caps affect plan, not the numerical payload identity.
PeriodicGaussianThreeCenterPlan plan_periodic_gaussian_three_center_tile(
    const PeriodicGaussianReciprocalSource&, const PeriodicGaussianMetricWhitener&,
    const BasisSet& ao, const BasisSet& auxiliary, const PeriodicGaussianThreeCenterSelection&,
    const PeriodicGaussianThreeCenterConfig&, const PeriodicGaussianMetricLiveInventory&,
    const PeriodicGaussianThreeCenterCaps&);
PeriodicGaussianThreeCenterTile build_periodic_gaussian_three_center_tile(
    const PeriodicGaussianReciprocalSource&, const PeriodicGaussianMetricWhitener&,
    const BasisSet& ao, const BasisSet& auxiliary, const PeriodicGaussianThreeCenterSelection&,
    const PeriodicGaussianThreeCenterConfig&, const PeriodicGaussianMetricLiveInventory&,
    const PeriodicGaussianThreeCenterCaps&);

} // namespace vibeqc
