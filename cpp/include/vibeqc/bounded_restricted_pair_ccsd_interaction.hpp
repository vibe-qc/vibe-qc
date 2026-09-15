#pragma once

// One REAL mixed-PNO semi-joint contribution, Riplinger and Neese,
// JCP 138, 034106 (2013), doi:10.1063/1.4773581, II.C Eq.(26):
// R_ab = sum_cd (K_ac - J_ca/2) (2 T_cd - T_dc) O_db.
// K[A,B]=(i a_A|k c_B), J[B,A]=(ik|c_B a_A), O[B,A]=<d_B|b_A>.
// This is an isolated numerical contribution, NOT a complete residual,
// physical-source/orthogonality certificate, energy, or DLPNO solver.
// Both local frames are retained; never pre-project B into A. No weights.

#include <string>
#include <vector>
#include "vibeqc/bounded_restricted_ccsd_target.hpp"

namespace vibeqc {

struct BoundedRestrictedPairCCSDInteractionInput {
    std::uint64_t target_dimension = 0, source_dimension = 0;
    BoundedRestrictedCCSDRealView k_ab, j_ba, overlap_ba, t_bb;
    // Apply Eq.(26) to stored T^T instead of T; K/J/O are NOT transposed.
    bool source_transposed = false;
};
struct BoundedRestrictedPairCCSDInteractionInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    // Borrowed K/J/O/T are counted once PER ROLE even if their views alias.
    // Declare other simultaneous owner storage here, excluding those roles.
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    // Explicit positive allowance; logical inventory is not exact OS RSS.
    std::uint64_t backend_margin_bytes_per_replica = 0;
};
struct BoundedRestrictedPairCCSDInteractionCaps {
    // Dimension/borrowed/owned/scalar-product caps may be zero when the
    // corresponding exact count is zero. All remaining caps are positive.
    std::uint64_t maximum_target_dimension = 0, maximum_source_dimension = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0, maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0, maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_scalar_products = 0, maximum_work_units = 0;
};
struct BoundedRestrictedPairCCSDInteractionMemoryPlan {
    std::uint64_t target_dimension = 0, source_dimension = 0;
    std::uint64_t borrowed_input_elements = 0, borrowed_numerical_bytes = 0;
    std::uint64_t output_bytes = 0, intermediate_bytes = 0, peak_owned_numerical_bytes = 0;
    std::uint64_t fixed_control_storage_bytes = 0, total_control_storage_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t backend_margin_bytes_per_replica = 0;
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    // Exact arithmetic multiplication count, including the 2 and 1/2
    // coefficients: 2*A*B*B + 2*A*A*B. No zero-value shortcuts.
    std::uint64_t intermediate_contraction_terms = 0, residual_contraction_terms = 0;
    std::uint64_t scalar_products = 0;
    // Two complete input finite/hash scans and one output payload hash.
    std::uint64_t input_scan_elements = 0, output_scan_elements = 0;
    std::uint64_t work_units_upper_bound = 0;
};
struct BoundedRestrictedPairCCSDInteractionDiagnostics {
    std::uint64_t scalar_products = 0, product_underflow_count = 0;
    std::uint64_t intermediate_contraction_terms = 0, residual_contraction_terms = 0;
    std::uint64_t charged_work_units = 0;
    double maximum_absolute_intermediate = 0.0, maximum_absolute_residual = 0.0;
};

class BoundedRestrictedPairCCSDInteractionResult {
public:
    BoundedRestrictedPairCCSDInteractionResult(const BoundedRestrictedPairCCSDInteractionResult&) = delete;
    BoundedRestrictedPairCCSDInteractionResult& operator=(const BoundedRestrictedPairCCSDInteractionResult&) = delete;
    BoundedRestrictedPairCCSDInteractionResult(BoundedRestrictedPairCCSDInteractionResult&&) noexcept;
    BoundedRestrictedPairCCSDInteractionResult& operator=(BoundedRestrictedPairCCSDInteractionResult&&) = delete;
    const BoundedRestrictedPairCCSDInteractionMemoryPlan& memory() const;
    const BoundedRestrictedPairCCSDInteractionDiagnostics& diagnostics() const;
    bool source_transposed() const;
    std::uint64_t target_dimension() const;
    std::uint64_t source_dimension() const;
    double residual(std::size_t a, std::size_t b) const;
    const double* residual_data() const;
    const std::string& input_payload_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    const std::string& identity_sha256() const;
    bool physical_source_certified() const noexcept { return false; }
private:
    BoundedRestrictedPairCCSDInteractionResult() = default;
    void require_live() const;
    bool live_ = false, source_transposed_ = false;
    BoundedRestrictedPairCCSDInteractionMemoryPlan memory_;
    BoundedRestrictedPairCCSDInteractionDiagnostics diagnostics_;
    std::vector<double> residual_;
    std::string input_, payload_, identity_;
    friend BoundedRestrictedPairCCSDInteractionResult bounded_restricted_pair_ccsd_interaction(
        const BoundedRestrictedPairCCSDInteractionInput&,
        const BoundedRestrictedPairCCSDInteractionInventory&,
        const BoundedRestrictedPairCCSDInteractionCaps&);
};

// Checked count-only plan, no floating reads or numerical allocation.
// Exact active numerical payload: owned8*(A*A+B*A), borrowed8*(3*A*B+B*B).
// Stages U[B,A]=(2T-T^T)O, then R[A,A]=(K-J^T/2)U. Each dot uses only
// fixed-size compensated scalar sums; there is no B*B copy or BLAS workspace.
BoundedRestrictedPairCCSDInteractionMemoryPlan plan_bounded_restricted_pair_ccsd_interaction(
    std::uint64_t target_dimension, std::uint64_t source_dimension,
    const BoundedRestrictedPairCCSDInteractionInventory&);

// Exact aligned row-major view extents are mandatory. Empty views have
// count0 (pointer ignored); all nonempty inputs are finite/hash-scanned even
// if A==0. B==0 yields a zero A*A output. Controls are copied first, complete
// memory/work/caps precede input scans and every size-dependent allocation.
// Inputs must stay immutable/alive for the WHOLE call; before/after hashes
// are a replay check, not permission for concurrent writes. The result owns
// its only retained numeric payload. Exceptions publish no partial result.
// Requires strict binary64/nearest/gradual underflow. Underflows are counted,
// not certified as an error bound; no symmetrization or clipping is performed.
// Digest: version1, big-endian length-prefixed strings/u64/binary64 (signed
// zero normalized). Input domain "vibeqc.bounded.pair-ccsd-interaction.input"
// binds A,B,source_transposed then K,J,O,T in declaration order. Payload domain
// ".payload" binds A,B and row-major R; identity domain ".identity" binds
// input/payload digests and the fixed numerical policy, not physical labels.
BoundedRestrictedPairCCSDInteractionResult bounded_restricted_pair_ccsd_interaction(
    const BoundedRestrictedPairCCSDInteractionInput&,
    const BoundedRestrictedPairCCSDInteractionInventory&,
    const BoundedRestrictedPairCCSDInteractionCaps&);

} // namespace vibeqc
