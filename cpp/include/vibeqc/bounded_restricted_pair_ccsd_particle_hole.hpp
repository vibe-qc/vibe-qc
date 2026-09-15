#pragma once

// Complete BARE particle-hole T2 contribution in real restricted CCSD.
// Stanton et al., JCP94,4334(1991), doi:10.1063/1.460620; restricted
// form: Riplinger/Neese JCP138,034106(2013), doi:10.1063/1.4773581 Eq.(7).
// For source ordered pair (l,m) in B, target A, and other occupied r:
// K[B,A]=(m b_B|r a_A), J[B,A]=(m r|b_B a_A), O[B,A]=<b_B|a_A>.
// F=O^T(2T-T^T)K-O^T T J-(O^T T^T J)^T.
// R_ij=sum_m[F(T_im,K_mj,J_mj,O_im)+F(T_jm,K_mi,J_mi,O_jm)^T].
// This REPLACES the amplitude-independent W1/W2/WX seeds in BOTH
// particle-hole passes of bounded_restricted_ccsd_target.cpp, not its
// amplitude-dependent dressing, F terms, ladders or explicit T1 products.
// Never append it to an unchanged full CCSD residual. This is not an entire
// CCSD residual, an energy, a physical-source certificate, or a solver.

#include <string>
#include <vector>
#include "vibeqc/bounded_restricted_ccsd_target.hpp"

namespace vibeqc {

struct BoundedRestrictedPairCCSDParticleHoleSource {
    // Strict sequence (leg=0,m=0..o-1), then (leg=1,m=0..o-1).
    // Leg0 represents ordered T_im, leg1 ordered T_jm. No omitted slots,
    // including m=i/j, equal targets, and source_dimension=0.
    std::uint64_t leg = 0, occupied_index = 0, source_dimension = 0;
    BoundedRestrictedCCSDRealView k_ba, j_ba, overlap_ba, t_bb;
    // Only the stored T is transposed, never K/J/O or the output leg.
    bool source_transposed = false;
};
struct BoundedRestrictedPairCCSDParticleHoleInventory {
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    // One simultaneous K/J/O/T source is counted PER ROLE (even if aliased).
    // Other live source owners/storage, including other cached sources,
    // belong here. The accumulator retains NO source pointer or table.
    std::uint64_t other_live_numerical_bytes_per_replica = 0;
    std::uint64_t other_live_control_bytes_per_replica = 0;
    std::uint64_t backend_margin_bytes_per_replica = 0;
};
struct BoundedRestrictedPairCCSDParticleHoleCaps {
    std::uint64_t maximum_target_dimension = 0, maximum_source_dimension = 0;
    std::uint64_t maximum_occupied_count = 0;
    std::uint64_t maximum_borrowed_numerical_bytes = 0, maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_per_replica_inventoried_bytes = 0, maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_scalar_products = 0, maximum_work_units = 0;
};
struct BoundedRestrictedPairCCSDParticleHoleSourcePlan {
    std::uint64_t target_dimension = 0, source_dimension = 0;
    std::uint64_t borrowed_input_elements = 0, borrowed_numerical_bytes = 0;
    std::uint64_t intermediate_contraction_terms = 0, projection_contraction_terms = 0;
    // Exact multiplications including 2*T: 4*A*B^2+2*A^2*B.
    std::uint64_t scalar_products = 0, input_scan_elements = 0;
    std::uint64_t work_units_upper_bound = 0;
};
struct BoundedRestrictedPairCCSDParticleHoleMemoryPlan {
    std::uint64_t target_dimension = 0, maximum_source_dimension = 0, occupied_count = 0;
    std::uint64_t source_slots = 0, maximum_borrowed_numerical_bytes = 0;
    std::uint64_t output_bytes = 0, compensation_bytes = 0, intermediate_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t fixed_control_storage_bytes = 0, total_control_storage_bytes = 0;
    std::uint64_t other_live_numerical_bytes_per_replica = 0, backend_margin_bytes_per_replica = 0;
    std::uint64_t numerical_replicas = 0, external_node_bytes = 0;
    std::uint64_t per_replica_inventoried_bytes = 0, required_node_inventoried_bytes = 0;
    std::uint64_t scalar_products_upper_bound = 0, input_scan_elements_upper_bound = 0;
    std::uint64_t finish_work_units = 0, work_units_upper_bound = 0;
};
struct BoundedRestrictedPairCCSDParticleHoleDiagnostics {
    std::uint64_t visited_sources = 0, zero_rank_sources = 0, scalar_products = 0;
    std::uint64_t input_scan_elements = 0, product_underflow_count = 0, charged_work_units = 0;
    double maximum_absolute_intermediate = 0.0, maximum_absolute_residual = 0.0;
};

class BoundedRestrictedPairCCSDParticleHoleAccumulator;
class BoundedRestrictedPairCCSDParticleHoleResult {
public:
    BoundedRestrictedPairCCSDParticleHoleResult(const BoundedRestrictedPairCCSDParticleHoleResult&) = delete;
    BoundedRestrictedPairCCSDParticleHoleResult& operator=(const BoundedRestrictedPairCCSDParticleHoleResult&) = delete;
    BoundedRestrictedPairCCSDParticleHoleResult(BoundedRestrictedPairCCSDParticleHoleResult&&) noexcept;
    BoundedRestrictedPairCCSDParticleHoleResult& operator=(BoundedRestrictedPairCCSDParticleHoleResult&&) = delete;
    const BoundedRestrictedPairCCSDParticleHoleMemoryPlan& memory() const;
    const BoundedRestrictedPairCCSDParticleHoleDiagnostics& diagnostics() const;
    std::uint64_t target_i() const;
    std::uint64_t target_j() const;
    double residual(std::size_t a, std::size_t b) const;
    const double* residual_data() const;
    const std::string& input_stream_identity_sha256() const;
    const std::string& payload_identity_sha256() const;
    const std::string& identity_sha256() const;
    bool physical_source_certified() const noexcept { return false; }
    bool entire_ccsd_residual() const noexcept { return false; }
private:
    BoundedRestrictedPairCCSDParticleHoleResult() = default;
    void require_live() const;
    bool live_ = false;
    std::uint64_t i_ = 0, j_ = 0;
    BoundedRestrictedPairCCSDParticleHoleMemoryPlan memory_;
    BoundedRestrictedPairCCSDParticleHoleDiagnostics diagnostics_;
    std::vector<double> residual_;
    std::string input_, payload_, identity_;
    friend class BoundedRestrictedPairCCSDParticleHoleAccumulator;
};

// Both are checked count-only plans, without floating reads/heap allocation.
BoundedRestrictedPairCCSDParticleHoleSourcePlan plan_bounded_restricted_pair_ccsd_particle_hole_source(
    std::uint64_t target_dimension, std::uint64_t source_dimension);
BoundedRestrictedPairCCSDParticleHoleMemoryPlan plan_bounded_restricted_pair_ccsd_particle_hole(
    std::uint64_t target_dimension, std::uint64_t occupied_count,
    std::uint64_t maximum_source_dimension, const BoundedRestrictedPairCCSDParticleHoleInventory&);

// Owned peak=16*A^2+8*A*M; one source borrowed=8*(B^2+3*A*B).
// Stage U=(2T-T^T)K-TJ then O^T U; overwrite U with T^T J and
// subtract (O^T U)^T. No common virtual dimension, dense common T or
// source-rank-squared copy. Final residual retains only 8*A^2 bytes.
// Inventory/caps are copied before all size-dependent allocations. Every
// source is shape/finite/hash-audited before and after its consumption;
// source payloads must remain immutable/alive during that call. No callback,
// concurrent-write permission or global shared-snapshot certification is
// implied. Reversal T_ml=T_lm^T and cross-slot frame/amplitude consistency
// are REQUIRED algebraic input contracts, authenticated by the outer owner.
// Zero-rank sources still consume their slot; A=0 still scans each T.
// Every exception poisons the stream; no partial result can be published.
// Strict binary64/nearest/gradual arithmetic with compensated scalar sums;
// underflow counts are diagnostics, not a rigorous error bound. No weights,
// source projection into A, clipping, diagonal repair or symmetrization.
class BoundedRestrictedPairCCSDParticleHoleAccumulator {
public:
    BoundedRestrictedPairCCSDParticleHoleAccumulator(std::uint64_t target_dimension,
        std::uint64_t occupied_count, std::uint64_t target_i, std::uint64_t target_j,
        std::uint64_t maximum_source_dimension,
        const BoundedRestrictedPairCCSDParticleHoleInventory&,
        const BoundedRestrictedPairCCSDParticleHoleCaps&);
    BoundedRestrictedPairCCSDParticleHoleAccumulator(const BoundedRestrictedPairCCSDParticleHoleAccumulator&) = delete;
    BoundedRestrictedPairCCSDParticleHoleAccumulator& operator=(const BoundedRestrictedPairCCSDParticleHoleAccumulator&) = delete;
    BoundedRestrictedPairCCSDParticleHoleAccumulator(BoundedRestrictedPairCCSDParticleHoleAccumulator&&) noexcept;
    BoundedRestrictedPairCCSDParticleHoleAccumulator& operator=(BoundedRestrictedPairCCSDParticleHoleAccumulator&&) = delete;
    void accumulate(const BoundedRestrictedPairCCSDParticleHoleSource&);
    BoundedRestrictedPairCCSDParticleHoleResult finish();
    const BoundedRestrictedPairCCSDParticleHoleMemoryPlan& memory() const;
    bool failed() const noexcept { return failed_; }
    bool finished() const noexcept { return finished_; }
private:
    void require_active() const;
    bool active_ = false, failed_ = false, finished_ = false;
    BoundedRestrictedPairCCSDParticleHoleCaps caps_;
    BoundedRestrictedPairCCSDParticleHoleResult result_;
    std::vector<double> compensation_, intermediate_;
};

} // namespace vibeqc
