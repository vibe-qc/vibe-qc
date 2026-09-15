#pragma once

/// Bounded physical pair Coulomb block in one common semicanonical PAO space.
/// Sun et al., doi:10.1063/1.4998644, Eqs.(3),(4),(13),(16),(21), and
/// Nejad et al., doi:10.1063/5.0290816, Eqs.(2)-(10),(26)-(28).
/// G_ab=(i* a|j* b)=sum_q,P conj(L_ai(q,P))*L_jb(q,P).
/// Both orientations and every q are actually evaluated. No molecular
/// same-orientation Gram, q weight or real-part shortcut is used.
/// This consumes the finite-image global-auxiliary RI store, not the local
/// chargeless auxiliary-domain approximation. The output is complex and is
/// NOT a real-orbital/permutation or omitted-image-error certificate.

#include "vibeqc/periodic_correlation_local_factors.hpp"

namespace vibeqc {

struct PeriodicCorrelationPairIntegralSelection {
    std::uint64_t occupied_i = 0, cell_i = 0;
    std::uint64_t occupied_j = 0, cell_j = 0;
    std::uint64_t virtual_begin = 0, virtual_count = 0;
    std::uint64_t virtual_translation_cell = 0;
    /// Positive virtual tile extent; auxiliary tiles follow the sealed store.
    std::uint64_t virtual_block = 0;
};

struct PeriodicCorrelationPairIntegralCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_work_units = 0;
    std::uint64_t maximum_factor_builds = 0;
    std::uint64_t maximum_tile_visits = 0;
};

struct PeriodicCorrelationPairIntegralMemoryPlan {
    std::uint64_t n_virtual = 0;
    std::uint64_t retained_output_bytes = 0, compensation_bytes = 0;
    std::uint64_t retained_left_factor_bytes = 0;
    std::uint64_t maximum_factor_owned_bytes = 0;
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t factor_builds = 0, tile_visits_upper_bound = 0;
    std::uint64_t work_units_upper_bound = 0;
    std::uint64_t required_node_memory_bytes = 0;
    PeriodicCorrelationLocalFactorMemoryPlan maximum_factor;
};

class PeriodicCorrelationPairIntegralBlock {
public:
    PeriodicCorrelationPairIntegralBlock(const PeriodicCorrelationPairIntegralBlock&) = delete;
    PeriodicCorrelationPairIntegralBlock& operator=(const PeriodicCorrelationPairIntegralBlock&) = delete;
    PeriodicCorrelationPairIntegralBlock(PeriodicCorrelationPairIntegralBlock&&) noexcept = default;
    PeriodicCorrelationPairIntegralBlock& operator=(PeriodicCorrelationPairIntegralBlock&&) noexcept = default;
    const PeriodicCorrelationPairIntegralMemoryPlan& memory() const noexcept { return memory_; }
    const PeriodicCorrelationPairIntegralSelection& selection() const noexcept { return selection_; }
    const std::complex<double>* data() const;
    const std::string& identity_sha256() const noexcept { return identity_; }
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& consumed_factors_sha256() const noexcept { return consumed_; }
    const std::string& store_identity_sha256() const noexcept { return store_; }
    std::uint64_t factor_builds() const noexcept { return factor_builds_; }
    std::uint64_t tile_visits() const noexcept { return tile_visits_; }
    bool finite_image_reference() const noexcept { return true; }
    bool real_orbital_integrals_certified() const noexcept { return false; }
private:
    PeriodicCorrelationPairIntegralBlock() = default;
    PeriodicCorrelationPairIntegralMemoryPlan memory_;
    PeriodicCorrelationPairIntegralSelection selection_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::vector<std::complex<double>> values_;
    std::uint64_t factor_builds_ = 0, tile_visits_ = 0;
    std::string identity_, payload_, consumed_, store_;
    friend PeriodicCorrelationPairIntegralBlock build_periodic_correlation_pair_integral_block(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
        const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
        const std::complex<double>*, std::size_t, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationPairIntegralSelection&,
        const PeriodicCorrelationPairIntegralCaps&);
};

/// O(1), allocation-free numerical preflight. The complete inner factor
/// inventory is counted once, plus outer G, compensation and retained VO.
/// All other live caller buffers must be in the admitted external inventory.
PeriodicCorrelationPairIntegralMemoryPlan plan_periodic_correlation_pair_integral_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationPairIntegralSelection&);

/// Admission precedes gauge hashing, allocations and disk reads. The gauges
/// remain borrowed immutable through return. Any failure publishes no block.
/// Only G+compensation, one retained VO block and one building OV block are
/// live; there is no all-q local-factor cache or full AO tensor in memory.
PeriodicCorrelationPairIntegralBlock build_periodic_correlation_pair_integral_block(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationFactorStreamSchedule&,
    const PeriodicCorrelationPrivateFactorReader&, const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges, std::size_t gauge_element_count,
    const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationPairIntegralSelection&, const PeriodicCorrelationPairIntegralCaps&);

}  // namespace vibeqc
