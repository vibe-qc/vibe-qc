#pragma once

/// Cross-overlap of two independently selected/translated orthonormal PAO
/// spaces. Nejad doi:10.1063/5.0290816 Eqs.(2)-(10),(26)-(28),(42)-(44):
/// O_AB=(1/Nk) sum_k C_A(k)^H S(k) C_B(k). Each C is expanded through the
/// same retained-virtual projector and exact cell-phase leaf used by the
/// local density factors. No complete AO projector or finite-supercell
/// matrix is constructed, and complex overlap values are never discarded.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vibeqc/periodic_correlation_density_factors.hpp"

namespace vibeqc {

inline constexpr std::uint32_t kPeriodicCorrelationPAOOverlapContractVersion = 1;

struct PeriodicCorrelationPAOOverlapCaps {
    std::uint64_t maximum_owned_numerical_bytes = 0;
    std::uint64_t maximum_work_units = 0;
};

struct PeriodicCorrelationPAOOverlapMemoryPlan {
    std::uint64_t n_cells = 0, n_basis = 0, left_count = 0, right_count = 0;
    std::uint64_t output_bytes = 0, compensation_bytes = 0;
    std::uint64_t coefficient_panel_bytes = 0, scratch_bytes = 0;
    /// 32*left_count*right_count +16*nao*(left_count+right_count+2).
    std::uint64_t peak_owned_numerical_bytes = 0;
    std::uint64_t unique_domain_owners = 0, unique_space_owners = 0;
    std::uint64_t live_domain_bytes = 0, live_space_bytes = 0;
    std::uint64_t work_units = 0, required_node_memory_bytes = 0;
};

/// Only the compact row-major left_count*right_count overlap is retained.
/// Domain/space owners are borrowed while constructing; identical object
/// ADDRESSES are charged once, but distinct objects with equal content hashes
/// are charged separately. Shared immutable mean-field state is charged once
/// by admission. Other live caller storage must already be in its inventory.
///
/// Digest wire: length-prefixed UTF-8 strings, big-endian unsigned integers
/// and finite binary64 lanes (signed zero canonicalized). Payload domain
/// "vibeqc.periodic.correlation.pao-overlap.payload", u32 version=1,
/// u64 Nk,left_count,right_count, then row-major real/imaginary lanes.
/// Identity domain "vibeqc.periodic.correlation.pao-overlap.identity", v1,
/// state/calculation/allocation strings; left domain/space strings followed
/// by u64 begin,count,translation; the corresponding right fields; payload
/// digest string; and the implementation's fixed projection/phase policy.
class PeriodicCorrelationPAOSpaceOverlap {
public:
    PeriodicCorrelationPAOSpaceOverlap(const PeriodicCorrelationPAOSpaceOverlap&) = delete;
    PeriodicCorrelationPAOSpaceOverlap& operator=(const PeriodicCorrelationPAOSpaceOverlap&) = delete;
    PeriodicCorrelationPAOSpaceOverlap(PeriodicCorrelationPAOSpaceOverlap&&) noexcept = default;
    PeriodicCorrelationPAOSpaceOverlap& operator=(PeriodicCorrelationPAOSpaceOverlap&&) noexcept = default;
    const PeriodicCorrelationPAOOverlapMemoryPlan& memory() const noexcept { return memory_; }
    PeriodicCorrelationVirtualBlockSelection left_selection() const noexcept { return left_; }
    PeriodicCorrelationVirtualBlockSelection right_selection() const noexcept { return right_; }
    std::complex<double> element(std::size_t left, std::size_t right) const;
    const std::complex<double>* data() const;
    const std::string& payload_sha256() const noexcept { return payload_; }
    const std::string& identity_sha256() const noexcept { return identity_; }
private:
    PeriodicCorrelationPAOSpaceOverlap() = default;
    PeriodicCorrelationPAOOverlapMemoryPlan memory_;
    PeriodicCorrelationVirtualBlockSelection left_, right_;
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> state_;
    std::vector<std::complex<double>> values_;
    std::string payload_, identity_;
    friend PeriodicCorrelationPAOSpaceOverlap make_periodic_correlation_pao_space_overlap(
        const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain&,
        const PeriodicCorrelationPAOSpace&, const PeriodicCorrelationVirtualBlockSelection&,
        const PeriodicCorrelationPAODomain&, const PeriodicCorrelationPAOSpace&,
        const PeriodicCorrelationVirtualBlockSelection&, const PeriodicCorrelationPAOOverlapCaps&);
};

/// No size-dependent scans or allocations. Shapes, seals, byte/work overflow
/// and address-based owner lifetimes are inventoried before native buffers.
PeriodicCorrelationPAOOverlapMemoryPlan plan_periodic_correlation_pao_space_overlap(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain& left_domain,
    const PeriodicCorrelationPAOSpace& left_space, const PeriodicCorrelationVirtualBlockSelection& left,
    const PeriodicCorrelationPAODomain& right_domain, const PeriodicCorrelationPAOSpace& right_space,
    const PeriodicCorrelationVirtualBlockSelection& right);
PeriodicCorrelationPAOSpaceOverlap make_periodic_correlation_pao_space_overlap(
    const PeriodicCorrelationAdmittedReference&, const PeriodicCorrelationPAODomain& left_domain,
    const PeriodicCorrelationPAOSpace& left_space, const PeriodicCorrelationVirtualBlockSelection& left,
    const PeriodicCorrelationPAODomain& right_domain, const PeriodicCorrelationPAOSpace& right_space,
    const PeriodicCorrelationVirtualBlockSelection& right, const PeriodicCorrelationPAOOverlapCaps& caps);

}  // namespace vibeqc
