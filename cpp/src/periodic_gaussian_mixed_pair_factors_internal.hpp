#pragma once

// Private arithmetic only: no physical-source factory or public numerical
// array route. Caller admits storage/work and verifies immutable input owners,
// source payloads, slots, kpoint and gauges before calling. No heap inside.
#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"

namespace vibeqc {
namespace periodic_gaussian_mixed_pair_detail {

struct GeometryInventory {
    std::uint64_t numerical_bytes = 0, control_bytes = 0, validation_work_units = 0;
};
// Metadata and checked arithmetic only. Numeric validation is separate and
// MUST follow the enclosing complete caps, including this inventory.
GeometryInventory plan_geometry(const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairPNOFrameView&,
    const PeriodicGaussianPairPNOFrameView&,const PeriodicCorrelationPAODomain&,
    const PeriodicCorrelationPAOSpace&,const PeriodicGaussianPairPNOGeometryView*,
    const PeriodicGaussianPairPNOGeometryView*);
void validate_geometry(const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairPNOFrameView&,
    const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOGeometryView*,
    const PeriodicGaussianPairPNOGeometryView*);

// Output: M*nAO complex lanes, contiguous columns, M=2+rA+rB in order
// [occupied i, occupied k, frame A, frame B]. The two occupied slots may be
// equal and the frames may alias; do NOT merge/orthogonalize duplicates.
// Workspace: nAO*(rA+rB+3) complex lanes if rA+rB>0, otherwise zero/null.
// Layout: PNO-column compensation, one streamed common virtual column,
// then two scratch columns required by fill_virtual_columns. All lanes are
// initialized/overwritten here; output/workspace must be disjoint. Input
// frames default to row-major real C[n_common,r] with the certified common
// selection/translation. An admitted optional geometry instead selects the
// original D[m,r] and that embedding's physical pair selection/translation.
// No D=X^T C recovery and no projection through the other pair is performed.
// The common path uses storage.coefficient_work_units_per_kpoint. Optional
// geometry uses the enclosing PLAN's doubled maximum-domain upper, once
// per complete fill, including both independent generation traversals.
void fill_coefficients(
    const PeriodicRestrictedMeanFieldState&,const std::complex<double>* gauges,
    const PeriodicCorrelationPAODomain&,const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairPNOFrameView&,const PeriodicGaussianPairPNOFrameView&,
    std::uint64_t occupied_slot_i,std::uint64_t occupied_slot_k,std::uint64_t kpoint,
    std::complex<double>* output,std::complex<double>* workspace,
    const PeriodicGaussianPairPNOGeometryView* geometry_a = nullptr,
    const PeriodicGaussianPairPNOGeometryView* geometry_b = nullptr);

} // namespace periodic_gaussian_mixed_pair_detail
} // namespace vibeqc
