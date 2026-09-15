#pragma once

// Actual all-q Gaussian replacement of ONLY the bare W1/W2/WX particle-hole
// group during pair-space CCSD iteration. The rest remains the bounded
// common-frame Galerkin reference. Not production DLPNO, not elimination of
// the origin provider/common owners, and not a bitwise reproduction of the
// original provider's separately certified real-row projection.
#include "vibeqc/periodic_gaussian_pair_particle_hole.hpp"

namespace vibeqc {

struct PeriodicGaussianPairCCSDPhysicalParticleHolePlan {
    PeriodicGaussianPairCCSDPlan ccsd;
    std::uint64_t occupied_count = 0, common_virtual_dimension = 0, pair_count = 0;
    std::uint64_t maximum_pair_rank = 0, pair_coefficient_bytes = 0;
    std::uint64_t maximum_frame_numerical_bytes = 0, maximum_geometry_numerical_bytes = 0;
    bool domain_generated = false;
    std::uint64_t borrowed_warmstart_numerical_bytes = 0, borrowed_basis_numerical_bytes = 0;
    // Additional to the complete source owners already counted by CCSD:
    // caller gauge, Wannier coefficients, common PAO domain/space and AO/aux.
    std::uint64_t additional_retained_numerical_bytes = 0;
    std::uint64_t duplicate_frame_geometry_padding_bytes = 0, duplicate_t_role_padding_bytes = 0;
    std::uint64_t producer_transient_upper_bytes = 0;
    std::uint64_t wrapper_fixed_control_storage_bytes = 0;
    std::uint64_t producer_additional_control_storage_bytes = 0;
    std::uint64_t metadata_work_units = 0, source_validation_work_units_per_pass = 0;
    std::uint64_t source_validation_passes_upper_bound = 0, finalization_work_units = 0;
    std::uint64_t wrapper_work_units_upper_bound = 0, work_units_upper_bound = 0;
    std::uint64_t particle_hole_calls_upper_bound = 0, factor_panels_upper_bound = 0;
    std::uint64_t tile_calls_upper_bound = 0, reciprocal_candidates_upper_bound = 0, image_candidates_upper_bound = 0;
    // The producer's transient declaration includes explicit nested role
    // padding; this remains a conservative reservation, not an RSS claim.
    std::uint64_t peak_owned_numerical_bytes = 0, per_worker_inventoried_bytes = 0;
    std::uint64_t required_node_memory_bytes = 0;
};

// Count/shape/work/control admission precedes a bounded pair-owner metadata
// census. No Reader, trial amplitudes, AO/gauge scan or numerical allocation.
// The actual current Reader reaches the producer only inside the admitted
// split solver. All positive resource/scientific controls remain explicit.
PeriodicGaussianPairCCSDPhysicalParticleHolePlan plan_periodic_gaussian_pair_ccsd_physical_particle_hole(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationPAODomain&,const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianRealLocalProvider&,
    const PeriodicGaussianPairMP2Result&,const PeriodicGaussianPairInteractionConfig&,
    const PeriodicGaussianPairInteractionOptions&,const PeriodicGaussianPairParticleHoleCaps&,
    const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
    const PeriodicGaussianPairCCSDCaps&);

// Borrows every physical owner and the immutable exact gauge view through
// return. Normal scalar progress is the only user callback; source payloads
// and owned-buffer addresses are checked around it before old raw pointers
// may be consumed. No user-defined particle-hole producer is accepted.
// Each target builds the native all-q current-snapshot contribution once,
// visits its immutable bare result once, then releases it. No all-q cache,
// completed common residual or temporary full common T2 is created here.
// Successful source/execution/consumed-stream checks privately qualify and
// re-seal the existing CCSD result. Nonconvergence remains last-evaluated;
// per-cell totals additionally require its ordinary complete-torus gate.
PeriodicGaussianPairCCSDResult run_periodic_gaussian_pair_ccsd_physical_particle_hole(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,
    const PeriodicCorrelationPAODomain&,const PeriodicCorrelationPAOSpace&,
    const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianRealLocalProvider&,
    const PeriodicGaussianPairMP2Result&,const PeriodicGaussianPairInteractionConfig&,
    const PeriodicGaussianPairInteractionOptions&,const PeriodicGaussianPairParticleHoleCaps&,
    const PeriodicGaussianPairCCSDOptions&,const PeriodicGaussianPairCCSDLiveInventory&,
    const PeriodicGaussianPairCCSDCaps&,PeriodicGaussianPairCCSDCallback progress=nullptr,
    void* progress_context=nullptr);

} // namespace vibeqc
