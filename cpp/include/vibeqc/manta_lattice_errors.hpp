#pragma once
#include "vibeqc/manta_extents.hpp"
#include "vibeqc/manta_lattice_tail.hpp"

namespace vibeqc::manta {
struct LatticeGaussianPlan {
    LatticeTailPlan coefficients;
    LatticeTailRemainderPlan compact;
    Count scalar_terms=0,kernel_terms=0,envelope_bytes=0,workspace_bytes=0,modeled_peak_bytes=0;
};
struct LatticeGaussianError {
    double compact_remainder=0,tail_penetration=0,full_moment_tail=0,total=0;
    double bounded_kernel_sup=0;
    // Isolated erfc only: interior absolute radial moments tighten the
    // conditional mass/radius compact bound. Lattice complement stays at one.
    double compact_moment_scale=1;
    std::string source_identity,target_envelope_identity,source_envelope_identity;
    std::string qualification="finite-domain-gaussian-approximation";
    bool roundoff_included=false;
    Count moment_products_evaluated=0,reciprocal_terms_evaluated=0;
};
struct LatticeGaussianInteraction {
    LatticeGaussianPlan plan;
    LatticeTailMatrix matrix;
    LatticeTailRemainder compact;
    LatticeGaussianError error;
};
LatticeGaussianPlan plan_lattice_gaussian_interaction(int order,Count real_cells,
    Count reciprocal_vectors,Count near_cells,const ResourceBudget&);
// Native producer envelopes, with centers and full/tail radial moments, give
// uniform per-component bounds. This is not an infinite-lattice certificate,
// density aggregation, source-specific Fock admission or a point-self policy.
#define MANTA_LATTICE_INTERACTION(A,B) \
LatticeGaussianInteraction lattice_gaussian_interaction(const Eigen::Matrix3d& cell, \
    double eta,const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal, \
    const std::vector<LatticeIndex>& near,const A& target,const B& source,int order,const ResourceBudget&);
MANTA_LATTICE_INTERACTION(PairExtent,PairExtent)
MANTA_LATTICE_INTERACTION(PairExtent,AuxiliaryExtent)
MANTA_LATTICE_INTERACTION(AuxiliaryExtent,PairExtent)
MANTA_LATTICE_INTERACTION(AuxiliaryExtent,AuxiliaryExtent)
#undef MANTA_LATTICE_INTERACTION
LatticeGaussianPlan plan_erfc_gaussian_interaction(int order,const ResourceBudget&);
// Same native envelopes and tail ledger, isolated erfc kernel only. No cell,
// Ewald gauge/near subtraction or caller-invented compact support is accepted.
#define MANTA_ERFC_INTERACTION(A,B) \
LatticeGaussianInteraction erfc_gaussian_interaction(double omega,const A& target,const B& source, \
    int order,const ResourceBudget&);
MANTA_ERFC_INTERACTION(PairExtent,PairExtent)
MANTA_ERFC_INTERACTION(PairExtent,AuxiliaryExtent)
MANTA_ERFC_INTERACTION(AuxiliaryExtent,PairExtent)
MANTA_ERFC_INTERACTION(AuxiliaryExtent,AuxiliaryExtent)
#undef MANTA_ERFC_INTERACTION
} // namespace vibeqc::manta
