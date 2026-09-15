#pragma once

// Native domain-generated extension of the finite-source coupled-MP2
// reference. It streams actual pair domains into Eq.39 PNO generation and
// retains each actual pair PAO domain, real space and embedding. Independent
// CCSD singles borrow the diagonal bundles without duplicate embeddings.
// The common-factor and Galerkin solver bottlenecks are NOT removed here.
#include "vibeqc/periodic_gaussian_pair_mp2.hpp"
#include "vibeqc/periodic_gaussian_pair_domain_builder.hpp"
#include "vibeqc/periodic_gaussian_embedded_pair_pnos.hpp"
#include "vibeqc/periodic_gaussian_gram_pair_pnos.hpp"

namespace vibeqc {
struct PeriodicGaussianDomainPairMP2Options {
    PeriodicGaussianPairDomainOptions domain;
    // Sole pair PNO controls for this branch, including explicit projection
    // audit budgets. No duplicate legacy/common-space generation settings.
    PeriodicGaussianEmbeddedPairPNOOptions pnos;
    PeriodicGaussianPairSpaceOptions projection;
    BoundedRestrictedPairMP2SolverOptions solver;
    double maximum_occupied_virtual_fock_norm = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicGaussianDomainPairMP2Caps {
    PeriodicGaussianPairDomainCaps domain;
    PeriodicGaussianEmbeddedPairPNOCaps pnos;
    PeriodicGaussianPairSpaceCaps projection;
    BoundedRestrictedPairMP2SolverCaps solver;
    std::uint64_t maximum_owned_numerical_bytes = 0;
    // Driver/known-owner controls only. Opaque caller other-live bytes may
    // include enclosing controls; the whole-worker/node caps count those.
    std::uint64_t maximum_control_storage_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0, maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_pair_count = 0, maximum_integral_calls = 0;
    std::uint64_t maximum_progress_callbacks = 0, maximum_work_units = 0;
};

// Constant-size worst-case full rank admission, no pair scans or allocation.
// Includes complete borrowed builder/basis/provider roles, all retained pair
// outputs and complete original pair geometries, then the ragged solver. Actual
// per-pair domain/PNO/projection plans must fit the admitted macro envelope.
PeriodicGaussianPairMP2Plan plan_periodic_gaussian_domain_pair_mp2(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairDomainBuilder&,
    const PeriodicGaussianDomainPairMP2Options&,const PeriodicGaussianPairMP2LiveInventory&,
    const PeriodicGaussianDomainPairMP2Caps&);
// All i<=j placed pairs are still computed in this reference. The builder
// reuses exact translation-unique topology routing but no representative-only
// energy or space-group acceleration is claimed. Inputs immutable, no array
// overrides; progress carries only scalar events. No result on callback error.
PeriodicGaussianPairMP2Result run_periodic_gaussian_domain_pair_mp2(
    const PeriodicCorrelationAdmittedReference&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianRealLocalProvider&,const PeriodicGaussianPairDomainBuilder&,
    const PeriodicGaussianDomainPairMP2Options&,const PeriodicGaussianPairMP2LiveInventory&,
    const PeriodicGaussianDomainPairMP2Caps&,PeriodicGaussianPairMP2Callback progress=nullptr,
    void* progress_context=nullptr);

// Genuine DirectPAOGram source, without a common real-row provider. Nejad
// Eqs.37-44: native local-PAO seed -> retained pair frame -> coupled solve.
// The common orthonormal export and all placed pairs are still retained;
// this is not representative-only storage or production DLPNO.
struct PeriodicGaussianGramDomainPairMP2Config {
    PeriodicGaussianGramPairPNOConfig pnos;
};
struct PeriodicGaussianGramDomainPairMP2Options {
    PeriodicGaussianPairDomainOptions domain;
    PeriodicGaussianGramPairPNOOptions pnos;
    PeriodicGaussianPairSpaceOptions projection;
    BoundedRestrictedPairMP2SolverOptions solver;
    double maximum_occupied_virtual_fock_norm = std::numeric_limits<double>::quiet_NaN();
};
struct PeriodicGaussianGramDomainPairMP2Caps {
    PeriodicGaussianPairDomainCaps domain;
    PeriodicGaussianGramPairPNOCaps pnos;
    PeriodicGaussianPairSpaceCaps projection;
    BoundedRestrictedPairMP2SolverCaps solver;
    // Bare (zero-extra-control) seed reservation, NOT its enclosing cap.
    // Actual per-geometry seed control plans must fit before any allocation.
    std::uint64_t maximum_pno_leaf_control_storage_bytes = 0;
    std::uint64_t maximum_owned_numerical_bytes = 0, maximum_control_storage_bytes = 0;
    std::uint64_t maximum_per_worker_inventoried_bytes = 0, maximum_node_inventoried_bytes = 0;
    std::uint64_t maximum_pair_count = 0, maximum_gram_builds = 0;
    std::uint64_t maximum_factor_panels = 0, maximum_tile_calls = 0;
    std::uint64_t maximum_progress_callbacks = 0, maximum_work_units = 0;
};
// Count-only whole-owner admission. Actual AO/auxiliary and gauge byte
// inventories derive from immutable native HF/Wannier metadata, not arrays.
PeriodicGaussianPairMP2Plan plan_periodic_gaussian_domain_pair_mp2(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const PeriodicCorrelationWannier&,const PeriodicCorrelationRealLocalBasis&,
    const PeriodicGaussianPairDomainBuilder&,const PeriodicGaussianGramDomainPairMP2Config&,
    const PeriodicGaussianGramDomainPairMP2Options&,const PeriodicGaussianPairMP2LiveInventory&,
    const PeriodicGaussianGramDomainPairMP2Caps&);
// All source owners/arrays immutable through return, including callbacks.
// Original pair geometries remain owned by the result for independent singles.
// No common provider receipt exists; the result seals consumed Gram receipts.
PeriodicGaussianPairMP2Result run_periodic_gaussian_domain_pair_mp2(
    const PeriodicGaussianRHFResult&,const PeriodicCorrelationAdmittedReference&,
    const BasisSet& ao,const BasisSet& auxiliary,const PeriodicCorrelationWannier&,
    const std::complex<double>* gauges,std::size_t gauge_count,
    const PeriodicCorrelationRealLocalBasis&,const PeriodicGaussianPairDomainBuilder&,
    const PeriodicGaussianGramDomainPairMP2Config&,const PeriodicGaussianGramDomainPairMP2Options&,
    const PeriodicGaussianPairMP2LiveInventory&,const PeriodicGaussianGramDomainPairMP2Caps&,
    PeriodicGaussianPairMP2Callback progress=nullptr,void* progress_context=nullptr);
} // namespace vibeqc
