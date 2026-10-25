#pragma once
#include "vibeqc/manta_periodic_nuclear_bounds.hpp"
#include "vibeqc/manta_periodic_exchange_bounds.hpp"
#include "vibeqc/manta_periodic_rhf.hpp"
#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/bounded_periodic_rhf_solver.hpp"

namespace vibeqc::manta {
// Fixed-projector RI supplies selected J/K components; the existing bounded
// restricted leaf, complementary exact and physical S/T/V/Ewald sources are reused.
struct PeriodicRIRHFOptions : PeriodicRHFOptions {
    PeriodicGDFShortRangeOptions fitting;
    RIOptions ri;
    PeriodicRIAccuracyOptions ri_accuracy;
    PeriodicRIFittingAccuracyOptions fitting_accuracy;
    PeriodicOneElectronAccuracyOptions one_electron_accuracy;
    PeriodicExchangeAccuracyOptions exchange_accuracy;
    PeriodicNuclearAccuracyOptions nuclear_accuracy;
    std::string coulomb_algorithm="ri",exchange_algorithm="ri";
    PeriodicRIRHFOptions() { pair_cutoff_bohr=fitting.pair_cutoff_bohr; }
};
struct PeriodicRIRHFPlan {
    BlochOneElectronPlan overlap,kinetic;
    PeriodicNuclearPlan nuclear;
    PeriodicRIFockPlan fock;
    PeriodicHartreePlan exact_hartree;
    PeriodicExchangePlan exact_exchange;
    Count mixed_driver_live_bytes=0,two_electron_peak_bytes=0,two_electron_kernel_terms_per_call=0;
    BoundedPeriodicRHFMemoryPlan solver;
    std::optional<BoundedPeriodicRHFMemoryPlan> density_envelope;
    std::optional<PeriodicRIFittingBoundPlan> fitting_accuracy;
    std::optional<PeriodicOneElectronAccuracyPlan> one_electron_accuracy;
    std::optional<PeriodicExchangeAccuracyPlan> exchange_accuracy;
    std::optional<PeriodicNuclearAccuracyPlan> nuclear_accuracy;
    PeriodicMadelungPlan madelung;
    Count n_basis=0,momenta=0,electrons=0,driver_live_bytes=0,modeled_peak_bytes=0;
    Count fock_calls_upper_bound=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
    Count nuclear_image_candidates_upper_bound=0;
    Count exchange_correction_kernel_terms_per_call=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count history_capacity=0,history_live_bytes=0;
};
struct PeriodicRIRHFReceipt : PeriodicRHFReceipt {
    PeriodicAssemblyRoundoff assembly_roundoff;
    PeriodicRIFockInfo fitting_info;
    std::optional<BoundedPeriodicDensityEnvelope> accuracy_density;
    std::optional<PeriodicOneElectronAccuracyResult> one_electron_accuracy;
    std::optional<PeriodicExchangeAccuracyResult> exchange_accuracy;
    std::optional<PeriodicNuclearAccuracyResult> nuclear_accuracy;
    PeriodicRIFockReceipt fitted_fock; // Last evaluated RI component, before exact K/correction.
    std::string coulomb_algorithm="ri",exchange_algorithm="ri",coulomb_identity,exchange_identity;
    std::optional<PeriodicHartreeReceipt> exact_hartree;
    std::optional<PeriodicExchangeReceipt> exact_exchange;
    PeriodicRIRHFReceipt() { algorithm="finite-3d-streamed-ri-rhf"; }
};
struct PeriodicRIRHFResult {
    PeriodicRIRHFPlan plan;
    PeriodicRIRHFReceipt receipt;
    std::optional<BoundedPeriodicRHFResult> numerical;
    std::vector<std::complex<double>> overlap,hcore;
    // Actual generated Cartesian coordinates, retained for exact frame reuse.
    md::Matrix momenta;
    // One copied scalar snapshot per evaluated solver iteration, no matrices.
    // The independent terminal Fock confirmation is not another iteration.
    std::vector<BoundedPeriodicRHFSnapshot> scf_history;
    bool converged() const { return numerical && numerical->converged() && receipt.confirmed; }
};
// RI K vectors use G+k_source-k_target, with empty diagonal segments.
// Exact K vectors use G+k_target-k_source, with nonempty diagonal segments.
// J and K independently select ri/exact; at least one target must use RI.
// Gamma/nuclear vectors share a primed G list. HF and fitting pair cutoffs
// must agree. Numerical fitting omega remains separate from nuclear alpha.
// Neutral all-electron 3D singlet on a uniform full mesh. The restricted
// solver owns spin-summed D[K,n,n], F[K,n,n], C[K,n,r], eps[K,r]. Its
// callback embeds D_alpha=D_beta=D/2 in the shared fitted spin builder;
// no unrestricted result, stopping criterion or occupation is relabeled.
// At least one retained virtual orbital; no smearing or stability claim.
PeriodicRIRHFPlan plan_periodic_ri_rhf(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,const PeriodicRIRHFOptions&,const ResourceBudget&);
PeriodicRIRHFResult run_periodic_ri_rhf(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& offsets,
    const md::Matrix& translations,const PeriodicRIRHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
