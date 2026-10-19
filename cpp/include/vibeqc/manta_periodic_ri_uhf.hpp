#pragma once
#include "vibeqc/manta_periodic_exchange_bounds.hpp"
#include "vibeqc/manta_periodic_uhf.hpp"
#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/bounded_periodic_uhf_solver.hpp"

namespace vibeqc::manta {
// Fixed-projector RI replaces the two-electron source; the existing bounded
// fixed-spin numerical leaf and physical S/T/V/Ewald producers are reused.
struct PeriodicRIUHFOptions : PeriodicUHFOptions {
    PeriodicGDFShortRangeOptions fitting;
    RIOptions ri;
    PeriodicRIAccuracyOptions ri_accuracy;
    PeriodicOneElectronAccuracyOptions one_electron_accuracy;
    PeriodicExchangeAccuracyOptions exchange_accuracy;
    std::string coulomb_algorithm="ri",exchange_algorithm="ri";
    PeriodicRIUHFOptions() { pair_cutoff_bohr=fitting.pair_cutoff_bohr; }
};
struct PeriodicRIUHFPlan {
    BlochOneElectronPlan overlap,kinetic;
    PeriodicNuclearPlan nuclear;
    PeriodicRIFockPlan fock;
    PeriodicHartreePlan exact_hartree;
    PeriodicExchangePlan exact_exchange;
    Count mixed_driver_live_bytes=0,two_electron_peak_bytes=0,two_electron_kernel_terms_per_call=0;
    BoundedPeriodicUHFMemoryPlan solver;
    std::optional<BoundedPeriodicRHFMemoryPlan> density_envelope;
    std::optional<PeriodicOneElectronAccuracyPlan> one_electron_accuracy;
    std::optional<PeriodicExchangeAccuracyPlan> exchange_accuracy;
    PeriodicMadelungPlan madelung;
    Count n_alpha=0,n_beta=0;
    Count n_basis=0,momenta=0,electrons=0,driver_live_bytes=0,modeled_peak_bytes=0;
    Count fock_calls_upper_bound=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
    Count nuclear_image_candidates_upper_bound=0;
    Count exchange_correction_kernel_terms_per_call=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count history_capacity=0,history_live_bytes=0;
};
struct PeriodicRIUHFReceipt : PeriodicUHFReceipt {
    PeriodicRIFockInfo fitting_info;
    std::optional<BoundedPeriodicDensityEnvelope> accuracy_density;
    std::optional<PeriodicOneElectronAccuracyResult> one_electron_accuracy;
    std::optional<PeriodicExchangeAccuracyResult> exchange_accuracy;
    PeriodicRIFockReceipt fitted_fock; // Last evaluated RI component, before exact K/correction.
    std::string coulomb_algorithm="ri",exchange_algorithm="ri",coulomb_identity,exchange_identity;
    std::optional<PeriodicHartreeReceipt> exact_hartree;
    std::optional<PeriodicExchangeReceipt> exact_exchange;
    PeriodicRIUHFReceipt() { algorithm="finite-3d-streamed-ri-uhf"; }
};
struct PeriodicRIUHFResult {
    PeriodicRIUHFPlan plan;
    PeriodicRIUHFReceipt receipt;
    std::optional<BoundedPeriodicUHFResult> numerical;
    std::vector<std::complex<double>> overlap,hcore;
    // Actual generated Cartesian coordinates, retained for exact frame reuse.
    md::Matrix momenta;
    // One copied scalar snapshot per evaluated solver iteration, no matrices.
    // The independent terminal Fock confirmation is not another iteration.
    std::vector<BoundedPeriodicUHFSnapshot> scf_history;
    bool converged() const { return numerical && numerical->converged() && receipt.confirmed; }
};
// RI K vectors use G+k_source-k_target, with empty diagonal segments.
// Exact K vectors use G+k_target-k_source, with nonempty diagonal segments.
// J and K independently select ri/exact; at least one target must use RI.
// Gamma/nuclear vectors share a primed G list. HF and fitting pair cutoffs
// must agree. Numerical fitting omega remains separate from nuclear alpha.
// Neutral all-electron 3D collinear fixed-spin UHF on a uniform full mesh.
// Multiplicity fixes N_alpha-N_beta; at least one virtual in each spin.
// D/F[2,K,n,n], C[2,K,n,r], eps[2,K,r]; S/H[K,n,n]. Both spin densities
// enter every physical build. No spin transfer, smearing or stability claim.
PeriodicRIUHFPlan plan_periodic_ri_uhf(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,const PeriodicRIUHFOptions&,const ResourceBudget&);
PeriodicRIUHFResult run_periodic_ri_uhf(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& offsets,
    const md::Matrix& translations,const PeriodicRIUHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
