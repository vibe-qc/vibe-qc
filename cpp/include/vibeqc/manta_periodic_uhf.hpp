#pragma once
#include "vibeqc/manta_periodic_rhf.hpp"
#include "vibeqc/bounded_periodic_uhf_solver.hpp"

namespace vibeqc::manta {
// Same finite physical controls; the SCF leaf uses per-spin occupations.
struct PeriodicUHFOptions : PeriodicRHFOptions {};
struct PeriodicUHFPlan {
    BlochOneElectronPlan overlap,kinetic;
    PeriodicNuclearPlan nuclear;
    PeriodicFockPlan fock;
    BoundedPeriodicUHFMemoryPlan solver;
    PeriodicMadelungPlan madelung;
    Count n_alpha=0,n_beta=0;
    Count n_basis=0,momenta=0,electrons=0,driver_live_bytes=0,modeled_peak_bytes=0;
    Count fock_calls_upper_bound=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
    Count nuclear_image_candidates_upper_bound=0;
    Count exchange_correction_kernel_terms_per_call=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count history_capacity=0,history_live_bytes=0;
};
struct PeriodicUHFReceipt : PeriodicRHFReceipt {
    PeriodicUHFReceipt() { algorithm="finite-3d-reciprocal-uhf"; }
    Count n_alpha=0,n_beta=0;
};
struct PeriodicUHFResult {
    PeriodicUHFPlan plan;
    PeriodicUHFReceipt receipt;
    std::optional<BoundedPeriodicUHFResult> numerical;
    std::vector<std::complex<double>> overlap,hcore;
    // One copied scalar snapshot per evaluated solver iteration, no matrices.
    // The independent terminal Fock confirmation is not another iteration.
    std::vector<BoundedPeriodicUHFSnapshot> scf_history;
    bool converged() const { return numerical && numerical->converged() && receipt.confirmed; }
};
// Neutral all-electron 3D collinear fixed-spin UHF on a uniform full mesh.
// Multiplicity fixes N_alpha-N_beta; at least one virtual in each spin.
// D/F[2,K,n,n], C[2,K,n,r], eps[2,K,r]; S/H[K,n,n]. Both spin densities
// enter every physical build. No spin transfer, smearing or stability claim.
PeriodicUHFPlan plan_periodic_uhf(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,const PeriodicUHFOptions&,const ResourceBudget&);
PeriodicUHFResult run_periodic_uhf(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& offsets,
    const md::Matrix& translations,const PeriodicUHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
