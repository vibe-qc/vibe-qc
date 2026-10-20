#pragma once
#include "vibeqc/manta_periodic_fock.hpp"
#include "vibeqc/manta_periodic_nuclear.hpp"
#include "vibeqc/bounded_periodic_rhf_solver.hpp"
#include "vibeqc/kmesh_address.hpp"

namespace vibeqc::manta {
struct PeriodicRHFOptions {
    PeriodicRHFOptions();
    BoundedPeriodicRHFOptions scf;
    double alpha=.7,nuclear_cutoff_bohr=8.;
    double confirmation_fock_tolerance=1e-8,confirmation_energy_tolerance=1e-10;
    Count reciprocal_batch=64;
    std::optional<double> pair_cutoff_bohr;
    std::string exchange_q0="strict-zero-mode";
    std::optional<double> madelung_alpha;
    double madelung_tolerance=1e-12;
};
struct PeriodicMadelungPlan {
    double alpha=0.,real_cutoff_bohr=0.,reciprocal_cutoff_bohr_inv=0.;
    Count real_candidates=0,reciprocal_candidates=0,modeled_peak_bytes=0,kernel_terms=0;
};
struct PeriodicRHFPlan {
    BlochOneElectronPlan overlap,kinetic;
    PeriodicNuclearPlan nuclear;
    PeriodicFockPlan fock;
    BoundedPeriodicRHFMemoryPlan solver;
    PeriodicMadelungPlan madelung;
    Count n_basis=0,momenta=0,electrons=0,driver_live_bytes=0,modeled_peak_bytes=0;
    Count fock_calls_upper_bound=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
    Count nuclear_image_candidates_upper_bound=0;
    Count exchange_correction_kernel_terms_per_call=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count history_capacity=0,history_live_bytes=0;
};
struct PeriodicRHFReceipt {
    std::string provider="manta",algorithm="finite-3d-reciprocal-rhf";
    std::string boundary="neutral-3d-tinfoil-zero-average;exchange-q0-omitted-no-madelung";
    std::string source_identity,overlap_identity,kinetic_identity,nuclear_identity,nuclear_energy_identity;
    std::string final_density_identity,final_fock_identity;
    std::string exchange_q0="strict-zero-mode",exchange_correction_identity;
    bool confirmed=false;
    double nuclear_real=0.,nuclear_reciprocal=0.,nuclear_self=0.,nuclear_background=0.,nuclear_energy=0.;
    double one_electron_energy=0.,coulomb_energy=0.,exchange_energy=0.,energy=0.;
    double madelung=0.,uncorrected_exchange_energy=0.,exchange_correction_energy=0.;
    double confirmation_fock_error=0.,confirmation_energy_error=0.;
    Count nuclear_pair_images=0,fock_calls=0,shell_calls=0,primitive_pairs=0,modeled_peak_bytes=0;
    Count madelung_real_vectors=0,madelung_reciprocal_vectors=0;
    std::string domain_selection="explicit-vectors",domain_identity;
    double reciprocal_energy_cutoff=0.;
    Count domain_translations=0,domain_hartree_vectors=0,domain_exchange_vectors=0,domain_conjugacy_vectors=0;
};
struct PeriodicRHFResult {
    PeriodicRHFPlan plan;
    PeriodicRHFReceipt receipt;
    std::optional<BoundedPeriodicRHFResult> numerical;
    std::vector<std::complex<double>> overlap,hcore;
    // One copied scalar snapshot per evaluated solver iteration, no matrices.
    // The independent terminal Fock confirmation is not another iteration.
    std::vector<BoundedPeriodicRHFSnapshot> scf_history;
    bool converged() const { return numerical && numerical->converged() && receipt.confirmed; }
};
// All-electron neutral singlet, uniform FULL RegularKMesh only. Explicit G/Q
// and AO-image supports define the finite Hamiltonian; no infinite-cutoff
// convergence is inferred. Optional BvK-ewald uses the shared point-charge
// Ewald producer with its own admitted cutoffs. S/T/V and nuclear energy are
// computed from the physical inputs, never supplied as reference matrices.
PeriodicRHFPlan plan_periodic_rhf(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,const PeriodicRHFOptions&,const ResourceBudget&);
PeriodicRHFResult run_periodic_rhf(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& offsets,
    const md::Matrix& translations,const PeriodicRHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
