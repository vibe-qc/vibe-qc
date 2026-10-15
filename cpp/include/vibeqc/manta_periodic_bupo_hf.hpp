#pragma once
#include "vibeqc/manta_periodic_electrostatic.hpp"
#include "vibeqc/manta_periodic_uhf.hpp"
#include "vibeqc/manta_periodic_one_electron_bounds.hpp"
#include "vibeqc/manta_periodic_exchange_bounds.hpp"

namespace vibeqc::manta {
struct PeriodicBupoHFOptions {
    BoundedPeriodicRHFOptions scf=PeriodicRHFOptions{}.scf;
    std::string spin_mode="restricted",exchange_q0="strict-zero-mode";
    std::optional<double> madelung_alpha;
    double madelung_tolerance=1e-12;
    Count reciprocal_batch=64;
    double charge_tolerance=1e-10;
    double maximum_electrostatic_energy_error=1e-8,maximum_electrostatic_matrix_error=1e-8;
    double maximum_overlap_image_error=1e-10,maximum_kinetic_image_error=1e-8,maximum_kinetic_energy_error=1e-8;
    bool bound_exchange=false;
    Count exchange_bound_cube_radius=0;
    double maximum_exchange_matrix_error=1e-8,maximum_exchange_energy_error=1e-8;
    double confirmation_fock_tolerance=1e-8,confirmation_energy_tolerance=1e-10;
};
struct PeriodicBupoHFPlan {
    PeriodicElectrostaticPlan electrostatic;
    PeriodicOneElectronImagePlan one_electron_images;
    BlochOneElectronPlan kinetic;
    PeriodicExchangePlan exchange;
    PeriodicExchangeBoundPlan exchange_bounds;
    PeriodicMadelungPlan madelung;
    BoundedPeriodicUHFMemoryPlan solver;
    Count n_basis=0,momenta=0,spins=0,electrons=0,n_alpha=0,n_beta=0,cells=0;
    Count fock_calls_upper_bound=0,provider_kernel_terms_per_call=0;
    Count driver_live_bytes=0,modeled_peak_bytes=0,shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
    Count history_capacity=0,history_live_bytes=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    Count source_live_bytes=0,source_shell_calls=0,source_primitive_pairs=0,source_scalar_terms=0,source_kernel_terms=0;
};
struct PeriodicBupoHFPreparationReceipt {
    std::string identity,pair_source_identity,tree_identity,kernel_identity,electronic_source_identity;
    Count pair_cube_radius=0,real_cube_radius=0,reciprocal_cube_radius=0,near_cube_radius=0,pairs=0,nodes=0,leaf_capacity=0;
    int order=0,branching=0,maximum_depth=0;
    double eta=0,opening=0,integral_tolerance=0,pointwise_tolerance=0,tail_mass_tolerance=0,minimum_radius=0,maximum_radius=0;
    std::optional<double> pair_cutoff_bohr;
};
struct PeriodicBupoHFReceipt : PeriodicUHFReceipt {
    PeriodicBupoHFReceipt() { algorithm="periodic-bupo-hf;finite-reciprocal-exchange"; }
    std::string bupo_source_identity;
    std::string accuracy_qualification="electrostatic-and-one-electron-images-at-evaluated-density;finite-exchange";
    PeriodicElectrostaticReceipt electrostatic;
    PeriodicExchangeReceipt exchange;
    std::optional<PeriodicExchangeBoundReceipt> exchange_bounds;
    PeriodicBupoReceipt coulomb;
    double maximum_electrostatic_energy_error=0,maximum_electrostatic_matrix_error=0;
    double maximum_observed_electrostatic_energy_error=0,maximum_observed_electrostatic_matrix_error=0;
    double maximum_overlap_image_error=0,maximum_kinetic_image_error=0,maximum_kinetic_energy_error=0;
    double kinetic_energy_error_bound=0,maximum_observed_kinetic_energy_error=0;
    bool one_electron_accuracy_passed=false;
    double maximum_exchange_matrix_error=0,maximum_exchange_energy_error=0;
    double exchange_matrix_error_bound=0,exchange_energy_error_bound=0;
    double maximum_observed_exchange_matrix_error=0,maximum_observed_exchange_energy_error=0;
    double exchange_correction_matrix_error=0,exchange_correction_energy_error=0;
    bool exchange_accuracy_passed=false;
    Count exchange_bound_cube_radius=0;
    bool electrostatic_accuracy_passed=false;
    std::optional<PeriodicBupoHFPreparationReceipt> preparation;
};
struct PeriodicBupoHFResult {
    PeriodicBupoHFPlan plan;
    PeriodicBupoHFReceipt receipt;
    PeriodicOneElectronImageResult one_electron_images;
    std::optional<BoundedPeriodicRHFResult> rhf;
    std::optional<BoundedPeriodicUHFResult> uhf;
    std::vector<std::complex<double>> overlap,hcore;
    std::vector<BoundedPeriodicRHFSnapshot> rhf_history;
    std::vector<BoundedPeriodicUHFSnapshot> uhf_history;
    bool converged() const { return receipt.confirmed && (rhf ? rhf->converged() : uhf && uhf->converged()); }
};
// Neutral all-electron RHF/UHF, original uniform full mesh, fixed occupations.
// The immutable native BUPO source owns S/T/J/V domains. Exact reciprocal K
// uses its original ordered AO images and cutoff, with explicit Q supports.
// Electrostatic and S/T image targets are enforced at the evaluated density,
// with optional full exchange certification at fixed mesh/Madelung. These
// are not promoted to basis, SCF-state or roundoff error bounds.
PeriodicBupoHFPlan plan_periodic_bupo_hf(const PeriodicBupoSource&,const PeriodicSystem&,const RegularKMesh&,
    Count exchange_vectors_per_pair,const PeriodicBupoHFOptions&,const ResourceBudget&);
PeriodicBupoHFResult run_periodic_bupo_hf(std::shared_ptr<const PeriodicBupoSource>,const PeriodicSystem&,
    const RegularKMesh&,const md::Matrix& exchange_vectors,const std::vector<Count>& offsets,
    const PeriodicBupoHFOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
