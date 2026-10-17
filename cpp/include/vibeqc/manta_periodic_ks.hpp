#pragma once
#include "vibeqc/manta_periodic_rhf.hpp"
#include "vibeqc/manta_periodic_quadrature.hpp"
#include "vibeqc/manta_periodic_semilocal_xc.hpp"
#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/bounded_periodic_functional_solver.hpp"

namespace vibeqc::manta {
struct PeriodicKSOptions {
    PeriodicKSOptions() : scf(PeriodicRHFOptions{}.scf) {}
    BoundedPeriodicRHFOptions scf;
    double alpha=.7,nuclear_cutoff_bohr=8.;
    double confirmation_fock_tolerance=1e-8,confirmation_energy_tolerance=1e-10;
    Count reciprocal_batch=64,grid_batch=64;
    std::optional<double> pair_cutoff_bohr;
};
// Composition deliberately prevents passing hybrid controls to the pure API.
struct PeriodicGlobalHybridKSOptions {
    PeriodicKSOptions ks;
    std::string exchange_q0="strict-zero-mode";
    std::optional<double> madelung_alpha;
    double madelung_tolerance=1e-12;
};
struct PeriodicRangeSeparatedKSOptions {
    PeriodicKSOptions ks;
    // This policy applies only to the singular Coulomb coefficient. Finite
    // erfc Q=0 terms are always present, including for screened-only hybrids.
    std::string exchange_q0="strict-zero-mode";
    std::optional<double> madelung_alpha;
    double madelung_tolerance=1e-12;
};
// Separate options keep numerical GDF splitting distinct from a functional's
// physical range. These interfaces qualify pure and full-Coulomb global KS.
struct PeriodicRIKSOptions {
    PeriodicKSOptions ks;
    PeriodicGDFShortRangeOptions fitting;
    RIOptions ri;
    PeriodicRIKSOptions() { ks.pair_cutoff_bohr=fitting.pair_cutoff_bohr; }
};
// Keep the fitted options a distinct type: passing them to a direct global
// interface must not silently discard the selected fitting source.
struct PeriodicRIGlobalHybridKSOptions {
    PeriodicKSOptions ks;
    std::string exchange_q0="strict-zero-mode";
    std::optional<double> madelung_alpha;
    double madelung_tolerance=1e-12;
    PeriodicGDFShortRangeOptions fitting;
    RIOptions ri;
    std::string coulomb_algorithm="ri",exchange_algorithm="ri";
    PeriodicRIGlobalHybridKSOptions() { ks.pair_cutoff_bohr=fitting.pair_cutoff_bohr; }
};
// Physical kernel metrics belong to K only. Even when both targets use RI,
// J keeps a separate bare-Coulomb source and every range keeps its own metric.
struct PeriodicRIRangeSeparatedKSOptions {
    PeriodicKSOptions ks;
    std::string exchange_q0="strict-zero-mode";
    std::optional<double> madelung_alpha;
    double madelung_tolerance=1e-12;
    PeriodicGDFShortRangeOptions fitting;
    RIOptions ri;
    std::string coulomb_algorithm="ri",exchange_algorithm="ri";
    PeriodicRIRangeSeparatedKSOptions() { ks.pair_cutoff_bohr=fitting.pair_cutoff_bohr; }
};
struct PeriodicKSExchangeComponentPlan {
    std::string kernel;
    double coefficient=0.,omega=0.;
    PeriodicExchangePlan exchange;
    std::optional<PeriodicRIFockPlan> fitting;
};
struct PeriodicKSExchangeComponentReceipt {
    std::string kernel;
    double coefficient=0.,omega=0.,energy=0.; // method-scaled, without Madelung
    PeriodicExchangeReceipt exchange; // unscaled child evidence
    std::optional<PeriodicRIFockInfo> fitting_info;
    std::optional<PeriodicRIFockReceipt> fitted_fock; // includes method coefficient
};
struct PeriodicKSPlan {
    BlochOneElectronPlan overlap,kinetic;
    PeriodicNuclearPlan nuclear;
    PeriodicHartreePlan hartree;
    std::optional<PeriodicRIFockPlan> fitting;
    PeriodicSemilocalXCPlan xc;
    std::optional<PeriodicExchangePlan> exchange;
    PeriodicMadelungPlan madelung;
    double exact_exchange_fraction=0.;
    bool range_separated=false;
    double long_range_exchange_fraction=0.,short_range_exchange_fraction=0.;
    std::vector<PeriodicKSExchangeComponentPlan> exchange_components;
    Count exchange_correction_kernel_terms_per_call=0,provider_kernel_terms_per_call=0;
    BoundedPeriodicRHFMemoryPlan solver;
    Count spins=0,n_alpha=0,n_beta=0,n_basis=0,momenta=0,electrons=0;
    Count driver_live_bytes=0,provider_workspace_bytes=0,modeled_peak_bytes=0;
    Count fock_calls_upper_bound=0,shell_calls=0,primitive_pairs=0,primitive_points=0;
    Count primitive_work=0,scalar_terms=0,kernel_terms=0;
    Count nuclear_image_candidates_upper_bound=0,history_capacity=0,history_live_bytes=0;
    Count domain_live_bytes=0,domain_scalar_terms=0,domain_kernel_terms=0;
    std::optional<PeriodicQuadraturePlan> quadrature;
    Count quadrature_live_bytes=0;
};
struct PeriodicKSReceipt {
    std::string provider="manta",algorithm="finite-3d-semilocal-ks";
    std::string boundary="neutral-3d-tinfoil-zero-average;pure-semilocal";
    std::string source_identity,overlap_identity,kinetic_identity,nuclear_identity,nuclear_energy_identity;
    std::string final_density_identity,final_fock_identity;
    std::string domain_selection="explicit-vectors",domain_identity;
    double reciprocal_energy_cutoff=0.,xc_image_cutoff_bohr=0.;
    Count domain_translations=0,domain_hartree_vectors=0,domain_xc_images=0,domain_conjugacy_vectors=0,domain_exchange_vectors=0;
    std::optional<PeriodicQuadratureReceipt> quadrature;
    int grid_radial_points=0,grid_lebedev_order=0;
    double grid_partition_radius_bohr=0.;
    bool grid_stratmann=false;
    PeriodicHartreeReceipt hartree;
    std::string coulomb_algorithm="exact",exchange_algorithm="none",coulomb_identity,exchange_identity;
    std::optional<PeriodicRIFockInfo> fitting_info;
    std::optional<PeriodicRIFockReceipt> fitted_fock;
    PeriodicSemilocalXCReceipt xc;
    std::optional<PeriodicExchangeReceipt> exchange;
    std::string exchange_q0="not-applicable",exchange_correction_identity;
    double exact_exchange_fraction=0.,madelung=0.;
    bool range_separated=false;
    double long_range_exchange_fraction=0.,short_range_exchange_fraction=0.;
    std::vector<PeriodicKSExchangeComponentReceipt> exchange_components;
    std::string exchange_q0_requested="not-applicable";
    Count madelung_real_vectors=0,madelung_reciprocal_vectors=0;
    // XC is the semilocal energy. Exchange includes its separately recorded
    // zero-mode correction; all three are evaluated on the exported density.
    double uncorrected_exchange_energy=0.,exchange_correction_energy=0.,exchange_energy=0.;
    bool confirmed=false;
    double nuclear_real=0.,nuclear_reciprocal=0.,nuclear_self=0.,nuclear_background=0.,nuclear_energy=0.;
    double one_electron_energy=0.,coulomb_energy=0.,xc_energy=0.,energy=0.;
    double confirmation_fock_error=0.,confirmation_energy_error=0.;
    // Quadrature populations need not equal metric populations at finite support.
    double maximum_grid_population_error=0.;
    Count nuclear_pair_images=0,fock_calls=0,shell_calls=0,primitive_pairs=0,primitive_points=0,modeled_peak_bytes=0;
};
struct PeriodicKSResult {
    PeriodicKSPlan plan;
    PeriodicKSReceipt receipt;
    std::optional<BoundedPeriodicRHFResult> rks;
    std::optional<BoundedPeriodicUHFResult> uks;
    std::vector<std::complex<double>> overlap,hcore;
    std::vector<BoundedPeriodicRHFSnapshot> rks_history;
    std::vector<BoundedPeriodicUHFSnapshot> uks_history;
    bool converged() const { return receipt.confirmed && (rks ? rks->converged() : uks && uks->converged()); }
};
// Neutral all-electron 3D, uniform full RegularKMesh, fixed occupations with
// one virtual per spin. Functional spin=1 selects singlet RKS; spin=2 UKS.
// Hartree pair translations and XC canonical AO images are DISTINCT finite
// sources. Supplied per-cell quadrature and both supports require refinement.
// No exchange/Madelung, hybrid, nonlocal, meta-GGA or ECP component is omitted.
PeriodicKSPlan plan_periodic_ks(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    Count reciprocal_vectors,Count cells,Count xc_images,Count grid_points,const PeriodicKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ks(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& translations,const Eigen::MatrixXd& xc_images,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,const PeriodicKSOptions&,const ResourceBudget&);
// Complete native global LDA/GGA hybrid, with explicit finite Coulomb exchange
// Q supports and their target/source offsets. Full meshes only. Range-separated,
// meta-GGA, double-hybrid and nonlocal components remain refused.
PeriodicKSPlan plan_periodic_global_hybrid_ks(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,Count xc_images,Count grid_points,
    const PeriodicGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_global_hybrid_ks(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& transfer_offsets,
    const md::Matrix& translations,const Eigen::MatrixXd& xc_images,const Eigen::MatrixXd& points,
    const std::vector<double>& quadrature_weights,const PeriodicGlobalHybridKSOptions&,const ResourceBudget&);
// Complete native range-separated LDA/GGA KS. Input Q supports omit Q=0;
// one exact zero per diagonal transfer is added for every finite erfc kernel.
// All component kernels share the caller's nonzero Q supports. No range is
// replaced by another, and screened-only exchange has no Madelung correction.
PeriodicKSPlan plan_periodic_range_separated_ks(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,Count xc_images,Count grid_points,
    const PeriodicRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_range_separated_ks(const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& transfer_offsets,
    const md::Matrix& translations,const Eigen::MatrixXd& xc_images,const Eigen::MatrixXd& points,
    const std::vector<double>& quadrature_weights,const PeriodicRangeSeparatedKSOptions&,const ResourceBudget&);
// Fixed-projector RI J with complete pure LDA/GGA XC. Physical AO-pair and
// fitting pair cutoffs must agree; canonical XC images remain independent.
PeriodicKSPlan plan_periodic_ri_ks(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    Count reciprocal_vectors,Count cells,Count xc_images,Count grid_points,const PeriodicRIKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_ks(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& translations,const Eigen::MatrixXd& xc_images,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,const PeriodicRIKSOptions&,const ResourceBudget&);
// Independent exact/RI J/K with at least one RI target. RI K uses
// G+k_source-k_target with empty diagonal segments; exact K retains its
// existing G+k_target-k_source domain. Method fractions/corrections apply once.
PeriodicKSPlan plan_periodic_ri_global_hybrid_ks(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,Count xc_images,Count grid_points,
    const PeriodicRIGlobalHybridKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_global_hybrid_ks(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& transfer_offsets,
    const md::Matrix& translations,const Eigen::MatrixXd& xc_images,const Eigen::MatrixXd& points,
    const std::vector<double>& quadrature_weights,const PeriodicRIGlobalHybridKSOptions&,const ResourceBudget&);
// Complete finite range KS with independently selected exact/RI J and K.
// Uses the same canonical Coulomb + erfc decomposition as direct range KS;
// distinct physical ranges retain separate fixed projectors. An erf term is
// resolved before fitting, not silently fitted in a different metric.
PeriodicKSPlan plan_periodic_ri_range_separated_ks(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    Count reciprocal_vectors,Count exchange_vectors_per_pair,Count cells,Count xc_images,Count grid_points,
    const PeriodicRIRangeSeparatedKSOptions&,const ResourceBudget&);
PeriodicKSResult run_periodic_ri_range_separated_ks(const BasisSet&,const BasisSet&,const PeriodicSystem&,const RegularKMesh&,const Functional&,
    const md::Matrix& reciprocal_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& transfer_offsets,
    const md::Matrix& translations,const Eigen::MatrixXd& xc_images,const Eigen::MatrixXd& points,
    const std::vector<double>& quadrature_weights,const PeriodicRIRangeSeparatedKSOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
