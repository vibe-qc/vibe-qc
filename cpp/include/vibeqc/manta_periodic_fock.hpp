#pragma once
#include "vibeqc/manta_periodic_hartree.hpp"
#include "vibeqc/manta_periodic_exchange.hpp"

namespace vibeqc::manta {
struct PeriodicFockPlan {
    PeriodicHartreePlan hartree;
    PeriodicExchangePlan exchange;
    Count n_basis=0,momenta=0,driver_live_bytes=0,modeled_peak_bytes=0;
    Count shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicFockReceipt {
    std::string provider="manta",algorithm="finite-periodic-spin-fock";
    std::string convention="F_sigma=H+J[D_alpha+D_beta]-scale*K[D_sigma]";
    std::string source_identity,density_identity,one_electron_identity,snapshot_identity;
    double exchange_scale=1.,one_electron_energy=0.,coulomb_energy=0.,exchange_energy=0.,electronic_energy=0.;
    double hcore_hermiticity_residual=0.,fock_hermiticity_residual=0.;
    Count modeled_peak_bytes=0;
};
struct PeriodicFockResult {
    PeriodicFockPlan plan;
    PeriodicFockReceipt receipt;
    PeriodicHartreeResult hartree;
    PeriodicExchangeResult exchange;
    std::vector<std::complex<double>> values;
};
// Immutable evaluated spin-density epoch. H[k,AO,AO] is supplied explicitly;
// this consumer does not claim to have generated a nuclear/one-electron source.
// D[spin=2,k,AO,AO] contains occupations for each spin, with no extra factor.
// Restricted callers use D_alpha=D_beta=D_total/2. Q and G are separate
// explicit finite sources. There is no XC, Madelung or nuclear energy here.
PeriodicFockPlan plan_periodic_fock(const BasisSet&,Count hartree_vectors,Count exchange_vectors_per_pair,
    Count cells,Count momenta,Count reciprocal_batch,std::optional<double> pair_cutoff,const ResourceBudget&,
    const std::string& kernel="coulomb",double omega=0.,double exchange_scale=1.);
PeriodicFockResult compute_periodic_fock(const BasisSet&,const md::Matrix& lattice,
    const md::Matrix& hartree_vectors,const md::Matrix& exchange_vectors,const std::vector<Count>& offsets,
    const md::Matrix& translations,const md::Matrix& momenta,const std::vector<double>& weights,
    const std::vector<std::complex<double>>& hcore,const std::vector<std::complex<double>>& spin_densities,
    Count reciprocal_batch,std::optional<double> pair_cutoff,const ResourceBudget&,
    const std::string& kernel="coulomb",double omega=0.,double exchange_scale=1.);
} // namespace vibeqc::manta
