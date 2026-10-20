#pragma once
#include "vibeqc/manta.hpp"
#include <complex>

namespace vibeqc::manta {
struct PeriodicLocalExchangePlan {
    Count n_basis=0,spins=0,momenta=0,images=0,grid_points=0,grid_batch=0;
    Count shell_calls=0,primitive_points=0,scalar_terms=0,kernel_terms=0;
    Count panel_bytes=0,input_bytes=0,output_bytes=0,modeled_peak_bytes=0;
};
struct PeriodicLocalExchangeReceipt {
    std::string algorithm="finite-bloch-dirac-exchange",functional="lda_x";
    std::string source_identity,density_identity,snapshot_identity;
    std::string convention="rho=sum_k w_k B_k D_k B_k^dagger; V=int B_k^dagger v B_k";
    std::string support="canonical-shell-images; explicit-per-cell-quadrature; no-tail-bound";
    double energy=0.,density_hermiticity_residual=0.,minimum_density=0.;
    std::vector<double> electrons_grid;
    Count panel_calls=0,shell_calls=0,primitive_points=0,rounded_negative_points=0,modeled_peak_bytes=0;
};
struct PeriodicLocalExchangeResult {
    PeriodicLocalExchangePlan plan;
    PeriodicLocalExchangeReceipt receipt;
    std::vector<std::complex<double>> values;
};
// Explicit finite AO-image and per-cell quadrature source, not an SCF driver.
// Lattice vectors are columns. Image rows are Cartesian lattice translations.
// Each shell's home image is reduced to fractional [0,1); its Bloch phase is
// transported back to the supplied shell gauge. Both AO indices use that same
// support. D and V are row-major [spin,k,AO,AO]. Spin=1 means total restricted
// density; spin=2 means separate alpha/beta one-particle densities.
// dE=sum_spin,k weights[k]*Re Tr(dD[k] V[k]); V contains no k weight.
// Inputs are borrowed for the call and must not be mutated concurrently.
PeriodicLocalExchangePlan plan_periodic_local_exchange(const BasisSet&,Count images,
    Count momenta,Count grid_points,Count spins,Count grid_batch,const ResourceBudget&);
PeriodicLocalExchangeResult compute_periodic_local_exchange(const BasisSet&,const Eigen::MatrixXd& lattice,
    const Eigen::MatrixXd& images,const Eigen::MatrixXd& momenta,const std::vector<double>& weights,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,
    const std::vector<std::complex<double>>& densities,Count spins,Count grid_batch,const ResourceBudget&);
} // namespace vibeqc::manta
