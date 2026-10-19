#pragma once
#include "vibeqc/manta_bloch_fourier.hpp"

namespace vibeqc::manta {
struct PeriodicHartreePlan {
    Count n_basis=0, vectors=0, cells=0, momenta=0, reciprocal_batch=0;
    Count batches=0, panel_calls=0, shell_calls=0, primitive_pairs=0;
    Count vector_primitive_products=0, contraction_terms=0, scalar_terms=0, kernel_terms=0;
    Count output_bytes=0, driver_live_bytes=0, modeled_peak_bytes=0;
};
struct PeriodicHartreeReceipt {
    std::string provider="manta", provider_version="md-fp64-v1";
    std::string normalization="libint-shell-axial-v1", algorithm="finite-reciprocal-hartree";
    std::string boundary="3d-zero-average-g0-omitted", source_identity, density_identity;
    Count panel_calls=0, shell_calls=0, skipped_shell_pairs=0, primitive_pairs=0;
    Count vector_primitive_products=0, phase_products=0, phase_accumulations=0;
    Count contraction_terms=0, modeled_peak_bytes=0;
    double energy=0, density_hermiticity_residual=0, cell_imaginary_residual=0;
    double response_hermiticity_residual=0, cell_volume=0;
};
struct PeriodicHartreeResult {
    PeriodicHartreePlan plan;
    PeriodicHartreeReceipt receipt;
    std::vector<std::complex<double>> cells, momenta;
};
// Sun et al., JCP147,164119 (2017), doi:10.1063/1.4998644, Eqs.16-17.
// Explicit finite image and reciprocal supports, each unique and inversion
// closed. Lattice vectors are columns. G=0 must be absent. There is no nuclear
// term, exchange, surface correction or claim of infinite-cutoff convergence.
// rho(G)=sum_k w_k sum_mn D_mn(k) F_mn(G;-k);
// J_R=1/V sum_G 4*pi/G^2 rho(G) conj(F(G;R)); J_k=sum_R exp(ik.R) J_R.
// Input D is row-major complex (k,mu,nu), weights sum to one. No density
// positivity or time-reversal assumption, and no Hermitization/real projection.
PeriodicHartreePlan plan_periodic_hartree(const BasisSet&,Count vectors,Count cells,Count momenta,
    Count reciprocal_batch,std::optional<double> pair_cutoff,const ResourceBudget&);
PeriodicHartreeResult compute_periodic_hartree(const BasisSet&,const md::Matrix& lattice,
    const md::Matrix& vectors,const md::Matrix& translations,const md::Matrix& momenta,
    const std::vector<double>& weights,const std::vector<std::complex<double>>& densities,
    Count reciprocal_batch,std::optional<double> pair_cutoff,const ResourceBudget&);
} // namespace vibeqc::manta
