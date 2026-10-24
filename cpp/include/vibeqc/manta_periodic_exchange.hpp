#pragma once
#include "vibeqc/manta_bloch_fourier.hpp"

namespace vibeqc::manta {
struct PeriodicExchangePlan {
    bool roundoff_requested=false;
    Count n_basis=0,momenta=0,densities=0,cells=0,vectors_per_pair=0,transfer_pairs=0,reciprocal_batch=0;
    Count shell_calls=0,primitive_pairs=0,vector_primitive_products=0;
    Count phase_products=0,phase_accumulations=0,matrix_products=0,contraction_terms=0;
    Count scalar_terms=0,kernel_terms=0,output_bytes=0,driver_live_bytes=0,modeled_peak_bytes=0;
};
struct PeriodicExchangeReceipt {
    bool roundoff_evaluated=false;
    std::string roundoff_scope="stored-bloch-panels-and-reciprocal-factors";
    double roundoff_bound=0.;
    std::vector<double> energy_roundoff_bounds;
    std::string provider="manta",provider_version="md-fp64-v1",normalization="libint-shell-axial-v1";
    std::string algorithm="3d-finite-reciprocal-exchange",boundary="full-coulomb-q0-omitted-no-madelung";
    std::string convention="Q=G+k_ket-k_bra;B=F(Q,k_ket);K=sum(w_bra*B^dagger*D_bra*B)";
    std::string kernel="coulomb";
    double omega=0.;
    std::string source_identity,density_identity;
    std::vector<std::string> density_identities;
    std::vector<double> energies;
    Count transfer_vectors=0,zero_vectors=0,panel_calls=0,shell_calls=0,skipped_shell_pairs=0,primitive_pairs=0;
    Count vector_primitive_products=0,phase_products=0,phase_accumulations=0,matrix_products=0;
    Count contraction_terms=0,modeled_peak_bytes=0;
    double cell_volume=0.,density_hermiticity_residual=0.,response_hermiticity_residual=0.;
};
struct PeriodicExchangeResult {
    PeriodicExchangePlan plan;
    PeriodicExchangeReceipt receipt;
    std::vector<std::complex<double>> values;
    std::vector<double> roundoff_bounds; // Per-entry complex L1, empty when unaudited.
};
// Each target/source momentum pair supplies its complete ordered Q=G+k_t-k_s
// list via offsets[t*nk+s:t*nk+s+2]. Reverse pair supports must contain the
// opposite reciprocal labels; diagonal supports are inversion closed.
// Coulomb/erf Q=0 is omitted explicitly. Positive-range erfc accepts its
// finite Q=0 coefficient pi/omega^2 only when supplied in the explicit domain.
// No singularity correction or
// reciprocal enumeration/rounding is inferred. D shape (density,k,AO,AO).
PeriodicExchangePlan plan_periodic_exchange(const BasisSet&,Count vectors_per_pair,Count cells,
    Count momenta,Count densities,Count reciprocal_batch,std::optional<double> pair_cutoff,const ResourceBudget&,
    const std::string& kernel="coulomb",double omega=0.,bool audit_roundoff=false);
PeriodicExchangeResult compute_periodic_exchange(const BasisSet&,const md::Matrix& lattice,
    const md::Matrix& vectors,const std::vector<Count>& offsets,const md::Matrix& translations,
    const md::Matrix& momenta,const std::vector<double>& weights,const std::vector<std::complex<double>>& densities,
    Count density_count,Count reciprocal_batch,std::optional<double> pair_cutoff,const ResourceBudget&,
    const std::string& kernel="coulomb",double omega=0.,bool audit_roundoff=false);
} // namespace vibeqc::manta
