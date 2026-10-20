#pragma once
#include "vibeqc/manta_fourier.hpp"
#include <optional>

namespace vibeqc::manta {
struct BlochFourierPlan {
    std::vector<Count> shape;
    Count cells=0, candidate_shell_pairs=0, shell_calls=0, primitive_pairs=0;
    Count vector_primitive_products=0, phase_products=0, phase_accumulations=0;
    Count scalar_terms=0, kernel_terms=0, output_bytes=0, modeled_peak_bytes=0;
};
struct BlochFourierReceipt {
    std::string provider="manta",provider_version="md-fp64-v1";
    std::string normalization="libint-shell-axial-v1",convention="exp(+i*k.R)*exp(-i*q.r)";
    std::string source_identity;
    Count candidate_shell_pairs=0, skipped_shell_pairs=0, shell_calls=0, primitive_pairs=0;
    Count vector_primitive_products=0, phase_products=0, phase_accumulations=0;
    Count integral_elements=0, modeled_peak_bytes=0;
};
struct BlochFourierResult {
    std::vector<Count> shape;
    std::vector<std::complex<double>> values;
    BlochFourierReceipt receipt;
};
// Sun et al., JCP 147,164119 (2017), doi:10.1063/1.4998644, Eq.16.
// Explicit finite translations only. q is the complete Fourier argument
// (e.g. G+k_ket-k_bra); k is the ket momentum. No reciprocal/lattice domain,
// Coulomb q=0/background, cell weight or periodic convergence is chosen here.
// Shape (k,mu,nu,q). One shell/cell Fourier panel is folded into all k rows.
// An optional cutoff retains exactly |O_mu-O_nu-R|<=cutoff, including zero.
BlochFourierPlan plan_bloch_fourier(const BasisSet&,Count vectors,Count cells,Count momenta,
                                  std::optional<double> pair_cutoff,const ResourceBudget&);
BlochFourierResult compute_bloch_fourier_panel(const BasisSet&,const md::Matrix& vectors,
    const md::Matrix& translations,const md::Matrix& momenta,std::optional<double> pair_cutoff,const ResourceBudget&);
} // namespace vibeqc::manta
