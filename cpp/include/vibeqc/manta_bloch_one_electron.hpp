#pragma once
#include "vibeqc/manta_one_electron.hpp"
#include <complex>
#include <optional>

namespace vibeqc::manta {
struct BlochOneElectronPlan {
    std::vector<Count> shape, cell_shape;
    Count candidate_shell_pairs=0, shell_calls=0, primitive_pairs=0;
    Count phase_products=0, phase_accumulations=0, integral_elements=0;
    Count scalar_terms=0, kernel_terms=0, output_bytes=0, modeled_peak_bytes=0;
};
struct BlochOneElectronReceipt {
    std::string provider="manta",provider_version="md-fp64-v1";
    std::string normalization="libint-shell-axial-v1",convention="exp(+i*k.R)";
    std::string operation,source_identity;
    Count candidate_shell_pairs=0, skipped_shell_pairs=0, shell_calls=0, primitive_pairs=0;
    Count phase_products=0, phase_accumulations=0, integral_elements=0, modeled_peak_bytes=0;
    double hermiticity_residual=0.;
};
struct BlochOneElectronResult {
    BlochOneElectronPlan plan;
    BlochOneElectronReceipt receipt;
    std::vector<std::complex<double>> values;
    std::vector<double> cell_values;
};
// Explicit finite overlap/kinetic matrices in caller order. S/T_R=<mu_0|S/T|nu_R>;
// S/T(k)=sum_R exp(+i*k.R) S/T_R. No support generation or matrix repair.
// Cell blocks are retained even for an empty momentum list, so the same
// producer can supply real-space periodic operator and symmetry consumers.
BlochOneElectronPlan plan_bloch_one_electron(const BasisSet&,const std::string& operation,
    Count cells,Count momenta,std::optional<double> pair_cutoff,const ResourceBudget&);
BlochOneElectronResult compute_bloch_one_electron(const BasisSet&,const std::string& operation,
    const md::Matrix& translations,const md::Matrix& momenta,
    std::optional<double> pair_cutoff,const ResourceBudget&);
} // namespace vibeqc::manta
