#pragma once
#include "vibeqc/manta.hpp"
#include "vibeqc/manta_ri_gradient.hpp"
#include <optional>

namespace vibeqc::manta {
struct HFGradientPlan {
    Count n_atoms=0,n_basis=0,shell_calls=0,primitive_evaluations_upper_bound=0;
    Count scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct HFGradientReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",algorithm="ordered-hf-lagrangian";
    std::string source_identity,density_identity,normalization="libint-shell-axial-v1";
    Count shell_calls=0,primitive_integrals_evaluated=0,modeled_peak_bytes=0;
};
struct HFGradientResult {
    Eigen::MatrixXd values,nuclear,one_electron,two_electron,pulay;
    HFGradientReceipt receipt;
};
HFGradientPlan plan_hf_lagrangian_gradient(const Molecule&,const BasisSet&,const ResourceBudget&);
// Derivative of the all-electron molecular HF Lagrangian at fixed D_alpha,
// D_beta and total energy-weighted density W. The caller owns stationarity.
HFGradientResult compute_hf_lagrangian_gradient(const Molecule&,const BasisSet&,
    const Eigen::MatrixXd& density_alpha,const Eigen::MatrixXd& density_beta,
    const Eigen::MatrixXd& energy_weighted_density,const ResourceBudget&);
struct RIHFGradientResult : HFGradientResult {
    Eigen::MatrixXd coulomb,exchange;
    std::optional<RIJGradientReceipt> coulomb_ri;
    std::optional<RIKGradientReceipt> exchange_ri;
    std::optional<RIMetricResponseInfo> coulomb_metric,exchange_metric;
};
HFGradientPlan plan_hf_ri_lagrangian_gradient(const Molecule&,const BasisSet&,const BasisSet& auxiliary,
    const RIOptions&,bool fit_j,bool fit_k,const ResourceBudget&);
RIHFGradientResult compute_hf_ri_lagrangian_gradient(const Molecule&,const BasisSet&,const BasisSet& auxiliary,
    const Eigen::MatrixXd& density_alpha,const Eigen::MatrixXd& density_beta,const Eigen::MatrixXd& energy_weighted_density,
    const RIOptions&,bool fit_j,bool fit_k,const ResourceBudget&);
} // namespace vibeqc::manta
