#pragma once
#include "vibeqc/manta_ri_metric.hpp"

namespace vibeqc::manta {
struct RIJGradientPlan {
    Count n_atoms=0,n_basis=0,n_auxiliary=0,shell_calls=0;
    Count primitive_evaluations_upper_bound=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct RIJGradientReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",algorithm="streamed-ri-coulomb-gradient";
    std::string source_identity,ri_source_identity,density_identity,kernel;
    double omega=0.,coulomb_energy=0.,metric_rhs_norm=0.,metric_residual_norm=0.;
    bool metric_residual_evaluated=false;
    Count metric_shell_calls=0,coupling_shell_calls=0,metric_derivative_shell_calls=0,coupling_derivative_shell_calls=0;
    Count primitive_integrals_evaluated=0,integral_elements_consumed=0,modeled_peak_bytes=0;
};
struct RIJGradientResult {
    Eigen::MatrixXd values,coupling,metric;
    RIMetricResponseInfo metric_response;
    RIJGradientReceipt receipt;
};
RIJGradientPlan plan_ri_coulomb_gradient(const Molecule&,const BasisSet& orbital,const BasisSet& auxiliary,
    const RIOptions&,const KernelRequest&,const ResourceBudget&);
// d[0.5 (B:D)^T V^+ (B:D)] at fixed real density D. This is a Coulomb
// energy component, not a complete stationary SCF force or Lagrangian.
RIJGradientResult compute_ri_coulomb_gradient(const Molecule&,const BasisSet& orbital,const BasisSet& auxiliary,
    const Eigen::MatrixXd& density,const RIOptions&,const KernelRequest&,const ResourceBudget&);

struct RIKGradientPlan : RIJGradientPlan {
    Count mode_block=0,auxiliary_blocks=0,maximum_coupling_passes=0,maximum_derivative_passes=0;
};
struct RIKGradientReceipt {
    std::string provider="manta",provider_version="md-fp64-v1",algorithm="streamed-ri-exchange-gradient";
    std::string source_identity,ri_source_identity,density_identity,kernel;
    std::array<std::string,2> density_identities;
    double omega=0.,exchange_energy=0.;
    Count metric_shell_calls=0,coupling_shell_calls=0,metric_derivative_shell_calls=0,coupling_derivative_shell_calls=0;
    Count coupling_passes=0,coupling_derivative_passes=0;
    Count primitive_integrals_evaluated=0,integral_elements_consumed=0,modeled_peak_bytes=0;
};
struct RIKGradientResult {
    Eigen::MatrixXd values,coupling,metric;
    RIMetricResponseInfo metric_response;
    RIKGradientReceipt receipt;
};
RIKGradientPlan plan_ri_exchange_gradient(const Molecule&,const BasisSet& orbital,const BasisSet& auxiliary,
    const RIOptions&,const KernelRequest&,const ResourceBudget&);
// d[-0.5 sum_spin Tr(D_spin K_RI[D_spin])] at fixed per-spin densities.
RIKGradientResult compute_ri_exchange_gradient(const Molecule&,const BasisSet& orbital,const BasisSet& auxiliary,
    const Eigen::MatrixXd& density_alpha,const Eigen::MatrixXd& density_beta,
    const RIOptions&,const KernelRequest&,const ResourceBudget&);
} // namespace vibeqc::manta
