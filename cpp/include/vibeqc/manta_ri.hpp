#pragma once

#include "vibeqc/manta_fitting.hpp"

namespace vibeqc::manta {
struct RIOptions {
    double absolute_cutoff=1e-12, relative_cutoff=1e-10;
    double negative_relative_tolerance=1e-10;
    Count mode_block=8;
};
struct RIPlan {
    Count n_basis=0, n_auxiliary=0, maximum_rank=0, mode_block=0;
    Count metric_shell_calls=0, coupling_shell_calls_per_pass=0;
    Count maximum_coupling_passes=0, shell_calls=0, primitive_products=0;
    Count scalar_terms=0, kernel_terms=0, modeled_peak_bytes=0;
};
struct RIMetricInfo {
    Count rank=0, shell_calls=0, primitive_products=0;
    double cutoff=0., minimum_eigenvalue=0., maximum_eigenvalue=0.;
    std::string factor_identity;
};
struct RIReceipt {
    std::string provider="manta", provider_version="md-fp64-v1";
    std::string algorithm="streamed-ri", normalization="libint-shell-axial-v1";
    std::string kernel, source_identity, density_identity;
    double omega=0., coulomb_energy=0., metric_rhs_norm=0., metric_residual_norm=0.;
    bool metric_residual_evaluated=false;
    Count retained_rank=0, coupling_passes=0, shell_calls=0;
    Count primitive_products=0, integral_elements_consumed=0, modeled_peak_bytes=0;
};
struct RIResult {
    Eigen::MatrixXd J,K;
    RIReceipt receipt;
};
struct RIExchangePair {
    Eigen::MatrixXd K_alpha,K_beta;
    std::array<std::string,2> density_identities;
    RIReceipt receipt;
};
RIPlan plan_ri(const BasisSet& orbital,const BasisSet& auxiliary,const RIOptions&,
               const KernelRequest&,const ResourceBudget&);

// Small-system dense-metric reference, with bounded mode blocks and streamed
// three-center passes. It never stores the complete auxiliary/AO-pair tensor.
class RIJKBuilder final : public JKBuilder {
public:
    RIJKBuilder(const BasisSet& orbital,const BasisSet& auxiliary,const RIOptions&,
                const KernelRequest&,const ResourceBudget&);
    const RIPlan& plan() const { return plan_; }
    const RIMetricInfo& metric_info() const { return info_; }
    const std::string& source_identity() const { return source_; }
    const Eigen::MatrixXd& metric() const { return metric_; }
    const Eigen::MatrixXd& factor() const { return factor_; }
    RIResult build(const Eigen::MatrixXd&,bool make_j=true,bool make_k=true) const;
    Eigen::MatrixXd build_J(const Eigen::MatrixXd& d) const override { return build(d,true,false).J; }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd& d) const override { return build(d,false,true).K; }
    std::pair<Eigen::MatrixXd,Eigen::MatrixXd> build_K_pair(const Eigen::MatrixXd&,const Eigen::MatrixXd&) const override;
    RIExchangePair build_exchange_pair(const Eigen::MatrixXd&,const Eigen::MatrixXd&) const;
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd&,double alpha_hf=1.) const override;
private:
    std::pair<RIResult,Eigen::MatrixXd> build_impl(const Eigen::MatrixXd&,const Eigen::MatrixXd*,bool make_j,bool make_k) const;
    RIPlan plan_;
    KernelRequest request_;
    FittingTiles coupling_;
    Eigen::MatrixXd metric_,factor_;
    RIMetricInfo info_;
    std::string source_;
};
} // namespace vibeqc::manta
