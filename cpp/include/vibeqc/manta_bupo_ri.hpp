#pragma once
#include "vibeqc/manta_bupo_fitting.hpp"
#include "vibeqc/manta_ri.hpp"

namespace vibeqc::manta {
struct BupoRIPlan {
    Count n_basis=0,n_auxiliary=0,shell_calls=0,primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct BupoRIReceipt {
    std::string source_identity,exact_ri_source_identity,density_identity;
    std::string algorithm="paired-ri-bupo-j",provider="manta",kernel="coulomb";
    Count retained_rank=0,modeled_peak_bytes=0;
    double coulomb_energy=0.,metric_rhs_norm=0.,metric_residual_norm=0.;
    double metric_inverse_norm_bound=0.,forward_error_l2_bound=0.;
    double approximation_energy_bound=0.,approximation_j_bound=0.;
    double adjoint_j_bound=0.,forward_propagated_j_bound=0.;
    BupoFittingReceipt forward,adjoint;
};
struct BupoRIResult {
    Eigen::MatrixXd J;
    Eigen::VectorXd rhs,coefficients;
    BupoRIReceipt receipt;
};
// Initial bounded dense exact metric. Both Coulomb passes use the same fixed
// approximate coupling. The retained exact RI producer also supplies an
// independently rebuilt same-basis reference, with identical metric cutoff.
class BupoRIJBuilder final : public JKBuilder {
public:
    BupoRIJBuilder(const BasisSet&,const BasisSet&,const BupoFittingOptions&,const RIOptions&,const ResourceBudget&);
    const BupoRIPlan& plan() const { return plan_; }
    const BupoFittingPlan& coupling_plan() const { return coupling_->plan(); }
    const RIMetricInfo& metric_info() const { return exact_->metric_info(); }
    const Eigen::MatrixXd& metric() const { return exact_->metric(); }
    const Eigen::MatrixXd& factor() const { return exact_->factor(); }
    const std::string& source_identity() const { return source_; }
    const std::string& exact_ri_source_identity() const { return exact_->source_identity(); }
    BupoRIResult build(const Eigen::MatrixXd&) const;
    RIResult build_exact(const Eigen::MatrixXd& density) const { return exact_->build(density,true,false); }
    Eigen::MatrixXd build_J(const Eigen::MatrixXd& density) const override { return build(density).J; }
    Eigen::MatrixXd build_K(const Eigen::MatrixXd&) const override;
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd&,double alpha=1.) const override;
private:
    BupoRIPlan plan_;
    std::shared_ptr<const BupoFittingCoupling> coupling_;
    std::shared_ptr<const RIJKBuilder> exact_;
    double inverse_bound_=0.;
    std::string source_;
};
} // namespace vibeqc::manta
