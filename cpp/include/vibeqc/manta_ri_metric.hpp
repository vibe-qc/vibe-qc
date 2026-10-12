#pragma once
#include "vibeqc/manta_ri.hpp"

namespace vibeqc::manta {
namespace ri_detail {
void validate_options(const RIOptions&);
struct Spectrum {
    Eigen::VectorXd values;
    Eigen::MatrixXd vectors, factor;
    RIMetricInfo info;
};
Spectrum decompose(const Eigen::MatrixXd&,const RIOptions&);
std::string matrix_identity(const Eigen::MatrixXd&);
std::string source_identity(const std::string& metric_source,const std::string& coupling_source,
                            const std::string& factor_identity,const RIOptions&);
} // namespace ri_detail

struct RIMetricResponsePlan {
    Count dimension=0, scalar_terms=0, kernel_terms=0, modeled_peak_bytes=0;
};
struct RIMetricResponseInfo {
    Count rank=0;
    double cutoff=0., cutoff_distance=0., spectral_guard=0., rank_stability_radius=0.;
    // Zero when one of the two spaces is empty; has_split distinguishes this.
    double retained_discarded_gap=0.;
    bool has_split=false;
    std::string source_identity, factor_identity;
};
struct RIMetricDerivative {
    Eigen::MatrixXd inverse, projector;
    std::string source_identity, direction_identity;
    Count modeled_peak_bytes=0;
};
RIMetricResponsePlan plan_ri_metric_response(Count dimension,const RIOptions&,const ResourceBudget&);

// Dense reference response of the *same* positive-eigenprojector metric as RI.
// Real symmetric directions and seeds use the Frobenius inner product. The
// lower triangle defines the numerical source; near-symmetry is never averaged.
class RIMetricResponse {
public:
    RIMetricResponse(const Eigen::MatrixXd&,const RIOptions&,const ResourceBudget&);
    const RIMetricResponsePlan& plan() const { return plan_; }
    const RIMetricResponseInfo& info() const { return info_; }
    const Eigen::MatrixXd& inverse() const { return inverse_; }
    const Eigen::MatrixXd& projector() const { return projector_; }
    const Eigen::MatrixXd& factor() const { return spectrum_.factor; }
    RIMetricDerivative apply(const Eigen::MatrixXd&) const;
    Eigen::MatrixXd adjoint(const Eigen::MatrixXd& inverse_seed,const Eigen::MatrixXd& projector_seed) const;
private:
    Eigen::MatrixXd rotate(const Eigen::MatrixXd&) const;
    Eigen::MatrixXd restore(const Eigen::MatrixXd&) const;
    RIMetricResponsePlan plan_;
    RIMetricResponseInfo info_;
    ri_detail::Spectrum spectrum_;
    Eigen::MatrixXd inverse_,projector_,inverse_divided_,projector_divided_;
};
} // namespace vibeqc::manta
