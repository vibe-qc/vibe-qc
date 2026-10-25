#pragma once
#include "vibeqc/manta_ri.hpp"
#include <memory>

namespace vibeqc::manta {
// Enclosure for the exact stored Hermitian matrix, not integral or Fock error.
struct HermitianSpectrumPlan {
    Count dimension=0, scalar_terms=0, kernel_terms=0, modeled_peak_bytes=0;
};
struct HermitianSpectrumCertificate {
    bool spectrum_certified=false, rank_certified=false;
    Count rank=0;
    double residual_bound=0., orthogonality_bound=0., eigenvalue_error_bound=0.;
    double cutoff_lower=0., cutoff_upper=0., rank_stability_radius=0.;
};
HermitianSpectrumPlan plan_hermitian_spectrum(Count,const RIOptions&,const ResourceBudget&);
HermitianSpectrumCertificate certify_hermitian_spectrum(const Eigen::MatrixXcd&,
    const Eigen::VectorXd&,const Eigen::MatrixXcd&,const RIOptions&,const ResourceBudget&);
// Bound ||F F^* - f(A)||_2 for the actual stored factor and exact stored A.
// f retains positive eigenvalues above the certified adaptive cutoff.
struct HermitianInverseCertificate {
    HermitianSpectrumCertificate spectrum;
    bool qualified=false;
    double retained_lower=0., inverse_norm_bound=0., divided_difference_bound=0.;
    double residual_frobenius_bound=0., residual_term=0., orthogonality_term=0.;
    double factor_product_bound=0., inverse_error_bound=0.;
};
HermitianSpectrumPlan plan_hermitian_inverse(Count,const RIOptions&,const ResourceBudget&);
HermitianInverseCertificate certify_hermitian_inverse(const Eigen::MatrixXcd&,
    const Eigen::VectorXd&,const Eigen::MatrixXcd&,const Eigen::MatrixXcd&,
    const RIOptions&,const ResourceBudget&);
// Per-column finite residual certificate for the *same stored* coupling map.
// Columns retain only scalar envelopes, never an auxiliary/AO-pair tensor.
struct HermitianInverseColumnBound {
    bool qualified=false;
    double coupling_norm=0.,factor_score=0.,residual_norm=0.,discarded_norm=0.;
private:
    std::shared_ptr<const void> owner;
    friend class HermitianInverseAction;
};
struct HermitianInverseFamilyBound {
    bool qualified=false;
    double j_error_bound=0.,energy_error_bound=0.;
    std::vector<double> j_error_bounds;
};
// Executed paired metric operations relative to exact arithmetic on stored F,x.
// Separate from inverse approximation, coupling-action and input errors.
struct PairedMetricRoundoff {
    bool qualified=false,input_error_included=false;
    double projection_bound=0.,coefficient_bound=0.,energy_bound=0.;
    std::vector<double> projection_bounds,coefficient_bounds;
};
HermitianSpectrumPlan plan_paired_metric_roundoff(Count dimension,Count rank,const ResourceBudget&);
PairedMetricRoundoff certify_paired_metric_roundoff(const Eigen::MatrixXcd& factor,
    const Eigen::VectorXcd& rhs,const Eigen::VectorXcd& projected,
    const Eigen::VectorXcd& coefficients,double energy,const ResourceBudget&);
// Same fixed graph with per-RHS complex L1 input uncertainty. The returned
// errors include both propagated input error and executed metric arithmetic.
HermitianSpectrumPlan plan_paired_metric_action(Count dimension,Count rank,const ResourceBudget&);
PairedMetricRoundoff certify_paired_metric_action(const Eigen::MatrixXcd& factor,
    const Eigen::VectorXcd& rhs,const Eigen::VectorXcd& projected,
    const Eigen::VectorXcd& coefficients,double energy,const std::vector<double>& rhs_errors,const ResourceBudget&);
HermitianSpectrumPlan plan_hermitian_inverse_action(Count,const RIOptions&,const ResourceBudget&);
HermitianSpectrumPlan plan_hermitian_inverse_column(Count,const ResourceBudget&);
HermitianSpectrumPlan plan_hermitian_inverse_family(Count,Count,const ResourceBudget&);
class PeriodicRIQBuilder;
class HermitianInverseAction {
public:
    HermitianInverseAction(const Eigen::MatrixXcd&,const Eigen::VectorXd&,const Eigen::MatrixXcd&,
        const Eigen::MatrixXcd&,const RIOptions&,const ResourceBudget&);
    Count dimension() const;
    const HermitianInverseCertificate& certificate() const;
    // Caller-supplied upper bounds on Euclidean coupling-column norms.
    HermitianInverseFamilyBound norm_family(const std::vector<double>&,
        double,const std::vector<double>&,const ResourceBudget&) const;
    HermitianInverseColumnBound column(const Eigen::VectorXcd&,const ResourceBudget&) const;
    HermitianInverseFamilyBound family(const std::vector<HermitianInverseColumnBound>&,
        double density_l1_limit,const std::vector<double>& density_entry_limits,const ResourceBudget&) const;
private:
    struct Data;
    std::shared_ptr<const Data> data_;
    HermitianInverseAction(const Eigen::MatrixXcd&,const Eigen::MatrixXcd&,
        const HermitianInverseCertificate&,const RIOptions&,const ResourceBudget&);
    friend class PeriodicRIQBuilder;
};
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
