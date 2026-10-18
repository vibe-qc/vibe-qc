#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <Eigen/Eigenvalues>
#include <cstring>

namespace vibeqc::manta {
namespace {
void hash_number(vibeqc::detail::Sha256& hash,double value) {
    if(value==0.) value=0.;
    std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits));
    std::array<std::uint8_t,8> bytes{};
    for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(bits>>(56-8*k));
    hash.update(bytes.data(),bytes.size());
}
void validate_matrix(const Eigen::MatrixXd& matrix,Eigen::Index n) {
    if(n<=0 || matrix.rows()!=n || matrix.cols()!=n || !matrix.allFinite())
        throw std::invalid_argument("MANTA RI metric response requires finite nonempty square matrices of matching dimensions");
    const double scale=std::max(1.,matrix.cwiseAbs().maxCoeff());
    for(Eigen::Index i=0;i<n;++i) for(Eigen::Index j=0;j<i;++j)
        if(std::abs(matrix(i,j)-matrix(j,i))>1e-12*scale)
            throw std::invalid_argument("MANTA RI metric response requires symmetric real matrices");
}
} // namespace
namespace ri_detail {
void validate_options(const RIOptions& options) {
    for(auto value:{options.absolute_cutoff,options.relative_cutoff,options.negative_relative_tolerance}) {
        if(!std::isfinite(value) || value<0) throw std::invalid_argument("MANTA RI cutoffs must be finite and nonnegative");
    }
    if(!options.mode_block || options.mode_block>64) throw std::invalid_argument("MANTA RI mode block must be in 1..64");
}
std::string matrix_identity(const Eigen::MatrixXd& matrix) {
    vibeqc::detail::Sha256 hash;
    hash_number(hash,matrix.rows()); hash_number(hash,matrix.cols());
    for(Eigen::Index i=0;i<matrix.rows();++i) for(Eigen::Index j=0;j<matrix.cols();++j) hash_number(hash,matrix(i,j));
    return hash.finish_hex();
}
std::string source_identity(const std::string& metric_source,const std::string& coupling_source,
                            const std::string& factor_identity,const RIOptions& options) {
    const std::string payload="manta.ri.v1;positive-eigenprojector;lower-triangle;"+
        metric_source+";"+coupling_source+";"+factor_identity;
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size());
    hash_number(hash,options.absolute_cutoff); hash_number(hash,options.relative_cutoff);
    return hash.finish_hex();
}
Spectrum decompose(const Eigen::MatrixXd& metric,const RIOptions& options) {
    validate_options(options);
    validate_matrix(metric,metric.rows());
    // Retain the existing RI lower-triangle eigensolve and factor construction
    // byte for byte. No diagonal shift or absolute-eigenvalue repair.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(metric,Eigen::ComputeEigenvectors);
    if(eigen.info()!=Eigen::Success || !eigen.eigenvalues().allFinite() || !eigen.eigenvectors().allFinite())
        throw std::runtime_error("MANTA RI metric eigensolve failed");
    Spectrum result;
    auto& info=result.info;
    info.minimum_eigenvalue=eigen.eigenvalues()[0];
    info.maximum_eigenvalue=eigen.eigenvalues()[metric.rows()-1];
    const double positive_scale=std::max(0.,info.maximum_eigenvalue);
    info.cutoff=std::max(options.absolute_cutoff,options.relative_cutoff*positive_scale);
    const double negative_limit=std::max(options.absolute_cutoff,options.negative_relative_tolerance*positive_scale);
    if(!std::isfinite(info.cutoff) || !std::isfinite(negative_limit))
        throw std::overflow_error("MANTA RI metric cutoff overflow");
    if(info.minimum_eigenvalue < -negative_limit) throw std::runtime_error("MANTA RI metric has a significant negative eigenvalue");
    for(Eigen::Index i=0;i<eigen.eigenvalues().size();++i) if(eigen.eigenvalues()[i]>info.cutoff) ++info.rank;
    result.factor.resize(metric.rows(),info.rank);
    Count column=0;
    for(Eigen::Index i=0;i<eigen.eigenvalues().size();++i) if(eigen.eigenvalues()[i]>info.cutoff)
        result.factor.col(column++)=eigen.eigenvectors().col(i)/std::sqrt(eigen.eigenvalues()[i]);
    if(!result.factor.allFinite()) throw std::overflow_error("MANTA RI nonfinite metric factor");
    info.factor_identity=matrix_identity(result.factor);
    result.values=eigen.eigenvalues(); result.vectors=eigen.eigenvectors();
    return result;
}
} // namespace ri_detail

RIMetricResponsePlan plan_ri_metric_response(Count n,const RIOptions& options,const ResourceBudget& budget) {
    using namespace admission;
    ri_detail::validate_options(options); margin(budget);
    if(!n || n>static_cast<Count>(std::numeric_limits<Eigen::Index>::max()))
        throw std::invalid_argument("MANTA RI metric response requires an admissible nonzero dimension");
    RIMetricResponsePlan plan;
    plan.dimension=n;
    // Eigensolver, factor, two spectral maps and targets, two caller inputs,
    // binding copies, rotated seeds, GEMM temporaries and exported results.
    const auto bytes=mul(8,add(mul(32,square(n)),mul(64,n)));
    plan.modeled_peak_bytes=add(bytes,add(budget.backend_margin_bytes,budget.other_live_bytes));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA RI metric response memory budget exceeded");
    const auto cubic=mul(n,square(n));
    plan.scalar_terms=add(mul(32,cubic),mul(16,square(n)));
    plan.kernel_terms=add(plan.scalar_terms,mul(768,cubic));
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA RI metric response scalar-work budget exceeded");
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA RI metric response kernel-work budget exceeded");
    return plan;
}
RIMetricResponse::RIMetricResponse(const Eigen::MatrixXd& metric,const RIOptions& options,const ResourceBudget& budget)
    : plan_(plan_ri_metric_response(metric.rows(),options,budget)),spectrum_(ri_detail::decompose(metric,options)) {
    const auto n=metric.rows();
    const auto& lambda=spectrum_.values;
    info_.rank=spectrum_.info.rank; info_.cutoff=spectrum_.info.cutoff;
    info_.factor_identity=spectrum_.info.factor_identity;
    info_.cutoff_distance=(lambda.array()-info_.cutoff).abs().minCoeff();
    const double scale=std::max(lambda.cwiseAbs().maxCoeff(),info_.cutoff);
    // A numerical separation check, not an interval eigensolver certificate.
    info_.spectral_guard=64.*std::numeric_limits<double>::epsilon()*n*scale;
    if(!std::isfinite(info_.cutoff_distance) || !std::isfinite(info_.spectral_guard)
       || info_.cutoff_distance<=info_.spectral_guard)
        throw std::runtime_error("MANTA RI metric response is unresolved at the rank cutoff");
    info_.rank_stability_radius=(info_.cutoff_distance-info_.spectral_guard)/(1.+options.relative_cutoff);
    if(!(info_.rank_stability_radius>0.) || !std::isfinite(info_.rank_stability_radius))
        throw std::runtime_error("MANTA RI metric response rank stability radius is unresolved");
    info_.has_split=info_.rank>0 && info_.rank<static_cast<Count>(n);
    if(info_.has_split) {
        const auto first=n-info_.rank;
        info_.retained_discarded_gap=lambda[first]-lambda[first-1];
        if(!std::isfinite(info_.retained_discarded_gap) || info_.retained_discarded_gap<=info_.spectral_guard)
            throw std::runtime_error("MANTA RI retained/discarded spectral gap is unresolved");
    }
    inverse_=spectrum_.factor*spectrum_.factor.transpose();
    const auto retained=spectrum_.vectors.rightCols(info_.rank);
    projector_=retained*retained.transpose();
    inverse_divided_=Eigen::MatrixXd::Zero(n,n);
    projector_divided_=Eigen::MatrixXd::Zero(n,n);
    // Daleckii-Krein spectral Frechet derivative; Higham, Functions of
    // Matrices (2008), Chapter 3. Divided differences transport the retained
    // projector, including nonzero discarded eigenvalues. Equal eigenvalues
    // within a retained block use -1/(lambda_i lambda_j), never eigenvector
    // derivatives or division by a within-block gap.
    for(Eigen::Index i=0;i<n;++i) for(Eigen::Index j=0;j<n;++j) {
        const bool ki=lambda[i]>info_.cutoff,kj=lambda[j]>info_.cutoff;
        if(ki && kj) inverse_divided_(i,j)=-(1./lambda[i])*(1./lambda[j]);
        else if(ki!=kj) {
            const auto r=ki?i:j,d=ki?j:i;
            const double gap=lambda[r]-lambda[d];
            inverse_divided_(i,j)=(1./lambda[r])/gap;
            projector_divided_(i,j)=1./gap;
        }
    }
    if(!inverse_.allFinite() || !projector_.allFinite() || !inverse_divided_.allFinite() || !projector_divided_.allFinite())
        throw std::overflow_error("MANTA RI metric response spectral map overflow");
    const Eigen::MatrixXd canonical=metric.selfadjointView<Eigen::Lower>();
    const std::string payload="manta.ri-metric-response.v1;lower-triangle;"+ri_detail::matrix_identity(canonical)+";"+info_.factor_identity;
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size());
    hash_number(hash,options.absolute_cutoff); hash_number(hash,options.relative_cutoff);
    hash_number(hash,options.negative_relative_tolerance);
    info_.source_identity=hash.finish_hex();
}
Eigen::MatrixXd RIMetricResponse::rotate(const Eigen::MatrixXd& direction) const {
    validate_matrix(direction,plan_.dimension);
    const Eigen::MatrixXd canonical=direction.selfadjointView<Eigen::Lower>();
    Eigen::MatrixXd rotated=(spectrum_.vectors.transpose()*canonical)*spectrum_.vectors;
    if(!rotated.allFinite()) throw std::overflow_error("MANTA RI metric response rotation overflow");
    return rotated;
}
Eigen::MatrixXd RIMetricResponse::restore(const Eigen::MatrixXd& rotated) const {
    Eigen::MatrixXd result=(spectrum_.vectors*rotated)*spectrum_.vectors.transpose();
    if(!result.allFinite()) throw std::overflow_error("MANTA RI metric response contraction overflow");
    return result;
}
RIMetricDerivative RIMetricResponse::apply(const Eigen::MatrixXd& direction) const {
    const auto rotated=rotate(direction);
    RIMetricDerivative result;
    result.inverse=restore(rotated.cwiseProduct(inverse_divided_));
    result.projector=restore(rotated.cwiseProduct(projector_divided_));
    result.source_identity=info_.source_identity;
    const Eigen::MatrixXd canonical=direction.selfadjointView<Eigen::Lower>();
    result.direction_identity=ri_detail::matrix_identity(canonical);
    result.modeled_peak_bytes=plan_.modeled_peak_bytes;
    return result;
}
Eigen::MatrixXd RIMetricResponse::adjoint(const Eigen::MatrixXd& inverse_seed,const Eigen::MatrixXd& projector_seed) const {
    // Both spectral maps are self-adjoint on real symmetric matrices with
    // the full Frobenius pairing (off-diagonal entries counted twice).
    const Eigen::MatrixXd a=rotate(inverse_seed),b=rotate(projector_seed);
    return restore(a.cwiseProduct(inverse_divided_)+b.cwiseProduct(projector_divided_));
}
} // namespace vibeqc::manta
