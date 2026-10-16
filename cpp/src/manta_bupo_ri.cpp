#include "vibeqc/manta_bupo_ri.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc::manta {
namespace {
using namespace admission;
double upper_product(std::initializer_list<double> factors) {
    double logarithm=0.;
    for(double x:factors) { if(x==0.) return 0.; require_finite(x); logarithm+=std::log(x); }
    const double result=std::nextafter(std::max(std::numeric_limits<double>::denorm_min(),
        std::exp(logarithm+512*std::numeric_limits<double>::epsilon()*(1+std::abs(logarithm)))),std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA RI-BUPO propagated error bound overflow");
    return result;
}
double upper_sum(double a,double b) {
    if(a==0.) return b; if(b==0.) return a;
    const double result=std::nextafter(a+b,std::numeric_limits<double>::infinity());
    if(!std::isfinite(result)) throw std::overflow_error("MANTA RI-BUPO propagated error sum overflow");
    return result;
}
void validate(const BupoRIPlan& p,const Eigen::MatrixXd& d) {
    if(d.rows()!=Eigen::Index(p.n_basis) || d.cols()!=d.rows() || !d.allFinite())
        throw std::invalid_argument("MANTA RI-BUPO density must be finite and match the orbital basis");
    const double scale=std::max(1.,d.cwiseAbs().maxCoeff());
    for(Eigen::Index i=0;i<d.rows();++i) for(Eigen::Index j=0;j<i;++j)
        if(std::abs(d(i,j)-d(j,i))>1e-12*scale)
            throw std::invalid_argument("MANTA RI-BUPO requires a symmetric real density");
}
} // namespace
BupoRIJBuilder::BupoRIJBuilder(const BasisSet& orbital,const BasisSet& auxiliary,
    const BupoFittingOptions& coupling_options,const RIOptions& metric_options,const ResourceBudget& budget) {
    plan_.n_basis=orbital.nbasis(); plan_.n_auxiliary=auxiliary.nbasis();
    const auto exact_plan=plan_ri(orbital,auxiliary,metric_options,KernelRequest{},budget);
    // Reserve the entire exact reference producer before the coupling setup.
    // Its conservative plan includes exact confirmation and RI-K workspaces;
    // this J-only consumer does not claim those upper bounds as executed work.
    ResourceBudget stage=budget;
    const Count extra=add(4096,add(mul(64,square(plan_.n_basis)),mul(128,plan_.n_auxiliary)));
    stage.other_live_bytes=add(exact_plan.modeled_peak_bytes,extra);
    coupling_=std::make_shared<BupoFittingCoupling>(orbital,auxiliary,coupling_options,stage);
    const auto& cp=coupling_->plan();
    plan_.modeled_peak_bytes=cp.modeled_peak_bytes;
    plan_.shell_calls=add(cp.shell_calls,exact_plan.shell_calls);
    plan_.primitive_products=add(cp.primitive_products,exact_plan.primitive_products);
    plan_.scalar_terms=add(cp.scalar_terms,add(exact_plan.scalar_terms,
        add(mul(16,square(plan_.n_auxiliary)),mul(16,square(plan_.n_basis)))));
    plan_.kernel_terms=add(cp.kernel_terms,add(exact_plan.kernel_terms,mul(4,plan_.scalar_terms)));
    cap(plan_.shell_calls,budget.shell_calls,"MANTA RI-BUPO combined shell/traversal budget exceeded");
    cap(plan_.primitive_products,budget.primitive_quartets,"MANTA RI-BUPO combined primitive budget exceeded");
    cap(plan_.scalar_terms,budget.scalar_terms,"MANTA RI-BUPO combined scalar budget exceeded");
    cap(plan_.kernel_terms,budget.kernel_terms,"MANTA RI-BUPO combined work budget exceeded");
    exact_=std::make_shared<RIJKBuilder>(orbital,auxiliary,metric_options,KernelRequest{},budget);
    // ||V+||_2 <= ||F||_F^2 for V+=F F^T. This intentionally conservative
    // amplification bound uses the actual retained factor, including rank zero.
    const double fnorm=exact_->factor().stableNorm();
    inverse_bound_=upper_product({fnorm,fnorm});
    const std::string payload="manta.ri-bupo-j.v1;fixed-positive-exact-metric;paired-coupling;"+
        exact_->source_identity()+";"+coupling_->source_identity();
    vibeqc::detail::Sha256 hash; hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size()); source_=hash.finish_hex();
}
BupoRIResult BupoRIJBuilder::build(const Eigen::MatrixXd& density) const {
    validate(plan_,density);
    BupoRIResult result; auto& r=result.receipt;
    r.source_identity=source_; r.exact_ri_source_identity=exact_->source_identity();
    r.retained_rank=exact_->metric_info().rank; r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    r.metric_inverse_norm_bound=inverse_bound_;
    auto forward=coupling_->forward(density);
    result.rhs=std::move(forward.values); r.forward=std::move(forward.receipt);
    r.density_identity=r.forward.input_identity;
    // SHARK (Neese, 2023, DOI 10.1002/jcc.26942) positive metric factor,
    // with the paired coupling required by MANTA's BUPO energy contract
    // (Neese et al., 2025, DOI 10.1021/acs.jpca.4c07415).
    const Eigen::VectorXd projected=exact_->factor().transpose()*result.rhs;
    result.coefficients=exact_->factor()*projected;
    if(!result.coefficients.allFinite() || !projected.allFinite()) throw std::overflow_error("MANTA RI-BUPO metric solve overflow");
    r.coulomb_energy=.5*projected.squaredNorm();
    r.metric_rhs_norm=result.rhs.stableNorm();
    r.metric_residual_norm=(exact_->metric().selfadjointView<Eigen::Lower>()*result.coefficients-result.rhs).stableNorm();
    auto adjoint=coupling_->adjoint(result.coefficients);
    result.J=std::move(adjoint.values); r.adjoint=std::move(adjoint.receipt);
    r.forward_error_l2_bound=upper_product({std::sqrt(double(plan_.n_auxiliary)),r.forward.approximation_bound});
    // Relative to exact B with this SAME retained metric factor:
    // |dE| <= ||c~|| db + .5 ||V+|| db^2,
    // |dJ|_inf <= eps ||c~||_1 + sqrt(m) Bmax ||V+|| db.
    // RI basis, eigensolve and floating-point errors are separate.
    r.approximation_energy_bound=upper_sum(upper_product({result.coefficients.stableNorm(),r.forward_error_l2_bound}),
        upper_product({.5,inverse_bound_,r.forward_error_l2_bound,r.forward_error_l2_bound}));
    r.adjoint_j_bound=r.adjoint.approximation_bound;
    r.forward_propagated_j_bound=upper_product({std::sqrt(double(plan_.n_auxiliary)),coupling_->plan().maximum_exact_coupling,
        inverse_bound_,r.forward_error_l2_bound});
    r.approximation_j_bound=upper_sum(r.adjoint_j_bound,r.forward_propagated_j_bound);
    if(!std::isfinite(r.coulomb_energy) || !std::isfinite(r.metric_rhs_norm) || !std::isfinite(r.metric_residual_norm))
        throw std::overflow_error("MANTA RI-BUPO nonfinite energy or metric diagnostic");
    return result;
}
Eigen::MatrixXd BupoRIJBuilder::build_K(const Eigen::MatrixXd&) const {
    throw std::invalid_argument("MANTA RI-BUPO supplies J only; compose an independent exchange provider");
}
Eigen::MatrixXd BupoRIJBuilder::build_g_rhf(const Eigen::MatrixXd& density,double alpha) const {
    if(alpha!=0.) throw std::invalid_argument("MANTA RI-BUPO requires an independent exchange provider");
    return build_J(density);
}
} // namespace vibeqc::manta
