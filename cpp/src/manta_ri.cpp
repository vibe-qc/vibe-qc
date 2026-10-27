#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc::manta {
namespace {
KernelRequest center_request(KernelRequest request,int centers) {
    request.centers=centers; return request;
}
void add_compensated(double value,double& sum,double& correction) {
    const double adjusted=value-correction,next=sum+adjusted;
    correction=(next-sum)-adjusted; sum=next;
}
void validate_density(const RIPlan& plan,const Eigen::MatrixXd& d) {
    if(d.rows()!=static_cast<Eigen::Index>(plan.n_basis) || d.cols()!=d.rows())
        throw std::invalid_argument("MANTA RI density dimensions");
    if(!d.allFinite()) throw std::invalid_argument("MANTA RI density must be finite");
    const double scale=std::max(1.,d.cwiseAbs().maxCoeff());
    for(Eigen::Index i=0;i<d.rows();++i) for(Eigen::Index j=0;j<i;++j)
        if(std::abs(d(i,j)-d(j,i))>1e-12*scale)
            throw std::invalid_argument("MANTA RI requires symmetric real density");
}
} // namespace

RIPlan plan_ri(const BasisSet& orbital,const BasisSet& auxiliary,const RIOptions& options,
               const KernelRequest& request,const ResourceBudget& budget) {
    using namespace admission;
    ri_detail::validate_options(options);
    const auto error=md_request_error(request,4);
    if(!error.empty()) throw std::invalid_argument(error);
    margin(budget);
    RIPlan plan;
    plan.n_basis=orbital.nbasis(); plan.n_auxiliary=auxiliary.nbasis();
    if(!plan.n_basis || !plan.n_auxiliary) throw std::invalid_argument("MANTA RI requires nonempty bases");
    plan.maximum_rank=plan.n_auxiliary;
    plan.mode_block=std::min(options.mode_block,plan.n_auxiliary);
    const Count n=plan.n_basis,m=plan.n_auxiliary;
    // Dense metric/eigensolver/factor and diagnostic copies, live input and
    // targets, compensated fitted blocks, GEMM and tile-transform temporaries.
    const Count matrices=add(mul(16,square(m)),mul(add(16,mul(4,plan.mode_block)),square(n)));
    const Count working=add(mul(8,add(matrices,mul(32,m))),mul(512,plan.mode_block));
    cap(add(working,add(budget.other_live_bytes,budget.backend_margin_bytes)),budget.memory_bytes,
        "MANTA RI metric/block memory budget exceeded");
    const auto metric=plan_fitting({&auxiliary,&auxiliary},center_request(request,2),budget);
    const auto coupling=plan_fitting({&auxiliary,&orbital,&orbital},center_request(request,3),budget);
    plan.metric_shell_calls=metric.shell_calls;
    plan.coupling_shell_calls_per_pass=coupling.shell_calls;
    plan.maximum_coupling_passes=add(2,add(m,plan.mode_block-1)/plan.mode_block);
    plan.shell_calls=add(metric.shell_calls,mul(plan.maximum_coupling_passes,coupling.shell_calls));
    plan.primitive_products=add(metric.primitive_products,mul(plan.maximum_coupling_passes,coupling.primitive_products));
    cap(plan.shell_calls,budget.shell_calls,"MANTA RI combined shell-call budget exceeded");
    cap(plan.primitive_products,budget.primitive_quartets,"MANTA RI combined primitive-product budget exceeded");
    // This is a conservative work model, not an executed instruction counter.
    plan.scalar_terms=add(metric.integral_elements,add(mul(add(2,mul(2,m)),coupling.integral_elements),
                          add(mul(8,mul(m,mul(n,square(n)))),mul(8,square(m)))));
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA RI scalar-contraction budget exceeded");
    plan.kernel_terms=add(add(metric.kernel_terms,mul(plan.maximum_coupling_passes,coupling.kernel_terms)),
                          add(plan.scalar_terms,mul(768,mul(m,square(m)))));
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA RI combined kernel-work budget exceeded");
    // Both subplans already include the same caller-owned live storage. Count
    // it once so a composed producer's admitted peak remains reproducible.
    plan.modeled_peak_bytes=add(working,add(metric.modeled_peak_bytes,coupling.modeled_peak_bytes-budget.other_live_bytes));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA RI combined workspace memory budget exceeded");
    return plan;
}

RIJKBuilder::RIJKBuilder(const BasisSet& orbital,const BasisSet& auxiliary,const RIOptions& options,
                         const KernelRequest& request,const ResourceBudget& budget)
    : plan_(plan_ri(orbital,auxiliary,options,request,budget)),request_(request),
      coupling_({&auxiliary,&orbital,&orbital},center_request(request,3),budget) {
    FittingTiles metric_source({&auxiliary,&auxiliary},center_request(request,2),budget);
    metric_=Eigen::MatrixXd::Zero(plan_.n_auxiliary,plan_.n_auxiliary);
    const auto offsets=metric_source.basis(0).libint().shell2bf();
    for(std::size_t p=0;p<auxiliary.nshells();++p) for(std::size_t q=0;q<auxiliary.nshells();++q) {
        const auto tile=metric_source.evaluate({p,q});
        metric_.block(offsets[p],offsets[q],tile.shape[0],tile.shape[1])=tile.values;
        ++info_.shell_calls; info_.primitive_products+=tile.primitive_products;
    }
    auto spectrum=ri_detail::decompose(metric_,options);
    const auto shell_calls=info_.shell_calls,primitive_products=info_.primitive_products;
    info_=std::move(spectrum.info);
    info_.shell_calls=shell_calls; info_.primitive_products=primitive_products;
    factor_=std::move(spectrum.factor);
    source_=ri_detail::source_identity(metric_source.source_identity(),coupling_.source_identity(),info_.factor_identity,options);
}

RIResult RIJKBuilder::build(const Eigen::MatrixXd& density,bool make_j,bool make_k) const {
    return std::move(build_impl(density,nullptr,make_j,make_k).first);
}
RIFactorExportPlan RIJKBuilder::plan_factor_export(const ResourceBudget& budget) const {
    using namespace admission;
    margin(budget);
    RIFactorExportPlan p;
    p.n_basis=plan_.n_basis; p.retained_rank=info_.rank;
    const Count pairs=square(p.n_basis);
    if(pairs>static_cast<Count>(std::numeric_limits<Eigen::Index>::max()))
        throw std::length_error("MANTA RI export AO-pair dimension exceeds index range");
    p.exported_values=mul(p.retained_rank,pairs);
    cap(p.exported_values,static_cast<Count>(std::numeric_limits<Eigen::Index>::max()),
        "MANTA RI export array exceeds index range");
    const auto& c=coupling_.plan();
    if(p.retained_rank) {
        p.shell_calls=c.shell_calls; p.primitive_products=c.primitive_products;
        p.integral_elements=c.integral_elements;
        p.contraction_products=mul(p.retained_rank,c.integral_elements);
        p.accumulation_terms=mul(p.exported_values,c.shell_extents[0]);
        p.kernel_terms=add(c.kernel_terms,add(mul(2,p.contraction_products),mul(6,p.accumulation_terms)));
    }
    cap(p.shell_calls,budget.shell_calls,"MANTA RI export shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA RI export primitive-product budget exceeded");
    cap(add(p.contraction_products,p.accumulation_terms),budget.scalar_terms,
        "MANTA RI export contraction budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA RI export kernel-work budget exceeded");
    // Existing builder and its fitting workspace, coefficient/compensation
    // arrays, a binding copy and a digest conversion, plus the transformed
    // shell tile. No complete unfitted (P|uv) tensor is allocated.
    const Count arrays=add(mul(32,p.exported_values),mul(8,mul(p.retained_rank,c.tile_elements)));
    p.modeled_peak_bytes=add(plan_.modeled_peak_bytes,
        add(arrays,add(budget.other_live_bytes,budget.backend_margin_bytes)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA RI export memory budget exceeded");
    return p;
}
RIFactorExport RIJKBuilder::export_factors(const ResourceBudget& budget) const {
    const auto p=plan_factor_export(budget); // Admission before arrays or tiles.
    RIFactorExport result;
    auto& receipt=result.receipt_;
    receipt.kernel=request_.kernel; receipt.omega=request_.omega;
    receipt.source_identity=source_; receipt.coupling_identity=coupling_.source_identity();
    receipt.metric_factor_identity=info_.factor_identity;
    receipt.n_basis=p.n_basis; receipt.retained_rank=p.retained_rank;
    receipt.factor_array_bytes=admission::mul(8,p.exported_values);
    receipt.modeled_peak_bytes=p.modeled_peak_bytes;
    result.coefficients_=md::Matrix::Zero(p.retained_rank,admission::square(p.n_basis));
    md::Matrix correction=result.coefficients_;
    const auto& auxiliary=coupling_.basis(0).libint();
    const auto& orbital=coupling_.basis(1).libint();
    const auto aux_offsets=auxiliary.shell2bf(),orb_offsets=orbital.shell2bf();
    // Melega et al., doi:10.1021/acs.jpca.5c08579, Eqs. (3)-(5):
    // fit the actual finite metric factor once. Preserve ordered T and its
    // pure/Cartesian normalization; no symmetry, screening or rank change.
    if(p.retained_rank) for(std::size_t a=0;a<auxiliary.size();++a)
        for(std::size_t u=0;u<orbital.size();++u) for(std::size_t v=0;v<orbital.size();++v) {
            const auto tile=coupling_.evaluate({a,u,v});
            const md::Matrix transformed=factor_.block(aux_offsets[a],0,tile.shape[0],p.retained_rank).transpose()*tile.values;
            ++receipt.shell_calls; receipt.primitive_products+=tile.primitive_products;
            receipt.integral_elements_consumed+=tile.values.size();
            receipt.contraction_products+=p.retained_rank*tile.values.size();
            for(Count k=0;k<p.retained_rank;++k) for(Count i=0;i<tile.shape[1];++i)
                for(Count j=0;j<tile.shape[2];++j) {
                    const Count pair=(orb_offsets[u]+i)*p.n_basis+orb_offsets[v]+j;
                    add_compensated(transformed(k,i*tile.shape[2]+j),result.coefficients_(k,pair),correction(k,pair));
                    ++receipt.accumulation_terms;
                }
        }
    if(!result.coefficients_.allFinite()) throw std::overflow_error("MANTA RI nonfinite factor export");
    if(receipt.shell_calls!=p.shell_calls || receipt.primitive_products!=p.primitive_products
       || receipt.integral_elements_consumed!=p.integral_elements || receipt.contraction_products!=p.contraction_products
       || receipt.accumulation_terms!=p.accumulation_terms)
        throw std::logic_error("MANTA RI factor export census mismatch");
    const std::string payload=receipt.algorithm+";"+source_+";"+ri_detail::matrix_identity(result.coefficients_);
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size());
    receipt.export_identity=hash.finish_hex();
    return result;
}
std::pair<Eigen::MatrixXd,Eigen::MatrixXd> RIJKBuilder::build_K_pair(const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta) const {
    auto result=build_exchange_pair(alpha,beta);
    return {std::move(result.K_alpha),std::move(result.K_beta)};
}
RIExchangePair RIJKBuilder::build_exchange_pair(const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta) const {
    auto result=build_impl(alpha,&beta,false,true);
    RIExchangePair pair;
    pair.K_alpha=std::move(result.first.K); pair.K_beta=std::move(result.second);
    pair.receipt=std::move(result.first.receipt);
    pair.density_identities={pair.receipt.density_identity,ri_detail::matrix_identity(beta)};
    const std::string payload="manta.ri.density-pair.v1;"+pair.density_identities[0]+";"+pair.density_identities[1];
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size());
    pair.receipt.density_identity=hash.finish_hex();
    pair.receipt.algorithm="streamed-ri-exchange-pair";
    return pair;
}
std::pair<RIResult,Eigen::MatrixXd> RIJKBuilder::build_impl(const Eigen::MatrixXd& density,const Eigen::MatrixXd* beta,bool make_j,bool make_k) const {
    if(!make_j && !make_k) throw std::invalid_argument("MANTA RI target selection");
    validate_density(plan_,density);
    if(beta) validate_density(plan_,*beta);
    RIResult result;
    Eigen::MatrixXd beta_k;
    const Count n=plan_.n_basis,m=plan_.n_auxiliary;
    if(make_j) result.J=Eigen::MatrixXd::Zero(n,n);
    if(make_k) result.K=Eigen::MatrixXd::Zero(n,n);
    if(beta) beta_k=Eigen::MatrixXd::Zero(n,n);
    auto& receipt=result.receipt;
    receipt.kernel=request_.kernel; receipt.omega=request_.omega;
    receipt.source_identity=source_; receipt.density_identity=ri_detail::matrix_identity(density);
    receipt.retained_rank=info_.rank; receipt.modeled_peak_bytes=plan_.modeled_peak_bytes;
    if(!info_.rank) return {std::move(result),std::move(beta_k)};
    const auto& auxiliary=coupling_.basis(0).libint();
    const auto& orbital=coupling_.basis(1).libint();
    const auto aux_offsets=auxiliary.shell2bf(),orb_offsets=orbital.shell2bf();
    auto pass=[&](const auto& consume) {
        ++receipt.coupling_passes;
        for(std::size_t p=0;p<auxiliary.size();++p) for(std::size_t a=0;a<orbital.size();++a)
        for(std::size_t b=0;b<orbital.size();++b) {
            const auto tile=coupling_.evaluate({p,a,b});
            ++receipt.shell_calls; receipt.primitive_products+=tile.primitive_products;
            receipt.integral_elements_consumed+=tile.values.size();
            consume(tile,aux_offsets[p],orb_offsets[a],orb_offsets[b]);
        }
    };
    // SHARK, Neese, J. Comput. Chem. 44, 381 (2023), Eqs. 42-50.
    // The two Coulomb passes use the identical ordered raw coupling T.
    if(make_j) {
        Eigen::VectorXd rhs=Eigen::VectorXd::Zero(m),correction=rhs;
        pass([&](const FittingTile& tile,Count p,Count a,Count b) {
            for(Count i=0;i<tile.shape[0];++i) for(Count j=0;j<tile.shape[1];++j) for(Count k=0;k<tile.shape[2];++k)
                add_compensated(tile.values(i,j*tile.shape[2]+k)*density(a+j,b+k),rhs[p+i],correction[p+i]);
        });
        const Eigen::VectorXd projected=factor_.transpose()*rhs;
        const Eigen::VectorXd solution=factor_*projected;
        receipt.coulomb_energy=.5*projected.squaredNorm();
        receipt.metric_rhs_norm=rhs.norm();
        receipt.metric_residual_norm=(metric_.selfadjointView<Eigen::Lower>()*solution-rhs).norm();
        receipt.metric_residual_evaluated=true;
        Eigen::MatrixXd compensation=Eigen::MatrixXd::Zero(n,n);
        pass([&](const FittingTile& tile,Count p,Count a,Count b) {
            for(Count i=0;i<tile.shape[0];++i) for(Count j=0;j<tile.shape[1];++j) for(Count k=0;k<tile.shape[2];++k)
                add_compensated(tile.values(i,j*tile.shape[2]+k)*solution[p+i],result.J(a+j,b+k),compensation(a+j,b+k));
        });
    }
    if(make_k) {
        Eigen::MatrixXd compensation=Eigen::MatrixXd::Zero(n,n);
        Eigen::MatrixXd beta_compensation;
        if(beta) beta_compensation=Eigen::MatrixXd::Zero(n,n);
        for(Count start=0;start<info_.rank;start+=plan_.mode_block) {
            const Count count=std::min(plan_.mode_block,info_.rank-start);
            std::vector<Eigen::MatrixXd> blocks(count,Eigen::MatrixXd::Zero(n,n)),corrections=blocks;
            pass([&](const FittingTile& tile,Count p,Count a,Count b) {
                const md::Matrix transformed=factor_.block(p,start,tile.shape[0],count).transpose()*tile.values;
                for(Count r=0;r<count;++r) for(Count j=0;j<tile.shape[1];++j) for(Count k=0;k<tile.shape[2];++k)
                    add_compensated(transformed(r,j*tile.shape[2]+k),blocks[r](a+j,b+k),corrections[r](a+j,b+k));
            });
            for(const auto& block:blocks) {
                auto accumulate=[&](const Eigen::MatrixXd& d,Eigen::MatrixXd& target,Eigen::MatrixXd& correction) {
                    const Eigen::MatrixXd contribution=(block*d)*block.transpose();
                    for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
                        add_compensated(contribution(i,j),target(i,j),correction(i,j));
                };
                accumulate(density,result.K,compensation);
                if(beta) accumulate(*beta,beta_k,beta_compensation);
            }
        }
    }
    if(!result.J.allFinite() || !result.K.allFinite() || !beta_k.allFinite() || !std::isfinite(receipt.coulomb_energy)
       || !std::isfinite(receipt.metric_rhs_norm) || !std::isfinite(receipt.metric_residual_norm))
        throw std::overflow_error("MANTA RI nonfinite contraction");
    return {std::move(result),std::move(beta_k)};
}
Eigen::MatrixXd RIJKBuilder::build_g_rhf(const Eigen::MatrixXd& density,double alpha) const {
    if(!std::isfinite(alpha)) throw std::invalid_argument("MANTA RI exchange fraction must be finite");
    auto result=build(density,true,alpha!=0.);
    if(alpha!=0.) result.J.noalias()-=.5*alpha*result.K;
    if(!result.J.allFinite()) throw std::overflow_error("MANTA RI nonfinite fused result");
    return std::move(result.J);
}
} // namespace vibeqc::manta
