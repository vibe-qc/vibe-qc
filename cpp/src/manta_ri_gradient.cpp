#include "vibeqc/manta_ri_gradient.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_derivatives.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc::manta {
namespace {
KernelRequest request_for(KernelRequest r,int centers,int derivative=0) {
    r.centers=centers; r.derivative=derivative; return r;
}
void geometry(const Molecule& molecule,const BasisSet& basis) {
    for(const auto& atom:molecule.atoms()) for(double x:atom.xyz) admission::require_finite(x);
    for(std::size_t s=0;s<basis.nshells();++s) {
        const auto owner=basis.shell_atom_index(s);
        if(owner<0 || static_cast<std::size_t>(owner)>=molecule.atoms().size() || basis.libint()[s].O!=molecule.atoms()[owner].xyz)
            throw std::invalid_argument("MANTA RI Coulomb gradient basis centers/atom map do not match the molecule");
    }
}
void density_valid(const Eigen::MatrixXd& d,Count n) {
    if(d.rows()!=static_cast<Eigen::Index>(n) || d.cols()!=d.rows() || !d.allFinite())
        throw std::invalid_argument("MANTA RI Coulomb gradient requires a finite density matching the orbital basis");
    const double scale=std::max(1.,d.cwiseAbs().maxCoeff());
    for(Eigen::Index i=0;i<d.rows();++i) for(Eigen::Index j=0;j<i;++j)
        if(std::abs(d(i,j)-d(j,i))>1e-12*scale)
            throw std::invalid_argument("MANTA RI Coulomb gradient requires symmetric real density");
}
void accumulate(double value,double& sum,double& correction) {
    const double adjusted=value-correction,next=sum+adjusted;
    correction=(next-sum)-adjusted; sum=next;
}
std::string gradient_identity(Count atoms,const BasisSet& orbital,const BasisSet& auxiliary,const std::string& source,const std::string& component="coulomb") {
    std::string payload="manta.ri-"+component+"-gradient.v1;"+source+";atoms="+std::to_string(atoms);
    for(const auto* basis:{&orbital,&auxiliary}) {
        payload+=';';
        for(std::size_t s=0;s<basis->nshells();++s) payload+=std::to_string(basis->shell_atom_index(s))+",";
    }
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size());
    return hash.finish_hex();
}
} // namespace
RIJGradientPlan plan_ri_coulomb_gradient(const Molecule& molecule,const BasisSet& orbital,const BasisSet& auxiliary,
    const RIOptions& options,const KernelRequest& request,const ResourceBudget& budget) {
    using namespace admission;
    ri_detail::validate_options(options); margin(budget);
    const auto error=md_request_error(request,4);
    if(!error.empty()) throw std::invalid_argument(error);
    RIJGradientPlan p;
    p.n_atoms=molecule.atoms().size(); p.n_basis=orbital.nbasis(); p.n_auxiliary=auxiliary.nbasis();
    if(!p.n_atoms) throw std::invalid_argument("MANTA RI Coulomb gradient requires a nonempty molecule");
    const auto capability=md_capability(request_for(request,3,1));
    if(!capability.supported) throw std::invalid_argument(capability.reason);
    const auto o=census(orbital,capability.maximum_l),a=census(auxiliary,capability.maximum_auxiliary_l);
    // Dense metric, adjoint/seeds and matrix products; input/binding copies,
    // rhs/solution, atom sums/corrections and independent provider snapshots.
    const Count resident=add(mul(8,add(add(mul(16,square(p.n_auxiliary)),mul(8,square(p.n_basis))),mul(32,p.n_auxiliary))),
                             add(mul(512,p.n_atoms),mul(16,add(orbital.nshells(),auxiliary.nshells()))));
    p.modeled_peak_bytes=add(resident,budget.other_live_bytes);
    cap(add(p.modeled_peak_bytes,budget.backend_margin_bytes),budget.memory_bytes,"MANTA RI Coulomb gradient resident memory budget exceeded");
    ResourceBudget inner=budget; inner.other_live_bytes=0;
    const auto metric=plan_fitting({&auxiliary,&auxiliary},request_for(request,2),inner);
    const auto coupling=plan_fitting({&auxiliary,&orbital,&orbital},request_for(request,3),inner);
    const auto dm=plan_derivatives({&auxiliary,&auxiliary},request_for(request,2,1),inner);
    const auto db=plan_derivatives({&auxiliary,&orbital,&orbital},request_for(request,3,1),inner);
    const auto response=plan_ri_metric_response(p.n_auxiliary,options,inner);
    // Sum complete subplan workspaces, including their margins. Conservative:
    // the integral recurrence workspaces do not actually coexist with one another.
    for(auto peak:{metric.modeled_peak_bytes,coupling.modeled_peak_bytes,dm.modeled_peak_bytes,db.modeled_peak_bytes,response.modeled_peak_bytes})
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,peak);
    // Each provider owns basis axes; the aggregate allowance also covers local
    // shell-offset vectors and source metadata retained around its workspace.
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(10,add(o.bytes,a.bytes)));
    p.shell_calls=add(add(metric.shell_calls,coupling.shell_calls),add(dm.shell_calls,db.shell_calls));
    p.primitive_evaluations_upper_bound=add(add(metric.primitive_products,coupling.primitive_products),
                                           add(dm.primitive_evaluations_upper_bound,db.primitive_evaluations_upper_bound));
    p.scalar_terms=add(response.scalar_terms,add(mul(16,square(p.n_auxiliary)),
                         add(mul(2,coupling.integral_elements),add(mul(3,dm.integral_elements),mul(3,db.integral_elements)))));
    p.kernel_terms=add(p.scalar_terms,add(response.kernel_terms,add(add(metric.kernel_terms,coupling.kernel_terms),add(dm.kernel_terms,db.kernel_terms))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA RI Coulomb gradient combined memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA RI Coulomb gradient shell-call budget exceeded");
    cap(p.primitive_evaluations_upper_bound,budget.primitive_quartets,"MANTA RI Coulomb gradient primitive-evaluation budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA RI Coulomb gradient scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA RI Coulomb gradient kernel-work budget exceeded");
    geometry(molecule,orbital); geometry(molecule,auxiliary);
    return p;
}
RIJGradientResult compute_ri_coulomb_gradient(const Molecule& molecule,const BasisSet& orbital,const BasisSet& auxiliary,
    const Eigen::MatrixXd& density,const RIOptions& options,const KernelRequest& request,const ResourceBudget& budget) {
    const auto p=plan_ri_coulomb_gradient(molecule,orbital,auxiliary,options,request,budget);
    density_valid(density,p.n_basis);
    RIJGradientResult result;
    auto& receipt=result.receipt;
    receipt.kernel=request.kernel; receipt.omega=request.omega; receipt.modeled_peak_bytes=p.modeled_peak_bytes;
    receipt.density_identity=ri_detail::matrix_identity(density);
    result.values=result.coupling=result.metric=Eigen::MatrixXd::Zero(p.n_atoms,3);
    Eigen::MatrixXd coupling_error=result.coupling,metric_error=result.metric;
    FittingTiles metric_source({&auxiliary,&auxiliary},request_for(request,2),budget);
    FittingTiles coupling_source({&auxiliary,&orbital,&orbital},request_for(request,3),budget);
    const auto ao=orbital.libint().shell2bf(),aux=auxiliary.libint().shell2bf();
    Eigen::MatrixXd metric=Eigen::MatrixXd::Zero(p.n_auxiliary,p.n_auxiliary);
    for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<auxiliary.nshells();++b) {
        const auto tile=metric_source.evaluate({a,b});
        metric.block(aux[a],aux[b],tile.shape[0],tile.shape[1])=tile.values;
        ++receipt.metric_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_products;
        receipt.integral_elements_consumed+=tile.values.size();
    }
    RIMetricResponse response(metric,options,budget);
    result.metric_response=response.info();
    receipt.ri_source_identity=ri_detail::source_identity(metric_source.source_identity(),coupling_source.source_identity(),
                                                         response.info().factor_identity,options);
    receipt.source_identity=gradient_identity(p.n_atoms,orbital,auxiliary,receipt.ri_source_identity);
    if(!response.info().rank) return result;
    Eigen::VectorXd rhs=Eigen::VectorXd::Zero(p.n_auxiliary),rhs_error=rhs;
    for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<orbital.nshells();++b) for(std::size_t c=0;c<orbital.nshells();++c) {
        const auto tile=coupling_source.evaluate({a,b,c});
        ++receipt.coupling_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_products;
        receipt.integral_elements_consumed+=tile.values.size();
        for(Count i=0;i<tile.shape[0];++i) for(Count j=0;j<tile.shape[1];++j) for(Count k=0;k<tile.shape[2];++k)
            accumulate(tile.values(i,j*tile.shape[2]+k)*density(ao[b]+j,ao[c]+k),rhs[aux[a]+i],rhs_error[aux[a]+i]);
    }
    const Eigen::VectorXd projected=response.factor().transpose()*rhs,solution=response.factor()*projected;
    receipt.coulomb_energy=.5*projected.squaredNorm(); receipt.metric_rhs_norm=rhs.norm();
    receipt.metric_residual_norm=(metric.selfadjointView<Eigen::Lower>()*solution-rhs).norm(); receipt.metric_residual_evaluated=true;
    // SHARK (Neese 2023), RI Coulomb energy; Higham (2008), Chapter 3:
    // dE = (G b)^T dB D + 0.5 b^T L_G(V,dV) b. The adjoint transports the
    // retained projector rather than assuming dG = -G dV G after truncation.
    const Eigen::MatrixXd seed=.5*(rhs*rhs.transpose());
    const Eigen::MatrixXd metric_adjoint=response.adjoint(seed,Eigen::MatrixXd::Zero(p.n_auxiliary,p.n_auxiliary));
    DerivativeTiles metric_derivative({&auxiliary,&auxiliary},request_for(request,2,1),budget);
    for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<auxiliary.nshells();++b) {
        const std::vector<std::size_t> indices{a,b};
        for(int center=0;center<2;++center) for(int axis=0;axis<3;++axis) {
            const auto tile=metric_derivative.evaluate(indices,center,axis);
            ++receipt.metric_derivative_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_evaluations;
            receipt.integral_elements_consumed+=tile.values.size();
            const int owner=auxiliary.shell_atom_index(indices[center]);
            for(Count i=0;i<tile.shape[0];++i) for(Count j=0;j<tile.shape[1];++j) {
                const auto row=aux[a]+i,col=aux[b]+j;
                if(row<col) continue; // Differentiate the actual lower-triangle source.
                const double coefficient=metric_adjoint(row,col)+(row==col?0.:metric_adjoint(col,row));
                accumulate(coefficient*tile.values(i,j),result.metric(owner,axis),metric_error(owner,axis));
            }
        }
    }
    DerivativeTiles coupling_derivative({&auxiliary,&orbital,&orbital},request_for(request,3,1),budget);
    for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<orbital.nshells();++b) for(std::size_t c=0;c<orbital.nshells();++c) {
        const std::vector<std::size_t> indices{a,b,c};
        for(int center=0;center<3;++center) for(int axis=0;axis<3;++axis) {
            const auto tile=coupling_derivative.evaluate(indices,center,axis);
            ++receipt.coupling_derivative_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_evaluations;
            receipt.integral_elements_consumed+=tile.values.size();
            const int owner=center==0?auxiliary.shell_atom_index(a):orbital.shell_atom_index(indices[center]);
            for(Count i=0;i<tile.shape[0];++i) for(Count j=0;j<tile.shape[1];++j) for(Count k=0;k<tile.shape[2];++k)
                accumulate(solution[aux[a]+i]*density(ao[b]+j,ao[c]+k)*tile.values(i,j*tile.shape[2]+k),
                           result.coupling(owner,axis),coupling_error(owner,axis));
        }
    }
    result.values=result.coupling+result.metric;
    if(!result.values.allFinite() || !result.coupling.allFinite() || !result.metric.allFinite() || !std::isfinite(receipt.coulomb_energy)
       || !std::isfinite(receipt.metric_rhs_norm) || !std::isfinite(receipt.metric_residual_norm))
        throw std::overflow_error("MANTA RI Coulomb gradient nonfinite contraction");
    return result;
}

RIKGradientPlan plan_ri_exchange_gradient(const Molecule& molecule,const BasisSet& orbital,const BasisSet& auxiliary,
    const RIOptions& options,const KernelRequest& request,const ResourceBudget& budget) {
    using namespace admission;
    RIKGradientPlan p;
    static_cast<RIJGradientPlan&>(p)=plan_ri_coulomb_gradient(molecule,orbital,auxiliary,options,request,budget);
    const Count n=p.n_basis,m=p.n_auxiliary;
    p.mode_block=std::min(m,options.mode_block);
    p.auxiliary_blocks=add(m,p.mode_block-1)/p.mode_block;
    p.maximum_coupling_passes=mul(p.auxiliary_blocks,add(p.auxiliary_blocks,1))/2;
    p.maximum_derivative_passes=p.auxiliary_blocks;
    const auto fit=plan_fitting({&auxiliary,&orbital,&orbital},request_for(request,3),budget);
    const auto derivative=plan_derivatives({&auxiliary,&orbital,&orbital},request_for(request,3,1),budget);
    const Count more_fit=p.maximum_coupling_passes-1,more_derivative=p.maximum_derivative_passes-1;
    p.shell_calls=add(p.shell_calls,add(mul(more_fit,fit.shell_calls),mul(more_derivative,derivative.shell_calls)));
    p.primitive_evaluations_upper_bound=add(p.primitive_evaluations_upper_bound,
        add(mul(more_fit,fit.primitive_products),mul(more_derivative,derivative.primitive_evaluations_upper_bound)));
    // Raw auxiliary blocks, density-weighted blocks, a second raw block, two
    // input densities/copies and multiplication temporaries. No m*n*n tensor.
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(mul(8,mul(add(mul(8,p.mode_block),24),square(n))),mul(512,p.mode_block)));
    const Count contractions=add(mul(24,mul(m,mul(n,square(n)))),
        add(mul(8,mul(square(m),square(n))),mul(8,mul(m,derivative.integral_elements))));
    p.scalar_terms=add(p.scalar_terms,contractions);
    p.kernel_terms=add(p.kernel_terms,add(contractions,add(mul(more_fit,fit.kernel_terms),mul(more_derivative,derivative.kernel_terms))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA RI exchange gradient combined memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA RI exchange gradient shell-call budget exceeded");
    cap(p.primitive_evaluations_upper_bound,budget.primitive_quartets,"MANTA RI exchange gradient primitive-evaluation budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA RI exchange gradient scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA RI exchange gradient kernel-work budget exceeded");
    return p;
}
RIKGradientResult compute_ri_exchange_gradient(const Molecule& molecule,const BasisSet& orbital,const BasisSet& auxiliary,
    const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta,const RIOptions& options,
    const KernelRequest& request,const ResourceBudget& budget) {
    const auto p=plan_ri_exchange_gradient(molecule,orbital,auxiliary,options,request,budget);
    density_valid(alpha,p.n_basis); density_valid(beta,p.n_basis);
    RIKGradientResult result;
    auto& receipt=result.receipt;
    receipt.kernel=request.kernel; receipt.omega=request.omega; receipt.modeled_peak_bytes=p.modeled_peak_bytes;
    receipt.density_identities={ri_detail::matrix_identity(alpha),ri_detail::matrix_identity(beta)};
    const std::string density_payload="manta.ri.density-pair.v1;"+receipt.density_identities[0]+";"+receipt.density_identities[1];
    vibeqc::detail::Sha256 density_hash;
    density_hash.update(reinterpret_cast<const std::uint8_t*>(density_payload.data()),density_payload.size());
    receipt.density_identity=density_hash.finish_hex();
    result.values=result.coupling=result.metric=Eigen::MatrixXd::Zero(p.n_atoms,3);
    Eigen::MatrixXd coupling_error=result.coupling,metric_error=result.metric;
    FittingTiles metric_source({&auxiliary,&auxiliary},request_for(request,2),budget);
    FittingTiles coupling_source({&auxiliary,&orbital,&orbital},request_for(request,3),budget);
    const auto ao=orbital.libint().shell2bf(),aux=auxiliary.libint().shell2bf();
    Eigen::MatrixXd metric=Eigen::MatrixXd::Zero(p.n_auxiliary,p.n_auxiliary);
    for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<auxiliary.nshells();++b) {
        const auto tile=metric_source.evaluate({a,b});
        metric.block(aux[a],aux[b],tile.shape[0],tile.shape[1])=tile.values;
        ++receipt.metric_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_products;
        receipt.integral_elements_consumed+=tile.values.size();
    }
    RIMetricResponse response(metric,options,budget);
    result.metric_response=response.info();
    receipt.ri_source_identity=ri_detail::source_identity(metric_source.source_identity(),coupling_source.source_identity(),
                                                         response.info().factor_identity,options);
    receipt.source_identity=gradient_identity(p.n_atoms,orbital,auxiliary,receipt.ri_source_identity,"exchange");
    if(!response.info().rank) return result;
    const auto& g=response.inverse();
    Eigen::MatrixXd seed=Eigen::MatrixXd::Zero(p.n_auxiliary,p.n_auxiliary);
    auto raw_block=[&](Count start,Count count) {
        ++receipt.coupling_passes;
        std::vector<Eigen::MatrixXd> block(count,Eigen::MatrixXd::Zero(p.n_basis,p.n_basis));
        for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<orbital.nshells();++b) for(std::size_t c=0;c<orbital.nshells();++c) {
            const auto tile=coupling_source.evaluate({a,b,c});
            ++receipt.coupling_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_products;
            receipt.integral_elements_consumed+=tile.values.size();
            for(Count i=0;i<tile.shape[0];++i) {
                const Count row=aux[a]+i;
                if(row<start || row>=start+count) continue;
                for(Count j=0;j<tile.shape[1];++j) for(Count k=0;k<tile.shape[2];++k)
                    block[row-start](ao[b]+j,ao[c]+k)=tile.values(i,j*tile.shape[2]+k);
            }
        }
        return block;
    };
    DerivativeTiles coupling_derivative({&auxiliary,&orbital,&orbital},request_for(request,3,1),budget);
    // E_K = sum_PQ G_PQ W_PQ, W_PQ = -0.5 sum_spin <D B_P D, B_Q>.
    // Form W in bounded raw auxiliary blocks, including discarded modes. A
    // retained-only construction would lose the cross-space metric response.
    for(Count start=0;start<p.n_auxiliary;start+=p.mode_block) {
        const Count count=std::min(p.mode_block,p.n_auxiliary-start);
        const auto raw=raw_block(start,count);
        std::vector<Eigen::MatrixXd> weighted; weighted.reserve(count);
        for(const auto& b:raw) {
            // The two transposed terms also retain the exact derivative of
            // Tr(D K[D]) for a roundoff-asymmetric admitted density.
            Eigen::MatrixXd c=.5*((alpha*b)*alpha+(alpha.transpose()*b)*alpha.transpose()
                                +(beta*b)*beta+(beta.transpose()*b)*beta.transpose());
            if(!c.allFinite()) throw std::overflow_error("MANTA RI exchange gradient density contraction overflow");
            weighted.push_back(std::move(c));
        }
        for(Count qstart=0;qstart<=start;qstart+=p.mode_block) {
            const Count qcount=std::min(p.mode_block,p.n_auxiliary-qstart);
            std::vector<Eigen::MatrixXd> fetched;
            if(qstart!=start) fetched=raw_block(qstart,qcount);
            const auto& other=qstart==start?raw:fetched;
            for(Count i=0;i<count;++i) for(Count j=0;j<qcount;++j) {
                const Count row=start+i,col=qstart+j;
                if(row<col) continue;
                double value=0.,error=0.;
                for(Count mu=0;mu<p.n_basis;++mu) for(Count nu=0;nu<p.n_basis;++nu)
                    accumulate(-.5*weighted[i](mu,nu)*other[j](mu,nu),value,error);
                seed(row,col)=seed(col,row)=value;
            }
        }
        // dE/dB_Q = -sum_P G_PQ sum_spin D_spin B_P D_spin.
        // Contract a block of P directly with derivative tiles for every Q.
        ++receipt.coupling_derivative_passes;
        for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<orbital.nshells();++b) for(std::size_t c=0;c<orbital.nshells();++c) {
            const std::vector<std::size_t> indices{a,b,c};
            for(int center=0;center<3;++center) for(int axis=0;axis<3;++axis) {
                const auto tile=coupling_derivative.evaluate(indices,center,axis);
                ++receipt.coupling_derivative_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_evaluations;
                receipt.integral_elements_consumed+=tile.values.size();
                const int owner=center==0?auxiliary.shell_atom_index(a):orbital.shell_atom_index(indices[center]);
                const Eigen::MatrixXd transformed=g.block(start,aux[a],count,tile.shape[0])*tile.values;
                for(Count i=0;i<count;++i) for(Count j=0;j<tile.shape[1];++j) for(Count k=0;k<tile.shape[2];++k)
                    accumulate(-weighted[i](ao[b]+j,ao[c]+k)*transformed(i,j*tile.shape[2]+k),
                               result.coupling(owner,axis),coupling_error(owner,axis));
            }
        }
    }
    double energy_error=0.;
    for(Count i=0;i<p.n_auxiliary;++i) for(Count j=0;j<p.n_auxiliary;++j)
        accumulate(g(i,j)*seed(i,j),receipt.exchange_energy,energy_error);
    const Eigen::MatrixXd metric_adjoint=response.adjoint(seed,Eigen::MatrixXd::Zero(p.n_auxiliary,p.n_auxiliary));
    DerivativeTiles metric_derivative({&auxiliary,&auxiliary},request_for(request,2,1),budget);
    for(std::size_t a=0;a<auxiliary.nshells();++a) for(std::size_t b=0;b<auxiliary.nshells();++b) {
        const std::vector<std::size_t> indices{a,b};
        for(int center=0;center<2;++center) for(int axis=0;axis<3;++axis) {
            const auto tile=metric_derivative.evaluate(indices,center,axis);
            ++receipt.metric_derivative_shell_calls; receipt.primitive_integrals_evaluated+=tile.primitive_evaluations;
            receipt.integral_elements_consumed+=tile.values.size();
            const int owner=auxiliary.shell_atom_index(indices[center]);
            for(Count i=0;i<tile.shape[0];++i) for(Count j=0;j<tile.shape[1];++j) {
                const auto row=aux[a]+i,col=aux[b]+j;
                if(row<col) continue;
                const double coefficient=metric_adjoint(row,col)+(row==col?0.:metric_adjoint(col,row));
                accumulate(coefficient*tile.values(i,j),result.metric(owner,axis),metric_error(owner,axis));
            }
        }
    }
    result.values=result.coupling+result.metric;
    if(!result.values.allFinite() || !result.coupling.allFinite() || !result.metric.allFinite() || !std::isfinite(receipt.exchange_energy))
        throw std::overflow_error("MANTA RI exchange gradient nonfinite contraction");
    return result;
}
} // namespace vibeqc::manta
