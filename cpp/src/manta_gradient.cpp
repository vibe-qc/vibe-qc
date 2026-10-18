#include "vibeqc/manta_gradient.hpp"
#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_derivatives.hpp"
#include "vibeqc/manta_one_electron.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
OneElectronRequest attraction(const Molecule& mol) {
    OneElectronRequest request; request.operation="potential";
    for(const auto& atom:mol.atoms()) request.charges.push_back({static_cast<double>(atom.Z),atom.xyz});
    return request;
}
void validate_geometry(const Molecule& mol,const BasisSet& basis) {
    for(const auto& a:mol.atoms()) {
        if(a.Z<0) throw std::invalid_argument("MANTA HF gradient requires nonnegative nuclear charges");
        for(auto x:a.xyz) admission::require_finite(x);
    }
    for(std::size_t i=0;i<basis.nshells();++i) {
        const int owner=basis.shell_atom_index(i);
        if(owner<0 || static_cast<std::size_t>(owner)>=mol.atoms().size() || basis.libint()[i].O!=mol.atoms()[owner].xyz)
            throw std::invalid_argument("MANTA HF gradient basis centers/atom map do not match the molecule");
    }
    for(std::size_t i=0;i<mol.atoms().size();++i) for(std::size_t j=0;j<i;++j) {
        const auto& a=mol.atoms()[i]; const auto& b=mol.atoms()[j];
        if(a.Z && b.Z && a.xyz==b.xyz) throw std::invalid_argument("MANTA HF gradient has coincident charged nuclei");
    }
}
void validate_matrix(const BasisSet& basis,const Eigen::MatrixXd& d) {
    if(d.rows()!=static_cast<Eigen::Index>(basis.nbasis()) || d.cols()!=d.rows() || !d.allFinite())
        throw std::invalid_argument("MANTA HF gradient requires finite matrices matching the basis");
    const double scale=std::max(1.,d.cwiseAbs().maxCoeff());
    for(Eigen::Index i=0;i<d.rows();++i) for(Eigen::Index j=0;j<i;++j)
        if(std::abs(d(i,j)-d(j,i))>1e-12*scale)
            throw std::invalid_argument("MANTA HF gradient requires symmetric real matrices");
}
void accumulate(double value,double& sum,double& correction) {
    const double adjusted=value-correction,next=sum+adjusted;
    correction=(next-sum)-adjusted; sum=next;
}
void hash_word(vibeqc::detail::Sha256& hash,std::uint64_t word) {
    std::array<std::uint8_t,8> bytes{};
    for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(word>>(56-8*k));
    hash.update(bytes.data(),bytes.size());
}
void hash_number(vibeqc::detail::Sha256& hash,double x) {
    if(x==0.) x=0.; std::uint64_t word; std::memcpy(&word,&x,sizeof(word)); hash_word(hash,word);
}
std::string source_identity(const Molecule& mol,const BasisSet& basis) {
    const std::string prefix="manta.hf-lagrangian-gradient.v1;all-electron;full-coulomb;md-fp64-v1;"+
        auxiliary_basis_content_identity_sha256(basis);
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    hash_word(hash,mol.atoms().size());
    for(const auto& a:mol.atoms()) { hash_word(hash,a.Z); for(auto x:a.xyz) hash_number(hash,x); }
    for(std::size_t i=0;i<basis.nshells();++i) hash_word(hash,basis.shell_atom_index(i));
    return hash.finish_hex();
}
std::string density_identity(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b,const Eigen::MatrixXd& w) {
    vibeqc::detail::Sha256 hash; hash_word(hash,a.rows());
    for(const auto* matrix:{&a,&b,&w})
        for(Eigen::Index i=0;i<matrix->rows();++i) for(Eigen::Index j=0;j<matrix->cols();++j) hash_number(hash,(*matrix)(i,j));
    return hash.finish_hex();
}
} // namespace
namespace {
HFGradientPlan plan_components(const Molecule& mol,const BasisSet& basis,const ResourceBudget& budget,bool exact_j,bool exact_k) {
    using namespace admission;
    margin(budget);
    HFGradientPlan p; p.n_atoms=mol.atoms().size(); p.n_basis=basis.nbasis();
    if(!p.n_atoms) throw std::invalid_argument("MANTA HF gradient requires a nonempty molecule");
    // Inputs and native copies, D_total, component sums/corrections/exports,
    // charge metadata and shell offsets stay live around each kernel workspace.
    const Count resident=add(mul(64,square(p.n_basis)),add(mul(512,p.n_atoms),mul(16,basis.nshells())));
    ResourceBudget inner=budget; inner.other_live_bytes=add(budget.other_live_bytes,resident);
    cap(add(inner.other_live_bytes,budget.backend_margin_bytes),budget.memory_bytes,"MANTA HF gradient input/output memory budget exceeded");
    if(exact_j || exact_k) {
        const auto two=plan_derivatives({&basis,&basis,&basis,&basis},derivative_request(4),inner);
        p.shell_calls=two.shell_calls; p.primitive_evaluations_upper_bound=two.primitive_evaluations_upper_bound;
        p.scalar_terms=mul(3,two.integral_elements); p.kernel_terms=two.kernel_terms; p.modeled_peak_bytes=two.modeled_peak_bytes;
    }
    const auto potential=attraction(mol);
    auto include=[&](const OneElectronRequest& request,const OneElectronDerivative& derivative,Count count) {
        const auto one=plan_one_electron_derivative(basis,basis,request,derivative,inner);
        p.shell_calls=add(p.shell_calls,mul(count,one.shell_calls));
        p.primitive_evaluations_upper_bound=add(p.primitive_evaluations_upper_bound,mul(count,one.primitive_evaluations_upper_bound));
        p.scalar_terms=add(p.scalar_terms,mul(count,square(p.n_basis)));
        p.kernel_terms=add(p.kernel_terms,mul(count,one.kernel_terms));
        p.modeled_peak_bytes=std::max(p.modeled_peak_bytes,one.modeled_peak_bytes);
    };
    for(const auto* operation:{"overlap","kinetic","potential"}) {
        OneElectronRequest request=operation==std::string("potential")?potential:OneElectronRequest{}; request.operation=operation;
        for(const auto* target:{"bra","ket"}) { OneElectronDerivative d; d.target=target; include(request,d,3); }
    }
    OneElectronDerivative charge; charge.target="charge"; include(potential,charge,mul(3,p.n_atoms));
    p.scalar_terms=add(p.scalar_terms,mul(3,square(p.n_atoms)));
    cap(p.shell_calls,budget.shell_calls,"MANTA HF gradient shell-call budget exceeded");
    cap(p.primitive_evaluations_upper_bound,budget.primitive_quartets,"MANTA HF gradient primitive-evaluation budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA HF gradient contraction-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA HF gradient kernel-work budget exceeded");
    validate_geometry(mol,basis);
    return p;
}
HFGradientResult compute_components(const Molecule& mol,const BasisSet& basis,
    const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta,const Eigen::MatrixXd& w,const ResourceBudget& budget,
    bool exact_j,bool exact_k) {
    const auto plan=plan_components(mol,basis,budget,exact_j,exact_k);
    validate_matrix(basis,alpha); validate_matrix(basis,beta); validate_matrix(basis,w);
    const Eigen::MatrixXd density=alpha+beta;
    HFGradientResult result;
    result.receipt.source_identity=source_identity(mol,basis);
    result.receipt.density_identity=density_identity(alpha,beta,w);
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    result.nuclear=result.one_electron=result.two_electron=result.pulay=Eigen::MatrixXd::Zero(plan.n_atoms,3);
    Eigen::MatrixXd one_error=result.one_electron,two_error=result.two_electron,pulay_error=result.pulay;
    const auto& atoms=mol.atoms();
    for(std::size_t i=0;i<atoms.size();++i) for(std::size_t j=0;j<i;++j) {
        if(!atoms[i].Z || !atoms[j].Z) continue;
        double r2=0.; for(int axis=0;axis<3;++axis) { const double d=atoms[i].xyz[axis]-atoms[j].xyz[axis]; r2+=d*d; }
        const double factor=-static_cast<double>(atoms[i].Z)*atoms[j].Z/(r2*std::sqrt(r2));
        for(int axis=0;axis<3;++axis) {
            const double value=factor*(atoms[i].xyz[axis]-atoms[j].xyz[axis]);
            result.nuclear(i,axis)+=value; result.nuclear(j,axis)-=value;
        }
    }
    const auto potential=attraction(mol);
    // Pople, Krishnan, Schlegel and Binkley, IJQC Symp. 13, 225 (1979):
    // dL = dE_nuc + D:dH + Gamma:dERI - W:dS.
    for(const auto* operation:{"overlap","kinetic","potential"}) {
        OneElectronRequest request=operation==std::string("potential")?potential:OneElectronRequest{}; request.operation=operation;
        for(const auto* target:{"bra","ket"}) for(int axis=0;axis<3;++axis) {
            OneElectronDerivative derivative; derivative.target=target; derivative.axis=axis;
            const auto component=compute_one_electron_derivative(basis,basis,request,derivative,budget);
            result.receipt.shell_calls+=component.receipt.shell_calls;
            result.receipt.primitive_integrals_evaluated+=component.receipt.primitive_integrals_evaluated;
            Eigen::Index start=0;
            for(std::size_t shell=0;shell<basis.nshells();++shell) {
                const int owner=basis.shell_atom_index(shell);
                const auto size=basis.libint()[shell].size();
                for(Eigen::Index local=0;local<static_cast<Eigen::Index>(size);++local) for(Eigen::Index other=0;other<density.rows();++other) {
                    const Eigen::Index row=derivative.target=="bra"?start+local:other,col=derivative.target=="bra"?other:start+local;
                    if(request.operation=="overlap") accumulate(-w(row,col)*component.values(row,col),result.pulay(owner,axis),pulay_error(owner,axis));
                    else accumulate(density(row,col)*component.values(row,col),result.one_electron(owner,axis),one_error(owner,axis));
                }
                start+=size;
            }
        }
    }
    for(std::size_t atom=0;atom<atoms.size();++atom) for(int axis=0;axis<3;++axis) {
        OneElectronDerivative derivative; derivative.target="charge"; derivative.axis=axis; derivative.charge_index=atom;
        const auto component=compute_one_electron_derivative(basis,basis,potential,derivative,budget);
        result.receipt.shell_calls+=component.receipt.shell_calls;
        result.receipt.primitive_integrals_evaluated+=component.receipt.primitive_integrals_evaluated;
        for(Eigen::Index i=0;i<density.rows();++i) for(Eigen::Index j=0;j<density.cols();++j)
            accumulate(density(i,j)*component.values(i,j),result.one_electron(atom,axis),one_error(atom,axis));
    }
    if(exact_j || exact_k) {
        DerivativeTiles provider({&basis,&basis,&basis,&basis},derivative_request(4),budget);
        std::vector<Eigen::Index> offsets(basis.nshells());
        for(std::size_t s=1;s<offsets.size();++s) offsets[s]=offsets[s-1]+basis.libint()[s-1].size();
        for(std::size_t a=0;a<basis.nshells();++a) for(std::size_t b=0;b<basis.nshells();++b)
        for(std::size_t c=0;c<basis.nshells();++c) for(std::size_t d=0;d<basis.nshells();++d) {
            const std::vector<std::size_t> indices{a,b,c,d};
            for(int center=0;center<4;++center) for(int axis=0;axis<3;++axis) {
                const auto tile=provider.evaluate(indices,center,axis);
                ++result.receipt.shell_calls; result.receipt.primitive_integrals_evaluated+=tile.primitive_evaluations;
                const int owner=basis.shell_atom_index(indices[center]);
                Count flat=0;
                for(Count i=0;i<tile.shape[0];++i) for(Count j=0;j<tile.shape[1];++j)
                for(Count k=0;k<tile.shape[2];++k) for(Count l=0;l<tile.shape[3];++l) {
                    const auto mu=offsets[a]+i,nu=offsets[b]+j,lambda=offsets[c]+k,sigma=offsets[d]+l;
                    const double coefficient=.5*((exact_j?density(mu,nu)*density(lambda,sigma):0.)
                        -(exact_k?alpha(mu,lambda)*alpha(nu,sigma):0.)-(exact_k?beta(mu,lambda)*beta(nu,sigma):0.));
                    accumulate(coefficient*tile.values.data()[flat++],result.two_electron(owner,axis),two_error(owner,axis));
                }
            }
        }
    }
    result.values=result.nuclear+result.one_electron+result.two_electron+result.pulay;
    if(!result.values.allFinite() || !result.nuclear.allFinite() || !result.one_electron.allFinite()
       || !result.two_electron.allFinite() || !result.pulay.allFinite())
        throw std::overflow_error("MANTA HF gradient nonfinite contraction");
    return result;
}
} // namespace
HFGradientPlan plan_hf_lagrangian_gradient(const Molecule& mol,const BasisSet& basis,const ResourceBudget& budget) {
    return plan_components(mol,basis,budget,true,true);
}
HFGradientResult compute_hf_lagrangian_gradient(const Molecule& mol,const BasisSet& basis,
    const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta,const Eigen::MatrixXd& w,const ResourceBudget& budget) {
    return compute_components(mol,basis,alpha,beta,w,budget,true,true);
}

HFGradientPlan plan_hf_ri_lagrangian_gradient(const Molecule& mol,const BasisSet& basis,const BasisSet& auxiliary,
    const RIOptions& options,bool fit_j,bool fit_k,const ResourceBudget& budget) {
    using namespace admission;
    if(!fit_j && !fit_k) throw std::invalid_argument("MANTA HF RI gradient requires at least one fitted component");
    auto p=plan_components(mol,basis,budget,!fit_j,!fit_k);
    auto include=[&](const RIJGradientPlan& q) {
        p.shell_calls=add(p.shell_calls,q.shell_calls);
        p.primitive_evaluations_upper_bound=add(p.primitive_evaluations_upper_bound,q.primitive_evaluations_upper_bound);
        p.scalar_terms=add(p.scalar_terms,q.scalar_terms); p.kernel_terms=add(p.kernel_terms,q.kernel_terms);
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,q.modeled_peak_bytes-budget.other_live_bytes);
    };
    if(fit_j) include(plan_ri_coulomb_gradient(mol,basis,auxiliary,options,{},budget));
    if(fit_k) include(plan_ri_exchange_gradient(mol,basis,auxiliary,options,{},budget));
    // Composition retains the exact/Pulay components and one additional RI
    // component result while the other producer executes. Count-only admission
    // precedes all input copies and generated integrals.
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(1024,p.n_atoms));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA HF RI gradient combined memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA HF RI gradient shell-call budget exceeded");
    cap(p.primitive_evaluations_upper_bound,budget.primitive_quartets,"MANTA HF RI gradient primitive-evaluation budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA HF RI gradient scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA HF RI gradient kernel-work budget exceeded");
    return p;
}
RIHFGradientResult compute_hf_ri_lagrangian_gradient(const Molecule& mol,const BasisSet& basis,const BasisSet& auxiliary,
    const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta,const Eigen::MatrixXd& w,
    const RIOptions& options,bool fit_j,bool fit_k,const ResourceBudget& budget) {
    const auto p=plan_hf_ri_lagrangian_gradient(mol,basis,auxiliary,options,fit_j,fit_k,budget);
    validate_matrix(basis,alpha); validate_matrix(basis,beta); validate_matrix(basis,w);
    RIHFGradientResult result;
    static_cast<HFGradientResult&>(result)=compute_components(mol,basis,alpha,beta,w,budget,!fit_j,!fit_k);
    result.coulomb=fit_j?Eigen::MatrixXd::Zero(p.n_atoms,3):result.two_electron;
    result.exchange=fit_k?Eigen::MatrixXd::Zero(p.n_atoms,3):result.two_electron;
    std::string source="manta.hf-ri-lagrangian-gradient.v1;"+result.receipt.source_identity
        +";J="+(fit_j?"ri":"exact")+";K="+(fit_k?"ri":"exact");
    auto include=[&](const auto& r) {
        result.receipt.shell_calls+=r.metric_shell_calls+r.coupling_shell_calls+r.metric_derivative_shell_calls+r.coupling_derivative_shell_calls;
        result.receipt.primitive_integrals_evaluated+=r.primitive_integrals_evaluated;
        source+=";"+r.source_identity;
    };
    if(fit_j) {
        auto component=compute_ri_coulomb_gradient(mol,basis,auxiliary,alpha+beta,options,{},budget);
        include(component.receipt);
        result.coulomb=std::move(component.values); result.coulomb_ri=std::move(component.receipt);
        result.coulomb_metric=std::move(component.metric_response);
    }
    if(fit_k) {
        auto component=compute_ri_exchange_gradient(mol,basis,auxiliary,alpha,beta,options,{},budget);
        include(component.receipt);
        result.exchange=std::move(component.values); result.exchange_ri=std::move(component.receipt);
        result.exchange_metric=std::move(component.metric_response);
    }
    result.two_electron=result.coulomb+result.exchange;
    result.values=result.nuclear+result.one_electron+result.two_electron+result.pulay;
    result.receipt.algorithm="composed-ri-hf-lagrangian";
    result.receipt.modeled_peak_bytes=p.modeled_peak_bytes;
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(source.data()),source.size());
    result.receipt.source_identity=hash.finish_hex();
    if(!result.values.allFinite() || !result.two_electron.allFinite())
        throw std::overflow_error("MANTA HF RI gradient nonfinite composition");
    return result;
}
} // namespace vibeqc::manta
