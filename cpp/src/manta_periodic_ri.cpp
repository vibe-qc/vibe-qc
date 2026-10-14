#include "vibeqc/manta_periodic_ri.hpp"
#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <Eigen/Eigenvalues>
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double v) { if(v==0)v=0.;Count bits;std::memcpy(&bits,&v,8);word(bits); }
    void value(Complex z) { real(z.real());real(z.imag()); }
    void matrix(const Eigen::MatrixXcd& m) { word(m.rows());word(m.cols());for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)value(m(i,j)); }
};
std::string matrix_id(const Eigen::MatrixXcd& m) { Digest d;d.text("manta.periodic-ri-matrix.v1");d.matrix(m);return d.hash.finish_hex(); }
std::string frame_id(const PeriodicGDFTiles& s,bool target) {
    Digest d;d.text("manta.periodic-ri-frame.v1");d.text(s.source().short_range.orbital_basis_identity);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)d.real(s.lattice()(i,j));
    d.word(s.momenta().rows());
    for(Eigen::Index k=0;k<s.momenta().rows();++k)for(int i=0;i<3;++i)d.real(s.momenta()(k,i)-(target?s.transfer()[i]:0.));
    return d.hash.finish_hex();
}
const PeriodicGDFTiles& require_source(const std::shared_ptr<const PeriodicGDFTiles>& s) {
    if(!s) throw std::invalid_argument("MANTA periodic RI requires an immutable GDF tile source");return *s;
}
void add_compensated(Complex v,Complex& sum,Complex& correction) {
    if(!std::isfinite(v.real()) || !std::isfinite(v.imag())) throw std::overflow_error("MANTA periodic RI contribution is nonfinite");
    const auto y=v-correction,t=sum+y;correction=(t-sum)-y;sum=t;
    if(!std::isfinite(sum.real()) || !std::isfinite(sum.imag())) throw std::overflow_error("MANTA periodic RI sum is nonfinite");
}
double hermiticity(const Complex* values,Count n) {
    double scale=1.,error=0.;
    for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
        const auto v=values[i*n+j];
        if(!std::isfinite(v.real()) || !std::isfinite(v.imag())) throw std::invalid_argument("MANTA periodic RI matrix must be finite");
        scale=std::max(scale,std::abs(v));error=std::max(error,std::abs(v-std::conj(values[j*n+i])));
    }
    if(!std::isfinite(scale) || !std::isfinite(error))
        throw std::overflow_error("MANTA periodic RI Hermiticity norm is not representable");
    return error/scale;
}
}

PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTiles& source,Count spins,const RIOptions& o,const ResourceBudget& budget) {
    if(source.source().metric_source_identity.empty())
        throw std::invalid_argument("MANTA periodic RI requires an admitted metric from the same source");
    return plan_periodic_ri_q(source.plan(),spins,o,budget);
}
PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTilesPlan& s,Count spins,const RIOptions& o,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);ri_detail::validate_options(o);
    if(spins!=1 && spins!=2) throw std::invalid_argument("MANTA periodic RI requires one or two spin channels");
    if(!s.compute_metric || !s.n_aux || !s.n_orb || !s.momenta)
        throw std::invalid_argument("MANTA periodic RI requires an admitted metric from the same source");
    PeriodicRIQPlan p;p.n_basis=s.n_orb;p.n_auxiliary=s.n_aux;p.momenta=s.momenta;p.spins=spins;
    const Count n=p.n_basis,m=p.n_auxiliary,nk=p.momenta,n2=square(n),m2=square(m);
    p.maximum_rank=m;p.mode_block=std::min(o.mode_block,m);p.maximum_coupling_passes=add(m,p.mode_block-1)/p.mode_block;
    const Count metric_bytes=mul(16,add(mul(24,m2),mul(64,m)));
    const Count build_bytes=mul(16,add(add(mul(8,m2),mul(64,m)),
        mul(nk,mul(n2,add(add(mul(12,spins),8),mul(4,p.mode_block))))));
    // The source's caller-owned reservation belongs to its construction,
    // not this consumer. Retain its immutable owners/workspaces once and
    // charge this call's external live storage separately.
    const Count construction=add(s.metric_peak_bytes-s.external_live_bytes,metric_bytes);
    const Count build=add(s.coupling_peak_bytes-s.external_live_bytes,build_bytes);
    p.modeled_peak_bytes=add(std::max(construction,build),add(budget.other_live_bytes,budget.backend_margin_bytes));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic RI metric/block memory budget exceeded");
    p.shell_calls=add(s.metric_shell_calls,mul(p.maximum_coupling_passes,s.coupling_shell_calls));
    p.primitive_products=add(s.metric_primitive_products,mul(p.maximum_coupling_passes,s.coupling_primitive_products));
    const Count cubic=mul(m,m2),whiten=mul(m2,mul(nk,n2)),exchange=mul(spins,mul(m,mul(nk,mul(n,n2))));
    const Count scalar=mul(128,add(cubic,add(whiten,add(exchange,mul(m,mul(nk,n2))))));
    p.scalar_terms=add(s.metric_scalar_terms,add(mul(p.maximum_coupling_passes,s.coupling_scalar_terms),scalar));
    p.kernel_terms=add(s.metric_kernel_terms,add(mul(p.maximum_coupling_passes,s.coupling_kernel_terms),
        add(mul(8,scalar),mul(4096,cubic))));
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic RI shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA periodic RI primitive-product budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic RI scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic RI kernel-work budget exceeded");
    return p;
}

PeriodicRIQBuilder::PeriodicRIQBuilder(std::shared_ptr<const PeriodicGDFTiles> source,Count spins,
    const RIOptions& options,const ResourceBudget& budget)
    : plan_(plan_periodic_ri_q(require_source(source),spins,options,budget)),source_(std::move(source)),options_(options) {
    const auto raw=source_->metric();const Count m=plan_.n_auxiliary;
    info_.metric_source_identity=source_->source().metric_source_identity;
    info_.zero_transfer=source_->source().zero_transfer;
    info_.shell_calls=raw.receipt.shell_calls;info_.primitive_products=raw.receipt.primitive_products;
    info_.hermiticity_residual=hermiticity(raw.values.data(),m);
    if(info_.hermiticity_residual>1e-12) throw std::runtime_error("MANTA periodic RI metric is not Hermitian");
    metric_.resize(m,m);
    for(Count i=0;i<m;++i) {
        metric_(i,i)=raw.values[i*m+i].real();
        for(Count j=0;j<i;++j) { metric_(i,j)=raw.values[i*m+j];metric_(j,i)=std::conj(metric_(i,j)); }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigen(metric_,Eigen::ComputeEigenvectors);
    if(eigen.info()!=Eigen::Success || !eigen.eigenvalues().allFinite() || !eigen.eigenvectors().allFinite())
        throw std::runtime_error("MANTA periodic RI metric eigensolve failed");
    info_.minimum_eigenvalue=eigen.eigenvalues()[0];info_.maximum_eigenvalue=eigen.eigenvalues()[m-1];
    const double positive=std::max(0.,info_.maximum_eigenvalue);
    info_.cutoff=std::max(options.absolute_cutoff,options.relative_cutoff*positive);
    info_.negative_limit=std::max(options.absolute_cutoff,options.negative_relative_tolerance*positive);
    if(!std::isfinite(info_.cutoff) || !std::isfinite(info_.negative_limit)) throw std::overflow_error("MANTA periodic RI metric cutoff overflow");
    if(info_.minimum_eigenvalue < -info_.negative_limit)
        throw std::runtime_error("MANTA periodic RI metric has a significant negative eigenvalue; converge the finite source");
    for(Count i=0;i<m;++i) if(eigen.eigenvalues()[i]>info_.cutoff) ++info_.rank;
    factor_.resize(m,info_.rank);Count column=0;
    for(Count i=0;i<m;++i) if(eigen.eigenvalues()[i]>info_.cutoff)
        factor_.col(column++)=eigen.eigenvectors().col(i)/std::sqrt(eigen.eigenvalues()[i]);
    projector_=Eigen::MatrixXcd::Zero(m,m);
    if(info_.rank) projector_=eigen.eigenvectors().rightCols(info_.rank)*eigen.eigenvectors().rightCols(info_.rank).adjoint();
    if(!factor_.allFinite() || !projector_.allFinite()) throw std::overflow_error("MANTA periodic RI metric factor is nonfinite");
    info_.canonical_metric_identity=matrix_id(metric_);info_.factor_identity=matrix_id(factor_);info_.projector_identity=matrix_id(projector_);
    Digest d;d.text("manta.periodic-ri-q.v1;fixed-positive-eigenprojector;lower-triangle");
    d.text(source_->source().source_identity);d.text(info_.canonical_metric_identity);d.text(info_.factor_identity);d.text(info_.projector_identity);
    d.real(options.absolute_cutoff);d.real(options.relative_cutoff);d.real(options.negative_relative_tolerance);
    info_.source_identity=d.hash.finish_hex();source_frame_=frame_id(*source_,false);target_frame_=frame_id(*source_,true);
}

void PeriodicRIQBuilder::validate_request(bool make_j,bool make_k) const {
    if(!make_j && !make_k) throw std::invalid_argument("MANTA periodic RI requires a Coulomb or exchange target");
    if(make_j && !info_.zero_transfer) throw std::invalid_argument("MANTA periodic RI Coulomb requires a reciprocal Gamma transfer");
}
PeriodicRIQResult PeriodicRIQBuilder::build(const std::vector<Complex>& d,const std::vector<double>& weights,
    bool make_j,bool make_k) const {
    using namespace admission;
    validate_request(make_j,make_k);
    const Count n=plan_.n_basis,m=plan_.n_auxiliary,nk=plan_.momenta,ns=plan_.spins,n2=square(n),mesh=mul(nk,n2);
    if(d.size()!=mul(ns,mesh) || weights.size()!=nk) throw std::invalid_argument("MANTA periodic RI density/weight dimensions");
    for(Count spin=0;spin<ns;++spin)for(Count k=0;k<nk;++k)
        if(hermiticity(d.data()+(spin*nk+k)*n2,n)>1e-12) throw std::invalid_argument("MANTA periodic RI requires Hermitian spin densities");
    double weight_sum=0.;
    for(double w:weights) { require_finite(w);if(w<0.) throw std::invalid_argument("MANTA periodic RI source weights must be nonnegative");weight_sum+=w; }
    if(!std::isfinite(weight_sum) || weight_sum>1.+1e-12 || (make_j && std::abs(weight_sum-1.)>1e-12))
        throw std::invalid_argument("MANTA periodic RI weights must sum to one for J, or at most one for a partial K channel");
    PeriodicRIQResult result;result.n_basis=n;result.momenta=nk;result.spins=ns;auto& r=result.receipt;
    r.kernel=source_->source().kernel;
    r.physical_omega=source_->source().physical_omega;r.zero_transfer=source_->source().zero_transfer;
    r.physical_zero_included=source_->source().physical_zero_included;
    r.has_j=make_j;r.has_k=make_k;r.source_identity=info_.source_identity;r.projector_identity=info_.projector_identity;
    r.source_frame_identity=source_frame_;r.target_frame_identity=target_frame_;r.retained_rank=info_.rank;r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    Digest density;density.text("manta.periodic-ri-density.v1");density.text(source_frame_);density.word(ns);density.word(nk);density.word(n);
    for(auto v:d)density.value(v);r.density_identity=density.hash.finish_hex();
    Digest weight;weight.text("manta.periodic-ri-weights.v1");weight.word(nk);for(double w:weights)weight.real(w);r.weight_identity=weight.hash.finish_hex();
    Digest snapshot;snapshot.text("manta.periodic-ri-q-snapshot.v1");snapshot.text(r.source_identity);snapshot.text(r.density_identity);
    snapshot.text(r.weight_identity);snapshot.word(make_j);snapshot.word(make_k);r.snapshot_identity=snapshot.hash.finish_hex();
    if(make_j) result.J.assign(mesh,{});if(make_k) result.K.assign(mul(ns,mesh),{});
    std::vector<Complex> j_error(result.J.size()),k_error(result.K.size()),total(mesh),rhs(m),rhs_error(m),rho(info_.rank);
    for(Count spin=0;spin<ns;++spin)for(Count i=0;i<mesh;++i)total[i]+=d[spin*mesh+i];
    const Count passes_rank=make_j?std::max<Count>(1,info_.rank):info_.rank;
    for(Count first=0;first<passes_rank;first+=plan_.mode_block) {
        const Count modes=std::min(plan_.mode_block,info_.rank-first);
        std::vector<Complex> fitted(mul(modes,mesh)),fitted_error(fitted.size());
        ++r.coupling_passes;
        for(Count ip=0;ip<source_->auxiliary().nshells();++ip)for(Count im=0;im<source_->orbital().nshells();++im)
            for(Count in=0;in<source_->orbital().nshells();++in) {
                const auto tile=source_->evaluate(ip,im,in);
                r.shell_calls=admission::add(r.shell_calls,tile.receipt.shell_calls);
                r.primitive_products=admission::add(r.primitive_products,tile.receipt.primitive_products);
                const Count np=tile.shape[1],nm=tile.shape[2],nn=tile.shape[3],p0=tile.offsets[1],m0=tile.offsets[2],n0=tile.offsets[3];
                for(Count k=0;k<nk;++k)for(Count p=0;p<np;++p)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j) {
                    const Count pair=(m0+i)*n+n0+j;const auto value=tile.values[((k*np+p)*nm+i)*nn+j];
                    if(make_j && first==0) add_compensated(weights[k]*std::conj(value)*total[k*n2+pair],rhs[p0+p],rhs_error[p0+p]);
                    for(Count mode=0;mode<modes;++mode) {
                        const Count out=(k*modes+mode)*n2+pair;
                        add_compensated(std::conj(factor_(p0+p,first+mode))*value,fitted[out],fitted_error[out]);
                    }
                }
            }
        for(Count mode=0;mode<modes;++mode) {
            if(make_j) {
                Complex value{},error{};
                for(Count k=0;k<nk;++k)for(Count pair=0;pair<n2;++pair)
                    add_compensated(weights[k]*std::conj(fitted[(k*modes+mode)*n2+pair])*total[k*n2+pair],value,error);
                rho[first+mode]=value;r.coulomb_energy+=.5*std::norm(value);
                for(Count k=0;k<nk;++k)for(Count pair=0;pair<n2;++pair)
                    add_compensated(fitted[(k*modes+mode)*n2+pair]*value,result.J[k*n2+pair],j_error[k*n2+pair]);
            }
            if(make_k) for(Count k=0;k<nk;++k)for(Count spin=0;spin<ns;++spin) {
                const Eigen::Map<const md::ComplexMatrix> B(fitted.data()+(k*modes+mode)*n2,n,n);
                const Eigen::Map<const md::ComplexMatrix> D(d.data()+(spin*nk+k)*n2,n,n);
                const md::ComplexMatrix contribution=(B*D)*B.adjoint();
                for(Count pair=0;pair<n2;++pair) {
                    const Count out=(spin*nk+k)*n2+pair;
                    add_compensated(weights[k]*contribution.data()[pair],result.K[out],k_error[out]);
                }
            }
        }
    }
    if(make_j) {
        const Eigen::Map<const Eigen::VectorXcd> raw_rhs(rhs.data(),m),projected(rho.data(),info_.rank);
        const Eigen::VectorXcd solution=factor_.conjugate()*projected;
        r.metric_rhs_norm=raw_rhs.norm();r.metric_residual_norm=(metric_.conjugate()*solution-raw_rhs).norm();r.metric_residual_evaluated=true;
        require_finite(r.coulomb_energy);require_finite(r.metric_rhs_norm);require_finite(r.metric_residual_norm);
        for(Count k=0;k<nk;++k)r.hermiticity_residual=std::max(r.hermiticity_residual,hermiticity(result.J.data()+k*n2,n));
        if(r.hermiticity_residual>1e-10) throw std::runtime_error("MANTA periodic RI Coulomb source is not Hermitian; qualify the reciprocal domain");
    }
    if(make_k) for(Count spin=0;spin<ns;++spin)for(Count k=0;k<nk;++k)
        r.hermiticity_residual=std::max(r.hermiticity_residual,hermiticity(result.K.data()+(spin*nk+k)*n2,n));
    if(r.hermiticity_residual>1e-10) throw std::runtime_error("MANTA periodic RI response is not Hermitian");
    cap(r.coupling_passes,plan_.maximum_coupling_passes,"MANTA periodic RI executed pass bound exceeded");
    cap(add(info_.shell_calls,r.shell_calls),plan_.shell_calls,"MANTA periodic RI executed shell bound exceeded");
    cap(add(info_.primitive_products,r.primitive_products),plan_.primitive_products,"MANTA periodic RI executed primitive bound exceeded");
    return result;
}
} // namespace vibeqc::manta
