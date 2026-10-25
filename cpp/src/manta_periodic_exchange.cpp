#include "vibeqc/manta_periodic_exchange.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <algorithm>
#include <cstring>
#include "manta_ri_interval.hpp"

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
using Matrix=Eigen::Matrix<Complex,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using Index=std::array<std::int64_t,3>;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) {
        std::array<std::uint8_t,8> b{}; for(int i=0;i<8;++i) b[i]=v>>(56-8*i);
        hash.update(b.data(),b.size());
    }
    void text(const std::string& s) { word(s.size()); hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double x) { if(x==0) x=0.; Count bits; std::memcpy(&bits,&x,8); word(bits); }
    void matrix(const md::Matrix& m) {
        word(m.rows()); word(m.cols());
        for(Eigen::Index i=0;i<m.rows();++i) for(Eigen::Index j=0;j<m.cols();++j) real(m(i,j));
    }
};
void finite(Complex z) {
    if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument("MANTA periodic exchange requires finite numerical values");
}
void accumulate(Complex x,Complex& sum,Complex& error) {
    const auto y=x-error,next=sum+y; error=(next-sum)-y; sum=next;
}
Index label(const Eigen::Vector3d& fractional) {
    Index result{};
    for(int d=0;d<3;++d) {
        const double x=fractional[d],nearest=std::round(x);
        if(!std::isfinite(x) || std::abs(x)>0x1p40 || std::abs(x-nearest)>1e-10)
            throw std::invalid_argument("MANTA periodic exchange support has an invalid lattice or transfer label");
        result[d]=static_cast<std::int64_t>(nearest);
    }
    return result;
}
void unique(std::vector<Index>& labels) {
    std::sort(labels.begin(),labels.end());
    for(std::size_t i=1;i<labels.size();++i) if(labels[i]==labels[i-1])
        throw std::invalid_argument("MANTA periodic exchange support contains duplicate vectors");
}
void paired(const std::vector<Index>& a,const std::vector<Index>& b) {
    if(a.size()!=b.size()) throw std::invalid_argument("MANTA periodic exchange reverse transfer supports differ");
    for(const auto& i:a) {
        const Index negative{-i[0],-i[1],-i[2]};
        if(!std::binary_search(b.begin(),b.end(),negative))
            throw std::invalid_argument("MANTA periodic exchange supports must be closed under transfer reversal");
    }
}
ResourceBudget unlimited(const ResourceBudget& budget) {
    auto leaf=budget;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    leaf.other_live_bytes=0; return leaf;
}
void validate_kernel(const std::string& kernel,double omega) {
    if((kernel!="coulomb" && kernel!="erf" && kernel!="erfc") || !std::isfinite(omega) || omega<0
       || (kernel=="coulomb" && omega!=0))
        throw std::invalid_argument("MANTA periodic exchange kernel must be coulomb with omega=0, or erf/erfc with finite omega>=0");
}
double fourier_kernel(double norm,const std::string& kernel,double omega) {
    // Fourier transform of erf(omega*r)/r and erfc(omega*r)/r, using
    // the Gaussian transform convention in Sun et al., JCP 147,164119 (2017).
    const double pi=std::acos(-1.);
    if(kernel=="coulomb" || omega==0) return kernel=="erf" ? 0. : 4*pi/norm;
    if(norm==0) return (pi/omega)/omega; // admitted only for positive-range erfc
    const double x=(.5*std::sqrt(norm))/omega,t=x*x;
    if(kernel=="erf") return std::exp(std::log(4*pi)-std::log(norm)-t);
    // Avoid both cancellation in 1-exp(-t) and underflow of t at large
    // omega. This retains the small positive erfc tail independently.
    if(t<=1) return ((pi/omega)/omega)*(t==0 ? 1. : -std::expm1(-t)/t);
    return (4*pi/norm)*(-std::expm1(-t));
}
// Same-source stored-panel contraction; the Fourier/kernel construction is
// not enclosed here. Rump, Acta Numerica 19 (2010), 287-449,
// DOI 10.1017/S096249291000005X; shared outward operations, no BLAS error model.
class ExchangeArithmetic {
    using CI=spectral_interval::ComplexInterval;
    Count n_,nk_,nd_;
    std::vector<CI> sum_,temporary_;
public:
    ExchangeArithmetic(Count n,Count nk,Count nd):n_(n),nk_(nk),nd_(nd),sum_(nd*nk*n*n),temporary_(n*n) {
        spectral_interval::require_arithmetic();
    }
    void panel(const Complex* b,const Complex* d,Count target,Count state,double factor) {
        using namespace spectral_interval;
        for(Count i=0;i<n_;++i)for(Count j=0;j<n_;++j) {
            CI value{};
            for(Count a=0;a<n_;++a)value=add(value,product(d[i*n_+a],b[a*n_+j]));
            temporary_[i*n_+j]=value;
        }
        for(Count i=0;i<n_;++i)for(Count j=0;j<n_;++j) {
            CI value{};
            for(Count a=0;a<n_;++a)value=add(value,product(point(std::conj(b[a*n_+i])),temporary_[a*n_+j]));
            const Count out=(state*nk_+target)*n_*n_+i*n_+j;
            sum_[out]=add(sum_[out],product(point(factor),value));
        }
    }
    void finish(PeriodicExchangeResult& result,const std::vector<Complex>& d,const std::vector<double>& weights) const {
        using namespace spectral_interval;
        auto& r=result.receipt;r.roundoff_evaluated=true;result.roundoff_bounds.resize(sum_.size());
        for(Count q=0;q<sum_.size();++q) {
            const double error=magnitude(sub(point(result.values[q]),sum_[q]));
            result.roundoff_bounds[q]=error;r.roundoff_bound=std::max(r.roundoff_bound,error);
        }
        r.energy_roundoff_bounds.resize(nd_);
        for(Count state=0;state<nd_;++state) {
            Interval energy{};
            for(Count k=0;k<nk_;++k)for(Count i=0;i<n_;++i)for(Count j=0;j<n_;++j) {
                const Count base=(state*nk_+k)*n_*n_;
                const auto weighted=product(point(weights[k]),point(d[base+j*n_+i]));
                energy=add(energy,product(point(-.5),product(weighted,sum_[base+i*n_+j])).real);
            }
            const auto error=sub(point(r.energies[state]).real,energy);
            r.energy_roundoff_bounds[state]=std::max(std::abs(error.lo),std::abs(error.hi));
        }
    }
};
} // namespace
PeriodicExchangePlan plan_periodic_exchange(const BasisSet& basis,Count nq,Count nc,Count nk,Count nd,
    Count batch,std::optional<double> cutoff,const ResourceBudget& budget,const std::string& kernel,double omega,bool audit_roundoff) {
    using namespace admission;
    validate_kernel(kernel,omega);
    margin(budget);
    if(!nk || !nd || !batch) throw std::invalid_argument("MANTA periodic exchange requires positive momentum, density and batch counts");
    PeriodicExchangePlan p;p.roundoff_requested=audit_roundoff;
    p.n_basis=basis.nbasis(); p.momenta=nk; p.densities=nd; p.cells=nc;
    p.vectors_per_pair=nq; p.transfer_pairs=square(nk); p.reciprocal_batch=batch;
    const Count n2=square(p.n_basis),elements=mul(nd,mul(nk,n2)),vectors=mul(p.transfer_pairs,nq);
    cap(elements,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic exchange output is not addressable");
    for(auto count:{vectors,nc,nk}) cap(count,std::numeric_limits<Eigen::Index>::max()/3,"MANTA periodic exchange vectors are not addressable");
    cap(add(p.transfer_pairs,1),std::numeric_limits<Eigen::Index>::max(),"MANTA periodic exchange offsets are not addressable");
    p.output_bytes=mul(16,elements);
    // Caller arrays + snapshots, response/compensation/exports, per-transfer
    // integer-label validation and three dense contraction operands plus
    // conservative Eigen multiplication scratch. No four-index AO tensor.
    const Count inputs=add(72,add(mul(24,add(vectors,add(nc,nk))),
        add(mul(8,add(nk,add(p.transfer_pairs,1))),mul(16,elements))));
    p.driver_live_bytes=add(mul(2,inputs),add(mul(3,p.output_bytes),
        add(mul(64,add(vectors,add(nc,p.transfer_pairs))),add(mul(128,n2),add(mul(256,nd),4096)))));
    const Count cube=mul(n2,p.n_basis);
    p.matrix_products=mul(2,mul(nd,vectors));
    p.contraction_terms=add(mul(p.matrix_products,cube),elements);
    p.scalar_terms=add(mul(8,p.contraction_terms),add(mul(1024,add(vectors,add(nc,p.transfer_pairs))),mul(8,elements)));
    if(audit_roundoff) {
        p.driver_live_bytes=add(p.driver_live_bytes,add(4096,add(mul(128,elements),mul(128,n2))));
        p.scalar_terms=add(p.scalar_terms,mul(1024,p.contraction_terms));
    }
    cap(p.driver_live_bytes,budget.memory_bytes,"MANTA periodic exchange driver memory budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic exchange scalar-work budget exceeded");
    auto leaf=unlimited(budget);
    Count peak=0;
    auto include=[&](Count count,Count repeats) {
        if(!repeats) return;
        const auto f=plan_bloch_fourier(basis,count,nc,1,cutoff,leaf);
        const Count calls=mul(repeats,p.transfer_pairs);
        p.shell_calls=add(p.shell_calls,mul(calls,f.shell_calls));
        p.primitive_pairs=add(p.primitive_pairs,mul(calls,f.primitive_pairs));
        p.vector_primitive_products=add(p.vector_primitive_products,mul(calls,f.vector_primitive_products));
        p.phase_products=add(p.phase_products,mul(calls,f.phase_products));
        p.phase_accumulations=add(p.phase_accumulations,mul(calls,f.phase_accumulations));
        p.scalar_terms=add(p.scalar_terms,mul(calls,f.scalar_terms));
        p.kernel_terms=add(p.kernel_terms,mul(calls,f.kernel_terms));
        peak=std::max(peak,f.modeled_peak_bytes);
    };
    include(batch,nq/batch); include(nq%batch,nq%batch!=0);
    if(!nq) peak=plan_bloch_fourier(basis,0,nc,1,cutoff,leaf).modeled_peak_bytes;
    p.kernel_terms=add(p.kernel_terms,mul(32,p.scalar_terms));
    p.modeled_peak_bytes=add(peak,add(p.driver_live_bytes,budget.other_live_bytes));
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic exchange shell-call budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic exchange primitive-pair budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic exchange scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic exchange kernel-work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic exchange memory budget exceeded");
    return p;
}

PeriodicExchangeResult compute_periodic_exchange(const BasisSet& basis,const md::Matrix& lattice,
    const md::Matrix& vectors,const std::vector<Count>& offsets,const md::Matrix& cells,const md::Matrix& k,
    const std::vector<double>& weights,const std::vector<Complex>& densities,Count nd,Count batch,
    std::optional<double> cutoff,const ResourceBudget& budget,const std::string& kernel,double omega,bool audit_roundoff) {
    using namespace admission;
    validate_kernel(kernel,omega);
    if(lattice.rows()!=3 || lattice.cols()!=3 || vectors.cols()!=3 || cells.cols()!=3 || k.cols()!=3)
        throw std::invalid_argument("MANTA periodic exchange requires a 3x3 lattice and (count,3) vectors");
    const Count nk=k.rows(),pairs=square(nk);
    if(offsets.size()!=add(pairs,1) || offsets.front()!=0 || offsets.back()!=static_cast<Count>(vectors.rows()))
        throw std::invalid_argument("MANTA periodic exchange offsets must delimit every target/source pair");
    Count maximum=0;
    for(Count pair=0;pair<pairs;++pair) {
        if(offsets[pair]>offsets[pair+1]) throw std::invalid_argument("MANTA periodic exchange offsets must be monotone");
        maximum=std::max(maximum,offsets[pair+1]-offsets[pair]);
    }
    const auto plan=plan_periodic_exchange(basis,maximum,cells.rows(),nk,nd,batch,cutoff,budget,kernel,omega,audit_roundoff);
    const Count n=plan.n_basis,n2=n*n;
    if(weights.size()!=nk || densities.size()!=nd*nk*n2)
        throw std::invalid_argument("MANTA periodic exchange weights/densities have incompatible shapes");
    if(!lattice.allFinite() || !vectors.allFinite() || !cells.allFinite() || !k.allFinite())
        throw std::invalid_argument("MANTA periodic exchange vectors must be finite");
    const Eigen::Matrix3d a=lattice;
    const double volume=std::abs(a.determinant());
    if(!std::isfinite(volume) || volume<=0 || !a.inverse().allFinite())
        throw std::invalid_argument("MANTA periodic exchange lattice must be finite and nonsingular");
    std::vector<Index> cell_labels; cell_labels.reserve(cells.rows());
    for(Eigen::Index i=0;i<cells.rows();++i) cell_labels.push_back(label(a.inverse()*cells.row(i).transpose()));
    unique(cell_labels); paired(cell_labels,cell_labels);
    std::vector<std::vector<Index>> labels(pairs);
    const Eigen::Matrix3d reciprocal_inverse=a.transpose()/(2*std::acos(-1.));
    for(Count target=0;target<nk;++target) for(Count source=0;source<nk;++source) {
        const Count pair=target*nk+source; auto& current=labels[pair];
        current.reserve(offsets[pair+1]-offsets[pair]);
        for(Count iq=offsets[pair];iq<offsets[pair+1];++iq) {
            const auto q=vectors.row(iq);
            const double norm=q.squaredNorm();
            if(!std::isfinite(norm) || (norm==0 && !(q.isZero(0) && kernel=="erfc" && omega>0)))
                throw std::invalid_argument("MANTA periodic exchange requires finite positive Q squared norm; exact Q=0 is admitted only for erfc with omega>0");
            require_finite(fourier_kernel(norm,kernel,omega));
            const Eigen::Vector3d delta=(q-k.row(target)+k.row(source)).transpose();
            current.push_back(label(reciprocal_inverse*delta));
        }
        unique(current);
    }
    for(Count target=0;target<nk;++target) for(Count source=0;source<nk;++source)
        paired(labels[target*nk+source],labels[source*nk+target]);
    double weight_sum=0;
    for(auto w:weights) { require_finite(w); if(w<0) throw std::invalid_argument("MANTA periodic exchange weights must be nonnegative"); weight_sum+=w; }
    if(!std::isfinite(weight_sum) || std::abs(weight_sum-1)>1e-12)
        throw std::invalid_argument("MANTA periodic exchange weights must sum to one");
    PeriodicExchangeResult result; result.plan=plan;
    auto& r=result.receipt; r.cell_volume=volume; r.modeled_peak_bytes=plan.modeled_peak_bytes;
    r.kernel=kernel; r.omega=omega;
    if(kernel=="erf") r.boundary="erf-q0-omitted-no-madelung";
    if(kernel=="erfc") r.boundary=omega>0 ? "erfc-finite-q0-on-explicit-support-no-madelung" : "erfc-zero-range-q0-omitted-no-madelung";
    for(Count state=0;state<nd;++state) for(Count ik=0;ik<nk;++ik) {
        double scale=0,residual_max=0;
        const Count start=(state*nk+ik)*n2;
        for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
            const auto v=densities[start+i*n+j]; finite(v);
            const double magnitude=std::abs(v),residual=std::abs(v-std::conj(densities[start+j*n+i]));
            require_finite(magnitude); require_finite(residual); scale=std::max(scale,magnitude);
            residual_max=std::max(residual_max,residual);
        }
        r.density_hermiticity_residual=std::max(r.density_hermiticity_residual,residual_max);
        if(residual_max>1e-12*std::max(1.,scale))
            throw std::invalid_argument("MANTA periodic exchange density blocks must be Hermitian");
    }
    Digest digest; digest.text("manta.periodic-exchange.v1;md-fp64-v1;libint-shell-axial-v1;full-coulomb-q0-omitted-no-madelung");
    digest.text(auxiliary_basis_content_identity_sha256(basis)); digest.matrix(lattice); digest.matrix(cells);
    digest.matrix(k); digest.matrix(vectors); for(auto offset:offsets) digest.word(offset);
    for(auto w:weights) digest.real(w);
    digest.word(cutoff.has_value()); if(cutoff) digest.real(*cutoff);
    // Preserve the established default full-Coulomb source identity bytes.
    if(kernel!="coulomb") { digest.text("manta.periodic-exchange.range.v1"); digest.text(kernel); digest.real(omega); }
    r.source_identity=digest.hash.finish_hex();
    Digest epoch; epoch.text("manta.periodic-exchange.epoch.v1"); epoch.text(r.source_identity); epoch.word(nd);
    for(Count state=0;state<nd;++state) {
        Digest density; density.text("manta.periodic-exchange.density.v1"); density.text(r.source_identity);
        for(Count i=0;i<nk*n2;++i) { const auto v=densities[state*nk*n2+i]; density.real(v.real()); density.real(v.imag()); }
        r.density_identities.push_back(density.hash.finish_hex()); epoch.text(r.density_identities.back());
    }
    r.density_identity=epoch.hash.finish_hex();
    result.values.resize(nd*nk*n2); std::vector<Complex> compensation(result.values.size());
    ResourceBudget leaf=budget; leaf.other_live_bytes=add(budget.other_live_bytes,plan.driver_live_bytes);
    Matrix b(n,n),temporary(n,n),response(n,n);
    std::optional<ExchangeArithmetic> arithmetic;
    if(audit_roundoff)arithmetic.emplace(n,nk,nd);
    // Sun et al., JCP 147,164119 (2017), Gaussian-pair Fourier convention.
    // B rows are the source-momentum bra, columns the target-momentum ket.
    // Each bounded Fourier panel is digested into every density channel.
    for(Count target=0;target<nk;++target) for(Count source=0;source<nk;++source) {
        const Count pair=target*nk+source;
        const md::Matrix momentum=k.middleRows(target,1);
        for(Count start=offsets[pair];start<offsets[pair+1];) {
            const Count count=std::min(batch,offsets[pair+1]-start);
            const md::Matrix q=vectors.middleRows(start,count);
            const auto panel=compute_bloch_fourier_panel(basis,q,cells,momentum,cutoff,leaf);
            ++r.panel_calls; r.shell_calls+=panel.receipt.shell_calls; r.skipped_shell_pairs+=panel.receipt.skipped_shell_pairs;
            r.primitive_pairs+=panel.receipt.primitive_pairs; r.vector_primitive_products+=panel.receipt.vector_primitive_products;
            r.phase_products+=panel.receipt.phase_products; r.phase_accumulations+=panel.receipt.phase_accumulations;
            for(Count iq=0;iq<count;++iq) {
                const double norm=q.row(iq).squaredNorm();
                const double factor=kernel=="coulomb" ? weights[source]*4*std::acos(-1.)/volume/norm
                    : weights[source]*fourier_kernel(norm,kernel,omega)/volume;
                require_finite(factor); ++r.transfer_vectors;
                if(norm==0) ++r.zero_vectors;
                if(factor==0) continue;
                for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) b(i,j)=panel.values[(i*n+j)*count+iq];
                for(Count state=0;state<nd;++state) {
                    const Eigen::Map<const Matrix> d(densities.data()+(state*nk+source)*n2,n,n);
                    temporary.noalias()=d*b; response.noalias()=b.adjoint()*temporary;
                    if(arithmetic)arithmetic->panel(b.data(),d.data(),target,state,factor);
                    if(!response.allFinite()) throw std::runtime_error("MANTA periodic exchange contraction is nonfinite");
                    r.matrix_products+=2; r.contraction_terms+=2*n2*n;
                    for(Count i=0;i<n2;++i) {
                        const Count index=(state*nk+target)*n2+i;
                        accumulate(factor*response.data()[i],result.values[index],compensation[index]);
                    }
                }
            }
            start+=count;
        }
    }
    r.energies.resize(nd);
    for(Count state=0;state<nd;++state) {
        Complex energy=0,error=0;
        for(Count ik=0;ik<nk;++ik) {
            double scale=0,residual_max=0;
            const Count start=(state*nk+ik)*n2;
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                const auto v=result.values[start+i*n+j]; finite(v);
                const double magnitude=std::abs(v),residual=std::abs(v-std::conj(result.values[start+j*n+i]));
                require_finite(magnitude); require_finite(residual); scale=std::max(scale,magnitude);
                residual_max=std::max(residual_max,residual);
                accumulate(-.5*weights[ik]*densities[start+j*n+i]*v,energy,error); ++r.contraction_terms;
            }
            r.response_hermiticity_residual=std::max(r.response_hermiticity_residual,residual_max);
            if(residual_max>1e-10*std::max(1.,scale))
                throw std::runtime_error("MANTA periodic exchange response violates Hermiticity");
        }
        finite(energy);
        if(std::abs(energy.imag())>1e-10*std::max(1.,std::abs(energy.real())))
            throw std::runtime_error("MANTA periodic exchange energy imaginary residual exceeds tolerance");
        r.energies[state]=energy.real();
    }
    if(arithmetic)arithmetic->finish(result,densities,weights);
    return result;
}
} // namespace vibeqc::manta
