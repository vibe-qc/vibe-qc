#include "vibeqc/manta_periodic_hartree.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <algorithm>
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
constexpr double tolerance=1e-10;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void text(const std::string& s) { word(s.size()); hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void word(Count v) {
        std::array<std::uint8_t,8> b{};
        for(int i=0;i<8;++i) b[i]=v>>(56-8*i);
        hash.update(b.data(),b.size());
    }
    void real(double x) { if(x==0) x=0.; Count bits; std::memcpy(&bits,&x,8); word(bits); }
    void matrix(const md::Matrix& m) {
        word(m.rows()); word(m.cols());
        for(Eigen::Index i=0;i<m.rows();++i) for(Eigen::Index j=0;j<m.cols();++j) real(m(i,j));
    }
};
void finite(Complex z) {
    if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument("MANTA periodic Hartree requires finite numerical values");
}
void accumulate(Complex x,Complex& sum,Complex& error) {
    const auto y=x-error,next=sum+y;
    error=(next-sum)-y; sum=next;
}
// Integer labels are validation only: actual caller vectors remain the source.
// No wrapping, completion, deduplication or round-to-lattice repair occurs.
void closed_support(const md::Matrix& points,const Eigen::Matrix3d& inverse,bool exclude_zero) {
    using Index=std::array<std::int64_t,3>;
    std::vector<Index> labels; labels.reserve(points.rows());
    for(Eigen::Index i=0;i<points.rows();++i) {
        const Eigen::Vector3d fractional=inverse*points.row(i).transpose();
        Index label{};
        for(int d=0;d<3;++d) {
            const double v=fractional[d],nearest=std::round(v);
            if(!std::isfinite(v) || std::abs(v)>0x1p40 || std::abs(v-nearest)>1e-10)
                throw std::invalid_argument("MANTA periodic Hartree support must contain lattice vectors");
            label[d]=static_cast<std::int64_t>(nearest);
        }
        if(exclude_zero && label==Index{}) throw std::invalid_argument("MANTA periodic Hartree requires G=0 omitted");
        labels.push_back(label);
    }
    std::sort(labels.begin(),labels.end());
    for(std::size_t i=0;i<labels.size();++i) {
        if(i && labels[i]==labels[i-1]) throw std::invalid_argument("MANTA periodic Hartree support contains duplicate vectors");
        const Index negative{-labels[i][0],-labels[i][1],-labels[i][2]};
        if(!std::binary_search(labels.begin(),labels.end(),negative))
            throw std::invalid_argument("MANTA periodic Hartree support must be inversion closed");
    }
}
void charge(PeriodicHartreeReceipt& r,const BlochFourierReceipt& b) {
    ++r.panel_calls; r.shell_calls+=b.shell_calls; r.skipped_shell_pairs+=b.skipped_shell_pairs;
    r.primitive_pairs+=b.primitive_pairs; r.vector_primitive_products+=b.vector_primitive_products;
    r.phase_products+=b.phase_products; r.phase_accumulations+=b.phase_accumulations;
}
ResourceBudget unlimited(const ResourceBudget& budget) {
    auto leaf=budget;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    leaf.other_live_bytes=0; return leaf;
}
} // namespace

PeriodicHartreePlan plan_periodic_hartree(const BasisSet& basis,Count ng,Count nc,Count nk,
    Count batch,std::optional<double> cutoff,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    if(!batch || !nk) throw std::invalid_argument("MANTA periodic Hartree requires positive batch and momentum counts");
    PeriodicHartreePlan p;
    p.n_basis=basis.nbasis(); p.vectors=ng; p.cells=nc; p.momenta=nk; p.reciprocal_batch=batch;
    const Count n2=square(p.n_basis),output_elements=mul(add(nc,nk),n2),density_elements=mul(nk,n2);
    cap(output_elements,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic Hartree output is not addressable");
    p.output_bytes=mul(16,output_elements);
    p.batches=(ng && nc)?ng/batch+(ng%batch!=0):0;
    p.panel_calls=mul(p.batches,add(1,nc));
    const Count active_g=p.batches?ng:0,active_batch=std::min(active_g,batch);
    p.contraction_terms=add(mul(active_g,mul(add(nk,nc),n2)),mul(mul(nc,nk),n2));
    // Original + binding snapshots; exported outputs + compensated sums;
    // sorted support labels, vector work, scalar reductions and diagnostics.
    const Count inputs=add(72,add(mul(24,add(ng,add(nc,nk))),add(mul(8,nk),mul(16,density_elements))));
    p.driver_live_bytes=add(mul(2,inputs),add(mul(3,p.output_bytes),add(mul(128,add(ng,nc)),
        add(mul(128,active_batch),add(mul(32,nk),4096)))));
    p.scalar_terms=add(mul(8,p.contraction_terms),add(mul(256,add(ng,nc)),mul(8,density_elements)));
    cap(p.driver_live_bytes,budget.memory_bytes,"MANTA periodic Hartree driver memory budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic Hartree scalar-work budget exceeded");
    auto leaf=unlimited(budget);
    Count peak=0;
    // At most two different batch lengths, no count-sized planning loop.
    auto include=[&](Count q,Count repeats) {
        if(!repeats) return;
        const auto density=plan_bloch_fourier(basis,q,nc,nk,cutoff,leaf);
        const auto output=plan_bloch_fourier(basis,q,1,1,cutoff,leaf);
        p.shell_calls=add(p.shell_calls,mul(repeats,add(density.shell_calls,mul(nc,output.shell_calls))));
        p.primitive_pairs=add(p.primitive_pairs,mul(repeats,add(density.primitive_pairs,mul(nc,output.primitive_pairs))));
        p.vector_primitive_products=add(p.vector_primitive_products,mul(repeats,add(density.vector_primitive_products,mul(nc,output.vector_primitive_products))));
        p.scalar_terms=add(p.scalar_terms,mul(repeats,add(density.scalar_terms,mul(nc,output.scalar_terms))));
        p.kernel_terms=add(p.kernel_terms,mul(repeats,add(density.kernel_terms,mul(nc,output.kernel_terms))));
        peak=std::max(peak,std::max(density.modeled_peak_bytes,output.modeled_peak_bytes));
    };
    if(p.batches) { include(batch,ng/batch); include(ng%batch,ng%batch!=0); }
    else peak=plan_bloch_fourier(basis,0,nc,nk,cutoff,leaf).modeled_peak_bytes;
    p.kernel_terms=add(p.kernel_terms,mul(32,p.scalar_terms));
    p.modeled_peak_bytes=add(peak,add(p.driver_live_bytes,budget.other_live_bytes));
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic Hartree shell-call budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic Hartree primitive-pair budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic Hartree scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic Hartree kernel-work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic Hartree memory budget exceeded");
    return p;
}

PeriodicHartreeResult compute_periodic_hartree(const BasisSet& basis,const md::Matrix& lattice,
    const md::Matrix& g,const md::Matrix& cells,const md::Matrix& k,const std::vector<double>& weights,
    const std::vector<Complex>& d,Count batch,std::optional<double> cutoff,const ResourceBudget& budget) {
    using namespace admission;
    if(lattice.rows()!=3 || lattice.cols()!=3 || g.cols()!=3 || cells.cols()!=3 || k.cols()!=3)
        throw std::invalid_argument("MANTA periodic Hartree requires a 3x3 lattice and (count,3) vectors");
    const auto plan=plan_periodic_hartree(basis,g.rows(),cells.rows(),k.rows(),batch,cutoff,budget);
    const Count n=plan.n_basis,n2=n*n,nk=k.rows(),nc=cells.rows();
    if(weights.size()!=nk || d.size()!=nk*n2) throw std::invalid_argument("MANTA periodic Hartree density/weight shapes disagree");
    if(!lattice.allFinite() || !g.allFinite() || !cells.allFinite() || !k.allFinite())
        throw std::invalid_argument("MANTA periodic Hartree vectors must be finite");
    const Eigen::Matrix3d a=lattice;
    const double volume=std::abs(a.determinant());
    if(!std::isfinite(volume) || volume<=0 || !a.inverse().allFinite())
        throw std::invalid_argument("MANTA periodic Hartree lattice must be finite and nonsingular");
    closed_support(g,a.transpose()/(2*std::acos(-1.)),true);
    closed_support(cells,a.inverse(),false);
    double weight_sum=0;
    for(auto w:weights) { require_finite(w); if(w<0) throw std::invalid_argument("MANTA periodic Hartree weights must be nonnegative"); weight_sum+=w; }
    if(!std::isfinite(weight_sum) || std::abs(weight_sum-1)>1e-12)
        throw std::invalid_argument("MANTA periodic Hartree weights must sum to one");
    PeriodicHartreeResult result; result.plan=plan;
    auto& r=result.receipt; r.cell_volume=volume; r.modeled_peak_bytes=plan.modeled_peak_bytes;
    double density_scale=0;
    for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto v=d[ik*n2+i*n+j]; finite(v);
        const double magnitude=std::abs(v),residual=std::abs(v-std::conj(d[ik*n2+j*n+i]));
        require_finite(magnitude); require_finite(residual);
        density_scale=std::max(density_scale,magnitude);
        r.density_hermiticity_residual=std::max(r.density_hermiticity_residual,residual);
    }
    if(r.density_hermiticity_residual>1e-12*std::max(1.,density_scale))
        throw std::invalid_argument("MANTA periodic Hartree densities must be Hermitian");
    Digest source; source.text("manta.periodic-hartree.v1;md-fp64-v1;libint-shell-axial-v1;3d-zero-average-g0-omitted");
    source.text(auxiliary_basis_content_identity_sha256(basis));
    source.matrix(lattice); source.matrix(g); source.matrix(cells); source.matrix(k);
    source.word(cutoff.has_value()); if(cutoff) source.real(*cutoff);
    for(auto w:weights) source.real(w);
    r.source_identity=source.hash.finish_hex();
    Digest density; density.text("manta.periodic-hartree.density.v1"); density.text(r.source_identity);
    density.word(nk); density.word(n); for(auto v:d) { density.real(v.real()); density.real(v.imag()); }
    r.density_identity=density.hash.finish_hex();
    result.cells.resize(nc*n2); result.momenta.resize(nk*n2);
    std::vector<Complex> cc(result.cells.size()),kc(result.momenta.size());
    ResourceBudget leaf=budget; leaf.other_live_bytes=add(budget.other_live_bytes,plan.driver_live_bytes);
    const md::Matrix minus_k=-k,zero=md::Matrix::Zero(1,3);
    if(nc) for(Count start=0;start<static_cast<Count>(g.rows());) {
        const Count size=std::min(batch,static_cast<Count>(g.rows())-start);
        const md::Matrix q=g.middleRows(start,size);
        std::vector<Complex> rho(size),rho_error(size);
        {
            const auto panel=compute_bloch_fourier_panel(basis,q,cells,minus_k,cutoff,leaf); charge(r,panel.receipt);
            for(Count ik=0;ik<nk;++ik) for(Count ij=0;ij<n2;++ij) for(Count iq=0;iq<size;++iq) {
                accumulate(weights[ik]*d[ik*n2+ij]*panel.values[(ik*n2+ij)*size+iq],rho[iq],rho_error[iq]);
                ++r.contraction_terms;
            }
        }
        for(Count iq=0;iq<size;++iq) {
            const double g2=q.row(iq).squaredNorm();
            const double factor=4*std::acos(-1.)/(volume*g2);
            if(!std::isfinite(factor) || factor<=0) throw std::invalid_argument("MANTA periodic Hartree Coulomb coefficient overflow");
            rho[iq]*=factor; finite(rho[iq]);
        }
        for(Count cell=0;cell<nc;++cell) {
            const md::Matrix shift=cells.middleRows(cell,1);
            const auto panel=compute_bloch_fourier_panel(basis,q,shift,zero,cutoff,leaf); charge(r,panel.receipt);
            for(Count ij=0;ij<n2;++ij) for(Count iq=0;iq<size;++iq) {
                const Count index=cell*n2+ij;
                accumulate(rho[iq]*std::conj(panel.values[ij*size+iq]),result.cells[index],cc[index]);
                ++r.contraction_terms;
            }
        }
        start+=size;
    }
    double cell_scale=0;
    for(auto v:result.cells) {
        finite(v); const double magnitude=std::abs(v); require_finite(magnitude);
        cell_scale=std::max(cell_scale,magnitude);
        r.cell_imaginary_residual=std::max(r.cell_imaginary_residual,std::abs(v.imag()));
    }
    if(r.cell_imaginary_residual>tolerance*std::max(1.,cell_scale))
        throw std::runtime_error("MANTA periodic Hartree real-space imaginary residual exceeds tolerance");
    for(Count ik=0;ik<nk;++ik) for(Count cell=0;cell<nc;++cell) {
        const double angle=k.row(ik).dot(cells.row(cell)); require_finite(angle);
        const auto phase=std::exp(Complex(0,angle));
        for(Count ij=0;ij<n2;++ij) { const Count index=ik*n2+ij;
            accumulate(phase*result.cells[cell*n2+ij],result.momenta[index],kc[index]); ++r.contraction_terms;
        }
    }
    double scale=0; Complex energy=0,energy_error=0;
    for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto v=result.momenta[ik*n2+i*n+j]; finite(v);
        const double magnitude=std::abs(v),residual=std::abs(v-std::conj(result.momenta[ik*n2+j*n+i]));
        require_finite(magnitude); require_finite(residual);
        scale=std::max(scale,magnitude);
        r.response_hermiticity_residual=std::max(r.response_hermiticity_residual,residual);
        accumulate(.5*weights[ik]*d[ik*n2+j*n+i]*v,energy,energy_error);
    }
    finite(energy);
    if(r.response_hermiticity_residual>tolerance*std::max(1.,scale) || std::abs(energy.imag())>tolerance*std::max(1.,std::abs(energy.real())))
        throw std::runtime_error("MANTA periodic Hartree response violates Hermiticity");
    r.energy=energy.real(); return result;
}
} // namespace vibeqc::manta
