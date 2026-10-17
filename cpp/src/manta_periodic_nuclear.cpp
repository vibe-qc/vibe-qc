#include "vibeqc/manta_periodic_nuclear.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <algorithm>
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) {
        std::array<std::uint8_t,8> b{}; for(int i=0;i<8;++i) b[i]=v>>(56-8*i);
        hash.update(b.data(),b.size());
    }
    void text(const std::string& s) { word(s.size()); hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double x) { if(x==0) x=0.; Count bits; std::memcpy(&bits,&x,8); word(bits); }
    template<class Matrix> void matrix(const Matrix& m) {
        word(m.rows()); word(m.cols());
        for(Eigen::Index i=0;i<m.rows();++i) for(Eigen::Index j=0;j<m.cols();++j) real(m(i,j));
    }
};
void finite(Complex z) {
    if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument("MANTA periodic nuclear operator requires finite values");
}
void accumulate(Complex x,Complex& sum,Complex& error) {
    const auto y=x-error,next=sum+y; error=(next-sum)-y; sum=next;
}
void closed_support(const md::Matrix& points,const Eigen::Matrix3d& inverse,bool exclude_zero) {
    using Index=std::array<std::int64_t,3>;
    std::vector<Index> labels; labels.reserve(points.rows());
    for(Eigen::Index i=0;i<points.rows();++i) {
        const Eigen::Vector3d fractional=inverse*points.row(i).transpose();
        Index label{};
        for(int d=0;d<3;++d) {
            const double v=fractional[d],nearest=std::round(v);
            if(!std::isfinite(v) || std::abs(v)>0x1p40 || std::abs(v-nearest)>1e-10)
                throw std::invalid_argument("MANTA periodic nuclear support must contain lattice vectors");
            label[d]=static_cast<std::int64_t>(nearest);
        }
        if(exclude_zero && label==Index{}) throw std::invalid_argument("MANTA periodic nuclear operator requires G=0 omitted");
        labels.push_back(label);
    }
    std::sort(labels.begin(),labels.end());
    for(std::size_t i=0;i<labels.size();++i) {
        if(i && labels[i]==labels[i-1]) throw std::invalid_argument("MANTA periodic nuclear support contains duplicate vectors");
        const Index negative{-labels[i][0],-labels[i][1],-labels[i][2]};
        if(!std::binary_search(labels.begin(),labels.end(),negative))
            throw std::invalid_argument("MANTA periodic nuclear support must be inversion closed");
    }
}
ResourceBudget unlimited(const ResourceBudget& budget) {
    auto leaf=budget;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    leaf.other_live_bytes=0; return leaf;
}
double volume(const PeriodicSystem& system) {
    const double v=std::abs(system.lattice.determinant());
    if(system.dim!=3 || !system.lattice.allFinite() || !std::isfinite(v) || v<=0 || !system.lattice.inverse().allFinite())
        throw std::invalid_argument("MANTA periodic nuclear operator requires a finite nonsingular 3D lattice");
    return v;
}
Count image_bound(const PeriodicSystem& system,double cutoff) {
    using namespace admission;
    const Eigen::Matrix3d inverse=system.lattice.inverse();
    Count bound=system.unit_cell.size();
    for(int d=0;d<3;++d) {
        const double reach=cutoff*inverse.row(d).norm();
        if(!std::isfinite(reach) || reach>1e9)
            throw std::invalid_argument("MANTA periodic nuclear image extent exceeds the supported range");
        // The shared selector refuses int-overflowed image bounds. For any
        // admitted center, its fractional coordinate is below INT_MAX and
        // its 1e-10*(1+abs(center)+reach) slack is less than 0.32 here.
        // Five extra indices conservatively cover floor/ceil and both slacks.
        bound=mul(bound,add(mul(2,static_cast<Count>(std::ceil(reach))),5));
    }
    return bound;
}
} // namespace

PeriodicNuclearPlan plan_periodic_nuclear(const BasisSet& basis,const PeriodicSystem& system,
    Count ng,Count nc,Count nk,double alpha,double nuclear_cutoff,Count batch,
    std::optional<double> cutoff,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget); volume(system);
    if(!std::isfinite(alpha) || alpha<=0 || !std::isfinite(nuclear_cutoff) || nuclear_cutoff<0 || !batch)
        throw std::invalid_argument("MANTA periodic nuclear operator requires positive finite alpha/batch and nonnegative finite nuclear cutoff");
    PeriodicNuclearPlan p;
    p.n_basis=basis.nbasis(); p.vectors=ng; p.cells=nc; p.momenta=nk; p.reciprocal_batch=batch;
    const Count n2=square(p.n_basis),ce=mul(nc,n2),ke=mul(nk,n2);
    for(auto count:{ce,ke}) cap(count,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic nuclear output is not addressable");
    for(auto count:{ng,nc,nk}) cap(count,std::numeric_limits<Eigen::Index>::max()/3,"MANTA periodic nuclear vectors are not addressable");
    p.output_bytes=add(mul(48,ce),mul(16,ke));
    const Count inputs=add(72,add(mul(32,system.unit_cell.size()),mul(24,add(ng,add(nc,nk)))));
    p.batches=nc?ng/batch+(ng%batch!=0):0;
    p.driver_live_bytes=add(mul(2,inputs),add(mul(3,p.output_bytes),add(mul(128,add(ng,nc)),
        add(mul(128,std::min(ng,batch)),4096))));
    cap(p.driver_live_bytes,budget.memory_bytes,"MANTA periodic nuclear driver memory budget exceeded");
    p.maximum_nuclear_candidates_per_pair=image_bound(system,nuclear_cutoff);
    const Count candidate_pairs=mul(nc,square(basis.nshells()));
    p.nuclear_image_candidates_upper_bound=mul(candidate_pairs,p.maximum_nuclear_candidates_per_pair);
    p.contraction_terms=add(mul(ng,add(system.unit_cell.size(),ce)),mul(nk,ce));
    p.scalar_terms=add(mul(32,p.nuclear_image_candidates_upper_bound),add(mul(8,p.contraction_terms),
        add(mul(256,add(ng,nc)),mul(16,add(ce,ke)))));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic nuclear scalar-work budget exceeded");
    auto leaf=unlimited(budget);
    OneElectronRequest request; request.operation="potential"; request.kernel="erfc"; request.omega=alpha;
    const auto empty=plan_one_electron(basis,basis,request,leaf);
    request.charges.push_back({1.,{0.,0.,0.}});
    const auto one=plan_one_electron(basis,basis,request,leaf);
    p.nuclear_shell_calls=mul(nc,one.shell_calls);
    const Count nuclear_pairs=mul(nc,one.primitive_pairs);
    p.primitive_pairs=nuclear_pairs;
    p.primitive_charge_products_upper_bound=mul(nuclear_pairs,p.maximum_nuclear_candidates_per_pair);
    p.kernel_terms=mul(nc,add(empty.kernel_terms,mul(p.maximum_nuclear_candidates_per_pair,one.kernel_terms-empty.kernel_terms)));
    // Two geometrically growing vectors (charges and atom indices) in the
    // shared selector occupy at most 80 bytes per candidate, plus scratch.
    Count peak=add(empty.modeled_peak_bytes,mul(80,p.maximum_nuclear_candidates_per_pair));
    const auto overlap=plan_bloch_one_electron(basis,"overlap",nc,0,cutoff,leaf);
    p.overlap_shell_calls=overlap.shell_calls;
    p.primitive_pairs=add(p.primitive_pairs,overlap.primitive_pairs);
    p.scalar_terms=add(p.scalar_terms,overlap.scalar_terms);
    p.kernel_terms=add(p.kernel_terms,overlap.kernel_terms);
    peak=std::max(peak,overlap.modeled_peak_bytes);
    auto include=[&](Count count,Count repeats) {
        if(!repeats) return;
        const auto ft=plan_bloch_fourier(basis,count,1,1,cutoff,leaf);
        const Count calls=mul(nc,repeats);
        p.fourier_shell_calls=add(p.fourier_shell_calls,mul(calls,ft.shell_calls));
        p.primitive_pairs=add(p.primitive_pairs,mul(calls,ft.primitive_pairs));
        p.vector_primitive_products=add(p.vector_primitive_products,mul(calls,ft.vector_primitive_products));
        p.scalar_terms=add(p.scalar_terms,mul(calls,ft.scalar_terms));
        p.kernel_terms=add(p.kernel_terms,mul(calls,ft.kernel_terms));
        peak=std::max(peak,ft.modeled_peak_bytes);
    };
    if(nc) { include(batch,ng/batch); include(ng%batch,ng%batch!=0); }
    p.shell_calls=add(p.nuclear_shell_calls,add(p.overlap_shell_calls,p.fourier_shell_calls));
    p.kernel_terms=add(p.kernel_terms,mul(32,p.scalar_terms));
    p.modeled_peak_bytes=add(peak,add(p.driver_live_bytes,budget.other_live_bytes));
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic nuclear shell-call budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic nuclear primitive-pair budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic nuclear scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic nuclear kernel-work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic nuclear memory budget exceeded");
    for(const auto& atom:system.unit_cell) for(auto x:atom.xyz) require_finite(x);
    return p;
}

PeriodicNuclearResult compute_periodic_nuclear(const BasisSet& basis,const PeriodicSystem& system,
    const md::Matrix& g,const md::Matrix& cells,const md::Matrix& k,double alpha,double nuclear_cutoff,
    Count batch,std::optional<double> cutoff,const ResourceBudget& budget) {
    using namespace admission;
    if(g.cols()!=3 || cells.cols()!=3 || k.cols()!=3)
        throw std::invalid_argument("MANTA periodic nuclear vectors must have three Cartesian components");
    const auto plan=plan_periodic_nuclear(basis,system,g.rows(),cells.rows(),k.rows(),alpha,nuclear_cutoff,batch,cutoff,budget);
    if(!g.allFinite() || !cells.allFinite() || !k.allFinite())
        throw std::invalid_argument("MANTA periodic nuclear vectors must be finite");
    closed_support(g,system.lattice.transpose()/(2*std::acos(-1.)),true);
    closed_support(cells,system.lattice.inverse(),false);
    PeriodicNuclearResult result; result.plan=plan;
    auto& r=result.receipt;
    r.alpha=alpha; r.nuclear_cutoff_bohr=nuclear_cutoff; r.cell_volume=volume(system);
    r.modeled_peak_bytes=plan.modeled_peak_bytes;
    Digest source; source.text("manta.periodic-nuclear.v1;md-fp64-v1;libint-shell-axial-v1;3d-zero-average-g0-omitted;pair-midpoint-sphere-v1");
    source.text(auxiliary_basis_content_identity_sha256(basis)); source.matrix(system.lattice);
    source.word(system.unit_cell.size());
    for(const auto& atom:system.unit_cell) {
        r.nuclear_charge+=atom.Z; source.real(atom.Z); for(auto x:atom.xyz) source.real(x);
    }
    source.matrix(g); source.matrix(cells); source.matrix(k); source.real(alpha); source.real(nuclear_cutoff);
    source.word(cutoff.has_value()); if(cutoff) source.real(*cutoff);
    r.source_identity=source.hash.finish_hex();
    r.constant_coefficient=(std::acos(-1.)/r.cell_volume/alpha)/alpha*r.nuclear_charge;
    require_finite(r.constant_coefficient);
    const Count n=plan.n_basis,n2=n*n,nc=cells.rows(),nk=k.rows();
    result.short_range_cells.resize(nc*n2); result.reciprocal_cells.resize(nc*n2);
    result.cells.resize(nc*n2); result.momenta.resize(nk*n2);
    std::vector<Complex> cell_error(nc*n2),k_error(nk*n2);
    ResourceBudget leaf=budget; leaf.other_live_bytes=add(budget.other_live_bytes,plan.driver_live_bytes);
    {
        auto overlap=compute_bloch_one_electron(basis,"overlap",cells,md::Matrix(0,3),cutoff,leaf);
        r.overlap_shell_calls=overlap.receipt.shell_calls;
        r.skipped_shell_pairs+=overlap.receipt.skipped_shell_pairs;
        r.primitive_pairs+=overlap.receipt.primitive_pairs;
        result.constant_cells=std::move(overlap.cell_values);
        for(auto& v:result.constant_cells) { v*=r.constant_coefficient; require_finite(v); }
    }
    const PairNuclearImageSelector selector(system,nuclear_cutoff);
    for(Count cell=0;cell<nc;++cell) {
        Count row=0;
        for(const auto& a:basis.libint()) {
            Count col=0;
            for(const auto& original:basis.libint()) {
                auto b=original;
                for(int d=0;d<3;++d) { b.O[d]+=cells(cell,d); require_finite(b.O[d]); }
                if(cutoff && std::hypot(a.O[0]-b.O[0],a.O[1]-b.O[1],a.O[2]-b.O[2])>*cutoff) {
                    ++r.skipped_shell_pairs; col+=b.size(); continue;
                }
                const auto images=selector.select(a,b);
                if(images.charges.size()>plan.maximum_nuclear_candidates_per_pair)
                    throw std::runtime_error("MANTA periodic nuclear image bound violated");
                const auto tile=md::shell_one_electron(a,b,"potential",images.charges,{}, {},md::RadialKernel::Erfc,alpha);
                if(!tile.allFinite()) throw std::runtime_error("MANTA periodic nuclear tile is nonfinite");
                for(Count i=0;i<a.size();++i) for(Count j=0;j<b.size();++j)
                    result.short_range_cells[cell*n2+(row+i)*n+col+j]=tile(i,j);
                ++r.nuclear_shell_calls; r.nuclear_images_retained+=images.charges.size();
                const Count pairs=a.alpha.size()*b.alpha.size();
                r.primitive_pairs+=pairs; r.primitive_charge_products+=pairs*images.charges.size();
                col+=b.size();
            }
            row+=a.size();
        }
    }
    // Sun et al., JCP 147,164119 (2017), Eq.16 and reciprocal Coulomb
    // convention. The Ewald G=0 short-range limit is -pi*sum(Z)/(alpha^2 V);
    // its subtraction adds the overlap term above to the nuclear operator.
    const md::Matrix zero=md::Matrix::Zero(1,3);
    if(nc) for(Count start=0;start<static_cast<Count>(g.rows());) {
        const Count count=std::min(batch,static_cast<Count>(g.rows())-start);
        const md::Matrix q=g.middleRows(start,count);
        std::vector<Complex> weight(count);
        for(Count iq=0;iq<count;++iq) {
            Complex rho=0,error=0;
            for(const auto& atom:system.unit_cell) {
                const double phase=q(iq,0)*atom.xyz[0]+q(iq,1)*atom.xyz[1]+q(iq,2)*atom.xyz[2];
                require_finite(phase); accumulate(static_cast<double>(atom.Z)*std::exp(Complex(0,-phase)),rho,error);
                ++r.contraction_terms;
            }
            const double g2=q.row(iq).squaredNorm(),argument=q.row(iq).norm()/(2*alpha);
            if(!std::isfinite(g2) || g2<=0) throw std::invalid_argument("MANTA periodic nuclear reciprocal norm overflow");
            const double factor=-4*std::acos(-1.)/r.cell_volume/g2*std::exp(-argument*argument);
            require_finite(factor); weight[iq]=factor*rho; finite(weight[iq]);
        }
        for(Count cell=0;cell<nc;++cell) {
            const md::Matrix shift=cells.middleRows(cell,1);
            const auto panel=compute_bloch_fourier_panel(basis,q,shift,zero,cutoff,leaf);
            r.fourier_shell_calls+=panel.receipt.shell_calls; r.skipped_shell_pairs+=panel.receipt.skipped_shell_pairs;
            r.primitive_pairs+=panel.receipt.primitive_pairs; r.vector_primitive_products+=panel.receipt.vector_primitive_products;
            for(Count ij=0;ij<n2;++ij) for(Count iq=0;iq<count;++iq) {
                const Count index=cell*n2+ij;
                accumulate(weight[iq]*std::conj(panel.values[ij*count+iq]),result.reciprocal_cells[index],cell_error[index]);
                ++r.contraction_terms;
            }
        }
        start+=count;
    }
    double scale=0.;
    for(Count i=0;i<nc*n2;++i) {
        result.cells[i]=result.short_range_cells[i]+result.reciprocal_cells[i]+result.constant_cells[i];
        const auto v=result.cells[i]; finite(v); require_finite(std::abs(v)); scale=std::max(scale,std::abs(v));
        r.cell_imaginary_residual=std::max(r.cell_imaginary_residual,std::abs(v.imag()));
    }
    if(r.cell_imaginary_residual>1e-10*std::max(1.,scale))
        throw std::runtime_error("MANTA periodic nuclear cell imaginary residual exceeds tolerance");
    for(Count ik=0;ik<nk;++ik) for(Count cell=0;cell<nc;++cell) {
        const double angle=k.row(ik).dot(cells.row(cell)); require_finite(angle);
        const auto phase=std::exp(Complex(0,angle));
        for(Count ij=0;ij<n2;++ij) {
            const Count index=ik*n2+ij;
            accumulate(phase*result.cells[cell*n2+ij],result.momenta[index],k_error[index]); ++r.contraction_terms;
        }
    }
    scale=0.;
    for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto v=result.momenta[ik*n2+i*n+j]; finite(v);
        const double magnitude=std::abs(v),residual=std::abs(v-std::conj(result.momenta[ik*n2+j*n+i]));
        require_finite(magnitude); require_finite(residual); scale=std::max(scale,magnitude);
        r.response_hermiticity_residual=std::max(r.response_hermiticity_residual,residual);
    }
    if(r.response_hermiticity_residual>1e-10*std::max(1.,scale))
        throw std::runtime_error("MANTA periodic nuclear response violates Hermiticity");
    return result;
}
} // namespace vibeqc::manta
