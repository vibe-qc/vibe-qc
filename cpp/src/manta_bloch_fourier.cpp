#include "vibeqc/manta_bloch_fourier.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
std::string identity(const BasisSet& basis,const md::Matrix& q,const md::Matrix& cells,
                     const md::Matrix& k,std::optional<double> cutoff) {
    vibeqc::detail::Sha256 hash;
    const auto prefix=std::string("manta.bloch-fourier.v1;md-fp64-v1;libint-shell-axial-v1;exp(+i*k.R)*exp(-i*q.r);")
        +auxiliary_basis_content_identity_sha256(basis)+";";
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    auto word=[&](std::uint64_t v) {
        std::array<std::uint8_t,8> b{};
        for(int j=0;j<8;++j) b[j]=static_cast<std::uint8_t>(v>>(56-8*j));
        hash.update(b.data(),b.size());
    };
    auto number=[&](double x) { if(x==0) x=0.; std::uint64_t v; std::memcpy(&v,&x,8); word(v); };
    word(cutoff.has_value()); if(cutoff) number(*cutoff);
    for(const auto* m:{&q,&cells,&k}) {
        word(m->rows());
        for(Eigen::Index i=0;i<m->rows();++i) for(int j=0;j<3;++j) number((*m)(i,j));
    }
    return hash.finish_hex();
}
} // namespace

BlochFourierPlan plan_bloch_fourier(const BasisSet& basis,Count vectors,Count cells,Count momenta,
                                  std::optional<double> cutoff,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    if(cutoff && (!std::isfinite(*cutoff) || *cutoff<0)) throw std::invalid_argument("MANTA Bloch pair cutoff must be finite and nonnegative");
    const Count n=basis.nbasis(), n2=square(n), elements=mul(momenta,mul(vectors,n2));
    BlochFourierPlan p;
    p.shape={momenta,n,n,vectors}; p.cells=cells;
    cap(elements,std::numeric_limits<Eigen::Index>::max(),"MANTA Bloch Fourier panel is not addressable");
    for(auto count:{vectors,cells,momenta})
        cap(count,std::numeric_limits<Eigen::Index>::max()/3,"MANTA Bloch vector count is not addressable");
    const bool active=vectors && cells && momenta;
    p.candidate_shell_pairs=active?mul(cells,square(basis.nshells())):0;
    p.shell_calls=p.candidate_shell_pairs;
    cap(p.shell_calls,budget.shell_calls,"MANTA Bloch shell-call budget exceeded");
    p.phase_products=active?mul(cells,momenta):0;
    p.phase_accumulations=active?mul(cells,elements):0;
    p.scalar_terms=add(elements,add(p.candidate_shell_pairs,add(p.phase_products,p.phase_accumulations)));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA Bloch scalar-work budget exceeded");
    p.output_bytes=mul(16,elements);
    const Count input_bytes=mul(48,add(vectors,add(cells,momenta)));
    p.modeled_peak_bytes=add(mul(3,p.output_bytes),add(input_bytes,add(budget.other_live_bytes,budget.backend_margin_bytes)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA Bloch output/input memory budget exceeded");
    // Reuse the conservative MD workspace inventory. Its full-AO output
    // allowance is deliberately retained although this driver uses tiles.
    ResourceBudget leaf=budget;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    leaf.other_live_bytes=0;
    const auto f=plan_fourier(basis,&basis,active?vectors:0,leaf);
    p.primitive_pairs=active?mul(cells,f.primitive_pairs):0;
    p.vector_primitive_products=mul(p.primitive_pairs,vectors);
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA Bloch primitive-pair budget exceeded");
    p.kernel_terms=add(active?mul(cells,f.kernel_terms):0,
        add(mul(48,p.phase_products),add(mul(24,p.phase_accumulations),mul(24,p.candidate_shell_pairs))));
    cap(p.kernel_terms,budget.kernel_terms,"MANTA Bloch kernel-work budget exceeded");
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(f.modeled_peak_bytes,mul(16,momenta)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA Bloch Fourier workspace memory budget exceeded");
    return p;
}

BlochFourierResult compute_bloch_fourier_panel(const BasisSet& basis,const md::Matrix& vectors,
    const md::Matrix& cells,const md::Matrix& momenta,std::optional<double> cutoff,const ResourceBudget& budget) {
    if(vectors.cols()!=3 || cells.cols()!=3 || momenta.cols()!=3)
        throw std::invalid_argument("MANTA Bloch vectors, translations and momenta require three Cartesian components");
    const auto plan=plan_bloch_fourier(basis,vectors.rows(),cells.rows(),momenta.rows(),cutoff,budget);
    if(!vectors.allFinite() || !cells.allFinite() || !momenta.allFinite())
        throw std::invalid_argument("MANTA Bloch vectors, translations and momenta must be finite");
    BlochFourierResult result;
    result.shape=plan.shape;
    result.receipt.source_identity=identity(basis,vectors,cells,momenta,cutoff);
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    result.values.resize(plan.output_bytes/sizeof(std::complex<double>));
    if(!vectors.rows() || !cells.rows() || !momenta.rows()) return result;
    std::vector<std::complex<double>> compensation(result.values.size()),phases(momenta.rows());
    const Count n=basis.nbasis(),nq=vectors.rows();
    for(Eigen::Index cell=0;cell<cells.rows();++cell) {
        for(Eigen::Index k=0;k<momenta.rows();++k) {
            const double angle=momenta.row(k).dot(cells.row(cell));
            if(!std::isfinite(angle)) throw std::invalid_argument("MANTA Bloch phase argument overflow");
            phases[k]=std::exp(std::complex<double>(0,angle));
            ++result.receipt.phase_products;
        }
        Count row=0;
        for(const auto& a:basis.libint()) {
            Count col=0;
            for(const auto& original:basis.libint()) {
                ++result.receipt.candidate_shell_pairs;
                auto b=original;
                for(int d=0;d<3;++d) {
                    b.O[d]+=cells(cell,d);
                    if(!std::isfinite(b.O[d])) throw std::invalid_argument("MANTA translated shell origin overflow");
                }
                if(cutoff && std::hypot(a.O[0]-b.O[0],a.O[1]-b.O[1],a.O[2]-b.O[2])>*cutoff) {
                    ++result.receipt.skipped_shell_pairs; col+=b.size(); continue;
                }
                const auto tile=md::shell_fourier_panel(&a,&b,vectors);
                if(!tile.allFinite()) throw std::runtime_error("MANTA Bloch Fourier tile is nonfinite");
                for(Eigen::Index k=0;k<momenta.rows();++k) for(Count i=0;i<a.size();++i)
                    for(Count j=0;j<b.size();++j) for(Count q=0;q<nq;++q) {
                        const Count index=((k*n+row+i)*n+col+j)*nq+q;
                        const auto contribution=phases[k]*tile(i*b.size()+j,q);
                        const auto y=contribution-compensation[index],t=result.values[index]+y;
                        compensation[index]=(t-result.values[index])-y; result.values[index]=t;
                    }
                ++result.receipt.shell_calls;
                const Count primitives=a.alpha.size()*b.alpha.size(),elements=a.size()*b.size()*nq;
                result.receipt.primitive_pairs+=primitives;
                result.receipt.vector_primitive_products+=primitives*nq;
                result.receipt.integral_elements+=elements;
                result.receipt.phase_accumulations+=elements*momenta.rows();
                col+=b.size();
            }
            row+=a.size();
        }
    }
    for(const auto v:result.values) if(!std::isfinite(v.real()) || !std::isfinite(v.imag()))
        throw std::runtime_error("MANTA Bloch Fourier accumulation is nonfinite");
    return result;
}
} // namespace vibeqc::manta
