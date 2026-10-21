#include "vibeqc/manta_bloch_one_electron.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
std::string identity(const BasisSet& basis,const std::string& operation,const md::Matrix& cells,
                     const md::Matrix& k,std::optional<double> cutoff) {
    vibeqc::detail::Sha256 hash;
    const auto prefix=std::string("manta.bloch-one-electron.v1;md-fp64-v1;libint-shell-axial-v1;exp(+i*k.R);")
        +auxiliary_basis_content_identity_sha256(basis)+";"+operation+";";
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    auto word=[&](std::uint64_t v) {
        std::array<std::uint8_t,8> b{};
        for(int j=0;j<8;++j) b[j]=static_cast<std::uint8_t>(v>>(56-8*j));
        hash.update(b.data(),b.size());
    };
    auto number=[&](double x) { if(x==0) x=0.; std::uint64_t v; std::memcpy(&v,&x,8); word(v); };
    word(cutoff.has_value()); if(cutoff) number(*cutoff);
    for(const auto* m:{&cells,&k}) {
        word(m->rows());
        for(Eigen::Index i=0;i<m->rows();++i) for(int j=0;j<3;++j) number((*m)(i,j));
    }
    return hash.finish_hex();
}
} // namespace
BlochOneElectronPlan plan_bloch_one_electron(const BasisSet& basis,const std::string& operation,
    Count cells,Count momenta,std::optional<double> cutoff,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    if(operation!="overlap" && operation!="kinetic")
        throw std::invalid_argument("MANTA Bloch one-electron operation must be overlap or kinetic");
    if(cutoff && (!std::isfinite(*cutoff) || *cutoff<0))
        throw std::invalid_argument("MANTA Bloch pair cutoff must be finite and nonnegative");
    const Count n=basis.nbasis(),n2=square(n),elements=mul(momenta,n2),cell_elements=mul(cells,n2);
    BlochOneElectronPlan p;
    p.shape={momenta,n,n}; p.cell_shape={cells,n,n};
    for(auto count:{elements,cell_elements})
        cap(count,std::numeric_limits<Eigen::Index>::max(),"MANTA Bloch one-electron matrix is not addressable");
    for(auto count:{cells,momenta})
        cap(count,std::numeric_limits<Eigen::Index>::max()/3,"MANTA Bloch vector count is not addressable");
    p.candidate_shell_pairs=mul(cells,square(basis.nshells()));
    p.shell_calls=p.candidate_shell_pairs;
    cap(p.shell_calls,budget.shell_calls,"MANTA Bloch one-electron shell-call budget exceeded");
    p.integral_elements=cell_elements;
    p.phase_products=mul(cells,momenta);
    p.phase_accumulations=mul(cell_elements,momenta);
    p.scalar_terms=add(mul(12,add(cells,momenta)),add(mul(2,add(elements,cell_elements)),
        add(p.candidate_shell_pairs,add(p.phase_products,p.phase_accumulations))));
    cap(p.scalar_terms,budget.scalar_terms,"MANTA Bloch one-electron scalar-work budget exceeded");
    p.output_bytes=add(mul(16,elements),mul(8,cell_elements));
    const Count input_bytes=mul(48,add(cells,momenta));
    p.modeled_peak_bytes=add(mul(3,p.output_bytes),add(input_bytes,add(budget.other_live_bytes,budget.backend_margin_bytes)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA Bloch one-electron output/input memory budget exceeded");
    // Retain the complete one-cell matrix allowance of the molecular plan
    // although the evaluator below materializes only a shell tile. This also
    // covers copied shell payloads and Cartesian/pure transformation scratch.
    ResourceBudget leaf=budget;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    leaf.other_live_bytes=0;
    OneElectronRequest request; request.operation=operation;
    const auto f=plan_one_electron(basis,basis,request,leaf);
    p.primitive_pairs=mul(cells,f.primitive_pairs);
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA Bloch one-electron primitive-pair budget exceeded");
    p.kernel_terms=add(mul(cells,f.kernel_terms),add(mul(48,p.phase_products),
        add(mul(24,p.phase_accumulations),add(mul(32,p.candidate_shell_pairs),mul(8,p.scalar_terms)))));
    cap(p.kernel_terms,budget.kernel_terms,"MANTA Bloch one-electron kernel-work budget exceeded");
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(f.modeled_peak_bytes,mul(16,momenta)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA Bloch one-electron workspace memory budget exceeded");
    return p;
}
BlochOneElectronResult compute_bloch_one_electron(const BasisSet& basis,const std::string& operation,
    const md::Matrix& cells,const md::Matrix& momenta,std::optional<double> cutoff,const ResourceBudget& budget) {
    if(cells.cols()!=3 || momenta.cols()!=3)
        throw std::invalid_argument("MANTA Bloch translations and momenta require three Cartesian components");
    const auto plan=plan_bloch_one_electron(basis,operation,cells.rows(),momenta.rows(),cutoff,budget);
    if(!cells.allFinite() || !momenta.allFinite())
        throw std::invalid_argument("MANTA Bloch translations and momenta must be finite");
    BlochOneElectronResult result; result.plan=plan;
    auto& receipt=result.receipt;
    receipt.operation=operation; receipt.source_identity=identity(basis,operation,cells,momenta,cutoff);
    receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    const Count n=basis.nbasis(),n2=n*n;
    result.values.resize(momenta.rows()*n2);
    result.cell_values.resize(cells.rows()*n2);
    std::vector<std::complex<double>> compensation(result.values.size()),phases(momenta.rows());
    for(Eigen::Index cell=0;cell<cells.rows();++cell) {
        for(Eigen::Index k=0;k<momenta.rows();++k) {
            const double angle=momenta.row(k).dot(cells.row(cell));
            if(!std::isfinite(angle)) throw std::invalid_argument("MANTA Bloch phase argument overflow");
            phases[k]=std::exp(std::complex<double>(0,angle)); ++receipt.phase_products;
        }
        Count row=0;
        for(const auto& a:basis.libint()) {
            Count col=0;
            for(const auto& original:basis.libint()) {
                ++receipt.candidate_shell_pairs;
                auto b=original;
                for(int d=0;d<3;++d) {
                    b.O[d]+=cells(cell,d);
                    if(!std::isfinite(b.O[d])) throw std::invalid_argument("MANTA translated shell origin overflow");
                }
                if(cutoff && std::hypot(a.O[0]-b.O[0],a.O[1]-b.O[1],a.O[2]-b.O[2])>*cutoff) {
                    ++receipt.skipped_shell_pairs; col+=b.size(); continue;
                }
                // McMurchie-Davidson, J. Comput. Phys. 26,218 (1978).
                // A single physical tile supplies the real-space block and
                // all Bloch rows, independent of momentum count.
                const auto tile=md::shell_one_electron(a,b,operation);
                if(!tile.allFinite()) throw std::runtime_error("MANTA Bloch one-electron tile is nonfinite");
                for(Count i=0;i<a.size();++i) for(Count j=0;j<b.size();++j) {
                    const Count index=(row+i)*n+col+j;
                    const double value=tile(i,j); result.cell_values[cell*n2+index]=value;
                    for(Eigen::Index k=0;k<momenta.rows();++k) {
                        const Count target=k*n2+index;
                        const auto contribution=phases[k]*value;
                        const auto y=contribution-compensation[target],t=result.values[target]+y;
                        compensation[target]=(t-result.values[target])-y; result.values[target]=t;
                    }
                }
                ++receipt.shell_calls; receipt.primitive_pairs+=a.alpha.size()*b.alpha.size();
                const Count elements=a.size()*b.size();
                receipt.integral_elements+=elements; receipt.phase_accumulations+=elements*momenta.rows();
                col+=b.size();
            }
            row+=a.size();
        }
    }
    // Report the raw residual even for a deliberately asymmetric finite
    // support. Never silently close the cell set or Hermitize its result.
    for(Eigen::Index k=0;k<momenta.rows();++k) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto v=result.values[k*n2+i*n+j];
        const double residual=std::abs(v-std::conj(result.values[k*n2+j*n+i]));
        if(!std::isfinite(v.real()) || !std::isfinite(v.imag()) || !std::isfinite(residual))
            throw std::runtime_error("MANTA Bloch one-electron accumulation is nonfinite");
        receipt.hermiticity_residual=std::max(receipt.hermiticity_residual,residual);
    }
    return result;
}
} // namespace vibeqc::manta
