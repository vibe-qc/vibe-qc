#include "vibeqc/manta_fourier.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
std::string source_identity(const BasisSet& bra,const BasisSet* ket,const md::Matrix& vectors) {
    vibeqc::detail::Sha256 hash;
    const std::string prefix="manta.fourier-panel.v1;md-fp64-v1;exp(-i*q.r);libint-shell-axial-v1;"
        +auxiliary_basis_content_identity_sha256(bra)+";"
        +(ket?auxiliary_basis_content_identity_sha256(*ket):"unit")+";";
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    auto word=[&hash](std::uint64_t bits) {
        std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(bits>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    };
    word(vectors.rows());
    for(Eigen::Index i=0;i<vectors.rows();++i) for(int j=0;j<3;++j) {
        const double value=vectors(i,j)==0?0.:vectors(i,j);
        std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits)); word(bits);
    }
    return hash.finish_hex();
}
} // namespace

FourierPlan plan_fourier(const BasisSet& bra,const BasisSet* ket,Count vectors,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    FourierPlan p;
    const Count na=bra.nbasis(),nb=ket?ket->nbasis():1;
    p.shape={vectors,na}; if(ket) p.shape.push_back(nb);
    cap(vectors,std::numeric_limits<Eigen::Index>::max(),"MANTA Fourier vector count is not addressable");
    const Count elements=mul(vectors,mul(na,nb));
    cap(elements,budget.scalar_terms,"MANTA Fourier element budget exceeded");
    cap(elements,std::numeric_limits<Eigen::Index>::max(),"MANTA Fourier panel is not addressable");
    p.shell_calls=vectors?mul(bra.nshells(),ket?ket->nshells():1):0;
    cap(p.shell_calls,budget.shell_calls,"MANTA Fourier shell-call budget exceeded");
    p.output_bytes=mul(16,elements);
    p.modeled_peak_bytes=add(mul(3,p.output_bytes),add(mul(48,vectors),add(budget.other_live_bytes,budget.backend_margin_bytes)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA Fourier matrix/input memory budget exceeded");
    const auto a=census(bra,ket?6:7);
    Census b; b.primitives=b.maximum_size=b.maximum_cartesian=1;
    if(ket) b=census(*ket,6);
    p.primitive_pairs=vectors?mul(a.primitives,b.primitives):0;
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA Fourier primitive-pair budget exceeded");
    p.vector_primitive_products=mul(p.primitive_pairs,vectors);
    const Count cart=mul(a.maximum_cartesian,b.maximum_cartesian);
    const Count l=a.maximum_l+b.maximum_l,h=(l+1)*(l+2)*(l+3)/6,e=mul(cart,h);
    const Count axes=3*(a.maximum_l+1)*(b.maximum_l+1)*(l+1);
    const Count primitive_work=add(mul(24,e),add(mul(12,axes),
        mul(vectors,add(mul(16,e),mul(48,mul(l+1,h))))));
    p.kernel_terms=add(mul(p.primitive_pairs,primitive_work),mul(p.shell_calls,mul(16,mul(vectors,square(cart)))));
    cap(p.kernel_terms,budget.kernel_terms,"MANTA Fourier kernel-work budget exceeded");
    p.tile_elements=mul(vectors,mul(a.maximum_size,b.maximum_size));
    const Count complex_elements=add(mul(2,e),add(mul(2,mul(h,vectors)),
        add(mul(10,mul(cart,vectors)),add(mul(2,square(cart)),4096))));
    const Count workspace=vectors?add(mul(8,add(mul(4,e),axes)),mul(16,complex_elements)):0;
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(workspace,add(a.bytes,b.bytes)));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA Fourier workspace memory budget exceeded");
    validate_basis(bra); if(ket) validate_basis(*ket);
    return p;
}
FourierResult compute_fourier_panel(const BasisSet& bra,const BasisSet* ket,
                                    const md::Matrix& vectors,const ResourceBudget& budget) {
    if(vectors.cols()!=3) throw std::invalid_argument("MANTA Fourier vectors must have three Cartesian components");
    const auto plan=plan_fourier(bra,ket,vectors.rows(),budget);
    if(!vectors.allFinite()) throw std::invalid_argument("MANTA Fourier vectors must be finite");
    FourierResult result;
    result.shape=plan.shape;
    result.receipt.source_identity=source_identity(bra,ket,vectors);
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    result.values.resize(plan.output_bytes/sizeof(std::complex<double>));
    if(vectors.rows()==0) return result;
    const Count na=bra.nbasis(),nb=ket?ket->nbasis():1;
    Count row=0;
    for(const auto& a:bra.libint()) {
        Count col=0;
        for(std::size_t s=0;s<(ket?ket->nshells():1);++s) {
            const auto* b=ket?&ket->libint()[s]:nullptr;
            const Count width=b?b->size():1;
            const auto values=md::shell_fourier_panel(&a,b,vectors);
            for(Eigen::Index q=0;q<vectors.rows();++q) for(Count i=0;i<a.size();++i) for(Count j=0;j<width;++j)
                result.values[(q*na+row+i)*nb+col+j]=values(i*width+j,q);
            ++result.receipt.shell_calls;
            const Count primitives=a.alpha.size()*(b?b->alpha.size():1);
            result.receipt.primitive_pairs+=primitives;
            result.receipt.vector_primitive_products+=primitives*vectors.rows();
            result.receipt.integral_elements+=a.size()*width*vectors.rows();
            col+=width;
        }
        row+=a.size();
    }
    return result;
}
} // namespace vibeqc::manta
