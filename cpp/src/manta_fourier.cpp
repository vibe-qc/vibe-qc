#include "vibeqc/manta_fourier.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <cstring>
#include "manta_ri_interval.hpp"

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

BlochPhasePlan plan_bloch_phase(const ResourceBudget& b) {
    using namespace admission;margin(b);BlochPhasePlan p;
    // At most 1028 binary squarings for any finite binary64 dot enclosure,
    // 12 Taylor steps per component, endpoints, remainders and owned exports.
    p.modeled_peak_bytes=add(add(b.other_live_bytes,b.backend_margin_bytes),4096);
    p.scalar_terms=262144;p.kernel_terms=2097152;
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA phase enclosure memory budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA phase enclosure scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA phase enclosure work budget exceeded");return p;
}
BlochPhaseCertificate certify_bloch_phase(const Eigen::Vector3d& q,const Eigen::Vector3d& r,
    std::complex<double> evaluated,const ResourceBudget& b) {
    plan_bloch_phase(b);namespace si=spectral_interval;si::require_arithmetic();
    if(!q.allFinite() || !r.allFinite() || !std::isfinite(evaluated.real()) || !std::isfinite(evaluated.imag()))
        throw std::invalid_argument("MANTA phase enclosure requires finite vectors and witness");
    si::Interval angle;
    for(int i=0;i<3;++i)angle=si::add(angle,si::product(q[i],r[i]));
    int exponent=0;std::frexp(std::max(std::abs(angle.lo),std::abs(angle.hi)),&exponent);
    const int scaling=(angle.lo==0. && angle.hi==0.)?0:std::max(0,exponent+4);
    const auto x=si::product(angle,si::Interval{std::ldexp(1.,-scaling),std::ldexp(1.,-scaling)});
    const auto squared=si::product(x,x);const si::Interval negative{-squared.hi,-squared.lo};
    auto divide=[](si::Interval a,double d) {
        if(a.lo==0. && a.hi==0.)return a;
        volatile double lo=a.lo/d,hi=a.hi/d;return si::Interval{si::down(lo),si::up(hi)};
    };
    si::Interval cosine{1.,1.},sine=x,ct{1.,1.},st=x;
    // Taylor/Lagrange remainder, |x| <= 1/16 (up to outward endpoint
    // expansion). Derivatives of exact sin/cos have magnitude <= 1.
    // No call to sin, cos, pi reduction or a correctly-rounded libm premise.
    // Verified arithmetic follows the shared Rump2010 convention;
    // full derivation and reference boundary: scientific contract 5.59.
    for(int j=1;j<=12;++j) {
        ct=divide(si::product(ct,negative),double((2*j-1)*(2*j)));
        st=divide(si::product(st,negative),double((2*j)*(2*j+1)));
        cosine=si::add(cosine,ct);sine=si::add(sine,st);
    }
    // Degree 25 cos polynomial has zero odd term, hence order-26 remainder;
    // degree 26 sin polynomial has zero even term, hence order-27 remainder.
    const auto cr=divide(si::product(ct,squared),26.*25.);
    const auto sr=divide(si::product(st,squared),27.*26.);
    const double ce=std::max(std::abs(cr.lo),std::abs(cr.hi)),se=std::max(std::abs(sr.lo),std::abs(sr.hi));
    si::ComplexInterval phase{si::add(cosine,{-ce,ce}),si::add(sine,{-se,se})};
    auto circle=[](si::ComplexInterval z) {
        z.real.lo=std::max(-1.,z.real.lo);z.real.hi=std::min(1.,z.real.hi);
        z.imag.lo=std::max(-1.,z.imag.lo);z.imag.hi=std::min(1.,z.imag.hi);
        if(z.real.lo>z.real.hi || z.imag.lo>z.imag.hi)throw std::logic_error("MANTA phase enclosure lost the unit circle");
        return z;
    };
    phase=circle(phase);
    for(int j=0;j<scaling;++j)phase=circle(si::product(phase,phase));
    BlochPhaseCertificate out;out.real_lower=phase.real.lo;out.real_upper=phase.real.hi;
    out.imag_lower=phase.imag.lo;out.imag_upper=phase.imag.hi;out.squarings=scaling;
    if(phase.real.lo!=evaluated.real() || phase.real.hi!=evaluated.real() ||
       phase.imag.lo!=evaluated.imag() || phase.imag.hi!=evaluated.imag())
        out.error_bound=si::magnitude(si::sub(si::point(evaluated),phase));
    return out;
}

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
    const auto capability=md_producer_capability({ket?"fourier":"auxiliary-fourier",0});
    if(!capability.supported) throw std::invalid_argument(capability.reason);
    const auto a=census(bra,ket?capability.maximum_l:capability.maximum_auxiliary_l);
    Census b; b.primitives=b.maximum_size=b.maximum_cartesian=1;
    if(ket) b=census(*ket,capability.maximum_l);
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
