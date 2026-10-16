#include "vibeqc/manta_one_electron.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
KernelRequest range_request(const OneElectronRequest& request) {
    KernelRequest radial; radial.kernel=request.kernel; radial.omega=request.omega;
    return radial;
}
int request_order(const OneElectronRequest& r) {
    if(r.operation!="overlap" && r.operation!="kinetic" && r.operation!="potential" && r.operation!="moment")
        throw std::invalid_argument("MANTA one-electron operation must be overlap, kinetic, potential or moment");
    const auto error=md_request_error(range_request(r),4);
    if(!error.empty()) throw std::invalid_argument(error);
    if(r.operation!="potential" && r.kernel!="coulomb")
        throw std::invalid_argument("MANTA range separation applies only to the potential operation");
    if(r.operation!="potential" && !r.charges.empty())
        throw std::invalid_argument("MANTA charges apply only to the potential operation");
    int degree=0;
    for(auto n:r.powers) {
        if(n<0 || n>6) throw std::invalid_argument("MANTA polynomial moment order must be between zero and six");
        degree+=n;
    }
    if(degree>6) throw std::invalid_argument("MANTA total polynomial moment order exceeds six");
    if(r.operation!="moment") {
        if(degree) throw std::invalid_argument("MANTA powers apply only to polynomial moments");
        for(auto x:r.origin) if(x!=0) throw std::invalid_argument("MANTA origin applies only to polynomial moments");
    }
    return degree;
}
std::string source_identity(const BasisSet& bra,const BasisSet& ket,const OneElectronRequest& r) {
    vibeqc::detail::Sha256 hash;
    const std::string prefix="manta.one-electron.v1;md-fp64-v1;libint-shell-axial-v1;"
        +auxiliary_basis_content_identity_sha256(bra)+";"+auxiliary_basis_content_identity_sha256(ket)+";"+r.operation+";"
        +(r.kernel=="coulomb"?"":r.kernel+range_identity_suffix(range_request(r))+";");
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    auto word=[&hash](std::uint64_t bits) {
        std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(bits>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    };
    auto number=[&word](double x) {
        if(x==0) x=0.;
        std::uint64_t bits; std::memcpy(&bits,&x,sizeof(bits)); word(bits);
    };
    for(auto n:r.powers) word(n);
    for(auto x:r.origin) number(x);
    word(r.charges.size());
    for(const auto& charge:r.charges) { number(charge.first); for(auto x:charge.second) number(x); }
    return hash.finish_hex();
}
void validate_derivative(const OneElectronRequest& request,const OneElectronDerivative& d) {
    if(d.target!="bra" && d.target!="ket" && d.target!="charge" && d.target!="origin")
        throw std::invalid_argument("MANTA one-electron derivative target must be bra, ket, charge or origin");
    if(d.axis<0 || d.axis>2) throw std::invalid_argument("MANTA one-electron derivative axis must be zero, one or two");
    if(d.target=="charge") {
        if(request.operation!="potential" || d.charge_index>=request.charges.size())
            throw std::invalid_argument("MANTA charge derivative requires a valid potential charge index");
    } else if(d.charge_index) throw std::invalid_argument("MANTA charge index applies only to charge derivatives");
    if(d.target=="origin" && request.operation!="moment")
        throw std::invalid_argument("MANTA origin derivative requires a polynomial moment");
}

OneElectronPlan plan_impl(const BasisSet& bra,const BasisSet& ket,
                                  const OneElectronRequest& request,const ResourceBudget& budget,const OneElectronDerivative* derivative) {
    using namespace admission;
    if(derivative) validate_derivative(request,*derivative);
    const int degree=request_order(request);
    const auto radial=range_request(request);
    margin(budget);
    OneElectronPlan p;
    p.shape={bra.nbasis(),ket.nbasis()};
    p.shell_calls=mul(bra.nshells(),ket.nshells());
    cap(p.shell_calls,budget.shell_calls,"MANTA one-electron shell-call budget exceeded");
    const Count elements=mul(p.shape[0],p.shape[1]);
    cap(elements,budget.scalar_terms,"MANTA one-electron element budget exceeded");
    cap(elements,std::numeric_limits<Eigen::Index>::max(),"MANTA one-electron matrix is not addressable");
    p.output_bytes=mul(8,elements);
    p.modeled_peak_bytes=add(mul(3,p.output_bytes),add(budget.other_live_bytes,budget.backend_margin_bytes));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(32,request.charges.size()));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA one-electron matrix/input memory budget exceeded");
    const auto a=census(bra,6),b=census(ket,6);
    p.primitive_pairs=mul(a.primitives,b.primitives);
    const bool center=derivative && (derivative->target=="bra" || derivative->target=="ket");
    p.primitive_evaluations_upper_bound=mul(center?2:1,p.primitive_pairs);
    cap(p.primitive_evaluations_upper_bound,budget.primitive_quartets,"MANTA one-electron primitive-pair budget exceeded");
    const int la=a.maximum_l+(center && derivative->target=="bra"),lb=b.maximum_l+(center && derivative->target=="ket");
    const Count ca=(la+1)*(la+2)/2,cb=(lb+1)*(lb+2)/2,tile=mul(ca,cb);
    const int l=la+lb+(derivative && derivative->target=="charge");
    const int ea=la+(request.operation=="kinetic"?1:degree),eb=lb+(request.operation=="kinetic"?1:0);
    const Count axes=3*(ea+1)*(eb+1)*(ea+eb+1);
    const Count h=(l+1)*(l+2)*(l+3)/6,rec=fourth(l+1),e=mul(tile,h);
    Count work=add(mul(12,axes),mul(24,mul(tile,degree+4)));
    if(request.operation=="potential") {
        work=add(work,mul(24,e));
        work=add(work,mul(request.charges.size(),add(mul(8,rec),add(mul(2,e),radial_work(radial,l)))));
    }
    p.kernel_terms=add(mul(p.primitive_evaluations_upper_bound,work),mul(p.shell_calls,mul(8,mul(tile,ca+cb))));
    cap(p.kernel_terms,budget.kernel_terms,"MANTA one-electron kernel-work budget exceeded");
    p.tile_elements=mul(a.maximum_size,b.maximum_size);
    const Count workspace=mul(8,add(add(mul(derivative?16:8,tile),mul(2,e)),add(mul(2,rec),add(mul(2,axes),add(8*h+4096,add(square(ca),square(cb)))))));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(radial_workspace(radial),add(workspace,add(a.bytes,b.bytes))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA one-electron workspace memory budget exceeded");
    validate_basis(bra); validate_basis(ket);
    for(auto x:request.origin) require_finite(x);
    for(const auto& charge:request.charges) { require_finite(charge.first); for(auto x:charge.second) require_finite(x); }
    return p;
}
OneElectronResult compute_impl(const BasisSet& bra,const BasisSet& ket,
                                       const OneElectronRequest& request,const ResourceBudget& budget,const OneElectronDerivative* derivative) {
    const auto plan=plan_impl(bra,ket,request,budget,derivative);
    OneElectronResult result;
    result.receipt.source_identity=source_identity(bra,ket,request);
    result.receipt.operation=request.operation;
    result.receipt.kernel=request.kernel; result.receipt.omega=request.omega==0?0.:request.omega;
    const auto radial=radial_kernel(range_request(request));
    if(derivative) {
        const std::string payload="manta.one-electron-derivative.v1;"+result.receipt.source_identity+";"+
            derivative->target+";"+std::to_string(derivative->axis)+";"+std::to_string(derivative->charge_index);
        vibeqc::detail::Sha256 hash;
        hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size());
        result.receipt.source_identity=hash.finish_hex();
        result.receipt.derivative_target=derivative->target;
        result.receipt.derivative_axis=derivative->axis;
        result.receipt.derivative_charge_index=derivative->charge_index;
    }
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    result.values.resize(plan.shape[0],plan.shape[1]);
    Eigen::Index row=0;
    for(const auto& a:bra.libint()) {
        Eigen::Index col=0;
        for(const auto& b:ket.libint()) {
            const auto values=derivative?
                md::shell_one_electron_derivative(a,b,request.operation,request.charges,request.origin,request.powers,
                                                 derivative->target,derivative->axis,derivative->charge_index,radial,request.omega):
                md::shell_one_electron(a,b,request.operation,request.charges,request.origin,request.powers,radial,request.omega);
            result.values.block(row,col,a.size(),b.size())=values;
            ++result.receipt.shell_calls;
            const Count pairs=a.alpha.size()*b.alpha.size();
            result.receipt.primitive_pairs+=pairs;
            Count evaluations=1;
            if(derivative) {
                if(derivative->target=="bra") evaluations=a.contr[0].l?2:1;
                else if(derivative->target=="ket") evaluations=b.contr[0].l?2:1;
                else if(derivative->target=="origin" && !request.powers[derivative->axis]) evaluations=0;
            }
            result.receipt.primitive_integrals_evaluated+=pairs*evaluations;
            result.receipt.integral_elements+=a.size()*b.size();
            col+=b.size();
        }
        row+=a.size();
    }
    return result;
}
} // namespace
OneElectronPlan plan_one_electron(const BasisSet& bra,const BasisSet& ket,
                                  const OneElectronRequest& request,const ResourceBudget& budget) {
    return plan_impl(bra,ket,request,budget,nullptr);
}
OneElectronResult compute_one_electron(const BasisSet& bra,const BasisSet& ket,
                                       const OneElectronRequest& request,const ResourceBudget& budget) {
    return compute_impl(bra,ket,request,budget,nullptr);
}
OneElectronPlan plan_one_electron_derivative(const BasisSet& bra,const BasisSet& ket,
                                  const OneElectronRequest& request,const OneElectronDerivative& derivative,const ResourceBudget& budget) {
    return plan_impl(bra,ket,request,budget,&derivative);
}
OneElectronResult compute_one_electron_derivative(const BasisSet& bra,const BasisSet& ket,
                                  const OneElectronRequest& request,const OneElectronDerivative& derivative,const ResourceBudget& budget) {
    return compute_impl(bra,ket,request,budget,&derivative);
}
} // namespace vibeqc::manta
