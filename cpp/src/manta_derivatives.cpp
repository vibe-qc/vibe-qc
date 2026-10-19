#include "vibeqc/manta_derivatives.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"

namespace vibeqc::manta {
namespace {
std::size_t slot(std::size_t axis,std::size_t arity) { return arity==4?axis:axis==0?0:axis+1; }
}
DerivativePlan plan_derivatives(const std::vector<const BasisSet*>& axes,const KernelRequest& request,const ResourceBudget& budget) {
    using namespace admission;
    if(axes.size()<2 || axes.size()>4) throw std::invalid_argument("MANTA derivatives require two, three or four basis axes");
    if(request.derivative!=1) throw std::invalid_argument("MANTA derivative tiles require derivative=1");
    auto scalar=request; scalar.derivative=0;
    const auto error=md_request_error(scalar,static_cast<int>(axes.size()));
    if(!error.empty()) throw std::invalid_argument(error);
    margin(budget);
    DerivativePlan p;
    p.components=3*axes.size(); p.shell_calls=p.primitive_products=p.integral_elements=p.tile_elements=1;
    Count bytes=0;
    std::array<int,4> angular{};
    for(const auto* basis:axes) {
        if(!basis) throw std::invalid_argument("MANTA derivative basis must not be null");
        p.shape.push_back(basis->nbasis()); p.shell_extents.push_back(basis->nshells());
        p.shell_calls=mul(p.shell_calls,basis->nshells());
        p.integral_elements=mul(p.integral_elements,basis->nbasis());
    }
    p.shell_calls=mul(p.shell_calls,p.components); p.integral_elements=mul(p.integral_elements,p.components);
    cap(p.shell_calls,budget.shell_calls,"MANTA derivative shell-call budget exceeded");
    cap(p.integral_elements,budget.scalar_terms,"MANTA derivative element-work budget exceeded");
    p.modeled_peak_bytes=add(budget.other_live_bytes,budget.backend_margin_bytes);
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA derivative live-memory budget exceeded");
    for(std::size_t i=0;i<axes.size();++i) {
        const auto c=census(*axes[i],axes.size()==4 || (axes.size()==3 && i>0)?6:7);
        p.primitive_products=mul(p.primitive_products,c.primitives);
        p.tile_elements=mul(p.tile_elements,c.maximum_size);
        bytes=add(bytes,c.bytes); angular[slot(i,axes.size())]=c.maximum_l;
    }
    p.primitive_products=mul(p.primitive_products,p.components);
    p.primitive_evaluations_upper_bound=mul(2,p.primitive_products);
    cap(p.primitive_evaluations_upper_bound,budget.primitive_quartets,"MANTA derivative primitive-evaluation budget exceeded");
    Count maximum_workspace=0,maximum_work=0,maximum_transform=0;
    for(std::size_t i=0;i<axes.size();++i) {
        auto shifted=angular; ++shifted[slot(i,axes.size())];
        std::array<Count,4> cart{};
        for(int k=0;k<4;++k) cart[k]=(shifted[k]+1)*(shifted[k]+2)/2;
        const Count ca=mul(cart[0],cart[1]),cb=mul(cart[2],cart[3]),tile=mul(ca,cb);
        const int la=shifted[0]+shifted[1],lb=shifted[2]+shifted[3],l=la+lb;
        const Count ha=(la+1)*(la+2)*(la+3)/6,hb=(lb+1)*(lb+2)*(lb+3)/6;
        const Count ea=mul(ca,ha),eb=mul(cb,hb),kernel=mul(ha,hb),rec=fourth(l+1);
        const Count axes_work=6*(l+1)*(l+1)*(l+1);
        const Count gemm=mul(2,add(mul(ca,kernel),mul(tile,hb)));
        const Count work=add(gemm,add(mul(8,rec),add(mul(24,add(ea,eb)),add(axes_work,add(radial_work(request,l),mul(16,tile))))));
        maximum_work=std::max(maximum_work,work);
        maximum_transform=std::max(maximum_transform,mul(8,mul(tile,ca+cb)));
        const Count workspace=mul(8,add(add(mul(4,add(ea,eb)),mul(2,kernel)),
              add(mul(10,tile),add(rec,add(axes_work,add(mul(4,add(square(ca),square(cb))),4096))))));
        maximum_workspace=std::max(maximum_workspace,workspace);
    }
    p.kernel_terms=add(mul(p.primitive_evaluations_upper_bound,maximum_work),mul(p.shell_calls,maximum_transform));
    cap(p.kernel_terms,budget.kernel_terms,"MANTA derivative kernel-work budget exceeded");
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(radial_workspace(request),
        add(maximum_workspace,add(mul(2,bytes),mul(16,p.tile_elements)))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA derivative provider/workspace memory budget exceeded");
    for(const auto* basis:axes) validate_basis(*basis);
    return p;
}
DerivativeTiles::DerivativeTiles(const std::vector<const BasisSet*>& axes,const KernelRequest& request,const ResourceBudget& budget)
    :plan_(plan_derivatives(axes,request,budget)),request_(request) {
    std::string payload="manta.coulomb-derivatives.v1;md-fp64-v1;"+request.kernel+
        ";real;derivative=1;images=0;libint-shell-axial-v1;"+std::to_string(axes.size());
    bases_.reserve(axes.size());
    for(const auto* basis:axes) {
        bases_.push_back(*basis); payload+=";"+auxiliary_basis_content_identity_sha256(bases_.back());
    }
    payload+=range_identity_suffix(request);
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size()); source_=hash.finish_hex();
}
DerivativeTile DerivativeTiles::evaluate(const std::vector<std::size_t>& indices,int center,int axis) const {
    if(indices.size()!=bases_.size()) throw std::invalid_argument("MANTA derivative shell-index arity mismatch");
    if(center<0 || center>=static_cast<int>(bases_.size()) || axis<0 || axis>2)
        throw std::out_of_range("MANTA derivative center/axis");
    std::array<const libint2::Shell*,4> shells{};
    DerivativeTile tile; tile.primitive_products=1;
    for(std::size_t i=0;i<indices.size();++i) {
        if(indices[i]>=bases_[i].nshells()) throw std::out_of_range("MANTA derivative shell index");
        const auto& s=bases_[i].libint()[indices[i]];
        shells[slot(i,indices.size())]=&s; tile.shape.push_back(s.size()); tile.primitive_products*=s.alpha.size();
    }
    auto derivative=md::shell_coulomb_derivative(shells,slot(center,indices.size()),axis,radial_kernel(request_),request_.omega);
    tile.values=std::move(derivative.values); tile.primitive_evaluations=derivative.primitive_evaluations;
    return tile;
}
} // namespace vibeqc::manta
