#include "vibeqc/manta_fitting.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"

namespace vibeqc::manta {
FittingPlan plan_fitting(const std::vector<const BasisSet*>& axes,const KernelRequest& request,const ResourceBudget& budget) {
    using namespace admission;
    if(axes.size()!=2 && axes.size()!=3) throw std::invalid_argument("MANTA fitting tiles require two or three basis axes");
    if(request.centers!=static_cast<int>(axes.size()))
        throw std::invalid_argument("MANTA fitting request center count must match the basis axes");
    if(request.derivative!=0) throw std::invalid_argument("MANTA fitting tiles require derivative=0; use derivative tiles for gradients");
    const auto capability=md_capability(request);
    if(!capability.supported) throw std::invalid_argument(capability.reason);
    margin(budget);
    FittingPlan p;
    p.shell_calls=p.primitive_products=p.integral_elements=p.tile_elements=1;
    Count bytes=0;
    std::array<Count,3> cart{1,1,1};
    std::array<int,3> angular{};
    for(std::size_t i=0;i<axes.size();++i) {
        if(!axes[i]) throw std::invalid_argument("MANTA fitting basis must not be null");
        p.shape.push_back(axes[i]->nbasis()); p.shell_extents.push_back(axes[i]->nshells());
        p.shell_calls=mul(p.shell_calls,p.shell_extents.back());
        p.integral_elements=mul(p.integral_elements,p.shape.back());
    }
    cap(p.shell_calls,budget.shell_calls,"MANTA fitting shell-call budget exceeded");
    cap(p.integral_elements,budget.scalar_terms,"MANTA fitting element-work budget exceeded");
    p.modeled_peak_bytes=add(budget.other_live_bytes,budget.backend_margin_bytes);
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA fitting live-memory budget exceeded");
    for(std::size_t i=0;i<axes.size();++i) {
        const auto c=census(*axes[i],md_axis_limit(capability,static_cast<int>(axes.size()),i));
        p.primitive_products=mul(p.primitive_products,c.primitives);
        p.tile_elements=mul(p.tile_elements,c.maximum_size);
        bytes=add(bytes,c.bytes);
        cart[i]=c.maximum_cartesian; angular[i]=c.maximum_l;
    }
    cap(p.primitive_products,budget.primitive_quartets,"MANTA fitting primitive-product budget exceeded");
    const Count ca=cart[0],cb=mul(cart[1],cart[2]),tile=mul(ca,cb);
    const int la=angular[0],lb=angular[1]+angular[2],l=la+lb;
    const Count ha=(la+1)*(la+2)*(la+3)/6,hb=(lb+1)*(lb+2)*(lb+3)/6;
    const Count ea=mul(ca,ha),eb=mul(cb,hb),kernel=mul(ha,hb),rec=fourth(l+1);
    const Count axes_work=6*(l+1)*(l+1)*(l+1);
    const Count gemm=mul(2,add(mul(ca,kernel),mul(tile,hb)));
    const Count primitive_work=add(gemm,add(mul(8,rec),add(mul(24,add(ea,eb)),add(axes_work,radial_work(request,l)))));
    p.kernel_terms=add(mul(p.primitive_products,primitive_work),mul(p.shell_calls,mul(8,mul(tile,ca+cb))));
    cap(p.kernel_terms,budget.kernel_terms,"MANTA fitting kernel-work budget exceeded");
    const Count workspace=mul(8,add(add(mul(4,add(ea,eb)),mul(2,kernel)),
          add(mul(8,tile),add(rec,add(axes_work,add(mul(4,add(square(ca),square(cb))),4096))))));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(radial_workspace(request),add(workspace,add(mul(2,bytes),mul(16,p.tile_elements)))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA fitting provider/workspace memory budget exceeded");
    for(const auto* basis:axes) validate_basis(*basis);
    return p;
}
FittingTiles::FittingTiles(const std::vector<const BasisSet*>& axes,const KernelRequest& request,const ResourceBudget& budget)
    : plan_(plan_fitting(axes,request,budget)),request_(request) {
    std::string payload="manta.fitting.v1;md-fp64-v1;"+request.kernel+";real;derivative=0;images=0;libint-shell-axial-v1;"
        +std::to_string(axes.size());
    bases_.reserve(axes.size());
    for(const auto* basis:axes) {
        bases_.push_back(*basis);
        payload+=";"+auxiliary_basis_content_identity_sha256(bases_.back());
    }
    payload+=range_identity_suffix(request);
    vibeqc::detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size()); source_=hash.finish_hex();
}
FittingTile FittingTiles::evaluate(const std::vector<std::size_t>& indices) const {
    if(indices.size()!=bases_.size()) throw std::invalid_argument("MANTA fitting shell-index arity mismatch");
    std::array<const libint2::Shell*,4> shells{};
    FittingTile tile;
    tile.primitive_products=1;
    for(std::size_t i=0;i<indices.size();++i) {
        if(indices[i]>=bases_[i].nshells()) throw std::out_of_range("MANTA fitting shell index");
        const auto& shell=bases_[i].libint()[indices[i]];
        shells[i==0?0:i+1]=&shell;
        tile.shape.push_back(shell.size());
        tile.primitive_products*=shell.alpha.size();
    }
    tile.values=md::shell_coulomb(shells,radial_kernel(request_),request_.omega);
    return tile;
}
} // namespace vibeqc::manta
