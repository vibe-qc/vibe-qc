#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/manta_radial.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::manta {
namespace {
Count add(Count a,Count b) {
    if(b>std::numeric_limits<Count>::max()-a) throw std::overflow_error("MANTA MD resource sum overflow");
    return a+b;
}
Count mul(Count a,Count b) {
    if(a && b>std::numeric_limits<Count>::max()/a) throw std::overflow_error("MANTA MD resource product overflow");
    return a*b;
}
Count square(Count a) { return mul(a,a); }
Count fourth(Count a) { return square(square(a)); }
void cap(Count a,Count b,const char* message) { if(a>b) throw std::length_error(message); }
void finite(double a) { if(!std::isfinite(a)) throw std::invalid_argument("MANTA MD requires finite basis data"); }
} // namespace

Capability md_capability(const KernelRequest& r) {
    Capability c;
    c.maximum_l=6;
    c.reason=md_request_error(r,4);
    c.supported=c.reason.empty();
    return c;
}
namespace {
BuildPlan plan_md_impl(const BasisSet& basis,const KernelRequest& request,const ResourceBudget& b,Count channel_batch,bool tiles_only) {
    const auto capability=md_capability(request);
    if(!capability.supported) throw std::invalid_argument(capability.reason);
    BuildPlan p; p.provider="manta";
    if(tiles_only) p.schedule="shell-tiles";
    if(!channel_batch || channel_batch>64) throw std::invalid_argument("MANTA channel batch must be in 1..64");
    if(tiles_only && channel_batch!=1) throw std::invalid_argument("MANTA shell-tile admission requires one quartet per call");
    p.channel_batch=channel_batch;
    p.n_basis=basis.nbasis(); p.n_shells=basis.nshells();
    p.basis_payload_bytes=basis.name().size();
    if(!p.n_basis || !p.n_shells) throw std::invalid_argument("MANTA MD requires a nonempty basis");
    if(b.backend_margin_bytes<(Count{64}<<10)) throw std::invalid_argument("MANTA MD backend margin must be at least 64 KiB");
    p.shell_calls=tiles_only?1:fourth(p.n_shells); p.scalar_terms=tiles_only?0:mul(2,fourth(p.n_basis));
    cap(p.shell_calls,b.shell_calls,"MANTA MD shell-call budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA MD scalar-contraction budget exceeded");
    p.target_bytes=tiles_only?0:mul(16,square(p.n_basis));
    p.modeled_peak_bytes=add(mul(5,p.target_bytes),add(b.other_live_bytes,b.backend_margin_bytes));
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA MD matrix/input memory budget exceeded");
    Count ao_count=0,primitive_count=0,max_size=0;
    for(const auto& s:basis.libint()) {
        if(s.alpha.empty() || s.contr.size()!=1 || s.contr[0].coeff.size()!=s.alpha.size())
            throw std::invalid_argument("MANTA MD requires nonempty single-contraction shell channels");
        const int l=s.contr[0].l;
        if(l<0 || l>capability.maximum_l) throw std::invalid_argument("MANTA MD angular capability exceeded");
        p.maximum_l=std::max(p.maximum_l,l);
        p.maximum_primitives=std::max<Count>(p.maximum_primitives,s.alpha.size());
        primitive_count=add(primitive_count,s.alpha.size());
        max_size=std::max<Count>(max_size,s.size()); ao_count=add(ao_count,s.size());
        p.basis_payload_bytes=add(p.basis_payload_bytes,add(24,mul(24,s.alpha.size())));
    }
    if(ao_count!=p.n_basis) throw std::invalid_argument("MANTA MD basis AO census mismatch");
    p.primitive_quartets_upper_bound=fourth(tiles_only?p.maximum_primitives:primitive_count);
    cap(p.primitive_quartets_upper_bound,b.primitive_quartets,"MANTA MD primitive-quartet budget exceeded");
    const Count l=p.maximum_l,c=(l+1)*(l+2)/2,h=(2*l+1)*(2*l+2)*(2*l+3)/6;
    const Count c2=square(c),c4=square(c2),h2=square(h),rec=fourth(4*l+1);
    const Count products=add(mul(2,mul(c2,h2)),mul(2,mul(c4,h)));
    // Count-based upper model includes E/R work, both GEMMs and shell
    // representation transforms. It is not a hardware instruction counter.
    p.kernel_terms=add(mul(p.primitive_quartets_upper_bound,add(radial_work(request,4*l),add(products,add(mul(6,rec),mul(24,mul(c2,h)))))),
                       mul(p.shell_calls,mul(4,mul(c4,c2))));
    if(channel_batch>1) {
        const Count lookup=mul(128,mul(square(channel_batch),add(p.maximum_primitives,1)));
        p.kernel_terms=add(p.kernel_terms,mul(p.primitive_quartets_upper_bound,add(lookup,mul(8,c4))));
    }
    cap(p.kernel_terms,b.kernel_terms,"MANTA MD kernel-work budget exceeded");
    p.tile_elements=fourth(max_size);
    if(tiles_only) {
        p.scalar_terms=p.tile_elements;
        cap(p.scalar_terms,b.scalar_terms,"MANTA MD tile-element budget exceeded");
    }
    const Count axes=mul(6,mul(square(l+1),2*l+1));
    const Count elements=add(add(mul(4,mul(c2,h)),h2),add(mul(8,c4),add(rec,add(axes,add(mul(6,h),4096)))));
    p.engine_bytes=add(mul(8,elements),radial_workspace(request));
    const Count controls=add(4096,mul(p.n_shells,sizeof(libint2::Shell)+sizeof(libint2::Shell::Contraction)+3*sizeof(std::size_t)));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(p.engine_bytes,add(mul(16,p.tile_elements),add(mul(2,controls),mul(2,p.basis_payload_bytes)))));
    if(channel_batch>1) {
        // Additional Cartesian sums/corrections, borrowed tile storage, map
        // nodes and bounded group controls. One primitive workspace is reused.
        const Count matrices=add(mul(16,c4),mul(8,p.tile_elements));
        const Count maps=mul(8192,add(p.maximum_primitives,1));
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(mul(channel_batch-1,matrices),mul(channel_batch,maps)));
    }
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA MD provider/workspace memory budget exceeded");
    for(const auto& s:basis.libint()) {
        for(auto x:s.O) finite(x);
        for(auto x:s.alpha) { finite(x); if(x<=0) throw std::invalid_argument("MANTA MD requires positive exponents"); }
        for(auto x:s.contr[0].coeff) finite(x);
    }
    return p;
}
} // namespace
BuildPlan plan_md(const BasisSet& basis,const KernelRequest& request,const ResourceBudget& budget,Count channel_batch) {
    return plan_md_impl(basis,request,budget,channel_batch,false);
}
BuildPlan plan_md_tiles(const BasisSet& basis,const KernelRequest& request,const ResourceBudget& budget) {
    return plan_md_impl(basis,request,budget,1,true);
}
MantaProvider::MantaProvider(const BasisSet& basis,const KernelRequest& r,const ResourceBudget& budget,Count channel_batch,bool tiles_only)
    : plan_(plan_md_impl(basis,r,budget,channel_batch,tiles_only)),basis_(basis),request_(r) {
    const std::string payload="manta.md.source.v1;"+auxiliary_basis_content_identity_sha256(basis_)
        +";md-fp64-v1;"+r.kernel+";real;derivative=0;images=0;precision=0;libint-shell-axial-v1"+range_identity_suffix(r);
    detail::Sha256 hash;
    hash.update(reinterpret_cast<const std::uint8_t*>(payload.data()),payload.size()); source_=hash.finish_hex();
}
TileView MantaProvider::evaluate(const std::array<std::size_t,4>& indices,Workspace& workspace) const {
    if(workspace.backend_!="manta" || workspace.values_.size()<plan_.tile_elements)
        throw std::invalid_argument("MANTA MD workspace provider or capacity mismatch");
    std::array<const libint2::Shell*,4> shells{};
    TileView tile;
    for(std::size_t i=0;i<4;++i) {
        if(indices[i]>=basis_.nshells()) throw std::out_of_range("MANTA shell index");
        shells[i]=&basis_.libint()[indices[i]]; tile.shape[i]=shells[i]->size();
    }
    const auto values=md::shell_coulomb(shells,radial_kernel(request_),request_.omega);
    std::copy_n(values.data(),values.size(),workspace.values_.data());
    tile.values=workspace.values_.data();
    return tile;
}
TileBatch MantaProvider::evaluate_batch(const std::vector<std::array<std::size_t,4>>& indices,Workspace& workspace) const {
    if(indices.empty() || indices.size()>plan_.channel_batch)
        throw std::invalid_argument("MANTA quartet batch exceeds admitted channel extent");
    if(workspace.backend_!="manta" || workspace.values_.size()<plan_.tile_elements*indices.size())
        throw std::invalid_argument("MANTA MD batch workspace provider or capacity mismatch");
    TileBatch result; result.primitive_evaluations_measured=true;
    result.tiles.resize(indices.size());
    const auto& basis=basis_.libint();
    if(plan_.channel_batch==1) {
        result.tiles[0]=evaluate(indices[0],workspace);
        Count count=1;
        for(auto shell:indices[0]) count*=basis[shell].alpha.size();
        result.primitive_quartets_evaluated=count;
        return result;
    }
    std::vector<std::array<const libint2::Shell*,4>> channels(indices.size());
    std::vector<std::vector<std::size_t>> groups;
    for(std::size_t i=0;i<indices.size();++i) {
        for(int axis=0;axis<4;++axis) {
            if(indices[i][axis]>=basis.size()) throw std::out_of_range("MANTA shell index");
            channels[i][axis]=&basis[indices[i][axis]];
            result.tiles[i].shape[axis]=channels[i][axis]->size();
        }
        bool inserted=false;
        for(auto& group:groups) {
            bool compatible=true;
            for(int axis=0;axis<4;++axis) {
                const auto* a=channels[i][axis]; const auto* b=channels[group[0]][axis];
                compatible=compatible && a->contr[0].l==b->contr[0].l && a->O==b->O;
            }
            if(compatible) { group.push_back(i); inserted=true; break; }
        }
        if(!inserted) groups.push_back({i});
    }
    for(const auto& group:groups) {
        std::vector<std::array<const libint2::Shell*,4>> selected;
        for(auto i:group) selected.push_back(channels[i]);
        auto tiles=md::shell_coulomb_shared(selected,radial_kernel(request_),request_.omega);
        result.primitive_quartets_evaluated+=tiles.primitive_evaluations;
        for(std::size_t g=0;g<group.size();++g) {
            const auto i=group[g];
            auto* target=workspace.values_.data()+i*plan_.tile_elements;
            std::copy_n(tiles.values[g].data(),tiles.values[g].size(),target);
            result.tiles[i].values=target;
        }
    }
    return result;
}
Eigen::MatrixXd DirectJKBuilder::build_K_erf(const Eigen::MatrixXd& density,double omega) const {
    KernelRequest request;
    request.kernel="erf"; request.omega=omega;
    const auto error=md_request_error(request,4);
    if(!error.empty()) throw std::invalid_argument(error);
    validate_density(density);
    return DirectJKBuilder(provider().basis(),request,budget(),plan().channel_batch).build_K(density);
}
} // namespace vibeqc::manta
