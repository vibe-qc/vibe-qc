#include "vibeqc/manta_molecular_source.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>
#include <typeinfo>

namespace vibeqc::manta {
namespace {
using namespace admission;
void require(bool ok,const char* message) { if(!ok) throw std::invalid_argument(message); }
const IntegralProvider& source(const OrderedJKBuilder& b) {
    const auto& p=b.provider();
    require(typeid(p)==typeid(MantaProvider) || typeid(p)==typeid(LibintProvider),
        "molecular source support requires an immutable native provider");
    const auto r=p.kernel_request();
    require(r.kernel=="coulomb" && r.omega==0. && r.scalar=="real" && r.centers==4 &&
        r.derivative==0 && r.images==decltype(r.images){},
        "molecular source support requires real four-center Coulomb without images or derivatives");
    require(p.plan().provider==p.name() && p.plan().kernel==r.kernel && p.plan().omega==r.omega,
        "molecular source support provider plan mismatch");
    return p;
}
void word(detail::Sha256& h,Count value) {
    std::uint8_t bytes[8];
    for(int j=0;j<8;++j) bytes[j]=value>>(56-8*j);
    h.update(bytes,8);
}
void real(detail::Sha256& h,double value) {
    // Versioned big-endian IEEE binary64; signed zero has one identity.
    if(value==0.) value=0.;
    Count bits; static_assert(sizeof(bits)==sizeof(value));
    std::memcpy(&bits,&value,sizeof(bits)); word(h,bits);
}
}
MolecularSourceSupportPlan plan_molecular_source_support(
    const OrderedJKBuilder& b,Count ng,const ResourceBudget& budget) {
    const auto& p=source(b); const auto& bp=p.plan();
    require(ng>0,"molecular source support requires operations");
    MolecularSourceSupportPlan out;
    out.operations=ng; out.shell_checks=mul(ng,bp.n_shells);
    // Count-only upper reservation before reading shell/operation values.
    out.radial_values=mul(2,mul(out.shell_checks,bp.maximum_primitives));
    out.scalar_terms=add(mul(256,out.shell_checks),mul(64,out.radial_values));
    // Input, binding snapshot and retained certificate map can overlap.
    out.additional_bytes=add(65536,add(mul(24,out.shell_checks),add(mul(192,ng),bp.n_shells)));
    out.modeled_peak_bytes=add(budget.other_live_bytes,add(bp.modeled_peak_bytes,out.additional_bytes));
    cap(out.modeled_peak_bytes,budget.memory_bytes,"molecular source support memory budget exceeded");
    cap(out.scalar_terms,budget.scalar_terms,"molecular source support work budget exceeded");
    cap(out.additional_bytes,std::numeric_limits<std::size_t>::max(),"molecular source support address overflow");
    return out;
}
MolecularSourceSupport::MolecularSourceSupport(std::shared_ptr<OrderedJKBuilder> b,Count ng,
    const double* rotations,const double* translations,const std::int64_t* destinations,
    double tolerance,const ResourceBudget& budget):builder_(std::move(b)) {
    require(bool(builder_),"molecular source support requires a provider");
    plan_=plan_molecular_source_support(*builder_,ng,budget);
    require(rotations && translations && destinations,"molecular source support input required");
    require(std::isfinite(tolerance) && tolerance>=0.,"invalid molecular source geometry tolerance");
    tolerance_=tolerance;
    const auto& p=source(*builder_); source_=p.source_identity();
    const auto& shells=p.basis().libint(); const Count ns=shells.size();
    std::vector<std::uint8_t> seen(ns);
    detail::Sha256 hash;
    const std::string tag="manta.molecular-source-support.v1;"+source_+
        ";libint-shell-axial-v1;precision=0;complete-ordered-support;no-density-screening;no-partition";
    hash.update(reinterpret_cast<const std::uint8_t*>(tag.data()),tag.size());
    word(hash,ng); word(hash,ns); real(hash,tolerance);
    Count radial=0;
    for(Count g=0;g<ng;++g) {
        const auto* r=rotations+9*g; const auto* t=translations+3*g;
        for(int i=0;i<9;++i) { require_finite(r[i]); real(hash,r[i]); }
        for(int i=0;i<3;++i) { require_finite(t[i]); real(hash,t[i]); }
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
            double v=0.; for(int k=0;k<3;++k) v+=r[3*k+i]*r[3*k+j];
            require(std::isfinite(v) && std::abs(v-(i==j?1.:0.))<=tolerance,
                "molecular source operation is not orthogonal");
        }
        std::fill(seen.begin(),seen.end(),0);
        for(Count s=0;s<ns;++s) {
            const auto d=destinations[g*ns+s];
            require(d>=0 && Count(d)<ns && !seen[d],"molecular source shell destinations must be bijective");
            seen[d]=1; word(hash,Count(d));
            const auto& a=shells[s]; const auto& z=shells[d];
            require(a.contr.size()==1 && z.contr.size()==1 && a.alpha==z.alpha &&
                a.contr[0].l==z.contr[0].l && a.contr[0].pure==z.contr[0].pure &&
                a.contr[0].coeff==z.contr[0].coeff && a.size()==z.size(),
                "molecular source operation changes a radial shell channel");
            radial=add(radial,mul(2,a.alpha.size()));
            double error=0.;
            for(int i=0;i<3;++i) {
                double x=t[i]; for(int j=0;j<3;++j) x+=r[3*i+j]*a.O[j];
                require(std::isfinite(x),"nonfinite molecular source center transform");
                error=std::hypot(error,x-z.O[i]);
            }
            require(std::isfinite(error) && error<=tolerance,"molecular source operation changes finite shell support");
            residual_=std::max(residual_,error);
        }
    }
    require(radial<=plan_.radial_values,"molecular source radial census exceeded");
    // Report actual inspected radial values; admission retains its upper cap.
    plan_.radial_values=radial;
    identity_=hash.finish_hex();
    destinations_.assign(destinations,destinations+plan_.shell_checks);
    validate(*builder_);
}
void MolecularSourceSupport::validate(const OrderedJKBuilder& b) const {
    require(&b.provider()==&builder_->provider() && source(b).source_identity()==source_,
        "molecular source support certificate belongs to a different provider");
}
void MolecularSourceSupport::validate_maps(Count ng,Count ns,const std::int64_t* maps) const {
    require(ng==plan_.operations && ns==builder_->plan().n_shells,
        "molecular source support operation shape mismatch");
    require(maps && std::equal(destinations_.begin(),destinations_.end(),maps),
        "molecular source support certificate belongs to different shell maps");
}
} // namespace vibeqc::manta
