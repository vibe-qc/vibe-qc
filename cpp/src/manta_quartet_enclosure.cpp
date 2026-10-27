#include "vibeqc/manta_molecular_source.hpp"
#include "vibeqc/manta_provider.hpp"
#include "vibeqc/manta_admission.hpp"
#include "manta_metric_enclosure.hpp"
#include "manta_verified_math.hpp"
#include <typeinfo>

namespace vibeqc::manta {
namespace {
namespace si=spectral_interval;
namespace vm=verified_math;
using I=si::Interval;
I point(double x) { return {x,x}; }
I positive(I x) {
    x.lo=std::max(0.,x.lo);
    if(x.lo>x.hi) throw std::logic_error("MANTA quartet enclosure lost positivity");
    return x;
}
I square(I x) { return positive(si::product(x,x)); }
const IntegralProvider& source(const OrderedJKBuilder& b) {
    const auto& p=b.provider(); const auto r=p.kernel_request();
    if(typeid(p)!=typeid(MantaProvider) && typeid(p)!=typeid(LibintProvider))
        throw std::invalid_argument("quartet construction requires an immutable native provider");
    if(r.kernel!="coulomb" || r.omega!=0. || r.scalar!="real" || r.centers!=4 ||
       r.derivative!=0 || r.images!=decltype(r.images){})
        throw std::invalid_argument("quartet construction requires molecular real four-center Coulomb");
    if(p.plan().provider!=p.name() || p.plan().kernel!=r.kernel || p.plan().omega!=r.omega)
        throw std::invalid_argument("quartet construction provider plan mismatch");
    return p;
}
struct PairReference { I exponent,decay; std::array<I,3> center; };
PairReference pair_reference(const libint2::Shell& a,std::size_t ia,
                            const libint2::Shell& b,std::size_t ib) {
    const I alpha=point(a.alpha[ia]),beta=point(b.alpha[ib]);
    PairReference out; out.exponent=si::add(alpha,beta);
    const auto fraction=vm::divide(beta,out.exponent);
    I squared;
    for(int axis=0;axis<3;++axis) {
        const auto delta=si::sub(point(b.O[axis]),point(a.O[axis]));
        squared=si::add(squared,square(delta));
        out.center[axis]=si::add(point(a.O[axis]),si::product(fraction,delta));
    }
    out.decay=vm::exp_negative(positive(si::product(si::product(alpha,fraction),squared)));
    return out;
}
I quartet_reference(const std::array<const libint2::Shell*,4>& s) {
    // Gaussian product theorem / McMurchie-Davidson1978,
    // DOI 10.1016/0021-9991(78)90092-X. For s shells:
    // 2*pi^(5/2) K_AB K_CD F_0(rho |P-Q|^2)/(p q sqrt(p+q)).
    // All finite coefficients/centers are exact reference inputs. Rump2010
    // outward arithmetic and MANTA's shared Boys proof enclose that reference;
    // the actual producer remains the existing IntegralProvider::evaluate.
    const auto pi=vm::pi();
    const auto prefactor=si::product(point(2.),si::product(square(pi),vm::sqrt_positive(pi)));
    I result;
    for(std::size_t ia=0;ia<s[0]->alpha.size();++ia)
    for(std::size_t ib=0;ib<s[1]->alpha.size();++ib) {
        const auto a=pair_reference(*s[0],ia,*s[1],ib);
        for(std::size_t ic=0;ic<s[2]->alpha.size();++ic)
        for(std::size_t id=0;id<s[3]->alpha.size();++id) {
            const auto b=pair_reference(*s[2],ic,*s[3],id);
            const auto pq=si::product(a.exponent,b.exponent),sum=si::add(a.exponent,b.exponent);
            const auto rho=vm::divide(pq,sum);
            I squared;
            for(int axis=0;axis<3;++axis) squared=si::add(squared,square(si::sub(a.center[axis],b.center[axis])));
            const auto boys=coulomb_boys_enclosure(0,positive(si::product(rho,squared)))[0];
            auto value=vm::divide(prefactor,si::product(pq,vm::sqrt_positive(sum)));
            value=si::product(value,si::product(si::product(a.decay,b.decay),boys));
            const auto weight=si::product(si::product(s[0]->contr[0].coeff[ia],s[1]->contr[0].coeff[ib]),
                                          si::product(s[2]->contr[0].coeff[ic],s[3]->contr[0].coeff[id]));
            result=si::add(result,si::product(weight,value));
        }
    }
    return result;
}
}
MolecularQuartetConstructionPlan plan_molecular_quartet_construction(
    const OrderedJKBuilder& b,const std::array<std::size_t,4>& key,const ResourceBudget& budget) {
    using namespace admission; margin(budget);
    const auto& p=source(b); const auto& shells=p.basis().libint();
    MolecularQuartetConstructionPlan out;out.primitive_products=1;
    for(auto k:key) {
        if(k>=shells.size()) throw std::invalid_argument("quartet construction shell index out of range");
        const auto& s=shells[k];
        if(s.contr.size()!=1 || s.contr[0].l!=0 || s.size()!=1 || s.alpha.empty() || s.alpha.size()!=s.contr[0].coeff.size())
            throw std::invalid_argument("quartet construction currently requires four contracted s shells");
        out.primitive_products=mul(out.primitive_products,s.alpha.size());
    }
    // Includes bounded radial proof (2048 terms), elementary proofs and all
    // signed contractions. Reserve an additional native workspace separately
    // from the live builder; proof storage is independent of primitive count.
    out.scalar_terms=mul(1048576,out.primitive_products);
    out.kernel_terms=add(mul(8,out.scalar_terms),p.plan().kernel_terms);
    out.additional_bytes=add(p.plan().modeled_peak_bytes,1048576);
    out.modeled_peak_bytes=add(budget.other_live_bytes,add(budget.backend_margin_bytes,
        add(p.plan().modeled_peak_bytes,out.additional_bytes)));
    cap(out.modeled_peak_bytes,budget.memory_bytes,"quartet construction memory budget exceeded");
    cap(out.scalar_terms,budget.scalar_terms,"quartet construction scalar budget exceeded");
    cap(out.kernel_terms,budget.kernel_terms,"quartet construction kernel budget exceeded");
    cap(out.shell_calls,budget.shell_calls,"quartet construction shell budget exceeded");
    cap(out.primitive_products,budget.primitive_quartets,"quartet construction primitive budget exceeded");
    return out;
}
MolecularQuartetConstruction::MolecularQuartetConstruction(std::shared_ptr<OrderedJKBuilder> b,
    const std::array<std::size_t,4>& key,const ResourceBudget& budget):builder_(std::move(b)),key_(key) {
    if(!builder_) throw std::invalid_argument("quartet construction requires a provider");
    plan_=plan_molecular_quartet_construction(*builder_,key,budget);
    si::require_arithmetic();const auto& p=source(*builder_);source_=p.source_identity();
    const auto& shells=p.basis().libint();
    const auto exact=quartet_reference({&shells[key[0]],&shells[key[1]],&shells[key[2]],&shells[key[3]]});
    Workspace workspace(p.plan());const auto actual=p.evaluate(key,workspace);
    if(actual.shape!=std::array<std::size_t,4>{1,1,1,1})
        throw std::runtime_error("quartet construction producer tile shape mismatch");
    null_=!actual.values;evaluated_=null_?0.:actual.values[0];
    admission::require_finite(evaluated_);
    lower_=exact.lo;upper_=exact.hi;
    const auto residual=si::sub(point(evaluated_),exact);
    error_=std::max(std::abs(residual.lo),std::abs(residual.hi));
}
void MolecularQuartetConstruction::validate(const OrderedJKBuilder& b) const {
    if(&b.provider()!=&builder_->provider() || source(b).source_identity()!=source_)
        throw std::invalid_argument("quartet construction certificate belongs to a different provider");
}
} // namespace vibeqc::manta
