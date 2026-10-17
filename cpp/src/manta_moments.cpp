#include "vibeqc/manta_moments.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>
namespace vibeqc::manta {
PairMomentPlan plan_pair_moments(const BasisSet& basis,const std::array<std::size_t,2>& shells,
                               int order,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    if(order<0 || order>12) throw std::invalid_argument("MANTA pair moment order must be zero through twelve");
    for(auto s:shells) if(s>=basis.nshells()) throw std::out_of_range("MANTA pair moment shell index");
    cap(1,budget.shell_calls,"MANTA pair moment shell-call budget exceeded");
    const auto inventory=census(basis,6);
    const auto& a=basis.libint()[shells[0]]; const auto& b=basis.libint()[shells[1]];
    PairMomentPlan plan; plan.order=order;
    plan.moments=Count(order+1)*(order+2)*(order+3)/6;
    plan.shape={mul(a.size(),b.size()),plan.moments};
    plan.scalar_terms=mul(plan.shape[0],plan.moments);
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA pair moment output budget exceeded");
    plan.primitive_pairs=mul(a.alpha.size(),b.alpha.size());
    cap(plan.primitive_pairs,budget.primitive_quartets,"MANTA pair moment primitive-pair budget exceeded");
    const int la=a.contr[0].l,lb=b.contr[0].l;
    const Count ca=(la+1)*(la+2)/2,cb=(lb+1)*(lb+2)/2,cart=mul(ca,cb);
    const Count axes=Count(3)*(la+order+1)*(lb+1)*(la+lb+order+1);
    const Count panel=mul(cart,plan.moments);
    plan.kernel_terms=add(mul(plan.primitive_pairs,add(mul(16,axes),mul(64,mul(panel,order+1)))),
        mul(8,mul(square(cart),plan.moments)));
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA pair moment work budget exceeded");
    // Cartesian sum/correction/primitive and transformed/exported panels,
    // pair transformation, one shared primitive Expansion and small controls.
    const Count workspace=mul(8,add(add(mul(8,panel),mul(4,square(cart))),add(mul(2,axes),mul(64,plan.moments))));
    plan.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(workspace,add(mul(2,inventory.bytes),8192)));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA pair moment memory budget exceeded");
    validate_basis(basis); return plan;
}
PairMomentResult compute_pair_moments(const BasisSet& basis,const std::array<std::size_t,2>& shells,
    const md::Point& origin,int order,const ResourceBudget& budget) {
    const auto plan=plan_pair_moments(basis,shells,order,budget);
    for(auto x:origin) admission::require_finite(x);
    PairMomentResult result; result.indices=md::hermites(order);
    const auto& a=basis.libint()[shells[0]]; const auto& b=basis.libint()[shells[1]];
    result.ao_shape={a.size(),b.size()};
    result.receipt.primitive_pairs=plan.primitive_pairs; result.receipt.primitive_expansions=plan.primitive_pairs;
    result.receipt.primitive_moment_products=admission::mul(plan.primitive_pairs,plan.moments);
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    vibeqc::detail::Sha256 hash;
    const auto prefix="manta.pair-moments.v1;"+auxiliary_basis_content_identity_sha256(basis)+";"+result.receipt.normalization;
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    auto word=[&](std::uint64_t value) {
        std::array<std::uint8_t,8> bytes{}; for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(value>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    };
    for(auto s:shells) word(s); word(order);
    for(double x:origin) { if(x==0.) x=0.; std::uint64_t bits; std::memcpy(&bits,&x,sizeof(bits)); word(bits); }
    result.receipt.source_identity=hash.finish_hex();
    result.values=md::shell_moments(a,b,origin,order);
    return result;
}
AuxiliaryMomentPlan plan_auxiliary_moments(const BasisSet& basis,std::size_t shell,int order,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    if(order<0 || order>12) throw std::invalid_argument("MANTA auxiliary moment order must be zero through twelve");
    if(shell>=basis.nshells()) throw std::out_of_range("MANTA auxiliary moment shell index");
    cap(1,budget.shell_calls,"MANTA auxiliary moment shell-call budget exceeded");
    const auto inventory=census(basis,7);
    const auto& a=basis.libint()[shell];
    AuxiliaryMomentPlan plan; plan.order=order;
    plan.moments=Count(order+1)*(order+2)*(order+3)/6;
    plan.shape={a.size(),plan.moments};
    plan.scalar_terms=mul(plan.shape[0],plan.moments);
    cap(plan.scalar_terms,budget.scalar_terms,"MANTA auxiliary moment output budget exceeded");
    plan.primitives=a.alpha.size();
    cap(plan.primitives,budget.primitive_quartets,"MANTA auxiliary moment primitive budget exceeded");
    const int l=a.contr[0].l;
    const Count cart=(l+1)*(l+2)/2,axes=Count(3)*(l+order+1)*(l+order+1),panel=mul(cart,plan.moments);
    plan.kernel_terms=add(mul(plan.primitives,add(mul(16,axes),mul(64,mul(panel,order+1)))),
                          mul(8,mul(square(cart),plan.moments)));
    cap(plan.kernel_terms,budget.kernel_terms,"MANTA auxiliary moment work budget exceeded");
    const Count workspace=mul(8,add(add(mul(8,panel),mul(4,square(cart))),add(mul(2,axes),mul(64,plan.moments))));
    plan.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
                               add(workspace,add(mul(2,inventory.bytes),8192)));
    cap(plan.modeled_peak_bytes,budget.memory_bytes,"MANTA auxiliary moment memory budget exceeded");
    validate_basis(basis); return plan;
}
AuxiliaryMomentResult compute_auxiliary_moments(const BasisSet& basis,std::size_t shell,
    const md::Point& origin,int order,const ResourceBudget& budget) {
    const auto plan=plan_auxiliary_moments(basis,shell,order,budget);
    for(auto x:origin) admission::require_finite(x);
    AuxiliaryMomentResult result; result.indices=md::hermites(order);
    result.receipt.primitives=result.receipt.primitive_expansions=plan.primitives;
    result.receipt.primitive_moment_products=admission::mul(plan.primitives,plan.moments);
    result.receipt.modeled_peak_bytes=plan.modeled_peak_bytes;
    vibeqc::detail::Sha256 hash;
    const auto prefix="manta.auxiliary-moments.v1;"+auxiliary_basis_content_identity_sha256(basis)+";"+result.receipt.normalization;
    hash.update(reinterpret_cast<const std::uint8_t*>(prefix.data()),prefix.size());
    auto word=[&](std::uint64_t value) {
        std::array<std::uint8_t,8> bytes{}; for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(value>>(56-8*k));
        hash.update(bytes.data(),bytes.size());
    };
    word(shell); word(order);
    for(double x:origin) { if(x==0.) x=0.; std::uint64_t bits; std::memcpy(&bits,&x,sizeof(bits)); word(bits); }
    result.receipt.source_identity=hash.finish_hex();
    result.values=md::shell_auxiliary_moments(basis.libint()[shell],origin,order);
    return result;
}
} // namespace vibeqc::manta
