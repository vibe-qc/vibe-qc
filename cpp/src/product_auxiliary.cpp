#include "vibeqc/product_auxiliary.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/init.hpp"
#include <array>
#include <cstring>
#include <set>
#include <tuple>

namespace vibeqc {
namespace {
using namespace manta::admission;
using Count=std::uint64_t;
struct Span {double lo=std::numeric_limits<double>::infinity(),hi=0.;};
using Spans=std::array<Span,33>;
Count cartesian(int l) {return mul(l+1,l+2)/2;}
void controls(const ProductAuxiliaryOptions& o) {
    if(!std::isfinite(o.beta) || o.beta<=1. || o.refinement>8 ||
       o.maximum_angular_momentum<0 || o.maximum_angular_momentum>32)
        throw std::invalid_argument("Product auxiliary controls require beta > 1, refinement <= 8 and angular limit in [0,32]");
}
void compatible(const Molecule& molecule,const BasisSet& basis) {
    validate_basis(basis);
    for(Count s=0;s<basis.nshells();++s) {
        const int a=basis.shell_atom_index(s);
        if(a<0 || static_cast<Count>(a)>=molecule.atoms().size() || basis.libint()[s].O!=molecule.atoms()[a].xyz)
            throw std::invalid_argument("Product auxiliary basis centers must exactly match the supplied molecule");
    }
}
Spans spans(const BasisSet& basis,Count atom,int maximum_l) {
    Spans out;
    for(Count i=0;i<basis.nshells();++i)if(static_cast<Count>(basis.shell_atom_index(i))==atom) {
        const auto& a=basis.libint()[i];
        for(Count j=i;j<basis.nshells();++j)if(static_cast<Count>(basis.shell_atom_index(j))==atom) {
            const auto& b=basis.libint()[j];const int l=a.contr[0].l+b.contr[0].l;
            if(l>maximum_l)throw std::invalid_argument("Product auxiliary angular limit would omit an orbital product channel");
            const auto [alo,ahi]=std::minmax_element(a.alpha.begin(),a.alpha.end());
            const auto [blo,bhi]=std::minmax_element(b.alpha.begin(),b.alpha.end());
            const double lo=*alo+*blo,hi=*ahi+*bhi;
            if(!std::isfinite(hi) || lo<=0.)throw std::overflow_error("Product auxiliary exponent range is not finite positive");
            out[l].lo=std::min(out[l].lo,lo);out[l].hi=std::max(out[l].hi,hi);
        }
    }
    return out;
}
double exponent(double minimum,double beta,double index) {
    if(index==0.)return minimum;
    const double a=std::exp(std::log(minimum)+index*std::log(beta));
    if(!std::isfinite(a) || a<=0.)throw std::overflow_error("Product auxiliary grid exponent is not finite positive");
    return a;
}
Count intervals(const Span& s,const ProductAuxiliaryOptions& o,const ProductAuxiliaryLimits& limits) {
    if(s.lo==s.hi)return 0;
    const double raw=std::ceil((std::log(s.hi)-std::log(s.lo))/std::log(o.beta));
    if(!std::isfinite(raw) || raw<0. || raw>=static_cast<double>(limits.maximum_shells))
        throw std::length_error("Product auxiliary shell budget exceeded by exponent range");
    Count n=static_cast<Count>(raw);
    // Include the upper product exponent even at a rounding boundary.
    if(exponent(s.lo,o.beta,n)<s.hi)++n;
    return n;
}
struct Digest {
    detail::Sha256 hash;
    void word(Count value) {std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=value>>(56-8*i);hash.update(b.data(),8);}
    void text(const std::string& s) {word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());}
    void real(double value) {if(value==0.)value=0.;Count bits;std::memcpy(&bits,&value,8);word(bits);}
};
std::string basis_id(const BasisSet& basis) {
    Digest d;d.text("product-auxiliary-basis-data.v1");d.word(basis.nbasis());d.word(basis.nshells());
    for(Count i=0;i<basis.nshells();++i) {
        const auto& s=basis.libint()[i];d.word(basis.shell_atom_index(i));d.word(s.contr[0].l);d.word(s.contr[0].pure);
        for(double x:s.O)d.real(x);d.word(s.alpha.size());
        for(double x:s.alpha)d.real(x);for(double x:s.contr[0].coeff)d.real(x);
    }
    return d.hash.finish_hex();
}
bool duplicate(const BasisSet* seed,Count atom,int l,double alpha) {
    if(!seed)return false;
    for(Count i=0;i<seed->nshells();++i) {
        const auto& s=seed->libint()[i];const auto& c=s.contr[0];
        // Pure s/p shells already span the same Cartesian space. Pure d+
        // lacks trace components, so it cannot replace a Cartesian shell.
        if(static_cast<Count>(seed->shell_atom_index(i))==atom && c.l==l && (!c.pure || l<2) &&
           s.alpha.size()==1 && s.alpha[0]==alpha && c.coeff[0]!=0.)return true;
    }
    return false;
}
} // namespace

ProductAuxiliaryPlan plan_product_auxiliary_basis(const Molecule& molecule,const BasisSet& orbital,
    const BasisSet* seed,const ProductAuxiliaryOptions& o,const ProductAuxiliaryLimits& limits) {
    controls(o);const auto ao=census(orbital,16);const auto aux=seed?census(*seed,32):Census{};
    ProductAuxiliaryPlan p;p.original_shells=seed?seed->nshells():0;p.original_functions=seed?seed->nbasis():0;
    const Count atoms=molecule.atoms().size(),n=orbital.nshells(),primitives=add(ao.primitives,aux.primitives);
    // Refuse excessive metadata scanning before the nested range scans.
    // Four scans cover planning, construction and source/receipt hashing.
    const Count scan=mul(4096,add(add(primitives,atoms),mul(atoms,mul(n,add(n,ao.primitives)))));
    cap(scan,limits.work_units,"Product auxiliary scan work budget exceeded");
    compatible(molecule,orbital);if(seed)compatible(molecule,*seed);
    for(Count atom=0;atom<atoms;++atom) {
        bool present=false;const auto channels=spans(orbital,atom,o.maximum_angular_momentum);
        for(int l=0;l<=o.maximum_angular_momentum;++l)if(channels[l].hi>0.) {
            present=true;++p.channels;
            const Count count=add(mul(intervals(channels[l],o,limits),Count{1}<<o.refinement),1);
            p.maximum_generated_shells=add(p.maximum_generated_shells,count);
            p.maximum_generated_functions=add(p.maximum_generated_functions,mul(count,cartesian(l)));
        }
        p.centers+=present;
    }
    p.maximum_shells=add(p.original_shells,p.maximum_generated_shells);
    p.maximum_functions=add(p.original_functions,p.maximum_generated_functions);
    cap(p.maximum_shells,limits.maximum_shells,"Product auxiliary shell budget exceeded");
    cap(p.maximum_functions,limits.maximum_functions,"Product auxiliary function budget exceeded");
    // Snapshots, libint/ShellInfo constructors, receipt copies, dedup keys
    // and Python metadata exports. Caller-owned inputs are not included.
    p.modeled_peak_bytes=add(65536,add(mul(8,add(ao.bytes,aux.bytes)),add(mul(4096,p.maximum_shells),mul(512,add(primitives,p.maximum_functions)))));
    p.work_units=add(scan,mul(4096,add(add(p.maximum_functions,p.channels),mul(p.maximum_shells,add(atoms,p.original_shells)))));
    cap(p.modeled_peak_bytes,limits.memory_bytes,"Product auxiliary memory budget exceeded");
    cap(p.work_units,limits.work_units,"Product auxiliary generation work budget exceeded");return p;
}
bool ProductAuxiliaryBasis::matches_orbital(const BasisSet& orbital) const {
    if(orbital.nshells()!=orbital_shells_ || orbital.nbasis()!=orbital_functions_)return false;
    Count primitives=0;
    for(const auto& s:orbital.libint())primitives=add(primitives,s.alpha.size());
    return primitives==orbital_primitives_ && basis_id(orbital)==receipt_.orbital_identity;
}
bool ProductAuxiliaryBasis::matches_seed(const BasisSet& seed) const {
    if(seed.nshells()!=plan_.original_shells || seed.nbasis()!=plan_.original_functions)return false;
    Count primitives=0;for(const auto& s:seed.libint())primitives=add(primitives,s.alpha.size());
    return primitives==seed_primitives_ && basis_id(seed)==receipt_.seed_identity;
}
bool ProductAuxiliaryBasis::matches_basis(const BasisSet& basis) const {
    if(basis.nshells()!=basis_->nshells() || basis.nbasis()!=basis_->nbasis())return false;
    Count primitives=0,expected=0;
    for(const auto& s:basis.libint())primitives=add(primitives,s.alpha.size());
    for(const auto& s:basis_->libint())expected=add(expected,s.alpha.size());
    return primitives==expected && basis_id(basis)==receipt_.basis_identity;
}
ProductAuxiliaryBasis::ProductAuxiliaryBasis(const Molecule& molecule,const BasisSet& orbital,const BasisSet* seed,
    const ProductAuxiliaryOptions& o,const ProductAuxiliaryLimits& limits)
    :plan_(plan_product_auxiliary_basis(molecule,orbital,seed,o,limits)) {
    orbital_shells_=orbital.nshells();orbital_functions_=orbital.nbasis();
    for(const auto& s:orbital.libint())orbital_primitives_=add(orbital_primitives_,s.alpha.size());
    if(seed)for(const auto& s:seed->libint())seed_primitives_=add(seed_primitives_,s.alpha.size());
    ensure_libint_initialized();receipt_.options=o;receipt_.orbital_identity=basis_id(orbital);
    receipt_.seed_identity=seed?basis_id(*seed):"none";
    std::vector<ShellInfo> shells=seed?seed->shells():std::vector<ShellInfo>{};shells.reserve(plan_.maximum_shells);
    receipt_.spans.reserve(plan_.channels);
    std::set<std::tuple<Count,int,double>> emitted;
    for(Count atom=0;atom<molecule.atoms().size();++atom) {
        const auto channels=spans(orbital,atom,o.maximum_angular_momentum);
        for(int l=0;l<=o.maximum_angular_momentum;++l)if(channels[l].hi>0.) {
            const auto& span=channels[l];const Count coarse=intervals(span,o,limits);
            receipt_.spans.push_back({atom,coarse,l,span.lo,span.hi,exponent(span.lo,o.beta,coarse)});
            auto append=[&](Count numerator,Count denominator) {
                const double alpha=exponent(span.lo,o.beta,static_cast<double>(numerator)/denominator);
                if(!emitted.emplace(atom,l,alpha).second)
                    throw std::invalid_argument("Product auxiliary refinement has indistinguishable floating exponents");
                if(duplicate(seed,atom,l,alpha)){++receipt_.duplicate_seed_shells;return;}
                // Reuse the project's libint normalization convention. Keep
                // every seed coefficient untouched when composing the basis.
                libint2::Shell s({alpha},{{l,false,{1.}}},molecule.atoms()[atom].xyz,true);
                ShellInfo info;info.atom_index=static_cast<int>(atom);info.l=l;info.pure=false;
                if(!std::isfinite(s.contr[0].coeff[0]) || s.contr[0].coeff[0]<=0.)
                    throw std::overflow_error("Product auxiliary normalization is not finite positive");
                info.origin=s.O;info.exponents={alpha};info.coefficients={s.contr[0].coeff[0]};shells.push_back(std::move(info));
                ++receipt_.added_shells;receipt_.added_functions=add(receipt_.added_functions,cartesian(l));
            };
            for(Count i=0;i<=coarse;++i)append(i,1);
            // Dyadic refinement leaves all coarser exponents bitwise intact.
            for(Count level=1;level<=o.refinement;++level) {
                const Count denominator=Count{1}<<level,points=mul(coarse,denominator);
                for(Count i=1;i<points;i+=2)append(i,denominator);
            }
        }
    }
    basis_=std::make_unique<BasisSet>(molecule,shells,"product-auxiliary-v1",true);
    cap(basis_->nshells(),plan_.maximum_shells,"Product auxiliary actual shell count exceeds plan");
    cap(basis_->nbasis(),plan_.maximum_functions,"Product auxiliary actual function count exceeds plan");
    receipt_.basis_identity=basis_id(*basis_);Digest d;d.text(receipt_.algorithm);
    d.text(receipt_.orbital_identity);d.text(receipt_.seed_identity);d.text(receipt_.basis_identity);
    d.real(o.beta);d.word(o.refinement);d.word(o.maximum_angular_momentum);receipt_.identity=d.hash.finish_hex();
}
ProductAuxiliarySequencePlan plan_product_auxiliary_sequence(const Molecule& mol,const BasisSet& orbital,
    const BasisSet& seed,const ProductAuxiliaryOptions& initial,Count refinements,const ProductAuxiliaryLimits& limits) {
    controls(initial);
    if(refinements>8 || initial.refinement>8-refinements)
        throw std::invalid_argument("Product auxiliary sequence exceeds refinement level eight");
    ProductAuxiliarySequencePlan out;out.candidates=refinements+1;Count retained=0,peak=0;
    for(Count i=0;i<=refinements;++i) {
        auto options=initial;options.refinement+=i;
        const auto p=plan_product_auxiliary_basis(mol,orbital,&seed,options,limits);
        retained=add(retained,p.modeled_peak_bytes);peak=std::max(peak,p.modeled_peak_bytes);
        out.work_units=add(out.work_units,p.work_units);
        out.maximum_shells=std::max(out.maximum_shells,p.maximum_shells);
        out.maximum_functions=std::max(out.maximum_functions,p.maximum_functions);
    }
    out.modeled_peak_bytes=add(65536,add(mul(256,mol.atoms().size()),add(retained,peak)));
    out.work_units=add(out.work_units,mul(4096,mol.atoms().size()));
    cap(out.modeled_peak_bytes,limits.memory_bytes,"Product auxiliary sequence memory budget exceeded");
    cap(out.work_units,limits.work_units,"Product auxiliary sequence work budget exceeded");return out;
}
ProductAuxiliarySequence::ProductAuxiliarySequence(const Molecule& mol,const BasisSet& orbital,const BasisSet& seed,
    const ProductAuxiliaryOptions& initial,Count refinements,const ProductAuxiliaryLimits& limits)
    :plan_(plan_product_auxiliary_sequence(mol,orbital,seed,initial,refinements,limits)),initial_(initial) {
    candidates_.reserve(plan_.candidates);
    for(Count i=0;i<plan_.candidates;++i) {
        auto options=initial;options.refinement+=i;
        candidates_.push_back(std::make_unique<ProductAuxiliaryBasis>(mol,orbital,&seed,options,limits));
    }
}
} // namespace vibeqc
