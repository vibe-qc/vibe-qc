#include "vibeqc/symmetry_shared.hpp"
#include <algorithm>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using I = std::int64_t;
using Z = std::complex<double>;
[[noreturn]] void bad(const char* message) { throw std::invalid_argument(message); }
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a) throw std::overflow_error("symmetry count overflow");
    return a+b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max()/a) throw std::overflow_error("symmetry count overflow");
    return a*b;
}
I plus(I a, I b) {
    if ((b>0 && a>std::numeric_limits<I>::max()-b) ||
        (b<0 && a<std::numeric_limits<I>::min()-b))
        throw std::overflow_error("symmetry integer overflow");
    return a+b;
}
I times(I a, I b) {
    if (!a || !b) return 0;
    if ((a == -1 && b == std::numeric_limits<I>::min()) ||
        (b == -1 && a == std::numeric_limits<I>::min()))
        throw std::overflow_error("symmetry integer overflow");
    if (a > 0 ? (b > 0 ? a > std::numeric_limits<I>::max()/b
                        : b < std::numeric_limits<I>::min()/a)
              : (b > 0 ? a < std::numeric_limits<I>::min()/b
                        : a < std::numeric_limits<I>::max()/b))
        throw std::overflow_error("symmetry integer overflow");
    return a*b;
}
template<class T> bool aligned(const T* p) {
    return !p || reinterpret_cast<std::uintptr_t>(p)%alignof(T)==0;
}
void admit(const SymmetryPlan& p, const SymmetryBudget& b) {
    if (!b.maximum_bytes || !b.maximum_work) bad("symmetry requires explicit positive budgets");
    if (p.bytes>b.maximum_bytes) throw std::length_error("symmetry byte budget exceeded");
    if (p.work>b.maximum_work) throw std::length_error("symmetry work budget exceeded");
    if (p.bytes>std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("symmetry address extent overflow");
}
void finite(Z z) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) bad("nonfinite symmetry payload");
}
void index(U i, U n) {
    if (i>=n) throw std::out_of_range("symmetry group index");
}
}
SymmetryPlan plan_symmetry_group(U n, bool lattice, const SymmetryBudget& b) {
    if (!n || n>static_cast<U>(std::numeric_limits<I>::max())) bad("invalid symmetry group order");
    const U square=mul(n,n), cube=mul(square,n);
    SymmetryPlan p;
    p.bytes=add(4096,add(mul(16,square),mul(18,n)));
    if (lattice) p.bytes=add(p.bytes,add(mul(72,n),mul(48,square)));
    // Includes table/range/Latin checks, associativity, rotation and cocycle.
    p.work=add(mul(lattice?256:16,cube),mul(128,square));
    admit(p,b);
    return p;
}
SymmetryGroup make_symmetry_group(U n, U identity, const I* table,
    const std::uint8_t* anti, const I* rotations, const I* cocycles,
    const SymmetryBudget& budget) {
    if (bool(rotations)!=bool(cocycles)) bad("rotations and cocycle must occur together");
    const auto plan=plan_symmetry_group(n,rotations!=nullptr,budget);
    if (!table || !anti || identity>=n) bad("invalid symmetry group descriptors");
    if (!aligned(table) || !aligned(anti) || !aligned(rotations) || !aligned(cocycles))
        bad("unaligned symmetry group descriptors");
    for (U g=0;g<n;++g) {
        if (anti[g]>1) bad("antiunitary grading must be zero or one");
        for (U h=0;h<n;++h) {
            const I k=table[g*n+h];
            if (k<0 || static_cast<U>(k)>=n) bad("symmetry product outside group");
        }
    }
    if (anti[identity]) bad("identity cannot be antiunitary");
    for (U g=0;g<n;++g) {
        if (table[identity*n+g]!=static_cast<I>(g) || table[g*n+identity]!=static_cast<I>(g))
            bad("symmetry identity law failed");
        U inverse_count=0;
        for (U h=0;h<n;++h) {
            if (table[g*n+h]==static_cast<I>(identity) && table[h*n+g]==static_cast<I>(identity))
                ++inverse_count;
            if ((anti[g]^anti[h])!=anti[table[g*n+h]]) bad("antiunitary grading law failed");
            for (U j=0;j<n;++j) {
                if (table[table[g*n+h]*n+j]!=table[g*n+table[h*n+j]])
                    bad("symmetry associativity law failed");
            }
        }
        if (inverse_count!=1) bad("symmetry group needs a unique two-sided inverse");
    }
    if (rotations) {
        for (U a=0;a<3;++a) for (U b=0;b<3;++b)
            if (rotations[9*identity+3*a+b]!=(a==b?1:0)) bad("lattice identity rotation failed");
        for (U g=0;g<n;++g) {
            for (U d=0;d<3;++d)
                if (cocycles[3*(identity*n+g)+d] || cocycles[3*(g*n+identity)+d])
                    bad("lattice identity cocycle failed");
            for (U h=0;h<n;++h) {
                const U gh=table[g*n+h];
                for (U a=0;a<3;++a) for (U b=0;b<3;++b) {
                    I value=0;
                    for (U c=0;c<3;++c)
                        value=plus(value,times(rotations[9*g+3*a+c],rotations[9*h+3*c+b]));
                    if (value!=rotations[9*gh+3*a+b]) bad("lattice rotation product failed");
                }
                // Exact twisted integer cocycle; never reduce shifts modulo a
                // mesh or infer them by rounding phases (Dovesi 1986, Sec.3).
                for (U j=0;j<n;++j) for (U a=0;a<3;++a) {
                    I rhs=cocycles[3*(g*n+table[h*n+j])+a];
                    for (U c=0;c<3;++c)
                        rhs=plus(rhs,times(rotations[9*g+3*a+c],cocycles[3*(h*n+j)+c]));
                    const I lhs=plus(cocycles[3*(g*n+h)+a],cocycles[3*(gh*n+j)+a]);
                    if (lhs!=rhs) bad("integer lattice cocycle law failed");
                }
            }
        }
    }
    SymmetryGroup out;
    out.order_=n; out.identity_=identity; out.memory_=plan;
    out.products_.assign(table,table+n*n);
    out.antiunitary_.assign(anti,anti+n);
    out.inverses_.resize(n);
    for (U g=0;g<n;++g) for (U h=0;h<n;++h)
        if (table[g*n+h]==static_cast<I>(identity) && table[h*n+g]==static_cast<I>(identity))
            out.inverses_[g]=h;
    if (cocycles) out.cocycles_.assign(cocycles,cocycles+3*n*n);
    return out;
}
U SymmetryGroup::product(U g,U h) const { index(g,order_);index(h,order_);return products_[g*order_+h]; }
U SymmetryGroup::inverse(U g) const { index(g,order_);return inverses_[g]; }
bool SymmetryGroup::antiunitary(U g) const { index(g,order_);return antiunitary_[g]!=0; }
std::array<I,3> SymmetryGroup::cocycle(U g,U h) const {
    index(g,order_);index(h,order_);
    if (cocycles_.empty()) return {0,0,0};
    const U i=3*(g*order_+h);
    return {cocycles_[i],cocycles_[i+1],cocycles_[i+2]};
}
SymmetryPlan plan_symmetry_blocks(U n,U blocks,U elements,U columns,const SymmetryBudget& b) {
    if (!n || !blocks || blocks>n) bad("invalid symmetry block dimensions");
    const U payload=mul(n,columns);
    SymmetryPlan p;
    p.bytes=add(4096,add(mul(32,payload),add(mul(16,elements),add(mul(17,blocks),16))));
    p.work=add(mul(32,mul(elements,columns)),add(mul(8,elements),mul(32,blocks)));
    admit(p,b); return p;
}
std::vector<Z> apply_symmetry_blocks(U n,U blocks,U elements,U columns,
    const I* offsets,const I* dest,const Z* matrices,const Z* input,
    bool anti,bool adjoint,const SymmetryBudget& budget) {
    plan_symmetry_blocks(n,blocks,elements,columns,budget);
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
    bad("symmetry transport requires strict floating point compilation");
#endif
    if (!std::numeric_limits<double>::is_iec559 || sizeof(double) != 8 ||
        std::numeric_limits<double>::digits != 53 || FLT_RADIX != 2 ||
        sizeof(Z) != 16 || std::fegetround() != FE_TONEAREST)
        bad("symmetry transport requires IEEE binary64 round-to-nearest arithmetic");
    if (!offsets || !dest || (elements && !matrices) || (columns && !input)) bad("null symmetry block payload");
    if (!aligned(offsets) || !aligned(dest) || !aligned(matrices) || !aligned(input))
        bad("unaligned symmetry block payload");
    if (offsets[0]!=0 || offsets[blocks]<0 || static_cast<U>(offsets[blocks])!=n)
        bad("symmetry offsets must partition the space");
    for (U s=0;s<blocks;++s)
        if (offsets[s]<0 || offsets[s+1]<=offsets[s]) bad("symmetry block offsets must increase");
    U actual=0;
    std::vector<std::uint8_t> seen(blocks,0);
    for (U s=0;s<blocks;++s) {
        if (dest[s]<0 || static_cast<U>(dest[s])>=blocks) bad("symmetry destination outside space");
        const U d=dest[s], size=offsets[s+1]-offsets[s];
        if (seen[d]) bad("symmetry destinations must be a permutation");
        seen[d]=1;
        if (offsets[d+1]-offsets[d]!=static_cast<I>(size)) bad("symmetry block rank mismatch");
        actual=add(actual,mul(size,size));
    }
    if (actual!=elements) bad("symmetry rotation payload extent mismatch");
    for (U i=0;i<elements;++i) finite(matrices[i]);
    for (U i=0;i<n*columns;++i) finite(input[i]);
    std::vector<Z> output(n*columns);
    U begin=0;
    for (U s=0;s<blocks;++s) {
        const U d=dest[s], size=offsets[s+1]-offsets[s];
        const U source=offsets[adjoint?d:s], target=offsets[adjoint?s:d];
        for (U a=0;a<size;++a) for (U col=0;col<columns;++col) {
            Z value=0, correction=0;
            for (U b=0;b<size;++b) {
                Z x=input[(source+b)*columns+col];
                if (anti && !adjoint) x=std::conj(x);
                const Z m=adjoint?std::conj(matrices[begin+b*size+a]):matrices[begin+a*size+b];
                const Z term=m*x-correction, next=value+term;
                correction=(next-value)-term; value=next;
            }
            if (anti && adjoint) value=std::conj(value);
            finite(value); output[(target+a)*columns+col]=value;
        }
        begin+=size*size;
    }
    return output;
}
} // namespace vibeqc
