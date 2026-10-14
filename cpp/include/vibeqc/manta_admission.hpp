#pragma once

// Count-only admission utilities shared by bounded MD adapters. Payload
// validation is a separate pass after output/workspace/work caps are checked.
#include "vibeqc/manta.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::manta::admission {
inline Count add(Count a,Count b) {
    if(b>std::numeric_limits<Count>::max()-a) throw std::overflow_error("MANTA resource sum overflow");
    return a+b;
}
inline Count mul(Count a,Count b) {
    if(a && b>std::numeric_limits<Count>::max()/a) throw std::overflow_error("MANTA resource product overflow");
    return a*b;
}
inline Count square(Count a) { return mul(a,a); }
inline Count fourth(Count a) { return square(square(a)); }
inline void cap(Count value,Count limit,const char* message) {
    if(value>limit) throw std::length_error(message);
}
inline void require_finite(double value) {
    if(!std::isfinite(value)) throw std::invalid_argument("MANTA requires finite numerical input");
}
struct Census {
    Count primitives=0,maximum_size=0,maximum_cartesian=0,bytes=0;
    int maximum_l=0;
};
inline Census census(const BasisSet& basis,int limit) {
    if(!basis.nbasis() || !basis.nshells()) throw std::invalid_argument("MANTA requires a nonempty basis");
    Census result;
    result.bytes=add(basis.name().size(),mul(basis.nshells(),sizeof(libint2::Shell)+sizeof(libint2::Shell::Contraction)+3*sizeof(std::size_t)));
    Count ao_count=0;
    for(const auto& s:basis.libint()) {
        if(s.alpha.empty() || s.contr.size()!=1 || s.contr[0].coeff.size()!=s.alpha.size())
            throw std::invalid_argument("MANTA requires nonempty single-contraction shell channels");
        const int l=s.contr[0].l;
        if(l<0 || l>limit) throw std::invalid_argument("MANTA angular capability exceeded");
        result.maximum_l=std::max(result.maximum_l,l);
        result.maximum_cartesian=std::max<Count>(result.maximum_cartesian,(l+1)*(l+2)/2);
        result.maximum_size=std::max<Count>(result.maximum_size,s.size());
        result.primitives=add(result.primitives,s.alpha.size());
        result.bytes=add(result.bytes,add(24,mul(24,s.alpha.size())));
        ao_count=add(ao_count,s.size());
    }
    if(ao_count!=basis.nbasis()) throw std::invalid_argument("MANTA basis AO census mismatch");
    return result;
}
inline void validate_basis(const BasisSet& basis) {
    for(const auto& s:basis.libint()) {
        for(auto x:s.O) require_finite(x);
        for(auto x:s.alpha) { require_finite(x); if(x<=0) throw std::invalid_argument("MANTA requires positive exponents"); }
        for(auto x:s.contr[0].coeff) require_finite(x);
    }
}
inline void margin(const ResourceBudget& budget) {
    if(budget.backend_margin_bytes<(Count{64}<<10)) throw std::invalid_argument("MANTA backend margin must be at least 64 KiB");
}
} // namespace vibeqc::manta::admission
