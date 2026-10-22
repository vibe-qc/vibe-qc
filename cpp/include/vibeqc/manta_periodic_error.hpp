#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::manta::periodic_error {
// Guard positive scalar envelope arithmetic, not an integral-roundoff proof.
// long double accumulation plus an outward binary64 guard; preserve exact zero.
inline void positive_add(long double value,long double& sum) {
    if(value!=0.) sum=std::nextafter(sum+value,std::numeric_limits<long double>::infinity());
}
inline double upper(long double value) {
    if(value==0) return 0.;
    value*=1+1024*std::numeric_limits<double>::epsilon();
    if(value<0 || !std::isfinite(value) || value>std::numeric_limits<double>::max())
        throw std::overflow_error("MANTA periodic coupling error bound overflow");
    const double out=std::nextafter(static_cast<double>(value),std::numeric_limits<double>::infinity());
    if(!std::isfinite(out)) throw std::overflow_error("MANTA periodic coupling error bound overflow");
    return out;
}
} // namespace vibeqc::manta::periodic_error
