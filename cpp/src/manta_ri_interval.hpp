#pragma once
#include "vibeqc/manta_ri_metric.hpp"
#include <cfenv>
#include <cstring>
#include <complex>

#if defined(__FAST_MATH__)
#error "MANTA interval bounds require IEEE floating-point arithmetic"
#endif

namespace vibeqc::manta::spectral_interval {
constexpr double infinity=std::numeric_limits<double>::infinity();
// Each elementary operation rounds separately to binary64 before expansion.
// No process-global rounding-mode change or dependence on BLAS arithmetic.
inline double outward(double x,double direction) {
    const double result=std::nextafter(x,direction);
    if(!std::isfinite(x) || !std::isfinite(result)) throw std::overflow_error("MANTA spectral enclosure overflow");
    return result;
}
inline double down(double x) { return outward(x,-infinity); }
inline double up(double x) { return outward(x,infinity); }
inline double plus_up(double a,double b) { if(a==0.) return b;if(b==0.) return a;volatile double x=a+b;return up(x); }
struct Interval { double lo=0.,hi=0.; };
inline Interval add(Interval a,Interval b) {
    if(b.lo==0. && b.hi==0.) return a;if(a.lo==0. && a.hi==0.) return b;
    volatile double lo=a.lo+b.lo,hi=a.hi+b.hi;return {down(lo),up(hi)};
}
inline Interval sub(Interval a,Interval b) { return add(a,{-b.hi,-b.lo}); }
inline Interval product(double a,double b) {
    if(a==0. || b==0.) return {};volatile double x=a*b;return {down(x),up(x)};
}
struct ComplexInterval { Interval real,imag; };
inline ComplexInterval product(std::complex<double> a,std::complex<double> b) {
    return {sub(product(a.real(),b.real()),product(a.imag(),b.imag())),
            add(product(a.real(),b.imag()),product(a.imag(),b.real()))};
}
inline ComplexInterval add(ComplexInterval a,ComplexInterval b) { return {add(a.real,b.real),add(a.imag,b.imag)}; }
inline ComplexInterval sub(ComplexInterval a,ComplexInterval b) { return {sub(a.real,b.real),sub(a.imag,b.imag)}; }
inline double magnitude(ComplexInterval z) {
    return plus_up(std::max(std::abs(z.real.lo),std::abs(z.real.hi)),std::max(std::abs(z.imag.lo),std::abs(z.imag.hi)));
}
inline void require_arithmetic() {
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1.;
    volatile double normal=std::numeric_limits<double>::min(),half=.5;
    const double smallest=tiny*one,subnormal=normal*half;
    std::uint64_t smallest_bits=0,subnormal_bits=0;
    std::memcpy(&smallest_bits,&smallest,sizeof(double));std::memcpy(&subnormal_bits,&subnormal,sizeof(double));
    if(!std::numeric_limits<double>::is_iec559 || std::fegetround()!=FE_TONEAREST ||
       smallest_bits!=1 || subnormal_bits!=(std::uint64_t{1}<<51))
        throw std::runtime_error("MANTA spectral enclosure requires round-to-nearest binary64 with gradual underflow");
}

inline Interval product(Interval a,Interval b) {
    const auto x=product(a.lo,b.lo),y=product(a.lo,b.hi),z=product(a.hi,b.lo),w=product(a.hi,b.hi);
    return {std::min({x.lo,y.lo,z.lo,w.lo}),std::max({x.hi,y.hi,z.hi,w.hi})};
}
inline ComplexInterval point(std::complex<double> z) { return {{z.real(),z.real()},{z.imag(),z.imag()}}; }
inline ComplexInterval product(ComplexInterval a,ComplexInterval b) {
    return {sub(product(a.real,b.real),product(a.imag,b.imag)),add(product(a.real,b.imag),product(a.imag,b.real))};
}
inline double times_up(double a,double b) { return product(a,b).hi; }
inline double quotient_up(double a,double b) { if(a==0.)return 0.;volatile double x=a/b;return up(x); }
// Exact power-of-two enclosure, no libm square-root accuracy assumption.
inline double sqrt_upper(double x) {
    if(x==0.)return 0.;if(x<0. || !std::isfinite(x))throw std::overflow_error("MANTA invalid squared norm");
    int e=0;std::frexp(x,&e);return std::ldexp(1.,e/2+(e>0 && e%2));
}
} // namespace vibeqc::manta::spectral_interval
