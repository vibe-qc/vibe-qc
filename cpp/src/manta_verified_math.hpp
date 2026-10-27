#pragma once
#include "manta_ri_interval.hpp"

// Shared finite transcendental proofs, Rump2010 verified arithmetic.
// Derivations: scientific contract 5.60-5.62. No libm accuracy premise.
namespace vibeqc::manta::verified_math {
namespace wi=spectral_interval;
inline wi::Interval divide(wi::Interval a,wi::Interval b) {
    if(!(b.lo>0.))throw std::runtime_error("MANTA weight denominator enclosure is not positive");
    if(a.lo==0. && a.hi==0.)return {};
    volatile double v0=a.lo/b.lo,v1=a.lo/b.hi,v2=a.hi/b.lo,v3=a.hi/b.hi;
    return {wi::down(std::min({double(v0),double(v1),double(v2),double(v3)})),
            wi::up(std::max({double(v0),double(v1),double(v2),double(v3)}))};
}

inline wi::Interval pi() {
    const auto arctangent=[](double denominator) {
        const auto x=divide({1.,1.},{denominator,denominator});
        const auto squared=wi::product(x,x);auto power=x;wi::Interval sum;
        for(int j=0;j<40;++j) {
            const double divisor=2*j+1;const auto term=divide(power,{divisor,divisor});
            sum=j%2?wi::sub(sum,term):wi::add(sum,term);power=wi::product(power,squared);
        }
        const auto remainder=divide(power,{81.,81.});
        return wi::add(sum,{-remainder.hi,remainder.hi});
    };
    return wi::sub(wi::product({16.,16.},arctangent(5.)),wi::product({4.,4.},arctangent(239.)));
}

inline wi::Interval exp_negative(wi::Interval x) {
    if(x.lo<0. || !std::isfinite(x.hi))throw std::runtime_error("MANTA weight decay interval is invalid");
    if(x.hi==0.)return {1.,1.};
    int exponent=0;std::frexp(x.hi,&exponent);const int scaling=std::max(0,exponent+4);
    const double factor=std::ldexp(1.,-scaling);
    x=wi::product(x,{factor,factor});x.lo=std::max(0.,x.lo);
    const wi::Interval negative{-x.hi,-x.lo};wi::Interval term{1.,1.},value=term;
    for(int j=1;j<=24;++j) {
        term=divide(wi::product(term,negative),{double(j),double(j)});
        value=wi::add(value,term);
    }
    const auto next=divide(wi::product(term,x),{25.,25.});
    const double remainder=std::max(std::abs(next.lo),std::abs(next.hi));
    value=wi::add(value,{-remainder,remainder});
    auto unit=[](wi::Interval a) {
        a.lo=std::max(0.,a.lo);a.hi=std::min(1.,a.hi);
        if(a.lo>a.hi)throw std::logic_error("MANTA weight enclosure lost positivity");return a;
    };
    value=unit(value);for(int j=0;j<scaling;++j)value=unit(wi::product(value,value));
    return value;
}

inline wi::Interval sqrt_positive(wi::Interval x) {
    if(x.lo<0. || !std::isfinite(x.hi))throw std::runtime_error("MANTA invalid square-root interval");
    const auto endpoint=[](double value,bool upper) {
        if(value==0.)return 0.;
        int exponent=0;double mantissa=std::frexp(value,&exponent);
        if(exponent%2) {mantissa*=2.;--exponent;}
        double lo=0.,hi=2.;
        // Search the two sufficient squared inequalities separately. A wide
        // rounded-product enclosure cannot accidentally certify equality.
        for(int j=0;j<64;++j) {
            const double mid=lo+(hi-lo)*.5;if(mid==lo || mid==hi)break;
            const auto square=wi::product(mid,mid);
            if(upper ? square.lo>=mantissa : square.hi>mantissa)hi=mid;else lo=mid;
        }
        const double scale=std::ldexp(1.,exponent/2);
        const auto result=wi::product(upper?hi:lo,scale);return upper?result.hi:std::max(0.,result.lo);
    };
    return {endpoint(x.lo,false),endpoint(x.hi,true)};
}
} // namespace vibeqc::manta::verified_math
