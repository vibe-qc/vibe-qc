#pragma once

#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace periodic_correlation_real_pao_detail {

namespace real = periodic_correlation_real_local_detail;
using Complex = std::complex<double>;

// Outward elementary intervals on the represented binary64 inputs. Exact
// zero operations and +/-1 products are preserved; no other cancellation or
// subnormal product is presumed exact. Every nonzero endpoint operation gets
// one nextafter step, under the shared explicit floating-environment policy.
struct Interval {
    double lower = 0.0, upper = 0.0;
    static Interval point(double value) { real::finite(value); return {value,value}; }
};
inline Interval plus(Interval a, Interval b) {
    if (a.lower == 0 && a.upper == 0) return b;
    if (b.lower == 0 && b.upper == 0) return a;
    return {real::down(a.lower+b.lower),real::up(a.upper+b.upper)};
}
inline Interval negate(Interval a) { return {-a.upper,-a.lower}; }
inline Interval minus(Interval a, Interval b) { return plus(a,negate(b)); }
inline Interval product(double a, double b) {
    real::finite(a); real::finite(b);
    if (a == 0 || b == 0) return {};
    if (a == 1) return Interval::point(b);
    if (b == 1) return Interval::point(a);
    if (a == -1) return Interval::point(-b);
    if (b == -1) return Interval::point(-a);
    const auto value = real::finite(a*b);
    return {real::down(value),real::up(value)};
}
inline Interval scale(Interval a, double b) {
    real::finite(b);
    if (b == 0 || (a.lower == 0 && a.upper == 0)) return {};
    const auto lo = product(a.lower,b), hi = product(a.upper,b);
    return {std::min(lo.lower,hi.lower),std::max(lo.upper,hi.upper)};
}
inline double magnitude(Interval a) { return std::max(std::abs(a.lower),std::abs(a.upper)); }
inline double error(Interval a, double target = 0) { return magnitude(minus(a,Interval::point(target))); }

struct ComplexInterval {
    Interval re, im;
    static ComplexInterval point(Complex value) { return {Interval::point(value.real()),Interval::point(value.imag())}; }
};
static_assert(sizeof(ComplexInterval) == 32, "real PAO interval column is four binary64 endpoints");
inline ComplexInterval plus(ComplexInterval a, ComplexInterval b) { return {plus(a.re,b.re),plus(a.im,b.im)}; }
inline ComplexInterval scale(ComplexInterval a, double b) { return {scale(a.re,b),scale(a.im,b)}; }
inline void norm(double& squared, Interval a, double target = 0) { real::norm_lane(squared,error(a,target)); }
inline void norm(double& squared, ComplexInterval a, Complex target = {}) {
    norm(squared,a.re,target.real()); norm(squared,a.im,target.imag());
}
inline double relative_up(double numerator, double denominator_lower) {
    if (numerator == 0) return 0;
    return real::up(numerator/(denominator_lower == 0 ? 1 : denominator_lower));
}
inline double polar_distance(double gram_upper) {
    if (!(gram_upper >= 0 && gram_upper < 1))
        throw std::runtime_error("real PAO eigensystem Gram bound must be below one");
    if (gram_upper == 0) return 0;
    const auto radicand_lower = std::max(0.0,real::down(1.0-gram_upper));
    const auto root_lower = radicand_lower == 0 ? 0 : std::max(0.0,real::down(std::sqrt(radicand_lower)));
    const auto denominator_lower = real::down(1.0+root_lower);
    return real::up(gram_upper/denominator_lower);  // Stable 1-sqrt(1-g).
}

}  // namespace periodic_correlation_real_pao_detail
}  // namespace vibeqc
