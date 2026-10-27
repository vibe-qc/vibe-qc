#pragma once
#include "manta_verified_math.hpp"

namespace vibeqc::manta {
// Positive-panel proof of Neese2023 Eq.54, Rump2010 outward arithmetic.
// Scientific contract 5.64. No change to the evaluated MD radial kernel.
inline std::vector<spectral_interval::Interval> positive_erfc_enclosure(
    int order,spectral_interval::Interval t,spectral_interval::Interval rho,double omega) {
    namespace si=spectral_interval;namespace vm=verified_math;using I=si::Interval;
    const auto point=[](double x)->I{return {x,x};};
    const auto nonnegative=[](I x)->I {x.lo=std::max(0.,x.lo);if(x.hi<x.lo)throw std::logic_error("MANTA positive radial enclosure is empty");return x;};
    const auto w2=si::product(omega,omega),denominator=si::add(w2,rho);
    auto kappa=nonnegative(vm::divide(w2,denominator));kappa.hi=std::min(1.,kappa.hi);
    auto complement=nonnegative(vm::divide(rho,denominator));complement.hi=std::min(1.,complement.hi);
    // Choose an exact stored fraction of the squared integration interval.
    // Choosing the partition needs no rounded-accuracy premise: every panel
    // and any omitted tail are enclosed independently below.
    double fraction=1.;
    if(t.hi>64. && complement.hi>64./t.hi)fraction=std::min(1.,(64./t.hi)/complement.hi);
    if(!(fraction>0.))throw std::runtime_error("MANTA radial partition underflow");
    const auto extent=si::product(complement,point(fraction));
    const auto gain=si::product(t,extent);
    const double count=std::max(1.,std::ceil(si::up(si::up(si::up(gain.hi/8.)))));
    if(count>9.)throw std::runtime_error("MANTA positive radial panel budget exceeded");
    const int panels=static_cast<int>(count);const auto increment=vm::divide(extent,point(panels));
    const auto panel_gain=si::product(t,increment);
    if(panel_gain.hi>8.)throw std::runtime_error("MANTA positive radial panel gain unresolved");
    // Multiplication in the Bernstein basis preserves nonnegative coefficients.
    // Integral_0^1 sum_i c_i B_i^d = sum_i c_i/(d+1).
    const auto linear=[&](const std::vector<I>& coefficients,I left,I right) {
        const Count degree=coefficients.size();std::vector<I> result(degree+1);
        for(Count i=0;i<=degree;++i) {
            if(i<degree)result[i]=si::product(si::product(left,coefficients[i]),vm::divide(point(degree-i),point(degree)));
            if(i)result[i]=si::add(result[i],si::product(si::product(right,coefficients[i-1]),vm::divide(point(i),point(degree))));
        }
        return result;
    };
    const auto integrate=[&](const std::vector<I>& coefficients) {
        I sum;for(const auto& x:coefficients)sum=si::add(sum,x);return vm::divide(sum,point(coefficients.size()));
    };
    std::vector<I> result(order+1);I right_squared=kappa,right=vm::sqrt_positive(kappa);
    for(int panel=0;panel<panels;++panel) {
        const auto left=right;
        right_squared=si::add(kappa,si::product(increment,point(panel+1)));
        // The full interval ends exactly at one; exploit that identity rather
        // than subtracting rounded theta from one in a narrow erfc interval.
        if(fraction==1. && panel+1==panels)right_squared=point(1.);
        right=vm::sqrt_positive(right_squared);
        const auto width=vm::divide(increment,si::add(left,right));
        const auto decay=vm::exp_negative(nonnegative(si::product(t,right_squared)));
        const auto multiplier=si::product(t,width);
        std::vector<I> monomial{point(1.)};
        for(int n=0;n<=order;++n) {
            if(n) {monomial=linear(monomial,left,right);monomial=linear(monomial,left,right);}
            auto coefficients=monomial;I sum=integrate(coefficients);bool finished=false;
            for(int k=1;k<=64;++k) {
                // Q=T*h*(1-y)*(a+b+h*y), with Q<=T*(b^2-a^2).
                coefficients=linear(coefficients,point(1.),point(0.));
                coefficients=linear(coefficients,si::add(left,right),si::product(point(2.),right));
                const auto factor=vm::divide(multiplier,point(k));
                for(auto& x:coefficients)x=si::product(x,factor);
                const auto next=integrate(coefficients);
                const auto ratio=vm::divide(panel_gain,point(k+1));
                if(ratio.hi<1.) {
                    const auto tail=vm::divide(next,si::sub(point(1.),ratio));
                    if(tail.hi<=sum.hi*1e-18 || k==64) {
                        sum=si::add(sum,{0.,tail.hi});finished=true;break;
                    }
                }
                sum=si::add(sum,next);
            }
            if(!finished)throw std::runtime_error("MANTA positive radial series budget exceeded");
            result[n]=si::add(result[n],si::product(si::product(width,decay),sum));
        }
    }
    if(fraction<1.) {
        const auto decay=vm::exp_negative(nonnegative(si::product(t,right_squared)));
        const auto remaining=vm::divide(si::product(complement,si::sub(point(1.),point(fraction))),si::add(point(1.),right));
        double tail=si::product(remaining,decay).hi;
        const auto slope=si::product(si::product(point(2.),t),right);
        if(slope.lo>0.)tail=std::min(tail,vm::divide(decay,slope).hi);
        // u^(2n)<=1 on the omitted interval. The second bound extends the
        // Gaussian integral to infinity and uses u/b>=1 for u>=b>0.
        for(auto& x:result)x=si::add(x,{0.,tail});
    }
    for(auto& x:result)x=nonnegative(x);return result;
}
} // namespace vibeqc::manta
