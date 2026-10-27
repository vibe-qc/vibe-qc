#pragma once
#include "manta_ri_interval.hpp"
#include <libint2/shell.h>
#include <Eigen/Core>
#include <array>
namespace vibeqc::manta {
using EnclosedPoint=std::array<spectral_interval::Interval,3>;
inline EnclosedPoint enclosed_point(const Eigen::Vector3d& p) {
    return {{{p[0],p[0]},{p[1],p[1]},{p[2],p[2]}}};
}
inline EnclosedPoint enclosed_center(const libint2::Shell& s) {
    return {{{s.O[0],s.O[0]},{s.O[1],s.O[1]},{s.O[2],s.O[2]}}};
}
inline EnclosedPoint enclosed_shift(const EnclosedPoint& a,const EnclosedPoint& b) {
    EnclosedPoint out;for(int i=0;i<3;++i)out[i]=spectral_interval::add(a[i],b[i]);return out;
}
inline EnclosedPoint enclosed_image(const Eigen::Matrix3d& lattice,const std::array<long long,3>& n) {
    namespace si=spectral_interval;EnclosedPoint out;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
        const double value=double(n[j]);const auto integer=(n[j]>=-(1ll<<53) && n[j]<=(1ll<<53))?
            si::Interval{value,value}:si::Interval{si::down(value),si::up(value)};
        out[i]=si::add(out[i],si::product(si::Interval{lattice(i,j),lattice(i,j)},integer));
    }
    return out;
}
} // namespace vibeqc::manta
