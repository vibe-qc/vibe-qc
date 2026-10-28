#pragma once
#include "vibeqc/manta_three_center.hpp"
#include "manta_ri_interval.hpp"
#include "manta_geometry_enclosure.hpp"
namespace vibeqc::manta {
// Shell centers are exact stored inputs. Caller admits full tile work first.
std::vector<spectral_interval::Interval> three_center_shell_enclosure(
    const libint2::Shell&,const libint2::Shell&,const libint2::Shell&,md::RadialKernel,double);
std::vector<spectral_interval::Interval> three_center_shell_enclosure(
    const libint2::Shell&,const libint2::Shell&,const libint2::Shell&,md::RadialKernel,double,
    const EnclosedPoint&,const EnclosedPoint&,const EnclosedPoint&);
} // namespace vibeqc::manta
