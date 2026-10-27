#pragma once
#include "vibeqc/manta_fitting.hpp"
#include "manta_ri_interval.hpp"
namespace vibeqc::manta {
// Shared exact Coulomb radial reference; callers admit work and arithmetic.
// This does not evaluate or certify a finite integral producer by itself.
std::vector<spectral_interval::Interval> coulomb_boys_enclosure(int,
    spectral_interval::Interval);
// Caller admits basis, shell and primitive work before requesting this proof.
std::vector<spectral_interval::Interval> metric_shell_enclosure(const libint2::Shell&,const libint2::Shell&,
    md::RadialKernel,double,const Eigen::Vector3d& translation=Eigen::Vector3d::Zero());
} // namespace vibeqc::manta
