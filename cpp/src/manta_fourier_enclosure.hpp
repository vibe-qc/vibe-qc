#pragma once
#include "vibeqc/manta_fourier.hpp"
#include "manta_ri_interval.hpp"
#include "manta_geometry_enclosure.hpp"
namespace vibeqc::manta {
std::vector<spectral_interval::ComplexInterval> pair_fourier_enclosure(
    const libint2::Shell&,const libint2::Shell&,const md::Matrix&);
std::vector<spectral_interval::ComplexInterval> pair_fourier_enclosure(
    const libint2::Shell&,const libint2::Shell&,const md::Matrix&,const EnclosedPoint&,const EnclosedPoint&);
spectral_interval::ComplexInterval bloch_phase_enclosure(const EnclosedPoint&,const EnclosedPoint&);
// Internal producer seam: caller admits complete basis/panel counts first.
std::vector<spectral_interval::ComplexInterval> auxiliary_fourier_enclosure(
    const libint2::Shell&,const md::Matrix&);
}
