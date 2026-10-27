#pragma once
#include "vibeqc/manta_fourier.hpp"
#include "manta_ri_interval.hpp"
namespace vibeqc::manta {
// Internal producer seam: caller admits complete basis/panel counts first.
std::vector<spectral_interval::ComplexInterval> auxiliary_fourier_enclosure(
    const libint2::Shell&,const md::Matrix&);
}
