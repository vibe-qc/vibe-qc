#pragma once

#include <stdexcept>
#include <string>

#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"

namespace vibeqc::semiempirical::xtb::detail {

struct ValenceOccupations {
    int alpha;
    int beta;
};

// GFN2 freezes element-dependent core subshells, so molecular spin must be
// compatible with the remaining valence electrons, not just the sum of Z.
inline ValenceOccupations valence_occupations(
    const Molecule& mol, int n_basis, const char* driver) {
    long long electrons = -static_cast<long long>(mol.charge());
    for (const auto& atom : mol.atoms())
        electrons += gfn2_valence_electrons(atom.Z);
    const long long unpaired = static_cast<long long>(mol.multiplicity()) - 1;
    const std::string prefix = std::string(driver) + ": ";
    if (electrons < 0)
        throw std::invalid_argument(prefix + "valence electron count must be nonnegative");
    if (unpaired < 0 || unpaired > electrons || (electrons - unpaired) % 2 != 0)
        throw std::invalid_argument(
            prefix + "valence electron count is incompatible with multiplicity");
    const long long alpha = (electrons + unpaired) / 2;
    const long long beta = (electrons - unpaired) / 2;
    if (alpha > n_basis || beta > n_basis)
        throw std::invalid_argument(prefix + "spin occupations exceed basis capacity");
    return {static_cast<int>(alpha), static_cast<int>(beta)};
}

}  // namespace vibeqc::semiempirical::xtb::detail
