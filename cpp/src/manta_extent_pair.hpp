#pragma once
#include "vibeqc/manta_extents.hpp"

namespace vibeqc::manta::detail {
// Internal shifted-shell entry point. The owner must first admit all work,
// validate the native basis/options and check translated centers. Reuses the
// exact molecular envelope, without constructing a replicated image basis.
PairExtent pair_extent(const libint2::Shell&,const libint2::Shell&,
    std::array<std::size_t,2>,const ExtentOptions&,Count& evaluations,Count& always_near);
AuxiliaryExtent auxiliary_extent(const libint2::Shell&,std::size_t,
    const ExtentOptions&,Count& evaluations,Count& always_near);
} // namespace vibeqc::manta::detail
