// Lebedev-Laikov angular quadrature grids.
//
// References (mathematical content; no proprietary source consulted):
//   Lebedev, V. I., "Quadratures on a sphere", USSR Comput. Math. Math.
//     Phys. 16, 10 (1976).
//   Laikov, D. N. & Lebedev, V. I., "A quadrature formula for the
//     sphere of the 131st algebraic order of accuracy", Russ. Acad.
//     Sci. Dokl. Math. 59, 477 (1999).
//
// Numerical values are checked into ``cpp/src/lebedev_data.cpp``, which
// is generated from ``scipy.integrate.lebedev_rule`` by
// ``scripts/generate_lebedev_data.py`` (re-run after a scipy upgrade as
// a self-check). The data is mathematical fact — not subject to copyright.
//
// Weight convention: each grid satisfies Σ w_k = 4π (full-sphere area),
// matching vibe-qc's existing product-grid angular weights so a Lebedev
// grid is a drop-in replacement at the build-grid layer.
//
// For each tier the data layout is interleaved (x, y, z, w) per point,
// row-stride 4 doubles. Points are unit vectors on the sphere.

#pragma once

#include <cstddef>

namespace vibeqc {

// One entry of the dispatch table that ``lebedev_tiers()`` exposes.
struct LebedevTier {
    int order;             // Algebraic order of accuracy (≥ this Y_LM ok)
    std::size_t n_points;  // Number of quadrature points on the sphere
    // Interleaved (x, y, z, w) per point, row-stride 4 doubles. The
    // pointer is to a static const array in ``lebedev_data.cpp``.
    const double* data;
};

// Pointer to the static dispatch table of available Lebedev tiers,
// ordered by ascending ``order``. The returned pointer is valid for
// the program lifetime.
const LebedevTier* lebedev_tiers() noexcept;

// Number of entries in the table returned by ``lebedev_tiers()``.
std::size_t lebedev_tier_count() noexcept;

}  // namespace vibeqc
