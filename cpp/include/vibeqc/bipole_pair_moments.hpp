// Per-shell-pair adjoined-Gaussian centre and standard binomial moment shift.
// Pisani-Dovesi-Roetti (1988), Ch. II.4c, places the periodic expansion at
// product-distribution centroids. Pisani-Dovesi (1980), Sec. 4, defines the
// adjoined diffuse s-Gaussian screening convention; the standard Gaussian
// product theorem supplies its approximate pair centre.

#pragma once

#include "basis.hpp"
#include "lattice_sum.hpp"
#include "multipole_moments_lattice.hpp"

#include <Eigen/Dense>
#include <array>
#include <utility>
#include <vector>

namespace vibeqc {

struct PairMultipoleMomentsCpp {
    int nbf = 0;
    int L_max = 0;
    std::vector<LatticeCell> cells;
    std::vector<std::vector<Eigen::MatrixXd>> blocks;  // [c][comp]
    std::vector<std::vector<std::array<double, 3>>> centres_flat;
    std::vector<std::pair<int, int>> shell_slices;
};

std::vector<std::vector<std::array<double, 3>>> compute_adjoined_pair_centres(
    const BasisSet& basis, const std::vector<LatticeCell>& cells);

PairMultipoleMomentsCpp shift_multipole_moments_to_pair_centres(
    const LatticeMultipoleSet& M_lat, const BasisSet& basis, int L_target,
    const std::array<double, 3>& origin = {0.0, 0.0, 0.0});

}  // namespace vibeqc
