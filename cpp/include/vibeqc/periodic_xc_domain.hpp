// Finite AO-image and density-difference support shared by XC and its callers.
#pragma once
#include "lattice_sum.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace vibeqc::detail {

// Active bra/ket cells for the cross-cell density sum: cells whose image
// atoms come within `radius` of a home-cell atom, so their lattice-shifted
// AOs overlap the home-atom Becke grid region. The home cell (g=0) is always
// active. Cells beyond `radius` contribute only at far, vanishing-weight grid
// points and are excluded by the same real-space cutoff the S/T/Fock lattice
// sums use (`opts.cutoff_bohr`). The MgO/STO-3G LDA E_xc(radius) study
// converges to < 0.1 mHa by ~12 bohr, well inside the production 14-15 bohr
// cutoff. The direct regression lives in tests/test_periodic_xc_cross_cell.py.
inline std::vector<int> active_density_cells(const std::vector<LatticeCell>& cells,
                                      const PeriodicSystem& system,
                                      double radius) {
    const auto& atoms = system.unit_cell;
    const double r2 = radius * radius;
    std::vector<int> active;
    active.reserve(cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c) {
        if (cells[c].index.isZero()) {
            active.push_back(static_cast<int>(c));
            continue;
        }
        const Eigen::Vector3d& R = cells[c].r_cart;
        bool near = false;
        for (const auto& a : atoms) {
            const Eigen::Vector3d pa(a.xyz[0] + R[0], a.xyz[1] + R[1],
                                     a.xyz[2] + R[2]);
            for (const auto& b : atoms) {
                const Eigen::Vector3d pb(b.xyz[0], b.xyz[1], b.xyz[2]);
                if ((pa - pb).squaredNorm() <= r2) { near = true; break; }
            }
            if (near) break;
        }
        if (near) active.push_back(static_cast<int>(c));
    }
    return active;
}


// Enumerate atom-pair-centred image spheres directly. Their integer bounds
// shift with atom image labels; a global origin ball would grow with a mere
// redescription of the same crystal. Apply the consumer's exact predicate
// after conservative enumeration, including ghost atoms and the home cell.
inline std::vector<LatticeCell> xc_ao_image_cells(
    const PeriodicSystem& system, double radius) {
    const Eigen::Matrix3d inverse = system.lattice.inverse();
    if (system.dim < 1 || system.dim > 3 || !system.lattice.allFinite()
        || !inverse.allFinite() || !std::isfinite(radius) || radius < 0.0)
        throw std::invalid_argument("Periodic XC images require a finite lattice and nonnegative radius");
    std::set<std::array<int, 3>> indices{{0, 0, 0}};
    for (const auto& a : system.unit_cell) for (const auto& b : system.unit_cell) {
        const Eigen::Vector3d offset(b.xyz[0] - a.xyz[0], b.xyz[1] - a.xyz[1], b.xyz[2] - a.xyz[2]);
        if (!offset.allFinite()) throw std::invalid_argument("Nonfinite periodic XC atom position");
        const Eigen::Vector3d center = inverse * offset;
        std::array<int, 3> lo{0, 0, 0}, hi{0, 0, 0};
        for (int axis = 0; axis < system.dim; ++axis) {
            const double reach = radius * inverse.row(axis).norm();
            const double slack = 1e-10 * (1.0 + std::abs(center[axis]) + reach);
            const double lower = std::floor(center[axis] - reach - slack);
            const double upper = std::ceil(center[axis] + reach + slack);
            constexpr int limit = std::numeric_limits<int>::max() / 4;
            if (!std::isfinite(lower) || !std::isfinite(upper) || lower <= -limit || upper >= limit)
                throw std::invalid_argument("Periodic XC image index exceeds the supported range");
            lo[axis] = static_cast<int>(lower);
            hi[axis] = static_cast<int>(upper);
        }
        for (int i = lo[0]; i <= hi[0]; ++i)
        for (int j = lo[1]; j <= hi[1]; ++j)
        for (int k = lo[2]; k <= hi[2]; ++k) {
            const Eigen::Vector3d shift = system.lattice * Eigen::Vector3d(i, j, k);
            const double enclosing = radius + 1e-12 * (1.0 + radius);
            if ((shift - offset).squaredNorm() <= enclosing * enclosing)
                indices.insert({i, j, k});
        }
    }
    std::vector<LatticeCell> candidates;
    for (const auto& i : indices) {
        LatticeCell cell;
        cell.index << i[0], i[1], i[2];
        cell.r_cart = system.lattice * cell.index.cast<double>();
        candidates.push_back(cell);
    }
    std::vector<LatticeCell> cells;
    for (int c : active_density_cells(candidates, system, radius)) cells.push_back(candidates[c]);
    return cells;
}

// rho(r) = sum_(a,s) chi_a(r) P(s-a) chi_s(r). The same difference
// labels carry dE_xc/dP, so an ERI enclosure alone is not an XC domain.
// Retain the existing operator order as a prefix, then add missing exact
// differences. Never pad density blocks with zeros: the SCF inverse-folds
// its actual D(k) onto this complete domain (Sun et al. 2017,
// doi:10.1063/1.4998644, Eq. 3). Operator zero extension is separate.
inline std::vector<LatticeCell> xc_density_cells(
    const PeriodicSystem& system, const LatticeSumOptions& opts,
    std::vector<LatticeCell> cells) {
    const auto ket = xc_ao_image_cells(system, opts.cutoff_bohr);
    const double bra_radius = opts.becke_image_radius_bohr > 0.0
        ? std::min(opts.becke_image_radius_bohr, opts.cutoff_bohr) : opts.cutoff_bohr;
    const auto bra = xc_ao_image_cells(system, bra_radius);
    std::set<std::array<int, 3>> present;
    for (const auto& c : cells) present.insert({c.index[0], c.index[1], c.index[2]});
    for (const auto& a : bra) for (const auto& s : ket) {
        const Eigen::Vector3i i = s.index - a.index;
        if (present.insert({i[0], i[1], i[2]}).second) {
            LatticeCell cell;
            cell.index = i;
            cell.r_cart = system.lattice * i.cast<double>();
            cells.push_back(cell);
        }
    }
    return cells;
}

} // namespace vibeqc::detail
