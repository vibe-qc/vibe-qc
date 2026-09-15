// Translation-invariant enumeration of the images a two-centre lattice sum
// needs.
//
// A periodic two-centre matrix element is a sum over lattice translations
// of the ket centre,
//
//     <a|K|b>^per = sum_P <a|K|b_P> ,
//
// and the lattice-dependent factor depends on P ONLY through the physical
// separation of the two centres, T(P) = rho * ||A - B - P||^2 (Sharma &
// Beylkin, J. Chem. Theory Comput. (2021),
// doi:10.1021/acs.jctc.0c01195, Eqs. 17-18 and the definition of T(P)
// beneath Eq. 18; they prescribe accumulating over increasing P until that
// sum converges). The translations that contribute therefore form a ball of
// radius R_cut centred on the intra-cell offset A - B, NOT on the origin.
//
// Enumerating |g| <= R_cut and then computing every shell pair at that g --
// which is what vibe-qc's periodic kernels did before 2026-08-03 -- both
// misses translations that bring a distant pair into range and visits ones
// that do not, asymmetrically between the (P,Q) and (Q,P) orderings. The
// resulting matrix is not translation invariant: re-describing the same
// crystal by moving an atom one lattice vector changes it. Measured
// consequences of the same defect elsewhere in the tree:
//
//   * the compensated 2c Coulomb metric on MgO primitive FCC / def2-svp-jk
//     (offset 6.89 bohr) had a -5.4e-05 eigenvalue although it is a Gram
//     matrix, and moved by 7.2e-02 under a lattice translation; MDF's
//     orthogonalisation divides by its square root, which is how MgO
//     reached +5.3e+05 Ha (fixed in aux_eri.cpp, 2026-08-03);
//   * the Gamma overlap of LiH rocksalt / def2-svp at the shipped
//     cutoff_bohr = 15 moved by 7.2e-01 -- an overlap element is O(1) --
//     under the same relabelling. STILL OPEN as of 2026-08-06: the fix
//     needs images beyond the |g| ball, so the one-electron cell list
//     grows ~2.7x and a family of consumers that pair it index-for-index
//     with the plain ball has to move with it. A prototype was built and
//     measured (invariance 7.1e-15; Si/def2-SVP becomes positive definite
//     at the shipped cutoff). See handovers/HANDOVER_OPEN_BUGS_V015.md for
//     the measurements, the consumer inventory, and which sums must NOT
//     get this treatment.
//
// A single-atom cell has zero intra-cell offset, so the |g| ball IS the
// correct ball there; every vacuum-box fixture is structurally blind to
// this class.
//
// The fix pattern these helpers implement: pad the enumeration radius by
// the largest intra-cell offset so the cell list is a superset of what any
// pair needs, then filter each (pair, g) on the true separation. The set of
// (pair, g) terms is then exactly {(p, q, g) : ||O_p - O_q - g|| <= R_cut},
// which is translation invariant by construction, and the engine-call count
// stays at the physically required number rather than growing with the
// padding.
//
// Kernels that share a consistency contract must move together: an energy
// sum with its gradient partner (the gradient must differentiate the sum
// the energy builds) and with any cell-resolved *_blocks variant (summing
// the blocks must reproduce the Gamma result).

#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include <libint2/basis.h>
#include <libint2/shell.h>

#include <stdexcept>
#include <string>
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

// Largest intra-cell separation between a bra-shell origin and a
// ket-shell origin, in bohr. This is how far the ball of contributing
// translations can be displaced from the origin.
inline double max_shell_offset(const libint2::BasisSet& bra,
                               const libint2::BasisSet& ket) {
    double d2_max = 0.0;
    for (const auto& p : bra) {
        for (const auto& q : ket) {
            const double dx = p.O[0] - q.O[0];
            const double dy = p.O[1] - q.O[1];
            const double dz = p.O[2] - q.O[2];
            d2_max = std::max(d2_max, dx * dx + dy * dy + dz * dz);
        }
    }
    return std::sqrt(d2_max);
}

// Lattice translations covering every (bra, ket) shell pair whose physical
// separation is within ``cutoff``. A superset: pair_in_range restores the
// exact physical set per pair.
inline std::vector<LatticeCell> pair_complete_cells(
    const PeriodicSystem& system, double cutoff,
    const libint2::BasisSet& bra, const libint2::BasisSet& ket) {
    return direct_lattice_cells(system,
                                cutoff + max_shell_offset(bra, ket));
}

// Is the physical separation of this shell pair within ``cutoff``? ``q`` is
// expected to be the ket shell already translated by the lattice vector, so
// the test is on |O_p - (O_q + g)|.
inline bool pair_in_range(const libint2::Shell& p, const libint2::Shell& q,
                          double cutoff) {
    const double dx = p.O[0] - q.O[0];
    const double dy = p.O[1] - q.O[1];
    const double dz = p.O[2] - q.O[2];
    return dx * dx + dy * dy + dz * dz <= cutoff * cutoff;
}

// As above, but taking the untranslated ket origin and the translation
// explicitly, so a caller can screen before materialising shifted shells.
inline bool pair_in_range(const libint2::Shell& p, const libint2::Shell& q,
                          const Eigen::Vector3d& g, double cutoff) {
    const double dx = p.O[0] - q.O[0] - g[0];
    const double dy = p.O[1] - q.O[1] - g[1];
    const double dz = p.O[2] - q.O[2] - g[2];
    return dx * dx + dy * dy + dz * dz <= cutoff * cutoff;
}

// Physical nuclear images for one AO product. Integer bounds are centered on
// the product midpoint, so relabelling an atom changes only image labels.
// For any accepted image, |n_i - (L^-1 offset)_i| <= R ||(L^-1)_i||.
// This encloses the shifted sphere without a growing origin-centered ball.
struct PairNuclearImageSet {
    std::vector<std::pair<double, std::array<double, 3>>> charges;
    std::vector<long> atoms;
};

class PairNuclearImageSelector {
public:
    PairNuclearImageSelector(const PeriodicSystem& system, double cutoff)
        : system_(system), cutoff_(cutoff), inverse_(system.lattice.inverse()) {
        if (system.dim < 1 || system.dim > 3 || !system.lattice.allFinite() ||
            !inverse_.allFinite() || !std::isfinite(cutoff) || cutoff < 0.0)
            throw std::invalid_argument("Physical nuclear images require a finite lattice and nonnegative cutoff");
        for (const auto& atom : system.unit_cell) {
            for (double coordinate : atom.xyz) {
                if (!std::isfinite(coordinate))
                    throw std::invalid_argument("Nonfinite nuclear coordinate");
            }
        }
        for (int axis = 0; axis < 3; ++axis)
            reach_[axis] = cutoff * inverse_.row(axis).norm();
    }

    PairNuclearImageSet select(const libint2::Shell& a, const libint2::Shell& b,
                              const std::vector<double>* weights = nullptr) const {
        const Eigen::Vector3d center(0.5 * (a.O[0] + b.O[0]),
                                     0.5 * (a.O[1] + b.O[1]),
                                     0.5 * (a.O[2] + b.O[2]));
        return select(center, weights);
    }

    PairNuclearImageSet select(const Eigen::Vector3d& center,
                              const std::vector<double>* weights = nullptr) const {
        if (!center.allFinite()) throw std::invalid_argument("Nonfinite AO-product center");
        PairNuclearImageSet result;
        for (std::size_t atom = 0; atom < system_.unit_cell.size(); ++atom) {
            const auto& nucleus = system_.unit_cell[atom];
            const double charge = weights && atom < weights->size()
                ? (*weights)[atom] : static_cast<double>(nucleus.Z);
            if (charge == 0.0) continue;
            const Eigen::Vector3d origin(nucleus.xyz[0], nucleus.xyz[1], nucleus.xyz[2]);
            const Eigen::Vector3d fractional = inverse_ * (center - origin);
            std::array<int, 3> lower{0, 0, 0}, upper{0, 0, 0};
            for (int axis = 0; axis < system_.dim; ++axis) {
                const double slack = 1e-10 * (1.0 + std::abs(fractional[axis]) + reach_[axis]);
                const double lo = std::floor(fractional[axis] - reach_[axis] - slack);
                const double hi = std::ceil(fractional[axis] + reach_[axis] + slack);
                if (!std::isfinite(lo) || !std::isfinite(hi) ||
                    lo <= static_cast<double>(std::numeric_limits<int>::min()) ||
                    hi >= static_cast<double>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("Physical nuclear image index exceeds the supported range");
                lower[axis] = static_cast<int>(lo);
                upper[axis] = static_cast<int>(hi);
            }
            for (int i = lower[0]; i <= upper[0]; ++i)
            for (int j = lower[1]; j <= upper[1]; ++j)
            for (int k = lower[2]; k <= upper[2]; ++k) {
                const Eigen::Vector3d position = origin + system_.lattice * Eigen::Vector3d(i, j, k);
                const double radius = cutoff_ + 1e-12 * (1.0 + cutoff_);
                if ((position - center).squaredNorm() > radius * radius) continue;
                result.charges.emplace_back(charge, std::array<double, 3>{position[0], position[1], position[2]});
                result.atoms.push_back(static_cast<long>(atom));
            }
        }
        return result;
    }
private:
    const PeriodicSystem& system_;
    double cutoff_;
    Eigen::Matrix3d inverse_;
    std::array<double, 3> reach_;
};

// One cache per worker. Repeated shells on the same atoms share source lists;
// an oversized list uses scratch storage without entering the bounded cache.
class PairNuclearImageCache {
public:
    explicit PairNuclearImageCache(const PairNuclearImageSelector& selector,
                                  const std::vector<double>* weights = nullptr)
        : selector_(selector), weights_(weights) {}

    const PairNuclearImageSet& get(const libint2::Shell& a,
                                   const libint2::Shell& b) {
        const Eigen::Vector3d center(0.5 * (a.O[0] + b.O[0]),
                                     0.5 * (a.O[1] + b.O[1]),
                                     0.5 * (a.O[2] + b.O[2]));
        for (const auto& entry : entries_) {
            if ((entry.center.array() == center.array()).all()) return entry.images;
        }
        scratch_ = selector_.select(center, weights_);
        const std::size_t bytes = scratch_.charges.capacity() * sizeof(scratch_.charges[0])
                                + scratch_.atoms.capacity() * sizeof(scratch_.atoms[0]);
        constexpr std::size_t budget = 8 * 1024 * 1024;
        if (bytes > budget) return scratch_;
        if (entries_.size() == 8 || bytes_ > budget - bytes) {
            entries_.clear();
            bytes_ = 0;
        }
        entries_.push_back({center, std::move(scratch_)});
        bytes_ += bytes;
        return entries_.back().images;
    }
private:
    struct Entry {
        Eigen::Vector3d center;
        PairNuclearImageSet images;
    };
    const PairNuclearImageSelector& selector_;
    const std::vector<double>* weights_;
    std::vector<Entry> entries_;
    PairNuclearImageSet scratch_;
    std::size_t bytes_ = 0;
};

// A finite ERI domain depends on the two physical AO products and their
// midpoint separation. Each product swap and bra/ket interchange preserves it.
inline bool pair_products_in_range(const libint2::Shell& a, const libint2::Shell& b,
                                    const libint2::Shell& c, const libint2::Shell& d,
                                    double pair_cutoff, double interaction_cutoff) {
    if (!pair_in_range(a, b, pair_cutoff) || !pair_in_range(c, d, pair_cutoff)) return false;
    double r2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double delta = 0.5 * ((a.O[axis] - c.O[axis]) + (b.O[axis] - d.O[axis]));
        r2 += delta * delta;
    }
    return r2 <= interaction_cutoff * interaction_cutoff;
}

inline double eri_interaction_cutoff(const LatticeSumOptions& opts) {
    if (!std::isfinite(opts.eri_interaction_cutoff_bohr) ||
        opts.eri_interaction_cutoff_bohr < 0.0)
        throw std::invalid_argument("ERI interaction cutoff must be finite and nonnegative");
    return opts.eri_interaction_cutoff_bohr > 0.0
        ? opts.eri_interaction_cutoff_bohr : opts.cutoff_bohr;
}

// |A-C| <= R_pair/2 + R_interaction + R_pair/2 for every accepted ERI.
inline std::vector<LatticeCell> pair_complete_eri_cells(
    const PeriodicSystem& system, const LatticeSumOptions& opts,
    const libint2::BasisSet& shells) {
    const double radius = opts.cutoff_bohr + eri_interaction_cutoff(opts);
    const Eigen::Matrix3d inverse = system.lattice.inverse();
    if (system.dim < 1 || system.dim > 3 || !system.lattice.allFinite() ||
        !inverse.allFinite() || !std::isfinite(opts.cutoff_bohr) ||
        opts.cutoff_bohr < 0.0 || !std::isfinite(radius))
        throw std::invalid_argument("Physical ERI images require a finite lattice and nonnegative cutoffs");

    // Every retained center relative to A is within R_pair + R_interaction.
    // Generate the union of these displaced spheres directly. A global ball
    // padded by the largest atom offset grows when only image labels change.
    std::set<std::array<double, 3>> offsets;
    for (const auto& a : shells) for (const auto& b : shells) {
        std::array<double, 3> offset;
        for (int axis = 0; axis < 3; ++axis) {
            offset[axis] = a.O[axis] - b.O[axis];
            if (!std::isfinite(offset[axis]))
                throw std::invalid_argument("Nonfinite shell origin in physical ERI images");
        }
        offsets.insert(offset);
    }
    std::set<std::array<int, 3>> indices;
    for (const auto& offset : offsets) {
        const Eigen::Vector3d center(offset[0], offset[1], offset[2]);
        const Eigen::Vector3d fractional = inverse * center;
        std::array<int, 3> lower{0, 0, 0}, upper{0, 0, 0};
        for (int axis = 0; axis < system.dim; ++axis) {
            const double reach = radius * inverse.row(axis).norm();
            const double slack = 1e-10 * (1.0 + std::abs(fractional[axis]) + reach);
            const double lo = std::floor(fractional[axis] - reach - slack);
            const double hi = std::ceil(fractional[axis] + reach + slack);
            // Subsequent quartet queries form differences of these labels.
            constexpr int index_limit = std::numeric_limits<int>::max() / 4;
            if (!std::isfinite(lo) || !std::isfinite(hi) ||
                lo <= -index_limit || hi >= index_limit)
                throw std::invalid_argument("Physical ERI image index exceeds the supported range");
            lower[axis] = static_cast<int>(lo);
            upper[axis] = static_cast<int>(hi);
        }
        const double enclosing_radius = radius + 1e-12 * (1.0 + radius);
        for (int i = lower[0]; i <= upper[0]; ++i)
        for (int j = lower[1]; j <= upper[1]; ++j)
        for (int k = lower[2]; k <= upper[2]; ++k) {
            const Eigen::Vector3d translation = static_cast<double>(i) * system.lattice.col(0)
                + static_cast<double>(j) * system.lattice.col(1)
                + static_cast<double>(k) * system.lattice.col(2);
            if ((translation - center).squaredNorm() <= enclosing_radius * enclosing_radius)
                indices.insert({i, j, k});
        }
    }
    std::vector<LatticeCell> cells;
    cells.reserve(indices.size());
    for (const auto& index : indices) {
        LatticeCell cell;
        cell.index << index[0], index[1], index[2];
        cell.r_cart = static_cast<double>(index[0]) * system.lattice.col(0)
            + static_cast<double>(index[1]) * system.lattice.col(1)
            + static_cast<double>(index[2]) * system.lattice.col(2);
        cells.push_back(cell);
    }
    std::stable_sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) {
        return a.r_cart.squaredNorm() < b.r_cart.squaredNorm();
    });
    return cells;
}

}  // namespace vibeqc
