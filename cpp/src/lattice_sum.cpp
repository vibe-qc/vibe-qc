#include "vibeqc/lattice_sum.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace vibeqc {

namespace {

// Perpendicular distance from origin to the hyperplane spanned by the other
// two lattice vectors. Equals V / ||a_j × a_k||. For the bounding box of a
// Cartesian-cutoff sphere we need max |n_i| = ceil(cutoff / d_i).
//
// In dim < 3 the vacuum columns are irrelevant; we only call this for the
// periodic columns.
double interplanar_distance(const Eigen::Matrix3d& lattice, int axis) {
    const Eigen::Vector3d a0 = lattice.col((axis + 1) % 3);
    const Eigen::Vector3d a1 = lattice.col((axis + 2) % 3);
    const Eigen::Vector3d cross = a0.cross(a1);
    const double det = lattice.determinant();
    const double cross_norm = cross.norm();
    if (!std::isfinite(det) || !std::isfinite(cross_norm)
        || cross_norm < 1.0e-14) {
        throw std::runtime_error(
            "direct_lattice_cells: degenerate lattice (cross product ≈ 0)");
    }
    const double distance = std::abs(det) / cross_norm;
    if (!std::isfinite(distance) || distance < 1.0e-14) {
        throw std::runtime_error(
            "direct_lattice_cells: degenerate lattice (zero interplanar "
            "distance)");
    }
    return distance;
}

}  // namespace

std::vector<LatticeCell> direct_lattice_cells(const PeriodicSystem& system,
                                              double cutoff_bohr) {
    if (!std::isfinite(cutoff_bohr) || cutoff_bohr < 0.0) {
        throw std::runtime_error(
            "direct_lattice_cells: cutoff must be finite and nonnegative");
    }
    if (system.dim < 1 || system.dim > 3) {
        throw std::runtime_error("direct_lattice_cells: dim must be 1, 2, or 3");
    }
    if (!system.lattice.allFinite()) {
        throw std::runtime_error(
            "direct_lattice_cells: lattice must contain only finite values");
    }

    // Bounding-box extents along each periodic axis. For non-periodic axes
    // we set the extent to 0 so the loop collapses to a single iteration.
    std::array<int, 3> n_max = {0, 0, 0};
    for (int i = 0; i < system.dim; ++i) {
        // Use 3D interplanar distance; for dim=1 or 2 this is slightly
        // pessimistic (the "perpendicular" direction includes vacuum axes
        // that don't actually restrict the bounding region), which means
        // we may enumerate a few extra cells and then reject by |r|. That
        // overhead is a rounding error in practice.
        const double d = interplanar_distance(system.lattice, i);
        const double extent = std::ceil(cutoff_bohr / d);
        if (!std::isfinite(extent)
            || extent >= static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "direct_lattice_cells: cutoff/lattice ratio is too large");
        }
        n_max[i] = static_cast<int>(extent);
    }

    std::vector<LatticeCell> cells;
    // A loose reservation: product of (2n_i+1). Fine for any realistic
    // cutoff, keeps single allocation for a periodic system.
    std::size_t cell_bound = 1;
    for (int extent : n_max) {
        const auto factor = static_cast<std::size_t>(extent) * 2 + 1;
        if (cell_bound > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::runtime_error(
                "direct_lattice_cells: requested cell count is too large");
        }
        cell_bound *= factor;
    }
    cells.reserve(cell_bound);

    const double cutoff_sq = cutoff_bohr * cutoff_bohr;
    const auto& L = system.lattice;

    for (int n1 = -n_max[0]; n1 <= n_max[0]; ++n1) {
        for (int n2 = -n_max[1]; n2 <= n_max[1]; ++n2) {
            for (int n3 = -n_max[2]; n3 <= n_max[2]; ++n3) {
                const Eigen::Vector3d r =
                    static_cast<double>(n1) * L.col(0) +
                    static_cast<double>(n2) * L.col(1) +
                    static_cast<double>(n3) * L.col(2);
                if (r.squaredNorm() <= cutoff_sq) {
                    LatticeCell c;
                    c.index << n1, n2, n3;
                    c.r_cart = r;
                    cells.push_back(c);
                }
            }
        }
    }

    // Sort by |r| so the zero cell comes first and distant cells last —
    // convenient for later screening / truncation-error studies.
    //
    // STABLE, deliberately. A crystal lattice has large groups of exactly
    // equal |r| (all 12 nearest neighbours of an FCC cell, say), and an
    // unstable sort may order such a group any way it likes — differently
    // between two cutoffs, and in principle differently between standard
    // library versions, which would make a cell-resolved result
    // platform-dependent. Stability buys two properties:
    //
    //   * a group's order is its input order, i.e. lexicographic in
    //     (n1, n2, n3), everywhere;
    //   * the result for a cutoff R is an exact PREFIX of the result for
    //     any larger cutoff, so a matrix built on a wider cell list stays
    //     index-aligned with one built on a narrower one.
    std::stable_sort(cells.begin(), cells.end(),
                     [](const LatticeCell& a, const LatticeCell& b) {
                         return a.r_cart.squaredNorm() < b.r_cart.squaredNorm();
                     });
    return cells;
}

void require_nonzero_lattice_image(const std::vector<LatticeCell>& cells,
                                   double cutoff_bohr,
                                   const char* context) {
    const bool has_nonzero_image = std::any_of(
        cells.begin(), cells.end(), [](const LatticeCell& cell) {
            return (cell.index.array() != 0).any();
        });
    if (has_nonzero_image) return;

    const std::string prefix = context ? std::string(context)
                                       : std::string("periodic calculation");
    throw std::invalid_argument(
        prefix + ": cutoff_bohr=" + std::to_string(cutoff_bohr)
        + " contains no nonzero lattice image; refusing to report a "
          "free-boundary cluster as periodic. Increase the cutoff, use a "
          "primitive cell with its matched k mesh where supported, or select "
          "an explicitly supported gamma_only_0 molecular mode");
}

}  // namespace vibeqc
