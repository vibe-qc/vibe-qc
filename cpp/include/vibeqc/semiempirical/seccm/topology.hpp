// Hamiltonian-independent SECCM Wigner-Seitz image topology.
//
// The builder preserves validated MSINDO ownership on ordinary cells and
// extends it to arbitrary full-rank skew bases with a cutoff-complete closest-
// image search. Image labels stay in the caller's original lattice basis.
// Method-family Hamiltonians, electrostatics, and derivatives do not belong
// in this header.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace vibeqc {
namespace semiempirical {
namespace seccm {

using ImageShellLabel = std::array<int, 3>;

struct WSImage {
    int origin;
    double weight;
    Eigen::Vector3d disp;
    ImageShellLabel image_shell_label{0, 0, 0};
    int ownership_multiplicity = 1;
};

struct WSTopology {
    std::vector<std::vector<WSImage>> cells;
    std::vector<Eigen::Vector3d> translations;

    double total_weight(std::size_t central) const {
        double total = 0.0;
        for (const auto& image : cells[central]) total += image.weight;
        return total;
    }

    bool is_valid(int natoms) const {
        for (int central = 0; central < natoms; ++central) {
            if (std::round(total_weight(static_cast<std::size_t>(central)))
                != natoms - 1) {
                return false;
            }
        }
        return true;
    }
};

namespace detail {

inline Eigen::Vector3d label_translation(
    const ImageShellLabel& label,
    const std::vector<Eigen::Vector3d>& translations) {
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    for (std::size_t axis = 0; axis < translations.size(); ++axis) {
        result += static_cast<double>(label[axis]) * translations[axis];
    }
    return result;
}

// Equidistant-image tie tolerance, in the caller's length unit (bohr on every
// vibe-qc route).  Bredow, Geudtner & Jug, J. Comput. Chem. 22, 89 (2001),
// pp. 90-91: in bulk MgO and NiO "there are always several neighbors N at the
// borders of the Wigner-Seitz unit cell around atom I with the same distance",
// and keeping only one of them "would lead to a non-symmetric region with a
// dipole moment", so all of them are retained with Evjen weight 1/n (also
// Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014), Theory, two-center
// weights).  "The same distance" must be judged far above the precision of any
// real input geometry and far below any physical distinction: unit conversions
// with different CODATA constants disagree at 4.4e-10 relative (5e-9 bohr on
// MgO 2x2x2), a six-decimal Angstrom file at 1e-6 bohr.  MSINDO neighbors.f
// shares every image whose distance to the central atom is within 1.0E-5 bohr
// of the minimum; the pure 256-eps roundoff guard that replaced that rule in
// 65f6b0048 broke the 4-, 2- and 6-fold ties of MgO 2x2x2 at a 5e-9 bohr
// mismatch, dropped 192 of 512 shared records while the weight-sum validity
// rule still passed, moved the MSINDO SECCM energy by 6e-2 Ha and made it
// depend on which lattice copy of an atom the input named (issues #592, #593).
// The Python builder (seccm/topology.py, _EQUIDISTANT_IMAGE_TOLERANCE) carries
// the same constant; the two must stay bit-identical.
constexpr double kEquidistantImageTolerance = 1.0e-5;

inline double closest_image_tolerance(
    const Eigen::Vector3d& displacement,
    const std::vector<Eigen::Vector3d>& translations) {
    double scale = std::max(1.0, displacement.norm());
    for (const auto& vector : translations) {
        scale = std::max(scale, vector.norm());
    }
    return std::max(
        kEquidistantImageTolerance,
        256.0 * std::numeric_limits<double>::epsilon() * scale);
}

// Enumerate a mathematically complete coefficient box in the caller's input
// basis. If f is the real least-squares lattice coefficient and sigma_min is
// the smallest singular value of L, every label n with residual <= R obeys
// |n_i-f_i| <= R/sigma_min. A rounded coefficient supplies a finite upper
// bound R, so this search cannot omit a closest image even for a skew basis.
inline std::vector<std::pair<ImageShellLabel, Eigen::Vector3d>>
closest_images(
    const Eigen::Vector3d& displacement,
    const std::vector<Eigen::Vector3d>& translations) {
    const int dimensionality = static_cast<int>(translations.size());
    if (dimensionality < 1 || dimensionality > 3) {
        throw std::invalid_argument(
            "SECCM closest-image search requires one to three translations");
    }
    Eigen::MatrixXd lattice(dimensionality, 3);
    for (int axis = 0; axis < dimensionality; ++axis) {
        lattice.row(axis) =
            translations[static_cast<std::size_t>(axis)].transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        lattice, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto singular_values = svd.singularValues();
    const double scale = std::max(1.0, singular_values[0]);
    const double rank_tolerance = std::max(
        1.0e-12,
        64.0 * std::numeric_limits<double>::epsilon() * scale);
    const double sigma_min = singular_values[dimensionality - 1];
    if (sigma_min <= rank_tolerance) {
        throw std::invalid_argument(
            "SECCM translations must be nonzero and linearly independent");
    }

    const Eigen::MatrixXd gram = lattice * lattice.transpose();
    const Eigen::VectorXd fractional =
        gram.ldlt().solve(lattice * (-displacement));
    ImageShellLabel rounded{0, 0, 0};
    for (int axis = 0; axis < dimensionality; ++axis) {
        const double value = std::nearbyint(fractional[axis]);
        if (value < static_cast<double>(std::numeric_limits<int>::min())
            || value > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "SECCM closest-image label exceeds integer range");
        }
        rounded[axis] = static_cast<int>(value);
    }
    const double upper =
        (displacement + label_translation(rounded, translations)).norm();
    const double coefficient_radius =
        (upper + closest_image_tolerance(displacement, translations))
            / sigma_min
        + 1.0;

    ImageShellLabel lower{0, 0, 0};
    ImageShellLabel upper_label{0, 0, 0};
    std::size_t candidate_count = 1;
    for (int axis = 0; axis < dimensionality; ++axis) {
        const double lo = std::floor(fractional[axis] - coefficient_radius);
        const double hi = std::ceil(fractional[axis] + coefficient_radius);
        if (lo < static_cast<double>(std::numeric_limits<int>::min())
            || hi > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "SECCM closest-image search exceeds integer label range");
        }
        lower[axis] = static_cast<int>(lo);
        upper_label[axis] = static_cast<int>(hi);
        const std::size_t width = static_cast<std::size_t>(
            static_cast<long long>(upper_label[axis])
            - static_cast<long long>(lower[axis]) + 1);
        if (width > 10000000 / candidate_count) {
            throw std::invalid_argument(
                "SECCM skew lattice is too ill-conditioned for exact native "
                "closest-image enumeration");
        }
        candidate_count *= width;
    }

    struct Candidate {
        ImageShellLabel label;
        Eigen::Vector3d residual;
        double distance;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(candidate_count);
    const int j_lower = dimensionality >= 2 ? lower[1] : 0;
    const int j_upper = dimensionality >= 2 ? upper_label[1] : 0;
    const int k_lower = dimensionality >= 3 ? lower[2] : 0;
    const int k_upper = dimensionality >= 3 ? upper_label[2] : 0;
    double minimum = std::numeric_limits<double>::infinity();
    for (int i = lower[0]; i <= upper_label[0]; ++i) {
        for (int j = j_lower; j <= j_upper; ++j) {
            for (int k = k_lower; k <= k_upper; ++k) {
                const ImageShellLabel label{i, j, k};
                const Eigen::Vector3d residual =
                    displacement + label_translation(label, translations);
                const double distance = residual.norm();
                minimum = std::min(minimum, distance);
                candidates.push_back({label, residual, distance});
            }
        }
    }

    const double tolerance =
        closest_image_tolerance(displacement, translations);
    std::vector<std::pair<ImageShellLabel, Eigen::Vector3d>> closest;
    for (const auto& candidate : candidates) {
        if (candidate.distance <= minimum + tolerance) {
            closest.push_back({candidate.label, candidate.residual});
        }
    }
    std::sort(
        closest.begin(), closest.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    return closest;
}

}  // namespace detail

inline WSTopology build_wigner_seitz(
    const std::vector<Eigen::Vector3d>& coords,
    const std::vector<Eigen::Vector3d>& translations) {
    int natoms = static_cast<int>(coords.size());
    WSTopology topology;
    topology.translations = translations;
    topology.cells.resize(natoms);

    struct IncludedImage {
        int origin;
        Eigen::Vector3d displacement;
        ImageShellLabel image_shell_label;
    };

    for (int central = 0; central < natoms; ++central) {
        std::vector<IncludedImage> included;
        for (int origin = 0; origin < natoms; ++origin) {
            const Eigen::Vector3d displacement =
                coords[origin] - coords[central];
            const auto closest =
                detail::closest_images(displacement, translations);
            for (const auto& image : closest) {
                const auto& label = image.first;
                if (origin == central && label == ImageShellLabel{0, 0, 0}) {
                    continue;
                }
                const Eigen::Vector3d position = coords[origin]
                    + detail::label_translation(label, translations);
                included.push_back(
                    {origin, position - coords[central], label});
            }
        }

        std::map<int, int> multiplicities;
        for (const auto& image : included) ++multiplicities[image.origin];
        for (const auto& image : included) {
            int multiplicity = multiplicities[image.origin];
            topology.cells[central].push_back({
                image.origin,
                1.0 / multiplicity,
                image.displacement,
                image.image_shell_label,
                multiplicity,
            });
        }
    }
    return topology;
}

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
