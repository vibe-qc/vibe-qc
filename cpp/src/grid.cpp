#include "vibeqc/grid.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <atomic>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vibeqc/lebedev_data.hpp"
#include "vibeqc/periodic.hpp"

namespace vibeqc {

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Gauss-Legendre nodes and weights on [-1, 1] via Golub-Welsch.
// ---------------------------------------------------------------------------
struct GLQuadrature {
    Eigen::VectorXd nodes;
    Eigen::VectorXd weights;
};

GLQuadrature gauss_legendre(int n) {
    if (n < 1) throw std::invalid_argument("Gauss-Legendre needs n >= 1");
    // Jacobi matrix for normalized Legendre polynomials: tri-diagonal with
    //   β_k = k / sqrt(4 k^2 - 1),   diagonals all 0.
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    for (int k = 1; k < n; ++k) {
        const double b = k / std::sqrt(4.0 * k * k - 1.0);
        J(k, k - 1) = b;
        J(k - 1, k) = b;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J);
    GLQuadrature q;
    q.nodes = solver.eigenvalues();                           // in [-1, 1]
    q.weights = Eigen::VectorXd(n);
    for (int k = 0; k < n; ++k) {
        const double v0 = solver.eigenvectors()(0, k);
        q.weights(k) = 2.0 * v0 * v0;
    }
    return q;
}

// ---------------------------------------------------------------------------
// Treutler-Ahlrichs M4 radial grid (Chem. Phys. Lett. 240, 283 (1995)).
//   Chebyshev 2nd-kind nodes on [-1, 1] mapped via
//     r(x) = (ξ / ln 2) (1 + x)^α ln(2 / (1 - x))
//   with α = 0.6 and element-dependent ξ (Bragg-Slater-like).
// Returns (r, w) where w already includes the r^2 jacobian and the
// Chebyshev angular weight, so ∫_0^∞ f(r) dr 4π r^2 ≈ 4π Σ w_k f(r_k).
// ---------------------------------------------------------------------------
struct RadialGrid {
    std::vector<double> r;
    std::vector<double> w;
};

// Modest element-specific scale factors ξ (atomic units). Values from the
// Treutler-Ahlrichs paper (rounded); reasonable for the 1st/2nd rows we
// currently exercise. Any Z without an entry falls back to 1.0.
double treutler_xi(int Z) {
    switch (Z) {
        case 1:  return 0.8;                     // H
        case 2:  return 0.9;                     // He
        case 3: case 4:                          // Li, Be
            return 1.8;
        case 5: case 6: case 7: case 8: case 9: case 10:  // B..Ne
            return 0.9;
        default:
            return 1.0;                          // 2nd row+; refine later
    }
}

RadialGrid treutler_ahlrichs(int n, double xi) {
    RadialGrid g;
    g.r.reserve(n);
    g.w.reserve(n);
    const double alpha = 0.6;
    const double ln2 = std::log(2.0);
    for (int k = 1; k <= n; ++k) {
        // Chebyshev 2nd kind nodes & plain-dx quadrature on [-1, 1]:
        //   ∫_{-1}^{1} f(x) dx = ∫_0^π f(cos θ) sin θ dθ
        //                     ≈ (π/(N+1)) Σ_k sin θ_k · f(x_k)
        // with θ_k = k π/(N+1), x_k = cos θ_k.
        const double theta = k * kPi / (n + 1);
        const double x = std::cos(theta);
        const double sin_t = std::sin(theta);
        const double cheb_w = (kPi / (n + 1)) * sin_t;

        // r(x) and dr/dx:
        const double one_m_x = 1.0 - x;
        const double one_p_x = 1.0 + x;
        const double ln_arg = std::log(2.0 / one_m_x);
        const double pow_p = std::pow(one_p_x, alpha);
        const double r = (xi / ln2) * pow_p * ln_arg;
        const double drdx =
            (xi / ln2)
            * (alpha * std::pow(one_p_x, alpha - 1.0) * ln_arg
               + pow_p / one_m_x);

        // Include r^2 Jacobian so this can feed straight into a 3D integral.
        g.r.push_back(r);
        g.w.push_back(cheb_w * drdx * r * r);
    }
    return g;
}

// ---------------------------------------------------------------------------
// PySCF-2.14 level-3 parity profile pinned by vibe-qc for SKALA-1.1.
//
// The period tables, atom-specific Treutler xi values, NWChem pruning rule,
// Bragg radii, and heteroatomic adjustment below are adapted from PySCF 2.14.0
// ``pyscf/dft/{gen_grid,radi}.py`` (Copyright 2014-2024 The PySCF
// Developers, Apache-2.0). The distributed license and upstream NOTICE are
// retained under ``LICENSES/``; implementation changes are the C++ data
// layout, validation, atom-major planning, and periodic image support.
// ---------------------------------------------------------------------------
constexpr double kPyscfBohrAngstrom = 0.52917721092;

// Z-indexed. PySCF 2.14.0 defines the atom-specific table through Lr (Z=103)
// and raises IndexError above it. The profile deliberately fails with an
// actionable error at the same boundary instead of inventing a fallback.
constexpr std::array<double, 104> kPyscfTreutlerXi = {
    1, 0.8, 0.9, 1.8, 1.4, 1.3, 1.1, 0.9,
    0.9, 0.9, 0.9, 1.4, 1.3, 1.3, 1.2, 1.1,
    1, 1, 1, 1.5, 1.4, 1.3, 1.2, 1.2,
    1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1,
    1, 0.9, 0.9, 0.9, 0.9, 2, 1.7, 1.5,
    1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5,
    1.5, 1.3, 1.2, 1.2, 1.15, 1.15, 1.15, 2.5,
    2.2, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5,
    2.1, 3.685, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
};

// Bragg-Slater radii in angstrom, also Z-indexed. Only ratios are used by
// the Treutler cell adjustment, while NWChem pruning converts the selected
// radius to bohr with PySCF's pinned conversion constant above.
constexpr std::array<double, 104> kPyscfBraggRadiiAngstrom = {
    1.999999, 0.35, 1.4, 1.45, 1.05, 0.85, 0.7, 0.65,
    0.6, 0.5, 1.5, 1.8, 1.5, 1.25, 1.1, 1,
    1, 1, 1.8, 2.2, 1.8, 1.6, 1.4, 1.35,
    1.4, 1.4, 1.4, 1.35, 1.35, 1.35, 1.35, 1.3,
    1.25, 1.15, 1.15, 1.15, 1.9, 2.35, 2, 1.8,
    1.55, 1.45, 1.45, 1.35, 1.3, 1.35, 1.4, 1.6,
    1.55, 1.55, 1.45, 1.45, 1.4, 1.4, 2.1, 2.6,
    2.15, 1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,
    1.8, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.55, 1.45, 1.35, 1.35, 1.3, 1.35, 1.35, 1.35,
    1.5, 1.9, 1.8, 1.6, 1.9, 1.45, 2.1, 1.8,
    2.15, 1.95, 1.8, 1.8, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
};

void validate_pyscf_level3_atomic_number(int Z) {
    if (Z < 1 || Z >= static_cast<int>(kPyscfTreutlerXi.size())) {
        throw std::invalid_argument(
            "grid: PySCF level-3 profile requires atomic numbers 1..103; "
            "PySCF 2.14.0 has no atom-specific Treutler xi value for Z="
            + std::to_string(Z));
    }
}

int pyscf_level3_radial_count(int Z) {
    validate_pyscf_level3_atomic_number(Z);
    if (Z <= 2) return 50;
    if (Z <= 10) return 75;
    if (Z <= 18) return 80;
    if (Z <= 36) return 90;
    if (Z <= 54) return 95;
    if (Z <= 86) return 100;
    return 105;
}

int pyscf_level3_max_lebedev_order(int Z) {
    validate_pyscf_level3_atomic_number(Z);
    return Z <= 10 ? 29 : 35;
}

RadialGrid pyscf_treutler_ahlrichs(int n, int Z) {
    validate_pyscf_level3_atomic_number(Z);
    std::vector<double> r(static_cast<std::size_t>(n));
    std::vector<double> dr(static_cast<std::size_t>(n));
    const double step = kPi / (n + 1);
    const double ln2 = kPyscfTreutlerXi[static_cast<std::size_t>(Z)]
                       / std::log(2.0);
    for (int i = 0; i < n; ++i) {
        const double angle = (i + 1) * step;
        const double x = std::cos(angle);
        const double log_term = std::log((1.0 - x) / 2.0);
        const double xpow = std::pow(1.0 + x, 0.6);
        r[static_cast<std::size_t>(i)] = -ln2 * xpow * log_term;
        dr[static_cast<std::size_t>(i)] =
            step * std::sin(angle) * ln2 * xpow
            * (-0.6 / (1.0 + x) * log_term + 1.0 / (1.0 - x));
    }

    // PySCF reverses both arrays: atom-local radii are ascending before the
    // NWChem radius bands are assigned.
    RadialGrid out;
    out.r.reserve(static_cast<std::size_t>(n));
    out.w.reserve(static_cast<std::size_t>(n));
    for (int i = n - 1; i >= 0; --i) {
        const double ri = r[static_cast<std::size_t>(i)];
        out.r.push_back(ri);
        out.w.push_back(dr[static_cast<std::size_t>(i)] * ri * ri);
    }
    return out;
}

std::vector<int> pyscf_nwchem_angular_counts(
    int Z, const std::vector<double>& radii, int max_lebedev_order) {
    validate_pyscf_level3_atomic_number(Z);
    const std::array<double, 4>* alpha = nullptr;
    static constexpr std::array<double, 4> kHHe = {0.25, 0.5, 1.0, 4.5};
    static constexpr std::array<double, 4> kLiNe = {0.1667, 0.5, 0.9, 3.5};
    static constexpr std::array<double, 4> kHeavy = {0.1, 0.4, 0.8, 2.5};
    if (Z <= 2) {
        alpha = &kHHe;
    } else if (Z <= 10) {
        alpha = &kLiNe;
    } else {
        alpha = &kHeavy;
    }

    const std::array<int, 5> tier_points = max_lebedev_order == 29
        ? std::array<int, 5>{50, 86, 266, 302, 266}
        : std::array<int, 5>{50, 86, 350, 434, 350};
    const double atom_radius =
        kPyscfBraggRadiiAngstrom[static_cast<std::size_t>(Z)]
        / kPyscfBohrAngstrom;
    std::vector<int> counts;
    counts.reserve(radii.size());
    for (double radius : radii) {
        int region = 0;
        for (double boundary : *alpha) {
            if (radius / atom_radius > boundary) ++region;
        }
        counts.push_back(tier_points[static_cast<std::size_t>(region)]);
    }
    return counts;
}

// ---------------------------------------------------------------------------
// Becke atomic partitioning (J. Chem. Phys. 88, 2547 (1988)).
// Returns P_A(r) for one point r and a set of atomic positions.
// ---------------------------------------------------------------------------
double becke_switch(double mu, int k) {
    // Iterated polynomial p(mu) = (3/2) mu - (1/2) mu^3, k times.
    for (int i = 0; i < k; ++i) {
        mu = 1.5 * mu - 0.5 * mu * mu * mu;
    }
    return 0.5 * (1.0 - mu);
}

std::vector<double> becke_cell_weights(
    const Eigen::Vector3d& r,
    const std::vector<Eigen::Vector3d>& atom_positions,
    int k_smooth) {
    const std::size_t N = atom_positions.size();
    std::vector<double> dist(N);
    for (std::size_t A = 0; A < N; ++A) {
        dist[A] = (r - atom_positions[A]).norm();
    }
    std::vector<double> P(N, 1.0);
    for (std::size_t A = 0; A < N; ++A) {
        for (std::size_t B = 0; B < N; ++B) {
            if (A == B) continue;
            const double RAB = (atom_positions[A] - atom_positions[B]).norm();
            const double mu = (dist[A] - dist[B]) / RAB;
            P[A] *= becke_switch(mu, k_smooth);
        }
    }
    double Z = 0.0;
    for (const double p : P) Z += p;
    // Normalize so Σ_A P_A = 1.
    if (Z > 0) {
        for (double& p : P) p /= Z;
    }
    return P;
}

double pyscf_treutler_pair_adjustment(int Zi, int Zj) {
    validate_pyscf_level3_atomic_number(Zi);
    validate_pyscf_level3_atomic_number(Zj);
    const double ri = std::sqrt(
        kPyscfBraggRadiiAngstrom[static_cast<std::size_t>(Zi)]
        / kPyscfBohrAngstrom);
    const double rj = std::sqrt(
        kPyscfBraggRadiiAngstrom[static_cast<std::size_t>(Zj)]
        / kPyscfBohrAngstrom);
    double a = 0.25 * (rj / ri - ri / rj);
    return std::clamp(a, -0.5, 0.5);
}

// Same pair traversal and polynomial association as PySCF 2.14.0's
// ``VXCgen_grid`` fast path. ``numbers`` belongs to the full partition list,
// including periodic images when present.
std::vector<double> pyscf_becke_cell_weights(
    const Eigen::Vector3d& r,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const std::vector<int>& numbers) {
    const std::size_t N = atom_positions.size();
    if (numbers.size() != N) {
        throw std::invalid_argument(
            "grid: PySCF level-3 Becke partition needs one atomic number "
            "for every home or image partition atom");
    }
    std::vector<double> dist(N);
    std::vector<double> P(N, 1.0);
    for (std::size_t i = 0; i < N; ++i) {
        validate_pyscf_level3_atomic_number(numbers[i]);
        dist[i] = (r - atom_positions[i]).norm();
    }
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const double Rij = (atom_positions[i] - atom_positions[j]).norm();
            if (!(Rij > 0.0)) {
                throw std::invalid_argument(
                    "grid: coincident atoms are invalid in a Becke "
                    "partition");
            }
            double g = (dist[i] - dist[j]) / Rij;
            const double a = pyscf_treutler_pair_adjustment(
                numbers[i], numbers[j]);
            g += a * (1.0 - g * g);
            double s = g;
            s = (3.0 - s * s) * s * 0.5;
            s = (3.0 - s * s) * s * 0.5;
            s = ((3.0 - s * s) * s * 0.5) * 0.5;
            P[i] *= 0.5 - s;
            P[j] *= 0.5 + s;
        }
    }
    double norm = 0.0;
    for (double value : P) norm += value;
    if (norm > 0.0) {
        for (double& value : P) value /= norm;
    }
    return P;
}

// ---------------------------------------------------------------------------
// Stratmann-Scuseria-Frisch atomic partitioning
//   (Chem. Phys. Lett. 257, 213 (1996), § 11)
// Replaces Becke's iterated 3μ - μ³ smoothing with a single 7th-order
// polynomial on |μ| ≤ a (= 0.64 per the paper) and a hard cutoff
// outside:
//
//      s(μ) = 1                                if μ ≤ -a
//             (1 - g(μ/a)) / 2                 if -a < μ < a
//             0                                if μ ≥ a
//
// where g(z) = (1/16) · (35z - 35z³ + 21z⁵ - 5z⁷) is the unique
// polynomial of degree 7 satisfying g(±1) = ±1 and g'(±1) = g''(±1) =
// g'''(±1) = 0 (so s is C³-smooth at the cutoff). The hard cutoff lets
// the per-grid-point partition loop early-exit when any neighbour-pair
// drives s to zero — asymptotically linear in the number of nearby
// atoms.
// ---------------------------------------------------------------------------
constexpr double kStratmannA = 0.64;

double stratmann_switch(double mu) {
    if (mu <= -kStratmannA) return 1.0;
    if (mu >=  kStratmannA) return 0.0;
    const double z = mu / kStratmannA;
    const double z2 = z * z;
    // g(z) = (z/16) · (35 - 35 z² + 21 z⁴ - 5 z⁶) — Horner-evaluated.
    const double g = (z * (35.0 - z2 * (35.0 - z2 * (21.0 - 5.0 * z2))))
                     / 16.0;
    return 0.5 * (1.0 - g);
}

std::vector<double> stratmann_cell_weights(
    const Eigen::Vector3d& r,
    const std::vector<Eigen::Vector3d>& atom_positions) {
    const std::size_t N = atom_positions.size();
    std::vector<double> dist(N);
    for (std::size_t A = 0; A < N; ++A) {
        dist[A] = (r - atom_positions[A]).norm();
    }
    std::vector<double> P(N, 1.0);
    for (std::size_t A = 0; A < N; ++A) {
        for (std::size_t B = 0; B < N; ++B) {
            if (A == B) continue;
            const double RAB =
                (atom_positions[A] - atom_positions[B]).norm();
            const double mu = (dist[A] - dist[B]) / RAB;
            const double s = stratmann_switch(mu);
            P[A] *= s;
            if (P[A] == 0.0) break;  // hard cutoff — no more work for A
        }
    }
    double Z = 0.0;
    for (const double p : P) Z += p;
    if (Z > 0) {
        for (double& p : P) p /= Z;
    }
    return P;
}

}  // namespace

// Shared core of build_grid / build_grid_periodic. ``grid_atom_pos``
// drives the per-atom radial × angular grid generation;
// ``partition_atom_pos`` is the (possibly larger) set used in the
// Becke partition denominator. For the molecular case the two are
// identical; for the periodic case ``partition_atom_pos`` includes
// image atoms.
//
// ``becke_index_of(grid_idx)`` maps a ``grid_atom_pos`` index ``A`` to
// its position in ``partition_atom_pos`` (which atom in the partition
// list this grid-atom corresponds to). For the molecular case it is
// the identity; for the periodic case the home-cell atoms come first,
// images after.
namespace {

// Cumulative shell-INDEX fractions delimiting the 5 regions of a 2021
// ORCA-style ``orca_angular_points`` AngularGrid. Region b covers radial
// shells [frac[b-1]·n_radial, frac[b]·n_radial); region 0 starts at shell
// 0 and region 4 ends at the last shell. The densest band (region 3) is
// placed over the valence shells (~45-80 % of the radial range), with a
// sparse core (region 0) and a thinned tail (region 4) — the standard
// pruning shape (Murray-Handy-Laming 1993 / SG-1 Gill 1993), and the
// qualitative shape of the Helmich-Paris 2021 5-region grids. These are
// the documented index-fraction stand-in for the paper's unpublished
// per-element Clementi-radius multipliers (see grid.hpp
// ``orca_angular_points``).
constexpr std::array<double, 4> kOrcaRegionFractions = {0.10, 0.25, 0.45,
                                                        0.80};

// Nearest bundled Lebedev tier to a target angular point count. The 2021
// Table I point counts (14/26/50/110/194/302/434/590/770) match bundled
// tiers exactly, so "nearest" is exact for every Table I value; the
// fallback keeps it total for any out-of-table request.
const LebedevTier* lebedev_tier_for_points(int target_npts) {
    const auto* tiers = lebedev_tiers();
    const std::size_t n_tiers = lebedev_tier_count();
    const LebedevTier* best = nullptr;
    int best_diff = std::numeric_limits<int>::max();
    for (std::size_t i = 0; i < n_tiers; ++i) {
        const int d =
            std::abs(static_cast<int>(tiers[i].n_points) - target_npts);
        if (d < best_diff) {
            best_diff = d;
            best = &tiers[i];
        }
    }
    return best;
}

const LebedevTier* exact_lebedev_tier_for_points(int npts) {
    const auto* tiers = lebedev_tiers();
    const std::size_t n_tiers = lebedev_tier_count();
    for (std::size_t i = 0; i < n_tiers; ++i) {
        if (static_cast<int>(tiers[i].n_points) == npts) return &tiers[i];
    }
    throw std::runtime_error(
        "grid: PySCF level-3 profile requires a bundled "
        + std::to_string(npts) + "-point Lebedev rule");
}

// PySCF 2.14.0 ``MakeAngularGrid`` direction position -> bundled SciPy
// Lebedev table row. These permutations were generated by exact coordinate
// matching (maximum distance 0) for the six tiers used by level 3. PySCF's
// Apache-2.0 license and NOTICE are retained under ``LICENSES/``. They are
// exact-profile metadata only: generic Lebedev grids retain table order.
constexpr std::array<std::uint16_t, 50> kPyscfLebedev50Order = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 21, 24,
    23, 25, 26, 27, 28, 30, 29, 31, 32, 33, 41, 34,
    35, 37, 36, 38, 39, 40, 42, 43, 44, 46, 45, 47,
    48, 49,
};

constexpr std::array<std::uint16_t, 86> kPyscfLebedev86Order = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12,
    11, 13, 14, 15, 16, 18, 17, 19, 20, 21, 29, 22,
    23, 25, 24, 26, 27, 28, 30, 31, 32, 34, 33, 35,
    36, 37, 38, 39, 40, 42, 41, 43, 44, 45, 53, 46,
    47, 49, 48, 50, 51, 52, 54, 55, 56, 58, 57, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85,
};

constexpr std::array<std::uint16_t, 266> kPyscfLebedev266Order = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 21, 24,
    23, 25, 26, 27, 28, 30, 29, 31, 32, 33, 41, 34,
    35, 37, 36, 38, 39, 40, 42, 43, 44, 46, 45, 47,
    48, 49, 50, 51, 52, 54, 53, 55, 56, 57, 65, 58,
    59, 61, 60, 62, 63, 64, 66, 67, 68, 70, 69, 71,
    72, 73, 74, 75, 76, 78, 77, 79, 80, 81, 89, 82,
    83, 85, 84, 86, 87, 88, 90, 91, 92, 94, 93, 95,
    96, 97, 98, 99, 100, 102, 101, 103, 104, 105, 113, 106,
    107, 109, 108, 110, 111, 112, 114, 115, 116, 118, 117, 119,
    120, 121, 122, 123, 124, 126, 125, 127, 128, 129, 137, 130,
    131, 133, 132, 134, 135, 136, 138, 139, 140, 142, 141, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
    156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 174, 173, 176, 175, 177, 202, 203,
    204, 206, 205, 208, 207, 209, 178, 179, 180, 182, 181, 184,
    183, 185, 210, 211, 212, 214, 213, 216, 215, 217, 186, 187,
    188, 190, 189, 192, 191, 193, 194, 195, 196, 198, 197, 200,
    199, 201, 218, 219, 220, 222, 221, 224, 223, 225, 250, 251,
    252, 254, 253, 256, 255, 257, 226, 227, 228, 230, 229, 232,
    231, 233, 258, 259, 260, 262, 261, 264, 263, 265, 234, 235,
    236, 238, 237, 240, 239, 241, 242, 243, 244, 246, 245, 248,
    247, 249,
};

constexpr std::array<std::uint16_t, 302> kPyscfLebedev302Order = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12,
    11, 13, 14, 15, 16, 18, 17, 19, 20, 21, 29, 22,
    23, 25, 24, 26, 27, 28, 30, 31, 32, 34, 33, 35,
    36, 37, 38, 39, 40, 42, 41, 43, 44, 45, 53, 46,
    47, 49, 48, 50, 51, 52, 54, 55, 56, 58, 57, 59,
    60, 61, 62, 63, 64, 66, 65, 67, 68, 69, 77, 70,
    71, 73, 72, 74, 75, 76, 78, 79, 80, 82, 81, 83,
    84, 85, 86, 87, 88, 90, 89, 91, 92, 93, 101, 94,
    95, 97, 96, 98, 99, 100, 102, 103, 104, 106, 105, 107,
    108, 109, 110, 111, 112, 114, 113, 115, 116, 117, 125, 118,
    119, 121, 120, 122, 123, 124, 126, 127, 128, 130, 129, 131,
    132, 133, 134, 135, 136, 138, 137, 139, 140, 141, 149, 142,
    143, 145, 144, 146, 147, 148, 150, 151, 152, 154, 153, 155,
    156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 206, 207, 208, 210, 209, 212, 211, 213, 238, 239,
    240, 242, 241, 244, 243, 245, 214, 215, 216, 218, 217, 220,
    219, 221, 246, 247, 248, 250, 249, 252, 251, 253, 222, 223,
    224, 226, 225, 228, 227, 229, 230, 231, 232, 234, 233, 236,
    235, 237, 254, 255, 256, 258, 257, 260, 259, 261, 286, 287,
    288, 290, 289, 292, 291, 293, 262, 263, 264, 266, 265, 268,
    267, 269, 294, 295, 296, 298, 297, 300, 299, 301, 270, 271,
    272, 274, 273, 276, 275, 277, 278, 279, 280, 282, 281, 284,
    283, 285,
};

constexpr std::array<std::uint16_t, 350> kPyscfLebedev350Order = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12,
    11, 13, 14, 15, 16, 18, 17, 19, 20, 21, 29, 22,
    23, 25, 24, 26, 27, 28, 30, 31, 32, 34, 33, 35,
    36, 37, 38, 39, 40, 42, 41, 43, 44, 45, 53, 46,
    47, 49, 48, 50, 51, 52, 54, 55, 56, 58, 57, 59,
    60, 61, 62, 63, 64, 66, 65, 67, 68, 69, 77, 70,
    71, 73, 72, 74, 75, 76, 78, 79, 80, 82, 81, 83,
    84, 85, 86, 87, 88, 90, 89, 91, 92, 93, 101, 94,
    95, 97, 96, 98, 99, 100, 102, 103, 104, 106, 105, 107,
    108, 109, 110, 111, 112, 114, 113, 115, 116, 117, 125, 118,
    119, 121, 120, 122, 123, 124, 126, 127, 128, 130, 129, 131,
    132, 133, 134, 135, 136, 138, 137, 139, 140, 141, 149, 142,
    143, 145, 144, 146, 147, 148, 150, 151, 152, 154, 153, 155,
    156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 206, 207, 208, 210, 209, 212, 211, 213, 238, 239,
    240, 242, 241, 244, 243, 245, 214, 215, 216, 218, 217, 220,
    219, 221, 246, 247, 248, 250, 249, 252, 251, 253, 222, 223,
    224, 226, 225, 228, 227, 229, 230, 231, 232, 234, 233, 236,
    235, 237, 254, 255, 256, 258, 257, 260, 259, 261, 286, 287,
    288, 290, 289, 292, 291, 293, 262, 263, 264, 266, 265, 268,
    267, 269, 294, 295, 296, 298, 297, 300, 299, 301, 270, 271,
    272, 274, 273, 276, 275, 277, 278, 279, 280, 282, 281, 284,
    283, 285, 302, 303, 304, 306, 305, 308, 307, 309, 334, 335,
    336, 338, 337, 340, 339, 341, 310, 311, 312, 314, 313, 316,
    315, 317, 342, 343, 344, 346, 345, 348, 347, 349, 318, 319,
    320, 322, 321, 324, 323, 325, 326, 327, 328, 330, 329, 332,
    331, 333,
};

constexpr std::array<std::uint16_t, 434> kPyscfLebedev434Order = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 21, 24,
    23, 25, 26, 27, 28, 30, 29, 31, 32, 33, 41, 34,
    35, 37, 36, 38, 39, 40, 42, 43, 44, 46, 45, 47,
    48, 49, 50, 51, 52, 54, 53, 55, 56, 57, 65, 58,
    59, 61, 60, 62, 63, 64, 66, 67, 68, 70, 69, 71,
    72, 73, 74, 75, 76, 78, 77, 79, 80, 81, 89, 82,
    83, 85, 84, 86, 87, 88, 90, 91, 92, 94, 93, 95,
    96, 97, 98, 99, 100, 102, 101, 103, 104, 105, 113, 106,
    107, 109, 108, 110, 111, 112, 114, 115, 116, 118, 117, 119,
    120, 121, 122, 123, 124, 126, 125, 127, 128, 129, 137, 130,
    131, 133, 132, 134, 135, 136, 138, 139, 140, 142, 141, 143,
    144, 145, 146, 147, 148, 150, 149, 151, 152, 153, 161, 154,
    155, 157, 156, 158, 159, 160, 162, 163, 164, 166, 165, 167,
    168, 169, 170, 171, 172, 174, 173, 175, 176, 177, 185, 178,
    179, 181, 180, 182, 183, 184, 186, 187, 188, 190, 189, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
    204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
    216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 246, 245, 248, 247, 249, 274, 275,
    276, 278, 277, 280, 279, 281, 250, 251, 252, 254, 253, 256,
    255, 257, 282, 283, 284, 286, 285, 288, 287, 289, 258, 259,
    260, 262, 261, 264, 263, 265, 266, 267, 268, 270, 269, 272,
    271, 273, 290, 291, 292, 294, 293, 296, 295, 297, 322, 323,
    324, 326, 325, 328, 327, 329, 298, 299, 300, 302, 301, 304,
    303, 305, 330, 331, 332, 334, 333, 336, 335, 337, 306, 307,
    308, 310, 309, 312, 311, 313, 314, 315, 316, 318, 317, 320,
    319, 321, 338, 339, 340, 342, 341, 344, 343, 345, 370, 371,
    372, 374, 373, 376, 375, 377, 346, 347, 348, 350, 349, 352,
    351, 353, 378, 379, 380, 382, 381, 384, 383, 385, 354, 355,
    356, 358, 357, 360, 359, 361, 362, 363, 364, 366, 365, 368,
    367, 369, 386, 387, 388, 390, 389, 392, 391, 393, 418, 419,
    420, 422, 421, 424, 423, 425, 394, 395, 396, 398, 397, 400,
    399, 401, 426, 427, 428, 430, 429, 432, 431, 433, 402, 403,
    404, 406, 405, 408, 407, 409, 410, 411, 412, 414, 413, 416,
    415, 417,
};

const std::uint16_t* pyscf_level3_angular_order(std::size_t n_points) {
    switch (n_points) {
        case 50: return kPyscfLebedev50Order.data();
        case 86: return kPyscfLebedev86Order.data();
        case 266: return kPyscfLebedev266Order.data();
        case 302: return kPyscfLebedev302Order.data();
        case 350: return kPyscfLebedev350Order.data();
        case 434: return kPyscfLebedev434Order.data();
        default:
            throw std::logic_error(
                "grid: missing PySCF angular-order permutation for "
                + std::to_string(n_points) + "-point Lebedev rule");
    }
}

// Generic NWChem pruning predates the extra Lebedev rules needed by the
// PySCF-level-3 profile.  Keep its historical tier ladder explicit so adding
// a profile-only rule (for example orders 15, 27, or 31) cannot silently
// change an existing generic grid's point count or numerical result.
bool is_generic_pruning_tier(int order) {
    switch (order) {
        case 11:
        case 17:
        case 23:
        case 29:
        case 35:
        case 41:
        case 47:
        case 53:
            return true;
        default:
            return false;
    }
}

struct PyscfLevel3AtomPlan {
    RadialGrid radial;
    std::vector<int> angular_points_by_shell;
    std::vector<int> shell_order;
    std::vector<const LebedevTier*> tier_by_shell;
    std::size_t point_count = 0;
};

PyscfLevel3AtomPlan make_pyscf_level3_atom_plan(int Z) {
    const int n_radial = pyscf_level3_radial_count(Z);
    const int max_order = pyscf_level3_max_lebedev_order(Z);
    PyscfLevel3AtomPlan plan;
    plan.radial = pyscf_treutler_ahlrichs(n_radial, Z);
    plan.angular_points_by_shell = pyscf_nwchem_angular_counts(
        Z, plan.radial.r, max_order);
    plan.tier_by_shell.resize(static_cast<std::size_t>(n_radial));
    for (int k = 0; k < n_radial; ++k) {
        const int npts = plan.angular_points_by_shell[static_cast<std::size_t>(k)];
        plan.tier_by_shell[static_cast<std::size_t>(k)] =
            exact_lebedev_tier_for_points(npts);
        plan.point_count += static_cast<std::size_t>(npts);
    }

    // PySCF's gen_atomic_grids does not emit radial shells in plain radial
    // order when pruning is active. It groups by ascending angular point
    // count and keeps ascending radial indices within each group. The emitter
    // below additionally reproduces PySCF's 12-radial-shell batches and its
    // tier-local angular-major, radial-minor flattening order.
    std::vector<int> unique_counts = plan.angular_points_by_shell;
    std::sort(unique_counts.begin(), unique_counts.end());
    unique_counts.erase(
        std::unique(unique_counts.begin(), unique_counts.end()),
        unique_counts.end());
    plan.shell_order.reserve(static_cast<std::size_t>(n_radial));
    for (int npts : unique_counts) {
        for (int k = 0; k < n_radial; ++k) {
            if (plan.angular_points_by_shell[static_cast<std::size_t>(k)]
                == npts) {
                plan.shell_order.push_back(k);
            }
        }
    }
    return plan;
}

std::vector<PyscfLevel3AtomPlan> make_pyscf_level3_atom_plans(
    const Molecule& mol) {
    std::vector<PyscfLevel3AtomPlan> plans;
    plans.reserve(mol.atoms().size());
    for (const auto& atom : mol.atoms()) {
        plans.push_back(make_pyscf_level3_atom_plan(atom.Z));
    }
    return plans;
}

Grid build_pyscf_level3_grid_core(
    const Molecule& grid_mol,
    const std::vector<Eigen::Vector3d>& grid_atom_pos,
    const std::vector<Eigen::Vector3d>& partition_atom_pos,
    const std::vector<int>& partition_atom_numbers,
    const std::vector<int>& becke_index_of) {
    const int n_grid_atoms = static_cast<int>(grid_atom_pos.size());
    if (static_cast<std::size_t>(n_grid_atoms) != grid_mol.atoms().size()
        || becke_index_of.size() != grid_atom_pos.size()) {
        throw std::invalid_argument(
            "grid: PySCF level-3 atom-layout metadata is inconsistent");
    }
    if (partition_atom_pos.size() != partition_atom_numbers.size()) {
        throw std::invalid_argument(
            "grid: PySCF level-3 periodic partition needs one atomic "
            "number for every home or image atom position");
    }
    for (int Z : partition_atom_numbers) {
        validate_pyscf_level3_atomic_number(Z);
    }
    for (std::size_t i = 0; i < partition_atom_pos.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if ((partition_atom_pos[i] - partition_atom_pos[j]).norm()
                <= 1e-14) {
                throw std::invalid_argument(
                    "grid: coincident home/image atoms are invalid in a "
                    "PySCF level-3 Becke partition");
            }
        }
    }

    const std::vector<PyscfLevel3AtomPlan> plans =
        make_pyscf_level3_atom_plans(grid_mol);
    std::vector<std::size_t> atom_offsets(
        static_cast<std::size_t>(n_grid_atoms) + 1, 0);
    for (int A = 0; A < n_grid_atoms; ++A) {
        atom_offsets[static_cast<std::size_t>(A + 1)] =
            atom_offsets[static_cast<std::size_t>(A)]
            + plans[static_cast<std::size_t>(A)].point_count;
    }
    const std::size_t total_points = atom_offsets.back();

    Grid g;
    g.atomic_grid_profile = AtomicGridProfile::PySCFLevel3;
    g.points.resize(static_cast<Eigen::Index>(total_points), 3);
    g.weights.resize(static_cast<Eigen::Index>(total_points));
    g.atomic_weights.resize(static_cast<Eigen::Index>(total_points));
    g.atom_of_point.resize(total_points);
    g.atom_coords.resize(n_grid_atoms, 3);
    g.atomic_numbers.resize(static_cast<std::size_t>(n_grid_atoms));
    for (int A = 0; A < n_grid_atoms; ++A) {
        g.atom_coords.row(A) = grid_atom_pos[static_cast<std::size_t>(A)].transpose();
        g.atomic_numbers[static_cast<std::size_t>(A)] =
            grid_mol.atoms()[static_cast<std::size_t>(A)].Z;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int A = 0; A < n_grid_atoms; ++A) {
        const PyscfLevel3AtomPlan& plan = plans[static_cast<std::size_t>(A)];
        const int A_part = becke_index_of[static_cast<std::size_t>(A)];
        const std::size_t atom_base = atom_offsets[static_cast<std::size_t>(A)];
        std::size_t local_out = 0;
        std::size_t tier_begin = 0;
        while (tier_begin < plan.shell_order.size()) {
            const std::size_t first_shell = static_cast<std::size_t>(
                plan.shell_order[tier_begin]);
            const LebedevTier* tier = plan.tier_by_shell[first_shell];
            std::size_t tier_end = tier_begin + 1;
            while (tier_end < plan.shell_order.size()) {
                const std::size_t shell = static_cast<std::size_t>(
                    plan.shell_order[tier_end]);
                if (plan.tier_by_shell[shell]->n_points != tier->n_points) {
                    break;
                }
                ++tier_end;
            }

            // PySCF 2.14 forms at most twelve radii at a time, then flattens
            // ``einsum("i,jk->jik", radii, angular_points)``. Consequently
            // the angular point is the outer index and the radial shell is
            // the inner index inside each batch. This ordering is observable
            // to float32 nonlocal models through their reduction order.
            for (std::size_t batch_begin = tier_begin;
                 batch_begin < tier_end; batch_begin += 12) {
                const std::size_t batch_end = std::min(
                    batch_begin + 12, tier_end);
                const std::uint16_t* angular_order =
                    pyscf_level3_angular_order(tier->n_points);
                for (std::size_t a = 0; a < tier->n_points; ++a) {
                    const double* row =
                        tier->data + 4 * angular_order[a];
                    const Eigen::Vector3d direction(row[0], row[1], row[2]);
                    for (std::size_t position = batch_begin;
                         position < batch_end; ++position) {
                        const std::size_t shell = static_cast<std::size_t>(
                            plan.shell_order[position]);
                        const double radius = plan.radial.r[shell];
                        const double radial_weight = plan.radial.w[shell];
                        const Eigen::Vector3d point =
                            grid_atom_pos[static_cast<std::size_t>(A)]
                            + radius * direction;
                        const double atomic_weight = radial_weight * row[3];
                        double partition_weight = 1.0;
                        if (partition_atom_pos.size() > 1) {
                            const auto cell_weights = pyscf_becke_cell_weights(
                                point, partition_atom_pos,
                                partition_atom_numbers);
                            partition_weight = cell_weights[
                                static_cast<std::size_t>(A_part)];
                        }
                        const std::size_t out = atom_base + local_out++;
                        g.points.row(static_cast<Eigen::Index>(out)) =
                            point.transpose();
                        g.weights(static_cast<Eigen::Index>(out)) =
                            atomic_weight * partition_weight;
                        g.atomic_weights(static_cast<Eigen::Index>(out)) =
                            atomic_weight;
                        g.atom_of_point[out] = A;
                    }
                }
            }
            tier_begin = tier_end;
        }
    }
    return g;
}

Grid build_grid_core(
    const Molecule& grid_mol,
    const std::vector<Eigen::Vector3d>& grid_atom_pos,
    const std::vector<Eigen::Vector3d>& partition_atom_pos,
    const std::vector<int>& partition_atom_numbers,
    const std::vector<int>& becke_index_of,
    const GridOptions& opts) {
    if (opts.atomic_grid_profile == AtomicGridProfile::PySCFLevel3) {
        return build_pyscf_level3_grid_core(
            grid_mol, grid_atom_pos, partition_atom_pos,
            partition_atom_numbers, becke_index_of);
    }
    if (opts.n_radial < 1 || opts.n_theta < 1 || opts.n_phi < 1) {
        throw std::invalid_argument("grid: resolution parameters must be >= 1");
    }
    if (opts.becke_k < 1) {
        throw std::invalid_argument("grid: becke_k must be >= 1");
    }

    const int N_grid_atoms = static_cast<int>(grid_atom_pos.size());
    const std::size_t N_partition = partition_atom_pos.size();

    // Pre-compute the angular grid(s). Two angular schemes — both
    // produce ``(ang_dir, ang_w)`` with the convention ``Σ w = 4π``
    // (full-sphere area) so the radial × angular product integrates
    // over R³ in (r²-Jacobian-included) Treutler-Ahlrichs units.
    //
    // When ``opts.angular_pruning == NWChem``, the Lebedev path
    // pre-builds up to three angular tiers (core / middle / tail)
    // and selects per radial shell — see ``select_pruned_lebedev_order``
    // below for the per-shell choice. The product-grid path doesn't
    // support pruning yet (kept opt-in only on Lebedev for the
    // first landing).
    std::vector<Eigen::Vector3d> ang_dir;
    std::vector<double> ang_w;
    int n_ang = 0;

    // Auxiliary tier arrays (only populated when pruning is on).
    std::vector<Eigen::Vector3d> ang_dir_inner;
    std::vector<double> ang_w_inner;
    int n_ang_inner = 0;
    bool have_pruning = false;

    // 2021 ORCA-style 5-region AngularGrid (``opts.orca_angular_points``).
    // When active, these 5 per-region tiers and the per-shell region map
    // OVERRIDE the 2-tier ``have_pruning`` scheme above. ``region_bounds``
    // are shell-index boundaries (b in 0..3) derived from
    // ``kOrcaRegionFractions``.
    std::array<std::vector<Eigen::Vector3d>, 5> region_dir;
    std::array<std::vector<double>, 5> region_w;
    std::array<int, 5> region_npts = {0, 0, 0, 0, 0};
    std::array<int, 4> region_bounds = {0, 0, 0, 0};
    bool have_orca_regions = false;

    if (opts.angular == AngularScheme::Lebedev) {
        // Lebedev-Laikov rule. The static dispatch table holds every
        // bundled tier (orders 5..53); we look up by exact ``order``.
        // Higher orders integrate higher-degree spherical harmonics
        // exactly — see ``cpp/include/vibeqc/lebedev_data.hpp``.
        const auto* tiers = lebedev_tiers();
        const std::size_t n_tiers = lebedev_tier_count();
        const LebedevTier* match = nullptr;
        for (std::size_t i = 0; i < n_tiers; ++i) {
            if (tiers[i].order == opts.lebedev_order) {
                match = &tiers[i];
                break;
            }
        }
        if (match == nullptr) {
            std::string available;
            for (std::size_t i = 0; i < n_tiers; ++i) {
                if (i > 0) available += ", ";
                available += std::to_string(tiers[i].order);
            }
            throw std::invalid_argument(
                "grid: unsupported Lebedev order "
                + std::to_string(opts.lebedev_order)
                + " (bundled tiers: " + available + ")");
        }
        n_ang = static_cast<int>(match->n_points);
        ang_dir.resize(n_ang);
        ang_w.resize(n_ang);
        for (int k = 0; k < n_ang; ++k) {
            const double* row = match->data + 4 * k;
            ang_dir[k] = Eigen::Vector3d(row[0], row[1], row[2]);
            ang_w[k]   = row[3];   // already normalised so Σ w = 4π
        }

        // NWChem-style 3-tier pruning: choose a smaller "inner" tier
        // (used at the innermost ~25 % and outermost ~10 % of radial
        // shells; full order in between). The inner-tier order is the
        // next-lower tier on the historical generic ladder, floored at 11.
        if (opts.angular_pruning == AngularPruning::NWChem) {
            // Pick the next-lower historical generic order strictly below
            // ``opts.lebedev_order`` (e.g. 29 -> 23, 35 -> 29, 41 -> 35).
            // Profile-only tiers must not perturb generic-grid behaviour.
            // If the requested order is already the smallest generic tier,
            // pruning is a no-op for this configuration.
            int chosen_inner = 0;
            for (std::size_t i = 0; i < n_tiers; ++i) {
                if (tiers[i].order < opts.lebedev_order
                    && is_generic_pruning_tier(tiers[i].order)) {
                    if (tiers[i].order > chosen_inner) {
                        chosen_inner = tiers[i].order;
                    }
                }
            }
            // Step down by ~one tier; standard NWChem rule for
            // mid-range orders. For requested order 29 (302 pts)
            // this gives inner-tier 23 (194 pts) — about half the
            // points in pruned regions.
            if (chosen_inner == 0) {
                // No bundled order is smaller — disable pruning.
                have_pruning = false;
            } else {
                const LebedevTier* inner_match = nullptr;
                for (std::size_t i = 0; i < n_tiers; ++i) {
                    if (tiers[i].order == chosen_inner) {
                        inner_match = &tiers[i];
                        break;
                    }
                }
                if (inner_match != nullptr) {
                    n_ang_inner = static_cast<int>(inner_match->n_points);
                    ang_dir_inner.resize(n_ang_inner);
                    ang_w_inner.resize(n_ang_inner);
                    for (int k = 0; k < n_ang_inner; ++k) {
                        const double* row = inner_match->data + 4 * k;
                        ang_dir_inner[k] =
                            Eigen::Vector3d(row[0], row[1], row[2]);
                        ang_w_inner[k] = row[3];
                    }
                    have_pruning = true;
                }
            }
        }

        // 2021 ORCA-style 5-region AngularGrid. When the caller supplies
        // exactly 5 point counts (``opts.orca_angular_points``), build one
        // Lebedev tier per region plus the shell-index region boundaries;
        // this supersedes the 2-tier ``have_pruning`` scheme above
        // (Helmich-Paris, de Souza, Neese & Izsák 2021, Table I).
        if (opts.orca_angular_points.size() == 5) {
            for (int b = 0; b < 5; ++b) {
                const auto bi = static_cast<std::size_t>(b);
                const LebedevTier* t =
                    lebedev_tier_for_points(opts.orca_angular_points[bi]);
                const int np = static_cast<int>(t->n_points);
                region_npts[bi] = np;
                region_dir[bi].resize(np);
                region_w[bi].resize(np);
                for (int k = 0; k < np; ++k) {
                    const double* row = t->data + 4 * k;
                    region_dir[bi][k] =
                        Eigen::Vector3d(row[0], row[1], row[2]);
                    region_w[bi][k] = row[3];
                }
            }
            // Shell-index region boundaries from the fixed fractions,
            // forced non-decreasing and within [0, n_radial] so every
            // region is well-defined even for very small radial counts.
            int prev = 0;
            for (int b = 0; b < 4; ++b) {
                int bound = static_cast<int>(std::lround(
                    kOrcaRegionFractions[static_cast<std::size_t>(b)]
                    * opts.n_radial));
                bound = std::clamp(bound, prev, opts.n_radial);
                region_bounds[static_cast<std::size_t>(b)] = bound;
                prev = bound;
            }
            have_orca_regions = true;
            have_pruning = false;  // 5-region scheme replaces 2-tier pruning
        }
    } else {
        // Legacy product grid:
        //   ∫ dΩ f(θ, φ) = ∫_{-1}^{1} d(cos θ) ∫_0^{2π} dφ f
        // Gauss-Legendre in x = cos θ, uniform in φ. Kept for parity-
        // matrix bit-reproducibility — superseded by Lebedev for any
        // new code path.
        GLQuadrature gl = gauss_legendre(opts.n_theta);
        const double dphi = 2.0 * kPi / opts.n_phi;
        n_ang = opts.n_theta * opts.n_phi;
        ang_dir.resize(n_ang);
        ang_w.resize(n_ang);
        for (int j = 0; j < opts.n_theta; ++j) {
            const double cos_t = gl.nodes(j);
            const double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t * cos_t));
            const double wj = gl.weights(j);
            for (int m = 0; m < opts.n_phi; ++m) {
                const double phi = m * dphi;
                const int idx = j * opts.n_phi + m;
                ang_dir[idx] = Eigen::Vector3d(sin_t * std::cos(phi),
                                               sin_t * std::sin(phi),
                                               cos_t);
                ang_w[idx] = wj * dphi;
            }
        }
    }

    // Per-radial-shell pruning boundaries (only used when
    // ``have_pruning`` is true). NWChem 3-tier scheme:
    //   inner core (low-ℓ enough — ρ nearly spherical):
    //       shell index < ``prune_inner_end``  (~25 % of n_radial)
    //   middle (full requested order):
    //       prune_inner_end ≤ shell index < prune_outer_start
    //   outer tail (low-ℓ enough — ρ smooth):
    //       shell index ≥ prune_outer_start  (~15 % of n_radial)
    const int prune_inner_end = have_pruning
        ? static_cast<int>(opts.n_radial / 4)
        : 0;
    const int prune_outer_start = have_pruning
        ? opts.n_radial - static_cast<int>(opts.n_radial * 15 / 100)
        : opts.n_radial;

    // Per-shell-index angular-tier selector. Returns the n_ang for
    // shell ``k`` (used both for total-points accumulation and for
    // looking up which ``ang_dir`` / ``ang_w`` to read inside the
    // radial loop).
    auto shell_is_pruned = [&](int k) -> bool {
        return have_pruning
            && (k < prune_inner_end || k >= prune_outer_start);
    };
    // 5-region selector (2021 AngularGrid). Maps a radial shell index to
    // its region 0..4 via the shell-index boundaries; used only when
    // ``have_orca_regions`` is set, in which case it supersedes the
    // 2-tier ``shell_is_pruned`` choice.
    auto region_of_shell = [&](int k) -> std::size_t {
        if (k < region_bounds[0]) return 0;
        if (k < region_bounds[1]) return 1;
        if (k < region_bounds[2]) return 2;
        if (k < region_bounds[3]) return 3;
        return 4;
    };
    auto angular_count_for_shell = [&](int k) -> int {
        if (have_orca_regions) return region_npts[region_of_shell(k)];
        return shell_is_pruned(k) ? n_ang_inner : n_ang;
    };

    // Total point count: when pruning is on, this varies per shell.
    std::size_t pts_per_atom = 0;
    for (int k = 0; k < opts.n_radial; ++k) {
        pts_per_atom += static_cast<std::size_t>(
            angular_count_for_shell(k));
    }
    const std::size_t total_pts =
        static_cast<std::size_t>(N_grid_atoms) * pts_per_atom;

    Grid g;
    g.atomic_grid_profile = opts.atomic_grid_profile;
    g.points.resize(total_pts, 3);
    g.weights.resize(total_pts);
    g.atomic_weights.resize(total_pts);
    g.atom_of_point.resize(total_pts);
    g.atom_coords.resize(N_grid_atoms, 3);
    g.atomic_numbers.resize(static_cast<std::size_t>(N_grid_atoms));
    for (int A = 0; A < N_grid_atoms; ++A) {
        g.atom_coords.row(A) = grid_atom_pos[A].transpose();
        g.atomic_numbers[static_cast<std::size_t>(A)] =
            grid_mol.atoms()[static_cast<std::size_t>(A)].Z;
    }

    // Per-shell offset within one atom's contiguous output block (prefix sum
    // of the angular counts), so each grid point's output row is a pure
    // function of (A, k, a):  out = A*pts_per_atom + shell_offset[k] + a.
    // Removing the sequential ``out++`` lets the per-atom loop run in
    // parallel — each atom owns a disjoint output block, so the result is
    // BIT-IDENTICAL to the old serial layout (same point at the same row).
    // The per-point Becke/Stratmann partition is O(N_partition^2) and
    // dominates for image-rich periodic grids (slabs), which is where this
    // pays off. Dynamic schedule: the Stratmann early-exit + angular pruning
    // make per-atom cost mildly uneven.
    std::vector<std::size_t> shell_offset(
        static_cast<std::size_t>(opts.n_radial));
    {
        std::size_t acc = 0;
        for (int k = 0; k < opts.n_radial; ++k) {
            shell_offset[static_cast<std::size_t>(k)] = acc;
            acc += static_cast<std::size_t>(angular_count_for_shell(k));
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int A = 0; A < N_grid_atoms; ++A) {
        const RadialGrid rad =
            treutler_ahlrichs(opts.n_radial, treutler_xi(grid_mol.atoms()[A].Z));
        const int A_part = becke_index_of[A];
        const std::size_t atom_base =
            static_cast<std::size_t>(A) * pts_per_atom;

        for (int k = 0; k < opts.n_radial; ++k) {
            const double r_k = rad.r[k];
            const double w_r = rad.w[k];
            // Per-shell angular-grid selection. With the 2021 5-region
            // scheme active, the region map picks one of 5 tiers; else the
            // 2-tier scheme (pruned shells use the smaller inner tier, the
            // middle band uses the full tier).
            const std::size_t reg =
                have_orca_regions ? region_of_shell(k) : 0;
            const std::vector<Eigen::Vector3d>& ang_dir_shell =
                have_orca_regions ? region_dir[reg]
                                  : (shell_is_pruned(k) ? ang_dir_inner
                                                        : ang_dir);
            const std::vector<double>& ang_w_shell =
                have_orca_regions ? region_w[reg]
                                  : (shell_is_pruned(k) ? ang_w_inner
                                                        : ang_w);
            const int n_ang_shell =
                have_orca_regions ? region_npts[reg]
                                  : (shell_is_pruned(k) ? n_ang_inner
                                                        : n_ang);
            const std::size_t shell_base =
                atom_base + shell_offset[static_cast<std::size_t>(k)];
            for (int a = 0; a < n_ang_shell; ++a) {
                const Eigen::Vector3d point =
                    grid_atom_pos[A] + r_k * ang_dir_shell[a];
                const double w_point = w_r * ang_w_shell[a];

                // Atomic-partition weight at this point. Two families:
                //   Becke (1988): O(N²) full iterated polynomial.
                //   Stratmann (1996): O(N · N_nearby) piecewise polynomial
                //     with |μ_AB| ≥ a hard cutoff.
                double becke_w = 1.0;
                if (N_partition > 1) {
                    if (opts.partition == AtomicPartition::Stratmann) {
                        const auto P = stratmann_cell_weights(
                            point, partition_atom_pos);
                        becke_w = P[A_part];
                    } else {
                        const auto P = becke_cell_weights(
                            point, partition_atom_pos, opts.becke_k);
                        becke_w = P[A_part];
                    }
                }

                const std::size_t out =
                    shell_base + static_cast<std::size_t>(a);
                g.points.row(out) = point.transpose();
                g.weights(out) = w_point * becke_w;
                g.atomic_weights(out) = w_point;
                g.atom_of_point[out] = A;   // index into grid_mol
            }
        }
    }
    return g;
}

}  // namespace

Grid build_grid(const Molecule& mol, const GridOptions& opts) {
    const int N_atoms = static_cast<int>(mol.atoms().size());
    std::vector<Eigen::Vector3d> atom_pos(N_atoms);
    std::vector<int> atom_numbers(N_atoms);
    std::vector<int> identity(N_atoms);
    for (int A = 0; A < N_atoms; ++A) {
        const auto& a = mol.atoms()[A];
        atom_pos[A] = Eigen::Vector3d(a.xyz[0], a.xyz[1], a.xyz[2]);
        atom_numbers[A] = a.Z;
        identity[A] = A;
    }
    return build_grid_core(
        mol, atom_pos, atom_pos, atom_numbers, identity, opts);
}

std::vector<int> grid_atomic_point_counts(
    const Molecule& mol, const GridOptions& opts) {
    if (opts.atomic_grid_profile == AtomicGridProfile::PySCFLevel3) {
        const auto plans = make_pyscf_level3_atom_plans(mol);
        std::vector<int> counts;
        counts.reserve(plans.size());
        for (const auto& plan : plans) {
            counts.push_back(static_cast<int>(plan.point_count));
        }
        return counts;
    }

    // The generic path has one shared radial/angular layout for all atoms.
    if (opts.n_radial < 1 || opts.n_theta < 1 || opts.n_phi < 1) {
        throw std::invalid_argument("grid: resolution parameters must be >= 1");
    }
    std::size_t points_per_atom = 0;
    if (opts.angular == AngularScheme::ProductGaussLegendre) {
        points_per_atom = static_cast<std::size_t>(opts.n_radial)
                          * static_cast<std::size_t>(opts.n_theta)
                          * static_cast<std::size_t>(opts.n_phi);
    } else if (opts.orca_angular_points.size() == 5) {
        std::array<int, 4> bounds = {0, 0, 0, 0};
        int previous = 0;
        for (int b = 0; b < 4; ++b) {
            int bound = static_cast<int>(std::lround(
                kOrcaRegionFractions[static_cast<std::size_t>(b)]
                * opts.n_radial));
            bound = std::clamp(bound, previous, opts.n_radial);
            bounds[static_cast<std::size_t>(b)] = bound;
            previous = bound;
        }
        const std::array<int, 5> shell_counts = {
            bounds[0], bounds[1] - bounds[0], bounds[2] - bounds[1],
            bounds[3] - bounds[2], opts.n_radial - bounds[3]};
        for (int b = 0; b < 5; ++b) {
            const LebedevTier* tier = lebedev_tier_for_points(
                opts.orca_angular_points[static_cast<std::size_t>(b)]);
            points_per_atom += static_cast<std::size_t>(
                shell_counts[static_cast<std::size_t>(b)]) * tier->n_points;
        }
    } else {
        const auto* tiers = lebedev_tiers();
        const std::size_t n_tiers = lebedev_tier_count();
        const LebedevTier* full = nullptr;
        const LebedevTier* inner = nullptr;
        for (std::size_t i = 0; i < n_tiers; ++i) {
            if (tiers[i].order == opts.lebedev_order) full = &tiers[i];
            if (tiers[i].order < opts.lebedev_order
                && is_generic_pruning_tier(tiers[i].order)
                && (inner == nullptr || tiers[i].order > inner->order)) {
                inner = &tiers[i];
            }
        }
        if (full == nullptr) {
            throw std::invalid_argument(
                "grid: unsupported Lebedev order "
                + std::to_string(opts.lebedev_order));
        }
        if (opts.angular_pruning != AngularPruning::NWChem
            || inner == nullptr) {
            points_per_atom = static_cast<std::size_t>(opts.n_radial)
                              * full->n_points;
        } else {
            const int inner_end = opts.n_radial / 4;
            const int outer_start =
                opts.n_radial - opts.n_radial * 15 / 100;
            const int n_pruned = inner_end + opts.n_radial - outer_start;
            points_per_atom = static_cast<std::size_t>(n_pruned)
                                  * inner->n_points
                              + static_cast<std::size_t>(opts.n_radial - n_pruned)
                                  * full->n_points;
        }
    }
    if (points_per_atom > static_cast<std::size_t>(
                              std::numeric_limits<int>::max())) {
        throw std::overflow_error("grid: per-atom point count exceeds int range");
    }
    return std::vector<int>(
        mol.atoms().size(), static_cast<int>(points_per_atom));
}

Grid build_grid_periodic(
    const Molecule& grid_mol,
    const std::vector<Eigen::Vector3d>& partition_atom_positions,
    const GridOptions& opts) {
    return build_grid_periodic(
        grid_mol, partition_atom_positions, std::vector<int>{}, opts);
}

Grid build_grid_periodic(
    const Molecule& grid_mol,
    const std::vector<Eigen::Vector3d>& partition_atom_positions,
    const std::vector<int>& partition_atomic_numbers,
    const GridOptions& opts) {
    const int N_grid = static_cast<int>(grid_mol.atoms().size());
    if (static_cast<int>(partition_atom_positions.size()) < N_grid) {
        throw std::invalid_argument(
            "build_grid_periodic: partition_atom_positions must include at "
            "least every grid_mol atom (home cell first, then images).");
    }
    if (!partition_atomic_numbers.empty()
        && partition_atomic_numbers.size() != partition_atom_positions.size()) {
        throw std::invalid_argument(
            "build_grid_periodic: partition_atomic_numbers must have one "
            "entry per partition_atom_positions entry.");
    }
    if (opts.atomic_grid_profile == AtomicGridProfile::PySCFLevel3
        && partition_atomic_numbers.size() != partition_atom_positions.size()) {
        throw std::invalid_argument(
            "build_grid_periodic: the PySCF level-3 profile requires "
            "partition_atomic_numbers for every home and image atom.");
    }
    std::vector<Eigen::Vector3d> grid_atom_pos(N_grid);
    std::vector<int> becke_index_of(N_grid);
    for (int A = 0; A < N_grid; ++A) {
        const auto& a = grid_mol.atoms()[A];
        grid_atom_pos[A] = Eigen::Vector3d(a.xyz[0], a.xyz[1], a.xyz[2]);
        // Convention: the first N_grid entries of partition_atom_positions
        // must coincide with the grid_mol atoms (home cell).
        const auto& p = partition_atom_positions[A];
        if ((p - grid_atom_pos[A]).norm() > 1e-10) {
            throw std::invalid_argument(
                "build_grid_periodic: the first N_grid entries of "
                "partition_atom_positions must equal the grid_mol atom "
                "positions in order (home cell first).");
        }
        if (!partition_atomic_numbers.empty()
            && partition_atomic_numbers[static_cast<std::size_t>(A)] != a.Z) {
            throw std::invalid_argument(
                "build_grid_periodic: the first N_grid entries of "
                "partition_atomic_numbers must equal the grid_mol atomic "
                "numbers in order (home cell first).");
        }
        becke_index_of[A] = A;
    }
    return build_grid_core(
        grid_mol, grid_atom_pos, partition_atom_positions,
        partition_atomic_numbers, becke_index_of, opts);
}

Grid build_periodic_point_grid(const PeriodicSystem& system,
                               double image_radius,
                               const GridOptions& opts) {
    // Becke (1988), doi:10.1063/1.454033, Eqs. 13 and 22: normalize
    // products of pair cell functions over the same atoms at each point.
    // Our periodic extension selects that set by physical distance from
    // the point, independently of the owning atom's lattice image label.
    if (!std::isfinite(image_radius) || !std::isfinite(4*image_radius*image_radius)
        || image_radius < 0.0)
        throw std::invalid_argument("Periodic partition radius must be finite and nonnegative");
    const Molecule molecule = system.unit_cell_molecule();
    if (image_radius == 0.0) return build_grid(molecule, opts);
    if (system.dim < 1 || system.dim > 3 || !system.lattice.allFinite())
        throw std::invalid_argument("Periodic partition requires a finite lattice in dimension 1..3");
    const Eigen::MatrixXd lattice = system.lattice.leftCols(system.dim);
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        lattice, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.rank() != system.dim)
        throw std::invalid_argument("Periodic partition requires independent active lattice vectors");
    const Eigen::MatrixXd inverse = svd.matrixV()
        * svd.singularValues().cwiseInverse().asDiagonal()
        * svd.matrixU().transpose();
    std::vector<Eigen::Vector3d> origins;
    std::vector<int> numbers;
    for (const auto& atom : system.unit_cell) {
        const Eigen::Vector3d origin(atom.xyz[0], atom.xyz[1], atom.xyz[2]);
        if (!origin.allFinite())
            throw std::invalid_argument("Periodic partition requires finite atomic positions");
        origins.push_back(origin);
        numbers.push_back(atom.Z);
    }
    for (std::size_t atom = 0; atom < origins.size(); ++atom) {
        for (std::size_t other = 0; other < atom; ++other) {
            const Eigen::Vector3d delta = origins[atom]-origins[other];
            const Eigen::VectorXd image = (inverse*delta).array().round();
            if ((delta-lattice*image).norm() <= 1e-12)
                throw std::invalid_argument("Coincident periodic atoms are invalid in a grid partition");
        }
    }

    // Generate the unchanged atom-major quadrature and raw weights first.
    // An empty partition list makes this a raw atomic-grid construction.
    Grid grid = build_grid_core(molecule, origins, {}, {},
                                 std::vector<int>(origins.size(), 0), opts);
    const std::vector<int> counts = grid_atomic_point_counts(molecule, opts);
    std::size_t begin = 0;
    std::atomic<bool> invalid_images{false};
    for (std::size_t owner = 0; owner < origins.size(); ++owner) {
        std::vector<Eigen::Vector3d> offsets;
        for (const auto& origin : origins) offsets.push_back(origin-origins[owner]);
        std::vector<Eigen::Vector3d> atlas;
        std::vector<int> atlas_numbers;
        std::size_t owner_image = 0;
        const double atlas_radius = 2*image_radius;
        for (std::size_t atom = 0; atom < origins.size(); ++atom) {
            const Eigen::Vector3d offset = origins[atom]-origins[owner];
            const Eigen::VectorXd fractional = -inverse*offset;
            std::array<int, 3> lower{0, 0, 0}, upper{0, 0, 0};
            for (int axis = 0; axis < system.dim; ++axis) {
                const double reach = atlas_radius*inverse.row(axis).norm();
                const double slack = 1e-10*(1+std::abs(fractional[axis])+reach);
                const double lo = std::floor(fractional[axis]-reach-slack);
                const double hi = std::ceil(fractional[axis]+reach+slack);
                constexpr int limit = std::numeric_limits<int>::max()/4;
                if (!std::isfinite(lo) || !std::isfinite(hi) || lo < -limit || hi > limit)
                    throw std::invalid_argument("Periodic partition image index exceeds the supported range");
                lower[axis] = static_cast<int>(lo);
                upper[axis] = static_cast<int>(hi);
            }
            for (int i = lower[0]; i <= upper[0]; ++i)
            for (int j = lower[1]; j <= upper[1]; ++j)
            for (int k = lower[2]; k <= upper[2]; ++k) {
                const Eigen::Vector3d relative = offset
                    + i*system.lattice.col(0) + j*system.lattice.col(1)
                    + k*system.lattice.col(2);
                const double limit = atlas_radius+1e-12*(1+atlas_radius);
                if (relative.squaredNorm() > limit*limit) continue;
                const bool is_owner = atom == owner && i == 0 && j == 0 && k == 0;
                if (!is_owner && relative.norm() <= 1e-12)
                    throw std::invalid_argument("Coincident periodic atoms are invalid in a grid partition");
                if (is_owner) owner_image = atlas.size();
                atlas.push_back(relative);
                atlas_numbers.push_back(numbers[atom]);
            }
        }
        const auto end = begin+static_cast<std::size_t>(counts[owner]);
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t index = static_cast<std::ptrdiff_t>(begin);
             index < static_cast<std::ptrdiff_t>(end); ++index) {
            const Eigen::Vector3d point = grid.points.row(index).transpose()-origins[owner];
            // Most atomic points lie inside R/2. Then the nearest image
            // is at most R/2 away and the effective radius is exactly R;
            // the precomputed 2R atlas contains the whole neighborhood.
            if (point.squaredNorm() <= image_radius*image_radius/4) {
                std::vector<Eigen::Vector3d> neighbors;
                std::vector<int> neighbor_numbers;
                std::size_t selected_owner = 0;
                for (std::size_t image = 0; image < atlas.size(); ++image) {
                    if ((point-atlas[image]).squaredNorm() > image_radius*image_radius) continue;
                    if (image == owner_image) selected_owner = neighbors.size();
                    neighbors.push_back(atlas[image]);
                    neighbor_numbers.push_back(atlas_numbers[image]);
                }
                std::vector<double> weights;
                if (opts.atomic_grid_profile == AtomicGridProfile::PySCFLevel3)
                    weights = pyscf_becke_cell_weights(point, neighbors, neighbor_numbers);
                else if (opts.partition == AtomicPartition::Stratmann)
                    weights = stratmann_cell_weights(point, neighbors);
                else
                    weights = becke_cell_weights(point, neighbors, opts.becke_k);
                grid.weights[index] = grid.atomic_weights[index]*weights[selected_owner];
                continue;
            }
            bool valid = point.allFinite();
            constexpr int image_limit = std::numeric_limits<int>::max()/4;
            // Rounded fractional coordinates supply an upper bound, not
            // a nearest-image assumption on a skew lattice.
            double nearest2 = point.squaredNorm();
            for (const auto& offset : offsets) {
                const Eigen::VectorXd fractional = inverse*(point-offset);
                Eigen::Vector3d image = Eigen::Vector3d::Zero();
                for (int axis = 0; axis < system.dim; ++axis) {
                    if (!std::isfinite(fractional[axis]) ||
                        std::abs(fractional[axis]) >= image_limit) {
                        valid = false;
                        break;
                    }
                    image += std::round(fractional[axis])*system.lattice.col(axis);
                }
                nearest2 = std::min(nearest2, (point-offset-image).squaredNorm());
            }
            const auto visit_images = [&](double radius, auto&& visit) {
                if (!valid || !std::isfinite(radius*radius)) return false;
                for (std::size_t atom = 0; atom < offsets.size(); ++atom) {
                    const Eigen::Vector3d delta = point-offsets[atom];
                    const Eigen::VectorXd fractional = inverse*delta;
                    const double perpendicular2 = (delta-lattice*fractional).squaredNorm();
                    const double slack = 1e-12*(1+radius);
                    const double enclosure = radius+slack;
                    if (perpendicular2 > enclosure*enclosure) continue;
                    const double active_radius = std::sqrt(std::max(
                        0.0, enclosure*enclosure-perpendicular2));
                    std::array<int, 3> lower{0, 0, 0}, upper{0, 0, 0};
                    for (int axis = 0; axis < system.dim; ++axis) {
                        const double reach = active_radius*inverse.row(axis).norm();
                        const double padding = 1e-10*(1+std::abs(fractional[axis])+reach);
                        const double lo = std::floor(fractional[axis]-reach-padding);
                        const double hi = std::ceil(fractional[axis]+reach+padding);
                        if (!std::isfinite(lo) || !std::isfinite(hi) ||
                            lo < -image_limit || hi > image_limit) return false;
                        lower[axis] = static_cast<int>(lo);
                        upper[axis] = static_cast<int>(hi);
                    }
                    for (int i = lower[0]; i <= upper[0]; ++i)
                    for (int j = lower[1]; j <= upper[1]; ++j)
                    for (int k = lower[2]; k <= upper[2]; ++k) {
                        const Eigen::Vector3d center = offsets[atom]
                            + i*system.lattice.col(0) + j*system.lattice.col(1)
                            + k*system.lattice.col(2);
                        const double distance2 = (point-center).squaredNorm();
                        if (distance2 > enclosure*enclosure) continue;
                        visit(center, atom, i == 0 && j == 0 && k == 0,
                              distance2);
                    }
                }
                return true;
            };
            valid = valid && visit_images(std::sqrt(nearest2),
                [&](const Eigen::Vector3d&, std::size_t, bool, double distance2) {
                    nearest2 = std::min(nearest2, distance2);
                });
            // The neighborhood must cover every physical point, including
            // diffuse density in vacuum. Its minimum radius is the user
            // cutoff; the nearest-image distance expands it continuously
            // where a fixed-radius domain would leave uncovered regions.
            const double radius = std::max(image_radius, 2*std::sqrt(nearest2));
            if (!valid) {
                invalid_images.store(true, std::memory_order_relaxed);
                grid.weights[index] = 0.0;
                continue;
            }
            if (point.squaredNorm() > radius*radius) {
                grid.weights[index] = 0.0;
                continue;
            }
            std::vector<Eigen::Vector3d> neighbors;
            std::vector<int> neighbor_numbers;
            std::size_t selected_owner = std::numeric_limits<std::size_t>::max();
            valid = visit_images(radius,
                [&](const Eigen::Vector3d& center, std::size_t atom,
                    bool home, double distance2) {
                    if (distance2 > radius*radius) return;
                    if (atom == owner && home) selected_owner = neighbors.size();
                    neighbors.push_back(center);
                    neighbor_numbers.push_back(numbers[atom]);
                });
            if (!valid || selected_owner == std::numeric_limits<std::size_t>::max()) {
                invalid_images.store(true, std::memory_order_relaxed);
                grid.weights[index] = 0.0;
                continue;
            }
            std::vector<double> weights;
            if (opts.atomic_grid_profile == AtomicGridProfile::PySCFLevel3)
                weights = pyscf_becke_cell_weights(point, neighbors, neighbor_numbers);
            else if (opts.partition == AtomicPartition::Stratmann)
                weights = stratmann_cell_weights(point, neighbors);
            else
                weights = becke_cell_weights(point, neighbors, opts.becke_k);
            grid.weights[index] = grid.atomic_weights[index]*weights[selected_owner];
        }
        begin = end;
    }
    if (invalid_images.load(std::memory_order_relaxed))
        throw std::invalid_argument("Periodic partition image index exceeds the supported range");
    return grid;
}

}  // namespace vibeqc
