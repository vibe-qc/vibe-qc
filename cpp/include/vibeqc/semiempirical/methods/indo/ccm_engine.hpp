// Periodic Cyclic Cluster Model (CCM) engine — faithful C++ port.
//
// Port of the validated Python reference in
// python/vibeqc/semiempirical/methods/msindo_ccm.py (oracle parity).
// Reuses the molecular SCF driver (scf_rhf_driver) from indo_engine.hpp
// and the STO kernel from msindo_integrals.hpp.
//
// Scope: closed-shell RHF (s/p/d), 1D/2D/3D, with optional Madelung
// embedding (1-D finite lattice sum + 2-D/3-D Ewald).
//
// © Mulliken Center for Theoretical Chemistry, University of Bonn (method);
// independent vibe-qc re-implementation.

#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/semiempirical/seccm/topology.hpp"
#include "vibeqc/semiempirical/methods/indo/indo_engine.hpp"
#include "vibeqc/semiempirical/methods/indo/msindo_gradient.hpp"

namespace vibeqc {
namespace semiempirical {
namespace indo {

using WSNeighbor = seccm::WSImage;
using WignerSeitzCells = seccm::WSTopology;
using seccm::build_wigner_seitz;

inline void validate_ccm_native_inputs(
    const std::vector<int>& Z,
    const std::vector<std::array<double, 3>>& coords_angstrom,
    const std::vector<std::array<double, 3>>& translations_angstrom,
    int max_iter,
    double conv_tol) {
    if (Z.empty() || coords_angstrom.size() != Z.size()) {
        throw std::invalid_argument(
            "MSINDO CCM atomic numbers and coordinates must have the same "
            "nonzero length");
    }
    if (translations_angstrom.empty()
        || translations_angstrom.size() > 3) {
        throw std::invalid_argument(
            "MSINDO CCM requires one to three cyclic translations");
    }
    if (max_iter < 1) {
        throw std::invalid_argument(
            "MSINDO CCM max_iter must be positive");
    }
    if (!std::isfinite(conv_tol) || conv_tol <= 0.0) {
        throw std::invalid_argument(
            "MSINDO CCM conv_tol must be positive and finite");
    }

    std::vector<Eigen::Vector3d> coordinates;
    coordinates.reserve(coords_angstrom.size());
    for (const auto& coordinate : coords_angstrom) {
        const Eigen::Vector3d vector(
            coordinate[0], coordinate[1], coordinate[2]);
        if (!vector.allFinite()) {
            throw std::invalid_argument(
                "MSINDO CCM atomic coordinates must be finite");
        }
        coordinates.push_back(vector);
    }
    for (std::size_t left = 0; left < coordinates.size(); ++left) {
        for (std::size_t right = left + 1; right < coordinates.size(); ++right) {
            if ((coordinates[left] - coordinates[right]).norm() <= 1.0e-12) {
                throw std::invalid_argument(
                    "MSINDO CCM does not accept coincident atoms");
            }
        }
    }

    Eigen::MatrixXd translation_matrix(
        static_cast<Eigen::Index>(translations_angstrom.size()), 3);
    for (std::size_t axis = 0; axis < translations_angstrom.size(); ++axis) {
        const auto& translation = translations_angstrom[axis];
        const Eigen::Vector3d vector(
            translation[0], translation[1], translation[2]);
        if (!vector.allFinite()) {
            throw std::invalid_argument(
                "MSINDO CCM cyclic translations must be finite");
        }
        translation_matrix.row(static_cast<Eigen::Index>(axis)) =
            vector.transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(translation_matrix);
    if (svd.rank()
        != static_cast<Eigen::Index>(translations_angstrom.size())) {
        throw std::invalid_argument(
            "MSINDO CCM cyclic translations must be nonzero and linearly "
            "independent");
    }
}

// --------------------------------------------------------------------------- //
// CCM core Hamiltonian + 2-electron integrals (ccmintov.f).                      //
// --------------------------------------------------------------------------- //

// Shell labels for the 9 STO basis functions
static const char _SHELLS_CCM[9] = {'s','p','p','p','d','d','d','d','d'};

inline void build_core_and_gamma_ccm(
    const std::vector<int>& Z,
    const std::vector<Eigen::Vector3d>& C,
    const std::vector<std::array<int, 2>>& blocks,
    int nsto,
    const MsindoParameterSet& p,
    const WignerSeitzCells& ws,
    Eigen::MatrixXd& H,
    Eigen::MatrixXd& G) {
    int natom = static_cast<int>(Z.size());
    H = Eigen::MatrixXd::Zero(nsto, nsto);
    G = Eigen::MatrixXd::Zero(nsto, nsto);

    // One-centre blocks — identical to the molecular engine.
    for (int i = 0; i < natom; ++i) {
        int lo = blocks[i][0], hi = blocks[i][1], z = Z[i];
        auto u = eneg(z, p);
        H(lo, lo) += u[0];
        int nb = hi - lo;
        if (nb >= 4)
            for (int pi = 1; pi < 4; ++pi)
                H(lo + pi, lo + pi) += u[1];
        if (nb >= 9)
            for (int d = 4; d < 9; ++d)
                H(lo + d, lo + d) += u[2];
        G.block(lo, lo, nb, nb) = one_center_gmunu(z, p).topLeftCorner(nb, nb);
    }

    // Two-centre periodic terms — WS-weighted sum over neighbour images.
    for (int k = 0; k < natom; ++k) {
        int lk = blocks[k][0], hk = blocks[k][1], nk = hk - lk;
        for (auto& nb : ws.cells[k]) {
            int j = nb.origin;
            double w = nb.weight;
            int lj = blocks[j][0], hj = blocks[j][1], nj = hj - lj;

            std::array<double, 3> rk_arr = {{C[k].x(), C[k].y(), C[k].z()}};
            Eigen::Vector3d rj = C[k] + nb.disp;
            std::array<double, 3> rj_arr = {{rj.x(), rj.y(), rj.z()}};

            PairBlocks pb = pair_blocks(Z[k], Z[j], rk_arr, rj_arr, p);

            // Core attraction on K from this image.
            H.block(lk, lk, nk, nk) += w * pb.HK1.topLeftCorner(nk, nk);

            if (k < j) {
                double R = pb.R;
                // Resonance + monopole γ
                H.block(lk, lj, nk, nj) += w * pb.HKL2.topLeftCorner(nk, nj);
                H.block(lj, lk, nj, nk) += w * pb.HKL2.topLeftCorner(nk, nj).transpose();
                for (int ia = 0; ia < nk; ++ia)
                    for (int ib = 0; ib < nj; ++ib) {
                        char sa = _SHELLS_CCM[ia], sb = _SHELLS_CCM[ib];
                        bool pa = (sa == 'p'), pbb = (sb == 'p');
                        double g = gamma_shell(Z[k], Z[j], pa, pbb, R, p);
                        G(lk + ia, lj + ib) += w * g;
                        G(lj + ib, lk + ia) += w * g;
                    }
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// Ewald / Madelung embedding.                                                  //
// --------------------------------------------------------------------------- //

// Ewald-WS set: Fock WS neighbours + the atom itself (weight 1.0, zero disp).
inline std::vector<std::vector<WSNeighbor>> _ewald_ws_cells(const WignerSeitzCells& ws) {
    int natom = static_cast<int>(ws.cells.size());
    std::vector<std::vector<WSNeighbor>> ews(natom);
    for (int i = 0; i < natom; ++i) {
        ews[i] = ws.cells[i];
        ews[i].push_back(WSNeighbor{i, 1.0, Eigen::Vector3d::Zero()});
    }
    return ews;
}

// Net atomic charges Z_I − Σ_{μ∈I} P(μ,μ)
inline Eigen::VectorXd _net_charges(const Eigen::MatrixXd& P,
                                     const std::vector<std::array<int, 2>>& blocks,
                                     const std::vector<int>& cz) {
    int natom = static_cast<int>(blocks.size());
    Eigen::VectorXd net(natom);
    for (int i = 0; i < natom; ++i) {
        int lo = blocks[i][0], hi = blocks[i][1];
        net(i) = cz[i] - P.block(lo, lo, hi - lo, hi - lo).trace();
    }
    return net;
}

// 1-D finite Madelung potential (ccm1dmadelsum.f).
inline Eigen::VectorXd _madelung_potential_1d(const Eigen::VectorXd& net_charges,
                                               const std::vector<std::vector<WSNeighbor>>& ews,
                                               const Eigen::Vector3d& T) {
    int natom = static_cast<int>(ews.size());
    Eigen::VectorXd pot = Eigen::VectorXd::Zero(natom);
    std::vector<Eigen::Vector3d> shells = {T, -T, 2.0 * T, -2.0 * T};
    for (int i = 0; i < natom; ++i) {
        double s = 0.0;
        for (auto& nb : ews[i]) {
            double contrib = 0.0;
            for (auto& sh : shells)
                contrib += 1.0 / (nb.disp + sh).norm();
            s += nb.weight * net_charges(nb.origin) * contrib;
        }
        pot(i) = s;
    }
    return pot;
}

// Reciprocal basis vectors + cell volume (3-D).
inline void _reciprocal_basis_3d(const std::vector<Eigen::Vector3d>& translations,
                                  Eigen::Matrix3d& recip, double& vol) {
    Eigen::Vector3d a1 = translations[0], a2 = translations[1], a3 = translations[2];
    double vol_signed = a1.dot(a2.cross(a3));
    vol = std::abs(vol_signed);
    double twopi = 2.0 * M_PI;
    recip.row(0) = twopi * a2.cross(a3) / vol_signed;
    recip.row(1) = twopi * a3.cross(a1) / vol_signed;
    recip.row(2) = twopi * a1.cross(a2) / vol_signed;
}

struct EwaldLattice3D {
    double volume = 0.0;
    double alpha = 0.0;
    double direct_cutoff = 0.0;
    std::vector<Eigen::Vector3d> reciprocal_vectors;
    std::vector<double> reciprocal_squared_norms;
    std::vector<Eigen::Vector3d> direct_vectors;
};

inline std::array<int, 3> _checked_ewald_bounds_3d(
    const std::array<double, 3>& unrounded_bounds,
    const char* space) {
    constexpr std::size_t max_candidate_vectors = 500000;
    std::array<int, 3> bounds = {0, 0, 0};
    std::size_t candidate_vectors = 1;
    for (int axis = 0; axis < 3; ++axis) {
        const double unrounded = unrounded_bounds[axis];
        if (!std::isfinite(unrounded) || unrounded < 0.0) {
            throw std::invalid_argument(
                std::string("3-D CCM Ewald ") + space
                + " bounds are not finite; use a reduced lattice basis");
        }
        const double rounded = std::ceil(unrounded) + 1.0;
        if (rounded > static_cast<double>(max_candidate_vectors)) {
            throw std::invalid_argument(
                std::string("3-D CCM Ewald ") + space
                + " enumeration exceeds the safe candidate budget; use a "
                  "reduced lattice basis");
        }
        bounds[axis] = static_cast<int>(rounded);
        const std::size_t width =
            2 * static_cast<std::size_t>(bounds[axis]) + 1;
        if (width > max_candidate_vectors / candidate_vectors) {
            throw std::invalid_argument(
                std::string("3-D CCM Ewald ") + space
                + " enumeration exceeds the safe candidate budget; use a "
                  "reduced lattice basis");
        }
        candidate_vectors *= width;
    }
    return bounds;
}

// Cutoff-complete direct and reciprocal vector inventories for the 3-D Ewald
// sums.  A fixed coefficient cube is not equivalent to a geometric cutoff: a
// unimodular shear can move a short physical vector to a large, near-cancelling
// coefficient tuple.  The dual-basis bounds below enumerate every vector
// inside the e^-30 Ewald cutoffs, independent of the basis chosen for the same
// lattice.
inline EwaldLattice3D _ewald_lattice_3d(
    const std::vector<Eigen::Vector3d>& translations,
    const std::vector<std::vector<WSNeighbor>>& ews) {
    constexpr double exponent_cutoff = 30.0;
    const double twopi = 2.0 * M_PI;

    Eigen::Matrix3d recip;
    EwaldLattice3D lattice;
    _reciprocal_basis_3d(translations, recip, lattice.volume);
    lattice.alpha = std::sqrt(M_PI) / std::cbrt(lattice.volume);

    const double reciprocal_cutoff =
        lattice.alpha * std::sqrt(4.0 * exponent_cutoff);
    const double reciprocal_cutoff_squared =
        reciprocal_cutoff * reciprocal_cutoff;
    const std::array<int, 3> reciprocal_bounds = _checked_ewald_bounds_3d({
        reciprocal_cutoff * translations[0].norm() / twopi,
        reciprocal_cutoff * translations[1].norm() / twopi,
        reciprocal_cutoff * translations[2].norm() / twopi,
    }, "reciprocal-space");
    for (int n1 = -reciprocal_bounds[0];
         n1 <= reciprocal_bounds[0]; ++n1) {
        for (int n2 = -reciprocal_bounds[1];
             n2 <= reciprocal_bounds[1]; ++n2) {
            for (int n3 = -reciprocal_bounds[2];
                 n3 <= reciprocal_bounds[2]; ++n3) {
                if (n1 == 0 && n2 == 0 && n3 == 0) continue;
                Eigen::Vector3d vector =
                    n1 * recip.row(0).transpose()
                    + n2 * recip.row(1).transpose()
                    + n3 * recip.row(2).transpose();
                const double squared_norm = vector.squaredNorm();
                if (squared_norm > reciprocal_cutoff_squared) continue;
                lattice.reciprocal_vectors.push_back(vector);
                lattice.reciprocal_squared_norms.push_back(squared_norm);
            }
        }
    }

    // For r = v + sum_i n_i a_i and |r| <= r_cut,
    // |n_i| <= |b_i| r_cut/(2 pi) + |b_i.v|/(2 pi).  Include the largest
    // fractional displacement in the Ewald-WS inventory so that translated
    // atom pairs cannot require a coefficient outside the bare-origin box.
    lattice.direct_cutoff = std::sqrt(exponent_cutoff) / lattice.alpha;
    double max_fractional_displacement[3] = {0.0, 0.0, 0.0};
    for (const auto& cell : ews) {
        for (const auto& neighbor : cell) {
            for (int axis = 0; axis < 3; ++axis) {
                const double component = std::abs(
                    recip.row(axis).transpose().dot(neighbor.disp) / twopi);
                if (component > max_fractional_displacement[axis]) {
                    max_fractional_displacement[axis] = component;
                }
            }
        }
    }
    const std::array<int, 3> direct_bounds = _checked_ewald_bounds_3d({
        lattice.direct_cutoff * recip.row(0).norm() / twopi
            + max_fractional_displacement[0],
        lattice.direct_cutoff * recip.row(1).norm() / twopi
            + max_fractional_displacement[1],
        lattice.direct_cutoff * recip.row(2).norm() / twopi
            + max_fractional_displacement[2],
    }, "direct-space");
    for (int n1 = -direct_bounds[0]; n1 <= direct_bounds[0]; ++n1) {
        for (int n2 = -direct_bounds[1]; n2 <= direct_bounds[1]; ++n2) {
            for (int n3 = -direct_bounds[2]; n3 <= direct_bounds[2]; ++n3) {
                lattice.direct_vectors.push_back(
                    n1 * translations[0]
                    + n2 * translations[1]
                    + n3 * translations[2]);
            }
        }
    }
    return lattice;
}

// 3-D Ewald Madelung-constant matrix (madelkonst.f).
inline Eigen::MatrixXd _madkonst_3d(const std::vector<std::vector<WSNeighbor>>& ews,
                                     const std::vector<Eigen::Vector3d>& translations,
                                     int n_atoms) {
    const EwaldLattice3D lattice = _ewald_lattice_3d(translations, ews);

    std::vector<double> kpref;
    kpref.reserve(lattice.reciprocal_vectors.size());
    for (double squared_norm : lattice.reciprocal_squared_norms) {
        const double factor =
            squared_norm / (4.0 * lattice.alpha * lattice.alpha);
        kpref.push_back(
            std::exp(-factor) / factor * M_PI
            / (lattice.alpha * lattice.alpha) / lattice.volume);
    }

    const double self_const = -2.0 * lattice.alpha / std::sqrt(M_PI);
    const double background =
        -M_PI / (lattice.volume * lattice.alpha * lattice.alpha);

    Eigen::MatrixXd mad = Eigen::MatrixXd::Zero(n_atoms, n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        for (auto& nb : ews[i]) {
            Eigen::Vector3d vij = -nb.disp;
            double recip_term = 0.0;
            for (size_t k = 0; k < lattice.reciprocal_vectors.size(); ++k)
                recip_term += kpref[k]
                    * std::cos(lattice.reciprocal_vectors[k].dot(vij));
            double direct_term = 0.0;
            for (const auto& dv : lattice.direct_vectors) {
                Eigen::Vector3d pos = vij + dv;
                double dist = pos.norm();
                if (dist > lattice.direct_cutoff) continue;
                if (dist < 1e-8) {
                    direct_term += self_const;
                } else {
                    direct_term += std::erfc(lattice.alpha * dist) / dist;
                }
            }
            mad(i, nb.origin) += nb.weight * (recip_term + direct_term + background);
        }
    }
    return mad;
}

struct EwaldLattice2D {
    double area = 0.0;
    double alpha = 0.0;
    double direct_cutoff = 0.0;
    Eigen::Vector3d nhat = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> reciprocal_vectors;
    std::vector<double> reciprocal_magnitudes;
    std::vector<Eigen::Vector3d> direct_vectors;
};

inline std::array<int, 2> _checked_ewald_bounds_2d(
    const std::array<double, 2>& unrounded_bounds,
    const char* space) {
    constexpr std::size_t max_candidate_vectors = 500000;
    std::array<int, 2> bounds = {0, 0};
    std::size_t candidate_vectors = 1;
    for (int axis = 0; axis < 2; ++axis) {
        const double unrounded = unrounded_bounds[axis];
        if (!std::isfinite(unrounded) || unrounded < 0.0) {
            throw std::invalid_argument(
                std::string("2-D CCM Ewald ") + space
                + " bounds are not finite; use a reduced lattice basis");
        }
        const double rounded = std::ceil(unrounded) + 1.0;
        if (rounded > static_cast<double>(max_candidate_vectors)) {
            throw std::invalid_argument(
                std::string("2-D CCM Ewald ") + space
                + " enumeration exceeds the safe candidate budget; use a "
                  "reduced lattice basis");
        }
        bounds[axis] = static_cast<int>(rounded);
        const std::size_t width =
            2 * static_cast<std::size_t>(bounds[axis]) + 1;
        if (width > max_candidate_vectors / candidate_vectors) {
            throw std::invalid_argument(
                std::string("2-D CCM Ewald ") + space
                + " enumeration exceeds the safe candidate budget; use a "
                  "reduced lattice basis");
        }
        candidate_vectors *= width;
    }
    return bounds;
}

// Cutoff-complete direct and reciprocal vector inventories for the 2-D (Parry)
// Ewald sums -- the surface twin of _ewald_lattice_3d, and the fix for issue
// #187.  A fixed +/-12 coefficient box is not a geometric cutoff: a unimodular
// shear moves a short physical vector to a large, near-cancelling coefficient
// pair, and the radius needed grows with the shear, so no fixed radius is
// correct.  The sets enumerated here -- {K != 0 : |K| <= K_cut} and
// {L : |v + L| <= r_cut} -- are sets of *physical* vectors and so are identical
// for every unimodular choice of the surface basis.
inline EwaldLattice2D _ewald_lattice_2d(
    const std::vector<Eigen::Vector3d>& translations,
    const std::vector<std::vector<WSNeighbor>>& ews) {
    constexpr double exponent_cutoff = 30.0;
    const double twopi = 2.0 * M_PI;

    EwaldLattice2D lattice;
    const Eigen::Vector3d a1 = translations[0], a2 = translations[1];
    const Eigen::Vector3d normal = a1.cross(a2);
    lattice.area = normal.norm();
    lattice.nhat = normal / lattice.area;
    // b_i . a_j = 2pi delta_ij, both b_i in the surface plane.
    const Eigen::Vector3d b1 = twopi * a2.cross(lattice.nhat) / lattice.area;
    const Eigen::Vector3d b2 = twopi * lattice.nhat.cross(a1) / lattice.area;
    lattice.alpha = 0.85 * std::sqrt(M_PI) / std::sqrt(lattice.area);

    const double reciprocal_cutoff =
        lattice.alpha * std::sqrt(4.0 * exponent_cutoff);
    // n_i = K.a_i/2pi, so |n_i| <= K_cut |a_i|/2pi.
    const std::array<int, 2> reciprocal_bounds = _checked_ewald_bounds_2d({
        reciprocal_cutoff * a1.norm() / twopi,
        reciprocal_cutoff * a2.norm() / twopi,
    }, "reciprocal-space");
    for (int n1 = -reciprocal_bounds[0]; n1 <= reciprocal_bounds[0]; ++n1) {
        for (int n2 = -reciprocal_bounds[1]; n2 <= reciprocal_bounds[1]; ++n2) {
            if (n1 == 0 && n2 == 0) continue;
            const Eigen::Vector3d vector = n1 * b1 + n2 * b2;
            const double magnitude = vector.norm();
            if (magnitude > reciprocal_cutoff) continue;
            lattice.reciprocal_vectors.push_back(vector);
            lattice.reciprocal_magnitudes.push_back(magnitude);
        }
    }

    // For r = v + sum_i n_i a_i and |r| <= r_cut,
    // |n_i| <= |b_i| r_cut/(2 pi) + |b_i.v|/(2 pi).  b_i . v picks up only the
    // in-plane part of v, which is what the coefficients span.  Include the
    // largest fractional displacement in the Ewald-WS inventory so that
    // translated atom pairs cannot need a coefficient outside the box.
    lattice.direct_cutoff = std::sqrt(exponent_cutoff) / lattice.alpha;
    double max_fractional_displacement[2] = {0.0, 0.0};
    const Eigen::Vector3d reciprocal_rows[2] = {b1, b2};
    for (const auto& cell : ews) {
        for (const auto& neighbor : cell) {
            for (int axis = 0; axis < 2; ++axis) {
                const double component = std::abs(
                    reciprocal_rows[axis].dot(neighbor.disp) / twopi);
                if (component > max_fractional_displacement[axis]) {
                    max_fractional_displacement[axis] = component;
                }
            }
        }
    }
    const std::array<int, 2> direct_bounds = _checked_ewald_bounds_2d({
        lattice.direct_cutoff * b1.norm() / twopi
            + max_fractional_displacement[0],
        lattice.direct_cutoff * b2.norm() / twopi
            + max_fractional_displacement[1],
    }, "direct-space");
    for (int n1 = -direct_bounds[0]; n1 <= direct_bounds[0]; ++n1) {
        for (int n2 = -direct_bounds[1]; n2 <= direct_bounds[1]; ++n2) {
            lattice.direct_vectors.push_back(n1 * a1 + n2 * a2);
        }
    }
    return lattice;
}

// Scaled complementary error function erfcx(x) = exp(x^2).erfc(x), evaluated
// without the intermediate exp(x^2) overflow / erfc underflow of the naive
// product.  Mirrors the shared slab-Ewald helper in cpp/src/ewald.cpp so both
// Parry kernels in the tree use one numerical form.
inline double _ccm_erfcx(double x) {
    if (x >= 25.0) {
        const double t = 1.0 / (x * x);
        return (1.0 / (x * std::sqrt(M_PI)))
               * (1.0 - 0.5 * t + 0.75 * t * t - 1.875 * t * t * t
                  + 6.5625 * t * t * t * t);
    }
    return std::exp(x * x) * std::erfc(x);
}

// The two Parry reciprocal factors e^{|K|z} erfc(az + |K|/2a) and
// e^{-|K|z} erfc(-az + |K|/2a) -- eq. (7) of Parry, Surf. Sci. 49, 433 (1975),
// = F_kl(z) of de Leeuw & Perram, Mol. Phys. 37, 1313 (1979) eq. (11).
// Evaluated as written, the growing exponential overflows while its erfc
// underflows (inf * 0 = nan); erfc(x) = e^{-x^2} erfcx(x) cancels the growth
// exactly:
//   e^{|K|z} erfc(az + |K|/2a) = e^{-a^2 z^2 - |K|^2/4a^2} erfcx(az + |K|/2a).
// The pair is symmetric under z -> -z with its members swapped, so it is
// evaluated at |z| (where erfcx sees only non-negative arguments) and swapped.
inline void _parry_recip_pair(double kmag, double alpha, double z,
                              double& rising, double& falling) {
    const double half = kmag / (2.0 * alpha);
    const double az = alpha * std::abs(z);
    const double common = std::exp(-(az * az) - half * half);
    const double rising_abs = common * _ccm_erfcx(half + az);
    const double falling_abs =
        std::exp(-kmag * std::abs(z)) * std::erfc(half - az);
    if (z < 0.0) {
        rising = falling_abs;
        falling = rising_abs;
    } else {
        rising = rising_abs;
        falling = falling_abs;
    }
}

// 2-D Ewald (Parry/Heyes) Madelung-constant matrix (madelkonst.f CCM2D branch).
// Eq. (7) of Parry, Surf. Sci. 49, 433 (1975) (erratum Surf. Sci. 54, 195
// (1976)), with the K=0 term Parry drops by cell neutrality restored in the
// per-pair form of de Leeuw & Perram, Mol. Phys. 37, 1313 (1979) eqs. (8)/(12);
// the direct sum and its -2a/sqrt(pi) self term are their eqs. (14a)-(14b).
// Summed over the cutoff-complete inventory of _ewald_lattice_2d, so the result
// is invariant under a unimodular change of surface basis (issue #187).
inline Eigen::MatrixXd _madkonst_2d(const std::vector<std::vector<WSNeighbor>>& ews,
                                     const std::vector<Eigen::Vector3d>& translations,
                                     int n_atoms) {
    const EwaldLattice2D lattice = _ewald_lattice_2d(translations, ews);
    const double area = lattice.area;
    const double alpha = lattice.alpha;
    const Eigen::Vector3d& nhat = lattice.nhat;
    const double twopi = 2.0 * M_PI;
    const std::vector<Eigen::Vector3d>& kvecs = lattice.reciprocal_vectors;
    const std::vector<double>& kmag = lattice.reciprocal_magnitudes;

    double self_const = -2.0 * alpha / std::sqrt(M_PI);

    Eigen::MatrixXd mad = Eigen::MatrixXd::Zero(n_atoms, n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        for (auto& nb : ews[i]) {
            Eigen::Vector3d vij = -nb.disp;
            double zc = vij.dot(nhat);

            // Reciprocal part
            double recip = 0.0;
            for (size_t kk = 0; kk < kvecs.size(); ++kk) {
                double krij = kvecs[kk].dot(vij);
                double km = kmag[kk];
                double rising = 0.0, falling = 0.0;
                _parry_recip_pair(km, alpha, zc, rising, falling);
                recip += std::cos(krij) / km * (rising + falling);
            }
            recip *= M_PI / area;

            // K=0 term
            double alpha_z = alpha * zc;
            double k0 = -twopi / area * (
                zc * std::erf(alpha_z)
                + std::exp(-alpha_z * alpha_z) / (alpha * std::sqrt(M_PI)));

            // Direct (real-space) 2-D sum, over |v + L| <= r_cut.
            double direct = 0.0;
            for (const auto& dv : lattice.direct_vectors) {
                Eigen::Vector3d pos = vij + dv;
                double dist = pos.norm();
                if (dist < 1e-8) {
                    direct += self_const;
                } else if (dist <= lattice.direct_cutoff) {
                    direct += std::erfc(alpha * dist) / dist;
                }
            }
            mad(i, nb.origin) += nb.weight * (recip + k0 + direct);
        }
    }
    return mad;
}

// Madelung potential at every atom (madelsum.f with SMADEL).
inline Eigen::VectorXd _madelung_potential_ewald(const Eigen::VectorXd& net_charges,
                                                  const Eigen::MatrixXd& madkonst,
                                                  const std::vector<std::vector<WSNeighbor>>& ews) {
    int natom = static_cast<int>(ews.size());
    Eigen::VectorXd full = madkonst * net_charges;
    Eigen::VectorXd smadel = Eigen::VectorXd::Zero(natom);
    for (int i = 0; i < natom; ++i) {
        for (auto& nb : ews[i]) {
            double d = nb.disp.norm();
            if (d > 1e-8)
                smadel(i) += net_charges(nb.origin) * nb.weight / d;
        }
    }
    return full - smadel;
}

// CCM core–core repulsion (NOEWALD): WS-weighted point-charge sum.
inline double core_repulsion_ccm(const std::vector<int>& Z,
                                  const std::vector<Eigen::Vector3d>& C,
                                  const std::vector<int>& cz,
                                  const WignerSeitzCells& ws) {
    int natom = static_cast<int>(Z.size());
    double e = 0.0;
    for (int i = 0; i < natom; ++i)
        for (auto& nb : ws.cells[i])
            e += 0.5 * cz[i] * cz[nb.origin] * nb.weight / nb.disp.norm();
    return e;
}

// --------------------------------------------------------------------------- //
// CCM SCF driver.                                                              //
// --------------------------------------------------------------------------- //

inline MsindoResult run_ccm_core(
    const std::vector<int>& Z,
    const std::vector<std::array<double, 3>>& coords_angstrom,
    const std::vector<std::array<double, 3>>& translations_angstrom,
    const MsindoParameterSet& p,
    bool madelung = false,
    int max_iter = 200, double conv_tol = 1e-9,
    int charge = 0) {
    validate_ccm_native_inputs(
        Z, coords_angstrom, translations_angstrom, max_iter, conv_tol);
    int natom = static_cast<int>(Z.size());
    int ndim = static_cast<int>(translations_angstrom.size());

    // Element scope guard (H-Kr, Z=1-36).
    for (int z : Z) {
        if (z < 1 || z > 36) {
            MsindoResult res;
            return res;
        }
    }

    // Convert to bohr
    std::vector<Eigen::Vector3d> C(natom);
    for (int i = 0; i < natom; ++i)
        C[i] = Eigen::Vector3d(coords_angstrom[i][0], coords_angstrom[i][1],
                               coords_angstrom[i][2]) * ANGSTROM_TO_BOHR;

    std::vector<Eigen::Vector3d> T(ndim);
    for (int d = 0; d < ndim; ++d)
        T[d] = Eigen::Vector3d(translations_angstrom[d][0], translations_angstrom[d][1],
                               translations_angstrom[d][2]) * ANGSTROM_TO_BOHR;

    std::vector<std::array<int, 2>> blocks;
    int nsto = 0;
    for (int z : Z) { int nb = p.n_basis(z); blocks.push_back({nsto, nsto + nb}); nsto += nb; }

    std::vector<int> cz(natom);
    int nelec = 0;
    for (int i = 0; i < natom; ++i) {
        cz[i] = MsindoParameterSet::eff_core_charge(Z[i]);
        nelec += cz[i];
    }
    nelec -= charge;

    // Closed-shell guard.
    if (nelec % 2 != 0) {
        MsindoResult res;
        return res; // not converged
    }
    int nocc = nelec / 2;

    // Build WS cells.
    WignerSeitzCells ws = build_wigner_seitz(C, T);
    if (!ws.is_valid(natom)) {
        MsindoResult res;
        return res;
    }

    // Build CCM H and G.
    Eigen::MatrixXd H, G;
    build_core_and_gamma_ccm(Z, C, blocks, nsto, p, ws, H, G);

    double e_core = core_repulsion_ccm(Z, C, cz, ws);

    // Madelung fock_extra callback.
    std::function<std::pair<Eigen::MatrixXd, double>(const Eigen::MatrixXd&)> fock_extra;
    Eigen::MatrixXd madkonst;
    std::vector<std::vector<WSNeighbor>> ews;

    if (madelung) {
        ews = _ewald_ws_cells(ws);
        if (ndim == 2)
            madkonst = _madkonst_2d(ews, T, natom);
        else if (ndim == 3)
            madkonst = _madkonst_3d(ews, T, natom);

        // Capture by copy for the lambda.
        fock_extra = [=, &blocks, &cz](const Eigen::MatrixXd& P) mutable
            -> std::pair<Eigen::MatrixXd, double> {
            Eigen::VectorXd net = _net_charges(P, blocks, cz);
            Eigen::VectorXd mad;
            if (ndim == 1)
                mad = _madelung_potential_1d(net, ews, T[0]);
            else
                mad = _madelung_potential_ewald(net, madkonst, ews);

            Eigen::MatrixXd F_add = Eigen::MatrixXd::Zero(P.rows(), P.cols());
            for (int i = 0; i < natom; ++i) {
                int lo = blocks[i][0], hi = blocks[i][1];
                for (int k = lo; k < hi; ++k)
                    F_add(k, k) = -mad(i);
            }
            Eigen::VectorXd czd(natom);
            for (int i = 0; i < natom; ++i) czd(i) = cz[i];
            double e_add = 0.5 * czd.dot(mad);
            return {F_add, e_add};
        };
    }

    MsindoResult res = scf_rhf_driver(H, G, blocks, Z, nocc, p, max_iter, conv_tol,
                                       fock_extra);

    if (res.converged) {
        res.total_energy = res.electronic_energy + e_core;
        res.binding_energy = res.total_energy;
        for (int z : Z) res.binding_energy -= ateng(z, p);
    }
    return res;
}

// Energy from a pre-built Wigner-Seitz set (for FD gradient with fixed
// WS topology — matching MSINDO's fixed-weight analytic gradient).
inline double ccm_energy_with_ws(
    const std::vector<int>& Z,
    const std::vector<Eigen::Vector3d>& C,
    const std::vector<Eigen::Vector3d>& T,
    const std::vector<std::array<int, 2>>& blocks,
    int nsto, const std::vector<int>& cz, int nocc,
    const MsindoParameterSet& p, const WignerSeitzCells& ws,
    bool madelung, int ndim,
    int max_iter, double conv_tol) {
    int natom = static_cast<int>(Z.size());
    Eigen::MatrixXd H, G;
    build_core_and_gamma_ccm(Z, C, blocks, nsto, p, ws, H, G);
    double e_core = core_repulsion_ccm(Z, C, cz, ws);

    std::function<std::pair<Eigen::MatrixXd, double>(const Eigen::MatrixXd&)> fock_extra;
    Eigen::MatrixXd madkonst;
    std::vector<std::vector<WSNeighbor>> ews;

    if (madelung) {
        ews = _ewald_ws_cells(ws);
        if (ndim == 2) madkonst = _madkonst_2d(ews, T, natom);
        else if (ndim == 3) madkonst = _madkonst_3d(ews, T, natom);
        fock_extra = [=, &blocks, &cz](const Eigen::MatrixXd& P) mutable
            -> std::pair<Eigen::MatrixXd, double> {
            Eigen::VectorXd net = _net_charges(P, blocks, cz);
            Eigen::VectorXd mad;
            if (ndim == 1) mad = _madelung_potential_1d(net, ews, T[0]);
            else mad = _madelung_potential_ewald(net, madkonst, ews);
            Eigen::MatrixXd F_add = Eigen::MatrixXd::Zero(P.rows(), P.cols());
            for (int i = 0; i < natom; ++i) {
                int lo = blocks[i][0], hi = blocks[i][1];
                for (int k = lo; k < hi; ++k) F_add(k, k) = -mad(i);
            }
            Eigen::VectorXd czd(natom);
            for (int i = 0; i < natom; ++i) czd(i) = cz[i];
            return std::make_pair(F_add, 0.5 * czd.dot(mad));
        };
    }

    MsindoResult res = scf_rhf_driver(H, G, blocks, Z, nocc, p, max_iter, conv_tol, fock_extra);
    if (!res.converged) return std::numeric_limits<double>::quiet_NaN();
    return res.electronic_energy + e_core;
}

// Finite-difference nuclear gradient of the CCM total energy (Ha/bohr).
// Holds Wigner-Seitz topology fixed at the reference geometry.
inline std::vector<std::array<double, 3>> ccm_gradient_fd(
    const std::vector<int>& Z,
    const std::vector<std::array<double, 3>>& coords_angstrom,
    const std::vector<std::array<double, 3>>& translations_angstrom,
    const MsindoParameterSet& p,
    bool madelung = false,
    const std::vector<int>& atoms = {},
    int max_iter = 200, double conv_tol = 1e-10, double step = 1e-3,
    int charge = 0) {
    validate_ccm_native_inputs(
        Z, coords_angstrom, translations_angstrom, max_iter, conv_tol);
    if (!std::isfinite(step) || step <= 0.0) {
        throw std::invalid_argument(
            "MSINDO CCM finite-difference step must be positive and finite");
    }
    int natom = static_cast<int>(Z.size());
    int ndim = static_cast<int>(translations_angstrom.size());
    double h = step * ANGSTROM_TO_BOHR;

    std::vector<Eigen::Vector3d> C0(natom);
    for (int i = 0; i < natom; ++i)
        C0[i] = Eigen::Vector3d(coords_angstrom[i][0], coords_angstrom[i][1],
                                coords_angstrom[i][2]) * ANGSTROM_TO_BOHR;

    std::vector<Eigen::Vector3d> T(ndim);
    for (int d = 0; d < ndim; ++d)
        T[d] = Eigen::Vector3d(translations_angstrom[d][0], translations_angstrom[d][1],
                               translations_angstrom[d][2]) * ANGSTROM_TO_BOHR;

    std::vector<std::array<int, 2>> blocks;
    int nsto = 0;
    for (int z : Z) { int nb = p.n_basis(z); blocks.push_back({nsto, nsto + nb}); nsto += nb; }

    std::vector<int> cz(natom);
    int nelec = 0;
    for (int i = 0; i < natom; ++i) { cz[i] = MsindoParameterSet::eff_core_charge(Z[i]); nelec += cz[i]; }
    nelec -= charge;
    if (nelec % 2 != 0) return {};
    int nocc = nelec / 2;

    WignerSeitzCells ws0 = build_wigner_seitz(C0, T);
    if (!ws0.is_valid(natom)) return {};

    // Fixed lattice translation per WS neighbour
    struct LatticeEntry { int origin; double weight; Eigen::Vector3d t; };
    std::vector<std::vector<LatticeEntry>> lattices(natom);
    for (int i = 0; i < natom; ++i) {
        lattices[i].reserve(ws0.cells[i].size());
        for (auto& nb : ws0.cells[i])
            lattices[i].push_back({nb.origin, nb.weight, nb.disp + C0[i] - C0[nb.origin]});
    }

    auto energy_at = [&](const std::vector<Eigen::Vector3d>& C) -> double {
        WignerSeitzCells ws;
        ws.translations = ws0.translations;
        ws.cells.resize(natom);
        for (int i = 0; i < natom; ++i) {
            ws.cells[i].reserve(lattices[i].size());
            for (auto& le : lattices[i])
                ws.cells[i].push_back(WSNeighbor{le.origin, le.weight,
                                                  C[le.origin] + le.t - C[i]});
        }
        return ccm_energy_with_ws(Z, C, T, blocks, nsto, cz, nocc, p, ws,
                                  madelung, ndim, max_iter, conv_tol);
    };

    std::vector<bool> which(natom, atoms.empty());
    for (int a : atoms) if (a >= 0 && a < natom) which[a] = true;

    std::vector<std::array<double, 3>> grad(natom, {0.0, 0.0, 0.0});
    for (int i = 0; i < natom; ++i) {
        if (!which[i]) continue;
        for (int d = 0; d < 3; ++d) {
            auto cp = C0; cp[i][d] += h;
            auto cm = C0; cm[i][d] -= h;
            grad[i][d] = (energy_at(cp) - energy_at(cm)) / (2.0 * h);
        }
    }
    return grad;
}

// =========================================================================== //
// Analytic CCM nuclear gradient — C++ port of the validated Python reference   //
// (msindo_ccm_gradient_analytic.py).  WS topology held fixed at the reference  //
// geometry; per-pair derivatives via pair_blocks_deriv (msindo_gradient.hpp);  //
// the Madelung gradient uses the DIRECT per-neighbour assembly (the exact      //
// fixed-charge derivative of e_mad = ½ Σ_I q_I V_I), NOT the central-atom-only  //
// dedmadelsum.f form (which is exact only on mirror-symmetric WS cells).       //
// =========================================================================== //

// 3-D Ewald Madelung matrix-element derivative dM/d(vij) (dmadkonst.f CCM3D).
inline Eigen::Vector3d _dmadkonst_3d(const Eigen::Vector3d& vij,
                                     const std::vector<Eigen::Vector3d>& kvecs,
                                     const std::vector<double>& kb,
                                     const std::vector<Eigen::Vector3d>& dvecs,
                                     double alpha, double volume,
                                     double direct_cutoff) {
    Eigen::Vector3d dM = Eigen::Vector3d::Zero();
    // Reciprocal: dM = -π/(α²V) Σ_K K sin(K·v) e^{-K²/4α²}/(K²/4α²)
    for (size_t k = 0; k < kvecs.size(); ++k) {
        double faktor = kb[k] / (4.0 * alpha * alpha);
        double pre = std::sin(kvecs[k].dot(vij)) * std::exp(-faktor) / faktor;
        dM += kvecs[k] * pre;
    }
    dM *= -M_PI / (alpha * alpha * volume);
    // Direct: -Σ_T (v+T)[erfc(αr)/r + 2α/√π e^{-α²r²}]/r²
    double tpre = 2.0 * alpha / std::sqrt(M_PI);
    for (auto& dv : dvecs) {
        Eigen::Vector3d vt = vij + dv;
        double r = vt.norm();
        if (r > direct_cutoff) continue;
        if (r > 1e-12) {
            double pref = (std::erfc(alpha * r) / r
                           + tpre * std::exp(-alpha * alpha * r * r)) / (r * r);
            dM -= vt * pref;
        }
    }
    return dM;
}

// 2-D (Parry/Heyes slab) Ewald Madelung matrix-element derivative dM/d(vij)
// (dmadkonst.f CCM2D).  Frame-free via the plane normal n̂ (z = v·n̂); the
// Gaussian cross terms in the recip ∂/∂z cancel exactly, leaving F_K / G_K.
// kvecs/dvecs/direct_cutoff come from _ewald_lattice_2d, so the derivative is
// summed over exactly the physical-vector sets the energy used and inherits its
// invariance under a unimodular change of surface basis (issue #187).
inline Eigen::Vector3d _dmadkonst_2d(const Eigen::Vector3d& vij,
                                     const std::vector<Eigen::Vector3d>& kvecs,
                                     const std::vector<double>& kmag,
                                     const std::vector<Eigen::Vector3d>& dvecs,
                                     double alpha, double area,
                                     const Eigen::Vector3d& nhat,
                                     double direct_cutoff) {
    Eigen::Vector3d dM = Eigen::Vector3d::Zero();
    double z = vij.dot(nhat);
    double out = 0.0;  // out-of-plane (n̂) accumulator: recip-z part + K=0 term
    for (size_t k = 0; k < kvecs.size(); ++k) {
        double km = kmag[k];
        double kdotv = kvecs[k].dot(vij);
        double ep = 0.0, em = 0.0;
        _parry_recip_pair(km, alpha, z, ep, em);
        double F = ep + em, Gg = ep - em;
        // in-plane: -(π/area) sin(K·v) F/|K| · K
        dM += (-(M_PI / area) * std::sin(kdotv) * F / km) * kvecs[k];
        // out-of-plane recip: (π/area) cos(K·v) G
        out += (M_PI / area) * std::cos(kdotv) * Gg;
    }
    out += -(2.0 * M_PI / area) * std::erf(alpha * z);  // K=0 term
    dM += out * nhat;
    // direct: -Σ_T (v+T)[erfc(αr)/r + 2α/√π e^{-α²r²}]/r², over |v+T| <= r_cut
    double tpre = 2.0 * alpha / std::sqrt(M_PI);
    for (auto& dv : dvecs) {
        Eigen::Vector3d pos = vij + dv;
        double r = pos.norm();
        if (r > 1e-12 && r <= direct_cutoff) {
            double pref = (std::erfc(alpha * r) / r
                           + tpre * std::exp(-alpha * alpha * r * r)) / (r * r);
            dM -= pos * pref;
        }
    }
    return dM;
}

// Add the Madelung/Ewald gradient (Ha/bohr) in place — direct assembly:
//   grad[o] += ½ q_i q_o w ∇h(disp),  grad[i] -= ½ q_i q_o w ∇h(disp).
inline void _add_madelung_gradient(std::vector<std::array<double, 3>>& grad,
                                   const Eigen::VectorXd& net,
                                   const std::vector<Eigen::Vector3d>& C0,
                                   const std::vector<Eigen::Vector3d>& T,
                                   const WignerSeitzCells& ws) {
    int natom = static_cast<int>(C0.size());
    int dim = static_cast<int>(T.size());

    std::vector<Eigen::Vector3d> shells;                 // 1-D
    Eigen::Vector3d nhat = Eigen::Vector3d::Zero();      // 2-D
    double area = 0.0, alpha = 0.0, vol = 0.0, direct_cutoff = 0.0;
    std::vector<Eigen::Vector3d> kvecs, dvecs;           // 2-D/3-D recip + direct
    std::vector<double> kmag, kb;

    if (dim == 1) {
        Eigen::Vector3d Tv = T[0];
        shells = {Tv, -Tv, 2.0 * Tv, -2.0 * Tv};
    } else if (dim == 2) {
        // Shares _ewald_lattice_2d with the energy, so the gradient is the
        // exact derivative of the sum that was actually evaluated.
        const EwaldLattice2D lattice =
            _ewald_lattice_2d(T, _ewald_ws_cells(ws));
        area = lattice.area;
        alpha = lattice.alpha;
        nhat = lattice.nhat;
        direct_cutoff = lattice.direct_cutoff;
        kvecs = lattice.reciprocal_vectors;
        kmag = lattice.reciprocal_magnitudes;
        dvecs = lattice.direct_vectors;
    } else {  // dim == 3
        const EwaldLattice3D lattice =
            _ewald_lattice_3d(T, _ewald_ws_cells(ws));
        vol = lattice.volume;
        alpha = lattice.alpha;
        direct_cutoff = lattice.direct_cutoff;
        kvecs = lattice.reciprocal_vectors;
        kb = lattice.reciprocal_squared_norms;
        dvecs = lattice.direct_vectors;
    }

    for (int i = 0; i < natom; ++i) {
        double qi = net(i);
        if (qi == 0.0) continue;
        for (auto& nb : ws.cells[i]) {
            int o = nb.origin;
            double qo = net(o);
            if (qo == 0.0) continue;
            double d = nb.disp.norm();
            Eigen::Vector3d gh = Eigen::Vector3d::Zero();
            if (dim == 1) {
                for (auto& sh : shells) {
                    Eigen::Vector3d u = nb.disp + sh;
                    double r = u.norm();
                    if (r > 1e-12) gh -= u / (r * r * r);
                }
            } else if (dim == 2) {
                gh = -_dmadkonst_2d(-nb.disp, kvecs, kmag, dvecs, alpha, area,
                                    nhat, direct_cutoff);
                if (d > 1e-12) gh += nb.disp / (d * d * d);
            } else {
                gh = -_dmadkonst_3d(
                    -nb.disp, kvecs, kb, dvecs, alpha, vol,
                    direct_cutoff);
                if (d > 1e-12) gh += nb.disp / (d * d * d);
            }
            Eigen::Vector3d term = 0.5 * qi * qo * nb.weight * gh;
            grad[o][0] += term.x(); grad[o][1] += term.y(); grad[o][2] += term.z();
            grad[i][0] -= term.x(); grad[i][1] -= term.y(); grad[i][2] -= term.z();
        }
    }
}

// Analytic CCM nuclear gradient (Ha/bohr).  Closed-shell (RHF) only.
inline std::vector<std::array<double, 3>> ccm_gradient_analytic(
    const std::vector<int>& Z,
    const std::vector<std::array<double, 3>>& coords_angstrom,
    const std::vector<std::array<double, 3>>& translations_angstrom,
    const MsindoParameterSet& p,
    bool madelung = false, int charge = 0,
    int max_iter = 200, double conv_tol = 1e-10) {
    validate_ccm_native_inputs(
        Z, coords_angstrom, translations_angstrom, max_iter, conv_tol);
    int natom = static_cast<int>(Z.size());
    int ndim = static_cast<int>(translations_angstrom.size());

    for (int z : Z) if (z < 1 || z > 36) return {};

    std::vector<Eigen::Vector3d> C(natom);
    for (int i = 0; i < natom; ++i)
        C[i] = Eigen::Vector3d(coords_angstrom[i][0], coords_angstrom[i][1],
                               coords_angstrom[i][2]) * ANGSTROM_TO_BOHR;
    std::vector<Eigen::Vector3d> T(ndim);
    for (int d = 0; d < ndim; ++d)
        T[d] = Eigen::Vector3d(translations_angstrom[d][0], translations_angstrom[d][1],
                               translations_angstrom[d][2]) * ANGSTROM_TO_BOHR;

    std::vector<std::array<int, 2>> blocks;
    int nsto = 0;
    for (int z : Z) { int nb = p.n_basis(z); blocks.push_back({nsto, nsto + nb}); nsto += nb; }

    std::vector<int> cz(natom);
    int nelec = 0;
    for (int i = 0; i < natom; ++i) { cz[i] = MsindoParameterSet::eff_core_charge(Z[i]); nelec += cz[i]; }
    nelec -= charge;
    if (nelec % 2 != 0) return {};  // closed-shell only
    int nocc = nelec / 2;

    WignerSeitzCells ws = build_wigner_seitz(C, T);
    if (!ws.is_valid(natom)) return {};

    // SCF (mirror run_ccm_core), optional Madelung embedding.
    Eigen::MatrixXd H, G;
    build_core_and_gamma_ccm(Z, C, blocks, nsto, p, ws, H, G);

    std::function<std::pair<Eigen::MatrixXd, double>(const Eigen::MatrixXd&)> fock_extra;
    Eigen::MatrixXd madkonst;
    std::vector<std::vector<WSNeighbor>> ews;
    if (madelung) {
        ews = _ewald_ws_cells(ws);
        if (ndim == 2) madkonst = _madkonst_2d(ews, T, natom);
        else if (ndim == 3) madkonst = _madkonst_3d(ews, T, natom);
        fock_extra = [=, &blocks, &cz](const Eigen::MatrixXd& P) mutable
            -> std::pair<Eigen::MatrixXd, double> {
            Eigen::VectorXd net = _net_charges(P, blocks, cz);
            Eigen::VectorXd mad;
            if (ndim == 1) mad = _madelung_potential_1d(net, ews, T[0]);
            else mad = _madelung_potential_ewald(net, madkonst, ews);
            Eigen::MatrixXd F_add = Eigen::MatrixXd::Zero(P.rows(), P.cols());
            for (int ii = 0; ii < natom; ++ii) {
                int lo = blocks[ii][0], hi = blocks[ii][1];
                for (int k = lo; k < hi; ++k) F_add(k, k) = -mad(ii);
            }
            Eigen::VectorXd czd(natom);
            for (int ii = 0; ii < natom; ++ii) czd(ii) = cz[ii];
            return {F_add, 0.5 * czd.dot(mad)};
        };
    }
    MsindoResult res = scf_rhf_driver(H, G, blocks, Z, nocc, p, max_iter, conv_tol, fock_extra);
    if (!res.converged) return {};
    Eigen::MatrixXd P = res.density;

    std::vector<std::array<double, 3>> grad(natom, {0.0, 0.0, 0.0});

    // WS-weighted per-pair derivative loop (CCMDEDXYZK + CCMDEDXYZL, both on K).
    for (int k = 0; k < natom; ++k) {
        int nk = blocks[k][1] - blocks[k][0];
        auto P_kk = P.block(blocks[k][0], blocks[k][0], nk, nk);
        std::array<double, 3> rk = {C[k].x(), C[k].y(), C[k].z()};
        for (auto& nb : ws.cells[k]) {
            int j = nb.origin;
            double w = nb.weight;
            int nj = blocks[j][1] - blocks[j][0];
            Eigen::Vector3d img = C[k] + nb.disp;
            std::array<double, 3> rl = {img.x(), img.y(), img.z()};
            PairBlocksDeriv pd = pair_blocks_deriv(Z[k], Z[j], rk, rl, p);

            auto P_ll = P.block(blocks[j][0], blocks[j][0], nj, nj);
            auto P_kl = P.block(blocks[k][0], blocks[j][0], nk, nj);

            double R = pd.R;
            Eigen::Vector3d E = nb.disp / R;
            double cost = E.z(), sint = std::sqrt(std::max(0.0, 1.0 - cost * cost));
            double cosphi = 1, sinphi = 0, dphidx = 0, dphidy = 0, dphidz = 0;
            if (sint > 1e-12) {
                cosphi = E.x() / sint; sinphi = E.y() / sint;
                double c1 = sint * sint * R;
                dphidx = -E.y() / c1; dphidy = E.x() / c1;
            }
            double edt1 = cost * cosphi, edt2 = cost * sinphi, edt3 = -sint;
            double drdx = E.x(), drdy = E.y(), drdz = E.z();
            double dthx = edt1 / R, dthy = edt2 / R, dthz = edt3 / R;

            double dx = 0, dy = 0, dz = 0;
            // K-diagonal + off-diagonal (HK1)
            for (int i = 0; i < nk; ++i) {
                double Pii = P_kk(i, i);
                dx += Pii * (pd.HK1_DR(i, i) * drdx + pd.HK1_DT(i, i) * dthx + pd.HK1_DP(i, i) * dphidx);
                dy += Pii * (pd.HK1_DR(i, i) * drdy + pd.HK1_DT(i, i) * dthy + pd.HK1_DP(i, i) * dphidy);
                dz += Pii * (pd.HK1_DR(i, i) * drdz + pd.HK1_DT(i, i) * dthz + pd.HK1_DP(i, i) * dphidz);
                for (int ii = i + 1; ii < nk; ++ii) {
                    double fac = 2.0 * P_kk(ii, i);
                    dx += fac * (pd.HK1_DR(ii, i) * drdx + pd.HK1_DT(ii, i) * dthx + pd.HK1_DP(ii, i) * dphidx);
                    dy += fac * (pd.HK1_DR(ii, i) * drdy + pd.HK1_DT(ii, i) * dthy + pd.HK1_DP(ii, i) * dphidy);
                    dz += fac * (pd.HK1_DR(ii, i) * drdz + pd.HK1_DT(ii, i) * dthz + pd.HK1_DP(ii, i) * dphidz);
                }
            }
            // L-diagonal + off-diagonal (HL1)
            for (int i = 0; i < nj; ++i) {
                double Pii = P_ll(i, i);
                dx += Pii * (pd.HL1_DR(i, i) * drdx + pd.HL1_DT(i, i) * dthx + pd.HL1_DP(i, i) * dphidx);
                dy += Pii * (pd.HL1_DR(i, i) * drdy + pd.HL1_DT(i, i) * dthy + pd.HL1_DP(i, i) * dphidy);
                dz += Pii * (pd.HL1_DR(i, i) * drdz + pd.HL1_DT(i, i) * dthz + pd.HL1_DP(i, i) * dphidz);
                for (int ii = i + 1; ii < nj; ++ii) {
                    double fac = 2.0 * P_ll(ii, i);
                    dx += fac * (pd.HL1_DR(ii, i) * drdx + pd.HL1_DT(ii, i) * dthx + pd.HL1_DP(ii, i) * dphidx);
                    dy += fac * (pd.HL1_DR(ii, i) * drdy + pd.HL1_DT(ii, i) * dthy + pd.HL1_DP(ii, i) * dphidy);
                    dz += fac * (pd.HL1_DR(ii, i) * drdz + pd.HL1_DT(ii, i) * dthz + pd.HL1_DP(ii, i) * dphidz);
                }
            }
            // Pair block + monopole γ + nuclear (×2 = DEDXYZK + DEDXYZL rounds).
            for (int i = 0; i < nk; ++i) {
                double Pii = P_kk(i, i);
                for (int jj = 0; jj < nj; ++jj) {
                    double Pjj = P_ll(jj, jj), Pij = P_kl(i, jj);
                    double Fg = 0.5 * pd.d_gamma_local(i, jj) * (Pii * Pjj - 0.5 * Pij * Pij);
                    double Fr = Pij * pd.HKL2_DR(i, jj);
                    double Ft = Pij * pd.HKL2_DT(i, jj);
                    double Fp = Pij * pd.HKL2_DP(i, jj);
                    dx += 2.0 * ((Fg + Fr) * drdx + Ft * dthx + Fp * dphidx);
                    dy += 2.0 * ((Fg + Fr) * drdy + Ft * dthy + Fp * dphidy);
                    dz += 2.0 * ((Fg + Fr) * drdz + Ft * dthz + Fp * dphidz);
                }
            }
            double zke = static_cast<double>(MsindoParameterSet::eff_core_charge(Z[k]));
            double zle = static_cast<double>(MsindoParameterSet::eff_core_charge(Z[j]));
            double dnuc = 2.0 * 0.5 * zke * zle * (-1.0 / (R * R));
            dx += dnuc * drdx; dy += dnuc * drdy; dz += dnuc * drdz;

            grad[k][0] -= w * dx; grad[k][1] -= w * dy; grad[k][2] -= w * dz;
        }
    }

    if (madelung) {
        Eigen::VectorXd net = _net_charges(P, blocks, cz);
        _add_madelung_gradient(grad, net, C, T, ws);
    }
    return grad;
}

}  // namespace indo
}  // namespace semiempirical
}  // namespace vibeqc
