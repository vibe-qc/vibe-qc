#include "vibeqc/ewald.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vibeqc/diagnostics.hpp"

namespace vibeqc {

namespace {

// Perpendicular interplanar distance (bohr for real lattice, bohr⁻¹ for
// reciprocal) from the origin to the plane spanned by the other two
// columns of ``M``. Used to size the bounding box of the summation.
double interplanar_distance(const Eigen::Matrix3d& M, int axis) {
    const Eigen::Vector3d a = M.col((axis + 1) % 3);
    const Eigen::Vector3d b = M.col((axis + 2) % 3);
    const double cross_norm = a.cross(b).norm();
    const double det = M.determinant();
    if (cross_norm < 1e-14) {
        throw std::runtime_error("ewald: degenerate lattice");
    }
    return std::abs(det) / cross_norm;
}

// Enumerate integer lattice points (n_1, n_2, n_3) such that
// ||M * n|| <= cutoff, where M's columns are lattice basis vectors in
// Cartesian coordinates. Returns one Cartesian vector per cell kept.
//
// ``include_origin`` controls whether (0,0,0) is returned; the real-space
// Ewald sum wants it (it holds the A ≠ B, g=0 intra-cell pairs), while
// the reciprocal sum explicitly excludes G = 0.
std::vector<Eigen::Vector3d> enumerate_cells(const Eigen::Matrix3d& M,
                                             double cutoff,
                                             bool include_origin) {
    std::array<int, 3> n_max;
    for (int i = 0; i < 3; ++i) {
        n_max[i] = static_cast<int>(std::ceil(cutoff / interplanar_distance(M, i)));
    }
    std::vector<Eigen::Vector3d> out;
    const auto extent0 = 2 * static_cast<std::size_t>(n_max[0]) + 1;
    const auto extent1 = 2 * static_cast<std::size_t>(n_max[1]) + 1;
    const auto extent2 = 2 * static_cast<std::size_t>(n_max[2]) + 1;
    out.reserve(extent0 * extent1 * extent2);
    const double cutoff_sq = cutoff * cutoff;
    for (int n1 = -n_max[0]; n1 <= n_max[0]; ++n1) {
        for (int n2 = -n_max[1]; n2 <= n_max[1]; ++n2) {
            for (int n3 = -n_max[2]; n3 <= n_max[2]; ++n3) {
                if (!include_origin && n1 == 0 && n2 == 0 && n3 == 0) continue;
                const Eigen::Vector3d v =
                    static_cast<double>(n1) * M.col(0) +
                    static_cast<double>(n2) * M.col(1) +
                    static_cast<double>(n3) * M.col(2);
                if (v.squaredNorm() <= cutoff_sq) out.push_back(v);
            }
        }
    }
    return out;
}

// Return a lattice-equivalent representative in the parallelepiped whose
// fractional coordinates lie in [-1/2, 1/2].  Ewald, Ann. Phys. 369, 253
// (1921), Sec. III.2, Eqs. (20)-(24), doi:10.1002/andp.19213690304, selects
// the real-space kernel by the complete source-observer distance
// |R_A - R_B + g|.  Reducing the displacement merely relabels g and keeps
// that infinite set unchanged.  It also makes the enumeration independent
// of which (possibly far-unwrapped) lattice representative the caller used.
Eigen::Vector3d centered_periodic_representative(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3d& lattice_inverse,
    const Eigen::Vector3d& cart) {
    Eigen::Vector3d fractional = lattice_inverse * cart;
    for (int axis = 0; axis < 3; ++axis) {
        fractional[axis] -= std::nearbyint(fractional[axis]);
    }
    return lattice * fractional;
}

// Visit a pair-complete integer box around one centered representative.
// If d = M(f + n), |f_i| <= 1/2, and |d| <= pair_cutoff, projection onto
// the plane normal to lattice axis i gives
//
//   |f_i + n_i| <= pair_cutoff / h_i,
//
// where h_i is that axis's interplanar distance.  Centering the integer
// bounds on -f keeps the work tied to the physical cutoff even for
// high-aspect cells, and reducing f keeps it fixed for unwrapped coordinates.
template <typename Visitor>
void for_each_pair_candidate(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3d& lattice_inverse,
    const Eigen::Vector3d& cart_displacement,
    const std::array<double, 3>& fractional_spans,
    Visitor&& visitor) {
    Eigen::Vector3d fractional = lattice_inverse * cart_displacement;
    for (int axis = 0; axis < 3; ++axis) {
        fractional[axis] -= std::nearbyint(fractional[axis]);
    }

    std::array<int, 3> n_min;
    std::array<int, 3> n_max;
    for (int axis = 0; axis < 3; ++axis) {
        const double span = fractional_spans[axis];
        const double lower = -fractional[axis] - span;
        const double upper = -fractional[axis] + span;
        const double padding = 16.0
            * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(lower), std::abs(upper)});
        n_min[axis] = static_cast<int>(std::ceil(lower - padding));
        n_max[axis] = static_cast<int>(std::floor(upper + padding));
    }

    for (int n1 = n_min[0]; n1 <= n_max[0]; ++n1) {
        for (int n2 = n_min[1]; n2 <= n_max[1]; ++n2) {
            for (int n3 = n_min[2]; n3 <= n_max[2]; ++n3) {
                const Eigen::Vector3d shifted_fractional =
                    fractional + Eigen::Vector3d(
                        static_cast<double>(n1),
                        static_cast<double>(n2),
                        static_cast<double>(n3));
                visitor(lattice * shifted_fractional);
            }
        }
    }
}

std::array<double, 3> pair_fractional_spans(
    const Eigen::Matrix3d& lattice,
    double pair_cutoff) {
    std::array<double, 3> spans;
    for (int axis = 0; axis < 3; ++axis) {
        spans[axis] =
            pair_cutoff / interplanar_distance(lattice, axis);
    }
    return spans;
}

// Pick α automatically so that both real- and reciprocal-space tails are
// screened to at least ``tol``. The real-space erfc(α·R) ≈ tol ⇒
// α·R ≈ √(−ln tol); this is the standard convention used by Allen-
// Tildesley, DL_POLY, LAMMPS, etc.
double auto_alpha(double real_cutoff, double tol) {
    const double t = -std::log(tol);
    return std::sqrt(t) / real_cutoff;
}

// For a given α and tolerance, pick a reciprocal-space cutoff so that
// the Gaussian prefactor exp(−G²/(4α²)) ≤ tol ⇒ G²/(4α²) ≥ −ln tol ⇒
// G ≥ 2α √(−ln tol).
double auto_recip_cutoff(double alpha, double tol) {
    return 2.0 * alpha * std::sqrt(-std::log(tol));
}

// Enumerate the 2D lattice points n = n1·a1 + n2·a2 with ||n|| <= cutoff,
// returned as Cartesian vectors. ``a1``/``a2`` are the in-plane lattice
// vectors (Cartesian, length bohr for the real lattice or bohr⁻¹ for the
// reciprocal one). ``include_origin`` keeps (0,0): the real-space sum needs
// it (intra-cell i≠j pairs), the reciprocal sum excludes g = 0 (its special
// term is added separately). The per-axis bound uses the inter-line spacing
// A/|a_other| of the 2D cell of area A = |a1 × a2|.
std::vector<Eigen::Vector3d> enumerate_cells_2d(const Eigen::Vector3d& a1,
                                                const Eigen::Vector3d& a2,
                                                double cutoff,
                                                bool include_origin) {
    const double area = a1.cross(a2).norm();
    if (area < 1e-14) {
        throw std::runtime_error("ewald 2D: degenerate in-plane lattice");
    }
    const int n1_max = static_cast<int>(std::ceil(cutoff * a2.norm() / area));
    const int n2_max = static_cast<int>(std::ceil(cutoff * a1.norm() / area));
    std::vector<Eigen::Vector3d> out;
    out.reserve(static_cast<std::size_t>((2 * n1_max + 1) * (2 * n2_max + 1)));
    const double cutoff_sq = cutoff * cutoff;
    for (int n1 = -n1_max; n1 <= n1_max; ++n1) {
        for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
            if (!include_origin && n1 == 0 && n2 == 0) continue;
            const Eigen::Vector3d v =
                static_cast<double>(n1) * a1 + static_cast<double>(n2) * a2;
            if (v.squaredNorm() <= cutoff_sq) out.push_back(v);
        }
    }
    return out;
}

// Scaled complementary error function erfcx(x) = exp(x²)·erfc(x), evaluated
// without the intermediate exp(x²)-overflow / erfc-underflow of the naive
// product. The 2D-Ewald reciprocal sum below probes the central region
// |x| ≲ 10 for physically sized slabs (the recip cutoff bounds g/2α and the
// slab thickness bounds α·z_ij), so the direct product is used there and a
// 5-term asymptotic tail beyond x = 25 keeps the few far-out evaluations
// finite.
inline double erfcx_stable(double x) {
    if (x >= 25.0) {
        const double t = 1.0 / (x * x);
        return (1.0 / (x * std::sqrt(M_PI))) *
               (1.0 - 0.5 * t + 0.75 * t * t - 1.875 * t * t * t +
                6.5625 * t * t * t * t);
    }
    return std::exp(x * x) * std::erfc(x);
}

}  // namespace

double ewald_point_charge_energy(const Eigen::Matrix3d& lattice,
                                 const Eigen::Matrix3Xd& positions_cart,
                                 const Eigen::VectorXd& charges,
                                 const EwaldOptions& opts) {
    const auto N = positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_point_charge_energy: charges length must match "
            "positions_cart.cols()");
    }
    if (N == 0) return 0.0;

    const double V = std::abs(lattice.determinant());
    if (V < 1e-14) {
        throw std::invalid_argument(
            "ewald_point_charge_energy: singular lattice (zero volume)");
    }

    const Eigen::Matrix3d lattice_inverse = lattice.inverse();
    // Reciprocal lattice with 2π convention (a_i · b_j = 2π δ_ij).
    const Eigen::Matrix3d B = 2.0 * M_PI * lattice_inverse.transpose();

    Eigen::Matrix3Xd periodic_positions(3, N);
    for (Eigen::Index a = 0; a < N; ++a) {
        periodic_positions.col(a) = centered_periodic_representative(
            lattice, lattice_inverse, positions_cart.col(a));
    }

    // Resolve α and the two cutoffs.
    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : auto_alpha(opts.real_cutoff_bohr, opts.tolerance);
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : auto_recip_cutoff(alpha, opts.tolerance);

    // ---- Real-space sum ----------------------------------------------
    // (1/2) Σ_{A,B} Σ'_g Z_A Z_B · erfc(α|r_AB + g|) / |r_AB + g|
    //
    // Reduce each basis displacement to a bounded lattice representative,
    // then enumerate its pair-complete integer box of translations.
    const auto real_fractional_spans =
        pair_fractional_spans(lattice, real_cutoff);
    const Eigen::Index n_pairs = N * N;
    double e_real = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:e_real)
    for (Eigen::Index pair = 0; pair < n_pairs; ++pair) {
        const Eigen::Index a = pair / N;
        const Eigen::Index b = pair % N;
        const double charge_product = charges[a] * charges[b];
        for_each_pair_candidate(
            lattice, lattice_inverse,
            periodic_positions.col(a) - periodic_positions.col(b),
            real_fractional_spans,
            [&](const Eigen::Vector3d& d) {
                const double r = d.norm();
                if (r < 1.0e-14) return;   // (A=B, g=0) self-term
                if (r > real_cutoff) return;
                e_real += charge_product * std::erfc(alpha * r) / r;
            });
    }
    e_real *= 0.5;

    // ---- Reciprocal-space sum ----------------------------------------
    // (2π/V) Σ_{G≠0} |S(G)|² exp(−|G|²/(4α²)) / |G|²
    // with S(G) = Σ_A Z_A exp(i G · R_A).
    const auto G_list = enumerate_cells(B, recip_cutoff, /*include_origin=*/false);
    const int n_G = static_cast<int>(G_list.size());

    VIBEQC_DIAG("ewald", vibeqc::DiagLevel::VERBOSE,
        "alpha=%.4f  rcut=%.1f  gcut=%.1f  n_G=%d",
        alpha, real_cutoff, recip_cutoff, n_G);

    double e_recip = 0.0;
    const double inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    #pragma omp parallel for schedule(static) reduction(+:e_recip)
    for (int Gi = 0; Gi < n_G; ++Gi) {
        const Eigen::Vector3d& G = G_list[Gi];
        double S_re = 0.0, S_im = 0.0;
        for (Eigen::Index a = 0; a < N; ++a) {
            const double phase = G.dot(periodic_positions.col(a));
            S_re += charges[a] * std::cos(phase);
            S_im += charges[a] * std::sin(phase);
        }
        const double G2 = G.squaredNorm();
        const double S2 = S_re * S_re + S_im * S_im;
        e_recip += S2 * std::exp(-G2 * inv_4alpha2) / G2;
    }
    e_recip *= 2.0 * M_PI / V;

    // ---- Self-correction ---------------------------------------------
    // -(α/√π) Σ_A Z_A²
    double e_self = 0.0;
    for (Eigen::Index a = 0; a < N; ++a) {
        e_self += charges[a] * charges[a];
    }
    e_self *= -alpha / std::sqrt(M_PI);

    // ---- Background (jellium) correction -----------------------------
    // For a non-neutral cell the reciprocal-space sum's excluded G=0 term
    // corresponds to a uniform negative background of density Q/V where
    // Q = Σ_A Z_A. Its contribution is -(π / (2 α² V)) Q².
    const double Q = charges.sum();
    const double e_bg = -M_PI * Q * Q / (2.0 * alpha * alpha * V);

    return e_real + e_recip + e_self + e_bg;
}

double ewald_nuclear_repulsion(const PeriodicSystem& system,
                               const EwaldOptions& opts) {
    if (system.dim != 3) {
        throw std::invalid_argument(
            "ewald_nuclear_repulsion: 3D Ewald requires dim == 3. "
            "Use DIRECT_TRUNCATED for 1D / 2D (Ewald variants for those "
            "dimensionalities arrive in a later phase).");
    }
    const auto N = static_cast<Eigen::Index>(system.unit_cell.size());
    Eigen::Matrix3Xd positions(3, N);
    Eigen::VectorXd charges(N);
    for (Eigen::Index a = 0; a < N; ++a) {
        const auto& at = system.unit_cell[static_cast<std::size_t>(a)];
        positions(0, a) = at.xyz[0];
        positions(1, a) = at.xyz[1];
        positions(2, a) = at.xyz[2];
        charges[a] = static_cast<double>(at.Z);
    }
    return ewald_point_charge_energy(system.lattice, positions, charges, opts);
}

// ---------------------------------------------------------------------------
// Ewald point-charge gradient  ∂E/∂R_C
// ---------------------------------------------------------------------------
//
// Differentiates ewald_point_charge_energy term by term. With
//   φ(r) = erfc(α r)/r,
//   φ'(r) = d/dr[erfc(α r)/r] = −erfc(α r)/r² − (2α/√π) e^{−α²r²}/r,
// so  ∇_{R_C} φ(|d|) = [φ'(r)/r] · d  with
//   φ'(r)/r = −erfc(α r)/r³ − (2α/√π) e^{−α²r²}/r².
//
// Real-space:  E_real = ½ Σ_{A,B,g}' Z_A Z_B φ(|R_A−R_B+g|).
//   ∂E_real/∂R_C = Σ_{B≠C} Σ_g Z_C Z_B [φ'(r)/r] d,  d = R_C − R_B + g.
//   (The ½ and the A↔B symmetry combine to 1; the B=C, g≠0 self-image
//    pairs move R_C in both charges so r=|g| is fixed → zero gradient,
//    hence skipped.)
//
// Reciprocal:  E_recip = (2π/V) Σ_{G≠0} |S(G)|² w(G),  w = e^{−G²/4α²}/G²,
//   S(G) = Σ_A Z_A e^{iG·R_A}.
//   ∂|S|²/∂R_C = 2 Z_C (S_im cos(G·R_C) − S_re sin(G·R_C)) G.
//   ∂E_recip/∂R_C = (2π/V) Σ_{G≠0} w(G) · 2 Z_C
//                      (S_im cos(G·R_C) − S_re sin(G·R_C)) G.
//
// Self-energy −(α/√π) Σ Z_A² and background −πQ²/(2α²V) are both
// position-independent → zero gradient.
Eigen::Matrix3Xd ewald_point_charge_gradient(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& positions_cart,
    const Eigen::VectorXd& charges,
    const EwaldOptions& opts) {
    const auto N = positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_point_charge_gradient: charges length must match "
            "positions_cart.cols()");
    }
    Eigen::Matrix3Xd grad = Eigen::Matrix3Xd::Zero(3, std::max<Eigen::Index>(N, 1));
    if (N == 0) return Eigen::Matrix3Xd::Zero(3, 0);

    const double V = std::abs(lattice.determinant());
    if (V < 1e-14) {
        throw std::invalid_argument(
            "ewald_point_charge_gradient: singular lattice (zero volume)");
    }
    const Eigen::Matrix3d lattice_inverse = lattice.inverse();
    const Eigen::Matrix3d B = 2.0 * M_PI * lattice_inverse.transpose();

    Eigen::Matrix3Xd periodic_positions(3, N);
    for (Eigen::Index a = 0; a < N; ++a) {
        periodic_positions.col(a) = centered_periodic_representative(
            lattice, lattice_inverse, positions_cart.col(a));
    }

    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : auto_alpha(opts.real_cutoff_bohr, opts.tolerance);
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : auto_recip_cutoff(alpha, opts.tolerance);

    const double two_alpha_over_sqrtpi = 2.0 * alpha / std::sqrt(M_PI);

    // ---- Real-space gradient -----------------------------------------
    const auto real_fractional_spans =
        pair_fractional_spans(lattice, real_cutoff);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index c = 0; c < N; ++c) {
        const Eigen::Vector3d R_C = periodic_positions.col(c);
        const double Z_C = charges[c];
        Eigen::Vector3d gc = Eigen::Vector3d::Zero();
        for (Eigen::Index b = 0; b < N; ++b) {
            if (b == c) continue;   // self-image pairs → zero gradient
            const Eigen::Vector3d R_B = periodic_positions.col(b);
            const double Z_B = charges[b];
            for_each_pair_candidate(
                lattice, lattice_inverse, R_C - R_B,
                real_fractional_spans,
                [&](const Eigen::Vector3d& d) {
                    const double r2 = d.squaredNorm();
                    if (r2 < 1.0e-28) return;
                    const double r = std::sqrt(r2);
                    if (r > real_cutoff) return;
                    // φ'(r)/r = −erfc(αr)/r³ − (2α/√π)e^{−α²r²}/r²
                    const double dphi_over_r =
                        -std::erfc(alpha * r) / (r2 * r)
                        - two_alpha_over_sqrtpi
                            * std::exp(-alpha * alpha * r2) / r2;
                    gc += (Z_C * Z_B * dphi_over_r) * d;
                });
        }
        grad.col(c) += gc;
    }

    // ---- Reciprocal-space gradient -----------------------------------
    const auto G_list = enumerate_cells(B, recip_cutoff, /*include_origin=*/false);
    const int n_G = static_cast<int>(G_list.size());
    const double inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    const double recip_pref = 2.0 * M_PI / V;
    // Precompute S(G) once (independent of which charge we differentiate).
    std::vector<double> S_re(n_G), S_im(n_G), w(n_G);
    for (int Gi = 0; Gi < n_G; ++Gi) {
        const Eigen::Vector3d& G = G_list[Gi];
        double re = 0.0, im = 0.0;
        for (Eigen::Index a = 0; a < N; ++a) {
            const double phase = G.dot(periodic_positions.col(a));
            re += charges[a] * std::cos(phase);
            im += charges[a] * std::sin(phase);
        }
        S_re[Gi] = re;
        S_im[Gi] = im;
        w[Gi] = std::exp(-G.squaredNorm() * inv_4alpha2) / G.squaredNorm();
    }
    #pragma omp parallel for schedule(static)
    for (Eigen::Index c = 0; c < N; ++c) {
        const Eigen::Vector3d R_C = periodic_positions.col(c);
        const double Z_C = charges[c];
        Eigen::Vector3d gc = Eigen::Vector3d::Zero();
        for (int Gi = 0; Gi < n_G; ++Gi) {
            const Eigen::Vector3d& G = G_list[Gi];
            const double phase = G.dot(R_C);
            const double s = S_im[Gi] * std::cos(phase) - S_re[Gi] * std::sin(phase);
            gc += (recip_pref * w[Gi] * 2.0 * Z_C * s) * G;
        }
        grad.col(c) += gc;
    }

    return grad;
}

Eigen::MatrixXd ewald_nuclear_repulsion_gradient(const PeriodicSystem& system,
                                                 const EwaldOptions& opts) {
    if (system.dim != 3) {
        throw std::invalid_argument(
            "ewald_nuclear_repulsion_gradient: 3D Ewald requires dim == 3. "
            "Use DIRECT_TRUNCATED for 1D / 2D.");
    }
    const auto N = static_cast<Eigen::Index>(system.unit_cell.size());
    Eigen::Matrix3Xd positions(3, N);
    Eigen::VectorXd charges(N);
    for (Eigen::Index a = 0; a < N; ++a) {
        const auto& at = system.unit_cell[static_cast<std::size_t>(a)];
        positions(0, a) = at.xyz[0];
        positions(1, a) = at.xyz[1];
        positions(2, a) = at.xyz[2];
        charges[a] = static_cast<double>(at.Z);
    }
    // Return N × 3 (row = atom) to match nuclear_repulsion_gradient_per_cell.
    return ewald_point_charge_gradient(system.lattice, positions, charges, opts)
        .transpose();
}

// ---------------------------------------------------------------------------
// Ewald potential at a set of points
// ---------------------------------------------------------------------------

Eigen::VectorXd ewald_point_charge_potential(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& charge_positions_cart,
    const Eigen::VectorXd& charges,
    const Eigen::MatrixX3d& eval_points_cart,
    const EwaldOptions& opts,
    bool include_short_range) {

    const auto N = charge_positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_point_charge_potential: charges length must match "
            "charge_positions_cart.cols()");
    }
    const auto M = eval_points_cart.rows();
    if (M == 0) return Eigen::VectorXd::Zero(0);

    const double V = std::abs(lattice.determinant());
    if (V < 1e-14) {
        throw std::invalid_argument(
            "ewald_point_charge_potential: singular lattice");
    }
    const Eigen::Matrix3d lattice_inverse = lattice.inverse();
    const Eigen::Matrix3d B = 2.0 * M_PI * lattice_inverse.transpose();

    Eigen::Matrix3Xd periodic_charge_positions(3, N);
    for (Eigen::Index a = 0; a < N; ++a) {
        periodic_charge_positions.col(a) = centered_periodic_representative(
            lattice, lattice_inverse, charge_positions_cart.col(a));
    }
    Eigen::MatrixX3d periodic_eval_points(M, 3);
    for (Eigen::Index i = 0; i < M; ++i) {
        periodic_eval_points.row(i) = centered_periodic_representative(
            lattice, lattice_inverse,
            eval_points_cart.row(i).transpose()).transpose();
    }

    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : std::sqrt(-std::log(opts.tolerance)) / opts.real_cutoff_bohr;
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : 2.0 * alpha * std::sqrt(-std::log(opts.tolerance));

    Eigen::VectorXd v = Eigen::VectorXd::Zero(M);

    // ---- Short-range: erfc-screened 1/r from nearby charge images. ----
    //
    // For each evaluation point, loop over charges + their lattice images
    // within ``real_cutoff`` and accumulate Z · erfc(α r) / r.
    //
    // Users building matrix elements ⟨μ | v | ν⟩ on a grid should set
    // ``include_short_range = false``: the short-range part has a 1/r
    // singularity at each nucleus that no finite grid can integrate
    // accurately, and is better computed analytically via libint's
    // erfc_nuclear operator (see compute_nuclear_erfc_lattice).
    if (include_short_range) {
        const auto real_fractional_spans =
            pair_fractional_spans(lattice, real_cutoff);
        #pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < M; ++i) {
            const Eigen::Vector3d r_i =
                periodic_eval_points.row(i).transpose();
            double v_short = 0.0;
            for (Eigen::Index a = 0; a < N; ++a) {
                const Eigen::Vector3d R_A =
                    periodic_charge_positions.col(a);
                const double Z_A = charges[a];
                for_each_pair_candidate(
                    lattice, lattice_inverse, r_i - R_A,
                    real_fractional_spans,
                    [&](const Eigen::Vector3d& d) {
                        const double r = d.norm();
                        if (r < 1e-14) return;
                        if (r > real_cutoff) return;
                        v_short += Z_A * std::erfc(alpha * r) / r;
                    });
            }
            v[i] = v_short;
        }
    }

    // ---- Long-range: reciprocal-space sum over G-vectors. ----
    //
    //   v_long(r) = (4π / V) Σ_{G ≠ 0} ρ̃(G) · exp(i G · r) ·
    //                                    exp(−|G|² / (4 α²)) / |G|²
    //
    // with ρ̃(G) = Σ_A Z_A exp(−i G · R_A).
    //
    // Precompute ρ̃(G) once — it doesn't depend on the evaluation points
    // — then accumulate the per-point contribution. For N_G ~ 1000 and
    // M ~ 50 000 this is ~5·10^7 complex operations per SCF setup call;
    // tractable without FFT. FFTW-accelerated variant lands with the
    // ERI Ewald in a later sub-phase.
    const auto G_list = enumerate_cells(B, recip_cutoff,
                                        /*include_origin=*/false);
    const int n_G = static_cast<int>(G_list.size());

    std::vector<double> rho_re(n_G), rho_im(n_G), w(n_G);
    const double inv_4alpha2 = 1.0 / (4.0 * alpha * alpha);
    for (int Gi = 0; Gi < n_G; ++Gi) {
        const Eigen::Vector3d& G = G_list[Gi];
        double re = 0.0, im = 0.0;
        for (Eigen::Index a = 0; a < N; ++a) {
            const double phase = G.dot(periodic_charge_positions.col(a));
            re += charges[a] * std::cos(phase);
            im -= charges[a] * std::sin(phase);    // ρ̃(G) has e^{−iG·R}
        }
        rho_re[Gi] = re;
        rho_im[Gi] = im;
        w[Gi] = std::exp(-G.squaredNorm() * inv_4alpha2) / G.squaredNorm();
    }
    const double prefactor = 4.0 * M_PI / V;

    #pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < M; ++i) {
        const Eigen::Vector3d r_i =
            periodic_eval_points.row(i).transpose();
        double acc = 0.0;
        for (int Gi = 0; Gi < n_G; ++Gi) {
            const Eigen::Vector3d& G = G_list[Gi];
            const double phase = G.dot(r_i);
            const double c = std::cos(phase);
            const double s = std::sin(phase);
            // Re[ρ̃(G) · e^{i G·r}] = rho_re · c − rho_im · s
            acc += w[Gi] * (rho_re[Gi] * c - rho_im[Gi] * s);
        }
        v[i] += prefactor * acc;
    }

    // ---- Background (jellium) term for non-neutral cells. ----
    //
    // The reciprocal-space sum excludes G = 0; its limit is taken by
    // imposing charge neutrality via a uniform compensating background
    // of density −Q/V where Q = Σ_A Z_A. The uniform potential from
    // that background is
    //
    //   v_bg = −(π / (α² V)) · Q
    //
    // applied at every evaluation point. For a neutral cell (Q = 0) it
    // vanishes automatically.
    const double Q = charges.sum();
    const double v_bg = -M_PI * Q / (alpha * alpha * V);
    v.array() += v_bg;

    return v;
}

Eigen::VectorXd ewald_nuclear_potential(const PeriodicSystem& system,
                                        const Eigen::MatrixX3d& eval_points_cart,
                                        const EwaldOptions& opts,
                                        bool include_short_range) {
    if (system.dim != 3) {
        throw std::invalid_argument(
            "ewald_nuclear_potential: 3D Ewald requires dim == 3.");
    }
    const auto N = static_cast<Eigen::Index>(system.unit_cell.size());
    Eigen::Matrix3Xd positions(3, N);
    Eigen::VectorXd charges(N);
    for (Eigen::Index a = 0; a < N; ++a) {
        const auto& at = system.unit_cell[static_cast<std::size_t>(a)];
        positions(0, a) = at.xyz[0];
        positions(1, a) = at.xyz[1];
        positions(2, a) = at.xyz[2];
        charges[a] = static_cast<double>(at.Z);
    }
    // Electronic nuclear-attraction potential: V_nuc(r) = -Σ Z_A / |r-R_A|
    Eigen::VectorXd v = ewald_point_charge_potential(
        system.lattice, positions, charges, eval_points_cart, opts,
        include_short_range);
    v.array() *= -1.0;
    return v;
}

// ---------------------------------------------------------------------------
// Rigorous 2D (slab) Ewald summation — Parry / de Leeuw–Perram–Smith
// ---------------------------------------------------------------------------
//
// See ewald.hpp for the full term-by-term formula and references. The four
// contributions (E_real, E_recip[g≠0], E_recip[g=0], E_self) sum to an
// α-independent total; α only trades real- vs reciprocal-space work. Only a
// charge-neutral cell is well-defined in 2D, so a non-neutral cell is
// rejected.
double ewald_2d_point_charge_energy(const Eigen::Matrix3d& lattice,
                                    const Eigen::Matrix3Xd& positions_cart,
                                    const Eigen::VectorXd& charges,
                                    const EwaldOptions& opts) {
    const auto N = positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_energy: charges length must match "
            "positions_cart.cols()");
    }
    if (N == 0) return 0.0;

    // A net-charged 2D-periodic plane has a divergent Madelung energy — there
    // is no finite jellium regularisation as in 3D. Reject rather than lie.
    const double Q = charges.sum();
    if (std::abs(Q) > 1e-8) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_energy: 2D (slab) Ewald requires a "
            "charge-neutral cell; got net charge Q = " + std::to_string(Q) +
            ". A net-charged periodic plane has a divergent electrostatic "
            "energy.");
    }

    // In-plane lattice vectors (cols 0,1) and the slab normal n̂ = (a1×a2)/A.
    const Eigen::Vector3d a1 = lattice.col(0);
    const Eigen::Vector3d a2 = lattice.col(1);
    const Eigen::Vector3d cross = a1.cross(a2);
    const double A = cross.norm();
    if (A < 1e-14) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_energy: degenerate in-plane lattice "
            "(lattice columns 0 and 1 are collinear)");
    }
    const Eigen::Vector3d nhat = cross / A;

    // 2D reciprocal lattice vectors b1,b2 (in-plane, b_i·a_j = 2π δ_ij):
    //   b1 = 2π (a2 × n̂)/A,   b2 = 2π (n̂ × a1)/A.
    const Eigen::Vector3d b1 = 2.0 * M_PI * a2.cross(nhat) / A;
    const Eigen::Vector3d b2 = 2.0 * M_PI * nhat.cross(a1) / A;

    // α and the two cutoffs — same auto-selection convention as the 3D sum.
    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : auto_alpha(opts.real_cutoff_bohr, opts.tolerance);
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : auto_recip_cutoff(alpha, opts.tolerance);

    // Per-atom normal coordinate z_i = r_i · n̂ (the in-plane part enters only
    // through cos(g·ρ_ij) = cos(g·r_ij), since g·n̂ = 0).
    Eigen::VectorXd z(N);
    for (Eigen::Index i = 0; i < N; ++i) z[i] = positions_cart.col(i).dot(nhat);

    // ---- Real-space sum: ½ Σ_{i,j} Σ'_n q_i q_j erfc(α|r_ij+n|)/|r_ij+n| --
    // Identical erfc/r kernel as the 3D sum, but n runs over the 2D lattice.
    const auto cells =
        enumerate_cells_2d(a1, a2, real_cutoff, /*include_origin=*/true);
    const int n_cell = static_cast<int>(cells.size());
    double e_real = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+:e_real)
    for (int ci = 0; ci < n_cell; ++ci) {
        const Eigen::Vector3d& n = cells[ci];
        for (Eigen::Index i = 0; i < N; ++i) {
            const Eigen::Vector3d R_i = positions_cart.col(i);
            const double q_i = charges[i];
            for (Eigen::Index j = 0; j < N; ++j) {
                const Eigen::Vector3d d = R_i - positions_cart.col(j) + n;
                const double r = d.norm();
                if (r < 1.0e-14) continue;        // (i=j, n=0) self-term
                if (r > real_cutoff) continue;
                e_real += q_i * charges[j] * std::erfc(alpha * r) / r;
            }
        }
    }
    e_real *= 0.5;

    // ---- Reciprocal-space sum, g ≠ 0 ---------------------------------------
    //   (π/2A) Σ_{g≠0} (1/g) Σ_{i,j} q_i q_j cos(g·ρ_ij) ×
    //          [ e^{+g z_ij} erfc(g/2α + α z_ij) + e^{−g z_ij} erfc(g/2α − α z_ij) ]
    // computed in the overflow-safe form
    //   e^{±g z} erfc(g/2α ± α z) = e^{−(g/2α)² − α²z²} erfcx(g/2α ± α z).
    // The z_ij dependence couples i and j, so this is O(N²) per g-vector
    // (no |S(g)|² factorisation as in 3D) — fine for the few atoms per cell.
    const auto g_list =
        enumerate_cells_2d(b1, b2, recip_cutoff, /*include_origin=*/false);
    const int n_g = static_cast<int>(g_list.size());
    const double inv_2alpha = 1.0 / (2.0 * alpha);
    const double alpha2 = alpha * alpha;
    double e_recip = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:e_recip)
    for (int gi = 0; gi < n_g; ++gi) {
        const Eigen::Vector3d& g = g_list[gi];
        const double gnorm = g.norm();
        const double g_over_2a = gnorm * inv_2alpha;
        const double g_over_2a_sq = g_over_2a * g_over_2a;
        double acc = 0.0;
        for (Eigen::Index i = 0; i < N; ++i) {
            const Eigen::Vector3d R_i = positions_cart.col(i);
            const double q_i = charges[i];
            const double z_i = z[i];
            for (Eigen::Index j = 0; j < N; ++j) {
                const double z_ij = z_i - z[j];
                const double cosg = std::cos(g.dot(R_i - positions_cart.col(j)));
                const double pref =
                    std::exp(-g_over_2a_sq - alpha2 * z_ij * z_ij);
                const double bracket =
                    erfcx_stable(g_over_2a + alpha * z_ij) +
                    erfcx_stable(g_over_2a - alpha * z_ij);
                acc += q_i * charges[j] * cosg * pref * bracket;
            }
        }
        e_recip += acc / gnorm;
    }
    e_recip *= M_PI / (2.0 * A);

    // ---- Reciprocal-space sum, g = 0 (the term with no 3D analogue) --------
    //   −(π/A) Σ_{i,j} q_i q_j [ z_ij erf(α z_ij) + (1/(α√π)) e^{−α²z_ij²} ]
    double e_g0 = 0.0;
    const double inv_alpha_sqrtpi = 1.0 / (alpha * std::sqrt(M_PI));
    for (Eigen::Index i = 0; i < N; ++i) {
        const double q_i = charges[i];
        const double z_i = z[i];
        for (Eigen::Index j = 0; j < N; ++j) {
            const double z_ij = z_i - z[j];
            e_g0 += q_i * charges[j] *
                    (z_ij * std::erf(alpha * z_ij) +
                     inv_alpha_sqrtpi * std::exp(-alpha2 * z_ij * z_ij));
        }
    }
    e_g0 *= -M_PI / A;

    // ---- Self-energy: −(α/√π) Σ_i q_i² -------------------------------------
    double e_self = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) e_self += charges[i] * charges[i];
    e_self *= -alpha / std::sqrt(M_PI);

    return e_real + e_recip + e_g0 + e_self;
}

// ---------------------------------------------------------------------------
// Rigorous 2D (slab) Ewald electrostatic potential at a set of points
// ---------------------------------------------------------------------------
//
// See ewald.hpp for the term-by-term formula. The potential is the
// functional derivative of the energy, so the reciprocal and g=0 terms carry
// twice the energy's prefactor (π/A and −2π/A vs π/2A and −π/A). At a point
// coincident with a charge the short-range 1/r self term is skipped (as in
// the 3D ewald_point_charge_potential); the smeared self-potential 2α/√π is
// the caller's responsibility when forming a site (Madelung) potential.
Eigen::VectorXd ewald_2d_point_charge_potential(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& charge_positions_cart,
    const Eigen::VectorXd& charges,
    const Eigen::MatrixX3d& eval_points_cart,
    const EwaldOptions& opts,
    bool include_short_range) {
    const auto N = charge_positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_potential: charges length must match "
            "charge_positions_cart.cols()");
    }
    const auto M = eval_points_cart.rows();
    if (M == 0) return Eigen::VectorXd::Zero(0);

    const double Q = charges.sum();
    if (std::abs(Q) > 1e-8) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_potential: 2D (slab) Ewald requires a "
            "charge-neutral cell; got net charge Q = " + std::to_string(Q) +
            ". A net-charged periodic plane has a divergent potential.");
    }

    const Eigen::Vector3d a1 = lattice.col(0);
    const Eigen::Vector3d a2 = lattice.col(1);
    const Eigen::Vector3d cross = a1.cross(a2);
    const double A = cross.norm();
    if (A < 1e-14) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_potential: degenerate in-plane lattice "
            "(lattice columns 0 and 1 are collinear)");
    }
    const Eigen::Vector3d nhat = cross / A;
    const Eigen::Vector3d b1 = 2.0 * M_PI * a2.cross(nhat) / A;
    const Eigen::Vector3d b2 = 2.0 * M_PI * nhat.cross(a1) / A;

    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : auto_alpha(opts.real_cutoff_bohr, opts.tolerance);
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : auto_recip_cutoff(alpha, opts.tolerance);
    const double inv_2alpha = 1.0 / (2.0 * alpha);
    const double alpha2 = alpha * alpha;
    const double inv_alpha_sqrtpi = 1.0 / (alpha * std::sqrt(M_PI));

    // Per-source normal coordinate z_j = r_j · n̂.
    Eigen::VectorXd zc(N);
    for (Eigen::Index j = 0; j < N; ++j) zc[j] = charge_positions_cart.col(j).dot(nhat);

    const auto g_list =
        enumerate_cells_2d(b1, b2, recip_cutoff, /*include_origin=*/false);
    const int n_g = static_cast<int>(g_list.size());
    const auto cells =
        enumerate_cells_2d(a1, a2, real_cutoff, /*include_origin=*/true);
    const int n_cell = static_cast<int>(cells.size());

    Eigen::VectorXd v = Eigen::VectorXd::Zero(M);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index p = 0; p < M; ++p) {
        const Eigen::Vector3d r = eval_points_cart.row(p).transpose();
        const double zr = r.dot(nhat);

        // Short-range erfc/r over the 2D lattice (skip coincident sources).
        double v_short = 0.0;
        if (include_short_range) {
            for (int ci = 0; ci < n_cell; ++ci) {
                const Eigen::Vector3d& n = cells[ci];
                for (Eigen::Index j = 0; j < N; ++j) {
                    const Eigen::Vector3d d =
                        r - charge_positions_cart.col(j) - n;
                    const double rr = d.norm();
                    if (rr < 1.0e-14) continue;
                    if (rr > real_cutoff) continue;
                    v_short += charges[j] * std::erfc(alpha * rr) / rr;
                }
            }
        }

        // Reciprocal (g≠0): the functional-derivative form (π/A prefactor).
        double v_recip = 0.0;
        for (int gi = 0; gi < n_g; ++gi) {
            const Eigen::Vector3d& g = g_list[gi];
            const double gnorm = g.norm();
            const double g_over_2a = gnorm * inv_2alpha;
            const double g_over_2a_sq = g_over_2a * g_over_2a;
            double acc = 0.0;
            for (Eigen::Index j = 0; j < N; ++j) {
                const double dz = zr - zc[j];
                const double cosg =
                    std::cos(g.dot(r - charge_positions_cart.col(j)));
                const double pref = std::exp(-g_over_2a_sq - alpha2 * dz * dz);
                const double bracket = erfcx_stable(g_over_2a + alpha * dz) +
                                       erfcx_stable(g_over_2a - alpha * dz);
                acc += charges[j] * cosg * pref * bracket;
            }
            v_recip += acc / gnorm;
        }
        v_recip *= M_PI / A;

        // g = 0 slab term (−2π/A prefactor).
        double v_g0 = 0.0;
        for (Eigen::Index j = 0; j < N; ++j) {
            const double dz = zr - zc[j];
            v_g0 += charges[j] *
                    (dz * std::erf(alpha * dz) +
                     inv_alpha_sqrtpi * std::exp(-alpha2 * dz * dz));
        }
        v_g0 *= -2.0 * M_PI / A;

        v[p] = v_short + v_recip + v_g0;
    }
    return v;
}

// ---------------------------------------------------------------------------
// Background-neutralised 2D (slab) Ewald energy
// ---------------------------------------------------------------------------
//
// See ewald.hpp for the full rationale. This is ewald_2d_point_charge_energy
// with the neutrality guard removed and a uniform in-plane neutralising sheet
// of total charge q_b = −Σq_i added at normal coordinate z_background. The
// four bare-point terms (real, recip g≠0, the slab g=0 term f(z), and self)
// are computed exactly as in the bare routine; the sheet — being a smooth
// in-plane-uniform plane — needs no Ewald screening and contributes only its
// exact bare field, the g=0 correction
//   ΔE_sheet = −(2π/A) q_b Σ_i q_i |z_i − z_b|.
double ewald_2d_point_charge_energy_with_background(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& positions_cart,
    const Eigen::VectorXd& charges,
    double z_background,
    const EwaldOptions& opts) {
    const auto N = positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_energy_with_background: charges length must "
            "match positions_cart.cols()");
    }
    if (N == 0) return 0.0;

    // NO neutrality guard: a net-charged ``charges`` is the intended input —
    // the sheet of charge q_b = −Σq_i neutralises it (the combined
    // {points + sheet} system is neutral, so the result is the physical,
    // α-invariant Madelung energy of a charged slab in a uniform background).

    const Eigen::Vector3d a1 = lattice.col(0);
    const Eigen::Vector3d a2 = lattice.col(1);
    const Eigen::Vector3d cross = a1.cross(a2);
    const double A = cross.norm();
    if (A < 1e-14) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_energy_with_background: degenerate in-plane "
            "lattice (lattice columns 0 and 1 are collinear)");
    }
    const Eigen::Vector3d nhat = cross / A;
    const Eigen::Vector3d b1 = 2.0 * M_PI * a2.cross(nhat) / A;
    const Eigen::Vector3d b2 = 2.0 * M_PI * nhat.cross(a1) / A;

    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : auto_alpha(opts.real_cutoff_bohr, opts.tolerance);
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : auto_recip_cutoff(alpha, opts.tolerance);

    Eigen::VectorXd z(N);
    for (Eigen::Index i = 0; i < N; ++i) z[i] = positions_cart.col(i).dot(nhat);

    // ---- Real-space sum (points only; the uniform sheet has no images) -----
    const auto cells =
        enumerate_cells_2d(a1, a2, real_cutoff, /*include_origin=*/true);
    const int n_cell = static_cast<int>(cells.size());
    double e_real = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+:e_real)
    for (int ci = 0; ci < n_cell; ++ci) {
        const Eigen::Vector3d& n = cells[ci];
        for (Eigen::Index i = 0; i < N; ++i) {
            const Eigen::Vector3d R_i = positions_cart.col(i);
            const double q_i = charges[i];
            for (Eigen::Index j = 0; j < N; ++j) {
                const Eigen::Vector3d d = R_i - positions_cart.col(j) + n;
                const double r = d.norm();
                if (r < 1.0e-14) continue;
                if (r > real_cutoff) continue;
                e_real += q_i * charges[j] * std::erfc(alpha * r) / r;
            }
        }
    }
    e_real *= 0.5;

    // ---- Reciprocal sum, g ≠ 0 (points only; the sheet has no g ≠ 0) -------
    const auto g_list =
        enumerate_cells_2d(b1, b2, recip_cutoff, /*include_origin=*/false);
    const int n_g = static_cast<int>(g_list.size());
    const double inv_2alpha = 1.0 / (2.0 * alpha);
    const double alpha2 = alpha * alpha;
    double e_recip = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:e_recip)
    for (int gi = 0; gi < n_g; ++gi) {
        const Eigen::Vector3d& g = g_list[gi];
        const double gnorm = g.norm();
        const double g_over_2a = gnorm * inv_2alpha;
        const double g_over_2a_sq = g_over_2a * g_over_2a;
        double acc = 0.0;
        for (Eigen::Index i = 0; i < N; ++i) {
            const Eigen::Vector3d R_i = positions_cart.col(i);
            const double q_i = charges[i];
            const double z_i = z[i];
            for (Eigen::Index j = 0; j < N; ++j) {
                const double z_ij = z_i - z[j];
                const double cosg = std::cos(g.dot(R_i - positions_cart.col(j)));
                const double pref =
                    std::exp(-g_over_2a_sq - alpha2 * z_ij * z_ij);
                const double bracket =
                    erfcx_stable(g_over_2a + alpha * z_ij) +
                    erfcx_stable(g_over_2a - alpha * z_ij);
                acc += q_i * charges[j] * cosg * pref * bracket;
            }
        }
        e_recip += acc / gnorm;
    }
    e_recip *= M_PI / (2.0 * A);

    // ---- Reciprocal sum, g = 0: point–point part --------------------------
    //   −(π/A) Σ_{i,j} q_i q_j [ z_ij erf(α z_ij) + (1/(α√π)) e^{−α²z_ij²} ]
    double e_g0 = 0.0;
    const double inv_alpha_sqrtpi = 1.0 / (alpha * std::sqrt(M_PI));
    for (Eigen::Index i = 0; i < N; ++i) {
        const double q_i = charges[i];
        const double z_i = z[i];
        for (Eigen::Index j = 0; j < N; ++j) {
            const double z_ij = z_i - z[j];
            e_g0 += q_i * charges[j] *
                    (z_ij * std::erf(alpha * z_ij) +
                     inv_alpha_sqrtpi * std::exp(-alpha2 * z_ij * z_ij));
        }
    }
    e_g0 *= -M_PI / A;

    // ---- g = 0 neutralising-sheet correction (exact uniform-sheet field) ---
    // A uniform in-plane sheet of total charge q_b = −Σq_i per cell at normal
    // coordinate z_b = z_background produces the exact electrostatic potential
    //   φ_sheet(z) = −(2π/A) q_b |z − z_b|
    // (Gauss's law for an infinite charged plane, surface density σ = q_b/A;
    // the field is the constant 2π|σ| pointing away from the plane). Being
    // smooth and in-plane-uniform, the sheet needs NO Ewald screening: it
    // contributes nothing to the real-space lattice sum (it has no discrete
    // images), nothing to the g ≠ 0 reciprocal sum (its in-plane FT is a δ at
    // g = 0), and nothing to the point self-energy. Its g = 0 contribution is
    // therefore the *bare* |z| field, NOT the screened f(z) kernel the point
    // pairs use — crucially, because the bare four-term above is already
    // α- and origin-invariant for a net-charged block (the finite slab g = 0
    // term f(z) regularises it without a background), so the sheet must add
    // the physical, α-independent field, not a second screened term.
    //
    // The point–sheet interaction (a cross-group term, counted once) plus the
    // sheet's zero self-energy (|z| = 0 at its own plane) is
    //   ΔE_sheet = −(2π/A) q_b Σ_i q_i |z_i − z_b|.
    // It is α-invariant, vanishes on neutral input (q_b = 0), and is
    // z_b-DEPENDENT for a charged block — the physical dependence on where the
    // neutralising background sits, which the BIPOLE surface gradient needs.
    // (Parry, Surf. Sci. 49, 433 (1975); de Leeuw & Perram, Mol. Phys. 37,
    // 1313 (1979).)
    const double q_b = -charges.sum();
    double sheet_dot = 0.0;  // Σ_i q_i |z_i − z_b|
    for (Eigen::Index i = 0; i < N; ++i)
        sheet_dot += charges[i] * std::abs(z[i] - z_background);
    const double e_g0_sheet = -(2.0 * M_PI / A) * q_b * sheet_dot;

    // ---- Self-energy: points only (the smooth sheet has no point self) -----
    double e_self = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) e_self += charges[i] * charges[i];
    e_self *= -alpha / std::sqrt(M_PI);

    return e_real + e_recip + e_g0 + e_g0_sheet + e_self;
}

// ---------------------------------------------------------------------------
// Background-neutralised 2D (slab) Ewald gradient  ∂E/∂R_C
// ---------------------------------------------------------------------------
//
// Differentiates ewald_2d_point_charge_energy_with_background term by term
// over the SAME truncated sums (identical α, cutoffs, cell enumeration).
// (Parry, Surf. Sci. 49, 433 (1975); de Leeuw & Perram, Mol. Phys. 37,
// 1313 (1979).)
//
// Real-space: identical in structure to the 3D ewald_point_charge_gradient
// real term, over the 2D cell list:
//   ∂E_real/∂R_C = Σ_{j≠C} Σ_n q_C q_j [φ'(r)/r] d,  d = R_C − R_j + n,
//   φ'(r)/r = −erfc(α r)/r³ − (2α/√π) e^{−α²r²}/r².
//   (Self-image pairs j=C, n≠0 have fixed r = |n| → zero gradient.)
//
// Reciprocal g ≠ 0 (the z-resolved Parry kernel): with
//   F(g,z) = e^{gz} erfc(g/2α + αz) + e^{−gz} erfc(g/2α − αz)
//          = e^{−(g/2α)² − α²z²} [erfcx(g/2α + αz) + erfcx(g/2α − αz)]
// the energy term is (π/2A) Σ_{g≠0} (1/g) Σ_{i,j} q_i q_j cos(g·r_ij) F(g,z_ij).
// F is even in z and cos even in r_ij, so the i=C and j=C contributions are
// equal and
//   ∂E_recip/∂R_C = (π/A) Σ_{g≠0} (1/g) Σ_j q_C q_j
//       [ −sin(g·(R_C − R_j)) F(g, z_Cj) · g
//         + cos(g·(R_C − R_j)) F'(g, z_Cj) · n̂ ].
// The z-derivative: d/dz[e^{±gz} erfc(g/2α ± αz)] = ±g e^{±gz} erfc(...)
// ∓ (2α/√π) e^{±gz} e^{−(g/2α±αz)²}, and both Gaussian factors reduce to
// the same e^{−(g/2α)² − α²z²} (since (g/2α ± αz)² = (g/2α)² ± gz + α²z²),
// so they cancel exactly between the two branches, leaving
//   F'(g,z) = g [e^{gz} erfc(g/2α + αz) − e^{−gz} erfc(g/2α − αz)]
//           = g e^{−(g/2α)² − α²z²} [erfcx(g/2α + αz) − erfcx(g/2α − αz)].
// The g-vector part is the in-plane force; the F' part is the normal force.
//
// Reciprocal g = 0: f(z) = z erf(αz) + e^{−α²z²}/(α√π) has f'(z) = erf(αz)
// (the two Gaussian terms cancel), so
//   ∂E_g0/∂R_C = −(2π/A) Σ_j q_C q_j erf(α z_Cj) · n̂.
//
// Sheet: ΔE_sheet = −(2π/A) q_b Σ_i q_i |z_i − z_b| gives
//   ∂ΔE_sheet/∂R_C = −(2π/A) q_b q_C sgn(z_C − z_b) · n̂
// (subgradient sgn(0) = 0 at the kink; q_b and z_b are position-independent).
//
// Self-energy −(α/√π) Σ q_i²: position-independent → zero.
Eigen::Matrix3Xd ewald_2d_point_charge_gradient_with_background(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& positions_cart,
    const Eigen::VectorXd& charges,
    double z_background,
    const EwaldOptions& opts) {
    const auto N = positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_gradient_with_background: charges length "
            "must match positions_cart.cols()");
    }
    if (N == 0) return Eigen::Matrix3Xd::Zero(3, 0);

    const Eigen::Vector3d a1 = lattice.col(0);
    const Eigen::Vector3d a2 = lattice.col(1);
    const Eigen::Vector3d cross = a1.cross(a2);
    const double A = cross.norm();
    if (A < 1e-14) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_gradient_with_background: degenerate "
            "in-plane lattice (lattice columns 0 and 1 are collinear)");
    }
    const Eigen::Vector3d nhat = cross / A;
    const Eigen::Vector3d b1 = 2.0 * M_PI * a2.cross(nhat) / A;
    const Eigen::Vector3d b2 = 2.0 * M_PI * nhat.cross(a1) / A;

    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : auto_alpha(opts.real_cutoff_bohr, opts.tolerance);
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : auto_recip_cutoff(alpha, opts.tolerance);
    const double inv_2alpha = 1.0 / (2.0 * alpha);
    const double alpha2 = alpha * alpha;
    const double two_alpha_over_sqrtpi = 2.0 * alpha / std::sqrt(M_PI);

    Eigen::VectorXd z(N);
    for (Eigen::Index i = 0; i < N; ++i) z[i] = positions_cart.col(i).dot(nhat);

    Eigen::Matrix3Xd grad = Eigen::Matrix3Xd::Zero(3, N);

    // ---- Real-space gradient (2D cell list) ---------------------------
    const auto cells =
        enumerate_cells_2d(a1, a2, real_cutoff, /*include_origin=*/true);
    const int n_cell = static_cast<int>(cells.size());
    #pragma omp parallel for schedule(static)
    for (Eigen::Index c = 0; c < N; ++c) {
        const Eigen::Vector3d R_C = positions_cart.col(c);
        const double q_C = charges[c];
        Eigen::Vector3d gc = Eigen::Vector3d::Zero();
        for (Eigen::Index j = 0; j < N; ++j) {
            if (j == c) continue;   // self-image pairs → zero gradient
            const Eigen::Vector3d R_j = positions_cart.col(j);
            const double q_j = charges[j];
            for (int ci = 0; ci < n_cell; ++ci) {
                const Eigen::Vector3d d = R_C - R_j + cells[ci];
                const double r2 = d.squaredNorm();
                if (r2 < 1.0e-28) continue;
                const double r = std::sqrt(r2);
                if (r > real_cutoff) continue;
                // φ'(r)/r = −erfc(α r)/r³ − (2α/√π) e^{−α²r²}/r²
                const double dphi_over_r =
                    -std::erfc(alpha * r) / (r2 * r)
                    - two_alpha_over_sqrtpi * std::exp(-alpha2 * r2) / r2;
                gc += (q_C * q_j * dphi_over_r) * d;
            }
        }
        grad.col(c) += gc;
    }

    // ---- Reciprocal g ≠ 0 gradient (z-resolved) -----------------------
    const auto g_list =
        enumerate_cells_2d(b1, b2, recip_cutoff, /*include_origin=*/false);
    const int n_g = static_cast<int>(g_list.size());
    const double recip_pref = M_PI / A;   // (π/2A) × the i↔j symmetry factor 2
    #pragma omp parallel for schedule(static)
    for (Eigen::Index c = 0; c < N; ++c) {
        const Eigen::Vector3d R_C = positions_cart.col(c);
        const double q_C = charges[c];
        Eigen::Vector3d gc = Eigen::Vector3d::Zero();
        for (int gi = 0; gi < n_g; ++gi) {
            const Eigen::Vector3d& g = g_list[gi];
            const double gnorm = g.norm();
            const double g_over_2a = gnorm * inv_2alpha;
            const double g_over_2a_sq = g_over_2a * g_over_2a;
            double acc_inplane = 0.0;   // Σ_j q_j sin(g·d) F(g,z)
            double acc_normal = 0.0;    // Σ_j q_j cos(g·d) F'(g,z)
            for (Eigen::Index j = 0; j < N; ++j) {
                if (j == c) continue;   // sin(0) = 0 and F'(0) = 0
                const double z_cj = z[c] - z[j];
                const double phase = g.dot(R_C - positions_cart.col(j));
                const double pref =
                    std::exp(-g_over_2a_sq - alpha2 * z_cj * z_cj);
                const double ep = erfcx_stable(g_over_2a + alpha * z_cj);
                const double em = erfcx_stable(g_over_2a - alpha * z_cj);
                acc_inplane +=
                    charges[j] * std::sin(phase) * pref * (ep + em);
                acc_normal +=
                    charges[j] * std::cos(phase) * gnorm * pref * (ep - em);
            }
            gc += (recip_pref * q_C / gnorm) *
                  (-acc_inplane * g + acc_normal * nhat);
        }
        grad.col(c) += gc;
    }

    // ---- Reciprocal g = 0 gradient (normal force only) ----------------
    const double g0_pref = -2.0 * M_PI / A;   // −(π/A) × symmetry factor 2
    for (Eigen::Index c = 0; c < N; ++c) {
        double acc = 0.0;
        for (Eigen::Index j = 0; j < N; ++j) {
            acc += charges[j] * std::erf(alpha * (z[c] - z[j]));
        }
        grad.col(c) += (g0_pref * charges[c] * acc) * nhat;
    }

    // ---- Neutralising-sheet gradient (normal force only) --------------
    const double q_b = -charges.sum();
    const double sheet_pref = -(2.0 * M_PI / A) * q_b;
    for (Eigen::Index c = 0; c < N; ++c) {
        const double dz = z[c] - z_background;
        const double sgn = (dz > 0.0) ? 1.0 : ((dz < 0.0) ? -1.0 : 0.0);
        grad.col(c) += (sheet_pref * charges[c] * sgn) * nhat;
    }

    // Self-energy: position-independent → no contribution.
    return grad;
}

// ---------------------------------------------------------------------------
// Background-neutralised 2D (slab) Ewald electrostatic potential
// ---------------------------------------------------------------------------
//
// ewald_2d_point_charge_potential with the neutrality guard removed and the
// uniform neutralising sheet's exact (un-screened) g = 0 field added:
//   Δv_sheet(r) = −(2π/A) q_b |z_r − z_b|,  q_b = −Σq_i.
Eigen::VectorXd ewald_2d_point_charge_potential_with_background(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& charge_positions_cart,
    const Eigen::VectorXd& charges,
    double z_background,
    const Eigen::MatrixX3d& eval_points_cart,
    const EwaldOptions& opts,
    bool include_short_range) {
    const auto N = charge_positions_cart.cols();
    if (charges.size() != N) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_potential_with_background: charges length "
            "must match charge_positions_cart.cols()");
    }
    const auto M = eval_points_cart.rows();
    if (M == 0) return Eigen::VectorXd::Zero(0);

    // NO neutrality guard (see the energy variant): the sheet neutralises the
    // sources, so the combined potential is well-defined for net-charged input.

    const Eigen::Vector3d a1 = lattice.col(0);
    const Eigen::Vector3d a2 = lattice.col(1);
    const Eigen::Vector3d cross = a1.cross(a2);
    const double A = cross.norm();
    if (A < 1e-14) {
        throw std::invalid_argument(
            "ewald_2d_point_charge_potential_with_background: degenerate "
            "in-plane lattice (lattice columns 0 and 1 are collinear)");
    }
    const Eigen::Vector3d nhat = cross / A;
    const Eigen::Vector3d b1 = 2.0 * M_PI * a2.cross(nhat) / A;
    const Eigen::Vector3d b2 = 2.0 * M_PI * nhat.cross(a1) / A;

    const double alpha = (opts.alpha > 0.0)
        ? opts.alpha
        : auto_alpha(opts.real_cutoff_bohr, opts.tolerance);
    const double real_cutoff = opts.real_cutoff_bohr;
    const double recip_cutoff = (opts.recip_cutoff_bohr_inv > 0.0)
        ? opts.recip_cutoff_bohr_inv
        : auto_recip_cutoff(alpha, opts.tolerance);
    const double inv_2alpha = 1.0 / (2.0 * alpha);
    const double alpha2 = alpha * alpha;
    const double inv_alpha_sqrtpi = 1.0 / (alpha * std::sqrt(M_PI));

    Eigen::VectorXd zc(N);
    for (Eigen::Index j = 0; j < N; ++j)
        zc[j] = charge_positions_cart.col(j).dot(nhat);

    const auto g_list =
        enumerate_cells_2d(b1, b2, recip_cutoff, /*include_origin=*/false);
    const int n_g = static_cast<int>(g_list.size());
    const auto cells =
        enumerate_cells_2d(a1, a2, real_cutoff, /*include_origin=*/true);
    const int n_cell = static_cast<int>(cells.size());

    // The sheet contributes its exact bare field −(2π/A) q_b |z_r − z_b| per
    // eval point (q_b = −Σq_i); zero on neutral input.
    const double q_b = -charges.sum();

    Eigen::VectorXd v = Eigen::VectorXd::Zero(M);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index p = 0; p < M; ++p) {
        const Eigen::Vector3d r = eval_points_cart.row(p).transpose();
        const double zr = r.dot(nhat);

        double v_short = 0.0;
        if (include_short_range) {
            for (int ci = 0; ci < n_cell; ++ci) {
                const Eigen::Vector3d& n = cells[ci];
                for (Eigen::Index j = 0; j < N; ++j) {
                    const Eigen::Vector3d d =
                        r - charge_positions_cart.col(j) - n;
                    const double rr = d.norm();
                    if (rr < 1.0e-14) continue;
                    if (rr > real_cutoff) continue;
                    v_short += charges[j] * std::erfc(alpha * rr) / rr;
                }
            }
        }

        double v_recip = 0.0;
        for (int gi = 0; gi < n_g; ++gi) {
            const Eigen::Vector3d& g = g_list[gi];
            const double gnorm = g.norm();
            const double g_over_2a = gnorm * inv_2alpha;
            const double g_over_2a_sq = g_over_2a * g_over_2a;
            double acc = 0.0;
            for (Eigen::Index j = 0; j < N; ++j) {
                const double dz = zr - zc[j];
                const double cosg =
                    std::cos(g.dot(r - charge_positions_cart.col(j)));
                const double pref = std::exp(-g_over_2a_sq - alpha2 * dz * dz);
                const double bracket = erfcx_stable(g_over_2a + alpha * dz) +
                                       erfcx_stable(g_over_2a - alpha * dz);
                acc += charges[j] * cosg * pref * bracket;
            }
            v_recip += acc / gnorm;
        }
        v_recip *= M_PI / A;

        // g = 0 slab term: point sources via the screened f(z) kernel...
        double v_g0 = 0.0;
        for (Eigen::Index j = 0; j < N; ++j) {
            const double dz = zr - zc[j];
            v_g0 += charges[j] *
                    (dz * std::erf(alpha * dz) +
                     inv_alpha_sqrtpi * std::exp(-alpha2 * dz * dz));
        }
        v_g0 *= -2.0 * M_PI / A;
        // ...plus the neutralising sheet's exact (un-screened) |z| field
        // −(2π/A) q_b |z_r − z_b| (see the energy variant); zero on neutral
        // input. The point sources are Ewald-split (screened f(z) kernel); the
        // smooth uniform sheet is not (bare |z| field).
        v_g0 += -(2.0 * M_PI / A) * q_b * std::abs(zr - z_background);

        v[p] = v_short + v_recip + v_g0;
    }
    return v;
}

}  // namespace vibeqc
