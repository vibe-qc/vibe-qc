#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "vibeqc/semiempirical/methods/indo/msindo_integrals.hpp"

namespace vibeqc {
namespace semiempirical {
namespace indo {

constexpr int MSINDO_GEPOL_NGRIDMAXRF = 45;

struct MsindoGepolCavity {
    Eigen::MatrixXd points;
    Eigen::VectorXd a_diag;
    Eigen::VectorXd area;
    Eigen::VectorXi seg_atom;
};

inline double msindo_cosmor_bohr(int z) {
    static const std::array<double, 54> kCosmorAng = {
        1.224, 1.540,
        1.991, 1.683, 2.112, 1.776, 1.678, 1.656, 1.595, 1.694,
        2.724, 2.076, 2.208, 2.520, 2.160, 2.160, 2.100, 2.256,
        3.300, 2.772,
        3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036,
        2.244, 2.532, 2.220, 2.280, 2.196, 2.424,
        3.636, 2.988,
        3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036, 3.036,
        2.352, 2.424, 2.484, 2.364, 2.424, 2.640,
    };
    if (z < 1 || z > static_cast<int>(kCosmorAng.size())) {
        throw std::invalid_argument("MSINDO GEPOL COSMOR radius is tabulated for Z=1..54");
    }
    return kCosmorAng[static_cast<std::size_t>(z - 1)] /
           MSINDO_BOHR_ANGSTROM;
}

inline std::array<Eigen::Vector3d, 32> msindo_gepol_seed_vertices() {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double phia = 2.0 + 2.0 * phi + 1.0 / phi;
    const double phib = 2.0 + phi;
    const double phiab = phia * phia + phib * phib;
    const double phib_n = phib * std::sqrt(3.0) / std::sqrt(phiab);
    const double phia_n = phia * std::sqrt(3.0) / std::sqrt(phiab);

    std::array<Eigen::Vector3d, 32> verts;
    int m = 0;
    for (int i = 0; i < 2; ++i) {
        const double si = (i == 0) ? 1.0 : -1.0;
        for (int j = 0; j < 2; ++j) {
            const double sj = (j == 0) ? 1.0 : -1.0;
            verts[m] = Eigen::Vector3d(si, sj, -1.0);
            verts[m + 1] = Eigen::Vector3d(si, sj, 1.0);
            verts[m + 2] = Eigen::Vector3d(0.0, si / phi, phi * sj);
            verts[m + 3] = Eigen::Vector3d(phi * sj, 0.0, si / phi);
            verts[m + 4] = Eigen::Vector3d(si / phi, phi * sj, 0.0);
            verts[m + 5] = Eigen::Vector3d(si * phia_n, sj * phib_n, 0.0);
            verts[m + 6] = Eigen::Vector3d(0.0, si * phia_n, sj * phib_n);
            verts[m + 7] = Eigen::Vector3d(si * phib_n, 0.0, sj * phia_n);
            m += 8;
        }
    }
    return verts;
}

inline double msindo_gepol_dneigh_coarse(double vdwr) {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double phia = 2.0 + 2.0 * phi + 1.0 / phi;
    const double phib = 2.0 + phi;
    const double phiab = phia * phia + phib * phib;
    return (1.0 + std::pow(1.0 - 1.0 / phi, 2.0) +
            std::pow(1.0 - phi, 2.0)) * vdwr * vdwr * 25.0 / phiab;
}

inline void msindo_gepol_cull_points(
        std::vector<Eigen::Vector3d>& points,
        std::vector<char>& angrid,
        const std::vector<Eigen::Vector3d>& neigh_pos,
        const std::vector<double>& neigh_r,
        const Eigen::Vector3d& atom_pos) {
    for (std::size_t in = 0; in < neigh_pos.size(); ++in) {
        const Eigen::Vector3d shift = atom_pos - neigh_pos[in];
        const double nr = neigh_r[in];
        for (std::size_t jx = 0; jx < points.size(); ++jx) {
            if (!angrid[jx]) continue;
            const Eigen::Vector3d g = points[jx] + shift;
            if (g.norm() <= nr) {
                angrid[jx] = 0;
            }
        }
    }
}

inline double msindo_gepol_refine(
        std::vector<Eigen::Vector3d>& points,
        std::vector<char>& angrid,
        int ngrid,
        int ngridmax,
        double dneigh,
        double vdwr,
        const std::vector<Eigen::Vector3d>& neigh_pos,
        const std::vector<double>& neigh_r,
        const Eigen::Vector3d& atom_pos,
        int refine) {
    int nlgrid = 0;
    int m = ngrid;
    for (int depth = 1; depth <= refine; ++depth) {
        double maxscal = 0.0;
        int m0 = m;
        for (int jx = 0; jx < ngrid; ++jx) {
            for (int kx = std::max(nlgrid, jx + 1); kx < ngrid; ++kx) {
                const Eigen::Vector3d d = points[jx] - points[kx];
                if (d.squaredNorm() > dneigh) continue;
                if (m0 >= ngridmax) {
                    throw std::runtime_error("MSINDO GEPOL refine overflow");
                }
                const Eigen::Vector3d mid = 0.5 * (points[jx] + points[kx]);
                const double scal = mid.squaredNorm();
                maxscal = std::max(maxscal, vdwr * vdwr / scal);
                points[static_cast<std::size_t>(m0)] = mid * vdwr / std::sqrt(scal);
                ++m0;
            }
        }
        nlgrid = ngrid;
        m = m0;
        if (depth < refine) {
            ngrid = m;
        }
        dneigh = dneigh * maxscal / 4.0;
    }

    msindo_gepol_cull_points(points, angrid, neigh_pos, neigh_r, atom_pos);
    return dneigh;
}

struct MsindoGepolFace {
    Eigen::Vector3d centre_local;
    int nsasrf = 0;
};

inline std::optional<MsindoGepolFace> msindo_gepol_refine_face(
        const Eigen::Vector3d& ci,
        const Eigen::Vector3d& cj,
        const Eigen::Vector3d& ck,
        bool ai,
        bool aj,
        bool ak,
        double vdwr,
        double dneigh,
        const std::vector<Eigen::Vector3d>& neigh_pos,
        const std::vector<double>& neigh_r,
        const Eigen::Vector3d& atom_pos,
        int refine) {
    std::vector<Eigen::Vector3d> pts(
        MSINDO_GEPOL_NGRIDMAXRF, Eigen::Vector3d::Zero());
    std::vector<char> angrid(MSINDO_GEPOL_NGRIDMAXRF, 1);
    pts[0] = ci;
    pts[1] = cj;
    pts[2] = ck;
    angrid[0] = ai ? 1 : 0;
    angrid[1] = aj ? 1 : 0;
    angrid[2] = ak ? 1 : 0;

    double dneigh_rf = dneigh;
    if (refine > 0) {
        dneigh_rf = msindo_gepol_refine(
            pts, angrid, 3, MSINDO_GEPOL_NGRIDMAXRF, dneigh, vdwr,
            neigh_pos, neigh_r, atom_pos, refine);
    }
    const int ngrid_rf = (refine > 0) ? 3 : 0;

    Eigen::Vector3d sas = Eigen::Vector3d::Zero();
    int nsasrf = 0;
    for (int irf = 0; irf < MSINDO_GEPOL_NGRIDMAXRF; ++irf) {
        if (!angrid[static_cast<std::size_t>(irf)]) continue;
        for (int jrf = std::max(irf + 1, ngrid_rf);
             jrf < MSINDO_GEPOL_NGRIDMAXRF; ++jrf) {
            if (!angrid[static_cast<std::size_t>(jrf)]) continue;
            if ((pts[jrf] - pts[irf]).squaredNorm() > dneigh_rf) continue;
            for (int krf = jrf + 1; krf < MSINDO_GEPOL_NGRIDMAXRF; ++krf) {
                if (!angrid[static_cast<std::size_t>(krf)]) continue;
                if ((pts[jrf] - pts[krf]).squaredNorm() > dneigh_rf) continue;
                if ((pts[krf] - pts[irf]).squaredNorm() > dneigh_rf) continue;
                sas += pts[irf] + pts[jrf] + pts[krf];
                ++nsasrf;
            }
        }
    }
    if (nsasrf == 0) {
        return std::nullopt;
    }
    return MsindoGepolFace{sas, nsasrf};
}

inline MsindoGepolCavity build_msindo_gepol_cavity(
        const std::vector<int>& Z,
        const Eigen::Ref<const Eigen::MatrixXd>& coords_bohr,
        int coarse = 0,
        int fine = 3,
        double rsolve = 0.0,
        double rscale = 0.1) {
    if (coords_bohr.cols() != 3 || coords_bohr.rows() != static_cast<int>(Z.size())) {
        throw std::invalid_argument("MSINDO GEPOL coords_bohr must have shape (n_atoms, 3)");
    }
    if (coarse != 0) {
        throw std::invalid_argument("MSINDO GEPOL native cavity supports coarse=0");
    }
    if (fine < coarse) {
        throw std::invalid_argument("MSINDO GEPOL fine grid must be >= coarse grid");
    }

    const int natom = static_cast<int>(Z.size());
    const double rsolvetmp =
        (rsolve != 0.0) ? rsolve / MSINDO_BOHR_ANGSTROM : 0.0;
    const int refine_fine = fine - coarse;
    const auto seed = msindo_gepol_seed_vertices();

    std::vector<Eigen::Vector3d> seg_pts;
    std::vector<double> seg_adiag;
    std::vector<double> seg_area;
    std::vector<int> seg_atom;

    for (int n = 0; n < natom; ++n) {
        const double vdwr = rsolvetmp + msindo_cosmor_bohr(Z[static_cast<std::size_t>(n)]);
        const Eigen::Vector3d atom_pos = coords_bohr.row(n).transpose();

        std::vector<Eigen::Vector3d> neigh_pos;
        std::vector<double> neigh_r;
        for (int mm = 0; mm < natom; ++mm) {
            if (mm == n) continue;
            const double neigh_radius =
                rsolvetmp + msindo_cosmor_bohr(Z[static_cast<std::size_t>(mm)]);
            const double vdwrn = std::pow(neigh_radius + vdwr, 2.0);
            const Eigen::Vector3d d =
                coords_bohr.row(n).transpose() - coords_bohr.row(mm).transpose();
            if (d.squaredNorm() <= vdwrn + vdwr) {
                neigh_pos.push_back(coords_bohr.row(mm).transpose());
                neigh_r.push_back(neigh_radius);
            }
        }

        std::vector<Eigen::Vector3d> cgrid(32, Eigen::Vector3d::Zero());
        for (int i = 0; i < 32; ++i) {
            cgrid[static_cast<std::size_t>(i)] = seed[static_cast<std::size_t>(i)] *
                                                 vdwr / std::sqrt(3.0);
        }
        std::vector<char> angrid(32, 1);
        const double dneigh = msindo_gepol_dneigh_coarse(vdwr);
        msindo_gepol_refine(
            cgrid, angrid, 32, 32, dneigh, vdwr, neigh_pos, neigh_r, atom_pos, 0);

        const double surfce = vdwr * vdwr / (60.0 * std::pow(4.0, fine));
        for (int i = 0; i < 32; ++i) {
            for (int j = i + 1; j < 32; ++j) {
                if ((cgrid[i] - cgrid[j]).squaredNorm() > dneigh) continue;
                for (int k = j + 1; k < 32; ++k) {
                    if ((cgrid[j] - cgrid[k]).squaredNorm() > dneigh) continue;
                    if ((cgrid[k] - cgrid[i]).squaredNorm() > dneigh) continue;

                    if (!(angrid[i] || angrid[j] || angrid[k])) {
                        Eigen::Vector3d centre = cgrid[i] + cgrid[j] + cgrid[k];
                        centre = centre * vdwr / std::sqrt(centre.squaredNorm());
                        bool buried = false;
                        for (std::size_t in = 0; in < neigh_pos.size(); ++in) {
                            const Eigen::Vector3d g = centre + atom_pos - neigh_pos[in];
                            if (g.norm() <= neigh_r[in]) {
                                buried = true;
                                break;
                            }
                        }
                        if (buried) continue;
                    }

                    auto face = msindo_gepol_refine_face(
                        cgrid[i], cgrid[j], cgrid[k],
                        static_cast<bool>(angrid[i]),
                        static_cast<bool>(angrid[j]),
                        static_cast<bool>(angrid[k]),
                        vdwr, dneigh, neigh_pos, neigh_r, atom_pos, refine_fine);
                    if (!face.has_value()) continue;

                    const double vdwrn =
                        rsolvetmp * rscale + msindo_cosmor_bohr(Z[static_cast<std::size_t>(n)]);
                    const double norm = face->centre_local.norm();
                    const Eigen::Vector3d pt = (vdwrn / norm) * face->centre_local + atom_pos;
                    const double area = surfce * static_cast<double>(face->nsasrf);
                    seg_pts.push_back(pt);
                    seg_adiag.push_back(1.07 / std::sqrt(area));
                    seg_area.push_back(area);
                    seg_atom.push_back(n);
                }
            }
        }
    }

    if (seg_pts.empty()) {
        throw std::runtime_error("MSINDO GEPOL cavity produced no segments");
    }

    const int nseg = static_cast<int>(seg_pts.size());
    MsindoGepolCavity out;
    out.points.resize(nseg, 3);
    out.a_diag.resize(nseg);
    out.area.resize(nseg);
    out.seg_atom.resize(nseg);
    for (int i = 0; i < nseg; ++i) {
        out.points.row(i) = seg_pts[static_cast<std::size_t>(i)].transpose();
        out.a_diag(i) = seg_adiag[static_cast<std::size_t>(i)];
        out.area(i) = seg_area[static_cast<std::size_t>(i)];
        out.seg_atom(i) = seg_atom[static_cast<std::size_t>(i)];
    }
    return out;
}

}  // namespace indo
}  // namespace semiempirical
}  // namespace vibeqc
