#include "vibeqc/semiempirical/methods/xtb/gfn2_multipole.hpp"

#include <libint2.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "vibeqc/basis.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/thread_pool.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

namespace {

// ---------------------------------------------------------------------------
// Compute dipole integrals for a single atom centre.
// Returns (dip_x, dip_y, dip_z) AO matrices.
// ---------------------------------------------------------------------------
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
compute_dipole_atom_centre(const BasisSet& basis,
                            const std::array<double, 3>& origin) {
    const auto& shells = basis.libint();
    const auto nbf = static_cast<Eigen::Index>(basis.nbasis());

    Eigen::MatrixXd mx = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd my = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd mz = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine prototype(libint2::Operator::emultipole1,
                              shells.max_nprim(), shells.max_l(), 0);
    prototype.set_params(origin);
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();

            engine.compute(shells[s1], shells[s2]);
            // buf[0] = overlap, buf[1..3] = ⟨μ|r_c−O_c|ν⟩
            const double* bx = buf[1];
            const double* by = buf[2];
            const double* bz = buf[3];
            if (!bx) continue;

            for (std::size_t i = 0; i < n1; ++i) {
                for (std::size_t j = 0; j < n2; ++j) {
                    const double vx = bx[i * n2 + j];
                    const double vy = by[i * n2 + j];
                    const double vz = bz[i * n2 + j];
                    mx(bf1 + i, bf2 + j) = vx;
                    my(bf1 + i, bf2 + j) = vy;
                    mz(bf1 + i, bf2 + j) = vz;
                    if (s1 != s2) {
                        mx(bf2 + j, bf1 + i) = vx;
                        my(bf2 + j, bf1 + i) = vy;
                        mz(bf2 + j, bf1 + i) = vz;
                    }
                }
            }
        }
    }
    return {mx, my, mz};
}

// ---------------------------------------------------------------------------
// Compute traceless quadrupole integrals for a single atom centre.
//
// libint emultipole2 returns (overlap, x, y, z, xx, xy, xz, yy, yz, zz)
// as the Cartesian quadrupole.  We convert to the GFN2 traceless
// convention:
//   Θ_ij = ³⁄₂·(r_i−O_i)(r_j−O_j) − ½·δ_ij·|r−O|²
//
// Returns (qxx, qxy, qxz, qyy, qyz, qzz).
// ---------------------------------------------------------------------------
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
           Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
compute_quadrupole_atom_centre(const BasisSet& basis,
                                const std::array<double, 3>& origin) {
    const auto& shells = basis.libint();
    const auto nbf = static_cast<Eigen::Index>(basis.nbasis());

    Eigen::MatrixXd qxx = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd qxy = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd qxz = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd qyy = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd qyz = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd qzz = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine prototype(libint2::Operator::emultipole2,
                              shells.max_nprim(), shells.max_l(), 0);
    prototype.set_params(origin);
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();

            engine.compute(shells[s1], shells[s2]);
            // buf layout: [0]=overlap,  [1]=x,  [2]=y,  [3]=z,
            //             [4]=xx,  [5]=xy,  [6]=xz,  [7]=yy,  [8]=yz,  [9]=zz
            const double* bxx = buf[4];
            if (!bxx) continue;
            const double* bxy = buf[5];
            const double* bxz = buf[6];
            const double* byy = buf[7];
            const double* byz = buf[8];
            const double* bzz = buf[9];

            for (std::size_t i = 0; i < n1; ++i) {
                for (std::size_t j = 0; j < n2; ++j) {
                    const auto idx = i * n2 + j;
                    const double xx = bxx[idx];
                    const double xy = bxy[idx];
                    const double xz = bxz[idx];
                    const double yy = byy[idx];
                    const double yz = byz[idx];
                    const double zz = bzz[idx];

                    // Convert Cartesian → traceless (GFN2 convention).
                    // Θ_ij = ³⁄₂ r_i r_j − ½ δ_ij r²
                    //    r² = xx + yy + zz
                    const double r2 = xx + yy + zz;
                    const double txx = 1.5 * xx - 0.5 * r2;
                    const double txy = 1.5 * xy;
                    const double txz = 1.5 * xz;
                    const double tyy = 1.5 * yy - 0.5 * r2;
                    const double tyz = 1.5 * yz;
                    const double tzz = 1.5 * zz - 0.5 * r2;

                    qxx(bf1 + i, bf2 + j) = txx;
                    qxy(bf1 + i, bf2 + j) = txy;
                    qxz(bf1 + i, bf2 + j) = txz;
                    qyy(bf1 + i, bf2 + j) = tyy;
                    qyz(bf1 + i, bf2 + j) = tyz;
                    qzz(bf1 + i, bf2 + j) = tzz;

                    if (s1 != s2) {
                        // Quadrupole operator is symmetric.
                        qxx(bf2 + j, bf1 + i) = txx;
                        qxy(bf2 + j, bf1 + i) = txy;
                        qxz(bf2 + j, bf1 + i) = txz;
                        qyy(bf2 + j, bf1 + i) = tyy;
                        qyz(bf2 + j, bf1 + i) = tyz;
                        qzz(bf2 + j, bf1 + i) = tzz;
                    }
                }
            }
        }
    }
    return {qxx, qxy, qxz, qyy, qyz, qzz};
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<Eigen::MatrixXd> build_gfn2_raw_global_quadrupole_integrals(
    const BasisSet& basis, const Molecule& mol) {
    // Raw Cartesian quadrupole integrals <p|r_k r_l|q> about the GLOBAL
    // origin, in xtb's component order (xx, yy, zz, xy, xz, yz) — the
    // convention xtb's build_SDQH0 fills qpint with and the AES Fock
    // (buildIsoAnisotropicH1) contracts vq against.  Distinct from the
    // traceless atom-centred set in build_gfn2_multipole_integrals.
    ensure_libint_initialized();
    (void)mol;

    const auto& shells = basis.libint();
    const auto nbf = static_cast<Eigen::Index>(basis.nbasis());
    std::vector<Eigen::MatrixXd> out;
    for (int c = 0; c < 6; ++c)
        out.push_back(Eigen::MatrixXd::Zero(nbf, nbf));

    libint2::Engine prototype(libint2::Operator::emultipole2,
                              shells.max_nprim(), shells.max_l(), 0);
    prototype.set_params(std::array<double, 3>{0.0, 0.0, 0.0});
    auto engines = make_engine_pool(prototype);

    const auto shell2bf = shells.shell2bf();
    const int n_shells = static_cast<int>(shells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
        const auto& buf = engine.results();
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        for (int s2 = 0; s2 <= s1; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            engine.compute(shells[s1], shells[s2]);
            const double* bxx = buf[4];
            if (!bxx) continue;
            const double* bxy = buf[5];
            const double* bxz = buf[6];
            const double* byy = buf[7];
            const double* byz = buf[8];
            const double* bzz = buf[9];
            for (std::size_t i = 0; i < n1; ++i) {
                for (std::size_t j = 0; j < n2; ++j) {
                    const auto idx = i * n2 + j;
                    out[0](bf1 + i, bf2 + j) = bxx[idx];
                    out[1](bf1 + i, bf2 + j) = byy[idx];
                    out[2](bf1 + i, bf2 + j) = bzz[idx];
                    out[3](bf1 + i, bf2 + j) = bxy[idx];
                    out[4](bf1 + i, bf2 + j) = bxz[idx];
                    out[5](bf1 + i, bf2 + j) = byz[idx];
                    if (s1 != s2) {
                        out[0](bf2 + j, bf1 + i) = bxx[idx];
                        out[1](bf2 + j, bf1 + i) = byy[idx];
                        out[2](bf2 + j, bf1 + i) = bzz[idx];
                        out[3](bf2 + j, bf1 + i) = bxy[idx];
                        out[4](bf2 + j, bf1 + i) = bxz[idx];
                        out[5](bf2 + j, bf1 + i) = byz[idx];
                    }
                }
            }
        }
    }
    return out;
}

GFN2MultipoleSet build_gfn2_multipole_integrals(
    const BasisSet& basis,
    const Molecule& mol,
    bool compute_quadrupole) {

    ensure_libint_initialized();

    const auto& atoms = mol.atoms();
    const auto n_atoms = atoms.size();
    const auto n_basis = basis.nbasis();

    GFN2MultipoleSet result;
    result.n_basis = static_cast<int>(n_basis);
    result.atoms.resize(n_atoms);

    for (std::size_t a = 0; a < n_atoms; ++a) {
        auto& ma = result.atoms[a];
        const std::array<double, 3> origin = atoms[a].xyz;

        // Dipole integrals (always needed).
        auto [dx, dy, dz] = compute_dipole_atom_centre(basis, origin);
        ma.dip_x = std::move(dx);
        ma.dip_y = std::move(dy);
        ma.dip_z = std::move(dz);

        // Quadrupole integrals (needed for GFN2).
        if (compute_quadrupole) {
            auto [qxx, qxy, qxz, qyy, qyz, qzz] =
                compute_quadrupole_atom_centre(basis, origin);
            ma.quad_xx = std::move(qxx);
            ma.quad_xy = std::move(qxy);
            ma.quad_xz = std::move(qxz);
            ma.quad_yy = std::move(qyy);
            ma.quad_yz = std::move(qyz);
            ma.quad_zz = std::move(qzz);
        }
    }

    return result;
}


GFN2ShellMoments compute_shell_multipole_moments(
    const Eigen::MatrixXd& density,
    const BasisSet& basis,
    const Molecule& mol,
    const GFN2MultipoleSet& mp_int,
    const std::vector<int>& ao_shell,
    const Eigen::VectorXd& n0_shell,
    int n_shells) {

    const auto n_basis = static_cast<Eigen::Index>(basis.nbasis());
    const auto n_atoms = static_cast<Eigen::Index>(mol.atoms().size());
    (void)n_atoms;

    GFN2ShellMoments moments;
    moments.resize(n_shells);
    // Zero all vectors.
    moments.q.setZero();
    moments.ox.setZero(); moments.oy.setZero(); moments.oz.setZero();
    moments.txx.setZero(); moments.txy.setZero(); moments.txz.setZero();
    moments.tyy.setZero(); moments.tyz.setZero(); moments.tzz.setZero();

    // Contract each atom's multipole integrals with the density.
    // Mulliken-style: for each AO μ∈A, Σ_ν D_μν · I^A_μν gives the
    // contribution of atom A's orbitals to the multipole moment on A.
    for (Eigen::Index mu = 0; mu < n_basis; ++mu) {
        int si = (mu < static_cast<Eigen::Index>(ao_shell.size()))
                     ? ao_shell[static_cast<std::size_t>(mu)] : -1;
        if (si < 0 || si >= n_shells) continue;

        for (Eigen::Index nu = 0; nu < n_basis; ++nu) {
            double D_mn = density(mu, nu);
            if (std::abs(D_mn) < 1e-20) continue;

            // Determine which atom μ belongs to for the integral origin.
            // We use the atom of μ (not ν) — this is the "bra" convention
            // from tblite: integrals centred on atom A, contracted by
            // μ ∈ A.
            const auto& shells = basis.shells();
            const auto& lib_shells = basis.libint();
            const auto shell2bf = lib_shells.shell2bf();
            int a_mu = -1;
            for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                int nf = lib_shells[s].size();
                int bf0 = shell2bf[s];
                if (mu >= bf0 && mu < bf0 + nf) {
                    a_mu = shells[s].atom_index;
                    break;
                }
            }
            if (a_mu < 0 || a_mu >= static_cast<int>(mp_int.atoms.size()))
                continue;

            const auto& ma = mp_int.atoms[static_cast<std::size_t>(a_mu)];

            // Dipole contribution to shell charge... wait, the shell charge
            // contribution comes from the OVERLAP (standard Mulliken), not
            // the dipole integrals. The dipole and quadrupole integrals
            // contribute to the shell dipole and quadrupole moments.

            moments.ox(si) -= D_mn * ma.dip_x(mu, nu);
            moments.oy(si) -= D_mn * ma.dip_y(mu, nu);
            moments.oz(si) -= D_mn * ma.dip_z(mu, nu);

            // Check if quadrupole integrals are populated (non-zero size).
            if (ma.quad_xx.rows() > 0) {
                moments.txx(si) -= D_mn * ma.quad_xx(mu, nu);
                moments.txy(si) -= D_mn * ma.quad_xy(mu, nu);
                moments.txz(si) -= D_mn * ma.quad_xz(mu, nu);
                moments.tyy(si) -= D_mn * ma.quad_yy(mu, nu);
                moments.tyz(si) -= D_mn * ma.quad_yz(mu, nu);
                moments.tzz(si) -= D_mn * ma.quad_zz(mu, nu);
            }
        }
    }

    // The shell charge is computed separately from the overlap-population
    // analysis (Δq = pop − n0).  This function only fills the multipole
    // parts; the caller must fill `moments.q` from its own SCC charge
    // analysis.
    return moments;
}

// ---------------------------------------------------------------------------
// Image-summed multipole integrals and their derivative contraction
// ---------------------------------------------------------------------------

namespace {

std::vector<libint2::Shell> shifted_shells(
    const libint2::BasisSet& shells, const Eigen::Vector3d& shift) {
    std::vector<libint2::Shell> out(shells.begin(), shells.end());
    for (auto& s : out) {
        s.O[0] += shift[0];
        s.O[1] += shift[1];
        s.O[2] += shift[2];
    }
    return out;
}

// Owning atom of every libint shell, from the AO-to-atom map.
std::vector<int> shell_atoms(const libint2::BasisSet& shells,
                             const std::vector<int>& ao_atom) {
    const auto shell2bf = shells.shell2bf();
    std::vector<int> out(shells.size());
    for (std::size_t s = 0; s < shells.size(); ++s) {
        out[s] = ao_atom[shell2bf[s]];
    }
    return out;
}

bool shell_pair_in_cutoff(const Atom& a, const Atom& b,
                          const Eigen::Vector3d& g, double cutoff) {
    if (!std::isfinite(cutoff)) return true;
    const double dx = a.xyz[0] - b.xyz[0] - g[0];
    const double dy = a.xyz[1] - b.xyz[1] - g[1];
    const double dz = a.xyz[2] - b.xyz[2] - g[2];
    return dx * dx + dy * dy + dz * dz <= cutoff * cutoff;
}

constexpr int kSym6[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

}  // namespace

GFN2MultipoleLatticeSums build_gfn2_multipole_lattice_sums(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    double pair_cutoff_bohr) {
    return build_gfn2_multipole_lattice_sums(
        basis, atoms, cells, ao_atom,
        [&](int a, int b, int cell_index) {
            return shell_pair_in_cutoff(
                       atoms[static_cast<std::size_t>(a)],
                       atoms[static_cast<std::size_t>(b)],
                       cells[static_cast<std::size_t>(cell_index)].r_cart,
                       pair_cutoff_bohr)
                ? 1.0 : 0.0;
        });
}

GFN2MultipoleLatticeSums build_gfn2_multipole_lattice_sums(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    const GFN2PairWeight& pair_weight) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (static_cast<int>(ao_atom.size()) != nbf) {
        throw std::invalid_argument(
            "build_gfn2_multipole_lattice_sums: ao_atom length mismatch");
    }
    const auto shell2bf = shells.shell2bf();
    const std::vector<int> s2a = shell_atoms(shells, ao_atom);
    const int n_shells = static_cast<int>(shells.size());

    GFN2MultipoleLatticeSums sums;
    sums.n_basis = nbf;
    sums.S0 = Eigen::MatrixXd::Zero(nbf, nbf);
    for (auto& m : sums.S1) m = Eigen::MatrixXd::Zero(nbf, nbf);
    for (auto& m : sums.S2) m = Eigen::MatrixXd::Zero(nbf, nbf);
    for (auto& m : sums.D0) m = Eigen::MatrixXd::Zero(nbf, nbf);
    for (auto& row : sums.D1)
        for (auto& m : row) m = Eigen::MatrixXd::Zero(nbf, nbf);
    for (auto& m : sums.Q0) m = Eigen::MatrixXd::Zero(nbf, nbf);

    libint2::Engine prototype(libint2::Operator::emultipole2,
                              shells.max_nprim(), shells.max_l(), 0);
    prototype.set_params(std::array<double, 3>{0.0, 0.0, 0.0});
    auto engines = make_engine_pool(prototype);

    for (int c = 0; c < static_cast<int>(cells.size()); ++c) {
        const Eigen::Vector3d g = cells[static_cast<std::size_t>(c)].r_cart;
        const auto shells_g = shifted_shells(shells, g);
        // Rows of every accumulator are owned by the bra shell, so the
        // parallel loop over s1 is race free.
        #pragma omp parallel for schedule(dynamic)
        for (int s1 = 0; s1 < n_shells; ++s1) {
            auto& engine = engines[static_cast<std::size_t>(omp_thread_index())];
            const auto& buf = engine.results();
            const auto bf1 = shell2bf[static_cast<std::size_t>(s1)];
            const auto n1 = shells[static_cast<std::size_t>(s1)].size();
            const int atom_a = s2a[static_cast<std::size_t>(s1)];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const int atom_b = s2a[static_cast<std::size_t>(s2)];
                const double weight = pair_weight(atom_a, atom_b, c);
                if (weight == 0.0) continue;
                const auto bf2 = shell2bf[static_cast<std::size_t>(s2)];
                const auto n2 = shells_g[static_cast<std::size_t>(s2)].size();
                engine.compute(shells[static_cast<std::size_t>(s1)],
                               shells_g[static_cast<std::size_t>(s2)]);
                if (!buf[0]) continue;
                for (std::size_t i = 0; i < n1; ++i) {
                    for (std::size_t j = 0; j < n2; ++j) {
                        const auto idx = i * n2 + j;
                        const Eigen::Index p = static_cast<Eigen::Index>(bf1 + i);
                        const Eigen::Index q = static_cast<Eigen::Index>(bf2 + j);
                        const double S = weight * buf[0][idx];
                        sums.S0(p, q) += S;
                        for (int a = 0; a < 3; ++a) {
                            sums.S1[static_cast<std::size_t>(a)](p, q) += g[a] * S;
                            for (int b = a; b < 3; ++b) {
                                sums.S2[static_cast<std::size_t>(kSym6[a][b])](p, q)
                                    += g[a] * g[b] * S;
                            }
                        }
                        for (int jdir = 0; jdir < 3; ++jdir) {
                            const double D = weight * buf[1 + jdir][idx];
                            sums.D0[static_cast<std::size_t>(jdir)](p, q) += D;
                            for (int a = 0; a < 3; ++a) {
                                sums.D1[static_cast<std::size_t>(a)][static_cast<std::size_t>(jdir)](p, q)
                                    += g[a] * D;
                            }
                        }
                        for (int k = 0; k < 6; ++k) {
                            sums.Q0[static_cast<std::size_t>(k)](p, q) += weight * buf[4 + k][idx];
                        }
                    }
                }
            }
        }
    }
    return sums;
}

GFN2MultipoleLatticeSums build_gfn2_multipole_molecular_sums(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<int>& ao_atom) {
    std::vector<LatticeCell> cells(1);
    return build_gfn2_multipole_lattice_sums(
        basis, atoms, cells, ao_atom,
        std::numeric_limits<double>::infinity());
}

GFN2MultipoleDerivativeContraction contract_gfn2_multipole_lattice_derivatives(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    double pair_cutoff_bohr,
    const GFN2MultipoleWeightFiller& filler) {
    return contract_gfn2_multipole_lattice_derivatives(
        basis, atoms, cells, ao_atom,
        [&](int a, int b, int cell_index) {
            return shell_pair_in_cutoff(
                       atoms[static_cast<std::size_t>(a)],
                       atoms[static_cast<std::size_t>(b)],
                       cells[static_cast<std::size_t>(cell_index)].r_cart,
                       pair_cutoff_bohr)
                ? 1.0 : 0.0;
        },
        filler);
}

GFN2MultipoleDerivativeContraction contract_gfn2_multipole_lattice_derivatives(
    const BasisSet& basis,
    const std::vector<Atom>& atoms,
    const std::vector<LatticeCell>& cells,
    const std::vector<int>& ao_atom,
    const GFN2PairWeight& pair_weight,
    const GFN2MultipoleWeightFiller& filler) {
    ensure_libint_initialized();
    const auto& shells = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    const int n_atoms = static_cast<int>(atoms.size());
    const auto shell2bf = shells.shell2bf();
    const std::vector<int> s2a = shell_atoms(shells, ao_atom);
    const int n_shells = static_cast<int>(shells.size());

    GFN2MultipoleDerivativeContraction out;
    out.gradient = Eigen::MatrixXd::Zero(n_atoms, 3);

    libint2::Engine prototype(libint2::Operator::emultipole2,
                              shells.max_nprim(), shells.max_l(), 1);
    prototype.set_params(std::array<double, 3>{0.0, 0.0, 0.0});
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        static_cast<std::size_t>(n_threads), Eigen::MatrixXd::Zero(n_atoms, 3));
    std::vector<Eigen::Matrix3d> virial_tls(
        static_cast<std::size_t>(n_threads), Eigen::Matrix3d::Zero());

    Eigen::MatrixXd wS;
    std::array<Eigen::MatrixXd, 3> wD;
    std::array<Eigen::MatrixXd, 6> wQ;
    for (int c = 0; c < static_cast<int>(cells.size()); ++c) {
        const Eigen::Vector3d g = cells[static_cast<std::size_t>(c)].r_cart;
        filler(c, g, wS, wD, wQ);
        if (wS.rows() != nbf || wS.cols() != nbf) {
            throw std::invalid_argument(
                "contract_gfn2_multipole_lattice_derivatives: weight shape");
        }
        const auto shells_g = shifted_shells(shells, g);
        #pragma omp parallel for schedule(dynamic)
        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& grad_local = grad_tls[tid];
            auto& virial_local = virial_tls[tid];
            const auto bf1 = shell2bf[static_cast<std::size_t>(s1)];
            const auto n1 = shells[static_cast<std::size_t>(s1)].size();
            const int atom_a = s2a[static_cast<std::size_t>(s1)];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const int atom_b = s2a[static_cast<std::size_t>(s2)];
                const double weight = pair_weight(atom_a, atom_b, c);
                if (weight == 0.0) continue;
                const auto bf2 = shell2bf[static_cast<std::size_t>(s2)];
                const auto n2 = shells_g[static_cast<std::size_t>(s2)].size();
                engine.compute(shells[static_cast<std::size_t>(s1)],
                               shells_g[static_cast<std::size_t>(s2)]);
                if (!buf[0]) continue;
                // libint lays the first-derivative multi-operator targets
                // out derivative-major: buffer d*10 + o for derivative d
                // (bra x, y, z, then ket x, y, z) and operator component o
                // (S, x, y, z, xx, xy, xz, yy, yz, zz); pinned by
                // tests/test_periodic_gfn2_aes.py against finite differences
                // through libint_emultipole2_derivative_layout_probe.
                for (int d = 0; d < 6; ++d) {
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            const auto idx = i * n2 + j;
                            const Eigen::Index p = static_cast<Eigen::Index>(bf1 + i);
                            const Eigen::Index q = static_cast<Eigen::Index>(bf2 + j);
                            double term = wS(p, q) * buf[d * 10 + 0][idx];
                            for (int jdir = 0; jdir < 3; ++jdir) {
                                term += wD[static_cast<std::size_t>(jdir)](p, q)
                                    * buf[d * 10 + 1 + jdir][idx];
                            }
                            for (int k = 0; k < 6; ++k) {
                                term += wQ[static_cast<std::size_t>(k)](p, q)
                                    * buf[d * 10 + 4 + k][idx];
                            }
                            acc += term;
                        }
                    }
                    acc *= weight;
                    if (d < 3) {
                        grad_local(atom_a, d) += acc;
                    } else {
                        grad_local(atom_b, d - 3) += acc;
                        for (int m = 0; m < 3; ++m) virial_local(d - 3, m) += acc * g[m];
                    }
                }
            }
        }
    }
    for (const auto& gl : grad_tls) out.gradient += gl;
    for (const auto& vl : virial_tls) out.image_virial += vl;
    return out;
}

Eigen::MatrixXd libint_emultipole2_derivative_layout_probe(double h) {
    ensure_libint_initialized();
    const auto make_shell = [](double exponent, const std::array<double, 3>& origin) {
        return libint2::Shell{{exponent}, {{0, false, {1.0}}}, origin};
    };
    const std::array<double, 3> origin_a = {0.0, 0.0, 0.0};
    const std::array<double, 3> origin_b = {1.3, 0.3, -0.2};
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(18, 10);

    libint2::Engine deriv(libint2::Operator::emultipole2, 1, 0, 1);
    deriv.set_params(std::array<double, 3>{0.0, 0.0, 0.0});
    deriv.compute(make_shell(1.0, origin_a), make_shell(0.8, origin_b));
    const auto& dbuf = deriv.results();
    for (int d = 0; d < 6; ++d) {
        for (int o = 0; o < 10; ++o) {
            const double* block = dbuf[static_cast<std::size_t>(d * 10 + o)];
            out(d, o) = block ? block[0] : 0.0;
        }
    }

    libint2::Engine plain(libint2::Operator::emultipole2, 1, 0, 0);
    plain.set_params(std::array<double, 3>{0.0, 0.0, 0.0});
    const auto& pbuf = plain.results();
    int row = 6;
    for (int centre = 0; centre < 2; ++centre) {
        for (int dir = 0; dir < 3; ++dir) {
            for (double sign : {1.0, -1.0}) {
                std::array<double, 3> a = origin_a;
                std::array<double, 3> b = origin_b;
                (centre == 0 ? a : b)[static_cast<std::size_t>(dir)] += sign * h;
                plain.compute(make_shell(1.0, a), make_shell(0.8, b));
                for (int o = 0; o < 10; ++o) {
                    const double* block = pbuf[static_cast<std::size_t>(o)];
                    out(row, o) = block ? block[0] : 0.0;
                }
                ++row;
            }
        }
    }
    return out;
}

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
