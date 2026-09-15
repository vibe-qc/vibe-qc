#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/dispersion.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"
#include "vibeqc/semiempirical/core/charge_mixer.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_multipole.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

namespace {

// ===========================================================================
// EXPERIMENTAL faithful GFN2 anisotropic electrostatics (AES).
// Bannwarth, Ehlert & Grimme, J. Chem. Theory Comput. 2019, 15, 1652,
// doi:10.1021/acs.jctc.8b01176.  Active only when XTBSccOptions::aes_faithful
// is set.  The flag-on path uses the atom-resolved AES potential in the SCC
// Fock build and evaluates the matching AES energy at the converged density
// (the ad-hoc shell-resolved AES is disabled on that path to avoid double
// counting).  Quantitative xtb parity is still tracked in
// handovers/HANDOVER_GFN2_AES.md.
// ===========================================================================

using Vec3 = std::array<double, 3>;
using Vec6 = std::array<double, 6>;  // order: xx, xy, xz, yy, yz, zz
// Off-diagonal Cartesian components are counted twice in the full tensor
// contraction Σ_ij Θ_ij² (and Σ_ij Θ_ij T_ij).
constexpr double kQScale[6] = {1.0, 2.0, 2.0, 1.0, 2.0, 1.0};

// Atom-resolved cumulative atomic multipole moments (CAMM), port of xtb
// aespot.F90 mmompop: Mulliken-partitioned moments of the GLOBAL-origin
// dipole/quadrupole operators (the R_A shift terms are part of the moment
// definition), followed by xtb's trace removal (x1.5, then -tr/2 on the
// diagonal).  Output th uses this file's Vec6 order (xx, xy, xz, yy, yz, zz),
// mapped from xtb's lin order (xx, xy, yy, xz, yz, zz).
void compute_atom_moments(const Eigen::MatrixXd& D,
                          const Eigen::MatrixXd& S,
                          const std::vector<Eigen::MatrixXd>& Dg,
                          const std::vector<Eigen::MatrixXd>& Qg,
                          const Molecule& mol,
                          const std::vector<int>& ao_atom,
                          int n_atoms, int n_basis,
                          std::vector<Vec3>& mu, std::vector<Vec6>& th) {
    const auto& atoms = mol.atoms();
    mu.assign(static_cast<std::size_t>(n_atoms), Vec3{0, 0, 0});
    th.assign(static_cast<std::size_t>(n_atoms), Vec6{0, 0, 0, 0, 0, 0});
    // lin order xx, xy, yy, xz, yz, zz
    std::vector<std::array<double, 6>> qp(
        static_cast<std::size_t>(n_atoms), {0, 0, 0, 0, 0, 0});
    for (int p = 0; p < n_basis; ++p) {
        const int ii = ao_atom[p];
        const double x1[3] = {atoms[ii].xyz[0], atoms[ii].xyz[1],
                              atoms[ii].xyz[2]};
        // off-diagonal pairs (q < p), matching xtb mmompop's lower triangle
        for (int q = 0; q < p; ++q) {
            const int jj = ao_atom[q];
            const double x2[3] = {atoms[jj].xyz[0], atoms[jj].xyz[1],
                                  atoms[jj].xyz[2]};
            const double pij = D(p, q);
            if (pij == 0.0) continue;
            const double ps = pij * S(p, q);
            for (int k = 0; k < 3; ++k) {
                const double pdmk = pij * Dg[k](p, q);
                mu[jj][k] += x2[k] * ps - pdmk;
                mu[ii][k] += x1[k] * ps - pdmk;
                for (int l = 0; l < k; ++l) {
                    // lin off-diagonal slot (xx, xy, yy, xz, yz, zz)
                    const int kl = k * (k + 1) / 2 + l;
                    const int kj = k + l + 2;  // xtb order xx, yy, zz, xy, xz, yz
                    const double pdml = pij * Dg[l](p, q);
                    const double pqm = pij * Qg[kj](p, q);
                    qp[jj][kl] += pdmk * x2[l] + pdml * x2[k]
                                  - x2[l] * x2[k] * ps - pqm;
                    qp[ii][kl] += pdmk * x1[l] + pdml * x1[k]
                                  - x1[l] * x1[k] * ps - pqm;
                }
                const int kl = k * (k + 1) / 2 + k;
                const double pqm = pij * Qg[k](p, q);
                qp[jj][kl] += 2.0 * pdmk * x2[k] - x2[k] * x2[k] * ps - pqm;
                qp[ii][kl] += 2.0 * pdmk * x1[k] - x1[k] * x1[k] * ps - pqm;
            }
        }
        // diagonal (p == p)
        {
            const double pij = D(p, p);
            const double ps = pij * S(p, p);
            for (int k = 0; k < 3; ++k) {
                const double pdmk = pij * Dg[k](p, p);
                mu[ii][k] += x1[k] * ps - pdmk;
                for (int l = 0; l < k; ++l) {
                    const int kl = k * (k + 1) / 2 + l;
                    const int kj = k + l + 2;
                    const double pdml = pij * Dg[l](p, p);
                    const double pqm = pij * Qg[kj](p, p);
                    qp[ii][kl] += pdmk * x1[l] + pdml * x1[k]
                                  - x1[l] * x1[k] * ps - pqm;
                }
                const int kl = k * (k + 1) / 2 + k;
                const double pqm = pij * Qg[k](p, p);
                qp[ii][kl] += 2.0 * pdmk * x1[k] - x1[k] * x1[k] * ps - pqm;
            }
        }
    }
    // trace removal (xtb mmompop): qp -> 1.5 qp, diagonals -= tr/2
    for (int a = 0; a < n_atoms; ++a) {
        const double tr = qp[a][0] + qp[a][2] + qp[a][5];
        const double half = 0.5 * tr;
        for (int k = 0; k < 6; ++k) qp[a][k] *= 1.5;
        qp[a][0] -= half;
        qp[a][2] -= half;
        qp[a][5] -= half;
        // lin -> Vec6 order: xx, xy, xz, yy, yz, zz
        th[a] = {qp[a][0], qp[a][1], qp[a][3], qp[a][2], qp[a][4], qp[a][5]};
    }
}

// Global-origin dipole and quadrupole integral matrices (origin at 0),
// built from the atom-centred multipole integrals: D(p,q) = M_A(p,q) +
// R_A S(p,q) for p on atom A; the quadrupoles are returned in xtb's
// integral order (xx, yy, zz, xy, xz, yz) to contract with setvsdq's vq.
void build_global_multipole_integrals(
    const Molecule& mol,
    const Eigen::MatrixXd& S,
    const GFN2MultipoleSet& mp,
    const BasisSet& basis,
    const std::vector<int>& ao_atom,
    int n_basis,
    std::vector<Eigen::MatrixXd>& Dg,
    std::vector<Eigen::MatrixXd>& Qg) {
    const auto& atoms = mol.atoms();
    Dg.assign(3, Eigen::MatrixXd::Zero(n_basis, n_basis));
    for (int a = 0; a < static_cast<int>(atoms.size()); ++a) {
        const auto& ma = mp.atoms[static_cast<std::size_t>(a)];
        const double Ra[3] = {atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]};
        const Eigen::MatrixXd Mc[3] = {
            ma.dip_x, ma.dip_y, ma.dip_z};
        for (int p = 0; p < n_basis; ++p) {
            if (ao_atom[p] != a) continue;
            for (int q = 0; q < n_basis; ++q) {
                const double s = S(p, q);
                for (int k = 0; k < 3; ++k)
                    Dg[k](p, q) += Mc[k](p, q) + Ra[k] * s;
            }
        }
    }
    // Quadrupoles: raw global-origin integrals straight from libint (xtb
    // qpint order), not derivable from the traceless atom-centred set.
    Qg = build_gfn2_raw_global_quadrupole_integrals(basis, mol);
}

// Faithful GFN2 AES energy at fixed moments + partial charges.
//   E_onsite = Σ_A [ dpol_A (μ_A·μ_A) + qpol_A Σ_ij Θ_A,ij² ]              (DPOL/QPOL kernel)
//   E_inter  = Σ_{A<B} [ charge-dipole·fdmp3 + (dipole-dipole + charge-quad)·fdmp5 ]
// Damping fdmp_n = 1/(1 + 6·(r̄/r)^kdmp_n), r̄ = ½(mrad_A+mrad_B), kdmp3=3, kdmp5=4.
// q_atom = partial charge (n0 − pop, positive for a cation).
double aes_energy(const Molecule& mol,
                  const Eigen::VectorXd& q_atom,
                  const std::vector<Vec3>& mu, const std::vector<Vec6>& th,
                  const std::vector<double>& dpol, const std::vector<double>& qpol,
                  const std::vector<double>& mrad) {
    const auto& at = mol.atoms();
    const int n = static_cast<int>(at.size());
    double E = 0.0;
    for (int a = 0; a < n; ++a) {
        const double m2 = mu[a][0]*mu[a][0] + mu[a][1]*mu[a][1] + mu[a][2]*mu[a][2];
        double t2 = 0.0;
        for (int k = 0; k < 6; ++k) t2 += kQScale[k] * th[a][k] * th[a][k];
        E += dpol[a] * m2 + qpol[a] * t2;
    }
    const double kdmp3 = 3.0, kdmp5 = 4.0;
    for (int A = 0; A < n; ++A) {
        for (int B = A + 1; B < n; ++B) {
            const double dx = at[A].xyz[0] - at[B].xyz[0];
            const double dy = at[A].xyz[1] - at[B].xyz[1];
            const double dz = at[A].xyz[2] - at[B].xyz[2];
            const double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < 1e-12) continue;
            const double r = std::sqrt(r2), r3 = r2 * r, r5 = r3 * r2;
            const double rr = 0.5 * (mrad[A] + mrad[B]) / r;
            const double fdmp3 = 1.0 / (1.0 + 6.0 * std::pow(rr, kdmp3));
            const double fdmp5 = 1.0 / (1.0 + 6.0 * std::pow(rr, kdmp5));
            const double qA = q_atom(A), qB = q_atom(B);
            const double mAv = mu[A][0]*dx + mu[A][1]*dy + mu[A][2]*dz;
            const double mBv = mu[B][0]*dx + mu[B][1]*dy + mu[B][2]*dz;
            // charge-dipole (vec = R_A - R_B; electronic moment convention)
            E += fdmp3 * (qA * mBv - qB * mAv) / r3;
            // dipole–dipole
            const double mAmB = mu[A][0]*mu[B][0] + mu[A][1]*mu[B][1] + mu[A][2]*mu[B][2];
            E += fdmp5 * (mAmB / r3 - 3.0 * mAv * mBv / r5);
            // charge-quadrupole. Atomic quadrupoles are already traceless, so
            // the interaction matrix contracts raw r_i r_j/r^5 components.
            const double T[6] = {dx*dx, dx*dy, dx*dz, dy*dy, dy*dz, dz*dz};
            double sThA = 0.0, sThB = 0.0;
            for (int k = 0; k < 6; ++k) {
                sThA += kQScale[k] * th[A][k] * T[k];
                sThB += kQScale[k] * th[B][k] * T[k];
            }
            E += fdmp5 / r5 * (qA * sThB + qB * sThA);
        }
    }
    return E;
}

// Port of xtb aespot.F90 setvsdq: global-origin AES potentials (vs, vd, vq)
// for the SCC Fock, including the origin-shift (R_C) terms and the on-site
// DPOL/QPOL CT corrections.  Input moments mu/th use the Vec6 order
// (xx, xy, xz, yy, yz, zz), mapped to xtb's lin order internally; vqp is
// returned in xtb's integral order (xx, yy, zz, xy, xz, yz) to contract with
// Qg of the same order.  Units: Hartree throughout.
void aes_potentials(const Molecule& mol,
                    const Eigen::VectorXd& q_atom,
                    const std::vector<Vec3>& mu, const std::vector<Vec6>& th,
                    const std::vector<double>& dpol, const std::vector<double>& qpol,
                    const std::vector<double>& mrad,
                    Eigen::VectorXd& vat,
                    std::vector<Vec3>& vdp,
                    std::vector<Vec6>& vqp) {
    const auto& at = mol.atoms();
    const int n = static_cast<int>(at.size());
    vat = Eigen::VectorXd::Zero(n);
    vdp.assign(static_cast<std::size_t>(n), Vec3{0, 0, 0});
    vqp.assign(static_cast<std::size_t>(n), Vec6{0, 0, 0, 0, 0, 0});

    const double kdmp3 = 3.0, kdmp5 = 4.0;
    Eigen::MatrixXd gab3 = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd gab5 = Eigen::MatrixXd::Zero(n, n);
    for (int A = 0; A < n; ++A) {
        for (int B = 0; B < n; ++B) {
            if (A == B) continue;
            const double dx = at[A].xyz[0] - at[B].xyz[0];
            const double dy = at[A].xyz[1] - at[B].xyz[1];
            const double dz = at[A].xyz[2] - at[B].xyz[2];
            const double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            const double rr = 0.5 * (mrad[A] + mrad[B]) / r;
            gab3(A, B) = 1.0 / ((1.0 + 6.0 * std::pow(rr, kdmp3)) * r * r * r);
            gab5(A, B) = 1.0 / ((1.0 + 6.0 * std::pow(rr, kdmp5)) * r * r * r * r * r);
        }
    }

    for (int i = 0; i < n; ++i) {
        const double ra[3] = {at[i].xyz[0], at[i].xyz[1], at[i].xyz[2]};
        double stmp = 0.0;
        double dtmp[3] = {0, 0, 0};
        double qtmp[6] = {0, 0, 0, 0, 0, 0};
        for (int j = 0; j < n; ++j) {
            const double g3 = gab3(j, i);
            const double g5 = gab5(j, i);
            const double dra[3] = {ra[0] - at[j].xyz[0], ra[1] - at[j].xyz[1],
                                   ra[2] - at[j].xyz[2]};
            const double r2a = ra[0]*ra[0] + ra[1]*ra[1] + ra[2]*ra[2];
            const double r2ab = dra[0]*dra[0] + dra[1]*dra[1] + dra[2]*dra[2];
            const double t1a = ra[0]*dra[0] + ra[1]*dra[1] + ra[2]*dra[2];
            const double t2a = mu[j][0]*dra[0] + mu[j][1]*dra[1] + mu[j][2]*dra[2];
            const double t3a = ra[0]*mu[j][0] + ra[1]*mu[j][1] + ra[2]*mu[j][2];
            // lin-ordered moments of atom j: xx, xy, yy, xz, yz, zz
            const double qpl[6] = {th[j][0], th[j][1], th[j][3],
                                   th[j][2], th[j][4], th[j][5]};
            double dum5a = 0.0;
            for (int l1 = 0; l1 < 3; ++l1) {
                for (int l2 = 0; l2 < 3; ++l2) {
                    // lin(l1, l2) for l1 >= l2
                    const int hi = std::max(l1, l2), lo = std::min(l1, l2);
                    const int ll = hi * (hi + 1) / 2 + lo;
                    dum5a += -qpl[ll] * dra[l1] * dra[l2];
                }
                for (int l2 = 0; l2 < l1; ++l2) {
                    const int ki = l1 + l2 + 2;  // xx, yy, zz, xy, xz, yz
                    qtmp[ki] += -3.0 * q_atom(j) * g5 * dra[l2] * dra[l1];
                }
                qtmp[l1] += -1.5 * q_atom(j) * g5 * dra[l1] * dra[l1];
            }
            // the double sum over (dra_l ra_l)(dra_k ra_k) = t1a^2
            dum5a += -1.5 * q_atom(j) * t1a * t1a;
            const double dum3a = -t1a * q_atom(j) - t2a;
            dum5a += t3a * r2ab - 3.0 * t1a * t2a + 0.5 * q_atom(j) * r2a * r2ab;
            stmp += dum5a * g5 + dum3a * g3;
            for (int l1 = 0; l1 < 3; ++l1) {
                const double dum3b = dra[l1] * q_atom(j);
                const double dum5b = 3.0 * dra[l1] * t2a - r2ab * mu[j][l1]
                    - q_atom(j) * r2ab * ra[l1]
                    + 3.0 * q_atom(j) * dra[l1] * t1a;
                dtmp[l1] += dum3b * g3 + dum5b * g5;
                qtmp[l1] += 0.5 * r2ab * q_atom(j) * g5;
            }
        }
        vat(i) = stmp;
        vdp[i] = {dtmp[0], dtmp[1], dtmp[2]};
        vqp[i] = {qtmp[0], qtmp[1], qtmp[2], qtmp[3], qtmp[4], qtmp[5]};

        // --- CT correction terms (on-site DPOL/QPOL kernel) ---
        const double qs1 = 2.0 * dpol[i];
        const double qs2 = 6.0 * qpol[i];
        const double qpl[6] = {th[i][0], th[i][1], th[i][3],
                               th[i][2], th[i][4], th[i][5]};
        double t3a = 0.0;
        double t2a = 0.0;
        for (int l1 = 0; l1 < 3; ++l1) {
            t3a += ra[l1] * mu[i][l1] * qs1;
            vdp[i][l1] -= qs1 * mu[i][l1];
            for (int l2 = 0; l2 < l1; ++l2) {
                const int ll = l1 * (l1 + 1) / 2 + l2;
                const int ki = l1 + l2 + 2;
                vqp[i][ki] -= qpl[ll] * qs2;
                t3a -= ra[l1] * ra[l2] * qpl[ll] * qs2;
                vdp[i][l1] += ra[l2] * qpl[ll] * qs2;
                vdp[i][l2] += ra[l1] * qpl[ll] * qs2;
            }
            const int ll = l1 * (l1 + 1) / 2 + l1;
            vqp[i][l1] -= qpl[ll] * qs2 * 0.5;
            t3a -= ra[l1] * ra[l1] * qpl[ll] * qs2 * 0.5;
            vdp[i][l1] += ra[l1] * qpl[ll] * qs2;
            t2a += qpl[ll];
        }
        vat(i) += t3a;
        t2a *= qpol[i];
        for (int l1 = 0; l1 < 3; ++l1) {
            vqp[i][l1] += t2a;
            vdp[i][l1] -= 2.0 * ra[l1] * t2a;
            vat(i) += t2a * ra[l1] * ra[l1];
        }
    }
}

void add_aes_potential_to_hamiltonian(
    Eigen::MatrixXd& H,
    const Eigen::MatrixXd& S,
    const std::vector<Eigen::MatrixXd>& Dg,
    const std::vector<Eigen::MatrixXd>& Qg,
    const std::vector<int>& ao_atom,
    const Eigen::VectorXd& vat,
    const std::vector<Vec3>& vdp,
    const std::vector<Vec6>& vqp,
    int n_basis) {
    // Port of xtb buildIsoAnisotropicH1's AES block (Ha units):
    //   H += 1/2 S (vs_i + vs_j)
    //      + 1/2 sum_l D_l (vd_l,i + vd_l,j)
    //      + 1/2 sum_k Q_k (vq_k,i + vq_k,j)
    // with GLOBAL-origin integrals (Dg/Qg) and potentials, both in xtb's
    // component order (Q: xx, yy, zz, xy, xz, yz).
    for (int p = 0; p < n_basis; ++p) {
        const int ap = ao_atom[p];
        for (int q = 0; q < n_basis; ++q) {
            const int aq = ao_atom[q];
            double shift = 0.5 * S(p, q) * (vat(ap) + vat(aq));
            for (int l = 0; l < 3; ++l) {
                shift += 0.5 * Dg[l](p, q) * (vdp[ap][l] + vdp[aq][l]);
            }
            for (int k = 0; k < 6; ++k) {
                shift += 0.5 * Qg[k](p, q) * (vqp[ap][k] + vqp[aq][k]);
            }
            H(p, q) += shift;
        }
    }
}

// The default molecular GFN2 path uses the exact closed-shell Aufbau map.
// A nonzero temperature is reserved for SCC stabilization retries: long-range
// ionic diatomics can bounce between neutral and charge-transfer fixed points
// when the third-order term is active and the frontier levels are nearly
// discontinuous under hard occupations.
Eigen::VectorXd closed_shell_occupations(
    const Eigen::VectorXd& eps,
    int n_electrons,
    double temperature) {
    const int n = static_cast<int>(eps.size());
    Eigen::VectorXd occ = Eigen::VectorXd::Zero(n);
    const double target = std::max(0.0, std::min(static_cast<double>(n_electrons),
                                                 2.0 * static_cast<double>(n)));

    if (temperature <= 0.0) {
        const int n_occ = std::min(n, static_cast<int>(target) / 2);
        for (int i = 0; i < n_occ; ++i) occ(i) = 2.0;
        return occ;
    }

    if (target <= 1e-12) return occ;
    if (target >= 2.0 * static_cast<double>(n) - 1e-12) {
        occ.setConstant(2.0);
        return occ;
    }

    auto particle_count = [&](double mu) {
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            double x = (eps(i) - mu) / temperature;
            x = std::max(-50.0, std::min(50.0, x));
            total += 2.0 / (1.0 + std::exp(x));
        }
        return total;
    };

    const double width = std::max(temperature, 1e-8);
    double lo = eps.minCoeff() - 80.0 * width - 1.0;
    double hi = eps.maxCoeff() + 80.0 * width + 1.0;
    for (int expand = 0; expand < 20; ++expand) {
        if (particle_count(lo) <= target && particle_count(hi) >= target) break;
        lo -= 2.0 * width;
        hi += 2.0 * width;
    }

    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (particle_count(mid) < target) lo = mid;
        else hi = mid;
    }
    const double mu = 0.5 * (lo + hi);
    for (int i = 0; i < n; ++i) {
        double x = (eps(i) - mu) / temperature;
        x = std::max(-50.0, std::min(50.0, x));
        occ(i) = 2.0 / (1.0 + std::exp(x));
    }
    return occ;
}

const char* eigen_status_name(Eigen::ComputationInfo info) {
    switch (info) {
        case Eigen::Success: return "Success";
        case Eigen::NumericalIssue: return "NumericalIssue";
        case Eigen::NoConvergence: return "NoConvergence";
        case Eigen::InvalidInput: return "InvalidInput";
    }
    return "Unknown";
}

bool matrix_all_finite(const Eigen::MatrixXd& M) {
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            if (!std::isfinite(M(i, j))) return false;
    return true;
}

double matrix_max_abs_finite(const Eigen::MatrixXd& M) {
    double value = 0.0;
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            if (std::isfinite(M(i, j)))
                value = std::max(value, std::abs(M(i, j)));
    return value;
}

double matrix_symmetry_residual(const Eigen::MatrixXd& M) {
    if (M.rows() != M.cols()) return std::numeric_limits<double>::infinity();
    double value = 0.0;
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        for (Eigen::Index j = i + 1; j < M.cols(); ++j) {
            const double a = M(i, j);
            const double b = M(j, i);
            if (std::isfinite(a) && std::isfinite(b)) {
                value = std::max(value, std::abs(a - b));
            } else {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    return value;
}

bool vector_all_finite(const Eigen::VectorXd& v) {
    for (Eigen::Index i = 0; i < v.size(); ++i)
        if (!std::isfinite(v(i))) return false;
    return true;
}

double vector_max_abs_finite(const Eigen::VectorXd& v) {
    double value = 0.0;
    for (Eigen::Index i = 0; i < v.size(); ++i)
        if (std::isfinite(v(i))) value = std::max(value, std::abs(v(i)));
    return value;
}

std::string element_inventory(const Molecule& mol) {
    std::vector<int> zs;
    std::vector<int> counts;
    for (const auto& atom : mol.atoms()) {
        auto it = std::find(zs.begin(), zs.end(), atom.Z);
        if (it == zs.end()) {
            zs.push_back(atom.Z);
            counts.push_back(1);
        } else {
            counts[static_cast<std::size_t>(it - zs.begin())] += 1;
        }
    }
    std::ostringstream os;
    for (std::size_t i = 0; i < zs.size(); ++i) {
        if (i) os << ",";
        os << "Z=" << zs[i] << "x" << counts[i];
    }
    return os.str();
}

bool has_extended_period_element(const Molecule& mol) {
    for (const auto& atom : mol.atoms())
        if (atom.Z > 18) return true;
    return false;
}

std::string overlap_spectrum_summary(const Eigen::MatrixXd& S) {
    std::ostringstream os;
    if (!matrix_all_finite(S)) {
        os << "S_eigs=skipped_nonfinite";
        return os.str();
    }
    Eigen::MatrixXd S_sym = 0.5 * (S + S.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s_solver(S_sym);
    if (s_solver.info() != Eigen::Success) {
        os << "S_eigs=failed(" << eigen_status_name(s_solver.info()) << ")";
        return os.str();
    }
    const Eigen::VectorXd evals = s_solver.eigenvalues();
    const double min_eval = evals.size() ? evals.minCoeff() : 0.0;
    const double max_eval = evals.size() ? evals.maxCoeff() : 0.0;
    const double cond = (min_eval != 0.0)
        ? std::abs(max_eval / min_eval)
        : std::numeric_limits<double>::infinity();
    os << "S_min_eig=" << min_eval
       << " S_max_eig=" << max_eval
       << " S_cond_abs=" << cond;
    return os.str();
}

std::string gfn2_diag_failure_message(
    const Molecule& mol,
    const Eigen::MatrixXd& H_scc,
    const Eigen::MatrixXd& S,
    const Eigen::VectorXd& V_shell,
    int iter,
    int n_basis,
    int n_shells,
    int n_val_elec,
    int n_occ,
    Eigen::ComputationInfo solver_info) {
    std::ostringstream os;
    os << std::setprecision(10);
    os << "GFN2: generalized eigensolver failed"
       << " iter=" << iter
       << " solver_status=" << eigen_status_name(solver_info)
       << " elements=" << element_inventory(mol)
       << " matrix_dim=" << H_scc.rows() << "x" << H_scc.cols()
       << " overlap_dim=" << S.rows() << "x" << S.cols()
       << " n_basis=" << n_basis
       << " n_shells=" << n_shells
       << " n_val_electrons=" << n_val_elec
       << " n_occ=" << n_occ
       << " H_finite=" << (matrix_all_finite(H_scc) ? "true" : "false")
       << " S_finite=" << (matrix_all_finite(S) ? "true" : "false")
       << " V_shell_finite=" << (vector_all_finite(V_shell) ? "true" : "false")
       << " H_sym_inf=" << matrix_symmetry_residual(H_scc)
       << " S_sym_inf=" << matrix_symmetry_residual(S)
       << " H_max_abs_finite=" << matrix_max_abs_finite(H_scc)
       << " S_max_abs_finite=" << matrix_max_abs_finite(S)
       << " V_shell_max_abs_finite=" << vector_max_abs_finite(V_shell)
       << " " << overlap_spectrum_summary(S);
    return os.str();
}

Eigen::VectorXd residual_trace_vector(const std::vector<double>& values) {
    Eigen::VectorXd trace(static_cast<Eigen::Index>(values.size()));
    for (std::size_t i = 0; i < values.size(); ++i) {
        trace(static_cast<Eigen::Index>(i)) = values[i];
    }
    return trace;
}

void stamp_single_attempt(
    GFN2Result& result,
    const XTBSccOptions& opts,
    int allocated_max_iter,
    const std::vector<double>& residuals,
    const std::string& solver) {
    result.scc_max_change_trace = residual_trace_vector(residuals);
    GFN2SCCAttempt attempt;
    attempt.solver = solver;
    attempt.allocated_max_iter = allocated_max_iter;
    attempt.n_iter = result.n_iter;
    attempt.ladder_charge_mixing = opts.charge_mixing;
    attempt.electronic_temperature = opts.electronic_temperature;
    attempt.scc_mixer = opts.scc_mixer;
    attempt.exit_reason = result.converged ? "converged" : "iteration_limit";
    attempt.scc_converged = result.converged;
    attempt.max_change_trace = result.scc_max_change_trace;
    result.attempts = {std::move(attempt)};
    result.selected_attempt_index = result.converged ? 0 : -1;
}

// Prefix a completed retry result with every attempt already consumed by the
// outer stabilization ladder.  The retry's electronic state remains the
// returned state while its diagnostics become a complete execution ledger.
void prepend_attempt_history(
    GFN2Result& retry,
    const GFN2Result& prefix) {
    Eigen::VectorXd trace(
        prefix.scc_max_change_trace.size()
        + retry.scc_max_change_trace.size());
    trace.head(prefix.scc_max_change_trace.size()) =
        prefix.scc_max_change_trace;
    trace.tail(retry.scc_max_change_trace.size()) =
        retry.scc_max_change_trace;
    retry.scc_max_change_trace = std::move(trace);

    const int attempt_offset = static_cast<int>(prefix.attempts.size());
    std::vector<GFN2SCCAttempt> attempts;
    attempts.reserve(prefix.attempts.size() + retry.attempts.size());
    attempts.insert(
        attempts.end(), prefix.attempts.begin(), prefix.attempts.end());
    attempts.insert(
        attempts.end(), retry.attempts.begin(), retry.attempts.end());
    retry.attempts = std::move(attempts);
    if (retry.selected_attempt_index >= 0) {
        retry.selected_attempt_index += attempt_offset;
    }
    retry.n_iter += prefix.n_iter;
}

}  // namespace

GFN2Result run_gfn2_xtb(
    const Molecule& mol,
    const GFN2ParameterSet& params,
    const XTBSccOptions& opts) {

    if (mol.multiplicity() != 1)
        throw std::invalid_argument("run_gfn2_xtb: only closed-shell supported");
    if (opts.scc_mixer == SCCMixer::Newton)
        throw std::invalid_argument(
            "run_gfn2_xtb: SCCMixer::Newton is unavailable for molecular "
            "GFN2; use Simple, DIIS, or Broyden");
    if (opts.scc_mixer == SCCMixer::BroydenEyert)
        throw std::invalid_argument(
            "run_gfn2_xtb: SCCMixer::BroydenEyert is unavailable for "
            "molecular GFN2 (SECCM-only); use Simple, DIIS, or Broyden");
    for (const auto& atom : mol.atoms())
        if (!params.has_element(atom.Z))
            throw std::runtime_error("run_gfn2_xtb: element Z="
                + std::to_string(atom.Z) + " not in parameter set");

    // n_primitives=0: GFN2-xTB per-element auto (H,He->3; else->4).
    BasisSet basis = SemiempiricalBasis::build(mol, params, 0);
    const int n_basis = static_cast<int>(basis.nbasis());
    const int n_atoms = static_cast<int>(mol.atoms().size());
    const auto& atoms = mol.atoms();

    Eigen::MatrixXd S = SemiempiricalHamiltonianBuilder::build_overlap(basis);
    Eigen::MatrixXd H0 = build_gfn2_hamiltonian_zero(basis, S, mol, params);

    double E_rep = 0.0;
    for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t b = a + 1; b < atoms.size(); ++b) {
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R > 1e-12) E_rep += params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
        }

    auto shell_info = gfn2_enumerate_shells(basis, mol, params);
    const int n_shells = static_cast<int>(shell_info.size());
    Eigen::MatrixXd gamma_shell = build_gfn2_shell_gamma(shell_info, mol, params);

    std::vector<int> ao_shell(n_basis, -1);
    for (int si = 0; si < n_shells; ++si)
        for (int i = 0; i < shell_info[si].n_funcs; ++i)
            ao_shell[shell_info[si].bf_start + i] = si;

    std::vector<int> ao_atom(n_basis, 0);
    for (int s = 0; s < static_cast<int>(basis.shells().size()); ++s) {
        int nf = basis.libint().shells()[s].size();
        int bf0 = basis.libint().shell2bf()[s];
        for (int i = 0; i < nf; ++i) ao_atom[bf0+i] = basis.shells()[s].atom_index;
    }

    GFN2MultipoleSet mp_int = build_gfn2_multipole_integrals(basis, mol, true);
    std::vector<Eigen::MatrixXd> aes_Dg, aes_Qg;
    if (opts.aes_faithful) {
        build_global_multipole_integrals(
            mol, S, mp_int, basis, ao_atom, n_basis, aes_Dg, aes_Qg);
    }

    // Faithful-AES per-atom parameters (experimental; only when aes_faithful).
    // CN-dependent multipole radius (Bannwarth-Ehlert-Grimme 2019; tblite gfn2.f90):
    //   mrad_A = rad_A + (rmax − rad_A)/(1 + exp(−aesexp·(CN_A − vcn_A − aesshift)))
    std::vector<double> dpol_at(n_atoms, 0.0), qpol_at(n_atoms, 0.0), mrad_at(n_atoms, 0.0);
    if (opts.aes_faithful) {
        // xtb's get_radcn consumes the GFN2 double-damped coordination
        // number (getCoordinationNumber, cnType%gfn), the same CN array the
        // H0 self-energy uses -- not the D4 dispersion CN.
        const Eigen::VectorXd cn = gfn2_h0_coordination_numbers(mol);
        const double aes_shift = 1.20, aes_exp = 4.0, aes_rmax = 5.0;
        for (int a = 0; a < n_atoms; ++a) {
            const auto* e = params.element_data(atoms[a].Z);
            dpol_at[a] = e ? e->dpol : 0.0;
            qpol_at[a] = e ? e->qpol : 0.0;
            const double rad = (e && e->mp_rad > 0.0) ? e->mp_rad : 3.0;  // fallback
            const double vcn = e ? e->mp_vcn : 0.0;
            const double t1 = std::exp(-aes_exp * (cn(a) - vcn - aes_shift));
            mrad_at[a] = rad + (aes_rmax - rad) / (1.0 + t1);
        }
    }

    Eigen::VectorXd n0_shell = Eigen::VectorXd::Zero(n_shells);
    for (int si = 0; si < n_shells; ++si) {
        const int Z = atoms[shell_info[si].atom_idx].Z;
        n0_shell(si) = gfn2_reference_occupation(Z, shell_info[si].l);
    }

    int n_val_elec = -mol.charge();
    for (int a = 0; a < n_atoms; ++a) n_val_elec += gfn2_valence_electrons(atoms[a].Z);
    const int n_occ = std::min(std::max(n_val_elec,0)/2, n_basis);

    // GFN2 globpar l-dependent GAM3 scaling (Bannwarth et al. 2019).
    // gam3s=1.0, gam3p=0.5, gam3d1=0.25, gam3d2=0.25.
    // Applied per shell so that p- and d-shell third-order terms are
    // correctly attenuated relative to the s-shell reference.
    static const double gam3_l_scale[4] = {1.0, 0.5, 0.25, 0.25};
    Eigen::VectorXd gam3_shell = Eigen::VectorXd::Zero(n_shells);
    for (int si = 0; si < n_shells; ++si) {
        auto* e = params.element_data(atoms[shell_info[si].atom_idx].Z);
        double gam3_raw = e ? e->gam3 : 0.0;
        int l = shell_info[si].l;
        gam3_shell(si) = gam3_raw * gam3_l_scale[std::min(l, 3)];
    }
    const bool has_gam3 = gam3_shell.cwiseAbs().maxCoeff() > 0.0;

    // --- Mixer ---
    // GFN2's atom-resolved third-order term (negative Γ for several elements,
    // e.g. Γ_O = -0.0517 after tblite's 0.1 scale factor) gives the charge SCC
    // a spurious over-polarized fixed
    // point alongside the physical one.  History-based accelerators
    // (Broyden/DIIS) launched from the neutral guess overshoot the
    // over-polarized first-iteration Mulliken charges and converge to the
    // spurious basin (O 2p over-filled to Δq≈+2.1, total E collapses by
    // ~0.25 Ha).  Damped simple mixing instead tracks the physical fixed point
    // from the neutral guess, so it is used whenever the third-order term is
    // active.  Without a third-order term (no Γ) the problem is single-fixed-
    // point and the configured accelerator is safe and fast.
    std::unique_ptr<ChargeMixer> mixer;
    std::unique_ptr<ChargeMixer> diis_mixer;
    const bool explicit_accelerator =
        opts.scc_mixer == SCCMixer::DIIS || opts.scc_mixer == SCCMixer::Broyden;
    // Hybrid polyalgorithm state (has_gam3 default path): damped simple
    // mixing tracks the physical basin for a fixed warmup and halves its
    // step on a stall; a damped DIIS then takes over from inside the basin.
    // DIIS engages only when the simple phase has actually contracted
    // (stalled/diverging phases skip it).  A DIIS phase that stops
    // improving (2-cycle or growth past its best residual) is permanently
    // abandoned: the driver restores the best DIIS charge state and
    // finishes with damped simple mixing.  All branches converge to the
    // same fixed point; only the iteration path differs.
    const bool hybrid = has_gam3 && !explicit_accelerator;
    std::string attempt_solver = "molecular_simple";
    if (hybrid) {
        attempt_solver = "molecular_hybrid";
    } else if (opts.scc_mixer == SCCMixer::DIIS) {
        attempt_solver = "molecular_diis";
    } else if (opts.scc_mixer == SCCMixer::Broyden) {
        attempt_solver = "molecular_broyden";
    }
    double hybrid_step = 0.1;
    bool diis_active = false;
    bool diis_failed = false;
    double residual_initial = -1.0;   // max_change at iteration 1
    double residual_best = -1.0;      // smallest simple-mixing residual
    double diis_best_residual = -1.0; // smallest residual during DIIS phase
    Eigen::VectorXd diis_best_dq;     // charge state at the DIIS best point
    int stall_count = 0;              // consecutive non-improving simple steps
    int diis_stall_count = 0;         // consecutive non-improving DIIS steps
    constexpr int kWarmupIters = 30;         // simple-mixing warmup before DIIS
    constexpr int kStallIters = 15;          // stall window before step halving
    constexpr double kStepFloor = 0.0125;    // adaptive halving floor
    constexpr int kDiisStallIters = 8;       // non-improving DIIS window before revert
    if (has_gam3 && !explicit_accelerator) {
        // Damped simple-mixing step from opts.charge_mixing (a caller may reduce
        // it and retry on a convergence failure).  Capped at 0.1: that is robust
        // across H2O/CO2/NH3/CH4/C2H4, while CO2 (large C→O charge transfer)
        // diverges above ~0.12 — so even if a caller asks for a larger step we
        // hold it at the safe ceiling rather than fall into the spurious basin.
        double requested = opts.charge_mixing > 0.0 ? opts.charge_mixing : 0.1;
        hybrid_step = std::min(requested, 0.1);
        mixer = std::make_unique<SimpleMixer>(hybrid_step);
    } else {
        switch (opts.scc_mixer) {
            case SCCMixer::DIIS:
                mixer = std::make_unique<ChargeDIISMixer>(
                    std::max(2, opts.mixer_memory),
                    opts.mixer_damping > 0 ? opts.mixer_damping : 0.5,
                    3); break;
            case SCCMixer::Broyden:
                mixer = std::make_unique<BroydenMixer>(
                    std::max(1, opts.mixer_memory),
                    opts.mixer_damping > 0 ? opts.mixer_damping : opts.charge_mixing); break;
            default:
                mixer = std::make_unique<SimpleMixer>(opts.charge_mixing); break;
        }
    }

    Eigen::VectorXd dq_shell = Eigen::VectorXd::Zero(n_shells);
    GFN2Result result;
    result.n_basis = n_basis;
    result.n_occ = n_occ;
    std::vector<double> max_change_trace;
    GFN2ShellMoments moments; moments.resize(n_shells);
    bool have_moments = false;
    // Faithful AES: xtb's first SCC iteration already carries the charge-only
    // AES channels (vd ~ -q_B dR/r^3, vqp ~ q_B T/r^5 — both depend on the
    // seeded charges alone, setvsdq runs with dipm = qp = 0 at iteration 1),
    // so the AES Fock must run from the first iteration with zero moments.
    // Gating on the first moment update (have_atom_moments) delayed the AES
    // channels by one iteration relative to xtb and pushed seeded starts off
    // xtb's SCC fixed point (GFN2-MOL-PARITY).
    std::vector<Vec3> atom_mu(static_cast<std::size_t>(n_atoms), Vec3{0, 0, 0});
    std::vector<Vec6> atom_th(static_cast<std::size_t>(n_atoms), Vec6{0, 0, 0, 0, 0, 0});
    // Damped AES-potential memory (faithful path): the on-site DPOL/QPOL
    // feedback is stiff, so each SCC iteration blends the fresh atom
    // charge/dipole/quadrupole potentials with the previous iterate
    // (opts.aes_damping). Path-only damping; the converged fixed point is
    // unchanged.
    Eigen::VectorXd vat_prev;
    std::vector<Vec3> vdp_prev;
    std::vector<Vec6> vqp_prev;
    bool have_aes_prev = false;
    const double aes_damp = std::max(0.0, std::min(1.0, opts.aes_damping));

    // max_iter is the total public SCC budget, not a per-attempt allowance.
    // The default 3600-step budget preserves the established molecular GFN2
    // stabilization ladder (700 hybrid-primary steps, 1800 slow T=0 retry,
    // 500 mild finite-T retry, and 600 high-T extended-period retry) while
    // explicit smaller caps fail closed at the requested iteration count.
    // The primary allowance grew 500 -> 700 when the default SCC path became
    // the simple+DIIS polyalgorithm: a stalled-then-halved simple phase plus
    // the fixed DIIS warmup costs a bounded number of iterations up front,
    // which the old 500-step window was cutting off for slow-start systems.
    const int total_max_iter = std::max(0, opts.max_iter);
    const bool use_stabilization =
        has_gam3 && opts.auto_stabilize && total_max_iter >= 2500;
    const int primary_max_iter =
        use_stabilization ? std::min(700, total_max_iter) : total_max_iter;

    for (int iter = 1; iter <= primary_max_iter; ++iter) {
        // Isotropic second-order (shell-resolved) potential: V²_l = Σ_l' γ_ll' Δq_l'.
        Eigen::VectorXd V_shell = gamma_shell * dq_shell;

        // Shell-resolved third-order on-site potential.  The driver stores
        // dq_l = pop_l - n0_l = -q_l, so the shell partial charge is
        // q_l = -dq_l and the potential added through the +0.5*S*(V_l+V_l')
        // SCC build is V3_l = -G_l*q_l^2 = -G_l*dq_l^2 with the l-dependent
        // GAM3 scaling (xtb: thirdorder.f90 addShift, shellGam branch —
        // per-SHELL charge, not the atomic charge).
        for (int si = 0; si < n_shells; ++si) {
            double G = gam3_shell(si);
            if (G != 0.0) {
                V_shell(si) += -G * dq_shell(si) * dq_shell(si);
            }
        }

        // Atomic partial charges (n0 - pop) for the faithful-AES potential.
        Eigen::VectorXd q_atom = Eigen::VectorXd::Zero(n_atoms);
        for (int si = 0; si < n_shells; ++si)
            q_atom(shell_info[si].atom_idx) -= dq_shell(si);

        // Anisotropic second-order (multipole / AES) potential — kept separate
        // so its energy (charge·field) can be formed cleanly in the assembly.
        // Anisotropic 2nd-order potential.  Default: the shipped ad-hoc
        // shell-resolved gamma^n kernel (self-consistent).  Faithful path
        // (experimental): V_mp stays zero here and the atom-resolved AES
        // potential is added below through dipole/quadrupole integral channels.
        Eigen::VectorXd V_mp = Eigen::VectorXd::Zero(n_shells);
        if (have_moments && !opts.aes_faithful) {
            for (int si = 0; si < n_shells; ++si) {
                int a = shell_info[si].atom_idx;
                for (int sj = 0; sj < n_shells; ++sj) {
                    int b = shell_info[sj].atom_idx;
                    if (a == b) continue;
                    double g = gamma_shell(si,sj);
                    if (g < 1e-15) continue;
                    double g3 = g*g*g, g5 = g3*g*g;
                    double dx = atoms[a].xyz[0]-atoms[b].xyz[0];
                    double dy = atoms[a].xyz[1]-atoms[b].xyz[1];
                    double dz = atoms[a].xyz[2]-atoms[b].xyz[2];
                    double R2 = dx*dx+dy*dy+dz*dz, R = std::sqrt(R2);
                    if (R < 1e-12) continue;
                    double invR3 = 1.0/(R2*R);
                    double pot_dip = -g3*(moments.ox(sj)*dx+moments.oy(sj)*dy+moments.oz(sj)*dz)*invR3;
                    double invR5 = invR3/R2;
                    double qxx=3*dx*dx-R2,qxy=3*dx*dy,qxz=3*dx*dz,qyy=3*dy*dy-R2,qyz=3*dy*dz,qzz=3*dz*dz-R2;
                    double pot_quad = 0.5*g5*(moments.txx(sj)*qxx+moments.txy(sj)*qxy*2
                        +moments.txz(sj)*qxz*2+moments.tyy(sj)*qyy+moments.tyz(sj)*qyz*2+moments.tzz(sj)*qzz)*invR5;
                    V_mp(si) += pot_dip + pot_quad;
                }
            }
        }
        V_shell += V_mp;

        Eigen::MatrixXd H_scc = H0;
        for (int mu = 0; mu < n_basis; ++mu) {
            int sm = ao_shell[mu];
            for (int nu = 0; nu < n_basis; ++nu) {
                int sn = ao_shell[nu];
                H_scc(mu,nu) += 0.5*S(mu,nu)*(V_shell(sm)+V_shell(sn));
            }
        }
        if (opts.aes_faithful) {
            Eigen::VectorXd vat;
            std::vector<Vec3> vdp;
            std::vector<Vec6> vqp;
            aes_potentials(mol, q_atom, atom_mu, atom_th,
                           dpol_at, qpol_at, mrad_at, vat, vdp, vqp);
            if (have_aes_prev && aes_damp < 1.0) {
                const double keep = 1.0 - aes_damp;
                vat = aes_damp * vat + keep * vat_prev;
                for (int a = 0; a < n_atoms; ++a) {
                    for (int k = 0; k < 3; ++k) {
                        vdp[a][k] = aes_damp * vdp[a][k]
                            + keep * vdp_prev[a][k];
                    }
                    for (int k = 0; k < 6; ++k) {
                        vqp[a][k] = aes_damp * vqp[a][k]
                            + keep * vqp_prev[a][k];
                    }
                }
            }
            vat_prev = vat;
            vdp_prev = vdp;
            vqp_prev = vqp;
            have_aes_prev = true;
            add_aes_potential_to_hamiltonian(
                H_scc, S, aes_Dg, aes_Qg, ao_atom, vat, vdp, vqp, n_basis);
        }

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(gfn2_diag_failure_message(
                mol, H_scc, S, V_shell, iter, n_basis, n_shells,
                n_val_elec, n_occ, solver.info()));
        }
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();
        Eigen::VectorXd occ = closed_shell_occupations(
            eps, 2 * n_occ, opts.electronic_temperature);
        Eigen::MatrixXd D = C * occ.asDiagonal() * C.transpose();
        Eigen::MatrixXd DS = D*S;

        Eigen::VectorXd dq_new = Eigen::VectorXd::Zero(n_shells);
        for (int mu = 0; mu < n_basis; ++mu) { int si = ao_shell[mu]; if (si>=0) dq_new(si)+=DS(mu,mu); }
        for (int si = 0; si < n_shells; ++si) dq_new(si) -= n0_shell(si);

        double max_change = (dq_new - dq_shell).cwiseAbs().maxCoeff();
        max_change_trace.push_back(max_change);

        if (max_change < opts.conv_tol_charge) {
            // Converged. Evaluate the energy functional at the converged
            // density's own Mulliken charges for self-consistency.
            dq_shell = dq_new;
            Eigen::VectorXd dq_atom = Eigen::VectorXd::Zero(n_atoms);
            for (int mu = 0; mu < n_basis; ++mu) dq_atom(ao_atom[mu]) += DS(mu,mu);
            for (int a = 0; a < n_atoms; ++a)
                dq_atom(a) = static_cast<double>(gfn2_valence_electrons(atoms[a].Z)) - dq_atom(a);

            // GFN2-xTB total energy as an explicit functional of the converged
            // density and charges (Bannwarth, Ehlert & Grimme, JCTC 2019,
            // Eqs. 1-9), NOT the band-structure trace.  E_rep is a separate
            // zeroth-order classical pairwise term in the published functional
            // (Eqs. 8-9, Sec. 2.1.2: it "does not contribute to the
            // tight-binding electronic energy") - the damped
            // effective-nuclear-charge repulsion - not something H0
            // "already folds in": live xtb 6.7.1 prints
            //   total = SCC + aniso ES + aniso XC + dispersion + repulsion
            // and vibe-qc's E_rep matches xtb's repulsion to ~1e-8 Ha, so
            // dropping E_rep removes the short-range repulsive wall (water
            // O-H PES collapse, 2026-08-12 audit; restored with the
            // repulsive gradient/stress).  Using the bare-H0 band
            // term Sum P.H0 plus explicit 2nd/3rd-order + AES corrections avoids
            // the double counting of the previous
            //   E = Sum 2eps_i(H_scc) - E_es + E_aes + E_3rd
            // assembly, where the SCC potential already folded into Sum 2eps_i was
            // re-added a second time through E_aes/E_3rd.
            double E_band0 = (D.cwiseProduct(H0)).sum();             // Sum_munu P_munu H0_munu
            double E_es  = 0.5 * dq_shell.dot(gamma_shell*dq_shell); // 2nd order, isotropic
            // 2nd-order anisotropic (AES).  Default: ad-hoc shell-resolved
            // (½ Δq·V_mp).  Faithful (experimental): functional of the
            // converged density's atomic CAMM + partial charges (dq_atom),
            // paired with the atom-resolved AES potential in the SCC Fock.
            double E_aes;
            if (opts.aes_faithful) {
                std::vector<Vec3> mu_at; std::vector<Vec6> th_at;
                compute_atom_moments(D, S, aes_Dg, aes_Qg, mol, ao_atom,
                                    n_atoms, n_basis, mu_at, th_at);
                E_aes = aes_energy(mol, dq_atom, mu_at, th_at,
                                   dpol_at, qpol_at, mrad_at);
            } else {
                E_aes = 0.5 * dq_shell.dot(V_mp);
            }
            double E_3rd = 0.0;  // 3rd order, shell-resolved (xtb thirdorder.f90)
            for (int si = 0; si < n_shells; ++si) {
                double G = gam3_shell(si);
                if (G == 0.0) continue;
                double q = -dq_shell(si);  // shell partial charge n0−pop
                E_3rd += G * q * q * q / 3.0;
            }

            result.energy = E_band0 + E_es + E_aes + E_3rd + E_rep;
            double entropy = 0.0;
            if (opts.electronic_temperature > 0.0) {
                for (int orbital = 0; orbital < occ.size(); ++orbital) {
                    const double fraction = std::clamp(
                        occ(orbital) / 2.0, 1.0e-300, 1.0 - 1.0e-15);
                    entropy -= 2.0 * (
                        fraction * std::log(fraction)
                        + (1.0 - fraction) * std::log(1.0 - fraction));
                }
            }
            result.entropy = entropy;
            result.smearing_temperature = opts.electronic_temperature;
            result.free_energy = result.energy
                - opts.electronic_temperature * entropy;
            result.e_electronic = E_band0 + E_es + E_aes + E_3rd;
            result.e_repulsive = E_rep; result.e_scc = E_es;
            result.e_band0 = E_band0; result.e_aes = E_aes; result.e_3rd = E_3rd;
            // Debug: RMS of V_mp and shell dipoles
            result.vmp_rms = std::sqrt(V_mp.squaredNorm() / std::max(1, n_shells));
            double dip2 = 0.0;
            for (int si = 0; si < n_shells; ++si)
                dip2 += moments.ox(si)*moments.ox(si)
                      + moments.oy(si)*moments.oy(si)
                      + moments.oz(si)*moments.oz(si);
            result.shell_dip_rms = std::sqrt(dip2 / std::max(1, n_shells));
            result.mo_energies = std::move(eps); result.mo_coeffs = std::move(C);
            result.density = std::move(D); result.overlap = S;
            result.hamiltonian = std::move(H_scc); result.charges = std::move(dq_atom);
            result.n_basis = n_basis; result.n_occ = n_occ;
            result.n_iter = iter; result.converged = true;
            stamp_single_attempt(
                result,
                opts,
                primary_max_iter,
                max_change_trace,
                attempt_solver);
            return result;
        }

        // Hybrid polyalgorithm bookkeeping (has_gam3 default path).
        if (hybrid) {
            if (residual_initial < 0.0) {
                residual_initial = max_change;
                residual_best = max_change;
            }
            if (diis_active) {
                if (max_change < diis_best_residual) {
                    diis_best_residual = max_change;
                    diis_best_dq = dq_shell;
                    diis_stall_count = 0;
                } else {
                    ++diis_stall_count;
                    if (diis_stall_count >= kDiisStallIters) {
                        // DIIS stalled or cycled: permanently abandon it,
                        // restore its best charge state, and finish with
                        // damped simple mixing at the current step.  The
                        // fresh SimpleMixer blends from the restored state
                        // (its caller-supplied dq vector), so the handoff
                        // is a soft continuation, not a charge jump.
                        diis_active = false;
                        diis_failed = true;
                        dq_shell = diis_best_dq;
                        mixer = std::make_unique<SimpleMixer>(hybrid_step);
                    }
                }
            } else {
                // Simple-phase bookkeeping (pre-DIIS warmup and post-revert
                // both land here, so the adaptive halving stays armed).
                if (max_change < residual_best) {
                    residual_best = max_change;
                    stall_count = 0;
                } else {
                    ++stall_count;
                    if (stall_count >= kStallIters) {
                        // Simple mixing stalled: halve the step (floor 0.0125).
                        hybrid_step = std::max(kStepFloor, 0.5 * hybrid_step);
                        mixer = std::make_unique<SimpleMixer>(hybrid_step);
                        stall_count = 0;
                        residual_best = max_change;
                    }
                }
                if (!diis_failed && iter >= kWarmupIters
                        && max_change < 0.5 * residual_initial) {
                    // Hand off to DIIS only from a genuinely contracting
                    // simple phase (a stalled/diverging phase skips DIIS and
                    // continues with the halved step).
                    diis_mixer = std::make_unique<ChargeDIISMixer>(
                        std::max(2, opts.mixer_memory),
                        opts.mixer_damping > 0 ? opts.mixer_damping : 0.5,
                        3);
                    diis_active = true;
                    diis_stall_count = 0;
                    diis_best_residual = max_change;
                    diis_best_dq = dq_shell;
                }
            }
        }

        // Not converged: mix the charges and refresh the multipole moments.
        ChargeMixer* active = diis_active ? diis_mixer.get() : mixer.get();
        active->mix(dq_shell, dq_new, iter);
        moments = compute_shell_multipole_moments(
            D, basis, mol, mp_int, ao_shell, n0_shell, n_shells);
        moments.q = dq_shell; have_moments = true;
        if (opts.aes_faithful) {
            compute_atom_moments(D, S, aes_Dg, aes_Qg, mol, ao_atom,
                                n_atoms, n_basis, atom_mu, atom_th);
        }
    }
    result.n_iter = primary_max_iter;
    result.smearing_temperature = opts.electronic_temperature;
    result.converged = false;
    stamp_single_attempt(
        result, opts, primary_max_iter, max_change_trace, attempt_solver);

    if (use_stabilization) {
        struct Retry {
            double charge_mixing;
            int max_iter;
            double electronic_temperature;
        };
        // BUG 45: the former ladder ran three increasingly long T=0 retries
        // (3000 + 5000 + 12000 iterations) before trying the finite-temperature
        // path that actually converges the affected neutral heterocycles in a
        // few hundred steps.  Eleven article-validation jobs consequently
        // reported 9828--20853 cumulative iterations.  One slow T=0 attempt
        // preserves the cyclopropene solution, then one short mild finite-T
        // attempt handles the heterocycles.  Extended-period TAE-PTComp anions
        // (K/Ca and post-Ar p-block cases) can still oscillate after the H0
        // covalent-radius fix because their near-degenerate frontier states
        // need stronger occupation smoothing, so they receive one final
        // bounded high-T retry.  Light-element automatic stabilization remains
        // bounded to 3000 SCC iterations; extended-period systems are bounded
        // to 3600 SCC iterations with the default max_iter.
        std::vector<Retry> retries = {
            {0.01, 1800, 0.0},
            {0.05, 500, 0.005},
        };
        if (has_extended_period_element(mol)) {
            retries.push_back({0.05, 600, 0.05});
        }
        for (const auto& retry : retries) {
            const int remaining_iters = total_max_iter - result.n_iter;
            if (remaining_iters <= 0) break;
            XTBSccOptions ropts = opts;
            ropts.charge_mixing = retry.charge_mixing;
            ropts.max_iter = std::min(retry.max_iter, remaining_iters);
            ropts.electronic_temperature = retry.electronic_temperature;
            ropts.scc_mixer = SCCMixer::Simple;
            ropts.auto_stabilize = false;
            GFN2Result r = run_gfn2_xtb(mol, params, ropts);
            prepend_attempt_history(r, result);
            if (r.converged) {
                return r;
            }
            result = std::move(r);
        }
    }
    return result;
}

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
