#include "vibeqc/semiempirical/methods/xtb/gfn2_aes.hpp"

#include <cmath>
#include <stdexcept>

namespace vibeqc {
namespace semiempirical {
namespace xtb {

namespace {

// 6-component (xx, xy, xz, yy, yz, zz) index of the symmetric pair (i, j).
inline int sym6(int i, int j) {
    static const int table[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return table[i][j];
}

inline Eigen::Matrix3d full_tensor(const Eigen::MatrixXd& rows, int atom) {
    Eigen::Matrix3d t;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) t(i, j) = rows(atom, 3 * i + j);
    return t;
}

inline Eigen::Matrix3d theta_tensor(const GFN2AesMoments& moments, int atom) {
    Eigen::Matrix3d t;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) t(i, j) = moments.theta(atom, sym6(i, j));
    return t;
}

// Damped pair kernels of Eq. 28 written without the R^-n prefactor split:
//   f3 = fdmp3 / R^3 = 1 / (R^3 + 6 R0^3),
//   f5 = fdmp5 / R^5 = 1 / (R^5 + 6 R0^4 R),
// with R0 = 1/2 (R0_A' + R0_B') (Eq. 29 radii).
struct PairKernels {
    double f3 = 0.0, f5 = 0.0;
    double df3_dr = 0.0, df5_dr = 0.0;    // d/dR
    double df3_drad = 0.0, df5_drad = 0.0;  // d/dR0
};

PairKernels pair_kernels(double r, double rad) {
    PairKernels k;
    const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;
    const double rad2 = rad * rad, rad3 = rad2 * rad, rad4 = rad3 * rad;
    const double den3 = r3 + 6.0 * rad3;
    const double den5 = r5 + 6.0 * rad4 * r;
    k.f3 = 1.0 / den3;
    k.f5 = 1.0 / den5;
    k.df3_dr = -3.0 * r2 * k.f3 * k.f3;
    k.df5_dr = -(5.0 * r4 + 6.0 * rad4) * k.f5 * k.f5;
    k.df3_drad = -18.0 * rad2 * k.f3 * k.f3;
    k.df5_drad = -24.0 * rad3 * r * k.f5 * k.f5;
    return k;
}

void require_record(const ImageRecord& record, int n_atoms) {
    if (record.a < 0 || record.a >= n_atoms || record.b < 0
        || record.b >= n_atoms) {
        throw std::invalid_argument(
            "GFN2 AES image record refers to an out-of-range atom");
    }
    if (!record.shift.allFinite() || !std::isfinite(record.weight)) {
        throw std::invalid_argument("GFN2 AES image record must be finite");
    }
}

}  // namespace

GFN2AesAtomParameters gfn2_aes_atom_parameters(
    const std::vector<Atom>& atoms,
    const GFN2ParameterSet& params,
    const Eigen::VectorXd& coordination_numbers) {
    const int n_atoms = static_cast<int>(atoms.size());
    if (coordination_numbers.size() != n_atoms) {
        throw std::invalid_argument(
            "gfn2_aes_atom_parameters: coordination-number length mismatch");
    }
    GFN2AesAtomParameters out;
    out.dpol = Eigen::VectorXd::Zero(n_atoms);
    out.qpol = Eigen::VectorXd::Zero(n_atoms);
    out.mrad = Eigen::VectorXd::Zero(n_atoms);
    out.dmrad_dcn = Eigen::VectorXd::Zero(n_atoms);
    for (int a = 0; a < n_atoms; ++a) {
        const auto* e = params.element_data(atoms[static_cast<std::size_t>(a)].Z);
        out.dpol(a) = e ? e->dpol : 0.0;
        out.qpol(a) = e ? e->qpol : 0.0;
        // Eq. 29: R0_A' = R0_A + (R_max - R0_A) / (1 + exp(-4 (CN'_A - N_val - Delta_val)))
        const double rad = (e && e->mp_rad > 0.0) ? e->mp_rad : 3.0;
        const double vcn = e ? e->mp_vcn : 0.0;
        const double t1 = std::exp(
            -kGFN2AesExp * (coordination_numbers(a) - vcn - kGFN2AesShift));
        out.mrad(a) = rad + (kGFN2AesRmax - rad) / (1.0 + t1);
        out.dmrad_dcn(a) = (kGFN2AesRmax - rad) * kGFN2AesExp * t1
            / ((1.0 + t1) * (1.0 + t1));
    }
    return out;
}

GFN2AesMoments gfn2_aes_moments(
    const Eigen::MatrixXd& density,
    const GFN2MultipoleLatticeSums& sums,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const Eigen::VectorXd& q_atom) {
    const int n_atoms = static_cast<int>(positions.size());
    const int n_basis = static_cast<int>(density.rows());
    if (static_cast<int>(ao_atom.size()) != n_basis || sums.n_basis != n_basis
        || q_atom.size() != n_atoms) {
        throw std::invalid_argument("gfn2_aes_moments: dimension mismatch");
    }
    GFN2AesMoments m;
    m.q = q_atom;
    m.mu = Eigen::MatrixXd::Zero(n_atoms, 3);
    Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(n_atoms, 6);
    // Eq. 27b-c with the bra-row partitioning: every AO pair with its bra on
    // atom A contributes to A with origin R_A.  D and Q are global-origin
    // integrals summed over the ket's lattice images.
    for (int p = 0; p < n_basis; ++p) {
        const int A = ao_atom[static_cast<std::size_t>(p)];
        const Eigen::Vector3d& R = positions[static_cast<std::size_t>(A)];
        for (int q = 0; q < n_basis; ++q) {
            const double P = density(p, q);
            if (P == 0.0) continue;
            const double S = sums.S0(p, q);
            const double D[3] = {sums.D0[0](p, q), sums.D0[1](p, q),
                                 sums.D0[2](p, q)};
            for (int k = 0; k < 3; ++k) m.mu(A, k) += P * (R(k) * S - D[k]);
            for (int k = 0; k < 3; ++k) {
                for (int l = k; l < 3; ++l) {
                    const int kl = sym6(k, l);
                    raw(A, kl) += P * (R(k) * D[l] + R(l) * D[k]
                                       - R(k) * R(l) * S - sums.Q0[kl](p, q));
                }
            }
        }
    }
    // Eq. 26 / xtb mmompop trace removal: Theta = 3/2 theta - 1/2 tr(theta) 1.
    m.theta = 1.5 * raw;
    for (int A = 0; A < n_atoms; ++A) {
        const double half_trace = 0.5 * (raw(A, 0) + raw(A, 3) + raw(A, 5));
        m.theta(A, 0) -= half_trace;
        m.theta(A, 3) -= half_trace;
        m.theta(A, 5) -= half_trace;
    }
    return m;
}

double gfn2_aes_energy(
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesMoments& moments,
    const GFN2AesAtomParameters& params,
    const ImageRecordSource& records) {
    const int n_atoms = static_cast<int>(positions.size());
    double energy = 0.0;
    // Eq. 31 on-site anisotropic XC.
    for (int A = 0; A < n_atoms; ++A) {
        double t2 = 0.0;
        for (int k = 0; k < 6; ++k) {
            t2 += kAesQuadrupoleWeights[static_cast<std::size_t>(k)]
                * moments.theta(A, k) * moments.theta(A, k);
        }
        energy += params.dpol(A) * moments.mu.row(A).squaredNorm()
            + params.qpol(A) * t2;
    }
    // Eq. 25 over directed records, 1/2 per record.
    records([&](const ImageRecord& rec) {
        require_record(rec, n_atoms);
        const int a = rec.a, b = rec.b;
        const Eigen::Vector3d v = positions[static_cast<std::size_t>(a)]
            - (positions[static_cast<std::size_t>(b)] + rec.shift);
        const double r = v.norm();
        if (r < 1.0e-12) {
            throw std::invalid_argument(
                "GFN2 AES image record has a zero pair displacement");
        }
        const PairKernels k = pair_kernels(r, 0.5 * (params.mrad(a) + params.mrad(b)));
        const Eigen::Vector3d mu_a = moments.mu.row(a).transpose();
        const Eigen::Vector3d mu_b = moments.mu.row(b).transpose();
        const Eigen::Matrix3d th_a = theta_tensor(moments, a);
        const Eigen::Matrix3d th_b = theta_tensor(moments, b);
        const double qa = moments.q(a), qb = moments.q(b);
        const double lambda3 = qa * mu_b.dot(v) - qb * mu_a.dot(v);
        const double lambda5 = mu_a.dot(mu_b) * r * r
            - 3.0 * mu_a.dot(v) * mu_b.dot(v)
            + qa * v.dot(th_b * v) + qb * v.dot(th_a * v);
        energy += 0.5 * rec.weight * (k.f3 * lambda3 + k.f5 * lambda5);
    });
    return energy;
}

GFN2AesPotentials gfn2_aes_potentials(
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesMoments& moments,
    const GFN2AesAtomParameters& params,
    const ImageRecordSource& records) {
    const int n_atoms = static_cast<int>(positions.size());
    GFN2AesPotentials pot;
    pot.v = Eigen::VectorXd::Zero(n_atoms);
    pot.w = Eigen::MatrixXd::Zero(n_atoms, 3);
    std::vector<Eigen::Matrix3d> G(static_cast<std::size_t>(n_atoms),
                                   Eigen::Matrix3d::Zero());
    // Inter-site derivatives.  With reverse-symmetric records, the two
    // appearances of each physical pair in 1/2 sum_records w e give the same
    // contribution, so the derivative of the energy with respect to the
    // moments of atom A is the full (unhalved) sum over the records whose
    // first atom is A.
    records([&](const ImageRecord& rec) {
        require_record(rec, n_atoms);
        const int a = rec.a, b = rec.b;
        const Eigen::Vector3d v = positions[static_cast<std::size_t>(a)]
            - (positions[static_cast<std::size_t>(b)] + rec.shift);
        const double r = v.norm();
        if (r < 1.0e-12) {
            throw std::invalid_argument(
                "GFN2 AES image record has a zero pair displacement");
        }
        const PairKernels k = pair_kernels(r, 0.5 * (params.mrad(a) + params.mrad(b)));
        const Eigen::Vector3d mu_b = moments.mu.row(b).transpose();
        const Eigen::Matrix3d th_b = theta_tensor(moments, b);
        const double qb = moments.q(b);
        const double w = rec.weight;
        pot.v(a) += w * (k.f3 * mu_b.dot(v) + k.f5 * v.dot(th_b * v));
        pot.w.row(a) += w * (-k.f3 * qb * v
                             + k.f5 * (mu_b * r * r - 3.0 * v * mu_b.dot(v)))
            .transpose();
        G[static_cast<std::size_t>(a)] += w * k.f5 * qb * (v * v.transpose());
    });
    // On-site Eq. 31 derivatives, then the chain to the raw tensor:
    // Theta = 3/2 theta - 1/2 tr(theta) 1  =>  dE/dtheta = 3/2 G - 1/2 tr(G) 1.
    pot.t_raw = Eigen::MatrixXd::Zero(n_atoms, 9);
    for (int A = 0; A < n_atoms; ++A) {
        pot.w.row(A) += 2.0 * params.dpol(A) * moments.mu.row(A);
        Eigen::Matrix3d Gfull = G[static_cast<std::size_t>(A)]
            + 2.0 * params.qpol(A) * theta_tensor(moments, A);
        const Eigen::Matrix3d t_raw = 1.5 * Gfull
            - 0.5 * Gfull.trace() * Eigen::Matrix3d::Identity();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) pot.t_raw(A, 3 * i + j) = t_raw(i, j);
    }
    return pot;
}

GFN2AesFockSide gfn2_aes_fock_side(
    const GFN2AesPotentials& potentials, int atom,
    const Eigen::Vector3d& origin) {
    // F^X(O) = -v S - w.(D - O S) - sum_ij t_ij (Q_ij - O_i D_j - O_j D_i + O_i O_j S)
    //        = [-v + w.O - O^T t O] S + [-w + 2 t O] . D - sum_ij t_ij Q_ij
    const Eigen::Matrix3d t = full_tensor(potentials.t_raw, atom);
    const Eigen::Vector3d w = potentials.w.row(atom).transpose();
    const Eigen::Vector3d tO = t * origin;
    GFN2AesFockSide side;
    side.cS = -potentials.v(atom) + w.dot(origin) - origin.dot(tO);
    side.cD = -w + 2.0 * tO;
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            side.cQ[static_cast<std::size_t>(sym6(i, j))] =
                -kAesQuadrupoleWeights[static_cast<std::size_t>(sym6(i, j))]
                * t(i, j);
        }
    }
    return side;
}

void gfn2_aes_add_fock(
    Eigen::MatrixXd& hamiltonian,
    const GFN2MultipoleLatticeSums& sums,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesPotentials& potentials) {
    const int n_atoms = static_cast<int>(positions.size());
    const int n_basis = sums.n_basis;
    if (hamiltonian.rows() != n_basis || hamiltonian.cols() != n_basis
        || static_cast<int>(ao_atom.size()) != n_basis) {
        throw std::invalid_argument("gfn2_aes_add_fock: dimension mismatch");
    }
    // Home-origin sides (the bra side; also the ket side's g-independent part).
    std::vector<GFN2AesFockSide> home(static_cast<std::size_t>(n_atoms));
    std::vector<Eigen::Matrix3d> t_of(static_cast<std::size_t>(n_atoms));
    std::vector<Eigen::Vector3d> w_minus_2tR(static_cast<std::size_t>(n_atoms));
    for (int A = 0; A < n_atoms; ++A) {
        home[static_cast<std::size_t>(A)] =
            gfn2_aes_fock_side(potentials, A, positions[static_cast<std::size_t>(A)]);
        t_of[static_cast<std::size_t>(A)] = full_tensor(potentials.t_raw, A);
        w_minus_2tR[static_cast<std::size_t>(A)] =
            potentials.w.row(A).transpose()
            - 2.0 * t_of[static_cast<std::size_t>(A)]
                * positions[static_cast<std::size_t>(A)];
    }
    for (int p = 0; p < n_basis; ++p) {
        const int A = ao_atom[static_cast<std::size_t>(p)];
        const GFN2AesFockSide& sa = home[static_cast<std::size_t>(A)];
        for (int q = 0; q < n_basis; ++q) {
            const int B = ao_atom[static_cast<std::size_t>(q)];
            const GFN2AesFockSide& sb = home[static_cast<std::size_t>(B)];
            const Eigen::Matrix3d& tB = t_of[static_cast<std::size_t>(B)];
            const Eigen::Vector3d& uB = w_minus_2tR[static_cast<std::size_t>(B)];
            // g-independent part of both sides.
            double value = (sa.cS + sb.cS) * sums.S0(p, q);
            for (int j = 0; j < 3; ++j) {
                value += (sa.cD(j) + sb.cD(j)) * sums.D0[static_cast<std::size_t>(j)](p, q);
            }
            for (int k = 0; k < 6; ++k) {
                value += (sa.cQ[static_cast<std::size_t>(k)]
                          + sb.cQ[static_cast<std::size_t>(k)])
                    * sums.Q0[static_cast<std::size_t>(k)](p, q);
            }
            // Ket-origin shift by g on the B side:
            //   cS_B(R_B + g) = cS_B(R_B) + (w_B - 2 t_B R_B) . g - g^T t_B g
            //   cD_B(R_B + g) = cD_B(R_B) + 2 t_B g
            for (int i = 0; i < 3; ++i) {
                value += uB(i) * sums.S1[static_cast<std::size_t>(i)](p, q);
                for (int j = 0; j < 3; ++j) {
                    value -= tB(i, j) * sums.S2[static_cast<std::size_t>(sym6(i, j))](p, q);
                    value += 2.0 * tB(i, j)
                        * sums.D1[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)](p, q);
                }
            }
            hamiltonian(p, q) += 0.5 * value;
        }
    }
}

GFN2AesKernelDerivatives gfn2_aes_kernel_derivatives(
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesMoments& moments,
    const GFN2AesAtomParameters& params,
    const ImageRecordSource& records) {
    const int n_atoms = static_cast<int>(positions.size());
    GFN2AesKernelDerivatives out;
    out.gradient = Eigen::MatrixXd::Zero(n_atoms, 3);
    out.dE_dmrad = Eigen::VectorXd::Zero(n_atoms);
    records([&](const ImageRecord& rec) {
        require_record(rec, n_atoms);
        const int a = rec.a, b = rec.b;
        const Eigen::Vector3d v = positions[static_cast<std::size_t>(a)]
            - (positions[static_cast<std::size_t>(b)] + rec.shift);
        const double r = v.norm();
        if (r < 1.0e-12) {
            throw std::invalid_argument(
                "GFN2 AES image record has a zero pair displacement");
        }
        const PairKernels k = pair_kernels(r, 0.5 * (params.mrad(a) + params.mrad(b)));
        const Eigen::Vector3d mu_a = moments.mu.row(a).transpose();
        const Eigen::Vector3d mu_b = moments.mu.row(b).transpose();
        const Eigen::Matrix3d th_a = theta_tensor(moments, a);
        const Eigen::Matrix3d th_b = theta_tensor(moments, b);
        const double qa = moments.q(a), qb = moments.q(b);
        const double mav = mu_a.dot(v), mbv = mu_b.dot(v);
        const double lambda3 = qa * mbv - qb * mav;
        const double lambda5 = mu_a.dot(mu_b) * r * r - 3.0 * mav * mbv
            + qa * v.dot(th_b * v) + qb * v.dot(th_a * v);
        // de/dv at fixed moments.
        const Eigen::Vector3d de_dv =
            k.df3_dr * (v / r) * lambda3 + k.f3 * (qa * mu_b - qb * mu_a)
            + k.df5_dr * (v / r) * lambda5
            + k.f5 * (2.0 * mu_a.dot(mu_b) * v - 3.0 * mu_a * mbv
                      - 3.0 * mu_b * mav + 2.0 * qa * (th_b * v)
                      + 2.0 * qb * (th_a * v));
        const Eigen::Vector3d f = 0.5 * rec.weight * de_dv;
        out.gradient.row(a) += f.transpose();
        out.gradient.row(b) -= f.transpose();
        out.strain += f * v.transpose();
        // de/dR0 through the damping (R0 = 1/2 (R0_a + R0_b)).
        const double de_drad = k.df3_drad * lambda3 + k.df5_drad * lambda5;
        out.dE_dmrad(a) += 0.25 * rec.weight * de_drad;
        out.dE_dmrad(b) += 0.25 * rec.weight * de_drad;
    });
    return out;
}

GFN2AesOriginDerivatives gfn2_aes_origin_derivatives(
    const Eigen::MatrixXd& density,
    const GFN2MultipoleLatticeSums& sums,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesPotentials& potentials) {
    const int n_atoms = static_cast<int>(positions.size());
    const int n_basis = sums.n_basis;
    GFN2AesOriginDerivatives out;
    out.gradient = Eigen::MatrixXd::Zero(n_atoms, 3);
    std::vector<Eigen::Matrix3d> t_of(static_cast<std::size_t>(n_atoms));
    std::vector<Eigen::Vector3d> u_of(static_cast<std::size_t>(n_atoms));
    for (int A = 0; A < n_atoms; ++A) {
        t_of[static_cast<std::size_t>(A)] = full_tensor(potentials.t_raw, A);
        u_of[static_cast<std::size_t>(A)] = potentials.w.row(A).transpose()
            - 2.0 * t_of[static_cast<std::size_t>(A)]
                * positions[static_cast<std::size_t>(A)];
    }
    // d/dO of [cS(O) S + cD(O) . D] = (w - 2 t O) S + 2 t D, with O = R_A on
    // the bra side and O = R_B + g on the ket side (whose g-dependence adds
    // -2 t g S to the first bracket).
    for (int p = 0; p < n_basis; ++p) {
        const int A = ao_atom[static_cast<std::size_t>(p)];
        const Eigen::Matrix3d& tA = t_of[static_cast<std::size_t>(A)];
        const Eigen::Vector3d& uA = u_of[static_cast<std::size_t>(A)];
        for (int q = 0; q < n_basis; ++q) {
            const double P = density(p, q);
            if (P == 0.0) continue;
            const int B = ao_atom[static_cast<std::size_t>(q)];
            const Eigen::Matrix3d& tB = t_of[static_cast<std::size_t>(B)];
            const Eigen::Vector3d& uB = u_of[static_cast<std::size_t>(B)];
            const double S0 = sums.S0(p, q);
            Eigen::Vector3d D0;
            for (int j = 0; j < 3; ++j) D0(j) = sums.D0[static_cast<std::size_t>(j)](p, q);
            Eigen::Vector3d S1;
            for (int j = 0; j < 3; ++j) S1(j) = sums.S1[static_cast<std::size_t>(j)](p, q);
            // Bra side (origin R_A, no images).
            out.gradient.row(A) += (0.5 * P * (uA * S0 + 2.0 * tA * D0)).transpose();
            // Ket side (origin R_B + g summed over images).
            out.gradient.row(B) +=
                (0.5 * P * (uB * S0 - 2.0 * tB * S1 + 2.0 * tB * D0)).transpose();
            // Image virial of the ket side: sum_g f_B(g) (x) g.
            for (int i = 0; i < 3; ++i) {
                for (int m = 0; m < 3; ++m) {
                    double value = uB(i) * S1(m);
                    for (int j = 0; j < 3; ++j) {
                        value -= 2.0 * tB(i, j)
                            * sums.S2[static_cast<std::size_t>(sym6(j, m))](p, q);
                        value += 2.0 * tB(i, j)
                            * sums.D1[static_cast<std::size_t>(m)][static_cast<std::size_t>(j)](p, q);
                    }
                    out.image_virial(i, m) += 0.5 * P * value;
                }
            }
        }
    }
    return out;
}

void gfn2_aes_fill_derivative_weights(
    int /*cell_index*/,
    const Eigen::Vector3d& shift,
    const Eigen::MatrixXd& density,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesPotentials& potentials,
    Eigen::MatrixXd& weight_S,
    std::array<Eigen::MatrixXd, 3>& weight_D,
    std::array<Eigen::MatrixXd, 6>& weight_Q) {
    const int n_atoms = static_cast<int>(positions.size());
    const int n_basis = static_cast<int>(density.rows());
    std::vector<GFN2AesFockSide> bra(static_cast<std::size_t>(n_atoms));
    std::vector<GFN2AesFockSide> ket(static_cast<std::size_t>(n_atoms));
    for (int A = 0; A < n_atoms; ++A) {
        bra[static_cast<std::size_t>(A)] = gfn2_aes_fock_side(
            potentials, A, positions[static_cast<std::size_t>(A)]);
        ket[static_cast<std::size_t>(A)] = gfn2_aes_fock_side(
            potentials, A, positions[static_cast<std::size_t>(A)] + shift);
    }
    weight_S.setZero(n_basis, n_basis);
    for (auto& w : weight_D) w.setZero(n_basis, n_basis);
    for (auto& w : weight_Q) w.setZero(n_basis, n_basis);
    for (int p = 0; p < n_basis; ++p) {
        const GFN2AesFockSide& sa = bra[static_cast<std::size_t>(ao_atom[static_cast<std::size_t>(p)])];
        for (int q = 0; q < n_basis; ++q) {
            const double P = density(p, q);
            if (P == 0.0) continue;
            const GFN2AesFockSide& sb = ket[static_cast<std::size_t>(ao_atom[static_cast<std::size_t>(q)])];
            weight_S(p, q) = 0.5 * P * (sa.cS + sb.cS);
            for (int j = 0; j < 3; ++j) {
                weight_D[static_cast<std::size_t>(j)](p, q) = 0.5 * P * (sa.cD(j) + sb.cD(j));
            }
            for (int k = 0; k < 6; ++k) {
                weight_Q[static_cast<std::size_t>(k)](p, q) = 0.5 * P
                    * (sa.cQ[static_cast<std::size_t>(k)] + sb.cQ[static_cast<std::size_t>(k)]);
            }
        }
    }
}

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
