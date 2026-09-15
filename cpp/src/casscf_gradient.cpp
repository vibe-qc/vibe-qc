#include "vibeqc/casscf_gradient.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <vector>

namespace vibeqc {

CasscfGradientResult casscf_orbital_gradient(
    const Eigen::MatrixXd& h1,
    const Eigen::VectorXd& eri_chem_flat,
    const Eigen::MatrixXd& dm1,
    const Eigen::VectorXd& dm2_flat,
    int n_core,
    int n_act,
    int norb,
    const std::vector<std::pair<int, int>>& pairs) {

    // Unpack 4-index ERI from flat row-major: index(p,q,r,s) = ((p*norb + q)*norb + r)*norb + s
    auto eri = [&](int p, int q, int r, int s) -> double {
        return eri_chem_flat(((p * norb + q) * norb + r) * norb + s);
    };

    // Unpack 4-index 2-RDM from flat: index(t,u,v,w) = ((t*n_act + u)*n_act + v)*n_act + w
    auto dm2 = [&](int t, int u, int v, int w) -> double {
        return dm2_flat(((t * n_act + u) * n_act + v) * n_act + w);
    };

    const int n_pairs = static_cast<int>(pairs.size());

    // Inactive Fock: FI = h1 + core mean field
    Eigen::MatrixXd FI = h1;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int p = 0; p < norb; ++p) {
        for (int q = 0; q < norb; ++q) {
            double acc = 0.0;
            for (int i = 0; i < n_core; ++i) {
                acc += 2.0 * eri(p, q, i, i) - eri(p, i, i, q);
            }
            FI(p, q) += acc;
        }
    }

    // Symmetric generalized Fock (1-RDM only): Favg = h1 + core + active(1-RDM)
    Eigen::MatrixXd Favg = h1;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int p = 0; p < norb; ++p) {
        for (int q = 0; q < norb; ++q) {
            double acc = 0.0;
            for (int i = 0; i < n_core; ++i) {
                acc += 2.0 * eri(p, q, i, i) - eri(p, i, i, q);
            }
            // Active contribution to Favg
            for (int t = 0; t < n_act; ++t) {
                int pt = n_core + t;
                for (int u = 0; u < n_act; ++u) {
                    int pu = n_core + u;
                    double d = dm1(t, u);
                    if (std::abs(d) > 1e-14) {
                        acc += d * (eri(p, q, pt, pu) - 0.5 * eri(p, pt, pu, q));
                    }
                }
            }
            Favg(p, q) += acc;
        }
    }

    // Full generalized Fock F (non-symmetric for active rows)
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(norb, norb);

    // Core rows: F[i,:] = 2 * Favg[i,:]
    for (int i = 0; i < n_core; ++i) {
        F.row(i) = 2.0 * Favg.row(i);
    }

    // Active rows: F[t,:] = dm1[t,u] FI[u,:] + dm2[t,u,v,w] * eri[:,u,v,w]
    // Parallelize over active orbital index t.
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int t = 0; t < n_act; ++t) {
        int pt = n_core + t;
        for (int q = 0; q < norb; ++q) {
            double val = 0.0;
            // Term 1: dm1[t,u] * FI[u,q]
            for (int u = 0; u < n_act; ++u) {
                int pu = n_core + u;
                double d = dm1(t, u);
                if (std::abs(d) > 1e-14) {
                    val += d * FI(pu, q);
                }
            }
            // Term 2: dm2[t,u,v,w] * eri(q,u,v,w)
            for (int u = 0; u < n_act; ++u) {
                int pu = n_core + u;
                for (int v = 0; v < n_act; ++v) {
                    int pv = n_core + v;
                    for (int w = 0; w < n_act; ++w) {
                        int pw = n_core + w;
                        double d2 = dm2(t, u, v, w);
                        if (std::abs(d2) > 1e-14) {
                            val += d2 * eri(q, pu, pv, pw);
                        }
                    }
                }
            }
            F(pt, q) = val;
        }
    }

    // Orbital gradient: g[pq] = 2*(F_qp - F_pq)
    Eigen::VectorXd gradient(n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
        int p = pairs[i].first;
        int q = pairs[i].second;
        gradient(i) = 2.0 * (F(q, p) - F(p, q));
    }

    // Diagonal Super-CI Hessian: H_diag[pq] = Favg_pp - Favg_qq
    Eigen::VectorXd H_diag(n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
        int p = pairs[i].first;
        int q = pairs[i].second;
        H_diag(i) = Favg(p, p) - Favg(q, q);
    }

    return {gradient, F, Favg, H_diag};
}

}  // namespace vibeqc
