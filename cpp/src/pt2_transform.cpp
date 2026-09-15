#include "vibeqc/pt2_transform.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>

namespace vibeqc {

Eigen::VectorXd transform_4index_mo(
    const Eigen::VectorXd& g_phys_flat,
    const Eigen::MatrixXd& U,
    int norb)
{
    // g_phys is physicist's (pr|qs) flat row-major: index(p,r,q,s) = ((p*norb + r)*norb + q)*norb + s
    // Transform: g_new[p,r,q,s] = sum_{a,b,c,d} U[a,p] U[b,r] U[c,q] U[d,s] g[a,b,c,d]
    const int norb4 = norb * norb * norb * norb;

    auto idx = [norb](int p, int r, int q, int s) -> int {
        return ((p * norb + r) * norb + q) * norb + s;
    };

    // Step 1: g1[p,r,q,s] = sum_a U[a,p] * g[a,r,q,s]
    Eigen::VectorXd g1(norb4);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int p = 0; p < norb; ++p) {
        for (int r = 0; r < norb; ++r) {
            for (int q = 0; q < norb; ++q) {
                for (int s = 0; s < norb; ++s) {
                    double acc = 0.0;
                    for (int a = 0; a < norb; ++a) {
                        acc += U(a, p) * g_phys_flat(idx(a, r, q, s));
                    }
                    g1(idx(p, r, q, s)) = acc;
                }
            }
        }
    }

    // Step 2: g2[p,r,q,s] = sum_b U[b,r] * g1[p,b,q,s]
    Eigen::VectorXd g2(norb4);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int p = 0; p < norb; ++p) {
        for (int r = 0; r < norb; ++r) {
            for (int q = 0; q < norb; ++q) {
                for (int s = 0; s < norb; ++s) {
                    double acc = 0.0;
                    for (int b = 0; b < norb; ++b) {
                        acc += U(b, r) * g1(idx(p, b, q, s));
                    }
                    g2(idx(p, r, q, s)) = acc;
                }
            }
        }
    }

    // Step 3: g3[p,r,q,s] = sum_c U[c,q] * g2[p,r,c,s]
    Eigen::VectorXd g3(norb4);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int p = 0; p < norb; ++p) {
        for (int r = 0; r < norb; ++r) {
            for (int q = 0; q < norb; ++q) {
                for (int s = 0; s < norb; ++s) {
                    double acc = 0.0;
                    for (int c = 0; c < norb; ++c) {
                        acc += U(c, q) * g2(idx(p, r, c, s));
                    }
                    g3(idx(p, r, q, s)) = acc;
                }
            }
        }
    }

    // Step 4: g4[p,r,q,s] = sum_d U[d,s] * g3[p,r,q,d]
    Eigen::VectorXd g4(norb4);
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int p = 0; p < norb; ++p) {
        for (int r = 0; r < norb; ++r) {
            for (int q = 0; q < norb; ++q) {
                for (int s = 0; s < norb; ++s) {
                    double acc = 0.0;
                    for (int d = 0; d < norb; ++d) {
                        acc += U(d, s) * g3(idx(p, r, q, d));
                    }
                    g4(idx(p, r, q, s)) = acc;
                }
            }
        }
    }

    return g4;
}

double pt2_energy_at_rotated_mo(
    const Eigen::MatrixXd& /* h1 */,
    const Eigen::VectorXd& /* g_phys_flat */,
    const Eigen::MatrixXd& /* kappa */,
    int /* n_core */,
    int /* n_act */,
    int /* n_act_elec */,
    int /* norb */,
    const std::string& /* variant */)
{
    // Stub: PT2 energy computation requires semicanonical prep + group
    // construction (Python-level).  The integral transform is done via
    // transform_4index_mo; the PT2 solve is called from Python.
    return 0.0;
}

}  // namespace vibeqc
