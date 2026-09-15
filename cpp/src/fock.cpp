#include "vibeqc/fock.hpp"

namespace vibeqc {

Eigen::MatrixXd build_fock_g(const Eri4D& eri, const Eigen::MatrixXd& D) {
    const int n = static_cast<int>(eri.n);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);

    // G_{mu nu} = sum_{lambda sigma} D_{lambda sigma}
    //             [ (mu nu | lambda sigma) - (1/2) (mu lambda | nu sigma) ]
    //
    // Parallelise over mu — each thread writes a distinct row of G, so no
    // races. The inner (nu, lam, sig) contraction is fully local.
    #pragma omp parallel for schedule(static)
    for (int mu = 0; mu < n; ++mu) {
        for (int nu = 0; nu < n; ++nu) {
            double g = 0.0;
            for (int lam = 0; lam < n; ++lam) {
                for (int sig = 0; sig < n; ++sig) {
                    g += D(lam, sig)
                         * (eri(mu, nu, lam, sig)
                            - 0.5 * eri(mu, lam, nu, sig));
                }
            }
            G(mu, nu) = g;
        }
    }
    return G;
}

Eigen::MatrixXd build_coulomb(const Eri4D& eri, const Eigen::MatrixXd& D) {
    const int n = static_cast<int>(eri.n);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    #pragma omp parallel for schedule(static)
    for (int mu = 0; mu < n; ++mu) {
        for (int nu = 0; nu < n; ++nu) {
            double j = 0.0;
            for (int lam = 0; lam < n; ++lam) {
                for (int sig = 0; sig < n; ++sig) {
                    j += D(lam, sig) * eri(mu, nu, lam, sig);
                }
            }
            J(mu, nu) = j;
        }
    }
    return J;
}

Eigen::MatrixXd build_exchange(const Eri4D& eri, const Eigen::MatrixXd& D) {
    const int n = static_cast<int>(eri.n);
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);
    #pragma omp parallel for schedule(static)
    for (int mu = 0; mu < n; ++mu) {
        for (int nu = 0; nu < n; ++nu) {
            double k = 0.0;
            for (int lam = 0; lam < n; ++lam) {
                for (int sig = 0; sig < n; ++sig) {
                    k += D(lam, sig) * eri(mu, lam, nu, sig);
                }
            }
            K(mu, nu) = k;
        }
    }
    return K;
}

}  // namespace vibeqc
