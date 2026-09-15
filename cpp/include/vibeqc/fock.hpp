// Fock-matrix building blocks in the AO basis.

#pragma once

#include <Eigen/Dense>

#include "integrals.hpp"

namespace vibeqc {

// RHF two-electron contribution:
// G_{mu nu} = sum_{lambda sigma} D_{lambda sigma}
//             [ (mu nu | lambda sigma) - (1/2) (mu lambda | nu sigma) ]
// D is the total (closed-shell) density, D = 2 C_occ C_occ^T,
// so trace(D·S) = n_electrons. F = Hcore + G is the standard RHF Fock.
Eigen::MatrixXd build_fock_g(const Eri4D& eri, const Eigen::MatrixXd& D);

// Coulomb matrix:  J_{mu nu} = sum_{lambda sigma} D_{lambda sigma}
//                              (mu nu | lambda sigma)
Eigen::MatrixXd build_coulomb(const Eri4D& eri, const Eigen::MatrixXd& D);

// Exchange matrix: K_{mu nu} = sum_{lambda sigma} D_{lambda sigma}
//                              (mu lambda | nu sigma)
Eigen::MatrixXd build_exchange(const Eri4D& eri, const Eigen::MatrixXd& D);

}  // namespace vibeqc
