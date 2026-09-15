// One- and two-electron integrals in the AO basis.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "basis.hpp"
#include "molecule.hpp"

namespace vibeqc {

// S_{mu nu} = <mu|nu>
Eigen::MatrixXd compute_overlap(const BasisSet& basis);

// Cross-basis AO overlap S_{mu nu} = <mu^(1)|nu^(2)> between two basis sets
// (which may differ in contraction and/or sit at different geometries).
// Shape (basis1.nbasis, basis2.nbasis); not symmetric. Used by the READ
// initial guess to project a prior density onto the current basis.
Eigen::MatrixXd compute_overlap_two_basis(const BasisSet& basis1,
                                          const BasisSet& basis2);

// T_{mu nu} = <mu|-1/2 nabla^2|nu>
Eigen::MatrixXd compute_kinetic(const BasisSet& basis);

// V_{mu nu} = sum_A <mu|-Z_A/|r-R_A||nu>
Eigen::MatrixXd compute_nuclear(const BasisSet& basis, const Molecule& mol);

// Dipole-integral matrices about a user-specified origin (bohr):
//   M_x,{mu nu} = <mu|x - O_x|nu>   (and analogues in y, z)
// Used to compute dipole moments <μ> = -tr(P · M_c) + Σ_A Z_A (R_A - O)_c.
// The default origin (0, 0, 0) is fine for anything neutral; charged
// systems should supply the center of mass / center of charge / nuclear
// centroid explicitly.
struct DipoleIntegrals {
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
    Eigen::MatrixXd z;
};
DipoleIntegrals compute_dipole(const BasisSet& basis,
                               const std::array<double, 3>& origin = {0.0, 0.0, 0.0});

// Four-center electron repulsion integrals in chemists' notation:
//   (mu nu | lambda sigma) = integral_r1,r2 phi_mu(r1) phi_nu(r1)
//                            * 1/r12 * phi_lambda(r2) phi_sigma(r2)
// Stored dense (n,n,n,n), row-major, with all 8 permutation-equivalent
// positions filled. Memory is 8*n^4 bytes — fine for M1 (H2O/STO-3G = 19kB).
struct Eri4D {
    std::vector<double> data;
    std::size_t n = 0;                // extent = nbasis

    double& operator()(std::size_t i, std::size_t j,
                       std::size_t k, std::size_t l) noexcept {
        return data[((i * n + j) * n + k) * n + l];
    }
    double operator()(std::size_t i, std::size_t j,
                      std::size_t k, std::size_t l) const noexcept {
        return data[((i * n + j) * n + k) * n + l];
    }
};

Eri4D compute_eri(const BasisSet& basis);

// Exact four-index AO -> MO transform of the chemists'-notation ERI tensor
// into pair-block matrix form:
//   W(x * n_bra + y, z * n_ket + w) = (x y | z w)
// with the bra pair (x, y) over C_bra's columns and the ket pair (z, w)
// over C_ket's columns.  Four successive quarter transforms, O(N^5) work;
// the returned matrix is (n_bra^2, n_ket^2) and the intermediate peaks at
// (nao^2, n_ket^2).  This is the canonical (non-DF) counterpart of
// DensityFitting::mo_transform for the conventional coupled-cluster path.
Eigen::MatrixXd eri_mo_pair_transform(const Eri4D& eri,
                                      const Eigen::MatrixXd& C_bra,
                                      const Eigen::MatrixXd& C_ket);

}  // namespace vibeqc
