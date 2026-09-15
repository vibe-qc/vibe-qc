// GFN2-xTB anisotropic electrostatics (AES) over directed image records.
//
// Bannwarth, Ehlert & Grimme, J. Chem. Theory Comput. 15, 1652 (2019),
// doi:10.1021/acs.jctc.8b01176:
//   * Eq. 25  E_AES = E_{q mu} + E_{q Theta} + E_{mu mu} (damped, Eq. 28)
//   * Eq. 26  traceless cumulative atomic quadrupole Theta = 3/2 theta
//             - 1/2 tr(theta) 1
//   * Eq. 27  cumulative atomic multipole moments (CAMM) from the density
//             and the global-origin dipole/quadrupole integrals
//   * Eq. 28  f_n(R) = 1 / (R^n (1 + 6 (R0_AB/R)^{a_n})), a_3 = 3, a_5 = 4
//   * Eq. 29  CN-dependent multipole radius R0_A'
//   * Eq. 31  on-site anisotropic XC energy f^mu |mu|^2 + f^Theta ||Theta||^2
//   * Eqs. 39-44  the AES/AXC Fock matrix through overlap, dipole and
//             quadrupole integrals with the origin-shift terms.
//
// Every routine here consumes a directed image-record source (see
// vibeqc/semiempirical/core/periodic_gamma.hpp): for a molecule the records
// are the ordered atom pairs with zero shift and unit weight; for the
// Gamma-periodic driver they are every atom pair and lattice image inside the
// pair cutoff; for the SECCM adapters they are the frozen Wigner-Seitz
// inventory with its fractional ownership weights.  The pair vector of a
// record is v = R_a - (R_b + shift), the molecular driver's R_A - R_B
// convention, and the energy is 1/2 sum_records w e(a, b, v) so that the
// reverse-symmetric record sets count each physical pair once.
//
// The Fock matrix is derived here as dE/dP rather than ported from xtb's
// setvsdq: with the atom-centred potentials v_A = dE/dq_A, w_A = dE/dmu_A and
// t~_A = dE/dtheta_A (theta the raw, non-traceless Eq. 27c tensor), the
// contribution of the AO block (kappa on A in the home cell, lambda on B in
// cell g) is
//     F_{kappa lambda}(g) = 1/2 [ F^A(O = R_A) + F^B(O = R_B + g) ],
//     F^X(O) = -v_X S - w_X . (D - O S)
//              - sum_ij t~_X,ij (Q_ij - O_i D_j - O_j D_i + O_i O_j S),
// with S, D, Q the global-origin overlap, dipole and raw quadrupole integrals
// of the block.  The origin of the ket side is the *image* position R_B + g:
// that is what makes the lattice-summed Fock exact for image blocks and is
// where a home-cell port of setvsdq goes wrong.

#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_multipole.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

// Global AES constants of the GFN2-xTB parameter file ($globpar).
constexpr double kGFN2AesShift = 1.20;   // Delta_val of Eq. 29
constexpr double kGFN2AesExp = 4.0;      // the exponent of Eq. 29
constexpr double kGFN2AesRmax = 5.0;     // R_max of Eq. 29 (bohr)
constexpr double kGFN2AesDamp3 = 3.0;    // a_3 of Eq. 28
constexpr double kGFN2AesDamp5 = 4.0;    // a_5 of Eq. 28

// Off-diagonal components of a symmetric 3x3 tensor appear twice in the full
// double sum; this is their weight in the 6-component (xx, xy, xz, yy, yz, zz)
// storage used throughout.
constexpr std::array<double, 6> kAesQuadrupoleWeights = {1.0, 2.0, 2.0, 1.0, 2.0, 1.0};

// Per-atom AES parameters at a given coordination number.
struct GFN2AesAtomParameters {
    Eigen::VectorXd dpol;       // f^mu_XC   (DPOL x 0.01)
    Eigen::VectorXd qpol;       // f^Theta_XC (QPOL x 0.01)
    Eigen::VectorXd mrad;       // R0_A' of Eq. 29 (bohr)
    Eigen::VectorXd dmrad_dcn;  // dR0_A'/dCN_A
};

GFN2AesAtomParameters gfn2_aes_atom_parameters(
    const std::vector<Atom>& atoms,
    const GFN2ParameterSet& params,
    const Eigen::VectorXd& coordination_numbers);

// Cumulative atomic multipole moments (Eq. 27, xtb mmompop trace removal).
struct GFN2AesMoments {
    Eigen::VectorXd q;      // partial charges (n0 - population; cation > 0)
    Eigen::MatrixXd mu;     // n_atoms x 3
    Eigen::MatrixXd theta;  // n_atoms x 6 traceless (xx, xy, xz, yy, yz, zz)
};

// Moments from the Gamma density and image-summed global-origin integrals.
// Bra-row (Mulliken) partitioning: atom A owns every AO pair whose bra sits
// on A, with the origin at R_A; summing the ket over lattice images keeps
// the ket's own origin shift inside the integral sums (see
// GFN2MultipoleLatticeSums).
GFN2AesMoments gfn2_aes_moments(
    const Eigen::MatrixXd& density,
    const GFN2MultipoleLatticeSums& sums,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const Eigen::VectorXd& q_atom);

double gfn2_aes_energy(
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesMoments& moments,
    const GFN2AesAtomParameters& params,
    const ImageRecordSource& records);

// Atom-centred AES potentials: v = dE/dq, w = dE/dmu, and t_raw = dE/dtheta
// conjugate to the RAW (non-traceless) Eq. 27c tensor, stored as full
// symmetric 3x3 tensors in row-major 9-vectors.
struct GFN2AesPotentials {
    Eigen::VectorXd v;       // n_atoms
    Eigen::MatrixXd w;       // n_atoms x 3
    Eigen::MatrixXd t_raw;   // n_atoms x 9
};

GFN2AesPotentials gfn2_aes_potentials(
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesMoments& moments,
    const GFN2AesAtomParameters& params,
    const ImageRecordSource& records);

// Coefficients of one Fock side F^X(O) = cS S + cD . D + sum_k cQ_k Q_k over
// global-origin integrals with the ket-origin O (the 6 quadrupole weights of
// kAesQuadrupoleWeights are folded into cQ).
struct GFN2AesFockSide {
    double cS = 0.0;
    Eigen::Vector3d cD = Eigen::Vector3d::Zero();
    std::array<double, 6> cQ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

GFN2AesFockSide gfn2_aes_fock_side(
    const GFN2AesPotentials& potentials, int atom,
    const Eigen::Vector3d& origin);

// Add the AES Fock matrix to H using image-summed integrals: the bra side at
// R_A and the ket side at every image origin R_B + g, expanded through the
// first and second lattice moments of the overlap and dipole blocks.
void gfn2_aes_add_fock(
    Eigen::MatrixXd& hamiltonian,
    const GFN2MultipoleLatticeSums& sums,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesPotentials& potentials);

// Explicit geometry derivatives of E_AES at fixed charges and moments: the
// pair-kernel force and strain (Eq. 25 + Eq. 28 damping), and dE/dR0_A' for
// the coordination-number chain of Eq. 29.
struct GFN2AesKernelDerivatives {
    Eigen::MatrixXd gradient;   // n_atoms x 3
    Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
    Eigen::VectorXd dE_dmrad;   // n_atoms
};

GFN2AesKernelDerivatives gfn2_aes_kernel_derivatives(
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesMoments& moments,
    const GFN2AesAtomParameters& params,
    const ImageRecordSource& records);

// Derivative of sum_g sum_{kappa lambda} P F_{kappa lambda}(g) through the
// explicit origin vectors R_A and R_B + g at fixed integrals (the companion
// of the integral-derivative contraction, which holds the origins fixed).
// Returns the atomic gradient and the image virial sum_g f_B(g) (x) g.
struct GFN2AesOriginDerivatives {
    Eigen::MatrixXd gradient;   // n_atoms x 3
    Eigen::Matrix3d image_virial = Eigen::Matrix3d::Zero();
};

GFN2AesOriginDerivatives gfn2_aes_origin_derivatives(
    const Eigen::MatrixXd& density,
    const GFN2MultipoleLatticeSums& sums,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesPotentials& potentials);

// Weight filler for contract_gfn2_multipole_lattice_derivatives: the
// density-weighted Fock coefficients of every AO pair in one cell.
void gfn2_aes_fill_derivative_weights(
    int cell_index,
    const Eigen::Vector3d& shift,
    const Eigen::MatrixXd& density,
    const std::vector<int>& ao_atom,
    const std::vector<Eigen::Vector3d>& positions,
    const GFN2AesPotentials& potentials,
    Eigen::MatrixXd& weight_S,
    std::array<Eigen::MatrixXd, 3>& weight_D,
    std::array<Eigen::MatrixXd, 6>& weight_Q);

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
