// Hamiltonian builders for semiempirical methods.
//
// Each method provides a HamiltonianBuilder that constructs H⁰
// from the overlap matrix and parameter set.  The builder is
// registered with the method plugin in the registry.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/basis.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/core/method_registry.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// DFTB Hamiltonian builder (Wolfsberg-Helmholtz)
// ---------------------------------------------------------------------------

Eigen::MatrixXd build_dftb_hamiltonian_zero(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const SemiempiricalParameters& params);

// ---------------------------------------------------------------------------
// GFN2-xTB Hamiltonian builder (electronegativity-based)
// ---------------------------------------------------------------------------

Eigen::MatrixXd build_gfn2_hamiltonian_zero(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params);

// GFN2 double-damped coordination number (ka=10, kb=20, r_shift=2 with the
// 4/3-scaled D3 radii), the same CN array that feeds the H0 self-energy and
// xtb's CN-dependent multipole radii (get_radcn).  Distinct from the D4
// dispersion CN in vibeqc/dispersion.hpp.
Eigen::VectorXd gfn2_h0_coordination_numbers(const Molecule& mol);

// Periodic Eq. 18 coordination numbers. The sum is over every atom and
// lattice image inside the existing 15-bohr physical pair-distance cutoff;
// only the home image of the atom itself is excluded.
Eigen::VectorXd gfn2_h0_periodic_coordination_numbers(
    const PeriodicSystem& system);

// One directed Eq. 18 contribution, including the shared 15-bohr cutoff.
// Boundary-specific assemblers (for example, the WS-weighted SECCM adapter)
// use this primitive without making the molecular H0 builder depend on their
// topology types.
double gfn2_h0_coordination_pair_contribution(
    int Z_a, int Z_b, double distance_bohr);

// GFN2 H0 block between home-cell AOs and AOs translated by image_shift.
// image_shift = 0 gives the molecular/home-cell H0 block; nonzero shifts are
// used by periodic Gamma GFN2 so image-cell couplings get the same EN /
// distance-polynomial shape terms as ordinary two-centre H0 couplings. A
// future k-point implementation must Bloch-sum these full image blocks.
Eigen::MatrixXd build_gfn2_hamiltonian_zero_image(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const Eigen::Vector3d& image_shift);

// As above, but consume an already assembled Eq. 18 coordination vector.
// Periodic and cyclic boundaries must compute that vector from their physical
// image inventory once and reuse it for every H0 block. The molecular entry
// point retains its historical behavior through the five-argument overload.
Eigen::MatrixXd build_gfn2_hamiltonian_zero_image_with_cn(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const Eigen::Vector3d& image_shift,
    const Eigen::VectorXd& coordination_numbers);

// ---------------------------------------------------------------------------
// Gamma matrix builders
// ---------------------------------------------------------------------------

// DFTB: Ohno-Klopman damped Coulomb
Eigen::MatrixXd build_dftb_gamma(
    const Molecule& mol,
    const SemiempiricalParameters& params);

// GFN2-xTB: Klopman-Ohno with Gaussian damping
Eigen::MatrixXd build_gfn2_gamma(
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params);

// ---------------------------------------------------------------------------
// Shell-resolved gamma (foundation for GFN2-xTB H¹ multipole)
// ---------------------------------------------------------------------------

// Descriptor for one shell in the GFN2-xTB basis
struct GFN2ShellInfo {
    int atom_idx = 0;      // parent atom index
    int Z = 0;             // atomic number
    int l = 0;             // angular momentum (0=s, 1=p, 2=d)
    int bf_start = 0;      // first basis function index
    int n_funcs = 0;       // number of basis functions
    double k_en = 1.0;     // electronegativity scaling
    double hardness = 0.5; // shell hardness (gam / k_en)
    double kcn = 0.0;      // CN self-energy shift (KCNS/KCNP/KCND, Ha)
    double poly = 0.0;     // shell distance-polynomial (POLYS/POLYP/POLYD ×0.01)
};

// Enumerate shells from a GFN2-xTB basis set + parameter set
std::vector<GFN2ShellInfo> gfn2_enumerate_shells(
    const BasisSet& basis,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params);

// Shell-pair Klopman-Ohno kernel at an explicit separation.
double gfn2_shell_gamma_at_distance(
    const GFN2ShellInfo& shell_a,
    const GFN2ShellInfo& shell_b,
    double distance_bohr);

// Shell-pair gamma: γ_{AB}^{ll'} using shell-specific hardness
Eigen::MatrixXd build_gfn2_shell_gamma(
    const std::vector<GFN2ShellInfo>& shells,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params);

// ---------------------------------------------------------------------------
// GFN2-xTB H⁰ geometry derivatives (Bannwarth, Ehlert & Grimme, JCTC 2019,
// doi:10.1021/acs.jctc.8b01176).
//
// H⁰_{μν} = ½ K_{ll'} S_{μν} (h_μ + h_ν) · X_ζ · (1 + 0.02 ΔEN²) · Π(R_AB)
// factorizes as H⁰_{μν} = F_{μν} · S_{μν} with F depending on geometry only
// through the CN-dependent self-energies h_μ = EN_l(A) − k_CN,l(A)·CN'_A and
// the distance polynomial Π = (1 + p_l(A)√(R/R₀))(1 + p_l'(B)√(R/R₀)).
// Hence  d(Σ D·H⁰)/dR = Σ D F dS/dR                     (fold F into the
//                                                        M·dS/dR contraction)
//                     + Σ_A (∂E/∂CN'_A)·dCN'_A/dR        (CN chain)
//                     + Σ_{AB} D·½K S (h+h) X_ζ EN·dΠ/dR (polynomial force).
// ---------------------------------------------------------------------------

// Per-image-block worker: returns the F block (∂H⁰_{μν}/∂S_{μν} shape factor)
// for AOs ⟨μ| in the home cell and |ν⟩ shifted by image_shift, and ACCUMULATES
// the fixed-S geometric derivative contractions with the density block D:
//   dE_dCN(A) += Σ_{μν} D_{μν} ∂H⁰_{μν}/∂CN'_A   (chain through
//                gfn2_cn_chain_gradient below)
//   grad_poly  += Σ_{μν} D_{μν} ∂H⁰_{μν}/∂R|_{S,CN fixed}  (distance
//                polynomial pair force, n_atoms × 3)
// image_shift = 0 gives the molecular / home-cell terms.
Eigen::MatrixXd gfn2_h0_image_derivative_terms(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& D,
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const Eigen::Vector3d& image_shift,
    const Eigen::VectorXd& coordination_numbers,
    Eigen::VectorXd& dE_dCN,
    Eigen::MatrixXd& grad_poly,
    Eigen::Matrix3d& strain_poly);

// Chain rule through the GFN2 H⁰ coordination number (the two-factor CN'
// with the 15-bohr hard cutoff used by build_gfn2_hamiltonian_zero_image):
// returns Σ_A dE_dCN(A)·dCN'_A/dR as an (n_atoms × 3) gradient.
Eigen::MatrixXd gfn2_cn_chain_gradient(
    const Molecule& mol,
    const Eigen::VectorXd& dE_dCN);

struct GFN2PeriodicCNDerivatives {
    Eigen::MatrixXd atomic_gradient;
    Eigen::Matrix3d strain_derivative = Eigen::Matrix3d::Zero();
};

// Periodic Eq. 18 chain rule over every directed atom/image pair used by
// gfn2_h0_periodic_coordination_numbers.  The strain tensor includes the
// image-vector derivative that cannot be reconstructed from an atomic virial.
GFN2PeriodicCNDerivatives gfn2_periodic_cn_chain_derivatives(
    const PeriodicSystem& system,
    const Eigen::VectorXd& dE_dCN);

// Wolfsberg-Helmholtz K_{ll'} from GFN2-xTB (Bannwarth et al. 2019,
// Table 2). The paper tabulates exact pair values: kss = 1.85,
// kpp = kdd = 2.23, ksp = 2.04, ksd = kpd = 2.00. The historical
// per-shell arithmetic average (kshell = [1.85, 2.23, 2.23, 2.23])
// reproduces ss/sp/pp by construction but gives the wrong 2.04/2.23
// for sd/pd pairs, over-coupling d-polarization functions into the
// band energy. Pairs involving f (and higher) polarization shells have
// no published table entry and keep the historical average.
inline double gfn2_kshell(int l1, int l2, double = 0.0, double = 0.0) {
    static const double kpair[3][3] = {
        {1.85, 2.04, 2.00},
        {2.04, 2.23, 2.00},
        {2.00, 2.00, 2.23},
    };
    if (l1 >= 0 && l1 <= 2 && l2 >= 0 && l2 <= 2) {
        return kpair[l1][l2];
    }
    static const double kshell[4] = {1.85, 2.23, 2.23, 2.23};
    return 0.5 * (kshell[std::min(l1, 3)] + kshell[std::min(l2, 3)]);
}

}  // namespace semiempirical
}  // namespace vibeqc
