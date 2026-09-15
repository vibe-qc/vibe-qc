// Translation-covariant, lattice-summed second-order charge kernels for the
// semiempirical SCC families (SCC-DFTB, GFN2-xTB) on periodic and cyclic
// boundaries.
//
// Every pair kernel gamma_ab(R) used by the SCC methods here is a damped
// Coulomb interaction that tends to 1/R at long range.  On a periodic
// boundary the 1/R tail must go through Ewald summation; only the remainder
//
//     gamma_ab(R) - 1/R
//
// may be summed directly over lattice images.  This header supplies both
// halves and the assembly that joins them:
//
//   * ShellGammaForm::KlopmanOhno -- the Klopman-Ohno / Mataga-Nishimoto
//     kernel of GFN2-xTB (Bannwarth, Ehlert & Grimme, J. Chem. Theory
//     Comput. 15, 1652 (2019), doi:10.1021/acs.jctc.8b01176, Eqs. 21-22)
//     and of the in-house SCC-DFTB set.  Its remainder decays only as
//     -eta^2/(2 R^3), so a three-dimensional image sum of the remainder
//     converges conditionally in the pair cutoff (issues #425, #444).  The
//     lattice-summed kernel is still translation covariant and continuous
//     in every atomic coordinate; it is cutoff-dependent by construction,
//     exactly as the tblite/xtb periodic GFN1 kernel is.
//   * ShellGammaForm::Elstner -- Elstner et al., Phys. Rev. B 58, 7260
//     (1998), doi:10.1103/PhysRevB.58.7260, Eqs. 17-18 and Appendix:
//     gamma = 1/R - S(tau_a, tau_b, R) with tau = 16/5 U.  S decays
//     exponentially, so the remainder sum converges absolutely in every
//     dimension and the kernel has a thermodynamic limit (the maintainer's
//     D1 decision of 2026-08-28, agentic-loop/asks/DECISION-maintainer-
//     2026-08-28-seccm-gamma-and-gates.md).
//
// The Ewald part is EwaldCoulombKernel: 3-D Ewald, 2-D Parry/Heyes and
// 1-D background-corrected wire sums, each lattice-periodic in the pair
// displacement so that relabelling an atom by a lattice vector is exactly
// a no-op.  The record-driven assembly build_periodic_shell_gamma serves
// the Gamma-periodic drivers (records = every atom pair and lattice image
// inside the pair cutoff, unit weight) and the SECCM adapters (records =
// the frozen Wigner-Seitz image inventory with its fractional ownership
// weights) with one arithmetic.

#pragma once

#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Scalar pair kernels
// ---------------------------------------------------------------------------

enum class ShellGammaForm {
    // gamma(R) = 1 / sqrt(R^2 + eta_ab^2); see KlopmanOhnoAverage for eta_ab.
    KlopmanOhno,
    // gamma(R) = 1/R - S(tau_a, tau_b, R), tau = 16/5 U (Elstner 1998).
    Elstner,
};

// How the Klopman-Ohno damping length eta_ab is formed from two hardnesses.
enum class KlopmanOhnoAverage {
    // GFN2-xTB (tblite effective_coulomb, gexp = 2, arithmetic average):
    //   eta_ab = 1 / (0.5 (U_a + U_b)).
    HardnessMean,
    // In-house SCC-DFTB (SemiempiricalHamiltonianBuilder::gamma_matrix):
    //   eta_ab = 0.5 (1/U_a + 1/U_b).
    InverseHardnessMean,
};

struct ShellGammaSpec {
    ShellGammaForm form = ShellGammaForm::KlopmanOhno;
    KlopmanOhnoAverage ko_average = KlopmanOhnoAverage::HardnessMean;
};

// Elstner Eq. 17/18 short-range function S(tau_a, tau_b, R) for R > 0 and
// its R-derivative.  Equal-tau and unequal-tau closed forms follow the
// paper's Appendix (the DFTB+ shortgammafuncs.F90 sign convention); the
// near-degenerate window |tau_a - tau_b| <= 0.02 max(tau) uses the exact
// series in delta = (tau_a - tau_b)/2 through delta^4, which is where the
// unequal closed form loses precision to cancellation.
double elstner_short_range(double tau_a, double tau_b, double R);
double elstner_short_range_derivative(double tau_a, double tau_b, double R);

// Same-site (R = 0) value of the pair kernel: the published on-site
// hardness parameterisation of each method (GFN2 Eq. 22 at R = 0: the
// arithmetic shell mean; SCC-DFTB: U for one shell per atom).  The
// Elstner form keeps this on-site convention unchanged because the on-site
// block is a method parameter, not a lattice-sum question.
double shell_gamma_onsite(const ShellGammaSpec& spec, double U_a, double U_b);

// Pair kernel gamma_ab(R) for R > 0 and its R-derivative.
double shell_gamma_pair(const ShellGammaSpec& spec, double U_a, double U_b,
                        double R);
double shell_gamma_pair_derivative(const ShellGammaSpec& spec, double U_a,
                                   double U_b, double R);

// Short-range remainder gamma_ab(R) - 1/R for R > 0 and its R-derivative:
// the part that is summed directly over lattice images.
double shell_gamma_remainder(const ShellGammaSpec& spec, double U_a,
                             double U_b, double R);
double shell_gamma_remainder_derivative(const ShellGammaSpec& spec,
                                        double U_a, double U_b, double R);

// Distance beyond which |gamma_ab(R) - 1/R| stays below `tolerance`.  For
// the Elstner form this is the exponential tail of S (a few tens of bohr
// for GFN2's softest shells, e.g. Cu 4s with U = 0.20: 40 bohr at 1e-10);
// the Klopman-Ohno remainder has no absolutely convergent tail and returns
// +infinity, so its callers fall back to the pair cutoff.
double shell_gamma_remainder_range(const ShellGammaSpec& spec, double U_a,
                                   double U_b, double tolerance);

// ---------------------------------------------------------------------------
// Ewald-summed 1/R lattice potential in one, two, or three dimensions
// ---------------------------------------------------------------------------

class EwaldCoulombKernel {
public:
    // `translations` holds the one to three periodic lattice vectors (bohr).
    // `alpha <= 0` selects the validated width convention of each dimension
    // (3-D: sqrt(pi)/V^(1/3); 2-D: 0.85 sqrt(pi/A); 1-D: 4/L); the summed
    // value is alpha-independent within the e^-30 series truncation.
    explicit EwaldCoulombKernel(
        const std::vector<Eigen::Vector3d>& translations,
        double alpha = 0.0);

    int dimension() const { return dim_; }
    double alpha() const { return alpha_; }
    const std::vector<Eigen::Vector3d>& translations() const {
        return translations_;
    }

    // Phi(d) = sum_g 1/|d + g| over the lattice, Ewald-regularised with the
    // neutral-cell background convention of each dimension.  With
    // `exclude_self` the g = 0 term is dropped (the same-site block, where
    // it would be singular) and the -2 alpha/sqrt(pi) self term is added.
    double potential(const Eigen::Vector3d& d, bool exclude_self) const;

    // dPhi/dd.  Zero at d = 0 with exclude_self (the lattice self potential
    // is stationary under a rigid displacement of the site).
    Eigen::Vector3d gradient(const Eigen::Vector3d& d, bool exclude_self) const;

private:
    int dim_ = 0;
    double alpha_ = 0.0;
    std::vector<Eigen::Vector3d> translations_;
    // Cartesian lattice/dual data for the fractional reduction of d.
    Eigen::MatrixXd lattice_;        // dim x 3 (rows = translations)
    Eigen::MatrixXd gram_inverse_;   // dim x dim
    double r_cut_ = 0.0;
    std::vector<int> n_real_;        // per-axis real-space box half-widths
    // 3-D reciprocal terms and constants.
    struct ReciprocalTerm {
        Eigen::Vector3d g;
        double weight;
    };
    std::vector<ReciprocalTerm> reciprocal_;
    double volume_ = 0.0;
    double background_ = 0.0;   // -pi/(V alpha^2) in 3-D (neutralising background), else 0
    // 2-D slab data.
    Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
    double area_ = 0.0;
    std::vector<Eigen::Vector3d> kvecs_;
    // 1-D wire data.
    Eigen::Vector3d axis_ = Eigen::Vector3d::Zero();
    double length_ = 0.0;
    int m_max_ = 0;

    Eigen::Vector3d reduce(const Eigen::Vector3d& d) const;
    void real_space(const Eigen::Vector3d& d_reduced, bool exclude_self,
                    double* value, Eigen::Vector3d* grad) const;
    void reciprocal_3d(const Eigen::Vector3d& d, double* value,
                       Eigen::Vector3d* grad) const;
    void reciprocal_2d(const Eigen::Vector3d& d, double* value,
                       Eigen::Vector3d* grad) const;
    void reciprocal_1d(const Eigen::Vector3d& d, double* value,
                       Eigen::Vector3d* grad) const;
};

// ---------------------------------------------------------------------------
// Record-driven assembly
// ---------------------------------------------------------------------------

// A directed image record: site b, translated by `shift`, as seen from site
// a with ownership `weight`.  The physical pair displacement is
// R_b + shift - R_a.  Record sets must be reverse-symmetric (every (a, b,
// shift) has its (b, a, -shift) partner with the same weight) for the
// assembled matrix to be symmetric; both the Gamma-periodic cell lists and
// the validated SECCM WS inventories satisfy this.
struct ImageRecord {
    int a = 0;
    int b = 0;
    Eigen::Vector3d shift = Eigen::Vector3d::Zero();
    double weight = 1.0;
};

using ImageRecordSink = std::function<void(const ImageRecord&)>;
using ImageRecordSource = std::function<void(const ImageRecordSink&)>;

// One charge site of the kernel: its parent atom and hardness (Ha).  For
// shell-resolved methods a site is a shell; for atomic SCC-DFTB one site per
// atom.
struct GammaSite {
    int atom = 0;
    double hardness = 0.5;
};

// Gamma_ij = Phi(R_b - R_a; exclude_self if a == b)
//         + sum_{records (a, b)} w [gamma_ij(|R_b + shift - R_a|) - 1/|...|]
//         + delta_{a == b} onsite(U_i, U_j)
// for sites i on atom a and j on atom b.
Eigen::MatrixXd build_periodic_shell_gamma(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec,
    const EwaldCoulombKernel& ewald,
    const ImageRecordSource& records);

// Molecular (free-boundary) counterpart with the same site/spec contract:
// Gamma_ij = gamma_ij(|R_b - R_a|) off-site, onsite(U_i, U_j) on-site.
Eigen::MatrixXd build_molecular_shell_gamma(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec);

// d/dR_atom of E = 1/2 dq^T Gamma dq at fixed dq (n_atoms x 3).
Eigen::MatrixXd periodic_shell_gamma_gradient(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec,
    const EwaldCoulombKernel& ewald,
    const ImageRecordSource& records,
    const Eigen::VectorXd& dq);

// dE/d eps_ij of E = 1/2 dq^T Gamma dq under a homogeneous strain of the
// lattice, the atomic positions and the record shifts together, at fixed dq
// and fixed record set.  The remainder part is analytic (pair virial); the
// Ewald part is a central difference of the closed-form lattice potential
// with alpha held fixed (h = 1e-5), so this is a derivative of an explicit
// function, not of the SCC.
Eigen::Matrix3d periodic_shell_gamma_strain_derivative(
    const std::vector<GammaSite>& sites,
    const std::vector<Eigen::Vector3d>& atom_positions,
    const ShellGammaSpec& spec,
    const EwaldCoulombKernel& ewald,
    const ImageRecordSource& records,
    const Eigen::VectorXd& dq);

}  // namespace semiempirical
}  // namespace vibeqc
