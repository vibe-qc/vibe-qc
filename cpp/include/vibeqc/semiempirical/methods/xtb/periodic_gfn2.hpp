// Periodic GFN2-xTB at the Gamma point.
//
// Lattice-summed analogue of the molecular GFN2-xTB driver.  Every energy
// term is a translation-covariant lattice sum over the pair-distance
// interaction set of pair_lattice.hpp:
//   * H0 and S: per-image blocks with the Eq. 18 coordination numbers;
//   * isotropic second order: the Ewald-split shell gamma of
//     core/periodic_gamma.hpp (Klopman-Ohno or Elstner remainder over the
//     same pair records);
//   * anisotropic second order: the Bannwarth 2019 AES with cumulative
//     atomic multipole moments from image-summed integrals and the damped
//     pair kernel over the same records (methods/xtb/gfn2_aes.hpp);
//   * third order and repulsion as in the molecular driver.
// No coordinate canonicalisation is applied: relabelling an atom by a
// lattice vector only relabels image indices, and an atom crossing a cell
// face changes nothing (issue #296).

#pragma once

#include <Eigen/Dense>
#include <limits>
#include <vector>

#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_aes.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_multipole.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

// ---------------------------------------------------------------------------
// Periodic GFN2-xTB result
// ---------------------------------------------------------------------------

struct PeriodicGFN2Result : ParameterIdentifiedResult {
    double energy = std::numeric_limits<double>::quiet_NaN();
    double free_energy = std::numeric_limits<double>::quiet_NaN();
    double e_electronic = std::numeric_limits<double>::quiet_NaN();
    double e_repulsive = std::numeric_limits<double>::quiet_NaN();
    double e_scc = std::numeric_limits<double>::quiet_NaN();
    double e_band0 = std::numeric_limits<double>::quiet_NaN();
    double e_aes = std::numeric_limits<double>::quiet_NaN();
    double e_3rd = std::numeric_limits<double>::quiet_NaN();
    double fermi_level = std::numeric_limits<double>::quiet_NaN();
    double entropy = std::numeric_limits<double>::quiet_NaN();
    double smearing_temperature = 0.0;  // k_B T in Hartree
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    Eigen::MatrixXd overlap_gamma;
    Eigen::MatrixXd hamiltonian_gamma;
    Eigen::VectorXd charges;          // atomic Mulliken charges
    Eigen::VectorXd dq_shell;         // shell charges (for gradient)
    Eigen::MatrixXd atom_dipoles;     // CAMM dipoles, n_atoms x 3
    Eigen::MatrixXd atom_quadrupoles; // CAMM traceless quadrupoles, n_atoms x 6
    Eigen::VectorXd occupations;      // converged MO occupations n_i (2 at T=0)
    int n_basis = 0;
    int n_occ = 0;
    int n_shells = 0;                 // number of shells (for gradient)
    int n_cells = 0;
    int n_iter = 0;
    bool converged = false;
    // Per-iteration max |X_new - X| over the reduced state (charges and
    // moments), one entry per executed iteration, restarts concatenated.
    Eigen::VectorXd scc_max_change_trace;
    // Effective lattice-sum provenance of the SCF that produced this result
    // (issue #340); see vibeqc::semiempirical::PeriodicDFTB0Result.
    double cutoff_bohr = 0.0;
    // Second-order kernel form the SCF used; the derivatives reuse it.
    ShellGammaForm gamma_form = ShellGammaForm::Elstner;
};

// Default mixer of the periodic GFN2 driver when the caller leaves
// XTBSccOptions::scc_mixer at Simple: the tblite/xtb Eyert Broyden scheme on
// the joint charge-and-moment state (issue #409), which converges the
// AES-coupled map where damped simple mixing stalls or lands in a
// polarised attractor (MgO: 87 iterations to the cubic-symmetric state
// against no convergence in 3000).  The Broyden history is capped at
// kPeriodicGFN2BroydenMemory vectors.
inline constexpr int kPeriodicGFN2BroydenMemory = 64;
inline constexpr double kPeriodicGFN2BroydenMixing = 0.4;

// Default second-order kernel of the periodic GFN2 driver when the caller
// leaves XTBSccOptions::gamma_form unset: the Elstner 1998 form, whose
// remainder sum converges absolutely in every dimension (maintainer decision
// D1, 2026-08-28).  The molecular driver's unset default stays the published
// Klopman-Ohno kernel.
inline constexpr ShellGammaForm kPeriodicGFN2DefaultGammaForm =
    ShellGammaForm::Elstner;

// ---------------------------------------------------------------------------
// Geometry-only assembly shared by the SCF and its derivatives
// ---------------------------------------------------------------------------

struct PeriodicGFN2Assembly {
    Molecule mol;
    BasisSet basis;
    std::vector<LatticeCell> cells;
    std::vector<Eigen::Vector3d> lattice_translations;  // the dim periodic vectors
    double cutoff_bohr = 0.0;
    // The short-range gamma remainder is summed over its own pair range:
    // the larger of the H0/S cutoff and the Elstner decay range
    // (shell_gamma_remainder_range at kGammaRemainderTolerance) over every
    // shell pair, so the lattice-summed gamma is converged independently of
    // the overlap cutoff.  For the Klopman-Ohno form both ranges coincide.
    double gamma_cutoff_bohr = 0.0;
    std::vector<LatticeCell> gamma_cells;
    ShellGammaSpec gamma_spec;
    std::vector<GFN2ShellInfo> shell_info;
    std::vector<int> ao_shell;
    std::vector<int> ao_atom;
    std::vector<Eigen::Vector3d> positions;
    std::vector<GammaSite> gamma_sites;
    Eigen::VectorXd n0_shell;
    Eigen::VectorXd gam3_shell;
    bool has_gam3 = false;
    int n_basis = 0;
    int n_atoms = 0;
    int n_shells = 0;
    int n_occ = 0;
    int n_valence_electrons = 0;
    LatticeMatrixSet overlap_blocks;      // masked S(g)
    Eigen::MatrixXd S_gamma;
    Eigen::MatrixXd H0_gamma;
    Eigen::VectorXd coordination_numbers; // Eq. 18, periodic
    Eigen::MatrixXd gamma_shell;          // lattice-summed shell gamma
    GFN2MultipoleLatticeSums multipole_sums;
    GFN2AesAtomParameters aes_params;
    double e_repulsive = 0.0;

    // Every atom pair and lattice image inside the pair cutoff (self images
    // included, the g = 0 self pair excluded), unit weight.
    ImageRecordSource records() const;
    // The same over the gamma remainder range.
    ImageRecordSource gamma_records() const;
    EwaldCoulombKernel ewald() const;
};

// Absolute tolerance on the neglected Elstner remainder tail per pair.
inline constexpr double kGammaRemainderTolerance = 1.0e-10;

PeriodicGFN2Assembly assemble_periodic_gfn2(
    const PeriodicSystem& system,
    const GFN2ParameterSet& params,
    double cutoff_bohr,
    ShellGammaForm gamma_form);

/// Reject periodic compositions whose current GFN2 implementation is known
/// to return nonphysical energies.
void validate_periodic_gfn2_physics_domain(const PeriodicSystem& system);

// ---------------------------------------------------------------------------
// Run periodic GFN2-xTB SCC calculation at Gamma-point
// ---------------------------------------------------------------------------

/// Run the Gamma-point periodic GFN2-xTB SCC calculation.
///
/// `scc_opts.max_iter` is a hard total iteration budget, including any
/// automatic stabilization restart.
PeriodicGFN2Result run_gfn2_xtb_gamma(
    const PeriodicSystem& system,
    const GFN2ParameterSet& params,
    const XTBSccOptions& scc_opts = {},
    double cutoff_bohr = 15.0);

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
