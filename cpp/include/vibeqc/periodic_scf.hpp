// Multi-k periodic SCF driver and density-matrix folding helpers.
//
// Generalises Phase 12b's Γ-only driver to a full Monkhorst–Pack mesh with
// the density matrix maintained in real space:
//
//   P_μν(g) = (1/N_k) Σ_k w_k  exp(−i k · g)  P_μν(k)
//
// with P(k) = 2 Σ_i^occ C_i(k) C_i(k)† the standard closed-shell density at
// each k, and w_k the k-point weight (sums to 1 over the IBZ or full mesh).
//
// For time-reversal-symmetric systems (all one-electron and two-electron
// operators here) P(g) is real; we store it as a LatticeMatrixSet of real
// blocks, and the SCF runs entirely in real arithmetic except at the
// per-k Fock diagonalisation step.
//
// Scope note. ``run_rhf_periodic`` is independent of
// ``run_rhf_periodic_gamma``: 12b kept the Γ-only molecular-limit path
// because it's easier to debug and captures the full periodic 1e + 2e
// infrastructure. Multi-k layers the BZ-average density and the general
// three-lattice-index Fock build on top.

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <vector>

#include <string>

#include "basis.hpp"
#include "bloch.hpp"
#include "davidson.hpp"     // Davidson iterative diagonalizer
#include "dft_plus_u.hpp"
#include "ecp.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"
#include "periodic_rhf_state_capture.hpp"
#include "periodic_rhf.hpp"  // SCFIteration, PeriodicRHFResult
#include "rks.hpp"           // RKSResult

namespace vibeqc {

struct PeriodicSCFOptions {
    int max_iter = 100;
    double conv_tol_energy = 1.0e-8;
    double conv_tol_grad = 1.0e-6;
    double damping = 0.5;
    // Dynamic damping (Zerner-Hehenberger 1979). See RHFOptions::dynamic_damping.
    // Default true: adaptive damping helps charge-sloshing-prone periodic SCF.
    bool dynamic_damping = true;
    double dynamic_damping_min = 0.0;
    double dynamic_damping_max = 0.95;
    // Fock matrix mixing; see RHFOptions::fock_mixing.
    double fock_mixing = 0.0;

    bool use_diis = true;
    int diis_start_iter = 2;
    std::size_t diis_subspace_size = 8;

    // SCF Fock-extrapolation accelerator. See RHFOptions::scf_accelerator
    // and cpp/include/vibeqc/ediis.hpp for the family. Default EDIIS_DIIS
    // (production hybrid, robust far from convergence).
    SCFAccelerator scf_accelerator = SCFAccelerator::EDIIS_DIIS;
    double ediis_diis_switch_threshold = 1e-1;

    // Adaptive-depth CDIIS parameters (Chupin et al. 2021). See
    // RHFOptions::diis_restart_tau / diis_adaptive_delta. Consulted only for
    // scf_accelerator R_CDIIS (τ) / AD_CDIIS (δ); both default to 1e-4.
    double diis_restart_tau = 1e-4;
    double diis_adaptive_delta = 1e-4;

    // v0.9.x: default flipped to AUTO. GuessEngine resolves AUTO → SAD
    // for any periodic system. The v0.6.2 test-suite calibration freeze
    // that kept HCORE in place is lifted.
    InitialGuess initial_guess = InitialGuess::AUTO;

    // READ payload in the target AO basis and target k-mesh order. Counts
    // are normalized with the complete BZ quadrature, never independently
    // at each k. The real-space driver requires time-reversal symmetry.
    std::vector<ComplexMatrix> read_density_k;

    // Phase C1a level shift (Hartree). See PeriodicRHFOptions for
    // the full description. Default 0.0 (no shift).
    double level_shift = 0.0;

    // CRYSTAL-style level-shift warm-up. See PeriodicRHFOptions for
    // the full description. ``-1`` auto, ``0`` persistent shift,
    // positive values explicit warm-up length.
    int level_shift_warmup_cycles = -1;

    // Explicit per-iteration level-shift curve. See PeriodicRHFOptions::
    // level_shift_schedule for the full description. Empty ⇒ derive from
    // the warm-up logic; non-empty ⇒ use it directly.
    std::vector<double> level_shift_schedule;

    // Phase C1b — Fermi-Dirac smearing temperature (Hartree).
    // See PeriodicRHFOptions for the full description. Default 0.0
    // (no smearing → standard Aufbau).
    double smearing_temperature = 0.0;

    // Phase C1c — second-order ("quadratic") SCF fallback. See
    // PeriodicRHFOptions for the full description. Default 0
    // (disabled).
    int quadratic_fallback_iter = 0;
    double quadratic_fallback_shift = 0.1;
    double quadratic_fallback_max_step = 0.1;

    LatticeSumOptions lattice_opts = {};

    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_home_centers;
    std::vector<double> ecp_effective_charges;
    int ecp_total_ncore = 0;

    // Davidson iterative diagonalization for large basis sets.
    bool use_davidson = false;
    DavidsonOptions davidson = {};
    int davidson_min_dim = 100;
};

// Real-space density matrix from a per-k list of MO coefficients.
//
// ``C_per_k`` has one complex nbf × nbf matrix per k-point with columns
// ordered by ascending band energy (same ordering the diagonaliser
// produces). ``n_occ_per_k`` is the number of occupied MOs at each k
// (for closed-shell insulators: n_elec / 2 at every k).
//
// Returns P(g) keyed by the supplied cell list — each block is real.
//
// ----------------------------------------------------------------------
// Fractional-occupation variant (Phase C1b — Fermi-Dirac smearing).
//
// Same contract, but ``occ_per_k`` is a per-k vector of *fractional*
// occupation numbers (doubles, typically in [0, 2] for closed-shell
// RHF). The density is built as
//    P(k) = Σ_i occ_i^k · C_i^k (C_i^k)†   (= C diag(occ) C†)
// with the same Bloch-summed real-space fold as the integer-occupation
// version. This is the path used by smearing-driven SCF on metals /
// small-gap insulators where Aufbau gives oscillating fixed points.
// ----------------------------------------------------------------------
LatticeMatrixSet real_space_density_from_kpoints_fractional(
    const std::vector<ComplexMatrix>& C_per_k,
    const std::vector<Eigen::VectorXd>& occ_per_k,
    const BlochKMesh& kmesh,
    const std::vector<LatticeCell>& cells);

LatticeMatrixSet real_space_density_from_kpoints(
    const std::vector<ComplexMatrix>& C_per_k,
    const std::vector<int>& n_occ_per_k,
    const BlochKMesh& kmesh,
    const std::vector<LatticeCell>& cells);

// Multi-k closed-shell periodic RHF. Uses the full Monkhorst–Pack mesh
// (IBZ reduction reserved for when per-basis D(R) transforms are in place).
PeriodicRHFResult run_rhf_periodic(const PeriodicSystem& system,
                                   const BasisSet& basis,
                                   const BlochKMesh& kmesh,
                                   const PeriodicSCFOptions& options = {});

// Experimental direct-truncated RHF producer for a future native periodic
// correlation orchestrator. The request is passed by value so its explicit
// per-k masks can be moved into the returned state. This entry point performs
// a retained-payload preflight, but does not claim total-SCF or correlation
// resource admission and is intentionally separate from ordinary SCF options.
PeriodicRHFResult run_rhf_periodic_with_state_capture(
    const PeriodicSystem& system,
    const BasisSet& basis,
    const BlochKMesh& kmesh,
    const PeriodicSCFOptions& options,
    PeriodicRHFStateCaptureRequest request);

// Multi-k closed-shell periodic Kohn–Sham DFT (LDA / pure GGA / hybrid).
//
// The XC potential is computed by numerical integration over the
// unit-cell DFT grid. The default partition (``use_periodic_becke =
// true``, since v0.9.x) is the image-aware periodic Becke fuzzy-cell:
// the denominator runs over home + image atoms within
// ``becke_image_radius_bohr``, so ∫_cell normalisation holds even
// when image atoms compete for partition weight near the unit-cell
// boundary. In the molecular-limit regime (no image atom within
// reach) it reduces exactly to the molecular Becke partition.
//
// Set ``use_periodic_becke = false`` only to reproduce v0.8.x and
// older numerics — on any tight crystal (lattice constant ≲ 12 bohr)
// that path silently corrupts E_xc by tens to hundreds of mHa per
// atom (the molecular partition over-counts weight at the unit-cell
// boundary; see docs/tutorial/23_tight_cell_dft.md).
struct PeriodicKSOptions {
    std::string functional = "LDA";
    GridOptions grid;

    int max_iter = 100;
    double conv_tol_energy = 1.0e-8;
    double conv_tol_grad = 1.0e-6;
    double damping = 0.5;
    // Dynamic damping (Zerner-Hehenberger 1979). See RHFOptions::dynamic_damping.
    // Default true: adaptive damping helps charge-sloshing-prone periodic SCF.
    bool dynamic_damping = true;
    double dynamic_damping_min = 0.0;
    double dynamic_damping_max = 0.95;
    // Kohn-Sham matrix mixing; see RHFOptions::fock_mixing.
    double fock_mixing = 0.0;

    bool use_diis = true;
    int diis_start_iter = 2;
    std::size_t diis_subspace_size = 8;

    // SCF Fock-extrapolation accelerator. See RHFOptions::scf_accelerator
    // and cpp/include/vibeqc/ediis.hpp for the family. Default EDIIS_DIIS
    // (production hybrid, robust far from convergence).
    SCFAccelerator scf_accelerator = SCFAccelerator::EDIIS_DIIS;
    double ediis_diis_switch_threshold = 1e-1;

    // Adaptive-depth CDIIS parameters (Chupin et al. 2021). See
    // RHFOptions::diis_restart_tau / diis_adaptive_delta. Consulted only for
    // scf_accelerator R_CDIIS (τ) / AD_CDIIS (δ); both default to 1e-4.
    double diis_restart_tau = 1e-4;
    double diis_adaptive_delta = 1e-4;

    // v0.9.x: default flipped to AUTO. GuessEngine resolves AUTO → SAD
    // for any periodic system. The v0.6.2 test-suite calibration freeze
    // that kept HCORE in place is lifted.
    InitialGuess initial_guess = InitialGuess::AUTO;

    // READ payload in the target AO basis and target k-mesh order. Counts
    // are normalized with the complete BZ quadrature, never independently
    // at each k. The real-space driver requires time-reversal symmetry.
    std::vector<ComplexMatrix> read_density_k;

    // Phase C1a level shift (Hartree). See PeriodicRHFOptions for
    // the full description. Default 0.0 (no shift).
    double level_shift = 0.0;

    // CRYSTAL-style level-shift warm-up. See PeriodicRHFOptions for
    // the full description. ``-1`` auto, ``0`` persistent shift,
    // positive values explicit warm-up length.
    int level_shift_warmup_cycles = -1;

    // Explicit per-iteration level-shift curve. See PeriodicRHFOptions::
    // level_shift_schedule for the full description. Empty ⇒ derive from
    // the warm-up logic; non-empty ⇒ use it directly.
    std::vector<double> level_shift_schedule;

    // Phase C1b — Fermi-Dirac smearing temperature (Hartree).
    // See PeriodicRHFOptions for the full description. Default 0.0.
    double smearing_temperature = 0.0;

    // Phase C1c — second-order ("quadratic") SCF fallback. See
    // PeriodicRHFOptions for the full description. Default 0
    // (disabled).
    int quadratic_fallback_iter = 0;
    double quadratic_fallback_shift = 0.1;
    double quadratic_fallback_max_step = 0.1;

    LatticeSumOptions lattice_opts = {};

    // 12f periodic Becke controls. When ``use_periodic_becke`` is
    // true (default, since v0.9.x) the partition denominator is
    // extended over image atoms within ``becke_image_radius_bohr``,
    // restoring the correct ∫_cell normalisation on tight crystals;
    // false uses the plain molecular partition (matches v0.8.x and
    // earlier — silently wrong on tight cells; see the class-level
    // comment above).
    bool use_periodic_becke = true;
    double becke_image_radius_bohr = 10.0;

    // DFT+U (Dudarev). Periodic +U on this driver is **Γ-only** in
    // Increment 4b (single k-point at the Γ-point). The Python wrapper
    // enforces the kmesh restriction; passing a multi-k mesh together
    // with a non-empty ``dft_plus_u_sites`` raises at the boundary.
    // Multi-k periodic +U (the user-goal target — Σ_k w_k S(k) P(k)
    // S(k) AO occupation) lands in Increment 4c.
    std::vector<HubbardSiteCxx> dft_plus_u_sites;
    std::vector<std::vector<int>> dft_plus_u_ao_groups;

    // ---- Initial-guess / convergence features (periodic case) -----------
    // Mirror of the molecular RKSOptions/UKSOptions fields; see
    // PeriodicRHFOptions for the full description. PeriodicKSOptions serves
    // both RKS (closed-shell) and UKS (open-shell), so it carries the
    // closed and per-spin READ densities.
    Eigen::MatrixXd read_density;        // closed-shell total density (RKS)
    Eigen::MatrixXd read_density_alpha;  // open-shell per-spin (UKS, alpha)
    Eigen::MatrixXd read_density_beta;   // open-shell per-spin (UKS, beta)
    std::string read_path;               // source .qvf / .molden (provenance)

    // ATOMSPIN broken-symmetry seed (open-shell UKS only). See
    // PeriodicRHFOptions::atomic_spins.
    std::vector<int> atomic_spins;

    // SPINLOCK (open-shell UKS). See PeriodicRHFOptions::spinlock_mode.
    SpinlockMode spinlock_mode = SpinlockMode::OFF;
    int spinlock_value = 0;
    int spinlock_iterations = 0;

    // ---- Effective core potentials ----------------------------------------
    // Inline-ECP primitive data (per-element libecpint arrays) and
    // per-home-atom centre positions, populated by the Python dispatch from
    // the pob-TZVP-rev2 CRYSTAL records or the basis' .ecp sidecar. This is
    // the only periodic ECP route: the driver adds the lattice-summed
    // V_ECP to the home-cell H_core, uses Z_eff = Z - n_core in V_ne and in
    // the ionic repulsion, and fills n_electrons - ecp_total_ncore
    // electrons. (An XML-library route was declared here until 2026-09 but
    // never consumed or bound.)
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_home_centers;
    std::vector<double> ecp_effective_charges;
    int ecp_total_ncore = 0;

    // Davidson iterative diagonalization for large basis sets.
    bool use_davidson = false;
    DavidsonOptions davidson = {};
    int davidson_min_dim = 100;
};

struct PeriodicKSResult {
    std::optional<BasisSet> restart_basis;
    Eigen::Matrix3d restart_lattice = Eigen::Matrix3d::Zero();
    Eigen::MatrixXd restart_kpoints;
    Eigen::VectorXd restart_weights;
    GuessSelection guess_selection;

    double energy = 0.0;
    double e_electronic = 0.0;
    double e_coulomb = 0.0;
    double e_hf_exchange = 0.0;   // −(α/2) tr(P·K), 0 for pure DFT
    double e_xc = 0.0;
    double e_nuclear = 0.0;
    // Dudarev DFT+U contribution per unit cell (Γ-only in 4b; multi-k
    // in 4c). 0 unless PeriodicKSOptions.dft_plus_u_sites is non-empty.
    double e_dft_plus_u = 0.0;
    int n_iter = 0;
    bool converged = false;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    // Complete Bloch orbitals; density below is only the home-cell block.
    std::vector<Eigen::MatrixXcd> mo_coeffs_k;
    std::vector<Eigen::VectorXd> occupations_k;
    std::vector<Eigen::VectorXd> mo_energies_k;
    Eigen::MatrixXd density;
    Eigen::MatrixXd fock;
    Eigen::MatrixXd overlap;
    std::vector<SCFIteration> scf_trace;
    std::string functional;
};

PeriodicKSResult run_rks_periodic(const PeriodicSystem& system,
                                  const BasisSet& basis,
                                  const BlochKMesh& kmesh,
                                  const PeriodicKSOptions& options = {});

}  // namespace vibeqc
