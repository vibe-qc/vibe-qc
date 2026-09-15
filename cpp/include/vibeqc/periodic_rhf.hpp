// Γ-only restricted Hartree-Fock for a periodic system, molecular-limit
// regime.
//
// Scope note. This driver solves a single-k (Γ) closed-shell SCF with the
// real-space density taken as Γ-only — i.e., P_μν(g) is nonzero only for
// g = 0. That's physically meaningful in the molecular-limit regime
// (unit cell large enough that cross-cell density matrix elements are
// numerically zero), where it must reproduce molecular RHF to machine
// precision.
//
// Bulk calculations with genuinely metallic or covalent inter-cell
// bonding need multi-k sampling of P(g) — that arrives in Phase 12c.
//
// Implementation reuses the existing DIIS machinery: at Γ the Fock /
// overlap / density matrices are real nbf × nbf, so Pulay's FDS - SDF
// error vector and the linear-combination extrapolation work unchanged.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "basis.hpp"
#include "ecp.hpp"       // ECPPrimitiveBlock
#include "davidson.hpp"     // Davidson iterative diagonalizer
#include "dft_plus_u.hpp"  // HubbardSiteCxx
#include "ediis.hpp"    // SCFAccelerator
#include "guess.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"
#include "periodic_mean_field_state.hpp"
#include "rhf.hpp"      // SCFIteration

namespace vibeqc {

struct PeriodicRHFOptions {
    // SCF convergence controls (same shape as molecular RHFOptions).
    int max_iter = 100;
    double conv_tol_energy = 1.0e-8;
    double conv_tol_grad = 1.0e-6;
    double damping = 0.5;
    // Dynamic damping (Zerner-Hehenberger 1979). See RHFOptions::dynamic_damping.
    // Default true: adaptive damping helps periodic SCF, which is prone to
    // charge-sloshing oscillation. ``damping`` starts at 0.5 and relaxes
    // toward 0.0 as the energy decreases monotonically.
    bool dynamic_damping = true;
    double dynamic_damping_min = 0.0;
    double dynamic_damping_max = 0.95;
    // Fock matrix mixing; see RHFOptions::fock_mixing. This mirrors
    // CRYSTAL's FMIXING keyword and is independent of density damping.
    double fock_mixing = 0.0;

    bool use_diis = true;
    int diis_start_iter = 2;
    std::size_t diis_subspace_size = 8;

    // SCF Fock-extrapolation accelerator (DIIS / KDIIS / EDIIS /
    // EDIIS_DIIS). See RHFOptions::scf_accelerator and
    // cpp/include/vibeqc/ediis.hpp for the family. Default EDIIS_DIIS
    // (the production hybrid, robust far from convergence). Ignored when
    // use_diis = false.
    SCFAccelerator scf_accelerator = SCFAccelerator::EDIIS_DIIS;
    // EDIIS → DIIS switch threshold for SCFAccelerator::EDIIS_DIIS.
    // See RHFOptions::ediis_diis_switch_threshold.
    double ediis_diis_switch_threshold = 1e-1;

    // Adaptive-depth CDIIS parameters (Chupin et al. 2021). See
    // RHFOptions::diis_restart_tau / diis_adaptive_delta. Consulted only for
    // scf_accelerator R_CDIIS (τ) / AD_CDIIS (δ); both default to 1e-4.
    double diis_restart_tau = 1e-4;
    double diis_adaptive_delta = 1e-4;

    // v0.9.x: default flipped to AUTO. The unified GuessEngine
    // (cpp/src/guess.cpp) resolves AUTO → SAD for any periodic system
    // — fixes the historic NaCl/MgO bombing class of failure where
    // Hcore's neglect of electron repulsion sends DIIS into
    // +30 000 / −16 000 Ha territory before recovery. The v0.6.2
    // calibration freeze that kept Hcore in place is lifted; tests
    // that were calibrated to Hcore iteration counts have been refreshed.
    InitialGuess initial_guess = InitialGuess::AUTO;

    // Phase C1a — Saunders-Hillier level shift (Hartree). Adds
    // ``b · (S - ½ S D S)`` to F before diagonalisation, which
    // raises virtual orbital eigenvalues by ``b`` while leaving
    // occupied orbitals unchanged. The shift is "inert at the
    // converged density" — it doesn't change the SCF fixed point,
    // only the iteration dynamics. Useful when DIIS oscillates
    // between near-degenerate occupied / virtual swaps on small-
    // HOMO–LUMO-gap insulators or weakly metallic cells. Default
    // 0.0 (no shift). Typical values: 0.1 – 0.5 Hartree.
    double level_shift = 0.0;

    // CRYSTAL-style level-shift warm-up. ``-1`` means auto: if
    // level_shift is nonzero, use up to five shifted startup cycles
    // and leave at least one unshifted tail cycle. ``0`` keeps the
    // historic persistent level-shift behavior. Positive values request
    // an explicit warm-up length.
    int level_shift_warmup_cycles = -1;

    // Explicit per-iteration level-shift curve (CRYSTAL ``LEVSHIFT B
    // IRESET`` style). Empty ⇒ derive the per-iteration shift from
    // ``level_shift`` + ``level_shift_warmup_cycles``; non-empty ⇒ use it
    // directly (``schedule[iter-1]``, last entry reused past its length,
    // 0.0 releases). Resolved by ``level_shift_at_iter`` in
    // level_shift.hpp, the same helper the molecular drivers use.
    std::vector<double> level_shift_schedule;

    // Phase C1b — Fermi-Dirac smearing temperature (Hartree).
    // ``T > 0`` replaces hard Aufbau (occupations ∈ {0, 2}) with a
    // smooth Fermi function ``n_i = 2 / (1 + exp((ε_i - μ) / T))``
    // and bisects for ``μ`` to satisfy the per-cell electron-count
    // constraint. Required for metals and small-gap insulators where
    // Aufbau gives oscillating fixed points. Adds an electronic-
    // entropy contribution to the free energy ``A = E - T S``;
    // converged calculations report both the internal energy and the
    // free energy. Default 0.0 (no smearing → standard Aufbau).
    // Typical values: 0.001–0.02 Hartree (~315–6300 K).
    double smearing_temperature = 0.0;

    // Phase C1c — second-order ("quadratic") SCF fallback.
    //
    // When standard SCF (damping + DIIS + level shift) fails to
    // converge — typically on small-gap insulators where DIIS
    // oscillates between near-degenerate occ/vir swaps — switch from
    // "diagonalize F" to a Newton step in MO space:
    //
    //   κ_{ai}  =  -F_{ai}^{MO} / (ε_a - ε_i + quadratic_fallback_shift)
    //   C_new   =  C_prev · exp(κ)            (skew-symmetric κ)
    //   D_new   =  2 · C_new_occ · C_new_occ^T
    //
    // The step is preconditioned by orbital-energy differences (the
    // diagonal of the orbital-rotation Hessian) and trust-region
    // capped at ``quadratic_fallback_max_step`` to keep the Taylor
    // exponential well-conditioned.
    //
    // Activation. ``quadratic_fallback_iter > 0`` enables the
    // fallback after iteration ``quadratic_fallback_iter``.
    // ``= 0`` (default) disables — SCF behaves exactly as before.
    // DIIS and level shift are skipped during the quadratic phase
    // (the Newton step is its own update mechanism, and mixing
    // with DIIS extrapolation undoes the trust-region cap).
    //
    // When this helps. Insulators with HOMO-LUMO gap < ~0.1 eV;
    // open-shell systems with near-degenerate ground / excited
    // states; high-symmetry crystals where many MO pairs are close.
    int quadratic_fallback_iter = 0;
    double quadratic_fallback_shift = 0.1;
    double quadratic_fallback_max_step = 0.1;

    // Lattice-sum controls applied to one-electron and J/K builds alike.
    LatticeSumOptions lattice_opts = {};

    // Inline ECP operator -- the one periodic ECP route (contract on
    // PeriodicKSOptions). run_periodic_job fills these from the basis'
    // .ecp sidecar or the pob-TZVP-rev2 CRYSTAL records; the k-point GDF
    // drivers add the lattice-summed V_ECP to H_core, use Z_eff = Z - n_core
    // in V_ne and the ionic repulsion, and fill n_electrons - ecp_total_ncore
    // electrons. Until 2026-09 this struct had no such fields, so every
    // periodic HF request on an ECP basis died on the assignment (#88).
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_home_centers;
    std::vector<double> ecp_effective_charges;
    int ecp_total_ncore = 0;

    // DFT+U (Dudarev). Γ-only periodic +U uses the *same* real-density
    // kernel as the molecular path: at Γ the density is a single real
    // nbf x nbf matrix, the AO overlap is a single real matrix, and
    // ao_group_indices(basis) returns the same shape because the basis
    // is built from PeriodicSystem.unit_cell_molecule(). Empty by
    // default. Multi-k periodic +U is documented in
    // docs/user_guide/dft_plus_u.md.
    std::vector<HubbardSiteCxx> dft_plus_u_sites;
    std::vector<std::vector<int>> dft_plus_u_ao_groups;

    // ---- Initial-guess / convergence features (periodic case) -----------
    // These mirror the molecular RHFOptions/UHFOptions fields. The periodic
    // SCF runs in the Python drivers, so the Python run_periodic_job wrapper
    // fills these and the driver consumes them; carrying them on the options
    // struct keeps the runner→driver flow uniform with ``initial_guess``.
    // PeriodicRHFOptions serves both Γ/multi-k RHF (closed-shell) and UHF
    // (open-shell), so it carries the closed and per-spin READ densities.

    // READ restart (Γ-only). The wrapper resolves the prior calculation's
    // g=0 cell density, projects it onto this cell's basis, and fills
    // ``read_density`` (closed-shell RHF) or ``read_density_alpha`` /
    // ``read_density_beta`` (open-shell UHF). The driver injects it at g=0;
    // it Bloch-sums to D(k) for the first Fock build. Empty = not a READ
    // start. Native multi-k RHF uses PeriodicSCFOptions.read_density_k;
    // the Python wrappers resolve complete result/QVF Bloch sources.
    Eigen::MatrixXd read_density;        // closed-shell total density
    Eigen::MatrixXd read_density_alpha;  // open-shell per-spin (alpha)
    Eigen::MatrixXd read_density_beta;   // open-shell per-spin (beta)
    std::string read_path;               // source .qvf / .molden (provenance)

    // ATOMSPIN broken-symmetry seed (open-shell UHF only): per-atom spin tag
    // (+1 majority alpha / -1 majority beta / 0 unpolarised) in unit-cell
    // atom order. Seeds an AFM / ferrimagnetic g=0 spin pattern that
    // Bloch-sums to a broken-symmetry D(k). Consulted for the SAD guess;
    // empty = spin-symmetric split. See GuessEngine::build_open_shell.
    std::vector<int> atomic_spins;

    // SPINLOCK (open-shell convergence aid for broken-symmetry magnetism):
    //   SPIN_SCHEDULE — two-phase SCF: converge at locked n_alpha-n_beta =
    //     spinlock_value for spinlock_iterations cycles, then restart at the
    //     multiplicity target from that density (CRYSTAL SPINLOCK n nstep).
    //   PATTERN_HOLD — hold the seeded broken-symmetry occupied set by
    //     maximum overlap (MOM, python/vibeqc/mom.py) for spinlock_iterations
    //     cycles, then release (protects an atomic_spins seed on the multi-k
    //     path). OFF = normal Aufbau. Mirrors molecular UHFOptions.
    SpinlockMode spinlock_mode = SpinlockMode::OFF;
    int spinlock_value = 0;        // target n_alpha-n_beta for SPIN_SCHEDULE
    int spinlock_iterations = 0;   // locked / held cycles before release

    // Davidson iterative diagonalization for large basis sets.
    bool use_davidson = false;
    DavidsonOptions davidson = {};
    int davidson_min_dim = 100;
};

struct PeriodicRHFResult {
    std::optional<BasisSet> restart_basis;
    Eigen::Matrix3d restart_lattice = Eigen::Matrix3d::Zero();
    Eigen::MatrixXd restart_kpoints;
    Eigen::VectorXd restart_weights;
    GuessSelection guess_selection;

    double energy = 0.0;           // Total Hartree–Fock energy per unit cell
    double e_electronic = 0.0;
    double e_nuclear = 0.0;        // Sum over unit-cell pairs within nuclear
                                   // cutoff (matches V lattice sum)
    // Dudarev DFT+U contribution per unit cell. 0 when +U is not active.
    double e_dft_plus_u = 0.0;
    int n_iter = 0;
    bool converged = false;

    // Γ-point molecular orbitals.
    Eigen::VectorXd mo_energies;   // ascending (Hartree)
    Eigen::MatrixXd mo_coeffs;     // real at Γ

    // Final Γ-aggregated matrices.
    // Complete Bloch orbitals; density below is only the home-cell block.
    std::vector<Eigen::MatrixXcd> mo_coeffs_k;
    std::vector<Eigen::VectorXd> occupations_k;
    std::vector<Eigen::VectorXd> mo_energies_k;
    Eigen::MatrixXd density;
    Eigen::MatrixXd fock;
    Eigen::MatrixXd overlap;

    // Present only for the explicit experimental multi-k correlation-capture
    // entry point. Ordinary and Gamma periodic RHF leave this null; their
    // restart orbitals above do not constitute a captured post-HF payload. Shared immutable ownership preserves the
    // result's historical copyability without copying the state matrices.
    std::shared_ptr<const PeriodicRestrictedMeanFieldState> mean_field_state;

    std::vector<SCFIteration> scf_trace;
};

// Nuclear-repulsion energy per unit cell, with the reference cell's nuclei
// interacting with all images within the lattice-sum cutoff. A half-sum
// handles the pair double-counting correctly: each pair of unit-cell atoms
// is counted once across all cell shifts (including g = 0, which reduces
// to the intra-cell molecular repulsion).
double nuclear_repulsion_per_cell(const PeriodicSystem& system,
                                  const LatticeSumOptions& opts);

PeriodicRHFResult run_rhf_periodic_gamma(const PeriodicSystem& system,
                                         const BasisSet& basis,
                                         const PeriodicRHFOptions& options = {});

}  // namespace vibeqc
