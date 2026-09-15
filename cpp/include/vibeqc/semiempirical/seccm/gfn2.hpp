// GFN2-xTB adapter for a frozen semiempirical cyclic-cluster topology,
// WS-weighted supercell SCC.
//
// The supercell Hamiltonian is the molecular GFN2-xTB machinery evaluated
// over the WS record set (the Bredow-Geudtner-Jug construction, the same
// assembly pattern as the PM6/OMx/SCC-DFTB SECCM adapters):
//   * S and H0: the one-center blocks are the molecular ones; every directed
//     two-center block is build_gfn2_hamiltonian_zero_image at the record
//     displacement, accumulated with its fractional WS weight. The image
//     blocks keep the H(g) = H(-g)^T nonsymmetric block convention of the
//     Gamma-periodic driver, and the validated reverse-image topology makes
//     the assembled matrices symmetric.
//   * Shell gamma: the same-atom block is the molecular one (Klopman-Ohno at
//     R = 0); every other atom contributes through its WS records at the
//     record distances with their fractional weights.
//   * AES: the molecular driver's shipped default path, the ad-hoc
//     shell-resolved gamma^n multipole potential, accumulated over the WS
//     records with their fractional weights (each record carries its weight
//     through the potential sums; the pair-additive AES has no three-center
//     terms, so the Peintinger-Bredow eq-13 union construction reduces to
//     the plain two-center record weight). The experimental atom-resolved
//     "faithful" AES (aes_potentials / aes_energy, gated behind
//     XTBSccOptions::aes_faithful in the molecular driver) is not part of
//     this adapter.
//   * GAM3 on-site third order and the WS-weighted pair repulsion complete
//     the energy functional; the per-primitive-cell energy is the raw
//     finite-cluster total divided by the finite-group order.
//
// The SCC loop mirrors run_gfn2_xtb: damped simple mixing (the GFN2
// third-order term makes every element's has_gam3 true, so the molecular
// driver's mixer selection always lands on simple mixing), the same
// stabilization retry ladder, and the same energy decomposition
// (band0 + 1/2 q.gamma.q + AES + GAM3 + repulsion). D4 dispersion stays out,
// exactly as in the molecular driver.
//
// The T=0 molecular limit (one replica, zero-translation images only) is
// evaluated with the validated molecular driver xtb::run_gfn2_xtb
// bit-for-bit. At finite T the state is identical but SECCM's energy field is
// the variational Mermin free energy rather than molecular GFN2's historical
// internal-energy field.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/charge_mixer.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"
#include "vibeqc/semiempirical/seccm/topology.hpp"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

struct GFN2SECCMResult : ParameterIdentifiedResult {
    double energy = 0.0;  // Total energy per primitive cell (Mermin A at T > 0).
    double free_energy = 0.0;  // Mermin A = E - T*S, per primitive cell
    double entropy = 0.0;      // electronic entropy, per primitive cell
    double smearing_temperature = 0.0;  // Ha (0 = hard Aufbau)
    // Free electronic component per cell: band0 + SCC + AES + third order
    // + optional Madelung - T*S.  Together with e_repulsive it closes to
    // energy on every path.
    double e_electronic = 0.0;
    double e_repulsive = 0.0;   // WS-weighted pair repulsion, per cell.
    double e_scc = 0.0;         // 1/2 q.gamma.q, per cell.
    double e_band0 = 0.0;       // sum D.H0, per cell.
    double e_aes = 0.0;         // anisotropic 2nd order, per cell.
    double e_3rd = 0.0;         // on-site third order, per cell.
    double e_madelung = 0.0;    // 1/2 dq . V_mad self-energy, per cell
                                // (0 when the embedding is off).
    double total_cyclic_energy = 0.0;
    double cyclic_electronic_energy = 0.0;
    double cyclic_repulsive_energy = 0.0;
    double cyclic_scc_energy = 0.0;
    double cyclic_aes_energy = 0.0;
    double cyclic_3rd_energy = 0.0;
    double cyclic_band0_energy = 0.0;
    double cyclic_madelung_energy = 0.0;
    // Frontier gap of the converged finite torus.  Measured and recorded
    // whenever a frontier exists (0 < n_occ < n_basis), at every
    // electronic temperature -- a smeared run records its real gap, not a
    // placeholder zero.  Left at 0.0 only when no frontier is defined.
    // NaN on a rejected screened attempt with no retained virtual orbital.
    double homo_lumo_gap = 0.0;
    // True when `homo_lumo_gap` is at or below the applied guard epsilon
    // (`run_controls.finite_torus_gap_tolerance`) and the positive-gap
    // requirement was waived because finite electronic temperature
    // resolves the occupation uniquely (Fermi-Dirac).  At T = 0 such a
    // state is rejected instead; a waived row is admissible but is NOT a
    // gapped row and must be screened accordingly.
    bool gap_guard_waived = false;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd hamiltonian;
    Eigen::VectorXd charges;
    // Shell Mulliken charges relative to the GFN2 reference occupations
    // (dq_shell = population - n0, supercell shell order). Exposed as a
    // diagnostic: intra-atomic shell transfer (for example s -> d) is
    // invisible in the atomic charges and is the signature of the
    // over-polarized SCC fixed point that metallic supercells can visit.
    Eigen::VectorXd shell_charges;
    // Per-iteration max |dq_new - dq| (SCC residual trace), one entry per
    // iteration up to the exit point. Exposed for convergence diagnostics:
    // a limit cycle shows as a sustained oscillation, a slow crawl as a
    // monotone decay.
    Eigen::VectorXd scc_max_change_trace;
    int n_basis = 0;
    int n_occ = 0;
    int n_iter = 0;
    int group_order = 0;
    int n_records = 0;
    bool converged = false;
    // Ordered SCC attempts plus their exact residual traces.  The aggregate
    // trace above is their concatenation and n_iter is their iteration sum.
    std::vector<xtb::GFN2SCCAttempt> attempts;
    int selected_attempt_index = -1;
    // Requested Hamiltonian selectors echoed from GFN2SECCMOptions. These
    // preserve the native option identity across the Python/C++ boundary;
    // smearing_temperature and molecular_delegated below describe the route
    // that actually executed after bounded retries or molecular delegation.
    double requested_electronic_temperature = 0.0;
    bool requested_madelung = false;
    bool requested_madelung_s_weighted = false;
    bool requested_madelung_no_self = false;
    bool requested_ewald_gamma = false;
    bool requested_include_aes = true;
    bool requested_ewald_gamma_molecular_onsite = false;
    // Requested solver/restart/compatibility controls echoed across the
    // native boundary.  Python canonicalizes these into immutable run
    // provenance and verifies that the core consumed the same request.
    int requested_max_iter = 0;
    double requested_conv_tol_charge = 0.0;
    double requested_charge_mixing = 0.0;
    SCCMixer requested_scc_mixer = SCCMixer::Simple;
    bool requested_ewald_gamma_k0_global = false;
    double requested_finite_torus_gap_tolerance = 0.0;
    Eigen::VectorXd requested_initial_shell_charges;
    // True when the zero-image record set executed through the exact
    // molecular GFN2 driver rather than a periodic electrostatics kernel.
    bool molecular_delegated = false;
    // False when the returned state violates shell-capacity bounds or the
    // conservative RMS/local charge-polarization heuristics in gfn2.cpp. These
    // catch over-polarized shell-transfer and charge-density-wave basins.
    bool physical_basin = true;
};

struct GFN2SECCMOptions {
    int max_iter = 3600;
    double conv_tol_charge = 1.0e-6;
    double charge_mixing = 0.1;
    // Fermi electronic temperature in Ha (0 = hard Aufbau, default).
    // A nonzero value smooths frontier crossings: metallic supercells
    // (bulk Cu/Pd/Ag, doped semiconductors) have a degenerate frontier
    // whose T=0 occupation can switch branches between neighbouring
    // geometries, making the energy surface discontinuous. The smeared
    // occupation converges branch-free and the reported energy is the
    // Mermin free energy A = E - T*S (free_energy alias, entropy and
    // smearing_temperature recorded). The insulating T=0 default is
    // unchanged.
    double electronic_temperature = 0.0;
    // Opt-in long-range embedding for neutral ionic cells: the
    // Madelung/Ewald potential of the supercell atomic Mulliken
    // fluctuations joins the H diagonal (MSINDO CCM convention
    // F_add(k,k) = -V_mad) and its classical self-energy 1/2 dq . V_mad
    // the total energy, with the SMADEL short-range subtraction. The
    // WS-truncated shell gamma already carries the damped short-range
    // Klopman-Ohno part; the embedding restores the Coulombic image-sum
    // tail beyond the WS cell that ionic charge patterns need (the
    // missing tail overbinds ionic crystals and feeds the over-polarized
    // shell-transfer basin). Kernels: 1-D background-corrected Parry-type
    // wire Ewald (seccm/ewald_1d.h), 2-D Parry/Heyes Ewald, 3-D Ewald
    // (methods/indo/ccm_engine.hpp) - the same shared layer the
    // SCC-DFTB-SECCM adapter uses. Charged cells remain unsupported.
    bool madelung = false;
    // Experimental knob: deposit the Madelung potential S-weighted
    // (0.5 S (V_A + V_B), the gamma channel's form - the variational
    // convention for the non-orthogonal basis) instead of the MSINDO
    // diagonal form. Default false is the shipped convention. true is
    // for channel isolation of SCC instabilities only (IID 150: a synthetic
    // aligned-plane stress cell makes the diagonal-form/Mulliken-response
    // mismatch violent); the reported energy then uses a different
    // deposit convention and must not be used for production numbers.
    bool madelung_s_weighted = false;
    // Experimental knob: zero the embedding self term (the madkonst
    // diagonal - the 2-D Parry/Heyes lattice self potential ~ -0.25 Ha
    // that SMADEL does not subtract because the self record sits at
    // zero displacement). Default false is the shipped convention. true
    // is for channel isolation of SCC instabilities only (IID 150: the
    // self term anti-screens each atom's own charge on the synthetic
    // aligned-plane stress map). The same diagonal-removed operator is used
    // in both the SCC potential and reported Madelung energy. This remains
    // a diagnostic model change and must not be used for production numbers.
    bool madelung_no_self = false;
    // Opt-in self-consistent periodic shell gamma for 1-D wires and 2-D
    // slabs: full bare-Coulomb Ewald plus WS-folded KO-minus-1/R and
    // on-site hardness. Non-molecular 3-D cyclic topologies fail closed
    // (#444): the finite WS remainder has no thermodynamic limit there.
    // Exact zero-image molecular delegation is unchanged; the internal
    // 3-D kernel remains diagnostic only. Mutually exclusive with madelung.
    // See docs/design_seccm_gfn2_long_range_gamma.md.
    bool ewald_gamma = false;
    // Experimental validation knob: disable the WS-folded anisotropic
    // second-order (AES) channel (the ad-hoc shell-resolved gamma^3/gamma^5
    // multipole potential and its energy term). Default true is the
    // molecular driver's shipped path. false is for channel isolation of
    // SCC instabilities only - the reported energy then misses a physical
    // term and must not be used for production numbers. It also disables the
    // exact molecular shortcut because that driver cannot remove this term.
    bool include_aes = true;
    // Experimental validation knob: restore the molecular on-site block
    // in the periodic shell gamma (skip the Ewald lattice self potential
    // at d = 0, which for the 2-D Parry/Heyes kernel carries a negative
    // k0/self shift). Default false is the shipped tblite-style
    // construction. true is for channel isolation of SCC instabilities only
    // (IID 150: the aligned-plane stress-map anti-screening runaway) - the
    // reported energy then uses a hybrid kernel and must not be used for
    // production numbers.
    bool ewald_gamma_molecular_onsite = false;
    // Deprecated compatibility flag. Both values now retain the complete
    // Parry/de Leeuw pairwise K=0 kernel. The former true branch used only
    // the neutral-cell quadratic Taylor term c*z*z^T, which is not exact for
    // higher perpendicular moments and could change energies by tens of mHa.
    bool ewald_gamma_k0_global = false;
    // Functional form of the short-range shell gamma under `ewald_gamma`.
    //
    // KlopmanOhno (the default) is the GFN2 paper's own kernel and keeps the
    // shipped 1-D/2-D numbers bit-for-bit.  Its remainder gamma - 1/R decays
    // as -eta^2/2R^3, so the Wigner-Seitz-folded sum is finite but has no
    // thermodynamic limit; 3-D cyclic topologies fail closed with it (#444).
    //
    // Elstner is the Elstner et al. 1998 Eq. 17/18 form (tau = 16/5 U),
    // whose remainder decays exponentially.  Its image sum converges
    // absolutely in every dimension, so it is the form 3-D cells must use,
    // and it changes the 1-D/2-D numbers.  Selecting it is the maintainer's
    // D1 decision of 2026-08-28, staged: the flag exists first, the default
    // moves only with a measured repin table.
    //
    // Without `ewald_gamma` the route is the unembedded WS-truncated
    // Klopman-Ohno sum and this field is not consulted.
    ShellGammaForm gamma_form = ShellGammaForm::KlopmanOhno;
    // Faithful record-resolved anisotropic electrostatics (issue #348).
    //
    // The shipped path (false) is the ad-hoc shell-resolved gamma^3/gamma^5
    // multipole kernel on moments built from the home-cell molecule.  Those
    // moments do not see the Wigner-Seitz image inventory, so retyping a
    // boundary site by a lattice vector moves the energy: measured on a
    // 4-unit polar HF chain, three representatives of one crystal give
    // -5.230514880891, -5.230364176724 and -5.230359290006 Ha, while with
    // include_aes=false all three agree bit-for-bit.  The AES channel is the
    // only remaining representative-dependent term in the route.
    //
    // true builds the multipole integrals over the same records as S and H0
    // and evaluates the Bannwarth 2019 Eq. 25 energy and Eqs. 39-44 Fock on
    // the resulting atom-resolved CAMM.  The two cannot be blended: the
    // ad-hoc kernel consumes shell-resolved moments, and a pair inventory
    // produces atom-resolved ones -- which is why this is a flag and not a
    // repair of the existing kernel.
    //
    // Default false while the molecular driver's own aes_faithful default is
    // false, so that the molecular-limit bit parity between the two routes
    // keeps meaning; the defaults move together.
    bool aes_faithful = false;
    // AES potential damping of the *molecular* driver, forwarded on a
    // zero-image delegating topology so the delegated solve runs the model
    // the caller asked for.  The SECCM engine itself does not use it: since
    // #409 the multipole moments are part of the mixed reduced state, so the
    // mixer supplies the damping and a separate potential lag would be a
    // second, uncontrolled one.
    double aes_damping = 0.5;
    // Charge mixer selection. Default Simple keeps the shipped behaviour
    // exactly. Broyden (and DIIS) are opt-in for maps whose fixed point
    // simple mixing cannot reach. Newton is a line-searched chord/quasi-
    // Newton step: its finite-difference charge-map Jacobian freezes the
    // lagged multipoles during each step, then refreshes them on the next
    // outer iteration. It uses a plain constrained Newton solve and
    // residual-reduction damping. It is the most expensive mixer (one
    // eigensolve per shell per step) and is meant for cycle-bound maps,
    // not routine runs.
    SCCMixer scc_mixer = SCCMixer::Simple;
    // Optional SCC restart: start the charge iteration from these shell
    // fluctuations (dq = population - n0) instead of the neutral zero
    // vector. Empty (the default) keeps the shipped neutral start. General
    // warm-start facility for continuation runs and defect studies; note
    // that map-level transients (IID 215) trigger from any near-physical
    // start, so a restart alone does not cure those. Must have exactly
    // n_shells finite entries when set; the driver validates before any SCC
    // work. Unavailable on an exact molecular-delegation topology because
    // the delegated molecular driver cannot consume a shell-charge restart.
    Eigen::VectorXd initial_shell_charges;
};

// Self-consistent periodic shell gamma for a cyclic SECCM topology, in one,
// two or three dimensions:
//
//   Gamma_ij = Phi_Ewald(R_b - R_a)
//            + sum_{WS records} w [gamma_ij(r) - 1/r]
//            + delta_{ab} onsite(U_i, U_j)
//
// This is semiempirical::build_periodic_shell_gamma driven by the frozen
// Wigner-Seitz image inventory as its directed record source, which is the
// same arithmetic (and the same Ewald code) the Gamma-periodic GFN2 driver
// runs with unit weights over its pair cutoff.  It replaced three separate
// hand-rolled Ewald implementations on 2026-09-06.
//
// `form` selects the short-range kernel.  ShellGammaForm::KlopmanOhno is the
// GFN2 paper's own, whose remainder decays as -eta^2/2R^3 and so has no
// three-dimensional thermodynamic limit (#444).  ShellGammaForm::Elstner is
// the Elstner 1998 form, whose remainder decays exponentially and does.
//
// `alpha <= 0` selects the validated per-dimension Ewald width (3-D
// sqrt(pi)/V^(1/3), 2-D 0.85 sqrt(pi/A), 1-D 4/L); any positive value is
// used as given, and the summed kernel is alpha-independent within the
// e^-30 series truncation.
//
// Test-visible: the eta->0 anchor test evaluates this kernel directly and
// compares its off-diagonal blocks to the bare-Coulomb Ewald kernel (the
// validated Madelung machinery).
Eigen::MatrixXd seccm_shell_gamma(
    const std::vector<GFN2ShellInfo>& shell_info,
    const std::vector<Eigen::Vector3d>& atom_coords,
    const WSTopology& topology,
    ShellGammaForm form = ShellGammaForm::KlopmanOhno,
    bool molecular_onsite = false,
    double alpha = 0.0);

// Test-visible local part of the physical-basin gate. Production combines
// this with element/shell population bounds; exposing the local predicate
// keeps the size-dilution regression tied to the actual native code.
bool gfn2_seccm_local_charge_state_is_physical(
    const Eigen::VectorXd& dq_atom,
    const Eigen::VectorXd& dq_shell);

GFN2SECCMResult run_gfn2_seccm(
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    const GFN2SECCMOptions& opts = {},
    double gap_tolerance = 1.0e-8);

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
