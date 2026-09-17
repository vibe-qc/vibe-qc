// GFN2-xTB energy driver.
//
// Implements the SCC loop for GFN2-xTB using the shared core
// infrastructure (basis, overlap, Hamiltonian builder, gamma
// builder, Mulliken charges).

#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/charge_mixer.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

// ---------------------------------------------------------------------------
// Utility: valence-electron count (used by all GFN2 drivers).

// GFN2's valence-electron count for element Z — i.e. the number of electrons
// that occupy the *valence* shells GFN2 actually parameterises, with the
// completed inner (n−1)d¹⁰ (and, throughout period 6, the 4f¹⁴) frozen in the
// core.  Reference: Bannwarth, Ehlert & Grimme, J. Chem. Theory Comput. 2019,
// 15, 1652, doi:10.1021/acs.jctc.8b01176.
//
// Why "Z − previous noble gas" is wrong past a completed d/f series: that rule
// counts the frozen (n−1)d¹⁰ / 4f¹⁴ as valence, which over-fills the small
// s/p(/d) valence basis.  Depending on how far the over-count exceeds the basis
// capacity it either silently corrupts the reference occupation (e.g. Kr filled
// to 4s²4p⁶3d¹⁰ instead of 4s²4p⁶) or blows up catastrophically — n_occ
// saturates the basis, the Mulliken charge collapses to a large spurious
// cation, and the band energy goes positive (closed-shell Pb → +530 Ha,
// Po → +6662 Ha, Os/Pt/Rn energies > 0).  This routine subtracts the frozen
// electrons so the count equals the reference occupation GFN2 loads.  Every
// rule below was checked against the per-element shell list in the parameter
// cache (~/.cache/vibeqc/gfn2_xtb_params.toml): the elements that drop a d/f
// subshell into the core load no shell for it.
//
// Atomic electron counts are separate from the shell reference occupations used
// by the shell-resolved SCC terms. Use gfn2_reference_occupation() for n0_shell:
// GFN2 intentionally uses hybridized references such as C s1 p3 and N s1.5 p3.5,
// not a greedy aufbau fill.
inline int gfn2_valence_electrons(int Z) {
    // Lanthanides Ce–Lu: GFN2 uses an f-in-core *trivalent* treatment.  The 4fⁿ
    // shell is entirely in the core and Ce–Lu are all parameterised with the
    // same {d,s,p} valence (en/zeta linearly interpolated across the row, a
    // constant Γ³ = −0.1 — verified in the param cache), so the valence count is
    // a flat 3, matching La's 5d¹6s².  (La itself, Z = 57, has no 4f electron;
    // the general rule below already returns 3 for it.)
    //
    // KNOWN EDGE CASE (even-Z lanthanides, isolated neutral atom): Ce (Z=58),
    // Nd (60), Sm (62), Gd (64), Dy (66), Er (68), Yb (70) are even-Z atoms
    // whose total electron count is even, but the f-in-core valence count is odd
    // (3).  An isolated neutral atom is an open-shell doublet; the closed-shell
    // driver (run_gfn2_xtb, multiplicity=1) silently truncates n_occ = 3/2 = 1,
    // leaving one valence electron unpaired → the energy is that of the +1
    // cation.  This is harmless in real molecules (where only the total electron
    // count matters and the molecule's overall charge + multiplicity controls
    // n_occ), but for a bare-atom energy benchmark, use the unrestricted driver
    // (run_ugfn2_xtb) with the correct multiplicity.
    if (Z >= 58 && Z <= 71) return 3;

    static const int nobles[] = {2, 10, 18, 36, 54, 86};
    int core = 0;
    for (int g : nobles) { if (g < Z) core = g; else break; }
    int v = Z - core;

    // Group 12 (Zn=30, Cd=48, Hg=80): the filled (n−1)d¹⁰ — and, for Hg, the
    // 4f¹⁴ — is frozen, and the parameter set loads only the ns,np shells (no
    // l = 2 shell), so GFN2's valence is just the ns² pair.  Left as the
    // noble-gas count (Zn,Cd → 12; Hg → 26) it forced ~12/26 electrons into the
    // 4-function s+p basis → a spurious +4 cation and closed-shell Zn ≈ +48.6 Ha.
    if (Z == 30 || Z == 48 || Z == 80) return 2;

    // Subtract the frozen subshell(s) for everything past a completed d/f series,
    // leaving the group-equivalent valence:
    //   period 4 p-block  Ga–Kr (31–36):  − 3d¹⁰          → 3 … 8
    //   period 5 p-block  In–Xe (49–54):  − 4d¹⁰          → 3 … 8
    //   period 6 d-block  Hf–Au (72–79):  − 4f¹⁴          → 4 … 11 (group no.)
    //   period 6 p-block  Tl–Rn (81–86):  − 4f¹⁴ − 5d¹⁰   → 3 … 8
    // The d-block of periods 4–5 (Sc–Cu, Y–Ag) and La need no subtraction: their
    // (n−1)d is the *valence* shell, not a frozen core, so Z − core is already
    // right (Sc → 3 … Cu → 11; La → 3).
    if ((Z >= 31 && Z <= 36) || (Z >= 49 && Z <= 54)) v -= 10;      // (n−1)d¹⁰
    else if (Z >= 72 && Z <= 79)                       v -= 14;      // 4f¹⁴
    else if (Z >= 81 && Z <= 86)                       v -= 10 + 14; // 5d¹⁰ + 4f¹⁴

    return v;
}

inline bool is_d_block(int Z) {
    return (Z >= 21 && Z <= 30) || (Z >= 39 && Z <= 48) || (Z >= 57 && Z <= 80);
}

inline double gfn2_reference_occupation(int Z, int l) {
    if (l < 0 || l > 2) return 0.0;

    auto sp = [l](double s, double p) {
        return l == 0 ? s : (l == 1 ? p : 0.0);
    };
    auto spd = [l](double s, double p, double d) {
        return l == 0 ? s : (l == 1 ? p : (l == 2 ? d : 0.0));
    };

    switch (Z) {
        case 1: case 3: case 11: case 19: case 37: case 55:
            return l == 0 ? 1.0 : 0.0;
        case 2: case 4: case 12: case 20: case 30: case 48: case 80:
            return l == 0 ? 2.0 : 0.0;
        case 5: case 13: case 31:
            return sp(2.0, 1.0);
        case 6:
            return sp(1.0, 3.0);
        case 7:
            return sp(1.5, 3.5);
        case 14: case 32:
            return sp(1.5, 2.5);
        case 15: case 33:
            return sp(1.5, 3.5);
        case 8: case 9: case 10:
            return sp(2.0, static_cast<double>(Z - 4));
        case 16: case 17: case 18:
            return sp(2.0, static_cast<double>(Z - 12));
        case 34: case 35: case 36:
            return sp(2.0, static_cast<double>(Z - 30));
        case 49: case 50: case 51: case 52: case 53: case 54:
            return sp(2.0, static_cast<double>(Z - 48));
        case 81: case 82: case 83: case 84: case 85: case 86:
            return sp(2.0, static_cast<double>(Z - 80));
        case 29: case 47: case 79:
            return spd(1.0, 0.0, 10.0);
    }

    if (Z >= 21 && Z <= 28) return spd(1.0, 1.0, static_cast<double>(Z - 20));
    if (Z >= 39 && Z <= 46) return spd(1.0, 1.0, static_cast<double>(Z - 38));
    if (Z >= 57 && Z <= 71) return spd(1.0, 1.0, 1.0);
    if (Z >= 72 && Z <= 78) return spd(1.0, 1.0, static_cast<double>(Z - 70));

    return 0.0;
}

// SCC options (shared with DFTB for now)
// ---------------------------------------------------------------------------

// Default electronic temperature (k_B T in Hartree, about 316 K) applied by
// the periodic GFN2-xTB drivers when the caller leaves
// ``electronic_temperature`` unset. Default-on frontier smearing keeps the
// SCC on one occupation branch when the Gamma frontier of a metallic cell
// crosses on a lattice sweep; the Bannwarth-Ehlert-Grimme 2019 Fermi term
// A = E - T*S is then the reported Mermin free energy. The molecular driver
// ignores this constant: its primary SCC attempt runs at the requested
// temperature (exact zero-temperature Aufbau by default). Only its
// auto-stabilisation ladder may retry at a finite temperature, and only when
// no temperature was requested explicitly (vibe-qc#3).
inline constexpr double kPeriodicGFN2DefaultElectronicTemperature = 0.001;

struct XTBSccOptions {
    // Total SCC iteration budget across the primary solve plus any automatic
    // stabilization retry. The historical hidden molecular retry ladder used
    // up to 3600 iterations; keep that robustness as an explicit default.
    int max_iter = 3600;
    double conv_tol_charge = 1e-6;
    double charge_mixing = 0.1;      // damping for Simple mixer
    // The default Simple path in the molecular driver runs the two-phase
    // polyalgorithm (damped simple mixing with stall-adaptive step halving
    // plus a guarded DIIS handoff) when the third-order term is active;
    // see the mixer block in gfn2_driver.cpp.  An explicit DIIS/Broyden
    // request bypasses the polyalgorithm and runs the unguarded
    // accelerator from the neutral guess. Newton is implemented only by the
    // nontrivial GFN2-SECCM engine and run_gfn2_xtb rejects it explicitly.
    SCCMixer scc_mixer = SCCMixer::Simple;
    int mixer_memory = 6;            // DIIS subspace / Broyden history
    double mixer_damping = 0.0;      // DIIS damping / Broyden step damping
    double electronic_temperature = 0.0;  // finite-T occupations (Ha), 0 = Aufbau

    // True when ``electronic_temperature`` was set explicitly (the pybind
    // property setter marks it on any assignment, including 0.0). The
    // periodic GFN2-xTB drivers smear the frontier by
    // ``kPeriodicGFN2DefaultElectronicTemperature`` when this flag is false,
    // and honour ``electronic_temperature`` exactly (0 = exact Aufbau) when it
    // is true. The molecular driver's primary attempt always uses the value as
    // given; its auto-stabilisation ladder may fall back to a finite-
    // temperature rung only when this flag is false, and honours an explicit
    // value on every rung (vibe-qc#3). The executed temperature is reported in
    // GFN2Result::smearing_temperature.
    bool electronic_temperature_explicit = false;

    bool auto_stabilize = true;      // retry hard Γ³ SCC failures with safer settings

    // Functional form of the isotropic second-order kernel
    // (core/periodic_gamma.hpp).  Unset: the molecular driver keeps the
    // published Klopman-Ohno kernel and the periodic drivers use the
    // Elstner 1998 form (kPeriodicGFN2DefaultGammaForm); an explicit value
    // is honoured by every driver, which is what makes the SECCM molecular
    // limit and a molecular run agree under one form.  Staged per the
    // maintainer's D1 decision (2026-08-28).
    ShellGammaForm gamma_form = ShellGammaForm::KlopmanOhno;
    bool gamma_form_explicit = false;

    // EXPERIMENTAL: faithful atom-resolved AES (Bannwarth, Ehlert & Grimme,
    // JCTC 2019).  When false (default) the shipped ad-hoc shell-resolved
    // gamma^n multipole path runs and the result is byte-identical to before
    // this flag existed.  When true, the on-site DPOL/QPOL XC kernel + the
    // CN-damped (fdmp3/fdmp5) inter-site multipoles replace it.  Gated off by
    // default until validated against the live-xtb decomposition oracle; see
    // handovers/HANDOVER_GFN2_AES.md.
    bool aes_faithful = false;
    // AES potential damping (faithful path): fraction of the NEW atom
    // charge/dipole/quadrupole potentials mixed into the Fock each SCC
    // iteration (1.0 = undamped). The damping changes only the iteration
    // path; the converged fixed point is unchanged. The on-site DPOL/QPOL
    // feedback is stiff on polarizable heterocycles (adenine two-cycles
    // under simple mixing and diverges under Broyden without it); see
    // handovers/HANDOVER_GFN2_AES.md.
    double aes_damping = 0.5;
};

// ---------------------------------------------------------------------------
// GFN2-xTB result
// ---------------------------------------------------------------------------

// One bounded SCC attempt.  The molecular GFN2 driver owns the automatic
// stabilization ladder used directly and by the exact molecular SECCM
// delegation path, so the attempt ledger lives on the shared native result
// rather than being reconstructed by an adapter after the fact.
struct GFN2SCCAttempt {
    // Canonical executed solver path: molecular_simple,
    // molecular_hybrid, molecular_diis, molecular_broyden, or the matching
    // supercell_* mixer path.  This distinguishes the molecular GAM3
    // simple/guarded-DIIS polyalgorithm from plain supercell Simple mixing.
    std::string solver;
    int allocated_max_iter = 0;
    int n_iter = 0;
    // The ladder rung's configured charge-mixing control.  The executed
    // solver name records when a polyalgorithm or accelerator interprets it.
    double ladder_charge_mixing = 0.0;
    double electronic_temperature = 0.0;
    SCCMixer scc_mixer = SCCMixer::Simple;
    bool restart_supplied = false;
    bool molecular_delegated = false;
    // Closed values: "converged", "iteration_limit", "stalled_checkpoint",
    // "unphysical_basin", and "gap_rejected".  Strings keep this internal
    // diagnostic record JSON-ready at the Python boundary without exposing
    // another public enum.  "stalled_checkpoint" is the SECCM primary attempt
    // handed to the stabilization ladder by the budget-independent checkpoint
    // (#294); like "iteration_limit" it never reached the residual tolerance.
    std::string exit_reason = "iteration_limit";
    // True when the SCC residual reached tolerance, including a fixed point
    // later rejected by the finite-torus gap or physical-basin gates.
    bool scc_converged = false;
    bool physical_basin = true;
    Eigen::VectorXd max_change_trace;
};

struct GFN2Result : ParameterIdentifiedResult {
    // Internal total E. Molecular GFN2 historically exposes this value as
    // energy; finite-temperature callers that need the variational potential
    // use free_energy = E - T*S below.
    double energy = 0.0;
    double free_energy = 0.0;
    double entropy = 0.0;
    double smearing_temperature = 0.0;
    double e_electronic = 0.0;
    double e_repulsive = 0.0;
    double e_scc = 0.0;
    double e_band0 = 0.0;       // Σ P·H⁰ (bare H0 band energy)
    double e_aes = 0.0;         // anisotropic 2nd-order (multipole / AES)
    double e_3rd = 0.0;         // 3rd-order on-site Σ (Γ/3)·q³
    double vmp_rms = 0.0;       // RMS of shell AES potential V_mp
    double shell_dip_rms = 0.0; // RMS of shell dipole moments, debug
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd hamiltonian;
    Eigen::VectorXd charges;
    int n_basis = 0;
    int n_occ = 0;
    int n_iter = 0;
    bool converged = false;
    // Concatenation of every attempt trace, in execution order.  The closure
    // invariant is n_iter == trace.size() == sum(attempt.n_iter).
    Eigen::VectorXd scc_max_change_trace;
    std::vector<GFN2SCCAttempt> attempts;
    // Index of the accepted attempt, or -1 when the whole bounded ladder
    // failed.  Successful single-attempt runs select index zero.
    int selected_attempt_index = -1;
};

// ---------------------------------------------------------------------------
// Run GFN2-xTB SCC calculation
// ---------------------------------------------------------------------------

GFN2Result run_gfn2_xtb(
    const Molecule& mol,
    const GFN2ParameterSet& params,
    const XTBSccOptions& opts = {});

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
