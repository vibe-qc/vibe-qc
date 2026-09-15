// Initial-guess methods for the SCF density matrix.
//
// ``GuessEngine`` is the molecular dispatch point for RHF / UHF / RKS / UKS
// and the low-level density seam used by periodic wrappers. Guesses that need
// a periodic lattice context are assembled by the shared Python helpers; a
// multi-k Fock-mode guess stays in the route driver so that its route-local
// occupation machinery remains authoritative.
//
// Why an engine and not a chain of ``if`` branches: the literature
// classification of guesses (Lehtola 2019; ORCA 6 manual §3.2;
// Q-Chem 5 manual §6.3) groups them by *what they produce*:
//
//   * Density-mode  — SAD, PATOM         — yields ρ(r) directly.
//   * Fock-mode     — HCORE, SAP, HUECKEL — yields a Fock that is
//                                          diagonalised into orbitals.
//   * MO-mode       — MINAO, READ         — yields orbital coefficients.
//
// All three modes plug into the same SCF iteration body the same way:
// the driver consumes whichever output the engine produced and starts
// the loop. AUTO inspects the system and picks per the table in
// docs/roadmap.md §G2e.
//
// A method unavailable on a particular driver throws ``std::runtime_error``
// rather than silently falling back to another guess.

#pragma once

#include <Eigen/Dense>
#include <map>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>

#include "basis.hpp"
#include "ecp.hpp"
#include "molecule.hpp"

namespace vibeqc {

// The atomic guess uses the same explicitly selected ECP operator as SCF.
// Keep chemical Z separate from its effective nuclear charge; neither a
// valence electron count nor a basis name identifies an ECP Hamiltonian.
struct GuessECPContext {
    std::vector<ECPCenter> xml_centers;
    std::string xml_library;
    std::vector<ECPPrimitiveBlock> primitive_blocks;
    std::vector<std::array<double, 3>> primitive_centers;
    std::vector<double> effective_charges;
    int total_ncore = 0;
    bool active() const noexcept {
        return !xml_centers.empty() || !primitive_blocks.empty();
    }
};

template<class Options>
GuessECPContext molecular_guess_ecp_context(const Options& opts) {
    return {opts.ecp_centers, opts.ecp_library, opts.ecp_primitive_blocks,
            opts.ecp_primitive_centers, opts.ecp_effective_charges,
            opts.ecp_total_ncore};
}

template<class Options>
GuessECPContext periodic_guess_ecp_context(const Options& opts,
                                         const Molecule* mol = nullptr) {
    GuessECPContext context{{}, {}, opts.ecp_primitive_blocks, opts.ecp_home_centers,
                            opts.ecp_effective_charges, opts.ecp_total_ncore};
    // POB all-electron records carry redundant physical charges. Accept only
    // exact atom-by-atom identity with no operator, centers or removed core;
    // malformed and reduced-charge records still reach the ECP validator.
    if (mol && context.primitive_blocks.empty() && context.primitive_centers.empty()
            && context.total_ncore == 0
            && context.effective_charges.size() == mol->atoms().size()) {
        bool all_electron = true;
        for (std::size_t a = 0; a < mol->atoms().size(); ++a)
            all_electron = all_electron && context.effective_charges[a] == mol->atoms()[a].Z;
        if (all_electron) context.effective_charges.clear();
    }
    return context;
}

enum class InitialGuess;
void validate_guess_ecp(InitialGuess kind, const GuessECPContext& ecp,
                        const Molecule* mol = nullptr);

class JKBuilder;  // forward decl — see jk_builder.hpp

// Density-mode guesses need not be idempotent, but must carry the requested
// population in the SCF metric. For a home-cell density reused at every k,
// metric is sum_k w_k S(k). Projection precedes normalization.
Eigen::MatrixXd normalize_guess_density(
    const Eigen::MatrixXd& density, const Eigen::MatrixXd& metric,
    double electrons);
Eigen::MatrixXcd normalize_guess_density(
    const Eigen::MatrixXcd& density, const Eigen::MatrixXcd& metric,
    double electrons);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> normalize_guess_spin_densities(
    const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta,
    const Eigen::MatrixXd& metric, int n_alpha, int n_beta);
std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> normalize_guess_spin_densities(
    const Eigen::MatrixXcd& alpha, const Eigen::MatrixXcd& beta,
    const Eigen::MatrixXcd& metric, int n_alpha, int n_beta);
// ATOMSPIN is a local sign request, not just a channel-shape preference.
// Reject population normalization that would erase or reverse its pattern.
void validate_guess_atomic_spins(
    const Molecule& mol, const BasisSet& basis,
    const Eigen::MatrixXcd& alpha, const Eigen::MatrixXcd& beta,
    const Eigen::MatrixXcd& metric, const std::vector<int>& atomic_spins);

std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> normalize_atomic_guess(
    const Molecule& mol, const BasisSet& basis,
    const Eigen::MatrixXcd& alpha, const Eigen::MatrixXcd& beta,
    const Eigen::MatrixXcd& metric, int n_alpha, int n_beta,
    const std::vector<int>& atomic_spins);

struct PeriodicSystem;     // forward decl — see periodic.hpp
struct LatticeMatrixSet;   // forward decl — see lattice_sum.hpp
struct LatticeSumOptions;  // forward decl — see lattice_sum.hpp

// One element's SAP (superposition-of-atomic-potentials) expansion, read from
// a SAP ``.g94`` table. The neutral-atom effective potential is fit as a sum
// of erf-screened Coulomb terms,
//
//   V_atom(r) = −Σ_i coeffs[i] · erf(√alphas[i] · r) / r .
//
// Exposed so the periodic lattice-summed SAP guess (``compute_vsap_lattice``)
// can reuse the same fitted data the molecular SAP guess uses, instead of
// re-parsing the file. ``sap_expansions`` returns the per-element table for
// the named SAP basis (e.g. "sap_helfem_large"), parsed once and cached.
struct SAPExpansion {
    std::vector<double> alphas;
    std::vector<double> coeffs;
};

const std::map<int, SAPExpansion>& sap_expansions(
    const std::string& sap_basis_name);

// Molecular SAP potential matrix V_SAP_μν = ⟨χ_μ | v^SAP | χ_ν⟩ (analytical,
// via libint). Exposed for parity-checking the periodic lattice-summed
// ``compute_vsap_lattice`` against the molecular reference.
Eigen::MatrixXd compute_sap_potential_molecular(
    const BasisSet& basis, const Molecule& mol,
    const std::string& sap_basis_name);

// Numerical LDA-on-valence-HF atomic potentials with the actual ECP operator.
Eigen::MatrixXd compute_vsap_ecp(const Molecule& mol, const BasisSet& basis,
    const GuessECPContext& ecp);
LatticeMatrixSet compute_vsap_ecp_lattice(const BasisSet& basis,
    const PeriodicSystem& system, const LatticeSumOptions& opts,
    const GuessECPContext& ecp);

// Periodic MINAO initial-guess density. Projects the ANO-RCC minimal-basis
// reference density onto the working basis using lattice-summed overlaps:
//
//   P = S(Γ)^{-1} S_tm(Γ) ,   D = P · D_ref · Pᵀ ,
//
// where S(Γ) is the Γ-folded target overlap (passed in by the driver, which
// already has it) and S_tm(Γ) = Σ_g ⟨χ_target(0) | χ_ref(g)⟩ is the
// lattice-summed cross-basis overlap to the reference minimal basis. This is
// the periodic analogue of the molecular MINAO guess; it is wired into the
// closed-shell Γ and multi-k RHF/RKS periodic drivers. D is rescaled so that
// tr(D · S(Γ)) = n_elec exactly (the projection is not norm-conserving when
// the target basis does not fully span the reference).
Eigen::MatrixXd compute_minao_density_periodic(
    const Molecule& mol, const BasisSet& basis,
    const PeriodicSystem& system, const Eigen::MatrixXd& S_gamma,
    int n_elec, const LatticeSumOptions& lattice_opts,
    const GuessECPContext& ecp = {});

// Periodic HUECKEL / GWH Fock-mode initial guess. Builds the lattice Fock
// blocks directly from the overlap lattice and per-AO atomic orbital energies:
//
//   H_GWH(g)_{uv} = 0.5 K S(g)_{uv} (eps_u + eps_v),   K = 1.75,
//
// with the home-cell diagonal set to eps_u. The closed-shell periodic Γ and
// multi-k RHF/RKS drivers diagonalise this Fock against S(k) to obtain the
// starting density. Open-shell and Python periodic drivers remain separately
// gated until they can inject per-spin / per-driver Fock-mode guesses.
LatticeMatrixSet compute_huckel_fock_lattice(
    const Molecule& mol, const BasisSet& basis,
    const LatticeMatrixSet& S_set, const GuessECPContext& ecp = {});

enum class InitialGuess {
    AUTO,     // pick per system type (see GuessEngine::resolve_auto)
    HCORE,    // diagonalise T + V_ne
    SAD,      // superposition of atomic densities (Van Lenthe 2006)
    SAP,      // superposition of atomic potentials (Lehtola 2019, JCTC 15, 1593)
    PATOM,    // SAD + in-field re-polarisation (ORCA PAtom)
    HUECKEL,  // extended Hückel minimal-basis Fock, projected to target
    MINAO,    // SAD projected onto a minimal-AO subset (Sun 2017)
    READ,     // restart — read MOs from a prior calculation
    FRAGMO,   // superposition of converged fragment densities (molecular or periodic;
              // assembled in the Python runner — see guess_fragmo.py)
};

class GuessCapabilityError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Physical selection is distinct from density transport on a restart/retry.
struct GuessSelection {
    InitialGuess requested = InitialGuess::AUTO;
    InitialGuess effective = InitialGuess::HCORE;
    InitialGuess transport = InitialGuess::HCORE;
};

// All native entry points validate even enum values constructed by a foreign
// binding or a cast. READ payload precedence never exempts the selector.
void validate_initial_guess(InitialGuess kind);
void validate_atomic_spin_selection(
    const Molecule* mol, InitialGuess effective, const std::vector<int>& tags);

// SPINLOCK: how the spin is held during early SCF for broken-symmetry
// magnetic convergence. OFF = normal aufbau. SPIN_SCHEDULE = converge at a
// locked n_alpha-n_beta for the first N cycles, then restart at the
// multiplicity target (CRYSTAL SPINLOCK n nstep; a two-phase SCF).
// PATTERN_HOLD = hold the seeded broken-symmetry occupied set by maximum
// overlap (MOM) for the first N cycles, then release (protects an atomic_spins
// seed on hard systems; see vibeqc/mom.hpp).
enum class SpinlockMode { OFF, SPIN_SCHEDULE, PATTERN_HOLD };

// Method-specific overrides. Most fields are ignored unless the matching
// ``kind`` is selected.
struct GuessRequest {
    InitialGuess kind = InitialGuess::AUTO;
    std::string read_path = "";          // READ
    double guessmix_angle_deg = 0.0;     // applies on top of any spin-aware guess
    int patom_iterations = 1;            // PATOM
};

// Hints used by the AUTO policy. The molecular SCF drivers fill these
// from ``Molecule``; the periodic drivers add ``is_periodic`` and an
// optional band-gap estimate. Adding hints does not break existing
// callers — defaults match the molecular closed-shell light-atom case.
struct SystemHints {
    bool is_periodic = false;
    bool has_transition_metal = false;
    bool is_open_shell = false;
    std::optional<double> band_gap_hartree;   // periodic only; if set, < 0.05 → metallic
};

// Closed-shell result for the context-free molecular engine. Implemented
// guesses currently return D after any required one-particle diagonalisation;
// an empty result means the driver retains its Hcore path. F_guess/C_guess are
// reserved for a future caller-owned artifact path. Periodic wrappers expose
// their lattice Fock through python/vibeqc/guess.py instead.
struct GuessClosedShellResult {
    InitialGuess resolved_kind = InitialGuess::HCORE;
    std::string provenance;
    Eigen::MatrixXd D;          // constructed starting density
    Eigen::MatrixXd F_guess;    // reserved Fock-mode artifact
    Eigen::MatrixXd C_guess;    // reserved MO-mode artifact
    Eigen::VectorXd eps_guess;
};

// Open-shell result. Implemented guesses currently return per-spin densities;
// empty means the driver retains its Hcore path. Fock/MO fields are reserved.
struct GuessOpenShellResult {
    InitialGuess resolved_kind = InitialGuess::HCORE;
    std::string provenance;
    Eigen::MatrixXd D_alpha, D_beta;                  // density-mode
    Eigen::MatrixXd F_guess_alpha, F_guess_beta;      // Fock-mode
    Eigen::MatrixXd C_guess_alpha, C_guess_beta;      // MO-mode
    Eigen::VectorXd eps_guess_alpha, eps_guess_beta;
};

struct PreparedClosedGuess {
    GuessSelection selection;
    Eigen::MatrixXd density;
};
struct PreparedOpenGuess {
    GuessSelection selection;
    Eigen::MatrixXd alpha, beta;
};

// Construction boundary shared by the molecular driver and its external-JK
// entry point. Explicit density transport and prepared construction metadata
// are separate arguments so nested adapters cannot relabel a READ retry.
PreparedClosedGuess prepare_closed_guess(
    const Molecule* mol, const BasisSet& basis, int n_occ, InitialGuess kind,
    const Eigen::MatrixXd& overlap, const Eigen::MatrixXd& hcore,
    const JKBuilder& jk, const Eigen::MatrixXd& initial_density,
    const Eigen::MatrixXd& read_density, double linear_dep_threshold,
    const GuessSelection* prepared = nullptr,
    const GuessECPContext& ecp = {});
PreparedOpenGuess prepare_open_guess(
    const Molecule* mol, const BasisSet& basis, int n_alpha, int n_beta,
    InitialGuess kind, const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& hcore, const JKBuilder& jk,
    const Eigen::MatrixXd& initial_alpha, const Eigen::MatrixXd& initial_beta,
    const Eigen::MatrixXd& read_alpha, const Eigen::MatrixXd& read_beta,
    const std::vector<int>& atomic_spins, double linear_dep_threshold,
    bool refine_sad, const GuessSelection* prepared = nullptr,
    const GuessECPContext& ecp = {});

class GuessEngine {
public:
    // ---- Public entry points used by SCF drivers ------------------------

    // Closed-shell (RHF, RKS, periodic-RHF/RKS Γ).
    //
    // ``S``, ``Hcore``, ``jk`` are optional inputs the engine uses
    // when the chosen guess needs them (molecular SAP needs S for the
    // F_SAP = T + V_SAP diagonalisation; PATOM needs S, Hcore, and a
    // JKBuilder for the in-field step). Pass
    // nullptr when the caller doesn't have them — the engine falls back to a
    // method that doesn't require them, or throws with a clear message.
    static GuessClosedShellResult build_closed_shell(
        const Molecule& mol, const BasisSet& basis,
        int n_occ,
        InitialGuess kind,
        const Eigen::MatrixXd* S = nullptr,
        const Eigen::MatrixXd* Hcore = nullptr,
        const JKBuilder* jk = nullptr,
        const SystemHints& hints = {},
        double linear_dep_threshold = 1e-7,
        const GuessECPContext& ecp = {});

    // Open-shell (UHF, UKS, periodic-UHF/UKS).
    //
    // The presence of ``jk`` enables the UKS-style SAD refinement through
    // F_SAD = Hcore + J(D_SAD) − ½K(D_SAD). Molecular SAP instead
    // diagonalises its spin-independent F_SAP once and fills the first
    // n_α / n_β orbitals, giving exact requested spin populations.
    static GuessOpenShellResult build_open_shell(
        const Molecule& mol, const BasisSet& basis,
        int n_alpha, int n_beta,
        InitialGuess kind,
        const Eigen::MatrixXd* S = nullptr,
        const Eigen::MatrixXd* Hcore = nullptr,
        const JKBuilder* jk = nullptr,
        const SystemHints& hints = {},
        double linear_dep_threshold = 1e-7,
        // ATOMSPIN: per-atom spin seed (+1/-1/0, in atom order) for a
        // broken-symmetry SAD start. nullptr/empty = spin-symmetric split.
        // Consulted only when the resolved guess is SAD.
        const std::vector<int>* atomic_spins = nullptr,
        const GuessECPContext& ecp = {});

    // Resolve AUTO to a concrete kind given system hints. Exposed for
    // tests and for the SCF-log provenance line — drivers don't need
    // to call this directly. Non-AUTO inputs pass through unchanged.
    static InitialGuess resolve_auto(InitialGuess kind, const SystemHints& hints);

    // Resolve AUTO using molecule-derived transition/f-block and open-shell
    // hints, including the isolated open-shell atom -> PATOM override. This is
    // the single policy entry point used by production molecular drivers and
    // Python routing/provenance code, so execution and citations cannot drift.
    static InitialGuess resolve_auto_for_molecule(
        const Molecule& mol,
        InitialGuess kind,
        bool is_periodic = false,
        bool is_open_shell = false,
        const std::optional<std::vector<InitialGuess>>& supported = std::nullopt);
};

// ---- Legacy free function ------------------------------------------------
//
// Kept for back-compat with code that built D_SAD outside the engine
// (notably the existing periodic-Python drivers via the pybind shim).
// New code should prefer the engine.
Eigen::MatrixXd sad_density(const Molecule& mol, const BasisSet& basis,
                            const GuessECPContext& ecp = {});

}  // namespace vibeqc
