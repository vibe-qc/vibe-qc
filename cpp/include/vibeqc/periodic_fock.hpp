// Periodic Fock-matrix building blocks — Γ-only, molecular-limit regime.
//
// Formalism. For closed-shell RHF with total density
// P_λσ (trace(P·S) = n_electrons, P = 2·Σ_occ C C†), the real-space AO-basis
// Fock matrix is
//
//   F_μν(g) = H_μν(g) + Σ_{λσ, g_λ, g_σ}
//                        P_λσ(g_σ − g_λ) [ (μ0 νg | λg_λ σg_σ)
//                                          − ½ (μ0 λg_λ | νg σg_σ) ]
//
// In the Γ-only molecular-limit regime (P(h) = P_Γ · δ_{h, 0}) the inner
// lattice sum collapses onto g_σ = g_λ:
//
//   F_μν(g) = H_μν(g) + Σ_{λσ, g_λ}
//                        P_λσ [ (μ0 νg | λg_λ σg_λ) − ½ (μ0 λg_λ | νg σg_λ) ]
//
// and the Bloch sum at Γ reduces to a molecular-shaped Fock matrix:
//
//   F_μν(Γ) = H_μν(Γ) + J_μν − ½ K_μν   with
//
//   J_μν = Σ_{λσ, g, g_λ} P_λσ · (μ0 νg | λg_λ σg_λ)
//   K_μν = Σ_{λσ, g, g_λ} P_λσ · (μ0 λg_λ | νg σg_λ)
//
// This file builds J and K by direct SCF — looping over cell pairs (g, g_λ)
// and AO shell quartets, letting libint emit the raw (μ, ν{g} | λ{g_λ}, σ{g_λ})
// blocks. No 4-index tensor is stored in memory; the pair loop is the only
// scalable scaffolding we'll carry through to 12c / 12e (where screening
// and Ewald splitting slot into the same loop).
//
// Γ-only molecular-limit is adequate for validation (box big enough that
// P(g ≠ 0) ≈ 0); full multi-k builds with real-space P(g) arrive in 12c.

#pragma once

#include <Eigen/Dense>

#include <utility>
#include <cstdint>
#include <vector>

#include "basis.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

struct JKMatrices {
    Eigen::MatrixXd J;   // Σ_{λσ, g, g_λ} P_λσ (μ0 νg | λg_λ σg_λ)
    Eigen::MatrixXd K;   // Σ_{λσ, g, g_λ} P_λσ (μ0 λg_λ | νg σg_λ)
};

// Direct-SCF J/K build in the Γ-only molecular-limit regime.
//
// ``P_gamma`` is the closed-shell density at Γ, convention
// P = 2 Σ_{i ∈ occ} C_i C_i† (trace(P·S) = n_electrons).
//
// The two lattice indices are capped independently by ``opts.cutoff_bohr``.
// The outer sum (g over ν-shift cells) and the inner sum (g_λ over density-
// image cells) share the same cell list in the Γ-only regime — callers do
// not need to distinguish them.
//
// ``omega`` controls the Coulomb kernel used for the ERIs:
//   omega == 0 (default) → full 1/r_12 Coulomb via libint's Operator::coulomb.
//   omega  > 0            → erfc(ω·r_12) / r_12 via Operator::erfc_coulomb.
// The erfc-screened variant is the short-range piece of the Ewald split
// for 3D bulk; the matching long-range erf piece is evaluated via the
// reciprocal-space / grid machinery in Phase 12e-c-3 and combined with
// this function's output.
JKMatrices build_jk_gamma_molecular_limit(const BasisSet& basis,
                                          const PeriodicSystem& system,
                                          const LatticeSumOptions& opts,
                                          const Eigen::MatrixXd& P_gamma,
                                          double omega = 0.0);

// Phase SYM3b — explicit cell list variant.  Same kernel, but accepts a
// caller-supplied cell list instead of computing one internally.
JKMatrices build_jk_gamma_molecular_limit_explicit(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const std::vector<LatticeCell>& cells,
        const LatticeSumOptions& opts,
        const Eigen::MatrixXd& P_gamma,
        double omega = 0.0);

// ---------------------------------------------------------------------------
// Phase M3b — per-cell-pair J/K contributions for symmetry-reduced Fock build.
//
// build_jk_gamma_molecular_limit_explicit sums every (c_g, c_p) cell pair into
// a single J and K. The symmetry-reduction layer needs the contributions
// *un-summed* — one J/K block per cell pair — so it can evaluate only a set of
// orbit representatives and scatter the rest via Wigner-D. This kernel runs the
// identical Γ-only molecular-limit shell-quartet loop but stores each pair's
// contribution separately instead of accumulating.
//
// This is additive: it does not modify or replace the existing J/K builders.
// It is called explicitly by the Python symmetry-orchestration layer.
struct PairJKContribution {
    int c_g = -1;                // index into the supplied cell list
    int c_p = -1;                // index into the supplied cell list
    Eigen::MatrixXd J_contrib;   // (nbf, nbf) J contribution from this pair
    Eigen::MatrixXd K_contrib;   // (nbf, nbf) K contribution from this pair
};

// Evaluate J and K contributions for each (c_g, c_p) pair in ``pairs``.
//
// ``cells`` is the full cell list; ``pairs[i] = {c_g, c_p}`` indexes into it.
// ``P_gamma`` is the Γ-only density matrix (nbf, nbf). ``omega`` selects the
// Coulomb kernel (0 = full 1/r, > 0 = erfc-screened short-range Ewald piece),
// same convention as build_jk_gamma_molecular_limit.
//
// The returned vector is aligned with ``pairs``: entry ``i`` carries the
// J/K blocks for ``pairs[i]``. Summing J_contrib (resp. K_contrib) over the
// full n_c × n_c pair set reproduces build_jk_gamma_molecular_limit_explicit's
// J (resp. K) exactly.
std::vector<PairJKContribution> build_jk_pair_contributions(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const std::vector<LatticeCell>& cells,
        const std::vector<std::pair<int, int>>& pairs,
        const LatticeSumOptions& opts,
        const Eigen::MatrixXd& P_gamma,
        double omega = 0.0);

// ---------------------------------------------------------------------------
// General multi-k real-space Fock build.
//
//   F^{2e}_μν(g) = Σ_{λσ, g_λ, g_σ}
//                    P_λσ(g_σ − g_λ) [ (μ0 νg | λg_λ σg_σ)
//                                      − ½ (μ0 λg_λ | νg σg_σ) ]
//
// Takes the full real-space density as a LatticeMatrixSet P(h) and returns
// F^{2e}(g) as a LatticeMatrixSet aligned with ``direct_lattice_cells(system,
// opts.cutoff_bohr)``. No ERI tensor is materialised — each (g, g_λ, g_σ)
// triple is emitted by libint on demand, so memory stays at O(nbf² × N_cells).
//
// The density-pair index ``h = g_σ − g_λ`` is looked up in ``P_real_space``
// using index arithmetic; if ``h`` falls outside the density cutoff, the
// contribution is taken as zero.
//
// Scope. Works for any dim ∈ {1, 2, 3}. Note that 3D bulk calculations with
// this direct-truncated lattice sum are cutoff-dependent (the Coulomb sum
// is conditionally convergent); proper Ewald splitting lives in Phase 12e.
// 1D and 2D insulating systems converge in practice with moderate cutoffs.
//
// ``exchange_scale`` is the prefactor on the K contribution:
//   F^{2e}(g) = J(g) − ½ · exchange_scale · K(g)
//
//   exchange_scale = 1.0  → RHF (default).
//   exchange_scale = 0.0  → pure DFT (K term skipped entirely, saving one
//                           libint call per shell quartet).
//   exchange_scale = α    → hybrid DFT with HF-exchange fraction α.
//
// ``omega`` selects the Coulomb kernel, same meaning as in
// ``build_jk_gamma_molecular_limit`` above: 0 (default) uses full 1/r_12,
// positive values use erfc(ω·r_12)/r_12 (the short-range Ewald piece).
LatticeMatrixSet build_fock_2e_real_space(const BasisSet& basis,
                                          const PeriodicSystem& system,
                                          const LatticeSumOptions& opts,
                                          const LatticeMatrixSet& P_real_space,
                                          double exchange_scale = 1.0,
                                          double omega = 0.0);

struct JKLatticeMatrixSets {
    LatticeMatrixSet J;  // direct Coulomb blocks J(g)
    LatticeMatrixSet K;  // full exchange blocks K(g), no -1/2 prefactor
    // Optional Gamma-density derivatives of the finite-domain bilinears
    //   E_J = 1/2 sum_g P : J_g[P]
    //   E_K = 1/2 sum_g P : K_g[P].
    // Empty for the ordinary builders; populated by the dedicated
    // build_jk_2e_real_space_domains_gamma_derivative entry point below.
    Eigen::MatrixXd J_gamma_energy_derivative;
    Eigen::MatrixXd K_gamma_energy_derivative;
    // Optional derivatives of the same bilinears with respect to every
    // independent real-space density block P(h). Empty for the ordinary and
    // Gamma-only builders; populated by the dedicated density-derivative
    // entry point below. The cell lists match P_real_space exactly.
    LatticeMatrixSet J_density_energy_derivative;
    LatticeMatrixSet K_density_energy_derivative;
    // Work counters, independent of timing and OpenMP scheduling (#21).
    std::uint64_t cell_triples_considered = 0;
    std::uint64_t cell_triples_possible = 0;
    std::uint64_t shell_quartets_considered = 0;
};

// Same cell-triple traversal as build_fock_2e_real_space, but returns
// J(g) and the unscaled K(g) separately. This avoids two full direct
// builds in Python callers that need both components, e.g. the BIPOLE
// Ewald-J split path.
JKLatticeMatrixSets build_jk_2e_real_space(const BasisSet& basis,
                                           const PeriodicSystem& system,
                                           const LatticeSumOptions& opts,
                                           const LatticeMatrixSet& P_real_space,
                                           double omega = 0.0);

// Phase SYM3b — explicit cell list variant.
JKLatticeMatrixSets build_jk_2e_real_space_explicit(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        double omega = 0.0);

// Phase SYM3b — point-group output-subset variant: builds J/K only for the
// `output_indices` bra cells (orbit representatives), full internal sum;
// non-emitted blocks left zero for caller-side reconstruction.
JKLatticeMatrixSets build_jk_2e_real_space_output_subset(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<int>& output_indices,
        double omega = 0.0);

// Phase SYM3b shell-pair mask: build only the atom-pair-orbit representative
// sub-blocks within each output (rep) cell. `output_shell_masks` is parallel
// to `output_indices`; each entry is an nshells*nshells row-major flag array
// (build the (s1_home, s2_cell) output shell pair iff non-zero).
JKLatticeMatrixSets build_jk_2e_real_space_output_subset_masked(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<int>& output_indices,
        const std::vector<std::vector<uint8_t>>& output_shell_masks,
        double omega = 0.0);

// ---------------------------------------------------------------------------
// Exact-ERI skip surface for the dormant PDR 1988 Ch. II.4c quartet prototype.
//
// Same cell-triple traversal as build_jk_2e_real_space, but accepts a
// per-cell-pair quartet-level skip mask: for each (c_g, c_p) cell pair,
// ``bipolar_skip_mask`` carries shell-quartet indices (s1, s2, s3, s4)
// that the prototype classifier assigns to its far path and that are SKIPPED by the
// exact ERI evaluation.  The caller computes those contributions
// separately via the Python multipole far-field contractor and adds them
// to the returned J/K blocks.
//
// ``bipolar_skip_mask`` is a vector parallel to the (c_g, c_p) pairs in
// row-major order (c_g * n_cells + c_p).  Each entry is a vector of
// shell-quartet indices, each encoded as (s1 * nsh^3 + s2 * nsh^2 +
// s3 * nsh + s4) with nsh = number of shells.  Quartets not in the mask
// are evaluated exactly (near-field, penetration zone).
//
// Provenance
// ----------
// Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
// doi:10.1007/978-3-642-93385-1, derives the quartet expansion. The skip
// mask and its classifier are implementation mechanisms, not source-derived.
JKLatticeMatrixSets build_jk_2e_real_space_bipolar_dispatch(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<std::vector<int>>& bipolar_skip_mask,
        double omega = 0.0,
        bool compute_exchange = true);

// Pair-resolved truncation (M1): full truncation-domain control. The caller
// supplies the internal summation cell list `cells` (the (c_g, c_λ, c_σ)
// traversal domain — the other entry points derive it radially from
// opts.cutoff_bohr), the output bra-cell subset `output_indices` (positions
// into `cells`; empty → every cell) and optional per-output shell-pair masks
// `output_shell_masks` (parallel to `output_indices`; empty → every shell
// pair). The third domain, the density support, is data-level: P(h) lookups
// follow P_real_space's own cell list. Each emitted (sub-)block is the exact
// sum over `cells`. `compute_exchange = false` skips the K contraction
// (pure-functional J-only builds; K blocks return zero).
JKLatticeMatrixSets build_jk_2e_real_space_domains(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        const std::vector<int>& output_indices,
        const std::vector<std::vector<uint8_t>>& output_shell_masks,
        double omega = 0.0,
        bool compute_exchange = true);

// Gamma-only finite-domain density adjoint. Uses the same internal/output/
// density-support traversal and screening decisions as the ordinary domain
// builder, but also differentiates the two bilinear energy contractions with
// respect to the single homogeneous Gamma density matrix. This is required
// when the density support is wider than the Fock-output domain: the forward
// J/K matrices alone are then not the derivative of the truncated energy.
// Pair-resolved callers supply the output and density shell masks plus the
// unmasked physical gamma_density; otherwise P_real_space must contain the
// same Gamma density block in every cell.
JKLatticeMatrixSets build_jk_2e_real_space_domains_gamma_derivative(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        const std::vector<int>& output_indices,
        double omega = 0.0,
        bool compute_exchange = true,
        const std::vector<std::vector<uint8_t>>& output_shell_masks = {},
        const std::vector<std::vector<uint8_t>>& density_shell_masks = {},
        const Eigen::MatrixXd& gamma_density = Eigen::MatrixXd());

// General finite-domain density adjoint. Differentiates
//   E_J = 1/2 sum_g P(g) : J_g[P]
//   E_K = 1/2 sum_g P(g) : K_g[P]
// with respect to every real-space density block P(h), using the same
// internal, output, density-support, shell-mask, and screening decisions as
// build_jk_2e_real_space_domains. This is the multi-k counterpart of the
// homogeneous Gamma derivative above.
JKLatticeMatrixSets build_jk_2e_real_space_domains_density_derivative(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        const std::vector<int>& output_indices = {},
        double omega = 0.0,
        bool compute_exchange = true,
        const std::vector<std::vector<uint8_t>>& output_shell_masks = {},
        const std::vector<std::vector<uint8_t>>& density_shell_masks = {});

// ---------------------------------------------------------------------------
// CCM (Cyclic Cluster Model) — WSSC-weighted four-center J/K build.
//
// Identical kernel shape to build_jk_gamma_molecular_limit_explicit, but
// applies the per-atom-pair WSSC four-center weights ω_{abcd} during the
// shell-quartet loop, before the density contraction. This avoids the O(n⁴)
// ERI tensor entirely — the weights are applied at the integral level.
//
// The four-center weight (Peintinger & Bredow, JCC 35, 839, eq 18):
//
//   ω_{μνλσ} = ω_{μν}(g) · (ω_{μλ}(p) + ω_{νλ}(p−g))/2 · ω_{λσ}(0)
//
// for J: (μ_0 ν_g | λ_p σ_p). For K: (μ_0 λ_p | ν_g σ_p) the indices are
// remapped accordingly (see the source).
//
// ``cells`` is the real-space lattice-sum cell list (must include the home
// cell at index 0). ``weight_cells`` and ``weight_matrices`` carry the WSSC
// two-center weights: for each minimum-image cell ``weight_cells[i]`` the
// associated ``weight_matrices[i]`` is an (n_atoms, n_atoms) matrix W[A,B]
// giving the pair weight ω_{AB} of atom B at that cell offset relative to
// atom A. Offsets absent from the list are taken as weight 0 (outside the
// WSSC). The home cell (0,0,0) must be present.
//
// ``atom_per_shell`` maps each AO shell to its supercell atom index (0-based,
// length = n_shells). This is available from the vibe-qc BasisSet's ShellInfo
// or from libint2's Shell::atom_index.
JKMatrices build_jk_ccm_weighted(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    const LatticeSumOptions& opts,
    const Eigen::MatrixXd& P_gamma,
    const std::vector<Eigen::Vector3i>& weight_cells,
    const std::vector<Eigen::MatrixXd>& weight_matrices,
    const std::string& method = "bra_home",
    double omega = 0.0);

}  // namespace vibeqc
