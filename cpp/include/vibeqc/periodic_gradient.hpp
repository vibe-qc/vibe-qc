// Periodic atomic gradients (Phase G1 — v0.6.0).
//
// Lattice-summed analogues of the molecular gradient primitives in
// gradient.hpp. For Γ-only periodic RHF the total atomic gradient
// decomposes into four contributions:
//
//   F_A = ∂E_nn^pc/∂R_A                                        (nuclear-rep)
//       + Σ_g  tr(D(g) · ∂h_core(g)/∂R_A)                      (1-e, kinetic + V)
//       - Σ_g  tr(W(g) · ∂S(g)/∂R_A)                           (overlap-Lagrangian)
//       + Σ_g Σ_h  Γ(g, h) · ∂(μν0|λg σh)/∂R_A                  (2-e Pulay; G1a-3)
//
// Each lattice-summed contribution is the molecular gradient with an
// outer loop over cell vectors g (and h, for the 2-e term) and a
// "translate the bra/ket basis function origins by g" step before
// calling libint with deriv_order = 1.
//
// G1a-1 ships the nuclear-rep + overlap-Lagrangian + 1-e Pulay parts
// (this header). G1a-2 will add the 2-e Pulay; together those are
// the full Γ-only RHF analytic gradient.

#pragma once

#include <Eigen/Dense>

#include "basis.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

// ----- Nuclear repulsion gradient (per-cell convention) -----------------
//
// Force on each unit-cell atom from every (other-atom, lattice-cell)
// pair. Uses ``opts.coulomb_method`` to dispatch (DIRECT_TRUNCATED →
// straight 1/r real-space sum; EWALD_3D → the derivative of the same
// Ewald energy returned by ``nuclear_repulsion_per_cell``; throws for
// SLAB_EWALD_2D / NEUTRALIZED_1D for now).
//
// Returns a (n_atoms, 3) matrix in Hartree / bohr.
Eigen::MatrixXd nuclear_repulsion_gradient_per_cell(
    const PeriodicSystem& system,
    const LatticeSumOptions& opts);

// ----- Slab (dim=2) bare 2D-Ewald nuclear-repulsion gradient ------------
//
// Analytic gradient of exactly what ``nuclear_repulsion_per_cell`` computes
// under ``CoulombMethod::SLAB_EWALD_2D``: the bare Parry / de Leeuw-Perram
// four-term point-charge block, evaluated as the background-neutralised
// primitive minus the exact uniform-sheet term. Mirrors the energy's
// composition (same α = ``slab_ewald_alpha`` or 0.4, same real cutoff
// ``nuclear_cutoff_bohr``, z_background = 0):
//   ∂E_bare/∂R_A = ∂E_with_background/∂R_A + (2π/A) q_bg Z_A sgn(z_A) n̂,
// where q_bg = −ΣZ and the added term removes the sheet's contribution
// −(2π/A) q_bg Z_A sgn(z_A − z_b) n̂ that the primitive's gradient carries.
//
// Returns a (n_atoms, 3) matrix in Hartree / bohr (same convention as
// ``nuclear_repulsion_gradient_per_cell``). Throws for ``dim != 2`` or a
// degenerate in-plane lattice.
Eigen::MatrixXd nuclear_repulsion_slab_ewald_2d_gradient(
    const PeriodicSystem& system,
    const LatticeSumOptions& opts);

// ----- Overlap lattice gradient -----------------------------------------
//
// Computes -Σ_g tr(W(g) · ∂S(g)/∂R) with S_μν(g) = ⟨χ_μ(0) | χ_ν(g)⟩.
//
// W_set is the energy-weighted density on the same cell list as
// ``S_set`` from ``compute_overlap_lattice``. For RHF/RKS:
// W(g) = 2 Σ_i ε_i C_μi C_νi (real-space Bloch-fold).
//
// Returns a (n_atoms, 3) matrix in Hartree / bohr.
//
// ``apply_pair_filter`` mirrors what compute_overlap_lattice enumerates:
// with it set (the default) the derivative runs over
// ``{(mu, nu, g) : |O_mu - O_nu - g| <= opts.cutoff_bohr}``, which is the
// term set that builder uses (see vibeqc/lattice_pair_cells.hpp). Clear it
// for a caller whose S is the all-pairs sum over the supplied cells -- the
// Ewald-split V_ne G=0 correction is the one such caller. Differentiating
// a different term set than the energy built is an analytic-vs-FD residual
// by construction.
Eigen::MatrixXd overlap_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& W_set,
    const LatticeSumOptions& opts,
    bool apply_pair_filter = true);

// ----- Kinetic lattice gradient -----------------------------------------
//
// Computes Σ_g tr(D(g) · ∂T(g)/∂R) with T_μν(g) = ⟨χ_μ(0) | -½∇² | χ_ν(g)⟩.
//
// D_set is the real-space density on the same cell list as ``T_set``.
//
// Returns a (n_atoms, 3) matrix in Hartree / bohr.
Eigen::MatrixXd kinetic_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts);

// ----- ERI (2-electron) lattice gradient --------------------------------
//
// Computes Σ_{g_a, g_b, g_c} Σ_μνλσ Γ(g_a, g_b, g_c)_μνλσ ·
// ∂(μ_0 ν_{g_a} | λ_{g_b} σ_{g_c}) / ∂R, with the cell-resolved Γ:
//
//   Γ(g_a, g_b, g_c)_μνλσ = (1/2)    · D(g_a)_μν · D(g_c - g_b)_λσ      (J)
//                          - (α_HF/4) · D(g_b)_μλ · D(g_c - g_a)_νσ      (K)
//
// In the molecular limit (D(g≠0) ≈ 0, ERIs with non-zero cell offsets ≈
// 0) this reduces bit-for-bit to the molecular two_electron_gradient_
// contribution.
//
// ``j_scale`` scales the J term in Γ ((j_scale/2)·D·D); set it to 0 for an
// exchange-only gradient. ``omega`` > 0 switches the 2-electron operator
// to the erfc-attenuated Coulomb kernel erfc(ω·r₁₂)/r₁₂ (the short-range
// J_SR of the BIPOLE Ewald split); 0 keeps the full 1/r₁₂ Coulomb.
// ``exchange_energy_convention`` selects the direct energy contraction
// -alpha/4 sum_g D(g):K(g) for the K term even when omega == 0. The default
// preserves the true-periodic DIRECT_TRUNCATED K convention used by the
// low-level periodic-gradient regression.
// ``internal_cells`` optionally replaces the radial cell list for the internal
// (c_lambda, c_sigma) image traversal. ``output_cells`` optionally selects the
// c_g Fock-output cells independently of ``D_set.cells``, which remain the
// density support used for every P(h) lookup. Together these differentiate the
// M5 padded erfc J_SR/K_SR energy domain exactly: output stays at the ordinary
// cutoff, density covers pair differences out to 2x cutoff, and translated
// ket-pair images reach the production short-range radius. Empty lists preserve
// the historical single-domain traversal. ``output_shell_masks`` optionally
// selects the emitted (s1_home, s2_output) shell pairs independently for every
// output cell, matching the pair-resolved SYM3b energy domain.
//
// Returns a (n_atoms, 3) matrix in Hartree / bohr.
Eigen::MatrixXd eri_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts,
    double alpha_hf = 1.0,
    double j_scale = 1.0,
    double omega = 0.0,
    bool exchange_energy_convention = false,
    const std::vector<LatticeCell>& internal_cells = {},
    const std::vector<LatticeCell>& output_cells = {},
    const std::vector<std::vector<uint8_t>>& output_shell_masks = {});

// ----- Nuclear-attraction lattice gradient -------------------------------
//
// Computes Σ_g tr(D(g) · ∂V(g)/∂R) with
// V_μν(g) = Σ_{A, h} -Z_A ⟨χ_μ(0) | 1/|r - (R_A + h)| | χ_ν(g)⟩.
//
// libint's "nuclear" derivative buffer family has 3 · (2 + N_q) entries
// per shell pair: 2 basis-center derivatives + 1 derivative per
// nuclear point charge. For periodic, N_q = N_atoms × N_cells_within_
// nuclear_cutoff.
//
// Returns a (n_atoms, 3) matrix in Hartree / bohr.
Eigen::MatrixXd nuclear_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts);

// Screened (erfc) nuclear-attraction lattice gradient — the short-range
// piece of the BIPOLE Ewald V_ne gradient. Identical structure to
// nuclear_lattice_gradient_contribution but with libint's erfc_nuclear
// operator (kernel erfc(ω·r_C)/r_C), so it differentiates exactly the
// V_short that compute_nuclear_erfc_lattice builds. The complementary
// long-range (reciprocal) + background pieces are differentiated in
// Python (bipole_gradient via the AO-pair-FT gradient + structure
// factor). Returns Σ_{g,μν} D(g)_μν ∂V_short_μν(g)/∂R_A as (n_atoms, 3).
Eigen::MatrixXd nuclear_erfc_lattice_gradient_contribution(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& D_set,
    const LatticeSumOptions& opts,
    double omega);

}  // namespace vibeqc
