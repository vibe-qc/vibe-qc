// Phase 17b-2 — second-derivative integral contractions for the
// analytic RHF Hessian.
//
// Each function takes a "weight" matrix (energy-weighted density W
// for overlap, total density D for kinetic + nuclear, two-particle
// density for ERI) and returns the corresponding contribution to the
// 3N × 3N skeleton Hessian:
//
//     H_{Aα,Bβ} = Σ_μν W_μν · ∂²I_μν / (∂R_Aα ∂R_Bβ)            (1-e)
//     H_{Aα,Bβ} = Σ_μνλσ Γ_μνλσ · ∂²(μν|λσ) / (∂R_Aα ∂R_Bβ)   (2-e)
//
// libint2 with deriv_order=2 emits one buffer per upper-triangle pair
// of perturbation indices (i ≤ j); we map each (i, j) pair to the
// (atom, cartesian) tuples on which the integral depends, contract
// with the weight matrix, and scatter into the Hessian (with the
// symmetric off-diagonal contribution for a ≠ b).
//
// Build prerequisite: the vendored libint at third_party/libint/
// must have been built with `LIBINT2_ENABLE_ONEBODY=2` and
// `LIBINT2_ENABLE_ERI=2`. See scripts/build_libint.sh — the default
// configuration covers max_am 5/4/4 for derivative orders 0/1/2, so
// the 2nd-derivative path reaches g functions (def2-TZVPP heavy
// atoms, cc-pVQZ). h functions (L = 5) are energy-only.
//
// The "skeleton" terminology distinguishes these from the "response"
// contributions that depend on CPHF orbital-rotation amplitudes —
// see hessian_assembly.hpp (Phase 17b-3) for the Hessian driver
// that combines skeleton + response.

#pragma once

#include <Eigen/Dense>

#include "basis.hpp"
#include "molecule.hpp"

namespace vibeqc {

// −Σ_μν W_μν · ∂²S/∂R_Aα∂R_Bβ_μν
// Sign matches the RHF energy formula
// ``E = ... − Σ_μν W_μν S_μν`` where W = 2 C_occ ε_occ C_occ^T is the
// energy-weighted density. Returns a (3N, 3N) symmetric matrix in
// Hartree/bohr².
Eigen::MatrixXd compute_overlap_hessian_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& W);

// Σ_μν D_μν · ∂²(T+V)/∂R_Aα∂R_Bβ_μν.
// T+V combined here because both depend on the same (basis center) and
// (nucleus center) coordinates; libint2 emits the 1-e Hamiltonian
// derivatives in two passes (kinetic with 2 centers, nuclear with
// 2 + N_nuclei centers).
Eigen::MatrixXd compute_kinetic_nuclear_hessian_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D);

// (1/2) Σ_μνλσ D_μν D_λσ · ∂²(μν|λσ)/∂R_Aα∂R_Bβ
// − (alpha_hf / 4) Σ_μνλσ D_μλ D_νσ · ∂²(μν|λσ)/∂R_Aα∂R_Bβ
// — the "2-electron skeleton". Pass alpha_hf = 1.0 for plain HF,
// alpha_hf ∈ (0, 1) for hybrid DFT, alpha_hf = 0 for pure DFT
// (Coulomb only; XC second-derivative contribution is separate and
// goes through the grid-based DFT Hessian — not implemented here).
Eigen::MatrixXd compute_eri_hessian_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D,
    double alpha_hf = 1.0);

// UHF / UKS counterpart with separate α and β densities. Two-particle
// density:
//   Γ_μνλσ = (1/2)(D_α+D_β)_μν (D_α+D_β)_λσ
//          − (α_HF/2) D_α_μλ D_α_νσ
//          − (α_HF/2) D_β_μλ D_β_νσ
// (matches ``two_electron_gradient_contribution_uhf`` for the gradient,
// applied to the 2nd-derivative ERI tensor here.)
Eigen::MatrixXd compute_eri_hessian_contribution_uhf(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D_alpha,
    const Eigen::MatrixXd& D_beta,
    double alpha_hf = 1.0);

// ∂²E_nuc / ∂R_Aα∂R_Bβ — the closed-form nuclear-repulsion second
// derivative. Returned as a (3N, 3N) symmetric matrix. No basis-set
// dependence; depends only on atomic numbers and positions.
Eigen::MatrixXd nuclear_repulsion_hessian(const Molecule& mol);

}  // namespace vibeqc
