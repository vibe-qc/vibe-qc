// Analytic nuclear gradient of the restricted Hartree-Fock energy.
//
// Pople-Binkley formula for closed-shell RHF:
//   dE/dR_A = dE_nuc/dR_A
//           + Σ_μν D_μν (dH_μν/dR_A)
//           + (1/2) Σ_μνλσ D_μν D_λσ d/dR_A[(μν|λσ) - (1/2)(μλ|νσ)]
//           - Σ_μν W_μν (dS_μν/dR_A)
//
// with D = 2 C_occ C_occ^T (closed-shell total density)
//      W = 2 C_occ · diag(ε_occ) · C_occ^T (energy-weighted density)
//
// Reference: Pople, Krishnan, Schlegel, Binkley, Int. J. Quantum Chem. Symp.
// 13, 225 (1979); Szabo & Ostlund, "Modern Quantum Chemistry", Section 3.4.8.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

#include "basis.hpp"
#include "cosx.hpp"
#include "ecp.hpp"
#include "grid.hpp"
#include "molecule.hpp"
#include "rhf.hpp"
#include "rks.hpp"
#include "uhf.hpp"
#include "uks.hpp"

namespace vibeqc {

// Options for the analytic-gradient drivers. Currently controls only
// density fitting; future options (Becke-weight derivatives, screening
// thresholds, ...) land here to keep the gradient API stable.
struct GradientOptions {
    // Density fitting for the two-electron gradient. When ``density_fit
    // = true``, ∂E_J + α_HF · ∂E_K route through the DF factorisation
    // (Weigend, PCCP 4, 4285 (2002)) instead of the four-index ERI
    // derivative. The Hcore (T + V_ne), overlap, and (for DFT) XC
    // gradient pieces are unchanged — DF only touches J / K.
    bool density_fit = false;

    // libint-recognised auxiliary basis name. Empty + density_fit=true
    // raises with a hint at the Python autodetect helper
    // ``vibeqc.default_aux_basis_for(orbital_basis_name, kind="jk")``.
    std::string aux_basis;

    // Chain-of-spheres exchange (RIJCOSX). When ``cosx = true`` and
    // ``density_fit = true``, the K piece of the two-electron
    // gradient routes through ``compute_cosx_k_gradient_contribution``
    // (frozen-grid / frozen-Q approximation; ~µHa/bohr per atom on
    // neutral organics). The J piece still goes through ``df.compute_j_gradient``.
    // Pure DFT (α_HF = 0) makes the flag a no-op.
    bool cosx = false;
    GridOptions cosx_grid = default_cosx_grid_options();

    // COSX variant / grid-level / accuracy threshold (RIJCOSX upgrade,
    // M1). Consulted only when ``cosx == true``. Mirrors the SCF Options
    // fields so a gradient uses the same K build as the energy. See
    // cosx.hpp.
    CosxVariant cosx_variant = CosxVariant::AUTO;
    int cosx_grid_level = 0;
    double thresh_cosx = 1e-6;

    // Effective core potential placement. When non-empty, the gradient
    // drivers apply the matching ``Z_eff = Z − n_core`` to the nuclear-
    // repulsion and nuclear-attraction derivative pieces and add the
    // ∂V_ECP/∂R contribution (computed via libecpint's first-derivative
    // API). Must match the SCF call's ``opts.ecp_centers`` / ``ecp_library``
    // to keep gradient ↔ energy consistent. The ASE calculator copies
    // these fields onto the GradientOptions automatically; manual
    // ``compute_gradient*`` callers running with ECPs must set them
    // explicitly.
    std::vector<ECPCenter> ecp_centers;
    std::string ecp_library;

    // Inline primitive ECPs (#574). Mirrors the ``RHFOptions`` /
    // ``UHFOptions`` / ``RKSOptions`` / ``UKSOptions`` fields of the same
    // name. XML-library and inline-primitive inputs are mutually exclusive,
    // matching the validated SCF dispatch; supplying both is ambiguous and
    // fails closed before integral work. The gradient must therefore carry
    // the one exact route used to build the SCF Hamiltonian.
    //
    // Bases whose per-element cores are not available as a libecpint XML
    // library (vDZP, def2-mSVP, CRYSTAL/pob data) reach the SCF exclusively
    // through this path -- ``python/vibeqc/ecp_metadata.py``
    // ``attach_inline_ecp_options_from_basis_sidecar`` populates it and
    // leaves ``ecp_centers`` empty. Before #574 ``GradientOptions`` could
    // not express it and the drivers silently fell back to the bare-Z
    // all-electron branch (measured: 0.16 Ha/bohr on CH4/vDZP).
    //
    // ``ecp_primitive_blocks`` / ``ecp_primitive_centers`` are per-ECP-centre
    // and must be the same length; ``ecp_effective_charges`` is per-ATOM
    // (bare Z on atoms carrying no ECP). Must match the SCF call's values.
    std::vector<ECPPrimitiveBlock> ecp_primitive_blocks;
    std::vector<std::array<double, 3>> ecp_primitive_centers;
    std::vector<double> ecp_effective_charges;
    int ecp_total_ncore = 0;
};

// Returns an (n_atoms, 3) matrix of gradient components in Hartree/bohr.
// Row A, column c = dE/dR_{A,c} with c ∈ {x, y, z}.
Eigen::MatrixXd compute_gradient(const Molecule& mol,
                                 const BasisSet& basis,
                                 const RHFResult& result,
                                 const GradientOptions& options = {});

// UHF counterpart. Uses the two-particle density
//   Γ_μνλσ = (1/2)(D_α+D_β)_μν (D_α+D_β)_λσ
//          − (1/2)(D_α)_μλ (D_α)_νσ
//          − (1/2)(D_β)_μλ (D_β)_νσ
// which collapses to the RHF Γ for closed-shell systems.
//
// When ``options.density_fit = true``, the two-electron piece routes
// through the DF factorisation: J on the *total* density D_α+D_β plus
// per-spin K each scaled by α_HF/2. Identical in the closed-shell
// limit (D_α = D_β = D_RHF/2) to the closed-shell DF gradient. The
// J/K pieces are summed across spins explicitly rather than fused
// (one ``compute_j_gradient`` + two ``compute_k_gradient`` calls;
// ~3× the 2c/3c kernel work of the closed-shell fused path — fuseable
// future optimisation tracked).
Eigen::MatrixXd compute_gradient_uhf(const Molecule& mol,
                                     const BasisSet& basis,
                                     const UHFResult& result,
                                     const GradientOptions& options = {});

// UKS (unrestricted open-shell DFT) gradient. Supports LDA, GGA, and
// hybrid functionals; the HF exchange fraction is read from the
// functional and applied to the UHF-style two-electron gradient with
// the appropriate scaling.
//
// ``options.density_fit = true`` routes the two-electron piece through
// the DF factorisation (see compute_gradient_uhf for the per-spin J/K
// composition); the XC Pulay-force piece is unchanged.
Eigen::MatrixXd compute_gradient_uks(const Molecule& mol,
                                     const BasisSet& basis,
                                     const UKSResult& result,
                                     const GridOptions& grid_options = {},
                                     const GradientOptions& options = {});

// Closed-shell Kohn-Sham (RKS) analytic gradient. Currently supports
// LDA functionals only — GGA adds a basis-Hessian contribution and lands
// in a follow-up commit. Becke-weight derivatives are not yet included
// either, so the gradient is grid-accuracy limited (~1e-5 Ha/bohr on
// the default grid). grid_options controls the integration grid; using
// the same options as the SCF call gives the cleanest numerical
// consistency.
Eigen::MatrixXd compute_gradient_rks(const Molecule& mol,
                                     const BasisSet& basis,
                                     const RKSResult& result,
                                     const GridOptions& grid_options = {},
                                     const GradientOptions& options = {});

// --- Individual contributions (exposed for unit testing / diagnostics) ---

Eigen::MatrixXd nuclear_repulsion_gradient(const Molecule& mol);

// Effective-charges overload — when ECPs replace core electrons, the
// nuclear-repulsion derivative must use ``Z_eff = Z − n_core``, not
// the bare atomic numbers. ``effective_charges`` is a per-atom vector
// in ``mol.atoms()`` order. Reduces to the bare-Z overload when every
// entry equals the corresponding ``atoms[i].Z``.
Eigen::MatrixXd nuclear_repulsion_gradient(
    const Molecule& mol,
    const std::vector<double>& effective_charges);

// -tr(W · dS/dR_A) contribution (from the orthonormality constraint).
Eigen::MatrixXd overlap_gradient_contribution(const BasisSet& basis,
                                              const Molecule& mol,
                                              const Eigen::MatrixXd& W);

// tr(D · d(T+V)/dR_A) contribution.
Eigen::MatrixXd one_electron_gradient_contribution(const BasisSet& basis,
                                                    const Molecule& mol,
                                                    const Eigen::MatrixXd& D);

// Effective-charges overload — when ECPs replace core electrons, the
// nuclear-attraction derivative kernel takes per-atom effective
// charges instead of bare atomic numbers, matching the SCF's
// ``compute_nuclear_with_charges`` Hcore build. The kinetic-energy
// piece is unchanged.
Eigen::MatrixXd one_electron_gradient_contribution(
    const BasisSet& basis, const Molecule& mol,
    const Eigen::MatrixXd& D,
    const std::vector<double>& effective_charges);

// RHF two-electron gradient contribution with arbitrary HF-exchange scaling.
//   Γ_μνλσ = (1/2) D_μν D_λσ − (alpha_hf / 4) D_μλ D_νσ
// For alpha_hf = 1 this is plain HF (J − (1/2) K). For alpha_hf ∈ (0, 1) it's
// a hybrid DFT exchange contribution. For alpha_hf = 0 only Coulomb remains.
Eigen::MatrixXd two_electron_gradient_contribution(
    const BasisSet& basis, const Molecule& mol,
    const Eigen::MatrixXd& D, double alpha_hf = 1.0);

// XC-Pulay piece of the RKS gradient: the basis-function derivatives of E_xc[ρ]
// on a *fixed* Becke grid (the same grid-fixed approximation compute_gradient_rks
// uses internally — no Becke weight derivative). LDA / GGA / meta-GGA. Exposed for
// the Γ-CCM force driver: the CCM XC term is the ordinary molecular functional on
// the cluster (supercell) density, so it does NOT carry the WSSC fold (unlike
// h^CCM / J^CCM) and its gradient is exactly this molecular XC-Pulay on the
// supercell. Returns (n_atoms, 3) in Hartree/bohr.
Eigen::MatrixXd xc_pulay_gradient_rks(const Molecule& mol,
                                      const BasisSet& basis,
                                      const Eigen::MatrixXd& D,
                                      const std::string& functional_name,
                                      const GridOptions& grid_options = {});

// Open-shell counterpart of xc_pulay_gradient_rks: the XC-Pulay piece of the UKS
// gradient on a fixed Becke grid (LDA / GGA / meta-GGA, no Becke weight derivative),
// for the Γ-CCM UKS force driver (the CCM XC is the ordinary molecular functional of
// the supercell spin densities and does not fold). Returns (n_atoms, 3) Ha/bohr.
Eigen::MatrixXd xc_pulay_gradient_uks(const Molecule& mol,
                                      const BasisSet& basis,
                                      const Eigen::MatrixXd& D_alpha,
                                      const Eigen::MatrixXd& D_beta,
                                      const std::string& functional_name,
                                      const GridOptions& grid_options = {});

// CPCM solvation-gradient helper. Given external point charges {q_i}
// at positions {s_i}, returns the per-center derivatives of
//   E_ext = Σ_i q_i · Tr(D · M_i),   M_i,μν = ⟨μ|1/|r − s_i||ν⟩
// (M_i is the *positive* Coulomb kernel). libint's nuclear-attraction
// gradient engine emits 3·(2 + n_points) derivative buffers per shell
// pair; this routine splits them into:
//   * atom_grad  (n_atoms, 3): the bra + ket basis-center derivatives,
//                 accumulated onto the atom carrying each shell.
//   * point_grad (n_points, 3): the derivative with respect to each
//                 external point-charge position s_i.
// The CPCM caller routes each point_grad[i] row onto the parent atom
// the cavity point i is attached to (cavity points move rigidly with
// their parent atomic sphere). Both pieces carry the sign of
// dE_ext/dR — libint's internal operator V = −Σ q_i ⟨1/r⟩ flips the
// sign of the raw engine output, which this routine undoes once.
struct ExternalChargeGradient {
    Eigen::MatrixXd atom_grad;   // (n_atoms, 3), Hartree/bohr
    Eigen::MatrixXd point_grad;  // (n_points, 3), Hartree/bohr
};

ExternalChargeGradient compute_external_charge_density_gradient(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D,
    const std::vector<double>& charges,
    const std::vector<std::array<double, 3>>& positions);

// UHF / UKS two-electron gradient with arbitrary HF-exchange scaling.
// The two-particle density is
//   Γ_μνλσ = (1/2)(D_α+D_β)_μν (D_α+D_β)_λσ
//          − (α_HF/2) D_α_μλ D_α_νσ
//          − (α_HF/2) D_β_μλ D_β_νσ
// α_HF = 1 → plain UHF. Hybrid DFT passes the functional's HF fraction,
// pure DFT passes 0 (only Coulomb survives).
Eigen::MatrixXd two_electron_gradient_contribution_uhf(
    const BasisSet& basis, const Molecule& mol,
    const Eigen::MatrixXd& D_alpha, const Eigen::MatrixXd& D_beta,
    double alpha_hf = 1.0);

// CASSCF general two-electron nuclear gradient.  Accepts a pre-computed
// AO 2-particle density matrix Γ_μνλσ in row-major order:
//   idx = ((mu * nb + nu) * nb + lam) * nb + sig
// where nb = basis.nbasis().  Uses the same shell-quartet derivative-
// integral loop as two_electron_gradient_contribution, but replaces the
// RHF density-product form with the supplied Γ — essential for CASSCF
// where the 2-RDM does not factorise into 1-RDM products.
//
// Reference: Helgaker, Jørgensen & Olsen, "Molecular Electronic-
// Structure Theory", §12.5 (MCSCF analytic gradients).  The general-
// tensor two-electron piece is the same one OpenMolcas's ALASKA module
// contracts with the RASSCF 2-RDM for CASSCF / RASSCF forces.
Eigen::MatrixXd two_electron_gradient_casscf(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::Ref<const Eigen::VectorXd>& gamma_ao_flat);

}  // namespace vibeqc
