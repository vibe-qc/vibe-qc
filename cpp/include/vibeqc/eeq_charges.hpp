// SPDX-License-Identifier: MPL-2.0
//
// Atomic partial charges from electronegativity equilibration.
//
// The "EEQ" model used here is the variant published in
//
//     Caldeweyher, Ehlert, Hansen, Neugebauer, Spicher, Bannwarth, Grimme,
//     "A generally applicable atomic-charge dependent London dispersion
//      correction",
//     J. Chem. Phys. 150, 154122 (2019), DOI 10.1063/1.5090222
//
// (Section II.A and Appendix A). It is the charge model the D4 dispersion
// correction reads to scale its reference C6 coefficients; it is also
// useful on its own as a stand-alone atomic-charge analysis tool
// (cheap O(N^3) factorisation, no SCF).
//
//
// === MATH SUMMARY ============================================================
//
// For a molecule of n atoms with positions r_A and atomic numbers Z_A,
// the EEQ atomic charges {q_A} minimise a model electrostatic energy
//
//                                                ┌  ──── ─── ──── ──── ┐
//   E_ES({q}) = Σ_A χ_eff(A) q_A  +  (1/2) Σ_AB │ A q q     +  η q²    │
//                                                └  AB A B       AA A  ┘
//
// subject to the charge-conservation constraint Σ_A q_A = Q_total.
// Introducing a Lagrange multiplier λ for that constraint and setting
// ∂L/∂q_A = 0 produces a linear system of size (n+1) × (n+1):
//
//   ⎡ A    1 ⎤ ⎡ q ⎤   ⎡ −χ_eff ⎤
//   ⎣ 1ᵀ   0 ⎦ ⎣ λ ⎦ = ⎣ Q_total ⎦                                 (1)
//
// The on-diagonal A_AA captures the self-Coulomb of a Gaussian density
// of width γ_A plus the on-site hardness η_A; the off-diagonal A_AB
// is the Coulomb interaction of two such Gaussians:
//
//   A_AA = η_A  +  √(2/π) / γ_A                                    (2a)
//   A_AB = erf(R_AB · κ_AB) / R_AB           (A ≠ B)               (2b)
//
// with κ_AB = 1 / sqrt(γ_A² + γ_B²) the inverse joint Gaussian width.
//
// The CN-modified effective electronegativity χ_eff(A) is
//
//   χ_eff(A) = χ_A  −  κχ_A · √(CN_A)                              (3)
//
// where CN_A is a fractional coordination number computed from a smooth
// counting function over neighbour distances:
//
//   CN_A = clip[ Σ_{B≠A}  ½·(1 + erf(−kcn · (R_AB − R0_AB) / R0_AB)) ]   (4)
//
// with R0_AB = (4/3) · (R_cov(A) + R_cov(B)). R_cov are the Pyykkö &
// Atsumi 2009 covalent radii (bohr); the 4/3 prefactor is the standard
// D3/D4 covalent-radius convention (Grimme 2010) carried into the
// EEQ-2019 coordination number — without it the EEQ charges are ~10 %
// off the reference. The clip is the soft-max saturation
//
//   clip[x] = log(1 + e^{cnmax})  −  log(1 + e^{cnmax − x})        (5)
//
// which is ≈ x for x ≪ cnmax and asymptotes smoothly to log(1+e^{cnmax}).
// The per-element parameters {χ_A, η_A, γ_A, κχ_A} are the fits
// reported in Caldeweyher 2019 SI (Table S2); covalent radii R_cov(A) are
// from Pyykkö & Atsumi, Chem. Eur. J. 15, 188 (2009). Default counting
// constants are kcn = 7.5, cnmax = 8.0, cutoff = 25.0 bohr — the same
// defaults the published reference implementation uses.
//
// Solving (1) is one symmetric LDLᵀ factorisation of a small dense
// matrix. We don't iterate (no self-consistency in q; the charges are
// the exact minimiser of a quadratic form once CN_A is fixed).
//
//
// === USE FROM vibe-qc ========================================================
//
//   #include "vibeqc/eeq_charges.hpp"
//   const auto r = vibeqc::eeq_atomic_charges(mol);
//   r.charges;                        // partial charges, sums to Q_total
//   r.chemical_potential;             // λ
//
// The Python entry is ``vibeqc.eeq_charges(mol)``.
//
//
// === SCOPE ==================================================================
//
// * Molecular (0D) only. Periodic (Ewald-summed) EEQ lives behind a
//   future entry point and reuses the matrix builders here.
// * Element coverage: Z = 1..103 (the range Caldeweyher 2019 fits).
//   Atoms with Z outside this range produce a clear ``std::invalid_argument``.
// * Energy + gradient w.r.t. nuclear coordinates is not yet wired up
//   here — only the charges. (The dispersion-energy gradient needs
//   dq/dR; this header will gain it in a subsequent step.)

#pragma once

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"

namespace vibeqc {

// -----------------------------------------------------------------------------
// Tunable parameters of the EEQ-CN counting function.
//
// Defaults reproduce the constants used in Caldeweyher 2019 (the
// reference D4 paper). Override only if you're recalibrating the model
// or running a sensitivity study.
// -----------------------------------------------------------------------------
struct EEQOptions {
    double cn_cutoff = 25.0;  // pair-distance cutoff for the CN sum (bohr)
    double cn_exp    = 7.5;   // kcn — steepness of the erf counting func
    double cn_max    = 8.0;   // cnmax — saturation parameter of clip[]
};


// -----------------------------------------------------------------------------
// Result of an EEQ charge solve.
//
//   charges            — atomic partial charges in units of e; sums to
//                        the total charge requested.
//   chemical_potential — λ, the Lagrange multiplier of the
//                        Σ_A q_A = Q_total constraint. Equal to the
//                        system's chemical potential (dE_ES/dN at the
//                        EEQ minimum).
// -----------------------------------------------------------------------------
struct EEQResult {
    Eigen::VectorXd charges;
    double chemical_potential = 0.0;
};


// =============================================================================
// Public API
// =============================================================================

// One-shot EEQ solve: build the matrix + RHS, factorise via LDLᵀ, return
// the charges + chemical potential. ``total_charge`` is the molecular
// net charge in electrons (default neutral). Throws ``std::invalid_argument``
// if any element is outside the parameter table (Z = 1..103) or if the
// input has coincident atoms; throws ``std::runtime_error`` if the
// symmetric system fails to factorise (would indicate degenerate
// geometry not caught by the coincident-atoms check).
EEQResult eeq_atomic_charges(const Molecule& mol,
                              double total_charge = 0.0,
                              const EEQOptions& opts = {});


// Standalone CN evaluator. Same formula (4) that ``eeq_atomic_charges``
// consumes internally; exposed for diagnostics, plotting CN-vs-bond-
// distance sweeps, or feeding into a different downstream model.
Eigen::VectorXd eeq_coordination_numbers(const Molecule& mol,
                                          const EEQOptions& opts = {});


// =============================================================================
// Per-element parameter accessors (defined in eeq_charges_data.cpp)
//
// Each returns NaN for an unsupported Z, which lets callers fail
// cleanly with their own error message instead of having a magic
// sentinel propagate through the math.
// =============================================================================

// χ_A — element electronegativity (Hartree). Caldeweyher 2019 fit.
double eeq_electronegativity(int Z);

// η_A — chemical hardness (Ha/e²). Caldeweyher 2019 fit.
double eeq_chemical_hardness(int Z);

// κχ_A — CN-scaling factor for χ_eff (Eq. 3). Caldeweyher 2019 fit.
double eeq_cn_scaling(int Z);

// γ_A — Gaussian-density width (bohr). Caldeweyher 2019 fit.
double eeq_gaussian_width(int Z);

// R_cov(A) — covalent radius in bohr (Pyykkö & Atsumi 2009; bare
// primary-source value). The EEQ CN counting function (Eq. 4)
// applies the standard D3/D4 4/3 scaling to this internally.
double eeq_covalent_radius(int Z);

// Highest Z covered by the shipped parameter tables.
int eeq_max_z();


// =============================================================================
// Backwards-compatible aliases — see eeq_charges.hpp git blame for the
// transition. To be removed in a future release once external callers
// migrate.
// =============================================================================
inline EEQResult eeq_charges(const Molecule& mol,
                              double total_charge = 0.0,
                              const EEQOptions& opts = {}) {
    return eeq_atomic_charges(mol, total_charge, opts);
}
inline double eeq_chi(int Z)     { return eeq_electronegativity(Z); }
inline double eeq_eta(int Z)     { return eeq_chemical_hardness(Z); }
inline double eeq_kcnchi(int Z)  { return eeq_cn_scaling(Z); }
inline double eeq_rad(int Z)     { return eeq_gaussian_width(Z); }
inline double eeq_rcov(int Z)    { return eeq_covalent_radius(Z); }
inline int    eeq_max_supported_Z() { return eeq_max_z(); }

}  // namespace vibeqc
