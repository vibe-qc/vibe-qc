// Ewald summation for 3D periodic lattices of point charges.
//
// Mathematical formulation
// ========================
//
// For a unit cell with N point charges at positions R_A (Cartesian) with
// values Z_A (signed), tiled periodically on lattice vectors {a_1, a_2, a_3},
// the Madelung energy per unit cell is
//
//   E = (1/2) Σ_{A,B} Σ'_g Z_A Z_B / |R_A − R_B + g|
//
// where g = n_1 a_1 + n_2 a_2 + n_3 a_3 runs over integer lattice vectors
// and the primed sum excludes the (A=B, g=0) self-term. The direct sum is
// conditionally convergent in 3D — different finite summation shapes give
// different limits unless a compensating uniform background is imposed.
// Ewald summation introduces a Gaussian screening function of width 1/(2α)
// to split the sum into two exponentially convergent parts:
//
//   E_real  = (1/2) Σ_{A,B} Σ'_g Z_A Z_B · erfc(α · |r_AB + g|) / |r_AB + g|
//   E_recip = (2π/V) · Σ_{G ≠ 0} |S(G)|² · exp(−|G|²/(4α²)) / |G|²
//   E_self  = −(α/√π) · Σ_A Z_A²
//   E_bg    = −(π/(2 α² V)) · (Σ_A Z_A)²
//
//   S(G)    = Σ_A Z_A · exp(i G · R_A)
//   V       = |det(a_1, a_2, a_3)|
//   G       = n_1 b_1 + n_2 b_2 + n_3 b_3,  a_i · b_j = 2π δ_{ij}
//
// E = E_real + E_recip + E_self + E_bg. The final answer is independent of
// α — α only balances how much work goes into each sum. The ``E_bg`` term
// makes the result well-defined for a non-neutral cell under the jellium
// convention (a uniform compensating charge at infinity). For a neutral
// cell (Σ_A Z_A = 0) the background term vanishes.
//
// Surface (dipole) term
// ---------------------
// This implementation uses the **tin-foil boundary condition** — the
// shape-dependent dipole-surface term is omitted. All mainstream
// solid-state QC codes (CRYSTAL, VASP, PySCF, Quantum ESPRESSO) default
// to tin-foil.
//
// Scope
// -----
// 3D periodic systems only. 1D polymers (needs neutralising-line
// prescription) and 2D slabs (needs slab Ewald) are separate future
// phases; pass such systems to ``ewald_point_charge_energy`` and it
// throws.

#pragma once

#include <Eigen/Dense>

#include "periodic.hpp"

namespace vibeqc {

struct EwaldOptions {
    // Gaussian width parameter. ``<= 0`` → auto-select from the real-space
    // cutoff (balanced for ~``tolerance`` accuracy in both sums).
    double alpha = -1.0;

    // Real-space cutoff in bohr. Real-space pair-interaction contributions
    // |r_AB + g| > real_cutoff_bohr are truncated.
    double real_cutoff_bohr = 15.0;

    // Reciprocal-space cutoff in bohr⁻¹. ``<= 0`` → auto-select to match
    // ``tolerance``.
    double recip_cutoff_bohr_inv = -1.0;

    // Target relative accuracy for auto-selected α and cutoffs. 1e-12 is
    // safely converged for small to medium unit cells.
    double tolerance = 1.0e-12;
};

// Ewald-summed Madelung energy for an arbitrary lattice of point charges.
//
// ``positions_cart`` is 3 × N (one atom per column) in bohr. ``charges`` is
// length N with the corresponding Z values (may be fractional, signed, and
// the list need not be neutral — ``E_bg`` handles non-neutral cells).
//
// Throws ``std::invalid_argument`` for non-3D periodic systems, singular
// lattices, or mismatched column count.
double ewald_point_charge_energy(const Eigen::Matrix3d& lattice,
                                 const Eigen::Matrix3Xd& positions_cart,
                                 const Eigen::VectorXd& charges,
                                 const EwaldOptions& opts = {});

// Convenience: nuclear Madelung energy for a PeriodicSystem. Uses each
// atom's ``Z`` as the point charge; calls ``ewald_point_charge_energy``
// under the hood.
double ewald_nuclear_repulsion(const PeriodicSystem& system,
                               const EwaldOptions& opts = {});

// Analytic gradient of ``ewald_point_charge_energy`` w.r.t. each charge
// position. Returns ∂E/∂R_C as a 3 × N matrix (column C is the gradient on
// charge C, in Hartree / bohr). Differentiates the *same* truncated Ewald
// sum the energy evaluates (identical α and real / reciprocal cutoffs), so
// it agrees with a central finite difference of the energy to the screening
// tolerance. The self-energy and jellium-background terms are
// position-independent and contribute nothing. See ewald.cpp for the
// per-term derivation.
Eigen::Matrix3Xd ewald_point_charge_gradient(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& positions_cart,
    const Eigen::VectorXd& charges,
    const EwaldOptions& opts = {});

// Convenience: nuclear Ewald-repulsion gradient for a PeriodicSystem. Uses
// each atom's ``Z`` as the point charge. Returns ∂E_nn/∂R_A as an N × 3
// matrix (row A is the gradient on atom A), matching the N × 3 convention of
// ``nuclear_repulsion_gradient_per_cell``.
Eigen::MatrixXd ewald_nuclear_repulsion_gradient(
    const PeriodicSystem& system,
    const EwaldOptions& opts = {});

// Ewald-summed electrostatic potential of a 3D-periodic lattice of point
// charges, evaluated at a list of Cartesian points.
//
// Returns ``v(r_i)`` for each input row of ``eval_points`` in units of
// Hartree / electron (i.e. the potential that couples to an electron via
// the usual Hamiltonian term ⟨μ | v | ν⟩). The potential decomposes as
//
//   v(r) = v_short(r) + v_long(r) + v_bg
//
//   v_short(r) = Σ_{A, g} Z_A · erfc(α |r − R_A − g|) / |r − R_A − g|
//   v_long(r)  = (4π/V) Σ_{G ≠ 0} ρ̃(G) · exp(i G · r) · exp(−|G|²/(4α²)) / |G|²
//   v_bg       = −(π / (α² V)) · Σ_A Z_A         (jellium-background term
//                                                  for non-neutral cells)
//
// with ρ̃(G) = Σ_A Z_A exp(−i G · R_A). The sign convention is the
// *unsigned* Coulomb potential — callers building the electronic
// nuclear-attraction operator (−Σ Z_A / |r − R_A|) must negate the
// returned values.
//
// Throws ``std::invalid_argument`` for non-3D systems (use the direct
// truncated sum for 1D / 2D, or wait for the slab / line variants).
Eigen::VectorXd ewald_point_charge_potential(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& charge_positions_cart,
    const Eigen::VectorXd& charges,
    const Eigen::MatrixX3d& eval_points_cart,
    const EwaldOptions& opts = {},
    bool include_short_range = true);

// Convenience wrapper: nuclear Coulomb potential (sign included: negative
// where electron attraction is attractive) of a PeriodicSystem evaluated
// at a grid of points.
//
// Equivalent to
//
//   −1 · ewald_point_charge_potential(lattice, positions, Z, eval_points)
//
// with Z and positions pulled from ``system.unit_cell``.
Eigen::VectorXd ewald_nuclear_potential(const PeriodicSystem& system,
                                        const Eigen::MatrixX3d& eval_points_cart,
                                        const EwaldOptions& opts = {},
                                        bool include_short_range = true);

// ---------------------------------------------------------------------------
// Rigorous 2D (slab) Ewald summation — Parry / de Leeuw–Perram–Smith
// ---------------------------------------------------------------------------
//
// Madelung energy per cell of a charge distribution that is periodic in two
// dimensions and finite in the third (a slab / layer). Columns 0 and 1 of
// ``lattice`` are the in-plane lattice vectors a1, a2; the slab normal is
// n̂ = (a1 × a2)/|a1 × a2| and the third lattice column is irrelevant (a
// dim == 2 ``PeriodicSystem`` carries a vacuum vector there). Unlike 3D
// Ewald this is the *true* 2D lattice sum — no vacuum padding and no
// spurious inter-image dipole field along n̂.
//
// With charges q_i at r_i, in-plane separation ρ_ij, normal separation
// z_ij = (r_i − r_j)·n̂, in-plane area A = |a1 × a2|, and 2D reciprocal
// vectors g (g·n̂ = 0), the energy decomposes as
//
//   E = E_real + E_recip(g≠0) + E_recip(g=0) + E_self
//
//   E_real    = ½ Σ_{i,j} Σ'_n q_i q_j erfc(α|r_ij+n|)/|r_ij+n|
//   E_recip   = (π/2A) Σ_{g≠0} (1/g) Σ_{i,j} q_i q_j cos(g·ρ_ij)
//                 [ e^{+g z_ij} erfc(g/2α + α z_ij)
//                 + e^{−g z_ij} erfc(g/2α − α z_ij) ]
//   E_{g=0}   = −(π/A) Σ_{i,j} q_i q_j
//                 [ z_ij erf(α z_ij) + (1/(α√π)) e^{−α²z_ij²} ]
//   E_self    = −(α/√π) Σ_i q_i²
//
// (Parry, Surf. Sci. 49, 433 (1975); de Leeuw & Perram, Mol. Phys. 37,
// 1313 (1979). The g=0 term has no 3D analogue.) The result is independent
// of α; α only balances real- vs reciprocal-space work, exactly as in 3D.
//
// REQUIRES A CHARGE-NEUTRAL CELL (Σ q_i = 0). A net-charged 2D-periodic
// plane has a divergent electrostatic energy with no finite jellium
// regularisation (a uniformly charged plane's potential grows without
// bound), so a non-neutral cell is rejected rather than silently
// regularised. The nuclei-only Madelung energy of a crystal slab is
// therefore *not* obtainable from this routine on its own; the
// self-consistent total (nuclei + electrons, which is neutral) is the
// well-defined object — see handovers/HANDOVER_SLAB_EWALD_2D.md for the planned
// SCF decomposition.
//
// Throws ``std::invalid_argument`` on a non-neutral cell, a degenerate
// in-plane lattice, or a charges/positions size mismatch.
double ewald_2d_point_charge_energy(const Eigen::Matrix3d& lattice,
                                    const Eigen::Matrix3Xd& positions_cart,
                                    const Eigen::VectorXd& charges,
                                    const EwaldOptions& opts = {});

// Rigorous 2D (slab) Ewald electrostatic potential of a charge distribution
// periodic in the plane (lattice columns 0,1) and finite along the normal,
// evaluated at a list of Cartesian points. The 2D analogue of
// ``ewald_point_charge_potential``; the long-range building block for the
// electron-nuclear attraction in a slab SCF. With z = (r − r_j)·n̂,
//
//   v(r) = v_short(r) + v_recip(r) + v_{g=0}(r)
//   v_short  = Σ_{j,n} q_j erfc(α|r − r_j − n|)/|r − r_j − n|
//   v_recip  = (π/A) Σ_{g≠0} (1/g) Σ_j q_j cos(g·(r − r_j)) ×
//                e^{−(g/2α)² − α²z²} [erfcx(g/2α + αz) + erfcx(g/2α − αz)]
//   v_{g=0}  = −(2π/A) Σ_j q_j [ z erf(α z) + (1/(α√π)) e^{−α²z²} ]
//
// Sign convention is the *unsigned* Coulomb potential (positive charges give
// a positive potential); callers building the electronic nuclear-attraction
// operator (−Σ Z_A/|r−R_A|) must negate. Set ``include_short_range=false`` to
// omit the erfc 1/r spikes at the charges (add them analytically via libint's
// erfc_nuclear when building ⟨μ|v|ν⟩ on a grid). Requires a charge-neutral
// cell; throws ``std::invalid_argument`` otherwise (see
// ewald_2d_point_charge_energy).
Eigen::VectorXd ewald_2d_point_charge_potential(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& charge_positions_cart,
    const Eigen::VectorXd& charges,
    const Eigen::MatrixX3d& eval_points_cart,
    const EwaldOptions& opts = {},
    bool include_short_range = true);

// ---------------------------------------------------------------------------
// Background-neutralised 2D (slab) Ewald — a charged layer + a co-located
// uniform neutralising sheet
// ---------------------------------------------------------------------------
//
// The two routines above reject a net-charged cell, because a net-charged
// 2D-periodic plane has a divergent electrostatic energy. The SCF, however,
// needs the rigorous 2D-Ewald energy of the *neutral total* expressed as a
// sum of physically-interpretable per-block pieces (nuclei-only, electron-
// only), and each block is individually net-charged. The standard cure
// (exactly the 3D jellium of ewald_point_charge_energy's e_bg term, but for a
// slab) is a uniform in-plane neutralising background: a charge sheet of
// total charge −Σq_i per cell, placed at a chosen reference plane.
//
// These ``*_with_background`` variants compute the rigorous 2D-Ewald energy /
// potential of the COMBINED neutral system {given point charges q_i} ∪ {a
// uniform in-plane sheet of total charge q_b = −Σq_i per cell at normal
// coordinate z_background}. The bare four-term sum over the points (real,
// recip g ≠ 0, the slab g = 0 term f(z) = z·erf(αz) + e^{−α²z²}/(α√π), and
// self) is computed exactly as in the bare routines. The sheet, being smooth
// and in-plane-uniform, has its in-plane FT a δ at g = 0 and needs NO Ewald
// screening: it adds nothing to the real-space lattice sum (no discrete
// images), nothing to the g ≠ 0 reciprocal sum, and nothing to the point
// self-energy. It enters only through its exact (un-screened) field
//   φ_sheet(z) = −(2π/A) q_b |z − z_b|                (Gauss, σ = q_b/A)
// so it adds
//
//   ΔE_sheet  = −(2π/A) q_b Σ_i q_i |z_i − z_b|       (energy)
//   Δv_sheet(r) = −(2π/A) q_b |z_r − z_b|             (potential)
//
// — the bare |z| field, NOT the screened f(z) kernel the point pairs use.
// (The point–point bare four-term is already α- and origin-invariant for a
// net-charged block: the finite slab g = 0 term regularises it without any
// background. So the physical neutralising sheet must contribute its bare,
// α-independent field; a second screened term would wrongly break that
// α-invariance.) On a charge-neutral input q_b = 0 and these reduce *exactly*
// to ``ewald_2d_point_charge_energy`` / ``..._potential``.
//
// The combined {points + sheet} system is neutral, so the with-background
// energy is the physical, α-invariant Madelung energy of a charged slab in a
// uniform neutralising background. Unlike the bare four-term (origin-
// invariant, no explicit background plane), the with-background energy is
// z_background-DEPENDENT for a net-charged block — the physical dependence on
// where the neutralising background sits, which the BIPOLE surface gradient
// consumes. In the neutral grand total every block's sheet cancels its
// partner's (−ΣZ for the nuclei against +ΣZ for the electrons, co-located at
// one shared z_background), so the total is independent of z_background — a
// single shared value must be used for all blocks of one cell.
//
// References: Parry, Surf. Sci. 49, 433 (1975); de Leeuw & Perram, Mol. Phys.
// 37, 1313 (1979). z_background is a *normal* coordinate (r·n̂), not a raw
// z-component; for an xy-plane slab (n̂ = ẑ) the two coincide.
//
// No neutrality guard: a net-charged ``charges`` is the intended input (the
// sheet neutralises it). Throws on a degenerate in-plane lattice or a
// charges/positions size mismatch.
double ewald_2d_point_charge_energy_with_background(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& positions_cart,
    const Eigen::VectorXd& charges,
    double z_background,
    const EwaldOptions& opts = {});

// Analytic gradient of ``ewald_2d_point_charge_energy_with_background``
// w.r.t. each charge position. Returns ∂E/∂R_C as a 3 × N matrix (column C
// is the gradient on charge C, in Hartree / bohr). Differentiates the SAME
// truncated sums the energy evaluates (identical α, real / reciprocal
// cutoffs, and cell enumeration), term by term:
//
//   real       — the 2D-lattice pair derivative, identical in structure to
//                the 3D ``ewald_point_charge_gradient`` real-space term;
//   recip g≠0  — the z-resolved Parry kernel F(g,z) = e^{gz} erfc(g/2α+αz)
//                + e^{−gz} erfc(g/2α−αz): its in-plane phase derivative
//                gives the in-plane force and its z-derivative
//                F'(g,z) = g [e^{gz} erfc(g/2α+αz) − e^{−gz} erfc(g/2α−αz)]
//                (the two Gaussian terms from erfc' cancel exactly) gives
//                the normal force;
//   g = 0      — d/dz of the slab kernel f(z) = z erf(αz) + e^{−α²z²}/(α√π)
//                is f'(z) = erf(αz) (the Gaussian terms cancel here too);
//   sheet      — ∂ΔE_sheet/∂R_C = −(2π/A) q_b q_C sgn(z_C − z_b) n̂
//                (subgradient sgn(0) = 0 at the kink z_C = z_b);
//   self       — position-independent, zero.
//
// On a neutral cell (q_b = 0) this is the gradient of the bare
// ``ewald_2d_point_charge_energy`` four-term block. References as for the
// energy: Parry, Surf. Sci. 49, 433 (1975); de Leeuw & Perram, Mol. Phys.
// 37, 1313 (1979).
Eigen::Matrix3Xd ewald_2d_point_charge_gradient_with_background(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& positions_cart,
    const Eigen::VectorXd& charges,
    double z_background,
    const EwaldOptions& opts = {});

// Background-neutralised 2D (slab) Ewald electrostatic potential — the
// potential of {point charges q_i} ∪ {uniform sheet of charge −Σq_i at
// z_background}, evaluated at a list of Cartesian points. The sheet's exact
// (un-screened) field −(2π/A) q_b |z_r − z_b| is added to the bare-source
// potential (see ewald_2d_point_charge_potential). Same sign convention
// (unsigned Coulomb;
// negate for the electron nuclear-attraction operator) and the same
// include_short_range switch as the bare routine. No neutrality guard.
Eigen::VectorXd ewald_2d_point_charge_potential_with_background(
    const Eigen::Matrix3d& lattice,
    const Eigen::Matrix3Xd& charge_positions_cart,
    const Eigen::VectorXd& charges,
    double z_background,
    const Eigen::MatrixX3d& eval_points_cart,
    const EwaldOptions& opts = {},
    bool include_short_range = true);

}  // namespace vibeqc
