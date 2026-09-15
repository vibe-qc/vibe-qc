// DFT-D3 dispersion correction (Grimme et al., J. Chem. Phys. 132, 154104
// (2010); J. Comp. Chem. 32, 1456-1465 (2011) for Becke–Johnson damping).
//
// D3(BJ) is a pairwise additive post-SCF correction of the form
//
//     E_disp = - Σ_{A<B} Σ_{n=6,8} s_n · C_n^AB / ( r_AB^n + f_d(R_0^AB)^n )
//
// where
//
//     f_d(R_0^AB) = a1 · R_0^AB + a2    (BJ damping radius)
//     R_0^AB      = sqrt( C_8^AB / C_6^AB )
//     C_8^AB      = 3 · C_6^AB · sqrt(Q_A · Q_B)
//     Q_A         = 0.5 · sqrt(Z_A) · r2r4_A
//
// C_6^AB depends on the fractional coordination numbers CN_A, CN_B of
// atoms A and B via a Gaussian-weighted interpolation of a pre-tabulated
// reference grid c6ab(Z_A, Z_B, CN_A_ref, CN_B_ref).
//
// Coordination number (smooth counting function):
//
//     CN_A = Σ_{B≠A} 1 / (1 + exp[-k1 · (k2 · (r_A^cov + r_B^cov) / r_AB - 1)])
//
// with k1 ≈ 16.0, k2 ≈ 4/3 (multiplied by r_cov ratios already). The
// covalent radii r_cov and ratio parameters r2r4 are tabulated per
// element; those live in dispersion_data.cpp so the framework code here
// stays data-free and testable in isolation.
//
// v0.5.0 scope (Phase D1): D3 with Becke–Johnson damping only ("zero
// damping" is the earlier D3(0) variant which modern papers recommend
// against).
//
// ---- Axilrod-Teller-Muto (ATM) three-body term ---------------------------
//
// The optional three-body dipole-dipole-dipole term is enabled by s9 != 0:
//
//     E_disp = E_disp^(2)  +  s9 · E_disp^(3)
//
// which is Eq. (3) of Grimme, Brandenburg, Bannwarth & Hansen, J. Chem.
// Phys. 143, 054107 (2015) (they call the scale factor a3, "which is unity
// by default"). Per-triple:
//
//     E^(3) = - Σ_{A<B<C} C9_ABC · ang_ABC · f_damp(ABC)
//
//     C9_ABC = - sqrt( |C6_AB · C6_AC · C6_BC| )        (geometric mean)
//
//                    3·cosθ_A·cosθ_B·cosθ_C + 1
//     ang_ABC =  ---------------------------------
//                      (r_AB · r_BC · r_AC)^3
//
// evaluated in the numerically convenient side-length form (see
// dispersion.cpp). The damping is the D3 *zero-damping* function built on
// the tabulated ab-initio cut-off radii r0ab — NOT the Becke–Johnson
// radius a1·R0 + a2 used by the two-body term above:
//
//     f_damp = 1 / ( 1 + 6 · [ (4/3) · (r0_prod / r_prod)^(1/3) ]^(alp+2) )
//
// with r_prod = r_AB·r_BC·r_AC, r0_prod = r0ab_AB·r0ab_BC·r0ab_AC and the
// D3 default alp = 14 (so the exponent is 16). This form is verified
// bit-for-bit against Grimme's reference implementation (simple-dftd3's
// `RationalDampingParam(..., s9=...)`) — see tests/test_dispersion_atm.py.
//
// NOTE for anyone tempted to reuse the D4 ATM code in
// python/vibeqc/dispersion_d4_model.py: D4's ATM damping uses the *BJ*
// radii (a1·sqrt(C8/C6) + a2) and alp = 16 directly. D3's uses r0ab and
// alp+2. They are different functions; do not cross-wire them.
//
// s9 defaults to 0.0: the published per-functional (s8, a1, a2) damping
// sets were fit *without* the three-body term, so plain "D3(BJ)" means
// two-body only. Recipes that define themselves to include ATM — the
// PBEh-3c / HSE-3c composites — set s9 = 1.0 explicitly.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "vibeqc/molecule.hpp"

namespace vibeqc {

// Damping parameters of D3(BJ). s6 is conventionally 1.0 (except for
// double-hybrid functionals); s8 carries the functional-specific weight
// of the C_8 term; a1, a2 are the BJ damping constants.
//
// s9 scales the Axilrod-Teller-Muto three-body term (Grimme 2015 Eq. 3,
// where it is called a3). It defaults to 0.0 because published D3(BJ)
// damping sets are fit two-body-only; set it to 1.0 for methods whose
// definition includes ATM (PBEh-3c, HSE-3c).
struct D3BJParams {
    double s6 = 1.0;
    double s8 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double s9 = 0.0;
};

// D3 zero-damping steepness. Grimme's D3 default alp = 14; the ATM damping
// uses the exponent alp + 2 = 16 (dftd3's `alp9`).
constexpr double kD3_alp = 14.0;

// Per-atom coordination number plus its derivative w.r.t. atom positions.
struct CoordinationNumbers {
    Eigen::VectorXd cn;                                 // (n_atoms,)
    std::vector<Eigen::MatrixX3d> dcn_dr;               // dCN_A / dr_B, one matrix per A
};

struct DispersionResult {
    double energy = 0.0;                                // Hartree
    Eigen::MatrixX3d gradient;                          // (n_atoms, 3), empty if not requested
};

// ---- Low-level building blocks ------------------------------------------
//
// These are exposed so the unit tests can exercise them independently of
// the full pairwise summation (which needs the C6 reference table).

// Becke–Johnson damping function f_d(R_0) = a1·R_0 + a2.
inline double bj_damping_radius(double R0, const D3BJParams& p) {
    return p.a1 * R0 + p.a2;
}

// Smooth coordination number for a single atom, given all atom positions
// (bohr) and atomic numbers. Uses the Grimme counting function with the
// standard k1 = 16.0 prefactor and covalent radii scaled by k2 = 4/3 as
// encoded in rcov().
//
// Returns both CN and its analytical Jacobian row for atom A so that the
// outer caller can reuse it in the gradient without re-evaluating distances.
CoordinationNumbers coordination_numbers(const Molecule& mol);

// C_6^AB at the fractional coordination numbers (CN_A, CN_B) via
// Gaussian-weighted interpolation over the H-Ar reference grid. Returns NaN
// when either element is outside the native range.
double interpolate_c6(int ZA, int ZB, double cnA, double cnB);

// ---- Data accessors (defined in dispersion_data.cpp) --------------------

// Atomic r2r4 ratio entering Q_A. Indexed by atomic number (1-based).
// Returns NaN for elements outside the shipped table.
double r2r4(int Z);

// Covalent radius (bohr). Indexed by atomic number (1-based).
double rcov(int Z);

// DFT-D3 ab-initio cut-off radius r0ab(ZA, ZB) in bohr, used by the ATM
// zero-damping function. Symmetric. Covers Z = 1..max_supported_Z();
// returns NaN outside that range.
//
// Same underlying table as python/vibeqc/_d3_r0ab.py (the full 94x94 grid
// needed by the gCP / SRB corrections); this is the Z <= 18 corner of it.
// tests/test_dispersion_atm.py pins the two against each other so they
// cannot drift.
double d3_r0ab(int ZA, int ZB);

// Highest atomic number covered by the shipped tables. Callers can check
// mol atoms against this to fail cleanly with a clear error.
int max_supported_Z();

// ---- Params registry (defined in dispersion_params.cpp) -----------------

// Tabulated D3(BJ) parameters for a DFT functional. Name matching is
// case-insensitive against the conventional labels (e.g. "pbe", "pbe0",
// "b3lyp"). Hyphens are ignored. Returns std::nullopt if the functional has
// no D3(BJ) parameter set in the pinned simple-dftd3 registry.
std::optional<D3BJParams> d3bj_params_for(const std::string& functional);

// ---- Top-level entry point ----------------------------------------------

// Pairwise D3(BJ) dispersion energy (and optionally gradient) for a
// molecule and a chosen functional's damping parameters. Throws
// std::invalid_argument if the molecule contains elements outside the
// shipped table.
DispersionResult compute_d3bj(const Molecule& mol,
                               const D3BJParams& params,
                               bool with_gradient = false);

// ---- Chai-Head-Gordon ("CHG") dispersion — the ωB97X-D "-D" term --------
//
// ωB97X-D (Chai & Head-Gordon, PCCP 10, 6615 (2008)) carries an
// intrinsic empirical dispersion correction of the older "DFT-D2"
// family — distinct from D3(BJ) above (different damping, the
// Grimme-2006 D2 atomic C6 / vdW-radius set):
//
//     E_disp = − s6 · Σ_{A<B} C6_AB / R_AB^6 · f_damp(R_AB)
//     C6_AB     = sqrt(C6_A · C6_B)            (geometric mean)
//     R0_AB     = RvdW_A + RvdW_B              (additive)
//     f_damp(R) = 1 / ( 1 + d · (R / R0_AB)^{-12} )
//
// with s6 = 1.0 and d = 6.0 for ωB97X-D. The C6 / RvdW tables are the
// Grimme-2006 D2 set (J. Comput. Chem. 27, 1787), covering Z = 1..54
// (H–Xe). The algorithm and the tabulated data are transcribed
// verbatim from Psi4's ``libdisp`` ("CHG" scheme) — see
// dispersion_chg.cpp. Geometry-only: no SCF / density coupling, so
// the ωB97X-D total is E_SCF(ωB97X-D XC) + E_disp.
//
// Throws std::invalid_argument if the molecule contains an element
// outside the Z ≤ 54 table.
DispersionResult compute_chg_dispersion(const Molecule& mol,
                                        bool with_gradient = false);

// Highest atomic number covered by the CHG / D2 table (54 = Xe).
int chg_max_supported_Z();

}  // namespace vibeqc
