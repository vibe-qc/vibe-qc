// OMx Fock matrix builder — proper OM1/OM2/OM3 Hamiltonian.
//
// Implements the orthogonalization-corrected NDDO Hamiltonians
// from the Thiel group:
//   OM1: Kolb & Thiel 1993, J. Comput. Chem. 14, 775.
//   OM2: Weber & Thiel 2000, Theor. Chem. Acc. 103, 495.
//   OM3: Scholten 2003, PhD thesis, Univ. Düsseldorf.
//   Dral 2016: J. Chem. Theory Comput. 12, 1082 — definitive formalism +
//   parameters; equation numbers below refer to this paper.
//
// Key elements (see omx_fock.cpp for the full equation-by-equation map):
//   1. Secular equation F C = C ε solved directly in the NDDO basis (eq 2);
//      the overlap matrix enters only the explicit correction terms.
//   2. Resonance integrals β_μλ = ½(β_μ+β_λ)√R·exp[−(α_μ+α_λ)R²] (eq 17),
//      with optional X–H pair-specific parameters.
//   3. Orthogonalization corrections V^ORT: one-centre (eq 8, F1/F2) and,
//      for OM2/OM3, three-centre (eq 9, G1/G2).
//   4. Semiempirical ECP for OM2/OM3 (Weber 2000 eqs 60/63).
//   5. Spherical Klopman–Ohno γ for all two-electron integrals and the
//      core–core repulsion E_AB = Z_A^c Z_B^c γ_AB (eq 20 with f_KO/R → γ;
//      no MNDO-style empirical exponential repulsion terms).

#pragma once

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/methods/nddo/pm6_fock.hpp"
#include "vibeqc/semiempirical/methods/nddo/omx_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// ---------------------------------------------------------------------------
// OMx resonance integral β_μν in the molecular frame (Ha).
//
// Radial kernel (eq 17 of Dral 2016): ½(β_μ+β_λ)·√R·exp[−(α_μ+α_λ)R²],
// Slater–Koster rotated with the same direction-cosine conventions as
// nddo_sto_overlap(), with the σ channels phased so the resonance matrix is
// elementwise opposite in sign to the overlap matrix.
// ---------------------------------------------------------------------------

double omx_resonance_integral(
    int t_mu, int t_nu,         // orbital types: 0=s, 1=px, 2=py, 3=pz
    int Z_mu, int Z_nu,         // atomic numbers
    double R,                   // interatomic distance (bohr)
    double dx, double dy, double dz,  // pos(mu-atom) − pos(nu-atom)
    const OMxParameterSet& params);

// ---------------------------------------------------------------------------
// OMx valence overlap matrix: STO overlaps with the per-element exponent ζ
// (Dral 2016 Tables 1–3); one-centre blocks are identity.  Used only inside
// the ORT/ECP correction terms — never in the secular equation.
// ---------------------------------------------------------------------------

Eigen::MatrixXd build_omx_overlap(
    const Molecule& mol,
    const OMxParameterSet& params,
    const std::vector<int>& ao_Z,
    const std::vector<int>& ao_atom,
    const std::vector<int>& ao_type);

// ---------------------------------------------------------------------------
// Closed-shell OMx SCF (v2): F C = C ε in the NDDO basis with Pulay DIIS.
// ---------------------------------------------------------------------------

PM6Result run_omx_v2(
    const Molecule& mol,
    const OMxParameterSet& params,
    int max_iter = 100,
    double conv_tol = 1e-7);

// ---------------------------------------------------------------------------
// Unrestricted OMx SCF (v2): UHF on the same corrected OMx Hamiltonian,
// stacked-spin Pulay DIIS.
// ---------------------------------------------------------------------------

UPM6Result run_uomx_v2(
    const Molecule& mol,
    const OMxParameterSet& params,
    int max_iter = 100,
    double conv_tol = 1e-7);

// Finite-difference OMx gradients (v2)
Eigen::MatrixXd compute_omx_v2_gradient_fd(
    const Molecule& mol,
    const OMxParameterSet& params,
    double h = 0.001,
    int max_iter = 100,
    double conv_tol = 1e-7);

Eigen::MatrixXd compute_uomx_v2_gradient_fd(
    const Molecule& mol,
    const OMxParameterSet& params,
    double h = 0.001,
    int max_iter = 100,
    double conv_tol = 1e-7);

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
