// SCC-DFTB adapter for a frozen semiempirical cyclic-cluster topology.
//
// Extends the DFTB0-SECCM supercell assembly (WS-weighted S, H0) with
// atomic charge self-consistency:
//   1. Supercell gamma: every other atom contributes through its single
//      nearest image inside the central atom's WS cell (fractionally
//      weighted); gamma(a, a) = U_a (on-site Hubbard).
//   2. SCF loop: H^SCC = H0 - 1/2 S (V_A + V_B), V = gamma . dq, where
//      dq are the supercell Mulliken charge fluctuations.
//   3. Mermin free energy per primitive cell =
//      (tr(D H0) - T*S + 1/2 dq.gamma.dq + E_mad + E_rep) / group_order,
//      with E_mad = 0 unless the opt-in embedding is active.
//
// The molecular limit (group_order = 1, zero-translation WS images only)
// reproduces run_scc_dftb on the same molecule. Neutral closed-shell
// clusters by default; no Madelung embedding (neutral cells do not
// need it). Finite electronic temperature is opt-in (default 0 = hard
// Aufbau, bit-identical to the molecular driver): charged polar 1-D
// chains with the converged background-corrected Ewald embedding sit at
// a near-degenerate T=0 frontier, so their SCC response is non-contractive
// under every mixer; a small Fermi temperature smooths the occupation and
// restores convergence (see handovers/HANDOVER_SECCM_ADAPTERS.md).

#pragma once

#include <array>

#include <Eigen/Dense>

#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/seccm/topology.hpp"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

// Roundoff epsilon applied to `aufbau_occupation_deviation` (issue 302).
// The quantity it gates is a dimensionless occupation number of order one, so
// this is negligibility on that number and not a threshold on any physical
// scale; it is recorded on every result as `aufbau_occupation_tolerance` so
// the verdict is reproducible from the record alone, the way
// `finite_torus_gap_tolerance` already makes the gap verdict reproducible.
inline constexpr double kAufbauOccupationTolerance = 1.0e-8;

struct SCCDFTBSECCMResult : ParameterIdentifiedResult {
    double energy = 0.0;  // Mermin free energy per primitive cell.
    // Free electronic band component tr(D H0) - T*S, per primitive cell.
    double e_electronic = 0.0;
    double e_repulsive = 0.0;   // WS-weighted pair repulsion, per cell.
    double e_scc = 0.0;         // 1/2 dq.gamma.dq, per primitive cell.
    double e_madelung = 0.0;    // 1/2 dq.V_mad, per primitive cell (0 without embedding).
    double total_cyclic_energy = 0.0;
    // Raw finite-cluster free electronic band component tr(D H0) - T*S.
    double cyclic_electronic_energy = 0.0;
    double cyclic_repulsive_energy = 0.0;
    double cyclic_scc_energy = 0.0;
    double cyclic_madelung_energy = 0.0;
    double homo_lumo_gap = 0.0;
    // Positive-gap guard outcome, recorded so a converged row is
    // self-describing.  `finite_torus_gap_tolerance` is the epsilon that
    // was actually applied to `homo_lumo_gap`; `gap_guard_waived` is true
    // when the converged frontier gap is at or below that epsilon and the
    // positive-gap requirement was waived because finite electronic
    // temperature resolves the occupation uniquely (Fermi-Dirac).  A
    // waived row is NOT a gapped row: its energy is a Mermin free energy
    // on a numerically degenerate frontier, so it must be screened out of
    // any Aufbau-referenced comparison rather than read as ordinary.
    double finite_torus_gap_tolerance = 0.0;
    bool gap_guard_waived = false;
    // Occupation actually applied to build the accepted density, and how far
    // it is from the integer Aufbau occupation (issue 302).  Weinert and
    // Davenport, Phys. Rev. B 45, 13709 (1992), Eqs. (8) and (10'): a
    // fractional-occupation functional differs from the fixed-integer one by
    // exactly the -T*S term this route subtracts, so a row whose applied
    // occupations are not the Aufbau integers is on a different energy
    // surface and is not comparable with an Aufbau row.
    //   aufbau_occupation_deviation = max_i |f_i - f_i^Aufbau|
    // is zero exactly when the two functionals coincide, which makes it the
    // record to gate on and costs no threshold on any energy scale.  It is
    // exactly 0 on the T = 0 branch by construction.  A value at or above 1
    // means the chemical potential has reached the Aufbau LUMO -- the
    // frontier is unresolved and the row is metallic, not merely thermally
    // broadened -- which is the state the gap guard above cannot see when the
    // gap is small in units of kT but still above the absolute epsilon.
    Eigen::VectorXd occupations;
    double aufbau_occupation_deviation = 0.0;
    double aufbau_occupation_tolerance = 0.0;
    bool aufbau_occupation = false;
    double free_energy = 0.0;   // Mermin A = E - T*S, per primitive cell
    double entropy = 0.0;       // electronic entropy (per cell, 0 at T = 0)
    double smearing_temperature = 0.0;  // Ha (0 = hard Aufbau)
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd mo_coeffs;
    Eigen::MatrixXd density;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd hamiltonian;
    Eigen::VectorXd charges;  // supercell Mulliken fluctuations dq
    Eigen::MatrixXd gradient;  // (n_atoms, 3) fixed-topology force, per cell
    bool has_gradient = false;
    int n_basis = 0;
    int n_occ = 0;
    int n_iter = 0;
    int group_order = 0;
    int n_records = 0;
    bool converged = false;
};

struct SCCDFTBSECCMOptions {
    int max_iter = 500;
    double conv_tol_charge = 1.0e-8;
    double charge_mixing = 0.2;   // fraction of new charges to mix in (0-1]
    bool use_diis = false;        // Pulay DIIS on the charge vector
    int diis_subspace = 6;
    // Modified Broyden quasi-Newton mixing (CP2K/tblite pattern,
    // core/charge_mixer.hpp) as an alternative to Aitken/DIIS. Broyden
    // accelerates the stiff embedded 2-cell charged chain ~5x
    // (see handovers/HANDOVER_SECCM_ADAPTERS.md).
    bool use_broyden = false;
    int broyden_memory = 6;       // history vectors kept
    double broyden_damping = 0.4; // initial mixing fraction (0-1]
    // Opt-in long-range embedding for charged cells: the Madelung/Ewald
    // potential of the supercell Mulliken fluctuations is added to the
    // diagonal of H^SCC and its classical self-energy (1/2 dq . V_mad) to
    // the total energy. Neutral cells do not need it. Kernels: 1-D
    // background-corrected wire Ewald (Parry-type, seccm/ewald_1d.h),
    // and 2-D Parry/Heyes Ewald. The public route rejects 3-D embedding.
    bool madelung = false;
    // Functional form of the SCC-DFTB gamma (maintainer decision D1,
    // 2026-08-28).  Klopman-Ohno is the shipped in-house form and keeps
    // every existing number; Elstner is the published Eq. 17/18 form whose
    // remainder decays exponentially.  It is the form that gives the
    // Wigner-Seitz truncation a thermodynamic limit, and therefore the one
    // 3-D embedding requires (#211).
    ShellGammaForm gamma_form = ShellGammaForm::KlopmanOhno;
    // Opt-in Ewald-split periodic gamma, the GFN2-SECCM ewald_gamma
    // construction for the DFTB family: full bare-Coulomb Ewald plus the
    // WS-folded remainder and the on-site hardness, in one, two or three
    // dimensions.  Mutually exclusive with `madelung`, which adds the
    // long-range tail as a separate embedding potential on top of an
    // untailed gamma instead of summing it into the kernel.
    bool ewald_gamma = false;
    // Fermi electronic temperature in Ha (0 = hard Aufbau, default).
    // A nonzero value smooths frontier-level crossings: charged polar 1-D
    // chains with the converged embedding have a near-degenerate T=0
    // frontier whose response no mixer contracts; the smeared occupation
    // converges and the Mermin free energy A = E - T*S is reported as
    // free_energy (molecular run_scc_dftb convention).
    double electronic_temperature = 0.0;
};

SCCDFTBSECCMResult run_scc_dftb_seccm(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    int group_order,
    const std::array<int, 3>& replicas,
    double geometry_tolerance,
    const SCCDFTBSECCMOptions& opts = {},
    double hermiticity_tolerance = 1.0e-10,
    double gap_tolerance = 1.0e-8,
    bool compute_gradient = false);

// Analytic fixed-topology nuclear gradient for a converged SCC-DFTB-SECCM
// result. Neutral cells use the fixed-charge form (identical in structure to
// the molecular compute_scc_dftb_gradient, which is exact at the SCC fixed
// point):
//   dE/dR = (D . dH0 - W . dS) supercell contraction over the frozen WS
//           record set (M = -W + D . 1/2 (kappa hbar - V_A - V_B) per record)
//         + WS-weighted repulsive derivative
//         + 1/2 dq . dgamma/dR . dq over the live record displacements
// normalized by the finite-group order. Embedded cells (madelung=True) add
// the fixed-charge Madelung derivative 1/2 dq . dM_eff/dR . dq (1-D wire
// Ewald kernel derivative from seccm/ewald_1d.h; 2-D reuses the validated
// indo::_add_madelung_gradient deposit) plus the
// coupled-perturbed SCC charge response: the diagonal Madelung shift
// leaves the energy non-stationary in the Mulliken charges, so the exact
// force solves J . (ddq*/dR) = (d dq_new/dR)|_fixed dq with J = I - A,
// A = d(dq_new)/ddq built by central differences of the charge map, and
// contracts (dE/ddq) = V + V_mad - A^T V + B^T V_mad (B = dt/ddq,
// t the orthogonal-basis atomic traces). The frozen topology keeps image
// labels and weights fixed while displacements track the geometry
// (topology.rebuild_displacements convention; the finite-difference
// regression in tests/test_ccm_semiempirical.py pins this contract).
Eigen::MatrixXd compute_scc_dftb_seccm_gradient(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    const SCCDFTBSECCMResult& result,
    int group_order,
    double geometry_tolerance,
    const SCCDFTBSECCMOptions& opts = {});

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
