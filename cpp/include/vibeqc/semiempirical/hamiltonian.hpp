// Hamiltonian builder for DFTB0 / SCC-DFTB.
//
// Stage 1: DFTB0 (non-SCC). S and H⁰ are built once.
// Stage 3: SCC adds Mulliken charge analysis, γ matrix, and
// charge-dependent diagonal correction H^{SCC}.

#pragma once

#include <Eigen/Dense>

#include "vibeqc/basis.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/semiempirical/parameters.hpp"

namespace vibeqc {
namespace semiempirical {

class SemiempiricalHamiltonianBuilder {
public:
    SemiempiricalHamiltonianBuilder() = default;

    // ---- Overlap --------------------------------------------------------
    static Eigen::MatrixXd build_overlap(const BasisSet& basis);

    // ---- H⁰ Hamiltonian (non-SCC) ---------------------------------------
    static Eigen::MatrixXd build_hamiltonian_zero(
        const BasisSet& basis,
        const Eigen::MatrixXd& S,
        const Molecule& mol,
        const SemiempiricalParameters& params);

    // ---- Valence electron count ------------------------------------------
    //
    // DFTB fills the valence-minimal basis with VALENCE electrons only:
    // N_val = Σ_A n_val(A) − molecular charge. Core electrons are implicit
    // in the on-site energies / repulsive potential and must NOT be
    // counted into the band filling (mol.n_electrons() counts all Z).
    static int valence_electron_count(
        const Molecule& mol,
        const SemiempiricalParameters& params);

    // ---- SCC charge analysis --------------------------------------------
    //
    // Mulliken atomic charges from density D and overlap S.
    //   q_A = n_val(A) − Σ_{μ∈A} (D S)_{μμ}
    //   Δq_A = q_A − q_A⁰  (charge fluctuation; q_A⁰ = 0 for neutrals)
    //
    // Sign convention: Δq_A > 0 means the atom is electron-DEFICIENT
    // (cationic) — the conventional chemical Mulliken charge. Note this
    // is the NEGATIVE of the Δq in Elstner et al., Phys. Rev. B 58,
    // 7260 (1998), who use excess electron population; every formula
    // below that is odd in Δq therefore carries a flipped sign.
    //
    // Returns a vector of length n_atoms with Δq_A values.
    static Eigen::VectorXd mulliken_charges(
        const BasisSet& basis,
        const Eigen::MatrixXd& D,
        const Eigen::MatrixXd& S,
        const Molecule& mol,
        const SemiempiricalParameters& params);

    // ---- gamma matrix (Coulomb interaction between atomic charges) ------
    //
    // One site per atom through the shared kernel of
    // semiempirical/core/periodic_gamma.hpp, so molecular, Gamma, k-point
    // and SECCM SCC-DFTB run one functional form under one method name
    // (maintainer decision D1, 2026-08-28).
    //
    //   KlopmanOhno (default): gamma_AB = 1 / sqrt(R^2 + eta_AB^2) with
    //     eta_AB = 0.5 (1/U_A + 1/U_B) -- the in-house form.
    //   Elstner: gamma_AB = 1/R - S(tau_A, tau_B, R), tau = 16/5 U --
    //     Elstner et al., Phys. Rev. B 58, 7260 (1998), Eqs. 17/18, the
    //     form the published method uses.
    //
    // For A = B both give gamma_AA = U_A (on-site Hubbard).
    //
    // Returns an n_atoms x n_atoms symmetric matrix.
    static Eigen::MatrixXd gamma_matrix(
        const Molecule& mol,
        const SemiempiricalParameters& params,
        ShellGammaForm form = ShellGammaForm::KlopmanOhno);

    // ---- SCC Hamiltonian correction -------------------------------------
    // ... (documentation unchanged) ...
    static Eigen::MatrixXd build_scc_hamiltonian(
        const Eigen::MatrixXd& H0,
        const Eigen::MatrixXd& S,
        const BasisSet& basis,
        const Molecule& mol,
        const SemiempiricalParameters& params,
        const Eigen::MatrixXd& gamma,
        const Eigen::VectorXd& delta_q);

    // ---- DFTB3: gamma3 matrix, third-order potential -------------------
    static Eigen::MatrixXd gamma3_matrix(
        const Molecule& mol,
        const SemiempiricalParameters& params);
    static Eigen::VectorXd third_order_potential(
        const Eigen::MatrixXd& gamma3,
        const Eigen::VectorXd& delta_q);
    static Eigen::VectorXd scc_potential_dftb3(
        const Eigen::MatrixXd& gamma,
        const Eigen::MatrixXd& gamma3,
        const Eigen::VectorXd& delta_q);

    // On-site energy accessor for a given atom.
    static std::vector<double> on_site_energies_for_atom(
        int atom_index, const BasisSet& basis,
        const SemiempiricalParameters& params,
        int Z_atom);
};

}  // namespace semiempirical
}  // namespace vibeqc
