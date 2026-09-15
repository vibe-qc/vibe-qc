// DFT+U (Dudarev rotationally-invariant) — C++ Eigen kernel.
//
// Mirrors python/vibeqc/dft_plus_u.py for the path that lives inside the
// C++ SCF Fock builder (Increment 2 — push +U into C++, maintainer call
// 2026-05-25). The Python module remains the user-facing surface and
// the post-SCF analysis path; this kernel is the one the SCF iteration
// calls each iter.
//
// Spin convention: per-spin. For closed-shell RHF/RKS the caller passes
// P_σ = P_total / 2 (so the occupation matrix eigenvalues land in [0,1]
// rather than [0,2]); the SCF then multiplies the returned energy by 2
// and uses the same V_U for both spins (V_U_α = V_U_β by closed-shell
// symmetry, since n_α = n_β).
//
// Definitions:
//
//     n^A_l_{mm'} = (S P_σ S)_{(A,l,m),(A,l,m')}
//     E_U_σ       = Σ_A (U_eff / 2) · ( tr n^A_l − tr (n^A_l)² )
//     V_U^A_{mm'} = U_eff · ( δ_{mm'} / 2 − n^A_l_{mm'} )
//
// V_U^A is rank-(2l+1) per centre; it is folded back into the AO basis
// only on the (A,l) block, which is what `apply_to_ao_basis` does
// below. The sum E_U_σ is over the spin σ this kernel is called for.
//
// Reference:
//     Dudarev, Botton, Savrasov, Humphreys, Sutton, PRB 57, 1505 (1998).

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace vibeqc {

// A single +U-active (atom, angular-shell) channel. Mirrors
// python/vibeqc/dft_plus_u.py::HubbardSite but stores U_eff already in
// Hartree atomic units (the eV → Hartree conversion happens at the
// Python-binding boundary, where the user surface lives).
struct HubbardSiteCxx {
    int atom_index = 0;      // zero-based index into Molecule.atoms()
    int l = 0;               // angular momentum (0=s, 1=p, 2=d, 3=f)
    double U_eff_au = 0.0;   // U − J, already in Hartree
};

// Output of one per-spin +U evaluation. ``V`` is the additive AO-basis
// Fock contribution (nbf × nbf, sparse — non-zero only on the (A,l)
// blocks of the configured sites); ``energy`` is E_U_σ for this spin
// channel.
struct DftPlusUResult {
    Eigen::MatrixXd V;
    double energy = 0.0;
};

// Compute the per-spin +U energy + AO-basis Fock contribution.
//
//   sites      — the configured (atom, l, U_eff) tuples.
//   ao_groups  — parallel array; ao_groups[i] is the list of AO indices
//                that belong to sites[i] (i.e. atom sites[i].atom_index,
//                angular momentum sites[i].l). Pre-computed once at SCF
//                setup by python/vibeqc/dft_plus_u.py::ao_group_indices
//                so we don't re-walk the basis every iteration.
//   P          — per-spin density matrix (nbf × nbf, symmetric).
//   S          — AO overlap matrix (nbf × nbf, symmetric positive
//                definite).
//
// Returns the per-spin DftPlusUResult{V, energy}. Empty ``sites`` →
// zero matrix + zero energy (a fast no-op so callers can unconditionally
// invoke us regardless of whether the user enabled +U).
DftPlusUResult compute_dft_plus_u(
    const std::vector<HubbardSiteCxx>& sites,
    const std::vector<std::vector<int>>& ao_groups,
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXd& S);

// ---------------------------------------------------------------------
// Multi-k periodic +U (Increment 4c).
// ---------------------------------------------------------------------
//
// Closed-shell multi-k Dudarev +U. The AO occupation matrix is averaged
// over the k-mesh:
//
//     n^A_l_{mm'} = (1/2) Σ_k w_k Re[(S(k) P(k) S(k))_{(A,l),mm'}]
//
// where the factor of (1/2) is the per-spin convention (P(k) is the
// total density). Energy:
//
//     E_U_total = U_eff × (tr n^A_l − tr (n^A_l)²)   (closed-shell, sum
//                                                    over both spins)
//
// V_AO_σ = U_eff (½ δ − n^A_l) is the per-spin Dudarev potential on
// the (A,l) blocks, scattered into the full AO basis. The variational
// per-k Fock contribution is ``S(k) V_AO_σ S(k)`` — k-dependent
// through ``S(k)``, but built from the k-independent ``V_AO_σ``. The
// caller is responsible for the per-k sandwich.
struct DftPlusUMultiK {
    Eigen::MatrixXd V_AO_per_spin;   // real, k-independent, (nbf × nbf)
    double energy_total = 0.0;       // closed-shell sum-over-spins E_U
};

DftPlusUMultiK compute_dft_plus_u_multi_k_closed_shell(
    const std::vector<HubbardSiteCxx>& sites,
    const std::vector<std::vector<int>>& ao_groups,
    const std::vector<Eigen::MatrixXcd>& S_k,
    const std::vector<Eigen::MatrixXcd>& P_k,
    const std::vector<double>& weights);

// Per-spin multi-k kernel — for UHF / UKS callers. ``P_sigma_k`` is
// the per-spin density (the caller does NOT halve P_total). Returns
// the per-spin V_AO and the per-spin Dudarev energy
// ``E_σ = (U_eff/2)(tr n_σ − tr n_σ²)``. Open-shell callers invoke
// this twice (once per spin) and sum the energies; closed-shell
// callers can use this with ``P_σ = P_total/2`` and double the
// energy themselves (or use the convenience
// ``..._closed_shell`` variant above).
DftPlusUMultiK compute_dft_plus_u_multi_k_per_spin(
    const std::vector<HubbardSiteCxx>& sites,
    const std::vector<std::vector<int>>& ao_groups,
    const std::vector<Eigen::MatrixXcd>& S_k,
    const std::vector<Eigen::MatrixXcd>& P_sigma_k,
    const std::vector<double>& weights);

}  // namespace vibeqc
