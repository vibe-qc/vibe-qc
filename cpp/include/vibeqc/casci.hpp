// Direct determinant CAS-CI engine: string-based sigma builds + Davidson.
//
// Solves the active-space CI problem H c = E c without materializing the
// determinant Hamiltonian, in the style of the determinant-FCI algorithms of
// Knowles & Handy, Chem. Phys. Lett. 111, 315 (1984) and Olsen, Roos,
// Jørgensen & Jensen, J. Chem. Phys. 89, 2185 (1988): the CI vector is a
// matrix C(I_α, I_β) over α/β occupation strings, and σ = H·C is assembled
// from single-excitation string tables contracted with the integrals.
//
// In spin-summed form with Ê_pq = Σ_σ a†_{pσ} a_{qσ} and chemist-notation
// integrals (pq|rs):
//
//   Ĥ = Σ_pq h'_pq Ê_pq + ½ Σ_pqrs (pq|rs) Ê_pq Ê_rs ,
//   h'_pq = h_pq − ½ Σ_r (pr|rq)
//
// (the one-body correction absorbs the δ_qr Ê_ps term of
// ê_pqrs = Ê_pq Ê_rs − δ_qr Ê_ps).  σ is then evaluated per determinant
// chunk as
//
//   D_rs = Ê_rs C            (gather over single-excitation tables)
//   G_pq = Σ_rs (pq|rs) D_rs (one GEMM per chunk)
//   σ   += Ê(h') C + ½ Σ_pq Ê_pq G_pq   (scatter over the same tables)
//
// Conventions are pinned to the Python determinant engine in
// python/vibeqc/solvers/ (the validation oracle):
//   * strings enumerate occupations in itertools.combinations
//     (lexicographic) order; the determinant index is I = I_α·n_β + I_β
//     (α-major), matching solvers._determinant.generate_determinants;
//   * fermionic phases are per-spin-sector Jordan-Wigner phases (number of
//     set bits below the operator's target within that sector's string),
//     matching solvers._rdm._apply_aq_adp_spin / solvers._slater_condon;
//   * RDM conventions match solvers._rdm.make_rdm12 (PySCF
//     fci.direct_spin1): rdm1[p,q] = ⟨a†_p a_q⟩ (spin-summed) and
//     rdm2[p,q,r,s] = ⟨a†_p a†_r a_s a_q⟩ = ⟨Ê_pq Ê_rs⟩ − δ_qr ⟨Ê_ps⟩.
//
// The eigensolver is a block Davidson with the determinant-diagonal
// preconditioner (Davidson, J. Comput. Phys. 17, 87 (1975)).
//
// Scope: the active space is at most 31 spatial orbitals (strings live in
// one uint64 per spin sector; practical sizes are bounded by ndet anyway).

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace vibeqc {

struct CASCIDirectOptions {
    int nroots = 1;
    // Davidson residual-norm convergence per root.
    double tol = 1e-9;
    int max_iter = 200;
    // Subspace collapse threshold (in vectors).
    int max_subspace = 24;
    // Determinant-index chunk width for the σ two-electron pass.  Bounds
    // the D/G work buffers at n_pair × chunk doubles per thread team.
    int chunk = 8192;
};

struct CASCIDirectResult {
    Eigen::VectorXd eigenvalues;  // (nroots) active-space CI eigenvalues
                                  // (no core/nuclear constant included)
    Eigen::MatrixXd ci;           // (ndet, nroots), α-major determinant order
    bool converged = false;
    int n_iter = 0;
    long long n_det = 0;
};

// Lowest-``nroots`` eigenpairs of the active-space CI Hamiltonian.
//
// ``h1`` is the (dressed) active one-electron matrix (n_act × n_act);
// ``eri_chem`` the active two-electron integrals in CHEMIST notation
// (pq|rs), flattened row-major to length n_act^4.  ``n_alpha``/``n_beta``
// are the active α/β electron counts.
//
// ``guess`` (optional, ndet × k) warm-starts the Davidson subspace —
// e.g. the previous macro-iteration's CI vector(s) in a CASSCF loop.
// Guess columns are orthonormalized and noise-dressed like the default
// determinant seeds; an empty matrix selects the cold start.
CASCIDirectResult casci_direct_solve(
    const Eigen::MatrixXd& h1,
    const std::vector<double>& eri_chem,
    int n_act,
    int n_alpha,
    int n_beta,
    const CASCIDirectOptions& opts = {},
    const Eigen::MatrixXd& guess = Eigen::MatrixXd());

// Spin-summed 1- and 2-RDM of a CI vector in the engine's determinant
// order (conventions above).  Returns (rdm1 [n_act²], rdm2 [n_act⁴]),
// both row-major.
void casci_direct_rdm12(const Eigen::VectorXd& ci,
                        int n_act,
                        int n_alpha,
                        int n_beta,
                        Eigen::MatrixXd& rdm1,
                        std::vector<double>& rdm2);

// σ = H·c for one vector (exposed for testing / power iterations).
Eigen::VectorXd casci_direct_sigma(const Eigen::VectorXd& ci,
                                   const Eigen::MatrixXd& h1,
                                   const std::vector<double>& eri_chem,
                                   int n_act,
                                   int n_alpha,
                                   int n_beta,
                                   int chunk = 8192);

}  // namespace vibeqc
