// Density fitting / resolution-of-the-identity (RI) integral kernels.
//
// References (one paragraph of the math; canonical citations):
//
//   J. L. Whitten, "Coulombic potential energy integrals and
//     approximations", J. Chem. Phys. 58, 4496 (1973). The original
//     Coulomb-metric fit of pair densities in an auxiliary basis.
//   B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, "On the applicability
//     of LCAO-Xα methods to molecules containing transition-metal
//     atoms", Int. J. Quantum Chem. 16, 81 (1979). Variational fit
//     formulation; established robustness of the auxiliary expansion.
//   K. Eichkorn, O. Treutler, H. Öhm, M. Häser, R. Ahlrichs,
//     "Auxiliary basis sets to approximate Coulomb potentials", Chem.
//     Phys. Lett. 240, 283 (1995). Modern atom-centred Gaussian fit
//     bases; foundational paper for RI-J in molecular HF / DFT.
//   F. Weigend, "Hartree-Fock exchange fitting basis sets for H to Rn",
//     J. Comput. Chem. 29, 167 (2008). The def2 JKfit family bundled
//     with vibe-qc (def2-svp-jk … def2-qzvpp-jk).
//
// The four-index electron-repulsion integral
//
//   (μν|λσ) = ∫∫ χ_μ(r1) χ_ν(r1) (1/r12) χ_λ(r2) χ_σ(r2) dr1 dr2
//
// is replaced by the three-index / two-index factorisation
//
//   (μν|λσ) ≈ Σ_{PQ} (μν|P) [V^{-1}]_{PQ} (Q|λσ),
//   V_{PQ}  = (P|Q) =  Coulomb metric on {ω_P}.
//
// Equivalent to fitting the orbital pair density χ_μ χ_ν in an
// auxiliary basis {ω_P} by minimising the Coulomb self-error
// ‖ρ - ρ̃‖_J = ⟨ρ - ρ̃ | r_{12}^{-1} | ρ - ρ̃⟩. The minimiser is the
// well-known
//
//   d^{μν}_P = Σ_Q [V^{-1}]_{PQ} (Q|μν).
//
// Cost: building the four-index ERI is O(n^4) memory and (with
// permutational symmetry) O(n^4) work. RI replaces this with the
// three-index tensor (P|μν) of size O(n_aux · n^2) and one
// O(n_aux^3) Cholesky of V — both built once per geometry — plus
// per-iteration O(n^2 · n_aux) contractions in the J / K / MP2
// builders.
//
// The downstream Cholesky factorisation V = L L^T, the half-
// transformed B-tensor B^P_μν = Σ_Q [L^{-1}]_PQ (Q|μν), and all
// J / K / MO contractions live on the Python side
// (vibeqc.density_fitting.DensityFitting). This C++ module is
// deliberately limited to the two raw libint2-driven kernels.

#pragma once

#include <Eigen/Dense>
#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

#include "basis.hpp"

namespace vibeqc {

// Three-index ERI tensor (n_aux, n_orb, n_orb), dense row-major.
// Symmetric in (μ, ν): (P|μν) = (P|νμ). Both off-diagonal positions
// are filled by compute_3c_eri so callers iterate over the full
// (n_aux, n_orb, n_orb) extent without symmetrisation logic.
struct Eri3D {
    std::vector<double> data;
    std::size_t n_aux = 0;
    std::size_t n_orb = 0;

    double& operator()(std::size_t p, std::size_t mu, std::size_t nu) noexcept {
        return data[(p * n_orb + mu) * n_orb + nu];
    }
    double operator()(std::size_t p, std::size_t mu, std::size_t nu) const noexcept {
        return data[(p * n_orb + mu) * n_orb + nu];
    }
};

// Two-centre Coulomb metric V_{PQ} = (P|Q) on the auxiliary basis.
// Symmetric positive-definite — V is the Gram matrix of {ω_P} under
// the Coulomb inner product, hence SPD whenever {ω_P} is linearly
// independent. libint2 BraKet::xs_xs.
Eigen::MatrixXd compute_2c_eri(const BasisSet& aux);

// Three-centre ERI T_{P,μν} = (P | μν) coupling auxiliary index P to
// orbital pair (μ, ν). libint2 BraKet::xs_xx. The aux basis sits on
// one side; the orbital pair density χ_μ χ_ν is on the other.
Eri3D compute_3c_eri(const BasisSet& orbital, const BasisSet& aux);

namespace detail {

// Apply the Coulomb-metric half transform to a matrix of independent right-
// hand sides. Large transforms use fixed-boundary RHS blocks so separate
// OpenMP workers can call the single-threaded BLAS backend without sharing
// output.
// ``force_blocked`` is reserved for the numerical regression seam.
std::pair<Eigen::MatrixXd, int> solve_df_half_transform(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& T_flat,
    bool force_blocked = false);

}  // namespace detail

// Per-atom contribution to ∂(2-centre Coulomb metric)/∂R contracted
// with a (n_aux, n_aux) weight matrix Ω:
//
//   grad(A, c) = Σ_{PQ} Ω_{PQ} ∂V_{PQ}/∂R_{A,c}
//
// The DF-J gradient picks Ω = -(1/2) γ γ^T (rank-1, where γ = V^{-1} ρ).
// The DF-K gradient picks Ω = +ω_{PQ} = (η^P : η^Q) (full-rank;
// η = V^{-1} M, M^P_{ij} = C_occ^T T^P C_occ). Combined J + K assembly
// uses the linear sum. libint2 BraKet::xs_xs, deriv_order=1.
Eigen::MatrixXd compute_2c_eri_gradient_weighted(
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::MatrixXd& omega);

// Convenience wrapper: contract with γ γ^T (rank-1 weight). Equivalent
// to compute_2c_eri_gradient_weighted(aux, mol, gamma * gamma.T) but
// avoids materialising the n_aux² outer product.
Eigen::MatrixXd compute_2c_eri_gradient_contribution(
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::VectorXd& gamma);

// Per-atom contribution to ∂(3-centre ERI)/∂R contracted with a
// (n_aux, n_orb, n_orb) weight tensor W:
//
//   grad(A, c) = Σ_{P, μν} W^P_{μν} ∂(P|μν)/∂R_{A,c}
//
// The DF-J gradient picks W^P_{μν} = γ_P D_{μν} (rank-1 over the P
// axis). The DF-K gradient picks W^P_{μν} = -2 Y^P_{μν} where
// Y^P = C_occ · η^P · C_occ^T. Combined J + K uses the linear sum.
// libint2 BraKet::xs_xx, deriv_order=1. Storage convention: W laid out
// as a (n_aux, n_orb²) row-major matrix; row P, column ``μ * n_orb +
// ν`` holds W^P_{μν}.
Eigen::MatrixXd compute_3c_eri_gradient_weighted(
    const BasisSet& orbital,
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>& W);

// Convenience wrapper: contract with γ_P · D_{μν} (rank-1 over P).
Eigen::MatrixXd compute_3c_eri_gradient_contribution(
    const BasisSet& orbital,
    const BasisSet& aux,
    const Molecule& mol,
    const Eigen::MatrixXd& D,
    const Eigen::VectorXd& gamma);

// -----------------------------------------------------------------------------
// DensityFitting object: the in-language consumer of the kernels above.
// -----------------------------------------------------------------------------
//
// Intended for use by the C++ SCF drivers (run_rhf / run_uhf / run_rks /
// run_uks) when ``density_fit=true``. Computes V = (P|Q), Cholesky
// V = L L^T, T = (P|μν), and the half-transformed B-tensor
// B^P_{μν} = Σ_Q [L^{-1}]_{PQ} (Q|μν) eagerly at construction. All
// downstream J / K / G builders consume B and are single
// matrix-multiplication contractions over the auxiliary axis P.
//
// A parallel Python implementation lives in
// vibeqc/density_fitting.py. The two are functionally equivalent; the
// Python class is the recommended interface for direct user code,
// MP2 amplitude assembly, and external test code, while this C++ class
// is the SCF drivers' internal workhorse where Python/C++ round-trips
// per iteration are unwelcome.

class DensityFitting {
public:
    // Build V, the Cholesky factor L, T, and B in one shot. Throws
    // std::runtime_error if V is not positive-definite (degenerate
    // auxiliary basis or duplicated atomic centres).
    DensityFitting(const BasisSet& orbital, const BasisSet& aux);

    // RI Coulomb matrix from a density:
    //   γ_P    = Σ_{μν} B^P_{μν} D_{μν}
    //   J_{μν} = Σ_P    B^P_{μν} γ_P
    Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const;

    // Shell-screened Coulomb build.  Same result as build_J(D) but
    // skips shell pairs (ish, jsh) whose density amplitude, weighted
    // by the per-shell B-tensor norm, falls below ``screen_tol``.
    // Reduces the per-iteration J-build cost by 30−70 % on extended
    // systems where the density matrix is sparse in the shell-pair
    // sense (default screen_tol = 1e-10).
    Eigen::MatrixXd build_J_screened(const Eigen::MatrixXd& D,
                                     double screen_tol = 1e-10) const;

    // RI Exchange matrix from a density (no MO factorisation needed).
    // K_{μν} = Σ_λσ D_λσ (μλ|νσ) ≈ Σ_P (B^P · D · B^P)_{μν}
    // (B^P is symmetric, so the triple product is symmetric when D is.)
    Eigen::MatrixXd build_K_density(const Eigen::MatrixXd& D) const;

    // Build the two independent open-shell exchange matrices together.
    // The density-fitting implementation parallelises only work that is
    // independent between auxiliary indices/spins, then preserves each
    // spin's ascending-P accumulation exactly.
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_K_density_pair(
        const Eigen::MatrixXd& D_alpha,
        const Eigen::MatrixXd& D_beta) const;

    // RI Exchange matrix from a list of occupied MOs. Faster than
    // build_K_density when n_occ << n_orb.
    //   B_occ^P_{μi} = Σ_ν B^P_{μν} C_{νi}
    //   K_{μν}       = Σ_{P, i} B_occ^P_{μi} B_occ^P_{νi}
    // C_occ has shape (n_orb, n_occ).
    Eigen::MatrixXd build_K_mo(const Eigen::MatrixXd& C_occ) const;

    // Streaming J / K builders that work directly from the 3-index
    // tensor T and Cholesky factor L, without materialising the full
    // B-tensor.  Block size controls the memory/ recomputation
    // tradeoff; default 64 processes 64 aux functions at a time.
    // Useful when n_aux · n_orb² exceeds available memory.
    Eigen::MatrixXd build_J_streaming(const Eigen::MatrixXd& D,
                                      int block_size = 64) const;
    Eigen::MatrixXd build_K_streaming(const Eigen::MatrixXd& D,
                                      int block_size = 64) const;

    // Composite RHF G builder: G = J(D) - (1/2) K_density(D), matching
    // the convention F = Hcore + G with D = 2 C_occ C_occ^T.
    // Uses K_density rather than K_mo because the SCF damped iterate
    // D_used = damping*D_prev + (1-damping)*D doesn't have a clean MO
    // factorisation. The two-X-flavoured K_mo path is reserved for
    // post-convergence rebuilds and gradient code.
    Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D) const;

    // MO-transformed B-tensor (the substrate for DF-MP2 / DF-CC):
    //   B^P_{pq} = Σ_{μν} C_left[μ,p] B^P_{μν} C_right[ν,q]
    // Returned as a row-major (n_aux, n_left * n_right) matrix; row P,
    // column ``p * n_right + q`` holds B^P_{pq}. The caller can reshape
    // via ``Eigen::Map`` for tensor-style indexing.
    //
    // The standard MP2 use:  ``mo_transform(C_occ, C_vir)`` returns
    // B^P_{ia} for forming (ia|jb) ≈ Σ_P B^P_ia B^P_jb.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        mo_transform(const Eigen::MatrixXd& C_left,
                     const Eigen::MatrixXd& C_right) const;

    // Project a symmetric matrix M through the B-tensor:
    //   result = Σ_P B^P · M · B^P
    // Used by the robust COSX correction to compute K_XvQ and K_QvX.
    Eigen::MatrixXd project_through_B(const Eigen::MatrixXd& M) const;

    // B-metric S_PQ = (B^P : B^Q) = Σ_μν B^P_μν B^Q_μν and its
    // inverse.  Computed from T_flat and L without materialising B.
    // Used to properly normalise projections through the B-tensor.
    Eigen::MatrixXd compute_B_metric() const;

    // α_P = (B^P : M) = tr(B^P · M) for all P simultaneously.
    // Computed from T_flat and L without materialising B.
    // Used by the robust COSX projector.
    Eigen::VectorXd contract_B_with(const Eigen::MatrixXd& M) const;

    // Access T_flat for streaming / robust operations.
    const Eigen::MatrixXd& T_flat() const noexcept { return T_flat_; }

    // Diagnostics / accessors.
    std::size_t n_aux() const noexcept { return n_aux_; }
    std::size_t n_orb() const noexcept { return n_orb_; }
    int last_j_workers_used() const noexcept {
        return last_j_workers_used_.load(std::memory_order_relaxed);
    }
    int last_k_workers_used() const noexcept {
        return last_k_workers_used_.load(std::memory_order_relaxed);
    }
    int last_k_pair_workers_used() const noexcept {
        return last_k_pair_workers_used_.load(std::memory_order_relaxed);
    }
    int last_df_transform_workers_used() const noexcept {
        return last_df_transform_workers_used_;
    }
    int last_df_pack_workers_used() const noexcept {
        return last_df_pack_workers_used_;
    }
    int last_df_unpack_workers_used() const noexcept {
        return last_df_unpack_workers_used_;
    }

    // For analytic-gradient code: the Cholesky factor is the bridge
    // between B-form and (P|μν) form, and ∂V comes in through the chain
    // rule for V^{-1} = (L L^T)^{-1}.
    const Eigen::MatrixXd& cholesky_factor() const noexcept { return L_; }

    // DF-J analytic-gradient contribution evaluated at converged density:
    //
    //   grad_J(A, c) = Σ_P γ_P ∂ρ_P/∂R_{A,c}
    //                  − (1/2) Σ_{PQ} γ_P γ_Q ∂V_{PQ}/∂R_{A,c}
    //
    // where γ = V^{-1} ρ, ρ_P = Σ_{μν} D_{μν} (μν|P). Closed-shell
    // convention: D = 2 C_occ C_occ^T.
    //
    // For pure DFT (α_HF = 0) this is the *complete* DF analytic
    // two-electron gradient.
    //
    // The mol passed here must match the BasisSets the DensityFitting
    // was constructed from; otherwise the shell-to-atom map mismatches.
    Eigen::MatrixXd compute_j_gradient(const Molecule& mol,
                                        const Eigen::MatrixXd& D) const;

    // DF-K analytic-gradient contribution evaluated at converged density:
    //
    //   E_K = -(α_HF/4) tr(D · K_DF)
    //       = -α_HF Σ_{PQ} V^{-1}_{PQ} (M^P : M^Q)
    //
    // with M^P_{ij} = C_occ^T T^P C_occ (closed-shell, D = 2 C_occ
    // C_occ^T baked into the prefactor). Differentiation at fixed D /
    // C_occ gives:
    //
    //   ∂E_K/∂R = +α_HF Σ_{PQ} ω_{PQ} ∂V_{PQ}/∂R
    //              − 2 α_HF Σ_{P, μν} Y^P_{μν} ∂(P|μν)/∂R
    //
    //   ω_{PQ} = (η^P : η^Q),  η = V^{-1} M,
    //   Y^P_{μν} = (C_occ · η^P · C_occ^T)_{μν}.
    //
    // ``alpha_hf`` is the HF-exchange fraction (1.0 for plain HF; the
    // functional's α_HF for hybrid DFT; 0 for pure DFT — pure DFT just
    // calls compute_j_gradient).
    //
    // ``C_occ`` has shape (n_orb, n_occ). For RHF closed-shell, pass
    // ``rhf.mo_coeffs.leftCols(nocc)``.
    Eigen::MatrixXd compute_k_gradient(const Molecule& mol,
                                        const Eigen::MatrixXd& C_occ,
                                        double alpha_hf = 1.0) const;

    // Combined J + K analytic gradient (HF + hybrid DFT case):
    //
    //   ∂E_2e/∂R = ∂E_J/∂R + α_HF · ∂E_K/∂R
    //
    // For pure DFT (α_HF = 0) this just calls compute_j_gradient. For
    // HF (α_HF = 1) and hybrid DFT (0 < α_HF < 1) it folds both
    // contractions into a single 2c + 3c contraction pair, sharing the
    // libint derivative kernel work.
    Eigen::MatrixXd compute_jk_gradient(
        const Molecule& mol,
        const Eigen::MatrixXd& D,
        const Eigen::MatrixXd& C_occ,
        double alpha_hf = 1.0) const;

private:
    // Need access to the orbital + aux BasisSets for gradient evaluation
    // (they hold the libint shell info and the per-shell atom origins).
    // Stored as raw pointers — DensityFitting borrows from caller-owned
    // BasisSets that must outlive it. Same lifetime contract as the
    // existing four-index ERI / SCF code.
    const BasisSet* orbital_basis_ = nullptr;
    const BasisSet* aux_basis_     = nullptr;
    std::size_t n_orb_ = 0;
    std::size_t n_aux_ = 0;
    Eigen::MatrixXd L_;                       // Cholesky factor of V, lower
    Eigen::MatrixXd V_inv_;                   // cached V^{-1} for gradient code
    bool V_inv_built_ = false;                // lazy on first gradient call
    std::vector<Eigen::MatrixXd> B_per_P_;    // length n_aux, each (n_orb, n_orb)
    Eigen::MatrixXd T_flat_;                  // (n_aux, n_orb²) column-major;
                                              // raw 3-index tensor for streaming
    int last_df_transform_workers_used_ = 1;
    int last_df_pack_workers_used_ = 1;
    int last_df_unpack_workers_used_ = 1;
    mutable std::atomic<int> last_j_workers_used_{1};
    mutable std::atomic<int> last_k_workers_used_{1};
    mutable std::atomic<int> last_k_pair_workers_used_{1};
};

}  // namespace vibeqc
