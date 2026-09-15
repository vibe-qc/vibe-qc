// Direct (integral-driven) Fock-build kernels for molecular SCF.
//
// Replaces the in-core 4-index ERI tensor (cpp/src/integrals.cpp::compute_eri
// + cpp/src/fock.cpp::build_*) with on-the-fly libint2 quartet evaluation
// inside an 8-fold-symmetric shell-quartet loop, screened by the strict
// Cauchy-Schwarz bound |⟨μν|λσ⟩| ≤ Q(s1, s2) · Q(s3, s4). Memory is
// O(n_shells²) for the precomputed Q matrix instead of O(n_basis⁴) for
// the ERI tensor. The closure point at ~250 basis functions / def2-SVP
// is the n-hexadecane OOM cell in benchmarks/orca_vs_vibeqc_speed.md.
//
// Three entry points mirror the JKBuilder ABC (cpp/include/vibeqc/jk_builder.hpp):
//   * direct_compute_j   — J(D)_{μν} = Σ_λσ D_λσ (μν|λσ)
//   * direct_compute_k   — K(D)_{μν} = Σ_λσ D_λσ (μλ|νσ)
//   * direct_compute_g_rhf — fused G = J − ½ · α_HF · K, one libint call
//                            per surviving quartet (≈ 2× cheaper than
//                            separate J + K when α_HF ≠ 0).
//
// All three accept a precomputed Q matrix (from compute_schwarz_factors
// in cpp/include/vibeqc/schwarz.hpp). The caller is expected to build Q
// once at JKBuilder construction and reuse it across SCF iterations —
// Q depends only on the basis, not on the density.
//
// Convention: D is vibe-qc's closed-shell spin density (D = 2·C_occ·C_occ^T),
// matching cpp/src/fock.cpp::build_fock_g + the JKBuilder ABC. The
// permutation-symmetry coefficients inside the kernel absorb the
// resulting factor of 2 vs the libint2 example's orbital-density form.

#pragma once

#include <Eigen/Dense>
#include <vector>

#include "basis.hpp"

namespace vibeqc {

// Sorted shell-pair list used by the direct-Fock quartet visitor for
// ORCA-style sort-and-break enumeration. Basis-only — caller should
// build once via ``build_q_sorted_pairs(shells, Q)`` and reuse across
// SCF iterations.
struct DirectShellPair {
    int s1;
    int s2;
    double q;       // Q(s1, s2) — the per-pair Schwarz factor.
};

// Build the K-sorted shell-pair list: enumerates all (s1, s2) with
// s1 ≥ s2 and sorts by descending Q. The inner-loop ``break`` in the
// direct-Fock visitor consumes this — once we walk below the
// threshold the rest of the list is also dropped.
//
// Cost: O(n_shells² · log n_shells²); basis-only, cached on the
// JKBuilder.
std::vector<DirectShellPair> build_q_sorted_pairs(
    const BasisSet& basis, const Eigen::MatrixXd& Q);

// Coulomb matrix J(D)_{μν} = Σ_{λσ} D_{λσ} (μν|λσ) via 8-fold-symmetric
// on-the-fly libint quartet evaluation + Schwarz screening. `Q` is the
// per-shell-pair Cauchy-Schwarz factor matrix (n_shells × n_shells)
// from compute_schwarz_factors; `sorted_pairs` is the corresponding
// K-sorted pair list from ``build_q_sorted_pairs``.
// `schwarz_threshold > 0` enables screening; `0` evaluates every
// quartet (useful as a parity reference).
Eigen::MatrixXd direct_compute_j(const BasisSet& basis,
                                 const Eigen::MatrixXd& D,
                                 const Eigen::MatrixXd& Q,
                                 const std::vector<DirectShellPair>& sorted_pairs,
                                 double schwarz_threshold);

// Exchange matrix K(D)_{μν} = Σ_{λσ} D_{λσ} (μλ|νσ) — same kernel
// shape as direct_compute_j but with the K-contraction permutations.
Eigen::MatrixXd direct_compute_k(const BasisSet& basis,
                                 const Eigen::MatrixXd& D,
                                 const Eigen::MatrixXd& Q,
                                 const std::vector<DirectShellPair>& sorted_pairs,
                                 double schwarz_threshold);

// Long-range (erf-attenuated) exchange matrix
// K_erf(D)_{μν} = Σ_{λσ} D_{λσ} (μλ| erf(ω·r₁₂)/r₁₂ |νσ).
// Identical contraction to ``direct_compute_k`` but with libint's
// ``Operator::erf_coulomb`` kernel — the long-range piece of a
// range-separated hybrid's exact exchange (ωB97X, ωB97X-D, …). The
// Coulomb Schwarz factors ``Q`` are a valid conservative screen
// (erf(ω·r)/r ≤ 1/r). ``omega`` must be > 0.
Eigen::MatrixXd direct_compute_k_erf(const BasisSet& basis,
                                     const Eigen::MatrixXd& D,
                                     const Eigen::MatrixXd& Q,
                                     const std::vector<DirectShellPair>& sorted_pairs,
                                     double schwarz_threshold,
                                     double omega);

// Closed-shell fused G = J − ½ · α_HF · K via a single quartet loop.
// One libint call per surviving quartet accumulates into both the J
// and K halves of G simultaneously; ≈ 2× cheaper than calling
// direct_compute_j + direct_compute_k separately. α_HF is the
// fraction of HF-exchange in the functional (1.0 for plain HF/RHF,
// 0.20 for B3LYP, 0.0 for pure GGA — for which the K piece is
// skipped entirely).
Eigen::MatrixXd direct_compute_g_rhf(const BasisSet& basis,
                                     const Eigen::MatrixXd& D,
                                     const Eigen::MatrixXd& Q,
                                     const std::vector<DirectShellPair>& sorted_pairs,
                                     double schwarz_threshold,
                                     double alpha_hf);

}  // namespace vibeqc
