// Cell-pair double-sum periodic COSX-K (M3b-1).
//
// Seminumerical exchange for the Γ-point cell-diagonal-density model
//
//   K_{μν} = Σ_{g,p} Σ_{λσ} D_{λσ} (μ_0 λ_p | ν_g σ_p)
//
// — the same object the direct-ERI reference
// ``build_jk_gamma_molecular_limit`` (periodic_fock.cpp) evaluates:
// bra pair (μ home, λ in cell p), ket pair (ν in cell g, σ in cell p),
// density connecting same-cell λσ pairs only ("cell-diagonal density",
// the Γ model valid when the density matrix decays between cells).
// Every lattice index is damped by AO-pair overlap, so the double sum
// is absolutely convergent — unlike the Γ-folded single-sum class
// (compute_cosx_k's ``image_cells`` mode, cosx.hpp).
//
// COSX quadrature on the bra integral with a home-cell Becke grid
// (vibe-qc periodic extension of Neese, Wennmohs, Hansen & Becker,
// Chem. Phys. 356, 98 (2009), doi:10.1016/j.chemphys.2008.10.036, §2):
//
//   (μ_0 λ_p | ν_g σ_p) ≈ Σ_G w_G χ_μ(r_G) χ_λ(r_G − p)
//                              · A^{(δ=g−p)}_{νσ}(r_G − p)
//
//   A^{(δ)}_{νσ}(C) = ∫ χ_ν(r' − δ) χ_σ(r') / |r' − C| dr'
//
// i.e. per grid point and bra cell p: density-weighted shifted AOs
// Dχ_p = D · χ(r_G − p), and per ket cell g the analytic
// nuclear-attraction-style block over the (ν shifted by δ, σ home)
// shell pair with the pseudo-nucleus at C = r_G − p, accumulated as
//
//   F_ν(r_G) += Σ_σ A^{(δ)}_{νσ}(r_G − p) · (Dχ_p)_σ
//   K_{μν}  += w_G · χ_μ(r_G) · F_ν(r_G)
//
// The δ-dependent inputs (shifted shells, primitive-pair data,
// Schwarz tables) are basis-only and SCF-invariant — build them once
// per geometry via ``build_cosx_cell_pair_caches`` and reuse across
// iterations. The δ difference set is Schwarz-screened at build time:
// shifts beyond the AO-pair overlap range drop out wholesale, which
// is what bounds the per-point (p, g) work for tight cells.
//
// This kernel is the M3b-1 building block for multi-k periodic COSX
// (handovers/HANDOVER_RIJCOSX_M3A.md § "M3b direction"): the multi-k exchange
// consumes the same machinery with per-cell-separation density blocks
// D(g) in place of the cell-diagonal density, and emits per-cell K(h)
// blocks instead of the Γ-folded sum.
//
// Current scope/assumptions (documented limits, not silent ones):
//   * D is assumed symmetric (the kernel contracts D·χ_p and
//     symmetrises K at the end). Generalised non-symmetric densities
//     (response/post-HF intermediates) are an M3b interface item.
//   * Per-cell shifted-χ tables are pre-evaluated over the full grid
//     (memory ≈ n_active_cells × n_pts × n_bf doubles). Fine at
//     validation scale; production chunking is M3b-2 work.
//   * Spin handled by the caller (pass D_α / D_β separately).

#pragma once

#include <Eigen/Dense>

#include <vector>

#include <libint2/shell.h>

#include "basis.hpp"
#include "cosx_kernel.hpp"
#include "grid.hpp"
#include "lattice_sum.hpp"

namespace vibeqc {

// Basis-only / SCF-invariant caches for ``compute_cosx_k_cell_pair``.
// Built once per geometry by ``build_cosx_cell_pair_caches``.
struct CosxCellPairCaches {
    // Coulomb-kernel range separation the caches were built for.
    // 0 = full 1/r kernel; ω > 0 = erfc(ω·r)/r short-range kernel
    // (M3b-4a). The Schwarz tables below use the matching metric, and
    // the kernels consuming these caches read the value from here —
    // one source of truth, no mismatched screens.
    double omega = 0.0;

    // The cell list both lattice indices (bra p, ket g) range over.
    // Must contain the zero cell (``direct_lattice_cells`` always
    // puts it first).
    std::vector<LatticeCell> cells;

    // Surviving relative shifts δ = g − p (Schwarz-screened pairwise
    // difference set). deltas[k] is the Cartesian shift in bohr.
    std::vector<Eigen::Vector3d> deltas;

    // Row-major (n_cells × n_cells) map: delta_index[p * n_cells + g]
    // = index into ``deltas`` for δ = cells[g] − cells[p], or −1 when
    // that shift was Schwarz-dropped (no (ν_δ, σ_0) pair survives).
    std::vector<int> delta_index;

    // Per-δ: the home shell list translated by +δ (the ket-side ν
    // shells of A^{(δ)}).
    std::vector<std::vector<libint2::Shell>> shells_delta;

    // Per-δ: primitive-pair data for (ν_δ, σ_0);
    // ``pp_delta[k].pairs[s_ν * n_shells + s_σ]``.
    std::vector<PrimitivePairCache> pp_delta;

    // Per-δ Schwarz tables: schwarz_delta[k](s_ν, s_σ) =
    // sqrt(max |(ν_δ σ_0 | ν_δ σ_0)|) — the Häser-Ahlrichs factor for
    // the shifted pair, used in the per-point pair screen.
    std::vector<Eigen::MatrixXd> schwarz_delta;

    // SR-only (omega > 0; empty otherwise): per-δ pair cutoff radii.
    // r_cut_delta[k](s_ν, s_σ) is the pseudo-nucleus distance beyond
    // which the SR A-block of the (ν_δ, σ_0) pair falls below the
    // screening floor — tabulated at cache build by probing the
    // validated kernel itself at increasing distances (so the table
    // automatically reflects kernel range AND pair-density extent;
    // no analytic extent model). The engine skips a pair when the
    // min shell distance to the pseudo-nucleus exceeds its radius,
    // and a whole (c_σ, δ) combination when both min-distances exceed
    // r_cut_max_delta[k] — the lever that makes basis-extent-sized
    // cell domains affordable (M3b-6).
    std::vector<Eigen::MatrixXd> r_cut_delta;
    std::vector<double> r_cut_max_delta;
};

// Build the per-δ caches for ``compute_cosx_k_cell_pair``. The δ
// difference set is screened at ``schwarz_drop_tol``: a shift whose
// whole Schwarz table is below the tolerance contributes < tol × ‖Dχ‖
// per grid point and is dropped from the cache (its delta_index
// entries are −1). Throws if ``cells`` is empty or lacks the zero
// cell.
//
// ``omega`` selects the Coulomb kernel (0 = full 1/r; ω > 0 = the
// erfc(ω·r)/r short-range kernel). With ω > 0 the Schwarz tables use
// the matching erfc metric. Note the Schwarz factor is the pair
// density's SELF-interaction — it decays with |δ| through the pair
// overlap only, so the δ set matches the full-kernel one; the SR
// locality appears in the K(g) block decay (bra-ket coupling at
// range 1/ω), not in the cache size.
CosxCellPairCaches build_cosx_cell_pair_caches(
    const BasisSet& basis,
    const std::vector<LatticeCell>& cells,
    double schwarz_drop_tol = 1e-10,
    double omega = 0.0);

// Cell-pair double-sum COSX exchange (formula at the top of this
// file). Returns the symmetrised, Q-corrected (n_bf, n_bf) K.
//
// ``q_cached``: Q-junction from ``build_cosx_q`` (basis + grid
// invariant — pass it on the hot path; an empty matrix triggers a
// per-call rebuild). ``boys_table``: from ``build_boys_table``;
// nullptr triggers a per-call local build.
//
// Validation anchor: agrees with the direct-ERI
// ``build_jk_gamma_molecular_limit`` K on the same cell list to COSX
// grid-quadrature accuracy, and reduces to the molecular
// ``compute_cosx_k`` exactly when ``caches.cells`` is the home cell
// only (tests/test_periodic_rijcosx.py, M3b-1 block).
//
// Implemented as a thin wrapper over ``compute_cosx_k_blocks``: the
// cell-diagonal density is the single-block set P = {0 ↦ D}, and the
// Γ-folded K is the sum of the raw K(g) blocks, symmetrised and
// Q-corrected.
Eigen::MatrixXd compute_cosx_k_cell_pair(
    const BasisSet& basis,
    const Eigen::MatrixXd& D,
    const Grid& cosx_grid,
    const CosxCellPairCaches& caches,
    const Eigen::MatrixXd& q_cached = Eigen::MatrixXd(),
    const BoysTable* boys_table = nullptr);

// Real-space exchange blocks from real-space density blocks (M3b-2)
// — the multi-k periodic COSX engine. Evaluates, for every output
// cell g in ``caches.cells``,
//
//   K(g)_{μν} = Σ_{c_λ, c_σ ∈ caches.cells} Σ_{λσ}
//                   P(c_σ − c_λ)_{λσ} (μ_0 λ_{c_λ} | ν_g σ_{c_σ})
//
// — the same object and the same summation domain as the direct-ERI
// reference ``build_jk_2e_real_space_explicit`` (periodic_fock.cpp):
// density blocks are looked up by integer cell difference into
// ``P_real_space.cells`` (absent difference ⇒ zero block), and the
// output set carries ``caches.cells``. Seminumerically:
//
//   V^{(c_σ)}(r_G) = Σ_{c_λ} P(c_σ − c_λ)ᵀ · χ(r_G − c_λ)
//   K(g)_{μν}     += w_G · χ_μ(r_G)
//                    · [A^{(δ = g − c_σ)}(r_G − c_σ) · V^{(c_σ)}]_ν
//
// (the transpose because P's row index is the λ/bra index,
// P(c_σ − c_λ)_{λσ}, matching the reference's contraction order —
// invisible for symmetric blocks, essential for off-diagonal ones)
//
// with the same per-δ caches, screens, and quadrature as
// ``compute_cosx_k_cell_pair``. For an insulator the density blocks
// decay with |c_σ − c_λ|, so the (c_λ, c_σ) work per grid point is
// bounded by the density decay range × AO-pair overlap — independent
// of the k-mesh size. This is the property that makes real-space
// exchange the production route at multi-k (handovers/HANDOVER_RIJCOSX_M3A.md
// § "M3b direction" — scope block).
//
// ``q_cached`` semantics (differs from the Γ-folded wrapper!):
//   * empty (default): return the RAW quadrature blocks — what the
//     validation tests compare against the direct-ERI reference.
//   * non-empty: apply the bra-side Q-junction from the left and
//     enforce the exact block symmetry K(g) = K(−g)ᵀ pairwise:
//       K(g) ← ½ [ Q·K_naive(g) + (Q·K_naive(−g))ᵀ ]
//     (the block generalisation of the molecular symmetrise + Q
//     post-processing; for a cell list missing −g the block keeps
//     the unsymmetrised left correction).
//
// The density blocks must satisfy P(h) = P(−h)ᵀ (a real-space
// density-matrix identity); per-spin and Γ-replicated inputs are
// both fine.
//
// The Coulomb kernel follows ``caches.omega``: 0 = full 1/r; ω > 0 =
// the erfc(ω·r)/r short-range kernel (M3b-4a) — validated ERI-exact
// against ``build_jk_2e_real_space_explicit(..., omega)``.
LatticeMatrixSet compute_cosx_k_blocks(
    const BasisSet& basis,
    const LatticeMatrixSet& P_real_space,
    const Grid& cosx_grid,
    const CosxCellPairCaches& caches,
    const Eigen::MatrixXd& q_cached = Eigen::MatrixXd(),
    const BoysTable* boys_table = nullptr);

}  // namespace vibeqc
