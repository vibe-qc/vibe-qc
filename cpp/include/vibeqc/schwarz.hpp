// Periodic Cauchy–Schwarz screening utilities.
//
// Shared between the periodic 2-e Fock build (``build_fock_2e_real_space``,
// ``build_jk_gamma_molecular_limit`` in ``periodic_fock.cpp``) and the
// periodic 2-e gradient pass (``eri_lattice_gradient_contribution`` in
// ``periodic_gradient.cpp``). Both walk the same shell-quartet × cell-
// triple loop and benefit from the same per-cell Q tensor.
//
// The Schwarz bound is the standard
//
//   |⟨μ_0 ν_g | λ_λ σ_σ⟩|  ≤  Q[c_g][μ, ν] · Q[c_σ−c_λ][λ, σ]
//
// where the per-cell Q tensor is
//
//   Q[c][s_a, s_b] = √( max_{ij} | ⟨s_a_0 s_b_c | s_a_0 s_b_c⟩_{ij,ij} | ).
//
// References: Häser & Ahlrichs 1989; CP2K's ``EPS_SCHWARZ`` (1e-7
// energies, tighter for forces); CRYSTAL's ``TOLINTEG`` 5-vector.

#pragma once

#include <Eigen/Dense>
#include <libint2.hpp>

#include <vector>

namespace vibeqc {

// Cell-translate every shell origin in `shells` by `dr` (bohr). Used
// by both the precompute and the main pass to position shells at a
// given lattice cell.
std::vector<libint2::Shell> shift_shells_to_cell(
    const libint2::BasisSet& shells, const Eigen::Vector3d& dr);

// Pre-compute the per-cell Schwarz factors
//
//   Q[c][s_a * nshells + s_b] = √( max_{ij} | ⟨s_a_0 s_b_c | s_a_0 s_b_c⟩_{ij,ij} | )
//
// for every cell c in `shells_at` and every (s_a, s_b) shell pair. The
// engine prototype carries the operator (Coulomb or erfc-Coulomb) +
// any kernel parameters (omega), so the precomputed Q matches whatever
// the main pass will evaluate. Cost: O(n_c · n_shells²) libint quartet
// evaluations — trivial vs the O(n_c³ · n_shells⁴) main pass.
//
// Use a fresh non-derivative engine for the precompute even when the
// main pass uses a derivative engine: the Schwarz bound applies to
// the integral magnitude, not the derivative.
std::vector<std::vector<double>> compute_schwarz_factors_per_cell(
    const libint2::BasisSet& shells_ref,
    const std::vector<std::vector<libint2::Shell>>& shells_at,
    const libint2::Engine& prototype);

// Single-cell (Γ-only / zero-displacement) Schwarz factors
//
//   Q(s_a, s_b) = √( max_{ij} | ⟨s_a s_b | s_a s_b⟩_{ij,ij} | )
//
// for every shell pair in `shells`. This is the molecular case — i.e.
// the c=0 collapse of compute_schwarz_factors_per_cell — surfaced as a
// dedicated entry point so molecular consumers (DirectJKBuilder in
// cpp/src/jk_direct.cpp; the COSX pair-screen at cpp/src/cosx.cpp)
// don't have to construct a one-element cell list and reshape a
// nshells² flat vector. Q is symmetric and stored as Eigen::MatrixXd
// for direct (s_a, s_b) indexing in the consumer's quartet / pair
// loops.
//
// The engine prototype carries the operator (Coulomb or erfc-Coulomb)
// + any parameters; pass a Coulomb engine for the standard bound.
// Parallelised over the upper triangle of (s_a, s_b); the symmetric
// half is mirrored.
Eigen::MatrixXd compute_schwarz_factors(
    const libint2::BasisSet& shells,
    const libint2::Engine& prototype);

// Conservative density envelope: max |D| over all blocks, floored at
// 1.0 to avoid spurious zero-suppression on Hcore-guess SCF iter 1
// (where the density may be tiny and unrepresentative).
double density_envelope(const std::vector<Eigen::MatrixXd>& blocks);

// Per-shell radial extent: for each shell ``s``, the smallest radius
// ``r*`` such that the contracted AO ``|χ_μ(r)| < tol`` for any
// ``μ ∈ s`` and ``|r - O_s| > r*``.
//
// Derivation (standard Gaussian-extent bound; see Stratmann/Scuseria/
// Frisch 1996 § 11 or Burow & Sierka 2011 § 2.3):
//
//   |χ_μ(r)| ≤ Σ_p |c_p · N(l, α_p)| · r^l · exp(-α_p r²)
//            ≤ exp(max_ln_amp + l ln r - α_min r²)
//
// solving the right-hand-side ≤ tol for r gives, with a couple of
// Newton-style iterations,
//
//   r* ≈ sqrt( (max_ln_amp + l · ln(r) − ln tol) / α_min ).
//
// libint's ``max_ln_coeff`` already carries
// ln(max_p |c_p · N(l, α_p)|) per primitive, so the bound is exact
// up to the polynomial-r prefactor that the iteration converges.
//
// Used by COSX-K and XC builders to build a per-grid-point list of
// "active shells" via a simple ``|r_g - O_s| < r*[s]`` distance check,
// reducing the shell-pair loop from O(n_shells²) to O(n_active²) on
// extended systems. Conservative (over-estimating) so screening is
// always safe.
//
// ``tol`` typically matches the consumer's AO threshold (``AO_TOL =
// 1e-7`` in ``cpp/src/cosx.cpp``). Returns a vector of length
// ``n_shells``.
std::vector<double> compute_shell_radial_cutoffs(
    const libint2::BasisSet& shells, double tol);

// Per-cell Q maximum, taken over all shell pairs:
//
//   Q_max[c] = max_{s_a, s_b} Q[c][s_a, s_b]
//
// Used to short-circuit the entire (c_g, c_λ, c_σ) cell triple in the
// inner Fock / gradient loop when ``Q_max[c_g] · Q_max[c_h] · D_max``
// already falls below the screening threshold, *before* iterating any
// of the n_shells^4 shell quartets. On real crystals where many cell
// triples have negligible shell-pair overlap this cuts the inner-loop
// hit count by another order of magnitude on top of per-quartet
// Schwarz.
std::vector<double> max_q_per_cell(
    const std::vector<std::vector<double>>& Q);

}  // namespace vibeqc
