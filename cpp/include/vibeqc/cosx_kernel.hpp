// Custom one-electron nuclear-attraction kernel for the COSX-K hot path.
//
// Purpose. ``compute_cosx_k`` in ``cpp/src/cosx.cpp`` evaluates, per
// grid point ``r_g``, an analytical-integral block A_{μν}(r_g) for
// every surviving shell pair via libint's ``Operator::nuclear`` engine
// parameterised by a single pseudo-charge q = −1 at r_g. The libint
// pair-loop is > 99 % of the K-build wall on the n-hexadecane
// B3LYP/RIJCOSX/def2-svp 5-iter probe (measured on
// ``origin/main @ fec7c6f``). Replacing that single call with an
// in-tree kernel — keeping the rest of the per-point structure
// (chi-screen / Dχ build / row-screened rank-1 K) intact — is the
// remaining 5-10× lever on the wall.
//
// Algorithm. Standard Boys-table + Obara-Saika recursion for the
// contracted (a|V_C|b) nuclear-attraction integral with a single
// nucleus C carrying charge q = −1.
//
// One primitive Gaussian pair (p, q):
//
//   α   = α_p + α_q
//   P   = (α_p · A + α_q · B) / α                  (product centre)
//   K   = exp(−α_p · α_q / α · |A − B|²)           (Gaussian product)
//   T   = α · |P − C|²
//
//   [0|V_C|0]^(m) = 2π/α · K · F_m(T)
//
//   where F_m(T) = ∫_0^1 t^{2m} exp(−T t²) dt is the m-th Boys
//   function. The sign matches libint's ``Operator::nuclear`` engine
//   for ``set_params([(−1, C)])`` because that engine implements
//   V(r) = −Σ_A Z_A / |r − R_A| in the attractive-nucleus convention
//   — feeding Z = −1 gives the *positive* ⟨μ|1/|r − C||ν⟩ that this
//   formula evaluates directly.
//
// Obara-Saika upward recursion in the angular momentum of bra (and
// symmetrically ket); the auxiliary index m runs 0 .. l_a + l_b:
//
//   [a+1_i | V_C | b]^(m) =  (P_i − A_i) [a | V_C | b]^(m)
//                          − (P_i − C_i) [a | V_C | b]^(m+1)
//                          + N_i(a) / (2α) · ( [a−1_i | V_C | b]^(m)
//                                              − [a−1_i | V_C | b]^(m+1) )
//                          + N_i(b) / (2α) · ( [a | V_C | b−1_i]^(m)
//                                              − [a | V_C | b−1_i]^(m+1) )
//
//   where ``a`` is a Cartesian-Gaussian multi-index (a_x, a_y, a_z),
//   ``1_i`` is the unit vector along Cartesian axis i, and N_i(·) is
//   the i-th component of the multi-index. The recursion terminates
//   at [0|V|0]^(m) for any m ≤ l_a + l_b.
//
// Boys function. The function is tabulated at a fixed Δt-grid in T
// for T ∈ [0, T_max]; 6th-order Taylor interpolation
//
//   F_n(T + h) = Σ_{k=0}^{6} (−h)^k / k! · F_{n+k}(T)
//
// reaches < 1e-12 relative accuracy with Δt = 0.05 over T ∈ [0, 30]
// (Gill-Head-Gordon-Pople 1991, IJQC Symp. 40, 269). For T > T_max,
// the leading asymptotic form
//
//   F_n(T) ≈ (2n − 1)!! / (2T)^n · √(π / (4T))
//
// is accurate to relative 1e-12 already at T = 30 for n ≤ 6; one
// downward Boys recursion fills lower n if needed:
//
//   F_{n-1}(T) = (2T · F_n(T) + exp(−T)) / (2n − 1).
//
// Contraction. The libint::Shell input carries a list of primitives
// (α_p, c_p) per contracted shell. The contracted nuclear-attraction
// block is
//
//   (a|V_C|b)_{μν} = Σ_{p, q}  c_p^a c_q^b N(l_a, α_p) N(l_b, α_q)
//                              · [a_p | V_C | b_q]_{μν}^{(0)}
//
// where N(l, α) is the standard Cartesian-Gaussian normalisation.
// libint stores the c · N product as a pre-normalised contraction
// coefficient in ``Shell::contr[*].coeff[*]``; the kernel consumes
// that directly.
//
// Spherical-AO transform. libint computes integrals in Cartesian
// Gaussians and applies the standard solid-harmonic → spherical
// transform at the contracted-block boundary when
// ``Shell::contr[*].pure == true``. The kernel mirrors that:
// Cartesian integrals come out of Obara-Saika, the spherical
// transform is applied just before return so the (n1, n2) block
// matches the libint engine's output element-for-element.
//
// References (no proprietary QC-code source consulted):
//
//   * Gill, P. M. W.; Head-Gordon, M.; Pople, J. A., *An efficient
//     algorithm for the generation of two-electron repulsion
//     integrals over Gaussian basis functions*, Int. J. Quantum
//     Chem. Symp. 40, 269 (1991) — Boys function tabulation + Taylor
//     interpolation; asymptotic blend.
//
//   * Obara, S.; Saika, A., *Efficient recursive computation of
//     molecular integrals over Cartesian Gaussian functions*, J.
//     Chem. Phys. 84, 3963 (1986) — the bra/ket recursion above.
//
//   * Head-Gordon, M.; Pople, J. A., *A method for two-electron
//     Gaussian integral and integral derivative evaluation using
//     recurrence relations*, J. Chem. Phys. 89, 5777 (1988) —
//     improved formulation for nuclear-attraction + ERI variants.
//
//   * Helgaker, T.; Jørgensen, P.; Olsen, J., *Molecular
//     Electronic-Structure Theory*, Wiley 2000, Ch. 9 — textbook
//     derivation of the Obara-Saika recursion + the Cartesian-to-
//     spherical transform.

#pragma once

#include <Eigen/Dense>
#include <libint2.hpp>
#include <libint2/shell.h>

#include <array>
#include <cstddef>
#include <vector>

namespace vibeqc {

// Boys-function lookup table. Tabulated F_n(T) over n ∈ [0, n_max]
// and T ∈ [0, t_max] on a uniform Δt grid; 6th-order Taylor
// interpolation in T; leading-order asymptotic for T > t_max
// (cut at t_max = 50 — past this the asymptotic is FP-roundoff
// accurate). n_max is determined by the maximum total angular
// momentum the consumer will request — for a basis with max_l = L,
// we need n_max ≥ 2L + 6 (the +6 buys the Taylor expansion's extra
// columns).
//
// Built once at COSXJKBuilder construction (basis-only / SCF-
// invariant). Lookup is thread-safe (const), heap-free, no locks.
struct BoysTable {
    int n_max = 0;            // highest derived order F_n_max
    double t_max = 0.0;       // last tabulated grid point
    double dt = 0.0;          // grid spacing in T
    int n_grid = 0;           // number of grid points = floor(t_max / dt) + 1

    // Row-major (n_grid × (n_max + 7)): the +7 is the 6 extra
    // columns needed for the 6th-order Taylor sum + the F_n_max
    // column itself.
    std::vector<double> values;

    // Inline lookup: F_n(T) with 6th-order Taylor interpolation
    // when T ≤ t_max, asymptotic-blend + downward recursion when
    // T > t_max. Argument bounds: 0 ≤ n ≤ n_max, T ≥ 0.
    double eval(int n, double T) const;
};

// Build a Boys-function table covering all orders needed for shells
// of max angular momentum ``max_l`` (so n_max = 2 · max_l + 6) and
// up to t_max = 30 (the default Gill-HG-Pople 1991 boundary). Uses
// double-exponential quadrature for the highest-order Boys column
// (numerically stable to 1e-15 at any T in the tabulated range);
// lower orders are filled by downward Boys recursion to preserve
// precision (Gill et al. 1991 § 2).
BoysTable build_boys_table(int max_l);

// Precomputed per-primitive-pair data for a single (s_a, s_b) shell
// pair. Stored per (primitive of a) × (primitive of b) — the kernel
// loops over this list inside the contraction sum.
struct PrimitivePairData {
    double alpha;             // α = α_p + α_q
    double inv_alpha;         // 1 / α (cached for speed)
    double inv_two_alpha;     // 1 / (2α) (Obara-Saika coefficient)
    std::array<double, 3> P;  // product centre (α_p A + α_q B) / α
    double prefactor;         // c_p · N_p · c_q · N_q · 2π/α · exp(−α_p α_q / α · |A − B|²)
    // 2π/α and the Gaussian product K are folded into ``prefactor``
    // alongside the contraction coefficients; the kernel only
    // multiplies by F_m(T) and the OS angular factors per term.
};

// Per-shell-pair primitive-pair cache. Built once at COSXJKBuilder
// construction by ``build_primitive_pair_cache``. Flat 2D layout —
// ``data[s1 * n_shells + s2]`` gives the primitive-pair list for
// pair (s1, s2). Both directions are stored (s1, s2) and (s2, s1);
// the kernel does not transpose on the fly.
struct PrimitivePairCache {
    int n_shells = 0;
    // Length n_shells² ; pairs[k] holds the contracted-pair
    // primitives for shell pair (k / n_shells, k % n_shells).
    std::vector<std::vector<PrimitivePairData>> pairs;
};

PrimitivePairCache build_primitive_pair_cache(
    const libint2::BasisSet& shells);

// Two-set variant: primitive-pair data between two distinct shell
// lists (e.g. lattice-shifted ν shells × home σ shells for the
// cell-pair periodic COSX kernel, cosx_cell_pair.hpp). Layout:
// ``pairs[s1 * shells_b.size() + s2]`` with s1 indexing ``shells_a``
// and s2 indexing ``shells_b``; the ``n_shells`` field holds the row
// stride (= shells_b.size()).
PrimitivePairCache build_primitive_pair_cache_two_set(
    const std::vector<libint2::Shell>& shells_a,
    const std::vector<libint2::Shell>& shells_b);

// Per-thread workspace for ``cosx_nuclear_pair_into``. Owns every
// per-call scratch buffer the kernel touches so the hot loop pays
// zero heap allocations per shell-pair call:
//
//   * ``bra_aux`` / ``ket_aux`` — Obara-Saika scratch tensors,
//     reused across primitive pairs within one call and across
//     calls of any (l_a, l_b) ≤ (reserved_max_l_a, reserved_max_l_b).
//
//   * ``K_cart`` — contracted Cartesian accumulator
//     (max_n_cart_a × max_n_cart_b), written in row-major order.
//
//   * ``K_mid`` — intermediate after the a-side spherical / norm
//     transform and before the b-side one. Sized to fit the
//     larger of (Cartesian, pure) on each axis.
//
//   * ``cart_lx`` / ``cart_ly`` / ``cart_lz`` — Cartesian-index
//     tables per shell l (libint order), indexed by l. Built once.
//
//   * ``T_pure`` — Cartesian-to-pure (solid-harmonic) transform
//     per l. Each ``T_pure[l]`` is a row-major (2l+1) × n_cart(l)
//     flat array. Built once via libint's
//     ``SolidHarmonicsCoefficients`` — same convention
//     ``ao_eval.cpp`` consumes for AO transforms.
//
//     Cartesian-output shells (``pure = false``) need neither a
//     transform nor any normalisation: Obara-Saika already produces
//     the convention libint's engine emits for them. A per-component
//     ``cart_norm`` factor lived here until 2026-08-05 and was wrong;
//     see the note in ``cosx_kernel.cpp``.
//
// Allocate one workspace per OMP worker; ``reserve`` sizes
// everything for the largest (l_a, l_b) the consumer will pass
// in. Re-use across millions of pair calls.
struct CosxKernelWorkspace {
    int reserved_max_l_a = -1;
    int reserved_max_l_b = -1;

    // OS recursion tensors.
    std::vector<double> bra_aux;
    std::vector<double> ket_aux;

    // Output-side scratch.
    std::vector<double> K_cart;
    std::vector<double> K_mid;
    int K_cart_stride = 0;    // row stride for K_cart (= max_n_cart_b)
    int K_mid_stride = 0;     // row stride for K_mid

    // Per-l Cartesian-index tables. Length ``reserved_max_l + 1``;
    // ``cart_lx[l]`` etc. have length n_cart(l) = (l+1)(l+2)/2 in
    // libint's standard order.
    std::vector<std::vector<int>> cart_lx;
    std::vector<std::vector<int>> cart_ly;
    std::vector<std::vector<int>> cart_lz;

    // Per-l Cartesian-to-pure transform; ``T_pure[l]`` is row-major
    // (2l+1) × n_cart(l).
    std::vector<std::vector<double>> T_pure;

    // Size the buffers to accommodate any shell pair with
    // l_a ≤ max_l_a, l_b ≤ max_l_b. Idempotent — calling with the
    // same or smaller bounds is a no-op.
    void reserve(int max_l_a, int max_l_b);
};

// Evaluate the contracted nuclear-attraction integral block
// (a|V_C|b) for shell pair (s1, s2) with a single pseudo-nucleus at
// ``C`` carrying charge q = −1. Matches libint's ``Operator::nuclear``
// engine output for the same inputs to ULP per matrix element (the
// claim is validated by ``tests/test_cosx_kernel.py``).
//
// Inputs:
//   * ``shell_a``, ``shell_b`` — the libint shells (used for
//     angular-momentum, pure/Cartesian flag, and as the lookup
//     handle for the primitive-pair cache).
//   * ``pair_data`` — the precomputed primitive-pair list for
//     (s1, s2); equal to ``cache.pairs[s1 * n_shells + s2]``.
//   * ``C`` — Cartesian coordinates of the pseudo-nucleus (bohr).
//   * ``boys`` — the basis-built Boys table.
//   * ``omega`` (default 0) — Coulomb-kernel range separation.
//     0 evaluates the full 1/r kernel; ω > 0 evaluates the
//     erfc-attenuated SHORT-RANGE kernel erfc(ω·r)/r (matching
//     libint's ``Operator::erfc_nuclear`` with the same ω), the
//     building block for range-separated periodic exchange
//     (M3b-4a; handovers/HANDOVER_RIJCOSX_M3A.md). Implemented as a seeding
//     change only — the attenuated Boys family ρ^{m+1/2}F_m(ρT),
//     ρ = ω²/(ω²+α), obeys the same downward derivative relation,
//     so the Obara-Saika recursion is shared.
//
// Returns an Eigen row-major (n1 × n2) matrix where n1 = shell_a.size()
// and n2 = shell_b.size() (spherical or Cartesian according to
// ``shell.contr[*].pure``).
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
cosx_nuclear_pair(const libint2::Shell& shell_a,
                  const libint2::Shell& shell_b,
                  const std::vector<PrimitivePairData>& pair_data,
                  const std::array<double, 3>& C,
                  const BoysTable& boys,
                  double omega = 0.0);

// Hot-loop variant of ``cosx_nuclear_pair``. Same output, no heap
// allocation per call: the caller provides a ``CosxKernelWorkspace``
// (sized via ``reserve``) for the Obara-Saika tensors, and a
// pre-allocated ``(n_a, n_b)`` row-major output buffer. ``n_a`` =
// ``shell_a.size()``, ``n_b`` = ``shell_b.size()``. The output
// matches libint's ``Operator::nuclear`` block element-for-element
// (``Operator::erfc_nuclear`` when ``omega > 0``).
//
// Intended call site: per-grid-point pair loop inside
// ``compute_cosx_k`` (cpp/src/cosx.cpp). One workspace per OMP
// worker; buffers reused across millions of shell-pair calls.
void cosx_nuclear_pair_into(
    const libint2::Shell& shell_a,
    const libint2::Shell& shell_b,
    const std::vector<PrimitivePairData>& pair_data,
    const std::array<double, 3>& C,
    const BoysTable& boys,
    CosxKernelWorkspace& ws,
    double* out,
    double omega = 0.0);

}  // namespace vibeqc
