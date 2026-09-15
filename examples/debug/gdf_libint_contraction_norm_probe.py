"""Characterize libint's per-shell contraction normalization empirically.

For each shell in def2-svp-jk on H:
  1. Build a single-shell BasisSet (one shell at the origin) using the
     libint coefficients from BasisSet(mol, "def2-svp-jk").shells().
  2. Compute libint's self-overlap S = compute_overlap (n_comp×n_comp).
  3. Compute the ANALYTICAL self-overlap assuming libint applies ONLY
     the per-primitive normalization N(α, L) (no contraction normalization).
  4. Report the ratio. If 1.0, no contraction normalization is applied.
     If different, libint applies extra contraction normalization with
     factor √(ratio_inv) per AO of that shell.

The factor is then exactly what needs to be undone in
:func:`make_modrho_aux_basis` and :func:`make_compensating_basis` for the
libint integrals to come out in libcint convention.

Reference primitive normalization (calibrated 2026-05-16, prior handover):
  N²(α, L) = C_L · α^{-(2L+3)/2}
  C_L: L=0 → (π/2)^(3/2) ≈ 1.9687
       L=1 → 0.4922  (calibration; ≈ (π/2)^(3/2) / 4)
       L=2 → 0.3691  ((π/2)^(3/2) · 3/16)
       L=3 → 0.4614  (calibration)
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import (
    BasisSet,
    ShellInfo,
    compute_2c_eri,
    compute_overlap,
)
from vibeqc.aux_basis import make_aux_basis_set


# Calibrated per-(α, L) primitive normalization coefficients.
# N²(α, L) = C_L · α^{-(2L+3)/2}. Reference:
# handovers/HANDOVER_GDF_V0_11_2026_05_29.md
# and the prompt's "single-primitive uncontracted shells" hint.
_LIBINT_PRIM_NORM_C_L = {
    0: 1.9687,   # = (π/2)^(3/2)
    1: 0.4922,
    2: 0.3691,
    3: 0.4614,
}


def libint_prim_norm_sq(alpha: float, L: int) -> float:
    """N²(α, L) for libint's per-primitive normalization."""
    return _LIBINT_PRIM_NORM_C_L[L] * (alpha ** (-(2 * L + 3) / 2))


def raw_overlap_pp(alpha_p: float, alpha_q: float, L: int) -> float:
    """⟨raw_prim(α_p, L) | raw_prim(α_q, L)⟩ — unnormalized primitive
    self-overlap. For libint's spherical convention with NO Y_lm
    factor at L=0 and Y_lm-included for L>0.

    For L=0: ⟨exp(-α_p r²) | exp(-α_q r²)⟩ = (π/(α_p+α_q))^(3/2).
    For L>0 the formula factorizes through the radial part.
    """
    p = alpha_p + alpha_q
    # The radial part for L is ∫r^(2L+2) exp(-p r²) dr · 4π
    #   = Γ(L+3/2) / (2 · p^(L+3/2)) · 4π
    # For NORMALIZED real Y_lm (libint convention for L>0), ∫|Y_lm|²dΩ = 1.
    from scipy.special import gamma as _gamma
    n_rad = _gamma(L + 1.5) / (2.0 * p ** (L + 1.5))
    if L == 0:
        # libint drops Y_00; raw_prim is just exp(-α r²); ∫ d³r:
        return (np.pi / p) ** 1.5
    # For L>0: raw_prim = r^L Y_lm e^{-α r²} → 4π·n_rad
    return 4.0 * np.pi * n_rad


def make_one_shell_basis(shell):
    """A 1-shell BasisSet with the supplied shell's primitives at the origin."""
    # Use a placeholder He atom (Z=2, charge=0, mult=1) so the Molecule
    # constructor is happy.
    mol = vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])])
    sh = ShellInfo(0, int(shell.l), bool(shell.pure),
                   list(shell.exponents), list(shell.coefficients),
                   [0.0, 0.0, 0.0])
    return BasisSet(mol, [sh], f"<probe-L{shell.l}>", True)


def analytical_self_overlap(shell) -> float:
    """∑_pq c_p c_q · N(α_p) · N(α_q) · raw_overlap_pp(α_p, α_q, L).

    This is what libint's compute_overlap should return if libint applies
    ONLY the per-primitive normalization (no contraction normalization).
    """
    L = int(shell.l)
    exps = np.asarray(shell.exponents, dtype=float)
    coeffs = np.asarray(shell.coefficients, dtype=float)
    s = 0.0
    for p in range(len(exps)):
        for q in range(len(exps)):
            Np = np.sqrt(libint_prim_norm_sq(exps[p], L))
            Nq = np.sqrt(libint_prim_norm_sq(exps[q], L))
            s_pq = raw_overlap_pp(exps[p], exps[q], L)
            s += coeffs[p] * coeffs[q] * Np * Nq * s_pq
    return float(s)


def main() -> int:
    mol = vq.Molecule([vq.Atom(2, [0, 0, 0])])  # He (charge=0, mult=1)
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    print(f"def2-svp-jk on H: {len(list(aux.shells()))} shells")
    print()
    print(f"{'ish':>4s}  {'L':>3s}  {'nprim':>6s}  "
          f"{'S_libint[0,0]':>14s}  {'S_analytical':>14s}  "
          f"{'ratio':>10s}  {'sqrt(1/ratio)':>14s}")
    print("-" * 90)

    for i, sh in enumerate(aux.shells()):
        try:
            probe = make_one_shell_basis(sh)
            S_libint = np.asarray(compute_overlap(probe))
            S_anal = analytical_self_overlap(sh)
            # libint returns (2L+1)×(2L+1) for pure spherical; diagonal
            # should be uniform by rotational symmetry.
            S_diag = float(S_libint[0, 0])
            ratio = S_diag / S_anal if S_anal != 0 else float("nan")
            extra_norm = np.sqrt(1.0 / ratio) if ratio > 0 else float("nan")
            print(f"{i:>4d}  {int(sh.l):>3d}  {len(list(sh.exponents)):>6d}  "
                  f"{S_diag:>14.6f}  {S_anal:>14.6f}  "
                  f"{ratio:>10.6f}  {extra_norm:>14.6f}")
        except Exception as exc:
            print(f"{i:>4d}  L={int(sh.l)}: ERR {type(exc).__name__}: {exc}")

    print()
    print("Interpretation:")
    print("  ratio = 1.0  → libint applies NO contraction normalization;")
    print("                primitive-normalization-only formula is correct.")
    print("  ratio ≠ 1.0  → libint applies extra per-shell contraction")
    print("                 normalization. The factor ``sqrt(1/ratio)`` is")
    print("                 what you'd multiply each coefficient by to undo it.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
