"""Per-L FT convention calibration for vibe-qc's AFT machinery.

Pre-requisite for the AFT long-range correction in build_lpq_compcell.
The in-source status note in python/vibeqc/aux_basis.py flags that
rsgdf_aux_fourier_transform has a documented absolute-scale mismatch
vs libint's compute_2c_eri at L=0 (claimed fixed by dropping the Y_00
factor) and "needs per-L convention factors" at L>0. This script
characterizes the empirical landscape.

Method
------
For each L in {0, 1, 2, 3}:

  1. Build a single-primitive shell ``g(r) = c · r^L Y_{l,m}(r̂) ·
     exp(-α r²)`` with c chosen so the radial part follows libint's
     ``coefficients_pre_normalized=True`` convention (we pass c as
     given, libint takes it verbatim).

  2. Compute (g|g) two ways:

       libint:    direct call to compute_2c_eri (Coulomb self-integral
                  of the shell).

       Parseval:  reciprocal-space reconstruction via
                  ``(g|g) = (4π/V) Σ_{G≠0} |ĝ(G)|² / G²`` on a fine
                  G-mesh in a large box (so the discrete sum
                  approximates the continuous integral). ĝ(G) comes
                  from rsgdf_aux_fourier_transform.

  3. Report the ratio libint(L) / Parseval(L) per L. If the ratio is
     constant per L (e.g., 4π for L=0 per the existing fix), it's a
     per-L scale we can bake into the FT. If it varies within a shell
     (i.e., across m at fixed L), there's a deeper convention issue.

The convention factor identified here is what we'll multiply
rsgdf_aux_fourier_transform's output by to make subsequent (chg|fused)
contractions match libint's direct lattice sum.
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import ShellInfo, compute_2c_eri
from vibeqc.aux_basis import rsgdf_aux_fourier_transform, rsgdf_g_mesh


def make_single_prim_basis(L: int, alpha: float, coef: float = 1.0,
                           origin=(0.0, 0.0, 0.0)):
    """A BasisSet with one single-primitive pure-spherical shell at L.

    Uses ``coefficients_pre_normalized=True`` (the default for shells
    extracted via .shells() round-trip). For a single primitive, this
    means libint takes the (α, c) pair verbatim with no extra
    contraction-normalization step.
    """
    # Use He (Z=2) for a charge=0 / mult=1 placeholder. The atom
    # carrier is irrelevant — we only use the BasisSet for integrals.
    mol = vq.Molecule([vq.Atom(2, list(origin))])
    sh = ShellInfo(0, int(L), True, [float(alpha)], [float(coef)], list(origin))
    return vq.BasisSet(mol, [sh], f"<single-L{L}-alpha{alpha}>", True), mol


def libint_pp(L: int, alpha: float, coef: float = 1.0) -> np.ndarray:
    """(P|P) Coulomb matrix from libint for the single-prim shell."""
    bs, _ = make_single_prim_basis(L, alpha, coef)
    return np.asarray(compute_2c_eri(bs))


def parseval_pp(L: int, alpha: float, coef: float = 1.0,
                box_bohr: float = 30.0,
                G_max: float = 8.0,
                G_density: int = 60) -> np.ndarray:
    """(P|P) reconstructed from the analytical FT via Parseval.

    For a NON-PERIODIC continuous Coulomb integral:
        (g_a|g_b) = (4π/(2π)³) ∫ ĝ_a*(G) · ĝ_b(G) / G²  d³G
                  = (1/(2π²)) ∫ ĝ_a*(G) · ĝ_b(G) / G²  d³G

    We approximate the integral with a uniform 3D mesh of G-vectors
    inside ``|G| ≤ G_max`` (uniform spacing = G_max / G_density),
    using midpoint quadrature with weight ``d³G = (Δk)³``. Box size
    is just a placeholder for the PeriodicSystem the rsgdf helpers
    expect; the result is geometry-independent.
    """
    bs, mol = make_single_prim_basis(L, alpha, coef)
    nbf = bs.nbasis

    # Build a uniform G-mesh inside the cube |G_i| <= G_max.
    grid = np.linspace(-G_max, G_max, 2 * G_density + 1)
    dk = grid[1] - grid[0]
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    G = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    G2 = (G ** 2).sum(axis=1)
    mask = G2 > 0
    G = G[mask]
    G2 = G2[mask]

    ft = rsgdf_aux_fourier_transform(bs, G)              # (nbf, n_G) complex
    # Continuous integral approximation: (1/(2π²)) Σ_G ĝ*·ĝ / G² · (Δk)³
    weight = (dk ** 3) / (2.0 * np.pi ** 2)
    out = np.zeros((nbf, nbf), dtype=complex)
    inv_G2 = 1.0 / G2
    for i in range(nbf):
        for j in range(nbf):
            out[i, j] = weight * np.sum(np.conj(ft[i]) * ft[j] * inv_G2)
    # Imaginary part is numerical noise for a self-shell at the origin.
    return out.real


def main() -> int:
    print(f"vibeqc {vq.__version__}: FT convention calibration")
    print()

    # Quadrature convergence study at L=0 (fixed alpha):
    print("Quadrature convergence at L=0, alpha=0.5:")
    print(f"  {'G_max':>6s}  {'density':>8s}  {'libint':>12s}  "
          f"{'parseval':>12s}  {'ratio':>10s}")
    for G_max, dens in [(8.0, 60), (15.0, 80), (20.0, 100), (25.0, 120)]:
        Ml = libint_pp(0, 0.5)
        Mp = parseval_pp(0, 0.5, G_max=G_max, G_density=dens)
        print(f"  {G_max:>6.1f}  {dens:>8d}  {Ml[0,0]:>12.6f}  "
              f"{Mp[0,0]:>12.6f}  {Ml[0,0]/Mp[0,0]:>10.6f}")
    print()

    # Alpha-dependence study at each L (converged quadrature):
    print("Alpha-dependence at converged quadrature (G_max=20, density=100):")
    print(f"  {'L':>3s}  {'alpha':>6s}  {'libint':>12s}  {'parseval':>12s}  "
          f"{'ratio':>10s}  {'spread':>10s}")
    for L in range(0, 4):
        for alpha in [0.25, 0.5, 1.0, 2.0]:
            try:
                Ml = libint_pp(L, alpha)
                Mp = parseval_pp(L, alpha, G_max=20.0, G_density=100)
                diag_l = np.diag(Ml)
                diag_p = np.diag(Mp)
                ratios = diag_l / np.where(diag_p != 0, diag_p, np.inf)
                spread = float(ratios.std() / max(abs(ratios.mean()), 1e-30))
                print(f"  {L:>3d}  {alpha:>6.2f}  {diag_l[0]:>12.4f}  "
                      f"{diag_p[0]:>12.4f}  {ratios[0]:>10.6f}  "
                      f"{spread:>10.3e}")
            except Exception as exc:
                print(f"  {L:>3d}  {alpha:>6.2f}  ERR: {exc}")
        print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
