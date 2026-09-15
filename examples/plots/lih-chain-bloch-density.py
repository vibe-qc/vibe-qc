"""1D LiH-chain Bloch-orbital |ψ|² at Γ vs X — tutorial 22 figure.

Two-panel 2D contour figure showing the *gauge-invariant* density
$|\\psi_{n,\\mathbf{k}}(\\mathbf{r})|^2$ of the bonding (lowest valence)
crystalline orbital of a 1D LiH chain at the two endpoints of the
band-structure path:

  * Left  — Γ-point: the orbital amplitude is identical in every
    unit cell (Bloch phase = 1 across all cells).
  * Right — X-point: same magnitude per cell but opposite sign
    between adjacent cells. The density |ψ|² hides the sign flip,
    so the two panels look almost identical *as densities* — that's
    the gauge-invariance pedagogy point this figure makes.

Renders directly via ``vq.evaluate_bloch_orbital`` (Phase V3 API);
no XSF / cube writer round-trip needed for this 2D inline preview.
The companion ``write_xsf_mo`` / ``write_cube_mo_periodic`` calls
are demonstrated in tutorial 22's text.

Run:
    .venv/bin/python examples/plots/lih-chain-bloch-density.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "lih-chain-bloch-density.png"

A = 5.5  # 1D LiH lattice parameter (bohr)


def diag_at(k_cart, S_lat, T_lat, V_lat):
    """Diagonalize the Bloch Hcore Hamiltonian at one k-point. Returns
    (eigenvalues, eigenvectors) sorted ascending in energy."""
    S_k = vq.bloch_sum(S_lat, k_cart)
    F_k = vq.bloch_sum(T_lat, k_cart) + vq.bloch_sum(V_lat, k_cart)
    sol = vq.diagonalize_bloch(F_k, S_k)
    eps = np.asarray(sol.energies)
    C = np.asarray(sol.coefficients)
    order = np.argsort(eps)
    return eps[order], C[:, order]


def main() -> None:
    t0 = time.perf_counter()

    # 1D LiH chain — two atoms per cell, sto-3g.
    system = vq.PeriodicSystem(
        1,
        [[A, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(3, [0.0,    0.0, 0.0]),
         vq.Atom(1, [0.5*A,  0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    # Hcore lattice matrices for the band-structure-style diagonalisation.
    opts = vq.LatticeSumOptions()
    S_lat = vq.compute_overlap_lattice(basis, system, opts)
    T_lat = vq.compute_kinetic_lattice(basis, system, opts)
    V_lat = vq.compute_nuclear_lattice(basis, system, opts)
    k_gamma = np.array([0.0,         0.0, 0.0])
    k_x     = np.array([np.pi / A,   0.0, 0.0])

    eps_g, C_g = diag_at(k_gamma, S_lat, T_lat, V_lat)
    eps_x, C_x = diag_at(k_x,     S_lat, T_lat, V_lat)

    bonding_idx = 1   # band 0 is the Li-1s core; band 1 is the valence
    print(f"  Li-1s core band:  ε(Γ) = {eps_g[0]*27.211386:7.2f} eV   "
          f"ε(X) = {eps_x[0]*27.211386:7.2f} eV   "
          f"(width = {(eps_x[0]-eps_g[0])*27.211386:+5.2f} eV)")
    print(f"  Bonding valence:  ε(Γ) = {eps_g[bonding_idx]*27.211386:7.2f} eV   "
          f"ε(X) = {eps_x[bonding_idx]*27.211386:7.2f} eV   "
          f"width = {(eps_x[bonding_idx]-eps_g[bonding_idx])*27.211386:+5.2f} eV")

    # 2D grid: chain axis x ∈ [-1.5 a, 1.5 a] (three cells), z ∈ [-2, 2]
    n_x, n_z = 360, 160
    xs = np.linspace(-1.5 * A, 1.5 * A, n_x)
    zs = np.linspace(-2.0, 2.0, n_z)
    X, Z = np.meshgrid(xs, zs)
    pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])

    # Bloch orbital at each k for the bonding *valence* band.
    # LiH has 4 electrons; band 0 is the Li-1s core (~-216 eV at sto-3g),
    # band 1 is the bonding valence band — that's the chemistry we want.
    bonding_idx = 1
    psi_g = vq.evaluate_bloch_orbital(basis, system, pts, C_g, k_gamma,
                                      bonding_idx)
    psi_x = vq.evaluate_bloch_orbital(basis, system, pts, C_x, k_x,
                                      bonding_idx)

    rho_g = (np.abs(psi_g) ** 2).reshape(X.shape)
    rho_x = (np.abs(psi_x) ** 2).reshape(X.shape)
    re_g  = psi_g.real.reshape(X.shape)
    re_x  = psi_x.real.reshape(X.shape)

    # Atom positions over the three displayed cells, for overlay.
    cell_centres = np.array([-A, 0.0, +A])
    li_pos = np.column_stack([cell_centres,                       np.zeros(3)])
    h_pos  = np.column_stack([cell_centres + 0.5 * A,             np.zeros(3)])

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), dpi=150,
                             sharex=True, sharey=True)

    # Top row: Re ψ — shows the gauge-arbitrary sign pattern
    amp_re = float(np.percentile(np.abs([re_g, re_x]), 99.5))
    levels_re = np.linspace(-amp_re, amp_re, 21)
    axes[0, 0].contourf(X, Z, re_g, levels=levels_re,
                        cmap="RdBu_r", extend="both")
    axes[0, 1].contourf(X, Z, re_x, levels=levels_re,
                        cmap="RdBu_r", extend="both")
    axes[0, 0].contour(X, Z, re_g, levels=[0.0],
                       colors="black", linewidths=0.4, linestyles="--")
    axes[0, 1].contour(X, Z, re_x, levels=[0.0],
                       colors="black", linewidths=0.4, linestyles="--")
    axes[0, 0].set_title(r"(a) $\mathrm{Re}\,\psi_{0,\Gamma}$ — "
                         "all cells in phase", fontsize=11)
    axes[0, 1].set_title(r"(b) $\mathrm{Re}\,\psi_{0,X}$ — "
                         "alternating sign between cells", fontsize=11)

    # Bottom row: |ψ|² — gauge-invariant density
    amp_rho = float(np.percentile(np.array([rho_g, rho_x]), 99.5))
    levels_rho = np.linspace(0.0, amp_rho, 21)
    axes[1, 0].contourf(X, Z, rho_g, levels=levels_rho,
                        cmap="viridis", extend="max")
    axes[1, 1].contourf(X, Z, rho_x, levels=levels_rho,
                        cmap="viridis", extend="max")
    axes[1, 0].set_title(r"(c) $|\psi_{0,\Gamma}|^2$ — "
                         "density is the same in every cell", fontsize=11)
    axes[1, 1].set_title(r"(d) $|\psi_{0,X}|^2$ — "
                         "density indistinguishable from $\\Gamma$",
                         fontsize=11)

    for ax in axes.flat:
        for x, _ in li_pos:
            ax.plot(x, 0.0, "o", ms=12, mfc="white", mec="black",
                    mew=1.2, zorder=5)
            ax.text(x, 0.0, "Li", ha="center", va="center",
                    fontsize=8, zorder=6)
        for x, _ in h_pos:
            ax.plot(x, 0.0, "o", ms=10, mfc="lightgrey", mec="black",
                    mew=1.0, zorder=5)
            ax.text(x, 0.0, "H", ha="center", va="center",
                    fontsize=7, zorder=6)
        # Cell-boundary guide lines
        for xb in (-0.5*A, 0.5*A, 1.5*A, -1.5*A):
            ax.axvline(xb, color="grey", linewidth=0.6, alpha=0.4,
                       linestyle=":")
        ax.set_aspect("equal")

    for ax in axes[1, :]:
        ax.set_xlabel("x along chain (bohr)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z (bohr)")

    fig.suptitle("1D LiH chain — bonding Bloch orbital at Γ vs X "
                 "(top: Re ψ shows gauge phase; bottom: |ψ|² is gauge-invariant)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}  "
          f"({time.perf_counter()-t0:.2f} s)")


if __name__ == "__main__":
    main()
