"""H2-chain crystalline orbitals at Gamma and X — tutorial 12 figure.

Builds the bonding and antibonding Bloch orbitals of the 1D H2
molecular crystal at the two endpoints of the band-structure path
(Gamma and X) and renders them as 2D contour slices in the chain
plane. The figure makes the abstract "almost-flat bonding band, wider
antibonding band" picture in the bands+DOS plot concrete: at Gamma
all unit cells are in phase, at X they alternate sign, and you can
see directly why the bonding band barely disperses (neighboring
cells barely overlap their bonding orbitals) while the antibonding
band disperses more (the antibonding lobes between H2 molecules
overlap noticeably).

Implementation: pulls the band coefficients C(k) from the Bloch
Hcore eigenvalue problem (the same one that
``vibeqc.bands.band_structure_hcore`` solves under the hood) and
evaluates the periodic Bloch orbital ψ_{n,k}(r) on the chain plane
via :func:`vibeqc.evaluate_bloch_orbital` (Phase V3).

Run:
    .venv/bin/python examples/plots/h-chain-crystalline-orbitals.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "h-chain-crystalline-orbitals.png"

A = 6.0  # lattice constant (bohr)
D = 1.4  # H2 bond length (bohr)


def main() -> None:
    system = vq.PeriodicSystem(
        1,
        [[A, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [D,   0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    # Real-space lattice matrices for the Hcore Fock + overlap
    opts = vq.LatticeSumOptions()
    S_lat = vq.compute_overlap_lattice(basis, system, opts)
    T_lat = vq.compute_kinetic_lattice(basis, system, opts)
    V_lat = vq.compute_nuclear_lattice(basis, system, opts)

    # Reciprocal lattice along x: b_x = 2*pi/a (Cartesian)
    # Gamma = (0, 0, 0) Cart;  X = (pi/a, 0, 0) Cart
    k_gamma = np.array([0.0, 0.0, 0.0])
    k_x     = np.array([np.pi / A, 0.0, 0.0])

    # Diagonalize at each k; coefficients is (n_bf, n_bands).
    def diag_at(k):
        S_k = vq.bloch_sum(S_lat, k)
        F_k = vq.bloch_sum(T_lat, k) + vq.bloch_sum(V_lat, k)
        sol = vq.diagonalize_bloch(F_k, S_k)
        return np.asarray(sol.energies), np.asarray(sol.coefficients)

    eps_g, C_g = diag_at(k_gamma)
    eps_x, C_x = diag_at(k_x)

    # 2D grid: chain plane (xz) at y = 0
    x_max = 1.6 * A   # show ~3 unit cells, +/- 1.5a
    z_max = 3.0
    nx, nz = 360, 140
    xs = np.linspace(-x_max, x_max, nx)
    zs = np.linspace(-z_max, z_max, nz)
    X, Z = np.meshgrid(xs, zs)
    points = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])

    panels = [
        ("bonding @ $\\Gamma$",     k_gamma, C_g, 0),
        ("bonding @ X",             k_x,     C_x, 0),
        ("antibonding @ $\\Gamma$", k_gamma, C_g, 1),
        ("antibonding @ X",         k_x,     C_x, 1),
    ]

    # Compute psi for each panel; both Gamma and X yield real Bloch
    # phases on a 1D chain, so the orbital is real up to a global sign.
    psi_panels = []
    for _, k, C, band in panels:
        psi = vq.evaluate_bloch_orbital(
            basis, system, points, C, k, band,
        ).reshape(X.shape)
        # Eigenvector phase is arbitrary; pin sign so the largest-amplitude
        # value in the central cell is positive.
        central = psi[len(zs) // 2, np.abs(xs - D / 2).argmin()]
        psi = psi * (1 if central.real >= 0 else -1)
        psi_panels.append(np.real(psi))

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.5), dpi=150,
                              sharex=True, sharey=True)
    axes_flat = axes.ravel()

    # Common color scale across all four panels for fair comparison.
    amp = max(float(np.percentile(np.abs(p), 99.5)) for p in psi_panels)
    levels = np.linspace(-amp, amp, 21)

    # Atom markers for the visible cells
    visible_cells = range(int(np.ceil(-x_max / A)) - 1,
                          int(np.floor(x_max / A)) + 2)
    atoms_xz = []
    for m in visible_cells:
        atoms_xz.append((m * A, 0.0))
        atoms_xz.append((m * A + D, 0.0))

    for ax, (title, _, _, _), psi in zip(axes_flat, panels, psi_panels):
        ax.contourf(X, Z, psi, levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(X, Z, psi, levels=[0.0],
                   colors="black", linewidths=0.4, linestyles="--")
        for (xa, za) in atoms_xz:
            ax.plot(xa, za, "o", ms=5, mfc="white", mec="black", mew=0.8,
                    zorder=5)
        # Visualize the unit-cell boundaries
        for m in visible_cells:
            ax.axvline(m * A - 0.0, color="grey", lw=0.4, ls=":", alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-z_max, z_max)
        ax.set_aspect("equal")

    for ax in axes[1, :]:
        ax.set_xlabel("x along chain (bohr)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z (bohr)")

    fig.suptitle(r"H$_2$ chain (STO-3G, Hcore) — crystalline orbitals at "
                 r"$\Gamma$ and X",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    HARTREE_EV = 27.211386
    print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"  bonding   bandwidth = "
          f"{(eps_x[0] - eps_g[0]) * HARTREE_EV:+.3f} eV "
          f"(eps(X) - eps(G), should be small/positive)")
    print(f"  antibond  bandwidth = "
          f"{(eps_x[1] - eps_g[1]) * HARTREE_EV:+.3f} eV "
          f"(eps(X) - eps(G), should be larger/negative)")
    print(f"  Direct gap @ G = {(eps_g[1] - eps_g[0]) * HARTREE_EV:.3f} eV")
    print(f"  Direct gap @ X = {(eps_x[1] - eps_x[0]) * HARTREE_EV:.3f} eV")


if __name__ == "__main__":
    main()
