"""H2O frontier-orbital cuts — tutorial 11 figure.

Renders 2D contour slices of the HOMO-1, HOMO, LUMO, and total electron
density of water (HF/6-31G*) directly via vibe-qc's ``evaluate_ao`` —
no external isosurface viewer needed. Writes
``docs/_static/plots/water-frontier-orbitals.png`` — the figure embedded
in tutorial 11.

The molecule sits with all atoms at x = 0 (molecular plane = yz). The
1b1 HOMO is the oxygen 2p lone pair pointing *out* of the molecular
plane, so it's plotted in the perpendicular xz-plane (y = 0); the
3a1 HOMO-1, 4a1 LUMO and the density are plotted in the molecular
plane itself.

Run:
    .venv/bin/python examples/plots/water-frontier-orbitals.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "water-frontier-orbitals.png"


def grid_in_plane(plane: str, n: int = 200, half_size: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, points) where X, Y are 2D meshgrids and points is the
    flat (n*n, 3) array of Cartesian coordinates in bohr.

    plane = "yz": x = 0, axes are (y, z)
    plane = "xz": y = 0, axes are (x, z)
    """
    u = np.linspace(-half_size, half_size, n)
    v = np.linspace(-half_size, half_size + 1.0, n)  # extra room above
    U, V = np.meshgrid(u, v)
    if plane == "yz":
        pts = np.column_stack([np.zeros(U.size), U.ravel(), V.ravel()])
    elif plane == "xz":
        pts = np.column_stack([U.ravel(), np.zeros(U.size), V.ravel()])
    else:
        raise ValueError(plane)
    return U, V, pts


def plot_orbital(ax, U, V, phi, title, atoms_in_plane, *, signed=True):
    """Filled contour of phi(r) in 2D. Symmetric divergent color map for
    signed orbitals; sequential for densities."""
    if signed:
        amp = float(np.percentile(np.abs(phi), 99.5))
        levels = np.linspace(-amp, amp, 21)
        cs = ax.contourf(U, V, phi, levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(U, V, phi, levels=[0.0], colors="black",
                   linewidths=0.4, linestyles="--")
    else:
        amp = float(np.percentile(phi, 99.5))
        levels = np.linspace(0.0, amp, 21)
        cs = ax.contourf(U, V, phi, levels=levels, cmap="viridis")
    for (label, x, y) in atoms_in_plane:
        ax.plot(x, y, "o", ms=11, mfc="white", mec="black", mew=1.2, zorder=5)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    return cs


def main() -> None:
    # Water at the same geometry the rest of the tutorials use (bohr).
    mol = vq.Molecule([
        vq.Atom(8, [0.0,  0.00,  0.00]),
        vq.Atom(1, [0.0,  1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])
    basis = vq.BasisSet(mol, "6-31g*")
    r = vq.run_rhf(mol, basis)
    C = np.asarray(r.mo_coeffs)
    P = np.asarray(r.density)
    n_occ = mol.n_electrons() // 2
    homo_idx = n_occ - 1

    # -- Three slices in the molecular (yz) plane --------------------------
    Uyz, Vyz, pts_yz = grid_in_plane("yz", n=240)
    chi_yz = vq.evaluate_ao(basis, pts_yz)        # (n_pts, n_bf)
    homo_m1_yz = (chi_yz @ C[:, homo_idx - 1]).reshape(Uyz.shape)
    lumo_yz   = (chi_yz @ C[:, homo_idx + 1]).reshape(Uyz.shape)
    rho_yz    = np.einsum("pi,ij,pj->p", chi_yz, P, chi_yz).reshape(Uyz.shape)

    # -- HOMO in the perpendicular (xz) plane through y = 0 ----------------
    Uxz, Vxz, pts_xz = grid_in_plane("xz", n=240)
    chi_xz = vq.evaluate_ao(basis, pts_xz)
    homo_xz = (chi_xz @ C[:, homo_idx]).reshape(Uxz.shape)

    # -- Atom markers for each panel ---------------------------------------
    atoms_yz = [("O", 0.0, 0.0), ("H", 1.43, -0.98), ("H", -1.43, -0.98)]
    atoms_xz = [("O", 0.0, 0.0)]  # both H atoms are out of this slice

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5), dpi=150)

    plot_orbital(axes[0, 0], Uyz, Vyz, homo_m1_yz,
                 r"HOMO$-$1 (3a$_1$, $\sigma$ bonding)" "\n"
                 r"slice: molecular plane ($x=0$)",
                 atoms_yz, signed=True)
    plot_orbital(axes[0, 1], Uxz, Vxz, homo_xz,
                 r"HOMO (1b$_1$, O lone pair)" "\n"
                 r"slice: $\perp$ molecular plane ($y=0$)",
                 atoms_xz, signed=True)
    plot_orbital(axes[1, 0], Uyz, Vyz, lumo_yz,
                 r"LUMO (4a$_1$, $\sigma^*$ antibonding)" "\n"
                 r"slice: molecular plane ($x=0$)",
                 atoms_yz, signed=True)
    plot_orbital(axes[1, 1], Uyz, Vyz, rho_yz,
                 r"total electron density $\rho(\mathbf{r})$" "\n"
                 r"slice: molecular plane ($x=0$)",
                 atoms_yz, signed=False)

    for ax in axes[:, 0]:
        ax.set_ylabel("z (bohr)")
    for ax in axes[1, :]:
        ax.set_xlabel("axis-in-plane (bohr)")
    axes[0, 0].set_xlabel("y (bohr)", labelpad=2)
    axes[0, 1].set_xlabel("x (bohr)", labelpad=2)
    axes[1, 0].set_xlabel("y (bohr)")
    axes[1, 1].set_xlabel("y (bohr)")

    fig.suptitle(r"H$_2$O / 6-31G* — frontier-orbital and density slices",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    eps = np.asarray(r.mo_energies)
    HARTREE_EV = 27.211386
    print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"  E(SCF)        = {r.energy:.6f} Ha")
    print(f"  ε(HOMO-1) = {eps[homo_idx-1]*HARTREE_EV:7.3f} eV")
    print(f"  ε(HOMO)   = {eps[homo_idx]*HARTREE_EV:7.3f} eV")
    print(f"  ε(LUMO)   = {eps[homo_idx+1]*HARTREE_EV:7.3f} eV")


if __name__ == "__main__":
    main()
