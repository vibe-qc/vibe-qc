"""NH₃ umbrella inversion — tutorial 19 animation.

Reads the 7-image NEB trajectory written by
``examples/workflows/input-nh3-umbrella-neb.py`` and renders an animated GIF
of the umbrella flip — ping-pong forward (0 → 6) then reverse
(6 → 0) so the loop reads naturally.

Each frame shows the molecule in 3D with bonds redrawn at every
step, plus a small inset reaction-coordinate marker indicating which
image is currently displayed and the energy relative to the reactant.
Output:

    docs/_static/plots/nh3-umbrella-neb-animation.gif

Requires ``ase`` for trajectory I/O and ``matplotlib`` (>= 3.7) for
the PillowWriter.

Run:
    .venv/bin/python examples/plots/nh3-umbrella-neb-animation.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D-projection registration)

from ase.io.trajectory import Trajectory


HERE = Path(__file__).resolve().parent
TRAJ_IN  = HERE.parent / "workflows" / "output-nh3-umbrella-neb.traj"
GIF_OUT  = HERE.parent.parent / "docs" / "_static" / "plots" / "nh3-umbrella-neb-animation.gif"

HARTREE_PER_EV = 1.0 / 27.211386
KCAL_PER_HARTREE = 627.509474063

# Atom drawing parameters
COLORS = {"N": "#3050a0", "H": "#dddddd"}
SIZES  = {"N": 280,        "H": 110}
BOND_THRESHOLD_ANGSTROM = 1.4   # N–H bonds at ~1.0 Å fall well inside


def find_bonds(symbols, positions):
    """Pair (i, j) for every i, j with i < j and r_ij < threshold."""
    pairs = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < BOND_THRESHOLD_ANGSTROM:
                pairs.append((i, j))
    return pairs


def main() -> None:
    t0 = time.perf_counter()

    if not TRAJ_IN.exists():
        raise SystemExit(
            f"NEB trajectory not found at {TRAJ_IN}. Re-run "
            f"examples/workflows/input-nh3-umbrella-neb.py first."
        )

    traj = Trajectory(str(TRAJ_IN))
    forward = list(range(len(traj)))
    reverse = list(range(len(traj) - 2, 0, -1))   # skip endpoints to avoid pause
    sequence = forward + reverse                  # ping-pong loop

    energies_eV = np.array([img.get_potential_energy() for img in traj])
    energies_kcal = (energies_eV - energies_eV[0]) * HARTREE_PER_EV * KCAL_PER_HARTREE

    print(f"  loaded {len(traj)} NEB images")
    print(f"  energies (kcal/mol, relative to reactant): "
          f"{np.array2string(energies_kcal, precision=2)}")
    print(f"  barrier: {energies_kcal.max():.2f} kcal/mol "
          f"at image {int(np.argmax(energies_kcal))}")

    fig = plt.figure(figsize=(7.5, 5.5), dpi=120)
    ax3d = fig.add_axes([0.02, 0.20, 0.70, 0.78], projection="3d")
    ax_e = fig.add_axes([0.78, 0.20, 0.20, 0.78])
    ax_text = fig.add_axes([0.02, 0.02, 0.96, 0.14])
    ax_text.axis("off")

    # Set fixed limits so the molecule doesn't reframe each step
    all_pos = np.vstack([img.get_positions() for img in traj])
    pad = 0.4
    xmin, xmax = all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad
    ymin, ymax = all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad
    zmin, zmax = all_pos[:, 2].min() - pad, all_pos[:, 2].max() + pad
    half = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    cx, cy, cz = 0.5 * (xmax + xmin), 0.5 * (ymax + ymin), 0.5 * (zmax + zmin)

    def setup_3d():
        ax3d.set_xlim(cx - half, cx + half)
        ax3d.set_ylim(cy - half, cy + half)
        ax3d.set_zlim(cz - half, cz + half)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_xlabel("x (Å)", labelpad=-12, fontsize=8)
        ax3d.set_ylabel("y (Å)", labelpad=-12, fontsize=8)
        ax3d.set_zlabel("z (Å)", labelpad=-12, fontsize=8)
        ax3d.tick_params(axis="both", labelsize=7, pad=-3)
        ax3d.view_init(elev=18, azim=-65)

    def setup_energy():
        ax_e.plot(range(len(energies_kcal)), energies_kcal,
                  "-o", color="#1f77b4", markersize=6)
        ax_e.set_xlabel("image", fontsize=9)
        ax_e.set_ylabel("ΔE (kcal/mol)", fontsize=9)
        ax_e.set_title("MEP", fontsize=10)
        ax_e.tick_params(labelsize=8)
        ax_e.grid(alpha=0.3)

    setup_energy()

    def render_frame(frame_idx: int):
        ax3d.cla()
        setup_3d()

        img = traj[frame_idx]
        pos = img.get_positions()
        sym = img.get_chemical_symbols()

        # Bonds first so they sit behind atoms
        for (i, j) in find_bonds(sym, pos):
            ax3d.plot([pos[i, 0], pos[j, 0]],
                      [pos[i, 1], pos[j, 1]],
                      [pos[i, 2], pos[j, 2]],
                      color="#404040", linewidth=2.0, zorder=1)
        for k, (s, p) in enumerate(zip(sym, pos)):
            ax3d.scatter(*p, s=SIZES.get(s, 100), c=COLORS.get(s, "magenta"),
                         edgecolor="black", linewidths=0.8, zorder=3)

        # Energy-marker dot on the inset
        for line in list(ax_e.lines):
            if getattr(line, "_marker_dot", False):
                line.remove()
        marker, = ax_e.plot(frame_idx, energies_kcal[frame_idx], "o",
                            ms=10, color="#d62728", zorder=10)
        marker._marker_dot = True

        # Bottom text strip
        ax_text.cla()
        ax_text.axis("off")
        ax_text.text(
            0.5, 0.55,
            f"NH$_3$ umbrella inversion — image {frame_idx + 1} / "
            f"{len(traj)}    "
            f"ΔE = {energies_kcal[frame_idx]:+.2f} kcal/mol",
            ha="center", va="center", fontsize=11,
        )

    # Initial frame
    render_frame(0)

    anim = FuncAnimation(
        fig, render_frame,
        frames=sequence,
        interval=400,           # ms per frame
        blit=False,
        repeat=True,
    )

    GIF_OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=4)
    anim.save(GIF_OUT, writer=writer, dpi=100)

    print(f"\nWrote {GIF_OUT.relative_to(HERE.parent.parent)}  "
          f"({time.perf_counter()-t0:.1f} s)")


if __name__ == "__main__":
    main()
