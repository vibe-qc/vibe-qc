"""H2O geometry-optimization trajectory animation — tutorial 08 figure.

Runs ``examples/molecular/input-h2o-opt.py`` (HF/6-31G* BFGS relaxation),
reads the resulting `.traj`, renders an animated GIF showing the
H2O molecule relaxing into its equilibrium geometry alongside an
energy-vs-step subplot. Writes
``docs/_static/plots/h2o-opt-trajectory-animation.gif``.

Run:
    .venv/bin/python examples/plots/h2o-opt-trajectory-animation.py

Wall: ~10 s (BFGS converges in <10 steps + matplotlib render).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
OUT_DIR = HERE.parent.parent / "docs" / "_static" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "h2o-opt-trajectory-animation.gif"

ATOM_COLORS = {"O": "#d62728", "H": "#7f7f7f"}
ATOM_RADII = {"O": 0.32, "H": 0.16}


def _ensure_traj() -> Path:
    """Run input-h2o-opt.py if the trajectory isn't already there."""
    traj = EXAMPLES / "output-h2o-opt.traj"
    if traj.is_file():
        return traj
    print("Running input-h2o-opt.py to generate the trajectory…")
    subprocess.check_call(
        [sys.executable, str(EXAMPLES / "input-h2o-opt.py")],
        cwd=str(EXAMPLES),
    )
    return traj


def render_frame(ax3d, ax_e, frame_idx, frames, energies):
    """Draw one frame: 3D structure + energy timeline marker."""
    atoms = frames[frame_idx]
    pos = atoms.positions
    syms = list(atoms.symbols)

    # 3D ball-and-stick
    ax3d.clear()
    bonds = [(0, 1), (0, 2)]
    for i, j in bonds:
        ax3d.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            [pos[i, 2], pos[j, 2]],
            color="#444", linewidth=2.5, zorder=1,
        )
    for i, sym in enumerate(syms):
        ax3d.scatter(
            *pos[i],
            color=ATOM_COLORS.get(sym, "#999"),
            s=900 * ATOM_RADII.get(sym, 0.2) ** 2,
            edgecolors="black", linewidths=0.5, zorder=10,
        )
    ax3d.set_xlim(-1.4, 1.4)
    ax3d.set_ylim(-1.4, 1.4)
    ax3d.set_zlim(-1.4, 1.0)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.xaxis.line.set_color("none")
    ax3d.yaxis.line.set_color("none")
    ax3d.zaxis.line.set_color("none")
    ax3d.xaxis.set_pane_color((1, 1, 1, 0))
    ax3d.yaxis.set_pane_color((1, 1, 1, 0))
    ax3d.zaxis.set_pane_color((1, 1, 1, 0))
    ax3d.view_init(elev=15, azim=-70)
    ax3d.set_title(f"step {frame_idx}", fontsize=11)

    # Energy curve + step marker
    ax_e.clear()
    e_rel_meV = (np.asarray(energies) - energies[-1]) * 1000
    ax_e.plot(range(len(energies)), e_rel_meV,
              marker="o", markersize=4, color="C0")
    ax_e.scatter([frame_idx], [e_rel_meV[frame_idx]],
                 color="C3", s=80, zorder=5)
    ax_e.set_xlabel("BFGS step")
    ax_e.set_ylabel("E - E_min  /  meV")
    ax_e.set_title("Energy convergence")
    ax_e.set_xlim(-0.5, len(energies) - 0.5)
    ax_e.grid(alpha=0.3)


def main() -> None:
    traj_path = _ensure_traj()
    print(f"Reading trajectory from {traj_path.relative_to(EXAMPLES.parent)}")
    frames = list(read(str(traj_path), index=":"))
    energies = [f.get_potential_energy() for f in frames]
    print(f"  {len(frames)} frames")
    print(f"  E[0] = {energies[0]:.6f} eV   E[-1] = {energies[-1]:.6f} eV")
    print(f"  ΔE   = {(energies[0] - energies[-1])*1000:.2f} meV")

    # Layout: 3D structure on left, energy plot on right
    fig = plt.figure(figsize=(9, 4))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_e = fig.add_subplot(1, 2, 2)

    fig.suptitle("H₂O / RHF / 6-31G* — BFGS geometry optimization",
                 fontsize=12, y=0.97)

    # Hold last frame for a moment so the loop visually pauses at
    # the converged geometry.
    n_frames = len(frames)
    HOLD = 3
    sequence = list(range(n_frames)) + [n_frames - 1] * HOLD

    def draw(i):
        render_frame(ax3d, ax_e, sequence[i], frames, energies)
        return [ax3d, ax_e]

    fps = 4
    print(f"Rendering {len(sequence)} frames at {fps} fps…")
    anim = animation.FuncAnimation(
        fig, draw, frames=len(sequence), interval=1000 // fps, blit=False,
    )
    anim.save(str(OUT_PATH), writer=animation.PillowWriter(fps=fps))
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
