"""H-chain Peierls dimerisation animation — tutorial 17 figure.

Animates the 1D hydrogen chain as the dimerisation parameter δ is
swept from 0 (uniform metallic chain) to 1.1 bohr (full
H₂-molecular-crystal limit). Each frame shows:

  - The chain geometry (top panel) — atoms positioned at
    ``R_short = 2.5 - δ`` and ``R_long  = 2.5 + δ`` bohr from the
    cell origin, replicated across 4 unit cells
  - The energy curve (bottom panel) — E(δ) being filled in
    step-by-step, with a marker on the current δ

The animation makes the *mechanism* of the Peierls instability
immediate: as the chain dimerises, the band gap opens (visible
as a vertical drop in the energy plot once δ crosses the
metal→insulator boundary at δ ≈ 0.8 bohr), and the SCF starts
converging where it failed before.

Writes ``docs/_static/plots/h-chain-peierls-animation.gif``.

Run:
    .venv/bin/python examples/plots/h-chain-peierls-animation.py

Uses locally generated energies + convergence flags from
``examples/periodic/output-h-chain-peierls.out``. If the `.out` file
isn't present, runs the upstream input script first.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
PERIODIC = EXAMPLES / "periodic"
OUT_DIR = HERE.parent.parent / "docs" / "_static" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "h-chain-peierls-animation.gif"

# Cell parameter (matches input-h-chain-peierls.py)
CELL = 5.0  # bohr
# Number of unit cells to draw for visual repetition
N_CELLS_VIZ = 4


def _ensure_scan_data() -> Path:
    """Run input-h-chain-peierls.py if the .out isn't present."""
    out = PERIODIC / "output-h-chain-peierls.out"
    if out.is_file():
        return out
    print("Running input-h-chain-peierls.py to generate the scan data…")
    subprocess.check_call(
        [sys.executable, str(PERIODIC / "input-h-chain-peierls.py")],
        cwd=str(PERIODIC),
    )
    return out


def _parse_scan_table(out_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull the (δ, E, converged) table from the .out file."""
    text = out_path.read_text()
    # Lines look like:  "    0.00    2.500    2.500       -1.0336385365      80  NO"
    pattern = re.compile(
        r"^\s+(?P<delta>[\d.]+)\s+[\d.]+\s+[\d.]+\s+"
        r"(?P<energy>-?[\d.]+)\s+\d+\s+(?P<conv>(?:yes|NO|no))\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    deltas = []
    energies = []
    converged = []
    for m in pattern.finditer(text):
        deltas.append(float(m.group("delta")))
        energies.append(float(m.group("energy")))
        converged.append(m.group("conv").lower() == "yes")
    if not deltas:
        raise ValueError(
            f"Could not parse scan table from {out_path} — re-run "
            "examples/periodic/input-h-chain-peierls.py and try again."
        )
    return np.asarray(deltas), np.asarray(energies), np.asarray(converged)


def _atom_positions(delta: float, n_cells: int = N_CELLS_VIZ) -> np.ndarray:
    """Return cartesian (x, y, z) positions for the dimerised chain
    over *n_cells* unit cells, atoms at z = 0."""
    r_short = (CELL - 2 * delta) / 2.0   # short H-H within cell
    r_long  = (CELL + 2 * delta) / 2.0   # long H-H between cells
    pos = []
    for i in range(n_cells):
        # Atoms at +/- r_short/2 around the cell midpoint
        cell_mid = i * CELL + CELL / 2
        pos.append((cell_mid - r_short / 2, 0.0, 0.0))
        pos.append((cell_mid + r_short / 2, 0.0, 0.0))
    return np.array(pos)


def main() -> None:
    out_path = _ensure_scan_data()
    deltas, energies, converged = _parse_scan_table(out_path)
    print(f"Loaded {len(deltas)} points: δ ∈ [{deltas.min()}, {deltas.max()}]")

    n_frames = len(deltas)
    HOLD = 4
    sequence = list(range(n_frames)) + [n_frames - 1] * HOLD

    fig = plt.figure(figsize=(10, 5))
    ax_chain = fig.add_subplot(2, 1, 1)
    ax_e = fig.add_subplot(2, 1, 2)

    fig.suptitle("Peierls instability of the 1D H-chain — RHF / pob-TZVP",
                 fontsize=12, y=0.97)

    e_ref = float(energies[0])

    def draw(i_frame):
        idx = sequence[i_frame]
        delta = float(deltas[idx])

        # --- chain (top panel) ---
        ax_chain.clear()
        pos = _atom_positions(delta)
        # Bonds: short within unit cell (i, i+1 with small separation),
        # long between cells (i+1, i+2)
        for i in range(0, len(pos) - 1, 2):
            # Short bond within cell
            ax_chain.plot(
                [pos[i, 0], pos[i + 1, 0]], [0, 0],
                color="#444", linewidth=2.5, zorder=1,
            )
        for i in range(1, len(pos) - 1, 2):
            # Long bond to next cell
            ax_chain.plot(
                [pos[i, 0], pos[i + 1, 0]], [0, 0],
                color="#bbb", linewidth=1.2, linestyle="--", zorder=1,
            )
        ax_chain.scatter(
            pos[:, 0], np.zeros_like(pos[:, 0]),
            s=180, color="#7f7f7f", edgecolors="black",
            linewidths=0.6, zorder=10,
        )
        ax_chain.set_xlim(0, N_CELLS_VIZ * CELL)
        ax_chain.set_ylim(-1.5, 1.5)
        ax_chain.set_yticks([])
        ax_chain.set_xlabel("z / bohr")
        title = (f"δ = {delta:.2f} bohr   "
                 f"R_short = {(CELL - 2*delta)/2:.2f} bohr   "
                 f"R_long  = {(CELL + 2*delta)/2:.2f} bohr")
        if converged[idx]:
            title += "   ✓ SCF converged"
        else:
            title += "   ✗ SCF stalled (metallic regime)"
        ax_chain.set_title(title, fontsize=10)

        # --- energy curve (bottom panel) ---
        ax_e.clear()
        de = (energies - e_ref) * 1000   # meV
        # Draw the whole curve as a faint guide
        ax_e.plot(deltas, de, color="#aaa", alpha=0.4,
                  marker="o", markersize=3, linestyle="-")
        # Highlight the points up to current frame
        ax_e.plot(deltas[: idx + 1], de[: idx + 1],
                  color="C0", marker="o", markersize=5)
        # Current point
        ax_e.scatter([delta], [de[idx]], color="C3", s=120, zorder=5)
        # Convergence-failure region as a colored band
        ax_e.axvspan(0, 0.7, alpha=0.10, color="red")
        ax_e.text(0.35, max(de) * 0.85, "metallic\n(SCF stalls)",
                  fontsize=9, color="darkred", ha="center")
        ax_e.set_xlabel("δ / bohr  (dimerisation parameter)")
        ax_e.set_ylabel("ΔE / meV  (rel. to δ = 0)")
        ax_e.set_title("Total energy per cell along the dimerisation path",
                       fontsize=10)
        ax_e.set_xlim(deltas.min() - 0.05, deltas.max() + 0.05)
        ax_e.grid(alpha=0.3)

    fps = 3
    print(f"Rendering {len(sequence)} frames at {fps} fps…")
    anim = animation.FuncAnimation(
        fig, draw, frames=len(sequence), interval=1000 // fps, blit=False,
    )
    anim.save(str(OUT_PATH), writer=animation.PillowWriter(fps=fps))
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
