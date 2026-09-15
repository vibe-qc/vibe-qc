"""H2O normal-mode animation — tutorial 09 figure.

Optimises water at HF/STO-3G, computes the analytic RHF Hessian
(Phase 17b-3), and renders an animated GIF showing the three
vibrational normal modes side-by-side — bend, symmetric stretch,
antisymmetric stretch — oscillating along their respective
eigenvectors. Drops the six near-zero translation/rotation modes.

Writes ``docs/_static/plots/water-normal-mode-animation.gif`` —
the animation embedded in tutorial 09. ~30 s wall (optimization +
analytic Hessian + 60-frame matplotlib render).

Run:
    .venv/bin/python examples/plots/water-normal-mode-animation.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive renderer for GIF output
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.optimize import BFGS

import vibeqc as vq
from vibeqc.ase import VibeQC


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent.parent / "docs" / "_static" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "water-normal-mode-animation.gif"


# Animation parameters
N_FRAMES = 24                  # one period of oscillation
AMPLITUDE = 0.30               # Å of cartesian displacement at peak
FPS = 12
LOOP_PERIODS = 1               # number of full periods to render
ATOM_COLORS = {"O": "#d62728", "H": "#7f7f7f"}
ATOM_RADII = {"O": 0.32, "H": 0.16}


def relax_h2o() -> Atoms:
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.000, 0.000,  0.000),
            (0.762, 0.000, -0.566),
            (-0.762, 0.000, -0.566),
        ],
    )
    atoms.calc = VibeQC(basis="sto-3g")
    BFGS(atoms, logfile=None).run(fmax=0.001, steps=80)
    return atoms


def compute_normal_modes(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """Run the analytic RHF Hessian, return (frequencies cm⁻¹,
    mass-weighted eigenvectors shape (3N, 3N))."""
    # Build vibe-qc Molecule + basis at the relaxed geometry
    pos_bohr = atoms.positions / 0.529177210903   # Å → bohr
    mol = vq.Molecule(
        [vq.Atom(int(z), list(p))
         for z, p in zip(atoms.numbers, pos_bohr)],
    )
    basis = vq.BasisSet(mol, "sto-3g")
    rhf_result = vq.run_rhf(mol, basis)
    hess = vq.compute_hessian_rhf_analytic(
        mol, basis, rhf_result, basis_name="sto-3g",
    )
    # HessianResult exposes: .hessian (Hartree/bohr²),
    # .frequencies_cm1 (cm⁻¹, complex for imaginary modes),
    # .normal_modes (3N, 3N) cartesian displacement vectors.
    freqs = np.asarray(hess.frequencies_cm1)
    modes = np.asarray(hess.normal_modes)
    return freqs, modes


def vibrational_modes(
    freqs: np.ndarray, modes: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    """Return the three real vibrational modes of water — sorted by
    frequency, dropping translations/rotations (|ω| < 100 cm⁻¹)."""
    real_idx = [i for i, w in enumerate(freqs)
                if not np.iscomplex(w) and abs(w.real) > 100.0]
    real_idx.sort(key=lambda i: float(freqs[i].real))
    return [(float(freqs[i].real), modes[:, i]) for i in real_idx[-3:]]


def setup_axes(fig, n_modes: int) -> list:
    axes = []
    for i in range(n_modes):
        ax = fig.add_subplot(1, n_modes, i + 1, projection="3d")
        axes.append(ax)
    return axes


def render_frame(
    ax, atoms_eq: Atoms, mode_vec: np.ndarray, displacement: float,
) -> None:
    """Draw a single frame: atoms + bonds at displaced geometry."""
    ax.clear()
    # Reshape mode vector (3N,) → (N, 3) cartesian displacement per atom
    disp = mode_vec.reshape(-1, 3)
    # Normalize
    disp = disp / np.linalg.norm(disp)
    # Apply
    pos = atoms_eq.positions + displacement * disp
    syms = list(atoms_eq.symbols)

    # Bonds (1-2 and 1-3 — O-H1 and O-H2 for water)
    bonds = [(0, 1), (0, 2)]
    for i, j in bonds:
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            [pos[i, 2], pos[j, 2]],
            color="#444", linewidth=2.5, zorder=1,
        )

    # Atoms — use small spheres (scatter with size, edge color)
    for i, sym in enumerate(syms):
        ax.scatter(
            *pos[i],
            color=ATOM_COLORS.get(sym, "#999"),
            s=900 * ATOM_RADII.get(sym, 0.2) ** 2,
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    # Hide the cube outline
    ax.xaxis.line.set_color("none")
    ax.yaxis.line.set_color("none")
    ax.zaxis.line.set_color("none")
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.view_init(elev=15, azim=-70)


def main() -> None:
    print("Optimizing H2O at HF/STO-3G…")
    atoms = relax_h2o()
    print(f"  E(min) = {atoms.get_potential_energy():.6f} eV")

    print("Computing analytic RHF Hessian…")
    freqs, modes = compute_normal_modes(atoms)
    vib_modes = vibrational_modes(freqs, modes)
    print("Vibrational modes (cm⁻¹):")
    for w, _ in vib_modes:
        print(f"  ω = {w:8.2f}")

    # Mode names. Order is bend (low) → asym stretch (mid) → sym stretch (high)
    # for water; harmonic ordering may swap sym/asym slightly with the basis.
    n_modes = len(vib_modes)
    mode_labels = ["bend", "asym stretch", "sym stretch"]
    if len(mode_labels) != n_modes:
        mode_labels = [f"mode {i+1}" for i in range(n_modes)]

    fig = plt.figure(figsize=(3.5 * n_modes, 4.0))
    axes = setup_axes(fig, n_modes)

    # Per-frame oscillation amplitude — sinusoid over LOOP_PERIODS periods
    total_frames = N_FRAMES * LOOP_PERIODS
    phases = np.linspace(0, 2 * np.pi * LOOP_PERIODS, total_frames,
                         endpoint=False)

    def draw(frame_idx: int):
        a = AMPLITUDE * np.sin(phases[frame_idx])
        for ax, (w, vec), label in zip(axes, vib_modes, mode_labels):
            render_frame(ax, atoms, vec, a)
            ax.set_title(f"{label}\nω = {w:.0f} cm⁻¹", fontsize=10,
                         pad=2)
        fig.suptitle(
            "H₂O / RHF / STO-3G — analytic CPHF Hessian normal modes",
            fontsize=11, y=0.98,
        )
        return list(axes)

    print(f"Rendering {total_frames} frames @ {FPS} fps …")
    anim = animation.FuncAnimation(
        fig, draw, frames=total_frames, interval=1000 // FPS, blit=False,
    )
    anim.save(str(OUT_PATH), writer=animation.PillowWriter(fps=FPS))
    print(f"Wrote {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
