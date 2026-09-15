"""Projected DOS for a 1D LiH crystal — tutorial 21 figure.

Builds the H-core band structure + atom- and orbital-projected DOS for
a uniform 1D LiH chain (alternating Li and H atoms, lattice parameter
5.5 bohr, sto-3g basis) and writes
``docs/_static/plots/lih-chain-pdos.png`` — the figure embedded in
tutorial 21. Pure Hcore (no SCF), runs in well under a second.

The 1D LiH chain is a clean two-element pedagogical system: the Li 1s
core sits ~50 eV below the valence band, the bonding band is a Li-2s
/ H-1s mixture biased toward H (the H atom is more electronegative),
and the Li-2p contribution lives in the conduction band. PDOS makes
all of that visible at a glance.

Run:
    .venv/bin/python examples/plots/lih-chain-pdos.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt

import vibeqc as vq
from vibeqc.plot import bands_pdos_figure

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "lih-chain-pdos.png"


def main() -> None:
    t0 = time.perf_counter()

    # 1D LiH chain: alternating Li and H, lattice parameter 5.5 bohr.
    a = 5.5
    atoms = [
        vq.Atom(3, [0.0,         0.0, 0.0]),   # Li at 0
        vq.Atom(1, [0.5 * a,     0.0, 0.0]),   # H at a/2
    ]
    system = vq.PeriodicSystem(
        1,
        [[a, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        atoms,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    # Band-structure path Γ → X (ka/π ∈ [0, 1]).
    kpath = vq.kpath_from_segments(
        system,
        [((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X")],
        points_per_segment=120,
    )
    bs = vq.band_structure_hcore(
        system, basis, kpath, n_electrons_per_cell=4,
    )

    # Atom-and-l projected DOS on a 200×1×1 mesh.
    pdos = vq.density_of_states_projected_hcore(
        system, basis, [200, 1, 1],
        projection="atoms_l",
        sigma=0.005,
        n_electrons_per_cell=4,
    )

    print(f"  basis: {basis.nbasis} AOs")
    print(f"  PDOS group labels: {pdos.group_labels}")
    print(f"  E_F = {pdos.e_fermi:.4f} Ha = "
          f"{pdos.e_fermi * 27.211386:.3f} eV")

    # Custom colors: Li in blue, H in red, by angular momentum
    colours = {
        "Li1-s": "#1f77b4",
        "Li1-p": "#9ecae1",
        "H2-s":  "#d62728",
    }

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = bands_pdos_figure(
        bs, pdos,
        units="eV",
        shift_to_fermi=True,
        stack=False,
        show_total=True,
        colors=colours,
        title="1D LiH chain (a = 5.5 bohr, sto-3g) — Hcore bands + PDOS",
    )
    # bands_pdos_figure returns a Figure with two axes that share y;
    # window them to skip the Li-1s core at ~-80 eV so the valence /
    # low-conduction structure is readable.
    fig.set_size_inches(9.0, 4.8)
    for ax in fig.axes:
        ax.set_ylim(-15.0, 35.0)
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}  "
          f"({time.perf_counter()-t0:.2f} s)")


if __name__ == "__main__":
    main()
