"""1D H2 chain — band structure + density of states.

Run:
    .venv/bin/python input-h-chain-bands.py

Produces:
    output-h-chain-bands.png    — bands (Γ → X) + horizontal DOS

Builds the same H2 molecular crystal as ``input-h-chain-uniform.py``
(one H2 per cell, lattice ``a = 6 bohr`` along x, vacuum elsewhere)
and plots the *non-interacting* (Hcore = T + V) bands and DOS. With a
two-orbital STO-3G basis there are exactly two bands: a bonding band
and an antibonding band, well-separated everywhere.

The aim is a quick visual sanity check of the periodic infrastructure
(Bloch sums, k-path, MP-mesh broadening), not a self-consistent
solid-state calculation.
"""

from pathlib import Path

import vibeqc as vq

OUT = Path(__file__).resolve().parent / "output-h-chain-bands.png"


def main() -> None:
    a = 6.0  # bohr
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.4, 0.0, 0.0]),
    ]
    system = vq.PeriodicSystem(
        1,
        [[a, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        atoms,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    # Γ → X path along the chain axis (only one direction matters in 1D).
    kpath = vq.kpath_from_segments(
        system,
        [((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X")],
        points_per_segment=40,
    )
    bands = vq.band_structure_hcore(
        system, basis, kpath, n_electrons_per_cell=2,
    )
    dos = vq.density_of_states_hcore(
        system, basis, [80, 1, 1],
        sigma=0.02, n_electrons_per_cell=2,
    )

    print(f"Γ:  {bands.energies[0]}")
    print(f"X:  {bands.energies[-1]}")
    print(f"E_F (HOMO): {bands.e_fermi:.6f} Ha")
    import numpy as np
    area = np.trapezoid(dos.dos, dos.energies)
    print(f"DOS integrated: {area:.4f} (should be ≈ {basis.nbasis})")

    # Plot — matplotlib is optional; skip gracefully if missing.
    try:
        from vibeqc.plot import bands_dos_figure
        fig = bands_dos_figure(bands, dos, title="H2 chain (STO-3G, Hcore)")
        fig.savefig(OUT, dpi=150)
        print(f"Wrote {OUT}")
    except ImportError as e:
        print(f"(matplotlib not available: {e}) — skipped figure")


if __name__ == "__main__":
    main()
