"""Si diamond — Hcore band structure + DOS — CRYSTAL Tutorial port.

Direct port of the canonical CRYSTAL "Band Structure / DOS" tutorial
onto vibe-qc. Silicon in the diamond structure (Fd-3m) is the
textbook covalent semiconductor — band-structure plots from this
system appear in every solid-state textbook (Kittel, Ashcroft-Mermin,
Marder).

We use **sto-3g** here, not pob-TZVP, to keep the example fast and
numerically modest. The minimal basis is not converged for silicon and
the resulting spectrum is not a quantitative solid-state prediction.

This script computes **non-interacting (Hcore) bands** along the
standard HPKOT k-path for the cubic-F Bravais lattice. Hcore contains
only kinetic and nuclear-attraction terms. Electron-electron terms can
change band ordering, dispersion, extrema, and gaps, so this plot is a
teaching exercise for k paths, DOS, and projections rather than a proxy
for an HF or DFT band structure.

Why no SCF? The Hcore route is deliberately inexpensive and isolates
the band-sampling infrastructure. Use a route-specific post-SCF band
helper for a scientific spectrum; do not compare the Hcore gap with an
experimental or self-consistent silicon gap.

What this exercises:

  - Build a conventional cubic Si cell (8 atoms, Fd-3m diamond)
  - Auto-detect Bravais via spglib + seekpath HPKOT k-path (Phase K3)
    — for cubic-F: Γ → X → U|K → Γ → L → W → X
  - Hcore band structure along the path
  - Hcore DOS + atom-l projected DOS (Si-s, Si-p)
  - Render bands + DOS + PDOS triple panel via matplotlib

Outputs (next to this script in ``examples/``):
    output-si-bands.png       — bands + DOS + PDOS triple panel

Wall: ~15-30 s — pure integral builds, no SCF iterations.

The CRYSTAL Tutorial parity matrix tracks this script under the
"Quick tour: band structure" row. See ``docs/roadmap.md`` §
"Tutorial parity (ORCA + CRYSTAL)".

Run:
    .venv/bin/python examples/periodic/input-si-bands.py
"""

from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PNG_OUT = HERE / "output-si-bands.png"

HARTREE_TO_EV = 27.211386245988

# Silicon lattice parameter (experimental, RT, diamond structure).
A_ANG = 5.4307
A_BOHR = A_ANG / 0.529177210903


def build_si_diamond() -> vq.PeriodicSystem:
    """Conventional cubic Si cell — 8 atoms, Fd-3m diamond structure.

    Atoms at the eight standard diamond positions: 4 FCC + 4 displaced
    by (1/4, 1/4, 1/4).
    """
    lat = A_BOHR * np.eye(3)
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    displaced = [(fx + 0.25, fy + 0.25, fz + 0.25) for fx, fy, fz in fcc]
    unit_cell = []
    for fx, fy, fz in fcc + displaced:
        unit_cell.append(vq.Atom(14, [fx * A_BOHR, fy * A_BOHR, fz * A_BOHR]))
    return vq.PeriodicSystem(dim=3, lattice=lat, unit_cell=unit_cell)


def main() -> None:
    plog = vq.ProgressLogger(verbose=True)
    plog.banner("Silicon diamond  /  Hcore bands  /  sto-3g")

    with plog.stage("setup", detail="build cell + basis + symmetry"):
        system = build_si_diamond()
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        vq.attach_symmetry(system, symprec=1e-4)
        plog.info(f"cell:       {A_ANG} Å cubic, {len(system.unit_cell)} atoms")
        plog.info(
            f"spacegroup: {system.symmetry.international_symbol} "
            f"(SG {system.symmetry.number}, "
            f"point group {system.symmetry.point_group})"
        )
        plog.info(f"basis:      sto-3g (minimal — see header note), "
                  f"{basis.nbasis} bf per cell")

    n_electrons = sum(int(at.Z) for at in system.unit_cell)

    # ---- 1. Band path via auto-detected HPKOT (cubic-F) -------------
    with plog.stage("band_path", detail="HPKOT path via seekpath"):
        band_path = vq.KPoints.band_path(system)
        plog.info(
            f"path: {' → '.join(label for _, label in band_path.labels)}"
        )

    # ---- 2. Hcore bands ---------------------------------------------
    with plog.stage("bands_hcore"):
        bands = vq.band_structure_hcore(
            system, basis, band_path.to_kpath(),
            n_electrons_per_cell=n_electrons,
        )

    # ---- 3. Hcore total DOS + atom-l projected DOS ------------------
    with plog.stage("dos_hcore"):
        dos = vq.density_of_states_hcore(
            system, basis, [4, 4, 4],
            sigma=0.01, n_electrons_per_cell=n_electrons,
        )
    with plog.stage("pdos_hcore"):
        pdos = vq.density_of_states_projected_hcore(
            system, basis, [4, 4, 4],
            projection="atoms_l",
            sigma=0.01, n_electrons_per_cell=n_electrons,
        )

    # Direct gap at the first k of the path (Γ for cubic-F).
    n_occ = n_electrons // 2
    gap_at_first_k = (
        float(bands.energies[0, n_occ])
        - float(bands.energies[0, n_occ - 1])
    ) * HARTREE_TO_EV
    plog.info(
        f"Hcore direct gap at Γ: {gap_at_first_k:.2f} eV  "
        "(teaching model only; not an HF, DFT, or experimental gap)"
    )

    # ---- 4. Plot bands + DOS + PDOS ---------------------------------
    with plog.stage("plot", detail="bands + DOS + PDOS triple panel"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            plog.warn("matplotlib not available - skipping plot")
            return

        fig = plt.figure(figsize=(11, 4.6), dpi=150)
        gs = fig.add_gridspec(
            1, 3, width_ratios=[3.0, 1.0, 1.6], wspace=0.05,
        )
        ax_b = fig.add_subplot(gs[0, 0])
        ax_d = fig.add_subplot(gs[0, 1], sharey=ax_b)
        ax_p = fig.add_subplot(gs[0, 2], sharey=ax_b)

        e_fermi = bands.e_fermi if bands.e_fermi is not None else 0.0
        e_b = (bands.energies - e_fermi) * HARTREE_TO_EV
        e_d = (dos.energies - e_fermi) * HARTREE_TO_EV
        e_p = (pdos.energies - e_fermi) * HARTREE_TO_EV

        # Highlight the lower (occupied) and upper (virtual) bands separately.
        for i in range(e_b.shape[1]):
            color = "#1f77b4" if i < n_occ else "#aaaaaa"
            ax_b.plot(bands.kpath.distances, e_b[:, i],
                      color=color, linewidth=1.0)
        ax_b.axhline(0.0, color="black", linewidth=0.8,
                     linestyle="--", alpha=0.7)
        tick_pos, tick_txt = zip(*bands.kpath.labels)
        ax_b.set_xticks(tick_pos)
        ax_b.set_xticklabels(tick_txt, fontsize=10)
        ax_b.set_xlim(bands.kpath.distances[0], bands.kpath.distances[-1])
        ax_b.set_ylabel(r"$E - E_F$  /  eV")
        ax_b.set_title("Hcore bands (HPKOT)")
        ax_b.grid(axis="y", alpha=0.3, linestyle=":")

        ax_d.fill_betweenx(e_d, dos.dos, color="#ff7f0e", alpha=0.55,
                           linewidth=0)
        ax_d.plot(dos.dos, e_d, color="#ff7f0e", linewidth=1.4)
        ax_d.axhline(0.0, color="black", linewidth=0.8,
                     linestyle="--", alpha=0.7)
        ax_d.set_xlim(left=0)
        ax_d.set_xlabel("DOS")
        ax_d.set_title("Total DOS")
        ax_d.tick_params(labelleft=False)

        # PDOS — aggregate across all 8 Si atoms by l character.
        si_s = sum(
            (c for label, c in pdos.contributions.items()
             if label.endswith("-s")),
            start=np.zeros_like(e_p),
        )
        si_p = sum(
            (c for label, c in pdos.contributions.items()
             if label.endswith("-p")),
            start=np.zeros_like(e_p),
        )
        si_d = sum(
            (c for label, c in pdos.contributions.items()
             if label.endswith("-d")),
            start=np.zeros_like(e_p),
        )
        if si_s.any():
            ax_p.plot(si_s, e_p, color="#1f77b4", linewidth=1.4, label="Si-s")
        if si_p.any():
            ax_p.plot(si_p, e_p, color="#d62728", linewidth=1.4, label="Si-p")
        if si_d.any():
            ax_p.plot(si_d, e_p, color="#2ca02c", linewidth=1.0,
                      label="Si-d", alpha=0.7)
        ax_p.axhline(0.0, color="black", linewidth=0.8,
                     linestyle="--", alpha=0.7)
        ax_p.set_xlabel("PDOS")
        ax_p.set_xlim(left=0)
        ax_p.set_title("Si-s / Si-p / Si-d PDOS")
        ax_p.tick_params(labelleft=False)
        ax_p.legend(loc="upper right", fontsize=9, frameon=True)

        ax_b.set_ylim(max(e_b.min(), -25.0), min(e_b.max(), 25.0))
        fig.suptitle(
            f"Silicon (diamond) — Hcore / sto-3g — a = {A_ANG} Å",
            fontsize=12, y=1.0,
        )
        fig.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plog.info(f"wrote {PNG_OUT.name}")

    plog.banner("Done")


if __name__ == "__main__":
    main()
