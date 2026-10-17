"""Rock-salt LiH / STO-3G: DLPNO-CCSD(T) on a crystal (preview).

``run_periodic_job(method="dlpno-ccsd(t)")`` runs the closed-shell GDF
reference on the multi-k engine with its classic RSGDF fit retained, folds a
Gamma-centred k mesh to the real Born-von Karman supercell, localises the
occupied orbitals with Pipek-Mezey and correlates the supercell with the
molecular DLPNO-CCSD(T) engine. Energies are reported per unit cell.

Measured with the vibe-qc 0.17.15 development build (main, 2026-10-11),
shipped defaults (NormalPNO, published frozen core, ``triples_mode="t1"``),
aux basis def2-SVP-JK:

    mesh       E_HF/cell        E_corr(CCSD)/cell  E(T)/cell   E_total/cell
    Gamma      -8.3351540806    -0.0068114824      -2.5e-06    -8.3419680624
    (2,2,2)    -7.9222475699    -0.0162333820      -2.8e-04    -7.9387617758

The Gamma reference of this compact primitive cell is 0.41 Ha per cell away
from the (2,2,2) value; never correlate a Gamma-only reference of a small
cell and call it the crystal. The route is preview: its accuracy at the
shipped thresholds on real crystals is not measured yet. See
docs/user_guide/periodic_local_correlation.md and
docs/tutorial/periodic_dlpno_ccsd_t.md.

Run from an empty scratch directory (vibe-qc writes its outputs into the
current directory); the (2,2,2) run takes about a minute on a laptop:

    python /path/to/examples/periodic/input-lih-dlpno-ccsd-t.py
"""

import numpy as np

import vibeqc as vq

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def lih_primitive(a_angstrom: float = 4.084) -> vq.PeriodicSystem:
    """Rock-salt LiH, primitive fcc cell, Li at the origin."""
    a = a_angstrom * BOHR_PER_ANGSTROM
    lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    centre = 0.5 * lattice.sum(axis=0)
    return vq.PeriodicSystem(
        dim=3,
        lattice=lattice,
        unit_cell=[vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, centre.tolist())],
    )


def main() -> None:
    system = lih_primitive()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    for mesh in [(1, 1, 1), (2, 2, 2)]:
        tag = "lih-dlpno-ccsd-t-" + "x".join(map(str, mesh))
        res = vq.run_periodic_job(
            system, basis, method="dlpno-ccsd(t)", kpoints=mesh,
            output=tag, progress=False,
        )
        c = res.correlation
        print(
            f"{mesh} cells={c.n_cells} source={c.source_kind} "
            f"E_HF/cell={c.e_hf_per_cell:.10f} "
            f"E_corr/cell={c.e_corr_per_cell:.10f} "
            f"E(T)/cell={c.e_t_per_cell:.2e} "
            f"E_total/cell={c.e_total_per_cell:.10f}"
        )


if __name__ == "__main__":
    main()
