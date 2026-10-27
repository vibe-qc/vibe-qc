"""LiH rocksalt at Gamma: the three GDF fit backends side by side.

System
    LiH in the rocksalt structure, FCC primitive cell, conventional lattice
    constant 4.0 angstrom (two atoms, six STO-3G functions).

Method and basis
    RHF/STO-3G, Gamma point only, on the Gaussian density-fitting route
    (`jk_method="gdf"`) with the fit backend chosen explicitly through
    `gdf_method`: `"rsgdf"` (range-separated GDF, the default), `"compcell"`
    (compensated-cell GDF) and `"mdf"` (mixed density fitting, which adds a
    plane-wave residual mesh controlled by `mdf_ke_cutoff`, default 40 Ha).
    All three explicit values route the Gamma closed-shell RHF job through
    the same driver with Ewald exchange treatment, so the energies differ
    only by the fit. A backend that refuses this cell raises; the script
    catches the exception, prints its message and continues, so a refusal is
    reported rather than hidden. MDF is gated off for a compact dense-core
    class of cells (MgO is the documented member, source note G-GDF-001);
    LiH at STO-3G was not refused in the build used here, so all three
    backends ran, and the try/except is kept for the cells that are.

Files (next to this script)
    input-lih-mdf-vs-rsgdf-rsgdf.*, -compcell.* and -mdf.*, one family per
    backend (.out, .system, .molden, .population.txt / .population.json,
    .bibtex, .references); the QVF archive and structure sidecars are
    switched off.

Success criterion (vibe-qc 0.17.10 development build, Apple M-series)
    Obtained E(rsgdf) = -8.34749578 Ha, E(compcell) = -8.34749827 Ha and
    E(mdf) = -8.34748586 Ha, each converged in 7 iterations. The pairwise
    differences are -0.0025 mHa (compcell - rsgdf), +0.0099 mHa (mdf - rsgdf)
    and +0.0124 mHa (mdf - compcell), so the fit backends agree to about
    0.01 mHa on this cell. The script asserts that every backend that ran
    converged and that the largest pairwise difference is below 0.05 mHa.
    Cost: the three single points took 3253 s, 1355 s and 778 s of wall time
    (2403 s of user CPU in total) on a shared host whose load average fell
    from about 250 to about 100 during the run, so the walls cannot be
    compared between backends; no idle-machine timing was taken.

Run:
    .venv/bin/python examples/periodic/input-lih-mdf-vs-rsgdf.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
A_ANG = 4.0

a = A_ANG * BOHR_PER_ANGSTROM
lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
body_centre = 0.5 * (lattice[0] + lattice[1] + lattice[2])
system = vq.PeriodicSystem(
    dim=3,
    lattice=lattice,
    unit_cell=[vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, body_centre.tolist())],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

energies: dict[str, float] = {}
for gdf_method in ("rsgdf", "compcell", "mdf"):
    t0 = time.perf_counter()
    try:
        res = vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gdf",
            gdf_method=gdf_method,
            output=str(HERE / f"{STEM}-{gdf_method}"),
            output_qvf=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
        )
    except (NotImplementedError, ValueError, RuntimeError) as exc:
        print(f"{gdf_method:>9}: refused after {time.perf_counter() - t0:.1f} s")
        print(f"           {type(exc).__name__}: {exc}")
        continue
    wall = time.perf_counter() - t0
    energies[gdf_method] = float(res.energy)
    print(f"{gdf_method:>9}: E = {res.energy:16.8f} Ha   n_aux = {res.n_aux:4d}   "
          f"{res.n_iter:2d} iterations   converged={res.converged}   wall {wall:6.1f} s")
    assert res.converged, f"{gdf_method} did not converge"

print()
names = list(energies)
max_diff_mha = 0.0
for i, lhs in enumerate(names):
    for rhs in names[i + 1:]:
        diff_mha = 1000.0 * (energies[rhs] - energies[lhs])
        max_diff_mha = max(max_diff_mha, abs(diff_mha))
        print(f"E({rhs}) - E({lhs}) = {diff_mha:+.4f} mHa")
if len(names) < 2:
    raise SystemExit("fewer than two backends ran; nothing to compare")
print(f"largest pairwise difference = {max_diff_mha:.4f} mHa (reference build: 0.0124 mHa)")
assert max_diff_mha < 0.05, "GDF backends disagree beyond the documented tolerance"
print("OK")
