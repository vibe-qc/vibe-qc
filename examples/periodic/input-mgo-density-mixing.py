"""MgO rocksalt: Fock DIIS versus Anderson, Broyden and Kerker density mixing.

System
    MgO in the rocksalt structure, FCC primitive cell (Mg at the origin, O at
    the cell-body centre), conventional lattice constant 4.21 angstrom.

Method and basis
    RKS with the LDA functional in STO-3G on the closed-shell multi-k
    Gaussian density-fitting route (`jk_method="gdf"`) with an explicit
    1x1x2 Gamma-centred mesh and `rsgdf_ke_cutoff=60.0` (the mesh and
    cutoff of the density-mixer regression test in
    tests/test_periodic_gdf_density_mixer.py, which runs LiH; a coarse mesh
    chosen for speed, not for convergence). The same SCF is run four times with the runner's
    density-mixing controls: `density_mixer=None` (the default Fock-matrix
    DIIS), `"anderson"`, `"broyden"`, and `"anderson"` with
    `density_mixer_kerker=True`, which adds Kerker preconditioning of the
    density residual. `run_periodic_job` accepts these keywords only on the
    closed-shell GDF (or multi-k RIJCOSX) route with an explicit `kpoints=`;
    every other route raises `NotImplementedError` instead of silently
    changing its update scheme.

Files (next to this script)
    input-mgo-density-mixing-diis.out / .system, -anderson.out / .system,
    -broyden.out / .system and -anderson-kerker.out / .system (QVF and
    structure sidecars are switched off).

Success criterion (vibe-qc 0.17.10 development build, Apple M-series)
    All four runs converge to the same fixed point. Obtained: DIIS
    E = -270.2870240638 Ha in 13 iterations, Anderson -270.2870240639 Ha in
    13, Broyden -270.2870240423 Ha in 13, Anderson with Kerker
    preconditioning -270.2870240372 Ha in 37; the largest spread is 2.7e-8 Ha.
    The traces differ (DIIS falls monotonically, the density mixers
    overshoot by up to 1e-2 Ha on the way in), but on this small
    insulating cell Fock DIIS is already as fast as the density mixers, and
    Kerker preconditioning, aimed at long-wavelength charge sloshing, only
    slows it. The example shows the switches and their equivalence; it does
    not exhibit a cell where DIIS fails. The script asserts convergence and
    a spread below 1e-6 Ha. The Kerker run emits a
    GAPWExperimentalWarning once: the preconditioner builds a plane-wave grid
    (`kerker_cutoff_ha`, default 120 Ha, a 60x60x60 grid here) from the
    experimental GAPW infrastructure. Cost: each SCF is a multi-minute job;
    the DIIS, Anderson, Broyden and Kerker runs took 5625 s, 2400 s, 1884 s
    and 2995 s of wall time (11848 s of user CPU for the whole script) on a
    shared host at load average 100 to 270, so the walls are not comparable
    between mixers; no idle-machine timing was taken.

Run:
    .venv/bin/python examples/periodic/input-mgo-density-mixing.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
A_ANG = 4.21

a = A_ANG * BOHR_PER_ANGSTROM
lattice = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
body_centre = 0.5 * (lattice[0] + lattice[1] + lattice[2])
system = vq.PeriodicSystem(
    dim=3,
    lattice=lattice,
    unit_cell=[vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, body_centre.tolist())],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

MIXERS = (
    ("diis", {}),
    ("anderson", dict(density_mixer="anderson")),
    ("broyden", dict(density_mixer="broyden")),
    ("anderson-kerker", dict(density_mixer="anderson", density_mixer_kerker=True)),
)

results = {}
print(f"{'mixer':>16}  {'E (Ha)':>16}  iters  conv  wall")
for label, mixer_kwargs in MIXERS:
    t0 = time.perf_counter()
    res = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gdf",
        kpoints=(1, 1, 2),
        rsgdf_ke_cutoff=60.0,
        output=str(HERE / f"{STEM}-{label}"),
        output_qvf=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        **mixer_kwargs,
    )
    wall = time.perf_counter() - t0
    results[label] = res
    print(f"{label:>16}  {res.energy:16.8f}  {res.n_iter:5d}  {str(res.converged):>4}  {wall:5.1f} s")

energies = np.array([r.energy for r in results.values()])
spread = float(energies.max() - energies.min())
print()
print(f"energy spread across mixers = {spread:.2e} Ha (reference build: 2.7e-8)")
assert all(r.converged for r in results.values()), "a mixer did not converge"
assert spread < 1e-6, "mixers reached different fixed points"
print("OK")
