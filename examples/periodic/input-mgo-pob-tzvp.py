"""MgO rocksalt — pob-TZVP / RHF / EWALD_3D — CRYSTAL Tutorial port.

Direct port of the canonical CRYSTAL "Quick tour: Single-point
Energy" tutorial onto vibe-qc. MgO is the most-cited pob-TZVP
testbed (PVO-Bredow 2013) — sits squarely in the "MgO / NaCl /
CaO" ionic-crystal benchmark family that every solid-state code
publishes numbers for.

We use the **conventional 8-atom cubic cell** rather than the
2-atom FCC primitive cell. The primitive cell would be 4× cheaper
in atom count, but vibe-qc's EWALD_3D Poisson solver currently
requires orthorhombic cells (FCC primitive vectors are not
orthorhombic). The conventional cell trades a 4× cost increase
for the unconditionally-convergent Ewald path. Triclinic Ewald is
on the roadmap.

What this exercises:

  - Conventional cubic rocksalt cell, 8 atoms (4 Mg + 4 O)
  - Multi-k RHF with EWALD_3D on the v0.5.0 `vibeqc.KPoints`
    builder (Phase K1-K7); 4×4×4 mesh, IBZ-reduced via spglib
    (Fm-3m → 4 unique k-points)
  - v0.5.1 ProgressLogger threading: live SCF iteration trace
    mirrored to BOTH stdout AND a persistent .out file
  - v0.5.1 .system manifest sibling pinning the runtime
    environment (vibe-qc + git SHA, host CPU/RAM, OMP threads,
    library versions)
  - v0.5.2 perf log accumulator: per-phase timings + per-iteration
    SCF rows + memory snapshots into the .perf file

Outputs (next to this script in ``examples/periodic/``):
    output-mgo-pob-tzvp.out      — line-buffered SCF log
    output-mgo-pob-tzvp.system   — runtime manifest (TOML; v0.5.1)
    output-mgo-pob-tzvp.perf     — post-mortem perf report (v0.5.2)

Wall: ~15–40 min on a laptop, ~3–5 min on a 32-core box. MgO has
~3× the basis function count of LiH at pob-TZVP (148 vs 52 bf for
the conventional cell), so wall is ~3-5× longer than
``input-lih-pob-tzvp.py``.

The CRYSTAL Tutorial parity matrix tracks this script under the
"Quick tour: single-point energy" row. See
``docs/roadmap.md`` § "Tutorial parity (ORCA + CRYSTAL)".

Run live + persistent + perf:
    .venv/bin/python examples/periodic/input-mgo-pob-tzvp.py

Run under nohup with tail-able output:
    nohup .venv/bin/python examples/periodic/input-mgo-pob-tzvp.py \\
        > MgO-stdout.log 2>&1 &
    tail -f examples/periodic/output-mgo-pob-tzvp.out

Disable progress (still writes .out / .system / .perf):
    VIBEQC_LIVE_LOGGING=0 .venv/bin/python \\
        examples/periodic/input-mgo-pob-tzvp.py
"""

import os
import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.progress import ProgressLogger

HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "output-mgo-pob-tzvp"

HARTREE_TO_EV = 27.211386245988

# Lattice parameter (experimental, room temperature, MgO).
A_ANG = 4.213
A_BOHR = A_ANG / 0.529177210903

# Rocksalt: Mg at FCC sites, O at displaced FCC.
MG_FRAC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
O_FRAC = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]


t_start = time.perf_counter()
plog = ProgressLogger(log_path=OUT_STEM.with_suffix(".out"), verbose=True)

with vq.perf_log(OUT_STEM.with_suffix(".perf")):

    plog.banner("MgO rocksalt  /  RHF  /  pob-TZVP  /  EWALD_3D")

    # ---- 1. Build the MgO conventional cubic cell -------------------
    lat = A_BOHR * np.eye(3)
    unit_cell = []
    for fx, fy, fz in MG_FRAC:
        unit_cell.append(vq.Atom(12, [fx*A_BOHR, fy*A_BOHR, fz*A_BOHR]))
    for fx, fy, fz in O_FRAC:
        unit_cell.append(vq.Atom(8,  [fx*A_BOHR, fy*A_BOHR, fz*A_BOHR]))

    system = vq.PeriodicSystem(dim=3, lattice=lat, unit_cell=unit_cell)
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")
    vq.attach_symmetry(system, symprec=1e-4)
    plog.info(f"cell:       {A_ANG} Å cubic, {len(system.unit_cell)} atoms (4 Mg + 4 O)")
    plog.info(
        f"spacegroup: {system.symmetry.international_symbol} "
        f"(SG {system.symmetry.number}, "
        f"point group {system.symmetry.point_group})"
    )
    plog.info(f"basis:      pob-TZVP, {basis.nbasis} bf per cell")

    # ---- 2. SCF with EWALD_3D + IBZ-reduced 4×4×4 mesh --------------
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.conv_tol_energy = 1e-8

    kpts = vq.KPoints.monkhorst_pack(system, [4, 4, 4], symmetry=True)
    plog.info(
        f"k-mesh:     4×4×4 → {len(kpts)} IBZ points "
        f"(out of 64 in the full mesh)"
    )

    result = vq.run_rhf_periodic_scf(
        system, basis, kpts, opts, progress=plog,
    )
    plog.info(f"E/cell = {result.energy:.8f} Ha   ({result.n_iter} iters)")

    # ---- 3. HOMO-LUMO gap from per-k orbital energies ---------------
    n_occ = sum(int(at.Z) for at in system.unit_cell) // 2
    homo = max(float(eps[n_occ - 1]) for eps in result.mo_energies)
    lumo = min(float(eps[n_occ])     for eps in result.mo_energies)
    plog.info(
        f"HOMO max = {homo*HARTREE_TO_EV:.3f} eV   "
        f"LUMO min = {lumo*HARTREE_TO_EV:.3f} eV"
    )
    plog.info(
        f"Gap     = {(lumo - homo) * HARTREE_TO_EV:.3f} eV "
        f"(HF over-estimates ionic gaps; MgO experimental optical ~7.8 eV)"
    )

    # ---- 4. Runtime manifest ----------------------------------------
    with plog.stage("write_system_manifest"):
        vq.write_system_manifest(
            OUT_STEM.with_suffix(".system"),
            wall_seconds=time.perf_counter() - t_start,
            basename=OUT_STEM.name,
            record_hostname=os.environ.get(
                "VIBEQC_NO_HOSTNAME", ""
            ).strip().lower() not in ("1", "true", "yes", "on"),
        )

    plog.banner("Done")
    plog.info(f".out:    {OUT_STEM.with_suffix('.out').name}")
    plog.info(f".system: {OUT_STEM.with_suffix('.system').name}")
    plog.info(f".perf:   {OUT_STEM.with_suffix('.perf').name}  "
              "(written when this perf_log block exits)")
