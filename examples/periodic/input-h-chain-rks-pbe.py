"""1D H2 chain — periodic RKS-PBE with multi-k IBZ-reduced mesh.

The DFT counterpart to ``input-h-chain-uniform.py``: same H2 unit
cell, same lattice, same k-mesh; switches the SCF from RHF to RKS
with the PBE GGA functional.

For a uniform 1D H-chain the PBE band structure is *qualitatively*
similar to HF (gapped molecular crystal — H2 dimers separated by a
larger inter-pair distance) but the band gap is smaller; PBE
under-binds the σ→σ* gap by 1-2 eV vs HF, which is the textbook
"DFT under-estimates band gaps" story.

Run:
    .venv/bin/python examples/periodic/input-h-chain-rks-pbe.py

Outputs (next to this script):
    input-h-chain-rks-pbe.out      — banner, SCF trace, energies
    input-h-chain-rks-pbe.system   — host / build / runtime manifest
"""

from pathlib import Path

import numpy as np

from vibeqc import (
    Atom,
    BasisSet,
    KPoints,
    PeriodicKSOptions,
    PeriodicSystem,
    perf_log,
    run_rks_periodic_scf,
    write_system_manifest,
)

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h-chain-rks-pbe"
OUT  = HERE / STEM

# H2 molecular crystal: 2 atoms in a 5-bohr unit cell along x,
# 30 bohr vacuum padding along y / z. Tutorial-17-style geometry
# (R_H-H = 1.4 bohr = H2 equilibrium; lattice spacing 5 bohr).
A = 5.0
PAD = 30.0

unit_cell = [
    Atom(1, [0.0, 0.0, 0.0]),
    Atom(1, [1.4, 0.0, 0.0]),
]
sysp  = PeriodicSystem(dim=1, lattice=np.diag([A, PAD, PAD]),
                       unit_cell=unit_cell)
basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
kpts  = KPoints.monkhorst_pack(sysp, [8, 1, 1])

opts = PeriodicKSOptions()
opts.functional       = "pbe"
opts.lattice_opts.cutoff_bohr         = 15.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.conv_tol_energy  = 1e-7
opts.max_iter         = 60

with perf_log(OUT.with_suffix(".perf")):
    result = run_rks_periodic_scf(sysp, basis, kpts, opts, progress=False)

# Headline numbers go into the .out we own
with open(OUT.with_suffix(".out"), "w") as fh:
    fh.write(f"H2 chain (1D, periodic) — RKS-PBE / pob-TZVP\n")
    fh.write(f"  lattice param a       = {A:.3f} bohr\n")
    fh.write(f"  H-H bond              = 1.4 bohr (H2 equilibrium)\n")
    fh.write(f"  k-mesh                = 8x1x1 Monkhorst-Pack\n")
    fh.write(f"  basis                 = pob-TZVP ({basis.nbasis} fns / cell)\n")
    fh.write("\n")
    fh.write(f"  E (per cell)          = {result.energy:.8f} Ha\n")
    fh.write(f"  converged             = {result.converged}\n")
    fh.write(f"  SCF iterations        = {result.n_iter}\n")

write_system_manifest(OUT.with_suffix(".system"))

print(f"  E (per cell) = {result.energy:.8f} Ha   "
      f"converged={result.converged}   n_iter={result.n_iter}")
