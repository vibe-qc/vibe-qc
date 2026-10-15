"""Al2O3 cubic stuffed-fluorite model — RHF / STO-3G / Γ-only GDF.

The vibe-qc half of the RHF parity test against PySCF.pbc (Γ-only GDF
with ``exxdiv='ewald'``) on an Al2O3-stoichiometry crystal. See
``examples/debug/mgo_al2o3_parity.py`` for the side-by-side comparison
and the |ΔE| verdict.

Why a cubic *model* and not actual α-Al2O3 (corundum)
-----------------------------------------------------
A 5-atom cubic stuffed-fluorite Al2O3 model: 2 Al at body-diagonal
positions, 3 O at face-centred positions, lattice constant chosen to
give a reasonable Al-O distance ~2 Å. This is NOT a real Al2O3
polymorph, but both vibe-qc and PySCF.pbc solve it with the same
Hamiltonian, so the *parity* between the two codes is the test, not
the physical realism of the crystal. vibe-qc's periodic routes are
lattice-general; real hexagonal corundum belongs in the GDF parity
suite.

SCF settings
------------
This cell is hard for RHF at Γ although its converged state has a
0.29 Ha gap. Plain DIIS stalls in a higher state (-699.2255 Ha,
orbital gradient ~7e-4). A 0.5 Ha level shift over the first few
cycles steers it to the lower state (-699.29141 Ha), but the orbital
gradient then decays slowly:
5e-6 after 800 cycles, while the energy has settled to 1e-11 Ha
(vibe-qc 0.17.15). The example therefore uses the k-point driver,
which takes an options object, with that shift and a gradient
tolerance of 1e-4: it stops after about 160 cycles with the energy
within 1e-7 Ha of the 800-cycle value. ``run_periodic_job`` keeps its
1e-6 gradient tolerance and refuses this SCF as unconverged.

Run:
    .venv/bin/python examples/periodic/input-al2o3-cubic-rhf.py

Outputs (next to this script):
    input-al2o3-cubic-rhf.out      — banner, SCF trace, energies
    input-al2o3-cubic-rhf.perf     — per-stage wall-clock
    input-al2o3-cubic-rhf.system   — host / build / runtime manifest
"""

import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc import banner, format_scf_trace, perf_log, write_system_manifest

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-al2o3-cubic-rhf"
OUT  = HERE / STEM

ANG2BOHR = 1.0 / 0.529177210903

# Cubic Al2O3 stuffed-fluorite model — 2 Al + 3 O per cell
A_ANG = 5.05
A     = A_ANG * ANG2BOHR
al_frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
o_frac  = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]

unit_cell = (
    [vq.Atom(13, [fx*A, fy*A, fz*A]) for fx, fy, fz in al_frac]
    + [vq.Atom( 8, [fx*A, fy*A, fz*A]) for fx, fy, fz in o_frac]
)
sysp  = vq.PeriodicSystem(3, np.diag([A, A, A]), unit_cell)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

opts = vq.PeriodicRHFOptions()
opts.level_shift     = 0.5      # Ha, over the first few cycles only
opts.conv_tol_energy = 1e-7
opts.conv_tol_grad   = 1e-4     # see "SCF settings" above
opts.max_iter        = 300

t_start = time.perf_counter()
with perf_log(OUT.with_suffix(".perf")):
    result = vq.run_krhf_periodic_gdf(sysp, basis, (1, 1, 1), opts,
                                      progress=False)

with open(OUT.with_suffix(".out"), "w") as fh:
    fh.write(banner() + "\n\n")
    fh.write("Al2O3 cubic stuffed-fluorite model — RHF / STO-3G / Γ-only GDF\n")
    fh.write(f"  lattice constant      = {A_ANG:.3f} Å ({A:.4f} bohr)\n")
    fh.write(f"  atoms per cell        = 2 Al + 3 O = 5\n")
    fh.write(f"  basis                 = STO-3G ({basis.nbasis} fns / cell)\n")
    fh.write("\n")
    fh.write(format_scf_trace(result, include_banner=False) + "\n")

write_system_manifest(OUT.with_suffix(".system"),
                      wall_seconds=time.perf_counter() - t_start,
                      basename=OUT.name)

print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
      f"n_iter={result.n_iter}")
