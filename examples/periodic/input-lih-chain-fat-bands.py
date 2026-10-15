"""LiH chain: Hcore fat bands (Li-s, Li-p and H-s character along the chain).

System
    A uniform ...Li-H-Li-H... chain along x with 1.6 angstrom Li-H spacing
    (3.2 angstrom period, two atoms per cell) and 30 bohr of vacuum in y and
    z, built as a true `dim=1` periodic system.

Method and operator
    The operator is Hcore = T + V (kinetic plus bare nuclear attraction),
    assembled block by block from the real-space lattice sets of
    `vq.compute_kinetic_lattice` and `vq.compute_nuclear_lattice` and handed
    with the overlap lattice set to `vibeqc.bands.band_structure_projected`,
    which returns eigenvalues and Mulliken weights on the (atom, l) channels
    Li1-s, Li1-p and H2-s along the segment Gamma -> X. These are
    non-interacting band characters in the STO-3G basis. They are not
    self-consistent: there is no Hartree or exchange term, and the bare 1-D
    nuclear sum is conditionally convergent, so absolute Hcore energies move
    with `nuclear_cutoff_bohr` (the valence-band top shifts by 23 eV between
    25 and 50 bohr) while energies relative to the valence-band top change by
    less than 0.4 eV and the Mulliken weights by less than 0.002. Read the
    table as band shape and orbital character only.
    `band_structure_projected` lives in `vibeqc.bands`; the top-level
    `vibeqc` namespace does not re-export it. A self-consistent fat band
    needs the converged Fock lattice blocks of a route that exposes them; the
    Gaussian density-fitting route refuses a `dim=1` cell with a k mesh (no
    consistent Coulomb gauge for bare 1-D sums) and a chain in a 3-D vacuum
    box with a k mesh is a multi-hour job on the build used here, so it is
    not attempted.

Files (next to this script)
    None by default: the table goes to stdout. With `WRITE_QVF = True` the
    script also runs one Gamma-point RHF/STO-3G GDF job on the chain in a
    three-dimensional vacuum box and embeds the band object with
    `run_periodic_job(..., band_structure=bands)`. That writes
    input-lih-chain-fat-bands.qvf, beside the usual .out, .system, .cif,
    .xyz, .xsf, .molden, .population.txt / .population.json, .bibtex and
    .references files. The QVF `bands` section holds `bands/eigenvalues.bin`,
    `bands/projections.bin` and the channel labels in `bands/kpath.json`
    (21 x 6 x 3 projection values here). The QVF then carries Hcore bands
    beside an SCF that is not their operator; the two are different spectra.
    That branch converged in 5 iterations to E = -7.95697499 Ha and took
    452 s of wall time (287 s user CPU) on a shared host at load average
    near 120; it also emits a UserWarning that the optional density-grid
    artifact did not converge its image sum, which leaves the `bands`
    section unaffected.

Success criterion (vibe-qc 0.17.10 development build, Apple M-series)
    The Mulliken weights of every band sum to one over the three channels
    (obtained max |sum - 1| = 2e-15). The occupied valence band (band 1,
    two electrons in the Li 1s band below it) is H-s at Gamma (H-s 0.966,
    Li-s 0.034) and picks up Li-p character toward X (H-s 0.845, Li-p 0.155).
    The lowest empty band switches character along the path: Li-p (weight
    1.00) at Gamma but Li-s (weight 1.00) at X. The script asserts the
    partition to 1e-10, the H-s weight of the valence band at Gamma above
    0.9, and the Li-p to Li-s switch of the lowest empty band between Gamma
    and X. With the default `WRITE_QVF = False` it takes about 6 s of wall
    time, nearly all of it importing vibeqc.

Run:
    .venv/bin/python examples/periodic/input-lih-chain-fat-bands.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.bands import band_structure_projected

HERE = Path(__file__).parent
STEM = Path(__file__).stem

WRITE_QVF = False  # True: also run a Gamma SCF in a vacuum box and write the QVF

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HARTREE_TO_EV = 27.211386245988

D_LIH = 1.6 * BOHR_PER_ANGSTROM  # Li-H spacing, bohr
A_CELL = 2.0 * D_LIH             # chain period, bohr
VACUUM = 30.0                    # y and z, bohr

unit_cell = [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [D_LIH, 0.0, 0.0])]
system = vq.PeriodicSystem(
    dim=1,
    lattice=[[A_CELL, 0.0, 0.0], [0.0, VACUUM, 0.0], [0.0, 0.0, VACUUM]],
    unit_cell=unit_cell,
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
n_occ = system.n_electrons() // 2

# Real-space blocks of S, T and V; Hcore = T + V block by block. A fresh
# kinetic set is reused as the container for Hcore so no private constructor
# is needed.
opts = vq.LatticeSumOptions()
S_real = vq.compute_overlap_lattice(basis, system, opts)
T_real = vq.compute_kinetic_lattice(basis, system, opts)
V_real = vq.compute_nuclear_lattice(basis, system, opts)
H_real = vq.compute_kinetic_lattice(basis, system, opts)
for i in range(len(H_real.cells)):
    H_real.set_block(i, np.asarray(T_real.blocks[i]) + np.asarray(V_real.blocks[i]))

kpath = vq.kpath_from_segments(
    system, [((0.0, 0.0, 0.0), "G", (0.5, 0.0, 0.0), "X")], points_per_segment=20
)
bands = band_structure_projected(
    H_real, S_real, system, basis, kpath, n_electrons_per_cell=system.n_electrons()
)

partition = float(np.abs(bands.projections.sum(axis=2) - 1.0).max())
print(f"LiH chain, Hcore operator, {basis.nbasis} basis functions, "
      f"{n_occ} occupied bands, {kpath.n_points} k points on G -> X")
print(f"channels: {bands.channels}   projections shape {bands.projections.shape}   "
      f"max |sum_c W - 1| = {partition:.1e}")
print()
print(f"{'k/b1':>6}  {'band':>4}  {'E-VBM (eV)':>10}   "
      + "   ".join(f"{c:>6}" for c in bands.channels))
rows = {}
for ik in (0, 5, 10, 15, 20):
    for ib, tag in ((n_occ - 1, "VB"), (n_occ, "CB")):
        w = bands.projections[ik, ib]
        rows[(ik, tag)] = w
        print(f"{kpath.kpoints_frac[ik, 0]:6.3f}  {tag:>4}  "
              f"{(bands.energies[ik, ib] - bands.e_fermi) * HARTREE_TO_EV:10.3f}   "
              + "   ".join(f"{x:6.3f}" for x in w))

ch = {name: i for i, name in enumerate(bands.channels)}
print()
print("valence band at G: H-s weight "
      f"{rows[(0, 'VB')][ch['H2-s']]:.3f}, at X: {rows[(20, 'VB')][ch['H2-s']]:.3f}")
print("lowest empty band: dominant channel "
      f"{bands.channels[int(np.argmax(rows[(0, 'CB')]))]} at G, "
      f"{bands.channels[int(np.argmax(rows[(20, 'CB')]))]} at X")

assert partition < 1e-10, "Mulliken channel weights do not partition each band"
assert rows[(0, "VB")][ch["H2-s"]] > 0.9, "valence band at G is not mostly H-s"
assert np.argmax(rows[(0, "CB")]) == ch["Li1-p"], "lowest empty band at G is not Li-p"
assert np.argmax(rows[(20, "CB")]) == ch["Li1-s"], "lowest empty band at X is not Li-s"

if WRITE_QVF:
    box = vq.PeriodicSystem(
        dim=3,
        lattice=[[A_CELL, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        unit_cell=unit_cell,
    )
    res = vq.run_periodic_job(
        box,
        vq.BasisSet(box.unit_cell_molecule(), "sto-3g"),
        method="RHF",
        jk_method="gdf",
        band_structure=bands,
        output=str(HERE / STEM),
        output_qvf=True,
    )
    print(f"QVF written: {STEM}.qvf   (Gamma SCF E = {res.energy:.8f} Ha)")
print("OK")
