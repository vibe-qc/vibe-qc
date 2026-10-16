"""NaCl from a CIF and from a POSCAR: two input formats, one calculation.

System
    Rocksalt NaCl. `nacl.cif` next to this script is a hand-written CIF of the
    conventional cubic cell (a = 5.64 angstrom) with the two symmetry-unique
    sites and the four F-centring operations, so the parser expands it to
    eight atoms; `nacl.POSCAR` is the two-atom FCC primitive cell of the same
    crystal. Both files are minimal public-domain inputs authored for this
    example.

Method and basis
    `vq.read_cif(..., symmetrise=True, to_primitive=True)` and
    `vq.read_poscar(..., symmetrise=True, to_primitive=True)` parse each file
    into a `Crystal`, run it through `vq.symmetrise` (spglib standardisation
    and idealisation) and return a `SymmetriseResult` whose `.system` is a
    calculation-ready two-atom `PeriodicSystem` with a `.report` of the space
    group before and after. `vq.Crystal.from_cif` is the same parser as a
    constructor. Each primitive cell is then run as a Gamma-point RHF/STO-3G
    single point on the Gaussian density-fitting route (`jk_method="gdf"`).

Files (next to this script)
    input-nacl-from-cif-cif.* and input-nacl-from-cif-poscar.* (.out,
    .system, .cif, .xyz, .xsf, .molden, .population.txt / .population.json,
    .bibtex, .references), written by `run_periodic_job` for the two routes;
    the QVF archive is switched off.

Success criterion (vibe-qc 0.17.10 development build, Apple M-series)
    Both readers report Fm-3m (No. 225) before and after symmetrisation, the
    CIF is reduced from 8 to 2 atoms with zero idealisation displacement, and
    the two primitive cells are the same lattice (302.673 bohr^3) and
    positions to machine precision (obtained difference 0.0 bohr). The two
    single points are therefore the same calculation: obtained
    E(CIF) = E(POSCAR) = -614.695472 Ha in 8 iterations each, |dE| = 0.0 Ha.
    The script asserts |dE| < 1e-6 Ha. Cost: each Gamma GDF single point on
    this tight ionic cell is not a seconds-scale job; the two took 2957 s
    and 943 s of wall time on a shared host with a load average between 100
    and 270 (1309 s of user CPU for the whole script), and no idle-machine
    timing was taken.

Run:
    .venv/bin/python examples/periodic/input-nacl-from-cif.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem

CIF_PATH = HERE / "nacl.cif"
POSCAR_PATH = HERE / "nacl.POSCAR"

# 1. Parse without symmetrisation to see what each file literally contains.
raw_cif = vq.Crystal.from_cif(CIF_PATH)          # same parser as vq.read_cif
raw_poscar = vq.read_poscar(POSCAR_PATH)
print(f"{CIF_PATH.name}: {raw_cif.n_atoms} atoms in the conventional cell, "
      f"volume {abs(np.linalg.det(np.asarray(raw_cif.lattice))):.3f} bohr^3")
print(f"{POSCAR_PATH.name}: {raw_poscar.n_atoms} atoms in the primitive cell, "
      f"volume {abs(np.linalg.det(np.asarray(raw_poscar.lattice))):.3f} bohr^3")

# 2. Symmetrise both to the standardised primitive cell.
from_cif = vq.read_cif(CIF_PATH, symmetrise=True, to_primitive=True)
from_poscar = vq.read_poscar(POSCAR_PATH, symmetrise=True, to_primitive=True)
for label, sym in (("CIF", from_cif), ("POSCAR", from_poscar)):
    rep = sym.report
    print(f"{label:>6}: {rep.spacegroup_before.international_symbol} (No. {rep.spacegroup_before.number}) "
          f"-> {rep.spacegroup_after.international_symbol}, {rep.n_atoms_before} -> {rep.n_atoms_after} atoms, "
          f"max displacement {rep.max_displacement_bohr:.1e} bohr, "
          f"volume {rep.volume_after_bohr3:.3f} bohr^3")

sys_cif, sys_poscar = from_cif.system, from_poscar.system
lattice_diff = np.abs(np.asarray(sys_cif.lattice) - np.asarray(sys_poscar.lattice)).max()
pos_diff = max(
    float(np.abs(np.asarray(a.xyz) - np.asarray(b.xyz)).max())
    for a, b in zip(sys_cif.unit_cell, sys_poscar.unit_cell)
)
print(f"primitive cells agree: max |dlattice| = {lattice_diff:.1e} bohr, "
      f"max |dposition| = {pos_diff:.1e} bohr")

# 3. One Gamma RHF/STO-3G single point from each structure.
energies = {}
for label, system in (("cif", sys_cif), ("poscar", sys_poscar)):
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    t0 = time.perf_counter()
    res = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gdf",
        output=str(HERE / f"{STEM}-{label}"),
        output_qvf=False,
    )
    energies[label] = float(res.energy)
    print(f"{label:>6}: E = {res.energy:16.8f} Ha   {res.n_iter:2d} iterations   "
          f"converged={res.converged}   wall {time.perf_counter() - t0:6.1f} s")
    assert res.converged

de = abs(energies["cif"] - energies["poscar"])
print(f"|E(CIF) - E(POSCAR)| = {de:.2e} Ha")
assert de < 1e-6, "the two input formats did not give the same calculation"
print("OK")
