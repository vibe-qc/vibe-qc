"""α-Al2O3 corundum — RHF / sto-3g / Γ via the new GDF driver.

Real corundum structure: hexagonal R-3c (#167), a = 4.7589 Å,
c = 12.991 Å. Conventional cell, 12 Al + 18 O atoms (6 Al2O3 f.u.),
300 electrons, 198 AOs.

Plain Roothaan converges in 15 iters; DIIS in 10. We keep DIIS on
for the canonical example. PySCF mirror in
``../../periodic_pyscf/Al2O3-corundum-RHF-sto3g/`` reproduces the
same energy to sub-µHa.

Run::
    .venv/bin/python examples/periodic/Al2O3-corundum-RHF-sto3g/Al2O3-corundum-RHF-sto3g.py

Produces (sibling files, sharing the script's stem):
    {stem}.out      — text log
    {stem}.system   — runtime manifest
    {stem}.molden   — Γ-point MOs
"""
from pathlib import Path

from ase.spacegroup import crystal as ase_crystal

from vibeqc import (
    Atom, BasisSet, PeriodicSystem,
    run_periodic_job,
)

ANG2BOHR = 1.0 / 0.529177210903

# Lattice parameters: Springer Materials sd_1400479 (α-Al2O3, R-3c).
A_ANG = 4.7589
C_ANG = 12.991
AL_Z = 0.35216
O_X = 0.30624

OUTPUT_STEM = Path(__file__).resolve().with_suffix("")

atoms_ase = ase_crystal(
    ["Al", "O"],
    basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
    spacegroup=167,
    cellpar=[A_ANG, A_ANG, C_ANG, 90, 90, 120],
)

cell_bohr = atoms_ase.cell.array * ANG2BOHR
Z_BY_SYMBOL = {"Al": 13, "O": 8}
atoms = [
    Atom(Z_BY_SYMBOL[sym], list(pos * ANG2BOHR))
    for sym, pos in zip(atoms_ase.get_chemical_symbols(),
                        atoms_ase.get_positions())
]
system = PeriodicSystem(3, cell_bohr, atoms)
basis = BasisSet(system.unit_cell_molecule(), "sto-3g")

run_periodic_job(
    system, basis,
    method="RHF",
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=False,   # XSF density writer needs orthorhombic;
                           # corundum is hexagonal — flip on when
                           # general-lattice XSF density lands.
)
