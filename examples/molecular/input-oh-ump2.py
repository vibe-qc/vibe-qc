"""OH radical — UHF + UMP2 post-SCF correlation energy.

Demonstrates the open-shell post-Hartree-Fock workflow:

  1. Run a UHF reference (using ``run_job`` for the standard files
     plus banner/MO table/.molden export).
  2. Re-run the underlying ``run_uhf`` to get the result object that
     ``run_ump2`` consumes (UMP2 needs the orbital + integral state
     that the high-level driver assembles internally).
  3. Print the correlation energy and the spin-contamination
     diagnostic — UMP2 inherits the UHF reference's <S²>.

Run:
    .venv/bin/python examples/molecular/input-oh-ump2.py

Outputs (next to this script):
    input-oh-ump2.out       — banner, UHF SCF trace, MO table
    input-oh-ump2.molden    — orbitals at the UHF solution
    input-oh-ump2.system    — host / build / runtime manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-oh-ump2"

# OH radical — open-shell doublet, equilibrium geometry
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.85]),   # ~0.98 Å in bohr
    ],
    charge=0,
    multiplicity=2,
)

# 1. Standard files via run_job (UHF reference)
vq.run_job(
    mol,
    basis="6-31g*",
    method="uhf",
    output=HERE / STEM,
)

# 2. UMP2 on top of the UHF reference. run_uhf is needed for the
#    intermediate-result handoff that run_ump2 consumes.
basis = vq.BasisSet(mol, "6-31g*")
uhf = vq.run_uhf(mol, basis)
ump2 = vq.run_ump2(mol, basis, uhf)

# 3. Report into the same .out the run_job above already wrote.
out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== UMP2 post-HF correlation ===\n")
    fh.write(f"  E(UHF)              = {uhf.energy:14.8f} Ha\n")
    fh.write(f"  E(UMP2 correlation) = {ump2.e_correlation:14.8f} Ha\n")
    fh.write(f"  E(UMP2 total)       = {ump2.e_total:14.8f} Ha\n")
    fh.write(f"  <S²>(UHF)           = {uhf.s_squared:14.6f}\n")
    fh.write(f"  ideal <S²>          = {0.5 * (mol.multiplicity - 1) * (mol.multiplicity + 1):14.6f}\n")

print(f"  E(UHF)              = {uhf.energy:14.8f} Ha")
print(f"  E(UMP2 correlation) = {ump2.e_correlation:14.8f} Ha")
print(f"  E(UMP2 total)       = {ump2.e_total:14.8f} Ha")
