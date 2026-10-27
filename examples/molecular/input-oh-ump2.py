"""OH radical at 6-31G*: one UHF + UMP2 job with standard output files.

Copy this script to a run directory outside the checkout, activate the
installed engine environment, and run ``python input-oh-ump2.py``.
Outputs use the ``output-oh-ump2`` prefix in the current working directory.

``run_job(method="mp2")`` selects UMP2 for this doublet and writes the
reference and correlation energies through the normal output system. The
Molden file contains UHF orbitals. All-electron correlation is explicit;
no second UHF run or manual append to the output is needed.
"""

import vibeqc as vq

# OH radical: open-shell doublet at an illustrative fixed geometry in bohr.
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.85]),   # approximately 0.98 angstrom
    ],
    charge=0,
    multiplicity=2,
)

job = vq.run_job(
    mol,
    basis="6-31g*",
    method="mp2",
    mp2_reference="uhf",
    frozen_core=0,
    output="output-oh-ump2",
    write_molden_file=True,
)
if not job.converged:
    raise RuntimeError("OH UHF reference did not converge; inspect output-oh-ump2.out")

# energy is the reference energy; energy_total includes the MP2 correction.
print(f"  E(UHF)              = {job.energy:14.8f} Ha")
print(f"  E(UMP2 correlation) = {job.mp2.e_correlation:14.8f} Ha")
print(f"  E(UMP2 total)       = {job.energy_total:14.8f} Ha")
print(f"  <S^2>(UHF)          = {job.s_squared:14.6f}")
print(f"  ideal <S^2>         = {job.s_squared_ideal:14.6f}")
