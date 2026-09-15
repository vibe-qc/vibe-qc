"""RHF with explicit conventional (in-core ERI tensor) Fock build on H2O / 6-31G*.

CONVENTIONAL mode materialises the full 4-index ERI tensor once and reuses
it across SCF iterations. On this 18-basis-function example the tensor is
18⁴ x 8 bytes = 0.801 MiB, so retaining it is inexpensive.

The same allocation is 6.223 GiB at 170 BF and 179.546 GiB at 394 BF.
Use AUTO or DIRECT once the complete memory estimate no longer fits the
job. See input-h2o-rhf-direct.py for the DIRECT counterpart.

Run:
    .venv/bin/python input-h2o-rhf-conventional.py
"""

from __future__ import annotations

from pathlib import Path

from vibeqc import Molecule, RHFOptions, SCFMode, run_rhf

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

opts = RHFOptions()
opts.scf_mode = SCFMode.CONVENTIONAL  # explicit: in-core 4-index ERI tensor
opts.conv_tol_energy = 1e-8

result = run_rhf(mol, "6-31g*", opts)

print(f"Energy: {result.energy:.10f} Ha")
print(f"Converged in {result.n_iter} iterations")
