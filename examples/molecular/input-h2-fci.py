"""H2 — exact Full CI vs Hartree-Fock.

The smallest interesting wavefunction-method calculation: Full CI
(``method="fci"``) diagonalises the Hamiltonian in the complete
determinant space of the orbital basis, so its energy is *exact*
within that basis. The gap to the Hartree-Fock energy is the
basis-set correlation energy.

``run_job(method="fci")`` builds the HF orbital basis internally,
transforms the Hamiltonian to the MO basis, generates every
determinant, and diagonalises. For H2 / 6-31G that is a handful of
determinants — instantaneous. Full CI is only tractable for small
(active) spaces; for larger systems use the approximate solvers
(``selected_ci`` / ``dmrg``) — see
``docs/user_guide/non_hf_solvers.md`` and
``docs/tutorial/non_hf_solvers.md``.

Run:
    .venv/bin/python examples/molecular/input-h2-fci.py

Outputs (next to this script):
    input-h2-fci.out      — banner + Full CI solver trace
    input-h2-fci.system   — provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2-fci"

# H2 near its equilibrium bond length (1.4 bohr ≈ 0.74 Angstrom).
mol = vq.Molecule(
    [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ]
)
basis = vq.BasisSet(mol, "6-31g")

# 1. Hartree-Fock reference energy.
e_hf = vq.run_rhf(mol, basis).energy

# 2. Full CI via run_job — writes input-h2-fci.out / .system, returns
#    the SolverResult.
fci = vq.run_job(
    mol,
    basis="6-31g",
    method="fci",
    output=HERE / STEM,
)

e_corr = fci.energy - e_hf

# 3. Append the HF-vs-FCI decomposition to the .out file.
out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== Hartree-Fock vs Full CI ===\n")
    fh.write(f"  E(RHF/6-31G)            = {e_hf:16.10f} Ha\n")
    fh.write(f"  E(FCI/6-31G)            = {fci.energy:16.10f} Ha\n")
    fh.write(f"  E_corr (basis-set)      = {e_corr:16.10f} Ha\n")
    fh.write(f"  Determinants in the CI  = {len(fci.ci_labels)}\n")

print(f"  E(RHF/6-31G)  = {e_hf:16.10f} Ha")
print(f"  E(FCI/6-31G)  = {fci.energy:16.10f} Ha")
print(f"  E_corr        = {e_corr:16.10f} Ha"
      f"   ({len(fci.ci_labels)} determinants)")
