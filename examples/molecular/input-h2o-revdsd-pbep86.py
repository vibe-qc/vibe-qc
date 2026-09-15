"""Water — revDSD-PBEP86 double-hybrid total energy.

``run_revdsd_pbep86(mol, basis)`` does the hybrid-DFT SCF step and the
asymmetrically-scaled RI-MP2 correlation correction in one call,
returning a ``DoubleHybridResult`` carrying both pieces and the total.

Recipe (Santra, Sylvetsky & Martin, *J. Phys. Chem. A* **123**, 5129
(2019)):

    SCF base : 0.69·HF + 0.31·PBE-X + 0.4210·P86-C
    MP2 corr : 0.5922·E_os + 0.0636·E_ss      (asymmetric: os ≠ ss)

revDSD-PBEP86 is the re-parametrized member whose coefficients were fit
jointly with **D4** damping, so D4 is its native dispersion. The basic
call below returns the SCF + MP2 total; add ``dispersion="d4"`` for the
published revDSD-PBEP86-D4 total (needs the optional ``dftd4`` package).
The older -D3BJ variant uses a different fit and is not this alias.

vibe-qc matches a PySCF reference to within 5e-5 Ha (validated in
``tests/test_revdsd_pbep86.py``). Molecular RKS reference only.

Run:
    .venv/bin/python examples/molecular/input-h2o-revdsd-pbep86.py

Outputs (next to this script):
    input-h2o-revdsd-pbep86.out      — banner + per-step trace + decomposition
    input-h2o-revdsd-pbep86.molden   — KS orbitals at the hybrid SCF solution
    input-h2o-revdsd-pbep86.system   — provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-revdsd-pbep86"

# Water, same geometry as the B2PLYP example next door (bohr).
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# 1. Standard files via run_job for the hybrid SCF step's artefacts
#    (the MP2 correction is post-SCF and does not change the orbitals).
vq.run_job(
    mol,
    basis="cc-pvdz",
    method="rks",
    functional="revdsd-pbep86",
    output=HERE / STEM,
)

# 2. The full double-hybrid dispatch: hybrid RKS SCF + scaled RI-MP2 in
#    one call. For the published revDSD-PBEP86-D4 total instead, use
#    vq.run_revdsd_pbep86(mol, basis, dispersion="d4")  (needs dftd4).
basis = vq.BasisSet(mol, "cc-pvdz")
result = vq.run_revdsd_pbep86(mol, basis)

fn = vq.Functional("revdsd-pbep86")

out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== revDSD-PBEP86 double-hybrid decomposition ===\n")
    fh.write(f"  hf_exchange_fraction  = {fn.hf_exchange_fraction:.4f}  "
             "(SCF: 0.69·HF + 0.31·PBE-X + 0.4210·P86-C)\n")
    fh.write(f"  mp2_c_os              = {fn.mp2_c_os:.4f}  "
             "(scales the opposite-spin pair)\n")
    fh.write(f"  mp2_c_ss              = {fn.mp2_c_ss:.4f}  "
             "(scales the same-spin pair)\n")
    fh.write("\n")
    fh.write(f"  E(hybrid RKS step)        = {result.rks.energy:14.8f} Ha\n")
    fh.write(f"  E_corr(MP2, asymmetric)   = {result.mp2.e_correlation:14.8f} Ha\n")
    fh.write(f"  E(revDSD-PBEP86 total)    = {result.e_total:14.8f} Ha\n")
    fh.write("  (add dispersion='d4' for the published revDSD-PBEP86-D4 total)\n")

print(f"  E(hybrid RKS step)     = {result.rks.energy:14.8f} Ha")
print(f"  E(MP2 correction)      = {result.mp2.e_correlation:14.8f} Ha")
print(f"  E(revDSD-PBEP86 total) = {result.e_total:14.8f} Ha")
