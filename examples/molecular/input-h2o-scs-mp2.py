"""Water — SCS-MP2 (Grimme 2003) via RI-MP2.

The simplest possible scaled-spin-component MP2 calculation: a closed-
shell RHF reference followed by SCS-MP2 (c_os = 6/5, c_ss = 1/3) on
top, with density fitting on for the MP2 step (the recommended
production setup — see the SCS-MP2 section of
``docs/user_guide/mp2_and_double_hybrids.md``).

The convenience wrapper ``run_scs_mp2`` defaults to ``density_fit=True``
and auto-resolves the per-zeta RIfit aux from the orbital basis name,
so this script reads as a one-call dispatch on top of a vanilla RHF.

Run:
    .venv/bin/python examples/molecular/input-h2o-scs-mp2.py

Outputs (next to this script):
    input-h2o-scs-mp2.out      — banner + RHF SCF trace + SCS-MP2 block
    input-h2o-scs-mp2.molden   — orbitals at the RHF solution
    input-h2o-scs-mp2.system   — provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-scs-mp2"

# H2O at the same experimental geometry used in tutorial 15.
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# 1. Standard files via run_job (RHF reference + banner / MO table /
#    .molden / .system manifest).
vq.run_job(
    mol,
    basis="cc-pvdz",
    method="rhf",
    output=HERE / STEM,
)

# 2. SCS-MP2 (RI-MP2 path) on top of the RHF reference. The wrapper
#    auto-resolves cc-pvdz → cc-pvdz-ri as the RI aux basis.
basis = vq.BasisSet(mol, "cc-pvdz")

rhf_opts = vq.RHFOptions()
rhf_opts.conv_tol_energy = 1e-10
hf = vq.run_rhf(mol, basis, rhf_opts)

scs = vq.run_scs_mp2(mol, basis, hf)  # density_fit=True by default

# Optional: SOS-MP2 for comparison (same dispatch shape).
sos = vq.run_sos_mp2(mol, basis, hf)

# Optional: canonical MP2 for parity (slower; the SCS/SOS sum
# invariant relates these numbers).
mp2 = vq.run_mp2(mol, basis, hf)

# 3. Append the post-SCF block to the .out file run_job already wrote.
out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== post-SCF correlation (RI on RHF/cc-pVDZ) ===\n")
    fh.write(f"  E(RHF)                = {hf.energy:14.8f} Ha\n")
    fh.write(f"  E_os (unscaled, αβ)   = {scs.e_os:14.8f} Ha\n")
    fh.write(f"  E_ss (unscaled, αα+ββ)= {scs.e_ss:14.8f} Ha\n")
    fh.write(f"  E_corr(MP2)           = {mp2.e_correlation:14.8f} Ha"
             "    (c_os=1,    c_ss=1)\n")
    fh.write(f"  E_corr(SCS-MP2)       = {scs.e_correlation:14.8f} Ha"
             "    (c_os=6/5,  c_ss=1/3)\n")
    fh.write(f"  E_corr(SOS-MP2)       = {sos.e_correlation:14.8f} Ha"
             "    (c_os=1.3,  c_ss=0  )\n")
    fh.write(f"  E(SCS-MP2 total)      = {scs.e_total:14.8f} Ha\n")

print(f"  E(RHF)           = {hf.energy:14.8f} Ha")
print(f"  E(SCS-MP2 corr)  = {scs.e_correlation:14.8f} Ha")
print(f"  E(SCS-MP2 total) = {scs.e_total:14.8f} Ha")
