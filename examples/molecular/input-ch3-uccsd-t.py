"""Methyl radical (CH3), open-shell UCCSD(T) on an automatic UHF reference.

For an open-shell molecule (multiplicity > 1), `run_job(method="ccsd(t)")`
runs UHF and then the spin-orbital UCCSD(T) kernel, with no extra flags;
the result is attached as `result.ccsd` with the same CCSDResult fields as
the closed-shell path. For a spin-pure restricted-open-shell reference
instead, add `ccsd_reference="rohf"`.

vibe-qc reproduces conventional ORCA 6.1 CCSD(T) for CH3/cc-pVDZ: the (T)
increment to under 3e-5 Ha and the total to within ~1 mHa (the cc-pVDZ
density-fitting error). The all-electron ORCA reference is
E(CCSD(T)) = -39.718197 Ha, E_(T) = -0.002624 Ha (cached in
tests/test_uccsd.py). run_job freezes chemical cores by default; pass
``frozen_core=False`` to match the all-electron reference exactly. This
script passes that escape explicitly so its comparison cannot inherit a
future default convention.

Run:
    .venv/bin/python examples/molecular/input-ch3-uccsd-t.py

Outputs (next to this script):
    input-ch3-uccsd-t.out      banner + UHF trace + UCCSD(T) block
    input-ch3-uccsd-t.molden   UHF orbitals
    input-ch3-uccsd-t.system   provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

# Planar D3h methyl radical, C-H about 1.079 angstrom (coordinates in bohr).
ch3 = vq.Molecule([
    vq.Atom(6, [ 0.000,  0.000, 0.0]),
    vq.Atom(1, [ 2.039,  0.000, 0.0]),
    vq.Atom(1, [-1.019,  1.766, 0.0]),
    vq.Atom(1, [-1.019, -1.766, 0.0]),
], multiplicity=2)

result = vq.run_job(
    ch3,
    basis="cc-pvdz",
    method="ccsd(t)",
    frozen_core=False,
    output=HERE / STEM,
)
cc = result.ccsd

out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== open-shell UCCSD(T) on a UHF reference ===\n")
    fh.write(f"  E(UHF)             = {cc.e_hf:16.8f} Ha\n")
    fh.write(f"  E_corr(UCCSD)      = {cc.e_ccsd_correlation:16.8f} Ha\n")
    fh.write(f"  E((T) increment)   = {cc.e_t:16.8f} Ha\n")
    fh.write(f"  E(UCCSD(T) total)  = {cc.e_ccsd_t:16.8f} Ha\n")
    fh.write(f"  converged={cc.converged} in {cc.n_iter} iters\n")
    fh.write("  (all-electron ORCA 6.1 ref: E(CCSD(T))=-39.718197, "
             "E_(T)=-0.002624; tests/test_uccsd.py)\n")

print(f"  E(UCCSD(T) total) = {cc.e_ccsd_t:16.8f} Ha")
