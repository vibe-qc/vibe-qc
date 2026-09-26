"""Water — B2PLYP double-hybrid total energy.

The simplest possible double-hybrid calculation:
``run_b2plyp(mol, basis)`` does the hybrid-DFT SCF step and the
scaled RI-MP2 correlation correction in one call, returning a
``DoubleHybridResult`` carrying both pieces and the combined total.

Recipe (Grimme, *J. Chem. Phys.* **124**, 034108 (2006)):

    E_B2PLYP = E_RKS[0.53·HF + 0.47·B88, 0.73·LYP] + 0.27 · E_MP2

The dispatcher defaults to density fitting on both steps — RI-J + RI-K
for the hybrid SCF (since α_HF = 0.53), RI-MP2 for the correction.
Aux bases are auto-resolved from the orbital basis name. See
``docs/user_guide/mp2_and_double_hybrids.md`` for the full surface
and ``docs/tutorial/double_hybrid_b2plyp.md`` for an annotated
walk-through.

Two equivalent routes are shown: ``run_job(method="b2plyp")`` for the
standard file set, and ``run_b2plyp`` when you want the result object.

Run:
    .venv/bin/python examples/molecular/input-h2o-b2plyp.py

Outputs (next to this script):
    input-h2o-b2plyp.out      — banner, construction line, energy components
    input-h2o-b2plyp.molden   — KS orbitals at the hybrid SCF solution
    input-h2o-b2plyp.system   — provenance manifest, incl. the SCF reference
    plus the usual .xyz / .bibtex / .references / population / .qvf siblings
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-b2plyp"

# H2O at the same experimental geometry used in tutorial 15 and the
# SCS-MP2 example next door.
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# 1. Standard files via run_job. "b2plyp" is a *method* here, not a
#    functional: run_job hands the whole double hybrid to the dispatcher,
#    so the .out carries the complete energy (SCF step + scaled MP2) and
#    the .molden the converged hybrid-SCF orbitals.
#
#    method="rks", functional="b2plyp" is a different request and raises:
#    it asks for the SCF half on its own, which is not B2PLYP.
vq.run_job(
    mol,
    basis="cc-pvdz",
    method="b2plyp",
    output=HERE / STEM,
)

# 2. The full B2PLYP dispatch — hybrid RKS SCF + RI-MP2 correction in
#    one call. RI on both sides by default; aux bases auto-resolved.
basis = vq.BasisSet(mol, "cc-pvdz")
result = vq.run_b2plyp(mol, basis)

# 3. What the dispatcher carries:
#       result.rks     — the full RKSResult of the hybrid SCF step
#       result.mp2     — the full MP2Result of the post-SCF correction
#       result.e_total — rks.energy + mp2.e_correlation
#       result.functional — "b2plyp"
#
#    The Functional resolver carries the recipe too:
fn = vq.Functional("b2plyp")

out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== B2PLYP double-hybrid decomposition ===\n")
    fh.write(f"  hf_exchange_fraction  = {fn.hf_exchange_fraction:.4f}  "
             "(SCF: 0.53·HF + 0.47·B88 + 0.73·LYP)\n")
    fh.write(f"  mp2_c_os              = {fn.mp2_c_os:.4f}  "
             "(post-SCF: scales the αβ pair)\n")
    fh.write(f"  mp2_c_ss              = {fn.mp2_c_ss:.4f}  "
             "(post-SCF: scales αα + ββ)\n")
    fh.write("\n")
    fh.write(f"  E(hybrid RKS step)        = {result.rks.energy:14.8f} Ha\n")
    fh.write(f"     E_xc (B88-X + LYP-C)   = {result.rks.e_xc:14.8f} Ha\n")
    fh.write(f"     E_HF_exchange (0.53)   = {result.rks.e_hf_exchange:14.8f} Ha\n")
    fh.write(f"  E_corr(MP2, on KS orbs)   = {result.mp2.e_correlation:14.8f} Ha"
             "    (= 0.27 × (E_os + E_ss))\n")
    fh.write(f"     E_os (unscaled)        = {result.mp2.e_os:14.8f} Ha\n")
    fh.write(f"     E_ss (unscaled)        = {result.mp2.e_ss:14.8f} Ha\n")
    fh.write(f"  E(B2PLYP total)           = {result.e_total:14.8f} Ha\n")

print(f"  E(hybrid RKS step)  = {result.rks.energy:14.8f} Ha")
print(f"  E(MP2 correction)   = {result.mp2.e_correlation:14.8f} Ha")
print(f"  E(B2PLYP total)     = {result.e_total:14.8f} Ha")
