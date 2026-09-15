"""Water — PWPB95 double-hybrid total energy.

PWPB95 (Goerigk & Grimme, *J. Chem. Theory Comput.* **7**, 291 (2011))
is vibe-qc's third double hybrid — and the first that is both a
**meta-GGA** and **spin-opposite-scaled** (SOS):

    E_PWPB95 = E_RKS[0.50·HF + 0.50·mPW(PW6), 0.731·B95] + 0.269·E_os

``run_pwpb95(mol, basis)`` does the hybrid meta-GGA SCF step and the
opposite-spin MP2 correction in one call, returning a
``DoubleHybridResult`` carrying both pieces and the combined total.

Two things make PWPB95 stand out from B2PLYP / DSD-PBEP86:

  * It is a **meta-GGA** — the B95 correlation component is
    τ-dependent, so the SCF step runs the τ-dependent Kohn-Sham path.
    The dispatcher handles that transparently.
  * It is **spin-opposite-scaled** — the same-spin MP2 coefficient is
    zero (c_ss = 0), so the correction is 0.269·E_os alone. The
    same-spin energy is still reported on ``result.mp2.e_ss``; it
    just does not enter the total.

The published method is **PWPB95-D3(BJ)**; this script prints both the
un-dispersed total and the D3(BJ)-corrected published total. See
``docs/user_guide/mp2_and_double_hybrids.md`` § PWPB95 for the full
surface.

Run:
    .venv/bin/python examples/molecular/input-h2o-pwpb95.py

Outputs (next to this script):
    input-h2o-pwpb95.out      — banner + per-step trace + PWPB95 block
    input-h2o-pwpb95.molden   — KS orbitals at the hybrid SCF solution
    input-h2o-pwpb95.system   — provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-pwpb95"

# H2O at the same experimental geometry as the B2PLYP example next door.
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# 1. Standard files via run_job. The "pwpb95" functional name drives
#    the SCF step's banner / MO table / .molden — these reflect the
#    hybrid meta-GGA SCF orbitals (the MP2 correction is post-SCF and
#    does not change the converged orbitals).
vq.run_job(
    mol,
    basis="cc-pvdz",
    method="rks",
    functional="pwpb95",
    output=HERE / STEM,
)

# 2. The full PWPB95 dispatch — hybrid meta-GGA RKS SCF + SOS-MP2
#    correction in one call. RI on both sides by default; aux bases
#    auto-resolved. dispersion="d3bj" folds in the published
#    PWPB95-D3(BJ) dispersion term.
basis = vq.BasisSet(mol, "cc-pvdz")
result = vq.run_pwpb95(mol, basis)
result_d3bj = vq.run_pwpb95(mol, basis, dispersion="d3bj")

# 3. The Functional resolver carries the recipe.
fn = vq.Functional("pwpb95")

out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== PWPB95 double-hybrid decomposition ===\n")
    fh.write(f"  kind                  = {fn.kind}  "
             "(meta-GGA — B95 correlation is τ-dependent)\n")
    fh.write(f"  hf_exchange_fraction  = {fn.hf_exchange_fraction:.4f}  "
             "(SCF: 0.50·HF + 0.50·mPW(PW6) + 0.731·B95)\n")
    fh.write(f"  mp2_c_os              = {fn.mp2_c_os:.4f}  "
             "(post-SCF: scales the αβ pair)\n")
    fh.write(f"  mp2_c_ss              = {fn.mp2_c_ss:.4f}  "
             "(SOS — same-spin MP2 dropped)\n")
    fh.write("\n")
    fh.write(f"  E(hybrid RKS step)        = {result.rks.energy:14.8f} Ha\n")
    fh.write(f"  E_corr(SOS-MP2, on KS orbs)= {result.mp2.e_correlation:14.8f} Ha"
             "    (= 0.269 × E_os)\n")
    fh.write(f"     E_os (unscaled)        = {result.mp2.e_os:14.8f} Ha\n")
    fh.write(f"     E_ss (unscaled, unused)= {result.mp2.e_ss:14.8f} Ha\n")
    fh.write(f"  E(PWPB95 total, no-D)     = {result.e_total:14.8f} Ha\n")
    fh.write(f"  E_dispersion (D3(BJ))     = "
             f"{result_d3bj.dispersion.energy:14.8f} Ha\n")
    fh.write(f"  E(PWPB95-D3(BJ) total)    = {result_d3bj.e_total:14.8f} Ha"
             "    (the published energy)\n")

print(f"  E(hybrid RKS step)      = {result.rks.energy:14.8f} Ha")
print(f"  E(SOS-MP2 correction)   = {result.mp2.e_correlation:14.8f} Ha")
print(f"  E(PWPB95 total, no-D)   = {result.e_total:14.8f} Ha")
print(f"  E(PWPB95-D3(BJ) total)  = {result_d3bj.e_total:14.8f} Ha")
