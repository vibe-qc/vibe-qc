"""Water, canonical CCSD(T), the gold-standard correlation reference.

`run_job(method="ccsd(t)")` runs RHF and then the spin-adapted DF-CCSD(T)
kernel, attaching the post-SCF result as `result.ccsd` (a CCSDResult with
e_hf / e_ccsd_correlation / e_ccsd / e_t / e_ccsd_t).

vibe-qc reproduces conventional ORCA 6.1.1 CCSD(T) for this system: the
(T) increment to under 3e-5 Ha (density-fitting insensitive) and the
total to within ~1 mHa (the cc-pVDZ density-fitting error). The
all-electron ORCA reference for H2O/cc-pVDZ is E(CCSD(T)) = -76.240621 Ha,
E_(T) = -0.003281 Ha (cached in tests/test_ccsd.py). This comparison script
passes ``frozen_core=False`` explicitly, so the calculated and quoted values
use the same all-electron convention even though the post-#140 public default
is the published chemical-core count.

Run:
    .venv/bin/python examples/molecular/input-h2o-ccsd-t.py

Outputs (next to this script):
    input-h2o-ccsd-t.out      banner + RHF trace + CCSD(T) block
    input-h2o-ccsd-t.molden   RHF orbitals
    input-h2o-ccsd-t.system   provenance manifest
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])

result = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="ccsd(t)",
    frozen_core=False,
    output=HERE / STEM,
)
cc = result.ccsd

out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== canonical CCSD(T) decomposition ===\n")
    fh.write(f"  E(RHF)             = {cc.e_hf:16.8f} Ha\n")
    fh.write(f"  E_corr(CCSD)       = {cc.e_ccsd_correlation:16.8f} Ha\n")
    fh.write(f"  E(CCSD)            = {cc.e_ccsd:16.8f} Ha\n")
    fh.write(f"  E((T) increment)   = {cc.e_t:16.8f} Ha\n")
    fh.write(f"  E(CCSD(T) total)   = {cc.e_ccsd_t:16.8f} Ha\n")
    fh.write(f"  converged={cc.converged} in {cc.n_iter} iters; "
             f"|T1|={cc.t1_norm:.4f}, |T2|={cc.t2_norm:.4f}\n")
    fh.write("  (all-electron ORCA 6.1.1 ref: E(CCSD(T))=-76.240621, "
             "E_(T)=-0.003281; tests/test_ccsd.py)\n")

print(f"  E(CCSD(T) total) = {cc.e_ccsd_t:16.8f} Ha")
print(f"  E((T) increment) = {cc.e_t:16.8f} Ha")
