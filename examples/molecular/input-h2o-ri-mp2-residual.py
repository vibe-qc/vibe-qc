"""Water — RI-MP2 fit-residual diagnostic across two aux bases.

Demonstrates ``MP2Options.report_ri_residual``: an opt-in flag that
also computes the canonical (ia|jb) tensor and reports the per-bucket
Coulomb-metric Dunlap fit residual on the MP2Result. See
``docs/user_guide/mp2_and_double_hybrids.md`` § "Verifying the RI fit"
for the math and the typical bucket-error signature.

The residual is **structurally biased** — E_os over-estimated and
E_ss under-estimated — and partially cancels in the unscaled sum but
survives SCS / SOS scaling. Running the same diagnostic on two aux
choices shows the residual shrinking monotonically with aux size, the
expected signature of a genuine fit residual (not a numerical bug).

Run:
    .venv/bin/python examples/molecular/input-h2o-ri-mp2-residual.py

Cost note: the diagnostic doubles the runtime of an RI-MP2 call (it
forces the O(N^5) canonical AO→MO transform alongside the O(n_aux ...)
RI build). Intended for verification — keep it off in production runs.
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

mol = vq.Molecule(
    [
        vq.Atom(8, [0.0,  0.00,  0.00]),
        vq.Atom(1, [0.0,  1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)
basis = vq.BasisSet(mol, "cc-pvtz")

rhf_opts = vq.RHFOptions()
rhf_opts.conv_tol_energy = 1.0e-10
rhf_opts.conv_tol_grad = 1.0e-08
hf = vq.run_rhf(mol, basis, rhf_opts)
assert hf.converged

print(f"\nH2O / cc-pvtz   E(RHF) = {hf.energy:.10f} Ha\n")
print(f"{'aux basis':<22s} {'n_aux':>6s}  {'E_os':>13s}  "
      f"{'E_ss':>13s}  {'ΔE_os':>11s}  {'ΔE_ss':>11s}  {'ΔE_SCS':>11s}")
print("-" * 100)

c_os, c_ss = 6.0 / 5.0, 1.0 / 3.0   # Grimme SCS-MP2

for aux_name in ("cc-pvtz-ri", "cc-pvqz-ri"):
    aux = vq.BasisSet(mol, aux_name)

    opts = vq.MP2Options()
    opts.density_fit = True
    opts.aux_basis = aux_name
    opts.c_os = c_os
    opts.c_ss = c_ss
    opts.report_ri_residual = True       # the opt-in flag

    mp2 = vq.run_mp2(mol, basis, hf, opts)
    assert mp2.ri_residual_reported

    # Reconstruct canonical SCS from the buckets (RI bucket − residual = canonical).
    e_os_can = mp2.e_os - mp2.e_os_ri_residual
    e_ss_can = mp2.e_ss - mp2.e_ss_ri_residual
    scs_can = c_os * e_os_can + c_ss * e_ss_can
    scs_ri  = c_os * mp2.e_os + c_ss * mp2.e_ss
    delta_scs = scs_ri - scs_can

    print(f"{aux_name:<22s} {aux.nbasis:>6d}  "
          f"{mp2.e_os:>13.8f}  {mp2.e_ss:>13.8f}  "
          f"{mp2.e_os_ri_residual:>+11.2e}  "
          f"{mp2.e_ss_ri_residual:>+11.2e}  "
          f"{delta_scs:>+11.2e}")

print("""
Reading the table:
  * ΔE_os > 0 and ΔE_ss < 0 on both rows — the structural fingerprint
    of the Coulomb-metric Dunlap fit.
  * The residual shrinks ~2x going cc-pvtz-ri → cc-pvqz-ri, monotonic
    with n_aux as expected for a genuine fit residual.
  * Δ_SCS stays positive and similar in magnitude to ΔE_os because
    SCS weights the OS bucket 3.6x more than SS (1.2 / 0.333).
""")
