"""Water — DLPNO-MP2 (Pinski 2015): local MP2 with PNO compression.

DLPNO-MP2 localises the occupied orbitals (Foster-Boys), gives every
occupied pair a compact pair-natural-orbital virtual space, and solves
the coupled local-MP2 equations. This example measures and reports the
current default's recovery rather than advertising an inherited historical
percentage. Tightening (or zeroing) the thresholds converges it to the
canonical answer — the zero-threshold limit reproduces RI-MP2 to better than
1 µHa for the same active occupied space and is asserted in
``tests/test_dlpno_mp2.py``.

This script runs the standard job once (``method="dlpno-mp2"``), then
sweeps TCutPNO to show the threshold convergence against canonical
RI-MP2 — the table every DLPNO user should see once. See
``docs/user_guide/dlpno_mp2.md`` and ``docs/tutorial/dlpno_local_correlation.md``.

Run:
    .venv/bin/python examples/molecular/input-h2o-dlpno-mp2.py

Outputs (next to this script):
    input-h2o-dlpno-mp2.out     — banner + RHF trace + DLPNO-MP2 block
                                  + appended threshold-sweep table
    input-h2o-dlpno-mp2.molden  — orbitals at the RHF solution
    input-h2o-dlpno-mp2.system  — provenance manifest
"""

from pathlib import Path

import vibeqc as vq
from vibeqc.dlpno.mp2 import DLPNOMP2Options, run_dlpno_mp2

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-dlpno-mp2"

# H2O at the same experimental geometry used across the tutorials.
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
)

# 1. The one-call production route: RHF + DLPNO-MP2 with the
#    auto-resolved RI fitting basis and default thresholds.
result = vq.run_job(
    mol,
    basis="cc-pvdz",
    method="dlpno-mp2",
    output=HERE / STEM,
)
dlpno = result.dlpno_mp2

# 2. Threshold sweep: rerun the DLPNO step at different TCutPNO against
#    the canonical RI-MP2 anchor (same fitting basis).
basis = vq.BasisSet(mol, "cc-pvdz")
rhf_opts = vq.RHFOptions()
rhf_opts.conv_tol_energy = 1e-10
hf = vq.run_rhf(mol, basis, rhf_opts)

n_frozen = vq.chemical_core_orbital_count(mol)
mp2_opts = vq.MP2Options()
mp2_opts.n_frozen_core = n_frozen
mp2_opts.density_fit = True
mp2_opts.aux_basis = "cc-pvdz-ri"
canonical = vq.run_mp2(mol, basis, hf, mp2_opts)

from vibeqc.density_fitting import DensityFitting

df = DensityFitting(basis, vq.BasisSet(mol, "cc-pvdz-ri"), aux_basis_name="cc-pvdz-ri")

rows = []
for label, opts in (
    (
        "1e-07",
        DLPNOMP2Options(
            n_frozen=n_frozen,
            tcut_pairs=1e-4,
            tcut_pairs_weak=1e-4,
            tcut_pno=1e-7,
            tcut_pno_weak=1e-6,
            tcut_mkn=1e-3,
        ),
    ),
    (
        "1e-08",
        DLPNOMP2Options(
            n_frozen=n_frozen,
            tcut_pairs=1e-4,
            tcut_pairs_weak=1e-4,
            tcut_pno=1e-8,
            tcut_pno_weak=1e-7,
            tcut_mkn=1e-3,
        ),
    ),
    (
        "1e-09",
        DLPNOMP2Options(
            n_frozen=n_frozen,
            tcut_pairs=1e-4,
            tcut_pairs_weak=1e-4,
            tcut_pno=1e-9,
            tcut_pno_weak=1e-8,
            tcut_mkn=1e-3,
        ),
    ),
    # The exactness limit: no PNO truncation AND full domains — this row
    # must (and does) reproduce canonical RI-MP2 to sub-µHa.
    (
        "0*",
        DLPNOMP2Options(
            n_frozen=n_frozen,
            tcut_pairs=0.0,
            tcut_pairs_weak=0.0,
            tcut_pno=0.0,
            tcut_pno_weak=0.0,
            tcut_mkn=0.0,
        ),
    ),
):
    r = run_dlpno_mp2(mol, basis, hf, df, opts)
    avg_pno = sum(r.pno_per_pair.values()) / max(len(r.pno_per_pair), 1)
    rows.append((label, r.e_corr, r.e_corr / canonical.e_correlation, avg_pno))

# 3. Append the sweep to the .out file run_job already wrote.
out_path = (HERE / STEM).with_suffix(".out")
with open(out_path, "a") as fh:
    fh.write("\n=== TCutPNO convergence vs canonical RI-MP2/cc-pVDZ ===\n")
    fh.write(f"  E_corr(canonical RI-MP2) = {canonical.e_correlation:14.8f} Ha\n")
    fh.write(f"  {'TCutPNO':>9s}  {'E_corr (Ha)':>14s}  {'recovery':>9s}  {'avg PNOs':>8s}\n")
    for label, e, rec, avg in rows:
        fh.write(f"  {label:>9s}  {e:14.8f}  {rec:9.4%}  {avg:8.1f}\n")
    fh.write("  (* = all screens zero: the exactness limit)\n")

print(f"  E(RHF)                  = {hf.energy:14.8f} Ha")
print(f"  E_corr(canonical RI-MP2)= {canonical.e_correlation:14.8f} Ha")
print(f"  {'TCutPNO':>9s}  {'E_corr (Ha)':>14s}  {'recovery':>9s}  {'avg PNOs':>8s}")
for label, e, rec, avg in rows:
    print(f"  {label:>9s}  {e:14.8f}  {rec:9.4%}  {avg:8.1f}")
print("  (* = all screens zero: the exactness limit)")
