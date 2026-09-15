---
myst:
  html_meta:
    "description": "Troubleshooting common AICCM failure modes: S^CCM non-positive-definite, Madelung over-binding, SCF oscillation, low-dimensional RI exxdiv shift, χ-CCM 4c-blocked, and how to diagnose each from the JSON output."
    "og:title": "vibe-qc - AICCM troubleshooting and diagnostics"
---

# AICCM troubleshooting and diagnostics

This page catalogues the common failure modes and diagnostic signals you
will encounter with Γ-CCM and χ-CCM[^xccm-convention]. Γ-CCM uses the
union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM uses the
finite-translation-group character construction. For each: what it looks like,
which JSON field or log line to
check, and the fix.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention. Their distinction is not a
    choice of Coulomb kernels, and equality for a specified operator/route is
    evidence to establish, not a naming premise.

## 1. `S^CCM` non-positive-definite - ill-conditioned cluster

**Symptom.** The SCF diverges immediately, or the energy is
unphysically low/high, or the run fails with a linear-dependence error.

**Cause.** The cyclic-cluster overlap matrix `S^CCM` has a negative or
near-zero eigenvalue. This happens when the cluster is too small for the
basis (atoms are too close across cell boundaries) or when the geometry
is compressed. The LiH chain at the CRYSTAL .d12 spacing of a=2.0 Å is
the canonical example - the overlap collapses to non-PD.

**Diagnose.** Run the pre-flight check:

```bash
cd studies/aiccm-2026
python testset.py --check
```

Or inspect programmatically:

```python
import numpy as np
from vibeqc.periodic.ccm import CCMSystem, ccm_overlap

ccm = CCMSystem(system, nrep, basis)
S = ccm_overlap(ccm)
eig_min = float(np.linalg.eigvalsh(S)[0])
print(f"min eig S^CCM = {eig_min:.2e}")
assert eig_min > 1e-3, "Cluster is ill-conditioned - increase nrep or spacing"
```

**Fix.** Increase the cluster size (`nrep`), increase the cell spacing,
or switch to a more compact basis. For 1-D chains, ensure the vacuum
padding in non-periodic directions is at least 20-40 bohr.

---

## 2. Union-and-weight Γ versus the neutral fitted-torus control

**Symptom.** The bare-1/r four-center energy is significantly lower
(more negative) than the GDF reference on an ionic crystal. E.g. LiH
rocksalt RKS-PBE/atom: 4-center −4.540 vs GDF −4.203 Ha - a 337 mHa/atom
gap.

**Cause.** These routes do not merely evaluate one named CCM approach with
two interchangeable backends. The four-center route constructs Γ-CCM by
union-and-weight/Wigner--Seitz integral weighting; the GDF route constructs a
neutral fitted-torus control. Their gap is therefore an operator/construction
gap. A leading monopole/Madelung interpretation is a useful hypothesis, not a
proof that one route can replace the other.

**Diagnose.** Compare bare-4c vs GDF on the same system:

```bash
cd studies/aiccm-2026
python run_case.py lih-rocksalt aiccm-hf --out /tmp/
python run_case.py lih-rocksalt aiccm-ri   --out /tmp/
# The aiccm-hf (bare 4c) energy will be ~hundreds of mHa/atom lower
```

Or in Python:

```python
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf
from vibeqc.periodic.ccm import ccm_eri_neutral

e_gamma = run_ccm_rhf(ccm, method="aiccm2026dev-a").energy_per_atom
e_control = run_ccm_rhf_gdf(ccm).energy_per_atom
e_neutral = run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm)).energy_per_atom
print(f"Gamma CCM - neutral control = {(e_gamma - e_control) * 1000:.1f} mHa/atom")
```

**Fix.** Decide which construction or control is being studied and label it
exactly. For Γ-CCM, keep the union-and-weight route and require its own
convergence and external validation before making quantitative ionic claims.
For a neutral fitted-torus control, use `run_ccm_rhf_gdf` / `aiccm-ri` and keep
both SCF and post-HF on the matching neutral operator. Never publish the GDF
control as Γ-CCM, and do not treat cancellation of this gap in an energy
difference as established without an explicit test.

The test-set `four_center_quantitative` flag is historical route guidance; it
does not turn the neutral GDF control into the Γ construction.

---

## 3. Low-dimensional RI exxdiv over-binding

**Symptom.** On 1-D and 2-D systems, the `aiccm-ri` and `aiccm-ks-ri`
(and `gdf`) routes produce an energy that is consistently ~192 mHa/atom
(1-D h-chain) or similar *below* the four-center / bipole reference,
roughly independent of cluster size.

**Cause.** The multi-k GDF RI routes carry a 3D-Ewald `exxdiv='ewald'`
correction that over-binds vacuum-padded low-dimensional cells. This is a
periodic-GDF low-D/vacuum limitation shared by the multi-k `gdf`
reference family, not a CCM-weighting issue.

**Diagnose.** Compare RI vs four-center on the same 1-D/2-D system:

```bash
python run_case.py h-chain aiccm-hf   # union-and-weight Γ-CCM
python run_case.py h-chain aiccm-ri   # GDF RI (over-binds ~192 mHa/atom)
python run_case.py h-chain bipole     # ordinary periodic BIPOLE control
```

**Fix.** For 1-D and 2-D, use the **four-center** (`aiccm-hf`/`aiccm-ks`)
or the real-space **bipole** reference. The RI routes are reliable for
3-D bulk. Do not assume an energy difference cancels the shift or that
correlation is unaffected. Both sides of a difference and every correlated
reference must use the same specified operator, and cancellation must be
tested explicitly.

The `run_case.py` docstring and the runner both print this caveat.

---

## 4. Every χ-CCM backend is blocked in 1-D and 2-D

**Symptom.** `run_periodic_job(..., jk_method="aiccm2026dev-b")` fails on a
1-D or 2-D system for `aiccm_backend="four_center"`, `"ri"`, and
`"rijcosx"`.

**Cause.** The direct four-center fallback is a truncated lower-dimensional
interaction, not the declared neutral finite-torus Hamiltonian. The shared
lower-dimensional RI/GDF mesh is not an alternative: it collapses transverse
reciprocal components and therefore is not a wire/slab Coulomb kernel.

**Diagnose.** The common dimensional guard raises before backend dispatch or
SCF. Treat any archived lower-dimensional B energy as pre-guard defect
evidence, not as a current result.

**Fix.** There is no current χ-CCM-B absolute-energy workaround for 1-D or
2-D. Use χ-CCM only in 3-D. Re-enabling lower dimensions requires one derived
mixed-boundary Hamiltonian with matched electron-electron, electron-nuclear,
nuclear, self-potential, and exchange-seam terms.

---

## 5. SCF oscillation on metals or near-metallic systems

**Symptom.** The SCF energy oscillates between iterations without
converging, or converges to a state with unphysical properties.

**Cause.** Metals (gapless systems) are outside the CCM's design
envelope. The CCM targets *gapped* insulators and semiconductors; a
vanishing gap makes the SCF ill-conditioned. The `na-bcc` system in the
test set is a metal and is flagged as tier-C / convergence-risky.

**Diagnose.** Inspect the SCF trace in the output JSON. In the χ-CCM
runner the `scf_trace_tail` field shows the last 8 iterations' energy
step and gradient norm.

**Fix.** If you must run a metal:
- Enable smearing (`--smearing-temperature 0.01`)
- Use a level shift (`--level-shift 0.3 --level-shift-warmup 5`)
- Try the EDIIS+DIIS accelerator (`--scf-accelerator EDIIS_DIIS`)
- Increase the cluster size (a Γ-only calculation on a small cluster of a
  metal may have no gap at all)

The test-set backup for `na-bcc` is `sc2o3-bixbyite` (a cI **insulator**,
same Bravais lattice).

---

## 6. DIIS subspace too small or too large

**Symptom.** The SCF converges slowly (many iterations) or diverges
after a few DIIS cycles.

**Diagnose.** The DIIS subspace size (`diis_subspace`, default 6) and
start iteration (`diis_start`, default 2) are printed in the JSON output.
A subspace that is too small discards useful history; one that is too
large admits near-linear-dependent error vectors.

**Fix.** For difficult systems, try a larger subspace:

```bash
python run_case.py mgo aiccm-hf --diis-dim 12    # Γ-CCM
python run_case_b.py mgo rhf-ri --diis-subspace 12  # χ-CCM
```

Or delay DIIS to build a better initial density:

```bash
python run_case_b.py mgo rhf-ri --diis-start 5
```

If DIIS consistently destabilises, try damping without DIIS:

```bash
python run_case_b.py mgo rhf-ri --damping 0.5 --no-diis
```

---

## 7. χ-CCM COSX requires at least two cyclic cells

**Symptom.** `run_case_b.py ... rhf-rijcosx` fails when the mesh is
`(1,1,1)`.

**Cause.** The native finite-torus chain-of-spheres exchange bridge needs
at least two cyclic cells per active direction to construct the
neighbour list.

**Diagnose.** The error is explicit.

**Fix.** Use a mesh of at least `(2,1,1)` or `(2,2,1)` for active
directions, or switch to `rhf-ri` for single-cell runs.

---

## 8. Post-HF OOM on dense 3-D four-center

**Symptom.** `run_ccm_mp2` or `run_ccm_ccsd` runs out of memory on a
3-D ionic crystal with more than ~16 atoms.

**Cause.** The padded four-center and neutral cderi builds are
small-cluster validation paths - they build the full `N⁴` ERI tensor.
Dense 3-D four-centers OOM beyond small clusters.

**Diagnose.** Monitor memory. The C++ scalable HF kernel handles the
SCF; the post-HF ERI builder does not yet have a scalable 3-D path.

**Fix.** For scalable Γ-CCM SCF, use the scalable union-and-weight route rather
than changing the construction. If the intended calculation is instead a
neutral fitted-torus control, use `run_ccm_rhf_gdf`; its neutral-cderi DLPNO
stack remains a control, not Γ-CCM. The 3-D χ-CCM post-HF routes use
pair-resolved RI tensors under the χ convention. Canonical four-center
correlation remains feasible only for small clusters.

---

## 9. χ-CCM post-HF is 3-D only

**Symptom.** `run_case_b.py h-chain ri-mp2` or any post-HF route on a
1-D or 2-D system fails closed.

**Cause.** χ-CCM's post-HF Coulomb derivation is validated only in 3-D.
Lower-dimensional long-range gauges are not yet matched.

**Diagnose.** The `b_routes.py` `unsupported_reason` function reports
this explicitly for each (system, route) combination.

**Fix.** Run post-HF on 3-D systems only. The coverage profile in
`make_jobs_b.py` intentionally restricts post-HF to the 3-D LiH anchor.

---

## 10. Neutral-control DLPNO requires a matched neutral reference

**Symptom.** `ccm_dlpno_mp2` or `ccm_dlpno_ccsd` produce correlation
energies that disagree with the canonical CCM MP2/CCSD(T) by more than
machine epsilon at zero PNO truncation.

**Cause.** This DLPNO control machinery expects the neutral `cderi` (from
`ccm_neutral_cderi`) and the neutral `g_neutral` (from
`ccm_eri_neutral`). Passing the bare-1/r four-center or its cderi shifts
the denominators.

**Diagnose.** Check the no-truncation gate:
`ccm_dlpno_mp2(..., tcut_pno=0.0).e_corr` must equal
`run_ccm_mp2(..., eri=g_neutral).e_correlation` to machine ε.

**Fix.** For this neutral-control calculation, construct a matched neutral
reference and do not label the result Γ-CCM:

```python
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi

g = ccm_eri_neutral(ccm, ke_cutoff=120.0)
L = ccm_neutral_cderi(ccm, ke_cutoff=120.0)
scf = run_ccm_rhf(ccm, eri=g)
dlpno = ccm_dlpno_mp2(ccm, scf, cderi=L, localize="pm")
```

---

## See also

- [Γ-CCM reference](../aiccm2026dev_a.md) - construction/control operator
  boundaries and matched-reference rules
- [χ-CCM user guide](../user_guide/aiccm2026dev_b.md) - backend caveats
- [Periodic SCF convergence](../tutorial/periodic_scf_convergence.md) -
  general SCF convergence aids (DIIS, level shift, damping, quadratic
  fallback)
- [Troubleshooting](../troubleshooting.md) - general vibe-qc
  troubleshooting (not CCM-specific)
