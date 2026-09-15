# Embedded-cluster CASSCF/CASPT2 spike

Research spike toward periodic multireference via embedded clusters
(the OpenMolcas-parity path; see `handovers/HANDOVER_PERIODIC_MULTIREF.md`,
milestones MR6-MR10). This is a research/validation study on `main`, not a
supported embedded-cluster runtime feature; it uses the molecular CAS solvers
unchanged and adds no core code.

## Milestone 1: molecular CAS in a static external point-charge field

The first brick of the embedded-cluster pipeline is the ability to run
a molecular RHF + CASSCF/CASPT2 with the QM region sitting in a field
of classical point charges (the embedding / Madelung potential).

**Key finding (enabler already exists, no new core code):** the
one-electron external-charge potential

```
V_ext_{mu,nu} = - sum_A q_A integral phi_mu(r) (1/|r - R_A|) phi_nu(r) dr
```

is provided by the existing pybind11 binding
`compute_nuclear_with_charges(basis, positions, charges)` (already used
by the CPCM solvation driver), and the low-level RHF entry
`run_rhf_scf_with_jk(basis, nelec, S, Hcore, E_nuc, jk, opts)` accepts a
custom `Hcore`. So the embedded reference is:

```
Hcore = T + V_nuc(QM) + V_ext(external charges)
E_nuc = E_nuc(QM-QM) + sum_{QM A, ext B} Z_A q_B / R_AB
```

run the SCF, transform the V_ext-aware AO integrals to the MO basis, and
feed them to the integral-driven CAS solvers (`casci` / `casscf` /
`caspt2`) unchanged.

`spike_external_field.py` validates this end to end:

1. **API cross-check** -- `compute_nuclear_with_charges` over the QM
   nuclei reproduces `compute_nuclear` (the external-charge integral and
   its sign convention are correct).
2. **V_ext = 0 limit** -- the custom-Hcore embedded RHF with zero
   external charges reproduces the standard `run_rhf` (the SCF plumbing
   is transparent).
3. **Field effect** -- RHF / CASCI / CASSCF / CASPT2 all run in the field
   and their energies shift smoothly with the charge magnitude (q-scan),
   recovering the bare result at q = 0.

**The Milestone 1 RHF/CASSCF external validation is DONE.**
`parity_openmolcas_xfield.py` runs OpenMolcas out-of-process (GATEWAY
`XField` point charges on the same cluster, atomic-unit coordinates) and
the vibe-qc embedded RHF and CASSCF match OpenMolcas SCF / RASSCF to
sub-nano-Hartree, both bare and in the field:

```
bare (no field):    RHF |Δ| = 9.2e-10   CASSCF |Δ| = 3.6e-9
XFIELD ±1 cage:     RHF |Δ| = 2.7e-10   CASSCF |Δ| = 2.6e-9
```

So molecular RHF/CASSCF point-charge embedding matches the reference code,
with no new core code. The same live OpenMolcas jobs also completed their
CASPT2 steps, but an OpenMolcas XField CASPT2 energy comparison remains part
of the full MR10 validation matrix.
The study delegates OpenMolcas input generation, execution, and energy parsing
to the canonical `examples/regression/core/runner_openmolcas.py`; the runner's
optional retained-artifact path and XField/charged-QM support are
regression-tested without requiring OpenMolcas.

## Milestone 2: cluster_carve (B1.5)

`cluster_carve.py` carves a finite cluster from a periodic cell
(`cluster_carve(system, center, radius)`): replicate the cell, keep
every image atom within `radius` of the center. Geometry-validated on
MgO rocksalt against the known anion neighbour shells (r = 0.51a ->
[OMg6] at a/2, 0.72a -> +12 O, 0.87a -> +8 Mg; exact counts +
distances), and the carve -> formal-point-charge embed -> embedded
Hamiltonian pipeline composes ([OMg6]/6-31G). A *physical* ionic CASSCF
is deferred to MR6 (a charge-neutral, boundary-corrected Madelung array
fit to a periodic SCF density), not this crude formal-charge shell.

## MR6: the Madelung array

`madelung_array.py` is **MR6a**: it anchors the physics (reproduces the
textbook NaCl Madelung constant, 1.7475645946, from the in-tree
`ewald_point_charge_energy` summer), defines the embedding TARGET (the
exact potential the QM region must feel, `V(infinite lattice) - V(QM
charges)` at off-site cluster probes via `ewald_point_charge_potential`),
and quantifies why the crude Milestone 2 formal-charge shell is
insufficient (non-neutral, conditionally convergent, misses the target
by ~2-4 Ha/e).

`madelung_embedding.py` is **MR6b**: two charge-neutral, boundary-corrected
finite arrays whose OFF-SITE potential over the cluster matches the MR6a
target, closing that gap. Both reproduce the same Ewald target to
< 1e-4 Ha/e on independent probes (so they cross-validate each other), and
both carry net charge `-q_QM` so that `QM_formal + array` is exactly neutral
(the cure for the MR6a conditional divergence):

* `evjen_embedding_array` -- Evjen (1932) fractional boundary weights. A
  finite cube of crystal sites with faces/edges/corners down-weighted
  (1/2, 1/4, 1/8) so the block is electroneutral, QM sites removed. Its
  off-site potential converges to the target as the cube grows
  (rms 1.7e-2 -> 4.9e-6 Ha/e over half-side 2 -> 8 in units of a/2),
  parameter-free.
* `fitted_array` -- Derenzo, Klintenberg & Weber (2000) least-squares fit.
  An inner zone of exact formal charges plus an outer shell whose values
  are fitted (neutrality-constrained KKT + small ridge) to reproduce the
  target. On [OMg6] it reaches rms 3.5e-5 Ha/e on independent test probes
  with the fitted charges within 0.04 e of formal (a tiny boundary
  correction, not invented charges) at a fixed, compact size -- the array
  MR6c hands to the embedded CASSCF.

`parity_mr6c.py` is **MR6c**: the physical embedded RHF/CASSCF on the [OMg6]
ionic cluster (central O(2-) + six Mg(2+), all-electron 6-31G, formal cluster
charge +10) in the MR6b fitted array (918 charges, net -10 -> QM + array
neutral), validated OUT OF PROCESS against OpenMolcas (GATEWAY XField with the
identical array):

* The **Madelung stabilization** dE = E(embedded) - E(bare) -- the array's
  effect alone, with the basis-library difference cancelled -- matches between
  the two codes to |Δ(dE)| = 4.2e-9 Ha. vibe-qc and OpenMolcas feel the SAME
  918-charge field to nano-Hartree; this is the rigorous embedding test.
* Embedded RHF matches OpenMolcas SCF to 6.1e-8 Ha and CASCI (CI in the fixed
  SCF orbitals) to 7.0e-8 Ha -- the 6-31G basis-library floor for Mg/O. The
  array adds no discrepancy of its own: with STO-3G the bare and embedded
  absolute |Δ| are both 1.10e-6 (the XField handling is exact; STO-3G simply
  has a ~1e-6 Mg/O basis mismatch between the codes, so MR6c uses 6-31G).
* Embedded CASSCF matches OpenMolcas RASSCF to ~1.4e-5 Ha. That residual is the
  cross-code orbital optimization converging to slightly different solutions on
  the 3-fold degenerate O 2p (t1u) active space -- it appears only when the
  orbitals relax (CASCI matches to the basis floor), not from the embedding.

So the embedded-cluster CAS pipeline (carve -> Madelung array -> molecular CAS
solvers) is oracle-validated end to end on a real ionic crystal cluster.

`embed_cluster.py` is the **MR8** driver (VALIDATED 2026-06-19): a single
`embed_cluster(system, center, qm_radius, basis, nae, nao, formal, ...)` call
that composes carve + the MR6b Madelung array + the embedded RHF/CASCI/CASSCF/
CASPT2 solvers and returns an `EmbeddedClusterResult` (energies + provenance:
cluster, charge, array size/neutrality, off-site potential-match rms). It is
the proposed public API surface for `vq.embed_cluster(...)`. End-to-end
validated: reproduces MR6c MgO [OMg6] energies to 5e-11 Ha and runs NaCl
[ClNa6]/STO-3G with all checks passing.

Then **MR7** adds the AIMP shell (from the local OpenMolcas AIMP library) for
the frontier-ion exchange/orthogonality the bare point charges omit. The
builders stay in `studies/` for now; productionizing them into
`python/vibeqc/embed/` (with the `database.toml` Evjen/Derenzo entries +
routes) is the MR8 follow-on, pending maintainer review (CLAUDE.md s9).

## Run

```
# Milestone 1 internal checks (integral cross-check, V_ext=0, field effect):
.venv/bin/python studies/embedded-cluster-cas/spike_external_field.py
# Milestone 1 OpenMolcas XFIELD parity (needs the local OpenMolcas build and
# OPENMOLCAS_PYMOLCAS, MOLCAS, and optionally OPENMOLCAS_PYTHON):
.venv/bin/python studies/embedded-cluster-cas/parity_openmolcas_xfield.py
# Milestone 2 cluster_carve geometry validation + carve->embed composition:
.venv/bin/python studies/embedded-cluster-cas/cluster_carve.py
# MR6a exact Madelung potential (target) + the naive-shell gap:
.venv/bin/python studies/embedded-cluster-cas/madelung_array.py
# MR6b charge-neutral boundary-corrected arrays (Evjen + fitted) vs the target:
.venv/bin/python studies/embedded-cluster-cas/madelung_embedding.py
# MR6c physical embedded RHF/CASSCF on [OMg6] vs OpenMolcas XFIELD (same array):
.venv/bin/python studies/embedded-cluster-cas/parity_mr6c.py
# MR8 (DRAFT) embed_cluster driver: carve + array + CAS in one call (MgO + NaCl):
.venv/bin/python studies/embedded-cluster-cas/embed_cluster.py
```
