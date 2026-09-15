---
myst:
  html_meta:
    "description": "How to study the distinct union-and-weight Γ-CCM and finite-character χ-CCM approaches side by side without assuming a matched finite Hamiltonian."
    "og:title": "vibe-qc - comparing Γ-CCM and χ-CCM"
---

# Comparing Γ-CCM and χ-CCM

Γ-CCM (`aiccm2026dev-a`) and χ-CCM[^xccm-convention]
(`aiccm2026dev-b`) are distinct approaches to the ab-initio Cyclic Cluster
Model. Γ-CCM uses union-and-weight/Wigner--Seitz integral weighting; χ-CCM
uses a finite-translation-group character construction. They are developed in
separate workstreams, share only the geometry registry (`testset.py`), and are
compared head-to-head to determine, rather than presume, where specified
finite operators and routes agree.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM approaches. Γ-CCM uses
    the union-and-weight/Wigner--Seitz integral-weighting construction; χ-CCM
    uses the finite-translation-group character construction. They are compared
    at a declared common exchange-q=0 convention. Their distinction is not a
    choice of Coulomb kernels, and equality for a specified operator/route is
    evidence to establish, not a naming premise.

```{important}
The reportable Γ/χ approach-comparison map is currently empty. The existing
`aiccm-hf-direct` versus B `rhf-ri` pair is a neutral-torus real-Gamma versus
character representation control, not evidence for the union-and-weight
Γ-CCM approach. Within the χ-defined Hamiltonian those two representations are
related by an exact finite Fourier transform. Cross-approach equality remains
evidence to establish at a declared common exchange-`q=0` convention.
```

## Running the same system through both approaches

### Via the test-set runners

The `studies/aiccm-2026/` directory has separate runners that share only
`testset.py` for the geometry:

```bash
cd studies/aiccm-2026

# Γ-CCM: four-center HF on diamond
python run_case.py c-diamond aiccm-hf --out results-a/

# χ-CCM: RHF with four-center backend on diamond
python run_case_b.py c-diamond rhf-4c --out results-b/

# Inspect the two records. The current comparator reports the approach
# comparison as not-defined; it does not form an energy delta.
python compare_b.py results-b/
```

`compare.py` still produces a single-line route table. Its old `--vs`
head-to-head mode is disabled because route names alone cannot prove a matched
Γ-CCM/χ-CCM operator. `compare_b.py` reports the
`gamma_ccm_chi_ccm_approach_comparison_status` explicitly as `not-defined`.

### Running the neutral-torus Fourier control

The one retained cross-run control is not an approach comparison. It compares
B `rhf-ri`, evaluated on the character mesh, with `aiccm-hf-direct`, evaluated
in a real-Gamma supercell, for a separately attested neutral fitted-torus
Hamiltonian:

```bash
python run_case.py c-diamond aiccm-hf-direct --out results-control/
python run_case_b.py c-diamond rhf-ri --out results-b/

python compare_b.py results-b/ \
    --real-gamma-control-results results-control/
```

When the complete control contract is present, its result is written only to
the `real_gamma_control_*` and
`character_minus_real_gamma_control_mha_per_atom` fields. It is never labelled
as a Γ-CCM/χ-CCM delta. The legacy `--a-results` spelling is accepted only as
an alias for `--real-gamma-control-results` and should not be used in new
commands.

### Fleet-scale comparison

Generate both batches and submit:

```bash
# Γ-CCM full matrix
python make_jobs.py | sh

# χ-CCM SCF matrix (all systems, all routes)
python make_jobs_b.py --profile scf | sh

# After results land, inspect the B records. No approach delta is formed.
python compare_b.py results-b/ --csv comparison.csv

# Optionally add the separately generated neutral-torus real-Gamma control.
python compare_b.py results-b/ \
    --real-gamma-control-results results-control/ --csv control.csv
```

### Adding a CRYSTAL23 reference

Populate the `crystal_refs_b.json` template with per-system
per-atom energies from actual CRYSTAL23 runs, then:

```bash
python compare_b.py results-b/ --crystal-refs crystal_refs_b.json
```

The template schema is:

```json
{
  "c-diamond": {"rhf": -38.07736, "pbe": -38.07736}
}
```

Values are energy per primitive-cell atom in Hartree. Only 3-D systems belong
in this B reference template while the all-backend 1-D/2-D fail-close is active.

## What to expect

### A matched convention is necessary, not sufficient

A meaningful approach comparison must hold the Coulomb operator and
exchange-`q=0` convention fixed. For current 3-D χ-CCM-B records those labels
are `coulomb_kernel="3d-periodic-g0"` and
`exchange_q0="bvk-ewald"`. Matching them does not prove that the
union-and-weight and finite-character constructions produced the same finite
Hamiltonian, variational space, or approximation. Those bindings require a
route-specific derivation and fingerprint.

No such cross-approach route is currently reportable. In particular,
`run_ccm_rhf_gdf` and `aiccm-hf-direct` are neutral fitted-torus controls; they
are not the union-and-weight Γ-CCM construction. Their agreement with B `rhf-ri`
tests a specified fitted operator or its Fourier evaluation, not Γ-CCM/χ-CCM
approach equality.

### Current comparison boundaries

| axis | current rule | reason |
|---|---|---|
| **Approach construction** | report `not-defined` | No route map currently binds a union-and-weight Γ-CCM result to a finite-character χ-CCM result for one fully attested operator. |
| **Neutral-torus Fourier control** | use only `real_gamma_control_*` fields | Character and real-Gamma evaluations of one specified block-circulant Hamiltonian are Fourier related. This is an internal representation theorem, not a construction identity. |
| **Exchange seam** | require the same active `exchange_q0` convention | Different seams define different finite-size operators. A matching label is still only one part of the contract. |
| **Low dimensions** | do not run or quote χ-CCM-B absolute energies | All 1-D/2-D B SCF backends fail closed until a shared wire/slab Coulomb convention is derived. |
| **Cluster size** | study each approach separately | No monotone cross-approach delta or common finite-size remainder is assumed. A thermodynamic-limit claim requires its own convergence evidence. |

The non-orthorhombic and ionic 3-D systems remain useful discriminating tests.
Agreement on them would be evidence only after both records pass a complete
construction, operator, numerical-input, and producer fingerprint.

## Interpreting a disagreement

1. **Check the construction identity.** A B record must carry the exact
   `ccm_approach="chi-ccm"`,
   `ccm_construction="finite-translation-group-character"`, and
   `evaluation_representation="gamma-centred-character-mesh"` fields. A future
   Γ record needs its own union-and-weight identity. A real-Gamma control must
   not be relabelled as Γ-CCM.

2. **Check the convention descriptor.** Both results must carry
   `coulomb_kernel="3d-periodic-g0"` and
   `exchange_q0="bvk-ewald"`. If not, the disagreement is a convention
   mismatch. If they do match, further operator and construction checks are
   still required.

3. **Check what is being compared.** If the pair is B `rhf-ri` versus
   `aiccm-hf-direct`, interpret it only as the neutral-torus character versus
   real-Gamma control. For an actual Γ-CCM/χ-CCM pair, stop unless a current
   route-specific comparison contract exists.

4. **Check every numerical fingerprint.** Basis, structure, mesh, AO space,
   finite operator, auxiliary fit, thresholds, smearing, source, native core,
   and producer attestation must match the applicable contract. Do not infer a
   missing field from a filename or route name.

5. **Check dimensional support.** A 1-D/2-D χ-CCM-B absolute energy is
   unreportable regardless of backend. Keep such rows as explicit unsupported
   coverage until the shared mixed-boundary kernel is derived.

6. **If a future complete approach contract passes and a delta remains,** the
   result is evidence of a construction, discretization, or implementation
   difference to localize. It is not automatically a bug in either approach.

## Cross-stream property comparison

| property | Γ-CCM | χ-CCM | current cross-approach status |
|---|---|---|---|
| HF/KS energy per atom | `energy_per_atom` | `energy / n_atoms` | `not-defined`; needs a matched construction/operator contract |
| Correlation energy | `e_correlation` | route-specific correlation fields | `not-defined`; reference and correlated spaces are not yet cross-bound |
| HOMO/LUMO gap | `ccm_homo_lumo_gap` | `fundamental_gap` in SCF record | `not-defined`; eigenvalue conventions and finite operators must be bound |
| Mulliken charges | `ccm_mulliken_charges` | `mulliken_charges` in properties | `not-defined`; overlap and density contracts must be established |
| Dipole moment | `ccm_dipole` (finite cluster) | not emitted | unavailable and gauge dependent |
| Forces | Γ route-specific force surface | analytic total gradient fails closed | unavailable |
| Density idempotency | not emitted | `density_idempotency_error` | N/A |

## See also

- [Γ-CCM reference](../aiccm2026dev_a.md)
- [χ-CCM user guide](../user_guide/aiccm2026dev_b.md)
- [AICCM-2026 benchmark README](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README.md)
- [χ-CCM fleet inputs](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README_B.md)
- [Diamond comparison](aiccm2026dev_diamond_compare.md) - a concrete
  worked example of cross-stream comparison on diamond
- [Archived Γ-CCM position memo](https://vibe-qc.com/docs/) - the
  debate rounds and settled items
