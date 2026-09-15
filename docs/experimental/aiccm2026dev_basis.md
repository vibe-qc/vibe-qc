---
myst:
  html_meta:
    "description": "Basis sets and auxiliary basis sets for Γ-CCM and χ-CCM: which orbital basis for which purpose, which auxiliary basis each backend needs, and the CLI flags to set them."
    "og:title": "vibe-qc — basis sets for AICCM (Γ-CCM and χ-CCM)"
---

# Basis sets for AICCM

Both CCM lines select their orbital and auxiliary bases through the same
vibe-qc basis-set machinery, but which basis you need depends on the
backend, the system, and the reference you are targeting.

## Orbital basis

The orbital basis labels are the same as everywhere else in vibe-qc:
built-in names (`sto-3g`, `6-31g*`, `cc-pvdz`) or pob-* solid-state
bases (`pob-tzvp-rev2`, `pob-dzvp-rev2`).

| basis | purpose | Δ vs CRYSTAL |
|---|---|---|
| `sto-3g` | **Cross-check.** Cheap, well-conditioned, works for every element H–Kr. The default for the AICCM-2026 benchmark so that Γ-CCM, χ-CCM, bipole, and GDF all run on the same basis. | large (minimal basis), not for production |
| `pob-tzvp-rev2` | **Production comparison.** The solid-state triple-zeta basis used in the CRYSTAL23 `.d12` library. Run the periodic references (`bipole`/`gdf`) at this basis to compare against CRYSTAL23 directly. | ~mHa/atom for insulators |
| `pob-dzvp-rev2` | Lighter solid-state double-zeta; faster SCF convergence than TZVP. | larger than TZVP, smaller than STO-3G |
| `6-31g*` | Molecular basis; usable for 1-D chains and vacuum-padded slabs. | N/A for bulk |

```{note}
The Γ-CCM test set uses `sto-3g` by default for the four-center routes
and `pob-tzvp-rev2` for the periodic reference routes (via
`--crystal-basis`). The χ-CCM test set uses `sto-3g` and specifies the
auxiliary basis explicitly.
```

### Specifying the orbital basis

**Γ-CCM test-set runner:**

```bash
python run_case.py c-diamond aiccm-hf --basis pob-tzvp-rev2
python run_case.py mgo        gdf     --basis pob-tzvp-rev2   # CRYSTAL23 comparison
```

**χ-CCM test-set runner:**

```bash
python run_case_b.py mgo rhf-ri --basis pob-tzvp-rev2
```

**Python API:**

```python
# Γ-CCM
ccm = CCMSystem(system, (2, 2, 2), "pob-tzvp-rev2")

# χ-CCM
basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
```

## Auxiliary basis (RI / density fitting)

The auxiliary basis is only needed for RI, RIJCOSX, and GDF routes.
The four-center backends need no auxiliary basis.

### Γ-CCM

| route | auxiliary basis | how it is chosen |
|---|---|---|
| `aiccm-hf`, `aiccm-ks` | none | four-center, direct |
| `aiccm-ri`, `aiccm-ks-ri` | **implicit** (GDF auto-builds the auxiliary from the orbital basis) | no user control — the GDF driver constructs the JK-fit basis |
| `aiccm-rijcosx` | implicit (RI-J uses GDF; COSX-K uses a grid, no auxiliary) | same as above |
| `ccm_neutral_cderi` | you supply the plane-wave cutoff `ke_cutoff` (Ha) | `ccm_neutral_cderi(ccm, ke_cutoff=200.0)` |

The GDF container auto-selects the auxiliary basis for each element.
For the neutral cderi the PW cutoff controls accuracy — the default
120 Ha is converged for STO-3G; for pob-TZVP use 200–300 Ha.

### χ-CCM

| backend | auxiliary basis | flag |
|---|---|---|
| `four_center` | none | — |
| `ri` | `def2-svp-jk` (default) or any JK-fit basis | `--aux-basis def2-svp-jk` |
| `rijcosx` | same RI-J auxiliary + COSX grid | `--aux-basis def2-svp-jk` |

```bash
# χ-CCM — explicitly set the auxiliary basis
python run_case_b.py mgo rhf-ri --aux-basis def2-svp-jk
python run_case_b.py mgo rhf-ri --aux-basis def2-tzvp-jk
```

The χ-CCM fleet runner (`make_jobs_b.py`) hardcodes `--aux-basis
def2-svp-jk` on every emitted command — the fastest JK-fit basis that
still gives ∼sub-mHa accuracy.

In the Python API:

```python
result = vq.run_periodic_job(
    system, basis,
    method="RHF", jk_method="aiccm2026dev-b",
    aiccm_backend="ri",
    aux_basis="def2-tzvp-jk",
    ...
)
```

## GDF settings (χ-CCM)

χ-CCM's RI/GDF route uses the range-separated GDF (RSGDF) method
with a plane-wave cutoff:

```bash
python run_case_b.py mgo rhf-ri --gdf-method rsgdf --rsgdf-ke-cutoff 200
```

The default is `rsgdf` with `ke_cutoff=200` Ha in the fleet runner.
Higher cutoff = more accurate RI fit, more expensive.

## Basis-set caveats

**Cs has no STO-3G.** The `cscl` test-set system defaults to
`pob-tzvp-rev2` for this reason. The element coverage of STO-3G stops at
Kr (Z=36); Cs (Z=55) and heavier elements need a pob-* or other
compiled basis.

**pob-* and linear dependence.** The pob-* bases are designed to avoid
linear dependence in crystals (they omit the most diffuse functions).
They are the recommended choice for solid-state work. Molecular bases
(cc-pV*Z, 6-31G*) can become linearly dependent in tight crystals —
vibe-qc's linear-dependence removal will catch this, but the removed
functions reduce variational freedom.

**Basis-set superposition error (BSSE).** STO-3G has large BSSE; do not
use it for binding energies or adsorption energies. The TZVP bases
reduce BSSE to chemically acceptable levels (~1–2 kcal/mol).

## See also

- [Γ-CCM reference](../aiccm2026dev_a.md)
- [χ-CCM user guide](../user_guide/aiccm2026dev_b.md)
- [Basis sets](../user_guide/basis_sets.md) — the general vibe-qc basis
  set guide
- [pob-TZVP tutorial](../tutorial/pob_tzvp.md) — using solid-state bases
- [Basis convergence tutorial](../tutorial/basis_convergence.md) —
  general basis-set convergence workflow
