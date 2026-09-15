# Fetching external data with `vqfetch`

**You'll learn:** how to pull crystal structures from open databases
(OPTIMADE / Materials Project / COD) and experimental reference data
from NIST CCCBDB into ready-to-run vibe-qc inputs, with full
per-record provenance (source ID + URL + DOI + license + fetched
timestamp) baked into every emitted file.

**Why:** stop hand-transcribing lattice constants from papers and
atomization energies from CCCBDB into your input files. The fetcher
does it for you, attaches the citation, caches the result, and emits
a regression-suite SPEC + an executable input script. Provenance is
in the artefact, reproducibility by construction, not retroactively.

**Prerequisites:** vibe-qc installed with the `fetch` extra:

```sh
pip install -e '.[fetch]'   # adds optimade-python-tools, ase, bs4, lxml
```

Two runnable example scripts ship alongside this tutorial:

* [`examples/input-vqfetch-optimade-mgo.py`](../../examples/input-vqfetch-optimade-mgo.py)
  - Phase 1 walkthrough (OPTIMADE structures).
* [`examples/input-vqfetch-reference-h2o.py`](../../examples/input-vqfetch-reference-h2o.py)
  - Phase 2 walkthrough (CCCBDB experimental references + D₀ → Dₑ
  correction + geometry bridge to a vibe-qc `Molecule`).

## Part 1: pull a crystal structure (Phase 1)

The shortest path to a ready-to-run vibe-qc input is the **canonical
set**, five round-trip-verified structures with pinned Materials
Project IDs:

```sh
vqfetch canonical mgo_rocksalt --quick
```

This prints exactly two paths to stdout (both **emitted at runtime**,
not checked into the repo):

```
examples/regression/systems/periodic/mgo_rocksalt.py
examples/input-mgo_rocksalt-sto-3g.py
```

The first is a regression-suite `PeriodicSpec` module; the second is
a standalone executable that calls `vq.run_rks_periodic_scf` on the
fetched cell. Run the second one:

```sh
.venv/bin/python examples/input-mgo_rocksalt-sto-3g.py
```

Every file emitted carries Provenance:

```python
provenance=Provenance(
    source_db='OPTIMADE/mp',
    source_id='mp-1265',
    source_url='https://optimade.materialsproject.org/structures/mp-1265',
    original_reference='',
    license='CC-BY-4.0',
    fetched_at='2026-05-16T16:16:34Z',
    fetcher_version='0.1.0',
    notes='',
)
```

### Programmatic equivalent

The same fetch-and-emit flow is available from Python. This looks up the
rocksalt MgO structure by its Materials Project ID, then writes both the
regression-suite spec module and a runnable input script:

```python
from vibeqc.fetch import (
    emit_input_script, emit_spec_module, fetch_optimade,
)

# ID-lookup is unambiguous — formula-based queries hit the
# polymorph-disambiguation footgun (MgO has rocksalt + CsCl + hex
# polymorphs all in MP; see § 11 of the structure-fetcher handover).
spec = fetch_optimade(optimade_id="mp/mp-1265", quick=True)[0]

emit_spec_module(spec,  "examples/regression/systems/periodic/")
emit_input_script(spec, "examples/", basis="sto-3g", method="rks-lda")
```

For when you only have a formula and want to see what polymorphs
exist, the multi-candidate API (v0.8.0) returns a deduped ranked
list:

```python
candidates = fetch_optimade(formula="MgO", max_results=5)
for s in candidates:
    print(f"  sg={s.space_group}  a={s.lattice_ang[0][0]:.3f}  "
          f"n_atoms={len(s.atoms)}  {s.provenance.source_id}")
```

Output (across the OPTIMADE federation):

```
  sg=Fm-3m  a=4.194  n_atoms=8  mp-1265        ← canonical rocksalt
  sg=Pm-3m  a=2.661  n_atoms=2  mp-1009127     ← HP CsCl phase
  sg=P6_3mc a=3.27   n_atoms=4  mp-1191789     ← HP hexagonal phase
  …
```

`docs/roadmap.md` § VFETCH-X1 tracks the `vqfetch list-candidates`
CLI surface for the same workflow (v0.8.x maintenance line).

## Part 2: pull experimental reference data (Phase 2)

The `vqfetch reference` command pulls experimental data for a molecule by
its CAS number from NIST CCCBDB. Here it fetches the water record:

```sh
vqfetch reference --cas 7732-18-5      # H₂O — 7732-18-5 is its CAS
```

Emits a JSON `ExperimentalReference` record under
`examples/regression/references/<cas>.json` containing
atomization energy, ΔH_f at 0K and 298K, vibrational fundamentals
+ harmonics, IR intensities, ionization energy, proton affinity,
dipole moment, polarizability, bond list, and the explicit
Cartesian geometry, all sourced from CCCBDB's `exp2x.asp` page,
all carrying the NIST DOI `doi:10.18434/T47C7Z` in Provenance.

### ⚠ The D₀ vs Dₑ footgun

CCCBDB's atomization energy is **D₀**, the thermodynamic value at
0 K, with zero-point vibrational energy already included. The
value most QC textbooks quote (and the value most QC method
papers report) is **Dₑ**, the electronic-only atomization
energy, ZPE removed. These differ by **+ZPE** = +½·Σν, which is
*not* small (≈ 13 kcal/mol for water, ≈ 28 kcal/mol for CH₄).

For water:

```
AE D₀ = 219.35 kcal/mol  (CCCBDB)
+ZPE  =  12.88 kcal/mol  (½ · (1595 + 3657 + 3756 cm⁻¹) · h c Nₐ / 4184)
─────────────────────────
AE Dₑ = 232.23 kcal/mol  ≈ 232.5 kcal/mol (Curtiss G2/3 reference)
```

The Provenance notes on every emitted record include this
distinction so it can't get conflated in downstream comparisons:

```
notes = 'fetched from CCCBDB exp2x; atomization_energy_kcal_per_mol
is D_0 (0 K, ZPE-included); for D_e add ZPE = 0.5 * sum(
vibrational_fundamentals_cm_inv) converted to kcal/mol'
```

### Programmatic equivalent + run on NIST geometry

From Python you can fetch the CCCBDB record, bridge its experimental
geometry straight into a vibe-qc `Molecule`, and run an SCF on it. This
fetches water, converts the NIST geometry, and runs RHF/STO-3G:

```python
from vibeqc.fetch.references import (
    experimental_geometry_to_molecule_spec,
    fetch_cccbdb,
)
import vibeqc as vq

ANG_TO_BOHR = 1.0 / 0.529177210903

ref = fetch_cccbdb(cas="7732-18-5")                          # H₂O
spec = experimental_geometry_to_molecule_spec(ref, slug="h2o_nist")

mol = vq.Molecule([
    vq.Atom(a.z, [c * ANG_TO_BOHR for c in a.xyz_ang]) for a in spec.atoms
])
result = vq.run_job(mol, basis="sto-3g", method="rhf")
print(f"E(NIST geometry) = {result.energy:.8f} Ha")
# → -74.96302314 Ha (NIST r=0.958 Å vs Szabo-Ostlund r=1.0 Å: Δ ≈ +1 mHa)
```

### Attaching CCCBDB columns to a regression-suite report

The regression-suite runner can fold the fetched NIST values into its
report. Passing `--include-experimental-reference cccbdb` adds an
experimental-reference column for each system in the run:

```sh
python -m examples.regression.run_suite \
    --systems h2o,ch4,hf --bases sto-3g --methods rhf,rks-lda \
    --include-experimental-reference cccbdb
```

The emitted `summary.md` gets an "Experimental references" section
with the NIST values + DOI citation:

| system | formula | AE D₀ (kcal/mol) | IE (eV) | μ (D) | vib fundamentals (cm⁻¹) | NIST source |
|---|---|---|---|---|---|---|
| h2o | H2O | 219.35 | 12.621 | 1.857 | 1595, 3657, 3756 | [CCCBDB 7732-18-5](https://cccbdb.nist.gov/exp2x.asp?casno=7732185) |

## Cache & offline mode

Every successful fetch lands at
`$XDG_CACHE_HOME/vibeqc/fetch/<source_db>/<source_id>.json` (default
`~/.cache/vibeqc/fetch/`). TTL is 30 days for OPTIMADE / MP / NOMAD
/ CCCBDB; **infinite for COD** because COD entries are immutable
post-publish.

Two env-var levers:

```sh
VIBEQC_FETCH_CACHE_ROOT=/tmp/myrun  vqfetch canonical mgo_rocksalt
VIBEQC_FETCH_CACHE_ONLY=1           vqfetch reference --cas 7732-18-5
```

`--cache-only` (set the env var or pass the flag) makes vqfetch
refuse to hit the network, useful for offline / CI / reproducibility.

## Polite rate limiting (CCCBDB)

NIST is a US-government public resource shared by the whole
community. `client_cccbdb.py` enforces:

- **≥ 1 second between HTTP requests** per process (per-process
  throttle).
- Declared User-Agent identifying vibe-qc + version + repo URL so
  NIST can find us if there's a problem.
- Retry-After honoured on 429/503 with exponential backoff (5 s →
  60 s, 3 attempts max).

The 30-day cache means re-fetches are rare in practice, and every
cached hit is a request NIST doesn't have to serve.

## Where to read more

* [`docs/user_guide/external_structures.md`](../user_guide/external_structures.md)
  - full Phase-1 reference (CLI flags, every source, troubleshooting).
* [`docs/user_guide/reference_data.md`](../user_guide/reference_data.md)
  - full Phase-2 reference (CCCBDB schema, every property, regression
  integration).
* The smoke commands in this tutorial, which cover the former structure-fetch
  and reference-data acceptance harnesses. Run them after a fresh install to
  verify the pipeline works on your machine.
* [`docs/roadmap.md`](../roadmap.md) §§ `v0.8.0 vqfetch external-data`
  + `v0.8.x vqfetch polish`, what shipped in v0.8.0 and what's
  queued in the maintenance line (`runner_crystal.py` for periodic
  parity, NIST WebBook fallback, ATcT integration, …).
