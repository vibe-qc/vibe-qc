---
myst:
  html_meta:
    "description": "vqfetch reference, pull experimental reference data from NIST CCCBDB into vibe-qc for QC method validation. Atomization energies, ΔHf°, vibrational frequencies, dipole, IE."
    "og:title": "vibe-qc, experimental reference data (vqfetch CCCBDB)"
    "og:description": "Pull NIST CCCBDB experimental data by CAS number. Atomization (D₀ vs Dₑ footgun documented). Eight-molecule canonical set. NIST SRD source."
---

# Experimental reference data (`vqfetch reference`)

`vqfetch reference` (Phase 2 of the v0.8.0 external-data
integration) pulls experimental and computed reference values
from the
[NIST Computational Chemistry Comparison and Benchmark DataBase
(CCCBDB)](https://cccbdb.nist.gov/). Use it to validate vibe-qc
calculations against the standard QC reference set without
hand-curating the numbers.

The source is **NIST Standard Reference Database 101**, DOI
[10.18434/T47C7Z](https://doi.org/10.18434/T47C7Z) (Release 22,
May 2022, Editor: R. D. Johnson III). Citation is required for
published work; the DOI travels with every fetched record. One
URL per molecule, one parser, one HTTP request, full provenance
preserved.

## What it pulls

For each molecule (looked up by CAS number), `fetch_cccbdb()`
returns an `ExperimentalReference` record carrying:

| Property | Field | Unit |
|----------|-------|------|
| Atomization energy (**D₀**, ZPE-included) | `atomization_energy_kcal_per_mol` | kcal/mol |
| Enthalpy of formation at 0 K | `enthalpy_of_formation_0_kj_per_mol` | kJ/mol |
| Enthalpy of formation at 298 K | `enthalpy_of_formation_298_kj_per_mol` | kJ/mol |
| Entropy at 298.15 K | `entropy_298_j_per_mol_per_k` | J/(mol·K) |
| Heat capacity at 298.15 K | `heat_capacity_298_j_per_mol_per_k` | J/(mol·K) |
| Ionization energy | `ionization_energy_ev` | eV |
| Proton affinity | `proton_affinity_kj_per_mol` | kJ/mol |
| Vibrational fundamentals | `vibrational_fundamentals_cm_inv` | cm⁻¹ (tuple) |
| Vibrational harmonics (where available) | `vibrational_harmonics_cm_inv` | cm⁻¹ (tuple) |
| IR intensities | `ir_intensities_km_per_mol` | km/mol (tuple) |
| Dipole moment | `dipole_moment_debye` | Debye |
| Polarizability (isotropic mean) | `polarizability_au` | atomic units (a₀³) |
| Cartesian geometry | `cartesian_geometry_ang` | tuple of `(label, z, x, y, z_coord)` in Å |
| Bond list (internal coords) | `bond_lengths_ang` / `bond_angles_deg` | Å / ° |

Each scalar field has a paired `_uncertainty_*` field where
NIST publishes one. Missing values come back as `None`; tuple
fields default to `()` when CCCBDB doesn't expose them.

## ⚠️ The D₀ vs Dₑ footgun

**`atomization_energy_kcal_per_mol` is D₀, not Dₑ.**

That is: it's the **thermodynamic** atomization energy
(includes zero-point vibrational energy), not the
**electronic** atomization energy (the bare energy difference
between molecule minimum and dissociated atoms, what most
QC textbooks and benchmark papers call "the atomization
energy" without qualification).

This matters because **what your SCF calculation produces is
Dₑ**, not D₀. Comparing CCCBDB's `atomization_energy_kcal_per_mol`
against your raw SCF atomization will be wrong by ZPE, typically
~3-10 kcal/mol for small molecules, more for large ones.

For H₂O specifically: `atomization_energy_kcal_per_mol` =
**219.35 kcal/mol** (D₀); the textbook Dₑ ≈ **232.4 kcal/mol**.
The difference is the ZPE ≈ 13.05 kcal/mol from
0.5 × (1595 + 3657 + 3756) cm⁻¹.

To get **Dₑ from CCCBDB** (matches your raw SCF):

```python
from vibeqc.fetch import fetch_cccbdb

ref = fetch_cccbdb(cas="7732-18-5")  # H₂O
# cm⁻¹ → kcal/mol uses h c N_A / (1000 cal/kcal · 4.184 J/cal):
CM_INV_TO_KCAL_PER_MOL = 0.0028591459
zpe_kcal_per_mol = 0.5 * sum(ref.vibrational_fundamentals_cm_inv) * CM_INV_TO_KCAL_PER_MOL
de_kcal_per_mol = ref.atomization_energy_kcal_per_mol + zpe_kcal_per_mol
# For H2O: de ≈ 232.4 kcal/mol — matches the QC-textbook value.
```

The footgun is also flagged in the `Provenance.notes` field of
every CCCBDB record, so callers reading the record cold should
see the warning. It's documented in the `ExperimentalReference`
dataclass docstring as well.

## Quick start: water

```sh
vqfetch reference --cas 7732-18-5
# Output (one path):
# examples/regression/references/7732-18-5.json
```

Then in Python:

```python
from vibeqc.fetch import fetch_cccbdb

ref = fetch_cccbdb(cas="7732-18-5")
print(ref.atomization_energy_kcal_per_mol)   # 219.35 (D_0!)
print(ref.dipole_moment_debye)               # 1.857
print(ref.vibrational_fundamentals_cm_inv)   # (1595.0, 3657.0, 3756.0)
print(ref.ionization_energy_ev)              # 12.621
```

The same record is also written to
`examples/regression/references/7732-18-5.json` so the
regression suite can reattach it to the H₂O test cases (see
*Regression-suite integration* below).

## Eight-molecule canonical set (round-trip-verified)

The v1 acceptance harness covers eight canonical molecules
representative of the small-molecule QC benchmark literature:

| Molecule | CAS | Formula | Why it's in the canonical set |
|----------|-----|---------|------------------------------|
| Water | 7732-18-5 | H₂O | Universal QC test molecule |
| Methane | 74-82-8 | CH₄ | Classic single-bond / Tₐ reference |
| Ammonia | 7664-41-7 | NH₃ | C₃ᵥ, lone pair, polar H bond |
| Hydrogen fluoride | 7664-39-3 | HF | Polar diatomic; high IE benchmark |
| Oxygen | 7782-44-7 | O₂ | Open-shell triplet ground state |
| Ozone | 10028-15-6 | O₃ | Multireference; classic DFT failure case |
| Carbon dioxide | 124-38-9 | CO₂ | Linear, π system |
| Formaldehyde | 50-00-0 | H₂CO | Smallest carbonyl; pyramidalisation tests |

Atomization energies (D₀) from the smoke harness:

```
h2o   219.35    ch4   392.44   nh3  276.68   hf   135.41   kcal/mol
o2    117.97    o3    142.43   co2  381.91   h2co 357.29   kcal/mol
```

These match published values for the molecule set (HF 135.4,
CH₄ 392.4, CO₂ 381.9, H₂O 219.3 kcal/mol etc.).

## Round-trip recipe: experimental geometry → vibe-qc Molecule

CCCBDB carries experimental Cartesian geometries. The
geometry-bridge helper builds a vibe-qc `MoleculeSpec` directly:

```python
from vibeqc.fetch import (
    fetch_cccbdb,
    experimental_geometry_to_molecule_spec,
)
import vibeqc as vq

ANG_TO_BOHR = 1.0 / 0.529177210903

ref      = fetch_cccbdb(cas="7732-18-5")                  # NIST H₂O
mol_spec = experimental_geometry_to_molecule_spec(ref)    # MoleculeSpec at
                                                          #   r(O–H) = 0.958 Å,
                                                          #   ∠HOH = 104.4776°
# MoleculeSpec is the regression-suite dataclass; build a
# vibe-qc Molecule from spec.atoms (positions are in Å):
mol = vq.Molecule([
    vq.Atom(a.z, [c * ANG_TO_BOHR for c in a.xyz_ang]) for a in mol_spec.atoms
])
result = vq.run_job(mol, basis="sto-3g", method="rhf")
# E = -74.96302314 Ha (8 SCF iters, 0.17 s on a laptop).
```

The fetched geometry differs slightly from textbook hand-curated
geometries (e.g. Szabo-Ostlund's H₂O is at r(O-H) = 0.957 Å,
∠HOH = 104.5°, gives a ~1 mHa shift at HF/STO-3G).

The bridge accepts overrides for open-shell species or charge:

```python
mol_spec = experimental_geometry_to_molecule_spec(
    ref,
    slug="o2_triplet",
    overrides={"multiplicity": 3, "family": "molecule_open_shell"},
)
```

Default multiplicity is electron-parity (even electrons → 1,
odd → 2). Open-shell ground states (O₂, NO, …) need the
explicit override.

## Regression-suite integration

Attach a CCCBDB reference to every molecular case in the
regression suite:

```sh
python -m examples.regression.run_suite \
    --systems h2o,ch4,hf --bases sto-3g --methods rhf,rks-lda \
    --include-experimental-reference cccbdb
```

The generated `summary.md` gains an "Experimental references"
section, with the NIST DOI footer and per-record permalinks
back to CCCBDB. Lookup is by `system_id` against the canonical
molecule table (h2o, ch4, nh3, hf, o2, o3, co2, h2co); molecular
cases without a canonical entry pass silently (no attachment).

The rendered section looks like:

```
## Experimental references

| system | formula | AE D₀ (kcal/mol) | IE (eV) | μ (D) | α (a.u.) | vib fundamentals (cm⁻¹) | NIST source |
|---|---|---|---|---|---|---|---|
| h2o | H2O | 219.35 | 12.621 | 1.857 | 1.501 | 1595, 3657, 3756 | [CCCBDB 7732-18-5](...) |
| ch4 | CH4 | 392.44 | 12.610 | …     | …       | 1306, 1534, 2917, 3019 | [CCCBDB 74-82-8](...) |
…
> NIST CCCBDB Release 22 (May 2022), R. D. Johnson III, ed.
> Standard Reference Database 101, doi:10.18434/T47C7Z.
```

## Cache + offline mode

Same XDG-compliant cache as
[`vqfetch` for structures](external_structures.md#cache--offline-mode)
(`$XDG_CACHE_HOME/vibeqc/fetch/CCCBDB/<cas>.json`). NIST records
are immutable in practice; default TTL is 30 days but the cache
hits for years.

* `--no-cache`, bypass cache reads (still writes through).
* `--cache-only`, refuse live HTTP entirely. Equivalent env
  var: `VIBEQC_FETCH_CACHE_ONLY=1`.

Useful for offline reproducibility on cluster compute nodes
without external network, pre-warm the cache locally, then
ssh / rsync `$XDG_CACHE_HOME/vibeqc/fetch/` to the cluster.

### GFN2 parameter caches on offline fleet nodes

GFN2 parameters use a separate cache from CCCBDB. They are fetched on demand
under their upstream LGPL terms and are never bundled in the repository.
Before a runtime is admitted for offline GFN2 work, deployment must stage the
reviewed parameter cache on shared storage and retain both its file SHA256 and
upstream source SHA256 in the deployment evidence.

`agentic-loop/fleet/seed-gfn2-cache.py` accepts a previously fetched cache and
those two independently reviewed hashes. It defaults to a read-only plan;
`--apply` publishes complete bytes under a directory named by the cache hash,
without replacing an existing cache. For example, from a checkout:

```sh
python3 agentic-loop/fleet/seed-gfn2-cache.py \
  --source /path/to/reviewed/gfn2_xtb_params.toml \
  --cache-root ~/.cache/vibeqc/gfn2-pinned \
  --expected-cache-sha256 "$GFN2_CACHE_SHA256" \
  --expected-source-sha256 "$GFN2_SOURCE_SHA256" --apply
```

Set `VIBEQC_GFN2_CACHE_DIR` to the directory in the returned receipt **before
importing vibeqc**. Existing jobs keep their original cache; no default cache
or runtime setting is switched by this staging command. Runtime rollout
integration remains in progress under #127. A copied-file receipt establishes
byte identity only: offline molecular and periodic jobs must also complete
and record the GFN2 lineage in their calculation manifests before fleet
acceptance.

## Polite rate limiting (required)

NIST is a US-government public resource shared by the whole
community. The CCCBDB client enforces:

* **≥ 1 s** spacing between consecutive requests in the same
  process (semaphored).
* Declared **User-Agent** carrying vibe-qc + version + repo
  URL + maintainer email so NIST can contact us if there's a
  problem.
* **Retry-After** honour on 429 / 503, with exponential backoff
  (5 s → 60 s, max 3 retries) before giving up.

Don't work around the throttle. Cache aggressively instead.

## Provenance

Every CCCBDB record carries:

```python
Provenance(
    source_db="CCCBDB",
    source_id="7732-18-5",                                # hyphenated CAS
    source_url="https://cccbdb.nist.gov/exp2x.asp?casno=7732185",
    original_reference="doi:10.18434/T47C7Z",             # NIST SRD 101
    license="NIST SRD",
    fetched_at="2026-05-09T14:00:00Z",
    fetcher_version="0.1.0",
    notes="fetched from CCCBDB exp2x; atomization_energy_kcal_per_mol is D_0 ...",
)
```

For published work, **cite NIST Standard Reference Database 101**:

> NIST Computational Chemistry Comparison and Benchmark Database,
> NIST Standard Reference Database Number 101, Release 22,
> May 2022, Editor: Russell D. Johnson III. DOI: 10.18434/T47C7Z.

The CCCBDB record also surfaces NIST's standard disclaimer about
data uncertainty in its [About page](https://cccbdb.nist.gov/about.asp);
none of the data in CCCBDB carries a warranty of any kind.

## Architecture note: one URL, one parser

The original handover doc proposed many table-code-specific
parsers (`ea2x.asp`, `enthalpyx.asp`, `vibs2x.asp`, …). Live
probing during the v0.8.0 implementation showed:

* `ea2x.asp` → 500 Internal Server Error
* `enthalpyx.asp` → 404
* `ea1x.asp` / `atomize1x.asp` / `exp1x.asp` → 200 but
  *form-only* landing pages that don't honour `?casno=`
* **`exp2x.asp` → the one-stop per-molecule master page** with
  thermochem + vibrational + geometry + IE + dipole +
  polarizability in 23 well-structured tables.

So v0.8.0 ships with one URL, one parser, one HTTP request per
molecule. Atomization energy isn't directly tabulated; it's
derived from Hf°(0K) + a small CODATA atomic-enthalpy table
([`atomic_enthalpies.py`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/fetch/references/atomic_enthalpies.py)).
Cleaner than scraping a second form-only page.

## Why this is part of vibe-qc

vibe-qc's molecular validation strategy uses two reference
sources:

1. **Software-vs-software parity**, vibe-qc vs PySCF / ORCA
   subprocesses to machine precision (see
   [`external_codes.md`](external_codes.md)). Catches
   regressions, validates numerical stability.
2. **Software-vs-experiment**, vibe-qc vs NIST CCCBDB on a
   well-defined small-molecule set. Catches *physical*
   wrongness that software-vs-software parity can't (e.g. a
   functional that gives the wrong atomization trend even
   though both vibe-qc and PySCF agree on the wrong number).

`vqfetch reference` makes (2) accessible without manually
maintaining a curated reference table.

## Deferred to the v0.8.x maintenance line

* **NIST Chemistry WebBook fallback** (`vqfetch reference
  --source webbook`), same parse-once / cache-30-days pattern
  for molecules CCCBDB doesn't cover.
* **ATcT** (Argonne Active Thermochemical Tables), network-
  corrected thermochemistry with explicit error bars and
  dependency graphs. Schema headroom is in place
  (`ReferenceKind = "evaluated"`); no client yet.
* **Bulk-sweep tooling** (`vqfetch reference --bulk
  <cas-list.txt>`) for pre-populating the cache.

Tracked in `docs/roadmap.md` under the v0.8.x maintenance
window.

## See also

* [`external_structures.md`](external_structures.md),
  vqfetch's structure subcommands (OPTIMADE / MP / COD /
  NOMAD).
* [`external_codes.md`](external_codes.md), vibe-qc's
  policy on external programs vs vendored libraries; how
  parity runs against PySCF / ORCA / CRYSTAL are wired.
* {ref}`external-data licensing <external-data-licensing>`
  - full per-source licensing inventory.
* [CCCBDB homepage](https://cccbdb.nist.gov/), browse
  records online.
