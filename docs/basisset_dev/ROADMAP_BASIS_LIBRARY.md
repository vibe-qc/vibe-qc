# Basis-set library roadmap

**Reconciled against current development: 2026-10-19.** The original
[2026-05-08 review](REVIEW_BASIS_SETS_2026-05-08.md) is historical. The
pre-split `basissetdev` branch is not a pending integration dependency.

This roadmap tracks library coverage, not the separately versioned optimizer
in [vibe-basis](https://github.com/vibe-qc/vibe-qc/blob/main/vibe-basis/ROADMAP.md). A bundled numerical record, an
implemented Hamiltonian and scientific acceptance are three different facts.

## GitLab milestone and weekly reconciliation

The development project's **Milestones** page tracks this work as
**QC basis-library authoritative store** (project milestone ID 8).
The existing umbrella #295 and its focused store/provenance/cache/import issues
remain the implementation and acceptance records. Missing library families are
separate additions, not an unlimited release-completion requirement.

Review library coverage and tracker acceptance weekly against actual QVF
records, generated renderings, resolver/import source and accepted checks.
Use the [main roadmap's weekly review policy](../roadmap.md#gitlab-milestone-tracking-and-weekly-review):
implementation, qualification, release readiness and tagged delivery are
different states. Advance QC's feature target when the selected criteria pass;
the library has no invented independent version line. The optimizer's separate
release milestones stay in [vibe-basis](https://github.com/vibe-qc/vibe-qc/blob/main/vibe-basis/ROADMAP.md).

## Current implementation and priorities

- QVF JSON records in `python/vibeqc/basis_library/qvf/` are authoritative;
  `.g94` and `.ecp` files are generated renderings. The
  [library guide](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/basis_library/README.md) documents
  conversion, provenance, citation routing and rendering checks.
- Molecular on-demand BSE lookup ships through the optional `[bse]` extra
  and `run_job(fetch_from_bse=True)`. It reads the installed BSE catalogue;
  it is not a missing HTTP downloader. Periodic `run_periodic_job` does not
  expose that keyword.
- ECP integration, the pc/pcseg families, diffuse Pople variants, modified
  3c bases, vDZP, LANL and the listed PP orbital families exist. They are
  not waiting on libecpint or on a new branch merge.
- The named user basis store, provenance records and resolver are implemented.
  Remaining tracker acceptance must be distinguished from missing source.
  Do not edit rendered files as the primary data or add a second store.
- Remaining work is targeted missing coverage, per-element provenance/parity,
  relativistic Hamiltonians for the all-electron relativistic families, and
  auxiliary-basis research. Automatic Stoychev-style AutoAux remains a
  proposal; a citation entry or a registry-selected fitting basis is not its
  implementation.

## Cross-cutting prerequisites

| Prerequisite | Current state | Remaining limitation |
|---|---|---|
| ECP integrals | Implemented through libecpint; basis/ECP records are loaded together | Per-element accuracy and route coverage, not initial integration |
| Scalar-relativistic X2C / DKH2 / ZORA | Future method work | Meaningful use of the corresponding all-electron recontracted sets |
| GPW/GAPW | Gaussian plus auxiliary plane-wave routes exist; GAPW remains experimental | MOLOPT/GTH pairing and route qualification are separate from merely adding a basis file |
| Pure-PW / PAW / ONCV / NAO | Not the current Gaussian library architecture | No implied commitment to import incompatible library families |
| NMR shielding / J coupling | Future property work | A bundled pcS/pcJ basis does not implement the observable |
| AutoAux and bespoke pob JK fits | Research / missing generation workflow | Published fit data, generation algorithm and numerical qualification |

The table below preserves the review's priority recommendations. ✅ means a
matching bundled orbital record is present, not full element coverage or a
validated method; ❌ means no matching bundled record was found; 🚫 identifies
a missing method or research prerequisite. Optional BSE catalogue availability
is separate from bundled coverage. Historical counts of "45 missing" are
removed because aliases and later additions made them unreliable.

## The big table

Everything the review recommends, organised by family. "BSE name"
is the catalogue identifier, not necessarily the bundled file stem. Use the
current BSE loader or basis conversion tools rather than the old proposed
fetch script.

### Pople family

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| STO-3G | `STO-3G` | ✅ | low | Pedagogy / SCF guess. UI label: "for guess only". |
| STO-6G | `STO-6G` | ✅ | low | Same role as STO-3G; slightly better atomic energies. |
| MINI | `MINI (Huzinaga)` | ✅ | low | Niche. |
| 3-21G | `3-21G` | ✅ | low | UI label: "unpolarised, avoid for production". |
| 6-31G | `6-31G` | ✅ | low | Same UI label as 3-21G. |
| 6-31G(d) | `6-31G*` | ✅ | medium | Legacy default; keep. |
| 6-31G(d,p) | `6-31G**` | ✅ | medium | Same. |
| **6-31+G(d,p)** | `6-31+G**` | ✅ | **high** | Diffuse-aug; `6-31+g**.g94` shipped. |
| **6-31++G(d,p)** | `6-31++G**` | ✅ | **high** | Best DZ basis per Pitman 2024; `6-31++g**.g94` shipped. |
| 6-311G(d,p) | `6-311G**` | ✅ | medium | UI label: "polarised 6-311G is poorly parameterised, prefer pcseg-2 / def2-TZVP" (Pitman 2024). |
| **6-311+G(d,p)** | `6-311+G**` | ✅ | medium | Diffuse variant; `6-311+g**.g94` shipped. |
| **6-311+G(2d,p)** | `6-311+G(2d,p)` | ✅ | low | `6-311+g(2d,p).g94` shipped. |
| **6-311+G(2df,p)** | `6-311+G(2df,p)` | ❌ | low | Same, still missing. |

### Karlsruhe def2 family

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| def2-SV | `def2-SV` | ✅ | medium | |
| def2-SV(P) | `def2-SV(P)` | ✅ | medium | |
| def2-SVP | `def2-SVP` | ✅ | high | Default DZ. |
| def2-SVPD | `def2-SVPD` | ✅ | medium | |
| def2-TZVP | `def2-TZVP` | ✅ | **highest** | Production default. |
| def2-TZVPD | `def2-TZVPD` | ✅ | high | |
| def2-TZVPP | `def2-TZVPP` | ✅ | high | |
| def2-TZVPPD | `def2-TZVPPD` | ✅ | high | Inorganic main-group default. |
| def2-QZVP | `def2-QZVP` | ✅ | high | Benchmark DFT. |
| def2-QZVPP | `def2-QZVPP` | ✅ | high | |
| def2-QZVPPD | `def2-QZVPPD` | ✅ | medium | |
| **ma-def2-TZVP** | `ma-def2-TZVP` | ❌ | high | Truhlar minimally-augmented. |
| **ma-def2-TZVPP** | `ma-def2-TZVPP` | ❌ | medium | |
| **ma-def2-SVP** | `ma-def2-SVP` | ❌ | medium | |
| def2-universal-J | (built into def2-J) | ✅ | high | aux-fit, Coulomb |
| def2-universal-JK | (built into def2-JK) | ✅ | high | aux-fit, Coulomb+exchange |

### Karlsruhe def2, relativistic (blocked on rel. Hamiltonian)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **x2c-SVPall** | `x2c-SVPall` | 🚫 | high | Pollak-Weigend 2017. Blocked on X2C / DKH. |
| **x2c-TZVPall** | `x2c-TZVPall` | ✅ | high | Same. |
| **x2c-QZVPall** | `x2c-QZVPall` | 🚫 | high | Franzke 2020. |
| **x2c-TZVPall-s** | `x2c-TZVPall-s` | ✅ | medium | NMR-tuned. |
| **dhf-SVP** | `dhf-SVP` | ✅ | medium | Dirac-Hartree-Fock-derived def2-equivalent; `dhf-svp.g94`/`.ecp` shipped. |
| **dhf-TZVP** | `dhf-TZVP` | ✅ | medium | `dhf-tzvp.g94`/`.ecp` shipped. |
| **dhf-QZVP** | `dhf-QZVP` | ✅ | low | `dhf-qzvp.g94`/`.ecp` shipped. |
| **def2-TZVP-DKH** | `def2-TZVP-DKH` | 🚫 | medium | DKH-recontracted. |

### Karlsruhe def2, modified for 3c (bundled)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **def2-mSVP** | `def2-mSVP` | ✅ | high | PBEh-3c, B3LYP-3c carrier. |
| **def2-mTZVP** | `def2-mTZVP` | ✅ | high | B97-3c carrier. |
| **def2-mTZVPP** | `def2-mTZVPP` | ✅ | high | r²SCAN-3c carrier. |

### Dunning correlation-consistent

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| cc-pVDZ | `cc-pVDZ` | ✅ | high | |
| cc-pVTZ | `cc-pVTZ` | ✅ | high | Post-HF default. |
| cc-pVQZ | `cc-pVQZ` | ✅ | high | CBS extrapolation. |
| cc-pV5Z | `cc-pV5Z` | ✅ | medium | |
| cc-pV6Z | `cc-pV6Z` | ✅ | low | |
| cc-pV7Z | `cc-pV7Z` | ✅ | low | |
| aug-cc-pVDZ | `aug-cc-pVDZ` | ✅ | high | |
| aug-cc-pVTZ | `aug-cc-pVTZ` | ✅ | high | Anions, response. |
| aug-cc-pVQZ | `aug-cc-pVQZ` | ✅ | medium | |
| aug-cc-pV5Z | `aug-cc-pV5Z` | ✅ | medium | |
| **cc-pV(D+d)Z** | `cc-pV(D+d)Z` | ✅ | medium | "tight d" for hypervalent S/P. |
| **cc-pV(T+d)Z** | `cc-pV(T+d)Z` | ✅ | medium | |
| **cc-pV(Q+d)Z** | `cc-pV(Q+d)Z` | ✅ | medium | |
| **aug-cc-pV(T+d)Z** | `aug-cc-pV(T+d)Z` | ✅ | medium | |
| **cc-pCVDZ** | `cc-pCVDZ` | ✅ | medium | Core-valence. |
| **cc-pCVTZ** | `cc-pCVTZ` | ✅ | medium | Core-valence. |
| **cc-pCVQZ** | `cc-pCVQZ` | ✅ | medium | Core-valence. |
| **aug-cc-pCVTZ** | `aug-cc-pCVTZ` | ✅ | medium | Core-valence. |
| cc-pwCVTZ | `cc-pwCVTZ` | ✅ | low | Weighted core-valence. |
| cc-pVTZ-F12 | `cc-pVTZ-F12` | ✅ | medium | Explicit-correlation. |
| cc-pVQZ-F12 | `cc-pVQZ-F12` | ✅ | medium | |

### Dunning pseudopotential-adapted (Peterson/Figgen small-core PPs)

Shipped 2026-09-08. Before that the library carried **sixteen** `-PP-RIFIT`
fitting sets and **zero** PP orbital sets: auxiliary bases for orbital bases
the registry would refuse to load. Each orbital set covers 39 elements
(Cu-Kr, Y-Xe, Hf-Rn) and carries its ECP blocks in the same BSE file;
`scripts/setup_basis_library.sh` splits them into a `.g94` + `.ecp` pair.

Note the coverage asymmetry that remains, by design: the RI fits are narrower
than the orbital sets (15 elements for `cc-pV*Z-PP-RIFIT`, 21 for
`cc-pwCV*Z-PP-RIFIT`, 8 for `aug-cc-pwCV*Z-PP-RIFIT`). An RI job on an element
the fit omits is refused by the aux coverage guard naming the element (#480).
No JK fit is published for the family at all.

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| cc-pVDZ-PP | `cc-pVDZ-PP` | ✅ | high | Small-core PP; heavy-element orbital set. |
| cc-pVTZ-PP | `cc-pVTZ-PP` | ✅ | high | |
| cc-pVQZ-PP | `cc-pVQZ-PP` | ✅ | medium | |
| cc-pV5Z-PP | `cc-pV5Z-PP` | ✅ | low | |
| cc-pwCVDZ-PP | `cc-pwCVDZ-PP` | ✅ | medium | PP core-valence. |
| cc-pwCVTZ-PP | `cc-pwCVTZ-PP` | ✅ | medium | |
| cc-pwCVQZ-PP | `cc-pwCVQZ-PP` | ✅ | low | |
| cc-pwCV5Z-PP | `cc-pwCV5Z-PP` | ✅ | low | |
| aug-cc-pVDZ-PP | `aug-cc-pVDZ-PP` | ✅ | high | PP + diffuse. |
| aug-cc-pVTZ-PP | `aug-cc-pVTZ-PP` | ✅ | high | |
| aug-cc-pVQZ-PP | `aug-cc-pVQZ-PP` | ✅ | medium | |
| aug-cc-pV5Z-PP | `aug-cc-pV5Z-PP` | ✅ | low | |
| aug-cc-pwCVDZ-PP | `aug-cc-pwCVDZ-PP` | ✅ | medium | |
| aug-cc-pwCVTZ-PP | `aug-cc-pwCVTZ-PP` | ✅ | medium | |
| aug-cc-pwCVQZ-PP | `aug-cc-pwCVQZ-PP` | ✅ | low | |
| aug-cc-pwCV5Z-PP | `aug-cc-pwCV5Z-PP` | ✅ | low | |

### Jensen polarization-consistent (segmented + general)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **pc-0** | `pc-0` | ✅ | medium | Unpol DZ. |
| **pc-1** | `pc-1` | ✅ | medium | DZ-pol. |
| **pc-2** | `pc-2` | ✅ | medium | TZ-pol. |
| **pc-3** | `pc-3` | ✅ | low | QZ-pol. |
| **pc-4** | `pc-4` | ✅ | low | 5Z-pol. |
| **pcseg-0** | `pcseg-0` | ✅ | high | Segmented contraction. |
| **pcseg-1** | `pcseg-1` | ✅ | **highest** | DFT-optimal DZ. |
| **pcseg-2** | `pcseg-2` | ✅ | **highest** | DFT-optimal TZ, best on diet-GMTKN55 (Pitman 2024). |
| **pcseg-3** | `pcseg-3` | ✅ | medium | |
| **pcseg-4** | `pcseg-4` | ✅ | low | |
| **aug-pcseg-1** | `aug-pcseg-1` | ✅ | high | |
| **aug-pcseg-2** | `aug-pcseg-2` | ✅ | high | TDDFT, anions. |
| **pcS-0…3** | `pcS-{0,1,2,3}` | ✅ | medium | pcs-0 through pcs-3 data bundled; NMR shielding kernel remains open. |
| **pcJ-0…3** | `pcJ-{0,1,2,3}` | 🚫 | medium | NMR J-coupling. |

### vDZP (Grimme 2023, bundled)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **vDZP** | `vDZP` | ✅ | high | ωB97X-3c carrier; uses ECPs; `vdzp.g94`/`.ecp` shipped. |

### ANO families

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| ANO-RCC (full) | `ANO-RCC` | ✅ | high | Generally contracted. |
| ANO-RCC-MB | `ANO-RCC-MB` | ✅ | low | Minimal. |
| **ANO-RCC-VDZP** | `ANO-RCC-VDZP` | ✅ | medium | Specific contraction. |
| **ANO-RCC-VTZP** | `ANO-RCC-VTZP` | ✅ | high | Multireference TM standard. |
| **ANO-RCC-VQZP** | `ANO-RCC-VQZP` | ✅ | medium | |
| **ANO-R0** | `ANO-R0` | ✅ | medium | Zobel-Widmark-Veryazov 2020. |
| **ANO-R1** | `ANO-R1` | ✅ | medium | |
| **ANO-R2** | `ANO-R2` | ✅ | medium | |
| **ANO-R3** | `ANO-R3` | ✅ | low | |

### LANL ECP (deprecated by review, ship for compatibility)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **LANL2DZ** | `LANL2DZ` | ✅ | low | UI label: "deprecated, use def2-TZVP + def2-ECP". ECP integration is available. |

### Periodic Gaussian

| Name | Source | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| pob-TZVP | Bredow tarball | ✅ | **highest** | Verified byte-clean f059b37. |
| pob-DZVP-rev2 | Bredow scrape | ✅ | high | Verified. |
| pob-TZVP-rev2 | Bredow tarball | ✅ | **highest** | Verified. S d-shell bug fixed. |
| **pob-DZVP** (original) | Bredow archive | ❌ | medium | Predecessor of rev2; for reproducibility. |
| **pob-DZVPP** | Bredow archive | ❌ | medium | Polar augmentation. |
| **pob-TZVPP** | Bredow archive | ❌ | medium | Polar augmentation. |
| **pob-TZVP-rev2 (Rb-I, fifth period)** | Bredow tarball | ✅ | high | ECP-backed fifth-period records are present; precision/parity remains per element. |
| **pob-TZVP-jk** | (to design) | 🚫 | high | Goal 5 of basissetdev. |
| **MOLOPT-SZV** | CP2K source | 🚫 | medium | Pairing and route qualification remain; GPW infrastructure exists. |
| **MOLOPT-DZVP** | CP2K source | 🚫 | medium | |
| **MOLOPT-TZVP** | CP2K source | 🚫 | medium | |
| **MOLOPT-TZV2P** | CP2K source | 🚫 | medium | |
| **dcm-TZVP** | Daga/Civalleri/Maschio 2020 | ❌ | low | System-specific (diamond, graphene, carbyne). |

### Property-specific (data and property implementation are separate)

| Name | BSE id | Status | Priority | Blocks |
|------|--------|--------|----------|---------|
| **EPR-II** | `EPR-II` | 🚫 | medium | EPR g-tensor / hyperfine kernel |
| **EPR-III** | `EPR-III` | 🚫 | medium | Same |
| **Sadlej-pVTZ** | `Sadlej pVTZ` | ✅ | medium | Polarizability, kernel exists |
| **LPolX (Maroulis)** | (BSE varied) | ❌ | low | Same |
| **ccX-DK** | `ccX-DK` | ❌ | low | Core-level X-ray |

## Plane-wave / NAO / LAPW (out of scope today)

The review recommends VASP-PAW, PseudoDojo (ONCV), SSSP, GBRV,
SG15, JTH PAW, FHI-aims tier 1-4, SIESTA NAO, and (L)APW+lo.
**vibe-qc is a Gaussian / atom-centred-orbital code today**; none
of these libraries fit the current architecture. The roadmap
captures them under "deferred to a future periodic-PW chat":

* **PseudoDojo (ONCV)** is the recommended starting point if /
  when vibe-qc grows a plane-wave path. Open-source, well-curated,
  has scalar and fully-relativistic variants.
* **SSSP precision** is the natural complement for high-throughput
  materials work.
* **PAW** (VASP / GPAW / ABINIT JTH) is a substantial
  implementation lift via the projector formalism; defer until
  ONCV is stable.
* **FHI-aims tier-1/2 NAO** would require a numerical-orbital
  integral path entirely separate from libint.

## Inventory and maintenance

The [library README](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/basis_library/README.md), registry
and QVF records define the live inventory. Check both element coverage and
ECP pairing before using a family. Adding a record does not close a numerical
acceptance issue, and a pending acceptance issue does not make existing data
absent.

1. Select a genuinely missing record or coverage gap, checking the bundle and
   optional BSE catalogue first.
2. Convert/import into the existing QVF record system; preserve upstream
   identity, numerical content hash and per-element citations.
3. Render through `scripts/render_basis_library.py` or the supported setup
   script and check record/rendering parity. Keep ECP blocks with the correct
   orbital family and refuse uncovered elements.
4. Validate the affected numerical use before making new accuracy claims.
5. Keep auxiliary-basis design and production optimized-basis campaigns in
   their own milestones; they do not hold unrelated library additions.

## Cross-references

* [PLAN.md](PLAN.md), top-level basissetdev plan; this roadmap
  is rolled in as a new long-term goal slot.
* [REQUIREMENTS-PERIODIC.md](REQUIREMENTS-PERIODIC.md), historical design record
  to the periodic chat; the periodic-Gaussian items above
  partly depend on R1-R5 there.
* [GOAL4_DESIGN.md](GOAL4_DESIGN.md), basis-optimisation engine
  design; note: the optimiser will eventually generate basis
  sets that themselves go into this library (Goal 5: bespoke
  pob-*-jk; Goal 6: periodic metals basis).
