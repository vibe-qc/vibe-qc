# Basis-set library roadmap

**Branch:** `basissetdev`. **Source review:** [REVIEW_BASIS_SETS_2026-05-08.md](REVIEW_BASIS_SETS_2026-05-08.md).

This document translates the 2026-05-08 review into a concrete
roadmap for the `python/vibeqc/basis_library/` bundle. Each row
below is one canonical basis-set name and gives:

* **Status** as of audit on `0ac5fb9`:
  * ✅ **shipped**, present under `basis/`
  * 🟡 **partial**, present but with gaps (e.g. only some elements)
  * ❌ **missing**, not bundled yet
  * 🚫 **blocked**, bundling depends on a feature that has not landed
* **Priority** matching the review's tiered recommendation
* **Source** the canonical place to fetch the numbers from
* **Blockers** any vibe-qc capability gap that must close first

## Status snapshot

* **142 basis-set files** were bundled at the original audit on `0ac5fb9`;
  **255 `.g94` files** are bundled today (2026-09-08) -- pcseg, dhf, vDZP
  and the 16 cc-pVnZ-PP orbital sets have since shipped (see the priorities
  list and table below); this snapshot paragraph itself was not kept current
  as those landed.
* **45 review-recommended bases missing** as listed below (this count
  has not been recomputed against current shipped status).
* Three pob bases are byte-clean against the upstream Bredow-group
  source after the Goal-1 audit (commit `f059b37`).
* The x2c and pob-aux blocks remain the largest outright gaps; pcseg,
  dhf, and vDZP are no longer gaps (shipped).

## Top-of-stack priorities (Q3 2026) -- items 1-4 shipped, verified against `python/vibeqc/basis_library/basis/`

1. ✅ **pcseg-n family** (pcseg-{0..4} and pcseg-s-{0..2} shipped; `aug-pcseg-n` not separately checked here).
2. ✅ **Pople diffuse variants** (6-31+G(d,p), 6-31++G(d,p), 6-311+G(d,p), 6-311+G(2d,p) all shipped; 6-311+G(2df,p) still missing).
3. ✅ **vDZP** shipped (`vdzp.g94`/`.ecp`).
4. ✅ **3c modified bases** (def2-mSVP, def2-mTZVP, def2-mTZVPP) shipped.
5. **Relativistic def2 family** (x2c-{S,TZ,QZ}VPall, dhf-XVPP,
   def2-TZVP-DKH) remains blocked on a relativistic Hamiltonian
   implementation (X2C / DKH2 / ZORA), out of scope for
   `basissetdev`; flag as a feature request for the molecular
   chat. (Note: plain dhf-{SVP,TZVP,QZVP} basis *data* has shipped --
   see the table below -- this item is specifically about the
   relativistic Hamiltonian needed to use them meaningfully.)
6. **pob auxiliary basis (pob-*-jk)**. Hand-tuned aux for periodic
   density fitting. Already in Goal 5 of the basissetdev plan;
   AutoAux owned by the DF dev chat, see
   `feedback_aux_basis_routing.md` (a private dev-memory note).

## Cross-cutting prerequisites

Several review recommendations are blocked on infrastructure that
the molecular / periodic dev chats own:

| Prerequisite | Blocks | Owner chat |
|---|---|---|
| **ECP integral support (libecpint)** | LANL2DZ, vDZP, dhf-{S,TZ,QZ}VPP, x2c-{S,TZ,QZ}VPall, pob-rev2 fifth-period | molecular DF / SCF chat |
| **Relativistic Hamiltonian (X2C / DKH2 / ZORA)** | x2c-XVPall, dhf-XVPP, def2-TZVP-DKH meaningful use | molecular relativity chat (not yet spawned) |
| **CP2K-style GPW / mixed Gaussian-PW basis** | MOLOPT family | periodic chat (deferred) |
| **PAW / ONCV pseudopotential infrastructure** | PseudoDojo, SSSP, VASP-PAW | periodic chat (deferred, vibe-qc is Gaussian) |
| **NMR shielding / J-coupling kernels** | pcS-n, pcJ-n meaningful use | properties chat (not yet spawned) |
| **AutoAux (Stoychev) implementation** | "auto-fit aux for any orbital basis" | density-fitting chat |

## The big table

Everything the review recommends, organised by family. "BSE name"
is the canonical Basis Set Exchange identifier; the fetch script
([`scripts/basisset_dev/fetch_from_bse.py`](../../scripts/basisset_dev/fetch_from_bse.py),
to be added) takes that as input.

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
| **x2c-TZVPall** | `x2c-TZVPall` | 🚫 | high | Same. |
| **x2c-QZVPall** | `x2c-QZVPall` | 🚫 | high | Franzke 2020. |
| **x2c-TZVPall-s** | `x2c-TZVPall-s` | 🚫 | medium | NMR-tuned. |
| **dhf-SVP** | `dhf-SVP` | ✅ | medium | Dirac-Hartree-Fock-derived def2-equivalent; `dhf-svp.g94`/`.ecp` shipped. |
| **dhf-TZVP** | `dhf-TZVP` | ✅ | medium | `dhf-tzvp.g94`/`.ecp` shipped. |
| **dhf-QZVP** | `dhf-QZVP` | ✅ | low | `dhf-qzvp.g94`/`.ecp` shipped. |
| **def2-TZVP-DKH** | `def2-TZVP-DKH` | 🚫 | medium | DKH-recontracted. |

### Karlsruhe def2, modified for 3c (blocked on gCP + D3/D4)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **def2-mSVP** | `def2-mSVP` | 🚫 | high | PBEh-3c, B3LYP-3c carrier. |
| **def2-mTZVP** | `def2-mTZVP` | 🚫 | high | B97-3c carrier. |
| **def2-mTZVPP** | `def2-mTZVPP` | 🚫 | high | r²SCAN-3c carrier. |

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
| **cc-pV(D+d)Z** | `cc-pV(D+d)Z` | ❌ | medium | "tight d" for hypervalent S/P. |
| **cc-pV(T+d)Z** | `cc-pV(T+d)Z` | ❌ | medium | |
| **cc-pV(Q+d)Z** | `cc-pV(Q+d)Z` | ❌ | medium | |
| **aug-cc-pV(T+d)Z** | `aug-cc-pV(T+d)Z` | ❌ | medium | |
| **cc-pCVDZ** | `cc-pCVDZ` | ❌ | medium | Core-valence. |
| **cc-pCVTZ** | `cc-pCVTZ` | ❌ | medium | Core-valence. |
| **cc-pCVQZ** | `cc-pCVQZ` | ❌ | medium | Core-valence. |
| **aug-cc-pCVTZ** | `aug-cc-pCVTZ` | ❌ | medium | Core-valence. |
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
| **pc-0** | `pc-0` | ❌ | medium | Unpol DZ. |
| **pc-1** | `pc-1` | ❌ | medium | DZ-pol. |
| **pc-2** | `pc-2` | ❌ | medium | TZ-pol. |
| **pc-3** | `pc-3` | ❌ | low | QZ-pol. |
| **pc-4** | `pc-4` | ❌ | low | 5Z-pol. |
| **pcseg-0** | `pcseg-0` | ❌ | high | Segmented contraction. |
| **pcseg-1** | `pcseg-1` | ❌ | **highest** | DFT-optimal DZ. |
| **pcseg-2** | `pcseg-2` | ❌ | **highest** | DFT-optimal TZ, best on diet-GMTKN55 (Pitman 2024). |
| **pcseg-3** | `pcseg-3` | ❌ | medium | |
| **pcseg-4** | `pcseg-4` | ❌ | low | |
| **aug-pcseg-1** | `aug-pcseg-1` | ❌ | high | |
| **aug-pcseg-2** | `aug-pcseg-2` | ❌ | high | TDDFT, anions. |
| **pcS-0…3** | `pcS-{0,1,2,3}` | 🚫 | medium | NMR shielding (blocked on NMR kernel). |
| **pcJ-0…3** | `pcJ-{0,1,2,3}` | 🚫 | medium | NMR J-coupling. |

### vDZP (Grimme 2023, blocked on libecpint)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **vDZP** | `vDZP` | ✅ | high | ωB97X-3c carrier; uses ECPs; `vdzp.g94`/`.ecp` shipped. |

### ANO families

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| ANO-RCC (full) | `ANO-RCC` | ✅ | high | Generally contracted. |
| ANO-RCC-MB | `ANO-RCC-MB` | ✅ | low | Minimal. |
| **ANO-RCC-VDZP** | `ANO-RCC-VDZP` | ❌ | medium | Specific contraction. |
| **ANO-RCC-VTZP** | `ANO-RCC-VTZP` | ❌ | high | Multireference TM standard. |
| **ANO-RCC-VQZP** | `ANO-RCC-VQZP` | ❌ | medium | |
| **ANO-R0** | `ANO-R0` | ❌ | medium | Zobel-Widmark-Veryazov 2020. |
| **ANO-R1** | `ANO-R1` | ❌ | medium | |
| **ANO-R2** | `ANO-R2` | ❌ | medium | |
| **ANO-R3** | `ANO-R3` | ❌ | low | |

### LANL ECP (deprecated by review, ship for compatibility)

| Name | BSE id | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| **LANL2DZ** | `LANL2DZ` | 🚫 | low | UI label: "deprecated, use def2-TZVP + def2-ECP". Blocked on libecpint. |

### Periodic Gaussian

| Name | Source | Status | Priority | Notes |
|------|--------|--------|----------|-------|
| pob-TZVP | Bredow tarball | ✅ | **highest** | Verified byte-clean f059b37. |
| pob-DZVP-rev2 | Bredow scrape | ✅ | high | Verified. |
| pob-TZVP-rev2 | Bredow tarball | ✅ | **highest** | Verified. S d-shell bug fixed. |
| **pob-DZVP** (original) | Bredow archive | ❌ | medium | Predecessor of rev2; for reproducibility. |
| **pob-DZVPP** | Bredow archive | ❌ | medium | Polar augmentation. |
| **pob-TZVPP** | Bredow archive | ❌ | medium | Polar augmentation. |
| **pob-TZVP-rev2 (Rb-I, fifth period)** | Bredow tarball | 🚫 | high | Blocked on libecpint (5th period uses ECPs). |
| **pob-TZVP-jk** | (to design) | 🚫 | high | Goal 5 of basissetdev. |
| **MOLOPT-SZV** | CP2K source | 🚫 | medium | Blocked on GPW infrastructure. |
| **MOLOPT-DZVP** | CP2K source | 🚫 | medium | |
| **MOLOPT-TZVP** | CP2K source | 🚫 | medium | |
| **MOLOPT-TZV2P** | CP2K source | 🚫 | medium | |
| **dcm-TZVP** | Daga/Civalleri/Maschio 2020 | ❌ | low | System-specific (diamond, graphene, carbyne). |

### Property-specific (blocked on property kernels)

| Name | BSE id | Status | Priority | Blocks |
|------|--------|--------|----------|---------|
| **EPR-II** | `EPR-II` | 🚫 | medium | EPR g-tensor / hyperfine kernel |
| **EPR-III** | `EPR-III` | 🚫 | medium | Same |
| **Sadlej-pVTZ** | `Sadlej pVTZ` | ❌ | medium | Polarizability, kernel exists |
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

## Inventory dashboard

A live summary belongs in [`python/vibeqc/basis_library/README.md`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/basis_library/README.md).
The current README inventory predates this roadmap; updating it
to flag missing-and-prioritised items is part of the next
`basissetdev` deliverable. The dashboard format I propose:

```
def2-TZVP        ✅  shipped
pcseg-2          ❌  missing — high priority — fetch from BSE
ma-def2-TZVP     ❌  missing — high priority — fetch from BSE
x2c-TZVPall      🚫  blocked on relativistic Hamiltonian
vDZP             🚫  blocked on libecpint integration
```

so users can see at a glance what they cannot use and why.

## Tooling: BSE fetcher

Most of the missing all-electron bases are one HTTP call away
from completion. Sketch (to land in
[`scripts/basisset_dev/fetch_from_bse.py`](../../scripts/basisset_dev/fetch_from_bse.py)):

```python
from basis_set_exchange import api
text = api.get_basis(name="pcseg-2", elements=None,
                     fmt="gaussian94", header=True)
Path(f"python/vibeqc/basis_library/custom/{name}.g94").write_text(text)
```

`pip install basis-set-exchange` ships a curated copy of the
BSE database; the call is offline-friendly. The fetcher will:

1. Read the "missing" list from this roadmap (parse the table).
2. Pull each via BSE.
3. Drop into `custom/` per the existing convention.
4. Run `scripts/setup_basis_library.sh` to copy `custom/` →
   `basis/`.
5. Add a regression-test that verifies every name resolves
   under `LIBINT_DATA_PATH` and is parseable by libint.

## Action items (ordered)

1. **Land BSE fetcher** + **commit pcseg-{0,1,2,3,4}** and **6-31++G(d,p)** + **6-31+G(d,p)** + **6-311+G(d,p)** + **ma-def2-{TZVP,SVP}** + **aug-pcseg-{1,2}**. Pure-Python additions; no infrastructure blockers.
2. **Update `basis_library/README.md`** with the dashboard view.
3. **File ECP-integral feature request** with the molecular DF / SCF chat (libecpint integration unlocks LANL2DZ + vDZP + dhf-XVP + x2c-XVPall + 5th-period pob).
4. **File relativistic-Hamiltonian feature request** with a future molecular-relativity chat (X2C / DKH2 / ZORA, unlocks the x2c / dhf families' meaningful use).
5. **3c composite methods** as a downstream goal once `vDZP` and `def2-m{S,TZ,TZP}VP` are in place AND gCP + D3/D4 are wired through the molecular runner.
6. **Goal 5 of basissetdev**: bespoke pob-*-jk auxiliary basis (separate paper; not blocked on the missing all-electron bases above).

## Cross-references

* [PLAN.md](PLAN.md), top-level basissetdev plan; this roadmap
  is rolled in as a new long-term goal slot.
* [REQUIREMENTS-PERIODIC.md](REQUIREMENTS-PERIODIC.md), handover
  to the periodic chat; the periodic-Gaussian items above
  partly depend on R1-R5 there.
* [GOAL4_DESIGN.md](GOAL4_DESIGN.md), basis-optimisation engine
  design; note: the optimiser will eventually generate basis
  sets that themselves go into this library (Goal 5: bespoke
  pob-*-jk; Goal 6: periodic metals basis).
