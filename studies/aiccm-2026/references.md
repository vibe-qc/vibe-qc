# CRYSTAL23 references for the AICCM-2026 test set

The external reference for every system is the CRYSTAL23 `.d12` in the
**QC input library** (`~/gitlab/qc-input-library`, a separate repo). Geometries in
[`testset.py`](testset.py) use the **same lattice constants** as these `.d12`
files (verified against the source), so the AICCM / bipole / gdf / CRYSTAL23
comparison is apples-to-apples.

Run a reference with CRYSTAL23:

```sh
cd ~/gitlab/qc-input-library/<crystal_ref_dir> && crystal < INPUT.d12 > out.out
```

| system | CRYSTAL23 `.d12` (under qc-input-library/) | level |
|---|---|---|
| h-chain | `crystal/h2-chain-1d/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| lih-chain | `crystal/lih-chain-1d/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| polyethylene | `crystal/polyethylene-1d/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| c-diamond | `crystal/c-diamond/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| si-diamond | `crystal/si-diamond/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| sic-zb | `crystal/sic-zincblende/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| bn-zb | `crystal/bn-zincblende/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| graphene | `crystal/graphene-2d/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 (SLAB) |
| mgo-slab | `crystal/mgo-001-slab/rks-pbe_tut/INPUT.d12` | PBE / 8-411G* (SLABCUT 001) |
| ice-ih | `crystal/ice-ih/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| co2-dryice | `crystal/co2-dryice/rks-pbe_sto-3g/INPUT.d12` | PBE / STO-3G |
| lih-rocksalt | `crystal/lih-rocksalt/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| lif-rocksalt | `crystal/lif-rocksalt/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| nacl-rocksalt | `crystal/nacl-rocksalt/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| mgo | `crystal/mgo-rocksalt/r2scan_seg/INPUT.d12` | r2SCAN / segmented (OPTGEOM) |
| tio2-rutile | `crystal/tio2-rutile/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| sio2-quartz | `crystal/sio2-alpha-quartz/pbe_pob-tzvp-rev2/INPUT.d12` | PBE / pob-TZVP-REV2 |
| nio-afm | `crystal/nio-rocksalt/uks-r2scan_pob-tzvp-rev2/INPUT.d12` | UKS r2SCAN / pob-TZVP-REV2 (AF2) |
| aln-wurtzite | `crystal/aln-wurtzite/b3lyp_pob-tzvp-rev2/INPUT.d12` | B3LYP / pob-TZVP-REV2 |
| black-phosphorus | `crystal/black-phosphorus/rhf_sto-3g/INPUT.d12` | RHF / STO-3G |
| caf2-fluorite | `crystal/caf2-fluorite/b3lyp_pob-tzvp-rev2/INPUT.d12` | B3LYP / pob-TZVP-REV2 |
| al2o3-corundum | `crystal/al2o3-corundum/b3lyp_pob-tzvp-rev2/INPUT.d12` | B3LYP / pob-TZVP-REV2 |
| alcl3 | `crystal/alcl3/pw1pw_pob-tzvp-rev2/INPUT.d12` | PW1PW / pob-TZVP-REV2 |
| cscl | `crystal/cscl/r2scan_pob-tzvp-rev2/INPUT.d12` | r2SCAN / pob-TZVP-REV2 |
| boric-acid | `crystal/boric-acid/rhf_sto-3g/INPUT.d12` | RHF / STO-3G |
| na-bcc | `crystal/na-bcc/r2scan_pob-tzvp-rev2/INPUT.d12` | r2SCAN / pob-TZVP-REV2 |
| tio2-anatase | *literature* (not in library) | Horn, Schwerdtfeger & Meagher, Z. Kristallogr. **136**, 273 (1972) |

### Bravais-lattice-coverage geometries (`crystal_specs.json`)

The expanded systems are built by `testset._crystal` from
[`crystal_specs.json`](crystal_specs.json) — `{space group, cell parameters,
asymmetric unit}` extracted from the CRYSTAL `.d12` above (and, for `tio2-anatase`,
the literature value), expanded to the primitive cell by ASE's
`spacegroup.crystal`. Grounded (every space group round-trips through spglib) and
portable (no library access at runtime). Adding a further library crystal = one
`crystal_specs.json` entry + one `SYSTEMS` row.

## Caveats (from the geometry extraction)

- **Level of theory:** the references are PBE / pob-TZVP-REV2 (a few r2SCAN /
  STO-3G). The AICCM four-center cross-check is run at **STO-3G** (so the dense
  four-center is tractable); to compare *absolute* energies against CRYSTAL23, run
  the **bipole / gdf** routes at `--basis pob-tzvp-rev2` (same basis as the `.d12`).
- **nio-afm:** the CRYSTAL23 reference is the **AF2 (type-II) antiferromagnetic**
  cell — a 2-atom rocksalt + `SUPERCEL [[0,1,1],[1,0,1],[1,1,0]]` with `ATOMSPIN`
  setting the two Ni sites to +1/−1 (`SPINLOCK 0 -6`). The `testset.py` geometry
  is the chemical primitive cell; reproducing the AFM number needs the matching
  spin pattern (open-shell UHF/UKS-CCM with the AF supercell + per-atom spins).
  Treat nio-afm as the open-shell demonstrator, not yet a quantitative AFM match.
- **hbn:** in the library `hbn-hexagonal` is **3-D bulk** h-BN (AA' stacked), not a
  2-D monolayer; `graphene-2d` and `mgo-001-slab` are genuine 2-D.
- **No stored reference energies** were available in the library outputs (the
  `.out` files are mostly absent / not converged) except NiO; regenerate the
  CRYSTAL23 totals with the harness below.

## Generating the CRYSTAL23 references (via vq)

CRYSTAL23 is an external program — run **out of process** (CLAUDE.md §10): we
never import it; we spawn it on its own `.d12` and parse the output.
[`make_crystal_jobs.py`](make_crystal_jobs.py) stages one self-contained payload
per system (its library `INPUT.d12` + [`crun.sh`](crun.sh)) and emits the vq
lines; [`crun.sh`](crun.sh) runs a **single point at the test-set geometry**
(any `OPTGEOM` block is stripped, so the reference matches the AICCM geometry
exactly) and writes `<system>__crystal23.json` (`energy_ha`, `converged`).

```sh
python make_crystal_jobs.py            # review the batch (stages payloads)
python make_crystal_jobs.py | sh       # launch on vq (needs CRYSTAL23 on the host)
# then: vq fetch <jobid>  →  feed the JSON into compare.py alongside the AICCM runs
```

Two caveats for the paper:

- **Level uniformity.** The library `.d12` levels differ per system (the table
  above: PBE / B3LYP / RHF / r2SCAN / PW1PW). For a *uniform*-level reference set
  (e.g. all PBE/pob-TZVP-REV2, matching the `gdf --basis pob-tzvp-rev2` periodic
  refs), regenerate the method block before submitting. The harness runs whatever
  level the `.d12` carries.
- **`tio2-anatase` has no library `.d12`** (literature geometry) — `make_crystal_jobs.py`
  lists it as SKIPPED. Build its `.d12` from the anatase geometry + the
  `tio2-rutile` pob-TZVP-REV2 Ti/O basis blocks (same elements) to add it.

## Comparison columns for the paper

Per system, tabulate per-atom energies: `AICCM-4c` (STO-3G), `AICCM-RI`,
`AICCM-RIJCOSX`, `bipole`, `gdf`, `CRYSTAL23` — plus the AICCM(nrep)→periodic and
k-mesh convergence. See [`README.md`](README.md) § Comparison protocol.
