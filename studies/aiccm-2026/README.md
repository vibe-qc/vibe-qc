# AICCM-2026 — benchmark test set for the ab-initio Cyclic Cluster Model paper

A curated, runnable benchmark for the **ab-initio Cyclic Cluster Model (AICCM)**,
`aiccm2026dev-a` development line. It spans dimensionality (1-D chains, 2-D sheets/slabs,
3-D bulk), bonding (covalent, ionic, molecular, oxide), and shell (closed / open),
and pits the AICCM against three references on each system:

1. **Γ-CCM** (`aiccm2026dev-a`) - the union-and-weight/Wigner-Seitz
   integral-weighting construction. Its construction routes and the neutral
   fitted-torus controls are reported separately.
2. **vibe-qc `bipole`** — the production periodic Bipole (Ewald) J/K route.
3. **vibe-qc `gdf`** — the production periodic Gaussian-density-fitting route.
4. **CRYSTAL23** — the external reference (`.d12` inputs in the QC input library).

The BIPOLE/GDF periodic routes and CRYSTAL23 are external controls. Agreement
with a same-Hamiltonian supercell or k-mesh evaluation is meaningful evidence,
but route names alone do not prove that a control realizes the union-and-weight
Γ-CCM construction.

> **This is the `aiccm2026dev-a` line.** The distinct χ-CCM approach uses the
> finite-translation-group character construction under `aiccm2026dev-b` and a
> separate 3-D-only fleet. Keep the lines separate. No public `aiccm` alias and
> no route-name-based Γ/χ approach delta are defined.

## Site configuration

Host assignments and interpreter locations are external private configuration.
Names such as `compute-small` in examples and historical notes are role labels,
not configured targets.
See [site-configuration.md](site-configuration.md) before rendering remote jobs.
The shipped example profile uses placeholders and disables remote defaults.

## Systems

Geometries are **primitive cells** with the lattice constants of the matching
CRYSTAL23 `.d12` (so the comparison is apples-to-apples). See
[`references.md`](references.md) for the per-system CRYSTAL23 paths and
[`manifest.yaml`](manifest.yaml) for the machine-readable table.

| system | dim | class | atoms/cell | tier | 4c quantitative? |
|---|---|---|---|---|---|
| h-chain | 1 | molecular | 2 | A | yes |
| lih-chain | 1 | ionic-1d | 2 | A | no (ionic; a=3.2 Å) |
| polyethylene | 1 | molecular | 6 | A | yes |
| c-diamond | 3 | covalent | 2 | A | yes |
| bn-zb | 3 | covalent | 2 | A | yes |
| si-diamond | 3 | covalent | 2 | B | yes |
| sic-zb | 3 | covalent | 2 | B | yes |
| graphene | 2 | covalent | 2 | A | yes |
| mgo-slab | 2 | ionic | 2 | B | no (ionic) |
| ice-ih | 3 | molecular | 6 | B | yes |
| co2-dryice | 3 | molecular | 3 | B | yes |
| lih-rocksalt | 3 | ionic | 2 | A | no (Madelung) |
| lif-rocksalt | 3 | ionic | 2 | A | no (Madelung) |
| nacl-rocksalt | 3 | ionic | 2 | B | no (Madelung) |
| mgo | 3 | ionic | 2 | B | no (Madelung) |
| tio2-rutile | 3 | oxide | 6 | C | no |
| sio2-quartz | 3 | oxide | 9 | C | no |
| nio-afm | 3 | ionic-afm (open-shell) | 2 | C | no |
| aln-wurtzite | 3 | covalent (hP) | 4 | B | yes |
| black-phosphorus | 3 | covalent (oS) | 4 | B | yes |
| caf2-fluorite | 3 | ionic (cF) | 3 | B | no |
| tio2-anatase | 3 | oxide (tI) | 6 | C | no |
| al2o3-corundum | 3 | oxide (hR) | 10 | C | no |
| alcl3 | 3 | molecular (mS) | 8 | C | yes |
| cscl | 3 | ionic (cP) | 2 | C | no |
| boric-acid | 3 | molecular (aP) | 28 | C | yes |
| na-bcc | 3 | metal (cI) | 1 | C | no |

### Bravais-lattice coverage

The set spans **all seven crystal systems** and the cubic centerings, to exercise
the AICCM's lattice-generality (the symmetric four-center + spglib symmetry are
lattice-general — the reference cell is the Wigner–Seitz cell):

| Bravais | system | representative(s) |
|---|---|---|
| **aP** triclinic | triclinic | `boric-acid` (P-1) |
| **mS** monoclinic (C) | monoclinic | `alcl3` (C2/m) |
| **oS** orthorhombic (C) | orthorhombic | `black-phosphorus` (Cmce) |
| **tP** / **tI** tetragonal | tetragonal | `tio2-rutile` (P4₂/mnm) / `tio2-anatase` (I4₁/amd) |
| **hR** rhombohedral | trigonal | `al2o3-corundum` (R-3c), `sio2-quartz` (P3₂21) |
| **hP** hexagonal | hexagonal | `aln-wurtzite` (P6₃mc), `graphene`, `ice-ih` |
| **cP** / **cI** / **cF** cubic | cubic | `cscl` (Pm-3m) / `na-bcc` (Im-3m) / `c-diamond`, rocksalts, `caf2-fluorite` (Fm-3m) |

The Bravais-coverage crystals are built from grounded space-group specs
(`crystal_specs.json`, extracted from the CRYSTAL `.d12` sources and expanded by
ASE — needs the `[ase]` extra). Adding any further library crystal is a one-line
registry entry plus its spec. The remaining centerings (mP, oP, oI, oF) are
one-line additions via the same `_crystal` builder. Larger / heavy / metallic
cells (`boric-acid`, `al2o3-corundum`, `na-bcc`) are tier-C and primarily exercise
the geometry / symmetry / RI routes; `cscl` uses `pob-tzvp-rev2` (Cs has no STO-3G).

## Γ-CCM and its neutral fitted-torus controls (read this)

The WSSC four-center constructs Γ-CCM through union-and-weight integral
weighting. The GDF/RI route constructs a neutral fitted-torus control. Their
large ionic gap (for example LiH rocksalt RKS-PBE/atom: four-center -4.540
versus GDF -4.203 Ha) is a construction/operator gap; a leading
Madelung/monopole explanation remains a hypothesis to test, not permission to
substitute the control for Γ-CCM.

The historical `four_center_quantitative` flag remains useful for triage, but
it is not a construction identity. Quantitative Γ-CCM claims require
construction-specific convergence and external validation. A neutral-control
study must keep both SCF and correlation on its neutral operator and must be
reported as a control, not Γ-CCM. Cancellation in an energy difference must be
demonstrated rather than assumed.

**Sample result** (`results_demo_convergence_1d.json`, STO-3G, local —
demonstration only; paper numbers come from the compute-study batch). Cluster-size
convergence of AICCM-HF, nrep 4→14:

| system | class | E/atom n=4 | E/atom n=14 | |ΔE| (n=14 vs 12) |
|---|---|---|---|---|
| h-chain | molecular/covalent | −0.558357 | −0.558357 | **0.00 µHa** |
| lih-chain | ionic | −3.973566 | −3.964876 | **456 µHa** |

The covalent chain is converged at the smallest cluster; the ionic chain still
drifts at nrep=14. This is route-specific Γ-CCM convergence evidence. It does
not authorize substitution of the separately constructed neutral GDF control
or assign the drift to one Coulomb term.

## Running

Each case is `python run_case.py <system> <route>`. Routes:

| route | what runs |
|---|---|
| `aiccm-hf` | HF, four-center cyclic cluster (no RI) |
| `aiccm-ks` | KS, four-center cyclic cluster (no RI; `--functional`, default `pbe`) |
| `aiccm-ri` | neutral fitted-torus HF control (multi-k GDF); not Γ-CCM |
| `aiccm-ks-ri` | neutral fitted-torus KS control (`--functional`); not Γ-CCM |
| `aiccm-rijcosx` | HF with RIJCOSX (WSSC RI-J + chain-of-spheres K) |
| `aiccm-mp2` | MP2 correlation (UMP2 for open-shell) on the CCM MO integrals |
| `aiccm-ccsd` | CCSD(T) correlation (closed-shell) on the CCM MO integrals |
| `aiccm-dlpno-mp2` | **DLPNO-MP2** research correlation on the neutral fitted-torus reference (`--tcut-pno`, `--ke-cutoff`); JSON reports `e_correlation`, `n_pairs`, `e_pno_correction`. The harness requires a 3-D system. |
| `aiccm-dlpno-ccsd` | **DLPNO-CCSD(T)** (subspace-projected, union-PNO) on the neutral reference; JSON reports `e_ccsd_correlation`, `e_t`, `n_virtual_pno/full` |
| `aiccm-uccsd` | **open-shell UCCSD(T)** on the UHF-CCM neutral reference (Task D); JSON reports `n_alpha/beta`, `e_ccsd_correlation`, `e_t` |
| `aiccm-properties` | **Task C properties** — gap / Mulliken+Löwdin per-cell charges + transl. spread / finite-cluster dipole (`--gradient` adds the numerical force); **+ vibe-view** `_props_canonical.qvf` + `_props_density.cube`. Scalable SCF → runs on every tier |
| `aiccm-viz` | HF + **vibe-view visualization of the crystalline orbitals AND the localized (Wannier) orbitals** — `_canonical.qvf` (COs) + `_wannier.qvf` (Wanniers) + density / CO / Wannier `.cube` + `.molden`; Wannier centers/spreads in the JSON |
| `aiccm-localize` | **demo:** Wannier localization (Task 1) — JSON reports `density_residual` (unitary-invariance gate ~0 = energy preserved), `objective_initial/final`, Wannier centers/spreads |
| `aiccm-symmetry` | **demo:** space-group symmetry (Task 3) — JSON reports the space group, cluster-invariant order, `overlap`/`kinetic` invariance (AO maps), the union-V breaking, and the symmetry-unique-pair `pair_reduction_factor` |
| `aiccm-pao` | **demo:** DLPNO projected atomic orbitals (Task 2) — JSON reports `n_pao` (= virtual dim) and `pao_occ_orthogonality` (S-orthogonal to occupied ~0) |
| `bipole` | vibe-qc periodic reference, `jk_method="bipole"` (k-mesh) |
| `gdf` | vibe-qc periodic reference, `jk_method="gdf"` (k-mesh) |

Any `aiccm-*` route may be sized by a **real-space interaction radius** instead of
an explicit mesh: `--interaction-range R_c` (bohr) / `--interaction-range-ang Å`
derives the minimal `nrep` whose WS supercell encloses a sphere of `R_c` around
every atom (the real-space dual of k-point density; see
`demo_interaction_range.py`).

The `aiccm-localize` / `aiccm-symmetry` / `aiccm-pao` routes are **demonstration
cases** — full, vq-runnable inputs that exercise each built feature on real
test-set crystals and write its validation metrics to JSON, proving the feature
works (e.g. c-diamond `aiccm-symmetry`: Fd-3m, S/T invariant to 1e-14, 256→19
pairs = 13.5× reduction; h-chain `aiccm-localize`: density residual 1e-15).

This harness contains the Γ-CCM union-and-weight routes, separately labelled
neutral fitted-torus controls, the **MP2 / UMP2 / CCSD(T)** research stack, and a
**vibe-view** visualisation artefact. `aiccm-viz` writes the cyclic-cluster
wavefunction (the visualised property is the HF density / orbitals — the CCM
post-HF stack returns correlation *energies* only, no relaxed density / natural
orbitals). Each energy route writes `<system>__<route>.json` (energy, energy/atom,
converged, walltime, correlation breakdown for mp2/ccsd, the CRYSTAL23
`crystal_ref`) into `$VQ_WORKDIR` (or `--out DIR`).

> **Low-D neutral-control policy.** The harness records every route in
> `NEEDS_3D_COULOMB` as `status="unavailable"` on a 1-D/2-D system. The shared
> neutral RI/GDF mesh collapses transverse reciprocal components and is not a
> wire/slab Coulomb kernel. Do not report its absolute energies, and do not
> treat a vacuum-padded 3-D calculation as a low-dimensional replacement. A
> shared mixed-boundary wire/slab Green function and matching exchange seam
> must be derived and validated first.

### On a configured queue host

**Probe the host before you dispatch a batch to it.**

```sh
vq submit <host> --branch main --cpus 4 probe_host.py    # exit 0 = campaign-ready
```

`probe_host.py` asks two preflight questions: does the host's `vibeqc` export
what the producers import, and is its **compiled core** numerically trustworthy.
The second matters because a stale core does not fail to import: it silently
returns wrong numbers. The full probe measures the direct-torus versus multi-k
GDF gap on the rocksalt-LiH control and checks that `required_g_max` uses the
tightest AO *pair* convention (23.83 for H/STO-3G, not the √2-too-small 16.85).
A fixed host agrees to ~4e-14 Ha/cell; a host predating `f8c213e8` disagrees by
2.04e-2 and its GDF/RI energies are corrupted.

`run.sh` also passes that cached preflight to the producer. The producer binds
it to the copied benchmark payload, records the current source/core/host/package
and native-library identities, and always runs a cheap shifted-mesh
native-versus-Python canary in the process holding the result. If the core-path
SHA256 changed after preflight, it reruns the entire v2 host probe in that
process, not just the cheap canary. It re-reads identity and payload immediately
before writing the result. The core SHA256 hashes the module path at the time of
the read; it is not a cryptographic identity of bytes already mapped into
memory, so the same-process numerical check is an essential part of the
composite attestation.

Always name the host. `vq submit` with no host falls back to `default_host`, and
when that host is down it falls back again to **localhost**, silently running a
cluster job on the laptop. `make_jobs.py` refuses `--machine localhost` for this
reason.

For a directory payload prefer a tarball: `-d studies/aiccm-2026/` uploads the whole
directory (35 MB with `qvf_samples/`, and it hangs); `tar czf` of just
`run_case.py testset.py run.sh crystal_specs.json` is 20 KB.

```sh
vq submit compute-small -c payload.tgz -- bash run.sh c-diamond aiccm-hf
```

Generate the whole batch with `make_jobs.py` (driven by the manifest — a
cost-aware route plan per tier and the target machine for each):

```sh
python make_jobs.py            # review the 92-job vq batch, grouped by machine
python make_jobs.py | sh       # launch it
python make_jobs.py --local    # tier-A as plain `python run_case.py` (no vq)
python make_jobs.py --crystal-basis   # periodic refs at pob-tzvp-rev2 (vs CRYSTAL23)
```

Or submit individual cases:

```sh
# four-center AICCM (small / 1-D on compute-small; 3-D on compute-study)
vq submit compute-small  -d studies/aiccm-2026/ -- python run_case.py h-chain     aiccm-hf
vq submit compute-study  -d studies/aiccm-2026/ -- python run_case.py c-diamond   aiccm-ks
# the periodic references (compute-study; multi-k GDF/bipole are the heavy ones)
vq submit compute-study  -d studies/aiccm-2026/ -- python run_case.py mgo         gdf
vq submit compute-study  -d studies/aiccm-2026/ -- python run_case.py mgo         bipole
# the 3-center hierarchy
vq submit compute-study  -d studies/aiccm-2026/ -- python run_case.py lif-rocksalt aiccm-ri
```

All cases fit a **16-core / 64 GB** node. Tier A is seconds-minutes (local/compute-small);
tier B is minutes-hours (compute-study; the four-center rebuilds each SCF iteration and the
multi-k GDF/bipole references dominate); tier C (oxides, AFM) is the heaviest
(compute-study / compute-reference). Override size with `--nrep i j k` (cluster) / `--kmesh i j k`
(periodic) and the basis with `--basis` (default STO-3G for the four-center;
use `pob-tzvp-rev2` to match CRYSTAL23 on the RI/bipole/gdf routes).

### Locally

```sh
cd studies/aiccm-2026 && python run_case.py h-chain aiccm-hf --out results/
python testset.py        # smoke-build every geometry (no SCF)
python testset.py --check  # pre-flight: cluster-overlap conditioning (Fig-6 guard)
```

Run `python testset.py --check` before queueing the batch — it builds each cluster
at its benchmark `nrep` and reports the minimum eigenvalue of `S^CCM` (must be
positive definite); it exits non-zero if any cell is non-PD. All 18 systems
currently pass (min eig > 1e-3).

## Comparison protocol

For each system, compare per-atom energies across the four references at the
**same basis** (STO-3G for the four-center cross-check; `pob-tzvp-rev2` for the
bipole/gdf/CRYSTAL23 cross-check). The headline tables for the paper:

- **AICCM vs gdf/bipole (same basis, STO-3G):** does the four-center reproduce the
  periodic energy? (yes for covalent/molecular/1-D; Madelung gap for ionic.)
- **AICCM RI vs gdf:** the 3-center routes agree to the RI fitting error.
- **bipole vs gdf vs CRYSTAL23 (pob-tzvp-rev2):** vibe-qc periodic vs CRYSTAL23.
- **cluster-size / k-mesh convergence:** AICCM(nrep) → periodic(converged).

Aggregate the per-case JSON results with `compare.py`:

```sh
python compare.py results/                      # per-system × per-route + the gaps
python compare.py results/ --crystal-refs x.json  # add a CRYSTAL23 column
```

The single-line table reports each route's energy/atom plus Δ4c-gdf (the
four-center vs RI Madelung gap), Δri-gdf, Δgdf-bipole, and Δgdf-CRYSTAL23 (mHa/
atom). D89 disables the historical `compare.py --vs` mode: Γ-CCM is the
union-and-weight construction, χ-CCM is the finite-character construction, and
route-name equality does not define a matched operator. Use `compare_b.py` for
an explicit `not-defined` approach status or for the separately named
`--real-gamma-control-results` neutral-torus Fourier control. That control is
not a Γ-CCM result.
CRYSTAL23 per-atom energies go in a `{system: E_per_atom_Ha}` JSON (populate by
running the `.d12` references; the library outputs are mostly absent).

Reference: Peintinger & Bredow, *J. Comput. Chem.* **35**, 839 (2014),
doi:10.1002/jcc.23550; AICCM theory in `AICCM_ALGORITHM.md` §13; method docs in
`docs/aiccm2026dev_a.md`.

---

## Molecular benchmark batch (methods × basis × RIJCOSX/GridX × convergers)

A separate full-matrix molecular benchmark that stress-tests every combination of
method, basis set, RIJCOSX/GridX tier, and SCF convergence accelerator on 10 small
molecules, distributed over the external profile's `molecular_hosts` rotation.

### Matrix

| Dimension | Values |
|---|---|
| Molecules | h2o, nh3, ch4, hf, co2, h2co, c2h4, hcn, h2s, ch3oh |
| Methods | rhf, rks/pbe, mp2, ccsd, ccsd(t) |
| Basis sets | sto-3g, 6-31g*, def2-svp, cc-pvdz |
| RIJCOSX/GridX | off, legacy-grid, auto-gridx, gridx3 |
| Convergers | ediis-diis, kdiis, ad-cdiis, ediis-diis+newton, ediis-diis+trah, quadratic, damping |
| **Total** | **5,600 jobs** (2,800 per host when using two equally assigned hosts) |

### Submit

```sh
# Pre-flight check (run once before launching)
python preflight_mol_batch.py

# Review first — all 5,600 commands
python batch_full_matrix.py

# Launch a test subset
python batch_full_matrix.py --method rhf --molecule h2o --basis sto-3g | sh

# Launch the full matrix
python batch_full_matrix.py | sh
```

### Analyze

```sh
# Collect results from $VQ_WORKDIR and print tables
python compare_mol_batch.py results/

# CSV for spreadsheet import
python compare_mol_batch.py results/ --csv > batch_results.csv

# Show only failures
python compare_mol_batch.py results/ --failures-only
```

### Scripts

| File | Purpose |
|---|---|
| `batch_full_matrix.py` | Generate 5,600 job scripts + emit `vq submit` commands |
| `preflight_mol_batch.py` | Pre-flight check: verify setup before launching 5,600 jobs |
| `batch_scripts/run_mol_case.sh` | Cluster launcher (finds vibeqc-dev venv) |
| `batch_scripts/*.py` | One self-contained job script per combination |
| `compare_mol_batch.py` | Results aggregator: parse `.out` → tables, CSV, JSON |
