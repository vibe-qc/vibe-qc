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

## Real-Gamma ladder parity against PySCF and CRYSTAL23 (RG-13)

The real-Gamma producer (`run_ccm_rhf_direct`) evaluates the neutral
fitted-torus Hamiltonian on the `nrep` supercell at Gamma. By band folding
that is the unit cell's periodic HF on the Gamma-centred `nrep` k-mesh, and
the real-Gamma and neutral-Bloch producers agree to 1e-10 Ha/cell in the test
suite. These results are evidence on that neutral control for the production
ledger (`docs/aiccm2026dev_a_production.md`, RG-13), not a Gamma-CCM result in
the sense of the section above.

* `ladder_pyscf.py` compares the real-Gamma energy per cell with PySCF's KRHF
  on the same mesh. Both codes use vibe-qc's AO and auxiliary shells
  (def2-svp-jk for STO-3G), transported to PySCF shell by shell, and
  `exxdiv='ewald'`. The real-Gamma SCF starts from SAD and is checked for an
  RHF internal instability. vibe-qc's SR/LR fit (`fit_source="sr"`) is
  compared with PySCF's range-separated GDF (`rs_density_fit`); both implement
  Ye and Berkelbach, J. Chem. Phys. 154, 131104 (2021). The classic fit is
  compared with PySCF's GDF (`density_fit`) where its tail is affordable.
* `ladder_crystal.py` runs CRYSTAL23 at converged meshes with the same STO-3G
  shells written into the input. At a finite mesh CRYSTAL's real-space
  exchange is not the torus exchange, so only the converged limits compare.

```sh
python ladder_pyscf.py                                   # diamond, LiH, MgO, fcc Be, MgO pob-DZVP-rev2
python ladder_pyscf.py --cases c-diamond lih-rocksalt --meshes 3,3,3 --no-sr --merge
python ladder_pyscf.py --cases c-diamond lih-rocksalt mgo --no-vibeqc --pyscf-meshes 3,3,3 4,4,4 --merge
python ladder_pyscf.py --cases lih-rocksalt --no-vibeqc --no-classic --pyscf-meshes 5,5,5 6,6,6 --merge
python ladder_crystal.py --crystal Pcrystal --mpi 3      # CRYSTAL23 at SHRINK 8
```

The results are in `results_ladder_pyscf.json` and `results_ladder_crystal.json`
(source identity in each row). Measured on 2026-10-14 with PySCF 2.14.0 and
CRYSTAL23 1.0.1, energies per primitive cell in Ha, differences vibe-qc minus
PySCF in uHa/cell, STO-3G unless named:

| cell | mesh | vibe-qc SR | PySCF RSGDF | SR difference | classic difference | PySCF GDF minus RSGDF | lowest RHF Hessian eigenvalue |
|---|---|---|---|---|---|---|---|
| diamond | 1x1x1 | -74.00247768 | -74.00248187 | +4.2 | +214.9 | +1.8 | -0.107 |
| diamond | 2x2x2 | -74.81479108 | -74.81480276 | +11.7 | +129.9 | +22.7 | +0.126 |
| LiH | 1x1x1 | -8.33518581 | -8.33518431 | -1.5 | +139.5 | +1.1 | +0.595 |
| LiH | 2x2x2 | -7.92203512 | -7.92201709 | -18.0 | +53.8 | +13.8 | +0.088 |
| MgO | 1x1x1 | -271.04835882 | -271.04836575 | +6.9 | not run |  | -0.117 |
| MgO | 2x2x2 | -271.20999469 | -271.20999781 | +3.1 | not run |  | +0.181 |
| fcc Be | 1x1x1 | -15.20310890 | -15.20310890 | -0.0 | +39.1 | +1.2 | +1.046 |
| fcc Be | 2x1x1 | -14.85599679 | -14.85599910 | +2.3 | +32.5 | +8.8 | +0.137 |
| fcc Be | 2x2x2 | -14.37762717 | -14.37763022 | +3.1 | +35.6 | +24.6 | -0.254 |
| MgO pob-DZVP-rev2 | 1x1x1 | -274.39053514 | -274.39053580 | +0.7 | not run |  | -0.004 |

Reading the table:

* The SR route agrees with PySCF's range-separated GDF to 0.0 to 18 uHa/cell on
  every rung. That is the size of PySCF's own spread between its two fits on
  the same cells (1.1 to 24.6 uHa/cell), so the parity is at the fitting
  floor of the two codes, not of the construction. On fcc Be (2,1,1) the PySCF
  run reproduces the independent anchor of
  `tests/test_ccm_neutral_tail_parity.py` (-14.855999005825) to 0.09 uHa.
* The Madelung constants of the exchange `q = 0` term (vibe-qc
  `ccm_exchange_q0_madelung`, PySCF `tools.pbc.madelung`) agree to 5e-16 on
  every rung: the exchange convention is the same.
* The classic fit sits 32 to 215 uHa/cell above PySCF's GDF. This is the
  documented dense-core truncation floor of the classic all-reciprocal fit
  (`vibeqc.pbc_gdf`), and the reason the SR route exists.
* Both codes converge to the same stationary point on every rung. The flagged
  states at (1,1,1) (diamond, MgO, MgO pob-DZVP-rev2) are singlet-stable RHF
  minima with an RHF-to-UHF (triplet) instability; fcc Be (2,2,2) has a
  negative singlet eigenvalue too, whose mode breaks the primitive translation
  symmetry, which a k-mesh KRHF cannot follow.
* Cost, not accuracy, is the open problem of the SR route at `N_c > 1`: on
  diamond, LiH and MgO (2,2,2) it takes 1374, 2815 and 2380 s against PySCF's
  29, 50 and 17 s, because the shared SR builder recomputes its real-space
  integrals for every transfer `q` and its reciprocal AO-pair transforms for
  every ket (vibe-qc issue 543). The classic fit takes 235 and 25 s on diamond
  and LiH (2,2,2).

**The CRYSTAL23 limit.** CRYSTAL23 at SHRINK 8 (`TOLINTEG 8 8 8 8 16`, the
transported STO-3G, `results_ladder_crystal.json`) gives -74.8771217 (diamond),
-7.9381931 (LiH) and -271.2148222 Ha/cell (MgO). Diamond at SHRINK 12 settles
within 1 uHa per cycle at -74.877128, 6 uHa lower, so CRYSTAL's own mesh is
converged; LiH equals the CRYSTAL14 internal-STO-3G baseline of
`examples/regression/crystal_parity/baseline_sto3g` (-7.93816810) to 0.02 mHa,
which checks the basis transport. PySCF's KRHF on the same Hamiltonian family
(the real-Gamma producer equals it rung by rung, above) then gives, as PySCF
minus CRYSTAL23 in mHa/cell:

| mesh | diamond | LiH | MgO |
|---|---|---|---|
| (2,2,2) | +62.3 | +16.2 | +4.8 |
| (3,3,3) | -1.1 | -10.2 | -8.1 |
| (4,4,4) | -5.1 | -2.3 | -4.4 |
| (5,5,5) | -3.7 | -1.8 | -2.5 |
| (6,6,6) | -2.5 | -1.0 | not run |

From (4,4,4) on every sequence approaches the CRYSTAL23 value from below, and at
the largest mesh run the gap is 2.5, 1.0 and 2.5 mHa/cell. The torus exchange
of these two-atom cells converges slowly and not yet in a single power law
(the diamond steps are 1.36 and 1.24 mHa), so no extrapolated limit is quoted:
the limits agree at the few-mHa level the meshes resolve, and the same-mesh
PySCF table above is the parity evidence. On LiH (3,3,3) the classic fit sits
142 uHa/cell below PySCF's RSGDF at its default 15-bohr cell list and 18 uHa/cell
above it at 25 and 35 bohr: the origin-ball pair domain of the classic fit
(vibe-qc issue 466) truncates the diffuse Li 2sp pairs once the torus resolves
them.

---

## The per-cell spin sector against PySCF (RG-14)

`CCMSystem(..., spin_state=)` names the spin sector of the torus (user guide,
"The spin sector of the torus"). Under the default `"legacy"` the real-Gamma
producer pairs a doublet cell by supercell parity while the Bloch producer
replicates the cell's counts, so a doublet cell on `(2,1,1)` runs two sectors
of one operator; `"per-cell"` makes both run the aligned uniform sector,
which is also what PySCF's k-point UHF runs when `nelec` is pinned to
`(N_k n_alpha, N_k n_beta)`.

* `spin_state_pyscf.py` runs the Li doublet cube (STO-3G, 7 bohr) at
  `(2,1,1)` in the per-cell sector on both producers and on PySCF KUHF and
  KUKS/PBE with pinned `nelec`, `exxdiv='ewald'` and the transported shells,
  SR fit against `rs_density_fit` and classic fit against `density_fit`.
* `spin_state_vq_job.py` is the vq job for the same rows through
  `run_periodic_job(method="aiccm", aiccm_spin_state="per-cell")`, UHF and
  UKS on both variants, refusing a runtime that predates the contract.

Measured on 2026-10-26 (source 12aed222 plus the RG-14 change, PySCF 2.14.0,
both Madelung constants 0.2579774015; energies per primitive cell in Ha,
differences in uHa/cell):

| row | real-Gamma | neutral-Bloch | PySCF | real-Gamma minus PySCF | real-Gamma minus Bloch |
|---|---|---|---|---|---|
| UHF, SR fit / RSGDF | -7.4055425896 | -7.4055425896 | -7.4055425896 | 0.000 | 2.6e-13 Ha |
| UKS PBE, SR fit / RSGDF | -7.4090022073 | -7.4090022073 | -7.4091926480 | 190.4 | 2.5e-13 Ha |
| UHF, classic fit / GDF | -7.4058450937 | -7.4058450937 | -7.4055425896 | -302.5 | -4.1e-14 Ha |
| UKS PBE, classic fit / GDF | -7.4087302147 | -7.4087302147 | -7.4091926480 | 462.4 | -1.1e-13 Ha |

All four PySCF runs converged with integer occupations, 4 alpha and 2 beta
electrons on the torus and `<S^2>` 2.000 for UHF. The UHF/SR row is the
matched comparison: fit, exchange zero mode, basis, mesh and sector agree
and the energies agree to below 1 nHa/cell. Unmatched approximations,
reported separately: the classic fit is a different fitted operator
(reciprocal support and tail; the ladder above measures its class), and the
KUKS rows use each code's own XC quadrature (PySCF's uniform FFT grid against
vibe-qc's periodic Becke grid), which the 190 uHa/cell of the SR row
measures. For the record, the legacy pair on the same cell and mesh is 4.83
mHa/cell apart (supercell singlet 3/3 against per-cell doublet 4/2, classic
fit) and is refused by `assert_compatible_spin_state`.

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
