# SKALA showcase calculation campaign

This directory contains independent, fixed-geometry inputs for an experimental
SKALA-1.1 showcase. The campaign is intentionally staged: validate the
molecular adapter first, then exercise a larger molecular property workflow,
then try bulk and surface periodic routes.

SKALA support is experimental. The molecular checkpoint protocol is the
published domain. Periodic calculations below test vibe-qc's general full-grid
XC boundary; they are not a published or accuracy-validated periodic SKALA
method.

## Inputs and run order

| Stage | Input | Purpose | Suggested compute-cluster allocation |
| --- | --- | --- | --- |
| 0 | `h2o_parity.py` | Reproduce Microsoft's H2O/def2-SVP energy and dipole test, with QVF, QTAIM, and one IBO set | `compute-cluster`, 20 CPUs, 90000 MB, 12 h |
| 1 | `s22_water_dimer.py` | Reproduce the S22B water-dimer reaction `-E(02) + 2 E(02a)` and compare its binding energy with 4.989 kcal/mol | `compute-cluster`, 20 CPUs, 90000 MB, 24 h |
| 2 | `benzene_properties.py` | Canonical orbitals, density, QTAIM, population analysis, and IBO/Boys/Pipek-Mezey comparison | `compute-cluster`, 20 CPUs, 90000 MB, 24 h |
| 3a | `lif_gamma.py` | Small bulk periodic gate at Gamma | `compute-cluster`, 20 CPUs, 90000 MB, 24 h |
| 3b | `lif_k222.py` | Repeat the same LiF primitive cell on a 2x2x2 Gamma-centred mesh | `compute-large`, 48 CPUs, 256000 MB, 24 h |
| 4a | `co_mgo_near.py` | Exploratory 3D vacuum-padded CO/MgO(001) model at Mg-C = 2.50 Angstrom | `compute-large`, 48 CPUs, 256000 MB, 24 h |
| 4b | `co_mgo_far.py` | Identical 3D cell and composition at Mg-C = 6.0 Angstrom | `compute-large`, 48 CPUs, 256000 MB, 24 h |

Each input is standalone. With its default output it reserves a new result
directory below `$VQ_WORKDIR`, proves that directory is writable, and writes
all vibe-qc artifacts there. Outside vq it does the same below the current
directory. `--output PATH_STEM` accepts an existing parent explicitly; normal
output creation remains the definitive writability check.

Run the periodic inputs only after H2O parity, the S22B reaction, and benzene
have converged to finite, chemically plausible results. Run `lif_k222.py` only
after the Gamma calculation passes that check. Run the surface pair only after
both periodic smoke jobs and the surface-model dry-run pass. These surface
inputs use 3D periodic GDF with a 30-Angstrom repeat, not the gated vacuum-free
2D slab route. Do not add damping or a level shift to hide periodic
oscillation; an oscillating or impossible periodic result is a bug to
investigate.

## Runtime gate and vq submission

A real SKALA evaluation currently requires Linux, Python 3.11 through 3.13,
the `skala` optional dependencies (including a compatible PyTorch), and the
verified Microsoft checkpoint. Python 3.14 and macOS are deliberately rejected
before model loading.

On compute-cluster, use only the isolated, receipt-verified `vibeqc-skala-dev` runtime.
Provisioning it is an operator action described in
`vibe-queue/docs/scheduler_runtime_deployment.md`; do not point these jobs at a
hand-maintained interpreter. After the implementation has landed, pin its full
commit and confirm that doctor, the managed-update ledger, and program
inventory are healthy:

```sh
SKALA_SHA=$(git rev-parse HEAD)
vq doctor compute-cluster --admin-update --json
vq admin status compute-cluster --json
vq programs compute-cluster --require vibeqc-skala-dev --json
vq doctor compute-large --admin-update --json
vq admin status compute-large --json
vq programs compute-large --require vibeqc-skala-dev --json
```

Example submissions:

```sh
vq submit compute-cluster --program vibeqc-skala-dev --expected-sha "$SKALA_SHA" \
  --cpus 20 --mem-mb 90000 --time 12:00:00 \
  studies/skala-showcase/h2o_parity.py

vq submit compute-cluster --program vibeqc-skala-dev --expected-sha "$SKALA_SHA" \
  --cpus 20 --mem-mb 90000 --time 24:00:00 \
  studies/skala-showcase/s22_water_dimer.py

vq submit compute-cluster --program vibeqc-skala-dev --expected-sha "$SKALA_SHA" \
  --cpus 20 --mem-mb 90000 --time 24:00:00 \
  studies/skala-showcase/benzene_properties.py

vq submit compute-cluster --program vibeqc-skala-dev --expected-sha "$SKALA_SHA" \
  --cpus 20 --mem-mb 90000 --time 24:00:00 \
  studies/skala-showcase/lif_gamma.py

vq submit compute-large --program vibeqc-skala-dev --expected-sha "$SKALA_SHA" \
  --cpus 48 --mem-mb 256000 --time 24:00:00 \
  studies/skala-showcase/lif_k222.py

vq submit compute-large --program vibeqc-skala-dev --expected-sha "$SKALA_SHA" \
  --cpus 48 --mem-mb 256000 --time 24:00:00 \
  studies/skala-showcase/co_mgo_near.py

vq submit compute-large --program vibeqc-skala-dev --expected-sha "$SKALA_SHA" \
  --cpus 48 --mem-mb 256000 --time 24:00:00 \
  studies/skala-showcase/co_mgo_far.py
```

Record each returned job ID. Wait for a terminal state and fetch into a fresh
destination; a timeout from `vq wait` leaves the calculation running:

```sh
vq wait compute-cluster JOBID --json
vq fetch compute-cluster JOBID -o results/h2o
```

These are generous first-run allocations, not measured peaks. Every input also
supports `--dry-run`, which exercises dispatch and output planning without
importing PyTorch or loading a checkpoint. Opt into the normal dry-run memory
estimate with `VIBEQC_DRY_RUN_ESTIMATE=1`:

```sh
VIBEQC_DRY_RUN_ESTIMATE=1 \
  python studies/skala-showcase/lif_gamma.py \
  --dry-run --output /tmp/lif-gamma-skala-plan
```

Keep vibe-qc's memory admission check enabled. `memory_override=True` is not
used anywhere in this campaign.

## Molecular reference and expected comparison

`h2o_parity.py` uses the exact geometry generated by `ase.build.molecule`
for Microsoft's upstream `skala/tests/test_ase.py` test:

| Quantity | Upstream SKALA-1.1 reference |
| --- | ---: |
| SKALA-1.1 + B3LYP5 D3(BJ) total energy | -2076.839069353949 eV = -76.3224280667 hartree |
| Dipole magnitude | 0.413545871474 e Angstrom = 1.986345479 debye |
| Force norm | 0.561464996883 eV/Angstrom |

The upstream ASE calculator enables the checkpoint's B3LYP5 D3(BJ)
correction by default. The input therefore requests `dispersion="b3lyp5"` and
compares the reference with `result.energy_total`; `result.energy` remains the
bare self-consistent SKALA energy. It checks the total energy and dipole but
deliberately does not request forces because analytic SKALA gradients are
gated in vibe-qc. The upstream calculation used def2-SVP with the def2-SVP
JK-fit auxiliary basis.

The defining SKALA paper is Luise et al., *Accurate and scalable
exchange-correlation with deep learning*, arXiv:2506.14665,
doi:10.48550/arXiv.2506.14665. The implementation is experimental and needs
independent verification before a parity result is presented as validation.

### S22B water-dimer benchmark entry

`s22_water_dimer.py` evaluates the public GMTKN55/S22 structures `02` and
`02a` with SKALA-1.1/def2-QZVP, the def2-universal-jkfit auxiliary basis, and
the recommended B3LYP5 D3(BJ) correction. The exact GMTKN55 reaction is

```text
D_e = -E(02) + 2 E(02a).
```

Structure `02b` exists in the public data directory but is not part of this
benchmark reaction; the GMTKN55 reaction file assigns coefficients `-1, +2`
to `02, 02a`. The revised S22B reference binding energy is 4.989 kcal/mol. The
script reports the bare SKALA contribution, the reaction-level D3(BJ)
contribution, their sum, and the signed difference from that reference. It
does not fail merely because the scientific error is nonzero.

This is one entry from the S22 subset, not a reproduction of the defining
SKALA paper's aggregate S22 mean absolute error of 0.14 kcal/mol. It is
nevertheless a useful independent check because it follows the paper's
def2-QZVP molecular benchmark protocol at a hydrogen-bonded geometry. No
geometry optimization, counterpoise correction, or analytic force is added by
the showcase input. Only the dimer requests QVF, density, frontier-orbital,
QTAIM, population, and IBO artifacts; the repeated monomer job remains lean.

The coordinates were copied unchanged from the
[`grimme-lab/GMTKN55` v1 data at commit `8d485b37`](https://github.com/grimme-lab/GMTKN55/tree/8d485b37a1ca8837e395042671ca5ba4e0714691/S22),
licensed CC-BY-4.0. Cite Goerigk et al., *Phys. Chem. Chem. Phys.* **19**,
32184-32215 (2017), doi:10.1039/C7CP04913G, for GMTKN55; Jurecka et al.,
*Phys. Chem. Chem. Phys.* **8**, 1985-1993 (2006),
doi:10.1039/B600027D, for S22; and Marshall, Burns, and Sherrill,
*J. Chem. Phys.* **135**, 194102 (2011), doi:10.1063/1.3659142, for the S22B
revision.

### Optional private water-hexamer paper cross-check

Pöschel et al., *Molecular Implementation of the Machine-Learned Skala
Exchange-Correlation Functional in CP2K through GauXC*, arXiv:2608.19033,
report the prism-to-cage relative energy in their Table S3. SKALA-1.1 places
the prism 0.37 kcal/mol above the cage; adding B3LYP5 D3(BJ) reduces that
difference to 0.06 kcal/mol. Their calculation used all-electron GAPW and a
QZVPP-quality MOLOPT-UZH basis, so a def2-QZVP vibe-qc calculation would be a
geometry-based cross-check rather than a strict basis/grid replication.

The exact prism and cage XYZ files are available in the authors' companion
repository at commit
[`fc671002`](https://github.com/DCM-Uni-Paderborn/Molecular-Skala-in-CP2K/tree/fc67100233f8e145dede656228377e429f473fa4/raw/begdb-water-hexamers/structures),
but neither that repository nor the underlying BEGDB record states an open
redistribution license. Do not copy those coordinates into this public
repository. A private calculation may fetch the two pinned files into an
untracked job directory, verify SHA-256 values
`0cbdfff2d7eae0366c66f3704a8a28d1820551ca462f7f77869bd730a4ab4ce1`
(prism) and
`eb1f54f42c28b9ee8c08dfa57a1a4aceaedf44394646d44ff1d47bbba3345c8a`
(cage), and cite both the paper and companion repository. Publication of the
coordinates themselves needs permission or a clear data-license grant.

`benzene_properties.py` takes the D6h geometry already used by
`examples/regression/systems/molecules/benzene.py`: C-C = 1.397 Angstrom and
C-H = 1.084 Angstrom. It requests `dispersion="b3lyp5"`, the checkpoint's
recommended molecular D3(BJ) settings. The raw `result.energy` remains the
bare SKALA SCF energy, while `result.e_dispersion` and `result.energy_total`
report the additive correction. D3 does not change the orbitals or density.

## Bulk reference context

Both LiF scripts use the two-atom rocksalt primitive cell from
`qc-input-library/scripts/_geometries.py`, with a = 4.03 Angstrom and STO-3G.
Neither input requests a population sidecar. The current non-BIPOLE periodic
writer would use a molecular Gamma-block proxy rather than a validated
periodic population formula, so it is excluded from scientific claims and
blog images.
The existing qc-input-library comparison matrix contains these converged
vibe-qc context values:

| Functional | Gamma (hartree/cell) | 2x2x2 (hartree/cell) |
| --- | ---: | ---: |
| PBE | -105.6600103809 | -106.1552799515 |
| r2SCAN | -105.7891787138 | -106.2437895735 |

They are not SKALA targets. Use them only to catch an obviously impossible
energy scale and to put the cost of the new run in context. Report the
executed GDF route, grid provenance, k-point convention, and experimental
periodic status with any LiF number.

## Surface model and interpretation limits

The surface pair uses an ASCENT-style one-layer MgO(001) checkerboard with
a = 4.21 Angstrom, one upright C-down CO, and a Gamma-only 3D GDF calculation
in a 30-Angstrom vacuum-padded repeat cell. The exact vacuum-free 2D external-
XC density domain is not wired, so these inputs deliberately do not use the
`SLAB_EWALD_2D` route. This is a six-atom, high-coverage, unrelaxed STO-3G
surrogate, not a converged adsorption model. The rounded C-O distance is
1.15 Angstrom, and the illustrative near/far Mg-C distances are 2.50 and
6.0 Angstrom.

For the exploratory fixed-composition scan, calculate

```text
Delta E_near-far = E_near - E_far
```

but do not label it a converged adsorption energy. It has no counterpoise
correction, relaxation, coverage convergence, slab-thickness convergence, or
periodic dispersion. Periodic SKALA with `dispersion=` is intentionally
rejected. No literature adsorption value is a pass/fail target for these
illustrative inputs.

## QVF images for the blog post

After fetching a completed job, inspect the archive before choosing a
supported capture target:

```sh
vibe-view capture-selftest
vibe-view info path/to/result.qvf --json
vibe-view capture path/to/result.qvf \
  -s vol_dens_0 -o density.png --size 1600x1000
```

Replace `vol_dens_0` with the exact density-volume section ID reported by
`vibe-view info`; omit `-s` to capture the structure view.

The command-line capture path currently supports structure, density-volume,
basis, and band views when those payloads exist. Wavefunction, localized-
orbital, and QTAIM sections must be opened in the interactive vibe-view
browser and captured there. Suggested figures are:

1. A command-line H2O density capture, plus interactive HOMO/LUMO and QTAIM
   views.
2. The S22 water-dimer density and an interactive QTAIM view showing whether
   the hydrogen-bond bond path and critical point are present; describe the
   topology only after inspecting the fetched archive.
3. Interactive benzene canonical frontier orbitals and a three-panel IBO,
   Foster-Boys, and Pipek-Mezey comparison of the occupied pi space.
4. LiF structure and density captures, plus selected Gamma crystalline
   orbitals in the interactive viewer.
5. Matching structure and density side views of near and far CO/MgO.

The periodic archive's `wavefunction.gto` section represents Gamma-point
crystalline orbitals only. The viewer evaluates central-cell AOs and omits
periodic-image AO tails, so orbitals crossing a cell face are approximate.

The generic periodic QVF DOS/PDOS, COOP/COHP, surrogate-Hamiltonian Mayer, and
Hcore-band rebuild is disabled for SKALA. That reconstruction cannot yet
include the nonlocal XC potential, so emitting it would create a
plausible-looking but scientifically wrong band structure. Non-BIPOLE
periodic population sidecars are also disabled in this campaign because their
current molecular Gamma-block proxy is not a validated periodic bond-analysis
formula. These scripts request neither `dos_kmesh`, `band_structure`,
`coop_cohp`, nor periodic population output; the blog should report those
gates explicitly.
