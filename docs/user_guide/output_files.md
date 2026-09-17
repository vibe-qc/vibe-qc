# Input scripts and output files

A classic quantum-chemistry workflow means running a job and getting back
a text log, plus files a viewer can open. `vibeqc.run_job` bundles that
up so you don't have to wire it together by hand: one call writes the
formatted output, the molden orbital file, and, for optimization
runs, a trajectory animation.

## Writing an input script

An input "script" in vibe-qc is just a Python file. The conventional
shape:

```python
# input-h2o.py
from pathlib import Path
from vibeqc import Molecule, run_job

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE / "h2o.xyz")

run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output=HERE / "output-h2o",
)
```

Run it like any Python file:

```sh
python3 input-h2o.py
```

A spread of ready-to-run example inputs lives under
[`examples/molecular/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/molecular)
and [`examples/periodic/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic)
- single-point RHF / DFT, open-shell UHF, BFGS geometry
optimization, MP2 / double hybrids, dispersion-corrected
optimization, cube output, the SCF Fock-build modes, and more.
Copy any of them as a template.

## Output files

Given `output="output-h2o"`, `run_job` writes up to a dozen file
families named by the same stem. Always-on for every molecular
run: `.out` (text log), `.system` (TOML manifest), `.molden`
(orbitals), `.xyz` (final geometry), `.bibtex` + `.references`
(citations, v0.8.x+, see [citations](citations.md)), and
`.population.{txt,json}` (Mulliken / Löwdin / Mayer / dipole, also
v0.8.x+), and **`.qvf`**, vibe-qc's native single-file visualization
archive for [vibe-view](#output-h2oqvf-native-visualization-archive-default).
Both high-level runners default to `output_qvf=True`; pass
`output_qvf=False` only when the archive is not wanted. Conditional or
opt-in siblings include `.traj` (geometry optimisation
only); `.perf` (opt-in via `perf_log=`); `.scf.jsonl` (opt-in via
`structured_log=`); `.density.cube` and `.{homo,lumo,…}.cube`
(opt-in via `write_cube=`); `.dump` (only on SCF failure).
Periodic runs (`run_periodic_job`) emit the same family plus
`.POSCAR` and `.xsf` for the crystal structure.

```{note}
`output=` is a **stem**, and every suffix above is *appended* to it
verbatim. A stem may therefore contain dots — a lattice constant, a
scale factor, a tolerance — and keeps them:
`output="output_scale_4.08"` writes `output_scale_4.08.out`, and
`output="lih-gdf-rhf-a4.0082-k222"` writes
`lih-gdf-rhf-a4.0082-k222.out`. Distinct stems always give distinct
files, so a scan can name its points by their parameter value.

Through v0.15.151 the suffix was *substituted* rather than appended, so
such a stem was truncated at its dot and the points of a scan overwrote
one another while every job still exited 0 (issue #254).
```

### `output-h2o.out`, the text log

Plain ASCII, readable in any editor. Sections, in order:

1. **Banner**, vibe-qc + libint + libxc + spglib versions, for
   provenance. Identical to `vibeqc.print_banner()`.
2. **Job header**, method + basis.
3. **Initial atom table**, Z + Cartesian bohr + charge + multiplicity
   + electron count.
4. **Optimization block** (only when `optimize=True`), target
   convergence, final optimized geometry.
5. **SCF trace**, iteration-by-iteration energy, ΔE, commutator norm,
   DIIS history length. Flags `converged` / `NOT converged`. Basis-free
   semiempirical engines currently expose only the final iteration count, so
   their log keeps the convergence/energy verdict and omits the unavailable
   per-iteration table.
6. **Energy components** (DFT only), nuclear repulsion, electronic,
   Coulomb *J*, HF-exchange *K*, XC, total. On the BIPOLE periodic routes
   the *K* row is followed, whenever exact exchange is present, by an
   indented `of which q=0 finite-size` row. See
   {ref}`reading-exchange-finite-size-row`.
7. **Orbital-energy table**, all occupied MOs and up to
   `n_virtual=5` virtual MOs (override via `n_virtual=...`), with
   HOMO/LUMO markers and the HOMO-LUMO gap in Ha and eV. For UHF/UKS,
   separate Alpha and Beta blocks with per-spin HO*S*MO / LU*S*MO
   markers and per-spin gaps.
8. **Linear dependence** (periodic GDF routes, v0.15.x+), what
   canonical orthogonalisation and the density fit actually kept: basis
   functions per k, retained dimension after orthogonalisation, how many
   directions were discarded, the threshold used, the smallest overlap
   eigenvalue and the overlap condition number, plus the auxiliary-fit
   retained dimension and its own threshold. Printed **even when nothing
   was discarded**, because "0 directions dropped" is what rules an
   ill-conditioned basis out as the explanation for a surprising energy.
   See {ref}`reading-linear-dependence-block`.
9. **References block** (v0.8.x+), `## References` section listing
   every paper that should be cited for the calculation just run.
   Auto-assembled from the bundled [citation database](citations.md);
   the matching BibTeX entries live in the `.bibtex` sibling.

The same content is available programmatically from
`vibeqc.format_scf_trace(result, molecule=...)`, which returns a string you
can log, print, or splice into your own output layout.

(reading-exchange-finite-size-row)=
#### Reading the `of which q=0 finite-size` row

Exact exchange on a finite k-mesh has a `q -> 0` singularity. vibe-qc treats
it with the standard probe-charge-Madelung (Gygi-Baldereschi) correction --
the same convention as PySCF's `exxdiv='ewald'` -- which is what lets a
Hartree-Fock or hybrid total converge quickly with mesh density. That
correction is part of the reported exchange energy and part of the reported
total, and on a coarse mesh **it is large**: on MgO rocksalt / STO-3G at
`(2,2,2)` it is **-2.88 Ha**.

The row reports it on its own, indented under `HF exchange` to show that it is
already counted there:

```
  HF exchange                          -24.6642094443
    of which q=0 finite-size            -2.8814780580
```

It is the amount that `exchange_exxdiv='none'` would remove, so it is also,
to leading order, **the amount by which a code that treats the singularity
differently will disagree with vibe-qc at the same k-mesh**. That is the row's
reason for existing. CRYSTAL, for example, applies no such correction and
truncates exchange in real space instead: at MgO `(2,2,2)` its Hartree-Fock
total sits 638 mHa from its own `SHRINK 8` value, while vibe-qc's is within
5 mHa of it. A cross-code comparison that does not account for the convention
is comparing two different quantities, and before this row nothing in the
output said so. The convention itself, and why CRYSTAL's coarse-mesh value is
not a target for this route, is written up in
[`docs/periodic_jk_routes.md`](../periodic_jk_routes.md) under the 2026-06-16
BIPOLE-CRYSTAL finding.

The row is absent for a pure functional (no exact exchange, so no such term)
and is exactly `0.0` under `exchange_exxdiv='none'`. It is also available
programmatically as `e_exchange_finite_size` on each entry of the result's
`energy_components`.

(reading-linear-dependence-block)=
#### Reading the linear-dependence block

```
  Linear dependence
  --------------------------------------------------------
    basis functions per k     = 14
    retained after orthog.    = 13-14
    directions discarded      = 1 (max over k)
    linear-dep threshold      = 1e-07
    min eigenvalue of S(k)    = 3.204e-06
    overlap condition number  = 9.709e+06
    auxiliary functions       = 189
    auxiliary fit retained    = 83
    auxiliary threshold       = 1e-09
```

A periodic SCF builds one overlap $S(k)$ per Brillouin-zone point and
canonically orthogonalises each one, dropping every direction whose
eigenvalue falls below `linear-dep threshold`. Those dropped directions
are removed from the variational space the SCF may use, so a run can be
solving a smaller problem than the basis implies.

What to look at, in order:

- **`directions discarded`.** Zero is the reassuring case. Non-zero means
  the basis is redundant at at least one k-point; a *large* value on a
  small basis usually means the lattice cutoff is too short for the
  Bloch sums rather than that the basis is genuinely redundant.
- **`min eigenvalue of S(k)`.** A true overlap is positive definite. A
  **negative** value is not near-degeneracy, it is a broken Bloch sum or
  too short a cutoff, and the drivers fail closed on it.
- **`auxiliary fit retained`.** This is the density-fitting metric, not
  the AO basis, and on compact dense-core cells it is the dominant
  failure mode: the fit is orthogonalised as $M^{-1/2}$, so a retained
  direction whose eigenvalue is really truncation noise gets amplified by
  $1/\sqrt{\lambda}$. If a periodic energy is wildly wrong and this line
  shows a large fraction discarded (or a suspiciously small threshold),
  try a looser `gdf_linear_dep_threshold` before suspecting the operators.
  The threshold is compared **absolutely**: a fit direction is dropped when
  its raw metric eigenvalue falls below it. Each fitting metric also carries
  a noise floor proportional to its largest eigenvalue, which guards against
  the metric's own construction error (that error scales with the metric,
  which is why the floor does too). Mixed density fitting
  (`gdf_method="mdf"`) uses a much larger floor than the other backends,
  because the plane-wave-dressed metric it decomposes is deliberately
  singular and its near-null modes lie inside that construction error.

The block is also what a non-physical-energy diagnostic quotes, so a
failing run says which of those it was rather than listing candidates.

### `output-h2o.qvf`: native visualization archive (default)

vibe-qc's native visualization format, **QVF** (`.qvf`), is a single
ZIP archive that bundles a calculation's structure, electron density,
orbitals, and basis + MO coefficients into one file, opened in
**vibe-view**, vibe-qc's GPU-accelerated viewer. It is the recommended
way to get visualization output: rather than wiring a `.molden` and a
handful of `.cube` files together for a third-party viewer, the default
`output_qvf=True` writes one coordinated archive. Set `output_qvf=False` to
skip it deliberately.

```python
# doc-audit: skip - output flags for the molecule defined earlier in this guide
run_job(mol, basis="6-31g*", method="rhf",
        write_cube=["density", "homo", "lumo"],  # which fields to evaluate
        output_qvf=True, output="output-h2o")
# → output-h2o.qvf
```

The archive carries typed sections: `structure`, `volume.density`, one
`volume.orbital` per requested MO, `wavefunction.gto` (basis +
coefficients, so the viewer resamples any orbital on demand), and the
`citations`. Open it with:

```sh
vibe-view open output-h2o.qvf
```

```{note}
In practice `wavefunction.gto` carries **spherical-harmonic (pure) shells**:
every basis vibe-qc builds by name is pure (`5d`/`7f`, not `6d`/`10f`).
Cartesian shells are written too: they arise from any explicit-`ShellInfo`
`BasisSet` with `pure=False`, including one built by
`vibeqc.basis_toolkit.to_libint_basis()` from a basis imported as
Cartesian (see [basis_toolkit](basis_toolkit.md)).

If you write your own QVF consumer, note the normalisation convention for
that case (QVF spec Appendix A.1): contraction coefficients are scaled by
**one** factor per shell, taken from the total `l`, so mixed Cartesian
components are deliberately *not* unit-normalised: a d shell has
`⟨xx|xx⟩ = 1` but `⟨xy|xy⟩ = 1/3`. Do not apply a further per-component
correction: doing so scales `d_xy` by `√3` and the rendered density by 3.
vibe-qc's own evaluator and vibe-view both made exactly that mistake until
2026-08-05, and no spherical calculation showed a symptom.
```

`run_periodic_job` likewise defaults to a periodic `.qvf` containing the
structure, density, and bands or DOS where computed. The
format is open and versioned: see the
[QVF tech spec](../design_qvf_format.md), [The QVF file format, end to end](../tutorial/qvf_file_format.md)
(format) and [vibe-view: an end-to-end walkthrough](../tutorial/vibe_view_walkthrough.md)
(viewer), and the motivation post [*Quantum chemistry needs a modern
file format*](https://vibe-qc.com/2026/05/21/qc-needs-a-modern-file-format-qvf/).

For concrete periodic archives, the static
[example-output catalog](../example_outputs.md) includes the curated
`chi-ccm-b-qvf` bundle: two finite-BvK chi-CCM-B QVF files, full `.out`
logs, sanitized `.system` manifests, and headless vibe-view captures for
structure, density, orbitals, replication, and Wannier-centre overlays.
It also includes `mgo-route-gdf-bipole`, a historical MgO primitive
RHF/STO-3G `2 x 2 x 2` GDF/BIPOLE route bundle with full logs, sanitized
manifests, QVFs, summaries, and vibe-view captures. The BIPOLE QVF is
structure/properties/SCF-history only because that run predates exact
returned-density export and its optional density/DOS artifact generation
warned after the SCF had converged. Current full-mesh BIPOLE runs write a
density volume after exact refold and numerical certification; this static
archive remains unchanged until regeneration.

### `output-h2o.bibtex` / `output-h2o.references`, auto-citations

Two siblings that pair with the in-`.out` references block (see
[the citations user guide](citations.md) for the full schema):

- **`.bibtex`**, one `@article` / `@software` entry per cited work,
  in citation order. Drop into `\bibliography{output-h2o.bibtex}` and
  `\cite{...}` each entry by its `bibtex_key`.
- **`.references`**, Chicago-style numbered list, human-readable.
  Open in a plain-text editor when you want to glance at the
  bibliography without firing up LaTeX.

Both are regenerated on every run, so a re-run with a different
functional / basis / dispersion automatically produces the updated
bibliography. Routing gaps (unknown basis, custom libxc id without a
route) surface as `# no citation route for …` warning lines at the
bottom of the `.references` file; the job itself never fails on a
gap.

### `output-h2o.molden`, molecular orbitals

Molden-format file carrying:

- The geometry in atomic units.
- The full basis set (per-atom, per-shell exponents and contraction
  coefficients, raw, primitive normalisation is reapplied by the
  reader).
- Every molecular orbital: symmetry label (`A` for now, vibe-qc is
  not yet symmetry-adapted), orbital energy in Hartree, spin, occupancy
  (2 for occupied restricted, 1 for each alpha/beta spin-occupied,
  0 otherwise), then the AO coefficients reordered from libint's
  `m = -L..+L` convention to Molden's `(m=0, +1, -1, +2, -2, ...)`
  ordering so the file is drop-in for any molden-aware viewer
  (Jmol, Avogadro, Molden itself, IQmol, MolView).

For unrestricted (UHF/UKS) results the file contains two MO blocks,
first the Alpha spin, then the Beta spin, as the Molden format
requires.

`write_molden_file=None` (the default) declares this file only when the
selected route exposes the required Gaussian AO wavefunction. Explicit `True`
is a guaranteed request and fails before calculation on basis-free
semiempirical, solver-only, and MLIP routes. `False` disables it.

For periodic calculations, Molden export writes one exact Gamma block. It is
available for a single-Gamma result and for Gamma-containing BIPOLE/GDF meshes
whose result metadata locates that block. Shifted meshes and routes without a
supported Gamma-block export omit the auto plan row; explicit `True` fails
before calculation.

Population output has a separate capability rule. Exact single-Gamma routes
use the molecular analysis, BIPOLE uses its full lattice-density/k-mesh
analysis, and χ-CCM (`jk_method="aiccm2026dev-b"`) uses its full finite-torus
density and overlap contractions. Other multi-k routes have no declared
crystal population convention and fail closed on an explicit request.

Open any .molden file via:

```sh
# Jmol (cross-platform, Java):
jmol output-h2o.molden

# Avogadro 2 (cross-platform):
avogadro output-h2o.molden

# MolTUI — terminal-only, no GUI required (great over SSH).
# Install via `pip install -e '.[viewer]'` from the vibe-qc checkout
# or via `./scripts/install_optional_tools.sh`. See the
# [installation page](../installation.md#optional-terminal-viewer-moltui).
moltui output-h2o.molden
```

### `output-h2o.trexio.h5`, TREXIO wavefunction (opt-in)

`run_job(..., trexio=True)` additionally writes the converged wavefunction
as a [TREXIO](trexio.md) file: nuclei, Gaussian basis with explicit
normalization factors, AO conventions, MO coefficients / energies /
occupations, the one-electron integrals and the total energy, in the
open container that Quantum Package, CHAMP, QMC=Chem, TurboRVB, QMCkl and
`trexio-tools` read directly. `trexio_backend="text"` writes the
`output-h2o.trexio` directory of the text back end instead. Requires the
optional `[trexio]` extra. ECPs and periodic Gaussian SCF results are
supported; routes without a Gaussian AO wavefunction are refused. Read it back with `vibeqc.read_trexio`. Details,
group coverage and the verified conventions: [TREXIO](trexio.md).

### `output-h2o.system`, runtime manifest

Plain TOML pinning the runtime environment that produced the
`.out`. The `.out` file carries the chemistry; the `.system`
sibling carries the hardware, linked-library, and timestamp
context needed to interpret a wall-time figure or reproduce a
calculation on a different box. It also records the validation
boundary: external QC programs are references only, run
out-of-process, and are not imported as vibe-qc backends. Without it,
an `.out` says
*"SCF total: 0.015 s"* with no indication whether that's a fast
machine, a slow one, or a single-threaded build.

Sample manifest:

```toml
# vibe-qc system manifest — written alongside output-<job>.out by run_job(...).
# Captures the runtime environment so generated calculation outputs are
# reproducible and wall-time numbers are interpretable. The [plan]
# section is the declared output contract (written once, at job start);
# the [outputs] section is the running status (rewritten as each file
# lands). External QC programs are validation references only.

[vibeqc]
version       = "0.12.0"
codename      = "Knuth's Beaver"
git_sha       = "0123456789abcdef0123456789abcdef01234567"
git_branch    = "main"
is_release    = true

[host]
hostname      = "workstation.invalid"
os            = "Darwin"
os_release    = "23.4.0"
os_pretty     = "macOS 14.4"
arch          = "arm64"

[cpu]
model         = "Apple M2 Pro"
physical_cores = 10
logical_cores  = 16
omp_threads_used = 12

[memory]
total_gb      = 32.0
available_gb  = 24.0

[python]
version       = "3.14.0"
implementation = "CPython"
executable    = "~/.venv/bin/python"

[libraries]
libint    = "2.13.1"
libxc     = "7.0.0"
spglib    = "2.7.0"
libecpint = "1.0.7"
fftw3     = "3.3.10"

[validation]
external_programs_policy = "External QC programs are validation references only."
execution_boundary = "Run external programs out-of-process and parse their outputs; do not import them as vibe-qc backends."
native_backend_policy = "vibe-qc runtime methods execute vibe-qc-owned native or Python code."

[run]
timestamp_iso  = "2026-04-29T21:42:14-04:00"
wall_seconds   = 0.084
basename       = "input-h2o-rhf"
pid            = 51284

# Present only when the job requested a Hessian (hessian=True). Names
# the potential-energy surface the frequencies describe, which is not
# always the method that was asked for: run_job resolves a correlated
# request down to the mean-field reference its SCF runs, and the
# finite-difference Hessian differentiates that reference. For
# method="mp2" the surface is RHF, so surface_is_requested_method is
# false. available = false means the requested method resolves to no
# surface with a finite-difference Hessian at all, and no frequencies
# were reported.

[hessian]
requested_method = "mp2"
surface        = "RHF/6-31g*"
surface_method = "rhf"
surface_is_requested_method = false
available      = true

# v0.8.x+ Phase-O1 additions: declarative pre-flight plan + running
# outputs status. vq's `--vibeqc-preflight` reads [plan] to know what
# files to expect; vq's status polling reads [outputs] for liveness.

[plan]
stem            = "output-h2o"
job_kind        = "molecular_scf"
method          = "RHF"
basis           = "6-31g*"
functional      = ""
options_digest  = "f3a2c1..."

[[plan.files]]
role        = "log"
path        = "output-h2o.out"
format      = "text"
always      = true
description = "Human-readable SCF log."

[[plan.files]]
role        = "manifest"
path        = "output-h2o.system"
format      = "toml"
always      = true
description = "Runtime manifest (this file)."

# … one row per declared artefact (.molden, .xyz, .bibtex,
# .references, .population.{txt,json}, optionally .traj / .perf /
# .scf.jsonl / .dump / .density.cube / .homo.cube / .lumo.cube / …)

[outputs]
status          = "complete"         # running | complete | crashed | dry_run
finished_at_iso = "2026-04-29T21:42:14-04:00"

[[outputs.files]]
path         = "output-h2o.out"
written      = true
bytes        = 4231
sha256       = "ab12cd34..."
checksum_status = "sha256"
wall_time_s  = 0.082

[[outputs.files]]
path         = "output-h2o.system"
written      = true
bytes        = 0
sha256       = ""
checksum_status = "self-excluded"
wall_time_s  = 0.084

# ... one row per declared artefact, with its write/checksum state and
# wall-time-since-job-start.
```

The `git_sha` is the full immutable 40-character commit ID when Git provenance
is available (builds without Git metadata use the `unknown` sentinel).
`checksum_status` is `pending`, `sha256`, `unavailable`, `failed`, or
`self-excluded`. The `.system` row is always `self-excluded`: a file cannot
truthfully contain the digest or byte count of its own final bytes, because
inserting either value changes those bytes. Its empty SHA and zero byte count
are explicit sentinels, not a failed validation. Older manifests that contain
a self-digest captured the pre-finalization file and that digest must not be
used to validate the fetched `.system` file.

An `[outputs].status = "complete"` manifest guarantees that every planned row
with `always = true` has `written = true`. Finalization cannot silently leave
a requested guaranteed sidecar pending: the manifest is instead written with
`status = "crashed"`, the missing row gets `checksum_status = "failed"`, and
the runner raises `IncompleteOutputError`. Rows with `always = false` describe
conditional files and may remain pending on a successful run.

The shape is fixed, every section + key listed above is always
present, even when a probe falls back to `"unknown"`. Downstream
parsers don't need to handle missing keys, only the unknown
sentinel value. The `[plan]` + `[outputs]` sections were added in
the v0.8.x output-module work (Phase O1 of
[`docs/design_output_module.md`](../design_output_module.md)) and
are strictly additive, pre-v0.8.0 parsers that only read the older
sections still work without changes.

The `[run]` section additionally carries route-specific scalar keys
that the runner merges in after the fixed ones: the requested,
resolved and executed J/K method (`jk_method_requested`,
`jk_method_resolved`, `jk_method_executed`; for the real-Gamma AICCM
route all three read `"real-gamma"`), the periodic `exchange_q0`
convention, `dft_plus_u_route`, the executed backend and its
`parity_held` flag, and the GAPW or finite-torus convention fields of
the route that ran. They are always scalars, so the fixed-shape rule
still holds: read the keys you know and ignore the rest. One of them
is a maturity marker. Every job dispatched through an AICCM route
(`run_periodic_job(method="aiccm", variant=...)`, or one of the
deprecated `jk_method` spellings `"real-gamma"`, `"aiccm2026dev-b"`,
`"chi"`) writes `method_status = "experimental"` into `[run]` from job
start, so a dry-run manifest, a crashed one and a completed one all
say the same thing. The value uses the test-gate maturity vocabulary
(`production`, `verified`, `under-review`, `experimental`). Routes
that carry no marker write no key; do not read an absent
`method_status` as a maturity claim. The same jobs also write
`aiccm_variant` (`"real-gamma"`, `"neutral-bloch"`, `"four-center"`,
`"chi"`: the formulation that ran) and `aiccm_selector`
(`"front-door"` for `method="aiccm"`, `"legacy-jk_method"` for a
deprecated spelling); for a front-door job `jk_method_requested`,
`_resolved` and `_executed` all record the variant's own route
value, and `[plan].method` records the inferred SCF reference
(`RHF`, `UHF`, `RKS` or `UKS`). The two neutral Gamma-CCM producers
also record their convention: `exchange_q0` and its applicability,
`ccm_construction`, `evaluation_representation`,
`lattice_vector_convention`, `bvk_nrep`, `bvk_n_cells` and the
executed mixing values; `neutral-bloch` adds the fit accounting
`lpq_pair_symmetry`, `gdf_pair_builds`, `gdf_pair_total`,
`gdf_pair_reduction_factor` and `rsgdf_tail_ke_cutoff_executed`.

Those finite-torus fields are listed above but their **values** carry the
distinctions the AICCM taxonomy exists to make, so read them as a vocabulary
rather than as free text:

* `exchange_q0` is the exchange gauge at the *q* = 0 singularity, and it is
  not cosmetic: `"BvK-ewald"` applies the finite-size Ewald constant,
  `"strict-zero"` deletes the free-space exchange monopole instead. The two
  differ by a term that is Ha-scale on a small torus (about 21.67 mHa at
  Γ on the measured MgO case) and tends to zero as the torus grows, so
  **two runs are comparable only if this key matches**.
  `exchange_q0_applicability` says whether that gauge was live at all
  (`"active"`, or inactive for a pure functional with no exact exchange).
* `ccm_construction` separates the constructions under ruling R1, and it is
  the key to check before comparing two AICCM numbers. The union-and-weight
  Wigner-Seitz construction (the 2014 lineage, `variant="four-center"`) and
  the neutral fitted torus are **different Hamiltonians**, not two
  representations of one; the neutral producers write
  `"none (neutral fitted-torus representation control)"` to say exactly that
  they are not making a union-and-weight claim.
* `evaluation_representation` separates the two *representations* of the one
  neutral construction; `real-gamma` writes
  `"real supercell-Gamma eigenproblem"` and `neutral-bloch` its per-*k* Bloch
  counterpart. Differing here is fine and expected; differing in
  `ccm_construction` is not.
* `bvk_nrep` is the Born-von-Kármán torus, and `bvk_n_cells` its cell count.
  On these routes it is **a supercell count, not a Bloch sampling mesh**: the
  Γ-centred mesh *is* the torus. A consumer that reads it as a *k*-mesh will
  mis-scale every extensive quantity by `bvk_n_cells`.

A job that ran a post-HF treatment also writes `aiccm_correlation` (what was
requested: `"mp2"`, `"ccsd"`, `"dlpno-mp2"` or `"dlpno-ccsd"`) and
`aiccm_correlation_route` (the citation route the driver actually stamped,
e.g. `"aiccm2026dev-a-ri-ccsd(t)"`). Read the second, not the first, when you
need the provenance: it carries the lineage (`-ri-` for the neutral fitted
torus, bare for the union-and-weight rows) and whether the perturbative
triples ran, neither of which the request determines. An SCF-only run writes
neither key.

**Privacy: hostname opt-out.** For runs you plan to share
publicly, pass `record_hostname=False` to `run_job`, or set the
`VIBEQC_NO_HOSTNAME=1` environment variable to opt out globally.
Either lever emits `hostname = "<redacted>"` (the field stays
present so the TOML shape is stable for parsers, only the value
is masked). Use the env var for public examples, paper artifacts,
and shared issue reproductions so local machine names do not leak.
Other manifest fields (CPU model, OS, memory, library versions) are
*not* redacted; the redaction is scoped to the hostname only.

Both levers cover **both** artefacts that carry the host: the
`.system` manifest and the `.qvf` archive's `provenance.hostname`.
That matters because the archive is the one you are most likely to
hand on, and it travels as a single file with no obvious place to
inspect. If you have archives written before this was fixed, check
them with:

```python
import json, zipfile
prov = json.loads(zipfile.ZipFile("output-h2o.qvf").read("manifest.json"))["provenance"]
print(prov.get("hostname"))
```

Read it from Python with stdlib `tomllib`:

```python
import tomllib
with open("output-h2o.system", "rb") as f:
    manifest = tomllib.load(f)
print(manifest["cpu"]["model"], manifest["run"]["wall_seconds"])
```

[Reproducing an example output](../tutorial/reference_outputs.md) walks through
using the manifest to compare two local runs of the same input.

### `output-h2o.xyz`, final geometry (v0.8.x+)

Always-on alongside `.molden`: an ASE-style **extended XYZ** with
the final geometry in Ångström, the SCF total energy on the
comment line as `energy=<Ha>`, and lattice metadata for periodic
jobs (Phase O5). Useful as input for downstream tools (a Gaussian
input generator, an ORCA parity run, a viewer that prefers
plain-text geometry over `.molden`):

```text
3
energy=-76.04939147 prop=energy
O    0.0000000000    0.0000000000    0.0000000000
H    0.0000000000    0.7569997744   -0.5184799434
H    0.0000000000   -0.7569997744   -0.5184799434
```

For periodic runs the comment line carries
`Lattice="ax ay az bx by bz cx cy cz" pbc="T T T"` so any
ASE-aware tool reads the cell correctly. Opt out with
`write_xyz_file=False` on `run_job` / `run_periodic_job`.

### `output-crystal.POSCAR` / `output-crystal.xsf` / `output-crystal.cif`, periodic structure

`run_periodic_job` emits three always-on crystal-structure
siblings (Phase O5 + D2, v0.8.x+):

* **`{stem}.POSCAR`**, VASP-5 POSCAR with selective-dynamics off;
  drop straight into VASP, pymatgen, or any tool that reads the
  POSCAR format.
* **`{stem}.xsf`**, XCrySDen XSF structure block (lattice in
  Ångström, atoms in fractional coordinates); read by VESTA and
  XCrySDen for crystal-structure visualisation.
* **`{stem}.cif`**, IUCr-standard Crystallographic Information
  File. Single `data_vibeqc` block with cell parameters in Å /
  degrees, `_symmetry_space_group_name_H-M = 'P 1'` and the
  identity symmetry op, and per-atom site labels indexed by
  element (`Mg1`, `O1`, `Mg2`, `O2`, …). Read by pymatgen, ASE,
  VESTA, Materials Project, and the Crystallography Open
  Database.

In addition to the four format-specific siblings,
`run_periodic_job` also emits a `{stem}.xyz` in Extended-XYZ form
(the ASE-convention `Lattice="..."` / `Properties=...` / `pbc=...`
keys carried in the comment line) so vanilla XYZ readers see the
geometry and ASE-aware readers recover the periodic cell.

All four files declare the **conventional** cell from the
`PeriodicSystem` input. Fixed-cell atomic optimization writes its relaxed
geometry to separate `.opt.POSCAR` / `.opt.xyz` files. Variable-cell
BIPOLE and GDF optimization currently fail closed. Opt out of any one with
`run_periodic_job(...,
write_poscar_file=False)` / `write_xsf_structure_file=False` /
`write_cif_file=False`.

### `{stem}.density.xsf`, periodic volumetric density (opt-in)

When ``write_density=True`` is passed to `run_periodic_job`,
vibe-qc evaluates the SCF electron density on a primitive-cell
real-space grid and writes it as a ``DATAGRID_3D_density`` XSF
block:

```python
# doc-audit: skip - output flags for an existing periodic calculation
run_periodic_job(
    system, basis,
    method="RHF",
    output="nacl",
    write_density=True,
    density_spacing_bohr=0.2,
)
# → nacl.density.xsf
```

Open in VESTA, XCrySDen, or `moltui nacl.density.xsf` to see
the isosurface inside the unit cell.  Control the voxel spacing
with ``density_spacing_bohr=`` (default 0.2 bohr ≈ 0.11 Å).

### `{stem}.bxsf`, band energies on a k-mesh (manual)

BXSF is not auto-emitted by `run_periodic_job`, you call
``write_bxsf`` yourself after a multi-k SCF run to produce
Fermi-surface data.  See
[Volumetric data: BXSF](volumetric_data.md#periodic-bxsf-files-regular-k-space-band-grids)
for the full recipe.

### `output-h2o.population.{txt,json}`, properties dump (v0.8.x+)

Two capability-aware default siblings carrying the population-analysis +
dipole data in machine-readable form. The matching block in `.out` is for
human reading; these files are the *parseable* form for
dashboards, regression scripts, and downstream analysis.

* **`.population.txt`**, tab-separated, seven `#`-commented
  sections: Mulliken charges, Löwdin charges, Hirshfeld charges, Mayer bond orders
  (top-N, default threshold 0.1), Wiberg bond indices (top-N),
  an NPA availability notice, and the dipole moment. Loadable into
  pandas / awk / spreadsheet importers with `#` as the comment
  marker.
* **`.population.json`**, one JSON object with top-level keys
  `mulliken` / `loewdin` / `hirshfeld` / `mayer` / `wiberg` / `npa` / `dipole`
  / `errors`, each shaped as a list of records (or a single
  object for `dipole` / `errors`). Drop-in for `json.load(...)`.

Wiberg indices are computed by the bond-analysis module. NPA is not yet
implemented: the `npa` array stays empty, an additional `unavailable` object
explains the missing NAO construction under `unavailable.npa`, and the text
section says `npa: not implemented`. This known limitation is not a
calculation error. See the [bond analysis guide](bond_analysis.md) for the
theory, provisional NBO APIs, EDA, orbital entanglement and citations.

With `iao_analysis=True`, molecular RHF/RKS/UHF/UKS jobs add an
`iao` object and a labeled IAO text section. This contains MINI/symmetric
IAO charges, optional alpha-minus-beta spin populations, dense spin-resolved
IAO-Wiberg bond orders, conventions and diagnostics. An unsupported analysis
has `available=false`, null numerical arrays and an `unavailable_reason`;
population summaries also record that reason under `unavailable.iao`.
The existing `wiberg` field keeps its original AO-based meaning.
See [IAO analysis](iao_population.md) for the API and QVF extension payload.

A property-computation failure (e.g. Mayer bond orders on a
near-singular overlap) on one section does NOT suppress the others
- partial success is preserved and the missing section is reported
in the `errors` dict / via a `# section: N/A - <error>` line.

`write_population_file=None` is capability-aware auto mode. Explicit `True`
guarantees the pair and fails before calculation when the selected route has
no compatible population convention; `False` opts out. On periodic meshes,
the multi-k exceptions are BIPOLE's lattice-density analysis and χ-CCM's
finite-torus analysis; other routes require an exact single-Gamma result.

(output-h2odensitycube--output-h2ohomolumocube--volumetric-data-opt-in-v08x)=
### `output-h2o.density.cube` / `output-h2o.{homo,lumo}.cube`, volumetric data (opt-in, v0.8.x+)

Gaussian-cube volumetric files for VMD / Avogadro / Jmol /
ChimeraX visualisation. Opt-in via `run_job(..., write_cube=...)`:

| `write_cube=` | Files written |
|---|---|
| `True` / `"density"` | `{stem}.density.cube` (total ρ(r)) |
| `"homo"` / `"lumo"` | The corresponding MO cube |
| `"homo-1"` / `"lumo+2"` / … | Offset MO labels |
| `<int>` | That MO index (0-based) |
| `["density", "homo", "lumo", 7]` | Any mix of the above |

Grid spacing + padding default to `0.2` / `4.0` bohr; pass
`cube_spacing=` / `cube_padding=` to tune. UHF/UKS density cubes
are the total density `D_α + D_β`. Each cube file is wrapped in
its own try/except so a single grid-evaluation failure on one MO
doesn't block the others.

```python
# doc-audit: skip - output flags for the molecule defined earlier in this guide
vq.run_job(mol, basis="6-31g*", method="rhf", output="h2o",
           write_cube=["density", "homo", "lumo"])
# → h2o.density.cube, h2o.homo.cube, h2o.lumo.cube
```

### `output-h2o.traj`, optimization trajectory

Emitted only when `optimize=True`. It's an ASE binary trajectory,
one frame per optimizer step, containing atomic positions + energy +
forces. View it as an animation with:

```sh
ase gui output-h2o.traj
```

Convert to XYZ for tools that prefer that format:

```sh
ase convert output-h2o.traj output-h2o-frames.xyz
```

Or iterate frames programmatically:

```python
from ase.io import read
frames = read("output-h2o.traj", index=":")
for step, atoms in enumerate(frames):
    print(step, atoms.get_potential_energy())
```

## Progress logging

Long calculations, multi-minute molecular SCFs on a big basis,
periodic bulk runs with EWALD_3D and a multi-k mesh, used to be
silent until the SCF returned. The headline question this answers:
*is the calculation stuck or actually running?*

`run_job` defaults to **live progress on**. The job emits a banner,
per-stage milestones, native molecular SCF iteration rows, requested
artifact-writing stages, and a final summary to stdout, flushing every line.
The canonical remote-job workflow therefore shows progress without extra
setup:

```sh
nohup python LiH.py > LiH.log 2>&1 &
tail -f LiH.log              # mirrors progress to the captured log
tail -f output-LiH.out       # calculation log; full SCF table after return
```

The captured stdout stream emits, in order:

- a banner naming the method, basis, functional, and thread count;
- one line per setup stage (`geometry_optimization`, `write_molden`,
  …) with elapsed wall-time on completion;
- the SCF banner ("Starting molecular SCF (RHF) …");
- one flushed row as each molecular RHF, UHF, RKS, or UKS iteration
  completes in the native driver;
- any requested large artifact stages, such as the final Molden export;
- a `Job total X.XXs - output written to …` summary line.

The ambient `.out` channel is a separate surface. It is line-buffered for
calculation milestones, then receives the canonical formatted SCF trace and
orbital tables when the native call returns. Native callback rows are not
duplicated into `.out`; use the captured stdout log or `.scf.jsonl` to follow
individual cycles while the call is active.

The same native callback writes an enabled `.scf.jsonl` structured-log row
and atomically updates the `.system` progress heartbeat before control returns
to the native loop. A job stopped by the scheduler therefore retains its last
completed iteration instead of an empty trace. `progress=False` silences the
terminal rows but does not suppress an explicitly requested structured log or
the manifest heartbeat.

An unrestricted post-SCF stability check has two distinct phases. The
orbital-Hessian Davidson analysis itself still has no per-iteration live
callback; when it finishes, its elapsed time is reported separately as
`SCF stability analysis`. If that analysis finds an instability and follows
it into another SCF, the reconvergence iterations do stream live. Their reset
iteration sequence receives a new `attempt` value. Structured rows also carry
a `phase` label for SCFs run by helpers such as geometry optimization or a
finite-difference Hessian. A terminal `scf_converged` record identifies the
canonical sequence with `selected_attempt`, or the contributing sequences of
an aggregating wrapper such as CPCM with `selected_attempts`.

### Disabling

Two equivalent levers, both restore the historical silent
behavior:

```python
# doc-audit: skip - per-call setting for an existing molecular input
# Per-call:
run_job(mol, basis="6-31g*", method="rhf", output="x",
        progress=False)
```

```sh
# Globally for a shell / batch job (only takes effect when
# `progress` is left at its default; explicit `progress=` kwargs win):
export VIBEQC_LIVE_LOGGING=0
```

A `VIBEQC_LIVE_LOGGING=0` env var is the right answer for batch
scripts that don't want to edit every input file. Explicit
`progress=True` / `progress=False` / a `ProgressLogger` instance
always wins, so a debugging session can re-enable progress for one
shell.

### Verbosity

The `verbose=` kwarg on `run_job` (added in v0.5.3) tunes how
much detail the live progress stream carries. Levels follow the
PySCF convention, each level is a strict superset of the one
below, so bumping `verbose` only adds output:

| level | what is emitted |
|---|---|
| 0 | silent, nothing live (the `.out` file is still written) |
| 1 | banner + warnings + final SCF status only |
| 2 | add per-stage milestones + `info()` lines |
| 3 | add per-stage timing on stage exit |
| 4 | **default**, add per-iteration SCF rows |
| 5 | add inline RSS-memory snapshots |
| 6+ | phase-level wall-clock breakdown live (overlaps the post-mortem `.perf` log) |

Two equivalent ways to set the level:

```python
# doc-audit: skip - per-call setting for an existing molecular input
# Per-call:
run_job(mol, basis="6-31g*", method="rhf", output="x",
        verbose=2)
```

```sh
# Globally for a shell / batch job (only takes effect when
# `verbose` is left at its default of None; explicit `verbose=`
# kwargs win):
export VIBEQC_VERBOSE=2
```

A junk env value (typo, leftover) silently falls back to the
package default, an overnight batch shouldn't die because of
`VIBEQC_VERBOSE=verbose`. Levels 1 and 2 are the right knob for
batch sweeps that want one summary per job without per-iter
spam; level 5+ is for debugging a specific run that's behaving
oddly. The level only gates the live emit, the
`format_scf_trace` block in `{output}.out` is unaffected and
always carries the full per-iteration history.

### Stdlib `logging` integration

When a project already pipes everything through `logging`
(rotating files, syslog, JSON-to-Loki, dictConfig), the
`use_logging=True` kwarg routes vibe-qc's progress through the
same stack instead of bare `stdout` writes:

```python
# doc-audit: skip - logging option for the molecule defined earlier in this guide
import logging
import vibeqc as vq

logging.basicConfig(level=logging.INFO)
vq.run_job(mol, basis="6-31g*", method="rhf", output="x",
           use_logging=True)
```

Banners, milestones, and the final SCF summary land at `INFO`;
per-iteration SCF rows at `DEBUG`; warnings at `WARNING`. The
logger name is `vibeqc.run_job`, so a `dictConfig` block can
target it specifically:

```python
import logging.config

logging.config.dictConfig({
    "version": 1,
    "handlers": {
        "scf_log": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "scf.log", "maxBytes": 10_000_000,
            "backupCount": 5,
        },
    },
    "loggers": {
        "vibeqc.run_job": {"handlers": ["scf_log"], "level": "INFO"},
    },
})
```

The verbose-level gate runs *before* the logging call, so
`verbose=2` + `use_logging=True` does not emit per-iter
`DEBUG` records, even if the active handler is set to
`DEBUG`. `progress=False` still wins as a hard kill switch, so
a silent run stays silent regardless of the active logging
config.

### Routing, `ProgressLogger`

For finer control, tee the same trace to a persistent file, or
thread a single logger through nested calls, instantiate a
[`vibeqc.ProgressLogger`](#progress-logging) directly:

```python
# doc-audit: skip - logger wiring for an existing periodic SCF setup
import vibeqc as vq

plog = vq.ProgressLogger(log_path="lih.progress.log", verbose=True)
vq.run_rhf_periodic_scf(system, basis, kpoints, opts, progress=plog)
```

The same `progress=` kwarg is accepted by every periodic SCF entry
point (`run_rhf_periodic_scf`, `run_rks_periodic_scf`, the
EWALD_3D variants, etc.), pass `True` to get stdout, or a logger
for routing.

### Coverage

Live per-iteration progress is available from Python-driven periodic loops
and from the four native molecular drivers when invoked through `run_job`:

| Entry point | Backend | Live per-iter? |
| --- | --- | --- |
| `run_rhf_periodic_scf` (EWALD_3D) | Python | yes |
| `run_rhf_periodic_gamma_scf` (EWALD_3D) | Python | yes |
| `run_rks_periodic_scf` / `run_rks_periodic_gamma_scf` (EWALD_3D) | Python | yes |
| `run_uhf_periodic_*_ewald3d` | Python | yes |
| `run_uks_periodic_*_ewald3d` | Python | yes |
| `*_periodic_*_scf` (DIRECT_TRUNCATED) | C++ | banner + post-hoc summary only |
| direct `run_rhf` / `run_uhf` / `run_rks` / `run_uks` calls | C++ | result trace on return |
| `run_job` with molecular RHF / UHF / RKS / UKS | wraps C++ | yes, native callback |

The native drivers still collect the canonical `result.scf_trace` used for
the final `.out` table. `run_job` scopes their diagnostics callback across the
whole molecular job, including convergence retries and native SCFs launched by
geometry optimization, atomization, Hessian, or property helpers. The final
manifest heartbeat is restored to the selected headline SCF or explicit
post-SCF correction, so a later helper cannot replace the job's terminal
result. Both molecular and periodic public runners restore the previous
execution-context callback after success or failure.

## Performance debugging

The post-mortem companion to live progress logging. Live logging
shows progress *during* a run; the perf log shows where the time
went *afterwards*. The two pair: live for "is the SCF stuck?",
perf for "why is my LiH / pob-TZVP run taking 20 minutes, is it J,
K, XC quadrature, or Bloch sums?".

`run_job` writes a `{output}.perf` sibling when you pass
`perf_log=True`:

```python
# doc-audit: skip - performance option for the molecule defined earlier in this guide
run_job(mol, basis="cc-pVDZ", method="rks", functional="pbe",
        output="output-h2o", perf_log=True)
# -> output-h2o.out / .molden / .system / .perf
```

The file is plain text, sortable by total wall-time. A typical
report:

```text
========================================================================
vibe-qc performance / debug log
========================================================================
  Total wall time:   19.599 s
  OMP threads:     12
  Phases tracked:  4

Phase summary  (sorted by total wall time, descending)
------------------------------------------------------------------------
  phase                                 n         wall          cpu  % wall    par
  ----------------------------------------------------------------------
  periodic.integrals_lattice            1      4.68 ms      9.86 ms    0.0%   0.18
  periodic.compute_nuclear_lattice      1      4.17 ms      8.82 ms    0.0%   0.18
  periodic.compute_overlap_lattice      1      0.30 ms      0.64 ms    0.0%   0.18
  periodic.compute_kinetic_lattice      1      0.16 ms      0.35 ms    0.0%   0.18

Memory snapshots  (RSS in MiB)
------------------------------------------------------------------------
  label                                  t (s)     RSS (MiB)
  ------------------------------------------------------------
  start_of_scf                           0.013         149.3
  end_of_scf                            19.598         152.1

SCF iterations
------------------------------------------------------------------------
  iter              E (Ha)           dE   ||[F,DS]||  DIIS   wall (s)
  ----------------------------------------------------------------------
     1      -28.9250484403        --     4.437e-02     -      3.895
     2      -28.9261305496  -1.082e-03  3.304e-02     1      7.747
     ...
========================================================================
```

Sections, in order:

- **Header**, total wall, OMP thread count, phase count.
- **Phase summary**, one row per `PerfScope` opened during the
  run, sorted by wall-time. The `par` column is *parallelism* =
  CPU time / (wall time × threads); 1.0 means perfect OpenMP
  scaling, < 0.7 flags an under-parallelised hot path.
- **Under-parallelised hot paths**, auto-flag block listing any
  phase that consumed more than 5% of wall time and ran at
  parallelism < 0.7×. The under-parallelised hot paths users care
  about.

- **Memory snapshots**, labeled RSS samples (`start_of_scf`,
  `end_of_scf`, …) so you can see RSS growth caused by the
  Fock build separate from the basis-set / pre-flight overhead.
- **SCF iterations**, per-iteration table (energy, ΔE,
  ‖[F,DS]‖, DIIS subspace, wall-time-since-SCF-start). The
  post-mortem analogue of live progress logging's per-iter
  emission.

The thread denominator comes from the native OpenMP runtime and explicit job
allocation limits such as `OMP_NUM_THREADS`, `SLURM_CPUS_PER_TASK`, or
`PBS_NP`; the tightest applicable limit wins. Host-wide available-core counts
are not treated as allocated CPUs. If no runtime/allocation signal is
available, the report conservatively uses one thread instead of guessing from
the machine size.

### Three ways to enable

```sh
# (1) Env var — common case for one-off jobs:
VIBEQC_PERFLOG=output.perf python my-calc.py
```

```python
# doc-audit: skip - instrumentation examples reuse an existing calculation setup
# (2) Programmatic context manager — wrap a region:
import vibeqc as vq

with vq.perf_log("output.perf"):
    result = vq.run_rhf(mol, basis, opts)
    hess = vq.compute_hessian_rhf_analytic(...)
# All work inside the block accumulates into the same tracker;
# the report is written when the block exits.

# (3) run_job kwarg — one-shot:
vq.run_job(mol, basis="cc-pVDZ", method="rks", functional="pbe",
           output="x", perf_log="x.perf")  # explicit path
vq.run_job(mol, basis="cc-pVDZ", method="rks", functional="pbe",
           output="x", perf_log=True)      # → x.perf sibling
```

The three knobs feed the same `vibeqc.PerfTracker` accumulator.
Explicit `perf_log=` always wins over the env var; off by default.

### JSON perf artefact: `perf_log="....json"`

The text report is written for humans. When you are harvesting
timings across hundreds of jobs, name the target `*.json` instead
and the **same tracker** is written as a JSON object:

```python
# doc-audit: skip - performance option for the molecule defined earlier in this guide
run_job(mol, basis="cc-pVDZ", method="rks", functional="pbe",
        output="output-h2o", perf_log="output-h2o.perf.json")
```

The suffix is the only switch: `.perf`, `.perf.txt`, or any other
name gets the text report, and only `.json` gets JSON. The `.system`
manifest's `plan.files` entry for `role = "perf"` declares
`format = "json"` accordingly, so a harvest can tell which
rendering it is about to open without sniffing the bytes.

Schema `vibeqc.perf/1`. Every duration is a raw float in **seconds**
and every memory figure a raw float in **MiB**, with no unit suffixes
to re-parse:

```json
{
  "schema": "vibeqc.perf/1",
  "pid": 2505082,
  "threads": 12,
  "total_wall_s": 19.599,
  "n_phases": 4,
  "phases": [
    {"name": "scf.rks", "n_calls": 1, "wall_s": 13.42, "cpu_s": 148.1,
     "pct_wall": 68.5, "parallelism": 0.92}
  ],
  "memory_snapshots": [
    {"label": "end_of_scf", "t_s": 14.02, "rss_mb": 1830.5}
  ],
  "scf_iters": [
    {"iter": 1, "energy": -76.0217, "dE": null, "grad": 1.2e-3,
     "diis": 0, "wall_s": 1.83}
  ]
}
```

| Field | Meaning |
|---|---|
| `schema` | Format tag. The integer bumps only when a field is removed or changes meaning; new fields are additive. |
| `pid` | OS process id that produced the run. |
| `threads` | OpenMP thread count snapshotted at tracker construction. |
| `total_wall_s` | Wall seconds from tracker construction to write. |
| `phases[]` | One entry per `PerfScope`, sorted by `wall_s` descending. |
| `phases[].parallelism` | `cpu_s / (wall_s × threads)`; 1.0 is perfect OpenMP scaling. `null` when `wall_s` is zero, never a fabricated `0.0`. |
| `memory_snapshots[]` | Labeled RSS samples, `rss_mb` in MiB, `t_s` relative to tracker start. |
| `scf_iters[]` | Per-iteration rows verbatim as the SCF driver recorded them. An unmeasured `wall_s` is **absent**, not `0.0`. |

`phases[].pct_wall` is `wall_s` as a percentage of `total_wall_s`,
matching the `% wall` column of the text report.

### Reading the report programmatically

The tracker is a plain Python object, call sites that want to
script around the perf data can read it directly:

```python
# doc-audit: skip - reads the tracker around an existing calculation setup
import vibeqc as vq

with vq.perf_log() as tracker:
    vq.run_rhf(mol, basis, opts)

for phase in sorted(tracker.phases.values(),
                    key=lambda p: -p.wall_s):
    print(f"{phase.name}: {phase.wall_s:.3f}s "
          f"({phase.n_calls} calls)")
```

`tracker.phases`, `tracker.scf_iters`, and
`tracker.memory_snapshots` are public attributes,
[full API in `vibeqc.PerfTracker`](../api/index.md).

### Coverage

What's instrumented today:

| Phase | Driver | Live perf rows? |
| --- | --- | --- |
| `run_job.total` | wraps everything inside `run_job` | yes |
| `geometry_optimization` | ASE BFGS | yes |
| `basis_set_construction` | libint | yes |
| `scf.{rhf,uhf,rks,uks}` | C++ molecular SCF (one row total) | one row total |
| `scf.{uhf,uks}.stability_analysis` | post-convergence internal-stability Davidson solve, charged by the C++ driver | one row total, only when the analysis ran |
| `write_molden` | molden writer | yes |
| `periodic.integrals_lattice` (+ S/T/V sub-scopes) | Python lattice integrals | yes |
| `periodic.*_periodic_*_ewald3d` SCF iteration loop | Python | per-iter wall in SCF table |

The C++ kernels (`compute_eri`, `build_coulomb`, `build_exchange`,
`xc_eval`, `diag_k`, `s_inverse_sqrt_complex`, `bloch_sum`) are
not instrumented yet, those scopes live in C++ and need a
compile-time `#ifdef VIBEQC_PERFLOG` hook to keep release builds
zero-cost. They arrive in a v0.5.2.x patch.

## Structured machine-readable log

A third observability surface (after the human-readable `.out` and
the post-mortem `.perf`): one JSON record per SCF transition,
written line-flushed to `{output}.scf.jsonl` so dashboards and
analysis scripts can ingest convergence data without screen-
scraping the text log.

Format: NDJSON (one JSON object per line, no enclosing array).
Every record carries `"event"` plus a per-event payload. Event
names + field names are append-only, never renamed or removed,
so v0.6 callers stay forward-compatible with v0.7+ additions.

A typical sequence for a successful molecular SCF (one JSON record
per line, the format Pygments calls `text`, not `jsonl`):

```text
{"event":"banner","timestamp":"...","vibeqc_version":"0.6.0.dev0","libint":"2.13.1","libxc":"7.0.0","spglib":"2.7.0","run_fingerprint":"abc1234567890abc"}
{"event":"job_start","timestamp":"...","method":"rhf","basis":"sto-3g","functional":null,"optimize":false,"threads":12,"n_atoms":2,"charge":0,"multiplicity":1,"n_electrons":2,"output_stem":"h2"}
{"event":"memory_estimate","timestamp":"...","total_gb":0.15,"raw_total_bytes":104858640,"headroom_factor":1.5,"by_category":{"ERI tensor":128,"Fock + density + 1e":256,"DIIS history":512,"MO workspace":144,"Python runtime + NumPy overhead":104857600}}
{"event":"scf_iter","timestamp":"...","iter":1,"energy":-0.71,"dE":null,"grad_norm":4.7e-16,"diis_subspace":1}
{"event":"scf_iter","timestamp":"...","iter":2,"energy":-1.117,"dE":-0.398,"grad_norm":0.0,"diis_subspace":2}
{"event":"scf_converged","timestamp":"...","n_iter":3,"energy":-1.1167143251,"converged":true}
{"event":"properties","timestamp":"...","mulliken":[6.7e-16,-4.4e-16],"loewdin":[7.8e-16,0.0],"dipole":{"x":0.0,"y":0.0,"z":-7.8e-16,"total":7.8e-16,"total_debye":2.0e-15}}
{"event":"job_end","timestamp":"...","total_wall_s":0.099,"scf_wall_s":0.004,"opt_wall_s":0.0,"n_iter":3,"converged":true,"energy":-1.1167143251,"out_path":"h2.out"}
```

Strict JSON: NaN / ±Infinity are coerced to `null` so the file
parses cleanly with `jq`, `jq -c '.'`, and any other strict-JSON
tool. The first iteration's `dE` is `null` (not 0), same
placeholder semantics as the human-readable trace.

The `run_fingerprint` field is a 16-character hex digest of the
identifying inputs (method + basis + functional + atoms + charge
+ multiplicity). Two runs with the same fingerprint are
calculating the same thing, useful for "did this run change?"
checks in CI dashboards.

### Three ways to enable

```sh
# (1) Env var — common case for one-off jobs:
VIBEQC_STRUCTURED_LOG=output.scf.jsonl python my-calc.py
```

```python
# doc-audit: skip - structured-logging examples reuse an existing calculation setup
# (2) Programmatic context manager — wrap a region:
import vibeqc as vq

with vq.structured_log("output.scf.jsonl"):
    vq.run_rhf(mol, basis, opts)
# All work inside the block emits to the same file; periodic SCFs
# emit per-iter rows live via the same context-var funnel that
# vibeqc.ProgressLogger uses.

# (3) run_job kwarg — one-shot:
vq.run_job(mol, basis="cc-pVDZ", method="rks", functional="pbe",
           output="x", structured_log=True)        # → x.scf.jsonl
vq.run_job(mol, basis="cc-pVDZ", method="rks", functional="pbe",
           output="x", structured_log="other.jsonl")  # explicit
```

Off by default, `run_job` only writes the file when the caller
opts in. Explicit `structured_log=` always wins over the env var.

### Tail-friendly

The file is line-flushed: a `tail -f output.scf.jsonl` shows
records as they're emitted (one per SCF iteration during the
SCF, plus banner / properties / job_end at boundaries). Pair
with `jq` for a live convergence monitor:

```sh
tail -f output.scf.jsonl | jq -c 'select(.event == "scf_iter") | [.iter, .energy, .dE, .grad_norm]'
```

### Coverage

| Event | Source | When emitted |
| --- | --- | --- |
| `banner` | `run_job` | first record, carries linked-library versions + run_fingerprint |
| `job_start` | `run_job` | after method resolution, before any work |
| `memory_estimate` | `run_job` | after each route-specific memory pre-flight; an automatic dense RKS TRAH retry emits a second record with `phase="scf.rks.trah_retry"` |
| `scf_iter` | C++ molecular RHF/UHF/RKS/UKS callbacks in `run_job` + Python periodic SCFs; terminal-trace replay for drivers without callbacks | per SCF iteration, flushed before the next cycle; native rows include `phase` and `attempt` |
| `scf_converged` | `run_job` / `ProgressLogger.converged` funnel | once after the SCF loop; native runs identify one chosen sequence as `selected_attempt`, or concatenated contributing sequences as `selected_attempts` |
| `properties` | `run_job` | post-SCF, when properties succeed; carries Mulliken / Löwdin / dipole |
| `scf_failed` | `run_job` | when the C++ SCF raises; the exception still propagates after the dump |
| `job_end` | `run_job` | last record, total_wall_s + out_path. `energy` is the bare SCF/solver energy; when a post-SCF correction is active (D3-BJ / D4 / gCP / SRB) the record also carries the explicit split `e_scf`, `e_dispersion`, `e_total` (+ `e_gcp` / `e_srb` for 3c composites). Harvesters should prefer `e_total` when present. |

## Crash dumps

When an SCF fails ungracefully, raised exception (NaN in the
density, severe linear dependence, OOM) or runs to `max_iter`
without converging, `run_job` writes a snapshot to
`{output}.dump` plus binary attachments. Three-line bug report:
attach `output.dump` + `output.dump.density.npy` + the input
script and the maintainer can reconstruct the exact failing
state via `vibeqc.load_dump`.

The dump is **on by default**: post-mortem reproducibility costs
zero bytes on success and saves a re-run on failure. Disable
per-call with `crash_dump=False` or globally with
`VIBEQC_NO_CRASH_DUMP=1` in the environment.

A custom nested `crash_dump=` path creates its parent before writing either
the TOML body or its density/Fock/MO attachments. If any optional diagnostic
file cannot be written, the calculation's original failure path is preserved
and the I/O problem is surfaced through the standard structured output
warning rather than an untracked console print.

### File layout

For `output="output-h2o"` and a NaN failure at SCF iteration 5:

* `output-h2o.dump`, TOML with `[crash]`, `[scf.last_iter]`,
  `[geometry]`, `[molecule]`, `[options]`, `[hint]`, and
  `[attachments]` sections.
* `output-h2o.dump.density.npy`, last-iteration density matrix.
* `output-h2o.dump.fock.npy`, last-iteration Fock matrix
  (when present).
* `output-h2o.dump.mo.npy`, current MO coefficients
  (when present).

A typical `.dump` body (real `.dump` files are TOML; the example
below is rendered as plain text because the illustrative `...` and
`nan` placeholders below aren't valid TOML literals on their own):

```text
[crash]
when = "2026-04-30T20:27:26-05:00"
phase = "scf_iteration_5"
exception_type = "RuntimeError"
exception = "NaN in density matrix"
n_iters_completed = 4

[scf.last_iter]
iter = 4
energy = -74.123
delta_e = nan
grad_norm = 1.7e+02
diis_subspace = 4

[geometry]
atoms = [
  { Z = 8, x = 0.0, y = 0.0, z = 0.0 },
  ...
]

[molecule]
charge = 0
multiplicity = 1
n_atoms = 3

[options]
max_iter = 100
damping = 0.0
...

[hint]
likely_cause = "DIIS instability — try damping=0.5, level_shift=0.5, or DIIS=False to fall back to plain Roothaan iterations."

[attachments]
files = "output-h2o.dump.density.npy, output-h2o.dump.fock.npy, output-h2o.dump.mo.npy"
```

The `[hint]` block runs a small heuristic against the exception
text + type to produce a one-line `likely_cause`. It's
best-effort and never blocks the dump if the keyword search
doesn't match anything specific.

### Reproducer recipe

```python
import vibeqc as vq

dump = vq.load_dump("output-h2o.dump")
density = dump["arrays"]["density"]   # numpy ndarray, last iteration
options = dump["options"]              # dict, ready to feed back in
print(dump["crash"]["phase"], "→", dump["hint"]["likely_cause"])
```

`vibeqc.load_dump` returns a nested dict shaped like the TOML
sections, plus an extra `"arrays"` key carrying every sibling
`.dump.<name>.npy` rebuilt with `numpy.load`. Pair with the
input script in the bug report and the maintainer reconstructs
the failing state bit-for-bit.

### Failure modes that trigger a dump

| Failure | Where the dump fires |
| --- | --- |
| C++ SCF raises (NaN, lin-dep, memory error) | `run_job`'s try/except wraps the SCF call → dump + re-raise |
| SCF returns non-converged (max_iter exceeded) | `run_job` checks `result.converged` after success; dumps if False, returns the result normally (does NOT raise) |
| Pre-SCF errors (basis-set construction, memory pre-flight abort) | not currently captured, the .out file holds the error message; a future patch will widen the dump scope |

The exception path always re-raises, `crash_dump=True` does NOT
swallow failures; it just makes them debuggable. The max-iter
path returns the non-converged result so callers that explicitly
want to inspect a failed iterate keep working.

## `run_job` parameters

| Parameter | Default | Purpose |
|---|---|---|
| `molecule` | required | `Molecule` in bohr coordinates |
| `basis=` | required | libint-recognized basis name |
| `method=` | `"auto"` | `"rhf"` / `"uhf"` / `"rks"` / `"uks"` / `"auto"` |
| `functional=` | `None` | XC functional name for RKS/UKS (e.g. `"PBE"`, `"B3LYP"`) |
| `output=` | `"output"` | path stem; files become `{output}.out`, `{output}.molden`, `{output}.traj` |
| `optimize=` | `False` | run BFGS (via ASE) before the final SCF |
| `fmax=` | `0.05` | optimizer convergence in eV/Å (ASE convention) |
| `max_opt_steps=` | `200` | optimizer iteration limit |
| `write_molden_file=` | `None` (auto) | emit `.molden` when the route exposes a Gaussian AO wavefunction; explicit `True` guarantees it or fails before calculation |
| `write_xyz_file=` | `True` | emit `{output}.xyz` final geometry (Å, plus `energy=<Ha>` in the comment line) |
| `write_population_file=` | `None` (auto) | emit `{output}.population.{txt,json}` on compatible routes; explicit `True` guarantees the pair or fails before calculation |
| `iao_analysis=` | `False` | request molecular determinant IAO charges, spin populations and IAO-Wiberg bonds independently of localization/QVF |
| `iao_bond_threshold=` | `0.05` | text display threshold for IAO bond pairs; leaves the dense API/JSON result unchanged |
| `write_cube=` | `False` | volumetric cubes; `True` / `"density"` / `"homo"` / `"lumo"` / int / list (see [cube section](#output-h2odensitycube--output-h2ohomolumocube--volumetric-data-opt-in-v08x)) |
| `cube_spacing=` / `cube_padding=` | `0.2` / `4.0` (bohr) | grid spacing + padding for cubes; ignored when `write_cube=False` |
| `output_qvf=` | `True` | bundle structure + density + the `write_cube=` grids + basis/coefficients into one `{output}.qvf` archive for vibe-view (vibe-qc's native visualization format) |
| `citations=` | `True` | emit `{output}.bibtex` + `{output}.references` siblings and append a `## References` block to `.out` |
| `dry_run=` | `False` | pre-flight only: build the OutputPlan, write `{output}.system` with `[outputs].status="dry_run"`, return `None` without running the SCF. `VIBEQC_DRY_RUN=1` env var does the same; used by `vq submit --vibeqc-preflight` |
| `progress=` | `None` (resolves to live-on) | live progress logger; `False` to silence stdout, a `ProgressLogger` to route, or set `VIBEQC_LIVE_LOGGING=0` globally for batch scripts |
| `verbose=` | `None` (resolves to 4) | integer 0..9 (PySCF convention) gating how much live detail emits; 0 silent, 4 default with per-iter rows, 5 adds memory snapshots; `None` defers to `VIBEQC_VERBOSE=N` env var |
| `use_logging=` | `False` | route progress through `logging.getLogger("vibeqc.run_job")` instead of bare `stdout`; composes with stdlib `RotatingFileHandler` / syslog / `dictConfig` |
| `perf_log=` | `None` | post-mortem perf breakdown; `True` writes `{output}.perf`, a path writes there explicitly, `None` defers to `VIBEQC_PERFLOG=path` env var |
| `structured_log=` | `False` | machine-readable NDJSON; `True` writes `{output}.scf.jsonl`, a path writes there explicitly, `None`/`False` defers to `VIBEQC_STRUCTURED_LOG=path` env var |
| `crash_dump=` | `True` | post-mortem dump on SCF failure (`{output}.dump` + `.dump.density.npy` etc.); pass `False` (or set `VIBEQC_NO_CRASH_DUMP=1`) to disable |
| `record_hostname=` | `True` | record live hostname in `{output}.system`; pass `False` (or set `VIBEQC_NO_HOSTNAME=1`) to emit `hostname = "<redacted>"` |
| `rhf_options=` / `uhf_options=` / `rks_options=` / `uks_options=` | `None` | fine SCF control, override the relevant options struct |

`method="auto"` resolves to:

- `functional` set + multiplicity 1 → RKS
- `functional` set + multiplicity ≥ 2 → UKS
- no `functional` + multiplicity 1 → RHF
- no `functional` + multiplicity ≥ 2 → UHF

The return value is the underlying SCF result object (`RHFResult`,
`UHFResult`, `RKSResult`, or `UKSResult`), so you can continue in
Python after the call to inspect MO coefficients, density matrices, or
feed the result to downstream post-SCF analysis.

## When to use `run_job` vs the low-level drivers

`run_job` optimises for the *80% case*: one method on one geometry,
producing a log and an orbital file. If you want to

- sweep over basis sets / functionals in one script,
- compose your own output format,
- or call the SCF drivers with non-default numerical integration grids,

reach for the low-level API (`run_rhf`, `run_rks`, etc.) and
`format_scf_trace` directly. For periodic systems, the matching
high-level entry point is `run_periodic_job`, same artefact
family (`.out` / `.system` / `.molden` / extended `.xyz` /
`.POSCAR` / `.xsf` / `.bibtex` / `.references`), same dry-run +
citation surfaces, accepts a `PeriodicSystem` instead of a
`Molecule`. Everything `run_job` does internally is
re-usable via the same public [API](../api/index.md).
