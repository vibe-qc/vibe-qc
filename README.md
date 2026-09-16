<!-- attribution -->

_Created and maintained by Dr. Michael F. Peintinger._

# vibe-qc

**vibe-qc** — quantum-chemistry code for molecules and solids.
Python frontend over a C++17 core backed by
[libint](https://github.com/evaleev/libint) (Gaussian integrals),
[libxc](https://libxc.gitlab.io/) (500+ XC functionals), and
[spglib](https://github.com/spglib/spglib) (crystal symmetry). Licensed
under [MPL 2.0](LICENSE).

> **Naming.** The project / brand is spelled **vibe-qc** (with hyphen,
> matching the domain [vibe-qc.com](https://vibe-qc.com)); the Python
> package is imported as `import vibeqc` (no hyphen — Python
> identifiers can't carry dashes). Both names refer to the same code.
> vibe-qc is **not on PyPI yet** — install by cloning this repo and
> running `./scripts/install.sh --dev` (full instructions below). The script
> creates and verifies the managed virtualenv for you.

Latest release line: **v0.17.x** (codename *Tew's Tern*), opened
2026-09-09; the current release is **v0.17.1**, cut 2026-09-10. The line lays
symmetry and periodic groundwork: shared group and
space infrastructure, audited BIPOLE Seitz supports and AO maps, exact
physical quartet supports for BIPOLE SR and independent product supports
for the native LR Gram kernel, and bridged exact-mesh HF long-range
contractions. These are supports and audits rather than a production
route -- production HF and complete source integration remain open, so
this line does not ship periodic local correlation. Alongside them,
v0.17.0 builds the COSMO FINE cavity in its molecule-fixed frame by
default with the analytic nuclear derivative running through that frame,
and adds the 16 cc-pVnZ-PP orbital sets.

The preceding **v0.16.x** line (codename *Pople's Puffin*), cut
2026-09-08, carried the structural change: vibe-qc, vibe-view,
vibe-queue and qvf became four independent repositories that can be
adopted separately; the table further down lists all four.
v0.16.0 also corrected the semiempirical periodic electrostatics
(GFN2-SECCM shell gamma and periodic Gamma GFN2-xTB as real lattice sums
over the Wigner-Seitz image records, and a converging SCC-DFTB gamma
lattice sum), unified initial-guess selection and construction, and added
analytic COSMO FINE cavity nuclear derivatives and the assembled analytic
CFC energy gradient.

Carried in from the v0.15 line: DLPNO local correlation (DLPNO-MP2, the
DLPNO-CCSD per-pair solver, DLPNO-(T1), open-shell U-DLPNO-MP2 /
U-DLPNO-UCCSD(T) pilots), a TD-DFT engine (Casida + TDA for
RHF/RKS/UHF/UKS with natural transition orbitals and UV/Vis spectrum
reconstruction), production CASSCF and exact IC-CASPT2 analytic
gradients, an eigensolver framework (Davidson, LOBPCG, experimental
Jacobi-Davidson/GPLHR), COOP/COHP + QTAIM + periodic Mayer bond-order
analysis, periodic COSX and Mixed Density Fitting, the AICCM cyclic
cluster model (experimental), the MSINDO semiempirical engine with
periodic CCM, the MACE ML interatomic potential, optional MPI
parallelisation, well-tempered metadynamics, and velocity-Verlet
molecular dynamics. See [`CHANGELOG.md`](CHANGELOG.md) and
[`docs/roadmap.md`](docs/roadmap.md) for the full timeline.

Long-term target: the **cyclic cluster model** for solid-state
calculations — vibe-qc's `v2.0` feature. `v1.0` delivers everything a
standard QC program offers, in both molecular and periodic forms. The
full milestone ladder lives in [docs/roadmap.md](docs/roadmap.md).

## Current capabilities

### Molecular (validated against [PySCF](https://pyscf.org/) and [ORCA 6.1.1](https://orcaforum.kofo.mpg.de/))

- **Hartree–Fock** — RHF (closed-shell) and UHF (open-shell) with
  DIIS / **EDIIS+DIIS** (default since v0.8.0) / KDIIS / quadratic
  fallback / Newton / TRAH / SOSCF, SAD / SAP / AUTO / HCORE / Hückel
  initial guesses, analytic nuclear gradients.
- **Kohn–Sham DFT** — RKS and UKS with LDA, GGA, and hybrid
  functionals via [libxc](https://libxc.gitlab.io/) (500+); analytic
  gradients for LDA / GGA / hybrid. The `b3lyp` keyword resolves to
  the **VWN5 / ORCA convention** as of v0.8.0; pass `b3lyp/g` for
  the Gaussian-VWN3 variant.
- **Møller–Plesset 2** — RMP2 and UMP2, with SCS / SOS decomposition,
  RI-MP2 with optional auxiliary-fit residual diagnostic.
- **Double hybrids** — **B2PLYP**, **DSD-PBEP86** (Grimme 2006,
  Kozuch-Martin 2011) via `run_b2plyp` / `run_dsd_pbep86` /
  `run_double_hybrid`.
- **MACE ML interatomic potential** — `method="mace"` drives
  [ACEsuit MACE](https://github.com/ACEsuit/mace) pre-trained
  *E(3)*-equivariant GNN potentials for fast geometry optimization,
  MD pre-screening, periodic energy/forces/**stress** + cell
  relaxation, and `run_neb` reaction paths. Optional `[mace]` extra
  (Python ≤3.13); foundation-model weights fetched on demand (MIT
  models ungated, ASL academic models gated). An attributed external
  pre-trained engine, not a vibe-qc total energy; see the
  [external-model guide](https://vibe-qc.com/docs/user_guide/mlip.html).
- **Density fitting + RIJCOSX** — `JKBuilder` polymorphic Fock build,
  three concrete kernels (`FourIndexJKBuilder`, `DFJKBuilder`,
  `COSXJKBuilder`); RIJCOSX validated to **0.13 mHa vs ORCA 6.1.1**.
- **ECPs** — libecpint 1.0.7; five Stuttgart MDF libraries
  (`ecp10mdf` / `ecp28mdf` / `ecp46mdf` / `ecp60mdf` / `ecp78mdf`)
  plus Hay-Wadt `lanl2dz` shipped on `main`. Pt UHF/LANL2DZ
  reference: −118.227 Ha.
- **Implicit solvation** — **CPCM / COSMO** continuum model (v0.9.0
  S1 landing); single-point energies + cavity surface +
  contribution-to-energy reporting.
- **Post-SCF properties** — Mulliken / Löwdin charges, Mayer bond
  orders, dipole moments, all SCF flavors; auto-written to
  `output-*.population.{txt,json}`.
- **Dispersion** — **DFT-D3 / D3(BJ) / D4** (Grimme et al. 2010 /
  2011, Caldeweyher-Bannwarth-Grimme 2019) wired through `run_job`
  and `run_b2plyp`, with analytic forces.
- **ASE integration** — drop-in
  [Calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html)
  for geometry optimization, MD, vibrations, structure I/O (XYZ, CIF,
  PDB, POSCAR).
- **Analytic Hessian** — for RHF / RKS-LDA / RKS-GGA / RKS-hybrid /
  UHF / UKS-LDA. Meta-GGA + UKS-GGA finite-difference fallbacks.

### Periodic (1D / 2D / 3D)

- **Γ-only RHF + KS-DFT via native Gaussian density fitting** —
  `run_rhf_periodic_gamma_gdf` / `run_rks_periodic_gamma_gdf` (LDA /
  GGA / hybrids — incl. PBE0, B3LYP, PW1PW); the production path for
  hybrid DFT on solids at Γ.
- **Multi-k closed-shell RHF / KS-DFT (Ewald-3D)** —
  `run_rhf_periodic`, `run_rks_periodic` with Monkhorst–Pack k-mesh.
- **Fermi-Dirac smearing** — `smearing_temperature` field with
  numeric / named / auto resolution; `"metal"` preset = 0.005 Ha.
  Reports free energy, internal energy, entropy contribution, Fermi
  level.
- **3D Ewald** — point-charge Madelung (NaCl / CsCl / ZnS / sc-jellium
  reproduced), erfc-screened nuclear attraction, Gaussian-charge V(g)
  via analytical+grid splitting.
- **CRYSTAL-style BIPOLE periodic RHF** — `run_pbc_bipole_rhf`,
  research-preview: exact Ewald-J production totals plus cell-multipole
  moments and IDIPC dispatch. CRYSTAL's reported `EXT EL-POLE`
  multipole/penetration decomposition remains gated; its former
  reciprocal-only helper now fails closed. See
  [`docs/user_guide/bipole.md`](docs/user_guide/bipole.md).
- **Visualization** — every job writes a **QVF** archive (`.qvf`) by
  default: one file carrying structure, wavefunction, scalar fields,
  spectra, bands, trajectories, provenance and citations, opened with
  the separately installed **vibe-view** viewer in a browser, in a terminal over
  SSH, as a desktop app, or headlessly from Python. Plus band
  structure + DOS helpers with matplotlib plotters, and extended-XYZ +
  POSCAR + XSF / BXSF writers for VESTA / XCrySDen. See
  [`docs/visualization.md`](docs/visualization.md).
- **Space-group analysis** via [spglib](https://github.com/spglib/spglib).

### Basis sets

- **142 bundled bases** under `python/vibeqc/basis_library/basis/` —
  ~90 standard molecular sets from libint (STO-nG, 6-31G\*\*,
  cc-pVXZ, def2-TZVP, ANO-RCC, …) plus the pob-* solid-state family
  and a curated set of auxiliary bases for JK-fit / RI-fit.
- **Peintinger–Vilela Oliveira–Bredow (pob-\*) solid-state bases** —
  `pob-TZVP`, `pob-TZVP-rev2`, `pob-DZVP-rev2` for H–Br, designed to
  avoid small-exponent linear dependencies in periodic calculations
  (Peintinger 2013, Vilela Oliveira 2019).
- **CRYSTAL-format parser** (`vibeqc.basis_crystal`) reads any
  CRYSTAL-style per-element file and emits NWChem/.g94 for libint.

### Output / tooling

- **`vibeqc.output` package** (v0.8.x) — unified output coordinator
  with declarative `OutputPlan` (the contract for which files a job
  will produce), `.system` TOML manifest with `[plan]` and
  `[outputs]` sections, atomic-write status updates for the
  `vq`-queue liveness detection. See
  [`docs/user_guide/output_files.md`](docs/user_guide/output_files.md).
- **Automatic citations** — every `run_job` writes a `{stem}.bibtex`
  and `{stem}.references` next to the `.out`, walking a TOML-backed
  database of every paper to cite for the functional / basis /
  dispersion / library combination actually exercised. See
  [`docs/user_guide/citations.md`](docs/user_guide/citations.md).
  Re-emit with `vibeqc-cite <stem>`.
- **`vq` queue** — companion job queue at
  [vibe-queue](https://github.com/vibe-qc/vibe-queue); submit to a remote machine over SSH
  with vibe-qc-aware dry-run preflight (`vq submit
  --vibeqc-preflight`). SLURM-style verb aliases. See
  [`docs/user_guide/queue.md`](docs/user_guide/queue.md).
- **OpenMP parallelism** across every hot kernel — integrals,
  gradients, Fock / XC builds, MP2 transform, lattice sums, Ewald,
  periodic XC. Bit-identical results across thread counts.
- **Memory pre-flight** — `estimate_memory()` + abort-on-overflow
  with `memory_override=` escape hatch.
- **Runtime banner** — vibe-qc version, codename, git provenance,
  linked-library versions; prepended to SCF trace output for
  reproducibility.
- **~8700 regression test functions** across 675 test files; runs in
  ~3 min on a modern laptop, ~50 s with `-n auto`.

## Quick start

A full `run_job` call — text log, MO file, geometry, citations,
manifest:

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("h2o.xyz")              # Å in file → bohr
run_job(
    mol,
    method="rks", functional="PBE0", basis="def2-tzvp",
    dispersion="d3bj",
    optimize=True,
    output="output-h2o-pbe0",
)
# → output-h2o-pbe0.out / .system / .molden / .xyz / .bibtex /
#   .references / .population.txt / .population.json / .traj
```

The lower-level driver gives you direct access to the result object:

```python
from vibeqc import Atom, Molecule, BasisSet, run_rhf, print_banner

print_banner()

mol = Molecule([
    Atom(8, [0.0,  0.0,  0.0]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])
basis = BasisSet(mol, "pob-tzvp")
result = run_rhf(mol, basis)
print(result.energy)          # -76.0493914670 Ha
```

Via ASE:

```python
from ase.build import molecule
from ase.optimize import BFGS
from vibeqc.ase import VibeQC

atoms = molecule("H2O")
atoms.calc = VibeQC(method="rks", functional="PBE", basis="6-31g*")
BFGS(atoms).run(fmax=1e-3)
```

Periodic 1D H-chain:

```python
import numpy as np
from vibeqc import (
    Atom, BasisSet, PeriodicSystem, PeriodicSCFOptions,
    monkhorst_pack, run_rhf_periodic,
)

sysp = PeriodicSystem(
    dim=1,
    lattice=np.diag([4.0, 30.0, 30.0]),
    unit_cell=[Atom(1, [0,0,0]), Atom(1, [0,0,1.4])],
)
basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
kmesh = monkhorst_pack(sysp, [6, 1, 1])

opts = PeriodicSCFOptions()
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 40.0
print(run_rhf_periodic(sysp, basis, kmesh, opts).energy)
```

See [docs/quickstart.md](docs/quickstart.md) and
[docs/tutorial/](docs/tutorial/) for a full tour.

## Installation

All native dependencies (libint, libxc, spglib, FFTW3, libecpint) are
**vendored** — `setup_native_deps.sh` fetches and builds each one with
the exact options vibe-qc needs, eliminating "wrong system version"
failures. Full instructions in
[docs/installation.md](docs/installation.md); the short version:

vibe-qc also links Eigen against an **optimised BLAS + LAPACK** —
Apple Accelerate on macOS (ships with the OS), OpenBLAS or MKL on
Linux. `setup_native_deps.sh` runs a **preflight check** that
verifies every tool / header / library below is present before it
starts building, and prints the exact per-distro install command for
anything missing.

After installing Git, clone the public source and enter its directory:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
```

**macOS (Homebrew)**
```sh
brew install cmake ninja pkg-config libomp boost eigen gmp git python@3.14
# Apple's /usr/bin/python3 is 3.9.x and too old — make sure brew's
# python3 wins on PATH:
eval "$(brew shellenv)"
./scripts/install.sh --dev                  # native deps + venv + pip install + banner
```

**Linux — Arch / Manjaro**
```sh
sudo pacman -S base-devel cmake ninja pkg-config git curl \
    gmp eigen boost python blas-openblas
./scripts/install.sh --dev
```

**Linux — Debian / Ubuntu**
```sh
sudo apt install build-essential cmake ninja-build pkg-config git curl \
    libeigen3-dev libboost-dev libgmp-dev libgmpxx4ldbl \
    libopenblas-dev liblapacke-dev \
    python3 python3-dev python3-venv
./scripts/install.sh --dev
```

`install.sh` accepts `--dev` (main), `--current` (do not switch refs),
`--branch NAME` (any branch or tag), `--extras dev` (richer pip extras),
`--extras basisopt` (the retained basis toolkit),
`--python python3.13`, `--venv PATH`, `--with-openblas`, `--force`. The bare command
targets `release` (or the newest stable tag when origin has no release branch), while `--dev` installs `main` into `./.venv/` with `[test]`
extras — for the manual `setup_native_deps.sh` + `python3 -m venv`
recipe see [docs/installation.md](docs/installation.md).

For an existing installation, use `./scripts/update.sh --dev`. To rebuild only
the environment while keeping a verified rollback, use
`./scripts/reinstall.sh`; `./scripts/uninstall.sh` removes only the
checkout-owned virtualenv and retains the checkout and native dependencies.
Pre-marker legacy installs require explicit `--adopt-legacy` and exact PEP 610
proof; a foreign environment is never replaced or removed by these scripts.
The standalone basis optimizer uses `./vibe-basis/scripts/install.sh` and
its own dedicated venv; it does not build the native vibe-qc core.

The viewer and queue have independent repositories and release tags. The
QVF format is also an independent Apache-2.0 reference project. Nothing
links its reference toolkit at runtime; vibe-qc implements the format and
is validated against its published contract and conformance corpus.
vibe-basis deliberately stays in this repository.

| Project | Checkout | Installation |
| --- | --- | --- |
| vibe-qc | this repository | `./scripts/install.sh --dev` |
| vibe-view | [vibe-qc/vibe-view](https://github.com/vibe-qc/vibe-view) | `./scripts/install.sh` from viewer root |
| vibe-queue (`vq`) | [vibe-qc/vibe-queue](https://github.com/vibe-qc/vibe-queue) | `./scripts/install.sh` from queue root |
| vibe-basis | `vibe-basis/` in this repository | `./vibe-basis/scripts/install.sh` from core root |
| QVF | [vibe-qc/qvf](https://github.com/vibe-qc/qvf) | format reference; no core runtime install |

The split repositories have GitHub mirrors:
[vibe-qc](https://github.com/vibe-qc/vibe-qc),
[vibe-view](https://github.com/vibe-qc/vibe-view),
[vibe-queue](https://github.com/vibe-qc/vibe-queue), and
[QVF](https://github.com/vibe-qc/qvf). These public source snapshots can be
cloned anonymously. GitLab requires authorized access and remains the
development source of truth. Public snapshots have separate commit history
and can lag GitLab; check the desired ref on the host you use.
See [repositories and downloads](docs/installation.md#repositories-and-downloads)
for each project's source tags and release artifacts. The archived
the archived monorepo repository retains historical work; do not push new changes there
or rebase its history onto these fresh repositories.

Each tool has install, update, reinstall, and uninstall scripts in its own
checkout. See the [toolset lifecycle guide](docs/toolset_lifecycle.md) for
working directories, verification, and the explicit two-checkout combined
viewer installation. `--extras basisopt` still installs the retained basis
toolkit; the removed viewer source cannot be resolved by `viewer-gpu` alone.

Every component stamps its dedicated environment with checkout ownership;
vibe-qc, vibe-view, and vq also provide an explicit, exact-PEP-610
`--adopt-legacy` migration for their older pre-marker installs.

The vibe-qc chemistry engine requires Python 3.11+ and a C++17 compiler.
The standalone companion installs do not compile the vibe-qc core. Full
troubleshooting, the Fedora / RHEL recipe, the vendored-OpenBLAS escape hatch,
and post-install verification steps are in
[docs/installation.md](docs/installation.md).

## Verify the install

```sh
.venv/bin/python -c "from vibeqc import print_banner; print_banner()"
.venv/bin/python -m pytest tests/
```

You should see the banner (vibe-qc version + codename + git provenance
+ linked libint / libxc / spglib / libecpint / FFTW3 versions) and
~8700 passing test functions (number grows with each phase).

## Architecture

```
python/vibeqc/             Python API — molecular + periodic drivers,
                           ASE, POSCAR / CRYSTAL-format I/O, SCF-log
                           formatter, output coordinator + citation
                           database (vibeqc.output, v0.8.x+).
python/vibeqc/basis_library/   255 bundled .g94 basis files (standard +
                               pob-* solid-state + JK / RI aux).
python/vibeqc/ecp_library/ Bundled libecpint XML libraries.
cpp/include/vibeqc/        C++17 public headers (integrals, SCF, DFT
                           grid, ECP, JKBuilder, …).
cpp/src/                   C++ implementation.
tests/                     pytest regression suite (~190 files,
                           ~1800 test functions).
docs/                      Sphinx-built documentation site (vibe-qc.com).
scripts/                   setup_native_deps.sh + per-dep build_*.sh helpers.
third_party/               Source-built libint, libxc, spglib, fftw,
                           libecpint (generated by
                           setup_native_deps.sh, gitignored).
vibe-basis/                Standalone external-code basis optimization driver.
```

## Documentation

- **User docs** — [`docs/`](docs/), built with Sphinx + Furo. Run
  `make -C docs html` for a local copy; the site is deployed at
  [vibe-qc.com](https://vibe-qc.com).
- **[docs/installation.md](docs/installation.md)** — full install.
- **[docs/toolset_lifecycle.md](docs/toolset_lifecycle.md)** — choose, install,
  update, repair, and uninstall the engine and separately installed companions.
- **[docs/quickstart.md](docs/quickstart.md)** — guided tour.
- **[docs/troubleshooting.md](docs/troubleshooting.md)** — common
  errors with reproducer symptoms + workarounds.
- **[docs/tutorial/](docs/tutorial/)** — 40+ worked tutorials
  covering molecular HF / DFT / MP2 / double hybrids, periodic SCF,
  band structure, geometry opt, vibrations, auto-citations, the
  `vq` queue, the EDIIS+DIIS hybrid, Fermi-Dirac smearing, and the
  RIJCOSX + ECP workflows.
- **[docs/user_guide/](docs/user_guide/)** — reference by concept,
  including the four-code migration guides ([PySCF](docs/user_guide/from_pyscf.md),
  [CRYSTAL](docs/user_guide/from_crystal.md),
  [ORCA](docs/user_guide/from_orca.md),
  [Gaussian](docs/user_guide/from_gaussian.md)).
- **[docs/visualization.md](docs/visualization.md)** — QVF and
  vibe-view in one place: what the `.qvf` your job just wrote
  contains, the four ways to read it (browser, terminal, desktop,
  Python), and how to emit QVF from another quantum-chemistry code
  using the optional Apache-2.0 reference implementations in the separate
  [QVF repository](https://github.com/vibe-qc/qvf).
- **[docs/roadmap.md](docs/roadmap.md)** — milestone ladder v0.1 → v2.0.

## Support the project

vibe-qc is a hobby project, built and maintained by one person on
personal hardware, with no institutional backing. If it's useful to
you, consider sponsoring development:

- **[GitHub Sponsors](https://github.com/sponsors/mpeintinger)** —
  recurring monthly support, zero fees.
- **[Ko-fi](https://ko-fi.com/mpeintinger)** — one-time donations, no
  GitHub account required.

Sponsorship funds the Claude Max subscription that drives day-to-day
development plus the self-hosted server behind
[vibe-qc.com](https://vibe-qc.com). The full pitch (author bio,
funding goals, other ways to help) is on the
[support page](https://vibe-qc.com/docs/support.html).

## Licensing and contributing

Source code: [MPL 2.0](LICENSE). Dependencies (libint, libxc, Eigen,
spglib, pybind11) are all MPL-2.0 or BSD-3 and MPL-compatible. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the short contributor note and
[SECURITY.md](SECURITY.md) for vulnerability reporting.

## Citation

vibe-qc auto-generates citation files for every job — `run_job(...)`
writes a `{stem}.bibtex` and `{stem}.references` next to the `.out`
listing the software citation, libint, libxc, every basis set,
every functional, the SCF accelerator, dispersion correction, and
any other linked library actually used by the run. Drop the
`.bibtex` into your LaTeX project and `\cite{...}` by the entry's
`bibtex_key`. See [`docs/user_guide/citations.md`](docs/user_guide/citations.md)
for the schema and routing details, plus
[`docs/citing.md`](docs/citing.md) for the canonical software
citation and a backup table for ad-hoc citations.

The repository also ships a [`CITATION.cff`](CITATION.cff) that
GitLab and citation managers parse automatically. A peer-reviewed
publication describing vibe-qc is forthcoming; until then, the
software citation in `CITATION.cff` is the canonical reference.


## History

This repository begins at the commit below. Development before that
point took place in a private monorepo, which is retained privately;
the history was deliberately not transferred.

Copyright (c) 2026 Michael F. Peintinger and vibe-qc contributors.
