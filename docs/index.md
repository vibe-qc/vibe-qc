---
myst:
  html_meta:
    "description": "vibe-qc, quantum-chemistry code for molecules and solids. Hartree-Fock, DFT, MP2, DLPNO-CCSD, TDDFT, CASSCF, dispersion, periodic systems. Python front-end over a validated C++17 core."
    "og:title": "vibe-qc, Quantum chemistry for molecules and solids"
    "og:description": "Python + C++17 electronic-structure code. RHF / UHF / DFT / MP2 validated against PySCF; 1D/2D/3D periodic HF and KS-DFT with GPW, GDF, and BIPOLE routes."
    "og:type": "website"
    "og:url": "https://vibe-qc.com/docs/"
    "og:image": "https://vibe-qc.com/docs/_static/logo/vibe-qc-social.png"
    "og:image:width": "1200"
    "og:image:height": "630"
    "og:image:alt": "vibe-qc: Quantum chemistry. Molecules to solids."
    "twitter:image:alt": "vibe-qc: Quantum chemistry. Molecules to solids."
    "twitter:card": "summary_large_image"
    "twitter:image": "https://vibe-qc.com/docs/_static/logo/vibe-qc-social.png"
---

# vibe-qc

**Quantum chemistry for molecules and solids.**

vibe-qc is a Python + C++17 electronic-structure code. The molecular
stack, Hartree-Fock, density-functional theory, Møller-Plesset
theory, analytic gradients, D3(BJ) dispersion, is validated against
[PySCF] to machine precision. The periodic stack delivers 1D / 2D / 3D
Hartree-Fock and Kohn-Sham DFT with Monkhorst-Pack k-meshes,
Ewald-summed Madelung, band structure and density of states, and is
growing toward CRYSTAL-style crystalline-orbital calculations at
full-SCF accuracy. COOP/COHP bonding analysis and periodic Mayer bond
orders are available for 1D/2D/3D systems.

Every calculation leaves behind **one file**: a [QVF](visualization.md)
archive carrying the structure, the wavefunction, scalar fields, spectra,
bands, trajectories, provenance and citations, which
[vibe-view](visualization.md) opens in a browser, in a terminal over SSH, as
a desktop app, or headlessly from Python. QVF is an open, versioned format
with an Apache-2.0 reference writer, meant for the whole
quantum-chemistry ecosystem rather than for vibe-qc alone.

Built on [libint] (Gaussian integrals), [libxc] (500+ XC
functionals), and [spglib] (crystal symmetry). Licensed MPL-2.0.

```{figure} _static/logo/vibe-qc-family-social.svg
:alt: Four independent tools for chemical drawing, submission, calculation and analysis.
:width: 1200px
:class: product-art

[vibe-sketch](https://vibe-qc.com/vibe-sketch/docs/) draws and prepares inputs,
[vq](https://vibe-qc.com/vibe-queue/docs/) submits and tracks work,
[vibe-qc](quickstart.md) calculates, and
[vibe-view](https://vibe-qc.com/vibe-view/docs/) analyzes results.
The workflow is sketch → submit → calculate → analyze; a local calculation
can skip queue submission.
Install each product independently and follow its own documentation.
```

## Choose where to begin

| Your goal | Start here |
|---|---|
| Install the calculation engine | [Start here](getting_started.md), then [Installation](installation.md) |
| Run a first calculation | [Quickstart](quickstart.md) |
| Open a `.qvf` without installing vibe-qc | [vibe-view quickstart](https://vibe-qc.com/vibe-view/docs/quickstart.html) |
| Submit, watch, and retrieve queued work | [vq user guide](https://vibe-qc.com/vibe-queue/docs/user/index.html) |
| Install or operate a queue host | [vq operator guide](https://vibe-qc.com/vibe-queue/docs/operator/index.html) |
| Choose a method and check its status | [Methods: theory and status](methods.md) |
| Learn methods through worked calculations | [Tutorial learning paths](tutorial/index.md#choose-a-learning-path) |
| Look up options, defaults, and limitations | [User guide task index](user_guide/index.md#find-the-page-from-the-task) |
| Browse complete runnable inputs | [Examples catalog](https://github.com/vibe-qc/vibe-qc/blob/main/examples/README.md) |
| Diagnose an installation or calculation | [Troubleshooting](troubleshooting.md) |
| Update an existing installation safely | [Updating](updating.md) |
| Report a reproducible failure | [Bug-report checklist](troubleshooting.md#prepare-a-reproducible-report) |
| Develop or document a change | [Contributor setup](contributor_setup.md), then [contribution guide](contributing.md) |

The [Start here](getting_started.md) page distinguishes the compiled vibe-qc
engine from the standalone vibe-view package and gives a verification
checklist for installing either one or both.

[libint]: https://github.com/evaleev/libint
[libxc]: https://libxc.gitlab.io/
[spglib]: https://github.com/spglib/spglib
[PySCF]: https://pyscf.org/

```{admonition} Current source: {{release}} - *{{codename}}*
:class: tip

This manual describes the current source, including changes that may follow
the package version shown above. Documentation refreshes do not qualify a
new runtime release or widen a method's validation envelope. See
[release_process](release_process.md) for tagged releases and the branch model.

The repository split makes the viewer, queue, and QVF format independently
adoptable. `vibe-basis/` deliberately stays in vibe-qc.

| Project | Version or releases | Role |
| --- | --- | --- |
| vibe-qc | {{release}} | quantum chemistry engine |
| vibe-view | [Companion releases](https://github.com/vibe-qc/vibe-view/releases) | visualization and QVF viewer |
| vibe-queue (`vq`) | [Companion releases](https://github.com/vibe-qc/vibe-queue/releases) | reusable job queue |
| vibe-basis | {{vibebasis_version}} | basis toolkit, retained here |
| QVF | [Format releases](https://github.com/vibe-qc/qvf/releases) | specification and conformance reference, not a runtime dependency |

See [toolset_lifecycle](toolset_lifecycle.md) to install or update any of
them individually.
```

```{admonition} 💚 vibe-qc is funded by individual sponsors
:class: tip

vibe-qc is built and maintained by [one person](support.md#about-the-author)
in evenings and weekends, on personal hardware, with no institutional
backing. If the project is useful to you - or if you think the Cyclic
Cluster Model reaching CCSD(T) on an open-source code is worth existing -
please consider supporting it via
[**GitHub Sponsors**](https://github.com/sponsors/mpeintinger) (recurring
monthly, zero fees) or
[**Ko-fi**](https://ko-fi.com/mpeintinger) (one-time, no GitHub account
required). Every sponsorship directly funds the **Claude Max
subscription** that drives day-to-day development, the **self-hosted
server** behind `vibe-qc.com`, and - **most urgently - bigger hardware**:
the project currently develops on a single Apple M2 laptop, and that's
what caps every regression test, CI benchmark, and CRYSTAL-Tutorial
port to small systems. **Smallest single actionable item right now:
the [NIST Crystal Data SRD 3 single-user
subscription](https://www.nist.gov/srd/nist-standard-reference-database-3)
at $200/year** - fully fundable by one sponsor for a year, and
unlocks programmatic access to a curated crystallographic
database for the CCM reference work. See the
[support page](support.md#what-your-sponsorship-funds) for the full
pitch and the near-/long-term hardware goals; public sponsors are
listed on the [sponsors page](sponsors.md).
```

```{admonition} 🖥️ macOS desktop auto-update is not currently supported
:class: note

The [vibe-view desktop app](user_guide/vibe_view_desktop.md) can detect a new
release on macOS, but it cannot install the update automatically. Self-update
requires Apple Developer registration and Developer ID signing, and that paid
registration is currently paywall-gated. Use the manual **Download…** button
for now; see the {ref}`macOS auto-update sponsorship opportunity
<macos-desktop-auto-update>` for details.
```

```{admonition} ✅ New in v0.18 (Berkelbach's Badger)
:class: tip

<p class="codename-art codename-art-home">
  <img src="_static/images-codenames/upcoming/18-vibe-qc-v0.18.0-berkelbachs-badger.png" alt="AI-generated Berkelbach's Badger codename artwork for vibe-qc v0.18">
</p>

v0.18 is a **bounded periodic-correctness release**. It freezes the periodic
routes and the four MANTA control profiles whose numbers were checked against
references, and states the rest as explicit exclusions; the
[0.18 scope](release_018_scope.md) and the
[qualified PBC configurations](user_guide/pbc_018_supported_configurations.md)
list exactly what is claimed. The v0.18.0 tag was cut on 2026-10-23; its
changelog records that full candidate qualification and publication were
still pending.

- **Periodic quadrature.** The periodic Becke partition switches atom images
  off smoothly, so periodic Kohn-Sham energies and weights are continuous in
  the atomic positions, and the Gamma-point GDF Kohn-Sham gradients now
  include the motion of that grid.
- **Periodic ECPs.** ECP single points run on the periodic routes, with a
  source-matched Ewald UKS analytic-gradient preview; other ECP gradients
  stay refused.
- **Refusals where a limit does not exist.** Dense-cell Gamma RIJCOSX
  exchange, whose image sum has no limit, is refused; a 1-D wire is modelled
  as a 3-D cell with vacuum.
- **Local correlation.** DLPNO runs on crystals at Gamma and on BvK-folded
  meshes, closed-shell DLPNO-CCSD gains LMP2 pair natural orbitals and a
  compiled pair-screening kernel, and open-shell pair screening defaults to
  0.25 tcut_pairs, which changes open-shell energies.
- **Also in the 0.17.18 to 0.18.0 line:** QVF toolkit 0.5 interoperability
  with canonical excited-state, NTO and PDOS sections; corrected TDDFT
  transition-dipole factors; native D4 optimisation and gradients;
  COSMO-RS infinite-dilution activity coefficients; K-commensurate hexagonal
  k meshes; sparse MRCI; Lebedev rules up to order 95; standalone macOS and
  Linux [distribution builds](distribution.md).

⚠️ **MANTA, AICCM/CCM, chi, SECCM and native periodic correlation stay
experimental.** MANTA's component certificates bound what they name, not a
relaxed total energy.

The preceding v0.17 line (*Tew's Tern*) laid the symmetry and periodic
groundwork and the shared operator contract; v0.16 (*Pople's Puffin*)
carried the repository split into `vibe-qc`, `vibe-view`, `vibe-queue` and
`qvf`.
```

```{admonition} 🔜 Current focus after 0.18
:class: note

The broader production goal for periodic HF, DFT and hybrids, including
CRYSTAL23 accuracy and time-to-solution comparisons, is the next theme; the
[CRYSTAL23 capability matrix](crystal23_capability_matrix.md) shows where
each route stands. MANTA continues alongside libint, and native periodic
local correlation is deferred to a later feature release. The vibe-qc
research paper is under peer review: read the
[ChemRxiv preprint](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15007558/v1).

See the [roadmap](roadmap.md) for the full plan.
```

For the current open-issues list (workarounds, regression-test
pointers, status of each known bug), see
[`troubleshooting`](troubleshooting.md) and the
[issue tracker](https://github.com/vibe-qc/vibe-qc/issues).

Periodic COOP/COHP is computed with the same spin Hamiltonian that produced
the analysed bands on the GDF routes, and the kernel refuses any other
operator; on BIPOLE, AICCM and external-XC routes the DOS and COOP/COHP
sections are skipped with a stated reason. Property output on ECP results
is tracked as issue 32 on the [issue tracker](https://github.com/vibe-qc/vibe-qc/issues)
(monorepo number 741); Hirshfeld charges on ECP results now build their
promolecule from valence-only free atoms.
GDF spectral output retains all accepted eigenstates and uses the requested
Gaussian width. If distant states would exceed the property memory limit on
one uniform energy grid, it uses nonuniform energy coordinates with resolved
Gaussian neighborhoods and exact zeros across empty gaps. The 128 MiB default
limit still applies; a genuine refusal is recorded in the run manifest.


## Install

Clone the public source over HTTPS; no account or deploy key is required:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
```

Authorized maintainers use their separately configured upstream remote.
Private access recipes belong in the external operations documentation.

Then enter the new checkout and install:

```sh
cd vibe-qc
./scripts/install.sh --dev                 # native deps + venv + pip install + banner
```

`install.sh` accepts `--dev` (main), `--branch NAME` (any branch
or tag), and other knobs, see [installation.md](installation.md)
for the full surface and the manual `setup_native_deps.sh` recipe.
`git clone` selects the project's default branch, `main`. The command above
keeps the development line with `--dev`; the flag-free installer selects
`release` when that branch exists on the selected origin, or its newest
stable `vX.Y.Z` tag otherwise. Use `--branch vX.Y.Z` to pin an exact
published tag for reproducibility.

Public source repositories are listed in
[repositories and downloads](installation.md#repositories-and-downloads),
including each project's tagged source downloads.

``setup_native_deps.sh`` builds and installs every native dependency
(libint, libxc, spglib, FFTW3, libecpint) into ``third_party/`` and
populates the bundled basis library. Re-running is a no-op if
everything's already built. Full per-platform dependency lists
(macOS / Arch / Manjaro / Debian / Ubuntu) are in
[installation](installation.md).

## Your first calculation

Make a working directory **outside the repo**, vibe-qc writes its
outputs into the current working directory, and you don't want
``.out`` / ``.molden`` / ``.traj`` files landing inside the source
tree:

```sh
mkdir -p ~/vibeqc-runs/water
cd ~/vibeqc-runs/water
```

Save the following as ``water.py`` in that directory (any filename
works, vibe-qc just runs whatever Python you point it at):

```python
from vibeqc import Atom, Molecule, run_job

mol = Molecule([
    Atom(8, [ 0.0,  0.00,  0.00]),
    Atom(1, [ 0.0,  1.43, -0.98]),
    Atom(1, [ 0.0, -1.43, -0.98]),
])

run_job(
    mol,
    basis="6-31g*",
    method="rks",
    functional="PBE",
    dispersion="d3bj",
    optimize=True,
    output="water",
)
```

Run it with the **virtual-env's Python** (the one ``pip install``
populated above, *not* your system ``python3``). Since you're no
longer in the repo, give the full path:

```sh
~/path/to/vibe-qc/.venv/bin/python water.py
```

Replace ``~/path/to/vibe-qc/`` with wherever ``git clone`` landed.
That ~3-second run writes a set of files into ``~/vibeqc-runs/water/``.
The four you will reach for first:

- ``water.out``, banner, SCF trace, energy breakdown, orbital table,
  HOMO-LUMO gap, Mulliken / Löwdin charges, Mayer bond orders, dipole,
  and wall-clock timings.
- ``water.qvf``, the whole calculation as one archive: structure,
  optimization trajectory, wavefunction, atom properties, bond orders,
  SCF history, the run record and the citations. Open it with
  [vibe-view](visualization.md), in a browser, in a terminal, or from
  Python. This one is written by default; pass ``output_qvf=False`` to
  skip it.
- ``water.molden``, molecular orbitals for Avogadro / Jmol.
- ``water.traj``, ASE trajectory for the optimization, viewable with
  ``ase gui water.traj``.

The run also leaves ``.xyz``, ``.bibtex`` / ``.references`` (the
citations for the level of theory you used), ``.population.txt`` /
``.population.json``, and a ``.system`` manifest. See
[output files](user_guide/output_files.md) for the full family.

```{tip}
**Skip the path prefix.** Activate the venv once per shell session
and the path resolves automatically - works from any directory:

    source ~/path/to/vibe-qc/.venv/bin/activate    # bash / zsh
    python water.py                                # uses the venv's python

Deactivate with ``deactivate`` when you're done.

**Common mistake:** ``ModuleNotFoundError: No module named 'vibeqc'``
means you ran the wrong Python. Either give the full
``~/path/to/vibe-qc/.venv/bin/python`` path or activate the venv
first. The bare ``.venv/bin/python`` shorthand only works when your
shell is sitting inside the repo.
```

See [quickstart](quickstart.md) for a 30-minute end-to-end
walkthrough (HF, periodic SCF, orbital cube), [running](running.md)
for the full "how to invoke vibe-qc scripts" reference (venv,
threading, output capture, SSH workflows),
[good practices](good_practices.md) for the working conventions
nobody tells you (file layout, naming, when to trust a number),
the [tour](tour.md) for the wider API surface (`run_job`, ASE
Calculator, logging, custom basis sets, tests), or dive into the
[tutorials](tutorial/index.md) for worked examples.

## See your results: QVF and vibe-view

The `water.qvf` your first job just wrote is the whole calculation in one
typed, checksummed archive. **[vibe-view](visualization.md)** is the viewer
for it, and it does not need vibe-qc: it is a wheel-based Python package, so
installing its browser, terminal, and headless modes takes one checkout
installer in the separate viewer repository, with no compiler or native build.
The distribution is not on PyPI yet. The Electron desktop shell is a separate
source-checkout component today.

```{figure} _static/plots/vibe_view/15-orbital-homo.png
vibe-view in the browser: the section browser on the left, the
GPU-accelerated 3D viewport in the middle, per-section controls on the
right. Here a molecular orbital with both signed lobes.
```

Four ways to read the same archive, so the environment never dictates
whether you can look at your results:

| | Command | Needs |
|---|---|---|
| **Browser** | `vibe-view open job.qvf` | an OpenGL context |
| **Terminal** | `vibe-view show` / `vibe-view tui` | terminal profile, no display server, no X forwarding |
| **Desktop app** | `vibe-view desktop job.qvf` | a source checkout and Electron |
| **Headless / Python** | `vibe-view capture`, `render_terminal`, `QVFReader` | nothing beyond the base install |

* **New here?**
  [Getting started with vibe-view alone](tutorial/vibe_view_getting_started.md)
  installs just the viewer on Linux or macOS. The current checkout creates a
  built-in demo; the hosted wheel opens files you already have. One wheel
  carries the released Python viewer modes, with no compiler and no vibe-qc;
  desktop mode uses the source checkout:
  [Companion release artifacts](https://github.com/vibe-qc/vibe-view/releases).
* **Want the tour?**
  [vibe-view: an end-to-end walkthrough](tutorial/vibe_view_walkthrough.md)
  goes panel by panel.
* **On a compute node?**
  [Terminal mode](user_guide/vibe_view_terminal_mode.md) renders structures,
  isosurfaces and charts as Unicode braille over SSH.
* **Writing your own code?** QVF is a published, versioned format with a
  normative specification, a JSON schema, reference writers in Python and
  zero-dependency C++17, a validator and a conformance corpus, distributed
  under **Apache 2.0** so it can be vendored into any code including a
  proprietary one. See
  [The QVF format toolkit](qvf/index.md) and
  [Adopting QVF in your own code](tutorial/qvf_adapt_and_writer.md).

Everything about the format and the viewer lives in one place:
**[QVF and vibe-view](visualization.md)**.

## Capabilities today

**Molecular.** Restricted and unrestricted HF / KS-DFT with analytic
nuclear gradients. The full libxc functional library, LDA, GGAs,
hybrids, the τ-dependent meta-GGA family (TPSS, M06-2X, SCAN,
r²SCAN, r²SCAN01), range-separated hybrids (ωB97X, ωB97X-D, and the
VV10-paired ωB97X-V / ωB97M-V), and the PW1PW
weighted-sum functional, all supported. Møller-Plesset theory
(MP2 / UMP2 / RI-MP2 / SCS-MP2 / SOS-MP2 / open-shell UMP2) and the
B2PLYP / DSD-PBEP86 / revDSD-PBEP86-D4 double hybrids. The dense correlated
routes are accuracy references for the demonstrated small-molecule envelope;
they are not yet claimed as production-size solvers. Density fitting with the RIJK and
RIJCOSX Fock-build kernels (RIJCOSX energies within 0.26 mHa of ORCA
6.1.1 on glycine/def2-TZVP). Effective core potentials for heavy-element chemistry.
CPCM / COSMO implicit solvation via the SolutePotentialProvider seam.
Grimme D3(BJ) / D4 dispersion. 255 bundled Gaussian basis files,
including the solid-state pob-\* family. Numerical anchors use out-of-process
PySCF and ORCA comparisons where documented.



**Wavefunction methods.** Canonical **CCSD(T)**, the gold-standard
molecular correlation reference: closed-shell, plus open-shell
(UCCSD / UCCSD(T) on a UHF reference, ROHF-reference CCSD / CCSD(T)) and
frozen natural orbitals for cheaper (T). **DLPNO-CCSD(T)**, the
near-linear-scaling local CCSD(T) route, with retained sub-kcal evidence for
explicitly pinned historical recipes (not yet a claim for the current
chemical-core/NormalPNO default), plus open-shell **DLPNO-UMP2** and
**DLPNO-UCCSD(T)** (full reduced-scaling local solver, pilot-anchored against
the dense spin-orbital oracle). DLPNO-(T1) restores off-diagonal localised-Fock
coupling and reaches canonical (T) in the full-domain limit. Multi-root CASCI with state-averaged
**CASSCF** (analytic nuclear gradient FD-tight
to ~1e-7, shipped in v0.15.0). **TDDFT** via the Casida linear-response
formalism and the Tamm-Dancoff approximation (RHF/RKS/UHF/UKS), with
Natural Transition Orbitals and FD excited-state gradients.
**Green's-function** quasiparticle ionisation energies (GF2 / OVGF,
renormalised GF2). An **eigensolver framework** (Davidson, LOBPCG, plus
experimental Jacobi-Davidson and GPLHR) selectable by keyword. The
`vibeqc.solvers` family, Full CI, selected CI, variational 2-RDM,
transcorrelated CI and an experimental DMRG toy solver, via
[`vibeqc.solvers`](user_guide/non_hf_solvers.md). General atomisation
energies and RRHO thermochemistry.

**Semiempirical + MLIP.** The MSINDO semiempirical engine covers
elements H-Xe (Z 1-54), supports NDDO mode for the main group, molecular
geometry optimisation, implicit solvation (COSMO), NEB, velocity-Verlet
molecular dynamics, well-tempered metadynamics, and penalty-function MECI
conical-intersection optimisation; it is production within that scope.
DFTB0 and SCC-DFTB ship as screening and preoptimisation routes (preview:
in-house parameters, no DFTB+ parity). GFN2-xTB and PM6 are experimental,
OM1 to OM3 are experimental and scheduled for retirement, and PM7 is gated.
The semiempirical cyclic cluster model (SECCM) is experimental. The MACE
machine-learning interatomic potential (`method="mace"`) is an external
pre-trained model available as an alternative energy surface. Every label
is defined in [Methods: theory and status](methods.md).


**Periodic.** 1D / 2D / 3D `PeriodicSystem` geometry, Monkhorst-Pack
k-meshes with IBZ reduction. Native Gaussian density fitting (GDF,
Γ-point RHF / RKS + hybrids, multi-k KRHF / KRKS; µHa parity vs PySCF
on LiH) and the production GPW (Gaussian Plane Waves) route through
`run_periodic_job` (Γ-only RHF / UHF / RKS / UKS, multi-k pure-DFT
RKS / UKS; ROHF / ROKS are maintained previews and the MPI grid overlay is experimental). The 3D BIPOLE route covers
Γ + multi-k RHF / UHF / RKS / UKS with the exact Ewald-J split.
Finite-difference forces are the production path; analytic gradients are
a gated preview. The quartet multipole prototype is fail-closed and is not
part of production energies or gradients. 3D Ewald uses a shared α gauge.
Fermi-Dirac, Methfessel-Paxton, and Marzari-Vanderbilt smearing for
metals, with Anderson / Broyden / Kerker density mixers and an AUTO
k-point and smearing recommender. MSINDO periodic CCM through bulk
MgO (Wigner-Seitz, periodic INDO, Ewald Madelung). **Experimental
ab-initio CCM:** two independent lines. The [Γ-CCM line](aiccm2026dev_a.md)
hosts the neutral finite-torus construction (variants `real-gamma` and
`neutral-bloch`) and the 2014 union-and-weight four-centre lineage
(`four-center`), HF→CCSD(T); [χ-CCM](user_guide/aiccm2026dev_b.md) is the
finite-character construction (3D SCF + finite-torus correlation). They are
under active side-by-side study on a 28-system benchmark, but no
cross-approach delta is currently reportable; every driver warns. Band structure
and density-of-states plotters. **COOP/COHP** bonding analysis
(Crystal Orbital Overlap/Hamilton Population; C++ kernels, QVF,
plotters, `vibeqc coop` CLI). **Periodic Mayer bond orders**
(k-space generalisation). **Fat bands** (Mulliken-projected band
weights). **QTAIM** topological analysis (critical-point search +
bond-path tracing). Gaussian cube + extended-XYZ +
POSCAR + XSF / BXSF writers, plus POSCAR and Extended-XYZ readers
and CIF readers with opt-in geometry symmetrisation.

**Tooling.** OpenMP parallelism throughout; optional `mpi4py`
substrate for future MPI parallelism (GPW grid overlay experimental;
production strategy in [MPI Parallelization](design_mpi_parallelization.md)).
Pre-flight memory budget estimator.
ASE Calculator integration for geometry optimization, vibrational
frequencies, and NEB. Automatic per-job citation files
([`.bibtex` / `.references`](user_guide/citations.md)). The
[`vibe-view`](user_guide/vibe_view.md) interactive 3D viewer for
structure / orbitals / densities / bands / COOP/COHP / spectra /
trajectories from supported QVF sections. The current viewer also displays
reaction schemes and preview images from vibe-sketch. Current viewer source
also renders `excited_states` tables in the browser, terminal and CLI, with
energies, strengths and orbital contributions. Excited-state energies label
NTOs; the
[viewer capability guide](https://vibe-qc.com/vibe-view/docs/capabilities.html)
describes support by surface. The
[`basis_toolkit`](qvf_basis_developer_readme.md) for
basis-set import/export (BSE, CRYSTAL, G94, NWChem, ORCA). The
[`vq` queue](user_guide/queue.md) for remote job submission.

See [Methods: theory and status](methods.md) for what each method computes
and which of the four status labels (production, preview, experimental,
gated) it carries, the [feature matrix](features.md) for the option tables,
and the [roadmap](roadmap.md) for what's next.

## Where to go next

```{toctree}
:maxdepth: 2
:caption: Getting started

getting_started
installation
distribution
toolset_lifecycle
quickstart
using_claude
running
qcschema
good_practices
troubleshooting
updating
cluster_setup
tour
```

```{toctree}
:maxdepth: 2
:caption: QVF and vibe-view

visualization
relocalization_worker
```

```{toctree}
:maxdepth: 1
:caption: Tutorial

tutorial/index
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/index
```

```{toctree}
:maxdepth: 1
:caption: Design documents -- architecture overviews

design_qvf_format
design_output_module
developer_shared_symmetry
symmetry_status
design_integrals_first
design_mpi_parallelization
design_vibeqc_vs_gpaw
```

```{toctree}
:maxdepth: 1
:caption: Design documents -- subsystem implementation notes

design_qvf_basis
design_native_gdf
design_rsgdf_3c_lr
design_dlpno_correlation_source
design_basis_ao_section
design_periodic_gapw
design_mdf
design_smearing
design_cosmors_run_job
design_aiccm2026dev_b
design_shared_symmetry
symmetry_literature
aiccm2026dev_a_scalability
aiccm2026dev_a_lowd_greens
aiccm2026dev_a_analytic_gradient
qvf_basis_developer_readme
qvf_basis_conversion_matrix
```

```{toctree}
:maxdepth: 1
:caption: Reference

methods
api/index
features
experimental/catalog
validation
roadmap
roadmap_history
```

```{toctree}
:maxdepth: 1
:caption: vq

user_guide/queue
```

```{eval-rst}
.. ifconfig:: is_dev

   .. toctree::
      :maxdepth: 1
      :caption: Experimental

      experimental/index
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
release_process
release_018_scope
github_publication
site_publishing
contributing
contributor_setup
developer_test_lanes
developer_pbc_performance
testing
support
sponsors
citing
license
example_outputs
```

## Status

vibe-qc is pre-release software heading toward a 1.0 feature-complete
milestone. Molecular HF/DFT and the documented analysis routes are stable.
MP2 and canonical CCSD(T) have small-molecule numerical validation, but their
production-size memory envelope remains gated by BUG 63 until bounded-memory
executions, rather than dry-run estimates, are archived. The
periodic stack supports three production routes: GDF (µHa parity vs
PySCF), GPW (Gaussian Plane Waves with Γ-point analytic gradients;
MPI grid overlay experimental), and BIPOLE (exact Ewald-J with production
finite-difference forces and fixed-cell atomic relaxation; analytic gradients
remain a gated preview). The eigensolver framework offers Davidson and
LOBPCG by keyword, with experimental Jacobi-Davidson and GPLHR at library
level.
Analysis tools (COOP/COHP, QTAIM, Mayer bond orders, fat bands)
and the basis_toolkit import/export system shipped in v0.15.0.
The ab-initio CCM (Γ-CCM and χ-CCM) ships as experimental.
Follow the [roadmap](roadmap.md) for what's next.

Licensed under the [Mozilla Public License 2.0](license.md). Source at
[github.com/vibe-qc/vibe-qc](https://github.com/vibe-qc/vibe-qc).

## Feedback and bug reports

Found a bug, have a feature request, or want to send a patch?
The decision tree lives in
[CONTRIBUTING.md](contributing.md). The short version:

- **Bugs / install problems / feature requests** →
  [GitHub issues](https://github.com/vibe-qc/vibe-qc/issues)
- **Security vulnerabilities** → email `mpei@vibe-qc.com` (see
  [SECURITY.md](https://github.com/vibe-qc/vibe-qc/blob/main/SECURITY.md))
- **General feedback / questions** → email `mpei@vibe-qc.com`
- **Sponsor the project** → see [Support vibe-qc](support.md)
  ([GitHub Sponsors](https://github.com/sponsors/mpeintinger),
  [Ko-fi](https://ko-fi.com/mpeintinger))
