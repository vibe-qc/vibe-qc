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
:alt: vibe-qc, vibe-view and vq: Calculate. Explore. Keep work moving. Three independent products.
:width: 1200px
:class: product-art

[vibe-qc](quickstart.md) calculates,
[vibe-view](https://vibe-qc.com/vibe-view/docs/) explores results, and
[vq](https://vibe-qc.com/vibe-queue/docs/) schedules work.
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
| Learn methods through worked calculations | [Tutorial learning paths](tutorial/index.md#choose-a-learning-path) |
| Look up options, defaults, and limitations | [User guide task index](user_guide/index.md#find-the-page-from-the-task) |
| Browse complete runnable inputs | [Examples catalog](https://github.com/vibe-qc/vibe-qc/blob/main/examples/README.md) |
| Diagnose an installation or calculation | [Troubleshooting](troubleshooting.md) |

The [Start here](getting_started.md) page distinguishes the compiled vibe-qc
engine from the standalone vibe-view package and gives a verification
checklist for installing either one or both.

[libint]: https://github.com/evaleev/libint
[libxc]: https://libxc.gitlab.io/
[spglib]: https://github.com/spglib/spglib
[PySCF]: https://pyscf.org/

```{admonition} Current source: {{release}} - *{{codename}}*
:class: tip

This documentation is built from the checked-out source. The `v0.17.1` tag
was published on 2026-09-10 and the `release` branch fast-forwarded to it;
the public docs are built from `release`. See [release_process](release_process.md) for the
branch model.

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

```{admonition} ✅ New in v0.17 (Tew's Tern)
:class: tip

<p class="codename-art codename-art-home">
  <img src="_static/images-codenames/17-vibe-qc-v0.17.0-tews-tern.png" alt="AI-generated Tew's Tern codename artwork for vibe-qc v0.17">
</p>

v0.17 lays **symmetry and periodic groundwork**. The shared group and space
infrastructure the periodic symmetry paths had been duplicating is restored;
retained group composition, Bloch factors and retained subspaces are audited
across panel backends, so the reduction a backend claims is the reduction it
performs; and periodic subspace probes are bound to immutable states.

On the BIPOLE side, finite Seitz supports and AO maps are audited, an exact
geometric builder supplies physical quartet supports for the short-range
kernel, the native long-range Gram kernel accepts independent left/right
AO-product cell lists with provenance for both, and exact-mesh HF long-range
contractions are bridged.

⚠️ **These are supports and audits, not a production route.** Production HF
and complete source integration remain open, and the quartet builder does not
authorize symmetry reduction. v0.17 does **not** ship periodic local
correlation; that milestone is [v0.18.0](roadmap.md).

Also in v0.17:

- **The COSMO FINE cavity is built in its molecule-fixed frame by default**,
  with the analytic nuclear derivative running through that frame, and the
  cavity assigns basis points to atoms smoothly.
- **The 16 cc-pVnZ-PP orbital sets**, and vibe-basis 0.11.0 with the
  cohesive objective.
- **`correlation=` on the AICCM front door** (four-centre arm), with
  Gamma-CCM correlation citation stamps and routes.
- **Per-product marketing pages** for vibe-qc, vibe-view and vibe-queue, with
  the website deploy no longer overwriting companion documentation subtrees.

The preceding v0.16 line (*Pople's Puffin*) carried the repository split:
`vibe-qc`, `vibe-view`, `vibe-queue` and `qvf` became four independently
adoptable projects, with vibe-qc reading its own QVF against the published
spec. Nothing depends on `qvf`.

See [CHANGELOG](https://github.com/vibe-qc/vibe-qc/blob/main/CHANGELOG.md)
for the full notes.
```

```{admonition} 🔜 Current focus: v0.17.x bug fixing and paper publication
:class: note

The 0.17.x line continues the 0.16.x commitment: fixing bugs and supporting
publication of the vibe-qc research paper, which is currently under peer
review. Read the
[ChemRxiv preprint](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15007558/v1).
AICCM and CCM remain experimental and are not part of that focus; they are
neither release gates nor validated methods.

See the [roadmap](roadmap.md) for the full plan.
```

For the current open-issues list (workarounds, regression-test
pointers, status of each known bug), see
[`troubleshooting`](troubleshooting.md) and the
[issue tracker](https://github.com/vibe-qc/vibe-qc/issues).

Periodic COHP/ICOHP output currently projects the core Hamiltonian rather
than the full Hamiltonian defining the analyzed bands. These curves are not
yet qualified as standard COHP; the correction is tracked in
[#765](https://github.com/vibe-qc/vibe-qc/issues), alongside the
ECP property-output defect [#741](https://github.com/vibe-qc/vibe-qc/issues).

## Install

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
./scripts/install.sh --dev                 # native deps + venv + pip install + banner
```

`install.sh` accepts `--dev` (main), `--branch NAME` (any branch
or tag), and other knobs, see [installation.md](installation.md)
for the full surface and the manual `setup_native_deps.sh` recipe.
`git clone` selects the project's default branch, `main`. The command above
keeps the development line with `--dev`; the flag-free installer selects
the newest stable tag advertised by origin. Use `--branch vX.Y.Z` to select
a tag that exists in this repository for reproducibility.

See [installation](installation.md) for GitHub clone instructions and
the publication access boundary.

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
~/path/to/vibeqc/.venv/bin/python water.py
```

Replace ``~/path/to/vibeqc/`` with wherever ``git clone`` landed.
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

    source ~/path/to/vibeqc/.venv/bin/activate    # bash / zsh
    python water.py                                # uses the venv's python

Deactivate with ``deactivate`` when you're done.

**Common mistake:** ``ModuleNotFoundError: No module named 'vibeqc'``
means you ran the wrong Python. Either give the full
``~/path/to/vibeqc/.venv/bin/python`` path or activate the venv
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
RIJCOSX Fock-build kernels (RIJCOSX validated to 0.13 mHa vs ORCA
6.1.1). Effective core potentials for heavy-element chemistry.
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
**Eigensolver framework**: Davidson, LOBPCG
(GF2 / OVGF, renormalised GF2). The `vibeqc.solvers` family, Full CI,
selected CI, DMRG, variational 2-RDM, via
[`vibeqc.solvers`](user_guide/non_hf_solvers.md). General atomisation
energies and RRHO thermochemistry.

**Semiempirical + MLIP.** The MSINDO semiempirical engine covers
elements H-Br (Z 1-35, including 3d and 4th-row p-block), supports
NDDO mode, molecular geometry optimisation, implicit solvation
(COSMO), NEB, velocity-Verlet molecular dynamics, well-tempered
metadynamics, and penalty-function MECI conical-intersection
optimisation. The MACE machine-learning interatomic potential
(`method="mace"`) is available as an alternative energy surface.
GFN2-xTB rounds out the semiempirical roster.


**Periodic.** 1D / 2D / 3D `PeriodicSystem` geometry, Monkhorst-Pack
k-meshes with IBZ reduction. Native Gaussian density fitting (GDF,
Γ-point RHF / RKS + hybrids, multi-k KRHF / KRKS; µHa parity vs PySCF
on LiH) and the production GPW (Gaussian Plane Waves) route through
`run_periodic_job` (Γ-only RHF / UHF / RKS / UKS, multi-k pure-DFT
RKS; the MPI grid overlay is experimental). The 3D BIPOLE route covers
Γ + multi-k RHF / UHF / RKS / UKS with the exact Ewald-J split.
Finite-difference forces are the production path; analytic gradients are
a gated preview. The quartet multipole prototype is fail-closed and is not
part of production energies or gradients. 3D Ewald uses a shared α gauge.
Fermi-Dirac, Methfessel-Paxton, and Marzari-Vanderbilt smearing for
metals, with Anderson / Broyden / Kerker density mixers and an AUTO
k-point and smearing recommender. MSINDO periodic CCM through bulk
MgO (Wigner-Seitz, periodic INDO, Ewald Madelung). **Experimental
ab-initio CCM:** two independent lines, [Γ-CCM](aiccm2026dev_a.md)
(union-and-weight/Wigner-Seitz integral weighting, HF→CCSD(T)) and
[χ-CCM](user_guide/aiccm2026dev_b.md) (finite-character, 3D SCF +
finite-torus correlation). They are under active side-by-side study on a
28-system benchmark, but no cross-approach delta is currently reportable. Band structure
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
trajectories out of every `.qvf` archive (QVF v1.2, 40 canonical
writer kinds: 39 first-class viewer kinds plus `bonds` via the
structure renderer). The
[`basis_toolkit`](qvf_basis_developer_readme.md) for
basis-set import/export (BSE, CRYSTAL, G94, NWChem, ORCA). The
[`vq` queue](user_guide/queue.md) for remote job submission.

See the [feature matrix](features.md) for details and the
[roadmap](roadmap.md) for what's next.

## Where to go next

```{toctree}
:maxdepth: 2
:caption: Getting started

getting_started
installation
toolset_lifecycle
quickstart
using_claude
running
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
design_basis_ao_section
design_periodic_gapw
design_mdf
design_smearing
design_aiccm2026dev_b
design_shared_symmetry
qvf_basis_developer_readme
qvf_basis_conversion_matrix
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
features
validation
roadmap
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
site_publishing
contributing
contributor_setup
developer_test_lanes
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
remain a gated preview). The eigensolver framework (Davidson / LOBPCG +
experimental Jacobi-Davidson / GPLHR) is selectable by keyword.
Analysis tools (COOP/COHP, QTAIM, Mayer bond orders, fat bands)
and the basis_toolkit import/export system shipped in v0.15.0.
The ab-initio CCM (Γ-CCM and χ-CCM) ships as experimental.
Follow the [roadmap](roadmap.md) for what's next.

Licensed under the [Mozilla Public License 2.0](license.md). Source at
[github.com/vibe-qc/vibe-qc](https://github.com/vibe-qc/vibe-qc).

## Feedback and bug reports

Found a bug, have a feature request, or want to send a patch?
The decision tree lives in
[CONTRIBUTING.md](CONTRIBUTING.md). The short version:

- **Bugs / install problems / feature requests** →
  [GitHub issues](https://github.com/vibe-qc/vibe-qc/issues)
- **Security vulnerabilities** → email `mpei@vibe-qc.com` (see
  [SECURITY.md](https://github.com/vibe-qc/vibe-qc/blob/main/SECURITY.md))
- **General feedback / questions** → email `mpei@vibe-qc.com`
- **Sponsor the project** → see [Support vibe-qc](support.md)
  ([GitHub Sponsors](https://github.com/sponsors/mpeintinger),
  [Ko-fi](https://ko-fi.com/mpeintinger))
