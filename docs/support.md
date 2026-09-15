---
myst:
  html_meta:
    "description": "Support vibe-qc, sponsor the development of an open-source quantum-chemistry code targeting the Cyclic Cluster Model at correlated levels of theory. GitHub Sponsors and Ko-fi."
    "og:title": "Support vibe-qc"
    "og:description": "Sponsor open-source quantum chemistry. Funds Claude Max + self-hosted server + future hardware. GitHub Sponsors and Ko-fi."
    "og:type": "website"
    "og:url": "https://vibe-qc.com/docs/support.html"
---

# Support vibe-qc

vibe-qc is free and open-source software (MPL 2.0), developed and
maintained by one person on personal hardware. Development happens
on a MacBook. Test calculations run on a personal gaming PC. The
server that hosts [vibe-qc.com](https://vibe-qc.com/), the GitLab
repository, and the CI/CD pipeline is self-hosted and self-funded.

There is no institutional backing, no grant, no university compute
allocation. If vibe-qc is useful to you, or if you think the
Cyclic Cluster Model reaching CCSD(T) accuracy on a publicly
available open-source code is worth existing, consider supporting
the project.

## About the author

I am a computational chemist with a PhD from the Mulliken Center
for Theoretical Chemistry in Bonn. My thesis work focused on
solid-state quantum chemistry: I developed the **pob-TZVP basis
sets** for periodic calculations with CRYSTAL ([Peintinger, Vilela
Oliveira, Bredow, *J. Comput. Chem.* **34**, 451,
2013](https://doi.org/10.1002/jcc.23153)) and implemented the
**Cyclic Cluster Model at the Hartree-Fock level** as a proof of
concept ([Peintinger, Bredow, *J. Comput. Chem.* **35**, 839,
2014](https://doi.org/10.1002/jcc.23550), free-access cover
article). Both papers, and the rest of my publication history,
are on my [Google Scholar
profile](https://scholar.google.com/citations?user=Y0SwM14AAAAJ&hl=en).

After my postdoc I left academia for the steel industry, where I
work today. **vibe-qc is a hobby project**, developed in the
evenings and on weekends. That context matters for understanding
what it is and where it is going.

vibe-qc is a modern reimplementation of the ideas behind that PhD
work, rebuilt from scratch using current tools and without the
constraints of a thesis timeline. The long-term goal is to take
the Cyclic Cluster Model as far up the post-Hartree-Fock ladder as
I can reasonably get: **MP2-CCM, local MP2, CCSD, and eventually
CCSD(T)**, with projection-based embedding for the environment.
That would mean correlated quantum chemistry on solid-state
defects at a cost that scales with cluster size rather than unit
cell size, available to anyone with a `pip install`. I wrote about
how the first day of development went here: [*I vibe-coded a
quantum-chemical program in one
day*](https://vibe-qc.com/2026/04/22/i-vibe-coded-a-quantum-chemical-program-in-one-day/).

The code is being built in the open using **Claude as the
implementation engine and my domain expertise as the steering
wheel**. What used to take months of PhD-level implementation
work, interfacing integral libraries, writing parallel code,
hunting down wrong signs in density matrices, now takes hours.
The science is still hard. The implementation overhead is no
longer the bottleneck. That asymmetry is what makes this tractable
as a hobby.

## What your sponsorship funds

There are recurring costs that keep development running at all,
smaller annual sponsorship opportunities, and a one-off **hardware
roadmap** that defines what *kinds of problems* vibe-qc can be
developed and tested against. All of them matter.

### Recurring costs (always-on)

- **Claude Max subscription** (~$200/month), the AI coding
  assistant that is the direct engine of development. Without it,
  vibe-qc does not get built at the current pace.
- **Self-hosted server costs**, [vibe-qc.com](https://vibe-qc.com/),
  the GitLab repository, and the CI/CD pipeline all run on
  personal hardware.

(macos-desktop-auto-update)=
### macOS desktop auto-update is donation-gated

The [vibe-view desktop app](user_guide/vibe_view_desktop.md) already checks
the update feed and notifies you when a newer build ships, but on **macOS**
it cannot install the update by itself. Apple only lets an app self-update if
it is signed with a paid **Apple Developer ID** (~USD 99/year); until that
certificate is funded, the macOS app detects the new version and offers a
manual **Download…** button instead of a hands-off restart-to-install. **A
single USD 100 donation** via
[**GitHub Sponsors**](https://github.com/sponsors/mpeintinger) or
[**Ko-fi**](https://ko-fi.com/mpeintinger) covers the certificate for a year
and enables real macOS auto-update, with no code change needed. Linux and
Windows do not require this Apple certificate, although signed and supported
download artifacts are not currently published for those platforms.

### 📚 Data licences (small annual items)

Some of the work needs paid data subscriptions that an academic
group would get bundled in their institutional licence, but
vibe-qc has none. These are the smallest, most actionable
sponsorship items: any single supporter can fund a year outright.

- **[NIST Crystal Data, SRD 3](https://www.nist.gov/srd/nist-standard-reference-database-3)**
  - single-user **annual subscription, $200/year**. Curated
  crystallographic database; the kind of authoritative
  reference-structure source needed to build the **solid-state
  quantum-chemical reference database** vibe-qc's v2.x Cyclic
  Cluster Model track is aimed at. With no institutional access,
  this subscription is the difference between *referencing the
  literature for every benchmark structure by hand* and
  *querying a curated database programmatically*. **A single
  sponsor can fully fund a year**, one of the easiest concrete
  contributions to make, and a year of access carries the
  v2.x reference-database build cleanly through the work it
  enables.

(compute-time-on-your-cluster)=
### 🧮 Compute time on your cluster

Direct financial sponsorship is the headline ask, but for
academic groups with cluster access there is a second,
equally valuable contribution: **let me use your cluster**.

vibe-qc has no institutional compute allocation. Reference
benchmarks against ORCA, PySCF, and CRYSTAL on production-
scale ionic crystals, the kind of comparison that makes the
release papers credible and that flushes out scaling bugs
the dev hardware never reaches, currently run on a personal
gaming PC and a single 16-core/128 GB self-hosted box.
That ceiling is the actual rate-limiter on a meaningful
chunk of the v0.x → v1.0 roadmap, ahead of the hardware
fund's own timeline.

The ask is intentionally simple: **a guest user account on
your group's cluster**. No formal allocation, no
service-grant paperwork, no priority queue, just an account
that lets me submit jobs. Even modest, low-priority access
to a big-memory node, an Apple Silicon node, or a GPU node
unblocks comparison work the dev hardware cannot reach.

Compute providers are credited in release notes and in the
acknowledgements of any paper whose results the access
enabled, the same convention CRYSTAL, Turbomole, and ORCA
papers use for CINECA, JSC, HLRS, etc.

If your group can spare a guest account, please reach out:
**[mpei@vibe-qc.com](mailto:mpei@vibe-qc.com)**.

### 🖥️ Hardware roadmap

vibe-qc is currently developed on a personal laptop. To accelerate
development and let the project tackle production-scale quantum
chemistry problems rather than toy systems, we are raising funds
for a dedicated development workstation. This section describes
**what the hardware enables, what the two funding tiers are, and
what your contribution buys in concrete terms.**

#### Why hardware matters for quantum chemistry

Quantum chemistry has unusual hardware requirements compared to
typical scientific software. Three properties dominate:

1. **Memory capacity.** Wavefunction tensors, two-electron
   integrals, and density matrices grow steeply with system size.
   A 30-40 atom molecule with a triple-zeta basis set can require
   60-100 GB of RAM for correlated methods like CCSD(T). On smaller
   hardware, development is limited to molecules with a handful of
   heavy atoms, fine for unit tests, inadequate for benchmarking
   against published reference data or the real molecules users
   actually care about.

2. **Memory bandwidth.** Most of the wall-clock time in correlated
   methods (MP2, CCSD, CCSD(T), CASPT2) is spent on tensor
   contractions. These operations are bandwidth-bound, not
   compute-bound. **Doubling memory bandwidth roughly doubles
   throughput on the rate-limiting steps.**

3. **Unified memory + Metal GPU support.** Apple Silicon shares
   one high-bandwidth memory pool between CPU and GPU, with no
   PCIe bottleneck. vibe-qc is being designed with first-class
   support for Apple's MLX framework and Metal GPU acceleration,
   currently a gap in the wider ecosystem (most GPU-accelerated QC
   codes target CUDA only). A capable Apple GPU lets us validate
   and tune these code paths properly.

#### Funding tiers

##### Tier 1, Development workstation, ~$3,499

**16-inch MacBook Pro, M5 Pro chip, 64 GB unified memory, 2 TB SSD,
standard display.**

| Spec | Value |
|---|---|
| CPU | 18 cores (6 performance + 12 efficiency) |
| GPU | 20 cores |
| Memory | 64 GB unified |
| Memory bandwidth | 307 GB/s |
| Storage | 2 TB SSD |

What this tier unlocks:

- DFT benchmarks on systems up to roughly 50 heavy atoms with
  triple-zeta basis sets.
- MP2 and CCSD on small-to-medium organic molecules.
- Initial MLX and Metal GPU development on the 20-core GPU.
- Fast iteration on core SCF, integral evaluation, and tensor
  contraction code paths.

**This is the threshold where development stops being toy-system-
bound and becomes useful for testing against real published
benchmarks.**

##### Tier 2, Full development capability, ~$5,599

**16-inch MacBook Pro, M5 Max chip, 128 GB unified memory, 2 TB
SSD, standard display.**

| Spec | Value |
|---|---|
| CPU | 18 cores (6 performance + 12 efficiency) |
| GPU | 40 cores |
| Memory | 128 GB unified |
| Memory bandwidth | 614 GB/s |
| Storage | 2 TB SSD |

What this tier *additionally* unlocks beyond Tier 1:

- Larger correlated-method validation cases and requested-memory scaling
  experiments. It does **not** establish production CCSD(T) coverage for
  60-100-heavy-atom molecules; BUG 63 remains gated on measured bounded-memory
  executions at those sizes.
- Roughly **2× speedup** on bandwidth-bound tensor contractions,
  the rate-limiting step for correlated methods.
- Serious GPU development with **40 Metal cores** and **614 GB/s**
  bandwidth shared between CPU and GPU.
- Headroom for the next four-to-five years of feature work without
  hitting hardware ceilings.

The Max tier is what allows vibe-qc to be **benchmarked against
ORCA, PySCF, and CRYSTAL on the kinds of problems users will
actually run**, not just the small molecules that fit on the Pro
tier.

#### Why a laptop, not a desktop or server?

A reasonable question. Three reasons:

1. **Conferences and demos.** vibe-qc is presented and demonstrated
   at scientific meetings. A portable workstation lets the same
   machine that runs production benchmarks also run live demos.
2. **Apple Silicon is the development target.** Since one of the
   goals is first-class MLX and Metal support, the development
   machine has to be Apple Silicon. The 16-inch MacBook Pro is the
   most capable Apple Silicon device available with 128 GB of
   unified memory.
3. **Power efficiency.** The same chip runs cool and quiet on
   battery and is competitive with desktop workstations on
   sustained load. There is no separate "travel laptop plus dev
   desktop" overhead.

#### Beyond the laptop, the long-term cluster

Once a Tier 1 or Tier 2 development workstation is funded, the next
hardware milestone is a **self-hosted mini-cluster** for routine
CI benchmarks against reference codes on real ionic crystals (MgO
supercells, perovskites, slab models), validated commit-by-commit
instead of by hand. This eventually grows into a multi-node MPI
testbed as vibe-qc moves toward the Cyclic Cluster Model at
correlated levels of theory. Estimated cost ~$15-25k for a
credible starting node count, but it makes more sense to fund
*after* the development bottleneck is solved.

## How to support

::::{grid} 2
:gutter: 3

:::{grid-item-card} 💚 GitHub Sponsors
:link: https://github.com/sponsors/mpeintinger
:link-type: url

Recurring monthly support. Zero fees. Preferred for ongoing
contributors.
:::

:::{grid-item-card} ☕ Ko-fi
:link: https://ko-fi.com/mpeintinger
:link-type: url

One-time donations. No GitHub account required.
:::

::::

Every contribution goes toward the **hardware fund** until the
relevant tier is reached. Progress updates are published, and we
document which features and benchmarks shipped as a result of each
tier being funded. Sponsors at any level are acknowledged in the
release notes for the versions their contribution helped fund.
Sponsors who choose to be listed publicly appear on the dedicated
[sponsors page](sponsors.md).

For organizations interested in larger sponsorships, commercial
support arrangements, or feature-directed funding, please reach
out directly: **[mpei@vibe-qc.com](mailto:mpei@vibe-qc.com)**.

## Other ways to help

- **Donate compute time** on your group's cluster, see the
  [🧮 Compute time](#compute-time-on-your-cluster) section
  above. For academic groups with allocations, this is often
  more impactful than direct funding.
- **Star the repository** on
  [GitLab](https://github.com/vibe-qc/vibe-qc).
- **Report bugs** and install failures via
  [GitLab Issues](https://github.com/vibe-qc/vibe-qc/issues),
  the v0.4 bug arc showed that non-dev machines surface problems
  the dev machine never will.
- **Contribute a tutorial**, the format is documented in the
  [tutorial guide](tutorial/index.md).
- **Cite vibe-qc** in publications, see the [citation
  guide](citing.md) for the software citation, the pob-TZVP basis
  paper to cite when you use those basis sets, and the libint /
  libxc / spglib references. The repository ships a
  [`CITATION.cff`](https://github.com/vibe-qc/vibe-qc/blob/main/CITATION.cff)
  that GitLab and citation managers parse automatically.
- **Spread the word**, if you work in computational chemistry or
  solid-state physics and vibe-qc covers your use case, telling
  colleagues is genuinely useful.
