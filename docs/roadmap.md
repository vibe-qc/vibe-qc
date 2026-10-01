# Roadmap

vibe-qc follows semantic versioning. Pre-1.0 releases are milestone-based:
each `0.N.0` tag marks the completion of a major capability. `1.0` is
reserved for "feature complete across both molecular and periodic
methods, everything a standard QC program offers." The **cyclic cluster
model**, the original motivating feature, now ships as experimental
Γ-CCM and χ-CCM lines in the `v0.15.0` release; the `v2.x` track is
re-scoped to production-grade correlated scaling and embedding.

**Parity targets, in order.** "Everything a standard QC program offers"
is measured against three real programs, in sequence, not against an
abstract feature list:

1. **ORCA** (molecular) **and CRYSTAL23** (periodic), together, as the
   `v1.0` bar. Both are tracked concurrently, not ORCA-then-CRYSTAL,
   because vibe-qc is molecular-and-periodic from the ground up (§ 14's
   molecular-first *gating* rule sequences individual features, it does
   not mean periodic parity waits for a finished molecular product).
   The [Tutorial parity matrix](#tutorial-parity-orca--crystal) below
   and the two [Cross-code feature gaps](#cross-code-feature-gaps-orca-611--crystal23-manual-sweep)
   sweeps are this target's live scorecard.
2. **CP2K**, after `v1.0`. CP2K is already vibe-qc's closest periodic
   sibling by lineage, GPW/GAPW (shipped, v0.10.x-v0.12.0), ADMM,
   OT-SCF, and Kerker/Anderson density mixing are all CP2K-originated
   methods already on this roadmap, cited throughout the Acceleration
   and SCF-convergence programs below. What CP2K parity adds *beyond*
   that overlap: production linear-scaling DFT for insulators (ONETEP/
   BigDFT/CONQUEST/SIESTA-style order-N, noted as "far future" under
   locality exploitation below), DBCSR-style sparse-tensor GPU
   contraction, and battle-tested `mpirun`-scale MPI (the production MPI milestone covers the
   vibe-qc-native version of the same target). Not scheduled as its own
   milestone; tracked as "the second parity target" so individual CP2K-
   originated items don't read as disconnected acceleration tricks.

No other program is a parity target. Comparisons to PySCF, Psi4, or
anyone else in this document are validation references (§ 10), not
goals to catch up to.

**Status.** The release table below is the one place this page states what
has shipped. It is checked against the repository rather than maintained by
hand: `tests/test_roadmap_release_table.py` fails if a codename in the table
disagrees with `RELEASE_CODENAMES` in `python/vibeqc/banner.py`, if a minor
that has a `CHANGELOG.md` section or a git tag has no row, or if a row marked
planned has either. The hand-maintained "this release is vX.Y.Z" line that
pass 1 left here is gone: it named `v0.17.3` six releases after the fact, and
no prose on this page names a current release any more. What each release
contains is recorded in [the changelog](changelog.md); the shipped milestones
below summarize and point there.

## Release table

Every released minor, checked against `RELEASE_CODENAMES`, `CHANGELOG.md` and
the repository's tags. Patch releases inherit their minor's codename and are
not listed; the changelog has them all.

| Version | Codename | Headline | Status |
|---|---|---|---|
| `v0.1.0` | Roothaan's Raven | Molecular HF/DFT/MP2 validated against PySCF; periodic scaffold with Gamma and multi-k RHF | shipped 2026-04-18 |
| `v0.2.0` | Ewald's Eagle | Quantitative 3D bulk: Ewald electrostatics and the exact BIPOLE J split | shipped, pre-split |
| `v0.2.5` | Wigner's Wolf | Space-group symmetry exploitation (core) | shipped, pre-split |
| `v0.3.0` | Mulliken's Magpie | Visualization and post-SCF properties | shipped, pre-split |
| `v0.4.0` | Hellmann's Hedgehog | Heavy elements through ECPs, open-shell periodic SCF, tight-cell convergence | shipped 2026-04-27 |
| `v0.5.0` | Wilson's Otter | Molecular Hessian, vibrations and thermochemistry; full k-point sampling | shipped 2026-04-30 |
| `v0.6.0` | Pulay's Owl | Periodic atomic gradients | shipped, pre-split |
| `v0.7.0` | Löwdin's Compass | Periodic SCF correctness and PySCF.pbc parity; linear-dependence handling | shipped, pre-split |
| `v0.8.0` | Grimme's Gecko | Periodic SCF chain over all Bravais lattices; D3/D4 dispersion and the double-hybrid foundation | shipped, pre-split |
| `v0.9.0` | Knowles's Kingfisher | Post-HF correlation ladder (MP3/MP4/CCD/CCSD) | shipped 2026-05-22 |
| `v0.10.0` | Pisani's Penguin | GPW/GAPW periodic route | shipped 2026-05-27 |
| `v0.11.0` | Sun's Stingray | Periodic GDF parity with PySCF, Gamma and multi-k; RSGDF | shipped 2026-06-03 |
| `v0.12.0` | Knuth's Beaver | Algorithmic-density release: DLPNO pair classification, selected CI, CPCM | shipped 2026-06-12 |
| `v0.13.0` | Wisesa's Fox | Generalised regular k-grids; mixed density fitting | shipped 2026-06-17 |
| `v0.14.0` | Bartlett's Goose | CCSD(T) and the coupled-cluster response framework; basis-set optimisation | shipped 2026-06-22 |
| `v0.15.0` | Neese's Cheetah | Comprehensive feature freeze: DLPNO-CCSD(T), QTAIM, COOP/COHP, fat bands | shipped 2026-06-26 |
| `v0.16.0` | Pople's Puffin | The repository split into vibe-qc, vibe-view, vibe-queue and qvf | shipped 2026-09-08 |
| `v0.17.0` | Tew's Tern | Symmetry and periodic foundations for periodic local correlation | shipped 2026-09-09 |

"Pre-split" marks a release tagged in the monorepo, before the 2026-09-08
split; its tag did not carry over, so this repository cannot date it and the
table does not guess. `CHANGELOG.md` starts at `v0.9.0`; earlier releases are
covered by its "Historical Codename Artwork (pre-v0.9.0)" section.

Open milestones are deliberately absent from this table. They are themes, and
a theme gets a version number only at the cut (see below), so a planned row
would be the drift this table exists to prevent.

## Experimental lines (not release-gated)

Maintainer decision, 2026-09-13: experimental work gets its own section
rather than sitting inside the production milestones, because a reader
scanning a shipped milestone should not have to know which of its bullets
carry a runtime warning.

**Nothing in this section is release-gated.** These routes run, they are
regression-tested, and a release is not held for them. They are not
validated for published numbers, and the milestone sections further down
describe their *plans*, not their status. Where a route warns at runtime,
the warning class is the authority; where it fails closed, the refusal is.

| Line | What says so at runtime | First in `CHANGELOG.md` | Scope, and what is not claimable |
|---|---|---|---|
| GAPW all-electron augmentation, range-separated hybrids, OT solver | `GAPWExperimentalWarning` | v0.10.1 | The GPW surface underneath is production for RHF / RKS / UHF / UKS with pure DFT at Gamma and multi-k. Only the augmentation, the range-separated hybrids and the OT solver are experimental. The GAPW/GPW stress is finite-difference either way |
| AICCM: Gamma-CCM and chi-CCM | `AICCM2026DevAExperimentalWarning`, `AICCM2026DevBExperimentalWarning`; every SCF driver emits one | v0.14.0 for the AICCM name and the `run_ccm_*` drivers; cyclic-cluster groundwork from v0.12.0 | Four-center, real-gamma, neutral-bloch and chi variants. Per-variant refusals are part of the construction, not gaps: `mp2` and `ccsd` run on four-center and real-gamma, the two DLPNO treatments are real-gamma only, neutral-bloch has no correlation at all, and `optimize_cell` and `hessian` fail closed. Results are research output |
| SECCM (semiempirical cyclic cluster) | route gates in `vibeqc.semiempirical.routes.SemiempiricalRoutePlan` | v0.15.41 | Per method and dimension, closed shell. DFTB0-SECCM is scoped to an explicit repulsive-pair table; PM6-SECCM and OM2/OM3-SECCM need an explicit `allow_truncated_electrostatics=True` acknowledgement for a nontrivial cyclic topology |
| GFN2-xTB | `GFN2ExperimentalWarning` | v0.10.0 (the v0.9.0 milestone note below dates the semiempirical stack to that release) | Molecular and Gamma-periodic. Full-k fails closed. Gamma derivatives are FD-validated on every gradient component and all nine stress components; faithful periodic AES electrostatics and external `xtb` parity are the production gates |
| OM1 | `NDDOExperimentalWarning` | v0.10.0 with the rest of the NDDO stack | The published analytic core-valence ECP is not implemented, so bonds to heavy atoms come out about 0.3 A short. PM6, OM2 and OM3 carry their full published Hamiltonians and do not warn. OM1, OM2 and OM3 are retirement targets |
| Native D4 dispersion | `D4NativeExperimentalWarning` | v0.9.0 | Parity coverage is H-Ne; elsewhere it returns zero with a warning. External `dftd4` remains the default |
| Green's-function surface embedding | `EmbeddedSurfaceExperimentalWarning` | v0.15.114 | The 1D analytic foundation and the 3D Gaussian pipeline are regression-tested, but the method has not been shown to reproduce a thick-slab GDF reference to tight tolerance. Treat absolute energies and densities as qualitative |
| BIPOLE analytic-gradient previews and the quartet multipole far field | driver-level refusals: all four drivers default the multipole branch to `False` and an explicit `True` raises before setup | see the BIPOLE sections below | Production BIPOLE forces finite-difference the executed energy. The corrected- and legacy-gauge analytic implementations are gated previews with a narrower maintained regression envelope. The BIPOLE support and long-range audits further down are diagnostics, not a production route |

"First in `CHANGELOG.md`" is the earliest released section that names the
line, which is the earliest date this repository can evidence. `CHANGELOG.md`
starts at v0.9.0, so a line that predates it shows v0.9.0.

Two rows are here that the pass-2 scope did not list, native D4 and the
surface embedding. They carry their own experimental warning class on the
same footing as the rest, so leaving them in the production milestones would
have reproduced the problem this section exists to fix.

**Open milestones are themes, not version numbers** (maintainer decision,
2026-09-13). They are listed in their intended order, and a version number
is attached only when a release is cut. v0.16.0 and v0.17.0 each took the
number of a planned milestone they did not deliver, which renumbered the old
ladder twice.

**v0.15.0 "Neese's Cheetah" was a comprehensive feature-freeze release**
(frozen at e94221f2): the maintainer's scope decision was "add everything
we have ready, even if not roadmap-due; rework the roadmap rather than gate
scope by it." The milestones through v0.15.0 reflect that rework.

The milestone list below is the *engineering* roadmap. The
[Tutorial parity matrix](#tutorial-parity-orca--crystal) at the bottom
is the *user-facing* roadmap, pinned against ORCA's molecular
tutorials and CRYSTAL's solid-state tutorials so we can see what works
today and what's blocking the next chunk of "real chemistry."

## Milestones

A release's contents are recorded in the
[release table](#release-table) above and in
[the changelog](changelog.md). The sections below are the milestones still in
play; the shipped inventories for `v0.1.0` to `v0.15.0`, and the plans written
for them, moved to [roadmap history](roadmap_history.md) so this page reads as
a plan rather than as a record.

## Acceleration program (multi-release)

Wall-clock performance is the gating axis between vibe-qc as a
research playground and vibe-qc as a production tool. The acceleration
work splits into **integral / Fock-build acceleration**, **locality
exploitation**, **reciprocal-space and basis tricks**, **SCF
convergence acceleration**, and **algorithmic / hardware**. Each
sub-release below targets one box and stacks multiplicatively on the
ones below. Pre-v0.5.5 the periodic Fock build had no integral
screening at all; the very first item here closed a 100-1000× wall-
clock gap on real crystals like LiH/STO-3G/4×4×4. The remaining items
buy the next 100-1000×.

### `v0.5.5`, Cauchy-Schwarz screening (foundation, **shipped**)

🎯 The bedrock under everything below. Pre-screening, periodic
``build_fock_2e_real_space`` was O(n_c³ · n_shells⁴) libint quartet
calls per Fock build × ~30 SCF iters × 3 builds-per-iter for the
Ewald-split HF path, minutes-to-hours on real crystals. Schwarz
caps the integral magnitude analytically per quartet × cell triple:

  | ⟨μ_0 ν_g | λ_λ σ_σ⟩ |  ≤  Q[c_g][μ,ν] · Q[c_σ−c_λ][λ,σ]

with Q[c][s_a,s_b] = √|⟨s_a_0 s_b_c | s_a_0 s_b_c⟩| precomputed
once. Quartets bounded below ``LatticeSumOptions.schwarz_threshold``
(default 1e-12 Ha) skip the libint call. Translational invariance
collapses the Q tensor to a function of one relative lattice
vector; Gaussian shell-pair products decay exponentially with
|R|, so the lattice-sum truncation becomes rigorous instead of
ad-hoc. Wired into both ``build_fock_2e_real_space`` (the full real-
space ERI path) and ``build_jk_gamma_molecular_limit`` (the Γ-only
periodic J/K builder); both share the same per-cell Q precompute
helper. Coverage stops at the energy-side Fock build for v0.5.5;
gradient ERIs use a separate threshold (see v0.6.x below).

References: Häser & Ahlrichs 1989; CP2K's ``EPS_SCHWARZ`` (1e-7
energies, tighter for forces); CRYSTAL's ``TOLINTEG`` 5-vector.

### `v0.6.x`, separate exchange threshold + gradient screening

- **Sx1** ``LatticeSumOptions.schwarz_threshold_forces``: separate
  threshold for the gradient ERI pass (CP2K's
  ``EPS_SCHWARZ_FORCES`` convention). Default ``1e-14``, tighter
  than the energy threshold because force convergence is dominated
  by the slowly-decaying tail of the integral *derivative* (loose
  Schwarz on derivatives is non-rigorous; the standard fix is to
  tighten by ~100× relative to the energy threshold).
- **Sx2** Wire screening into ``eri_lattice_gradient_contribution``
  (the periodic Pulay-derivative pass). Same Q precompute, with
  ``schwarz_threshold_forces`` controlling the per-quartet skip.
  Required for production-quality periodic forces; the unscreened
  gradient pass is currently the dominant cost on a relaxation
  step.
- **Sx3** Density-weighted screening (LinK-style, Ochsenfeld 1998):
  fold the local |D_λσ| block into the bound, replacing the
  conservative ``D_max`` envelope. In an insulator the density
  matrix decays exponentially in real space, which is what enables
  genuine **O(N) exchange** for periodic gapped systems. Metallic
  systems remain harder by construction, the density matrix does
  not decay, and any production-quality periodic hybrid for metals
  needs ADMM (see v0.9.x acceleration row) or smearing-aware
  screening.

### `v0.7.x`, RI / density fitting + RIJ-COSX (1st acceleration wave)

🎯 The single biggest multiplicative lever. Expands the product
density ρ_μν(r) = χ_μ(r)·χ_ν(r) in an auxiliary basis, turning
4-index ERIs into contractions of 3-index and 2-index quantities.
Formal cost drops from N⁴ to N³ with a small prefactor; accuracy is
controllable via the auxiliary basis.

- **RI1** RI-J for the Coulomb build, essentially free at chemical
  accuracy (Eichkorn et al. 1995; Whitten 1973). Auxiliary basis
  loading from libint's built-in JKfit / Cfit families.
- **RI2** RI-K for exchange (loose-fit / chain-of-spheres-style).
- **RIJCOSX** seminumerical exchange via Neese's chain-of-spheres
  (COSX), pairs RI-J with COSX-K for 10-100× speedups on hybrid
  DFT in molecules. The workhorse in ORCA. Variant: sn-LinK
  (Laqua / Ochsenfeld) with rigorous integral screening.
- **PBC-RI** Periodic RI, 3-index ERIs over real-space lattice
  sums + Ewald-split aux Coulomb metric (Sun, Berkelbach et al.
  for PySCF.pbc). Couples to the v0.5.x Ewald split machinery.

### `v0.8.x`, fast multipole methods (long-range Coulomb)

> **Re-versioned to v1.x / research direction (2026-06 catch-up audit).** This
> whole cluster, production molecular **FMM1 / CSAM** (the near-field plus
> far-field continuous-FMM split) and **PBC-FMM** (periodic FMM, Kudin &
> Scuseria), was optimistically dated `v0.8.x` but was never committed as a
> stub: no enum, kwarg, or raising code path reaches it. Molecular FMM/CFMM
> needs the `distance_cutoff` C++ JKBuilder work and is self-dated v0.12 to
> v0.13 in its own `MultipoleJKBuilder` foundation module; PBC-FMM exists only
> as driver-unreachable BIPOLE quartet prototypes, not a periodic FMM tree.
> A research direction, not missed v0.8 to v0.13 catch-up work.

For the long-range 1/r₁₂ part of the J build. Schwarz alone fails
here because the operator decays only conditionally; the standard
fix is to hierarchically group distant charge distributions into
multipole expansions.

- **FMM1** Continuous fast multipole method (Strain / White-Head-
  Gill / GvFMM). O(N) for the Coulomb part of J.
- **PBC-FMM** Periodic FMM (Kudin & Scuseria 1998 / 2004),
  handles lattice sums directly without Ewald reciprocal-space
  machinery. Alternative to the v0.5.x Ewald split for systems
  where FMM is more efficient.
- **CSAM** Combined Schwarz + multipole (CFMM-style). Schwarz in
  the near field, multipole estimates in the far field.

### Post-v0.15: BIPOLE integral acceleration (POLIPO)

The exact Ewald-J BIPOLE route is the supported production path. The
quartet-level bipolar prototype is not enabled at any cutoff: explicit
`use_multipole_far_field=True` raises because its two-translation dispatch
does not preserve the exact three-translation Fock domain. The quartet
formula is in Pisani, Dovesi, and Roetti (1988), Chapter II.4c; Saunders et
al. 1992 supplies related electrostatic-potential and penetration context,
not the dispatcher. Two research avenues remain, both contingent on a
correct and route-certified redesign:

- **POLIPO** (post-v0.15): a native C++ alternative to the vendored
  libint for Gaussian multipole-moment integrals and their derivatives.
  libint currently supplies `emultipole2`/`emultipole3`/`sphemultipole`
  which are sufficient for L≤4 (hexadecapole).  POLIPO engine
  landed 2026-08-11 (McMurchie-Davidson Hermite kernel, OpenMP
  shell-pair lattice loop, pybind11 bindings, `multipole_engine` parameter).
  All 6 POLIPO-vs-libint Cartesian parity tests pass at machine precision
  (H₂ L=2/3, MgO L=2).  Current performance: ~200× slower than
  libint (107 ms C++ + 102 ms Python overhead vs 1 ms for MgO/STO-3G at
  cutoff 10 bohr).  Optimization targets: pre-computed Hermite tables,
  recurrence relations replacing pow/binom/factorial, and moving the Python
  post-processing (Eigen→numpy copy + pure-shell permutation) into C++.
  Within the low-level prototype, `multipole_engine="libint"` remains the
  comparison default. Neither engine is used by the supported exact SCF route.
  POLIPO would:
  - Unify Cartesian and spherical moment computation in a single
    self-contained code path (eliminating the libint HRR-gate patch).
  - Support higher L (8-12) for tighter multipole convergence.
  - Exploit shell-pair recurrence relations for faster L-scaling.
  - Integrate with the existing C++ pair-centre shift and buffer
    infrastructure.
  - Be selectable via a runtime parameter
    (`multipole_engine="libint"` or `"polipo"`) for cross-validation.

- **BIPOLE far-field MPI** (production MPI milestone, MPI4): after route certification,
  farm far-field quartet batches across MPI ranks for very large supercells
  (200k+ quartets).
  Architecture: master rank distributes dispatch entries across workers;
  each worker computes local far-field Fock contributions; reduction
  via MPI_Allreduce on the per-cell Fock blocks.  The infrastructure
  is designed to reuse the existing OpenMP C++ contractor with an
  MPI-aware dispatch splitter.  Python bindings via mpi4py or a
  lightweight C API. This is blocked on the numerical route redesign; the
  current OpenMP contractor is research-only infrastructure.

### `v0.9.x`, locality exploitation (hybrid DFT for big cells)

Once exchange is the bottleneck and gapped systems are the target,
the locality of the density matrix becomes the dominant lever.

- **ADMM** Auxiliary density matrix method (VandeVondele, Krack,
  Mohamed et al., CP2K). Computes exact exchange in a *smaller*
  auxiliary basis and corrects with a GGA exchange difference
  term. The reason HSE06 calculations on 1000+ atom periodic
  systems are feasible in CP2K. Critical for the catalysis /
  electrochemistry use case.
  **Re-versioned to v1.x / research direction (2026-06 audit):** ADMM exists
  only as a `database.toml` citation entry with a dormant
  `routes.acceleration."admm"` route and no compute path, never a committed
  stub, so not v0.9.x catch-up debt.
- **OT-SCF** Direct minimisation via orbital transformation
  (VandeVondele & Hutter, JCP 2003). Avoids diagonalisation,
  scales linearly in nbf for large cells, eliminates the Aufbau
  / occupation ambiguity. The standard for large CP2K runs.
- **Linear-scaling DFT** for insulators (nearsightedness, Kohn).
  Localised orbitals (NGWFs / support functions) optimized in
  situ, ONETEP, BigDFT, CONQUEST, SIESTA's order-N. Far-future,
  but the exponential decay of P(R) in gapped systems makes this
  natural for vibe-qc's CCM target.

### `v0.10.x`, reciprocal-space and basis tricks

**✅ GPW shipped on `main` (production as of v0.12); GAPW shipped
experimental.** The Gaussian + plane-wave route landed end-to-end:
`run_periodic_{rhf,rks,uhf,uks}_gpw` (Γ + multi-k pure-DFT), with DIIS,
density damping, analytic forces, finite-difference Hessians, DOS /
band-path helpers, `.npz` restart, and the `VibeqcGPW` ASE calculator,
reproducing the molecular RHF limit to sub-µHa. The all-electron
**GAPW** augmentation (`run_periodic_*_gapw`) also ships but stays
**opt-in behind `GAPWExperimentalWarning`**: single atoms match the
molecular all-electron reference to <1 mHa, but molecules over-bind
(overlapping augmentation spheres), and meta-GGA / open-shell GAPW DFT
raise `NotImplementedError`. The bonded-molecule overlapping-augmentation
double-count itself was fixed via the partition-of-unity weight
(`_partition_weight`, 2026-06-26; see the ✅ entry under v0.15.0 below) --
the remaining accuracy residual is multi-atom overlapping-sphere
precision on tight crystals, not the partition-of-unity mechanism itself.
**Remaining:** that multi-atom accuracy residual, the PAW
atomic-data path, and the **PAW**, **ACE**, and range-separated
refinements below (PAW and ACE re-versioned to v1.x / research, see their
notes below). See [user_guide/gapw.md](user_guide/gapw.md).

- **PAW** Projector augmented wave (Blöchl 1994). All-electron
  accuracy at modest plane-wave cutoffs. Standard for VASP /
  ABINIT / GPAW.
  **Re-versioned to v1.x / research direction (2026-06 audit):** citation-only;
  a Gaussian-orbital code has no PAW atomic datasets
  (`periodic_gapw_augment.py`), a research direction, not missed v0.10.x work.
- **GAPW** Gaussian-augmented plane-wave (Lippert / Hutter,
  CP2K). Pure-Gaussian periodic codes plus an auxiliary plane-
  wave grid for the Hartree term, linear scaling for J.
- **GPW** Pure GPW (the Hartree-only variant of GAPW; pseudo only).
- **ACE** Adaptively compressed exchange (Lin Lin 2016).
  Reformulates the non-local exchange operator into a low-rank
  form. Order-of-magnitude speedups on plane-wave HSE / PBE0
  in Quantum ESPRESSO. Translation to Gaussian-based periodic is
  an open research question; revisit when ACE bibliography
  matures.
  **Re-versioned to v1.x / research direction (2026-06 audit):** the roadmap
  itself flags the Gaussian-periodic translation as an open research question
  (above), so not v0.10.x catch-up debt.
- **Range-separated split refinements** Beyond the v0.5.x Ewald-
  3D split: tighter ω heuristics + per-system ω optimization.

### `v1.x`, native McMurchie-Davidson integral engine (SHARK-style)

Research direction, not a committed stub. ORCA replaced its exclusive
libint dependence with **SHARK** (ORCA ≥5.0, Neese group), a native
McMurchie-Davidson integral engine that factorizes the four-center ERI
into a triple matrix product (μν|κτ) = (E_bra · R · E_ket)_μν,κτ, with
the bra/ket E-coefficient matrices folding contraction and Cartesian→
spherical transforms into themselves. Two implications for vibe-qc:

- **General contraction is the actual payoff.** ANO-style basis sets
  (many primitives, few contracted functions, every primitive feeding
  every function of a given l) are redundant under naive quartet loops
  (N_l⁴ duplicate integral evaluations); SHARK's factorization
  generates a whole atom-quadruple/angular-momentum-quadruple block
  via two matrix multiplications instead. vibe-qc's own
  general-contraction basis work (v0.8.0 basis-set library expansion,
  above) would gain the same speedup from a native ERI path built this
  way, instead of routing everything through libint.
- **High angular momentum (l>2) is where matrix multiplication wins.**
  ORCA's own crossover data: quartets of f-/g-functions run up to 5×
  faster than traditional recursive quartet code; intermediate l stays
  libint's turf, low l performs fine either way. ORCA ships both
  engines and dispatches per-quartet (`FockFlag SHARK_libint_hybrid`).

This is **not** an ask to run ORCA: § 10 forbids it as a runtime
dependency outright, and SHARK itself is closed-source, unavailable to
vendor even if the license permitted it. The item on the table is a
**from-scratch native implementation** of the McMurchie-Davidson
factorization (a public algorithm, textbook Helgaker/Jørgensen/Olsen
§9.10, independently re-derivable from first principles) as an
*additional* internal ERI backend alongside vendored libint, mirroring
ORCA's own hybrid-dispatch idea rather than a wholesale swap. Revisit
once general-contraction basis sets (the ANO family) or heavy f-/g-
shell workloads (lanthanide/actinide ECPs, high-l correlation-
consistent bases) become a real bottleneck; until then libint's
segmented-contraction performance is not the limiting factor anywhere
in the current SCF profile.

Reference: Neese group, *SHARK Integral Package and Task Driver*, ORCA
6.1.1 Manual §2.15; McMurchie & Davidson, *J. Comput. Phys.* 26, 218
(1978), doi:10.1016/0021-9991(78)90092-X.

### `v1.x`, algorithmic and hardware acceleration

- **GPU acceleration** Integral evaluation, tensor contractions,
  cubic-scaling diagonalisations all map well to GPUs (TeraChem /
  BrianQC / GPU4PySCF / GPU CP2K via DBCSR). Reduced precision
  (mixed FP32/FP64, FP16 for some steps) gives further 2-4×.
- **Tensor decompositions** THC (tensor hypercontraction) /
  Cholesky decomposition / ISDF (interpolative separable density
  fitting, increasingly popular for periodic GW / BSE). CD is
  essentially RI with a numerically generated auxiliary basis,
  controllable accuracy for free.

### Beyond SCF, local correlation methods

Pinned here for completeness; the post-HF analog of v0.9.x
locality. Already on the canonical roadmap below as v0.15+:

- **DLPNO-CCSD(T)** (Riplinger / Neese), near-canonical accuracy
  at a small fraction of the cost. PNO-MP2 / LNO-CCSD(T) /
  Kallay's variants / CIM (Li) are siblings. Periodic translation
  via fragment methods + embedded cluster (CCM!).

### SCF convergence accelerators (mostly at v0.5.x scope)

These come earlier than the multi-release waves above because they
are mostly Python-level + small C++ hooks:

- **EDIIS / ADIIS / A-DIIS** (Kudin / Garza), improvements on
  Pulay DIIS for difficult cases (transition metals, broken
  symmetry).
- **Second-order SCF** (Newton-Raphson / SOSCF / Augmented
  Hessian). Essential for transition-metal / open-shell and BS
  cases that DIIS plateaus on.
- **Better initial guesses** SAD (atomic density superposition),
  SAP (atomic potential), GWH (generalised Wolfsberg-Helmholz)
  for molecules. Density mixing across k-points + careful k-mesh
  for solids.

### Prioritisation summary

If standing up the periodic capability for catalysis /
electrochemistry / solid-state today, the **first wave** to deploy
on top of Schwarz is:

  1. **RI-J** + **ADMM** for hybrids (covers HSE06 on 1000+ atoms).
  2. **OT-based SCF** for large systems (avoids the diagonalisation
     wall).
  3. **DLPNO-CCSD(T)** on cluster models for benchmarking.

The **second wave** layered on top:

  4. **COSX** / **sn-LinK** if molecular hybrid DFT dominates
     workload, **ACE** if plane-wave hybrids do.
  5. A serious **GPU** build of whichever code is the production
     tool.

Schwarz screening sits underneath all of this as the foundation;
RI + ADMM + OT + GPU stacked on top is where the wall-clock
factors of 100× and 1000× actually come from.

## SCF guess + convergence program (multi-release)

Performance gets you a fast Fock build per iter; **a good guess plus
a robust SCF accelerator gets you fewer iters and higher convergence
rate on hard cases**. Both axes need work. This program tracks the
end-to-end SCF workflow: where you start (initial guess), how you
progress (damping / DIIS / second-order), and how you stop (how
robustly + how cheaply). The whole program is **wired uniformly
across all four flavors**, RHF / UHF / RKS / UKS, and across
**both molecular and periodic** entry points. CRYSTAL-style **manual
atomic occupations** are exposed as an opt-in override on top of all
guess methods.

The work splits along two axes:

| Axis | Item                                                   | Status / target |
|------|--------------------------------------------------------|-----------------|
| Guess | CORE (Hcore)                                           | shipped pre-v0.5 |
| Guess | Level shift on virtual block (Saunders-Hillier 1973)   | partial, molecular only at v0.5; periodic at v0.6.x |
| Guess | SAD (atomic density superposition)                     | v0.6.x flagship |
| Guess | SAP (atomic potential superposition)                   | v0.6.x sibling   |
| Guess | Extended Hückel (parameter-free) / PModel              | v0.7.x          |
| Guess | MINAO (minimal-AO projection of SAD)                   | v0.7.x          |
| Guess | FRAGMO (fragment-MO assembly)                          | v0.8.x          |
| Guess | READ / restart (orbitals from prior calc)              | v0.8.x          |
| Guess | Broken-symmetry (high-spin → GUESSMIX → 45° rotation)  | v0.7.x          |
| Guess | Manual atomic occupation override (CRYSTAL-style)      | v0.6.x          |
| Conv | Linear damping (density-mixing on D)                    | shipped         |
| Conv | CRYSTAL-style FMIXING (static Fock-mixing on F)         | v0.6.x          |
| Conv | Dynamic damping (Zerner-Hehenberger 1979)               | v0.6.x          |
| Conv | Pulay DIIS                                              | shipped         |
| Conv | EDIIS (Kudin / Scuseria / Cancès 2002)                  | ✅ shipped (D1a) |
| Conv | EDIIS+DIIS hybrid (Garza / Scuseria 2012)               | ✅ shipped (D1c) |
| Conv | KDIIS (Kollmar)                                         | ✅ shipped       |
| Conv | ADIIS (Hu / Yang 2010)                                   | ✅ shipped (D1b) |
| Conv | ODA (Cancès & Le Bris 2000)                             | ✅ shipped (D1d) |
| Conv | LIST family (Wang et al. 2011)                          | v0.7.x          |
| Conv | Saunders-Hillier level shift (level shift on F_vv)      | v0.6.x          |
| Conv | SOSCF (Newton-Raphson; Bacskay 1981, Fischer-Almlöf 1992, Neese 2000) | ✅ shipped (D2c) |
| Conv | Augmented Hessian / TRAH (Helmich-Paris 2022)           | ✅ shipped (D2e) |
| Conv | ARH (Host et al. 2008) / CIAH (Sun, PySCF 2017)         | v0.8.x sibling   |
| Conv | OT direct minimisation (VandeVondele & Hutter 2003), periodic | v0.9.x flagship |
| Conv | Anderson mixing (Anderson 1965)                         | v0.10.x        |
| Conv | Broyden mixing (Johnson 1988, Eyert 1996)               | v0.10.x        |
| Conv | Kerker preconditioner (Kerker 1981)                     | v0.10.x        |
| Conv | Pulay-Kerker mixing (Kresse & Furthmüller 1996)          | v0.10.x        |
| Conv | Periodic Pulay (Banerjee et al. 2016)                   | v0.10.x        |
| Conv | Fermi-Dirac / Methfessel-Paxton / Marzari-Vanderbilt smearing | partial, extend at v0.10.x |

### `v0.6.x`, initial-guess overhaul (flagship: **SAD + SAP**)

> 🎯 **The user-visible win.** Closed the v0.5.6 NaCl-bombing-with-
> Hcore-guess class of failure mode: ionic insulators where the
> Hcore guess is so far from physical that DIIS amplifies the
> swing into the +30 000 / −16 000 Ha territory and never
> recovers. SAD / SAP made NaCl / MgO / Na metal converge in 5-15
> iters from the first call.

- **G2a** Atomic density library, precomputed spherical atomic
  RHF / UHF densities for Z = 1 … 86 across STO-3G, 6-31G(d),
  cc-pVDZ, def2-TZVP, pob-TZVP. Stored as `(nbf_atom, nbf_atom)`
  numpy arrays keyed on `(Z, basis_name)`. Prebuilt via the
  molecular RHF/UHF path on isolated atoms.
- **G2b** SAD assembly, sum atomic densities into the
  unit-cell density matrix at AO indices keyed by atom-bf
  ranges. For periodic, replicate the unit-cell SAD into all
  cells in the lattice cutoff (D(R) = D₀ · δ(R) is the Γ-only
  convention; the multi-cell SAD picks up the same atomic
  density at every image).
- **G2c** SAP assembly, sum tabulated radial atomic effective
  potentials (Lehtola 2019 / Lehtola, Visscher, Engel 2020 erfc
  fits) on the AO grid. Skip the Fock build entirely for the
  guess, diagonalize `T + V_SAP` → C₀ → D₀.
  Cheaper than SAD, especially attractive for solids where the
  atomic Fock build per Z is dominated by ERIs.
- **G2d** AUTOSAD fallback, for atoms or basis sets without a
  precomputed atomic density / potential, build it on the fly
  via a single isolated-atom RHF/UHF run. Cached per
  `(Z, basis, charge, multiplicity)` for the rest of the
  process.
- **G2e** Smart defaults per system type:

  | System type                    | Default guess |
  |--------------------------------|---------------|
  | Closed-shell molecule          | SAP           |
  | Open-shell molecule            | SAD-UHF       |
  | Transition-metal complex       | SAD-UHF + Saunders-Hillier shift |
  | Ionic insulator (any periodic) | SAD           |
  | Covalent semiconductor         | SAD           |
  | Metallic crystal               | SAD + Fermi-Dirac smearing |
  | Slab / surface                 | SAD; warn if metallic |
  | Wire                           | SAD; raise on un-implemented Madelung |

  All overridable via `opts.initial_guess = "sap" | "sad" | "core"
  | "huckel" | "minao" | "fragmo" | "read"`.

- **G2f** **CRYSTAL-style manual atomic occupations**. New API:
  ```python
  opts.atomic_occupations = {
      "Cl": {"3s": 2, "3p": [2, 2, 2]},
      "Na": {"3s": 0},
      "Fe": {"3d": [1, 1, 1, 1, 1], "4s": 1},  # high-spin Fe³⁺
  }
  ```
  Overrides the SAD-built atomic density for the named
  elements. Critical for transition-metal SCF stability and
  for testing specific configurations (e.g. broken-symmetry
  start). Mirrors CRYSTAL's `EIGSHIFT` / `ATOMSPIN` / manual
  shell occupation directives, see the CRYSTAL23 manual,
  §SCF Convergence, for the convention.

- **G2g** **Saunders-Hillier level shift** (Int. J. Quantum
  Chem. 7, 699 (1973)). New `opts.level_shift_v` (Hartree)
  added to the virtual-block of F in the MO basis. Provably
  convergent for sufficiently large shift. Wired across
  RHF / UHF / RKS / UKS, mol + periodic. Default 0.0; auto-set
  to 0.2 Ha when system type is "transition metal" or
  "ionic insulator" + heuristic detection of small gap.

- **G2h** **Dynamic damping** (Zerner-Hehenberger, Chem. Phys.
  Lett. 62, 550 (1979)). α adjusted on the fly based on
  energy decrease. Replaces the current static damping for
  difficult cases. Conservative default for molecules; more
  aggressive for ionic / metallic systems.

- **G2h-bis** **CRYSTAL-style FMIXING** (static Fock-matrix
  mixing). New `opts.fock_mixing_pct = N` (0..99, integer):
  ```
  F^(it)  =  (N/100) · F^(it-1)  +  (1 − N/100) · F'^(it)
  ```
  where `F'^(it)` is the freshly-built Fock from the latest
  density. Mirrors CRYSTAL's `FMIXING N` directive (see
  CRYSTAL23 manual, §SCF Convergence). **Distinct from the
  shipped `damping` option**, which mixes the density matrix
  D, not F, both are useful, and on tight ionic crystals
  Fock-mixing often stabilises the residual that
  density-mixing leaves oscillating. Standard CRYSTAL setting
  is 30-80% old Fock; vibe-qc default 0 (off). Wired uniformly
  across RHF / UHF / RKS / UKS, Γ-only + multi-k. Plays
  cooperatively with DIIS, applied before the DIIS extrapolation
  on every iter when both are on, identically to CRYSTAL's
  ordering. The implementation is a one-line addition in each
  SCF driver; the user-facing value is the recognised CRYSTAL
  name + numerical compatibility with CRYSTAL input decks.

- **G2i** SCF divergence-detection on every periodic SCF entry
  point, generalisation of the v0.5.6 hook on
  `run_rks_periodic_gamma_ewald3d` to RHF / UHF / UKS Γ-only
  + multi-k. Same heuristics (NaN / |E| > 1e6 / |dE| > 1e4
  Ha after iter 5) and same remediation hint targeting
  initial-guess + damping + level-shift advice.

- **G2j** Broken-symmetry guesses (open-shell singlets,
  antiferromagnetic coupling). New `opts.guess_mix` that
  rotates α HOMO into α LUMO by a user-specified angle (45°
  default = 50% mixing). Mirrors ORCA's `GUESSMIX`. Wired
  through UHF + UKS only.

References: Lehtola, *J. Chem. Theory Comput.* 15, 1593 (2019);
Lehtola/Visscher/Engel *J. Chem. Phys.* 152, 144105 (2020); Van
Lenthe et al. *J. Comput. Chem.* 27, 926 (2006); Saunders &
Hillier *Int. J. Quantum Chem.* 7, 699 (1973); Zerner &
Hehenberger *Chem. Phys. Lett.* 62, 550 (1979).

### `v0.7.x`, DIIS family expansion (flagship: **EDIIS+DIIS hybrid**)

> 🎯 **Robust by default on transition-metal complexes**. Pulay
> DIIS plateaus or oscillates near stationary points that aren't
> minima. EDIIS minimises the OD energy functional rather than
> the commutator, dominates DIIS far from convergence. The
> standard production pattern is EDIIS-early, switch-to-DIIS-near.

- ✅ **D1a** EDIIS (Kudin, Scuseria, Cancès, *J. Chem. Phys.* 116,
  8255 (2002)), quadratic energy functional from the Optimal
  Damping Algorithm. Ships ``vibeqc::EDIIS`` (closed-shell + UHF
  open-shell extrapolate; exact active-face simplex QP solver), wired
  into all four ``run_*_scf_with_jk`` molecular drivers (RHF,
  UHF, RKS, UKS), periodic-Γ callers driving via the JKBuilder
  interface inherit it for free. ``RHFOptions::scf_accelerator =
  SCFAccelerator::EDIIS`` activates it.
- ✅ **D1b** ADIIS (Hu & Yang, *J. Chem. Phys.* 132, 054109 (2010))
 - augmented Roothaan-Hall energy functional. Sibling to
  EDIIS; a later analysis (Garza & Scuseria, *J. Chem. Phys.*
  137, 054110 (2012)) showed they're identical at the HF level.
- ✅ **D1c** **EDIIS+DIIS** hybrid, the production default.
  EDIIS until error is below threshold, then switch to plain
  DIIS for the asymptotic regime. Garza & Scuseria's analysis
  identifies this as the method of choice.
  ``RHFOptions::scf_accelerator = SCFAccelerator::EDIIS_DIIS``
  with ``ediis_diis_switch_threshold`` (default 1e-1, PySCF
  convention).
- ✅ **D1d** Optimal Damping Algorithm (Cancès & Le Bris, *Int. J.
  Quantum Chem.* 79, 82 (2000)) as a standalone option,
  primarily for diagnostic comparison and educational use.
- **D1e** LIST family (Wang et al., *J. Chem. Theory Comput.* 7,
  3045 (2011)), for systems where DIIS stagnates on its own.
- ✅ **D1f** Extended Hückel / PModel guess (Hoffmann 1963 + the
  parameter-free variant). Mostly equivalent to SAP at first
  order but useful as a back-up for systems where SAP/SAD
  numerics struggle (deep core states with very high Z).
- ✅ **D1g** MINAO guess: project SAD onto a minimal-basis
  reference, then re-expand. Cheap; very stable for molecules.

### `v0.8.x`, second-order convergence (flagship: **SOSCF / TRAH**)

> 🎯 **Quadratic convergence for the hard cases.** DIIS family
> methods stall on transition-metal complexes, broken-symmetry
> solutions, and small-gap systems. Once the gradient is below
> a "trust threshold", switch to a Newton-step on the orbital
> rotation manifold. Quadratic convergence in ~3-5 final iters.

- **D2a** Orbital-rotation parametrisation: C(t) = C₀ · exp(A)
  with A antihermitian, so the constraint C† S C = I is
  trivially preserved. Standard convention.
- **D2b** Pre-SOSCF DIIS warm-up, gradient must be below
  ~1.0 (Fischer-Almlöf 1992 convention) before SOSCF activates.
  Otherwise the Newton step overshoots.
- ✅ **D2c** SOSCF strict (Bacskay 1981, Fischer & Almlöf, *J. Phys.
  Chem.* 96, 9768 (1992)), full orbital Hessian, exact
  Newton step.
- **D2d** Approximate SOSCF (Neese, *Chem. Phys. Lett.* 325, 93
  (2000)), diagonal-dominant orbital-Hessian approximation.
  ORCA's standard. Cheaper than strict, almost as fast in
  iters.
- ✅ **D2e** Augmented Hessian (AH) / Trust-Region (TRAH;
  Helmich-Paris, *J. Chem. Phys.* 156, 204104 (2022)), step
  restriction prevents Newton overshoot. Critical for
  CASSCF / state-averaged variants.
- **D2f** ARH, Augmented Roothaan-Hall quasi-Newton (Host et
  al., *J. Chem. Phys.* 129, 124106 (2008)). Iterative Hessian
  build avoids the explicit-Hessian bottleneck.
- **D2g** CIAH, Co-Iterative Augmented Hessian (Sun, *J. Chem.
  Theory Comput.* 13, 5053 (2017)), PySCF's general
  second-order solver, decorator-style API.
- **D2h** FRAGMO guess, converge SCF on isolated fragments,
  superimpose. Excellent for non-covalent complexes; foundation
  of ALMO / EDA. Pairs naturally with second-order convergence
  for the fragmented start.
- ✅ **D2i** READ / restart from prior orbitals, for opt /
  scan / extrapolation workflows. Re-orthogonalize across
  basis-set or geometry changes. Project from smaller to
  larger basis when stepping up. Supersedes any guess method.
  Shipped (molecular): `InitialGuess.READ` from an in-memory
  result, a `.qvf`, or a `.molden`, with cross-basis/geometry
  density projection. Shipped (periodic, Γ-point): the prior
  g=0 cell density (projected onto the current cell basis) is
  injected at g=0 and Bloch-sums to `D(k)`, the slab/bulk
  NEB / geometry-scan restart path, on the Ewald / GDF / BIPOLE
  Γ drivers. Shipped (periodic, closed-shell multi-k): native GDF / GPW /
  GAPW results and current `.qvf` archives restart from their per-k complex
  Bloch coefficients and occupations by rebuilding `D(k)` for the first Fock
  build. Still pending: `.molden` multi-k READ, legacy QVF archives without
  `x_vibeqc.bloch_wavefunction`, and open-shell multi-k READ (per-spin per-k
  density blocks).
- ✅ **ATOMSPIN / SPINLOCK (periodic)** broken-symmetry magnetic
  start + convergence for periodic UHF/UKS. `atomic_spins` seeds
  an AFM/ferrimagnetic g=0 pattern (Bloch-sums to broken-symmetry
  `D(k)`) on the Γ UHF/UKS Ewald, GDF, BIPOLE drivers + multi-k
  UKS Ewald; `SPINLOCK` PATTERN_HOLD (per-k MOM hold) protects it
  on multi-k UKS + Γ UHF/UKS, and SPIN_SCHEDULE (two-phase) runs
  on Γ UHF/UKS Ewald, Γ GDF, and Γ BIPOLE. Still pending: ATOMSPIN
  on the multi-k UHF Ewald path (no broken-symmetry guess hook),
  SPIN_SCHEDULE on the multi-k Ewald/BIPOLE routes, and `BETALOCK`.

### `v0.8.x`, multi-secant / quasi-Newton SCF acceleration (Anderson + Broyden)

> 🎯 **Density-mixing acceleration beyond DIIS.** Anderson
> acceleration and Broyden quasi-Newton mixing are derivative-
> free, multi-secant techniques that excel on the cases where
> DIIS oscillates: small-gap or near-metallic periodic systems,
> charge-sloshing regimes, and any SCF where the residual
> spectrum has multiple comparable modes. Standard in periodic
> DFT codes (SIESTA, CP2K, ADF/BAND, VASP) and increasingly in
> molecular codes (ORCA's DIIS_GDM hybrid, Q-Chem's GDM/RCA_DIIS).

Background: Anderson (Donald G. Anderson, J. ACM 12, 547 (1965))
minimises a least-squares linear model over the m most recent
residuals fₖ = g(xₖ) − xₖ, effectively a quasi-Newton update
without forming an explicit Jacobian. Broyden (J. Math. Phys. 12,
347 (1965); "good Broyden" Type-I updates the Jacobian, "bad
Broyden" Type-II updates the inverse) builds a secant-condition-
based Jacobian approximation. **Type-II Broyden is mathematically
equivalent to Anderson** at the multi-secant level; they
diverge in implementation detail (history weighting, restart
criteria, line-search safeguards).

- **D3a** Anderson acceleration on the density-matrix iteration,
  a density-space alternative to the Fock-space DIIS family.
  Shipped in v0.14.0 as ``AndersonMixer`` in
  ``python/vibeqc/periodic_density_mixing.py``. History length
  m=5-10; Tikhonov-regularised least-squares to handle near-
  linear-dependent residual columns. Anderson-restart on
  residual norm increase (Walker-Ni safeguard).
- **D3b** Broyden Type-II mixing, same multi-secant least-
  squares but updates the inverse Jacobian. Equivalent in
  theory to Anderson but with different numerical stability;
  worth shipping both for benchmarking and as fallbacks.
  Shipped in v0.14.0 as ``BroydenMixer``, same module.
- **D3c** Hybrid policy: warm-start with FMIXING/damping +
  Anderson, transition to DIIS once the gradient is "tame"
  (||[F,DS]|| < some threshold). Mirrors the CRYSTAL recipe
  pattern (FMIXING-then-DIIS) but with a smarter accelerator
  during the rough phase.
- **D3d** Periodic-friendly variant: Kerker preconditioning
  (damp the small-G charge modes that drive charge-sloshing
  oscillation on metallic systems). Mainly relevant for
  conducting / small-gap targets; pairs naturally with
  Fermi-Dirac smearing.
- **D3e** Multi-secant on per-k density list, when running
  multi-k SCF, the residual vector is the concatenation of
  per-k Fock errors; Anderson / Broyden act on the full
  concatenated vector, weighted by k-mesh.

Anderson and Broyden mixers shipped in v0.14.0 as the second-line
solver for cases where DIIS struggles but SOSCF/TRAH is overkill
(no second-order Hessian needed). Users have the
``scf_accelerator = "diis" | "anderson" | "broyden" | "soscf"``
selector with sensible AUTO defaults per system type.

### `v0.8.0`, SCF diagnostics: "possibly conducting state" warning

> 🎯 **Surface the SCF's view of the band gap.** CRYSTAL emits
> ``POSSIBLY CONDUCTING STATE - EFERMI(AU) <ε_F> (RES. CHARGE
> <q>; IT. <n>)`` when HOMO and LUMO collapse and fractional
> occupations are needed. vibe-qc should do the same, silent
> SCF convergence on a "no gap" Aufbau-filled state is the
> wrong answer dressed up as right.

- Compute the band gap from converged ``mo_energies`` (per k
  for multi-k; global Fermi level via the existing
  ``occupations.py`` helpers).
- If gap < ``conducting_state_warning_threshold`` (default
  1e-4 Ha ≈ 3 meV, narrow enough that it's an actual
  metal, not a small-gap insulator), emit a warning to the
  SCF log + add a ``warnings`` field on the result object.
- Document the warning as a hint to enable Fermi-Dirac
  smearing on metallic / near-degenerate cases.

### `v0.9.x`, direct minimisation for big solids (flagship: **OT**)

> 🎯 **Skip diagonalisation entirely.** Unconstrained
> minimisation on the manifold of orthonormal orbitals via
> exponential parametrisation. Linear-scaling alternative to
> SCF; the workhorse for 1000+ atom periodic CP2K runs.

- **D6a** OT framework (VandeVondele & Hutter, *J. Chem. Phys.*
  118, 4365 (2003)). Cubic in nbf for the basis transform but
  no diagonalisation in the inner loop. Critical for big
  cells.
- **D6b** OT preconditioners, FULL_ALL, FULL_SINGLE_INVERSE,
  FULL_KINETIC, with `ENERGY_GAP` as the primary tuning knob.
  Quality of the preconditioner dominates iteration count.
- **D6c** OT minimisers, CG (default), DIIS-OT (faster but
  less reliable), Broyden-OT (Weber et al., *J. Chem. Phys.*
  128, 084113 (2008)). CG with Wolfe line search.
- **D6d** ADMM, auxiliary density matrix method
  (VandeVondele / Krack / Mohamed) for hybrid DFT compatibility
  with OT. The HSE06-on-1000-atoms enabling technology.
  Already on the acceleration roadmap as a v0.9.x sibling;
  **re-versioned to v1.x / research direction (2026-06 audit)**: citation-only,
  dormant route, no compute path.

### `v0.10.x`, density mixing for periodic metals (flagship: **Anderson + Kerker + Pulay-Kerker**)

> 🎯 **Make periodic metals tractable.** OT is contraindicated
> for metals (no smearing-aware variant). Density mixing on the
> charge-density grid is the standard plane-wave path. Without
> Kerker preconditioning, charge sloshing at small G destroys
> SCF convergence in any metallic cell with > 50 atoms.

- **D4a** Anderson mixing (Anderson, *J. ACM* 12, 547 (1965))
  on the real-space density. Closely related to DIIS in
  spirit.
- **D4b** Kerker preconditioner (Kerker, *Phys. Rev. B* 23,
  3082 (1981)). Damps small-G modes that cause charge
  sloshing in metallic cells. Indispensable for metals.
- **D4c** Modified Broyden II / Johnson-Eyert (Johnson, *Phys.
  Rev. B* 38, 12807 (1988); Eyert, *J. Comput. Phys.* 124,
  271 (1996)). Quasi-Newton dielectric Jacobian. Default in
  many plane-wave codes.
- **D4d** Pulay-Kerker (Kresse & Furthmüller, *Phys. Rev. B*
  54, 11169 (1996)). Pulay extrapolation + Kerker
  preconditioning. VASP's default for metals.
- **D4e** Periodic Pulay (Banerjee, Suryanarayana, Pask, *Chem.
  Phys. Lett.* 647, 31 (2016)), the recent plane-wave variant
  with provably better robustness for diverse materials.
- **D4f** Fermi-Dirac / Methfessel-Paxton / Marzari-Vanderbilt
  smearing, already partially in place; extended here to
  smearing-aware DIIS / Pulay so the SCF inner loop respects
  the smeared occupations consistently.

**This program targets self-consistent SMEARING metals (large-cell charge
sloshing).** It does **not** enable self-consistent GR-*driven* metals:
empirically (2026-06-16, Be fcc RKS/PBE 4×4×4) a GR-driven metal SCF does not
converge with *any* density mixer, linear damping, Fock-DIIS, and Anderson
(D4a) all oscillate (cold, heavy-damped, and warm-started from a smeared
density), because sharp T=0 occupations make the SCF map **discontinuous**,
which no mixer can converge. Metals are driven with SMEARING (a smooth map) and
GR is applied POST-SCF (the standard tetrahedron/Gilat usage). `AndersonMixer`
(D4a, commit `111f96da`) is landed as the numerical core, but did not converge
even a small *smeared* metal in a quick wiring, D4 needs Kerker preconditioning
(D4b) + careful representation, i.e. genuine research, not a quick drop-in.

### `v0.13.x`, metallic BZ integration quality (Gilat net), *partially shipped*

> 🎯 **Resolve the Fermi surface, parameter-free.** The Gilat-Raubenheimer cell
> quadrature (Gilat & Raubenheimer, *Phys. Rev.* 144, 390 (1966); the CRYSTAL
> `SHRINK IS ISP` second-net analogue) integrates the BZ occupation analytically
> per microcell, no smearing width to converge. Shared k-layer infrastructure
> (`vibeqc.bz_integration`), engine-agnostic across BIPOLE/GDF/GPW.

**Shipped (updated 2026-08-02):** the GR occupation kernel +
per-k-to-full-grid scatter, gated `bz_integration="gilat"` hooks in the RHF,
RKS, UHF, and UKS multi-k Ewald drivers, public BIPOLE RKS/UHF/UKS dispatch,
the GR density of states (`gilat_dos`), and IBZ-reduced-mesh support
(`expand_ibz_eigenvalues` + `gilat_occupations_for_kmesh` auto-dispatch).
Self-consistent GR works for **insulators** (RHF + RKS + UHF + UKS, full *or*
IBZ meshes); **post-SCF** GR gives the parameter-free Fermi level + occupations
+ DOS on a smearing-converged density. Commits `8ea8332d` `203b81ba`
`d66aa7f5` `14ac08df` `47f92b24` `a314cf67` `f5ce3e64`; full status in
`handovers/HANDOVER_GILAT_NET.md`.

**Remaining:**

- **Metal total-energy parity vs CRYSTAL `SHRINK IS ISP`**, via SMEARING-SCF +
  post-SCF GR, **not** a GR-driven SCF (which is fundamentally non-convergent:
  sharp T=0 occupations → discontinuous map, unconvergeable by any density mixer,
  confirmed 2026-06-16). Metal total energies come from smearing (T→0 is well
  defined); GR gives the parameter-free Fermi level / occupations / DOS post-SCF.
  Remaining: a CRYSTAL parity run (Al/Cu fcc, out-of-process).
- **Additional public backends.** BIPOLE UHF/UKS now share the existing
  per-spin kernel. GDF, RIJCOSX, GPW/GAPW, and the restricted-open-shell
  public routes remain separately gated and should reuse the same occupation
  layer when their route contracts are ready.

### Post-dimensional-method SCF parity agenda

Once the general 1D / 2D / 3D method stack is in place for HF, MP2,
DFT, hybrids, and their open-shell variants, the SCF-control surface
should be consolidated against CRYSTAL and the wider code ecosystem.
The goal is not a pile of ad-hoc knobs; it is a small set of
composable, method-independent strategies that work across molecular
and periodic drivers.

| Method | Strengths | Weaknesses | Best for | CRYSTAL notes |
|--------|-----------|------------|----------|---------------|
| FMIXING | Simple Fock/KS damping; stabilizes early iterations | Slow alone | Initial cycles, metals | `FMIXING <value>`; CRYSTAL default behavior |
| LEVSHIFT | Stabilizes small HOMO-LUMO gaps and open shells | Can slow final precision | Semiconductors, open-shell systems | `LEVSHIFT <n> <e>`; combinable with FMIXING |
| DIIS | Fast asymptotic convergence | Orbital swapping, early failure on poor guesses | Good guesses, non-metals | `DIIS`; CRYSTAL disables in some massive-parallel defaults |
| FMIXING + LEVSHIFT | Robust for stubborn cases | Slower than pure DIIS | Difficult periodic SCF | Common CRYSTAL tutorial recipe |
| FMIXING + DIIS | Good speed/stability compromise | DIIS can still oscillate | Default hybrid strategy | FMIXING warm-up before DIIS |

Observed operating principle: DIIS is usually 2-3x faster than linear
mixing on easy cases, but fails more often on hard systems unless
damping, level shifting, smearing, or a hybrid strategy prepares the
SCF state first. CRYSTAL's periodic stability bias is FMIXING +
LEVSHIFT first, DIIS when the state is sane; vibe-qc should expose the
same pattern as reusable policy rather than per-driver special cases.

CRYSTAL-specific follow-up items:

* Retain FMIXING / LEVSHIFT / DIIS as the CRYSTAL14-compatible core.
* Add post-CRYSTAL14 metallic smearing semantics where they affect
  parity inputs; keep CRYSTAL14 references separate from CRYSTAL23.
* Provide pure-mixing fallback profiles for DIIS oscillations.
* Investigate `DIISALLK`-style full-k-space DIIS for hybrids, guarded
  by memory diagnostics because the full k-space history is expensive.

Cross-code convergence controls to mirror or learn from:

| Code | Key methods | Notes |
|------|-------------|-------|
| Q-Chem | DIIS, GDM, RCA, RCA_DIIS, DIIS_GDM, MOM | RCA is a reliable downhill fallback; MOM guards orbital swaps. |
| ORCA | DIIS, damping, level shift, SOSCF-style fallbacks | Strong open-shell stability bias. |
| ADF/BAND | DIIS, MultiSecant, density mixing | BAND defaults to conservative mixing and smearing for metals. |
| PySCF | DIIS, SOSCF / CIAH, Newton variants | External reference only; useful design comparison. |
| Gaussian | DIIS, damping, SCF=QC/XQC | QC/XQC is the canonical quadratic fallback pattern. |

For metals, occupation treatment and charge-density response must come
before plain Fock extrapolation:

| Approach | Description | Examples / target |
|----------|-------------|-------------------|
| Smearing | Fractional occupations via Fermi-Dirac and later Gaussian / Methfessel-Paxton / Marzari-Vanderbilt variants | First-line setting for any metallic periodic calculation |
| Broyden mixing | Quasi-Newton density mixer | Robust against charge sloshing; SIESTA / CP2K style |
| MultiSecant | DIIS-like multi-secant density update | ADF/BAND-style post-DIIS fallback |
| Kerker / preconditioned mixing | Damp small-G charge modes | Plane-wave metal standard; CP2K / VASP style |
| LEVSHIFT + FMIXING | Virtual shift plus Fock/KS damping | CRYSTAL / ORCA-compatible stabilizer for difficult starts |

Practical metallic workflow target: start with smearing (typical
electronic broadening 0.01-0.1 Ry), use FMIXING in the 0.5-0.8 range
for difficult charge-sloshing starts, combine with LEVSHIFT when
occupied/virtual ordering is unstable, and delay pure DIIS until the
charge response is tame. This should become a named policy profile
usable from molecular, 1D, 2D, and 3D drivers rather than a periodic
one-off.

### Recommended workflows by system

| System type | Initial guess | Convergence accelerator |
|-------------|---------------|-------------------------|
| Routine closed-shell mol | SAP | DIIS |
| Difficult mol (anything else) | SAD | EDIIS+DIIS + level shift |
| Transition-metal complex | SAD-UHF + manual occupations | EDIIS+DIIS → SOSCF (warm-up below grad ~1) |
| Open-shell singlet / BS | SAD-UHF + GUESSMIX | EDIIS+DIIS → SOSCF + AH |
| Periodic insulator (small-medium cell) | SAD | DIIS |
| Periodic insulator (large cell) | SAD | OT-CG with FULL_ALL |
| Periodic metal | SAD + smearing | Pulay-Kerker (small) / Periodic Pulay (large) |
| Periodic hybrid DFT | First converge semilocal, then read for hybrid | DIIS (small) / OT + ADMM (large) |
| Slab (insulating) | SAD | DIIS |
| Slab (metallic) | SAD + smearing | Pulay-Kerker |
| Wire | SAD + 1D Madelung | DIIS |

### Key reference list (initial guess + convergence)

This table is the single source of truth for citations across the
program; each section above references this list.

| Method | Reference |
|--------|-----------|
| Hcore guess | tradition (Roothaan 1951) |
| GWH | Wolfsberg & Helmholz, *J. Chem. Phys.* 20, 837 (1952) |
| Extended Hückel | Hoffmann, *J. Chem. Phys.* 39, 1397 (1963) |
| Anderson mixing | Anderson, *J. ACM* 12, 547 (1965) |
| Saunders-Hillier level shift | *Int. J. Quantum Chem.* 7, 699 (1973) |
| Zerner-Hehenberger damping | *Chem. Phys. Lett.* 62, 550 (1979) |
| Pulay DIIS | *Chem. Phys. Lett.* 73, 393 (1980); *J. Comput. Chem.* 3, 556 (1982) |
| Bacskay quadratic SCF | *Chem. Phys.* 61, 385 (1981) |
| Kerker preconditioner | *Phys. Rev. B* 23, 3082 (1981) |
| Johnson modified Broyden II | *Phys. Rev. B* 38, 12807 (1988) |
| Fischer-Almlöf SOSCF | *J. Phys. Chem.* 96, 9768 (1992) |
| Eyert Broyden mixing | *J. Comput. Phys.* 124, 271 (1996) |
| Kresse-Furthmüller VASP mixing | *Phys. Rev. B* 54, 11169 (1996) |
| Kudin/Scuseria/Cancès EDIIS / Cancès-Le Bris ODA | *J. Chem. Phys.* 116, 8255 (2002); *Int. J. Quantum Chem.* 79, 82 (2000) |
| Neese SOSCF (ORCA) | *Chem. Phys. Lett.* 325, 93 (2000) |
| VandeVondele-Hutter OT | *J. Chem. Phys.* 118, 4365 (2003) |
| Van Lenthe et al. SAD | *J. Comput. Chem.* 27, 926 (2006) |
| Host et al. ARH | *J. Chem. Phys.* 129, 124106 (2008) |
| Weber et al. OT refinements | *J. Chem. Phys.* 128, 084113 (2008) |
| Hu & Yang ADIIS | *J. Chem. Phys.* 132, 054109 (2010) |
| Wang et al. LIST | *J. Chem. Theory Comput.* 7, 3045 (2011) |
| Garza & Scuseria EDIIS+DIIS analysis | *J. Chem. Phys.* 137, 054110 (2012) |
| Sun PySCF CIAH | *J. Chem. Theory Comput.* 13, 5053 (2017) |
| Banerjee-Suryanarayana-Pask Periodic Pulay | *Chem. Phys. Lett.* 647, 31 (2016) |
| Lehtola SAP guess | *J. Chem. Theory Comput.* 15, 1593 (2019) |
| Lehtola-Visscher-Engel SAP Gaussian implementation | *J. Chem. Phys.* 152, 144105 (2020) |
| Helmich-Paris TRAH | *J. Chem. Phys.* 156, 204104 (2022) |

CRYSTAL23 manual (CRYSTAL solid-state code) is the reference for
manual atomic occupation conventions, EIGSHIFT, ATOMSPIN, broken-
symmetry workflows for solids, and the canonical wisdom on the
shell-by-shell guess override that vibe-qc's `opts.atomic_occupations`
is modeled on.

## Recent releases

The two most recent minors stay here because their scope still explains what
the next one builds on. Everything older is in
[roadmap history](roadmap_history.md).

### `v0.16.0`, shipped 2026-09-08, the repository split

> **Pople's Puffin.** John Pople (1925-2004), Nobel 1998: ab initio methods
> made usable by anyone, which is what splitting these components into
> separately adoptable projects is for. Cut at 9c34674b, proven by the
> release-gate pipeline on `release-candidate/v0.16.0`.

🎯 A **structural** release. It consumed a version number without consuming a
milestone from the ladder below -- which is why the native BvK milestone that
previously held this number moved to v0.17.0, and the periodic-CC, MR,
properties and MPI milestones each moved up one with it. (v0.17.0 then
consumed a number the same way, so that ladder shifted once more; native BvK
now leads the open milestones. See the v0.17.0 section below.)

**The split.** the archived monorepo became four fresh-history repositories, each
adoptable on its own:

- ✅ **vibe-qc** -- `python/ cpp/ tests/ docs/ examples/ scripts/`, with
  **vibe-basis** deliberately retained as a subpackage.
- ✅ **vibe-view** -- the viewer, an optional companion.
- ✅ **vibe-queue** -- the job queue; CLI and import package stay `vq`.
- ✅ **qvf** -- the Quantum Visualization Format: spec, schema, worked
  examples, conformance corpus and an Apache-2.0 reference implementation.
  **Nothing depends on it**, at runtime or install time. vibe-qc implements
  the format and validates against the published contract; the viewer reads
  it independently. Two implementations, one spec.

The couplings that only worked inside one checkout are gone: vibe-qc reads
its own QVF (`vibeqc_naming._qvf_read`, sha256-verified per member, validated
against the 12-archive corpus) instead of importing the viewer through a
fallback that had always been dead; the viewer's schema files are real files
rather than symlinks into `python/vibeqc/output/formats/`; and
`vibe-view[queue]` is declared instead of resolving by shared venv. A
`qvf-conformance` job in both vibe-qc and vibe-view is what now holds the
three schema copies together, since no dependency edge does.

**Also in v0.16.0:**

- ✅ **Semiempirical periodic electrostatics corrected** (GFN2-xTB and
  GFN2-SECCM are gated experimental) -- the GFN2-SECCM
  shell gamma and the periodic Γ GFN2-xTB electrostatics are real lattice
  sums over the Wigner-Seitz image records on the shared periodic kernel,
  replacing three hand-rolled Ewald implementations; the periodic SCC-DFTB
  gamma lattice sum converges.
- ✅ **Unified initial-guess selection and construction**, closing a
  fourteen-issue family.
- ✅ **Analytic COSMO FINE cavity nuclear derivatives** and the assembled
  analytic **CFC energy gradient**, with Becke-partitioned segment joins and
  Gaussian segment charges.
- ✅ **Periodic fixes across the stack** -- physical grid-weight
  neighborhoods, custom basis contractions retained through XC gradients,
  multi-k GPW/GAPW live SCF progress restored to the `.system` manifest,
  multi-k MOM occupied-subspace retention, and full-k DFTB optimization at
  finite temperature minimising the Mermin free energy.
- ✅ **CAS / MR-PT and OVGF accept an ECP reference.**

### `v0.17.0`, shipped 2026-09-09, symmetry and periodic foundations

> **Tew's Tern.** David P. Tew, periodic local correlation (the
> Nejad-Zhu-Tew 2025 Paper I lineage). Tern: navigates by the lattice it
> crosses. Cut at 6421ed34, proven by the release-gate pipeline on
> `release-candidate/v0.17.0` (5094).

🎯 The second release in a row to consume a version number without consuming
the milestone that held it. The native BvK route this codename honours is
*not* in this tag; what is here is the symmetry and periodic groundwork that
route has to stand on, plus two unrelated deliverables that were ready. The
BvK milestone stayed next in line, and the ladder below it shifted once more.

The roadmap anticipated exactly this: "if native BvK DLPNO-MP2 is not ready
after the paper freeze, choose a different minor headline rather than
re-selling the megacell route that already shipped in v0.15.x."

**Symmetry, shared group and space infrastructure.**

- ✅ **Shared group and space infrastructure restored**, the common
  substrate the periodic symmetry paths had been duplicating.
- ✅ **Retained group composition and Bloch factors audited**, and
  **retained subspaces audited across panel backends**, so the reduction a
  backend claims is the reduction it performs.
- ✅ **Periodic subspace probes bound to immutable states**, removing the
  mutable-state route by which a probe could disagree with the run it
  described.

**Periodic: BIPOLE supports and long-range contractions.**

- ✅ **Finite BIPOLE Seitz supports audited**, and **BIPOLE support audits
  bound to validated AO maps**.
- ✅ **Exact physical quartet supports for BIPOLE SR** (private diagnostic) -- a bounded
  geometric builder enumerating SR quartet images from two AO-product
  distance limits and a geometric midpoint limit, on exact rational
  interpretations of the supplied binary64 data. Pair-dependent supports
  pass ERI-permutation and screw-reanchoring tests, including native multi-k
  SR integral comparisons.
- ✅ **Independent physical product supports for the native BIPOLE LR
  Gram** kernel: separate left/right AO-product cell lists with complete
  admission and provenance for both.
- ✅ **Exact-mesh HF long-range contractions bridged** (private), and **zero-mode
  overlap checks bound to the finite source.**

> ⚠️ These are *supports and audits*, not a production route. Production HF
> and complete source integration remain open, and the quartet builder does
> not authorize symmetry reduction. Do not read this section as periodic
> local correlation having shipped.

**Solvation: the COSMO FINE cavity gets a molecule-fixed frame.**

- ✅ **The CFC is built in its molecule-fixed frame by default**, and the
  **analytic nuclear derivative runs through that frame**, completing the
  frame-consistent gradient begun with the v0.16.0 analytic CFC work.
- ✅ **The CFC assigns basis points to atoms smoothly**, closing a
  discontinuity in the step-5 partition.

**Basis sets.**

- ✅ **The 16 cc-pVnZ-PP orbital sets** shipped.
- ✅ **vibe-basis 0.11.0**, the cohesive objective (M2 code).

**Also in v0.17.0:** experimental `correlation=` on the AICCM front door (four-centre
arm) and Gamma-CCM correlation citation stamps and routes; per-product
marketing pages for vibe-qc, vibe-view and vibe-queue, with the website
deploy no longer overwriting companion documentation subtrees.

**Patch releases.** v0.17.1 (2026-09-10) added on-demand Basis Set Exchange
basis fetching, vibe-basis 0.12.0 (the topology search), periodic and ECP
initial-guess coverage (valence ECP references for SAP and MINAO, FRAGMO
with explicit periodic atom images, native periodic READ and full-HF PATOM
seeds), open-shell all-k QVF restart, the COSMO FINE Cavity's C1 outward
projection, and experimental AICCM correlation on the real-Gamma arm with
CCSD and DLPNO. v0.17.2 (2026-09-13) added Direct COSMO-RS, completed
pob-TZVP (Rb-I) and pob-DZVP-rev2, and fixed the pob-TZVP-rev2
effective-core potentials. It ships the periodic GDF cost regressions #206
and #196 as known issues. v0.17.3 expands optional TREXIO exchange while
keeping QVF as the default, fixes periodic GFN2 budget handling and CCM
tolerance validation, and consumes published vibe-view v2.17.0. The GDF
issues remain open; experimental MSINDO CCM stability has partial evidence,
without accepted public force or Hessian coverage. All patches are itemized
in [the changelog](changelog.md).

## Open milestones

Themes, in their intended order. A version number is attached only when a
release is cut (maintainer decision, 2026-09-13), so none of these carries
one: `v0.16.0` and `v0.17.0` each took the number of a planned milestone they
did not deliver, and renumbering the ladder twice is what this avoids.

### Next minor: native BvK periodic local correlation

> **Codename:** not yet named, and no artwork is prepared.

**Roadmap adjustment after v0.15.x.** The old headline, "periodic
correlation", is now too broad: the Paper II megacell/toroidal route shipped
across v0.13-v0.15. `vibeqc.periodic_megacell_mp2` and the AICCM/CCM
experimental lines already cover periodic MP2 -> DLPNO-CCSD(T) routes with
TDL-style extrapolation and PySCF.pbc KMP2 parity checks. Those routes remain
important, but they are no longer a clean headline for this milestone.

The remaining deliverable is the **native BvK /
translational-symmetry route** from the CRYSCOR/Turbomole periodic local
correlation lineage: unique-pairs-only, k-aware, no explicit molecular
megacell expansion, consuming the Stage 1-4 periodic-DF foundation.

- **Native BvK RI-MP2**, correlated solid-state energies in the periodic
  representation, benchmarkable against PySCF.pbc KMP2 and CRYSCOR-style
  limits. Molecular-first gate (v0.14.0 rule) already met: molecular RI-MP2
  shipped in v0.7.2.
- **Native BvK DLPNO-MP2**, the scalable local-correlation variant from the
  Nejad, Zhu, Tew et al. 2025 Paper I lineage. Molecular DLPNO-MP2 is shipped;
  the missing work is the periodic PAO/PNO domain machinery, pair screening,
  and parity surface in the translational representation.
- **Decision point for the next minor's headline:** if native BvK DLPNO-MP2 is
  not ready after the paper freeze, choose a different minor headline rather
  than re-selling the megacell route that already shipped in v0.15.x.
- **FCIQMC in a periodic mean field** (Christlmaier et al. 2022) remains a
  candidate follow-on, not the release gate.

**Historical implementation notes:** the former
`handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md` workstream handover is no
longer tracked in this repository. For the currently supported routes and
limits, see the [DLPNO-MP2 guide](user_guide/dlpno_mp2.md).

### Planned: periodic coupled cluster

> **Codename candidate:** Berkelbach's Badger (provisional artwork proposal).

- **Periodic canonical CCSD(T)** for small unit cells (McClain, Sun, Chan &
  Berkelbach 2017; Gruber et al. 2018 plane-wave TDL parity target).
  Molecular-first gate (v0.14.0): molecular CCSD(T) ✅.
- **Periodic LNO-/DLPNO-CCSD(T)** as the scalable variant (Ye & Berkelbach 2024).
  Gated on molecular DLPNO-CCSD (✅) and periodic DLPNO-MP2 (the native BvK milestone).
  Stages 7-8 of the periodic-local-correlation guide.

### Planned: multireference + excited-state extensions

> **Codename candidate:** Runge's Raccoon, for the Runge-Gross TDDFT theorem.

> Molecular MR foundation (25a-f plus IC-CASPT2) already shipped across
> v0.12-v0.15. What remains here:

**TDDFT (T1):**
- UV/Vis spectrum reconstruction (Lorentzian/Gaussian broadening,
  oscillator strengths) -- the compute engine (Casida + TDA) shipped in
  v0.12.0; this is the post-processing + tutorial surface.
- Transition densities + NTOs (NTO writer path ready in QVF v1.2).
- ECD (Electronic Circular Dichroism) on top of TDDFT.

**Larger active spaces + MR extensions:**
- Selected-CI past 2M determinants (C++ kernel ✅, scaling work remains).
- ICE-CI (iterative configuration expansion).
- DMRG interface (BLOCK / CheMPS2).
- MR-EOM-CC, MCRPA.

**Excited-state extras:**
- MECP (minimum-energy crossing point) optimization.
- Conical intersection optimization (CI-search).
- ΔSCF for excited states.

**Local/DLPNO-MR bridge (molecular target):**
- DLPNO-NEVPT2 (Guo et al. 2016).
- PNO-CASPT2 / PNO-MS-CASPT2.
- Local NEVPT2 via LVMOs.

### Planned: properties suite

> **Codename candidate:** Stone's Sloth, for A. J. Stone and molecular
> response properties.

The user-facing observables milestone. Several items already shipped ahead
of this slot (Mulliken/Löwdin/Mayer in v0.3.0; dipole in v0.3.0; molecular
Hessian/IR in v0.5.0; QTAIM in v0.15.0; COOP/COHP in v0.15.0; Mayer bond
orders in v0.15.0). This milestone gathers the remainder.

**Charge analysis (extended):**
- **PR1** Iterative-Hirshfeld and periodic Hirshfeld charges. Classical
  molecular Hirshfeld charges already ship (`vq.hirshfeld_charges`, also in
  the `run_job` population output).
- **PR3** ESP-fit charges (CHELPG, Merz-Kollman, RESP).
- **PR4** Production NBO and NPA. Provisional NBO helpers (`vibeqc.nbo`)
  already ship as experimental; NPA is deliberately unavailable and raises
  `NotImplementedError`.

**Multipole moments + ESP:**
- **PR5** quadrupole + octupole moments.
- **PR6** ESP cube / isosurface export.

**Response substrate:**
- **PR7** CPHF / CPKS solver.

**Static response (molecular):**
- **PR8-PR9** dipole polarisability α, hyperpolarisability β, γ.
- **PR10-PR12** NMR shielding, spin-spin coupling, EPR g-tensor/hyperfine.

**Vibrational extensions:**
- **PR13-PR15** Raman activities, VCD, ROA.

**Periodic response:**
- **PR16-PR22** Born effective charges, dielectric tensor, elastic/piezo/
  photoelastic tensors, periodic IR/Raman intensities.

**Topological / scalar fields:**
- **PR23-PR25** ELF, NCI/RDG, Laplacian + kinetic-energy density.

**Energy decomposition:**
- **PR26** LMOEDA / Hayes-Stone EDA (molecular, SCF/DFT variant), wired into
  `run_job` and checked against a reference. The closed-shell RHF helpers
  (`vibeqc.eda.eda_lmo`, `eda_morokuma`) already ship, take caller-supplied
  fragment SCF results and are checked against an independent PySCF
  construction. What PR26 adds is the Kohn-Sham form, open-shell fragments,
  the Kitaura-Morokuma PL/CT/MIX separation and the `run_job` wiring.

**Excited-state properties:**
- **PR27** Transition densities + NTOs (QVF writer ready).
- **PR28** Oscillator strengths + UV/Vis spectrum reconstruction.
- **PR29** ECD (Electronic Circular Dichroism).

### Planned: production MPI and large-system scaling

> **Codename candidate:** Amdahl's Axolotl (provisional artwork proposal).

Production distributed-memory support, after the v0.15.x release-paper
hardening line and after the periodic correlation/properties work has
settled enough that the hot paths are clear. Detailed strategy:
[`docs/design_mpi_parallelization.md`](design_mpi_parallelization.md).

- **MPI0** Existing substrate hardening: real `mpirun` smoke tests,
  rank-0 output discipline, MPI metadata in `.out` / `.system`, and
  real multi-rank validation of `GpwJBuilder(mpi_aware=True)`.
- **MPI1** Hybrid MPI x OpenMP policy: scheduler-visible
  threads-per-rank, BLAS-thread controls, and clear "OpenMP only" vs
  "MPI active" logging.
- **MPI2** Periodic k-point farming for multi-k SCF and properties,
  with ordered gathers for per-k matrices and allreduces for weighted
  energies, free energies, entropies, and residual norms. First slice
  landed: `KPointPartition` (`vibeqc.mpi`, 2026-07-28) provides the
  split/reassembly primitive (`local_indices` + `allgather_ordered`,
  real multi-rank tests), and the multi-k GDF cderi build farms
  k-point work across MPI ranks on top of it. D114 adds a separate exact
  complete-output-block farming slice for restricted χ-CCM four-center RHF.
  General SCF-loop farming and the properties reductions above remain open.
- **MPI3** Periodic k/q tuple batching for GDF, COSX, periodic MP2,
  and CCM-style finite-character correlation.
- **MPI4** Molecular coarse-grained MPI: DFT grid batches, HF shell
  tasks for very large molecules, MP2 occupied-pair farming, DLPNO
  pair-domain farming, selected-CI determinant/PT2 batches, and TDDFT
  state/property farming. **BIPOLE far-field MPI** is deferred until a
  route-certified quartet implementation preserves the exact periodic Fock
  domain. Only then can quartet batches be farmed across MPI ranks for very
  large supercells. Tracked post-v0.15 as a conditional MPI4 sub-item.
- **MPI5** Optional distributed dense linear algebra only if replicated
  matrices become the measured memory wall. ScaLAPACK is not a phase-1
  prerequisite.

Hardware policy: single-node OpenMP remains the default. Ethernet
clusters get coarse task farming and infrequent reductions; low-latency
interconnects can opt into finer shell/matrix reductions once benchmarks
show a win.

### `v1.0.0`, feature complete

> **Codename candidate:** Schrödinger's Llama (provisional artwork proposal).

Final polish before declaring "feature complete across both molecular
and periodic." Remaining items:

- ROHF/ROKS ✅ (shipped in v0.13.0).
- meta-GGA + range-separated hybrid validation across the full
  functional table.
- Thermochemistry polish (corrections, low-frequency treatment,
  rigid-rotor / quasi-harmonic switches).
- Miscellaneous bug fixes uncovered in the larger correlation /
  excited-state / periodic / properties stacks.
- Production MPI parallelism is tracked in the production MPI milestone; the v0.12.0 work
  shipped only the optional `mpi4py` substrate and the experimental GPW
  grid overlay.

(ccm-aiccm-roadmap)=
### CCM / AICCM -- the ab-initio Cyclic Cluster Model

> **Shipped as experimental in v0.15.0.** The ab-initio CCM now ships as
> Γ-CCM (`aiccm2026dev-a`) and χ-CCM (`aiccm2026dev-b`) lines in the
> v0.15.x release, gated behind experimental warnings, rather than waiting
> for v2.0.

The Cyclic Cluster Model is vibe-qc's signature real-space approach to
periodic systems. Two distinct approaches are under active head-to-head
comparison:

- **Γ-CCM / `aiccm2026dev-a`** - the union-and-weight/Wigner--Seitz
  integral-weighting CCM
  (`vibeqc.periodic.ccm`). Ships HF, KS-DFT (any libxc functional),
  MP2, UMP2, CCSD(T), properties (gap, charges, dipole, forces), Wannier
  localization, and space-group symmetry analysis. The symmetric four-center is exactly 8-fold
  permutationally symmetric for any lattice. Its RIJCOSX route is a
  construction-specific research approximation; the neutral GDF controls are
  not a three-center hierarchy for Γ-CCM.
- **A-namespace neutral fitted-torus controls** - `run_ccm_uccsd` and the
  `run_ccm_ri_*` and `ccm_dlpno_*` APIs provide restricted/unrestricted
  canonical and local correlation on a separately declared neutral operator.
  They are neither Γ-CCM nor χ-CCM construction results.
- **χ-CCM / `aiccm2026dev-b`** - the finite-translation-group character
  CCM[^xccm-convention]
  (`vibeqc.periodic.chi`). Ships RHF/RKS/UHF/UKS via
  four-center, RI, and RIJCOSX backends in 3D, plus finite-torus RI-MP2,
  DLPNO-MP2, DLPNO-CCSD, and DLPNO-CCSD(T) in 3D, with explicit finite-torus
  convention recording. Every 1D/2D B absolute-energy backend fails closed
  until one shared wire/slab Coulomb convention is derived. Restricted
  four-center RHF farms complete direct-ERI output blocks under MPI; this is
  not yet character/residue or representative-pair local-correlation scaling.

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.

Both lines use the 28-system
[AICCM-2026 benchmark test set](https://github.com/vibe-qc/vibe-qc/blob/main/studies/aiccm-2026/README.md)
for route-specific validation and future CRYSTAL23 thermodynamic-limit
comparisons. No Γ-CCM/χ-CCM approach delta is currently reportable. See the
[Γ-CCM reference](aiccm2026dev_a.md),
[χ-CCM user guide](user_guide/aiccm2026dev_b.md), and the
[experimental AICCM section](experimental/index.md).

Methodological lineage:
[Peintinger & Bredow, *J. Comput. Chem.* **35**, 839
(2014)](https://doi.org/10.1002/jcc.23550).

**Remaining on the CCM roadmap (the production-grade track):**

- **Scalable dense-3-D four-center** - the current padded-ERI and
  neutral-cderi builders are small-cluster validation paths; a scalable
  dense-3-D neutral four-center is needed for production ionic
  correlation.
- **Mixed-boundary Green's functions** (1-D wire / 2-D slab) - both
  lines currently over-bind low-dimensional systems under the 3-D Ewald
  gauge. A shared wire/slab Green's function is the next joint
  deliverable.
- **Production reduced-scaling correlation** - for Γ-CCM
  (`aiccm2026dev-a`) both halves of this item are implemented and gated:
  `ccm_dlpno_ccsd` is the subspace-projected (union-PNO) route, while
  `ccm_dlpno_ccsd_coupled` provides genuine per-pair PNO bases with a
  BvK-torus minimum-image `coupling_radius` screen and per-triple TNO
  (T). What remains open here is the *production* qualifier - the
  symmetry-unique pair reduction, and scaling evidence at production
  bases beyond the small research fixtures these are gated on. Status
  for the χ-CCM (`aiccm2026dev-b`) line is that line's to state.
- **Scalable 3-D analytic CCM gradient** - analytic forces already
  exist for Γ-CCM (`gradient_analytic.py`: RHF/UHF/RKS/UKS, MP2/UMP2,
  CCSD(T)/UCCSD(T), gated against finite difference at 5e-6-class with
  O(h²) Richardson confirmation), and the real-Γ neutral fitted-torus
  control has parity-verified analytic forces plus fixed-lattice
  relaxation at the API level (`run_ccm_direct_gradient` /
  `run_ccm_direct_optimize`). What is genuinely open is production
  scale: the Γ-CCM four-center gradient term is a dense `n_pad_ao⁴`
  validation path (small / 1-D / 2-D clusters only, `_check_padded_eri_size`),
  so 3-D production forces need a scalable weighted ERI-gradient C++
  kernel. Runner-level (`run_periodic_job`) CCM geometry optimization and
  Hessians are also still unwired. Gradient status for the χ-CCM line is
  that line's to state.
- **Bulk polarization (Berry phase / Resta)** - the finite-cluster
  dipole is a symmetry indicator only.
- **Projection-based embedding** - a correlated CCM region inside a
  DFT-treated matrix, the long-term capstone.

## Linear-dependence + screening program (multi-release)

Mathematically the AO overlap matrix `S` must be positive definite. In
Gaussian-basis periodic codes it routinely becomes ill-conditioned and,
with truncated lattice sums and finite arithmetic, can even acquire
**numerically negative eigenvalues**, at which point a silent
canonical-orthogonalisation truncation gives a "converged" SCF on a
basis with the wrong metric signature, and the total energy is off by
orders of magnitude.

CRYSTAL's stance, **refuse to silently truncate; push the problem to
basis-set design**, is the right one for solid-state SCF. PySCF's
optional `remove_linear_dep_` is too quiet by default. Vibe-qc has
adopted CRYSTAL's "loud and explicit" philosophy across this program.

### What landed in v0.7 (October 2025, May 2026 work)

| Feature | API | What it does |
|---|---|---|
| **Critical-severity preflight** | `vq.scf_preflight_overlap_check`, wired into all 8 periodic SCF drivers | Detects non-PSD `S(k)` per k-point. Aborts SCF by default with a citation-rich error message. Override with `allow_critical=True`. |
| **Standalone EIGS diagnostic** | `vq.eigs_preflight(system, basis, kmesh)` | CRYSTAL's `EIGS` keyword equivalent. Builds `S(k)` at every k, prints the spectrum, returns without running SCF. |
| **TOLINTEG distinction** | `vq.disambiguate_critical_overlap(...)` | Re-checks `S` with tightened cutoffs to distinguish basis-set linear dependence from under-converged exchange screening (CRYSTAL ITOL4 / ITOL5 manual p. 130). LiH conventional rocksalt diagnoses as `screening_undertight`, not `basis_set_problem`. |
| **Joint cutoff / Schwarz optimisation** | `vq.optimize_truncation(...)` (default ON via `auto_optimize_truncation=True` in all 8 drivers) | Bisects both knobs jointly to find loosest settings keeping `S` PSD at every k-point. Sanity-checked ranges, max 8 evaluations, joint scaling so neither knob runs far from default while the other stays put. |
| **Primitive filter (`exp_to_discard`)** | `vq.make_basis(mol, name, exp_to_discard=0.1)` | PySCF-style runtime filter, drops Gaussian primitives below threshold, persisted via `_vibeqc_filtered_*.g94` cache for the SAD initial-guess re-load path. |
| **Per-primitive drop transparency** | `vq.format_basis_filter_report(rep)` | Lists every dropped primitive (element, shell, l-letter, exponent, libint coefficient) so the SCF log self-documents any basis-content modification. |
| **PySCF-style canonical orth** | `_canonical_orthogonalizer(*, normalize_diag_first=True)` | Sqrt(diag(S)) pre-conditioning so the threshold operates on a unit-diagonal matrix (Lykos & Schmeising 1961, Löwdin 1970 conventions). |
| **Opt-in orthogonalisation methods** | `vq.orthogonalise_overlap(S, method="auto")`; `vq.canonical_orth`, `vq.pivoted_cholesky_orth`, `vq.symmetric_orth` | Explicit fallbacks (Lehtola 2019 pivoted Cholesky for extreme overcompleteness) when the user opts past the critical-severity abort. |
| **Pob basis sets bundled** | `vq.BasisSet(mol, "pob-tzvp" \| "pob-tzvp-rev2" \| "pob-dzvp-rev2")` | Pre-existing in vibe-qc; the critical-severity error message now actively recommends them as the basis-design fix (Peintinger 2013, Vilela Oliveira 2019). |
| **Active-settings dump** | `vq.format_options(opts)`, `vq.dump_active_settings(plog, ...)` | Every SCF run self-documents, the user never has to read driver source to know what defaults were in effect. Required by the v0.7 transparency directive. |

References landed in code + docstrings:

* Lykos & Schmeising *J. Chem. Phys.* **35**, 288 (1961)
* Löwdin *Adv. Quantum Chem.* **5**, 185 (1970)
* Peintinger, Vilela Oliveira, Bredow *J. Comput. Chem.* **34**, 451 (2013), pob-TZVP
* VandeVondele & Hutter *J. Chem. Phys.* **127**, 114105 (2007), MOLOPT
* Searle, Bernasconi, Harrison, ARCHER eCSE04-16 (2017), k-dependent linear dependence
* Vilela Oliveira et al *J. Comput. Chem.* **40**, 2364 (2019), pob-TZVP-rev2
* Daga, Civalleri, Maschio *J. Chem. Theory Comput.* **16**, 2192 (2020)
* Lehtola *J. Chem. Phys.* **151**, 241102 (2019); *Phys. Rev. A* **101**, 032504 (2020), pivoted Cholesky
* Ye & Berkelbach *J. Chem. Theory Comput.* **18**, 1595 (2022), GTH-cc-pVXZ
* CRYSTAL23 manual: EIGS (p. 103, 398), TOLINTEG (p. 130), OPTBASIS (p. 65), basis-set design discipline (Ch. 17, pp. 394-398)

### `v0.7.x`, module consolidation (deferred)

The linear-dep work currently lives across five modules
(`linear_dependence.py`, `basis_filter.py`, `eigs_preflight.py`,
`orthogonalisation.py`, `options_dump.py`). The user-requested
consolidation into a single coherent module / namespace is deferred
to a v0.7.x maintenance release. Functionality won't change; imports
will get a nicer `vq.basis_screening.*` umbrella.

### Unified `FockMatrix` object across SCF kernels (refactor proposal, deferred)

Every SCF kernel today (`rhf.cpp`, `uhf.cpp`, `rks.cpp`, `uks.cpp`, the
periodic `periodic_scf.cpp` / `bloch.cpp` pair, and the initial-guess
path in `guess.cpp`) independently assembles its own {H_core, S, F,
J, K, (Vxc)} and calls its own
`Eigen::GeneralizedSelfAdjointEigenSolver` (or the periodic
Bloch-block-diagonal analogue) on F/S to get MO coefficients and
orbital energies. FACCTS' ORCA Python Interface (OPI) tutorial *Fock
Matrix Diagonalization* externalises exactly this step for ORCA users:
extract `h_core`/`F`/`S`/`J`/`K` from a finished calculation as named
matrices, assemble the full Fock matrix, solve the generalised
eigenproblem with `scipy.linalg.eigh(F, S)`, and read HOMO/LUMO off the
resulting orbital energies (a Koopmans'-theorem ionisation-energy
estimate is the notebook's worked example).

The pattern worth adopting internally isn't OPI's API (a post-hoc
Python/JSON extraction layer bolted onto ORCA's own C++, not something
to import, see § 10), it's the **shape**: a small, named container
bundling {H_core, S, F, J, K, Vxc} with the diagonalisation step owned
once instead of duplicated per kernel. Candidate design: a
`FockMatrix` (or `ScfMatrices`) struct/class under
`cpp/include/vibeqc/` holding the assembled pieces plus a
`diagonalize()` method wrapping the generalised eigenproblem (the
existing `JKBuilder` ABC already consolidates *building* J/K, this
would sit one layer up and consolidate what happens to the assembled
F). Molecular RHF/UHF/RKS/UKS and the periodic SCF kernels would
construct one instead of hand-rolling their own solver call; Python-
side diagnostics (`DIAG1`/`DIAG2` SCF-stability work above, any future
Koopmans-style estimate) would get a single place to read
`h_core`/`F`/`S`/`J`/`K` back out instead of threading raw arrays
through `bindings.cpp` ad hoc.

This is a refactor spanning every SCF kernel that ships today, squarely
a § 9 "grand refactor, needs maintainer approval before starting" item
rather than a Decide-it-yourself cleanup: a regression introduced here
is a correctness bug across every method vibe-qc offers, not a style
nit. Parked here as a design proposal only; do not start it without
sign-off.

Reference: FACCTS, *Fock Matrix Diagonalization*, ORCA Python Interface
(OPI) 2.0 documentation, notebook tutorial.

### Arbitrary-dimensional tensor engine (refactor proposal, deferred)

The C++ core has no rank-N tensor abstraction at all: `grep`-ing
`cpp/include` and `cpp/src` for `Eigen::Tensor` or
`unsupported/Eigen` returns zero hits. Every object of rank 3 or
higher, the 4-index AO/MO ERI blocks in `mp2.cpp`, the doubles
amplitudes in `ccsd.cpp` (the variable is *named* `T2_flat`, the name
documents the workaround), is carried as a flat `std::vector<double>`
buffer and manually reinterpreted per contraction step via
`Eigen::Map<...>(ptr + offset, rows, cols)` with the index arithmetic
for that particular reshape written out at each call site. It works
(CCSD's `update_amplitudes` is validated to machine precision against
a reference), but the index-order convention for a given tensor lives
implicitly in whichever function last reshaped it, not in a type.

This is exactly the kind of duplication the `FockMatrix` proposal
above targets one level down: as CASSCF/NEVPT2 RDMs, DMRG tensor
networks, periodic multi-k-point ERI blocks, and the **PRNLO**
nonlinear-optical response tensors above (third- and fourth-order
CPHF/CPKS derivatives are rank-3/4 objects by construction) all
accumulate their own from-scratch flat-buffer-plus-manual-reshape
code, the duplication compounds. Candidate design: a minimal
named-index tensor class wrapping `Eigen::Tensor` (from
`unsupported/Eigen/CXX11/Tensor`) for the contraction and reshape
machinery, with **zero new dependency** to justify under § 9,
`unsupported/Eigen` ships inside the same Eigen distribution
`setup_native_deps.sh` already requires (confirmed present alongside
regular Eigen headers wherever Eigen is installed; nothing new to
vendor, license-check, or probe in the banner). Scope would start
narrow, MP2/CCSD's existing rank-3/4 objects as the first migration,
proving the abstraction on code with a machine-precision reference
before touching anything else, not a big-bang rewrite of every
tensor-shaped array in the codebase at once.

Like `FockMatrix`, this is a § 9 "grand refactor, needs maintainer
approval before starting" item: it touches validated correlated-method
kernels, and a transposition bug in a generic reshape layer is a
silent correctness bug in every method built on top of it, not a
style nit. Parked here as a design proposal only.

**A harder follow-on, once the container exists** (NWChem sweep
finding): NWChem's Tensor Contraction Engine (TCE) is not just a
tensor container, it symbolically derives the working equations for
an arbitrary CI/MBPT/CC method and generates the contraction code from
that derivation, spin/spatial/permutation symmetry exploited
automatically, rather than hand-coded per method. That is precisely
what makes CCSDT, CCSDTQ, MBPT(4), and triples-corrected multireference
CC (BW-MRCC, Mk-MRCC) tractable to implement at all, hand-deriving and
hand-coding each one is, in NWChem's own framing, "economically
prohibitive" past CCSD(T). vibe-qc's own CCSDTQ/FCIQMC exclusion
("What's explicitly out of scope," above) is partly a symptom of not
having this: those methods are out of reach by hand-coding, not
necessarily out of reach in principle. Worth remembering as the reason
to build the tensor engine as a *code-generation* target, not merely a
runtime contraction library, if and when it gets built at all; a much
later-stage idea than the container itself, not a parallel proposal to
schedule alongside it.

### OPTBASIS-style basis optimiser: **shipped**; the endeavour moved to `vibe-basis`

CRYSTAL's `OPTBASIS` keyword (manual p. 65) optimises a basis set's
exponents and contraction coefficients while penalising the condition
number of the overlap matrix:

> Ω({α,d}) = E_tot({α,d}) + γ · ln κ({α,d})

This steers the basis-optimiser away from regions of basis-function
space that produce ill-conditioned overlap matrices, the cure is
**built into how the basis is constructed**, not patched at run time.

**Shipped on `main`** (the vibe-qc half of the work):

* `vibeqc.basis_optimization.optimize_bdiis`, the BDIIS driver of
  Daga/Civalleri/Maschio, hardened past the CRYSTAL baseline with a BFGS
  inverse-Hessian, a trust-region + Armijo guard, quasi-Newton fallback on an
  ill-conditioned DIIS `B`, and box-projected convergence.
* `ld_diagnostics.condition_number_penalty`, the γ·ln κ(S) term itself.
* `energy_gradient.py`, **analytic** molecular basis-parameter gradients for
  RHF/UHF/RKS/UKS over exponents *and* contraction coefficients.
* `periodic_energy_gradient.py`, the periodic (k-point) sibling: P1 to P3 landed,
  end-to-end gated on the R13 Ewald-gauge kernel
  ([Archived `handovers/HANDOVER_PERIODIC_BASIS_GRADIENT.md`](https://vibe-qc.com/docs/)).

**Where the rest of it lives.** Matching `OPTBASIS` was never the goal; the
endeavour is *beating* it, via cohesive-energy objectives, basis-topology
search, and cross-validation against a plane-wave limit. That work is driven by
the sibling **`vibe-basis`** package, which carries its own version line and its
own roadmap:

* [`vibe-basis/ROADMAP.md`](https://github.com/vibe-qc/vibe-qc/blob/main/vibe-basis/ROADMAP.md):
  milestones M-0 to M6, engine policy, and the dependencies it has on vibe-qc.

That roadmap depends on this one (periodic gradients, GAPW plane-wave
atomization, BSSE for solids, hybrid periodic frequencies); its § 5 is the
single place those dependencies are listed, and its § 6 is the sync contract
between the two documents. **Do not restate its milestones here**: record on
this side only what vibe-qc itself ships.

References:

* CRYSTAL `OPTBASIS` documentation (manual p. 65).
* Daga, Civalleri, Maschio *J. Chem. Theory Comput.* **16**, 2192 (2020), DIIS-based basis-set optimiser with condition-number penalty.

### Automated regression / parity test suite

Built in a dedicated chat before the split, and shipped: waves 1.0 through
1.7 landed across v0.7.0 to v0.7.3, covering about 30 molecules and 4
periodic systems with three-way vibe-qc / PySCF / ORCA parity at
single-point SCF. The runner lives in `examples/regression/`; its
bibliography map is `examples/regression/REFERENCES.md`.

The wave-by-wave record, the queued test-set expansion and the
which-chat-owns-what notes moved to
[roadmap history](roadmap_history.md#automated-regression--parity-test-suite-separate-chat).

## Long-term vision: language-model-assisted user interaction

Looking past the v2.x feature-complete CCM stack, the project's
ultimate ergonomic target is a **language-model front-end that
removes the input-file barrier** between a chemist's intent and a
running calculation. The user describes what they want to compute,
in plain language, the way they'd explain it to a colleague, and
the system:

1. **Suggests a model.** Picks a sensible starting method
   (functional, basis set, k-mesh, dispersion, solvation) given the
   chemistry described and the resources available, with reasoning
   surfaced so the user can override. Defaults to the most-cited
   choice for the chemistry class (e.g. ωB97X-D/def2-TZVP for
   organic thermochemistry; PBE+D3/pob-TZVP for periodic
   transition-metal solids; HF/6-31G* + MP2 single-point for
   benchmarking).

2. **Generates the input.** Translates the natural-language
   specification into a vibe-qc Python script, molecule from XYZ
   or SMILES, ``run_job(...)`` with the right options, post-
   processing for the requested observables (energies, geometry,
   properties, spectra). The script is human-readable and human-
   editable; the LM is a starting-point assistant, not an opaque
   wrapper.

3. **Submits and monitors.** Bridges to the user's compute
   environment (laptop, workstation, HPC cluster via SLURM /
   PBS / LSF, eventually cloud back-ends), submits the job with
   appropriate resource requests inferred from the memory
   estimator + scaling heuristics (Phase P2), and reports status
   back to the user.

4. **Interprets the output.** Reads the ``.out`` / ``.molden`` /
   ``.cube`` artefacts back, summarises the result in plain
   language ("the SCF converged in 14 iterations to ‑76.0107 Ha;
   the dipole moment is 1.85 Debye, in good agreement with the
   experimental value of 1.85 D"), and surfaces follow-up
   questions ("would you like to verify this with a tighter basis
   or run a frequency calculation?").

This sits *on top of* the Python API, never replacing it. The
Python entry points stay the canonical, scriptable interface for
power users and programmatic workflows; the LM front-end is the
ergonomic on-ramp for chemists who'd otherwise be paralysed by
input-file syntax. The architecture decouples cleanly: a separate
Python package depending on vibe-qc, presumably called
``vibe-qc-assistant`` or similar, with the LM provider configurable
and the underlying engine unchanged.

Concrete prerequisites the rest of the roadmap delivers toward
this vision:

* **Banner provenance** (shipped v0.1.0+), every persisted run
  log identifies the exact build, so the LM can cite it
  unambiguously and regenerate identical calculations.
* **Memory estimator + scaling heuristics** (shipped Phase P2),
  feeds the resource-request logic in step 3.
* **``run_job`` ``.out`` format** (shipped v0.1.0+), the
  structured output the LM reads back in step 4.
* **Tutorial corpus** (15+ tutorials shipping by v0.4.x),
  training-data foundation; every tutorial is a "what does this
  chemistry look like in vibe-qc" worked example.
* **Bug-hunt brief** (committed at ``docs/bug_hunt_brief.md``),
  the same machine-readable policy artefacts that govern the
  bug-hunt chats can govern the LM front-end.

No commitment to ship dates yet, this is a vision target, not a
milestone. The phasing question (does it land alongside v1.0
feature-complete, or after v2.x CCM?) is open.

## Release codenames

The [upcoming artwork gallery](codename_artwork.md) contains prepared images
for four open milestones and v1.0.0. Their names are candidates paired with
milestone themes; a codename is assigned when its release is cut.
``RELEASE_CODENAMES`` in ``python/vibeqc/banner.py`` is the source of truth
for released codenames, and the catalog table below mirrors it.

```{toctree}
:hidden:

codename_artwork
```

Starting with v0.5.0, every **major** and **minor** release gets a
codename of the form *"Scientist's Animal"*. Patch releases (v0.5.1,
v0.5.2, ...) inherit their parent minor's codename.

Naming rules:

* **Scientist** is chosen for thematic fit with the release's
  flagship feature, typically the person whose work the milestone
  most directly builds on. For v0.5.0 (vibrations / Hessian /
  thermochemistry) that's E. Bright Wilson (FG-matrix vibrations).
  For v0.15.0 (DLPNO-CCSD(T)) that was Frank Neese (DLPNO-CCSD(T)
  in ORCA).
* **Animal** is random and intentionally a bit silly. The
  juxtaposition of a famous physical chemist with an unexpected
  animal is the joke ("Schrödinger's Llama", "Ewald's Cheetah",
  "Mulliken's Cat"). Avoid animals already used by other projects'
  codename schemes (Ubuntu, Debian) where there's clear conflict;
  otherwise anything goes.
* **Format on every public artefact** that mentions the release:
  ``v0.5.0 "Wilson's Otter"``, version number primary, codename
  in quotes after.

Where the codename appears:

* The **runtime banner**: ``Release v0.5.0 "Wilson's Otter"`` on a
  tagged clean tree, ``dev 0.5.0.dev0 "Wilson's Otter" (main @
  <sha>)`` on a development build heading toward that release. Dev
  builds inherit the codename of the upcoming release so the banner
  is fun even before the tag drops. The lookup logic strips
  ``.devN`` / ``aN`` / ``bN`` / ``rcN`` PEP-440 suffixes; see
  :func:`vibeqc.banner.codename_for_version` for the catalog.
* The CHANGELOG header for that release: ``## [v0.5.0] - YYYY-MM-DD -
  *Wilson's Otter*``.
* The ``git tag -a`` annotation: title line includes the codename.
* The release-notes blog post / website announcement.

**Pre-staging + referencing (for chats writing docs or code).** A
codename is added to ``RELEASE_CODENAMES`` as soon as it is decided,
which is often *before* the cut (e.g. Bartlett's Goose for v0.14.0 was
pre-staged while main was still on the 0.13.x line). A pre-staged entry
renders nothing until a build of that ``X.Y.x`` line exists, so a dev
build of main shows the *current* upcoming-release codename, not a later
pre-staged one. When you need the codename in docs or code:

* **Prefer the lookup** ``codename_for_version("0.14.0")`` over a
  hardcoded literal, so the text stays correct across the cut.
* **Forward-looking content** about an unreleased version uses that
  version's (possibly pre-staged) codename.
* **References to the current released version** use the latest tagged
  codename, i.e. whatever ``codename_for_version`` returns for the
  released version number, not the pre-staged next one.

Pre-policy releases (v0.1.0 through v0.4.0) were **retroactively
assigned codenames** in the v0.15.1 cycle. Their original tagged
banners read ``Release v0.4.0`` etc. without a quoted suffix.

Implementation. The catalog lives in
``python/vibeqc/banner.py`` as ``RELEASE_CODENAMES`` (plain Python
dict). When cutting a new minor release, add the entry there in
the same commit that bumps ``pyproject.toml``. Patch releases
inherit their parent minor's codename automatically: the lookup tries
the full version, then falls back to the minor. Three patch releases
carry their own entry (v0.7.1 to v0.7.3).
``tests/test_banner_codename.py`` pins the catalog contract.

Codename catalog (filled in as releases ship):

| Tag | Codename | Scientist | What they gave us |
|---|---|---|---|
| ``v0.1.0`` | **"Roothaan's Raven"** | Clemens C. J. Roothaan (1918-2019) | Formulated the **Roothaan-Hall equations** (1951) with George G. Hall -- the matrix-eigenvalue formulation of the Hartree-Fock integro-differential equations that made HF computationally tractable and is the foundation of every molecular quantum-chemistry code. The Raven: the first bird, intelligent, lays the foundation everything else builds on. |
| ``v0.2.0`` | **"Ewald's Eagle"** | Paul Peter Ewald (1888-1985) | Invented the **Ewald summation** (1921) for conditionally convergent lattice sums -- the real-space/reciprocal-space split that makes periodic Coulomb calculations possible. The Eagle: soars above the periodic landscape with a Fourier perspective, seeing both the near and the far field at once. |
| ``v0.2.5`` | **"Wigner's Wolf"** | Eugene P. Wigner (1902-1995) | Nobel Prize 1963 for group theory in quantum mechanics. His **Wigner D-matrices** are the mathematical core of the symmetry exploitation phase: every space-group operator maps AO shells through Wigner's representation theory, enabling orbit compression and |G|-fold storage reduction. The Wolf: hunts in packs -- group theory's power is in the collective action of operators. |
| ``v0.3.0`` | **"Mulliken's Magpie"** | Robert S. Mulliken (1896-1986) | Nobel Prize 1966 for **molecular orbital theory**. Mulliken population analysis, the workhorse charge-partitioning scheme, anchors this release's property suite alongside band-structure plotting, density-of-states, COOP/COHP, and Mayer bond orders. The Magpie: collects shiny observables from the wavefunction and arranges them for display. |
| ``v0.4.0`` | **"Hellmann's Hedgehog"** | Hans Hellmann (1903-1938) | The **Hellmann-Feynman theorem** (1937) proves that analytic nuclear gradients require only the electron density and the derivative of the Hamiltonian -- no response of the wavefunction parameters -- making geometry optimization and molecular dynamics computationally feasible. This release added heavy-element capability via ECPs (libecpint), open-shell periodic SCF, and tight-cell convergence machinery. The Hedgehog: well-protected by spines (ECPs replace core electrons); prickly to handle but gets the job done. |
| ``v0.5.0`` | **"Wilson's Otter"** | E. Bright Wilson, Jr. (1908-1992) | Co-author of *Molecular Vibrations* (1955); the **FG-matrix method** for normal-mode analysis that every quantum-chemistry code uses to compute vibrational frequencies. Wilson's formalism is the mathematical core of Phase 17 (analytic Hessian, IR intensities, thermochemistry). |
| ``v0.6.0`` | **"Pulay's Owl"** | Peter Pulay (b. 1941) | Invented **DIIS** (direct inversion in the iterative subspace, 1980), the convergence accelerator that makes SCF practical; formulated the **Pulay correction terms** (1969) that underpin analytic nuclear gradients. The Owl sees what others miss -- sub-meV displacement-derived forces. |
| ``v0.7.0`` | **"Löwdin's Compass"** | Per-Olov Löwdin (1916-2000) | Gave quantum chemistry **canonical orthogonalisation** (Adv. Quantum Chem. 5, 185, 1970) and the symmetric S^{-1/2} transformation that anchors vibe-qc's linear-dependence diagnostic, screening optimiser, and the Lehtola pivoted-Cholesky fallback. The Compass navigates the basis-set/cutoff space and finds the loosest PSD-keeping configuration without silent truncation. |
| ``v0.8.0`` | **"Grimme's Gecko"** | Stefan Grimme (b. 1963) | His group's dispersion-correction lineage -- **DFT-D3** (2010), **DFT-D4** (2017/2019), the D3-BJ damping, the EEQ charge model -- turned DFT from a qualitative to a quantitative tool for noncovalent interactions. Also co-developed the B2PLYP double-hybrid family (2006) with the Martin group. The Gecko is small, fast, and sticks -- the molecular-methods arc built on a clean dispersion + double-hybrid foundation. |
| ``v0.9.0`` | **"Knowles's Kingfisher"** | Peter J. Knowles | Co-author of the **MOLPRO** MRCI/direct-CI machinery, the Knowles-Handy 1984 FCI algorithm, and the Werner-Knowles CASSCF. His coupled-cluster and configuration-interaction methods are the computational foundation of the post-HF correlation ladder (MP2 -> MP3 -> MP4 -> CCD -> CCSD). The Kingfisher is precise, patient, strikes once. |
| ``v0.10.0`` | **"Pisani's Penguin"** | Cesare Pisani | Founded the **CRYSTAL code** at Turin (1976-), the Gaussian-basis periodic-SCF program whose methodology vibe-qc inherits: crystalline orbitals, real-space lattice sums, Ewald-accelerated Coulomb, and the "refuse to silently truncate" philosophy. The Penguin lives at extremes -- structured, solid-state, low-T. |
| ``v0.11.0`` | **"Sun's Stingray"** | Qiming Sun | First author of the **Sun-Berkelbach RSGDF** algorithm (JCP 147, 164119, 2017) for range-separated Gaussian density fitting with analytic-FT long-range terms, and creator of **PySCF**, vibe-qc's microhartree-parity reference. The Stingray glides over the periodic landscape via Fourier transform -- k-space agile. |
| ``v0.12.0`` | **"Knuth's Beaver"** | Donald E. Knuth (b. 1938) | Author of *The Art of Computer Programming* (1968-), the foundational text of algorithmic analysis. Knuth's rigorous design philosophy -- data structures, combinatorial enumeration, amortised analysis -- is the intellectual substrate of DLPNO pair-classification, integral-transformation scheduling, per-pair solvers, and the heat-bath selected-CI kernel. The Beaver is the industrious builder: nine headline features in a single minor cycle. |
| ``v0.13.0`` | **"Wisesa's Fox"** | Pandu Wisesa | First author of the **generalised-regular k-grid** algorithm (Comp. Mat. Sci. 110, 1, 2016, with McGill & Mueller), which searches over reciprocal sublattices for ~2x more efficient k-point sampling than axis-aligned Monkhorst-Pack. The Fox is nimble and efficient. |
| ``v0.14.0`` | **"Bartlett's Goose"** | Rodney J. Bartlett (b. 1944) | Co-author of the **Purvis-Bartlett CCSD** equations (1982), the foundation of gold-standard molecular coupled-cluster theory. His group at the University of Florida has driven CC method development for four decades -- CCSD, CCSD(T), EOM-CC, and the coupled-cluster response framework. The Goose: the accuracy reference every cheaper method is measured against. |
| ``v0.15.0`` | **"Neese's Cheetah"** | Frank Neese (b. 1967) | Co-inventor of **DLPNO-CCSD(T)** (Riplinger-Neese 2013), the domain-based local pair natural orbital approximation that makes canonical-accuracy coupled cluster feasible on realistic molecules. His group's SparseMaps framework and the ORCA program's DLPNO family (DLPNO-MP2, DLPNO-CCSD, DLPNO-CCSD(T), DLPNO-NEVPT2) are the gold standard in local correlation. The Cheetah: the speed local correlation buys. |
| ``v0.16.0`` | **"Pople's Puffin"** | John Pople (1925-2004) | Nobel Prize 1998 for **ab initio methods made usable by anyone**, which is what splitting one project into vibe-qc, vibe-view, vibe-queue and qvf, each adoptable on its own, is for. |
| ``v0.17.0`` | **"Tew's Tern"** | David P. Tew | **Periodic local correlation** (the Nejad-Zhu-Tew 2025 Paper I lineage). v0.17.0 shipped the symmetry and periodic foundations that route will consume. The Tern: navigates by the lattice it crosses. |

---

We stand on the shoulders of these scientists. Every codename is chosen
to honour the work that made the release possible -- not just the
method it ships, but the conceptual breakthroughs that the method rests
on. The animal is intentionally a bit silly (the juxtaposition of a
famous physical chemist with an unexpected animal is the joke), but the
scientist tribute is serious.

## Dependencies

Each milestone tag implies everything from earlier milestones. The
dependency graph is almost strictly linear:

* **v0.2.0** needs nothing beyond v0.1.0.
* **v0.2.5** (symmetry core) needs v0.2.0, the real-space matrix
  reduction operates on the Ewald-composed periodic Fock / J. Strictly
  additive: symmetry-disabled code paths continue to work unchanged.
* **v0.3.0** is independent of the remaining 12e-c work and of v0.2.5
 - visualization is useful even before quantitative bulk SCF or
  symmetry speedups land.
* **v0.4.0** needs v0.2.0 for the Ewald infrastructure that tight-cell
  SCF will lean on. Phase SYM4 (IBZ Fock build) bundles in here and
  depends on v0.2.5 SYM1/SYM2.
* **v0.5.0** dispersion + solvation are independent. Periodic gradients
  / phonons need **v0.4.0 C1** for the underlying SCFs to converge
  cleanly.
* **v0.12.0** can be started in parallel with periodic development because
  molecular correlation + MR methods are independent of the periodic stack.
* **v0.14.0** extends v0.12.0 with canonical CCSD(T) + DLPNO + eigensolvers.
* **v0.15.0** extends v0.14.0 with the comprehensive feature freeze.
* **Native BvK periodic local correlation** (next minor) needs molecular
  DLPNO-MP2 (✅) and the periodic-DF foundation from v0.8-v0.13. The
  megacell/toroidal periodic-correlation route already shipped in v0.13-v0.15.
* **Periodic CC** needs native BvK periodic local correlation.
* **MR + excited-state extensions** build on the MR foundation
  ✅ shipped in v0.12-v0.15; TDDFT engine ✅ shipped in v0.12.0.
* **The properties suite** needs v0.5.0 (Phase 17 Hessian for
  Raman / VCD) and an analytic periodic stress for elastic / piezo /
  photoelastic (the FD stress ships and is not enough for these),
  periodic CC (periodic CCSD if PR16-PR22 want correlated dielectric /
  Born), and the MR + excited-state milestone (TDDFT for PR27-PR29). Within the milestone,
  PR7 (CPHF/CPKS) gates PR8-PR22; PR1-PR6 and PR23-PR25 are
  CPHF-independent and can ship first.
* **Production MPI and large-system scaling** needs the
  v0.15.x hardening baseline, the periodic local-correlation and
  periodic CC hot paths, and the properties-suite workloads for k-resolved
  post-processing.
* **v1.0.0** sums everything above.
* **v2.x** (CCM track), pulled forward into v0.15.x as experimental
  Γ-CCM and χ-CCM lines. The remaining production-grade items
  (scalable dense-3-D neutral four-center, mixed-boundary Green's
  functions, production reduced-scaling correlation, analytic
  gradients, bulk polarization, projection-based embedding) are
  individually sequenced rather than gated as a monolithic track.

## Tutorial parity (ORCA + CRYSTAL)

This section is the *user-facing* lens on the engineering roadmap: how
many ORCA / CRYSTAL tutorials a user could reproduce in vibe-qc today,
which ones are blocked by which milestone, and which ones we could
write up *now* without any new vibe-qc code. The audit was compiled
from the ORCA tutorial index and the CRYSTAL tutorial index.

**Audit tags.** Each table below carries its own tag, because the three were
last checked at different times and against different evidence. A tag says
what was checked and how, so a reader can tell a re-derived row from an
inherited one.

**Last full audit: 2026-06-26.** On 2026-09-13 milestone version numbers
below were replaced by milestone names, and blockers contradicted by later
releases were corrected; the remaining rows were not re-audited for
v0.16.0 to v0.17.2. (That audit was post-v0.15.0 *Neese's Cheetah*
comprehensive feature freeze: canonical CCSD(T), DLPNO-CCSD(T) + U-DLPNO,
CASSCF analytic gradients + geom-opt framework, COOP/COHP, QTAIM, fat bands,
vibe-view QVF v1.2, TD-DFT engine + NTOs, periodic-XC cross-cell fix,
CASPT2/NEVPT2 case-partitioned, AICCM Γ-CCM + χ-CCM (experimental), basis_toolkit,
eigensolver framework, GAPW, GFN2, native-D4 (experimental), periodic COSX,
MDF, BIPOLE gradient hardening, MSINDO to Xe, MACE MLIP, experimental MPI substrate,
GR k-grids, Gilat net). Per the
[per-quarter docs cadence](release_process.md),
this audit refreshes at every release tag.

Status legend: ✅ works · 🟡 works with glue (ASE / small wrapper) ·
❌ needs vibe-qc implementation.

### ORCA, molecular chemistry

> **Audit tag: API sweep at `v0.17.6`, 2026-09-18.** Every name, example path
> and tutorial reference in this table resolves: checked mechanically across
> all 99 rows of the three parity tables, 0 unresolved. Status markers were
> re-derived from the public API and the runner's own refusals, which settles
> "the capability exists and is reachable" but not "the tutorial reproduces",
> since only a run settles that. A ✅ here is the first claim, not the second.
> See [what this sweep changed](#what-the-2026-09-18-sweep-changed).


| Tutorial | Status | Blocker |
|---|---|---|
| Hello water | ✅ | - |
| Input / Output | ✅ | - |
| Running in parallel | ✅ | P1.1 (`num_threads=`) |
| GUI / orbital viewers | 🟡 | External (Avogadro, Jmol), `.molden` works |
| Single points | ✅ | - |
| Geometry optimization | ✅ | - |
| Charges (Mulliken, Löwdin) | ✅ | Phase 18 |
| Bond analysis (Mayer) | ✅ | Phase 18 |
| Dipole moment | ✅ | Phase 19 |
| Vibrational frequencies | ✅ | Phase 17a-e: analytic CPHF/CPKS Hessian for RHF/UHF/RKS/UKS; the `vibrational_frequencies` tutorial |
| Thermodynamics | ✅ | Phase 17a-3: full ZPE/U/H/S/G(T,p) on top of analytic Hessian; the `thermodynamics` tutorial |
| Conformer search (GOAT) | ❌ | Needs a GOAT-style conformer driver; GFN2-xTB itself ships gated experimental |
| Implicit solvation | ✅ | CPCM shipped at v0.12.0 via `SolutePotentialProvider`; the `solvation_water` tutorial covers CPCM water |
| Explicit solvation (SOLVATOR) | ❌ | Out of scope (use ASE + classical FF) |
| Dispersion (D3 / D4) | ✅ | D3-BJ shipped at v0.6.x; D4 shipped at v0.8.0 (D5a-D5d), validated against CRYSTAL23 on 87 benchmark crystals |
| NEB-TS | ✅ | `neb_reaction_path.md` tutorial (ammonia umbrella inversion at HF/STO-3G, barrier 11.14 kcal/mol) |
| IRC | ✅ | `vq.run_irc`, molecular only (damped steepest descent; no Gonzalez-Schlegel integrator yet) |
| KIE | 🟡 | Analytic Hessian + thermo ship (Phase 17); ratio script is user-side |
| Fukui functions | 🟡 | Scriptable today; native via PR1 (properties milestone) |
| LED energy decomposition | 🟡 | DLPNO-CCSD(T) shipped (v0.14.0); closed-shell RHF EDA helpers ship (LMO-EDA, Kitaura-Morokuma); Kohn-Sham form and `run_job` wiring **PR26** (properties milestone) |
| FOD analysis | ❌ | Needs finite-T DFT (post-1.0) |
| IR intensities | ✅ | Phase 17a-2: dipole-derivative tensor + Wilson-Decius-Cross |
| Raman intensities | ❌ | Phase **PR13** (properties milestone), needs polarisability derivatives |
| UV/Vis (TDDFT) | 🟡 | TDDFT engine (Casida + TDA) shipped (v0.12.0); spectrum reconstruction + NTO post-processing in the MR + excited-state milestone |
| ECD / VCD | ❌ | T1 + Phase **PR29** (ECD) / **PR14** (VCD) (properties milestone) |
| NMR (GIAO) | ❌ | Phase **PR10** (properties milestone) |
| Spin-spin coupling (J) | ❌ | Phase **PR11** (properties milestone) |
| EPR / hyperfine | ❌ | Phase **PR12** (properties milestone); spin-orbit terms post-1.0 |
| ESP charges (CHELPG / RESP) | ❌ | Phase **PR3** (properties milestone) |
| Hirshfeld / Bader / NBO | 🟡 | QTAIM (PR2) + COOP/COHP shipped (v0.15.0); classical molecular Hirshfeld ships; NBO helpers are provisional and NPA is unavailable; iterative Hirshfeld (PR1) and production NBO (PR4) are in the properties milestone |
| Quadrupole / higher multipoles | ❌ | Phase **PR5** (properties milestone) |
| Polarisability / hyperpolarisability | 🟡 | The static dipole polarisability ships for closed-shell RHF (`dipole_polarizability_rhf`, via CPHF). UHF / KS polarisability is **PR8** and hyperpolarisability **PR9**, both on the CPHF substrate **PR7** (properties milestone) |
| ELF / NCI / RDG plots | ❌ | Phases **PR23** / **PR24** (properties milestone) |
| Relativistic (ZORA / DKH) | ❌ | Post-1.0 stack |
| ONIOM / QM-XTB | ❌ | Needs ONIOM coupling; GFN2-xTB itself ships gated experimental |

### CRYSTAL, solid-state chemistry

> **Audit tag: API sweep at `v0.17.6`, 2026-09-18.** Every name, example path
> and tutorial reference in this table resolves: checked mechanically across
> all 99 rows of the three parity tables, 0 unresolved. Status markers were
> re-derived from the public API and the runner's own refusals, which settles
> "the capability exists and is reachable" but not "the tutorial reproduces",
> since only a run settles that. A ✅ here is the first claim, not the second.
> See [what this sweep changed](#what-the-2026-09-18-sweep-changed).


| Tutorial | Status | Blocker |
|---|---|---|
| Geometry input | ✅ | POSCAR I/O + `PeriodicSystem` |
| Basis set input | ✅ | pob-\* + CRYSTAL-format parser |
| SCF & other input | ✅ | `PeriodicSCFOptions` |
| Single-point energy (1D molecular) | ✅ | H-chain / H₂-chain examples |
| Single-point energy (2D / 3D tight cell) | ✅ | C1a (level shift) + C1b (smearing) + **C1c (EDIIS + EDIIS+DIIS hybrid, v0.8.0)** all shipped; quadratic-Newton fallback for the hardest oscillating cases |
| SCF convergence tools | ✅ | Damping + Pulay DIIS (default) + Saunders-Hillier level shift (C1a) + Fermi-Dirac smearing (C1b) + **EDIIS+DIIS hybrid (C1c, v0.8.0)** + quadratic-Newton fallback all shipped |
| Geometry optimization (periodic) | ✅ | Full Phase **G1** (a-e) shipped on main: 1e + nuclear + S analytic, J piece via true-periodic ERIs, RKS pure-DFT XC Pulay, multi-k assembly, UKS path, ASE bridge for `atoms.get_forces()`. K-gradient (HF / hybrid exchange) deferred to v0.6.x, pure-DFT periodic relaxations work today |
| Quick tour: single-point energy (CRYSTAL Tutorial port) | ✅ | `examples/periodic/input-mgo-pob-tzvp.py` (canonical pob-TZVP testbed); **v0.8.0 alternate**: `vqfetch mp --id mp-1265` end-to-end (The `mgo_from_materials_project` tutorial) |
| Quick tour: band structure (CRYSTAL Tutorial port) | ✅ | `examples/periodic/input-si-bands.py` (silicon diamond Hcore bands) |
| Complete solid-state walkthrough (CRYSTAL Tutorial port) | ✅ | `the `lih_pob_tzvp_solid_state` tutorial` + `examples/periodic/input-lih-pob-tzvp.py`, multi-k periodic HF, IBZ reduction, Hcore bands + DOS + PDOS, HOMO Bloch cube, full v0.5.x logging surface. **v0.8.0 update**: the `lih_multi_k` tutorial (LiH multi-k via `run_krhf_periodic_gdf`) targets Peintinger 2013 SI Table 2 reference (-8.149050 Ha/cell) at sub-µHa parity to PySCF.pbc.KRHF |
| Debug-friendly periodic DFT input (CRYSTAL Tutorial port) | ✅ | `examples/periodic/input-nacl-sto3g-dft.py`, RKS-LDA / sto-3g / NaCl, `VIBEQC_FAST_DEBUG=1` knob, threads ProgressLogger + perf log + .system manifest |
| Equation of state | 🟡 | Birch-Murnaghan fit helper (`vibeqc.eos.fit_birch_murnaghan`) on user-run volume points; no Vinet form, no sweep driver |
| Surfaces / slabs | ✅ | `slab_adsorbate_dft` and `surface_embedding` tutorials; Phase C1 + G1 BIPOLE all shipped |
| Defects / supercells | ✅ | Phase 15 (UHF/UKS) shipped; Madelung / per-k charge-correction (12e-c-4c-iv) shipped; **v0.8.0 the `open_shell_mg_cation` tutorial** (open-shell Mg⁺• in vacuum-padded cell + Makov-Payne correction) is the worked example. AFM ground-state initial guess (R3) still queued for v0.x.x |
| Magnetic systems | ✅ | `open_shell_mg_cation` tutorial (Mg+ in vacuum-padded cell + Makov-Payne correction); Phase 15a (Γ-UHF) / 15b (multi-k UHF) / 15c-3 (Γ + multi-k UKS) all shipped; multi-k UHF/UKS un-gated at v0.8.0 |
| Metallic systems | ✅ | `fermi_dirac_smearing` and `gilat_net_metals` tutorials; GR k-grids shipped at v0.13.0; Gilat-net metallic BZ integration partially shipped at v0.13.x |
| Metal/oxide interfaces | 🟡 | Phase 14a-c+e (ECPs) + C1a/C1b + Phase 15a/b/c shipped, and Phase 14f, the CRYSTAL `INPUT` ECP-block parser, is in `vibeqc.basis_crystal` (`parse_crystal_atom_basis`, `CrystalECP`, `crystal_ecp_to_libecpint_arrays`), not on a branch. The old note here gated this row on a `basissetdev` merge; that branch was never transferred to this repository. What keeps the row at 🟡 is that there is no dedicated tutorial |
| Nanorods / nanocrystals (finite) | 🟡 | Already work via molecular `run_job` |
| Fullerenes | 🟡 | Molecular; runnable, slow |
| Band structure | ✅ | Phase **V4** |
| Density of states | ✅ | Phase **V5** |
| PDOS (atom / l-projected) | ✅ | Phase **V5b** (the `pdos` tutorial) |
| Charge-density map (molecular) | ✅ | Phase **V1** (cube) |
| Periodic Bloch-orbital map | ✅ | Phase **V3** (the `periodic_orbital_cubes` tutorial) |
| Madelung constants (Ewald) | ✅ | Phase 12e-a + example |
| Vibrational frequencies (periodic) | 🟡 | Γ-point finite-difference frequencies (`run_periodic_job(hessian=True)`, GAPW `PhononCalculator`); no ZPE from `run_periodic_job` |
| Phonon dispersion | ❌ | Phase **21b**; only Γ-point phonons ship |
| Thermodynamics (QHA, EOS at T) | ❌ | Phase **21c** |
| Dielectric / polarisability | ❌ | Phases **PR8** and **PR17** / **PR18** (properties milestone) |
| Elastic / piezo / photoelastic | ❌ | Phases **PR19**-**PR21** via CPHF / CPKS (properties milestone) |
| TS search (periodic) | ❌ | Periodic Hessian, post-Phase 21 |
| Periodic MP2 (CRYSCOR) | 🟡 | Megacell/toroidal route shipped (v0.13-v0.15); native BvK translational route is the next minor's milestone |
| 4f-in-core ECPs | ✅ | **Phase 14a-d shipped at v0.7.x to v0.8.0**: libecpint 1.0.7 vendored + `ecp60mdf` library covers f-block (post-Yb). Manual `ECPCenter` recipe, or `vibeqc.auto_ecp_centers`, which is a shipped top-level helper rather than the `basissetdev`-conditional one the old note described |
| Born effective charges | ❌ | Phase **PR16** (properties milestone) |
| Dielectric tensor (ε∞ / εstatic) | ❌ | Phases **PR17** / **PR18** (properties milestone) |
| Elastic constants | ❌ | Phase **PR19** (properties milestone; needs stress on the target route, where finite-difference stress ships only for GPW/GAPW) |
| Piezoelectric tensor | ❌ | Phase **PR20** (properties milestone) |
| Photoelastic tensor | ❌ | Phase **PR21** (properties milestone) |
| Periodic IR / Raman intensities | ❌ | Phase **PR22** (properties milestone; needs Born charges **PR16** on top of the shipped Γ-point phonons) |
| Spin-orbit / noncollinear | ❌ | Post-1.0 |
| Electron transport | ❌ | Out of scope |
| **CCM track** (defects, embedded clusters) | 🟡 | Experimental Γ-CCM + χ-CCM shipped (v0.15.0); production-grade items individually sequenced |

### ASE, workflow tutorials

> **Audit tag: API sweep at `v0.17.6`, 2026-09-18.** Every name, example path
> and tutorial reference in this table resolves: checked mechanically across
> all 99 rows of the three parity tables, 0 unresolved. Status markers were
> re-derived from the public API and the runner's own refusals, which settles
> "the capability exists and is reachable" but not "the tutorial reproduces",
> since only a run settles that. A ✅ here is the first claim, not the second.
> See [what this sweep changed](#what-the-2026-09-18-sweep-changed).


ASE ships its own [tutorial set](https://ase-lib.org/examples_generated/tutorials/index.html)
focused on **workflows** rather than methods, geometry optimization
loops, constrained relaxations, NEB, basin hopping, MD,
EOS-from-stress, defect supercells, etc. They're calculator-agnostic
by design: any code that exposes the ASE Calculator interface
(energy + forces, optionally stress / dipole / Hessian / magmoms)
plugs in directly.

Now that vibe-qc ships ``vibeqc.ase.VibeQC`` with energy + forces +
dipole + Hessian + polarizability (Phase A, v0.5), most molecular
ASE tutorials are reproducible with a one-line calculator swap.
Periodic-stress-driven tutorials are partly reachable: a periodic stress
ships, but only as a finite difference (`compute_stress_gpw` /
`compute_stress_gapw` over strain, and one native batched FD pass on the
semiempirical routes), so a tutorial that scans volumes and fits works while
one that relaxes the cell against a stress does not. QM/MM and
Wannier-functions tutorials need extensions still out of scope.

| ASE tutorial | Status | Blocker |
|---|---|---|
| ASE Databases (introduction) | ✅ |, (db is calc-agnostic) |
| ASE Databases, surface adsorption study | 🟡 | Periodic energy works; surfaces benefit from **G1** for relaxation |
| Atomization energy | ✅ |, (energy only, ships) |
| Band Structures of Bulk Structures | 🟡 | We compute bands directly; ASE's wrapper assumes plane-wave codes, small adapter needed |
| Bulk Structures and Relaxations | ✅ | G1 periodic gradients shipped; `periodic_geometry_optimization` tutorial + ASE `BFGS`/`BFGSLineSearch` optimizers work |
| Constrained Calculations, Surface diffusion energy barriers | 🟡 | G1 shipped, needs constraint + cell-fixing wrapper (minor ASE glue) |
| Defect calculations, ASE Tools | ❌ | Needs **G1** + bulk relaxation; defects are a v0.6+ tutorial topic anyway |
| Surface diffusion via NEB | 🟡 | Same, G1 shipped, needs constraint wrapper |
| Dimensionality analysis | ✅ |, (post-processing, no calc) |
| EOS, Equation of state introduction | ✅ | G1 relax plus `vibeqc.eos.fit_birch_murnaghan` over a volume scan; third-order Birch-Murnaghan only |
| EOS, Calculating Delta-values | 🟡 | Same, G1 shipped for relax |
| Equilibrating acetonitrile via MD | 🟡 | NVE MD works on vibe-qc forces; energy-conservation diagnostic in Phase D |
| Equilibrating a TIPnP water box | 🟡 | Same, works for QM forces, classical FF needs ASE's own calculator |
| Global optimization, basin hopping | ✅ |, (energy + forces, ships) |
| Lattice constants from EOS / stress | 🟡 | The EOS half works (volume scan plus BM3 fit). The stress half does not: the periodic stress is FD-only and `optimize_cell=True` fails closed on GPW / GAPW / GDF / BIPOLE |
| Molecular dynamics (general) | ✅ |, (Phase D ships an NVE example) |
| NEB, Dissociation of a molecule | ✅ |, (Phase D ships) |
| NEB with IDPP | ✅ |, (IDPP is interpolation-only, calc-agnostic) |
| NEB + Dimer on Al(110) self-diffusion | 🟡 | G1 shipped, needs constraint wrapper |
| QM/MM Simulations | ❌ | Needs MM region, out of v0.5 scope; **post-1.0** |
| Vibrational Modes of a Molecule | ✅ | Analytic CPHF/CPKS Hessian via Phase 17a-e (RHF/UHF/RKS/UKS); FD via `ase.vibrations.Vibrations` as a fallback for hybrid-UKS |
| Partly occupied Wannier Functions | ❌ | Needs Wannier projector, far future |

The Phase D scripts under
[`examples/ase_workflows/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/ase_workflows)
will cover the ✅ rows directly (BFGS opt, Vibrations,
molecular NEB, NVE MD), and the user guide page
[`docs/user_guide/ase_integration.md`](user_guide/ase_integration.md)
will document the ``run_*_ase`` adapter pattern that lets users
plug VibeQC into any of these workflows. The 🟡 rows are
"works with a small adapter", for example, ASE's band-structure
plot expects calculators to expose particular grid attributes;
a tiny adapter is still owed. Periodic gradients (**G1**) shipped; the
remaining stress-blocked rows wait on an *analytic* periodic stress, not on
a stress at all, since the FD one ships.

Re-audit on every minor release; cross-link any newly-✅ rows to
the corresponding tutorial under `docs/tutorial/` or example
script under `examples/ase_workflows/`.

### What the 2026-09-18 sweep changed

Three rows moved, each because the repository had overtaken the note:

* **ORCA, "Polarisability / hyperpolarisability": ❌ to 🟡.** The static
  dipole polarisability ships for closed-shell RHF through CPHF
  (`dipole_polarizability_rhf`). The row had it entirely behind the CPHF
  substrate **PR7**, which is still true of the UHF / KS case and of
  hyperpolarisability, but not of what ships.
* **CRYSTAL, "Metal/oxide interfaces".** Stays 🟡, for a different reason. The
  row gated on "Phase 14f CRYSTAL ECP-block parser ships on `basissetdev`;
  flips to ✅ if/when `basissetdev` merges". That parser is in
  `vibeqc.basis_crystal` on `main`, and `basissetdev` was never transferred to
  this repository, so the condition could not have been met and did not need
  to be. What holds the row at 🟡 is the missing tutorial.
* **CRYSTAL, "4f-in-core ECPs".** Stays ✅. Its note called
  `auto_ecp_centers` "`basissetdev`-conditional"; it is a shipped top-level
  helper.

Elsewhere in this section, the stress, EOS and phonon rows in the ASE table
and the critical-path summary were corrected in the same pass: they said a
periodic stress was needed that has shipped as a finite difference for
releases. What they should have said, and now do, is that the FD stress ships
and carries EOS by volume scan, and that an *analytic* periodic stress is what
variable-cell relaxation and the elastic / piezo / photoelastic workflows are
actually waiting on.

**What the sweep did not do.** It did not re-run a tutorial. The cross-code
gap sweeps below carry their own dates and were not re-derived at all.

### Critical-path summary

The two features that historically unblocked the most CRYSTAL tutorials,
**Phase C1** (periodic SCF convergence tooling: level shift,
Fermi-Dirac smearing, second-order SCF) and **Phase G1** (periodic
analytic gradients), both shipped across v0.5.0-v0.15.0. The periodic
SCF front is now at production quality: every standard CRYSTAL tutorial
that needs an SCF energy or geometry optimization is reproducible.

What still gates the remainder: an **analytic** periodic stress for elastic
/ piezo / photoelastic workflows and for variable-cell relaxation (the FD
stress ships and carries EOS by volume scan, but no route accepts
`optimize_cell=True` outside full-k semiempirical); **phonon dispersion**,
since the GAPW phonon route is Gamma-point only and its q-path helper
returns the Gamma frequencies repeated rather than an interpolation; and the
native-BvK periodic local-correlation chain (the native BvK and periodic CC
milestones) for CRYSCOR parity. The megacell periodic-MP2->DLPNO-CCSD(T)
route (Paper II) is already shipped.

On the molecular side, **Phase 17** (analytic Hessian) shipped and
unlocks vibrational frequencies, thermochemistry, TS confirmation,
IR intensities. S1 solvation (CPCM) shipped in v0.12.0. D1/D2 dispersion
(D3-BJ, D4) shipped across v0.5.x-v0.8.0. The remaining gap is spectroscopy
(NMR / EPR / ECD / VCD / Raman) and the advanced charge-analysis tools
(Hirshfeld / NBO / ESP-fit charges). QTAIM + COOP/COHP + Mayer bond orders
shipped in v0.15.0.

The **properties suite** milestone closes the spectroscopy gap and
the CRYSTAL "Response Properties" line in one coherent pass on
top of a CPHF/CPKS substrate -- the highest-leverage late-stage
milestone on the road to v1.0.

The **CCM/AICCM** lines shipped as experimental in v0.15.0: union-and-weight
Γ-CCM with HF, DFT, MP2/UMP2, and CCSD(T), and χ-CCM with 3D SCF and
finite-torus correlation. The A namespace also ships neutral fitted-torus
UCCSD(T) and DLPNO controls without a Γ-CCM or χ-CCM construction identity.
The route-specific benchmark set spans 28 systems across all 7 crystal
systems. Production-grade
items (scalable dense-3-D, mixed-boundary Green's functions, analytic gradients,
projection-based embedding) are individually sequenced rather than gated as a
monolithic v2.x track.

### Tutorials we can write *today* with no new vibe-qc code

These are documentation-only items, every underlying capability
already ships. Status tracks what has landed in `docs/tutorial/`.

| # | Tutorial | Status |
| ---: | --- | --- |
| 1 | Hello water → `input-h2o-rhf.py` promoted into tutorial index | ✅ `molecular_hf.md` |
| 2 | Geometry optimization (H2O / dimer / trimer) | ✅ `geometry_optimization.md` |
| 3 | Running in parallel (`num_threads=`, timing block) | ✅ `parallel_execution.md` |
| 4 | Bond analysis + charges + dipole (Phase 18 + 19) | ✅ `post_scf_properties.md` |
| 5 | Vibrational frequencies, analytic CPHF/CPKS Hessian (Phase 17a-e), ASE-FD as fallback | ✅ `vibrational_frequencies.md` |
| 6 | Thermodynamics at T via `ase.thermochemistry` | ✅ `thermodynamics.md` |
| 7 | NEB-TS reaction path via `ase.neb` | ✅ `neb_reaction_path.md` (ammonia umbrella inversion at HF/STO-3G) |
| 8 | 3D Madelung constants via Ewald | ✅ `madelung_with_ewald.md` |
| 9 | Basis-set convergence sweep | ✅ `basis_convergence.md` |
| 10 | Solid-state hello world (1D molecular crystal) | ✅ `periodic_hf.md` + `periodic_dft.md` |
| 11 | Peierls dimerisation scan | ✅ `peierls.md` (also `input-h-chain-peierls.py` example) |
| 12 | pob-TZVP rationale (solid-state basis hygiene) | ✅ `pob_tzvp.md` |
| 13 | Orbital & density visualization (molden + cube) | ✅ `orbital_visualization.md` |
| 14 | Band structure + DOS (1D H-chain) | ✅ `band_structure.md` |
| 15 | DFT functional comparison | ✅ `functional_comparison.md` |
| 16 | Dispersion (D3-BJ via `dispersion="d3bj"`) | ✅ `dispersion.md` |
| 17 | Non-HF wavefunction solvers (Selected-CI, DMRG, v2RDM, transcorrelated) | ✅ `non_hf_solvers.md` |
| 18 | CASSCF / CASPT2 / NEVPT2 multireference methods | ✅ `casscf_multireference.md` |
| 19 | DLPNO-CCSD(T) local correlation | ✅ `dlpno_local_correlation.md` |
| 20 | EDIIS+DIIS hybrid SCF acceleration | ✅ `ediis_diis_hybrid.md` |
| 21 | Second-order SCF (SOSCF / TRAH) | ✅ `second_order_scf.md` |
| 22 | RIJCOSX chain-of-spheres exchange | ✅ `rijcosx_glycine.md` |
| 23 | TDDFT excited states (UV/Vis) | ✅ `excited_states_tddft.md` |
| 24 | Slab / adsorbate DFT | ✅ `slab_adsorbate_dft.md` |
| 25 | Surface embedding | ✅ `surface_embedding.md` |
| 26 | Periodic geometry optimization | ✅ `periodic_geometry_optimization.md` |
| 27 | Multi-k LiH solid-state walkthrough | ✅ `lih_multi_k.md` |
| 28 | Materials Project fetch + calculate | ✅ `mgo_from_materials_project.md` |
| 29 | Fermi-Dirac smearing | ✅ `fermi_dirac_smearing.md` |
| 30 | Gilat-net metallic BZ integration | ✅ `gilat_net_metals.md` |
| 31 | Double-hybrid (B2PLYP) | ✅ `double_hybrid_b2plyp.md` |
| 32 | Range-separated (ωB97X) | ✅ `range_separated_wb97x.md` |
| 33 | VV10 modern functionals | ✅ `vv10_modern_functionals.md` |
| 34 | Semiempirical methods (MSINDO / DFTB / PM6 / GFN2) | ✅ `msindo.md` + `semiempirical_dftb.md` + `pm6_and_gfn2.md` |
| 35 | MACE MLIP integration | ✅ `mace_mlip.md` |
| 36 | Solvation (CPCM water) | ✅ `solvation_water.md` |
| 37 | Cyclic cluster model foundations | ✅ (experimental) `cyclic_cluster_model.md` |
| 38 | AICCM-2026dev-B (localized-orbital CCM) | ✅ (experimental) `aiccm2026dev_b.md` |
| 39 | K-points + Brillouin-zone sampling | ✅ `kpoints_brillouin_bloch.md` |
| 40 | Relaxed PES scan | ✅ `relaxed_pes_scan.md` |

**"No new code" tutorial backlog cleared (40 of 40).** All 16
originally-scoped items plus the 24 tutorials that landed with
new features across the v0.8.x--v0.14.0 cycle are now in
`docs/tutorial/` with theory sections and references. Future
tutorial additions will track new vibe-qc features as they land
(periodic correlation, CPHF/CPKS response properties, phonons,
stress tensor, periodic multireference).

### Theory-layer roadmap

Each tutorial today teaches *how to run the calculation*. The next
pass adds a **theory layer**, the underlying equations, the key
canonical references, and where relevant the algorithmic improvements
that vibe-qc inherits. The goal is that a chemistry undergrad who
reads one tutorial end-to-end leaves knowing both "how to do it" and
"why this is the right procedure." Math is MathJax-rendered via
`dollarmath` (already enabled in `docs/conf.py`).

Target depth per tutorial is one focused theory section (≤ ½ page,
3-6 key equations) plus a `References` block. Not a derivation from
first principles, a working summary with pointers to the papers.

| Tutorial | Theory section content | Canonical refs | Status |
| --- | --- | --- | :---: |
| `molecular_hf` | Roothaan-Hall equations, Fock matrix, self-consistent field, DIIS extrapolation | Roothaan 1951, Hall 1951, Pulay 1980/1982 (DIIS) | ✅ |
| `molecular_dft` | Kohn-Sham mapping, exchange-correlation decomposition, Becke grid, LDA/GGA hierarchy | Kohn-Sham 1965, Becke 1988 (grid), Treutler-Ahlrichs 1995 (radial grid) | ✅ |
| `open_shell` | UHF vs RHF vs ROHF, spin contamination, ⟨S²⟩ diagnostic | Pople-Nesbet 1954, Pauncz 1967 | ✅ |
| `periodic_hf` | Bloch's theorem, real-space vs reciprocal-space representation, k-sampling in a Γ-only cell | Ashcroft-Mermin Ch. 8, Monkhorst-Pack 1976 | ✅ |
| `periodic_dft` | Same as 04, plus KS-DFT in a periodic basis; grid integration per unit cell | Pisani 1988 (CRYSTAL-style crystalline orbitals) | ✅ |
| `madelung_with_ewald` | Ewald decomposition (real + reciprocal), α-invariance, convergence-rate trade-off | Ewald 1921, Allen-Tildesley Ch. 5 | ✅ |
| `post_scf_properties` | Mulliken / Löwdin population analysis, Mayer bond order as a basis-set-consistent generalisation, dipole moment formula | Mulliken 1955, Löwdin 1950, Mayer 1983 | ✅ |
| `geometry_optimization` | Hellmann-Feynman theorem, BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton update, Wolfe conditions for line search | Hellmann 1937 / Feynman 1939, Broyden 1970 (or textbook Nocedal-Wright) | ✅ |
| `vibrational_frequencies` | Normal-mode analysis, mass-weighted Hessian, analytic CPHF/CPKS construction, IR intensities via dipole-derivative tensor (double-harmonic), FD fallback, harmonic-to-IR scaling factors | Wilson 1955 (normal modes), Pople 1979 (analytic 2nd derivatives + CPHF), Wong 1996 (HF scaling factor 0.89) | ✅ |
| `thermodynamics` | Ideal-gas partition function factorisation (trans × rot × vib × elec), Gibbs free energy, entropy decomposition | Atkins-de-Paula Ch. 13, McQuarrie Ch. 8 | ✅ |
| `orbital_visualization` | MO isosurface interpretation, sign arbitrariness, Gaussian-cube file format | Gaussian cube format spec (public) | ✅ |
| `band_structure` | Bloch sum, dispersion relation E(k), density of states histogram, Fermi level, gap closure | Ashcroft-Mermin Ch. 8, 9 | ✅ |
| `basis_convergence` | Contracted Gaussian expansion, correlation-consistent hierarchy cc-pVXZ, Helgaker CBS extrapolation exp(-αX) | Dunning 1989, Helgaker-Klopper-Koch-Noga 1997 | ✅ |
| `dispersion` | Grimme D3 pairwise expression with BJ damping, C₆ coefficients, damping-parameter fit | Grimme 2010 (D3), Grimme 2011 (BJ damping), Becke-Johnson 2006 | ✅ |
| `functional_comparison` | Jacob's ladder hierarchy, self-interaction error, B3LYP / PBE0 hybrid construction | Perdew 2001 (Jacob's ladder), Becke 1993 (B3LYP), Adamo-Barone 1999 (PBE0) | ✅ |

**Progress (15 of 15).** All tutorials now carry a Theory + References
section matching the template from the `dispersion` tutorial. Open follow-up: a
round of reference spot-checks (hunt down every `[verify]` marker,
confirm or correct the journal-volume-page triples) and maintenance
passes as new features ship (e.g., analytic Hessian will refresh
the `vibrational_frequencies` tutorial, ROHF will refresh the `open_shell` tutorial).

**Working convention for citations.** Foundation-era papers (Roothaan,
Kohn-Sham, Pulay, Becke, etc.) I cite directly, those are
stable and well known. Anything post-2010 I'll mark with
`[Ref: verify]` in draft; the user confirms before publication. This
catches my weakest failure mode (hallucinated year/volume/page)
without slowing the foundation content down.

**Order of attack.** Start with the most-visited tutorials (01, 02,
08, 14) where theory depth pays back most immediately for new users.
Defer 10-12 until the core set lands.

That's 12 tutorial theory sections + 4 remaining "no-code" tutorials
= the documentation backlog before any new C++ feature work is
required.

### Cross-code feature gaps (ORCA 6.1.1 + CRYSTAL23 manual sweep)

A May 2026 inventory pass against the ORCA 6 (1655 pp) and
CRYSTAL23 (524 pp) manuals surfaced features neither already
shipped in vibe-qc nor explicitly enumerated under one of the
milestones above. Grouped into thematic buckets so future
milestones can absorb whole buckets at once instead of one-off
keywords. Items that ARE already on the roadmap (DLPNO-CCSD(T),
NMR/EPR/Mössbauer, CASSCF/NEVPT2, periodic phonons, slab builder,
elastic / piezoelectric / photoelastic tensors, periodic Berry
charges, periodic MP2, TDDFT, implicit solvation, D4) are
omitted from this list to keep the focus on genuinely-new gaps.

**Buckets at a glance** (~165 items total across two sweeps;
ID-prefix in **bold**; see body for items):

| First sweep (~70 items) | Second sweep (~95 items) |
|---|---|
| **3c-** Composite-3c methods family | **CMP-** ORCA Compound scripting (workflow lang) |
| **DIAG-** SCF diagnostics + provenance | **AUTOCI-** AutoCI + advanced correlation knobs |
| **B1.5-12** Geometry editing + nanostructure builders | **IO-** Restart, interop, on-disk formats |
| **SS-** Solid-solution enumeration | **MD-** / **ESD-** MD + excited-state dynamics |
| **EOS-** / **QHA-** Equation-of-state + quasi-harmonic | **TOOL-** orca_* utility binaries |
| **VIB-** Beyond-harmonic vibrations | **SPEC7-12** Spectroscopy depth (extra) |
| **BOND-** / **XRD-** / **EMD-** Bond-order, X-ray, EMD | **PLT-** Density / property cube exports |
| **MR-** / **EX-** Multireference + excited-state extras | **CGEO-** CRYSTAL geometry builders (extra) |
| **EMB-** Embedded-cluster + scalable QM/MM | **CBAS-** CRYSTAL basis-set library |
| **S5-8** Solvation extras | **CALC-** CRYSTAL calculation-type + SCF knobs |
| **SPEC1-6** Spectroscopy beyond what's planned | **PROP-** CRYSTAL properties keywords |
| **BSSE-** Boys-Bernardi counterpoise + BSSE | **CTOOL-** CRYSTAL post-processing utilities |
| **CHG-** Hirshfeld + ESP-fit charge variants | **PAR-** CRYSTAL job-control + parallelism |
| **POL-** Spontaneous polarisation + Berry phase | **CECO-** CRYSTAL ecosystem tools |
| **BAS-** Basis-set automation | |
| **CORR-** A-posteriori correlation correction | |

The first sweep is the **chemistry surface** (methods, properties,
builders); the second sweep is the **infrastructure surface**
(workflow languages, file-format interop, restart protocols,
parallel-execution knobs, post-processing utilities). Most
infrastructure items are small individually; collectively they're
the post-1.0 maturity story.

#### Composite-3c methods family (extends Phase D + DFT stack)

Direct ORCA / CRYSTAL composites that sit on top of dispersion +
gCP + a tuned DFT functional. Each pair (`HF-3c`, `r2SCAN-3c`,
`PBEh-3c`, `B97-3c`, `HSE-3c`) is a fixed recipe, a small wrapper
sets the basis + functional + dispersion + counterpoise and prints
"3c" in the .out for the user. Modest C++ work.

* **3c-1** Wrapper functions `vq.run_hf_3c`, `run_r2scan_3c`,
  `run_pbeh_3c`, `run_b97_3c`, `run_hse_3c` (last two: solid-state
  hybrid composites already in the pob-TZVP family).
* **3c-2** Solid-state revisions: `HFsol-3c`, `PBEsol0-3c`,
  `HSEsol-3c` (distinct from the molecular composites; the
  parametrisation is for periodic).
* **3c-3** ✅ *shipped*: `ABC` (D3 three-body Axilrod-Teller-Muto) as
  the `s9` field on `D3BJParams`, honoured by both the builtin C++ and
  the `dftd3` backends. `pbeh-3c` / `hse-3c` set `s9 = 1.0`; plain
  `dispersion="d3bj"` stays two-body. Energy + analytic gradient.
* **3c-4** `gCP` geometrical-counterpoise correction as a public
  function (component of the 3c stack but useful standalone).

#### SCF diagnostics + provenance (extends v0.5.x observability)

Cheap, high-leverage diagnostics ORCA/CRYSTAL surface that
vibe-qc doesn't expose yet:

* **DIAG1** `vq.scf_stability(result)`, real / complex / spin
  stability analysis (orbital-Hessian eigenvalues; flags
  triplet instability for closed-shell SCFs that should have
  been UHF).
* **DIAG2** Spin-contamination diagnostic for UHF/UKS results
  (`<Ŝ²>` deviation from `S(S+1)`; CRYSTAL `SPINCNTM`).
* **DIAG3** Broken-symmetry workflow scaffold: `BROKENSYM` ORCA-
  style, with explicit per-pair antiferromagnetic coupling J via
  Yamaguchi's formula.
* **DIAG4** ORCA `Compound` script equivalent, chained-job runner
  for canonical workflows (CBS extrapolation, BSSE-corrected
  binding curves, DLPNO+composite stacks).

#### Geometry editing + nanostructure builders (extends Phase B1)

CRYSTAL23 has an unusually rich geometry-editing toolkit. Each
function is a few-line numpy operation but together they
streamline solid-state model building:

* **B1.5** `vq.cluster_carve(system, radius_bohr,
  saturate_dangling="H")`, finite cluster carved from a
  periodic cell, optionally H-saturated (CRYSTAL `CLUSTER` /
  `HYDROSUB`).
* **B1.6** `vq.nanotube_from_slab(slab, n, m)`, single-wall
  nanotube builder; `vq.multiwall_nanotube` for concentric
  variants. CRYSTAL `NANOTUBE` / `SWCNT` / `NANOMULTI`.
* **B1.7** `vq.nanorod_from_crystal(...)`, `nanocrystal_from_crystal(...)`
 - 1D / 0D nanostructure builders. CRYSTAL `NANOROD` / `NANOCRYSTAL`.
* **B1.8** `vq.fullerene_from_hexagonal_slab(...)`, fullerene
  builder. CRYSTAL `FULLE`.
* **B1.9** `vq.wulff_polyhedron(surface_energies)`, Gibbs-Wulff
  equilibrium-shape construction. CRYSTAL `WULFF`.
* **B1.10** Atomic-edit primitives: `substitute`, `insert`,
  `remove`, `displace`, `rotate` on `PeriodicSystem` (CRYSTAL
  `ATOMSUBS` / `ATOMINSE` / `ATOMREMO` / `ATOMDISP` / `ATOMROT`).
* **B1.11** `vq.embed_point_charges(system, charges)`,
  external point charge array (CRYSTAL `POINTCHG`).
* **B1.12** `vq.apply_sawtooth_field(system, field_au, axis)`,
  external electric field along a periodic axis (CRYSTAL
  `FIELD` / `FIELDCON`).

#### Solid-solution enumeration

* **SS1** `vq.enumerate_configurations(system,
  substitutions, symprec=...)`, list symmetry-inequivalent
  configurations (CRYSTAL `CONFCNT`).
* **SS2** `vq.sample_configurations(system, n_samples)`,
  Monte-Carlo sampling for large composition spaces
  (CRYSTAL `CONFRAND`).
* **SS3** `vq.run_configurations(configs, calculator)`, batch
  driver across the enumerated set (CRYSTAL `RUNCONFS`).

#### Equation-of-state automation

EOS never needed the stress: it is "fit a curve to E(V)", and the fit half
shipped as `vibeqc.eos.fit_birch_murnaghan`. What is left here is the model
coverage and the scan driver.

* **EOS1** `vq.fit_equation_of_state(volumes, energies,
  model="birch-murnaghan")` - accepts `vinet`, `birch-murnaghan`,
  `murnaghan`, `poirier-tarantola`, `polynomial`. Returns
  `EOSFit` with `V0`, `B0`, `B0p`, `E0` + reduced-χ². Mirrors
  CRYSTAL `EOS`. **Partly shipped:** `vibeqc.eos.fit_birch_murnaghan`
  returns `v0`, `b0`, `b0_prime` and `e0` for the third-order
  Birch-Murnaghan model with ill-conditioned-fit guards. The other four
  models, the reduced-chi-squared field and the `EOSFit` shape are open.
* **EOS2** `vq.run_eos_scan(system, calculator,
  volume_range_pct=(-10, +10), n_points=11)`, automated
  workflow: scale lattice, relax internal coords at each volume
  (needs G1), fit. Output a dataclass with the fit + raw points.
* **QHA1** `vq.run_qha(...)`, quasi-harmonic approximation
  workflow chaining EOS + phonons. Already mentioned as
  v0.7-ish; this entry pins the API shape.
* **QHA2** Quasi-harmonic thermo-elasticity (T-dependent Cij).

#### Beyond-harmonic vibrations (extends Phase 21)

* **VIB1** `vq.compute_vpt2(hess)`, vibrational perturbation
  theory anharmonic corrections. ORCA `VPT2` / `GVPT2`.
* **VIB2** `vq.compute_anharm_xh_stretch(...)`, 1D anharmonic
  treatment of a selected X-H stretch (CRYSTAL `ANHARM`).
* **VIB3** `vq.compute_vscf(...)` and `vq.compute_vci(...)`,
  vibrational SCF / CI on a 4th-order PES. CRYSTAL `VSCF` / `VCI`.
* **VIB4** `vq.compute_phonon_dos(..., neutron_weighted=True)`,
  neutron-weighted vibrational DOS for inelastic-neutron-scattering
  comparison. CRYSTAL phonon DOS option.
* **VIB5** `vq.compute_adp(hess, T_kelvin)`, anisotropic atomic
  displacement parameters (Debye-Waller) at temperature.
  CRYSTAL `ADP`.

#### Bond-order / topology / X-ray analyses

* **BOND1** ✅ Crystal Orbital Overlap Population (`vq.coop`),
  bond-strength analysis for periodic systems. CRYSTAL `COOP`.
  Shipped 2026-06-23 in `python/vibeqc/coop_cohp.py`.
* **BOND2** ✅ Crystal Orbital Hamiltonian Population (`vq.cohp`)
 - energy-resolved bonding analysis. CRYSTAL `COHP` (also
  popular via the LOBSTER post-processor on plane-wave codes).
  Shipped 2026-06-23 alongside BOND1; spin-polarized support
  included.
* **XRD1** `vq.compute_xray_structure_factors(system, density)`
 - F(hkl) for X-ray diffraction; CRYSTAL `XFAC`.
* **XRD2** Compton profile + reciprocal form factors (CRYSTAL
  `BIDIERD` / `PROF`).
* **EMD1** Electron-momentum-density maps (`vq.compute_emd_line`,
  `compute_emd_plane`); from the density matrix or per-Wannier-
  function. CRYSTAL `EMDL` / `EMDP`.

#### Multireference + excited-state extras (extends the MR + excited-state milestone)

* **MR1** ICE-CI (iterative configuration expansion), approximate
  FCI scaling to ~30 orbitals (ORCA `ICE-CI`).
* **MR2** DMRG interface, for FCI in larger active spaces.
  Likely an interface to BLOCK / CheMPS2 rather than a native
  implementation.
* **MR3** Multi-reference EOM-CC (`MR-EOM-CC`).
* **MR4** Multiconfigurational RPA (`MCRPA`), extends MCSCF to
  excited states without explicit state-averaging.
* **MR5** ΔSCF for excited states by orbital occupation
  (ORCA `DELTASCF`), much cheaper than TDDFT for the lowest
  few states of a single character.
* **EX1** Minimum-energy crossing point (`MECP`) optimization
  between two states (e.g. singlet/triplet, two TDDFT roots).
* **EX2** Conical intersection optimization (CI-search).

#### Periodic multireference via embedded clusters (OpenMolcas parity)

A true k-space CASSCF/CASPT2 analogous to periodic HF or KS-DFT does
not exist. CASSCF needs a finite active space of localized, chemically
interpretable orbitals with an FCI expansion built over them, which is
structurally incompatible with delocalized Bloch functions; and the
FCI step's exponential scaling already caps molecular CASSCF near
CAS(20,20). Only single-reference wavefunction methods (periodic MP2,
CCSD) have been generalized to full periodic boundary conditions so
far. The established route to multireference physics in a solid is
therefore the **embedded cluster**: carve a finite quantum region out
of the crystal, treat it with the CASSCF / CASPT2 / NEVPT2 solvers the
shipped molecular CAS family already provides, and represent the
surrounding crystal by an embedding potential. This is the path to
parity with OpenMolcas, the de-facto standard for embedded-cluster
CASSCF/CASPT2 on ionic and covalent solids. It is well suited to local
physics: ligand-field and crystal-field states, defect and color
centers, dopant spectra, charge-transfer excitations.

The standard construction is three nested regions:

1. a fully quantum-mechanical inner cluster (the CAS region);
2. an **ab initio model potential (AIMP)** shell on the frontier ions
   that reproduces exchange repulsion and orthogonality at the
   boundary (the **Fragment AIMP / FAIMP** variant carries the full
   embedding-ion density for covalently bonded hosts, where spherical
   AIMPs fail);
3. a point-charge array reproducing the long-range Madelung potential.

Milestones (extend the shipped molecular CAS family; reuse the
embedded-cluster infrastructure in EMB1-EMB4 and the `cluster_carve`
helper in B1.5):

* **MR6** Madelung point-charge embedding: fit a finite point-charge
  array (Evjen / Ewald-matched) that reproduces the crystal Madelung
  potential across the cluster, driven from a periodic SCF density.
* **MR7** AIMP boundary shell: ab initio model potentials on the
  frontier ions for exchange and orthogonality, plus the FAIMP
  full-density variant for covalent hosts (Swerts 2008). A reusable
  AIMP/FAIMP library for common lattices is the bulk of the work.
* **MR8** `vq.embed_cluster(system, center, radius, ...)` driver that
  composes carve + Madelung array + AIMP shell into an embedded
  cluster and hands the finite CAS region to the molecular
  CASSCF/CASPT2/NEVPT2 solvers unchanged.
* **MR9** Cluster-size and AIMP-shell convergence protocol (Larsson
  2022): the systematic study that defines "converged" for an
  embedded-cluster result, mirroring the CCM cluster-size convergence
  suite (v2.0).
* **MR10** Validation out-of-process against OpenMolcas (the parity
  target; large AIMP-library precedent) on published dopant spectra
  and defect multiplets, and against periodic DFT where a direct
  comparison exists (Fasulo 2023, Benedek 2025).

**Plane-wave / Bloch to active-space bridge.** When the periodic mean
field is a plane-wave calculation rather than vibe-qc's Gaussian
periodic SCF, the active space is built by localizing the relevant
bands into Wannier functions (the periodic analogue of localized MOs):
maximally localized Wannier functions (Marzari-Vanderbilt 1997;
entangled-band disentanglement Souza 2001; review Marzari 2012), the
Bloch intrinsic-atomic-orbital variant that better preserves sigma/pi
character for a cleaner active-space guess (Zhu-Tew 2024), or the
LOBSTER embedded-LMO route that sidesteps the Wannier gauge problem
(Muller 2023). vibe-qc's own Gaussian periodic SCF (the v0.5-v0.7 GDF
stack) together with the v2.3 CCM localization machinery is the
in-house analogue of this bridge, so for vibe-qc this is mainly an
interoperability concern.

**Outlook.** A genuinely periodic CASSCF remains the open problem; the
actively pursued near-term routes are density matrix embedding theory
(DMET) with a CASSCF/NEVPT2 impurity solver (detailed in its own
subsection below), and more speculative compact-manifold
reformulations. For vibe-qc the embedded-cluster route is the pragmatic
target now, and CAS within the cyclic cluster model is the long-term
in-house meeting point of this work with the v2.x CCM track (see "CAS
within CCM" there).

#### Density matrix embedding (DMET), the entanglement-bath route to periodic MR

The second practical route to multireference physics in a solid, and
mathematically distinct from the AIMP/FAIMP embedded cluster above:
instead of an electrostatic + projection potential, DMET represents the
environment through a bath of *entangled* orbitals built from a Schmidt
decomposition of the mean-field density matrix. The impurity subspace
(fragment + bath) is then solved with a high-level, and possibly
multireference, solver. Currently **absent from vibe-qc**; this is the
roadmap entry.

* **Periodic ab initio DMET**, Cui, Zhu & Chan (2020, *JCTC*):
  realistic Gaussian-basis periodic DMET with up to ~300 embedding
  orbitals; hBN, silicon, NiO. The periodic-DMET parity target.
* **CASSCF/NEVPT2-DMET for defect excited states**, Mitra, Pham,
  Hermes & Gagliardi (2022, *JPCL*): O-vacancy in MgO and Si-vacancy in
  diamond, within ~0.05 eV of non-embedded CASSCF/NEVPT2.
* **MC-PDFT-DMET**, Mitra, Hermes & Gagliardi (2023, *JCTC*):
  multiconfiguration pair-density functional theory as a cheaper
  impurity solver than NEVPT2 inside DMET (O-vacancy MgO).
* **Optical properties of defects via quantum embedding**, Berkelbach
  group (2024, *J. Phys. Chem. C*): active-space selection + DMET-style
  localisation feeding EOM-CCSD for solid defects.

Sequenced molecular-first (v0.14.0 rule): the impurity solvers DMET
needs, CASSCF, NEVPT2, FCI/CASCI, are ✅ already shipped molecularly;
MC-PDFT is a citable prerequisite sub-item (a cheaper impurity solver).
The missing piece is the **DMET framework itself** (Schmidt-bath
construction, correlation-potential fitting, fragment assembly): build
and validate a **molecular DMET** against PySCF's DMET first, *then* the
**periodic DMET** extension (Cui-Zhu-Chan 2020). Slots post-1.0, reusing
the embedded-cluster (MR6-MR10) and v2.x CCM infrastructure.

#### Local / DLPNO multireference, the molecular bridge to periodic

No DLPNO-CASSCF or DLPNO-NEVPT2 for periodic systems exists *anywhere*
yet. The molecular local-MR methods are mature, and they are the
bridge: they are how the embedded-cluster and DMET routes scale, and,
joined to periodic DLPNO-MP2 (the native BvK milestone), they are the visible path to a
future periodic DLPNO-NEVPT2. These are **molecular** targets extending
the v0.15.0 DLPNO-CCSD(T) milestone into the multireference regime:

* **DLPNO-NEVPT2**, Guo, Sivalingam, Valeev & Neese (2016, *JCP*):
  near-linear-scaling NEVPT2 on the SparseMaps-III framework,
  demonstrated past 5,400 basis functions. (The SparseMaps reference it
  builds on, `riplinger_sparsemaps_2016`, is already in the citation
  database for DLPNO-CCSD(T).)
* **DLPNO-NEVPT2-F12**, Guo, Pavošević, Sivalingam, Becker, Valeev &
  Neese (2023, *JCP*): explicit correlation for faster basis-set
  convergence (rhodopsin-scale excited states).
* **PNO-CASPT2 / PNO-MS-CASPT2**, Kats & Werner (Molpro):
  linear-scaling (multistate / extended-multistate) CASPT2 on the local
  integrated tensor framework (LITF).
* **Local NEVPT2 via LVMOs**, Uemura, Saitow, Ishimaru & Yanai (2023,
  *JCP*): orthonormal localised virtual MOs instead of PAOs; stable on
  large Co-containing biological systems.
* **CASSCF/DLPNO-NEVPT2 as a black box** (ORCA, Neese group), routine
  for single-molecule magnets and transition metals. Practical caveat
  to carry into our own implementation: CASSCF orbitals can break
  CC/PNO convergence, so **ROHF / ROKS is the safer reference** in many
  cases (relevant to the v1.0.0 ROHF/ROKS item).

The molecular halves, CASSCF/NEVPT2/CASPT2 ✅ and the DLPNO machinery
🔨 (`python/vibeqc/dlpno/`), already exist; combining them yields
molecular DLPNO-NEVPT2 / PNO-CASPT2. The bottleneck for the *periodic*
version is generating PNOs from a CASSCF reference in the periodic
setting, which needs either a periodic CASSCF (nonexistent) or a
Wannier-based active space (see the plane-wave/Bloch bridge above),
which is exactly why DMET and the embedded cluster are the practical
periodic-MR proxies today.

**Active groups + parity oracles (literature compass).** Where this
state of the art is being made, for parity targets and paper tracking:
**Tew** (Oxford, Turbomole periodic DLPNO-MP2); **Berkelbach**
(Columbia, PySCF periodic CC, periodic LNO, quantum-embedding optics);
**Grüneis** (Vienna, plane-wave CC at the thermodynamic limit);
**Gagliardi** (Chicago, DMET / MC-PDFT in PySCF); **Neese** (MPI
Mülheim, DLPNO-NEVPT2 in ORCA); **Veryazov** (Lund, OpenMolcas
embedded-cluster AIMP/FAIMP). Where CASPT2 is insufficient,
**compound-tunable embedding potentials** (Oleynichenko et al. 2024,
*Phys. Rev. B*) bring relativistic-coupled-cluster quality to f-element
dopants in solid hosts. Per §10 each of these is an *out-of-process
parity oracle*, never a runtime dependency: vibe-qc implements every
method itself and validates against the parsed output.

#### Embedded-cluster + scalable QM/MM (extends post-1.0 QM/MM)

The CCM track (v2.x) overlaps with this, items here are the
*infrastructure* it sits on:

* **EMB1** ORCA-style `MOL-CRYSTAL-QMMM` / `IONIC-CRYSTAL-QMMM`,
  embedded-cluster periodic property calculations on a QM
  region inside an MM crystal.
* **EMB2** Fast-multipole-method QM/MM (`FMM-QMMM`) for >10⁴
  point charges, needed for any realistic MM environment.
* **EMB3** Auto link-atom + boundary-treatment selection
  (mechanical / electrostatic / IOD / Z1/Z2/Z3 schemes).
* **EMB4** Partial Hessian Vibrational Analysis (`PHVA`) for
  the active region, frequencies from a sub-block of the
  full Hessian. Pairs naturally with QM/MM.

#### Solvation extras (extends Phase S)

* **S5** `DRACO`, dynamic radii adjustment for continuum
  solvation, gives noticeably better ΔG_solv than fixed radii.
* **S6** openCOSMO-RS interface, solvent screening and fluid-
  thermodynamics post-processing on top of the COSMO/CPCM
  output.
* **S7** Explicit-solvent shell builder (`SOLVATOR`-style),
  out of v0.x core scope; could be a thin wrapper over
  `packmol` or similar.
* **S8** Docking driver (`DOCKER`-style), out of v0.x core
  scope; would interface to `vina` / equivalent.

#### Spectroscopy beyond what's already planned

* **SPEC1** Resonance Raman simulation (analogous to ORCA
  `orca_asa`), vibronic structure with a fixed reference
  excited state.
* **SPEC2** ROCIS / ROCIS-DFT for X-ray and UV-Vis of
  transition-metal complexes.
* **SPEC3** STEOM-CCSD and DLPNO-STEOM-CCSD for excited states
  (similarity-transformed EOM family). Pairs with the MR +
  excited-state milestone.
* **SPEC4** MCD (magnetic CD) on top of TDDFT (extends PR29 ECD
  in the properties suite).
* **SPEC5** LO/TO splitting in IR for polar crystals, extends
  Phase 21 phonons.
* **SPEC6** SOMF (spin-orbit mean-field) operator for property
  evaluation, cheaper than full 2-component SCF for SOC matrix
  elements. Useful for EPR g-tensors of heavy elements.

#### Boys-Bernardi counterpoise + BSSE infrastructure

* **BSSE1** `vq.compute_bsse(system_a, system_b, calculator)`
 - Boys-Bernardi counterpoise for intermolecular interactions.
* **BSSE2** `MOLEBSSE` / `ATOMBSSE` for molecular crystals
  (CRYSTAL): counterpoise corrections for molecules-in-crystal
  packing energies.

#### Hirshfeld variant + ESP-fit charge extras (extends PR-suite)

* **CHG1** MBIS (minimal-basis iterative stockholder) charges,
  a Hirshfeld variant with often-better atoms-in-molecules
  fidelity. Extends the shipped molecular Hirshfeld charges and PR1.
* **CHG2** Updated CHELPG / RESP infrastructure: extend the
  existing molecular dipole/ESP machinery to fit per-atom
  point charges to the ESP on a vdW grid.

#### Spontaneous polarization + Berry phase

* **POL1** `SPOLBP`, spontaneous polarization via Berry phase
  (King-Smith / Vanderbilt) on the periodic SCF.
* **POL2** `SPOLWF`, spontaneous polarization via localised
  Wannier orbitals, pairs with the LOCALWF Wannier-function
  story already on the post-1.0 list.

#### Basis-set automation

* **BAS1** `BASOPT` equivalent, internal basis-set optimizer
  for unusual element / pseudopotential combinations.
* **BAS2** `AUTOAUX` equivalent, automatic auxiliary-basis
  generation for RI-J / RI-K once those land.

#### A-posteriori correlation correction

* **CORR1** `EDFT` / `ADFT`, apply a DFT correlation functional
  on top of an HF density without re-converging. Cheap "DFT
  correlation correction" workflow CRYSTAL exposes.

The total here is roughly 70 distinct items spanning ~10 themes.
Most slot naturally into existing milestones (3c stack into
v0.5.x or v0.6.x; geometry editing into the B1 surface-builder
phase; EOS into the v0.7 G2 follow-up; bond-order analyses into
the properties suite). The few that need their own
milestone, embedded cluster QM/MM, multireference extras, fit
the post-1.0 stack already in the roadmap.

### Cross-code feature gaps, second-pass deep dive

The first sweep above caught the user-facing chemistry: methods,
properties, geometry builders. A second deeper read of the same
two manuals (May 2026, focusing on chapters skimmed the first
time, Compound scripting, AutoCI, the orca_* utility binaries,
the property exports, the CRYSTAL job-control + tools layer)
surfaced ~95 additional items, mostly *infrastructure*: workflow
languages, file-format interop, restart protocols, parallel-
execution knobs, post-processing utilities. None of it is bleeding-
edge chemistry, it's the scaffolding that turns vibe-qc from "a
correct code" into "a code engineering teams can build on." Most
items are small individually; collectively they're the post-1.0
maturity story.

#### ORCA Compound scripting (workflow language)

ORCA's `%Compound` block is a Turing-complete mini-language for
chaining calculations: variables, control flow, geometry
manipulation, file I/O, system calls. Vibe-qc's equivalent today
is "write a Python script", which is fine, but Compound is
deliberately persistable and self-contained. A small subset of
this surface is high-leverage; the full surface is post-1.0.

* **CMP1** `vq.workflow.Compound`, declarative chained-job
  runner with `New_Step` (sub-job), `Variable` (typed value),
  `For` / `If` / `Break` / `Continue` (control flow), `&{var}`
  string interpolation. Python class with a fluent API, not a
  parser, the goal is the workflow, not the syntax.
* **CMP2** `Read` primitive, pull values from any prior step's
  output (energy, gradient, dipole, frequency, occupation,
  arbitrary scalar) by structured key.
* **CMP3** `Geometry` object methods, `add_atom`, `remove_atom`,
  `bond_length`, `bond_angle`, `dihedral`, `to_zmat`, `to_xyz`,
  `apply_displacement`, `superimpose`. Mirrors ORCA's geometry-
  manipulation calls inside Compound.
* **CMP4** `Dataset` primitive, multi-row table with column-
  typed cells, the natural object for scan / fit workflows
  (CBS extrapolation, BSSE binding curves, EOS).
* **CMP5** Linear-algebra primitives in workflow scope,
  `dot`, `norm`, `det`, `inv`, `eig`, `solve` on small matrices.
  Lets a Compound script do Yamaguchi-style spin coupling
  arithmetic without dropping to Python.
* **CMP6** `Timer` + file primitives (`file_exists`, `file_read`,
  `file_write`, `file_delete`, `file_append`), workflow-scope
  filesystem access. Needed for restart logic.
* **CMP7** `Alias` + `Basenames[i]`, name reuse and step-
  indexed file naming. Cheap; pure ergonomics.
* **CMP8** `GoTo` / `Abort` / `EndRun` / `sys_cmd`, control-
  flow escape hatches. `sys_cmd` is the "shell out" backdoor.
* **CMP9** GOAT (global optimization) hooks, make the
  Compound runner a first-class consumer of the v0.7 GOAT
  conformer-search output: `For (conformer in goat_results)`.
* **CMP10** Bundled examples gallery, `helloWorld`,
  `ccCA_CBS_2`, `scan_1D`, `BSSEOptimization`, `binding_curve`.
  Each example doubles as a regression test for the workflow
  runtime.
* **CMP11** Property-table coverage, every property the SCF +
  post-SCF stack writes should be Compound-readable by
  structured key. Audit what's exposed today vs what ORCA's
  `%output` block surfaces.

#### AutoCI + advanced correlation knobs

ORCA's AutoCI module exposes CC-family methods with an unusually
rich knob surface. v0.5.x ships single-reference CCSD(T); the
items here flesh out the AutoCI-equivalent control surface for
the multireference + excited-state milestone and the post-1.0 correlation push.

* **AUTOCI1** `citype` selector, `CISD`, `CCSD`, `CCSD(T)`,
  `CC2`, `CC3`, `CCSDT`, `LCCSD`, `LCCD`, `BCCD` available
  through one entry point with a string flag.
* **AUTOCI2** Density variants, `unrelaxed` vs `Z-vector
  relaxed` densities for properties (gradients require the
  relaxed variant; one-electron expectation values often only
  need unrelaxed).
* **AUTOCI3** Convergence options, DIIS / Jacobi diagonal-
  shifted iteration, residual norm + amplitude norm both
  monitored, max_iter + level shift exposed in options struct.
* **AUTOCI4** Analytical gradients for CC family, currently
  CCSD only on the roadmap; AutoCI has CCSD(T), CC2, CC3
  gradients via response theory.
* **AUTOCI5** Linear response on top of CC reference, TD-CC
  excitation energies, CC2 / CC3 oscillator strengths.
* **AUTOCI6** RDM2 export, two-particle density matrix to
  HDF5 / JSON / FCIDUMP for downstream analysis (NOFs,
  decomposition into two-electron contributions).
* **AUTOCI7** `%moinp` restart, start CC from a converged
  prior wavefunction, refine with tighter convergence or
  larger basis. Couples to the orbital-restart story below.

#### Restart, interop, and on-disk format protocol

vibe-qc's restart story today is `BasisSet.checkpoint` + `MOSet`
serialization. ORCA's interop surface is broader: `orca_2json`,
multiple binary encodings, ASCII fallback, and a JSON config
schema readable by any tooling. This bucket is the v0.7+
"play well with the ecosystem" story.

* **IO1** `vq.export_calc(result, format="json")`, one-line
  export of a converged calculation to JSON. Mirrors
  `orca_2json -property` (properties only) and `-gbw` (full
  wavefunction). Default format JSON, optional BSON / UBJSON /
  MessagePack via flag for byte efficiency on large
  wavefunctions.
* **IO2** `vq.import_calc(path)`, round-trip importer. Schema-
  versioned; rejects mismatched majors with a clear error.
* **IO3** Public JSON-schema for the export, published in
  `docs/file_formats.md` and pinned to a major version. Other
  tools (analysis scripts, ML models, web frontends) consume
  this contract.
* **IO4** ASCII-fallback `dump_text(result, path)`, human-
  readable plaintext form for environments that can't ingest
  binary (containers, locked-down clusters).
* **IO5** Integral exports, `vq.export_integrals(basis,
  kind="ao_2e")` writing AO 2e integrals to FCIDUMP /
  HDF5. Same for `kind="mo_2e"`, `kind="ri_3c"`. Needed for
  external CI / DMRG / ML-correlation post-processors.
* **IO6** Fock-matrix decomposition export, per-contribution
  Fock pieces (J, K, XC, ECP, dispersion) saved separately so
  downstream tools can re-weight or analyse contributions.
* **IO7** Orbital-window export (`OrbWin`), write a
  contiguous-or-explicit MO range to disk, the natural shape
  for active-space pre-selection workflows.
* **IO8** `AuxBasisType` exposed in exports, record which
  auxiliary basis (RI-J / RI-K / RIJCOSX) was used so a
  restart picks the same one.
* **IO9** TDDFT data export, Casida `(X+Y)`, `(X-Y)`
  vectors, transition densities, NTOs, with the same
  schema-versioned shape as the SCF export.
* **IO10** OPI Python interface equivalent, a documented
  Python wrapper around the on-disk format, so external
  consumers don't reinvent the parser. `vq.io.OrcaCompat`
  module providing read-only adapters.
* **IO11** ASE `read("file.json", format="vibeqc")` round-
  trip, extends ASE's existing per-code parsers. Pairs with
  the existing PySCFCalculator / vibeqc ASE bridge.

#### Molecular dynamics + ESD (excited-state dynamics)

ORCA's MD chapter ships a SANscript-like input language for
classical-and-mixed dynamics (BOMD, AIMD, MTD, restart, region-
based thermostats). vibe-qc has `examples/ase_compare/` showing
ASE-driven MD; native MD comes after the production MPI milestone, but the API shape
matters now so future engineering doesn't reinvent it.

* **MD1** `vq.MDDriver`, top-level Python class with
  `Run(steps)`, `Restart(path)`, `Initvel(temp)`,
  `Thermostat(kind, params)`, `Timestep(fs)`, `Cell(npt|nvt|nve)`.
  Mirrors ORCA's MD command surface in Python idioms.
* **MD2** Region API, `MDRegion(atoms, thermostat=...,
  constraint=...)` for per-region thermostats / constraints.
  Needed for QM/MM and surface MD where bulk and slab want
  different couplings.
* **MD3** `Constraint`, bond / angle / dihedral / SHAKE /
  RATTLE constraints applied during integration.
* **MD4** `Dump`, trajectory write at configurable cadence,
  including selective per-region dumps and per-property
  dumps (forces, dipole, charges).
* **MD5** Metadynamics, `Manage_Colvar` + `Metadynamics`
  primitives; well-tempered MTD as a flag. Pairs with the
  free-energy methods on the post-1.0 list.
* **MD6** `Restart` protocol, checkpoint that captures
  positions + velocities + thermostat state + colvar history
  in one file. Pairs with IO1's schema versioning.
* **MD7** Units parser, accept times, energies, lengths,
  temperatures with explicit units in the API; reject
  ambiguous bare floats.
* **ESD1** `vq.compute_esd(...)`, excited-state dynamics
  driver wrapping the ORCA `%ESD` block functionality.
* **ESD2** `HESSFLAG` Hessian-flow options, `VG` (vertical
  gradient), `AHAS` (adiabatic Hessian, adiabatic shift),
  `AH` (adiabatic Hessian), `VH` (vertical Hessian), `HHBS`
  / `HHAS` (half-and-half), `UFBS` / `UFAS` (uniform-field
  variants). Maps onto how the excited-state PES is
  approximated.
* **ESD3** `USEJ` Duschinsky rotation, full mode-mixing
  between ground- and excited-state normal modes, beyond the
  parallel-mode approximation.
* **ESD4** `ESD(FLUOR)`, `ESD(PHOSP)`, `ESD(ISC)`,
  fluorescence / phosphorescence / intersystem-crossing
  rate calculation modes.
* **ESD5** `LINES` parser, line-shape function selection
  (Lorentzian / Gaussian / Voigt) for the simulated
  vibronic spectrum.
* **ESD6** `MODELIST` selector, restrict vibronic coupling
  to a chosen mode subset; needed for chromophores embedded
  in large systems.

#### orca_* utility binaries (post-processing toolbelt)

ORCA ships 20+ standalone post-processing executables. Most
are small, single-purpose, and compose with the main run via
parsed output. vibe-qc's `vq.tools` namespace is the natural
home; many of these are already implicit in the
properties-suite milestone, but pinning the API shape now keeps the
naming consistent.

* **TOOL1** ✅ `vq.tools.aim`, atoms-in-molecules / QTAIM
  topology (critical points, basins, integration). ORCA
  `orca_2aim`. Pairs with TOPOND on the CRYSTAL side.
  Shipped 2026-06-24 in `python/vibeqc/qtaim.py`; CP search +
  bond-path tracing via finite-difference Hessian.
  QVF `topology.qtaim` writer shipped in QVF v1.2 (2026-06-24).
* **TOOL2** `vq.tools.to_molden(result)`, Molden / MKL /
  Multiwfn export format. ORCA `orca_2mkl`.
* **TOOL3** `vq.tools.asa(...)`, automatic-spectrum-analysis
  driver for vibronic / resonance-Raman simulation. ORCA
  `orca_asa`.
* **TOOL4** `vq.tools.chelpg(result)`, CHELPG ESP-fit
  charges. Already covered by CHG2 above; this entry pins
  the standalone CLI shape.
* **TOOL5** `vq.tools.euler(...)`, Euler-angle conversion
  utilities for tensor-property output. Tiny; pure
  ergonomics. ORCA `orca_euler`.
* **TOOL6** `vq.tools.export_basis(basis)`, basis-set export
  in NWChem / Gaussian / ORCA / Molpro / CRYSTAL formats.
  ORCA `orca_exportbasis`. Pairs with BAS1 / BAS2.
* **TOOL7** `vq.tools.fit_pes(points, model="morse")`, 1D
  PES fitter (Morse / harmonic / quartic / spline). ORCA
  `orca_fitpes`. Already partly covered by EOS1.
* **TOOL8** `vq.tools.mapspc(...)`, broadens stick spectra
  to envelope spectra (Lorentzian / Gaussian) with
  configurable width and lineshape. ORCA `orca_mapspc`.
* **TOOL9** `vq.tools.merge_fragments(frag_results)`,
  stitches per-fragment SCF results into a fragment-
  decomposed total (useful for SAPT-style decomposition,
  embedding workflows). ORCA `orca_mergefrag`.
* **TOOL10** `vq.tools.plot_vibmodes(hess, mode_indices)`
 - Cartesian displacement vectors as XYZ frames per mode.
  ORCA `orca_pltvib`.
* **TOOL11** `vq.tools.pnmr(...)`, paramagnetic NMR shifts
  (combines hyperfine + g-tensor + ZFS). ORCA `orca_pnmr`.
  Pairs with v0.12 EPR / NMR.
* **TOOL12** `vq.tools.vib_thermochem(hess, T, P)`,
  rigid-rotor / harmonic-oscillator thermochemistry post-
  processor. ORCA `orca_vib`. Probably already in vibeqc
  via thermo helpers; this entry standardises the CLI.
* **TOOL13** `vq.tools.gcp_correction(system, basis)`,
  geometrical-counterpoise standalone (companion to 3c-4
  above). ORCA `otool_gcp`.
* **TOOL14** `vq.tools.xtb_singlepoint(system)`,
  semiempirical xTB single-point inside vibe-qc workflow
  (driver-level fast preview before DFT). ORCA `otool_xtb`.
  Pairs with the existing xTB-driver discussion in the
  benchmarks plans.
* **TOOL15** `vq.tools.localize(orbitals, method=...)`,
  orbital localisation with `PM` (Pipek-Mezey), `FB`
  (Foster-Boys), `NEWBOYS`, `AHFB`, `IAOIBO`, `IAOBOYS`,
  `PMVVO`, `LIVVO` selectors. ORCA `orca_loc`. Mostly
  covered today; this pins the full method roster.
* **TOOL16** `vq.tools.fock_block_eig(fock, block)`,
  diagonalise a sub-block of the Fock matrix. ORCA
  `orca_blockf`.
* **TOOL17** `vq.tools.plot_orbital(orbital, grid)`,
  cube-grid orbital / density rendering, matplotlib-
  ready slices. ORCA `orca_plot`. Already partly covered
  by the `peierls` tutorial; this entry standardises the API.

#### Spectroscopy depth (extends Phase 19 + post-1.0 spec stack)

Beyond the user-facing methods (TDDFT, EOM-CC, MCD), ORCA
exposes a long tail of specialised spectroscopy modules. Most
are post-1.0 territory but worth itemising so we don't rebuild
the API surface when each lands.

* **SPEC7** `ABSFFMIO` / `ABSQ`, full-frequency-domain
  absorption with multipole-and-quadrupole oscillator
  strengths.
* **SPEC8** MCD / XMCD, magnetic CD on top of TDDFT, with
  X-ray (XMCD) extension for transition-metal core-level
  spectroscopy.
* **SPEC9** XAS / XES family, full-COC, RAS-CI core-hole,
  damped-response, and ROCIS variants of X-ray absorption /
  emission. Pairs with the MR + excited-state milestone.
* **SPEC10** RIXS / RIXSSOC, resonant inelastic X-ray
  scattering, with spin-orbit coupling for heavy atoms.
* **SPEC11** TRANSABS / TRANSCD, transient-absorption /
  transient-CD on top of pump-probe excited-state dynamics.
* **SPEC12** NRVS / VDOS, nuclear-resonance vibrational
  spectroscopy, vibrational density-of-states partitioning
  per-element. Pairs with VIB4 above.

#### Density / property exports (extends IO bucket)

The `%plots` and `%loc` blocks in ORCA enumerate density-cube
shapes and localisation-method options that the IO layer needs
to be able to write.

* **PLT1** `Format` selector, Gaussian-cube, Molden,
  netCDF, HDF5, raw-binary as output formats for any cube
  property.
* **PLT2** Density selectors, `MOCoefficients`,
  `2elIntegrals`, `Densities`, `Vaux` (auxiliary-density
  flag for hybrid + RI workflows). Each is a flag on
  `vq.plot_density(...)`.
* **PLT3** `%loc` full block, localisation method roster
  on the SCF orbitals (covered by TOOL15 above; this is
  the input-side analogue).

#### CRYSTAL geometry input, extra builders

The first sweep covered the user-facing builders (CLUSTER,
NANOTUBE, FULLE, WULFF, ATOMSUBS family). A second pass
catches the more specialised input keywords for symmetry
manipulation and finite-cluster construction.

* **CGEO1** `EXTERNAL`, read pre-built geometry from a file
  in CRYSTAL's external geometry format. Pairs with CIF /
  ASE-Atoms readers vibe-qc already has; this is the
  CRYSTAL-native dialect.
* **CGEO2** `SLABCUT` / `SLABINFO`, extract a slab from a
  bulk crystal by Miller indices, with explicit termination
  control. Already in B1.x phase; this names the
  specifically-CRYSTAL options.
* **CGEO3** `NANOROD` / `NANOCRYSTAL` (covered by B1.7
  above; this entry is just the CRYSTAL keyword cross-ref).
* **CGEO4** `HYDROSUB`, hydrogen substitution at dangling
  bonds for cluster carving. Already in B1.5; this names
  the CRYSTAL knob.
* **CGEO5** `MOLEXP` / `MOLSPLIT`, split a molecular
  crystal into per-molecule subsystems for fragment
  workflows.
* **CGEO6** `ROTCRY`, rotate the unit cell to align a
  given crystallographic axis with a Cartesian direction
  (for slab + interface setups).
* **CGEO7** `SUPERCEL` / `SUPERCON` / `SCELCONF` /
  `SCELPHONO`, supercell construction with various
  conventions (conventional / primitive, phonon-supercell
  with explicit q-mesh).
* **CGEO8** `ATOMORDE`, reorder atoms by Z, by name, or
  by an explicit list. Pure ergonomics but stabilises
  output formatting.
* **CGEO9** `BREAKELAS` / `ELASTIC`, symmetry-breaking
  strains for elastic-constant calculations. Pairs with
  ELASTCON in the existing roadmap.
* **CGEO10** `MAKESAED`, generate the symmetry-adapted
  external displacements (SAED) for IR / Raman
  intensity-symmetry analysis.
* **CGEO11** `RAYCOV`, modify covalent radii used by the
  bond / connectivity detector. Niche but cheap.
* **CGEO12** `MOLEBSSE` / `ATOMBSSE`, molecular-in-
  crystal counterpoise (covered by BSSE2 above).
* **CGEO13** `PLANES`, define crystallographic planes for
  property calculation (potentials, density slices),
  pairs with the `POTM` / `ECHG` properties.
* **CGEO14** `PRIMITIV` / `MODISYMM` / `PURIFY` /
  `SYMMREMO` / `TRASREMO`, symmetry-manipulation knobs:
  reduce to primitive cell, modify symmetry, purify
  numerically-noisy symmetry, remove symmetry / remove
  translational symmetry.
* **CGEO15** `FRAGMENT`, define explicit fragment
  partitions for fragment-correlation methods.
* **CGEO16** `EXTPRT` / `CIFPRT` / `CIFPRTSYM` /
  `COORPRT`, output-format selectors (CRYSTAL external,
  CIF, CIF+symmetry, plain coordinates).
* **CGEO17** `FINDSYM`, auto-detect the highest-symmetry
  setting consistent with the input geometry. Pairs with
  the spglib integration vibe-qc already has but
  surfaces a CRYSTAL-shaped API.
* **CGEO18** `STRUCPRT`, print the structure tensor
  (used in elastic-tensor post-processing).

#### CRYSTAL basis-set library

CRYSTAL23 ships a basis-set library with several entries that
vibe-qc's existing basis-loader doesn't carry yet.

* **CBAS1** `STO-nG` family with n = 3, 4, 5, 6, minimal
  Slater-fit basis variants. STO-3G is shipped; n > 3 is
  trivially accessible.
* **CBAS2** Pople-style `3-21G`, `6-21G`, covered for
  most rows by the existing Pople loader; audit gaps.
* **CBAS3** `POB-TZVP` family, periodic-optimised TZVP for
  solid-state work; CRYSTAL's recommended default for
  many DFT-on-crystals studies.
* **CBAS4** f-electron lanthanide / actinide bases, new in
  CRYSTAL23; needed for f-block solid-state.
* **CBAS5** g-type AOs, new in CRYSTAL23 (g-functions in
  the angular-momentum stack). Vibe-qc's libint integration
  supports g; verify the basis-loader path doesn't truncate.
* **CBAS6** ECP libraries, Hay-Wadt, Durand-Barthelat,
  Columbus, Stuttgart-Cologne SOREP. Pairs with the
  existing ECP roadmap.
* **CBAS7** `BASOPT`, internal basis optimiser (covered
  by BAS1 above; this is the CRYSTAL keyword).
* **CBAS8** `GHOSTS`, ghost-atom basis-functions for
  counterpoise / mid-bond functions. Pairs with BSSE1.

#### CRYSTAL calculation-type + SCF knobs

* **CALC1** `OPTGEOM` block options, full geometry-
  optimisation control surface (constraints, line-search
  variants, redundant internals, Hessian update schemes).
  Pairs with the v0.7 geometry-opt phase.
* **CALC2** `FREQCALC` block, frequency-calculation knobs
  (numerical-derivative step size, mode-by-mode integration,
  partial Hessian by atom selection).
* **CALC3** `EOS` block, built-in volume scan + EOS fit
  (covered by EOS2 above; this is the CRYSTAL block).
* **CALC4** `QHA` block, quasi-harmonic thermo workflow
  (covered by QHA1).
* **CALC5** `ELASTCON` with `AWESOME`, automatic-strain
  elastic-constant determination via the AWESOME-code
  symmetry analyser. Pairs with the existing ELASTCON entry.
* **CALC6** `PIEZOCON` / `PHOTOELA`, piezoelectric and
  photoelastic tensor calculation blocks (covered in the
  existing Berry-phase / properties roadmap; this names
  the CRYSTAL-side trigger).
* **CALC7** `CPHF` block, coupled-perturbed HF for
  response properties. Pairs with the post-1.0 response-
  theory roadmap.
* **CALC8** `CONFCNT` / `CONFRAND` / `RUNCONFS` (covered
  by SS1 / SS2 / SS3 above; this is the keyword cross-ref).
* **CALC9** SCF settings, `SPINLOCK`, `LEVSHIFT`,
  `BROYDEN`, `ANDERSON`, `ATOMSPIN`. Most are covered by
  vibe-qc's existing SCF options; audit for missing
  Anderson / Broyden mixing.
* **CALC10** `TWOCOMPON` / `SOC`, 2-component SCF with
  spin-orbit coupling. Pairs with the existing X2C / DKH
  relativistic roadmap.

#### CRYSTAL properties keywords (extends the properties suite)

CRYSTAL's `properties` step is its post-SCF analysis layer
(separate executable in CRYSTAL; one `vq.properties` module
in the natural vibe-qc analogue).

* **PROP1** `NEWK`, k-mesh densification post-SCF for
  property evaluation. Pairs with the existing k-point
  story.
* **PROP2** `COMMENS` / `NOSYMADA` / `PATO` / `PBAN` /
  `PGEOMW` / `PDIDE` / `PSCF` / `RDFMWF`, properties-step
  control flags (commensurate, suppress symmetry-
  adaptation, project to atomic, project to band, geometry
  weights, projected DOS, project SCF, read formatted
  wavefunction). Most are flags on existing analysis
  functions.
* **PROP3** `ADFT` / `ACOR` (covered by CORR1 above).
* **PROP4** `BAND`, band-structure plotter along k-paths.
  Band structures already ship (Phase V4); this is the CRYSTAL keyword.
* **PROP5** `BIDIERD`, directional Compton profile
  (covered by XRD2 above).
* **PROP6** `CLAS`, classical (Madelung) potential at
  selected points; useful for embedded-cluster QM/MM
  setup (pairs with EMB1).
* **PROP7** `ECHG` / `ECH3`, charge-density 2D / 3D
  cube output. Related cube output already ships (Phase V1).
* **PROP8** `EDFT` (covered by CORR1).
* **PROP9** `EMDLDM` / `EMDPDM`, electron-momentum
  density on density-matrix (covered by EMD1 above; these
  are the explicit CRYSTAL flags).
* **PROP10** `HIRSHCHG`, Hirshfeld charges (molecular Hirshfeld
  ships; periodic Hirshfeld is PR1 in the properties suite).
* **PROP11** `KINETEMD`, kinetic-energy density / EMD on
  the same grid. Useful for non-covalent index post-
  processing.
* **PROP12** `PMP2`, periodic-MP2 single-point on the
  converged SCF (covered in the existing periodic-MP2
  roadmap).
* **PROP13** `POLI`, Mulliken-style population analysis
  on a periodic system.
* **PROP14** `POTM` / `POT3` / `POTC`, electrostatic
  potential 2D / 3D / classical-only cube output.
* **PROP15** `PPAN`, projected population-analysis
  printout.
* **PROP16** `XFAC` (covered by XRD1).
* **PROP17** `BOLTZTRA`, Boltzmann-transport coefficients
  (Seebeck, electrical / thermal conductivity) from the
  band structure. Pairs with the post-1.0 transport story.
* **PROP18** `ANISOTRO` / `ISOTROPIC`, anisotropic /
  isotropic hyperfine coupling tensors (pairs with EPR
  in v0.12).
* **PROP19** `POLSPIN` / `SPINCNTM` (covered by DIAG2
  above).
* **PROP20** `ANBD` / `BWIDTH` / `DOSS`, band-by-band
  analysis, bandwidth printout, density-of-states.
* **PROP21** `EMDL` / `EMDP` / `PROF`, line / plane /
  profile EMD output formats (covered by EMD1).
* **PROP22** ✅ `COOP` / `COHP` (covered by BOND1 / BOND2, shipped 2026-06-23).
* **PROP23** `SPOLBP` / `SPOLWF` (covered by POL1 / POL2).
* **PROP24** `LOCALWF`, Wannier-localisation post-
  processing (covered in the post-1.0 Wannier story).
* **PROP25** `DIEL`, frequency-dependent dielectric
  function. Pairs with v0.12 spectroscopy.
* **PROP26** `ISO` + `POTC`, isotropic + classical
  potential printout for QM/MM coupling.
* **PROP27** ✅ `TOPO`, topology of the electron density
  (covered by TOOL1, shipped).
* **PROP28** `ATOMIRR` / `CRYAPI_OUT` / `FMWF` /
  `INFOGUI` / `XML`, output-format selectors (irreducible
  atoms, CRYSTAL Python API output, formatted wavefunction,
  GUI metadata, XML). Pairs with the IO bucket.

#### CRYSTAL post-processing utilities

* **CTOOL1** `MOLDRAW` / `COORPRT`, geometry-export
  formats (covered by CGEO16).
* **CTOOL2** `crysplot`, built-in matplotlib-style
  plotter for CRYSTAL output. vibe-qc has tutorial-level
  plotting helpers; this entry pins a unified
  `vq.plot.crystal_style(...)` namespace.
* **CTOOL3** `cryapi_inp`, programmatic input-builder
  Python module; mirrors vibe-qc's existing input-builder
  ergonomics.
* **CTOOL4** TOPOND module, full QTAIM topology /
  basin-integration (covered by TOOL1).
* **CTOOL5** `runcry06` / `runprop06`, wrapper scripts
  for chained SCF + properties. Compound covers this
  (CMP1).
* **CTOOL6** AWESOME-code coupling, symmetry-aware
  strain / Hessian post-processing. Pairs with CALC5.

#### CRYSTAL job-control + parallelism

CRYSTAL ships five executables for different parallel models
(`Pcrystal`, `MPPcrystal`, `crystalOMP`, `PcrystalOMP`,
`MPPcrystalOMP`). vibe-qc has a single executable + OpenMP +
optional MPI; the items below cover the operational knobs.

* **PAR1** Document the replicated-data vs distributed-data
  modes, replicated (each rank holds full Fock / density;
  cheap on small systems) vs distributed (ScaLAPACK; needed
  for large systems). Pairs with the v0.5.x performance docs.
* **PAR2** ScaLAPACK distributed Fock diagonalisation,
  for systems where the dense Fock exceeds per-rank memory.
  work for after the production MPI milestone, and only after replicated-data MPI benchmarks
  show dense matrices are the measured wall.
* **PAR3** Dual-level parallelism (MPI x OpenMP) tuning
  guide, when to bias toward MPI vs OMP. Pairs with the
  perf-knob scaling scan in `examples/debug/`.
* **PAR4** Defaults audit, match `MPPcrystal`'s default
  knob settings on equivalent vibe-qc invocations as a
  sanity reference.
* **PAR5** `REPLDATA` flag, explicit replicated-data
  selector for benchmarking against distributed mode.
* **PAR6** `OMP_NUM_THREADS` / `MKL_DEBUG_CPU_TYPE`
  cookbook entries in installation.md, already partly
  there; expand the recommendations for AMD CPUs (the
  `MKL_DEBUG_CPU_TYPE=5` workaround).
* **PAR7** Memory knobs, `BIPOSIZE` (bielectronic-
  integral buffer), `EXCHSIZE` (exchange buffer),
  `LOWMEM` (memory-conservative path), `CHUNKS`
  (integral chunking). Pairs with the existing
  `LatticeOptions.cutoff_bohr` knob; surface a
  `MemoryOptions` struct.

#### CRYSTAL ecosystem tools beyond the main exec

* **CECO1** Cryscor interface, periodic-MP2 /
  periodic-CCSD post-processor. Pairs with the
  existing periodic-MP2 roadmap.
* **CECO2** Pproperties, the parallel `properties`
  executable. Mirrored by parallelising vibe-qc's
  property step.
* **CECO3** `INPUT` keyword, universal escape hatch
  in CRYSTAL input that includes another file. The
  Python API equivalent is just `**kwargs` forwarding;
  document the pattern.
* **CECO4** Migration appendices, CRYSTAL ships
  detailed migration notes between major versions.
  Pin a `MIGRATION.md` doc that does the same for
  vibe-qc (currently we have `CHANGELOG.md` but no
  migration narrative).
* **CECO5** Appendix D, formal file-formats list
  (every on-disk format CRYSTAL reads / writes).
  Pairs with the `docs/file_formats.md` from IO3.

The second-pass total is roughly 95 items spanning ~12
themes. Themes 1-4 (Compound, AutoCI, IO, MD/ESD) are the
"workflow infrastructure" story, most of the C++ work is
already done; this is API-shape design + Python plumbing.
Themes 5-7 (orca_*, spectroscopy, density exports) flesh
out post-processing. Themes 8-12 (CRYSTAL geom / basis /
calc / props / parallelism / ecosystem) are the solid-
state-side counterparts. Combined with the first-sweep
~70 items, the cross-code feature gap is ~165 items
total. This is a multi-year backlog at current cadence;
prioritisation will come from user requests, not from
ticking off the manuals top-to-bottom.

### Cross-code feature gaps, third pass (ORCA 6.1.1 + CRYSTAL23 re-sweep, 2026-07)

A dedicated re-read of the current ORCA 6.1.1 manual
(`orca-manual.mpi-muelheim.mpg.de`), chapter by chapter, plus the
CRYSTAL23 User's Manual (October 2023 edition, including its own
curated §1.2 "Program Features" checklist), looking specifically for
methods that survived the first two sweeps above. **None of this is
scheduled ahead of the open milestones.** The release-paper hardening
work and the refactor / unification proposals above (`FockMatrix` and
its siblings) come first; nothing below starts
before that work lands, regardless of how small an item looks. Treat
every entry here as backlog, not as a claim on a specific minor.

#### Bonding + noncovalent-interaction analysis (extends LED, D3/D4, PR26)

vibe-qc already ships pairwise D3-BJ/D4 dispersion and DLPNO-CCSD(T)
with LED (v0.14.0). Three ORCA analysis layers build on exactly that
foundation and are not yet itemised:

* **NCOV1** ETS-NOCV / EDA-NOCV (ORCA manual §5.35): a fragment-pair
  energy decomposition (electrostatic + Pauli + orbital + dispersion
  terms) combined with Natural Orbitals for Chemical Valence to
  visualise the electron-density rearrangement on bond formation.
  Distinct from the already-planned LMOEDA / Hayes-Stone **PR26**
  decomposition, a different partition of the same interaction energy.
  Reference: Mitoraj, Michalak & Ziegler's EDA-NOCV formalism; see
  ORCA manual §5.35 for the full citation list.
* **NCOV2** HFLD, Hartree-Fock plus London Dispersion (§5.37): solve
  DLPNO-CC on a fragment pair with intramonomer correlation switched
  off, extract the intermolecular dispersion via the same LED
  machinery already shipped, and add it to a plain HF interaction
  energy. Cheap (HF cost plus one intermolecular-only DLPNO-CC pass),
  and per ORCA's own benchmarking comparable in accuracy to full
  DLPNO-CCSD(T) for H-bonded and dispersion-bound complexes. A natural
  LED follow-on rather than a new correlation method.
* **NCOV3** ADLD(D), atomic decomposition of the D2/D3/D4 dispersion
  energy (§5.38): since the pairwise dispersion terms are already
  computed for the total D3-BJ/D4 energy, assigning half of each
  pairwise contribution to each atom (plus the three-body Axilrod-
  Teller-Muto term already shipped as `s9`, see **3c-3** above) is a
  small post-processing pass over data vibe-qc already has, not a new
  energy expression.

#### Post-Kohn-Sham direct-RPA correlation energy

* **RPA1** RPAC, the direct random-phase-approximation correlation
  energy evaluated as a numerical frequency integral over the
  Kohn-Sham response matrix in an auxiliary (`/C`-type) basis (ORCA
  manual §3.8). Distinct from **MR4** MCRPA above, which is an
  excited-state extension of MCSCF, not a ground-state correlation
  functional. Essentially non-empirical, comparable cost to RI-MP2,
  and a genuinely different accuracy/cost point from the DFT and
  DLPNO-MP2/CC stack vibe-qc already ships. ORCA's own implementation
  is explicitly a pilot (energies only, no analytic gradients, no
  excited states), so a first vibe-qc cut could reasonably match that
  scope. References: Langreth & Perdew, *Solid State Commun.* **17**,
  1425 (1975); Furche, *Phys. Rev. B* **64**, 195120 (2001) and
  *J. Chem. Phys.* **129**, 114105 (2008).

#### Relativistic Hamiltonian program (promotes an existing placeholder)

**BS6** and **BSC1** above already schedule the relativistic basis-set
families (x2c-\*, dhf-\*, SARC, Sapporo-DKH) and both note they "pair
with an X2C / DKH2 / ZORA Hamiltonian in the relativistic phase," but
that Hamiltonian work has no roadmap entry of its own, only that one
forward reference. Splitting it out:

* **REL1** A scalar-relativistic one-component Hamiltonian route:
  X2C (exact two-component, spin-free/one-component variant per
  Franzke & Weigend), DKH2 (Douglas-Kroll-Hess, second order), and
  ZORA (zeroth-order regular approximation, including the scaled
  variant), as alternative routes to the same goal, correct valence
  energetics and properties for heavy elements without a full 4- or
  2-component treatment. Gates **BS6**/**BSC1** actually mattering:
  those basis sets are only useful once one of these Hamiltonians
  exists to pair them with.
* **REL2** Finite Nucleus Model (§2.12.8): replace the point-charge
  nucleus with a Gaussian or Fermi charge distribution. Matters most
  for hyperfine tensors and other core-region-sensitive properties on
  heavy elements, where the point-nucleus singularity is the dominant
  source of error, not basis incompleteness.
* **REL3** Picture-change corrections (§2.12.7): once a wavefunction
  is computed under a transformed (X2C/DKH/ZORA) Hamiltonian, the
  non-relativistic property operator no longer lives in the same
  representation as the transformed wavefunction. Evaluating a
  property (NMR shielding, EFG, hyperfine coupling) without correcting
  for this is a silent, systematic error, the ORCA manual is explicit
  that skipping it "is clearly not a physical effect" but a
  book-keeping one. If **REL1** ships, **REL3** has to ship with it or
  every downstream property built on top of it is wrong by construction,
  the same class of bug as § 7's periodic-SCF oscillation rule: papering
  over it is not an option.

#### Ab initio molecular magnetism / single-molecule-magnet suite (post-1.0, far out)

Builds on machinery vibe-qc already has (CASSCF, NEVPT2, the spin-orbit
coupling operator, broken-symmetry exchange couplings via **DIAG3**).
**Not an external interface**: ORCA's `SINGLE_ANISO`/`POLY_ANISO`
modules literally shell out to Liviu Ungur's separate codes, which
§ 10 forbids outright for vibe-qc (another QC-adjacent program, not a
library). The item on the table is a **native implementation of the
same underlying physics**, ab initio crystal-field / pseudospin
Hamiltonian extraction from a CASSCF+SOC calculation, not a wrapper
around anyone else's binary.

* **MAG1** Quasi-Degenerate Perturbation Theory (QDPT, §5.29): build
  an effective pseudospin Hamiltonian (g-tensor, D-tensor / zero-field
  splitting, exchange parameters beyond simple Yamaguchi J) from a set
  of CASSCF/NEVPT2 states plus the spin-orbit coupling operator.
  Broader than the existing EPR g-tensor work; QDPT is the general
  machine EPR, D-tensor, and magnetic-relaxation inputs all come from.
* **MAG2** Magnetic relaxation rates (§5.30): Fermi's golden rule over
  a QDPT-derived spin Hamiltonian coupled to phonons (Herzberg-Teller
  expansion of the spin Hamiltonian in normal coordinates), giving
  parameter-free Orbach / Raman / QTM relaxation-time estimates for
  single-molecule magnets. Needs **MAG1** plus the existing analytic
  Hessian/phonon machinery as inputs.
* **MAG3** Single-ion and multi-ion magnetic anisotropy analysis in
  the style of `SINGLE_ANISO`/`POLY_ANISO`, built natively on **MAG1**
  rather than interfaced. Lanthanide/actinide single-molecule-magnet
  research is a real and growing user base for this kind of analysis.

References: Chibotaru & Ungur's ab initio SMM formalism and the
Neese-group QDPT/magnetic-relaxation papers cited in ORCA manual
§5.29-§5.30; specific DOIs to be pinned when this item is actually
scoped for implementation, not guessed at roadmap-writing time.

#### X-ray spectroscopy, a cheap one-electron route (extends SPEC9)

* **XSPEC1** SGS-DFT, Static Ground State DFT (§5.39): reduces
  core-level spectroscopy (XAS/XES/RIXS) to Kohn-Sham orbital-energy
  differences instead of an explicit excited-state or multireference
  treatment (contrast **SPEC9**'s RAS-CI core-hole route, and the
  existing ROCIS/TDDFT machinery). Much cheaper per spectrum; a
  natural complement once **SPEC9** exists, not a replacement for it,
  ORCA documents it as trading some accuracy for cost, valuable for
  screening large numbers of structures before a RAS-CI or MRCI
  follow-up on the interesting ones.

#### Periodic two-component / spin-orbit / non-collinear magnetism (promotes an existing placeholder)

The periodic tutorial-parity table above already carries a bare
"Spin-orbit / noncollinear | Post-1.0" line with no supporting detail
anywhere in this document. CRYSTAL23 chapter 6, "Spin-Orbit Coupling,
Non-Collinear Magnetism, Two-Component Spinors," is the concrete
target to promote it against. vibe-qc's periodic magnetism today is
collinear RHF/UHF/RKS/UKS only (a scalar spin-up/spin-down occupation
per site); this is the next rung up, and the periodic-side sibling of
the molecular **REL1**-**REL3** relativistic Hamiltonian program above,
not a separate research direction.

* **PSOC1** Two-component periodic SCF: a generalized-HF-style
  two-component spinor basis (CRYSTAL `GHF`), self-consistent
  spin-orbit coupling inside the SCF itself rather than as a post-hoc
  perturbation, and a non-collinear initial guess for the
  magnetization. Distinct from **REL1**'s molecular scalar-relativistic
  route; periodic two-component SCF needs its own convergence handling
  (Fock-matrix mixing is CRYSTAL's stabiliser of choice there) and its
  own SOC-capable pseudopotentials (**PSOC3** below).
* **PSOC2** Non-collinear (and spin-current) periodic DFT: the
  magnetization as a vector field rather than a scalar per site,
  needed for frustrated, canted, or helical magnetic structures that
  no collinear UKS calculation can represent. CRYSTAL's spin-current
  DFT (SCDFT) variant goes one step further and makes the XC
  functional depend on the orbital-current density too, not just the
  density and its gradient.
* **PSOC3** SOREP pseudopotentials, spin-orbit relativistic ECPs, as
  distinct from the averaged-relativistic (AREP) ECPs vibe-qc already
  supports. Required input for **PSOC1** on heavy elements; same
  libecpint-based machinery, a different parameter set to source and
  license-check (§ 1) before bundling.

This is squarely post-1.0 research-direction territory, exactly what
the existing tutorial-parity table already says; nothing here should
be read as a commitment for any open milestone.

#### Periodic nonlinear-optical response via CPHF/CPKS (extends PR16-PR22)

The properties-suite periodic-response bucket (**PR16**-**PR22**) covers Born
effective charges, the (linear) dielectric tensor, elastic/piezo/
photoelastic tensors, and periodic IR/Raman intensities, all first- or
second-order response quantities. CRYSTAL23 chapter 10 ("Dielectric
Properties up to Fourth Order via the Coupled Perturbed HF/KS Method")
goes one and two orders further:

* **PRNLO1** Static first and second hyperpolarizability tensors
  (third- and fourth-order CPHF/CPKS energy derivatives with respect
  to an applied electric field), the periodic analogue of the
  molecular **PR9** hyperpolarizability work, gated on the same
  **PR7** CPHF/CPKS substrate already planned.
* **PRNLO2** Frequency-dependent (dynamic) CPHF/CPKS: Second-Harmonic
  Generation and the Pockels effect, the same hyperpolarizability
  tensors evaluated at a finite applied-field frequency instead of the
  static limit.
* **PRNLO3** Mixed derivatives (field plus nuclear displacement, field
  plus cell strain): Raman polarizability tensors and the electronic
  term of the direct piezoelectric tensor, sharing machinery with
  **PR20** and **PR22** rather than duplicating it.

### CP2K parity sweep (second parity target, post-`v1.0`)

A pass over CP2K's current `methods/` manual (`manual.cp2k.org`, the
narrative tree, not the auto-generated `CP2K_INPUT` keyword reference,
which is a per-keyword dump the way sweeping ORCA's raw input-block
listing would be, not the right altitude for a gap sweep). This is the
**second** parity target from the intro above: nothing here starts
before `v1.0` (ORCA + CRYSTAL23), and most of it lands well after,
alongside or following the production MPI and scaling milestone. A large fraction
of CP2K's surface is already on this roadmap by lineage, GPW/GAPW
(shipped), ADMM, OT-SCF, and Kerker/Anderson mixing are all
CP2K-originated methods vibe-qc already tracks; this sweep is only the
remainder.

#### GW quasiparticle band structure + Bethe-Salpeter optical spectra

The single biggest method-level gap found. Neither appears anywhere
else in this document, only a passing mention of GW as a beneficiary
of tensor-decomposition acceleration (Acceleration program above).

* **GW1** Quasiparticle energies via the GW approximation: G₀W₀
  (non-iterative, on top of KS/HF orbitals), evGW₀ (eigenvalue
  self-consistency in the Green's function only), and evGW (full
  self-consistency in both G and the screened interaction W).
  Corrects DFT's systematic band-gap underestimate; CP2K's own
  guidance recommends ev GW₀@PBE as the practical default for both
  molecules and periodic solids. Three system-size regimes matter:
  molecules, small periodic cells with k-point sampling, and large
  supercells via a Γ-point-only route once the density matrix decays
  enough to make that valid, the last of these pairs naturally with
  vibe-qc's existing periodic-locality work. Self-consistency level
  is a separate axis from *how the self-energy's frequency integral
  is evaluated*, PySCF's `gw`/`pbc.gw` offer analytic continuation
  (Padé, N⁴, the fast default), contour deformation (N⁴, slower but
  more robust, recommended for core/high-energy states), and exact
  full diagonalization of the response matrix (N⁶, no frequency-grid
  error, and general enough to swap in a TDDFT kernel for the
  screening instead of plain dRPA). A first implementation should
  pick one (AC is the practical starting point) but the three-way
  split is worth keeping visible rather than baking in one choice.
* **GW2** Bethe-Salpeter equation (BSE) on top of **GW1**'s
  quasiparticle energies: excitation energies, oscillator strengths,
  and natural transition orbitals from a generalized eigenvalue
  problem over the screened electron-hole interaction W(ω=0).
  Captures excitonic binding that plain TDDFT (already shipped,
  v0.12.0) structurally cannot, since Casida/TDA excitations are a
  mean-field response, not an interacting electron-hole pair.
  Molecular-first gate (§ 14): CP2K's own BSE is molecule-only today,
  periodic BSE is explicitly "work in progress" even there.

References: Hedin's original GW formalism; van Setten, Caruso, Sharifzadeh
et al.'s survey and benchmark papers on GW implementations; Blase,
Duchemin, Jacquemin's review of BSE for molecular excitations.

#### Real-time propagation for broadband spectra (fixed nuclei only)

CP2K's real-time-propagation (RTP) machinery covers two genuinely
different things, and only one belongs on this roadmap:

* **RTP1** δ-kick real-time TDDFT: an instantaneous broadband electric-
  field pulse excites all transitions at once at a fixed geometry, the
  induced dipole is propagated and recorded, and a Fourier transform
  converts the time-domain signal into the full absorption spectrum in
  one propagation, instead of one Casida/TDA diagonalization per
  excited state of interest. Same physical observable as the existing
  linear-response TDDFT (v0.12.0), a different, complementary
  algorithm, better suited to broadband spectra or systems where many
  states are needed at once. Static nuclei throughout, no new scope
  concern.
* **Not proposed: Ehrenfest MD** (moving nuclei on a mean-field,
  non-stationary electronic density). This is a non-adiabatic method
  by construction, nuclei respond to an electronic state that is not
  an eigenstate, exactly the class of method this roadmap already
  excludes explicitly ("Non-adiabatic methods (CASSCF + SOC, surface
  hopping)," What's explicitly out of scope above). Being a mean-field
  single trajectory rather than a stochastic multi-surface hop doesn't
  move it out of that exclusion.

#### Constrained DFT (CDFT)

* **CDFT1** Charge/spin density constrained to atom-centered fragment
  regions via Lagrange multipliers on the KS functional, an outer
  root-finding loop (Newton + backtracking) around the inner SCF.
  Produces diabatic charge/spin-localized states standard functionals
  incorrectly delocalize (a self-interaction-error symptom). Primary
  uses: Marcus-theory electronic couplings for charge-transfer rates,
  Heisenberg-model J parametrization from constrained states (a
  different route to the same J's **DIAG3**'s broken-symmetry approach
  targets), and CDFT-CI (constrained states as a small CI basis to fix
  spurious delocalization in dissociation curves).

#### ALMO-SCF and a third EDA flavor

* **ALMO1** Absolutely Localized Molecular Orbitals SCF: orbitals
  restricted to individual molecular fragments (block-diagonal
  optimization) with an optional delocalization correction (XALMO or
  full delocalization) layered back on. A linear-scaling SCF route for
  large weakly-interacting systems (molecular liquids, crystals),
  belongs alongside OT-SCF and ADMM in the v0.9.x locality-exploitation
  bucket above as a third technique in the same family.
  * **ALMO2** ALMO-EDA, energy decomposition analysis built on the
    ALMO block-localized starting point. A third distinct partition of
    the interaction energy alongside **NCOV1** (ETS-NOCV) and **PR26**
    (LMOEDA/Hayes-Stone) above, not a duplicate: three different
    starting points (NOCV natural orbitals, Hayes-Stone perturbative
    polarization/exchange, and ALMO's fragment-localized orbitals)
    genuinely disagree on how to split the same number, which is
    exactly why all three get cited independently in the EDA
    literature.

#### Subtractive (Kim-Gordon) embedding and local-RI density fitting

Two lighter-weight extensions of machinery vibe-qc is already building:

* **EMB6** Kim-Gordon subtractive embedding: E_tot = E_HK[ρ_tot] -
  Σ_A E_HK[ρ_A] + Σ_A E_KS[ρ_A], with the non-additive kinetic-energy
  coupling between fragments handled through a local-potential
  approximation rather than a full orbital-free functional. A
  cheaper, less accurate sibling of the DMET/projection-based
  embedding already on the post-1.0 multireference-embedding list
  above, useful when the fragments are weakly enough coupled that the
  approximate kinetic-energy term is good enough.
* **RI3** Local RI (atom-pair-local density fitting): fits the
  pair density ρ_AB in functions centered on atoms A and B only,
  instead of a global auxiliary expansion. Sparse by construction,
  trades some accuracy for linear-scaling grid operations; most
  valuable for condensed-phase, non-orthorhombic, large-cutoff GPW
  runs with many SCF steps, least valuable for metals (where
  diagonalization dominates) or short MD runs (where wavefunction
  extrapolation already converges fast). A variant to evaluate
  alongside, not instead of, the existing periodic GDF work.

#### Sub-cubic RPA, and a concrete GPU-XC pointer

* **RPA2** CP2K's own answer to making **RPA1** (the direct-RPA
  correlation energy added in the ORCA sweep above) practical past
  pilot scale: RI with a short-range metric for 3-center sparsity, a
  numerical Laplace transform in place of the frequency integral
  (weighted sum over a handful of MINIMAX-placed grid points), and an
  AO-basis sparse-tensor formulation, together reaching sub-cubic
  scaling against the naive quartic cost. Heavy prefactor, so only
  pays off past a few hundred atoms; **RPA1** should track this as its
  large-system algorithm from the start rather than discovering the
  scaling wall later.
* **GPU-XC1** GauXC-style external XC quadrature offload: CP2K
  optionally routes exchange-correlation energy/potential/gradient
  evaluation through an external GPU-capable integrator instead of its
  own grid code. Not a new roadmap item on its own, a concrete target
  for the already-generic "GPU acceleration" entry in the `v1.x`
  algorithmic/hardware section above, XC-grid quadrature specifically,
  alongside integral evaluation and diagonalization.

#### Explicitly not proposed (already covered or already excluded elsewhere)

Checked and deliberately left off this list, to keep it from padding
itself with things already decided:

* **Wannier90 interface**: CP2K does not build Wannier functions
  itself, it writes `.win`/`.mmn`/`.eig`/`.amn` files and hands
  construction to the separate Wannier90 program. vibe-qc already
  plans *native* Wannier-function construction (Boys localization,
  CRYSTAL-parity work referenced above); shelling out to Wannier90
  instead would be exactly the external-program pattern § 10 forbids.
* **Image charges and polarizable-force-field QM/MM**: both are CP2K
  QM/MM variants, and general QM/MM is already blanket-excluded before
  `v1.0` (What's explicitly out of scope, above); nothing here changes
  that call.
* **Path-integral MD (PIMD) and Gibbs-ensemble Monte Carlo (GEMC)**:
  real, distinct sampling paradigms beyond classical/BOMD trajectories
  (nuclear quantum effects via ring-polymer discretization; vapor-
  liquid phase equilibria via particle-swap Monte Carlo,
  respectively), but vibe-qc has no MD at all yet (**MD1**-**MD6**
  above still come after the production MPI milestone). Worth remembering as two further
  sampling modes once a classical MD driver exists, not worth a
  numbered item ahead of the driver itself; CP2K's own PIMD manual
  page is honest about being unwritten, which is a fair indicator of
  how far down its own priority list it sits too.

### PySCF sweep (validation-reference cross-check, not a third parity target)

A pass over PySCF's `user/` guide (`pyscf.org`), the same altitude as
the ORCA/CRYSTAL23/CP2K sweeps above. **This does not make PySCF a
parity target**, the intro's "Parity targets, in order" section is
explicit that PySCF stays a validation reference (§ 10, and
[Dependencies](#dependencies) below), not a program to catch up to
feature-for-feature. But PySCF carries several genuinely distinct
methods vibe-qc already validates *against* without yet *offering*,
and those are worth tracking regardless of which tier PySCF sits in.
Nothing here is scheduled before `v1.0`; most of it slots in wherever
its prerequisite method already sits (noted per item).

#### A second Green's-function/propagator family for charged + neutral excitations

Distinct from **GW1**/**GW2** above (screened-Coulomb self-energy) and
from the EOM-CC / CASSCF-based excited-state stack already shipped:
these are perturbative-propagator methods with their own accuracy/cost
point, o(N⁵)-ish rather than CC-scaling, and their own native outputs
(Dyson orbitals, spectroscopic factors) that neither the CC nor the
GW stack produces directly.

* **ADC1** Algebraic Diagrammatic Construction, ADC(2), ADC(2)-x
  (extended second order), and ADC(3): excitation energies (neutral,
  via the polarization propagator), ionization potentials and electron
  affinities (charged, via the one-particle Green's function), plus
  transition intensities, spectroscopic factors, and Dyson orbitals as
  a side effect of the propagator formalism, not a separate
  post-processing step the way NTOs are for TDDFT. Core-valence
  separation (CVS) gives core-ionization energies without solving for
  every lower state first, relevant alongside the **XSPEC1**/**SPEC9**
  X-ray-spectroscopy items above. RHF/UHF/ROHF references.
* **AGF1** Auxiliary second-order Green's function perturbation theory
  (AGF2): "loosely an iterative approximate ADC(2)," but self-
  consistent, solves the Dyson equation with a coarse-grained
  self-energy (compressing the first two spectral moments of the MP2
  self-energy into renormalized auxiliary states, O[N³] to O[N]) *and*
  feeds the resulting correlated density back into a Fock-matrix
  update, converging the whole charged-excitation spectrum
  simultaneously rather than one IP/EA at a time. Cheaper than GW
  (second-order self-energy, not screened-Coulomb), a genuinely
  different cost/accuracy point from **ADC1**, not a duplicate.

#### A cheaper CASSCF dynamic-correlation route: MC-PDFT

* **PDFT1** Multi-configuration pair-density functional theory:
  a CASSCF/CASCI reference plus a density functional evaluated on the
  *on-top pair density* (the two-electrons-at-one-point probability)
  instead of the spin density, letting one functional encode both
  strong correlation and spin symmetry without CASPT2/NEVPT2's
  perturbative dynamic-correlation step. Translated on-top functionals
  carry a `t`/`ft` prefix (`tPBE`, `ftBLYP`) over the existing GGA/
  hybrid table already shipped, so the XC-functional side is mostly
  already there; the on-top-density machinery and the CASSCF-energy-
  not-PDFT-energy minimisation detail are the new pieces. Multi-state
  variants (L-PDFT, XMS-PDFT, CMS-PDFT) target conical-intersection
  and multi-state work already on the MR + excited-state list, a cheaper alternative
  route there, not a new scope item.

#### Electron-phonon coupling (extends shipped phonons + the PR7 CPHF/CPKS substrate)

* **EPC1** First-order electron-phonon coupling matrix elements
  (how electronic states respond to a phonon normal-mode
  displacement), via the linear electron-phonon coupling
  approximation. Two routes, analytic (CPHF/CPKS, reusing the
  response substrate **PR7** already plans) or finite-difference
  (displace along each phonon mode, reusing the phonon machinery
  already shipped at v0.8.0), both needing a geometry-relaxed
  reference the way any coupled-perturbed method does. Foundational
  for carrier-mobility and polaron-physics properties, and for
  superconductivity's Eliashberg-theory route, neither on this roadmap
  yet, but this is the prerequisite matrix element either would need.
  Periodic scope starts Γ-point-only (matches PySCF's own current
  scope); k-resolved electron-phonon coupling is a later refinement.

#### Intrinsic Atomic/Bond Orbitals, and the localization schemes around them

Population analysis, Mulliken/Löwdin/Mayer/Hirshfeld/NBO, and Boys-
localized Wannier functions are already covered above. IAO/IBO are a
distinct, smaller idea worth calling out on their own:

* ✅ **IAO1** Intrinsic Atomic Orbitals (IAO) and Intrinsic Bond Orbitals
  (IBO): a minimal-basis projection of the converged SCF density that
  reproduces it *exactly* (unlike a truncated minimal-basis
  approximation), giving a population-analysis and orbital-
  localization scheme with no free parameters to tune, "an unbiased
  bridge between quantum theory and chemical concepts" in Knizia's own
  framing. Distinct from NBO (which optimizes a different objective)
  and worth having alongside it, not instead of it, the literature
  uses both for different questions. Shipped (`python/vibeqc/iao.py`,
  validated against Knizia Table 1/Fig. 5), reachable via
  `run_job(localize="ibo")`, which fills the QVF `wf_localized` slot;
  see [bond_analysis.md](user_guide/bond_analysis.md).
* **LOC1** Pipek-Mezey (maximizes atomic population charges, preserves
  σ/π separation, the reason it is preferred over Boys for systems
  where that separation matters) and Edmiston-Ruedenberg (maximizes
  orbital self-repulsion; correct but expensive) as two more
  localization objectives alongside the Boys/Wannier work already
  planned. Cheap to add once one localization framework exists, since
  the objective function is the only thing that changes.

#### Seminumerical exchange (SGX), a second grid-based exchange route alongside RIJCOSX

* **SGX1** A different way to avoid analytic 4-index exchange
  integrals than the already-planned RIJCOSX (**v0.7.x** RI / DF
  acceleration wave above): evaluate one of the two electron
  coordinates in the exchange integral analytically and the other on
  a real-space grid (not COSX's numerical-quadrature-of-the-whole-
  operator scheme), with density-matrix-based "P-junction" screening
  and overlap-fitting to control aliasing error. O(N³) before
  screening, trending toward O(N²) (or O(N) for gapped systems) once
  screening kicks in, reported at roughly 0.03 mEh against analytic
  exchange in PySCF's own accuracy check. A genuinely different
  algorithm from COSX worth evaluating alongside it once the RI/DF
  wave lands, not a re-implementation of the same idea under a new
  name.

#### Particle-particle RPA, a second RPA channel alongside RPA1

* **PPRPA1** particle-particle RPA (ppRPA): builds the response matrix
  from particle-particle and hole-hole pairs instead of **RPA1**'s
  particle-hole pairs, correlation-energy-exact to second order in the
  electron-electron interaction, and *equivalent to ladder-CCD*, a
  genuinely different formal object from direct RPA, not a
  reformulation of it. Its distinguishing use case is what makes it
  worth a separate item: excitation energies of an N-electron system
  computed as differences between (N±2)-electron total energies,
  which handles diradicals and bond-breaking (where a single-reference
  N-electron description struggles) more naturally than particle-hole
  RPA or standard TDDFT. O(N⁴) with a Davidson solver; active-space
  truncation is the accuracy-preserving speed-up PySCF itself uses.

#### Explicitly not proposed (already covered elsewhere)

Confirmed covered and left off: standard HF/DFT/MP2/CC/CASSCF/TDDFT
(all shipped), density fitting and RI (v0.7.x acceleration wave),
periodic boundary conditions (vibe-qc's other half from day one, not
something borrowed from PySCF), implicit solvation (CPCM, v0.12.0),
QM/MM (already blanket out-of-scope pre-`v1.0`), geometry optimisation
(shipped), CISD (already one of **AUTOCI1**'s `citype` values), and
molecular dynamics (already the **MD1**-**MD6** post-MPI-milestone bucket, GPU
acceleration already the generic `v1.x` item GPU4PySCF is one more
concrete instance of, alongside GauXC above).

### Psi4 sweep (open-source molecular reference, not a parity target)

A pass over Psi4's manual (`psicode.org/psi4manual/master/`), same
altitude and same framing as the PySCF sweep above: Psi4 is not a
parity target either (the intro's "Parity targets, in order" section
already covers this), and unlike PySCF it isn't even vibe-qc's
existing validation reference, it is a second open-source molecular
program whose specialty (perturbative intermolecular interaction
theory, and a cluster of post-HF flavors none of the other three
sweeps emphasized) happens to fill in gaps the ORCA/CRYSTAL23/CP2K/
PySCF sweeps did not surface. Nothing here is scheduled before `v1.0`.

#### SAPT: a genuinely different noncovalent-interaction decomposition

Three EDA flavors are already on this roadmap, **NCOV1** (ETS-NOCV),
**PR26** (LMOEDA/Hayes-Stone), and **ALMO2** (ALMO-EDA), and all three
are *supermolecular*: run the dimer and the fragments, subtract
energies. Symmetry-Adapted Perturbation Theory is a fourth, structurally
different approach, not a fourth flavor of the same idea, worth
tracking on its own rather than folding into the EDA cluster above.

* **SAPT1** SAPT proper: expand the dimer Hamiltonian around the
  monomer Fock operators and compute the interaction energy directly
  as a perturbation series, electrostatics and exchange at first
  order, induction and dispersion at second order and beyond, *without
  ever computing a monomer or dimer total energy*. No BSSE by
  construction, the supermolecular EDAs above all have to correct for
  it after the fact. Levels from **SAPT0** (cheap, large-system
  default) up through **SAPT2+(3)** (higher-order, smaller systems),
  plus a **SAPT(DFT)** variant using KS operators instead of HF ones.
* **SAPT2** F/I-SAPT (functional-group and/or intramolecular SAPT):
  partitions a single molecule into functional-group "monomers" to get
  a SAPT-style decomposition of an *intramolecular* interaction
  (conformational preferences, substituent effects), the same
  machinery aimed at a different partitioning question.

#### Explicitly correlated methods and basis-set-limit extrapolation

Two ways of getting basis-set-limit accuracy without a basis-set-limit
calculation, both missing today:

* **F121** MP2-F12: adds an explicitly correlated geminal term
  (Gaussian-type geminals approximating a Slater-type correlation
  factor f₁₂) that captures the electron-electron cusp directly,
  instead of relying on ever-larger basis sets to approximate it.
  Needs the orbital basis, a complementary auxiliary basis (CABS,
  **BS11** in the basis-set sweep above now covers it) and, for
  practical cost, density fitting. O(N⁵) with a
  heavy prefactor, RHF-reference only in Psi4's own implementation;
  pays for itself by removing most of the basis-set-extrapolation
  error **CBS1** below exists to estimate.
* **CBS1** Complete-basis-set extrapolation as a first-class utility,
  not a manual workflow: Helgaker two- and three-point (exponential,
  the SCF default), Truhlar (power-law), and Karton (root-power) two-
  point schemes, applied separately to the HF/SCF and correlation
  energy components since they converge at different rates. Exposed
  the way Psi4 does it, `energy("mp2/cc-pv[dt]z")` triggers the two
  underlying single-point calculations and extrapolates automatically,
  is the right API shape: `vq.run_cbs("mp2/cc-pv[dt]z")` reusing the
  `Compound`-style workflow runner (**DIAG4**/**CMP-** bucket above)
  rather than a bespoke extrapolation-only entry point.

#### Density Cumulant Theory and orbital-optimized perturbation/CC

* **DCT1** Density Cumulant Theory: parametrizes the density cumulant
  λ (the connected part of the two-particle density matrix) directly
  instead of wavefunction amplitudes, size-extensive and size-
  consistent by construction, with simpler N-representability
  constraints than optimizing a general 2-RDM outright (contrast
  **V2RDM1** below, which optimizes the full 2-RDM and pays for that
  generality with an SDP solve). ODC-12 is the balanced default;
  ODC-13(λ₃) adds a three-particle correction for open-shell accuracy.
* **OOPT1** Orbital-optimized MP2/MP3/MP2.5 and OLCCD: re-optimizing
  the orbitals against the correlated energy (not just the HF energy)
  fixes real, documented pathologies of the plain methods, artificial
  symmetry-breaking instabilities in restricted references, natural
  occupation numbers straying outside [0, 2] (an N-representability
  violation plain MP2 can produce), and discontinuous dipole/force
  surfaces where that symmetry-breaking kicks in along a coordinate.
  Directly relevant to vibe-qc's own transition-metal / open-shell
  convergence work above, this is the same failure mode DIIS/SOSCF
  fixes at the SCF level, one layer up at the correlated-method level.

#### v2RDM-CASSCF and Mukherjee state-specific multireference CC

* **V2RDM1** Variational 2-RDM-driven CASSCF (DePrince, Mazziotti):
  optimizes the two-particle reduced density matrix directly subject
  to N-representability conditions (enforced via a semidefinite-
  program solve) instead of expanding a full CI wavefunction, scaling
  polynomially with active-space size rather than exponentially.
  Trades some accuracy (the N-representability conditions used in
  practice are necessary, not sufficient) for reaching active spaces
  conventional CASSCF and even DMRG cannot, worth having as a third
  large-active-space route alongside **MR1** (ICE-CI) and **MR2**
  (DMRG interface) above, not a replacement for either.
* **MKMRCC1** Mukherjee state-specific multireference CC (Mk-MRCC):
  a state-specific alternative to the internally-contracted CASPT2/
  NEVPT2/MRCI stack already shipped and to ORCA's state-universal
  **MR3** (MR-EOM-CC) above. Uses the Jeziorski-Monkhorst ansatz, a
  separate excitation operator T^μ per reference determinant, feeding
  into an effective Hamiltonian; being state-specific rather than
  state-universal sidesteps the intruder-state problem multi-root
  state-universal MRCC formulations are prone to. CCSD and CCSD(T)
  variants; best-behaved for low-spin (singlet / open-shell-singlet)
  references, an explicit limitation worth inheriting rather than
  silently working around.

#### Distributed multipole analysis, and the many-body cluster expansion

* **GDMA1** Distributed multipole analysis (Stone): partitions the
  molecular multipole expansion into atom-centered (or bond-centered)
  multipole moments, rather than the single-origin expansion **PR5**
  above already covers. The natural source of physically-meaningful
  parameters for a polarizable force field or a GDMA-derived embedding
  potential, distinct from and complementary to **PR5**'s global
  quadrupole/octupole moments.
* **NBODY1** Many-body (N-body) expansion for clusters: decompose a
  cluster's total binding energy into 1-body (monomer relaxation),
  2-body, 3-body, ... contributions, each optionally counterpoise-
  corrected (CP, NoCP, or Valiron-Mayer Function Counterpoise), and
  optionally computed at a *cheaper* method past some body-order
  cutoff (`supersystem_ie_only`-style truncation: CCSD(T) for 2-body,
  MP2 or SCF for everything past it). This is not a generic borrowed
  feature, it is directly on-mission: the cyclic cluster model is
  vibe-qc's original motivating idea, and a many-body decomposition of
  an embedded-cluster binding energy is exactly the diagnostic that
  would show whether a given CCM truncation radius is capturing the
  cooperativity that matters, or truncating before it converges.
  Worth prioritizing above the rest of this sweep for that reason.

#### Explicitly not proposed (already covered, out of scope, or superseded)

* **Effective Fragment Potential (EFP)**: an ab initio-derived
  polarizable force field for representing a solvent/environment
  region, close enough to the QM/MM family that the existing blanket
  "QM/MM and embedding, other flavors deferred" exclusion already
  covers it.
* **FNOCC (frozen natural orbitals) and CEPA**: CEPA is a superseded
  predecessor to the DLPNO-CCSD(T) stack already shipped; frozen
  natural orbital truncation is one of several NO-truncation schemes,
  DLPNO's PNO domains already cover the same idea in the variant
  vibe-qc uses.
* **Interfaces to other QC programs** (Psi4's CFOUR and MRCC
  interfaces): correctly not a pattern to copy, § 10 forbids shelling
  out to another QC program as a runtime dependency; where vibe-qc
  wants the same physics (arbitrary-order CC, **MRCC-lineage** ideas)
  it goes in as a native implementation, as already noted for
  Kállay's methods elsewhere in this document.
* **ADC** and **scalar-relativistic Hamiltonians**: already **ADC1**
  (PySCF sweep) and **REL1**-**REL3** (ORCA sweep) respectively.
* **TDSCF (real-time TDHF/TDDFT)**: already **RTP1** (CP2K sweep).
* **Diatomic spectroscopic constants** (Dunham analysis from a PES
  scan): real and cheap, but a post-processing utility over machinery
  (PES scans, vibrational analysis) already shipped, not worth a
  numbered item; note it here so it isn't independently rediscovered.

### NWChem sweep

A pass over NWChem's manual (`nwchemgit.github.io`), same altitude and
framing as the CP2K/PySCF/Psi4 sweeps: not a parity target. Smaller
yield than the previous four sweeps, which is expected six sweeps in,
most of NWChem's surface (DFT, MP2, CCSD, MCSCF, GW, XTB semiempirical,
QM/MM, real-time TDDFT, relativistic all-electron approximations,
VSCF, FCIDUMP export) is already covered above, confirmed by name
before writing this section rather than assumed. One finding folded
into the existing tensor-engine refactor proposal instead of listed
here (TCE, see above); the rest follows.

#### Non-equilibrium solvation for vertical excitations (extends CPCM + TDDFT)

* **VEM1** Vertical Excitation/Emission (VEM) model: split the
  continuum reaction field into a fast (electronic) component that
  re-equilibrates with the excited-state density, and a slow (nuclear/
  orientational) component frozen at the ground-state equilibrium.
  Equilibrium CPCM (already shipped, v0.12.0) implicitly assumes the
  *whole* medium re-equilibrates with whichever state is being
  computed, which is the wrong physics for a vertical transition (the
  solvent nuclei have not had time to move). Needed for quantitatively
  correct solvatochromic shifts on top of the TDDFT stack already
  shipped; a physically-motivated refinement of existing machinery,
  not a new solvation model to maintain alongside CPCM.

#### A third solvation theory, alongside continuum and (declined) explicit-solvent

* **RISM1** 1D reference interaction site model (RISM): a statistical-
  mechanical integral-equation theory of solvation, site-site
  correlation functions and a closure relation (HNC or a Gaussian
  approximation) in place of either a dielectric continuum (CPCM) or
  explicit solvent coordinates (SOLVATOR-style, already declined as
  out of core scope above). Sits at a genuine middle point, molecular-
  level solvent structure without paying for explicit-solvent sampling,
  worth having as a third option once CPCM's limits (structured
  solvents, specific first-shell effects) actually bite, not before.

#### A second diabatization route for electron-transfer couplings

* **ETRAN1** Corresponding Orbital Transformation: construct diabatic
  charge-localized states from a pair of user-supplied, pre-localized
  molecular orbital sets (verified via Mulliken population, not
  constraint-enforced), then compute the Marcus-theory electronic
  coupling V_RP between them. A lighter-weight, less automatic
  alternative to **CDFT1** above (Psi4 sweep), which enforces the same
  localization through an optimization constraint instead of manual
  orbital preparation, worth having as a cheap cross-check for
  **CDFT1** rather than a primary route on its own.

#### A wavefunction-based composite thermochemistry recipe (extends the 3c-family)

* **CCCA1** Correlation-consistent Composite Approach (ccCA):
  E = ΔE(MP2/CBS) + ΔE(CCSD(T) correction) + ΔE(core-valence) +
  ΔE(scalar-relativistic) + ΔE(ZPE), a fixed multi-step wavefunction
  recipe in the Gaussian-n / CBS-QB3 / W1 tradition. Genuinely
  different in kind from the **3c-** family already shipped/planned
  above (a single DFT point plus a dispersion/gCP/basis correction):
  ccCA is a stack of *separate* single-point jobs at increasing levels
  of theory, closer to a `Compound`-script recipe (**CMP-** bucket
  above) than to a one-shot method flag. Main-group scope only, by
  design, the same limitation the original method carries.

#### Explicitly declined: pure plane-wave DFT (NWPW: PSPW / Band / PAW)

Considered and deliberately not proposed, worth saying why rather than
silently omitting it. NWChem's `nwpw` module is a genuine pure-
plane-wave, pseudopotential-basis periodic DFT engine (Γ-point PSPW,
multi-k-point Band, gamma-point PAW), architecturally unrelated to
vibe-qc's Gaussian-basis periodic code the way GPW/GAPW's Gaussian-
plus-auxiliary-plane-wave-grid approach is not: vibe-qc represents
every crystalline orbital as a linear combination of Bloch functions
built from local (Gaussian) functions from the ground up (`AGENTS.md`'s
architecture description), and a real plane-wave engine would mean a
second periodic SCF stack with no shared integral code, no shared
basis-set infrastructure, and its own pseudopotential-projector
machinery, not an extension of
anything that exists today. That is a different product, not a
roadmap item; the closest thing already on this roadmap, the GPW/GAPW
Gaussian-augmented route, gets the practical benefit (a plane-wave
grid for the Hartree term) without that cost. Revisit only if a
concrete, specific user need for pure-plane-wave accuracy shows up that
GPW/GAPW genuinely cannot serve, and treat it then as the § 9 "grand
refactor" conversation it would actually be.

### Long-term documentation goal, the Lecture Series

A multi-year goal beyond the existing tutorial set: an **educational
tour through computational chemistry**, structured as a lecture
series, that introduces every concept from first principles and uses
vibe-qc as the running "lab notebook." The current 25 tutorials are
*task-oriented* ("how do I run an open-shell DFT calculation?"); the
lecture series is *concept-oriented* ("what is exchange-correlation,
why does the Kohn-Sham mapping work, and what does PBE actually
parameterize?"). They complement each other rather than overlap.

**Audience.** Graduate students in chemistry / physics / materials
science, advanced undergraduates with a quantum-mechanics background,
researchers from adjacent fields (computational biology, materials
informatics, ML+chem) who need a working understanding of what an
electronic-structure code is doing under the hood, not just how to
drive one.

**Differentiation from existing material.** Textbooks (Szabo-Ostlund,
Helgaker-Jørgensen-Olsen, Martin) are excellent but offer no
runnable code. Online courses (TCCM-EJD, MolSSI BootCamps) are good
introductions but don't go deep enough on solid-state. CRYSTAL /
ORCA / Gaussian have manuals, not curricula. The lecture series fills
the "I want to *understand* what's happening when I press SCF, with
code I can run alongside" gap.

**Indicative outline** (subject to revision when we actually start
writing, currently a backlog item, not a committed structure):

1. *Why electronic structure?*, what we're computing and why,
   Born-Oppenheimer, the many-body problem
2. *Hartree-Fock from scratch*, Slater determinants, antisymmetry,
   Roothaan-Hall, SCF as a fixed-point iteration
3. *Basis sets*, Gaussians vs Slaters, contracted-Gaussian
   construction, polarization, diffuse functions, cc-pVxZ extrapolation
4. *DFT and the Kohn-Sham mapping*, Hohenberg-Kohn theorems, why
   non-interacting reference, exchange-correlation functionals,
   Jacob's ladder
5. *Numerical integration*, Becke fuzzy partitioning, Lebedev
   quadrature, the radial grid choice
6. *Open shells*, UHF / ROHF / spin contamination / why DFT often
   "does the right thing" anyway
7. *Periodic systems*, Bloch's theorem, k-points, Brillouin zone,
   density of states
8. *The Coulomb problem in 3D*, why the periodic Madelung sum is
   conditionally convergent, Ewald decomposition, ω-invariance
9. *Tight-cell pathologies*, what breaks when atoms are close,
   periodic Becke partition, level shifts, smearing for metals
10. *Symmetry exploitation*, space groups, Bloch sums symmetrized
    over the little group, IBZ k-mesh reduction
11. *Beyond the mean field*, MP2, coupled cluster, the CC hierarchy
12. *Local correlation methods*, DLPNO, RI / density fitting, why
    they make canonical CCSD(T) usable
13. *Periodic correlation*, CRYSCOR, periodic local CC, the
    cyclic cluster model

Each lecture is ~30-45 minutes of reading + 15 minutes of running
the embedded vibe-qc examples. The reader leaves with both the
theory and a working `examples/lecture_NN_*.py` they can modify.

**Cadence.** This is a multi-year project, not a v0.5 deliverable.
A reasonable starting target is one lecture per quarter (so the
full curriculum lands roughly aligned with v1.0). Earlier lectures
benefit from earlier publication, we don't need to write them
in order.

**Status:** queued, not started. The existing 25 tutorials and 15
theory sections are prerequisites, they cover the "lab notebook"
side and the equation references the lectures will lean on.

## Upstream feature requests

These are features we'd like to see in *external* tools that
vibe-qc users rely on. Tracked here so we don't lose them; cross-
referenced from the relevant docs page so users discover the
limitation + the upstream issue together.

| Tool | Feature | Why it matters | Status |
|---|---|---|---|
| [MolTUI](https://github.com/kszenes/moltui) | Periodic / crystalline-orbital support, read lattice vectors from CUBE (or new periodic format), render unit-cell box, replicate atoms across PBC, optionally support k-labeled Bloch orbitals | Lets users SSH into a remote machine + triage a periodic-SCF result the same way they triage a molecular one. Currently MolTUI is molecular-only; periodic users have to drop to VESTA / XCrySDen / matplotlib plots. | proposed; track at [github.com/kszenes/moltui](https://github.com/kszenes/moltui) |
| MolTUI | Vibe-qc's `.molden` extension for vibrational data, confirm round-trip reading | Lets users view normal modes via `moltui my.molden`; works in principle since MolTUI parses the `[FREQ]` block. Engineering tracks the round-trip test. | needs verification once vibe-qc Hessian → molden is in CI |
| ASE | First-class PySCF wrapper in `ase.calculators.pyscf` | Today third-party `ase-pyscf` exists but isn't pinned; we ship a built-in shim in `vibeqc.benchmark.PySCFCalculator` to avoid the extra dep. An upstream wrapper would let us drop the shim. | low priority, our shim is small + maintained |

## Licensing + redistribution

vibe-qc is MPL 2.0; the linked-library and bundled-data
licensing inventory is at [`docs/license.md`](license.md). Two
roadmap items address known caveats in that inventory:

### `v0.x.x`, FFT backend abstraction (commercial-friendly binaries)

vibe-qc currently dynamic-links FFTW3 (GPL v2 or later) for the
FFT-Poisson reciprocal-space Hartree solver. **The combined
binary is therefore effectively GPL v2** when redistributed,
fine for academic users running their own data, restrictive for
anyone wanting to embed vibe-qc in a closed-source product or
ship a non-GPL precompiled distribution.

Plan: add a CMake build-time option `VIBEQC_FFT_BACKEND={fftw3, pocketfft, kissfft}`
and abstract the four FFTW3 calls in
[`cpp/src/fft_poisson.cpp`](https://github.com/vibe-qc/vibe-qc/blob/main/cpp/src/fft_poisson.cpp)
behind a thin internal interface. Default stays FFTW3 (best
performance). [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft)
(BSD 3-Clause) and [KissFFT](https://github.com/mborgerding/kissfft)
(BSD 3-Clause) cover the GPL-free binary distribution case.
Estimated effort: ~1 day for the abstraction + parity tests on
the 3D-periodic SCF benchmarks. Tracked as **post-v1.0** since
no current user has flagged it as blocking; surface here so the
intent is visible.

### Shipped in v0.17.1: on-demand BSE basis fetcher

**Shipped** as `vibeqc.basis_fetch`, wired into
`run_job(..., fetch_from_bse=True)`, molecular only: `run_periodic_job` has no
`fetch_from_bse`. See [basis sets](user_guide/basis_sets.md) and the
[feature matrix](features.md).

It sits under licensing because the decision it records is a licensing one,
and that decision went the other way from the pre-split proposal. The proposal,
which gated on the `basissetdev` branch that was never transferred, is in
[roadmap history](roadmap_history.md#shipped-in-v0171-on-demand-bse-basis-fetcher).
Two of its premises do not survive contact with what shipped:

* **It is not a network fetch.** The `basis-set-exchange` distribution ships
  the whole catalogue as package data (~334 MB, 4085 files) and answers with
  sockets disabled. So there is nothing to pull "on first use", the result is
  pinned by the *package version* rather than by what a server serves, and it
  works on an air-gapped fleet host. The XDG cache is still used, but only
  for the rendered `.g94` / `.ecp`, not to avoid a download.
* **The licensing argument no longer carries the proposal.** Its "avoids any
  per-basis-set redistribution clearance question" rationale was settled the
  other way by `docs/license.md` § 3a: basis parameters are numerical data,
  not creative expression, BSE states this itself, and vibe-qc has since
  bundled 107+ BSE-fetched sets on exactly that basis. What remains is wheel
  size, and reach: **BSE 0.12 catalogues 776 sets against the 255 bundled**,
  so ~595 are otherwise unreachable.
* Consequently the dependency is the optional `[bse]` extra, imported lazily,
  and the default install is unchanged.

## What's explicitly out of scope (before v1.0)

- **QM/MM and embedding**, CCM itself is an embedding method for
  solids; other flavors deferred.
- **Coupled-cluster beyond the shipped dense CC3 and full-CCSDT benchmark
  routes**, including CCSDTQ and FCIQMC, remains outside normal production.
- **Non-adiabatic methods** (CASSCF + SOC, surface hopping), excited
  states are covered through TDDFT / NEVPT2 for practical
  ground-state work.
- **Relativistic methods** (DKH, X2C, spin-orbit coupling), important
  for heavy-element chemistry, but a specialised stack of its own.
  Tracked as a post-1.0 follow-on; ECPs (Phase 14) cover most heavy-
  element chemistry without it.
- **Distributed dense linear algebra**, such as ScaLAPACK-backed
  Fock diagonalisation, is not part of the first MPI milestone unless
  replicated-data benchmarks prove it is the blocking memory wall.

These aren't ruled out forever, they're simply not prerequisites for
a "standard QC program" per v1.0's scope, nor for CCM in v2.0.

**Previously listed as out of scope, now in scope after the parity
audit:**

- Dispersion corrections (DFT-D3, DFT-D4), too central to defensible
  DFT to omit. Now Phase **D1/D2** in v0.5.0.
- Implicit solvation (CPCM/COSMO), too central to organic / aqueous
  chemistry to omit. Now Phase **S1** in v0.5.0.
