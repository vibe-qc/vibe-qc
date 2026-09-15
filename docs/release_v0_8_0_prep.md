---
orphan: true
---

# v0.8.0 release-day docs sprint, checklist + pre-staged copy

```{note}
This is a historical pre-split release checklist for the archived monorepo.
Its co-located component paths and old branch/tag instructions are not a
current setup recipe. Use [release process](release_process.md) and
[toolset lifecycle](toolset_lifecycle.md) for the independent repositories.
```

This page is the **docs chat's release-day playbook for the
v0.8.0 vibe-qc release**. Marked `orphan: true` so it doesn't
appear in the toctree; intended for the release / docs chat to
consult on the day v0.8.0 is tagged.

```{warning}
**Historical — v0.8.0 shipped 2026-05-17 (codename *Grimme's
Gecko*, tag `050d2a2`); v0.8.1 followed 2026-05-18.** This page
is kept as the copy-and-adapt template for future per-tag
sprints (see `docs/release_process.md` § Documentation cadence)
and as a record of the v0.8.0 sprint.

**Do NOT paste the pre-staged copy below verbatim.** Several
blocks were drafted *before* the tag-day scope decision and are
now stale:

* **Block 2** ("Periodic SCF chain — production-grade in
  v0.8.0") is wrong — periodic compcell-GDF + multi-k GDF +
  open-shell GDF were **deferred to v0.9.0** at tag time
  (commit `ad9d95e`). v0.8.0's actual headline is molecular
  methods polish, double hybrids, direct SCF, D4/EEQ, and the
  grid / XC / basis-ECP library expansions.
* **Block 3** lists the DF integral-kernel SIGSEGV as
  "carries" — but the promoted CHANGELOG `[v0.8.0]` "Fixed"
  section records it as **closed** by the molecular-methods
  wave-1.8 work. It also references `run_uhf_periodic_gamma_gdf`,
  which does not exist (open-shell GDF is v0.9.0).

Sprint status as of 2026-05-20: CHANGELOG `[Unreleased]` →
`[v0.8.0]` promotion **done** (`af9338a`); homepage warning
admonition given correctness fixes (`bdc3102`) but its full
editorial re-curation + the green-admonition replacement are
still open. When adapting this page for v0.9.0, re-derive every
pre-staged block from the actual tag-day state.
```

## Pre-flight (before tagging)

Confirm:

1. **`CHANGELOG.md`** `[Unreleased]` block is the v0.8.0 release-
   notes draft (current state: 9 sections covering the 10 v0.8.0
   roadmap entries + Fixed + Open lists; ~306 lines added in
   commit `7cddd5a`). On tag day:
   - Promote the `[Unreleased]` content to `## [v0.8.0], 
     <YYYY-MM-DD>, *<Codename>*, periodic SCF chain + basis +
     functional + JKBuilder + vqfetch (structures + CCCBDB) + asbestos`
   - Replace `[Unreleased]` with the v0.8.x maintenance banner
     (template below)
   - Pick the codename from the candidates spread across roadmap
     entries (Bloch's Gnu, Wannier's Marmoset, Brillouin's Gerbil,
     Pople's Beaver, Becke's Hummingbird, etc.)

2. **`docs/roadmap.md`** v0.8.0 entries, flip status markers from
   "in flight" / "queued" / "pending" to "shipped" / "✓" for
   anything that's actually in the v0.8.0 cut. Anything that
   slipped → move to a v0.8.1 / v0.9.0 section. Roadmap currently
   has **10 parallel v0.8.0 sibling entries**; release chat
   triages.

3. **`docs/index.md`** homepage admonitions, swap to the
   pre-staged v0.8.0 versions below.

4. **`pyproject.toml`** version bump to `0.8.0`.

5. **`CITATION.cff`** version + `date-released` bump.

6. **`docs/citing.md`** version-bump in the BibTeX example.

7. **README.md**, verify the headline-feature paragraph mentions
   the v0.8.0 deliverables (currently says v0.6.x).

## On tag day, order of operations

1. Engineering pushes `v0.8.0` tag on `release` branch (per
   `docs/release_process.md`). Wait for the tag to be visible.
2. Docs chat applies this checklist's flips in **one commit per
   surface** so each is independently reviewable + revert-able.
3. Confirm `vibe-qc.com` rebuilds via the docs-deploy CI.
4. Verify each admonition renders correctly (live).
5. Refresh the paper-input package's file 04 proof table from
   the live regression suite (`python examples/regression/run_suite.py
   --all --output-md`); paper-writer chat re-pulls.

## Pre-staged copy: homepage admonitions for v0.8.0

### Block 1, current "release banner" admonition (already auto-substituted via MyST {{release}} / {{codename}}; verify the substitutions resolve correctly)

The MyST substitution block at the top of `docs/index.md` reads:

```markdown
\`\`\`{admonition} Current release: {{release}} — *{{codename}}*
:class: tip
…
\`\`\`
```

These resolve from `conf.py`'s `myst_substitutions` (which in turn
read from `pyproject.toml` + the codename catalog in
`docs/roadmap.md`). After the `pyproject.toml` bump + the codename
pick, this admonition auto-updates on rebuild. **No manual edit
needed** unless the substitution mechanism breaks.

### Block 2, replace the v0.7.0 "Periodic-SCF gauge fix landed" green admonition

**Currently** (search `Periodic-SCF gauge fix landed` in
`docs/index.md`):

> ✅ Periodic-SCF gauge fix landed in v0.7.0
>
> The v0.6.x Madelung self-image leak, periodic SCF systematically
> over-bound in the molecular limit by an `α_M·(Q_n² + Q_e²)/(2L)`
> shift, is fixed in v0.7.0 (codename *Löwdin's Compass*). … v0.7
> also ships a comprehensive linear-dependence + screening
> program …

**Replace with** (v0.8.0 headline):

> ✅ Periodic SCF chain, production-grade in v0.8.0
>
> v0.8.0 (codename *<TBD>*) lands the periodic SCF chain at
> production scale: **Gaussian density fitting (GDF) is now the
> default periodic J/K builder**, matching PySCF.pbc.GDF to
> machine precision (ΔE = −6.8 × 10⁻¹² Ha on MgO conventional
> rocksalt / sto-3g / Γ RHF; 20 of 23 systems × methods × bases
> at sub-µHa). **Open-shell periodic UHF and UKS** ship via
> Γ-only GDF. **Multi-k SCF** (KRHF + KRKS) is in tree; the
> LiH/pob-TZVP/[8,8,8] sweep targets the 2013 paper SI Table 2
> reference (−8.149050 Ha/cell). The native-GDF Lpq builder
> drops the PySCF integral dependency.
>
> v0.8.0 also ships the **basis-set library expansion** (87 new
> BSE-fetched bases bundled, 142→236 .g94 files, full libecpint
> end-to-end ECP integration, the latent-since-2013 pob S
> d-polarisation column-swap fix), the **XC functional library
> expansion** (RSH machinery for ωB97X-V/M-V/HSE06; PW1PW
> custom hybrid; r²SCAN family; D4 dispersion), the
> **molecular methods polish** (`JKBuilder` polymorphic
> dispatch unifying molecular + periodic-Γ J/K; RIJCOSX SCF +
> analytic gradient validated to <0.13 mHa vs ORCA on
> glycine/def2-TZVP; B3LYP corrected to libxc-475 / VWN5 /
> Becke 1993 convention), the **`vqfetch` external-data
> integration** (OPTIMADE / Materials Project / COD / NOMAD
> structure pulls + NIST CCCBDB experimental references with
> full provenance), and the **vq queue v0.4+** with cgroup-v2
> enforcement, web dashboard, pause/resume, and multi-venv
> dispatch. See
> [CHANGELOG](https://vibe-qc.com/docs/)
> for the full release notes.

### Block 3, replace the v0.7.x "Open in the v0.7.x maintenance window" warning admonition

**Currently** (search `Open in the v0.7.x maintenance window`
in `docs/index.md`): a long warning block
naming 8 open issues from the v0.7.x window.

**Note on closures vs carryovers from the v0.7.x → v0.8.x window
transition** (informational; not part of the admonition):

* **Closes at v0.8.0** (drop from the warning):
  * Tight-ionic multi-cell density bug, closed by the GDF/
    multi-k path landing.
  * `pob-TZVP` missing Ne, closes when `fix/pob-tzvp-add-ne`
    merges (release chat owns the merge).
  * vDZP / dhf-* / x2c-* manual `ecp_centers` boilerplate, 
    closes when basissetdev's Phase 14e (`auto_ecp_centers`)
    merges. **Conditional on the basissetdev merge decision**;
    if basissetdev is held out per the user's standing rule,
    this bullet stays in the v0.8.x admonition.
* **Carries from v0.7.x → v0.8.x** (kept):
  * Si/C-diamond mHa drift, multi-k Lpq RAM ceiling, DF
    gradient on large bases, heavy-atom basis-load OOM,
    composite-3c methods, hexagonal-lattice FFT-Poisson,
    analytic-gradient-with-f-shells, B3LYP behaviour-change
    informational, `build_xc_periodic` V_xc gauge bug, DF
    integral-kernel SIGSEGV.
* **New at v0.8.0** (added below):
  * compute-reference libint ERI2/ERI3 deficit (DF segfault on L≥1 aux
    on compute-reference specifically).
  * `ProgressLogger.warning` crash on
    `auto_optimize_truncation` non-convergence.

**Replace with** (v0.8.x maintenance window):

> ⚠️ Open in the v0.8.x maintenance window
>
> Eleven known issues still gating the next major release. All
> are caught by the regression suite at
> [`examples/regression/`](https://vibe-qc.com/docs/)
> or fail loudly at runtime, so a calculation that hits them
> doesn't silently produce wrong numbers.
>
> * **Si-diamond × 3 + C-diamond / PBE / pob-TZVP** converge to a
>   different stationary point than PySCF (~mHa). Suspected
>   DIIS-tightness difference, not algorithmic. **Workaround**:
>   stick with sto-3g for these systems, or tighten DIIS.
>
> * **Multi-k dense `Lpq[k1, k2]` RAM ceiling**. Production-scale
>   multi-k periodic SCF with dense Lpq storage hits multi-TB
>   RAM on real systems (lizardite/[2,2,2] ≈ 553 GB;
>   antigorite-m17/[2,2,2] ≈ 5 TB). **Workaround**: small system
>   × full kmesh OR full system × small kmesh, never both.
>   Permanent fix: streaming-Lpq queued for v0.8.x.
>
> * **~~DF gradient on large bases~~**, FIXED in v0.8.0. The
>   `density_fit=True` gradient was ~115 mHa/bohr off on
>   glycine / def2-TZVP / RHF due to a libint engine-state leak
>   in `compute_3c_eri_gradient_weighted`.
>
> * **~~Analytic RHF/UHF/RKS/UKS gradient with f-shells~~**, 
>   FIXED in v0.8.0. The direct 4-index ERI gradient kernel had
>   a buggy libint `DerivMapGenerator` routing for high-l
>   mixed-l shell quartets (Fix C: canonical 1/8 + l-canonical
>   reorder rewrite). Post-fix, H2CO/def2-tzvp matches PySCF to
>   ~5e-11 Ha/bohr.
>
> * **`build_xc_periodic` native periodic V_xc has O(10 Ha)/atom
>   error**. Loud, not silent. The default `run_rhf_periodic_gamma_gdf`
>   driver routes around it via the PySCF Becke grid + libxc path.
>   Affects only callers who invoke `build_xc_periodic` directly,
>   or who use the legacy `run_uks_periodic_gamma_ewald3d` driver.
>   **Workaround**: use `run_rhf_periodic_gamma_gdf` /
>   `run_uhf_periodic_gamma_gdf` / `run_uks_periodic_gamma_gdf`.
>
> * **DF integral-kernel SIGSEGV**, `vq.compute_2c_eri` and
>   `vq.compute_3c_eri` segfault on any aux basis containing
>   `l ≥ 1` shells, before SCF starts. The crash manifests through
>   DF SCF because that's what most users hit first.
>   **Workaround**: only s-only aux bases (sto-3g, sto-6g, 6-31g,
>   …) are safe in the DF / ADFT path until the fix lands on
>   `feature/density-fitting`. Tighter reproducer at
>   [`examples/experimental/coulomb-method-zoo/bug_compute_2c_eri_pshell.py`](https://vibe-qc.com/docs/).
>
> * **compute-reference libint ERI2/ERI3 deficit** *(new at v0.8.0)*. The
>   compute-reference libint install was built **without** ERI2 + ERI3
>   support; any DF feature on aux containing L≥1 functions
>   (essentially all production aux bases) segfaults on compute-reference.
>   Confirmed by inspecting `third_party/libint/install/include/libint2/config.h`
>   on compute-reference. Fresh laptop builds are unaffected. **Fix**:
>   rebuild compute-reference libint via `bash scripts/build_libint.sh`
>   from current `third_party/libint/` source (which includes the
>   ERI2/ERI3-enable patches from `2ef9682`); one-time, benefits
>   every dev tree on compute-reference. **Workaround until rebuilt**: don't
>   route DF features through `vq submit` to compute-reference.
>
> * **`ProgressLogger.warning` crash** *(new at v0.8.0)*, 
>   `python/vibeqc/periodic_rhf_ewald.py:406` calls
>   `plog.warning(...)` on a `ProgressLogger` that has no
>   `warning` attribute. Fires when `auto_optimize_truncation`
>   doesn't converge. **Workaround**: pass
>   `auto_optimize_truncation=False` to the periodic SCF call.
>   1-line fix queued for v0.8.x.
>
> * **Heavy-atom basis-load test** cumulative-OOMs the laptop on
>   the full 366-case batch. Affects only the basisset-dev test
>   pass; user-facing SCF is unaffected. **Workaround**: route
>   via `vq submit` to compute-reference (after the libint rebuild above),
>   not on the laptop.
>
> * **Composite "3c" methods** (r²SCAN-3c, B97-3c, ωB97X-3c), 
>   basis carriers ship (`def2-mTZVP/mTZVPP`, `vDZP`) but gCP +
>   D3/D4 aren't wired through the molecular runner.
>   **Don't promise turnkey 3c yet.** **Workaround**: use
>   `def2-TZVP` + matching ECP + D3(BJ) until v0.9.0
>   composite-methods milestone.
>
> * **Hexagonal lattice support**, the native FFT-Poisson kernel
>   now supports skew cells via the full reciprocal metric, but
>   production periodic parity for tight ionic crystals is still
>   tracked under native FFTDF/GDF benchmarking.
>
> * **B3LYP behaviour change at v0.8.0** *(informational, not a bug)*.
>   The `b3lyp` keyword now resolves to libxc id 475 (VWN5, the
>   ORCA / ADF convention) rather than id 402 (VWN-RPA / VWN3, the
>   libxc default and Gaussian convention). Effect: B3LYP energies
>   move by ~10-15 mHa per heavy atom, a noticeable shift for
>   users comparing pre- vs post-v0.8.0 numbers. **For the
>   Gaussian-compatible VWN3 variant** (e.g. for PySCF parity), pass
>   `functional="b3lyp/g"`, the new alias mirroring ORCA's `B3LYP`
>   / `B3LYP/G` keyword pair.

### Block 4, green funding admonition (small refresh)

The funding admonition (search
`vibe-qc is funded by individual sponsors` in `docs/index.md`)
(GitHub Sponsors +
Ko-fi + bigger hardware + NIST SRD 3) is mostly unchanged. **One
addition for v0.8.0**: insert a "**cluster guest accounts**"
sentence near the existing "bigger hardware" sentence, mirroring
the new
{ref}`Compute time on your cluster <compute-time-on-your-cluster>`
section in `docs/support.md` (added 2026-05-10, commit
`4ed5f04`). One-line insertion; keep the admonition's overall
length unchanged.

Verify the URLs still resolve.

## Landing-page rework, beyond admonition swaps

The four-block admonition swap above keeps the top-of-page
banners current, but **v0.8.0 is a large enough release
that the body of `docs/index.md` also needs a once-over**.
Three sections need attention beyond the admonition flips:

### 1. Intro paragraph (top of `docs/index.md`, just above the admonitions)

> **Position note.** The intro paragraph was moved to the top
> of the page (right after the "**Quantum chemistry for
> molecules and solids.**" tagline, before the admonitions)
> in commit pending, so a landing user sees "what is this
> thing?" before the release-banner / funding / open-bugs
> admonitions. The paragraph is now between the tagline and
> the four-block admonition stack.

**Currently** ends with:

> The periodic stack delivers 1D / 2D / 3D Hartree-Fock and
> Kohn-Sham DFT with Monkhorst-Pack k-meshes, Ewald-summed
> Madelung, band structure and density of states, and is
> **growing toward CRYSTAL-style crystalline-orbital
> calculations at full-SCF accuracy**.

The "growing toward" framing is wrong as of v0.8.0, multi-k
KRHF + KRKS + open-shell UHF/UKS landed. **Replace with**:

> The periodic stack delivers 1D / 2D / 3D Hartree-Fock and
> Kohn-Sham DFT with Monkhorst-Pack k-meshes (closed- and
> open-shell), Gaussian density fitting, hybrid functionals
> (B3LYP, PBE0, PW1PW), Ewald-summed Madelung, band structure
> and density of states. Reference benchmarks land at
> machine precision against PySCF.pbc.

### 2. "Capabilities today" section (search `## Capabilities today`)

**Currently** lists three buckets: Molecular / Periodic /
Tooling. v0.8.0 adds material to **all three**.

**Molecular bucket**, append:
- `JKBuilder` polymorphic dispatch unifying molecular and
  periodic-Γ J/K (RIJ + RIK + RIJK + RIJCOSX, validated to
  <0.13 mHa vs ORCA on glycine/def2-TZVP).
- B3LYP follows the ORCA convention, `b3lyp` → libxc-475 /
  VWN5 (was libxc-402 / VWN3); new `b3lyp/g` alias → libxc-402
  for the Gaussian-compatible VWN3 variant (note: ~10-15 mHa
  shift per heavy atom vs the pre-v0.8.0 behaviour).
- D4 dispersion alongside the existing D3(BJ).

**Periodic bucket**, replace the Γ-only RHF/RKS framing
with:
- Multi-k KRHF + KRKS (Monkhorst-Pack with IBZ reduction).
- Open-shell periodic UHF + UKS (Γ-only via GDF).
- Hybrid functionals at the periodic level (B3LYP, PBE0,
  PW1PW, Bredow's 20%-HF + PW91 hybrid).
- Native-GDF Lpq builder (drops the PySCF integral
  dependency on the dominant code paths).

**Tooling bucket**, append:
- `vq` v0.4 job queue, submit vibe-qc runs to a remote
  cluster with cgroup-v2 enforcement, web dashboard,
  CRYSTAL14 dispatcher.
- `vqfetch` external-data integration, pull crystal
  structures from OPTIMADE / Materials Project / COD /
  NOMAD with full provenance.
- `examples/regression/` suite (since v0.7.2 *Boys'
  Crucible*) with v0.8.0 coverage matrix expansion.

### 3. "Status" section (search `## Status`)

**Currently** says:

> vibe-qc is pre-release software. Molecular HF/DFT/MP2 is
> stable and production-ready for small-to-medium systems;
> **periodic calculations converge cleanly in the
> molecular-limit regime and are being hardened for
> tight-cell bulk work**.

Reframe, molecular limit was the v0.7 story. v0.8.0
delivers production multi-k periodic SCF with sub-µHa
PySCF.pbc parity:

> vibe-qc is pre-release software. The molecular stack
> (HF/DFT/MP2 + analytic gradients + dispersion) is stable
> and PySCF-validated to machine precision. The periodic
> stack delivers production-grade multi-k SCF (closed- and
> open-shell, hybrid functionals, native GDF) with sub-µHa
> parity to PySCF.pbc on the in-tree benchmark suite. Open
> v0.8.x maintenance items are listed in the warning
> admonition above; the [roadmap](roadmap.md) covers what's
> next.

### 4. Sanity-check the "Your first calculation" walkthrough

The water/PBE/6-31g\*/D3BJ/opt example (search the
`Save the following as` block in `docs/index.md`) is
unchanged-API. **Verification step**:
on tag day, actually run the snippet against the v0.8.0 build
and confirm the three output files (`water.out`,
`water.molden`, `water.traj`) still produce as documented.
Banner-format change (libecpint coupled fix) means
`water.out` will show the four-library banner, verify the
sample-output comments elsewhere don't show the old
three-library format inconsistently.

### Effort estimate (landing-page rework)

- ~15 min for the three text rewrites above (intro paragraph,
  capabilities buckets, status paragraph).
- ~5 min for the funding-admonition cluster-account sentence.
- ~5 min for the water-example sanity check.
- **Total: ~25 min**, on top of the four-block admonition
  swap. Folds into the ~2-hour tag-day docs sprint.

## Pre-staged copy: CHANGELOG `[Unreleased]` reset

After promoting the current `[Unreleased]` content to a
`## [v0.8.0]` section, replace `[Unreleased]` with:

```markdown
## [Unreleased]

> Heading toward v0.8.x maintenance + v0.9.0. v0.8.x will close:
> the multi-k Lpq RAM ceiling (streaming-Lpq); the Si/C-diamond
> mHa-drift vs PySCF (DIIS investigation);~~ the DF gradient
> disagreement on large bases (~115 mHa/bohr on glycine/def2-TZVP);
> the analytic-gradient-with-f-shells bug;~~ the compute-reference libint
> ERI2/ERI3 deficit (one-time rebuild); the
> `ProgressLogger.warning` crash on `auto_optimize_truncation`
> non-convergence. Both gradient bugs (DF and f-shell direct)
> **already closed in v0.8.0** — see the CHANGELOG Fixed entries.
```

## Pre-staged copy: roadmap.md "shipped" sweep

For each v0.8.0 entry in `docs/roadmap.md`:

1. Add a leading line at the top of each entry: `**Status:
   shipped in v0.8.0 (<YYYY-MM-DD>) as part of *<Codename>*.**`
2. Move any sub-items that *slipped* to a new v0.8.1 entry below.
3. Update the v0.7.x → v0.8.x maintenance entries to reflect what
   carries forward.

If asbestos polymorphs Paper 1 calculations are still in
flight at tag time (i.e. the `studies/asbestos-polymorphs/`
scaffolding shipped but Paper 1 results haven't been generated
yet), keep that entry's "shipped: scaffolding only; Paper 1
calculations in flight" framing, don't claim more than is
true.

## Pre-staged copy: paper-input package refresh

The paper-writer chat consumes `vibeqc-article/input-from-claude-code/`.
On the day v0.8.0 ships:

1. Refresh **`04_validation.md`** comprehensive method proof
   table from the live regression suite, re-pull MgO
   ΔE numbers, RIJCOSX max-|Δ| numbers, etc., against the
   actually-tagged build SHA.
2. Refresh **`02_methods.md`** § 8 (`JKBuilder`), verify the
   API names + factory functions match what shipped.
3. Refresh **`08_software_availability.md`**, version
   `0.8.0`, date, citation BibTeX.
4. Re-pull **`13_molecular_methods_handover_2026-05-09.md`**
   commit-SHA references, the 8 stacked branches' SHAs become
   merged-into-main commits at v0.8.0.
5. Notify the paper-writer chat that the package is current.

## Banner / libecpint visibility, coupled fix (IN v0.8.0)

> **Decision (Mike, 2026-05-10): in scope for v0.8.0.**
> Engineering extends `banner.py` to probe libecpint; docs
> side then refreshes the banner-example surfaces. Sequenced
> in the on-tag-day operations as a tag-day prerequisite
> (engineering side) + flip (docs side).

### The gap

- **`docs/release_process.md` lines 38-46** show the example
  banner as `Release v0.1.0` with only `libint 2.13.1 ·
  libxc 7.0.0 · spglib 2.7.0`. No libecpint, no current
  version. New readers might think libecpint isn't shipped
  (it's been bundled since v0.4.0 per the basissetdev chat
  + commit `add8163`).
- **`python/vibeqc/banner.py`**, `library_versions()`
  function (around line 247) probes only libint / libxc /
  spglib. **`libecpint` appears nowhere in `banner.py`**
  (verified: `grep libecpint python/vibeqc/banner.py`
  returns zero hits). So the **live banner** today *also*
  doesn't show libecpint. The "linked: …" line in the
  banner only lists three libraries, not four.

### Why this is a coupled fix

Updating the doc banner example to show libecpint **without**
first fixing `banner.py` to actually probe libecpint would
mean shipping a doc banner format that users can't reproduce
on their install, the live `vibeqc.print_banner()` would
keep showing only three libraries while the doc shows four.
**Both sides must land together in v0.8.0.**

### Sequenced fix order (within the v0.8.0 tag-day workflow)

1. **Engineering** (pre-tag prerequisite, ~30 min):
   - Extend `library_versions()` in
     `python/vibeqc/banner.py` to probe libecpint version
     alongside libint/libxc/spglib. ~10 lines of additive
     code.
   - Update the docstring at line 247 to add "libecpint" to
     the listed libraries.
   - Extend the banner format string in
     `vibeqc.banner.banner()` so the "linked: …" line
     includes libecpint.
   - Add a unit test asserting libecpint is probed.
   - Verify the live banner renders the four-library format
     before the v0.8.0 tag commit.
2. **Docs** (tag-day flip, ~10 min):
   - Update `docs/release_process.md` example banner
     blocks (lines 38-46 release-mode and the dev-mode block
     immediately below) to `v0.8.0` + the four-library
     format. Same edit shape as the v0.6.3 → v0.7.0 banner
     refresh that landed in `docs/updating.md`
     (commit `5d743a1`).
   - Run `grep -r "linked: libint" docs/ examples/` to find
     any other doc surface that reproduces the banner, 
     `docs/installation.md`, sample-output comments in
     example scripts, etc., and refresh each to the
     four-library format.

### Effort estimate

- Engineering side: ~30 min (function + docstring + format
  string + unit test).
- Docs side: ~10 min (sed-style multi-file refresh once
  banner.py lands).

**Total: ~40 min for the coupled fix.** Folds into the
~2-hour tag-day docs sprint cleanly; engineering side is the
hard prerequisite that gates the docs-side flip.

## What's NOT in this checklist

- **Engineering tasks** (merging the 8 molecular branches,
  rebasing basissetdev, resolving the tender-wright conflict,
  picking the v0.8.0 codename): release/engineering chat owns.
- **CI verification** (does the docs deploy succeed? does the
  banner render with the new version?): docs CI owns; verify
  by visiting https://vibe-qc.com/ after the deploy completes.
- **The asbestos paper or the JCC release paper**: separate
  publications on separate timelines; not part of the v0.8.0
  release-day docs sprint.

## Estimated wall-clock for the docs sprint

Assuming engineering has merged + tagged cleanly:

- CHANGELOG promotion + reset: ~10 min
- Homepage admonitions swap (3 blocks): ~15 min
- Landing-page rework (intro / capabilities / status /
  funding-admonition cluster sentence / water-example
  sanity check): ~25 min
- Roadmap "shipped" sweep (10 entries): ~30 min
- pyproject.toml + CITATION.cff + citing.md version bumps: ~5 min
- README.md headline refresh: ~10 min
- Paper-input package refresh: ~30 min
- Banner / libecpint coupled fix (engineering 30 min pre-tag
  + docs 10 min tag-day): ~10 min docs-side on tag day
- Verify the deploy + cross-link audit: ~30 min

**Total: ~3 hours** of focused docs-chat work on tag day
(+ ~30 min engineering pre-tag for the banner/libecpint
extension).

## After the dust settles

Items to queue for v0.8.x maintenance docs work:

- **`h8_chain_ccm_ancestor` tutorial (H-chain BvK) revival** when the multi-cell
  density bug closes, the small-spacing rows in the table
  collapse from mHa to µHa drift. Tutorial gets re-run; the
  "two regimes" framing becomes "BvK works to µHa across all
  spacings."
- **`docs/user_guide/density_fitting.md`** new chapter, covers
  the JKBuilder API, RIJCOSX, gCP wiring (when it lands),
  `run_*_scf_with_jk`. Currently no dedicated user-guide page.
- **`docs/user_guide/external_structures.md`**, vqfetch user
  guide; drafted in design-worktree
  `claude/awesome-noether-2de783`; should land on main if v0.8
  ships the fetcher.
- **`docs/user_guide/ecp.md`**, covers libecpint + ECPCenter +
  Phase 14e auto-population.
- **Cross-link audit** when v0.8.0 ships, every "post-merge"
  placeholder in the docs becomes a real cross-ref.
- **`docs/features.md` rebuild**, currently lags the v0.7.x
  state; refresh to v0.8.0 capabilities.
