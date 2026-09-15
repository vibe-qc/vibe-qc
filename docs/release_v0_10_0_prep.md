---
orphan: true
---

# v0.10.0 release-day docs sprint, checklist + pre-staged copy

```{note}
This is a historical pre-split release checklist for the archived monorepo.
Its co-located component paths and old branch/tag instructions are not a
current setup recipe. Use [release process](release_process.md) and
[toolset lifecycle](toolset_lifecycle.md) for the independent repositories.
```

This page is the **release-day playbook for the v0.10.0 vibe-qc
release**. Copy-and-adapted from `docs/release_v0_9_0_prep.md` per the
per-tag mechanical-sprint convention in
[`docs/release_process.md`](release_process.md) § "Documentation
cadence". Marked `orphan: true` so it doesn't appear in the toctree.

## Scope, what v0.10.0 is, and what it is NOT

> **Read this before writing any user-facing copy.** The v0.8.0 cut
> shipped late corrections because the prep doc overstated
> periodic-GDF maturity. The v0.9.0 cut hit the same trap on the
> polarised-GGA xc-kernel and the M06 meta-GGA parity. v0.10.0
> starts from an honest scope so the pattern does not repeat, and
> v0.10.0 prep is **explicitly conditional** on engineering-gate
> sign-offs from the GDF / XC / D4 / QA chats.

**v0.10.0 codename: _Pisani's Penguin_** (maintainer-confirmed, 
Cesare Pisani, founder of CRYSTAL; the all-electron Gaussian-orbital
periodic-SCF lineage vibe-qc's BIPOLE / multi-k Ewald / GDF /
crystal-symmetry arc descends from. Penguin: lives at extremes, 
solid-state, low-T, structured, and is unmatched in its native
crystalline regime even if it waddles in the gas phase.)

### v0.10.0 headline + included tracks (already on `main`)

| Track | Status on `main` | Notes |
|---|---|---|
| **Semiempirical periodic stack**, SE1 (USCC-DFTB gradient), SE2 (periodic open-shell DFTB), SE3 (periodic analytic gradients), SE3a (periodic analytic stress, partial) | Landed (`715d302a`, `0e895a12`, `d7d0101f`, `39861f0b`) | DFTB0 / SCC-DFTB across 87 elements; honest-scope "approximate / screening-grade" framing inherited from v0.9.0 |
| **Multi-k periodic SCF acceleration**, block-vector multi-k EDIIS + real Python accelerator family | Landed (`eb8f25bd`) | Closes the multi-k convergence-rate gap vs Γ-only |
| **Crystal symmetry SYM6**, character tables (Td, D4h, C6v, D3d, D6h, D2h, C2h, Ci, C3v, C4v, Th) + M3b per-cell-pair J/K kernel + Python orchestration | Landed (`7339dcec`, `cbd134d3`, `aa2ecf80`, `fb7cc74d`) | Reduces multi-k Fock-build cost by exploiting symmetry-equivalent k-pairs |
| **BIPOLE L_max≥2 multipole**, Cartesian→spherical conversion + wiring into UHF/RKS/UKS drivers | Landed (`eb818354`, `d7560225`, `bcce51f2`) | Trailered v0.9.x but feature-sized; rides v0.10.0 |
| **PWPB95 double hybrid** | Landed (`57506637`) | First meta-GGA / SOS double hybrid in vibe-qc |
| **basissetdev integration**, 87 BSE-fetched basis sets across 21 families | Landed in the v0.9.0 cycle (§4 maintainer override + §1 Feist clearance documented in `docs/license.md`) | First minor where basissetdev is on `main` |
| **QVF/viewer v0.9 kinds**, bookmarks, in-memory handover, writer hardening | Landed (`60287902`, `9dfe6f4f`, `2fd23c35`) | Producer/consumer contract tightened |
| **Audit follow-ups**, basis-opt LD penalty, NLopt path, atoms_keys; solver active_space / DMRG particle-number / v2RDM Q/G | Landed (`67d93669`, `792ef83d` + v0.9.1 cherry-picks) | Closes the v0.9.0 post-tag audit findings |
| **Semiempirical methods diversification**, **GFN2-xTB** (parameter parser/fetcher, Fock builder, SCC driver, analytic gradient, repulsive pairs, periodic Γ, UGFN2-xTB, **periodic gradient + stress**); **PM6** (parameter set, NDDO Fock builder, STO-overlap resonance integrals, FD gradient, energy-formula correction); **OM1 / OM2 / OM3** skeletons + parameter loaders + Loewdin-correction Fock builder + 77-test benchmark suite | Landed (Phase-0 framework + ~20 follow-on commits; representative SHAs: `5be3c782`, `1e7f3035`, `a6eeecba`, `01d27119`, `19f5527f`, `458a99fb`, `aec6916d`, `498bd4a5`, `db5a6d7b`, `16090fa3`, `7faad7ed`, `5cedec94`, `5cd89b79`, `80bcd15b`, `5859a47a`, `10f0175d`, `c2cbf26a`) | Architectural Phase-0 framework, method registry + shared parameter interfaces + Hamiltonian builders + `CoreParameterSet` basis builder, is what makes the diversification clean. Each new method inherits the same honest-scope rubric as the v0.9.0 DFTB stack, see `docs/license.md` § 3e |
| **DFT+U** (Hubbard correction), Increments 1-4 | Landed (`f8ae03e8` Inc 1 occupation matrix → `b04a5278` Inc 2a C++ kernel + pybind11 → `b8d55982` Inc 2b RHF SCF → `1da55422` Inc 2c RKS + run_job → `921b959c` Inc 2d UHF/UKS molecular → `235ff4e0` Inc 4a periodic Γ RHF → `f2777b37` Inc 4b periodic Γ RKS) | Molecular RHF/UHF/RKS/UKS + **periodic Γ-only RHF/RKS**, substantial new feature for transition-metal and localised-orbital systems |
| **NEB transition-state search** (nudged-elastic-band), M1 path primitives (interpolate_linear + IDPP) → M2 improved-tangent driver with joblib per-image SCFs → M3 climbing-image NEB (Henkelman + Uberuaga + Jónsson 2000) → M4 **periodic dispatch via BIPOLE + FD gradient** | Landed (`d47a3b55`, `3a5c38c1`, `d3e70f73`, `99fac10a`) | Molecular + periodic; FD gradient (analytic NEB-gradient is a future increment). See `docs/user_guide/neb.md` |
| **GPW iterative SCF** (M2-full), minimal iterative GPW RHF SCF on closed-shell cells | Landed (`dccfa60a` M2c single-point → `9a27933b` M2-full iterative SCF → `1d9d8590` validate + SCF Madelung test; paired with `dbe49c46` GPW user-guide page) | First CP2K-style GPW SCF method in vibe-qc |
| **Periodic-methods supporting infrastructure**, periodic D3-BJ dispersion via dftd3 + builtin supercell fallback (`5dbfbfc3`); **native slab + adsorbate builder** with frozen-atom relaxation (`eb83c6c7`); **relaxed coordinate scans** molecular + periodic (`f9c2453e`); Fermi-Dirac smearing consolidation (`e790f558`, `377d2cd1`); BIPOLE accelerator wiring + EDIIS bridge M3 (`2c1617d3`) | Landed as cited | Surface-reactions and band-structure workflows now have an end-to-end story (cf. `handovers/HANDOVER_SURFACE_REACTIONS.md` + the BIPOLE / GDF / GPW-GAPW comparison guide at `58de7c37`) |

### CONDITIONAL on engineering-gate sign-offs

These three define whether the headline writes itself or needs surgery:

| Track | Gating chat | If YES (tag-ready) | If NO |
|---|---|---|---|
| **Periodic GDF parity**, compcell + multi-k + AFT, µHa parity vs `PySCF.pbc.GDF` | GDF chat (`.release-status/v0.10.0/gdf.md`) | (YES branch not taken) | ✅ **DEFER LOCKED 2026-05-27** by maintainer. GDF chat verdict: `dfa34b4a` fixed the Bloch-summed lattice pair-FT bug in `_compcell_aft_correction_3c` (real fix); `b8f1e509` complete-M1-investigation documented the remaining bugs (RSGDF SR+LR calibration: 98.5% error for L=0, 23.8% for L=1; compcell metric near-singular for tight ionic crystals, cond 1e10-1e14). GDF moves to v0.11.0. **At tag time**: drop all "multi-k GDF µHa parity" / "compcell GDF" claims from CHANGELOG `[v0.10.0]` + homepage admonitions; record the **fourth GDF deferral chain** (v0.8.0 → v0.9.0 → v0.10.0 → v0.11.0) explicitly in `docs/roadmap.md` so a fifth slip doesn't quietly happen. **Codename**: *Pisani's Penguin* stays, anchors the periodic-SCF spine (BIPOLE / multi-k EDIIS / SYM6 / GPW M2-full / CRYSTAL-α unification / A1 Ewald-gauge fix `859efe0a`) which is itself a real maturity step. Use the **fallback Block 2 admonition** (pre-staged below). |
| **Native D4 default-backend flip**, production-dataset upgrade (aug-cc-pVDZ-level reference) | D4 chat (`.release-status/v0.10.0/d4.md`) | (YES branch not taken) | ✅ **OPT-IN LOCKED 2026-05-27** by maintainer. No D4 chat prompt needed for v0.10.0. The native backend stays available via `backend="native"`; the LGPL `dftd4` package remains the default until the production-dataset upgrade lands. **No CHANGELOG entry needed beyond a one-line note** that `backend="native"` is available for users who want the in-tree MPL-2.0 path. Flip queued for v0.10.x or v0.11.0. |
| **v0.9.0 Known-issues residuals**, polarised-GGA xc-kernel + M06 meta-GGA parity | XC chat (`.release-status/v0.10.0/xc.md`) | ✅ **CLOSED 2026-05-24** by XC-chat verdict. Fixes were already in v0.9.0's tree, `b34f051c` bundled the XC fixes alongside the test-gate restore (despite its misleading `fix(release):` subject). M06 drift root-caused to a vibe-qc/PySCF grid mismatch; both sides now pin a matched fine grid (`_fine_m06_grid`, 120/23/46). Tolerance kept at 2e-5, § 8 discipline respected. 10 pinning tests all green. v0.10.0 CHANGELOG carries a one-line retraction (pre-staged below). | (NO branch not taken) |

### Deferred OUT of v0.10.0, do NOT claim these in v0.10.0 copy

| Deferred item | Target | Why |
|---|---|---|
| **ωB97X-D** (RSH milestone B) | v0.10.x / v0.11.0 | Still blocked on a D2/CHG dispersion reference; PySCF lists `wb97x-d` UNSUPPORTED_DISP; dftd3/dftd4 do D3/D4 only |
| **Solvation analytic gradients** | v0.10.x | Originally v0.9.1; slipped through v0.9.x |
| **Coupled cluster (canonical CCSD(T))** | v0.14.0 | Roadmap unchanged |
| **vq queue v0.6.44-v0.6.51 security commits** | rides naturally in v0.10.0's tree | vq queue has its own sub-project release line; these commits live in `vibe-queue/` and are integrated transparently |
| **viewer-gpu PEP 508 restore** (`f0593ab6`) | v0.10.0 cherry-pick if it lands cleanly; otherwise v0.10.x | Was deferred from v0.9.1 |

## Pre-flight (before tagging)

Confirm:

1. **Dev-chat tag-ready signals** (CLAUDE.md § 3). Each contributing
   chat drops `.release-status/v0.10.0/<chat-id>.md` stating tag-ready
   Y/N. Gating chats:
   - **GDF chat** → `.release-status/v0.10.0/gdf.md`
   - **XC chat** → `.release-status/v0.10.0/xc.md`
   - **QA chat** → `.release-status/v0.10.0/qa-cross-code-parity.md`
   - **D4 chat** → `.release-status/v0.10.0/d4.md` (gating only if
     the default-backend flip is in scope)
   - **semiempirical chat** → SE1-SE3a surface sign-off
   - **basissetdev chat** → BSE integration + `docs/license.md`
     section sign-off
   - **periodic-SCF / BIPOLE / multi-k chat** → BIPOLE multipole +
     EDIIS sign-off
   **Do not tag without QA's tag-ready YES.**

2. **`CHANGELOG.md` `[Unreleased]` block** is the v0.10.0 release-notes
   draft. On tag day:
   - Promote `[Unreleased]` → `## [v0.10.0], <YYYY-MM-DD>, 
     *Pisani's Penguin*, <one-line theme>`.
   - Replace `[Unreleased]` with the v0.10.x maintenance banner
     (pre-staged below).
   - **Audit every GDF claim** against the GDF chat's verdict, if
     they came back NO, every `multi-k GDF`, `compcell`, `µHa parity
     vs PySCF.pbc.GDF` line gets either deleted or moved to a
     "Pending v0.11.0" forward-looking note.
   - **Audit the Known-issues section**, close v0.9.0 polarised-GGA
     + M06 residuals if XC says YES; carry forward honestly if NO.
   - **Carry forward `[v0.8.1]`/`[v0.8.2]`/`[v0.8.3]`** from
     `origin/release` into `main`'s CHANGELOG (the v0.9.0 follow-up
     that was deferred; the `merge -s ours` reconciliation brings
     the ancestry but not the tree, so the sections need to be
     transcribed manually). Same for `[v0.9.1]`.

3. **`docs/roadmap.md`**, flip v0.10.0 entries from "in flight" /
   "queued" to "shipped" / "✓". Anything not in the cut → v0.10.x or
   v0.11.0 sections. If GDF defers a *fourth* time
   (v0.8.0 → v0.9.0 → v0.10.0 → v0.11.0), record the deferral
   pattern honestly in the roadmap entry so it does not quietly
   slip a fifth.

4. **`docs/index.md`** homepage admonitions, swap to the pre-staged
   v0.10.0 versions below.

5. **`pyproject.toml`** version bump `0.9.2.dev0` → `0.10.0`.

6. **`CITATION.cff`** version + `date-released` bump. On `main` these
   are still at `0.8.0` from the v0.9.0 follow-up gap; the v0.10.0
   commit retroactively syncs to 0.10.0, accept this; don't try to
   back-patch a v0.9.0 / v0.9.1 sync now.

7. **`docs/citing.md`** version-bump in APA + BibTeX examples (same
   gap as `CITATION.cff`, retro-sync to 0.10.0).

8. **README.md**, verify the headline-feature paragraph mentions
   the v0.10.0 semiempirical-periodic stack + Pisani's-Penguin theme.

9. **No banner coupled-fix this cycle** unless a new vendored native
   dep landed (none believed to have between v0.9.0 and v0.10.0
   prep). If something did, run § 6 of `CLAUDE.md` checklist.

10. **`docs/license.md`**, add the **semiempirical
    in-house-parameters entry** (one paragraph: in-house Slater-rule
    + periodic-trend construction, zero dftd.org-derived values, no
    redistribution concern; cf. semiempirical chat sign-off at
    `2d2c24eb`). Deferred from v0.9.0 follow-ups; close at v0.10.0.

## Release-branch reconciliation, the `merge -s ours` step

Same procedure as the v0.9.0 cut. `release` carries the v0.9.x patch
line (currently at v0.9.1, commit `50f8a2bf`) that is NOT on `main`
as those SHAs. The v0.10.0 cut commit must have `release`'s tip in
its ancestry so `release` can ff cleanly to the v0.10.0 tag.

```sh
# On a fresh v0.10.0-cut branch off the v0.10.0 candidate SHA
# (or current origin/main if the QA gate validates current main):
git merge -s ours --no-ff origin/release \
  -m "Merge v0.9.1 release line into v0.10.0 cut (history-only)"
# Then the release: v0.10.0 commit goes on top; tag it; release ffs.
```

`merge -s ours` records the ancestry **without importing release's
tree**, which would conflict with everything that has landed on
`main` since v0.9.1.

## On tag day, order of operations

1. **Pre-flight + reconciliation.** Branch off the QA-gated SHA
   (`v0.10.0-cut` branch); run the `merge -s ours` step above; QA gate
   must already be 0-failed per the QA chat's status file.
2. **Version + metadata commit.** Bump `pyproject.toml`,
   `CITATION.cff`, `docs/citing.md`; carry the v0.8.x + v0.9.x
   CHANGELOG sections forward into `main`'s `CHANGELOG.md`; promote
   `[Unreleased]` → `[v0.10.0]`; add `"0.10.0": "Pisani's Penguin"`
   to `RELEASE_CODENAMES` in `banner.py`. Commit
   `release: v0.10.0 — Pisani's Penguin`.
3. **Smoke**, `import vibeqc` + codename check + a tiny RHF run.
   No full rebuild needed beyond the QA gate's; this commit only
   touches docs/metadata + the banner codename entry.
4. **Tag `v0.10.0`** annotated:
   `vibe-qc 0.10.0 — Pisani's Penguin`. Push the tag.
5. **Fast-forward `release`** to the tag commit:
   `git push origin v0.10.0^{}:refs/heads/release` (peel the
   annotated tag to its commit, same gotcha as v0.9.0).
6. **Post-release:** bump `main` to `0.10.1.dev0` (separate commit
   on `main`'s tip, not on the cut branch).
7. **Watch the docs-deploy CI**, `release` push fires
   `docs-build` + `docs-deploy` (no test gate). Expect ~5-12 min for
   `vibe-qc.com` to flip.

## Pre-staged copy: homepage admonitions for v0.10.0

### Block 1, release-banner admonition

Auto-substituted via MyST `{{release}}` / `{{codename}}` from
`pyproject.toml` + `RELEASE_CODENAMES`. **No manual edit needed**, 
but the `"0.10.0": "Pisani's Penguin"` catalog entry MUST be added
to `banner.py` in the same commit that bumps `pyproject.toml`.

### Block 2, replace the v0.9.0 wavefunction-methods headline admonition

**Currently** (search `Knowles's Kingfisher` / the v0.9.0 headline
block in `docs/index.md`): the v0.9.0 wavefunction-methods headline.

**Replace with, IF GDF lands** (v0.10.0 headline, marquee version):

> ✅ Periodic SCF maturity, in v0.10.0
>
> v0.10.0 (codename *Pisani's Penguin*) brings vibe-qc's
> all-electron Gaussian-orbital periodic-SCF stack into the
> CRYSTAL-class regime: **multi-k Gaussian density fitting** (compcell
> + AFT, µHa parity vs ``PySCF.pbc.GDF``), **block-vector multi-k
> EDIIS** for fast convergence, **crystal-symmetry SYM6** with full
> character-table support and a per-cell-pair J/K kernel, and
> **BIPOLE L_max≥2** Cartesian→spherical multipole conversion.
> v0.10.0 also delivers the **semiempirical periodic stack**
> (SE1-SE3a: USCC-DFTB gradient, periodic open-shell DFTB, periodic
> analytic gradients + stress), the **PWPB95 double hybrid** (first
> meta-GGA / SOS double hybrid), and integrates **87 BSE-fetched
> basis sets** under the §1 Feist clearance. See
> [CHANGELOG](https://vibe-qc.com/docs/)
> for the full notes.

**Replace with, IF GDF defers** (v0.10.0 headline, fallback version):

> ✅ Periodic-SCF maturity arc, in v0.10.0
>
> v0.10.0 (codename *Pisani's Penguin*) advances the all-electron
> Gaussian-orbital periodic-SCF stack with **block-vector multi-k
> EDIIS** for fast convergence, **crystal-symmetry SYM6** with full
> character-table support and a per-cell-pair J/K kernel, and
> **BIPOLE L_max≥2** Cartesian→spherical multipole conversion.
> v0.10.0 also delivers the **semiempirical periodic stack**
> (SE1-SE3a: USCC-DFTB gradient, periodic open-shell DFTB, periodic
> analytic gradients + stress), the **PWPB95 double hybrid** (first
> meta-GGA / SOS double hybrid), and integrates **87 BSE-fetched
> basis sets** under the §1 Feist clearance.
> **Periodic GDF (compcell + multi-k + AFT) remains deferred** to
> v0.11.0, see the roadmap entry for the deferral chain.

### Block 3, replace the v0.9.x "maintenance window" warning admonition

Refresh the open-issues warning to the v0.10.x window. Three buckets:

* **Closes at v0.10.0** (drop): items closed by the v0.9.x patch
  line + any v0.10.0 fixes in the cut. Audit against the v0.9.1
  CHANGELOG + the actual v0.10.0 cherry-pick set.
* **Carries v0.9.x → v0.10.x** (keep): items still unresolved. If
  GDF defers again, *that's* the headline carry-forward item.
* **New at v0.10.0** (add): polarised-GGA + M06 if XC says NO;
  v0.9.x deferred items that didn't land in v0.10.0; any new known
  issues surfaced during the QA gate.

### Block 4, funding admonition

Unchanged from v0.9.0 unless the sponsor surface changed. Verify URLs.

## Pre-staged copy: CHANGELOG `[Unreleased]` reset

After promoting current `[Unreleased]` to `## [v0.10.0]`, replace
`[Unreleased]` with:

```markdown
## [Unreleased]

> Heading toward v0.10.x maintenance + v0.11.0. v0.10.x will close
> ωB97X-D (pending a D2/CHG dispersion reference), solvation analytic
> gradients (slipped from v0.9.1), and any v0.10.0 Known-issues
> residuals. v0.11.0 is reserved for [GDF if it slipped from v0.10.0
> — record the four-time deferral pattern honestly]. Coupled cluster
> (canonical CCSD(T)) is roadmapped for v0.14.0.
```

## Pre-staged copy: v0.10.0 CHANGELOG, XC-residuals retraction note

Add this under the v0.10.0 `### Fixed` block at cut time (it's a
*retraction* of an earlier release note, not new code). If the
v0.10.0 cut has no `### Known issues` section at all, this still
goes under `### Fixed`:

> * **Retraction of the v0.9.0 "Known issues" XC note.** v0.9.0's
>   CHANGELOG flagged the polarised-GGA xc-kernel API +
>   M06 meta-GGA UKS parity as residuals "tracked for 0.9.1."
>   Subsequent investigation by the XC chat confirmed both were
>   **already closed at the v0.9.0 cut commit** by `b34f051c`, 
>   which bundled the XC fixes alongside the test-gate restore + the
>   k-GDF rewrite, despite its misleading `fix(release):` subject.
>   The M06 drift was a vibe-qc/PySCF grid-mismatch artefact,
>   resolved by pinning a matched fine grid (`_fine_m06_grid`:
>   120/23/46; PySCF `atom_grid=(120,590)`, `prune=None`,
>   `use_newton`). The tolerance was kept at the published 2e-5, 
>   CLAUDE.md § 8 discipline respected, not loosened. The
>   polarised-GGA `eval_polarised_gga_fxc` binding was likewise live
>   in v0.9.0 already. **No code change in v0.10.0**, this entry
>   retracts an overcautious release note.

## Pre-staged copy: roadmap "shipped" sweep

For each v0.10.0 entry in `docs/roadmap.md`:

1. Add a leading `**Status: shipped in v0.10.0 (<YYYY-MM-DD>) as part
   of *Pisani's Penguin*.**` line.
2. Move slipped sub-items to a v0.10.x / v0.11.0 section.
3. **If GDF defers a fourth time** (v0.8.0 → v0.9.0 → v0.10.0 →
   v0.11.0): the roadmap entry should record the deferral chain
   explicitly, `Deferred v0.8.0 → v0.9.0 → v0.10.0 → v0.11.0`.
   Honesty about repeated slips is what stops a fifth one.

## What's NOT in this checklist

- **Engineering tasks** owned by dev chats: GDF parity itself, XC
  residuals, the D4 production-dataset upgrade, the semiempirical
  audit follow-ups. The chats sign off via their status files; this
  prep doc reads those signals, it does not generate them.
- **CI verification**, verify by visiting <https://vibe-qc.com/> a
  few minutes after the `release` push. The deploy job has no test
  gate; it runs `docs-build` + `docs-deploy` only.
- **Periodic GDF**, if conditional verdict came back NO, GDF is
  explicitly v0.11.0+; not part of v0.10.0 at all.
- **Coupled cluster**, v0.14.0; do not mention as imminent.

## Estimated wall-clock for the docs sprint

Assuming engineering has merged + the QA gate is green + the
reconciliation is done:

- `merge -s ours` reconciliation + CHANGELOG v0.8.x + v0.9.x carry-forward: ~25 min
- CHANGELOG `[Unreleased]` → `[v0.10.0]` promotion + `[Unreleased]` reset: ~10 min
- Homepage admonitions swap (Blocks 2-3), choose YES/NO-GDF version: ~15 min
- Roadmap "shipped" sweep + deferral-chain honesty: ~30 min
- `pyproject.toml` + `CITATION.cff` + `docs/citing.md` + `banner.py`
  codename catalog entry: ~10 min
- README.md headline refresh: ~10 min
- `docs/license.md` semiempirical in-house-parameters entry: ~10 min
- Verify deploy + cross-link audit: ~30 min

**Total: ~2.5-3 hours** of focused release/docs work on tag day.

## After the dust settles

Queue for v0.10.x maintenance:

- v0.9.0 + v0.9.1 post-tag docs sprint items still pending:
  `CITATION.cff` + `docs/citing.md` on `main` were stuck at 0.8.0 (the
  v0.9.0/v0.9.1 bumps lived on the side-branch tag commits, never
  on `main`). The v0.10.0 cut retroactively syncs them, close that
  follow-up.
- `[v0.8.1]`/`[v0.8.2]`/`[v0.8.3]`/`[v0.9.1]` CHANGELOG sections
  carry-forward into `main`, performed in the v0.10.0 cut commit.
- `scripts/doc_examples.toml` manifest fix (stale since before v0.8.0).
- ωB97X-D once a dispersion reference exists.
- Solvation analytic gradients.
- The standing `merge -s ours` reconciliation will be needed again
  at every minor while `release` carries a patch line, consider
  whether a periodic `main` ↔ `release` true-merge is worth doing
  once to reset.
