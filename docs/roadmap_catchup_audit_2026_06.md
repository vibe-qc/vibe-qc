# Roadmap catch-up audit, v0.14 scope (2026-06-17)

**Audited tree:** `main` @ `f30addb9` (current head; v0.13.0 *Wisesa's Fox*
shipped). Read-only audit, no code changed.

**Question answered:** which features the roadmap scheduled for **v0.5.0
through v0.13.x** (i.e. should *already* be shipped) are still
`NotImplementedError` stubs, built-but-unwired orphans, or absent **on
current `main`**, so the release chat can scope a v0.14 "catch-up"
milestone.

**Method.** Every line below was checked against the *current* code
(function bodies, dispatch tables, pybind bindings), not against the
roadmap prose or any prior audit. The roadmap text and the citation
database are treated as *claims to verify*, not ground truth. Five
parallel search passes (SCF mixers, Coulomb/accel dispatch, functionals,
properties/IO, roadmap version-mapping) fed candidate sites; every
load-bearing verdict was then re-read by hand at the cited `file:line`.

**Calibration (confirms the tree is current).** A prior run of this
audit used a ~110-commit-stale checkout and wrongly flagged the whole
SCF initial-guess overhaul as missing. On `f30addb9` those are all
**implemented**: `cpp/src/guess.cpp` builds SAD/SAP/PATOM/HUECKEL/MINAO
for molecular systems (`build_closed_shell` / `build_open_shell`, lines
1277-1383 / 1399-1645); `READ` is intercepted in Python
(`guess.py:55`, `guess_read.py`, 478 lines) before the engine; ATOMSPIN
(`atomic_spins`) and SPINLOCK (`spinlock_periodic.py`, PATTERN_HOLD +
SPIN_SCHEDULE wired on the Γ UHF/UKS Ewald + multi-k UKS drivers) are
live. Calibration passes, the gaps below are real, not staleness
artefacts.

**Reconciliation note (main moved during the audit).** The findings were
read at `f30addb9`; while writing, main advanced to `7d46dd22` (this doc
sits on top of it). One intervening commit touches a finding directly:
`111f96da feat(periodic): Anderson density mixer, start the v0.10.x
metal-mixing program (D4a)` landed a fresh, tested-but-unwired
`AndersonMixer` in the new module `python/vibeqc/periodic_density_mixing.py`.
So **the periodic-mixer cluster (Tier 2 #1-#4) is now an active in-flight
workstream, not dormant debt**, see the updated Tier 2 #1. The other
four intervening commits (`7d46dd22`, `0d4770b0`, `1ec5d914`, `82295bd9`:
megacell-MP2 docs/test, GAPW soft/hard partition fix, Γ ASE Ewald-3D
routing) do not touch any finding here.

---

## Executive summary

The catch-up surface is **smaller and more concentrated than the
roadmap's unmarked rows suggest**, and falls into a few clusters:

1. **A handful of exposed enum/kwarg values that raise (or silently
   lie) when a user selects them**, the worst class, because users hit
   them. Tier 1.
2. **Two whole dead modules** (`scf_acceleration.py`, `kerker.py`) of
   periodic density mixers that are fully classed but *never imported*
   anywhere, Tier 2. (As of `7d46dd22` these are being superseded by a
   fresh component-first line: `periodic_density_mixing.py:AndersonMixer`
   just landed as the start of the v0.10.x D4a program, so treat this
   cluster as *coordinate with the active workstream*, not cold pickup.)

   **Resolved 2026-07-10.** `scf_acceleration.py` was deleted: all three
   of its classes are superseded by the now-shipped
   `periodic_density_mixing.py` mixers, and `SmearingAwareDIIS` was a
   fourth numpy DIIS in violation of CLAUDE.md §10. `kerker.py` is no
   longer a dead module: `periodic_density_mixing.py` imports
   `kerker_kernel` from it. Whether the other `kerker.py` symbols
   (Tier 2 rows 3 and 4) still lack callers was not re-verified.
3. **A short list of genuinely-absent named convergers/functionals**
   (LIST, ARH, CIAH, FRAGMO; VV10 + the ωB97M-V/ωB97M(2)/revDSD double
   hybrids that depend on it). Tier 3.
4. **Large research subsystems the roadmap optimistically dated to
   v0.8-v0.10** (real FMM/CFMM/PBC-FMM, PAW, ACE, ADMM, 2D/1D Ewald), 
   these were never "committed-and-stubbed", they are research
   directions whose dates have simply passed. **Re-version, don't
   catch up.** Tier 4.

**Important meta-finding:** *no roadmap **✅-shipped** claim audited
turned out to be a stub.* The ✅ marks are honest. Conversely the
roadmap **under**-claims: SOSCF, TRAH, ADIIS, ODA and KDIIS are wired
and shipped but sit in the roadmap as unmarked future-version rows (see
Corrections § B). So the catch-up work is "finish the unmarked rows and
re-date the research rows", not "the shipped claims are lies."

---

## Tier 1, exposed API/enum that raises (or silently substitutes)

> **Update (2026-06-18):** the *honest-surface* fixes for #1-#4 landed (CHANGELOG
> `[Unreleased]`). #1 no longer silently substitutes DIIS (raises
> `NotImplementedError`); #4's Python periodic-driver seam raised a clean
> `NotImplementedError` for SAP/PATOM/HUECKEL/MINAO at that point
> (dispatch-layer parity with #2/#3, the C++ Γ drivers keep the engine's clear
> `RuntimeError`, a `NotImplementedError` superclass); #2/#3 already raised cleanly and are
> unchanged. This makes the surfaces honest, it does **not** implement the
> underlying physics (real Anderson/Broyden mixing, 2D/1D Ewald, lattice-summed
> periodic SAP), which remains the per-area v0.14 work below.
>
> **Update (2026-07-04):** closed-shell C++ periodic RHF/RKS now build explicit
> SAP, HUECKEL, MINAO, and PATOM guesses at Gamma and multi-k. Closed-shell
> Python periodic drivers also pass the context needed by the SAP/HUECKEL
> lattice Fock helpers and the MINAO density seam. PATOM open-shell/Python
> paths and open-shell SAP/HUECKEL/MINAO integrations remain gated.

A user can select these through a documented enum value or kwarg and
either hits `NotImplementedError` or, worse, gets a *different* method
with no error.

| # | feature | roadmap version | status (`file:line`) | effort |
|---|---------|-----------------|----------------------|--------|
| 1 | **`scf_accelerator="anderson"` / `"broyden"`** silently resolve to **DIIS** | v0.10.x (D4a/D4c) | **SILENT-SUBSTITUTE**, `python/vibeqc/__init__.py:410-411` map both strings to `SCFAccelerator.DIIS` with a `# placeholder` comment; the user asked for Anderson/Broyden and gets DIIS with **no warning**. | S (make honest) / M (deliver) |
| 2 | **`CoulombMethod.SLAB_EWALD_2D`** (2D slab Ewald) | v0.8 (descriptive only; dispatch msg says "unscheduled") | **CURRENT (2026-07-05):** Gamma-only and multi-k RHF/RKS/UKS slab Ewald SCF paths are active. Analytic gradients still raise cleanly. Earlier audit state was a full stub. | L |
| 3 | **`CoulombMethod.NEUTRALIZED_1D`** (1D wire Ewald) | v0.8 (descriptive only) | **STUB**, same enum/dispatch as #2; `periodic_rhf.cpp:98`, `periodic_rhf_dispatch.py:242`. | L |
| 4 | **Periodic `initial_guess` = SAP / PATOM / HUECKEL / MINAO** | v0.6.x (SAP); v0.14/v0.15 catch-up | **PARTIAL SHIP**, molecular paths ship; closed-shell C++ periodic Gamma and multi-k RHF/RKS now build SAP, PATOM, HUECKEL, and MINAO; closed-shell Python periodic drivers also reach SAP, HUECKEL, and MINAO through the density seam. `AUTO` safely picks SAD for periodic. Remaining gates: PATOM on open-shell/Python periodic drivers and SAP/HUECKEL/MINAO on open-shell drivers. | M |
| 5 | **`xc="revdsd-pbep86"`** (revDSD-PBEP86 double hybrid) | v0.8.0 (F4) | **ABSENT compute path**, no alias in `cpp/src/xc.cpp`; resolver throws `Functional: unknown name` (`xc.cpp:468`). Surface exists only as a D4-param row (`dispersion_d4_parameters.py:185`) + citation route. | S-M |
| 6 | **`xc="wb97m-v"` / `"wb97m(2)"`** (ωB97M-V / ωB97M(2)) | v0.8.0 (F1 / F4) | **ABSENT**, no `wb97m` alias anywhere in `cpp/src/xc.cpp`; throws `unknown name`. Blocked on VV10 (Tier 3 #5). | L (needs VV10 first) |

> **Note on #5/#6:** `xc=` is a free-form string, so "unknown name" is
> the generic resolver failure, not a dedicated stub, but the feature
> *is* named/parametrised (D4 rows, citations, roadmap F4), so it reads
> as Tier 1 to a user who copied the name from the dispersion table.
> The runnable double hybrids (B2PLYP, DSD-PBEP86, PWPB95) **do** work, 
> see Corrections § C.

---

## Tier 2, built but not wired (cheap-ish to finish)

Implementation code exists in-tree but **no driver or dispatch ever
calls it** (verified: neither `scf_acceleration` nor `kerker` is
imported anywhere under `python/vibeqc/`).

> **Stale as of 2026-07-10.** `scf_acceleration.py` has since been
> deleted (rows 1, 2, 5 below), and `kerker.py` is now imported by
> `periodic_density_mixing.py`. Re-verify before picking up rows 3-4.

| # | feature | roadmap version | status (`file:line`) | effort |
|---|---------|-----------------|----------------------|--------|
| 1 | **Anderson mixing** | v0.10.x (D4a) | **DONE (v0.14.0).** `AndersonMixer` at `python/vibeqc/periodic_density_mixing.py` is wired and shipped. The dead orphan `AndersonAccelerator` it superseded was deleted with `scf_acceleration.py` (2026-07-10). | done |
| 2 | **Broyden mixing** (`BroydenAccelerator`) | v0.10.x (D4c) | **DONE (v0.14.0).** Superseded by `BroydenMixer` (`periodic_density_mixing.py`), which mixes the density directly instead of re-diagonalising with the guessed `n_occ = D_new.shape[0] // 2`. Orphan deleted 2026-07-10. | done |
| 3 | **Kerker preconditioner / mixing** (`apply_kerker_preconditioner`, `kerker_mix_density`, `kerker_density_matrix_mix`, `JohnsonEyertMixer`) | v0.10.x (D4b/D4c/D4e) | **ORPHAN**, `kerker.py:109/148/366/279`; zero callers outside `kerker.py`. | M |
| 4 | **Pulay-Kerker mixer** (`PulayKerkerMixer`) | v0.10.x (D4d) | **ORPHAN**, `kerker.py:184`; docstring claims "VASP's default", nothing instantiates it. | M |
| 5 | **`SmearingAwareDIIS`** | (unscheduled; D4 family) | **DELETED 2026-07-10, do not resurrect.** Never imported, never ran: `extrapolate` raised `ValueError` from `np.trace` on a raveled 1-D error vector as soon as the history reached 2 entries. Its premise was also wrong: the commutator error `FDS-SDF` *does* vanish at the fixed point under fractional occupations (`ε` and `f(ε)` are both diagonal, so they commute), so the density-residual switch bought nothing. Nothing to port either: canonical `vibeqc::DIIS::extrapolate(fock, error)` already takes the error vector as a caller argument. | done |
| 6 | **Molecular FMM foundation** (`MultipoleJKBuilder`) | v0.8.x (FMM1) | **PARTIAL/ORPHAN**, `multipole_jk_builder.py` header self-labels "**Status, foundation, not production-ready**" (78% error on H₂O/STO-3G); multipole building blocks exist but **no near-field/far-field split** and no `method=`/`coulomb=` routes to it. Production builder self-dated "v0.12-v0.13" in its own docstring. | L |

> Tier 2 #1-#5 are "wire an existing class into the periodic SCF loop +
> validate vs PySCF", the algorithm code is the cheap part; the cost is
> a correctness gate (per CLAUDE.md §7, a periodic mixer that hides an
> oscillation is a bug, not a fix). #6 is only *foundation* code, so its
> effort is L despite living here.

---

## Tier 3, named in roadmap (≤ v0.13.x), genuinely absent

No implementation exists (not even orphaned code), but the roadmap
scheduled them for a version that has shipped.

| # | feature | roadmap version | status | effort |
|---|---------|-----------------|--------|--------|
| 1 | **LIST** (Lagrange-interpolated SCF) | v0.7.x (D1e) | **ABSENT**, no `LIST*` class in Python or C++; only ordinary `list` usage matches. | M |
| 2 | **ARH** (augmented Roothaan-Hall) | v0.8.x (D2f) | **ABSENT**, exists only as citation `database.toml` ("roadmap v0.9.x D2f") + an ADIIS expansion comment; no optimizer. | M-L |
| 3 | **CIAH** (co-iterative augmented Hessian) | v0.8.x (D2g) | **ABSENT**, citation-only. The 2nd-order machinery that *does* exist is TRAH/SOSCF (shipped, see Corrections § B), which is a different algorithm. | L |
| 4 | **FRAGMO** (fragment-MO assembly guess) | v0.8.x (D2h) | **ABSENT**, no fragment-SCF-superposition guess path. | M |
| 5 | **VV10 nonlocal correlation** | v0.5.0 (D1/D2 prereq list) | **ABSENT**, referenced only in `cpp/src/xc.cpp:410-414` comments; never evaluated. Root-cause blocker for the three double hybrids below. `xc="wb97x-v"` *resolves* (libxc id) but **silently omits VV10**, its energy is the semilocal+RSH part only, complete only inside the ωB97X-3c recipe (where D4 stands in). | M-L |
| 6 | **ωB97M-V / ωB97M(2)** | v0.8.0 (F1/F4) | **ABSENT**, gated on #5 (no `wb97m` alias). | L (after VV10) |
| 7 | **revDSD-PBEP86** | v0.8.0 (F4) | **ABSENT compute path**, D4-param/citation surface only (see Tier 1 #5). Double-hybrid machinery exists, so this is just an alias + coefficient wire-up + validation. | S-M |
| 8 | **Molecular GUESSMIX broken-symmetry** (high-spin → HOMO-LUMO 45° mix) | v0.7.x (G2j) / v0.6.x | **PARTIAL**, a broken-symmetry *start* ships via per-atom spins (ATOMSPIN/`atomic_spins`, periodic + molecular block-diagonal SAD, `guess.py:91`); the standalone ORCA-style GUESSMIX HOMO-LUMO-rotation option for molecular UHF is **not a named/wired path** (no `guessmix`/`homo_lumo_mix` symbol). *(Lower confidence, the ATOMSPIN route covers most use cases.)* | S-M |

---

## Tier 4, optimistically dated; re-version the roadmap (don't "catch up")

These were scheduled v0.8-v0.10 but are large research subsystems, never
committed as stubs: there is **no enum value, no kwarg, no raising code
path** a user can reach, only citation-database entries with dormant
routes (the `acceleration=` producer `runner.py:_detect_acceleration`
can only ever emit `rij`/`rijcosx`/`[]`, so those rows never fire). The
right action is to **move their roadmap dates to v0.14+/post-1.0**, not
to treat them as overdue deliverables.

| # | feature | roadmap version | status | note |
|---|---------|-----------------|--------|------|
| 1 | **Production FMM / CFMM** (molecular, near+far split) | v0.8.x (FMM1/CSAM) | ROADMAP-ONLY beyond the Tier-2 foundation | needs the `distance_cutoff` C++ JKBuilder work; self-dated v0.12-v0.13 |
| 2 | **PBC-FMM** (periodic FMM, Kudin-Scuseria) | v0.8.x | ROADMAP-ONLY | At the time of this audit, BIPOLE exposed an experimental `use_multipole_far_field` opt-in, not a periodic FMM tree. Superseded 2026-08-13: every BIPOLE SCF driver now rejects explicit activation before setup. |
| 3 | **ADMM** (auxiliary density matrix method) | v0.9.x (D3d) | ROADMAP-ONLY | citation `database.toml` + dormant `routes.acceleration."admm"`; no code path |
| 4 | **PAW** (projector augmented wave) | v0.10.x | ROADMAP-ONLY | citation-only; a Gaussian-orbital code has no PAW datasets (`periodic_gapw_augment.py:42`) |
| 5 | **ACE** (adaptively compressed exchange) | v0.10.x | ROADMAP-ONLY | roadmap itself calls the Gaussian-periodic translation "an open research question" (`roadmap.md:710`) |
| 6 | **2D-Ewald / 1D-Madelung physics** behind `SLAB_EWALD_2D` / `NEUTRALIZED_1D` | v0.8 (descriptive) | the *enum* is Tier 1 (#2/#3); the *physics* is research-grade | either implement, or remove the enum members + raise a clean "unsupported" at the API edge |

> The properties suite (ESP/CHELPG/RESP, NBO/NPA, NMR, EPR, Raman, VCD,
> ROA, Bader/QTAIM, ELF/NCI, hyperpolarisability, iterative-Hirshfeld,
> Born charges, COHP/COOP, VPT2, Berry-phase polarisation) is **not**
> in catch-up scope: the roadmap honestly marks every one ❌ and dates
> them **v0.19.0 (Phases PR1-PR29)** or post-1.0, correctly future.
> See Corrections § D.

---

## Corrections

### § A, roadmap ✅-shipped claims that are actually stubs

**None found.** Every roadmap **✅** mark spot-checked is honest:
READ guess (`guess_read.py`), MINAO/Extended-Hückel (`guess.cpp`),
ATOMSPIN/SPINLOCK (`spinlock_periodic.py`), RIJCOSX (v0.8.0), the Sx2
gradient-screening threshold (`schwarz_threshold_forces`, below), the
cube/XSF/molden writers, and the closed-shell DF-CCSD(T) pilot all do
what the ✅ claims. The catch-up gaps live entirely in **unmarked**
(status-none) future-version rows, *not* in over-claimed ✅ rows.

### § B, roadmap **under**-claims (shipped, but unmarked, do NOT re-implement)

The roadmap's SCF guess/convergence table (`roadmap.md:794-824`) lists
these as bare future-version cells with no ✅, but the code ships them.
The fix is a **roadmap edit**, not implementation work:

| feature | roadmap row | actually shipped at (`file:line`) |
|---------|-------------|-----------------------------------|
| **SOSCF** (2nd-order Newton SCF) | v0.8.x flagship (D2c) | `runner.py:807-828` arms it via `soscf_threshold`; C++ `cpp/include/vibeqc/soscf.hpp` |
| **TRAH** (trust-region augmented Hessian) | v0.8.x flagship (D2e) | `runner.py:807-828` via `trah_threshold`; `cpp/include/vibeqc/trah.hpp` |
| **ADIIS** (Hu-Yang 2010) | v0.7.x (D1b) | `periodic_scf_accelerators.py:355`; reachable via `scf_accelerator=ADIIS` |
| **ODA** (Cancès-Le Bris) | v0.7.x | `oda.py`; opt-in `use_oda=True` in `periodic_rhf_multi_k_ewald.py`; the former BIPOLE hook now fails closed because mixed densities lack a single orbital representation |
| **KDIIS** (Kollmar) | not in roadmap | `periodic_scf_accelerators.py`; reachable via `scf_accelerator=KDIIS` |
| **EDIIS / EDIIS+DIIS** | shipped (D1a/D1c) ✅ | already ✅, confirmed wired (production default) |

*(SOSCF/TRAH are explicitly **disabled** on the periodic GAPW path, 
`periodic_gapw_cpp_host.py` zeroes both thresholds, but that is a
documented periodic limitation, not a roadmap "shipped" claim.)*

### § C, confirmed fine (do not re-investigate)

* **Runnable double hybrids:** B2PLYP, DSD-PBEP86, PWPB95 all resolve in
  `cpp/src/xc.cpp` (mp2 coefficients set) and have `run_*` wrappers;
  `run_double_hybrid(...)` (`__init__.py:1236`) drives hybrid-SCF + RI-MP2.
  Molecular-only by design.
* **RSH machinery (F1):** HSE06, ωB97X, ωB97X-D, CAM-B3LYP and the
  erf-attenuated-K path are wired (`cpp/src/xc.cpp:378-579`). Only the
  **VV10-paired** members (ωB97X-V full / ωB97M-V) are incomplete
  (Tier 3 #5).
* **Periodic gradient screening (Sx2):** a *separate, tighter* forces
  threshold ships, `schwarz_threshold_forces = 1e-14` vs energy `1e-12`
  (`cpp/include/vibeqc/lattice_sum.hpp:86`, consumed
  `cpp/src/periodic_gradient.cpp:654`). The molecular gradient relies on
  libint's internal nullptr-buffer screen with no separate user knob
  (roadmap Sx1), minor and low-priority (engine screens anyway), not a
  correctness gap.
* **CC `citype=` / `triples=` stubs** (`runner.py:504`, `cc.py:91`):
  these raise honestly and are scheduled **v0.14.0+ / post-1.0 AutoCI
  wishlist** (CCD/CC2/CC3/CCSDT/QCISD/CEPA/BCCD, A-CCSD(T)/CCSD[T]), 
  *not* in the v0.5-v0.13 catch-up window. Leave them.
* **MSINDO `NotImplementedError` sites** (`msindo*.py`, ~14 of them) are
  legitimate input guards (Z>54 not parameterised; RHF-only post-SCF
  paths reject open-shell), **not** roadmap gaps. The genuine MSINDO
  feature stubs (open-shell periodic CCM `msindo_ccm.py:495`; analytic
  CIS gradient `msindo_cis.py:826`; coarse GEPOL grid `msindo_gepol.py:191`)
  are basissetdev/MSINDO-track increments, not main-line catch-up.
* **GFN2-xTB / OM1** run but are experimental-gated
  (`gfn2.py:5` warns "not quantitative"; `runner.py:1620` warns OM1's
  core-valence ECP is unimplemented), honest gating, by design.
* **Writers/readers:** cube, XSF, BXSF, POSCAR, CIF (write+read), QVF,
  molden all write/read fully. The only writer stub is `.g94` for
  ECP-bearing atoms (`basis_crystal.py:589`, "libecpint integration
  pending"), a known basissetdev follow-on. No wfn/wfx/fchk writer
  exists, but nothing claims one.

### § D, out of catch-up scope (correctly future)

The entire **v0.19.0 properties suite** (ESP charges, NBO/NPA, GIAO NMR
shielding + J-coupling, EPR g/A-tensors, Raman, VCD, ROA, Bader/QTAIM,
ELF/NCI, hyperpolarisability, iterative-Hirshfeld, MBIS, Born charges)
is honestly ❌ in the roadmap and dated v0.19.0 / post-1.0. The
`[routes.properties]` table in `database.toml` pre-stages ~30 citation
routes for these, which can look like an implemented surface, but they
have **no kernel and no `run_job` input**, they only fire if a caller
passes `properties=[...]` straight into `assemble()`. This is
pre-staged provenance, not a stub a user can hit. **Leave as future.**

Genuinely-future correlation/excited-state milestones likewise out of
scope: canonical CCSD(T) (v0.14.0, closed-shell pilot already shipped),
DLPNO-CCSD(T) (v0.15.0, in active dev under `dlpno/`), periodic
correlation (v0.16-v0.17), TDDFT/MR (v0.18.0). A handful of
within-reach behavioural stubs a user *can* hit, TDDFT+range-separated
hybrid (`tddft.py:291`), ROKS Hessian (`runner.py:3198`),
ROKS+RSH/double-hybrid/meta-GGA (`roks.py:172-184`), open-shell
excited-state gradient (`excited_gradient.py:251`), ROHF analytic
gradient + ECP (`rohf.py:794`), are **not roadmap-scheduled items**
(no version slot), so they are limitation-documented refusals rather
than catch-up debt; surface them as their own small backlog if desired,
but they don't belong to a v0.5-v0.13 "should-be-shipped" milestone.

---

## Suggested v0.14 catch-up ordering (for the release chat)

1. **Tier 1 first, stop users hitting raises/lies.** Make
   `"anderson"`/`"broyden"` honest (raise or warn, S); decide
   SLAB_EWALD_2D / NEUTRALIZED_1D = implement-or-remove-enum (L, or S to
   gate cleanly); finish the remaining periodic initial-guess gates
   (PATOM open-shell/Python drivers and SAP/HUECKEL/MINAO open-shell drivers)
   or keep their current clean errors.
2. **Tier 3 #7 revDSD-PBEP86 (S-M)** and **#5 VV10 (M-L) → then #6
   ωB97M-V/ωB97M(2)**, finishes the v0.8.0 functional sweep with the
   machinery already in place.
3. **Tier 2 mixers (M each)**, **coordinate with the active v0.10.x D4a
   program** (`periodic_density_mixing.py`, `111f96da`) before scoping;
   the immediate open task is wiring the already-landed `AndersonMixer`
   into the RKS multi-k driver *with a PySCF correctness gate* (CLAUDE.md
   §7). Broyden/Kerker/Pulay-Kerker (D4b-e) are the still-dormant tail;
   the old `scf_acceleration.py` / `kerker.py` orphans can be deleted once
   the new line covers them. (Anderson + Broyden shipped in v0.14.0;
   `scf_acceleration.py` deleted 2026-07-10. `kerker.py` stays: it is now
   imported by `periodic_density_mixing.py`.)
4. **Tier 3 convergers (LIST/ARH/CIAH/FRAGMO)**, schedule per appetite;
   SOSCF/TRAH already cover the second-order niche, so CIAH/ARH are
   lower-value.
5. **Re-version Tier 4** in `docs/roadmap.md` (move FMM/PBC-FMM/ADMM/
   PAW/ACE dates to v0.14+/research) and apply Corrections § B (mark
   SOSCF/TRAH/ADIIS/ODA/KDIIS shipped). Both are pure doc edits.
