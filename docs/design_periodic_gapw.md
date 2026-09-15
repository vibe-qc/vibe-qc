# Design, periodic GAPW / GPW (and a path to PAW + ACE)

> **Status:** M0 design doc, **signed off 2026-05-24** (decisions
> log in §9). GPW and GAPW routes shipped as experimental on `main`
> (M1-M3 landed; M4 metallic GAPW and M5 ACE/PAW remain gated).
>
> **Original target:** v0.10.x acceleration program
> ([`docs/roadmap.md`](roadmap.md) § "Acceleration program" →
> "`v0.10.x`, reciprocal-space and basis tricks").
>
> **Chat:** `gapw`.

This document scopes the introduction of a Gaussian-augmented
plane-wave (GAPW) route, and its Hartree-only progenitor GPW,
as a NEW periodic-SCF strategy in vibe-qc, alongside the existing
Gaussian-only Ewald-3D, GDF, and BIPOLE routes. PAW and ACE are
**stretch goals** scoped from the start so the architecture
admits them later, but they are not first-cut deliverables.

## 1. What lands, what doesn't (scope)

| | M1 | M2 | M3 | M4 | M5 (stretch) |
|---|---|---|---|---|---|
| FFT-grid + PW-basis infrastructure (experimental flag) | ✅ | | | | |
| GPW Hartree-J, Γ-only, closed-shell LDA | | ✅ | | | |
| GAPW (Gaussian augmentation on the smooth grid) | | | ✅ | | |
| Metallic GAPW + smearing (composes with scf-mix chat) | | | | ✅ | |
| ACE-decorated K for hybrids | | | | | ✅ |
| PAW pseudopotentials + on-demand fetcher | | | | | ✅ |

Hard non-goals of this chat (owned elsewhere):

- Multi-k extension of GPW/GAPW. Designed-for here, but the
  first wave lands Γ-only. Multi-k GAPW becomes its own
  milestone once dual-grid is solid.
- Anything in Ewald-3D / GDF / BIPOLE drivers. We coexist.
- The molecular SCF surface.
- The v0.10.x density-mixing flagship, that is the scf-mix
  chat. Our metallic-system test set (M4) consumes their work.
- An `FFTW3 → {pocketfft, kissfft}` backend abstraction
  (a separate roadmap item, CLAUDE.md §1). GAPW will use the
  same internal FFT wrapper that `cpp/src/fft_poisson.cpp`
  uses, so it inherits whatever lands there.

## 2. The four pre-flight questions

### Q1, Where does GAPW plug into JKBuilder dispatch?

**Answer:** Add two new values to
[`PeriodicJKMethod`](../python/vibeqc/periodic_jk_method.py),
`GPW` and `GAPW`, and a new driver module
`python/vibeqc/periodic_gapw.py`. The user opts in via
`jk_method="gapw"` (or `"gpw"`) on `run_periodic_job`. We do
**not** change the `AUTO` heuristic
([periodic_jk_method.py:151](../python/vibeqc/periodic_jk_method.py))
- GDF stays the closed-shell default, BIPOLE stays the
open-shell default. GAPW is opt-in for as long as it has
narrower system coverage than the incumbents.

Current GAPW HF adds a second, physics-level opt-in:
`gapw_molecular_limit=True` declares an isolated single-Gamma box so the
method-aware one-centre default can use the fit-free analytic augmentation.
The declaration is required because that implementation does not yet include
dense-cell image sums and geometry-only classifiers cannot safely infer the
envelope. RKS/UKS retain the block construction and do not require it.

**J vs K vs J+K.** GPW/GAPW replaces **the J build** only.
Exchange in the first cut goes through whichever K-builder the
existing periodic SCF would use for the same calculation
(direct-K from the molecular-limit periodic builder; later RI-K
in hybrids; later ACE on top, see Q4). The split lets us:

- ship a defensible LDA / PBE GPW driver before any K work,
- reuse the GDF/BIPOLE chats' K machinery untouched,
- keep ACE as a decorator over the inner K, not a new kernel.

**Concrete construction shape.** Mirroring
[`make_periodic_gamma_jk_builder`](../cpp/include/vibeqc/periodic_jk_builder.hpp)
(the abstract base lives in C++,
`cpp/include/vibeqc/jk_builder.hpp`; Python orchestrates):

```
make_periodic_gapw_jk_builder(
    basis, periodic_system, lattice_opts,
    fft_grid,                      # see Q2
    pseudopotentials = None,       # None ⇒ all-electron (GAPW)
    augmentation = "gapw",         # "gpw" disables Gaussian augmentation
    omega = 0.0,
    inner_k_builder = ...,         # the K kernel; defaults to direct-K
)
```

The returned object implements the molecular `JKBuilder`
interface ([jk_builder.hpp](../cpp/include/vibeqc/jk_builder.hpp))
- `build_J(D)`, `build_K(D)`, so existing SCF drivers see no
behavioural change beyond which concrete kernel was instantiated.

### Q2, FFT-grid management (single vs dual, cell-relative coordinates, sizing)

**Single grid (GPW, M2).** A uniform 3D real-space grid in
cell-relative fractional coordinates `s = A⁻¹ r`, sampled at
`s = (i/Nx, j/Ny, k/Nz)`. Exactly the grid layout that
[`fft_poisson.cpp`](../cpp/src/fft_poisson.cpp) already uses
([`ScalarField3D` at fft_poisson.hpp:49](../cpp/include/vibeqc/fft_poisson.hpp)).
Skew cells supported via the full reciprocal-lattice metric;
already exercised by [`tests/test_fft_poisson.py`](../tests/test_fft_poisson.py).

```
  GPW J build (M2, schematic):
  ┌───────────────┐  collocate   ┌───────────────┐  FFT-Poisson  ┌───────────────┐
  │ AO density Dμν│ ───────────▶ │ ρ(r) on grid  │ ─────────────▶│ V_H(r) on grid│
  │  (compact)    │ (Gaussian-on- │ (real, single │  (1/r kernel  │ (zero-mean,   │
  │               │  uniform-grid │  grid Nx×Ny×Nz│  if neutral;  │  G=0 gauged)  │
  │               │  collocation) │ ; G-mean = 0 │  erfc/erf if  │               │
  │               │               │  for neutral │  Ewald-split) │               │
  │               │               │  cell)        │               │               │
  └───────────────┘               └───────────────┘               └───────┬───────┘
                                                                          │ project
                                                                          ▼
                                                                 ┌───────────────┐
                                                                 │ J_μν =        │
                                                                 │ ∫ φ_μ φ_ν V_H │
                                                                 └───────────────┘
```

The collocate step sums each AO over its periodic images,
`χ_μ(r) = Σ_R χ_μ^mol(r − R)` over the lattice vectors within a small
image radius (the grid-native twin of
[`ewald_j.evaluate_ao_periodic`](../python/vibeqc/ewald_j.py)), before
forming `ρ = Σ_μν Dμν χ_μ χ_ν`. An AO centred near a cell face then
folds its escaping Gaussian tail back into the box, so
`∫ρ = tr(D·S) = N` for *any* placement of the atoms in the cell, and the
density is translation-invariant. The same periodic AOs are reused for
the `J_μν = ∫ φ_μ φ_ν V_H` projection (and the V_xc projection on the
RKS path), so the Hartree / XC energies are translation-invariant too.
Without the image sum an edge-centred AO under-integrates,
`∫ρ ≈ 0.4 e` for H₂/STO-3G at a cell corner instead of 2 e (audit
2026-05-31; the molecular `evaluate_ao` is fine for centred molecules,
which is why it surfaced only off-centre).

**Dual grid (GAPW, M3b).** Lippert-Hutter:

- a *smooth* real-space grid (coarse) that carries the soft
  pseudo-density ρ̃, identical to the GPW grid;
- per-atom *radial × Lebedev* grids (fine) that carry the
  hard atomic density ρ_a and the soft atomic density ρ̃_a
  inside an augmentation sphere around each nucleus.
- a per-atom *multipole-compensating Gaussian density* ρ_0
  added to *both* the soft FFT density and the soft atomic
  density.

The Hartree potential decomposes as
`V_H[ρ] = V_H[ρ̃ + ρ_0] + Σ_a (V_H[ρ_a] − V_H[ρ̃_a + ρ_0,a])`,
where the FFT-Poisson solve runs once on the *globally*
multipole-compensated (ρ̃ + ρ_0), and each augmentation sphere
contributes a hard-minus-soft local correction. The atomic
pieces are local radial-Lebedev Poisson solves on per-atom
grids, no FFT involved.

**Three load-bearing design constraints, taken from the CP2K
audit (gapw chat 2026-05-27):**

- **D1, ρ_0 multipole compensator is mandatory, not optional.**
  Without it the soft FFT density carries spurious long-range
  multipoles even on a neutral cell, contaminating V_H[ρ̃].
  CP2K's source comments (`qs_rho0_methods.F`) pin `lmax0_kind`
  as a per-element correctness knob that must cover the basis's
  effective angular content. Implications for vibe-qc M3b: ρ_0
  has to be in the architecture from day 1, not retrofitted.
  The natural surface is a sibling of `GpwJBuilder` that owns
  the per-atom ρ_0 / Q_L state alongside the smooth-grid
  builder.

- **D2, hard/soft cancellation must be pointwise on the
  radial grid, not matrix-level.** Both `V_H[ρ_a]` and
  `V_H[ρ̃_a + ρ_0,a]` are huge near the nucleus (Coulomb tail
  through ρ_a); subtracting independently-integrated AO matrix
  elements at the end *loses orders of magnitude* of
  significance. CP2K's `add_vhxca_to_ks` (`qs_ks_atom.F`) does
  the pointwise subtraction `V_H[ρ_a](r_g) − V_H[ρ̃_a +
  ρ_0,a](r_g)` on each radial-grid point before projecting
  onto the AO basis. vibe-qc M3b must do the same; subtraction
  at the integrated-matrix level is the canonical
  paper-only-implementer mistake.

- **D5, V_ne convention for M3b. Both paths are now built; the
  default stays (a) Ewald.** Two paths supported:
  (a) the M3a Ewald lattice V_ne (`compute_nuclear_lattice_ewald`),
  selected via ``v_ne_convention="ewald"`` (default); and
  (b) the CP2K-style `Z_a · [erfc(α r) + erf(α r)] / r` split,
  selected via ``v_ne_convention="smeared_erfc"``,
  ``smeared_v_ne_full_gamma`` in
  ``python/vibeqc/periodic_gapw_smearing.py`` decomposes V_ne as
  long-range FFT-Poisson on the smeared ρ_core, plus libint's
  erfc-attenuated nuclear-attraction one-electron operator
  (``compute_nuclear_erfc_lattice``), plus the standard PW-DFT
  α-correction ``c_α · S`` where
  ``c_α = π · (Σ Z) / (α² · V_cell)``.

  Both are physically equivalent on neutral cells: an SCF parity
  sweep over He / H2 STO-3G in a 16-bohr cube at
  ``α ∈ {0.5, 1.0, 1.5, 2.0, 2.5}`` agrees to < 1 mHa, and to
  < 1 µHa on H2 across the whole sweep. The α-correction is
  empirically pinned by the same sweep and matches the closed-
  form formula exactly. Default stays (a) because it's the path
  already exercised through M3a; (b) is the convention the GAPW
  augmentation work plugs into, every Hartree-class quantity
  there will route through the FFT-Poisson kernel.

**Cutoff-driven sizing.** A user-facing PW cutoff `E_cut` (Ry
or Ha) sets `|G_max| = √(2 E_cut)` in atomic units, from which
the grid extents `Nx, Ny, Nz` are determined by the reciprocal-
lattice metric. Auto-default: probe the basis for the tightest
primitive exponent `α_max` and set `E_cut = K · α_max`. Calibration
note (B2, audit): CP2K's `REL_CUTOFF` is applied to the *fused*
product exponent `ζ_p = ζ_a + ζ_b ≈ 2 · α_max` after the
Gaussian-product theorem, not to the orbital `α_max`. vibe-qc's
single-grid default `E_cut = 60 · α_max` Ha (no multigrid; no
Gaussian-product fusion) is roughly equivalent to CP2K's
recommended `REL_CUTOFF = 60 Ry ≈ 30 Ha` applied to `ζ_p`, i.e.,
deliberately conservative. The docstring of
`recommend_cutoff_from_basis` should make this calibration
explicit rather than handwaving "CP2K-style". Users override via
`cutoff_ha=` on the driver.

**Grid life-cycle.** The grid is constructed once per SCF when
the `GAPWJKBuilder` is built; carried as an immutable
`PlaneWaveGrid` Python object that owns the (Nx, Ny, Nz),
reciprocal-lattice metric, and a cached FFTW3 plan (plan
caching is already in
[`fft_poisson.cpp`](../cpp/src/fft_poisson.cpp), mutex-guarded).
Atomic radial grids are owned by the `PAWAtomData` /
`GAPWAtomData` records.

### Q3, Coexistence with Gaussian-only routes

**Dispatch.** Three-state input flag, `jk_method ∈
{"auto", "gdf", "bipole", "direct", "fft_poisson", "gpw", "gapw"}`:

- `"gpw"` / `"gapw"` → new driver `periodic_gapw.py`.
- `"auto"` keeps its current behaviour
  ([periodic_jk_method.py:178](../python/vibeqc/periodic_jk_method.py)):
  GDF for closed-shell, BIPOLE for open-shell. **No GAPW in
  AUTO at first cut.** Promote later, after the cross-system
  parity track is solid.
- `validate_jk_method` gets `GPW` / `GAPW` cases that hard-error
  on combinations not yet wired (e.g., GPW without
  pseudopotentials on heavy atoms, GPW alone is
  pseudopotential-only).

**Reproducibility.** As today, the *resolved* method (not
`AUTO`) is logged into the banner and `.system` manifest. The
GAPW route also logs the PW cutoff, grid `(Nx, Ny, Nz)`, and
the augmentation-sphere parameters. This is what a future
re-run reads to reproduce; AUTO's heuristic version is not
load-bearing.

**Test coexistence.** Every existing periodic-SCF test stays
on its current method. GAPW gets its own
`tests/test_periodic_gapw_*.py` files. Any test that breaks
because of dispatch-rule changes is ours to fix or back out
before pushing (the brief).

### Q4, Where does ACE plug in once GAPW exists?

**ACE is a decorator over a K-builder, not a new kernel.**
Lin Lin (2016) reformulates the non-local exchange operator
into a low-rank `K_ACE = Σ_k |ξ_k⟩⟨ξ_k|` form that can be applied
in `O(N)` per orbital. The compression rank scales with
occupied space, not basis size; the per-SCF construction is one
heavy K-on-orbitals action plus a Cholesky.

So the plug-in shape is:

```
inner_K = ...                       # any concrete K (direct, RI, COSX, …)
ace_K   = ACEKBuilder(inner_K, rank=n_occ, refresh_every=k_iters)
ace_K.build_K(D)                    # cheap; uses cached compression
```

- and the `GAPWJKBuilder` (or any other periodic JKBuilder)
opts in via composition rather than inheritance. This means
ACE is **not GAPW-specific**: the same machinery is what the
v0.9.x locality-exploitation chat (ADMM / OT) consumes for
periodic hybrids. We design the API so that chat can pull it
in without a GAPW dependency.

For M5 we ship a minimal `ACEKBuilder` that wraps the existing
direct-K, validated against PBE0 / HSE06 on a small molecular
test set first (where reference Ks are cheap), THEN periodic.

## 3. PAW pseudopotential data, license + provenance

CLAUDE.md §1 is the binding rule: "Before bundling, vendoring,
or fetching anything new: check the redistribution terms. If
the terms are unclear or restrictive: don't bundle. Use the
on-demand fetcher pattern (modeled on `vqfetch`)."

The honest reading of the candidate PAW dataset families:

| Dataset family | License (as of 2026-05) | Bundle? |
|---|---|---|
| **GBRV-PAW** (Rutgers; Garrity, Bennett, Rabe, Vanderbilt) | CC-BY-SA 4.0 (per dataset metadata, distributed at `physics.rutgers.edu/gbrv/`) | **NO**, `SA` (share-alike) is restrictive against MPL-2.0 combined-work concerns; verify per-record before any redistribution. |
| **ABINIT JTH-PAW** (Jollet, Torrent, Holzwarth) | LGPL (per ABINIT) for most; some CC-BY | **NO**, license varies per record; on-demand fetch with per-record provenance is the only safe pattern. |
| **GPAW datasets** | GPL | **NO**, incompatible with MPL-2.0 bundling. |
| **PSlibrary** (Quantum ESPRESSO) | Mixed; many GPL | **NO**, same as GPAW. |
| **VASP POTCARs** | Proprietary, redistribution prohibited | **NO**, never. |

**Decision (signed off 2026-05-24): bundle nothing; build the
on-demand fetcher as part of M5 scope.** The vqfetch-style
fetcher gets pulled forward into M5 rather than deferred to
v0.11+, so M5 ships with a clean license surface AND no user
burden.

Concrete M5 fetcher shape (mirroring `vqfetch`):

- **`python/vibeqc/paw_fetcher.py`**, fetch + cache module.
  XDG-cached at `~/.cache/vibe-qc/paw/<dataset>/<element>.xml`
  (or equivalent per-format on-disk layout). Fetch backends
  for GBRV (Rutgers HTTP) and ABINIT JTH (LibPAW XML
  archive). Per-record `Provenance` bundle: source URL, DOI,
  license string, SHA256 hash, fetch timestamp.
- **Provenance threading.** Every fetched record's
  `Provenance` is preserved into the SCF log + `.system`
  manifest, same pattern as the existing `vqfetch` plumbing
  for basis sets / ECPs (CLAUDE.md §1, §8).
- **Default recommended dataset:** GBRV-PAW for the small
  test set; user can swap to JTH on opt-in via
  `paw_dataset="jth"`.
- **GPL / proprietary datasets explicitly refused.** The
  fetcher will not download GPAW datasets, PSlibrary GPL
  records, or VASP POTCARs, license filter at the fetcher
  boundary, with a clear error message pointing at the
  policy in `docs/license.md`.
- **`docs/license.md`** gets a new section explaining the
  per-record license surface, mirroring the existing basis-
  set / ECP entries; the fetched records are *not* bundled
  in the wheel.
- **Local-supplied fallback.** `paw_data_path=` remains
  available for users who want to point at a private /
  in-house PAW dataset; the fetcher is the *default* path,
  not the only path.

Risk to flag at M5 planning time: the fetcher carries
non-trivial implementation cost (HTTP+cache+verify+filter)
that the original "user-supplied only" path avoided.
Acceptable cost for a clean final surface; flagged here so
M5 is scoped honestly.

## 4. Architecture sketch (Python module layout)

```
python/vibeqc/
  periodic_gapw.py            ← M2/M3 driver entry point
                                run_periodic_gapw_rhf(), etc.
  periodic_gapw_grid.py       ← M1 PlaneWaveGrid + cutoff heuristics
  periodic_gapw_density.py    ← Gaussian-on-grid collocation
                                (uses cpp/src/periodic_pw/collocate.cpp)
  periodic_gapw_augment.py    ← M3 dual-grid Hartree decomposition
  periodic_gapw_paw.py        ← M5 PAW projector machinery
  paw_fetcher.py              ← M5 on-demand PAW dataset fetcher
                                (vqfetch-style; XDG-cached)
  exchange_ace.py             ← M5 ACE decorator (not GAPW-specific;
                                shared with locality chat)

cpp/src/periodic_pw/
  collocate.cpp               ← Gaussian-on-grid collocation
                                (the M2 hot path)
  fft_poisson.cpp             ← already exists; reused as-is
  paw_projector.cpp           ← M5

tests/
  test_periodic_gapw_grid.py        # M1
  test_periodic_gapw_collocate.py   # M1
  test_periodic_gapw_hartree.py     # M1 (Hartree-only J vs Ewald-3D)
  test_periodic_gapw_lda_si.py      # M2 (Si bulk vs CP2K)
  test_periodic_gapw_lda_mgo.py     # M2 (MgO conv vs CP2K)
  test_periodic_gapw_pbe_al.py      # M4 (metallic Al fcc)
  test_periodic_gapw_pbe0_mgo.py    # M5 (hybrid via ACE)

docs/user_guide/
  gapw.md                     ← user-guide page (lands at M2)
```

## 5. Validation oracle (open ask)

The brief specifies **CP2K** as the parity oracle. The existing
parity infrastructure ([`examples/regression/core/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/core/))
ships subprocess runners for **CRYSTAL14** and **PySCF.pbc** but
**not CP2K**. Both existing oracles are GTO-only; neither can
validate a plane-wave Hartree-J, comparing GAPW against a GTO
code is non-trivial because the cell-coordinate gauge and
Madelung treatment differ. CP2K, by contrast, *is* the
reference implementation of GAPW (Lippert-Hutter) and the
natural oracle.

**Recommendation (asks the release chat to sign off):** add
`examples/regression/core/runner_cp2k.py` as part of M1, so M2
parity numbers can land. The runner follows the existing
subprocess pattern; no `import cp2k` anywhere in `python/vibeqc/`
or `cpp/` (CLAUDE.md §10). License: CP2K is GPL, but as an
external program invoked via subprocess and parsed at the
output-file boundary, it is a build/dev tool, same status as
`pyscf` / `crystal14`, and outside the vibe-qc runtime
dependency surface. The `.system` manifest will record CP2K as
the external-validation boundary, mirroring the existing
PySCF / CRYSTAL pattern.

Until that runner lands, M1's Hartree-only test (Ewald-3D
nuclear-repulsion energy recovery on a known cell) gives an
internal-consistency target that does not need an external
oracle. Real GPW parity (M2+) is gated on the runner.

## 6. Banner / library-version probing

[`python/vibeqc/banner.py`](../python/vibeqc/banner.py)
`library_versions()` now probes FFTW3 (fixed `0904f84bd`,
2026-05-24) alongside libint, libxc, spglib, and libecpint,
matching the `find_package(FFTW3 CONFIG REQUIRED)` dependency in
`cpp/CMakeLists.txt`. This closed the gap M1 originally called
out per CLAUDE.md §6: `library_versions()`, the banner format
string, the banner unit test, and the sample-output blocks in
[`docs/release_process.md`](release_process.md) and
[`docs/installation.md`](installation.md).

## 7. Out-of-design (deliberately deferred)

- Multi-k GAPW (designed-for; not first-cut).
- GAPW analytical gradients (a separate chat once the energy
  surface is stable).
- ADMM (locality chat owns it; ACE is the closest sibling).
- GPU FFT backends (the FFT abstraction is the prerequisite).
- A k-resolved PAW-projector cache (a multi-k follow-on).

## 8. (was: asks back to the release chat, now resolved; see §9)

## 9. Decisions log

The five M0 asks, resolved by the maintainer 2026-05-24:

| # | Decision | Resolution |
|---|---|---|
| 1 | CP2K subprocess oracle | **Yes**, add `examples/regression/core/runner_cp2k.py` as part of M1. Mirrors the existing runner_pyscf.py / runner_crystal.py pattern. CP2K is GPL but as a subprocess parsed at the output-file boundary it's a build/dev tool, not a runtime dep. |
| 2 | PAW dataset bundling | **Bundle nothing; build the on-demand fetcher into M5** (pulled forward from v0.11+). M5 ships with a vqfetch-style `paw_fetcher.py`: XDG-cached, per-record provenance, GPL / proprietary filter at the boundary. `paw_data_path=` remains as a local-supplied fallback. |
| 3 | Two enum values vs one | **Two enum values**, `PeriodicJKMethod.GPW` and `PeriodicJKMethod.GAPW` as distinct values. GPW = pseudo-only Hartree route; GAPW = all-electron via augmentation. |
| 4 | AUTO promotion in M2-M4 | **No**, GAPW stays opt-in through M4. Promotion to AUTO is a deliberate decision after cross-system parity is broadly demonstrated; revisit at M5. |
| 5 | FFTW3 banner probe timing | **Land at M1** alongside the PlaneWaveGrid + collocation infrastructure. Extends `library_versions()`, banner format string, banner unit test, and the sample-output blocks per CLAUDE.md §6. |

## See also

- [`docs/roadmap.md`](roadmap.md), Acceleration program,
  `v0.10.x` row.
- [Archived `HANDOVER_GDF_V0_11_2026_05_29.md`](https://vibe-qc.com/docs/)
  - GDF route state (the closed-shell Γ-only Gaussian route
  we coexist with).
- [`docs/user_guide/bipole.md`](user_guide/bipole.md)
  - BIPOLE route reference (the multi-k open-shell Gaussian route
  we coexist with).
- [`docs/license.md`](license.md), bundled-data inventory; the
  PAW row will be added here once M5 lands.
- Lippert & Hutter, *Theor. Chem. Acc.* 103, 124 (1999),
  GAPW formulation.
- Blöchl, *Phys. Rev. B* 50, 17953 (1994), PAW.
- Lin Lin, *J. Chem. Theory Comput.* 12, 2242 (2016), ACE.
