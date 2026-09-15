---
myst:
  html_meta:
    "description": "Density fitting (RIJ / RIJK / RIJCOSX) in vibe-qc, JKBuilder polymorphic dispatch, recommended aux bases, validated to 0.13 mHa vs ORCA on glycine def2-TZVP."
    "og:title": "vibe-qc, density fitting (RIJK + RIJCOSX)"
    "og:description": "JKBuilder polymorphic Fock build (FourIndex / DF / COSX). Recommended aux-basis table per orbital basis. Known DF gradient bug above 50 BFs (def2-TZVP class) flagged loudly."
---

# Density fitting (RIJ / RIJK / RIJCOSX)

vibe-qc's molecular SCF Fock build is dispatched through a
polymorphic `JKBuilder` interface (introduced at v0.8.0). Three
concrete kernels cover the production paths:

| Kernel | When to use | Memory | Wall-clock |
|--------|-------------|--------|-----------|
| `FourIndexJKBuilder` | small systems (≤ ~250 BFs); reference / debugging | O(N⁴) | O(N⁴) per iter |
| `DFJKBuilder` (RIJK) | medium systems (~250-1000 BFs); hybrid DFT | O(N²·M) | O(N³) per iter |
| `COSXJKBuilder` (RIJCOSX) | large hybrid-DFT systems (~500-3000 BFs) | O(N²·M) | O(N²·M) per iter |

(`N` = orbital basis size; `M` = aux basis or grid points.)

The JKBuilder dispatch is automatic, set the right
`RHFOptions` flags and vibe-qc picks the kernel. The interface
is also the insertion point for the **periodic-Γ** JKBuilders
(see [multi-k SCF](multi_k_scf.md)) so the same
infrastructure drives molecular and periodic-Γ Fock builds.
[`examples/jk_builder/periodic_gamma_via_jk_builder.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/jk_builder/periodic_gamma_via_jk_builder.py)
makes that sharing explicit: it drives a Γ-only periodic RHF SCF by
building `Hcore`/`E_nuc`/`S` from lattice-summed integrals and a
`jk = vq.make_periodic_gamma_jk_builder(basis, system, lattice_opts)`,
then calls the same `vq.run_rhf_scf_with_jk(...)` C++ entry point the
molecular `run_rhf` wraps: DIIS, damping, level-shift, and the
quadratic fallback all come along unchanged.

## Quick start

### Direct (four-index, the default)

```python
from vibeqc import Atom, Molecule, run_rhf, RHFOptions

mol = Molecule([Atom(8, [0, 0, 0]),
                Atom(1, [0,  1.43, -0.98]),
                Atom(1, [0, -1.43, -0.98])])

result = run_rhf(mol, basis="def2-svp", options=RHFOptions())
# JKBuilder = FourIndexJKBuilder (default)
```

### Density fitting (RIJK)

```python
opts = RHFOptions(
    density_fit=True,
    aux_basis="def2-svp-jk",     # any libint-recognised JK-fit aux
)
result = run_rhf(mol, basis="def2-svp", options=opts)
# JKBuilder = DFJKBuilder
```

### RIJCOSX (RI-J for Coulomb + COSX for exchange)

```python
opts = RHFOptions(
    density_fit=True,
    aux_basis="def2-svp-jk",
    cosx=True,                    # turns DF path into RIJCOSX
    # cosx_grid defaults to default_cosx_grid_options()
)
result = run_rhf(mol, basis="def2-svp", options=opts)
# JKBuilder = COSXJKBuilder
```

For hybrid DFT replace `run_rhf` with `run_rks`:

```python
from vibeqc import run_rks
result = run_rks(mol, basis="def2-tzvp", functional="b3lyp",
                 options=RHFOptions(density_fit=True,
                                    aux_basis="def2-tzvp-jk",
                                    cosx=True))
```

The same flags work for `run_uhf` / `run_uks` (open-shell).

## Auto-picking an aux basis

If you don't want to remember the matching aux for each orbital
basis, use the helper:

```python
from vibeqc import default_aux_basis_for

orbital = "def2-tzvp"
aux = default_aux_basis_for(orbital, kind="jk")
# → "def2-tzvp-jk"
```

Recommended aux bases per orbital basis (the helper's table):

| Orbital basis | Recommended `aux_basis` (kind="jk") | Recommended `aux_basis` (kind="ri") |
|---------------|--------------------------------------|--------------------------------------|
| `def2-svp` | `def2-svp-jk` | `def2-svp-rifit` |
| `def2-svpd` | `def2-svp-jk` | `def2-svpd-rifit` |
| `def2-tzvp` | `def2-tzvp-jk` | `def2-tzvp-rifit` |
| `def2-tzvpp` | `def2-tzvpp-jk` | `def2-tzvpp-rifit` |
| `def2-qzvp` | `def2-qzvp-jk` | `def2-qzvpp-rifit` (see note) |
| `cc-pvdz` | `cc-pvdz-jkfit` | `cc-pvdz-rifit` |
| `cc-pvtz` | `cc-pvtz-jkfit` | `cc-pvtz-rifit` |
| `cc-pvqz` | `cc-pvqz-jkfit` | `cc-pvqz-rifit` |
| `aug-cc-pvdz` | `aug-cc-pvdz-jkfit` | `aug-cc-pvdz-rifit` |
| `pob-tzvp` | (no matching aux yet) | (no matching aux yet) |

**Why `def2-qzvp` resolves to `def2-qzvpp-rifit`.** The Basis Set
Exchange record `def2-QZVP-RIFIT` (version 1, data from Turbomole 7.3)
covers only 42 main-group elements -- H to Ca, Ga to Sr, In to Ba and
Tl to Rn -- and every 3d, 4d and 5d transition metal is absent
upstream, not lost in vibe-qc's fetch. BSE itself describes that record
as the "RIMP2 auxiliary basis for def2-QZVPP", and its 42 blocks are
byte-identical to the same elements of `def2-qzvpp-rifit`, which covers
all 72 elements of the def2 RI convention (Hättig 2005 for Li to Kr;
Hellweg, Hättig, Höfener and Klopper 2007 for Rb to Rn). Auto-resolving
`def2-qzvp` to the QZVPP fit therefore changes nothing for a main-group
molecule and lets a def2-QZVP transition-metal DF-MP2 / DF-CC / DLPNO
run proceed instead of being refused by the coverage guard below
(GitLab #483). `aux_basis="def2-qzvp-rifit"` remains selectable by
name for the 42 elements it covers.

`kind="jk"` aux basis = bigger, designed for the Coulomb +
exchange J/K Fock build. `kind="ri"` aux basis = smaller,
designed for the Coulomb-only (correlation-side) RI used in
MP2. Use the right one, JK aux on MP2 is wasteful, RI aux on
SCF is inaccurate.

**`pob-tzvp` does not have a matching aux yet.** Designing a
pob-tzvp-jk fitting basis is a separate-paper item per
[`docs/roadmap.md`](../roadmap.md). Workaround until then:
fall back to `def2-tzvp-jk` (aux basis size scales with
orbital basis size, not orbital identity, so the def2 aux
works as a generic), or use `density_fit=False` for periodic
work.

### Element coverage: the run is refused, not approximated

Not every shipped fitting family covers every element of the
orbital basis it pairs with. The known gaps:

| Auxiliary family | Elements it does **not** cover |
|---|---|
| `cc-pV{D,T,Q,5}Z-JKFIT` | He, Li, Be, Na, Mg; and K, Ca, Sc to Zn |
| `cc-pVnZ-RI`, `cc-pVnZ-RIFIT` | K, Ca (and Sc to Zn at DZ/TZ/QZ) |
| `cc-pV6Z-RI`, `cc-pV6Z-RIFIT` | Li, Be, Na, Mg |
| `aug-cc-pVnZ-RIFIT` | Li, Na, K, Ca (varies by zeta) |
| `def2-qzvp-rifit` (BSE's 42-element record; not auto-selected) | Sc to Zn, Y to Cd, La, Hf to Hg |
| `cc-pwCVnZ-RIFIT`, `aug-cc-pwCVnZ-RIFIT` | H, He, Li, Be, Na, Mg |

The `def2-universal-jkfit`, `def2-universal-jfit` and the other
`def2-*-jk` / `def2-*-rifit` families cover H to Rn.

libint2 does not raise for an element a fitting file omits: it
returns an auxiliary basis with **zero functions on that centre**.
Before v0.15.x that fit was built anyway, and the SCF converged
cleanly, with a tight gradient and no warning, to an energy up to
**49.6 Ha** below the conventional answer (NaH/cc-pVDZ). vibe-qc
now checks coverage before any fitting work and **refuses the
run**, naming the uncovered elements and the auxiliary basis:

```text
run_rhf: auxiliary basis 'cc-pvdz-jkfit' auto-selected for orbital
basis 'cc-pvdz' has no functions on 1 atom of this molecule:
Na (atom 0).
```

The refusal is deliberate: substituting a covering auxiliary
would change the fit under a caller who asked for a specific one.
To proceed, name a covering auxiliary explicitly
(`aux_basis="def2-universal-jkfit"` for the JK/SCF route,
`aux_basis="def2-tzvpp-rifit"` for the RI/correlation route),
switch to a def2 orbital basis, or run with `density_fit=False`.
On LiH, BeH₂, NaH and MgH₂ the `def2-universal-jkfit` control
reproduces the conventional SCF energy to ~1e-5 Ha, so the
density-fitting machinery itself was never the problem; only the
element coverage was.

The same check guards the post-SCF RI route, which the DLPNO
methods and `run_rohf_mp2` reach with no `density_fit` flag at
all, and the periodic GDF auxiliary constructor.

## RIJCOSX: chain-of-spheres exchange

RIJCOSX (Neese, *Chem. Phys.* **356**, 98, 2009; Bykov,
Petrenko, Izsák, Kossmann, Becker, Valeev, Neese, *Mol. Phys.*
**113**, 1961, 2015) replaces the O(N⁴) ERI assembly for the
exchange K matrix with an O(N²·M) seminumerical chain-of-
spheres algorithm:

```
K_{μν} ≈ Σ_g w_g χ_μ(r_g) Σ_ρ A_{νρ}(r_g) Σ_λ D_{λρ} χ_λ(r_g)
```

where `A_{νρ}(r_g) = ∫dr' χ_ν(r') χ_ρ(r') / |r' − r_g|` is
the analytic 1/|r-r'| integral with a unit charge at grid
point `r_g`. RI-J is used for the Coulomb piece; the same
B-tensor that the `DFJKBuilder` precomputes is reused.

### Default COSX grid

```
n_radial =  35   (Treutler-Ahlrichs M4)
n_theta  =   9   (Gauss-Legendre)
n_phi    =  18   (uniform φ)
```

= ~5670 angular pts/atom before Becke pruning, which is
**~1/8 the XC integration grid** (75 × 17 × 36 = ~46 000
pts/atom). Substantially sparser is fine because COSX needs
only the smooth 1/|r−r'| kernel through `χ_μ(r)`, no XC
gradient response.

This default tier matches **ORCA's default GridX**. Override
via `RHFOptions.cosx_grid`:

```python
from vibeqc import GridOptions
opts = RHFOptions(
    density_fit=True, aux_basis="def2-tzvp-jk",
    cosx=True,
    cosx_grid=GridOptions(n_radial=50, n_theta=11, n_phi=23),
                                          # finer than default
)
```

### Variant + grid-level selection

Beyond the grid itself, RIJCOSX exposes three controls that let
you trade COSX accuracy against cost without hand-tuning the
grid. They live on every SCF Options struct (`RHFOptions`,
`UHFOptions`, `RKSOptions`, `UKSOptions`) and on
`GradientOptions`, and are consulted only when `cosx=True`.

| Field | Values | Meaning |
|-------|--------|---------|
| `cosx_variant` | `CosxVariant.AUTO` (default), `STANDARD`, `FITTED` | K build: STANDARD = Neese 2009 seminumerical K, no overlap fit; FITTED = overlap-fitted K (global Q-junction). |
| `cosx_grid_level` | `-1` (SCF default), `0`, `1` to `4` | COSX grid tier: `-1` auto-selects by basis (the SCF-options default since 2026-06-26; see note below); `0` keeps `cosx_grid` as supplied (the `GradientOptions` default); `1` to `4` select the GridX1-4 tiers. |
| `thresh_cosx` | float, default `1e-6` | COSX accuracy target that drives `AUTO` variant selection. |

`AUTO` always resolves to **FITTED**, the robust overlap-fitted build.
The standard (no-fit) build is accurate per grid point on a GridX tier
(DZ gradient ~4e-5 Ha/bohr vs the legacy grid's ~1.5e-3) but it is **not
robust enough to auto-select**: as an auto default it caused RIJCOSX SCF
convergence failures (open-shell especially), so it stays an **explicit
opt-in** (`cosx_variant=STANDARD`) for callers who want its speed and
accept the convergence risk. Both variants apply the one-centre analytic
correction.

When you opt into a GridX tier (`cosx_grid_level` 1 to 4, or `-1`), the auto
mapping is DZ/TZ→GridX2, QZ→GridX3 (GridX1 is the explicit quick-SP tier
only, too coarse to auto-select). Note the GridX2 caveat below.

```python
from vibeqc import CosxVariant
opts = RHFOptions(
    density_fit=True, aux_basis="def2-qzvp-jk",
    cosx=True,
    cosx_variant=CosxVariant.FITTED,  # or AUTO (default)
    cosx_grid_level=3,                # GridX3, recommended for QZ
)
```

The GridX1-4 tiers are the **2021 5-region pruned-Lebedev "AngularGrid"**
grids of Helmich-Paris, de Souza, Neese & Izsák (*J. Chem. Phys.*
**155**, 104109, 2021), Table I. Each tier splits every atom's radial
shells into 5 bands with a distinct Lebedev order per band (sparse core,
dense valence, thinned tail). The Table I **angular point counts are
exact**; the radial counts and the region boundaries (taken as shell-index
fractions) are documented approximations, because ORCA's per-element
Clementi-radius multipliers and differential-evolution radial parameters
are not openly published (ORCA-binary only). The tiers map to ORCA's
DefGrid levels: GridX2 = DefGrid2 (ORCA's default), GridX3 = DefGrid3, etc.

**Multi-stage SCF grid progression.** When you opt into a GridX tier, the
SCF does not use one fixed grid; it steps through the DefGrid stage triple
(coarse to middle to fine) following the published scheme (Helmich-Paris,
de Souza, Neese & Izsák, *J. Chem. Phys.* **155**, 104109 (2021), Sec.
IV.A): the coarse stage converges loosely, the *middle* stage converges to
your SCF tolerance, and the fine (final) grid is used for exactly **one**
post-convergence exchange rebuild that yields the reported energy: the
SCF never iterates on the fine grid. (If a degenerate open-shell system
hits a commutator floor on the middle grid above your tolerance, the SCF
automatically falls back to converging on the fine grid instead.) The SCF
trace in the `.out`/`.perf` shows all stages concatenated; the last
iteration is the final-grid recompute, whose commutator reflects the
inter-grid exchange difference rather than SCF error.
`vibeqc.cosx_grid_stages_for_level(level)` returns the stage grids;
`vibeqc.cosx_grid_options_for_level(level)` returns the fine (last) grid
on which the reported exchange energy is built.

**Reach across methods.** RIJCOSX accelerates the mean-field exchange
build, so it applies to **RHF, UHF, RKS, UKS** (C++ drivers, with the
multi-stage progression) and to **ROHF** (`ROHFOptions(cosx=True,
cosx_grid_level=...)`, single-grid). Every post-HF method that builds on a
mean-field reference (**MP2, RI-MP2, DLPNO-MP2, CCSD, CCSD(T), DLPNO-CC,
TDDFT**) inherits RIJCOSX automatically through its reference SCF: set
`cosx=True` on the reference's options (`rhf_options` / `uhf_options` /
`rohf_options`) and the correlated step runs on the RIJCOSX MOs. There are
no `cosx` fields on the MP2/CCSD option structs because COSX is purely an
SCF concern. The CAS family (CASCI/CASSCF/NEVPT2/CASPT2) is the exception:
it needs exact MO integrals, so its reference is built on the exact/DF path
regardless of `cosx`.

```{note}
**Default (since 2026-06-26): `cosx_grid_level = -1` (AUTO).** When
`cosx=True`, the SCF energy now uses a 2021 GridX tier with the multi-stage
progression by default (DZ/TZ to GridX2, QZ to GridX3): more accurate than
the old legacy grid, and faster. The selection is *conv-tolerance-aware*: if
you tighten `conv_tol_grad` below `1e-6` (the seminumerical commutator-noise
floor sits between 1e-6 and 1e-7), the SCF automatically falls back to the
denser legacy grid, which can reach a tight gradient. So tight-SCF runs and
`conv_tol_grad <= 1e-7` open-shell runs keep their old behaviour
automatically; you do not need to set anything.

`cosx_grid_level = 0` forces the legacy grid; `1..4` force a GridX tier at
any tolerance. **Gradients** (`GradientOptions`, which has no
`conv_tol_grad`) still default to the legacy grid; the ASE calculator's
force path mirrors the SCF's resolved grid so geometry optimisations stay
energy/gradient consistent, but a hand-built `GradientOptions` for a GridX
optimisation should set `cosx_grid_level` to match the SCF.
```

### Validation: glycine / def2-TZVP

| Method | Energy (Ha) | Δ vs ORCA 6.1.1 RIJCOSX |
|--------|-------------|---------------------------|
| RHF, 4-index | −282.823485 | 0.00 mHa (reference) |
| RHF, RIJK (def2-tzvp-jk) | −282.823484 | 0.001 mHa |
| RHF, RIJCOSX | −282.823611 | **0.13 mHa** |
| Analytic gradient max\|Δ\| (RIJCOSX) | - | **0.13 mHa/bohr** |

Hits the standard "sub-mHa accuracy" target Neese 2009 cited.
For larger systems (e.g. cellulose hexamer / def2-TZVP), the
COSX path becomes ~10× faster than 4-index for hybrid DFT
SCF + analytic gradient combined.


### One-centre exact-exchange correction (v0.8.0)

The COSX grid under-samples the 1/r₁₂ Coulomb singularity
for shell quartets where all four basis-function centres lie
on the same atom.  At v0.8.0, vibe-qc applies a one-centre
correction that replaces the COSX-approximate intra-atom
K-matrix blocks with exact four-index ERI contractions:

```
K_corrected[μ∈A, ν∈A] = K_exact_1c[μ,ν]
                       + (K_cosx_full − K_cosx_1c)[μ∈A, ν∈A]
```

The exact one-centre ERIs are precomputed at JKBuilder
construction (cost: one libint quartet evaluation per
intra-atom shell quartet, negligible).  The COSX-approximate
one-centre contribution K_cosx_1c is computed from the same
GridBatches chi cache and the in-tree nuclear-attraction
kernel used by the main COSX-K path, giving bit-identical AO
values.

The correction is applied **once post-SCF-convergence** in all
four molecular SCF drivers, so it does not perturb DIIS or
SOSCF acceleration during iterations.  It is always active
when `cosx=True`; there is no user-facing toggle.

For light atoms (H, C, N, O), the correction is sub-μHa
because the COSX grid already samples intra-atom shell pairs
adequately.  For heavier atoms (3d / 4d transition metals)
with contracted core shells, the improvement can reach
~10⁻⁴ Ha per atom in the total energy.

### What improved between v0.7.x and v0.8.0

RIJCOSX got **~3.65× faster** on the n-hexadecane / B3LYP /
def2-svp / 5-iter anchor between the v0.7.x baseline and the
v0.8.0 tip (matched-load median, M5Max, 10 perf cores: ~603 s →
~165 s wall). Energy bit-identical end-to-end (-416.178230 Ha
at iter 5). Nothing in the user-facing API changes, all the
improvements are automatic when `cosx=True` is set. The
contributing landings, in chronological order:

* **Drop dense per-grid-point A_g**, build F_g block-sparse on
  the fly instead of materialising the full n_bf × n_bf
  analytical-integral matrix per point.
* **OMP `schedule(guided)`** for the per-grid-point loop,
  absorbs the per-point cost variance (chi-magnitude / shell-
  pair-survival distribution swings ~100× between nuclear and
  outer-grid regions). ~15-17 % wall.
* **Per-shell radial cutoff** (Schwarz × distance × dχ) on the
  COSX-K pair loop, drops pair-loop iteration count by 5-50×
  on extended systems. ~11 % wall on n-hex.
* **Reusable per-batch grid + chi-cache infrastructure**
  (`GridBatch` / `GridBatches`), basis-only AO cache evaluated
  once at SCF setup instead of per K-build.
* **Per-batch L1 (primary-shell) hierarchy**, `COSXJKBuilder`
  consumes the chi cache on every `build_K(D)` call. L1-only
  first landing.
* **Per-batch L2 (density-coupled) shell screen**, drops
  pairs where both shells fall outside the batch's density-
  coupled set (Burow & Sierka 2011 § 2; Stratmann-Scuseria-
  Frisch 1996 § 11). **The headline structural win** at ~20 %
  wall over the pre-batch baseline.

The full per-commit measurement table + references lives in
`CHANGELOG.md` under `[Unreleased]` for v0.8.0. Further perf
work (custom Boys-table integral kernel replacing libint) was
recorded in [Archived `handovers/HANDOVER_PERF_OPT.md`](https://vibe-qc.com/docs/).

## DF analytic gradient status

(historical-gradient-bug-status)=
The historical ~115 mHa/bohr `density_fit=True` gradient bug on
glycine / def2-TZVP / RHF closed in v0.8.0: the root cause was a
libint engine-state leak in `compute_3c_eri_gradient_weighted`
(stale derivative-buffer state between inner ket-shell calls).
After the fix the 3c-kernel residual vs finite-difference is
~1e-7 Ha / bohr (the FD truncation floor); the regression guard
is
`tests/test_df_gradient.py::test_df_rhf_gradient_hcooh_def2_tzvp_matches_direct`.
RIJCOSX gradients are independent of this fix and were
unaffected throughout.

One RIJCOSX-specific gap closed in the 2026-05-18 audit
remediation: the unrestricted (UHF / UKS) DF gradient silently
ignored `cosx=True` and returned the plain RI-JK gradient. K now
routes through the chain-of-spheres kernel per spin
(`alpha_kernel = 2 alpha_HF`); regression-pinned in
`tests/test_rijcosx.py`. The ASE calculator and the all-electron FD Hessian
also copy `density_fit` / `aux_basis` / `cosx` off the SCF options onto the
gradient call, so forces differentiate the same JK Hamiltonian the SCF solved.
The ASE force path additionally copies exact XML ECP fields. Molecular ECP FD
Hessians remain unsupported and fail closed before any displacement.

A separate **direct-gradient bug on f-shells with two or more
different second-row elements** (CO / CH₂O / glycine + def2-TZVP)
is still open for the UHF / UKS open-shell path. Closed-shell
direct gradients now auto-route through the DF gradient path
when the basis contains f-functions (`max l >= 3`), fully
thread-deterministic and cross-platform reproducible; the
open-shell auto-route is queued. Fall back to FD on the total
energy for open-shell f-shell systems, or restrict to def2-SVP /
single heavy-atom-type systems. Tracked in
`tests/test_gradient_f_bug.py`.

## When to use which kernel

Decision tree:

* **System ≤ 250 BFs**: `FourIndexJKBuilder` (the default).
  Simpler, no aux-basis dependency, exactly as accurate as the
  4-index ERI. No DF or COSX overhead worth introducing.
* **System ~250-1000 BFs, hybrid DFT**: `DFJKBuilder` (RIJK).
  RIJCOSX is overkill at this size.
* **System > 1000 BFs, hybrid DFT**: `COSXJKBuilder` (RIJCOSX).
  The exchange-K piece is where the wall-clock savings come
  from.
* **System > 1000 BFs, pure GGA DFT** (no exact exchange):
  `DFJKBuilder` with `cosx=False`. Skipping COSX is fine
  because there's no exchange to assemble.
* **Need analytic gradient AND > 50 BFs def2-TZVP-class**:
  any of RIJCOSX / direct / RIJK works (the historical RIJK
  glycine bug closed in v0.8.0). On f-shell basis sets,
  closed-shell direct gradients auto-route through the DF
  gradient path for cross-platform reproducibility; open-shell
  UHF / UKS gradients on f-shells should still fall back to FD.

## API surface

```python
# Python-side flags on RHFOptions / UHFOptions / RKSOptions / UKSOptions:
RHFOptions(
    density_fit=True,            # → DFJKBuilder
    aux_basis="def2-tzvp-jk",
    cosx=True,                   # → COSXJKBuilder (paired with density_fit)
    cosx_grid=default_cosx_grid_options(),
)
```

```cpp
// C++-side factory (cpp/include/vibeqc/jk_builder.hpp):
std::unique_ptr<JKBuilder> make_four_index_jk_builder(const BasisSet& basis);
std::unique_ptr<JKBuilder> make_df_jk_builder(const BasisSet& basis,
                                              const BasisSet& aux);
std::unique_ptr<JKBuilder> make_cosx_jk_builder(const BasisSet& basis,
                                                const BasisSet& aux,
                                                Grid cosx_grid);
```

The strategy interface:

```cpp
class JKBuilder {
public:
    virtual Eigen::MatrixXd build_J(const Eigen::MatrixXd& D) const = 0;
    virtual Eigen::MatrixXd build_K(const Eigen::MatrixXd& D) const = 0;
    virtual Eigen::MatrixXd build_g_rhf(const Eigen::MatrixXd& D,
                                        double alpha_hf = 1.0) const;
};
```

`build_g_rhf` defaults to `J − ½·α·K`; concrete kernels override
when the J/K can fuse work (DF amortises the B-tensor
contraction).

## Citations

Cite for any published RIJ / RIJK / RIJCOSX work:

* **RI / DF in Hartree-Fock**: Whitten, *J. Chem. Phys.* **58**,
  4496 (1973); Dunlap, *J. Chem. Phys.* **71**, 3396 (1979);
  Eichkorn, Treutler, Öhm, Häser, Ahlrichs, *Chem. Phys. Lett.*
  **240**, 283 (1995).
* **JK auxiliary bases (def2 family)**: Weigend,
  *Phys. Chem. Chem. Phys.* **8**, 1057 (2006).
* **RI auxiliary bases (def2 family)**: Hellweg, Hättig,
  Höfener, Klopper, *Theor. Chem. Acc.* **117**, 587 (2007).
* **COSX**: Neese, Wennmohs, Hansen, Becker, *Chem. Phys.*
  **356**, 98 (2009).
* **COSX-K analytic gradient**: Bykov, Petrenko, Izsák,
  Kossmann, Becker, Valeev, Neese, *Mol. Phys.* **113**, 1961
  (2015).

Plus your basis-set + functional citations per the rest of
[`docs/license.md`](../license.md).

## See also

* [`periodic_methods.md`](periodic_methods.md): comparative tour of
  vibe-qc's periodic-SCF kernels (BIPOLE, native GDF, GPW/GAPW)
  and the surface-reactions workflow walkthrough.
* [`functionals.md`](functionals.md), XC functional library
  (B3LYP-VWN5 sidebar, PW1PW, hybrid functional aliases).
* [`multi_k_scf.md`](multi_k_scf.md), periodic-Γ JKBuilder
  + multi-k SCF.
* [`scf_convergence.md`](scf_convergence.md), DIIS / EDIIS+DIIS
  / level-shift / quadratic-fallback that the JKBuilder
  dispatch lands inside.
