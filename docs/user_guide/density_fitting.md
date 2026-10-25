---
myst:
  html_meta:
    "description": "Density fitting (RIJ / RIJK / RIJCOSX) in vibe-qc: the Coulomb-metric RI theory, three-index tensors and error bounds, chain-of-spheres exchange, JKBuilder polymorphic dispatch, recommended aux bases, RIJCOSX energy validated to 0.26 mHa vs ORCA on glycine def2-TZVP."
    "og:title": "vibe-qc, density fitting (RIJK + RIJCOSX)"
    "og:description": "JKBuilder polymorphic Fock build (FourIndex / DF / COSX). Coulomb-metric RI theory and chain-of-spheres exchange. Recommended aux-basis table per orbital basis and the element-coverage guard."
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

All three kernels are production routes for molecular SCF. The status
vocabulary used across the manual is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

## Theory

**The resolution of the identity.** Every two-electron integral in
Hartree-Fock and hybrid DFT is a Coulomb interaction between two orbital
product densities, $\rho_{\mu\nu}(\mathbf r) = \chi_\mu(\mathbf r)\chi_\nu(\mathbf r)$,
and there are $O(N^2)$ such products for $N$ basis functions, hence
$O(N^4)$ integrals. Density fitting expands each product in an
auxiliary basis $\{P(\mathbf r)\}$ of size $M \approx 3N$:

$$
\rho_{\mu\nu}(\mathbf r) \approx \tilde\rho_{\mu\nu}(\mathbf r)
  = \sum_P C^P_{\mu\nu}\, P(\mathbf r).
$$

Whitten ({vibeqc-cite}`whitten_1973`) and Dunlap
({vibeqc-cite}`dunlap_1979`) showed that the right way to choose the
coefficients is to minimise the Coulomb self-repulsion of the fitting
error, $(\rho - \tilde\rho\,|\,\rho - \tilde\rho)$, which gives
$C^P_{\mu\nu} = \sum_Q (\mu\nu|Q)\,[V^{-1}]_{QP}$ with the Coulomb metric
$V_{PQ} = (P|Q) = \iint P(\mathbf r_1)\, r_{12}^{-1}\, Q(\mathbf r_2)$. Substituting
turns the four-index tensor into a product of three-index tensors,

$$
(\mu\nu|\lambda\sigma) \approx \sum_{PQ} (\mu\nu|P)\,[V^{-1}]_{PQ}\,(Q|\lambda\sigma)
  = \sum_P B^P_{\mu\nu} B^P_{\lambda\sigma},
\qquad
B^P_{\mu\nu} = \sum_Q (\mu\nu|Q)\,[V^{-1/2}]_{QP},
$$

and it is the $N^2 M$ tensor $B$ that `DFJKBuilder` precomputes and
holds in memory (the `O(N^2 M)` column of the table above). This is the
"RI" of RI-J, RI-K and RI-MP2; the same $B$ feeds the correlation
methods with a smaller `-ri` auxiliary basis.

**Error bounds.** Because the Coulomb metric is positive definite, the
fit is variational in the Coulomb energy: for a density $\rho$ the
fitted self-repulsion obeys $(\tilde\rho|\tilde\rho) \le (\rho|\rho)$ and
the error in the Coulomb energy is second order in the fitting error,
$E_J[\rho] - E_J[\tilde\rho] = \tfrac12 (\rho-\tilde\rho\,|\,\rho-\tilde\rho) \ge 0$.
RI-J energies therefore lie slightly above the exact ones and converge
quadratically with the quality of the auxiliary basis; with the
Eichkorn-Ahlrichs fitting sets the error is a few tens of µHa per atom
in the total energy and far less in energy differences
({vibeqc-cite}`eichkorn_rij_1995`, {vibeqc-cite}`eichkorn_rij_1997`).
Exchange contracts $B$ over the "wrong" index pair,
$K_{\mu\nu} = \sum_P \sum_i B^P_{\mu i} B^P_{\nu i}$ with
$B^P_{\mu i} = \sum_\lambda B^P_{\mu\lambda} C_{\lambda i}$, so the same
argument does not bound its error; that is why the JK fitting bases of
Weigend ({vibeqc-cite}`weigend_jkfit_2002`) are larger than the
Coulomb-only ones and why `kind="jk"` and `kind="ri"` are different
tables below. Fitting is exact in the limit of a complete auxiliary
basis, and the coverage guard on this page exists because an auxiliary
basis with *no* functions on an atom is the one case where the fit is
silently and grossly wrong rather than slightly biased.

**Cost.** With $B$ in hand, the Coulomb matrix costs $O(N^2 M)$ per
iteration, $J_{\mu\nu} = \sum_P B^P_{\mu\nu} \sum_{\lambda\sigma} B^P_{\lambda\sigma} D_{\lambda\sigma}$,
and exchange $O(N^2 M\, n_{\rm occ})$ through the half-transformed
$B^P_{\mu i}$. RI-K is therefore still formally quartic, and its
three-index intermediates are what fills memory first on large hybrid
DFT jobs.

**Chain-of-spheres exchange (COSX).** Neese and coworkers
({vibeqc-cite}`neese_rijcosx_2009`) replaced the fitted exchange by a
*seminumerical* one: one of the two electron coordinates is integrated
on a molecular quadrature grid $\{\mathbf r_g, w_g\}$, the other
analytically,

$$
K_{\mu\nu} \approx \sum_g w_g\, \chi_\mu(\mathbf r_g)
  \sum_\lambda A_{\nu\lambda}(\mathbf r_g)\, F_{\lambda g},
\qquad
A_{\nu\lambda}(\mathbf r_g) = \int \frac{\chi_\nu(\mathbf r')\chi_\lambda(\mathbf r')}{|\mathbf r' - \mathbf r_g|}\, d\mathbf r',
\qquad
F_{\lambda g} = \sum_\sigma D_{\lambda\sigma}\chi_\sigma(\mathbf r_g).
$$

$A$ is an ordinary nuclear-attraction integral with a unit charge at the
grid point, so the Coulomb singularity is handled analytically and the
grid only has to resolve the smooth orbital products. The "chain of
spheres" is the screening: around each grid point only the basis
functions whose value is non-negligible there (a sphere per shell)
enter, and only density-coupled shell pairs survive, so the work per
grid point stops growing with the molecule and the build scales as
$O(N^2 M_{\rm grid})$ formally and close to linearly in practice. The
overlap-fitted variant of Izsák and Neese
({vibeqc-cite}`izsak_neese_cosx_2011`) replaces the grid values
$X_{\mu g} = \chi_\mu(\mathbf r_g)$ by $\tilde X = S Q^{-1} X$, where
$Q_{\mu\nu} = \sum_g w_g \chi_\mu(\mathbf r_g)\chi_\nu(\mathbf r_g)$ is the
numerical overlap, so that the quadrature reproduces the analytic
overlap exactly; that is `CosxVariant.FITTED`, the robust default here.
The pruned multi-tier grids that vibe-qc uses for the SCF progression
are the 2021 AngularGrid family of Helmich-Paris, de Souza, Neese and
Izsák ({vibeqc-cite}`helmich_paris_cosx_2021`).

**Why RIJCOSX scales better than RIJK.** RI-J is cheap and accurate,
so both routes keep it. The difference is exchange: RI-K needs the
$N^2 M$ tensor and an $O(N^2 M n_{\rm occ})$ contraction every
iteration, while COSX needs only AO values on a grid and a screened
$O(N^2 M_{\rm grid})$ loop with a small prefactor. For hybrid DFT above
roughly a thousand basis functions the exchange build is the wall-clock
bottleneck, and COSX is where the order-of-magnitude savings come from
(the cellulose-hexamer comparison below).

**Where it fails.** RI errors are systematic, not random: they cancel
in energy differences on the same auxiliary basis and reappear when two
codes use different fitting sets. COSX adds grid noise: the SCF
commutator floor sits between 1e-6 and 1e-7, so a run that demands
`conv_tol_grad` below that has to fall back to a denser grid, and the
COSX gradient neglects the grid-motion and weight-derivative terms
(the gradient validation numbers below quote that band explicitly).
Neither route pays off on small molecules, where the four-index build
is already fast and exact.

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
from vibeqc import run_job

result = run_job(
    mol,
    basis="def2-svp",
    method="rhf",
    density_fit=True,
    aux_basis="def2-svp-jk",
    cosx=True,                    # turns the DF path into RIJCOSX
    output="rijcosx_rhf",
)
# JKBuilder = COSXJKBuilder
```

For hybrid DFT switch to `method="rks"` with a functional:

```python
result = run_job(mol, basis="def2-tzvp", method="rks", functional="b3lyp",
                 density_fit=True, aux_basis="def2-tzvp-jk", cosx=True,
                 output="rijcosx_b3lyp")
```

The same keywords work with `method="uhf"` and `method="uks"` (open shell). The
option objects (`RHFOptions` and friends) carry the same fields as attributes,
set on a default-constructed object (`opts = RHFOptions(); opts.density_fit =
True`); they are not accepted as constructor keywords.

## Auto-picking an aux basis

If you don't want to remember the matching aux for each orbital
basis, use the helper:

```python
from vibeqc import default_aux_basis_for

orbital = "def2-tzvp"
aux = default_aux_basis_for(orbital, kind="jk")
# → "def2-tzvp-jk"
```

`run_job(density_fit=True)` calls this for you when `aux_basis` is empty,
and since v0.17.7 so do every route that reaches the bare SCF drivers from
that request: the geometry optimiser (`optimize=True`), the `VibeQC` ASE
calculator given an options struct with `density_fit=True`, and
`compute_hessian_fd`. Before that only the single point resolved the aux and
an optimisation failed with "density_fit=true requires aux_basis" (#362). An
orbital basis with no registered default (`sto-3g`, Pople bases, `pob-*`)
still raises that explicit error; pass `aux_basis=` there.

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
multi-stage progression) and to **ROHF and ROKS** (`ROHFOptions(cosx=True,
cosx_grid_level=...)` / `ROKSOptions`, single-grid; `run_job(cosx=True)`
sets them for `method="rohf"` / `"roks"` as it does for the C++ drivers,
#361). Every post-HF method that builds on a
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

The authoritative numbers are kept in [Features](../features.md); this
table repeats them.

| Quantity | Result |
|----------|--------|
| RHF, 4-index versus RIJK (def2-tzvp-jk) | 0.001 mHa (the RI-J/K fitting error) |
| RIJCOSX SCF **energy** versus ORCA 6.1.1 RIJCOSX | **0.26 mHa** |
| RIJCOSX analytic gradient versus finite differences of the RIJCOSX energy | agrees up to the neglected grid-motion band: ~2e-4 Ha/bohr on H2O/def2-TZVP, ~3e-3 worst case on OH/def2-SVP (`tests/test_rijcosx_gradient_fd.py`) |
| RIJCOSX analytic gradient versus ORCA 6.1.1 EnGrad | \|grad\| 0.08696 versus 0.08703 Ha/bohr; componentwise RMS 3.3e-3, max 1.1e-2 Ha/bohr |

An earlier revision of this table claimed 0.13 mHa for the energy and
"0.13 mHa/bohr" for the gradient on the strength of the energy
comparison alone; the gradient claim was unproven and, before
2026-07-29, wrong. The present gradient residual against ORCA is the
documented fitted-Fock response term plus the neglected grid-motion and
weight derivatives (see the `compute_cosx_k_gradient_contribution`
notes in `cpp/include/vibeqc/cosx.hpp`); the gradient is consistent
with vibe-qc's own RIJCOSX energy surface to the band quoted above,
which is what a geometry optimisation needs.

The energy agreement is within the "sub-mHa" target Neese 2009 set for
COSX. For larger systems (e.g. cellulose hexamer / def2-TZVP), the COSX
path becomes ~10× faster than 4-index for hybrid DFT SCF + analytic
gradient combined.


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
`CHANGELOG.md` under `[Unreleased]` for v0.8.0.

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
opts = RHFOptions()              # likewise UHFOptions / RKSOptions / UKSOptions
opts.density_fit = True          # → DFJKBuilder
opts.aux_basis = "def2-tzvp-jk"
opts.cosx = True                 # → COSXJKBuilder (paired with density_fit)
opts.cosx_grid_level = -1        # GridX tier; -1 auto-selects by basis
# opts.cosx_grid is the GridOptions used when cosx_grid_level is 0
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

Cite for any published RIJ / RIJK / RIJCOSX work (the `.bibtex`
sidecar of a `run_job` call assembles the route's entries for you):

* **RI / DF in Hartree-Fock**: {vibeqc-cite}`whitten_1973`;
  {vibeqc-cite}`dunlap_1979`; {vibeqc-cite}`eichkorn_rij_1995`;
  {vibeqc-cite}`eichkorn_rij_1997`.
* **RI-JK exchange fitting and the JK auxiliary bases**:
  {vibeqc-cite}`weigend_jkfit_2002`; the def2 JK sets are Weigend,
  *Phys. Chem. Chem. Phys.* **8**, 1057 (2006).
* **RI auxiliary bases (def2 family)**: Hellweg, Hättig,
  Höfener, Klopper, *Theor. Chem. Acc.* **117**, 587 (2007).
* **COSX**: {vibeqc-cite}`neese_rijcosx_2009`; overlap-fitted COSX:
  {vibeqc-cite}`izsak_neese_cosx_2011`; the GridX tiers:
  {vibeqc-cite}`helmich_paris_cosx_2021`.
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


## Evaluated states from the native Gamma RHF/RKS API

`run_rhf_periodic_gamma_gdf` returns `result.fock_build` for bare-fit RHF/RKS in
3D molecular-limit cells, with no active symmetry operations, stabilization or
Fock reduction. A 3D cell is a molecular-limit cell when no atom pair reaches a
periodic image within the integral cutoff. A lattice vector longer than the
cutoff is not enough: in a box just longer than the cutoff, atoms near
opposite faces can still be within it of each other's images. Such a cell runs
the bulk Gamma GDF Hamiltonian instead (#98).
The builder owns the fitted tensor and one-electron matrices. Its receipt
records the actual auxiliary shells, fitting threshold, modrho setting and
resolved integral/nuclear options. Existing libint fitting is unchanged.

RKS adds the joint libxc energy and potential for the evaluated density. Its
receipt records the actual functional, grid points/weights, lattice options,
AO shells and XC density convention. Changes to live XC inputs require a new
builder; completed states own their values. The XC energy remains an explicit
component and is never inferred from half a density-potential trace. PATOM
uses a separate full-HF build and eigensolver history, including for pure DFT.
Pure functionals omit the fitted exchange contraction during KS builds;
global hybrids retain the functional's exchange fraction.

The returned orbitals and occupations define `result.density`, including when
SCF reaches its iteration limit. The physical Fock and energy are evaluated
on that density. Orbital energies are its physical Rayleigh quotients;
`next_density` records the further unaccelerated orbital/occupation update.
Convergence checks this density closure as well as energy and stationarity.
Legacy arrays remain writable, separate from the immutable build.

At finite temperature, entropy belongs to the returned occupations and
`free_energy = energy - smearing_temperature * entropy`. The SCF trace records
free energy; the FockBuild and QVF energy record the physical internal energy.
Dense and Davidson solves both request the complete retained spectrum for
Fermi occupations. Positive `max_iter` is required on this migrated route.

This state migration does not change fitting accuracy or admit additional
methods. External XC providers, RSGDF, private slab and torus routes, symmetry-projected sources
and multi-k GDF retain their existing behavior and do not yet provide this
FockBuild result. Native bounded MANTA sources and BIPOLE domains remain
separately qualified.
