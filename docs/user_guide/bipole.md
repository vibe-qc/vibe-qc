# BIPOLE: Ewald-J-split periodic HF and DFT

```{admonition} Feature status
:class: note

The BIPOLE drivers (`run_pbc_bipole_rhf`, `_uhf`, `_rks`, `_uks`)
form vibe-qc's BIPOLE periodic workstream. All four methods support
multi-k. The exact Ewald-J-split route is the supported production
path. The experimental quartet-level bipolar far-field is disabled;
`use_multipole_far_field=True` fails before SCF setup. Its current
two-translation dispatch does not cover the same periodic quartet
domain as the exact three-translation Fock contraction.

Production gradients and fixed-cell atomic optimization use the
finite-difference BIPOLE force path by default. Variable-cell and coupled
atom/cell optimization fail closed. Analytic gradients are available as a
gated research-preview path on the maintained regression surface.

For routine calculations, prefer:

* [`run_rhf_periodic_gamma_gdf`](density_fitting.md) for Γ-only
  hybrid / pure DFT with native Gaussian density fitting.
* [`run_rhf_periodic_multi_k_ewald3d`](multi_k_scf.md) for multi-k
  with 3D Ewald for the long-range Coulomb piece.
```

## What BIPOLE is

CRYSTAL splits the periodic Fock build into two halves and keeps
them in **separate gauges**:

* **One-electron Coulomb** (`V_ne`) and the **nuclear-nuclear
  energy** (`E_nn`), 3D Ewald (point-charge ↔ Gaussian-pair
  doesn't decay on the nucleus side, so direct truncation diverges
  on charged-nucleus crystals like MgO). Both share a single
  Ewald α (a single shared Ewald state).
* **Two-electron Coulomb + exchange** (`F^{2e}` = J + K),
  CRYSTAL's native code uses direct-space BIPOLE screening and
  multipole-far-pair replacement. The current vibe-qc parity path
  implements the same electrostatic gauge as
  `J_SR(ω) + J_LR(ω) + V_bg·S - 1/2 K_full`, where `ω` is the same
  Ewald α used by `V_ne` / `E_nn` and `V_bg = -π N_e/(α²V)`.
  This replaces the older broken `run_rhf_periodic_multi_k_ewald3d`
  composition, whose long-range J did not share CRYSTAL's gauge.

## What today's implementation does

`vibeqc.run_pbc_bipole_rhf` ships:

* The full real-space one-electron pipeline at Ewald gauge:
  $S(g)$, $T(g)$, $V_{ne}(g)$ at `opts.lattice_opts.cutoff_bohr`,
  Bloch-summed to per-k $S(k)$ and $H_{core}(k)$, canonical
  orthogonalisation $X(k)$. The singular erfc part of $V_{ne}$ is
  analytic via libint, and the smooth reciprocal/background part is
  analytic by shifted AO-pair Fourier transforms. A tightened
  unpruned Lebedev grid is still available as an explicit diagnostic
  fallback via `v_ne_grid_options`.

  The **real-space** half of that one-electron Ewald sum is sized from
  the Ewald `α`, not from `nuclear_cutoff_bohr`. Because BIPOLE pins
  CRYSTAL's `α = 2.8 / V^{1/3}`, the screened kernel at the nearest
  image is `erfc(α·L) = erfc(2.8) ≈ 7.5e-5` for a cubic cell of *any*
  side, and it never shrinks with the box, so the driver raises the
  one-electron cutoff to `sqrt(-ln ewald_precision) / α` when your
  `nuclear_cutoff_bohr` falls short of it, and says so in the SCF log.
  Tightening `ewald_precision` therefore widens the real-space
  envelope as well as the reciprocal one. Your `cutoff_bohr`, the
  electronic J/K cell set, is unchanged. Before this (issue #478) a
  fixed cutoff truncated an image tail decaying only as `1/L`, leaving
  a `+0.226·N_e²/L` mHa monopole self-image in the Γ molecular limit.
* A shared-gauge Ewald-J two-electron build by default for 3D
  systems (`use_ewald_j_split=None`, or `True` explicitly):
  short-range J from direct erfc-screened ERIs, long-range J from
  reciprocal AO-pair Fourier transforms, the matching electron
  background potential, and full direct-space exchange.
* The standard SCF inner loop: Bloch-sum, direct real-space
  energy evaluation (`Σ_g D(g)M(g)`),
  optional `_MultiKPulayDIIS`, optional `LEVSHIFT`, diagonalisation,
  optional MOM occupied-subspace reorder, and density rebuild. ODA is
  fail-closed on this route: a line-search mixture generally has no
  single orbital representation, so accepting it would make the returned
  density and orbital payload describe different states.
* The matching nuclear-nuclear Ewald sum so $E_{nn}$ shares the α
  used by $V_{ne}$.

For result auditing, `energy_components[-1]` and `scf_trace[-1]` describe
the terminal density returned by the driver. The separate
`initial_density_energy_components` field preserves the component breakdown
from the density entering the first SCF cycle. In a legacy-gauge SAD run this
is the density convention corresponding to CRYSTAL CYC0; after even one
diagonalisation it is generally not the same state as the terminal row.

The direct erfc build selects potentially contributing lattice pairs before
traversing them. It uses the existing Schwarz and separation tests, preserves
their accepted contributions and summation order, and retains the requested
internal, output, and density domains. This does not loosen a cutoff or an
SCF tolerance. For numerical comparisons, setting
`options.lattice_opts.sr_sparse_traversal = False` retains exhaustive
enumeration with the same screening. Low-level J/K results expose
`cell_triples_considered`, `cell_triples_possible`, and
`shell_quartets_considered` to compare work independently of machine load.
Shell-prefix bounds reject groups before quartet expansion, and surviving
integrals reuse primitive shell pairs in a bounded worker-local cache.
Dense candidate lists switch to a bitmap when the full pair domain fits
within the same 2 MiB scratch budget. Larger domains fall back to exhaustive
enumeration if their candidate list fills.

The loop's Fock is built incrementally (`use_incremental_fock`, the
J$_{SR}$/K$_{SR}$ increments from $\Delta D$ under density-envelope
screening), and the accumulated screening error of that chain makes its
energy differ from a non-incremental build of the same density by a small,
structural amount, 2.1e-8 Ha on LiH/STO-3G (2,2,2) at a 25-bohr cutoff. The
convergence decision is therefore taken on the exact operator (GitLab #116):
when the loop's own criteria are met, the committed density is rebuilt
non-incrementally, its energy and commutator norm are judged against
`conv_tol_energy` and `conv_tol_grad`, and if they fail the incremental
chain is re-synced to that exact build and the loop continues (typically two
more iterations, one of them a $\Delta D = 0$ build). The post-loop refresh
re-applies the same test to the same rebuild, so a loop can no longer exit on
a state the terminal check would refuse, and no tolerance is widened. Both
values are logged and emitted as `scf_terminal_check` structured events
(`phase` `in_loop` / `post_loop`, `energy_loop`, `energy_exact`, `delta_e`,
`grad_norm_loop`, `grad_norm_exact`, `passed`), and a refused run's
`RuntimeError` names the terminal delta and commutator norm against the
tolerances. `scf_trace[-1].delta_e` is that terminal delta.

That exact rebuild is the expensive post-SCF phase. It screens the full
density, whose image support can be much larger than the localized initial
guess or a late density increment. Its cost is measured separately.
Before it starts, the ordinary `.out` file also receives a flushed record
of the provisional SCF energy and iteration, including runs without a
structured log. The phase completion records build status and wall time;
it does not itself certify convergence, which the terminal check judges.
The historical #115 LiH (2,2,2) case with six basis functions spent 22,000 s
there after late iterations of 819 s. The build is announced before it
runs, and `scf_exact_confirmation_begin` records the loop's provisional
energy and iteration first, so a job killed inside it retains that
provisional result in the structured log; `scf_exact_confirmation_end` carries the
wall time. The post-loop `scf_final_density_begin`/`_end` pair then
reports `reused` (no second build) on a converged run, or the one
mandatory cold build on a max-iteration exit if no exact build already exists
for that density. A failed confirmation at the iteration cap also reuses its
exact build and remains non-converged. A build that raises ends
with `status="raised"` rather than a successful end.

`vibeqc.run_pbc_bipole_uhf` uses the same one-electron and Ewald-J
machinery with unrestricted spin densities:

* $D_{tot}(g) = D_\alpha(g) + D_\beta(g)$ drives the Hartree operator.
* $F^{2e}_\alpha(g) = J_{tot}(g) - K[D_\alpha](g)$ and
  $F^{2e}_\beta(g) = J_{tot}(g) - K[D_\beta](g)$.
* The UHF energy is evaluated as
  $E_{2e} = \frac{1}{2}\mathrm{tr}[D_\alpha F^{2e}_\alpha] +
  \frac{1}{2}\mathrm{tr}[D_\beta F^{2e}_\beta]$ in the same
  real-space lattice contraction convention as RHF.
* Even-electron SAD guesses are spin-averaged for CRYSTAL CYC0 parity:
  $D_\alpha = D_\beta = D_{SAD}/2$. The requested multiplicity enters
  through the alpha/beta occupation counts after diagonalisation.

What is still gated:

* True IBZ orbit expansion for the multi-k long-range-J density
  transform. Until that lands, symmetry-reduced Monkhorst-Pack inputs
  are accepted by internally expanding them back to the full mesh for
  Ewald-J correctness. This is a usability bridge, not a performance
  speedup.
* ECP-bearing bases. BIPOLE does not yet consume the pseudopotential
  one-electron operator, effective charges, removed-core count, and ionic
  energy as one Hamiltonian, so public and direct requests fail before SCF.
* Methfessel-Paxton and Marzari-Vanderbilt smearing. BIPOLE implements
  Fermi-Dirac occupations (including the Mermin label); other flavours fail
  rather than being relabelled Fermi-Dirac calculations.
* Periodic Hessians, TDDFT, and COOP/COHP. The former generic paths used
  molecular or fixed-cutoff surrogate operators and now fail before SCF.
  QVF structure, wavefunction, population, and SCF-history content remains
  available, but surrogate DOS/PDOS/COOP/COHP arrays are omitted.
* Closed-shell multi-k READ restarts. A complete conversion from the saved
  Bloch density to the driver's real-space block convention is still needed,
  so the unsupported request fails instead of reading only Gamma.

The driver is an equation- and component-validation surface plus a supported
exact Ewald-J SCF implementation. Results from a different periodic Coulomb
algorithm are useful references, but total-energy agreement is not assumed
until gauge, basis, k mesh, and numerical supports have all been matched.

## Why ship it at all?

Two reasons:

1. **CRYSTAL parity surface.** Comparisons against external reference
   calculations help track the multipole branch's accuracy. The historical
   `tests/demos/` path is no longer present; current diagnostics live in
   [`examples/regression/bipole_parity/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/bipole_parity).

2. **Multipole infrastructure.** Moment, interaction-tensor, dispatch,
   and contractor prototypes remain available for method development.
   They are not called by a production driver. The legacy G1 cell-level
   path is retired, and the quartet-level replacement fails closed until
   its periodic translation domain, penetration criterion, screened
   kernel, symmetry reconstruction, exchange composition, and gradients
   have equation-level end-to-end validation.

## BIPOLE phase status

| Phase | Status | What |
|---|---|---|
| 1 | landed | Shell-pair Cartesian multipole moments (C++ via libint) |
| 2 | landed | Multipole-multipole Coulomb interaction tensor |
| 3 | landed | IDIPC geometric dispatch per quartet |
| 4a | landed | Cell-level multipole moments from density |
| 4b | landed | Neutral-cell spherical-sample dipole surface diagnostic |
| 4b proper | gated | CRYSTAL EXT EL-POLE shell-multipole/penetration decomposition; the former complete-AO-density reciprocal helper fails closed |
| 5 (Γ-only) | landed (2026-05-18) | Γ-only end-to-end SCF with `use_ewald_j_split=True` |
| 5 + DIIS | landed (2026-05-18 late) | DIIS-compatible via D-consistent energy/error formulation |
| 5 (multi-k) | landed (dense-k sign-off, 2026-05-20) | Per-k `F_J^LR(k)`, MgO/diamond/Si STO-3G within 1 mHa of CRYSTAL14 |
| 5K (RKS) | landed (2026-05-20) | BIPOLE RKS driver with libxc V_xc + hybrid support |
| 5U | landed (2026-05-20) | UHF BIPOLE with MOM/level-shift parity to RHF; historical ODA hook now fail-closed |
| 5UK | landed (2026-05-20) | BIPOLE UKS driver with spin-polarised V_xc |
| 5b | retired | Cell-level multipole far-field research artifact; not driver-reachable |
| 6a | **fail-closed (2026-08-13)** | Quartet-level prototype retained for redesign; explicit driver requests raise before SCF |
| 6b | landed | `compute_ext_el_spheropole` higher-l terms |
| POLIPO | **landed (2026-08-11)** | Native C++ multipole moment engine (McMurchie-Davidson Hermite kernel, OpenMP); 6/6 libint parity tests pass; ~200\u00d7 slower than libint (optimization pending) |
| 7 | roadmap | CRYSTAL14 numerical-parity sign-off on the 15 demo set; BIPOLE parity tests cover LiH/NaCl/MgO/diamond/Si to < 1 mHa |

Phase 1-5 plus the exact Ewald-J split give the production BIPOLE
energy route; it does not call the gated EXT EL-POLE helper or either
multipole far-field prototype. Phase 4b-proper and Phase 6-7 remain
method-development work.

Dense-k STO-3G sign-off results at cutoff 14 bohr:

| Demo | vibe-qc E_total (Ha/cell) | Δ vs CRYSTAL14 |
|---|---:|---:|
| MgO SHRINK 8 | -271.2177748509 | +0.369 mHa |
| diamond SHRINK 8 | -74.8771393842 | -0.145 mHa |
| silicon SHRINK 8 | -571.3214715798 | -0.659 mHa |

## Quick start: SCF with the Ewald J-split

The Ewald-J split is automatic for 3D parity work. Full multi-k meshes
remain fastest to reason about, but IBZ-reduced Monkhorst-Pack inputs
are accepted and expanded internally to the full mesh until true IBZ
orbit expansion is wired.
Pass `use_ewald_j_split=False` only when intentionally reproducing the
legacy direct-only diagnostic branch.

```python
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

system, basis = build_mgo_sto3g()    # 15 demo geometries in crystal_demos/

# Γ-only:
kmesh = monkhorst_pack(system, [1, 1, 1])
# Multi-k:
# kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
# kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
# The second form is expanded internally to the full mesh for Ewald-J.

opts = PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 14.0
opts.lattice_opts.nuclear_cutoff_bohr = 14.0
opts.initial_guess = InitialGuess.SAD
opts.use_diis = True
opts.diis_start_iter = 2
opts.damping = 0.0
opts.max_iter = 15

result = run_pbc_bipole_rhf(
    system, basis, kmesh, opts,
    # use_ewald_j_split=None defaults to the exact Ewald-J-split
    # F^{2e} build for 3D systems.
    ewald_precision=1e-8,
)
print(f"E_total = {result.energy:+.6f} Ha (CRYSTAL: -271.218)")
```

For open-shell systems, use the UHF scaffold:

```python
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

result = run_pbc_bipole_uhf(
    system, basis, kmesh, opts,
    ewald_precision=1e-6,
)
```

Constraints (2026-05-20):

* **True IBZ acceleration is not implemented.** The k-space ρ̂(K)
  formula does not yet expand IBZ symmetry orbits directly. If an
  IBZ-reduced Monkhorst-Pack mesh carries `ir_mapping` metadata, the
  driver expands it to the corresponding full mesh internally.
  Explicit non-uniform custom k-meshes without that metadata are still
  rejected on the Ewald-J path.
* **The public BIPOLE route requires dim = 3.**
  `run_periodic_job(..., jk_method="bipole")` rejects dim < 3 before
  setup. The four direct `run_pbc_bipole_*` APIs retain a low-dimensional
  `DIRECT_TRUNCATED` diagnostic when `use_ewald_j_split=None`; passing
  `True` for 2D / 1D raises.

## Restricted exact bielectronic zone (`exact_zone_bohr`)

On diffuse bases the S(k) fold gate legitimately demands a large
`cutoff_bohr` (LiH/STO-3G needs ~37 bohr for a 1e-4 fold drift), but
the *fold* requirement is one-electron and cheap; historically the
expensive exact erfc J/K traversal grew with it. `exact_zone_bohr`
(all four RHF/RKS/UHF/UKS drivers, opt-in; screened hybrids and
ROHF/ROKS fail closed) bounds the exact-ERI output zone
independently: operator cells beyond the zone carry the reciprocal
`J_LR` channel plus the neutralising background alone -- the Ewald
far-field model of the density. This is the bielectronic-zone /
monoelectronic-zone partition of the CRYSTAL Coulomb machinery
(Dovesi, Pisani, Roetti, Saunders, Phys. Rev. B 28, 5781 (1983),
Eq. (24a); Pisani, Dovesi, Roetti, Lecture Notes in Chemistry 48
(1988), Sec. II.4b/II.4d) expressed in the Ewald-split gauge.

```python
result = run_pbc_bipole_rhf(
    system, basis, kmesh, opts,      # opts.lattice_opts.cutoff_bohr = 20.0
    exact_zone_bohr=12.0,            # exact erfc quartets only to 12 bohr
    ewald_precision=1e-8,
)
```

The omitted short-range tail is a *measured*, per-case quantity --
choose the zone with a small ladder exactly like the SR pad
(`sr_image_extent_bohr`). Measured fixed-density anchors (pad 20,
2026-08-06): MgO/STO-3G tail 2.7e-3 Ha at zone 10 and 3.2e-4 Ha at
zone 12; LiH/STO-3G (most-diffuse exponent 0.048) keeps mHa-scale
tail out to ~14 bohr and needs a correspondingly wider zone. Guard
rails: requires the corrected exchange split and the padded SR path,
must sit strictly below `cutoff_bohr`, is rejected with SYM3b
symmetry reduction / pair-resolved domains, and the analytic-gradient
preview fails closed on it (FD forces remain production). The
resolved zone is recorded on the result (`result.exact_zone_bohr`)
and in the run log.

## Where to look in the code

```
python/vibeqc/pbc_bipole.py
    run_pbc_bipole_rhf: the closed-shell RHF driver.

python/vibeqc/pbc_bipole_uhf.py
    run_pbc_bipole_uhf: the spin-unrestricted UHF scaffold.

python/vibeqc/pbc_bipole_rks.py
    run_pbc_bipole_rks: the closed-shell RKS (DFT) driver.

python/vibeqc/pbc_bipole_uks.py
    run_pbc_bipole_uks: the open-shell UKS (spin-DFT) driver.

python/vibeqc/bipole_multipole.py
    Shell-pair multipole moments (Phase 1) + interaction tensor
    (Phase 2).

python/vibeqc/bipole_dispatch.py
    IDIPC geometric dispatch (Phase 3).

python/vibeqc/bipole_cell_moments.py
    Cell-level multipole moments from a density block (Phase 4a).

python/vibeqc/bipole_ext_el_pole.py
    AO-density Fourier helpers, a reciprocal electronic Ewald diagnostic,
    the spheropole, and a fail-closed EXT EL-POLE compatibility entry point.

python/vibeqc/bipole_fock_multipole.py
    Retired multipole far-field J research artifact (Phase 5b):
    cell-level multipole-multipole interactions for far cell pairs.
    The exact Ewald-J split is the production path.

python/vibeqc/bipole_fock_ewald.py
    Ewald J-split F²e build: short-range erfc + long-range
    reciprocal + background (used by all four drivers).

python/vibeqc/periodic_jk_method.py
    PeriodicJKMethod.BIPOLE enum, selectable via
    run_periodic_job(..., jk_method="bipole").

examples/periodic/input-bipole-*.py
    Example inputs: MgO RHF, MgO RKS PBE, Li UHF, and
    run_periodic_job with jk_method="bipole".
```

## Forces and analytic-gradient preview

Production BIPOLE atomic forces use
``compute_bipole_gradient_fd``.  It finite-differences the same total
energy reported by the SCF driver, so it is the default path used by
geometry optimization and NEB for all four BIPOLE methods.

```python
from vibeqc.bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
    compute_bipole_gradient_uhf,
    compute_bipole_gradient_rks,
    compute_bipole_gradient_uks,
)

# Production force path:
grad = compute_bipole_gradient_fd(system, "sto-3g", kmesh, opts, method="RHF")
print(f"max|grad| = {np.max(np.abs(grad)):.4e} Ha/bohr")
```

The analytic drivers are still a research-preview surface.  The
**corrected (Ewald-exchange-split) gauge, the BIPOLE default, is the
FD-validated analytic route**: RHF/UHF/RKS/UKS at Γ *and* multi-k,
including finite-temperature/fractional (Mermin free-energy) occupations,
all pinned against FD.  Meta-GGA τ-Pulay is landed (the term validated to
1e-9 against an independent reimplementation; SCAN/r2SCAN are
well-behaved, TPSS/M06-L carry a known SCF-eigenvalue residual).  In the
legacy
(``use_exchange_ewald_split=False``) gauge the maintained preview covers
RHF/UHF Γ (general crystals) and maintained RHF/UHF multi-k regressions,
Gamma-local zero-smearing RKS/UKS (LDA/GGA XC Pulay, moving-grid, and
KS-response terms), and RKS/UKS multi-k (diagonal-Z + corrected W + J^LR +
XC Pulay), which **warns** that the full multi-k KS coupled-perturbed
response is not included.  Legacy-gauge RKS/UKS calls with
finite-temperature/fractional occupations raise ``NotImplementedError``.
The sole remaining gated analytic case is the **legacy multi-k KS-CPHF**:
that path uses a diagonal-Z approximation and warns rather than solving the
full coupled-perturbed response, and it is deferred (the corrected gauge
already covers multi-k KS variationally).  Use the FD force path for
production forces in all cases.

``compute_bipole_gradient_fd`` costs about 6N SCFs for N atoms. It
fails fast if any displaced SCF point does not converge, rather than
differentiating a failed iterate; use ``require_converged=False`` only
for diagnostics.

## Structure optimization

Atomic positions can be optimized at a fixed lattice using
:mod:`vibeqc.bipole_optimize`:

```python
from vibeqc.bipole_optimize import relax_atoms

# Atomic positions only (FD gradients by default, L-BFGS-B)
result = relax_atoms(system, "sto-3g", kmesh, method="RHF",
                     max_iter=30, conv_tol_grad=1e-4)

# One-shot via high-level API
from vibeqc.periodic_runner import run_periodic_job
result = run_periodic_job(system, basis, method="RHF", jk_method="bipole",
                          optimize=True, optimize_cell=False)
```

Atomic relaxation preserves the input lattice. The historical
`relax_cell`, `relax_cell_gradient`, and `relax_full` entry points remain
importable but raise `NotImplementedError`: their strain convention and
coupled convergence were not certified on one terminal geometry. The
force-virial helper is available only as a diagnostic and is not the
periodic stress:

```python
from vibeqc.bipole_gradient import compute_stress_tensor
virial = compute_stress_tensor(system, gradient)  # diagnostic 3×3, Ha/bohr³
```

## Dimensionality support

| Dim | Coulomb | Gradient | Optimization | Notes |
|---|---|---|---|---|
| 3D | Ewald J-split (default) or DIRECT_TRUNCATED | FD production; analytic research preview | Fixed-cell atoms | Variable-cell entry points fail closed |
| 2D (surfaces) | DIRECT_TRUNCATED only | Diagnostic coverage only | Fails closed | Public BIPOLE rejects; use SLAB_EWALD_2D or slab GDF |
| 1D (wires) | DIRECT_TRUNCATED only | Diagnostic coverage only | Fails closed | Public BIPOLE rejects; AUTO uses GDF |

For 2D/1D, only the direct BIPOLE drivers automatically fall back to
``DIRECT_TRUNCATED`` when ``use_ewald_j_split=None``. Treat that branch as a
vacuum-padded molecular-limit diagnostic, not a low-dimensional Coulomb
model. Use GDF for wires and SLAB_EWALD_2D (or the bounded closed-shell slab
GDF route) for genuine surfaces.

## Orbital and population sidecars

BIPOLE runs (RHF, RKS, UHF, UKS, restricted-open, Γ-only and
multi-k alike) write both sidecars. What each one means differs,
and the difference matters:

| Sidecar | Option | What BIPOLE writes |
|---|---|---|
| `{out}.population.{txt,json}` | `write_population_file` | A **crystal** population. Mulliken contracts the real-space lattice density and overlap blocks in the same convention as the BIPOLE SCF energy; Löwdin and Mayer contract over the full SCF k-mesh. No Γ restriction. Dipole stays unsupported: the bulk position operator needs a Berry-phase polarization. |
| `{out}.molden` | `write_molden_file` | The **Γ-block orbitals only**, truncated to the home cell. Requires the k-mesh to contain Γ. |

```{admonition} Prefer QVF and vibe-view over Molden for crystals
:class: important

Molden is a molecular format with no lattice vectors, no k-points, and
no complex coefficients, so a periodic `.molden` is necessarily a
narrow projection: the Γ block, evaluated over home-cell basis
functions with no image sum, and therefore clipped wherever an orbital
straddles a cell face. Orbitals at k ≠ 0 are irreducibly complex and
cannot be written at all; vibe-qc refuses rather than emitting a file
a viewer would misread.

Set `output_qvf=True` and open the archive with `vibe-view`. QVF
carries the lattice, the k-point metadata, the complex Bloch
wavefunction, and torus-periodic grids that wrap across cell faces.
See [QVF and vibe-view](../visualization.md) and
[Molden and periodic systems](../tutorial/orbital_visualization.md#molden-and-periodic-systems).
```

BIPOLE density volumes are evaluated from the density returned by the SCF,
including every translated lattice block. For a full uniform
Monkhorst-Pack mesh, vibe-qc performs an exact Born-von-Karman transform of
that returned density and verifies the inverse transform against every
source block before evaluating Bloch AOs. Alpha and beta densities are
verified separately. The archive receives `volume.density` only when the
same-cell lattice trace, periodic AO-image convergence, and the numerical
primitive-cell electron count all agree. If k-point provenance is incomplete,
the returned blocks cannot be refolded exactly, or the requested real-space
grid is too coarse, the calculation remains successful and QVF is written
without a density volume, together with a nonfatal output diagnostic. This
boundary never regenerates density from orbitals or a surrogate Hamiltonian.

All three options are tri-state. `None` (the default) emits the
sidecar when the route can produce it truthfully; `True` is a
guarantee and raises before the SCF starts if the route cannot;
`False` opts out. A shifted Monkhorst-Pack mesh has no Γ point, so
`write_molden_file=True` raises there; pass `None` to let the run
proceed without the orbital file.

## References

- **Pisani, Dovesi, Roetti 1988**, *Hartree-Fock Ab Initio Treatment
  of Crystalline Systems*, Lecture Notes in Chemistry 48, Springer.
  [doi:10.1007/978-3-642-93385-1](https://doi.org/10.1007/978-3-642-93385-1).
  Chapter II.4c defines the periodic bipolar quartet expansion and its
  common lattice-translation sum.
- **Saunders et al. 1992**, "On the electrostatic potential in
  crystalline systems where the charge density is expanded in Gaussian
  functions," *Molecular Physics* **77**, 629.
  [doi:10.1080/00268979200102671](https://doi.org/10.1080/00268979200102671).
  Defines the Gaussian-tail and penetration quantities used by the
  electrostatic-potential construction.
- **CRYSTAL14**, R. Dovesi, V. R. Saunders, C. Roetti, R.
  Orlando, C. M. Zicovich-Wilson, F. Pascale, B. Civalleri, K.
  Doll, N. M. Harrison, I. J. Bush, P. D'Arco, M. Llunell, M.
  Causà, Y. Noël, "CRYSTAL14: A program for the *ab initio*
  investigation of crystalline solids," *Int. J. Quantum Chem.*
  **114**, 1287 (2014).
  [doi:10.1002/qua.24658](https://doi.org/10.1002/qua.24658). The
  modern reference for the EXT EL-POLE / EXT EL-SPHEROPOLE
  conventions vibe-qc is targeting.
- **CRYSTAL23**, A. Erba et al., "CRYSTAL23: a program for
  computational solid state physics and chemistry," *J. Chem.
  Theory Comput.* **19**, 6891 (2023).
  [doi:10.1021/acs.jctc.2c00958](https://doi.org/10.1021/acs.jctc.2c00958).

BIPOLE runs through `run_periodic_job(..., jk_method="bipole")`
now route the methodology references through the citation database
([user_guide/citations](citations.md)) via `routes.methods["bipole"]`.
The 1988 monograph, Saunders 1992, and Dovesi 2014 fire automatically
on every BIPOLE run.

## See also

- [Periodic-SCF methods (comparative tour)](periodic_methods.md):
  side-by-side comparison of BIPOLE vs GDF vs GPW/GAPW, plus
  the surface-reactions workflow walkthrough.
- [Multi-k periodic SCF](multi_k_scf.md), the production multi-k
  RHF/RKS path today.
- [Density fitting](density_fitting.md), the production Γ-only
  hybrid-DFT path via native GDF.
- [Ewald summation](ewald.md), the long-range Coulomb machinery
  used by the legacy `run_*_periodic_*_ewald3d` drivers.

## Quartet-level bipolar far-field

The quartet-level implementation is currently unavailable from all four
drivers. The supported setting is the default:

```python
result = run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="bipole",
    use_multipole_far_field=False,
)
```

Passing `True` raises `NotImplementedError`. A future implementation must
first show that its skipped exact quartets and multipole add-back cover the
same three-translation periodic domain in Eq. II.4.10 of the 1988 monograph.
It must also validate independent Coulomb and exchange penetration criteria,
screened interactions, nontrivial symmetry operations, and energy-gradient
consistency. The low-level prototype modules are not a supported numerical
API and their isolated unit tests do not certify an SCF route.
