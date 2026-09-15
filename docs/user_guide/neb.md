(neb)=
# Nudged Elastic Band (NEB)

vibe-qc is growing a native Nudged Elastic Band driver for finding
minimum-energy paths between reactant and product geometries. It is
the right transition-state finder for systems where the reactive
event is delocalised, most notably for surface catalysis (the
flagship target is N₂ dissociation on Fe(100)), where a cluster
model is a poor approximation of the actual catalytic surface and
single-point TS searches struggle.

## Status

NEB is landing across five increments on `main`. This page tracks
what is shippable today; the rest is signposted as roadmap.

| Increment | Status | Public API surface |
|-----------|--------|--------------------|
| 1. Interpolators + dataclasses | **Available** | `interpolate_linear`, `interpolate_idpp`, `NEBImage`, `NEBPath` |
| 2. NEB driver (improved tangent + spring) | **Available** | `run_neb(...)`, `NEBResult` |
| 3. Climbing image (CI-NEB) | **Available** | `run_neb(..., climbing_image=True)` |
| 4. Periodic dispatch (BIPOLE + FD gradient) | **Available** | `run_neb(periodic_r, periodic_p, ..., kpoints=...)` |
| 5. QVF `reaction.path` emitter | **Available** | `result.write_qvf(stem)` |
| + Density warm-start | **Available** | `run_neb(..., warm_start=True)` |
| + DFT+U (Dudarev) | **Available** (molecular + periodic UHF/UKS) | `run_neb(..., dft_plus_u=[HubbardSite(...)])` |
| + MACE backend (ML potential) | **Available** (molecular + periodic; needs `[mace]`) | `run_neb(..., method="mace", mlip_options=MLIPOptions(...))` |
| + MSINDO backend (semiempirical INDO) | **Available** (molecular) | `run_neb(..., method="msindo")` |
| + Unified molecular semiempirical backends | **Available** | `run_neb(..., method="dftb0" / "scc-dftb" / "gfn2-xtb" / "pm6" / "upm6" / "om1" / "om2" / "om3")` |
| + Unified periodic semiempirical backends | **Available** (Gamma DFTB0/SCC-DFTB/GFN2-xTB/PM6; full-k DFTB0/SCC-DFTB) | Bloch-periodic OMx and all PM7 routes fail closed until their published Hamiltonians are implemented. |
| + MSINDO-SECCM backend | **Available** (closed-shell, explicit cyclic cell) | `run_neb(..., method="seccm", ccm_options=...)` |
| + DFTB0-SECCM backend | **Available** (gated 1-D H/C, explicit frozen topology) | `run_neb(..., method="dftb0", seccm_topology=...)` |

## Increment 1, path-construction primitives (available today)

Two interpolators construct an initial chain of images between a
reactant and a product geometry. They do *not* run any SCF, they
operate purely on atomic positions.

### Linear Cartesian

```python
from vibeqc import Atom, Molecule, interpolate_linear

reactant = Molecule([Atom(1, [0.0, 0.0, 0.0]),
                     Atom(1, [0.0, 0.0, 1.4])])
product  = Molecule([Atom(1, [0.0, 0.0, 0.0]),
                     Atom(1, [0.0, 0.0, 4.0])])

path = interpolate_linear(reactant, product, n_images=5)
#   len(path) == 7   (5 intermediate + the two endpoints)
```

Endpoints are returned as the original objects (no copy), so
`path[0] is reactant` and `path[-1] is product`.

### IDPP (Image-Dependent Pair Potential)

Linear interpolation between two bonded structures often places
atoms inside each other in the intermediate images. IDPP
(Smidstrup, Pedersen, Stokbro, Jónsson 2014) constructs a smooth
chain by minimising

```
S^(i)(R) = sum_{j<k}  (d^(i)_{jk} - r_{jk}(R))^2 / r_{jk}(R)^4
```

for each intermediate image `i`, where `d^(i)_{jk}` is the linear
interpolation between reactant and product pair-distance matrices.
The `1/r⁴` weighting drives images away from atom-atom clashes
while still tracking the interpolated distance manifold.

```python
from vibeqc import interpolate_idpp

path = interpolate_idpp(reactant, product, n_images=7)
```

IDPP is the recommended default starting path for NEB whenever
reactant and product differ in bonding connectivity.

### Dataclasses

```python
from vibeqc import NEBImage, NEBPath

images = [NEBImage(system=s) for s in path]
neb = NEBPath(images=images, spring_constant=0.1)
neb.n_images        # 7 (intermediate + endpoints)
neb.n_intermediate  # 5
neb.energies()      # np.ndarray of NaN (energies set by the driver)
```

`NEBImage` carries `(system, energy, gradient, tangent)`, the
energy/gradient/tangent slots stay `None` after interpolation; they
are populated by the NEB driver (Increment 2+).

### Periodic systems

Both interpolators accept `PeriodicSystem`. The lattice must match
between reactant and product (variable-cell NEB is out of scope,
fix the cell to the reactant's). Periodic IDPP uses minimum-image
pair distances, and the NEB driver's tangent and spring use
minimum-image inter-image displacements, so a reactant and product
that differ by an atom hopping across the PBC, the surface
self-diffusion case, interpolate and relax along the short,
through-the-boundary path instead of being dragged across the cell.
The single-round minimum image is exact for orthorhombic cells and
the standard close approximation for mildly skewed ones.
`interpolate_linear` is a plain Cartesian straight line (no
wrapping); use IDPP for cross-boundary hops.

## Increment 2, NEB driver (available today, molecular only)

`run_neb` runs an improved-tangent NEB end to end: it computes
per-image SCFs (in parallel via joblib), assembles the NEB force
(tangent + spring + perpendicular-true-force projection), and
optimises with a damped quick-min outer loop (the same scheme ASE's
MDMin uses).

```python
from vibeqc import Atom, Molecule, run_neb

# H + H₂ → H₂ + H, collinear.
reactant = Molecule(
    [Atom(1, [0.0, 0.0, 0.0]),
     Atom(1, [0.0, 0.0, 1.4]),
     Atom(1, [0.0, 0.0, 4.4])],
    0, 2,  # neutral doublet
)
product = Molecule(
    [Atom(1, [0.0, 0.0, 0.0]),
     Atom(1, [0.0, 0.0, 3.0]),
     Atom(1, [0.0, 0.0, 4.4])],
    0, 2,
)
result = run_neb(
    reactant, product,
    basis="sto-3g",
    n_images=5,
    method="UHF",
    spring_constant=0.1,
    interpolation="idpp",
    max_iter=80,
    conv_tol_force=2e-3,
)
print(result.converged, result.n_iter, result.max_force)
print("TS energy:", result.energies[result.transition_state_index])
ts = result.path.images[result.transition_state_index].system
```

`NEBResult` carries the final path, the per-image energies, a
boolean `converged`, the index of the highest-energy intermediate
image (`transition_state_index`), the number of outer iterations,
and the final maximum NEB force magnitude.

### Parallelism

Per-image SCFs run inside a `joblib.Parallel` block per outer
iteration. `n_jobs=0` (default) uses a bounded automatic worker count
(serial for periodic NEB, up to a small cap for molecular NEB); pass
`n_jobs=1` for serial or `n_jobs=-1` to explicitly use every available
core. The endpoints are evaluated once and cached - they don't move
while `free_endpoints=False` (the default).

### Frozen atoms

Pass `freeze_indices=[i, j, …]` to zero the NEB force on the
listed atoms; their geometry stays at the endpoint position
throughout the optimisation. This is implemented at the NEB layer
(the per-image SCF still sees the full atom set, we just zero the
force components before the quick-min step). Useful for keeping
slab-substrate atoms fixed during surface NEB once Increment 4
lands.

### Method support

`method="RHF" | "UHF" | "ROHF" | "RKS" | "UKS" | "ROKS"`, matching the
molecular mean-field set in `optimize_molecule`. ROHF follows the validated
analytic ROHF gradient; ROKS uses central differences of the ROKS energy
(`fd_step_bohr=1e-3` by default) until its analytic XC gradient is available.
Both preserve one restricted spatial-orbital set and exact spin-pure metadata.
Periodic ROHF/ROKS NEB and DFT+U on these two routes remain fail-closed. Pass
the matching `*_options` (for example,
`uhf_options=UHFOptions()`) for SCF controls that the image evaluator
supports, such as `max_iter` and convergence thresholds. The Gaussian NEB
image loop is deliberately narrower than the corresponding single-point or
geometry-optimization driver.

**Molecular Gaussian validity envelope.** Every molecular Gaussian NEB is
all-electron: named or explicit ECP requests fail before image evaluation.
The RHF/UHF/RKS/UKS warm-start loop also rejects `density_fit=True` and
`cosx=True`; all molecular Gaussian paths reject multi-guess basin selection,
and UHF/UKS reject `SPIN_SCHEDULE`. Double-hybrid KS paths fail because the
image objective omits their MP2 term, while molecular RKS/UKS VV10 paths fail
because their analytic gradient omits the nonlocal derivative. Path-backed
READ is supported for RHF/UHF/RKS/UKS through `options.read_path` (or complete
preloaded density matrices), but `run_neb` has no `read_from=` keyword.
FRAGMO likewise has no `fragments=` seam here and requires complete
precomputed density matrices. These checks also run for `dry_run=True`.

UKS image stability is check-only: the lowest Hessian eigenvalue and verdict
are retained, but `stability_max_retries` is internally set to zero on a copy
of the image options. Independently following each image into its lowest local
basin could break electronic-state continuity along the path, for the same
reason multi-guess basin selection is rejected. If the check finds an unstable
image, or if its stability eigensolver does not converge, NEB aborts with the
image index and diagnostic instead of silently using its energy and gradient.
The caller's options object is not changed.

### Algorithmic notes

* **Improved tangent.** The tangent at each intermediate image is
  determined by the energy ordering of its neighbours (eqs. 8-11
  of Henkelman & Jónsson 2000). When the central image is the
  local energy extremum, the tangent is a weighted combination of
  the forward and backward differences with the magnitude of the
  larger energy difference upfront.
* **Outer loop.** Quick-min (damped MD) with adaptive step. Each
  intermediate image carries a velocity; per step the velocity is
  projected onto the NEB force and zeroed if the projection is
  negative. Step size grows by 1.1 when aligned and resets on
  direction flips. Equivalent in spirit to ASE's `MDMin`. L-BFGS-B
  on the concatenated coordinate would also work but assumes the
  NEB force is conservative, it isn't (the spring contribution
  depends only on the parallel projection of the displacement),
  so quick-min is the standard choice.
* **Slow convergence tail.** Near convergence the geometry can be
  essentially locked in (TS energy stable to mHa) while the
  spring-component max-force decays slowly. This is intrinsic to
  the quick-min / MDMin scheme; for tighter convergence increase
  `max_iter`, or pre-relax the band with a coarser
  `conv_tol_force` and finish the saddle with a quasi-Newton TS
  search.

## Increment 3, climbing image (available today)

Pass `climbing_image=True` to `run_neb` to promote the highest-
energy intermediate image to a "climbing" image partway through
the optimisation:

```python
result = vibeqc.run_neb(
    reactant, product, basis="sto-3g",
    n_images=5, method="UHF",
    climbing_image=True,
    climbing_image_start_fraction=0.3,  # default; warm-up fraction
    conv_tol_force=1e-3, max_iter=80,
)
ts_image = result.path.images[result.path.climbing_image_index]
```

The CI image's spring contribution is removed and the tangent-
parallel component of its true force is inverted, so it climbs
uphill *along* the path to the saddle while still relaxing
perpendicular. Other images keep standard NEB dynamics, which
keeps the path itself anchored at the right shape. Henkelman,
Uberuaga, Jónsson 2000 is the original reference.

* **Warm-up.** The highest-energy intermediate image is selected
  *after* `climbing_image_start_fraction * max_iter` iterations
  of plain NEB and then fixed for the remainder of the run. The
  warm-up keeps the climbing selection from flipping between
  iterations before the band has found its rough shape; the
  selection lock keeps the climber from losing momentum to a
  later re-selection. Default fraction is 0.3.
* **Accuracy.** On the textbook H + H₂ → H₂ + H benchmark the
  climbing image lands at the symmetric saddle to ≤ 1e-4 bohr,
  six orders of magnitude tighter than the plain-NEB highest-
  energy image (which Inc 2's test bounds at 0.05 bohr).
* **`NEBResult.path.climbing_image_index`** records which path
  image was promoted; `transition_state_index` then equals that
  index. Both are `None` on a plain (non-climbing) run.

The climbing image is the recommended saddle estimator from
`run_neb`: it's accurate enough to seed a follow-up quasi-Newton
TS search (or in many cases to just report directly as the
saddle).

## Increment 4, periodic dispatch (available today, BIPOLE + FD gradient)

`run_neb` accepts `PeriodicSystem` endpoints. The path is laid
out exactly as in the molecular case (improved-tangent forces,
quick-min outer loop, optional climbing image); per-image SCFs
go through `run_pbc_bipole_rhf/uhf/rks/uks` and per-image
gradients through a central-difference fallback (the J^LR
reciprocal-Ewald contribution is still missing from the analytic
BIPOLE gradient; the current periodic path therefore keeps the
finite-difference fallback).

```python
import numpy as np
from vibeqc import (
    Atom, PeriodicSystem, run_neb,
    PeriodicRHFOptions, LatticeSumOptions,
)

L = np.diag([10.0, 10.0, 10.0])
reactant = PeriodicSystem(3, L, [
    Atom(1, [0.0, 0.0, 0.0]),
    Atom(1, [0.0, 0.0, 1.4]),
])
product = PeriodicSystem(3, L, [
    Atom(1, [0.0, 0.0, 0.0]),
    Atom(1, [0.0, 0.0, 1.8]),
])

# Periodic SCF options. Use the same options object the
# non-NEB BIPOLE drivers take.
rhf_opts = PeriodicRHFOptions()
rhf_opts.max_iter = 50
result = run_neb(
    reactant, product,
    basis="sto-3g",
    n_images=5,
    method="RHF",
    rhf_options=rhf_opts,
    kpoints=(2, 2, 2),       # Monkhorst-Pack mesh; passed to monkhorst_pack
    fd_step_bohr=1e-3,       # central-difference half-step
    interpolation="idpp",
    max_iter=20,
    conv_tol_force=2e-3,
)
ts = result.path.images[result.transition_state_index].system
```

**The FD-gradient surface**. Per-image gradient cost is 6N + 1
BIPOLE SCFs (one per Cartesian degree of freedom, ± displaced,
plus the reference). For N = 10 atoms × 5 intermediate images × 50
outer iterations that's ~15 000 BIPOLE SCFs total, slow, but
correct in the limit `fd_step_bohr → 0`. When the J^LR derivative
lands in the analytic BIPOLE gradient, the per-image cost drops to
2 SCFs (one for the energy + one for the analytic gradient).

**`kpoints`** accepts either a 3-tuple of ints (converted internally
via `vibeqc.monkhorst_pack`), a pre-built `BlochKMesh`, or a
`KPoints` object. `None` ⇒ Γ-only mesh (only useful as a
sanity-check; pick a real mesh for production).

**BIPOLE image padding** uses `sr_image_precision=1e-6` by default, the same
production padded ket-image contract as the direct BIPOLE drivers. The value
is forwarded to the reference and every finite-difference SCF. Set
`sr_image_precision=None` only for a historical-domain diagnostic or a
bounded-cost dispatch test whose numerical accuracy is not being tested. It
does not cap the production image domain, and results from that opt-out should
not be used as production energies or forces.

**Same lattice for both endpoints**. Variable-cell NEB is out of
scope, fix the cell to the reactant's lattice. Endpoint lattices
that don't match raise.

**Periodic Gaussian validity envelope.** The public BIPOLE NEB route is
3D-only and all-electron. Before dry-run or image evaluation it rejects
`dim=1`/`dim=2`, POB basis names, explicit ECP metadata, element/basis pairs
that require an ECP, and periodic Gaussian `dispersion_params`. It also
requires zero-temperature occupations: positive `smearing_temperature` and
`KPoints.smearing` / `KPoints.bz_integration` metadata fail closed because the
NEB result does not yet distinguish a Mermin free-energy objective. Periodic
Gaussian `InitialGuess.READ` and `FRAGMO` also fail closed: the image and
displaced-geometry loop does not yet project a complete lattice restart or
fragment density onto every geometry. Closed-shell RHF/RKS additionally reject
`PATOM` because their periodic driver has no in-field PATOM seed. Run the
requested guess as a supported single point first, then begin the band from SAD
or HCORE; UHF/UKS may also use PATOM or ATOMSPIN. Use a
BIPOLE-supported semilocal, global-hybrid, or validated screened-hybrid
functional and a basis with a supported all-electron record for every
element. Long-range-corrected range-separated hybrids, VV10/nonlocal
correlation, and double hybrids are outside this envelope. Dispersion support
documented for molecular or semiempirical NEB does not extend to periodic
Gaussian NEB.

**Frozen substrate atoms**. The same `freeze_indices` mechanism
works for surface NEB: pass the indices of the substrate atoms
you want to keep fixed. The frozen-atom forces are zeroed at the
NEB layer; the BIPOLE SCF still sees the full system.

**Mixing molecular + periodic endpoints raises** at the
type-dispatch guard inside `run_neb`. The endpoints must be the
same type.

## Increment 5, QVF reaction.path emitter (available today)

`NEBResult` carries a `write_qvf(stem)` method that emits a
vibe-view-renderable archive of the converged (or last-evaluated)
path:

```python
result = vibeqc.run_neb(reactant, product, basis="sto-3g", ...)
qvf_path = result.write_qvf("my_neb_run")
#   → Path('my_neb_run.qvf')
```

The archive contains three sections:

* **`structure`**, the reactant geometry (single-frame fallback
  for viewers that don't yet animate `reaction.path`).
* **`reaction.path`**, per-image coords + energies + reaction
  coordinate (cumulative arc length normalised to 0-1) + waypoint
  annotations (`reactant` / `product` / `transition_state`). The
  TS waypoint points at the climbing image when CI-NEB ran, else
  at the highest-energy intermediate.
* **`citations`**, BibTeX assembled with `uses_neb=True` (plus
  `uses_ci_neb=True` when the run was climbing-image). Always
  includes Henkelman+Jónsson 2000 (improved tangent) + Smidstrup
  2014 (IDPP); CI-NEB adds Henkelman+Uberuaga+Jónsson 2000.

For periodic NEB the archive ships as **QVF v2** automatically:
the writer detects periodic frames and emits the per-frame lattice
+ dim on the `reaction.path` section so vibe-view can draw the
unit cell. See the
"[Periodic reaction paths (QVF v2)](vibe_view.md)" subsection of
the vibe-view guide for the schema details. Inc B (vibe-view
renderer track) will add the cell-rendering + atom-wrapping logic
to the renderer itself.

## Density warm-start (available today)

`run_neb` reuses each image's converged SCF density across outer
iterations: the density at outer iter N is fed in as the SCF
initial guess for the same image at outer iter N+1. The geometry
change between iterations is typically small, so the SCF
converges in many fewer iterations than from a cold SAD/Hcore
guess.

```python
result = vq.run_neb(
    reactant, product, basis="def2-svp",
    method="UKS", functional="pbe",
    n_images=7,
    warm_start=True,   # default; pass False to force cold SCFs (benchmarking)
)
```

Warm-start changes only the SCF initial guess, but that does not make a
finite calculation bit-exact to a cold start. In a unique, well-separated SCF
basin the two modes should converge to the same physical state within the
requested thresholds. Roundoff, different inner-SCF iteration histories, and
symmetry-equivalent broken-symmetry solutions can produce small numerical
differences; a genuinely multi-basin problem can converge to a different
electronic state. At a fixed outer `max_iter` budget those differences can
propagate into forces and positions. Treat a cold-start comparison as a useful
basin audit for sensitive barriers, not as an identity guaranteed by the API.

### Coverage today

| Method   | Molecular | Periodic (BIPOLE) |
|----------|-----------|-------------------|
| RHF      | yes       | yes               |
| UHF      | yes       | yes               |
| RKS      | yes       | yes               |
| UKS      | yes       | yes               |

Periodic RKS / UKS warm-start was unblocked when the upstream
`build_xc_periodic` arg-order bug in
`run_pbc_bipole_rks` / `run_pbc_bipole_uks` got fixed alongside
Increment 4d-bipole UKS DFT+U. Same dispatch as RHF / UHF,
`_evaluate_image_periodic` hands the cached density (or
``(α, β)`` blocks for open-shell) into the BIPOLE driver's
warm-start kwarg per outer iteration.

### Observed speedups

* **Molecular UKS** on the H + H₂ → H₂ + H benchmark
  (3 intermediate images, 10 outer iters): ~2× wall-time
  reduction (~84 s → ~42 s). Open-shell DFT SCFs are the
  biggest win, expensive per iteration AND slow to converge
  from SAD/Hcore on a doublet manifold.
* **Periodic UHF** on H₃-in-cubic-box (single SCF call):
  76 SCF iters → 25 SCF iters, **~3× speedup**.
* **Molecular RHF / RKS / periodic RHF**: speedups in the
  flat-to-modest range on small systems, the cold SCF
  already converges in 3-5 iters at the test scale. Real
  surface NEB workloads should see 2-4× the maintainer
  originally scoped for the warm-start milestone.

### Bonus: FD-gradient inner warm-start (periodic only)

Periodic NEB computes per-image gradients by central
differences (6N + 1 SCFs per image per outer iteration; see
Increment 4). Each of those 6N displaced SCFs additionally
warm-starts from the reference SCF's converged density at the
same image, geometrically small perturbations
(default `fd_step_bohr=1e-3`) are essentially the same
electronic structure, so the displaced SCFs converge in a
couple of iterations instead of from cold. This stacking
multiplies the warm-start savings for periodic NEB
specifically.

### When to turn warm-start off

* **Benchmarking** the cold-start cost (`warm_start=False`).
* **Basin validation** for open-shell, symmetry-broken, near-degenerate, or
  otherwise metastable paths. Compare independently converged warm and cold
  bands, including state diagnostics, before trusting a sensitive barrier.
* **Bisection** when an unrelated convergence problem is suspected and you
  want to remove warm-start as a variable.

Warm-start remains the production default because it normally saves many SCF
iterations. It is an acceleration strategy, not a guarantee that two initial
guesses select the same SCF basin.

## DFT+U (Hubbard correction; available today)

Pass `dft_plus_u=[HubbardSite(...)]` to `run_neb` to add the
Dudarev rotationally-invariant per-spin V_U potential to every
per-image SCF:

```python
import vibeqc as vq

result = vq.run_neb(
    reactant, product,
    basis="def2-svp",
    method="UKS", functional="pbe",
    n_images=7,
    dft_plus_u=[
        # U_eff = 4 eV on the Fe 3d channel (atom index 0 in the
        # cluster ordering).
        vq.HubbardSite(atom_index=0, l=2, U_ev=4.0),
    ],
)
```

The kwarg accepts the same `HubbardSite` dataclass as
`vibeqc.run_rhf` / `vibeqc.run_rks` / `vibeqc.run_uhf` /
`vibeqc.run_uks`, eV inputs converted internally to Hartree,
``(atom_index, l)`` → AO-group lookup done once per NEB call
(the AO grouping is geometry-invariant for a fixed basis +
atom ordering).

The Dudarev energy ``E_U = 2 Σ_A (U_eff/2) (tr n − tr n²)``
folds into each image's ``result.energy``; the corresponding
``e_dft_plus_u`` field of the per-image SCF result is the V_U
contribution alone. Combines cleanly with warm-start, the
converged density already encodes V_U, so feeding it back as
the next outer iter's initial guess works the same way.

**Coverage today**:

| Method | Molecular | Periodic |
|--------|-----------|----------|
| RHF    | yes       | yes      |
| UHF    | yes       | yes      |
| RKS    | yes       | yes      |
| UKS    | yes       | yes      |

All four periodic methods route ``dft_plus_u=`` through to the
matching BIPOLE driver via ``run_pbc_bipole_{rhf,uhf,rks,uks}``;
the per-spin Dudarev V_U Fock contribution lands in every
per-image BIPOLE SCF, and the FD-gradient finite differences
pick up ``∂E_U/∂R`` correctly because the displaced SCFs carry
the same ``dft_plus_u=`` list. Closed-shell BIPOLE +U landed in
v0.9.0; the open-shell variants landed alongside Increment
4d-bipole.

**Small-cell pathology warning.** Periodic +U on diffuse AO
channels in tight boxes (e.g. H 1s in a 4-bohr cubic cell) can
exhibit an unphysically large +U Fock contribution: heavy
periodic AO image overlap pumps the AO projector and the
``e_dft_plus_u`` contribution swings by orders of magnitude
relative to the molecular reference. The observed pathology on
the v0.9.0 NEB test fixture (H₂⁺ + U=4 eV on 1s): ΔE_TS ≈ -224
eV at L=4 bohr, vs. the physically-correct +0.27 eV at L≥8
bohr. Until the periodic +U projector is hardened against
small-cell image overlap, periodic +U recipes should stay at
cell sizes ≥ 2 × max AO decay length (~8 bohr for H 1s; much
larger for transition-metal d-channels).

**Citations**: when `dft_plus_u` is non-empty the per-image
SCFs route the Dudarev 1998 + Cococcioni-Gironcoli 2005
papers into the BibTeX assembled by `NEBResult.write_qvf`
(same `routes.methods.dft_plus_u` route the molecular SCF
runners use).

**Invalid Hubbard sites raise at the NEB boundary**: a
`HubbardSite(atom_index, l, ...)` pointing at a channel
absent from the basis raises `ValueError` before any SCF
runs, useful when the user picks the wrong `l` for a
minimal basis (e.g. `l=1` on STO-3G H, which only has an
s-shell).

## MACE backend, ML-potential reaction paths (available today)

Pass `method="mace"` to drive the band with a pre-trained
[MACE](https://github.com/ACEsuit/mace) machine-learned interatomic
potential instead of per-image SCFs. MACE returns **analytic energy
and forces** from a single forward pass, no SCF, no Gaussian basis,
no k-mesh, so a MACE-NEB is dramatically cheaper than the SCF path,
and for **periodic** bands it bypasses the 6N+1 finite-difference
gradient (Increment 4) entirely.

```python
import vibeqc as vq
from vibeqc.mlip import MLIPOptions

# Molecular: an organic reaction path with the MACE-OFF23 model.
result = vq.run_neb(
    reactant, product,            # Molecule or PeriodicSystem endpoints
    method="mace",                # no basis / functional needed
    n_images=7,
    climbing_image=True,
    mlip_options=MLIPOptions(model="medium-mpa-0"),  # default: MIT MACE-MPA-0
    conv_tol_force=5e-4,
)
ts = result.path.images[result.transition_state_index].system
```

The improved-tangent force, climbing image, frozen atoms, and the
[minimum-image handling](#periodic-systems) for cross-boundary periodic
hops all work exactly as for the SCF methods, only the per-image
energy/force evaluation changes.

**How it runs.** The MACE model (torch weights) is loaded **once** and
its ASE calculator reused for every image and outer iteration;
constructing a fresh model per evaluation would reload the weights each
call. Because the loaded calculator is a live torch object, MACE-NEB
evaluates the band **serially** (`n_jobs` is forced to 1) rather than
pickling the model across worker processes, each evaluation is one
forward pass, so serial is cheap.

**Energy scale.** MACE energies are on a model-specific reference scale
(each model subtracts its own per-element atomic energies) and are
**not** comparable across models or to a vibe-qc total energy. They are
meaningful for *relative* energetics, exactly what a reaction barrier
is, so the MEP and barrier are well-defined, but don't compare a MACE
absolute energy to an SCF one.

**Model selection + licensing.** `MLIPOptions(model=...)` picks the
foundation model (default `"medium-mpa-0"`, MIT, materials). The
organic `off23-*` models are under the Academic Software License
(academic, non-commercial) and are **gated**: selecting one raises
`PermissionError` unless you acknowledge the licence
(`MLIPOptions(accept_academic_license=True)` or `VIBEQC_ACCEPT_ASL=1`).
See [the MLIP guide](mlip.md) / `vibeqc.mlip.mace` for the model
registry.

**Requirements.** MACE is the optional `[mace]` extra (PyTorch + e3nn);
install with `pip install 'vibe-qc[mace]'`. It currently requires
**Python ≤ 3.13** (its `matscipy` dependency ships no 3.14 wheel). A
`run_neb(method="mace")` on an interpreter without the extra raises a
clear, actionable `ImportError`.

**Citations.** A MACE-NEB run cites the MACE *method* paper (Batatia
2022) plus the *foundation-model* paper for the model actually used
(MACE-MP/MPA → Batatia 2024; MACE-OFF23 → Kovács 2023), alongside the
NEB papers, and **not** the Gaussian-integral library (no integrals
are evaluated). All of this lands in `result.write_qvf(...)`'s
references block automatically.

## MSINDO backend, semiempirical reaction paths (available today)

Pass `method="msindo"` to drive the band with
[MSINDO](msindo.md), vibe-qc's own Bredow/Geudtner/Jug INDO engine
over Slater orbitals, instead of an SCF over a Gaussian basis. Like
MACE it needs **no basis, functional, or k-mesh**; unlike MACE it is
vibe-qc's own quantum-chemistry method (not an external model), so its
energies *are* MSINDO total electronic energies and are directly
comparable run-to-run. MSINDO-NEB is **molecular only** (periodic MSINDO
is the Cyclic Cluster Model, a separate API, see [the MSINDO
guide](msindo.md)).

```python
import vibeqc as vq

# Ammonia umbrella inversion: pyramidal NH3 flips through the planar
# D3h transition state. Endpoints carry the atomic numbers / charge /
# multiplicity; no basis or functional is passed.
result = vq.run_neb(
    reactant, product,            # Molecule endpoints (H-Br supported)
    method="msindo",
    n_images=5,
    climbing_image=True,
    conv_tol_force=1e-3,
)
ts = result.path.images[result.transition_state_index].system
```

The improved-tangent force, climbing image, frozen atoms, warm-start
flag (a no-op here, MSINDO carries no SCF state across geometries), and
the spring machinery are all shared with the SCF path; only the
per-image energy/gradient evaluation changes.

**Gradient + cost.** The default `method="msindo"` INDO path uses the
finite-difference nuclear gradient (`msindo_gradient_fd`, central-differenced
from the total energy and oracle-validated against MSINDO's analytic `CARTOPT
ANALY` gradient). Each INDO image therefore costs 6N `run_msindo` SCFs per
outer iteration. The FD half-step is `fd_step_bohr` (default 1e-3 bohr,
converted to the engine's Angstrom). `method="nddo"` selects the distinct
closed-shell NDDO Hamiltonian and its analytic gradient for H, Li-F, and Na-Cl,
including the source SPDD terms for Al-Cl; it never falls through to the INDO
FD evaluator. Open-shell NDDO gradients fail closed. Because both engines
are stateless, the band evaluates in parallel across images (`n_jobs`
honoured, default bounded auto).

**Open shell + dispersion.** Open-shell radicals route to MSINDO's UHF
automatically (set the endpoint `multiplicity`); the s/p elements are
covered, open-shell d raises. Pass `dispersion_params=D3BJParams(...)`
to fold a D3-BJ correction (energy **and** gradient) into every image.

**Citations.** An MSINDO-NEB run cites the MSINDO method papers (Ahlswede &
Jug 1999, Parts I + II) and the NEB papers, plus the Pulay DIIS paper
(MSINDO's SCF accelerator), and **not** the Gaussian-integral library (MSINDO
evaluates no Gaussian integrals) or any XC functional. NDDO additionally cites
Voigt's coordinate-invariant INDO/NDDO multipole decomposition and Dewar and
Thiel's two-centre multipole model, and records `MSINDO-NDDO` as the QVF method.
All of this lands in `result.write_qvf(...)`'s references block automatically.

A complete, runnable example, endpoint relaxation, a converged
climbing-image band, and a Hessian check confirming the saddle has a
single imaginary mode, is at
[`examples/semiempirical/22_msindo_neb.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/semiempirical/22_msindo_neb.py).

## Unified semiempirical reaction paths

`run_neb` accepts every semiempirical method whose requested boundary, spin,
energy, and gradient combination is already executable. Method aliases are
resolved by `SemiempiricalRoutePlan` before parameters or a heavy kernel are
loaded.

| Boundary | Methods | Gradient used by NEB |
| --- | --- | --- |
| Molecular | DFTB0, SCC-DFTB | Existing restricted or unrestricted analytic gradient |
| Molecular | GFN2-xTB | Existing closed-shell analytic gradient; open shell remains gated |
| Molecular | PM6/UPM6, OM1/OM2/OM3 | Existing native finite-difference batch |
| Molecular | MSINDO | Existing molecular NEB finite difference, including supported open shells |
| Periodic Gamma | DFTB0/UDFTB0 | Native analytic gradient at the validated 15 bohr image cutoff |
| Periodic Gamma | SCC-DFTB/USCC-DFTB, GFN2-xTB | Central difference of each converged total energy |
| Periodic Gamma | PM6 | Existing native finite-difference batch; closed shell only. Bloch-periodic OMx is gated. |
| Periodic full-k | DFTB0, SCC-DFTB | Zero-temperature native batched finite difference over the Bloch kernel; closed shell only |
| SECCM | Current MSINDO adapter | Analytic fixed-topology cyclic gradient; closed shell only |
| SECCM | Gated DFTB0 adapter | Analytic fixed-topology cyclic gradient; neutral closed-shell insulating 1-D H/C only |

For example, a Gamma-periodic DFTB0 slab path needs no Gaussian basis:

```python
result = run_neb(
    slab_reactant,
    slab_product,
    method="dftb0",
    kpoints=(1, 1, 1),
    semiempirical_cutoff_bohr=15.0,
    n_images=7,
    climbing_image=True,
)
result.write_qvf("dftb0_surface_path")
```

The 15 bohr DFTB0 analytic route is pinned against independent
total-energy differences on an asymmetric MgO slab. Selecting another cutoff
is allowed, but is explicitly evaluated through the total-energy central
difference controlled by `fd_step_bohr`. SCC-DFTB and GFN2-xTB deliberately
use total-energy differences: their lower-level periodic analytic prototypes
do not yet include a validated self-consistent response suitable for reaction
forces.

DFTB0 and SCC-DFTB also accept an explicit full-k mesh. This route is
zero-temperature and closed-shell, and keeps the same fractional k-mesh for
every displaced image in the native batched finite-difference gradient:

```python
result = run_neb(
    chain_reactant,
    chain_product,
    method="scc-dftb",
    kpoints=(3, 1, 1),
    n_images=7,
)
result.write_qvf("scc_dftb_full_k_path")
```

GFN2-xTB, PM6/UPM6, OMx, and generalized NDDO full-k paths remain gated and
fail before parameters or image kernels are loaded. Full-k stress, analytic
derivatives, and finite-temperature free-energy gradients are independent
capabilities and remain unavailable.

SECCM is a boundary, not a generic correction. The `method="seccm"` alias is
the production MSINDO adapter:

```python
from vibeqc.semiempirical.methods.msindo_ccm import CCMOptions

options = CCMOptions(
    translations=[[a_angstrom, 0.0, 0.0]],
    madelung=True,
)
result = run_neb(
    periodic_reactant,
    periodic_product,
    method="seccm",
    ccm_options=options,
    n_images=7,
)
result.write_qvf("msindo_seccm_path")
```

The explicitly topology-bound DFTB0 adapter is also wired into NEB for its
narrow neutral closed-shell insulating 1-D H/C envelope:

```python
result = run_neb(
    reactant,
    product,
    method="dftb0",
    seccm_topology=topology,
    n_images=7,
)
```

For `PeriodicSystem` endpoints, the explicit SECCM translations are in
Angstrom and must match the active lattice vectors exactly after unit
conversion. This prevents a reaction-path QVF from describing a different
cell than the cyclic Hamiltonian. Non-DFTB full k-point semiempirical NEB,
periodic semiempirical D3 image sums, open-shell periodic NDDO, and the
remaining direct SECCM adapters fail closed in NEB before evaluation. Energy, gradient, and stress
remain independent capabilities; NEB support does not promote stress or any
broader SECCM family claim.

## What is not in vibe-qc today

- The J^LR analytic-gradient + periodic-NEB-specific
  optimisations are still future work.
  The `run_neb(PeriodicSystem, ...)` path landed in Inc 4 +
  density warm-start lands the most impactful pieces; the
  dedicated chat will close the remaining gap (FD →
  analytic gradient).
- The vibe-view renderer's lattice-box + atom-wrap logic for
  periodic reaction paths (QVF "Inc B"). The archive already
  carries everything the renderer needs; the rendering pass is
  scheduled separately.
- Frame-to-frame anchored atom wrap in vibe-view's renderer.
  The current modulo-1 wrap can produce a visual "jump" when
  an atom crosses a cell boundary mid-animation. Rare in
  practice for chemistry NEBs; cosmetic.
- Full k-point semiempirical NEB for GFN2-xTB, PM6/UPM6, or OMx, and SECCM
  reaction-path adapters beyond MSINDO and the gated DFTB0 slice.
  Gamma-periodic or full-k support is not an SECCM
  implementation.

For TS finders that ship today see the geometry-optimization +
Hessian tooling. The relaxed-scan workflow (`relaxed_scan` /
`ScanResult`) can also provide an interpolated starting path that
NEB consumes.

## Citations

Both NEB papers fire automatically whenever `run_neb` runs (via
the `routes.drivers.neb` route registered against
`uses_neb=True`):

> Henkelman, G.; Jónsson, H.
> *Improved tangent estimate in the nudged elastic band method
> for finding minimum energy paths and saddle points.*
> J. Chem. Phys. **113**, 9978 (2000).
> [doi:10.1063/1.1323224](https://doi.org/10.1063/1.1323224)

> Smidstrup, S.; Pedersen, A.; Stokbro, K.; Jónsson, H.
> *Improved initial guess for minimum energy path calculations.*
> J. Chem. Phys. **140**, 214106 (2014).
> [doi:10.1063/1.4878664](https://doi.org/10.1063/1.4878664)

For `climbing_image=True`, additionally:

> Henkelman, G.; Uberuaga, B. P.; Jónsson, H.
> *A climbing image nudged elastic band method for finding
> saddle points and minimum energy paths.*
> J. Chem. Phys. **113**, 9901 (2000).
> [doi:10.1063/1.1329672](https://doi.org/10.1063/1.1329672)

The matching `routes.drivers.ci_neb` route fires in addition to
the base `routes.drivers.neb` route when CI-NEB is enabled.
