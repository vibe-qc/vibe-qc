# v0.6.x periodic-SCF bug — root-cause analysis

**Status (v0.7 development branch).** Translation invariance of the
periodic-SCF total energy is broken. H₂ in a 30 bohr cubic box gives a
+0.587 Ha shift between "atoms at the box origin" and "atoms at the box
centre". PySCF.pbc gives the same energy regardless (diff ≈ 1e-13 Ha).

This file documents what the bug **is** and **where** it lives, so the
v0.7 fix can be implemented without retracing the diagnosis.

## Pre-fix repro

```
$ .venv/bin/python examples/debug/scf_translation_invariance_check.py
…
  h2_at_origin   E = -1.702447 Ha   (5 iters)
  h2_centered    E = -1.114878 Ha   (2 iters)
  E(centered) - E(origin) = +0.587568 Ha
  verdict: ✗ TRANSLATION INVARIANCE BROKEN
```

## What was *not* the bug

The summary the v0.6.2 handover hands to the next dev chat said the
gauge mismatch lived in V_ne (electron-nuclear) — `compute_nuclear_lattice`
(direct truncated libint, no G=0 omission) vs `build_j_ewald_3d` (FFT-Poisson,
G=0 dropped). That hypothesis is **wrong**. Verified against the actual
matrices:

```
$ .venv/bin/python examples/debug/check_v_ne_translation.py
Frobenius diffs (||M_origin - M_centered||):
  S       = 1.6e-16
  T       = 2.2e-16
  V_bare  = 6.7e-16     ← bare libint V_ne
  V_ewald = 9.4e-16     ← Ewald-built V_ne
```

All one-electron matrix elements **are** translation-invariant within
machine precision, both in the bare libint path and via
`compute_nuclear_lattice_ewald`. Switching V_ne to dispatch on
`coulomb_method` is a clean improvement (V_ne now matches the J gauge
on G=0 dropping) but is **not** the translation-invariance fix.

## What the bug actually is

`build_jk_gamma_molecular_limit(...).J` (the short-range J via real-space
ERIs) is fine — translation-invariant by construction. The break is
exclusively in **the long-range J built via FFT-Poisson**:

```
$ .venv/bin/python examples/debug/check_j_translation.py
||J_SR diff|| = 5.8e-16          ← real-space ERIs are clean
||J_LR diff|| = 9.94e-1          ← FFT-Poisson J is broken
                tr(D·J_LR): origin=0.154   centred=1.327
```

The 1.17 Ha drift in `tr(D·J_LR)` between configurations propagates as
0.5 × 1.17 = 0.587 Ha into the SCF total energy — exact match.

### Smoking gun: ρ on the grid doesn't integrate to N_e

```
$ .venv/bin/python examples/debug/probe_v_lr.py
  origin   rho.sum * dV = 0.624541   ← H₂ has 2 electrons, but grid integral says 0.62
  centre   rho.sum * dV = 2.000020   ← centered: correct
```

When the molecule sits at the **box origin** the grid integration
captures only the AO-density octant inside `[0, L)^3` — the other
seven octants of the AO Gaussian (mathematically extending into
`[-L/2, 0)` along each axis) are simply lost because the grid only
samples `[0, L)`. For an isotropic Gaussian centered at the origin
exactly 1/8 of the integral lives in the home octant; for an AO at
z=1.4 a non-trivial fraction still spills into z<0. The two AOs of
H₂ together end up with ~0.62 of their normalisation captured.

When the molecule sits at the **box centre** every AO Gaussian's full
extent fits inside `[0, L)^3` and ρ integrates to exactly 2.

This is *not* a sampling-error issue — it's a consequence of evaluating
non-periodic AOs on a finite-extent grid that only covers one home cell.
The FFT-Poisson then thinks it's solving Poisson on a charge density
of 0.62 electrons spread oddly across the home cell, which is wrong by
construction.

### The fix path

The grid-sampled ρ has to be the **periodic** density:

    ρ(r) = Σ_g Σ_{μν} D(g)_{μν} χ_μ(r − R_μ − g) χ_ν(r − R_ν)

with the sum over images **g** spanning enough lattice cells that the
AO Gaussian's tail is accounted for everywhere on the grid.

`evaluate_periodic_density_on_grid` (in `python/vibeqc/periodic_density.py`)
already iterates over `D_real.cells`, but those cells are determined by
`lat_opts.cutoff_bohr` — for `cutoff_bohr=12, L=30` only the home cell
makes the cut, so no images are summed. For tight bases like STO-3G
(α_min ≈ 0.16 → tail extends a few bohr) this misses the AO-image
contributions at the box boundary.

Two ways to fix it, in order of preference:

1. **Sum AO images directly inside `evaluate_periodic_density_on_grid`**
   regardless of `D_real.cells`. The image-cell radius needed is just
   the AO Gaussian RMS (a few bohr for STO-3G, ~10 bohr for cc-pVDZ),
   independent of `lat_opts.cutoff_bohr`. This is exactly what PySCF
   does in `pyscf.pbc.df.fft.get_pp` / `get_nuc`: the AOs are evaluated
   periodically on the FFT grid, the density is built, the Poisson
   kernel is applied, the AO×V×AO integrals are summed back.
2. **Bump `cutoff_bohr`** so the cells list includes images at distance L.
   Wasteful (it pulls into the lattice sum a lot of cells whose D-block
   is essentially the home-cell D anyway), but a one-line bandaid.

Path 1 is the v0.7 deliverable. The implementation site is
`evaluate_periodic_density_on_grid` plus the matching loop in
`build_j_long_range_periodic`'s integration step (the ν-side AO at the
home cell needs to become a periodic AO sum just like the μ-side).

After the fix lands:

* `examples/debug/scf_translation_invariance_check.py` should report
  diff < 1e-5 Ha (currently +0.587 Ha).
* The v0.6.1 Madelung energy correction in
  `python/vibeqc/madelung.py::madelung_energy_correction` becomes the
  *only* leftover correction; for sufficiently-large boxes it should
  drop to <1 mHa as well, at which point we can retire it.
* `tests/test_periodic_atomic_limit_bug.py` and
  `tests/test_periodic_dense_ionic_bug.py` xfails can be flipped to
  pass.

## Diagnostic kit

The four scripts in this directory anchor the regression-test fixture:

| Script | What it isolates |
|---|---|
| `scf_translation_invariance_check.py` | End-to-end: H₂ at origin vs centre, full SCF, total-energy diff. |
| `check_v_ne_translation.py` | One-electron integrals (S/T/V_bare/V_ewald). Shows V_ne is *not* the bug. |
| `check_j_translation.py` | Decomposes J = J_SR + J_LR and shows J_LR is the breakage. |
| `probe_v_lr.py` | Direct probe of ρ-on-grid showing the 0.62 vs 2.0 integration mismatch. |

Run all four after the fix lands; expect the J_LR diff and the SCF diff
to both drop to ~1e-5 Ha.
