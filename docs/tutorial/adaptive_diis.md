# Letting DIIS choose its own depth: R-CDIIS and AD-CDIIS

Pulay's DIIS keeps a fixed-size rolling window of the last `m` iterates
(`diis_subspace_size`, default 8) and extrapolates from all of them.
That fixed `m` is a compromise: too small and the extrapolation has too
little history to work with; too large and stale, near-linearly-dependent
iterates make the least-squares problem ill-conditioned and can stall or
oscillate the tail of the SCF. This tutorial introduces two accelerators
that remove the guesswork by letting the history depth adapt on the fly,
following Chupin, Dupuy, Legendre and Sere, *Convergence analysis of
adaptive DIIS algorithms with application to electronic ground state
calculations*, ESAIM: M2AN **55**, 2785 (2021),
[doi:10.1051/m2an/2021069](https://doi.org/10.1051/m2an/2021069).

Both variants are drop-in replacements for `SCFAccelerator.DIIS`: the
extrapolation step is byte-for-byte Pulay's commutator DIIS, and only the
set of retained iterates differs. They are the adaptive-depth companions
to the [EDIIS+DIIS stiff-convergence recipe](ediis_diis_hybrid.md), which
instead switches the mixing *rule* near convergence.

## One-paragraph theory

Standard commutator DIIS minimises the norm of the extrapolated error
vector $\sum_i c_i \mathbf{e}_i$ (with $\mathbf{e} = \mathbf{FDS} -
\mathbf{SDF}$) over the last `m` iterates subject to $\sum_i c_i = 1$.
The adaptive variants keep the same minimisation but choose *which*
iterates to retain each step:

- **R-CDIIS (restarted, Algorithm 3)** lets the depth grow every
  iteration until the stored error *differences* become nearly linearly
  dependent, then restarts (drops all history but the last iterate and
  regrows). The near-dependence test projects the newest difference onto
  the span of the stored ones; a restart fires when the orthogonal
  residual is small relative to the difference itself, controlled by a
  parameter $\tau \in (0, 1)$.
- **AD-CDIIS (adaptive-depth, Algorithm 4)** keeps only the recent
  iterates whose residual is within a factor $1/\delta$ of the current
  residual, so the window shrinks smoothly as the SCF converges and the
  current residual drops. It has no restart discontinuity and, in the
  paper's numerics, tends to give the fastest convergence.

Both hold `diis_subspace_size` as a hard upper bound on stored iterates
(a memory safety cap); in practice the adaptive rule trims first, and the
mean depth is usually smaller than the fixed default.

## Selecting the accelerator

Both variants are ordinary `scf_accelerator` values, selectable by enum
or by keyword string, exactly like `DIIS` or `EDIIS_DIIS`:

```python
import vibeqc as vq

opts = vq.RHFOptions()

# Restarted CDIIS with the paper's default restart parameter.
opts.scf_accelerator = vq.SCFAccelerator.R_CDIIS
opts.diis_restart_tau = 1e-4          # tau in (0, 1); smaller => deeper history

# Or adaptive-depth CDIIS.
opts.scf_accelerator = vq.SCFAccelerator.AD_CDIIS
opts.diis_adaptive_delta = 1e-4       # delta > 0; smaller => deeper history
```

The keyword-string form (handy for YAML or CLI-driven runs) accepts
`"r_cdiis"` and `"ad_cdiis"` plus a few ergonomic spellings:

```python
opts.scf_accelerator = vq.SCFAccelerator.from_string("ad_cdiis")
```

## A molecular example

The adaptive variants converge to the same fixed point as plain DIIS;
they change the path, not the answer. Water in a minimal basis makes that
concrete:

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [0.0,  0.000,  0.221]),
    vq.Atom(1, [0.0,  1.427, -0.890]),
    vq.Atom(1, [0.0, -1.427, -0.890]),
])  # coordinates in bohr
basis = vq.BasisSet(mol, "sto-3g")

def energy(accel):
    opts = vq.RHFOptions()
    opts.scf_accelerator = accel
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    return vq.run_rhf(mol, basis, opts).energy

e_diis    = energy(vq.SCFAccelerator.DIIS)
e_rcdiis  = energy(vq.SCFAccelerator.R_CDIIS)
e_adcdiis = energy(vq.SCFAccelerator.AD_CDIIS)

assert abs(e_rcdiis  - e_diis) < 1e-9
assert abs(e_adcdiis - e_diis) < 1e-9
```

The same holds for UHF, RKS and UKS: the accelerator is a solver detail,
not a change to the energy functional.

## A periodic example

Because the depth policy lives on the single canonical DIIS kernel, the
adaptive variants are available on the periodic drivers too, at a single
k-point and across a k-mesh. Here is an H2 chain sampled at a
Monkhorst-Pack mesh:

```python
import numpy as np
import vibeqc as vq

sysp = vq.PeriodicSystem(
    1, np.diag([6.0, 30.0, 30.0]),
    [vq.Atom(1, [0.0, 15.0, 15.0]), vq.Atom(1, [1.4, 15.0, 15.0])],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])

opts = vq.PeriodicRHFOptions()
opts.scf_accelerator = vq.SCFAccelerator.AD_CDIIS
opts.diis_adaptive_delta = 1e-4

res = vq.run_rhf_periodic_multi_k_ewald3d(
    sysp, basis, kmesh, opts, omega=0.5, spacing_bohr=0.4,
)
print("converged:", res.converged, "E =", res.energy)
```

For the multi-k drivers the depth policy operates on the k-weighted DIIS
history, so the restart and residual-ratio tests use the same weighted
inner product the Fock extrapolation already minimises.

## Choosing the parameters

The paper recommends $\tau = \delta = 10^{-4}$ as a good default, and
those are the vibe-qc defaults. The knobs behave monotonically:

| Parameter | Effect of a *smaller* value | Effect of a *larger* value |
|---|---|---|
| `diis_restart_tau` (R-CDIIS) | restarts less often, larger mean depth | restarts more often, shallower history |
| `diis_adaptive_delta` (AD-CDIIS) | retains more iterates, larger mean depth | trims more aggressively, shallower history |

In the paper's experiments the convergence rate improves as the
parameters shrink, but the mean depth (and therefore the per-iteration
cost) grows roughly linearly, so $10^{-4}$ is the reported sweet spot.
If a run is memory-bound, `diis_subspace_size` still caps the worst-case
history length.

## When to reach for these

Use the adaptive variants when a fixed `diis_subspace_size` is a poor fit
for the problem: when a large window becomes ill-conditioned and stalls,
or when a small window has too little history to accelerate well. They
self-tune the depth to the problem and, in the paper's tests, converge at
least as fast as, and often faster than, a hand-picked fixed depth. For
the orthogonal failure mode, oscillation or plateauing far from the
minimum on stiff open-shell or small-gap systems, prefer the
[EDIIS+DIIS hybrid](ediis_diis_hybrid.md), which changes the mixing rule
rather than the depth. The two ideas are independent and target different
problems.

## See also

- [SCF convergence](../user_guide/scf_convergence.md), the full
  accelerator reference and per-backend support table.
- [Stiff molecular SCF: EDIIS+DIIS](ediis_diis_hybrid.md), the
  rule-switching companion for oscillating stiff cases.
