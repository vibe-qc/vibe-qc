# Letting DIIS choose its own depth: R-CDIIS and AD-CDIIS

Pulay's DIIS keeps a fixed-size rolling window of the last `m` iterates
(`diis_subspace_size`, default 8) and extrapolates from all of them.
That fixed `m` is a compromise: too small and the extrapolation has too
little history to work with; too large and stale, near-linearly-dependent
iterates make the least-squares problem ill-conditioned and can stall or
oscillate the tail of the SCF. This tutorial introduces two accelerators
that remove the guesswork by letting the history depth adapt on the fly.
Both are production routes (label vocabulary in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)),
and the [Theory](#theory) section gives the restart and window criteria.
They follow Chupin, Dupuy, Legendre and Sere, *Convergence analysis of
adaptive DIIS algorithms with application to electronic ground state
calculations*, ESAIM: M2AN **55**, 2785 (2021),
[doi:10.1051/m2an/2021069](https://doi.org/10.1051/m2an/2021069).

Both variants are drop-in replacements for `SCFAccelerator.DIIS`: the
extrapolation step is byte-for-byte Pulay's commutator DIIS, and only the
set of retained iterates differs. They are the adaptive-depth companions
to the [EDIIS+DIIS stiff-convergence recipe](ediis_diis_hybrid.md), which
instead switches the mixing *rule* near convergence.

## Theory

**What plain DIIS does.** Pulay's commutator DIIS
({vibeqc-cite}`pulay_diis_1980`, {vibeqc-cite}`pulay_diis_1982`) stores the
Fock matrices $\mathbf F_j$ and error vectors
$\mathbf e_j = \mathbf F_j\mathbf D_j\mathbf S - \mathbf S\mathbf D_j\mathbf F_j$
of the last $m$ iterates and extrapolates
$\mathbf F = \sum_j c_j\mathbf F_j$ with the coefficients that minimise the
error of the combination under the normalisation constraint,

$$
\min_{\mathbf c}\;\Bigl\lVert \sum_{j} c_j\,\mathbf e_j \Bigr\rVert^2
\quad\text{subject to}\quad \sum_j c_j = 1 ,
$$

a small bordered linear system built from the overlaps
$\langle\mathbf e_i,\mathbf e_j\rangle$ (written out on the
[EDIIS+DIIS page](ediis_diis_hybrid.md#theory)). The window length $m$
(`diis_subspace_size`, default 8) is fixed. A short window discards history
that would still help; a long window keeps early iterates whose errors are
far larger than the current one and whose differences are nearly linearly
dependent, so the overlap matrix becomes ill-conditioned and the fitted
Fock matrix noisy. The adaptive variants of
{vibeqc-cite}`chupin_adaptive_diis_2021` keep the same minimisation and
change only *which* iterates are in the window at each step.

**R-CDIIS (restarted).** The window grows every iteration. Write
$\Delta\mathbf e_k = \mathbf e_{k+1} - \mathbf e_k$ for the newest error
difference and $\Pi_k$ for the orthogonal projector onto the span of the
stored differences. The history is restarted (everything dropped but the
last iterate, then regrown) when the new difference is nearly contained in
that span,

$$
\bigl\lVert (1 - \Pi_k)\,\Delta\mathbf e_k \bigr\rVert
\;<\; \tau\,\lVert \Delta\mathbf e_k \rVert ,
\qquad \tau \in (0, 1),
$$

which is exactly the situation in which the least-squares problem is about
to become ill-conditioned. `diis_restart_tau` sets $\tau$; the default
$10^{-4}$ is the paper's recommendation.

**AD-CDIIS (adaptive depth).** Instead of restarting, keep only the
iterates whose error is comparable to the current one,

$$
\mathcal W_k = \bigl\{\, j \le k \;:\; \lVert \mathbf e_j \rVert \le \delta^{-1}\,\lVert \mathbf e_k \rVert \,\bigr\},
\qquad \delta > 0 ,
$$

so the window shrinks smoothly as the residual falls and early, large-error
iterates leave it. There is no restart discontinuity, and in the paper's
numerics this variant tends to give the fastest convergence.
`diis_adaptive_delta` sets $\delta$; the default is again $10^{-4}$.
`diis_subspace_size` stays a hard cap on the stored iterates for both
variants (a memory safety bound); in practice the adaptive rule trims first
and the mean depth is usually below the fixed default.

**Where this helps and where it does not.** The adaptive rules address the
fixed-history compromise, not the far-from-convergence failure of the
commutator objective: they change how the tail of a well-behaved SCF is
extrapolated, and they leave the answer unchanged (the accelerator is a
solver detail, not a change to the energy functional). On a small, easy
molecule they can even take a few more iterations than plain DIIS, as the
example below shows. For oscillation far from the minimum, the
[EDIIS+DIIS hybrid](ediis_diis_hybrid.md) changes the objective instead;
the two ideas are independent.

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

On a vibe-qc 0.17.10 development build all accelerators converge to
`-74.9629979688` Ha. The iteration counts on this easy problem are 7 for
plain DIIS, 7 for the default EDIIS+DIIS, 9 for R-CDIIS and 8 for AD-CDIIS,
so the adaptive variants do not pay off on a system that converges in seven
iterations; their purpose is the stiff tail of harder SCFs.

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

On a vibe-qc 0.17.10 development build this chain (two k-points) converged
in 3 iterations to `E = -2.22162183` Ha with plain DIIS, with
`AD_CDIIS` and with `R_CDIIS` alike, so on a system this small the
accelerator changes nothing. This is the multi-k Ewald route; a
one-dimensional cell with a k mesh fails closed on the GDF route at present,
so keep GDF examples three-dimensional or Gamma-only.

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
