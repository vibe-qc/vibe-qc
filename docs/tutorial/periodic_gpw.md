---
myst:
  html_meta:
    "description": "Tutorial for vibe-qc's GPW (Gaussian + plane-wave) periodic route: the FFT-Poisson Hartree build derived in reciprocal space, how to invoke it with jk_method='gpw', its experimental status, and why it assumes a smooth (pseudopotential) density. The all-electron fix is GAPW."
    "og:title": "vibe-qc tutorial - the GPW periodic route (plane-wave Hartree by FFT-Poisson)"
---

# Periodic SCF on a grid: the GPW route

GPW (Gaussian and plane waves) builds the periodic Hartree potential by
mapping the density onto a real-space grid and solving Poisson's equation
by FFT in reciprocal space. It is the foundation of vibe-qc's plane-wave
family, and it assumes a smooth, pseudopotential-like density; the
all-electron fix is [GAPW](periodic_gapw.md).

```{warning}
**GPW is experimental.** It emits a `GAPWExperimentalWarning` that you have
to opt past, and its absolute energies are not yet validated for production
work. For production periodic SCF use **GDF** or **BIPOLE** (see
[choosing a periodic method](periodic_methods_compared.md)). Reach for GPW
to explore the plane-wave family or to validate vibe-qc's grid kernels.
```

GPW, **Gaussian + plane-wave**, is the first of vibe-qc's two plane-wave
periodic routes (the second, all-electron one, is
[GAPW](periodic_gapw.md)). The orbitals are still Gaussians, but the
Coulomb (Hartree) potential is built on a uniform real-space grid by
solving Poisson's equation with an FFT. The construction goes back to
Lippert and Hutter (1997) and is the engine underneath codes like CP2K.

## Why a grid: Poisson in reciprocal space

The Hartree potential of a charge density obeys Poisson's equation,

$$
\nabla^2 V_H(\mathbf{r}) = -4\pi\,\rho(\mathbf{r}).
$$

In a periodic cell this is almost free to solve, *if* you work in
reciprocal space. Expand both $\rho$ and $V_H$ in plane waves
$e^{i\mathbf{G}\cdot\mathbf{r}}$ over the reciprocal lattice
$\{\mathbf{G}\}$. The Laplacian of a plane wave just brings down
$-|\mathbf{G}|^2$, so Poisson's equation turns from a differential equation
into one division per reciprocal-lattice vector:

$$
V_H(\mathbf{G}) = \frac{4\pi}{|\mathbf{G}|^2}\,\rho(\mathbf{G}),
\qquad V_H(\mathbf{G}=\mathbf{0}) \equiv 0 .
$$

Setting the $\mathbf{G}=\mathbf{0}$ term to zero is the uniform
neutralising background that a charge-neutral cell carries. The recipe is
therefore: lay the density on a grid, FFT it to reciprocal space, divide by
$|\mathbf{G}|^2$, and FFT back. That is an $\mathcal{O}(N_g\log N_g)$
operation in the number of grid points $N_g$, the scaling that makes large
cells affordable. The collocation (density-onto-grid) and quadrature
(potential-back-to-matrix) steps, and how the smooth-cutoff is chosen, are
derived in full in the [GPW / GAPW reference](../user_guide/gapw.md).

## Running it

The simplest end-to-end GPW calculation is H₂ in a vacuum-padded cell:

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", GAPWExperimentalWarning)   # opt in to the route

# H2 in a 16-bohr cube
lattice = 16.0 * np.eye(3)
atoms = [vq.Atom(1, [-0.7, 0.0, 0.0]),
         vq.Atom(1, [+0.7, 0.0, 0.0])]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system, basis,
    method="RHF",
    jk_method="gpw",          # <- selects the GPW route
    max_iter=50,
    conv_tol_energy=1e-7,
)
print(result.energy, result.converged, result.n_iter)
```

The SCF converges (`result.converged is True`), and `result.energy` is the
periodic GPW total energy in the Ewald gauge. Because a periodic cell
carries gauge terms that a finite molecule does not, that number is **not**
the molecular H₂ energy and should not be compared with one; treat it as a
method demonstration while the route is experimental. The production routes,
where the absolute energy *is* the validated physical quantity, are
[GDF and BIPOLE](periodic_methods_compared.md).

## The catch: GPW wants a smooth density

GPW assumes the density is **smooth**. It is built for the
pseudopotential picture, where the sharp $1s$ core has been removed and what
is left varies gently enough to sit on a uniform grid. With an
all-electron Gaussian basis, the cusp at the nucleus is expensive to
resolve on any uniform grid. The fix is a per-atom augmentation that treats
the hard core on its own fine radial grid: that is [GAPW](periodic_gapw.md),
and it is the route to use when you want all-electron accuracy without
pseudopotential files.

## Runnable examples

All of these are runnable source examples. Run them in a scratch
directory when you need fresh output to inspect or diff:

- [`examples/periodic/gpw-h2-sto3g-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/gpw-h2-sto3g-rhf.py), the script above.
- [`examples/periodic/gpw-he-sto3g-lda.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/gpw-he-sto3g-lda.py), He, LDA.
- [`examples/periodic/gpw-h2-sto3g-pbe-multi-k.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/gpw-h2-sto3g-pbe-multi-k.py), multi-k pure DFT.
- [`examples/periodic/gpw-h2-sto3g-dos.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/gpw-h2-sto3g-dos.py), density of states on the grid.

## See also

- [Choosing a periodic method](periodic_methods_compared.md), GPW in
  context next to GDF and BIPOLE.
- [GAPW: the all-electron plane-wave route](periodic_gapw.md).
- [GPW / GAPW reference](../user_guide/gapw.md), the full theory and the
  grid machinery.
