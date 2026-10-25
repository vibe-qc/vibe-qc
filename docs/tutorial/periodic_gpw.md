---
myst:
  html_meta:
    "description": "Tutorial for vibe-qc's GPW (Gaussian + plane-wave) periodic route: the FFT-Poisson Hartree build derived in reciprocal space, how to invoke it with jk_method='gpw', its production envelope and how its status shows up at runtime, and why it assumes a smooth (pseudopotential) density. The all-electron fix is GAPW (experimental)."
    "og:title": "vibe-qc tutorial - the GPW periodic route (plane-wave Hartree by FFT-Poisson)"
---

# Periodic SCF on a grid: the GPW route

GPW (Gaussian and plane waves) builds the periodic Hartree potential by
mapping the density onto a real-space grid and solving Poisson's equation
by FFT in reciprocal space. It is the foundation of vibe-qc's plane-wave
family, and it assumes a smooth, pseudopotential-like density; the
all-electron fix is [GAPW](periodic_gapw.md).

GPW is a **production** route within its shipped envelope: Gamma-point RHF,
UHF, RKS and UKS; multi-k pure-DFT RKS, ROKS and UKS; Gamma analytic
gradients; DFT+U; D3-BJ dispersion. GPW ROHF and ROKS are a maintained
preview, multi-k UHF and multi-k hybrids fail closed, and the all-electron
augmentation [GAPW](periodic_gapw.md) is experimental. GDF and BIPOLE remain
the longer-validated periodic routes and the reference for cross-checks
([choosing a periodic method](periodic_methods_compared.md)).

```{note}
**How the status shows up at runtime.** GPW is production within its shipped
envelope. The `.out` route description for `jk_method="gpw"` still ends in the
word "experimental", and the plane-wave grid and J-builder infrastructure emit
`GAPWExperimentalWarning` once per grid build (three notices per run through
`run_periodic_job`; the runner's `quiet=True` silences only the driver-level
warning), while the GPW/GAPW family is completed. Read both as scope reminders
for the parts not yet in the envelope above, not as a verdict on the envelope
itself; the shipped example silences the notices with
`warnings.simplefilter("ignore", GAPWExperimentalWarning)`. GDF and BIPOLE are
the longer-validated periodic routes. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

GPW, **Gaussian + plane-wave**, is the first of vibe-qc's two plane-wave
periodic routes (the second, all-electron one, is
[GAPW](periodic_gapw.md)). The orbitals are still Gaussians, but the
Coulomb (Hartree) potential is built on a uniform real-space grid by
solving Poisson's equation with an FFT. The construction goes back to
Lippert and Hutter (1997, {vibeqc-cite}`lippert_hutter_gpw_1997`) and is the
engine underneath codes like CP2K (Quickstep,
{vibeqc-cite}`vandevondele_hutter_quickstep_2005`).

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

warnings.simplefilter("ignore", GAPWExperimentalWarning)   # silence the grid notices

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

The SCF converges in three iterations (`result.converged is True`) to
`result.energy = -1.11671345 Ha` on a vibe-qc 0.17.x development build
(Apple M2), the periodic GPW total energy in the Ewald gauge. Because a
periodic cell carries gauge terms that a finite molecule does not, that
number is **not** the molecular H₂ energy and should not be compared with
one; the meaningful cross-check is the same cell on
[GDF or BIPOLE](periodic_methods_compared.md), the longer-validated routes.
Two practical notes: the default plane-wave cutoff (`cutoff_ha=300.0`) puts
a 192x192x192 grid on this 16-bohr cube, about 1.7 GB of grid caches (the
`.out` memory block itemises them) and several minutes of wall time on a
busy machine, and the `.out` route line reads
`J/K method: GPW -- Gaussian + plane-wave J via FFT-Poisson ... ; experimental)`,
the tag discussed at the top of the page. Lower `cutoff_ha` or shrink the
box for a quicker first run.

## The catch: GPW wants a smooth density

GPW assumes the density is **smooth**. It is built for the
pseudopotential picture, where the sharp $1s$ core has been removed and what
is left varies gently enough to sit on a uniform grid (in CP2K, the
Goedecker-Teter-Hutter pseudopotentials: S. Goedecker, M. Teter and
J. Hutter, *Phys. Rev. B* **54**, 1703 (1996),
doi:[10.1103/PhysRevB.54.1703](https://doi.org/10.1103/PhysRevB.54.1703);
C. Hartwigsen, S. Goedecker and J. Hutter, *Phys. Rev. B* **58**, 3641
(1998), doi:[10.1103/PhysRevB.58.3641](https://doi.org/10.1103/PhysRevB.58.3641)). With an
all-electron Gaussian basis, the cusp at the nucleus is expensive to
resolve on any uniform grid. The fix is a per-atom augmentation that treats
the hard core on its own fine radial grid: that is [GAPW](periodic_gapw.md)
({vibeqc-cite}`lippert_hutter_parrinello_gapw_1999`,
{vibeqc-cite}`krack_parrinello_allelectron_2000`),
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
