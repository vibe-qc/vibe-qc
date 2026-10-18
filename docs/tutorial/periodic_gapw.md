---
myst:
  html_meta:
    "description": "Tutorial for vibe-qc's GAPW (Gaussian-augmented plane-wave) periodic route: how a per-atom radial augmentation makes the GPW FFT-Poisson Hartree build all-electron without pseudopotentials, the hard/soft density split, how to invoke it with jk_method='gapw', and its experimental status (GPW itself is production within its envelope)."
    "og:title": "vibe-qc tutorial - the GAPW all-electron plane-wave route"
---

# All-electron plane waves: the GAPW route

GAPW (Gaussian-augmented plane waves) restores all-electron accuracy to
vibe-qc's plane-wave grid: it keeps the smooth [GPW](periodic_gpw.md)
FFT-Poisson Hartree build but adds a per-atom radial augmentation around
each nucleus, so no pseudopotential is needed.

```{admonition} Status: experimental
:class: warning

GAPW (`jk_method="gapw"`) is an experimental route. The `.out` route line ends
in "(experimental)", and `run_periodic_job` emits `GAPWExperimentalWarning`
(the plane-wave grid and J-builder infrastructure warn once per grid build);
the compact-crystal augmentation is not yet validated, which is why RHF/UHF GAPW
requires the `gapw_molecular_limit=True` declaration and fails closed without it.
Its numbers are research output, not validated results, and it is not part of the
release gate. Its smooth-grid parent [GPW](periodic_gpw.md) is production within
its shipped envelope; GDF and BIPOLE are the longer-validated periodic routes
([how to choose](periodic_methods_compared.md)). The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

GAPW, **Gaussian-augmented plane-wave**, is [GPW](periodic_gpw.md) made
all-electron. GPW needs a smooth density and is built for pseudopotentials;
GAPW keeps GPW's fast FFT-Poisson Hartree build for the smooth part and adds
a small per-atom correction for the sharp core, so you get all-electron
accuracy **without any pseudopotential files**, only atomic numbers. The
construction is due to Lippert and Hutter (1999,
{vibeqc-cite}`lippert_hutter_parrinello_gapw_1999`) and Krack and Parrinello
(2000, {vibeqc-cite}`krack_parrinello_allelectron_2000`), building on GPW
({vibeqc-cite}`lippert_hutter_gpw_1997`).

## The hard/soft split

The obstacle GPW hits is the cusp in the density at each nucleus, which a
uniform grid cannot resolve cheaply. GAPW sidesteps it by writing the
density as a smooth grid part plus, on each atom $a$, a hard part that is
cancelled again outside that atom's augmentation sphere:

$$
\rho = \tilde{\rho} \;+\; \sum_a \big(\rho_a - \tilde{\rho}_a\big),
$$

where $\tilde{\rho}$ is the smooth density carried on the shared FFT grid,
$\rho_a$ is the hard all-electron atomic density on a fine per-atom radial
grid, and $\tilde{\rho}_a$ is a soft atomic density that matches
$\tilde{\rho}$ outside the sphere. Because the Hartree operator is linear,
the potential splits the same way:

$$
V_H[\rho] = V_H[\tilde{\rho}]
            \;+\; \sum_a \big(V_H[\rho_a] - V_H[\tilde{\rho}_a]\big).
$$

The first term is exactly the GPW FFT-Poisson solve, reused verbatim. The
per-atom corrections are cheap analytic radial Poisson solves integrated
with Lebedev angular quadrature ({vibeqc-cite}`lebedev_laikov_1999`) on
Mura-Knowles radial shells ({vibeqc-cite}`mura_knowles_radial_1996`) inside
each sphere. The cusp is handled
exactly on the atom's own grid, the smooth tail on the shared grid, and the
augmentation radii are set automatically per element. The full machinery is
in the [GPW / GAPW reference](../user_guide/gapw.md).

## Running it

He is the simplest closed-shell all-electron test:

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", GAPWExperimentalWarning)   # opt in

lattice = 12.0 * np.eye(3)
atoms = [vq.Atom(2, [0.0, 0.0, 0.0])]            # He
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system, basis,
    method="RHF",
    jk_method="gapw",          # <- selects the all-electron augmented route
    gapw_molecular_limit=True, # <- declares an isolated single-Gamma box
    max_iter=30,
    conv_tol_energy=1e-7,
)
print(result.energy, result.converged)
```

On a vibe-qc 0.17.x development build (Apple M2) this converges to
-2.35194424 Ha; the same He cell on plain GPW gives -2.80778396 Ha, so the
augmentation shift printed by the next block is +0.4558 Ha. Both runs emit
`GAPWExperimentalWarning`. The GPW number is the molecular RHF/STO-3G He
energy (-2.80778396 Ha) to within 1e-8 Ha (rechecked on a vibe-qc 0.17.17
development build), so in this basis the GAPW energy lies 0.456 Ha above
the all-electron answer.

## Seeing the augmentation work

The point of GAPW is the per-atom correction, so the instructive experiment
is to run the *same* cell through plain GPW and through GAPW and look at the
difference:

```python
gpw  = vq.run_periodic_job(system, basis, method="RHF", jk_method="gpw",
                           max_iter=30, conv_tol_energy=1e-7)
gapw = vq.run_periodic_job(system, basis, method="RHF", jk_method="gapw",
                           gapw_molecular_limit=True,
                           max_iter=30, conv_tol_energy=1e-7)
print("augmentation shift:", gapw.energy - gpw.energy, "Ha")
```

For He/STO-3G this is a consistency check, not a demonstration of a
correction: the STO-3G He density is smooth enough that plain GPW already
reproduces the molecular all-electron energy, and the hard/soft split above
only regroups the same density, so a correct augmentation should leave the
energy essentially unchanged. The +0.456 Ha shift the current build prints is therefore an
error of the experimental GAPW HF route, not physics the smooth grid misses,
and it is one reason GAPW stays experimental. Read the shift as a defect
indicator that should go to zero, not as a GAPW result. For a
validated absolute number on a compact crystal, use
[GDF or BIPOLE](periodic_methods_compared.md); GPW itself is production for the
Gamma and multi-k pure-DFT envelope listed on [its page](periodic_gpw.md).

`gapw_molecular_limit=True` is a physical-regime declaration, not a tuning
knob. The fit-free GAPW HF augmentation used here omits dense-cell image terms,
and no geometry-only heuristic safely distinguishes every isolated box from a
dense crystal supercell. Omit the declaration and RHF/UHF GAPW fails closed;
use GDF or BIPOLE for compact crystals.

## Runnable example

- [`examples/periodic/gapw-he-sto3g-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/gapw-he-sto3g-rhf.py), the script above; it runs both
  GAPW and GPW and prints the augmentation shift.

## See also

- [The GPW route](periodic_gpw.md), the smooth-grid foundation GAPW builds
  on.
- [Choosing a periodic method](periodic_methods_compared.md).
- [GPW / GAPW reference](../user_guide/gapw.md).
