---
myst:
  html_meta:
    "description": "Tutorial for running Born-Oppenheimer molecular dynamics in vibe-qc: the velocity-Verlet integrator behind the NVE microcanonical ensemble, the Langevin thermostat that controls temperature in NVT, and the Berendsen barostat that controls pressure in NPT, driven from the periodic GAPW route through run_nve / run_nvt / run_npt."
    "og:title": "vibe-qc tutorial - molecular dynamics (NVE, NVT, NPT)"
---

# Molecular dynamics (NVE, NVT, NPT)

Molecular dynamics moves the nuclei. Where the SCF tutorials stop at a
single energy and the [geometry optimizer](periodic_geometry_optimization.md)
walks downhill to the nearest minimum, MD integrates Newton's equations
forward in time so the atoms actually move, sampling the thermal motion
that geometry optimization freezes out. This page runs three short
trajectories on the same small molecule and shows the real printed output
of each: NVE (constant energy), NVT (constant temperature), and a
description of NPT (constant pressure). The theory section explains the
velocity-Verlet integrator that propagates every step, the thermostat that
holds temperature, and the barostat that holds pressure.

```{warning}
The `run_nve` / `run_nvt` / `run_npt` helpers drive MD through the
**periodic GAPW** route, which is experimental: the underlying SCF emits a
`GAPWExperimentalWarning` you opt past, and its absolute energies are not
yet production-validated (see the [GAPW tutorial](periodic_gapw.md)). MD on
this route is for learning the mechanics and for small exploratory runs,
not production trajectories. The H2 examples below explicitly use block-mode
augmentation, so every step is a full all-electron SCF plus an analytic
gradient. Keep `n_steps` and the system small.
```

## The driver and its ensembles

vibe-qc exposes molecular dynamics as four functions at the top level,
`vibeqc.run_md` and the three ensemble shorthands `run_nve`, `run_nvt`,
`run_npt`. Each takes an ASE `Atoms` object that must be fully 3D-periodic
(the GAPW SCF underneath only runs in a cell), attaches the
[`VibeqcGAPW`](periodic_gapw.md) calculator, seeds velocities from a
Maxwell-Boltzmann distribution at the requested temperature, removes the
centre-of-mass drift, and propagates the requested number of steps. The
same `Atoms` object comes back with updated positions and velocities.

| Function | Ensemble | What is held fixed | Integrator / coupling |
|---|---|---|---|
| `run_nve` | NVE (microcanonical) | energy E | velocity Verlet |
| `run_nvt` | NVT (canonical) | temperature T | Langevin thermostat |
| `run_npt` | NPT (isothermal-isobaric) | temperature T, pressure P | Berendsen thermostat + barostat |

The full general driver, its `MDTrajectory` return object, the alternative
Berendsen and Nose-Hoover thermostats, and well-tempered metadynamics on
top of MD are documented in the
[molecular-dynamics user guide](../user_guide/molecular_dynamics.md). This
tutorial stays on the three periodic-GAPW ensemble helpers.

## A molecule to move

Any closed-shell molecule placed in a periodic box works. The cheapest
system that still runs in minutes is H2 in the STO-3G basis, in a 5 Angstrom
cubic box with Gamma-only Hartree-Fock. So that the bond oscillates gently
around its minimum rather than being flung inward by a large starting force,
place the two atoms at the STO-3G periodic equilibrium bond length, which a
six-point energy scan in this box puts at about 0.617 Angstrom (the scanned
energy minimum is -35.286 eV near 0.60 Angstrom; the parabola through the
three lowest points peaks at d = 0.6166 Angstrom). Starting far from this
length instead pours that potential energy into the bond and the trajectory
heats up fast.

```python
import warnings
import numpy as np
from ase import Atoms
from vibeqc.periodic_gapw_md import run_md
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", GAPWExperimentalWarning)  # opt in to GAPW
np.random.seed(0)                                         # reproducible velocities

a = 5.0          # cubic box edge, Angstrom
d = 0.6166       # H-H bond at the STO-3G periodic minimum, Angstrom
atoms = Atoms("H2",
              positions=[[0.0, 0.0, 0.0], [0.0, 0.0, d]],
              cell=[a, a, a], pbc=True)
atoms.center()
```

`np.random.seed(0)` fixes the Maxwell-Boltzmann velocity draw so the numbers
below reproduce; drop it for an independent sample. The calculator keywords
(`basis`, `functional`, `cutoff_ha`, ...) are forwarded verbatim through the
MD helpers to `VibeqcGAPW`; `functional=None` selects Hartree-Fock.

The two HF examples below deliberately pass
`gapw_kwargs={"one_centre": "block"}`. Hydrogen has no pruned core under the
GAPW softening split, so the legacy block and fit-free analytic one-centre
corrections are physically equivalent for H2. Explicit block mode keeps the
analytic-gradient path used for the printed timings. For a core-bearing
molecule in a declared molecular-limit cell, leave `one_centre` at `"auto"`
and pass `gapw_kwargs={"molecular_limit": True}` instead. `VibeqcGAPW` then
selects analytic augmentation and obtains forces from central differences of
full SCF energies because a matching analytic derivative is not implemented.

## NVE: constant energy

The microcanonical ensemble conserves the total energy. Nothing is added or
removed: the velocity-Verlet integrator just propagates the atoms on the
Born-Oppenheimer surface, and the total energy `E_tot = E_pot + E_kin`
should stay put while energy sloshes between potential and kinetic as the
bond stretches and compresses. `run_md(..., md_engine="nve")` with
`log_interval=1` prints every step (the `run_nve` shorthand prints only
every tenth step, which hides the per-step detail on a four-step run).

```python
run_md(atoms, md_engine="nve", temperature_K=60.0, timestep_fs=0.25,
       n_steps=4, log_interval=1,
       basis="sto-3g", functional=None, cutoff_ha=60.0,
       gapw_kwargs={"one_centre": "block"})
```

```text
=== NVE via run_md, log_interval=1, 4 steps, dt=0.25 fs, T0=60 K, d=0.6166 Ang (equilibrium), cutoff 60 Ha ===
  step    0:  E_pot = -1.297131 Ha  E_tot = -1.295830 Ha  T =  273.9 K
  step    1:  E_pot = -1.297102 Ha  E_tot = -1.296190 Ha  T =  192.0 K
  step    2:  E_pot = -1.297012 Ha  E_tot = -1.296407 Ha  T =  127.2 K
  step    3:  E_pot = -1.296982 Ha  E_tot = -1.296458 Ha  T =  110.1 K
  step    4:  E_pot = -1.297046 Ha  E_tot = -1.296339 Ha  T =  148.9 K
NVE wall: 527.25 s
```

The MD status rows follow vibe-qc's active output policy: they use Ha by
default, while `VIBEQC_OUTPUT_UNITS=eV` converts both values and labels. The
total energy holds near -1.296 Ha across the run, its full excursion under
0.7 mHa (from -1.295830 Ha at step 0 to -1.296458 Ha at step 3), while the
potential energy barely moves (the bond is at its minimum) and the
temperature swings as kinetic energy trades with potential. That bounded wobble, not a flat line, is the correct signature of
a symplectic integrator at finite timestep: velocity Verlet has no secular
energy drift, and the wobble amplitude shrinks with the timestep (it scales
as the square of the step). Halve `timestep_fs` and the energy excursion
falls by about four.

```{note}
The instantaneous temperature of a two-atom molecule swings wildly because
it has only three momentum degrees of freedom (`3N - 3 = 3`). Temperature
is a kinetic-energy average over many degrees of freedom, so on H2 it is a
noisy per-step number, illustrative of the mechanics, not a converged
thermodynamic average. Real thermostat statistics need a larger system and
a far longer run.
```

## NVT: constant temperature

The canonical ensemble holds the temperature fixed instead of the energy. A
real system exchanges heat with its surroundings; an NVT thermostat mimics
that bath. The GAPW `run_nvt` path uses ASE's **Langevin** thermostat: each
step adds a friction term that bleeds energy away and a random force that
feeds energy in, tuned so the long-run kinetic temperature relaxes toward
the target. The default friction is 0.01 inverse femtoseconds (about
10 inverse picoseconds, a relaxation time near 100 fs).

```python
np.random.seed(0)
atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, d]],
              cell=[a, a, a], pbc=True)
atoms.center()

run_md(atoms, md_engine="nvt", temperature_K=300.0, timestep_fs=0.25,
       n_steps=6, log_interval=1,
       basis="sto-3g", functional=None, cutoff_ha=60.0,
       gapw_kwargs={"one_centre": "block"})
```

ASE prints a one-time `FutureWarning` here that its `fixcm=True`
centre-of-mass handling does not strictly sample the canonical
distribution for small systems (it is fine for large ones), a fair warning
for a two-atom demo. The opening frames of the run:

```text
=== NVT (Langevin) via run_md, log_interval=1, dt=0.25 fs, T=300 K, d=0.6166 Ang, cutoff 60 Ha ===
  step    0:  E_pot = -1.297131 Ha  E_tot = -1.290626 Ha  T = 1369.4 K
  step    1:  E_pot = -1.296826 Ha  E_tot = -1.291744 Ha  T = 1069.8 K
```

The initial Maxwell-Boltzmann draw at 300 K landed this 3-degree-of-freedom
system at an instantaneous 1369 K (huge fluctuations are intrinsic to so few
degrees of freedom), and one Langevin step already bleeds it down toward the
target: T falls to 1070 K while the total energy drops from -35.120 to
-35.150 eV (-1.290626 to -1.291744 Ha in the status rows) as the friction
removes kinetic energy.

Unlike the NVE run, the total energy is no longer conserved here, and it
should not be: the thermostat is deliberately injecting and removing energy
to steer the temperature. On a system this small and a run this short you
see the thermostat acting (the random force kicks the temperature around
the 300 K target) rather than a converged canonical average; the same call
on a condensed-phase system run for tens of picoseconds is what samples the
true ensemble. The `run_nvt` shorthand wraps exactly this call with the
Langevin thermostat and a ten-step logging cadence.

## NPT: constant temperature and pressure

The isothermal-isobaric ensemble holds both temperature and pressure fixed
by letting the cell volume change. This is the ensemble for equation-of-state
work, thermal-expansion coefficients, and any property measured at a set
pressure rather than a set volume. `run_npt` couples ASE's **NPTBerendsen**
engine: a Berendsen thermostat rescales velocities toward the target
temperature while a Berendsen barostat rescales the cell toward the target
pressure, each with its own weak-coupling time (about 50 fs for temperature
and 100 fs for pressure by default, with a water-like isothermal
compressibility).

```python
from vibeqc import run_npt

# Schematic: NPT needs a condensed cell whose volume can respond to
# pressure. A single molecule in a fixed box is not a meaningful NPT
# system, so use a bulk solid here.
# run_npt(bulk_atoms, temperature_K=300.0, pressure_GPa=0.0,
#         timestep_fs=0.5, n_steps=100,
#         basis="sto-3g", functional="lda", cutoff_ha=300.0)
```

Pressure is given in GPa (0 GPa is ambient) and converted internally. NPT
only makes physical sense for a condensed phase whose volume can respond to
the barostat, so unlike the NVE and NVT demos above it is not meaningful on
an isolated molecule in a large box, the barostat would simply rescale empty
vacuum. Reach for it on a bulk solid or a dense liquid, with a denser FFT
grid (`cutoff_ha` around 300) and a pure functional for any multi-k mesh.

## Theory

The three ensembles share one engine, the velocity-Verlet integrator, and
differ only in what extra bookkeeping they bolt on, a thermostat for NVT, a
thermostat plus a barostat for NPT. The sections below derive the
integrator and sketch each coupling.

### Velocity Verlet

MD integrates Newton's second law, $\ddot{\mathbf{R}} = \mathbf{F}/M$ with
$\mathbf{F} = -\nabla_{\mathbf{R}} E$, by discrete timesteps $\Delta t$.
The **velocity-Verlet** scheme (Swope et al. 1982) advances positions and
velocities as

$$
\mathbf{R}(t + \Delta t) = \mathbf{R}(t) + \mathbf{v}(t)\,\Delta t
  + \tfrac{1}{2}\,\frac{\mathbf{F}(t)}{M}\,\Delta t^2,
$$
$$
\mathbf{v}(t + \Delta t) = \mathbf{v}(t)
  + \frac{\mathbf{F}(t) + \mathbf{F}(t + \Delta t)}{2 M}\,\Delta t.
$$

It needs one force evaluation per step. In the explicit block-mode H2 runs
above, that evaluation is one full GAPW SCF plus its analytic gradient. The
scheme is time-reversible and **symplectic**: it conserves a quantity close to
the true energy exactly, so the total energy oscillates within a bounded band
rather than drifting away over time. The band width scales as $\Delta t^2$,
which is why halving the timestep shrinks the energy wobble roughly fourfold.
This bounded-but-nonzero fluctuation, not a perfectly flat energy, is what
correct NVE looks like, and it is exactly what the run above shows.

### The thermostat (NVT)

A bare velocity-Verlet run conserves energy, so it samples NVE, not the
canonical NVT ensemble a fixed-temperature experiment lives in. A
**thermostat** couples the system to a heat bath so the time-averaged
kinetic temperature
$T = 2 E_\text{kin} / (N_\text{dof} k_B)$ matches a target.

vibe-qc's GAPW `run_nvt` uses the **Langevin** thermostat, which augments
the equation of motion with a friction and a random force,

$$
M\,\ddot{\mathbf{R}} = \mathbf{F}(\mathbf{R})
  - \gamma M\,\dot{\mathbf{R}} + \sqrt{2\,\gamma M k_B T}\;\boldsymbol{\xi}(t),
$$

where $\gamma$ is the friction coefficient and $\boldsymbol{\xi}$ is
Gaussian white noise. The friction drains energy, the noise feeds it back,
and the fluctuation-dissipation balance between them fixes the stationary
distribution at temperature $T$. ASE integrates this with the
quasi-symplectic Vanden-Eijnden-Ciccotti propagator (2006). The general
`vibeqc.md` driver additionally offers the **Berendsen** weak-coupling
thermostat (velocities rescaled toward the target each step) and the
**Nose-Hoover** extended-system thermostat (a genuine conserved quantity,
samples the exact canonical distribution); see the
[user guide](../user_guide/molecular_dynamics.md).

### The barostat (NPT)

To hold pressure rather than volume, NPT adds a **barostat** that lets the
cell breathe. vibe-qc's `run_npt` uses the **Berendsen** scheme (Berendsen
et al. 1984): each step the cell vectors (and the atomic coordinates with
them) are scaled by a factor that relaxes the instantaneous pressure toward
the target with a weak-coupling time $\tau_P$,

$$
\mu = \Bigl[1 - \frac{\beta\,\Delta t}{\tau_P}\,(P_0 - P)\Bigr]^{1/3},
\qquad
\mathbf{h} \leftarrow \mu\,\mathbf{h},
$$

with $\beta$ the isothermal compressibility, $\mathbf{h}$ the cell matrix,
$P$ the instantaneous (virial) pressure, and $P_0$ the target. A Berendsen
thermostat runs alongside it for the temperature. Berendsen coupling is
robust and strongly damping but does not reproduce the exact NPT
fluctuations, so it is a fine equilibrator and a serviceable production
barostat for averages, while distributional quantities call for an
extended-system (Parrinello-Rahman style) barostat.

## Resources

For the explicit block-mode H2 examples, each MD step is one periodic GAPW
SCF plus one analytic all-electron gradient. Cost therefore scales with
`n_steps` times the per-step SCF+gradient cost. For H2 / STO-3G in a 5
Angstrom box at `cutoff_ha=60` on an Apple-M class core, a step lands in
roughly one to two minutes here, dominated by the gradient. Analytic
molecular-limit mode instead uses a central finite difference of the full SCF
energy: one force request adds `6N` displaced SCFs for `N` atoms, in addition
to the central energy. That makes the GAPW MD helpers usable for short
demonstrations and tens-of-step exploratory runs, not for production
trajectories of thousands of steps. The two levers on per-step cost are the
basis size and the FFT grid (`cutoff_ha`); the timestep trades accuracy
(energy-conservation quality) against how much simulated time each step
buys.

## References

- **Velocity-Verlet integrator.** W. C. Swope, H. C. Andersen, P. H.
  Berens, and K. R. Wilson, "A computer simulation method for the
  calculation of equilibrium constants for the formation of physical
  clusters of molecules," *J. Chem. Phys.* **76**, 637 (1982). The velocity
  form of L. Verlet, "Computer experiments on classical fluids," *Phys.
  Rev.* **159**, 98 (1967).
- **Langevin thermostat (the NVT integrator used here).** The propagator
  ASE integrates is Eq. 23 of E. Vanden-Eijnden and G. Ciccotti,
  "Second-order integration scheme for Langevin equations with holonomic
  constraints," *Chem. Phys. Lett.* **429**, 310 (2006).
- **Berendsen thermostat and barostat (NPT).** H. J. C. Berendsen, J. P. M.
  Postma, W. F. van Gunsteren, A. DiNola, and J. R. Haak, "Molecular
  dynamics with coupling to an external bath," *J. Chem. Phys.* **81**, 3684
  (1984).
- **Nose-Hoover thermostat (alternative canonical thermostat).** S. Nose,
  "A unified formulation of the constant temperature molecular dynamics
  methods," *J. Chem. Phys.* **81**, 511 (1984); W. G. Hoover, "Canonical
  dynamics: equilibrium phase-space distributions," *Phys. Rev. A* **31**,
  1695 (1985).
- **The GAPW route under the MD driver.** G. Lippert, J. Hutter, and M.
  Parrinello, "The Gaussian and augmented-plane-wave density functional
  method for ab initio molecular dynamics simulations," *Theor. Chem. Acc.*
  **103**, 124 (1999); M. Krack and M. Parrinello, "All-electron ab-initio
  molecular dynamics," *Phys. Chem. Chem. Phys.* **2**, 2105 (2000).
- **ASE (the integrators and Atoms object).** A. H. Larsen et al., "The
  atomic simulation environment, a Python library for working with atoms,"
  *J. Phys. Condens. Matter* **29**, 273002 (2017).

## Next

- [Periodic geometry optimization](periodic_geometry_optimization.md), find
  the minimum the dynamics fluctuates around before you heat it up.
- [Vibrational frequencies](vibrational_frequencies.md), the harmonic modes
  whose thermal population MD samples directly.
- [The GAPW route](periodic_gapw.md), the all-electron plane-wave SCF that
  supplies the energy and forces at every MD step.
- [Molecular-dynamics user guide](../user_guide/molecular_dynamics.md), the
  general driver, the `MDTrajectory` object, the Berendsen and Nose-Hoover
  thermostats, and well-tempered metadynamics.
