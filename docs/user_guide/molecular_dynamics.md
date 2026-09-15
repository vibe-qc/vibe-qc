# Molecular dynamics

See also the [tutorial](../tutorial/molecular_dynamics.md) for a worked
NVE/NVT/NPT walkthrough.

`vibeqc.md` is a **general, method-agnostic** Born-Oppenheimer molecular
dynamics driver. It integrates Newton's equations for any callable that
returns an energy and a nuclear gradient, so the same driver runs MD on the
MSINDO semiempirical surface or on any other vibe-qc potential-energy
surface, exactly the way [`run_neb`](neb.md) drives a band over an
energy+gradient seam.

```{note}
The top-level names `vibeqc.run_md` / `run_nve` / `run_nvt` are the
**periodic GAPW** Born–Oppenheimer helpers (they wrap an ASE calculator —
see {mod}`vibeqc.periodic_gapw_md`). The general driver described here lives
in its own module and is imported as `from vibeqc.md import run_md`.
```

## The energy+gradient interface

The driver consumes a `ForceProvider`, a callable

```python
force_fn(coords) -> (energy, gradient)
```

where `coords` is an `(n_atoms, 3)` array of positions **in bohr**, `energy`
is in **Hartree**, and `gradient = dE/dR` has the same shape as `coords` **in
Ha/bohr** (forces are `-gradient`; the driver negates internally, matching
the `run_neb` convention). Anything that can produce an energy and a gradient
- a semiempirical engine, an SCF + analytic gradient, a model potential, 
plugs in through this one interface.

Everything is in **atomic units**: bohr, Hartree, electron masses, and atomic
time units (ħ/E_h ≈ 0.0242 fs). The timestep and thermostat coupling time are
given in femtoseconds and converted internally; trajectory velocities are
stored in bohr / atomic-time-unit.

## Ensembles and thermostats

| Ensemble | `thermostat=` | Reference |
|----------|---------------|-----------|
| **NVE** (microcanonical) | `None` | velocity Verlet (Swope 1982) |
| **NVT** (canonical) | `"berendsen"` | Berendsen weak coupling (1984) |
| **NVT** (canonical) | `"nose_hoover"` | Nosé 1984 / Hoover 1985 |

* **NVE** uses the symplectic velocity-Verlet integrator, which conserves the
  total energy with no secular drift (the oscillation amplitude scales ∝ Δt²).
* **Berendsen** rescales the velocities each step to relax the kinetic
  temperature toward the target with a coupling time `thermostat_tau_fs`. It
  is robust and strongly damping but does **not** sample the exact canonical
  ensemble.
* **Nosé-Hoover** is a single extended-system thermostat with a genuine
  conserved quantity `H_NH = KE + PE + ½Qζ² + N k_B T η`; it samples the true
  canonical distribution. (A single thermostat on purely harmonic modes is
  non-ergodic, Martyna-Klein-Tuckerman 1992, so realistic anharmonic
  systems sample best.)

## Running MD on the MSINDO surface

`run_msindo_md` is the MSINDO-aware convenience wrapper. It takes the initial
geometry in Ångström, builds the MSINDO energy+gradient provider, and returns
an `MDTrajectory`:

```python
import numpy as np
from vibeqc.semiempirical.methods.msindo import run_msindo_md

Z = [8, 1, 1]                      # H2O
xyz = np.array([[0.0,  0.0,    0.0],
                [0.0,  0.7572, 0.5865],
                [0.0, -0.7572, 0.5865]])   # Ångström

# Microcanonical (NVE) — check energy conservation.
traj = run_msindo_md(Z, xyz, timestep_fs=0.3, n_steps=50,
                     temperature_K=300.0, thermostat=None, seed=1)
print("E drift:", traj.energy_drift(), "Ha")

# Canonical (NVT) with a Nosé–Hoover thermostat.
traj = run_msindo_md(Z, xyz, timestep_fs=0.3, n_steps=200,
                     temperature_K=300.0, thermostat="nose_hoover",
                     thermostat_tau_fs=25.0, seed=1, output="h2o_md")
print("mean T:", traj.mean_temperature(), "K")
```

```{warning}
MSINDO exposes a **finite-difference** gradient, so every MD step costs
`6N + 1` SCFs. This is fine for short trajectories on small molecules but is
not a production MD engine — keep `n_steps` modest.
```

### `MDTrajectory`

`run_md` returns an `MDTrajectory` dataclass with one entry per recorded
frame (`record_stride` controls the recording cadence): `times_fs`,
`positions`, `velocities`, `potential_energy`, `kinetic_energy`,
`total_energy`, `temperature`, and `conserved_energy` (the total plus the
thermostat bath energy, for NVE this equals the total energy; for
Nosé-Hoover it is `H_NH`). Convenience methods:

* `energy_drift()`, max deviation of the conserved quantity from its initial
  value (the NVE energy drift, or the `H_NH` drift under Nosé-Hoover).
* `mean_temperature(skip_fraction=0.5)`, mean temperature over the last
  portion of the run (discarding equilibration).

## Driving any method

Because the driver only needs a `(coords) -> (energy, gradient)` callable, you
can wrap any vibe-qc potential. For MSINDO, `msindo_force_provider` builds the
callable directly:

```python
from vibeqc.md import run_md, run_nvt
from vibeqc.semiempirical.methods.msindo import msindo_force_provider

provider = msindo_force_provider([8, 1, 1])     # -> fn(coords_bohr)
traj = run_nvt(provider, coords_bohr, atomic_numbers=[8, 1, 1],
               temperature_K=300.0, thermostat="berendsen", n_steps=100)
```

## Well-tempered metadynamics

`vibeqc.metadynamics` layers **well-tempered metadynamics** on the MD driver:
a history-dependent Gaussian bias is deposited on a small set of **collective
variables** (CVs), pushing the system out of free-energy minima it has already
visited so it explores rare events and reconstructs the free-energy surface.

Collective variables are interatomic `DistanceCV(i, j)` and `AngleCV(i, j, k)`
(each supplies its value and the Cartesian gradient ∂s/∂R); the bias is added
to the energy+gradient the MD driver integrates, and hills are deposited on a
fixed stride. In the well-tempered scheme (Barducci-Bussi-Parrinello 2008)
each hill's height is damped by the bias already present, so the bias
converges and the free energy follows as

```
F(s) = −γ / (γ − 1) · V_bias(s) + const          (γ = bias factor)
```

```python
import numpy as np
from vibeqc.metadynamics import DistanceCV, run_metadynamics

# Bias the distance between atoms 0 and 1 (force_fn = any energy+gradient).
res = run_metadynamics(
    force_fn, coords_bohr, [DistanceCV(0, 1)], atomic_numbers=[8, 1, 1],
    bias_factor=8.0, hill_height=8e-4, hill_sigma=0.15,
    deposition_stride=10, n_steps=16000, temperature_K=300.0,
    thermostat="berendsen")

grid = np.linspace(1.9, 4.1, 221)          # CV values to evaluate
fes = res.free_energy(grid)                # reconstructed free energy (min 0)
```

`run_metadynamics` returns a `MetadynamicsResult` with the (biased) MD
`trajectory`, the `cv_trajectory` (CV values per frame), and the accumulated
`bias`; `result.free_energy(grid)` reconstructs the profile and
`result.bias_potential(grid)` returns the raw bias. Metadynamics runs NVT, so
a thermostat is required (default Nosé-Hoover).

On the MSINDO surface use `run_msindo_metadynamics(Z, xyz_angstrom, cvs, ...)`
(`vibeqc.semiempirical.methods.msindo`). Because each MD step costs `6N + 1`
SCFs and filling a basin needs many steps, MSINDO metadynamics is for tiny
systems / short demonstrations; the headline free-energy recovery is validated
on an analytic double well in
`examples/semiempirical/25_msindo_metadynamics.py`.

## Citations

A run cites the velocity-Verlet integrator (Swope et al. 1982) and, when a
thermostat is used, the Berendsen (1984) or Nosé-Hoover (Nosé 1984 + Hoover
1985) papers, in addition to the underlying method (e.g. the MSINDO papers).
A metadynamics run additionally cites metadynamics (Laio-Parrinello 2002) and
the well-tempered variant (Barducci-Bussi-Parrinello 2008).
`run_msindo_md(..., output="stem")` / `run_msindo_metadynamics(..., output=...)`
write the `stem.bibtex` / `stem.references` sidecars; see
[Citations](citations.md).

## See also

* [Geometry optimization](geometry_optimization.md) and [NEB](neb.md), the
  other energy+gradient drivers.
* [Semiempirical methods](semiempirical.md) and the MSINDO guide for the
  underlying engine.
