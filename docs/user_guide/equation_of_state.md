---
myst:
  html_meta:
    "description": "Reference for equation-of-state fitting in vibe-qc: the third-order Birch-Murnaghan fit (vibeqc.eos.fit_birch_murnaghan), its quality gates and the ill_conditioned_fit return, how to build an energy-volume scan with run_periodic_job, unit conversions, the QVF equation_of_state section that vibe-view draws, and the limits of the helper."
    "og:title": "vibe-qc user guide - equation of state (Birch-Murnaghan fits)"
---

# Equation of state: Birch-Murnaghan fits

`vibeqc.eos` fits a third-order Birch-Murnaghan (BM3) equation of state to an
energy-volume series you supply and returns the equilibrium volume, the energy at
the minimum, the bulk modulus and its pressure derivative. It also refuses to
report a minimum when the curve is too flat, or the fit too poor, for the number to
mean anything. The fit is production: it is a small, deterministic, tested helper.
It fits a series that you compute; there is no scan driver (`run_eos_scan` is on
the [roadmap](../roadmap.md), not shipped), no Vinet or Murnaghan form, and no
equation of state from the stress tensor. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

## Quick start

The fit takes volumes in bohr$^3$ and energies in Hartree, one pair per calculation.
First a self-consistency check on synthetic data with known parameters, so you can
see what a good fit looks like (this is not a calculation):

```python
import numpy as np
from vibeqc.eos import birch_murnaghan_energy, fit_birch_murnaghan

HA_PER_BOHR3_TO_GPA = 29421.0157            # 1 Ha/bohr^3 in GPa

# Synthetic E(V) with known parameters and 2 microhartree of noise: a check of the
# fit itself, not a calculation.
e0, v0, b0, b0p = -8.0, 110.0, 0.0020, 4.2   # Ha, bohr^3, Ha/bohr^3, dimensionless
rng = np.random.default_rng(7)
volumes = np.linspace(0.90, 1.10, 7) * v0
energies = birch_murnaghan_energy(volumes, e0, v0, b0, b0p) + 2.0e-6 * rng.standard_normal(7)

fit = fit_birch_murnaghan(volumes, energies)
print(fit["status"], "| minimum bracketed:", fit["minimum_bracketed"])
print(f"V0 = {fit['v0_bohr3']:.3f} bohr^3   "
      f"B0 = {fit['b0_hartree_per_bohr3'] * HA_PER_BOHR3_TO_GPA:.2f} GPa   "
      f"B0' = {fit['b0_prime']:.2f}   E0 = {fit['e0_hartree']:.6f} Ha")
print(f"input: V0 = {v0:.3f}   B0 = {b0 * HA_PER_BOHR3_TO_GPA:.2f} GPa   B0' = {b0p}")
print(f"rmse = {fit['rmse_hartree']:.1e} Ha   span = {fit['energy_span_hartree']:.4f} Ha   "
      f"rmse/span = {fit['rmse_to_span']:.1e}")

# A curve with almost no energy variation is refused, not reported as a minimum.
flat = fit_birch_murnaghan(volumes, 1.0e-9 * np.cos(np.arange(7)))
print(flat["status"], "|", flat["reason"])
```

```text
fit_succeeded | minimum bracketed: True
V0 = 110.012 bohr^3   B0 = 58.92 GPa   B0' = 4.13   E0 = -8.000001 Ha
input: V0 = 110.000   B0 = 58.84 GPa   B0' = 4.2
rmse = 4.5e-07 Ha   span = 0.0013 Ha   rmse/span = 3.4e-04
ill_conditioned_fit | energy_span_below_threshold:1.98999e-09<1e-06; rmse_exceeds_span_fraction:301.345>0.25
```

The volume and energy are recovered to $10^{-4}$ relative; the bulk modulus to 0.15
percent, and $B_0'$ to 0.07, the usual weak determination of the pressure
derivative. The second call is the gate at work.

## A real scan, and what a failure looks like

A scan is a loop over cell volumes with one `run_periodic_job` per point, using the
same basis, k mesh and numerical settings at every volume (and, for cells with free
internal coordinates, relaxed positions at each volume). The loop below scans the
lattice constant of diamond with SCC-DFTB ({vibeqc-cite}`elstner_scc_dftb_1998`),
chosen because a point takes seconds:

```python
import warnings
import numpy as np
import vibeqc as vq
from vibeqc.eos import fit_birch_murnaghan

warnings.simplefilter("ignore")
ANG = 1.8897259886

def diamond(a_ang):                                     # fcc primitive cell, 2 atoms
    a = a_ang * ANG
    lat = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    return vq.PeriodicSystem(3, lat, [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [a / 4, a / 4, a / 4])])

volumes, energies = [], []
for a_ang in np.linspace(3.40, 3.76, 7):                # conventional lattice constant / Angstrom
    r = vq.run_periodic_job(diamond(a_ang), None, method="scc_dftb", kpoints=(3, 3, 3),
                            output=f"c_{a_ang:.3f}")
    assert r.converged
    volumes.append((a_ang * ANG) ** 3 / 4.0)            # primitive-cell volume, bohr^3
    energies.append(r.energy)
    print(f"a = {a_ang:.3f} A   V = {volumes[-1]:8.3f} bohr^3   E = {energies[-1]:.8f} Ha")

fit = fit_birch_murnaghan(volumes, energies)
print(fit["status"], "| minimum bracketed:", fit["minimum_bracketed"])
print(f"V0 = {fit['v0_bohr3']:.2f} bohr^3 (outside {min(volumes):.1f} to {max(volumes):.1f})")
print(f"rmse/span = {fit['rmse_to_span']:.1e}, span = {fit['energy_span_hartree']:.3f} Ha")
```

```text
a = 3.400 A   V =   66.309 bohr^3   E = -4.92247196 Ha
a = 3.460 A   V =   69.882 bohr^3   E = -4.69301269 Ha
a = 3.520 A   V =   73.581 bohr^3   E = -4.49460070 Ha
a = 3.580 A   V =   77.408 bohr^3   E = -4.32303764 Ha
a = 3.640 A   V =   81.366 bohr^3   E = -4.17386462 Ha
a = 3.700 A   V =   85.456 bohr^3   E = -4.04323783 Ha
a = 3.760 A   V =   89.681 bohr^3   E = -3.92827627 Ha
fit_succeeded | minimum bracketed: False
V0 = 59.68 bohr^3 (outside 66.3 to 89.7)
rmse/span = 1.8e-02, span = 0.994 Ha
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) the scan takes 25 to 55 s. It
is a deliberately honest failure. The energy falls monotonically as the cell
shrinks: the in-house SCC-DFTB parameters (a preview route with a placeholder
repulsive, GitLab #306) have no minimum in the physical window, so the fit
extrapolates. `status` is `fit_succeeded` because the residual is small compared with
the large span, but `minimum_bracketed` is `False`, $V_0$ lies outside the sampled
range and the implied bulk modulus is thousands of gigapascals. **The quality gates do
not test whether the minimum is inside the data**; always check `minimum_bracketed`
and that `v0_bohr3` lies within your volumes before you use a result. The data set
that needs this care is exactly the one this module was written for: semiempirical
curves that are nearly flat or monotonic.

For an ab initio scan budget the cost. On the same machine, one Gamma-only LiH
point (rock-salt primitive cell, STO-3G, two atoms) took 864 s with
`jk_method="gdf"` ({vibeqc-cite}`sun_berkelbach_gdf_2017`) and 177 to 270 s with
`jk_method="gpw"` ({vibeqc-cite}`lippert_hutter_gpw_1997`; LDA, 60 to 300 Ha cutoff;
the runner prints a `GAPWExperimentalWarning` once per GPW build). A seven-point scan is
therefore minutes to hours, and is best run as independent jobs (see
[Queue](queue.md)) whose energies you collect into the list passed to the fit. Use
five to nine volumes spanning about $\pm10$ percent around the expected $V_0$.

## Theory

The third-order Birch-Murnaghan equation of state
(F. Birch, *Phys. Rev.* **71**, 809 (1947)) expands the energy in the Eulerian strain
$f=\tfrac12\big[(V_0/V)^{2/3}-1\big]$ to third order. With
$\eta^2=(V_0/V)^{2/3}$ the energy that `birch_murnaghan_energy` evaluates is

$$
E(V)=E_0+\frac{9V_0B_0}{16}\Big\{B_0'\,(\eta^2-1)^3+(\eta^2-1)^2\,(6-4\eta^2)\Big\},
$$

where $E_0$ is the energy at the equilibrium volume $V_0$, $B_0=-V\,\partial P/\partial V|_{V_0}=V_0\,\partial^2E/\partial V^2|_{V_0}$
is the bulk modulus and $B_0'=\partial B/\partial P|_{P=0}$ its pressure derivative. The
pressure follows from $P=-\partial E/\partial V$,

$$
P(V)=\frac{3B_0}{2}\Big[\Big(\frac{V_0}{V}\Big)^{7/3}-\Big(\frac{V_0}{V}\Big)^{5/3}\Big]
\Big\{1+\tfrac34\,(B_0'-4)\Big[\Big(\frac{V_0}{V}\Big)^{2/3}-1\Big]\Big\}.
$$

Four parameters $(E_0,V_0,B_0,B_0')$ are fitted to the $N$ sampled points by bounded
nonlinear least squares (`scipy.optimize.curve_fit`; SciPy is a hard dependency of
vibe-qc). The bounds are $V_0\in[0.9\,V_\text{min},\,1.1\,V_\text{max}]$,
$B_0>0$ and $B_0'\in[0,12]$, and the start is the lowest-energy point with
$B_0=10^{-3}$ Ha/bohr$^3$ and $B_0'=4$. Atomic units convert with
1 Ha/bohr$^3$ = 29421.0157 GPa. The approximation is the truncation: BM3 describes
moderate compression well, and $B_0'$ is only weakly
constrained by data within a few percent of $V_0$. With exactly four points the system is exactly
determined and the residual is zero whatever the data, so use at least six.

**Why a nearly flat curve is refused.** The curvature at the minimum is
$\partial^2E/\partial V^2=B_0/V_0$, so a curve whose total energy variation is tiny has a
tiny $B_0$, and a four-parameter model can place an interior stationary point almost
anywhere in it. The energy noise (SCF thresholds, grid and integral cutoffs) then
determines $V_0$ and $B_0$, not the physics. Flat curves are typical of very weakly
bound systems and of semiempirical methods for them (the rare-gas SCC-DFTB curves that
motivated the module). Two gates catch this and return `ill_conditioned_fit` with a
`reason` string:

- `min_energy_span_hartree` (default $10^{-6}$ Ha): the raw energy span
  $\max E-\min E$ of the data must exceed this.
- `max_rmse_to_span` (default 0.25): the fit's RMS residual divided by that span must not
  exceed this; a residual that is a material fraction of the total signal means the
  parameters are descriptive only.

Where the method helps and where it fails: a smooth, well-bracketed scan of a bound
solid gives $V_0$ to a fraction of a percent and $B_0$ to a few percent, limited by
the electronic-structure method, the basis and the k mesh, far more than by the
form. It cannot describe a phase transition inside the window, a nonmonotonic pressure
response, or a scan whose points differ in more than the volume.

## Options

`fit_birch_murnaghan(volumes, energies, *, min_energy_span_hartree=1.0e-6, max_rmse_to_span=0.25) -> dict`

| Argument | Meaning |
|---|---|
| `volumes` | One-dimensional finite positive volumes, bohr$^3$, at least two distinct values. |
| `energies` | Matching total energies, Hartree. |
| `min_energy_span_hartree` | Smallest raw energy span accepted; non-negative. |
| `max_rmse_to_span` | Largest RMSE-to-span ratio accepted; finite and positive. |

The returned dict has these keys on success (`status == "fit_succeeded"`): `status`,
`model` (`vibeqc.eos.BM3_MODEL == "third_order_birch_murnaghan"`), `e0_hartree`,
`v0_bohr3`, `b0_hartree_per_bohr3`, `b0_prime`, `rmse_hartree`, `energy_span_hartree`,
`rmse_to_span` and `minimum_bracketed` (whether $V_0$ lies within the sampled range).
`ill_conditioned_fit` (`vibeqc.eos.ILL_CONDITIONED_FIT`) has the same keys plus a
`reason`; if the optimiser itself fails the dict contains only `status`, `model`,
`energy_span_hartree`, `reason` (`fit_failed: <ExceptionType>: <message>`) and
`minimum_bracketed=False`. `birch_murnaghan_energy(volume, e0, v0, b0, b0_prime)`
evaluates the curve for scalars or arrays. The module is imported as `vibeqc.eos`;
nothing is re-exported at the top level.

## Output

The fit returns a dict and writes nothing, and no runner keyword produces an
equation-of-state file. To view a fit in vibe-view, pass the data to the low-level QVF
writer: `vibeqc.output.formats.qvf.write_qvf(..., eos_data=...)` writes an
`equation_of_state` section (QVF format section 4.14, used in
`examples/vibe_view/showcase_qvf_all_sections.py`). The dict carries `volumes` and
`energies` in Å$^3$ and eV and a `fit` dict with `model`, `V0`, `E0`, `B0` (GPa),
`B0_prime` and optionally `energy_unit`, `volume_unit`, `pressure_unit`, `residual_rms`
and `pressures_gpa`. The conversion from a `fit_birch_murnaghan` result is:

```python
BOHR_TO_ANG, HA_TO_EV = 0.529177210903, 27.211386245988
eos_data = {
    "volumes": np.asarray(volumes) * BOHR_TO_ANG ** 3,          # Angstrom^3
    "energies": np.asarray(energies) * HA_TO_EV,                # eV
    "fit": {
        "model": "birch_murnaghan",          # the tag the shipped showcase uses, not BM3_MODEL
        "V0": fit["v0_bohr3"] * BOHR_TO_ANG ** 3, "E0": fit["e0_hartree"] * HA_TO_EV,
        "B0": fit["b0_hartree_per_bohr3"] * 29421.0157, "B0_prime": fit["b0_prime"],
        "energy_unit": "eV", "volume_unit": "angstrom^3", "pressure_unit": "GPa",
        "residual_rms": fit["rmse_hartree"] * HA_TO_EV,
    },
}
```

vibe-view draws the sampled points as markers, overlays the fitted curve evaluated
from `V0`, `E0`, `B0` and `B0'`, marks $V_0$ and annotates the bulk modulus, its
derivative and the model reference. Write the section only for a fit whose
`status` is `fit_succeeded` and whose minimum is bracketed.

## Limits and gates

- **Model**: BM3 only. There is no Vinet, Murnaghan, Poirier-Tarantola or
  polynomial form and no `EOSFit` class; the fit returns a plain dict.
- **Inputs** (`ValueError`): fewer than four points ("requires at least four
  points"), non-1-D or differently shaped arrays, non-finite values, non-positive
  volumes, all volumes equal, negative `min_energy_span_hartree`, non-finite or
  non-positive `max_rmse_to_span`.
- **Quality**: the two gates above return `ill_conditioned_fit`, not an exception; test
  `fit["status"]`. They do not check that the minimum is bracketed or that the
  parameters are physical.
- **Scan**: the fit runs no calculations. Keep basis, k mesh, Coulomb route and cutoffs
  fixed across the series; a series that mixes settings has a step where the settings
  change and a poor `rmse_to_span`.
- **Thermal effects**: static lattice only. There is no quasi-harmonic free energy or
  thermal equation of state (planned with phonons, see the roadmap).
- **Stress**: the periodic stress is finite-difference only and `optimize_cell=True`
  fails closed on GPW, GAPW, GDF and BIPOLE, so a pressure-based route to $V_0$ is not
  available; the volume scan is the route.

## See also

- [Periodic methods](periodic_methods.md) and [k-points](k_points.md): choosing the
  route and mesh for the scan, and [Periodic systems](periodic_systems.md) for cell
  construction.
- [Vibrational analysis](vibrational_analysis.md): the Gamma-point phonons that a
  quasi-harmonic treatment would combine with the fit.
- [Cohesive energies](cohesive_energies.md), [Queue](queue.md).
- [QVF and vibe-view](../visualization.md) for the `equation_of_state` panel, and the
  [roadmap](../roadmap.md) for the planned scan driver and further models.
- [Methods: theory and status](../methods.md) and the [feature matrix](../features.md).
