# Smearing (fractional occupations for periodic SCF)

> **TL;DR.** For a **metallic** periodic system (Fermi level cuts
> through bands), turn smearing on, set
> `opts.smearing_temperature = 0.005` (Hartree, ~1580 K), or pass
> `smearing_temperature="metal"` to `vq.run_periodic_job(...)`.
> For a **clear-gap insulator**, leave smearing off, the default of
> `0.0` is correct. **Smearing is *not* a remedy for an oscillating
> or diverging SCF**, if your SCF runs cleanly with smearing on but
> diverges with it off, you are masking a backend bug, not solving a
> convergence problem (see [§ When smearing is the wrong knob](#when-smearing-is-the-wrong-knob)).

Smearing replaces the integer step occupation
$n_i \in \{0, 2\}$ at the Fermi level with a smooth distribution
that lets fractional occupations relax around $\mu$. Mathematically
this regularises the Brillouin-zone integral
$\sum_{k\in\text{BZ}} \theta(\mu - \varepsilon_i(k))$ that is ill-
defined when any band crosses $\mu$, i.e., for every metal. The
trade is a finite electronic temperature or broadening width $T$:
the SCF converges to $A = E - TS$ instead of the strict ground-state
energy $E_0$. For Fermi-Dirac this is Mermin's physical finite-
temperature free energy (Mermin 1965); Methfessel-Paxton and
Marzari-Vanderbilt use the corresponding stationary generalized
free-energy functional.

For metals the smearing energy $TS$ is **necessary** to make the
band integral well-defined at a finite k-mesh; the price is that
you must either pick $T$ small enough that $TS$ is negligible at
your accuracy target, or extrapolate $T \to 0$ from a ladder of
runs. The recommended workflow per flavor is in
[§ Choosing T](#choosing-t).

## Quick start

### High-level wrapper

```python
import vibeqc as vq

# "metal" preset → k_B T = 0.005 Ha (~0.136 eV, ~1580 K).
vq.run_periodic_job(
    system, basis,
    method="rks", functional="pbe",
    smearing_temperature="metal",
    kpoints=(4, 4, 4),
)
```

### Direct driver call

```python
opts = vq.PeriodicSCFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.smearing_temperature = 0.005   # k_B T in Hartree

result = vq.run_rhf_periodic_multi_k_ewald3d(
    system, basis, kmesh=(4, 4, 4), options=opts,
)

print(f"Energy        = {result.energy:.8f} Ha")
print(f"Free energy   = {result.free_energy:.8f} Ha   (Mermin A = E - TS)")
print(f"Entropy S/k_B = {result.entropy:.6f}")
print(f"Fermi level μ = {result.fermi_level:.6f} Ha")
print(f"Occupations   = {result.occupations}")  # list[ndarray], per k-point
```

The SCF converges Mermin's free energy $A$, **not** $E$. Both are
on the result struct, use $A$ for variational comparisons (e.g.,
forces, lattice optimisations under fixed $T$); use $E$ for "what
would the strict ground-state energy be at this geometry" only
after T → 0 extrapolation.

Since 2026-07-30 the GDF analytic gradients (Γ open-shell
`run_pbc_gdf_uhf`/`run_pbc_gdf_uks` and the multi-k
`run_krhf/krks/kuhf/kuks_periodic_gdf` drivers) accept Fermi-Dirac
smeared runs: `compute_gradient=True` returns exactly $dA/dR$ at
fixed $(T, N)$: by Mermin's variational principle the occupation-
and $\mu$-response terms vanish at self-consistency, so the forces
are the Hellmann-Feynman + Pulay assembly at the
fractional-occupation density $D = \sum_i f_i c_i c_i^\dagger$ and
energy-weighted density $W = \sum_i f_i \epsilon_i c_i c_i^\dagger$.
The present analytic-gradient implementation is Fermi-Dirac-only.
Methfessel-Paxton and Marzari-Vanderbilt now report generalized
entropies conjugate to their occupation functions, but their GDF
force paths still need separate full-SCF validation; MP occupations
can also be negative, which the current square-root fractional-block
assembly cannot represent. See [`periodic_methods.md`](periodic_methods.md)
for the gradient envelope and the Fermi-Dirac gates.

## Input forms

`smearing_temperature` on the high-level wrapper
(`vq.run_periodic_job`) accepts any of:

| Form | Example | Notes |
|---|---|---|
| Numeric width in Hartree (canonical) | `0.005` | Same units as the on-options-struct field |
| Named preset (string) | `"metal"`, `"small-gap"`, `"insulator"`, `"debug"`, `"off"` | See `vq.SMEARING_PRESETS` |
| Numeric string with unit | `"1000 K"`, `"0.1 eV"`, `"0.02 Ha"`, `"0.2 Ry"` | Resolved via `vq.resolve_smearing_temperature` |
| `"auto"` + hints | `smearing_temperature="auto", smearing_metallic=True` | Cautious, defaults to 0.0 if no hint |
| `None` / `"none"` / `"off"` | `None` | Disables smearing |

The on-options-struct field
`PeriodicRHFOptions.smearing_temperature` (and siblings on
`PeriodicSCFOptions` / `PeriodicKSOptions`) is **always** the
canonical electronic temperature $k_B T$ in **Hartree**, no
implicit unit conversion happens there.

`run_periodic_job` also accepts the pre-v0.10 migration surface:

```python
vq.run_periodic_job(
    system, basis,
    method="rks", functional="pbe",
    smearing=vq.SmearingOptions.from_user("metal"),
)
```

### Common conversions

| Macroscopic value | $k_B T$ in Hartree |
|---|---|
| 100 K | $3.17 \times 10^{-4}$ |
| 300 K (room temperature) | $9.50 \times 10^{-4}$ |
| 1000 K | $3.17 \times 10^{-3}$ |
| 1580 K (the `"metal"` preset) | $5.00 \times 10^{-3}$ |
| 3000 K | $9.50 \times 10^{-3}$ |
| 0.1 eV | $3.67 \times 10^{-3}$ |
| 0.01 Ry | $5.00 \times 10^{-3}$ |

Helpers:
```python
vq.kelvin_to_hartree_temperature(300.0)         # → 9.5e-4
vq.electronvolt_to_hartree_temperature(0.1)     # → 3.67e-3
vq.rydberg_to_hartree_temperature(0.01)         # → 5.0e-3
vq.hartree_to_kelvin_temperature(0.005)         # → 1579.3
```

## Choosing T

The Mermin free-energy formalism is exact in $T$, but **only the
$T \to 0$ limit gives the strict ground-state energy**. Finite $T$
adds a $TS$ shift that for Fermi-Dirac smearing decays only linearly
in $T$ as $T \to 0$ (Marzari et al. 1999), so the strategy is one
of:

* **Pick $T$ small enough that the $TS$ correction is below your
  accuracy target.** For ~mHa/atom accuracy on typical
  3d / 4d transition metals at room temperature, the `"metal"`
  preset ($k_B T = 5 \times 10^{-3}$ Ha) sits at the upper edge of
  the "small" range. Tighter targets need $T \sim 10^{-3}$ Ha
  (~300 K), which slows convergence but keeps the entropy
  correction near µHa.
* **Extrapolate $T \to 0$ from a ladder of runs.** Two to three
  $T$ values plus a polynomial fit in the asymptotic regime gives
  a defensible ground-state estimate. The order of the
  extrapolation depends on flavor (linear for Fermi-Dirac,
  quadratic for Methfessel-Paxton, cubic for Marzari-Vanderbilt
  cold). A user-facing helper (`vq.extrapolate_t_zero`) is queued
  for a later milestone, until then, fit manually.

### Recommended starting points

| System type | Suggested `smearing_temperature` | Why |
|---|---|---|
| Clear-gap insulator (gap ≳ 1 eV) | `0.0` (default) | Integer Aufbau is exact; smearing only adds noise |
| Small-gap semiconductor (gap ~0.1 eV) | `"small-gap"` ($2 \times 10^{-3}$ Ha) | Avoid integer-occupation jitter when the gap is comparable to k-mesh resolution |
| Metal | `"metal"` ($5 \times 10^{-3}$ Ha) | Standard for converged metallic SCF |
| Convergence debugging | `"debug"` ($10^{-2}$ Ha) | Useful to confirm a hard SCF *can* converge at high T; **not** for production runs (see [§ When smearing is the wrong knob](#when-smearing-is-the-wrong-knob)) |

## Output surface

When `smearing_temperature > 0`, the SCF result struct populates:

| Field | Meaning |
|---|---|
| `result.energy` | Strict electronic energy $E$ at the converged density (no $TS$ subtraction) |
| `result.free_energy` | Mermin (Fermi-Dirac) or generalized (MP/MV) free energy $A = E - TS$, the **variational** quantity under smearing |
| `result.entropy` | Dimensionless electronic (Fermi-Dirac) or generalized (MP/MV) entropy $S/k_B$ per unit cell |
| `result.fermi_level` | Chemical potential $\mu$ in Hartree |
| `result.occupations` | List of per-$k$ occupation arrays; nominally in $[0, 2]$ for closed-shell, with generalized MP/MV tails able to overshoot |
| `result.smearing_temperature` | Echo of the temperature actually used |

For multi-k GDF KRHF/KRKS and KUHF/KUKS, the reported Fermi-Dirac entropy
comes from the natural occupations of the accepted density in the retained
orthonormal AO space. This includes densities mixed during SCF. The energy,
entropy and free energy therefore describe the same accepted density, even
when the iteration limit stops the calculation before convergence. If an
atomic or restart guess has natural occupations outside the Fermi interval,
the driver first builds that selected guess's Fock and refills it with
Fermi occupations. This seed step precedes the counted SCF iterations;
`max_iter=0` still returns the selected initial guess without this step.

When `smearing_temperature == 0`, the smearing fields collapse:
`free_energy == energy` and `entropy == 0`. On **every closed-shell multi-k
route**, which dispatches through `vibeqc.smearing.apply_smearing`, Aufbau
uses one Fermi level across the full Brillouin zone. A gapped insulator
therefore retains the usual per-k hard-Aufbau
`{0, 2}` arrays and conventional midgap value. If bands overlap between k
points, the number of occupied bands may differ by k point; states that are
equal at the Fermi boundary to roundoff receive the same fractional occupation
so mesh ordering cannot break their symmetry. This is a sharp, zero-temperature
ensemble, not positive-temperature smearing. Analytic GDF gradients currently
require either the gapped integer pattern or supported positive-temperature
Fermi-Dirac occupations; a band-overlap T=0 ensemble fails closed for
`compute_gradient=True`.

A Γ-only calculation is a single spectrum, so there is nothing for a shared
chemical potential to reorder; it keeps the historical hard-Aufbau path
unchanged. Open-shell (UHF/UKS) `T = 0` is still a per-spin, per-k Aufbau and
has **not** moved to the global fill; that remains an open ask recorded on
GitLab #85.

Not every route can yet *represent* a band-overlap `T = 0` state. The Ewald-3D
multi-k (`run_rhf_periodic_multi_k_ewald3d`,
`run_rks_periodic_multi_k_ewald3d`) and BIPOLE RKS multi-k
(`run_pbc_bipole_rks`) drivers build their zero-temperature density from a
single fixed occupied subspace `C[:, :n_occ]` at every k, which cannot express
a differing per-k occupied count. Rather than converge one density and report
the band edges of another, those routes now **fail closed** with a
`NotImplementedError` naming the per-k occupied counts and pointing at
`smearing_temperature > 0`, `bz_integration="gilat"`, or the multi-k GDF route,
which builds its density from the occupations directly. Giving them the same
density build is tracked as GitLab #509. Gapped meshes never reach this guard.

Why it matters: filling the lowest `n_occ` bands at each k point separately
occupies a state above the Fermi level at one k while leaving a state below it
empty at another. The band-extrema block then reports a **negative** indirect
gap, because the valence-band maximum it finds genuinely does sit above the
state it has been told is the conduction-band minimum. That is a wrong answer,
not a convergence complaint (GitLab #85).

## Per-backend availability

Smearing is being rolled out backend-by-backend as the v0.10.x
smearing program (see [`docs/design_smearing.md`](../design_smearing.md)
for the full design and milestone plan). The state as shipped on
the current `main`:

| Backend | Spin | k-mesh | Fermi-Dirac smearing | Notes |
|---|---|---|---|---|
| Ewald-3D | RHF / RKS | multi-k | ✅ | `run_rhf_periodic_multi_k_ewald3d`, `run_rks_periodic_multi_k_ewald3d` |
| Ewald-3D (Python Γ-dispatch) | RHF / RKS | Γ-only | ✅ (inherits via `n_kpts=1`) | `run_rhf_periodic_gamma_scf`, `run_rks_periodic_gamma_scf` |
| Ewald-3D (C++-direct Γ) | RHF / RKS | Γ-only | ❌ (queued M2) | `run_rhf_periodic_gamma_ewald3d`, `run_rks_periodic_gamma_ewald3d` |
| Ewald-3D | UHF / UKS | any | ❌ (queued M3) | Driver raises on `smearing_temperature > 0` today |
| GDF | RHF / RKS | multi-k | ✅ | `run_krhf_periodic_gdf`, `run_krks_periodic_gdf` |
| GDF | RHF / RKS | Γ-only | ✅ | `run_rhf_periodic_gamma_gdf` |
| GDF | UHF / UKS | any | ✅ | `run_pbc_gdf_uhf`, `run_pbc_gdf_uks` (Γ), `run_kuhf_periodic_gdf`, `run_kuks_periodic_gdf` (multi-k); per-spin $\mu_\alpha$/$\mu_\beta$; analytic $dA/dR$ gradients since 2026-07-30 |
| BIPOLE | RKS / UHF / UKS | Γ-only + multi-k | ✅ | `run_pbc_bipole_rks`, `run_pbc_bipole_uhf`, `run_pbc_bipole_uks`; analytic KS gradients reject smeared results, so production forces use FD for forces |
| BIPOLE | RHF | any | ❌ | Closed-shell HF BIPOLE keeps integer occupations today |
| GAPW | - | - | n/a (driver not yet a thing, queued M8) | |

### Choosing a smearing flavor

`run_periodic_job` accepts `smearing_method=` with four implemented
flavors, all dispatched through `vibeqc.smearing.apply_smearing`:

| flavor | occupation | notes |
|---|---|---|
| `"fermi-dirac"` | $1/(1+e^x)$ | default; physical Fermi-Dirac occupation |
| `"mermin"` | $1/(1+e^x)$ | Mermin finite-temperature free-energy framing ($A = E - TS$); same occupation shape as Fermi-Dirac, distinct citation route; `"meremin"` is accepted as a spelling alias |
| `"methfessel-paxton"` | Hermite-polynomial cold smearing (orders 1 and 2 via `mp_order=`) | accelerates k-point convergence for metals; occupations can go negative |
| `"marzari-vanderbilt"` | first-order cold smearing | minimizes Fermi-surface broadening; zero generalized entropy at $\mu$ |

The free-energy correction $-TS$ is computed from the flavor's own
generalized entropy, so `result.free_energy = result.energy - T S` is
variational for every flavor. Analytic GDF gradients currently accept
Fermi-Dirac and Mermin (both strictly positive occupations); the cold
flavors fail closed pending their own force validation.

For molecular single-point runs, `run_job` accepts
`smearing_temperature=` and `smearing_method=` and applies the same
free-energy correction post-SCF from the converged eigenvalues (for
gapped molecules the density change is negligible).

## When smearing is the wrong knob

**Smearing fixes integer-occupation discontinuities at metallic
Fermi surfaces. It does NOT fix:**

* a wrong Hartree term (e.g., a Madelung self-image leak, a gauge
  bug in the lattice sum);
* a wrong exchange term (a wrong sign in the Coulomb integral, a
  divergent omega in the Ewald split);
* a wrong nuclear-attraction term;
* an over-tight DIIS subspace that's amplifying noise.

If your periodic SCF oscillates with smearing **off** but converges
cleanly with smearing **on**, the SCF is probably converging to a
non-physical stationary point that happens to be smooth under
fractional occupations, the underlying bug is masked, not fixed.
File it against the backend chat (CLAUDE.md § 7).

**Occupation self-consistency is part of GDF convergence.** A periodic GDF
SCF with global occupations (multi-k T=0 Aufbau or finite-temperature
smearing; Γ open-shell smearing) only reports
`converged=True` once the stored occupations are the Fermi filling of
the converged Fock's own eigenvalues, within
`max(sqrt(conv_tol_energy), 1e-9)` per occupation
(`vibeqc.smearing.smeared_occupation_selfconsistency_tolerance`). The
energy and commutator tests alone cannot see a frozen-occupation fixed
point on symmetry-locked cells whose `[F, D]` commutator vanishes
identically (e.g. a symmetric H2-in-a-box fixture, where DIIS receives
zero error vectors from iteration 1). At T=0 the same check prevents a
globally misoccupied band-overlap state from passing merely because each
per-k density commutes with its local Fock. The SCF accelerators also skip
extrapolation entirely when the commutator error is at numerical
noise, so such cells iterate plainly to the true occupation fixed
point instead of terminating early on a stale extrapolated Fock
(fixed 2026-08-01; regression:
`tests/test_periodic_gdf_smearing_diis.py`).

The high-level `run_periodic_job` wrapper now emits a warning before it
applies a positive smearing temperature to a system identified as an ionic
or covalent insulator, or to a run with an explicitly supplied positive band
gap. The automatic convergence strategy never enables smearing for those
profiles. The warning is also written into the `.out` convergence-strategy
block so a batch calculation cannot hide the choice.

The canonical cautionary tale: the v0.7.0 Madelung self-image leak,
which over-bound H₂/STO-3G in a 30-bohr box by ~0.587 Ha (a
Madelung-scale shift). Smearing made the symptom *look* like a
convergence issue. The actual fix was in the gauge of the
periodic Hartree term, once that landed (in *Löwdin's Compass*),
the SCF converged without any smearing tweaks.

## Parameter-free alternative: the Gilat-Raubenheimer net

Smearing needs a width $T$ that you converge / extrapolate to $T \to 0$.
The **Gilat-Raubenheimer net** (`bz_integration="gilat"` on the multi-k
Ewald drivers) instead integrates the Brillouin zone analytically per
microcell, the CRYSTAL `SHRINK IS ISP` second-net analogue, with **no width
to converge**. It lives in `vibeqc.bz_integration`.

* **Insulators / gapped systems**, GR runs self-consistently (RHF / RKS /
  UKS, full or symmetry-reduced meshes). For a gap its occupations are
  integer, so it reproduces the Aufbau result, the value is the
  parameter-free machinery, not a different energy.
* **Metals, post-SCF only.** GR *cannot drive* a metallic SCF: the sharp
  $T=0$ Fermi surface makes the occupation-versus-density map
  **discontinuous**, so the SCF oscillates and *no* density mixer (linear
  damping, DIIS, or Anderson) converges it. **Drive the metal SCF with
  smearing, then evaluate the GR Fermi level / occupations / DOS on the
  converged eigenvalues**, the standard tetrahedron/Gilat usage:

  ```python
  import numpy as np
  from vibeqc.bz_integration import (
      gilat_occupations_on_kmesh, fractional_kpoints,
  )

  # 1. Converge the metal with smearing (a smooth map - robust).
  opts.smearing_temperature = 0.005
  r = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts)

  # 2. Parameter-free GR Fermi level + occupations on the converged spectrum.
  #    (Full mesh; for an IBZ mesh expand first via expand_ibz_eigenvalues.)
  frac = fractional_kpoints(sysp.reciprocal_lattice(),
                            np.asarray(kmesh.kpoints))
  mesh = tuple(int(x) for x in kmesh.mesh)
  occ, e_fermi = gilat_occupations_on_kmesh(
      frac, mesh, r.mo_energies, sysp.n_electrons())
  # DOS: scatter the spectrum with eigenvalues_to_full_grid, then gilat_dos.
  ```

The metal *total energy* still comes from the smearing run (its $T \to 0$
limit is well-defined); GR supplies the parameter-free spectral quantities.
Reference: Gilat & Raubenheimer, *Phys. Rev.* **144**, 390 (1966).

## Theory references

* **Mermin, N. D.** *Thermal properties of the inhomogeneous
  electron gas.* Phys. Rev. **137**, A1441 (1965)., The
  free-energy formalism $A = E - TS$ vibe-qc uses for all flavors.
* **Methfessel, M. & Paxton, A. T.** *High-precision sampling for
  Brillouin-zone integration in metals.* Phys. Rev. B **40**, 3616
  (1989)., MP smearing (M6).
* **Marzari, N., Vanderbilt, D., De Vita, A. & Payne, M. C.**
  *Thermal contraction and disordering of the Al(110) surface.*
  Phys. Rev. Lett. **82**, 3296 (1999)., Cold smearing (M7) and
  the original $T \to 0$ extrapolation analysis.
* **dos Santos, F. P. & Marzari, N.** *Fermi-surface effects and
  the electronic entropy of crystals at finite temperature.*
  Phys. Rev. B **107**, 195122 (2023)., Modern review of
  smearing-induced error and the practical $T \to 0$
  extrapolation ladder.

## See also

* [Metallic BZ integration](metallic_bz_integration.md), the
  Gilat-Raubenheimer net in full: parameter-free DOS / Fermi level,
  the post-SCF metal workflow, and the API.
* [SCF convergence](scf_convergence.md), when smearing composes
  with damping / DIIS / level shift / Newton.
* [k-points](k_points.md), k-mesh density is the dual knob to
  smearing for metals (denser mesh → smaller $T$ acceptable).
* [Moving from CRYSTAL](from_crystal.md), the `SHRINK IS ISP` →
  Gilat-net (`bz_integration="gilat"`) mapping.
* [`docs/design_smearing.md`](../design_smearing.md), the
  authoritative design contract, milestone plan, and CP2K parity
  target list for the v0.10.x smearing program.
