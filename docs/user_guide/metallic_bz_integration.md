---
myst:
  html_meta:
    "description": "The Gilat-Raubenheimer net (bz_integration='gilat'), parameter-free Brillouin-zone integration for periodic metals in vibe-qc: self-consistent for insulators, post-SCF Fermi level / occupations / DOS for metals."
    "og:title": "Metallic BZ integration: the Gilat-Raubenheimer net"
---

# Metallic BZ integration: the Gilat-Raubenheimer net

The **Gilat-Raubenheimer (GR) net** integrates the Brillouin-zone occupation
analytically per microcell, the integrator behind CRYSTAL's `SHRINK IS ISP`
second net. Unlike [smearing](smearing.md) it carries **no width parameter** to
converge or extrapolate. It lives in `vibeqc.bz_integration`, where the
occupation kernel itself is engine-agnostic.

```{admonition} Status: production on BIPOLE, GDF and RIJCOSX; gated on GPW and GAPW
:class: caution

On `run_periodic_job` the flag is honoured by **`bipole`, `gdf` and
`rijcosx`**. On **`gpw` and `gapw`** it is recognised but fails closed
(since 2026-08-02) with `NotImplementedError: run_periodic_job:
bz_integration='gilat' is not implemented ...`: those drivers do not thread
the flag into their occupation logic, and before the gate they accepted it,
ran Fermi-Dirac, and still cited Gilat-Raubenheimer in the `.references`
sidecar. The missing term is the GR occupation step inside the GPW/GAPW SCF
loops. Use `jk_method="gdf"`, `"bipole"` or `"rijcosx"` for GR, or drop
`bz_integration` on the GPW/GAPW routes. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

GR and smearing are the two ways vibe-qc resolves the metallic Fermi-surface
integration, and they are complementary:

| | Smearing (Fermi-Dirac / MP / cold) | Gilat-Raubenheimer net |
|---|---|---|
| Parameter | width *T* (converge / extrapolate *T* → 0) | none |
| Drives a metal SCF? | **yes** (smooth occupation map) | **no**, post-SCF only |
| Best for | converging the SCF; forces | parameter-free DOS / Fermi level / final integration |

## Theory

The Brillouin-zone integral behind every occupation, Fermi level and density
of states is, per band $n$ and with spin degeneracy $g$,

$$
N = g\sum_n \int_{\mathrm{BZ}} \frac{V\,\mathrm d^3k}{(2\pi)^3}\,
\theta\!\left(E_F - \varepsilon_n(\mathbf k)\right),
\qquad
D(E) = g\sum_n \int_{\mathrm{BZ}} \frac{V\,\mathrm d^3k}{(2\pi)^3}\,
\delta\!\left(E - \varepsilon_n(\mathbf k)\right)
= \frac{gV}{(2\pi)^3}\sum_n \int_{S_n(E)} \frac{\mathrm dS}{|\nabla_{\mathbf k}\varepsilon_n|},
$$

where $S_n(E)$ is the constant-energy surface of band $n$ and $V$ the cell
volume. On a finite mesh the step function is the problem: counting each
mesh point as fully occupied or empty makes $N(E_F)$ a staircase, and the
Fermi surface is resolved only as well as the mesh spacing.

The Gilat-Raubenheimer net ({vibeqc-cite}`gilat_raubenheimer_1966`,
{vibeqc-cite}`gilat_spectral_1972`) partitions the zone into one
parallelepiped microcell per regular-mesh point $\mathbf k_0$ and linearises
each band inside its cell,

$$
\varepsilon_n(\mathbf k_0 + \delta\mathbf k) \approx \varepsilon_n(\mathbf k_0)
+ \mathbf v_n(\mathbf k_0)\cdot\delta\mathbf k,
\qquad
\varepsilon - \varepsilon_0 = \sum_{i=1}^{3} g_i\,t_i,
\qquad
g_i = \mathbf v_n\cdot\frac{\mathbf b_i}{N_i},
$$

with $t_i \in [-\tfrac12, \tfrac12]$ the offset in the cell's fractional
coordinates, $\mathbf b_i$ the reciprocal-lattice vectors and $N_i$ the mesh
sizes. vibe-qc takes each directional slope $g_i$ as the periodic central
difference of the eigenvalue grid along mesh axis $i$
(`grid_band_slopes`), so the eigenvalues alone fix the geometry and a
non-orthogonal reciprocal cell needs no special treatment: the cell Jacobian
is constant and cancels in the occupied *fraction*. That fraction is the
volume of the unit box below a plane,

$$
f_{n\mathbf k_0}(E_F) = \mathrm{Vol}\Big\{\,t\in[-\tfrac12,\tfrac12]^3 :
\sum_i g_i t_i < E_F - \varepsilon_n(\mathbf k_0)\Big\}
= \frac{1}{6A_1A_2A_3}\sum_{S\subseteq\{1,2,3\}} (-1)^{|S|}\,
\mathrm{relu}\Big(y - \sum_{i\in S}A_i\Big)^{3},
$$

with $A_i = |g_i|$ and $y = E_F - \varepsilon_0 + \tfrac12(A_1 + A_2 + A_3)$,
the cumulative distribution of a weighted sum of three uniform variables
(`occupied_fraction`; directions with a negligible slope fall back to the
two-, one- or zero-dimensional box formula to avoid cancellation). The
occupation at the grid point is $g f_{n\mathbf k_0}$, uniform mesh weights
$1/N_k$ reproduce the zone integral, and $E_F$ is found by bisecting
$N(E_F) = g\sum_{n,\mathbf k_0} f_{n\mathbf k_0}/N_k$ to the electron count
(`gilat_raubenheimer_occupations`). The density of states is the energy
derivative of the same object, $D(E) = g\sum_{n,\mathbf k_0}
\partial f_{n\mathbf k_0}(E)/\partial E\,/N_k$: for a linear band the area of
the plane $\varepsilon = E$ inside the microcell divided by $|\nabla_{\mathbf
k}\varepsilon_n|$, which is exactly the Fermi-surface area weight in the
surface integral above, evaluated analytically per cell (`gilat_dos`).

Smearing answers the same integral by replacing the step with a smooth
function, $f = [1 + e^{(\varepsilon-\mu)/T}]^{-1}$ for Fermi-Dirac, and by
minimising the Mermin free energy $F = E - TS$ instead of $E$; it introduces a
width $T$ that has to be converged or extrapolated, and it produces an
entropy. GR introduces neither: it is a $T = 0$ quadrature whose only
approximation is the linear band inside each microcell, so its error falls
with the mesh spacing rather than with a broadening parameter. The price is
the one stated below: the resulting occupation map is discontinuous in the
density, so it can analyse a converged metal but cannot drive its SCF.
Which approximation enters, then, is the piecewise-linear band; where it
helps is the parameter-free Fermi level, occupations and DOS of a converged
spectrum, and where it fails is inside the metallic SCF loop and in force
expressions, which need the smooth occupations of smearing.

## Quick start

Pass `bz_integration="gilat"` to a multi-k Ewald driver:

```python
import vibeqc as vq

r = vq.run_rks_periodic_multi_k_ewald3d(
    system, basis, kmesh, opts, bz_integration="gilat")
```

It is available on `run_rhf_periodic_multi_k_ewald3d`,
`run_rks_periodic_multi_k_ewald3d`, `run_uhf_periodic_multi_k_ewald3d`, and
`run_uks_periodic_multi_k_ewald3d`. UHF and UKS integrate alpha and beta
occupations independently at their fixed spin populations. The default
(`None` / `"smearing"`) path is unchanged. GR is a *T* = 0 integrator, so it
is mutually exclusive with `smearing_temperature > 0`.

The public runner also exposes the supported BIPOLE routes:

```python
r = vq.run_periodic_job(
    system,
    basis,
    method="UHF",              # or RKS / UKS
    jk_method="bipole",
    kpoints=(4, 4, 4),
    bz_integration="gilat",
)
```

BIPOLE UHF and UKS use the same independent per-spin integral described
above. BIPOLE RHF, ROHF, and ROKS remain fail closed for this selector.

## When to use GR

### Insulators / gapped systems, self-consistent

For a gapped system GR runs **self-consistently** (RHF / RKS / UHF / UKS, full
or symmetry-reduced meshes). Its occupations are integer across the gap, so it
reproduces the Aufbau result. The value is the parameter-free machinery and a
clean Fermi level / DOS, not a different total energy.

### Metals, post-SCF only

**GR cannot drive a metallic SCF.** A sharp *T* = 0 Fermi surface makes the
occupation-versus-density map *discontinuous* (occupations jump as bands cross
*E*_F), so the SCF oscillates and **no density mixer**, linear damping, DIIS,
or Anderson, converges it. This is intrinsic, not a tuning gap, and is exactly
why every code drives metals with smearing.

The standard (tetrahedron/Gilat) workflow is therefore **smearing to converge,
GR to analyse**:

```python
import numpy as np
from vibeqc.bz_integration import (
    eigenvalues_to_full_grid,
    fractional_kpoints,
    gilat_dos,
    gilat_occupations_on_kmesh,
)

# 1. Converge the metal with Fermi-Dirac smearing (a smooth, robust map).
opts.smearing_temperature = 0.005
r = vq.run_rks_periodic_multi_k_ewald3d(system, basis, kmesh, opts)

# 2. Parameter-free GR Fermi level + occupations + DOS on the converged spectrum.
frac = fractional_kpoints(system.reciprocal_lattice(), np.asarray(kmesh.kpoints))
mesh = tuple(int(x) for x in kmesh.mesh)
occ, e_fermi = gilat_occupations_on_kmesh(
    frac, mesh, r.mo_energies, system.n_electrons())
eps_grid = eigenvalues_to_full_grid(frac, mesh, r.mo_energies)
energies = np.linspace(e_fermi - 0.3, e_fermi + 0.3, 61)
dos = gilat_dos(eps_grid, energies)
```

The metal *total energy* comes from the smearing run (its *T* → 0 limit is well
defined); GR supplies the parameter-free spectral quantities. A complete worked
script is
[`examples/periodic/gilat-be-fcc-dos.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/gilat-be-fcc-dos.py),
and the [Gilat-net tutorial](../tutorial/gilat_net_metals.md) walks through both
the insulator and metal cases.

## Full vs symmetry-reduced (IBZ) meshes

GR integrates over the **full** Brillouin zone (one k-point per microcell). The
driver hooks accept a full Monkhorst-Pack mesh directly. For a symmetry-reduced
(IBZ) mesh, `vq.monkhorst_pack(..., use_symmetry=True)`, the driver expands the
eigenvalues to the full BZ internally (`eps_n(R k) = eps_n(k)`). For a manual
post-SCF call on an IBZ result, expand first with
`expand_ibz_eigenvalues(system, kmesh, eps_per_k)`.

## API (`vibeqc.bz_integration`)

* `gilat_occupations_for_kmesh(system, kmesh, eps_per_k, n_elec, g=2.0)`,
  driver-facing entry: occupations + Fermi level for a full **or** IBZ mesh.
* `gilat_occupations_on_kmesh(kpoints_frac, mesh, eps_per_k, n_elec, g=2.0)`,
  lower-level (full mesh only).
* `gilat_dos(eps_grid, energies, g=2.0)`, density of states *D*(*E*).
* `gilat_raubenheimer_occupations(eps_grid, n_elec, g=2.0)`, the dense-grid
  kernel.
* `eigenvalues_to_full_grid`, `expand_ibz_eigenvalues`, `fractional_kpoints`,
  helpers to assemble the full eigenvalue grid.

## Caveats

* **Metals are post-SCF only**, see above.
* **DOS band-ordering artifact**, `gilat_dos` sorts eigenvalues per k-point, so
  band crossings introduce small spurious features away from the Fermi level
  (the DOS is clamped non-negative). The Fermi level and occupations, the
  integration that matters, are unaffected.
* **Forces**, GR (like tetrahedron) occupation derivatives break the simple
  Hellmann-Feynman + Pulay force expression; for periodic-metal *gradients* use
  smearing, whose Mermin free energy gives clean forces.

## Citations

Cite **Gilat, G. & Raubenheimer, L. J.** *Accurate Numerical Method for
Calculating Frequency-Distribution Functions in Solids.* Phys. Rev. **144**, 390
(1966), {vibeqc-cite}`gilat_raubenheimer_1966`, whenever you use the Gilat net
(`bz_integration="gilat"` or the `vibeqc.bz_integration` post-SCF helpers),
and Gilat's 1972 analysis of the method, {vibeqc-cite}`gilat_spectral_1972`,
for the DOS helper. The runner registers both automatically on the routes
that honour the flag.

## See also

* [Smearing](smearing.md), the SCF-driving counterpart; converge metals there,
  then bring the converged spectrum here.
* [Moving from CRYSTAL](from_crystal.md), the `SHRINK IS ISP` → Gilat-net
  mapping.
* [k-points](k_points.md) and [multi-k SCF](multi_k_scf.md), building the mesh
  and running the SCF that produces the eigenvalues.
