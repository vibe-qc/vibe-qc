---
myst:
  html_meta:
    "description": "The Gilat-Raubenheimer net in vibe-qc: parameter-free Brillouin-zone integration for metals by linear interpolation of the bands inside each microcell, how it compares with Fermi-Dirac and cold smearing, where bz_integration='gilat' is accepted (bipole, gdf, rijcosx) and where it fails closed (gpw, gapw), used self-consistently for an insulator and post-SCF for Be metal."
    "og:title": "vibe-qc tutorial - the Gilat-Raubenheimer net"
---

# The Gilat-Raubenheimer net: metallic Brillouin-zone integration

The **Gilat-Raubenheimer (GR) net** integrates the Brillouin-zone occupation
analytically per microcell, the integrator behind CRYSTAL's `SHRINK IS ISP`
second net, with **no smearing width** to converge. This tutorial shows the two
ways you use it: self-consistently for an insulator, and post-SCF for a metal.

Select it with `bz_integration="gilat"` on any multi-k Ewald driver
(`run_rhf_periodic_multi_k_ewald3d`, `run_rks_...`, `run_uhf_...`,
`run_uks_...`). Through `run_periodic_job` the flag is honoured on the
`bipole`, `gdf` and `rijcosx` routes (BIPOLE RHF, periodic ROHF/ROKS and
any run with a finite smearing temperature refuse it), and it **fails
closed on `gpw` and `gapw`**: those drivers do not thread it into their
occupation logic, so rather than silently running Fermi-Dirac and citing
Gilat-Raubenheimer they raise. The occupation kernel itself is production
(see [Methods: theory and status](../methods.md#how-to-read-the-status-labels)).
Full reference: [Metallic BZ integration](../user_guide/metallic_bz_integration.md).
The numbers below come from a vibe-qc 0.17.x development build on Apple M2.

## Theory

**The problem.** Every ground-state quantity of a crystal is a
Brillouin-zone integral of a band function, and for a metal the integrand
is discontinuous: the occupation $n_i(\mathbf k)$ jumps from 2 to 0 where
the band $\varepsilon_i(\mathbf k)$ crosses the Fermi level $E_F$. A
Monkhorst-Pack sum with integer occupations converges only as a power of
the mesh spacing and, worse, the electron count as a function of $E_F$ is a
staircase, so the Fermi level itself is ill-defined on a coarse mesh.
[Smearing](fermi_dirac_smearing.md) cures both by replacing the step with a
smooth function of width $k_BT$, at the price of a width to converge and a
zero-temperature extrapolation. The Gilat-Raubenheimer net
({vibeqc-cite}`gilat_raubenheimer_1966`, analysed and generalised in
{vibeqc-cite}`gilat_spectral_1972`) attacks the integral geometrically
instead.

**Linear interpolation per microcell.** Divide the zone into one
parallelepiped microcell per regular-mesh point $\mathbf k_0$, of the same
shape as the mesh cell. Inside each microcell the band is linearised,

$$
\varepsilon_n(\mathbf k_0 + \delta\mathbf k) \approx \varepsilon_n(\mathbf k_0) + \mathbf v_n(\mathbf k_0)\cdot\delta\mathbf k ,
\qquad
\mathbf v_n = \nabla_{\mathbf k}\,\varepsilon_n ,
$$

so the constant-energy surfaces inside the cell are planes and the volume
of the cell below $E_F$ can be written down in closed form. In the cell's
fractional coordinates $t_i \in [-\tfrac12, \tfrac12]$ along the three mesh
axes the plane is $\sum_i g_i t_i = E_F - \varepsilon_n(\mathbf k_0)$ with
directional slopes $g_i = \mathbf v_n\cdot\mathbf b_i/N_i$ ($\mathbf b_i$
the reciprocal-lattice vectors, $N_i$ the mesh divisions), and the
occupied fraction $f_n(\mathbf k_0) \in [0, 1]$ is the volume of the unit
cube cut off by that plane, a piecewise cubic polynomial in
$E_F - \varepsilon_n(\mathbf k_0)$ (equivalently the cumulative distribution
of a weighted sum of three uniform random variables). The slopes are taken
from the eigenvalue grid itself as periodic central differences along each
mesh axis, so no analytic band gradient is needed and the cell Jacobian
cancels in the fraction; a non-orthogonal reciprocal lattice is carried
entirely by the $g_i$. The GR occupation at the grid point is $2 f_n$, the
electron count is

$$
N_e(E_F) = \frac{2}{N_1N_2N_3}\sum_{\mathbf k_0}\sum_n f_n(\mathbf k_0; E_F) ,
$$

a continuous, monotonic function of $E_F$ that is bisected to the target
electron count, and the density of states follows as
$D(E) = \mathrm dN_e/\mathrm dE$, which was the original purpose of the
method (phonon and electron spectral functions). The error of the
linearisation is second order in the mesh spacing where the bands are
smooth and concentrated near band crossings, where the energy-ordered
central differences pick up a spurious slope; vibe-qc drops a mesh
direction from the box when its slope is negligible against the steepest
one, which keeps the cubic-fraction arithmetic stable for flat or
one-dimensional bands.

**GR versus smearing.** The two methods answer different questions.
Smearing gives a *smooth* occupation map $n_i(\varepsilon_i - E_F)$ and
therefore a well-posed SCF fixed point, an electronic entropy and a free
energy, and it needs a width: converge it, or extrapolate it away. GR
gives the *sharp* $T = 0$ occupations of the linearised bands: no
parameter, no entropy, no extrapolation, a Fermi level and a DOS that
converge with the mesh alone, but an occupation-versus-density map that is
still discontinuous wherever a band crosses $E_F$ inside a cell, which is
exactly why it cannot drive a metallic SCF (section 2 and "Why not just
drive the metal SCF with GR?" below). The practical division of labour,
inherited from the tetrahedron-method literature, is therefore: smearing
to converge the density and the total energy, GR (or a tetrahedron method)
to integrate the converged spectrum. For an insulator both give integer
occupations and the same energy, which section 1 checks.

**Where it helps and where it fails.** GR is the right tool for a
parameter-free Fermi level and DOS of a converged metal, for checking that
a smearing extrapolation landed in the right place, and for insulators
where it costs nothing. It does not help the SCF of a metal, it needs a
*full* regular mesh (an IBZ-reduced mesh is expanded internally, and a
mesh whose reduced points cannot be mapped back is refused), it does not
provide forces (the multi-k GDF gradient refuses `bz_integration="gilat"`),
and its Fermi level moves by a few mHa between nearly equivalent
converged spectra because a sharp Fermi surface is sensitive to
eigenvalue changes that the variational total energy is not.

## 1. Insulators, self-consistent GR

For a gapped system GR drives the SCF directly. Its occupations are integer
across the gap, so it reproduces the ordinary Aufbau result, the value is the
parameter-free machinery and a clean Fermi level, not a different energy. Here is
H₂ in a 30-bohr box at a full 2×2×2 mesh:

```python
import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import PeriodicRHFOptions, monkhorst_pack

box = 30.0
c = box / 2
sysp = vq.PeriodicSystem(
    3, np.eye(3) * box,
    [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)

opts = PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0

r_aufbau = vq.run_rhf_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts)
r_gilat = vq.run_rhf_periodic_multi_k_ewald3d(
    sysp, basis, kmesh, opts, bz_integration="gilat")

print(f"Aufbau : {r_aufbau.energy:.8f} Ha")
print(f"Gilat  : {r_gilat.energy:.8f} Ha")   # identical for a gapped system
```

```text
Aufbau : -1.11676072 Ha
Gilat  : -1.11676072 Ha
```

The two energies agree to every printed digit. The same call works for RKS, UHF, and UKS, and
for symmetry-reduced (IBZ) meshes, the driver expands the eigenvalues to the
full Brillouin zone internally.

## 2. Metals, smearing to converge, GR to analyse

**GR cannot drive a metallic SCF.** A sharp *T* = 0 Fermi surface makes the
occupation-versus-density map *discontinuous* (occupations jump as bands cross
*E*_F), so the SCF oscillates and no density mixer converges it. That is exactly
why metals are driven with [smearing](fermi_dirac_smearing.md), a smooth map.
The GR net then gives the **parameter-free** Fermi level, occupations, and
density of states on the converged spectrum (the standard tetrahedron/Gilat
usage).

Here is Be fcc (a free-electron-like metal, 4 electrons/cell):

```python
import numpy as np
import vibeqc as vq
from vibeqc.bz_integration import (
    eigenvalues_to_full_grid, fractional_kpoints,
    gilat_dos, gilat_occupations_on_kmesh,
)

a = 3.2 / 0.529177210903                      # bohr, metallic density
lattice = (a / 2) * np.array([[0., 1, 1], [1, 0, 1], [1, 1, 0]])
sysp = vq.PeriodicSystem(3, lattice, [vq.Atom(4, [0, 0, 0])])
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(sysp, [4, 4, 4], use_symmetry=False)  # full mesh

# Step 1: converge with Fermi-Dirac smearing (smooth, robust).
opts = vq.PeriodicKSOptions()
opts.functional = "pbe"
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.smearing_temperature = 0.005
r = vq.run_rks_periodic_multi_k_ewald3d(sysp, basis, kmesh, opts)
print(f"smearing SCF: E = {r.energy:.6f} Ha/cell  ({r.n_iter} iters)")

# Step 2: parameter-free GR Fermi level + DOS on the converged eigenvalues.
frac = fractional_kpoints(sysp.reciprocal_lattice(), np.asarray(kmesh.kpoints))
mesh = tuple(int(x) for x in kmesh.mesh)
occ, e_fermi = gilat_occupations_on_kmesh(
    frac, mesh, r.mo_energies, sysp.n_electrons())
eps_grid = eigenvalues_to_full_grid(frac, mesh, r.mo_energies)
energies = np.linspace(e_fermi - 0.3, e_fermi + 0.3, 61)
dos = gilat_dos(eps_grid, energies)

print(f"GR Fermi level = {e_fermi:.3f} Ha")
print(f"DOS at E_F = {float(np.interp(e_fermi, energies, dos)):.1f} states/Ha/cell")
```

```text
smearing SCF: E = -14.568391 Ha/cell  (6 iters)
GR Fermi level = 0.370 Ha
DOS at E_F = 10.8 states/Ha/cell
```

The Fermi level is an eigenvalue of the periodic Fock operator, so it sits on
that operator's constant (Ewald) gauge and its absolute value is not a work
function; only differences against the same operator's bands mean anything.
On a busy machine the 4x4x4 SCF takes many minutes; use a 2x2x2 mesh for a
first run.

`DOS(E_F)` is nonzero, the signature of a metal. The metal *total energy* comes
from the smearing run (its *T* → 0 limit is well defined and stable); GR supplies
the parameter-free spectral quantities. Note the GR Fermi level / DOS can shift
by a few mHa between runs, a metal's sharp Fermi surface is sensitive to tiny
eigenvalue changes, while the variational total energy is not.

A complete script is
[`examples/periodic/gilat-be-fcc-dos.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/gilat-be-fcc-dos.py).

## Why not just drive the metal SCF with GR?

Because it provably will not converge. Sharp *T* = 0 occupations are a
discontinuous function of the density, and *no* density mixer (linear damping,
DIIS, or Anderson) converges a discontinuous fixed-point map, confirmed for Be
fcc with all three. Smearing replaces the step with a smooth Fermi-Dirac
function, which converges; GR then recovers the *T* = 0 integration without a
width to extrapolate. See
[Metallic BZ integration § Metals](../user_guide/metallic_bz_integration.md).

## Citation

Cite **Gilat, G. & Raubenheimer, L. J.** *Accurate Numerical Method for
Calculating Frequency-Distribution Functions in Solids.* Phys. Rev. **144**, 390
(1966) ({vibeqc-cite}`gilat_raubenheimer_1966`) when you use the Gilat net;
Gilat's 1972 review ({vibeqc-cite}`gilat_spectral_1972`) compares the
linear-interpolation net with the other spectral-integration schemes.
`run_periodic_job` writes both into the `.bibtex` sidecar when the flag is
honoured.

## See also

- [Fermi-Dirac smearing](fermi_dirac_smearing.md), the SCF-driving counterpart
  for metals.
- [Metallic BZ integration](../user_guide/metallic_bz_integration.md), full
  reference, API, and caveats.
- [Moving from CRYSTAL](../user_guide/from_crystal.md), the `SHRINK IS ISP`
  mapping.
