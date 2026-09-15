# The Gilat-Raubenheimer net: metallic Brillouin-zone integration

The **Gilat-Raubenheimer (GR) net** integrates the Brillouin-zone occupation
analytically per microcell, the integrator behind CRYSTAL's `SHRINK IS ISP`
second net, with **no smearing width** to converge. This tutorial shows the two
ways you use it: self-consistently for an insulator, and post-SCF for a metal.

Select it with `bz_integration="gilat"` on any multi-k Ewald driver
(`run_rhf_periodic_multi_k_ewald3d`, `run_rks_...`, `run_uhf_...`,
`run_uks_...`). The public BIPOLE RKS/UHF/UKS route accepts the same selector.
Full reference:
[Metallic BZ integration](../user_guide/metallic_bz_integration.md).

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

The two energies match to µHa. The same call works for RKS, UHF, and UKS, and
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
(1966) when you use the Gilat net.

## See also

- [Fermi-Dirac smearing](fermi_dirac_smearing.md), the SCF-driving counterpart
  for metals.
- [Metallic BZ integration](../user_guide/metallic_bz_integration.md), full
  reference, API, and caveats.
- [Moving from CRYSTAL](../user_guide/from_crystal.md), the `SHRINK IS ISP`
  mapping.
