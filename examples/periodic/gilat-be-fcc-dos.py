"""Be fcc metal — parameter-free Gilat-Raubenheimer Fermi level + DOS (post-SCF).

Demonstrates the Gilat-Raubenheimer net (``vibeqc.bz_integration``), the
integrator behind CRYSTAL's ``SHRINK IS ISP`` second net. GR resolves the
Brillouin-zone occupation analytically per microcell, with *no smearing width*
to converge.

The metal workflow (the standard tetrahedron/Gilat usage):

  1. Drive the metallic SCF with Fermi-Dirac **smearing** — a smooth occupation
     map that converges robustly. (GR *cannot* drive a metal SCF: its sharp
     T=0 occupations make the SCF map discontinuous, so it oscillates and no
     density mixer converges it. Use smearing to converge, GR to analyse.)
  2. Evaluate the GR Fermi level / occupations / DOS on the converged
     eigenvalues — parameter-free, no T to extrapolate.

The metal *total energy* comes from step 1 (its T -> 0 limit is well defined);
GR supplies the parameter-free spectral quantities in step 2.

Reference: G. Gilat & L. J. Raubenheimer, "Accurate Numerical Method for
Calculating Frequency-Distribution Functions in Solids", Phys. Rev. 144, 390
(1966), doi:10.1103/PhysRev.144.390.

Run:
    .venv/bin/python examples/periodic/gilat-be-fcc-dos.py
"""

import numpy as np
import vibeqc as vq
from vibeqc.bz_integration import (
    eigenvalues_to_full_grid,
    fractional_kpoints,
    gilat_dos,
    gilat_occupations_on_kmesh,
)

# Be fcc at a metallic density. 4 electrons/cell (even) -> closed-shell RKS.
ANG2BOHR = 1.0 / 0.529177210903
a = 3.2 * ANG2BOHR
lattice = (a / 2.0) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
system = vq.PeriodicSystem(3, lattice, [vq.Atom(4, [0.0, 0.0, 0.0])])
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# GR needs the FULL Brillouin zone (one k-point per microcell). For a
# symmetry-reduced (IBZ) mesh, expand first with
# vibeqc.bz_integration.expand_ibz_eigenvalues.
kmesh = vq.monkhorst_pack(system, [4, 4, 4], use_symmetry=False)

# --- Step 1: converge the metal with Fermi-Dirac smearing (smooth -> robust).
opts = vq.PeriodicKSOptions()
opts.functional = "pbe"
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0
opts.smearing_temperature = 0.005  # k_B T in Hartree (~0.14 eV)
opts.use_diis = True
opts.max_iter = 100
result = vq.run_rks_periodic_multi_k_ewald3d(system, basis, kmesh, opts)
print(f"smearing SCF : E = {result.energy:.6f} Ha/cell  "
      f"converged={result.converged}  n_iter={result.n_iter}")

# --- Step 2: parameter-free Gilat-Raubenheimer analysis on the converged spectrum.
n_electrons = float(system.n_electrons())
frac = fractional_kpoints(system.reciprocal_lattice(),
                          np.asarray(kmesh.kpoints))
mesh = tuple(int(x) for x in kmesh.mesh)

occ, e_fermi = gilat_occupations_on_kmesh(frac, mesh, result.mo_energies, n_electrons)
print(f"GR Fermi level = {e_fermi:.4f} Ha")

eps_grid = eigenvalues_to_full_grid(frac, mesh, result.mo_energies)
energies = np.linspace(e_fermi - 0.30, e_fermi + 0.30, 13)
dos = gilat_dos(eps_grid, energies)
print("  E - E_F (Ha)   DOS (states/Ha/cell)")
for e, d in zip(energies, dos):
    print(f"   {e - e_fermi:+.3f}        {d:8.3f}")

# Expected output (Be is a metal -> nonzero DOS at the Fermi level):
#   smearing SCF : E = -14.247658 Ha/cell  converged=True  n_iter=6
#   GR Fermi level = ~0.51 Ha
#   DOS table: DOS(E_F) ~ 15-20 states/Ha/cell (nonzero -> metallic).
# The smearing total energy is reproducible; the GR Fermi level / DOS shift by
# ~mHa run-to-run because a metal's sharp Fermi surface is sensitive to tiny
# (e.g. threading-order) eigenvalue changes -- expected for metals, and the
# reason the *energy* (variational, stable) is the quantity to compare.
