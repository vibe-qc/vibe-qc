"""NaCl rocksalt: write an SCF density XSF and a teaching-model BXSF.

Run::
    .venv/bin/python examples/periodic/input-xsf-bxsf-nacl.py

Produces:
    output-nacl-xsf.xsf             SCF density on a primitive-cell grid
    output-nacl-xsf.structure.xsf   structure-only companion
    output-nacl-xsf.bxsf            Hcore energies on an 8 x 8 x 8 grid

The XSF density is an RHF result.  The BXSF section deliberately uses the
one-electron core Hamiltonian only so that this example stays on the public
API and focuses on the file format.  Hcore bands are not SCF bands and must
not be interpreted as a physical NaCl band structure or Fermi surface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vibeqc as vq


HERE = Path(__file__).resolve().parent
OUTPUT_STEM = HERE / "output-nacl-xsf"

ANG2BOHR = 1.0 / 0.529177210903


# True rocksalt primitive cell.  PeriodicSystem stores lattice vectors as
# columns, so each vector is one column of ``lattice``.
a = 5.640 * ANG2BOHR
a1 = np.array([0.0, a / 2.0, a / 2.0])
a2 = np.array([a / 2.0, 0.0, a / 2.0])
a3 = np.array([a / 2.0, a / 2.0, 0.0])
lattice = np.column_stack((a1, a2, a3))
cl_position = 0.5 * (a1 + a2 + a3)

system = vq.PeriodicSystem(
    3,
    lattice,
    [vq.Atom(11, [0.0, 0.0, 0.0]), vq.Atom(17, cl_position)],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")


# Part 1: converged RHF density in XSF format.
vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    output=OUTPUT_STEM,
    output_qvf=False,
    write_density=True,
    density_spacing_bohr=0.2,
    use_diis=True,
    max_iter=80,
    conv_tol_energy=1e-7,
)
density_path = OUTPUT_STEM.with_suffix(".xsf")
print(f"Wrote {density_path.name}")
print(f"  Open with: moltui {density_path.name}")


# Part 2: public-API Hcore band grid for demonstrating BXSF output.
# This does not reuse the RHF Fock matrix from Part 1.  Electron-electron
# Coulomb and exchange terms are absent, so these energies are non-SCF.
lattice_options = vq.LatticeSumOptions()
lattice_options.cutoff_bohr = 12.0
lattice_options.nuclear_cutoff_bohr = 12.0

overlap_lattice = vq.compute_overlap_lattice(
    basis, system, lattice_options,
)
kinetic_lattice = vq.compute_kinetic_lattice(
    basis, system, lattice_options,
)
nuclear_lattice = vq.compute_nuclear_lattice(
    basis, system, lattice_options,
)

mesh = (8, 8, 8)
energies = np.empty((*mesh, basis.nbasis), dtype=float)
reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)

print(f"Sampling {np.prod(mesh)} k-points for the Hcore BXSF model")
for index in np.ndindex(mesh):
    fractional_k = np.array(
        [index[axis] / mesh[axis] for axis in range(3)],
        dtype=float,
    )
    cartesian_k = reciprocal @ fractional_k

    overlap_k = np.asarray(
        vq.bloch_sum(overlap_lattice, cartesian_k),
        dtype=complex,
    )
    hcore_k = np.asarray(
        vq.bloch_sum(kinetic_lattice, cartesian_k)
        + vq.bloch_sum(nuclear_lattice, cartesian_k),
        dtype=complex,
    )
    overlap_k = 0.5 * (overlap_k + overlap_k.conj().T)
    hcore_k = 0.5 * (hcore_k + hcore_k.conj().T)

    solution = vq.diagonalize_bloch(hcore_k, overlap_k)
    energies[index] = np.asarray(solution.energies, dtype=float)

# For visualization only, put the BXSF reference energy midway between the
# Hcore occupied and virtual band edges.  Hcore is not guaranteed to retain
# NaCl's physical gap, so this midpoint may intersect an Hcore band.  It is
# not an RHF Fermi energy or a physical gap.
n_occupied = system.n_electrons() // 2
valence_maximum = float(np.max(energies[..., n_occupied - 1]))
conduction_minimum = float(np.min(energies[..., n_occupied]))
reference_energy = 0.5 * (valence_maximum + conduction_minimum)
hcore_edge_separation = conduction_minimum - valence_maximum

bxsf_path = vq.write_bxsf(
    OUTPUT_STEM.with_suffix(".bxsf"),
    system,
    energies,
    e_fermi=reference_energy,
)
print(f"Wrote {bxsf_path.name}")
print("  Teaching data: non-SCF Hcore energies, not a physical Fermi surface")
print(
    "  Sampled Hcore edge separation: "
    f"{hcore_edge_separation:.6f} Ha (diagnostic only)"
)
if hcore_edge_separation <= 0.0:
    print("  Hcore manifolds overlap; any displayed surfaces are nonphysical")
print(f"  Open with: moltui {bxsf_path.name}")
print(f"  Or:        xcrysden --bxsf {bxsf_path.name}")
