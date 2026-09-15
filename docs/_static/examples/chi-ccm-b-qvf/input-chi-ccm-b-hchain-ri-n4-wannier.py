from __future__ import annotations

import numpy as np

import vibeqc as vq


system = vq.PeriodicSystem(
    3,
    np.diag([5.0, 20.0, 20.0]),
    [
        vq.Atom(1, [1.8, 10.0, 10.0]),
        vq.Atom(1, [3.2, 10.0, 10.0]),
    ],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="aiccm2026dev-b",
    aiccm_backend="ri",
    aiccm_lattice_extension=(4, 1, 1),
    convergence="off",
    output="chi-ccm-b-hchain-ri-n4-wannier",
    output_qvf=True,
    qvf_wannier_centers=True,
    density_spacing_bohr=0.5,
    write_density=False,
    write_molden_file=False,
    write_xyz_file=False,
    write_poscar_file=False,
    write_xsf_structure_file=False,
    write_cif_file=False,
    write_population_file=False,
    citations=False,
    record_hostname=False,
    progress=False,
)
