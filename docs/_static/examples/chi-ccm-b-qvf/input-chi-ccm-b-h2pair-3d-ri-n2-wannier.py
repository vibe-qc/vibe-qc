from __future__ import annotations

import numpy as np

import vibeqc as vq


system = vq.PeriodicSystem(
    3,
    np.diag([12.0, 12.0, 12.0]),
    [
        vq.Atom(1, [6.0, 6.0, 5.3]),
        vq.Atom(1, [6.0, 6.0, 6.7]),
    ],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="aiccm2026dev-b",
    aiccm_backend="ri",
    aiccm_lattice_extension=(2, 1, 1),
    convergence="off",
    output="chi-ccm-b-h2pair-3d-ri-n2-wannier",
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
