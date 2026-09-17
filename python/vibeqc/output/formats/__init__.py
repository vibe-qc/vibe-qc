"""``vibeqc.output.formats`` -- per-format writers.

Each module here implements one output file format. The thin-layer
phase only contains the formats new to v0.8.x (`.xyz`); the legacy
writers (`io.molden`, `system_info`, `structured_log`, `perf`,
`crash_dump`, `io.trajectory`) keep their current locations until
the pre-v1.0 coordinator rewrite relocates them under this
subpackage.

Public-API entry points are re-exported by :mod:`vibeqc.output` so
callers do not have to import sub-modules directly.
"""

from __future__ import annotations

from .cif import write_cif
from .cube import (
    CubeRequest,
    parse_write_cube_kwarg,
    requested_mo_indices,
    write_cube_density_for_run_job,
    write_cube_mo_for_run_job,
)
from .extended_xyz import write_extended_xyz
from .population import (
    PopulationSummary,
    compute_population_summary,
    write_population,
)
from .poscar import write_poscar
from .qvf import (
    QVF_FORMAT_VERSION,
    qvf_bytes,
    qvf_density_data,
    qvf_mo_data,
    validate_qvf,
    write_qvf,
)
from .qcschema import write_qcschema
from .vibrational import (
    FD_HESSIAN_SURFACES,
    hessian_surface_label,
    hessian_surface_lines,
    hessian_surface_manifest_fields,
    hessian_unsupported_surface_lines,
    thermochemistry_energy_labels,
)
from .trexio import (
    TrexioData,
    TrexioMOBlock,
    TrexioSparse,
    read_trexio,
    read_trexio_fields,
    write_trexio,
    write_trexio_fields,
)
from .xyz import write_xyz

__all__ = [
    "FD_HESSIAN_SURFACES",
    "hessian_surface_label",
    "hessian_surface_lines",
    "hessian_surface_manifest_fields",
    "hessian_unsupported_surface_lines",
    "thermochemistry_energy_labels",
    "write_qcschema",
    "write_xyz",
    "write_extended_xyz",
    "write_trexio",
    "read_trexio",
    "read_trexio_fields",
    "write_trexio_fields",
    "TrexioData",
    "TrexioMOBlock",
    "TrexioSparse",
    "write_poscar",
    "write_cif",
    # qvf.py
    "write_qvf",
    "qvf_bytes",
    "validate_qvf",
    "QVF_FORMAT_VERSION",
    # cube.py (high-level wrappers for run_job)
    "CubeRequest",
    "parse_write_cube_kwarg",
    "requested_mo_indices",
    "write_cube_density_for_run_job",
    "write_cube_mo_for_run_job",
    # population.py
    "PopulationSummary",
    "compute_population_summary",
    "write_population",
]
