"""lizardite-1T — RKS-PBE0 / pob-DZVP-rev2 / Γ via GDF.

Mg₃Si₂O₅(OH)₄, P3̄1m (trigonal). Mellini & Zanazzi 1987.

Closed-shell — runnable today on the v0.7.1-spike GDF chain.

Caveats
-------
* Γ-only sampling — for k-converged numbers comparable to literature
  (CRYSTAL/PW1PW) wait for ``feature/multi-k-periodic-scf``.
* PBE0 is the canonical solid-state hybrid (α=0.25). Sister script
  ``lizardite-1T-RKS-PW1PW-pobdzvp.py`` uses PW1PW (α=0.20, PWGGA
  exchange + correlation) once that functional lands via
  ``feature/uks-periodic-gdf``.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _minerals import lizardite_1T

import vibeqc as vq

OUTPUT_STEM = Path(__file__).resolve().with_suffix("")

system, label = lizardite_1T()
basis = vq.BasisSet(system.unit_cell_molecule(), "pob-dzvp-rev2")

print(f"lizardite-1T: {label}, nbf={basis.nbasis}, "
      f"n_e={system.n_electrons()}")

vq.run_periodic_job(
    system, basis,
    method="RKS",
    functional="pbe0",
    jk_method="gdf",
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=False,
)
