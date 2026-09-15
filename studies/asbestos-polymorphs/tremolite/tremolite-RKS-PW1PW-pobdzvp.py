"""tremolite — RKS-PW1PW / pob-DZVP-rev2 / Γ via GDF.

Ca₂Mg₅Si₈O₂₂(OH)₂, C2/m (monoclinic). Hawthorne & Grundy 1976. Mg endmember of the tremolite-actinolite series.

PW1PW is Bredow's 1-parameter hybrid (α=0.20, PWGGA exchange + PWGGA
correlation; Bredow & Gerson, *Phys. Rev. B* **61**, 5194 (2000)).
The canonical CRYSTAL hybrid used in the inspirational 2014 Al₂O₃
paper that this asbestos study is modeled on, and in the pob-TZVP
paper SI Table 1 (Peintinger-Vilela Oliveira-Bredow JCC 2013).

Status: **gated on `feature/uks-periodic-gdf` landing** —
Functional("pw1pw") is not yet defined in vibe-qc. Will run as soon
as that chip merges. The PBE0 sister script runs today.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _minerals import tremolite

import vibeqc as vq

OUTPUT_STEM = Path(__file__).resolve().with_suffix("")

system, label = tremolite()
basis = vq.BasisSet(system.unit_cell_molecule(), "pob-dzvp-rev2")

print(f"tremolite: {label}, nbf={basis.nbasis}, "
      f"n_e={system.n_electrons()}")

vq.run_periodic_job(
    system, basis,
    method="RKS",
    functional="pw1pw",
    jk_method="gdf",
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=False,
)
