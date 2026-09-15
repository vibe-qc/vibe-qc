"""PySCF mirror — TiO2-rutile RKS-B3LYP / pob-tzvp / Γ via GDF
(exxdiv=ewald). Same geometry / basis as
``../../periodic/TiO2-rutile/TiO2-rutile-RKS-B3LYP-pobtzvp.py``.

Run::
    .venv/bin/python examples/periodic_pyscf/TiO2-rutile/TiO2-rutile-RKS-B3LYP-pobtzvp.py
Produces:
    {stem}.out
"""
from pathlib import Path

import numpy as np

from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import df as pbc_df
from vibeqc._basis_g94 import load_g94_for_pyscf

OUTPUT_STEM = Path(__file__).resolve().with_suffix("")
out_path = OUTPUT_STEM.with_suffix(".out")

CELL_ANG = [[4.5937, 0.0, 0.0], [0.0, 4.5937, 0.0], [0.0, 0.0, 2.9587]]
ATOM_DATA = [('Ti', [0.0, 0.0, 0.0]),
 ('Ti', [2.29685, 2.29685, 1.47935]),
 ('O', [1.4024566100000002, 1.4024566100000002, 0.0]),
 ('O', [3.19124339, 3.19124339, 0.0]),
 ('O', [0.89439339, 3.6993066100000003, 1.47935]),
 ('O', [3.6993066100000003, 0.89439339, 1.47935])]

atom_str = "; ".join(
    f"{sym} {x:.6f} {y:.6f} {z:.6f}"
    for sym, (x, y, z) in ATOM_DATA
)
unique_syms = sorted({sym for sym, _pos in ATOM_DATA})
basis_dict = load_g94_for_pyscf("pob-tzvp", unique_syms)

# Force the GDF FFT mesh to a sane fixed value matching what the
# vibe-qc driver uses internally. Auto-mesh from PySCF + a tight pob-TZVP
# valence exponent picks 1961^3 (~60 GB intermediates) — fills disk.
cell = pbc_gto.M(
    atom=atom_str,
    a=np.array(CELL_ANG, dtype=float),
    basis=basis_dict,
    unit="A",
    verbose=4,
    output=str(out_path),
    mesh=[31, 31, 31],
    precision=1e-8,
)
from pyscf.pbc import dft as pbc_dft
mf = pbc_dft.RKS(cell)
mf.xc = "b3lyp5"
mf.exxdiv = "ewald"
mf.with_df = pbc_df.GDF(cell)
mf.with_df.build()
mf.max_cycle = 80
mf.conv_tol = 1e-7
e = mf.kernel()
print(f"\nFinal: E = {e:.10f} Ha  converged={mf.converged}")
