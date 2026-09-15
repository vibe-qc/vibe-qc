"""PySCF mirror — SiO2-alpha-quartz RHF / pob-tzvp / Γ via GDF
(exxdiv=ewald). Same geometry / basis as
``../../periodic/SiO2-alpha-quartz/SiO2-alpha-quartz-RHF-pobtzvp.py``.

Run::
    .venv/bin/python examples/periodic_pyscf/SiO2-alpha-quartz/SiO2-alpha-quartz-RHF-pobtzvp.py
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

CELL_ANG = [[4.9134, 0.0, 0.0],
 [-2.456699999999999, 4.255129218954461, 0.0],
 [0.0, 0.0, 5.4052]]
ATOM_DATA = [('Si', [2.30782398, 0.0, 1.8017333333333332]),
 ('Si', [-1.1539119899999997, 1.9986341941429104, 0.0]),
 ('Si', [1.3027880100000004, 2.2564950248115507, 3.6034666666666664]),
 ('Si', [-1.1539119899999997, 1.9986341941429104, 3.603466666666667]),
 ('Si', [2.30782398, 0.0, 0.0]),
 ('Si', [1.3027880100000004, 2.2564950248115507, 1.8017333333333332]),
 ('O', [1.3759976700000003, 1.1356939885389457, 0.6437593199999999]),
 ('O', [3.2418613200000004, 0.6238019434987238, 4.247225986666666]),
 ('O', [2.752241010000001, 2.4956332869167914, 2.445492653333333]),
 ('O', [0.2955410100000006, 1.7594959320376695, 4.76144068]),
 ('O', [-1.0807023299999996, 3.1194352304155153, 1.1579740133333334]),
 ('O', [0.7851613200000007, 3.631327275455737, 2.9597073466666663])]

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
from pyscf.pbc import scf as pbc_scf
mf = pbc_scf.RHF(cell)
mf.exxdiv = "ewald"
mf.with_df = pbc_df.GDF(cell)
mf.with_df.build()
mf.max_cycle = 80
mf.conv_tol = 1e-7
e = mf.kernel()
print(f"\nFinal: E = {e:.10f} Ha  converged={mf.converged}")
