"""PySCF mirror — MgO rocksalt RHF / sto-3g / Γ via GDF (exxdiv=ewald).

Same geometry and basis as
``../../periodic/MgO-rocksalt-RHF-sto3g/MgO-rocksalt-RHF-sto3g.py``.

Run::
    .venv/bin/python examples/periodic_pyscf/MgO-rocksalt-RHF-sto3g/MgO-rocksalt-RHF-sto3g.py

Produces:
    {stem}.out  — full PySCF log + final energy
"""
from pathlib import Path

from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
from pyscf.pbc import df as pbc_df

A_ANG = 4.211
OUTPUT_STEM = Path(__file__).resolve().with_suffix("")
out_path = OUTPUT_STEM.with_suffix(".out")

mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
atom_lines = []
for fx, fy, fz in mg_frac:
    atom_lines.append(f"Mg {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")
for fx, fy, fz in o_frac:
    atom_lines.append(f"O  {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")

cell = pbc_gto.M(
    atom="; ".join(atom_lines),
    a=[[A_ANG, 0, 0], [0, A_ANG, 0], [0, 0, A_ANG]],
    basis="sto-3g",
    unit="A",
    verbose=4,            # full SCF trace
    output=str(out_path),
    mesh=[31, 31, 31],
    precision=1e-8,
)
mf = pbc_scf.RHF(cell)
mf.exxdiv = "ewald"
mf.with_df = pbc_df.GDF(cell)
mf.with_df.build()
mf.max_cycle = 80
mf.conv_tol = 1e-7
e = mf.kernel()
print(f"\nFinal: E = {e:.10f} Ha  converged={mf.converged}")
