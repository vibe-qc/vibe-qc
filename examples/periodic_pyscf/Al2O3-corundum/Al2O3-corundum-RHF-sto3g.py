"""PySCF mirror — α-Al2O3 corundum RHF / sto-3g / Γ via GDF
(exxdiv=ewald). Same geometry and basis as
``../../periodic/Al2O3-corundum-RHF-sto3g/Al2O3-corundum-RHF-sto3g.py``.

Run::
    .venv/bin/python examples/periodic_pyscf/Al2O3-corundum-RHF-sto3g/Al2O3-corundum-RHF-sto3g.py

Produces:
    {stem}.out  — full PySCF log + final energy
"""
from pathlib import Path

from ase.spacegroup import crystal as ase_crystal

from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
from pyscf.pbc import df as pbc_df

A_ANG = 4.7589
C_ANG = 12.991
AL_Z = 0.35216
O_X = 0.30624

OUTPUT_STEM = Path(__file__).resolve().with_suffix("")
out_path = OUTPUT_STEM.with_suffix(".out")

atoms = ase_crystal(
    ["Al", "O"],
    basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
    spacegroup=167,
    cellpar=[A_ANG, A_ANG, C_ANG, 90, 90, 120],
)

atom_str = "; ".join(
    f"{sym} {x:.6f} {y:.6f} {z:.6f}"
    for sym, (x, y, z) in zip(atoms.get_chemical_symbols(),
                              atoms.get_positions())
)

cell = pbc_gto.M(
    atom=atom_str,
    a=atoms.cell.array,
    basis="sto-3g",
    unit="A",
    verbose=4,
    output=str(out_path),
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
