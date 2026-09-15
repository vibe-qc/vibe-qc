"""Periodic H₂ in a vacuum box — DFT+U quick demo.

Closed-shell RKS / LDA / cc-pVDZ on H₂ in a 50 bohr cubic box,
Γ-only kmesh. Demonstrates the +U surface lighting up in a
periodic job, with ``U=4 eV`` on H's p-channel: ``e_dft_plus_u``
appears in the result and Dudarev + Cococcioni land in the
``.bibtex``. Designed to run in a few seconds on a laptop —
unlike the full NiO recipe in
``input-bipole-nio-uks-plus-u.py``.

Physically meaningless (H has no occupied p) but exercises every
piece of the +U pipeline including the citation surface.

Run:
    .venv/bin/python examples/periodic/input-h-chain-plus-u.py
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.periodic_runner import run_periodic_job

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

sysp = vq.PeriodicSystem(
    3,
    np.eye(3) * 50.0,
    [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")  # H has 2s+1p

for U_ev in [0.0, 4.0]:
    out_stem = HERE / f"{STEM}_U{int(U_ev):d}"
    dft_plus_u = (
        [vq.HubbardSite(atom_index=0, l=1, U_ev=U_ev)] if U_ev > 0.0 else None
    )
    print(f"\n=== H2 / RKS-LDA  U={U_ev:.1f} eV ===")
    result = run_periodic_job(
        sysp, basis,
        method="RKS", functional="lda",
        jk_method="bipole", kpoints=(1, 1, 1),
        output=str(out_stem),
        dft_plus_u=dft_plus_u,
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    e_du = float(getattr(result, "e_dft_plus_u", 0.0))
    print(
        f"  converged={result.converged}  n_iter={result.n_iter}  "
        f"E_total={result.energy:.6f} Ha  "
        f"e_dft_plus_u={e_du:.6f} Ha"
    )
