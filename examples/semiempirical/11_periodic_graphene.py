#!/usr/bin/env python3
"""Periodic GFN2-xTB on graphene — energy and band gap.

Demonstrates periodic semiempirical calculations on a 2-atom graphene
unit cell with all three method families: DFTB0, SCC-DFTB, GFN2-xTB.

Run:
    .venv/bin/python examples/semiempirical/11_periodic_graphene.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical import SemiempiricalParameters


def main():
    # ── Graphene 2-atom unit cell (AA = 2.46 Å ≈ 4.65 bohr) ──
    a = 4.65
    c = 20.0  # vacuum layer
    lattice = np.array(
        [[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    )
    atoms = [
        Atom(6, [0.0, 0.0, 0.0]),
        Atom(6, [a / 2, a * np.sqrt(3) / 6, 0.0]),
    ]
    system = PeriodicSystem(2, lattice, atoms, charge=0, multiplicity=1)

    params = SemiempiricalParameters.dftb0_default()

    # ── DFTB0 ──
    r0 = _se.run_dftb0_gamma(system, params)
    print(f"DFTB0:          E = {r0.energy:12.6f} Ha")

    # ── SCC-DFTB ──
    sopts = _se.PeriodicSCCOptions()
    sopts.max_iter = 80
    rs = _se.run_scc_dftb_gamma(system, params, sopts)
    print(f"SCC-DFTB:       E = {rs.energy:12.6f} Ha  conv={rs.converged}")

    # ── GFN2-xTB ──
    try:
        from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
        from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

        gp = load_gfn2_params()
        xopts = _xtb.XTBSccOptions()
        xopts.max_iter = 400
        rg = _xtb.run_gfn2_xtb_gamma(system, gp, xopts)
        print(f"GFN2-xTB:       E = {rg.energy:12.6f} Ha  conv={rg.converged}  "
              f"n_iter={rg.n_iter}  n_cells={rg.n_cells}")
    except Exception as e:
        print(f"GFN2-xTB:       skipped ({e})")

    # ── PM6 ──
    try:
        from vibeqc.semiempirical import PeriodicPM6Model

        pm6 = PeriodicPM6Model(system)
        e_pm6 = pm6.energy()
        print(f"PM6:            E = {e_pm6:12.6f} Ha")
    except Exception as e:
        print(f"PM6:            skipped ({e})")

    # ── Performance note ──
    print(f"\nGraphene unit cell: {len(atoms)} atoms, C-C ≈ 1.42 Å")
    print("Semiempirical methods are 10³−10⁴× faster than DFT for periodic systems.")


if __name__ == "__main__":
    main()
