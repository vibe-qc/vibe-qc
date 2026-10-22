#!/usr/bin/env python3
"""Generate the independent Cl2/ECP periodic UHF references.

Run in an environment with optional PySCF. The reference uses direct
range-separated J/K integrals, not Gaussian density fitting of J or K.
The finite-k Ewald energy is deliberately not an isolated molecular energy.
See Carrier, Rohra and Goerling, PRB 75, 205126 (2007), Sec. II, Eq. (15),
DOI: 10.1103/PhysRevB.75.205126.

The separate isolated-molecule ORCA 6.1.1 control is reproducible with:
    ! UHF LANL2DZ VeryTightSCF
    * xyz 0 1
    Cl 0 0 0
    Cl 0 0 1.99
    *
Its energy is -29.335833072480 Ha. It is not the reference for the
periodic calculation's default exchange_exxdiv='ewald'.
"""
from __future__ import annotations

import json


def main() -> None:
    import numpy as np
    import pyscf
    from pyscf import lib
    from pyscf.pbc import gto, scf
    from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder

    lib.num_threads(2)
    bond = 1.99 * 1.8897261246257702
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.a = np.eye(3) * 20.0
    cell.atom = [("Cl", (10., 10., 10.-bond/2)),
                 ("Cl", (10., 10., 10.+bond/2))]
    cell.basis = "lanl2dz"
    cell.ecp = "lanl2dz"
    cell.charge = cell.spin = 0
    cell.precision = 1e-10
    cell.max_memory = 1500
    cell.verbose = 0
    cell.build()
    ionic_energy = float(cell.energy_nuc())
    results = []
    for mesh in ((1, 1, 1), (3, 1, 1)):
        kpts = cell.make_kpts(mesh, with_gamma_point=True)
        mf = scf.KUHF(cell, kpts).density_fit(auxbasis="def2-svp-jkfit")
        # GDF supplies nuclear attraction; RSJK overrides both J and K.
        mf.rsjk = RangeSeparatedJKBuilder(cell, kpts)
        mf.exxdiv = "ewald"
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        # Some PySCF SCF wrappers select molecular nuclear repulsion whenever
        # rsjk is present, even for dimension=3. Match the periodic ionic gauge
        # explicitly, using the independently evaluated cell Ewald sum.
        mf.energy_nuc = lambda: ionic_energy
        energy = float(mf.kernel())
        if not mf.converged:
            raise RuntimeError(f"Reference did not converge for mesh {mesh}")
        results.append({"mesh": mesh, "energy_hartree": energy})
    print(json.dumps({"pyscf_version": pyscf.__version__, "basis": "LANL2DZ",
                      "ecp": "Hay-Wadt", "valence_electrons": cell.nelectron,
                      "nao": cell.nao, "ionic_energy_hartree": ionic_energy,
                      "exchange": "ewald", "results": results}, indent=2))


if __name__ == "__main__":
    main()
