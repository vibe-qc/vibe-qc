"""Run the χ-CCM / aiccm2026dev-b canonical 3D RI-MP2 route."""

from __future__ import annotations

import numpy as np

import vibeqc as vq


def main() -> None:
    system = vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 6.0]),
        [
            vq.Atom(1, [10.0, 10.0, 2.3]),
            vq.Atom(1, [10.0, 10.0, 3.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    result = vq.run_aiccm2026dev_b_mp2(
        system,
        basis,
        (1, 1, 2),
        progress=False,
    )
    print(f"RHF/cell       {result.e_hf_per_cell: .12f} Ha")
    print(f"RI-MP2 SS/cell {result.e_corr_ss_per_cell: .12f} Ha")
    print(f"RI-MP2 OS/cell {result.e_corr_os_per_cell: .12f} Ha")
    print(f"RI-MP2/cell    {result.e_corr_per_cell: .12f} Ha")
    print(f"Total/cell     {result.e_total_per_cell: .12f} Ha")
    print(
        "checks          "
        f"momentum={result.momentum_conservation_error:.3e} "
        f"imaginary={result.max_energy_imaginary_residual:.3e}"
    )


if __name__ == "__main__":
    main()
