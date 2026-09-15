"""Experimental χ-CCM occupied localization.

Run from the repository root after an editable install. The example uses a
small 3D periodic H2 control in an elongated cell and prints only invariants,
not reference energies. Genuine 1D/2D χ-CCM-B SCF fails closed.
"""

from __future__ import annotations

import numpy as np

import vibeqc as vq


def main() -> None:
    system = vq.PeriodicSystem(
        3,
        np.diag([5.0, 20.0, 20.0]),
        [
            vq.Atom(1, [1.8, 10.0, 10.0]),
            vq.Atom(1, [3.2, 10.0, 10.0]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.max_iter = 80
    scf = vq.run_aiccm2026dev_b_rhf(
        system,
        basis,
        mesh=(4, 1, 1),
        options=options,
        backend="four_center",
        progress=False,
    )
    localized = vq.localize_aiccm2026dev_b_occupied(
        scf,
        system,
        basis,
        method="wannier",
    )
    print(f"density invariance: {localized.density_invariance_error:.3e}")
    print(f"unitarity:          {localized.unitary_error:.3e}")
    print(f"translation gauge: {localized.translation_covariance_error:.3e}")
    print("centers / bohr:")
    print(localized.centers_bohr)
    print("spreads / bohr^2:")
    print(localized.spreads_bohr2)
    print("wrap flags:")
    print(localized.aliasing_detected)


if __name__ == "__main__":
    main()
