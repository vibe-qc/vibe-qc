"""Run χ-CCM local-PNO MP2 and CCSD(T) on a finite torus."""

from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
from vibeqc.dlpno.mp2 import DLPNOMP2Options


def main() -> None:
    system = vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 7.0]),
        [
            vq.Atom(3, [10.0, 10.0, 2.5]),
            vq.Atom(1, [10.0, 10.0, 4.1]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = (1, 1, 2)

    mp2 = vq.run_aiccm2026dev_b_dlpno_mp2(
        system,
        basis,
        mesh,
        dlpno_options=DLPNOMP2Options(
            localise="wannier",
            n_frozen=0,
            tcut_pno=0.0,
            tcut_pno_weak=0.0,
            tcut_mkn=0.0,
            tcut_pairs=0.0,
            tcut_pairs_weak=0.0,
        ),
        progress=False,
    )
    ccsd_t = vq.run_aiccm2026dev_b_dlpno_ccsd_t(
        system,
        basis,
        mesh,
        cc_options=LocalCCSDOptions(
            localise="pipek-mezey",
            n_frozen=0,
            tcut_pno=1e-7,
            tcut_mkn=0.0,
            tcut_pairs=0.0,
            coupling_radius=0.0,
            residual_domain="pair",
            compute_triples=True,
            tcut_tno=0.0,
            triples_mode="t1",
        ),
        progress=False,
    )

    print(f"local-PNO MP2 correlation/cell {mp2.e_corr_per_cell: .12f} Ha")
    print(
        "raw/corrected complete-space difference "
        f"{mp2.complete_space_correction_per_cell: .3e} Ha/cell"
    )
    if mp2.local_correlation_space is not None:
        local = mp2.local_correlation_space
        print(
            "PAO rank and translation-pair orbits "
            f"{local.pao_rank}, {local.n_translation_unique_pairs}/"
            f"{local.n_pairs}"
        )
    print(f"local-PNO CCSD correlation/cell {ccsd_t.e_corr_per_cell: .12f} Ha")
    print(f"local-PNO (T)/cell             {ccsd_t.e_t_per_cell: .12f} Ha")
    print(f"local-PNO CCSD(T)/cell         {ccsd_t.e_total_per_cell: .12f} Ha")
    print(
        "finite-torus transform residuals "
        f"imag={ccsd_t.cderi_imaginary_residual:.3e} "
        f"sym={ccsd_t.cderi_symmetry_residual:.3e}"
    )


if __name__ == "__main__":
    main()
