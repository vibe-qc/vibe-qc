"""Run the χ-CCM / aiccm2026dev-b RHF/RKS backend matrix.

This example is intentionally separate from ``aiccm2026dev_a_demo.py``.
In 3D it runs the finite-character (Γ-centred character-mesh) CCM stream
through the public periodic dispatcher with four-center, RI, or RIJCOSX. A
``--dim 1`` or ``--dim 2`` invocation instead verifies the intentional
all-backend fail-close and never prints an energy.
"""

from __future__ import annotations

import argparse

import numpy as np

import vibeqc as vq


def h2_system(dim: int) -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    if dim == 1:
        lattice = np.diag([8.0, 20.0, 20.0])
    elif dim == 2:
        lattice = np.diag([8.0, 8.0, 20.0])
    elif dim == 3:
        lattice = np.diag([8.0, 12.0, 12.0])
    else:
        raise ValueError("dim must be 1, 2, or 3")
    system = vq.PeriodicSystem(
        dim,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--backend",
        choices=("four_center", "ri", "rijcosx"),
        default="four_center",
    )
    parser.add_argument("--functional", default="pbe0")
    args = parser.parse_args()

    system, basis = h2_system(args.dim)
    mesh = (2, 1, 1)
    common = dict(
        jk_method="aiccm2026dev-b",
        kpoints=mesh,
        aiccm_backend=args.backend,
        max_iter=40,
        progress=False,
        citations=False,
        write_xyz_file=False,
        output_qvf=False,
        write_molden_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
    )

    if args.dim < 3:
        for method in ("RHF", "RKS"):
            try:
                if method == "RHF":
                    vq.run_aiccm2026dev_b_rhf(
                        system,
                        basis,
                        lattice_extension=mesh,
                        backend=args.backend,
                        progress=False,
                    )
                else:
                    vq.run_aiccm2026dev_b_rks(
                        system,
                        basis,
                        args.functional,
                        lattice_extension=mesh,
                        backend=args.backend,
                        progress=False,
                    )
            except NotImplementedError as exc:
                print(
                    f"EXPECTED FAIL-CLOSED: {args.dim}D {method} "
                    f"with {args.backend}: {exc}"
                )
            else:
                raise RuntimeError(
                    f"{args.dim}D χ-CCM-B {method}/{args.backend} "
                    "unexpectedly returned an energy"
                )
        return

    rhf = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        output=f"aiccm2026dev-b-{args.dim}d-{args.backend}-rhf",
        **common,
    )
    rks = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional=args.functional,
        output=f"aiccm2026dev-b-{args.dim}d-{args.backend}-rks",
        **common,
    )

    for label, result in (("RHF", rhf), (f"RKS/{args.functional}", rks)):
        diagnostic = result.aiccm2026dev_b
        print(
            f"{label:10s} E/cell={result.energy: .12f} Ha "
            f"idempotency={diagnostic.density_idempotency_error:.3e} "
            f"electrons={diagnostic.electron_count_error:.3e}"
        )


if __name__ == "__main__":
    main()
