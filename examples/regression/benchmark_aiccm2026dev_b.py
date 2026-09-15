#!/usr/bin/env python3
"""Inspect χ-CCM, historical CCM weights, Γ-CCM, and a k-mesh control.

The default system is the alternating H4 chain used in the CCM literature.
The transverse cell is vacuum padded; increase ``--vacuum`` to test the
simulation-lattice embedding independently of the cyclic length.

This is a side-by-side diagnostic, not a mapped Γ-CCM/χ-CCM approach
comparison. It emits the approach status as ``not-defined`` and forms no
cross-construction energy delta.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import vibeqc as vq
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def alternating_h4_chain(
    vacuum_bohr: float,
) -> tuple[vq.PeriodicSystem, vq.BasisSet]:
    positions = [
        [x_angstrom * ANGSTROM_TO_BOHR, 0.0, 0.0]
        for x_angstrom in (0.0, 0.8, 2.0, 2.8)
    ]
    lattice = np.diag([4.0 * ANGSTROM_TO_BOHR, vacuum_bohr, vacuum_bohr])
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, position) for position in positions],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def options(max_iter: int, tolerance: float) -> vq.PeriodicRHFOptions:
    value = vq.PeriodicRHFOptions()
    value.max_iter = max_iter
    value.conv_tol_energy = tolerance
    value.damping = 0.0
    value.use_diis = True
    return value


def timed(
    callable_: Callable[[], Any],
) -> tuple[Any | None, float, str | None]:
    started = perf_counter()
    try:
        result = callable_()
        return result, perf_counter() - started, None
    except Exception as exc:  # benchmark must report both peers even if one fails
        return None, perf_counter() - started, f"{type(exc).__name__}: {exc}"


def run_ladder(
    meshes: Sequence[int],
    *,
    backend: str,
    vacuum: float,
    max_iter: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    system, basis = alternating_h4_chain(vacuum)
    records = []
    for length in meshes:
        mesh = (length, 1, 1)
        new, new_time, new_error = timed(
            lambda: vq.run_aiccm2026dev_b_rhf(
                system,
                basis,
                mesh,
                options(max_iter, tolerance),
                backend=backend,
                progress=False,
            )
        )
        reference, reference_time, reference_error = timed(
            lambda: vq.run_krhf_periodic_gdf(
                system,
                basis,
                vq.cyclic_gamma_mesh(system, mesh),
                options(max_iter, tolerance),
                progress=False,
            )
        )
        historical, historical_time, historical_error = timed(
            lambda: run_ccm_rhf(
                CCMSystem(system, mesh, "sto-3g"),
                method="union12",
                max_iter=max_iter,
                conv_tol=tolerance,
            )
        )
        peer, peer_time, peer_error = timed(
            lambda: run_ccm_rhf(
                CCMSystem(system, mesh, "sto-3g"),
                method="aiccm2026dev-a",
                max_iter=max_iter,
                conv_tol=tolerance,
            )
        )

        new_energy = None if new is None else float(new.energy)
        reference_energy = None if reference is None else float(reference.energy)
        historical_energy = (
            None
            if historical is None
            else float(historical.energy_per_atom * len(system.unit_cell))
        )
        peer_energy = (
            None
            if peer is None
            else float(peer.energy_per_atom * len(system.unit_cell))
        )
        diagnostics = None if new is None else new.aiccm2026dev_b
        convention = None if new is None else new.finite_torus_convention
        records.append(
            {
                "mesh": list(mesh),
                "aiccm2026dev-b": {
                    "ccm_approach": (
                        None if convention is None else convention.ccm_approach
                    ),
                    "ccm_construction": (
                        None
                        if convention is None
                        else convention.ccm_construction
                    ),
                    "evaluation_representation": (
                        None
                        if convention is None
                        else convention.evaluation_representation
                    ),
                    "finite_torus_convention": (
                        None if convention is None else asdict(convention)
                    ),
                    "backend": backend,
                    "energy_per_cell_ha": new_energy,
                    "converged": None if new is None else bool(new.converged),
                    "iterations": None if new is None else int(new.n_iter),
                    "seconds": new_time,
                    "idempotency_error": (
                        None
                        if diagnostics is None
                        else diagnostics.density_idempotency_error
                    ),
                    "electron_count_error": (
                        None if diagnostics is None else diagnostics.electron_count_error
                    ),
                    "error": new_error,
                },
                "direct_gamma_centred_mesh": {
                    "energy_per_cell_ha": reference_energy,
                    "converged": (
                        None if reference is None else bool(reference.converged)
                    ),
                    "iterations": (
                        None if reference is None else int(reference.n_iter)
                    ),
                    "seconds": reference_time,
                    "error": reference_error,
                },
                "historical_union12": {
                    "energy_per_cell_ha": historical_energy,
                    "converged": (
                        None if historical is None else bool(historical.converged)
                    ),
                    "iterations": (
                        None if historical is None else int(historical.n_iter)
                    ),
                    "seconds": historical_time,
                    "idempotency_error": (
                        None
                        if historical is None
                        else float(historical.idempotency_error)
                    ),
                    "error": historical_error,
                },
                "peer_aiccm2026dev_a": {
                    "energy_per_cell_ha": peer_energy,
                    "converged": None if peer is None else bool(peer.converged),
                    "iterations": None if peer is None else int(peer.n_iter),
                    "seconds": peer_time,
                    "idempotency_error": (
                        None if peer is None else float(peer.idempotency_error)
                    ),
                    "error": peer_error,
                },
                "gamma_ccm_chi_ccm_approach_comparison_status": "not-defined",
            }
        )
    return records


def print_table(records: Sequence[dict[str, Any]]) -> None:
    header = (
        "mesh       E(new) / Ha       E(kmesh) / Ha     E(old) / Ha       "
        "E(peer) / Ha      time new/old/peer / s"
    )
    print(
        "Γ-CCM/χ-CCM approach comparison: not-defined "
        "(side-by-side diagnostics only; no delta formed)"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        mesh = "x".join(str(value) for value in record["mesh"])
        new = record["aiccm2026dev-b"]
        reference = record["direct_gamma_centred_mesh"]
        historical = record["historical_union12"]
        peer = record["peer_aiccm2026dev_a"]

        def number(value: float | None) -> str:
            return "FAILED" if value is None else f"{value: .10f}"

        print(
            f"{mesh:<10} {number(new['energy_per_cell_ha']):>16} "
            f"{number(reference['energy_per_cell_ha']):>16} "
            f"{number(historical['energy_per_cell_ha']):>16} "
            f"{number(peer['energy_per_cell_ha']):>16} "
            f"{new['seconds']:.2f}/{historical['seconds']:.2f}/"
            f"{peer['seconds']:.2f}"
        )
        for label, payload in (
            ("new", new),
            ("kmesh", reference),
            ("historical", historical),
            ("peer", peer),
        ):
            if payload["error"]:
                print(f"  {label}: {payload['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", default="1,2,4")
    parser.add_argument(
        "--backend",
        choices=("four_center", "ri", "rijcosx"),
        default="ri",
        help=(
            "χ-CCM electron-repulsion backend. The default RI route runs "
            "beside the k-mesh control, but this unattested harness forms no delta."
        ),
    )
    parser.add_argument("--vacuum", type=float, default=40.0)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    meshes = [int(value) for value in args.meshes.split(",")]
    records = run_ladder(
        meshes,
        backend=args.backend,
        vacuum=args.vacuum,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
    )
    print_table(records)
    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
