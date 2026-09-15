#!/usr/bin/env python3
"""Validate two-rank χ direct output-cell farming for D114/D116.

Every rank runs the replicated, unfarmed BIPOLE reference and the farmed
``aiccm2026dev-b`` selector with identical 3-D c-diamond inputs.  D114 covers
RHF; D116 covers PBE RKS under its versioned numerical-parity contract.
The producer gathers compact per-rank execution evidence and permits only MPI
rank zero to atomically write the JSON artifact.  This is an
implementation-parity calculation, not an independently validated
absolute-energy benchmark.

Launch only through ``bash run.sh --d114-mpi`` or
``bash run.sh --d116-rks-mpi`` so the source/core attestation and mpi4py
preflight execute once before ``mpirun`` starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import probe_host
import testset
import vibeqc as vq
from vibeqc.mpi import mpi_allgather, mpi_bcast, mpi_rank, mpi_size
from vibeqc.periodic.chi.scf import cyclic_gamma_mesh


D114_SCHEMA = "vibeqc.aiccm2026dev-b.d114-mpi-validation/v1"
D116_RKS_SCHEMA = "vibeqc.aiccm2026dev-b.d116-rks-mpi-validation/v2"
D116_PARITY_CONTRACT_SCHEMA = (
    "vibeqc.aiccm2026dev-b.d116-parity-contract/v2"
)
# Historical public test/import alias for the original D114 producer.
SCHEMA = D114_SCHEMA
EXPECTED_TASK_KIND = "chi-direct-output-cell"
EXPECTED_DISTRIBUTION = "allgather-complete-blocks"
EXPECTED_FARMING_SCHEMA = (
    "vibeqc.pbc-bipole.output-cell-farming-execution/v1"
)
D114_PARALLEL_EXECUTION_SCHEMA = (
    "vibeqc.aiccm2026dev-b.d114-parallel-execution/v1"
)
PARALLEL_EXECUTION_SCHEMA = D114_PARALLEL_EXECUTION_SCHEMA
D116_PARALLEL_EXECUTION_SCHEMA = (
    "vibeqc.aiccm2026dev-b.d116-parallel-execution/v1"
)
ENERGY_TOLERANCE = 1.0e-8
GRADIENT_TOLERANCE = 1.0e-6
FOCK_MIXING = 0.0
SR_IMAGE_PRECISION = 1.0e-6
D116_ENERGY_PARITY_TOLERANCE_HA = 1.0e-10
D116_ARRAY_PARITY_TOLERANCE = 1.0e-12


def _d116_parity_contract() -> dict[str, object]:
    """Describe the versioned D116 floating/exact acceptance boundary."""

    return {
        "schema": D116_PARITY_CONTRACT_SCHEMA,
        "comparison": "absolute",
        "inclusive": True,
        "energy_components": ["total", "electronic", "nuclear", "xc"],
        "energy_abs_tolerance_ha": D116_ENERGY_PARITY_TOLERANCE_HA,
        "fock_element_abs_tolerance": D116_ARRAY_PARITY_TOLERANCE,
        "density_element_abs_tolerance": D116_ARRAY_PARITY_TOLERANCE,
        "exact_requirements": [
            "rank-and-task-census",
            "canonical-task-order",
            "ordered-cell-fingerprint",
            "array-block-shapes",
            "scf-controls",
            "scf-state",
            "exchange-q0-applicability",
            "parallel-execution-controls",
            "source-core-process-identity",
        ],
    }


def _producer_payload() -> dict[str, object]:
    """Bind every executable campaign input to the producer attestation."""

    here = Path(__file__).resolve().parent
    return {
        "version": probe_host.PRODUCER_PAYLOAD_VERSION,
        "stream": "aiccm2026dev-b",
        "files": {
            "producer": probe_host.file_sha256(__file__),
            "probe_host": probe_host.file_sha256(probe_host.__file__),
            "testset": probe_host.file_sha256(testset.__file__),
            "b_routes": probe_host.file_sha256(here / "b_routes.py"),
            "launcher": probe_host.file_sha256(here / "run.sh"),
        },
    }


def _provenance(attestation: dict[str, object]) -> dict[str, object]:
    """Return the same composite provenance aliases as campaign producers."""

    return {
        "host": attestation["host"],
        "vibeqc_version": attestation["vibeqc_version"],
        "vibeqc_commit": attestation["source_commit"],
        "source_clean": attestation["source_clean"],
        "core_build_id": attestation["core_build_id"],
        "probe_version": attestation["attestation_version"],
        "probe_script_id": attestation["attestor_script_id"],
        "probe_passed": attestation["probe_passed"],
        "probe_attestation_id": probe_host.probe_attestation_id(attestation),
        "producer_payload_id": attestation["producer_payload_id"],
        "probe_attestation": attestation,
    }


def _options(max_iter: int, cutoff_bohr: float) -> vq.PeriodicRHFOptions:
    options = vq.PeriodicRHFOptions()
    options.max_iter = int(max_iter)
    options.conv_tol_energy = ENERGY_TOLERANCE
    options.conv_tol_grad = GRADIENT_TOLERANCE
    options.fock_mixing = FOCK_MIXING
    options.lattice_opts.cutoff_bohr = float(cutoff_bohr)
    options.lattice_opts.nuclear_cutoff_bohr = float(cutoff_bohr)
    return options


def _ks_options(max_iter: int, cutoff_bohr: float) -> vq.PeriodicKSOptions:
    options = vq.PeriodicKSOptions()
    options.functional = "pbe"
    options.max_iter = int(max_iter)
    options.conv_tol_energy = ENERGY_TOLERANCE
    options.conv_tol_grad = GRADIENT_TOLERANCE
    options.fock_mixing = FOCK_MIXING
    options.lattice_opts.cutoff_bohr = float(cutoff_bohr)
    options.lattice_opts.nuclear_cutoff_bohr = float(cutoff_bohr)
    return options


def _validation_route(mode: str) -> dict[str, object]:
    """Resolve the launcher-owned calculation mode to immutable metadata."""

    routes: dict[str, dict[str, object]] = {
        "d114-rhf": {
            "schema": D114_SCHEMA,
            "parallel_schema": D114_PARALLEL_EXECUTION_SCHEMA,
            "milestone": "D114",
            "method": "RHF",
            "functional": None,
            "artifact": "chi-d114-c-diamond-rhf4c-mpi2.json",
        },
        "d116-rks-pbe": {
            "schema": D116_RKS_SCHEMA,
            "parallel_schema": D116_PARALLEL_EXECUTION_SCHEMA,
            "parity_contract": _d116_parity_contract(),
            "milestone": "D116",
            "method": "RKS",
            "functional": "PBE",
            "artifact": "chi-d116-c-diamond-rkspbe4c-mpi2.json",
        },
    }
    try:
        return routes[mode]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported χ output-cell MPI validation mode {mode!r}"
        ) from exc


def _validate_requested_inputs(
    mode: str,
    *,
    mesh: Sequence[int],
    basis: str,
    cutoff_bohr: float,
    max_iter: int,
) -> None:
    """Keep the D116 compute-cluster acceptance profile scientifically immutable."""

    if mode != "d116-rks-pbe":
        return
    if (
        tuple(int(value) for value in mesh) != (2, 2, 2)
        or basis.strip().lower() != "sto-3g"
        or float(cutoff_bohr) != 15.0
        or int(max_iter) != 1
    ):
        raise RuntimeError(
            "D116 fixes c-diamond/STO-3G, mesh 2 2 2, 15-bohr cutoffs, "
            "and one SCF cycle"
        )


def _run_reference_and_farmed(
    mode: str,
    system: object,
    basis: object,
    mesh: tuple[int, int, int],
    *,
    max_iter: int,
    cutoff_bohr: float,
) -> tuple[object, object]:
    """Execute the immutable unfarmed/farmed pair for one profile."""

    kmesh = cyclic_gamma_mesh(system, mesh).to_bloch_kmesh()
    if mode == "d114-rhf":
        reference = vq.run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            _options(max_iter, cutoff_bohr),
            fock_mixing=FOCK_MIXING,
            use_ewald_j_split=True,
            use_exchange_ewald_split=True,
            exchange_exxdiv="ewald",
            use_multipole_far_field=False,
            sr_image_precision=SR_IMAGE_PRECISION,
            farm_output_cells=False,
            progress=False,
        )
        farmed = vq.run_aiccm2026dev_b_rhf(
            system,
            basis,
            mesh,
            _options(max_iter, cutoff_bohr),
            backend="four_center",
            fock_mixing=FOCK_MIXING,
            progress=False,
        )
        return reference, farmed
    if mode == "d116-rks-pbe":
        reference = vq.run_pbc_bipole_rks(
            system,
            basis,
            kmesh,
            _ks_options(max_iter, cutoff_bohr),
            functional="pbe",
            fock_mixing=FOCK_MIXING,
            use_ewald_j_split=True,
            use_exchange_ewald_split=True,
            exchange_exxdiv="ewald",
            use_multipole_far_field=False,
            sr_image_precision=SR_IMAGE_PRECISION,
            farm_output_cells=False,
            use_fock_symmetry=False,
            use_fock_symmetry_reduce=False,
            progress=False,
        )
        farmed = vq.run_aiccm2026dev_b_rks(
            system,
            basis,
            "pbe",
            mesh,
            _ks_options(max_iter, cutoff_bohr),
            backend="four_center",
            fock_mixing=FOCK_MIXING,
            symmetry_mode="off",
            progress=False,
        )
        return reference, farmed
    raise RuntimeError(f"unsupported validation execution mode {mode!r}")


def _positive_int_environment(name: str) -> int:
    value = os.environ.get(name, "")
    if not value.isascii() or not value.isdecimal():
        raise RuntimeError(f"{name} must be a positive decimal integer")
    parsed = int(value)
    if parsed < 1:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _parallel_execution_record(mode: str | None = None) -> dict[str, Any]:
    """Validate the launcher-owned hybrid MPI/OpenMP execution contract."""

    selected_mode = mode or os.environ.get("AICCM_MPI_VALIDATION_PROFILE", "")
    route = _validation_route(selected_mode)
    launch_flags = {
        "d114-rhf": os.environ.get("AICCM_D114_MPI_LAUNCHED") == "1",
        "d116-rks-pbe": os.environ.get("AICCM_D116_MPI_LAUNCHED") == "1",
    }
    if launch_flags != {
        name: name == selected_mode for name in launch_flags
    }:
        raise RuntimeError(
            "χ output-cell MPI profile and launcher flags are inconsistent"
        )
    threads = _positive_int_environment("AICCM_MPI_THREADS_PER_RANK")
    declared_cpus = _positive_int_environment("AICCM_MPI_DECLARED_CPUS")
    launcher = os.environ.get("AICCM_MPI_LAUNCHER_KIND")
    binding = os.environ.get("AICCM_MPI_BINDING_POLICY")
    semantics = os.environ.get("AICCM_MPI_RESOURCE_SEMANTICS")
    if launcher not in {"intel-mpi", "open-mpi"}:
        raise RuntimeError("χ output-cell MPI launcher is not attested")
    expected_binding = {
        "intel-mpi": "intel-domain-omp-compact",
        "open-mpi": "openmpi-ppr-node-pe-core",
    }[launcher]
    if binding != expected_binding:
        raise RuntimeError("χ MPI binding policy contradicts the launcher")
    scheduler_tasks_raw = os.environ.get("VQ_SCHEDULER_TASKS")
    scheduler_tasks = (
        None
        if scheduler_tasks_raw in (None, "")
        else _positive_int_environment("VQ_SCHEDULER_TASKS")
    )
    if semantics == "vq-total-cpus-divided-by-ranks":
        if scheduler_tasks is not None or declared_cpus != 2 * threads:
            raise RuntimeError("χ total-CPU resource declaration is inconsistent")
    else:
        raise RuntimeError("χ MPI resource semantics are not attested")
    thread_environment = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    mismatches = {
        key: (os.environ.get(key), value)
        for key, value in thread_environment.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"χ thread environment is inconsistent: {mismatches}")
    native_threads = int(vq.get_num_threads())
    if native_threads != threads:
        raise RuntimeError(
            "χ native OpenMP thread count does not match the launcher contract"
        )
    return {
        "schema": route["parallel_schema"],
        "mpi_world_size": mpi_size(),
        "launcher_implementation": launcher,
        "binding_policy": binding,
        "resource_semantics": semantics,
        "declared_cpus": declared_cpus,
        "scheduler_tasks": scheduler_tasks,
        "omp_threads_per_rank": threads,
        "native_max_threads": native_threads,
        "thread_environment": thread_environment,
    }


def _execution_record(execution: object) -> dict[str, Any]:
    """Serialize and validate one rank's executed output-cell schedule."""

    if execution is None:
        raise RuntimeError("χ result did not expose output-cell telemetry")
    record = {
        "schema": execution.schema,
        "active": bool(execution.active),
        "task_kind": execution.task_kind,
        "strategy": execution.strategy,
        "world_size": int(execution.world_size),
        "rank": int(execution.rank),
        "global_task_count": int(execution.global_task_count),
        "local_task_count": int(execution.local_task_count),
        "local_task_counts": [
            int(value) for value in execution.local_task_counts
        ],
        "ordered_cell_fingerprint": execution.ordered_cell_fingerprint,
        "complete_internal_translation_sum": bool(
            execution.complete_internal_translation_sum
        ),
        "result_distribution": execution.result_distribution,
    }
    rank = mpi_rank()
    expected = {
        "schema": EXPECTED_FARMING_SCHEMA,
        "active": True,
        "task_kind": EXPECTED_TASK_KIND,
        "strategy": "cyclic",
        "world_size": 2,
        "rank": rank,
        "complete_internal_translation_sum": True,
        "result_distribution": EXPECTED_DISTRIBUTION,
    }
    mismatches = {
        key: (record[key], value)
        for key, value in expected.items()
        if record[key] != value
    }
    if mismatches:
        raise RuntimeError(f"invalid χ output-cell farming telemetry: {mismatches}")
    counts = record["local_task_counts"]
    if (
        len(counts) != 2
        or any(count <= 0 for count in counts)
        or sum(counts) != record["global_task_count"]
        or counts[rank] != record["local_task_count"]
    ):
        raise RuntimeError(
            "χ telemetry does not prove non-empty, complete two-rank ownership"
        )
    fingerprint = record["ordered_cell_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise RuntimeError("χ ordered-cell scheduling fingerprint is invalid")
    return record


def _max_abs_delta(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> float | None:
    if len(left) != len(right):
        return None
    value = max(
        (
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(left, right)
        ),
        default=0.0,
    )
    return value if np.isfinite(value) else None


def _finite_float(value: object) -> float | None:
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _within_abs_tolerance(value: object, tolerance: float) -> bool:
    if value is None or isinstance(value, (bool, np.bool_)):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(parsed) and abs(parsed) <= tolerance)


def _block_shapes(values: Sequence[np.ndarray]) -> list[list[int]]:
    return [list(np.asarray(value).shape) for value in values]


def _rank_parity_record(
    reference: object,
    farmed: object,
    *,
    require_xc: bool = False,
) -> dict[str, Any]:
    """Build one rank's farmed-versus-replicated parity evidence."""

    reference_fock = [np.asarray(value) for value in reference.fock]
    farmed_fock = [np.asarray(value) for value in farmed.fock]
    reference_density = [np.asarray(value) for value in reference.density.blocks]
    farmed_density = [np.asarray(value) for value in farmed.density.blocks]
    reference_fock_shapes = _block_shapes(reference_fock)
    farmed_fock_shapes = _block_shapes(farmed_fock)
    reference_density_shapes = _block_shapes(reference_density)
    farmed_density_shapes = _block_shapes(farmed_density)
    energy_delta = float(farmed.energy) - float(reference.energy)
    electronic_delta = float(farmed.e_electronic) - float(reference.e_electronic)
    nuclear_delta = float(farmed.e_nuclear) - float(reference.e_nuclear)
    xc_delta = (
        float(farmed.e_xc) - float(reference.e_xc)
        if require_xc
        else None
    )
    reference_fock_mixing = (
        float(reference.fock_mixing) if require_xc else None
    )
    farmed_fock_mixing = float(farmed.fock_mixing) if require_xc else None
    exchange_q0_applicability = (
        str(
            farmed.aiccm2026dev_b.finite_torus_convention
            .exchange_q0_applicability
        )
        if require_xc
        else None
    )
    fock_delta = (
        _max_abs_delta(farmed_fock, reference_fock)
        if farmed_fock_shapes == reference_fock_shapes
        else None
    )
    density_delta = (
        _max_abs_delta(farmed_density, reference_density)
        if farmed_density_shapes == reference_density_shapes
        else None
    )
    exact = bool(
        energy_delta == 0.0
        and electronic_delta == 0.0
        and nuclear_delta == 0.0
        and (xc_delta == 0.0 if require_xc else True)
        and (
            reference_fock_mixing == FOCK_MIXING
            and farmed_fock_mixing == FOCK_MIXING
            and exchange_q0_applicability == "inactive"
            if require_xc
            else True
        )
        and bool(farmed.converged) == bool(reference.converged)
        and int(farmed.n_iter) == int(reference.n_iter)
        and len(farmed_fock) == len(reference_fock)
        and len(farmed_density) == len(reference_density)
        and all(
            np.array_equal(got, want)
            for got, want in zip(farmed_fock, reference_fock)
        )
        and all(
            np.array_equal(got, want)
            for got, want in zip(farmed_density, reference_density)
        )
    )
    record = {
        "rank": mpi_rank(),
        "exact_parity": exact,
        "energy_delta_ha": _finite_float(energy_delta),
        "electronic_energy_delta_ha": _finite_float(electronic_delta),
        "nuclear_energy_delta_ha": _finite_float(nuclear_delta),
        "max_abs_fock_delta": fock_delta,
        "max_abs_density_delta": density_delta,
        "reference_energy_ha": _finite_float(reference.energy),
        "farmed_energy_ha": _finite_float(farmed.energy),
        "reference_nuclear_energy_ha": _finite_float(reference.e_nuclear),
        "farmed_nuclear_energy_ha": _finite_float(farmed.e_nuclear),
        "reference_converged": bool(reference.converged),
        "farmed_converged": bool(farmed.converged),
        "reference_iterations": int(reference.n_iter),
        "farmed_iterations": int(farmed.n_iter),
        "farming": _execution_record(
            farmed.aiccm2026dev_b.direct_output_cell_farming
        ),
    }
    if require_xc:
        contract_parity = bool(
            all(
                _within_abs_tolerance(
                    delta,
                    D116_ENERGY_PARITY_TOLERANCE_HA,
                )
                for delta in (
                    energy_delta,
                    electronic_delta,
                    nuclear_delta,
                    xc_delta,
                )
            )
            and _within_abs_tolerance(
                fock_delta,
                D116_ARRAY_PARITY_TOLERANCE,
            )
            and _within_abs_tolerance(
                density_delta,
                D116_ARRAY_PARITY_TOLERANCE,
            )
            and reference_fock_shapes == farmed_fock_shapes
            and reference_density_shapes == farmed_density_shapes
            and reference_fock_mixing == FOCK_MIXING
            and farmed_fock_mixing == FOCK_MIXING
            and exchange_q0_applicability == "inactive"
            and bool(farmed.converged) == bool(reference.converged)
            and int(farmed.n_iter) == int(reference.n_iter)
        )
        record.update(
            {
                "contract_parity": contract_parity,
                "xc_energy_delta_ha": _finite_float(xc_delta),
                "reference_electronic_energy_ha": _finite_float(
                    reference.e_electronic
                ),
                "farmed_electronic_energy_ha": _finite_float(
                    farmed.e_electronic
                ),
                "reference_xc_energy_ha": _finite_float(reference.e_xc),
                "farmed_xc_energy_ha": _finite_float(farmed.e_xc),
                "reference_fock_mixing": _finite_float(
                    reference_fock_mixing
                ),
                "farmed_fock_mixing": _finite_float(farmed_fock_mixing),
                "exchange_q0_applicability": exchange_q0_applicability,
                "reference_fock_block_shapes": reference_fock_shapes,
                "farmed_fock_block_shapes": farmed_fock_shapes,
                "reference_density_block_shapes": reference_density_shapes,
                "farmed_density_block_shapes": farmed_density_shapes,
            }
        )
    return record


def _require_numeric_rank_agreement(
    records: Sequence[dict[str, Any]],
    key: str,
    tolerance: float,
) -> None:
    values = [record.get(key) for record in records]
    if not all(_within_abs_tolerance(value, float("inf")) for value in values):
        raise RuntimeError(f"MPI rank value for {key} is not finite")
    baseline = float(values[0])
    if any(abs(float(value) - baseline) > tolerance for value in values[1:]):
        raise RuntimeError(f"MPI ranks disagree on {key}")


def _valid_block_shapes(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(
            isinstance(shape, list)
            and len(shape) == 2
            and all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in shape
            )
            for shape in value
        )
    )


def _validate_d116_rank_record(record: dict[str, Any]) -> None:
    """Recompute the D116 v2 verdict rather than trusting its boolean."""

    if not isinstance(record.get("exact_parity"), bool):
        raise RuntimeError("D116 exact_parity observation must be boolean")
    if record.get("contract_parity") is not True:
        raise RuntimeError("D116 parity contract failed on at least one rank")
    energy_fields = (
        ("reference_energy_ha", "farmed_energy_ha", "energy_delta_ha"),
        (
            "reference_electronic_energy_ha",
            "farmed_electronic_energy_ha",
            "electronic_energy_delta_ha",
        ),
        (
            "reference_nuclear_energy_ha",
            "farmed_nuclear_energy_ha",
            "nuclear_energy_delta_ha",
        ),
        (
            "reference_xc_energy_ha",
            "farmed_xc_energy_ha",
            "xc_energy_delta_ha",
        ),
    )
    for reference_key, farmed_key, delta_key in energy_fields:
        reference_value = record.get(reference_key)
        farmed_value = record.get(farmed_key)
        recorded_delta = record.get(delta_key)
        if not all(
            _within_abs_tolerance(value, float("inf"))
            for value in (reference_value, farmed_value, recorded_delta)
        ):
            raise RuntimeError(
                f"D116 {reference_key}, {farmed_key}, and {delta_key} "
                "must be finite"
            )
        observed_delta = float(farmed_value) - float(reference_value)
        if not _within_abs_tolerance(
            observed_delta,
            D116_ENERGY_PARITY_TOLERANCE_HA,
        ):
            raise RuntimeError(
                f"D116 {farmed_key} differs from {reference_key} beyond "
                "the absolute tolerance"
            )
        if float(recorded_delta) != observed_delta:
            raise RuntimeError(
                f"D116 {delta_key} contradicts {farmed_key} - {reference_key}"
            )
    for key in (
        "energy_delta_ha",
        "electronic_energy_delta_ha",
        "nuclear_energy_delta_ha",
        "xc_energy_delta_ha",
    ):
        if not _within_abs_tolerance(
            record.get(key),
            D116_ENERGY_PARITY_TOLERANCE_HA,
        ):
            raise RuntimeError(f"D116 {key} exceeds its absolute tolerance")
    for key in ("max_abs_fock_delta", "max_abs_density_delta"):
        if not _within_abs_tolerance(
            record.get(key),
            D116_ARRAY_PARITY_TOLERANCE,
        ):
            raise RuntimeError(f"D116 {key} exceeds its absolute tolerance")
    for prefix in ("fock", "density"):
        reference_shapes = record.get(f"reference_{prefix}_block_shapes")
        farmed_shapes = record.get(f"farmed_{prefix}_block_shapes")
        if not _valid_block_shapes(reference_shapes) or (
            reference_shapes != farmed_shapes
        ):
            raise RuntimeError(f"D116 {prefix} block shapes do not agree")
    convergence = (
        record.get("reference_converged"),
        record.get("farmed_converged"),
    )
    iterations = (
        record.get("reference_iterations"),
        record.get("farmed_iterations"),
    )
    if not all(isinstance(value, bool) for value in convergence):
        raise RuntimeError("D116 convergence state must be boolean")
    if not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == 1
        for value in iterations
    ):
        raise RuntimeError("D116 iteration state must equal the fixed one cycle")
    if convergence[0] != convergence[1] or iterations[0] != iterations[1]:
        raise RuntimeError("D116 SCF state does not agree within one rank")
    mixing = (
        record.get("reference_fock_mixing"),
        record.get("farmed_fock_mixing"),
    )
    if not all(
        _within_abs_tolerance(value, FOCK_MIXING) for value in mixing
    ):
        raise RuntimeError("D116 did not execute zero Fock mixing")
    if record.get("exchange_q0_applicability") != "inactive":
        raise RuntimeError("D116 PBE exchange-q=0 applicability is not inactive")


def _is_lower_hex(value: object, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_d116_farming_record(
    farming: dict[str, Any],
    rank: int,
) -> None:
    if (
        farming.get("active") is not True
        or farming.get("complete_internal_translation_sum") is not True
        or not isinstance(farming.get("world_size"), int)
        or isinstance(farming.get("world_size"), bool)
        or not isinstance(farming.get("rank"), int)
        or isinstance(farming.get("rank"), bool)
    ):
        raise RuntimeError("invalid D116 farming telemetry types")
    expected = {
        "schema": EXPECTED_FARMING_SCHEMA,
        "active": True,
        "task_kind": EXPECTED_TASK_KIND,
        "strategy": "cyclic",
        "world_size": 2,
        "rank": rank,
        "complete_internal_translation_sum": True,
        "result_distribution": EXPECTED_DISTRIBUTION,
    }
    mismatches = {
        key: (farming.get(key), value)
        for key, value in expected.items()
        if farming.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid D116 farming telemetry: {mismatches}")
    counts = farming.get("local_task_counts")
    global_count = farming.get("global_task_count")
    local_count = farming.get("local_task_count")
    if (
        not isinstance(counts, list)
        or len(counts) != 2
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for count in counts
        )
        or not isinstance(global_count, int)
        or isinstance(global_count, bool)
        or global_count != sum(counts)
        or not isinstance(local_count, int)
        or isinstance(local_count, bool)
        or local_count != counts[rank]
    ):
        raise RuntimeError("invalid D116 exact task census")
    if not _is_lower_hex(farming.get("ordered_cell_fingerprint"), 64):
        raise RuntimeError("invalid D116 ordered-cell fingerprint")


def _validate_d116_parallel_execution(record: dict[str, Any]) -> None:
    launcher = record.get("launcher_implementation")
    binding = record.get("binding_policy")
    expected_bindings = {
        "intel-mpi": "intel-domain-omp-compact",
        "open-mpi": "openmpi-ppr-node-pe-core",
    }
    if (
        record.get("schema") != D116_PARALLEL_EXECUTION_SCHEMA
        or record.get("mpi_world_size") != 2
        or launcher not in expected_bindings
        or binding != expected_bindings.get(launcher)
        or record.get("resource_semantics")
        != "vq-total-cpus-divided-by-ranks"
        or record.get("scheduler_tasks") is not None
    ):
        raise RuntimeError("invalid D116 parallel execution contract")
    threads = record.get("omp_threads_per_rank")
    declared_cpus = record.get("declared_cpus")
    native_threads = record.get("native_max_threads")
    if (
        not isinstance(threads, int)
        or isinstance(threads, bool)
        or threads <= 0
        or not isinstance(declared_cpus, int)
        or isinstance(declared_cpus, bool)
        or declared_cpus != 2 * threads
        or not isinstance(native_threads, int)
        or isinstance(native_threads, bool)
        or native_threads != threads
    ):
        raise RuntimeError("invalid D116 parallel thread allocation")
    expected_environment = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    if record.get("thread_environment") != expected_environment:
        raise RuntimeError("invalid D116 parallel thread environment")


def _validate_d116_process_identity(identity: dict[str, Any]) -> None:
    if not _is_lower_hex(identity.get("vibeqc_commit"), 40):
        raise RuntimeError("invalid D116 source commit identity")
    for key in ("core_build_id", "producer_attestation_id"):
        value = identity.get(key)
        if not isinstance(value, str) or not value.startswith("sha256:") or (
            not _is_lower_hex(value.removeprefix("sha256:"), 64)
        ):
            raise RuntimeError(f"invalid D116 {key}")


def _validate_rank_records(
    records: Sequence[dict[str, Any]],
    *,
    require_xc: bool = False,
) -> None:
    """Require contract-valid, mutually consistent evidence from both ranks."""

    ranks = [record.get("rank") for record in records]
    if (
        len(records) != 2
        or any(
            not isinstance(rank, int) or isinstance(rank, bool)
            for rank in ranks
        )
        or set(ranks) != {0, 1}
    ):
        raise RuntimeError("χ validation did not gather ranks 0 and 1 exactly once")
    if require_xc:
        for record in records:
            _validate_d116_rank_record(record)
    elif not all(record.get("exact_parity") is True for record in records):
        raise RuntimeError("χ farmed/unfarmed parity failed on at least one rank")
    farming = [record["farming"] for record in records]
    if require_xc:
        for record, item in zip(records, farming):
            rank = int(record["rank"])
            _validate_d116_farming_record(item, rank)
    invariant_keys = (
        "schema",
        "active",
        "task_kind",
        "strategy",
        "world_size",
        "global_task_count",
        "local_task_counts",
        "ordered_cell_fingerprint",
        "complete_internal_translation_sum",
        "result_distribution",
    )
    for key in invariant_keys:
        if len({json.dumps(item[key], sort_keys=True) for item in farming}) != 1:
            raise RuntimeError(f"χ ranks disagree on farming field {key!r}")
    if {item["rank"] for item in farming} != {0, 1}:
        raise RuntimeError("χ farming telemetry does not cover both ranks")
    if require_xc:
        for energy_key in (
            "reference_energy_ha",
            "farmed_energy_ha",
            "energy_delta_ha",
            "reference_nuclear_energy_ha",
            "farmed_nuclear_energy_ha",
            "nuclear_energy_delta_ha",
            "reference_xc_energy_ha",
            "farmed_xc_energy_ha",
            "xc_energy_delta_ha",
            "reference_electronic_energy_ha",
            "farmed_electronic_energy_ha",
            "electronic_energy_delta_ha",
        ):
            _require_numeric_rank_agreement(
                records,
                energy_key,
                D116_ENERGY_PARITY_TOLERANCE_HA,
            )
        for exact_key in (
            "reference_fock_mixing",
            "farmed_fock_mixing",
            "exchange_q0_applicability",
            "reference_converged",
            "farmed_converged",
            "reference_iterations",
            "farmed_iterations",
            "reference_fock_block_shapes",
            "farmed_fock_block_shapes",
            "reference_density_block_shapes",
            "farmed_density_block_shapes",
        ):
            if len(
                {
                    json.dumps(record.get(exact_key), sort_keys=True)
                    for record in records
                }
            ) != 1:
                raise RuntimeError(f"MPI ranks disagree on {exact_key}")
    else:
        for energy_key in (
            "reference_energy_ha",
            "farmed_energy_ha",
            "reference_nuclear_energy_ha",
            "farmed_nuclear_energy_ha",
        ):
            if len({record[energy_key] for record in records}) != 1:
                raise RuntimeError(f"MPI ranks disagree on {energy_key}")
    parallel_records = [record.get("parallel_execution") for record in records]
    if any(not isinstance(item, dict) for item in parallel_records):
        raise RuntimeError("at least one MPI rank lacks parallel execution evidence")
    if require_xc:
        for item in parallel_records:
            _validate_d116_parallel_execution(item)
    if len({json.dumps(item, sort_keys=True) for item in parallel_records}) != 1:
        raise RuntimeError("MPI ranks disagree on the parallel execution contract")
    identities = [record.get("process_identity") for record in records]
    if any(not isinstance(identity, dict) for identity in identities):
        raise RuntimeError("at least one MPI rank lacks finalized process identity")
    if require_xc:
        for identity in identities:
            _validate_d116_process_identity(identity)
    if len({json.dumps(identity, sort_keys=True) for identity in identities}) != 1:
        raise RuntimeError("MPI ranks disagree on source/core process identity")


def _atomic_write_rank_zero(path: Path, record: dict[str, Any]) -> None:
    if mpi_rank() != 0:
        raise RuntimeError("only MPI rank zero may write the χ validation artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--cutoff-bohr", type=float, default=15.0)
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument(
        "--out",
        default=os.environ.get("VQ_WORKDIR", "."),
        help="Artifact directory; defaults to VQ_WORKDIR or the current directory",
    )
    args = parser.parse_args()

    mode = os.environ.get("AICCM_MPI_VALIDATION_PROFILE", "")
    try:
        route = _validation_route(mode)
    except RuntimeError as exc:
        if mpi_rank() == 0:
            print(f"run_d114_mpi.py: {exc}", file=sys.stderr)
        raise SystemExit(4) from exc
    require_xc = route["functional"] is not None
    try:
        _validate_requested_inputs(
            mode,
            mesh=args.mesh,
            basis=args.basis,
            cutoff_bohr=args.cutoff_bohr,
            max_iter=args.max_iter,
        )
    except RuntimeError as exc:
        if mpi_rank() == 0:
            print(f"run_d114_mpi.py: {exc}", file=sys.stderr)
        raise SystemExit(4) from exc

    if mpi_size() != 2:
        if mpi_rank() == 0:
            print(
                "run_d114_mpi.py: exactly two active MPI ranks are required",
                file=sys.stderr,
            )
        raise SystemExit(4)
    parallel_execution = _parallel_execution_record(mode)

    payload = _producer_payload()
    attestation_path = os.environ.get("AICCM_PROBE_ATTESTATION")
    try:
        attestation = probe_host.producer_attestation(attestation_path, payload)
        attestation_error = None
    except Exception as exc:  # noqa: BLE001 -- coordinate failure across ranks
        attestation = None
        attestation_error = f"{type(exc).__name__}: {exc}"
    attestation_states = mpi_allgather(
        {
            "rank": mpi_rank(),
            "passed": attestation is not None,
            "error": attestation_error,
        }
    )
    if not all(state["passed"] for state in attestation_states):
        if mpi_rank() == 0:
            print(
                "run_d114_mpi.py: producer attestation failed: "
                + json.dumps(attestation_states, sort_keys=True),
                file=sys.stderr,
            )
        raise SystemExit(3)
    assert attestation is not None

    mesh = tuple(int(value) for value in args.mesh)
    system = testset.build("c-diamond")
    basis = vq.BasisSet(system.unit_cell_molecule(), args.basis)

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            vq.AICCM2026DevBExperimentalWarning,
        )
        reference, farmed = _run_reference_and_farmed(
            mode,
            system,
            basis,
            mesh,
            max_iter=args.max_iter,
            cutoff_bohr=args.cutoff_bohr,
        )

    try:
        local_record = _rank_parity_record(
            reference,
            farmed,
            require_xc=require_xc,
        )
        local_record["parallel_execution"] = parallel_execution
    except Exception as exc:  # noqa: BLE001 -- gather one coherent failure
        local_record = {
            "rank": mpi_rank(),
            "exact_parity": False,
            "validation_error": f"{type(exc).__name__}: {exc}",
        }
        if require_xc:
            local_record["contract_parity"] = False
    try:
        finalized = probe_host.finalize_producer_attestation(attestation, payload)
    except Exception as exc:  # noqa: BLE001 -- gather one coherent failure
        finalized = None
        local_record["attestation_error"] = f"{type(exc).__name__}: {exc}"
    if finalized is None or finalized.get("probe_passed") is not True:
        local_record.setdefault(
            "attestation_error",
            "identity changed during calculation",
        )
        local_record["exact_parity"] = False
        if require_xc:
            local_record["contract_parity"] = False
    else:
        local_record["process_identity"] = {
            "vibeqc_commit": finalized["source_commit"],
            "core_build_id": finalized["core_build_id"],
            "producer_attestation_id": probe_host.probe_attestation_id(finalized),
        }
    rank_records = mpi_allgather(local_record)

    root_outcome: dict[str, Any] | None = None
    if mpi_rank() == 0:
        output = Path(args.out) / str(route["artifact"])
        try:
            record = {
                "schema": route["schema"],
                "status": "pending-validation",
                "scope": (
                    f"{route['milestone']} implementation parity only; "
                    "not an independent "
                    "absolute-energy benchmark"
                ),
                "selector": "aiccm2026dev-b",
                "system": "c-diamond",
                "method": route["method"],
                "backend": "four_center",
                "mesh": list(mesh),
                "basis": args.basis,
                "cutoff_bohr": float(args.cutoff_bohr),
                "max_iter": int(args.max_iter),
                "world_size": 2,
                "progress": False,
                "parallel_execution": parallel_execution,
                "scf_controls": {
                    "energy_tolerance_ha": ENERGY_TOLERANCE,
                    "gradient_tolerance": GRADIENT_TOLERANCE,
                    "fock_mixing": FOCK_MIXING,
                    "use_ewald_j_split": True,
                    "use_exchange_ewald_split": True,
                    "exchange_exxdiv": "ewald",
                    "use_multipole_far_field": False,
                    "sr_image_precision": SR_IMAGE_PRECISION,
                },
                "wall_time_s": time.perf_counter() - started,
                "provenance": (
                    _provenance(finalized) if finalized is not None else None
                ),
                "rank_records": rank_records,
            }
            if route["functional"] is not None:
                record["functional"] = route["functional"]
                record["parity_contract"] = route["parity_contract"]
                record["scope"] = (
                    "D116 implementation parity only; not an independent "
                    "absolute-energy or whole-SCF scaling benchmark"
                )
                record["scf_controls"].update(
                    {
                        "functional": "pbe",
                        "exchange_q0_applicability": "inactive",
                        "symmetry_mode": "off",
                        "use_fock_symmetry": False,
                        "use_fock_symmetry_reduce": False,
                    }
                )
            try:
                _validate_rank_records(rank_records, require_xc=require_xc)
            except Exception as exc:  # noqa: BLE001 -- persist failure evidence
                record["status"] = "failed"
                record["validation_error"] = f"{type(exc).__name__}: {exc}"
            else:
                record["status"] = "passed"
            _atomic_write_rank_zero(output, record)
            root_outcome = {
                "passed": record["status"] == "passed",
                "artifact": str(output),
            }
            if record["status"] != "passed":
                root_outcome["error"] = record["validation_error"]
        except Exception as exc:  # noqa: BLE001 -- broadcast one rank-safe verdict
            root_outcome = {"passed": False, "error": str(exc)}
    outcome = mpi_bcast(root_outcome, root=0)
    if mpi_rank() == 0:
        print(json.dumps(outcome, sort_keys=True))
    if not isinstance(outcome, dict) or outcome.get("passed") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
