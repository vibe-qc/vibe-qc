#!/usr/bin/env python3
"""Run one full χ-CCM / aiccm2026dev-b benchmark input and write JSON.

The benchmark launch/probe infrastructure is shared with the sibling stream,
but the electronic-structure implementation remains separate. All such calls
go through the finite-character (Γ-centred character-mesh) CCM API selected by
``aiccm2026dev-b``. The directory is self-contained for ``vq submit -d
studies/aiccm-2026/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import probe_host
import testset
import vibeqc as vq
from b_routes import (
    DIRECT_DOMAIN_POLICY,
    EWALD_SHIFTED_PAIR_CANARY_FIXTURE,
    EWALD_SHIFTED_PAIR_CANARY_SCHEMA,
    EWALD_SHIFTED_PAIR_CANARY_TOLERANCE,
    EWALD_SHIFTED_PAIR_IMPLEMENTATION,
    EWALD_SHIFTED_PAIR_REPAIR_COMMIT,
    EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA,
    OVERLAP_FOLD_DIAGNOSTIC,
    OVERLAP_FOLD_QUANTITATIVE_TARGET,
    OVERLAP_FOLD_REFERENCE_CUTOFF_FACTOR,
    OVERLAP_FOLD_STOP_THRESHOLD,
    OVERLAP_FOLD_SUPPORT_SCHEMA,
    ROUTES,
    TWO_ELECTRON_SUPPORT_SCHEMA,
    unsupported_reason,
)


#: Per-job path exported by ``run.sh`` inside a fresh private directory.
_attestation_path = os.environ.get("AICCM_PROBE_ATTESTATION")
_ATTESTATION = Path(_attestation_path) if _attestation_path else None


def _core_built_at() -> str:
    """Return the current core-path mtime as a human-readable audit field."""

    import datetime

    try:
        from vibeqc import _vibeqc_core

        core = Path(getattr(_vibeqc_core, "__file__", "") or "")
        if core.is_file():
            return datetime.datetime.fromtimestamp(core.stat().st_mtime).isoformat(
                timespec="seconds"
            )
    except Exception:  # noqa: BLE001 -- audit metadata must fail closed
        pass
    return "unknown"


def _provenance() -> dict[str, object]:
    """Record process-bound producer evidence before any calculation."""

    import platform

    commit, source_clean = probe_host.imported_source_state()
    provenance: dict[str, object] = {
        "host": (platform.node() or "unknown").split(".")[0],
        "vibeqc_version": str(getattr(vq, "__version__", "unknown")),
        "vibeqc_commit": commit,
        "source_clean": source_clean,
        "core_built": _core_built_at(),
        "core_build_id": probe_host.core_build_id(),
        "probe_version": "not-recorded",
        "probe_script_id": "unknown",
        "probe_passed": False,
        "probe_attestation_id": "unknown",
        "probe_attestation": None,
        "producer_payload_id": "unknown",
    }
    if os.environ.get("AICCM_SKIP_PROBE") != "1":
        attestation = probe_host.producer_attestation(
            _ATTESTATION,
            _producer_payload(),
        )
        if attestation is not None:
            provenance.update(
                host=attestation["host"],
                vibeqc_version=attestation["vibeqc_version"],
                vibeqc_commit=attestation["source_commit"],
                source_clean=attestation["source_clean"],
                core_build_id=attestation["core_build_id"],
                core_built=_core_built_at(),
            )
            provenance["probe_version"] = probe_host.PRODUCER_ATTESTATION_VERSION
            provenance["probe_script_id"] = attestation["attestor_script_id"]
            provenance["probe_passed"] = True
            provenance["probe_attestation_id"] = (
                probe_host.probe_attestation_id(attestation)
            )
            provenance["probe_attestation"] = attestation
            provenance["producer_payload_id"] = attestation[
                "producer_payload_id"
            ]
    return provenance


def _producer_payload() -> dict[str, object]:
    """Identify every executable input copied by the directory launcher."""

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


def _attestation_required() -> bool:
    return _ATTESTATION is not None and os.environ.get("AICCM_SKIP_PROBE") != "1"


def _require_attested_launch(provenance: dict[str, object]) -> None:
    if _attestation_required() and provenance.get("probe_passed") is not True:
        raise SystemExit(3)


def _finalize_provenance(provenance: dict[str, object]) -> bool:
    finalized = probe_host.finalize_producer_attestation(
        provenance.get("probe_attestation"),
        _producer_payload(),
    )
    provenance["core_built"] = _core_built_at()
    if finalized is None:
        provenance.update(
            probe_passed=False,
            probe_attestation_id="unknown",
            probe_attestation=None,
            producer_payload_id="unknown",
        )
    else:
        provenance.update(
            host=finalized["host"],
            vibeqc_version=finalized["vibeqc_version"],
            vibeqc_commit=finalized["source_commit"],
            source_clean=finalized["source_clean"],
            core_build_id=finalized["core_build_id"],
            probe_version=probe_host.PRODUCER_ATTESTATION_VERSION,
            probe_script_id=finalized["attestor_script_id"],
            probe_passed=finalized.get("probe_passed") is True,
            probe_attestation_id=probe_host.probe_attestation_id(finalized),
            probe_attestation=finalized,
            producer_payload_id=finalized["producer_payload_id"],
        )
    return provenance.get("probe_passed") is True


def _scf_options(method: str, args: argparse.Namespace):
    options = vq.PeriodicKSOptions() if method == "RKS" else vq.PeriodicRHFOptions()
    # ---- convergence controls ----
    options.max_iter = args.max_iter
    options.conv_tol_energy = args.energy_tol
    options.conv_tol_grad = args.gradient_tol
    # ---- DIIS family ----
    options.use_diis = args.use_diis
    options.diis_start_iter = args.diis_start
    options.diis_subspace_size = args.diis_subspace
    options.scf_accelerator = vq.scf_accelerator_from_string(args.scf_accelerator)
    options.ediis_diis_switch_threshold = args.ediis_diis_switch
    # ---- damping ----
    options.damping = args.damping
    options.dynamic_damping = args.dynamic_damping
    options.dynamic_damping_min = args.dynamic_damping_min
    options.dynamic_damping_max = args.dynamic_damping_max
    options.fock_mixing = args.fock_mixing
    # ---- level shift ----
    options.level_shift = args.level_shift
    options.level_shift_warmup_cycles = args.level_shift_warmup
    # ---- quadratic fallback (C1c Newton) ----
    options.quadratic_fallback_iter = args.quadratic_fallback_iter
    options.quadratic_fallback_shift = args.quadratic_fallback_shift
    options.quadratic_fallback_max_step = args.quadratic_fallback_max_step
    # ---- smearing ----
    options.smearing_temperature = args.smearing_temperature
    return options


def _orbital_properties(result: object, n_occ: int) -> dict[str, Any]:
    energies = [np.asarray(value, dtype=float) for value in result.mo_energies]
    occupied = np.concatenate([value[:n_occ] for value in energies])
    virtual = np.concatenate([value[n_occ:] for value in energies])
    homo = float(np.max(occupied)) if occupied.size else None
    lumo = float(np.min(virtual)) if virtual.size else None
    return {
        "n_kpoints": len(energies),
        "n_occupied_bands": n_occ,
        "homo_max_ha": homo,
        "lumo_min_ha": lumo,
        "fundamental_gap_ha": None if homo is None or lumo is None else lumo - homo,
    }


def _json_float(value: object | None) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _runtime_backend(result: object) -> str:
    """Return the executed backend label available on the result object."""

    value = getattr(result, "runtime_backend", None)
    if value is None:
        value = getattr(
            getattr(result, "hf_result", None),
            "runtime_backend",
            None,
        )
    return value if isinstance(value, str) and value.strip() else "not-recorded"


def _direct_two_electron_support(
    result: object,
    route_name: str,
    direct_lattice_cutoffs: dict[str, float],
) -> dict[str, Any]:
    """Serialize the resolved physical and M5 internal direct domains."""

    electronic = _json_float(direct_lattice_cutoffs["electronic_cutoff_bohr"])
    nuclear = _json_float(direct_lattice_cutoffs["nuclear_cutoff_bohr"])
    extent = _json_float(getattr(result, "sr_image_extent_bohr", None))
    precision = _json_float(getattr(result, "sr_image_precision", None))
    domain_policy = getattr(result, "sr_image_domain_policy", None)
    runtime_backend = _runtime_backend(result)
    qualified = (
        runtime_backend == "pbc-bipole"
        and domain_policy == DIRECT_DOMAIN_POLICY
        and precision == 1.0e-6
        and electronic is not None
        and electronic > 0.0
        and nuclear is not None
        and 0.0 < nuclear <= electronic
        and extent is not None
        and extent > electronic
    )
    return {
        "schema": TWO_ELECTRON_SUPPORT_SCHEMA,
        "qualification": "qualified" if qualified else "not-qualified",
        "reason": (
            "executed pbc-bipole M5 QQR-padded erfc image extent exceeds "
            "the coherent physical electronic and nuclear cutoffs"
            if qualified
            else "the executed four-center route did not expose one complete "
            "finite M5 QQR-padded direct-domain realization"
        ),
        "backend": ROUTES[route_name].backend,
        "runtime_backend": runtime_backend,
        "direct": {
            "domain_policy": domain_policy,
            "sr_image_precision": precision,
            "electronic_cutoff_bohr": electronic,
            "nuclear_cutoff_bohr": nuclear,
            "sr_image_extent_bohr": extent,
        },
        "fitted": None,
    }


def _fitted_two_electron_support(
    result: object,
    route_name: str,
    *,
    gdf_method: str,
    rsgdf_ke_cutoff: float,
    mdf_ke_cutoff: float,
) -> dict[str, Any]:
    """Serialize active fitted support without claiming tail convergence."""

    route = ROUTES[route_name]
    return {
        "schema": TWO_ELECTRON_SUPPORT_SCHEMA,
        "qualification": "not-qualified",
        "reason": (
            "no accepted high-G RSGDF tail, COSX exchange-domain, MDF, or "
            "post-HF correlation-factor support contract is active"
        ),
        "backend": route.backend,
        "runtime_backend": _runtime_backend(result),
        "direct": None,
        "fitted": {
            "gdf_method": gdf_method,
            "rsgdf_base_ke_cutoff_ha": (
                _json_float(rsgdf_ke_cutoff) if gdf_method == "rsgdf" else None
            ),
            "rsgdf_tail_ke_cutoff_ha": None,
            "mdf_ke_cutoff_ha": (
                _json_float(mdf_ke_cutoff) if gdf_method == "mdf" else None
            ),
            "cosx_exchange_support": (
                "not-attested" if route.backend == "rijcosx" else "not-applicable"
            ),
            "correlation_support": (
                "not-attested" if route.post_hf else "not-applicable"
            ),
        },
    }


def _overlap_fold_support(
    result: object,
    route_name: str,
    direct_lattice_cutoffs: dict[str, float] | None,
    mesh: tuple[int, int, int],
) -> dict[str, Any]:
    """Serialize direct corrected-gauge overlap-fold support evidence."""

    route = ROUTES[route_name]
    if route.backend != "four_center":
        return {
            "schema": OVERLAP_FOLD_SUPPORT_SCHEMA,
            "qualification": "not-applicable",
            "reason": (
                "the fitted backend does not use the corrected-gauge "
                "BIPOLE overlap fold"
            ),
            "backend": route.backend,
            "applicability": "not-applicable",
            "max_k_drift": None,
            "cutoff_bohr": None,
            "diagnostic": "not-applicable",
            "reference_cutoff_factor": None,
            "character_mesh_shape": None,
            "quantitative_target": OVERLAP_FOLD_QUANTITATIVE_TARGET,
            "stop_threshold": OVERLAP_FOLD_STOP_THRESHOLD,
        }

    cutoff = (
        None
        if direct_lattice_cutoffs is None
        else _json_float(direct_lattice_cutoffs.get("electronic_cutoff_bohr"))
    )
    drift = _json_float(getattr(result, "overlap_fold_drift", None))
    qualified = (
        cutoff is not None
        and cutoff > 0.0
        and drift is not None
        and 0.0 <= drift <= OVERLAP_FOLD_QUANTITATIVE_TARGET
    )
    if qualified:
        reason = (
            "executed corrected-gauge max-k overlap-fold drift meets the "
            "quantitative target"
        )
    elif drift is None:
        reason = "the executed corrected-gauge overlap-fold drift was not recorded"
    else:
        reason = (
            "the executed corrected-gauge max-k overlap-fold drift exceeds "
            "the quantitative target"
        )
    return {
        "schema": OVERLAP_FOLD_SUPPORT_SCHEMA,
        "qualification": "qualified" if qualified else "not-qualified",
        "reason": reason,
        "backend": route.backend,
        "applicability": "active",
        "max_k_drift": drift,
        "cutoff_bohr": cutoff,
        "diagnostic": OVERLAP_FOLD_DIAGNOSTIC,
        "reference_cutoff_factor": OVERLAP_FOLD_REFERENCE_CUTOFF_FACTOR,
        "character_mesh_shape": list(mesh),
        "quantitative_target": OVERLAP_FOLD_QUANTITATIVE_TARGET,
        "stop_threshold": OVERLAP_FOLD_STOP_THRESHOLD,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _repair_is_ancestor(producer_commit: object) -> bool:
    """Verify the D102 repair against the imported source checkout."""

    if not probe_host._is_full_commit(producer_commit):
        return False
    location = getattr(vq, "__file__", None)
    if not location:
        return False
    root = next(
        (
            parent
            for parent in Path(location).resolve().parents
            if (parent / ".git").exists()
        ),
        None,
    )
    if root is None:
        return False
    try:
        completed = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                EWALD_SHIFTED_PAIR_REPAIR_COMMIT,
                str(producer_commit),
            ],
            cwd=root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _ewald_shifted_pair_canary() -> dict[str, Any]:
    """Run a cheap same-process discriminator for the repaired native core."""

    record: dict[str, Any] = {
        "schema": EWALD_SHIFTED_PAIR_CANARY_SCHEMA,
        "fixture": EWALD_SHIFTED_PAIR_CANARY_FIXTURE,
        "real_cutoff_bohr": 18.0,
        "lattice_shift": [37, 0, 0],
        "energy_ha": None,
        "shifted_energy_ha": None,
        "abs_delta_ha": None,
        "tolerance_ha": EWALD_SHIFTED_PAIR_CANARY_TOLERANCE,
        "passed": False,
    }
    try:
        a_bohr = 4.211 / 0.529177210903
        lattice = 0.5 * a_bohr * np.asarray(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=float,
        )
        oxygen = 0.5 * (lattice[0] + lattice[1] + lattice[2])
        positions = np.column_stack(([0.0, 0.0, 0.0], oxygen))
        shifted = positions.copy()
        shifted[:, 1] += 37.0 * lattice[:, 0]
        charges = np.asarray([12.0, 8.0])
        options = vq.EwaldOptions()
        options.real_cutoff_bohr = 18.0
        energy = float(
            vq.ewald_point_charge_energy(
                lattice,
                positions,
                charges,
                options,
            )
        )
        shifted_energy = float(
            vq.ewald_point_charge_energy(
                lattice,
                shifted,
                charges,
                options,
            )
        )
        delta = abs(shifted_energy - energy)
        passed = (
            np.isfinite(energy)
            and np.isfinite(shifted_energy)
            and np.isfinite(delta)
            and delta <= EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
        )
        record.update(
            energy_ha=energy,
            shifted_energy_ha=shifted_energy,
            abs_delta_ha=delta,
            passed=bool(passed),
        )
    except Exception:  # noqa: BLE001 -- support evidence fails closed
        pass
    return record


def _ewald_shifted_pair_support(
    provenance: object,
    *,
    canary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the pair-complete Ewald repair to finalized D77 identity."""

    value = provenance if isinstance(provenance, dict) else {}
    producer_commit = value.get("vibeqc_commit")
    core_build_id = value.get("core_build_id")
    probe_attestation_id = value.get("probe_attestation_id")
    repair_is_ancestor = _repair_is_ancestor(producer_commit)
    if canary is None:
        canary = _ewald_shifted_pair_canary()
    qualified = (
        value.get("source_clean") is True
        and value.get("probe_passed") is True
        and probe_host._is_full_commit(producer_commit)
        and _is_sha256(core_build_id)
        and _is_sha256(probe_attestation_id)
        and repair_is_ancestor
        and canary.get("passed") is True
    )
    if qualified:
        reason = (
            "the finalized producer identity and same-process canary contain "
            "the pair-complete centered-displacement Ewald repair"
        )
    elif canary.get("passed") is not True:
        reason = (
            "the repair-specific same-process Ewald canary failed; rebuild "
            "the native core from the post-repair source"
        )
    else:
        reason = "the post-repair producer identity or ancestry is not attested"
    return {
        "schema": EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA,
        "qualification": "qualified" if qualified else "not-qualified",
        "reason": reason,
        "implementation": EWALD_SHIFTED_PAIR_IMPLEMENTATION,
        "repair_commit": EWALD_SHIFTED_PAIR_REPAIR_COMMIT,
        "producer_commit": (
            producer_commit
            if probe_host._is_full_commit(producer_commit)
            else None
        ),
        "core_build_id": core_build_id if _is_sha256(core_build_id) else None,
        "probe_attestation_id": (
            probe_attestation_id
            if _is_sha256(probe_attestation_id)
            else None
        ),
        "repair_is_ancestor": repair_is_ancestor,
        "canary": canary,
    }


def _ewald_shifted_pair_preflight_failure(
    support: object,
) -> str | None:
    """Reject a stale source/core pair before entering an expensive SCF."""

    if not isinstance(support, dict):
        return "post-repair Ewald support evidence was not assembled"
    if support.get("repair_is_ancestor") is not True:
        return "the loaded source identity does not attest the D104 Ewald repair"
    canary = support.get("canary")
    if not isinstance(canary, dict) or canary.get("passed") is not True:
        return (
            "the repair-specific same-process Ewald canary failed; rebuild "
            "the native core before running SCF"
        )
    return None


def _convention_record(result: object) -> dict[str, Any]:
    convention = getattr(result, "finite_torus_convention", None)
    if convention is None:
        diagnostics = getattr(result, "aiccm2026dev_b", None)
        convention = getattr(diagnostics, "finite_torus_convention", None)
    if convention is None:
        return {"status": "not-recorded"}
    return {
        "ccm_approach": convention.ccm_approach,
        "ccm_construction": convention.ccm_construction,
        "evaluation_representation": convention.evaluation_representation,
        "coulomb_kernel": convention.coulomb_kernel,
        "exchange_q0": convention.exchange_q0,
        "exchange_q0_applicability": convention.exchange_q0_applicability,
        "boundary_model": convention.boundary_model,
        "periodic_dimension": int(convention.periodic_dimension),
        "character_mesh_shape": list(convention.character_mesh_shape),
        "bvk_madelung_supercell_repetitions": list(
            convention.bvk_madelung_supercell_repetitions
        ),
        "bvk_madelung_supercell_lattice_bohr": [
            list(row) for row in convention.bvk_madelung_supercell_lattice_bohr
        ],
        "lattice_vector_convention": getattr(
            convention,
            "lattice_vector_convention",
            "not-recorded",
        ),
        "orbital_energy_convention": convention.orbital_energy_convention,
    }


def _direct_output_cell_farming_record(
    result: object,
) -> dict[str, Any] | None:
    """Serialize executed D114 output-cell scheduling evidence, if any."""

    diagnostics = getattr(result, "aiccm2026dev_b", None)
    execution = getattr(diagnostics, "direct_output_cell_farming", None)
    if execution is None:
        execution = getattr(result, "output_cell_farming_execution", None)
    if execution is None:
        return None
    return {
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


def _exact_exchange_assembly_record(result: object) -> dict[str, Any]:
    """Serialize route-resolved exchange metadata from the B reference."""

    candidates = [
        getattr(result, "exact_exchange_assembly", None),
        getattr(
            getattr(result, "aiccm2026dev_b", None),
            "exact_exchange_assembly",
            None,
        ),
    ]
    hf_result = getattr(result, "hf_result", None)
    if hf_result is not None:
        candidates.extend(
            [
                getattr(hf_result, "exact_exchange_assembly", None),
                getattr(
                    getattr(hf_result, "aiccm2026dev_b", None),
                    "exact_exchange_assembly",
                    None,
                ),
            ]
        )
    assembly = next(
        (candidate for candidate in candidates if candidate is not None),
        None,
    )
    if not isinstance(assembly, vq.AICCM2026DevBExactExchangeAssembly):
        raise RuntimeError(
            "aiccm2026dev-b result did not expose its route-resolved "
            "exact-exchange assembly"
        )
    return {
        "schema": assembly.schema,
        "resolver": assembly.resolver,
        "c_full": float(assembly.c_full),
        "c_sr": float(assembly.c_sr),
        "omega_screen_bohr_inv": float(assembly.omega_screen_bohr_inv),
        "screened_exchange_applicability": (
            assembly.screened_exchange_applicability
        ),
        "screened_exchange_assembly": assembly.screened_exchange_assembly,
    }


def _trace_field(item: object, name: str, index: int) -> object | None:
    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, (tuple, list)) and len(item) > index:
        return item[index]
    return None


def _scf_trace_tail(result: object, *, n_tail: int = 8) -> list[dict[str, Any]]:
    tail = list(getattr(result, "scf_trace", []) or [])[-n_tail:]
    records: list[dict[str, Any]] = []
    for fallback_iter, item in enumerate(tail, start=1):
        iter_value = _trace_field(item, "iter", 0)
        records.append(
            {
                "iter": int(iter_value) if iter_value is not None else fallback_iter,
                "energy_ha": _json_float(_trace_field(item, "energy", 1)),
                "delta_e_ha": _json_float(_trace_field(item, "delta_e", 2)),
                "grad_norm": _json_float(_trace_field(item, "grad_norm", 3)),
                "diis_subspace": (
                    None
                    if _trace_field(item, "diis_subspace", 4) is None
                    else int(_trace_field(item, "diis_subspace", 4))
                ),
            }
        )
    return records


def _damping_interaction_note(args: argparse.Namespace) -> str | None:
    if not args.use_diis or args.damping == 0.0:
        return None
    return (
        "In the current periodic GDF SCF loop, static density damping is "
        "bypassed from the DIIS start iteration onward; use --no-diis or a "
        "larger --diis-start when testing damping as an independent variable."
    )


def _scf_record(
    system: vq.PeriodicSystem,
    basis: vq.BasisSet,
    mesh: tuple[int, int, int],
    route_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    route = ROUTES[route_name]
    options = _scf_options(route.method, args)
    if route.backend == "four_center":
        cutoff_bohr = float(args.direct_cutoff_bohr)
        options.lattice_opts.cutoff_bohr = cutoff_bohr
        options.lattice_opts.nuclear_cutoff_bohr = cutoff_bohr
    common = dict(
        system=system,
        basis=basis,
        mesh=mesh,
        options=options,
        backend=route.backend,
        aux_basis=args.aux_basis,
        gdf_method=args.gdf_method,
        rsgdf_ke_cutoff=args.rsgdf_ke_cutoff,
        mdf_ke_cutoff=args.mdf_ke_cutoff,
        progress=args.progress,
    )
    if route.method == "RHF":
        result = vq.run_aiccm2026dev_b_rhf(**common)
    else:
        result = vq.run_aiccm2026dev_b_rks(functional=route.functional, **common)

    diagnostics = result.aiccm2026dev_b
    n_occ = int(system.n_electrons()) // 2
    trace_tail = _scf_trace_tail(result)
    direct_lattice_cutoffs = None
    if route.backend == "four_center":
        lattice_options = options.lattice_opts
        direct_lattice_cutoffs = {
            "electronic_cutoff_bohr": float(lattice_options.cutoff_bohr),
            "nuclear_cutoff_bohr": float(lattice_options.nuclear_cutoff_bohr),
        }
        two_electron_support = _direct_two_electron_support(
            result,
            route_name,
            direct_lattice_cutoffs,
        )
    else:
        two_electron_support = _fitted_two_electron_support(
            result,
            route_name,
            gdf_method=args.gdf_method,
            rsgdf_ke_cutoff=args.rsgdf_ke_cutoff,
            mdf_ke_cutoff=args.mdf_ke_cutoff,
        )
    return {
        "primitive_lattice_bohr": np.asarray(
            system.lattice,
            dtype=float,
        ).tolist(),
        "energy_per_cell_ha": float(result.energy),
        "electronic_energy_per_cell_ha": float(result.e_electronic),
        "nuclear_energy_per_cell_ha": float(result.e_nuclear),
        "converged": bool(result.converged),
        "iterations": int(result.n_iter),
        "finite_torus_convention": _convention_record(result),
        "exact_exchange_assembly": _exact_exchange_assembly_record(result),
        "direct_output_cell_farming": _direct_output_cell_farming_record(result),
        "direct_lattice_cutoffs": direct_lattice_cutoffs,
        "two_electron_support": two_electron_support,
        "overlap_fold_support": _overlap_fold_support(
            result,
            route_name,
            direct_lattice_cutoffs,
            mesh,
        ),
        "convergence_diagnostics": {
            "scf_trace_length": int(diagnostics.scf_trace_length),
            "final_delta_e_ha": _json_float(diagnostics.final_scf_delta_e_ha),
            "final_grad_norm": _json_float(diagnostics.final_scf_grad_norm),
            "final_diis_subspace": diagnostics.final_scf_diis_subspace,
            "scf_accelerator": diagnostics.scf_accelerator,
            "use_diis": diagnostics.use_diis,
            "diis_start_iter": diagnostics.diis_start_iter,
            "diis_subspace_size": diagnostics.diis_subspace_size,
            "damping": _json_float(diagnostics.damping),
            "dynamic_damping": diagnostics.dynamic_damping,
            "fock_mixing": _json_float(diagnostics.fock_mixing),
            "level_shift": _json_float(diagnostics.level_shift),
            "level_shift_warmup_cycles": diagnostics.level_shift_warmup_cycles,
            "smearing_temperature": _json_float(diagnostics.smearing_temperature),
            "damping_diis_note": _damping_interaction_note(args),
        },
        "scf_trace_tail": trace_tail,
        "properties": {
            **_orbital_properties(result, n_occ),
            "density_idempotency_error": float(diagnostics.density_idempotency_error),
            "electron_count_error": float(diagnostics.electron_count_error),
            "final_scf_delta_e_ha": _json_float(diagnostics.final_scf_delta_e_ha),
            "final_scf_grad_norm": _json_float(diagnostics.final_scf_grad_norm),
            "wigner_seitz_partition_error": float(
                diagnostics.wigner_seitz_partition_error
            ),
            "inverse_bloch_imaginary_residual": float(
                diagnostics.inverse_bloch_imaginary_residual
            ),
        },
    }


def _exact_dlpno_mp2_options():
    from vibeqc.dlpno.mp2 import DLPNOMP2Options

    return DLPNOMP2Options(
        localise="none",
        tcut_pno=0.0,
        tcut_pno_weak=0.0,
        tcut_mkn=0.0,
        tcut_pairs=0.0,
        tcut_pairs_weak=0.0,
        conv_tol_energy=1e-10,
        conv_tol_residual=1e-8,
    )


def _exact_cc_options(with_triples: bool):
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

    return LocalCCSDOptions(
        localise="none",
        tcut_pno=0.0,
        tcut_mkn=0.0,
        tcut_pairs=0.0,
        coupling_radius=0.0,
        residual_domain="full",
        compute_triples=with_triples,
        tcut_tno=0.0,
        triples_mode="exact",
        conv_tol_energy=1e-10,
        conv_tol_residual=1e-7,
    )


def _post_hf_record(
    system: vq.PeriodicSystem,
    basis: vq.BasisSet,
    mesh: tuple[int, int, int],
    route_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    options = _scf_options("RHF", args)
    common = dict(
        system=system,
        basis=basis,
        mesh=mesh,
        options=options,
        aux_basis=args.aux_basis,
        rsgdf_ke_cutoff=args.rsgdf_ke_cutoff,
        progress=args.progress,
    )
    if route_name == "ri-mp2":
        result = vq.run_aiccm2026dev_b_mp2(**common)
    elif route_name == "dlpno-mp2":
        local_options = None if args.local_mode == "pno" else _exact_dlpno_mp2_options()
        result = vq.run_aiccm2026dev_b_dlpno_mp2(
            dlpno_options=local_options,
            **common,
        )
    else:
        with_triples = route_name == "dlpno-ccsd-t"
        local_options = (
            None if args.local_mode == "pno" else _exact_cc_options(with_triples)
        )
        runner = (
            vq.run_aiccm2026dev_b_dlpno_ccsd_t
            if with_triples
            else vq.run_aiccm2026dev_b_dlpno_ccsd
        )
        result = runner(cc_options=local_options, **common)

    record = {
        "primitive_lattice_bohr": np.asarray(
            system.lattice,
            dtype=float,
        ).tolist(),
        "energy_per_cell_ha": float(result.energy),
        "hf_energy_per_cell_ha": float(result.e_hf_per_cell),
        "correlation_energy_per_cell_ha": float(result.e_corr_per_cell),
        "triples_energy_per_cell_ha": float(getattr(result, "e_t_per_cell", 0.0)),
        "converged": bool(getattr(result, "converged", result.hf_result.converged)),
        "iterations": int(getattr(result, "n_iter", result.hf_result.n_iter)),
        "finite_torus_convention": _convention_record(result),
        "exact_exchange_assembly": _exact_exchange_assembly_record(result),
        "direct_lattice_cutoffs": None,
        "two_electron_support": _fitted_two_electron_support(
            result,
            route_name,
            gdf_method="rsgdf",
            rsgdf_ke_cutoff=args.rsgdf_ke_cutoff,
            mdf_ke_cutoff=args.mdf_ke_cutoff,
        ),
        "overlap_fold_support": _overlap_fold_support(
            result,
            route_name,
            None,
            mesh,
        ),
        "properties": {
            "n_cyclic_cells": int(result.n_cyclic_cells),
            "n_pairs": getattr(result, "n_pairs", None),
            "n_pairs_screened": getattr(result, "n_pairs_screened", None),
            "t1_norm": getattr(result, "t1_norm", None),
            "localization": getattr(result, "localization", None),
            "cderi_imaginary_residual": getattr(
                result, "cderi_imaginary_residual", None
            ),
            "cderi_symmetry_residual": getattr(result, "cderi_symmetry_residual", None),
            "matrix_imaginary_residual": getattr(
                result, "matrix_imaginary_residual", None
            ),
            "momentum_conservation_error": getattr(
                result, "momentum_conservation_error", None
            ),
            "energy_imaginary_residual": getattr(
                result, "max_energy_imaginary_residual", None
            ),
        },
    }
    if route_name == "ri-mp2":
        record["same_spin_correlation_per_cell_ha"] = float(result.e_corr_ss_per_cell)
        record["opposite_spin_correlation_per_cell_ha"] = float(
            result.e_corr_os_per_cell
        )
    elif route_name == "dlpno-mp2":
        solver = result.solver_result
        record["properties"].update(
            pno_correction_per_cell_ha=float(solver.e_pno_correction)
            / result.n_cyclic_cells,
            distant_pair_energy_per_cell_ha=float(solver.e_distant)
            / result.n_cyclic_cells,
        )
    return record


def _write(
    path: Path,
    record: dict[str, Any],
    *,
    ewald_canary: dict[str, Any] | None = None,
) -> bool:
    provenance = record.get("provenance")
    trusted = isinstance(provenance, dict) and _finalize_provenance(provenance)
    quantitative_status = record.get("status") in ("ok", "not_converged")
    support_qualified = True
    if quantitative_status or ewald_canary is not None:
        support = _ewald_shifted_pair_support(
            provenance,
            canary=ewald_canary,
        )
        record["ewald_shifted_pair_support"] = support
        support_qualified = support.get("qualification") == "qualified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    identity_accepted = trusted or not _attestation_required()
    return identity_accepted and (not quantitative_status or support_qualified)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=sorted(testset.SYSTEMS))
    parser.add_argument("route", choices=tuple(ROUTES))
    parser.add_argument("--basis", default=None)
    parser.add_argument("--mesh", type=int, nargs=3, default=None)
    parser.add_argument("--aux-basis", default=None)
    parser.add_argument("--gdf-method", choices=("rsgdf", "mdf"), default="rsgdf")
    parser.add_argument("--rsgdf-ke-cutoff", type=float, default=200.0)
    parser.add_argument("--mdf-ke-cutoff", type=float, default=40.0)
    parser.add_argument("--direct-cutoff-bohr", type=float, default=15.0)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--energy-tol", type=float, default=1e-8)
    parser.add_argument("--gradient-tol", type=float, default=1e-6)
    # ---- SCF convergence options (bug-finding matrix) ----
    parser.add_argument(
        "--no-diis",
        action="store_false",
        dest="use_diis",
        help="Disable DIIS convergence acceleration",
    )
    parser.add_argument(
        "--diis-start", type=int, default=2, help="Iteration at which DIIS begins"
    )
    parser.add_argument(
        "--diis-subspace", type=int, default=8, help="DIIS subspace dimension (4-12)"
    )
    parser.add_argument(
        "--scf-accelerator",
        default="DIIS",
        choices=("DIIS", "KDIIS", "EDIIS", "EDIIS_DIIS", "ADIIS"),
        help="SCF accelerator family",
    )
    parser.add_argument(
        "--ediis-diis-switch",
        type=float,
        default=1e-2,
        help="EDIIS→DIIS switch threshold (energy gradient norm)",
    )
    parser.add_argument(
        "--damping", type=float, default=0.0, help="Static damping fraction (0 = off)"
    )
    parser.add_argument(
        "--dynamic-damping",
        action="store_true",
        default=False,
        help="Adaptive Zerner-Hehenberger density mixing",
    )
    parser.add_argument(
        "--dynamic-damping-min",
        type=float,
        default=0.0,
        help="Lower bound for dynamic-damping alpha",
    )
    parser.add_argument(
        "--dynamic-damping-max",
        type=float,
        default=0.95,
        help="Upper bound for dynamic-damping alpha",
    )
    parser.add_argument(
        "--fock-mixing",
        type=float,
        default=0.0,
        help="Fock/Kohn-Sham matrix mixing fraction. 0 = off.",
    )
    parser.add_argument(
        "--level-shift",
        type=float,
        default=0.0,
        help="Saunders-Hillier level shift (Hartree). 0 = off.",
    )
    parser.add_argument(
        "--level-shift-warmup",
        type=int,
        default=-1,
        help="Level-shift warmup cycles: -1=auto, 0=persistent, N=startup len",
    )
    parser.add_argument(
        "--quadratic-fallback-iter",
        type=int,
        default=0,
        help=(
            "Reserved quadratic-SCF activation; χ routes currently require "
            "0 because no B backend executes the fallback"
        ),
    )
    parser.add_argument(
        "--quadratic-fallback-shift",
        type=float,
        default=0.1,
        help="Reserved denominator damping (inactive while activation is 0)",
    )
    parser.add_argument(
        "--quadratic-fallback-max-step",
        type=float,
        default=0.1,
        help="Reserved trust-region cap (inactive while activation is 0)",
    )
    parser.add_argument(
        "--smearing-temperature",
        type=float,
        default=0.0,
        help="Fermi-Dirac smearing temperature (Hartree). 0 = off.",
    )
    parser.add_argument("--local-mode", choices=("exact", "pno"), default="exact")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", default=os.environ.get("VQ_WORKDIR", "."))
    args = parser.parse_args()

    provenance = _provenance()

    meta = testset.SYSTEMS[args.system]
    mesh = tuple(args.mesh or meta["nrep_4c"])
    route = ROUTES[args.route]
    reason = unsupported_reason(meta, args.route, mesh)
    out = Path(args.out)
    output = out / f"{args.system}__b-{args.route}.json"
    base = {
        "schema": "vibeqc.aiccm2026dev-b.benchmark/v1",
        "stream": "aiccm2026dev-b",
        "selector": "aiccm2026dev-b",
        "system": args.system,
        "route": args.route,
        "method": route.method,
        "backend": route.backend,
        "functional": route.functional,
        "basis": args.basis or meta["basis"],
        "aux_basis": args.aux_basis,
        "mesh": list(mesh),
        "dim": int(meta["dim"]),
        "class": meta["klass"],
        "atoms_per_cell": int(meta["atoms"]),
        "crystal_ref": meta["crystal_ref"],
        "direct_lattice_cutoffs": None,
        "provenance": provenance,
        "crystal_reference_basis_note": (
            "Compare only to a CRYSTAL calculation using this record's basis, "
            "functional, geometry, and reciprocal mesh."
        ),
        "local_mode": args.local_mode if route.post_hf else None,
        "scf_options": {
            "use_diis": args.use_diis,
            "diis_start": args.diis_start,
            "diis_subspace": args.diis_subspace,
            "scf_accelerator": args.scf_accelerator,
            "ediis_diis_switch": args.ediis_diis_switch,
            "damping": args.damping,
            "damping_diis_note": _damping_interaction_note(args),
            "dynamic_damping": args.dynamic_damping,
            "dynamic_damping_min": args.dynamic_damping_min,
            "dynamic_damping_max": args.dynamic_damping_max,
            "fock_mixing": args.fock_mixing,
            "level_shift": args.level_shift,
            "level_shift_warmup": args.level_shift_warmup,
            "quadratic_fallback_iter": args.quadratic_fallback_iter,
            "quadratic_fallback_shift": args.quadratic_fallback_shift,
            "quadratic_fallback_max_step": args.quadratic_fallback_max_step,
            "smearing_temperature": args.smearing_temperature,
        },
    }
    if reason:
        base.update(status="unsupported", converged=False, reason=reason)
        _write(output, base)
        raise SystemExit(3)

    _require_attested_launch(provenance)

    ewald_canary = _ewald_shifted_pair_canary()
    preflight_support = _ewald_shifted_pair_support(
        provenance,
        canary=ewald_canary,
    )
    preflight_failure = _ewald_shifted_pair_preflight_failure(preflight_support)
    if preflight_failure is not None:
        base.update(
            status="error",
            converged=False,
            error_type="EwaldShiftedPairSupportError",
            error=preflight_failure,
        )
        _write(output, base, ewald_canary=ewald_canary)
        raise SystemExit(3)

    started = time.perf_counter()
    try:
        system = testset.build(args.system)
        basis = vq.BasisSet(system.unit_cell_molecule(), base["basis"])
        payload = (
            _post_hf_record(system, basis, mesh, args.route, args)
            if route.post_hf
            else _scf_record(system, basis, mesh, args.route, args)
        )
        energy = float(payload["energy_per_cell_ha"])
        base.update(
            status="ok" if payload["converged"] else "not_converged",
            wall_time_s=time.perf_counter() - started,
            energy_per_atom_ha=energy / int(meta["atoms"]),
            **payload,
        )
        if not _write(output, base, ewald_canary=ewald_canary):
            raise SystemExit(3)
        if not payload["converged"]:
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        base.update(
            status="error",
            converged=False,
            wall_time_s=time.perf_counter() - started,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        _write(output, base, ewald_canary=ewald_canary)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
