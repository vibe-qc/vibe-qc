"""Contract tests for the independently runnable B-stream fleet inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_TESTSET_DIR = _ROOT / "studies" / "aiccm-2026"
sys.path.insert(0, str(_TESTSET_DIR))

import audit_b  # noqa: E402
import audit_cmp_b  # noqa: E402
import b_routes  # noqa: E402
import compare as legacy_compare  # noqa: E402
import compare_b  # noqa: E402
import curate  # noqa: E402
import make_jobs_b  # noqa: E402
import probe_host  # noqa: E402
import run_case  # noqa: E402
import run_case_b  # noqa: E402
import run_case_cmp  # noqa: E402
import run_case_cmp_b  # noqa: E402
import run_d114_mpi  # noqa: E402
import testset  # noqa: E402
import site_settings  # noqa: E402
from examples.periodic import (  # noqa: E402
    aiccm2026dev_diamond_bonds_bands_compare as diamond_compare,
)
from vibeqc.periodic.chi.scf import (  # noqa: E402
    AICCM2026DevBExactExchangeAssembly,
    AICCM2026DevBFiniteTorusConvention,
    cyclic_gamma_mesh,
)


@pytest.fixture(autouse=True)
def synthetic_study_site(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise configured routing without any operator or fleet identifiers."""
    profile = json.loads((_TESTSET_DIR / "site-profile.example.json").read_text())
    profile["allow_submission"] = True
    monkeypatch.setattr(site_settings, "_profile", site_settings.validate_profile(profile))


_SOURCE_COMMIT = "a" * 40
_CURRENT_CORE_ID = "sha256:" + "b" * 64
_OLD_CORE_ID = "sha256:" + "d" * 64
_PROBE_SCRIPT_ID = "sha256:" + "c" * 64
_NATIVE_VERSIONS = {
    "libint": "2.13.1",
    "libxc": "7.0.0",
    "spglib": "2.7.0",
    "libecpint": "1.0.7",
    "fftw3": "3.3.9",
    "blas": "test-blas",
}
_EXPECTED_EXACT_EXCHANGE_RESOLVER = (
    "vibeqc.periodic_screened_exchange.resolve_periodic_exchange"
)
_EXPECTED_EXACT_EXCHANGE_SCHEMA = (
    "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
)
_EXPECTED_EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE = {
    "rhf-4c": (1.0, 0.0, 0.0),
    "rhf-ri": (1.0, 0.0, 0.0),
    "rhf-rijcosx": (1.0, 0.0, 0.0),
    "rks-pbe-4c": (0.0, 0.0, 0.0),
    "rks-pbe-ri": (0.0, 0.0, 0.0),
    "rks-pbe-rijcosx": (0.0, 0.0, 0.0),
    "rks-pbe0-4c": (0.25, 0.0, 0.0),
    "rks-pbe0-ri": (0.25, 0.0, 0.0),
    "rks-pbe0-rijcosx": (0.25, 0.0, 0.0),
    "ri-mp2": (1.0, 0.0, 0.0),
    "dlpno-mp2": (1.0, 0.0, 0.0),
    "dlpno-ccsd": (1.0, 0.0, 0.0),
    "dlpno-ccsd-t": (1.0, 0.0, 0.0),
}
_EXPECTED_B_ROUTES = {
    "rhf-4c": b_routes.Route(
        "RHF", "four_center", None, False, "active"
    ),
    "rhf-ri": b_routes.Route("RHF", "ri", None, False, "active"),
    "rhf-rijcosx": b_routes.Route(
        "RHF", "rijcosx", None, False, "active"
    ),
    "rks-pbe-4c": b_routes.Route(
        "RKS", "four_center", "pbe", False, "inactive"
    ),
    "rks-pbe-ri": b_routes.Route(
        "RKS", "ri", "pbe", False, "inactive"
    ),
    "rks-pbe-rijcosx": b_routes.Route(
        "RKS", "rijcosx", "pbe", False, "inactive"
    ),
    "rks-pbe0-4c": b_routes.Route(
        "RKS", "four_center", "pbe0", False, "active"
    ),
    "rks-pbe0-ri": b_routes.Route(
        "RKS", "ri", "pbe0", False, "active"
    ),
    "rks-pbe0-rijcosx": b_routes.Route(
        "RKS", "rijcosx", "pbe0", False, "active"
    ),
    "ri-mp2": b_routes.Route("RI-MP2", "ri", None, True, "active"),
    "dlpno-mp2": b_routes.Route(
        "DLPNO-MP2", "ri", None, True, "active"
    ),
    "dlpno-ccsd": b_routes.Route(
        "DLPNO-CCSD", "ri", None, True, "active"
    ),
    "dlpno-ccsd-t": b_routes.Route(
        "DLPNO-CCSD(T)", "ri", None, True, "active"
    ),
}


def _qualified_direct_support() -> dict[str, Any]:
    return {
        "schema": b_routes.TWO_ELECTRON_SUPPORT_SCHEMA,
        "qualification": "qualified",
        "reason": (
            "executed pbc-bipole M5 QQR-padded erfc image extent exceeds "
            "the coherent physical electronic and nuclear cutoffs"
        ),
        "backend": "four_center",
        "runtime_backend": "pbc-bipole",
        "direct": {
            "domain_policy": b_routes.DIRECT_DOMAIN_POLICY,
            "sr_image_precision": b_routes.DIRECT_SR_IMAGE_PRECISION,
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
            "sr_image_extent_bohr": 28.0,
        },
        "fitted": None,
    }


def _qualified_overlap_fold_support(
    route_name: str = "rhf-4c",
    *,
    drift: float = 5.0e-5,
    cutoff: float = 15.0,
    mesh: tuple[int, int, int] = (2, 2, 2),
) -> dict[str, Any]:
    return {
        "schema": b_routes.OVERLAP_FOLD_SUPPORT_SCHEMA,
        "qualification": "qualified",
        "reason": (
            "executed corrected-gauge max-k overlap-fold drift meets the "
            "quantitative target"
        ),
        "backend": b_routes.ROUTES[route_name].backend,
        "applicability": "active",
        "max_k_drift": drift,
        "cutoff_bohr": cutoff,
        "diagnostic": b_routes.OVERLAP_FOLD_DIAGNOSTIC,
        "reference_cutoff_factor": (
            b_routes.OVERLAP_FOLD_REFERENCE_CUTOFF_FACTOR
        ),
        "character_mesh_shape": list(mesh),
        "quantitative_target": b_routes.OVERLAP_FOLD_QUANTITATIVE_TARGET,
        "stop_threshold": b_routes.OVERLAP_FOLD_STOP_THRESHOLD,
    }


def _not_applicable_overlap_fold_support(route_name: str) -> dict[str, Any]:
    return {
        "schema": b_routes.OVERLAP_FOLD_SUPPORT_SCHEMA,
        "qualification": "not-applicable",
        "reason": (
            "the fitted backend does not use the corrected-gauge BIPOLE "
            "overlap fold"
        ),
        "backend": b_routes.ROUTES[route_name].backend,
        "applicability": "not-applicable",
        "max_k_drift": None,
        "cutoff_bohr": None,
        "diagnostic": "not-applicable",
        "reference_cutoff_factor": None,
        "character_mesh_shape": None,
        "quantitative_target": b_routes.OVERLAP_FOLD_QUANTITATIVE_TARGET,
        "stop_threshold": b_routes.OVERLAP_FOLD_STOP_THRESHOLD,
    }


def _qualified_ewald_shifted_pair_support(
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = provenance or _producer_provenance("aiccm2026dev-b")
    producer_commit = value.get("vibeqc_commit", value.get("source_commit"))
    return {
        "schema": b_routes.EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA,
        "qualification": "qualified",
        "reason": (
            "the finalized producer identity and same-process canary contain "
            "the pair-complete centered-displacement Ewald repair"
        ),
        "implementation": b_routes.EWALD_SHIFTED_PAIR_IMPLEMENTATION,
        "repair_commit": b_routes.EWALD_SHIFTED_PAIR_REPAIR_COMMIT,
        "producer_commit": producer_commit,
        "core_build_id": value["core_build_id"],
        "probe_attestation_id": value["probe_attestation_id"],
        "repair_is_ancestor": True,
        "canary": {
            "schema": b_routes.EWALD_SHIFTED_PAIR_CANARY_SCHEMA,
            "fixture": b_routes.EWALD_SHIFTED_PAIR_CANARY_FIXTURE,
            "real_cutoff_bohr": 18.0,
            "lattice_shift": [37, 0, 0],
            "energy_ha": -73.0,
            "shifted_energy_ha": -73.0,
            "abs_delta_ha": 0.0,
            "tolerance_ha": (
                b_routes.EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
            ),
            "passed": True,
        },
    }


def _legacy_unqualified_ewald_shifted_pair_support_v1() -> dict[str, Any]:
    return {
        "schema": "vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v1",
        "qualification": "not-qualified",
        "reason": "the pre-repair helper can omit shifted pairs",
        "implementation": (
            "unshifted-radius-preselection-before-shifted-pair-filter"
        ),
        "repair_commit": None,
    }


def _unqualified_fitted_support(
    route_name: str,
    *,
    gdf_method: str = "rsgdf",
) -> dict[str, Any]:
    route = b_routes.ROUTES[route_name]
    reference_method = "rhf" if route.post_hf else route.method.lower()
    exchange_backend = "cosx" if route.backend == "rijcosx" else "gdf"
    return {
        "schema": b_routes.TWO_ELECTRON_SUPPORT_SCHEMA,
        "qualification": "not-qualified",
        "reason": "high-G and correlation support are not attested",
        "backend": route.backend,
        "runtime_backend": (
            f"native-multi-k-gdf-{exchange_backend}-{reference_method}"
        ),
        "direct": None,
        "fitted": {
            "gdf_method": gdf_method,
            "rsgdf_base_ke_cutoff_ha": 200.0 if gdf_method == "rsgdf" else None,
            "rsgdf_tail_ke_cutoff_ha": None,
            "mdf_ke_cutoff_ha": 40.0 if gdf_method == "mdf" else None,
            "cosx_exchange_support": (
                "not-attested" if route.backend == "rijcosx" else "not-applicable"
            ),
            "correlation_support": (
                "not-attested" if route.post_hf else "not-applicable"
            ),
        },
    }


def _rhf_exact_exchange_assembly() -> AICCM2026DevBExactExchangeAssembly:
    return AICCM2026DevBExactExchangeAssembly(
        c_full=1.0,
        c_sr=0.0,
        omega_screen_bohr_inv=0.0,
    )


def _fleet_exact_exchange_assembly(route_name: str) -> dict[str, object]:
    coefficients = _EXPECTED_EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE[route_name]
    return {
        "schema": _EXPECTED_EXACT_EXCHANGE_SCHEMA,
        "c_full": coefficients[0],
        "c_sr": coefficients[1],
        "omega_screen_bohr_inv": coefficients[2],
        "resolver": _EXPECTED_EXACT_EXCHANGE_RESOLVER,
        "screened_exchange_applicability": "inactive",
        "screened_exchange_assembly": "not-applicable",
    }


def _d89_chi_convention() -> AICCM2026DevBFiniteTorusConvention:
    return AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic",
        periodic_dimension=3,
        character_mesh_shape=(2, 2, 2),
        bvk_madelung_supercell_repetitions=(2, 2, 2),
        bvk_madelung_supercell_lattice_bohr=(
            (8.0, 0.0, 0.0),
            (0.0, 8.0, 0.0),
            (0.0, 0.0, 8.0),
        ),
    )


def _d89_chi_result(electronic_method: str = "RHF") -> SimpleNamespace:
    convention = _d89_chi_convention()
    exact_exchange_assembly = _rhf_exact_exchange_assembly()
    return SimpleNamespace(
        backend="aiccm2026dev-b-ri",
        finite_torus_convention=convention,
        exact_exchange_assembly=exact_exchange_assembly,
        aiccm2026dev_b=SimpleNamespace(
            finite_torus_convention=convention,
            exact_exchange_assembly=exact_exchange_assembly,
            backend="ri",
            electronic_method=electronic_method,
            mesh=(2, 2, 2),
            n_cyclic_cells=8,
            n_kpoints=8,
        ),
    )


def _preflight_attestation(*, core_id: str = _CURRENT_CORE_ID) -> dict[str, Any]:
    return {
        "probe_version": probe_host.PROBE_VERSION,
        "probe_script_id": _PROBE_SCRIPT_ID,
        "probe_passed": True,
        "host": "verified-host",
        "vibeqc_version": "0.15.34",
        "core_build_id": core_id,
        "source_commit": _SOURCE_COMMIT,
        "source_clean": True,
    }


def _producer_payload(stream: str) -> dict[str, Any]:
    files = {
        "producer": "sha256:" + "1" * 64,
        "probe_host": _PROBE_SCRIPT_ID,
        "testset": "sha256:" + "3" * 64,
        "launcher": "sha256:" + "4" * 64,
    }
    if stream == "aiccm2026dev-b":
        files["b_routes"] = "sha256:" + "5" * 64
    return {
        "version": probe_host.PRODUCER_PAYLOAD_VERSION,
        "stream": stream,
        "files": files,
    }


def _loaded_core_check() -> dict[str, Any]:
    return {
        "version": probe_host.LOADED_CORE_CHECK_VERSION,
        "passed": True,
        "fixture": "lih-sto3g-shifted-mesh/v1",
        "n_ao": 6,
        "max_l": 1,
        "n_cells": 3,
        "n_frequencies": 3,
        "atol": probe_host.CANARY_ATOL,
        "rtol": probe_host.CANARY_RTOL,
        "max_abs_cxx_python": 1.0e-15,
        "max_scaled_error": 1.0e-5,
        "reference_transpose_asymmetry": 0.38,
    }


def _composite_attestation(
    stream: str,
    *,
    preflight_core_id: str = _CURRENT_CORE_ID,
) -> dict[str, Any]:
    preflight = _preflight_attestation(core_id=preflight_core_id)
    current = _preflight_attestation()
    payload = _producer_payload(stream)
    return {
        "attestation_version": probe_host.PRODUCER_ATTESTATION_VERSION,
        "attestor_script_id": _PROBE_SCRIPT_ID,
        "probe_passed": True,
        "host": current["host"],
        "vibeqc_version": current["vibeqc_version"],
        "core_build_id": current["core_build_id"],
        "source_commit": current["source_commit"],
        "source_clean": current["source_clean"],
        "native_library_versions": dict(_NATIVE_VERSIONS),
        "producer_payload": payload,
        "producer_payload_id": probe_host.probe_attestation_id(payload),
        "preflight_attestation": preflight,
        "preflight_attestation_id": probe_host.probe_attestation_id(preflight),
        "current_core_attestation": current,
        "current_core_attestation_id": probe_host.probe_attestation_id(current),
        "core_changed_since_preflight": preflight_core_id != _CURRENT_CORE_ID,
        "loaded_core_check": _loaded_core_check(),
        "result_identity_stable": True,
    }


def _producer_provenance(
    stream: str,
    *,
    preflight_core_id: str = _CURRENT_CORE_ID,
) -> dict[str, Any]:
    composite = _composite_attestation(
        stream,
        preflight_core_id=preflight_core_id,
    )
    return {
        "host": composite["host"],
        "vibeqc_version": composite["vibeqc_version"],
        "vibeqc_commit": composite["source_commit"],
        "source_clean": True,
        "core_built": "2026-07-11T12:00:00+00:00",
        "core_build_id": composite["core_build_id"],
        "probe_version": probe_host.PRODUCER_ATTESTATION_VERSION,
        "probe_script_id": composite["attestor_script_id"],
        "probe_passed": True,
        "probe_attestation_id": probe_host.probe_attestation_id(composite),
        "probe_attestation": composite,
        "producer_payload_id": composite["producer_payload_id"],
    }


def _cmp_b_source_binding() -> dict[str, Any]:
    payload_files = run_case_cmp_b._producer_payload()["files"]
    assert isinstance(payload_files, dict)
    tracked_files = {}
    for name, path in run_case_cmp_b._PAYLOAD_TRACKED_PATHS.items():
        digest = payload_files[name]
        tracked_files[name] = {
            "tracked_path": path,
            "tracked_sha256": digest,
            "copied_sha256": digest,
            "matches": True,
        }
    return {
        "expected_source_commit": _SOURCE_COMMIT,
        "origin_main_commit": _SOURCE_COMMIT,
        "required_ancestors": {
            name: {"commit": commit, "is_ancestor": True}
            for name, commit in run_case_cmp_b.SOURCE_REQUIREMENTS.items()
        },
        "tracked_files": tracked_files,
        "passed": True,
    }


def _cmp_b_success_record(
    monkeypatch: pytest.MonkeyPatch,
    system_name: str = "3d",
) -> dict[str, Any]:
    payload = _producer_payload("aiccm2026dev-b")
    monkeypatch.setattr(
        run_case_cmp_b,
        "_producer_payload",
        lambda: json.loads(json.dumps(payload)),
    )
    provenance = _producer_provenance("aiccm2026dev-b")
    provenance.update(
        attestation="git-checkout",
        expected_source_commit=_SOURCE_COMMIT,
    )
    meta = run_case_cmp_b.SYSTEMS[system_name]
    mesh = tuple(int(value) for value in meta["mesh"])
    primitive = np.asarray(meta["vectors"], dtype=float).T
    bvk_lattice = (primitive @ np.diag(np.asarray(mesh, dtype=float))).tolist()
    bvk_madelung = run_case_cmp_b._recomputed_bvk_madelung(
        run_case_cmp_b.build_system(system_name),
        bvk_lattice,
    )
    assert bvk_madelung is not None
    source_thresholds, threshold_failures = (
        run_case_cmp_b._source_resolved_thresholds()
    )
    assert threshold_failures == []
    receipt = None
    if system_name != "3d":
        receipt = {
            "schema": run_case_cmp_b.SCHEMA,
            "sha256": "sha256:" + "9" * 64,
            "system": "3d",
            "vq_job_id": "123456abcdef",
            "scheduler_job_id": "1001",
            "probe_attestation_id": provenance["probe_attestation_id"],
        }
    record = {
        "schema": run_case_cmp_b.SCHEMA,
        "spec": run_case_cmp_b.SPEC,
        "stream": run_case_cmp_b.STREAM,
        "selector": run_case_cmp_b.SELECTOR,
        "system": system_name,
        "system_label": meta["label"],
        "declared_model": meta["declared_model"],
        "dim": 3,
        "route": run_case_cmp_b.ROUTE,
        "method": "RHF",
        "functional": None,
        "backend": "ri",
        "entry_point": run_case_cmp_b.ENTRY_POINT,
        "evidence_role": run_case_cmp_b.EVIDENCE_ROLE,
        "quantitative_status": run_case_cmp_b.QUANTITATIVE_STATUS,
        "comparison_status": run_case_cmp_b.COMPARISON_STATUS,
        "status": "ok",
        "basis": run_case_cmp_b.BASIS,
        "aux_basis": run_case_cmp_b.EXPECTED_AUX,
        "primitive_lattice_bohr": primitive.tolist(),
        "atoms": [
            [int(atomic_number), list(position)]
            for atomic_number, position in meta["atoms"]
        ],
        "charge": 0,
        "multiplicity": 1,
        "physical_electrons_per_cell": int(meta["n_electrons"]),
        "atoms_per_cell": len(meta["atoms"]),
        "mesh": list(mesh),
        "requested_input": run_case_cmp_b._requested_input(
            mesh,
            run_case_cmp_b.EXPECTED_AUX,
        ),
        "executed_input": {
            "runtime_backend": "native-multi-k-gdf-gdf-rhf",
            "aux_basis_name": run_case_cmp_b.EXPECTED_AUX,
            "gdf_method": {
                "value": "rsgdf",
                "evidence": "B-selector-resolved-forwarded-setting",
            },
            "rsgdf_ke_cutoff_ha": {
                "value": 200.0,
                "evidence": "B-selector-resolved-forwarded-setting",
            },
            "fock_mixing_result": 0.0,
            "fock_mixing_diagnostics": 0.0,
            "converger": {
                "use_diis": True,
                "scf_accelerator": "EDIIS_DIIS",
                "diis_start_iter": 2,
                "diis_subspace_size": 8,
            },
            "static_scf_controls": {
                "damping": 0.0,
                "dynamic_damping": False,
                "level_shift": 0.0,
                "smearing_temperature_ha": 0.0,
            },
            "initial_guess": {
                "semantics": "per-k-hcore-diagonalisation",
                "evidence": "executed-progress-event",
                "event": "initial guess: HCORE (per-k Hcore diagonalisation)",
            },
            "linear_dependence_thresholds": {
                "linear_dep_threshold": {
                    "value": 1.0e-7,
                    "evidence": "executed-progress-event",
                },
                "gdf_linear_dep_threshold": {
                    "value": 1.0e-9,
                    "evidence": "executed-progress-event",
                },
                "conv_tol_grad": {
                    "value": 1.0e-6,
                    "evidence": "executed-progress-event",
                },
            },
            "source_resolved_thresholds": source_thresholds,
        },
        "finite_torus_convention": {
            "ccm_approach": "chi-ccm",
            "ccm_construction": "finite-translation-group-character",
            "evaluation_representation": "gamma-centred-character-mesh",
            "coulomb_kernel": "3d-periodic-g0",
            "exchange_q0": "bvk-ewald",
            "exchange_q0_applicability": "active",
            "boundary_model": "3d-periodic",
            "periodic_dimension": 3,
            "character_mesh_shape": list(mesh),
            "bvk_madelung_supercell_repetitions": list(mesh),
            "bvk_madelung_supercell_lattice_bohr": bvk_lattice,
            "lattice_vector_convention": "columns",
            "orbital_energy_convention": (
                "HF/Kohn-Sham eigenvalues include the declared "
                "exchange_q0 seam"
            ),
        },
        "exact_exchange_assembly": _fleet_exact_exchange_assembly("rhf-ri"),
        "two_electron_support": {
            **_unqualified_fitted_support("rhf-ri"),
            "reason": run_case_cmp_b.TWO_ELECTRON_SUPPORT_REASON,
        },
        "kmesh_audit": {
            "n_expected_characters": int(np.prod(mesh)),
            "n_executed_characters": int(np.prod(mesh)),
            "complete_character_net_matches": True,
            "fractional_residue_bijection": True,
            "cartesian_reciprocal_matches": True,
            "uniform_weights": True,
            "gamma_included": True,
            "bvk_lattice_matches_a_times_diag_n": True,
        },
        "exchange_seam_audit": {
            "routing_event_count": 1,
            "routing_event": run_case_cmp_b.ROUTING_EVENT,
            "separately_recomputed_bvk_madelung_positive": True,
            "bvk_madelung_xi": bvk_madelung,
            "route_oracle_c_full": 1.0,
            "evidence_level": "dispatch-evidence-not-independent-kernel-proof",
        },
        "energy_audit": {
            "absolute_energy_in_campaign_json": False,
            "components_finite": True,
            "total_decomposition_matches": True,
            "nuclear_matches_separately_recomputed_3d_ewald": True,
            "assembly_tolerance_ha": 1.0e-12,
        },
        "convergence": {
            "converged": True,
            "iterations": 3,
            "scf_trace_length": 3,
            "final_delta_e_ha": 5.0e-11,
            "final_grad_norm": 5.0e-7,
            "source_resolved_conv_tol_grad": 1.0e-6,
        },
        "walltime_s": 1.0,
        "peak_rss_mb": 100.0,
        "provenance": provenance,
        "source_binding": _cmp_b_source_binding(),
        "validation_receipt": receipt,
        "vq_target_host": "compute-large",
        "vq_job_id": "abcdef123456",
        "scheduler_job_id": "12345",
        "utc_started": "2026-08-01T12:00:00+00:00",
        "utc_finished": "2026-08-01T12:01:00+00:00",
        "pin_violations": [],
    }
    assert set(record) == run_case_cmp_b._SUCCESS_KEYS
    return record


def _set_nested(record: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor: dict[str, Any] = record
    for key in path[:-1]:
        child = cursor[key]
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = value


def test_chi_campaign_accepts_exact_nonquantitative_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _cmp_b_success_record(monkeypatch)

    assert run_case_cmp_b.campaign_record_failures(
        record,
        expected_source_sha=_SOURCE_COMMIT,
        required_system="3d",
    ) == []
    assert "energy_per_cell_ha" not in record
    assert "energy_per_atom_ha" not in record
    assert record["energy_audit"]["absolute_energy_in_campaign_json"] is False
    assert record["quantitative_status"] == "not-qualified"
    assert record["validation_receipt"] is None


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("charge",), False),
        (("executed_input", "fock_mixing_result"), 0.30),
        (("executed_input", "fock_mixing_result"), False),
        (("executed_input", "fock_mixing_diagnostics"), 0.30),
        (("executed_input", "fock_mixing_diagnostics"), False),
        (("executed_input", "aux_basis_name"), "wrong-aux"),
        (("executed_input", "rsgdf_ke_cutoff_ha", "value"), 180.0),
        (("executed_input", "static_scf_controls", "dynamic_damping"), True),
        (("finite_torus_convention", "exchange_q0"), "none"),
        (
            (
                "finite_torus_convention",
                "bvk_madelung_supercell_lattice_bohr",
            ),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        (("exchange_seam_audit", "routing_event_count"), 0),
        (("exchange_seam_audit", "routing_event_count"), True),
        (
            ("exchange_seam_audit", "routing_event"),
            run_case_cmp_b.ROUTING_EVENT + " E_total=-7.92 Ha",
        ),
        (
            ("executed_input", "initial_guess", "event"),
            run_case_cmp_b.INITIAL_GUESS_EVENT + " E_total=-7.92 Ha",
        ),
        (
            (
                "exchange_seam_audit",
                "separately_recomputed_bvk_madelung_positive",
            ),
            False,
        ),
        (("exchange_seam_audit", "route_oracle_c_full"), True),
        (("exchange_seam_audit", "bvk_madelung_xi"), 7.92),
        (("source_binding", "tracked_files", "producer", "matches"), False),
        (
            (
                "source_binding",
                "required_ancestors",
                "d88_column_lattice",
                "is_ancestor",
            ),
            1,
        ),
        (("convergence", "final_grad_norm"), 2.0e-6),
        (("convergence", "final_grad_norm"), -1.0),
        (
            ("two_electron_support", "reason"),
            run_case_cmp_b.TWO_ELECTRON_SUPPORT_REASON + " E_total=-7.92 Ha",
        ),
        (("two_electron_support", "fitted", "rsgdf_base_ke_cutoff_ha"), 201.0),
        (("provenance", "energy_per_cell_ha"), -7.92),
        (("provenance", "source_clean"), 1),
        (("provenance", "core_built"), "2026-07-11T12:00:00+00:00 E=-7.92 Ha"),
        (("vq_job_id",), "not-recorded"),
        (("scheduler_job_id",), "not-recorded"),
        (("utc_started",), "2026-08-01T12:00:00.123+00:00"),
        (("utc_started",), "2026-08-01T08:00:00-04:00"),
        (("utc_finished",), "2026-08-01T11:59:59+00:00"),
        (("validation_receipt",), {"forged": True}),
        (
            ("provenance", "probe_attestation", "core_changed_since_preflight"),
            True,
        ),
    ),
)
def test_chi_campaign_rejects_nested_receipt_tampering(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: Any,
) -> None:
    record = _cmp_b_success_record(monkeypatch)
    _set_nested(record, path, value)

    assert run_case_cmp_b.campaign_record_failures(
        record,
        expected_source_sha=_SOURCE_COMMIT,
    )


def test_chi_campaign_rejects_any_absolute_energy_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _cmp_b_success_record(monkeypatch)
    record["energy_per_cell_ha"] = -8.0

    assert run_case_cmp_b.campaign_record_failures(
        record,
        expected_source_sha=_SOURCE_COMMIT,
    )


def test_chi_campaign_cross_binds_source_and_attested_payload_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _cmp_b_success_record(monkeypatch)
    forged = "sha256:" + "f" * 64
    for item in record["source_binding"]["tracked_files"].values():
        item["tracked_sha256"] = forged
        item["copied_sha256"] = forged

    failures = run_case_cmp_b.campaign_record_failures(
        record,
        expected_source_sha=_SOURCE_COMMIT,
    )
    assert any("source/payload digest differs" in failure for failure in failures)


def test_chi_campaign_dedicated_auditor_never_needs_an_energy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _cmp_b_success_record(monkeypatch)
    path = tmp_path / "d101.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    assert audit_cmp_b.audit_path(path, _SOURCE_COMMIT) == []
    record["energy_per_atom_ha"] = -4.0
    path.write_text(json.dumps(record), encoding="utf-8")
    assert audit_cmp_b.audit_path(path, _SOURCE_COMMIT)


def test_chi_campaign_auditor_rejects_duplicate_key_energy_smuggling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _cmp_b_success_record(monkeypatch)
    encoded = json.dumps(record)
    encoded = '{"system_label":"E_total=-7.92 Ha",' + encoded[1:]
    path = tmp_path / "duplicate.json"
    path.write_text(encoded, encoding="utf-8")

    failures = audit_cmp_b.audit_path(path, _SOURCE_COMMIT)
    assert failures == ["cannot read strict D101 JSON"]
    assert "-7.92" not in " ".join(failures)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_chi_campaign_auditor_rejects_nonstandard_json_constants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    constant: str,
) -> None:
    record = _cmp_b_success_record(monkeypatch)
    encoded = json.dumps(record)
    needle = '"walltime_s": 1.0'
    assert needle in encoded
    encoded = encoded.replace(needle, f'"walltime_s": {constant}', 1)
    path = tmp_path / "nonstandard.json"
    path.write_text(encoded, encoding="utf-8")

    assert audit_cmp_b.audit_path(path, _SOURCE_COMMIT) == [
        "cannot read strict D101 JSON"
    ]


@pytest.mark.parametrize(
    "malformed",
    ([], {"provenance": None}),
)
def test_chi_campaign_malformed_validation_receipt_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    malformed: object,
) -> None:
    current = _cmp_b_success_record(monkeypatch)["provenance"]
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(malformed), encoding="utf-8")

    assert run_case_cmp_b._validation_receipt_failures(
        path,
        _SOURCE_COMMIT,
        current,
    )


def test_chi_campaign_binds_same_identity_passing_3d_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _cmp_b_success_record(monkeypatch)
    path = tmp_path / "d101-3d.json"
    encoded = json.dumps(record, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded)

    binding, failures = run_case_cmp_b._validation_receipt(
        path,
        _SOURCE_COMMIT,
        record["provenance"],
    )

    assert failures == []
    assert binding == {
        "schema": run_case_cmp_b.SCHEMA,
        "sha256": run_case_cmp_b._sha256_bytes(encoded),
        "system": "3d",
        "vq_job_id": record["vq_job_id"],
        "scheduler_job_id": record["scheduler_job_id"],
        "probe_attestation_id": record["provenance"]["probe_attestation_id"],
    }

    changed = json.loads(json.dumps(record["provenance"]))
    changed["probe_attestation"]["core_build_id"] = _OLD_CORE_ID
    assert run_case_cmp_b._validation_receipt_failures(
        path,
        _SOURCE_COMMIT,
        changed,
    )


def test_chi_campaign_auditor_requires_and_checks_exact_3d_receipt_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = _cmp_b_success_record(monkeypatch)
    parent_path = tmp_path / "d101-3d.json"
    parent_bytes = json.dumps(parent, sort_keys=True).encode("utf-8")
    parent_path.write_bytes(parent_bytes)

    child = _cmp_b_success_record(monkeypatch, "1d")
    child["vq_job_id"] = "abcdef123457"
    child["scheduler_job_id"] = "12346"
    child["validation_receipt"] = {
        "schema": run_case_cmp_b.SCHEMA,
        "sha256": run_case_cmp_b._sha256_bytes(parent_bytes),
        "system": "3d",
        "vq_job_id": parent["vq_job_id"],
        "scheduler_job_id": parent["scheduler_job_id"],
        "probe_attestation_id": parent["provenance"]["probe_attestation_id"],
    }
    child_path = tmp_path / "d101-1d.json"
    child_path.write_text(json.dumps(child), encoding="utf-8")

    assert audit_cmp_b.audit_path(child_path, _SOURCE_COMMIT)
    assert (
        audit_cmp_b.audit_path(
            child_path,
            _SOURCE_COMMIT,
            validation_record=parent_path,
        )
        == []
    )

    child["validation_receipt"]["probe_attestation_id"] = "sha256:" + "d" * 64
    child_path.write_text(json.dumps(child), encoding="utf-8")
    assert audit_cmp_b.audit_path(
        child_path,
        _SOURCE_COMMIT,
        validation_record=parent_path,
    )


def test_chi_campaign_receipt_accepts_distinct_node_attestation_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def retarget_host(provenance: dict[str, Any], host: str) -> None:
        attestation = provenance["probe_attestation"]
        attestation["host"] = host
        for field, digest_field in (
            ("preflight_attestation", "preflight_attestation_id"),
            ("current_core_attestation", "current_core_attestation_id"),
        ):
            attestation[field]["host"] = host
            attestation[digest_field] = probe_host.probe_attestation_id(
                attestation[field]
            )
        provenance["host"] = host
        provenance["probe_attestation_id"] = probe_host.probe_attestation_id(
            attestation
        )

    parent = _cmp_b_success_record(monkeypatch)
    retarget_host(parent["provenance"], "compute-node-a")
    parent_path = tmp_path / "d101-parent.json"
    parent_bytes = json.dumps(parent, sort_keys=True).encode("utf-8")
    parent_path.write_bytes(parent_bytes)

    child = _cmp_b_success_record(monkeypatch, "1d")
    retarget_host(child["provenance"], "compute-node-b")
    child["vq_job_id"] = "abcdef123457"
    child["scheduler_job_id"] = "12346"
    child["validation_receipt"] = {
        "schema": run_case_cmp_b.SCHEMA,
        "sha256": run_case_cmp_b._sha256_bytes(parent_bytes),
        "system": "3d",
        "vq_job_id": parent["vq_job_id"],
        "scheduler_job_id": parent["scheduler_job_id"],
        "probe_attestation_id": parent["provenance"]["probe_attestation_id"],
    }
    child_path = tmp_path / "d101-child.json"
    child_path.write_text(json.dumps(child), encoding="utf-8")

    assert (
        parent["provenance"]["probe_attestation_id"]
        != child["provenance"]["probe_attestation_id"]
    )
    assert audit_cmp_b.audit_path(
        child_path,
        _SOURCE_COMMIT,
        validation_record=parent_path,
    ) == []


def test_chi_campaign_log_number_parser_is_strict_and_finite() -> None:
    log = "  linear_dep_threshold = 1e-07\n  other = value\n"
    values = run_case_cmp_b._setting_values(log, "linear_dep_threshold")

    assert values == ["1e-07"]
    assert run_case_cmp_b._finite_numeric_text(values[0]) == 1.0e-7
    assert run_case_cmp_b._finite_numeric_text("nan") is None
    assert run_case_cmp_b._finite_numeric_text("inf") is None
    assert run_case_cmp_b._finite_numeric_text(True) is None


def test_chi_campaign_pins_high_level_route_and_declares_every_cell_3d(
    tmp_path: Path,
) -> None:
    progress = run_case_cmp_b.ProgressLogger(verbose=0)
    for name, meta in run_case_cmp_b.SYSTEMS.items():
        system = run_case_cmp_b.build_system(name)
        np.testing.assert_allclose(
            system.lattice,
            np.asarray(meta["vectors"], dtype=float).T,
            rtol=0.0,
            atol=1.0e-12,
        )
        assert system.dim == 3
        stem = run_case_cmp_b._campaign_stem(name)
        assert "__" not in f"{stem}.json"

    kwargs = run_case_cmp_b._runner_kwargs(
        tmp_path / "runner",
        (2, 2, 2),
        run_case_cmp_b.EXPECTED_AUX,
        progress,
    )
    assert kwargs["method"] == "RHF"
    assert kwargs["functional"] is None
    assert kwargs["jk_method"] == "aiccm2026dev-b"
    assert kwargs["aiccm_backend"] == "ri"
    assert kwargs["aiccm_lattice_extension"] == (2, 2, 2)
    assert kwargs["gdf_method"] == "rsgdf"
    assert kwargs["rsgdf_ke_cutoff"] == 200.0
    assert kwargs["fock_mixing"] == 0.0
    assert kwargs["damping"] == 0.0
    assert kwargs["dynamic_damping"] is False
    assert kwargs["initial_guess"] == "HCORE"
    assert kwargs["exchange_exxdiv"] == "ewald"
    assert kwargs["output_qvf"] is False
    launcher = (_TESTSET_DIR / "run.sh").read_text(encoding="utf-8")
    assert 'case_script="run_case_cmp_b.py"' in launcher
    assert '"${1:-}" = "--comparison-b"' in launcher
    assert 'PY="$VIBEQC_PYTHON"' in launcher


def test_chi_campaign_refuses_non_vq_or_non_slurm_context_before_scf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("VQ_JOB_ID", "VQ_WORKDIR", "SLURM_JOB_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        run_case_cmp_b.vq,
        "run_periodic_job",
        lambda *_args, **_kwargs: pytest.fail("invalid context reached SCF"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_cmp_b.py",
            "3d",
            "--expected-source-sha",
            _SOURCE_COMMIT,
            "--vq-host",
            "compute-large",
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="3"):
        run_case_cmp_b.main()

    path = tmp_path / f"{run_case_cmp_b._campaign_stem('3d')}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "input-failed"
    assert record["absolute_energy_in_campaign_json"] is False


def test_chi_campaign_replaces_stale_success_on_preflight_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VQ_JOB_ID", "abcdef123456")
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    path = tmp_path / f"{run_case_cmp_b._campaign_stem('3d')}.json"
    path.write_text(
        json.dumps({"status": "ok", "energy_per_cell_ha": -7.92}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_case_cmp_b,
        "_source_binding",
        lambda _sha: (_ for _ in ()).throw(
            RuntimeError("preflight failed; E_total=-7.92 Ha")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_cmp_b.py",
            "3d",
            "--expected-source-sha",
            _SOURCE_COMMIT,
            "--vq-host",
            "compute-large",
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="3"):
        run_case_cmp_b.main()

    encoded = path.read_text(encoding="utf-8")
    record = json.loads(encoded)
    assert record["status"] == "attestation-error"
    assert "energy_per_cell_ha" not in encoded
    assert "-7.92" not in encoded


def test_chi_campaign_serializes_unexpected_declared_3d_fail_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VQ_JOB_ID", "abcdef123456")
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    binding = _cmp_b_source_binding()
    provenance = {"probe_attestation": {}}
    monkeypatch.setattr(
        run_case_cmp_b,
        "_source_binding",
        lambda _sha: (json.loads(json.dumps(binding)), []),
    )
    monkeypatch.setattr(
        run_case_cmp_b,
        "_provenance",
        lambda _sha: (provenance, []),
    )
    monkeypatch.setattr(
        run_case_cmp_b,
        "_finalize_provenance",
        lambda _provenance, _sha: [],
    )
    monkeypatch.setattr(
        run_case_cmp_b.vq,
        "run_periodic_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            NotImplementedError("unexpected declared-3d gate; E_total=-7.92 Ha")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_cmp_b.py",
            "3d",
            "--expected-source-sha",
            _SOURCE_COMMIT,
            "--vq-host",
            "compute-large",
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="4"):
        run_case_cmp_b.main()

    path = tmp_path / f"{run_case_cmp_b._campaign_stem('3d')}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "unexpected-not-implemented"
    assert record["absolute_energy_in_campaign_json"] is False
    assert "NotImplementedError" in record["pin_violations"][0]
    assert "-7.92" not in json.dumps(record)


def _cmp_b_fake_live_result() -> tuple[object, SimpleNamespace, str]:
    system = run_case_cmp_b.build_system("3d")
    mesh = tuple(run_case_cmp_b.SYSTEMS["3d"]["mesh"])
    character_mesh = cyclic_gamma_mesh(system, mesh)
    bvk_lattice = np.asarray(system.lattice) @ np.diag(np.asarray(mesh))
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic",
        periodic_dimension=3,
        character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=mesh,
        bvk_madelung_supercell_lattice_bohr=tuple(
            tuple(float(value) for value in row) for row in bvk_lattice
        ),
    )
    diagnostics = SimpleNamespace(
        mesh=mesh,
        n_cyclic_cells=8,
        n_kpoints=8,
        backend="ri",
        electronic_method="RHF",
        use_diis=True,
        diis_start_iter=2,
        diis_subspace_size=8,
        scf_accelerator="EDIIS_DIIS",
        damping=0.0,
        dynamic_damping=False,
        fock_mixing=0.0,
        level_shift=0.0,
        smearing_temperature=0.0,
        physical_electron_count=4,
        effective_electron_count=4,
        ecp_total_ncore=0,
        wigner_seitz_partition_error=0.0,
        density_idempotency_error=0.0,
        electron_count_error=0.0,
        inverse_bloch_imaginary_residual=0.0,
        effective_net_charge=0.0,
        scf_trace_length=3,
        final_scf_delta_e_ha=5.0e-11,
        final_scf_grad_norm=5.0e-7,
    )
    result = SimpleNamespace(
        backend="aiccm2026dev-b-ri",
        runtime_backend="native-multi-k-gdf-gdf-rhf",
        finite_torus_convention=convention,
        exact_exchange_assembly=_rhf_exact_exchange_assembly(),
        aiccm2026dev_b=diagnostics,
        kpoints_frac=np.asarray(character_mesh.kpoints_frac),
        kpoints_cart=np.asarray(character_mesh.kpoints_cart),
        kpoint_weights=np.asarray(character_mesh.weights),
        fock_mixing=0.0,
        aux_basis_name=run_case_cmp_b.EXPECTED_AUX,
        aiccm_resolved_gdf_method="rsgdf",
        aiccm_resolved_rsgdf_ke_cutoff=200.0,
        energy=-7.0,
        e_electronic=-8.0,
        e_nuclear=1.0,
        converged=True,
        n_iter=3,
        scf_trace=[object(), object(), object()],
    )
    log = "\n".join(
        (
            run_case_cmp_b.ROUTING_EVENT,
            run_case_cmp_b.INITIAL_GUESS_EVENT,
            "linear_dep_threshold = 1e-07",
            "gdf_linear_dep_threshold = 1e-09",
            "conv_tol_grad = 1e-06",
        )
    )
    return system, result, log


def test_chi_campaign_live_audit_fails_closed_on_mixing_and_energy_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system, result, log = _cmp_b_fake_live_result()
    source_thresholds, failures = run_case_cmp_b._source_resolved_thresholds()
    assert failures == []
    monkeypatch.setattr(
        run_case_cmp_b,
        "madelung_constant_for_cell",
        lambda _system: 0.25,
    )
    monkeypatch.setattr(
        run_case_cmp_b.vq,
        "ewald_nuclear_repulsion",
        lambda _system: 1.0,
    )

    evidence, failures = run_case_cmp_b._audit_result(
        "3d",
        system,
        result,
        log,
        run_case_cmp_b.EXPECTED_AUX,
        source_thresholds,
    )
    assert failures == []
    assert evidence["energy_audit"]["absolute_energy_in_campaign_json"] is False
    assert "energy_per_cell_ha" not in evidence

    result.fock_mixing = 0.30
    assert any(
        "fock_mixing" in failure
        for failure in run_case_cmp_b._audit_result(
            "3d",
            system,
            result,
            log,
            run_case_cmp_b.EXPECTED_AUX,
            source_thresholds,
        )[1]
    )
    result.fock_mixing = 0.0
    result.energy = -6.9
    assert any(
        "electronic plus nuclear" in failure
        for failure in run_case_cmp_b._audit_result(
            "3d",
            system,
            result,
            log,
            run_case_cmp_b.EXPECTED_AUX,
            source_thresholds,
        )[1]
    )
    result.energy = -7.0
    assert any(
        "routing event" in failure
        for failure in run_case_cmp_b._audit_result(
            "3d",
            system,
            result,
            log.replace("routing", "route"),
            run_case_cmp_b.EXPECTED_AUX,
            source_thresholds,
        )[1]
    )


def _write_retracted_b_result(
    root: Path,
    *,
    basis: str = "sto-3g",
    marker: object = True,
) -> Path:
    record = (
        root
        / "aiccm2026dev-b"
        / "h-chain"
        / basis
        / "rhf-4c"
        / "result.json"
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "selector": "aiccm2026dev-b",
                "system": "h-chain",
                "route": "rhf-4c",
                "basis": basis,
                "mesh": [1, 1, 1],
                "status": "ok",
                "converged": True,
                "dim": 3,
                "energy_per_atom_ha": -1.0,
                "retracted": marker,
                "retraction_reason": "invalid validation provenance",
                "finite_torus_convention": {
                    "coulomb_kernel": "3d-periodic-g0",
                    "exchange_q0": "bvk-ewald",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return record


def _matching_real_gamma_control_records(
    b_route_name: str,
    *,
    control_route_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    route = b_routes.ROUTES[b_route_name]
    resolved_control_route = (
        control_route_name
        or compare_b.REAL_GAMMA_CONTROL_ROUTE_FOR_B.get(b_route_name)
    )
    if resolved_control_route is None:
        raise ValueError(f"no real-Gamma control declared for {b_route_name}")
    b_provenance = _producer_provenance("aiccm2026dev-b")
    control_provenance = _producer_provenance("aiccm2026dev-a")
    comparison_input = {
        # Frozen legacy schema spelling; this fixture validates provenance for
        # a representation control, not Gamma-CCM construction evidence.
        "schema": compare_b.REAL_GAMMA_CONTROL_INPUT_VERSION,
        "system_name": "lih-rocksalt",
        "system": {
            "dim": 3,
            "charge": 0,
            "multiplicity": 1,
            "lattice_bohr": [
                [0.0, 3.86, 3.86],
                [3.86, 0.0, 3.86],
                [3.86, 3.86, 0.0],
            ],
            "atoms": [
                {"atomic_number": 3, "xyz_bohr": [0.0, 0.0, 0.0]},
                {"atomic_number": 1, "xyz_bohr": [3.86, 3.86, 3.86]},
            ],
        },
        "electronic_method": route.method,
        "functional": route.functional,
        "orbital_basis": "sto-3g",
        "auxiliary_basis": "def2-svp-jk",
        "finite_torus_mesh": [2, 2, 2],
        "boundary_model": "3d-periodic",
        "coulomb_kernel": "3d-periodic-g0",
        "exchange_q0": "bvk-ewald",
        "two_electron_operator": route.backend,
        "gdf_method": "rsgdf",
        "rsgdf_ke_cutoff": 200.0,
        "ao_linear_dependence_threshold": 1.0e-7,
        "auxiliary_metric_linear_dependence_threshold": 1.0e-9,
        "auxiliary_phase_convention": "bloch-cell-periodic/v1",
        "smearing_temperature": 0.0,
    }
    input_sha256 = compare_b._comparison_input_sha256(comparison_input)
    assert input_sha256 is not None
    contract = {
        "version": compare_b.REAL_GAMMA_CONTROL_CONTRACT_VERSION,
        "input_sha256": input_sha256,
        "dim": 3,
        "boundary_model": "3d-periodic",
        "coulomb_kernel": "3d-periodic-g0",
        "exchange_q0": "bvk-ewald",
        "two_electron_operator": route.backend,
        "aux_basis": "def2-svp-jk",
        "gdf_method": "rsgdf",
        "rsgdf_ke_cutoff": 200.0,
        "ao_linear_dependence_threshold": 1.0e-7,
        "auxiliary_metric_linear_dependence_threshold": 1.0e-9,
        "auxiliary_phase_convention": "bloch-cell-periodic/v1",
        "smearing_temperature": 0.0,
    }
    return (
        {
            "system": "lih-rocksalt",
            "route": b_route_name,
            "status": "ok",
            "converged": True,
            "method": route.method,
            "backend": route.backend,
            "functional": route.functional,
            "basis": "sto-3g",
            "aux_basis": "def2-svp-jk",
            "gdf_method": "rsgdf",
            "rsgdf_ke_cutoff": 200.0,
            "ao_linear_dependence_threshold": 1.0e-7,
            "auxiliary_metric_linear_dependence_threshold": 1.0e-9,
            "auxiliary_phase_convention": "bloch-cell-periodic/v1",
            "two_electron_operator": route.backend,
            "scf_options": {"smearing_temperature": 0.0},
            "convergence_diagnostics": {"smearing_temperature": 0.0},
            "dim": 3,
            "mesh": [2, 2, 2],
            "primitive_lattice_bohr": [
                [0.0, 3.86, 3.86],
                [3.86, 0.0, 3.86],
                [3.86, 3.86, 0.0],
            ],
            "finite_torus_convention": {
                "ccm_approach": "chi-ccm",
                "ccm_construction": "finite-translation-group-character",
                "evaluation_representation": (
                    "gamma-centred-character-mesh"
                ),
                "exchange_q0": "bvk-ewald",
                "exchange_q0_applicability": (
                    route.exchange_q0_applicability
                ),
                "boundary_model": "3d-periodic",
                "coulomb_kernel": "3d-periodic-g0",
                "lattice_vector_convention": "columns",
                "periodic_dimension": 3,
                "character_mesh_shape": [2, 2, 2],
                "bvk_madelung_supercell_repetitions": [2, 2, 2],
                "bvk_madelung_supercell_lattice_bohr": [
                    [0.0, 7.72, 7.72],
                    [7.72, 0.0, 7.72],
                    [7.72, 7.72, 0.0],
                ],
            },
            "exact_exchange_assembly": _fleet_exact_exchange_assembly(
                b_route_name
            ),
            "two_electron_support": _unqualified_fitted_support(b_route_name),
            "ewald_shifted_pair_support": (
                _qualified_ewald_shifted_pair_support(b_provenance)
            ),
            "provenance": json.loads(json.dumps(b_provenance)),
            "comparison_input": json.loads(json.dumps(comparison_input)),
            "comparison_contract": dict(contract),
            "energy_per_atom_ha": -4.0,
        },
        {
            "system": "lih-rocksalt",
            "route": resolved_control_route,
            "ccm_approach": "not-applicable",
            "ccm_construction": "neutral-fitted-torus",
            "evaluation_representation": "real-gamma-supercell",
            "route_role": "representation-control",
            "converged": True,
            "method": route.method,
            "functional": route.functional,
            "basis": "STO-3G",
            "aux_basis": "def2-svp-jk",
            "gdf_method": "rsgdf",
            "rsgdf_ke_cutoff": 200.0,
            "ao_linear_dependence_threshold": 1.0e-7,
            "auxiliary_metric_linear_dependence_threshold": 1.0e-9,
            "auxiliary_phase_convention": "bloch-cell-periodic/v1",
            "two_electron_operator": route.backend,
            "scf_options": {"smearing_temperature": 0.0},
            "convergence_diagnostics": {"smearing_temperature": 0.0},
            "dim": 3,
            "nrep": [2, 2, 2],
            "finite_torus_convention": {
                "exchange_q0": "BvK-ewald",
                "boundary_model": "3d-periodic",
                "coulomb_kernel": "3d-periodic-g0",
                "periodic_dimension": 3,
                "character_mesh_shape": [2, 2, 2],
            },
            "ewald_shifted_pair_support": (
                _qualified_ewald_shifted_pair_support(control_provenance)
            ),
            "provenance": json.loads(json.dumps(control_provenance)),
            "comparison_input": json.loads(json.dumps(comparison_input)),
            "comparison_contract": dict(contract),
            "energy_per_atom_ha": -4.0,
        },
    )


def test_b_route_matrix_covers_every_implemented_family() -> None:
    assert b_routes.ROUTES == _EXPECTED_B_ROUTES
    assert (
        b_routes.EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE
        == _EXPECTED_EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE
    )
    assert b_routes.EXACT_EXCHANGE_RESOLVER == _EXPECTED_EXACT_EXCHANGE_RESOLVER
    assert set(b_routes.ROUTES) == {
        "rhf-4c",
        "rhf-ri",
        "rhf-rijcosx",
        "rks-pbe-4c",
        "rks-pbe-ri",
        "rks-pbe-rijcosx",
        "rks-pbe0-4c",
        "rks-pbe0-ri",
        "rks-pbe0-rijcosx",
        "ri-mp2",
        "dlpno-mp2",
        "dlpno-ccsd",
        "dlpno-ccsd-t",
    }
    assert {route.backend for route in b_routes.ROUTES.values()} == {
        "four_center",
        "ri",
        "rijcosx",
    }
    assert {route.functional for route in b_routes.ROUTES.values()} >= {
        None,
        "pbe",
        "pbe0",
    }
    assert {
        name: route.exchange_q0_applicability
        for name, route in b_routes.ROUTES.items()
        if route.functional == "pbe"
    } == {
        "rks-pbe-4c": "inactive",
        "rks-pbe-ri": "inactive",
        "rks-pbe-rijcosx": "inactive",
    }
    assert all(
        route.exchange_q0_applicability == "active"
        for route in b_routes.ROUTES.values()
        if route.functional != "pbe"
    )


@pytest.mark.parametrize(
    "route_name",
    ("rhf-4c", "rks-pbe-ri", "rks-pbe0-rijcosx", "dlpno-ccsd-t"),
)
def test_exact_exchange_assembly_route_oracle_accepts_supported_families(
    route_name: str,
) -> None:
    assert (
        b_routes.exact_exchange_assembly_failure(
            _fleet_exact_exchange_assembly(route_name),
            route_name,
        )
        is None
    )


def test_screened_exchange_route_oracle_covers_every_registered_route() -> None:
    assert b_routes.EXACT_EXCHANGE_ASSEMBLY_SCHEMA == (
        _EXPECTED_EXACT_EXCHANGE_SCHEMA
    )
    assert b_routes.SCREENED_EXCHANGE_ASSEMBLY_BY_ROUTE == {
        route_name: ("inactive", "not-applicable")
        for route_name in _EXPECTED_B_ROUTES
    }


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("missing", "missing or is not an object"),
        ("malformed", "missing or is not an object"),
        ("extra-key", "exact keys"),
        ("wrong-schema", "wrong schema"),
        ("spoofed-resolver", "wrong resolver"),
        ("boolean", "excluding booleans"),
        ("overflow", "finite JSON numbers"),
        ("wrong-coefficients", "contradict the route selector"),
        ("wrong-screened-applicability", "screened branch contradicts"),
        ("wrong-screened-assembly", "screened branch contradicts"),
        ("wrong-route", "contradict the route selector"),
    ),
)
def test_exact_exchange_assembly_route_oracle_fails_closed(
    case: str,
    expected: str,
) -> None:
    route_name = "rks-pbe0-ri"
    value: object = _fleet_exact_exchange_assembly(route_name)
    validated_route = route_name
    if case == "missing":
        value = None
    elif case == "malformed":
        value = ["invalid"]
    elif case == "extra-key":
        value["unexpected"] = True
    elif case == "wrong-schema":
        value["schema"] = (
            "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v999"
        )
    elif case == "spoofed-resolver":
        value["resolver"] = "spoofed.resolver"
    elif case == "boolean":
        value["c_full"] = True
    elif case == "overflow":
        value["c_full"] = 10**400
    elif case == "wrong-coefficients":
        value["c_full"] = 1.0
    elif case == "wrong-screened-applicability":
        value["screened_exchange_applicability"] = "active"
    elif case == "wrong-screened-assembly":
        value["screened_exchange_assembly"] = "short-range-direct"
    else:
        validated_route = "rks-pbe-ri"

    failure = b_routes.exact_exchange_assembly_failure(value, validated_route)

    assert failure is not None
    assert expected in failure


def test_exact_exchange_assembly_oracle_cross_checks_seam_applicability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = b_routes.ROUTES["rhf-4c"]
    monkeypatch.setitem(
        b_routes.ROUTES,
        "rhf-4c",
        b_routes.Route(
            method=route.method,
            backend=route.backend,
            functional=route.functional,
            post_hf=route.post_hf,
            exchange_q0_applicability="inactive",
        ),
    )

    failure = b_routes.exact_exchange_assembly_failure(
        _fleet_exact_exchange_assembly("rhf-4c"),
        "rhf-4c",
    )

    assert failure is not None
    assert "exchange_q0_applicability contradicts" in failure


def test_research_wssc_route_does_not_claim_the_production_exchange_seam() -> None:
    assert run_case._exchange_q0_for_route("aiccm-rijcosx") == (
        "research-wssc-cosx-undeclared"
    )
    assert run_case._exchange_q0_for_route("aiccm-ri-cosx") == "BvK-ewald"


@pytest.mark.parametrize(
    ("route", "expected"),
    (
        (
            "aiccm-hf",
            {
                "ccm_approach": "gamma-ccm",
                "ccm_construction": "union-and-weight",
                "evaluation_representation": "real-supercell-four-center",
                "route_role": "ccm-approach",
            },
        ),
        (
            "aiccm-hf-direct",
            {
                "ccm_approach": "not-applicable",
                "ccm_construction": "neutral-fitted-torus",
                "evaluation_representation": "real-gamma-supercell",
                "route_role": "representation-control",
            },
        ),
        (
            "aiccm-ri",
            {
                "ccm_approach": "not-applicable",
                "ccm_construction": "neutral-fitted-torus",
                "evaluation_representation": "bloch-gdf",
                "route_role": "representation-control",
            },
        ),
        (
            "aiccm-ri-cosx",
            {
                "ccm_approach": "not-applicable",
                "ccm_construction": "neutral-fitted-torus-cosx",
                "evaluation_representation": "bloch-gdf-cosx",
                "route_role": "approximate-exchange-control",
            },
        ),
        (
            "aiccm-dlpno-mp2",
            {
                "ccm_approach": "not-applicable",
                "ccm_construction": "neutral-fitted-torus-research",
                "evaluation_representation": "real-supercell-neutral-cderi",
                "route_role": "post-hf-research-route",
            },
        ),
        (
            "aiccm-rijcosx",
            {
                "ccm_approach": "not-applicable",
                "ccm_construction": "research-wssc-cosx-undeclared",
                "evaluation_representation": "real-supercell-wssc-cosx",
                "route_role": "research-control",
            },
        ),
        (
            "gdf",
            {
                "ccm_approach": "not-applicable",
                "ccm_construction": "not-applicable",
                "evaluation_representation": "periodic-k-mesh",
                "route_role": "periodic-reference",
            },
        ),
    ),
)
def test_a_harness_declares_exact_route_semantics(
    route: str,
    expected: dict[str, str],
) -> None:
    assert run_case._route_semantics(route) == expected


def test_a_harness_route_semantics_cover_every_route() -> None:
    assert all(run_case._route_semantics(route) for route in run_case.ROUTES)
    with pytest.raises(ValueError, match="semantics are not declared"):
        run_case._route_semantics("undeclared")


def test_imported_source_state_uses_the_loaded_vibeqc_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "loaded-vibeqc"
    package = source_root / "python" / "vibeqc"
    package.mkdir(parents=True)
    (source_root / ".git").mkdir()
    monkeypatch.setattr(probe_host.vibeqc, "__file__", str(package / "__init__.py"))

    calls: list[tuple[tuple[str, ...], Path]] = []

    def fake_check_output(argv, *, cwd, **_kwargs):
        calls.append((tuple(argv), Path(cwd)))
        return "d" * 40 + "\n" if "rev-parse" in argv else ""

    monkeypatch.setattr(probe_host.subprocess, "check_output", fake_check_output)

    assert probe_host.imported_source_state() == ("d" * 40, True)
    assert calls == [
        (
            (
                "git",
                "ls-files",
                "--error-unmatch",
                "--",
                "python/vibeqc/__init__.py",
            ),
            source_root,
        ),
        (("git", "rev-parse", "HEAD"), source_root),
        (("git", "status", "--porcelain"), source_root),
    ]


def test_imported_source_state_rejects_an_untracked_in_repo_wheel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "checkout"
    package = source_root / ".venv" / "site-packages" / "vibeqc"
    package.mkdir(parents=True)
    (source_root / ".git").mkdir()
    monkeypatch.setattr(probe_host.vibeqc, "__file__", str(package / "__init__.py"))

    def fake_check_output(argv, **_kwargs):
        if "ls-files" in argv:
            raise probe_host.subprocess.CalledProcessError(1, argv)
        raise AssertionError(f"unexpected Git command after untracked file: {argv}")

    monkeypatch.setattr(probe_host.subprocess, "check_output", fake_check_output)

    assert probe_host.imported_source_state() == ("unknown", False)


def test_probe_script_id_hashes_the_executing_implementation() -> None:
    expected = hashlib.sha256(Path(probe_host.__file__).read_bytes()).hexdigest()
    assert probe_host.probe_script_id() == "sha256:" + expected


def _patch_current_probe_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {
        key: value
        for key, value in _preflight_attestation().items()
        if key != "probe_passed"
    }
    monkeypatch.setattr(probe_host, "probe_identity", lambda: dict(identity))
    monkeypatch.setattr(
        probe_host,
        "native_library_versions",
        lambda: dict(_NATIVE_VERSIONS),
    )
    monkeypatch.setattr(
        probe_host,
        "imported_source_state",
        lambda: (_SOURCE_COMMIT, True),
    )
    monkeypatch.setattr(probe_host, "core_build_id", lambda: _CURRENT_CORE_ID)
    monkeypatch.setattr(probe_host.platform, "node", lambda: "verified-host")
    monkeypatch.setattr(probe_host.vibeqc, "__version__", "0.15.34")


@pytest.mark.parametrize("producer", (run_case, run_case_b), ids=("gamma", "chi"))
@pytest.mark.parametrize("core_changed", (False, True), ids=("exact", "core-churn"))
def test_producers_emit_process_bound_composite_attestation(
    producer: object,
    core_changed: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stream = "aiccm2026dev-b" if producer is run_case_b else "aiccm2026dev-a"
    preflight = _preflight_attestation(
        core_id=_OLD_CORE_ID if core_changed else _CURRENT_CORE_ID
    )
    attestation_path = tmp_path / "preflight.json"
    attestation_path.write_text(json.dumps(preflight), encoding="utf-8")
    payload = _producer_payload(stream)
    monkeypatch.delenv("AICCM_SKIP_PROBE", raising=False)
    monkeypatch.setattr(producer, "_ATTESTATION", attestation_path)
    monkeypatch.setattr(producer, "_producer_payload", lambda: payload)
    _patch_current_probe_identity(monkeypatch)
    monkeypatch.setattr(probe_host, "check_symbols", lambda: [])
    monkeypatch.setattr(
        probe_host,
        "check_loaded_core_mirror",
        _loaded_core_check,
    )
    full_checks = 0

    def full_host_checks(**_kwargs: object) -> dict[str, object]:
        nonlocal full_checks
        full_checks += 1
        return {"passed": True}

    monkeypatch.setattr(probe_host, "_full_host_checks", full_host_checks)

    provenance = producer._provenance()

    assert provenance["host"] == "verified-host"
    assert provenance["vibeqc_commit"] == _SOURCE_COMMIT
    assert provenance["source_clean"] is True
    assert provenance["core_build_id"] == _CURRENT_CORE_ID
    assert provenance["probe_passed"] is True
    assert provenance["probe_version"] == probe_host.PRODUCER_ATTESTATION_VERSION
    assert provenance["probe_script_id"] == _PROBE_SCRIPT_ID
    composite = provenance["probe_attestation"]
    assert isinstance(composite, dict)
    assert composite["preflight_attestation"] == preflight
    assert composite["core_changed_since_preflight"] is core_changed
    assert composite["result_identity_stable"] is None
    assert provenance["probe_attestation_id"] == probe_host.probe_attestation_id(
        composite
    )
    assert provenance["producer_payload_id"] == probe_host.probe_attestation_id(
        payload
    )
    assert full_checks == int(core_changed)
    assert producer._finalize_provenance(provenance) is True
    assert provenance["probe_attestation"]["result_identity_stable"] is True


@pytest.mark.parametrize(
    ("field", "value", "also_change_core"),
    (
        ("host", "other-host", False),
        ("host", "other-host", True),
        ("vibeqc_version", "0.15.35", False),
        ("source_commit", "d" * 40, False),
        ("source_clean", False, False),
        ("probe_version", "aiccm-host-probe/v999", False),
        ("probe_script_id", "sha256:" + "e" * 64, False),
        ("probe_passed", False, False),
    ),
)
def test_producer_attestation_rejects_every_noncore_preflight_mismatch(
    field: str,
    value: object,
    also_change_core: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preflight = _preflight_attestation(
        core_id=_OLD_CORE_ID if also_change_core else _CURRENT_CORE_ID
    )
    preflight[field] = value
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(preflight), encoding="utf-8")
    _patch_current_probe_identity(monkeypatch)
    monkeypatch.setattr(
        probe_host,
        "check_symbols",
        lambda: pytest.fail("forbidden mismatch reached producer checks"),
    )

    assert probe_host.producer_attestation(
        path,
        _producer_payload("aiccm2026dev-b"),
    ) is None


def test_core_churn_requires_a_passing_full_v2_recheck(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(_preflight_attestation(core_id=_OLD_CORE_ID)),
        encoding="utf-8",
    )
    _patch_current_probe_identity(monkeypatch)
    monkeypatch.setattr(probe_host, "check_symbols", lambda: [])
    monkeypatch.setattr(
        probe_host,
        "check_loaded_core_mirror",
        _loaded_core_check,
    )
    monkeypatch.setattr(
        probe_host,
        "_full_host_checks",
        lambda **_kwargs: {"passed": False},
    )

    assert probe_host.producer_attestation(
        path,
        _producer_payload("aiccm2026dev-b"),
    ) is None


def test_producer_rejects_payload_from_a_different_attestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_preflight_attestation()), encoding="utf-8")
    payload = _producer_payload("aiccm2026dev-b")
    payload["files"]["probe_host"] = "sha256:" + "f" * 64
    _patch_current_probe_identity(monkeypatch)
    monkeypatch.setattr(
        probe_host,
        "check_symbols",
        lambda: pytest.fail("mismatched attestor payload reached checks"),
    )

    assert probe_host.producer_attestation(path, payload) is None


@pytest.mark.parametrize("malformed", ("canary", "libraries"))
def test_producer_rejects_malformed_passing_process_evidence(
    malformed: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_preflight_attestation()), encoding="utf-8")
    _patch_current_probe_identity(monkeypatch)
    monkeypatch.setattr(probe_host, "check_symbols", lambda: [])
    monkeypatch.setattr(
        probe_host,
        "check_loaded_core_mirror",
        (lambda: {"passed": True})
        if malformed == "canary"
        else _loaded_core_check,
    )
    libraries = dict(_NATIVE_VERSIONS)
    if malformed == "libraries":
        libraries.pop("blas")
    monkeypatch.setattr(
        probe_host,
        "native_library_versions",
        lambda: dict(libraries),
    )

    assert probe_host.producer_attestation(
        path,
        _producer_payload("aiccm2026dev-b"),
    ) is None


@pytest.mark.parametrize("producer", (run_case, run_case_b), ids=("gamma", "chi"))
def test_producers_leave_explicit_probe_skip_untrusted(
    producer: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICCM_SKIP_PROBE", "1")
    monkeypatch.setattr(
        probe_host,
        "producer_attestation",
        lambda *_args, **_kwargs: pytest.fail("explicit skip ran attestation"),
    )

    provenance = producer._provenance()

    assert provenance["probe_passed"] is False
    assert provenance["probe_attestation"] is None
    assert provenance["producer_payload_id"] == "unknown"


@pytest.mark.parametrize("producer", (run_case, run_case_b), ids=("gamma", "chi"))
def test_result_finalization_fails_closed_on_identity_churn(
    producer: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = "aiccm2026dev-b" if producer is run_case_b else "aiccm2026dev-a"
    payload = _producer_payload(stream)
    provenance = _producer_provenance(stream)
    provenance["probe_attestation"]["result_identity_stable"] = None
    provenance["probe_attestation_id"] = probe_host.probe_attestation_id(
        provenance["probe_attestation"]
    )
    changed = {
        key: value
        for key, value in _preflight_attestation(core_id=_OLD_CORE_ID).items()
        if key != "probe_passed"
    }
    monkeypatch.setattr(probe_host, "probe_identity", lambda: dict(changed))
    monkeypatch.setattr(
        probe_host,
        "native_library_versions",
        lambda: dict(_NATIVE_VERSIONS),
    )
    monkeypatch.setattr(producer, "_producer_payload", lambda: payload)

    assert producer._finalize_provenance(provenance) is False
    assert provenance["probe_passed"] is False
    assert provenance["probe_attestation"]["result_identity_stable"] is False


def test_producer_payloads_bind_every_directory_executable() -> None:
    a_payload = run_case._producer_payload()
    b_payload = run_case_b._producer_payload()
    d114_payload = run_d114_mpi._producer_payload()

    assert set(a_payload["files"]) == {
        "producer",
        "probe_host",
        "testset",
        "launcher",
    }
    assert set(b_payload["files"]) == {
        "producer",
        "probe_host",
        "testset",
        "b_routes",
        "launcher",
    }
    assert set(d114_payload["files"]) == set(b_payload["files"])
    for payload in (a_payload, b_payload, d114_payload):
        assert payload["files"]["probe_host"] == probe_host.probe_script_id()
        assert all(
            compare_b._is_sha256(value) for value in payload["files"].values()
        )


@pytest.mark.parametrize("producer", (run_case, run_case_b), ids=("gamma", "chi"))
def test_producers_snapshot_provenance_before_compute(
    producer: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    provenance = {"probe_passed": False, "snapshot": "before"}
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(producer, "_ATTESTATION", None)
    monkeypatch.setattr(
        producer,
        "_provenance",
        lambda: events.append("provenance") or provenance,
    )

    if producer is run_case:
        ccm = SimpleNamespace(
            nrep=(1, 1, 1),
            wsc_inscribed_radius=2.0,
        )
        monkeypatch.setattr(
            run_case,
            "_ccm",
            lambda *_args, **_kwargs: events.append("compute") or ccm,
        )
        monkeypatch.setattr(
            run_case,
            "_aiccm",
            lambda *_args, **_kwargs: (-1.0, -0.5, True, {}),
        )

        def write_a(_path, record, captured):  # noqa: ANN001
            events.append("write")
            assert captured is provenance
            emitted.append(record)

        monkeypatch.setattr(run_case, "_write_result", write_a)
        argv = [
            "run_case.py",
            "lih-rocksalt",
            "aiccm-hf",
            "--out",
            str(tmp_path),
        ]
    else:
        system = SimpleNamespace(
            unit_cell_molecule=lambda: SimpleNamespace(),
        )
        passing_canary = _qualified_ewald_shifted_pair_support()["canary"]
        monkeypatch.setattr(
            run_case_b,
            "_ewald_shifted_pair_canary",
            lambda: json.loads(json.dumps(passing_canary)),
        )
        monkeypatch.setattr(
            run_case_b,
            "_repair_is_ancestor",
            lambda _commit: True,
        )
        monkeypatch.setattr(
            testset,
            "build",
            lambda _name: events.append("compute") or system,
        )
        monkeypatch.setattr(run_case_b.vq, "BasisSet", lambda *_args: object())
        monkeypatch.setattr(
            run_case_b,
            "_scf_record",
            lambda *_args, **_kwargs: {
                "energy_per_cell_ha": -1.0,
                "converged": True,
            },
        )

        def write_b(_path, record, **_kwargs):  # noqa: ANN001
            events.append("write")
            emitted.append(record)
            return True

        monkeypatch.setattr(run_case_b, "_write", write_b)
        argv = [
            "run_case_b.py",
            "lih-rocksalt",
            "rhf-ri",
            "--out",
            str(tmp_path),
        ]

    monkeypatch.setattr(sys, "argv", argv)
    producer.main()

    assert events == ["provenance", "compute", "write"]
    assert emitted[0]["provenance"] is provenance
    if producer is run_case:
        assert emitted[0]["ccm_approach"] == "gamma-ccm"
        assert emitted[0]["ccm_construction"] == "union-and-weight"
        assert emitted[0]["route_role"] == "ccm-approach"


def test_loaded_core_canary_is_backend_environment_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")

    check = probe_host.check_loaded_core_mirror()

    assert check["version"] == probe_host.LOADED_CORE_CHECK_VERSION
    assert check["passed"] is True
    assert check["n_ao"] == 6
    assert check["max_l"] == 1
    assert check["max_scaled_error"] <= 1.0
    assert check["reference_transpose_asymmetry"] > (
        probe_host.CANARY_ASYMMETRY_MIN
    )
    assert os.environ["VIBEQC_AOPAIR_FT_BACKEND"] == "python"


def test_host_probe_emits_versioned_success_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_current_probe_identity(monkeypatch)
    monkeypatch.setattr(probe_host, "check_symbols", lambda: [])
    monkeypatch.setattr(probe_host, "check_g_max_convention", lambda: 23.8298)
    monkeypatch.setattr(
        probe_host,
        "check_mirror_fix",
        lambda: (1.0e-14, -1.0, -1.0),
    )
    monkeypatch.setattr(sys, "argv", ["probe_host.py"])

    assert probe_host.main() == 0
    output = capsys.readouterr().out
    assert f"probe     : {probe_host.PROBE_VERSION}" in output
    assert "VERDICT   : campaign-ready" in output


def test_host_probe_covers_the_b_producer_dependency_surface() -> None:
    required = {
        "vibeqc": {
            "Atom",
            "BasisSet",
            "get_num_threads",
            "PeriodicKSOptions",
            "PeriodicRHFOptions",
            "PeriodicSystem",
            "run_aiccm2026dev_b_rhf",
            "run_aiccm2026dev_b_rks",
            "run_aiccm2026dev_b_mp2",
            "run_aiccm2026dev_b_dlpno_mp2",
            "run_aiccm2026dev_b_dlpno_ccsd",
            "run_aiccm2026dev_b_dlpno_ccsd_t",
            "run_pbc_bipole_rhf",
            "run_pbc_bipole_rks",
            "scf_accelerator_from_string",
        },
        "vibeqc.dlpno.mp2": {"DLPNOMP2Options"},
        "vibeqc.dlpno.ccsd_local_solver": {"LocalCCSDOptions"},
    }

    assert set(probe_host.PRODUCER_MODULES) == {
        "run_case",
        "run_case_b",
        "run_d114_mpi",
    }
    for module, symbols in required.items():
        assert symbols <= set(probe_host.NEEDED[module])


@pytest.mark.parametrize(
    ("field", "updated"),
    (
        ("source_commit", "d" * 40),
        ("host", "other-host"),
        ("vibeqc_version", "0.15.35"),
        ("core_build_id", "sha256:" + "d" * 64),
        ("probe_script_id", "sha256:" + "e" * 64),
        ("probe_version", "aiccm-host-probe/v999"),
    ),
)
def test_host_probe_cache_is_bound_to_every_identity_field(
    field: str,
    updated: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = {
        "source_commit": "a" * 40,
        "host": "verified-host",
        "vibeqc_version": "0.15.34",
        "core_build_id": "sha256:" + "b" * 64,
        "probe_script_id": "sha256:" + "c" * 64,
    }
    symbol_checks = 0

    def check_symbols() -> list[str]:
        nonlocal symbol_checks
        symbol_checks += 1
        return []

    monkeypatch.setattr(probe_host, "check_symbols", check_symbols)
    monkeypatch.setattr(probe_host, "check_g_max_convention", lambda: 23.8298)
    monkeypatch.setattr(
        probe_host,
        "check_mirror_fix",
        lambda: (1.0e-14, -1.0, -1.0),
    )
    monkeypatch.setattr(
        probe_host,
        "imported_source_state",
        lambda: (identity["source_commit"], True),
    )
    monkeypatch.setattr(
        probe_host,
        "core_build_id",
        lambda: identity["core_build_id"],
    )
    monkeypatch.setattr(
        probe_host,
        "probe_script_id",
        lambda: identity["probe_script_id"],
    )
    monkeypatch.setattr(probe_host.platform, "node", lambda: identity["host"])
    monkeypatch.setattr(
        probe_host.vibeqc,
        "__version__",
        identity["vibeqc_version"],
    )

    first = tmp_path / "first.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_host.py",
            "--emit",
            str(first),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert probe_host.main() == 0

    if field == "vibeqc_version":
        monkeypatch.setattr(probe_host.vibeqc, "__version__", updated)
    elif field == "probe_version":
        monkeypatch.setattr(probe_host, "PROBE_VERSION", updated)
    else:
        identity[field] = updated
    second = tmp_path / "second.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_host.py",
            "--emit",
            str(second),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert probe_host.main() == 0

    assert symbol_checks == 2
    first_attestation = json.loads(first.read_text())
    second_attestation = json.loads(second.read_text())
    attestation_field = field
    assert first_attestation[attestation_field] != updated
    assert second_attestation[attestation_field] == updated
    assert len(list((tmp_path / "cache").glob("probe-*.json"))) == 2


def test_host_probe_reuses_only_an_exact_identity_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    symbol_checks = 0
    numerical_checks = 0

    def check_symbols() -> list[str]:
        nonlocal symbol_checks
        symbol_checks += 1
        return []

    def check_mirror_fix() -> tuple[float, float, float]:
        nonlocal numerical_checks
        numerical_checks += 1
        return 1.0e-14, -1.0, -1.0

    monkeypatch.setattr(probe_host, "check_symbols", check_symbols)
    monkeypatch.setattr(probe_host, "check_g_max_convention", lambda: 23.8298)
    monkeypatch.setattr(
        probe_host,
        "check_mirror_fix",
        check_mirror_fix,
    )
    monkeypatch.setattr(
        probe_host,
        "imported_source_state",
        lambda: ("a" * 40, True),
    )
    monkeypatch.setattr(
        probe_host,
        "core_build_id",
        lambda: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        probe_host,
        "probe_script_id",
        lambda: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(probe_host.platform, "node", lambda: "verified-host")
    monkeypatch.setattr(probe_host.vibeqc, "__version__", "0.15.34")

    emitted = []
    for name in ("first.json", "second.json"):
        path = tmp_path / name
        emitted.append(path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "probe_host.py",
                "--emit",
                str(path),
                "--cache-dir",
                str(tmp_path / "cache"),
            ],
        )
        assert probe_host.main() == 0

    assert symbol_checks == 2
    assert numerical_checks == 1
    assert json.loads(emitted[0].read_text()) == json.loads(emitted[1].read_text())
    assert len(list((tmp_path / "cache").glob("probe-*.json"))) == 1
    assert "cached identity-bound attestation" in capsys.readouterr().out


def test_host_probe_does_not_cache_an_untracked_or_dirty_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    symbol_checks = 0

    def check_symbols() -> list[str]:
        nonlocal symbol_checks
        symbol_checks += 1
        return []

    monkeypatch.setattr(probe_host, "check_symbols", check_symbols)
    monkeypatch.setattr(probe_host, "check_g_max_convention", lambda: 23.8298)
    monkeypatch.setattr(
        probe_host,
        "check_mirror_fix",
        lambda: (1.0e-14, -1.0, -1.0),
    )
    monkeypatch.setattr(
        probe_host,
        "imported_source_state",
        lambda: ("a" * 40, False),
    )
    monkeypatch.setattr(
        probe_host,
        "core_build_id",
        lambda: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        probe_host,
        "probe_script_id",
        lambda: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(probe_host.platform, "node", lambda: "verified-host")
    monkeypatch.setattr(probe_host.vibeqc, "__version__", "0.15.34")

    cache = tmp_path / "cache"
    for name in ("first.json", "second.json"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "probe_host.py",
                "--emit",
                str(tmp_path / name),
                "--cache-dir",
                str(cache),
            ],
        )
        assert probe_host.main() == 1

    assert symbol_checks == 2
    assert not cache.exists()


def test_coverage_profile_exercises_all_routes_and_keeps_low_d_fail_closed(
) -> None:
    jobs, unsupported = make_jobs_b.iter_jobs("coverage")
    assert len(unsupported) == 21
    low_dimensional = [
        (system, route, reason)
        for system, route, reason in unsupported
        if "SCF is restricted to 3D" in reason
    ]
    assert len(low_dimensional) == 18
    assert all(
        "SCF is restricted to 3D" in reason
        for _, _, reason in low_dimensional
    )
    assert {
        (system, route)
        for system, route, _ in low_dimensional
    } == {
        (system, route)
        for system in ("h-chain", "graphene")
        for route in b_routes.SCF_ROUTES
    }
    assert len(jobs) == 10
    assert {
        route
        for system, route, reason in unsupported
        if system == "lih-rocksalt" and "overlap-fold drift" in reason
    } == {"rhf-4c", "rks-pbe-4c", "rks-pbe0-4c"}
    assert {job.route for job in jobs} == set(b_routes.ROUTES) - {
        "rhf-4c",
        "rks-pbe-4c",
        "rks-pbe0-4c",
    }
    assert {testset.SYSTEMS[job.system]["dim"] for job in jobs} == {3}
    assert {job.host for job in jobs} == {"compute-large"}
    post_hf = [job for job in jobs if b_routes.ROUTES[job.route].post_hf]
    assert len(post_hf) == 4
    assert all("--mesh 2 1 1" in job.command for job in post_hf)
    assert all("--direct-cutoff-bohr 15.0" in job.command for job in jobs)


def test_known_critical_fold_skip_is_bound_to_exact_fixed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = testset.SYSTEMS["lih-rocksalt"]
    reason = make_jobs_b._known_input_support_reason(
        "lih-rocksalt",
        "sto-3g",
        "rhf-4c",
        meta,
        (2, 2, 2),
        15.0,
    )
    assert reason is not None
    assert "LiH-rocksalt/STO-3G 2x2x2, 15-bohr" in reason

    assert (
        make_jobs_b._known_input_support_reason(
            "lih-rocksalt",
            "sto-3g",
            "rhf-4c",
            meta,
            (1, 1, 1),
            15.0,
        )
        is None
    )
    assert (
        make_jobs_b._known_input_support_reason(
            "lih-rocksalt",
            "sto-3g",
            "rhf-4c",
            meta,
            (2, 2, 2),
            18.0,
        )
        is None
    )

    monkeypatch.setattr(
        make_jobs_b,
        "_geometry_signature",
        lambda _meta: ("changed-geometry",),
    )
    assert (
        make_jobs_b._known_input_support_reason(
            "lih-rocksalt",
            "sto-3g",
            "rhf-4c",
            meta,
            (2, 2, 2),
            15.0,
        )
        is None
    )


def test_fleet_resource_claim_fits_every_coverage_host() -> None:
    assert make_jobs_b.JOB_CPUS == 4
    assert make_jobs_b.JOB_MEMORY_MB == 8000


def test_fleet_preview_does_not_invite_unqualified_submission(
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_jobs_b.main(
        [
            "--profile",
            "coverage",
            "--system",
            "lih-rocksalt",
            "--route",
            "rhf-ri",
        ]
    )

    out = capsys.readouterr().out
    assert "Preview only: do not pipe this output to a shell" in out
    assert "then pipe it to sh" not in out


def test_fleet_jobs_select_the_vibeqc_development_interpreter() -> None:
    jobs, unsupported = make_jobs_b.iter_jobs("coverage")
    assert unsupported
    assert all(
        job.command.startswith(
            "bash run.sh --b "
        )
        for job in jobs
    )


@pytest.mark.parametrize("host", ("localhost", "127.0.0.1", "::1", "workstation"))
def test_vq_batch_refuses_localhost_host_override(
    host: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        make_jobs_b.main(["--host", host])

    assert excinfo.value.code == 2
    assert (
        "would submit the χ-CCM-B batch to this laptop"
        in capsys.readouterr().err
    )


def test_local_mode_remains_available_for_direct_smoke_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_jobs_b.main(
        [
            "--local",
            "--profile",
            "coverage",
            "--system",
            "lih-rocksalt",
            "--route",
            "rhf-ri",
            "--host",
            "localhost",
        ]
    )

    out = capsys.readouterr().out
    assert "bash run.sh --b lih-rocksalt rhf-ri" in out
    assert "vq submit" not in out


def test_probed_runner_uses_a_fresh_private_attestation_directory() -> None:
    wrapper = (_TESTSET_DIR / "run.sh").read_text(encoding="utf-8")
    create = wrapper.index('ATTESTATION_DIR="$(mktemp -d ')
    probe = wrapper.index('"$PY" "$_here/probe_host.py"')
    execute = wrapper.index('"$PY" "$_here/$case_script" "$@"')

    assert 'if [ "${1:-}" = "--b" ]' in wrapper
    assert 'case_script="run_case_b.py"' in wrapper
    assert 'ATTESTATION_ROOT="${AICCM_PROBE_TMPDIR:-${TMPDIR:-/tmp}}"' in wrapper
    assert 'aiccm-probe-attestation.XXXXXX' in wrapper
    assert 'ATTESTATION="$ATTESTATION_DIR/attestation.json"' in wrapper
    assert 'export AICCM_PROBE_ATTESTATION="$ATTESTATION"' in wrapper
    assert '--emit "$ATTESTATION"' in wrapper
    assert 'trap \'rm -rf "$ATTESTATION_DIR"\' EXIT' in wrapper
    assert "aiccm-probe-attestation-$$" not in wrapper
    assert create < probe < execute


def test_probed_runner_has_narrow_rank_safe_d114_d116_mpi_paths() -> None:
    wrapper = (_TESTSET_DIR / "run.sh").read_text(encoding="utf-8")
    d114_selector = wrapper.index('if [ "${1:-}" = "--d114-mpi" ]')
    d116_selector = wrapper.index(
        'elif [ "${1:-}" = "--d116-rks-mpi" ]'
    )
    probe = wrapper.index('"$PY" "$_here/probe_host.py"')
    mpi4py = wrapper.index('"$PY" -c "import mpi4py"')
    required = wrapper.index("export VIBEQC_MPI_REQUIRED=1")
    expected = wrapper.index('export VIBEQC_MPI_EXPECTED_SIZE="$mpi_ranks"')
    launch = wrapper.index('"$mpi_launcher" "${mpi_args[@]}"')

    assert wrapper.count('case_script="run_d114_mpi.py"') == 2
    assert "mpi_ranks=2" in wrapper
    assert 'mpi_validation_mode="d114-rhf"' in wrapper
    assert 'mpi_validation_mode="d116-rks-pbe"' in wrapper
    assert 'export OMP_NUM_THREADS="$threads_per_rank"' in wrapper
    assert "export I_MPI_PIN_DOMAIN=omp" in wrapper
    assert '--map-by "ppr:${mpi_ranks}:node:PE=${threads_per_rank}"' in wrapper
    assert "--bind-to core" in wrapper
    assert 'VQ_SCHEDULER_TASKS="$mpi_ranks"' not in wrapper
    assert 'scheduler_tasks="${VQ_SCHEDULER_TASKS:-}"' in wrapper
    assert "$mpi_selector owns both ranks; omit --scheduler-tasks" in wrapper
    assert 'export AICCM_MPI_VALIDATION_PROFILE="$mpi_validation_mode"' in wrapper
    assert "export AICCM_D114_MPI_LAUNCHED=1" in wrapper
    assert "export AICCM_D116_MPI_LAUNCHED=1" in wrapper
    assert "export VIBEQC_MPI_REQUIRED=1" in wrapper
    assert 'export VIBEQC_MPI_EXPECTED_SIZE="$mpi_ranks"' in wrapper
    assert "export OPENBLAS_NUM_THREADS=1" in wrapper
    assert d114_selector < d116_selector < probe < mpi4py < required < expected
    assert expected < launch


def test_runner_wrapper_selects_explicit_or_portable_interpreter() -> None:
    wrapper = (_TESTSET_DIR / "run.sh").read_text(encoding="utf-8")
    explicit = wrapper.index('PY="$VIBEQC_PYTHON"')
    checkout = wrapper.index('PY="$_here/../../.venv/bin/python"')
    fallback = wrapper.index('PY="$(command -v python)"')
    assert explicit < checkout < fallback
    assert '"${VIBEQC_PYTHON+x}" = x' in wrapper
    assert "VIBEQC_PYTHON must name an executable interpreter" in wrapper
    assert "/home/" not in wrapper
    assert "/Users/" not in wrapper


def test_fleet_record_helpers_preserve_scf_residual_context() -> None:
    result = SimpleNamespace(
        scf_trace=[
            SimpleNamespace(
                iter=1,
                energy=-1.0,
                delta_e=0.0,
                grad_norm=0.25,
                diis_subspace=0,
            ),
            SimpleNamespace(
                iter=2,
                energy=-1.1,
                delta_e=-0.1,
                grad_norm=0.04,
                diis_subspace=2,
            ),
        ]
    )
    tail = run_case_b._scf_trace_tail(result, n_tail=1)
    assert tail == [
        {
            "iter": 2,
            "energy_ha": -1.1,
            "delta_e_ha": -0.1,
            "grad_norm": 0.04,
            "diis_subspace": 2,
        }
    ]
    note = run_case_b._damping_interaction_note(
        SimpleNamespace(use_diis=True, damping=0.5)
    )
    assert note is not None
    assert "--diis-start" in note


def test_b_producer_serializes_d114_output_cell_farming() -> None:
    execution = SimpleNamespace(
        schema="vibeqc.pbc-bipole.output-cell-farming-execution/v1",
        active=True,
        task_kind="chi-direct-output-cell",
        strategy="cyclic",
        world_size=2,
        rank=1,
        global_task_count=7,
        local_task_count=3,
        local_task_counts=(4, 3),
        ordered_cell_fingerprint="a" * 64,
        complete_internal_translation_sum=True,
        result_distribution="allgather-complete-blocks",
    )
    result = SimpleNamespace(
        aiccm2026dev_b=SimpleNamespace(
            direct_output_cell_farming=execution,
        )
    )

    assert run_case_b._direct_output_cell_farming_record(result) == {
        "schema": "vibeqc.pbc-bipole.output-cell-farming-execution/v1",
        "active": True,
        "task_kind": "chi-direct-output-cell",
        "strategy": "cyclic",
        "world_size": 2,
        "rank": 1,
        "global_task_count": 7,
        "local_task_count": 3,
        "local_task_counts": [4, 3],
        "ordered_cell_fingerprint": "a" * 64,
        "complete_internal_translation_sum": True,
        "result_distribution": "allgather-complete-blocks",
    }
    assert (
        run_case_b._direct_output_cell_farming_record(SimpleNamespace())
        is None
    )


def test_d116_mpi_profile_is_fixed_and_distinct_from_d114() -> None:
    route = run_d114_mpi._validation_route("d116-rks-pbe")
    parity_contract = {
        "schema": "vibeqc.aiccm2026dev-b.d116-parity-contract/v2",
        "comparison": "absolute",
        "inclusive": True,
        "energy_components": ["total", "electronic", "nuclear", "xc"],
        "energy_abs_tolerance_ha": 1.0e-10,
        "fock_element_abs_tolerance": 1.0e-12,
        "density_element_abs_tolerance": 1.0e-12,
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
    assert route == {
        "schema": run_d114_mpi.D116_RKS_SCHEMA,
        "parallel_schema": run_d114_mpi.D116_PARALLEL_EXECUTION_SCHEMA,
        "parity_contract": parity_contract,
        "milestone": "D116",
        "method": "RKS",
        "functional": "PBE",
        "artifact": "chi-d116-c-diamond-rkspbe4c-mpi2.json",
    }
    assert route["schema"] == (
        "vibeqc.aiccm2026dev-b.d116-rks-mpi-validation/v2"
    )
    assert run_d114_mpi._d116_parity_contract() == parity_contract
    assert run_d114_mpi.D116_PARALLEL_EXECUTION_SCHEMA.endswith("/v1")
    assert run_d114_mpi._validation_route("d114-rhf")["schema"] == (
        run_d114_mpi.D114_SCHEMA
    )
    with pytest.raises(RuntimeError, match="unsupported"):
        run_d114_mpi._validation_route("d116-rks-hse06")

    fixed = {
        "mesh": (2, 2, 2),
        "basis": "sto-3g",
        "cutoff_bohr": 15.0,
        "max_iter": 1,
    }
    run_d114_mpi._validate_requested_inputs("d116-rks-pbe", **fixed)
    for field, value in (
        ("mesh", (3, 3, 3)),
        ("basis", "def2-svp"),
        ("cutoff_bohr", 18.0),
        ("max_iter", 2),
    ):
        changed = dict(fixed)
        changed[field] = value
        with pytest.raises(RuntimeError, match="D116 fixes"):
            run_d114_mpi._validate_requested_inputs(
                "d116-rks-pbe",
                **changed,
            )


def test_d116_mpi_profile_calls_only_fixed_pbe_rks_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    bloch_mesh = object()

    monkeypatch.setattr(
        run_d114_mpi,
        "cyclic_gamma_mesh",
        lambda _system, mesh: SimpleNamespace(
            requested_mesh=mesh,
            to_bloch_kmesh=lambda: bloch_mesh,
        ),
    )

    def fake_reference(*args, **kwargs):
        captured["reference_args"] = args
        captured["reference_kwargs"] = kwargs
        return "reference"

    def fake_farmed(*args, **kwargs):
        captured["farmed_args"] = args
        captured["farmed_kwargs"] = kwargs
        return "farmed"

    monkeypatch.setattr(run_d114_mpi.vq, "run_pbc_bipole_rks", fake_reference)
    monkeypatch.setattr(
        run_d114_mpi.vq,
        "run_aiccm2026dev_b_rks",
        fake_farmed,
    )

    reference, farmed = run_d114_mpi._run_reference_and_farmed(
        "d116-rks-pbe",
        object(),
        object(),
        (2, 2, 2),
        max_iter=1,
        cutoff_bohr=15.0,
    )

    assert (reference, farmed) == ("reference", "farmed")
    reference_args = captured["reference_args"]
    reference_kwargs = captured["reference_kwargs"]
    farmed_args = captured["farmed_args"]
    farmed_kwargs = captured["farmed_kwargs"]
    assert reference_args[2] is bloch_mesh
    assert reference_kwargs == {
        "functional": "pbe",
        "fock_mixing": 0.0,
        "use_ewald_j_split": True,
        "use_exchange_ewald_split": True,
        "exchange_exxdiv": "ewald",
        "use_multipole_far_field": False,
        "sr_image_precision": 1.0e-6,
        "farm_output_cells": False,
        "use_fock_symmetry": False,
        "use_fock_symmetry_reduce": False,
        "progress": False,
    }
    assert farmed_args[2:4] == ("pbe", (2, 2, 2))
    assert farmed_kwargs == {
        "backend": "four_center",
        "fock_mixing": 0.0,
        "symmetry_mode": "off",
        "progress": False,
    }
    reference_options = reference_args[3]
    farmed_options = farmed_args[4]
    assert reference_options is not farmed_options
    for options in (reference_options, farmed_options):
        assert options.functional == "pbe"
        assert options.max_iter == 1
        assert options.conv_tol_energy == pytest.approx(1.0e-8)
        assert options.conv_tol_grad == pytest.approx(1.0e-6)
        assert options.fock_mixing == 0.0
        assert options.lattice_opts.cutoff_bohr == 15.0
        assert options.lattice_opts.nuclear_cutoff_bohr == 15.0


def test_d114_mpi_payload_requires_exact_rank_parity_and_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = SimpleNamespace(
        schema=run_d114_mpi.EXPECTED_FARMING_SCHEMA,
        active=True,
        task_kind=run_d114_mpi.EXPECTED_TASK_KIND,
        strategy="cyclic",
        world_size=2,
        rank=0,
        global_task_count=7,
        local_task_count=4,
        local_task_counts=(4, 3),
        ordered_cell_fingerprint="b" * 64,
        complete_internal_translation_sum=True,
        result_distribution=run_d114_mpi.EXPECTED_DISTRIBUTION,
    )
    reference = SimpleNamespace(
        energy=-75.0,
        e_electronic=-80.0,
        e_nuclear=5.0,
        converged=False,
        n_iter=1,
        fock=(np.asarray([[1.0, 0.25], [0.25, 2.0]]),),
        density=SimpleNamespace(
            blocks=(np.asarray([[0.5, 0.0], [0.0, 0.5]]),)
        ),
    )
    farmed = SimpleNamespace(
        energy=reference.energy,
        e_electronic=reference.e_electronic,
        e_nuclear=reference.e_nuclear,
        converged=reference.converged,
        n_iter=reference.n_iter,
        fock=tuple(value.copy() for value in reference.fock),
        density=SimpleNamespace(
            blocks=tuple(value.copy() for value in reference.density.blocks)
        ),
        aiccm2026dev_b=SimpleNamespace(
            direct_output_cell_farming=execution,
        ),
    )
    monkeypatch.setattr(run_d114_mpi, "mpi_rank", lambda: 0)
    rank_zero = run_d114_mpi._rank_parity_record(reference, farmed)
    rank_zero["process_identity"] = {
        "vibeqc_commit": "c" * 40,
        "core_build_id": "sha256:" + "d" * 64,
        "producer_attestation_id": "sha256:" + "e" * 64,
    }
    rank_zero["parallel_execution"] = {
        "schema": run_d114_mpi.PARALLEL_EXECUTION_SCHEMA,
        "mpi_world_size": 2,
        "launcher_implementation": "intel-mpi",
        "binding_policy": "intel-domain-omp-compact",
        "omp_threads_per_rank": 10,
    }
    rank_one = json.loads(json.dumps(rank_zero))
    rank_one["rank"] = 1
    rank_one["farming"]["rank"] = 1
    rank_one["farming"]["local_task_count"] = 3

    run_d114_mpi._validate_rank_records([rank_zero, rank_one])
    assert rank_zero["exact_parity"] is True
    assert rank_zero["max_abs_fock_delta"] == 0.0
    assert rank_zero["max_abs_density_delta"] == 0.0

    farmed.fock[0][0, 0] += 1.0e-12
    assert run_d114_mpi._rank_parity_record(reference, farmed)[
        "exact_parity"
    ] is False


def test_d116_mpi_payload_applies_numeric_tolerances_and_exact_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy_tolerance_ha = 1.0e-10
    array_tolerance = 1.0e-12
    execution = SimpleNamespace(
        schema=run_d114_mpi.EXPECTED_FARMING_SCHEMA,
        active=True,
        task_kind=run_d114_mpi.EXPECTED_TASK_KIND,
        strategy="cyclic",
        world_size=2,
        rank=0,
        global_task_count=7,
        local_task_count=4,
        local_task_counts=(4, 3),
        ordered_cell_fingerprint="b" * 64,
        complete_internal_translation_sum=True,
        result_distribution=run_d114_mpi.EXPECTED_DISTRIBUTION,
    )
    reference = SimpleNamespace(
        energy=-75.0,
        e_electronic=-80.0,
        e_nuclear=5.0,
        e_xc=-0.25,
        fock_mixing=0.0,
        converged=False,
        n_iter=1,
        fock=(np.asarray([[1.0, 0.25], [0.25, 2.0]]),),
        density=SimpleNamespace(
            blocks=(np.asarray([[0.5, 0.0], [0.0, 0.5]]),)
        ),
    )
    farmed = SimpleNamespace(
        **{
            key: getattr(reference, key)
            for key in (
                "energy",
                "e_electronic",
                "e_nuclear",
                "e_xc",
                "fock_mixing",
                "converged",
                "n_iter",
            )
        },
        fock=tuple(value.copy() for value in reference.fock),
        density=SimpleNamespace(
            blocks=tuple(value.copy() for value in reference.density.blocks)
        ),
        aiccm2026dev_b=SimpleNamespace(
            direct_output_cell_farming=execution,
            finite_torus_convention=SimpleNamespace(
                exchange_q0_applicability="inactive",
            ),
        ),
    )
    farmed.energy += 0.25 * energy_tolerance_ha
    farmed.e_electronic -= 0.50 * energy_tolerance_ha
    farmed.e_nuclear += 0.25 * energy_tolerance_ha
    farmed.e_xc += 0.25 * energy_tolerance_ha
    farmed.fock[0][0, 0] += 0.50 * array_tolerance
    farmed.density.blocks[0][1, 1] -= 0.50 * array_tolerance
    monkeypatch.setattr(run_d114_mpi, "mpi_rank", lambda: 0)
    rank_zero = run_d114_mpi._rank_parity_record(
        reference,
        farmed,
        require_xc=True,
    )
    rank_zero["process_identity"] = {
        "vibeqc_commit": "c" * 40,
        "core_build_id": "sha256:" + "d" * 64,
        "producer_attestation_id": "sha256:" + "e" * 64,
    }
    rank_zero["parallel_execution"] = {
        "schema": run_d114_mpi.D116_PARALLEL_EXECUTION_SCHEMA,
        "mpi_world_size": 2,
        "launcher_implementation": "intel-mpi",
        "binding_policy": "intel-domain-omp-compact",
        "resource_semantics": "vq-total-cpus-divided-by-ranks",
        "declared_cpus": 20,
        "scheduler_tasks": None,
        "omp_threads_per_rank": 10,
        "native_max_threads": 10,
        "thread_environment": {
            "OMP_NUM_THREADS": "10",
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "spread",
            "OMP_PLACES": "cores",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
    }
    rank_one = json.loads(json.dumps(rank_zero))
    rank_one["rank"] = 1
    rank_one["farming"]["rank"] = 1
    rank_one["farming"]["local_task_count"] = 3
    # The production D116 run reproduced the floating observables on both
    # ranks to about 1e-14, but not bitwise.  Model that independent rank
    # ordering while leaving every discrete control byte-identical.
    cross_rank_delta = 2.5e-14
    for field in (
        "reference_energy_ha",
        "farmed_energy_ha",
        "reference_nuclear_energy_ha",
        "farmed_nuclear_energy_ha",
        "reference_electronic_energy_ha",
        "farmed_electronic_energy_ha",
        "reference_xc_energy_ha",
        "farmed_xc_energy_ha",
    ):
        rank_one[field] += cross_rank_delta
    for reference_field, farmed_field, delta_field in (
        ("reference_energy_ha", "farmed_energy_ha", "energy_delta_ha"),
        (
            "reference_nuclear_energy_ha",
            "farmed_nuclear_energy_ha",
            "nuclear_energy_delta_ha",
        ),
        (
            "reference_electronic_energy_ha",
            "farmed_electronic_energy_ha",
            "electronic_energy_delta_ha",
        ),
        (
            "reference_xc_energy_ha",
            "farmed_xc_energy_ha",
            "xc_energy_delta_ha",
        ),
    ):
        rank_one[delta_field] = (
            rank_one[farmed_field] - rank_one[reference_field]
        )

    run_d114_mpi._validate_rank_records(
        [rank_zero, rank_one],
        require_xc=True,
    )
    assert rank_zero["exact_parity"] is False
    assert rank_zero["contract_parity"] is True
    assert abs(rank_zero["xc_energy_delta_ha"]) <= energy_tolerance_ha
    assert rank_zero["max_abs_fock_delta"] <= array_tolerance
    assert rank_zero["max_abs_density_delta"] <= array_tolerance
    assert rank_zero["reference_fock_mixing"] == 0.0
    assert rank_zero["farmed_fock_mixing"] == 0.0
    assert rank_zero["exchange_q0_applicability"] == "inactive"

    boundary = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    for item in boundary:
        for reference_field, farmed_field, delta_field, delta in (
            (
                "reference_energy_ha",
                "farmed_energy_ha",
                "energy_delta_ha",
                1.0e-10,
            ),
            (
                "reference_nuclear_energy_ha",
                "farmed_nuclear_energy_ha",
                "nuclear_energy_delta_ha",
                -1.0e-10,
            ),
            (
                "reference_electronic_energy_ha",
                "farmed_electronic_energy_ha",
                "electronic_energy_delta_ha",
                1.0e-10,
            ),
            (
                "reference_xc_energy_ha",
                "farmed_xc_energy_ha",
                "xc_energy_delta_ha",
                -1.0e-10,
            ),
        ):
            item[reference_field] = 0.0
            item[farmed_field] = delta
            item[delta_field] = delta
        item["max_abs_fock_delta"] = array_tolerance
        item["max_abs_density_delta"] = array_tolerance
        item["contract_parity"] = True
    run_d114_mpi._validate_rank_records(boundary, require_xc=True)

    disagree = json.loads(json.dumps(rank_one))
    disagree["farmed_xc_energy_ha"] = (
        rank_zero["farmed_xc_energy_ha"] + 1.01 * energy_tolerance_ha
    )
    with pytest.raises(RuntimeError, match="farmed_xc_energy_ha"):
        run_d114_mpi._validate_rank_records(
            [rank_zero, disagree],
            require_xc=True,
        )

    false_delta = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    for item in false_delta:
        item["farmed_energy_ha"] = item["reference_energy_ha"] + 1.0
        item["energy_delta_ha"] = 0.0
        item["contract_parity"] = True
    with pytest.raises(RuntimeError, match="farmed_energy_ha"):
        run_d114_mpi._validate_rank_records(false_delta, require_xc=True)

    outside_fock = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    outside_fock[1]["max_abs_fock_delta"] = np.nextafter(
        array_tolerance,
        np.inf,
    )
    with pytest.raises(RuntimeError, match="max_abs_fock_delta"):
        run_d114_mpi._validate_rank_records(outside_fock, require_xc=True)

    wrong_shape = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    wrong_shape[1]["farmed_density_block_shapes"] = [[1, 4]]
    with pytest.raises(RuntimeError, match="density block shapes"):
        run_d114_mpi._validate_rank_records(wrong_shape, require_xc=True)

    wrong_q0 = [json.loads(json.dumps(item)) for item in (rank_zero, rank_one)]
    for item in wrong_q0:
        item["exchange_q0_applicability"] = "active"
    with pytest.raises(RuntimeError, match="applicability is not inactive"):
        run_d114_mpi._validate_rank_records(wrong_q0, require_xc=True)

    boolean_mixing = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    for item in boolean_mixing:
        item["reference_fock_mixing"] = False
        item["farmed_fock_mixing"] = False
    with pytest.raises(RuntimeError, match="zero Fock mixing"):
        run_d114_mpi._validate_rank_records(
            boolean_mixing,
            require_xc=True,
        )

    wrong_identity = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    wrong_identity[1]["process_identity"]["core_build_id"] = (
        "sha256:" + "f" * 64
    )
    with pytest.raises(RuntimeError, match="source/core process identity"):
        run_d114_mpi._validate_rank_records(wrong_identity, require_xc=True)

    wrong_fingerprint = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    wrong_fingerprint[1]["farming"]["ordered_cell_fingerprint"] = "a" * 64
    with pytest.raises(RuntimeError, match="farming field"):
        run_d114_mpi._validate_rank_records(
            wrong_fingerprint,
            require_xc=True,
        )

    wrong_parallel = [
        json.loads(json.dumps(item)) for item in (rank_zero, rank_one)
    ]
    wrong_parallel[1]["parallel_execution"]["binding_policy"] = "unbound"
    with pytest.raises(RuntimeError, match="parallel execution contract"):
        run_d114_mpi._validate_rank_records(wrong_parallel, require_xc=True)

    farmed.e_xc = reference.e_xc + 1.01 * energy_tolerance_ha
    outside_energy = run_d114_mpi._rank_parity_record(
        reference,
        farmed,
        require_xc=True,
    )
    assert outside_energy["contract_parity"] is False


def test_d114_mpi_payload_binds_hybrid_resource_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "AICCM_MPI_VALIDATION_PROFILE": "d114-rhf",
        "AICCM_D114_MPI_LAUNCHED": "1",
        "AICCM_MPI_LAUNCHER_KIND": "intel-mpi",
        "AICCM_MPI_BINDING_POLICY": "intel-domain-omp-compact",
        "AICCM_MPI_THREADS_PER_RANK": "10",
        "AICCM_MPI_DECLARED_CPUS": "20",
        "AICCM_MPI_RESOURCE_SEMANTICS": "vq-total-cpus-divided-by-ranks",
        "OMP_NUM_THREADS": "10",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("VQ_SCHEDULER_TASKS", raising=False)
    monkeypatch.delenv("AICCM_D116_MPI_LAUNCHED", raising=False)
    monkeypatch.setattr(run_d114_mpi, "mpi_size", lambda: 2)
    monkeypatch.setattr(run_d114_mpi.vq, "get_num_threads", lambda: 10)

    total = run_d114_mpi._parallel_execution_record()
    assert total == {
        "schema": run_d114_mpi.PARALLEL_EXECUTION_SCHEMA,
        "mpi_world_size": 2,
        "launcher_implementation": "intel-mpi",
        "binding_policy": "intel-domain-omp-compact",
        "resource_semantics": "vq-total-cpus-divided-by-ranks",
        "declared_cpus": 20,
        "scheduler_tasks": None,
        "omp_threads_per_rank": 10,
        "native_max_threads": 10,
        "thread_environment": {
            key: environment[key]
            for key in (
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
    }

    monkeypatch.setenv("AICCM_MPI_VALIDATION_PROFILE", "d116-rks-pbe")
    monkeypatch.delenv("AICCM_D114_MPI_LAUNCHED")
    monkeypatch.setenv("AICCM_D116_MPI_LAUNCHED", "1")
    d116 = run_d114_mpi._parallel_execution_record()
    assert d116["schema"] == run_d114_mpi.D116_PARALLEL_EXECUTION_SCHEMA

    monkeypatch.setenv("AICCM_D114_MPI_LAUNCHED", "1")
    with pytest.raises(RuntimeError, match="profile and launcher flags"):
        run_d114_mpi._parallel_execution_record()
    monkeypatch.delenv("AICCM_D114_MPI_LAUNCHED")

    monkeypatch.setenv("VQ_SCHEDULER_TASKS", "2")
    with pytest.raises(RuntimeError, match="total-CPU resource"):
        run_d114_mpi._parallel_execution_record()
    monkeypatch.delenv("VQ_SCHEDULER_TASKS")

    monkeypatch.setenv("AICCM_MPI_BINDING_POLICY", "unbound")
    with pytest.raises(RuntimeError, match="binding policy"):
        run_d114_mpi._parallel_execution_record()


def test_d114_mpi_payload_writes_atomically_only_on_rank_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    monkeypatch.setattr(run_d114_mpi, "mpi_rank", lambda: 0)
    run_d114_mpi._atomic_write_rank_zero(output, {"status": "passed"})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "passed"
    }
    assert not list(tmp_path.glob(".*.tmp-*"))

    monkeypatch.setattr(run_d114_mpi, "mpi_rank", lambda: 1)
    with pytest.raises(RuntimeError, match="only MPI rank zero"):
        run_d114_mpi._atomic_write_rank_zero(output, {"status": "wrong-rank"})


def test_b_campaign_help_marks_quadratic_fallback_inactive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_case_b.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        run_case_b.main()

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert "--quadratic-fallback-iter" in help_text
    assert "no B backend executes the fallback" in normalized_help
    assert "currently require 0" in normalized_help


def test_investigation_profile_targets_pbe_ri_diamond_convergence() -> None:
    jobs, unsupported = make_jobs_b.iter_jobs("investigate")
    assert not unsupported
    assert len(jobs) == 10
    assert {job.system for job in jobs} == {"c-diamond", "si-diamond"}
    assert {job.route for job in jobs} == {"rks-pbe-ri"}
    assert {job.variant for job in jobs} == {
        "default-diagnostics",
        "ediis-diis",
        "damping-diis40",
        "level-shift",
        "stress",
    }
    damping = next(job for job in jobs if job.variant == "damping-diis40")
    assert "--damping 0.5" in damping.command
    assert "--diis-start 40" in damping.command
    ediis = next(job for job in jobs if job.variant == "ediis-diis")
    assert "--scf-accelerator EDIIS_DIIS" in ediis.command


def test_paper1_profile_can_emit_crystal_matched_basis_override() -> None:
    jobs, unsupported = make_jobs_b.iter_jobs(
        "paper1",
        basis_override="pob-tzvp-rev2",
    )
    assert len(jobs) == 32
    assert len(unsupported) == 16
    assert {job.host for job in jobs} == {"compute-large"}
    assert all("--basis pob-tzvp-rev2" in job.command for job in jobs)
    assert {job.system for job in jobs} == {
        "c-diamond",
        "mgo",
        "al2o3-corundum",
        "nacl-rocksalt",
    }
    assert {
        (system, route)
        for system, route, _ in unsupported
    } == {
        (system, route)
        for system in ("uniform-h-chain", "h-chain")
        for route in make_jobs_b.PAPER1_ROUTES
    }
    assert any(job.route == "dlpno-ccsd-t" for job in jobs)


def test_full_profile_is_complete_and_marks_theory_gaps() -> None:
    jobs, unsupported = make_jobs_b.iter_jobs("full")
    assert jobs
    assert unsupported
    assert any(system == "nio-afm" for system, _, _ in unsupported)
    assert any("post-HF" in reason for _, _, reason in unsupported)
    assert all(not testset.SYSTEMS[job.system].get("open_shell", False)
               for job in jobs)
    for job in jobs:
        assert "--basis " in job.command
        assert "--aux-basis def2-svp-jk" in job.command
        assert "--mesh " in job.command
        assert "--gdf-method rsgdf" in job.command
        assert "--rsgdf-ke-cutoff 200" in job.command
        assert "--energy-tol 1e-8" in job.command
        assert "--gradient-tol 1e-6" in job.command


def test_skew_fleet_builders_preserve_fractional_coordinates() -> None:
    frac3d = np.asarray([0.17, 0.23, 0.31])
    bulk = testset._frac3d(
        4.1,
        5.2,
        6.3,
        73.0,
        [(2, *frac3d)],
    )
    bulk_position = np.asarray(bulk.unit_cell[0].xyz, dtype=float)
    np.testing.assert_allclose(
        np.linalg.solve(np.asarray(bulk.lattice, dtype=float), bulk_position),
        frac3d,
        atol=1.0e-14,
    )

    frac2d = np.asarray([0.19, 0.37, 0.0])
    sheet = testset._hex2d(2.46, [(6, frac2d[0], frac2d[1])])
    sheet_position = np.asarray(sheet.unit_cell[0].xyz, dtype=float)
    np.testing.assert_allclose(
        np.linalg.solve(np.asarray(sheet.lattice, dtype=float), sheet_position),
        frac2d,
        atol=1.0e-14,
    )


@pytest.mark.parametrize("route", b_routes.POST_HF_ROUTES)
def test_lower_dimensional_post_hf_is_rejected_explicitly(route: str) -> None:
    meta = testset.SYSTEMS["h-chain"]
    reason = b_routes.unsupported_reason(meta, route, tuple(meta["nrep_4c"]))
    assert reason is not None
    assert "restricted to 3D" in reason


@pytest.mark.parametrize("route", b_routes.SCF_ROUTES)
@pytest.mark.parametrize("system", ("h-chain", "graphene"))
def test_every_lower_dimensional_scf_route_is_rejected_explicitly(
    system: str,
    route: str,
) -> None:
    meta = testset.SYSTEMS[system]
    reason = b_routes.unsupported_reason(meta, route, tuple(meta["nrep_4c"]))
    assert reason is not None
    assert "SCF" in reason
    assert "restricted to 3D" in reason
    assert "wire/slab Coulomb kernel" in reason


def test_design_and_fleet_docs_do_not_claim_low_d_ri_is_runnable() -> None:
    design = (_ROOT / "docs" / "design_aiccm2026dev_b.md").read_text(
        encoding="utf-8"
    )
    guide = (_ROOT / "docs" / "user_guide" / "aiccm2026dev_b.md").read_text(
        encoding="utf-8"
    )
    readme = (_TESTSET_DIR / "README_B.md").read_text(encoding="utf-8")
    comparison_producer = (_TESTSET_DIR / "run_case_cmp.py").read_text(
        encoding="utf-8"
    )

    assert "RI and RIJCOSX routes\nremain executable" not in design
    assert "blocks every lower-dimensional SCF backend" in design
    assert (
        "Quick start - 3-D MgO RHF/RI route smoke; default KE is not quantitative"
        in guide
    )
    assert "vibeqc.aiccm2026dev-b.two-electron-support/v1" in design
    assert "vibeqc.aiccm2026dev-b.two-electron-support/v1" in readme
    assert _EXPECTED_EXACT_EXCHANGE_SCHEMA in design
    assert _EXPECTED_EXACT_EXCHANGE_SCHEMA in guide
    assert _EXPECTED_EXACT_EXCHANGE_SCHEMA in readme
    assert "short-range-direct" in design
    assert "full-range-minus-long-range" in guide
    assert "They remain `not-qualified`" in guide
    assert "`audit_b.py` and `compare_b.py` reject" in guide
    assert "Two-electron numerical-support qualification" in design
    assert "exact executed M5" in design
    assert "route-inconsistent, or unqualified quantitative rows fail" in design
    assert "Numerical-support status" in readme
    assert "independently applies the D93 and D103 support gates" in readme
    assert b_routes.OVERLAP_FOLD_SUPPORT_SCHEMA in design
    assert b_routes.OVERLAP_FOLD_SUPPORT_SCHEMA in guide
    assert b_routes.OVERLAP_FOLD_SUPPORT_SCHEMA in readme
    assert "route-specific" in readme
    assert "exact M5 `sr_image_precision=1e-6` policy" in readme
    assert "sr_image_extent_bohr" in readme
    assert "10 runnable jobs" in readme
    assert "All 1D/2D SCF" in readme
    assert "none is runnable" in readme
    assert "gamma-ccm-bloch-gdf" not in comparison_producer
    assert 'ROUTE = "neutral-fitted-torus-bloch-gdf"' in comparison_producer
    assert "route=ROUTE" in comparison_producer
    assert 'route_role="representation-control"' in comparison_producer
    assert 'ccm_approach="not-applicable"' in comparison_producer
    assert 'ccm_construction="neutral-fitted-torus"' in comparison_producer
    assert 'evaluation_representation="bloch-gdf"' in comparison_producer
    assert (
        'nuclear_repulsion_model="shared-3d-ewald-same-helper"'
        in comparison_producer
    )
    assert "vq.ewald_nuclear_repulsion(cell)" in comparison_producer
    assert "nuclear_repulsion_per_cell`` +" not in comparison_producer
    assert 'exchange_q0_derivation="wrapper-field+executed-path"' in (
        comparison_producer
    )
    assert "wrapper/executed-path mismatch" in comparison_producer
    crystal_template = json.loads(
        (_TESTSET_DIR / "crystal_refs_b.template.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        testset.SYSTEMS[system]["dim"] == 3
        for system in crystal_template
    )


@pytest.mark.parametrize(
    ("executed", "reference", "expected"),
    [
        (1.0, 1.0, True),
        (1.0 + 5e-13, 1.0, True),
        (1.0 + 2e-12, 1.0, False),
        (float("nan"), 1.0, False),
        (1.0, float("nan"), False),
        (float("inf"), float("inf"), False),
    ],
)
def test_comparison_producer_nuclear_match_fails_closed_on_nonfinite_evidence(
    executed: float,
    reference: float,
    expected: bool,
) -> None:
    _delta, matches = run_case_cmp._shared_e_nuclear_match(
        executed,
        reference,
    )
    assert matches is expected


@pytest.mark.parametrize(
    ("xi", "expected_label", "expected_applicability"),
    [
        (0.25, "BvK-ewald", "active"),
        (0.0, "unverified", "unverified"),
        (float("nan"), "unverified", "unverified"),
        (float("inf"), "unverified", "unverified"),
    ],
)
def test_comparison_producer_q0_derivation_requires_finite_madelung(
    monkeypatch,
    xi: float,
    expected_label: str,
    expected_applicability: str,
) -> None:
    monkeypatch.setattr(
        "vibeqc.periodic_k_gdf._madelung_for_kmesh",
        lambda *_args, **_kwargs: xi,
    )
    label, applicability, evidence = run_case_cmp.derive_exchange_q0(
        SimpleNamespace(),
        (2, 1, 1),
        "executed route: exxdiv='ewald'",
    )

    assert label == expected_label
    assert applicability == expected_applicability
    if np.isnan(xi):
        assert np.isnan(evidence["bvk_madelung_xi"])
    else:
        assert evidence["bvk_madelung_xi"] == xi


@pytest.mark.parametrize(
    ("artifact", "electronic_method"),
    (("rhf", "RHF"), ("rks", "RKS/PBE"), ("localized", "RHF")),
)
def test_diamond_b_qvf_artifacts_use_runner_owned_d89_vendor_payload(
    artifact: str,
    electronic_method: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _d89_chi_result(electronic_method)
    captured: dict[str, Any] = {}

    def fake_write_qvf(stem, _plan, **context):
        captured.update(context)
        return Path(stem).with_suffix(".qvf")

    monkeypatch.setattr(diamond_compare, "write_qvf", fake_write_qvf)
    if artifact == "localized":
        monkeypatch.setattr(diamond_compare, "qvf_wf_data", lambda *_a, **_k: {})
        diamond_compare.write_localized_qvf(
            tmp_path / "diamond-b-localized",
            molecule=SimpleNamespace(),
            basis=SimpleNamespace(),
            coefficients=np.eye(1),
            method="RHF/aiccm2026dev-b/ri/localized",
            basis_name="sto-3g",
            aiccm_b_result=result,
        )
    else:
        diamond_compare.write_periodic_qvf(
            tmp_path / f"diamond-b-{artifact}",
            system=SimpleNamespace(),
            basis_name="sto-3g",
            method=f"{electronic_method}/aiccm2026dev-b/ri",
            functional="pbe" if artifact == "rks" else None,
            bands=SimpleNamespace(),
            bonds=SimpleNamespace(bonds=[]),
            aiccm_b_result=result,
        )

    assert captured["extensions"]["x_vibeqc"]["version"] == "1.0"
    sections = captured["vendor_json_sections"]
    assert len(sections) == 1
    assert sections[0]["kind"] == "x_vibeqc.aiccm2026dev_b_convention"
    payload = sections[0]["payload"]
    expected = {
        "ccm_approach": "chi-ccm",
        "ccm_construction": "finite-translation-group-character",
        "evaluation_representation": "gamma-centred-character-mesh",
    }
    assert {field: payload[field] for field in expected} == expected
    assert payload["schema_version"] == 2
    assert payload["exact_exchange_assembly"] == {
        "c_full": 1.0,
        "c_sr": 0.0,
        "omega_screen_bohr_inv": 0.0,
        "screened_exchange_applicability": "inactive",
        "screened_exchange_assembly": "not-applicable",
        "schema": _EXPECTED_EXACT_EXCHANGE_SCHEMA,
        "resolver": _EXPECTED_EXACT_EXCHANGE_RESOLVER,
    }
    convention = payload["finite_torus_convention"]
    assert {field: convention[field] for field in expected} == expected


def test_diamond_b_qvf_fails_closed_without_exact_exchange_schema() -> None:
    result = _d89_chi_result()
    del result.aiccm2026dev_b.exact_exchange_assembly

    with pytest.raises(RuntimeError, match="versioned exact-exchange assembly"):
        diamond_compare._aiccm_b_qvf_context(result)


@pytest.mark.parametrize("localized", (False, True))
def test_diamond_b_qvf_artifacts_fail_closed_without_source_identity(
    localized: bool,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="construction identity cannot be omitted"):
        if localized:
            diamond_compare.write_localized_qvf(
                tmp_path / "diamond-b-localized",
                molecule=SimpleNamespace(),
                basis=SimpleNamespace(),
                coefficients=np.eye(1),
                method="RHF/aiccm2026dev-b/ri/localized",
                basis_name="sto-3g",
            )
        else:
            diamond_compare.write_periodic_qvf(
                tmp_path / "diamond-b-rhf",
                system=SimpleNamespace(),
                basis_name="sto-3g",
                method="RHF/aiccm2026dev-b/ri",
                functional=None,
                bands=SimpleNamespace(),
                bonds=SimpleNamespace(bonds=[]),
            )


def test_diamond_b_stream_attaches_results_to_all_three_qvf_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rhf = _d89_chi_result("RHF")
    rks = _d89_chi_result("RKS/PBE")
    periodic_sources: list[object] = []
    localized_sources: list[object] = []
    system = diamond_compare.diamond_primitive()
    basis = diamond_compare.vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    monkeypatch.setattr(
        diamond_compare.vq,
        "run_aiccm2026dev_b_rhf",
        lambda *_a, **_k: rhf,
    )
    monkeypatch.setattr(
        diamond_compare.vq,
        "run_aiccm2026dev_b_rks",
        lambda *_a, **_k: rks,
    )
    bonds = SimpleNamespace(bonds=[])
    bands = SimpleNamespace()
    monkeypatch.setattr(
        diamond_compare.vq,
        "aiccm2026dev_b_mayer_bond_orders",
        lambda *_a, **_k: bonds,
    )
    monkeypatch.setattr(
        diamond_compare.vq,
        "aiccm2026dev_b_band_structure",
        lambda *_a, **_k: bands,
    )
    monkeypatch.setattr(
        diamond_compare.vq,
        "localize_aiccm2026dev_b_occupied",
        lambda *_a, **_k: SimpleNamespace(coefficients=np.eye(2)),
    )
    monkeypatch.setattr(
        diamond_compare,
        "write_periodic_qvf",
        lambda *_a, **kwargs: periodic_sources.append(kwargs["aiccm_b_result"]),
    )
    monkeypatch.setattr(
        diamond_compare,
        "write_localized_qvf",
        lambda *_a, **kwargs: localized_sources.append(kwargs["aiccm_b_result"]),
    )
    monkeypatch.setattr(diamond_compare, "save_band_plot", lambda *_a, **_k: None)
    monkeypatch.setattr(
        diamond_compare,
        "summarize_b_result",
        lambda **kwargs: {"label": kwargs["label"]},
    )

    diamond_compare.run_b_stream(
        system=system,
        basis=basis,
        basis_name="sto-3g",
        extension=(1, 1, 1),
        functional="pbe",
        backend="ri",
        kpath=SimpleNamespace(),
        outdir=tmp_path,
        max_iter=2,
    )

    assert periodic_sources == [rhf, rks]
    assert localized_sources == [rhf]


def test_diamond_b_pno_json_carries_exact_finite_torus_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    convention = _d89_chi_convention()
    result = SimpleNamespace(
        e_corr_per_cell=-0.01,
        e_total_per_cell=-1.01,
        finite_torus_convention=convention,
        n_pairs=4,
        n_pairs_screened=0,
        localization="wannier",
        local_correlation_space=None,
    )
    seen = {}

    def _run(*_args, **kwargs):
        seen["options"] = kwargs["dlpno_options"]
        return result

    monkeypatch.setattr(
        diamond_compare.vq,
        "run_aiccm2026dev_b_dlpno_mp2",
        _run,
    )

    payload = diamond_compare.maybe_run_b_pno(
        system=SimpleNamespace(),
        basis=SimpleNamespace(),
        extension=(2, 2, 2),
        outdir=tmp_path,
    )

    assert payload is not None
    options = seen["options"]
    assert (
        options.n_frozen,
        options.tcut_pairs,
        options.tcut_pairs_weak,
        options.tcut_pno,
        options.tcut_pno_weak,
        options.tcut_mkn,
    ) == (0, 0.0, 1e-4, 1e-8, 1e-7, 1e-3)
    assert payload["dlpno_recipe"] == {
        "convention": "pre-#140/#448-explicit-all-electron",
        "n_frozen": 0,
        "tcut_pairs": 0.0,
        "tcut_pairs_weak": 1e-4,
        "tcut_pno": 1e-8,
        "tcut_pno_weak": 1e-7,
        "tcut_mkn": 1e-3,
    }
    expected = {
        "ccm_approach": "chi-ccm",
        "ccm_construction": "finite-translation-group-character",
        "evaluation_representation": "gamma-centred-character-mesh",
    }
    assert {
        field: payload["finite_torus_convention"][field]
        for field in expected
    } == expected
    stored = json.loads(
        (tmp_path / "diamond-aiccm2026dev-b-dlpno-mp2.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["finite_torus_convention"] == payload["finite_torus_convention"]
    assert stored["dlpno_recipe"] == payload["dlpno_recipe"]


def test_validation_docs_do_not_relabel_pbe0_ri_as_rijcosx() -> None:
    decisions = " ".join(
        (_ROOT / "docs" / "aiccm2026dev_b_decisions.md")
        .read_text(encoding="utf-8")
        .split()
    )
    handover = " ".join(
        (_ROOT / "handovers" / "HANDOVER_AICCM2026DEV_B.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "matching STO-3G/PBE0/RIJCOSX runs converged" not in decisions
    assert (
        "direct hybrid COSX smoke is the PBE0/RIJCOSX block above"
        not in handover
    )
    assert "PBE0/RIJCOSX validation remains open" in decisions
    assert "PBE0/RIJCOSX remains unrun" in handover


def test_four_center_runner_records_resolved_direct_lattice_cutoffs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lattice_options = SimpleNamespace(
        cutoff_bohr=15.0,
        nuclear_cutoff_bohr=25.0,
    )
    options = SimpleNamespace(lattice_opts=lattice_options)
    monkeypatch.setattr(run_case_b, "_scf_options", lambda *_args: options)

    convention = SimpleNamespace(
        ccm_approach="chi-ccm",
        ccm_construction="finite-translation-group-character",
        evaluation_representation="gamma-centred-character-mesh",
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability="active",
        boundary_model="3d-periodic",
        periodic_dimension=3,
        character_mesh_shape=(1, 1, 1),
        bvk_madelung_supercell_repetitions=(1, 1, 1),
        bvk_madelung_supercell_lattice_bohr=(
            (8.0, 0.0, 0.0),
            (0.0, 8.0, 0.0),
            (0.0, 0.0, 8.0),
        ),
        lattice_vector_convention="columns",
        orbital_energy_convention="declared exchange-q=0 seam",
    )
    exact_exchange_assembly = _rhf_exact_exchange_assembly()
    diagnostics = SimpleNamespace(
        scf_trace_length=1,
        final_scf_delta_e_ha=0.0,
        final_scf_grad_norm=0.0,
        final_scf_diis_subspace=0,
        scf_accelerator="DIIS",
        use_diis=True,
        diis_start_iter=2,
        diis_subspace_size=8,
        damping=0.0,
        dynamic_damping=False,
        fock_mixing=0.25,
        level_shift=0.0,
        level_shift_warmup_cycles=0,
        smearing_temperature=0.0,
        density_idempotency_error=0.0,
        electron_count_error=0.0,
        wigner_seitz_partition_error=0.0,
        inverse_bloch_imaginary_residual=0.0,
        exact_exchange_assembly=exact_exchange_assembly,
    )
    result = SimpleNamespace(
        backend="aiccm2026dev-b-four_center",
        runtime_backend="pbc-bipole",
        energy=-1.0,
        e_electronic=-1.5,
        e_nuclear=0.5,
        converged=True,
        n_iter=1,
        finite_torus_convention=convention,
        exact_exchange_assembly=exact_exchange_assembly,
        aiccm2026dev_b=diagnostics,
        mo_energies=([-0.5, 0.2],),
        scf_trace=(),
        sr_image_extent_bohr=28.0,
        sr_image_domain_policy=b_routes.DIRECT_DOMAIN_POLICY,
        sr_image_precision=b_routes.DIRECT_SR_IMAGE_PRECISION,
        overlap_fold_drift=5.0e-5,
    )

    executed: list[dict[str, Any]] = []

    def fake_run(**kwargs):
        # The 3-D direct driver resolves the neutral-cell cutoff policy by
        # mutating the caller-owned options before returning.
        executed.append(kwargs)
        lattice_options.nuclear_cutoff_bohr = lattice_options.cutoff_bohr
        backend = kwargs["backend"]
        if backend == "four_center":
            result.runtime_backend = "pbc-bipole"
        else:
            exchange_backend = "cosx" if backend == "rijcosx" else "gdf"
            result.runtime_backend = (
                f"native-multi-k-gdf-{exchange_backend}-rhf"
            )
        return result

    monkeypatch.setattr(run_case_b.vq, "run_aiccm2026dev_b_rhf", fake_run)
    args = SimpleNamespace(
        aux_basis=None,
        direct_cutoff_bohr=15.0,
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0,
        mdf_ke_cutoff=40.0,
        progress=False,
        use_diis=True,
        damping=0.0,
    )
    system = SimpleNamespace(
        lattice=np.eye(3) * 8.0,
        n_electrons=lambda: 2,
    )

    record = run_case_b._scf_record(
        system,
        SimpleNamespace(),
        (1, 1, 1),
        "rhf-4c",
        args,
    )

    assert record["direct_lattice_cutoffs"] == {
        "electronic_cutoff_bohr": 15.0,
        "nuclear_cutoff_bohr": 15.0,
    }
    assert record["two_electron_support"] == _qualified_direct_support()
    assert record["overlap_fold_support"] == _qualified_overlap_fold_support(
        mesh=(1, 1, 1)
    )
    assert "ewald_shifted_pair_support" not in record
    assert (
        b_routes.overlap_fold_support_reportability_failure(
            record["overlap_fold_support"],
            "rhf-4c",
            record["direct_lattice_cutoffs"],
            [1, 1, 1],
        )
        is None
    )
    assert (
        b_routes.two_electron_support_reportability_failure(
            record["two_electron_support"],
            "rhf-4c",
            record["direct_lattice_cutoffs"],
        )
        is None
    )
    np.testing.assert_allclose(
        record["primitive_lattice_bohr"],
        np.eye(3) * 8.0,
    )
    assert record["finite_torus_convention"]["exchange_q0_applicability"] == (
        "active"
    )
    assert record["finite_torus_convention"]["ccm_approach"] == "chi-ccm"
    assert (
        record["finite_torus_convention"]["ccm_construction"]
        == "finite-translation-group-character"
    )
    assert (
        record["finite_torus_convention"]["evaluation_representation"]
        == "gamma-centred-character-mesh"
    )
    assert record["finite_torus_convention"]["lattice_vector_convention"] == (
        "columns"
    )
    assert record["exact_exchange_assembly"] == _fleet_exact_exchange_assembly(
        "rhf-4c"
    )
    assert record["convergence_diagnostics"]["fock_mixing"] == pytest.approx(
        0.25
    )

    lattice_options.nuclear_cutoff_bohr = 25.0
    args.gdf_method = "mdf"
    args.mdf_ke_cutoff = 55.0
    ri_record = run_case_b._scf_record(
        system,
        SimpleNamespace(),
        (1, 1, 1),
        "rhf-ri",
        args,
    )
    assert ri_record["direct_lattice_cutoffs"] is None
    assert executed[-1]["gdf_method"] == "mdf"
    assert executed[-1]["mdf_ke_cutoff"] == pytest.approx(55.0)
    assert ri_record["two_electron_support"]["qualification"] == "not-qualified"
    assert ri_record["overlap_fold_support"] == (
        _not_applicable_overlap_fold_support("rhf-ri")
    )
    assert "ewald_shifted_pair_support" not in ri_record
    assert ri_record["two_electron_support"]["fitted"] == {
        "gdf_method": "mdf",
        "rsgdf_base_ke_cutoff_ha": None,
        "rsgdf_tail_ke_cutoff_ha": None,
        "mdf_ke_cutoff_ha": 55.0,
        "cosx_exchange_support": "not-applicable",
        "correlation_support": "not-applicable",
    }
    assert (
        b_routes.two_electron_support_schema_failure(
            ri_record["two_electron_support"],
            "rhf-ri",
        )
        is None
    )

    args.gdf_method = "rsgdf"
    args.rsgdf_ke_cutoff = 360.0
    rijcosx_record = run_case_b._scf_record(
        system,
        SimpleNamespace(),
        (2, 1, 1),
        "rhf-rijcosx",
        args,
    )
    assert rijcosx_record["two_electron_support"]["qualification"] == (
        "not-qualified"
    )
    assert rijcosx_record["overlap_fold_support"] == (
        _not_applicable_overlap_fold_support("rhf-rijcosx")
    )
    assert "ewald_shifted_pair_support" not in rijcosx_record
    assert rijcosx_record["two_electron_support"]["fitted"] == {
        "gdf_method": "rsgdf",
        "rsgdf_base_ke_cutoff_ha": 360.0,
        "rsgdf_tail_ke_cutoff_ha": None,
        "mdf_ke_cutoff_ha": None,
        "cosx_exchange_support": "not-attested",
        "correlation_support": "not-applicable",
    }
    assert (
        b_routes.two_electron_support_schema_failure(
            rijcosx_record["two_electron_support"],
            "rhf-rijcosx",
        )
        is None
    )


@pytest.mark.parametrize(
    "extent",
    (None, float("nan"), float("inf"), 15.0, 14.0),
    ids=("missing", "nan", "infinite", "equal-cutoff", "inside-cutoff"),
)
def test_four_center_producer_does_not_qualify_invalid_m5_extent(
    extent: float | None,
) -> None:
    result = SimpleNamespace(
        backend="aiccm2026dev-b-four_center",
        runtime_backend="pbc-bipole",
        sr_image_extent_bohr=extent,
        sr_image_domain_policy=b_routes.DIRECT_DOMAIN_POLICY,
        sr_image_precision=b_routes.DIRECT_SR_IMAGE_PRECISION,
    )

    support = run_case_b._direct_two_electron_support(
        result,
        "rhf-4c",
        {
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
        },
    )

    assert support["qualification"] == "not-qualified"
    assert (
        b_routes.two_electron_support_reportability_failure(support, "rhf-4c")
        is not None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime_backend", "native-multi-k-gdf-gdf-rhf"),
        ("domain_policy", "m4a-explicit-radius/v1"),
        ("domain_policy", "m5-physical-pair-midpoint-erfc/v1"),
        ("sr_image_precision", None),
        ("sr_image_precision", 1.0e-5),
    ),
)
def test_four_center_producer_does_not_qualify_unbound_m5_provenance(
    field: str,
    value: object,
) -> None:
    result = SimpleNamespace(
        runtime_backend="pbc-bipole",
        sr_image_extent_bohr=28.0,
        sr_image_domain_policy=b_routes.DIRECT_DOMAIN_POLICY,
        sr_image_precision=b_routes.DIRECT_SR_IMAGE_PRECISION,
    )
    if field == "domain_policy":
        result.sr_image_domain_policy = value
    else:
        setattr(result, field, value)

    support = run_case_b._direct_two_electron_support(
        result,
        "rhf-4c",
        {
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
        },
    )

    assert support["qualification"] == "not-qualified"
    assert (
        b_routes.two_electron_support_reportability_failure(support, "rhf-4c")
        is not None
    )


@pytest.mark.parametrize(
    ("drift", "qualification", "reportable"),
    (
        (0.0, "qualified", True),
        (1.0e-4, "qualified", True),
        (1.0001e-4, "not-qualified", False),
        (1.0e-3, "not-qualified", False),
        (1.0e-2, "not-qualified", False),
        (None, "not-qualified", False),
    ),
)
def test_overlap_fold_support_binds_quantitative_target(
    drift: float | None,
    qualification: str,
    reportable: bool,
) -> None:
    support = _qualified_overlap_fold_support(
        drift=0.0 if drift is None else drift,
    )
    support["max_k_drift"] = drift
    support["qualification"] = qualification
    support["reason"] = "exact test evidence"

    assert (
        b_routes.overlap_fold_support_schema_failure(support, "rhf-4c")
        is None
    )
    failure = b_routes.overlap_fold_support_reportability_failure(
        support,
        "rhf-4c",
        {
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
        },
        [2, 2, 2],
    )
    assert (failure is None) is reportable


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.pop("reason"), "exact v1 keys"),
        (
            lambda value: value.__setitem__("backend", "ri"),
            "backend contradicts",
        ),
        (
            lambda value: value.__setitem__("max_k_drift", True),
            "finite and nonnegative",
        ),
        (
            lambda value: value.__setitem__("max_k_drift", 1.01e-2),
            "exceeds the fail-closed threshold",
        ),
        (
            lambda value: value.__setitem__("quantitative_target", 1.0e-3),
            "wrong quantitative target",
        ),
        (
            lambda value: value.__setitem__("stop_threshold", 1.0e-1),
            "wrong stop threshold",
        ),
        (
            lambda value: value.__setitem__("diagnostic", "frobenius"),
            "wrong diagnostic",
        ),
        (
            lambda value: value.__setitem__("reference_cutoff_factor", 2.0),
            "wrong reference factor",
        ),
    ),
)
def test_overlap_fold_support_rejects_tampering(
    mutation,
    message: str,
) -> None:
    support = _qualified_overlap_fold_support()
    mutation(support)

    assert message in (
        b_routes.overlap_fold_support_schema_failure(support, "rhf-4c") or ""
    )


def test_overlap_fold_support_binds_character_mesh() -> None:
    support = _qualified_overlap_fold_support()
    support["character_mesh_shape"] = [2, 2, 1]

    failure = b_routes.overlap_fold_support_reportability_failure(
        support,
        "rhf-4c",
        {
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
        },
        [2, 2, 2],
    )

    assert failure == "overlap_fold_support contradicts the character mesh"


@pytest.mark.parametrize(
    "mesh",
    ([True, 2, 2], [2, 0, 2], [2, 2], "2x2x2"),
)
def test_overlap_fold_support_rejects_malformed_row_mesh(mesh: object) -> None:
    failure = b_routes.overlap_fold_support_reportability_failure(
        _qualified_overlap_fold_support(),
        "rhf-4c",
        {
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
        },
        mesh,
    )

    assert failure == "the row character mesh must be three positive integers"


@pytest.mark.parametrize(
    ("drift", "qualification"),
    (
        (5.0e-5, "qualified"),
        (1.0e-3, "not-qualified"),
        (None, "not-qualified"),
        (float("nan"), "not-qualified"),
        (True, "not-qualified"),
    ),
)
def test_four_center_producer_serializes_overlap_fold_support(
    drift: object,
    qualification: str,
) -> None:
    support = run_case_b._overlap_fold_support(
        SimpleNamespace(overlap_fold_drift=drift),
        "rhf-4c",
        {
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
        },
        (2, 2, 2),
    )

    assert support["qualification"] == qualification
    assert (
        b_routes.overlap_fold_support_schema_failure(support, "rhf-4c")
        is None
    )


def test_fitted_overlap_fold_support_is_explicitly_not_applicable() -> None:
    support = run_case_b._overlap_fold_support(
        SimpleNamespace(),
        "rhf-ri",
        None,
        (2, 2, 2),
    )

    assert support == _not_applicable_overlap_fold_support("rhf-ri")
    assert (
        b_routes.overlap_fold_support_reportability_failure(
            support,
            "rhf-ri",
            None,
            [2, 2, 2],
        )
        is None
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.pop("reason"), "exact v2 keys"),
        (
            lambda value: value.__setitem__("schema", "legacy-unversioned"),
            "wrong schema",
        ),
        (
            lambda value: value.__setitem__("qualification", "claimed"),
            "invalid qualification",
        ),
        (
            lambda value: value.__setitem__("qualification", []),
            "invalid qualification",
        ),
        (
            lambda value: value.__setitem__("implementation", "legacy"),
            "wrong implementation state",
        ),
        (
            lambda value: value.__setitem__("repair_commit", "a" * 40),
            "wrong repair commit",
        ),
        (
            lambda value: value.__setitem__("core_build_id", "unknown"),
            "core_build_id is malformed",
        ),
        (
            lambda value: value.__setitem__("repair_is_ancestor", False),
            "complete post-repair producer identity",
        ),
        (
            lambda value: value["canary"].__setitem__(
                "lattice_shift", [37.0, 0.0, 0.0]
            ),
            "wrong lattice shift",
        ),
        (
            lambda value: value["canary"].__setitem__("passed", False),
            "canary contradicts its result",
        ),
        (
            lambda value: value["canary"].__setitem__("abs_delta_ha", 1e-11),
            "canary contradicts its result",
        ),
    ),
)
def test_d104_shifted_pair_support_v2_rejects_tampering(
    mutation,
    message: str,
) -> None:
    support = _qualified_ewald_shifted_pair_support()
    mutation(support)

    assert message in (
        b_routes.ewald_shifted_pair_support_schema_failure(support) or ""
    )


def test_d102_v1_record_stays_quarantined_after_d104_repair() -> None:
    failure = b_routes.d102_absolute_energy_revision_failure(
        _legacy_unqualified_ewald_shifted_pair_support_v1()
    )

    assert failure is not None
    assert "exact v2 keys" in failure


def test_d104_v2_failed_canary_is_well_formed_but_not_reportable() -> None:
    support = _qualified_ewald_shifted_pair_support()
    support.update(
        qualification="not-qualified",
        reason="the same-process repair canary failed",
    )
    support["canary"].update(
        shifted_energy_ha=-72.0,
        abs_delta_ha=1.0,
        passed=False,
    )

    assert b_routes.ewald_shifted_pair_support_schema_failure(support) is None
    assert "not-qualified" in (
        b_routes.ewald_shifted_pair_support_reportability_failure(support) or ""
    )


def test_d104_v2_support_is_cross_bound_to_final_provenance() -> None:
    provenance = _producer_provenance("aiccm2026dev-b")
    support = _qualified_ewald_shifted_pair_support(provenance)
    provenance["core_build_id"] = "sha256:" + "f" * 64

    assert "contradicts producer provenance" in (
        b_routes.ewald_shifted_pair_support_reportability_failure(
            support,
            provenance,
        )
        or ""
    )


def test_d104_v2_rejects_a_minimal_fabricated_probe_attestation() -> None:
    provenance = _producer_provenance("aiccm2026dev-b")
    support = _qualified_ewald_shifted_pair_support(provenance)
    fabricated_id = "sha256:" + "d" * 64
    provenance["probe_attestation"] = {
        "source_commit": support["producer_commit"],
        "core_build_id": support["core_build_id"],
        "source_clean": True,
        "probe_passed": True,
    }
    provenance["probe_attestation_id"] = fabricated_id
    support["probe_attestation_id"] = fabricated_id

    assert "finalized D77 provenance" in (
        b_routes.ewald_shifted_pair_support_reportability_failure(
            support,
            provenance,
        )
        or ""
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda attestation: attestation["producer_payload"]["files"].__setitem__(
            "producer", "sha256:" + "e" * 64
        ),
        lambda attestation: attestation["preflight_attestation"].__setitem__(
            "host", "forged-preflight"
        ),
        lambda attestation: attestation["current_core_attestation"].__setitem__(
            "host", "forged-current"
        ),
    ),
    ids=("payload", "preflight", "current"),
)
def test_d104_v2_rejects_nested_d77_attestation_tampering(mutation) -> None:
    provenance = _producer_provenance("aiccm2026dev-b")
    support = _qualified_ewald_shifted_pair_support(provenance)
    mutation(provenance["probe_attestation"])

    assert "finalized D77 provenance" in (
        b_routes.ewald_shifted_pair_support_reportability_failure(
            support,
            provenance,
        )
        or ""
    )


def test_d104_native_shift_canary_passes_on_the_repaired_core() -> None:
    canary = run_case_b._ewald_shifted_pair_canary()

    assert canary["schema"] == b_routes.EWALD_SHIFTED_PAIR_CANARY_SCHEMA
    assert canary["fixture"] == b_routes.EWALD_SHIFTED_PAIR_CANARY_FIXTURE
    assert canary["passed"] is True
    assert canary["abs_delta_ha"] <= (
        b_routes.EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
    )


def test_same_helper_control_emits_the_exact_d104_v2_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        run_case_cmp.EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA
        == b_routes.EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA
    )
    assert (
        run_case_cmp.EWALD_SHIFTED_PAIR_IMPLEMENTATION
        == b_routes.EWALD_SHIFTED_PAIR_IMPLEMENTATION
    )
    assert (
        run_case_cmp.EWALD_SHIFTED_PAIR_REPAIR_COMMIT
        == b_routes.EWALD_SHIFTED_PAIR_REPAIR_COMMIT
    )
    assert (
        run_case_cmp.EWALD_SHIFTED_PAIR_CANARY_SCHEMA
        == b_routes.EWALD_SHIFTED_PAIR_CANARY_SCHEMA
    )
    assert (
        run_case_cmp.EWALD_SHIFTED_PAIR_CANARY_FIXTURE
        == b_routes.EWALD_SHIFTED_PAIR_CANARY_FIXTURE
    )
    assert (
        run_case_cmp.EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
        == b_routes.EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
    )
    provenance = _producer_provenance("aiccm2026dev-a")
    provenance["source_commit"] = provenance.pop("vibeqc_commit")
    expected = _qualified_ewald_shifted_pair_support(provenance)
    monkeypatch.setattr(run_case_cmp, "_repair_is_ancestor", lambda _sha: True)
    monkeypatch.setattr(
        run_case_cmp,
        "_ewald_shifted_pair_canary",
        lambda: json.loads(json.dumps(expected["canary"])),
    )
    assert run_case_cmp._ewald_shifted_pair_support(provenance) == (
        expected
    )


@pytest.mark.parametrize("route_name", ("rhf-ri", "rhf-rijcosx", "ri-mp2"))
def test_fitted_support_rejects_runtime_backend_contradiction(
    route_name: str,
) -> None:
    support = _unqualified_fitted_support(route_name)
    support["runtime_backend"] = "pbc-bipole"

    assert "runtime_backend contradicts" in (
        b_routes.two_electron_support_schema_failure(support, route_name) or ""
    )


def test_fleet_record_distinguishes_requested_and_executed_fock_mixing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    emitted: list[dict[str, Any]] = []
    system = SimpleNamespace(unit_cell_molecule=lambda: SimpleNamespace())
    monkeypatch.setattr(run_case_b, "_ATTESTATION", None)
    monkeypatch.setattr(
        run_case_b,
        "_provenance",
        lambda: {"probe_passed": False},
    )
    passing_canary = _qualified_ewald_shifted_pair_support()["canary"]
    monkeypatch.setattr(
        run_case_b,
        "_ewald_shifted_pair_canary",
        lambda: json.loads(json.dumps(passing_canary)),
    )
    monkeypatch.setattr(
        run_case_b,
        "_repair_is_ancestor",
        lambda _commit: True,
    )
    monkeypatch.setattr(testset, "build", lambda _name: system)
    monkeypatch.setattr(run_case_b.vq, "BasisSet", lambda *_args: object())
    monkeypatch.setattr(
        run_case_b,
        "_scf_record",
        lambda *_args, **_kwargs: {
            "energy_per_cell_ha": -1.0,
            "converged": True,
            "convergence_diagnostics": {"fock_mixing": 0.30},
        },
    )
    monkeypatch.setattr(
        run_case_b,
        "_write",
        lambda _path, record, **_kwargs: emitted.append(record) or True,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_b.py",
            "lih-rocksalt",
            "rks-pbe-4c",
            "--no-diis",
            "--fock-mixing",
            "0.0",
            "--out",
            str(tmp_path),
        ],
    )

    run_case_b.main()

    assert len(emitted) == 1
    assert emitted[0]["scf_options"]["fock_mixing"] == pytest.approx(0.0)
    assert emitted[0]["convergence_diagnostics"]["fock_mixing"] == pytest.approx(
        0.30
    )


def test_fleet_stops_before_scf_when_d104_canary_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = _producer_provenance("aiccm2026dev-b")
    failed_canary = _qualified_ewald_shifted_pair_support(provenance)["canary"]
    failed_canary.update(
        shifted_energy_ha=-72.0,
        abs_delta_ha=1.0,
        passed=False,
    )
    monkeypatch.setattr(run_case_b, "_ATTESTATION", None)
    monkeypatch.setattr(run_case_b, "_provenance", lambda: provenance)
    monkeypatch.setattr(
        run_case_b,
        "_repair_is_ancestor",
        lambda _commit: True,
    )
    monkeypatch.setattr(
        run_case_b,
        "_ewald_shifted_pair_canary",
        lambda: json.loads(json.dumps(failed_canary)),
    )
    monkeypatch.setattr(run_case_b, "_finalize_provenance", lambda _value: True)
    monkeypatch.setattr(
        testset,
        "build",
        lambda _name: pytest.fail("D104 preflight failure reached SCF setup"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_b.py",
            "c-diamond",
            "rhf-4c",
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="3"):
        run_case_b.main()

    record = json.loads(
        (tmp_path / "c-diamond__b-rhf-4c.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "error"
    assert record["converged"] is False
    assert record["error_type"] == "EwaldShiftedPairSupportError"
    assert record["ewald_shifted_pair_support"]["qualification"] == (
        "not-qualified"
    )
    assert "energy_per_cell_ha" not in record


def test_fleet_exact_exchange_assembly_fails_closed_when_missing() -> None:
    with pytest.raises(
        RuntimeError,
        match="route-resolved exact-exchange assembly",
    ):
        run_case_b._exact_exchange_assembly_record(SimpleNamespace())


def test_post_hf_runner_records_unqualified_reference_and_correlation_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_case_b, "_scf_options", lambda *_args: object())
    hf_result = SimpleNamespace(
        converged=True,
        n_iter=4,
        backend="aiccm2026dev-b-ri",
        runtime_backend="native-multi-k-gdf-gdf-rhf",
        exact_exchange_assembly=_rhf_exact_exchange_assembly(),
    )
    result = SimpleNamespace(
        energy=-8.1,
        e_hf_per_cell=-8.0,
        e_corr_per_cell=-0.1,
        e_corr_ss_per_cell=-0.02,
        e_corr_os_per_cell=-0.08,
        converged=True,
        n_iter=4,
        n_cyclic_cells=8,
        hf_result=hf_result,
        finite_torus_convention=_d89_chi_convention(),
    )
    monkeypatch.setattr(
        run_case_b.vq,
        "run_aiccm2026dev_b_mp2",
        lambda **_kwargs: result,
    )
    args = SimpleNamespace(
        aux_basis="def2-svp-jk",
        rsgdf_ke_cutoff=320.0,
        mdf_ke_cutoff=40.0,
        progress=False,
    )
    system = SimpleNamespace(lattice=np.eye(3), n_electrons=lambda: 2)

    record = run_case_b._post_hf_record(
        system,
        SimpleNamespace(),
        (2, 2, 2),
        "ri-mp2",
        args,
    )

    support = record["two_electron_support"]
    assert "ewald_shifted_pair_support" not in record
    assert record["exact_exchange_assembly"] == _fleet_exact_exchange_assembly(
        "ri-mp2"
    )
    assert record["direct_lattice_cutoffs"] is None
    assert support["runtime_backend"] == "native-multi-k-gdf-gdf-rhf"
    assert support["qualification"] == "not-qualified"
    assert support["fitted"] == {
        "gdf_method": "rsgdf",
        "rsgdf_base_ke_cutoff_ha": 320.0,
        "rsgdf_tail_ke_cutoff_ha": None,
        "mdf_ke_cutoff_ha": None,
        "cosx_exchange_support": "not-applicable",
        "correlation_support": "not-attested",
    }
    assert (
        b_routes.two_electron_support_schema_failure(support, "ri-mp2") is None
    )
    assert "not-qualified" in (
        b_routes.two_electron_support_reportability_failure(support, "ri-mp2")
        or ""
    )


@pytest.mark.parametrize("route", ("rhf-ri", "rhf-rijcosx"))
def test_low_dimensional_direct_runner_writes_unsupported_before_scf(
    route: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        run_case_b,
        "_ATTESTATION",
        tmp_path / "missing-probe-attestation.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_b.py",
            "h-chain",
            route,
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_case_b.main()

    assert excinfo.value.code == 3
    record = json.loads(
        (tmp_path / f"h-chain__b-{route}.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "unsupported"
    assert record["provenance"]["probe_passed"] is False
    assert isinstance(record["provenance"]["source_clean"], bool)
    assert "wire/slab Coulomb kernel" in record["reason"]


def test_normal_b_run_rejects_missing_attestation_before_scf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AICCM_SKIP_PROBE", raising=False)
    monkeypatch.setattr(
        run_case_b,
        "_ATTESTATION",
        tmp_path / "missing-probe-attestation.json",
    )
    monkeypatch.setattr(
        testset,
        "build",
        lambda _name: pytest.fail("unattested launch reached system construction"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_b.py",
            "lih-rocksalt",
            "rhf-ri",
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_case_b.main()

    assert excinfo.value.code == 3
    assert not (tmp_path / "lih-rocksalt__b-rhf-ri.json").exists()


def test_normal_a_run_rejects_missing_attestation_before_scf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("AICCM_SKIP_PROBE", raising=False)
    monkeypatch.setattr(
        run_case,
        "_ATTESTATION",
        tmp_path / "missing-probe-attestation.json",
    )
    monkeypatch.setattr(
        run_case,
        "_ccm",
        lambda *_args, **_kwargs: pytest.fail(
            "unattested launch reached cyclic-cluster construction"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case.py",
            "lih-rocksalt",
            "aiccm-hf",
            "--out",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_case.main()

    assert excinfo.value.code == 3
    assert not (tmp_path / "lih-rocksalt__aiccm-hf.json").exists()


@pytest.mark.parametrize("nested", (False, True), ids=("raw", "curated"))
def test_audit_rejects_legacy_low_dimensional_absolute_energy(
    nested: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = (
        tmp_path
        / "aiccm2026dev-b"
        / "h-chain"
        / "sto-3g"
        / "rhf-ri"
        / "result.json"
        if nested
        else tmp_path / "h-chain__b-rhf-ri.json"
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        """
{
  "selector": "aiccm2026dev-b",
  "system": "h-chain",
  "route": "rhf-ri",
  "status": "ok",
  "converged": true,
  "dim": 1,
  "energy_per_atom_ha": -1.0,
  "finite_torus_convention": {
    "coulomb_kernel": "3d-periodic-g0",
    "exchange_q0": "bvk-ewald"
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert (
        "lower-dimensional chi-CCM-B absolute energies"
        in capsys.readouterr().out
    )


def test_audit_discovers_pre_normalization_curated_record_from_stream_path(
    tmp_path: Path,
) -> None:
    record = (
        tmp_path
        / "aiccm2026dev-b"
        / "uniform-h-chain"
        / "sto-3g"
        / "rhf-ri"
        / "result.json"
    )
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "system": "uniform-h-chain",
                "route": "rhf-ri",
                "status": "unsupported",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = audit_b._json_records(tmp_path)

    assert records == [(record, json.loads(record.read_text(encoding="utf-8")))]


@pytest.mark.parametrize(
    ("payload", "expected"),
    (("{", "malformed chi-CCM-B JSON"), ("[]", "is not an object")),
)
def test_audit_rejects_corrupt_strong_b_path_beside_valid_record(
    payload: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_b_audit_record(tmp_path)
    (tmp_path / "corrupt__b-rhf-ri.json").write_text(
        payload,
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "# audit failures:" in output
    assert expected in output


def test_audit_rejects_retracted_curated_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_retracted_b_result(tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert "retracted result is failure evidence only" in capsys.readouterr().out


def _write_b_audit_record(root: Path, **updates: object) -> Path:
    provenance = _producer_provenance("aiccm2026dev-b")
    record = {
        "selector": "aiccm2026dev-b",
        "system": "lih-rocksalt",
        "route": "rhf-4c",
        "method": "RHF",
        "backend": "four_center",
        "functional": None,
        "status": "ok",
        "converged": True,
        "dim": 3,
        "mesh": [2, 2, 2],
        "energy_per_atom_ha": -4.0,
        "primitive_lattice_bohr": [
            [0.0, 3.86, 3.86],
            [3.86, 0.0, 3.86],
            [3.86, 3.86, 0.0],
        ],
        "finite_torus_convention": {
            "ccm_approach": "chi-ccm",
            "ccm_construction": "finite-translation-group-character",
            "evaluation_representation": "gamma-centred-character-mesh",
            "coulomb_kernel": "3d-periodic-g0",
            "exchange_q0": "bvk-ewald",
            "exchange_q0_applicability": "active",
            "lattice_vector_convention": "columns",
            "periodic_dimension": 3,
            "character_mesh_shape": [2, 2, 2],
            "bvk_madelung_supercell_repetitions": [2, 2, 2],
            "bvk_madelung_supercell_lattice_bohr": [
                [0.0, 7.72, 7.72],
                [7.72, 0.0, 7.72],
                [7.72, 7.72, 0.0],
            ],
        },
        "exact_exchange_assembly": _fleet_exact_exchange_assembly("rhf-4c"),
        "two_electron_support": _qualified_direct_support(),
        "overlap_fold_support": _qualified_overlap_fold_support(),
        "ewald_shifted_pair_support": (
            _qualified_ewald_shifted_pair_support(provenance)
        ),
        "provenance": provenance,
        "direct_lattice_cutoffs": {
            "electronic_cutoff_bohr": 15.0,
            "nuclear_cutoff_bohr": 15.0,
        },
    }
    record.update(updates)
    path = root / "lih-rocksalt__b-rhf-4c.json"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_audit_reports_and_accepts_consistent_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        audit_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    _write_b_audit_record(tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    audit_b.main()

    output = capsys.readouterr().out
    assert "| status | retracted | converged |" in output
    assert "# audit failures:" not in output


@pytest.mark.parametrize("status_case", ("missing", "unknown", "non-text"))
def test_audit_rejects_unknown_or_missing_status(
    status_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if status_case == "missing":
        record.pop("status")
    elif status_case == "non-text":
        record["status"] = []
    else:
        record["status"] = "mystery"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert "unexpected status=" in capsys.readouterr().out


@pytest.mark.parametrize("status", ("ok", "not_converged"))
@pytest.mark.parametrize(
    "assembly_case",
    (
        "missing",
        "malformed",
        "wrong-schema",
        "spoofed",
        "overflow",
        "wrong-screened-applicability",
        "wrong-screened-assembly",
        "wrong-route",
    ),
)
def test_audit_rejects_invalid_exact_exchange_assembly(
    status: str,
    assembly_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(
        tmp_path,
        status=status,
        converged=status == "ok",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    if assembly_case == "missing":
        record.pop("exact_exchange_assembly")
    elif assembly_case == "malformed":
        record["exact_exchange_assembly"] = ["invalid"]
    elif assembly_case == "wrong-schema":
        record["exact_exchange_assembly"]["schema"] = "legacy-unversioned"
    elif assembly_case == "spoofed":
        record["exact_exchange_assembly"]["resolver"] = "spoofed.resolver"
    elif assembly_case == "overflow":
        record["exact_exchange_assembly"]["c_full"] = 10**400
    elif assembly_case == "wrong-screened-applicability":
        record["exact_exchange_assembly"][
            "screened_exchange_applicability"
        ] = "active"
    elif assembly_case == "wrong-screened-assembly":
        record["exact_exchange_assembly"][
            "screened_exchange_assembly"
        ] = "full-range-minus-long-range"
    else:
        record["exact_exchange_assembly"] = _fleet_exact_exchange_assembly(
            "rks-pbe-4c"
        )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    argv = ["audit_b.py", str(tmp_path)]
    if status == "not_converged":
        argv.append("--allow-not-converged")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert "invalid exact-exchange assembly provenance" in capsys.readouterr().out


@pytest.mark.parametrize("status", ("ok", "not_converged"))
@pytest.mark.parametrize(
    "support_case",
    (
        "missing",
        "malformed",
        "spoofed",
        "cutoff-mismatch",
        "overflow",
        "not-qualified",
    ),
)
def test_audit_rejects_unqualified_two_electron_support(
    status: str,
    support_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(
        tmp_path,
        status=status,
        converged=status == "ok",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    if support_case == "missing":
        record.pop("two_electron_support")
    elif support_case == "malformed":
        record["two_electron_support"] = ["invalid"]
    elif support_case == "spoofed":
        record["two_electron_support"]["backend"] = "ri"
    elif support_case == "cutoff-mismatch":
        record["direct_lattice_cutoffs"]["electronic_cutoff_bohr"] = 14.0
    elif support_case == "overflow":
        record["two_electron_support"]["direct"][
            "sr_image_extent_bohr"
        ] = 10**400
    else:
        route = b_routes.ROUTES["rhf-ri"]
        record.update(
            route="rhf-ri",
            method=route.method,
            backend=route.backend,
            functional=route.functional,
            two_electron_support=_unqualified_fitted_support("rhf-ri"),
        )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    argv = ["audit_b.py", str(tmp_path)]
    if status == "not_converged":
        argv.append("--allow-not-converged")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "unreportable two-electron numerical support" in output


@pytest.mark.parametrize("status", ("ok", "not_converged"))
@pytest.mark.parametrize(
    "support_case",
    (
        "missing",
        "malformed",
        "spoofed",
        "cutoff-mismatch",
        "not-qualified",
        "above-stop",
    ),
)
def test_audit_rejects_unqualified_overlap_fold_support(
    status: str,
    support_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(
        tmp_path,
        status=status,
        converged=status == "ok",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    if support_case == "missing":
        record.pop("overlap_fold_support")
    elif support_case == "malformed":
        record["overlap_fold_support"]["unexpected"] = True
    elif support_case == "spoofed":
        record["overlap_fold_support"]["backend"] = "ri"
    elif support_case == "cutoff-mismatch":
        record["overlap_fold_support"]["cutoff_bohr"] = 14.0
    elif support_case == "not-qualified":
        record["overlap_fold_support"].update(
            qualification="not-qualified",
            reason="executed drift exceeds the quantitative target",
            max_k_drift=1.0e-3,
        )
    else:
        record["overlap_fold_support"].update(
            qualification="not-qualified",
            reason="invalid evidence above the runtime stop threshold",
            max_k_drift=1.01e-2,
        )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    argv = ["audit_b.py", str(tmp_path)]
    if status == "not_converged":
        argv.append("--allow-not-converged")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "unreportable overlap-fold numerical support" in output


def test_audit_rejects_pre_repair_d102_absolute_energy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["ewald_shifted_pair_support"] = (
        _legacy_unqualified_ewald_shifted_pair_support_v1()
    )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    csv_path = tmp_path / "audit.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_b.py", str(tmp_path), "--csv", str(csv_path)],
    )

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "revision-bound absolute energy" in output
    assert "exact v2 keys" in output
    assert "| -4.0 |" not in output
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["energy_per_atom_ha"] == ""
    assert "_raw_energy_per_atom_ha" not in row


@pytest.mark.parametrize("status", ("unsupported", "error"))
def test_audit_does_not_require_success_support_from_failure_evidence(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path, status=status, converged=False)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("two_electron_support")
    record.pop("overlap_fold_support")
    record.pop("ewald_shifted_pair_support")
    record.pop("exact_exchange_assembly")
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    argv = ["audit_b.py", str(tmp_path)]
    if status == "error":
        argv.append("--allow-errors")
    monkeypatch.setattr(sys, "argv", argv)

    audit_b.main()

    output = capsys.readouterr().out
    assert "# audit failures:" not in output
    assert "two_electron_support is missing" not in output
    assert "overlap_fold_support is missing" not in output
    assert "ewald_shifted_pair_support is missing" not in output
    assert "exact_exchange_assembly is missing" not in output
    assert "revision-bound absolute energy" not in output


def test_audit_accepts_route_expected_inactive_pbe_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        audit_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    route = b_routes.ROUTES["rks-pbe-4c"]
    record.update(
        route="rks-pbe-4c",
        method=route.method,
        backend=route.backend,
        functional=route.functional,
    )
    record["finite_torus_convention"][
        "exchange_q0_applicability"
    ] = route.exchange_q0_applicability
    record["exact_exchange_assembly"] = _fleet_exact_exchange_assembly(
        "rks-pbe-4c"
    )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    audit_b.main()

    output = capsys.readouterr().out
    assert "| inactive | inactive |" in output
    assert "# audit failures:" not in output


@pytest.mark.parametrize(
    ("route_name", "invalid_applicability"),
    (
        ("rhf-ri", None),
        ("rhf-ri", "inactive"),
        ("rks-pbe-ri", "active"),
    ),
)
def test_audit_rejects_route_inconsistent_exchange_q0_applicability(
    route_name: str,
    invalid_applicability: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    route = b_routes.ROUTES[route_name]
    record.update(
        route=route_name,
        method=route.method,
        backend=route.backend,
        functional=route.functional,
    )
    convention = record["finite_torus_convention"]
    if invalid_applicability is None:
        convention.pop("exchange_q0_applicability")
    else:
        convention["exchange_q0_applicability"] = invalid_applicability
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "applicability-" in output
    assert f"expected-{route.exchange_q0_applicability}" in output


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "remove"),
    (
        ("route", "invented", False),
        ("route", ["rhf-ri"], False),
        ("method", "RKS", False),
        ("backend", "ri", False),
        ("functional", "pbe", False),
        ("functional", None, True),
    ),
)
@pytest.mark.parametrize("status", ("ok", "not_converged"))
def test_audit_rejects_route_identity_contradictions(
    field_name: str,
    invalid_value: object,
    remove: bool,
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(
        tmp_path,
        status=status,
        converged=status == "ok",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    if remove:
        record.pop(field_name)
    else:
        record[field_name] = invalid_value
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    argv = ["audit_b.py", str(tmp_path)]
    if status == "not_converged":
        argv.append("--allow-not-converged")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    output = capsys.readouterr().out
    assert "route identity=" in output
    assert "must match b_routes.ROUTES exactly" in output


def test_audit_malformed_convention_fails_closed_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_b_audit_record(tmp_path, finite_torus_convention=["invalid"])
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert "construction identity" in capsys.readouterr().out


def test_audit_ignores_malformed_optional_diagnostic_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        audit_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    _write_b_audit_record(
        tmp_path,
        properties=["invalid"],
        convergence_diagnostics="invalid",
    )
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    audit_b.main()

    assert "# audit failures:" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("ccm_approach", None),
        ("ccm_approach", "representation-label"),
        ("ccm_construction", None),
        ("ccm_construction", "union-and-weight"),
        ("evaluation_representation", None),
        ("evaluation_representation", "real-gamma-supercell"),
    ),
)
def test_audit_rejects_missing_or_wrong_construction_identity(
    field_name: str,
    invalid_value: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    convention = record["finite_torus_convention"]
    if invalid_value is None:
        convention.pop(field_name)
    else:
        convention[field_name] = invalid_value
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert "construction identity" in capsys.readouterr().out


def test_audit_rejects_pre_d88_lattice_convention_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["finite_torus_convention"].pop("lattice_vector_convention")
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert "Hamiltonian convention" in capsys.readouterr().out


def test_audit_rejects_row_scaled_bvk_lattice_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["finite_torus_convention"][
        "bvk_madelung_supercell_lattice_bohr"
    ][0][1] = 11.58
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert "bvk-match-False" in capsys.readouterr().out


@pytest.mark.parametrize("converged", (False, None, 1, "true"))
def test_audit_requires_ok_converged_to_be_exactly_true(
    converged: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_b_audit_record(tmp_path, converged=converged)
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert (
        "status=ok requires converged=true exactly" in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "energy",
    (None, True, "-4.0", float("nan"), float("inf")),
    ids=("missing", "boolean", "string", "nan", "infinity"),
)
def test_audit_requires_ok_energy_to_be_finite_numeric(
    energy: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_b_audit_record(tmp_path, energy_per_atom_ha=energy)
    monkeypatch.setattr(sys, "argv", ["audit_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert (
        "status=ok requires finite numeric energy_per_atom_ha"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("status", "allow_flag"),
    (
        ("not_converged", "--allow-not-converged"),
        ("unsupported", None),
        ("error", "--allow-errors"),
    ),
)
def test_audit_rejects_non_success_status_claiming_convergence(
    status: str,
    allow_flag: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_b_audit_record(tmp_path, status=status, converged=True)
    argv = ["audit_b.py", str(tmp_path)]
    if allow_flag is not None:
        argv.append(allow_flag)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        audit_b.main()

    assert excinfo.value.code == 2
    assert (
        f"status={status} must not claim converged=true"
        in capsys.readouterr().out
    )


def test_legacy_compare_direct_head_to_head_fails_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        legacy_compare.head_to_head({}, {})

    message = str(excinfo.value)
    assert "D89 disables compare.py head-to-head deltas" in message
    assert "compare_b.py --real-gamma-control-results" in message
    assert capsys.readouterr().out == ""


def test_legacy_compare_qualifies_four_center_control_gap(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = legacy_compare.single_table(
        {
            "h2": {
                "aiccm-hf": {
                    "energy_per_atom": -1.0,
                    "klass": "molecular",
                },
                "gdf": {"energy_per_atom": -0.999},
            }
        },
        {},
    )

    assert rows[0]["d_gamma_ccm_minus_gdf_control"] == pytest.approx(-1.0)
    assert "d_4c_gdf" not in rows[0]
    legacy_compare.print_single(rows)
    output = capsys.readouterr().out
    assert "ΔΓCCM-GDFctrl" in output
    assert "construction/control route gap with causality unassigned" in output
    assert "Madelung gap" not in output


def test_legacy_compare_vs_cli_fails_closed_before_writing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "forbidden-comparison.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare.py",
            str(tmp_path / "a"),
            "--vs",
            str(tmp_path / "b"),
            "--csv",
            str(csv_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        legacy_compare.main()

    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert captured.out == ""
    assert "D89 disables compare.py head-to-head deltas" in captured.err
    assert "compare_b.py --real-gamma-control-results" in captured.err
    assert not csv_path.exists()


def test_compare_refuses_retracted_curated_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_retracted_b_result(tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    assert "refusing to compare retracted" in capsys.readouterr().err


@pytest.mark.parametrize("marker", (1, "true"), ids=("integer", "string"))
def test_compare_conservatively_refuses_truthy_retraction_marker(
    marker: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_retracted_b_result(tmp_path, marker=marker)
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    assert "refusing to compare retracted" in capsys.readouterr().err


def test_compare_refuses_duplicate_system_route_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_retracted_b_result(tmp_path)
    second = _write_retracted_b_result(
        tmp_path,
        basis="pob-tzvp-rev2",
        marker=False,
    )
    second_record = json.loads(second.read_text(encoding="utf-8"))
    second_record["mesh"] = [2, 2, 2]
    second.write_text(
        json.dumps(second_record, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    error = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "duplicate result records" in error
    assert "sto-3g" in error
    assert "pob-tzvp-rev2" in error


@pytest.mark.parametrize(
    ("payload", "expected"),
    (("{", "malformed JSON result candidate"), ("[]", "is not a JSON object")),
)
def test_compare_load_rejects_corrupt_result_candidates(
    payload: str,
    expected: str,
    tmp_path: Path,
) -> None:
    (tmp_path / "corrupt__b-rhf-ri.json").write_text(
        payload,
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        compare_b.load(tmp_path)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize("field", ("system", "route"))
def test_compare_load_rejects_non_text_result_identity(
    field: str,
    tmp_path: Path,
) -> None:
    record: dict[str, object] = {
        "system": "lih-rocksalt",
        "route": "rhf-4c",
        "status": "unsupported",
    }
    record[field] = []
    (tmp_path / "invalid__b-rhf-4c.json").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        compare_b.load(tmp_path)

    assert f"missing or invalid {field}" in str(excinfo.value)


def test_compare_main_reports_malformed_candidate_as_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "corrupt__b-rhf-ri.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "malformed JSON result candidate" in captured.err


def test_compare_loads_distinct_functionals_but_excludes_same_driver_routes(
    tmp_path: Path,
) -> None:
    a_route = "aiccm-ks-ri-cosx"
    pbe_b, pbe_a = _matching_real_gamma_control_records(
        "rks-pbe-rijcosx",
        control_route_name=a_route,
    )
    pbe0_b, pbe0_a = _matching_real_gamma_control_records(
        "rks-pbe0-rijcosx",
        control_route_name=a_route,
    )
    (tmp_path / "pbe__aiccm-ks-ri-cosx.json").write_text(
        json.dumps(pbe_a) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "pbe0__aiccm-ks-ri-cosx.json").write_text(
        json.dumps(pbe0_a) + "\n",
        encoding="utf-8",
    )

    records = compare_b.load(tmp_path)

    assert set(records) == {
        ("lih-rocksalt", "aiccm-ks-ri-cosx", "pbe"),
        ("lih-rocksalt", "aiccm-ks-ri-cosx", "pbe0"),
    }
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rks-pbe-rijcosx",
        pbe_b,
        records,
    ) == (None, None)
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rks-pbe0-rijcosx",
        pbe0_b,
        records,
    ) == (None, None)


def test_compare_main_rejects_chi_route_metadata_contradiction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    b_record, _ = _matching_real_gamma_control_records(
        "rks-pbe0-ri",
        control_route_name="aiccm-ks-ri",
    )
    b_record["route"] = "rks-pbe-ri"
    (tmp_path / "lih__b-rks-pbe-ri.json").write_text(
        json.dumps(b_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    assert "contradicts the route selector" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field",
    ("ccm_approach", "ccm_construction", "evaluation_representation"),
)
def test_compare_main_rejects_missing_chi_construction_identity(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    b_record, _ = _matching_real_gamma_control_records("rhf-ri")
    b_record["finite_torus_convention"].pop(field)
    (tmp_path / "lih__b-rhf-ri.json").write_text(
        json.dumps(b_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    assert "exact construction identity" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("route_name", "invalid_applicability", "control_route_name"),
    (
        ("rhf-ri", None, None),
        ("rhf-ri", "inactive", None),
        ("rks-pbe-ri", "active", "aiccm-ks-ri"),
    ),
)
def test_compare_main_rejects_wrong_exchange_q0_applicability(
    route_name: str,
    invalid_applicability: str | None,
    control_route_name: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    b_record, _ = _matching_real_gamma_control_records(
        route_name,
        control_route_name=control_route_name,
    )
    convention = b_record["finite_torus_convention"]
    if invalid_applicability is None:
        convention.pop("exchange_q0_applicability")
    else:
        convention["exchange_q0_applicability"] = invalid_applicability
    (tmp_path / f"lih__b-{route_name}.json").write_text(
        json.dumps(b_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    assert "exchange-q=0 applicability" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("b_route_name", "control_route_name"),
    (
        ("rhf-ri", "aiccm-hf-direct"),
    ),
)
def test_compare_matches_character_and_real_gamma_control_representations(
    b_route_name: str,
    control_route_name: str,
) -> None:
    b_record, control_record = _matching_real_gamma_control_records(b_route_name)

    matched_route, matched_record = compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        b_route_name,
        b_record,
        {("lih-rocksalt", control_route_name): control_record},
    )

    assert (
        compare_b.REAL_GAMMA_CONTROL_ROUTE_FOR_B[b_route_name]
        == control_route_name
    )
    assert matched_route == control_route_name
    assert matched_record is control_record


def test_compare_main_marks_missing_real_gamma_control_not_defined(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    b_results = tmp_path / "b"
    b_results.mkdir()
    _write_b_audit_record(b_results)
    csv_path = tmp_path / "comparison.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_b.py", str(b_results), "--csv", str(csv_path)],
    )

    compare_b.main()

    output = capsys.readouterr().out
    assert "Γ-CCM/χ-CCM approach comparison status" in output
    assert "real-Γ control status" in output
    assert "χ character minus real-Γ control" in output
    assert "Γ-CCM route" not in output
    assert "χ-Γ" not in output
    table_row = next(
        line for line in output.splitlines() if line.startswith("| lih-rocksalt |")
    )
    cells = [cell.strip() for cell in table_row.strip("|").split("|")]
    assert cells[2] == "ok"
    assert cells[3] == "not-defined"
    assert cells[5] == "not-defined"
    assert cells[6] == "not-defined"
    assert cells[7] == "not-defined"
    assert cells[9] == "not-defined"

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["status"] == "ok"
    assert (
        csv_row["gamma_ccm_chi_ccm_approach_comparison_status"]
        == "not-defined"
    )
    assert csv_row["real_gamma_control_status"] == "not-defined"
    assert csv_row["real_gamma_control_route"] == "not-defined"
    assert csv_row["real_gamma_control_ha_per_atom"] == "not-defined"
    assert (
        csv_row["character_minus_real_gamma_control_mha_per_atom"]
        == "not-defined"
    )


@pytest.mark.parametrize("status_case", ("missing", "unknown", "non-text"))
def test_compare_main_rejects_unknown_or_missing_b_status(
    status_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if status_case == "missing":
        record.pop("status")
    elif status_case == "non-text":
        record["status"] = []
    else:
        record["status"] = "mystery"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown or missing status" in captured.err


@pytest.mark.parametrize("status", ("unsupported", "error"))
def test_compare_main_rejects_unregistered_failure_evidence_route(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path, status=status, converged=False)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["route"] = "unknown"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    crystal_refs = tmp_path / "crystal.json"
    crystal_refs.write_text(
        json.dumps({"lih-rocksalt": {"rhf-4c": -4.0}}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_b.py",
            str(tmp_path),
            "--crystal-refs",
            str(crystal_refs),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unregistered route" in captured.err


@pytest.mark.parametrize("status", ("ok", "not_converged"))
@pytest.mark.parametrize(
    "assembly_case",
    (
        "missing",
        "malformed",
        "wrong-schema",
        "spoofed",
        "overflow",
        "wrong-screened-applicability",
        "wrong-screened-assembly",
        "wrong-route",
    ),
)
def test_compare_main_rejects_invalid_exact_exchange_assembly(
    status: str,
    assembly_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    path = _write_b_audit_record(
        tmp_path,
        status=status,
        converged=status == "ok",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    if assembly_case == "missing":
        record.pop("exact_exchange_assembly")
    elif assembly_case == "malformed":
        record["exact_exchange_assembly"] = ["invalid"]
    elif assembly_case == "wrong-schema":
        record["exact_exchange_assembly"]["schema"] = "legacy-unversioned"
    elif assembly_case == "spoofed":
        record["exact_exchange_assembly"]["resolver"] = "spoofed.resolver"
    elif assembly_case == "overflow":
        record["exact_exchange_assembly"]["c_full"] = 10**400
    elif assembly_case == "wrong-screened-applicability":
        record["exact_exchange_assembly"][
            "screened_exchange_applicability"
        ] = "active"
    elif assembly_case == "wrong-screened-assembly":
        record["exact_exchange_assembly"][
            "screened_exchange_assembly"
        ] = "full-range-minus-long-range"
    else:
        record["exact_exchange_assembly"] = _fleet_exact_exchange_assembly(
            "rks-pbe-4c"
        )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exact-exchange assembly provenance" in captured.err


@pytest.mark.parametrize(
    "support_case",
    (
        "missing",
        "malformed",
        "spoofed",
        "cutoff-mismatch",
        "overflow",
        "not-qualified",
    ),
)
def test_compare_main_rejects_unqualified_two_electron_support(
    support_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if support_case == "missing":
        record.pop("two_electron_support")
    elif support_case == "malformed":
        record["two_electron_support"]["direct"]["unexpected"] = True
    elif support_case == "spoofed":
        record["two_electron_support"]["runtime_backend"] = "native-multi-k-gdf"
    elif support_case == "cutoff-mismatch":
        record["direct_lattice_cutoffs"]["nuclear_cutoff_bohr"] = 14.0
    elif support_case == "overflow":
        record["two_electron_support"]["direct"][
            "electronic_cutoff_bohr"
        ] = 10**400
    else:
        route = b_routes.ROUTES["rhf-ri"]
        record.update(
            route="rhf-ri",
            method=route.method,
            backend=route.backend,
            functional=route.functional,
            two_electron_support=_unqualified_fitted_support("rhf-ri"),
        )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "without qualified two-electron numerical support" in captured.err


@pytest.mark.parametrize(
    "support_case",
    (
        "missing",
        "malformed",
        "spoofed",
        "cutoff-mismatch",
        "not-qualified",
        "above-stop",
    ),
)
def test_compare_main_rejects_unqualified_overlap_fold_support(
    support_case: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if support_case == "missing":
        record.pop("overlap_fold_support")
    elif support_case == "malformed":
        record["overlap_fold_support"]["unexpected"] = True
    elif support_case == "spoofed":
        record["overlap_fold_support"]["backend"] = "ri"
    elif support_case == "cutoff-mismatch":
        record["overlap_fold_support"]["cutoff_bohr"] = 14.0
    elif support_case == "not-qualified":
        record["overlap_fold_support"].update(
            qualification="not-qualified",
            reason="executed drift exceeds the quantitative target",
            max_k_drift=1.0e-3,
        )
    else:
        record["overlap_fold_support"].update(
            qualification="not-qualified",
            reason="invalid evidence above the runtime stop threshold",
            max_k_drift=1.01e-2,
        )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "without qualified overlap-fold numerical support" in captured.err


def test_compare_main_rejects_pre_repair_d102_absolute_energy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["ewald_shifted_pair_support"] = (
        _legacy_unqualified_ewald_shifted_pair_support_v1()
    )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "revision-bound chi-CCM-B absolute energies" in captured.err
    assert "exact v2 keys" in captured.err


def test_compare_main_rejects_revision_bound_real_gamma_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    b_results = tmp_path / "b"
    control_results = tmp_path / "control"
    b_results.mkdir()
    control_results.mkdir()
    _write_b_audit_record(b_results)

    _, control_record = _matching_real_gamma_control_records("rhf-ri")
    control_record["ewald_shifted_pair_support"] = (
        _legacy_unqualified_ewald_shifted_pair_support_v1()
    )
    (control_results / "lih__control.json").write_text(
        json.dumps(control_record) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_b.py",
            str(b_results),
            "--real-gamma-control-results",
            str(control_results),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "revision-bound real-Gamma representation-control" in captured.err
    assert "exact v2 keys" in captured.err


@pytest.mark.parametrize("status", ("unsupported", "error"))
def test_compare_main_ignores_support_on_failure_evidence(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_b_audit_record(tmp_path, status=status, converged=False)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("two_electron_support")
    record.pop("overlap_fold_support")
    record.pop("ewald_shifted_pair_support")
    record.pop("exact_exchange_assembly")
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    compare_b.main()

    assert f"| {status} |" in capsys.readouterr().out


@pytest.mark.parametrize(
    "control_option",
    ("--real-gamma-control-results", "--a-results"),
    ids=("explicit-control", "legacy-control-alias"),
)
def test_compare_main_rejects_unqualified_real_gamma_control_candidate(
    control_option: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    b_record, control_record = _matching_real_gamma_control_records("rhf-ri")
    b_results = tmp_path / "b"
    control_results = tmp_path / "control"
    b_results.mkdir()
    control_results.mkdir()
    (b_results / "lih__b-rhf-ri.json").write_text(
        json.dumps(b_record) + "\n",
        encoding="utf-8",
    )
    (control_results / "lih__aiccm-hf-direct.json").write_text(
        json.dumps(control_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_b.py",
            str(b_results),
            control_option,
            str(control_results),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "without qualified two-electron numerical support" in captured.err


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("method", "RKS"),
        ("functional", "pbe"),
        ("basis", "pob-tzvp-rev2"),
        ("nrep", [1, 1, 1]),
        ("exchange_q0", "bare-four-center"),
        ("retracted", True),
    ),
)
def test_compare_rejects_mismatched_real_gamma_control_metadata(
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record[field] = value

    matched_route, matched_record = compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    )

    assert matched_route is None
    assert matched_record is None


def test_compare_main_rejects_unqualified_row_before_rendering_control_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        compare_b,
        "d102_absolute_energy_revision_failure",
        lambda *_args: None,
    )
    b_record, control_record = _matching_real_gamma_control_records("rhf-ri")
    control_record["basis"] = "pob-tzvp-rev2"
    b_results = tmp_path / "b"
    control_results = tmp_path / "control"
    b_results.mkdir()
    control_results.mkdir()
    (b_results / "lih__b-rhf-ri.json").write_text(
        json.dumps(b_record) + "\n",
        encoding="utf-8",
    )
    (control_results / "lih__aiccm-hf-direct.json").write_text(
        json.dumps(control_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_b.py",
            str(b_results),
            "--real-gamma-control-results",
            str(control_results),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "without qualified two-electron numerical support" in captured.err


def test_compare_maps_only_the_independent_rhf_scf_representation_pair() -> None:
    assert compare_b.GAMMA_CHI_APPROACH_ROUTE_FOR_B == {}
    assert compare_b.REAL_GAMMA_CONTROL_ROUTE_FOR_B == {"rhf-ri": "aiccm-hf-direct"}
    assert "aiccm-ri" not in compare_b.REAL_GAMMA_CONTROL_ROUTE_FOR_B.values()
    assert "aiccm-ri-cosx" not in compare_b.REAL_GAMMA_CONTROL_ROUTE_FOR_B.values()
    assert "aiccm-ks-ri" not in compare_b.REAL_GAMMA_CONTROL_ROUTE_FOR_B.values()
    assert "aiccm-ks-ri-cosx" not in compare_b.REAL_GAMMA_CONTROL_ROUTE_FOR_B.values()

    b_record, same_driver_a = _matching_real_gamma_control_records(
        "rhf-ri",
        control_route_name="aiccm-ri",
    )
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-ri"): same_driver_a},
    ) == (None, None)


@pytest.mark.parametrize(
    ("side", "spoofed_route"),
    (("b", "rhf-4c"), ("control", "aiccm-ri")),
)
def test_compare_rejects_embedded_route_spoof(
    side: str,
    spoofed_route: str,
) -> None:
    b_record, control_record = _matching_real_gamma_control_records("rhf-ri")
    record = b_record if side == "b" else control_record
    record["route"] = spoofed_route

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): control_record},
    ) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend", "four_center"),
        ("aux_basis", "other-jkfit"),
        ("gdf_method", "mdf"),
        ("rsgdf_ke_cutoff", 180.0),
    ),
)
def test_compare_rejects_result_fields_that_conflict_with_contract(
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record[field] = value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize("side", ("b", "a"))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("aux_basis", "other-jkfit"),
        ("gdf_method", "mdf"),
        ("rsgdf_ke_cutoff", 180.0),
        ("ao_linear_dependence_threshold", 1.0e-6),
        ("auxiliary_metric_linear_dependence_threshold", 1.0e-8),
        ("auxiliary_phase_convention", "other-phase/v1"),
        ("two_electron_operator", "rijcosx"),
    ),
)
def test_compare_rejects_resolved_payload_alias_conflict_on_either_side(
    side: str,
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    record = b_record if side == "b" else a_record
    record[field] = value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize("side", ("b", "a"))
@pytest.mark.parametrize(
    "field",
    (
        "aux_basis",
        "gdf_method",
        "rsgdf_ke_cutoff",
        "ao_linear_dependence_threshold",
        "auxiliary_metric_linear_dependence_threshold",
        "auxiliary_phase_convention",
        "two_electron_operator",
    ),
)
def test_compare_requires_resolved_payload_aliases_on_both_sides(
    side: str,
    field: str,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    record = b_record if side == "b" else a_record
    record.pop(field)

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    "field",
    ("method", "backend", "aux_basis", "dim", "primitive_lattice_bohr"),
)
def test_compare_requires_emitted_chi_payload_identity(field: str) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record.pop(field)

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_requires_emitted_chi_functional_identity() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record.pop("functional")

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    "field",
    (
        "ccm_approach",
        "ccm_construction",
        "evaluation_representation",
        "boundary_model",
        "coulomb_kernel",
        "lattice_vector_convention",
        "periodic_dimension",
        "character_mesh_shape",
        "bvk_madelung_supercell_repetitions",
        "bvk_madelung_supercell_lattice_bohr",
    ),
)
def test_compare_requires_emitted_chi_convention_identity(field: str) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["finite_torus_convention"].pop(field)

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    ("field", "near_miss"),
    (
        ("ccm_approach", "CHI-CCM"),
        ("ccm_approach", "chi-ccm "),
        ("ccm_construction", "Finite-Translation-Group-Character"),
        ("ccm_construction", " finite-translation-group-character"),
        ("evaluation_representation", "Gamma-centred-character-mesh"),
        ("evaluation_representation", "gamma-centred-character-mesh\n"),
    ),
)
def test_compare_rejects_near_miss_construction_identity(
    field: str,
    near_miss: str,
) -> None:
    b_record, control_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["finite_torus_convention"][field] = near_miss

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): control_record},
    ) == (None, None)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("ccm_approach", "gamma-ccm"),
        ("ccm_construction", "union-and-weight"),
        ("evaluation_representation", "real-supercell-four-center"),
        ("route_role", "ccm-approach"),
    ),
)
def test_compare_rejects_wrong_real_gamma_control_role(
    field: str,
    wrong_value: str,
) -> None:
    b_record, control_record = _matching_real_gamma_control_records("rhf-ri")
    control_record[field] = wrong_value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): control_record},
    ) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ccm_approach", "gamma-ccm"),
        ("ccm_construction", "union-and-weight"),
        ("evaluation_representation", "real-gamma-supercell"),
        ("lattice_vector_convention", "rows"),
        ("periodic_dimension", 2),
        ("character_mesh_shape", [3, 2, 2]),
        ("bvk_madelung_supercell_repetitions", [3, 2, 2]),
        (
            "bvk_madelung_supercell_lattice_bohr",
            [[0.0, 7.72, 7.72], [11.58, 0.0, 7.72], [11.58, 7.72, 0.0]],
        ),
    ),
)
def test_compare_rejects_inconsistent_chi_bvk_lattice(
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["finite_torus_convention"][field] = value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_binds_skew_chi_bvk_lattice_by_columns() -> None:
    lattice = np.asarray(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    mesh = (2, 3, 1)
    record = {
        "mesh": list(mesh),
        "primitive_lattice_bohr": lattice.tolist(),
        "comparison_input": {"system": {"lattice_bohr": lattice.tolist()}},
        "finite_torus_convention": {
            "lattice_vector_convention": "columns",
            "periodic_dimension": 3,
            "character_mesh_shape": list(mesh),
            "bvk_madelung_supercell_repetitions": list(mesh),
            "bvk_madelung_supercell_lattice_bohr": (
                lattice @ np.diag(mesh)
            ).tolist(),
        },
    }

    assert compare_b._b_finite_torus_lattice_matches_input(record)

    record["finite_torus_convention"][
        "bvk_madelung_supercell_lattice_bohr"
    ] = (np.diag(mesh) @ lattice).tolist()
    assert not compare_b._b_finite_torus_lattice_matches_input(record)


def test_compare_rejects_conflicting_primitive_lattice_alias() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["primitive_lattice_bohr"][0][1] = 4.0

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("exchange_q0", "bare-four-center"),
        ("boundary_model", "slab"),
        ("coulomb_kernel", "other-kernel"),
    ),
)
def test_compare_rejects_conflicting_convention_aliases(
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record[field] = value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize("side", ("b", "a"))
@pytest.mark.parametrize(
    "updates",
    (
        {"status": "not_converged", "converged": False},
        {"status": "ok", "converged": False},
        {"status": None, "converged": False},
    ),
)
def test_compare_rejects_nonconverged_real_gamma_control_records(
    side: str,
    updates: dict[str, object],
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    record = b_record if side == "b" else a_record
    record.update(updates)

    assert compare_b._energy_per_atom(
        record,
        allow_missing_status=side == "a",
    ) is None
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_accepts_current_real_gamma_control_without_status() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")

    assert "status" not in a_record
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == ("aiccm-hf-direct", a_record)
    assert compare_b._energy_per_atom(
        a_record,
        allow_missing_status=True,
    ) == -4.0


@pytest.mark.parametrize("value", (True, "-4.0", float("nan"), float("inf")))
def test_compare_rejects_non_numeric_or_nonfinite_energies(value: object) -> None:
    b_record, _ = _matching_real_gamma_control_records("rhf-ri")
    b_record["energy_per_atom_ha"] = value

    assert compare_b._energy_per_atom(b_record) is None


@pytest.mark.parametrize(
    "missing",
    ("provenance", "comparison_input", "comparison_contract"),
)
def test_compare_requires_paired_provenance_and_contract(missing: str) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record.pop(missing)

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_comparison_input_hash_is_canonical_over_mapping_order() -> None:
    b_record, _ = _matching_real_gamma_control_records("rhf-ri")
    comparison_input = b_record["comparison_input"]
    reversed_input = dict(reversed(list(comparison_input.items())))

    assert compare_b._comparison_input_sha256(reversed_input) == (
        b_record["comparison_contract"]["input_sha256"]
    )


def test_compare_rejects_comparison_input_tampering_without_digest_update() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record["comparison_input"]["finite_torus_mesh"] = [3, 2, 2]

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_unknown_comparison_input_schema_even_with_new_digest() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    for record in (b_record, a_record):
        record["comparison_input"]["schema"] = "unversioned-input"
        fingerprint = compare_b._comparison_input_sha256(
            record["comparison_input"]
        )
        assert fingerprint is not None
        record["comparison_contract"]["input_sha256"] = fingerprint

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_rehashed_incomplete_comparison_input() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    for record in (b_record, a_record):
        record["comparison_input"] = {
            "schema": compare_b.REAL_GAMMA_CONTROL_INPUT_VERSION,
        }
        fingerprint = compare_b._comparison_input_sha256(
            record["comparison_input"]
        )
        assert fingerprint is not None
        record["comparison_contract"]["input_sha256"] = fingerprint

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("system_name", "other-system"),
        ("electronic_method", "RKS"),
        ("orbital_basis", "pob-tzvp-rev2"),
        ("finite_torus_mesh", [3, 2, 2]),
        ("two_electron_operator", "rijcosx"),
        ("gdf_method", "mdf"),
        ("rsgdf_ke_cutoff", 999.0),
    ),
)
def test_compare_rejects_rehashed_input_that_conflicts_with_contract_or_payload(
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    for record in (b_record, a_record):
        record["comparison_input"][field] = value
        fingerprint = compare_b._comparison_input_sha256(
            record["comparison_input"]
        )
        assert fingerprint is not None
        record["comparison_contract"]["input_sha256"] = fingerprint

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize("mesh", ([0, 2, 2], [-1, 2, 2]))
def test_compare_rejects_nonpositive_mesh_even_when_every_alias_matches(
    mesh: list[int],
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["mesh"] = list(mesh)
    a_record["nrep"] = list(mesh)
    for record in (b_record, a_record):
        record["finite_torus_convention"]["character_mesh_shape"] = list(mesh)
        record["comparison_input"]["finite_torus_mesh"] = list(mesh)
        fingerprint = compare_b._comparison_input_sha256(
            record["comparison_input"]
        )
        assert fingerprint is not None
        record["comparison_contract"]["input_sha256"] = fingerprint

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_singular_lattice_even_with_matching_input_hashes() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    for record in (b_record, a_record):
        record["comparison_input"]["system"]["lattice_bohr"] = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        fingerprint = compare_b._comparison_input_sha256(
            record["comparison_input"]
        )
        assert fingerprint is not None
        record["comparison_contract"]["input_sha256"] = fingerprint

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_nonzero_smearing_even_when_every_alias_matches() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    for record in (b_record, a_record):
        record["scf_options"]["smearing_temperature"] = 0.5
        record["comparison_input"]["smearing_temperature"] = 0.5
        record["comparison_contract"]["smearing_temperature"] = 0.5
        fingerprint = compare_b._comparison_input_sha256(
            record["comparison_input"]
        )
        assert fingerprint is not None
        record["comparison_contract"]["input_sha256"] = fingerprint

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize("side", ("b", "a"))
@pytest.mark.parametrize(
    "section",
    ("scf_options", "convergence_diagnostics"),
)
def test_compare_requires_requested_and_reported_smearing_on_both_sides(
    side: str,
    section: str,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    record = b_record if side == "b" else a_record
    record[section].pop("smearing_temperature")

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_reported_smearing_that_conflicts_with_request() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["convergence_diagnostics"] = {"smearing_temperature": 0.5}

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    "missing",
    (
        "ao_linear_dependence_threshold",
        "auxiliary_metric_linear_dependence_threshold",
    ),
)
def test_compare_contract_v2_requires_both_distinct_thresholds(
    missing: str,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record["comparison_contract"].pop(missing)
    a_record["comparison_contract"]["linear_dependence_threshold"] = 1.0e-9

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_strips_raw_json_run_meta_spoof(tmp_path: Path) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["_run_meta"] = b_record.pop("provenance")
    a_record["_run_meta"] = a_record.pop("provenance")
    b_root = tmp_path / "b"
    a_root = tmp_path / "a"
    b_root.mkdir()
    a_root.mkdir()
    (b_root / "lih__b-rhf-ri.json").write_text(
        json.dumps(b_record) + "\n",
        encoding="utf-8",
    )
    (a_root / "lih__aiccm-hf-direct.json").write_text(
        json.dumps(a_record) + "\n",
        encoding="utf-8",
    )

    b_records = compare_b.load(b_root)
    a_records = compare_b.load(a_root)

    assert "_run_meta" not in b_records[("lih-rocksalt", "rhf-ri", None)]
    assert "_run_meta" not in a_records[
        ("lih-rocksalt", "aiccm-hf-direct", None)
    ]
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_records[("lih-rocksalt", "rhf-ri", None)],
        a_records,
    ) == (None, None)


def test_compare_rejects_conflicting_nested_and_top_level_provenance() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record["vibeqc_commit"] = "d" * 40

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    "field",
    (
        "host",
        "vibeqc_version",
        "vibeqc_commit",
        "source_clean",
        "core_build_id",
        "probe_version",
        "probe_script_id",
        "probe_passed",
        "probe_attestation_id",
        "producer_payload_id",
    ),
)
def test_compare_requires_complete_nested_provenance_even_with_alias(
    field: str,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    value = a_record["provenance"].pop(field)
    a_record[field] = value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_same_unknown_probe_version_on_both_records() -> None:
    assert compare_b.EXPECTED_PROBE_VERSION == probe_host.PROBE_VERSION
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["provenance"]["probe_version"] = "legacy-attestor/v0"
    a_record["provenance"]["probe_version"] = "legacy-attestor/v0"

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_same_malformed_probe_attestation_on_both_records() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    for record in (b_record, a_record):
        record["provenance"]["probe_attestation_id"] = "not-recorded"

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_same_forged_probe_attestation_digest() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    for record in (b_record, a_record):
        record["provenance"]["probe_attestation_id"] = "sha256:" + "d" * 64

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_requires_the_embedded_probe_attestation() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record["provenance"].pop("probe_attestation")

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-preflight",
        "stale-preflight-digest",
        "invalid-loaded-core-check",
        "unstable-result-identity",
        "unknown-native-library",
        "extra-composite-field",
        "wrong-payload-stream",
    ),
)
def test_compare_independently_validates_composite_attestation(
    mutation: str,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    provenance = a_record["provenance"]
    attestation = provenance["probe_attestation"]
    if mutation == "missing-preflight":
        attestation.pop("preflight_attestation")
    elif mutation == "stale-preflight-digest":
        attestation["preflight_attestation"]["host"] = "forged-host"
    elif mutation == "invalid-loaded-core-check":
        attestation["loaded_core_check"]["max_scaled_error"] = 1.01
    elif mutation == "unstable-result-identity":
        attestation["result_identity_stable"] = False
        attestation["probe_passed"] = False
        provenance["probe_passed"] = False
    elif mutation == "unknown-native-library":
        attestation["native_library_versions"]["libint"] = "unknown"
    elif mutation == "extra-composite-field":
        attestation["unvalidated"] = True
    elif mutation == "wrong-payload-stream":
        attestation["producer_payload"]["stream"] = "aiccm2026dev-b"
        attestation["producer_payload_id"] = probe_host.probe_attestation_id(
            attestation["producer_payload"]
        )
        provenance["producer_payload_id"] = attestation["producer_payload_id"]
    else:  # pragma: no cover - exhaustive parameter list above
        raise AssertionError(mutation)
    provenance["probe_attestation_id"] = probe_host.probe_attestation_id(
        attestation
    )

    assert compare_b._producer_identity(
        a_record,
        expected_stream="aiccm2026dev-a",
    ) is None
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("host", "other-host"),
        ("vibeqc_version", "0.15.35"),
        ("vibeqc_commit", "d" * 40),
        ("source_clean", False),
        ("core_build_id", "sha256:" + "d" * 64),
        ("probe_version", "aiccm-producer-attestation/v999"),
        ("probe_script_id", "sha256:" + "e" * 64),
        ("probe_passed", False),
        ("probe_attestation_id", "sha256:" + "d" * 64),
        ("producer_payload_id", "sha256:" + "e" * 64),
    ),
)
def test_compare_rejects_real_gamma_control_on_producer_mismatch(
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record["provenance"][field] = value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_rejects_two_individually_valid_different_producers() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_attestation = json.loads(
        json.dumps(a_record["provenance"]["probe_attestation"])
    )
    for section in ("preflight_attestation", "current_core_attestation"):
        a_attestation[section]["host"] = "other-verified-host"
    a_attestation["host"] = "other-verified-host"
    a_attestation["preflight_attestation_id"] = probe_host.probe_attestation_id(
        a_attestation["preflight_attestation"]
    )
    a_attestation["current_core_attestation_id"] = (
        probe_host.probe_attestation_id(a_attestation["current_core_attestation"])
    )
    a_record["provenance"]["host"] = "other-verified-host"
    a_record["provenance"]["probe_attestation"] = a_attestation
    a_record["provenance"]["probe_attestation_id"] = (
        probe_host.probe_attestation_id(a_attestation)
    )

    b_identity = compare_b._producer_identity(b_record)
    a_identity = compare_b._producer_identity(a_record)
    assert b_identity is not None
    assert a_identity is not None
    assert b_identity != a_identity
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_accepts_different_valid_preflight_core_histories() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record["provenance"] = _producer_provenance(
        "aiccm2026dev-a",
        preflight_core_id=_OLD_CORE_ID,
    )

    b_identity = compare_b._producer_identity(
        b_record,
        expected_stream="aiccm2026dev-b",
    )
    a_identity = compare_b._producer_identity(
        a_record,
        expected_stream="aiccm2026dev-a",
    )
    assert b_identity is not None
    assert a_identity == b_identity
    matched_route, matched_record = compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    )
    assert matched_route == "aiccm-hf-direct"
    assert matched_record is a_record


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("input_sha256", "sha256:" + "d" * 64),
        ("boundary_model", "slab"),
        ("coulomb_kernel", "other-kernel"),
        ("two_electron_operator", "rijcosx"),
        ("aux_basis", "other-jkfit"),
        ("gdf_method", "mdf"),
        ("rsgdf_ke_cutoff", 180.0),
        ("ao_linear_dependence_threshold", 1.0e-6),
        ("auxiliary_metric_linear_dependence_threshold", 1.0e-8),
        ("auxiliary_phase_convention", "other-phase/v1"),
    ),
)
def test_compare_rejects_real_gamma_control_on_contract_mismatch(
    field: str,
    value: object,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record["comparison_contract"][field] = value

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_contract_v2_rejects_mdf_without_its_active_cutoff() -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    b_record["gdf_method"] = "mdf"
    for record in (b_record, a_record):
        record["comparison_contract"]["gdf_method"] = "mdf"

    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        {("lih-rocksalt", "aiccm-hf-direct"): a_record},
    ) == (None, None)


def test_compare_curated_control_cannot_gain_provenance_from_sidecar(
    tmp_path: Path,
) -> None:
    b_record, a_record = _matching_real_gamma_control_records("rhf-ri")
    a_record.pop("provenance")
    result_path = (
        tmp_path
        / "aiccm2026dev-a"
        / "lih-rocksalt"
        / "sto-3g"
        / "aiccm-hf-direct"
        / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(a_record) + "\n", encoding="utf-8")
    sidecar = result_path.with_name("RUN.meta")
    sidecar.write_text(
        "host: legacy-host\ndate: 2026-07-08\n",
        encoding="utf-8",
    )

    legacy_records = compare_b.load(tmp_path / "aiccm2026dev-a")
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        legacy_records,
    ) == (None, None)

    sidecar.write_text(
        "host: verified-host\n"
        "vibeqc_version: 0.15.33\n"
        f"vibeqc_commit: {'a' * 40}\n"
        "source_clean: true\n"
        "core_built: 2026-07-10T12:00:00\n"
        f"core_build_id: sha256:{'b' * 64}\n"
        f"probe_version: {probe_host.PROBE_VERSION}\n"
        "probe_passed: true\n",
        encoding="utf-8",
    )
    sidecar_only_records = compare_b.load(tmp_path / "aiccm2026dev-a")
    assert compare_b._matched_real_gamma_control_record(
        "lih-rocksalt",
        "rhf-ri",
        b_record,
        sidecar_only_records,
    ) == (None, None)


@pytest.mark.parametrize("nested", (False, True), ids=("raw", "curated"))
def test_compare_refuses_legacy_low_dimensional_b_absolute_energy(
    nested: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = (
        tmp_path
        / "aiccm2026dev-b"
        / "h-chain"
        / "sto-3g"
        / "rhf-ri"
        / "result.json"
        if nested
        else tmp_path / "h-chain__b-rhf-ri.json"
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        """
{
  "selector": "aiccm2026dev-b",
  "system": "h-chain",
  "route": "rhf-ri",
  "status": "ok",
  "dim": 1,
  "energy_per_atom_ha": -1.0
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["compare_b.py", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        compare_b.main()

    assert excinfo.value.code == 2
    assert "refusing to compare lower-dimensional" in capsys.readouterr().err


def test_b_inputs_use_the_shared_geometry_registry() -> None:
    jobs, _ = make_jobs_b.iter_jobs(
        "scf",
        system_filter="c-diamond",
        route_filter="rhf-4c",
    )
    assert len(jobs) == 1
    assert "--mesh 2 2 2" in jobs[0].command
    assert "--basis sto-3g" in jobs[0].command


def test_curate_b_writes_nested_testset_layout(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    producer_provenance = _producer_provenance("aiccm2026dev-b")

    def fake_run(cmd, capture_output, text, env):  # noqa: ANN001
        assert capture_output is True
        assert text is True
        assert env["VIBEQC_PYTHON"] == sys.executable
        calls.append(list(cmd))
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "lih-rocksalt__b-rhf-ri.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "converged": True,
                    "dim": 3,
                    "energy_per_atom_ha": -0.5001,
                    "provenance": producer_provenance,
                    "finite_torus_convention": {
                        "coulomb_kernel": "3d-periodic-g0",
                        "exchange_q0": "bvk-ewald",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(curate.subprocess, "run", fake_run)
    monkeypatch.setattr(curate, "_sha", lambda: "abc1234")
    monkeypatch.setattr(
        curate,
        "_provenance",
        lambda: pytest.fail("curator identity must not replace producer identity"),
    )

    ok = curate.curate(
        "lih-rocksalt",
        "rhf-ri",
        "sto-3g",
        ["--mesh", "2", "1", "1"],
        stream=curate.STREAMS["aiccm2026dev-b"],
        testset_root=tmp_path / "testset",
        tmp_dir=tmp_path / "scratch",
    )

    assert ok is True
    assert calls
    assert calls[0][0] == "bash"
    assert calls[0][1].endswith("run.sh")
    assert calls[0][2:5] == ["--b", "lih-rocksalt", "rhf-ri"]
    dest = (
        tmp_path
        / "testset"
        / "aiccm2026dev-b"
        / "lih-rocksalt"
        / "sto-3g"
        / "rhf-ri"
    )
    record = json.loads((dest / "result.json").read_text(encoding="utf-8"))
    assert record["selector"] == "aiccm2026dev-b"
    assert record["system"] == "lih-rocksalt"
    assert record["route"] == "rhf-ri"
    assert record["basis"] == "sto-3g"
    assert record["finite_torus_convention"]["exchange_q0"] == "bvk-ewald"
    meta = (dest / "RUN.meta").read_text(encoding="utf-8")
    assert "code: aiccm2026dev-b" in meta
    assert "state: ok" in meta
    assert "host: verified-host" in meta
    assert f"vibeqc_commit: {'a' * 40}" in meta
    assert "source_clean: true" in meta
    assert "core_built: 2026-07-11T12:00:00" in meta
    assert f"core_build_id: sha256:{'b' * 64}" in meta
    assert f"probe_version: {probe_host.PRODUCER_ATTESTATION_VERSION}" in meta
    assert f"probe_script_id: sha256:{'c' * 64}" in meta
    assert "probe_passed: true" in meta
    assert (
        f"probe_attestation_id: {producer_provenance['probe_attestation_id']}"
        in meta
    )
    assert (
        f"producer_payload_id: {producer_provenance['producer_payload_id']}"
        in meta
    )
    assert "exchange_q0: bvk-ewald" in meta
    assert "/Users/" not in meta
    audited = audit_b._json_records(tmp_path / "testset")
    assert [(path, item["route"]) for path, item in audited] == [
        (dest / "result.json", "rhf-ri")
    ]
    compared = compare_b.load(
        tmp_path / "testset" / "aiccm2026dev-b"
    )
    assert compared[("lih-rocksalt", "rhf-ri", None)]["selector"] == (
        "aiccm2026dev-b"
    )


def test_curate_refuses_untrusted_success_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    testset_root = tmp_path / "testset"
    existing = (
        testset_root
        / "aiccm2026dev-b"
        / "lih-rocksalt"
        / "sto-3g"
        / "rhf-ri"
        / "result.json"
    )
    existing.parent.mkdir(parents=True)
    existing.write_text('{"trusted": true}\n', encoding="utf-8")
    sidecar = existing.with_name("RUN.meta")
    sidecar.write_text("trusted metadata\n", encoding="utf-8")

    def fake_run(cmd, capture_output, text, env):  # noqa: ANN001
        assert capture_output is True
        assert text is True
        assert env["VIBEQC_PYTHON"] == sys.executable
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "lih-rocksalt__b-rhf-ri.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "converged": True,
                    "energy_per_atom_ha": -0.5,
                    "provenance": {"probe_passed": False},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(curate.subprocess, "run", fake_run)

    assert curate.curate(
        "lih-rocksalt",
        "rhf-ri",
        "sto-3g",
        [],
        stream=curate.STREAMS["aiccm2026dev-b"],
        testset_root=testset_root,
        tmp_dir=tmp_path / "scratch",
    ) is False
    assert existing.read_text(encoding="utf-8") == '{"trusted": true}\n'
    assert sidecar.read_text(encoding="utf-8") == "trusted metadata\n"


def test_curate_b_does_not_reuse_stale_scratch_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stale = scratch / "lih-rocksalt__b-rhf-ri.json"
    stale.write_text(
        json.dumps(
            {
                "status": "ok",
                "energy_per_atom_ha": -999.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run(cmd, capture_output, text, env):  # noqa: ANN001
        assert capture_output is True
        assert text is True
        assert env["VIBEQC_PYTHON"] == sys.executable
        assert not stale.exists()
        return SimpleNamespace(returncode=0, stdout="", stderr="no result")

    monkeypatch.setattr(curate.subprocess, "run", fake_run)
    monkeypatch.setattr(curate, "_provenance", lambda: {})

    ok = curate.curate(
        "lih-rocksalt",
        "rhf-ri",
        "sto-3g",
        [],
        stream=curate.STREAMS["aiccm2026dev-b"],
        testset_root=tmp_path / "testset",
        tmp_dir=scratch,
    )

    assert ok is False
    assert not stale.exists()
    assert not (tmp_path / "testset").exists()


def test_curate_reads_only_nested_producer_provenance() -> None:
    record = {
        "provenance": {
            "probe_version": probe_host.PRODUCER_ATTESTATION_VERSION,
            "probe_passed": True,
        },
        "probe_passed": False,
    }

    provenance = curate._recorded_provenance(record)
    assert provenance["probe_version"] == probe_host.PRODUCER_ATTESTATION_VERSION
    assert provenance["probe_passed"] == "true"
    assert provenance["host"] == "not-recorded"


def test_curate_b_stores_unsupported_fail_closed_records(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run(cmd, capture_output, text, env):  # noqa: ANN001
        assert env["VIBEQC_PYTHON"] == sys.executable
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "uniform-h-chain__b-rhf-4c.json").write_text(
            """
{
  "status": "unsupported",
  "converged": false,
  "reason": "four-center restricted to 3D"
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=3, stdout="", stderr="")

    monkeypatch.setattr(curate.subprocess, "run", fake_run)
    monkeypatch.setattr(curate, "_sha", lambda: "abc1234")

    ok = curate.curate(
        "uniform-h-chain",
        "rhf-4c",
        "sto-3g",
        [],
        stream=curate.STREAMS["aiccm2026dev-b"],
        testset_root=tmp_path / "testset",
        tmp_dir=tmp_path / "scratch",
    )

    assert ok is True
    dest = (
        tmp_path
        / "testset"
        / "aiccm2026dev-b"
        / "uniform-h-chain"
        / "sto-3g"
        / "rhf-4c"
    )
    assert '"status": "unsupported"' in (
        dest / "result.json"
    ).read_text(encoding="utf-8")
    meta = (dest / "RUN.meta").read_text(encoding="utf-8")
    assert "state: unsupported" in meta
    assert "return_code: 3" in meta


def test_curate_b_ingests_fetched_result_tree(monkeypatch, tmp_path: Path) -> None:
    fetched = tmp_path / "fetched"
    job_dir = fetched / "job-1234"
    job_dir.mkdir(parents=True)
    (job_dir / "lih-rocksalt__b-rhf-ri.json").write_text(
        """
{
  "selector": "aiccm2026dev-b",
  "status": "ok",
  "system": "lih-rocksalt",
  "route": "rhf-ri",
  "basis": "sto-3g",
  "converged": true,
  "dim": 3,
  "energy_per_atom_ha": -0.5001,
  "finite_torus_convention": {
    "coulomb_kernel": "3d-periodic-g0",
    "exchange_q0": "bvk-ewald"
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "lih-rocksalt__aiccm-ri.json").write_text(
        """
{
  "system": "lih-rocksalt",
  "route": "aiccm-ri",
  "basis": "sto-3g"
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(curate, "_sha", lambda: "abc1234")
    monkeypatch.setattr(
        curate,
        "_provenance",
        lambda: pytest.fail("ingest must not claim the curator's local build"),
    )

    summary = curate.ingest_results(
        fetched,
        stream=curate.STREAMS["aiccm2026dev-b"],
        testset_root=tmp_path / "testset",
    )

    assert summary == curate.IngestSummary(matched=1, written=1)
    dest = (
        tmp_path
        / "testset"
        / "aiccm2026dev-b"
        / "lih-rocksalt"
        / "sto-3g"
        / "rhf-ri"
    )
    record = (dest / "result.json").read_text(encoding="utf-8")
    assert '"selector": "aiccm2026dev-b"' in record
    meta = (dest / "RUN.meta").read_text(encoding="utf-8")
    assert "state: ok" in meta
    assert "return_code: not-recorded" in meta
    assert "host: not-recorded" in meta
    assert "vibeqc_version: not-recorded" in meta
    assert "vibeqc_commit: not-recorded" in meta
    assert "source_clean: not-recorded" in meta
    assert "core_built: not-recorded" in meta
    assert "core_build_id: not-recorded" in meta
    assert "probe_version: not-recorded" in meta
    assert "probe_passed: not-recorded" in meta
    assert "source: ingest job-1234/lih-rocksalt__b-rhf-ri.json" in meta
    assert "exchange_q0: bvk-ewald" in meta


def test_ingest_rejects_duplicate_destination_identity_before_writing(
    tmp_path: Path,
) -> None:
    fetched = tmp_path / "fetched"
    for job, energy in (("job-1", -0.5001), ("job-2", -0.5002)):
        path = fetched / job / "lih-rocksalt__b-rhf-ri.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "selector": "aiccm2026dev-b",
                    "system": "lih-rocksalt",
                    "route": "rhf-ri",
                    "basis": "sto-3g",
                    "status": "ok",
                    "energy_per_atom_ha": energy,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="duplicate destination identity"):
        curate.ingest_results(
            fetched,
            stream=curate.STREAMS["aiccm2026dev-b"],
            testset_root=tmp_path / "testset",
        )

    assert not (tmp_path / "testset").exists()


def test_ingest_refuses_to_overwrite_differing_existing_record(
    tmp_path: Path,
) -> None:
    fetched_record = {
        "selector": "aiccm2026dev-b",
        "system": "lih-rocksalt",
        "route": "rhf-ri",
        "basis": "sto-3g",
        "status": "ok",
        "energy_per_atom_ha": -0.5001,
    }
    fetched = tmp_path / "fetched" / "job-1" / "lih-rocksalt__b-rhf-ri.json"
    fetched.parent.mkdir(parents=True)
    fetched.write_text(json.dumps(fetched_record) + "\n", encoding="utf-8")

    existing = (
        tmp_path
        / "testset"
        / "aiccm2026dev-b"
        / "lih-rocksalt"
        / "sto-3g"
        / "rhf-ri"
        / "result.json"
    )
    existing.parent.mkdir(parents=True)
    existing_record = dict(fetched_record, energy_per_atom_ha=-0.4999)
    existing.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differing record"):
        curate.ingest_results(
            tmp_path / "fetched",
            stream=curate.STREAMS["aiccm2026dev-b"],
            testset_root=tmp_path / "testset",
        )

    assert json.loads(existing.read_text(encoding="utf-8")) == existing_record


def test_ingest_identical_existing_record_is_a_true_noop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = {
        "selector": "aiccm2026dev-b",
        "system": "lih-rocksalt",
        "route": "rhf-ri",
        "basis": "sto-3g",
        "status": "ok",
        "energy_per_atom_ha": -0.5001,
    }
    fetched = tmp_path / "fetched" / "job-1" / "lih-rocksalt__b-rhf-ri.json"
    fetched.parent.mkdir(parents=True)
    fetched.write_text(json.dumps(record) + "\n", encoding="utf-8")

    existing = (
        tmp_path
        / "testset"
        / "aiccm2026dev-b"
        / "lih-rocksalt"
        / "sto-3g"
        / "rhf-ri"
        / "result.json"
    )
    existing.parent.mkdir(parents=True)
    original_result = json.dumps(record) + "\n"
    original_meta = (
        f"vibeqc_commit: {'a' * 40}\n"
        "source_clean: true\n"
        f"core_build_id: sha256:{'b' * 64}\n"
        f"probe_version: {probe_host.PROBE_VERSION}\n"
        "probe_passed: true\n"
    )
    existing.write_text(original_result, encoding="utf-8")
    sidecar = existing.with_name("RUN.meta")
    sidecar.write_text(original_meta, encoding="utf-8")

    summary = curate.ingest_results(
        tmp_path / "fetched",
        stream=curate.STREAMS["aiccm2026dev-b"],
        testset_root=tmp_path / "testset",
    )

    assert summary == curate.IngestSummary(matched=1, written=0)
    assert curate.main(
        [
            "--stream",
            "aiccm2026dev-b",
            "--testset-root",
            str(tmp_path / "testset"),
            "--ingest-results",
            str(tmp_path / "fetched"),
        ]
    ) == 0
    assert "1 matched; 0 written" in capsys.readouterr().out
    assert existing.read_text(encoding="utf-8") == original_result
    assert sidecar.read_text(encoding="utf-8") == original_meta


@pytest.mark.parametrize("nonmatching", (False, True), ids=("empty", "nonmatching"))
def test_ingest_cli_fails_when_no_records_match(
    nonmatching: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    if nonmatching:
        (fetched / "lih-rocksalt__aiccm-ri.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

    assert curate.main(
        [
            "--stream",
            "aiccm2026dev-b",
            "--testset-root",
            str(tmp_path / "testset"),
            "--ingest-results",
            str(fetched),
        ]
    ) == 1
    assert "0 matched; 0 written" in capsys.readouterr().out


def test_ingest_preserves_producer_provenance_from_result() -> None:
    record = {
        "provenance": {
            "host": "compute-large",
            "vibeqc_version": "0.15.33",
            "vibeqc_commit": "a" * 40,
            "source_clean": "true",
            "core_built": "2026-07-10T12:00:00",
            "core_build_id": "sha256:" + "b" * 64,
            "probe_version": probe_host.PROBE_VERSION,
            "probe_script_id": "sha256:" + "c" * 64,
            "probe_passed": "true",
            "probe_attestation_id": "sha256:" + "d" * 64,
            "producer_payload_id": "sha256:" + "e" * 64,
        }
    }

    assert curate._recorded_provenance(record) == record["provenance"]


def test_curate_helper_has_no_private_default_path() -> None:
    helper = (_TESTSET_DIR / "curate.py").read_text(encoding="utf-8")
    assert "/Users/" not in helper
    assert "AICCM2026_TESTSET_ROOT" in helper
