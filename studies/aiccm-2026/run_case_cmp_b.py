#!/usr/bin/env python3
"""Run one D101 χ-CCM RI route-control diagnostic without admitting energy.

This is the χ-owned producer for
``COMPARISON_1D2D3D_2026-07-15.md``.  It deliberately differs from
``run_case_cmp.py``, which is the neutral fitted-torus Bloch/GDF control.
The only electronic-structure entry point used here is
``run_periodic_job(..., jk_method="aiccm2026dev-b", aiccm_backend="ri")``.

The current fitted RI support is D93 ``not-qualified``.  A successful run is
therefore route-plumbing evidence only: the producer audits the total-energy
assembly internally but puts no absolute energy in the campaign JSON.  The
ordinary runner output and SCF log remain nonreportable diagnostic artifacts.
It also refuses
bundle provenance (the normative D91 bundle contract is not implemented),
requires the imported clean
checkout to equal an explicitly pinned ``origin/main`` SHA, and checks that
every executable payload byte is the tracked byte at that SHA.

The current managed-host managed runtime is a bundle and cannot satisfy this D77
contract even after a routine refresh.  The following workflow applies only
after the interpreter selected by ``run.sh`` is a separate clean Git checkout
(either through the managed wrapper or explicit ``VIBEQC_PYTHON``; System 3D
first)::

    git fetch origin
    SHA=$(git rev-parse origin/main)
    vq submit --host managed-host --job-name d101-chi-3d-ri -d studies/aiccm-2026/ -- \
      bash run.sh --comparison-b 3d --expected-source-sha "$SHA" \
      --vq-host managed-host

After fetching ``d101-cmp1d2d3d-3d-chi-rhf-ri.json`` into the submitted
directory, pass that exact file as ``--validation-record`` for the ``1d`` and
``2d`` jobs, using distinct queue job names.  Those labels describe the
embedded object; every cell remains declared ``dim=3`` periodic-in-vacuum.

For example, the first follow-up is::

    vq submit --host managed-host --job-name d101-chi-1d-ri -d studies/aiccm-2026/ -- \
      bash run.sh --comparison-b 1d --expected-source-sha "$SHA" \
      --vq-host managed-host \
      --validation-record d101-cmp1d2d3d-3d-chi-rhf-ri.json

Use the analogous ``2d`` selector and ``d101-chi-2d-ri`` job name for the
second follow-up.
"""

from __future__ import annotations

import site_settings

import argparse
import hashlib
import inspect
import json
import os
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import product as cartesian_product
from math import isfinite, prod
from pathlib import Path
from typing import Any

import numpy as np
import probe_host
import vibeqc as vq
from b_routes import (
    ROUTES,
    TWO_ELECTRON_SUPPORT_SCHEMA,
    exact_exchange_assembly_failure,
    two_electron_support_reportability_failure,
    two_electron_support_schema_failure,
)
from vibeqc import Atom, BasisSet, PeriodicSystem
from vibeqc.aux_basis import default_aux_for
from vibeqc.madelung import madelung_constant_for_cell
from vibeqc.progress import ProgressLogger


SCHEMA = "vibeqc.aiccm2026dev-b.campaign-1d2d3d/v1"
FAILURE_SCHEMA = "vibeqc.aiccm2026dev-b.campaign-1d2d3d-failure/v1"
SPEC = "COMPARISON_1D2D3D_2026-07-15"
STREAM = "aiccm2026dev-b"
SELECTOR = "aiccm2026dev-b"
ROUTE = "rhf-ri"
ENTRY_POINT = "run_periodic_job"
EVIDENCE_ROLE = "route-plumbing-diagnostic"
QUANTITATIVE_STATUS = "not-qualified"
COMPARISON_STATUS = "no-gamma-chi-construction-comparison-defined"
EXPECTED_AUX = "def2-svp-jk"
BASIS = "sto-3g"
ROUTING_EVENT = (
    "multi-k HF/hybrid exchange: routing to the compcell GDF path "
    "with exxdiv='ewald' (the use_compcell=False Ewald-3D-K path "
    "omits the exxdiv Madelung exchange-divergence correction)."
)
INITIAL_GUESS_EVENT = "initial guess: HCORE (per-k Hcore diagonalisation)"
TWO_ELECTRON_SUPPORT_REASON = (
    "no accepted high-G RSGDF tail support contract is active; "
    "the campaign record is route-plumbing evidence only"
)
_PROVENANCE_KEYS = {
    "attestation",
    "host",
    "vibeqc_version",
    "vibeqc_commit",
    "source_clean",
    "core_build_id",
    "core_built",
    "probe_version",
    "probe_script_id",
    "probe_passed",
    "probe_attestation_id",
    "probe_attestation",
    "producer_payload_id",
    "expected_source_commit",
}

SYSTEMS: dict[str, dict[str, object]] = {
    "1d": {
        "label": "LiH chain, declared 3-D periodic-in-vacuum",
        "declared_model": "3d-periodic-in-vacuum",
        "vectors": [
            [6.0000, 0.0, 0.0],
            [0.0, 30.0000, 0.0],
            [0.0, 0.0, 30.0000],
        ],
        "atoms": [
            (3, [0.0000, 15.0000, 15.0000]),
            (1, [3.0000, 15.0000, 15.0000]),
        ],
        "mesh": (8, 1, 1),
        "n_electrons": 4,
    },
    "2d": {
        "label": "h-BN monolayer, declared 3-D periodic-in-vacuum",
        "declared_model": "3d-periodic-in-vacuum",
        "vectors": [
            [4.7319, 0.0, 0.0],
            [-2.3660, 4.0979, 0.0],
            [0.0, 0.0, 30.0000],
        ],
        "atoms": [
            (5, [0.0000, 2.7319, 15.0000]),
            (7, [2.3659, 1.3660, 15.0000]),
        ],
        "mesh": (3, 3, 1),
        "n_electrons": 12,
    },
    "3d": {
        "label": "LiH rocksalt, fcc primitive cell",
        "declared_model": "3d-periodic",
        "vectors": [
            [0.0, 3.8579, 3.8579],
            [3.8579, 0.0, 3.8579],
            [3.8579, 3.8579, 0.0],
        ],
        "atoms": [
            (3, [0.0000, 0.0000, 0.0000]),
            (1, [3.8579, 3.8579, 3.8579]),
        ],
        "mesh": (2, 2, 2),
        "n_electrons": 4,
    },
}

PINS: dict[str, object] = {
    "method": "RHF",
    "functional": None,
    "jk_method": SELECTOR,
    "aiccm_backend": "ri",
    "basis": BASIS,
    "gdf_method": "rsgdf",
    "rsgdf_ke_cutoff_ha": 200.0,
    "conv_tol_energy_ha": 1.0e-10,
    "max_iter": 128,
    "use_diis": True,
    "diis_start_iter": 2,
    "diis_subspace_size": 8,
    "damping": 0.0,
    "dynamic_damping": False,
    "fock_mixing": 0.0,
    "level_shift": 0.0,
    "smearing_temperature_ha": 0.0,
    "initial_guess": "HCORE",
    "exchange_exxdiv": "ewald",
    "convergence_strategy": "off",
}

SOURCE_RESOLVED_PINS: dict[str, float] = {
    "ao_linear_dependence_threshold": 1.0e-7,
    "auxiliary_metric_linear_dependence_threshold": 1.0e-9,
    "conv_tol_grad": 1.0e-6,
}

# Full ancestry requirements already present when this campaign was issued or
# repaired.  The expected tip is separately required to equal origin/main.
SOURCE_REQUIREMENTS: dict[str, str] = {
    "d88_column_lattice": "d746d238cce5fa8cf868cda27b79ffb5025237fe",
    "rsgdf_column_bound": "85b5afe532a1f6835a9ab4252469dd5d6cffb5ff",
    "cosx_one_center": "605efb6954252ac0a18751c600c2c1ae46305b88",
    "d99_ewald_nuclear": "ddbe859d1fbe7b8be0e408cfb31a5147cafba0ac",
    "d99_dense_core_guard": "6ce1423392afc1b1f8c63bdd63d41bbdb8e3a628",
    "d99_shared_q_cache": "ad96a2f63d456632b993b6ccd81c757b1a807191",
}

_PAYLOAD_TRACKED_PATHS = {
    "producer": "studies/aiccm-2026/run_case_cmp_b.py",
    "probe_host": "studies/aiccm-2026/probe_host.py",
    "testset": "studies/aiccm-2026/testset.py",
    "b_routes": "studies/aiccm-2026/b_routes.py",
    "launcher": "studies/aiccm-2026/run.sh",
}
_PAYLOAD_LOCAL_PATHS = {
    "producer": Path(__file__),
    "probe_host": Path(probe_host.__file__),
    "testset": Path(__file__).resolve().parent / "testset.py",
    "b_routes": Path(__file__).resolve().parent / "b_routes.py",
    "launcher": Path(__file__).resolve().parent / "run.sh",
}

_attestation_path = os.environ.get("AICCM_PROBE_ATTESTATION")
_ATTESTATION = Path(_attestation_path) if _attestation_path else None


def _is_full_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_vq_job_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{12}", value) is not None


def _is_slurm_job_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _finite_numeric_text(value: object) -> float | None:
    """Parse one finite number emitted by the plain-text settings dump."""

    if not isinstance(value, str):
        return None
    try:
        number = float(value.strip())
    except (OverflowError, ValueError):
        return None
    return number if isfinite(number) else None


def _json_exact(actual: object, expected: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_exact(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate member names."""

    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON constant")


def _strict_json_loads(value: str | bytes) -> object:
    """Decode standards-compliant JSON with no duplicate object members."""

    text = value.decode("utf-8") if isinstance(value, bytes) else value
    return json.loads(
        text,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_json_constant,
    )


def _is_aware_iso_timestamp(value: object) -> bool:
    """Return whether *value* is one canonical UTC second-resolution timestamp."""

    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
        and parsed.isoformat(timespec="seconds") == value
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _source_root() -> Path | None:
    location = getattr(vq, "__file__", None)
    if not location:
        return None
    return next(
        (parent for parent in Path(location).resolve().parents if (parent / ".git").exists()),
        None,
    )


def _git_output(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_binding(expected_source_sha: str) -> tuple[dict[str, object], list[str]]:
    """Bind the copied producer payload to exact tracked bytes on origin/main."""

    failures: list[str] = []
    root = _source_root()
    if root is None:
        return {"status": "source-root-not-found"}, [
            "imported vibeqc source root is unavailable"
        ]

    origin_main = _git_output(root, "rev-parse", "origin/main")
    if origin_main != expected_source_sha:
        failures.append(
            "expected source SHA does not equal the imported checkout's "
            f"origin/main: expected={expected_source_sha!r}, "
            f"origin/main={origin_main!r}"
        )

    ancestry: dict[str, bool] = {}
    for name, required in SOURCE_REQUIREMENTS.items():
        try:
            completed = subprocess.run(
                ["git", "merge-base", "--is-ancestor", required, expected_source_sha],
                cwd=root,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            held = completed.returncode == 0
        except OSError:
            held = False
        ancestry[name] = held
        if not held:
            failures.append(
                f"expected source SHA does not contain required ancestor {required} ({name})"
            )

    tracked_files: dict[str, dict[str, object]] = {}
    for key, tracked_path in _PAYLOAD_TRACKED_PATHS.items():
        try:
            tracked_bytes = subprocess.check_output(
                ["git", "show", f"{expected_source_sha}:{tracked_path}"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            tracked_bytes = b""
        tracked_digest = _sha256_bytes(tracked_bytes) if tracked_bytes else "unknown"
        copied_digest = probe_host.file_sha256(_PAYLOAD_LOCAL_PATHS[key])
        matches = tracked_digest != "unknown" and copied_digest == tracked_digest
        tracked_files[key] = {
            "tracked_path": tracked_path,
            "tracked_sha256": tracked_digest,
            "copied_sha256": copied_digest,
            "matches": matches,
        }
        if not matches:
            failures.append(
                f"copied payload file {key!r} does not match "
                f"{expected_source_sha}:{tracked_path}"
            )

    return {
        "expected_source_commit": expected_source_sha,
        "origin_main_commit": origin_main,
        "required_ancestors": {
            name: {"commit": commit, "is_ancestor": ancestry[name]}
            for name, commit in SOURCE_REQUIREMENTS.items()
        },
        "tracked_files": tracked_files,
        "passed": not failures,
    }, failures


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _source_binding_failure(
    value: object,
    expected_source_sha: str,
) -> str | None:
    """Validate the serialized source-to-payload binding exactly."""

    if not isinstance(value, dict) or set(value) != {
        "expected_source_commit",
        "origin_main_commit",
        "required_ancestors",
        "tracked_files",
        "passed",
    }:
        return "source_binding does not have the exact D101 keys"
    if (
        value.get("expected_source_commit") != expected_source_sha
        or value.get("origin_main_commit") != expected_source_sha
        or value.get("passed") is not True
    ):
        return "source_binding does not bind the passing origin/main tip"

    ancestors = value.get("required_ancestors")
    if not isinstance(ancestors, dict) or set(ancestors) != set(
        SOURCE_REQUIREMENTS
    ):
        return "source_binding does not contain the exact required ancestors"
    for name, commit in SOURCE_REQUIREMENTS.items():
        if not _json_exact(
            ancestors.get(name),
            {"commit": commit, "is_ancestor": True},
        ):
            return f"source_binding does not establish required ancestor {name}"

    files = value.get("tracked_files")
    if not isinstance(files, dict) or set(files) != set(_PAYLOAD_TRACKED_PATHS):
        return "source_binding does not contain the exact payload files"
    for name, tracked_path in _PAYLOAD_TRACKED_PATHS.items():
        item = files.get(name)
        if not isinstance(item, dict) or set(item) != {
            "tracked_path",
            "tracked_sha256",
            "copied_sha256",
            "matches",
        }:
            return f"source_binding payload entry {name!r} has the wrong keys"
        if (
            item.get("tracked_path") != tracked_path
            or item.get("matches") is not True
            or not _is_sha256(item.get("tracked_sha256"))
            or item.get("copied_sha256") != item.get("tracked_sha256")
        ):
            return f"source_binding payload entry {name!r} is not exact"
    return None


def _source_payload_binding_failure(
    source_binding: object,
    provenance: object,
) -> str | None:
    """Cross-bind tracked source bytes to the attested producer payload."""

    if not isinstance(source_binding, dict) or not isinstance(provenance, dict):
        return "source/payload cross-binding inputs are missing"
    tracked = source_binding.get("tracked_files")
    attestation = provenance.get("probe_attestation")
    payload = (
        attestation.get("producer_payload")
        if isinstance(attestation, dict)
        else None
    )
    payload_files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(tracked, dict) or not isinstance(payload_files, dict):
        return "source/payload cross-binding evidence is missing"
    if set(tracked) != set(_PAYLOAD_TRACKED_PATHS) or set(payload_files) != set(
        _PAYLOAD_TRACKED_PATHS
    ):
        return "source/payload cross-binding file sets contradict D101"
    for name in _PAYLOAD_TRACKED_PATHS:
        item = tracked.get(name)
        if (
            not isinstance(item, dict)
            or not _json_exact(item.get("tracked_sha256"), payload_files.get(name))
            or not _json_exact(item.get("copied_sha256"), payload_files.get(name))
        ):
            return f"source/payload digest differs for {name!r}"
    return None


def _producer_payload() -> dict[str, object]:
    return {
        "version": probe_host.PRODUCER_PAYLOAD_VERSION,
        "stream": STREAM,
        "files": {
            key: probe_host.file_sha256(path)
            for key, path in _PAYLOAD_LOCAL_PATHS.items()
        },
    }


def _core_built_at() -> str:
    try:
        from vibeqc import _vibeqc_core

        core = Path(getattr(_vibeqc_core, "__file__", "") or "")
        if core.is_file():
            return datetime.fromtimestamp(
                core.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001 - provenance must fail closed
        pass
    return "unknown"


def _provenance(expected_source_sha: str) -> tuple[dict[str, object], list[str]]:
    """Require one D77 clean-checkout attestation before computation."""

    failures: list[str] = []
    if os.environ.get("VIBEQC_BUNDLE"):
        failures.append("D91 bundle attestation is not implemented for B campaign rows")
    if os.environ.get("AICCM_SKIP_PROBE") == "1":
        failures.append("AICCM_SKIP_PROBE cannot be used for this campaign")
    if _ATTESTATION is None:
        failures.append("AICCM_PROBE_ATTESTATION is required")

    attestation = None
    if not failures:
        attestation = probe_host.producer_attestation(
            _ATTESTATION,
            _producer_payload(),
        )
        if attestation is None:
            failures.append("D77 producer attestation failed")
        elif attestation.get("source_commit") != expected_source_sha:
            failures.append(
                "D77 source commit does not equal expected origin/main SHA: "
                f"{attestation.get('source_commit')!r} != {expected_source_sha!r}"
            )

    provenance: dict[str, object] = {
        "attestation": "git-checkout",
        "host": "unknown",
        "vibeqc_version": str(getattr(vq, "__version__", "unknown")),
        "vibeqc_commit": "unknown",
        "source_clean": False,
        "core_build_id": probe_host.core_build_id(),
        "core_built": _core_built_at(),
        "probe_version": probe_host.PRODUCER_ATTESTATION_VERSION,
        "probe_script_id": probe_host.probe_script_id(),
        "probe_passed": False,
        "probe_attestation_id": "unknown",
        "probe_attestation": attestation,
        "producer_payload_id": probe_host.probe_attestation_id(_producer_payload()),
        "expected_source_commit": expected_source_sha,
    }
    if attestation is not None:
        provenance.update(
            host=attestation["host"],
            vibeqc_version=attestation["vibeqc_version"],
            vibeqc_commit=attestation["source_commit"],
            source_clean=attestation["source_clean"],
            core_build_id=attestation["core_build_id"],
            probe_script_id=attestation["attestor_script_id"],
            probe_passed=not failures,
            probe_attestation_id=probe_host.probe_attestation_id(attestation),
            producer_payload_id=attestation["producer_payload_id"],
        )
    return provenance, failures


def _finalize_provenance(
    provenance: dict[str, object],
    expected_source_sha: str,
) -> list[str]:
    failures: list[str] = []
    finalized = probe_host.finalize_producer_attestation(
        provenance.get("probe_attestation"),
        _producer_payload(),
    )
    if finalized is None or finalized.get("probe_passed") is not True:
        failures.append("D77 producer identity was not stable through result assembly")
        provenance.update(
            probe_passed=False,
            probe_attestation=None,
            probe_attestation_id="unknown",
        )
        return failures
    if finalized.get("source_commit") != expected_source_sha:
        failures.append("final D77 source commit no longer equals expected origin/main SHA")
    provenance.update(
        host=finalized["host"],
        vibeqc_version=finalized["vibeqc_version"],
        vibeqc_commit=finalized["source_commit"],
        source_clean=finalized["source_clean"],
        core_build_id=finalized["core_build_id"],
        core_built=_core_built_at(),
        probe_script_id=finalized["attestor_script_id"],
        probe_passed=not failures,
        probe_attestation_id=probe_host.probe_attestation_id(finalized),
        probe_attestation=finalized,
        producer_payload_id=finalized["producer_payload_id"],
    )
    return failures


def _serialized_provenance_failure(
    provenance: object,
    expected_source_sha: str,
) -> str | None:
    """Validate a stored D77 record for the 3D-first receipt gate."""

    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_KEYS:
        return "serialized D77 campaign provenance does not have the exact keys"
    failure = probe_host.serialized_producer_attestation_failure(
        provenance,
        _producer_payload(),
        expected_source_commit=expected_source_sha,
    )
    if failure is not None:
        return failure
    attestation = provenance.get("probe_attestation")
    if not isinstance(attestation, dict):
        return "serialized D77 campaign attestation is missing"
    exact_aliases = {
        "host": attestation.get("host"),
        "vibeqc_version": attestation.get("vibeqc_version"),
        "vibeqc_commit": attestation.get("source_commit"),
        "source_clean": attestation.get("source_clean"),
        "core_build_id": attestation.get("core_build_id"),
        "probe_version": attestation.get("attestation_version"),
        "probe_script_id": attestation.get("attestor_script_id"),
        "probe_passed": attestation.get("probe_passed"),
        "probe_attestation_id": probe_host.probe_attestation_id(attestation),
        "producer_payload_id": attestation.get("producer_payload_id"),
    }
    if any(
        not _json_exact(provenance.get(name), expected)
        for name, expected in exact_aliases.items()
    ):
        return "serialized D77 campaign aliases are not type-exact"
    if (
        provenance.get("attestation") != "git-checkout"
        or provenance.get("expected_source_commit") != expected_source_sha
        or not _is_aware_iso_timestamp(provenance.get("core_built"))
    ):
        return "serialized D77 campaign aliases contradict the source contract"
    return None


def build_system(key: str) -> PeriodicSystem:
    """Build one exact campaign cell and enforce column-vector semantics."""

    meta = SYSTEMS[key]
    vectors = np.asarray(meta["vectors"], dtype=float)
    lattice = vectors.T
    atoms = [Atom(int(z), list(position)) for z, position in meta["atoms"]]
    system = PeriodicSystem(3, lattice, atoms, 0, 1)
    stored = np.asarray(system.lattice, dtype=float)
    if not np.allclose(stored, lattice, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            f"{key} lattice did not round-trip as columns (D88 fail-close)"
        )
    molecule = system.unit_cell_molecule()
    if int(molecule.n_electrons()) != int(meta["n_electrons"]):
        raise RuntimeError(f"{key} electron count contradicts the campaign input")
    if int(system.dim) != 3 or int(system.charge) != 0 or int(system.multiplicity) != 1:
        raise RuntimeError(f"{key} is not the required neutral closed-shell dim=3 cell")
    return system


def _source_resolved_thresholds() -> tuple[dict[str, object], list[str]]:
    from vibeqc._vibeqc_core import PeriodicRHFOptions
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    failures: list[str] = []
    signature = inspect.signature(run_krhf_periodic_gdf)
    resolved: dict[str, object] = {}
    expected = {
        "linear_dep_threshold": SOURCE_RESOLVED_PINS[
            "ao_linear_dependence_threshold"
        ],
        "gdf_linear_dep_threshold": SOURCE_RESOLVED_PINS[
            "auxiliary_metric_linear_dependence_threshold"
        ],
    }
    for parameter, value in expected.items():
        actual = signature.parameters[parameter].default
        resolved[parameter] = {
            "value": float(actual),
            "evidence": "source-resolved-default+executed-progress-event",
            "source": f"vibeqc.periodic_k_gdf.run_krhf_periodic_gdf:{parameter}",
        }
        if actual != value:
            failures.append(
                f"source-resolved {parameter}={actual!r}, campaign pin is {value!r}"
            )
    grad_threshold = float(PeriodicRHFOptions().conv_tol_grad)
    resolved["conv_tol_grad"] = {
        "value": grad_threshold,
        "evidence": "source-resolved-option-default+executed-progress-event",
        "source": "vibeqc._vibeqc_core.PeriodicRHFOptions:conv_tol_grad",
    }
    expected_grad = SOURCE_RESOLVED_PINS["conv_tol_grad"]
    if grad_threshold != expected_grad:
        failures.append(
            "source-resolved conv_tol_grad="
            f"{grad_threshold!r}, campaign source pin is {expected_grad!r}"
        )
    return resolved, failures


def _runner_kwargs(
    output_stem: Path,
    mesh: tuple[int, int, int],
    aux_basis: str,
    progress: ProgressLogger,
) -> dict[str, object]:
    return {
        "method": PINS["method"],
        "functional": PINS["functional"],
        "jk_method": PINS["jk_method"],
        "aiccm_backend": PINS["aiccm_backend"],
        "aiccm_lattice_extension": mesh,
        "aux_basis": aux_basis,
        "gdf_method": PINS["gdf_method"],
        "rsgdf_ke_cutoff": PINS["rsgdf_ke_cutoff_ha"],
        "convergence": PINS["convergence_strategy"],
        "use_diis": PINS["use_diis"],
        "diis_start_iter": PINS["diis_start_iter"],
        "diis_subspace_size": PINS["diis_subspace_size"],
        "damping": PINS["damping"],
        "dynamic_damping": PINS["dynamic_damping"],
        "fock_mixing": PINS["fock_mixing"],
        "level_shift": PINS["level_shift"],
        "smearing_temperature": PINS["smearing_temperature_ha"],
        "max_iter": PINS["max_iter"],
        "conv_tol_energy": PINS["conv_tol_energy_ha"],
        "initial_guess": PINS["initial_guess"],
        "exchange_exxdiv": PINS["exchange_exxdiv"],
        "aiccm_symmetry": "off",
        "solver": "dense",
        "output": output_stem,
        "output_qvf": False,
        "write_molden_file": False,
        "write_density": False,
        "write_xyz_file": False,
        "write_poscar_file": False,
        "write_xsf_structure_file": False,
        "write_cif_file": False,
        "write_population_file": False,
        "citations": False,
        "progress": progress,
    }


def _requested_input(
    mesh: tuple[int, int, int],
    aux_basis: str,
) -> dict[str, object]:
    """Separate explicit high-level inputs from inherited route defaults."""

    return {
        "explicit": {
            **PINS,
            "aiccm_lattice_extension": list(mesh),
            "aux_basis_name": aux_basis,
            "aux_basis_resolution": "explicit-default_aux_for(sto-3g)",
        },
        "source_resolved": {
            "ao_linear_dependence_threshold": {
                "value": SOURCE_RESOLVED_PINS[
                    "ao_linear_dependence_threshold"
                ],
                "source": (
                    "vibeqc.periodic_k_gdf.run_krhf_periodic_gdf:"
                    "linear_dep_threshold"
                ),
            },
            "auxiliary_metric_linear_dependence_threshold": {
                "value": SOURCE_RESOLVED_PINS[
                    "auxiliary_metric_linear_dependence_threshold"
                ],
                "source": (
                    "vibeqc.periodic_k_gdf.run_krhf_periodic_gdf:"
                    "gdf_linear_dep_threshold"
                ),
            },
            "conv_tol_grad": {
                "value": SOURCE_RESOLVED_PINS["conv_tol_grad"],
                "source": (
                    "vibeqc._vibeqc_core.PeriodicRHFOptions:conv_tol_grad"
                ),
            },
        },
    }


def _convention_record(result: object) -> dict[str, object]:
    convention = getattr(result, "finite_torus_convention", None)
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
        "lattice_vector_convention": convention.lattice_vector_convention,
        "orbital_energy_convention": convention.orbital_energy_convention,
    }


def _exact_exchange_record(result: object) -> dict[str, object]:
    assembly = getattr(result, "exact_exchange_assembly", None)
    if assembly is None:
        return {"status": "not-recorded"}
    return {
        "schema": assembly.schema,
        "c_full": float(assembly.c_full),
        "c_sr": float(assembly.c_sr),
        "omega_screen_bohr_inv": float(assembly.omega_screen_bohr_inv),
        "resolver": assembly.resolver,
        "screened_exchange_applicability": assembly.screened_exchange_applicability,
        "screened_exchange_assembly": assembly.screened_exchange_assembly,
    }


def _runtime_backend(result: object) -> str:
    value = getattr(result, "runtime_backend", None)
    return value if isinstance(value, str) and value.strip() else "not-recorded"


def _two_electron_support(result: object) -> dict[str, object]:
    return {
        "schema": TWO_ELECTRON_SUPPORT_SCHEMA,
        "qualification": "not-qualified",
        "reason": TWO_ELECTRON_SUPPORT_REASON,
        "backend": "ri",
        "runtime_backend": _runtime_backend(result),
        "direct": None,
        "fitted": {
            "gdf_method": "rsgdf",
            "rsgdf_base_ke_cutoff_ha": 200.0,
            "rsgdf_tail_ke_cutoff_ha": None,
            "mdf_ke_cutoff_ha": None,
            "cosx_exchange_support": "not-applicable",
            "correlation_support": "not-applicable",
        },
    }


def _recomputed_bvk_madelung(
    system: PeriodicSystem,
    bvk_lattice: object,
) -> float | None:
    """Recompute the BvK Madelung scalar through the shared code helper."""

    try:
        lattice = np.asarray(bvk_lattice, dtype=float)
        if lattice.shape != (3, 3):
            return None
        atoms = [
            Atom(int(atom.Z), [float(value) for value in atom.xyz])
            for atom in system.unit_cell
        ]
        bvk_system = PeriodicSystem(
            3,
            lattice,
            atoms,
            int(system.charge),
            int(system.multiplicity),
        )
        return _finite_number(madelung_constant_for_cell(bvk_system))
    except Exception:  # noqa: BLE001 - campaign audit must fail closed
        return None


def _setting_values(log_text: str, name: str) -> list[str]:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(.*?)\s*$")
    return [match.group(1) for line in log_text.splitlines() if (match := pattern.match(line))]


def _audit_result(
    key: str,
    system: PeriodicSystem,
    result: object,
    log_text: str,
    aux_basis: str,
    source_thresholds: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Build non-quantitative campaign evidence and return every violation."""

    meta = SYSTEMS[key]
    mesh = tuple(int(value) for value in meta["mesh"])
    failures: list[str] = []
    convention = _convention_record(result)
    assembly = _exact_exchange_record(result)
    support = _two_electron_support(result)

    if getattr(result, "backend", None) != "aiccm2026dev-b-ri":
        failures.append("result backend is not aiccm2026dev-b-ri")
    assembly_failure = exact_exchange_assembly_failure(assembly, ROUTE)
    if assembly_failure:
        failures.append(f"exact-exchange assembly: {assembly_failure}")
    support_failure = two_electron_support_schema_failure(support, ROUTE)
    if support_failure:
        failures.append(f"two-electron support: {support_failure}")
    reportability_failure = two_electron_support_reportability_failure(support, ROUTE)
    if reportability_failure is None:
        failures.append("D93 fitted support unexpectedly claimed reportability")

    expected_convention = {
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
        "lattice_vector_convention": "columns",
    }
    for name, expected in expected_convention.items():
        if convention.get(name) != expected:
            failures.append(
                f"finite-torus convention {name}={convention.get(name)!r}, "
                f"expected {expected!r}"
            )
    primitive = np.asarray(system.lattice, dtype=float)
    expected_bvk = primitive @ np.diag(np.asarray(mesh, dtype=float))
    try:
        recorded_bvk = np.asarray(
            convention["bvk_madelung_supercell_lattice_bohr"],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        recorded_bvk = np.empty((0, 0))
    bvk_matches = recorded_bvk.shape == (3, 3) and np.allclose(
        recorded_bvk,
        expected_bvk,
        rtol=0.0,
        atol=1.0e-12,
    )
    if not bvk_matches:
        failures.append("recorded BvK lattice does not equal A @ diag(mesh)")

    n_characters = prod(mesh)
    try:
        result_frac = np.asarray(
            getattr(result, "kpoints_frac"), dtype=float
        ).reshape(-1, 3)
        result_cart = np.asarray(
            getattr(result, "kpoints_cart"), dtype=float
        ).reshape(-1, 3)
        result_weights = np.asarray(
            getattr(result, "kpoint_weights"), dtype=float
        ).reshape(-1)
        shape_matches = (
            result_frac.shape == (n_characters, 3)
            and result_cart.shape == (n_characters, 3)
            and result_weights.shape == (n_characters,)
        )
        scaled = result_frac * np.asarray(mesh, dtype=float)[None, :]
        integral_residues = shape_matches and np.allclose(
            scaled,
            np.rint(scaled),
            rtol=0.0,
            atol=1.0e-12,
        )
        residue_rows = {
            tuple(
                int(np.rint(row[axis])) % mesh[axis]
                for axis in range(3)
            )
            for row in scaled
        }
        expected_residues = set(
            cartesian_product(*(range(value) for value in mesh))
        )
        residue_bijection = bool(
            integral_residues and residue_rows == expected_residues
        )
        reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
        cartesian_matches = shape_matches and np.allclose(
            result_cart,
            (reciprocal @ result_frac.T).T,
            rtol=0.0,
            atol=1.0e-12,
        )
    except (AttributeError, TypeError, ValueError):
        result_frac = np.empty((0, 3))
        result_weights = np.empty((0,))
        residue_bijection = False
        cartesian_matches = False
    kmesh_matches = bool(residue_bijection and cartesian_matches)
    uniform_weights = (
        result_weights.shape == (n_characters,)
        and np.allclose(
            result_weights,
            1.0 / n_characters,
            rtol=0.0,
            atol=1.0e-15,
        )
        and np.isclose(result_weights.sum(), 1.0, rtol=0.0, atol=1.0e-15)
    )
    gamma_included = result_frac.shape[0] > 0 and bool(
        np.any(np.all(np.isclose(result_frac, 0.0, atol=1.0e-14), axis=1))
    )
    if not kmesh_matches:
        failures.append("executed k mesh does not equal the complete cyclic character net")
    if not uniform_weights:
        failures.append("executed character weights are not complete and uniform")
    if not gamma_included:
        failures.append("executed character net does not include Gamma")

    diagnostics = getattr(result, "aiccm2026dev_b", None)
    if diagnostics is None:
        failures.append("result has no B-owned diagnostics")
    else:
        expected_diagnostics = {
            "mesh": mesh,
            "n_cyclic_cells": n_characters,
            "n_kpoints": n_characters,
            "backend": "ri",
            "electronic_method": "RHF",
            "use_diis": True,
            "diis_start_iter": 2,
            "diis_subspace_size": 8,
            "scf_accelerator": "EDIIS_DIIS",
            "damping": 0.0,
            "dynamic_damping": False,
            "fock_mixing": 0.0,
            "level_shift": 0.0,
            "smearing_temperature": 0.0,
            "physical_electron_count": int(meta["n_electrons"]),
            "effective_electron_count": int(meta["n_electrons"]),
            "ecp_total_ncore": 0,
        }
        for name, expected in expected_diagnostics.items():
            actual = getattr(diagnostics, name, None)
            if actual != expected:
                failures.append(
                    f"executed diagnostic {name}={actual!r}, expected {expected!r}"
                )
        invariant_limits = {
            "wigner_seitz_partition_error": 1.0e-12,
            "density_idempotency_error": 1.0e-7,
            "electron_count_error": 1.0e-7,
            "inverse_bloch_imaginary_residual": 1.0e-7,
            "effective_net_charge": 1.0e-12,
        }
        for name, limit in invariant_limits.items():
            number = _finite_number(getattr(diagnostics, name, None))
            if number is None or abs(number) > limit:
                failures.append(
                    f"diagnostic {name} is missing, nonfinite, or exceeds {limit:g}"
                )

    raw_fock_mixing = _finite_number(getattr(result, "fock_mixing", None))
    diagnostics_fock_mixing = _finite_number(
        getattr(diagnostics, "fock_mixing", None) if diagnostics is not None else None
    )
    if raw_fock_mixing != 0.0 or diagnostics_fock_mixing != 0.0:
        failures.append(
            "executed fock_mixing must be present as exactly 0.0 on both "
            "the backend result and B diagnostics"
        )
    resolved_gdf_method = getattr(result, "aiccm_resolved_gdf_method", None)
    resolved_rsgdf_ke = _finite_number(
        getattr(result, "aiccm_resolved_rsgdf_ke_cutoff", None)
    )
    if resolved_gdf_method != "rsgdf" or resolved_rsgdf_ke != 200.0:
        failures.append(
            "B selector-resolved forwarded settings do not confirm "
            "gdf_method='rsgdf' and rsgdf_ke_cutoff=200.0 Ha"
        )
    resolved_aux = getattr(result, "aux_basis_name", None)
    if resolved_aux != aux_basis or resolved_aux != EXPECTED_AUX:
        failures.append(
            f"executed auxiliary basis {resolved_aux!r} does not equal {EXPECTED_AUX!r}"
        )

    routing_lines = [
        line.strip()
        for line in log_text.splitlines()
        if line.strip() == ROUTING_EVENT
    ]
    initial_guess_lines = [
        line.strip()
        for line in log_text.splitlines()
        if line.strip() == INITIAL_GUESS_EVENT
    ]
    if len(routing_lines) != 1:
        failures.append("the unique executed exxdiv='ewald' routing event is missing")
    if len(initial_guess_lines) != 1:
        failures.append("the unique executed HCORE initial-guess event is missing")

    madelung = _recomputed_bvk_madelung(system, expected_bvk)
    if madelung is None or madelung <= 0.0:
        failures.append(
            "separately recomputed BvK Madelung coefficient is not finite and positive"
        )

    setting_expectations = {
        "linear_dep_threshold": SOURCE_RESOLVED_PINS[
            "ao_linear_dependence_threshold"
        ],
        "gdf_linear_dep_threshold": SOURCE_RESOLVED_PINS[
            "auxiliary_metric_linear_dependence_threshold"
        ],
        "conv_tol_grad": SOURCE_RESOLVED_PINS["conv_tol_grad"],
    }
    executed_thresholds: dict[str, object] = {}
    for name, expected in setting_expectations.items():
        values = _setting_values(log_text, name)
        parsed = _finite_numeric_text(values[0]) if len(values) == 1 else None
        executed_thresholds[name] = {
            "value": parsed,
            "evidence": "executed-progress-event",
        }
        if parsed != expected:
            failures.append(
                f"executed progress value for {name} is {parsed!r}, expected {expected!r}"
            )

    total = _finite_number(getattr(result, "energy", None))
    electronic = _finite_number(getattr(result, "e_electronic", None))
    nuclear = _finite_number(getattr(result, "e_nuclear", None))
    recomputed_nuclear = _finite_number(vq.ewald_nuclear_repulsion(system))
    energy_finite = all(
        value is not None for value in (total, electronic, nuclear, recomputed_nuclear)
    )
    decomposition_matches = bool(
        energy_finite
        and abs(float(total) - (float(electronic) + float(nuclear))) <= 1.0e-12
    )
    nuclear_matches = bool(
        energy_finite
        and abs(float(nuclear) - float(recomputed_nuclear)) <= 1.0e-12
    )
    if not energy_finite:
        failures.append("SCF energy components are missing or nonfinite")
    if not decomposition_matches:
        failures.append("total energy does not equal electronic plus nuclear energy")
    if not nuclear_matches:
        failures.append(
            "executed nuclear scalar does not equal the shared 3D Ewald helper"
        )

    converged = getattr(result, "converged", None) is True
    iterations = getattr(result, "n_iter", None)
    trace = list(getattr(result, "scf_trace", []) or [])
    final_delta = _finite_number(
        getattr(diagnostics, "final_scf_delta_e_ha", None)
        if diagnostics is not None
        else None
    )
    final_grad = _finite_number(
        getattr(diagnostics, "final_scf_grad_norm", None)
        if diagnostics is not None
        else None
    )
    if converged and (final_delta is None or abs(final_delta) > 1.0e-10):
        failures.append("converged result does not carry a final |delta E| <= 1e-10 Ha")
    if converged and (
        final_grad is None
        or final_grad < 0.0
        or final_grad > SOURCE_RESOLVED_PINS["conv_tol_grad"]
    ):
        failures.append(
            "converged result does not carry a final gradient below the "
            "source-resolved SCF threshold"
        )
    if diagnostics is not None and getattr(diagnostics, "scf_trace_length", None) != len(trace):
        failures.append("B diagnostic SCF trace length contradicts the result trace")
    if type(iterations) is not int or iterations < 1:
        failures.append("SCF iteration count is missing or invalid")
    elif iterations != len(trace):
        failures.append("SCF iteration count contradicts the result trace length")

    evidence = {
        "executed_input": {
            "runtime_backend": _runtime_backend(result),
            "aux_basis_name": resolved_aux,
            "gdf_method": {
                "value": resolved_gdf_method,
                "evidence": "B-selector-resolved-forwarded-setting",
            },
            "rsgdf_ke_cutoff_ha": {
                "value": resolved_rsgdf_ke,
                "evidence": "B-selector-resolved-forwarded-setting",
            },
            "fock_mixing_result": raw_fock_mixing,
            "fock_mixing_diagnostics": diagnostics_fock_mixing,
            "converger": {
                "use_diis": getattr(diagnostics, "use_diis", None),
                "scf_accelerator": getattr(
                    diagnostics, "scf_accelerator", None
                ),
                "diis_start_iter": getattr(
                    diagnostics, "diis_start_iter", None
                ),
                "diis_subspace_size": getattr(
                    diagnostics, "diis_subspace_size", None
                ),
            },
            "static_scf_controls": {
                "damping": getattr(diagnostics, "damping", None),
                "dynamic_damping": getattr(
                    diagnostics, "dynamic_damping", None
                ),
                "level_shift": getattr(diagnostics, "level_shift", None),
                "smearing_temperature_ha": getattr(
                    diagnostics, "smearing_temperature", None
                ),
            },
            "initial_guess": {
                "semantics": "per-k-hcore-diagonalisation",
                "evidence": "executed-progress-event",
                "event": initial_guess_lines[0] if len(initial_guess_lines) == 1 else None,
            },
            "linear_dependence_thresholds": executed_thresholds,
            "source_resolved_thresholds": source_thresholds,
        },
        "finite_torus_convention": convention,
        "exact_exchange_assembly": assembly,
        "two_electron_support": support,
        "kmesh_audit": {
            "n_expected_characters": n_characters,
            "n_executed_characters": int(result_frac.shape[0]),
            "complete_character_net_matches": bool(kmesh_matches),
            "fractional_residue_bijection": bool(residue_bijection),
            "cartesian_reciprocal_matches": bool(cartesian_matches),
            "uniform_weights": bool(uniform_weights),
            "gamma_included": bool(gamma_included),
            "bvk_lattice_matches_a_times_diag_n": bool(bvk_matches),
        },
        "exchange_seam_audit": {
            "routing_event_count": len(routing_lines),
            "routing_event": routing_lines[0] if len(routing_lines) == 1 else None,
            "separately_recomputed_bvk_madelung_positive": bool(
                madelung is not None and madelung > 0.0
            ),
            "bvk_madelung_xi": madelung,
            "route_oracle_c_full": 1.0,
            "evidence_level": "dispatch-evidence-not-independent-kernel-proof",
        },
        "energy_audit": {
            "absolute_energy_in_campaign_json": False,
            "components_finite": bool(energy_finite),
            "total_decomposition_matches": decomposition_matches,
            "nuclear_matches_separately_recomputed_3d_ewald": nuclear_matches,
            "assembly_tolerance_ha": 1.0e-12,
        },
        "convergence": {
            "converged": converged,
            "iterations": iterations,
            "scf_trace_length": len(trace),
            "final_delta_e_ha": final_delta,
            "final_grad_norm": final_grad,
            "source_resolved_conv_tol_grad": SOURCE_RESOLVED_PINS[
                "conv_tol_grad"
            ],
        },
    }
    return evidence, failures


_SUCCESS_KEYS = {
    "schema",
    "spec",
    "stream",
    "selector",
    "system",
    "system_label",
    "declared_model",
    "dim",
    "route",
    "method",
    "functional",
    "backend",
    "entry_point",
    "evidence_role",
    "quantitative_status",
    "comparison_status",
    "status",
    "basis",
    "aux_basis",
    "primitive_lattice_bohr",
    "atoms",
    "charge",
    "multiplicity",
    "physical_electrons_per_cell",
    "atoms_per_cell",
    "mesh",
    "requested_input",
    "executed_input",
    "finite_torus_convention",
    "exact_exchange_assembly",
    "two_electron_support",
    "kmesh_audit",
    "exchange_seam_audit",
    "energy_audit",
    "convergence",
    "walltime_s",
    "peak_rss_mb",
    "provenance",
    "source_binding",
    "validation_receipt",
    "vq_target_host",
    "vq_job_id",
    "scheduler_job_id",
    "utc_started",
    "utc_finished",
    "pin_violations",
}


def _campaign_evidence_failures(
    record: dict[str, object],
    key: str,
) -> list[str]:
    """Revalidate nested evidence before a record can act as a receipt."""

    failures: list[str] = []
    meta = SYSTEMS[key]
    mesh = tuple(int(value) for value in meta["mesh"])
    primitive = np.asarray(meta["vectors"], dtype=float).T
    expected_bvk = (primitive @ np.diag(np.asarray(mesh, dtype=float))).tolist()

    convention = record.get("finite_torus_convention")
    expected_convention = {
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
        "bvk_madelung_supercell_lattice_bohr": expected_bvk,
        "lattice_vector_convention": "columns",
        "orbital_energy_convention": (
            "HF/Kohn-Sham eigenvalues include the declared exchange_q0 seam"
        ),
    }
    if not _json_exact(convention, expected_convention):
        failures.append("campaign finite-torus convention contradicts D101")

    expected_source_thresholds = {
        "linear_dep_threshold": {
            "value": SOURCE_RESOLVED_PINS["ao_linear_dependence_threshold"],
            "evidence": "source-resolved-default+executed-progress-event",
            "source": (
                "vibeqc.periodic_k_gdf.run_krhf_periodic_gdf:"
                "linear_dep_threshold"
            ),
        },
        "gdf_linear_dep_threshold": {
            "value": SOURCE_RESOLVED_PINS[
                "auxiliary_metric_linear_dependence_threshold"
            ],
            "evidence": "source-resolved-default+executed-progress-event",
            "source": (
                "vibeqc.periodic_k_gdf.run_krhf_periodic_gdf:"
                "gdf_linear_dep_threshold"
            ),
        },
        "conv_tol_grad": {
            "value": SOURCE_RESOLVED_PINS["conv_tol_grad"],
            "evidence": (
                "source-resolved-option-default+executed-progress-event"
            ),
            "source": "vibeqc._vibeqc_core.PeriodicRHFOptions:conv_tol_grad",
        },
    }
    expected_executed_thresholds = {
        "linear_dep_threshold": {
            "value": SOURCE_RESOLVED_PINS["ao_linear_dependence_threshold"],
            "evidence": "executed-progress-event",
        },
        "gdf_linear_dep_threshold": {
            "value": SOURCE_RESOLVED_PINS[
                "auxiliary_metric_linear_dependence_threshold"
            ],
            "evidence": "executed-progress-event",
        },
        "conv_tol_grad": {
            "value": SOURCE_RESOLVED_PINS["conv_tol_grad"],
            "evidence": "executed-progress-event",
        },
    }
    executed = record.get("executed_input")
    support = record.get("two_electron_support")
    runtime_backend = (
        support.get("runtime_backend") if isinstance(support, dict) else None
    )
    if not isinstance(executed, dict) or set(executed) != {
        "runtime_backend",
        "aux_basis_name",
        "gdf_method",
        "rsgdf_ke_cutoff_ha",
        "fock_mixing_result",
        "fock_mixing_diagnostics",
        "converger",
        "static_scf_controls",
        "initial_guess",
        "linear_dependence_thresholds",
        "source_resolved_thresholds",
    }:
        failures.append("campaign executed_input does not have the exact D101 keys")
    else:
        initial_guess = executed.get("initial_guess")
        initial_guess_ok = (
            isinstance(initial_guess, dict)
            and set(initial_guess) == {"semantics", "evidence", "event"}
            and initial_guess.get("semantics") == "per-k-hcore-diagonalisation"
            and initial_guess.get("evidence") == "executed-progress-event"
            and initial_guess.get("event") == INITIAL_GUESS_EVENT
        )
        converger = executed.get("converger")
        static_controls = executed.get("static_scf_controls")
        if (
            executed.get("runtime_backend") != runtime_backend
            or executed.get("aux_basis_name") != EXPECTED_AUX
            or not _json_exact(
                executed.get("gdf_method"),
                {
                    "value": "rsgdf",
                    "evidence": "B-selector-resolved-forwarded-setting",
                },
            )
            or not _json_exact(
                executed.get("rsgdf_ke_cutoff_ha"),
                {
                    "value": 200.0,
                    "evidence": "B-selector-resolved-forwarded-setting",
                },
            )
            or not _json_exact(executed.get("fock_mixing_result"), 0.0)
            or not _json_exact(executed.get("fock_mixing_diagnostics"), 0.0)
            or not _json_exact(
                converger,
                {
                    "use_diis": True,
                    "scf_accelerator": "EDIIS_DIIS",
                    "diis_start_iter": 2,
                    "diis_subspace_size": 8,
                },
            )
            or not _json_exact(
                static_controls,
                {
                    "damping": 0.0,
                    "dynamic_damping": False,
                    "level_shift": 0.0,
                    "smearing_temperature_ha": 0.0,
                },
            )
            or not initial_guess_ok
            or not _json_exact(
                executed.get("linear_dependence_thresholds"),
                expected_executed_thresholds,
            )
            or not _json_exact(
                executed.get("source_resolved_thresholds"),
                expected_source_thresholds,
            )
        ):
            failures.append("campaign executed_input contradicts the executed pins")

    kmesh = record.get("kmesh_audit")
    expected_characters = prod(mesh)
    if not isinstance(kmesh, dict) or set(kmesh) != {
        "n_expected_characters",
        "n_executed_characters",
        "complete_character_net_matches",
        "fractional_residue_bijection",
        "cartesian_reciprocal_matches",
        "uniform_weights",
        "gamma_included",
        "bvk_lattice_matches_a_times_diag_n",
    }:
        failures.append("campaign kmesh_audit does not have the exact D101 keys")
    elif (
        not _json_exact(kmesh.get("n_expected_characters"), expected_characters)
        or not _json_exact(
            kmesh.get("n_executed_characters"), expected_characters
        )
        or any(
            kmesh.get(flag) is not True
            for flag in (
                "complete_character_net_matches",
                "fractional_residue_bijection",
                "cartesian_reciprocal_matches",
                "uniform_weights",
                "gamma_included",
                "bvk_lattice_matches_a_times_diag_n",
            )
        )
    ):
        failures.append("campaign kmesh_audit contradicts the complete character net")

    seam = record.get("exchange_seam_audit")
    recomputed_madelung = _recomputed_bvk_madelung(
        build_system(key),
        expected_bvk,
    )
    if not isinstance(seam, dict) or set(seam) != {
        "routing_event_count",
        "routing_event",
        "separately_recomputed_bvk_madelung_positive",
        "bvk_madelung_xi",
        "route_oracle_c_full",
        "evidence_level",
    }:
        failures.append("campaign exchange_seam_audit has the wrong keys")
    else:
        serialized_madelung = _finite_number(seam.get("bvk_madelung_xi"))
        seam_invalid = (
            not _json_exact(seam.get("routing_event_count"), 1)
            or seam.get("routing_event") != ROUTING_EVENT
            or seam.get("separately_recomputed_bvk_madelung_positive") is not True
            or type(seam.get("bvk_madelung_xi")) is not float
            or serialized_madelung is None
            or serialized_madelung <= 0.0
            or recomputed_madelung is None
            or abs(serialized_madelung - recomputed_madelung) > 1.0e-12
            or not _json_exact(seam.get("route_oracle_c_full"), 1.0)
            or seam.get("evidence_level")
            != "dispatch-evidence-not-independent-kernel-proof"
        )
        if seam_invalid:
            failures.append(
                "campaign exchange-seam evidence contradicts D83/D101"
            )

    convergence = record.get("convergence")
    if not isinstance(convergence, dict) or set(convergence) != {
        "converged",
        "iterations",
        "scf_trace_length",
        "final_delta_e_ha",
        "final_grad_norm",
        "source_resolved_conv_tol_grad",
    }:
        failures.append("campaign convergence does not have the exact D101 keys")
    else:
        iterations = convergence.get("iterations")
        trace_length = convergence.get("scf_trace_length")
        final_delta = _finite_number(convergence.get("final_delta_e_ha"))
        final_grad = _finite_number(convergence.get("final_grad_norm"))
        if (
            convergence.get("converged") is not True
            or type(iterations) is not int
            or type(trace_length) is not int
            or not 1 <= iterations <= int(PINS["max_iter"])
            or trace_length != iterations
            or final_delta is None
            or abs(final_delta) > float(PINS["conv_tol_energy_ha"])
            or final_grad is None
            or final_grad < 0.0
            or final_grad > SOURCE_RESOLVED_PINS["conv_tol_grad"]
            or convergence.get("source_resolved_conv_tol_grad")
            != SOURCE_RESOLVED_PINS["conv_tol_grad"]
        ):
            failures.append("campaign convergence does not satisfy the pinned gate")

    walltime = _finite_number(record.get("walltime_s"))
    peak_rss = _finite_number(record.get("peak_rss_mb"))
    if walltime is None or walltime < 0.0 or peak_rss is None or peak_rss <= 0.0:
        failures.append("campaign runtime measurements are missing or invalid")
    if not _is_vq_job_id(record.get("vq_job_id")):
        failures.append("campaign record lacks a canonical vq_job_id")
    if not _is_slurm_job_id(record.get("scheduler_job_id")):
        failures.append("campaign record lacks a canonical scheduler_job_id")
    receipt = record.get("validation_receipt")
    if key == "3d":
        if receipt is not None:
            failures.append("the 3d validation record must have no prior receipt")
    elif (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema",
            "sha256",
            "system",
            "vq_job_id",
            "scheduler_job_id",
            "probe_attestation_id",
        }
        or receipt.get("schema") != SCHEMA
        or receipt.get("system") != "3d"
        or not _is_sha256(receipt.get("sha256"))
        or not _is_vq_job_id(receipt.get("vq_job_id"))
        or not _is_slurm_job_id(receipt.get("scheduler_job_id"))
        or not _is_sha256(receipt.get("probe_attestation_id"))
    ):
        failures.append(
            "campaign record lacks its exact contract-valid nonquantitative "
            "3d receipt binding"
        )
    else:
        if (
            receipt.get("vq_job_id") == record.get("vq_job_id")
            or receipt.get("scheduler_job_id") == record.get("scheduler_job_id")
        ):
            failures.append(
                "campaign 3d receipt does not identify a distinct parent job"
            )
    timestamps: dict[str, datetime] = {}
    for name in ("utc_started", "utc_finished"):
        value = record.get(name)
        if not _is_aware_iso_timestamp(value):
            failures.append(
                f"campaign {name} is not a canonical timezone-aware timestamp"
            )
        else:
            assert isinstance(value, str)
            timestamps[name] = datetime.fromisoformat(value)
    if (
        set(timestamps) == {"utc_started", "utc_finished"}
        and timestamps["utc_finished"] < timestamps["utc_started"]
    ):
        failures.append("campaign finish timestamp precedes its start timestamp")
    return failures


def campaign_record_failures(
    record: object,
    *,
    expected_source_sha: str,
    required_system: str | None = None,
) -> list[str]:
    """Validate one serialized D101 record without exposing its SCF energy.

    For child records this validates the stored receipt shape and cross-fields.
    :func:`audit_cmp_b.audit_path` additionally requires the referenced 3D file
    and recomputes its exact byte binding.
    """

    failures: list[str] = []
    if not isinstance(record, dict) or set(record) != _SUCCESS_KEYS:
        return ["campaign record does not have the exact D101 top-level keys"]
    expected_scalars = {
        "schema": SCHEMA,
        "spec": SPEC,
        "stream": STREAM,
        "selector": SELECTOR,
        "route": ROUTE,
        "method": "RHF",
        "functional": None,
        "backend": "ri",
        "entry_point": ENTRY_POINT,
        "evidence_role": EVIDENCE_ROLE,
        "quantitative_status": QUANTITATIVE_STATUS,
        "comparison_status": COMPARISON_STATUS,
        "status": "ok",
        "basis": BASIS,
        "aux_basis": EXPECTED_AUX,
        "dim": 3,
        "charge": 0,
        "multiplicity": 1,
        "vq_target_host": site_settings.comparison_host(),
    }
    for name, expected in expected_scalars.items():
        if not _json_exact(record.get(name), expected):
            failures.append(f"campaign record {name} contradicts D101")
    key = record.get("system")
    if not isinstance(key, str) or key not in SYSTEMS:
        failures.append("campaign record has an unknown system")
        return failures
    if required_system is not None and key != required_system:
        failures.append(f"campaign validation receipt must be system={required_system!r}")
    meta = SYSTEMS[key]
    exact_inputs = {
        "system_label": meta["label"],
        "declared_model": meta["declared_model"],
        "primitive_lattice_bohr": np.asarray(meta["vectors"], dtype=float).T.tolist(),
        "atoms": [[int(z), list(position)] for z, position in meta["atoms"]],
        "physical_electrons_per_cell": int(meta["n_electrons"]),
        "atoms_per_cell": len(meta["atoms"]),
        "mesh": list(meta["mesh"]),
        "requested_input": _requested_input(
            tuple(int(value) for value in meta["mesh"]),
            EXPECTED_AUX,
        ),
    }
    for name, expected in exact_inputs.items():
        if not _json_exact(record.get(name), expected):
            failures.append(f"campaign record {name} contradicts the pinned input")
    if record.get("pin_violations") != []:
        failures.append("campaign record contains pin violations")
    if "energy_per_cell_ha" in record or "energy_per_atom_ha" in record:
        failures.append("D93-not-qualified campaign record serialized an absolute energy")
    energy_audit = record.get("energy_audit")
    if not _json_exact(
        energy_audit,
        {
            "absolute_energy_in_campaign_json": False,
            "components_finite": True,
            "total_decomposition_matches": True,
            "nuclear_matches_separately_recomputed_3d_ewald": True,
            "assembly_tolerance_ha": 1.0e-12,
        },
    ):
        failures.append("campaign energy audit is incomplete or contradictory")
    support_failure = two_electron_support_schema_failure(
        record.get("two_electron_support"),
        ROUTE,
    )
    if support_failure:
        failures.append(f"campaign support payload: {support_failure}")
    elif two_electron_support_reportability_failure(
        record.get("two_electron_support"),
        ROUTE,
    ) is None:
        failures.append("campaign fitted support improperly claims reportability")
    support = record.get("two_electron_support")
    if isinstance(support, dict):
        expected_support = {
            "schema": TWO_ELECTRON_SUPPORT_SCHEMA,
            "qualification": "not-qualified",
            "reason": TWO_ELECTRON_SUPPORT_REASON,
            "backend": "ri",
            "runtime_backend": support.get("runtime_backend"),
            "direct": None,
            "fitted": {
                "gdf_method": "rsgdf",
                "rsgdf_base_ke_cutoff_ha": 200.0,
                "rsgdf_tail_ke_cutoff_ha": None,
                "mdf_ke_cutoff_ha": None,
                "cosx_exchange_support": "not-applicable",
                "correlation_support": "not-applicable",
            },
        }
        if not _json_exact(support, expected_support):
            failures.append("campaign fitted support contradicts the exact D101 record")
    assembly_failure = exact_exchange_assembly_failure(
        record.get("exact_exchange_assembly"),
        ROUTE,
    )
    if assembly_failure:
        failures.append(f"campaign exact-exchange payload: {assembly_failure}")
    failures.extend(_campaign_evidence_failures(record, key))
    provenance_failure = _serialized_provenance_failure(
        record.get("provenance"),
        expected_source_sha,
    )
    if provenance_failure:
        failures.append(provenance_failure)
    source_binding_failure = _source_binding_failure(
        record.get("source_binding"),
        expected_source_sha,
    )
    if source_binding_failure:
        failures.append(source_binding_failure)
    cross_binding_failure = _source_payload_binding_failure(
        record.get("source_binding"),
        record.get("provenance"),
    )
    if cross_binding_failure:
        failures.append(cross_binding_failure)
    return failures


def _validation_receipt(
    path: Path | None,
    expected_source_sha: str,
    current_provenance: dict[str, object],
) -> tuple[dict[str, object] | None, list[str]]:
    if path is None:
        return None, [
            "1d/2d campaign runs require a fetched contract-valid "
            "nonquantitative 3d validation record"
        ]
    try:
        receipt_bytes = path.read_bytes()
        record = _strict_json_loads(receipt_bytes)
    except Exception:  # noqa: BLE001 - receipt parsing must fail closed
        return None, ["cannot read or decode the 3d validation record"]
    failures = campaign_record_failures(
        record,
        expected_source_sha=expected_source_sha,
        required_system="3d",
    )
    if not isinstance(record, dict):
        return None, failures
    previous_provenance = record.get("provenance")
    previous = (
        previous_provenance.get("probe_attestation")
        if isinstance(previous_provenance, dict)
        else None
    )
    current = current_provenance.get("probe_attestation")
    if not isinstance(previous, dict) or not isinstance(current, dict):
        failures.append("3d validation receipt lacks comparable D77 identity")
        return None, failures
    for field in (
        "source_commit",
        "vibeqc_version",
        "core_build_id",
        "attestor_script_id",
        "native_library_versions",
        "producer_payload_id",
    ):
        if not _json_exact(previous.get(field), current.get(field)):
            failures.append(f"3d validation receipt identity differs for {field}")
    if failures:
        return None, failures
    return {
        "schema": SCHEMA,
        "sha256": _sha256_bytes(receipt_bytes),
        "system": "3d",
        "vq_job_id": record["vq_job_id"],
        "scheduler_job_id": record["scheduler_job_id"],
        "probe_attestation_id": record["provenance"]["probe_attestation_id"],
    }, []


def _validation_receipt_failures(
    path: Path | None,
    expected_source_sha: str,
    current_provenance: dict[str, object],
) -> list[str]:
    """Compatibility wrapper used by the focused contract tests."""

    _binding, failures = _validation_receipt(
        path,
        expected_source_sha,
        current_provenance,
    )
    return failures


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        value /= 1024.0 * 1024.0
    else:
        value /= 1024.0
    return value


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(rendered, end="")


def _campaign_stem(key: str) -> str:
    """Return a name excluded from the generic quantitative fleet globs."""

    if key not in SYSTEMS:
        raise KeyError(key)
    return f"d101-cmp1d2d3d-{key}-chi-rhf-ri"


def _failure_record(
    *,
    key: str,
    status: str,
    failures: list[str],
    provenance: dict[str, object] | None,
    source_binding: dict[str, object] | None,
    expected_source_sha: str,
    vq_target_host: str,
) -> dict[str, object]:
    return {
        "schema": FAILURE_SCHEMA,
        "spec": SPEC,
        "stream": STREAM,
        "selector": SELECTOR,
        "system": key,
        "route": ROUTE,
        "evidence_role": EVIDENCE_ROLE,
        "quantitative_status": QUANTITATIVE_STATUS,
        "comparison_status": COMPARISON_STATUS,
        "status": status,
        "absolute_energy_in_campaign_json": False,
        "pin_violations": failures,
        "provenance": provenance,
        "source_binding": source_binding,
        "expected_source_commit": expected_source_sha,
        "vq_target_host": vq_target_host,
        "vq_job_id": os.environ.get("VQ_JOB_ID", "not-recorded"),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID", "not-recorded"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=sorted(SYSTEMS))
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument(
        "--vq-host",
        default=os.environ.get("AICCM_CAMPAIGN_VQ_HOST"),
        help="required operator assertion; must match the external site profile",
    )
    parser.add_argument(
        "--validation-record",
        type=Path,
        help=(
            "fetched contract-valid nonquantitative 3d JSON required before "
            "1d/2d"
        ),
    )
    parser.add_argument("--out", default=os.environ.get("VQ_WORKDIR", "."))
    args = parser.parse_args()

    expected_source_sha = args.expected_source_sha.strip().lower()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = _campaign_stem(args.system)
    result_path = out / f"{stem}.json"
    log_path = out / f"{stem}.scf.log"
    result_path.unlink(missing_ok=True)

    preflight_failures: list[str] = []
    try:
        site_settings.require_submission()
    except ValueError as error:
        preflight_failures.append(str(error))
    if not _is_full_commit(expected_source_sha):
        preflight_failures.append("--expected-source-sha must be one full 40-hex SHA")
    if not site_settings.comparison_host() or args.vq_host != site_settings.comparison_host():
        preflight_failures.append("--vq-host must match comparison_host in the external site profile")
    if not _is_vq_job_id(os.environ.get("VQ_JOB_ID")):
        preflight_failures.append(
            "D101 requires the canonical 12-hex VQ_JOB_ID from vq"
        )
    if not os.environ.get("VQ_WORKDIR", "").strip():
        preflight_failures.append("D101 must execute inside a vq workdir")
    if not _is_slurm_job_id(os.environ.get("SLURM_JOB_ID")):
        preflight_failures.append(
            "the operator-asserted managed-host target lacks numeric SLURM_JOB_ID evidence"
        )
    if preflight_failures:
        _write_json(
            result_path,
            _failure_record(
                key=args.system,
                status="input-failed",
                failures=preflight_failures,
                provenance=None,
                source_binding=None,
                expected_source_sha=expected_source_sha,
                vq_target_host=args.vq_host or "not-recorded",
            ),
        )
        raise SystemExit(3)

    source_binding: dict[str, object] | None = None
    provenance: dict[str, object] | None = None
    try:
        source_binding, binding_failures = _source_binding(expected_source_sha)
        provenance, provenance_failures = _provenance(expected_source_sha)
    except Exception as exc:  # noqa: BLE001 - attestation must serialize closed
        _write_json(
            result_path,
            _failure_record(
                key=args.system,
                status="attestation-error",
                failures=[f"preflight attestation raised {type(exc).__name__}"],
                provenance=provenance,
                source_binding=source_binding,
                expected_source_sha=expected_source_sha,
                vq_target_host=args.vq_host,
            ),
        )
        raise SystemExit(3) from exc
    assert source_binding is not None
    assert provenance is not None
    preflight_failures.extend(binding_failures)
    preflight_failures.extend(provenance_failures)
    validation_receipt = None
    if args.system != "3d":
        validation_receipt, receipt_failures = _validation_receipt(
            args.validation_record,
            expected_source_sha,
            provenance,
        )
        preflight_failures.extend(receipt_failures)
    elif args.validation_record is not None:
        preflight_failures.append("the 3d validation run must not consume a prior receipt")
    if preflight_failures:
        _write_json(
            result_path,
            _failure_record(
                key=args.system,
                status="attestation-failed",
                failures=preflight_failures,
                provenance=provenance,
                source_binding=source_binding,
                expected_source_sha=expected_source_sha,
                vq_target_host=args.vq_host,
            ),
        )
        raise SystemExit(3)

    meta = SYSTEMS[args.system]
    mesh = tuple(int(value) for value in meta["mesh"])
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()
    try:
        source_thresholds, threshold_failures = _source_resolved_thresholds()
        aux_basis = str(default_aux_for(BASIS))
        if aux_basis != EXPECTED_AUX:
            threshold_failures.append(
                f"default_aux_for({BASIS!r})={aux_basis!r}, "
                f"expected {EXPECTED_AUX!r}"
            )
        if threshold_failures:
            raise RuntimeError("; ".join(threshold_failures))
        system = build_system(args.system)
        basis = BasisSet(system.unit_cell_molecule(), BASIS)
        progress = ProgressLogger(log_path=log_path, verbose=5)
        output_stem = out / f"{stem}-runner"
        result = vq.run_periodic_job(
            system,
            basis,
            **_runner_kwargs(output_stem, mesh, aux_basis, progress),
        )
        walltime = time.perf_counter() - started
        finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log_text = (
            log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        )
        evidence, pin_failures = _audit_result(
            args.system,
            system,
            result,
            log_text,
            aux_basis,
            source_thresholds,
        )
    except Exception as exc:  # noqa: BLE001 - campaign errors need JSON evidence
        exception_failures = [f"SCF execution raised {type(exc).__name__}"]
        try:
            exception_failures.extend(
                _finalize_provenance(provenance, expected_source_sha)
            )
        except Exception as final_exc:  # noqa: BLE001 - preserve first failure
            exception_failures.append(
                f"provenance finalization raised {type(final_exc).__name__}"
            )
        try:
            final_binding, final_binding_failures = _source_binding(
                expected_source_sha
            )
            source_binding = final_binding
            exception_failures.extend(final_binding_failures)
        except Exception as binding_exc:  # noqa: BLE001 - fail closed
            exception_failures.append(
                f"final source binding raised {type(binding_exc).__name__}"
            )
        _write_json(
            result_path,
            _failure_record(
                key=args.system,
                status=(
                    "unexpected-not-implemented"
                    if isinstance(exc, NotImplementedError)
                    else "execution-error"
                ),
                failures=exception_failures,
                provenance=provenance,
                source_binding=source_binding,
                expected_source_sha=expected_source_sha,
                vq_target_host=args.vq_host,
            ),
        )
        raise SystemExit(4) from exc

    try:
        pin_failures.extend(
            _finalize_provenance(provenance, expected_source_sha)
        )
        final_binding, final_binding_failures = _source_binding(
            expected_source_sha
        )
        pin_failures.extend(final_binding_failures)
        if final_binding != source_binding:
            pin_failures.append(
                "source-to-payload binding changed during computation"
            )
        source_binding = final_binding
    except Exception as exc:  # noqa: BLE001 - postflight must serialize failure
        _write_json(
            result_path,
            _failure_record(
                key=args.system,
                status="postflight-error",
                failures=[f"postflight raised {type(exc).__name__}"],
                provenance=provenance,
                source_binding=source_binding,
                expected_source_sha=expected_source_sha,
                vq_target_host=args.vq_host,
            ),
        )
        raise SystemExit(4) from exc
    if pin_failures:
        _write_json(
            result_path,
            _failure_record(
                key=args.system,
                status="stop-and-report",
                failures=pin_failures,
                provenance=provenance,
                source_binding=source_binding,
                expected_source_sha=expected_source_sha,
                vq_target_host=args.vq_host,
            ),
        )
        raise SystemExit(4)

    record: dict[str, object] = {
        "schema": SCHEMA,
        "spec": SPEC,
        "stream": STREAM,
        "selector": SELECTOR,
        "system": args.system,
        "system_label": meta["label"],
        "declared_model": meta["declared_model"],
        "dim": 3,
        "route": ROUTE,
        "method": "RHF",
        "functional": None,
        "backend": "ri",
        "entry_point": ENTRY_POINT,
        "evidence_role": EVIDENCE_ROLE,
        "quantitative_status": QUANTITATIVE_STATUS,
        "comparison_status": COMPARISON_STATUS,
        "status": "ok",
        "basis": BASIS,
        "aux_basis": aux_basis,
        "primitive_lattice_bohr": np.asarray(system.lattice, dtype=float).tolist(),
        "atoms": [[int(z), list(position)] for z, position in meta["atoms"]],
        "charge": 0,
        "multiplicity": 1,
        "physical_electrons_per_cell": int(meta["n_electrons"]),
        "atoms_per_cell": len(meta["atoms"]),
        "mesh": list(mesh),
        "requested_input": _requested_input(mesh, aux_basis),
        **evidence,
        "walltime_s": walltime,
        "peak_rss_mb": _peak_rss_mb(),
        "provenance": provenance,
        "source_binding": source_binding,
        "validation_receipt": validation_receipt,
        "vq_target_host": args.vq_host,
        "vq_job_id": os.environ.get("VQ_JOB_ID", "not-recorded"),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID", "not-recorded"),
        "utc_started": started_utc,
        "utc_finished": finished_utc,
        "pin_violations": [],
    }
    validation_failures = campaign_record_failures(
        record,
        expected_source_sha=expected_source_sha,
        required_system=args.system,
    )
    if validation_failures:
        _write_json(
            result_path,
            _failure_record(
                key=args.system,
                status="stop-and-report",
                failures=validation_failures,
                provenance=provenance,
                source_binding=source_binding,
                expected_source_sha=expected_source_sha,
                vq_target_host=args.vq_host,
            ),
        )
        raise SystemExit(4)
    _write_json(result_path, record)


if __name__ == "__main__":
    main()
