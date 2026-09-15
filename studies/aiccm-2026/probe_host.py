"""Host-readiness probe for the AICCM-2026 benchmark campaign.

Run this on a compute host BEFORE dispatching a batch there. It provides a
cached preflight answer to two questions:

1. **Does the host's vibeqc export the routines the A and B producers need?**
   The probe covers the public and internal APIs exercised across both runners.
   Guessing those paths is how an earlier revision reported false "missing"
   symbols on compute-small.

2. **Is the host's compiled core new enough to be numerically trustworthy?**
   This is the question a symbol probe cannot ask. Symbols do not change when a
   C++ kernel silently returns wrong numbers.

   The 2026-07-10 case (``f8c213e8``, ``cpp/src/aopair_ft.cpp``): the general-L
   AO-pair FT mirrored ``FT_mu,nu(G) == FT_nu,mu(G)`` at ``k == 0`` without
   checking that every evaluation frequency is a reciprocal-lattice vector. That
   identity fails on the ``q = -k_bra`` shifted meshes ``build_lpq_bloch_native_fft``
   uses for each ``(k_bra != Gamma, k_ket = Gamma)`` multi-k exchange pair, where the
   ``mu <-> nu`` swap picks up ``exp(i q.R)`` phases on inter-cell terms. It corrupted
   those fits wherever inter-cell AO-pair overlap is non-negligible, which is every
   compact or ultra-diffuse basis on a tight cell. On rocksalt LiH/STO-3G at
   ``(2,1,1)`` it moved ``run_krhf_periodic_gdf`` by 2.04e-2 Ha/cell.

   The tell: on a fixed host the direct-torus fold cderi and the multi-k GDF must
   agree to ~4e-14 Ha/cell. On the affected stale build the mirror-insensitive
   direct route and corrupted GDF side disagree by 2.04e-2. This is a targeted
   regression discriminator, not independent validation of their shared per-q
   fit objects and one-electron primitives.

Usage:

    vq submit <host> --branch main --cpus 4 probe_host.py

Exit 0 means the probed process is campaign-ready. A producer still binds that
preflight to its own process: it checks the copied payload, runs a cheap
shifted-mesh native/Python canary, and reruns the full checks if deployment
churn changed the core-path digest. Non-zero means do not dispatch to the host.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import platform
import subprocess
import sys

import numpy as np
import vibeqc

PROBE_VERSION = "aiccm-host-probe/v2"
PRODUCER_ATTESTATION_VERSION = "aiccm-producer-attestation/v1"
PRODUCER_PAYLOAD_VERSION = "aiccm-producer-payload/v1"
LOADED_CORE_CHECK_VERSION = "aiccm-loaded-core-mirror/v1"

PREFLIGHT_ATTESTATION_FIELDS = {
    "probe_version",
    "probe_script_id",
    "probe_passed",
    "host",
    "vibeqc_version",
    "core_build_id",
    "source_commit",
    "source_clean",
}
NATIVE_LIBRARY_KEYS = (
    "libint",
    "libxc",
    "spglib",
    "libecpint",
    "fftw3",
    "blas",
)
LOADED_CORE_CHECK_FIELDS = {
    "version",
    "passed",
    "fixture",
    "n_ao",
    "max_l",
    "n_cells",
    "n_frequencies",
    "atol",
    "rtol",
    "max_abs_cxx_python",
    "max_scaled_error",
    "reference_transpose_asymmetry",
}
PRODUCER_ATTESTATION_FIELDS = {
    "attestation_version",
    "attestor_script_id",
    "probe_passed",
    "host",
    "vibeqc_version",
    "core_build_id",
    "source_commit",
    "source_clean",
    "native_library_versions",
    "producer_payload",
    "producer_payload_id",
    "preflight_attestation",
    "preflight_attestation_id",
    "current_core_attestation",
    "current_core_attestation_id",
    "core_changed_since_preflight",
    "loaded_core_check",
    "result_identity_stable",
}
CANARY_ATOL = 1.0e-12
CANARY_RTOL = 1.0e-9
CANARY_ASYMMETRY_MIN = 1.0e-3


def imported_source_state() -> tuple[str, bool]:
    """Return the Git identity of the checkout providing imported ``vibeqc``.

    Directory-submitted benchmark payloads contain their own copied scripts,
    so resolving Git from ``__file__`` in ``studies/aiccm-2026`` can attest the payload
    repository rather than the package that executed the calculation. Walk up
    from ``vibeqc.__file__`` instead; a wheel without a source checkout fails
    closed with an unknown commit and ``source_clean=False``.
    """

    location = getattr(vibeqc, "__file__", None)
    if not location:
        return "unknown", False
    package_file = pathlib.Path(location).resolve()
    root = next(
        (
            parent
            for parent in package_file.parents
            if (parent / ".git").exists()
        ),
        None,
    )
    if root is None:
        return "unknown", False
    try:
        relative_package = package_file.relative_to(root)
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", str(relative_package)],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 -- provenance must never break a run
        return "unknown", False
    return commit, not bool(status.strip())


def probe_attestation_id(value: object) -> str:
    """Return a canonical digest binding one emitted probe attestation."""

    if not isinstance(value, dict) or not value:
        return "unknown"
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | pathlib.Path) -> str:
    """Return a SHA256 for one regular file, or ``"unknown"``."""

    try:
        digest = hashlib.sha256()
        with pathlib.Path(path).resolve().open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except Exception:  # noqa: BLE001 -- provenance must fail closed
        return "unknown"


def probe_script_id() -> str:
    """Return the SHA256 of the executing probe implementation."""

    return file_sha256(__file__)


def core_build_id():
    """SHA256 of bytes at the imported core module's filesystem path.

    A stale ``.so`` returns wrong numbers rather than failing to import, so a version
    string or a commit hash cannot attest a host. This path digest is therefore one
    required part of the source/core/host/package/probe cache identity. It does not
    cryptographically identify bytes already mapped into the process; the producer's
    same-process numerical canary covers the loaded behavior, while a literal loaded
    image identity would require a native build ID or an OS mapping hash.
    """
    try:
        from vibeqc import _vibeqc_core

        so = pathlib.Path(getattr(_vibeqc_core, "__file__", "") or "")
        if not so.is_file():
            return "unknown"
        return file_sha256(so)
    except Exception:  # noqa: BLE001
        return "unknown"


def probe_identity() -> dict[str, object]:
    """Return the exact identity fields carried by a v2 preflight."""

    source_commit, source_clean = imported_source_state()
    return {
        "probe_version": PROBE_VERSION,
        "probe_script_id": probe_script_id(),
        "host": (platform.node() or "unknown").split(".")[0],
        "vibeqc_version": str(vibeqc.__version__),
        "core_build_id": core_build_id(),
        "source_commit": source_commit,
        "source_clean": source_clean,
    }


def native_library_versions() -> dict[str, str]:
    """Return stable linked-library identities for producer provenance."""

    try:
        from vibeqc.banner import library_versions

        versions = library_versions()
        return {key: str(versions.get(key, "unknown")) for key in NATIVE_LIBRARY_KEYS}
    except Exception:  # noqa: BLE001 -- unknown versions fail producer validation
        return {key: "unknown" for key in NATIVE_LIBRARY_KEYS}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _is_full_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_identity_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and value.casefold() not in {"", "unknown", "not-recorded"}
    )


def _valid_probe_identity(identity: object) -> bool:
    return (
        isinstance(identity, dict)
        and set(identity) == PREFLIGHT_ATTESTATION_FIELDS - {"probe_passed"}
        and identity.get("probe_version") == PROBE_VERSION
        and _is_sha256(identity.get("probe_script_id"))
        and _is_identity_text(identity.get("host"))
        and _is_identity_text(identity.get("vibeqc_version"))
        and _is_sha256(identity.get("core_build_id"))
        and _is_full_commit(identity.get("source_commit"))
        and identity.get("source_clean") is True
    )


def _valid_preflight(attestation: object) -> bool:
    return (
        isinstance(attestation, dict)
        and set(attestation) == PREFLIGHT_ATTESTATION_FIELDS
        and attestation.get("probe_passed") is True
        and _valid_probe_identity(
            {key: value for key, value in attestation.items() if key != "probe_passed"}
        )
    )


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _valid_native_library_versions(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(NATIVE_LIBRARY_KEYS)
        and all(
            isinstance(version, str)
            and version == version.strip()
            and version.casefold() not in {"", "unknown", "not-recorded"}
            for version in value.values()
        )
    )


def _valid_loaded_core_check(value: object) -> bool:
    """Validate the exact process-local shifted-mesh canary contract."""

    if not isinstance(value, dict) or set(value) != LOADED_CORE_CHECK_FIELDS:
        return False
    if (
        value.get("version") != LOADED_CORE_CHECK_VERSION
        or value.get("passed") is not True
        or value.get("fixture") != "lih-sto3g-shifted-mesh/v1"
        or type(value.get("n_ao")) is not int
        or value.get("n_ao") != 6
        or type(value.get("max_l")) is not int
        or value.get("max_l") != 1
        or type(value.get("n_cells")) is not int
        or value.get("n_cells") != 3
        or type(value.get("n_frequencies")) is not int
        or value.get("n_frequencies") != 3
    ):
        return False
    atol = _finite_float(value.get("atol"))
    rtol = _finite_float(value.get("rtol"))
    max_abs = _finite_float(value.get("max_abs_cxx_python"))
    max_scaled = _finite_float(value.get("max_scaled_error"))
    asymmetry = _finite_float(value.get("reference_transpose_asymmetry"))
    return (
        atol == CANARY_ATOL
        and rtol == CANARY_RTOL
        and max_abs is not None
        and max_abs >= 0.0
        and max_scaled is not None
        and 0.0 <= max_scaled <= 1.0
        and asymmetry is not None
        and asymmetry > CANARY_ASYMMETRY_MIN
    )


def _atomic_write_json(path: str | pathlib.Path, value: dict[str, object]) -> None:
    """Atomically replace one JSON file so concurrent probes never see a prefix."""

    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

# Public and internal APIs exercised by the host probe and benchmark producers.
NEEDED = {
    "vibeqc": [
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
        "run_periodic_job",
        "scf_accelerator_from_string",
        "write_cube_density",
        "write_cube_mo",
        "write_molden",
    ],
    "vibeqc.dlpno.ccsd_local_solver": ["LocalCCSDOptions"],
    "vibeqc.dlpno.mp2": ["DLPNOMP2Options"],
    "vibeqc.output.formats.qvf": ["qvf_wf_data", "write_qvf"],
    "vibeqc.output.plan": ["OutputPlan"],
    "vibeqc.periodic.ccm": [
        "CCMSystem",
        "ccm_dipole",
        "ccm_dlpno_ccsd_coupled",
        "ccm_dlpno_mp2",
        "ccm_homo_lumo_gap",
        "ccm_lowdin_charges",
        "ccm_mulliken_charges",
        "ccm_numerical_gradient",
        "run_ccm_uccsd",
    ],
    "vibeqc.periodic.ccm.ccsd": ["run_ccm_ccsd"],
    "vibeqc.periodic.ccm.dft": ["run_ccm_rks", "run_ccm_uks"],
    "vibeqc.periodic.ccm.direct": ["run_ccm_rhf_direct"],
    "vibeqc.periodic.ccm.dlpno": ["ccm_pao", "pao_occupied_orthogonality"],
    "vibeqc.periodic.ccm.integrals": ["ccm_kinetic", "ccm_overlap"],
    "vibeqc.periodic.ccm.localize": [
        "localise_ccm",
        "localization_density_residual",
    ],
    "vibeqc.periodic.ccm.lowd_four_center": ["required_g_max"],
    "vibeqc.periodic.ccm.mp2": ["run_ccm_mp2"],
    "vibeqc.periodic.ccm.neutral": [
        "ccm_eri_neutral",
        "ccm_neutral_cderi",
        "ccm_neutral_cderi_fold",
    ],
    "vibeqc.periodic.ccm.padded": ["ccm_nuclear"],
    "vibeqc.periodic.ccm.qvf": ["write_ccm_periodic_qvf"],
    "vibeqc.periodic.ccm.ri": [
        "run_ccm_rhf_gdf",
        "run_ccm_rhf_ri_neutral",
        "run_ccm_rhf_rijcosx",
        "run_ccm_rks_gdf",
    ],
    "vibeqc.periodic.ccm.scf": ["run_ccm_rhf", "run_ccm_rhf_scalable"],
    "vibeqc.periodic.ccm.symmetry": [
        "analyze_ccm_symmetry",
        "ccm_symmetry_unique_atom_pairs",
    ],
    "vibeqc.periodic.ccm.uhf": ["run_ccm_uhf"],
    "vibeqc.periodic.ccm.ump2": ["run_ccm_ump2"],
    "vibeqc.periodic.exchange_convention": [
        "assert_matched_exchange_q0",
        "exchange_q0_label",
    ],
}

PRODUCER_MODULES = ("run_case", "run_case_b", "run_d114_mpi")

# Rocksalt LiH, fcc primitive, a = 7.72 bohr -- the ultra-diffuse Li/sto-3g control.
LIH_A = 7.72
MIRROR_TOL = 1e-8            # fixed hosts land at ~4e-14; stale hosts at 2.04e-2
E_MIRROR_STALE = 2.04e-2     # the documented corrupted-vs-fixed gap


def check_symbols():
    missing = []
    producer_root = pathlib.Path(__file__).resolve().parent
    for module_name in PRODUCER_MODULES:
        if not (producer_root / f"{module_name}.py").is_file():
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{module_name} (import failed: {exc})")
    for mod, syms in NEEDED.items():
        try:
            m = importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{mod} (import failed: {exc})")
            continue
        missing += [f"{mod}.{s}" for s in syms if not hasattr(m, s)]
    return missing


def check_g_max_convention():
    """``required_g_max`` must use the tightest AO *pair* ``p = 2*e_max``, giving
    23.83 for H/STO-3G at tau = 1e-9, not the sqrt(2)-too-small 16.85."""
    from vibeqc.periodic.ccm.lowd_four_center import required_g_max

    class _Shell:
        exponents = np.array([3.42525091])   # H/STO-3G tightest primitive

    class _Basis:
        def shells(self):
            return [_Shell()]

    return float(required_g_max(_Basis()))


def check_mirror_fix():
    """direct(fold cderi) vs multi-k GDF on the LiH control. Returns the gap."""
    from vibeqc import Atom, PeriodicSystem
    from vibeqc.periodic.ccm import CCMSystem
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    lat = 0.5 * LIH_A * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    cell = PeriodicSystem(
        3, lat, [Atom(3, [0, 0, 0]), Atom(1, [0.5 * LIH_A] * 3)], 0, 1)
    ccm = CCMSystem(cell, (2, 1, 1), "sto-3g")
    r = run_ccm_rhf_direct(ccm)
    g = run_ccm_rhf_gdf(ccm)
    return abs(r.energy / ccm.n_cells - g.energy), r.energy / ccm.n_cells, g.energy


def check_loaded_core_mirror() -> dict[str, object]:
    """Exercise the loaded native AO-pair kernel on the shifted-mesh defect.

    This is deliberately tiny: three LiH/STO-3G lattice cells and three
    half-reciprocal shifted frequencies. The native general-L binding is called
    directly, independently of ``VIBEQC_AOPAIR_FT_BACKEND``, and compared with
    the pure-Python per-cell construction in the same producer process.
    """

    from vibeqc import Atom, BasisSet, PeriodicSystem
    from vibeqc._aopair_ft import ao_pair_fourier_transform_at_cells
    from vibeqc._vibeqc_core import ao_pair_fourier_transform_bloch_cxx

    lattice = 0.5 * LIH_A * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    cell = PeriodicSystem(
        3,
        lattice,
        [Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.5 * LIH_A] * 3)],
        0,
        1,
    )
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    translations = np.asarray(
        [np.zeros(3), lattice[:, 0], -lattice[:, 0]], dtype=float
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    q_shift = 0.5 * reciprocal[0]
    frequencies = np.asarray(
        [q_shift - reciprocal[0], q_shift, q_shift + reciprocal[0]],
        dtype=float,
    )
    k_gamma = np.zeros(3, dtype=float)

    native = np.asarray(
        ao_pair_fourier_transform_bloch_cxx(
            basis,
            frequencies,
            translations,
            k_gamma,
            0.0,
        ),
        dtype=np.complex128,
    )
    per_cell = np.asarray(
        ao_pair_fourier_transform_at_cells(basis, frequencies, translations),
        dtype=np.complex128,
    )
    reference = np.sum(per_cell, axis=0)
    expected_shape = (6, 6, 3)
    shapes_ok = native.shape == reference.shape == expected_shape
    finite = bool(
        np.all(np.isfinite(native.real))
        and np.all(np.isfinite(native.imag))
        and np.all(np.isfinite(reference.real))
        and np.all(np.isfinite(reference.imag))
    )
    if shapes_ok and finite:
        absolute = np.abs(native - reference)
        scaled = absolute / (CANARY_ATOL + CANARY_RTOL * np.abs(reference))
        max_abs = float(np.max(absolute))
        max_scaled = float(np.max(scaled))
        asymmetry = float(
            np.max(np.abs(reference - reference.transpose(1, 0, 2)))
        )
    else:
        max_abs = float("inf")
        max_scaled = float("inf")
        asymmetry = 0.0
    max_l = max(int(shell.l) for shell in basis.shells())
    passed = bool(
        shapes_ok
        and finite
        and basis.nbasis == 6
        and max_l == 1
        and max_scaled <= 1.0
        and asymmetry > CANARY_ASYMMETRY_MIN
    )
    return {
        "version": LOADED_CORE_CHECK_VERSION,
        "passed": passed,
        "fixture": "lih-sto3g-shifted-mesh/v1",
        "n_ao": int(basis.nbasis),
        "max_l": max_l,
        "n_cells": int(translations.shape[0]),
        "n_frequencies": int(frequencies.shape[0]),
        "atol": CANARY_ATOL,
        "rtol": CANARY_RTOL,
        "max_abs_cxx_python": max_abs,
        "max_scaled_error": max_scaled,
        "reference_transpose_asymmetry": asymmetry,
    }


def _full_host_checks(*, missing: list[str] | None = None) -> dict[str, object]:
    """Run the uncached v2 symbol, g-max, and LiH integration checks."""

    missing_symbols = check_symbols() if missing is None else list(missing)
    result: dict[str, object] = {
        "missing": missing_symbols,
        "g_max": None,
        "g_max_error": None,
        "mirror": None,
        "mirror_error": None,
        "passed": False,
    }
    try:
        result["g_max"] = check_g_max_convention()
    except Exception as exc:  # noqa: BLE001
        result["g_max_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["mirror"] = tuple(float(value) for value in check_mirror_fix())
    except Exception as exc:  # noqa: BLE001
        result["mirror_error"] = f"{type(exc).__name__}: {exc}"
    g_max = result["g_max"]
    mirror = result["mirror"]
    result["passed"] = bool(
        not missing_symbols
        and isinstance(g_max, float)
        and np.isfinite(g_max)
        and g_max > 20.0
        and isinstance(mirror, tuple)
        and len(mirror) == 3
        and all(np.isfinite(value) for value in mirror)
        and mirror[0] < MIRROR_TOL
    )
    return result


def _preflight_from_checks(
    identity: dict[str, object], checks: dict[str, object]
) -> dict[str, object]:
    return {**identity, "probe_passed": checks.get("passed") is True}


def _valid_payload_manifest(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"version", "stream", "files"}:
        return False
    if value.get("version") != PRODUCER_PAYLOAD_VERSION:
        return False
    if value.get("stream") not in {"aiccm2026dev-a", "aiccm2026dev-b"}:
        return False
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        return False
    required = {"producer", "probe_host", "testset", "launcher"}
    if value.get("stream") == "aiccm2026dev-b":
        required.add("b_routes")
    return set(files) == required and all(_is_sha256(item) for item in files.values())


def serialized_producer_attestation_failure(
    provenance: object,
    expected_payload: object,
    *,
    expected_source_commit: str | None = None,
) -> str | None:
    """Validate one finalized process-bound producer attestation.

    This is the reusable serialized composite core of the D77 contract. It
    validates every nested digest/identity relationship; a caller can
    additionally bind the source commit and exact payload bytes. It does not
    validate a caller's top-level transport declaration such as
    ``attestation="git-checkout"``. The route-specific wrapper must enforce
    that layer before treating the result as full D77 evidence.
    """

    if not isinstance(provenance, dict):
        return "producer provenance is missing or is not an object"
    required_aliases = {
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
    }
    if not required_aliases.issubset(provenance):
        return "producer provenance aliases are incomplete"
    attestation = provenance.get("probe_attestation")
    if (
        not isinstance(attestation, dict)
        or set(attestation) != PRODUCER_ATTESTATION_FIELDS
        or attestation.get("attestation_version")
        != PRODUCER_ATTESTATION_VERSION
        or not _is_sha256(attestation.get("attestor_script_id"))
        or attestation.get("probe_passed") is not True
        or not isinstance(attestation.get("host"), str)
        or not attestation["host"].strip()
        or not isinstance(attestation.get("vibeqc_version"), str)
        or not attestation["vibeqc_version"].strip()
        or not _is_sha256(attestation.get("core_build_id"))
        or not _is_full_commit(attestation.get("source_commit"))
        or attestation.get("source_clean") is not True
        or type(attestation.get("core_changed_since_preflight")) is not bool
        or attestation.get("result_identity_stable") is not True
    ):
        return "producer attestation has an invalid finalized shape or identity"
    if (
        expected_source_commit is not None
        and attestation.get("source_commit") != expected_source_commit
    ):
        return "producer attestation source commit does not match the required source"

    payload = attestation.get("producer_payload")
    preflight = attestation.get("preflight_attestation")
    current = attestation.get("current_core_attestation")
    libraries = attestation.get("native_library_versions")
    loaded_core = attestation.get("loaded_core_check")
    if not _valid_payload_manifest(payload) or payload != expected_payload:
        return "producer attestation payload does not match the expected bytes"
    if not _valid_preflight(preflight) or not _valid_preflight(current):
        return "producer attestation preflight/current-core identity is invalid"
    if not _valid_native_library_versions(libraries):
        return "producer attestation native-library identity is invalid"
    if not _valid_loaded_core_check(loaded_core):
        return "producer attestation loaded-core canary is invalid"
    assert isinstance(payload, dict)
    assert isinstance(preflight, dict)
    assert isinstance(current, dict)

    digest_fields = (
        ("producer_payload", "producer_payload_id"),
        ("preflight_attestation", "preflight_attestation_id"),
        ("current_core_attestation", "current_core_attestation_id"),
    )
    for value_key, digest_key in digest_fields:
        if probe_attestation_id(attestation.get(value_key)) != attestation.get(
            digest_key
        ):
            return f"producer attestation {digest_key} is inconsistent"
    if payload["files"].get("probe_host") != attestation.get(
        "attestor_script_id"
    ):
        return "producer payload does not bind the executing attestor"

    for key in PREFLIGHT_ATTESTATION_FIELDS - {"core_build_id"}:
        if preflight[key] != current[key]:
            return "producer preflight/current-core non-core identity changed"
    core_changed = preflight["core_build_id"] != current["core_build_id"]
    if attestation["core_changed_since_preflight"] is not core_changed:
        return "producer core-churn flag contradicts the two attestations"

    current_aliases = {
        "host": attestation["host"],
        "vibeqc_version": attestation["vibeqc_version"],
        "core_build_id": attestation["core_build_id"],
        "source_commit": attestation["source_commit"],
        "source_clean": attestation["source_clean"],
        "probe_script_id": attestation["attestor_script_id"],
    }
    if any(current.get(key) != value for key, value in current_aliases.items()):
        return "producer current-core identity contradicts the attestation"

    attestation_id = probe_attestation_id(attestation)
    aliases = {
        "host": attestation["host"],
        "vibeqc_version": attestation["vibeqc_version"],
        "vibeqc_commit": attestation["source_commit"],
        "source_clean": attestation["source_clean"],
        "core_build_id": attestation["core_build_id"],
        "probe_version": attestation["attestation_version"],
        "probe_script_id": attestation["attestor_script_id"],
        "probe_passed": attestation["probe_passed"],
        "probe_attestation_id": attestation_id,
        "producer_payload_id": attestation["producer_payload_id"],
    }
    if any(
        type(provenance.get(key)) is not type(value)
        or provenance.get(key) != value
        for key, value in aliases.items()
    ):
        return "producer provenance aliases contradict the attestation"
    return None


def producer_attestation(
    preflight_path: str | pathlib.Path | None,
    payload_manifest: dict[str, object],
) -> dict[str, object] | None:
    """Build process-bound evidence from one successful v2 preflight.

    Only a core-path SHA change is recoverable. In that case the full v2
    numerical/API probe is rerun in this process; the cheap shifted-mesh canary
    always runs. Identity is reread after the checks so an update during the
    attestation window fails closed.
    """

    if preflight_path is None or not _valid_payload_manifest(payload_manifest):
        return None
    path = pathlib.Path(preflight_path)
    if not path.is_file():
        return None
    try:
        preflight = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not _valid_preflight(preflight):
        return None

    identity = probe_identity()
    if not _valid_probe_identity(identity):
        return None
    payload_files = payload_manifest["files"]
    if payload_files.get("probe_host") != identity["probe_script_id"]:
        return None
    for key in (
        "probe_version",
        "probe_script_id",
        "host",
        "vibeqc_version",
        "source_commit",
        "source_clean",
    ):
        if preflight.get(key) != identity.get(key):
            return None
    if not _is_sha256(preflight.get("core_build_id")):
        return None

    missing = check_symbols()
    if missing:
        return None
    try:
        loaded_core_check = check_loaded_core_mirror()
    except Exception:  # noqa: BLE001
        return None
    if not _valid_loaded_core_check(loaded_core_check):
        return None

    core_changed = preflight["core_build_id"] != identity["core_build_id"]
    if core_changed:
        current_checks = _full_host_checks(missing=[])
        current_attestation = _preflight_from_checks(identity, current_checks)
        if not _valid_preflight(current_attestation):
            return None
    else:
        current_attestation = dict(preflight)

    libraries = native_library_versions()
    if not _valid_native_library_versions(libraries):
        return None
    payload_id = probe_attestation_id(payload_manifest)
    if not _is_sha256(payload_id):
        return None

    # Catch a deployment or payload edit while the checks themselves ran.
    if probe_identity() != identity or native_library_versions() != libraries:
        return None
    if probe_attestation_id(payload_manifest) != payload_id:
        return None

    return {
        "attestation_version": PRODUCER_ATTESTATION_VERSION,
        "attestor_script_id": identity["probe_script_id"],
        "probe_passed": True,
        "host": identity["host"],
        "vibeqc_version": identity["vibeqc_version"],
        "core_build_id": identity["core_build_id"],
        "source_commit": identity["source_commit"],
        "source_clean": identity["source_clean"],
        "native_library_versions": libraries,
        "producer_payload": payload_manifest,
        "producer_payload_id": payload_id,
        "preflight_attestation": preflight,
        "preflight_attestation_id": probe_attestation_id(preflight),
        "current_core_attestation": current_attestation,
        "current_core_attestation_id": probe_attestation_id(current_attestation),
        "core_changed_since_preflight": core_changed,
        "loaded_core_check": loaded_core_check,
        "result_identity_stable": None,
    }


def finalize_producer_attestation(
    attestation: object,
    payload_manifest: dict[str, object],
) -> dict[str, object] | None:
    """Record whether producer identity stayed stable through result assembly."""

    if not isinstance(attestation, dict):
        return None
    finalized = dict(attestation)
    identity = probe_identity()
    expected_identity = {
        "probe_version": PROBE_VERSION,
        "probe_script_id": finalized.get("attestor_script_id"),
        "host": finalized.get("host"),
        "vibeqc_version": finalized.get("vibeqc_version"),
        "core_build_id": finalized.get("core_build_id"),
        "source_commit": finalized.get("source_commit"),
        "source_clean": finalized.get("source_clean"),
    }
    stable = bool(
        identity == expected_identity
        and _valid_native_library_versions(
            finalized.get("native_library_versions")
        )
        and native_library_versions() == finalized.get("native_library_versions")
        and _valid_loaded_core_check(finalized.get("loaded_core_check"))
        and _valid_payload_manifest(payload_manifest)
        and payload_manifest == finalized.get("producer_payload")
        and probe_attestation_id(payload_manifest)
        == finalized.get("producer_payload_id")
    )
    finalized["result_identity_stable"] = stable
    finalized["probe_passed"] = finalized.get("probe_passed") is True and stable
    return finalized


def main():
    ap = argparse.ArgumentParser(description="AICCM-2026 host-readiness probe")
    ap.add_argument("--emit", metavar="PATH",
                    help="write the JSON attestation curate.py requires of a "
                         "producing run")
    ap.add_argument("--cache-dir", metavar="DIR",
                    help="reuse a passing attestation for this exact clean "
                         "source/core-path/host/package/probe-script identity, so "
                         "a 59-job batch probes once rather than 59 times")
    args = ap.parse_args()

    cache_identity = probe_identity()
    build_id = str(cache_identity["core_build_id"])
    cache = None
    source_is_cacheable = _valid_probe_identity(cache_identity)
    if args.cache_dir and build_id != "unknown" and source_is_cacheable:
        cache_key = probe_attestation_id(cache_identity).removeprefix("sha256:")
        cache = (pathlib.Path(args.cache_dir).expanduser()
                 / f"probe-{cache_key}.json")
    # Producer imports and API-shape checks are cheap and belong to the copied
    # benchmark payload. Run them for every job, even when the expensive
    # numerical checks below have an identity-matched cache entry.
    missing = check_symbols()
    if cache is not None and cache.is_file():
        try:
            cached = json.loads(cache.read_text())
        except Exception:  # noqa: BLE001
            cached = {}
        if (
            not missing
            and set(cached) == PREFLIGHT_ATTESTATION_FIELDS
            and cached.get("probe_passed") is True
            and _valid_preflight(cached)
            and all(cached.get(key) == value for key, value in cache_identity.items())
        ):
            print(f"probe     : {PROBE_VERSION}")
            print(f"host      : {platform.node()}")
            print(f"core      : {build_id[:23]}...")
            print("symbols   : all present")
            print("VERDICT   : campaign-ready (cached identity-bound attestation)")
            if args.emit:
                _atomic_write_json(args.emit, cached)
            return 0

    print(f"probe     : {PROBE_VERSION}")
    print(f"host      : {platform.node()}")
    print(f"python    : {sys.version.split()[0]}  ({sys.executable})")
    print(f"vibeqc    : {vibeqc.__version__}  ({vibeqc.__file__})")

    print(f"symbols   : {'all present' if not missing else missing}")
    checks = _full_host_checks(missing=missing)
    g = checks["g_max"]
    if isinstance(g, float):
        good = np.isfinite(g) and g > 20.0
        print(f"g_max(H)  : {g:.4f}  ({'pair convention, current' if good else 'STALE: sqrt(2) too small'})")
    else:
        print(f"g_max(H)  : probe failed: {checks['g_max_error']}")

    mirror = checks["mirror"]
    if isinstance(mirror, tuple):
        d, e_direct, e_gdf = mirror
        good = d < MIRROR_TOL
        print(f"LiH direct: {e_direct:.12f}")
        print(f"LiH gdf   : {e_gdf:.12f}")
        print(f"|delta|   : {d:.3e}  ({'mirror fix present' if good else 'MIRROR FIX ABSENT'})")
        if not good:
            print(f"            expected < {MIRROR_TOL:.0e}; a stale core gives "
                  f"~{E_MIRROR_STALE:.2e} (f8c213e8). GDF/RI energies from this host "
                  "are CORRUPTED; the direct route is the trustworthy side.")
    else:
        print(f"LiH mirror: probe failed: {checks['mirror_error']}")

    ok = checks["passed"] is True and _valid_probe_identity(cache_identity)
    print(f"VERDICT   : {'campaign-ready' if ok else 'DO NOT DISPATCH'}")

    attestation = {**cache_identity, "probe_passed": bool(ok)}
    if args.emit:
        _atomic_write_json(args.emit, attestation)
    if ok and cache is not None:
        _atomic_write_json(cache, attestation)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
