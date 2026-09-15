#!/usr/bin/env python
"""1D/2D/3D neutral fitted-torus Bloch/GDF control producer.

Implements the neutral Bloch/GDF representation-control row of
``studies/aiccm-2026/COMPARISON_1D2D3D_2026-07-15.md`` (§3, entry point
``run_ccm_rhf_gdf``). One pinned system per invocation:

    vq submit managed-host -d <payload>/ -- bash run.sh --case run_case_cmp.py 3d
    vq submit managed-host -d <payload>/ -- bash run.sh --case run_case_cmp.py 1d
    vq submit managed-host -d <payload>/ -- bash run.sh --case run_case_cmp.py 2d

Every §2 pin is passed EXPLICITLY (rule 0.4): the bare
``PeriodicRHFOptions()`` defaults are ``damping=0.5`` /
``conv_tol_energy=1e-8`` / ``max_iter=100``, all of which violate the pinned
realization, so this producer never relies on route defaults for a pinned
value. The executed ``fock_mixing`` and the resolved auxiliary basis name are
re-read from the result and verified; a mismatch is a stop-and-report
condition (exit 4, ``status="stop-and-report"`` in the JSON).

Exchange-q=0 convention (§0.3): the multi-k HF GDF driver routes HF exchange
to the compcell path with ``exxdiv='ewald'`` automatically (the
``use_compcell=False`` Ewald-3D-K path omits the Madelung correction, so
HF/hybrid is auto-routed; see ``periodic_k_gdf.py``, maintainer decision
2026-06-04) and applies ``apply_exxdiv_ewald_to_K`` whenever the exchange
fraction is nonzero. ``CCMGDFResult`` now carries ``exchange_q0`` and
``exchange_q0_applicability`` fields, while its raw
``PeriodicKRHFGDFResult`` does not. This producer requires those wrapper fields
to agree with an independent derivation from executed evidence:

* the captured SCF progress log must contain the exxdiv routing line
  ("exxdiv='ewald'") emitted by the executed HF multi-k branch, and
* the k-mesh-aware BvK Madelung constant ``xi`` (``_madelung_for_kmesh``)
  must be nonzero for the run's (system, nrep), and
* the method is RHF (exchange fraction alpha = 1, so the K-shift branch
  executes).

All three together record ``exchange_q0='BvK-ewald'`` /
``exchange_q0_applicability='active'`` with
``exchange_q0_derivation='wrapper-field+executed-path'``. A missing or
contradictory wrapper field, or incomplete executed evidence, is a
stop-and-report condition.

Energy normalization: ``CCMGDFResult.energy`` is per unit cell — the wrapped
3-D multi-k KRHF uses the shared Ewald nuclear helper + weighted k-averaged
electronic energy, and ``PeriodicKRHFGDFResult.energy_per_cell_ha`` is an
identity property on ``energy``. The producer independently recomputes the
same-helper Ewald nuclear term, then asserts
``res.energy == res.raw.energy == res.raw.energy_per_cell_ha`` and
``res.energy_per_atom == res.energy / n_atoms_per_cell`` at machine precision
and records all of them.

Provenance: the primary path is the same producer-attestation contract as
``run_case.py`` (D77): ``run.sh`` preflights the host with ``probe_host.py``
and exports ``AICCM_PROBE_ATTESTATION``; this process binds the copied
payload, reruns the shifted-mesh loaded-core canary, and finalizes identity
after result assembly.

**managed-host bundle fallback (2026-07-16 finding).** On managed-host the vq deployment
is an immutable runtime bundle: a sha256-checksummed tarball containing ONLY
``.venv`` (vibeqc installed into site-packages), ``third_party``, an *empty*
``.git`` marker directory, and ``BUILD-INFO.txt`` — no tracked source tree.
``probe_host.imported_source_state()`` therefore fails closed
(``git ls-files`` on the installed package cannot succeed), so the v2
identity is STRUCTURALLY unsatisfiable on this host even though every v2
numerical check passes (observed: symbols all present, g_max 23.8298,
direct/GDF mirror gap 1.243e-14 on vq job f890c8bb96de). When — and only
when — the wrapper env (``VIBEQC_BUNDLE``) is present and the v2 preflight is
unavailable, this producer runs a bundle attestation instead
(``aiccm-managed-host-bundle-attestation/v1``):

* the FULL v2 numerical checks (``check_symbols`` + g_max + direct/GDF
  mirror + the shifted-mesh loaded-core canary) executed in the SAME process
  as the production calculation, all validated with probe_host's own
  validators; plus
* fail-closed bundle identity: ``BUILD-INFO.txt`` (variant/version/
  git_branch/git_sha) must parse, ``git_sha`` must be 40-hex, branch must be
  ``main``, version must equal ``vibeqc.__version__``, the wrapper env
  (``VIBEQC_EXPECTED_SHA`` / ``VIBEQC_EXPECTED_VERSION``) must match, and the
  artifact bytes are INDEPENDENTLY re-hashed and compared to the sibling
  ``.sha256`` checksum file.

A failed attestation (either mode) exits 3 without reporting a number. The
bundle mode is flagged in the result (``attestation_mode``) so the curator
can distinguish it from a v2 producer attestation.
"""

from __future__ import annotations

import site_settings

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import probe_host
import vibeqc as vq
from vibeqc import Atom, PeriodicRHFOptions, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf
from vibeqc.periodic.exchange_convention import BVK_EWALD, exchange_q0_label
from vibeqc.progress import ProgressLogger

METHOD = "aiccm2026dev-a"  # producer stream (payload manifest contract)
ROUTE = "neutral-fitted-torus-bloch-gdf"
SPEC = "COMPARISON_1D2D3D_2026-07-15"
EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA = (
    "vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v2"
)
EWALD_SHIFTED_PAIR_IMPLEMENTATION = (
    "centered-displacement-interplanar-pair-bounds"
)
EWALD_SHIFTED_PAIR_REPAIR_COMMIT = (
    "e578b86c00268a172b651cae29c7845836e31e17"
)
EWALD_SHIFTED_PAIR_CANARY_SCHEMA = (
    "vibeqc.ewald.shifted-pair-canary/v1"
)
EWALD_SHIFTED_PAIR_CANARY_FIXTURE = (
    "mgo-point-charges-basis-shift-37a1/c18"
)
EWALD_SHIFTED_PAIR_CANARY_TOLERANCE = 1.0e-10

# ----- §1: pinned geometries (bohr, column-vector lattice, all dim=3) -----
#
# Each entry: lattice VECTORS a1,a2,a3 given as rows here, assembled into the
# column-vector engine matrix via np.array([a1,a2,a3]).T (PeriodicSystem
# stores columns = Cartesian lattice vectors, post-D88 convention).
SYSTEMS = {
    "1d": dict(
        label="LiH chain, declared 3-D periodic-in-vacuum",
        vectors=[[6.0000, 0.0, 0.0], [0.0, 30.0000, 0.0], [0.0, 0.0, 30.0000]],
        atoms=[(3, [0.0000, 15.0000, 15.0000]), (1, [3.0000, 15.0000, 15.0000])],
        nrep=(8, 1, 1),
        n_electrons=4,
    ),
    "2d": dict(
        label="h-BN monolayer, declared 3-D periodic-in-vacuum",
        vectors=[[4.7319, 0.0, 0.0], [-2.3660, 4.0979, 0.0], [0.0, 0.0, 30.0000]],
        atoms=[(5, [0.0000, 2.7319, 15.0000]), (7, [2.3659, 1.3660, 15.0000])],
        nrep=(3, 3, 1),
        n_electrons=12,
    ),
    "3d": dict(
        label="LiH rocksalt, fcc primitive cell",
        vectors=[[0.0, 3.8579, 3.8579], [3.8579, 0.0, 3.8579], [3.8579, 3.8579, 0.0]],
        atoms=[(3, [0.0000, 0.0000, 0.0000]), (1, [3.8579, 3.8579, 3.8579])],
        nrep=(2, 2, 2),
        n_electrons=4,
    ),
}

BASIS = "sto-3g"
EXPECTED_AUX = "def2-svp-jk"  # default_aux_for('sto-3g'); re-verified at runtime

# §2 pins, recorded verbatim into the result JSON.
PINS = dict(
    method="RHF",
    basis=BASIS,
    rsgdf_ke_cutoff=200.0,
    conv_tol_energy=1e-10,
    max_iter=128,
    diis_subspace_size=8,
    damping=0.0,
    level_shift=0.0,
    fock_mixing=0.0,
    smearing_temperature=0.0,
    linear_dep_threshold=1e-7,
    gdf_linear_dep_threshold=1e-9,
    k_exchange="gdf",
    exxdiv="ewald",
)

#: Per-job path exported by ``run.sh`` inside a fresh private directory.
_attestation_path = os.environ.get("AICCM_PROBE_ATTESTATION")
_ATTESTATION = Path(_attestation_path) if _attestation_path else None


def _producer_payload() -> dict[str, object]:
    """Identify every executable input copied by the directory launcher."""

    here = Path(__file__).resolve().parent
    return {
        "version": probe_host.PRODUCER_PAYLOAD_VERSION,
        "stream": METHOD,
        "files": {
            "producer": probe_host.file_sha256(__file__),
            "probe_host": probe_host.file_sha256(probe_host.__file__),
            "testset": probe_host.file_sha256(here / "testset.py"),
            "launcher": probe_host.file_sha256(here / "run.sh"),
        },
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _repair_is_ancestor(producer_commit: object) -> bool:
    """Verify the pair-complete Ewald repair in a D77 source checkout."""

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


def _ewald_shifted_pair_canary() -> dict[str, object]:
    """Run the repair-specific discriminator in the producer process."""

    record: dict[str, object] = {
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
        record.update(
            energy_ha=energy,
            shifted_energy_ha=shifted_energy,
            abs_delta_ha=delta,
            passed=bool(
                np.isfinite(energy)
                and np.isfinite(shifted_energy)
                and np.isfinite(delta)
                and delta <= EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
            ),
        )
    except Exception:  # noqa: BLE001 -- support evidence fails closed
        pass
    return record


def _ewald_shifted_pair_support(
    provenance: object,
    *,
    canary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Bind the same-helper control to the repaired Ewald/core identity."""

    value = provenance if isinstance(provenance, dict) else {}
    producer_commit = value.get("source_commit")
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


def _ewald_shifted_pair_preflight_failure(support: object) -> str | None:
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


BUNDLE_ATTESTATION_VERSION = site_settings.PUBLIC_BUNDLE_MODE


def _parse_build_info(text: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip()
    return info


def _bundle_runtime_root() -> Path | None:
    """Locate the unpacked bundle root by walking up from ``vibeqc.__file__``."""
    location = getattr(vq, "__file__", None)
    if not location:
        return None
    for parent in Path(location).resolve().parents:
        if (parent / "BUILD-INFO.txt").is_file() and (parent / "run-python").exists():
            return parent
    return None


def _bundle_attestation() -> dict | None:
    """Fail-closed identity + full v2 numerical checks for a bundle deployment.

    Returns the attestation dict with ``probe_passed=True`` only when EVERY
    check passes; otherwise the dict carries ``probe_passed=False`` and a
    ``failures`` list (never None, so the reason is always recorded).
    """
    import platform

    failures: list[str] = []
    art_path = os.environ.get("VIBEQC_BUNDLE", "")
    expected_sha = os.environ.get("VIBEQC_EXPECTED_SHA", "")
    expected_version = os.environ.get("VIBEQC_EXPECTED_VERSION", "")

    root = _bundle_runtime_root()
    build_info: dict[str, str] = {}
    if root is None:
        failures.append("bundle runtime root (BUILD-INFO.txt) not found")
    else:
        build_info = _parse_build_info(
            (root / "BUILD-INFO.txt").read_text(encoding="utf-8")
        )
    git_sha = build_info.get("git_sha", "")
    if not probe_host._is_full_commit(git_sha):
        failures.append(f"BUILD-INFO git_sha not a full commit: {git_sha!r}")
    if build_info.get("git_branch") != "main":
        failures.append(
            f"BUILD-INFO git_branch = {build_info.get('git_branch')!r}, need 'main'"
        )
    version = str(getattr(vq, "__version__", "unknown"))
    if build_info.get("version") != version:
        failures.append(
            f"BUILD-INFO version {build_info.get('version')!r} != "
            f"imported vibeqc {version!r}"
        )
    if expected_sha and git_sha and expected_sha != git_sha:
        failures.append(
            f"wrapper VIBEQC_EXPECTED_SHA {expected_sha!r} != BUILD-INFO {git_sha!r}"
        )
    if expected_version and expected_version != version:
        failures.append(
            f"wrapper VIBEQC_EXPECTED_VERSION {expected_version!r} != {version!r}"
        )

    # Independent artifact integrity re-verification (the wrapper already ran
    # sha256sum -c before unpacking; this re-hashes the bytes ourselves).
    artifact_sha256 = "unknown"
    artifact_sha256_expected = "unknown"
    art = Path(art_path) if art_path else None
    if art is None or not art.is_file():
        failures.append(f"artifact not found: {art_path!r}")
    else:
        artifact_sha256 = probe_host.file_sha256(art)
        checksum_file = art.with_name(art.name + ".sha256")
        try:
            artifact_sha256_expected = (
                "sha256:"
                + checksum_file.read_text(encoding="utf-8").split()[0].lower()
            )
        except Exception:  # noqa: BLE001 -- provenance must fail closed
            failures.append(f"cannot read checksum file {checksum_file}")
        if (
            artifact_sha256 == "unknown"
            or artifact_sha256 != artifact_sha256_expected
        ):
            failures.append(
                f"artifact sha256 mismatch: hashed {artifact_sha256}, "
                f"expected {artifact_sha256_expected}"
            )

    # FULL v2 numerical checks, in the SAME process as the production run.
    checks = probe_host._full_host_checks()
    if checks.get("passed") is not True:
        failures.append(f"v2 numerical checks failed: {checks!r}")
    try:
        loaded_core = probe_host.check_loaded_core_mirror()
    except Exception as exc:  # noqa: BLE001
        loaded_core = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    if not probe_host._valid_loaded_core_check(loaded_core):
        failures.append(f"loaded-core canary failed: {loaded_core!r}")

    mirror = checks.get("mirror")
    return {
        "attestation_mode": site_settings.bundle_mode(),
        "probe_passed": not failures,
        "failures": failures,
        "host": (platform.node() or "unknown").split(".")[0],
        "vibeqc_version": version,
        "source_commit": git_sha or "unknown",
        "source_branch": build_info.get("git_branch", "unknown"),
        "source_identity": (
            "immutable sha256-verified deployment bundle (BUILD-INFO + wrapper "
            "env + independent artifact re-hash); probe_host v2 git identity is "
            "structurally unsatisfiable on bundle deployments (installed "
            "package is not a tracked file, bundle ships no source tree)"
        ),
        "build_info": build_info,
        "artifact_path": str(art) if art else "unknown",
        "artifact_sha256": artifact_sha256,
        "artifact_sha256_expected": artifact_sha256_expected,
        "core_build_id": probe_host.core_build_id(),
        "core_path": _core_path(),
        "python": sys.version.split()[0],
        "numerical_checks": {
            "missing_symbols": checks.get("missing"),
            "g_max_h_sto3g": checks.get("g_max"),
            "mirror_delta_ha": (mirror[0] if isinstance(mirror, tuple) else None),
            "mirror_e_direct_per_cell": (
                mirror[1] if isinstance(mirror, tuple) else None
            ),
            "mirror_e_gdf_per_cell": (
                mirror[2] if isinstance(mirror, tuple) else None
            ),
            "loaded_core_check": loaded_core,
        },
        "producer_payload": _producer_payload(),
        "producer_payload_id": probe_host.probe_attestation_id(
            _producer_payload()
        ),
    }


def _core_path() -> str:
    try:
        from vibeqc import _vibeqc_core

        return str(Path(getattr(_vibeqc_core, "__file__", "") or ""))
    except Exception:  # noqa: BLE001
        return "unknown"


def _provenance() -> dict:
    """Process-bound producer evidence.

    Primary: the D77 v2 producer attestation (requires the run.sh preflight).
    Fallback: the managed-host bundle attestation (requires the wrapper env).
    Fail-closed: anything else reports ``probe_passed=False``.
    """
    import platform

    if _ATTESTATION is not None and os.environ.get("AICCM_SKIP_PROBE") != "1":
        attestation = probe_host.producer_attestation(
            _ATTESTATION, _producer_payload()
        )
        if attestation is not None:
            return {
                "attestation_mode": probe_host.PRODUCER_ATTESTATION_VERSION,
                "probe_passed": True,
                "host": attestation["host"],
                "vibeqc_version": attestation["vibeqc_version"],
                "source_commit": attestation["source_commit"],
                "source_clean": attestation["source_clean"],
                "core_build_id": attestation["core_build_id"],
                "probe_script_id": attestation["attestor_script_id"],
                "probe_attestation_id": probe_host.probe_attestation_id(
                    attestation
                ),
                "probe_attestation": attestation,
                "producer_payload_id": attestation["producer_payload_id"],
            }
    if os.environ.get("VIBEQC_BUNDLE"):
        return _bundle_attestation()
    commit, source_clean = probe_host.imported_source_state()
    return {
        "attestation_mode": "none",
        "probe_passed": False,
        "failures": [
            "no v2 preflight attestation and no bundle wrapper env present"
        ],
        "host": (platform.node() or "unknown").split(".")[0],
        "vibeqc_version": str(getattr(vq, "__version__", "unknown")),
        "source_commit": commit,
        "source_clean": source_clean,
        "core_build_id": probe_host.core_build_id(),
    }


def _finalize_provenance(provenance: dict) -> None:
    """Re-check identity stability after result assembly (both modes)."""
    if provenance.get("attestation_mode") == probe_host.PRODUCER_ATTESTATION_VERSION:
        finalized = probe_host.finalize_producer_attestation(
            provenance.get("probe_attestation"), _producer_payload()
        )
        if finalized is None:
            provenance.update(probe_passed=False, probe_attestation=None)
        else:
            provenance.update(
                probe_passed=finalized.get("probe_passed") is True,
                probe_attestation=finalized,
                probe_attestation_id=probe_host.probe_attestation_id(finalized),
            )
        return
    if provenance.get("attestation_mode") in site_settings.accepted_bundle_modes():
        stable = (
            probe_host.core_build_id() == provenance.get("core_build_id")
            and str(getattr(vq, "__version__", "unknown"))
            == provenance.get("vibeqc_version")
        )
        provenance["result_identity_stable"] = stable
        if not stable:
            provenance["probe_passed"] = False
            provenance.setdefault("failures", []).append(
                "core/version identity changed during result assembly"
            )


def _write_result(
    path: Path,
    result: dict,
    provenance: dict,
    *,
    ewald_canary: dict[str, object] | None = None,
) -> None:
    _finalize_provenance(provenance)
    support = _ewald_shifted_pair_support(
        provenance,
        canary=ewald_canary,
    )
    result["ewald_shifted_pair_support"] = support
    result["ewald_shifted_pair_status"] = support["qualification"]
    result["quantitative_status"] = "not-qualified"
    result["provenance"] = provenance
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if (
        provenance.get("probe_passed") is not True
        or support.get("qualification") != "qualified"
    ):
        raise SystemExit(3)


def build_system(key: str) -> PeriodicSystem:
    """Build one §1 cell and verify the column-vector lattice round-trips."""
    meta = SYSTEMS[key]
    vectors = np.asarray(meta["vectors"], dtype=float)  # rows = a1,a2,a3
    lattice = vectors.T                                 # columns = a1,a2,a3
    atoms = [Atom(z, list(pos)) for z, pos in meta["atoms"]]
    cell = PeriodicSystem(3, lattice, atoms, 0, 1)

    stored = np.asarray(cell.lattice, dtype=float)
    for j in range(3):
        if not np.allclose(stored[:, j], vectors[j], rtol=0.0, atol=1e-12):
            raise SystemExit(
                f"lattice round-trip FAILED for {key}: column {j} of the "
                f"stored lattice is {stored[:, j].tolist()}, pinned a{j+1} is "
                f"{vectors[j].tolist()} — refusing to run (D88 defect class)."
            )
    mol = cell.unit_cell_molecule()
    n_e = int(mol.n_electrons())
    if n_e != int(meta["n_electrons"]):
        raise SystemExit(
            f"electron-count check FAILED for {key}: built cell has {n_e} "
            f"electrons per cell, spec pins {meta['n_electrons']}."
        )
    if int(cell.dim) != 3:
        raise SystemExit(f"dim check FAILED for {key}: dim={cell.dim}, pinned 3.")
    return cell


def pinned_options() -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.conv_tol_energy = PINS["conv_tol_energy"]
    opts.max_iter = PINS["max_iter"]
    opts.damping = PINS["damping"]
    opts.level_shift = PINS["level_shift"]
    opts.diis_subspace_size = PINS["diis_subspace_size"]
    opts.smearing_temperature = PINS["smearing_temperature"]
    opts.use_diis = True  # §2: DIIS on (default True; pinned explicitly)
    return opts


def derive_exchange_q0(
    cell: PeriodicSystem, nrep, log_text: str
) -> tuple[str, str, dict]:
    """Derive the exchange-q=0 record from executed evidence (see docstring)."""
    from vibeqc.periodic_k_gdf import _madelung_for_kmesh

    routing_line = next(
        (
            line.strip()
            for line in log_text.splitlines()
            if "exxdiv='ewald'" in line
        ),
        None,
    )
    xi = float(_madelung_for_kmesh(cell, list(nrep)))
    evidence = {
        "exxdiv_routing_line": routing_line,
        "bvk_madelung_xi": xi,
        "hf_exchange_fraction": 1.0,  # RHF: alpha = 1, K-shift branch executes
        "raw_result_field_present": False,
    }
    if routing_line is not None and np.isfinite(xi) and xi != 0.0:
        return exchange_q0_label("ewald"), "active", evidence
    return "unverified", "unverified", evidence


def _shared_e_nuclear_match(
    executed: float,
    reference: float,
    *,
    atol: float = 1e-12,
) -> tuple[float, bool]:
    """Return the nuclear-energy delta and a finite fail-closed match flag."""

    delta = float(executed) - float(reference)
    finite = all(np.isfinite(value) for value in (executed, reference, delta))
    return delta, bool(finite and abs(delta) <= float(atol))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("system", choices=sorted(SYSTEMS))
    ap.add_argument("--out", default=os.environ.get("VQ_WORKDIR", "."))
    args = ap.parse_args()

    provenance = _provenance()
    if provenance.get("probe_passed") is not True:
        print(
            json.dumps({"status": "attestation-failed", "provenance": provenance},
                       indent=2)
        )
        raise SystemExit(3)

    meta = SYSTEMS[args.system]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"cmp1d2d3d__{args.system}__gdf.json"
    log_path = out / f"cmp1d2d3d__{args.system}__gdf.scf.log"

    ewald_canary = _ewald_shifted_pair_canary()
    preflight_support = _ewald_shifted_pair_support(
        provenance,
        canary=ewald_canary,
    )
    preflight_failure = _ewald_shifted_pair_preflight_failure(preflight_support)
    if preflight_failure is not None:
        _write_result(
            result_path,
            {
                "spec": SPEC,
                "route": ROUTE,
                "system": args.system,
                "status": "stop-and-report",
                "pin_violations": [preflight_failure],
            },
            provenance,
            ewald_canary=ewald_canary,
        )
        raise SystemExit(3)

    cell = build_system(args.system)
    ccm = CCMSystem(cell, tuple(meta["nrep"]), BASIS)
    opts = pinned_options()
    plog = ProgressLogger(log_path=log_path, verbose=5)

    from vibeqc.aux_basis import default_aux_for

    expected_aux = str(default_aux_for(BASIS))

    t0 = time.time()
    res = run_ccm_rhf_gdf(
        ccm,
        options=opts,
        fock_mixing=PINS["fock_mixing"],
        rsgdf_ke_cutoff=PINS["rsgdf_ke_cutoff"],
        linear_dep_threshold=PINS["linear_dep_threshold"],
        gdf_linear_dep_threshold=PINS["gdf_linear_dep_threshold"],
        k_exchange=PINS["k_exchange"],
        progress=plog,  # logging passthrough only (executed-path evidence)
    )
    walltime = time.time() - t0
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":  # ru_maxrss is bytes on macOS, KB on Linux
        peak_rss_kb //= 1024

    raw = res.raw
    violations: list[str] = []

    # --- D86: executed fock_mixing must be exactly 0.0 --------------------
    executed_fock_mixing = float(getattr(raw, "fock_mixing", float("nan")))
    if executed_fock_mixing != 0.0:
        violations.append(
            f"executed fock_mixing = {executed_fock_mixing!r}, pinned 0.0"
        )

    # --- resolved aux basis must match the -b RI default -------------------
    aux_name = str(getattr(raw, "aux_basis_name", ""))
    if aux_name != expected_aux or aux_name != EXPECTED_AUX:
        violations.append(
            f"resolved aux_basis_name = {aux_name!r}; default_aux_for"
            f"({BASIS!r}) = {expected_aux!r}; expected {EXPECTED_AUX!r}"
        )

    # --- energy normalization: per unit cell, not N_c-total ----------------
    n_unit_atoms = ccm.n_atoms // ccm.n_cells
    raw_energy = float(raw.energy)
    per_cell_prop = float(getattr(raw, "energy_per_cell_ha", raw_energy))
    if not (
        res.energy == raw_energy
        and per_cell_prop == raw_energy
        and abs(res.energy_per_atom - res.energy / n_unit_atoms) < 1e-12
    ):
        violations.append(
            "energy normalization ambiguous: "
            f"res.energy={res.energy!r} raw.energy={raw_energy!r} "
            f"raw.energy_per_cell_ha={per_cell_prop!r} "
            f"res.energy_per_atom={res.energy_per_atom!r}"
        )

    # --- D99: fitted 3-D routes use the shared Ewald nuclear helper ---------
    executed_e_nuclear = float(getattr(raw, "e_nuclear", float("nan")))
    shared_e_nuclear = float(vq.ewald_nuclear_repulsion(cell))
    e_nuclear_delta, e_nuclear_matches = _shared_e_nuclear_match(
        executed_e_nuclear,
        shared_e_nuclear,
    )
    if not e_nuclear_matches:
        violations.append(
            "nuclear-repulsion assembly mismatch: "
            f"raw.e_nuclear={executed_e_nuclear!r}, shared 3-D Ewald helper="
            f"{shared_e_nuclear!r}, delta={e_nuclear_delta!r}"
        )

    # --- §0.3: exchange-q=0 convention, derived from executed evidence -----
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    q0_label, q0_applicability, q0_evidence = derive_exchange_q0(
        cell, meta["nrep"], log_text
    )
    wrapper_q0 = str(getattr(res, "exchange_q0", ""))
    wrapper_q0_applicability = str(
        getattr(res, "exchange_q0_applicability", "")
    )
    q0_evidence.update(
        wrapper_result_field_present=(
            hasattr(res, "exchange_q0")
            and hasattr(res, "exchange_q0_applicability")
        ),
        wrapper_exchange_q0=wrapper_q0,
        wrapper_exchange_q0_applicability=wrapper_q0_applicability,
    )
    if q0_label != BVK_EWALD or q0_applicability != "active":
        violations.append(
            f"exchange_q0 derivation failed closed: label={q0_label!r} "
            f"applicability={q0_applicability!r} evidence={q0_evidence!r}"
        )
    if wrapper_q0 != q0_label or wrapper_q0_applicability != q0_applicability:
        violations.append(
            "exchange_q0 wrapper/executed-path mismatch: "
            f"wrapper=({wrapper_q0!r}, {wrapper_q0_applicability!r}), "
            f"derived=({q0_label!r}, {q0_applicability!r})"
        )
    # Now that the base mesh is READ BACK from the driver, this comparison has
    # a reachable negative branch; it did not while the field was a copy of
    # PINS. None means the driver reported nothing, which is also a violation.
    _ke_executed = getattr(raw, "rsgdf_ke_cutoff", None)
    if _ke_executed is None or float(_ke_executed) != float(
        PINS["rsgdf_ke_cutoff"]
    ):
        violations.append(
            "rsgdf_ke_cutoff requested != executed: "
            f"requested={PINS['rsgdf_ke_cutoff']!r} executed={_ke_executed!r}"
        )

    # --- final SCF trace tail (evidence for the 1e-10 convergence pin) -----
    trace_tail = []
    for it in list(getattr(raw, "scf_trace", []) or [])[-3:]:
        trace_tail.append(
            {
                "iteration": int(getattr(it, "iteration", -1)),
                "energy": float(getattr(it, "energy", float("nan"))),
                "delta_e": float(getattr(it, "delta_e", float("nan"))),
            }
        )

    result = dict(
        spec=SPEC,
        route=ROUTE,
        route_role="representation-control",
        ccm_approach="not-applicable",
        ccm_construction="neutral-fitted-torus",
        evaluation_representation="bloch-gdf",
        entry_point="run_ccm_rhf_gdf",
        system=args.system,
        system_label=meta["label"],
        basis=BASIS,
        nrep=list(meta["nrep"]),
        dim=3,
        n_atoms_per_cell=n_unit_atoms,
        n_cells=int(ccm.n_cells),
        nbf_unit_cell=int(ccm.nbf // ccm.n_cells),
        pins=PINS,
        # §4 report fields
        energy_per_cell_ha=float(res.energy),
        energy_per_atom_ha=float(res.energy_per_atom),
        energy_normalization="per-unit-cell (verified == raw.energy == raw.energy_per_cell_ha)",
        nuclear_repulsion_model="shared-3d-ewald-same-helper",
        nuclear_repulsion_energy_ha=executed_e_nuclear,
        nuclear_repulsion_shared_helper_ha=shared_e_nuclear,
        nuclear_repulsion_delta_ha=e_nuclear_delta,
        converged=bool(res.converged),
        n_iter=int(res.n_iter),
        executed_fock_mixing=executed_fock_mixing,
        aux_basis_name=aux_name,
        n_aux=int(getattr(raw, "n_aux", 0)),
        rsgdf_ke_cutoff_requested=PINS["rsgdf_ke_cutoff"],
        # MEASURED from the run, not copied from the request. Until 2026-08-28
        # this read `PINS["rsgdf_ke_cutoff"]` with the comment "kwarg forwarded
        # verbatim" -- an "_executed" field that could not disagree with its
        # "_requested" twin, so it was never evidence. The comment was also
        # false on the Gamma path, which resolves a TAIL the producer never saw
        # (GitLab IID 307). `None` = the driver did not report it.
        rsgdf_ke_cutoff_executed=(
            None if getattr(raw, "rsgdf_ke_cutoff", None) is None
            else float(raw.rsgdf_ke_cutoff)
        ),
        # The reciprocal-support convention actually APPLIED. run_ccm_rhf_gdf
        # at nrep=(1,1,1) dispatches to run_pbc_gdf_rhf, which auto-sizes this
        # on tight-core cells; the multi-k loop applies none. A row without it
        # cannot be compared against a direct-route row at all.
        rsgdf_tail_ke_cutoff_executed=(
            None if getattr(raw, "rsgdf_tail_ke_cutoff", None) is None
            else float(raw.rsgdf_tail_ke_cutoff)
        ),
        parity_held=bool(getattr(res, "parity_held", False)),
        linear_dep_threshold=PINS["linear_dep_threshold"],
        gdf_linear_dep_threshold=PINS["gdf_linear_dep_threshold"],
        exchange_q0=q0_label,
        exchange_q0_applicability=q0_applicability,
        exchange_q0_derivation="wrapper-field+executed-path",
        exchange_q0_evidence=q0_evidence,
        backend=str(getattr(raw, "backend", "unknown")),
        scf_trace_tail=trace_tail,
        smearing_temperature_executed=float(
            getattr(raw, "smearing_temperature", 0.0)
        ),
        level_shift_executed=float(getattr(raw, "level_shift", 0.0)),
        walltime_s=round(walltime, 1),
        peak_rss_mb=round(peak_rss_kb / 1024.0, 1),
        status="ok" if not violations and res.converged else (
            "stop-and-report" if violations else "unconverged"
        ),
        pin_violations=violations,
    )
    _write_result(
        result_path,
        result,
        provenance,
        ewald_canary=ewald_canary,
    )
    if violations:
        raise SystemExit(4)
    if not res.converged:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
