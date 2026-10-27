"""QVF job containers: build, load, and run a ``.qvf`` that *describes*
a calculation (QVF spec § 5.9 ``job.spec`` + § 3.2 ``run_status``).

A *pending container* is a QVF archive carrying a ``structure`` section,
a ``job.spec`` section (the declarative request: job type, method, basis,
functional, charge, multiplicity, k-mesh, tasks, engine options), and
``provenance.run_status = "pending"``. ``vibeqc run job.qvf`` — or
:func:`run_container` — opens the container, reconstructs the job, and
executes it through the ordinary :func:`vibeqc.run_job` /
:func:`vibeqc.run_periodic_job` surface.

The job specification is declarative data, never code: nothing in this
module executes an embedded script, and a ``run.record`` ``input`` member
is never evaluated (QVF spec § 5.9).

Producer counterpart: :func:`write_pending_qvf` builds a pending
container from a :class:`Molecule` / :class:`PeriodicSystem` plus the
spec fields.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import time
import zipfile
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Optional, Union

from .molecule import ANGSTROM_TO_BOHR, _atomic_number
from ._qvf_options import hydrate_option
from .output._text_safety import safe_json_bytes
from .output._stem_paths import stem_sibling

__all__ = [
    "QvfJobContainer",
    "load_job_container",
    "build_system",
    "run_container",
    "write_pending_qvf",
    "job_spec_method_fields",
    "job_spec_name",
]

# Task identifiers understood by vibe-qc's runners. ``single_point`` is
# the implicit default; ``optimize`` / ``hessian`` map to the runners'
# same-named switches and may be combined (opt + freq).
_KNOWN_TASKS = ("single_point", "optimize", "hessian")

# Spec fields owned by the typed JobSpecPayload surface. ``options`` may
# not restate them (a spec that disagrees with itself is refused rather
# than silently resolved), and they never pass through as raw kwargs.
_RESERVED_OPTION_KEYS = frozenset(
    {
        "job_type",
        "method",
        "basis",
        "functional",
        "charge",
        "multiplicity",
        "kpoints",
        "tasks",
        "options",
        "molecule",
        "system",
        "output",
        "optimize",
        "hessian",
        # The container's own result archive must exist for the in-place
        # update; a spec cannot opt out of it.
        "output_qvf",
    }
)

# QVF spec Sec. 5.9: a job.spec ``functional`` / ``basis`` is a non-empty
# name, ``null`` ("the method takes none": a runner MUST NOT supply one), or
# absent (the engine's default). ``""`` is a deprecated spelling of null that
# earlier vibe-qc versions wrote; readers accept it, producers never write it.
#
# Methods that take no exchange-correlation functional: Hartree-Fock and the
# correlated wavefunction methods (the semiempirical and MLIP methods, which
# take neither a functional nor a basis, are recognised separately). A
# Kohn-Sham method, a DFT composite or a double hybrid takes one, and
# ``auto`` may; an absent functional then leaves the choice to the engine.
_NO_FUNCTIONAL_METHODS = frozenset(
    {
        "hf",
        "rhf",
        "uhf",
        "rohf",
        "ghf",
        "hf-3c",
        "selected_ci",
        "dmrg",
        "v2rdm",
        "transcorrelated_ci",
        "casci",
        "mrci",
        "casscf",
        "nevpt2",
        "caspt2",
        "fci",
        "ci",
        "cisd",
        "ccsd",
        "ccsd(t)",
        "cc3",
        "ccsdt",
        "bccd",
        "bccd(t)",
        "cc2",
        "ccd",
        "lccd",
        "lccsd",
        "cepa(0)",
        "cepa(1)",
        "cepa(2)",
        "cepa(3)",
        "qcisd",
        "qcisd(t)",
        "mp2",
        "scs-mp2",
        "sos-mp2",
        "dlpno-mp2",
        "dlpno-ccsd",
        "dlpno-ccsd(t)",
        "ovgf",
    }
)
# Machine-learning interatomic potentials: no basis set, no functional.
_MLIP_METHOD_KEYS = frozenset({"mace"})


def _method_key(method: object) -> Optional[str]:
    if method is None:
        return None
    key = str(method).strip().lower()
    return key or None


def _method_is_basis_free(key: str) -> bool:
    """Semiempirical / tight-binding and MLIP methods take no basis set."""
    if key in _MLIP_METHOD_KEYS:
        return True
    from .semiempirical.routes import is_semiempirical_method

    return is_semiempirical_method(key)


def job_spec_method_fields(
    method: object,
    *,
    basis: object = None,
    functional: object = None,
) -> dict[str, Any]:
    """The ``basis`` / ``functional`` entries of a job.spec payload.

    QVF spec Sec. 5.9: a given name is written as a string; a method that
    takes no basis set (semiempirical, tight-binding, MLIP) writes
    ``basis: null``, and a method that takes no functional (Hartree-Fock, a
    correlated wavefunction method, a semiempirical or MLIP method) writes
    ``functional: null``; otherwise an unnamed input is left out, which means
    the engine's default. Never writes the deprecated empty string. A
    basis-free method records ``basis: null`` even when a placeholder label
    was passed, because the method takes no basis set.
    """
    key = _method_key(method)
    basis_free = key is not None and _method_is_basis_free(key)
    fields: dict[str, Any] = {}

    basis_name = None if basis is None else str(basis).strip()
    if basis_free:
        fields["basis"] = None
    elif basis_name:
        fields["basis"] = basis_name

    functional_name = None if functional is None else str(functional).strip()
    if functional_name:
        fields["functional"] = functional_name
    elif key is not None and (basis_free or key in _NO_FUNCTIONAL_METHODS):
        fields["functional"] = None
    return fields


def job_spec_name(
    spec: dict[str, Any], key: str, *, source: str = "job.spec"
) -> Optional[str]:
    """Read a job.spec ``basis`` / ``functional`` as a name or None.

    ``None`` covers an absent key, ``null`` ("the method takes none") and the
    deprecated empty string, which QVF spec Sec. 5.9 reads as ``null``. The
    runner supplies no such input in any of these cases. Anything other than
    a string or null is refused rather than stringified.
    """
    value = spec.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{source}: job.spec {key} must be a string or null, got {value!r}"
        )
    name = value.strip()
    return name or None


def _is_json_unsafe(value: object) -> bool:
    """Return True if *value* is not safe to embed in ``job.spec.options``.

    The QVF serialises the options dict as JSON (QVF spec § 5.9).  Any value
    that cannot round-trip through ``json.dumps → json.loads`` without loss of
    type or data is "unsafe" here — embedding it will corrupt the container at
    write time or cause a wrong-answer / abort at read time.

    pybind11-wrapped C++ option classes (RHFOptions, CCSDOptions, …) are not
    JSON-serializable and must be passed as direct ``run_job`` kwargs, never
    inside ``options``.
    """
    # Direct type check: pybind11 classes are not instances of any JSON-safe
    # Python primitive or standard container.
    if isinstance(value, (
        str, int, float, bool, type(None),
        list, tuple, dict,
    )):
        return False
    # numpy arrays / ndarrays also survive round-trip (via __array__ → list)
    # but are not standard JSON types; we flag them for clarity even though
    # json.dumps will raise TypeError later.
    try:
        import numpy as np  # local to avoid startup cost when safe
        if isinstance(value, np.ndarray):
            return False
    except ImportError:
        pass
    # Everything else (pybind11 types, custom classes, …) is unsafe.
    return True


@dataclasses.dataclass
class QvfJobContainer:
    """A parsed QVF job container (not yet a live system)."""

    path: Path
    manifest: dict[str, Any]
    spec: dict[str, Any]
    structure: dict[str, Any]
    run_status: Optional[str]
    job_spec_section: Optional[dict[str, Any]] = None
    job_spec_bytes: Optional[bytes] = None

    @property
    def job_type(self) -> str:
        return str(self.spec.get("job_type"))


def _read_member(
    zf: zipfile.ZipFile, section: dict[str, Any], role: str, *, kind: str
) -> bytes:
    member = section.get("members", {}).get(role)
    if not isinstance(member, dict) or "path" not in member:
        raise ValueError(
            f"QVF container: {kind} section has no {role!r} member"
        )
    raw = zf.read(member["path"])
    want = member.get("sha256")
    if want and hashlib.sha256(raw).hexdigest() != want:
        raise ValueError(
            f"QVF container: member {member['path']!r} fails its sha256 "
            f"checksum -- refusing to run a corrupted container"
        )
    return raw


def _job_structure_section(
    manifest: dict[str, Any], spec: dict[str, Any], job: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the declared input, with section-level reference precedence."""
    structures = [
        section for section in manifest.get("sections", [])
        if section.get("kind") == "structure"
    ]
    reference = job.get("structure_ref", spec.get("structure_ref"))
    if "structure_ref" in job or "structure_ref" in spec:
        matches = [s for s in structures if s.get("id") == reference]
        if not isinstance(reference, str) or len(matches) != 1:
            raise ValueError(
                f"QVF container: structure_ref {reference!r} must name "
                "exactly one structure section"
            )
        return matches[0]
    if len(structures) != 1:
        raise ValueError(
            "QVF container: a job without structure_ref requires exactly "
            f"one structure section, found {len(structures)}"
        )
    return structures[0]


def load_job_container(path: Union[str, os.PathLike]) -> QvfJobContainer:
    """Parse ``path`` into a :class:`QvfJobContainer`.

    Requires exactly one ``job.spec`` section. Its section-level
    ``structure_ref`` takes precedence over its payload reference; a job
    without either requires exactly one structure. Verifies both input
    members against their declared sha256 before use.
    """
    from .output.formats.qvf import qvf_read_version

    p = Path(os.fspath(path))
    with zipfile.ZipFile(p, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        # Spec Sec. 3.1: the withdrawn version 2 is read as 1 (with a
        # W_VERSION_WITHDRAWN warning); any other version is refused.
        qvf_read_version(manifest, source=f"QVF container {p.name}")
        sections = manifest.get("sections", [])
        job_specs = [s for s in sections if s.get("kind") == "job.spec"]
        if len(job_specs) != 1:
            raise ValueError(
                f"QVF container {p.name}: expected exactly one job.spec "
                f"section, found {len(job_specs)} -- not a runnable job "
                f"container"
            )
        spec_bytes = _read_member(
            zf, job_specs[0], "spec", kind="job.spec"
        )
        spec = json.loads(spec_bytes)
        structure_section = _job_structure_section(manifest, spec, job_specs[0])
        structure = json.loads(
            _read_member(zf, structure_section, "structure", kind="structure")
        )
    if spec.get("job_type") not in ("molecular", "periodic"):
        raise ValueError(
            f"QVF container {p.name}: job.spec job_type must be "
            f"'molecular' or 'periodic', got {spec.get('job_type')!r}"
        )
    run_status = manifest.get("provenance", {}).get("run_status")
    return QvfJobContainer(
        path=p,
        manifest=manifest,
        spec=spec,
        structure=structure,
        run_status=run_status,
        job_spec_section=deepcopy(job_specs[0]),
        job_spec_bytes=spec_bytes,
    )


def build_system(container: QvfJobContainer):
    """Reconstruct the :class:`Molecule` / :class:`PeriodicSystem`.

    Positions and lattice rows in the QVF structure payload are Angstrom
    (spec § 4.3); the native types want bohr, with the lattice as column
    vectors. ``charge`` / ``multiplicity`` come from the job.spec (the
    request is authoritative; the structure payload does not carry them).
    """
    import numpy as np

    from ._vibeqc_core import Atom, Molecule, PeriodicSystem

    atoms = []
    for i, a in enumerate(container.structure.get("atoms", [])):
        z = a.get("atomic_number")
        if z is None:
            z = _atomic_number(str(a.get("symbol", "")))
        pos = a.get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            raise ValueError(
                f"QVF container {container.path.name}: atom {i} has no "
                f"3-vector position"
            )
        atoms.append(
            Atom(int(z), [float(c) * ANGSTROM_TO_BOHR for c in pos])
        )
    if not atoms:
        raise ValueError(
            f"QVF container {container.path.name}: structure has no atoms"
        )

    charge = int(container.spec.get("charge", 0))
    multiplicity = int(container.spec.get("multiplicity", 1))

    pbc = container.structure.get("pbc") or [False, False, False]
    dim = sum(bool(x) for x in pbc)
    if container.job_type == "molecular":
        if dim:
            raise ValueError(
                f"QVF container {container.path.name}: job_type is "
                f"'molecular' but the structure is periodic (pbc={pbc})"
            )
        return Molecule(atoms, charge, multiplicity)

    if dim == 0:
        raise ValueError(
            f"QVF container {container.path.name}: job_type is 'periodic' "
            f"but the structure carries no periodic axis (pbc={pbc})"
        )
    # PeriodicSystem represents prefix masks only (T../TT./TTT); the QVF
    # spec allows any axis pattern (§ 5.1), so refuse the rest explicitly.
    if not all(pbc[i] for i in range(dim)):
        raise ValueError(
            f"QVF container {container.path.name}: pbc={pbc} is not a "
            f"prefix mask (periodic axes first); vibe-qc's PeriodicSystem "
            f"cannot represent it -- permute the structure's axes"
        )
    lattice_rows = container.structure.get("lattice_vectors")
    if lattice_rows is None:
        raise ValueError(
            f"QVF container {container.path.name}: periodic structure "
            f"has no lattice_vectors"
        )
    lat = np.asarray(lattice_rows, dtype=float)
    if lat.shape != (3, 3):
        raise ValueError(
            f"QVF container {container.path.name}: lattice_vectors must "
            f"be 3x3, got shape {lat.shape}"
        )
    # Row vectors in Angstrom -> column vectors in bohr.
    return PeriodicSystem(
        dim=int(dim),
        lattice=np.asarray(
            lat.T * ANGSTROM_TO_BOHR, dtype=float, order="F"
        ),
        unit_cell=atoms,
        charge=charge,
        multiplicity=multiplicity,
    )


def _runner_kwargs(
    container: QvfJobContainer, runner
) -> dict[str, Any]:
    """Map the JobSpecPayload onto ``runner``'s keyword surface.

    Unknown tasks and unknown / reserved option keys are refused (QVF
    spec § 5.9: a runner MUST refuse rather than silently drop).  Native
    option records are hydrated through their public constructors, with
    unsupported fields refused rather than silently discarded (issue #366).
    """
    spec = container.spec
    kwargs: dict[str, Any] = {}
    if spec.get("method") is not None:
        kwargs["method"] = str(spec["method"])
    functional = job_spec_name(
        spec, "functional", source=f"QVF container {container.path.name}"
    )
    if functional is not None:
        kwargs["functional"] = functional
    if container.job_type == "periodic":
        if spec.get("kpoints") is not None:
            kwargs["kpoints"] = [int(n) for n in spec["kpoints"]]
    elif spec.get("kpoints") is not None:
        raise ValueError(
            f"QVF container {container.path.name}: kpoints given for a "
            f"molecular job"
        )

    tasks = [str(t) for t in (spec.get("tasks") or ["single_point"])]
    for t in tasks:
        if t not in _KNOWN_TASKS:
            raise ValueError(
                f"QVF container {container.path.name}: unknown task {t!r} "
                f"(this vibe-qc understands {', '.join(_KNOWN_TASKS)})"
            )
    if "optimize" in tasks:
        kwargs["optimize"] = True
    if "hessian" in tasks:
        kwargs["hessian"] = True

    params = inspect.signature(runner).parameters
    accepted = set(params)
    # A runner with a **kwargs catch-all cannot be signature-validated;
    # vibe-qc's real runners have no catch-all, so this only relaxes
    # test doubles and wrappers.
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    options = spec.get("options", {})
    if not isinstance(options, dict):
        raise ValueError(
            f"QVF container {container.path.name}: job.spec options must "
            f"be an object"
        )
    for key, value in options.items():
        if key in _RESERVED_OPTION_KEYS:
            raise ValueError(
                f"QVF container {container.path.name}: option {key!r} "
                f"restates a typed job.spec field -- set it at the spec "
                f"top level instead"
            )
        if key not in accepted and not has_var_keyword:
            raise ValueError(
                f"QVF container {container.path.name}: option {key!r} is "
                f"not understood by {runner.__name__} -- refusing to "
                f"silently drop it (QVF spec 5.9)"
            )
        kwargs[key] = hydrate_option(key, value)
    return kwargs


def _utc_now() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _section_member_paths(section: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    members = section.get("members")
    if not isinstance(members, dict):
        return paths
    for member in members.values():
        if isinstance(member, dict) and isinstance(member.get("path"), str):
            paths.add(member["path"])
    return paths


def _temporary_sibling(path: Path, label: str) -> Path:
    fd, name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.{label}.",
        suffix=".tmp",
    )
    os.close(fd)
    return Path(name)


def _copy_zip_member(
    source: zipfile.ZipFile,
    target: zipfile.ZipFile,
    name: str,
    written: set[str],
) -> None:
    """Stream one member, preserving its ZIP metadata and compression."""
    if name in written:
        return
    info = copy(source.getinfo(name))
    with source.open(name, "r") as src, target.open(
        info, "w", force_zip64=True
    ) as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    written.add(name)


def _settle_withdrawn_version(manifest: dict[str, Any]) -> None:
    """Rewrite a withdrawn ``qvf_version`` 2 as 1 before an in-place update.

    QVF spec Sec. 3.1: a tool that updates such an archive in place writes
    ``qvf_version: 1`` and the v1 ``schema_uri`` into its new manifest.
    """
    from .output.formats.qvf import (
        QVF_FORMAT_VERSION,
        _SCHEMA_URI,
        _WITHDRAWN_QVF_VERSION,
    )

    version = manifest.get("qvf_version")
    if (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version == _WITHDRAWN_QVF_VERSION
    ):
        manifest["qvf_version"] = QVF_FORMAT_VERSION
        manifest["schema_uri"] = _SCHEMA_URI


def _write_manifest(
    target: zipfile.ZipFile, manifest: dict[str, Any]
) -> None:
    target.writestr(
        "manifest.json",
        safe_json_bytes(manifest, indent=2),
        compress_type=zipfile.ZIP_STORED,
    )


def _validate_qvf_or_raise(path: Path, *, action: str) -> None:
    from .output.formats.qvf import validate_qvf

    report = validate_qvf(path)
    if not report["valid"]:
        raise ValueError(
            f"QVF container {action} produced an invalid archive:\n  - "
            + "\n  - ".join(report["errors"][:8])
        )


def _mark_container_running(path: Path) -> None:
    """Atomically preserve every submitted member while marking it running."""
    temp = _temporary_sibling(path, "running")
    try:
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
            temp, "w"
        ) as target:
            written: set[str] = set()
            for info in source.infolist():
                if info.filename == "manifest.json":
                    continue
                _copy_zip_member(source, target, info.filename, written)
            manifest = json.loads(source.read("manifest.json"))
            _settle_withdrawn_version(manifest)
            provenance = manifest.setdefault("provenance", {})
            provenance["run_status"] = "running"
            provenance.pop("checkpoint", None)
            _write_manifest(target, manifest)
        _validate_qvf_or_raise(temp, action="running-state update")
        os.replace(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def _next_run_sequence(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    sequences = [
        int(record.get("sequence", index))
        for index, record in enumerate(records)
    ]
    return max(sequences) + 1


def _write_opaque_bytes(
    target: zipfile.ZipFile,
    path: str,
    payload: bytes,
) -> dict[str, Any]:
    target.writestr(path, payload, compress_type=zipfile.ZIP_DEFLATED)
    return {
        "path": path,
        "format": "binary",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_opaque_file(
    target: zipfile.ZipFile,
    archive_path: str,
    source_path: Path,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    info = zipfile.ZipInfo(archive_path)
    info.compress_type = zipfile.ZIP_DEFLATED
    with source_path.open("rb") as source, target.open(
        info, "w", force_zip64=True
    ) as destination:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            destination.write(chunk)
    return {
        "path": archive_path,
        "format": "binary",
        "sha256": digest.hexdigest(),
    }


def _runner_result_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path, "r") as archive:
            manifest = json.loads(archive.read("manifest.json"))
    except (OSError, ValueError, zipfile.BadZipFile, KeyError):
        return None
    records = [
        section
        for section in manifest.get("sections", [])
        if section.get("kind") == "run.record"
    ]
    if not records:
        return None
    if manifest.get("provenance", {}).get("run_status") == "running":
        return None
    return manifest


def _run_attachment_paths(
    *,
    output: Path,
    system_path: Path,
) -> dict[str, tuple[Path, str, str]]:
    """Return non-canonical sidecars that belong in the run record."""
    candidates: dict[str, tuple[Path, str, str]] = {
        "system": (
            system_path,
            "application/toml",
            "vibe-qc runtime environment and output contract.",
        ),
        "perf": (
            stem_sibling(output, ".perf"),
            "text/plain; charset=utf-8",
            "vibe-qc performance timing log.",
        ),
        "structured": (
            stem_sibling(output, ".scf.jsonl"),
            "application/x-ndjson",
            "vibe-qc structured event log.",
        ),
    }
    return {
        role: value
        for role, value in candidates.items()
        if value[0].is_file()
    }


# Section-id references defined by the pinned QVF v1 contract. Never rewrite
# arbitrary strings or vendor payloads as though they were references.
_SECTION_REFERENCE_KEYS = frozenset({
    "structure_ref", "run_ref", "job_ref", "trajectory_ref", "wavefunction_ref",
    "hessian_ref", "states_ref", "excited_states_ref", "state_ref", "nto_ref",
    "density_ref", "operand_a", "operand_b",
})


def _remap_result_references(value: Any, identifiers: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_remap_result_references(item, identifiers) for item in value]
    if not isinstance(value, dict):
        return value
    mapped = {}
    for key, item in value.items():
        if key in _SECTION_REFERENCE_KEYS and isinstance(item, str):
            mapped[key] = identifiers.get(item, item)
        elif key.startswith("x_"):
            if _mentions_remapped_id(item, identifiers):
                raise ValueError(
                    "QVF finalization cannot safely relocate references "
                    f"inside vendor field {key!r}"
                )
            mapped[key] = item
        else:
            mapped[key] = _remap_result_references(item, identifiers)
    return mapped


def _mentions_remapped_id(value: Any, identifiers: dict[str, str]) -> bool:
    if isinstance(value, str):
        return value in identifiers and identifiers[value] != value
    if isinstance(value, dict):
        return any(_mentions_remapped_id(v, identifiers) for v in value.values())
    if isinstance(value, list):
        return any(_mentions_remapped_id(v, identifiers) for v in value)
    return False


def _unused_prefix(prefix: str, names: set[str]) -> str:
    candidate = prefix
    suffix = 0
    while any(name == candidate or name.startswith(candidate + "/") for name in names):
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


def _merge_result_sections(
    base: zipfile.ZipFile, target: zipfile.ZipFile,
    original_manifest: dict[str, Any], base_manifest: dict[str, Any],
    job: dict[str, Any], spec: dict[str, Any], sequence: int, record_id: str,
    written: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Keep the submitted graph fixed and relocate colliding result members.

    Legacy unreferenced jobs retain one structure, under its submitted id;
    their original geometry is attached to the run record. Explicit jobs keep
    their input geometry and receive a separate result structure.
    """
    original_sections = deepcopy(original_manifest.get("sections", []))
    input_structure = _job_structure_section(original_manifest, spec, job)
    implicit = "structure_ref" not in job and "structure_ref" not in spec
    result_sections = [
        deepcopy(s) for s in base_manifest.get("sections", [])
        if s.get("kind") not in ("job.spec", "run.record")
    ]
    result_structures = [s for s in result_sections if s.get("kind") == "structure"]
    if len(result_structures) != 1:
        raise ValueError("QVF finalization requires exactly one runner result structure")
    input_attachments = {}
    if implicit:
        # Prior results describe the prior geometry. Keep the historical
        # record and its attachments, rather than mislabel old results as
        # properties of the new sole structure.
        original_sections = [
            section for section in original_sections
            if section["kind"] in ("job.spec", "run.record", "preview")
            or (section["kind"].startswith("x_") and "run_ref" not in section)
        ]
        input_attachments = deepcopy(input_structure.get("members", {}))
    used_ids = {s["id"] for s in original_sections} | {record_id}
    identifiers = {}
    for section in result_sections:
        old = section["id"]
        if implicit and section["kind"] == "structure":
            new = input_structure["id"]
        else:
            new = old
            suffix = sequence
            while new in used_ids:
                new = f"result{suffix}_{old}"
                suffix += 1
        identifiers[old] = new
        used_ids.add(new)
    for section in base_manifest.get("sections", []):
        if section.get("kind") == "run.record":
            identifiers[section["id"]] = record_id
        elif section.get("kind") == "job.spec":
            identifiers[section["id"]] = job["id"]
    result_structure_id = identifiers[result_structures[0]["id"]]
    prefix = _unused_prefix(f"results/{sequence}", set(written))
    relocated = {}
    for section in result_sections:
        old_id = section["id"]
        section.update(_remap_result_references(section, identifiers))
        section["id"] = identifiers[old_id]
        section["run_ref"] = record_id
        if section["kind"] != "structure" and not section["kind"].startswith("x_"):
            section.setdefault("structure_ref", result_structure_id)
        for member in section.get("members", {}).values():
            path = member["path"]
            destination = f"{prefix}/{path}"
            member["path"] = destination
            if path in relocated:
                member.update(relocated[path])
                continue
            if member.get("format") == "json":
                raw = base.read(path)
                payload = json.loads(raw)
                if str(section.get("kind", "")).startswith("x_"):
                    if _mentions_remapped_id(payload, identifiers):
                        raise ValueError(
                            "QVF finalization cannot safely relocate references "
                            f"inside vendor result {old_id!r}"
                        )
                    updated = payload
                else:
                    updated = _remap_result_references(payload, identifiers)
                if updated != payload:
                    raw = safe_json_bytes(updated, indent=2)
                target.writestr(destination, raw, compress_type=zipfile.ZIP_DEFLATED)
                member["sha256"] = hashlib.sha256(raw).hexdigest()
                if "size" in member:
                    member["size"] = len(raw)
            else:
                info = copy(base.getinfo(path))
                info.filename = destination
                with base.open(path) as source, target.open(info, "w", force_zip64=True) as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
            written.add(destination)
            relocated[path] = {key: member[key] for key in ("path", "sha256", "size") if key in member}
    return original_sections + result_sections, identifiers, input_attachments


def _finalize_container(
    *,
    container: QvfJobContainer,
    original_path: Path,
    produced_qvf: Path,
    log_path: Path,
    status: str,
    started_utc: str,
    wall_seconds: float,
    attachments: dict[str, tuple[Path, str, str]],
) -> None:
    """Atomically merge the latest results with immutable job history.

    The original archive stays in a private sibling until this function
    validates the replacement. Large grids and logs are copied as streams,
    so completing a large QVF does not duplicate the whole archive in RAM.
    """
    if status not in ("converged", "failed"):
        raise ValueError(f"unsupported terminal QVF status {status!r}")

    result_manifest = _runner_result_manifest(produced_qvf)
    base_path = produced_qvf if result_manifest is not None else original_path
    if status == "converged" and result_manifest is None:
        raise ValueError(
            "QVF runner returned without a settled result archive "
            "containing its run.record"
        )

    target_path = container.path
    temp = _temporary_sibling(target_path, "final")
    try:
        with (
            zipfile.ZipFile(original_path, "r") as original,
            zipfile.ZipFile(base_path, "r") as base,
            zipfile.ZipFile(temp, "w") as target,
        ):
            original_manifest = json.loads(original.read("manifest.json"))
            base_manifest = json.loads(base.read("manifest.json"))

            original_job_specs = [
                section
                for section in original_manifest.get("sections", [])
                if section.get("kind") == "job.spec"
            ]
            if len(original_job_specs) != 1:
                raise ValueError(
                    "QVF finalization requires exactly one original "
                    f"job.spec section, found {len(original_job_specs)}"
                )
            job_spec_section = deepcopy(original_job_specs[0])
            prior_records = [
                deepcopy(section)
                for section in original_manifest.get("sections", [])
                if section.get("kind") == "run.record"
            ]
            result_records = [
                section
                for section in base_manifest.get("sections", [])
                if section.get("kind") == "run.record"
            ]

            written: set[str] = set()
            for info in original.infolist():
                if info.filename != "manifest.json":
                    _copy_zip_member(original, target, info.filename, written)

            sequence = _next_run_sequence(prior_records)
            section_ids = {s["id"] for s in original_manifest.get("sections", [])}
            record_id = f"run_record{sequence}"
            suffix = sequence
            while record_id in section_ids:
                suffix += 1
                record_id = f"run_record{suffix}"
            input_attachments: dict[str, Any] = {}
            identifiers: dict[str, str] = {}
            if result_manifest is not None:
                retained_sections, identifiers, input_attachments = _merge_result_sections(
                    base, target, original_manifest, base_manifest,
                    job_spec_section, container.spec, sequence, record_id, written,
                )
            else:
                retained_sections = deepcopy(original_manifest.get("sections", []))
            base_dir = _unused_prefix(f"run_record/{sequence}", written)
            spec_member = job_spec_section["members"]["spec"]
            spec_bytes = original.read(spec_member["path"])
            members: dict[str, Any] = {
                "input": _write_opaque_bytes(
                    target, f"{base_dir}/input.json", spec_bytes
                )
            }
            files_index: dict[str, Any] = {
                "input": {
                    "filename": Path(spec_member["path"]).name,
                    "description": "Executed declarative job specification.",
                    "media_type": "application/json",
                }
            }
            for role, member in input_attachments.items():
                member_role = f"attachment.input_structure_{role}"
                members[member_role] = {
                    "path": member["path"], "format": "binary",
                    "sha256": hashlib.sha256(original.read(member["path"])).hexdigest(),
                }
                files_index[member_role] = {
                    "filename": Path(member["path"]).name,
                    "description": "Submitted geometry for an unreferenced job.",
                    "media_type": "application/json" if member.get("format") == "json"
                    else "application/octet-stream",
                }
            if log_path.is_file():
                members["log"] = _write_opaque_file(
                    target, f"{base_dir}/log.txt", log_path
                )
            else:
                # A runner may fail before opening its output channel.
                # The complete log of that invocation is then the empty
                # byte string, not an absent canonical log member.
                members["log"] = _write_opaque_bytes(
                    target, f"{base_dir}/log.txt", b""
                )
            files_index["log"] = {
                "filename": log_path.name,
                "description": "Complete vibe-qc run log.",
                "media_type": "text/plain; charset=utf-8",
            }

            for role, (
                attachment_path,
                media_type,
                description,
            ) in attachments.items():
                member_role = f"attachment.{role}"
                archive_path = f"{base_dir}/attachments/{role}"
                members[member_role] = _write_opaque_file(
                    target, archive_path, attachment_path
                )
                files_index[member_role] = {
                    "filename": attachment_path.name,
                    "description": description,
                    "media_type": media_type,
                }

            files_bytes = safe_json_bytes(files_index, indent=2)
            files_path = f"{base_dir}/files.json"
            target.writestr(
                files_path,
                files_bytes,
                compress_type=zipfile.ZIP_DEFLATED,
            )
            members["files"] = {
                "path": files_path,
                "format": "json",
                "sha256": hashlib.sha256(files_bytes).hexdigest(),
            }

            template = (
                result_records[-1]
                if result_manifest is not None and result_records
                else {}
            )
            record: dict[str, Any] = {
                "id": record_id,
                "kind": "run.record",
                "job_ref": job_spec_section["id"],
                "run_status": status,
                "program": str(template.get("program") or "vibe-qc"),
                "exit_status": 0 if status == "converged" else 1,
                "started_utc": str(
                    template.get("started_utc") or started_utc
                ),
                "finished_utc": _utc_now(),
                "sequence": sequence,
                "members": members,
            }
            program_version = template.get("program_version")
            if program_version:
                record["program_version"] = str(program_version)

            manifest = deepcopy(original_manifest)
            _settle_withdrawn_version(manifest)
            manifest["sections"] = retained_sections + [record]
            if result_manifest is not None:
                for key, value in base_manifest.items():
                    if key not in {"sections", "metadata", "extensions", "viewer_defaults", "provenance"}:
                        if not key.startswith("x_"):
                            manifest[key] = deepcopy(value)
                extensions = manifest.setdefault("extensions", {})
                for namespace, declaration in base_manifest.get("extensions", {}).items():
                    if namespace in extensions and extensions[namespace] != declaration:
                        raise ValueError(
                            "QVF finalization has conflicting extension declarations "
                            f"for {namespace!r}"
                        )
                    extensions[namespace] = deepcopy(declaration)
                defaults = manifest.setdefault("viewer_defaults", {})
                result_defaults = base_manifest.get("viewer_defaults", {})
                for key, value in result_defaults.items():
                    if key != "auto_open":
                        value = deepcopy(value)
                        if key == "crossfade" and isinstance(value, dict):
                            for operand in ("a", "b"):
                                if isinstance(value.get(operand), str):
                                    value[operand] = identifiers.get(value[operand], value[operand])
                        defaults[identifiers.get(key, key)] = value
                result_structure_id = next(
                    identifiers[s["id"]] for s in base_manifest["sections"]
                    if s["kind"] == "structure"
                )
                ordered = [result_structure_id] + [
                    identifiers.get(i, i) for i in result_defaults.get("auto_open", [])
                ] + defaults.get("auto_open", [])
                current_ids = {s["id"] for s in retained_sections}
                defaults["auto_open"] = list(dict.fromkeys(i for i in ordered if i in current_ids))
                for key in list(defaults):
                    if key not in {"auto_open", "bookmarks", "camera", "crossfade"} and key not in current_ids:
                        del defaults[key]
                manifest.setdefault("provenance", {}).update(
                    deepcopy(base_manifest.get("provenance", {}))
                )
            manifest.setdefault("metadata", {})["modified_utc"] = _utc_now()
            provenance = manifest.setdefault("provenance", {})
            provenance["run_status"] = status
            provenance["wall_seconds"] = float(max(0.0, wall_seconds))
            provenance.pop("checkpoint", None)
            _write_manifest(target, manifest)

        _validate_qvf_or_raise(temp, action=f"{status} finalization")
        os.replace(temp, target_path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise

    if produced_qvf.absolute() != target_path.absolute():
        try:
            produced_qvf.unlink()
        except FileNotFoundError:
            pass


def run_container(
    path: Union[str, os.PathLike],
    *,
    force: bool = False,
    output: Union[str, os.PathLike, None] = None,
    update_in_place: bool = True,
    **overrides: Any,
):
    """Execute the job a QVF container describes, updating it in place.

    Refuses any container whose ``provenance.run_status`` is not
    ``"pending"`` unless ``force=True`` -- a settled archive is a record,
    and re-running it must be an explicit decision (a forced re-run
    carries the earlier ``run.record`` sections over, ordered by
    ``sequence``). ``output`` defaults to the container's stem
    (``job.qvf`` -> ``job.out`` siblings); ``overrides`` are extra
    runner keywords that win over spec options (local knobs like
    ``progress=``, never part of the container).

    After the runner returns, the same archive is updated in place
    (atomic write-new + rename): result sections and the ``run.record``
    are folded in, the original ``job.spec`` section is preserved
    verbatim, the embedded log is refreshed to the complete on-disk
    ``.out`` (including epilogues appended after the runner's own
    archive write), and ``provenance.run_status`` moves to
    ``"converged"`` / ``"failed"``. If the runner raises, the container
    is stamped ``"failed"`` and the exception propagates.
    ``update_in_place=False`` skips all of that and just runs the job.

    Returns the runner's result object.
    """
    container = load_job_container(path)
    if container.run_status != "pending" and not force:
        status = container.run_status or "not stated"
        raise ValueError(
            f"QVF container {container.path.name}: run_status is "
            f"{status!r}, not 'pending' -- this archive does not describe "
            f"a job waiting to run. Pass force=True (CLI: --force) to "
            f"re-run it anyway."
        )

    from . import run_job, run_periodic_job

    system = build_system(container)
    if container.job_type == "periodic":
        runner = run_periodic_job
    else:
        runner = run_job
    if output is None:
        output = container.path.with_suffix("")
    output = Path(os.fspath(output))

    def invoke_runner():
        # Hydration is part of the attempted execution: refusal settles the
        # archive as failed while preserving the submitted spec verbatim.
        kwargs = _runner_kwargs(container, runner)
        kwargs.update(overrides)
        if update_in_place:
            kwargs["output_qvf"] = True
        # ``null``, absent and the deprecated ``""`` all mean "supply no
        # basis set" (QVF spec Sec. 5.9).
        basis_name = job_spec_name(
            container.spec,
            "basis",
            source=f"QVF container {container.path.name}",
        )
        if container.job_type == "periodic":
            # run_periodic_job wants a BasisSet object (or None for
            # basis-free semiempirical methods), not a name string.
            if basis_name is None:
                basis = None
            else:
                from . import BasisSet

                basis = BasisSet(system.unit_cell_molecule(), basis_name)
            result = runner(
                system, basis, output=os.fspath(output), **kwargs
            )
            return result
        if basis_name is not None:
            kwargs["basis"] = basis_name
        return runner(system, output=os.fspath(output), **kwargs)

    if not update_in_place:
        return invoke_runner()

    # Match OutputPlan / write_qvf exactly: output is a path stem and each
    # artifact suffix is APPENDED to it (``stem_sibling``), so a stem that
    # carries a dot keeps its whole name -- see issue #254.
    produced = stem_sibling(output, ".qvf")
    log_path = stem_sibling(output, ".out")
    system_path = stem_sibling(output, ".system")
    backup_path = _temporary_sibling(container.path, "original")
    shutil.copyfile(container.path, backup_path)
    started_utc = _utc_now()
    started_monotonic = time.monotonic()
    finalized = False
    try:
        _mark_container_running(container.path)
        try:
            result = invoke_runner()
        except BaseException as run_error:
            try:
                _finalize_container(
                    container=container,
                    original_path=backup_path,
                    produced_qvf=produced,
                    log_path=log_path,
                    status="failed",
                    started_utc=started_utc,
                    wall_seconds=time.monotonic() - started_monotonic,
                    attachments=_run_attachment_paths(
                        output=output,
                        system_path=system_path,
                    ),
                )
                finalized = True
            except BaseException as final_error:
                run_error.add_note(
                    "QVF failed-state finalization also failed: "
                    f"{final_error!r}; the submitted archive was restored."
                )
            raise

        converged = getattr(result, "converged", None)
        status = (
            "failed"
            if converged is not None and not bool(converged)
            else "converged"
        )
        _finalize_container(
            container=container,
            original_path=backup_path,
            produced_qvf=produced,
            log_path=log_path,
            status=status,
            started_utc=started_utc,
            wall_seconds=time.monotonic() - started_monotonic,
            attachments=_run_attachment_paths(
                output=output,
                system_path=system_path,
            ),
        )
        finalized = True
        return result
    finally:
        if not finalized:
            os.replace(backup_path, container.path)
        else:
            try:
                backup_path.unlink()
            except FileNotFoundError:
                pass


def write_pending_qvf(
    system,
    stem: Union[str, os.PathLike],
    *,
    method: Optional[str] = None,
    basis: Optional[str] = None,
    functional: Optional[str] = None,
    kpoints: Optional[list] = None,
    tasks: Optional[list] = None,
    options: Optional[dict] = None,
) -> Path:
    """Write a *pending* QVF job container for ``system``.

    ``system`` is a :class:`Molecule` (molecular job) or
    :class:`PeriodicSystem` (periodic job); its ``charge`` /
    ``multiplicity`` are recorded in the job.spec. ``basis`` and
    ``functional`` follow QVF spec Sec. 5.9 (:func:`job_spec_method_fields`):
    a given name is recorded; for a ``method`` that takes no basis set
    (semiempirical, tight-binding, MLIP) or no functional (Hartree-Fock,
    correlated wavefunction methods) the spec records ``null``; otherwise an
    omitted input is left out, meaning the engine's default. The archive carries
    the structure, the declarative ``job.spec``, and
    ``provenance.run_status = "pending"`` -- ready for
    ``vibeqc run <stem>.qvf`` locally or through vq.
    """
    from .output.formats.qvf import write_qvf
    from .output.plan import OutputPlan

    periodic = hasattr(system, "unit_cell") or hasattr(system, "dim")
    spec: dict[str, Any] = {
        "job_type": "periodic" if periodic else "molecular",
    }
    if method is not None:
        spec["method"] = str(method)
    # ``basis: null`` / ``functional: null`` for a method that takes none,
    # the given name otherwise, absent for "the engine's default".
    spec.update(
        job_spec_method_fields(method, basis=basis, functional=functional)
    )
    charge = getattr(system, "charge", None)
    if charge is not None:
        spec["charge"] = int(charge)
    multiplicity = getattr(system, "multiplicity", None)
    if multiplicity is not None:
        spec["multiplicity"] = int(multiplicity)
    if kpoints is not None:
        if not periodic:
            raise ValueError(
                "write_pending_qvf: kpoints given for a molecular system"
            )
        spec["kpoints"] = [int(n) for n in kpoints]
    if tasks is not None:
        unknown = [t for t in tasks if str(t) not in _KNOWN_TASKS]
        if unknown:
            raise ValueError(
                f"write_pending_qvf: unknown tasks {unknown!r} "
                f"(understood: {', '.join(_KNOWN_TASKS)})"
            )
        spec["tasks"] = [str(t) for t in tasks]
    if options is not None:
        bad = sorted(set(options) & _RESERVED_OPTION_KEYS)
        if bad:
            raise ValueError(
                f"write_pending_qvf: options {bad!r} restate typed "
                f"job.spec fields -- pass them as keyword arguments"
            )
        # Guard against non-JSON-safe values in the container's options dict.
        # pybind11-wrapped native option classes cannot round-trip through
        # QVF's JSON serialisation path; embedding them would silently corrupt
        # the archive and cause aborts at read time (issue #366).
        for key, value in sorted(options.items()):
            if _is_json_unsafe(value):
                raise TypeError(
                    f"write_pending_qvf: options[{key!r}] is a "
                    f"{type(value).__name__} which cannot be embedded in a "
                    "QVF job container. Native option classes must be passed "
                    "as direct run_job() kwargs (e.g. run_job(mol, "
                    "rhf_options=opts)), not inside the ``options`` dict of "
                    "write_pending_qvf()."
                )
        spec["options"] = dict(options)

    # OutputPlan wants string method/basis for provenance labels; a
    # pending container may legitimately omit both.
    plan = OutputPlan.from_run_job_kwargs(
        output=stem,
        method=str(method) if method is not None else "auto",
        basis=str(basis) if basis else "",
        functional=str(functional) if functional else None,
        job_kind="periodic_scf" if periodic else "molecular_scf",
        output_qvf=True,
    )
    context: dict[str, Any] = {"job_spec": spec, "run_status": "pending"}
    if periodic:
        context["system"] = system
    else:
        context["molecule"] = system
    return write_qvf(stem, plan, **context)
