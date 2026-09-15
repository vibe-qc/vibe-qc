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
from .output._text_safety import safe_json_bytes
from .output._stem_paths import stem_sibling

__all__ = [
    "QvfJobContainer",
    "load_job_container",
    "build_system",
    "run_container",
    "write_pending_qvf",
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


def load_job_container(path: Union[str, os.PathLike]) -> QvfJobContainer:
    """Parse ``path`` into a :class:`QvfJobContainer`.

    Requires exactly one ``job.spec`` section and exactly one
    ``structure`` section; verifies both members against their declared
    sha256 before use. Raises :class:`ValueError` on anything malformed.
    """
    p = Path(os.fspath(path))
    with zipfile.ZipFile(p, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        sections = manifest.get("sections", [])
        job_specs = [s for s in sections if s.get("kind") == "job.spec"]
        structures = [s for s in sections if s.get("kind") == "structure"]
        if len(job_specs) != 1:
            raise ValueError(
                f"QVF container {p.name}: expected exactly one job.spec "
                f"section, found {len(job_specs)} -- not a runnable job "
                f"container"
            )
        if len(structures) != 1:
            raise ValueError(
                f"QVF container {p.name}: expected exactly one structure "
                f"section, found {len(structures)}"
            )
        spec_bytes = _read_member(
            zf, job_specs[0], "spec", kind="job.spec"
        )
        spec = json.loads(spec_bytes)
        structure = json.loads(
            _read_member(zf, structures[0], "structure", kind="structure")
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
    spec § 5.9: a runner MUST refuse rather than silently drop).
    """
    spec = container.spec
    kwargs: dict[str, Any] = {}
    if spec.get("method") is not None:
        kwargs["method"] = str(spec["method"])
    if spec.get("functional") is not None:
        kwargs["functional"] = str(spec["functional"])
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
    options = spec.get("options") or {}
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
        kwargs[key] = value
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

            excluded_base_paths: set[str] = set()
            retained_sections: list[dict[str, Any]] = []
            for section in base_manifest.get("sections", []):
                if section.get("kind") in ("job.spec", "run.record"):
                    excluded_base_paths.update(
                        _section_member_paths(section)
                    )
                    continue
                retained_sections.append(deepcopy(section))

            written: set[str] = set()
            for info in base.infolist():
                if (
                    info.filename == "manifest.json"
                    or info.filename in excluded_base_paths
                ):
                    continue
                _copy_zip_member(base, target, info.filename, written)

            for section in prior_records:
                for member_path in _section_member_paths(section):
                    _copy_zip_member(
                        original, target, member_path, written
                    )
            for member_path in _section_member_paths(job_spec_section):
                _copy_zip_member(original, target, member_path, written)

            sequence = _next_run_sequence(prior_records)
            base_dir = f"run_record/{sequence}"
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
            section_ids = {
                str(section.get("id"))
                for section in retained_sections
                + prior_records
                + [job_spec_section]
            }
            record_id = f"run_record{sequence}"
            suffix = sequence
            while record_id in section_ids:
                suffix += 1
                record_id = f"run_record{suffix}"
            record: dict[str, Any] = {
                "id": record_id,
                "kind": "run.record",
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

            manifest = deepcopy(base_manifest)
            manifest["sections"] = (
                retained_sections
                + [job_spec_section]
                + prior_records
                + [record]
            )
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
    kwargs = _runner_kwargs(container, runner)
    kwargs.update(overrides)
    if update_in_place:
        # The runner-produced QVF supplies the latest result sections for
        # the terminal merge; a container execution cannot opt out of it.
        kwargs["output_qvf"] = True
    if output is None:
        output = container.path.with_suffix("")
    output = Path(os.fspath(output))

    def invoke_runner():
        if container.job_type == "periodic":
            # run_periodic_job wants a BasisSet object (or None for
            # basis-free semiempirical methods), not a name string.
            basis_name = container.spec.get("basis")
            if basis_name is None:
                basis = None
            else:
                from . import BasisSet

                basis = BasisSet(
                    system.unit_cell_molecule(), str(basis_name)
                )
            result = runner(
                system, basis, output=os.fspath(output), **kwargs
            )
            return result
        if container.spec.get("basis") is not None:
            kwargs["basis"] = str(container.spec["basis"])
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
    ``multiplicity`` are recorded in the job.spec. The archive carries
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
    if basis is not None:
        spec["basis"] = str(basis)
    if functional is not None:
        spec["functional"] = str(functional)
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
        spec["options"] = dict(options)

    # OutputPlan wants string method/basis for provenance labels; a
    # pending container may legitimately omit both.
    plan = OutputPlan.from_run_job_kwargs(
        output=stem,
        method=str(method) if method is not None else "auto",
        basis=str(basis) if basis is not None else "",
        functional=str(functional) if functional is not None else None,
        job_kind="periodic_scf" if periodic else "molecular_scf",
        output_qvf=True,
    )
    context: dict[str, Any] = {"job_spec": spec, "run_status": "pending"}
    if periodic:
        context["system"] = system
    else:
        context["molecule"] = system
    return write_qvf(stem, plan, **context)
