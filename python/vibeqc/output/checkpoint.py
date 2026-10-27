"""Opt-in live QVF checkpointing for :func:`vibeqc.run_job` /
:func:`vibeqc.run_periodic_job`.

For live job visualization, vibe-view wants to hot-reload a QVF *as a
calculation progresses* -- SCF convergence climbing, an optimization
trajectory growing, the geometry morphing toward the relaxed structure.
The runners normally write the QVF once, at the end. This module adds a
small handle that a runner drives to also refresh a *checkpoint* QVF on a
cadence, each snapshot carrying:

* ``provenance.run_status = "running"`` while the job is in flight, then
  ``"converged"`` (success) or ``"failed"`` (crash) at the end;
* a monotonic ``provenance.checkpoint.seq`` so a reader can order snapshots
  and tell a fresh one from a stale one without diffing bytes;
* ``partial: true`` on any still-growing section (e.g. an optimization
  ``trajectory``).

Every checkpoint write goes through :func:`vibeqc.output.formats.qvf.write_qvf`
with ``atomic=True`` (temp file + ``os.replace``) so a concurrent reader
never observes a half-written zip.

These provenance/section fields validate against the *current* QVF v1
manifest schema (``provenance`` is ``additionalProperties: true`` and
sections carry no ``additionalProperties: false``), so checkpointing does
not depend on the QVF v3 schema landing first.

Design contract
---------------

A live QVF checkpoint write must **never** take down the calculation it is
observing. Every write is wrapped so a filesystem hiccup or a transient
writer error is surfaced as a warning and swallowed -- the SCF keeps
running and the authoritative end-of-run output QVF is unaffected.

The numeric ``.vqr`` restart APIs below are separate authoritative solver
state. They use the same atomic-publication convention but propagate I/O and
integrity errors, so a failed requested restart save is never reported as valid.
A numeric restart archive is not a QVF visualization document.
"""

from __future__ import annotations

import datetime as _dt
import os
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Union


def _now_iso() -> str:
    """UTC timestamp, ISO-8601 with a trailing ``Z``."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class QvfCheckpointer:
    """Drive periodic + terminal QVF snapshots to a checkpoint path.

    Parameters
    ----------
    path
        Where to write the checkpoint archive (the ``.qvf`` suffix is
        appended if absent, matching :func:`write_qvf`). ``None`` disables
        checkpointing entirely -- every method becomes a cheap no-op.
    every
        Refresh the checkpoint every ``every`` SCF / optimization
        iterations. ``0`` (or negative) disables the *cadence* snapshots;
        :meth:`snapshot` (initial frame) and :meth:`finalize` (terminal
        frame) still fire when a non-``None`` ``path`` is given, so a job
        with ``every=0`` but a ``path`` gets a start + end frame only.
    plan
        The :class:`~vibeqc.output.plan.OutputPlan` for the job, forwarded
        to :func:`write_qvf`.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike, None],
        every: int,
        *,
        plan: Any,
    ) -> None:
        self._path: Optional[Path] = Path(os.fspath(path)) if path else None
        self._every = int(every) if every else 0
        self._plan = plan
        self._seq = 0
        self._t0 = time.monotonic()

    # ----- state ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when a checkpoint ``path`` was configured."""
        return self._path is not None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def seq(self) -> int:
        """The last sequence number written (0 before the first write)."""
        return self._seq

    def due(self, iteration: int) -> bool:
        """Whether a cadence snapshot is due at ``iteration``.

        False when checkpointing is disabled, ``every`` is 0, or the
        iteration is not a positive multiple of ``every``.
        """
        if not self.enabled or self._every <= 0:
            return False
        return iteration > 0 and iteration % self._every == 0

    # ----- writes --------------------------------------------------------

    def snapshot(
        self,
        *,
        scf_iteration: Optional[int] = None,
        energy_eh: Optional[float] = None,
        partial_sections: Any = None,
        **context: Any,
    ) -> None:
        """Write a running ("in flight") snapshot.

        Use for the initial geometry frame and for each cadence hit. The
        caller supplies whatever section context is available so far
        (``molecule=`` / ``system=`` at minimum; ``result=``,
        ``trajectory_frames=``, ``scf_history_data=`` when they exist).
        ``partial_sections`` flags still-growing sections (see
        :func:`write_qvf`).
        """
        if not self.enabled:
            return
        self._write(
            "running",
            scf_iteration=scf_iteration,
            energy_eh=energy_eh,
            partial_sections=partial_sections,
            **context,
        )

    def maybe_snapshot(
        self,
        iteration: int,
        *,
        energy_eh: Optional[float] = None,
        partial_sections: Any = None,
        **context: Any,
    ) -> bool:
        """Write a running snapshot only if :meth:`due` at ``iteration``.

        Returns ``True`` if a snapshot was written. Convenience wrapper for
        the SCF / optimization loop hooks.
        """
        if not self.due(iteration):
            return False
        self.snapshot(
            scf_iteration=iteration,
            energy_eh=energy_eh,
            partial_sections=partial_sections,
            **context,
        )
        return True

    def finalize(self, status: str, **context: Any) -> None:
        """Write the terminal snapshot with ``run_status=status``.

        ``status`` is ``"converged"`` (success) or ``"failed"`` (the job
        raised -- mirrors :meth:`ManifestUpdater.crash`). Always writes when
        checkpointing is enabled, regardless of ``every``, so the final
        checkpoint archive is always labeled and readers can stop watching.
        """
        if not self.enabled:
            return
        if status not in ("converged", "failed"):
            raise ValueError(
                "QvfCheckpointer.finalize: status must be 'converged' or "
                f"'failed', got {status!r}."
            )
        self._write(status, partial_sections=None, **context)

    # ----- internals -----------------------------------------------------

    def _write(
        self,
        status: str,
        *,
        scf_iteration: Optional[int] = None,
        energy_eh: Optional[float] = None,
        partial_sections: Any = None,
        **context: Any,
    ) -> None:
        # Import lazily so importing this module doesn't pull the (heavy)
        # QVF writer + its optional native deps before they're needed.
        from .formats.qvf import write_qvf

        self._seq += 1
        checkpoint: dict[str, Any] = {
            "seq": self._seq,
            "wall_time_s": time.monotonic() - self._t0,
            "written_at": _now_iso(),
        }
        if scf_iteration is not None:
            checkpoint["scf_iteration"] = int(scf_iteration)
        if energy_eh is not None:
            checkpoint["energy_eh"] = float(energy_eh)

        try:
            write_qvf(
                self._path,
                self._plan,
                atomic=True,
                run_status=status,
                checkpoint=checkpoint,
                partial_sections=partial_sections,
                **context,
            )
        except Exception as exc:  # noqa: BLE001 -- never tank the job
            # A checkpoint is a best-effort visualization aid. Surface the
            # failure but keep the calculation (and its authoritative
            # end-of-run output QVF) going.
            warnings.warn(
                f"QVF checkpoint write to {self._path} failed "
                f"(seq={self._seq}, status={status}): {exc!r}. The "
                f"calculation continues; the final output QVF is "
                f"unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Roll the seq back so the next successful write is still
            # monotonic-without-gaps from a reader's point of view.
            self._seq -= 1


class _CheckpointingProgressProxy:
    """Transparent proxy over a :class:`ProgressLogger`.

    Forwards every attribute to the wrapped logger, but on each
    ``iteration(n, **fields)`` call also invokes ``on_iteration(n, fields)``
    after the real log line is emitted. Used by the periodic runner so a
    checkpoint snapshot can fire as each Python-driven SCF cycle lands,
    without any driver having to know about checkpointing. The callback is
    fully guarded -- a checkpoint error can never disturb the SCF's own
    progress logging.
    """

    def __init__(self, inner: Any, on_iteration: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_on_iteration", on_iteration)

    def iteration(self, n: int, **fields: Any) -> None:
        self._inner.iteration(n, **fields)
        try:
            self._on_iteration(n, fields)
        except Exception:  # noqa: BLE001 -- never disturb the SCF
            pass

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the proxy itself.
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


def wrap_progress_for_checkpoints(plog: Any, on_iteration: Any) -> Any:
    """Return ``plog`` wrapped so per-iteration checkpoints fire.

    ``on_iteration`` is called as ``on_iteration(n, fields)`` after each
    ``plog.iteration(n, **fields)``; ``fields`` is the dict the driver
    passed (typically containing ``energy``). Returns the original ``plog``
    unchanged when ``on_iteration`` is falsy.
    """
    if not on_iteration:
        return plog
    return _CheckpointingProgressProxy(plog, on_iteration)


def normalize_partial_iterable(spec: Any) -> Optional[Union[bool, set]]:
    """Coerce a ``partial_sections`` argument to the shape write_qvf wants.

    ``None`` / falsy -> ``None``; ``True`` -> ``True``; an iterable of
    strings -> a ``set`` of them. Kept here so both runners share one
    interpretation.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        return True
    if isinstance(spec, str):
        return {spec}
    if isinstance(spec, Iterable):
        return {str(x) for x in spec}
    return None


# Numeric restart archives share the QVF atomic-publication convention, but are
# deliberately not QVF visualization documents. Unknown versions have no implicit
# migration. Callers own and validate their method-specific metadata schema.
_RESTART_SCHEMA = 'vibeqc.restart.archive/v1'
_RESTART_SCHEMA_V2 = 'vibeqc.restart.archive/v2'
_RESTART_CHUNK_BYTES = 1024 * 1024
_RESTART_MANIFEST_LIMIT = 32 * 1024 * 1024
_RESTART_METADATA_INLINE_LIMIT = 1024 * 1024


def _restart_json(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _restart_chunks(array):
    """At most one MiB of staging, including for noncontiguous inputs."""
    import numpy as np
    for panel in np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                           op_flags=['readonly'], order='C',
                           buffersize=max(1, _RESTART_CHUNK_BYTES // array.dtype.itemsize)):
        yield panel.tobytes(order='C')


def write_restart_state(path, metadata: dict, arrays: dict):
    """Atomically publish finite JSON and numeric arrays at the exact path.

    The conventional suffix is ``.vqr``. No pickle, object arrays, executable
    types or compression are used. A failed write preserves the last complete
    archive. Numeric staging is bounded by one MiB; retained input arrays,
    solver state, JSON metadata and archive bookkeeping are additional memory.
    I/O errors propagate: a requested restart checkpoint is authoritative state,
    unlike the best-effort live QVF visualization above.
    """
    import hashlib
    import tempfile
    import zipfile
    import numpy as np
    if not isinstance(metadata, dict) or not isinstance(arrays, dict):
        raise TypeError('restart metadata and arrays must be dictionaries')
    metadata_bytes = _restart_json(metadata)
    prepared = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('restart array names must be nonempty portable identifiers')
        a = np.asarray(value)
        if a.dtype.kind not in 'biufc' or a.dtype.hasobject or a.dtype.fields or a.dtype.itemsize not in (1, 2, 4, 8, 16):
            raise ValueError('restart arrays require plain numeric dtypes')
        # Validation is panelled too; do not allocate an array-sized boolean mask.
        for chunk in _restart_chunks(a):
            if not np.isfinite(np.frombuffer(chunk, dtype=a.dtype)).all():
                raise ValueError('restart arrays must be finite')
        prepared[name] = a
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(descriptor, 'w+b') as raw:
            with zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                records = {}
                for name, a in sorted(prepared.items()):
                    checksum = hashlib.sha256()
                    with archive.open('arrays/' + name, 'w', force_zip64=True) as stream:
                        for chunk in _restart_chunks(a):
                            checksum.update(chunk)
                            stream.write(chunk)
                    records[name] = dict(shape=list(a.shape), dtype=a.dtype.str,
                                         nbytes=a.nbytes, sha256=checksum.hexdigest())
                # Executed support can exceed the bounded manifest even for a
                # small orbital basis. Keep its complete JSON as data, without
                # truncating source identity or relaxing the manifest limit.
                schema, stored_metadata = _RESTART_SCHEMA, metadata
                if len(metadata_bytes) > _RESTART_METADATA_INLINE_LIMIT:
                    schema = _RESTART_SCHEMA_V2
                    with archive.open('metadata.json', 'w', force_zip64=True) as stream:
                        view = memoryview(metadata_bytes)
                        for start in range(0, len(view), _RESTART_CHUNK_BYTES):
                            stream.write(view[start:start + _RESTART_CHUNK_BYTES])
                    stored_metadata = dict(nbytes=len(metadata_bytes),
                        sha256=hashlib.sha256(metadata_bytes).hexdigest())
                payload = dict(schema=schema, metadata=stored_metadata, arrays=records)
                payload['sha256'] = hashlib.sha256(_restart_json(payload)).hexdigest()
                encoded = _restart_json(payload)
                if len(encoded) > _RESTART_MANIFEST_LIMIT:
                    raise ValueError('restart manifest exceeds 32 MiB limit')
                archive.writestr('manifest.json', encoded)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def read_restart_state(path):
    """Read supported v1/v2 archives, without executable deserialization.

    Arrays allocate their declared final storage once and stream into it in
    one-MiB panels. Reading returns all retained arrays; this is not a claim of
    constant total memory. Declared byte counts must match the actual archive
    members before allocation. Unknown envelope versions are refused.
    """
    import hashlib
    import json
    import math
    import zipfile
    import numpy as np
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate restart metadata key')
            result[key] = value
        return result
    try:
        archive_bytes = Path(path).stat().st_size
        with zipfile.ZipFile(path, 'r') as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('duplicate restart archive entry')
            info = archive.getinfo('manifest.json')
            if info.file_size > _RESTART_MANIFEST_LIMIT:
                raise ValueError('restart manifest exceeds size limit')
            if (info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size
                    or info.file_size > archive_bytes):
                raise ValueError('restart manifest must be stored uncompressed')
            manifest = json.loads(archive.read(info), object_pairs_hook=unique_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite restart metadata')))
            if not isinstance(manifest, dict) or manifest.get('schema') not in (_RESTART_SCHEMA, _RESTART_SCHEMA_V2):
                raise ValueError('unsupported restart archive schema; no implicit migration')
            checksum = manifest.pop('sha256', None)
            if set(manifest) != {'schema', 'metadata', 'arrays'} or hashlib.sha256(_restart_json(manifest)).hexdigest() != checksum:
                raise ValueError('restart manifest checksum or structure mismatch')
            if not isinstance(manifest['metadata'], dict) or not isinstance(manifest['arrays'], dict):
                raise ValueError('invalid restart metadata or array table')
            members = {'manifest.json', *('arrays/' + n for n in manifest['arrays'])}
            if manifest['schema'] == _RESTART_SCHEMA_V2:
                members.add('metadata.json')
            if set(names) != members:
                raise ValueError('restart archive array coverage mismatch')
            metadata = manifest['metadata']
            if manifest['schema'] == _RESTART_SCHEMA_V2:
                if set(metadata) != {'nbytes', 'sha256'}:
                    raise ValueError('invalid restart metadata descriptor')
                item = archive.getinfo('metadata.json')
                count = metadata['nbytes']
                if (type(count) is not int or count < 0 or count > archive_bytes
                        or item.file_size != count or item.compress_size != count
                        or item.compress_type != zipfile.ZIP_STORED):
                    raise ValueError('restart metadata byte count mismatch')
                # Validate actual stored size before allocation, and integrity
                # before parsing. JSON and decoded records remain resident;
                # only I/O staging is bounded by the panel size.
                data = bytearray(count)
                digest = hashlib.sha256()
                with archive.open(item) as stream:
                    for start in range(0, count, _RESTART_CHUNK_BYTES):
                        chunk = stream.read(min(_RESTART_CHUNK_BYTES, count - start))
                        if len(chunk) != min(_RESTART_CHUNK_BYTES, count - start):
                            raise ValueError('truncated restart metadata')
                        digest.update(chunk)
                        data[start:start + len(chunk)] = chunk
                if digest.hexdigest() != metadata['sha256']:
                    raise ValueError('restart metadata checksum mismatch')
                metadata = json.loads(data, object_pairs_hook=unique_pairs,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite restart metadata')))
                del data
                if not isinstance(metadata, dict):
                    raise ValueError('invalid restart metadata object')
                # Also reject overflowed finite JSON numbers such as 1e999.
                _restart_json(metadata)
            arrays = {}
            for name, record in manifest['arrays'].items():
                if not isinstance(record, dict) or set(record) != {'shape', 'dtype', 'nbytes', 'sha256'}:
                    raise ValueError('invalid restart array descriptor')
                shape = record['shape']
                if not isinstance(shape, list) or len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError('invalid restart array shape')
                dtype = np.dtype(record['dtype'])
                if dtype.kind not in 'biufc' or dtype.hasobject or dtype.fields or dtype.itemsize not in (1, 2, 4, 8, 16):
                    raise ValueError('unsafe restart array dtype')
                count = math.prod(shape) * dtype.itemsize
                item = archive.getinfo('arrays/' + name)
                if (type(record['nbytes']) is not int or record['nbytes'] != count or item.file_size != count or item.compress_size != count
                        or count > archive_bytes or item.compress_type != zipfile.ZIP_STORED):
                    raise ValueError('restart array shape/dtype/byte count mismatch')
                a = np.empty(shape, dtype=dtype)
                view = memoryview(a.reshape(-1)).cast('B')
                digest = hashlib.sha256()
                with archive.open(item) as stream:
                    for start in range(0, count, _RESTART_CHUNK_BYTES):
                        chunk = stream.read(min(_RESTART_CHUNK_BYTES, count - start))
                        if len(chunk) != min(_RESTART_CHUNK_BYTES, count - start):
                            raise ValueError('truncated restart array')
                        digest.update(chunk)
                        view[start:start + len(chunk)] = chunk
                if digest.hexdigest() != record['sha256']:
                    raise ValueError('restart array checksum mismatch')
                for chunk in _restart_chunks(a):
                    if not np.isfinite(np.frombuffer(chunk, dtype=dtype)).all():
                        raise ValueError('nonfinite restart array')
                a.setflags(write=False)
                arrays[name] = a
            return metadata, arrays
    except (OSError, EOFError, KeyError, TypeError, OverflowError, UnicodeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid or truncated restart archive: {exc}') from exc
