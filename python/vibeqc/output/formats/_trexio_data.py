"""General TREXIO data transport, using the installed library's public API.

Posenitskiy et al., J. Chem. Phys. 158, 174801 (2023),
doi:10.1063/5.0148161; https://trex-coe.github.io/trexio/trex.html.
No schema is vendored: fields and storage kinds follow the optional library.
Scientific conversion to vibe-qc objects lives in :mod:`.trexio`.
"""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass
class TrexioSparse:
    """Sparse tensor in TREXIO index order, without symmetry expansion.

    ``indices`` is an integer ``(nnz, rank)`` array, ``values`` is ``(nnz,)``.
    Two-electron quantities use physicists' indices, as specified by TREXIO.
    """

    indices: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        indices = np.asarray(self.indices)
        values = np.asarray(self.values)
        if indices.ndim != 2 or indices.dtype.kind not in "iu":
            raise ValueError("TREXIO sparse indices must be a 2-D integer array.")
        if np.any(indices < 0) or np.any(indices > np.iinfo(np.int32).max):
            raise ValueError("TREXIO sparse indices must be nonnegative int32 indices.")
        if values.ndim != 1 or len(values) != len(indices):
            raise ValueError("TREXIO sparse indices and values must have equal lengths.")
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError("TREXIO sparse values must be finite and real.")
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.values = np.ascontiguousarray(values, dtype=np.float64)

    @classmethod
    def from_dense(cls, array, *, threshold=0.0):
        """Store entries whose absolute value exceeds ``threshold``."""
        array = np.asarray(array)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Sparse threshold must be finite and nonnegative.")
        if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
            raise ValueError("TREXIO sparse values must be finite and real.")
        indices = np.argwhere(np.abs(array) > threshold)
        return cls(indices, array[tuple(indices.T)])


def _api(trexio):
    # read_*_size helpers have no has_* counterpart; has_<group> has no
    # corresponding reader. This intersection is exactly the field API.
    return {
        name[5:]: getattr(trexio, name)
        for name in dir(trexio)
        if name.startswith("read_") and hasattr(trexio, "has_" + name[5:])
    }


def _chunk_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("TREXIO chunk_size must be a positive integer.")
    return value


def _string_kind(reader):
    annotation = inspect.signature(reader).return_annotation
    return annotation if annotation in (str, list) else None


@lru_cache(maxsize=4)
def _c_library(path):
    import ctypes
    return ctypes.CDLL(path)


def _read_string_array(trexio, handle, name, size):
    # TREXIO 2.6's SWIG array wrapper allocates a fixed 4096-byte output,
    # irrespective of the array length: even many short MO classes can
    # overflow it. Call the exported C API with its required full buffer.
    import ctypes
    dimensions = {
        "metadata_author": "metadata_author_num", "metadata_code": "metadata_code_num",
        "nucleus_label": "nucleus_num", "mo_class": "mo_num", "mo_symmetry": "mo_num",
        "state_label": "state_num", "state_file_name": "state_num",
    }
    if name not in dimensions:
        raise ValueError(f"TREXIO: unknown string-array dimension for {name}.")
    count = int(getattr(trexio, "read_" + dimensions[name])(handle))
    if count <= 0:
        raise ValueError(f"TREXIO: invalid string-array dimension for {name}.")
    library = _c_library(trexio.pytr._pytrexio.__file__)
    reader = getattr(library, "trexio_read_" + name + "_low")
    reader.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    reader.restype = ctypes.c_int32
    buffer = ctypes.create_string_buffer(count * (size + 1) + 1)
    rc = reader(int(handle.pytrexio_s.this), buffer, size)
    return rc, buffer.value.decode("utf-8", "surrogateescape"), count


def _read_strings(trexio, handle, name, kind):
    # The high-level Python wrapper hard-codes a 2048-byte read buffer.
    # Use its public low-level binding with a growing buffer instead.
    reader = getattr(trexio.pytr, "trexio_read_" + name) if kind is str else None
    size = 2048
    while True:
        if kind is list:
            rc, value, count = _read_string_array(trexio, handle, name, size)
        else:
            rc, value = reader(handle.pytrexio_s, size)
        if rc != trexio.TREXIO_SUCCESS:
            raise trexio.Error(rc)
        values = value.split(trexio.pytr.TREXIO_DELIM)[:-1] if kind is list else [value]
        if kind is list and len(values) != count:
            raise ValueError(f"TREXIO: invalid delimiter or missing strings in {name}.")
        lengths = [len(s.encode("utf-8", "surrogateescape")) for s in values]
        if handle.back_end == trexio.TREXIO_TEXT and any(n >= 1023 for n in lengths):
            raise ValueError(f"TREXIO text backend may have truncated {name}; use HDF5 for long strings.")
        if all(n < size - 1 for n in lengths):
            return values if kind is list else value
        size *= 2


def _write_strings(trexio, handle, name, value, kind):
    values = list(value) if kind is list else [value]
    if not values or any(not isinstance(s, str) or "\0" in s for s in values):
        raise ValueError(f"{name} requires strings without NUL characters.")
    lengths = [len(s.encode("utf-8")) for s in values]
    if kind is list and any(not s or trexio.pytr.TREXIO_DELIM in s for s in values):
        raise ValueError(f"{name}: string-array entries must be nonempty and cannot contain the TREXIO newline delimiter.")
    if handle.back_end == trexio.TREXIO_TEXT and any(
            n >= 1023 or not s or s[0].isspace() or "\n" in s or "\r" in s
            for s, n in zip(values, lengths)):
        raise ValueError(f"{name}: the TREXIO text backend cannot preserve this string; use HDF5.")
    # The high-level wrapper counts characters, but the C API requires bytes.
    writer = getattr(trexio.pytr, "trexio_write_" + name)
    rc = writer(handle.pytrexio_s, values if kind is list else value, max(lengths) + 1)
    if rc != trexio.TREXIO_SUCCESS:
        raise trexio.Error(rc)


def read_fields(trexio, handle, *, fields=None, chunk_size=65536):
    """Read all present fields, or an explicit subset of API field names."""
    _chunk_size(chunk_size)
    api = _api(trexio)
    selected = api if fields is None else dict.fromkeys(fields)
    unknown = set(selected) - set(api)
    if unknown:
        raise ValueError(f"Unknown TREXIO fields: {sorted(unknown)}")
    result = {}
    for name in selected:
        if not getattr(trexio, "has_" + name)(handle):
            continue
        reader = api[name]
        kind = _string_kind(reader)
        if kind is not None:
            result[name] = _read_strings(trexio, handle, name, kind)
            continue
        if "offset_file" not in inspect.signature(reader).parameters:
            result[name] = reader(handle)
            continue
        size = (trexio.read_determinant_num(handle) if name == "determinant_list"
                else getattr(trexio, "read_" + name + "_size")(handle))
        chunks, indices = [], []
        sparse = "indices" in inspect.signature(getattr(trexio, "write_" + name)).parameters
        for offset in range(0, size, chunk_size):
            count = min(chunk_size, size - offset)
            part = reader(handle, offset, count)
            if int(part[-2]) != count:
                raise ValueError(f"Incomplete TREXIO field {name} at offset {offset}.")
            if sparse:
                indices.append(part[0])
                chunks.append(part[1])
            else:
                chunks.append(part[0])
        if not chunks:
            raise ValueError(f"TREXIO field {name} is present but empty.")
        values = np.concatenate(chunks)
        result[name] = TrexioSparse(np.concatenate(indices), values) if sparse else values
    return result


def write_fields(trexio, handle, fields: Mapping, *, chunk_size=65536):
    """Write a complete field mapping, dimensions before dependent arrays.

    TREXIO owns package_version, unsafe and determinant_num. The latter is
    checked against the written list; the former two describe the new file.
    """
    _chunk_size(chunk_size)
    api = _api(trexio)
    unknown = set(fields) - set(api)
    if unknown:
        raise ValueError(f"Unknown TREXIO fields: {sorted(unknown)}")
    readonly = {name for name in api if not hasattr(trexio, "write_" + name)}
    readonly |= {"metadata_package_version", "metadata_unsafe"}
    names = [name for name in fields if name not in readonly]

    def order(name):
        if len(inspect.signature(api[name]).parameters) == 1:
            return (0, name != "state_num", name)
        return (1 if name == "determinant_list" else 2, False, name)

    for name in sorted(names, key=order):
        writer = getattr(trexio, "write_" + name)
        value = fields[name]
        params = inspect.signature(writer).parameters
        try:
            kind = _string_kind(api[name])
            if kind is not None:
                _write_strings(trexio, handle, name, value, kind)
            elif "offset_file" in params:
                sparse = "indices" in params
                if sparse:
                    if not isinstance(value, TrexioSparse):
                        raise TypeError(f"{name} requires TrexioSparse indices and values.")
                    n = len(value.values)
                else:
                    value = np.asarray(value)
                    if np.iscomplexobj(value) or (value.dtype.kind == "f" and not np.all(np.isfinite(value))):
                        raise ValueError(f"{name} requires finite real values.")
                    if name == "determinant_list" and value.dtype.kind not in "iu":
                        raise ValueError("determinant_list requires integer bitfields.")
                    n = len(value)
                if n == 0:
                    raise ValueError(f"{name}: omit empty buffered/sparse fields.")
                for offset in range(0, n, chunk_size):
                    end = min(offset + chunk_size, n)
                    args = ((value.indices[offset:end], value.values[offset:end])
                            if sparse else (value[offset:end],))
                    writer(handle, offset, end - offset, *args)
            else:
                arr = np.asarray(value)
                if np.iscomplexobj(arr):
                    raise ValueError(f"{name}: split complex data into real and *_im fields.")
                if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
                    raise ValueError(f"{name} contains non-finite values.")
                writer(handle, arr.item() if arr.ndim == 0 else value)
        except Exception as exc:
            raise ValueError(f"Cannot write TREXIO field {name}: {exc}") from exc
    if "determinant_num" in fields:
        if (not trexio.has_determinant_num(handle)
                or trexio.read_determinant_num(handle) != fields["determinant_num"]):
            raise ValueError("determinant_num disagrees with determinant_list.")


def _check_target(trexio, path, overwrite):
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"write_trexio: {path} already exists.")
        if path.is_symlink():
            raise FileExistsError(f"write_trexio: refusing to replace symlink {path}.")
        if path.is_dir():
            # A directory of arbitrary .txt files is not proof of ownership.
            try:
                with trexio.File(str(path), "r", trexio.TREXIO_TEXT) as f:
                    if not trexio.has_metadata_package_version(f):
                        raise ValueError("missing TREXIO metadata")
                known = {name[7:] + ".txt" for name in dir(trexio)
                         if name.startswith("delete_")}
                known |= {name + ".txt" for name in _api(trexio)}
                known |= {name + ".txt.size" for name in _api(trexio)}
                if any(not p.is_file() or p.is_symlink() or p.name not in known | {".lock"}
                       for p in path.iterdir()):
                    raise ValueError("unrecognized directory contents")
            except Exception as exc:
                raise FileExistsError(
                    f"write_trexio: {path} does not look like a TREXIO text artefact."
                ) from exc


@contextmanager
def atomic_target(trexio, target, *, overwrite):
    """Stage a sibling artefact, then replace; roll back a directory rename."""
    target = Path(target)
    _check_target(trexio, target, overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    temporary, backup = staging / "data", staging / "previous"
    try:
        yield temporary
        _check_target(trexio, target, overwrite)
        if target.is_dir() or (target.exists() and temporary.is_dir()):
            target.rename(backup)
            try:
                temporary.rename(target)
            except BaseException:
                backup.rename(target)
                raise
        elif overwrite:
            os.replace(temporary, target)
        elif temporary.is_dir():
            temporary.rename(target)
        else:
            # Atomic no-clobber installation of a regular file.
            os.link(temporary, target)
    finally:
        shutil.rmtree(staging)
