"""Immutable, byte-identified materializations for filtered named bases."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

_RECEIPT = "filtered-basis-receipt.json"
_KIND = "vibeqc.filtered-basis-root.v1"
_MAX_FILE_BYTES = 64 * 1024**2
_MAX_LIBRARY_BYTES = 256 * 1024**2
_MAX_FILES = 10000


def _payload(directory: Path, *, allow_links: bool) -> dict[str, bytes]:
    entries = sorted(directory.iterdir())
    if len(entries) > _MAX_FILES:
        raise RuntimeError("filtered basis library exceeds the file-count bound")
    result = {}
    folded = set()
    total = 0
    for entry in entries:
        if entry.name.casefold() in folded:
            raise RuntimeError(f"case-fold collision in basis library: {entry.name}")
        folded.add(entry.name.casefold())
        if entry.is_symlink() and not allow_links:
            raise RuntimeError(f"symlink in immutable filtered basis cache: {entry.name}")
        status = entry.stat()
        if not stat.S_ISREG(status.st_mode) or status.st_size > _MAX_FILE_BYTES:
            raise RuntimeError(f"unsafe or oversized basis library entry: {entry.name}")
        with entry.open("rb") as stream:
            data = stream.read(_MAX_FILE_BYTES + 1)
        total += len(data)
        if len(data) > _MAX_FILE_BYTES or total > _MAX_LIBRARY_BYTES:
            raise RuntimeError("filtered basis library exceeds the byte bound")
        result[entry.name] = data
    return result


def _identity(payload: dict[str, bytes]) -> tuple[str, dict[str, str]]:
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return key, hashes


def validate_root(root: Path, expected: dict[str, bytes] | None = None) -> dict[str, bytes]:
    """Refuse changed, extra, missing or linked cache content before reuse."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("unsafe filtered basis cache root")
    if {p.name for p in root.iterdir()} != {"basis", _RECEIPT}:
        raise RuntimeError("filtered basis cache root has unexpected entries")
    if (root / "basis").is_symlink() or (root / _RECEIPT).is_symlink():
        raise RuntimeError("symlink in filtered basis cache root")
    receipt_status = (root / _RECEIPT).stat()
    if not stat.S_ISREG(receipt_status.st_mode) or receipt_status.st_size > 8 * 1024**2:
        raise RuntimeError("unsafe or oversized filtered basis receipt")
    try:
        with (root / _RECEIPT).open("rb") as stream:
            receipt_bytes = stream.read(8 * 1024**2 + 1)
        if len(receipt_bytes) > 8 * 1024**2:
            raise RuntimeError("oversized filtered basis receipt")
        receipt = json.loads(receipt_bytes)
        payload = _payload(root / "basis", allow_links=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError("unreadable filtered basis cache") from exc
    key, hashes = _identity(payload)
    if receipt != {"kind": _KIND, "key": key, "files": hashes} or root.name != key:
        raise RuntimeError("filtered basis cache content hash mismatch")
    if expected is not None and payload != expected:
        raise RuntimeError("filtered basis cache differs from requested materialization")
    return payload


def materialize(cache: Path, source: Path, filename: str, text: str) -> Path:
    """Publish a complete stock-plus-filtered library without source writes."""
    cache, source = cache.expanduser().resolve(), source.expanduser().resolve()
    package = Path(__file__).resolve().parent
    repository = package.parent.parent if package.parent.name == "python" else package
    protected_source = source.parent if source.name == "basis" else source
    if cache.is_relative_to(protected_source) or cache.is_relative_to(package) or cache.is_relative_to(repository):
        raise RuntimeError("filtered basis cache must be outside the installed library and package")
    if filename != Path(filename).name or not filename.endswith(".g94") or filename in (".", ".."):
        raise ValueError("filtered basis filename must be a plain .g94 basename")
    if source.name == "basis" and (
        (source.parent / _RECEIPT).exists()
        or (source.parent / _RECEIPT).is_symlink()
        or re.fullmatch(r"[0-9a-f]{64}", source.parent.name) is not None
    ):
        source = source.parent
    basis_dir = source / "basis" if (source / "basis").is_dir() else source
    receipt_path = source / _RECEIPT
    owned_source = (receipt_path.exists() or receipt_path.is_symlink()
                    or source.is_relative_to(cache)
                    or re.fullmatch(r"[0-9a-f]{64}", source.name) is not None)
    payload = (validate_root(source) if owned_source
               else _payload(basis_dir, allow_links=True))
    data = text.encode()
    for name, previous in payload.items():
        if name.casefold() == filename.casefold() and (name != filename or previous != data):
            raise RuntimeError(f"filtered basis name conflicts with existing library file: {filename}")
    if len(data) > _MAX_FILE_BYTES:
        raise RuntimeError("filtered basis file exceeds the byte bound")
    payload[filename] = data
    if len(payload) > _MAX_FILES or sum(map(len, payload.values())) > _MAX_LIBRARY_BYTES:
        raise RuntimeError("filtered basis materialization exceeds the size bound")
    key, hashes = _identity(payload)
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / key
    if root.exists() or root.is_symlink():
        validate_root(root, payload)
        return root
    temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=cache))
    try:
        (temporary / "basis").mkdir()
        for name, content in payload.items():
            (temporary / "basis" / name).write_bytes(content)
        (temporary / _RECEIPT).write_text(json.dumps({"kind": _KIND, "key": key, "files": hashes}, sort_keys=True))
        try:
            temporary.rename(root)
        except OSError:
            if not root.exists() and not root.is_symlink():
                raise
        validate_root(root, payload)
        return root
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def purge(cache: Path, selected: Path | None) -> int:
    """Explicit purge after the caller has stopped all readers and writers."""
    cache = cache.expanduser().resolve()
    package = Path(__file__).resolve().parent
    repository = package.parent.parent if package.parent.name == "python" else package
    if cache.is_relative_to(repository):
        raise RuntimeError("cannot purge a cache inside the installed package or repository")
    if not cache.exists():
        return 0
    roots = sorted(cache.iterdir())
    removed = 0
    for root in roots:
        if not re.fullmatch(r"[0-9a-f]{64}", root.name):
            raise RuntimeError("filtered basis cache contains unknown or active staging entries")
        if selected is not None and selected.resolve().is_relative_to(root):
            raise RuntimeError("cannot clear the selected filtered basis root; stop all readers and restore LIBINT_DATA_PATH first")
        payload = validate_root(root)
        removed += sum(name.startswith("_vibeqc_filtered_") and name.endswith(".g94") for name in payload)
    # Validation of every owned root precedes deletion. Normal filtering never
    # calls purge; explicit callers must ensure no other process uses the roots.
    for root in roots:
        shutil.rmtree(root)
    return removed
