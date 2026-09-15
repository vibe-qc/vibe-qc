"""QVF-Basis serializer/deserializer -- text mode (``.qvf.json``) and
packaged mode (``.qvf``).

The canonical data model lives in :mod:`.model`; validation is delegated
to :func:`.validator.validate_basis_dict`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .model import BasisSetData, Reference
from .validator import validate_basis_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(data: bytes) -> str:
    """Return the sha256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _build_manifest(
    data: BasisSetData,
    *,
    program: str = "vibe-qc",
    version: str = "0.14.0",
    basis_json_bytes: bytes | None = None,
    refs_json_bytes: bytes | None = None,
) -> Dict[str, Any]:
    """Build the QVF manifest dict.

    When *basis_json_bytes* / *refs_json_bytes* are provided the
    corresponding sha256 hashes are embedded in the manifest.  Callers
    that only need the shape for inspection can omit them (hashes will
    be absent).
    """
    manifest: Dict[str, Any] = {
        "qvf_version": 1,
        "qvf_profile": "basis",
        "profile_version": "1.0",
        "source": {
            "program": program,
            "version": version,
        },
        "sections": [],
    }

    # --- basis section -------------------------------------------------
    basis_section: Dict[str, Any] = {
        "id": "basis",
        "kind": "basis",
        "members": {
            "basis": {
                "path": "basis/basis.qvf.json",
                "format": "json",
            }
        },
    }
    if basis_json_bytes is not None:
        basis_section["members"]["basis"]["sha256"] = _sha256_hex(basis_json_bytes)
    manifest["sections"].append(basis_section)

    # --- references section (optional) ---------------------------------
    if data.references:
        ref_section: Dict[str, Any] = {
            "id": "references",
            "kind": "references",
            "members": {
                "references": {
                    "path": "references/references.json",
                    "format": "json",
                }
            },
        }
        if refs_json_bytes is not None:
            ref_section["members"]["references"]["sha256"] = _sha256_hex(
                refs_json_bytes
            )
        manifest["sections"].append(ref_section)

    return manifest


def _resolve_source(source: str | Path) -> str:
    """Return JSON text from a string-or-path source.

    - If *source* is a :class:`Path`, read the file at that path.
    - If *source* is a ``str`` that names an existing file, read it.
    - Otherwise treat *source* as a JSON string literal.
    """
    if isinstance(source, Path):
        return Path(source).read_text(encoding="utf-8")
    # str: try file first, then inline JSON
    if os.path.isfile(source):
        return Path(source).read_text(encoding="utf-8")
    return source


# ---------------------------------------------------------------------------
# Text mode  (.qvf.json)
# ---------------------------------------------------------------------------


def to_qvf_json(data: BasisSetData, *, indent: int = 2) -> str:
    """Serialize a :class:`BasisSetData` to a plain JSON string."""
    d = data.to_dict()
    return json.dumps(d, indent=indent, sort_keys=False)


def from_qvf_json(source: str | Path) -> BasisSetData:
    """Deserialize a ``.qvf.json`` file or inline JSON string.

    Parameters
    ----------
    source:
        A file path (``str`` or :class:`Path`) pointing to a ``.qvf.json``
        file, or a JSON string containing a serialised basis set.

    Returns
    -------
    BasisSetData

    Raises
    ------
    jsonschema.ValidationError
        If the data does not conform to the QVF-Basis v1 schema.
    """
    raw = _resolve_source(source)
    d = json.loads(raw)
    validate_basis_dict(d)
    return BasisSetData.from_dict(d)


def save_qvf_json(data: BasisSetData, path: str | Path, *, indent: int = 2) -> Path:
    """Write *data* to a ``.qvf.json`` file.  Returns the resolved path."""
    path = Path(path)
    path.write_text(to_qvf_json(data, indent=indent), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Packaged mode  (.qvf)
# ---------------------------------------------------------------------------


def to_qvf_bytes(
    data: BasisSetData,
    *,
    program: str = "vibe-qc",
    version: str = "0.14.0",
) -> bytes:
    """Serialize *data* to an in-memory ``.qvf`` archive (ZIP bytes).

    The archive contains ``manifest.json``, ``basis/basis.qvf.json``, and
    (if references are present) ``references/references.json``.
    """
    # 1. Serialize the basis payload & optional references
    basis_json_text = to_qvf_json(data)
    basis_json_bytes = basis_json_text.encode("utf-8")

    refs_json_bytes: bytes | None = None
    if data.references:
        refs_json_text = json.dumps(
            [r.to_dict() for r in data.references], indent=2, sort_keys=False
        )
        refs_json_bytes = refs_json_text.encode("utf-8")

    # 2. Build manifest (with hashes)
    manifest = _build_manifest(
        data,
        program=program,
        version=version,
        basis_json_bytes=basis_json_bytes,
        refs_json_bytes=refs_json_bytes,
    )
    manifest_json_bytes = json.dumps(manifest, indent=2, sort_keys=False).encode(
        "utf-8"
    )

    # 3. Assemble sorted zip entries for deterministic output
    entries: List[Tuple[str, bytes]] = [
        ("manifest.json", manifest_json_bytes),
        ("basis/basis.qvf.json", basis_json_bytes),
    ]
    if refs_json_bytes is not None:
        entries.append(("references/references.json", refs_json_bytes))

    # Sort by normpath for deterministic archive order
    entries.sort(key=lambda e: os.path.normpath(e[0]))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload in entries:
            zf.writestr(arcname, payload)

    return buf.getvalue()


def save_qvf(
    data: BasisSetData,
    path: str | Path,
    *,
    program: str = "vibe-qc",
    version: str = "0.14.0",
) -> Path:
    """Write *data* to a ``.qvf`` file.  Returns the resolved path."""
    path = Path(path)
    raw = to_qvf_bytes(data, program=program, version=version)
    path.write_bytes(raw)
    return path


def from_qvf_bytes(source: bytes) -> BasisSetData:
    """Deserialize a ``.qvf`` archive from bytes into :class:`BasisSetData`.

    Raises
    ------
    ValueError
        If the archive is missing a manifest or a basis payload.
    jsonschema.ValidationError
        If the basis payload does not conform to the QVF-Basis v1 schema.
    """
    with zipfile.ZipFile(io.BytesIO(source), "r") as zf:
        # 1. Read manifest
        try:
            manifest_raw = zf.read("manifest.json")
        except KeyError:
            raise ValueError("QVf archive is missing manifest.json")

        manifest = json.loads(manifest_raw)

        # 2. Locate the basis section
        basis_path: str | None = None
        for section in manifest.get("sections", []):
            if section.get("kind") == "basis":
                members = section.get("members", {})
                basis_member = members.get("basis", {})
                basis_path = basis_member.get("path")
                break

        if basis_path is None:
            raise ValueError(
                "QVf manifest has no 'basis' section with a member 'basis'"
            )

        # 3. Read and parse the basis payload
        try:
            basis_raw = zf.read(basis_path)
        except KeyError:
            raise ValueError(f"QVf archive is missing basis payload at {basis_path!r}")

        basis_dict = json.loads(basis_raw)
        validate_basis_dict(basis_dict)
        data = BasisSetData.from_dict(basis_dict)

        # 4. Optionally read references from the archive (they are
        #    already embedded in the basis payload's .references field,
        #    but the QVF may carry a separate references section that
        #    we use as an override if present).
        for section in manifest.get("sections", []):
            if section.get("kind") == "references":
                members = section.get("members", {})
                refs_member = members.get("references", {})
                refs_path = refs_member.get("path")
                if refs_path:
                    try:
                        refs_raw = zf.read(refs_path)
                        refs_list = json.loads(refs_raw)
                        if isinstance(refs_list, list):
                            data.references = [
                                Reference.from_dict(r) for r in refs_list
                            ]
                    except KeyError:
                        pass  # references section declared but file missing
                break

        return data


def load_qvf(path: str | Path) -> BasisSetData:
    """Load a ``.qvf`` file into :class:`BasisSetData`."""
    return from_qvf_bytes(Path(path).read_bytes())


def qvf_manifest(
    data: BasisSetData,
    *,
    program: str = "vibe-qc",
    version: str = "0.14.0",
) -> Dict[str, Any]:
    """Return the manifest dict for *data* (for inspection, without hashes)."""
    return _build_manifest(data, program=program, version=version)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def load_any(path: str | Path) -> BasisSetData:
    """Load from ``.qvf.json`` or ``.qvf``, auto-detecting from extension.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        # .qvf.json ends with .json
        return from_qvf_json(path)
    elif suffix == ".qvf":
        return load_qvf(path)
    else:
        raise ValueError(
            f"Unrecognised basis file extension {path.suffix!r}; "
            f"expected .qvf.json or .qvf"
        )


def save_any(
    data: BasisSetData,
    path: str | Path,
    *,
    program: str = "vibe-qc",
    version: str = "0.14.0",
) -> Path:
    """Save to ``.qvf.json`` or ``.qvf``, auto-detecting from extension.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".qvf.json"):
        return save_qvf_json(data, path)
    elif name.endswith(".qvf"):
        return save_qvf(data, path, program=program, version=version)
    else:
        raise ValueError(
            f"Unrecognised basis file extension {path.suffix!r}; "
            f"expected .qvf.json or .qvf"
        )
