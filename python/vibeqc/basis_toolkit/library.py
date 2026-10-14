"""The bundled basis library as a store of QVF-Basis records.

``python/vibeqc/basis_library/qvf/<name>.qvf.json`` is the source of truth
for every bundled basis set (#295, #406). ``basis_library/basis/<name>.g94``
and ``<name>.ecp`` are renderings of those records for libint and
libecpint, written by :func:`render_library` and nothing else. The
tracked renderings are pinned byte for byte to the records by
``tests/test_basis_qvf_parity.py``, so the text cannot drift from the
record and a record cannot change without its rendering following.

This module imports nothing from the rest of ``vibeqc``, so the install
scripts can run it before the native core exists
(``scripts/render_basis_library.py`` bootstraps the package by path).

The public surface:

* :func:`iter_records` / :func:`load_record` / :func:`save_record`
* :func:`render_record` -> ``{filename: text}`` for one record
* :func:`render_library` -> write every rendering into a ``basis/`` dir
* :func:`check_library` -> the renderings that differ from the records
* :func:`library_root` -> ``python/vibeqc/basis_library``
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .exporter_g94 import to_ecp_sidecar, to_g94
from .model import (
    SCHEMA_VERSION,
    BasisSetData,
    CitationRoute,
    Provenance,
    ProvenanceKind,
    Reference,
    Upstream,
)

__all__ = [
    "RECORD_SUFFIX",
    "RenderReport",
    "check_library",
    "iter_records",
    "library_root",
    "load_record",
    "record_from_text",
    "record_path",
    "records_dir",
    "render_record",
    "render_library",
    "renderings_dir",
    "save_record",
    "stems",
]

RECORD_SUFFIX = ".qvf.json"


def library_root() -> Path:
    """``python/vibeqc/basis_library`` of this installation."""
    return Path(__file__).resolve().parent.parent / "basis_library"


def records_dir(root: Path | None = None) -> Path:
    return (root or library_root()) / "qvf"


def renderings_dir(root: Path | None = None) -> Path:
    return (root or library_root()) / "basis"


def record_path(stem: str, root: Path | None = None) -> Path:
    return records_dir(root) / f"{stem}{RECORD_SUFFIX}"


def stems(root: Path | None = None) -> list[str]:
    """Record names in the library, sorted."""
    directory = records_dir(root)
    if not directory.is_dir():
        return []
    return sorted(
        p.name[: -len(RECORD_SUFFIX)]
        for p in directory.iterdir()
        if p.name.endswith(RECORD_SUFFIX)
    )


def load_record(stem_or_path: str | os.PathLike, root: Path | None = None) -> BasisSetData:
    """Load one record by stem (from the library) or by path.

    Validation is structural (:meth:`BasisSetData.from_dict`); the JSON
    Schema check lives in :mod:`.validator` and runs in the test suite.
    """
    path = Path(stem_or_path)
    if not path.name.endswith(RECORD_SUFFIX):
        path = record_path(str(stem_or_path), root)
    with open(path, "r", encoding="utf-8") as fh:
        return BasisSetData.from_dict(json.load(fh))


def iter_records(root: Path | None = None) -> Iterator[tuple[str, BasisSetData]]:
    for stem in stems(root):
        yield stem, load_record(stem, root)


def record_json(data: BasisSetData) -> str:
    """The canonical on-disk JSON of a record: two-space indent, insertion
    order, trailing newline. One spelling, so a re-save is a no-op diff."""
    return json.dumps(data.to_dict(), indent=2, ensure_ascii=False) + "\n"


def save_record(data: BasisSetData, path: os.PathLike | str) -> Path:
    """Write ``data`` as a record, stamping schema version and content hash.

    The record's ``name`` must be the file stem: ``BasisSet(mol, name)``
    opens ``<name>.g94``, and the rendering is named after the record.
    """
    path = Path(path)
    stem = path.name[: -len(RECORD_SUFFIX)] if path.name.endswith(RECORD_SUFFIX) else path.stem
    if data.name != stem:
        raise ValueError(
            f"record name {data.name!r} must equal the file stem {stem!r}"
        )
    data.schema_version = SCHEMA_VERSION
    if data.provenance is not None:
        data.provenance.content_sha256 = data.content_sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(record_json(data), encoding="utf-8")
    os.replace(tmp, path)
    return path


def render_record(data: BasisSetData) -> dict[str, str]:
    """``{"<name>.g94": text, "<name>.ecp": text}`` (the sidecar only with ECPs)."""
    out = {f"{data.name}.g94": to_g94(data)}
    sidecar = to_ecp_sidecar(data)
    if sidecar is not None:
        out[f"{data.name}.ecp"] = sidecar
    return out


@dataclass
class RenderReport:
    written: list[str]
    unchanged: list[str]
    removed: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.written or self.removed)


def render_library(
    target: Path, *, root: Path | None = None, prune: bool = True,
) -> RenderReport:
    """Render every record of the library into ``target`` (a ``basis/`` dir).

    Files whose content already matches are left untouched, so mtimes stay
    meaningful. With ``prune`` (the default) a ``.g94`` or ``.ecp`` in
    ``target`` that no record renders is removed: a rendering without a
    record would be a second original, which is what this module exists to
    prevent.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    expected: dict[str, str] = {}
    for _stem, data in iter_records(root):
        expected.update(render_record(data))
    written: list[str] = []
    unchanged: list[str] = []
    for name in sorted(expected):
        path = target / name
        text = expected[name]
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            unchanged.append(name)
            continue
        tmp = path.with_name(path.name + ".partial")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        written.append(name)
    removed: list[str] = []
    if prune:
        for path in sorted(target.iterdir()):
            if path.suffix in (".g94", ".ecp") and path.name not in expected:
                path.unlink()
                removed.append(path.name)
    return RenderReport(written=written, unchanged=unchanged, removed=removed)


def check_library(
    target: Path | None = None, *, root: Path | None = None,
) -> dict[str, str]:
    """Compare the renderings in ``target`` with the records.

    Returns ``{filename: reason}`` for every difference: a rendering whose
    bytes differ from the record's, a rendering with no record, or a
    record with no rendering. Empty means the text and the records agree.
    """
    target = Path(target) if target is not None else renderings_dir(root)
    expected: dict[str, str] = {}
    for _stem, data in iter_records(root):
        expected.update(render_record(data))
    problems: dict[str, str] = {}
    for name, text in expected.items():
        path = target / name
        if not path.is_file():
            problems[name] = "no rendering for this record"
        elif path.read_text(encoding="utf-8") != text:
            problems[name] = "rendering differs from the record"
    if target.is_dir():
        for path in target.iterdir():
            if path.suffix in (".g94", ".ecp") and path.name not in expected:
                problems[path.name] = "rendering has no record"
    return dict(sorted(problems.items()))


def record_from_text(
    name: str,
    g94_text: str,
    ecp_text: str | None = None,
    *,
    kind: ProvenanceKind | str,
    upstream: list[Upstream] | None = None,
    retrieval_date: str | None = None,
    citation_keys: list[str] | None = None,
    references: list[Reference] | None = None,
    description: str = "",
    role: str | None = None,
    run_id: str | None = None,
    notes: str | None = None,
) -> BasisSetData:
    """One door from Gaussian94 text to a record with provenance (#410).

    ``g94_text`` may carry ``<Sym>-ECP`` blocks in the BSE layout; they
    join the record's ``ecps`` together with any ``ecp_text`` sidecar, and
    every element with a block is linked through ``ecp_id``. ``kind`` is
    the record's class; ``upstream``, ``retrieval_date`` (today when
    omitted) and ``citation_keys`` (``database.toml`` entry keys, routed
    over every element) fill the provenance block. The record is not saved;
    pass it to :func:`save_record`.
    """
    import datetime as _dt

    from .importer_g94 import from_g94

    data = from_g94(g94_text, name=name, role=role or "orbital")
    if ecp_text and ecp_text.strip():
        sidecar = from_g94(ecp_text, name=name, role="ecp")
        merged = dict(data.ecps or {})
        merged.update(sidecar.ecps or {})
        data.ecps = merged or None
    for sym, element in data.elements.items():
        element.ecp_id = sym if sym in (data.ecps or {}) else None
    data.name = name
    data.description = description
    data.schema_version = SCHEMA_VERSION
    if references is not None:
        data.references = list(references)
    data.provenance = Provenance(
        origin="file",
        kind=ProvenanceKind(kind),
        upstream=list(upstream or []),
        retrieval_date=retrieval_date or _dt.date.today().isoformat(),
        citations=[CitationRoute(keys=list(citation_keys))] if citation_keys else [],
        run_id=run_id,
        notes=notes,
    )
    data.provenance.content_sha256 = data.content_sha256()
    return data
